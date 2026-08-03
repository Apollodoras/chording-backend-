"""Authentication — verify the Bearer token into a stable user id.

**Ported from the Mo backend's ``app/auth.py``, against the SAME Firebase
project** (handoff §16.2: "reuse Firebase, do not invent"). That is the whole
point: the player is already signed in, the client's token provider is reused
as-is, and this service gets the account plumbing for free.

Identity lives in **Firebase Auth** (email/password + Apple/Google/X/Facebook).
The iOS app signs in through the FirebaseAuth SDK and sends the resulting
**Firebase ID token** as the Bearer; here we verify it with ``firebase-admin``
and take the ``uid``. Rosetta's backend never sees a password.

Two notes from Mo's build that §16.2 says will save a day, both still true here:
the ID token **expires hourly**, so the client must fetch it per request; and it
is read from **off the main actor** on the client (`nonisolated`, behind a lock)
— re-asserting the main actor on that path is what crashed Mo.

Separate credentials from Mo's, deliberately (§19.2): same Firebase project for
identity, but this deployment's service-account key is its own, so the audio
pipeline never inherits Mo's blast radius.

Two escape hatches keep this testable and locally runnable:
- ``CHORDS_DEV_TOKEN`` — a literal bearer accepted as a fixed dev user (matches the
  app's DEBUG ``signInAsDeveloper()`` token ``"dev-token"``). Never set in prod.
- ``firebase_verifier`` is injected, so tests drive verification without a real
  Firebase project or network.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

from fastapi import Depends, Header, HTTPException

from .config import Settings

log = logging.getLogger("chords.auth")

# uid used for the dev-token bypass.
DEV_UID = "dev:local"

# A verifier maps a raw ID token → its **decoded claims** (or raises on invalid).
#
# It returns the whole claim set rather than the two fields we happened to need
# first: A4's abuse gate gets its answer from `email_verified` and
# `firebase.sign_in_provider`, and every later phase that wants a claim would
# otherwise have to widen this signature again.
#
# `check_revoked` asks Firebase whether the *account* behind a still-unexpired
# token has been deleted or disabled. It costs a network round-trip, so it is a
# parameter rather than a default — see `Authenticator.__call__`.
FirebaseVerifier = Callable[..., dict[str, Any]]

# The `firebase.sign_in_provider` value for an email/password account. Everything
# else (apple.com, google.com, twitter.com, facebook.com) is a federated identity
# whose email the provider itself already proved.
PASSWORD_PROVIDER = "password"


@dataclass
class Principal:
    uid: str
    display_name: Optional[str]
    # --- A4 abuse claims ---
    email: Optional[str] = None
    # Firebase's own `email_verified` claim, verbatim.
    email_verified: bool = False
    # `firebase.sign_in_provider` — "password", "apple.com", "google.com", …
    sign_in_provider: Optional[str] = None

    @property
    def is_verified(self) -> bool:
        """Whether this identity is trustworthy enough to spend money on.

        The hole A4 closes (finding F4): anyone can mint a Firebase password
        account from the public REST API with an address they do not own, and each
        one carried a full daily quota of ~$0.20–1.50 calls. So a **password**
        account must prove its address before it can spend an analysis.

        A **federated** account is verified by construction — Apple/Google/X/
        Facebook only issue an identity after authenticating the person at their
        end, and Firebase does not always set `email_verified` on those tokens
        (Apple's private-relay addresses in particular). Gating them on a claim
        they may legitimately lack would lock out real players to no benefit.

        The dev-token principal has no provider at all and stays usable, which is
        what keeps the local flow drivable.
        """
        if self.sign_in_provider == PASSWORD_PROVIDER:
            return self.email_verified
        return True


def _bearer(authorization: Optional[str]) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        # 401 → the client shows its sign-in CTA (RemoteMoRosettaService maps it
        # to .notSignedIn).
        raise HTTPException(status_code=401, detail={"message": "Sign in to analyze a video."})
    return authorization[7:].strip()


def _service_account_info(raw: str) -> dict[str, Any]:
    """Parse the service-account key given as env-var *content*. Accepts raw JSON
    or base64-wrapped JSON (some secret stores mangle multi-line values)."""
    text = raw.strip()
    if not text.startswith("{"):
        try:
            text = base64.b64decode(text, validate=True).decode("utf-8").strip()
        except (binascii.Error, UnicodeDecodeError) as exc:
            raise ValueError("FIREBASE_SERVICE_ACCOUNT_JSON is neither JSON nor base64 JSON") from exc
    info = json.loads(text)
    if not isinstance(info, dict) or "private_key" not in info:
        raise ValueError("FIREBASE_SERVICE_ACCOUNT_JSON is not a service-account key")
    return info


def firebase_credential(settings: Settings) -> tuple[Any, str]:
    """Pick the Admin-SDK credential and say which path won.

    Modal (and most secret stores) hand us env *values*, not files, so the
    JSON-content path is the production one; the file path stays for mounted-key
    deploys; project-id-only is the last resort (ID-token verification needs only
    Google's public certs, but anything calling the Admin API will fail).

    Returns (credential | None, label). Never logs key material.
    """
    from firebase_admin import credentials

    if settings.firebase_service_account_json:
        info = _service_account_info(settings.firebase_service_account_json)
        return credentials.Certificate(info), "service-account-json"

    path = settings.google_application_credentials
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"GOOGLE_APPLICATION_CREDENTIALS points at a missing file: {path}")
        return credentials.Certificate(path), "service-account-file"

    return None, "project-id-only"


def firebase_admin_verifier(settings: Settings) -> FirebaseVerifier:
    """The production verifier. Initializes the Admin SDK from whichever
    credential path `firebase_credential` selects."""
    import firebase_admin
    from firebase_admin import auth as fb_auth

    project_id = settings.firebase_project_id
    credential, source = firebase_credential(settings)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(credential, options={"projectId": project_id})
    log.info("Firebase verifier active — project=%s credential=%s", project_id, source)

    def verify(token: str, check_revoked: bool = False) -> dict[str, Any]:
        # `check_revoked=True` costs a call to Firebase to ask whether the account
        # still exists and is enabled. A1's smoke run proved why it matters: a
        # deleted user's already-minted ID token keeps verifying for up to an hour
        # (finding F13). We pay for it only where that hour is dangerous.
        return fb_auth.verify_id_token(token, check_revoked=check_revoked)

    return verify


class Authenticator:
    """Turns a request's Authorization header into a `Principal`. Constructed
    once at startup and used as a FastAPI dependency."""

    def __init__(self, settings: Settings, verifier: Optional[FirebaseVerifier] = None):
        self.settings = settings
        self._verifier = verifier

    @property
    def mode(self) -> str:
        """What can actually authenticate a request, derived from what BUILT —
        never from config intent. This is the value `/healthz` reports, and it is
        the diagnostic that would have caught F1 (a dead verifier hiding behind a
        green healthz)."""
        if self._verifier is not None:
            return "firebase"
        if self.settings.dev_token:
            return "dev-token"
        return "none"

    def _verify(self, token: str, check_revoked: bool = False) -> Principal:
        if self.settings.dev_token and token == self.settings.dev_token:
            # No provider, and `is_verified` is True for that — the dev principal
            # must stay usable or the whole local flow stops being drivable.
            return Principal(uid=DEV_UID, display_name="Developer", email_verified=True)
        if self._verifier is None:
            raise HTTPException(status_code=401, detail={"message": "Sign in to analyze a video."})
        try:
            claims = self._verifier(token, check_revoked)
        except HTTPException:
            raise
        except Exception as exc:  # firebase raises a variety of token errors
            log.info("token verification failed: %s", exc)
            raise HTTPException(status_code=401, detail={"message": "Your session expired — sign in again."})
        return self._principal(claims)

    @staticmethod
    def _principal(claims: dict[str, Any]) -> Principal:
        """Decoded claims → the identity the routes reason about.

        Every field is read defensively: a claim set is whatever Firebase chose to
        put in the token, and a missing `firebase` block (or a hand-built one in a
        test) must not 500 the request.
        """
        firebase = claims.get("firebase")
        provider = firebase.get("sign_in_provider") if isinstance(firebase, dict) else None
        return Principal(
            uid=claims["uid"],
            display_name=claims.get("name"),
            email=claims.get("email"),
            email_verified=bool(claims.get("email_verified", False)),
            sign_in_provider=provider,
        )

    def __call__(self, authorization: Optional[str] = Header(default=None),
                 *, check_revoked: bool = False) -> Principal:
        return self._verify(_bearer(authorization), check_revoked)


class AuthConfigurationError(RuntimeError):
    """Startup refused: production demanded real auth and didn't get it."""


def build_authenticator(settings: Settings) -> Authenticator:
    """Build the request authenticator, and in production refuse to start rather
    than serve 401s to every real user (the F1 failure mode: a swallowed verifier
    error left the service up, green, and unusable)."""
    verifier = None
    failure: Exception | None = None
    if settings.firebase_project_id:
        try:
            verifier = firebase_admin_verifier(settings)
        except Exception as exc:  # missing creds locally shouldn't crash startup
            failure = exc
            log.warning("Firebase verifier unavailable (%s); only CHORDS_DEV_TOKEN will authenticate", exc)
    elif settings.require_auth:
        failure = AuthConfigurationError("FIREBASE_PROJECT_ID is not set")

    authenticator = Authenticator(settings, verifier)

    if settings.require_auth:
        # A dev bypass in production is a hole, not a fallback — refuse both the
        # missing-verifier case and the dev-token-in-prod case, loudly.
        if verifier is None:
            raise AuthConfigurationError(
                f"CHORDS_REQUIRE_AUTH=1 but no Firebase verifier could be built ({failure}). "
                "Set FIREBASE_PROJECT_ID and FIREBASE_SERVICE_ACCOUNT_JSON."
            )
        if settings.dev_token:
            raise AuthConfigurationError(
                "CHORDS_REQUIRE_AUTH=1 but CHORDS_DEV_TOKEN is set — a literal bearer must never be accepted in production."
            )

    log.info("auth mode: %s", authenticator.mode)
    return authenticator


def principal_dependency(authenticator: Authenticator):
    """Wrap the authenticator so FastAPI treats it as a dependency callable."""

    def dependency(authorization: Optional[str] = Header(default=None)) -> Principal:
        return authenticator(authorization)

    return Depends(dependency)
