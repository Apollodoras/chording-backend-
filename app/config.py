"""Runtime configuration, all from the environment (12-factor). Nothing secret
is committed; ``README.md`` documents each key.

Deliberately shaped like the Mo backend's ``app/config.py`` (handoff §16 — mirror
Mo, don't invent), with three additions this service needs and Mo does not:

- ``analysis_enabled`` — **the kill switch** (§3). One flag, no deploy.
- ``max_video_seconds`` — the 10-minute cap (§18).
- the worker's isolation knobs (scratch root, hard timeout).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# handoff §18: "Cap at 10 minutes. Songs, not DJ sets."
DEFAULT_MAX_VIDEO_SECONDS = 600


@dataclass(frozen=True)
class Settings:
    # --- The kill switch (§3) ------------------------------------------------
    # False disables *new* analysis jobs and answers a clean "feature
    # unavailable"; cached maps keep serving, because a cache hit costs nothing
    # and never touches a recording. Read per request, never captured at
    # startup, so flipping the env var on the deployment is enough — no deploy.
    analysis_enabled: bool = True

    # --- Auth (mirrors Mo exactly — same Firebase project, §16.2) -------------
    firebase_project_id: str | None = None
    firebase_service_account_json: str | None = None
    google_application_credentials: str | None = None
    # Fail loud instead of silently serving 401s: with this on, startup CRASHES
    # unless a real Firebase verifier was built.
    require_auth: bool = False
    # A literal bearer accepted in dev so local curl works without a Firebase
    # project. Refused at startup when require_auth is on.
    dev_token: str | None = None

    # --- Storage / quota -----------------------------------------------------
    db_path: str = "chords.sqlite3"
    # Set this and the store is Postgres; leave it unset and it is SQLite at
    # `db_path`. Unset is the default so local dev and CI need no database.
    database_url: str | None = None
    # §16.4: analysis gets its OWN daily quota (owner default until told
    # otherwise), and **cache hits do not count against it** — a cached map
    # costs us nothing, so it must not cost the player anything either.
    daily_quota: int = 10

    # --- Abuse hardening (mirrors Mo's A4) -----------------------------------
    rate_limit_per_min: int = 0
    rate_limit_ip_per_min: int = 0
    rate_limit_window_s: float = 60.0

    # --- Admin (§3 takedown intake) ------------------------------------------
    # A shared secret for /v1/admin/*. Unset ⇒ the admin routes answer 503 and
    # /healthz says `admin: "unconfigured"`. Deliberately NOT a Firebase uid
    # allow-list: a takedown must be satisfiable in minutes by whoever is
    # holding the pager, including from a laptop that isn't signed into the app.
    admin_token: str | None = None

    # --- Analysis ------------------------------------------------------------
    max_video_seconds: int = DEFAULT_MAX_VIDEO_SECONDS
    # Hard wall-clock budget for one job (§5.1 suggests 180 s). Breaching it
    # kills the job and cleans the scratch dir.
    job_deadline_s: float = 180.0
    # Where decoded audio may live for the seconds it exists. MUST be a tmpfs
    # mount in the worker (§4: `--read-only` + an explicit tmpfs for scratch).
    # app/analysis/scratch.py refuses to run if this looks durable.
    scratch_root: str = "/tmp/chords-scratch"
    # Mean-confidence floor below which the map is flagged `lowConfidence` and
    # videoSync is withheld entirely (§5.4.6, §13.3).
    confidence_floor: float = 0.5
    # Which chord engine / beat tracker to use. Unset until the §8-step-2
    # benchmark is run and the owner chooses — an unset engine makes the
    # analysis path answer a clean 503 rather than guessing (see
    # app/analysis/engines.py).
    chord_engine: str | None = None
    beat_tracker: str | None = None

    cors_allow_origins: str = ""

    @property
    def has_admin(self) -> bool:
        return bool(self.admin_token)


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_settings() -> Settings:
    return Settings(
        analysis_enabled=_bool("CHORDS_ANALYSIS_ENABLED", True),
        firebase_project_id=os.environ.get("FIREBASE_PROJECT_ID") or None,
        firebase_service_account_json=os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON") or None,
        google_application_credentials=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or None,
        require_auth=_bool("CHORDS_REQUIRE_AUTH", False),
        dev_token=os.environ.get("CHORDS_DEV_TOKEN") or None,
        db_path=os.environ.get("CHORDS_DB_PATH", "chords.sqlite3"),
        database_url=os.environ.get("CHORDS_DATABASE_URL") or None,
        daily_quota=int(os.environ.get("CHORDS_DAILY_QUOTA", "10")),
        rate_limit_per_min=int(os.environ.get("CHORDS_RATE_LIMIT_PER_MIN", "0")),
        rate_limit_ip_per_min=int(os.environ.get("CHORDS_RATE_LIMIT_IP_PER_MIN", "0")),
        rate_limit_window_s=float(os.environ.get("CHORDS_RATE_LIMIT_WINDOW_S", "60")),
        admin_token=os.environ.get("CHORDS_ADMIN_TOKEN") or None,
        max_video_seconds=int(os.environ.get("CHORDS_MAX_VIDEO_SECONDS", str(DEFAULT_MAX_VIDEO_SECONDS))),
        job_deadline_s=float(os.environ.get("CHORDS_JOB_DEADLINE_S", "180")),
        scratch_root=os.environ.get("CHORDS_SCRATCH_ROOT", "/tmp/chords-scratch"),
        confidence_floor=float(os.environ.get("CHORDS_CONFIDENCE_FLOOR", "0.5")),
        chord_engine=os.environ.get("CHORDS_CHORD_ENGINE") or None,
        beat_tracker=os.environ.get("CHORDS_BEAT_TRACKER") or None,
        cors_allow_origins=os.environ.get("CHORDS_CORS_ORIGINS", ""),
    )
