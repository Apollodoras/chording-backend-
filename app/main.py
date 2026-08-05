"""The HTTP service — handoff §16's contract, deliberately shaped like Mo's.

| Endpoint | Body | Returns |
|---|---|---|
| `POST /v1/analyze` | `{videoId \\| url, difficulty?}` | `{song, videoSync}` or `{jobId, status}` |
| `GET /v1/analyze/{jobId}` | — | `{status, progress, song?, videoSync?, message?}` |
| `GET /v1/maps/{videoId}` | — | cached result or 404 |
| `GET /v1/me` | — | identity + quota (Mo's shape) |
| `GET /healthz` | — | `{"status": "ok", …}` |
| `POST /v1/admin/block` | `{videoId \\| channelId, reason}` | purge + block |
| `DELETE /v1/admin/maps/{videoId}` | — | purge |

§16 is explicit about why this file looks like Mo's `app/main.py`: the client's
HTTP layer, error enum and account plumbing **already exist** and will be copied,
not designed. "Match these and the client work is a day; diverge and it's a
week." So: the same `Authorization: Bearer <Firebase ID token>`, the same
`{"message": …}` error body, the same per-uid and per-IP limiters, the same
"report what actually built" `/healthz`.

The one deliberate divergence is the job model. Mo is a long synchronous request
the client awaits for up to 120 s; analysis has a fetch + decode + DSP tail that
deserves **job-id + poll** (§16.1). On the client that difference lands entirely
inside one service class, exactly as `RemoteMoRosettaService` isolates its
transport today.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .analysis import engines
from .analysis.fetch import build_source, parse_video_id
# Safe on the API image: `file_source` imports no audio dependency at module
# level (numpy and ffmpeg are reached inside `decode`), which is what CI's
# "the API surface cannot touch audio" check asserts.
from .analysis.file_source import MAX_UPLOAD_BYTES, upload_id
from .auth import Authenticator, Principal, build_authenticator
from .chords import DIFFICULTIES, NORMAL
from .config import Settings, load_settings
from .errors import (
    CODE_ANALYSIS_FAILED,
    CODE_BAD_REQUEST,
    CODE_EMAIL_UNVERIFIED,
    CODE_FEATURE_DISABLED,
    CODE_NOT_FOUND,
    CODE_QUOTA_EXHAUSTED,
    CODE_RATE_LIMITED,
    CODE_VIDEO_BLOCKED,
    AnalysisError,
    fail,
)
from .jobs import JobRunner, ThreadJobRunner
from .store import (
    BLOCK_CHANNEL,
    BLOCK_VIDEO,
    RATE_SCOPE_IP,
    RATE_SCOPE_UID,
    STATUS_FAILED,
    STATUS_READY,
    Store,
    build_store,
    next_utc_reset,
)

log = logging.getLogger("chords")


class AnalyzeRequest(BaseModel):
    """`videoId` or `url` — the app's entry point is "paste or pick a video", and
    a player pastes whatever the share sheet handed them."""

    videoId: str | None = Field(default=None, max_length=64)
    url: str | None = Field(default=None, max_length=2048)
    difficulty: str | None = None


class BlockRequest(BaseModel):
    videoId: str | None = Field(default=None, max_length=64)
    channelId: str | None = Field(default=None, max_length=64)
    reason: str | None = Field(default=None, max_length=1000)


class OffsetRequest(BaseModel):
    """§6's admin knob: shift a video's chart against its recording."""

    offsetMs: int | None = Field(default=None, ge=-30000, le=30000)


# Sentinel for "the caller didn't say", so that passing `source=None` can mean
# **explicitly no audio stack** — which is what the API container does, and what
# a test asserting the cache-only path needs. Defaulting `None` to "go build one"
# would make that impossible to express.
_UNSET = object()


def create_app(
    settings: Settings | None = None,
    *,
    authenticator: Authenticator | None = None,
    store: Store | None = None,
    runner: JobRunner | None = None,
    source=_UNSET,
) -> FastAPI:
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(instance: FastAPI):
        yield
        # Only the thread runner owns anything; the inline and Modal runners
        # have nothing to release.
        shutdown = getattr(instance.state.runner, "shutdown", None)
        if callable(shutdown):
            shutdown()

    app = FastAPI(title="Rosetta GP — chord analysis", version="1.0", lifespan=lifespan)

    app.state.settings = settings
    app.state.store = store or build_store(settings)
    app.state.authenticator = authenticator or build_authenticator(settings)
    # None on the API container by design (§4): it has no audio stack, cannot
    # fetch or decode, and serves cached maps only. The worker builds its own.
    app.state.source = build_source(settings) if source is _UNSET else source
    # A background thread, not the base runner. `JobRunner.submit` executes the
    # analysis **inline**, so defaulting to it would run fetch + decode + DSP on
    # the request thread and return the 202 only once the job it describes had
    # already finished — which is the job-id-and-poll contract (§16.1) inverted.
    # Modal replaces this with its own; tests that want determinism pass the
    # inline one explicitly.
    app.state.runner = runner or ThreadJobRunner(settings, app.state.store, app.state.source)

    _install_cors(app, settings)
    _install_error_handlers(app)
    _install_ip_rate_limit(app)
    _install_routes(app)
    return app


def _install_cors(app: FastAPI, settings: Settings) -> None:
    """Only installed when an origin is configured: the iOS app isn't a browser
    and needs no CORS, so an unconfigured deployment adds no middleware rather
    than defaulting to a wildcard someone has to remember to lock down."""
    origins = [o.strip() for o in settings.cors_allow_origins.split(",") if o.strip()]
    if not origins:
        return
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,   # bearer only, no cookies of ours
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )


# ---------------------------------------------------------------------------
# Abuse hardening
# ---------------------------------------------------------------------------

def client_ip(request: Request) -> str:
    """The caller's IP, as far as it can be trusted behind Modal's proxy.

    `X-Forwarded-For` grows left to right as it crosses proxies, so the
    **rightmost** entry is the one our trusted proxy appended and everything left
    of it is caller-supplied text. Reading the leftmost value — the common
    shorthand — hands an attacker a free bypass: a different fake IP per request
    starts a fresh window every time.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        parts = [part.strip() for part in forwarded.split(",") if part.strip()]
        if parts:
            return parts[-1]
    return request.client.host if request.client else "unknown"


def _too_many(retry_after: float, message: str) -> JSONResponse:
    response = JSONResponse(status_code=429, content={"message": message, "code": CODE_RATE_LIMITED})
    response.headers["Retry-After"] = str(max(1, int(retry_after + 0.5)))
    return response


def _install_ip_rate_limit(app: FastAPI) -> None:
    """Per-IP burst limiting, ahead of authentication — an unauthenticated flood
    still costs container time, and the per-uid limit cannot see a caller who
    never presents a valid token.

    `/healthz` is the only exemption: a liveness probe that can rate-limit itself
    into red is worse than useless. Everything else fails **closed**.
    """

    @app.middleware("http")
    async def _limit(request: Request, call_next):
        settings: Settings = app.state.settings
        if request.url.path == "/healthz" or settings.rate_limit_ip_per_min <= 0:
            return await call_next(request)

        ip = client_ip(request)
        try:
            allowed, retry_after = app.state.store.hit_rate_limit(
                RATE_SCOPE_IP, ip, settings.rate_limit_ip_per_min, settings.rate_limit_window_s
            )
        except Exception:
            # Fail closed: if the limiter can't be evaluated we don't know
            # whether this caller is inside its budget, and the endpoints it
            # guards are the expensive ones.
            log.exception("ip rate limiter unavailable")
            return _too_many(settings.rate_limit_window_s, "Too many requests just now — try again shortly.")

        if not allowed:
            return _too_many(retry_after, "Too many requests just now — try again shortly.")
        return await call_next(request)


def _guard(request: Request, principal: Principal) -> None:
    """The per-uid burst limit, after authentication.

    Per-uid *and* per-IP because they catch different attackers: one account
    behind many addresses, and many accounts behind one.
    """
    settings: Settings = request.app.state.settings
    if settings.rate_limit_per_min <= 0:
        return
    allowed, retry_after = request.app.state.store.hit_rate_limit(
        RATE_SCOPE_UID, principal.uid, settings.rate_limit_per_min, settings.rate_limit_window_s
    )
    if not allowed:
        seconds = max(1, int(retry_after + 0.5))
        # Deliberately different wording from the daily-quota 429 so the two are
        # tellable apart in the app and in a log: one says "slow down", the other
        # says "come back tomorrow".
        raise fail(429, f"Too many requests just now — try again in {seconds}s.",
                   CODE_RATE_LIMITED, headers={"Retry-After": str(seconds)})


# ---------------------------------------------------------------------------
# Errors — always {"message": ...}
# ---------------------------------------------------------------------------

def _install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def _http(_: Request, exc: HTTPException):
        detail = exc.detail
        body = {"message": detail["message"]} if isinstance(detail, dict) and "message" in detail \
            else {"message": str(detail)}
        if isinstance(detail, dict) and detail.get("code"):
            body["code"] = detail["code"]
        # Pass through any headers the raiser set (the rate limits carry
        # `Retry-After`) — reshaping the body must not silently drop them.
        return JSONResponse(status_code=exc.status_code, content=body,
                            headers=getattr(exc, "headers", None))

    @app.exception_handler(AnalysisError)
    async def _analysis(_: Request, exc: AnalysisError):
        return JSONResponse(status_code=exc.status,
                            content={"message": exc.message, "code": exc.code})

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        return JSONResponse(status_code=400,
                            content={"message": "That request wasn’t understood.",
                                     "code": CODE_BAD_REQUEST})

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception):
        log.exception("unhandled error")
        return JSONResponse(status_code=500, content={"message": "Something went wrong on our side."})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def _principal(request: Request) -> Principal:
    principal = request.app.state.authenticator(request.headers.get("Authorization"))
    _guard(request, principal)
    return principal


def _principal_checked(request: Request) -> Principal:
    """Same, but asks Firebase whether the account still exists and is enabled.

    Mo's A1 smoke run proved a deleted user's ID token keeps verifying for up to
    an hour, so the **spending** routes pay the round-trip: disabling an account
    in the Firebase console is the runbook's only abuse lever, and without this
    the ban does nothing for an hour on exactly the endpoints that cost money.
    Cheap read-only routes (`/v1/me`) keep the unchecked path — it is called on
    every app launch and the worst a stale token buys there is reading a balance.
    """
    principal = request.app.state.authenticator(
        request.headers.get("Authorization"), check_revoked=True
    )
    _guard(request, principal)
    return principal


def _admin(request: Request) -> str:
    """Authorize a takedown.

    A shared secret rather than a Firebase uid allow-list, deliberately: §3 wants
    a takedown satisfiable **in minutes**, including from a laptop that isn't
    signed into the app. Unconfigured ⇒ 503, never open.
    """
    settings: Settings = request.app.state.settings
    if not settings.has_admin:
        raise fail(503, "Admin actions aren’t configured on this deployment.", CODE_FEATURE_DISABLED)
    header = request.headers.get("X-Admin-Token", "")
    # Constant-time-ish: compare full strings rather than short-circuiting on the
    # first differing byte.
    if not header or not _constant_eq(header, settings.admin_token or ""):
        raise fail(401, "Not authorized.")
    return request.headers.get("X-Admin-Actor") or "admin"


def _constant_eq(a: str, b: str) -> bool:
    import hmac
    return hmac.compare_digest(a, b)


def _db_health(store: Store) -> str:
    """One round-trip to the database, reported rather than raised.

    Everything this service does past `/healthz` needs the store, so a green
    health check in front of an unreachable Postgres is the same class of lie as
    a green one in front of a dead authenticator — the failure §16 says this
    endpoint exists to prevent. It stays a *field* rather than a non-200 because
    the probe's other answers (auth mode, kill switch, `canAnalyze`) are exactly
    what an operator needs while the database is the thing that's broken.
    """
    try:
        store.ping()
        return "ok"
    except Exception:
        log.exception("healthz: store unreachable")
        return "unavailable"


def _install_routes(app: FastAPI) -> None:

    @app.get("/healthz")
    async def healthz():
        settings: Settings = app.state.settings
        # Reports what actually BUILT, not what config asked for. A green
        # healthz hiding a dead authenticator is the failure Mo learned from.
        return {
            "status": "ok",
            "auth": app.state.authenticator.mode,
            "store": type(app.state.store).__name__.replace("Store", "").lower(),
            # §3's kill switch, visible from outside. A protection that is
            # silently off is worse than one that is visibly off.
            "analysis": "enabled" if settings.analysis_enabled else "disabled",
            # `fetch`/`engines`/`enginesReady` describe THIS container, which on
            # the API image is correctly "none of it" (§4). `canAnalyze` is the
            # one to read for "would a new analysis be accepted" — it asks the
            # runner, which is what actually does the work.
            "fetch": "configured" if app.state.source is not None else "unconfigured",
            # How THIS container leaves the network, on the same terms as
            # `fetch` and `engines` above: `"proxy"`, `"direct"`, or `None` when
            # there is no source to ask. `"direct"` is not a neutral answer — it
            # means every analysis is drawing at YouTube's per-IP bot check at
            # roughly 1-in-6 and paying for the misses in cold starts.
            #
            # **On the deployed two-container shape this is always `None`**, and
            # that is correct rather than a gap: fetching lives in the worker,
            # this image has `source=None` by design (§4), and the proxy
            # credential is deliberately absent from this container's secret.
            # Reporting the worker's posture from here would mean either holding
            # a credential with no use for it or publishing config intent as
            # though it were built reality — and this endpoint exists precisely
            # because the second one is a lie that reads as a green tick.
            # `scripts/smoke.py` says "cannot see from here" rather than
            # guessing. Unlike `canAnalyze`, there is no runner-mediated channel
            # to ask the worker down; if that is ever wanted, it needs one.
            "egress": getattr(app.state.source, "egress", None),
            "engines": engines.available(),
            "enginesReady": engines.is_ready(settings),
            "canAnalyze": app.state.runner.can_analyze(),
            # Reported separately because the two genuinely differ: an upload
            # needs ffmpeg and the engines but no fetch source, so it survives
            # a dead YouTube path (see `analysis/file_source.py`).
            "canAcceptUploads": app.state.runner.can_accept_uploads(),
            "db": _db_health(app.state.store),
            "admin": "configured" if settings.has_admin else "unconfigured",
            "maxVideoSeconds": settings.max_video_seconds,
            "rateLimit": {
                "uidPerMin": settings.rate_limit_per_min,
                "ipPerMin": settings.rate_limit_ip_per_min,
            },
        }

    @app.get("/v1/me")
    async def me(principal: Principal = Depends(_principal)):
        """Mirrors Mo's `/v1/me` shape exactly (§16.1), so the client's
        `MoMeService` is reused rather than rewritten.

        The quota here is **this service's own** (§16.4's default until told
        otherwise): analysis costs real money per call and a Mo song and a video
        analysis are different spends.
        """
        store: Store = app.state.store
        settings: Settings = app.state.settings
        store.upsert_user(principal.uid, principal.display_name)
        return {
            "userID": principal.uid,
            "displayName": store.display_name(principal.uid) or principal.display_name,
            "quota": {
                "used": store.usage_today(principal.uid),
                "limit": effective_quota(settings, principal),
                "resetsAtUTC": next_utc_reset().isoformat().replace("+00:00", "Z"),
            },
            "emailVerified": principal.is_verified,
            "signInProvider": principal.sign_in_provider,
            # Same question `POST /v1/analyze` answers, asked ahead of time — so
            # it must be the same test. The app hides the analyze affordance on
            # a false, and reading the API container's own engine registry here
            # hid the feature on every healthy Modal deployment.
            "analysisEnabled": settings.analysis_enabled and app.state.runner.can_analyze(),
        }

    @app.post("/v1/analyze")
    async def analyze(body: AnalyzeRequest, request: Request,
                      principal: Principal = Depends(_principal_checked)):
        """Cache hit ⇒ `200` with the result inline. Otherwise `202` + a job id.

        The cache hit is the common case once warm, and it is **free** (§16.4):
        it costs us nothing, so it must not cost the player a daily analysis
        either. That is also why the blocklist is re-checked here — a video
        blocked after it was analyzed has to stop being served immediately, not
        once its map happens to be purged.
        """
        settings: Settings = app.state.settings
        store: Store = app.state.store

        video_id = _video_id(body)
        difficulty = _difficulty(body.difficulty)

        cached = store.get_map(video_id, difficulty)
        if cached is not None:
            if store.is_blocked(video_id=video_id, channel_id=cached.channel_id):
                raise fail(403, "That video isn’t available for chord analysis.", CODE_VIDEO_BLOCKED)
            return _result(cached)

        if store.is_blocked(video_id=video_id):
            raise fail(403, "That video isn’t available for chord analysis.", CODE_VIDEO_BLOCKED)
        if not settings.analysis_enabled:
            raise fail(503, "Chord analysis is unavailable right now.", CODE_FEATURE_DISABLED)
        # The *runner's* capability, not this container's. On Modal the API image
        # has no source and no engines by design (§4) and delegates to a worker
        # that has both; asking `app.state.source is None` here refused every
        # analysis on a healthy deployment. See `JobRunner.can_analyze`.
        if not app.state.runner.can_analyze():
            raise fail(503, "Chord analysis isn’t available on this deployment yet.",
                       CODE_FEATURE_DISABLED)

        # Two players asking for the same video at once join one job rather than
        # decoding the same recording twice for an identical result.
        existing = store.active_job_for(video_id, difficulty)
        if existing is not None:
            return JSONResponse(status_code=202,
                                content={"jobId": existing.job_id, "status": existing.status})

        _charge(app, principal)
        job_id = uuid.uuid4().hex
        store.create_job(job_id=job_id, uid=principal.uid, video_id=video_id, difficulty=difficulty)
        try:
            app.state.runner.submit(job_id=job_id, video_id=video_id, difficulty=difficulty,
                                    uid=principal.uid)
        except Exception:
            # The dispatch itself failed — Modal refusing a spawn, the thread
            # pool shutting down. Without this the caller is charged for a job
            # row that nothing will ever pick up, and `active_job_for` then hands
            # that dead id to everyone else who asks for the same video.
            log.exception("failed to submit job %s (%s)", job_id, video_id)
            store.update_job(job_id, status=STATUS_FAILED, progress=1.0,
                             error_code=CODE_ANALYSIS_FAILED,
                             error_message="That video couldn’t be analyzed.")
            store.refund_use(principal.uid)
            raise fail(503, "Chord analysis is busy right now — try again in a moment.",
                       CODE_FEATURE_DISABLED)
        return JSONResponse(status_code=202, content={"jobId": job_id, "status": "queued"})

    @app.post("/v1/analyze/upload")
    async def analyze_upload(request: Request, file: UploadFile = File(...),
                             difficulty: str | None = Form(default=None),
                             principal: Principal = Depends(_principal_checked)):
        """Analyze audio the player supplied, rather than a video we fetched.

        The second input path, and the one with **no YouTube-terms exposure** —
        see `app/analysis/file_source.py` for why that matters beyond
        convenience. Everything downstream is identical: the same gate, the same
        pipeline, the same envelope, the same §2.1 guarantee that the audio is
        destroyed with the job.

        The id is a **content hash**, so re-uploading the same file is a cache
        hit and therefore free (§16.4) — the player is charged for analyzing a
        piece of audio, not for sending it twice.

        The bytes are held in this container's memory for the length of one
        request and forwarded to the worker. They are never written to a Volume,
        never stored, and never readable by a later request; §4's isolation is
        unchanged because this container still never *decodes* anything.
        """
        settings: Settings = app.state.settings
        store: Store = app.state.store

        if not settings.analysis_enabled:
            raise fail(503, "Chord analysis is unavailable right now.", CODE_FEATURE_DISABLED)
        if not app.state.runner.can_accept_uploads():
            raise fail(503, "Uploads aren’t available on this deployment yet.",
                       CODE_FEATURE_DISABLED)

        tier = _difficulty(difficulty)
        data = await _read_upload(file, settings)
        video_id = upload_id(data)

        cached = store.get_map(video_id, tier)
        if cached is not None:
            return _result(cached)
        if store.is_blocked(video_id=video_id):
            raise fail(403, "That audio isn’t available for chord analysis.", CODE_VIDEO_BLOCKED)

        existing = store.active_job_for(video_id, tier)
        if existing is not None:
            return JSONResponse(status_code=202,
                                content={"jobId": existing.job_id, "status": existing.status})

        _charge(app, principal)
        job_id = uuid.uuid4().hex
        store.create_job(job_id=job_id, uid=principal.uid, video_id=video_id, difficulty=tier)
        try:
            app.state.runner.submit(job_id=job_id, video_id=video_id, difficulty=tier,
                                    uid=principal.uid, audio=data, filename=file.filename)
        except Exception:
            log.exception("failed to submit upload job %s", job_id)
            store.update_job(job_id, status=STATUS_FAILED, progress=1.0,
                             error_code=CODE_ANALYSIS_FAILED,
                             error_message="That audio couldn’t be analyzed.")
            store.refund_use(principal.uid)
            raise fail(503, "Chord analysis is busy right now — try again in a moment.",
                       CODE_FEATURE_DISABLED)
        return JSONResponse(status_code=202, content={"jobId": job_id, "status": "queued"})

    @app.get("/v1/analyze/{job_id}")
    async def analyze_status(job_id: str, principal: Principal = Depends(_principal)):
        store: Store = app.state.store
        job = store.get_job(job_id)
        if job is None:
            raise fail(404, "That analysis has expired — ask for it again.", CODE_NOT_FOUND)
        if job.uid != principal.uid:
            # 404 rather than 403: whether someone else's job exists is not this
            # caller's business.
            raise fail(404, "That analysis has expired — ask for it again.", CODE_NOT_FOUND)

        body: dict = {"status": job.status, "progress": round(job.progress, 3)}
        if job.status == STATUS_READY:
            cached = store.get_map(job.video_id, job.difficulty)
            if cached is None:
                # The map was purged between finishing and polling — a takedown
                # landing mid-flight. Not an error we caused; say so plainly.
                raise fail(404, "That video isn’t available for chord analysis.", CODE_VIDEO_BLOCKED)
            body.update(_result(cached))
        elif job.error_message:
            body["message"] = job.error_message
            if job.error_code:
                body["code"] = job.error_code
        return body

    @app.get("/v1/maps/{video_id}")
    async def get_map(video_id: str, difficulty: str | None = None,
                      principal: Principal = Depends(_principal)):
        store: Store = app.state.store
        cached = store.get_map(video_id, _difficulty(difficulty))
        if cached is None:
            raise fail(404, "No analysis for that video yet.", CODE_NOT_FOUND)
        if store.is_blocked(video_id=video_id, channel_id=cached.channel_id):
            raise fail(403, "That video isn’t available for chord analysis.", CODE_VIDEO_BLOCKED)
        return _result(cached)

    # -- Admin: §3's takedown surface ----------------------------------------

    @app.post("/v1/admin/block")
    async def admin_block(body: BlockRequest, request: Request):
        """Block **and purge**, in that order.

        Order is the point: blocking first means that even if the purge
        half-fails, nothing is served from the moment the request lands. The
        response reports the row counts so the operator can verify the cascade
        actually cascaded rather than trusting that it did.
        """
        actor = _admin(request)
        store: Store = app.state.store
        if not body.videoId and not body.channelId:
            raise fail(400, "Give a videoId or a channelId.", CODE_BAD_REQUEST)

        purged = {"maps": 0, "jobs": 0}
        if body.videoId:
            store.block(BLOCK_VIDEO, body.videoId, reason=body.reason, actor=actor)
            counts = store.purge(body.videoId, actor=actor, reason=body.reason)
            purged["maps"] += counts["maps"]
            purged["jobs"] += counts["jobs"]
        if body.channelId:
            store.block(BLOCK_CHANNEL, body.channelId, reason=body.reason, actor=actor)
            counts = store.purge_channel(body.channelId, actor=actor, reason=body.reason)
            purged["maps"] += counts["maps"]
            purged["jobs"] += counts["jobs"]
        log.warning("blocked video=%s channel=%s by %s: %s",
                    body.videoId, body.channelId, actor, body.reason)
        return {"blocked": True, "purged": purged}

    @app.delete("/v1/admin/block")
    async def admin_unblock(body: BlockRequest, request: Request):
        actor = _admin(request)
        store: Store = app.state.store
        removed = 0
        if body.videoId:
            removed += int(store.unblock(BLOCK_VIDEO, body.videoId, actor=actor))
        if body.channelId:
            removed += int(store.unblock(BLOCK_CHANNEL, body.channelId, actor=actor))
        return {"unblocked": removed}

    @app.delete("/v1/admin/maps/{video_id}")
    async def admin_purge(video_id: str, request: Request):
        actor = _admin(request)
        return {"purged": app.state.store.purge(video_id, actor=actor)}

    @app.post("/v1/admin/maps/{video_id}/offset")
    async def admin_offset(video_id: str, body: OffsetRequest, request: Request):
        """§6: set the per-video sync offset by hand.

        The app has **no latency calibration** — it was deleted along with the
        scoring it corrected — so this and the player's own nudge (§17.7) are the
        only two correction paths that exist.
        """
        actor = _admin(request)
        store: Store = app.state.store
        rows = store.set_offset(video_id, body.offsetMs)
        store.audit("set_offset", video_id, actor=actor, detail=str(body.offsetMs))
        return {"updated": rows, "offsetMs": body.offsetMs}

    @app.get("/v1/admin/audit")
    async def admin_audit(request: Request, limit: int = 100):
        _admin(request)
        return {"entries": app.state.store.audit_entries(min(limit, 1000))}

    @app.get("/v1/admin/blocklist")
    async def admin_blocklist(request: Request):
        _admin(request)
        return {"entries": [
            {"kind": k, "key": key, "reason": reason, "createdAt": created}
            for k, key, reason, created in app.state.store.blocked_entries()
        ]}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _video_id(body: AnalyzeRequest) -> str:
    raw = body.videoId or body.url or ""
    video_id = parse_video_id(raw)
    if video_id is None:
        raise fail(400, "That doesn’t look like a YouTube video.", CODE_BAD_REQUEST)
    return video_id


async def _read_upload(file: UploadFile, settings: Settings) -> bytes:
    """The uploaded audio, read with a hard ceiling.

    Read in chunks and abandoned the moment the cap is passed, rather than read
    whole and measured afterwards: `await file.read()` on an unbounded body is
    how a container with a memory cap dies instead of returning a 400. The cap is
    generous for a 10-minute song (§18) even lossless.
    """
    limit = MAX_UPLOAD_BYTES
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise fail(413, f"That file is larger than {limit // (1024 * 1024)} MB.",
                       CODE_BAD_REQUEST)
        chunks.append(chunk)
    if total == 0:
        raise fail(400, "That upload was empty.", CODE_BAD_REQUEST)
    return b"".join(chunks)


def _difficulty(value: str | None) -> str:
    if value is None:
        return NORMAL
    if value not in DIFFICULTIES:
        raise fail(400, f"Difficulty must be one of {', '.join(DIFFICULTIES)}.", CODE_BAD_REQUEST)
    return value


def _result(cached) -> dict:
    """The §13.1 envelope, from a stored row.

    `offsetMs` is applied from the map's own column rather than from the stored
    sidecar JSON, so an admin correction takes effect on the next request without
    re-analyzing anything.
    """
    sync = dict(cached.sync) if cached.sync else None
    if sync is not None and cached.offset_ms is not None:
        sync["offsetMs"] = cached.offset_ms
    return {"song": cached.song, "videoSync": sync}


def effective_quota(settings: Settings, principal: Principal) -> int:
    """Today's allowance for this identity.

    **Zero for an unverified password account**: anyone can sign up as anyone
    through Firebase's public Auth REST API, and until the address is proven that
    account may not spend money. Federated accounts were authenticated by their
    provider and keep the full allowance.
    """
    return settings.daily_quota if principal.is_verified else 0


def _charge(app: FastAPI, principal: Principal) -> None:
    """Enforce the daily quota and record the use in ONE atomic statement, before
    spending an analysis. Cache hits never reach here (§16.4)."""
    settings: Settings = app.state.settings
    store: Store = app.state.store
    store.upsert_user(principal.uid, principal.display_name)

    if not principal.is_verified:
        # 403, not 429: a limit says "come back later", this says "do something
        # first", and the app renders a different state for each. Waiting will
        # never clear it — checking their inbox will.
        raise fail(403, "Verify your email to analyze a video — check your inbox.",
                   CODE_EMAIL_UNVERIFIED)

    charged, _count = store.try_record_use(principal.uid, effective_quota(settings, principal))
    if not charged:
        raise fail(429, "You’ve hit today’s limit — try again tomorrow.", CODE_QUOTA_EXHAUSTED)


# The ASGI entrypoint (uvicorn app.main:app).
app = create_app()
