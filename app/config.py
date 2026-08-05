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
    # Cheap authenticated reads — `GET /v1/analyze/{jobId}` and `GET
    # /v1/maps/{videoId}` — get their **own** budget rather than sharing the one
    # above, and this is a correctness fix rather than a tuning knob.
    #
    # The client models an analysis as one long await and runs the poll loop
    # inside `RemoteChordAnalysisService`, so a single job is one POST followed
    # by a poll every few seconds for as long as the job takes. Against a shared
    # budget of ten that arithmetic is fatal: ~14 requests land in the first
    # minute, and **every job that runs longer than about forty seconds
    # rate-limits its own status checks** — which is every YouTube job there is.
    # The player saw "too many requests" for a job that was running perfectly.
    #
    # The two limits guard genuinely different things. `rate_limit_per_min`
    # gates work and spend: a fetch, a decode, a DSP pipeline, a quota
    # decrement. A poll is one indexed row read whose whole purpose is to be
    # repeated. Sizing the second from the first is how the second one ends up
    # wrong.
    rate_limit_poll_per_min: int = 0
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
    # §20.4 — let a song's repeated sections vote their engine mistakes out.
    # A flag rather than a constant because this is the one part of the theory
    # layer that *edits chords the engine reported*, and the honest way to ship
    # something that can only be judged by measurement is to be able to turn it
    # off in production without a deploy. Everything else in §20 — the meter
    # reconciliation, the form detection, the pooled patterns — only ever
    # rearranges or re-derives; this one overwrites.
    theory_consensus: bool = True
    # §20.2 — let a tempo that reads an octave out be halved (or doubled) instead
    # of only reported. **Off** by default, and for the opposite reason to
    # `theory_consensus`: this one is unmeasured. Correcting the octave rewrites
    # the beat grid, so every bar line and every anchor in the song moves, and
    # the corpus has no track that triggers it — so there is no benchmark verdict
    # to turn it on with. With it off, a suspect tempo the container can still
    # carry ships low-confidence, and one it cannot fails with a message that
    # names the tempo (`analysis/meter.py`, `pipeline.assemble`).
    theory_tempo_octave: bool = False
    # Which chord engine / beat tracker to use. Chosen by the §8-step-2
    # benchmark (`bench/run_bench.py`, results in README):
    #
    #   chords  BTC        0.805 vs chroma's 0.531 on real tracks, and 3× faster
    #   beats   beat_this  downbeat F 0.893 vs librosa's 0.486 — and §13.2's
    #                      anchors *are* downbeats, so that is the whole margin
    #
    # Naming an engine that this image did not install is still a clean 503, not
    # a guess: `engines.is_ready()` checks the registry, not this string.
    chord_engine: str | None = "btc"
    beat_tracker: str | None = "beat_this"

    cors_allow_origins: str = ""

    @property
    def has_admin(self) -> bool:
        return bool(self.admin_token)


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _poll_limit() -> int:
    """The budget for cheap authenticated reads (see `rate_limit_poll_per_min`).

    Defaulted rather than required, because the alternative is a fix that only
    works on a deployment somebody remembered to update: this ships as a code
    change, and the secret it would otherwise depend on is edited by hand in the
    Modal dashboard. An unset `CHORDS_RATE_LIMIT_POLL_PER_MIN` therefore means
    "sixty a minute, or six times the spend budget if that is larger" — one
    poll a second, comfortably above any sane client loop and still a ceiling on
    a runaway one.

    `0` is honoured when it is set **explicitly**, and means off, the same as its
    two siblings. That distinction is why this reads the variable rather than
    passing a default to `os.environ.get`: "unset" and "set to zero" have to mean
    different things here, and only one of them can be spelled `"0"`.

    Off when the per-uid limit is off, too — that is the local-dev shape, and a
    limiter appearing on one route in a deployment that has disabled limiting
    everywhere else is a surprise nobody asked for.
    """
    raw = os.environ.get("CHORDS_RATE_LIMIT_POLL_PER_MIN")
    if raw is not None and raw.strip():
        return int(raw)
    uid_per_min = int(os.environ.get("CHORDS_RATE_LIMIT_PER_MIN", "0"))
    return max(60, uid_per_min * 6) if uid_per_min > 0 else 0


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
        rate_limit_poll_per_min=_poll_limit(),
        rate_limit_window_s=float(os.environ.get("CHORDS_RATE_LIMIT_WINDOW_S", "60")),
        admin_token=os.environ.get("CHORDS_ADMIN_TOKEN") or None,
        max_video_seconds=int(os.environ.get("CHORDS_MAX_VIDEO_SECONDS", str(DEFAULT_MAX_VIDEO_SECONDS))),
        job_deadline_s=float(os.environ.get("CHORDS_JOB_DEADLINE_S", "180")),
        scratch_root=os.environ.get("CHORDS_SCRATCH_ROOT", "/tmp/chords-scratch"),
        confidence_floor=float(os.environ.get("CHORDS_CONFIDENCE_FLOOR", "0.5")),
        theory_consensus=_bool("CHORDS_THEORY_CONSENSUS", True),
        theory_tempo_octave=_bool("CHORDS_THEORY_TEMPO_OCTAVE", False),
        chord_engine=os.environ.get("CHORDS_CHORD_ENGINE") or "btc",
        beat_tracker=os.environ.get("CHORDS_BEAT_TRACKER") or "beat_this",
        cors_allow_origins=os.environ.get("CHORDS_CORS_ORIGINS", ""),
    )
