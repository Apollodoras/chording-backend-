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
    # How many proxy hops in front of this service are **ours**, i.e. how far from
    # the right of `X-Forwarded-For` the real client address is. 1 is today's
    # deployment (Modal's own proxy) and today's behaviour. See
    # `main.client_ip` for what getting it wrong costs in each direction —
    # too low and every caller shares one bucket, too high and a caller can
    # forge their own.
    trusted_proxy_hops: int = 1

    # --- Admin (§3 takedown intake) ------------------------------------------
    # A shared secret for /v1/admin/*. Unset ⇒ the admin routes answer 503 and
    # /healthz says `admin: "unconfigured"`. Deliberately NOT a Firebase uid
    # allow-list: a takedown must be satisfiable in minutes by whoever is
    # holding the pager, including from a laptop that isn't signed into the app.
    admin_token: str | None = None

    # --- Analysis ------------------------------------------------------------
    max_video_seconds: int = DEFAULT_MAX_VIDEO_SECONDS

    # --- The time budget, which has to add up ---------------------------------
    #
    # These five numbers describe one job's wall clock, and they used to
    # contradict each other in every direction. The job deadline was 180 s while
    # the fetch stage alone was allowed 240 s; the three subprocess ceilings summed
    # to 405 s while the Modal worker was killed at 300 s. Each number was
    # defensible on its own and the set was not, and the failure it produced is the
    # worst-shaped one available: a *slow but successful* fetch got the container
    # SIGKILLed with no terminal status written, so the job sat in `analyzing`
    # until the 900 s lease reaper found it — fifteen minutes of a player watching
    # a spinner for an analysis that had been working.
    #
    # The invariant, asserted by `tests/test_deployment.py` rather than left as
    # prose:
    #
    #     probe + fetch + decode + dsp_reserve  ≤  job_deadline_s
    #                                              < the container's own timeout
    #                                              < the store's job lease
    #
    # Read outward: the stages fit inside the deadline, so the deadline is what
    # fails a slow job — with a terminal status, a message, and a refund. The
    # container timeout is strictly larger, so it only ever fires for something the
    # deadline cannot see (a hang inside pure compute, which no in-process check
    # can interrupt). The lease is larger again, and is the backstop for a
    # container that died without writing anything at all.
    #
    # `job_deadline_s` is 450 rather than §5.1's suggested 180 because 180 is not a
    # budget a real analysis fits in: 10 minutes of audio has to be fetched,
    # decoded and run through two neural models. A typical job finishes in a small
    # fraction of this; the number is a ceiling on the pathological case, and the
    # cost of it being generous is paid only by jobs that were going to fail.
    job_deadline_s: float = 450.0
    # Metadata lookup — one yt-dlp call with `--skip-download`, or one ffprobe on
    # an upload. Short on purpose: the gate depends on it, so a video that cannot
    # be looked up quickly should be refused rather than waited on.
    probe_timeout_s: float = 45.0
    # The download. A 10-minute audio-only stream is a few MB; a fetch still
    # running after two minutes is a stream that is not going to arrive.
    fetch_timeout_s: float = 120.0
    # ffmpeg to mono 22.05 kHz PCM. CPU-bound and roughly linear in duration.
    decode_timeout_s: float = 90.0
    # Headroom kept for everything after the audio: two models, the theory layer,
    # the compiler, the lint. Not enforced anywhere — it is the term that makes the
    # inequality above meaningful, and the thing that shrinks if a stage ceiling
    # grows.
    dsp_reserve_s: float = 180.0
    # Where decoded audio may live for the seconds it exists. MUST be a tmpfs
    # mount in the worker (§4: `--read-only` + an explicit tmpfs for scratch).
    # app/analysis/scratch.py refuses to run if this looks durable.
    scratch_root: str = "/tmp/chords-scratch"
    # Mean-confidence floor below which the map is flagged `lowConfidence`
    # (§5.4.6, §13.3).
    #
    # It no longer withholds the sidecar. It used to, and what that did in
    # practice was take "play with the video" away from songs whose beat map was
    # measured on the very recording the player wanted to play along with —
    # handing back the same chart with a metronome, which repairs nothing. The
    # flag now travels *on* the sidecar, so the client can caveat a weak reading
    # instead of the service silently deleting the feature. See
    # `analysis/pipeline.py`.
    confidence_floor: float = 0.5
    # §20.4 — let a song's repeated sections vote their engine mistakes out.
    # A flag rather than a constant because this is the one part of the theory
    # layer that *edits chords the engine reported*, and the honest way to ship
    # something that can only be judged by measurement is to be able to turn it
    # off in production without a deploy. Everything else in §20 — the meter
    # reconciliation, the form detection, the pooled patterns — only ever
    # rearranges or re-derives; this one overwrites.
    theory_consensus: bool = True
    # §20.8 — let the song's own chord vocabulary correct a brief, doubtful
    # reading of a root the rest of the song contradicts. A flag for the same
    # reason as `theory_consensus`, and it is the other half of the same job:
    # the vote speaks where a section repeats and the passes disagree, this
    # speaks where they don't — an intro, a bridge, a section that occurs twice,
    # or a mistake the engine made identically in every pass. Measured the same
    # way, by `bench/run_bench.py --theory`.
    theory_vocabulary: bool = True
    # §20.9 — let belief settle a repeated slot the *count* cannot: a tie between
    # two readings, or a plurality that points the other way from the confidence.
    # A flag for the same reason as the two above, and it is the third face of
    # the same job — but it is the one that answers the complaint those two were
    # reported for and did not fix ("the engine adds variants and the song ends
    # up with more chords than it has"), because that complaint is a *tie*: the
    # engine hears the seventh in half the passes, so no majority forms, and
    # nothing in §20.4 or §20.8 has anything to count.
    theory_belief: bool = True
    # §20.10 — audit the finished chart against its own key, and settle a root
    # the song reads two incompatible ways. A flag for the same reason as the
    # three above, and **on** by default because it is the only layer that can
    # answer the owner's "major and minor chords in the same key": the vote needs
    # two passes of a section to disagree, the vocabulary needs a landslide of
    # mass, and a systematic mishearing offers neither. The key is the one piece
    # of evidence in the building that did not come from counting engine output
    # (`analysis/keyaudit.py`).
    theory_key_audit: bool = True
    # §20.2 — let a tempo that reads an octave out be halved (or doubled) instead
    # of only reported. **On**, since the 2026-08-18 audit, and the reason it was
    # off is worth keeping: correcting the octave rewrites the beat grid, so every
    # bar line and every anchor in the song moves, and there was no measurement
    # to turn it on with.
    #
    # There is now, and it is the shape that settles this kind of question: the
    # correction only fires on a tempo *outside* 40–220 BPM, no track in the
    # eleven-song chart corpus is, and turning it on is a **no-op to four decimal
    # places on all of them** (root 0.854 / triad 0.849 / form 0.747 either way).
    # So the risk of moving every anchor is bounded to the songs that today do not
    # ship at all: `pipeline.assemble` raises `TempoUnreadable` outside that range,
    # and the user pays a quota charge for a failure the halve/double machinery
    # right beside it could have repaired. Trading "certainly no song" for "a song
    # on a grid we corrected, flagged `tempoOctaveSuspect`" is the same trade
    # §13.3 already makes for the sidecar.
    theory_tempo_octave: bool = True
    # §21 — let every occurrence of a repeated section play that section's own
    # progression, so the chart states the song's form instead of transcribing
    # each pass separately. A flag for the same reason as the three above, and
    # **on** by default because without it `repeats` never fires on a real
    # recording and a four-chord song compiles as eighty-eight distinct bars.
    # It is not a correction layer: unlike `theory_consensus` it makes no claim
    # that the engine misheard anything, only that a verse is the same verse
    # every time the song plays it (`analysis/canon.py`).
    theory_form: bool = True
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
    # Which onset detector feeds §14's strumming extraction, and this one is a
    # *musical* choice rather than an accuracy one. `harmonic` reads the attacks
    # off the harmonic component, so the drums stop voting on what the guitarist
    # played; `librosa` reads the full mix, which is right for a solo recording
    # and is what turns a groove into straight eighths on anything else.
    # `bench/run_bench.py --strum` is where the default comes from: on the kit
    # specimens the full-mix detector emits every eighth, including a beat the
    # guitar never touches.
    onset_detector: str | None = "harmonic"
    # §15's section *names*, which need a loudness envelope to find. Off means
    # every section is `Part N` and nothing else changes — the chart, the bars and
    # the sidecar are all identical either way (see `pipeline.analyze`).
    #
    # This flag was read by `engines.build_structure_probe` via
    # `getattr(settings, "structure_probe", True)` and `Settings` had no such
    # field, so the knob was a permanent no-op: the probe could not be turned off,
    # and a `getattr` default is indistinguishable from a working setting at every
    # call site. Declaring it is the fix; the `getattr` can go.
    structure_probe: bool = True

    cors_allow_origins: str = ""

    @property
    def has_admin(self) -> bool:
        return bool(self.admin_token)

    @property
    def stage_budget_s(self) -> float:
        """What one job is allowed to spend before the deadline must have fired.

        The sum the invariant above is about. A property rather than a constant so
        it stays true when a deployment moves one of the parts.
        """
        return (self.probe_timeout_s + self.fetch_timeout_s
                + self.decode_timeout_s + self.dsp_reserve_s)


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
        trusted_proxy_hops=int(os.environ.get("CHORDS_TRUSTED_PROXY_HOPS", "1")),
        admin_token=os.environ.get("CHORDS_ADMIN_TOKEN") or None,
        max_video_seconds=int(os.environ.get("CHORDS_MAX_VIDEO_SECONDS", str(DEFAULT_MAX_VIDEO_SECONDS))),
        job_deadline_s=float(os.environ.get("CHORDS_JOB_DEADLINE_S", "450")),
        probe_timeout_s=float(os.environ.get("CHORDS_PROBE_TIMEOUT_S", "45")),
        fetch_timeout_s=float(os.environ.get("CHORDS_FETCH_TIMEOUT_S", "120")),
        decode_timeout_s=float(os.environ.get("CHORDS_DECODE_TIMEOUT_S", "90")),
        dsp_reserve_s=float(os.environ.get("CHORDS_DSP_RESERVE_S", "180")),
        scratch_root=os.environ.get("CHORDS_SCRATCH_ROOT", "/tmp/chords-scratch"),
        confidence_floor=float(os.environ.get("CHORDS_CONFIDENCE_FLOOR", "0.5")),
        theory_consensus=_bool("CHORDS_THEORY_CONSENSUS", True),
        theory_vocabulary=_bool("CHORDS_THEORY_VOCABULARY", True),
        theory_belief=_bool("CHORDS_THEORY_BELIEF", True),
        theory_key_audit=_bool("CHORDS_THEORY_KEY_AUDIT", True),
        theory_tempo_octave=_bool("CHORDS_THEORY_TEMPO_OCTAVE", True),
        theory_form=_bool("CHORDS_THEORY_FORM", True),
        chord_engine=os.environ.get("CHORDS_CHORD_ENGINE") or "btc",
        beat_tracker=os.environ.get("CHORDS_BEAT_TRACKER") or "beat_this",
        onset_detector=os.environ.get("CHORDS_ONSET_DETECTOR") or "harmonic",
        structure_probe=_bool("CHORDS_STRUCTURE_PROBE", True),
        cors_allow_origins=os.environ.get("CHORDS_CORS_ORIGINS", ""),
    )
