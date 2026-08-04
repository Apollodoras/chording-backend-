"""Job lifecycle — the seam between "the API accepted it" and "a worker ran it".

§16.1 chose job-id + poll over Mo's long synchronous request, because analysis
has a fetch + decode + DSP tail. That means something has to carry the work off
the request thread, and **which** something is a deployment decision, so it lives
behind `JobRunner.submit`:

- `ThreadJobRunner` — a background thread in the API process. Fine for local dev
  and a single-container run, and **not** what production should use: it shares
  the API container's filesystem and credentials, which is precisely what §4's
  worker isolation exists to prevent.
- `ModalJobRunner` (`modal_app.py`) — spawns the isolated worker function:
  separate image, separate secrets, read-only root, tmpfs scratch, its own
  timeout and memory cap.

Both call the same `run_job`, so the thing being tested locally is the thing that
runs in production, minus the isolation.

**On the hard timeout.** §5.1 wants a hard wall-clock budget with a kill-and-clean
on breach. A Python thread cannot be killed from outside, so `ThreadJobRunner`
only *checks* the deadline between stages — the real guarantee is the container's
(Modal's `timeout=`, or Docker's), which is another reason the thread runner is
for development. The scratch directory is destroyed either way: its cleanup runs
in a `finally`, and a killed container takes its tmpfs with it.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .analysis import engines
from .analysis.pipeline import analyze as run_pipeline
from .errors import AnalysisError, CODE_ANALYSIS_FAILED, CODE_VIDEO_BLOCKED
from .store import (
    STATUS_ANALYZING,
    STATUS_BLOCKED,
    STATUS_FAILED,
    STATUS_READY,
    STATUS_UNAVAILABLE,
    Store,
)

log = logging.getLogger("chords.jobs")

# Failures that cost us nothing and are not the player's fault: the video was
# blocked, private, region-locked, or too long — all decided from metadata,
# before a byte is fetched. Charging a daily analysis for an error message would
# be charging for nothing.
REFUNDABLE_CODES = {
    CODE_VIDEO_BLOCKED,
    "video_unavailable",
    "video_too_long",
    "feature_disabled",
}

# Which terminal job status each error code maps to. `blocked` and `unavailable`
# are separate from `failed` because the client renders them as calm states, not
# errors (§17.8) — nothing went wrong, the video just isn't analyzable.
_STATUS_FOR_CODE = {
    CODE_VIDEO_BLOCKED: STATUS_BLOCKED,
    "video_unavailable": STATUS_UNAVAILABLE,
    "video_too_long": STATUS_UNAVAILABLE,
}


def run_job(*, job_id: str, video_id: str, difficulty: str, uid: str,
            settings, store: Store, source) -> None:
    """Execute one analysis and record the outcome. Never raises.

    Never raises on purpose: this runs detached from any request, so an escaping
    exception would leave the job row stuck in `analyzing` forever and the client
    polling something that will never move. Every exit path writes a terminal
    status.
    """
    started = time.monotonic()

    def progress(status: str, fraction: float) -> None:
        store.update_job(job_id, status=status, progress=fraction)
        if time.monotonic() - started > settings.job_deadline_s:
            raise AnalysisError("That video took too long to analyze — try a shorter one.")

    try:
        store.update_job(job_id, status=STATUS_ANALYZING, progress=0.05)
        outcome = run_pipeline(
            video_id=video_id,
            settings=settings,
            store=store,
            source=source,
            chord_engine=engines.build_chord_engine(settings),
            beat_tracker=engines.build_beat_tracker(settings),
            onset_detector=engines.build_onset_detector(settings),
            progress=progress,
        )
    except AnalysisError as exc:
        _finish_failed(store, job_id, uid, exc.code, exc.message)
        return
    except Exception:
        log.exception("job %s (%s) crashed", job_id, video_id)
        _finish_failed(store, job_id, uid, CODE_ANALYSIS_FAILED,
                       "That video couldn’t be analyzed.")
        return

    # One row per difficulty: all three tiers are computed from one analysis
    # (§5.5 — "compute once, store all three, let the client pick") and stored
    # together, so switching difficulty later is a cache hit rather than a job.
    sync_wire = outcome.sync.model_dump() if outcome.sync else None
    for tier, song in outcome.songs.items():
        store.put_map(
            video_id=video_id, difficulty=tier, song=song, sync=sync_wire,
            engine_chords=outcome.engine_chords, engine_beats=outcome.engine_beats,
            analyzed_at=outcome.analyzed_at, channel_id=outcome.meta.channel_id,
            title=outcome.meta.title, duration_ms=outcome.duration_ms,
            low_confidence=outcome.low_confidence,
        )

    store.update_job(job_id, status=STATUS_READY, progress=1.0)
    log.info("job %s ready: %s (%s) in %.1fs", job_id, video_id,
             ", ".join(sorted(outcome.songs)), time.monotonic() - started)


def _finish_failed(store: Store, job_id: str, uid: str, code: str, message: str) -> None:
    status = _STATUS_FOR_CODE.get(code, STATUS_FAILED)
    store.update_job(job_id, status=status, progress=1.0,
                     error_code=code, error_message=message)
    if code in REFUNDABLE_CODES:
        store.refund_use(uid)


class JobRunner:
    """Base: run inline. Only sensible in tests, where determinism beats
    concurrency and a job that has finished by the time `submit` returns makes
    assertions readable."""

    def __init__(self, settings, store: Store, source):
        self.settings = settings
        self.store = store
        self.source = source

    def submit(self, *, job_id: str, video_id: str, difficulty: str, uid: str,
               audio: bytes | None = None, filename: str | None = None) -> None:
        run_job(job_id=job_id, video_id=video_id, difficulty=difficulty, uid=uid,
                settings=self.settings, store=self.store,
                source=self._source_for(audio, filename))

    def _source_for(self, audio: bytes | None, filename: str | None):
        """The source this job should run against.

        An upload brings its own audio, so it brings its own source — there is
        nothing to fetch and, per §2.1, nowhere the bytes could have been left
        for a shared source to pick up later. The YouTube path keeps the
        long-lived source built at startup.
        """
        if audio is None:
            return self.source
        from .analysis.file_source import FileSource
        return FileSource(audio, filename=filename, settings=self.settings)

    def can_analyze(self) -> bool:
        """Whether submitting a job here could actually produce a map.

        **The capability belongs to the runner, not to the container asking.**
        An in-process runner analyzes with its own hands, so it needs its own
        source and its own engines. A runner that dispatches to an isolated
        worker (`RemoteJobRunner`) has neither and does not need either — which
        is the whole point of §4, and is exactly what the API container is.

        `app/main.py` asks this before choosing between a 202 and a clean 503.
        Asking the *container* instead — "do I have a source, are my engines
        registered" — reads as the same question and is not: on Modal the API
        image is built without ffmpeg, yt-dlp or an engine deliberately, so that
        version answers "no" on a perfectly healthy deployment and refuses every
        analysis it was built to delegate.
        """
        return self.source is not None and engines.is_ready(self.settings)

    def can_accept_uploads(self) -> bool:
        """An upload needs ffmpeg and the engines, but no fetch source — that is
        the whole distinction, so it is asked separately from `can_analyze`."""
        from .analysis.file_source import available
        return available() and engines.is_ready(self.settings)


class ThreadJobRunner(JobRunner):
    """Background threads in the API process — development and single-container
    runs only. See the module docstring on why this is not production."""

    def __init__(self, settings, store: Store, source, *, max_workers: int = 2):
        super().__init__(settings, store, source)
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="analysis")
        self._lock = threading.Lock()

    def submit(self, *, job_id: str, video_id: str, difficulty: str, uid: str,
               audio: bytes | None = None, filename: str | None = None) -> None:
        with self._lock:
            self._pool.submit(
                run_job, job_id=job_id, video_id=video_id, difficulty=difficulty,
                uid=uid, settings=self.settings, store=self.store,
                source=self._source_for(audio, filename),
            )

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)


class RemoteJobRunner(JobRunner):
    """Hands the job to a worker in another container, and reports itself capable.

    The base class for `modal_app.py`'s `ModalJobRunner`, and it lives here rather
    than there so the capability rule above is covered by the audio-free suite —
    `modal` is not a dependency of this package, so nothing in `modal_app.py` can
    be imported by a test.

    `can_analyze` is True without consulting `source` or the engine registry
    because in this deployment shape **neither one is this container's to have**:
    the worker image installs them, and the deploy fails at build time if it
    can't (see the `test -f` on BTC's weights in `modal_app.py`). A container
    that cannot decode audio is the §4 guarantee working, not a degraded state.

    What this can still get wrong is a worker image that built without an engine
    while the API image came up fine. That surfaces as a job that fails with the
    engine's own 503 message instead of a refused submission — one poll later,
    and refunded either way (`REFUNDABLE_CODES` includes `feature_disabled`).
    """

    def submit(self, *, job_id: str, video_id: str, difficulty: str, uid: str,
               audio: bytes | None = None, filename: str | None = None) -> None:
        raise NotImplementedError

    def can_analyze(self) -> bool:
        return True

    def can_accept_uploads(self) -> bool:
        """Whether an upload could be analyzed if one arrived.

        Separate from `can_analyze` because the two can differ: the kill switch
        and a dead YouTube path both take the fetch route out without touching
        this one, which is most of the point of having it (see
        `analysis/file_source.py`). A remote runner answers for its worker, the
        same way `can_analyze` does.
        """
        return True
