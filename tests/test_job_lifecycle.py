"""Job rows expire, and a job never runs on the request thread.

Two fixes to the same seam, and both are about a promise the service was making
without keeping:

- `GET /v1/analyze/{jobId}` answers an unknown id with *"That analysis has
  expired — ask for it again."* Nothing expired: job rows were only ever removed
  by a takedown purge, so the table grew forever and the message was fiction.
- `create_app` defaulted to `JobRunner`, whose `submit` runs the analysis
  **inline**. With no engines registered that was invisible — every request 503'd
  before reaching it — but the moment one is configured, `POST /v1/analyze`
  blocks for the whole fetch + decode + DSP and then returns a 202 announcing a
  job that has already finished. That inverts §16.1's contract.
"""

from __future__ import annotations

import time
from dataclasses import replace

from fastapi.testclient import TestClient

from app import jobs
from app.jobs import JobRunner, ThreadJobRunner
from app.main import create_app
from app.store import (
    STATUS_ANALYZING,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_READY,
    SQLiteStore,
)
from tests.conftest import FakeSource

VIDEO = "dQw4w9WgXcQ"


# --- expiry ------------------------------------------------------------------

def make_job(store: SQLiteStore, job_id: str, status: str, *, age_s: float) -> None:
    store.create_job(job_id=job_id, uid="u1", video_id=VIDEO)
    store.update_job(job_id, status=status)
    # Backdate `updated_at` directly: the prune is a time-based sweep and the
    # alternative is a test that sleeps for a day.
    from datetime import datetime, timedelta, timezone

    when = (datetime.now(timezone.utc) - timedelta(seconds=age_s))
    stamp = when.isoformat().replace("+00:00", "Z")
    with store._cursor(write=True) as cur:
        cur.execute("UPDATE jobs SET updated_at = ? WHERE job_id = ?", (stamp, job_id))


def test_finished_jobs_are_swept_once_they_are_old(store):
    make_job(store, "old-ready", STATUS_READY, age_s=200_000)
    make_job(store, "old-failed", STATUS_FAILED, age_s=200_000)
    make_job(store, "recent-ready", STATUS_READY, age_s=60)

    removed = store.prune_jobs(older_than_s=86_400)

    assert removed == 2
    assert store.get_job("old-ready") is None
    assert store.get_job("old-failed") is None
    assert store.get_job("recent-ready") is not None


def test_an_unfinished_job_is_never_swept_however_old(store):
    """The one row in this table someone is actually waiting on. A worker that
    has been running for a long time is the case where losing the row is worst:
    the client would poll a 404 for a job that is still going to finish."""
    make_job(store, "stuck", STATUS_ANALYZING, age_s=1_000_000)
    make_job(store, "queued", STATUS_QUEUED, age_s=1_000_000)

    store.prune_jobs(older_than_s=1.0)

    assert store.get_job("stuck") is not None
    assert store.get_job("queued") is not None


def test_the_sweep_is_rate_limited_so_it_does_not_run_per_request(store):
    make_job(store, "old", STATUS_READY, age_s=200_000)
    store._last_job_prune = time.time()      # pretend one just happened

    store._maybe_prune_jobs()

    assert store.get_job("old") is not None


def test_a_failing_sweep_never_fails_the_request(store, monkeypatch):
    """Housekeeping, not correctness — the same contract as the rate-event
    prune. A caller creating a job must not see someone else's cleanup fail."""
    def explode(*args, **kwargs):
        raise RuntimeError("database on fire")

    monkeypatch.setattr(store, "prune_jobs", explode)
    store._last_job_prune = 0.0

    store._maybe_prune_jobs()   # must not raise

    job = store.create_job(job_id="fresh", uid="u1", video_id=VIDEO)
    assert job.job_id == "fresh"


# --- the runner --------------------------------------------------------------

def test_the_default_runner_does_not_run_jobs_on_the_request_thread(settings, store):
    app = create_app(settings, store=store, source=FakeSource())
    assert isinstance(app.state.runner, ThreadJobRunner)


def test_an_explicit_runner_still_wins(settings, store):
    """Tests want determinism; Modal wants its own. Neither should have to
    fight the default."""
    inline = JobRunner(settings, store, FakeSource())
    app = create_app(settings, store=store, source=FakeSource(), runner=inline)
    assert app.state.runner is inline


def test_the_thread_pool_is_released_on_shutdown(settings, store):
    """A pool per app instance, left running, is a leak that only shows up in a
    long-lived process — i.e. the one place it matters."""
    app = create_app(settings, store=store, source=FakeSource())
    runner = app.state.runner

    with TestClient(app):
        pass    # entering and leaving drives startup + shutdown

    assert runner._pool._shutdown


def test_a_job_id_that_was_swept_reads_as_expired_not_as_an_error(settings, store):
    """The message the fix makes true: an id we no longer hold is 404 + the copy
    the client already renders, not a 500."""
    app = create_app(replace(settings, rate_limit_ip_per_min=0), store=store,
                     source=FakeSource(), runner=JobRunner(settings, store, FakeSource()))
    with TestClient(app) as client:
        response = client.get("/v1/analyze/does-not-exist",
                              headers={"Authorization": "Bearer dev-token"})

    assert response.status_code == 404
    assert response.json()["code"] == "not_found"
    assert "expired" in response.json()["message"].lower()


# --- a job that analyzed and could not be filed ------------------------------
#
# `run_job` promises never to raise, because it runs detached from any request:
# an escaping exception leaves the row at `analyzing` and the client polling
# something that will never move. Every failure path honoured that. The success
# path did not — persistence sat outside the try — so the one case where the
# expensive work had already been done was the one case that could hang.

class _Meta:
    channel_id = "UC123"
    title = "A Song"


class _Outcome:
    """A successful analysis, as `run_pipeline` would return it."""

    song = {"id": "yt:x", "title": "A Song"}
    engine_chords = "btc"
    engine_beats = "beat_this"
    analyzed_at = "2026-08-05T00:00:00Z"
    meta = _Meta()
    duration_ms = 180_000
    low_confidence = False
    # None here: this stub is about the job's bookkeeping, not about §13.
    sync = None


def _analyzed_job(store, settings, monkeypatch, *, put_map):
    """Run one job whose analysis succeeds and whose persistence is `put_map`.

    The engine builders are stubbed alongside the pipeline because `run_job`
    calls them before it calls the pipeline, so without this these tests would
    be asserting on whichever engines the registry happens to hold — which is a
    different module's business. What is under test here is the bookkeeping
    around a completed analysis, so the analysis is the part that gets faked.
    """
    monkeypatch.setattr(jobs, "run_pipeline", lambda **kw: _Outcome())
    for builder in ("build_chord_engine", "build_beat_tracker",
                    "build_onset_detector", "build_structure_probe"):
        monkeypatch.setattr(jobs.engines, builder, lambda _s: None)
    monkeypatch.setattr(store, "put_map", put_map)
    store.create_job(job_id="j1", uid="u1", video_id=VIDEO)
    store.try_record_use("u1", 10)
    return jobs.run_job(job_id="j1", video_id=VIDEO, uid="u1",
                        settings=settings, store=store, source=FakeSource())


def test_a_store_failure_after_a_good_analysis_still_answers_the_poller(store, settings, monkeypatch):
    """The reproduction: `put_map` raises, and the job used to be abandoned
    mid-flight — status `analyzing`, progress 0.05, no message, and the charge
    still spent. The lease reaper collected it fifteen minutes later; the client
    polled a corpse until then."""
    def explode(**kwargs):
        raise RuntimeError("database connection lost")

    outcome = _analyzed_job(store, settings, monkeypatch, put_map=explode)

    assert outcome == jobs.OUTCOME_FAILED
    job = store.get_job("j1")
    assert job.is_terminal, "an abandoned row is the thing this must never leave"
    assert job.status == STATUS_FAILED
    assert job.progress == 1.0
    assert "couldn’t be saved" in job.error_message


def test_an_analysis_we_could_not_file_is_not_charged_for(store, settings, monkeypatch):
    """It cost us the whole analysis, so it is not in `REFUNDABLE_CODES` — but
    the player got nothing and has to ask again, which is the reaper's reasoning
    for the same shape of failure."""
    def explode(**kwargs):
        raise RuntimeError("database connection lost")

    _analyzed_job(store, settings, monkeypatch, put_map=explode)

    assert store.usage_today("u1") == 0


def test_a_store_too_broken_to_record_the_failure_still_does_not_raise(store, settings, monkeypatch):
    """The far end of the same promise. If the database is unreachable there is
    nowhere to write a terminal status either — so this returns and lets the
    lease reaper answer from a healthier container, rather than raising into a
    thread pool that will not log it."""
    def explode(*args, **kwargs):
        raise RuntimeError("database on fire")

    monkeypatch.setattr(store, "update_job", explode)

    outcome = _analyzed_job(store, settings, monkeypatch, put_map=explode)

    assert outcome == jobs.OUTCOME_FAILED


def test_the_happy_path_files_the_chart_and_reports_ready(store, settings, monkeypatch):
    """The guard must not have changed what success does."""
    filed = []
    outcome = _analyzed_job(store, settings, monkeypatch,
                            put_map=lambda **kw: filed.append(kw["video_id"]))

    assert outcome == jobs.OUTCOME_READY
    assert filed == [VIDEO]
    assert store.get_job("j1").status == STATUS_READY
    assert store.usage_today("u1") == 1
