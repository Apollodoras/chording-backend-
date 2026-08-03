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
    store.create_job(job_id=job_id, uid="u1", video_id=VIDEO, difficulty="normal")
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

    job = store.create_job(job_id="fresh", uid="u1", video_id=VIDEO, difficulty="normal")
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
