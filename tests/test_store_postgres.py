"""The Postgres store, against a real Postgres.

Every other test in this suite runs on SQLite, which means `PostgresStore` — the
backend the service actually deploys with — had no coverage at all: its
`_migrate()` had never executed, and neither had a single one of its statements
in the dialect they run in. The SQL is written once and translated by `_sql`, so
the *logic* is shared and well covered; what is untested is exactly the part
translation cannot guarantee — that `ON CONFLICT … EXCLUDED` means what we think
in both, that an integer flag round-trips through a real boolean-ish column, that
`rowcount` is populated after a DELETE, that BIGSERIAL replaces AUTOINCREMENT
cleanly.

Skipped unless `CHORDS_TEST_DATABASE_URL` points at a throwaway database. CI
supplies one from a service container; locally, any scratch database works:

    CHORDS_TEST_DATABASE_URL=postgresql://localhost/chords_test \\
        .venv/bin/python -m pytest tests/test_store_postgres.py

**It drops and recreates its tables**, so it refuses anything that does not look
like a test database — losing a production `chord_maps` to a stray environment
variable is a bad way to learn this file exists.
"""

from __future__ import annotations

import os
import uuid

import pytest

from app.store import (
    BLOCK_CHANNEL,
    BLOCK_VIDEO,
    STATUS_READY,
    PostgresStore,
)

DSN = os.environ.get("CHORDS_TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not DSN, reason="set CHORDS_TEST_DATABASE_URL to run the Postgres store tests"
)

_SAFE_NAMES = ("test", "ci", "tmp", "scratch")


@pytest.fixture(scope="module")
def pg_store():
    pytest.importorskip("psycopg", reason="psycopg is needed for the Postgres store")

    database = DSN.rsplit("/", 1)[-1].split("?")[0].lower()
    if not any(token in database for token in _SAFE_NAMES):
        pytest.fail(
            f"refusing to run against database {database!r}: this test drops tables. "
            f"Name it with one of {_SAFE_NAMES}."
        )

    store = PostgresStore(DSN)
    with store._cursor(write=True) as cur:
        for table in ("audit_log", "blocklist", "jobs", "chord_maps",
                      "rate_events", "usage", "users"):
            cur.execute(f"DROP TABLE IF EXISTS {table}")
    store._migrate()          # the statement that had never run
    yield store
    store.close()


@pytest.fixture
def clean(pg_store):
    with pg_store._cursor(write=True) as cur:
        for table in ("audit_log", "blocklist", "jobs", "chord_maps",
                      "rate_events", "usage", "users"):
            cur.execute(f"DELETE FROM {table}")
    return pg_store


SONG = {"schemaVersion": 2, "title": "Test"}


def test_the_schema_applies(clean):
    with clean._cursor() as cur:
        cur.execute("SELECT 1 FROM chord_maps LIMIT 1")
    with clean._cursor() as cur:
        cur.execute("SELECT id FROM audit_log LIMIT 1")   # BIGSERIAL, not AUTOINCREMENT


def test_upsert_user_is_idempotent_and_preserves_a_known_name(clean):
    clean.upsert_user("u1", "Ada")
    clean.upsert_user("u1", None)
    assert clean.display_name("u1") == "Ada"


def test_the_quota_counts_and_refunds(clean):
    clean.upsert_user("u1", "Ada")
    assert clean.try_record_use("u1", 2) == (True, 1)
    assert clean.try_record_use("u1", 2) == (True, 2)
    charged, count = clean.try_record_use("u1", 2)
    assert charged is False and count == 2

    clean.refund_use("u1")
    assert clean.usage_today("u1") == 1


def test_a_map_round_trips_including_the_low_confidence_flag(clean):
    """The flag is an INTEGER column written as 1/0 and read back through
    `bool()` — the kind of thing that works in SQLite and surprises you in
    Postgres if the column type ever drifts."""
    clean.put_map(video_id="v1", difficulty="normal", song=SONG, sync=None,
                  engine_chords="btc@1", engine_beats="beat_this@1",
                  analyzed_at="2026-08-03T00:00:00Z", channel_id="c1",
                  title="Test", duration_ms=1000, low_confidence=True)

    cached = clean.get_map("v1", "normal")
    assert cached.song == SONG
    assert cached.low_confidence is True
    assert cached.channel_id == "c1"
    assert cached.sync is None


def test_re_analysis_upserts_rather_than_duplicating(clean):
    for engine in ("btc@1", "btc@2"):
        clean.put_map(video_id="v1", difficulty="normal", song=SONG, sync=None,
                      engine_chords=engine, engine_beats="beat_this@1",
                      analyzed_at="2026-08-03T00:00:00Z", channel_id="c1",
                      title="Test", duration_ms=1000, low_confidence=False)

    with clean._cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM chord_maps WHERE video_id = 'v1'")
        assert cur.fetchone()[0] == 1


def test_an_admin_offset_survives_re_analysis(clean):
    """`put_map`'s ON CONFLICT deliberately preserves `offset_ms` — a hand
    correction (§6) must not be silently undone by a re-run."""
    clean.put_map(video_id="v1", difficulty="normal", song=SONG, sync=None,
                  engine_chords="btc@1", engine_beats="b@1",
                  analyzed_at="2026-08-03T00:00:00Z", channel_id="c1",
                  title="T", duration_ms=1000, low_confidence=False)
    clean.set_offset("v1", 250)
    clean.put_map(video_id="v1", difficulty="normal", song=SONG, sync=None,
                  engine_chords="btc@2", engine_beats="b@1",
                  analyzed_at="2026-08-04T00:00:00Z", channel_id="c1",
                  title="T", duration_ms=1000, low_confidence=False)

    assert clean.get_map("v1", "normal").offset_ms == 250


def test_the_blocklist_covers_video_and_channel(clean):
    clean.block(BLOCK_VIDEO, "v1", reason="dmca", actor="ops")
    clean.block(BLOCK_CHANNEL, "c9", reason="dmca", actor="ops")

    assert clean.is_blocked(video_id="v1") is True
    assert clean.is_blocked(video_id="other", channel_id="c9") is True
    assert clean.is_blocked(video_id="other", channel_id="c1") is False

    assert clean.unblock(BLOCK_VIDEO, "v1", actor="ops") is True
    assert clean.is_blocked(video_id="v1") is False


def test_purge_reports_what_it_actually_removed(clean):
    """§3 asks the operator to verify the cascade cascaded, which is only
    possible if the counts are real — `rowcount` after DELETE, per backend."""
    clean.put_map(video_id="v1", difficulty="normal", song=SONG, sync=None,
                  engine_chords="e", engine_beats="b",
                  analyzed_at="2026-08-03T00:00:00Z", channel_id="c1",
                  title="T", duration_ms=1000, low_confidence=False)
    clean.create_job(job_id="j1", uid="u1", video_id="v1", difficulty="normal")

    counts = clean.purge("v1", actor="ops", reason="dmca")

    assert counts["maps"] == 1
    assert counts["jobs"] == 1
    assert clean.get_map("v1", "normal") is None


def test_a_channel_purge_reaches_every_video(clean):
    for video in ("v1", "v2"):
        clean.put_map(video_id=video, difficulty="normal", song=SONG, sync=None,
                      engine_chords="e", engine_beats="b",
                      analyzed_at="2026-08-03T00:00:00Z", channel_id="c1",
                      title="T", duration_ms=1000, low_confidence=False)

    counts = clean.purge_channel("c1", actor="ops", reason="dmca")

    assert counts["videos"] == 2
    assert counts["maps"] == 2


def test_the_audit_log_outlives_what_it_records(clean):
    """§3: `purge` deliberately does not touch the audit log."""
    clean.put_map(video_id="v1", difficulty="normal", song=SONG, sync=None,
                  engine_chords="e", engine_beats="b",
                  analyzed_at="2026-08-03T00:00:00Z", channel_id="c1",
                  title="T", duration_ms=1000, low_confidence=False)
    clean.block(BLOCK_VIDEO, "v1", reason="dmca", actor="ops")
    clean.purge("v1", actor="ops", reason="dmca")

    actions = [entry["action"] for entry in clean.audit_entries(50)]
    assert "block" in actions
    assert "purge" in actions


def test_the_rate_limiter_admits_then_refuses(clean):
    key = uuid.uuid4().hex
    allowed_first, _ = clean.hit_rate_limit("uid", key, 2, 60.0)
    allowed_second, _ = clean.hit_rate_limit("uid", key, 2, 60.0)
    allowed_third, retry_after = clean.hit_rate_limit("uid", key, 2, 60.0)

    assert allowed_first and allowed_second
    assert allowed_third is False
    assert retry_after > 0


def test_jobs_move_through_their_lifecycle_and_expire(clean):
    clean.create_job(job_id="j1", uid="u1", video_id="v1", difficulty="normal")
    clean.update_job("j1", status=STATUS_READY, progress=1.0)

    job = clean.get_job("j1")
    assert job.status == STATUS_READY
    assert job.progress == pytest.approx(1.0)

    # Nothing is old enough yet, so a sweep must remove nothing.
    assert clean.prune_jobs(older_than_s=86_400) == 0
    # Everything terminal is older than "one second ago".
    assert clean.prune_jobs(older_than_s=0) == 1
    assert clean.get_job("j1") is None


def test_two_callers_asking_for_one_video_share_a_job(clean):
    clean.create_job(job_id="j1", uid="u1", video_id="v1", difficulty="normal")
    existing = clean.active_job_for("v1", "normal")
    assert existing is not None and existing.job_id == "j1"

    clean.update_job("j1", status=STATUS_READY)
    assert clean.active_job_for("v1", "normal") is None
