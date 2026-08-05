"""The store — quota, limiter, and §3's operational surface.

§3 is blunt about why these are in the first milestone: *"They're cheap now and
impossible to retrofit under pressure."* The purge tests in particular are
written the way the handoff asks for — **verify it actually cascades** — because
a purge that silently matched nothing is the failure you find out about from a
lawyer rather than from a log.
"""

from __future__ import annotations

import pytest

from app.store import (
    AUDIT_BLOCK,
    AUDIT_PURGE,
    BLOCK_CHANNEL,
    BLOCK_VIDEO,
    RATE_SCOPE_UID,
    STATUS_QUEUED,
    STATUS_READY,
)


def put(store, video_id="dQw4w9WgXcQ", difficulty="normal", channel_id="UCtest"):
    store.put_map(
        video_id=video_id, difficulty=difficulty,
        song={"version": 2, "id": f"yt:{video_id}"}, sync={"videoId": video_id},
        engine_chords="fake@1", engine_beats="fake@1",
        analyzed_at="2026-08-03T10:00:00Z", channel_id=channel_id,
        title="Known Song", duration_ms=32_000,
    )


# --- quota ------------------------------------------------------------------

def test_the_quota_is_charged_atomically(store):
    for _ in range(3):
        charged, _ = store.try_record_use("uid", 3)
        assert charged
    charged, count = store.try_record_use("uid", 3)
    assert not charged and count == 3


def test_a_zero_quota_refuses_everything(store):
    """An unverified password account's allowance is genuinely zero."""
    assert store.try_record_use("uid", 0) == (False, 0)


def test_a_refund_gives_back_a_charge_whose_analysis_never_happened(store):
    """A video that turns out to be blocked, private or too long is rejected
    before a byte is decoded — charging a daily analysis for an error message
    would be charging for nothing."""
    store.try_record_use("uid", 3)
    store.refund_use("uid")
    assert store.usage_today("uid") == 0


def test_a_refund_never_goes_negative(store):
    store.refund_use("uid")
    assert store.usage_today("uid") == 0


# --- rate limiting ----------------------------------------------------------

def test_the_window_slides_rather_than_resetting_on_a_boundary(store):
    """A fixed bucket lets a caller fire `limit` requests at 11:59:59 and `limit`
    more at 12:00:00 — twice the intended rate at exactly the moment a burst
    hurts most."""
    for i in range(3):
        allowed, _ = store.hit_rate_limit(RATE_SCOPE_UID, "uid", 3, 60.0, now=1000.0 + i)
        assert allowed
    allowed, retry_after = store.hit_rate_limit(RATE_SCOPE_UID, "uid", 3, 60.0, now=1003.0)
    assert not allowed and retry_after == pytest.approx(57.0)
    allowed, _ = store.hit_rate_limit(RATE_SCOPE_UID, "uid", 3, 60.0, now=1061.0)
    assert allowed


def test_a_refused_request_does_not_hold_the_window_open(store):
    """Counting refusals would let a caller who keeps hammering lock themselves
    out forever."""
    store.hit_rate_limit(RATE_SCOPE_UID, "uid", 1, 60.0, now=1000.0)
    for i in range(5):
        store.hit_rate_limit(RATE_SCOPE_UID, "uid", 1, 60.0, now=1001.0 + i)
    allowed, _ = store.hit_rate_limit(RATE_SCOPE_UID, "uid", 1, 60.0, now=1061.0)
    assert allowed


def test_pruning_removes_windows_nobody_is_inside(store):
    """In the IP scope the key IS an IP address — personal data accumulating
    forever in a table nobody reads."""
    store.hit_rate_limit(RATE_SCOPE_UID, "seen-once", 5, 60.0, now=1000.0)
    assert store.prune_rate_events(older_than_s=60.0, now=2000.0) == 1


# --- the cache --------------------------------------------------------------

def test_a_map_round_trips(store):
    put(store)
    cached = store.get_map("dQw4w9WgXcQ", "normal")
    assert cached.song["id"] == "yt:dQw4w9WgXcQ"
    assert cached.sync["videoId"] == "dQw4w9WgXcQ"
    assert cached.channel_id == "UCtest"


def test_re_analysis_replaces_rather_than_duplicates(store):
    put(store)
    store.put_map(video_id="dQw4w9WgXcQ", difficulty="normal",
                  song={"version": 2, "id": "yt:dQw4w9WgXcQ", "title": "Better"},
                  sync=None, engine_chords="fake@2", engine_beats="fake@1",
                  analyzed_at="2026-08-04T10:00:00Z")
    cached = store.get_map("dQw4w9WgXcQ", "normal")
    assert cached.song["title"] == "Better"
    assert cached.engine_chords == "fake@2"


def test_an_admin_offset_survives_a_re_analysis(store):
    """It is a human correction, and a fresh run has no better information than
    the person who watched the video."""
    put(store)
    store.set_offset("dQw4w9WgXcQ", -250)
    put(store)
    assert store.get_map("dQw4w9WgXcQ", "normal").offset_ms == -250


def test_the_offset_moves_every_difficulty_of_a_video(store):
    for difficulty in ("easy", "normal", "hard"):
        put(store, difficulty=difficulty)
    assert store.set_offset("dQw4w9WgXcQ", 120) == 3


# --- blocklist (§3) ---------------------------------------------------------

def test_a_video_can_be_blocked(store):
    store.block(BLOCK_VIDEO, "dQw4w9WgXcQ", reason="DMCA", actor="agent")
    assert store.is_blocked(video_id="dQw4w9WgXcQ")


def test_a_whole_channel_can_be_blocked(store):
    """A takedown may name a channel, and then every video from it has to stop
    being served — including ones nobody has analyzed yet."""
    store.block(BLOCK_CHANNEL, "UCtest", reason="label request", actor="agent")
    assert store.is_blocked(video_id="never-seen-1", channel_id="UCtest")


def test_unblocking_works(store):
    store.block(BLOCK_VIDEO, "dQw4w9WgXcQ", reason=None, actor="agent")
    assert store.unblock(BLOCK_VIDEO, "dQw4w9WgXcQ")
    assert not store.is_blocked(video_id="dQw4w9WgXcQ")


def test_blocking_the_same_thing_twice_is_not_an_error(store):
    store.block(BLOCK_VIDEO, "dQw4w9WgXcQ", reason="first", actor="a")
    store.block(BLOCK_VIDEO, "dQw4w9WgXcQ", reason="second", actor="b")
    assert store.is_blocked(video_id="dQw4w9WgXcQ")


# --- purge (§3) -------------------------------------------------------------

def test_a_purge_actually_cascades(store):
    """"Verify it actually cascades" — the handoff's own instruction. Blocking
    while a cached map keeps being served is the failure that gets a DMCA agent's
    attention."""
    for difficulty in ("easy", "normal", "hard"):
        put(store, difficulty=difficulty)
    store.create_job(job_id="j1", uid="uid", video_id="dQw4w9WgXcQ", difficulty="normal")

    counts = store.purge("dQw4w9WgXcQ", actor="agent", reason="DMCA")

    assert counts == {"maps": 3, "jobs": 1}
    assert store.get_map("dQw4w9WgXcQ", "normal") is None
    assert store.get_job("j1") is None


def test_purging_a_channel_reaches_every_video_we_know_of_it(store):
    put(store, video_id="aaaaaaaaaaa", channel_id="UCtest")
    put(store, video_id="bbbbbbbbbbb", channel_id="UCtest")
    put(store, video_id="ccccccccccc", channel_id="UCother")

    counts = store.purge_channel("UCtest", actor="agent", reason="label request")

    assert counts["videos"] == 2 and counts["maps"] == 2
    assert store.get_map("ccccccccccc", "normal") is not None


def test_a_purge_that_matched_nothing_says_so(store):
    assert store.purge("never-analyzed") == {"maps": 0, "jobs": 0}


# --- audit log (§3) ---------------------------------------------------------

def test_every_block_and_purge_is_recorded_with_who_and_when(store):
    store.block(BLOCK_VIDEO, "dQw4w9WgXcQ", reason="DMCA", actor="agent@example.com")
    store.purge("dQw4w9WgXcQ", actor="agent@example.com", reason="DMCA")

    actions = [e["action"] for e in store.audit_entries()]
    assert AUDIT_BLOCK in actions and AUDIT_PURGE in actions
    entry = next(e for e in store.audit_entries() if e["action"] == AUDIT_BLOCK)
    assert entry["actor"] == "agent@example.com"
    assert entry["reason"] == "DMCA"
    assert entry["createdAt"]


def test_the_audit_log_outlives_what_it_records(store):
    """Append-only: `purge` deliberately does not touch it. The record of a
    takedown has to survive the thing taken down."""
    put(store)
    store.block(BLOCK_VIDEO, "dQw4w9WgXcQ", reason="DMCA", actor="agent")
    store.purge("dQw4w9WgXcQ", actor="agent", reason="DMCA")
    assert any(e["key"] == "dQw4w9WgXcQ" for e in store.audit_entries())


def test_the_store_exposes_no_way_to_edit_or_delete_the_audit_log():
    """Structural, not aspirational — there is no update or delete path to that
    table anywhere in the store."""
    import inspect

    from app import store as store_module

    source = inspect.getsource(store_module)
    for statement in ("DELETE FROM audit_log", "UPDATE audit_log"):
        assert statement not in source


# --- jobs -------------------------------------------------------------------

def test_a_job_moves_through_its_lifecycle(store):
    job = store.create_job(job_id="j1", uid="uid", video_id="dQw4w9WgXcQ", difficulty="normal")
    assert job.status == STATUS_QUEUED
    store.update_job("j1", status=STATUS_READY, progress=1.0)
    assert store.get_job("j1").status == STATUS_READY


def test_two_players_asking_at_once_find_one_another_s_job(store):
    """Rather than decoding the same recording twice for an identical result —
    which is both the expensive thing and the thing §2 wants to happen as rarely
    as possible."""
    store.create_job(job_id="j1", uid="a", video_id="dQw4w9WgXcQ", difficulty="normal")
    found = store.active_job_for("dQw4w9WgXcQ", "normal")
    assert found is not None and found.job_id == "j1"


def test_a_finished_job_is_not_joined(store):
    store.create_job(job_id="j1", uid="a", video_id="dQw4w9WgXcQ", difficulty="normal")
    store.update_job("j1", status=STATUS_READY)
    assert store.active_job_for("dQw4w9WgXcQ", "normal") is None


# --- the schema itself ------------------------------------------------------

def test_no_column_could_hold_audio(store):
    """§2.2 as a schema property. The absence is the enforcement: a future "just
    cache the chroma so re-analysis is cheap" change has to add a column first,
    and argue with this test."""
    with store._cursor() as cur:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cur.fetchall()]
        columns: list[str] = []
        for table in tables:
            cur.execute(f"PRAGMA table_info({table})")
            columns += [row[1].lower() for row in cur.fetchall()]

    forbidden = {"pcm", "audio", "samples", "waveform", "chroma", "spectrogram", "audio_path"}
    assert forbidden.isdisjoint(columns), sorted(forbidden & set(columns))
