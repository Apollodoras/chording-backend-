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


# --- the catalog -------------------------------------------------------------

def _put_many(store, count, *, prefix="vid", owner_uid=None):
    for index in range(count):
        video_id = f"{prefix}{index:07d}"
        store.put_map(
            video_id=video_id, difficulty="normal",
            song={"version": 2, "id": f"yt:{video_id}", "artist": "A Band",
                  "tempo": 120, "tonic": "G", "mode": "major",
                  "chordNames": ["G", "D", "Em", "C"]},
            sync=None, engine_chords="fake@1", engine_beats="fake@1",
            # Descending, so "newest first" has something to order by.
            analyzed_at=f"2026-08-{(index % 28) + 1:02d}T{index % 24:02d}:00:00Z",
            channel_id="UCtest", title=f"Song {index}", duration_ms=32_000,
            owner_uid=owner_uid,
        )


def test_the_catalog_pages_in_sql(store):
    """`limit`/`offset` must reach the database.

    They used to be a Python slice over the *whole table*: every row fetched, every
    `song_json` decoded, deduped in memory, and then sixty of them returned. So a
    catalog hit cost time linear in the size of the catalog forever, and `offset`
    bought nothing at all — asking for page 30 cost exactly what asking for page 1
    did.
    """
    _put_many(store, 25)

    first = store.list_catalog(limit=10, offset=0)
    second = store.list_catalog(limit=10, offset=10)
    tail = store.list_catalog(limit=10, offset=20)

    assert [len(page) for page in (first, second, tail)] == [10, 10, 5]

    # Disjoint, exhaustive, and in the same order one unpaged read gives — which is
    # the property that makes paging mean anything. Without it a client scrolling
    # the home screen sees rows twice and misses others.
    paged = [row.video_id for page in (first, second, tail) for row in page]
    whole = [row.video_id for row in store.list_catalog(limit=100)]
    assert paged == whole
    assert len(set(paged)) == 25

    # Newest first, tie-broken on video id, both descending.
    keys = [(row.analyzed_at, row.video_id) for row in store.list_catalog(limit=100)]
    assert keys == sorted(keys, reverse=True)


def test_the_catalog_reads_its_scalars_from_columns_not_from_the_payload(store):
    """The five fields a card needs, off their own columns.

    Reaching them through `song_json` meant `json.loads` per row — the cost that
    made paging pointless, for a listing that never looks at anything else in the
    payload.
    """
    _put_many(store, 1)
    row = store.list_catalog()[0]

    assert (row.song_id, row.artist, row.tempo, row.tonic, row.mode) == \
        ("yt:vid0000000", "A Band", 120, "G", "major")
    assert row.chord_names == ["G", "D", "Em", "C"]
    # And it is deliberately not a ChordMap: no payload came back at all.
    assert not hasattr(row, "song")


def test_a_video_at_two_difficulties_is_one_catalog_row(store):
    """The catalog lists songs, not analyses — collapsed in SQL now, and the page
    size therefore means "songs", which is what the caller asked for."""
    put(store, difficulty="easy")
    put(store, difficulty="hard")

    rows = store.list_catalog()
    assert [row.video_id for row in rows] == ["dQw4w9WgXcQ"]


def test_the_collapse_survives_paging(store):
    """The bug the SQL rewrite could plausibly have introduced: collapsing *after*
    the limit would make a page of ten duplicated rows come back as fewer than ten
    songs, and would drop songs off the end of the listing entirely."""
    for index in range(12):
        for tier in ("easy", "normal", "hard"):
            put(store, video_id=f"vid{index:07d}", difficulty=tier)

    page = store.list_catalog(limit=5, offset=0)
    assert len(page) == 5
    assert len({row.video_id for row in page}) == 5


def test_uploads_are_excluded_from_the_catalog_in_sql(store):
    """Filtered in the query rather than after, for the same reason the blocklist
    is: a listing is as much a way of serving something as the detail route."""
    _put_many(store, 3)
    _put_many(store, 2, prefix="up_abcdef", owner_uid="alice")

    assert len(store.list_catalog()) == 3
    assert all(not row.video_id.startswith("up_") for row in store.list_catalog())


def test_a_blocked_video_is_excluded_from_the_catalog(store):
    """Unchanged by the rewrite, and re-asserted because the rewrite moved the
    filter into a subquery where it could plausibly have stopped applying."""
    _put_many(store, 3)
    store.block(BLOCK_VIDEO, "vid0000001", reason="DMCA", actor="agent")

    assert [row.video_id for row in store.list_catalog()] == ["vid0000002", "vid0000000"]


def test_the_catalog_version_counts_only_public_rows(store):
    put(store)
    public = store.catalog_version()

    store.put_map(video_id="up_abcdef0123456789", difficulty="normal",
                  song={"version": 2, "id": "yt:up_x"}, sync=None,
                  engine_chords="f@1", engine_beats="f@1",
                  analyzed_at="2027-01-01T00:00:00Z", owner_uid="alice")

    assert store.catalog_version() == public


# --- the migration -----------------------------------------------------------

def test_a_database_written_before_the_new_columns_still_opens(tmp_path):
    """`CREATE TABLE IF NOT EXISTS` is silent about a table that already exists with
    fewer columns, so a deployed database gets the additions from `_migrate_columns`
    or not at all — and the first query naming `owner_uid` fails without them."""
    import sqlite3

    from app.store import SQLiteStore

    path = tmp_path / "old.sqlite3"
    legacy = sqlite3.connect(path)
    legacy.execute(
        """
        CREATE TABLE chord_maps (
            video_id TEXT NOT NULL, difficulty TEXT NOT NULL, channel_id TEXT,
            title TEXT, duration_ms INTEGER NOT NULL DEFAULT 0,
            song_json TEXT NOT NULL, sync_json TEXT, offset_ms INTEGER,
            low_confidence INTEGER NOT NULL DEFAULT 0, engine_chords TEXT NOT NULL,
            engine_beats TEXT NOT NULL, analyzed_at TEXT NOT NULL,
            PRIMARY KEY (video_id, difficulty)
        )
        """
    )
    legacy.execute(
        "INSERT INTO chord_maps VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        ("dQw4w9WgXcQ", "normal", "UCtest", "Known Song", 32_000,
         '{"version": 2, "id": "yt:dQw4w9WgXcQ", "artist": "A Band", "tempo": 120,'
         ' "tonic": "G", "mode": "major", "chordNames": ["G", "D"]}',
         None, None, 0, "fake@1", "fake@1", "2026-08-03T10:00:00Z"),
    )
    legacy.commit()
    legacy.close()

    store = SQLiteStore(str(path))
    try:
        # It opens, and the pre-existing row was backfilled rather than left blank —
        # otherwise the catalog would serve a card with no chords and no title.
        row = store.list_catalog()[0]
        assert row.video_id == "dQw4w9WgXcQ"
        assert row.chord_names == ["G", "D"]
        assert (row.tempo, row.tonic, row.mode, row.artist) == (120, "G", "major", "A Band")
        # And the legacy row is public: it predates uploads existing at all.
        assert store.get_map("dQw4w9WgXcQ", "normal").owner_uid is None
    finally:
        store.close()


def test_an_undecodable_payload_does_not_stop_the_store_from_opening(tmp_path):
    """One malformed cached song must not be able to refuse the whole service.

    A card with no chords on it is a far better failure than a store that will not
    open — and retrying the decode on every container start would make it permanent
    rather than one-off, which is what the flag column prevents.
    """
    import sqlite3

    from app.store import SQLiteStore

    path = tmp_path / "broken.sqlite3"
    store = SQLiteStore(str(path))
    put(store)
    store.close()

    handle = sqlite3.connect(path)
    handle.execute("UPDATE chord_maps SET song_json = 'not json', denormalized = 0")
    handle.commit()
    handle.close()

    reopened = SQLiteStore(str(path))
    try:
        assert reopened.list_catalog()[0].chord_names == []
    finally:
        reopened.close()


# --- refunds land on the day the charge did ---------------------------------

def test_a_refund_credits_the_day_the_charge_was_made(store):
    """An analysis can be queued and run across midnight.

    `refund_use` always decremented *today's* row, so a job charged at 23:59 and
    failed at 00:01 did the player double harm: yesterday stayed exhausted, and
    today's fresh allowance absorbed a refund it was owed nothing for.
    """
    from app.store import utc_day

    yesterday = "2026-08-16"
    with store._cursor(write=True) as cur:
        cur.execute(store._sql("INSERT INTO usage (uid, day, count) VALUES (?, ?, ?)"),
                    ("u1", yesterday, 3))

    store.refund_use("u1", yesterday)

    with store._cursor() as cur:
        cur.execute(store._sql("SELECT count FROM usage WHERE uid = ? AND day = ?"),
                    ("u1", yesterday))
        assert cur.fetchone()[0] == 2
    # And today is untouched — it was never charged.
    assert store.usage_today("u1") == 0
    assert utc_day() != yesterday


def test_the_charge_day_comes_off_the_job_row(store):
    """The charge happens in the request that creates the row, so the row's
    `created_at` *is* the day it was charged to."""
    from app.store import day_of

    job = store.create_job(job_id="j1", uid="u1", video_id="dQw4w9WgXcQ",
                           difficulty="normal")
    assert store.charge_day_for_job("j1") == day_of(job.created_at)
    assert store.charge_day_for_job("nope") is None


# --- job followers -----------------------------------------------------------

def test_a_follower_may_read_the_job_they_joined(store):
    job = store.create_job(job_id="j1", uid="alice", video_id="dQw4w9WgXcQ",
                           difficulty="normal")

    assert store.may_read_job(job, "alice")
    assert not store.may_read_job(job, "bob")

    store.follow_job("j1", "bob")
    assert store.may_read_job(job, "bob")
    assert not store.may_read_job(job, "carol")


def test_following_twice_is_not_an_error(store):
    """The client may well ask for the same video twice while it is running."""
    job = store.create_job(job_id="j1", uid="alice", video_id="dQw4w9WgXcQ",
                           difficulty="normal")
    store.follow_job("j1", "bob")
    store.follow_job("j1", "bob")

    assert store.may_read_job(job, "bob")


def test_followers_are_collected_with_the_jobs_they_point_at(store):
    """There is no foreign key in this schema, so an orphaned follower row would
    accumulate forever holding a uid."""
    store.create_job(job_id="j1", uid="alice", video_id="dQw4w9WgXcQ",
                     difficulty="normal")
    store.follow_job("j1", "bob")
    store.update_job("j1", status=STATUS_READY, progress=1.0)

    store.prune_jobs(older_than_s=-1)

    with store._cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM job_followers")
        assert cur.fetchone()[0] == 0


def test_a_purge_takes_the_followers_with_it(store):
    store.create_job(job_id="j1", uid="alice", video_id="dQw4w9WgXcQ",
                     difficulty="normal")
    store.follow_job("j1", "bob")

    store.purge("dQw4w9WgXcQ")

    with store._cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM job_followers")
        assert cur.fetchone()[0] == 0


def test_concurrent_callers_cannot_both_squeeze_past_the_limit(store):
    """The limiter admits exactly `limit`, under real contention.

    SQLite gets this for free — `SQLiteStore` holds a process-wide lock for the whole
    cursor block, and SQLite takes a database write lock on top — so this passes here
    even against the racy version. It is pinned anyway because it is the *definition*
    of the limiter, and because the Postgres backend needed an advisory lock
    (`_serialize_rate_key`) to make the same statement true. A sequential test cannot
    tell the two apart, which is why this one uses threads.
    """
    import threading

    limit = 5
    callers = 24
    allowed: list[bool] = []
    lock = threading.Lock()
    start = threading.Barrier(callers)

    def hammer():
        start.wait()
        ok, _retry = store.hit_rate_limit(RATE_SCOPE_UID, "uid", limit, 60.0)
        with lock:
            allowed.append(ok)

    threads = [threading.Thread(target=hammer) for _ in range(callers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sum(allowed) == limit


def test_sqlite_needs_no_extra_serialization(store):
    """`_serialize_rate_key` is a no-op here on purpose, and saying so out loud is
    what stops someone adding a lock this backend does not need — or removing the one
    Postgres does."""
    with store._cursor() as cur:
        assert store._serialize_rate_key(cur, RATE_SCOPE_UID, "uid") is None


def test_the_catalog_can_be_filtered_to_one_difficulty(store):
    """An unused parameter is still a parameter, and the SQL rewrite moved it into
    a subquery where its placeholder order could plausibly have gone wrong."""
    put(store, video_id="aaaaaaaaaaa", difficulty="easy")
    put(store, video_id="bbbbbbbbbbb", difficulty="hard")

    easy = store.list_catalog(difficulty="easy")
    assert [row.video_id for row in easy] == ["aaaaaaaaaaa"]
    assert easy[0].difficulty == "easy"
    assert len(store.list_catalog()) == 2


def test_the_backfill_covers_more_rows_than_one_batch(tmp_path):
    """Read in batches so the migration does not hold every payload in the table at
    once — which is the cost the columns exist to remove, reintroduced at startup
    where it is least visible. So it has to loop correctly past the batch size."""
    import sqlite3

    from app.store import SQLiteStore

    path = tmp_path / "many.sqlite3"
    store = SQLiteStore(str(path))
    for index in range(450):          # > the 200-row batch
        put(store, video_id=f"vid{index:08d}")
    store.close()

    handle = sqlite3.connect(path)
    handle.execute("UPDATE chord_maps SET denormalized = 0, song_id = NULL")
    handle.commit()
    handle.close()

    reopened = SQLiteStore(str(path))
    try:
        with reopened._cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chord_maps WHERE denormalized = 0")
            assert cur.fetchone()[0] == 0
            cur.execute("SELECT COUNT(*) FROM chord_maps WHERE song_id IS NULL")
            assert cur.fetchone()[0] == 0
    finally:
        reopened.close()
