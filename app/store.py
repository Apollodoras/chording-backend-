"""The persistent store — and the one place handoff §2's invariants become code.

**Two backends, one definition of the semantics** (the shape Mo's ``app/store.py``
settled on, kept deliberately): ``Store`` holds every statement and every rule;
``SQLiteStore`` and ``PostgresStore`` supply only a connection and a placeholder
style. SQLite is the zero-config default for local runs and the test-suite;
Postgres switches on when ``CHORDS_DATABASE_URL`` is set.

What this store may hold is not a design preference — it is §2.2, and the schema
is where it is enforced:

- ``chord_maps`` holds the **derived song only**: the `CompositionPayload` JSON
  and the sidecar. Chord symbols, timestamps, a beat grid, key, tempo,
  confidences. **No PCM, no spectrogram, no chroma matrix, no path to a decoded
  file** — there is no column that could hold one, which is the point. A future
  "just cache the chroma so re-analysis is cheap" change has to add a column and
  argue with this docstring first.
- ``blocklist`` + ``audit_log`` are §3's takedown surface. The audit log is
  **append-only**: there is no update or delete path to it anywhere in this file.

``purge`` is the operation a takedown request actually needs, and it is written
to cascade — maps, jobs, and the analyses that reference them — because "we
blocked it" while a cached map keeps being served is the failure mode that gets a
DMCA agent's attention.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from datetime import time as dt_time
from typing import Any, Optional

from .errors import CODE_ANALYSIS_FAILED

log = logging.getLogger("chords.store")

RATE_SCOPE_UID = "uid"
RATE_SCOPE_IP = "ip"
# Cheap authenticated reads, counted against their own budget and keyed by uid
# like `RATE_SCOPE_UID`. A separate *scope* rather than a separate limit on the
# same one: sharing the scope would mean a poll still consumed a slot the
# spending routes need, which is the defect this exists to fix.
RATE_SCOPE_POLL = "poll"

# Blocklist kinds (§3): a takedown may name one video or a whole channel.
BLOCK_VIDEO = "video"
BLOCK_CHANNEL = "channel"

# Audit actions — the append-only record of who did what (§3).
AUDIT_BLOCK = "block"
AUDIT_UNBLOCK = "unblock"
AUDIT_PURGE = "purge"
AUDIT_OFFSET = "set_offset"

# Job statuses (§16.1). `blocked` and `unavailable` are terminal states that are
# not failures of ours, and the client renders them as calm states, not errors
# (§17.8).
STATUS_QUEUED = "queued"
STATUS_FETCHING = "fetching"
STATUS_ANALYZING = "analyzing"
STATUS_READY = "ready"
STATUS_FAILED = "failed"
STATUS_BLOCKED = "blocked"
STATUS_UNAVAILABLE = "unavailable"

TERMINAL_STATUSES = {STATUS_READY, STATUS_FAILED, STATUS_BLOCKED, STATUS_UNAVAILABLE}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def utc_day() -> str:
    """The quota day key, in **UTC** — so the reset the app promises the player
    doesn't depend on the deployment's timezone."""
    return datetime.now(timezone.utc).date().isoformat()


def next_utc_reset() -> datetime:
    """When today's quota rolls over: the next UTC midnight."""
    tomorrow = datetime.now(timezone.utc).date() + timedelta(days=1)
    return datetime.combine(tomorrow, dt_time.min, tzinfo=timezone.utc)


def day_of(iso_timestamp: str | None) -> str:
    """The UTC quota day an ISO-8601 timestamp falls in.

    Used to refund the charge on the day it was **made**. `refund_use` used to
    always decrement today's row, so a job that started at 23:59 and failed at
    00:01 credited a day the player had not spent anything on — and left
    yesterday's exhausted quota exhausted.
    """
    if not iso_timestamp:
        return utc_day()
    # Everything this store writes is `_now_iso()`'s `…Z`; be tolerant anyway,
    # since a hand-edited or migrated row is not worth an exception.
    return iso_timestamp[:10] if len(iso_timestamp) >= 10 else utc_day()


def _decode_chord_names(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        names = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return [str(name) for name in names] if isinstance(names, list) else []


def _catalog_scalars(song: dict) -> tuple[str | None, str | None, int | None,
                                          str | None, str | None, str]:
    """The five catalog fields, read off a `CompositionPayload` wire dict.

    One function so `put_map` and the backfill cannot disagree about what the
    columns mean. `chord_names` is stored as JSON because it is the one non-scalar
    — a short list of symbols the card prints, never queried on.
    """
    if not isinstance(song, dict):
        song = {}
    tempo = song.get("tempo")
    return (
        song.get("id"),
        song.get("artist"),
        int(tempo) if isinstance(tempo, (int, float)) else None,
        song.get("tonic"),
        song.get("mode"),
        json.dumps(song.get("chordNames") or [], ensure_ascii=False),
    )


# ---------------------------------------------------------------------------
# Schema — dialect-neutral except the auto-increment key, which each backend
# supplies via `_ID_COLUMN`.
# ---------------------------------------------------------------------------

_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS users (
        uid          TEXT PRIMARY KEY,
        display_name TEXT,
        created_at   TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS usage (
        uid   TEXT NOT NULL,
        day   TEXT NOT NULL,
        count INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (uid, day)
    )
    """,
    # One row per accepted request inside the sliding window. In the database
    # rather than a process dictionary: a limiter that resets on container
    # recycle is a limiter an attacker resets by waiting for a cold start.
    """
    CREATE TABLE IF NOT EXISTS rate_events (
        scope TEXT NOT NULL,
        key   TEXT NOT NULL,
        ts    DOUBLE PRECISION NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS rate_events_lookup ON rate_events (scope, key, ts)",
    # §2.2 — the derived map, and NOTHING from which audio could be
    # reconstructed. **One row per video**: the §5.5 difficulty tiers used to put
    # three here, keyed `(video_id, difficulty)`, and there is one chart now.
    # `song_json` is a CompositionPayload v2; `sync_json` is the §13 sidecar
    # (nullable — anchors that are not a usable map are filed without one rather
    # than with a faked grid; since the §13.3 amendment that is the only thing
    # that empties this column). `engine_chords`/`engine_beats` are stored per §5.3 so a
    # cache can be invalidated selectively when one engine is upgraded.
    #
    # `owner_uid` is what makes an **upload** private. Uploaded audio lands in
    # this table exactly like a fetched video, and until this column existed the
    # catalog listed it and `GET /v1/maps/{id}` served it to anyone who guessed
    # the hash — one player's private recording, with their own filename on it,
    # on every other player's home screen. NULL means "public, came from a
    # video"; a uid means "only this player may read it, and it is not catalog
    # material". Nullable rather than defaulted so the distinction is a fact
    # about the row and not about when the row was written.
    #
    # `song_id`/`artist`/`tempo`/`tonic`/`mode`/`chord_names` duplicate five
    # scalars out of `song_json`, and the duplication is deliberate: the catalog
    # reads exactly those and nothing else, and reading them *through*
    # `song_json` meant `json.loads` on every row of the table to return a page
    # of sixty. They are written from the payload by `put_map`, which is the only
    # writer, so the two cannot drift.
    """
    CREATE TABLE IF NOT EXISTS chord_maps (
        video_id      TEXT NOT NULL,
        channel_id    TEXT,
        title         TEXT,
        duration_ms   INTEGER NOT NULL DEFAULT 0,
        song_json     TEXT NOT NULL,
        sync_json     TEXT,
        offset_ms     INTEGER,
        low_confidence INTEGER NOT NULL DEFAULT 0,
        engine_chords TEXT NOT NULL,
        engine_beats  TEXT NOT NULL,
        analyzed_at   TEXT NOT NULL,
        owner_uid     TEXT,
        song_id       TEXT,
        artist        TEXT,
        tempo         INTEGER,
        tonic         TEXT,
        mode          TEXT,
        chord_names   TEXT,
        PRIMARY KEY (video_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS chord_maps_channel ON chord_maps (channel_id)",
    # Job lifecycle (§16.1). A job is identified by its video: `video_id` points
    # at the chord_maps row a finished job produced, so a purge can find and
    # clear the jobs that would otherwise keep handing out a map that no longer
    # exists.
    """
    CREATE TABLE IF NOT EXISTS jobs (
        job_id            TEXT PRIMARY KEY,
        uid               TEXT NOT NULL,
        video_id          TEXT NOT NULL,
        status            TEXT NOT NULL,
        progress          DOUBLE PRECISION NOT NULL DEFAULT 0,
        error_code        TEXT,
        error_message     TEXT,
        created_at        TEXT NOT NULL,
        updated_at        TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS jobs_video ON jobs (video_id)",
    "CREATE INDEX IF NOT EXISTS jobs_uid ON jobs (uid)",
    # Who else is waiting on a job they did not start.
    #
    # `active_job_for` deliberately joins a second caller onto one in-flight
    # analysis rather than decoding the same recording twice — but the job row
    # carries a single `uid`, and `GET /v1/analyze/{jobId}` refuses a poll from
    # anyone else. So the second player was handed an id and then told, forever,
    # that it had expired; they retried, got the same dead id, and the video was
    # un-analyzable for everybody except whoever asked first.
    #
    # A follower row is the smallest thing that fixes it without weakening the
    # check: you may poll a job because you *joined* it, not because you guessed
    # its id. Nobody is charged for following — joining a running analysis is
    # free for the same reason a cache hit is (§16.4).
    """
    CREATE TABLE IF NOT EXISTS job_followers (
        job_id     TEXT NOT NULL,
        uid        TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (job_id, uid)
    )
    """,
    # §3 — per-video-ID and per-channel-ID.
    """
    CREATE TABLE IF NOT EXISTS blocklist (
        kind       TEXT NOT NULL,
        key        TEXT NOT NULL,
        reason     TEXT,
        actor      TEXT,
        created_at TEXT NOT NULL,
        PRIMARY KEY (kind, key)
    )
    """,
    # §3 — append-only. Nothing in this file updates or deletes from it, and
    # `purge` deliberately does not touch it: the record of a takedown must
    # outlive the thing taken down.
    """
    CREATE TABLE IF NOT EXISTS audit_log (
        {id_column},
        action     TEXT NOT NULL,
        kind       TEXT,
        key        TEXT NOT NULL,
        actor      TEXT,
        reason     TEXT,
        detail     TEXT,
        created_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS audit_log_key ON audit_log (key)",
    "CREATE INDEX IF NOT EXISTS audit_log_created ON audit_log (created_at)",
]

# Columns added to an existing table after it was first deployed.
#
# `CREATE TABLE IF NOT EXISTS` is silent about a table that already exists with
# *fewer* columns, so a database written by an earlier version gets these here or
# not at all — and "not at all" means the privacy filter below cannot see
# `owner_uid` and every query naming it fails. Additive and nullable only: this
# is a migration path, not a schema tool, and anything needing a backfill or a
# rewrite deserves to be written by hand and read by a person.
_ADDED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("chord_maps", "owner_uid", "TEXT"),
    ("chord_maps", "song_id", "TEXT"),
    ("chord_maps", "artist", "TEXT"),
    ("chord_maps", "tempo", "INTEGER"),
    ("chord_maps", "tonic", "TEXT"),
    ("chord_maps", "mode", "TEXT"),
    ("chord_maps", "chord_names", "TEXT"),
    # Set once the row's five catalog scalars have been lifted out of
    # `song_json`. A flag rather than "is `song_id` still NULL?" so a song that
    # legitimately has no id is not re-read from JSON on every container start.
    ("chord_maps", "denormalized", "INTEGER NOT NULL DEFAULT 0"),
)

# Indexes that name a column from `_ADDED_COLUMNS`, and so cannot be created until
# after the migration has run. `CREATE TABLE IF NOT EXISTS` is a no-op against an
# existing table, so putting these in `_SCHEMA` meant every already-deployed
# database failed to open on `no such column: owner_uid` — the schema list and the
# migration are two different mechanisms and only one of them adds columns.
_POST_MIGRATION_SCHEMA = [
    # The catalog's own ordering, so paging it is an index walk rather than a sort
    # of the whole table. Ends in `video_id` because that is the tiebreak
    # `list_catalog` orders by, and a partial index match would leave the sort in
    # place for exactly the rows the first page is made of.
    "CREATE INDEX IF NOT EXISTS chord_maps_catalog ON chord_maps (analyzed_at DESC, video_id DESC)",
    "CREATE INDEX IF NOT EXISTS chord_maps_owner ON chord_maps (owner_uid)",
]


@dataclass(frozen=True)
class ChordMap:
    """A cached analysis, as it comes back out of the store."""

    video_id: str
    song: dict
    sync: Optional[dict]
    offset_ms: Optional[int]
    low_confidence: bool
    engine_chords: str
    engine_beats: str
    analyzed_at: str
    channel_id: Optional[str] = None
    title: Optional[str] = None
    duration_ms: int = 0
    # None for a fetched video — public, and catalog material. A uid for
    # uploaded audio, which only its uploader may read (see the schema note).
    owner_uid: Optional[str] = None

    @property
    def is_private(self) -> bool:
        return self.owner_uid is not None


@dataclass(frozen=True)
class CatalogEntry:
    """One row of the catalog listing — the five scalars a card needs, and
    nothing else.

    Deliberately **not** a `ChordMap`: a `ChordMap` carries the whole
    `CompositionPayload`, and the listing was decoding two thousand of them to
    print sixty titles. These fields come straight out of their own columns.
    """

    video_id: str
    title: Optional[str]
    artist: Optional[str]
    duration_ms: int
    song_id: Optional[str]
    chord_names: list[str]
    tempo: Optional[int]
    tonic: Optional[str]
    mode: Optional[str]
    analyzed_at: str
    low_confidence: bool


@dataclass(frozen=True)
class Job:
    job_id: str
    uid: str
    video_id: str
    status: str
    progress: float
    error_code: Optional[str]
    error_message: Optional[str]
    created_at: str
    updated_at: str

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES


class Store:
    """Every statement and every rule. Backends supply only the connection.

    Statements are written with `?` placeholders (the SQLite style) and
    translated per-backend by `_sql`, so there is exactly one copy of each query.
    """

    _ID_COLUMN = "id INTEGER PRIMARY KEY AUTOINCREMENT"
    _PARAMSTYLE = "qmark"

    _RATE_PRUNE_INTERVAL_S = 600.0
    _last_rate_prune = 0.0

    # How long a finished job row is kept. `GET /v1/analyze/{jobId}` tells a
    # caller with an unknown id that the analysis "has expired" — which was a
    # lie for as long as nothing expired, since rows were only ever removed by a
    # takedown. A day is comfortably longer than any client would keep polling
    # and short enough that the table stays small.
    _JOB_TTL_S = 86_400.0
    _JOB_PRUNE_INTERVAL_S = 3_600.0
    _last_job_prune = 0.0

    # -- backend seam --------------------------------------------------------

    @contextmanager
    def _cursor(self, *, write: bool = False):
        """Yield a cursor; commit on clean exit of a write."""
        raise NotImplementedError

    def _sql(self, sql: str) -> str:
        if self._PARAMSTYLE == "qmark":
            return sql
        return sql.replace("?", "%s")

    def _columns(self, cur, table: str) -> set[str]:
        """Which columns `table` currently has. Backend-specific, and the reason
        `_migrate_columns` can be additive without guessing."""
        raise NotImplementedError

    def _serialize_rate_key(self, cur, scope: str, key: str) -> None:
        """Serialize concurrent limiter checks on one `(scope, key)`.

        A no-op here, because `SQLiteStore` already holds a process-wide lock for
        the whole cursor block and SQLite itself takes a database write lock —
        the read-count-then-insert below is atomic there by construction.
        Postgres is the backend that needs it: under READ COMMITTED two
        concurrent transactions both see a count below the limit, both insert,
        and the limiter admits `2 × limit` requests. See `PostgresStore`.
        """

    def ping(self) -> None:
        """One round-trip, for `/healthz`. Raises if the store is unreachable.

        Deliberately a real query rather than "is the object constructed": the
        Postgres store builds its pool lazily, so an unreachable database looks
        perfectly healthy right up until the first request that needs a row.
        """
        with self._cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()

    def _migrate(self) -> None:
        self._drop_difficulty()
        with self._cursor(write=True) as cur:
            for statement in _SCHEMA:
                cur.execute(statement.format(id_column=self._ID_COLUMN))
        self._migrate_columns()
        with self._cursor(write=True) as cur:
            for statement in _POST_MIGRATION_SCHEMA:
                cur.execute(statement)
        self._denormalize_catalog()

    def _drop_difficulty(self) -> None:
        """Collapse the §5.5 difficulty tiers out of an already-written database.

        The one migration in this file that is neither additive nor nullable, and
        so the one written by hand and meant to be read by a person. It runs
        **before** `_SCHEMA`, because it drops and recreates the two tables and
        `CREATE TABLE IF NOT EXISTS` then puts back every index that went with
        them. On a database that never had the column — a fresh one, or one
        already migrated — `_columns` comes back without it and this is two
        cheap lookups.

        `chord_maps` cannot be migrated with `ALTER TABLE … DROP COLUMN`:
        `difficulty` is half of its primary key. So the table is rebuilt, and
        rebuilding forces the question of *which* of a video's three rows is the
        one row it keeps. It keeps **`hard`** — that was the reference render,
        the one built at the full grammar, and the only one of the three that
        ever claimed to state what was played. `easy` and `normal` were
        reductions of it, and a reduction is exactly what this change exists to
        stop shipping, so they are dropped rather than merged or preferred. A
        video that somehow has no `hard` row keeps `normal`, then `easy`, then
        the most recent of whatever is left; the `ORDER BY` says so.

        `jobs` is rebuilt the same way rather than with `DROP COLUMN`, which is
        available on both backends but not on every SQLite a container might
        carry (3.35+). Job rows survive the copy; they are short-lived and
        lease-reaped either way, but losing one would strand a client polling it.
        """
        keep = ("CASE difficulty WHEN 'hard' THEN 0 WHEN 'normal' THEN 1 "
                "ELSE 2 END, analyzed_at DESC")
        rebuilds = (
            # table, the row to keep per key, the key
            ("chord_maps", keep, "video_id"),
            ("jobs", None, None),
        )
        for table, order, key in rebuilds:
            with self._cursor(write=True) as cur:
                present = self._columns(cur, table)
                if "difficulty" not in present:
                    continue
                ddl = next(st for st in _SCHEMA
                           if f"CREATE TABLE IF NOT EXISTS {table} (" in st)
                staging = f"{table}_migrating"
                cur.execute(ddl.format(id_column=self._ID_COLUMN)
                            .replace(f"IF NOT EXISTS {table} (", f"{staging} ("))
                # The intersection, so this is correct whichever columns the old
                # database happens to have — `_migrate_columns` has not run yet.
                columns = [c for c in self._columns(cur, staging) if c in present]
                names = ", ".join(columns)
                where = "" if order is None else (
                    f" WHERE difficulty = (SELECT d.difficulty FROM {table} d "
                    f"WHERE d.{key} = {table}.{key} ORDER BY "
                    f"{order.replace('difficulty', 'd.difficulty').replace('analyzed_at', 'd.analyzed_at')}"
                    " LIMIT 1)"
                )
                cur.execute(f"INSERT INTO {staging} ({names}) "
                            f"SELECT {names} FROM {table}{where}")
                cur.execute(f"DROP TABLE {table}")
                cur.execute(f"ALTER TABLE {staging} RENAME TO {table}")
            log.info("migrating: dropped the difficulty tiers from %s", table)

    def _migrate_columns(self) -> None:
        """Add any column in `_ADDED_COLUMNS` this database is missing."""
        wanted: dict[str, list[tuple[str, str]]] = {}
        for table, column, ddl in _ADDED_COLUMNS:
            wanted.setdefault(table, []).append((column, ddl))
        with self._cursor(write=True) as cur:
            for table, columns in wanted.items():
                present = self._columns(cur, table)
                for column, ddl in columns:
                    if column in present:
                        continue
                    log.info("migrating: adding %s.%s", table, column)
                    cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def _denormalize_catalog(self) -> None:
        """Lift the catalog's five scalars out of `song_json`, once per row.

        The listing used to read them by decoding every payload in the table, so
        the columns have to be populated for rows written before they existed.
        Bounded and one-off: the flag column means a container start after the
        backfill does one indexed count, not a table scan.

        Best-effort by design. A row whose payload cannot be decoded is left
        flagged as done with empty scalars rather than retried on every start —
        the alternative is a store that refuses to open because one cached song
        is malformed, which is a worse failure than one card with no chords on
        it.

        Read in **batches**, because the whole point of the columns is not holding
        every payload in the table at once: selecting them all here would
        reintroduce the memory cost of the query being replaced, at startup, where
        it is least visible.
        """
        batch = 200
        total = 0
        while True:
            with self._cursor() as cur:
                cur.execute(self._sql(
                    "SELECT video_id, song_json FROM chord_maps "
                    "WHERE denormalized = 0 LIMIT ?"
                ), (batch,))
                rows = cur.fetchall()
            if not rows:
                break
            for video_id, song_json in rows:
                try:
                    song = json.loads(song_json) if song_json else {}
                except (TypeError, ValueError):
                    log.warning("chord map %s has undecodable song_json", video_id)
                    song = {}
                with self._cursor(write=True) as cur:
                    cur.execute(self._sql(
                        """
                        UPDATE chord_maps
                        SET song_id = ?, artist = ?, tempo = ?, tonic = ?, mode = ?,
                            chord_names = ?, denormalized = 1
                        WHERE video_id = ?
                        """
                    ), (*_catalog_scalars(song), video_id))
            total += len(rows)
        if total:
            log.info("migrating: denormalized %d chord map(s) for the catalog", total)

    # -- users ---------------------------------------------------------------

    def upsert_user(self, uid: str, display_name: str | None) -> None:
        with self._cursor(write=True) as cur:
            cur.execute(self._sql(
                """
                INSERT INTO users (uid, display_name, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(uid) DO UPDATE SET
                    display_name = COALESCE(EXCLUDED.display_name, users.display_name)
                """
            ), (uid, display_name, _now_iso()))

    def display_name(self, uid: str) -> str | None:
        with self._cursor() as cur:
            cur.execute(self._sql("SELECT display_name FROM users WHERE uid = ?"), (uid,))
            row = cur.fetchone()
        return row[0] if row else None

    # -- daily quota ---------------------------------------------------------

    def usage_today(self, uid: str) -> int:
        with self._cursor() as cur:
            cur.execute(
                self._sql("SELECT count FROM usage WHERE uid = ? AND day = ?"),
                (uid, utc_day()),
            )
            row = cur.fetchone()
        return row[0] if row else 0

    def try_record_use(self, uid: str, limit: int) -> tuple[bool, int]:
        """Charge one analysis if the user is under `limit` — check and record in
        ONE conditional statement, so two concurrent requests can't both squeeze
        past a cap only one of them was entitled to.

        **Cache hits never reach this** (§16.4): a cached map costs nothing, so
        it must not cost the player anything either.
        """
        if limit <= 0:
            return False, self.usage_today(uid)
        day = utc_day()
        with self._cursor(write=True) as cur:
            cur.execute(self._sql(
                """
                INSERT INTO usage (uid, day, count) VALUES (?, ?, 1)
                ON CONFLICT(uid, day) DO UPDATE SET count = usage.count + 1
                    WHERE usage.count < ?
                RETURNING count
                """
            ), (uid, day, limit))
            row = cur.fetchone()
            if row is not None:
                return True, row[0]
            cur.execute(
                self._sql("SELECT count FROM usage WHERE uid = ? AND day = ?"), (uid, day)
            )
            blocked = cur.fetchone()
        return False, blocked[0] if blocked else limit

    def refund_use(self, uid: str, day: str | None = None) -> None:
        """Give back a charge whose analysis never happened.

        Analysis can fail for reasons that are not the player's doing and cost us
        nothing worth billing — a video that turns out to be blocked, private, or
        too long is rejected before a single second is decoded. Charging for that
        would burn a daily quota on an error message.

        `day` is **the day the charge was made**, not today. Analyses outlive
        midnight: a job that started at 23:59 and failed at 00:01 was charged to
        yesterday's row, and crediting today's instead did the player double harm
        — yesterday stayed exhausted, and today's fresh allowance absorbed a
        refund it was owed nothing for. Callers that know the job pass
        `charge_day_for_job`; the default is today, which is right for a failure
        that happens in the same request that charged.
        """
        with self._cursor(write=True) as cur:
            cur.execute(self._sql(
                "UPDATE usage SET count = count - 1 WHERE uid = ? AND day = ? AND count > 0"
            ), (uid, day or utc_day()))

    def charge_day_for_job(self, job_id: str) -> str | None:
        """Which quota day this job's charge landed on, or None if it is gone.

        The charge happens in the request that creates the row, so the row's
        `created_at` *is* the day it was charged to.
        """
        with self._cursor() as cur:
            cur.execute(self._sql("SELECT created_at FROM jobs WHERE job_id = ?"), (job_id,))
            row = cur.fetchone()
        return day_of(row[0]) if row else None

    # -- rate limiting -------------------------------------------------------

    def hit_rate_limit(self, scope: str, key: str, limit: int, window_s: float,
                       now: float | None = None) -> tuple[bool, float]:
        """Record one request against a **sliding window** and say whether it is
        allowed. Returns `(allowed, retry_after_s)`.

        Sliding rather than a fixed bucket because a fixed bucket lets a caller
        fire `limit` requests at 11:59:59 and `limit` more at 12:00:00.

        Count-then-insert, which is only atomic because the whole block is one
        transaction **and** concurrent checks of the same key are serialized —
        see `_serialize_rate_key`. Without that, Postgres under READ COMMITTED
        lets two simultaneous requests both read a count below the limit and both
        insert, so the budget quietly doubles under exactly the load it exists
        for. SQLite never had the problem; Postgres is what this deployment runs.
        """
        if limit <= 0:
            return True, 0.0
        now = time.time() if now is None else now
        self._maybe_prune_rate_events(window_s, now)
        cutoff = now - window_s
        with self._cursor(write=True) as cur:
            self._serialize_rate_key(cur, scope, key)
            cur.execute(
                self._sql("DELETE FROM rate_events WHERE scope = ? AND key = ? AND ts <= ?"),
                (scope, key, cutoff),
            )
            cur.execute(
                self._sql("SELECT COUNT(*), MIN(ts) FROM rate_events WHERE scope = ? AND key = ?"),
                (scope, key),
            )
            row = cur.fetchone()
            count, oldest = (row[0] or 0), row[1]
            if count >= limit:
                # Refused requests are NOT recorded: counting them would let a
                # caller who keeps hammering hold their own window open forever.
                retry_after = max(0.0, (oldest + window_s) - now) if oldest is not None else window_s
                return False, retry_after
            cur.execute(
                self._sql("INSERT INTO rate_events (scope, key, ts) VALUES (?, ?, ?)"),
                (scope, key, now),
            )
        return True, 0.0

    def _maybe_prune_rate_events(self, window_s: float, now: float) -> None:
        """Occasional global prune, from whoever happens to be writing.

        `hit_rate_limit` prunes only the key it was asked about, so a key seen
        once keeps its row forever — and in the IP scope that key **is an IP
        address**, i.e. personal data accumulating in a table nobody reads.
        Best-effort: housekeeping must never fail a request that was allowed.
        """
        if now - self._last_rate_prune < self._RATE_PRUNE_INTERVAL_S:
            return
        self._last_rate_prune = now
        try:
            self.prune_rate_events(older_than_s=window_s, now=now)
        except Exception:  # pragma: no cover - housekeeping, never fatal
            log.warning("rate-event prune failed (request unaffected)", exc_info=True)

    def prune_rate_events(self, older_than_s: float, now: float | None = None) -> int:
        now = time.time() if now is None else now
        with self._cursor(write=True) as cur:
            cur.execute(self._sql("DELETE FROM rate_events WHERE ts <= ?"), (now - older_than_s,))
            return cur.rowcount

    def _maybe_prune_jobs(self, now: float | None = None) -> None:
        """Occasional sweep of finished job rows, from whoever creates the next.

        Same shape and same reasoning as the rate-event prune: no scheduler to
        depend on, and housekeeping must never fail the request that triggered
        it. **Only terminal rows** — a job still queued or analyzing is the one
        thing in this table that someone is waiting on.
        """
        now = time.time() if now is None else now
        if now - self._last_job_prune < self._JOB_PRUNE_INTERVAL_S:
            return
        self._last_job_prune = now
        try:
            self.prune_jobs(older_than_s=self._JOB_TTL_S, now=now)
        except Exception:  # pragma: no cover - housekeeping, never fatal
            log.warning("job prune failed (request unaffected)", exc_info=True)

    def prune_jobs(self, older_than_s: float, now: float | None = None) -> int:
        now = time.time() if now is None else now
        cutoff = datetime.fromtimestamp(now - older_than_s, tz=timezone.utc)
        placeholders = ", ".join("?" for _ in TERMINAL_STATUSES)
        stamp = cutoff.isoformat().replace("+00:00", "Z")
        with self._cursor(write=True) as cur:
            # Followers first, while the rows they point at still exist to be
            # selected by. There is no foreign key here (neither backend's schema
            # in this file declares one), so an orphaned follower row would simply
            # accumulate forever holding a uid — see `_maybe_prune_rate_events` on
            # why that matters beyond tidiness.
            cur.execute(
                self._sql(
                    f"""DELETE FROM job_followers WHERE job_id IN (
                            SELECT job_id FROM jobs
                            WHERE status IN ({placeholders}) AND updated_at < ?
                        )"""
                ),
                (*sorted(TERMINAL_STATUSES), stamp),
            )
            cur.execute(
                self._sql(
                    f"DELETE FROM jobs WHERE status IN ({placeholders}) AND updated_at < ?"
                ),
                (*sorted(TERMINAL_STATUSES), stamp),
            )
            return cur.rowcount

    # -- the abandoned-job lease ---------------------------------------------
    #
    # `run_job` writes a terminal status on every exit path *it* controls. A
    # container killed from outside controls none of them: Modal's `timeout=`
    # SIGKILLs the worker, an OOM kill takes it at the memory cap, and a spot
    # reclaim takes it for nothing at all. In each case the row is left mid-flight
    # forever, and two things then compound:
    #
    #   - the player polls `analyzing` until they give up; nothing ever answers,
    #   - `active_job_for` keeps handing that dead id to *everyone else* asking
    #     for the same video, so one killed worker makes one video permanently
    #     un-analyzable — and `prune_jobs` can't help, since it only deletes rows
    #     that already reached a terminal status.
    #
    # So a non-terminal row carries a lease. Past it, the job is presumed dead:
    # marked failed (giving the poller a terminal answer and the pruner something
    # to collect) and refunded, because the player got nothing for it.
    #
    # Comfortably above the worker's own 300 s timeout — the lease must expire
    # only after the container that would have written the row is certainly gone,
    # including time spent queued waiting for one to start.
    _JOB_LEASE_S = 900.0
    _JOB_REAP_INTERVAL_S = 60.0
    _last_job_reap = 0.0

    def _maybe_reap_stale_jobs(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        if now - self._last_job_reap < self._JOB_REAP_INTERVAL_S:
            return
        self._last_job_reap = now
        try:
            reaped = self.reap_stale_jobs(older_than_s=self._JOB_LEASE_S, now=now)
            if reaped:
                log.warning("reaped %d abandoned job(s) — a worker died without "
                            "writing a terminal status", reaped)
        except Exception:  # pragma: no cover - housekeeping, never fatal
            log.warning("job reap failed (request unaffected)", exc_info=True)

    def reap_stale_jobs(self, older_than_s: float, now: float | None = None) -> int:
        """Fail every non-terminal job whose lease has expired, and refund it.

        The refund is read-then-write rather than one statement because the uid
        is needed to credit it back, and `usage` is a different table. A row is
        claimed by the UPDATE before its refund is issued, so two containers
        reaping the same job concurrently cannot both refund it.
        """
        now = time.time() if now is None else now
        cutoff = datetime.fromtimestamp(now - older_than_s, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        placeholders = ", ".join("?" for _ in TERMINAL_STATUSES)

        with self._cursor(write=True) as cur:
            cur.execute(self._sql(
                f"""
                SELECT job_id, uid, created_at FROM jobs
                WHERE status NOT IN ({placeholders}) AND updated_at < ?
                """
            ), (*sorted(TERMINAL_STATUSES), cutoff))
            candidates = cur.fetchall()

        reaped = 0
        for job_id, uid, created_at in candidates:
            with self._cursor(write=True) as cur:
                # The `status NOT IN (terminal)` repeat is the claim: if the
                # worker was merely slow and has since finished, this matches
                # nothing and we neither overwrite its result nor refund a job
                # that delivered one.
                cur.execute(self._sql(
                    f"""
                    UPDATE jobs
                    SET status = ?, progress = 1.0, error_code = ?, error_message = ?,
                        updated_at = ?
                    WHERE job_id = ? AND status NOT IN ({placeholders})
                    """
                ), (STATUS_FAILED, CODE_ANALYSIS_FAILED,
                    "That analysis stopped unexpectedly — try again.",
                    _now_iso(), job_id, *sorted(TERMINAL_STATUSES)))
                claimed = cur.rowcount
            if claimed:
                reaped += 1
                # The day the charge was made, not the day the lease expired. A
                # lease is fifteen minutes and a job can be queued for longer, so
                # this reaper is the path most likely to cross midnight.
                self.refund_use(uid, day_of(created_at))
        return reaped

    # -- chord maps (the cache) ----------------------------------------------

    def put_map(self, *, video_id: str, song: dict,
                sync: dict | None, engine_chords: str, engine_beats: str,
                analyzed_at: str, channel_id: str | None = None,
                title: str | None = None, duration_ms: int = 0,
                offset_ms: int | None = None, low_confidence: bool = False,
                owner_uid: str | None = None) -> None:
        """Store (or replace) one analysis.

        Upsert, not insert: re-analyzing a video **replaces** its map, exactly as
        `import` upserts the Library row on the same deterministic id (§12.5).
        An admin-set `offset_ms` is preserved across a re-analysis — it is a
        human correction and a fresh run has no better information than the
        person who watched the video.

        `owner_uid` makes the row private (see the schema note). It is preserved
        across a re-analysis for the same reason `offset_ms` is: a second upload
        of the same audio must not be able to hand someone else's recording to
        whoever uploads it next, and re-analysis must not silently publish it.
        """
        with self._cursor(write=True) as cur:
            cur.execute(self._sql(
                "SELECT offset_ms, owner_uid FROM chord_maps WHERE video_id = ?"
            ), (video_id,))
            row = cur.fetchone()
            preserved = row[0] if row and row[0] is not None else offset_ms
            owner = row[1] if row and row[1] is not None else owner_uid
            cur.execute(self._sql(
                """
                INSERT INTO chord_maps (video_id, channel_id, title, duration_ms,
                                        song_json, sync_json, offset_ms, low_confidence,
                                        engine_chords, engine_beats, analyzed_at, owner_uid,
                                        song_id, artist, tempo, tonic, mode, chord_names,
                                        denormalized)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(video_id) DO UPDATE SET
                    channel_id = EXCLUDED.channel_id,
                    title = EXCLUDED.title,
                    duration_ms = EXCLUDED.duration_ms,
                    song_json = EXCLUDED.song_json,
                    sync_json = EXCLUDED.sync_json,
                    offset_ms = EXCLUDED.offset_ms,
                    low_confidence = EXCLUDED.low_confidence,
                    engine_chords = EXCLUDED.engine_chords,
                    engine_beats = EXCLUDED.engine_beats,
                    analyzed_at = EXCLUDED.analyzed_at,
                    owner_uid = EXCLUDED.owner_uid,
                    song_id = EXCLUDED.song_id,
                    artist = EXCLUDED.artist,
                    tempo = EXCLUDED.tempo,
                    tonic = EXCLUDED.tonic,
                    mode = EXCLUDED.mode,
                    chord_names = EXCLUDED.chord_names,
                    denormalized = 1
                """
            ), (video_id, channel_id, title, duration_ms,
                json.dumps(song, ensure_ascii=False), json.dumps(sync, ensure_ascii=False) if sync else None,
                preserved, 1 if low_confidence else 0, engine_chords, engine_beats, analyzed_at,
                owner, *_catalog_scalars(song)))

    def get_map(self, video_id: str) -> ChordMap | None:
        with self._cursor() as cur:
            cur.execute(self._sql(
                """
                SELECT video_id, song_json, sync_json, offset_ms, low_confidence,
                       engine_chords, engine_beats, analyzed_at, channel_id, title, duration_ms,
                       owner_uid
                FROM chord_maps WHERE video_id = ?
                """
            ), (video_id,))
            row = cur.fetchone()
        if row is None:
            return None
        return ChordMap(
            video_id=row[0],
            song=json.loads(row[1]), sync=json.loads(row[2]) if row[2] else None,
            offset_ms=row[3], low_confidence=bool(row[4]),
            engine_chords=row[5], engine_beats=row[6], analyzed_at=row[7],
            channel_id=row[8], title=row[9], duration_ms=row[10] or 0,
            owner_uid=row[11],
        )

    # The catalog is public rows only: `owner_uid IS NULL`. Spelled once, because
    # the listing and the version token have to agree about what "the catalog" is
    # — a private upload that moved the version everyone polls would tell every
    # other player their home screen had changed, and then show them nothing.
    _PUBLIC = "m.owner_uid IS NULL"

    def list_catalog(self, *, limit: int = 60, offset: int = 0) -> list[CatalogEntry]:
        """The **catalog**: everything analyzed so far, newest first.

        This is what the app's Home screen is built from. Every row here is a
        cache hit — already analyzed, free to open, instant — which is exactly why
        the client leads with it and treats YouTube as the long tail.

        Three things it must get right, and all three are about not serving
        something we shouldn't, or not doing it a table at a time:

        - **Uploads are excluded.** Uploaded audio is stored in this same table,
          and it is somebody's private recording: their file name, their chart,
          their key and tempo. It has no `videoId` a YouTube player could resolve
          either, so a catalog row for one is both a privacy failure and a card
          that cannot be opened. `owner_uid IS NULL` is the whole filter, and it
          is in SQL for the same reason the blocklist is.
        - **Blocked videos and channels are excluded in SQL**, not filtered after.
          §3's takedown surface has to hold on a *listing* as firmly as it does on
          ``GET /v1/maps/{id}``; a blocked video that vanishes from the detail
          endpoint but still sits on the home screen is a takedown that didn't
          happen.
        - **The page is a page.** `LIMIT`/`OFFSET` in SQL, never a slice in
          Python: fetching and decoding the entire table to return sixty rows
          cost a second of CPU at two thousand maps and grew from there, with
          `offset` buying nothing at all. One row per video is now the primary
          key rather than a `ROW_NUMBER` collapse over the difficulty tiers, so
          this walks `chord_maps_catalog` in order and stops.

        The five scalars come from their own columns rather than from
        ``song_json``: this endpoint reads exactly those, and decoding a whole
        ``CompositionPayload`` per row to reach them is the cost that made paging
        pointless. See the schema note on why the duplication is safe.
        """
        where = [self._PUBLIC, """NOT EXISTS (
                      SELECT 1 FROM blocklist b
                      WHERE (b.kind = ? AND b.key = m.video_id)
                         OR (b.kind = ? AND b.key = m.channel_id)
                  )"""]
        params: list = [BLOCK_VIDEO, BLOCK_CHANNEL]

        sql = f"""
            SELECT m.video_id, m.title, m.artist, m.duration_ms, m.song_id,
                   m.chord_names, m.tempo, m.tonic, m.mode, m.analyzed_at,
                   m.low_confidence
            FROM chord_maps m
            WHERE {' AND '.join(where)}
            ORDER BY m.analyzed_at DESC, m.video_id DESC
            LIMIT ? OFFSET ?
        """
        params += [max(0, limit), max(0, offset)]

        with self._cursor() as cur:
            cur.execute(self._sql(sql), tuple(params))
            rows = cur.fetchall()
        return [
            CatalogEntry(
                video_id=row[0], title=row[1], artist=row[2],
                duration_ms=row[3] or 0, song_id=row[4],
                chord_names=_decode_chord_names(row[5]),
                tempo=row[6], tonic=row[7], mode=row[8],
                analyzed_at=row[9], low_confidence=bool(row[10]),
            )
            for row in rows
        ]

    def catalog_version(self) -> str:
        """A cheap token that changes whenever the catalog does.

        The client polls this to answer "has anyone added a song?" without
        pulling the whole list — the catalog is shared, so a song analyzed by one
        player should appear for everyone without anybody restarting anything.

        It is ``<count>:<newest analyzed_at>``, which moves on an addition (count)
        **and** on a re-analysis of an existing video (timestamp, since ``put_map``
        upserts and refreshes ``analyzed_at``). A deletion moves the count too.
        Deliberately not a hash of the whole table: this is the endpoint that gets
        called most often, so it has to stay one aggregate.

        Counts the same rows the listing serves — public ones. A private upload
        moving this token would wake every client to fetch a list that had not
        changed.
        """
        with self._cursor() as cur:
            cur.execute(self._sql(
                f"SELECT COUNT(*), COALESCE(MAX(analyzed_at), '') "
                f"FROM chord_maps m WHERE {self._PUBLIC}"
            ))
            row = cur.fetchone()
        count = row[0] if row else 0
        newest = row[1] if row and row[1] else ""
        return f"{count}:{newest}"

    def set_offset(self, video_id: str, offset_ms: int | None) -> int:
        """The §6 admin knob: shift this video's chart against
        the recording. Returns how many rows moved."""
        with self._cursor(write=True) as cur:
            cur.execute(self._sql("UPDATE chord_maps SET offset_ms = ? WHERE video_id = ?"),
                        (offset_ms, video_id))
            return cur.rowcount

    # -- jobs ----------------------------------------------------------------

    def create_job(self, *, job_id: str, uid: str, video_id: str) -> Job:
        self._maybe_prune_jobs()
        self._maybe_reap_stale_jobs()
        now = _now_iso()
        with self._cursor(write=True) as cur:
            cur.execute(self._sql(
                """
                INSERT INTO jobs (job_id, uid, video_id, status, progress,
                                  error_code, error_message, created_at, updated_at)
                VALUES (?, ?, ?, ?, 0, NULL, NULL, ?, ?)
                """
            ), (job_id, uid, video_id, STATUS_QUEUED, now, now))
        return Job(job_id, uid, video_id, STATUS_QUEUED, 0.0, None, None, now, now)

    def update_job(self, job_id: str, *, status: str | None = None,
                   progress: float | None = None, error_code: str | None = None,
                   error_message: str | None = None) -> None:
        sets, params = ["updated_at = ?"], [_now_iso()]
        if status is not None:
            sets.append("status = ?")
            params.append(status)
        if progress is not None:
            sets.append("progress = ?")
            params.append(progress)
        if error_code is not None:
            sets.append("error_code = ?")
            params.append(error_code)
        if error_message is not None:
            sets.append("error_message = ?")
            params.append(error_message)
        params.append(job_id)
        with self._cursor(write=True) as cur:
            cur.execute(self._sql(f"UPDATE jobs SET {', '.join(sets)} WHERE job_id = ?"), tuple(params))

    def get_job(self, job_id: str) -> Job | None:
        # The poll path reaps too, so a client watching a job whose worker was
        # killed reaches a terminal answer even when nobody else is starting
        # analyses. Interval-guarded, so a 2-second poll is not 2-second sweeps.
        self._maybe_reap_stale_jobs()
        with self._cursor() as cur:
            cur.execute(self._sql(
                """
                SELECT job_id, uid, video_id, status, progress,
                       error_code, error_message, created_at, updated_at
                FROM jobs WHERE job_id = ?
                """
            ), (job_id,))
            row = cur.fetchone()
        return Job(*row) if row else None

    def follow_job(self, job_id: str, uid: str) -> None:
        """Record that `uid` is waiting on a job somebody else started.

        Called when `active_job_for` joins a second caller onto an in-flight
        analysis. Idempotent — the client may well ask twice.
        """
        with self._cursor(write=True) as cur:
            cur.execute(self._sql(
                """
                INSERT INTO job_followers (job_id, uid, created_at) VALUES (?, ?, ?)
                ON CONFLICT(job_id, uid) DO NOTHING
                """
            ), (job_id, uid, _now_iso()))

    def may_read_job(self, job: Job, uid: str) -> bool:
        """Whether `uid` is entitled to this job's status.

        The owner, or anyone the API handed this job id to because they asked for
        the same video while it was running. Deliberately not "anyone who knows
        the id": the id is a uuid4 and unguessable in practice, but *in practice*
        is not an authorization rule.
        """
        if job.uid == uid:
            return True
        with self._cursor() as cur:
            cur.execute(
                self._sql("SELECT 1 FROM job_followers WHERE job_id = ? AND uid = ?"),
                (job.job_id, uid),
            )
            return cur.fetchone() is not None

    def active_job_for(self, video_id: str) -> Job | None:
        """An in-flight job for this video, if one exists.

        Two players asking for the same video at the same time should join one
        job, not start two: the second would decode the same recording again for
        an identical result, which is both the expensive thing and the thing §2
        wants to happen as rarely as possible.

        **Only jobs within their lease count.** This is the read that turns one
        killed worker into a permanently un-analyzable video: without the
        staleness bound, a row abandoned mid-flight is "in flight" forever and
        every later request joins a job that will never finish.
        """
        self._maybe_reap_stale_jobs()
        placeholders = ", ".join("?" for _ in TERMINAL_STATUSES)
        # Belt and braces with the reaper: that runs on an interval and this must
        # be right on every call, including the one that arrives between sweeps.
        fresh_since = datetime.fromtimestamp(
            time.time() - self._JOB_LEASE_S, tz=timezone.utc
        ).isoformat().replace("+00:00", "Z")
        with self._cursor() as cur:
            cur.execute(self._sql(
                f"""
                SELECT job_id, uid, video_id, status, progress,
                       error_code, error_message, created_at, updated_at
                FROM jobs
                WHERE video_id = ? AND status NOT IN ({placeholders})
                  AND updated_at >= ?
                ORDER BY created_at DESC
                """
            ), (video_id, *sorted(TERMINAL_STATUSES), fresh_since))
            row = cur.fetchone()
        return Job(*row) if row else None

    # -- blocklist + audit (§3) ----------------------------------------------

    def block(self, kind: str, key: str, *, reason: str | None, actor: str | None) -> None:
        with self._cursor(write=True) as cur:
            cur.execute(self._sql(
                """
                INSERT INTO blocklist (kind, key, reason, actor, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(kind, key) DO UPDATE SET
                    reason = EXCLUDED.reason, actor = EXCLUDED.actor
                """
            ), (kind, key, reason, actor, _now_iso()))
        self.audit(AUDIT_BLOCK, key, kind=kind, actor=actor, reason=reason)

    def unblock(self, kind: str, key: str, *, actor: str | None = None) -> bool:
        with self._cursor(write=True) as cur:
            cur.execute(self._sql("DELETE FROM blocklist WHERE kind = ? AND key = ?"), (kind, key))
            removed = cur.rowcount > 0
        self.audit(AUDIT_UNBLOCK, key, kind=kind, actor=actor)
        return removed

    def is_blocked(self, *, video_id: str | None = None, channel_id: str | None = None) -> bool:
        """Whether either identifier is on the list.

        Checked before a fetch **and** before serving a cache hit: a video
        blocked after it was analyzed must stop being served immediately, not
        once its map happens to be purged.
        """
        pairs = []
        if video_id:
            pairs.append((BLOCK_VIDEO, video_id))
        if channel_id:
            pairs.append((BLOCK_CHANNEL, channel_id))
        if not pairs:
            return False
        with self._cursor() as cur:
            for kind, key in pairs:
                cur.execute(self._sql("SELECT 1 FROM blocklist WHERE kind = ? AND key = ?"), (kind, key))
                if cur.fetchone():
                    return True
        return False

    def blocked_entries(self) -> list[tuple[str, str, str | None, str]]:
        with self._cursor() as cur:
            cur.execute("SELECT kind, key, reason, created_at FROM blocklist ORDER BY created_at DESC")
            return [tuple(row) for row in cur.fetchall()]

    def purge(self, video_id: str, *, actor: str | None = None, reason: str | None = None) -> dict[str, int]:
        """Delete a video's map and everything that references it (§3).

        Returns the row counts so the caller can *verify the cascade actually
        cascaded* rather than trusting that it did — the handoff asks for exactly
        that, and a purge that silently matched nothing is the failure you find
        out about from a lawyer.

        Deliberately does NOT touch `audit_log`: the record of the takedown has
        to outlive the thing taken down.
        """
        with self._cursor(write=True) as cur:
            cur.execute(self._sql("DELETE FROM chord_maps WHERE video_id = ?"), (video_id,))
            maps = cur.rowcount
            cur.execute(self._sql(
                "DELETE FROM job_followers WHERE job_id IN "
                "(SELECT job_id FROM jobs WHERE video_id = ?)"
            ), (video_id,))
            cur.execute(self._sql("DELETE FROM jobs WHERE video_id = ?"), (video_id,))
            jobs = cur.rowcount
        self.audit(AUDIT_PURGE, video_id, actor=actor, reason=reason,
                   detail=json.dumps({"maps": maps, "jobs": jobs}))
        return {"maps": maps, "jobs": jobs}

    def purge_channel(self, channel_id: str, *, actor: str | None = None,
                      reason: str | None = None) -> dict[str, int]:
        """Purge every cached video known to belong to a channel.

        Only reaches maps whose `channel_id` we recorded — a video analyzed
        before its channel was known is not found here, which is why the block
        (checked on every serve) is the real guarantee and the purge is hygiene.
        """
        with self._cursor() as cur:
            cur.execute(self._sql("SELECT DISTINCT video_id FROM chord_maps WHERE channel_id = ?"),
                        (channel_id,))
            video_ids = [row[0] for row in cur.fetchall()]
        totals = {"maps": 0, "jobs": 0, "videos": len(video_ids)}
        for video_id in video_ids:
            counts = self.purge(video_id, actor=actor, reason=reason)
            totals["maps"] += counts["maps"]
            totals["jobs"] += counts["jobs"]
        return totals

    def audit(self, action: str, key: str, *, kind: str | None = None,
              actor: str | None = None, reason: str | None = None,
              detail: str | None = None) -> None:
        """Append one row. There is no update and no delete for this table."""
        with self._cursor(write=True) as cur:
            cur.execute(self._sql(
                """
                INSERT INTO audit_log (action, kind, key, actor, reason, detail, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """
            ), (action, kind, key, actor, reason, detail, _now_iso()))

    def audit_entries(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute(self._sql(
                """
                SELECT action, kind, key, actor, reason, detail, created_at
                FROM audit_log ORDER BY id DESC LIMIT ?
                """
            ), (limit,))
            rows = cur.fetchall()
        return [
            {"action": r[0], "kind": r[1], "key": r[2], "actor": r[3],
             "reason": r[4], "detail": r[5], "createdAt": r[6]}
            for r in rows
        ]


class SQLiteStore(Store):
    """The zero-config default: local dev, the test-suite, and any
    single-container deployment."""

    _ID_COLUMN = "id INTEGER PRIMARY KEY AUTOINCREMENT"
    _PARAMSTYLE = "qmark"

    def __init__(self, db_path: str):
        # check_same_thread=False + a lock: FastAPI runs the sync store calls in
        # a threadpool, and the writes are trivially short.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.Lock()
        self._migrate()

    def _columns(self, cur, table: str) -> set[str]:
        cur.execute(f"PRAGMA table_info({table})")
        return {row[1] for row in cur.fetchall()}

    @contextmanager
    def _cursor(self, *, write: bool = False):
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
            except Exception:
                if write:
                    self._conn.rollback()
                raise
            else:
                if write:
                    self._conn.commit()
            finally:
                cur.close()

    def close(self) -> None:
        self._conn.close()


class PostgresStore(Store):
    """The deployable store.

    A connection **pool** rather than one shared connection: Modal runs the sync
    store calls across a threadpool, so concurrent callers each need their own.
    """

    _ID_COLUMN = "id BIGSERIAL PRIMARY KEY"
    _PARAMSTYLE = "pyformat"

    def __init__(self, dsn: str, *, min_size: int = 1, max_size: int = 10):
        # Imported lazily so the SQLite path — local dev, CI, the whole
        # test-suite — needs no psycopg installed at all.
        from psycopg_pool import ConnectionPool

        self._pool = ConnectionPool(dsn, min_size=min_size, max_size=max_size, open=True)
        self._pool.wait(timeout=30)
        self._migrate()

    @contextmanager
    def _cursor(self, *, write: bool = False):
        # psycopg's connection context manager commits on clean exit and rolls
        # back on exception; `write` is accepted for interface symmetry.
        with self._pool.connection() as conn, conn.cursor() as cur:
            yield cur

    def _columns(self, cur, table: str) -> set[str]:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = %s AND table_schema = current_schema()",
            (table,),
        )
        return {row[0] for row in cur.fetchall()}

    def _serialize_rate_key(self, cur, scope: str, key: str) -> None:
        """A transaction-scoped advisory lock on this one `(scope, key)`.

        The cheapest correct answer to the limiter's count-then-insert race. Not
        `SELECT … FOR UPDATE`: there is no row to lock — the check's whole point
        is deciding whether to *create* one — and locking the table would
        serialize every caller against every other. `pg_advisory_xact_lock`
        releases at commit, which is the end of the `_cursor` block, so nothing
        here can leak a lock.

        Contended only by requests from the same uid or the same IP, which is
        precisely the case where serializing is the intended behaviour.
        """
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"{scope}:{key}",))

    def close(self) -> None:
        self._pool.close()


# Which half of the deployment is asking for a store. The worker is not merely a
# second process — it is a second *container*, with its own ephemeral filesystem
# and (deliberately, §4) no Volume mounted.
ROLE_API = "api"
ROLE_WORKER = "worker"


class StoreUnusable(RuntimeError):
    """This container cannot reach a store the rest of the deployment shares."""


def build_store(settings, *, role: str = ROLE_API) -> Store:
    """Pick the backend from configuration: DSN set ⇒ Postgres, absent ⇒ SQLite.

    **A remote worker may not use SQLite, and this refuses rather than pretends.**
    On Modal the worker runs `build_store` in its own container. `db_path`
    defaults to a *relative* path and the `chords-data` Volume is mounted on the
    API function only, so a SQLite worker opens a brand-new database file on a
    disk that dies with the call: it writes `analyzing`, then `ready`, then the
    map, into a file nothing will ever read. From the API's side every job sits at
    `queued` until the 900 s lease reaper fails it — a total outage of the
    analysis feature, behind a `/healthz` that is green in both containers,
    because each one is individually fine.

    Nothing enforced this before. The `MAX_CONTAINERS = 1` pin and its deploy-time
    warning are about the *API* side of the same single-writer question, and are
    silent here: the worker is a separate container whether or not the API is
    pinned, so the pin cannot make this shape work. Postgres is not an
    optimisation for this deployment, it is the requirement, and the honest place
    to say so is where the store is built.

    Raising leaves the job row untouched at `queued`, which is the right failure:
    the reaper on the API container gives the player a terminal answer and a
    refund, and the exception is in the worker's log with the remedy in it.
    """
    dsn = getattr(settings, "database_url", None)
    if dsn:
        log.info("store: postgres (%s)", role)
        return PostgresStore(dsn)
    if role == ROLE_WORKER:
        raise StoreUnusable(
            "This worker has no CHORDS_DATABASE_URL, so the only store it could "
            "build is a SQLite file on its own ephemeral disk — which the API "
            "container cannot read, so every job it ran would stay 'queued' until "
            "the lease reaper failed it. Set CHORDS_DATABASE_URL on "
            "chords-worker-secrets (the same Postgres DSN the API uses)."
        )
    log.info("store: sqlite at %s (%s)", settings.db_path, role)
    return SQLiteStore(settings.db_path)
