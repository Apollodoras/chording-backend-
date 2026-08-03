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

log = logging.getLogger("chords.store")

RATE_SCOPE_UID = "uid"
RATE_SCOPE_IP = "ip"

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
    # reconstructed. `song_json` is a CompositionPayload v2; `sync_json` is the
    # §13 sidecar (nullable — §13.3 returns the song without it rather than
    # faking a grid). `engine_chords`/`engine_beats` are stored per §5.3 so a
    # cache can be invalidated selectively when one engine is upgraded.
    """
    CREATE TABLE IF NOT EXISTS chord_maps (
        video_id      TEXT NOT NULL,
        difficulty    TEXT NOT NULL,
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
        PRIMARY KEY (video_id, difficulty)
    )
    """,
    "CREATE INDEX IF NOT EXISTS chord_maps_channel ON chord_maps (channel_id)",
    # Job lifecycle (§16.1). `result_difficulty` points at the chord_maps row a
    # finished job produced, so a purge can find and clear the jobs that would
    # otherwise keep handing out a map that no longer exists.
    """
    CREATE TABLE IF NOT EXISTS jobs (
        job_id            TEXT PRIMARY KEY,
        uid               TEXT NOT NULL,
        video_id          TEXT NOT NULL,
        difficulty        TEXT NOT NULL,
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


@dataclass(frozen=True)
class ChordMap:
    """A cached analysis, as it comes back out of the store."""

    video_id: str
    difficulty: str
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


@dataclass(frozen=True)
class Job:
    job_id: str
    uid: str
    video_id: str
    difficulty: str
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

    def _migrate(self) -> None:
        with self._cursor(write=True) as cur:
            for statement in _SCHEMA:
                cur.execute(statement.format(id_column=self._ID_COLUMN))

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

    def refund_use(self, uid: str) -> None:
        """Give back a charge whose analysis never happened.

        Analysis can fail for reasons that are not the player's doing and cost us
        nothing worth billing — a video that turns out to be blocked, private, or
        too long is rejected before a single second is decoded. Charging for that
        would burn a daily quota on an error message.
        """
        with self._cursor(write=True) as cur:
            cur.execute(self._sql(
                "UPDATE usage SET count = count - 1 WHERE uid = ? AND day = ? AND count > 0"
            ), (uid, utc_day()))

    # -- rate limiting -------------------------------------------------------

    def hit_rate_limit(self, scope: str, key: str, limit: int, window_s: float,
                       now: float | None = None) -> tuple[bool, float]:
        """Record one request against a **sliding window** and say whether it is
        allowed. Returns `(allowed, retry_after_s)`.

        Sliding rather than a fixed bucket because a fixed bucket lets a caller
        fire `limit` requests at 11:59:59 and `limit` more at 12:00:00.
        """
        if limit <= 0:
            return True, 0.0
        now = time.time() if now is None else now
        self._maybe_prune_rate_events(window_s, now)
        cutoff = now - window_s
        with self._cursor(write=True) as cur:
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
        with self._cursor(write=True) as cur:
            cur.execute(
                self._sql(
                    f"DELETE FROM jobs WHERE status IN ({placeholders}) AND updated_at < ?"
                ),
                (*sorted(TERMINAL_STATUSES), cutoff.isoformat().replace("+00:00", "Z")),
            )
            return cur.rowcount

    # -- chord maps (the cache) ----------------------------------------------

    def put_map(self, *, video_id: str, difficulty: str, song: dict,
                sync: dict | None, engine_chords: str, engine_beats: str,
                analyzed_at: str, channel_id: str | None = None,
                title: str | None = None, duration_ms: int = 0,
                offset_ms: int | None = None, low_confidence: bool = False) -> None:
        """Store (or replace) one analysis.

        Upsert, not insert: re-analyzing a video **replaces** its map, exactly as
        `import` upserts the Library row on the same deterministic id (§12.5).
        An admin-set `offset_ms` is preserved across a re-analysis — it is a
        human correction and a fresh run has no better information than the
        person who watched the video.
        """
        with self._cursor(write=True) as cur:
            cur.execute(self._sql("SELECT offset_ms FROM chord_maps WHERE video_id = ? AND difficulty = ?"),
                        (video_id, difficulty))
            row = cur.fetchone()
            preserved = row[0] if row and row[0] is not None else offset_ms
            cur.execute(self._sql(
                """
                INSERT INTO chord_maps (video_id, difficulty, channel_id, title, duration_ms,
                                        song_json, sync_json, offset_ms, low_confidence,
                                        engine_chords, engine_beats, analyzed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(video_id, difficulty) DO UPDATE SET
                    channel_id = EXCLUDED.channel_id,
                    title = EXCLUDED.title,
                    duration_ms = EXCLUDED.duration_ms,
                    song_json = EXCLUDED.song_json,
                    sync_json = EXCLUDED.sync_json,
                    offset_ms = EXCLUDED.offset_ms,
                    low_confidence = EXCLUDED.low_confidence,
                    engine_chords = EXCLUDED.engine_chords,
                    engine_beats = EXCLUDED.engine_beats,
                    analyzed_at = EXCLUDED.analyzed_at
                """
            ), (video_id, difficulty, channel_id, title, duration_ms,
                json.dumps(song, ensure_ascii=False), json.dumps(sync, ensure_ascii=False) if sync else None,
                preserved, 1 if low_confidence else 0, engine_chords, engine_beats, analyzed_at))

    def get_map(self, video_id: str, difficulty: str) -> ChordMap | None:
        with self._cursor() as cur:
            cur.execute(self._sql(
                """
                SELECT video_id, difficulty, song_json, sync_json, offset_ms, low_confidence,
                       engine_chords, engine_beats, analyzed_at, channel_id, title, duration_ms
                FROM chord_maps WHERE video_id = ? AND difficulty = ?
                """
            ), (video_id, difficulty))
            row = cur.fetchone()
        if row is None:
            return None
        return ChordMap(
            video_id=row[0], difficulty=row[1],
            song=json.loads(row[2]), sync=json.loads(row[3]) if row[3] else None,
            offset_ms=row[4], low_confidence=bool(row[5]),
            engine_chords=row[6], engine_beats=row[7], analyzed_at=row[8],
            channel_id=row[9], title=row[10], duration_ms=row[11] or 0,
        )

    def set_offset(self, video_id: str, offset_ms: int | None) -> int:
        """The §6 admin knob: shift every difficulty of this video's chart against
        the recording. Returns how many rows moved."""
        with self._cursor(write=True) as cur:
            cur.execute(self._sql("UPDATE chord_maps SET offset_ms = ? WHERE video_id = ?"),
                        (offset_ms, video_id))
            return cur.rowcount

    # -- jobs ----------------------------------------------------------------

    def create_job(self, *, job_id: str, uid: str, video_id: str, difficulty: str) -> Job:
        self._maybe_prune_jobs()
        now = _now_iso()
        with self._cursor(write=True) as cur:
            cur.execute(self._sql(
                """
                INSERT INTO jobs (job_id, uid, video_id, difficulty, status, progress,
                                  error_code, error_message, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 0, NULL, NULL, ?, ?)
                """
            ), (job_id, uid, video_id, difficulty, STATUS_QUEUED, now, now))
        return Job(job_id, uid, video_id, difficulty, STATUS_QUEUED, 0.0, None, None, now, now)

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
        with self._cursor() as cur:
            cur.execute(self._sql(
                """
                SELECT job_id, uid, video_id, difficulty, status, progress,
                       error_code, error_message, created_at, updated_at
                FROM jobs WHERE job_id = ?
                """
            ), (job_id,))
            row = cur.fetchone()
        return Job(*row) if row else None

    def active_job_for(self, video_id: str, difficulty: str) -> Job | None:
        """An in-flight job for this exact analysis, if one exists.

        Two players asking for the same video at the same time should join one
        job, not start two: the second would decode the same recording again for
        an identical result, which is both the expensive thing and the thing §2
        wants to happen as rarely as possible.
        """
        placeholders = ", ".join("?" for _ in TERMINAL_STATUSES)
        with self._cursor() as cur:
            cur.execute(self._sql(
                f"""
                SELECT job_id, uid, video_id, difficulty, status, progress,
                       error_code, error_message, created_at, updated_at
                FROM jobs
                WHERE video_id = ? AND difficulty = ? AND status NOT IN ({placeholders})
                ORDER BY created_at DESC
                """
            ), (video_id, difficulty, *sorted(TERMINAL_STATUSES)))
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

    def close(self) -> None:
        self._pool.close()


def build_store(settings) -> Store:
    """Pick the backend from configuration: DSN set ⇒ Postgres, absent ⇒ SQLite."""
    dsn = getattr(settings, "database_url", None)
    if dsn:
        log.info("store: postgres")
        return PostgresStore(dsn)
    log.info("store: sqlite at %s", settings.db_path)
    return SQLiteStore(settings.db_path)
