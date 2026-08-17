"""The upload path — `POST /v1/analyze/upload` and `FileSource`.

The second input source, and the one that carries **no YouTube-terms exposure**
(`app/analysis/file_source.py` says why that is the point rather than a
convenience). So the properties worth pinning are the ones that make it a
genuine alternative rather than a variant spelling of the fetch path:

- it is reachable when the fetch path is not,
- it obeys every §2/§3 rule the fetch path obeys — the gate, the cap, the
  blocklist, the quota, the scratch guarantee,
- and its ids are content-addressed, so re-sending a file is a cache hit and
  therefore free (§16.4).

The decode half needs ffmpeg and is skipped without it, exactly like the other
adapter tests — the CI job that guards §4 installs no audio stack at all.
"""

from __future__ import annotations

import shutil
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.analysis import engines, file_source
from app.analysis.file_source import (
    MAX_UPLOAD_BYTES,
    FileSource,
    available,
    is_upload_id,
    upload_id,
)
from app.errors import VideoTooLong
from app.jobs import JobRunner
from app.main import create_app
from app.store import BLOCK_VIDEO
from tests.conftest import FakeBeatTracker, FakeChordEngine, FakeOnsetDetector, FakeSource

AUTH = {"Authorization": "Bearer dev-token"}
ADMIN = {"X-Admin-Token": "admin-secret", "X-Admin-Actor": "agent@example.com"}

AUDIO = b"ID3\x04\x00" + b"\x00" * 4096


@pytest.fixture(autouse=True)
def registered_engines():
    engines.register_chord_engine("fake-chords", FakeChordEngine)
    engines.register_beat_tracker("fake-beats", FakeBeatTracker)
    engines.register_onset_detector("fake-beats", FakeOnsetDetector)
    yield
    engines._CHORD_ENGINES.clear()
    engines._BEAT_TRACKERS.clear()
    engines._ONSET_DETECTORS.clear()


class UploadCapableRunner(JobRunner):
    """The inline runner, reporting itself able to take uploads.

    `JobRunner.can_accept_uploads` asks whether *this image* has ffmpeg, which is
    the right question in production and the wrong one in a suite that must run
    with no audio stack (see the CI job). Overridden rather than mocked so the
    route under test is the real one.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.submitted: list[dict] = []

    def can_accept_uploads(self) -> bool:
        return True

    def submit(self, *, job_id, video_id, difficulty, uid, audio=None, filename=None):
        self.submitted.append({"video_id": video_id, "audio": audio, "filename": filename})
        # Deliberately runs against the fake source rather than building a real
        # `FileSource`: this covers the route, the gate and the job bookkeeping.
        # `FileSource` itself is covered below, against real ffmpeg.
        from app.jobs import run_job
        run_job(job_id=job_id, video_id=video_id, difficulty=difficulty, uid=uid,
                settings=self.settings, store=self.store, source=self.source)


@pytest.fixture
def client(settings, store):
    source = FakeSource()
    runner = UploadCapableRunner(settings, store, source)
    app = create_app(settings, store=store, source=source, runner=runner)
    with TestClient(app) as client:
        client.runner = runner
        yield client


def post(client, data: bytes = AUDIO, name: str = "song.mp3", **form):
    return client.post("/v1/analyze/upload", headers=AUTH,
                       files={"file": (name, data, "audio/mpeg")}, data=form)


# --- ids --------------------------------------------------------------------

def test_the_id_is_the_content_so_the_same_file_is_the_same_song():
    """§12.5's idempotency key, derived from the audio rather than from the
    filename or the clock: re-uploading must **replace** the Library row, and a
    second send of the same bytes must cost nothing."""
    assert upload_id(AUDIO) == upload_id(AUDIO)
    assert upload_id(AUDIO) != upload_id(AUDIO + b"\x01")
    assert is_upload_id(upload_id(AUDIO))


def test_an_upload_id_is_never_mistaken_for_a_youtube_id():
    """They share an id column, a blocklist and a payload namespace, so they must
    not be able to collide: a YouTube id is exactly 11 of [A-Za-z0-9_-]."""
    from app.analysis.fetch import parse_video_id

    assert parse_video_id(upload_id(AUDIO)) is None
    assert not is_upload_id("dQw4w9WgXcQ")


# --- the route --------------------------------------------------------------

def test_an_upload_is_analyzed_and_polls_ready(client):
    response = post(client)
    assert response.status_code == 202
    job_id = response.json()["jobId"]

    status = client.get(f"/v1/analyze/{job_id}", headers=AUTH).json()
    assert status["status"] == "ready"
    assert status["song"]["id"].startswith("yt:up_")
    assert client.runner.submitted[0]["audio"] == AUDIO
    assert client.runner.submitted[0]["filename"] == "song.mp3"


def test_resending_the_same_audio_is_a_free_cache_hit(client, store, settings):
    """§16.4 and §2.6: a cached map costs us nothing, so it must not cost the
    player a daily analysis either."""
    assert post(client).status_code == 202
    used = store.usage_today("dev-user")

    again = post(client)
    assert again.status_code == 200
    assert again.json()["song"]["id"].startswith("yt:up_")
    assert store.usage_today("dev-user") == used


def test_the_kill_switch_stops_uploads_too(settings, store):
    """§3's switch is one flag for the whole feature. An input path it did not
    cover would be a hole in the only lever there is."""
    from dataclasses import replace

    source = FakeSource()
    app = create_app(replace(settings, analysis_enabled=False), store=store, source=source,
                     runner=UploadCapableRunner(settings, store, source))
    with TestClient(app) as client:
        response = post(client)
    assert response.status_code == 503
    assert response.json()["code"] == "feature_disabled"


def test_blocked_audio_is_refused(client, store):
    """§3's blocklist keyed on the content hash — the takedown lever an upload
    path needs, since there is no channel to block."""
    store.block(BLOCK_VIDEO, upload_id(AUDIO), reason="DMCA", actor="admin")
    response = post(client)
    assert response.status_code == 403
    assert response.json()["code"] == "video_blocked"


def test_an_empty_upload_is_a_clean_400(client):
    response = post(client, data=b"")
    assert response.status_code == 400
    assert "message" in response.json()


def test_an_oversized_upload_is_refused_without_being_read_whole(client):
    """The cap is enforced while reading, not after: an unbounded `await
    file.read()` is how a container with a memory cap dies instead of answering."""
    response = post(client, data=b"\x00" * (MAX_UPLOAD_BYTES + 1024))
    assert response.status_code == 413


def test_upload_requires_authentication(client):
    response = client.post("/v1/analyze/upload",
                           files={"file": ("song.mp3", AUDIO, "audio/mpeg")})
    assert response.status_code == 401


def test_healthz_reports_upload_capability_separately(client):
    """`canAnalyze` and `canAcceptUploads` genuinely differ — an upload needs
    ffmpeg and the engines but no fetch source, which is what lets it survive a
    dead YouTube path."""
    body = client.get("/healthz").json()
    assert body["canAcceptUploads"] is True
    assert "canAnalyze" in body


# --- the source itself ------------------------------------------------------

def test_a_filename_never_reaches_the_title_unsanitised():
    """Player-supplied text that lands in a database column and on screen."""
    source = FileSource(AUDIO, filename="../../etc/passwd.mp3")
    assert "/" not in source.title
    assert source.title == "passwd"
    assert FileSource(AUDIO, filename=None).title == "Uploaded audio"
    assert FileSource(AUDIO, filename="a\x00b\nc.wav").title == "abc"


def test_the_source_id_matches_the_content_hash():
    assert FileSource(AUDIO).video_id == upload_id(AUDIO)


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
                    reason="needs ffmpeg + ffprobe")
def test_probe_and_decode_round_trip(tmp_path, settings):
    """The real thing, when the image can do it: a WAV in, samples out, and
    nothing left behind in the scratch directory (§2.1)."""
    import struct
    import wave

    wav_path = tmp_path / "tone.wav"
    with wave.open(str(wav_path), "w") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(22050)
        out.writeframes(struct.pack("<22050h", *([1000] * 22050)))
    data = wav_path.read_bytes()

    source = FileSource(data, filename="tone.wav", settings=settings)
    meta = source.probe(source.video_id)
    assert meta.video_id == upload_id(data)
    assert 0.9 < meta.duration_s < 1.1
    assert meta.channel_id is None      # an upload has no channel to block

    workdir = tmp_path / "scratch"
    workdir.mkdir()
    pcm, rate = source.decode(source.video_id, workdir)
    assert rate == 22050
    assert len(pcm) > 0
    assert list(workdir.iterdir()) == [], "decode must leave nothing behind"


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
                    reason="needs ffmpeg + ffprobe")
def test_garbage_is_a_calm_player_facing_error_not_a_crash(settings):
    from app.errors import VideoUnavailable

    source = FileSource(b"not audio at all" * 100, settings=settings)
    with pytest.raises(VideoUnavailable):
        source.probe(source.video_id)


def test_available_reports_this_image_honestly():
    """Same contract as `ytdlp_source.available()`: report what this container
    can do, so the API image refuses an upload it could not process rather than
    accepting it and failing later."""
    assert available() == bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


# --- the decode is bounded before it reaches memory --------------------------

def _tone(path, seconds: float) -> bytes:
    """`seconds` of a steady tone, as WAV bytes."""
    import struct
    import wave

    frames = int(22050 * seconds)
    with wave.open(str(path), "w") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(22050)
        out.writeframes(struct.pack(f"<{frames}h", *([1000] * frames)))
    return path.read_bytes()


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
                    reason="needs ffmpeg + ffprobe")
def test_a_decode_is_bounded_rather_than_measured_afterwards(tmp_path, settings, monkeypatch):
    """`gate()` refuses a long file on the duration **ffprobe claimed**, and a
    claim is not a measurement. A container that understates its real length
    used to decode unbounded into the scratch dir — tmpfs, so RAM on the worker
    — and then be read into memory a second time by `_read_wav`. With a 4 GB cap
    that is a dead container instead of a clean refusal.

    The assertion that matters is the second one: the ceiling has to sit *above*
    the length tolerance. Clamping at the same number would truncate an
    over-length file to exactly the limit, and the measured check would then
    never fire — a refusal silently turned into a shortened song.
    """
    commands = []
    real_run = file_source._run

    def spy(command, *, timeout):
        commands.append(command)
        return real_run(command, timeout=timeout)

    monkeypatch.setattr(file_source, "_run", spy)

    capped = replace(settings, max_video_seconds=1)
    data = _tone(tmp_path / "long.wav", seconds=5)
    source = FileSource(data, filename="long.wav", settings=capped)
    workdir = tmp_path / "scratch"
    workdir.mkdir()

    with pytest.raises(VideoTooLong):
        source.decode(source.video_id, workdir)

    decode = commands[-1]
    assert "-t" in decode, "the decode itself has to be bounded, not just checked"
    ceiling = float(decode[decode.index("-t") + 1])
    assert ceiling > capped.max_video_seconds * file_source._LENGTH_TOLERANCE, \
        "a ceiling at the tolerance would truncate an over-long file into legality"
    assert list(workdir.iterdir()) == [], "and it still leaves nothing behind (§2.1)"


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
                    reason="needs ffmpeg + ffprobe")
def test_the_bound_never_shortens_audio_that_is_within_the_cap(tmp_path, settings):
    """The other half: a legal file has to come back whole."""
    data = _tone(tmp_path / "ok.wav", seconds=2)
    source = FileSource(data, filename="ok.wav",
                        settings=replace(settings, max_video_seconds=600))
    workdir = tmp_path / "scratch"
    workdir.mkdir()

    pcm, rate = source.decode(source.video_id, workdir)

    assert 1.9 < len(pcm) / rate < 2.1, "a song well inside the cap is untouched"


# --- an upload is private ----------------------------------------------------
#
# Uploaded audio lands in the same `chord_maps` table as a fetched video, and until
# `owner_uid` existed nothing distinguished the two. `list_catalog` selected every
# row, so one player's private recording — their own filename, their chart, their
# key and tempo — appeared on every other player's home screen; and `GET
# /v1/maps/{id}` served it to anyone who produced the right hash. `is_upload_id`
# had been written for exactly this and was never called anywhere.
#
# The catalog row was functionally broken too: `embeddable: true` beside a
# `videoId` of `up_<hash>`, which no YouTube player can resolve.

class _TwoPlayers:
    """`Bearer <name>` → uid `<name>`, so "somebody else" is expressible."""

    mode = "test-multi-user"

    def __call__(self, authorization, *, check_revoked: bool = False):
        from fastapi import HTTPException

        from app.auth import Principal

        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail={"message": "Sign in."})
        name = authorization[7:].strip()
        return Principal(uid=name, display_name=name, email_verified=True)


@pytest.fixture
def two_player_client(settings, store):
    source = FakeSource()
    runner = UploadCapableRunner(settings, store, source)
    app = create_app(settings, store=store, source=source,
                     authenticator=_TwoPlayers(), runner=runner)
    with TestClient(app) as client:
        client.runner = runner
        yield client


ALICE = {"Authorization": "Bearer alice"}
BOB = {"Authorization": "Bearer bob"}


def test_an_upload_never_appears_in_the_public_catalog(two_player_client):
    """The reproduction. Alice uploads a rehearsal recording; Bob opens the app."""
    response = two_player_client.post(
        "/v1/analyze/upload", headers=ALICE,
        files={"file": ("my band rehearsal 2026.mp3", AUDIO, "audio/mpeg")})
    assert response.status_code == 202

    catalog = two_player_client.get("/v1/catalog", headers=BOB).json()
    assert catalog["results"] == [], (
        f"an upload reached the shared catalog: {catalog['results']}"
    )


def test_an_upload_does_not_move_the_version_everyone_polls(two_player_client):
    """The client polls `/v1/catalog/version` to learn "has anyone added a song?".
    A private upload moving it wakes every client to fetch a list that has not
    changed."""
    before = two_player_client.get("/v1/catalog/version", headers=BOB).json()["version"]
    two_player_client.post("/v1/analyze/upload", headers=ALICE,
                           files={"file": ("demo.mp3", AUDIO, "audio/mpeg")})
    after = two_player_client.get("/v1/catalog/version", headers=BOB).json()["version"]

    assert before == after


def test_a_stranger_cannot_read_an_upload_by_its_hash(two_player_client):
    """`up_<sha256[:16]>` is unguessable in practice, and *in practice* is not an
    authorization rule. 404 rather than 403: whether someone else's upload exists is
    not this caller's business either."""
    two_player_client.post("/v1/analyze/upload", headers=ALICE,
                           files={"file": ("demo.mp3", AUDIO, "audio/mpeg")})
    video_id = upload_id(AUDIO)

    assert two_player_client.get(f"/v1/maps/{video_id}", headers=BOB).status_code == 404


def test_the_uploader_can_still_read_their_own(two_player_client):
    """The privacy fix must not have locked the owner out of their own analysis."""
    two_player_client.post("/v1/analyze/upload", headers=ALICE,
                           files={"file": ("demo.mp3", AUDIO, "audio/mpeg")})

    response = two_player_client.get(f"/v1/maps/{upload_id(AUDIO)}", headers=ALICE)
    assert response.status_code == 200
    assert response.json()["song"]["id"].startswith("yt:up_")


def test_re_sending_the_same_bytes_is_still_a_free_cache_hit(two_player_client):
    """Possession of the audio is what authorizes the read on *this* route, which
    is why a cache hit here is fine even for a row someone else owns — the caller
    supplied the bytes, and the id is their hash. §16.4 makes it free."""
    two_player_client.post("/v1/analyze/upload", headers=ALICE,
                           files={"file": ("demo.mp3", AUDIO, "audio/mpeg")})
    store = two_player_client.app.state.store
    spent_before = store.usage_today("bob")

    again = two_player_client.post("/v1/analyze/upload", headers=BOB,
                                   files={"file": ("demo.mp3", AUDIO, "audio/mpeg")})
    assert again.status_code == 200
    assert again.json()["song"]["id"].startswith("yt:up_")
    assert store.usage_today("bob") == spent_before


def test_a_re_analysis_cannot_quietly_publish_an_upload(two_player_client):
    """`put_map` upserts, so re-analysis must preserve `owner_uid` for the same
    reason it preserves an admin's `offset_ms`: a fresh run has no better
    information, and losing it here would publish a private recording."""
    two_player_client.post("/v1/analyze/upload", headers=ALICE,
                           files={"file": ("demo.mp3", AUDIO, "audio/mpeg")})
    store = two_player_client.app.state.store
    video_id = upload_id(AUDIO)

    cached = store.get_map(video_id, "normal")
    store.put_map(video_id=video_id, difficulty="normal", song=cached.song, sync=None,
                  engine_chords="x", engine_beats="y", analyzed_at="2026-09-01T00:00:00Z",
                  owner_uid=None)

    assert store.get_map(video_id, "normal").owner_uid == "alice"
    assert store.list_catalog() == []


def test_a_fetched_video_is_still_public(two_player_client):
    """The other side of the same rule: a video analysis is catalog material, and
    the catalog is the app's home screen."""
    two_player_client.post("/v1/analyze", json={"videoId": "dQw4w9WgXcQ"}, headers=ALICE)

    results = two_player_client.get("/v1/catalog", headers=BOB).json()["results"]
    assert [row["videoId"] for row in results] == ["dQw4w9WgXcQ"]
    assert two_player_client.get("/v1/maps/dQw4w9WgXcQ", headers=BOB).status_code == 200


# --- bounded memory on the API container ------------------------------------

def test_the_body_is_buffered_once_not_twice():
    """Peak memory while reading an upload is about one copy of it, not two.

    Accumulating into a list and then `b"".join(chunks)` allocates the whole payload
    a second time while the first copy is still referenced, so the peak was **twice**
    the upload — 128 MB per request at the 64 MB cap, before the bytes had gone
    anywhere. With `max_inputs=10` that is 1.28 GB of a 4 GB container.

    Measured with `tracemalloc`, at a size large enough that the payload dominates
    anything the harness allocates around it. The bound is 1.6× rather than 1.0×
    because a `bytearray` grows geometrically and so briefly holds its old buffer
    alongside the new one — which is a fraction of a copy, not another whole one.
    """
    import asyncio
    import tracemalloc

    from app.config import Settings
    from app.main import _read_upload

    size = 8 * 1024 * 1024
    payload = b"\x00" * size

    class _Body:
        """The `UploadFile.read(n)` half of the interface, and nothing else."""

        def __init__(self, data):
            self._data = memoryview(data)
            self._at = 0

        async def read(self, n):
            chunk = bytes(self._data[self._at:self._at + n])
            self._at += len(chunk)
            return chunk

    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    result = asyncio.run(_read_upload(_Body(payload), Settings()))
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    assert len(result) == size
    overhead = (peak - before) / size
    assert overhead < 1.6, (
        f"reading an upload peaked at {overhead:.2f}× its size — a second full copy "
        f"of every body is what makes ten concurrent uploads a memory problem"
    )


def test_too_many_simultaneous_uploads_are_refused_rather_than_admitted(client,
                                                                       monkeypatch):
    """A 503 with `Retry-After` is a far better answer than an OOM kill, which takes
    the other nine in-flight requests with it.

    Refused immediately rather than queued: the client is holding a request open with
    a 64 MB body in it, so making it *wait* for a slot means the socket, the buffer
    and the slot are all occupied by something that has not started yet.
    """
    import asyncio

    from app import main

    # Every slot already taken.
    monkeypatch.setattr(main, "_upload_slots", asyncio.Semaphore(0))

    response = post(client)
    assert response.status_code == 503
    assert response.json()["code"] == "feature_disabled"
    assert int(response.headers["Retry-After"]) >= 1


def test_the_slot_is_released_when_an_upload_fails(client, monkeypatch):
    """A refused or failed upload must not leak its admission slot, or the third
    bad request closes the route for everyone."""
    import asyncio

    from app import main

    monkeypatch.setattr(main, "_upload_slots", asyncio.Semaphore(1))

    # 413: over the cap, raised from inside the slot.
    assert post(client, data=b"\x00" * (MAX_UPLOAD_BYTES + 1)).status_code == 413
    # The slot came back, so a good upload still works.
    assert post(client).status_code in {200, 202}


# The cap and the empty-body cases are already covered above
# (`test_an_oversized_upload_is_refused_without_being_read_whole`,
# `test_an_empty_upload_is_a_clean_400`) and still pass against the rewritten
# buffering, which is the point — the ceiling did not move, only the footprint.
