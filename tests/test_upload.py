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

import pytest
from fastapi.testclient import TestClient

from app.analysis import engines
from app.analysis.file_source import (
    MAX_UPLOAD_BYTES,
    FileSource,
    available,
    is_upload_id,
    upload_id,
)
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
