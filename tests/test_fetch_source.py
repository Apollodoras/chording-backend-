"""The fetch/decode implementation (§8 step 4), tested without a network.

yt-dlp and ffmpeg are driven as subprocesses, which is what makes this testable
at all: `_run` is the single seam, so every branch — unavailable video, live
stream, missing duration, oversized download, empty audio — can be exercised by
handing back the process result the real tool would have produced. None of these
tests touch YouTube, and the suite stays runnable in CI with no credentials.

The point of most of them is the §16.3 split: a video that is private, removed
or region-locked is a **player-visible outcome** (`video_unavailable`), not a
server fault, and reporting it as "something went wrong on our side" would be
both wrong and unactionable.
"""

from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path

import pytest

from app.analysis import ytdlp_source
from app.analysis.ytdlp_source import FetchError, YtDlpSource, _read_wav
from app.errors import CODE_VIDEO_UNAVAILABLE, VideoTooLong, VideoUnavailable

VIDEO = "dQw4w9WgXcQ"


def result(returncode: int = 0, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(args=["yt-dlp"], returncode=returncode,
                                       stdout=stdout, stderr=stderr)


def patch_run(monkeypatch, handler):
    monkeypatch.setattr(ytdlp_source, "_run", handler)


# --- probe: metadata only ----------------------------------------------------

def test_probe_returns_the_metadata_the_gate_needs(monkeypatch):
    """§3's blocklist is per-channel, so `channel_id` is not optional decoration."""
    payload = json.dumps({"title": " Let It Be ", "duration": 243.3,
                          "channel_id": "UCabc", "is_live": False})
    patch_run(monkeypatch, lambda *a, **k: result(stdout=payload))

    meta = YtDlpSource().probe(VIDEO)

    assert meta.video_id == VIDEO
    assert meta.title == "Let It Be"
    assert meta.duration_s == pytest.approx(243.3)
    assert meta.channel_id == "UCabc"


def test_probe_never_downloads_media():
    """The ordering §5.1 insists on: the blocklist and the length cap must both
    be decidable before a byte is fetched."""
    source = YtDlpSource()
    captured: list[list[str]] = []

    def spy(command, **kwargs):
        captured.append(command)
        return result(stdout=json.dumps({"title": "x", "duration": 10}))

    ytdlp_source._run, original = spy, ytdlp_source._run
    try:
        source.probe(VIDEO)
    finally:
        ytdlp_source._run = original

    assert "--skip-download" in captured[0]


@pytest.mark.parametrize("stderr", [
    "ERROR: [youtube] xyz: Video unavailable",
    "ERROR: [youtube] xyz: Private video. Sign in if you've been granted access",
    "ERROR: This video is not available in your country",
    "ERROR: Sign in to confirm your age",
])
def test_an_unreachable_video_is_the_players_problem_not_a_server_error(monkeypatch, stderr):
    patch_run(monkeypatch, lambda *a, **k: result(returncode=1, stderr=stderr))

    with pytest.raises(VideoUnavailable) as caught:
        YtDlpSource().probe(VIDEO)

    assert caught.value.code == CODE_VIDEO_UNAVAILABLE
    assert caught.value.status == 422


def test_an_unrecognised_failure_is_reported_as_ours(monkeypatch):
    """The inverse: don't launder a real bug into "that video is private"."""
    patch_run(monkeypatch, lambda *a, **k: result(returncode=1, stderr="ERROR: boom"))

    with pytest.raises(FetchError):
        YtDlpSource().probe(VIDEO)


def test_a_live_stream_is_refused(monkeypatch):
    payload = json.dumps({"title": "live", "duration": 60, "is_live": True})
    patch_run(monkeypatch, lambda *a, **k: result(stdout=payload))

    with pytest.raises(VideoUnavailable):
        YtDlpSource().probe(VIDEO)


def test_a_video_with_no_duration_is_refused(monkeypatch):
    """§18's cap is not advisory. No duration means it cannot be enforced, and
    fetching something unbounded is the failure the cap exists to prevent."""
    patch_run(monkeypatch, lambda *a, **k: result(stdout=json.dumps({"title": "x"})))

    with pytest.raises(VideoUnavailable):
        YtDlpSource().probe(VIDEO)


def test_unreadable_metadata_does_not_crash_the_worker(monkeypatch):
    patch_run(monkeypatch, lambda *a, **k: result(stdout="not json"))

    with pytest.raises(FetchError):
        YtDlpSource().probe(VIDEO)


def test_a_timeout_becomes_a_message_the_player_can_read(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="yt-dlp", timeout=1)

    monkeypatch.setattr(subprocess, "run", timeout)

    with pytest.raises(FetchError) as caught:
        YtDlpSource().probe(VIDEO)
    assert "too long" in str(caught.value).lower()


def test_a_missing_tool_says_so_rather_than_blaming_the_video(monkeypatch):
    """Wrong image — the API container, or a worker built without ffmpeg. A
    deployment error must not read as a video error."""
    def missing(*args, **kwargs):
        raise FileNotFoundError("yt-dlp")

    monkeypatch.setattr(subprocess, "run", missing)

    with pytest.raises(FetchError) as caught:
        YtDlpSource().probe(VIDEO)
    assert "missing from this image" in str(caught.value)


# --- decode ------------------------------------------------------------------

def write_wav(path: Path, samples: list[int], rate: int = 22050) -> None:
    import struct

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"".join(struct.pack("<h", s) for s in samples))


def test_read_wav_returns_normalised_mono_floats(tmp_path):
    import numpy as np

    path = tmp_path / "a.wav"
    write_wav(path, [0, 16384, -16384, 32767])
    pcm = _read_wav(np, path)

    assert pcm.dtype == np.dtype("float32")
    assert pcm[0] == pytest.approx(0.0)
    assert pcm[1] == pytest.approx(0.5, abs=1e-4)
    assert pcm[2] == pytest.approx(-0.5, abs=1e-4)
    assert abs(pcm).max() <= 1.0


def test_decode_leaves_nothing_behind_in_the_scratch_directory(tmp_path, monkeypatch):
    """§2.1 at the level below `scratch()`: even inside the directory that is
    about to be destroyed, the decoder frees what it no longer needs, because the
    worker's memory cap is sized for one copy of the audio and not three."""
    workdir = tmp_path / "work"
    workdir.mkdir()

    def fake_run(command, **kwargs):
        if command[0] == "ffmpeg":
            write_wav(workdir / "decoded.wav", [0, 1000, -1000, 0])
            return result()
        (workdir / "media.webm").write_bytes(b"fake")
        return result()

    patch_run(monkeypatch, fake_run)

    pcm, rate = YtDlpSource().decode(VIDEO, workdir)

    assert rate == 22050
    assert len(pcm) == 4
    assert list(workdir.iterdir()) == []


def test_decode_rejects_audio_that_outruns_the_cap(tmp_path, monkeypatch):
    """The cap re-checked against what was actually decoded — `probe`'s duration
    is metadata, and metadata can lie."""
    from dataclasses import dataclass

    @dataclass
    class Limited:
        max_video_seconds: int = 1

    workdir = tmp_path / "work"
    workdir.mkdir()

    def fake_run(command, **kwargs):
        if command[0] == "ffmpeg":
            write_wav(workdir / "decoded.wav", [0] * (22050 * 3))
            return result()
        (workdir / "media.webm").write_bytes(b"fake")
        return result()

    patch_run(monkeypatch, fake_run)

    with pytest.raises(VideoTooLong):
        YtDlpSource(Limited()).decode(VIDEO, workdir)


def test_an_empty_audio_track_is_an_error_not_an_empty_song(tmp_path, monkeypatch):
    workdir = tmp_path / "work"
    workdir.mkdir()

    def fake_run(command, **kwargs):
        if command[0] == "ffmpeg":
            write_wav(workdir / "decoded.wav", [])
            return result()
        (workdir / "media.webm").write_bytes(b"fake")
        return result()

    patch_run(monkeypatch, fake_run)

    with pytest.raises(FetchError):
        YtDlpSource().decode(VIDEO, workdir)


def test_an_aborted_oversized_download_is_caught(tmp_path, monkeypatch):
    """yt-dlp exits 0 when `--max-filesize` aborts, so a zero exit code is not
    proof that anything arrived."""
    workdir = tmp_path / "work"
    workdir.mkdir()
    patch_run(monkeypatch, lambda *a, **k: result())

    with pytest.raises(FetchError) as caught:
        YtDlpSource().decode(VIDEO, workdir)
    assert "too large" in str(caught.value).lower()


# --- wiring ------------------------------------------------------------------

def test_build_source_refuses_when_the_tools_are_absent(monkeypatch):
    """`/healthz` reports `fetch: configured` from this. Importable is not the
    same as usable, and a green health check hiding a dead fetch stage is exactly
    the failure the health check exists to prevent."""
    from app.analysis import fetch

    monkeypatch.setattr(ytdlp_source, "available", lambda: False)
    assert fetch.build_source(object()) is None
