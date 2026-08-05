"""A `VideoSource` over audio the player supplied themselves.

Second input path, beside `ytdlp_source`, and the reason it exists is stated
plainly: **it is the only input this service has that carries no YouTube-terms
problem.** §2 concedes that fetching and decoding a recording separated from its
video contravenes the API terms as written, and the owner has knowingly accepted
that risk for the YouTube path. A file the player already has raises no such
question — the copyright analysis is the ordinary one (§2's invariants still do
all the work: audio never persisted, only the derived map stored, the map never a
product), and none of it depends on Google's tolerance.

That makes this more than a convenience feature. It is what §3's kill switch
degrades *to*: with `CHORDS_ANALYSIS_ENABLED=0`, or against YouTube's bot check,
the feature stops working entirely today. With this path it keeps working for
anyone holding their own audio.

That day is not hypothetical and cookies were never the thing holding it off:
the bot check is per egress IP, measured at ~20% of Modal containers, and
identical with and without cookies. This path is the only fetch route that does
not depend on YouTube's tolerance at all.

**Nothing here weakens §2.1.** The bytes live in the worker's memory for the
duration of one job, are written only into the caller's scratch directory (which
is destroyed on every exit path), and are never stored, never uploaded onward,
and never reachable by a second request — a re-upload sends the bytes again. The
id is a content hash, so the *derived map* caches; the audio does not.

**Ids are content-addressed.** `up_<sha256[:16]>` — deterministic, so:

- re-uploading the same file is a cache hit, which §16.4 makes free;
- `import` upserts the same Library row instead of duplicating it (§12.5);
- §3's blocklist works unchanged, keyed on the hash, so a takedown against
  specific audio is satisfiable the same way a video takedown is.

Distinguishable from a YouTube id by construction: those are exactly 11
characters of `[A-Za-z0-9_-]`, and this is 19 with a `up_` prefix.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

from ..errors import AnalysisError, VideoTooLong, VideoUnavailable
from .types import PCM, VideoMeta

log = logging.getLogger("chords.fetch.file")

_SAMPLE_RATE = 22050
_PROBE_TIMEOUT_S = 30
_DECODE_TIMEOUT_S = 120

# Two bounds on how much audio a decode may produce, and the gap between them is
# deliberate.
#
# `_LENGTH_TOLERANCE` is the **policy** check: §18's cap plus enough slack that a
# file whose declared duration is a second or two out is not refused over it.
# Exceeding it raises `VideoTooLong`, which is an answer the player can act on.
#
# `_DECODE_CEILING` is the **memory** bound, and it is needed because the policy
# check runs on samples that are already in this process. `gate()` decides on the
# duration ffprobe *claimed*, and a claim is not a measurement — a container that
# understates its real length decodes unbounded into the scratch dir (tmpfs, so
# RAM on the worker) and is then read into memory a second time by `_read_wav`.
# On a container with a 4 GB cap that is how a file that should have been a 400
# becomes a dead worker instead. `-t` bounds what ffmpeg will emit at all.
#
# It sits **above** the tolerance on purpose. Clamping at the same number would
# truncate an over-length file to exactly the limit, and the check below — which
# asks whether the decode came out *longer* than the tolerance — would then never
# fire: a refusal would have quietly become a shortened song.
_LENGTH_TOLERANCE = 1.1
_DECODE_CEILING = 1.25

UPLOAD_PREFIX = "up_"
UPLOAD_ID = re.compile(r"^up_[0-9a-f]{16}$")

# What the player is allowed to hand us. A 10-minute song (§18's cap) at a sane
# bitrate is a few MB; 64 MB is generous for lossless and still bounded, which is
# what keeps a worker with a memory cap erroring instead of dying.
MAX_UPLOAD_BYTES = 64 * 1024 * 1024


def upload_id(data: bytes) -> str:
    """The content-addressed id for a piece of audio."""
    return f"{UPLOAD_PREFIX}{hashlib.sha256(data).hexdigest()[:16]}"


def is_upload_id(value: str) -> bool:
    return bool(UPLOAD_ID.match(value or ""))


class FileSource:
    """`VideoSource` over bytes already in hand.

    Constructed per job with the audio it is to analyze, unlike `YtDlpSource`
    which is built once and fetches on demand — because there is nothing to fetch
    and nowhere the bytes could be fetched *from* without persisting them, which
    §2.1 forbids.
    """

    name = "upload"
    version = "1"

    def __init__(self, data: bytes, *, filename: str | None = None, settings=None) -> None:
        self._data = data
        self._settings = settings
        self._ffmpeg = os.environ.get("CHORDS_FFMPEG", "ffmpeg")
        self._ffprobe = os.environ.get("CHORDS_FFPROBE", "ffprobe")
        self.video_id = upload_id(data)
        # Only ever used as a display title, and stripped of any path the client
        # may have sent. It is player-supplied text that ends up in a database
        # column and on screen, so it is treated as untrusted.
        self.title = _clean_title(filename) or "Uploaded audio"

    # -- metadata only ------------------------------------------------------

    def probe(self, video_id: str) -> VideoMeta:
        """Duration and title, without decoding.

        Runs before the §3 gate, exactly as the YouTube path does, so the length
        cap and the blocklist are both decidable before any audio is decoded —
        the ordering is not an optimisation here either (see `fetch.py`).
        """
        with _temp_media(self._data) as media:
            result = _run([
                self._ffprobe, "-v", "error", "-show_entries", "format=duration",
                "-of", "json", str(media),
            ], timeout=_PROBE_TIMEOUT_S)
        if result.returncode != 0:
            log.warning("ffprobe rejected an upload: %s", (result.stderr or "").strip()[:400])
            raise VideoUnavailable("That file couldn’t be read as audio.")
        try:
            duration = float(json.loads(result.stdout)["format"]["duration"])
        except (ValueError, KeyError, TypeError) as exc:
            raise VideoUnavailable("That file’s length couldn’t be determined.") from exc
        if duration <= 0:
            raise VideoUnavailable("That file has no audio in it.")

        return VideoMeta(
            video_id=self.video_id,
            title=self.title,
            duration_s=duration,
            # An upload has no channel. §3's per-channel blocklist simply does not
            # apply; the per-id half does, keyed on the content hash.
            channel_id=None,
        )

    # -- the only code here that holds audio --------------------------------

    def decode(self, video_id: str, workdir: Path, *,
               sample_rate: int = _SAMPLE_RATE) -> tuple[PCM, int]:
        """Decode into `workdir`, which the caller destroys.

        Same contract as `YtDlpSource.decode`: nothing is written outside the
        scratch directory, and no path is returned — only samples, which die with
        the worker.
        """
        import numpy as np

        workdir = Path(workdir)
        source = workdir / "upload.bin"
        source.write_bytes(self._data)
        wav = workdir / "decoded.wav"

        limit = getattr(self._settings, "max_video_seconds", None)
        command = [self._ffmpeg, "-y", "-loglevel", "error", "-i", str(source)]
        if limit:
            command += ["-t", f"{limit * _DECODE_CEILING:.3f}"]
        command += ["-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(wav)]

        result = _run(command, timeout=_DECODE_TIMEOUT_S)
        try:
            source.unlink()
        except OSError:
            pass
        if result.returncode != 0 or not wav.is_file():
            log.warning("decode failed for an upload: %s", (result.stderr or "").strip()[:400])
            raise VideoUnavailable("That file couldn’t be decoded — try a normal audio file.")

        pcm = _read_wav(np, wav)
        try:
            wav.unlink()
        except OSError:
            pass
        if pcm.size == 0:
            raise VideoUnavailable("That file had no audio track.")

        # The cap re-checked against what actually decoded, for the same reason
        # the YouTube path does it: a container's metadata is a claim, and this
        # is the measurement. Still reachable with `-t` in place above, which is
        # why the ceiling is the looser of the two numbers.
        if limit and pcm.size / sample_rate > limit * _LENGTH_TOLERANCE:
            raise VideoTooLong(int(limit))

        return pcm, sample_rate


def _clean_title(filename: str | None) -> str:
    """A player-supplied filename → something safe to show and to store."""
    if not filename:
        return ""
    stem = Path(filename.replace("\\", "/")).name
    stem = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", stem)
    stem = re.sub(r"[\x00-\x1f\x7f]", "", stem).strip()
    return stem[:120]


class _temp_media:
    """The bytes as a file ffprobe can open, in the container's own tmp.

    Not in the §2.1 scratch root: `probe` runs *before* the pipeline opens one
    (the gate has to be decidable first). Removed on every exit path, including
    the exception ones, and a Modal container's filesystem dies with it either
    way.
    """

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._path: Path | None = None

    def __enter__(self) -> Path:
        import tempfile
        handle, path = tempfile.mkstemp(prefix="chords-upload-", suffix=".bin")
        with os.fdopen(handle, "wb") as target:
            target.write(self._data)
        os.chmod(path, 0o600)
        self._path = Path(path)
        return self._path

    def __exit__(self, *_exc) -> None:
        if self._path is not None:
            try:
                self._path.unlink()
            except OSError:
                pass


def _run(command: list[str], *, timeout: int) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(command, capture_output=True, text=True,
                              timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise AnalysisError("That file took too long to read — try a shorter one.") from exc
    except FileNotFoundError as exc:
        raise AnalysisError(f"Required tool missing from this image: {command[0]}") from exc


def _read_wav(np, path: Path):
    """16-bit PCM WAV → mono float32 in [-1, 1]. Same reasoning as the yt-dlp
    path: ffmpeg has produced exactly the format we asked for, so the stdlib
    `wave` module needs no format guessing and no extra C library."""
    import wave

    with wave.open(str(path)) as source:
        frames = source.readframes(source.getnframes())
    if not frames:
        return np.zeros(0, dtype="float32")
    return np.frombuffer(frames, dtype="<i2").astype("float32") / 32768.0


def available() -> bool:
    """Whether this image can decode an upload at all.

    Checked the same way `ytdlp_source.available()` is, and for the same reason:
    the API container must report honestly that it cannot, rather than accepting
    an upload it has no way to process.
    """
    return (shutil.which(os.environ.get("CHORDS_FFMPEG", "ffmpeg")) is not None
            and shutil.which(os.environ.get("CHORDS_FFPROBE", "ffprobe")) is not None)
