"""The fetch/decode seam — §5.1, and the sharpest edge in the whole service.

Two things live behind this interface, and they are separated on purpose:

- **`probe`** learns a video's duration, title and **channel id** without
  downloading anything. That ordering is not an optimisation: §3's blocklist is
  per-video *and* per-channel, and the 10-minute cap (§18) is a rejection. Both
  have to be decidable *before* a byte of audio is fetched, or the service
  downloads recordings it was never allowed to touch.
- **`decode`** produces mono float32 PCM at 22.05 kHz into a caller-supplied
  scratch directory (`yt-dlp` → `ffmpeg`), and is the only code in the service
  that ever holds audio.

**This is the part §2 concedes openly**: fetching and decoding audio separated
from video contravenes the YouTube API terms as written, and the owner is
knowingly accepting that risk. The mitigation is blast radius, not deniability —
so this module is imported *only* by the worker, its dependencies live in the
`audio` extra rather than the base install, and the API container is not built
with them. If you find yourself importing this from `app/main.py`, that is the
mistake.

The implementation is §8 step 4's work and is not written yet, which is why the
default source below refuses rather than pretends. `build_source` returning None
is a legitimate state: the API still serves cached maps, `/healthz` reports
`fetch: "unconfigured"`, and a new analysis answers a clean "unavailable".
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..errors import AnalysisError, CODE_FEATURE_DISABLED
from .types import PCM, VideoMeta

log = logging.getLogger("chords.fetch")

# YouTube ids are exactly 11 characters of [A-Za-z0-9_-]. Validated because the
# id reaches a subprocess argv and a database key, and because a caller that
# sends a URL should get a clear error rather than a mysterious fetch failure.
_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")

# The URL forms a player might paste, in the order they turn up.
_URL_PATTERNS = (
    re.compile(r"(?:youtube\.com|youtube-nocookie\.com)/watch\?(?:.*&)?v=([A-Za-z0-9_-]{11})"),
    re.compile(r"youtu\.be/([A-Za-z0-9_-]{11})"),
    re.compile(r"(?:youtube\.com|youtube-nocookie\.com)/(?:embed|shorts|live|v)/([A-Za-z0-9_-]{11})"),
)


def parse_video_id(text: str) -> str | None:
    """A bare id or any common YouTube URL → the id. None when it isn't one.

    Accepting URLs is not politeness: the app's entry point is "paste or pick a
    video", and a player pastes what the share sheet gave them.
    """
    candidate = text.strip()
    if _VIDEO_ID.match(candidate):
        return candidate
    for pattern in _URL_PATTERNS:
        match = pattern.search(candidate)
        if match:
            return match.group(1)
    return None


@runtime_checkable
class VideoSource(Protocol):
    """Everything that touches a recording, behind one name."""

    name: str
    version: str

    def probe(self, video_id: str) -> VideoMeta:
        """Metadata only — no media. Raises `VideoUnavailable` if unreachable."""
        ...

    def decode(self, video_id: str, workdir: Path, *, sample_rate: int = 22050) -> tuple[PCM, int]:
        """Fetch and decode to mono PCM inside `workdir`.

        `workdir` is a `scratch.scratch()` directory: it exists for the duration
        of the call and is destroyed on every exit path afterwards. Nothing here
        may write outside it, and nothing may return a path.
        """
        ...


class SourceUnavailable(AnalysisError):
    def __init__(self, message: str = "Chord analysis isn’t available on this deployment."):
        super().__init__(message, CODE_FEATURE_DISABLED, status=503)


def build_source(settings) -> VideoSource | None:
    """The configured source, or None when this build has no audio stack.

    None is the API container's normal state and must stay serviceable: cached
    maps still return, `/v1/me` still answers, and only a *new* analysis is
    refused. That is also what makes §3's kill switch cheap — a deployment with
    no source behaves exactly like a deployment with the switch off.
    """
    try:
        from .ytdlp_source import YtDlpSource, available
    except ImportError:
        log.info("fetch: no audio source in this build (API container, or `audio` extra not installed)")
        return None
    # Importable is not the same as usable: `ytdlp_source` drives yt-dlp and
    # ffmpeg as subprocesses, so it imports fine in an image that has neither.
    # Checking the tools rather than the import is what keeps `/healthz`'s
    # `fetch: configured` from being a lie the first job discovers.
    if not available():
        log.info("fetch: yt-dlp or ffmpeg missing — no audio source in this build")
        return None
    return YtDlpSource(settings)
