"""Shared fixtures — and the fake engines that make the whole pipeline testable.

The design point from `app/analysis/pipeline.py` pays off here: everything after
`decode` is pure, so a chord engine is just "a function that returns a list of
labelled spans". These fakes stand in for BTC/madmom/librosa and let the suite
exercise quantization, structure, difficulty tiers, the compiler and §13.2's
anchor invariant with **no audio, no model weights and no network** — which is
also what makes it runnable in CI before the §8-step-2 benchmark has happened.

`KNOWN_SONG` is deliberately exact: 120 bpm, 4/4, a G–D–Em–C progression of four
bars played four times, strummed D-DU-UD-U. Every number downstream is therefore
predictable, so a test can assert "the third bar's chord is Em" rather than
"something plausible came out".
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.analysis.types import BeatGrid, Onset, RawChordSpan, VideoMeta  # noqa: E402
from app.config import Settings  # noqa: E402
from app.store import SQLiteStore  # noqa: E402

# --- the known song ---------------------------------------------------------

BPM = 120
MS_PER_BEAT = 500          # 60_000 / 120
BAR_BEATS = 4
PROGRESSION = ["G:maj", "D:maj", "E:min", "C:maj"]
PASSES = 4                 # 16 bars total
BEATS_PER_CHORD = 4

TOTAL_BEATS = len(PROGRESSION) * BEATS_PER_CHORD * PASSES   # 64
DURATION_S = TOTAL_BEATS * MS_PER_BEAT / 1000.0             # 32.0

# The pattern everybody actually plays, as bar-local beat offsets (0-indexed):
# down, down, up, up, down, up.
DDUUDU = [0.0, 1.0, 1.5, 2.5, 3.0, 3.5]


def known_beats() -> list[int]:
    return [i * MS_PER_BEAT for i in range(TOTAL_BEATS + 1)]


def known_downbeats() -> list[int]:
    return [i * MS_PER_BEAT for i in range(0, TOTAL_BEATS + 1, BAR_BEATS)]


def known_grid(*, confidence: float = 0.95) -> BeatGrid:
    return BeatGrid(
        beats_ms=known_beats(),
        downbeats_ms=known_downbeats(),
        bpm=float(BPM),
        confidence=confidence,
        time_signature="4/4",
    )


def known_chords(*, confidence: float = 0.9) -> list[RawChordSpan]:
    spans: list[RawChordSpan] = []
    t = 0
    for _ in range(PASSES):
        for label in PROGRESSION:
            spans.append(RawChordSpan(
                start_ms=t, end_ms=t + BEATS_PER_CHORD * MS_PER_BEAT,
                label=label, confidence=confidence,
            ))
            t += BEATS_PER_CHORD * MS_PER_BEAT
    return spans


def known_onsets() -> list[Onset]:
    """D-DU-UD-U in every bar, with the downbeat struck hardest."""
    onsets: list[Onset] = []
    for bar in range(TOTAL_BEATS // BAR_BEATS):
        bar_start = bar * BAR_BEATS * MS_PER_BEAT
        for offset in DDUUDU:
            onsets.append(Onset(
                t_ms=int(bar_start + offset * MS_PER_BEAT),
                strength=1.6 if offset == 0.0 else 1.0,
            ))
    return onsets


def known_meta(video_id: str = "dQw4w9WgXcQ") -> VideoMeta:
    return VideoMeta(video_id=video_id, title="Known Song",
                     duration_s=DURATION_S, channel_id="UCtest")


# --- fakes ------------------------------------------------------------------

@dataclass
class FakeChordEngine:
    name: str = "fake-chords"
    version: str = "1.0.0"
    spans: list[RawChordSpan] | None = None

    def analyze(self, pcm, sr):
        return list(self.spans) if self.spans is not None else known_chords()


@dataclass
class FakeBeatTracker:
    name: str = "fake-beats"
    version: str = "1.0.0"
    grid: BeatGrid | None = None

    def track(self, pcm, sr):
        return self.grid or known_grid()


@dataclass
class FakeOnsetDetector:
    name: str = "fake-onsets"
    version: str = "1.0.0"
    onsets: list[Onset] | None = None

    def detect(self, pcm, sr):
        return list(self.onsets) if self.onsets is not None else known_onsets()


class FakeSource:
    """Stands in for yt-dlp + ffmpeg.

    `decode` writes a file into the scratch directory it is handed — not because
    anything reads it, but so the scratch tests are exercising a directory with
    real contents in it, which is the case that matters.
    """

    name = "fake-source"
    version = "1.0.0"

    def __init__(self, meta: VideoMeta | None = None, *, error: Exception | None = None):
        self.meta = meta or known_meta()
        self.error = error
        self.decoded: list[Path] = []

    def probe(self, video_id: str) -> VideoMeta:
        if self.error:
            raise self.error
        return VideoMeta(video_id=video_id, title=self.meta.title,
                         duration_s=self.meta.duration_s, channel_id=self.meta.channel_id)

    def decode(self, video_id: str, workdir: Path, *, sample_rate: int = 22050):
        scratch_file = Path(workdir) / "audio.raw"
        scratch_file.write_bytes(b"\x00" * 1024)
        self.decoded.append(scratch_file)
        return object(), sample_rate


# --- fixtures ---------------------------------------------------------------

@pytest.fixture
def scratch_root(tmp_path_factory) -> str:
    """A scratch root the §2.1 guard will accept.

    `tmp_path_factory` lands under /private/var/folders on macOS and /tmp on
    Linux — both recognised ephemeral prefixes, so this exercises the real
    validator rather than bypassing it.
    """
    return str(tmp_path_factory.mktemp("scratch"))


@pytest.fixture
def settings(tmp_path, scratch_root) -> Settings:
    return Settings(
        db_path=str(tmp_path / "test.sqlite3"),
        dev_token="dev-token",
        daily_quota=5,
        scratch_root=scratch_root,
        admin_token="admin-secret",
        chord_engine="fake-chords",
        beat_tracker="fake-beats",
    )


@pytest.fixture
def store(settings) -> SQLiteStore:
    store = SQLiteStore(settings.db_path)
    yield store
    store.close()
