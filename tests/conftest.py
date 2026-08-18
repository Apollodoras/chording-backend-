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

import random
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.analysis.axis import BeatAxis, build_axis  # noqa: E402
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


def known_axis() -> BeatAxis:
    """The known song's beat axis — what the chart is actually built on.

    Everything downstream of the beat tracker addresses song beats through a
    `BeatAxis` rather than through a raw `BeatGrid`, because the grid alone does
    not say where beat 0 is (see `app/analysis/axis.py`). For the known song the
    two coincide, which is exactly why the distinction went unnoticed for so
    long.
    """
    axis = build_axis(known_grid())
    assert axis is not None, "the known song's grid must yield an axis"
    return axis


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


def jittered_onsets(*, spread_ms: int = 30, bias_ms: int = -12) -> list[Onset]:
    """`known_onsets`, played by hands instead of by a sequencer.

    Two things separate a real detection from the exact grid every strumming
    fixture used before this one, and the pattern extractor has to survive both:

    - **spread** — no onset lands exactly on its cell;
    - **bias** — the detector's onsets skew *early*. That is the direction that
      matters, because an onset ahead of the downbeat wraps to the far end of the
      bar when folded, and for a long time nothing there could claim it.

    Deterministic (a fixed seed): a fixture that fails one run in twenty is worse
    than no fixture.
    """
    rng = random.Random(20260804)
    onsets: list[Onset] = []
    for bar in range(TOTAL_BEATS // BAR_BEATS):
        bar_start = bar * BAR_BEATS * MS_PER_BEAT
        for offset in DDUUDU:
            wobble = rng.uniform(-spread_ms, spread_ms) + bias_ms
            onsets.append(Onset(
                t_ms=int(bar_start + offset * MS_PER_BEAT + wobble),
                strength=1.6 if offset == 0.0 else 1.0,
            ))
    return sorted(onsets, key=lambda o: o.t_ms)


def ghost_sixteenths(*, position: float = 1.25) -> list[Onset]:
    """One 16th-note hi-hat per bar — what a full mix always adds to the strums.

    Quiet (a ghost note is not a strum) and perfectly consistent, which is the
    combination that used to drag the whole grid to 16ths on the strength of a
    sixth of the onsets.
    """
    return [
        Onset(t_ms=int((bar * BAR_BEATS + position) * MS_PER_BEAT), strength=0.35)
        for bar in range(TOTAL_BEATS // BAR_BEATS)
    ]


def known_meta(video_id: str = "dQw4w9WgXcQ") -> VideoMeta:
    return VideoMeta(video_id=video_id, title="Known Song",
                     duration_s=DURATION_S, channel_id="UCtest")


# --- a recording with a pickup ----------------------------------------------
#
# `known_grid` above starts the song at t=0 on beat 0 AND downbeat 0. That is the
# one alignment case that needs no reconciliation between the chart's beat axis
# and the sidecar's anchors, and for a long time it was the only case the suite
# had — so a chart that was uniformly phase-shifted against its recording scored
# 303 green tests.
#
# Real recordings almost never oblige: a beat tracker emits beats from the first
# audible pulse, and the first *downbeat* is typically one to three beats later
# (a pickup, a count-in, an intro fill). This builder makes that difference a
# parameter, so a test can ask for the normal case rather than the lucky one.

@dataclass(frozen=True)
class Recording:
    """A synthetic recording with exact ground truth, in the units the pipeline
    consumes: engine-native chord labels in ms, and a beat grid."""

    grid: BeatGrid
    chords: list[RawChordSpan]
    meta: VideoMeta
    # (start_ms, end_ms, engine label) per bar — what is actually sounding.
    truth: list[tuple[int, int, str]]

    @property
    def duration_ms(self) -> int:
        return int(round(self.meta.duration_s * 1000))

    def label_at(self, t_ms: float) -> str | None:
        for start, end, label in self.truth:
            if start <= t_ms < end:
                return label
        return None


def recording(
    *,
    progression: list[str] | None = None,
    bars: int = 16,
    pickup_beats: int = 0,
    ms_per_beat: int = MS_PER_BEAT,
    first_beat_ms: int = 0,
    bar_beats: int = BAR_BEATS,
    confidence: float = 0.95,
    odd_bars: dict[int, int] | None = None,
) -> Recording:
    """A recording whose first downbeat is `pickup_beats` beats after beat 0.

    `pickup_beats=0` reproduces `known_grid`'s geometry. Anything above 0 is the
    case a real tracker produces, and the case the chart has to reconcile.

    `odd_bars` maps a bar index to its real beat count, for songs that do not
    hold one meter throughout — Here Comes The Sun famously drops in 11/8 and
    15/8 bars, and it was the worst-scoring track in the corpus for exactly that
    reason, independently of any pickup.
    """
    progression = progression or PROGRESSION
    odd_bars = odd_bars or {}

    # Lay the beats out bar by bar, so a bar can be a different length from its
    # neighbours without the ones after it losing their downbeat.
    beats_ms: list[int] = [first_beat_ms + i * ms_per_beat for i in range(pickup_beats)]
    downbeats_ms: list[int] = []
    cursor = first_beat_ms + pickup_beats * ms_per_beat
    for bar in range(bars):
        downbeats_ms.append(cursor)
        for _ in range(odd_bars.get(bar, bar_beats)):
            beats_ms.append(cursor)
            cursor += ms_per_beat
    downbeats_ms.append(cursor)
    beats_ms.append(cursor)
    chords: list[RawChordSpan] = []
    truth: list[tuple[int, int, str]] = []
    for index in range(len(downbeats_ms) - 1):
        start, end = downbeats_ms[index], downbeats_ms[index + 1]
        label = progression[index % len(progression)]
        chords.append(RawChordSpan(start_ms=start, end_ms=end, label=label,
                                   confidence=confidence))
        truth.append((start, end, label))

    grid = BeatGrid(
        beats_ms=beats_ms,
        downbeats_ms=downbeats_ms,
        bpm=60_000.0 / ms_per_beat,
        confidence=confidence,
        time_signature=f"{bar_beats}/4",
    )
    meta = VideoMeta(video_id="dQw4w9WgXcQ", title="Known Song",
                     duration_s=(beats_ms[-1] + ms_per_beat) / 1000.0,
                     channel_id="UCtest")
    return Recording(grid=grid, chords=chords, meta=meta, truth=truth)


# --- grids a real tracker produces ------------------------------------------
#
# Every beat grid above lays downbeats at `beats[::bar_beats]`, and so did every
# grid in the suite — `known_downbeats`, `test_model`, `test_meter`,
# `test_pipeline`, all of them. A perfectly regular downbeat sequence appeared in
# 519 green tests and an irregular one appeared in none, which is exactly why a
# defect that put 15–20% of the catalog's bars at the wrong length was invisible
# here while the player could see it plainly.
#
# `beat_this` gets the pulse right and the bar wrong, in both directions: an
# extra downbeat one beat into a real bar (a half-bar, then a short one) and a
# missing one (a bar of double length). Both are below.

def broken_downbeats(*, spurious: tuple[int, ...] = (),
                     dropped: tuple[int, ...] = (),
                     beat_into_bar: int = 1) -> list[int]:
    """`known_downbeats` with tracker mistakes in it.

    `spurious` names bars that get an **extra** downbeat `beat_into_bar` beats
    after their real one; `dropped` names bars whose downbeat the tracker missed.
    Both are given as bar indices, so a test reads as "bar 10 is broken" rather
    than as an arithmetic puzzle about milliseconds.
    """
    downbeats = [t for i, t in enumerate(known_downbeats()) if i not in dropped]
    extra = [bar * BAR_BEATS * MS_PER_BEAT + beat_into_bar * MS_PER_BEAT
             for bar in spurious]
    return sorted(set(downbeats + extra))


def broken_grid(**kwargs) -> BeatGrid:
    """`known_grid` with a downbeat sequence a real tracker might have produced.

    The **beats** are untouched, which is the point: the pulse is the half a
    tracker gets right, and the repair in `downbeats.py` is allowed to choose
    among these beats but never to invent a time that isn't one of them.
    """
    grid = known_grid()
    return BeatGrid(
        beats_ms=grid.beats_ms,
        downbeats_ms=broken_downbeats(**kwargs),
        bpm=grid.bpm,
        confidence=grid.confidence,
        time_signature=grid.time_signature,
    )


# --- fakes ------------------------------------------------------------------

@dataclass
class FakeChordEngine:
    name: str = "fake-chords"
    version: str = "1.0.0"
    spans: list[RawChordSpan] | None = None

    def analyze(self, pcm, sr, *, tuning=None):
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
    # What the real source reports with no proxy configured, which is the
    # deployment's default. Present so `/healthz`'s `egress` field is exercised
    # by something other than the absent-source case — those two answers are
    # different (`"direct"` vs `None`) and a fake that omitted this would make
    # them look the same.
    egress = "direct"

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
