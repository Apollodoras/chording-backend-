"""What the analysis stages hand each other.

Everything here is **plain data with no audio in it**. That is deliberate and it
is handoff §2.1/§2.2 expressed as types: PCM exists only as a local variable
inside a worker call, and the moment a stage returns, what crosses the boundary
is chord symbols, timestamps and confidences. There is no type in this module
that could carry a spectrogram, a chroma matrix, or a path to a decoded file, so
"accidentally persist the audio" is not a mistake this pipeline can make by
forgetting something — it would take adding a field here first.

The engine protocols are §5.3's, kept swappable on purpose: the chord engine and
the beat tracker are chosen by the §8-step-2 benchmark, and `engine_name` +
`engine_version` are persisted on every stored map so a cache can be invalidated
*selectively* when one of them is upgraded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

# The decoded-audio handle. Typed as Any rather than np.ndarray because numpy is
# in the `audio` extra: the API container must not import it, and the type
# signature must not be what drags it in.
PCM = Any


@dataclass(frozen=True)
class RawChordSpan:
    """One chord as an engine reports it — **engine-native label**, unquantized.

    The label is whatever vocabulary the engine speaks (BTC and Chordino emit
    Harte: ``C:maj``, ``F#:min7``, ``G:maj/3``, ``N``). Normalization into the
    app's grammar happens in postprocess, not here, so an engine adapter never
    has to know what the app can play.
    """

    start_ms: int
    end_ms: int
    label: str
    confidence: float = 1.0

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass(frozen=True)
class BeatGrid:
    """The song's pulse: beat times, downbeat times, tempo, meter.

    Computed **before** chords and used to quantize them (§5: "order matters").
    ``downbeats_ms`` is a subset of ``beats_ms``; both are milliseconds from the
    start of the recording.
    """

    beats_ms: list[int]
    downbeats_ms: list[int]
    bpm: float
    confidence: float
    time_signature: str = "4/4"

    @property
    def is_usable(self) -> bool:
        """Enough grid to quantize against. Two downbeats is the floor — one
        cannot express a bar length, and §13.2 needs bars."""
        return len(self.beats_ms) >= 2 and len(self.downbeats_ms) >= 2


@dataclass(frozen=True)
class Onset:
    """A detected attack — the raw material of strumming-pattern extraction (§14).

    ``strength`` is relative within the track (a normalized onset-envelope peak),
    not an absolute level: accent is decided by comparing a stroke against its own
    bar's mean, so only the ordering has to be meaningful.
    """

    t_ms: int
    strength: float = 1.0


@dataclass(frozen=True)
class GridSpan:
    """A chord after quantization — addressed in **beat indices**, not time.

    This is the type post-processing works in, and the switch of units is the
    point: once a chord starts at beat 8 and lasts 4 beats, it can be merged,
    dropped, simplified and laid into bars without ever reasoning about
    milliseconds again. The trip back to milliseconds happens once, in the
    sidecar's anchors.
    """

    start_beat: int
    length_beats: int
    root_pc: int
    quality: str
    confidence: float = 1.0
    # False when normalization threw information away (an extension flattened, a
    # slash bass dropped). Counted, not acted on: a track where most chords
    # needed reducing is a track whose `hard` tier is a fiction.
    exact: bool = True

    @property
    def end_beat(self) -> int:
        return self.start_beat + self.length_beats


@dataclass(frozen=True)
class EngineInfo:
    """``name@version`` for one stage, persisted with every map (§5.3)."""

    name: str
    version: str

    def __str__(self) -> str:
        return f"{self.name}@{self.version}"


@runtime_checkable
class ChordEngine(Protocol):
    """§5.3's adapter interface, verbatim.

    Candidates to benchmark, roughly in order of expected quality: **BTC**
    (strongest on typical pop/rock, needs a GPU — available on Modal per §18),
    **Chordino/NNLS-Chroma** (predictable, no GPU), **autochord** (easy, weaker).
    """

    name: str
    version: str

    def analyze(self, pcm: PCM, sr: int) -> list[RawChordSpan]: ...


@runtime_checkable
class BeatTracker(Protocol):
    """The same seam for §5.2.

    ``madmom`` is the accuracy benchmark but is effectively unmaintained and
    breaks on NumPy 2.x (hence the `<2` pin in the `audio` extra); ``beat_this``
    and ``BeatNet`` are worth benchmarking first; ``librosa.beat.beat_track`` is
    the fallback — fast, weaker on downbeats.
    """

    name: str
    version: str

    def track(self, pcm: PCM, sr: int) -> BeatGrid: ...


@runtime_checkable
class OnsetDetector(Protocol):
    """§14's input. Separate from the beat tracker because the two answer
    different questions — where the pulse is, versus where the hand actually
    struck — and a track can have a confident grid with unreadable strumming."""

    name: str
    version: str

    def detect(self, pcm: PCM, sr: int) -> list[Onset]: ...


@dataclass(frozen=True)
class VideoMeta:
    """What the fetch stage learns about a video *before* deciding to decode it.

    ``channel_id`` is here because §3's blocklist is per-video **and
    per-channel**, and a channel-level takedown can only be honoured if the
    channel is known at fetch time.
    """

    video_id: str
    title: str
    duration_s: float
    channel_id: str | None = None
