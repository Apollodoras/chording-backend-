"""§15 — the bar grid everything downstream stands on, and the section it fills.

This module used to hold both halves of structure: slicing the chord timeline
into bars, and deciding where the sections are. The second half moved to
`form.py` when §20 replaced it — exact-equality, adjacent-only segmentation
could not survive an engine mistake, and could not tell that verse 1 and verse 3
were the same music. What stays here is the part that was never in question:

**`bars_from_spans` puts everything on whole bars.** Section lengths must be a
whole number of bars or §13.2's beat axis drifts (see `app/sync.py`), and the
cheapest way to guarantee that is to stop working in loose beats the moment the
chords are quantized. Everything after this function counts bars.

`BarChord` and `Section` live here because both halves need them and because
they are what `compile.py` reads — keeping the types next to the bar slicing,
and the algorithms in `form.py`, is what stops the two from drifting apart the
way three private copies of "which beat is this?" once did (see `axis.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import GridSpan

# §15's practical floor: "one section per structural segment, minimum ~4 bars".
MIN_SECTION_BARS = 4


@dataclass(frozen=True)
class BarChord:
    """One chord's slice of one bar. Beats are **bar-local**, which is exactly
    what ``Bar.chordSpans[]`` wants (§12.1)."""

    root_pc: int
    quality: str
    start_beat: float
    length_beats: float
    # Carried down from the `GridSpan` this slice came from, because §20's
    # consensus vote needs it: a chord the engine was *confident* about is
    # evidence the music changed, and a doubtful one is evidence it misheard.
    # Without this the vote could only count occurrences, and counting alone
    # cannot tell a real difference from a repeated mistake.
    confidence: float = 1.0

    @property
    def signature(self) -> tuple[int, str]:
        """What "the same chord" means for structure — root and quality, not
        duration. A verse whose last bar holds an extra beat is still that verse.
        """
        return (self.root_pc, self.quality)


@dataclass
class Section:
    """One pass of the section's bars, plus how many times it repeats.

    Holding one pass rather than the expanded bars is what lets the compiler use
    ``repeats`` (§15: "a 4-bar progression played 4× is one section with
    repeats: 4, not 16 bars of explicit chords") — which is both smaller on the
    wire and what makes the campfire header count bars the way a player would.

    ``group`` is §20's addition: the rehearsal letter of the repeat group this
    section belongs to. Two sections carrying the same letter are two
    occurrences of one piece of music, which is the fact the container has no
    field for and which `consensus.py` needs in order to vote across them.
    """

    kind: str = "custom"
    name: str = ""
    bars: list[list[BarChord]] = field(default_factory=list)
    repeats: int = 1
    start_bar: int = 0
    group: str = ""

    @property
    def total_bars(self) -> int:
        return len(self.bars) * self.repeats

    @property
    def signature(self) -> tuple[tuple[tuple[int, str], ...], ...]:
        return tuple(tuple(c.signature for c in bar) for bar in self.bars)


def bars_from_spans(spans: list[GridSpan], bar_beats: int) -> list[list[BarChord]]:
    """Slice a contiguous chord timeline into bars.

    A chord crossing a barline becomes one `BarChord` in each bar it touches —
    the same shape ``SongSection.materializedBars`` produces on device, so a
    chord held for two bars renders as two spans rather than one that overruns
    its bar (which the lint rejects, and the app would truncate).

    Trailing beats that don't complete a bar are dropped: a partial bar has no
    downbeat to anchor and would put every later bar off the grid.
    """
    if not spans or bar_beats <= 0:
        return []

    end_beat = max(s.end_beat for s in spans)
    bar_count = end_beat // bar_beats
    bars: list[list[BarChord]] = [[] for _ in range(bar_count)]

    for span in spans:
        for bar_index in range(span.start_beat // bar_beats, bar_count):
            bar_start = bar_index * bar_beats
            if bar_start >= span.end_beat:
                break
            lo = max(span.start_beat, bar_start)
            hi = min(span.end_beat, bar_start + bar_beats)
            if hi <= lo:
                continue
            bars[bar_index].append(BarChord(
                root_pc=span.root_pc, quality=span.quality,
                start_beat=float(lo - bar_start), length_beats=float(hi - lo),
                confidence=span.confidence,
            ))
    return [bar for bar in bars if bar]
