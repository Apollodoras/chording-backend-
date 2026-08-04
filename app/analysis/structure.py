"""§15 — song structure, and the bar grid everything downstream stands on.

Section labels are **player-visible**: campfire's header prints "Verse · bar 2 of
8" and the rail marks section boundaries. So structure is worth getting roughly
right, and §15 is explicit that it is also the cheapest place to be
wrong-but-harmless — a song labelled "Part 1 / Part 2" plays exactly as well as
one labelled "Verse / Chorus".

Two jobs here, and the first one is the load-bearing one:

**`bars_from_spans` puts everything on whole bars.** Section lengths must be a
whole number of bars or §13.2's beat axis drifts (see `app/sync.py`), and the
cheapest way to guarantee that is to stop working in loose beats the moment the
chords are quantized. Everything after this function counts bars.

**`segment` finds the repeats.** Structural segmentation proper (`msaf`,
self-similarity over audio features) is a §8-step-2 concern; this works on the
*chords*, which is free, deterministic, and — for the folk/pop material this
feature is for — usually the same answer. It also cannot contradict the chart,
which matters more than being right: a section boundary in the middle of a
repeated progression is visible to the player as a mislabelled bar.

The `energy` hint is how the audio layer gets a vote when it has one: §15 says
naming the repeated high-energy segment `chorus` is defensible, and naming
nothing `custom`/"Part 1" is the honest fallback. **Never** name a section from
lyrics — §2.4 means we must not have lyrics at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import GridSpan

# §15's practical floor: "one section per structural segment, minimum ~4 bars".
MIN_SECTION_BARS = 4

# Progression lengths worth testing, most-likely first. Four bars is the folk
# default; eight covers a long verse; two catches a vamp. Sixteen exists so a
# through-composed section isn't chopped into four identical-looking quarters.
CANDIDATE_UNITS = (4, 8, 2, 16)

# How alike two adjacent blocks must be to be called the same section. §15 wants
# merging: "near-identical adjacent segments read better than a 14-section song".
MERGE_SIMILARITY = 0.75


@dataclass(frozen=True)
class BarChord:
    """One chord's slice of one bar. Beats are **bar-local**, which is exactly
    what ``Bar.chordSpans[]`` wants (§12.1)."""

    root_pc: int
    quality: str
    start_beat: float
    length_beats: float

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
    """

    kind: str = "custom"
    name: str = ""
    bars: list[list[BarChord]] = field(default_factory=list)
    repeats: int = 1
    start_bar: int = 0

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
            ))
    return [bar for bar in bars if bar]


def segment(bars: list[list[BarChord]], *, energy: list[float] | None = None) -> list[Section]:
    """Bars → sections, with repeats collapsed and labels assigned.

    `energy` is one value per bar (a loudness proxy from the audio stage) and is
    optional. With it, the loudest repeated block is called `chorus` and the rest
    `verse`; without it every section is `custom` with a "Part N" name, which
    §15 says is the honest answer when the segmentation can't tell.
    """
    if not bars:
        return []

    unit = _unit_length(bars)
    blocks = _chunk(bars, unit)
    sections = _collapse_repeats(blocks)
    sections = _merge_similar(sections)
    sections = _absorb_runts(sections)
    _assign_positions(sections)
    _assign_labels(sections, energy=energy, bar_beats_hint=unit)
    return sections


# --- repetition -------------------------------------------------------------

def _unit_length(bars: list[list[BarChord]]) -> int:
    """The progression length that best explains the song.

    Scored by self-similarity at a lag: how often bar *i* matches bar *i+p*. The
    shortest candidate that scores near the best wins, because a song whose real
    unit is 4 bars also matches perfectly at 8 and 16, and picking the longest
    would bury the repeat structure the `repeats` field exists to express.
    """
    signatures = [tuple(c.signature for c in bar) for bar in bars]
    n = len(signatures)
    scores: dict[int, float] = {}
    for period in CANDIDATE_UNITS:
        comparisons = n - period
        # At least two full cycles, or the score is measured on too few
        # comparisons to mean anything: a 16-bar period in a 20-bar song is
        # judged on four bars and reaches 1.0 by luck, which would bury the real
        # 4-bar progression under one giant section.
        if comparisons < period:
            continue
        matches = sum(1 for i in range(comparisons) if signatures[i] == signatures[i + period])
        scores[period] = matches / comparisons

    if not scores:
        return min(len(bars), CANDIDATE_UNITS[0])
    best = max(scores.values())
    if best <= 0.0:
        return CANDIDATE_UNITS[0]
    # "Near the best": within a small margin, so a 4-bar unit that matches 90% of
    # the time beats an 8-bar unit that matches 92%.
    for period in sorted(scores):
        if scores[period] >= best - 0.05:
            return period
    return CANDIDATE_UNITS[0]


def _chunk(bars: list[list[BarChord]], unit: int) -> list[list[list[BarChord]]]:
    """Split into blocks of `unit` bars; a short tail joins the previous block
    rather than becoming a section of its own."""
    unit = max(1, unit)
    blocks = [bars[i:i + unit] for i in range(0, len(bars), unit)]
    if len(blocks) > 1 and len(blocks[-1]) < unit:
        tail = blocks.pop()
        blocks[-1] = blocks[-1] + tail
    return blocks


def _collapse_repeats(blocks: list[list[list[BarChord]]]) -> list[Section]:
    """Consecutive identical blocks become one section with `repeats`."""
    sections: list[Section] = []
    for block in blocks:
        signature = tuple(tuple(c.signature for c in bar) for bar in block)
        if sections and sections[-1].signature == signature:
            sections[-1].repeats += 1
        else:
            sections.append(Section(bars=block, repeats=1))
    return sections


def _merge_similar(sections: list[Section]) -> list[Section]:
    """Fold an adjacent section that is *nearly* the same into its neighbour.

    A verse whose second pass changes one chord is one verse played twice, not
    two sections — and the player reads the section rail, so an extra boundary is
    a visible mistake in a way a slightly-wrong chord is not.

    **Merging the boundary must not merge the harmony.** This used to keep the
    earlier block's bars and bump ``repeats``, which replays pass one in place of
    pass two — so at the 0.75 threshold, one differing bar in a four-bar unit was
    silently overwritten by the bar it "nearly" matched. On a self-paced song
    that is defensible smoothing; on a song played against the recording it is
    showing the player a chord that is not being played, and neither ``lint`` nor
    ``lint_sync`` can see it (the chart is perfectly self-consistent and wrong).

    So the two cases are now told apart:

    - **identical** — collapse with ``repeats``, which is lossless and is what
      §15 wants on the wire ("a 4-bar progression played 4× is one section with
      repeats: 4");
    - **merely similar** — merge the *section*, keep both passes' bars explicitly.
      One boundary on the rail, every bar its own harmony.
    """
    if not sections:
        return []
    out = [sections[0]]
    for section in sections[1:]:
        previous = out[-1]
        if len(previous.bars) != len(section.bars):
            out.append(section)
            continue
        if previous.signature == section.signature:
            previous.repeats += section.repeats
        elif _similarity(previous.signature, section.signature) >= MERGE_SIMILARITY:
            previous.bars = _expanded(previous) + _expanded(section)
            previous.repeats = 1
        else:
            out.append(section)
    return out


def _similarity(a: tuple, b: tuple) -> float:
    if not a or len(a) != len(b):
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def _absorb_runts(sections: list[Section]) -> list[Section]:
    """Enforce §15's ~4-bar floor by expanding a runt's bars into its neighbour.

    A two-bar fragment gets appended to the previous section as extra bars (its
    repeats expanded first, so nothing is lost), or prepended to the next when it
    is the first section. A single section shorter than the floor is left alone —
    a two-bar song is a two-bar song.
    """
    if len(sections) <= 1:
        return sections

    out: list[Section] = []
    for section in sections:
        if section.total_bars >= MIN_SECTION_BARS or not out:
            out.append(section)
            continue
        # Expanding the runt's repeats keeps every bar it actually contained.
        out[-1].bars = _expanded(out[-1]) + _expanded(section)
        out[-1].repeats = 1

    if len(out) > 1 and out[0].total_bars < MIN_SECTION_BARS:
        head = out.pop(0)
        out[0].bars = _expanded(head) + _expanded(out[0])
        out[0].repeats = 1
    return out


def _expanded(section: Section) -> list[list[BarChord]]:
    return list(section.bars) * section.repeats


def _assign_positions(sections: list[Section]) -> None:
    cursor = 0
    for section in sections:
        section.start_bar = cursor
        cursor += section.total_bars


# --- labels -----------------------------------------------------------------

def _assign_labels(sections: list[Section], *, energy: list[float] | None,
                   bar_beats_hint: int) -> None:
    """Give each section a `kindRaw` and, when we can't justify one, a name.

    Without an energy hint everything is `custom` + "Part N" (§15's fallback).
    With one, the loudest *repeated* block becomes `chorus`, other repeated
    blocks `verse`, and a short unique block at either end `intro`/`outro`.
    """
    if not sections:
        return

    if energy is None:
        for index, section in enumerate(sections, start=1):
            section.kind = "custom"
            section.name = f"Part {index}"
        return

    means = [_mean_energy(energy, s) for s in sections]
    repeated = [i for i, s in enumerate(sections) if s.repeats > 1]
    chorus_index = max(repeated, key=lambda i: means[i]) if repeated else None

    for index, section in enumerate(sections):
        section.name = ""   # "" = use the kind's own display name
        if index == chorus_index:
            section.kind = "chorus"
        elif index == 0 and section.repeats == 1 and section.total_bars <= bar_beats_hint:
            section.kind = "intro"
        elif index == len(sections) - 1 and section.repeats == 1 and section.total_bars <= bar_beats_hint:
            section.kind = "outro"
        else:
            section.kind = "verse"


def _mean_energy(energy: list[float], section: Section) -> float:
    window = energy[section.start_bar:section.start_bar + section.total_bars]
    return sum(window) / len(window) if window else 0.0
