"""§20.3 — finding the song's form, in a way one wrong chord cannot destroy.

§15's segmentation worked, and its two structural instincts were right: put
everything on whole bars, and prefer the shortest progression that explains the
song. What it could not survive was the engine being wrong. It compared bars by
**exact equality** of `(root_pc, quality)` and it only ever compared **adjacent**
blocks, which has two consequences on real recordings:

- one misheard chord in verse 3 makes verse 3 a different section, so the very
  repetition the layer exists to exploit is destroyed by the noise it exists to
  clean up;
- verse 1 at bar 0 and verse 2 at bar 16 are never compared at all, so the song
  has no idea they are the same music. `repeats` could only ever fire on
  neighbours.

This module fixes both, and the fix is what makes `consensus.py` possible at
all: you cannot vote across the occurrences of a section until you know which
sections *are* occurrences of one another.

Three changes, in order of how much they matter:

**Bars are compared by sound, not by name** (`harmony.similarity`). A verse whose
third bar the engine heard as Am instead of C is 87% the same verse, not a
different one.

**Blocks are matched globally.** Every block is compared with every other and
clustered into repeat groups, so a song's form comes out as A A B A B C rather
than as six unrelated parts. The group is what the player sees on the rail and
what the consensus vote is taken over.

**The block grid is phase-aligned.** A song that opens with a two-bar intro used
to put *every* boundary two bars out — the 4-bar verse got chopped across two
blocks, matched nothing, and the whole form dissolved. The grid's offset is now
chosen by how much repetition it exposes, which is exactly the thing a wrong
offset destroys.

What is deliberately unchanged: §15's ~4-bar floor, the tail-joins-previous
rule, and the honest `Part N` fallback. Labels are player-visible — campfire's
header prints them — and §15 is explicit that being wrong-but-harmless is
cheaper than being confidently wrong. Naming still requires evidence, and
`keyfinder`/`harmony` supply some of it now (see `label`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import harmony
from .structure import MIN_SECTION_BARS, BarChord, Section

# Progression lengths worth testing, most-likely first — §15's list, unchanged.
# Four bars is the folk default; eight covers a long verse; two catches a vamp;
# sixteen exists so a through-composed section isn't chopped into quarters.
CANDIDATE_UNITS = (4, 8, 2, 16)

# How alike two blocks must sound to be called the same section. Measured with
# `harmony.similarity` now, so 0.75 means "three of four bars identical, or four
# of four near-missed" rather than the old "three of four bars byte-equal".
CLUSTER_SIMILARITY = 0.75

# A period only wins if it explains the song this much better than nothing. The
# shortest candidate within this margin of the best wins, because a song whose
# real unit is 4 bars also matches at 8 and 16, and picking the longest would
# bury the repeat structure `repeats` exists to express.
PERIOD_MARGIN = 0.05

# Beats sampled per bar when comparing two bars. Comparing at a fixed grid is
# what lets a bar holding one chord be compared with a bar holding two without
# either representation being privileged.
COMPARE_SUBDIVISION = 4


@dataclass(frozen=True)
class Block:
    """One candidate section-sized run of bars, before it is named."""

    start_bar: int
    bars: tuple[tuple[BarChord, ...], ...]

    @property
    def length(self) -> int:
        return len(self.bars)


@dataclass
class RepeatGroup:
    """A set of blocks the song plays as the same music.

    This is the object the rest of §20 hangs off: `consensus.py` votes over
    `occurrences`, `compile` emits one pattern per group, and the player's rail
    reads its label. `canonical` is filled in by the consensus pass — until then
    it is simply the first occurrence.
    """

    label: str
    length_bars: int
    occurrences: list[int] = field(default_factory=list)
    canonical: list[list[BarChord]] = field(default_factory=list)
    # Mean pairwise similarity of the occurrences, 0…1. How much of a repeat
    # this group really is, which is what decides whether it is worth voting on.
    cohesion: float = 1.0
    # Filled by `consensus.apply`; carried here so the sidecar can report it.
    rewritten_bars: int = 0
    contested_bars: tuple[int, ...] = ()

    @property
    def is_repeat(self) -> bool:
        return len(self.occurrences) > 1


# --- comparing bars ---------------------------------------------------------

def sampled(bar: list[BarChord] | tuple[BarChord, ...], bar_beats: float
            ) -> tuple[tuple[int, str] | None, ...]:
    """A bar as a fixed-length list of "which chord is sounding here".

    Sampling rather than comparing chord lists directly is what makes a bar
    holding `C` and a bar holding `C C` the same bar, and what lets a bar with a
    mid-bar change be compared with one without. `None` marks a slot no chord
    covers, which only happens on a malformed bar.
    """
    cells = max(1, int(round(bar_beats * COMPARE_SUBDIVISION)))
    step = bar_beats / cells
    out: list[tuple[int, str] | None] = []
    for index in range(cells):
        at = (index + 0.5) * step
        found = None
        for chord in bar:
            if chord.start_beat <= at < chord.start_beat + chord.length_beats:
                found = (chord.root_pc, chord.quality)
                break
        out.append(found)
    return tuple(out)


def bar_similarity(a: list[BarChord] | tuple[BarChord, ...],
                   b: list[BarChord] | tuple[BarChord, ...], bar_beats: float) -> float:
    """How alike two bars sound, in 0…1.

    Mean harmonic similarity across the sampled grid, so a bar that differs only
    in its second half scores about 0.5 and a bar heard as the relative minor
    scores 0.5 rather than 0. That gradient is the whole point: exact equality
    made one wrong chord as damaging as a completely different section.
    """
    left, right = sampled(a, bar_beats), sampled(b, bar_beats)
    if not left or len(left) != len(right):
        return 0.0
    total = 0.0
    for x, y in zip(left, right):
        if x is None or y is None:
            total += 1.0 if x is y else 0.0
        else:
            total += harmony.similarity(x, y)
    return total / len(left)


def block_similarity(a: Block, b: Block, bar_beats: float) -> float:
    """Mean bar-by-bar similarity. Blocks of different lengths score 0 —
    a 4-bar verse and a 6-bar verse are not two passes of one thing."""
    if a.length != b.length or a.length == 0:
        return 0.0
    return sum(bar_similarity(x, y, bar_beats) for x, y in zip(a.bars, b.bars)) / a.length


def _signature(block: Block) -> tuple:
    """Exact chord identity, which still matters alongside the fuzzy measure:
    only an identical repeat may be collapsed with `repeats`, because that
    encoding replays one pass in place of the other. A merely *similar* repeat
    has to keep both passes' bars."""
    return tuple(tuple((c.root_pc, c.quality) for c in bar) for bar in block.bars)


# --- period and phase -------------------------------------------------------

def period(bars: list[list[BarChord]], bar_beats: float) -> int:
    """The progression length that best explains the song.

    §15's rule with §20's metric: self-similarity at a lag, scored with
    `bar_similarity` instead of equality, and the shortest candidate within
    `PERIOD_MARGIN` of the best wins.
    """
    n = len(bars)
    scores: dict[int, float] = {}
    for candidate in CANDIDATE_UNITS:
        comparisons = n - candidate
        # At least two full cycles or the score means nothing: a 16-bar period
        # in a 20-bar song is judged on four bars and reaches 1.0 by luck.
        if comparisons < candidate:
            continue
        scores[candidate] = sum(
            bar_similarity(bars[i], bars[i + candidate], bar_beats) for i in range(comparisons)
        ) / comparisons

    if not scores:
        return min(len(bars), CANDIDATE_UNITS[0]) or CANDIDATE_UNITS[0]
    best = max(scores.values())
    if best <= 0.0:
        return CANDIDATE_UNITS[0]
    for candidate in sorted(scores):
        if scores[candidate] >= best - PERIOD_MARGIN:
            return candidate
    return CANDIDATE_UNITS[0]


def _chunk(bars: list[list[BarChord]], unit: int, phase: int) -> list[Block]:
    """Split into blocks of `unit` bars, starting the grid at `phase`.

    The bars before `phase` become a block of their own — that is the intro the
    phase exists to separate — and a short tail joins the previous block rather
    than becoming a section of its own (§15, unchanged).
    """
    unit = max(1, unit)
    edges = list(range(phase, len(bars), unit))
    if phase:
        edges.insert(0, 0)
    blocks: list[Block] = []
    for i, start in enumerate(edges):
        end = edges[i + 1] if i + 1 < len(edges) else len(bars)
        if end > start:
            blocks.append(Block(start_bar=start, bars=tuple(tuple(b) for b in bars[start:end])))
    if len(blocks) > 1 and blocks[-1].length < unit:
        tail = blocks.pop()
        merged = blocks[-1]
        blocks[-1] = Block(start_bar=merged.start_bar, bars=merged.bars + tail.bars)
    return blocks


def _repetition_score(blocks: list[Block], bar_beats: float) -> float:
    """How much repetition a block layout exposes.

    Mean over blocks of the best match each one finds elsewhere. This is the
    objective the phase search maximizes, and it is the right one because a
    misaligned grid destroys exactly this: chop a repeating 4-bar verse two bars
    late and every block becomes half of one verse and half of the next, which
    matches nothing.
    """
    if len(blocks) < 2:
        return 0.0
    total = 0.0
    for i, block in enumerate(blocks):
        total += max(
            (block_similarity(block, other, bar_beats)
             for j, other in enumerate(blocks) if i != j),
            default=0.0,
        )
    return total / len(blocks)


def _layout(bars: list[list[BarChord]], unit: int, bar_beats: float) -> list[Block]:
    """Blocks of `unit` bars, at the phase that exposes the most repetition.

    Ties resolve to the smallest phase, so a song with no intro keeps the
    obvious grid and the result is deterministic (§16.5 wants byte-stable
    output; a phase chosen by dict order would not be).
    """
    best_blocks = _chunk(bars, unit, 0)
    best_score = _repetition_score(best_blocks, bar_beats)
    for phase in range(1, min(unit, len(bars))):
        blocks = _chunk(bars, unit, phase)
        score = _repetition_score(blocks, bar_beats)
        # Strictly better, so phase 0 wins every tie.
        if score > best_score + 1e-9:
            best_blocks, best_score = blocks, score
    return best_blocks


# --- clustering -------------------------------------------------------------

def cluster(blocks: list[Block], bar_beats: float) -> tuple[list[RepeatGroup], list[int]]:
    """Blocks → repeat groups, plus which group each block belongs to.

    Greedy and single-pass, in song order: a block joins the first group whose
    representative it matches, or starts a new one. Deterministic by
    construction, which §16.5 requires — and in song order specifically, so
    group A is always the music the song opens with.
    """
    groups: list[RepeatGroup] = []
    assignment: list[int] = []
    representatives: list[Block] = []

    for block in blocks:
        found = None
        for index, representative in enumerate(representatives):
            if block_similarity(block, representative, bar_beats) >= CLUSTER_SIMILARITY:
                found = index
                break
        if found is None:
            found = len(groups)
            groups.append(RepeatGroup(
                label=_label_for(found), length_bars=block.length,
                canonical=[list(bar) for bar in block.bars],
            ))
            representatives.append(block)
        groups[found].occurrences.append(block.start_bar)
        assignment.append(found)

    for group, representative in zip(groups, representatives):
        members = [b for b, a in zip(blocks, assignment) if groups[a] is group]
        group.cohesion = round(_cohesion(members, bar_beats), 3)
    return groups, assignment


def _cohesion(members: list[Block], bar_beats: float) -> float:
    if len(members) < 2:
        return 1.0
    scores = [block_similarity(a, b, bar_beats)
              for i, a in enumerate(members) for b in members[i + 1:]]
    return sum(scores) / len(scores) if scores else 1.0


def _label_for(index: int) -> str:
    """A, B, … Z, then AA. Rehearsal letters, which is what a musician would
    write on the chart."""
    letters = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return letters


# --- assembling sections ----------------------------------------------------

def detect(bars: list[list[BarChord]], *, bar_beats: float = 4.0,
           energy: list[float] | None = None) -> tuple[list[Section], list[RepeatGroup]]:
    """Bars → the song's sections and the repeat groups behind them.

    The sections are what goes on the wire; the groups are what `consensus.py`
    votes over and what makes two distant sections know they are the same music.
    """
    if not bars:
        return [], []

    unit = period(bars, bar_beats)
    blocks = _layout(bars, unit, bar_beats)
    groups, assignment = cluster(blocks, bar_beats)

    sections = _sections_from(blocks, assignment, groups)
    sections = _absorb_runts(sections)
    _assign_positions(sections)
    _assign_labels(sections, groups, energy=energy, unit=unit)
    return sections, groups


def segment(bars: list[list[BarChord]], *, energy: list[float] | None = None,
            bar_beats: float = 4.0) -> list[Section]:
    """`detect`, for callers that only want the sections (§15's old entry point)."""
    return detect(bars, bar_beats=bar_beats, energy=energy)[0]


def _sections_from(blocks: list[Block], assignment: list[int],
                   groups: list[RepeatGroup]) -> list[Section]:
    """Consecutive blocks of one group become one section.

    **Identical** consecutive blocks collapse with `repeats`, which is lossless
    and what §15 wants on the wire. Consecutive blocks that are merely *similar*
    are merged into one section with **both passes' bars kept explicitly** —
    collapsing those would replay pass one in place of pass two, which on a song
    played against a recording means showing the player a chord that is not
    being played. `lint` cannot see that, because the chart is perfectly
    self-consistent and wrong; `structure._merge_similar` learned it first and
    the rule is carried forward here unchanged.
    """
    sections: list[Section] = []
    for block, group_index in zip(blocks, assignment):
        group = groups[group_index]
        if sections and sections[-1].group == group.label:
            # Compared by *signature* — root and quality — and deliberately not
            # by the bars themselves, which now carry a per-chord confidence
            # that differs between two passes of music the engine heard
            # identically. Comparing the dataclasses would make `repeats` fire
            # only when two occurrences happened to be equally believed, which
            # on a real recording is never.
            if sections[-1].signature == _signature(block):
                sections[-1].repeats += 1
            else:
                sections[-1].bars = _expanded(sections[-1]) + [list(b) for b in block.bars]
                sections[-1].repeats = 1
        else:
            sections.append(Section(bars=[list(b) for b in block.bars], repeats=1,
                                    group=group.label))
    return sections


def _expanded(section: Section) -> list[list[BarChord]]:
    return [list(bar) for bar in section.bars] * section.repeats


def _absorb_runts(sections: list[Section]) -> list[Section]:
    """§15's ~4-bar floor. A runt's bars are expanded into a neighbour rather
    than dropped, so no bar of the song is ever lost."""
    if len(sections) <= 1:
        return sections
    out: list[Section] = []
    for section in sections:
        if section.total_bars >= MIN_SECTION_BARS or not out:
            out.append(section)
            continue
        out[-1].bars = _expanded(out[-1]) + _expanded(section)
        out[-1].repeats = 1
    if len(out) > 1 and out[0].total_bars < MIN_SECTION_BARS:
        head = out.pop(0)
        out[0].bars = _expanded(head) + _expanded(out[0])
        out[0].repeats = 1
        out[0].group = head.group
    return out


def _assign_positions(sections: list[Section]) -> None:
    cursor = 0
    for section in sections:
        section.start_bar = cursor
        cursor += section.total_bars


# --- naming -----------------------------------------------------------------

def _assign_labels(sections: list[Section], groups: list[RepeatGroup], *,
                   energy: list[float] | None, unit: int) -> None:
    """Give each section a `kindRaw`, and a name when we can't justify a kind.

    Without an energy hint everything is `custom` + "Part N" — §15's fallback,
    and still the honest answer, because "which repeated block is the chorus" is
    a question about *loudness and density* that the chords alone cannot settle.
    What §20 adds is that the N is now the **group's** number rather than the
    section's, so the same music carries the same name wherever it returns; the
    player reading the rail can see that part 1 has come back.

    With an energy hint the loudest repeated group is the chorus, other repeated
    groups are verses, and a short unique block at either end is the
    intro/outro.
    """
    if not sections:
        return

    order: dict[str, int] = {}
    for section in sections:
        order.setdefault(section.group, len(order) + 1)

    if energy is None:
        for section in sections:
            section.kind = "custom"
            section.name = f"Part {order.get(section.group, 1)}"
        return

    means = {s.group: _mean_energy(energy, s) for s in sections}
    repeated = {g.label for g in groups if g.is_repeat}
    # A group can also be "repeated" by having been merged into one section with
    # several passes, which is what happens when its occurrences were adjacent.
    repeated |= {s.group for s in sections if s.repeats > 1}
    chorus = max(repeated, key=lambda g: (means.get(g, 0.0), g), default=None)

    for index, section in enumerate(sections):
        section.name = ""   # "" = use the kind's own display name
        if section.group == chorus:
            section.kind = "chorus"
        elif index == 0 and section.repeats == 1 and section.total_bars <= unit:
            section.kind = "intro"
        elif index == len(sections) - 1 and section.repeats == 1 and section.total_bars <= unit:
            section.kind = "outro"
        else:
            section.kind = "verse"


def _mean_energy(energy: list[float], section: Section) -> float:
    window = energy[section.start_bar:section.start_bar + section.total_bars]
    return sum(window) / len(window) if window else 0.0
