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

§15's honest `Part N` fallback is unchanged. Labels are player-visible —
campfire's header prints them — and §15 is explicit that being wrong-but-harmless
is cheaper than being confidently wrong. Naming still requires evidence, and
`keyfinder`/`harmony` supply some of it now (see `_assign_labels`).

**Two of §15's rules did have to move**, and both for the same reason: they were
written for a pipeline with no clustering step in it, so both quietly destroyed
the groups everything above depends on. The tail no longer joins the previous
block (`_chunk`), and the ~4-bar floor yields rather than flatten a collapsed
repeat (`_absorb_runts`). Each function carries the case that forced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache

from . import harmony
from .structure import MIN_SECTION_BARS, BarChord, Section

# Progression lengths worth testing, most-likely first. Four bars is the folk
# default; eight covers a long verse; **twelve is the blues**, which is core
# campfire repertoire and was missing — without it the twelve-bar form is chopped
# at period 4 and its I/IV/V phrases scatter across three groups that are not
# really sections; two catches a vamp; six covers a six-bar phrase and the 12/8
# blues; sixteen exists so a through-composed section isn't chopped into quarters.
#
# Order matters only for the fallbacks below (`CANDIDATE_UNITS[0]`) — the winner
# is chosen shortest-first from the scores, not from this order.
CANDIDATE_UNITS = (4, 8, 12, 2, 6, 16)

# How alike two blocks must sound to be called the same section. Measured with
# `harmony.similarity` now, so 0.75 means "three of four bars identical, or four
# of four near-missed" rather than the old "three of four bars byte-equal".
CLUSTER_SIMILARITY = 0.75

# A period only wins if it explains the song this much better than nothing. The
# shortest candidate within this margin of the best wins, because a song whose
# real unit is 4 bars also matches at 8 and 16, and picking the longest would
# bury the repeat structure `repeats` exists to express.
#
# **And only a shortest candidate that divides the best-scoring one.** That is the
# whole content of the argument above: 4 beating 16 is the same loop written more
# compactly, and 4 divides 16. A shorter candidate that does *not* divide the best
# is a different hypothesis about the song, not a tidier statement of the same
# one, and it has to win on the score rather than on being shorter. Country Roads
# scores 0.497 at 8 bars and 0.467 at 6, and without the divisibility test it was
# charted in six-bar sections: eighteen sections for a song with two progressions
# in it, and not one of its blocks the length of one of its phrases.
#
# Score-neutral on the chart corpus, and worth saying so — root and form both come
# out identical to four decimal places, because Country Roads' remaining error is
# in its beat grid rather than in its blocks. What changes is that the one song
# this rule touches is now cut into phrase-length pieces (eight bars of a doubled
# grid, which is its four-bar phrase) instead of six-bar ones, which is the answer
# a musician would give and the one `repeats` can express.
#
# **The width is measured, and it was too narrow at 0.05.** Swept over the chart
# corpus (root / form): 0.05 → 0.857 / 0.713, then a flat band from 0.08 to 0.15
# at 0.854–0.855 / **0.747**, then 0.18 → 0.853 / 0.732. 0.12 is the middle of
# the band rather than its edge.
#
# Three and a half points of form for three thousandths of root, and the trade is
# the right way round for what this constant decides: the shorter unit is what
# lets `_sections_from` collapse a repeat into `repeats`, so widening the margin
# is not "accept a worse period", it is "prefer the compact statement of the same
# period more often". The songs that move are the ones whose section is two
# passes of a loop — which in this repertoire is most of them.
#
# Worth noting what the 2026-08-18 audit expected here and did not get. It read
# the shortest-within-margin rule as the cause of fragmented sections and
# recommended preferring the **longest**; measured, that is a no-op on all ten
# songs (the divisibility test above already prevents the case it describes), and
# moving the margin the *other* way is what pays. The period search is not where
# a song's sections fragment — `folded()` and the divisibility rule got there
# first.
PERIOD_MARGIN = 0.12

# Beats sampled per bar when comparing two bars. Comparing at a fixed grid is
# what lets a bar holding one chord be compared with a bar holding two without
# either representation being privileged.
COMPARE_SUBDIVISION = 4

# Whether to cut the song at novelty peaks instead of tiling it with one unit.
# **Off**, and the measurement that settled it is on `novelty_blocks` below.
NOVELTY = False


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

    **Memoized on the bar's identity**, because the same bar is sampled thousands
    of times for one song and the answer cannot change. `_layout` tries every phase
    of the period; each phase scores every block against every other block
    (`_repetition_score`, O(blocks²)); each block comparison samples every bar in
    both blocks — and then `form.detect` runs the whole thing twice, before and
    after the consensus vote (see `model.py`). The product is O(unit² × blocks²)
    samplings of a small fixed set of bars.

    `BarChord` is a frozen dataclass, so a tuple of them is hashable and its hash
    *is* the bar's identity: same chords, same offsets, same answer. Lists are
    coerced — cheap, since a bar holds at most a handful of chords. Bounded rather
    than unbounded: a long song has thousands of distinct bars and this cache
    outlives the analysis, so it is sized to hold one song's worth comfortably and
    then behave like an LRU rather than like a leak.
    """
    return _sampled(tuple(bar), bar_beats)


@lru_cache(maxsize=8192)
def _sampled(bar: tuple[BarChord, ...], bar_beats: float
             ) -> tuple[tuple[int, str] | None, ...]:
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


# --- the structural view ----------------------------------------------------

def folded(bars: list[list[BarChord]]) -> list[list[BarChord]]:
    """The bars as *structure* sees them: every chord reduced to its triad.

    **Where the form is found, colour is noise.** `bar_similarity` scores `Em`
    against `Em7` at 0.75 — a real gradient, and the right answer to "how alike
    do these two sound". It is the wrong answer to "are these two the same bar of
    music", and the difference is not academic: a four-bar loop whose `Em` is
    heard as `Em7` on alternate passes scores **0.9375** at lag 4 and a perfect
    1.0 at lag 8, and `PERIOD_MARGIN` is 0.05. The song loses its own period by
    0.0125 and comes out as an eight-bar section played four times instead of a
    four-bar one played eight.

    That is the failure the user sees as "the chart is twice as long as the
    song", and it is the *same* engine wobble that shows up as "there are more
    chords than the song has" — one cause, two symptoms. Folding here fixes the
    structural half; `lexicon.py` fixes the naming half, and it can only run at
    all once the groups are right, because its evidence is what the *other
    passes of this slot* say.

    Only the period, the phase and the clustering read this. The sections are
    built from the real bars, `_signature` compares the real chords, and the vote
    counts the real readings — because those are the places where the difference
    between `Em` and `Em7` is the thing being decided rather than the thing
    getting in the way.
    """
    return [[replace(chord, quality=harmony.triad(chord.quality)) for chord in bar]
            for bar in bars]


def _restored(blocks: list[Block], bars: list[list[BarChord]]) -> list[Block]:
    """The same block boundaries, over the real bars instead of the folded ones."""
    return [Block(start_bar=block.start_bar,
                  bars=tuple(tuple(bar) for bar in
                             bars[block.start_bar:block.start_bar + block.length]))
            for block in blocks]


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
    # The longest candidate that scored best, so "divides the winner" is asked of
    # the fullest statement of the song's period rather than of whichever of its
    # own multiples happened to tie first.
    winner = max((c for c in scores if scores[c] >= best - 1e-9), default=CANDIDATE_UNITS[0])
    for candidate in sorted(scores):
        if scores[candidate] >= best - PERIOD_MARGIN and winner % candidate == 0:
            return candidate
    return winner


def _chunk(bars: list[list[BarChord]], unit: int, phase: int) -> list[Block]:
    """Split into blocks of `unit` bars, starting the grid at `phase`.

    The bars before `phase` become a block of their own — that is the intro the
    phase exists to separate — and **so do the bars after the last full block**.

    §15's rule was that a short tail joins the previous block. That is right at
    *section* level and wrong here, because clustering happens in between: a
    four-bar verse with a two-bar tag stuck on its end is a six-bar block, and
    `block_similarity` scores blocks of unequal length 0. So the song's last
    verse stopped being an occurrence of the verse — it was excluded from the
    consensus vote, its onsets never pooled into the verse's strum pattern, and
    the rail labelled it as different music. Real songs rarely end on an exact
    multiple of their own period, so this cost the *last section of most songs*.

    The tail is now left standing as its own short block, clusters on its own
    terms, and `_absorb_runts` applies the ~4-bar floor afterwards, at section
    level, where it belongs.
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
    return blocks


# --- boundaries from novelty ------------------------------------------------
#
# §20.3b — the alternative to cutting the song into equal blocks.
#
# `_layout` below tiles the whole song with one unit at one phase, and that model
# cannot express a standard song form: 4-bar intro + 8-bar verse + 16-bar chorus
# has no valid representation in it, so boundaries land mid-section and every
# stage after this one — clustering, the vote, the canon — reasons about slots
# that are half of one section and half of the next.
#
# The standard answer is Foote novelty: build the bar-by-bar self-similarity
# matrix, slide a checkerboard kernel down its diagonal, and read the peaks as
# boundaries. Where the music changes, the kernel sees two blocks that are alike
# within and unlike across, and it spikes.
#
# **It is implemented here, it is measured, and it loses on every axis.** Over the
# ten-song chart corpus, against the fixed grid:
#
#            root    triad    form
#   grid    0.857    0.852   0.713
#   Foote   0.846    0.835   0.610
#
# and the way it loses says why. Creep goes from 1.000 root and four chords to
# 0.949 and **nine**: the song is one eight-bar loop for its whole length, the
# fixed grid at the measured period lands on it exactly, and the novelty curve
# peaks on the chord changes *inside* the loop because at bar resolution that is
# what a checkerboard kernel sees. Foote novelty is a good segmenter of audio
# features, where timbre and density change at a section boundary and not within
# one; run over *harmony* in a repertoire whose sections are the same four chords
# at different volumes, the signal it needs is not there.
#
# So the argument in the audit is right about the fixed grid's limits — a 4-bar
# intro, an 8-bar verse and a 16-bar chorus genuinely have no representation in
# it — and wrong about this being the way out. What would need to change first is
# the feature: novelty over the *energy* curve (which the structure probe already
# produces and which nothing but the chorus label reads today) is the untried
# version, and it is untried rather than rejected.
#
# Left in place, off, and covered by a test that runs it — the measurement is the
# reason not to ship it, and a measurement nobody can re-run is a claim.

# Half-width of the checkerboard kernel, in bars. Eight bars either side is a
# section-scale question; smaller kernels find phrase boundaries, which is what
# the fixed grid already finds.
NOVELTY_KERNEL_BARS = 8

# A peak has to stand this far above the local median novelty to be a boundary.
NOVELTY_PROMINENCE = 0.08


def novelty(bars: list[list[BarChord]], bar_beats: float) -> list[float]:
    """Foote novelty over the bar-level self-similarity matrix, one value per bar.

    High where the music before a bar is unlike the music after it, which is what
    a section boundary *is* — as opposed to a chord change, which is what the
    period search finds.
    """
    n = len(bars)
    if n < 4:
        return [0.0] * n
    similarity = [[0.0] * n for _ in range(n)]
    for i in range(n):
        similarity[i][i] = 1.0
        for j in range(i + 1, n):
            value = bar_similarity(bars[i], bars[j], bar_beats)
            similarity[i][j] = similarity[j][i] = value

    half = min(NOVELTY_KERNEL_BARS, max(2, n // 4))
    out = [0.0] * n
    for centre in range(n):
        lo, hi = centre - half, centre + half
        if lo < 0 or hi > n:
            continue
        same = across = 0.0
        pairs = 0
        for i in range(lo, centre):
            for j in range(lo, centre):
                same += similarity[i][j]
            for j in range(centre, hi):
                across += similarity[i][j]
            pairs += 1
        for i in range(centre, hi):
            for j in range(centre, hi):
                same += similarity[i][j]
        if pairs:
            out[centre] = (same - 2.0 * across) / (2.0 * half * half)
    return out


def novelty_blocks(bars: list[list[BarChord]], bar_beats: float,
                   *, minimum: int = 4) -> list[Block]:
    """Variable-length blocks, cut where the novelty peaks.

    Boundaries no closer together than `minimum` bars, because a section shorter
    than that is a tag or a turnaround (§15's floor, and `_absorb_runts`' one).
    """
    n = len(bars)
    if n < 2 * minimum:
        return _chunk(bars, n, 0)
    curve = novelty(bars, bar_beats)
    ordered = sorted(curve)
    median = ordered[len(ordered) // 2]
    boundaries = [0]
    for index in range(1, n):
        if curve[index] < median + NOVELTY_PROMINENCE:
            continue
        if index - boundaries[-1] < minimum or n - index < minimum:
            continue
        # Local maximum only, so one broad rise produces one boundary.
        if curve[index] < max(curve[max(0, index - 2):index + 3]):
            continue
        boundaries.append(index)
    boundaries.append(n)

    blocks: list[Block] = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end > start:
            blocks.append(Block(start_bar=start,
                                bars=tuple(tuple(b) for b in bars[start:end])))
    return blocks


def _repetition_score(blocks: list[Block], bar_beats: float) -> float:
    """How much repetition a block layout exposes.

    Mean over blocks of the best match each one finds elsewhere. This is the
    objective the phase search maximizes, and it is the right one because a
    misaligned grid destroys exactly this: chop a repeating 4-bar verse two bars
    late and every block becomes half of one verse and half of the next, which
    matches nothing.

    Runt blocks are scored too, at the 0 they inevitably earn — and that is
    deliberate, not an oversight. A runt is a bar the grid failed to explain, so
    the penalty is real information about the offset. Excluding them (tried, and
    reverted) hands the search a way to cheat: it can shift the grid until the
    song's one misheard bar falls into the tail runt, score a perfect 1.0 on
    what is left, and adopt an offset that is wrong everywhere.
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

    **First match rather than best match**, and that is measured rather than
    careless — the obvious improvement was tried and does not pay. Giving each
    block its *best* available match over the ten-song chart corpus moves form by
    +0.012 and root by −0.008, which is the shape of a change that has bought
    nothing. Three Little Birds loses five points of root to it: a block
    reassigned to a better-matching group is then made to agree with that group
    (§21), so a better cluster produces a worse chart. The corpus cannot resolve
    an effect this size in either direction, and first-match is what every
    threshold here was calibrated against.
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
           energy: list[float] | None = None, tonic_pc: int | None = None
           ) -> tuple[list[Section], list[RepeatGroup]]:
    """Bars → the song's sections and the repeat groups behind them.

    The sections are what goes on the wire; the groups are what `consensus.py`
    votes over and what makes two distant sections know they are the same music.

    `tonic_pc` is optional and only ever *adds* a label: without it the half
    cadence that distinguishes a pre-chorus cannot be read, and those sections
    stay verses.
    """
    if not bars:
        return [], []

    # Period, phase and clustering are decided on the triad-folded bars (see
    # `folded`); everything the caller gets back is the real ones.
    structure = folded(bars)
    unit = period(structure, bar_beats)
    if NOVELTY:
        blocks = novelty_blocks(structure, bar_beats)
    else:
        blocks = _layout(structure, unit, bar_beats)
    groups, assignment = cluster(blocks, bar_beats)

    blocks = _restored(blocks, bars)
    for group in groups:
        # `cluster` seeded `canonical` from the folded block that created the
        # group. The vote overwrites it, but until it does this is what the
        # sidecar reports, and it must be chords the song actually contains.
        if group.occurrences:
            start = group.occurrences[0]
            group.canonical = [list(bar) for bar in bars[start:start + group.length_bars]]

    sections = _sections_from(blocks, assignment, groups)
    sections = _absorb_runts(sections, groups)
    _assign_positions(sections)
    _assign_labels(sections, groups, energy=energy, unit=unit, tonic_pc=tonic_pc)
    return sections, groups


def segment(bars: list[list[BarChord]], *, energy: list[float] | None = None,
            bar_beats: float = 4.0, tonic_pc: int | None = None) -> list[Section]:
    """`detect`, for callers that only want the sections (§15's old entry point)."""
    return detect(bars, bar_beats=bar_beats, energy=energy, tonic_pc=tonic_pc)[0]


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


def _absorb_runts(sections: list[Section],
                  groups: list[RepeatGroup] | None = None) -> list[Section]:
    """§15's ~4-bar floor, applied only where it costs nothing.

    A runt's bars are expanded into a neighbour rather than dropped, so no bar of
    the song is ever lost. Two rules make that safe, and the absence of both is
    what turned a two-bar intro into a song with no form at all:

    **The host keeps its own group.** A head merge used to hand the merged
    section the *runt's* group, so `intro + verse ×4` came out as one section
    carrying the intro's letter — and `compile` then attached the intro's strum
    pattern (two bars of evidence, so in practice the fallback) to sixteen bars
    of verse, while the verse group's pooled pattern was computed and never
    referenced by anything.

    **A collapsed repeat is never expanded.** Absorbing into a host with
    `repeats > 1` has to flatten it, which destroys the `repeats: 4` encoding
    §15 wants on the wire, erases the repeat from the player's rail, and — the
    real cost — merges bars belonging to one group into a section belonging to
    another. So when the neighbour is a collapsed repeat the runt simply stays a
    short section of its own. The ~4-bar floor is a §15 *preference*; the
    container has no such rule, and a standalone two-bar intro is a far smaller
    lie than an eighteen-bar section that claims to be one piece of music.

    **And a host that is a clean occurrence of a repeat group is not lengthened.**
    Third rule, from the 2026-08-18 audit. A 2-bar turnaround absorbed into the
    16-bar verse beside it makes an 18-bar section, and 18 bars is not an
    occurrence of a 16-bar group any more: `block_similarity` scores blocks of
    unequal length 0, so that verse stops matching the others, drops out of the
    consensus vote and out of §21's canon, and the repetition the whole layer
    runs on is quietly one pass short. The runt stays where it is instead —
    which is the same trade the rule above already makes, applied to the
    occurrence rather than to the `repeats` encoding.

    **A repeat group specifically**, and that qualifier is measured rather than
    tidy. Protecting every clean host — including the ones whose "group" has a
    single occurrence, where there is no sibling to stop matching — costs Smooth
    Criminal a quarter of its form score (0.750 → 0.500) and gains nothing
    anywhere: a lone section is not made worse by two more bars, and leaving the
    runt standing turns one section into two. With the guard restricted to groups
    that actually repeat, the corpus is unchanged to four decimals and the
    failure above is still closed.
    """
    if len(sections) <= 1:
        return sections
    lengths = {group.label: group.length_bars for group in (groups or [])
               if group.is_repeat}
    out: list[Section] = []
    for section in sections:
        if section.total_bars >= MIN_SECTION_BARS or not out or out[-1].repeats > 1:
            out.append(section)
            continue
        if lengths.get(out[-1].group) == out[-1].total_bars:
            out.append(section)      # the host is a clean occurrence — leave it
            continue
        out[-1].bars = _expanded(out[-1]) + _expanded(section)
        out[-1].repeats = 1
    # The head has no previous section to join, so it merges forwards instead —
    # under the same two rules.
    if (len(out) > 1 and out[0].total_bars < MIN_SECTION_BARS and out[1].repeats == 1
            and lengths.get(out[1].group) != out[1].total_bars):
        head = out.pop(0)
        out[0].bars = _expanded(head) + _expanded(out[0])
        out[0].repeats = 1
    return out


def _assign_positions(sections: list[Section]) -> None:
    cursor = 0
    for section in sections:
        section.start_bar = cursor
        cursor += section.total_bars


# --- naming -----------------------------------------------------------------

def _assign_labels(sections: list[Section], groups: list[RepeatGroup], *,
                   energy: list[float] | None, unit: int,
                   tonic_pc: int | None = None) -> None:
    """Give each section a `kindRaw`, and a name when we can't justify a kind.

    Without an energy hint everything is `custom` + "Part N" — §15's fallback,
    and still the honest answer, because "which repeated block is the chorus" is
    a question about *loudness and density* that the chords alone cannot settle.
    What §20 adds is that the N is now the **group's** number rather than the
    section's, so the same music carries the same name wherever it returns; the
    player reading the rail can see that part 1 has come back.

    With an energy hint the loudest repeated group is the chorus, a unique
    mid-song group is the bridge, a repeated group leaning on the chorus is the
    pre-chorus, a short unique block at either end is the intro/outro, and
    everything else is a verse.

    The last two of those are new, and they are here because the vocabulary was
    unfinished rather than because it was deliberately small: `preChorus` and
    `bridge` are both in the container's `SECTION_KINDS`, and
    `harmony.is_dominant_of` was written for the pre-chorus cue and then called
    by nothing. Each is gated on a structural fact rather than a guess —

    - **bridge**: a group the song plays *once*, in the middle. That is what the
      word means, and calling it `verse` (which is what happened before) is
      confidently wrong about the one section a listener would never confuse.
    - **preChorus**: a repeated non-chorus group that sits immediately before the
      chorus and ends on the key's V — a half cadence leaning on what follows.
      Guarded on the song having *more than one* repeated non-chorus group,
      because a lone verse ending on the dominant to lead into the chorus is the
      most ordinary thing in this repertoire, and without that guard every folk
      verse in the corpus would be relabelled.

    `solo` stays unreachable on purpose. Telling a solo from a verse is a
    question about *timbre* — who is playing the melody — and nothing that
    crosses the audio boundary here (one loudness scalar per hop) can answer it.
    The 2026-08-18 audit noted it as dead (F23), which it is; it is dead because
    the evidence for it does not cross §2.1's audio boundary, and the fix is a
    second scalar out of the structure probe rather than anything in this file.
    `preChorus` is nearly dead for a different and better reason — its guard
    demands two repeated non-chorus groups *and* a half cadence *and* a chorus
    after it, and songs that satisfy all three are genuinely rare.
    """
    if not sections:
        return

    order: dict[str, int] = {}
    for section in sections:
        order.setdefault(section.group, len(order) + 1)

    means = ({s.group: _mean_energy(energy, s) for s in sections}
             if energy is not None else {})
    repeated = {g.label for g in groups if g.is_repeat}
    # A group can also be "repeated" by having been merged into one section with
    # several passes, which is what happens when its occurrences were adjacent.
    repeated |= {s.group for s in sections if s.repeats > 1}

    if not repeated:
        # No repetition anywhere: there is no chorus to find and no verse to name,
        # so §15's honest fallback stands. This is now the *only* way to reach it
        # — it used to be reached by every song analyzed without a loudness probe.
        for section in sections:
            section.kind = "custom"
            section.name = f"Part {order.get(section.group, 1)}"
        return

    chorus = _chorus(sections, repeated, means)
    # The song's other repeated material — what a pre-chorus has to be one of.
    siblings = repeated - {chorus}

    for index, section in enumerate(sections):
        section.name = ""   # "" = use the kind's own display name
        following = sections[index + 1] if index + 1 < len(sections) else None
        mid_song = 0 < index < len(sections) - 1
        if section.group == chorus:
            section.kind = "chorus"
        elif index == 0 and section.repeats == 1 and section.total_bars <= unit:
            section.kind = "intro"
        elif index == len(sections) - 1 and section.repeats == 1 and section.total_bars <= unit:
            section.kind = "outro"
        elif _leans_on(section, following, chorus, siblings, tonic_pc):
            section.kind = "preChorus"
        elif section.group not in repeated and mid_song:
            section.kind = "bridge"
        else:
            section.kind = "verse"


def _chorus(sections: list[Section], repeated: set[str],
            means: dict[str, float]) -> str | None:
    """Which repeated group is the chorus, from several cheap signals.

    Loudness alone was the whole rule, and it is the right *first* signal —
    a chorus is louder and denser than the verse it follows — but it fails on
    exactly the songs where it is easiest to be confident: a quiet chorus after a
    building verse, a final verse sung at full band, a coda louder than anything.
    Worse, it was the only signal, so a build without a structure probe had no
    labelling at all and every song came out `Part 1`, `Part 2`.

    Three structural signals stand beside it, each cheap and each a different
    argument for the same conclusion:

    - **how often it comes round.** A chorus recurs more than any other group; a
      verse comes back too, but a song has more verse *material* than chorus
      material and the chorus is the part that repeats identically.
    - **whether it is interleaved.** A chorus alternates with something else
      (V C V C). A group whose occurrences are all consecutive is a verse played
      twice, or a vamp.
    - **where it first appears.** Choruses rarely open a song. A group whose first
      occurrence is the first section is far more likely the intro or the verse,
      and this carries the most weight of the three structural terms because it is
      the one that decides `A B A` — where the other two both point at `A`, which
      opens the song, recurs, and is interleaved, and is the verse.

    Scored as a weighted sum rather than a cascade of tie-breaks, so a chorus that
    is only slightly the loudest but plainly the most repeated still wins, and no
    single signal can carry the decision on its own. The label falls back to the
    alphabetically-first repeated group only when everything ties, which keeps the
    output deterministic (§16.5).
    """
    if not repeated:
        return None
    positions: dict[str, list[int]] = {}
    for index, section in enumerate(sections):
        positions.setdefault(section.group, []).append(index)

    loudest = max(means.values(), default=0.0)
    quietest = min(means.values(), default=0.0)
    spread = (loudest - quietest) or 1.0

    def score(group: str) -> tuple[float, str]:
        where = positions.get(group, [])
        passes = sum(sections[i].repeats for i in where)
        interleaved = any(b - a > 1 for a, b in zip(where, where[1:]))
        loudness = ((means[group] - quietest) / spread) if group in means else 0.0
        return (
            1.0 * loudness
            + 0.5 * min(passes, 4) / 4.0
            + 0.3 * float(interleaved)
            - 0.6 * float(bool(where) and where[0] == 0),
            group,
        )

    return max(repeated, key=score)


def _leans_on(section: Section, following: Section | None, chorus: str | None,
              siblings: set[str], tonic_pc: int | None) -> bool:
    """Whether this section is a pre-chorus: repeated, before the chorus, on V."""
    if tonic_pc is None or chorus is None or following is None:
        return False
    if following.group != chorus or section.group == chorus:
        return False
    if len(siblings) < 2 or section.group not in siblings:
        return False
    last_bar = section.bars[-1] if section.bars else []
    if not last_bar:
        return False
    final = last_bar[-1]
    return harmony.is_dominant_of(final.root_pc, final.quality, tonic_pc)


def _mean_energy(energy: list[float], section: Section) -> float:
    window = energy[section.start_bar:section.start_bar + section.total_bars]
    return sum(window) / len(window) if window else 0.0
