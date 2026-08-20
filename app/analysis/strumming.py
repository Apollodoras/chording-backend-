"""§14 — strumming patterns: what is recoverable from a mix, and what is invented.

The app *requires* a pattern (a section with no resolvable pattern is silently
dropped, §12.3), so this always produces one. What it must never do is present
the invented half as if it were measured, so the split is kept explicit here and
in every emitted pattern's `tags`:

| Dimension | Recoverable | How |
|---|---|---|
| onset positions | yes | fold this section's onsets onto one bar of the grid |
| subdivision (8ths vs 16ths) | yes | quantize onsets modulo the bar; keep the coarsest grid they sit on |
| accent | roughly | onset strength against the bar's own mean |
| **band (bass/chordal)** | yes | which frequency band the attack arrived in |
| **direction (down/up)** | **no — convention** | the alternating-hand rule, below |
| mute / percussive | not in a full mix | **never emitted** |

**The band is the one dimension here that is instrument-neutral *and* measured**
(§14.1), and it is worth saying why it earns a place beside direction, which is
neither. You cannot hear which way a hand moved, but you can absolutely hear
whether the bass moved on its own — a root on the beat and a chord off it are
separated by an octave and a half, and a band split finds them. So the extraction
says *what was struck*, in bands, and leaves *who struck it* to whichever
instrument the player picked: a guitar reads a low-only stroke as a bass note
(boom-chick), a piano reads it as the left hand. Neither reading is taken here,
and that is deliberate — one analysis has to serve both, because the catalog is
shared and a song is analyzed once.

**Direction is a convention, not a measurement.** You cannot hear which way a
hand moved in a mixed recording. §14 states the rule as "an onset on a beat is a
downstroke, an onset on the '&' (or the second/fourth 16th) is an upstroke" — two
halves of one idea, and on a 16th grid they only agree under the reading taken
here: **strokes alternate from each beat**, so within a beat the even
subdivisions are down and the odd ones are up. On an 8th grid that gives D on the
beat and U on the "&"; on a 16th grid it gives D-U-D-U, with the 'e' and 'a' as
ups — exactly §14's parenthetical. It is also *correct* far more often than not,
because that is how the instrument is physically played.

That reading is a rule about **grid cells**, and it holds only where the grid is
a *duple* division of the beat — there the hand crosses the strings once per cell
whether or not the cell is struck, which is precisely why the "&" of a bar of
16ths is a downstroke. On a **triple** grid there is no such pendulum: three
cells cannot be walked by a hand alternating at a fixed rate, so cell parity made
a shuffle — the beat and the "let", the commonest triple feel there is — come out
Down-Down, which no one has ever played or taught. Under a triple subdivision the
strokes therefore alternate over what is actually *sounded*, restarting downward
on each beat. `direction_for` is the cell rule; `directions_for` is the one the
extraction calls, and it picks between them.

**Don't over-fit.** Campfire draws the pattern as direction triangles under the
bar and the player strums through it. A 16-onset syncopated transcription of a
strummed acoustic is less playable — and less true to the song — than the
D-DU-UD-U everyone actually plays, so support thresholds here are deliberately
strict and the fallback is deliberately boring: a plain quarter-note downstroke
bar that plays beats a confident pattern that's wrong.

That was the stated intent for a long time before the thresholds delivered it.
Support is a share of *bars*, and on a full mix nearly every cell is supported —
the kit plays in every bar — so "keep everything over the threshold" kept
fifteen of sixteen cells on a recording whose guitar plays six. Three rules now
stand between the grid and the pattern, in this order:

1. **contrast** — a cell must reach a fraction of the bar's strongest cell, which
   is what tells a hand from a hi-hat when support cannot;
2. **a budget** — at most two strokes per beat, spent on the strongest and most
   metrically important cells, so a saturated bar reads as the skeleton
   underneath it;
3. **a vocabulary** — an extraction that comes within a cell of a strum people
   actually play is snapped onto it, tagged `snapped-to-idiom`, the same
   measure-then-snap discipline `vocabulary.SNAP_TO` follows for chords.

The user's acceptance criterion for all of this is worth keeping in the file:
*if a pattern doesn't repeat, it isn't a pattern.* The half of that this module
can enforce is here; the half about how many patterns a song gets to have is in
`model._patterns`.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import NamedTuple

from ..payload import PATTERN_PREFIX, PatternPayload, Stroke, derived_uuid
from .axis import position_in
from .types import FULL, LOW, MID, Onset

log = logging.getLogger("chords.strumming")

# Subdivisions worth testing: quarters, eighths, triplets, sixteenths (cells per
# quarter-note beat). Ordered coarsest-first so a tie resolves toward the simpler
# pattern, per "don't over-fit".
SUBDIVISIONS = (1, 2, 3, 4)

# A grid position is kept only if it carries an onset in at least this share of
# the section's bars. High on purpose: an onset that happens in a third of the
# bars is a fill, and putting a fill in the repeating pattern makes every bar
# wrong instead of one bar right.
SUPPORT_THRESHOLD = 0.5

# ...and it must also hold up against the bar's *strongest* cell. Support alone
# has no opinion about a bar where everything is supported, and on a full mix
# that is the normal case: `LibrosaOnsetDetector` fires on the drum kit, a hi-hat
# playing 8ths is present in every bar, and so more than half the bars carry an
# onset near nearly every 16th cell. Measured on a clean 120 bpm grid, a 6-stroke
# D-DU-UD-U with a kit behind it came out as fifteen of the sixteen cells — the
# literal opposite of what this module's docstring promises.
#
# A groove is contrast: which cells are struck, against which are not. Compared
# with the bar's *mean* that test is self-defeating — six struck cells out of
# eight drag the mean up until the six can no longer clear it — so the reference
# is the peak, and a cell under this fraction of it is the kit, not the hand.
#
# Measured on **loudness**, not on `prominence`. The two axes answer different
# questions and combining them here charged a cell twice for the same fact:
# `SUPPORT_THRESHOLD` has already ruled on how *often* a cell is struck, and
# `prominence` folds that same support back in as a multiplier, so a stroke
# played in half the bars arrived at the contrast test with half the score before
# anything about its loudness was considered.
#
# That is not a rounding error, it is the audit's open question about §14 and it
# has a specimen: `folk-kit-human` varies which syncopation it plays, so the "&"
# of 2 lands in half the bars and the "&" of 4 in the other half. Both clear
# support — 0.5 is the threshold and they sit on it — and both were then cut by
# contrast for being *occasional*, leaving `0 1 2.5 3`: the metronome, with the
# groove averaged out of it. The complaint that started this was that the
# patterns are not rhythmic, and this is one of the mechanisms that made them so.
#
# Loudness alone still separates the case contrast exists for, because a hi-hat
# is not occasional — it is *quiet*, and quiet in every bar. Measured: the hat
# reaches 0.36 of the bar's loudest cell and a real strum played half the time
# reaches about 1.0.
CONTRAST_RATIO = 0.45

# ...but a cell struck in **every** bar is exempt from it, and the two halves of
# that sentence have to be read together with the onset detector.
#
# Contrast exists to answer "everything is supported, so support says nothing",
# and that was true while onsets came off the full mix — a hi-hat is present in
# every bar, so it earns support 1.0 without a hand going anywhere near it. Under
# `HarmonicOnsetDetector` it is no longer true: on the kit specimens the drum-only
# cells come back with support **0.00**, because the kit is not in the harmonic
# component at all. Support means what it says again, and "the player struck this
# in every single bar" is not something a ratio should be allowed to overrule.
#
# It was overruling it, and this is the measurement that forced the exemption:
# with the drums gone, the two *upstrokes* of D-DU-UD-U scored prominence 0.613
# and 0.653 against a peak of 1.479 — under the 0.666 the ratio demands, by 0.05
# and by 0.01. An upstroke is quieter than a downstroke; that is what makes it an
# upstroke. Separating the drums widens the bar's dynamic range, and a rule
# calibrated on the flatter full-mix distribution then cuts exactly the strokes
# that carry the groove — leaving D-D--D- and calling it a pattern.
#
# So the relief is deliberately narrow, and its condition is this comment's own
# first sentence turned around: **support may overrule contrast only where
# support discriminates.** On a grid where every cell is struck, support has no
# opinion and contrast is the only thing that can speak — the full-mix case,
# unchanged. On a grid with *empty* cells in it, support has already told us
# which ones the player struck, and asking those to prove themselves a second
# time on loudness is asking the wrong question.
#
# **Loudness cannot make this distinction and the sparsity of the grid can**,
# which is the whole reason the condition is shaped this way. Measured on the
# specimens, as ratios of the bar's loudest cell:
#
#     a hi-hat, full mix, every bar          0.359
#     a real upstroke played in every bar    0.414, 0.442
#     a real upstroke played in half of them 0.335, 0.414
#
# Those overlap, so no contrast threshold exists that keeps the third row and
# drops the first — a fact worth stating plainly, because the obvious fix is to
# nudge `CONTRAST_RATIO` and there is no value of it that works. On grid sparsity
# the same cases are not close at all: the kit fills every cell of the grid it is
# on (0% empty) and the guitar leaves a quarter of them empty (25%), because
# leaving cells empty is what playing a rhythm *is*. An eighth of the grid — one
# cell in eight — is the floor, comfortably under the 25% the real grooves show
# and comfortably over the 0% a kit leaves.
#
# On a sparse grid `SUPPORT_THRESHOLD` is then the only gate, and deliberately so:
# once support is known to discriminate, "the player struck this in half the
# bars" is the same *kind* of evidence as "in all of them", and it is the
# half-the-bars case that carries a groove. A human varies which syncopation they
# play; `folk-kit-human` is that specimen, and holding it to a higher bar left it
# reading `0 1 2.5 3` — the metronome underneath the groove, which is the
# complaint this module was reported for.
SPARSE_GRID_SHARE = 0.125

# And a hard ceiling, in strokes per quarter-note beat — eight in 4/4. Nobody
# strums sixteen times a bar on the material this service accepts, and a pattern
# that says so is unplayable however well-supported each cell was.
MAX_STROKES_PER_BEAT = 2

# How close (in beats) an onset must sit to a grid cell to count as landing on it.
TOLERANCE_BEATS = 0.12

# How much *average* quantization error a coarser grid may cost before the finer
# one is worth taking (see `choose_subdivision`). 0.05 beats is 25 ms at 120 bpm
# — under the tolerance above, so a grid is never refined for timing jitter, and
# small enough that a real subdivision (which costs a quarter-beat per onset it
# cannot place) always clears it.
SUBDIVISION_MARGIN_BEATS = 0.05

# And how much better a **non-nesting rival** has to fit before the feel changes.
# Relative, not in beats, because the two comparisons are not the same question:
# a finer grid that nests is asking "is there more detail here", where an
# absolute floor is right; a rival is asking "is this song in threes or in twos",
# where the honest test is whether it explains the timing *substantially* better.
#
# It has to be relative because an absolute floor cannot separate the two cases
# that matter, and both are real:
#
#   a genuine 16th feel  (0, .75, 1, 1.75, 2, 3) — only two of its six onsets are
#       off the triplet grid, so triplets score 0.028 against 16ths' 0.000. An
#       absolute margin of 0.05 says "close enough" and prints triplets.
#   a duple groove read late — every onset detected a shade after it began, so
#       eighths score 0.070 and triplets 0.060 by luck. A bare "better wins"
#       hands the song to triplets over a *thousandth* of a beat.
#
# Halving is comfortably clear of both: 0.000 is less than half of 0.028 so 16ths
# still win, 0.060 is not less than half of 0.070 so the duple groove survives,
# and a real triplet feel (0.000 against the eighth grid's 0.167) wins by a mile.
SUBDIVISION_RIVAL_SHARE = 0.5

# Louder than this multiple of the bar's mean onset strength reads as an accent.
ACCENT_RATIO = 1.35

# Below this many strokes-worth of agreement the extraction is not trusted and
# the quarter-note fallback is used instead.
MIN_STROKES = 2

# ...unless the one stroke is played this reliably. A chord struck on the downbeat
# of nine bars in ten is a real (if sparse) pattern — slow ballads are played that
# way — and replacing it with the four-downstroke fallback puts *more* strokes in
# the bar than the recording has.
SOLID_SUPPORT = 0.9

DOWN, UP = "down", "up"

# What share of a cell's onsets have to agree that a band was struck before the
# cell claims that band was there (§14.1).
#
# It gates **presence**, not exclusivity, which is the opposite of how it reads
# and is why the first value here was wrong. Raising it does not make the labels
# more cautious — it makes it harder for *both* bands to be claimed, so the first
# thing a high threshold destroys is `FULL`, and a plain strum starts coming back
# as chord-only. Measured across the specimens, as the labels the bar emits:
#
#     majority   oom-pah              folk strum
#     0.50       full mid low mid     one stroke mid, rest full
#     0.67       low  mid low mid     one stroke mid, rest full
#     0.75       low  mid low mid     three strokes mid
#
# Two thirds is where the oom-pah's own downbeat stops being masked by the
# previous bar still ringing under it, and it is still well short of where a
# strum starts losing its bass. The stray `mid` both of the lower values leave on
# a strummed bar is not fixed here at all — see `_hands_apart`, which is the rule
# that actually answers it.
BAND_MAJORITY = 2.0 / 3.0

# Tags every emitted pattern carries, so nobody downstream — or in six months —
# mistakes the direction assignment for detection.
CONVENTION_TAGS = ["yt", "extracted", "directions-by-convention"]

# ...and the tag a snapped pattern carries as well, for the same reason: the
# strokes it names were chosen from the library below, not measured off this
# recording. Exactly the discipline `vocabulary.SNAP_TO` follows on the chord
# side, and named the same way on purpose.
SNAPPED_TAG = "snapped-to-idiom"

class Idiom(NamedTuple):
    """One strum from the library, and optionally how it is fingered.

    `directions` exists because the pendulum rule below is a rule about *grid
    cells*, and a named idiom is allowed to know better than the grid it happens
    to land on. "6/8 in two" is the specimen: its two strokes are the dotted-
    quarter pulses of a bar counted in two, both of them main beats and both
    played downward, but they sit on cells 0 and 3 of a 2-per-beat grid, and the
    pendulum reads cell 3 as an offbeat and hands back an upstroke. Nothing in
    the grid can see the difference — only the name can.

    Left None for every entry the pendulum already fingers correctly, which is
    most of them: an idiom that is a plain subdivision of the beat *is* the case
    the pendulum was written for, and duplicating its answer here would be one
    more thing to keep in step.
    """

    name: str
    positions: tuple[float, ...]
    directions: tuple[str, ...] | None = None


# The strums people actually play, in bar-local quarter-note beats, keyed by the
# bar's length. Small and deliberately unambitious: this is a *vocabulary*, not a
# generator, and its job is to catch an extraction that came within one cell of
# something idiomatic — not to have an entry for every groove in the world.
IDIOMS: dict[float, tuple[Idiom, ...]] = {
    4.0: (
        Idiom("half notes", (0.0, 2.0)),
        Idiom("quarters", (0.0, 1.0, 2.0, 3.0)),
        Idiom("the campfire pattern", (0.0, 1.0, 1.5, 2.5, 3.0, 3.5)),
        Idiom("offbeats", (0.5, 1.5, 2.5, 3.5)),
        Idiom("eighths", (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)),
    ),
    3.0: (
        Idiom("dotted half", (0.0,)),
        Idiom("waltz quarters", (0.0, 1.0, 2.0)),
        Idiom("waltz eighths", (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)),
        # Both strokes are the bar's own pulses, so both are downstrokes — see
        # `Idiom.directions`.
        Idiom("6/8 in two", (0.0, 1.5), (DOWN, DOWN)),
    ),
    2.0: (
        Idiom("halves", (0.0,)),
        Idiom("quarters", (0.0, 1.0)),
        Idiom("eighths", (0.0, 0.5, 1.0, 1.5)),
    ),
}

# How alike an extraction and a library entry must be before the entry is taken
# instead. High: a 16th feel of (0, .75, 1, 1.75, 2, 3) is two thirds of the way
# to plain quarters by this measure and it is **not** plain quarters, so anything
# looser would flatten real playing into the nearest cliché. What clears it is an
# extraction that missed or gained a single cell of something idiomatic, which is
# the case worth rescuing.
SNAP_SIMILARITY = 0.7

# How far apart two stroke positions may be and still be the same stroke, when
# an extraction is matched against the library.
SNAP_TOLERANCE_BEATS = 0.05

# How much support a cell the library wants to **add** must already have.
#
# The snap's own docstring promises it is "a correction and never an invention",
# and the rule had no way to keep that promise: it took the library entry whole,
# including positions the recording showed *nothing* at. On a groove with a
# deliberate hole in it — nothing on beat 3, which is a very ordinary thing to
# play — the nearest idiom is the campfire pattern with beat 3 in it, and the
# player got handed a stroke they had pointedly not played.
#
# The argument for adding a cell is real, but it is narrower than the rule was:
# an extraction one cell short of an idiom is *probably* that idiom with one
# stroke that fell under `SUPPORT_THRESHOLD`. That sentence is a claim about
# evidence, so it can be checked — a stroke that nearly cleared the bar left
# support behind, and a stroke nobody played left none. Half the threshold, so
# "nearly made it" is the whole of what qualifies.
#
# Positions the library *drops* need no evidence, and removing the gate there
# would be the same mistake pointing the other way: the extraction saying less
# than the library is exactly the case where the library should not be trusted
# over the recording.
SNAP_EVIDENCE_SUPPORT = SUPPORT_THRESHOLD / 2


class FoldedOnset(NamedTuple):
    """One onset laid onto the pattern bar, with the bar it came from.

    The bar identity is not decoration: support is "in how many *bars* was this
    stroke played", and counting onsets instead lets one bar with a flam or a
    doubled drum hit stand in for two bars of evidence.
    """

    bar: int
    position: float     # bar-local, in quarter-note beats
    strength: float
    band: str = FULL


class _Cell(NamedTuple):
    """One position on the pattern grid, with everything scored about it.

    `prominence` is support scaled by how loud the cell is against the bar's own
    mean onset — the number `_with_contrast` thresholds on. Support alone cannot
    tell a strum from a hi-hat, because on a full mix both are present in every
    bar; loudness can, and it is the same evidence `ACCENT_RATIO` already trusts.
    """

    position: float
    support: float
    strength: float
    prominence: float
    # Which band(s) this cell was struck in, across the bars that struck it — the
    # majority verdict of its onsets (`_band_of`). `FULL` for a cell whose
    # onsets disagree, which is the honest answer and also the harmless one.
    band: str = FULL


@dataclass(frozen=True)
class ExtractedPattern:
    pattern: PatternPayload
    # 0…1. Carried into `videoSync.patternConfidence` so the client can say the
    # strumming is a guess when it is one (§14).
    confidence: float
    # True when the quarter-note fallback was used — no usable onset support.
    is_fallback: bool = False


# Where `t_ms` falls on a beat axis, interpolating. Defined in `axis.py` so the
# chart, the anchors and the strumming extractor all read one implementation —
# three private copies of "which beat is this?" is what put the chart out of
# phase with its own recording in the first place.
beat_position = position_in


def fold_onsets(onsets: list[Onset], axis, *, bar_beats: float,
                first_beat: float, last_beat: float) -> list[FoldedOnset]:
    """Onsets inside a beat range → `FoldedOnset`s on one bar of the grid.

    "Folding" is the whole trick: every bar of a section is laid on top of every
    other, so a stroke played in all eight bars shows up as eight onsets at the
    same bar-local position and a one-off fill shows up as one.

    `axis` is a `BeatAxis` (a plain list of beat times is also accepted, which is
    what the unit tests hand it). Passing the axis matters: the bar-local
    position is taken modulo the bar, so folding against a beat list whose
    origin differs from the chart's would put every stroke on the wrong side of
    the beat.

    **A downbeat stroke is rarely late and often early.** Taken modulo the bar,
    a hand that arrives 20 ms *ahead* of the "one" folds to ~3.98 rather than
    ~0.0 — the far end of the bar, supporting nothing. Nobody plays there; they
    played the downbeat of the *next* bar and got there first. So an onset within
    tolerance of the top of the bar is rolled forward onto that downbeat, which
    is both where it belongs on the grid and which bar it is evidence for. The
    bar index is global (it counts from the song's beat 0, not from
    `first_beat`), so pooling several occurrences of the same section keeps their
    bars distinct.
    """
    locate = axis.position_at if hasattr(axis, "position_at") else \
        (lambda t: position_in(axis, t))
    origin_bar = int(round(first_beat / bar_beats))
    # A stroke anticipating the downbeat that *ends* the range belongs to the
    # next section, not this one — it is evidence for a bar we are not folding.
    bar_limit = origin_bar + int(round((last_beat - first_beat) / bar_beats))
    folded: list[FoldedOnset] = []
    for onset in onsets:
        position = locate(onset.t_ms)
        if position < first_beat - TOLERANCE_BEATS or position >= last_beat:
            continue
        offset = position - first_beat
        bar = origin_bar + int(offset // bar_beats)
        local = offset % bar_beats
        if local > bar_beats - TOLERANCE_BEATS:
            local -= bar_beats          # a small negative: just shy of the "one"
            bar += 1
        if bar >= bar_limit:
            continue
        folded.append(FoldedOnset(bar=bar, position=local, strength=onset.strength,
                                  band=onset.band))
    return folded


def choose_subdivision(positions: list[float]) -> int:
    """The coarsest grid the onsets sit on, scored by mean quantization error.

    Each grid is scored by how far, on average, an onset lies from its nearest
    cell — and coarsest-first wins, but only where "coarser" actually means
    "less detail about the same grid":

    - Against a grid it **nests inside** (every 8th is also a 16th) the coarser
      one wins unless the finer one is better by more than
      `SUBDIVISION_MARGIN_BEATS`. This is §14's "don't over-fit": without the
      bias, timing jitter alone would always favour the finest grid and invent
      syncopation out of it.
    - Against a grid it **doesn't** nest inside, the better fit wins — but it has
      to be *substantially* better (`SUBDIVISION_RIVAL_SHARE`). Triplets and 16ths
      are different feels, not two resolutions of one grid, and 3 is only
      nominally coarser than 4, so the ordering here cannot express a preference
      for the coarser one. What it can express is a refusal to change the song's
      feel over nothing: an attack is detected slightly after it begins, so onsets
      carry a systematic late bias, and with a bare "better wins" a bias of 0.07
      beats was enough for the triplet grid to beat the eighth grid by a
      *thousandth* and take the whole song with it. `direction_for` reads this
      number, so every "&" in every bar changed hand. Observed on the noisy
      specimen, where it turned a clean D-DU-UD-U into `0 1 1.667 2.667 3 3.667`.

    Mean error is also what makes this survive a full mix. Scored by *share of
    onsets explained*, one consistent hi-hat 16th per bar — a sixth of the
    onsets, and present in every drummed recording — was enough to carry the vote
    to 16ths, and since `direction_for` reads the grid, every "&" in the bar
    flipped from an upstroke to a downstroke. Averaged instead, that ghost costs
    a quarter-beat spread across the bar's strokes and changes nothing; two of
    them per bar, which is a real 16th feel, still wins.
    """
    if not positions:
        return 1
    scores: dict[int, float] = {}
    for subdivision in SUBDIVISIONS:
        cell = 1.0 / subdivision
        error = sum(abs(p / cell - round(p / cell)) * cell for p in positions)
        scores[subdivision] = error / len(positions)

    for subdivision in SUBDIVISIONS:
        finer = [s for s in SUBDIVISIONS if s > subdivision]
        nested = [s for s in finer if s % subdivision == 0]
        rivals = [s for s in finer if s % subdivision != 0]
        if any(scores[subdivision] > scores[s] + SUBDIVISION_MARGIN_BEATS for s in nested):
            continue
        if any(scores[s] < scores[subdivision] * SUBDIVISION_RIVAL_SHARE for s in rivals):
            continue
        return subdivision
    return SUBDIVISIONS[-1]


def direction_for(position: float, subdivision: int) -> str:
    """The alternating-hand rule — see the module docstring.

    Strokes alternate from each beat: even subdivisions within the beat are down,
    odd ones are up. This is a **convention**, not a measurement.

    Correct on its own only for a **duple** grid, where the hand really does
    cross the strings once per cell whether or not the cell is struck. On a
    triple grid it is not, and `directions_for` — which is what `extract` calls —
    handles that case instead of this one.
    """
    within_beat = position - int(position)
    cell = round(within_beat * subdivision)
    return DOWN if cell % 2 == 0 else UP


def directions_for(positions: list[float], subdivision: int) -> list[str]:
    """Directions for a whole bar of strokes, which is the only scale at which
    the question is answerable.

    **A duple grid keeps the pendulum**, and it has to. There the hand crosses
    the strings once per cell regardless of whether that cell is struck, so a
    stroke's direction is a property of *where it is* and its neighbours cannot
    change it. That is exactly what makes the "&" of a bar of 16ths a
    **downstroke** — 1 e & a is D-U-D-U — and any rule that alternated over the
    sounded strokes instead would call it an upstroke and be wrong. This is also
    why `extract` re-reads the subdivision off the strokes it kept before asking:
    the pendulum's answer is only as good as the grid it is swinging on.

    **A triple grid has no such pendulum**, and this is the bug that was here.
    Under `subdivision == 3` the cells of a beat are 0, 1, 2, and parity makes
    cell 2 a downstroke — so a **shuffle**, the beat and the "let" and nothing in
    between, came out Down-Down. That is the single most common triple feel there
    is (blues, 12/8 ballads, anything swung) and it is played, taught and
    notated Down-Up; the hand cannot make two downstrokes in a row at tempo
    without a wasted pass through the strings. The reason parity fails is that a
    triplet's three cells cannot be walked by a hand alternating at a fixed rate
    — one of the three has to be taken in the same direction as its neighbour,
    and which one depends on which cells are *played*.

    So on a triple grid the strokes alternate over what is actually sounded,
    restarting downward on every beat. That gives D-U for the shuffle, D-U-D for
    a full triplet (which is what parity gave too, and is right), and D for a
    beat struck alone — and the restart is what keeps the hand coming back down
    on the beat, which is the part of the convention nobody argues about.
    """
    if subdivision % 2 == 0:
        return [direction_for(position, subdivision) for position in positions]

    out = [DOWN] * len(positions)
    struck_in_beat: dict[int, int] = {}
    for index in sorted(range(len(positions)), key=lambda i: positions[i]):
        # Via the grid cell rather than the raw float, so a position carrying a
        # rounding hair below the beat is still counted into that beat.
        beat = int(round(positions[index] * subdivision)) // subdivision
        rank = struck_in_beat.get(beat, 0)
        struck_in_beat[beat] = rank + 1
        out[index] = DOWN if rank % 2 == 0 else UP
    return out


def extract(onsets_in_bar: list[FoldedOnset], *, bar_beats: float, bars: int,
            tempo: int, name: str, time_signature: str = "4/4") -> ExtractedPattern:
    """Folded onsets → one bar of strokes (§14's method, end to end).

    `bars` is how many bars were folded together, and it is what turns a raw
    count into support: a stroke played in three bars means everything across
    three bars and nothing across sixteen.

    `bar_beats` is quarter-note beats (the app's unit — ``n × 4/d``), so 6/8
    arrives here as 3.0 and its strokes land where 3/4's would. `time_signature`
    rides through unchanged so the emitted pattern names the meter the song
    actually uses.
    """
    if bars <= 0 or bar_beats <= 0:
        return fallback(bar_beats=max(1.0, bar_beats), tempo=tempo, name=name,
                        time_signature=time_signature)

    folded = [FoldedOnset(*onset) for onset in onsets_in_bar]
    subdivision = choose_subdivision([onset.position for onset in folded])
    cell = 1.0 / subdivision
    cells = int(round(bar_beats * subdivision))

    # The bar's own loudness scale, so `prominence` below is a ratio and not a
    # level. A cell struck in every bar by something the mix barely registers —
    # a hi-hat — is not a stroke of the pattern however reliable it is.
    loudness = [o.strength for o in folded]
    mean_onset_strength = sum(loudness) / len(loudness) if loudness else 1.0

    scored: list[_Cell] = []
    for index in range(cells):
        position = index * cell
        matched = [o for o in folded
                   if _distance_in_bar(o.position, position, bar_beats) <= TOLERANCE_BEATS]
        # Support is a share of *bars*, not of onsets: a cell struck twice in half
        # the bars is half-supported, however many onsets that is.
        support = min(1.0, len({o.bar for o in matched}) / bars)
        strength = sum(o.strength for o in matched) / len(matched) if matched else 0.0
        relative = strength / mean_onset_strength if mean_onset_strength else 0.0
        scored.append(_Cell(position=position, support=support, strength=strength,
                            prominence=support * relative,
                            band=_band_of(matched)))

    # Before anything reads a band: if this bar has no two hands in it, it has no
    # bands either (`_hands_apart`). Applied here rather than at the end so that
    # `_with_contrast`'s per-band reference collapses back to the bar-wide peak
    # too — an unsplit bar takes the identical path it took before §14.1.
    if not _hands_apart(scored):
        scored = [c._replace(band=FULL) for c in scored]

    kept = _with_contrast(scored)
    if not _is_a_pattern(kept):
        # Too thin, too noisy, or too *uniform* to be a pattern. A boring bar
        # that plays is worth more than a confident one that's wrong.
        return fallback(bar_beats=bar_beats, tempo=tempo, name=name,
                        time_signature=time_signature)

    kept, snapped, fingering = snap_to_idiom(kept, bar_beats=bar_beats, scored=scored)

    # Re-read the grid off the strokes that survived, because that is the grid
    # the pattern is actually on and `direction_for` reads directions from it.
    # A bar whose 16ths were all struck chooses a 16th grid, keeps the eight-note
    # skeleton, and would otherwise have every "&" come out a downstroke — the
    # same failure the one-ghost-hi-hat guard exists to prevent, arriving from
    # the other side.
    subdivision = choose_subdivision([c.position for c in kept])

    # The library's fingering when the snap supplied one, the convention
    # otherwise — and only when it fits, so a stale entry can never hand out
    # more or fewer directions than there are strokes.
    if fingering and len(fingering) == len(kept):
        directions = list(fingering)
    else:
        directions = directions_for([c.position for c in kept], subdivision)

    mean_strength = sum(c.strength for c in kept) / len(kept)
    strokes = [
        Stroke(
            beat=round(c.position, 4),
            direction=direction,
            accent=c.strength >= mean_strength * ACCENT_RATIO,
            # Only ever `low` or `mid` on the wire. `FULL` is both the commonest
            # answer and the one that asks the client for nothing, so it travels
            # as an absent field — which also keeps every song that is simply
            # strummed byte-identical to what this module emitted before §14.1.
            band=c.band if c.band in (LOW, MID) else None,
        )
        for c, direction in zip(kept, directions)
    ]
    # Confidence is the mean support of the strokes we kept: how reliably this
    # bar's shape actually repeated across the section.
    confidence = sum(c.support for c in kept) / len(kept)
    pattern = _pattern(strokes, time_signature=time_signature, tempo=tempo, name=name,
                       extra_tags=[SNAPPED_TAG] if snapped else None)
    return ExtractedPattern(pattern=pattern, confidence=round(confidence, 3))


def _with_contrast(scored: list[_Cell]) -> list[_Cell]:
    """The cells that are actually the pattern, out of every cell of the grid.

    Two rules on top of `SUPPORT_THRESHOLD`, and they answer different halves of
    the same complaint:

    **Contrast.** A cell has to be struck at `CONTRAST_RATIO` of the loudness of
    the bar's strongest cell. That is what separates the hand from the kit when
    both are present in every bar and support therefore says nothing: on the
    measured case the six strums sit near the peak and the hi-hat's own cells sit
    at a third of it, so the six survive and the wall of eighths does not.

    On *loudness* and not on `prominence`, which is the same number with support
    folded in — see `CONTRAST_RATIO` for the specimen that forced the
    distinction. Support has already had its say one line above, and charging a
    cell for it twice is what turned a groove whose syncopation varies from bar
    to bar into the metronome underneath it.

    **A ceiling.** Whatever survives, at most `MAX_STROKES_PER_BEAT` per beat,
    ranked by prominence and then by metrical weight — so when the budget bites
    it spends what it has on the downbeat and the beats before the offbeats.
    This is the rule that catches the genuinely saturated bar, where every cell
    is struck *equally* and contrast has nothing to discriminate with: sixteen
    equal cells become the eight-note skeleton underneath them, which is what
    the recording is actually playing.

    **Relief**, and only on a grid where support means something. When
    `SPARSE_GRID_SHARE` of the cells are *empty*, support has already selected
    which cells the player struck and contrast steps aside. On a grid where
    everything is struck — the full mix, where a hi-hat guarantees it — nothing
    is relieved and this is the rule it always was.

    See `SPARSE_GRID_SHARE` for why the condition is sparsity and not loudness:
    a hi-hat and a real upstroke overlap on loudness, so there is no threshold
    that separates them, and they do not overlap on this at all.

    **Contrast is measured within a band** (§14.1), and it has to be, because the
    rule above compares a cell against the loudest thing in the bar and a bass
    note is quieter than the chord over it. Measured on the synthetic oom-pah — a
    root octave on beats 1 and 3, a right hand on 2 and 4 — judging every cell
    against one bar-wide peak deleted *both bass strokes* and emitted the two
    chord stabs as the whole pattern. That is the correct answer to "which cells
    are loudest" and the wrong answer to "what was played", and it would have
    handed a piano an accompaniment with no left hand in it.

    So a cell's reference is the loudest cell **that was struck in the same band
    as it**: a bass note competes with the other bass notes, a chord stab with
    the other chord stabs. On a recording where nothing is labelled — which is
    every strummed song, and every specimen this rule was tuned against — all
    cells share one band and the reference is the bar's peak exactly as before,
    so this generalizes the rule without moving it.

    The cost is real and worth naming: a band with only one cell in it is its own
    reference and therefore always clears contrast, leaving `SUPPORT_THRESHOLD`
    as the only gate on it. That is the right trade here — support still demands
    the player struck it in half the bars, and the alternative is the measured
    failure above — but it is why this reads the band a cell *has* rather than
    trying to infer one.
    """
    if not scored:
        return []
    empty = sum(1 for c in scored if c.support < SUPPORT_THRESHOLD)
    sparse = empty >= max(1, round(SPARSE_GRID_SHARE * len(scored)))
    peaks: dict[str, float] = {}
    for cell in scored:
        peaks[cell.band] = max(peaks.get(cell.band, 0.0), cell.strength)
    kept = [c for c in scored
            if c.support >= SUPPORT_THRESHOLD
            and (sparse or c.strength >= peaks[c.band] * CONTRAST_RATIO)]

    budget = max(1, int(round(MAX_STROKES_PER_BEAT * _beats_in(scored))))
    if len(kept) > budget:
        kept = sorted(sorted(kept, key=lambda c: c.position),
                      key=lambda c: (-c.prominence, -_metrical_weight(c.position)))[:budget]
    return sorted(kept, key=lambda c: c.position)


def _hands_apart(scored: list[_Cell]) -> bool:
    """Whether this bar genuinely has two hands in it (§14.1).

    **A `MID` label means nothing without a `LOW` somewhere to mean it against.**
    A song whose bass is simply quiet, or buried, produces bar after bar of
    chord-band-only strokes — and read literally that says "this accompaniment
    has no left hand", which is not something any recording has ever meant. It is
    the one systematic error the band split makes on ordinary strummed material,
    and no threshold fixes it, because on those bars the measurement is not
    ambiguous — it is confidently reporting a fact about the mix rather than
    about the playing.

    So a bar's labels survive only when the bar actually splits: something was
    struck in the bass band alone, and something else was not. Otherwise every
    cell is `FULL` and the rest of this module runs exactly as it did before
    §14.1 existed — same contrast reference, same strokes, same content-addressed
    id. That last property is worth the rule on its own: it means a strummed song
    cannot change because bands were added.

    Read off the *supported* cells only. An empty cell has no onsets and so no
    band, and letting a grid position nobody struck vote on whether the player
    has two hands would decide it by how fine the grid happened to be.
    """
    supported = [c for c in scored if c.support >= SUPPORT_THRESHOLD]
    return (any(c.band == LOW for c in supported)
            and any(c.band != LOW for c in supported))


def _band_of(matched: list[FoldedOnset]) -> str:
    """The band(s) a cell was struck in, from the onsets that landed on it.

    Each onset is read as the *set* of bands it moved — `LOW` and `MID` name one,
    `FULL` names both — and a band belongs to the cell when a majority of its
    onsets moved it. That framing is what makes the three labels combine sanely:
    a cell struck by a bass note in every bar and by the whole band in one is
    still a bass stroke, and a cell where half the bars caught the bass and half
    did not is `FULL`, because it genuinely is both.

    Bars, not onsets, would be the stricter denominator — `support` counts that
    way for a reason. It is not used here because a cell's onsets have already
    been reduced to one per bar in every case that matters, and because the
    failure this would guard against (one bar with a flam voting twice) changes a
    label rather than the existence of a stroke.
    """
    if not matched:
        return FULL
    low = sum(1 for o in matched if o.band in (LOW, FULL))
    mid = sum(1 for o in matched if o.band in (MID, FULL))
    threshold = len(matched) * BAND_MAJORITY
    has_low, has_mid = low > threshold, mid > threshold
    if has_low and has_mid:
        return FULL
    if has_low:
        return LOW
    if has_mid:
        return MID
    return FULL


def _beats_in(scored: list[_Cell]) -> float:
    """The bar's length in quarter-note beats, read back off the cell grid."""
    if len(scored) < 2:
        return float(len(scored))
    cell = scored[1].position - scored[0].position
    return len(scored) * cell if cell > 0 else float(len(scored))


def _metrical_weight(position: float) -> float:
    """How strong a beat this is — the tie-break when the budget bites.

    A downbeat outranks a beat, a beat outranks an "&", an "&" outranks an "e"
    or an "a". This is the one place a musical prior is applied rather than
    measured, and it only ever chooses *between* cells the evidence already
    likes equally.
    """
    if abs(position) < 1e-6:
        return 3.0
    if abs(position - round(position)) < 1e-6:
        return 2.0
    if abs(position * 2 - round(position * 2)) < 1e-6:
        return 1.0
    return 0.0


def snap_to_idiom(kept: list[_Cell], *, bar_beats: float,
                  scored: list[_Cell] | None = None,
                  ) -> tuple[list[_Cell], bool, tuple[str, ...] | None]:
    """Pull an extraction onto the nearest strum people actually play (§14).

    The direct answer to "patterns should be more musical". The chord side
    already has this shape in `vocabulary.SNAP_TO`: measure first, then move the
    measurement onto what the repertoire actually contains — and only when the
    two are already close, so the snap is a *correction* and never an invention.
    An extraction that lands a cell short of D-DU-UD-U almost certainly is
    D-DU-UD-U with one stroke under the support threshold; an extraction that is
    nothing like anything in the library is left exactly as measured.

    A snapped position keeps the support and strength of the extracted cell it
    came from, so accents stay a measurement even where the positions are not,
    and a position the library added but the recording did not play inherits the
    bar's mean — it is a stroke we are asserting, and its confidence should say
    so rather than borrow a neighbour's.
    """
    library = IDIOMS.get(round(bar_beats, 3))
    if not library or not kept:
        return kept, False, None

    positions = [c.position for c in kept]
    best_idiom, best_score = None, 0.0
    for idiom in library:
        candidate = idiom.positions
        if not _supportable(candidate, kept, scored or kept, bar_beats):
            continue
        score = stroke_similarity(positions, candidate)
        # Ties go to the sparser entry: "don't over-fit" applies to the library
        # exactly as it applies to the grid.
        if (score, -len(candidate)) > (best_score,
                                       -len(best_idiom.positions) if best_idiom else 0):
            best_idiom, best_score = idiom, score

    if best_idiom is None or best_score < SNAP_SIMILARITY:
        return kept, False, None
    best_positions = best_idiom.positions
    if best_score >= 1.0 and len(positions) == len(best_positions):
        # Already idiomatic, so the positions need no correction — but the
        # fingering still can, since nothing before this point has consulted the
        # library and the pendulum has already had its say.
        return kept, False, best_idiom.directions
    log.info("strum snapped to %s (%.2f similar): %s → %s", best_idiom.name, best_score,
             [round(p, 3) for p in positions], list(best_positions))

    mean_support = sum(c.support for c in kept) / len(kept)
    mean_strength = sum(c.strength for c in kept) / len(kept)
    snapped: list[_Cell] = []
    for position in best_positions:
        source = min(kept, key=lambda c: abs(c.position - position))
        near = abs(source.position - position) <= SNAP_TOLERANCE_BEATS
        snapped.append(_Cell(
            position=position,
            support=source.support if near else mean_support,
            strength=source.strength if near else mean_strength,
            prominence=source.prominence if near else 0.0,
            # A stroke the library *added* was not played, so nothing was measured
            # about its register and it must not inherit a neighbour's: `FULL` is
            # the "no claim" value. A stroke that merely moved by a cell keeps what
            # its own onsets said.
            band=source.band if near else FULL,
        ))
    return snapped, True, best_idiom.directions


def _supportable(candidate: tuple[float, ...], kept: list[_Cell],
                 scored: list[_Cell], bar_beats: float) -> bool:
    """Whether every stroke this library entry would *add* has evidence for it.

    A position the extraction already holds needs nothing. A position it does not
    has to have left `SNAP_EVIDENCE_SUPPORT` behind on the grid — the recording
    saying "there was something here" even though it did not clear
    `SUPPORT_THRESHOLD`. See that constant for why the entry is rejected outright
    rather than trimmed: the library's entries are grooves people play, and half
    of one is not.
    """
    for position in candidate:
        if any(_distance_in_bar(position, cell.position, bar_beats) <= SNAP_TOLERANCE_BEATS
               for cell in kept):
            continue
        near = [cell for cell in scored
                if _distance_in_bar(position, cell.position, bar_beats) <= SNAP_TOLERANCE_BEATS]
        if not near or max(cell.support for cell in near) < SNAP_EVIDENCE_SUPPORT:
            return False
    return True


def stroke_similarity(a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]) -> float:
    """Jaccard over two stroke sets, matched with a tolerance rather than by
    equality — a cell at 1.4999 and one at 1.5 are the same stroke."""
    if not a or not b:
        return 0.0
    remaining = list(b)
    hits = 0
    for position in a:
        match = next((q for q in remaining
                      if abs(q - position) <= SNAP_TOLERANCE_BEATS), None)
        if match is not None:
            remaining.remove(match)
            hits += 1
    return hits / (len(a) + len(b) - hits)


def _distance_in_bar(position: float, cell: float, bar_beats: float) -> float:
    """How far an onset is from a cell, **around** the bar rather than along it.

    The bar is a loop, not a line: the last cell and the downbeat are adjacent,
    one grid step apart. Measured linearly, an onset a hair ahead of the "one"
    sits a whole bar away from it and no cell can claim it — which is exactly how
    the beat-1 stroke used to vanish from patterns extracted off real recordings
    while the extraction still reported full confidence in the rest.
    """
    direct = abs(position - cell)
    return min(direct, bar_beats - direct)


def _is_a_pattern(kept: list[_Cell]) -> bool:
    """Whether the kept cells are worth emitting instead of the fallback."""
    if len(kept) >= MIN_STROKES:
        return True
    # One stroke is a pattern only when it is played almost every bar; anything
    # less is a coincidence with nothing else in the bar to corroborate it.
    return len(kept) == 1 and kept[0].support >= SOLID_SUPPORT


def fallback(*, bar_beats: float, tempo: int, name: str,
             time_signature: str = "4/4") -> ExtractedPattern:
    """A plain downstroke on every beat — §14's honest floor."""
    strokes = [Stroke(beat=float(b), direction=DOWN) for b in range(int(bar_beats))]
    pattern = _pattern(strokes, time_signature=time_signature, tempo=tempo,
                       name=name, extra_tags=["fallback"])
    return ExtractedPattern(pattern=pattern, confidence=0.0, is_fallback=True)


def _pattern(strokes: list[Stroke], *, time_signature: str, tempo: int, name: str,
             extra_tags: list[str] | None = None) -> PatternPayload:
    """Wrap strokes in a `PatternPayload` with a **content-addressed id**.

    Hashing the meter and the strokes means an unchanged groove compiles to an
    unchanged id, which is §12.5's "keep embedded pattern ids stable when their
    strokes are unchanged" — held by construction rather than by bookkeeping.
    Mo's `grid.py` does the same thing for the same reason.
    """
    # The register joins the fingerprint **only when a stroke has one**, so every
    # pattern that predates §14.1 — and every ordinary strummed song, which emits
    # no registers at all — hashes to exactly the id it hashed to before. Two
    # grooves that differ only in which hand plays them are genuinely different
    # grooves, though, and content-addressing has to see that or §12.5's promise
    # ("an unchanged groove keeps its id") quietly becomes "a groove we happened
    # to hash the same".
    body = f"{time_signature}|" + ";".join(
        f"{s.beat:.4f},{s.direction},{int(s.accent)}"
        + (f",{s.band}" if s.band else "")
        for s in strokes
    )
    fingerprint = hashlib.sha1(body.encode("utf-8")).hexdigest()[:12]
    pattern_id = f"{PATTERN_PREFIX}{fingerprint}"
    # The strokes' own ids are derived too, so the whole pattern is a function of
    # its content — two runs over the same recording produce the same bytes.
    identified = [
        s.model_copy(update={"id": derived_uuid(pattern_id, "stroke", index)})
        for index, s in enumerate(strokes)
    ]
    return PatternPayload(
        id=pattern_id,
        name=name,
        timeSignature=time_signature,
        tempo=tempo,
        strokes=identified,
        tags=CONVENTION_TAGS + (extra_tags or []),
    )
