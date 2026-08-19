"""§20.8 — the song's own chord vocabulary, and the noise it makes visible.

`consensus.py` cleans a chord up by comparing a bar against **the same bar in
another pass of the same section**. That is the strongest evidence there is, and
it is also the narrowest: it can only speak where the song repeats, where the
passes disagree, and where enough of them agree to carry a vote. Three ordinary
situations fall straight through it:

- the section occurs **twice** — one reading against one reading, which the vote
  is right to refuse (`consensus._vote`, gate 1);
- the section occurs **once** — an intro, a bridge, a tag;
- the engine made the *same* mistake in **every** pass, so there is nothing to
  disagree with. A recognizer's errors are only independent in the way the vote
  needs when the audio differs; a doubled guitar part that confuses it in verse 1
  confuses it identically in verse 3.

What remains after the vote is therefore not a residue. It is the ordinary case,
and it is what the user sees: a song whose chart is Ebm–Db–Ab with an `Ebm7` in
one bar and an `Eb` in another, both of which the recording plays as Ebm.

**This module answers those with a different kind of evidence: the rest of the
song.** A song has a small chord vocabulary — four or five chords, played over
and over — and that vocabulary is a fact about the recording measured over
minutes, not over one bar. So when one brief, doubtful reading of a root
disagrees with what the song plays on that root everywhere else, and the
disagreement is the kind a recognizer makes (a third or a seventh, never a
different root), the song's own answer is better evidence than the frame it came
from.

Two rules, in order, both operating on `GridSpan`s before anything has been cut
into bars:

1. **`absorb_islands`** — an interior span shorter than the chords on either side
   of it, flanked by *the same chord on both sides*, a near-miss of it, and
   believed less. Ebm | Eb | Ebm is one chord with a hole punched in it, and the
   hole is what the player sees as a chord change that isn't there. §5.4's
   `drop_short` already does this for spans below one beat; this is the same
   argument one level up, and it needs the harmonic test because at two beats a
   real passing chord is possible and must survive.

2. **`snap`** — the vocabulary rule proper. Per **root**, the qualities the song
   plays on it are weighed by duration × confidence, and a minority reading is
   pulled onto the dominant one when every one of seven conditions holds (listed
   on `snap` itself). Same root always: the root is the load-bearing fact (§12.2),
   and a rule that moved it would be inventing harmony rather than spelling it.

**What keeps this safe** is the same discipline as §20.4, gate for gate:

- it never crosses harmonic space — `harmony.is_near_miss` bounds every edit, so
  a real IV can never be pulled onto a I;
- it never moves a root;
- it demands a **lopsided** ratio of evidence (`MASS_DOMINANCE`), not a majority,
  because "the song plays Ebm here" and "the song plays Ebm7 in the bridge" are
  both true of a song that has both, and only a landslide distinguishes noise
  from vocabulary;
- it never snaps *away* from the key, and never uses the key on its own
  (`harmony.diatonic_fit`, same standing as in the vote — a tie-break, not a
  filter, because borrowed chords are normal music);
- and it requires the loser to have been **believed less** than the winner, by
  `CONFIDENCE_MARGIN`. That last one is what carries the property this layer is
  judged by, exactly as it does for the vote: **on perfect input every rule here
  is provably a no-op**, because ground truth arrives at a flat confidence of 1.0
  and no span is ever less believed than any other. `bench/run_bench.py`'s
  truth-as-engines run asserts it, and the assertion is a consequence of the
  construction rather than a number that came out well.

What this module deliberately does **not** do is the relative-major/minor
confusion — hearing Gb where the song plays Ebm. Those two share two notes, so
`is_near_miss` admits the pair, and it is a real and common engine error. But it
is a *different root*, and both chords are usually in the vocabulary of the same
song (they are in "The Silence": the verse is on Ebm and the chorus opens on Gb),
so vocabulary mass cannot tell which one belongs in a given bar. Deciding that
needs bar-position evidence — which is `consensus.py`'s job, where the same slot
in another pass is available to compare against. Guessing it from mass would put
a chord nobody played into the chart, which is the one failure this whole layer
exists to avoid.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from ..chords import (
    AUGMENTED,
    DIMINISHED,
    DIMINISHED7,
    DOMINANT7,
    HALF_DIM7,
    MAJOR,
    MAJOR7,
    MINOR,
    MINOR7,
    SUS2,
    SUS4,
)
from . import harmony
from .postprocess import merge
from .types import GridSpan

log = logging.getLogger("chords.vocabulary")

# Which readings may be pulled onto which, and this table is **measured, not
# reasoned**. `harmony.is_near_miss` says whether two chords are close enough
# that a recognizer could slide between them; it does not say which direction the
# engine is more often wrong in, and that is the only question that decides
# whether an edit pays. So the answer comes from the corpus: for every quality
# BTC reports, how often the truth is what it said, versus the plain triad of the
# same root (`bench/run_bench.py --calibration`, 9 real tracks, beats-weighted):
#
#   engine says   as read   as plain triad   root wrong
#   major            0.93             —            0.03
#   minor            0.80             —            0.12
#   dominant7        0.31           0.60           0.01
#   minor7           0.30           0.39           0.30
#   diminished7      0.90           0.00           0.10
#   augmented        0.12           0.00           0.88
#   sus4             0.00           0.30           0.70
#   major7           0.33           0.00           0.67
#   sus2             0.00           1.00           0.00
#
# Read the two middle columns as the wager. A reported `dominant7` is the plain
# major twice as often as it is a seventh, so flattening a doubtful one is a bet
# at 2:1 on — that is the single biggest win available here, and it is the
# opposite of what the near-miss relation alone would suggest (which is that the
# two are simply interchangeable). `minor7` is a narrower win in the same
# direction. `sus2`/`sus4` are never right as read, so flattening them cannot
# lose, though the samples are small (three and eight spans).
#
# And three qualities are **deliberately absent, on the same evidence**:
#
# - `major7` is right a third of the time and is *never* the plain triad — the
#   engine hearing a major 7th is either hearing a real one or hearing the wrong
#   root entirely, and flattening it can only lose. This is the case that made
#   "Let It Be"'s opening Fmaj7 disappear while the rule was still generic.
# - `augmented` is right 12% of the time, and 88% of the time the *root* is wrong,
#   so nothing that keeps the root can help it. Worse, an augmented triad is one
#   pitch-class set with three names (Caug, Eaug, G#aug are the same three
#   notes), so flattening the one the engine chose throws away a chord that was
#   arguably right — which is exactly what it did to "Michelle" five times.
# - `diminished`/`diminished7` are either right 90% of the time (dim7) or a wrong
#   root (dim), and neither is ever the plain triad.
#
# The half-diminished 7th is absent for want of a single example in the corpus,
# which is the honest reason to leave a chord alone.
SNAP_TO: dict[str, tuple[str, ...]] = {
    # The seventh the engine probably did not hear.
    DOMINANT7: (MAJOR,),
    MINOR7: (MINOR,),
    # The suspension it heard instead of the third.
    SUS4: (MAJOR, MINOR),
    SUS2: (MAJOR, MINOR),
    # And the third itself, which is the failure this whole layer was reported
    # for: a mix whose third is buried reads as the parallel major or minor, and
    # `Eb` in a song that plays `Ebm` twenty times is that mistake.
    MAJOR: (MINOR,),
    MINOR: (MAJOR,),
}

# Named so the exclusions are legible at the call site rather than only in the
# table above, and so a future engine change has one place to revisit.
NEVER_SNAPPED = (MAJOR7, AUGMENTED, DIMINISHED, DIMINISHED7, HALF_DIM7)

# How many times more evidence the song's answer needs before a minority reading
# of the same root is treated as noise. Deliberately steep: at 6× a chord played
# in one bar of eight is left alone, and only a reading the song contradicts
# overwhelmingly is moved. Two-thirds — `consensus.SUPPORT`'s standard — is the
# right bar for *counting passes of one section*; it is far too low a bar for
# "this quality is rarer than that one across a whole song", where a genuine
# secondary dominant or a borrowed chord is always in the minority.
MASS_DOMINANCE = 6.0

# And the most of a root's evidence a reading may hold and still be called noise.
# Belt to `MASS_DOMINANCE`'s braces, and the one of the two that survives a song
# with only two chords in it: a root whose two readings split 80/20 is a root the
# song plays two ways.
MINORITY_SHARE = 0.15

# How many separate places in the song may carry a reading before it stops being
# a one-off and starts being the song's language. This is the gate that decides
# the hardest case in the whole module, and it is worth stating what it settled.
#
# "In My Life" is in A, plays A for seventy beats, and touches A7 for nine — four
# brief passes, each one the I7 that leans into the IV. Every duration-based
# measure calls those nine beats noise: they are 4% of the root's evidence, the
# engine hedged at 0.45–0.67 on all of them, and flattening them scores as the
# 2:1 bet `SNAP_TO` describes. They are also the song. Meanwhile "Something"
# reports G7 twice, just as briefly and just as doubtfully, and both times the
# record plays a plain G.
#
# Nothing about the *amount* of evidence separates those two. What separates them
# is how many **occasions** carry the reading: an engine artefact happens once, or
# twice; a chord the arrangement actually contains keeps coming back. Counting
# occasions rather than beats is the same "the song is the evidence" argument the
# rest of this module makes, applied in the unit that answers the question.
#
# Measured over the nine real tracks, counting every edit this module makes: at
# ≤ 2 it corrects six, damages one, and is neutral on four (spans that were wrong
# before and after, so the edit costs nothing). With no limit at all it corrects
# seven, damages **five**, and is neutral on eight. Nearly all of the damage lives
# above the line, and the extra correction below it is one span.
MAX_OCCASIONS = 2

# How much less believed the minority must be. Identical to
# `consensus.CONFIDENCE_MARGIN`, and load-bearing for the same reason: any value
# ≥ 0 makes every rule here a no-op on flat-confidence input.
CONFIDENCE_MARGIN = 0.02

# An island may be at most this share of the chord it interrupts. A span half the
# length of its neighbours is a chord in its own right; a quarter of it is a
# wobble at a boundary.
ISLAND_SHARE = 0.5


@dataclass(frozen=True)
class VocabularyReport:
    """What the consolidation changed — provenance, like `ConsensusReport`.

    Every number here is an edit the player cannot see and neither lint can
    check, which is precisely why it is carried out to the sidecar instead of
    being logged and forgotten.
    """

    snapped_spans: int = 0
    absorbed_islands: int = 0
    # §20.8b's own count, kept apart from `absorbed_islands` because the two make
    # different claims. That one says a chord had a hole punched in its colour;
    # this one says the engine slipped a semitone for a moment, which is a
    # stronger claim about a bigger mistake and the one worth watching.
    absorbed_semitones: int = 0

    @property
    def touched(self) -> bool:
        return bool(self.snapped_spans or self.absorbed_islands
                    or self.absorbed_semitones)


@dataclass(frozen=True)
class Reading:
    """One (root, quality) the song plays, and how much the song stands behind it."""

    beats: float
    belief: float          # duration-weighted mean confidence
    spans: int

    @property
    def mass(self) -> float:
        """Duration × belief — evidence, not airtime.

        A chord heard for eight beats at 0.3 and one heard for three beats at 0.9
        are close to the same amount of evidence, and airtime alone would say the
        first is nearly three times the second.
        """
        return self.beats * self.belief


def profile(spans: list[GridSpan]) -> dict[int, dict[str, Reading]]:
    """root pitch class → quality → what the song plays there.

    The whole track. Not per section: the point of this
    layer is to be able to speak about a bar that has no second occurrence to
    compare against, so its evidence has to come from somewhere other than the
    section.
    """
    beats: dict[tuple[int, str], float] = {}
    weighted: dict[tuple[int, str], float] = {}
    counts: dict[tuple[int, str], int] = {}
    for span in spans:
        if span.length_beats <= 0:
            continue
        key = (span.root_pc % 12, span.quality)
        beats[key] = beats.get(key, 0.0) + span.length_beats
        weighted[key] = weighted.get(key, 0.0) + span.length_beats * span.confidence
        counts[key] = counts.get(key, 0) + 1

    out: dict[int, dict[str, Reading]] = {}
    for (root, quality), total in beats.items():
        out.setdefault(root, {})[quality] = Reading(
            beats=total,
            belief=weighted[(root, quality)] / total if total else 0.0,
            spans=counts[(root, quality)],
        )
    return out


def _dominant(readings: dict[str, Reading], quality: str) -> tuple[str, Reading] | None:
    """The song's answer for this root, if it is a reading `quality` may move to.

    The winner has to be the *most* played quality at that root and an allowed
    destination for this one. Both halves matter: the first is what makes the edit
    the song's own answer rather than a guess, and the second is what keeps it to
    the moves the corpus says are worth making (`SNAP_TO`).
    """
    allowed = SNAP_TO.get(quality)
    if not allowed or not readings:
        return None
    ranked = sorted(readings.items(), key=lambda item: (-item[1].mass, item[0]))
    winner, evidence = ranked[0]
    return (winner, evidence) if winner != quality and winner in allowed else None


def snap(spans: list[GridSpan], *, tonic_pc: int = 0, mode: str = "ionian") -> tuple[list[GridSpan], int]:
    """Pull minority readings of a root onto the one the song actually plays.

    Returns the new spans and how many were moved. Seven conditions, all of which
    must hold: the move is one `SNAP_TO` allows, the two chords are near
    neighbours in harmonic space, the reading turns up on no more than
    `MAX_OCCASIONS` separate occasions, the song's reading of that root is
    overwhelmingly the other one, this reading holds only a sliver of the root's
    evidence, it was believed less than the song's answer, and the move is not
    away from the key. The module docstring says why each is there; the
    near-miss test is redundant against `SNAP_TO` today and kept anyway, because
    it is the invariant the table has to satisfy and not a fact about this
    version of it.
    """
    if not spans:
        return [], 0

    readings = profile(spans)
    out: list[GridSpan] = []
    snapped = 0
    for span in spans:
        root = span.root_pc % 12
        candidates = readings.get(root, {})
        mine = candidates.get(span.quality)
        found = _dominant(candidates, span.quality)
        if mine is None or found is None:
            out.append(span)
            continue
        quality, winner = found

        total = sum(reading.mass for reading in candidates.values())
        near = harmony.is_near_miss((root, span.quality), (root, quality))
        one_off = mine.spans <= MAX_OCCASIONS
        lopsided = mine.mass <= 0 or winner.mass >= MASS_DOMINANCE * mine.mass
        minority = total <= 0 or mine.mass / total <= MINORITY_SHARE
        believed_less = span.confidence < winner.belief - CONFIDENCE_MARGIN
        in_key = (harmony.diatonic_fit(root, quality, tonic_pc, mode)
                  >= harmony.diatonic_fit(root, span.quality, tonic_pc, mode))

        if not (near and one_off and lopsided and minority and believed_less and in_key):
            out.append(span)
            continue

        # `exact` goes with it: this is no longer the quality the engine reported,
        # so the span can no longer claim to have reached the container intact.
        # Overstating `exactRatio` would make the one field that can say "`hard`
        # is a fiction on this recording" quietly optimistic.
        out.append(replace(span, quality=quality, exact=False))
        snapped += 1

    if snapped:
        log.info("vocabulary: %d span(s) snapped onto the song's own reading", snapped)
    return merge(out), snapped


def absorb_islands(spans: list[GridSpan], *, bar_beats: int = 4) -> tuple[list[GridSpan], int]:
    """Fill in a brief near-miss span flanked by the same chord on both sides.

    §5.4's `drop_short` is this rule with a one-beat floor and no harmonic test.
    Raising the floor without adding the test would delete real music — a
    two-beat passing chord is ordinary — so the floor here is relative (the
    island must be a fraction of what surrounds it) and the harmonic test does
    the discriminating: C | G | C is a passing dominant and survives, because a
    fifth is not a mishearing; Ebm | Eb | Ebm is one chord with a hole in it.

    **Same root only**, which was not the first version of this rule and is the
    corpus's correction to it. Letting the flanks claim a different root looks
    defensible — they surround it, they are the same chord as each other, and the
    reading between them is doubtful — and on "Michelle" it turned a correctly
    heard augmented chord into the Fm either side of it. An augmented triad is one
    set of notes under three names, so the engine had it right and the rule could
    not tell, because it was reasoning about the *label*. Keeping the root makes
    the edit a spelling correction, which is all this rule has the evidence to be.
    """
    if len(spans) < 3:
        return list(spans), 0

    out = list(spans)
    absorbed = 0
    for index in range(1, len(out) - 1):
        before, island, after = out[index - 1], out[index], out[index + 1]
        if (before.root_pc, before.quality) != (after.root_pc, after.quality):
            continue
        if island.root_pc % 12 != before.root_pc % 12:
            continue
        flank = min(before.length_beats, after.length_beats)
        if island.length_beats > ISLAND_SHARE * flank or island.length_beats > bar_beats:
            continue
        if not harmony.is_near_miss((island.root_pc, island.quality),
                                    (before.root_pc, before.quality)):
            continue
        # Against the *weaker* flank, so "believed less than the chord it
        # interrupts" means less than either side of it rather than less than the
        # more confident one.
        if island.confidence >= min(before.confidence, after.confidence) - CONFIDENCE_MARGIN:
            continue
        # Its own confidence is kept, not the flank's. `merge` will take the
        # minimum across the three spans it is about to join, which is
        # §5.4's rule and the right one: merging must not launder a doubtful
        # reading into a confident one, and the chord now spans a stretch the
        # engine was demonstrably unsure about in the middle.
        out[index] = replace(island, quality=before.quality, exact=False)
        absorbed += 1

    if absorbed:
        log.info("vocabulary: %d island(s) absorbed into the chord around them", absorbed)
    return merge(out), absorbed


def absorb_semitone_islands(spans: list[GridSpan], *, bar_beats: int = 4
                            ) -> tuple[list[GridSpan], int]:
    """Absorb a brief span **a semitone away** from the identical chords around it.

    The one mistake nothing else in this pipeline can touch, and the owner's third
    symptom — `A` and `A#` in the same chart. Every near-miss gate in the building
    is closed to it *by construction*: `harmony.similarity(A, A#)` is 0, because
    the two triads share no pitch class at all, so `is_near_miss` says they are as
    far apart as any two chords can be. That is the right answer to "could a
    recognizer slide between these by hearing a third wrong" and the wrong answer
    to "could a recognizer land a semitone out", which is a different failure with
    a different cause: a recording between two CQT bins, or a bass note the model
    took for a root. `absorb_islands` above cannot help either — it requires the
    island to be on the flanks' own root, which is the whole point of it.

    So the harmonic test is replaced rather than relaxed, and what replaces it is
    **the semitone itself plus the flanks being identical**. `A | A# | A` is not a
    progression anyone plays; it is one chord with a slip in the middle of it. A
    fourth or a fifth away would be an ordinary passing chord and is refused here
    exactly as it is refused there.

    Three further gates, and each one is a case this rule would otherwise damage:

    - **the island must be brief**, relative to what surrounds it and never longer
      than a bar — a real chromatic chord gets a bar of its own;
    - **it must be believed less** than either flank, which is the gate every
      correcting layer here carries and the reason this is a no-op on perfect
      input;
    - **neither side may be a quality `NEVER_SNAPPED` protects.** An augmented
      triad is one set of notes under three names, so "the chord a semitone away"
      is not a well-formed statement about it — this is `absorb_islands`'
      Michelle case arriving by a different road, and it is refused on the same
      grounds.
    """
    if len(spans) < 3:
        return list(spans), 0

    out = list(spans)
    absorbed = 0
    for index in range(1, len(out) - 1):
        before, island, after = out[index - 1], out[index], out[index + 1]
        if (before.root_pc, before.quality) != (after.root_pc, after.quality):
            continue
        distance = (island.root_pc - before.root_pc) % 12
        if distance not in (1, 11):
            continue
        if island.quality in NEVER_SNAPPED or before.quality in NEVER_SNAPPED:
            continue
        flank = min(before.length_beats, after.length_beats)
        if island.length_beats > ISLAND_SHARE * flank or island.length_beats > bar_beats:
            continue
        if island.confidence >= min(before.confidence, after.confidence) - CONFIDENCE_MARGIN:
            continue
        out[index] = replace(island, root_pc=before.root_pc, quality=before.quality,
                             exact=False)
        absorbed += 1

    if absorbed:
        log.info("vocabulary: %d semitone slip(s) absorbed into the chord around them",
                 absorbed)
    return merge(out), absorbed


def consolidate(spans: list[GridSpan], *, tonic_pc: int = 0, mode: str = "ionian",
                bar_beats: int = 4) -> tuple[list[GridSpan], VocabularyReport]:
    """Both rules, in the order that makes each one see the other's work.

    Islands first, because an island is a hole in a chord rather than a reading
    of a root: absorbing it removes a span that would otherwise contribute its
    (small, doubtful) mass to the profile, and — more importantly — it removes a
    *boundary*, which is what the player actually sees. Then `snap`, over a
    profile that now describes the song rather than the wobbles in it.

    Every rule re-merges, so what comes back is a contiguous timeline with no
    adjacent duplicates — the same invariant `postprocess.process` hands over,
    which is what lets this run as a step inside that chain.
    """
    cleaned, absorbed = absorb_islands(spans, bar_beats=bar_beats)
    # The semitone rule after the same-root one, so a slip that is also a colour
    # island has already been handled by the narrower rule. Both before `snap`,
    # for the reason the docstring gives: an island is a hole rather than a
    # reading, and removing it stops it contributing its small, doubtful mass to
    # the profile `snap` is about to weigh.
    cleaned, slipped = absorb_semitone_islands(cleaned, bar_beats=bar_beats)
    cleaned, snapped = snap(cleaned, tonic_pc=tonic_pc, mode=mode)
    return cleaned, VocabularyReport(snapped_spans=snapped, absorbed_islands=absorbed,
                                     absorbed_semitones=slipped)
