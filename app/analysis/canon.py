"""§21 — a chart states the song's form, not a transcript of every pass.

The complaint this module exists for, in the owner's words: *"clearly our system
doesn't know that a verse (or any section) is the same across the whole song"*.
It was right, and the output showed it. Creep is four chords over eight bars for
its whole length, and the pipeline emitted **eighty-eight distinct bars** in one
unbroken section with a ten-chord vocabulary — every wobble the recognizer made
on every pass preserved as though it were music.

Everything needed to fix that was already in the building. `form.py` finds the
repeat groups; `consensus.py` votes over them. What was missing is the step
between "these eleven blocks are the same music" and "so print them the same
way", and it was missing for a reason worth writing down, because the reason is
also the boundary of what this module may do.

Why `consensus` could not be it
-------------------------------

§20.4 is a **correction** layer: it asks whether the engine misheard a bar, and
overwrites only where three independent gates all say yes — the disagreement is
a near miss, the dissenter was believed less, and a plurality agreed. Those
gates exist because a correction that is wrong shows the player a chord nobody
played, and they buy a property the benchmark leans on: on perfect input the
vote is provably a no-op.

They also mean it stays silent on the ordinary case. Creep's eleventh bar is
heard as `Cm` on six passes, `Cm D#` on three and `Cm F` on two; nothing there
is a near miss of anything (the bars differ in how many chords they hold, not in
a third), so gate 2 refuses and all eleven readings ship. That is the correct
answer to *"did the engine mishear bar 8 of pass 7"* and the wrong answer to
*"what does this song play in bar 8"*.

This module asks the second question. It is not a correction layer and it does
not try to be: it does not ask whether any occurrence was misheard, it asks what
the group **agrees** on and writes that everywhere. A chart is a claim about the
song, and a song does not play its verse eleven subtly different ways.

The property that keeps it honest is the same one §20 is judged by, and it comes
free from the construction rather than from a gate: **on perfect input this is a
no-op**, because every occurrence is already identical and the agreed reading is
the reading. What it costs, and this is the real trade and not a hedge, is the
occurrence that genuinely differs — a fourth verse with a substitution in bar 6
is flattened onto the other three. Three things bound that cost:

- **Only groups that already cohere.** `MIN_COHESION` — a group whose
  occurrences merely resemble each other is not one piece of music played
  repeatedly, and is left exactly as it was heard.
- **Only where the group agrees.** A beat needs a *strict* plurality — one
  reading ahead of every other — before it may speak for all the occurrences. A
  tie is not resolved by a coin toss and it is not left as a hole either: the bar
  **holds** the chord it has most recently settled on, which is §5.4.4's own rule
  (`postprocess` fills every gap in the timeline; §18 has no rest primitive, so a
  stroke always sounds *something*) and which is also what a musician writing the
  chart does with a passing chord half the passes disagree about. A bar whose
  beats all tie is declined outright and every occurrence keeps its own reading.
- **Never invented.** The winner at every beat is a chord some occurrence
  actually played *in that bar* — held beats included, since what they hold is
  the neighbouring beat's winner. Nothing here can synthesise a symbol the
  recording did not produce, which is the rule `consensus._weigh` already works
  under.

Voting at beat resolution, not bar resolution
---------------------------------------------

The obvious implementation — take the most common *bar* — is worse than it
looks, and the failure is the interesting one. Occurrences disagree about
**where inside the bar** a change happens at least as often as they disagree
about what the chord is: eight passes of `| G |` and four of `| G D |` are not
two competing bars, they are twelve passes that all say G on beats 1–3 and
disagree only about beat 4. Whole-bar voting throws the agreement away and picks
a winner on a 8-vs-4 split of composite objects; beat voting keeps it, and
settles beat 4 on its own evidence.

Beats are the right atom rather than an approximation of one, because
`postprocess.quantize` has already snapped every boundary to a beat index —
there is no sub-beat information left in a `BarChord` to lose. The bar is then
rebuilt from runs of equal beats, so `[G G G D]` comes back out as `| G D |`
with the D one beat long, and `[G G G G]` as `| G |`.

Where this runs, and why there
------------------------------

Between the two `form.detect` passes `model.build` already makes, in the slot
the second pass was written for: §20.6 explains that the first pass finds the
groups and the second encodes them, and that the encoding pass exists because
*"after the vote they often **are** identical, and the second pass is what turns
that into the compact encoding the container wants"*. Before this module, "often"
meant "on synthetic input"; on a recording the occurrences were never identical
and `repeats` never fired, so every song compiled as one flat run of bars. That
is why Creep's eighty-eight bars came out as a single section: `_sections_from`
merged eleven consecutive blocks of one group and could not collapse any of
them.

So this is the step that makes the encoding pass mean what it says. It runs on
the groups found *after* the vote, and a further `form.detect` then re-reads the
sections out of the canonical bars.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from . import harmony
from .form import RepeatGroup
from .structure import BarChord
from .types import GridSpan
from .vocabulary import SNAP_TO

log = logging.getLogger("chords.canon")

# How much of a repeat a group has to be before its occurrences are made to
# agree. `form.CLUSTER_SIMILARITY` (0.75) is the bar for two blocks *joining* a
# group, and this is deliberately below it: cohesion is the **mean pairwise**
# similarity over every member, so a group of eight occurrences each of which
# cleared 0.75 against the representative can sit in the low 0.7s against each
# other and still be one verse the engine read eight ways. What this rejects is
# the group that clustered on a technicality — a couple of shared bars and a
# different second half — where "they are the same music" is the premise that
# has failed rather than the conclusion being drawn.
MIN_COHESION = 0.6

# How many occurrences must read a beat the same way before their reading may
# speak for all of them. Two, and it is the same floor and the same argument as
# `consensus.MIN_AGREEING`: one pass agreeing with itself is not a vote. Above
# this the rule is a **strict plurality** rather than a share, for the reason
# §20.4 records — errors do not arrive one per song, and a threshold expressed as
# a share of the occurrences gets *harder* to meet the more times the engine
# repeats a mistake, which is precisely backwards.
MIN_AGREEING = 2


@dataclass(frozen=True)
class CanonReport:
    """What the form cost, in bars — provenance for an edit nothing can see.

    `canonical_bars` counts bars this module rewrote, and it is not comparable
    with `ConsensusReport.rewritten_bars`: that one counts corrections the engine
    is thought to have got wrong, this one counts bars brought into line with
    their own section. A song reporting a large number here is one whose form is
    strong and whose per-pass readings were noisy — the ordinary case, and the
    one the layer was built for.

    The other two say where the agreement ran out. `held_beats` counts beats the
    occurrences tied on, where the bar held the chord it had already settled on;
    `split_bars` counts slots where *nothing* settled and every occurrence kept
    its own reading. Both are worth watching on a song whose sections genuinely
    vary, and a song with a high `split_bars` is one this layer has correctly
    decided it cannot speak for.
    """

    groups_canonical: int = 0
    groups_declined: int = 0
    canonical_bars: int = 0
    held_beats: int = 0
    split_bars: int = 0
    # §21b. `bar_rhythm` is whether the song was found to change chord a bar at a
    # time at all; `settled_bars` is how many bars were then given the chord that
    # held most of them. Zero settled on a song that moves faster than its bar,
    # which is a fact about the song rather than a failure to act.
    bar_rhythm: bool = False
    settled_bars: int = 0

    @property
    def touched(self) -> bool:
        return bool(self.canonical_bars or self.settled_bars)


def voice(bars: list[list[BarChord]]) -> dict[tuple[int, str], float]:
    """How much of the song sounds as each `(root, quality)` — the vocabulary a
    tied beat is settled against. Duration-weighted, over the whole chart."""
    out: dict[tuple[int, str], float] = {}
    for bar in bars:
        for chord in bar:
            key = (chord.root_pc, chord.quality)
            out[key] = out.get(key, 0.0) + max(0.0, chord.length_beats)
    return out


def canonicalize(bars: list[list[BarChord]], groups: list[RepeatGroup], *,
                 bar_beats: int, tonic_pc: int = 0, mode: str = "ionian",
                 record: bool = True) -> tuple[list[list[BarChord]], CanonReport]:
    """Make every occurrence of a repeat group play the group's progression.

    Returns a **new** bar list, like `consensus.apply` and for the same reason:
    a caller that wants to know what changed can compare the two.

    `record` writes the agreed progression back onto each `RepeatGroup` as its
    `canonical`, which is what the sidecar reports and what `compile` names. It
    is off for a render for exactly the reason it is off in
    `consensus.apply` — those groups are shared objects describing the reference
    analysis, and a render is a replay of it rather than a new finding.
    """
    out = [list(bar) for bar in bars]
    spoken = voice(bars)
    made = declined = rewritten = held = split = 0

    for group in groups:
        if not group.is_repeat or group.length_bars <= 0:
            continue
        if group.cohesion < MIN_COHESION:
            declined += 1
            log.info("group %s left as heard: cohesion %.2f below %.2f",
                     group.label, group.cohesion, MIN_COHESION)
            continue
        made += 1

        for slot in range(group.length_bars):
            positions = [start + slot for start in group.occurrences
                         if start + slot < len(out)]
            if len(positions) < 2:
                continue
            agreed, tied = _agreed([out[p] for p in positions], bar_beats=bar_beats,
                                    tonic_pc=tonic_pc, mode=mode, spoken=spoken)
            held += tied
            if agreed is None:
                split += 1
                continue
            for position in positions:
                if _signature(out[position]) != _signature(agreed):
                    out[position] = _adopt(agreed)
                    rewritten += 1

        if record:
            first = group.occurrences[0]
            group.canonical = [list(out[first + i]) for i in range(group.length_bars)
                               if first + i < len(out)]

    if rewritten:
        log.info("form applied: %d bar(s) across %d group(s) now play their "
                 "section's progression (%d slot(s) left split)",
                 rewritten, made, split)
    return out, CanonReport(groups_canonical=made, groups_declined=declined,
                            canonical_bars=rewritten, held_beats=held,
                            split_bars=split)


# --- one bar slot -----------------------------------------------------------

def _agreed(candidates: list[list[BarChord]], *, bar_beats: int, tonic_pc: int,
            mode: str, spoken: dict[tuple[int, str], float] | None = None
            ) -> tuple[list[BarChord] | None, int]:
    """What a group's occurrences agree this bar is, and how many beats tied.

    `None` means nothing in the bar settled, so every occurrence keeps what it
    played. Otherwise the bar is contiguous by construction: a beat with no
    strict plurality holds its neighbour rather than opening a hole, because a
    bar with a gap in the middle of it is not a bar the rest of the pipeline can
    carry (`_as_flat` refuses a chord that doesn't fill its bar, and §18 has no
    rest to put there).

    Holding *backwards* — the chord already settled on — rather than forwards is
    the same asymmetry `postprocess.fill` uses, and for the same reason: a chord
    that has begun sounding keeps sounding until something replaces it, which is
    what a held instrument does and what the container's spans mean. Only a tie
    at the head of the bar, where there is nothing behind it yet, reaches
    forwards instead.

    **Voting per beat can in principle assemble a bar no occurrence played** —
    each chord in it is one some occurrence played *in that bar*, but the
    sequence can be new. The 2026-08-18 audit flagged this (F16) and recommended
    constraining the output to whichever candidate bar is closest to the per-beat
    consensus. Measured over the chart corpus: **3 slots out of 1744** come back
    as a sequence no candidate held, 0.2%. That is not worth constraining the
    vote for, and constraining it would give up the property the whole method was
    chosen for — that occurrences disagreeing about *where* a change falls are
    settled beat by beat rather than by a whole-bar plurality of composite
    objects, which is where §21's +0.049 root came from.
    """
    settled: list[tuple[int, str] | None] = []
    confidences: list[float] = []
    tied = 0

    for beat in range(bar_beats):
        readings = [found for bar in candidates
                    if (found := _sounding(bar, beat)) is not None]
        tally: dict[tuple[int, str], list[float]] = {}
        for chord, confidence in readings:
            tally.setdefault(chord, []).append(confidence)

        # Count first, then belief, then the key — the same ordering
        # `consensus._vote` uses, and for the same reasons. The last two terms
        # keep the result independent of dict insertion order (§16.5 wants
        # byte-stable output).
        winner = max(tally, key=lambda chord: (
            len(tally[chord]),
            sum(tally[chord]) / len(tally[chord]),
            harmony.diatonic_fit(chord[0], chord[1], tonic_pc, mode),
            -chord[0], chord[1],
        ), default=None)
        runner_up = max((len(votes) for chord, votes in tally.items() if chord != winner),
                        default=0)
        if winner is not None and len(tally[winner]) <= runner_up:
            # The count has tied. The rest of the song may still speak.
            winner = _spoken_for(tally, spoken)
        if winner is None or len(tally[winner]) < MIN_AGREEING:
            tied += 1
            settled.append(None)
            confidences.append(0.0)
            continue
        settled.append(winner)
        confidences.append(sum(tally[winner]) / len(tally[winner]))

    if all(chord is None for chord in settled):
        return None, tied
    return _bar_from_beats(_held(settled), confidences), tied


def _spoken_for(tally: dict[tuple[int, str], list[float]],
                spoken: dict[tuple[int, str], float] | None) -> tuple[int, str] | None:
    """Settle a tied beat with the song's own vocabulary, where it may speak.

    A count that ties has run out of evidence *inside* the group, and the rest of
    the song is the obvious next witness — this is §20.8's argument, in the one
    place §20.8 itself cannot reach, because the readings here are contained in a
    single bar slot rather than spread over a whole root's history.

    Someone Like You is the case. Its verse plays A / C#m / F#m / D and the engine
    hears the third bar as `F#` on two passes and `F#m` on the other two: an exact
    2–2 split, so the count declines and both readings ship. Meanwhile the chorus
    plays F#m twenty times over. "This song plays F#m on that root" is not a close
    call, and it is the only evidence anywhere that can settle the slot.

    Two gates, and they are the ones that make this the vocabulary rule rather
    than a licence:

    - the tied readings must share a **root** and differ in quality alone, and the
      move must be one `vocabulary.SNAP_TO` allows — the measured table, so a
      seventh may flatten onto its triad and never the reverse;
    - the song's answer must be the one it plays **more** of. Not a landslide:
      §20.8 demands `MASS_DOMINANCE` because it is overruling a reading nothing
      else disputes, and here the reading is already tied against an alternative
      the same section played.
    """
    if not spoken or len(tally) != 2:
        return None
    (first, _), (second, _) = tally.items()
    if first[0] != second[0]:
        return None
    winner, loser = ((first, second) if spoken.get(first, 0.0) > spoken.get(second, 0.0)
                     else (second, first))
    if spoken.get(winner, 0.0) <= spoken.get(loser, 0.0):
        return None
    if winner[1] not in SNAP_TO.get(loser[1], ()):
        return None
    return winner


def _held(settled: list[tuple[int, str] | None]) -> list[tuple[int, str]]:
    """Fill every tied beat with the chord the bar had already settled on."""
    out: list[tuple[int, str]] = []
    carry: tuple[int, str] | None = None
    for chord in settled:
        carry = chord if chord is not None else carry
        out.append(carry)                       # type: ignore[arg-type]
    # A tie at the head has nothing behind it, so it takes the first thing ahead.
    lead = next(chord for chord in out if chord is not None)
    return [chord if chord is not None else lead for chord in out]


def _sounding(bar: list[BarChord], beat: int) -> tuple[tuple[int, str], float] | None:
    """Which chord is sounding on a beat of this bar, and how much it was
    believed. `None` where the bar covers that beat with nothing, which happens
    on a truncated final occurrence."""
    for chord in bar:
        if chord.start_beat <= beat < chord.start_beat + chord.length_beats:
            return (chord.root_pc, chord.quality), chord.confidence
    return None


def _bar_from_beats(beats: list[tuple[int, str]],
                    confidences: list[float]) -> list[BarChord]:
    """Per-beat readings → the bar they describe, as runs.

    A run of one chord becomes one `BarChord` however many beats it covers, so
    four beats of G come back as `| G |` and not as `| G G G G |`, and
    `| G G G D |` comes back as two spans with the D one beat long. The bar is
    contiguous by construction — `_agreed` has already refused anything with a
    beat missing from it.
    """
    out: list[BarChord] = []
    index = 0
    while index < len(beats):
        chord = beats[index]
        run = index
        while run + 1 < len(beats) and beats[run + 1] == chord:
            run += 1
        # A held beat contributes no confidence of its own — it was a tie — so
        # the run is believed as much as the beats that actually settled it.
        span = [confidences[i] for i in range(index, run + 1) if confidences[i] > 0.0]
        out.append(BarChord(
            root_pc=chord[0], quality=chord[1],
            start_beat=float(index), length_beats=float(run + 1 - index),
            confidence=sum(span) / len(span) if span else 0.0,
        ))
        index = run + 1
    return out


def _adopt(agreed: list[BarChord]) -> list[BarChord]:
    """A fresh copy of the agreed bar. Copied rather than shared because the
    bars are mutable lists downstream, and one occurrence's later edit must not
    reach through into every other pass of the same section."""
    return [BarChord(root_pc=c.root_pc, quality=c.quality, start_beat=c.start_beat,
                     length_beats=c.length_beats, confidence=c.confidence)
            for c in agreed]


def _signature(bar: list[BarChord]) -> tuple:
    return tuple((c.root_pc, c.quality, round(c.start_beat, 4), round(c.length_beats, 4))
                 for c in bar)


# --- harmonic rhythm --------------------------------------------------------
#
# §21b — where the song changes chord, as opposed to which chord it changes to.
#
# `postprocess.quantize` snaps every boundary to the nearest **beat**, and §5.4
# argues for that: "a chord change that lands 40 ms before the beat is the same
# musical event as one that lands on it". The argument does not stop at the beat.
# In a song whose harmony moves once a bar, a change the engine put on beat 4 is
# the same musical event as one on beat 1 of the next bar — an anticipation, the
# most ordinary thing in this repertoire, and the reason a published chart of
# Don't Stop Believin' reads `| E | B | C#m | A |` and not `| E B | B | C# C#m A |`.
#
# That second chart is what the pipeline emitted. On the bench corpus 43% of Don't
# Stop Believin's bars and 28% of Viva La Vida's held more than one chord, against
# reference charts where every bar holds exactly one — and the cost is not only
# cosmetic: a bar holding the tail of one chord and the head of the next matches
# *neither* reference bar, so it loses the root twice.
#
# What makes this safe to do at all is that it is **not** applied unless the song
# says so. Wonderwall really does change twice a bar (89% of its bars hold two
# chords, and its reference says so), and the same rule applied there would
# destroy the song. So the harmonic unit is measured first, from the song's own
# chord durations, and the settling only runs where the measurement says the
# harmony moves a bar at a time or slower.

# How much of a bar one chord must hold before the bar is called that chord.
# Three beats in four: enough that the bar is unambiguously *about* that chord,
# and tight enough that a genuine two-chord bar (2+2, or 3+1 where the 1 is on a
# different beat of a song that moves twice a bar) is never touched — those songs
# are excluded by the harmonic-unit gate before this constant is consulted.
SETTLE_SHARE = 0.75

def harmonic_unit(spans: list[GridSpan]) -> float:
    """How long this song holds a chord, in beats — its harmonic rhythm.

    The **duration-weighted median** span length: the length of the chord you
    would land on by dropping a pin somewhere in the song. Weighting by duration
    rather than counting spans is what makes it a statement about the music
    instead of about the engine's flicker rate — a recognizer that stutters four
    times inside one held chord adds four short spans to a plain median and moves
    it, and adds almost nothing to this one.

    Measured over the bench corpus, against bar_beats = 4:

        Wonderwall            2      changes twice a bar — and its reference says so
        Smooth Criminal       2      the same, `| Am G | F E |`
        Zombie                4      one chord a bar
        Don't Stop Believin'  4      one chord a bar
        Viva La Vida          4      one chord a bar
        Someone Like You      4      one chord a bar
        Country Roads         8      the tracker's octave is doubled; two of its
                                     bars are one of the song's
        Creep, I'm Yours      8      two bars a chord

    The separation is clean and it is the whole gate: 2 on both songs that really
    move twice a bar, 4 or more on every song that does not.

    An earlier version of this measured the *share of bars holding one chord*, and
    it fails in the most instructive way available — the songs whose bars are
    messiest are exactly the ones that most need settling, so the measurement they
    fail is the one that would have fixed them. Don't Stop Believin' held two
    chords in 43% of its bars and was refused for it. Measuring the chords rather
    than the bars they were sliced into asks the question before the damage.
    """
    lengths = sorted(s.length_beats for s in spans if s.length_beats > 0)
    if not lengths:
        return 0.0
    half = sum(lengths) / 2.0
    running = 0.0
    for length in lengths:
        running += length
        if running >= half:
            return length
    return lengths[-1]


def settles_on_barlines(spans: list[GridSpan], bar_beats: int) -> bool:
    """Does this song change chord a bar at a time, or faster?"""
    return bar_beats > 0 and harmonic_unit(spans) >= bar_beats - 1e-6


def settle_to_bars(bars: list[list[BarChord]], bar_beats: int,
                   groups: list[RepeatGroup] | None = None
                   ) -> tuple[list[list[BarChord]], int]:
    """Give every bar the chord that holds most of it, where one clearly does.

    Only called on songs `settles_on_barlines` accepts. A bar where no chord
    reaches `SETTLE_SHARE` is left exactly as it was heard: two chords sharing a
    bar down the middle is a real two-chord bar even in a song that mostly moves
    once a bar, and there is no reading of "the bar's chord" that covers it.

    **And a split the song plays every time round is not settled either.** The
    `SETTLE_SHARE` test asks one bar in isolation whether its second chord looks
    like an anticipation, and one bar cannot tell an anticipation from a cadence:
    `| IV V |` at the end of a phrase reads exactly like `| IV |` with the next
    chord pushed early. What separates them is the *other passes of the same
    slot* — an engine putting a change a beat early does it on some passes and
    not others, and a song that really splits that bar splits it every time.
    So where `groups` is supplied, a bar whose siblings mostly hold two chords is
    left alone however lopsided this one occurrence happens to be.

    `groups` optional so a caller with no form in hand (a test fixture, or the
    replay in `model.render` before the groups exist) gets the old behaviour
    rather than an error.
    """
    if bar_beats <= 0:
        return [list(bar) for bar in bars], 0
    corroborated = _corroborated_splits(bars, groups, bar_beats)
    out: list[list[BarChord]] = []
    settled = 0
    for index, bar in enumerate(bars):
        winner = _longest(bar)
        if winner is None or len(bar) == 1:
            out.append(list(bar))
            continue
        if winner.length_beats / bar_beats < SETTLE_SHARE:
            out.append(list(bar))
            continue
        if index in corroborated:
            out.append(list(bar))
            continue
        out.append([BarChord(root_pc=winner.root_pc, quality=winner.quality,
                             start_beat=0.0, length_beats=float(bar_beats),
                             confidence=winner.confidence)])
        settled += 1
    return out, settled


def _corroborated_splits(bars: list[list[BarChord]],
                         groups: list[RepeatGroup] | None,
                         bar_beats: int) -> frozenset[int]:
    """Bar indices whose slot is split in most of the occurrences that hold it.

    A slot is "split" in an occurrence when no chord in that bar reaches
    `SETTLE_SHARE` — the same test `settle_to_bars` applies, asked of the
    siblings. A strict majority is required, so a slot two passes disagree about
    is still settled and only a slot the song keeps splitting is protected.
    """
    if not groups:
        return frozenset()
    protected: set[int] = set()
    for group in groups:
        if not group.is_repeat or group.length_bars <= 0:
            continue
        for slot in range(group.length_bars):
            positions = [start + slot for start in group.occurrences
                         if start + slot < len(bars)]
            if len(positions) < 2:
                continue
            split = 0
            for position in positions:
                winner = _longest(bars[position])
                if (len(bars[position]) > 1 and winner is not None
                        and winner.length_beats / bar_beats < SETTLE_SHARE):
                    split += 1
            if split * 2 > len(positions):
                protected.update(positions)
    return frozenset(protected)


def _longest(bar: list[BarChord]) -> BarChord | None:
    """The bar's chord, by how much of the bar it holds. Ties go to the earlier
    one — the bar is named for what it starts on."""
    best: BarChord | None = None
    for chord in bar:
        if best is None or chord.length_beats > best.length_beats:
            best = chord
    return best
