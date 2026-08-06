"""§20.4 — using a song's repeats to clean up the engine's mistakes.

This is the payoff of `form.py` and the most dangerous module in the service.

The idea is sound and old: a chord recognizer makes *independent* mistakes, but
a song plays its verse the same way every time, so four passes of one verse are
four noisy readings of one signal and the disagreements between them are mostly
noise. Averaging them is free accuracy — that is why structure-aware chord
transcription works at all.

The danger is equally simple. The fourth verse might genuinely be different, and
a layer that "standardizes" it has shown the player a chord that is not being
played. Nothing downstream can catch that: `lint` and `lint_sync` both check the
song against *itself*, and a chart that has been made uniformly self-consistent
is exactly the thing they are least able to complain about. `axis.py` records
what that class of failure already cost this service once — 23 points of
delivered accuracy behind three hundred green tests — and
`form._sections_from` carries the same warning about `repeats`.

So overwriting one occurrence with another is gated on **three** independent
conditions, and all three must hold:

1. **The vote is decisive** — at least `MIN_AGREEING` occurrences read the slot
   *identically*, and no other reading is agreed by as many. Two occurrences that
   disagree can never satisfy this, which is correct: with one reading against one
   reading there is nothing to learn from counting.

   This used to be a share — two-thirds of the occurrences had to agree — and that
   is the wrong shape for the thing being counted, in a way that made the layer
   fail on exactly the input it exists for. Errors do not arrive one per song. A
   verse played four times with the engine mishearing the same bar in *two* of
   them, differently each time, gave a 2-of-4 majority and was filed as
   `contested`, so both mistakes shipped — and the fourfold repetition that was
   supposed to be the evidence counted against it, because every extra bad
   reading pushed the share further down. Yet two passes agreeing exactly while
   the dissenters disagree with the majority *and with each other* is the
   signature of noise, not of music: a song that really changes its fourth verse
   changes it the same way every time it plays it.

   So the requirement is a **plurality with a floor**, and the safety is carried
   where it belongs — by gates 2 and 3, which every dissenter must still clear
   individually. Any single occurrence a fifth away from the winner, or believed
   as much as it, still contests the whole slot for every occurrence.

2. **The disagreement is a near-miss** (`harmony.is_near_miss`). C against Am is
   the mix being ambiguous about a third; C against F is the music changing. The
   first is worth voting on and the second is not, and no amount of agreement
   makes it so — a chord four passes away in harmonic space is not a mishearing.

3. **The dissenter was believed less than the winner.** This is the condition
   that does the real work, because it is the only one that consults evidence
   from *outside* the repetition. An engine that was confident about the F in
   verse 3 is telling us the F is there; an engine that hedged is telling us it
   could not hear. Counting alone cannot tell a real change from a repeated
   mistake — the confidence can.

Condition 3 also gives the layer a property worth stating plainly, because it is
what makes the whole thing testable: **on perfect input, consensus is provably a
no-op.** Feed ground truth in as both engines and every span carries confidence
1.0, so no dissenter is ever less believed than a winner, so nothing is ever
rewritten. The benchmark asserts exactly that (`bench/run_bench.py`, the
truth-as-engines run), and it is a guarantee from the construction rather than a
number that happened to come out right.

Where the gates do not all hold, the slot is **contested**: nothing is
rewritten, for any occurrence, and the group is reported as having a real
difference in it. Contested slots are counted, not hidden — a song whose verses
genuinely vary is a fact about the song, and the sidecar says so.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from . import harmony
from .form import RepeatGroup, bar_similarity
from .strumming import ExtractedPattern, FoldedOnset, extract, fallback, fold_onsets
from .structure import BarChord
from .types import Onset

log = logging.getLogger("chords.consensus")

# How many occurrences must read a slot identically before their reading may
# overwrite another. Two, which is the smallest number that is evidence at all:
# one pass agreeing with itself is not a vote, and a 1-of-2 split has nothing to
# count. It has to be a *plurality* as well (see `_vote`), so 2-of-4 carries only
# while no other reading also has two.
MIN_AGREEING = 2

# How much less believed a dissenting bar must be before the majority may
# replace it. Small on purpose: the requirement is that the winner was believed
# *more*, not that the dissenter was hopeless. Any value ≥ 0 preserves the
# no-op-on-perfect-input property, since ground truth arrives at a flat 1.0.
CONFIDENCE_MARGIN = 0.02


@dataclass(frozen=True)
class ConsensusReport:
    """What the vote did — provenance, not decoration.

    Every one of these numbers describes an edit the player cannot see and the
    lint cannot check, which is exactly why they are carried out of the module
    and into the sidecar instead of being logged and forgotten.
    """

    groups_voted: int = 0
    rewritten_bars: int = 0
    contested_bars: int = 0

    @property
    def touched(self) -> bool:
        return bool(self.rewritten_bars)


def apply(bars: list[list[BarChord]], groups: list[RepeatGroup], *,
          bar_beats: float, tonic_pc: int = 0, mode: str = "ionian",
          record: bool = True) -> tuple[list[list[BarChord]], ConsensusReport]:
    """Vote each repeat group's occurrences into agreement, where allowed.

    Returns a **new** bar list; the input is not mutated, so a caller can always
    compare the two — which is what the benchmark does to count what changed.

    `record` is the exception to "not mutated": what the vote did is written back
    onto each `RepeatGroup` (`rewritten_bars`, `contested_bars`, `canonical`) so
    the provenance travels with the group it describes. Those objects are
    **shared** — `model.render` votes over the same groups once per difficulty
    tier — and a tier's vote is a render of the reference one, not a new finding
    about the song. So renders pass `record=False` and the numbers on the groups
    keep describing the reference vote rather than whichever tier compiled last.
    """
    out = [list(bar) for bar in bars]
    rewritten = contested = voted = 0

    for group in groups:
        if not group.is_repeat or group.length_bars <= 0:
            continue
        voted += 1
        group_rewritten = 0
        group_contested: list[int] = []

        for slot in range(group.length_bars):
            positions = [start + slot for start in group.occurrences
                         if start + slot < len(out)]
            if len(positions) < 2:
                continue
            outcome = _vote([out[p] for p in positions], bar_beats=bar_beats,
                            tonic_pc=tonic_pc, mode=mode)
            if outcome is None:
                group_contested.append(slot)
                contested += 1
                continue
            winner, losers = outcome
            for index in losers:
                out[positions[index]] = _adopt(winner, out[positions[index]])
                group_rewritten += 1
                rewritten += 1

        if record:
            group.rewritten_bars = group_rewritten
            group.contested_bars = tuple(group_contested)
            # The canonical pass is read back out *after* voting, so it is the
            # progression the song settled on rather than whichever occurrence
            # happened to come first.
            first = group.occurrences[0]
            group.canonical = [list(out[first + i]) for i in range(group.length_bars)
                               if first + i < len(out)]
            if group_rewritten or group_contested:
                log.info("group %s: %d bar(s) voted into line, %d contested",
                         group.label, group_rewritten, len(group_contested))

    return out, ConsensusReport(groups_voted=voted, rewritten_bars=rewritten,
                                contested_bars=contested)


def _vote(candidates: list[list[BarChord]], *, bar_beats: float,
          tonic_pc: int, mode: str) -> tuple[list[BarChord], list[int]] | None:
    """One bar slot across a group's occurrences → (winner, indices to rewrite).

    None means **contested**: the gates did not all hold, so this slot is left
    exactly as every occurrence played it. An empty loser list means there was
    nothing to decide — every occurrence already agreed.

    Those two are emphatically not the same thing, and conflating them makes the
    reported numbers lie: a song whose verses match perfectly would report every
    bar as "contested", which reads as "this song's sections genuinely vary"
    when it means the exact opposite.
    """
    tally: dict[tuple, list[int]] = {}
    for index, bar in enumerate(candidates):
        tally.setdefault(_signature(bar), []).append(index)

    if len(tally) == 1:
        return candidates[0], []   # unanimous — nothing to do, nothing contested

    ranked = sorted(
        tally.items(),
        key=lambda item: (
            len(item[1]),                                   # how many agree
            _mean_confidence([candidates[i] for i in item[1]]),
            _diatonic(candidates[item[1][0]], tonic_pc, mode),
            -item[1][0],                                    # earliest, for determinism
        ),
        reverse=True,
    )
    winning_signature, winning_indices = ranked[0]
    runner_up = len(ranked[1][1]) if len(ranked) > 1 else 0
    if len(winning_indices) < MIN_AGREEING or len(winning_indices) <= runner_up:
        return None                                          # gate 1

    winner = candidates[winning_indices[0]]
    winner_confidence = _mean_confidence([candidates[i] for i in winning_indices])

    losers: list[int] = []
    for signature, indices in ranked[1:]:
        for index in indices:
            other = candidates[index]
            if bar_similarity(winner, other, bar_beats) < harmony.NEAR_MISS:
                return None                                  # gate 2
            if _mean_confidence([other]) >= winner_confidence - CONFIDENCE_MARGIN:
                return None                                  # gate 3
            losers.append(index)
    return winner, losers


def _adopt(winner: list[BarChord], loser: list[BarChord]) -> list[BarChord]:
    """Replace a bar with the group's agreed one.

    The adopted bar keeps the **winner's** confidence, because that is now what
    the chart is claiming: this bar says what the group says, and it is believed
    as much as the group believed it. Carrying the loser's confidence forward
    would understate a bar we have more evidence for than any single pass.
    """
    return [BarChord(root_pc=c.root_pc, quality=c.quality, start_beat=c.start_beat,
                     length_beats=c.length_beats, confidence=c.confidence)
            for c in winner]


def _signature(bar: list[BarChord]) -> tuple:
    return tuple((c.root_pc, c.quality, round(c.start_beat, 4), round(c.length_beats, 4))
                 for c in bar)


def _mean_confidence(bars: list[list[BarChord]]) -> float:
    chords = [c for bar in bars for c in bar]
    if not chords:
        return 0.0
    total = sum(c.length_beats for c in chords)
    if total <= 0:
        return sum(c.confidence for c in chords) / len(chords)
    return sum(c.confidence * c.length_beats for c in chords) / total


def _diatonic(bar: list[BarChord], tonic_pc: int, mode: str) -> float:
    """How well a bar sits in the key — the tie-break, and only ever that.

    Reached only when two readings are already near-misses of each other and
    already tied on count and on confidence. At that point the song's own key is
    the best remaining evidence, and preferring the diatonic reading is what a
    musician transcribing by ear would do. It is never allowed to reject a chord
    on its own: borrowed chords and secondary dominants are normal music, and a
    rule that called them errors would do more damage than the errors it caught.
    """
    if not bar:
        return 0.0
    return sum(harmony.diatonic_fit(c.root_pc, c.quality, tonic_pc, mode)
               for c in bar) / len(bar)


# --- patterns ---------------------------------------------------------------

def pattern_for_group(group: RepeatGroup, *, onsets: list[Onset], axis,
                      bar_beats: float, tempo: int, name: str,
                      time_signature: str) -> ExtractedPattern:
    """§14's extraction, folded over **every bar of every occurrence**.

    The unambiguous half of this module. §14's method already averages a
    section's bars onto one bar — that is what folding *is* — so pooling the
    other occurrences of the same music changes nothing about the estimator and
    multiplies its sample. A four-bar verse played four times goes from four
    bars of evidence to sixteen, which is the difference between
    `SUPPORT_THRESHOLD` meaning something and it being a coin toss.

    There is no "overwrote real music" risk here of the kind the chord vote has
    to guard against: a pattern was always a per-section average, and every
    occurrence of the group already shared one. The only thing that changes is
    how much evidence stands behind it.
    """
    if not onsets or group.length_bars <= 0:
        return fallback(bar_beats=bar_beats, tempo=tempo, name=name,
                        time_signature=time_signature)

    folded: list[FoldedOnset] = []
    bars = 0
    for start in group.occurrences:
        first_beat = start * bar_beats
        last_beat = first_beat + group.length_bars * bar_beats
        folded.extend(fold_onsets(onsets, axis, bar_beats=bar_beats,
                                  first_beat=first_beat, last_beat=last_beat))
        bars += group.length_bars

    return extract(folded, bar_beats=bar_beats, bars=bars, tempo=tempo,
                   name=name, time_signature=time_signature)
