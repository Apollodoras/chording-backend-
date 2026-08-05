"""§20.2 — the song's timing properties, reconciled against the harmony.

The beat tracker is the only witness to tempo and meter, and on most tracks it
is a good one. But it is a *rhythmic* witness making a partly *harmonic* claim,
and where the two kinds of evidence disagree the harmonic one is often right.
This module is the second opinion.

**Why the phase check exists.** `axis.py` records what a wrong downbeat costs:
a chart laid a beat or two out of phase with its own recording scored 0.768 with
a perfect engine, and neither `lint` nor `lint_sync` could see it, because both
check the song against itself and a uniformly shifted chart is perfectly
self-consistent. That defect came from two modules disagreeing about beat 0 and
is fixed. This is the *other* way to arrive at the same failure: both modules
agree, and the tracker put the "one" in the wrong place to begin with.

The second opinion is free and it is strong. Chord changes overwhelmingly land
on barlines — that is most of what a barline *is* in this repertoire — so the
residue of chord-change positions modulo the bar is a vote on where the "one"
belongs. If the tracker's downbeats collect most of that mass, it was right. If
some other rotation collects it instead, and the tracker's own share is poor,
the tracker found the pulse and missed the bar.

**The gate is deliberately strict, in both directions.** Beat This! scores a
downbeat F of 0.893 on the real corpus; a rule that second-guessed it casually
would lose more than it won. So a rotation is applied only when the tracker's
share is genuinely poor *and* the challenger is decisively better. Everything
else — including every case where the two are merely close — leaves the
tracker's answer alone.

Tempo and meter get a lighter touch on purpose. A meter override changes how
many beats are in a bar, and a tempo octave correction changes what a beat *is*;
both rewrite the axis wholesale, and neither has the clean "chord changes are on
barlines" evidence the phase check has. So the meter is arbitrated only when the
tracker itself reports low confidence, and the tempo octave is **reported and
never rewritten** — see `Meter.tempo_octave_suspect`.
"""

from __future__ import annotations

import logging
from bisect import bisect_left
from dataclasses import dataclass

from ..chords import normalize
from ..payload import bar_beats as parse_bar_beats
from .types import BeatGrid, RawChordSpan

log = logging.getLogger("chords.meter")

# A chord change must land within this of a beat to count as being "on" it. One
# beat's worth of tolerance would make every rotation score identically; this is
# tight enough that only changes genuinely on the pulse vote.
SNAP_TOLERANCE_MS = 90

# The tracker keeps its downbeats unless *both* of these fail: it collects less
# than FLOOR of the chord-change mass, and some rotation beats it by MARGIN.
# Two independent conditions because either alone is too easy to trip — a song
# with very few chord changes has a noisy histogram, and a song whose changes are
# genuinely mid-bar (syncopated soul, say) has a low share for everyone.
PHASE_FLOOR = 0.45
PHASE_MARGIN = 0.15

# Below/above these the reported tempo is more likely an octave error than a
# real reading, for the folk/pop material this service accepts. Reported only.
TEMPO_MIN, TEMPO_MAX = 55.0, 200.0

# Meters worth arbitrating between when the tracker is unsure. Not a general
# meter finder: these are the two that cover almost everything here, and a wrong
# answer between them is recoverable in a way that inventing 7/8 is not.
CANDIDATE_METERS = (4, 3)


@dataclass(frozen=True)
class Meter:
    """The timing half of the song model, and how much of it we chose."""

    grid: BeatGrid
    bar_beats: int
    tempo: int
    time_signature: str
    # Beats the downbeat grid was rotated by. 0 means the tracker's own answer
    # was kept, which is the overwhelmingly common case.
    phase_shift: int = 0
    # Share of chord changes landing on a barline, after any rotation. This is
    # the number that says how well the harmony and the pulse agree, and it is
    # worth logging even when nothing was changed.
    phase_evidence: float = 0.0
    # True when the tempo is outside the range this material plausibly occupies.
    # Never acted on — see the module docstring.
    tempo_octave_suspect: bool = False
    meter_source: str = "tracker"


def reconcile(grid: BeatGrid, raw: list[RawChordSpan]) -> Meter:
    """A tracker's `BeatGrid` → the timing the rest of the model is built on.

    Returns the grid unchanged in every case where the evidence does not clearly
    say otherwise. `build_axis` is called on `Meter.grid`, not on the tracker's,
    so a correction here reaches the chart, the bars and the anchors together —
    there is still exactly one origin.
    """
    beats = parse_bar_beats(grid.time_signature)
    bar_beats = int(round(beats)) if beats else 4
    if not beats or bar_beats < 1 or abs(beats - bar_beats) > 1e-6:
        # A meter the axis can't lay bars over. Left alone: `build_axis` returns
        # None and the pipeline declines the song honestly (§13.3) rather than
        # forcing it into a grid it isn't in.
        return Meter(grid=grid, bar_beats=max(1, bar_beats), tempo=int(round(grid.bpm)),
                     time_signature=grid.time_signature)

    time_signature = grid.time_signature
    source = "tracker"
    if grid.confidence < 0.5:
        arbitrated = _meter_from_harmony(grid, raw, bar_beats)
        if arbitrated is not None and arbitrated != bar_beats:
            log.info("meter arbitrated from harmony: %d/4 (tracker said %d/4, confidence %.2f)",
                     arbitrated, bar_beats, grid.confidence)
            bar_beats = arbitrated
            time_signature = f"{arbitrated}/4"
            source = "harmony"

    shift, evidence = _phase(grid, raw, bar_beats)
    corrected = grid
    if shift:
        rotated = _rotate(grid, bar_beats, shift)
        if rotated is not None:
            log.info("downbeat phase rotated by %d beat(s); chord changes on barlines %.2f",
                     shift, evidence)
            corrected = rotated
        else:
            shift = 0

    if time_signature != corrected.time_signature:
        corrected = BeatGrid(
            beats_ms=corrected.beats_ms, downbeats_ms=corrected.downbeats_ms,
            bpm=corrected.bpm, confidence=corrected.confidence,
            time_signature=time_signature,
        )

    suspect = bool(corrected.bpm) and not (TEMPO_MIN <= corrected.bpm <= TEMPO_MAX)
    if suspect:
        log.warning("tempo %.1f bpm is outside %.0f–%.0f — possible octave error, not corrected",
                    corrected.bpm, TEMPO_MIN, TEMPO_MAX)

    return Meter(
        grid=corrected,
        bar_beats=bar_beats,
        tempo=int(round(corrected.bpm)),
        time_signature=time_signature,
        phase_shift=shift,
        phase_evidence=round(evidence, 3),
        tempo_octave_suspect=suspect,
        meter_source=source,
    )


# --- the phase vote ---------------------------------------------------------

def _changes_ms(raw: list[RawChordSpan]) -> list[int]:
    """Chord-change times, ignoring no-chord labels and repeats of one chord.

    A boundary between two spans the engine gave the same name is not a change
    and must not vote — a recognizer that flickers `C C C` across three frames
    would otherwise stuff the histogram with events the music never had.
    """
    changes: list[int] = []
    previous: tuple[int, str] | None = None
    for span in sorted(raw, key=lambda s: s.start_ms):
        parsed = normalize(span.label)
        if parsed is None:
            continue
        chord = (parsed[0], parsed[1])
        if previous is not None and chord != previous:
            changes.append(span.start_ms)
        previous = chord
    return changes


def _residues(beats: list[int], changes: list[int], bar_beats: int) -> list[int]:
    """Each chord change → which beat of the bar it landed on, or dropped.

    Dropped when it is not close to any beat at all: a change that happens
    between beats is evidence about nothing, and letting it snap to a neighbour
    would let syncopation vote on the downbeat.
    """
    out: list[int] = []
    for t in changes:
        index = bisect_left(beats, t)
        candidates = [i for i in (index - 1, index) if 0 <= i < len(beats)]
        if not candidates:
            continue
        nearest = min(candidates, key=lambda i: abs(beats[i] - t))
        if abs(beats[nearest] - t) <= SNAP_TOLERANCE_MS:
            out.append(nearest % bar_beats)
    return out


def _tracker_residue(beats: list[int], downbeats: list[int], bar_beats: int) -> int | None:
    """Which residue class the tracker's own downbeats occupy.

    None when its downbeats don't sit on its own beats consistently, in which
    case there is nothing to compare a rotation against and the phase check
    declines to run.
    """
    counts: dict[int, int] = {}
    for t in downbeats:
        index = bisect_left(beats, t)
        candidates = [i for i in (index - 1, index) if 0 <= i < len(beats)]
        if not candidates:
            continue
        nearest = min(candidates, key=lambda i: abs(beats[i] - t))
        if abs(beats[nearest] - t) <= SNAP_TOLERANCE_MS:
            counts[nearest % bar_beats] = counts.get(nearest % bar_beats, 0) + 1
    if not counts:
        return None
    best = max(counts.values())
    # Ties resolve to the lowest residue so the result is deterministic.
    return min(r for r, c in counts.items() if c == best)


def _phase(grid: BeatGrid, raw: list[RawChordSpan], bar_beats: int) -> tuple[int, float]:
    """How far to rotate the downbeat grid, and the winning share.

    Returns `(0, share)` — change nothing — unless the tracker's own share is
    below `PHASE_FLOOR` *and* a rotation beats it by `PHASE_MARGIN`.
    """
    beats = sorted(set(int(t) for t in grid.beats_ms))
    if len(beats) < bar_beats * 2 or bar_beats < 2:
        return 0, 0.0

    residues = _residues(beats, _changes_ms(raw), bar_beats)
    # Too few changes to hold a vote. A song with three chord changes in it can
    # produce a 1.0 share by luck, and rotating a whole song on that is exactly
    # the kind of confident mistake this layer exists to avoid.
    if len(residues) < bar_beats * 2:
        return 0, 0.0

    counts = [residues.count(r) / len(residues) for r in range(bar_beats)]
    tracker = _tracker_residue(beats, sorted(set(int(t) for t in grid.downbeats_ms)), bar_beats)
    if tracker is None:
        return 0, 0.0

    best = max(range(bar_beats), key=lambda r: (counts[r], -r))
    if counts[tracker] >= PHASE_FLOOR or counts[best] - counts[tracker] < PHASE_MARGIN:
        return 0, counts[tracker]
    return (best - tracker) % bar_beats, counts[best]


def _rotate(grid: BeatGrid, bar_beats: int, shift: int) -> BeatGrid | None:
    """Re-derive the downbeats `shift` beats later in the tracker's beat list.

    The beats themselves are untouched — only which of them are called bar
    starts. Returns None if the rotation would not leave enough bars to build an
    axis on, in which case the caller keeps the tracker's answer.
    """
    beats = sorted(set(int(t) for t in grid.beats_ms))
    downbeats = sorted(set(int(t) for t in grid.downbeats_ms))
    first = _tracker_residue(beats, downbeats, bar_beats)
    if first is None:
        return None
    start = (first + shift) % bar_beats
    rotated = beats[start::bar_beats]
    if len(rotated) < 3:
        return None
    return BeatGrid(
        beats_ms=grid.beats_ms, downbeats_ms=rotated, bpm=grid.bpm,
        confidence=grid.confidence, time_signature=grid.time_signature,
    )


# --- meter arbitration ------------------------------------------------------

def _meter_from_harmony(grid: BeatGrid, raw: list[RawChordSpan], tracked: int) -> int | None:
    """Pick the meter whose barlines the chord changes best respect.

    Only consulted when the tracker reports low confidence, and only between 4
    and 3 — the two that cover this repertoire. Returns None when the evidence
    is too thin to prefer either, which leaves the tracker's answer standing.
    """
    beats = sorted(set(int(t) for t in grid.beats_ms))
    changes = _changes_ms(raw)
    if len(beats) < 8 or len(changes) < 8:
        return None

    scores: dict[int, float] = {}
    for candidate in CANDIDATE_METERS:
        residues = _residues(beats, changes, candidate)
        if not residues:
            continue
        counts = [residues.count(r) / len(residues) for r in range(candidate)]
        scores[candidate] = max(counts)

    if not scores:
        return None
    best = max(scores, key=lambda m: (scores[m], m == tracked, -m))
    # A meter change is a big intervention; require it to be clearly better than
    # the tracker's, not merely different.
    if best != tracked and scores[best] - scores.get(tracked, 0.0) < PHASE_MARGIN:
        return tracked
    return best
