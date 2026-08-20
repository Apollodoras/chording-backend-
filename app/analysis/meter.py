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

**What runs before the vote.** The vote reasons about chord-change residues
modulo the bar, so it assumes the bars are the right length — and on real
recordings they often are not. `downbeats.repair` (§20.2a) therefore runs first
and settles which of the tracker's beats are bar starts; only then is it
meaningful to ask which residue class the "one" belongs to.

Tempo and meter get a lighter touch on purpose. A meter override changes how
many beats are in a bar, and a tempo octave correction changes what a beat *is*;
both rewrite the axis wholesale, and neither has the clean "chord changes are on
barlines" evidence the phase check has. So the meter is arbitrated only when the
tracker itself reports low confidence, and the tempo octave is **reported and
not rewritten unless it is asked for** — see `Meter.tempo_octave_suspect` and
`correct_octave` below.

**Why the octave correction is opt-in.** Reporting-and-not-correcting was not
actually neutral, which is what the audit found: a tracker reading 230 bpm for a
115 bpm song is outside the range `lint` accepts, so the song was refused
outright and the player was told nothing useful — while this module was holding
the diagnosis. The correction that fixes that halves (or doubles) the whole beat
grid, which moves every bar line and every anchor in the song, and there is no
way to know from here whether that trades a refused song for a wrong one. So it
ships behind `Settings.theory_tempo_octave`, off by default, the same way
`theory_consensus` gates the only other stage that overwrites rather than
re-derives. What is *not* optional is the honest degradation: `pipeline.assemble`
now reads `tempo_octave_suspect` and either flags the song low-confidence or
fails it with a message that names the tempo.
"""

from __future__ import annotations

import logging
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import NamedTuple

from ..chords import normalize
from ..payload import PLAUSIBLE_TEMPO_MAX, PLAUSIBLE_TEMPO_MIN
from ..payload import bar_beats as parse_bar_beats
from .downbeats import DownbeatReport
from .downbeats import repair as repair_downbeats
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
# real reading, for the folk/pop material this service accepts. Defined in
# `payload.py` beside the container's own range and the pattern's, because the
# three have to nest — this one is the innermost, so anything this module calls
# plausible is something `lint` will accept.
TEMPO_MIN, TEMPO_MAX = float(PLAUSIBLE_TEMPO_MIN), float(PLAUSIBLE_TEMPO_MAX)

# **The meter vocabulary is 4/4 against 3/4, and that is a ceiling rather than an
# oversight** (F26). The trackers emit `n/4` and nothing else, so a 6/8 or 12/8
# song is approximated as one of those two — usually 3/4 with a doubled bar, or
# 4/4 with the bar length the downbeat repair measures. For a strumming app that
# is survivable and sometimes right: §14's idiom set includes "6/8 in two", and
# the anchors are absolute times taken off the recording either way, so playback
# follows the video whatever the signature *says*. What it costs is the printed
# time signature and, on a compound-time song, the sense of where beat 3 is.
# Fixing it needs a tracker that reports compound meter, not a rule here.

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
    # True when the tempo is outside the range this material plausibly occupies,
    # *after* any correction — so it means "still suspect", and a song carrying
    # it is one the pipeline degrades honestly rather than one it fixed.
    tempo_octave_suspect: bool = False
    # Octaves the beat grid was moved by: -1 halved, +1 doubled, 0 untouched
    # (the default, and the only value reachable without `correct_octave`).
    tempo_octave_shift: int = 0
    meter_source: str = "tracker"
    # What §20.2a's downbeat repair found and did, before the phase vote below
    # ever ran. Empty on a grid it declined to touch, which is also what a
    # hand-assembled `Meter` reports.
    downbeats: DownbeatReport = field(default_factory=DownbeatReport)


def reconcile(grid: BeatGrid, raw: list[RawChordSpan], *,
              correct_octave: bool = False) -> Meter:
    """A tracker's `BeatGrid` → the timing the rest of the model is built on.

    Returns the grid unchanged in every case where the evidence does not clearly
    say otherwise. `build_axis` is called on `Meter.grid`, not on the tracker's,
    so a correction here reaches the chart, the bars and the anchors together —
    there is still exactly one origin.

    `correct_octave` opts into rewriting a tempo that reads an octave out. Off by
    default: it is the one thing in here that changes what a beat *is*, and the
    diagnosis is worth reporting whether or not it is acted on.
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
        if arbitrated is not None and arbitrated.meter != bar_beats:
            # **The downbeats move with the meter, or the meter does not move.**
            # Relabelling `bar_beats` while leaving the tracker's bar starts
            # where they were is not a smaller intervention, it is a broken one:
            # every bar then holds the wrong number of beats, `axis` resamples
            # all of them, and the whole song is quantized against a grid running
            # at `arbitrated/tracked` of its real beat. Measured, a tracked 2/4
            # relabelled 4/4 gave an axis beat of 250 ms on a 500 ms song.
            rebuilt = _rebuild_downbeats(grid, arbitrated)
            if rebuilt is None:
                log.info("meter arbitration declined: %d/4 leaves too little grid to "
                         "lay bars on (tracker said %d/4)", arbitrated.meter, bar_beats)
            else:
                log.info("meter arbitrated from harmony: %d/4 phase %d "
                         "(tracker said %d/4, confidence %.2f)",
                         arbitrated.meter, arbitrated.phase, bar_beats, grid.confidence)
                grid = rebuilt
                bar_beats = arbitrated.meter
                time_signature = f"{arbitrated.meter}/4"
                source = "harmony"

    # The octave first, because it decides which beats exist: halving the grid
    # removes every other beat, and the phase vote counts residues over the beats
    # that are left. Rotating first and rescaling after would rotate a grid that
    # is about to stop being the grid.
    working = grid
    octave_shift = 0
    if correct_octave:
        rescaled = _octave(grid, bar_beats)
        if rescaled is not None:
            working, octave_shift = rescaled
            log.info("tempo octave corrected: %.1f → %.1f bpm (%s)", grid.bpm, working.bpm,
                     "halved" if octave_shift < 0 else "doubled")

    # §20.2a, and it has to run **here**: after the octave, which decides which
    # beats exist at all, and before the phase vote, which assumes the bars are
    # right — a vote taken over a grid holding half-bars is counting residues
    # modulo a bar the song does not have.
    working, downbeat_report = repair_downbeats(working, bar_beats=bar_beats)

    shift, evidence = _phase(working, raw, bar_beats)
    corrected = working
    if shift:
        rotated = _rotate(working, bar_beats, shift)
        if rotated is not None:
            log.info("downbeat phase rotated by %d beat(s); chord changes on barlines %.2f",
                     shift, evidence)
            corrected = rotated
        else:
            shift = 0

    # The tempo the *beats* run at, derived rather than carried. See
    # `_measured_bpm`: it agrees with the tracker's headline on every track
    # measured, and the point is that from here on it cannot stop agreeing.
    measured = _measured_bpm(corrected)
    if measured is not None and abs(measured - corrected.bpm) > 0.5:
        log.info("tempo measured from the grid: %.1f bpm (tracker reported %.1f)",
                 measured, corrected.bpm)

    if time_signature != corrected.time_signature or measured is not None:
        corrected = BeatGrid(
            beats_ms=corrected.beats_ms, downbeats_ms=corrected.downbeats_ms,
            bpm=measured if measured is not None else corrected.bpm,
            confidence=corrected.confidence,
            time_signature=time_signature,
        )

    suspect = bool(corrected.bpm) and not (TEMPO_MIN <= corrected.bpm <= TEMPO_MAX)
    if suspect:
        log.warning("tempo %.1f bpm is outside %.0f–%.0f — possible octave error, %s",
                    corrected.bpm, TEMPO_MIN, TEMPO_MAX,
                    "correction declined" if correct_octave else "not corrected")

    return Meter(
        grid=corrected,
        bar_beats=bar_beats,
        tempo=int(round(corrected.bpm)),
        time_signature=time_signature,
        phase_shift=shift,
        phase_evidence=round(evidence, 3),
        tempo_octave_suspect=suspect,
        tempo_octave_shift=octave_shift,
        meter_source=source,
        downbeats=downbeat_report,
    )


# --- the tempo the beats actually run at -------------------------------------

def _measured_bpm(grid: BeatGrid) -> float | None:
    """The song's tempo, read off its own beat list.

    Not a correction — on the ten-song bench corpus this reproduces
    `BeatGrid.bpm` to two decimal places on every track, including the four whose
    grids are otherwise poor. What it buys is that the number the payload
    publishes is **derived from the timeline the chart is actually on** rather
    than carried alongside it, so the two cannot drift apart:

    - `axis.py` lays its timeline on `beats_ms`, the bars are sliced out of that
      axis, and the anchors are read off it. Everything internal already uses the
      beats. The client is the one consumer that does not — it derives its whole
      bar grid from the single song-level `tempo` (`JamSongSheet.from`, and see
      `lint_sync` on why an override is forbidden) — so that number has to be a
      statement about the same grid.
    - `_halve` and `_double` already rewrite `bpm` by hand alongside the beats
      they thin or thicken. Deriving it removes the chance of those two lines
      disagreeing, which is the kind of arithmetic that is right until it isn't.

    The **median** interval, not the mean, and the corpus is emphatic about why:
    a tracker emits no beats over material that has no pulse, and the mean sees
    that as tempo. Smooth Criminal's grid contains a single **51-second** gap
    (the short film's spoken opening), Country Roads has a rubato intro, and Don't
    Stop Believin' has an 80 ms stutter. Their mean beat rates are 91, 149 and 130
    against true rates of 115, 167 and 120; every one of their medians is exact.

    `None` when there is not enough grid to measure, which leaves the tracker's
    own answer standing — the only other answer available.
    """
    beats = sorted(set(int(t) for t in grid.beats_ms))
    if len(beats) < 8:
        return None
    gaps = sorted(b - a for a, b in zip(beats, beats[1:]) if b > a)
    if not gaps:
        return None
    middle = len(gaps) // 2
    interval = (gaps[middle] if len(gaps) % 2
                else (gaps[middle - 1] + gaps[middle]) / 2)
    return round(60000.0 / interval, 2) if interval > 0 else None


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


def _nearest_beat(beats: list[int], t: int) -> int | None:
    """Index of the beat `t` sits on, or None when it sits on none of them.

    "Sits on" is `SNAP_TOLERANCE_MS`, not "is nearest to": a time that falls
    between beats is evidence about nothing, and letting it snap to a neighbour
    anyway would let syncopation vote on where the bar starts.
    """
    index = bisect_left(beats, t)
    candidates = [i for i in (index - 1, index) if 0 <= i < len(beats)]
    if not candidates:
        return None
    nearest = min(candidates, key=lambda i: abs(beats[i] - t))
    return nearest if abs(beats[nearest] - t) <= SNAP_TOLERANCE_MS else None


def _residues(beats: list[int], changes: list[int], bar_beats: int) -> list[int]:
    """Each chord change → which beat of the bar it landed on, or dropped."""
    out: list[int] = []
    for t in changes:
        index = _nearest_beat(beats, t)
        if index is not None:
            out.append(index % bar_beats)
    return out


def _tracker_residue(beats: list[int], downbeats: list[int], bar_beats: int) -> int | None:
    """Which residue class the tracker's own downbeats occupy.

    None when its downbeats don't sit on its own beats consistently, in which
    case there is nothing to compare a rotation against and the phase check
    declines to run.
    """
    counts: dict[int, int] = {}
    for t in downbeats:
        index = _nearest_beat(beats, t)
        if index is not None:
            counts[index % bar_beats] = counts.get(index % bar_beats, 0) + 1
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
    """Move each of the tracker's downbeats `shift` beats later in its beat list.

    The beats themselves are untouched — only which of them are called bar
    starts. Returns None if the rotation would not leave enough bars to build an
    axis on, in which case the caller keeps the tracker's answer.

    **Each downbeat is moved relative to itself**, which is the whole point.
    Re-deriving the grid as `beats[start::bar_beats]` instead assumes the beat
    list is metrically uniform, and a real one is not: one inserted or dropped
    beat — Here Comes The Sun has 11/8 and 15/8 bars inside a 4/4 song — shifts
    every later downbeat by one, throwing away exactly what a downbeat-aware
    tracker is for. Rotating in place keeps the tracker's bar structure and
    changes only its phase.

    The head is the one place the tracker's own answer runs out: rotating forward
    means the music before the first rotated downbeat belongs to no bar. Bars are
    extended back over it while there are whole bars' worth of beats to cover,
    which is the same coverage the old slice produced on a uniform grid, bounded
    to the run of beats before the first bar the tracker actually found.
    """
    beats = sorted(set(int(t) for t in grid.beats_ms))
    downbeats = sorted(set(int(t) for t in grid.downbeats_ms))

    rotated_indices: list[int] = []
    for t in downbeats:
        index = _nearest_beat(beats, t)
        if index is None:
            continue
        if index + shift < len(beats):
            rotated_indices.append(index + shift)
    if not rotated_indices:
        return None

    head = rotated_indices[0] - bar_beats
    while head >= 0:
        rotated_indices.insert(0, head)
        head -= bar_beats

    rotated = sorted({beats[i] for i in rotated_indices})
    if len(rotated) < 3:
        return None
    return BeatGrid(
        beats_ms=grid.beats_ms, downbeats_ms=rotated, bpm=grid.bpm,
        confidence=grid.confidence, time_signature=grid.time_signature,
    )


def _rebuild_downbeats(grid: BeatGrid, vote: _MeterVote) -> BeatGrid | None:
    """The grid's bar starts, re-derived for a meter the harmony just chose.

    `_rotate` argues at length against deriving downbeats as
    `beats[start::bar_beats]`, and it is right — *for the case it is in*. There
    the tracker's bar structure is being kept and only its phase changed, so a
    song with an 11/8 bar in the middle of it must not have every later downbeat
    dragged one beat off by a uniform slice.

    Here the premise is the opposite: the meter itself was overruled, so the
    tracker's bar starts are bars of the wrong length and there is no structure
    left to preserve. What there is instead is the hypothesis that just won —
    "changes land on residue `phase` of an `meter`-beat bar" — and that
    hypothesis *is* a uniform slice. Scoring one grid and then shipping a
    different one is how the two came apart in the first place, so the grid
    returned here is the one the score was measured on.

    Returns None when the slice leaves too little to build an axis on, in which
    case the caller keeps the tracker's meter as well as its downbeats.
    """
    beats = sorted({int(t) for t in grid.beats_ms})
    if len(beats) < vote.meter * 2:
        return None
    downbeats = beats[vote.phase % vote.meter::vote.meter]
    if len(downbeats) < 3:
        return None
    return BeatGrid(
        beats_ms=grid.beats_ms, downbeats_ms=downbeats, bpm=grid.bpm,
        confidence=grid.confidence, time_signature=f"{vote.meter}/4",
    )


# --- the tempo octave -------------------------------------------------------

def _octave(grid: BeatGrid, bar_beats: int) -> tuple[BeatGrid, int] | None:
    """Halve or double the beat grid when the tempo reads an octave out.

    Returns None — change nothing — unless a **single** octave lands the tempo
    inside the plausible band. More than an octave out is not an octave error,
    it is a tracker that failed, and quartering a beat grid on the strength of a
    number no music has is the confident mistake this module exists to avoid.

    A halved grid is a different song structurally, not just a slower one: two of
    the tracker's bars become one, so the downbeats are thinned with the beats
    and each surviving bar keeps `bar_beats` of them. Doubling is the same
    operation backwards — a beat between every pair of beats, and a downbeat in
    the middle of every bar, because an old bar is now two.
    """
    bpm = grid.bpm
    if not bpm:
        return None
    if bpm > TEMPO_MAX and TEMPO_MIN <= bpm / 2 <= TEMPO_MAX:
        rescaled = _halve(grid, bar_beats)
        return (rescaled, -1) if rescaled is not None else None
    if bpm < TEMPO_MIN and TEMPO_MIN <= bpm * 2 <= TEMPO_MAX:
        rescaled = _double(grid, bar_beats)
        return (rescaled, 1) if rescaled is not None else None
    return None


def _halve(grid: BeatGrid, bar_beats: int) -> BeatGrid | None:
    """Every other beat, and every other bar line.

    Walked bar by bar rather than sliced, for the same reason `_rotate` is: the
    tracker's bars are not all the same length, and a slice off a fixed origin
    would drift out of phase with them at the first irregular one.
    """
    beats = sorted(set(int(t) for t in grid.beats_ms))
    downbeats = sorted(set(int(t) for t in grid.downbeats_ms))
    kept_downbeats = downbeats[::2]
    if len(kept_downbeats) < 3:
        return None

    kept_beats: list[int] = []
    for start, end in zip(kept_downbeats, kept_downbeats[1:]):
        inside = [t for t in beats if start <= t < end]
        kept_beats.extend(inside[::2])
    tail = [t for t in beats if t >= kept_downbeats[-1]]
    kept_beats.extend(tail[::2])
    if len(kept_beats) < bar_beats * 2:
        return None

    return BeatGrid(
        beats_ms=sorted(set(kept_beats)), downbeats_ms=kept_downbeats,
        bpm=grid.bpm / 2, confidence=grid.confidence,
        time_signature=grid.time_signature,
    )


def _double(grid: BeatGrid, bar_beats: int) -> BeatGrid | None:
    """A beat between every pair of beats, and a bar line every `bar_beats` of
    the beats that result — so each of the tracker's bars becomes two."""
    beats = sorted(set(int(t) for t in grid.beats_ms))
    downbeats = sorted(set(int(t) for t in grid.downbeats_ms))
    if len(beats) < 2 or len(downbeats) < 2:
        return None

    dense: list[int] = []
    for left, right in zip(beats, beats[1:]):
        dense.extend((left, (left + right) // 2))
    dense.append(beats[-1])

    new_downbeats: list[int] = []
    for start, end in zip(downbeats, downbeats[1:]):
        inside = [t for t in dense if start <= t < end]
        new_downbeats.extend(inside[::bar_beats])
    new_downbeats.append(downbeats[-1])
    if len(new_downbeats) < 3:
        return None

    return BeatGrid(
        beats_ms=dense, downbeats_ms=sorted(set(new_downbeats)),
        bpm=grid.bpm * 2, confidence=grid.confidence,
        time_signature=grid.time_signature,
    )


# --- meter arbitration ------------------------------------------------------

class _MeterVote(NamedTuple):
    """A candidate meter, where its "one" falls, and how well it explains the
    chord changes. `phase` is a residue class over the beat list — the same
    coordinate `_residues` counts in — so `beats[phase::meter]` is exactly the
    barline hypothesis `score` was measured on."""

    meter: int
    phase: int
    score: float


def _meter_from_harmony(grid: BeatGrid, raw: list[RawChordSpan],
                        tracked: int) -> _MeterVote | None:
    """Pick the meter whose barlines the chord changes best respect.

    Only consulted when the tracker reports low confidence. Returns None when the
    evidence is too thin to prefer anything, which leaves the tracker's answer
    standing.

    **The tracker's own meter is scored, not assumed to be zero.** It used to be
    absent from the candidate list, so `scores.get(tracked, 0.0)` was 0.0 for any
    tracker reading outside `CANDIDATE_METERS` and *any* candidate clearing
    `PHASE_MARGIN` on its own beat it by default. A genuine 2/4 — which
    `beat_this` does emit — has its chord changes on every second beat, which
    lands them on residue 0 of a 4-beat bar half the time and hands meter 4 a
    score of 0.5 automatically. Cut-time and marches were being overruled by
    evidence that agreed with them.

    **Scored against chance, not raw.** A share of the changes falling in the
    best of `m` residue classes is worth less the larger `m` is: spread the
    changes uniformly and meter 3 scores 1/3 while meter 4 scores 1/4, so the
    raw comparison carried a standing bias toward the shorter bar. What each
    candidate is charged with explaining is its excess over chance.
    """
    beats = sorted(set(int(t) for t in grid.beats_ms))
    changes = _changes_ms(raw)
    if len(beats) < 8 or len(changes) < 8:
        return None

    votes: dict[int, _MeterVote] = {}
    for candidate in dict.fromkeys((*CANDIDATE_METERS, tracked)):
        if candidate < 2:
            continue
        residues = _residues(beats, changes, candidate)
        if not residues:
            continue
        counts = [residues.count(r) / len(residues) for r in range(candidate)]
        phase = max(range(candidate), key=lambda r: (counts[r], -r))
        votes[candidate] = _MeterVote(meter=candidate, phase=phase,
                                      score=counts[phase] - 1.0 / candidate)

    if not votes:
        return None
    best = max(votes.values(), key=lambda v: (v.score, v.meter == tracked, -v.meter))
    # A meter change is a big intervention; require it to be clearly better than
    # the tracker's, not merely different.
    if best.meter != tracked and tracked in votes:
        if best.score - votes[tracked].score < PHASE_MARGIN:
            return votes[tracked]
    return best
