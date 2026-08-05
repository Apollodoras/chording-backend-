"""§20.2 — the harmony's second opinion on where the bar starts.

`axis.py` records what a wrong downbeat costs: a chart laid a beat out of phase
with its own recording scored 0.768 with a *perfect* engine, and neither lint
could see it, because both check the song against itself and a uniformly shifted
chart is perfectly self-consistent. That defect came from two modules disagreeing
about beat 0. This module guards the other road to the same place — the tracker
finding the pulse and putting the "one" in the wrong spot.

Most of these tests are about **not** intervening. Beat This! scores a downbeat
F of 0.893 on the real corpus; a rule that second-guessed it casually would lose
far more than it won.
"""

from __future__ import annotations

from app.analysis.meter import reconcile
from app.analysis.types import BeatGrid, RawChordSpan

BEAT_MS = 500
BAR_MS = BEAT_MS * 4


def beats(count: int = 64) -> list[int]:
    return [i * BEAT_MS for i in range(count)]


def grid(downbeat_offset: int = 0, *, confidence: float = 0.9,
         time_signature: str = "4/4", count: int = 64) -> BeatGrid:
    """A steady 120 bpm grid whose downbeats sit `downbeat_offset` beats into
    each bar — the tracker's answer, right or wrong."""
    all_beats = beats(count)
    return BeatGrid(
        beats_ms=all_beats,
        downbeats_ms=all_beats[downbeat_offset::4],
        bpm=120.0, confidence=confidence, time_signature=time_signature,
    )


def changes_on_barlines(offset_beats: int = 0, bars: int = 16) -> list[RawChordSpan]:
    """One chord per bar, changing exactly on the barline — the harmonic
    evidence for where the "one" is. Alternating names so every boundary is a
    real change and not a repeat of one chord."""
    names = ["C", "G", "Am", "F"]
    start = offset_beats * BEAT_MS
    return [RawChordSpan(start_ms=start + i * BAR_MS, end_ms=start + (i + 1) * BAR_MS,
                         label=names[i % len(names)], confidence=0.9)
            for i in range(bars)]


# --- leaving the tracker alone -----------------------------------------------

def test_a_tracker_that_agrees_with_the_harmony_is_untouched():
    meter = reconcile(grid(0), changes_on_barlines(0))
    assert meter.phase_shift == 0
    assert meter.grid.downbeats_ms == grid(0).downbeats_ms
    assert meter.phase_evidence > 0.9


def test_too_few_chord_changes_to_hold_a_vote_changes_nothing():
    """A song with three chord changes can produce a perfect share by luck, and
    rotating a whole song on that is the confident mistake this layer exists to
    avoid."""
    meter = reconcile(grid(2), changes_on_barlines(0, bars=2))
    assert meter.phase_shift == 0


def test_a_tracker_whose_share_is_merely_imperfect_is_left_alone():
    """Both conditions have to fail before anything moves: the tracker's own
    share must be poor *and* a rotation must be decisively better. Syncopated
    material has a low share for every rotation and must not be rotated."""
    scattered = [
        RawChordSpan(start_ms=i * 700, end_ms=(i + 1) * 700,
                     label=["C", "G", "Am", "F"][i % 4], confidence=0.9)
        for i in range(20)
    ]
    meter = reconcile(grid(0), scattered)
    assert meter.phase_shift == 0


# --- the intervention it exists for ------------------------------------------

def test_a_downbeat_two_beats_out_is_rotated_back():
    """The tracker put the "one" on beat 3 of every bar; every chord change in
    the song says otherwise. This is the failure that is invisible to `lint`,
    to `lint_sync`, and to every self-consistency check there is."""
    meter = reconcile(grid(2), changes_on_barlines(0))
    assert meter.phase_shift == 2
    assert meter.grid.downbeats_ms[0] == 0
    assert meter.phase_evidence > 0.9


def test_a_rotation_moves_only_the_downbeats():
    """The pulse was never in question — only which of its beats are bar starts."""
    original = grid(1)
    meter = reconcile(original, changes_on_barlines(0))
    assert meter.grid.beats_ms == original.beats_ms
    assert meter.grid.downbeats_ms != original.downbeats_ms


# --- meter and tempo ---------------------------------------------------------

def test_the_meter_is_only_arbitrated_when_the_tracker_is_unsure():
    """A confident tracker keeps its meter even where the harmony might suggest
    another: a meter override rewrites the whole bar grid, which is a far bigger
    intervention than a rotation."""
    confident = reconcile(grid(0, confidence=0.95), changes_on_barlines(0))
    assert confident.time_signature == "4/4"
    assert confident.meter_source == "tracker"


def test_a_meter_the_axis_cannot_use_is_passed_through_untouched():
    """7/8 has no whole number of quarter-note beats, so `build_axis` returns
    None and the pipeline declines the song honestly (§13.3) rather than forcing
    it onto a grid it is not in."""
    meter = reconcile(grid(0, time_signature="7/8"), changes_on_barlines(0))
    assert meter.time_signature == "7/8"
    assert meter.phase_shift == 0


def test_an_implausible_tempo_is_reported_and_never_corrected():
    """A tempo octave error rewrites what a beat *is*, and there is no clean
    harmonic evidence for it the way there is for the phase. So it is flagged
    for the sidecar and left alone."""
    fast = BeatGrid(beats_ms=[i * 130 for i in range(64)],
                    downbeats_ms=[i * 130 for i in range(0, 64, 4)],
                    bpm=460.0, confidence=0.9, time_signature="4/4")
    meter = reconcile(fast, [])
    assert meter.tempo_octave_suspect
    assert meter.tempo == 460
    assert meter.grid.beats_ms == fast.beats_ms


def test_a_normal_tempo_is_not_flagged():
    assert not reconcile(grid(0), changes_on_barlines(0)).tempo_octave_suspect


def test_repeats_of_one_chord_do_not_vote():
    """A recognizer that flickers `C C C` across three spans has not found three
    chord changes, and letting it stuff the histogram would let engine jitter
    decide where the bar starts."""
    flicker = [RawChordSpan(start_ms=i * 250, end_ms=(i + 1) * 250, label="C",
                            confidence=0.9) for i in range(40)]
    meter = reconcile(grid(2), flicker)
    assert meter.phase_shift == 0
