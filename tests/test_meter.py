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


def irregular_song(*, early_beats: int = 2, bars: int = 16, long_bar: int = 6):
    """A 4/4 song with one 5-beat bar, tracked by a tracker that found the bars
    and put the "one" `early_beats` too early in each of them.

    The irregular bar is the whole point. `axis.py` records that Here Comes The
    Sun has 11/8 and 15/8 bars inside a 4/4 song, and a downbeat-aware tracker
    survives that — it reports the bar starts it heard, whatever is between them.
    Re-deriving the grid from a fixed origin does not: after one inserted beat
    every later "downbeat" is a beat out.

    Returns the tracker's grid, the chord changes (on the *real* barlines), and
    the real barlines.
    """
    lengths = [4] * bars
    lengths[long_bar] = 5
    starts: list[int] = []
    cursor = 0
    for length in lengths:
        starts.append(cursor)
        cursor += length
    all_beats = [i * BEAT_MS for i in range(cursor + 1)]

    tracked = [all_beats[s - early_beats] for s in starts if s - early_beats >= 0]
    tracker = BeatGrid(beats_ms=all_beats, downbeats_ms=tracked,
                       bpm=120.0, confidence=0.9, time_signature="4/4")

    names = ["C", "G", "Am", "F"]
    changes = [
        RawChordSpan(start_ms=all_beats[start], end_ms=all_beats[start] + lengths[i] * BEAT_MS,
                     label=names[i % len(names)], confidence=0.9)
        for i, start in enumerate(starts)
    ]
    return tracker, changes, [all_beats[s] for s in starts]


def test_a_rotation_survives_an_irregular_bar():
    """Every downbeat moves relative to *itself*, not to a re-derived grid.

    `beats[start::bar_beats]` assumes a metrically uniform beat list. One 5-beat
    bar and every later bar line lands a beat off the music — the tail of the
    song silently corrupted by a correction that was right about the phase.
    """
    tracker, changes, real_barlines = irregular_song()
    meter = reconcile(tracker, changes)
    assert meter.phase_shift == 2
    assert meter.grid.downbeats_ms == real_barlines


def test_a_rotation_still_covers_the_music_before_the_first_bar_it_moved():
    """Rotating forward leaves the head of the recording in no bar at all, so
    bars are extended back over it — the coverage the old fixed-origin grid had,
    kept without the assumption that produced it."""
    original = grid(2)
    meter = reconcile(original, changes_on_barlines(0))
    assert meter.grid.downbeats_ms[0] == 0
    assert meter.grid.downbeats_ms == beats()[0::4][:len(meter.grid.downbeats_ms)]


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


def steady(bpm: float, count: int = 64) -> BeatGrid:
    """A steady grid at `bpm`, 4/4, downbeats every fourth beat."""
    spacing = int(round(60_000 / bpm))
    all_beats = [i * spacing for i in range(count)]
    return BeatGrid(beats_ms=all_beats, downbeats_ms=all_beats[::4],
                    bpm=bpm, confidence=0.9, time_signature="4/4")


def test_an_implausible_tempo_is_reported_and_never_corrected():
    """A tempo octave error rewrites what a beat *is*, and there is no clean
    harmonic evidence for it the way there is for the phase. So by default it is
    flagged for the sidecar and left alone — the pipeline degrades on the flag
    rather than this module acting on it."""
    fast = BeatGrid(beats_ms=[i * 130 for i in range(64)],
                    downbeats_ms=[i * 130 for i in range(0, 64, 4)],
                    bpm=460.0, confidence=0.9, time_signature="4/4")
    meter = reconcile(fast, [])
    assert meter.tempo_octave_suspect
    # 462 and not the 460 the fixture's own header claims: the beats are 130 ms
    # apart, which is 461.5 bpm, and the tempo is measured off them. See
    # `test_the_tempo_is_measured_off_the_beats_not_taken_from_the_header`.
    assert meter.tempo == 462
    assert meter.grid.beats_ms == fast.beats_ms
    assert meter.tempo_octave_shift == 0


def test_the_tempo_is_measured_off_the_beats_not_taken_from_the_header():
    """`BeatGrid.bpm` is the tracker's headline estimate; `beats_ms` is what it
    emitted. They agree on every track in the bench corpus — this is not a
    correction — and the guarantee being bought is that they cannot come apart.

    The payload carries one song-level tempo and the client derives its whole bar
    grid from it (`JamSongSheet.from`), while every internal stage is built on the
    beats. Two numbers for one grid is a divergence waiting to happen, and
    `_halve`/`_double` already maintain the second one by hand.
    """
    lying = BeatGrid(beats_ms=[i * 500 for i in range(64)],       # 120 bpm
                     downbeats_ms=[i * 500 for i in range(0, 64, 4)],
                     bpm=90.0, confidence=0.9, time_signature="4/4")
    meter = reconcile(lying, [])
    assert meter.tempo == 120
    assert meter.grid.bpm == 120.0, "the sidecar reads this, and it must agree"
    assert meter.grid.beats_ms == lying.beats_ms, "measuring must not move a beat"


def test_a_dropped_beat_does_not_drag_the_measured_tempo():
    """The median interval, not the mean. A tracker emits no beats over material
    with no pulse, and the mean reads that silence as tempo: Smooth Criminal's
    grid has one **51-second** gap in it (the short film's spoken opening), and
    its mean beat rate is 91 bpm against a true 115. The median is exact."""
    beats = [i * 500 for i in range(64)]
    beats.remove(31 * 500)
    holed = BeatGrid(beats_ms=beats, downbeats_ms=beats[::4],
                     bpm=120.0, confidence=0.9, time_signature="4/4")
    assert reconcile(holed, []).tempo == 120


def test_a_normal_tempo_is_not_flagged():
    assert not reconcile(grid(0), changes_on_barlines(0)).tempo_octave_suspect


def test_a_doubled_tempo_is_halved_when_the_correction_is_asked_for():
    """230 bpm is the tracker counting the eighths of a 115 bpm song. Halving it
    is not a slower tempo, it is a different bar: two of the tracker's bars are
    one, so the downbeats thin out with the beats."""
    fast = steady(230.0)
    meter = reconcile(fast, [], correct_octave=True)
    assert meter.tempo == 115
    assert meter.tempo_octave_shift == -1
    assert not meter.tempo_octave_suspect
    assert meter.grid.beats_ms == fast.beats_ms[::2]
    assert meter.grid.downbeats_ms == fast.beats_ms[::8]


def test_a_halved_tempo_is_doubled_when_the_correction_is_asked_for():
    """The same operation backwards: a beat between every pair of beats, and a
    bar line in the middle of every bar, because one of them is now two."""
    slow = steady(45.0)
    meter = reconcile(slow, [], correct_octave=True)
    assert meter.tempo == 90
    assert meter.tempo_octave_shift == 1
    assert not meter.tempo_octave_suspect
    assert meter.grid.beats_ms[::2] == slow.beats_ms
    assert slow.downbeats_ms[1] in meter.grid.downbeats_ms
    # A new bar line halfway through each of the tracker's bars.
    assert len(meter.grid.downbeats_ms) == 2 * len(slow.downbeats_ms) - 1


def test_a_tempo_more_than_an_octave_out_is_left_alone_even_then():
    """460 bpm halves to 230, which is still not a tempo. That is not an octave
    error, it is a tracker that failed, and quartering a beat grid on the
    strength of it is exactly the confident mistake this layer avoids."""
    wild = steady(460.0)
    meter = reconcile(wild, [], correct_octave=True)
    assert meter.tempo == 462     # measured; `steady` rounds 460 bpm to a 130 ms beat
    assert meter.tempo_octave_shift == 0
    assert meter.tempo_octave_suspect
    assert meter.grid.beats_ms == wild.beats_ms


def test_a_plausible_tempo_is_never_rescaled():
    meter = reconcile(grid(0), changes_on_barlines(0), correct_octave=True)
    assert meter.tempo == 120
    assert meter.tempo_octave_shift == 0
    assert meter.grid.downbeats_ms == grid(0).downbeats_ms


def test_repeats_of_one_chord_do_not_vote():
    """A recognizer that flickers `C C C` across three spans has not found three
    chord changes, and letting it stuff the histogram would let engine jitter
    decide where the bar starts."""
    flicker = [RawChordSpan(start_ms=i * 250, end_ms=(i + 1) * 250, label="C",
                            confidence=0.9) for i in range(40)]
    meter = reconcile(grid(2), flicker)
    assert meter.phase_shift == 0
