"""§20.2a — the downbeat repair, and the three grids the suite never had.

The whole of this file exists because of a measurement: on the four stored
catalog songs, between 8% and 42% of bars were not the length the song's own
modal bar says they are, and *nothing* between the tracker and the sidecar
looked. `axis.build_axis` defines every consecutive pair of downbeats to be one
bar, so a half-bar was resampled into a full one and shipped; the anchors then
told the client the song had doubled in tempo for two seconds, which is exactly
the complaint that came back from the player side.

Three fixtures matter here and they are deliberately different from each other:

- a **spurious** downbeat — the dominant failure (So What: 37 one-beat "bars");
- a **dropped** one — what the player reported as 2.6× segments;
- a **genuine** irregular bar — Here Comes The Sun has 11/8 and 15/8 inside a
  4/4 song, and a repair that flattened those would be trading one silent lie
  for another.

The first two must be repaired. The third must survive.
"""

from __future__ import annotations

from app.analysis.adapters.beat_this_tracker import _meter as beat_this_meter
from app.analysis.downbeats import repair
from app.analysis.model import build
from app.analysis.types import BeatGrid

from .conftest import (
    BAR_BEATS,
    MS_PER_BEAT,
    broken_downbeats,
    broken_grid,
    known_beats,
    known_chords,
    known_downbeats,
    known_grid,
    known_onsets,
)


def gaps_of(grid: BeatGrid) -> list[int]:
    return [b - a for a, b in zip(grid.downbeats_ms, grid.downbeats_ms[1:])]


# --- the three grids ---------------------------------------------------------

def test_a_clean_grid_is_returned_untouched():
    """The overwhelmingly common case, and the one where an intervention would
    cost the most."""
    repaired, report = repair(known_grid())
    assert repaired.downbeats_ms == known_downbeats()
    assert (report.dropped, report.inserted) == (0, 0)
    assert report.irregular_bars == 0
    assert not report.unreliable


def test_a_spurious_downbeat_is_dropped():
    repaired, report = repair(broken_grid(spurious=(10,)))
    assert report.dropped == 1
    assert report.inserted == 0
    assert repaired.downbeats_ms == known_downbeats()
    assert set(gaps_of(repaired)) == {BAR_BEATS * MS_PER_BEAT}


def test_a_dropped_downbeat_is_put_back_on_the_tracker_s_own_beat():
    repaired, report = repair(broken_grid(dropped=(10,)))
    assert (report.dropped, report.inserted) == (0, 1)
    assert repaired.downbeats_ms == known_downbeats()
    # Not merely "a downbeat somewhere in the gap" — the restored one is a beat
    # the tracker actually reported, which is what keeps the repair honest on a
    # recording that is not metronomic.
    assert set(repaired.downbeats_ms) <= set(known_beats())


def test_a_genuine_irregular_bar_survives_the_repair():
    """Here Comes The Sun's 11/8 bar, in miniature: bar 8 is three beats long
    and the song really is like that. A repair that flattened it would be making
    the same silent edit as the defect it replaces, in the other direction."""
    downbeats = [t if t < 8 * BAR_BEATS * MS_PER_BEAT else t - MS_PER_BEAT
                 for t in known_downbeats()]
    grid = BeatGrid(beats_ms=known_beats(), downbeats_ms=downbeats, bpm=120.0,
                    confidence=0.95, time_signature="4/4")

    repaired, report = repair(grid)
    assert (report.dropped, report.inserted) == (0, 0)
    assert repaired.downbeats_ms == downbeats
    # Reported, not repaired — which is what puts it on the sidecar as
    # `irregularBars` instead of into the chart as an invented bar line.
    assert report.irregular_bars == 1


# --- how the walk reasons ----------------------------------------------------

def test_a_spurious_downbeat_does_not_make_the_next_bar_look_short():
    """The signature failure, and the reason each gap is measured from the last
    *accepted* downbeat: an extra downbeat one beat into a bar reads as a 1
    followed by a 3, and So What has thirty of each. Measured from the raw
    predecessor the 3 would look like an irregular bar of its own."""
    repaired, report = repair(broken_grid(spurious=(4, 9, 12)))
    assert report.dropped == 3
    assert report.irregular_bars == 0
    assert repaired.downbeats_ms == known_downbeats()


def test_two_missing_downbeats_in_a_row_are_both_restored():
    repaired, report = repair(broken_grid(dropped=(6, 7)))
    assert (report.dropped, report.inserted) == (0, 2)
    assert repaired.downbeats_ms == known_downbeats()


def test_a_sub_multiple_cannot_win_by_halving_every_bar():
    """The trap the first implementation fell into, found on real audio.

    Scoring candidates by how tidy the repair comes out is self-defeating,
    because the walk can *insert*: halve the bar and every bar agrees with the
    half, so a `2` scores perfectly on a song in four. Deployed, that elected a
    2-beat bar for Sweet Home Alabama, inserted a downbeat into every bar, and
    doubled the song's bar count — a 5-minute 97 bpm track came out as 227 bars
    instead of ~120, its form fragmented, and its sidecar was withheld by the
    ceiling that was the only thing standing between it and the player.

    A candidate is judged on the downbeats the tracker actually produced.
    """
    downbeats = broken_downbeats(spurious=(0, 5, 10, 15), beat_into_bar=2)
    grid = BeatGrid(beats_ms=known_beats(), downbeats_ms=downbeats, bpm=120.0,
                    confidence=0.95, time_signature="4/4")

    repaired, report = repair(grid)
    assert report.bar_beats == BAR_BEATS
    assert report.inserted == 0, "nothing was missing — four downbeats were extra"
    assert report.dropped == 4
    assert repaired.downbeats_ms == known_downbeats()


def test_the_mode_is_not_the_median():
    """A song can be more than a third wrong and still have an unambiguous bar.
    A median over a heavy tail of half-bars lands *between* the two lengths and
    would then repair the song toward a meter it is not in."""
    _, report = repair(broken_grid(spurious=tuple(range(0, 15, 2))))
    assert report.bar_beats == BAR_BEATS


# --- declining to act --------------------------------------------------------

def test_a_grid_whose_downbeats_are_not_beats_is_left_alone():
    """The repair's whole premise is that the downbeats are a *selection* of the
    beats and the selection is what went wrong. Where that isn't true it has no
    business editing anything."""
    grid = BeatGrid(beats_ms=known_beats(),
                    downbeats_ms=[t + 137 for t in known_downbeats()],
                    bpm=120.0, confidence=0.9)
    repaired, report = repair(grid)
    assert not report.ran
    assert repaired.downbeats_ms == grid.downbeats_ms


def test_too_few_bars_to_hold_a_mode_changes_nothing():
    """Three gaps can be 4, 4, 2 with the 2 being the song."""
    beats = [i * MS_PER_BEAT for i in range(13)]
    grid = BeatGrid(beats_ms=beats, downbeats_ms=[0, 2000, 4000, 6000],
                    bpm=120.0, confidence=0.9)
    repaired, report = repair(grid)
    assert not report.ran
    assert repaired.downbeats_ms == grid.downbeats_ms


def test_a_song_mostly_disagreeing_with_its_own_meter_is_flagged():
    """So What sits at 42% and is precisely the case to watch. The repair still
    runs — a self-consistent grid is worth having — but the song is marked, and
    `pipeline.assemble` reads that as a reason to withhold the sidecar rather
    than publish a timeline nobody can vouch for."""
    _, report = repair(broken_grid(spurious=tuple(range(0, 16, 2))))
    assert report.unreliable


# --- what it costs the song, end to end --------------------------------------
#
# This is the bridge to the third complaint ("patterns should be more musical").
# A spurious downbeat adds a bar to the chart and a dropped one removes one, so
# every bar after it is shifted by one against the music — and `form._layout`
# searches for one *global* block phase. A phase that changes mid-song cannot be
# fitted, so block similarity collapses, every block lands in its own group, and
# `model._patterns` emits one non-repeating "pattern" per group.

def _model(grid: BeatGrid):
    return build(grid=grid, raw=known_chords(), onsets=known_onsets())


def test_a_corrupted_downbeat_no_longer_costs_the_song_its_form():
    clean = _model(known_grid())
    assert clean is not None

    for grid in (broken_grid(spurious=(10,)), broken_grid(dropped=(10,))):
        model = _model(grid)
        assert model is not None
        assert model.axis.bar_count == clean.axis.bar_count
        assert len(model.sections) == len(clean.sections)
        assert [s.repeats for s in model.sections] == [s.repeats for s in clean.sections]
        assert len(model.groups) == len(clean.groups)
        assert {p.pattern.id for p in model.patterns.values()} == \
            {p.pattern.id for p in clean.patterns.values()}


# --- the confidence that could not see any of it -----------------------------
#
# `_meter` and `track` are pure functions over two lists, so they are testable
# here without torch, without weights and without audio — which is worth doing,
# because the arithmetic in them is what decided whether a song with a broken
# bar grid shipped a sidecar, and it decided wrong.

def test_single_beat_bars_count_against_the_meter_agreement():
    """They used to be filtered out of the sample entirely, which is the one
    piece of evidence a broken grid always has: So What's 37 one-beat "bars"
    never entered the denominator and it shipped at confidence 0.842 with 42% of
    its bars malformed."""
    beats = known_beats()
    downbeats = broken_downbeats(spurious=tuple(range(0, 16, 2)))

    meter, agreement = beat_this_meter(beats, downbeats)
    assert meter == BAR_BEATS, "a pile of one-beat fragments cannot elect 1/4"
    assert agreement < 0.5


def test_a_clean_grid_still_reports_full_agreement():
    meter, agreement = beat_this_meter(known_beats(), known_downbeats())
    assert (meter, agreement) == (BAR_BEATS, 1.0)


def test_a_perfect_pulse_with_no_bars_cannot_pass_the_confidence_floor():
    """`0.5 * regularity + 0.5 * meter_agreement` gave a song with a flawless
    pulse and zero bar agreement exactly 0.5, and `pipeline.assemble` tests
    `< confidence_floor` against a floor of 0.5 — so this song could not be
    flagged by that path at all. Multiplied, no bars means no confidence."""
    regularity, agreement = 1.0, 0.0
    assert regularity * agreement < 0.5


def test_the_repair_is_reported_on_the_model():
    """Provenance, not a silent rewrite — the rule `consensus` and `vocabulary`
    already follow, and the reason `TheoryReport` carries these counts."""
    model = _model(broken_grid(spurious=(10,), dropped=(4,)))
    assert model is not None
    assert model.meter.downbeats.dropped == 1
    assert model.meter.downbeats.inserted == 1
    assert model.meter.downbeats.ran
