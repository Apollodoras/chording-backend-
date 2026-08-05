"""§20.6 — one analysis, three renders.

The claim under test is that a song has **one** structure and the difficulty
tier only changes which chord names are printed inside it. Before §20 the whole
of §5.4, §15 and §14 ran once per tier, so the three songs were related only by
having come from the same recording — and `easy`'s simplification can merge two
bars into identical ones, which changes what the segmenter collapses. The
pipeline had to re-check `lint_sync` against every tier to catch the fallout.

That check is still there and still passes. These tests assert the stronger
thing it could never establish: that the tiers agree **by construction**.
"""

from __future__ import annotations

import pytest

from app.analysis import model as song_model
from app.analysis.types import BeatGrid, EnergyCurve, RawChordSpan
from app.chords import DIFFICULTIES, EASY, HARD
from app.errors import AnalysisError

BEAT_MS = 500
BAR_MS = BEAT_MS * 4


def _grid(bars: int = 16) -> BeatGrid:
    beats = [i * BEAT_MS for i in range(bars * 4 + 1)]
    return BeatGrid(beats_ms=beats, downbeats_ms=beats[::4], bpm=120.0,
                    confidence=0.9, time_signature="4/4")


def _chords(bars: int = 16, *, names=("C", "G", "Am", "F")) -> list[RawChordSpan]:
    return [RawChordSpan(start_ms=i * BAR_MS, end_ms=(i + 1) * BAR_MS,
                         label=names[i % len(names)], confidence=0.9)
            for i in range(bars)]


def _model(**kwargs):
    return song_model.build(grid=_grid(), raw=_chords(), onsets=[], **kwargs)


# --- the model ---------------------------------------------------------------

def test_a_model_is_built_from_a_steady_grid_and_readable_chords():
    model = _model()
    assert model is not None
    assert model.sections and model.groups
    assert model.meter.bar_beats == 4
    assert model.axis.bar_count == 16


def test_no_usable_rhythm_returns_none_rather_than_raising():
    """The caller turns this into an honest refusal (§13.3). It is a different
    failure from having no chords, and the player is told a different thing."""
    thin = BeatGrid(beats_ms=[0, 500], downbeats_ms=[0], bpm=120.0,
                    confidence=0.9, time_signature="4/4")
    assert song_model.build(grid=thin, raw=_chords(), onsets=[]) is None


def test_no_readable_chords_raises_rather_than_blaming_the_rhythm():
    """"That track's beat wasn't clear enough" is wrong and confusing on a
    perfectly steady recording the chord engine simply heard as silence."""
    silent = [RawChordSpan(start_ms=0, end_ms=32000, label="N")]
    with pytest.raises(AnalysisError, match="No chords"):
        song_model.build(grid=_grid(), raw=silent, onsets=[])


# --- the tiers are renders, not analyses -------------------------------------

def test_every_tier_shares_the_models_section_boundaries():
    model = _model()
    expected = [(s.start_bar, s.total_bars) for s in model.sections]
    for difficulty in DIFFICULTIES:
        rendered = song_model.render(model, _chords(), difficulty)
        assert [(s.start_bar, s.total_bars) for s in rendered] == expected, difficulty


def test_every_tier_shares_the_models_repeat_groups():
    model = _model()
    expected = [s.group for s in model.sections]
    for difficulty in DIFFICULTIES:
        assert [s.group for s in song_model.render(model, _chords(), difficulty)] == expected


def test_the_tiers_tile_the_song_identically():
    """One sidecar serves whichever tier the player asked for, so a tier that
    tiled differently would be addressed by anchors meant for another."""
    model = _model()
    for difficulty in DIFFICULTIES:
        cursor = 0
        for section in song_model.render(model, _chords(), difficulty):
            assert section.start_bar == cursor
            cursor += section.total_bars
        assert cursor == model.total_bars


def test_simplification_changes_the_names_not_the_shape():
    """`easy` really does drop passing chords shorter than a bar — duration work
    that cannot be done by renaming qualities in place — but the boundaries come
    from the model and do not move."""
    raw = _chords(names=("Cmaj7", "G7", "Am7", "Fmaj7"))
    model = song_model.build(grid=_grid(), raw=raw, onsets=[])
    hard = song_model.render(model, raw, HARD)
    easy = song_model.render(model, raw, EASY)
    assert [(s.start_bar, s.total_bars) for s in hard] == [(s.start_bar, s.total_bars) for s in easy]

    hard_qualities = {c.quality for s in hard for bar in s.bars for c in bar}
    easy_qualities = {c.quality for s in easy for bar in s.bars for c in bar}
    assert hard_qualities != easy_qualities, "the tiers really do differ in vocabulary"
    assert easy_qualities <= {"major", "minor"}


# --- patterns ----------------------------------------------------------------

def test_one_pattern_per_repeat_group():
    model = _model()
    assert set(model.patterns) == {g.label for g in model.groups}


def test_sections_of_one_group_share_a_groove():
    """They are the same music, which is also why the pooled extraction had more
    evidence to work from."""
    model = _model()
    for section in model.sections:
        assert section.group in model.patterns


# --- energy ------------------------------------------------------------------

def test_a_loudness_curve_becomes_one_number_per_bar():
    """Averaged between the axis's own downbeats, so bar k's energy is measured
    over exactly the span bar k occupies — the same correspondence the anchors
    publish."""
    model = _model()
    curve = EnergyCurve(hop_ms=50, values=[0.0] * 320 + [1.0] * 320)
    per_bar = song_model.per_bar_energy(curve, model.axis)
    assert per_bar is not None and len(per_bar) == model.axis.bar_count
    assert per_bar[0] < per_bar[-1]


def test_no_probe_means_no_energy_and_that_is_a_supported_configuration():
    model = _model()
    assert song_model.per_bar_energy(None, model.axis) is None
    assert all(s.kind == "custom" for s in model.sections), "§15's honest fallback"


def test_an_empty_curve_is_treated_as_no_curve():
    model = _model()
    assert song_model.per_bar_energy(EnergyCurve(hop_ms=50, values=[]), model.axis) is None
    assert song_model.per_bar_energy(EnergyCurve(hop_ms=0, values=[1.0]), model.axis) is None


# --- the consensus switch ----------------------------------------------------

def test_the_vote_can_be_turned_off():
    """`CHORDS_THEORY_CONSENSUS=off` is a supported posture: this is the one part
    of §20 that edits chords the engine reported, and the honest way to ship
    something judged only by measurement is to be able to turn it off."""
    model = song_model.build(grid=_grid(), raw=_chords(), onsets=[], vote=False)
    assert model.consensus.rewritten_bars == 0
    assert model.consensus.groups_voted == 0
