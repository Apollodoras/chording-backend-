"""§14 — strumming patterns.

The tests split the same way §14 does: what is **measured** (onset positions,
subdivision, accent) and what is **convention** (direction). The convention half
is asserted as a convention, not as truth — you cannot hear which way a hand
moved in a mixed recording, and a test that claimed otherwise would be the exact
mistake §14 warns against.
"""

from __future__ import annotations

from app.analysis.strumming import (
    CONVENTION_TAGS,
    beat_position,
    choose_subdivision,
    direction_for,
    extract,
    fallback,
    fold_onsets,
)
from app.chords import MAJOR  # noqa: F401  (keeps the import surface honest)
from tests.conftest import BAR_BEATS, DDUUDU, MS_PER_BEAT, known_beats, known_onsets


# --- the convention ---------------------------------------------------------

def test_directions_alternate_from_each_beat_on_an_eighth_grid():
    """§14's rule as written: on the beat is a downstroke, on the "&" an
    upstroke."""
    assert direction_for(0.0, 2) == "down"
    assert direction_for(0.5, 2) == "up"
    assert direction_for(1.0, 2) == "down"
    assert direction_for(3.5, 2) == "up"


def test_directions_alternate_from_each_beat_on_a_sixteenth_grid():
    """§14's parenthetical — "or the second/fourth 16th" — only agrees with the
    "&"-is-up half under this reading: strokes alternate from the beat, so within
    a beat the even subdivisions are down and the odd ones are up. That gives
    D-U-D-U across 1-e-&-a, with 'e' and 'a' as the ups."""
    assert [direction_for(x, 4) for x in (0.0, 0.25, 0.5, 0.75)] == ["down", "up", "down", "up"]


def test_every_pattern_says_out_loud_that_directions_are_a_convention():
    """So nobody downstream — or in six months — mistakes the assignment for
    detection."""
    result = fallback(bar_beats=4, tempo=120, name="Test")
    assert "directions-by-convention" in result.pattern.tags
    assert set(CONVENTION_TAGS).issubset(set(result.pattern.tags))


# --- what is measured -------------------------------------------------------

def test_beat_position_interpolates_between_beats():
    beats = known_beats()
    assert beat_position(beats, 0) == 0.0
    assert beat_position(beats, MS_PER_BEAT) == 1.0
    assert beat_position(beats, MS_PER_BEAT * 1.5) == 1.5


def test_subdivision_prefers_the_coarsest_grid_that_explains_the_onsets():
    """16ths explain everything 8ths explain, so picking the finest grid would
    invent syncopation out of timing jitter."""
    assert choose_subdivision([0.0, 1.0, 2.0, 3.0]) == 1
    assert choose_subdivision([0.0, 1.0, 1.5, 2.5, 3.0, 3.5]) == 2
    assert choose_subdivision([0.0, 0.25, 0.5, 0.75, 1.0]) == 4


def test_folding_lays_every_bar_on_top_of_every_other():
    folded = fold_onsets(known_onsets(), known_beats(), bar_beats=BAR_BEATS,
                         first_beat=0, last_beat=16)
    positions = sorted({round(p, 3) for p, _ in folded})
    assert positions == DDUUDU


def test_the_known_song_extracts_as_d_du_ud_u():
    """The pattern everybody actually plays — and the one §14 says to prefer over
    a 16-onset transcription of the same bar."""
    folded = fold_onsets(known_onsets(), known_beats(), bar_beats=BAR_BEATS,
                         first_beat=0, last_beat=64)
    result = extract(folded, bar_beats=BAR_BEATS, bars=16, tempo=120, name="Verse strum")
    assert not result.is_fallback
    assert [s.beat for s in result.pattern.strokes] == DDUUDU
    assert [s.direction for s in result.pattern.strokes] == \
        ["down", "down", "up", "up", "down", "up"]
    assert result.confidence == 1.0


def test_the_downbeat_reads_as_accented():
    folded = fold_onsets(known_onsets(), known_beats(), bar_beats=BAR_BEATS,
                         first_beat=0, last_beat=64)
    result = extract(folded, bar_beats=BAR_BEATS, bars=16, tempo=120, name="Verse strum")
    assert result.pattern.strokes[0].accent
    assert not any(s.accent for s in result.pattern.strokes[1:])


def test_a_one_off_fill_never_enters_the_repeating_pattern():
    """An onset that happens in a third of the bars is a fill; putting it in the
    pattern makes every bar wrong instead of one bar right."""
    folded = [(0.0, 1.0)] * 8 + [(2.0, 1.0)] * 8 + [(1.25, 1.0)] * 2
    result = extract(folded, bar_beats=BAR_BEATS, bars=8, tempo=120, name="Verse strum")
    assert [s.beat for s in result.pattern.strokes] == [0.0, 2.0]


# --- degrading honestly -----------------------------------------------------

def test_thin_support_falls_back_to_quarter_note_downstrokes():
    """A boring pattern that plays is worth more than a confident one that's
    wrong — and the app *requires* a pattern, or the section is silently
    dropped."""
    result = extract([(0.3, 1.0), (2.7, 1.0)], bar_beats=BAR_BEATS, bars=16,
                     tempo=120, name="Verse strum")
    assert result.is_fallback
    assert [s.beat for s in result.pattern.strokes] == [0.0, 1.0, 2.0, 3.0]
    assert all(s.direction == "down" for s in result.pattern.strokes)
    assert result.confidence == 0.0


def test_no_onsets_at_all_still_produces_a_playable_bar():
    result = extract([], bar_beats=BAR_BEATS, bars=8, tempo=120, name="Verse strum")
    assert result.is_fallback
    assert result.pattern.strokes


def test_mute_is_never_emitted():
    """§14: not reliably recoverable in a full mix, so it is not guessed at."""
    folded = fold_onsets(known_onsets(), known_beats(), bar_beats=BAR_BEATS,
                         first_beat=0, last_beat=64)
    result = extract(folded, bar_beats=BAR_BEATS, bars=16, tempo=120, name="Verse strum")
    assert {s.direction for s in result.pattern.strokes} <= {"down", "up"}


# --- ids --------------------------------------------------------------------

def test_an_unchanged_groove_keeps_its_id():
    """§12.5's "keep embedded pattern ids stable when their strokes are
    unchanged", held by construction: the id is a hash of the strokes."""
    a = fallback(bar_beats=4, tempo=120, name="Verse strum")
    b = fallback(bar_beats=4, tempo=120, name="Chorus strum")
    assert a.pattern.id == b.pattern.id


def test_a_different_groove_gets_a_different_id():
    a = fallback(bar_beats=4, tempo=120, name="x")
    b = fallback(bar_beats=3, tempo=120, name="x", time_signature="3/4")
    assert a.pattern.id != b.pattern.id


def test_pattern_ids_are_namespaced_away_from_mo():
    """Mo mints `mo:pat-…`; this service mints `yt:pat-…`. The two backends write
    into the same Library and `import` upserts on id."""
    assert fallback(bar_beats=4, tempo=120, name="x").pattern.id.startswith("yt:pat-")


def test_strokes_never_fall_outside_one_bar():
    """The app silently drops out-of-range strokes, so a pattern that overflows
    its bar plays short with no error anywhere."""
    for bar_beats, signature in ((4, "4/4"), (3, "3/4")):
        result = fallback(bar_beats=bar_beats, tempo=120, name="x", time_signature=signature)
        assert all(0 <= s.beat < bar_beats for s in result.pattern.strokes)
