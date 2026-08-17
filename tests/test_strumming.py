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
from app.analysis.types import Onset
from app.chords import MAJOR  # noqa: F401  (keeps the import surface honest)
from tests.conftest import (
    BAR_BEATS,
    DDUUDU,
    MS_PER_BEAT,
    ghost_sixteenths,
    jittered_onsets,
    known_beats,
    known_onsets,
)


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
    positions = sorted({round(o.position, 3) for o in folded})
    assert positions == DDUUDU


def test_folding_remembers_which_bar_each_onset_came_from():
    """Support is a share of *bars*, so the fold has to carry bar identity — a
    bar with a flam or a doubled drum hit must not stand in for two bars of
    evidence."""
    folded = fold_onsets(known_onsets(), known_beats(), bar_beats=BAR_BEATS,
                         first_beat=0, last_beat=16)
    assert sorted({o.bar for o in folded}) == [0, 1, 2, 3]
    downbeats = [o for o in folded if abs(o.position) < 0.01]
    assert sorted(o.bar for o in downbeats) == [0, 1, 2, 3]


def test_a_bar_that_repeats_a_stroke_is_still_one_bar_of_evidence():
    """`support = onsets / bars` let a flam count twice: struck in 7 bars of 16
    but hit 14 times, these cells scored 0.875 and entered the pattern. Counted
    as a share of bars they are 0.44 — a fill, and below the threshold."""
    played_in = range(7)
    doubled = [(bar, position, 1.0) for bar in played_in for position in (0.0, 2.0)] + \
              [(bar, position + 0.02, 1.0) for bar in played_in for position in (0.0, 2.0)]
    result = extract(doubled, bar_beats=BAR_BEATS, bars=16, tempo=120, name="x")
    assert result.is_fallback


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
    folded = [(bar, 0.0, 1.0) for bar in range(8)] + \
             [(bar, 2.0, 1.0) for bar in range(8)] + \
             [(bar, 1.25, 1.0) for bar in range(2)]
    result = extract(folded, bar_beats=BAR_BEATS, bars=8, tempo=120, name="Verse strum")
    assert [s.beat for s in result.pattern.strokes] == [0.0, 2.0]


# --- what real recordings do to all of the above -----------------------------

def test_the_known_song_survives_jitter_around_the_barline():
    """The defect this file could not see. Onsets never land exactly on the grid,
    and a downbeat is far likelier to be early than late — a hand 20 ms ahead of
    the "one" folds to ~3.98, the far end of the bar, where no cell could claim
    it. The beat-1 stroke silently disappeared and the extraction still reported
    confidence 1.0 in what was left."""
    folded = fold_onsets(jittered_onsets(), known_beats(), bar_beats=BAR_BEATS,
                         first_beat=0, last_beat=64)
    result = extract(folded, bar_beats=BAR_BEATS, bars=16, tempo=120, name="Verse strum")
    assert not result.is_fallback
    assert [s.beat for s in result.pattern.strokes] == DDUUDU
    assert [s.direction for s in result.pattern.strokes] == \
        ["down", "down", "up", "up", "down", "up"]


def test_an_onset_ahead_of_the_downbeat_supports_the_bar_it_anticipates():
    """Rolled forward onto the "one" it was reaching for — so it is evidence for
    the next bar, not for the bar it was played at the end of. Getting this wrong
    would show up as a downbeat supported by every bar but the first."""
    early = [Onset(t_ms=int(bar * BAR_BEATS * MS_PER_BEAT) - 20, strength=1.0)
             for bar in range(1, 8)]
    folded = fold_onsets(early, known_beats(), bar_beats=BAR_BEATS,
                         first_beat=0, last_beat=32)
    assert all(abs(o.position) <= 0.12 for o in folded)
    assert sorted(o.bar for o in folded) == [1, 2, 3, 4, 5, 6, 7]


def test_one_sixteenth_ghost_does_not_flip_every_upstroke_to_a_downstroke():
    """A hi-hat on one 16th of every bar is present in every drummed recording.
    Scored by share of onsets it carried the grid from 8ths to 16ths, and since
    direction is read off the grid, the "&"s at 1.5 and 2.5 flipped from up to
    down — D-DU-UD-U degrading toward all-downstrokes."""
    folded = fold_onsets(known_onsets() + ghost_sixteenths(), known_beats(),
                         bar_beats=BAR_BEATS, first_beat=0, last_beat=64)
    result = extract(folded, bar_beats=BAR_BEATS, bars=16, tempo=120, name="Verse strum")
    assert [s.beat for s in result.pattern.strokes] == DDUUDU
    assert [s.direction for s in result.pattern.strokes] == \
        ["down", "down", "up", "up", "down", "up"]


def test_a_real_sixteenth_feel_still_reads_as_sixteenths():
    """The guard above must not deafen the extractor to genuine 16ths — two per
    bar is a feel, not a ghost."""
    positions = (0.0, 0.75, 1.0, 1.75, 2.0, 3.0)
    folded = [(bar, p, 1.0) for bar in range(16) for p in positions]
    result = extract(folded, bar_beats=BAR_BEATS, bars=16, tempo=120, name="x")
    assert [s.beat for s in result.pattern.strokes] == list(positions)
    assert [s.direction for s in result.pattern.strokes][:2] == ["down", "up"]


# --- degrading honestly -----------------------------------------------------

def test_thin_support_falls_back_to_quarter_note_downstrokes():
    """A boring pattern that plays is worth more than a confident one that's
    wrong — and the app *requires* a pattern, or the section is silently
    dropped."""
    result = extract([(0, 0.3, 1.0), (0, 2.7, 1.0)], bar_beats=BAR_BEATS, bars=16,
                     tempo=120, name="Verse strum")
    assert result.is_fallback
    assert [s.beat for s in result.pattern.strokes] == [0.0, 1.0, 2.0, 3.0]
    assert all(s.direction == "down" for s in result.pattern.strokes)
    assert result.confidence == 0.0


def test_one_chord_a_bar_is_a_pattern_when_it_is_played_every_bar():
    """A slow ballad struck once a bar was replaced by the four-downstroke
    fallback — *more* strokes than the recording has, presented as the honest
    floor. One stroke is a pattern when the evidence is unambiguous."""
    folded = [(bar, 0.0, 1.0) for bar in range(16)]
    result = extract(folded, bar_beats=BAR_BEATS, bars=16, tempo=120, name="Verse strum")
    assert not result.is_fallback
    assert [s.beat for s in result.pattern.strokes] == [0.0]


def test_one_stroke_in_half_the_bars_is_still_not_a_pattern():
    """The floor moved for the unambiguous case only; a lone half-supported cell
    has nothing else in the bar to corroborate it."""
    folded = [(bar, 0.0, 1.0) for bar in range(9)]
    result = extract(folded, bar_beats=BAR_BEATS, bars=16, tempo=120, name="Verse strum")
    assert result.is_fallback


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


# --- one to few strokes, and musical ones -----------------------------------
#
# Support is a share of *bars*, and on a full mix nearly every cell is supported:
# `LibrosaOnsetDetector` fires on the kit, the kit plays in every bar, and so
# "keep every cell over the threshold" kept fifteen of the sixteen 16th cells on
# a recording whose guitar plays six. The three rules below are what stands
# between the grid and the pattern now.

def drum_kit(*, bars: int = 16) -> list[tuple[int, float, float]]:
    """A guitar playing D-DU-UD-U with a hi-hat on every 8th behind it.

    The hi-hat is quiet and perfectly reliable, which is exactly the combination
    support cannot see through: it is present in *every* bar, so every cell it
    touches is fully supported.
    """
    folded: list[tuple[int, float, float]] = []
    for bar in range(bars):
        for position in DDUUDU:
            folded.append((bar, position, 1.6 if position == 0.0 else 1.0))
        for eighth in range(8):
            folded.append((bar, eighth * 0.5, 0.35))
    return folded


def test_a_kit_behind_the_guitar_does_not_become_the_pattern():
    """The measured case, and the complaint: fifteen strokes where the recording
    has six. What separates them is not support — both are in every bar — but
    how hard the cell is struck."""
    result = extract(drum_kit(), bar_beats=BAR_BEATS, bars=16, tempo=120, name="x")
    assert [s.beat for s in result.pattern.strokes] == DDUUDU


def test_a_bar_with_an_onset_on_everything_reads_as_the_skeleton_underneath():
    """Sixteen equally-struck cells carry no groove at all — there is nothing to
    contrast — so the honest reading is the eight-note skeleton, not sixteen
    strokes. Two per beat is the ceiling and this is the case it exists for."""
    saturated = [(bar, cell / 4, 1.0) for bar in range(16) for cell in range(16)]
    result = extract(saturated, bar_beats=BAR_BEATS, bars=16, tempo=120, name="x")
    assert len(result.pattern.strokes) == 8
    assert [s.direction for s in result.pattern.strokes[:2]] == ["down", "up"], \
        "and the directions are re-read off the grid the strokes actually landed on"


def test_an_extraction_one_cell_short_of_an_idiom_is_snapped_onto_it():
    """The direct answer to "patterns should be more musical", and the same
    measure-then-snap discipline `vocabulary.SNAP_TO` follows for chords: an
    extraction that lands a cell short of D-DU-UD-U almost certainly *is*
    D-DU-UD-U with one stroke under the support threshold."""
    short = [(bar, position, 1.0) for bar in range(16)
             for position in (0.0, 1.0, 1.5, 2.5, 3.0)]
    result = extract(short, bar_beats=BAR_BEATS, bars=16, tempo=120, name="x")
    assert [s.beat for s in result.pattern.strokes] == DDUUDU


def test_a_snapped_pattern_says_so_out_loud():
    """A snap is not a measurement, and every emitted pattern has to be readable
    as which of the two it is — the rule the direction tags already follow."""
    short = [(bar, position, 1.0) for bar in range(16)
             for position in (0.0, 1.0, 1.5, 2.5, 3.0)]
    snapped = extract(short, bar_beats=BAR_BEATS, bars=16, tempo=120, name="x")
    measured = extract([(bar, p, 1.0) for bar in range(16) for p in DDUUDU],
                       bar_beats=BAR_BEATS, bars=16, tempo=120, name="x")
    assert "snapped-to-idiom" in snapped.pattern.tags
    assert "snapped-to-idiom" not in measured.pattern.tags


def test_a_groove_that_is_nothing_like_an_idiom_is_left_as_measured():
    """The library corrects; it does not overwrite. A real 16th feel is two
    thirds of the way to plain quarters by the similarity measure, and it is not
    plain quarters."""
    positions = (0.0, 0.75, 1.0, 1.75, 2.0, 3.0)
    folded = [(bar, position, 1.0) for bar in range(16) for position in positions]
    result = extract(folded, bar_beats=BAR_BEATS, bars=16, tempo=120, name="x")
    assert [s.beat for s in result.pattern.strokes] == list(positions)
    assert "snapped-to-idiom" not in result.pattern.tags
