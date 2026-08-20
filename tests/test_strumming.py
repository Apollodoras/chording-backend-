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
    SNAPPED_TAG,
    FoldedOnset,
    _band_of,
    beat_position,
    choose_subdivision,
    direction_for,
    directions_for,
    extract,
    fallback,
    fold_onsets,
)
from app.analysis.types import FULL, LOW, MID, Onset
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
    D-DU-UD-U with one stroke under the support threshold.

    "Under the support threshold" is the load-bearing half, and it is now what
    the fixture actually builds: the 3.5 is played in five bars of sixteen, which
    is real evidence that misses `SUPPORT_THRESHOLD`. The rule used to fire with
    the cell **empty**, which is not this claim — see the test below.
    """
    short = [(bar, position, 1.0) for bar in range(16)
             for position in (0.0, 1.0, 1.5, 2.5, 3.0)]
    short += [(bar, 3.5, 1.0) for bar in range(5)]
    result = extract(short, bar_beats=BAR_BEATS, bars=16, tempo=120, name="x")
    assert [s.beat for s in result.pattern.strokes] == DDUUDU


def test_the_library_never_adds_a_stroke_the_recording_does_not_show():
    """A groove with a hole in it keeps the hole.

    Nothing on beat 3 — an ordinary thing to play, and the nearest idiom is the
    campfire pattern *with* beat 3 in it. Taking the entry whole would hand the
    player a stroke they had pointedly not played, which is the opposite of what
    a chart is for, and it is what `snap_to_idiom`'s own docstring already
    forbade ("a correction and never an invention") while the code did it anyway.
    """
    holed = [(bar, position, 1.0) for bar in range(16)
             for position in (0.0, 1.0, 1.5, 2.5, 3.5)]
    result = extract(holed, bar_beats=BAR_BEATS, bars=16, tempo=120, name="x")
    assert [s.beat for s in result.pattern.strokes] == [0.0, 1.0, 1.5, 2.5, 3.5]
    assert SNAPPED_TAG not in result.pattern.tags


def test_a_snapped_pattern_says_so_out_loud():
    """A snap is not a measurement, and every emitted pattern has to be readable
    as which of the two it is — the rule the direction tags already follow."""
    short = [(bar, position, 1.0) for bar in range(16)
             for position in (0.0, 1.0, 1.5, 2.5, 3.0)]
    short += [(bar, 3.5, 1.0) for bar in range(5)]      # under the threshold, not absent
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


def test_a_syncopation_the_player_varies_survives_the_averaging():
    """The audit's open question about §14, and the second half of "the patterns
    aren't rhythmic".

    A human does not play the same bar sixteen times. Here the "&" of 2 lands in
    half the bars and the "&" of 4 in the other half, so each sits exactly on
    `SUPPORT_THRESHOLD` while the groove is unmistakably syncopated. Both used to
    be cut — support passed them and contrast then charged them a second time for
    the same fact, because `prominence` folds support back in as a multiplier —
    and what came out was `0 1 2.5 3`: the metronome underneath the groove.

    One pattern per repeat group is §14's design, so the honest answer for a
    section played two ways is the union of them, and `confidence` is what says
    the section was less consistent than the others. It reports about 0.8 here
    against 0.99 for a groove played the same way every bar.
    """
    folded = []
    for bar in range(16):
        played = (0.0, 1.0, 1.5, 2.5, 3.0) if bar % 2 == 0 else (0.0, 1.0, 2.5, 3.0, 3.5)
        folded += [(bar, position, 1.0) for position in played]
    result = extract(folded, bar_beats=BAR_BEATS, bars=16, tempo=120, name="x")
    assert [s.beat for s in result.pattern.strokes] == DDUUDU
    assert result.confidence < 0.9, "and it says the section was played two ways"


def test_a_quiet_hi_hat_is_still_cut_on_a_grid_where_everything_is_struck():
    """The other side of the same rule, and why the condition is the grid's
    *sparsity* rather than a loudness threshold.

    A real upstroke played in half the bars reaches 0.335 of the bar's loudest
    cell and a hi-hat reaches 0.359, so no contrast threshold separates them.
    What separates them is that a rhythm leaves cells empty and a hi-hat does
    not: this grid has an onset on every eighth, so support is telling us
    nothing, no relief applies, and loudness decides — which is the full-mix
    behaviour `test_a_kit_behind_the_guitar_does_not_become_the_pattern` pins.
    """
    saturated = drum_kit()
    scored_positions = {round(position, 3) for _, position, _ in saturated}
    assert scored_positions == {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5}, \
        "every cell of the grid carries an onset — support cannot discriminate"
    result = extract(saturated, bar_beats=BAR_BEATS, bars=16, tempo=120, name="x")
    assert [s.beat for s in result.pattern.strokes] == DDUUDU


# --- the convention on a triple grid ----------------------------------------
#
# The rule above is a rule about *grid cells*, and it is right on a duple grid
# because the hand really does cross the strings once per cell there whether or
# not the cell is struck. A triple grid has no such pendulum, and reading parity
# off it produced a shuffle strummed Down-Down.

def test_a_shuffle_is_down_up_and_not_down_down():
    """The single most common triple feel there is — the beat and the "let",
    nothing in between — and every chart in the world notates it D-U. Parity over
    a 3-cell beat makes cell 2 a downstroke and got this exactly wrong; the hand
    cannot take two downstrokes in a row at tempo without a wasted pass."""
    shuffle = [0.0, 2 / 3, 1.0, 1 + 2 / 3, 2.0, 2 + 2 / 3, 3.0, 3 + 2 / 3]
    assert directions_for(shuffle, 3) == ["down", "up"] * 4


def test_a_full_triplet_is_down_up_down():
    """Three strokes to the beat leaves the hand where it started, so the next
    beat is a downstroke again. Parity happened to get this one right and the
    alternation has to keep getting it right."""
    assert directions_for([0.0, 1 / 3, 2 / 3], 3) == ["down", "up", "down"]
    assert directions_for([0.0, 1 / 3, 2 / 3, 1.0], 3)[-1] == "down"


def test_a_triple_grid_restarts_downward_on_every_beat():
    """The half of the convention nobody argues about: whatever happened inside
    the last beat, the hand comes back down on the next one."""
    assert directions_for([0.0, 1.0, 2.0, 3.0], 3) == ["down"] * 4


def test_a_duple_grid_still_answers_from_the_grid_and_not_the_neighbours():
    """The pendulum is not replaced, and must not be. On a 16th grid the "&" is
    a *downstroke* — 1 e & a is D-U-D-U — so a bar holding only 1, & and a is
    D-D-U. Alternating over the sounded strokes instead would call the "&" an
    upstroke, which is the bug this rule exists to avoid."""
    assert directions_for([0.0, 0.5, 0.75], 4) == ["down", "down", "up"]
    assert directions_for([0.0, 0.5, 1.0, 1.5], 2) == ["down", "up", "down", "up"]
    assert directions_for(list(DDUUDU), 2) == ["down", "down", "up", "up", "down", "up"]


def test_six_eight_in_two_is_two_downstrokes():
    """Both strokes are the bar's own dotted-quarter pulses — main beats, both
    played downward. They sit on cells 0 and 3 of a 2-per-beat grid, where the
    pendulum reads cell 3 as an offbeat and hands back an upstroke; nothing in
    the grid can see the difference, so the idiom carries its own fingering."""
    onsets = [FoldedOnset(bar=bar, position=position, strength=1.0)
              for bar in range(12) for position in (0.0, 1.5)]
    result = extract(onsets, bar_beats=3.0, bars=12, tempo=90, name="Verse",
                     time_signature="6/8")
    assert [(s.beat, s.direction) for s in result.pattern.strokes] == [
        (0.0, "down"), (1.5, "down"),
    ]


# --- §14.1 bands: what was struck, in which half of the spectrum -------------
#
# The band is the one dimension of an accompaniment that is both measurable and
# instrument-neutral, so these tests are about two separate claims and keep them
# separate: that a bar with two hands in it comes out with two hands, and that a
# bar with one comes out **exactly** as it did before bands existed.

def _banded_onsets(layout: dict[float, str], *, bars: int = 8,
                   strengths: dict[str, float] | None = None) -> list[Onset]:
    """`bars` bars of one accompaniment: {bar-local beat: band}.

    The default strengths are the shape that matters — the bass quieter than the
    chord over it, which is what a left hand actually is and what a bar-wide
    contrast test throws away.
    """
    levels = strengths or {LOW: 0.5, MID: 1.0, FULL: 1.0}
    out: list[Onset] = []
    for bar in range(bars):
        start = bar * BAR_BEATS * MS_PER_BEAT
        for offset, band in layout.items():
            out.append(Onset(t_ms=int(start + offset * MS_PER_BEAT),
                             strength=levels[band], band=band))
    return sorted(out, key=lambda o: o.t_ms)


def _extract(onsets, *, bars: int = 8, name: str = "Verse"):
    folded = fold_onsets(onsets, known_beats(), bar_beats=BAR_BEATS,
                         first_beat=0, last_beat=bars * BAR_BEATS)
    return extract(folded, bar_beats=BAR_BEATS, bars=bars, tempo=120, name=name)


def test_a_band_is_claimed_only_when_two_thirds_of_the_onsets_agree():
    """`BAND_MAJORITY` gates *presence*, so it is what decides `FULL` too."""
    low_only = [FoldedOnset(bar=b, position=0.0, strength=1.0, band=LOW) for b in range(9)]
    assert _band_of(low_only) == LOW
    # Six of nine bars caught the bass alone, three caught the whole band: two
    # thirds exactly is not *more* than two thirds, so the chord band is not
    # claimed and this stays a bass stroke.
    mixed = ([FoldedOnset(bar=b, position=0.0, strength=1.0, band=LOW) for b in range(6)]
             + [FoldedOnset(bar=b, position=0.0, strength=1.0, band=FULL) for b in range(6, 9)])
    assert _band_of(mixed) == LOW
    assert _band_of([]) == FULL


def test_a_full_onset_is_evidence_for_both_bands():
    """`FULL` means both bands moved, so it has to vote in both tallies — a cell
    struck by the whole band every bar is not a bass stroke and not a chord
    stroke, it is both."""
    every_bar = [FoldedOnset(bar=b, position=0.0, strength=1.0, band=FULL)
                 for b in range(8)]
    assert _band_of(every_bar) == FULL


def test_folding_carries_the_band_from_the_onset():
    onsets = _banded_onsets({0.0: LOW, 1.0: MID})
    folded = fold_onsets(onsets, known_beats(), bar_beats=BAR_BEATS,
                         first_beat=0, last_beat=32)
    assert {round(f.position, 3): f.band for f in folded} == {0.0: LOW, 1.0: MID}


def test_an_oom_pah_extracts_as_a_bass_and_a_chord():
    """§14.1's specimen: a root on 1 and 3, a chord on 2 and 4 — the shape of
    nearly every piano accompaniment there is."""
    result = _extract(_banded_onsets({0.0: LOW, 1.0: MID, 2.0: LOW, 3.0: MID}))
    assert not result.is_fallback
    assert [(s.beat, s.band) for s in result.pattern.strokes] == [
        (0.0, LOW), (1.0, MID), (2.0, LOW), (3.0, MID)
    ]


def test_a_quiet_bass_stroke_survives_beside_a_loud_chord():
    """Contrast is measured **within** a band (`_with_contrast`).

    Against one bar-wide peak the bass strokes here are less than half the
    loudness of the chord over them, so the rule that separates a hand from a
    hi-hat deletes them and the pattern comes back as the two chord stabs — the
    measured failure that made contrast band-relative in the first place.
    """
    quiet_bass = _banded_onsets({0.0: LOW, 1.0: MID, 2.0: LOW, 3.0: MID},
                                strengths={LOW: 0.3, MID: 1.0, FULL: 1.0})
    beats = [s.beat for s in _extract(quiet_bass).pattern.strokes]
    assert beats == [0.0, 1.0, 2.0, 3.0]


def test_a_strummed_bar_emits_no_bands_at_all():
    """The commonest case, and the one that must not move: every stroke covers
    the whole spectrum, so nothing is claimed and the wire stays as it was."""
    result = _extract(_banded_onsets({0.0: FULL, 1.0: FULL, 2.0: FULL, 3.0: FULL}))
    assert [s.band for s in result.pattern.strokes] == [None, None, None, None]


def test_chord_band_labels_collapse_without_a_bass_to_mean_them_against():
    """`_hands_apart` — a bar of chord-band-only strokes is a song whose bass is
    quiet, not a song played with no left hand, and the honest label for every
    one of its strokes is "the whole range"."""
    result = _extract(_banded_onsets({0.0: MID, 1.0: MID, 2.0: MID, 3.0: MID}))
    assert [s.band for s in result.pattern.strokes] == [None, None, None, None]


def test_an_unbanded_pattern_keeps_the_id_it_had_before_bands_existed():
    """§12.5 — a groove that has not changed keeps its id.

    Bands join the content-addressed fingerprint only when a stroke carries one,
    so every song analyzed before §14.1 and every song that is simply strummed
    hashes to exactly what it hashed to. The literal is the point: if this test
    has to be updated, every cached song in the catalog has been invalidated.

    The literal is not a copy of what the code currently prints — that would pin
    the behaviour to itself and pass no matter what. It is sha1 of the body the
    fingerprint had *before* bands existed, computed from the old formula:

        "4/4|0.0000,down,0;1.0000,down,0;2.0000,down,0;3.0000,down,0"
    """
    result = _extract(_banded_onsets({0.0: FULL, 1.0: FULL, 2.0: FULL, 3.0: FULL}))
    assert result.pattern.id == "yt:pat-8b3c163231b9"


def test_two_grooves_that_differ_only_in_hand_get_different_ids():
    """...and the other half of §12.5: a groove that *has* changed must not keep
    it. Same beats, same directions, different hands — different music."""
    strummed = _extract(_banded_onsets({0.0: FULL, 1.0: FULL, 2.0: FULL, 3.0: FULL}))
    oom_pah = _extract(_banded_onsets({0.0: LOW, 1.0: MID, 2.0: LOW, 3.0: MID}))
    assert [s.beat for s in strummed.pattern.strokes] == \
        [s.beat for s in oom_pah.pattern.strokes]
    assert strummed.pattern.id != oom_pah.pattern.id
