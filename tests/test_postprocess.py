"""§5.4 — the stage the handoff calls "where playability is won".

Every rule here is the difference between a chart that plays and one that
technically contains the right chords. The assertions are written as the musical
statement rather than the data statement wherever possible: "a run of jitter
reads as one held chord", not "len(result) == 1".
"""

from __future__ import annotations

from app.analysis.postprocess import (
    drop_short,
    exact_ratio,
    hold_through_gaps,
    mean_confidence,
    merge,
    process,
    quantize,
)
from app.analysis.types import GridSpan, RawChordSpan
from app.chords import MAJOR, MAJOR7, MINOR
from tests.conftest import known_axis, known_chords


def span(start, length, root=0, quality=MAJOR, confidence=1.0, exact=True):
    return GridSpan(start_beat=start, length_beats=length, root_pc=root,
                    quality=quality, confidence=confidence, exact=exact)


# --- quantize ---------------------------------------------------------------

def test_quantize_snaps_boundaries_that_wobble_off_the_beat():
    """A chord change 40 ms early is the same musical event as one on the beat.
    Left unquantized it shows up as a chord that changes *between* two strokes,
    which reads as a glitch rather than as detail."""
    axis = known_axis()
    raw = [
        RawChordSpan(start_ms=0, end_ms=1960, label="C:maj"),      # wants beat 4
        RawChordSpan(start_ms=1960, end_ms=4030, label="G:maj"),   # wants beat 8
    ]
    out = quantize(raw, axis)
    assert [(s.start_beat, s.length_beats) for s in out] == [(0, 4), (4, 4)]


def test_quantize_drops_no_chord_spans_rather_than_emitting_them():
    """There is no rest primitive in the container (§18) — the hole `N` leaves is
    closed later by holding the previous chord."""
    axis = known_axis()
    raw = [
        RawChordSpan(start_ms=0, end_ms=2000, label="C:maj"),
        RawChordSpan(start_ms=2000, end_ms=4000, label="N"),
        RawChordSpan(start_ms=4000, end_ms=6000, label="G:maj"),
    ]
    out = quantize(raw, axis)
    assert [s.root_pc for s in out] == [0, 7]


def test_quantize_discards_spans_too_short_to_own_a_beat():
    axis = known_axis()
    raw = [RawChordSpan(start_ms=0, end_ms=80, label="C:maj")]
    assert quantize(raw, axis) == []


# --- merge ------------------------------------------------------------------

def test_merge_collapses_a_run_of_the_same_chord():
    out = merge([span(0, 2), span(2, 2), span(4, 4)])
    assert len(out) == 1
    assert (out[0].start_beat, out[0].length_beats) == (0, 8)


def test_merge_keeps_a_real_change():
    out = merge([span(0, 4, root=0), span(4, 4, root=7)])
    assert [s.root_pc for s in out] == [0, 7]


def test_merge_never_launders_a_doubtful_chord_into_a_confident_one():
    out = merge([span(0, 4, confidence=0.9), span(4, 4, confidence=0.2)])
    assert out[0].confidence == 0.2


def test_merge_marks_a_merged_span_inexact_if_either_half_was():
    out = merge([span(0, 4, exact=True), span(4, 4, exact=False)])
    assert not out[0].exact


def test_two_chords_quantized_onto_one_beat_keep_the_one_with_more_evidence():
    """A 300 ms Am inside a bar of C rounds onto C's own downbeat, and then the
    two are not two chords — they are two readings of one slot. The second used
    to be dropped silently, so which chord the bar showed was decided by the
    sort: a passing chord could evict the chord the bar is actually in. The
    longer one wins, because it is the one the bar is spent on."""
    out = merge([span(0, 1, root=9, quality=MINOR), span(0, 4, root=0)])
    assert [(s.root_pc, s.length_beats) for s in out] == [(0, 4)]


def test_and_on_a_tie_the_one_the_engine_believed():
    """Same start, same length — nothing left to separate them but how sure the
    engine was, which is the same evidence the consensus vote consults."""
    out = merge([span(0, 4, root=9, quality=MINOR, confidence=0.3),
                 span(0, 4, root=0, confidence=0.9)])
    assert [(s.root_pc, s.confidence) for s in out] == [(0, 0.9)]


# --- drop_short -------------------------------------------------------------

def test_dropped_spans_give_their_beats_to_the_neighbour():
    """Not "delete": the beats have to go somewhere, and handing them back is
    what makes a run of jitter read as one held chord."""
    out = drop_short([span(0, 4, root=0), span(4, 1, root=1), span(5, 4, root=7)],
                     min_beats=2)
    assert [(s.root_pc, s.length_beats) for s in out] == [(0, 5), (7, 4)]


def test_a_too_short_first_span_is_absorbed_forward():
    out = drop_short([span(0, 1, root=1), span(1, 4, root=7)], min_beats=2)
    assert [(s.start_beat, s.length_beats, s.root_pc) for s in out] == [(0, 5, 7)]


# --- N/C handling -----------------------------------------------------------

def test_a_gap_is_filled_by_holding_the_previous_chord():
    """§18: hold the previous chord. Never emit a section with empty chords — the
    importer silently drops it."""
    out = hold_through_gaps([span(0, 4, root=0), span(12, 4, root=7)])
    assert [(s.start_beat, s.length_beats, s.root_pc) for s in out] == [(0, 12, 0), (12, 4, 7)]


def test_a_gap_at_the_start_pulls_the_first_chord_back_to_beat_zero():
    out = hold_through_gaps([span(4, 4, root=7)])
    assert (out[0].start_beat, out[0].length_beats) == (0, 8)


def test_the_timeline_is_extended_to_a_known_total():
    out = hold_through_gaps([span(0, 4, root=0)], total_beats=16)
    assert out[-1].end_beat == 16


# --- no simplification ------------------------------------------------------

def test_the_chain_leaves_the_harmony_alone():
    """The §5.5 tiers used to be step 5 of this chain, and `easy` collapsed a
    Cmaj7 next to a C onto one held C. Both chords were played, so both are
    charted: this chain quantizes, merges and fills, and it never renames."""
    spans = [span(0, 4, root=0, quality=MAJOR7), span(4, 4, root=0, quality=MAJOR),
             span(8, 4, root=9, quality=MINOR)]
    out = merge(spans)
    assert [(s.root_pc, s.quality) for s in out] == [
        (0, MAJOR7), (0, MAJOR), (9, MINOR)]


def test_a_passing_chord_of_a_full_bar_is_never_dropped():
    """`easy` dropped changes shorter than a bar, so a two-beat passing chord
    disappeared. The floor is one *beat* again — the jitter floor — and a chord
    the band played for half a bar is a chord the band played."""
    spans = [span(0, 8, root=0), span(8, 2, root=2, quality=MINOR), span(10, 8, root=7)]
    assert [s.root_pc for s in drop_short(spans)] == [0, 2, 7]


# --- confidence -------------------------------------------------------------

def test_confidence_is_weighted_by_how_long_a_chord_is_heard():
    """A track with one doubtful two-beat chord among thirty confident ones is a
    confident track; a flat mean would say otherwise."""
    spans = [span(0, 30, confidence=1.0), span(30, 2, root=7, confidence=0.0)]
    assert mean_confidence(spans) > 0.9


def test_exact_ratio_reports_how_much_survived_normalization():
    spans = [span(0, 8, exact=True), span(8, 8, root=7, exact=False)]
    assert exact_ratio(spans) == 0.5


# --- the whole chain --------------------------------------------------------

def test_the_known_song_survives_the_chain_unchanged():
    """G–D–Em–C, four beats each, four times. Nothing in §5.4 should touch a
    chart that was already clean."""
    out = process(known_chords(), known_axis())
    assert len(out) == 16
    assert [s.root_pc for s in out[:4]] == [7, 2, 4, 0]
    assert all(s.length_beats == 4 for s in out)
