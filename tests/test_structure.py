"""§15 — bars and sections.

`bars_from_spans` is the quiet load-bearing one: it is what puts the whole song
on whole bars, which is what §13.2's beat axis needs (see `app/sync.py`). If it
ever emits a partial bar, the sidecar's anchors start addressing beats the cursor
never lands on, and the failure shows up as a cursor drifting mid-song rather
than as anything that looks like a bug here.
"""

from __future__ import annotations

from app.analysis.postprocess import process
from app.analysis.structure import (
    MIN_SECTION_BARS,
    BarChord,
    bars_from_spans,
    segment,
)
from app.analysis.types import GridSpan
from app.chords import MAJOR, MINOR, NORMAL
from tests.conftest import known_chords, known_grid


def span(start, length, root=0, quality=MAJOR):
    return GridSpan(start_beat=start, length_beats=length, root_pc=root, quality=quality)


# --- bars -------------------------------------------------------------------

def test_one_chord_per_bar_becomes_one_span_per_bar():
    bars = bars_from_spans([span(0, 4, root=0), span(4, 4, root=7)], 4)
    assert len(bars) == 2
    assert [b[0].root_pc for b in bars] == [0, 7]
    assert all(b[0].start_beat == 0 and b[0].length_beats == 4 for b in bars)


def test_a_chord_held_across_a_barline_becomes_one_span_in_each_bar():
    """The same shape `SongSection.materializedBars` produces on device. A single
    span overrunning its bar would be rejected by the lint and truncated by the
    app."""
    bars = bars_from_spans([span(0, 8, root=0)], 4)
    assert len(bars) == 2
    for bar in bars:
        assert bar[0].start_beat == 0 and bar[0].length_beats == 4


def test_a_mid_bar_change_produces_two_spans_in_one_bar():
    bars = bars_from_spans([span(0, 2, root=0), span(2, 2, root=7)], 4)
    assert len(bars) == 1
    assert [(c.root_pc, c.start_beat, c.length_beats) for c in bars[0]] == [
        (0, 0.0, 2.0), (7, 2.0, 2.0),
    ]


def test_a_trailing_partial_bar_is_dropped():
    """A partial bar has no downbeat to anchor and would put every later bar off
    the grid."""
    bars = bars_from_spans([span(0, 4, root=0), span(4, 2, root=7)], 4)
    assert len(bars) == 1


def test_no_bar_ever_overflows_its_meter():
    spans = process(known_chords(), known_grid(), difficulty=NORMAL)
    for bar in bars_from_spans(spans, 4):
        for chord in bar:
            assert chord.start_beat + chord.length_beats <= 4 + 1e-9


# --- sections ---------------------------------------------------------------

def _bar(root, quality=MAJOR):
    return [BarChord(root_pc=root, quality=quality, start_beat=0.0, length_beats=4.0)]


def test_a_repeated_progression_collapses_into_repeats():
    """§15: "a 4-bar progression played 4× is one section with repeats: 4, not 16
    bars of explicit chords"."""
    progression = [_bar(7), _bar(2), _bar(4, MINOR), _bar(0)]
    sections = segment(progression * 4)
    assert len(sections) == 1
    assert sections[0].repeats == 4
    assert sections[0].total_bars == 16


def test_two_different_progressions_become_two_sections():
    verse = [_bar(7), _bar(2), _bar(4, MINOR), _bar(0)]
    chorus = [_bar(0), _bar(0), _bar(7), _bar(7)]
    sections = segment(verse * 2 + chorus * 2)
    assert len(sections) == 2
    assert [s.repeats for s in sections] == [2, 2]
    assert sections[1].start_bar == 8


def test_a_near_identical_neighbour_is_folded_in_rather_than_split_off():
    """The player reads the section rail, so an extra boundary is a visible
    mistake in a way one slightly-wrong chord is not."""
    verse = [_bar(7), _bar(2), _bar(4, MINOR), _bar(0)]
    variant = [_bar(7), _bar(2), _bar(4, MINOR), _bar(9, MINOR)]   # last bar differs
    sections = segment(verse + variant)
    assert len(sections) == 1
    assert sections[0].total_bars == 8


def test_no_section_is_shorter_than_the_floor():
    verse = [_bar(7), _bar(2), _bar(4, MINOR), _bar(0)]
    runt = [_bar(5), _bar(9, MINOR)]
    sections = segment(verse * 2 + runt)
    assert all(s.total_bars >= MIN_SECTION_BARS for s in sections)


def test_sections_tile_the_song_with_no_gaps():
    """Sections concatenate on device, so a gap or overlap here slides every
    later bar off the recording."""
    verse = [_bar(7), _bar(2), _bar(4, MINOR), _bar(0)]
    chorus = [_bar(0), _bar(0), _bar(7), _bar(7)]
    bars = verse * 2 + chorus * 2 + verse
    sections = segment(bars)
    cursor = 0
    for section in sections:
        assert section.start_bar == cursor
        cursor += section.total_bars
    assert cursor == len(bars)


def test_without_an_energy_hint_sections_are_honestly_unnamed():
    """§15's fallback. Naming a section requires evidence; "Part 1" is what we
    have. **Never** name a section from lyrics — §2.4 means we must not have
    lyrics at all."""
    verse = [_bar(7), _bar(2), _bar(4, MINOR), _bar(0)]
    chorus = [_bar(0), _bar(0), _bar(7), _bar(7)]
    sections = segment(verse * 2 + chorus * 2)
    assert [s.kind for s in sections] == ["custom", "custom"]
    assert [s.name for s in sections] == ["Part 1", "Part 2"]


def test_with_an_energy_hint_the_loudest_repeated_block_is_the_chorus():
    verse = [_bar(7), _bar(2), _bar(4, MINOR), _bar(0)]
    chorus = [_bar(0), _bar(0), _bar(7), _bar(7)]
    bars = verse * 2 + chorus * 2
    energy = [0.3] * 8 + [0.9] * 8
    sections = segment(bars, energy=energy)
    assert [s.kind for s in sections] == ["verse", "chorus"]
    assert all(s.name == "" for s in sections), "empty name = use the kind's own label"
