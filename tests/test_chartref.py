"""The reference-chart format, and the corpus written in it.

`bench/reference/*.chart` is the graded truth for `bench/lab.py grade`, so a
parser bug here does not read as a parser bug — it reads as an engine that cannot
hear a sharp. That is not hypothetical; it is what the first version did, and the
first test below is that defect.

The corpus itself is checked too, because a chart file that does not parse, or
that names a chord this grammar cannot read, is a broken measuring stick rather
than a failing measurement.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bench.chartref import ChartError, load_chart, parse_chart, render_chart

REFERENCE = Path(__file__).resolve().parent.parent / "bench" / "reference"


def _chart(body: str):
    return parse_chart("Title: T\nKey: A major\nTime: 4/4\n\n" + body)


# ---------------------------------------------------------------------------
# the format
# ---------------------------------------------------------------------------

def test_a_sharp_is_a_chord_and_not_the_start_of_a_comment():
    """`#` means both things in this format, and telling them apart is the whole
    of it.

    Read as a comment unconditionally — which is what `line.split("#", 1)[0]`
    does — every `F#` in the corpus silently became `F` and every `G#m` became
    `G`. I'm Yours parsed as a fifty-bar song with a two-chord vocabulary, and
    the grade sheet said the analysis was wrong about a song it had exactly
    right.
    """
    chart = _chart("[Verse]\n| B | F# | G#m | C#m |\n")
    assert [bar.head for bar in chart.bars] == ["B", "F#", "G#m", "C#m"]
    assert len(chart.bars) == 4


def test_a_comment_still_ends_the_line():
    chart = _chart("[Verse]\n| A | D |   # two bars, and this is not one of them\n")
    assert [bar.head for bar in chart.bars] == ["A", "D"]


def test_a_whole_line_comment_is_ignored():
    chart = _chart("# a note about the recording\n[Verse]\n| A |\n")
    assert [bar.head for bar in chart.bars] == ["A"]


def test_a_cell_with_several_chords_divides_the_bar():
    chart = _chart("[Verse]\n| Am G | D |\n")
    assert [bar.chords for bar in chart.bars] == [("Am", "G"), ("D",)]


def test_a_percent_repeats_the_previous_bar():
    chart = _chart("[Verse]\n| Am | % | D | |\n")
    assert [bar.head for bar in chart.bars] == ["Am", "Am", "D", "D"]


def test_one_line_is_one_phrase():
    chart = _chart("[Verse]\n| A | D |\n| A | E |\n")
    assert chart.sections[0].phrase_count == 2
    assert chart.sections[0].bars_per_phrase == [2, 2]


def test_the_parenthetical_wins_when_the_key_says_two_things():
    chart = parse_chart("Key: E minor (Dorian)\n\n[Verse]\n| Em |\n")
    assert (chart.tonic, chart.mode) == ("E", "dorian")


def test_an_unreadable_chord_names_its_line():
    with pytest.raises(ChartError) as error:
        _chart("[Verse]\n| A |\n| H |\n")
    assert "line 7" in str(error.value)


def test_render_round_trips():
    source = _chart("[Verse]\n| Am G | D |\n| Am | E |\n\n[Chorus]\n| F | C |\n")
    again = parse_chart(render_chart(source))
    assert [bar.chords for bar in again.bars] == [bar.chords for bar in source.bars]
    assert [s.bars_per_phrase for s in again.sections] == \
           [s.bars_per_phrase for s in source.sections]


def test_form_is_read_from_the_chords_and_not_from_the_labels():
    """Two sections with the same first phrase are the same letter however they
    are labelled — which is what lets a chart be compared against a payload that
    calls everything `Part N`."""
    chart = _chart("[Verse 1]\n| A | D |\n\n[Chorus]\n| F | C |\n\n[Interlude]\n| A | D |\n")
    assert chart.form == ["A", "B", "A"]


# ---------------------------------------------------------------------------
# the corpus
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", sorted(REFERENCE.glob("*.chart")), ids=lambda p: p.stem)
def test_every_reference_chart_parses(path):
    chart = load_chart(path)
    assert chart.title, f"{path.name}: no Title"
    assert chart.tonic, f"{path.name}: no Key"
    assert chart.bars, f"{path.name}: no bars"
    assert chart.tempo, f"{path.name}: no Tempo"


@pytest.mark.parametrize("path", sorted(REFERENCE.glob("*.chart")), ids=lambda p: p.stem)
def test_every_reference_chart_has_a_songs_worth_of_vocabulary(path):
    """Three to eight distinct chords. Not a style rule: a hand-written reference
    with fifteen chords in it is a transcription mistake, and grading against one
    would report the mistake as an engine defect."""
    assert 2 <= len(load_chart(path).vocabulary) <= 8, sorted(load_chart(path).vocabulary)
