"""§21 — the chart states the song's form, not a transcript of every pass.

The layer that answers "our system doesn't know that a verse is the same across
the whole song". Two things are under test and they are separate claims:

- **`canonicalize`** — every occurrence of a repeat group plays the group's own
  progression, decided beat by beat over the occurrences;
- **`settle_to_bars`** — in a song measured to change chord a bar at a time, a
  change the engine put a beat early belongs on the barline.

The second one's gate is the interesting half: applied to a song that really does
move twice a bar it would destroy the song, so `harmonic_unit` has to be able to
tell those apart from the chords alone.
"""

from __future__ import annotations

from app.analysis.canon import (
    MIN_COHESION,
    canonicalize,
    harmonic_unit,
    settle_to_bars,
    settles_on_barlines,
)
from app.analysis.form import RepeatGroup, detect
from app.analysis.structure import BarChord
from app.analysis.types import GridSpan
from app.chords import DOMINANT7, MAJOR, MAJOR7, MINOR

C, D, E, F, G, A, B = 0, 2, 4, 5, 7, 9, 11


def _bar(root, quality=MAJOR, confidence=1.0):
    return [BarChord(root_pc=root, quality=quality, start_beat=0.0,
                     length_beats=4.0, confidence=confidence)]


def _split(first, second, at=2.0, quality=MAJOR, confidence=1.0):
    """A bar holding two chords, `at` beats in."""
    return [BarChord(root_pc=first, quality=quality, start_beat=0.0,
                     length_beats=at, confidence=confidence),
            BarChord(root_pc=second, quality=quality, start_beat=at,
                     length_beats=4.0 - at, confidence=confidence)]


def _names(bar):
    return [(c.root_pc, c.quality, c.start_beat, c.length_beats) for c in bar]


def _group(length, occurrences, cohesion=1.0):
    return RepeatGroup(label="A", length_bars=length, occurrences=list(occurrences),
                       cohesion=cohesion)


# ---------------------------------------------------------------------------
# canonicalize
# ---------------------------------------------------------------------------

def test_every_pass_of_a_section_ends_up_playing_the_same_progression():
    """The whole point. Three passes of `C G Am F` where the engine heard the
    fourth bar three different ways come out as three identical passes."""
    bars = (
        [_bar(C), _bar(G), _bar(A, MINOR), _bar(F)]
        + [_bar(C), _bar(G), _bar(A, MINOR), _split(F, D)]
        + [_bar(C), _bar(G), _bar(A, MINOR), _split(F, E)]
    )
    out, report = canonicalize(bars, [_group(4, (0, 4, 8))], bar_beats=4)

    assert [_names(b) for b in out[0:4]] == [_names(b) for b in out[4:8]]
    assert [_names(b) for b in out[4:8]] == [_names(b) for b in out[8:12]]
    assert _names(out[3]) == _names(_bar(F)), "the majority reading, not a synthesis"
    assert report.canonical_bars == 2


def test_the_agreed_bar_is_settled_beat_by_beat_not_as_a_whole():
    """Four passes: two say `| G |`, two say `| G D |` with the D on beat 4.

    Whole-bar voting sees a 2–2 split of composite objects and gives up. Beat
    voting sees four passes agreeing on beats 1–3 and disagreeing on beat 4, and
    the disagreement is settled — or declined — on its own.
    """
    bars = [_bar(G), _split(G, D, at=3.0), _bar(G), _split(G, D, at=3.0)]
    out, _ = canonicalize(bars, [_group(1, (0, 1, 2, 3))], bar_beats=4)

    first, second, third, fourth = (c[0] for c in (out[0], out[1], out[2], out[3]))
    assert first.root_pc == second.root_pc == third.root_pc == fourth.root_pc == G
    assert all(len(bar) == len(out[0]) for bar in out), "all four passes now agree"


def _beats(*roots):
    """A bar of four one-beat chords."""
    return [BarChord(root_pc=root, quality=MAJOR, start_beat=float(index),
                     length_beats=1.0, confidence=1.0)
            for index, root in enumerate(roots)]


def test_a_tied_beat_holds_the_chord_the_bar_had_already_settled_on():
    """Four passes agreeing on beats 1, 2 and 4 and splitting four ways on beat 3.

    The tie is not left as a hole — `_as_flat` refuses a chord that does not fill
    its bar, and §18 has no rest to put there — and it is not filled with the
    bar's *first* chord either. It holds beat 2, because a chord that has begun
    sounding keeps sounding until something replaces it, which is the same
    asymmetry `postprocess.fill` uses.
    """
    bars = [_beats(C, G, D, F), _beats(C, G, E, F),
            _beats(C, G, A, F), _beats(C, G, B, F)]
    out, report = canonicalize(bars, [_group(1, (0, 1, 2, 3))], bar_beats=4)

    assert report.held_beats == 1
    for bar in out:
        assert [(c.root_pc, c.start_beat, c.length_beats) for c in bar] == [
            (C, 0.0, 1.0), (G, 1.0, 2.0), (F, 3.0, 1.0),
        ], "beat 3 held the G, and the two G beats came back as one span"


def test_a_tie_on_the_first_beat_reaches_forwards_instead():
    """The one case with nothing behind it. Held backwards there is no answer, so
    the head takes the first thing the bar does settle on."""
    bars = [_beats(D, G, F, F), _beats(E, G, F, F),
            _beats(A, G, F, F), _beats(B, G, F, F)]
    out, _ = canonicalize(bars, [_group(1, (0, 1, 2, 3))], bar_beats=4)

    for bar in out:
        assert bar[0].root_pc == G and bar[0].start_beat == 0.0
        assert sum(c.length_beats for c in bar) == 4.0, "and the bar is still whole"


def test_no_canonical_bar_ever_has_a_hole_in_it():
    bars = [_split(G, D, at=3.0), _split(G, E, at=3.0),
            _split(G, F, at=3.0), _split(G, A, at=3.0)]
    out, _ = canonicalize(bars, [_group(1, (0, 1, 2, 3))], bar_beats=4)

    for bar in out:
        covered = sorted((c.start_beat, c.start_beat + c.length_beats) for c in bar)
        assert covered[0][0] == 0.0
        assert covered[-1][1] == 4.0
        for (_, end), (start, _) in zip(covered, covered[1:]):
            assert end == start, "no hole anywhere in a canonical bar"


def test_a_group_that_barely_coheres_is_left_exactly_as_it_was_heard():
    """Below `MIN_COHESION` the premise has failed: these blocks clustered on a
    technicality and are not one piece of music played twice."""
    bars = [_bar(C), _bar(G), _bar(F), _bar(D)]
    out, report = canonicalize(bars, [_group(2, (0, 2), cohesion=MIN_COHESION - 0.01)],
                               bar_beats=4)

    assert [_names(b) for b in out] == [_names(b) for b in bars]
    assert report.groups_declined == 1
    assert report.canonical_bars == 0


def test_a_group_that_occurs_once_is_never_touched():
    bars = [_bar(C), _bar(G)]
    out, report = canonicalize(bars, [_group(2, (0,))], bar_beats=4)
    assert [_names(b) for b in out] == [_names(b) for b in bars]
    assert not report.touched


def test_identical_occurrences_are_a_no_op():
    """The property the whole §20/§21 stack is judged by. Perfect input arrives
    with every pass identical, so the agreed reading *is* the reading."""
    verse = [_bar(C), _bar(G), _bar(A, MINOR), _bar(F)]
    bars = verse * 4
    out, report = canonicalize(bars, [_group(4, (0, 4, 8, 12))], bar_beats=4)

    assert [_names(b) for b in out] == [_names(b) for b in bars]
    assert report.canonical_bars == 0
    assert report.split_bars == 0


def test_the_winner_is_always_a_chord_some_pass_actually_played():
    """Never invents. Four passes reading a bar four ways still produce one of
    those four, or nothing."""
    bars = [_bar(C), _bar(D), _bar(E), _bar(F)]
    out, _ = canonicalize(bars, [_group(1, (0, 1, 2, 3))], bar_beats=4)
    for bar in out:
        for chord in bar:
            assert chord.root_pc in (C, D, E, F)


def test_a_tie_is_settled_by_what_the_song_plays_everywhere_else():
    """The verse's third bar is heard `F#` twice and `F#m` twice — an exact tie
    the count cannot break. The rest of the song plays F#m, and that is the only
    evidence there is."""
    verse = [_bar(A), _bar(D), _bar(A + 9, MAJOR), _bar(E)]        # F# major
    minor = [_bar(A), _bar(D), _bar(A + 9, MINOR), _bar(E)]
    chorus = [_bar(A + 9, MINOR), _bar(D), _bar(A), _bar(E)] * 4   # F# minor, a lot
    bars = verse + minor + verse + minor + chorus

    out, _ = canonicalize(bars, [_group(4, (0, 4, 8, 12))], bar_beats=4)
    assert [bar[0].quality for bar in (out[2], out[6], out[10], out[14])] == [MINOR] * 4


def test_the_tie_break_never_moves_a_root():
    """Same gate as §20.8: the song's vocabulary may respell a chord and may not
    replace it.

    `C` against `Am` is the case that needs guarding, because `SNAP_TO` *does*
    allow minor to become major — so the quality gate waves this pair through and
    only the root test stands between the song's mass and a chord nobody played
    in that bar. The relative-major/minor confusion is the largest error bucket
    BTC has, and mass cannot settle it: both chords are in the vocabulary of the
    same song.
    """
    bars = [_bar(C), _bar(A, MINOR), _bar(C), _bar(A, MINOR)] + [_bar(C)] * 8
    out, report = canonicalize(bars, [_group(1, (0, 1, 2, 3))], bar_beats=4)

    assert [(bar[0].root_pc, bar[0].quality) for bar in out[:4]] == [
        (C, MAJOR), (A, MINOR), (C, MAJOR), (A, MINOR)]
    assert report.split_bars == 1


def test_the_tie_break_only_makes_moves_the_corpus_measured():
    """`SNAP_TO` is the measured table, so a seventh may flatten onto its triad
    and a triad may never grow one — however much of the song is spent on the
    seventh."""
    bars = ([_bar(C, MAJOR7), _bar(C, MAJOR), _bar(C, MAJOR7), _bar(C, MAJOR)]
            + [_bar(C, MAJOR7)] * 8)
    out, report = canonicalize(bars, [_group(1, (0, 1, 2, 3))], bar_beats=4)

    assert [bar[0].quality for bar in out[:4]] == [MAJOR7, MAJOR, MAJOR7, MAJOR]
    assert report.split_bars == 1


def test_two_occurrences_that_disagree_have_nothing_to_count():
    """One reading against one reading is not a vote. Both passes keep what they
    played, and the slot is reported as split."""
    bars = [_bar(C), _bar(G)]
    out, report = canonicalize(bars, [_group(1, (0, 1))], bar_beats=4)

    assert [bar[0].root_pc for bar in out] == [C, G]
    assert report.split_bars == 1


def test_one_occurrence_covering_a_beat_alone_does_not_speak_for_the_others():
    """`MIN_AGREEING`, on the case that reaches it: a beat only *one* occurrence
    covers has no dissenter, so a plurality test cannot refuse it and a floor has
    to. The second pass here is a truncated tail — it holds C for two beats and
    nothing after — and the first pass's G on beats 3 and 4 is one pass agreeing
    with itself.
    """
    truncated = [BarChord(root_pc=C, quality=MAJOR, start_beat=0.0,
                          length_beats=2.0, confidence=1.0)]
    bars = [_beats(C, C, G, G), truncated]
    out, report = canonicalize(bars, [_group(1, (0, 1))], bar_beats=4)

    assert [(c.root_pc, c.length_beats) for c in out[0]] == [(C, 4.0)], \
        "beats 3 and 4 had one witness, so they held the C the pair agreed on"
    assert report.held_beats == 2


def test_canonical_occurrences_let_the_form_pass_collapse_them_with_repeats():
    """What §21 is *for*, end to end. Before it, `repeats` fired only on
    synthetic input: real occurrences were never byte-identical, so a song
    compiled as one flat run of every bar the engine emitted."""
    noisy = (
        [_bar(C), _bar(G), _bar(A, MINOR), _bar(F)]
        + [_bar(C), _bar(G), _bar(A, MINOR), _split(F, D)]
        + [_bar(C), _bar(G), _bar(A, MINOR), _bar(F)]
        + [_bar(C), _bar(G), _bar(A, MINOR), _split(F, E)]
    )
    before, _ = detect(noisy, bar_beats=4.0)
    assert max(s.repeats for s in before) == 1, "the defect, pinned"

    _, groups = detect(noisy, bar_beats=4.0)
    canonical, _ = canonicalize(noisy, groups, bar_beats=4)
    after, _ = detect(canonical, bar_beats=4.0)

    assert len(after) == 1
    assert after[0].repeats == 4
    assert len(after[0].bars) == 4


# ---------------------------------------------------------------------------
# harmonic rhythm
# ---------------------------------------------------------------------------

def _span(start, length, root, quality=MAJOR):
    return GridSpan(start_beat=start, length_beats=length, root_pc=root, quality=quality)


def test_a_song_that_changes_once_a_bar_is_recognised_as_one():
    spans = [_span(i * 4, 4, root) for i, root in enumerate([C, G, A, F] * 8)]
    assert harmonic_unit(spans) == 4
    assert settles_on_barlines(spans, 4)


def test_a_song_that_changes_twice_a_bar_is_left_alone():
    """Wonderwall's case, and the reason the gate exists at all: `| F#m7 A |` is
    a real two-chord bar and settling it would delete half the song."""
    spans = [_span(i * 2, 2, root) for i, root in enumerate([C, G, A, F] * 8)]
    assert harmonic_unit(spans) == 2
    assert not settles_on_barlines(spans, 4)


def test_the_harmonic_unit_is_not_moved_by_engine_flicker():
    """Duration-weighted, so a recognizer stuttering four times inside one held
    chord does not turn a one-chord-a-bar song into a fast one. A plain median
    over the same spans returns 1."""
    spans = []
    for index, root in enumerate([C, G, A, F] * 8):
        if index % 4 == 0:
            spans += [_span(index * 4, 1, root), _span(index * 4 + 1, 1, D),
                      _span(index * 4 + 2, 2, root)]
        else:
            spans.append(_span(index * 4, 4, root))
    assert harmonic_unit(spans) == 4


def test_a_change_a_beat_early_is_pulled_onto_the_barline():
    """The anticipation. `| E B | B |` is what the engine reports and `| E | B |`
    is what every chart of the song prints."""
    bars = [_split(E, B, at=3.0), _bar(B)]
    out, settled = settle_to_bars(bars, 4)

    assert settled == 1
    assert _names(out[0]) == _names(_bar(E))
    assert _names(out[1]) == _names(_bar(B))


def test_a_bar_genuinely_split_down_the_middle_is_not_settled():
    """Two chords holding half a bar each is a two-chord bar however the song's
    harmonic rhythm was measured — there is no reading of "the bar's chord" that
    covers it."""
    bars = [_split(C, G, at=2.0)]
    out, settled = settle_to_bars(bars, 4)

    assert settled == 0
    assert _names(out[0]) == _names(_split(C, G, at=2.0))


def test_settling_keeps_the_quality_it_settled_on():
    bars = [_split(G, D, at=3.5)]
    bars[0][0] = BarChord(root_pc=G, quality=DOMINANT7, start_beat=0.0,
                          length_beats=3.5, confidence=0.8)
    out, settled = settle_to_bars(bars, 4)

    assert settled == 1
    assert (out[0][0].root_pc, out[0][0].quality) == (G, DOMINANT7)
    assert out[0][0].length_beats == 4.0
