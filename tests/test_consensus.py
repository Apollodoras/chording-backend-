"""§20.4 — the vote, and the three gates that make overwriting safe.

This is the most dangerous module in the service: it edits chords the engine
reported, and nothing downstream can catch a bad edit. `lint` and `lint_sync`
both check the song against *itself*, and a chart that has been made uniformly
self-consistent is precisely what they are least able to complain about — the
failure `axis.py` records as having cost 23 points behind three hundred green
tests.

So the tests here are mostly about what consensus **refuses** to do. Each gate
gets its own case, stated as the musical situation it exists to protect:

1. one reading against one reading teaches nothing, and neither does an even
   split (the plurality, with its floor of two agreeing passes),
2. a distant chord is the music changing, not a mishearing (near-miss),
3. an engine that was *confident* is telling us something (confidence).

Gate 1 has a second set of cases below the ones it refuses, because it is the
gate that was **wrong** rather than merely strict: as a two-thirds share it threw
away the slot whenever the engine misheard the same bar in two passes of four,
which is the ordinary noise rate and the reason this layer was reported as not
working. Every extra bad reading pushed the share further down, so repetition —
the whole evidence the vote runs on — counted against it.

And the property the whole design rests on: **on perfect input, consensus is a
no-op** — provable from gate 3 rather than observed, and asserted here as well as
in `bench/run_bench.py`'s truth-as-engines run.
"""

from __future__ import annotations

from app.analysis import consensus
from app.analysis.form import RepeatGroup, detect
from app.analysis.strumming import SUPPORT_THRESHOLD
from app.analysis.structure import BarChord
from app.analysis.types import Onset
from app.chords import MAJOR, MINOR, MINOR7

C, D, E, F, G, A = 0, 2, 4, 5, 7, 9


def bar(root, quality=MAJOR, confidence=1.0) -> list[BarChord]:
    return [BarChord(root_pc=root, quality=quality, start_beat=0.0,
                     length_beats=4.0, confidence=confidence)]


def group(*starts, length=4) -> RepeatGroup:
    return RepeatGroup(label="A", length_bars=length, occurrences=list(starts))


def verse(last=(C, MAJOR), confidence=1.0) -> list[list[BarChord]]:
    """G D Em <last> — the folk default, with the final bar parameterised
    because that is the bar every test below disagrees about."""
    return [bar(G), bar(D), bar(E, MINOR),
            bar(last[0], last[1], confidence=confidence)]


# The two ways the last bar can disagree with the other verses, and the whole
# point of `harmony.is_near_miss`: Am shares two notes with C (a mishearing),
# F shares one (a chord change).
NEAR = (A, MINOR)
DISTANT = (F, MAJOR)
# A *second* near-miss of C, for the cases about two dissenters who disagree with
# each other as well as with the majority: Em shares E and G with C.
OTHER_NEAR = (E, MINOR)


# --- the property the design rests on ----------------------------------------

def test_perfect_input_is_a_no_op():
    """Ground truth arrives at a flat confidence of 1.0, so no dissenter is ever
    *less* believed than a winner and gate 3 can never open. Nothing is
    rewritten — not because the corpus happened to come out that way, but
    because the construction forbids it. This is what makes the benchmark's
    truth-as-engines run a regression guard rather than a hopeful number."""
    bars = verse() + verse(DISTANT) + verse() + verse()     # verse 2 genuinely differs
    out, report = consensus.apply(bars, [group(0, 4, 8, 12)], bar_beats=4.0)
    assert report.rewritten_bars == 0
    assert out == bars


# --- gate 1: the vote has to be decisive -------------------------------------

def test_one_reading_against_one_reading_never_votes():
    """With two occurrences a disagreement is 1–1. Counting cannot tell which is
    right, so the slot is contested and both are left exactly as played."""
    bars = verse() + verse(NEAR, confidence=0.1)
    out, report = consensus.apply(bars, [group(0, 4)], bar_beats=4.0)
    assert report.rewritten_bars == 0
    assert report.contested_bars == 1
    assert out[7] == bars[7]


# --- gate 2: a distant chord is the music, not a mistake ---------------------

def test_a_genuinely_different_chord_is_never_flattened():
    """Three verses end on C and the fourth ends on F — a fifth away, which no
    recognizer arrives at by accident. Even at rock-bottom confidence, and even
    outvoted three to one, the F stands: this is the case where "standardizing"
    would show the player a chord that is not being played."""
    bars = verse() + verse() + verse() + verse(DISTANT, confidence=0.05)
    out, report = consensus.apply(bars, [group(0, 4, 8, 12)], bar_beats=4.0)
    assert report.rewritten_bars == 0
    assert report.contested_bars == 1
    assert out[15][0].root_pc == F, "the real chord change survived the vote"


# --- gate 3: a confident dissenter is evidence -------------------------------

def test_a_confident_dissenter_outranks_the_majority():
    """The engine heard Am in verse 4 and was as sure of it as it was of the C
    in the other three. That is evidence the music changed, and counting alone
    cannot see it — which is exactly why the vote consults confidence."""
    bars = verse() + verse() + verse() + verse(NEAR, confidence=1.0)
    for one_bar in bars[:12]:
        one_bar[0] = BarChord(root_pc=one_bar[0].root_pc, quality=one_bar[0].quality,
                              start_beat=0.0, length_beats=4.0, confidence=1.0)
    out, report = consensus.apply(bars, [group(0, 4, 8, 12)], bar_beats=4.0)
    assert report.rewritten_bars == 0
    assert out[15][0].root_pc == A


# --- what it is actually for -------------------------------------------------

def test_a_doubtful_near_miss_is_voted_into_line():
    """The case the layer exists for. Three confident passes say C; the fourth
    says Am — two of three notes shared, and the engine hedged. That is a
    mishearing, and four verses of one song are four readings of one signal."""
    bars = verse() + verse() + verse() + verse(NEAR, confidence=0.2)
    out, report = consensus.apply(bars, [group(0, 4, 8, 12)], bar_beats=4.0)
    assert report.rewritten_bars == 1
    assert out[15][0].root_pc == C, "the misheard bar now agrees with its repeats"
    assert report.contested_bars == 0


def test_the_corrected_bar_carries_the_groups_confidence():
    """The bar now claims what the group claims, and is believed as much as the
    group believed it — carrying the loser's confidence forward would understate
    a bar we have more evidence for than any single pass."""
    bars = verse() + verse() + verse() + verse(NEAR, confidence=0.2)
    out, _ = consensus.apply(bars, [group(0, 4, 8, 12)], bar_beats=4.0)
    assert out[15][0].confidence == 1.0


def test_only_the_offending_bar_moves():
    bars = verse() + verse() + verse() + verse(NEAR, confidence=0.2)
    out, _ = consensus.apply(bars, [group(0, 4, 8, 12)], bar_beats=4.0)
    assert [b[0].root_pc for b in out[12:16]] == [G, D, E, C]


# --- gate 1, the other half: a plurality, not a share ------------------------

def test_two_agreeing_passes_carry_against_two_dissenters_who_disagree():
    """The case the old two-thirds share got wrong, and the one the layer was
    reported for. Four verses, and the engine misheard the same bar in two of
    them — Am once, Em the other time, hedging on both. A share rule counts that
    as 2-of-4 and files the slot as contested, so *both* mistakes ship; and every
    additional bad reading pushes the share further down, which means the
    repetition that was supposed to be the evidence counts against it.

    Two passes agreeing exactly while the dissenters disagree with the majority
    and with each other is the signature of noise: a song that really changes its
    fourth verse changes it the same way every time it plays it."""
    bars = verse() + verse(NEAR, confidence=0.2) + verse() + verse(OTHER_NEAR, confidence=0.3)
    out, report = consensus.apply(bars, [group(0, 4, 8, 12)], bar_beats=4.0)
    assert report.rewritten_bars == 2
    assert report.contested_bars == 0
    assert [b[3][0].root_pc for b in (out[0:4], out[4:8], out[8:12], out[12:16])] == [C] * 4


def test_an_even_split_is_still_contested():
    """Two passes end on C and two on Am, all four believed the same. That is not
    a mishearing with a majority against it — it is a song with two versions of
    its verse, and the plurality rule has to see the difference."""
    bars = verse() + verse(NEAR) + verse() + verse(NEAR)
    out, report = consensus.apply(bars, [group(0, 4, 8, 12)], bar_beats=4.0)
    assert report.rewritten_bars == 0
    assert report.contested_bars == 1
    assert out == bars


def test_a_lone_agreeing_pass_is_not_a_vote():
    """Three passes, three different readings. Nothing agrees with anything, so
    there is no majority to speak with however doubtful the others are."""
    bars = verse() + verse(NEAR, confidence=0.2) + verse(OTHER_NEAR, confidence=0.2)
    out, report = consensus.apply(bars, [group(0, 4, 8)], bar_beats=4.0)
    assert report.rewritten_bars == 0
    assert report.contested_bars == 1
    assert out == bars


def test_a_group_with_one_occurrence_is_left_alone():
    bars = verse()
    out, report = consensus.apply(bars, [group(0)], bar_beats=4.0)
    assert report.groups_voted == 0 and out == bars


def test_the_input_is_never_mutated():
    """The benchmark diffs before against after; a vote that edited in place
    would make "what did consensus change?" unanswerable."""
    bars = verse() + verse() + verse() + verse(NEAR, confidence=0.2)
    before = [[(c.root_pc, c.quality) for c in one_bar] for one_bar in bars]
    consensus.apply(bars, [group(0, 4, 8, 12)], bar_beats=4.0)
    assert [[(c.root_pc, c.quality) for c in b] for b in bars] == before


def test_the_canonical_progression_is_read_back_after_voting():
    """So it is the progression the song settled on, not whichever occurrence
    happened to come first."""
    bars = verse(NEAR, confidence=0.2) + verse() + verse() + verse()
    repeat = group(0, 4, 8, 12)
    consensus.apply(bars, [repeat], bar_beats=4.0)
    assert [b[0].root_pc for b in repeat.canonical] == [G, D, E, C]


# --- patterns ----------------------------------------------------------------

def _onsets_every_beat(bars: int, bar_beats: int = 4, beat_ms: int = 500) -> list[Onset]:
    return [Onset(t_ms=i * beat_ms, strength=1.0) for i in range(bars * bar_beats)]


def test_a_pattern_pools_every_occurrence_of_its_group():
    """§14 already averages a section's bars onto one bar — that is what folding
    *is* — so pooling the other occurrences changes nothing about the estimator
    and multiplies its sample. A four-bar verse played four times goes from four
    bars of evidence to sixteen, which is the difference between the support
    threshold meaning something and it being a coin toss."""
    axis = [i * 500 for i in range(65)]

    class _Axis:
        times_ms = axis
        bar_beats = 4

        def position_at(self, t_ms):
            from app.analysis.axis import position_in
            return position_in(axis, t_ms)

    extracted = consensus.pattern_for_group(
        group(0, 4, 8, 12), onsets=_onsets_every_beat(16), axis=_Axis(),
        bar_beats=4.0, tempo=120, name="Verse strum", time_signature="4/4",
    )
    assert not extracted.is_fallback
    assert extracted.confidence >= SUPPORT_THRESHOLD
    assert [s.beat for s in extracted.pattern.strokes] == [0.0, 1.0, 2.0, 3.0]


def test_no_onsets_still_produces_a_playable_pattern():
    """The app requires a pattern; a boring one that plays is worth more than no
    song (§14)."""
    extracted = consensus.pattern_for_group(
        group(0, 4), onsets=[], axis=None, bar_beats=4.0, tempo=120,
        name="Verse strum", time_signature="4/4",
    )
    assert extracted.is_fallback and extracted.pattern.strokes


# --- end to end through the form layer ---------------------------------------

def test_voting_lets_a_noisy_repeat_collapse_into_repeats():
    """The compact encoding §15 asks for ("a 4-bar progression played 4× is one
    section with repeats: 4") is unreachable while one bar of one pass is wrong,
    because `repeats` demands the passes be identical. Fixing the bar is what
    makes the encoding available — which is why the model detects the form,
    votes, and then detects it again."""
    bars = verse() + verse() + verse() + verse(NEAR, confidence=0.2)
    before = detect(bars, bar_beats=4.0)[0]
    assert sum(s.repeats for s in before) < 4, "the noisy pass blocks collapsing"

    voted, _ = consensus.apply(bars, detect(bars, bar_beats=4.0)[1], bar_beats=4.0)
    after = detect(voted, bar_beats=4.0)[0]
    assert len(after) == 1 and after[0].repeats == 4


# --- §20.9: the slots the count cannot decide --------------------------------
#
# The user-visible complaint these answer: "the engine adds variants to a chord
# and the song ends up with more chords than it has". That complaint is a *tie*.
# BTC hears the seventh in roughly half the passes of a section, so no plurality
# ever forms, gate 1 files every one of those slots as contested, and both
# readings ship. `vocabulary.snap` cannot reach them either — it wants a 6:1 mass
# ratio and no more than `MAX_OCCASIONS` sightings, and a variant heard in four
# passes of eight is neither rare nor lopsided.

def test_a_tie_between_two_spellings_is_settled_by_which_was_believed():
    """Four passes read Em, four read Em7, and the vote has nothing to count.

    The seventh is the note the recognizer is least sure it heard, and here it
    says so: the plain readings carry 0.8 and the sevenths 0.6. That gap is the
    only evidence in the slot that was not copied along with the mistake — four
    passes of a doubled guitar part confuse BTC four times identically, so the
    repetition the vote runs on is four copies of one error.
    """
    bars = []
    for pass_index in range(8):
        seventh = pass_index % 2 == 1
        bars += [bar(G), bar(D),
                 bar(E, MINOR7 if seventh else MINOR,
                     confidence=0.6 if seventh else 0.8),
                 bar(C)]

    voted, report = consensus.apply(bars, [group(*range(0, 32, 4))], bar_beats=4.0)
    assert {b[0].quality for b in voted[2::4]} == {MINOR}, "Em7 should be gone"
    assert report.weighed_bars == 4 and report.contested_bars == 0


def test_belief_may_overrule_a_plurality_that_points_the_other_way():
    """Five hesitant sevenths against three confident triads.

    The count says Em7 and the confidences say Em. Gate 3 refuses the count's
    answer — correctly — and before §20.9 that left both readings in the chart.
    """
    bars = []
    for pass_index in range(8):
        seventh = pass_index >= 3
        bars += [bar(G), bar(D),
                 bar(E, MINOR7 if seventh else MINOR,
                     confidence=0.62 if seventh else 0.79),
                 bar(C)]

    voted, report = consensus.apply(bars, [group(*range(0, 32, 4))], bar_beats=4.0)
    assert {b[0].quality for b in voted[2::4]} == {MINOR}
    assert report.weighed_bars == 5


def test_belief_never_grows_a_seventh_onto_a_triad():
    """The direction is `vocabulary.SNAP_TO`'s, which is measured rather than
    reasoned: a reported seventh is the plain triad about twice as often as it is
    a seventh, and nothing in the corpus supports the move the other way. So a
    confident Em7 against a doubtful Em leaves both alone rather than spreading
    the seventh across the song."""
    bars = []
    for pass_index in range(8):
        seventh = pass_index % 2 == 1
        bars += [bar(G), bar(D),
                 bar(E, MINOR7 if seventh else MINOR,
                     confidence=0.85 if seventh else 0.55),
                 bar(C)]

    voted, report = consensus.apply(bars, [group(*range(0, 32, 4))], bar_beats=4.0)
    assert {b[0].quality for b in voted[2::4]} == {MINOR, MINOR7}
    assert report.weighed_bars == 0 and report.contested_bars == 1


def test_belief_never_moves_a_root_however_sure_it_is():
    """The gate that makes the extra power safe.

    A recognizer wobbling over colour leaves the roots and the barlines exactly
    where they were; music that changes moves one of them. So a slot where a
    *root* differs is never shown to the belief reduction at all, whatever the
    confidences say.

    Set up as the tie §20.9 is otherwise allowed to settle — four passes against
    four, the minority believed far less — differing this time in the root. The
    count refuses it (no plurality) and belief must refuse it too, so the slot
    stays contested and both readings survive.
    """
    bars = []
    for pass_index in range(8):
        wandered = pass_index % 2 == 1
        bars += [bar(G), bar(D), bar(E, MINOR),
                 bar(A if wandered else C, MINOR if wandered else MAJOR,
                     confidence=0.3 if wandered else 0.9)]

    voted, report = consensus.apply(bars, [group(*range(0, 32, 4))], bar_beats=4.0)
    assert voted == bars, "a root may never be voted away by confidence alone"
    assert report.weighed_bars == 0 and report.contested_bars == 1


def test_belief_is_a_no_op_on_perfect_input():
    """The property every layer in §20 is judged by, inherited here by
    construction: ground truth arrives at a flat confidence of 1.0, so no reading
    is ever *less believed* than another and gate C can never open. Asserted on
    the shape §20.9 exists for — an even split that the count also refuses."""
    bars = []
    for pass_index in range(8):
        seventh = pass_index % 2 == 1
        bars += [bar(G), bar(D), bar(E, MINOR7 if seventh else MINOR), bar(C)]

    voted, report = consensus.apply(bars, [group(*range(0, 32, 4))], bar_beats=4.0)
    assert voted == bars
    assert report.rewritten_bars == 0 and report.weighed_bars == 0


def test_the_belief_reduction_can_be_turned_off():
    """`CHORDS_THEORY_BELIEF=off`, the same posture the other two layers support.
    With it off the slot stays contested, which is this module's behaviour before
    §20.9."""
    bars = []
    for pass_index in range(8):
        seventh = pass_index % 2 == 1
        bars += [bar(G), bar(D),
                 bar(E, MINOR7 if seventh else MINOR,
                     confidence=0.6 if seventh else 0.8),
                 bar(C)]

    voted, report = consensus.apply(bars, [group(*range(0, 32, 4))],
                                    bar_beats=4.0, weigh=False)
    assert voted == bars
    assert report.weighed_bars == 0 and report.contested_bars == 1
