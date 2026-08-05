"""§20.4 — the vote, and the three gates that make overwriting safe.

This is the most dangerous module in the service: it edits chords the engine
reported, and nothing downstream can catch a bad edit. `lint` and `lint_sync`
both check the song against *itself*, and a chart that has been made uniformly
self-consistent is precisely what they are least able to complain about — the
failure `axis.py` records as having cost 23 points behind three hundred green
tests.

So the tests here are mostly about what consensus **refuses** to do. Each gate
gets its own case, stated as the musical situation it exists to protect:

1. one reading against one reading teaches nothing (support),
2. a distant chord is the music changing, not a mishearing (near-miss),
3. an engine that was *confident* is telling us something (confidence).

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
from app.chords import MAJOR, MINOR

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
