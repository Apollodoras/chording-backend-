"""§20.3 — finding the form when the engine got some of it wrong.

§15's segmentation was correct on clean input and helpless on real input,
because it compared bars by **exact equality** and only ever compared
**adjacent** blocks. These tests are mostly the cases that used to break it:
a verse with one misheard chord, a verse that comes back after the chorus, and
a song that starts with a two-bar intro.

The §15 behaviours that were right are still asserted in `test_structure.py`,
which drives `segment` — the same code, entered by the older door.
"""

from __future__ import annotations

from app.analysis.form import (
    CLUSTER_SIMILARITY,
    bar_similarity,
    cluster,
    detect,
    period,
    segment,
)
from app.analysis.structure import BarChord
from app.chords import MAJOR, MINOR

C, D, E, F, G, A = 0, 2, 4, 5, 7, 9


def _bar(root, quality=MAJOR):
    return [BarChord(root_pc=root, quality=quality, start_beat=0.0, length_beats=4.0)]


def _split(root_a, root_b):
    """A bar holding two chords, for the sampling comparison."""
    return [BarChord(root_pc=root_a, quality=MAJOR, start_beat=0.0, length_beats=2.0),
            BarChord(root_pc=root_b, quality=MAJOR, start_beat=2.0, length_beats=2.0)]


VERSE = [_bar(G), _bar(D), _bar(E, MINOR), _bar(C)]
CHORUS = [_bar(C), _bar(C), _bar(G), _bar(G)]
# The same verse with its last bar misheard as the relative minor — the single
# most common thing a chord recognizer does.
MISHEARD = [_bar(G), _bar(D), _bar(E, MINOR), _bar(A, MINOR)]

# Verse, verse, chorus, chorus, verse, verse. Deliberately not the alternating
# `V C V C`: with a verse and a chorus of equal length, alternation makes the
# real repeating unit **eight** bars rather than four, and the honest answer is
# one section played twice. That is a property of the music, not a bug — and it
# is what §15's "shortest progression that explains the song" has always said.
SONG = VERSE * 2 + CHORUS * 2 + VERSE * 2


# --- comparing bars by sound rather than by name -----------------------------

def test_a_misheard_bar_is_mostly_the_same_bar():
    """0.5, not 0. Under exact equality one wrong chord was as damaging as a
    completely different section, which is what destroyed the repetition the
    layer exists to exploit."""
    assert bar_similarity(_bar(C), _bar(A, MINOR), 4.0) == 0.5
    assert bar_similarity(_bar(C), _bar(C), 4.0) == 1.0
    assert bar_similarity(_bar(C), _bar(6), 4.0) == 0.0


def test_bars_are_compared_on_a_grid_so_shapes_can_differ():
    """A bar holding one chord and a bar holding it twice are the same bar; a
    bar that changes half way through is half the same."""
    assert bar_similarity(_bar(C), _split(C, C), 4.0) == 1.0
    assert bar_similarity(_bar(C), _split(C, 6), 4.0) == 0.5


# --- the three failures this module was written for --------------------------

def test_a_verse_with_one_wrong_chord_still_groups_with_its_clean_repeats():
    """The core new capability. Exact equality made this four unrelated blocks;
    now it is one group of four, which is what `consensus.py` needs before it
    can vote the wrong bar back into line."""
    sections, groups = detect(VERSE + VERSE + MISHEARD + VERSE, bar_beats=4.0)
    assert len(groups) == 1
    assert len(groups[0].occurrences) == 4


def test_a_verse_that_returns_after_the_chorus_is_the_same_group():
    """Non-adjacent repeats. The old segmenter only ever compared neighbours, so
    verse 2 and verse 1 were unrelated by construction — the song had no way to
    know it had come back."""
    sections, groups = detect(SONG, bar_beats=4.0)
    assert [s.group for s in sections] == ["A", "B", "A"]
    assert len(groups) == 2
    assert len(groups[0].occurrences) == 4, "all four verse blocks are one group"


def test_a_two_bar_intro_does_not_destroy_the_form():
    """Chop a repeating four-bar verse two bars late and every block becomes
    half of one verse and half of the next, matching nothing. The block grid's
    offset is chosen by how much repetition it exposes, which is exactly the
    thing a wrong offset destroys."""
    intro = [_bar(F), _bar(F)]
    sections, groups = detect(intro + VERSE * 4, bar_beats=4.0)
    verses = [g for g in groups if g.length_bars == 4 and len(g.occurrences) >= 4]
    assert verses, f"the verse was not recovered: {[(g.label, g.occurrences) for g in groups]}"


def test_a_two_bar_intro_does_not_destroy_the_form_at_section_level_either():
    """The assertion above passed all along — and the song was still ruined.

    Clustering got it right and *section assembly* threw the answer away: the
    runt intro was merged forwards into the verse, the merged section was handed
    the **intro's** group, and `repeats: 4` was flattened. The whole song came
    out as one eighteen-bar section carrying two bars' worth of strum evidence,
    while the verse group's pooled sixteen-bar pattern was computed and
    referenced by nothing. Groups are not the deliverable; sections are.
    """
    intro = [_bar(F), _bar(F)]
    sections, groups = detect(intro + VERSE * 4, bar_beats=4.0)

    assert len(sections) == 2, "the intro is still its own section"
    assert sections[0].total_bars == 2 and sections[0].repeats == 1
    assert sections[1].repeats == 4, "the verse keeps its collapsed encoding"
    assert sections[0].group != sections[1].group

    verse_group = next(g for g in groups if g.label == sections[1].group)
    assert len(verse_group.occurrences) == 4, "and the section points at the verse's group"


def test_the_last_occurrence_stays_in_its_group_when_the_song_ends_on_a_tag():
    """Real songs rarely end on an exact multiple of their own period, so this
    was the *last section of most songs*: the short tail was merged into the
    final block before clustering, that block no longer matched the group's
    length, and `block_similarity` scores unequal lengths 0. The last verse
    stopped being a verse — excluded from the consensus vote, its onsets never
    pooled into the verse's pattern, and drawn on the rail as different music."""
    tag = [_bar(F), _bar(F)]
    sections, groups = detect(VERSE * 4 + tag, bar_beats=4.0)

    verse_group = next(g for g in groups if g.length_bars == 4)
    assert verse_group.occurrences == [0, 4, 8, 12], "all four passes, including the last"
    assert [s.total_bars for s in sections] == [16, 2]
    assert sections[0].repeats == 4


def test_the_twelve_bar_blues_is_one_section_and_not_three():
    """`CANDIDATE_UNITS` had no 12, so the core blues form was chopped at period
    4 and its I/IV/V phrases scattered across groups that are not sections —
    three choruses came out as `A B B A B B A B B`."""
    blues = ([_bar(C)] * 4 + [_bar(F)] * 2 + [_bar(C)] * 2
             + [_bar(G)] + [_bar(F)] + [_bar(C)] * 2)
    assert period(blues * 3, 4.0) == 12
    sections, groups = detect(blues * 3, bar_beats=4.0)
    assert len(groups) == 1 and groups[0].length_bars == 12
    assert len(sections) == 1 and sections[0].repeats == 3


# --- the encoding ------------------------------------------------------------

def test_identical_repeats_still_collapse_with_repeats():
    """§15's compact encoding survives the rewrite."""
    sections = segment(VERSE * 4)
    assert len(sections) == 1 and sections[0].repeats == 4


def test_a_merely_similar_repeat_keeps_both_passes_explicitly():
    """`repeats` replays one pass in place of the other, so it may only be used
    when they are identical. A near-repeat merges the *boundary* and keeps every
    bar's own harmony — on a song played against a recording, the alternative is
    showing a chord that is not being played."""
    sections = segment(VERSE + MISHEARD)
    assert len(sections) == 1
    assert sections[0].repeats == 1
    assert len(sections[0].bars) == 8
    assert sections[0].bars[7][0].root_pc == A, "pass two kept its own last bar"


def test_sections_tile_the_song_with_no_gaps():
    bars = VERSE * 2 + CHORUS * 2 + VERSE
    sections = segment(bars)
    cursor = 0
    for section in sections:
        assert section.start_bar == cursor
        cursor += section.total_bars
    assert cursor == len(bars)


# --- period and clustering ---------------------------------------------------

def test_the_shortest_progression_that_explains_the_song_wins():
    """A song whose real unit is 4 bars also matches perfectly at 8 and 16;
    picking the longest would bury the repeat structure."""
    assert period(VERSE * 4, 4.0) == 4


def test_clustering_is_deterministic_and_in_song_order():
    """§16.5 wants byte-stable output, so group letters cannot depend on dict
    order — and group A is always the music the song opens with."""
    for _ in range(3):
        sections, groups = detect(SONG, bar_beats=4.0)
        assert [g.label for g in groups] == ["A", "B"]
        assert [s.group for s in sections] == ["A", "B", "A"]


def test_blocks_of_different_lengths_are_never_the_same_group():
    """A 4-bar verse and a 6-bar verse are not two passes of one thing."""
    from app.analysis.form import Block
    four = Block(start_bar=0, bars=tuple(tuple(b) for b in VERSE))
    six = Block(start_bar=4, bars=tuple(tuple(b) for b in VERSE + VERSE[:2]))
    groups, assignment = cluster([four, six], 4.0)
    assert len(groups) == 2 and assignment == [0, 1]


def test_cohesion_reports_how_much_of_a_repeat_a_group_really_is():
    _, groups = detect(VERSE + MISHEARD, bar_beats=4.0)
    assert CLUSTER_SIMILARITY <= groups[0].cohesion < 1.0


# --- naming ------------------------------------------------------------------

def test_without_energy_the_name_follows_the_group_not_the_position():
    """§20's one improvement to the honest fallback: the same music carries the
    same name wherever it returns, so a player reading the rail can see that
    part 1 has come back."""
    sections = segment(SONG)
    assert [s.kind for s in sections] == ["custom"] * 3
    assert [s.name for s in sections] == ["Part 1", "Part 2", "Part 1"]


def test_with_energy_the_loudest_repeated_group_is_the_chorus():
    energy = [0.3] * 8 + [0.9] * 8 + [0.3] * 8
    sections = segment(SONG, energy=energy)
    assert [s.kind for s in sections] == ["verse", "chorus", "verse"]
    assert all(s.name == "" for s in sections), "empty name = use the kind's own label"


def test_a_section_the_song_plays_once_in_the_middle_is_a_bridge():
    """It was `verse` before — confidently wrong about the one section a
    listener would never confuse with anything else."""
    bridge = [_bar(F), _bar(C), _bar(F), _bar(C)]
    bars = VERSE * 2 + CHORUS * 2 + bridge + CHORUS * 2
    energy = [0.3] * 8 + [0.9] * 8 + [0.5] * 4 + [0.9] * 8
    sections = segment(bars, energy=energy)
    assert [s.kind for s in sections] == ["verse", "chorus", "bridge", "chorus"]


def test_a_repeated_group_that_ends_on_the_dominant_before_the_chorus_is_a_prechorus():
    """`harmony.is_dominant_of` was written for exactly this half-cadence cue and
    then called by nothing — its docstring pointed at a `form.label` that does
    not exist. In G, the section leaning on D before the chorus is the one the
    player is being set up for."""
    pre = [_bar(C), _bar(C), _bar(D), _bar(D)]      # … ends on V of G
    bars = VERSE * 2 + pre * 2 + CHORUS * 2 + VERSE * 2 + pre * 2 + CHORUS * 2
    energy = ([0.3] * 8 + [0.5] * 8 + [0.9] * 8) * 2
    sections = segment(bars, energy=energy, tonic_pc=G)
    assert [s.kind for s in sections] == ["verse", "preChorus", "chorus"] * 2


def test_a_lone_verse_ending_on_the_dominant_is_still_a_verse():
    """The guard that makes the cue safe. A verse ending on V to lead into the
    chorus is the most ordinary thing in this repertoire — without requiring a
    *second* repeated non-chorus group, every folk verse becomes a pre-chorus."""
    verse = [_bar(G), _bar(E, MINOR), _bar(C), _bar(D)]     # ends on V of G
    bars = verse * 2 + CHORUS * 2 + verse * 2
    energy = [0.3] * 8 + [0.9] * 8 + [0.3] * 8
    sections = segment(bars, energy=energy, tonic_pc=G)
    assert [s.kind for s in sections] == ["verse", "chorus", "verse"]


def test_without_a_key_the_prechorus_cue_is_simply_unavailable():
    """`tonic_pc` is optional and only ever *adds* a label."""
    pre = [_bar(C), _bar(C), _bar(D), _bar(D)]
    bars = VERSE * 2 + pre * 2 + CHORUS * 2 + VERSE * 2 + pre * 2 + CHORUS * 2
    energy = ([0.3] * 8 + [0.5] * 8 + [0.9] * 8) * 2
    sections = segment(bars, energy=energy)
    assert [s.kind for s in sections] == ["verse", "verse", "chorus"] * 2
