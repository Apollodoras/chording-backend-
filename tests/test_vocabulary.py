"""§20.8 — the song's own vocabulary, and the seven gates that make it safe.

This layer is the second one in the service that **edits chords the engine
reported**, so the tests are written the way `test_consensus.py` is: mostly about
what it refuses to do, each refusal stated as the musical situation it protects,
and each one named after the real recording that taught it.

Four of them come straight off the corpus, and they are the reason `SNAP_TO` is a
measured table rather than "anything `harmony.is_near_miss` admits":

- **"In My Life"** plays A for seventy beats and A7 for nine, in four brief
  passes, and the engine hedges on every one. Every measure of *amount* calls
  those nine beats noise. They are the song, and what distinguishes them is that
  they keep coming back — `MAX_OCCASIONS`.
- **"Let It Be"** opens on Fmaj7. A reported major 7th is never the plain triad
  when it is wrong, so flattening it can only lose — the quality is not in the
  table at all.
- **"Michelle"** has a real augmented chord, and an augmented triad is one set of
  notes under three names. A rule reasoning about labels cannot tell Caug from
  Eaug, so it must not touch either.
- **"Here Comes The Sun"** has a Bm the engine hedges on in a song full of B. The
  diatonic guard is what stops it becoming B major: Bm belongs to A major and B
  does not.

And the property the whole layer is judged by, exactly as in §20.4: **on perfect
input every rule here is a no-op**, because ground truth arrives at a flat
confidence and no reading is ever believed less than another.
"""

from __future__ import annotations

from app.analysis import harmony, vocabulary
from app.analysis.postprocess import merge
from app.analysis.types import GridSpan
from app.chords import (
    AUGMENTED,
    DOMINANT7,
    MAJOR,
    MAJOR7,
    MINOR,
    MINOR7,
    SUS4,
)

C, D, Eb, E, F, Gb, G, A, Bb, B = 0, 2, 3, 4, 5, 6, 7, 9, 10, 11


def span(start, length, root=C, quality=MAJOR, confidence=0.9, exact=True) -> GridSpan:
    return GridSpan(start_beat=start, length_beats=length, root_pc=root,
                    quality=quality, confidence=confidence, exact=exact)


def timeline(*chords, beats: int = 4, confidence: float = 0.9) -> list[GridSpan]:
    """`(root, quality)` or `(root, quality, confidence)` per bar, laid end to end."""
    out: list[GridSpan] = []
    for index, chord in enumerate(chords):
        root, quality = chord[0], chord[1]
        out.append(span(index * beats, beats, root, quality,
                        confidence=chord[2] if len(chord) > 2 else confidence))
    return out


def names(spans: list[GridSpan]) -> list[tuple[int, str]]:
    return [(s.root_pc, s.quality) for s in spans]


# The verse of "The Silence" — Ebm Db Ab Ebm, in Eb dorian. Ab major rather than
# Ab minor is what makes it dorian, and it is why the diatonic guard has to read
# modes at all: scored against Eb aeolian, the Ab the song leans on is foreign.
SILENCE = ((Eb, MINOR), (1, MAJOR), (8, MAJOR), (Eb, MINOR))


def silence(passes: int = 4, **swaps) -> list[GridSpan]:
    """The verse played `passes` times, with `bar=chord` substitutions by index."""
    bars = [chord for _ in range(passes) for chord in SILENCE]
    for index, chord in swaps.items():
        bars[int(index[3:])] = chord
    return timeline(*bars)


# --- the property the design rests on ----------------------------------------

def test_perfect_input_is_a_no_op():
    """Every gate that consults confidence needs a *gap* to open, and ground truth
    has none — so a correct chart cannot be edited. This is what makes the
    benchmark's truth-as-engines run a regression guard rather than a hope."""
    spans = timeline(*[chord for _ in range(4) for chord in SILENCE], confidence=1.0)
    out, report = vocabulary.consolidate(spans, tonic_pc=Eb, mode="dorian")
    assert not report.touched
    assert names(out) == names(merge(spans))


def test_a_perfect_chart_with_extensions_is_also_left_alone():
    """The same, on the input most likely to tempt the layer: a song that really
    does play both the triad and its seventh."""
    spans = timeline((C, MAJOR), (C, DOMINANT7), (F, MAJOR), (C, MAJOR),
                     (G, MAJOR), (C, MAJOR), (F, MAJOR), (C, MAJOR), confidence=1.0)
    out, report = vocabulary.consolidate(spans)
    assert not report.touched
    assert names(out) == names(merge(spans))


# --- what it exists for ------------------------------------------------------

def test_the_reported_defect():
    """The complaint this module was written for. "The Silence" is Ebm–Db–Ab–Ebm
    throughout; the engine reads one bar of the tonic as `Ebm7` and another as
    `Eb`, hedging on both. Neither is in the song, and neither is a bar any repeat
    group can outvote — the passes disagree two ways, which is what defeats a
    majority. The song itself has the answer twenty bars over."""
    spans = silence(bar4=(Eb, MINOR7, 0.55), bar8=(Eb, MAJOR, 0.61))
    out, report = vocabulary.consolidate(spans, tonic_pc=Eb, mode="dorian")
    assert report.snapped_spans == 2
    assert all(quality == MINOR for root, quality in names(out) if root == Eb)


def test_a_snapped_span_stops_claiming_to_be_exact():
    """`exactRatio` is the one field that can say "`hard` is a fiction on this
    recording", and a span whose quality we chose is not the quality the engine
    reported. Overstating it would make that field quietly optimistic."""
    spans = silence(bar4=(Eb, MINOR7, 0.55))
    out, _ = vocabulary.consolidate(spans, tonic_pc=Eb, mode="dorian")
    assert any(not s.exact for s in out)


def test_the_edited_timeline_stays_contiguous_and_merged():
    """The invariant `postprocess.process` hands over and this stage has to hand
    on: no gaps, no adjacent duplicates. Without the re-merge the chart shows a
    chord change from Ebm to Ebm at the bar line."""
    spans = silence(bar4=(Eb, MINOR7, 0.55))
    out, _ = vocabulary.consolidate(spans, tonic_pc=Eb, mode="dorian")
    assert all(a.end_beat == b.start_beat for a, b in zip(out, out[1:]))
    assert all(names([a]) != names([b]) for a, b in zip(out, out[1:]))


# --- the refusals ------------------------------------------------------------

def test_a_recurring_seventh_is_the_song_not_noise():
    """"In My Life": four brief, doubtful A7s in a song that is otherwise all A.
    Duration says noise, recurrence says arrangement, and recurrence is right."""
    bars = []
    for _ in range(4):
        bars += [(A, MAJOR), (A, DOMINANT7, 0.5), (D, MAJOR), (A, MAJOR)]
    out, report = vocabulary.consolidate(timeline(*bars), tonic_pc=A, mode="ionian")
    assert report.snapped_spans == 0
    assert (A, DOMINANT7) in names(out)


def test_a_seventh_heard_twice_is_flattened():
    """The other side of the same gate, and the case the corpus says pays: two
    doubtful G7s in a song of G is the engine adding a seventh the record does not
    play ("Something", where it did exactly that twice, a beat at a time).

    Built as beats rather than bars, because that is the shape the mistake has: the
    seventh appears for one beat inside a song that holds plain G for bars."""
    spans: list[GridSpan] = []
    beat = 0
    for index in range(6):
        spans.append(span(beat, 4, C, MAJOR))
        beat += 4
        if index in (1, 4):
            spans.append(span(beat, 1, G, DOMINANT7, confidence=0.5))
            spans.append(span(beat + 1, 3, G, MAJOR))
        else:
            spans.append(span(beat, 4, G, MAJOR))
        beat += 4
    out, report = vocabulary.consolidate(spans, tonic_pc=C, mode="ionian")
    assert report.snapped_spans == 2
    assert (G, DOMINANT7) not in names(out)


def test_a_major_seventh_is_never_flattened():
    """"Let It Be" opens on Fmaj7. Measured, a reported major 7th is right a third
    of the time and is *never* the plain triad when it is wrong — so this edit
    cannot win, whatever the rest of the song looks like."""
    bars = [(F, MAJOR7, 0.48)] + [(F, MAJOR)] * 6 + [(C, MAJOR)]
    out, report = vocabulary.consolidate(timeline(*bars), tonic_pc=C, mode="ionian")
    assert report.snapped_spans == 0
    assert names(out)[0] == (F, MAJOR7)


def test_an_augmented_chord_is_left_alone():
    """"Michelle". Caug, Eaug and G#aug are the same three notes, so a rule
    reasoning about names cannot tell which spelling was meant — and 88% of the
    time the engine reporting one has the root wrong anyway, which nothing here
    can fix."""
    bars = [(C, AUGMENTED, 0.5)] + [(C, MAJOR)] * 6 + [(F, MAJOR)]
    out, report = vocabulary.consolidate(timeline(*bars), tonic_pc=C, mode="ionian")
    assert report.snapped_spans == 0
    assert names(out)[0] == (C, AUGMENTED)


def test_the_key_is_never_snapped_away_from():
    """"Here Comes The Sun": a hedged Bm in a song full of B, in A major. The
    counting all points at B; the key says Bm belongs to A major and B does not,
    and the key is right. This gate turned three wrong edits into none."""
    bars = [(B, MINOR, 0.41)] + [(B, MAJOR)] * 6 + [(A, MAJOR)]
    out, report = vocabulary.consolidate(timeline(*bars), tonic_pc=A, mode="ionian")
    assert report.snapped_spans == 0
    assert names(out)[0] == (B, MINOR)


def test_a_root_the_song_plays_two_ways_keeps_both():
    """A minority reading with real weight behind it is vocabulary, not noise —
    `MASS_DOMINANCE` and `MINORITY_SHARE` between them. Three bars of Cm against
    five of C is a song with modal mixture in it, not a song with a mistake."""
    bars = [(C, MINOR, 0.6)] * 3 + [(C, MAJOR)] * 5
    out, report = vocabulary.consolidate(timeline(*bars), tonic_pc=C, mode="ionian")
    assert report.snapped_spans == 0
    assert (C, MINOR) in names(out)


def test_a_confident_minority_survives():
    """The engine was as sure of the Eb as it was of every Ebm around it. That is
    evidence the music changed, and it outranks the counting — the same gate, and
    the same reasoning, as `consensus`'s third."""
    spans = silence(bar4=(Eb, MAJOR, 0.95))
    out, report = vocabulary.consolidate(spans, tonic_pc=Eb, mode="dorian")
    assert report.snapped_spans == 0
    assert (Eb, MAJOR) in names(out)


def test_the_root_is_never_moved():
    """The relative-major confusion — Gb heard where the song plays Ebm — is a
    real and common engine error, and it is deliberately out of scope: both chords
    are usually in the same song's vocabulary (they are in "The Silence"), so mass
    cannot tell which belongs in this bar. Deciding that needs the same bar in
    another pass, which is `consensus`'s evidence, not this module's."""
    spans = silence(bar4=(Gb, MAJOR, 0.4))
    out, report = vocabulary.consolidate(spans, tonic_pc=Eb, mode="dorian")
    assert report.snapped_spans == 0
    assert (Gb, MAJOR) in names(out)


# --- the table's own invariants ----------------------------------------------

def test_every_allowed_move_keeps_the_root_and_stays_a_near_miss():
    """`SNAP_TO` is measured, so it can be extended by measurement — but no
    measurement licenses an edit across harmonic space or onto another root. This
    is the bound the table has to satisfy, asserted rather than assumed."""
    for quality, targets in vocabulary.SNAP_TO.items():
        for target in targets:
            assert harmony.is_near_miss((C, quality), (C, target)), (quality, target)


def test_the_qualities_the_corpus_ruled_out_are_absent():
    """The exclusions are load-bearing, not omissions — each one is a measured
    negative result (see `NEVER_SNAPPED`)."""
    for quality in vocabulary.NEVER_SNAPPED:
        assert quality not in vocabulary.SNAP_TO


# --- islands -----------------------------------------------------------------

def test_a_hole_in_a_held_chord_is_filled():
    """Two bars of Ebm with one doubtful beat of Eb in the middle is one chord,
    and the hole is what the player sees as a chord change that isn't there.
    §5.4's `drop_short` cannot reach it: a beat is not shorter than a beat."""
    spans = [span(0, 4, Eb, MINOR), span(4, 1, Eb, MAJOR, confidence=0.45),
             span(5, 7, Eb, MINOR)]
    out, absorbed = vocabulary.absorb_islands(spans, bar_beats=4)
    assert absorbed == 1
    assert names(out) == [(Eb, MINOR)], "the whole stretch is one held chord"


def test_a_passing_dominant_is_not_an_island():
    """C | G | C is a beat of the dominant, which is ordinary music. A fifth is not
    a mishearing, so the harmonic test is what keeps this rule from deleting it —
    and it is the reason the span floor could not simply be raised instead."""
    spans = [span(0, 4, C), span(4, 1, G, confidence=0.4), span(5, 7, C)]
    out, absorbed = vocabulary.absorb_islands(spans, bar_beats=4)
    assert absorbed == 0
    assert names(out) == [(C, MAJOR), (G, MAJOR), (C, MAJOR)]


def test_an_island_must_share_the_flanks_root():
    """The corpus's correction to the first version of this rule. Fm | Caug | Fm
    looks like a hole surrounded by one chord, and the augmented chord in the
    middle of it was right."""
    spans = [span(0, 4, F, MINOR), span(4, 1, C, AUGMENTED, confidence=0.5),
             span(5, 7, F, MINOR)]
    out, absorbed = vocabulary.absorb_islands(spans, bar_beats=4)
    assert absorbed == 0
    assert (C, AUGMENTED) in names(out)


def test_a_confident_island_is_left_where_it_is():
    spans = [span(0, 4, Eb, MINOR), span(4, 1, Eb, MAJOR, confidence=0.95),
             span(5, 7, Eb, MINOR)]
    _, absorbed = vocabulary.absorb_islands(spans, bar_beats=4)
    assert absorbed == 0


def test_a_chord_as_long_as_its_neighbours_is_not_an_island():
    """"Brief" is relative to what surrounds it: a bar of Ebm7 between two bars of
    Ebm is a chord in its own right, and if it is wrong it is `snap`'s business
    rather than this rule's."""
    spans = [span(0, 4, Eb, MINOR), span(4, 4, Eb, MINOR7, confidence=0.4),
             span(8, 4, Eb, MINOR)]
    _, absorbed = vocabulary.absorb_islands(spans, bar_beats=4)
    assert absorbed == 0


def test_a_sus_between_two_readings_of_its_own_chord_is_a_hole():
    """The shape §5.4 leaves behind at a boundary: the suspension the engine hears
    while the strum is still ringing."""
    spans = [span(0, 4, D, MINOR), span(4, 1, D, SUS4, confidence=0.35),
             span(5, 7, D, MINOR)]
    out, absorbed = vocabulary.absorb_islands(spans, bar_beats=4)
    assert absorbed == 1
    assert names(out) == [(D, MINOR)]


# --- the profile -------------------------------------------------------------

def test_evidence_is_duration_times_belief_not_airtime():
    """Eight beats at 0.3 and three at 0.9 are close to the same amount of
    evidence; airtime alone would call the first nearly three times the second."""
    profile = vocabulary.profile([span(0, 8, C, MAJOR, confidence=0.3),
                                  span(8, 3, C, MINOR, confidence=0.9)])
    assert profile[C][MAJOR].mass == 2.4
    assert profile[C][MINOR].mass == 2.7


def test_the_profile_counts_occasions_as_well_as_beats():
    """`MAX_OCCASIONS` reads this, and it is the number that separates "In My
    Life"'s A7 from "Something"'s."""
    profile = vocabulary.profile([span(0, 2, A, DOMINANT7), span(8, 2, A, DOMINANT7),
                                  span(16, 2, A, DOMINANT7), span(24, 2, A, DOMINANT7)])
    assert profile[A][DOMINANT7].spans == 4


def test_an_empty_track_profiles_to_nothing():
    assert vocabulary.profile([]) == {}
    assert vocabulary.consolidate([])[0] == []
