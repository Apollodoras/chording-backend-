"""§20.10 — the chart against its own key.

The third layer in the service that **edits chords the engine reported**, so
these tests are written the way `test_consensus.py` and `test_vocabulary.py` are:
mostly about what it refuses to do, each refusal named after the recording that
taught it.

The layer exists for one symptom the owner reported — "major and minor chords in
the same key" — and the reason neither of the other two could answer it is
worth restating, because it is what these tests are guarding. §20.4 needs two
passes of one section to *disagree*; §20.8 needs a landslide of mass. A
systematic mishearing — BTC reading `C#m` as `C#` every time that passage comes
round — offers neither. The key is the only evidence in the building that did not
come from counting the engine's own output.
"""

from __future__ import annotations

from app.analysis import keyaudit
from app.analysis.types import GridSpan
from app.chords import DOMINANT7, MAJOR, MAJOR7, MINOR

C, Db, D, Eb, E, F, Gb, G, Ab, A, Bb, B = range(12)


def span(start, length, root, quality=MAJOR, confidence=0.9) -> GridSpan:
    return GridSpan(start_beat=start, length_beats=length, root_pc=root,
                    quality=quality, confidence=confidence, exact=True)


def timeline(*chords, beats: int = 4) -> list[GridSpan]:
    """`(root, quality)` or `(root, quality, confidence)` per bar."""
    return [span(i * beats, beats, c[0], c[1], c[2] if len(c) > 2 else 0.9)
            for i, c in enumerate(chords)]


def names(spans):
    return [(s.root_pc, s.quality) for s in spans]


# --- where a chord stands in the key ------------------------------------------

def test_the_three_standings():
    """Diatonic, borrowed and foreign are three answers, not two. A rule with only
    the first and the last calls half this repertoire an error — bVII and iv in a
    major key are in most of the folk canon."""
    assert keyaudit.standing(C, MAJOR, C, "ionian") == keyaudit.DIATONIC
    assert keyaudit.standing(A, MINOR, C, "ionian") == keyaudit.DIATONIC
    assert keyaudit.standing(Bb, MAJOR, C, "ionian") == keyaudit.BORROWED   # bVII
    assert keyaudit.standing(F, MINOR, C, "ionian") == keyaudit.BORROWED    # iv
    assert keyaudit.standing(C, DOMINANT7, C, "ionian") == keyaudit.BORROWED  # I7
    assert keyaudit.standing(A, MAJOR, C, "ionian") == keyaudit.FOREIGN     # VI as a triad


def test_a_secondary_dominant_is_borrowed_and_a_bare_major_is_not():
    """The distinction that decides the whole layer. `A7` in C is V/ii and reads
    as a secondary dominant; a bare `A` in C is the same root with no seventh to
    explain it, and in this repertoire it is far more often the vi misheard.

    Widening the table to admit the bare triad — which is what it said first —
    protects exactly the chords this layer was reported for: Don't Stop
    Believin's `C#` and `G#` against its `C#m` and `G#m`."""
    assert keyaudit.standing(A, DOMINANT7, C, "ionian") == keyaudit.BORROWED
    assert keyaudit.standing(A, MAJOR, C, "ionian") == keyaudit.FOREIGN


# --- what it repairs ----------------------------------------------------------

def test_a_minority_foreign_reading_is_pulled_onto_the_keys_answer():
    """Don't Stop Believin', in E: the verse is `C#m` and the engine hears `C#`
    in one passage. Same root, colour only, out of key, doubtful, and a small
    share of what that root ever sounds."""
    spans = timeline((E, MAJOR), (Db, MINOR), (E, MAJOR), (Db, MINOR),
                     (E, MAJOR), (Db, MINOR), (E, MAJOR), (Db, MAJOR, 0.4))
    out, report = keyaudit.resolve(spans, tonic_pc=E, mode="ionian")
    assert report.resolved_spans == 1
    assert (Db, MAJOR) not in names(out)
    assert report.conflicts == ("C#(VI) vs C#m(vi)",)


def test_the_conflict_names_the_degrees():
    """A chord-symbol pair alone does not say which of the two the key expects,
    and that is the whole content of the finding."""
    spans = timeline((E, MAJOR), (Db, MINOR), (E, MAJOR), (Db, MINOR),
                     (E, MAJOR), (Db, MINOR), (E, MAJOR), (Db, MAJOR, 0.4))
    _, report = keyaudit.resolve(spans, tonic_pc=E, mode="ionian")
    assert report.conflicts == ("C#(VI) vs C#m(vi)",)


# --- what it refuses ----------------------------------------------------------

def test_a_foreign_reading_the_song_leans_on_is_reported_and_kept():
    """**Creep.** It is in G and plays G B C Cm; the key finder returns something
    else (C major and E minor both score it identically on every term it has), and
    from there the real `Cm` is the foreign one. `Cm` is a third of everything
    that root sounds, so the song is telling us the key is wrong rather than the
    chord — and the audit's job there is to say so, not to delete the chord the
    song is known for."""
    spans = timeline((G, MAJOR), (B, MAJOR), (C, MAJOR), (C, MINOR, 0.6),
                     (G, MAJOR), (B, MAJOR), (C, MAJOR), (C, MINOR, 0.6))
    out, report = keyaudit.resolve(spans, tonic_pc=E, mode="aeolian")
    assert report.resolved_spans == 0
    assert (C, MINOR) in names(out), "the chord survives"
    assert report.conflicts, "and the conflict is reported"


def test_a_borrowed_seventh_is_never_touched():
    """**"In My Life."** It is in A, plays A for seventy beats and A7 for nine, in
    four brief doubtful passes — and those nine beats are the song. I7 leaning
    into IV is in the borrowings table for this one case."""
    spans = timeline((A, MAJOR), (A, DOMINANT7, 0.45), (D, MAJOR), (A, MAJOR),
                     (A, MAJOR), (A, DOMINANT7, 0.5), (D, MAJOR), (A, MAJOR))
    out, report = keyaudit.resolve(spans, tonic_pc=A, mode="ionian")
    assert report.resolved_spans == 0
    assert (A, DOMINANT7) in names(out)
    assert report.conflicts == ()


def test_a_believed_reading_is_left_alone():
    """The gate every correcting layer here carries, and the reason this one is
    **a provable no-op on perfect input**: ground truth arrives at a flat
    confidence and no reading is ever believed less than another."""
    spans = timeline((E, MAJOR), (Db, MINOR), (E, MAJOR), (Db, MINOR),
                     (E, MAJOR), (Db, MINOR), (E, MAJOR), (Db, MAJOR, 1.0),
                     beats=4)
    spans = [GridSpan(start_beat=s.start_beat, length_beats=s.length_beats,
                      root_pc=s.root_pc, quality=s.quality, confidence=1.0,
                      exact=True) for s in spans]
    out, report = keyaudit.resolve(spans, tonic_pc=E, mode="ionian")
    assert report.resolved_spans == 0
    assert names(out) == names(spans)


def test_a_quality_the_corpus_says_never_to_flatten_is_never_flattened():
    """`vocabulary.NEVER_SNAPPED` is a measured table and it binds here too — a
    reported major 7th is either a real one or a wrong root, and neither is helped
    by moving its colour."""
    spans = timeline((C, MAJOR), (F, MAJOR), (C, MAJOR), (Ab, MAJOR7, 0.3),
                     (C, MAJOR), (F, MAJOR), (C, MAJOR), (G, MAJOR))
    out, report = keyaudit.resolve(spans, tonic_pc=C, mode="ionian")
    assert report.resolved_spans == 0


def test_a_root_the_song_reads_one_way_is_not_a_conflict():
    """A foreign chord on its own is not an error — that is what `BORROWED` and
    the graded `diatonic_fit` are for. A conflict needs the same root read *two*
    ways, one at home and one not."""
    spans = timeline((C, MAJOR), (F, MAJOR), (Ab, MAJOR), (G, MAJOR))
    out, report = keyaudit.resolve(spans, tonic_pc=C, mode="ionian")
    assert report.resolved_spans == 0
    assert report.conflicts == ()
    assert names(out) == names(spans)
