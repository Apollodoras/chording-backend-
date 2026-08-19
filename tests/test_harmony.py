"""§20.1 — the primitives the smart layer votes with.

The one that carries weight is `is_near_miss`. It is the only thing standing
between "the engine misheard a repeat" and "the music changed", and
`consensus.py` is allowed to overwrite the first and forbidden from touching the
second. Get this table wrong in the permissive direction and the service starts
quietly rewriting real chord changes into whatever the other verses did.
"""

from __future__ import annotations

from app.analysis import harmony
from app.chords import (
    DOMINANT7,
    MAJOR,
    MAJOR7,
    MINOR,
    MINOR7,
    SUS4,
)

C, D, E, F, G, A, B = 0, 2, 4, 5, 7, 9, 11


# --- what a recognizer can plausibly confuse ---------------------------------

def test_the_classic_confusions_are_near_misses():
    """Every pair here shares two of three notes, which is exactly why a
    recognizer slides between them: in a dense mix the third really is
    ambiguous. These are the disagreements worth voting on."""
    for label, a, b in [
        ("relative minor", (C, MAJOR), (A, MINOR)),
        ("mediant", (C, MAJOR), (E, MINOR)),
        ("parallel minor", (C, MAJOR), (C, MINOR)),
        ("suspension", (C, MAJOR), (C, SUS4)),
        ("added seventh", (C, MAJOR), (C, MAJOR7)),
        ("dominant seventh", (G, MAJOR), (G, DOMINANT7)),
        ("subdominant/relative", (A, MINOR), (F, MAJOR)),
    ]:
        assert harmony.is_near_miss(a, b), label


def test_a_different_chord_is_not_a_near_miss():
    """I against IV, I against V, and anything a tritone away. Consensus must
    never vote across these — a chord this far off is the music changing, and
    overwriting it shows the player something that is not being played."""
    for label, a, b in [
        ("tonic vs subdominant", (C, MAJOR), (F, MAJOR)),
        ("tonic vs dominant", (C, MAJOR), (G, MAJOR)),
        ("tritone", (C, MAJOR), (6, MAJOR)),
        ("whole step", (C, MAJOR), (D, MAJOR)),
    ]:
        assert not harmony.is_near_miss(a, b), label


def test_similarity_is_symmetric_and_bounded():
    for a in [(C, MAJOR), (A, MINOR), (G, DOMINANT7)]:
        for b in [(C, MAJOR), (F, MAJOR), (E, MINOR7)]:
            assert harmony.similarity(a, b) == harmony.similarity(b, a)
            assert 0.0 <= harmony.similarity(a, b) <= 1.0
    assert harmony.similarity((C, MAJOR), (C, MAJOR)) == 1.0


# --- scales and keys ---------------------------------------------------------

def test_a_modes_scale_is_rooted_at_its_own_tonic():
    assert harmony.scale_pcs(C, "ionian") == frozenset({0, 2, 4, 5, 7, 9, 11})
    assert harmony.scale_pcs(G, "mixolydian") == frozenset({7, 9, 11, 0, 2, 4, 5})


def test_modes_of_one_collection_share_their_notes():
    """The fact that makes key detection hard, and the reason `keyfinder` scores
    only the four modes that really occur as keys: membership cannot tell these
    apart at all, so every extra mode is another way to pick a wrong tonic."""
    assert harmony.scale_pcs(C, "ionian") == harmony.scale_pcs(G, "mixolydian")
    assert harmony.scale_pcs(C, "ionian") == harmony.scale_pcs(D, "dorian")


def test_minor_carries_its_harmonic_seventh():
    """The V and vii° of a minor key are everywhere in this repertoire; a scorer
    that called them foreign would fail every minor song."""
    assert 8 in harmony.scale_pcs(A, "aeolian")     # G#, the leading tone


def test_every_mode_projects_to_a_mode_the_container_knows():
    for mode in harmony.SCALES:
        assert harmony.MODE_PROJECTION[mode] in ("major", "minor")


def test_diatonic_fit_is_graded_not_boolean():
    """Borrowed chords and secondary dominants are normal music. A rule that
    called them errors would do more damage than the errors it caught."""
    assert harmony.diatonic_fit(G, MAJOR, C, "ionian") == 1.0
    assert harmony.diatonic_fit(6, MAJOR, C, "ionian") == 0.0
    partial = harmony.diatonic_fit(D, DOMINANT7, C, "ionian")     # V/V
    assert 0.0 < partial < 1.0


# --- naming ------------------------------------------------------------------

def test_roman_numerals_read_the_way_a_chart_prints_them():
    assert harmony.roman(C, MAJOR, C, "ionian") == "I"
    assert harmony.roman(E, MINOR, C, "ionian") == "iii"
    assert harmony.roman(10, MAJOR, C, "ionian") == "bVII"
    assert harmony.roman(F, MAJOR, A, "aeolian") == "bVI"


def test_every_interval_has_a_name():
    """Total by construction — the mode-relative alternative had to invent a
    spelling for the harmonic-minor leading tone and indexed off the end."""
    for interval in range(12):
        for mode in harmony.SCALES:
            assert harmony.roman(interval, MAJOR, 0, mode)


def test_the_dominant_is_recognised_for_the_half_cadence():
    assert harmony.is_dominant_of(G, MAJOR, C)
    assert harmony.is_dominant_of(G, DOMINANT7, C)
    assert not harmony.is_dominant_of(F, MAJOR, C)
