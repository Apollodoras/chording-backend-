"""The chord grammar and the normalization §12.2 demands.

The grammar half is a port of the app's `ChordSymbol(name:)` and is tested the
same way Mo tests it. The normalization half is this service's own, and it is
where the real risk sits: **an unparseable name doesn't fail loudly, it makes the
chord silently misfire on device** (§12.2). So the assertions below are mostly
"what does a real chord recognizer emit, and does it survive the trip".
"""

from __future__ import annotations

import pytest

from app.chords import (
    AUGMENTED,
    DIMINISHED,
    DIMINISHED7,
    DOMINANT7,
    HALF_DIM7,
    MAJOR,
    MAJOR7,
    MINOR,
    MINOR7,
    SUS2,
    SUS4,
    is_valid_chord,
    normalize,
    normalize_name,
    normalize_quality,
    parse_chord,
    prefers_flats,
    render,
)


# --- the ported grammar -----------------------------------------------------

@pytest.mark.parametrize("name", [
    "C", "Cm", "C7", "Cmaj7", "Cm7", "Cdim", "Cdim7", "Cm7b5", "Caug", "Csus4", "Csus2",
    "F#", "Bb", "G#m7", "Ebmaj7", "A-", "D+", "Bo7",
])
def test_grammar_accepts_every_documented_spelling(name):
    assert is_valid_chord(name)


@pytest.mark.parametrize("name", ["", "H", "bogus", "Cmaj9", "G/B", "C5", "Cadd9", "X"])
def test_grammar_rejects_what_the_app_cannot_read(name):
    """Strict on purpose: a junk token whose first letter happens to be a note
    must not be silently read as a major chord."""
    assert not is_valid_chord(name)


def test_render_is_the_inverse_of_parse():
    for quality in (MAJOR, MINOR, DOMINANT7, MAJOR7, MINOR7, DIMINISHED,
                    DIMINISHED7, HALF_DIM7, AUGMENTED, SUS4, SUS2):
        name = render(7, quality)
        parsed = parse_chord(name)
        assert parsed == (7, quality), f"{name} did not round-trip"


def test_every_rendered_name_parses_under_the_apps_grammar():
    """The guarantee that matters: nothing this service can emit is a name the
    importer would warn about."""
    for pc in range(12):
        for flats in (False, True):
            for quality in (MAJOR, MINOR, DOMINANT7, MAJOR7, MINOR7, DIMINISHED,
                            DIMINISHED7, HALF_DIM7, AUGMENTED, SUS4, SUS2):
                assert is_valid_chord(render(pc, quality, flats=flats))


# --- normalization ----------------------------------------------------------

@pytest.mark.parametrize("label,expected", [
    # Harte — what BTC and Chordino actually speak.
    ("C:maj", (0, MAJOR)),
    ("F#:min7", (6, MINOR7)),
    ("Bb:maj7", (10, MAJOR7)),
    ("A:hdim7", (9, HALF_DIM7)),
    ("G:sus4", (7, SUS4)),
    ("E:dim7", (4, DIMINISHED7)),
    # Symbolic — what a chord sheet or a simpler engine emits.
    ("Am", (9, MINOR)),
    ("Cmaj7", (0, MAJOR7)),
    ("Dm7", (2, MINOR7)),
])
def test_normalize_reads_both_vocabularies(label, expected):
    root, quality, exact = normalize(label)
    assert (root, quality) == expected
    assert exact


@pytest.mark.parametrize("label", ["N", "N.C.", "X", "", "  "])
def test_no_chord_labels_become_none(label):
    """`N` is Harte's "no chord". It is not an error and not a rest — §18 answers
    it with "hold the previous chord", which postprocess does."""
    assert normalize(label) is None


@pytest.mark.parametrize("label,expected", [
    # §12.2's list, verbatim: these are what "a chord recognizer will hand you
    # constantly", and each has a defined landing place inside the grammar.
    ("G/B", (7, MAJOR)),          # slash → root triad
    ("C:maj/3", (0, MAJOR)),      # Harte slash
    ("C9", (0, DOMINANT7)),       # extension → nearest 7th
    ("C11", (0, DOMINANT7)),
    ("C13", (0, DOMINANT7)),
    ("Cmaj9", (0, MAJOR7)),
    ("Cm9", (0, MINOR7)),
    ("Cadd9", (0, MAJOR)),        # add9 keeps the triad
    ("C6", (0, MAJOR)),
    ("Cm6", (0, MINOR)),
    ("C5", (0, MAJOR)),           # power chord → major
    ("C7#5", (0, DOMINANT7)),     # altered dominant → plain dominant
    ("C7b9", (0, DOMINANT7)),
    ("CmMaj7", (0, MINOR)),       # no minor-major slot; its triad is minor
])
def test_normalize_reduces_what_the_grammar_cannot_say(label, expected):
    root, quality, exact = normalize(label)
    assert (root, quality) == expected
    assert not exact, "a reduction must report that information was lost"


def test_normalized_names_always_parse():
    """The end-to-end property: whatever an engine emits, what we write is
    readable by the app."""
    engine_output = ["C:maj", "G/B", "F#:min7", "Bb13", "A:hdim7", "Dsus4", "E5", "Cadd9"]
    for label in engine_output:
        name = normalize_name(label)
        assert name is not None and is_valid_chord(name), label


def test_normalize_rejects_things_that_are_not_chords():
    assert normalize("hello") is None
    assert normalize("123") is None


def test_quality_reduction_reports_exactness():
    assert normalize_quality("maj7") == (MAJOR7, True)
    assert normalize_quality("13") == (DOMINANT7, False)


# --- no simplification ------------------------------------------------------

def test_the_grammar_offers_no_way_to_make_a_chord_easier():
    """The §5.5 difficulty tiers are gone, and this is the guard on them staying
    gone: the chart states what was played, so nothing in this module may take a
    quality and hand back a simpler one. `simplify`, `DIFFICULTIES`, `EASY`,
    `NORMAL` and `HARD` were that API."""
    import app.chords as chords

    for name in ("simplify", "DIFFICULTIES", "EASY", "NORMAL", "HARD"):
        assert not hasattr(chords, name), f"app.chords.{name} is back"


def test_normalization_is_the_only_reduction_and_it_reports_itself():
    """What survives is the app's grammar, which is a ceiling rather than a
    choice — and every quality inside it comes back untouched. A chord that had
    to be reduced to fit says so, which is what `exactRatio` counts."""
    for token, quality in (("maj", MAJOR), ("min", MINOR), ("7", DOMINANT7),
                           ("maj7", MAJOR7), ("min7", MINOR7), ("dim", DIMINISHED),
                           ("dim7", DIMINISHED7), ("hdim7", HALF_DIM7),
                           ("aug", AUGMENTED), ("sus4", SUS4), ("sus2", SUS2)):
        assert normalize_quality(token) == (quality, True), token
    # Cmaj7 stays a major 7. It is not a C, at any level of anything.
    assert normalize("C:maj7") == (0, MAJOR7, True)
    assert normalize("A:min7") == (9, MINOR7, True)


# --- spelling ---------------------------------------------------------------

def test_flat_keys_spell_flats():
    """Not cosmetic: the player reads these names off the campfire bands, and
    "Bb" in an F-major song is right where "A#" is wrong."""
    assert prefers_flats("F", "major")
    assert prefers_flats("Bb", "major")
    assert prefers_flats("D", "minor")
    assert not prefers_flats("G", "major")
    assert not prefers_flats("E", "minor")
    assert render(10, MAJOR, flats=True) == "Bb"
    assert render(10, MAJOR, flats=False) == "A#"
