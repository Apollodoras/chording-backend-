"""Key detection.

The app uses `tonic`/`mode` for exactly one thing — note spelling — so getting it
wrong doesn't break playback, it just prints "A#m" where every chord sheet in the
world prints "Bbm". These tests are scoped to that: the right *spelling*, and an
honest confidence when the answer is genuinely ambiguous.
"""

from __future__ import annotations

from app.analysis.keyfinder import detect_key
from app.analysis.types import GridSpan
from app.chords import DOMINANT7, MAJOR, MINOR


def track(*chords) -> list[GridSpan]:
    """(root_pc, quality) pairs → a four-beat-each chord track."""
    return [
        GridSpan(start_beat=i * 4, length_beats=4, root_pc=root, quality=quality)
        for i, (root, quality) in enumerate(chords)
    ]


def test_a_plain_major_progression():
    key = detect_key(track((7, MAJOR), (2, MAJOR), (9, MINOR), (0, MAJOR), (7, MAJOR)))
    assert (key.tonic, key.mode) == ("G", "major")


def test_a_minor_progression():
    key = detect_key(track((9, MINOR), (5, MAJOR), (0, MAJOR), (7, MAJOR), (9, MINOR)))
    assert (key.tonic, key.mode) == ("A", "minor")


def test_a_flat_key_is_spelled_with_flats():
    """The reason this module exists at all."""
    key = detect_key(track((5, MAJOR), (10, MAJOR), (0, MAJOR), (5, MAJOR)))
    assert key.tonic == "F"
    key = detect_key(track((10, MAJOR), (3, MAJOR), (5, DOMINANT7), (10, MAJOR)))
    assert key.tonic == "Bb"


def test_a_sharp_key_is_spelled_with_sharps():
    key = detect_key(track((6, MAJOR), (11, MAJOR), (1, DOMINANT7), (6, MAJOR)))
    assert key.tonic == "F#"


def test_the_tonic_at_both_ends_separates_relative_major_from_minor():
    """The two share every diatonic chord, so without the endpoint cue they score
    identically — and this is the single most common way a key-finder is wrong in
    a way a player notices."""
    major = detect_key(track((0, MAJOR), (5, MAJOR), (7, MAJOR), (9, MINOR), (0, MAJOR)))
    minor = detect_key(track((9, MINOR), (5, MAJOR), (0, MAJOR), (7, MAJOR), (9, MINOR)))
    assert major.mode == "major" and minor.mode == "minor"


def test_confidence_is_low_when_the_answer_is_genuinely_ambiguous():
    """Reported as the margin over the runner-up, not the winner's raw score:
    "this key fits 80% of the song" says little when a neighbour fits 79%."""
    ambiguous = detect_key(track((0, MAJOR), (2, MAJOR), (4, MAJOR), (6, MAJOR)))
    decisive = detect_key(track((7, MAJOR), (2, MAJOR), (9, MINOR), (0, MAJOR), (7, MAJOR)))
    assert ambiguous.confidence < decisive.confidence


def test_an_empty_track_does_not_crash():
    key = detect_key([])
    assert key.mode in ("major", "minor") and key.confidence == 0.0


def test_the_mode_is_never_a_church_mode():
    """§12.2 — the song container knows only major and minor (the jam room knows
    the rest)."""
    for chords in (((7, MAJOR), (0, MAJOR)), ((2, MINOR), (7, MAJOR)), ((4, MINOR),)):
        assert detect_key(track(*chords)).mode in ("major", "minor")
