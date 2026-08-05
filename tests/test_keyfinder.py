"""Key detection.

The app uses `tonic`/`mode` for exactly one thing — note spelling — so getting it
wrong doesn't break playback, it just prints "A#m" where every chord sheet in the
world prints "Bbm". These tests are scoped to that: the right *spelling*, and an
honest confidence when the answer is genuinely ambiguous.
"""

from __future__ import annotations

from app.analysis.keyfinder import KEY_MODES, detect_key
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


# --- §20.5: the modes are how the tonic is found -----------------------------

def test_a_mixolydian_song_finds_its_own_tonic():
    """`G F C G` scored against major and minor only comes out **A minor**: every
    chord is diatonic to it, so it ties with C major and beats G major (whose F♯
    the song never plays), and the strongest tonic cue there is — the song starts
    and ends on G — is outvoted by membership. This is why the modes are scored."""
    key = detect_key(track((7, MAJOR), (5, MAJOR), (0, MAJOR), (7, MAJOR)))
    assert key.tonic == "G"
    assert key.scale == "mixolydian"
    assert key.mode == "major", "projected for the container (§12.2)"
    assert key.is_modal


def test_a_dorian_song_finds_its_own_tonic():
    key = detect_key(track((2, MINOR), (7, MAJOR), (2, MINOR)))
    assert (key.tonic, key.mode) == ("D", "minor")
    assert key.scale == "dorian"


def test_only_the_modes_that_occur_as_keys_are_scored():
    """Modes of one collection contain the same notes, so membership cannot tell
    them apart and the tonic gets decided by whatever residue is left. Every mode
    admitted has to earn its place by really occurring as a key — lydian,
    phrygian and locrian do not, and including them cost a correct tonic on the
    real corpus."""
    assert set(KEY_MODES) == {"ionian", "aeolian", "mixolydian", "dorian"}
    for chords in (((0, MAJOR), (5, MAJOR)), ((9, MINOR), (2, MINOR)), ((7, MAJOR),)):
        assert detect_key(track(*chords)).scale in KEY_MODES


def test_the_tonic_bonus_is_not_swamped_by_a_long_final_chord():
    """The endpoint bonus is a claim about *position* — the song comes to rest on
    its tonic — so it is weighted like one typical chord. Uncapped, a song ending
    on a long-held chord gives its final root a bonus several times any evidence
    the rest of the song can offer, and the key becomes a fact about the outro."""
    normal = track((0, MAJOR), (5, MAJOR), (7, MAJOR), (0, MAJOR))
    dragged = normal[:-1] + [GridSpan(start_beat=12, length_beats=400,
                                      root_pc=9, quality=MINOR)]
    assert detect_key(dragged).tonic == detect_key(normal).tonic
