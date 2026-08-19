"""Chord-name grammar — a line-for-line port of the app's ``ChordSymbol(name:)``
(MIDI_Tab_Game/Models/ChordSymbol.swift).

This is the *reference* parser for the lint: a chord name the app can't read is
flagged as an import warning on-device, so the backend must reject exactly the
same names. Parsing is STRICT — a junk token whose first letter happens to be a
note ("bogus") is rejected, never silently read as a major chord.

The first half (through ``spell``) is a verbatim port shared with the Mo
backend. **The second half is this service's own**, and it exists because of the
one place the two backends genuinely differ: Mo *authors* chords inside the
grammar, whereas a chord recognizer hands us Harte labels, slash chords and
extensions by the hundred (handoff §12.2 — "a chord recognizer will hand you
these constantly ... normalize before emitting"). So:

``normalize`` maps anything an engine emits into the app's closed grammar, and
that grammar is the **only** reduction this file performs. There used to be a
second one — ``simplify``, the §5.5 easy/normal/hard tiers — and it is gone: the
chart states what was played, and nothing downstream is allowed to make the
harmony easier than the recording. (§12.2 had already cut those tiers down to
three shapes inside the app's grammar, which is a fair description of how little
they were ever worth.)
"""

from __future__ import annotations

import re

# Letter → pitch class, matching `pitchClass(forNoteName:)`.
_LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

_SHARPS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLATS = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# Suffix → quality, verbatim from ChordSymbol.qualitySuffixes. Keys are matched
# against the LOWERCASED remainder of the name (so "MAJ7" parses; the trade-off
# — "M7" reads as minor 7 — matches the app).
QUALITY_SUFFIXES = {
    "": "major", "maj": "major", "major": "major",
    "m": "minor", "min": "minor", "mi": "minor", "-": "minor",
    "maj7": "major7", "ma7": "major7", "major7": "major7", "δ": "major7",
    "7": "dominant7", "dom7": "dominant7",
    "m7": "minor7", "min7": "minor7", "mi7": "minor7", "-7": "minor7",
    "dim": "diminished", "°": "diminished", "o": "diminished",
    "dim7": "diminished7", "°7": "diminished7", "o7": "diminished7",
    "m7b5": "halfDiminished7", "ø": "halfDiminished7", "halfdim": "halfDiminished7",
    "aug": "augmented", "+": "augmented",
    "sus": "sus4", "sus4": "sus4", "sus2": "sus2",
}

_ACCIDENTALS = {"#": 1, "♯": 1, "b": -1, "♭": -1}


def pitch_class(note_name: str) -> int | None:
    """``pitchClass(forNoteName:)`` — "C", "F#", "Bb", "C##" → 0–11, junk → None."""
    trimmed = note_name.strip()
    if not trimmed:
        return None
    pc = _LETTER_PC.get(trimmed[0].upper())
    if pc is None:
        return None
    for ch in trimmed[1:]:
        delta = _ACCIDENTALS.get(ch)
        if delta is None:
            return None
        pc += delta
    return pc % 12


def parse_chord(name: str) -> tuple[int, str] | None:
    """``ChordSymbol(name:)`` — returns ``(root_pitch_class, quality)`` or None."""
    trimmed = name.strip()
    if not trimmed or trimmed[0].upper() not in _LETTER_PC:
        return None
    # Root = the letter plus any immediately-following accidentals.
    i = 1
    while i < len(trimmed) and trimmed[i] in _ACCIDENTALS:
        i += 1
    root = pitch_class(trimmed[:i])
    if root is None:
        return None
    quality = QUALITY_SUFFIXES.get(trimmed[i:].lower())
    if quality is None:
        return None
    return root, quality


def is_valid_chord(name: str) -> bool:
    return parse_chord(name) is not None


def spell(pc: int, flats: bool = False) -> str:
    """Pitch class → note name (the app's NoteSpelling tables)."""
    return (_FLATS if flats else _SHARPS)[pc % 12]


# ===========================================================================
# Below here is analysis-side only — Mo has no equivalent.
# ===========================================================================

# The eleven qualities the app can actually sound. Everything a recognizer emits
# is reduced to one of these; there is no twelfth option, and inventing one means
# a silently misfiring chord on device (§12.2).
MAJOR = "major"
MINOR = "minor"
DOMINANT7 = "dominant7"
MAJOR7 = "major7"
MINOR7 = "minor7"
DIMINISHED = "diminished"
DIMINISHED7 = "diminished7"
HALF_DIM7 = "halfDiminished7"
AUGMENTED = "augmented"
SUS4 = "sus4"
SUS2 = "sus2"

# Quality → the canonical suffix to emit. Several spellings parse (see
# QUALITY_SUFFIXES) but output uses exactly one, so two runs of the pipeline on
# the same audio produce byte-identical songs.
CANONICAL_SUFFIX = {
    MAJOR: "", MINOR: "m",
    DOMINANT7: "7", MAJOR7: "maj7", MINOR7: "m7",
    DIMINISHED: "dim", DIMINISHED7: "dim7", HALF_DIM7: "m7b5",
    AUGMENTED: "aug", SUS4: "sus4", SUS2: "sus2",
}

# What a recognizer emits for "no chord here" — Harte's `N` (and `X` for
# "unknown"). Both mean *no harmony to show*, which §18 answers with "hold the
# previous chord"; that decision lives in postprocess, not here.
NO_CHORD_LABELS = {"n", "n.c.", "nc", "x", "none", "-", ""}


def render(root_pc: int, quality: str, *, flats: bool = False) -> str:
    """(pitch class, quality) → a name the app parses. The inverse of
    ``parse_chord``, and by construction always inside the grammar."""
    return spell(root_pc, flats) + CANONICAL_SUFFIX[quality]


def prefers_flats(tonic: str, mode: str) -> bool:
    """Whether to spell accidentals as flats in this key.

    Not cosmetic: the player reads these names off the campfire bands, and "Bb"
    in an F-major song is right where "A#" is wrong. The codebase's own
    convention (MO_BACKEND_HANDOFF.md §5) is that minor keys spell their flats
    explicitly, so the minor list is the wider one.
    """
    if "b" in tonic[1:] or "♭" in tonic:
        return True
    pc = pitch_class(tonic)
    if pc is None:
        return False
    return prefers_flats_for(pc, mode)


# The key-signature table itself, by pitch class, so a caller holding a pitch
# class rather than a spelled name does not have to keep a second copy of it.
# `analysis/keyfinder.py` had that second copy — the same twelve numbers, written
# out again — which is the shape of thing that stays right until one of them is
# edited.
_FLAT_MINOR_TONICS = frozenset({2, 7, 0, 5, 10, 3})   # d, g, c, f, bb, eb
_FLAT_MAJOR_TONICS = frozenset({5, 10, 3, 8, 1})      # F, Bb, Eb, Ab, Db


def prefers_flats_for(tonic_pc: int, mode: str) -> bool:
    """`prefers_flats`, for a tonic that is still a pitch class."""
    if mode == "minor":
        return tonic_pc % 12 in _FLAT_MINOR_TONICS
    return tonic_pc % 12 in _FLAT_MAJOR_TONICS


# --- Normalization ----------------------------------------------------------

# Alterations and colour tones that carry no weight in a grammar with no slot
# for them. Stripped before the core quality is read, so "7b9" reduces via "7".
_ALTERATION = re.compile(
    r"\((?:[^)]*)\)"                      # C7(b9)
    r"|\[(?:[^\]]*)\]"
    r"|(?:[#b♯♭](?:5|9|11|13))"           # b5 #5 b9 #9 #11 b13
    r"|(?:no[35])"                        # no3 no5
    r"|alt|omit\d*",
    re.IGNORECASE,
)

# The exact spellings worth matching before the reduction rules run — either
# because reduction would get them wrong (`hdim7`) or because they are the
# overwhelmingly common case and an exact hit is cheaper to reason about.
_EXACT_QUALITY = {
    "": MAJOR, "maj": MAJOR, "major": MAJOR, "M": MAJOR,
    "min": MINOR, "m": MINOR, "mi": MINOR, "-": MINOR, "minor": MINOR,
    "7": DOMINANT7, "dom7": DOMINANT7, "dom": DOMINANT7,
    "maj7": MAJOR7, "ma7": MAJOR7, "major7": MAJOR7, "Δ": MAJOR7, "δ": MAJOR7,
    "min7": MINOR7, "m7": MINOR7, "mi7": MINOR7, "-7": MINOR7, "minor7": MINOR7,
    "dim": DIMINISHED, "o": DIMINISHED, "°": DIMINISHED,
    "dim7": DIMINISHED7, "o7": DIMINISHED7, "°7": DIMINISHED7,
    "hdim7": HALF_DIM7, "hdim": HALF_DIM7, "m7b5": HALF_DIM7,
    "ø": HALF_DIM7, "ø7": HALF_DIM7, "halfdim": HALF_DIM7, "min7b5": HALF_DIM7,
    "aug": AUGMENTED, "+": AUGMENTED, "aug7": AUGMENTED,
    "sus": SUS4, "sus4": SUS4, "sus2": SUS2,
}


def normalize_quality(text: str) -> tuple[str, bool]:
    """A recognizer's quality token → (grammar quality, was-it-exact).

    ``exact`` is False whenever information was thrown away (an extension
    flattened, an alteration dropped, a token not understood at all). The caller
    keeps the count: a track where most chords needed reducing is a track whose
    chart is a fiction, and that is worth knowing before shipping it.
    """
    raw = text.strip()
    if raw in _EXACT_QUALITY:
        return _EXACT_QUALITY[raw], True
    lowered = raw.lower()
    if lowered in _EXACT_QUALITY:
        return _EXACT_QUALITY[lowered], True

    stripped = _ALTERATION.sub("", lowered).strip()
    if stripped in _EXACT_QUALITY:
        # Only alterations were dropped ("7b9" → "7") — the core survived, but
        # the chord we emit is not the chord that was heard.
        return _EXACT_QUALITY[stripped], False

    # Reduction, most specific first. Order matters: "sus" before the minor test
    # (there is no "m" in sus), "maj" before the bare-number test.
    if "sus2" in stripped:
        return SUS2, False
    if "sus" in stripped:
        return SUS4, False
    if stripped.startswith(("hdim", "ø")) or "m7b5" in stripped:
        return HALF_DIM7, False
    if stripped.startswith(("dim", "o", "°")):
        return DIMINISHED7 if "7" in stripped else DIMINISHED, False
    if stripped.startswith(("aug", "+")):
        return AUGMENTED, False

    # `add9`/`add11` name a colour tone WITHOUT the seventh below it, so the
    # number must not be read as implying one — Cadd9 is a major triad, and
    # emitting C7 for it would put a note in the chord that isn't in the record.
    extension = None if "add" in stripped else _highest_extension(stripped)
    if _is_minorish(stripped):
        # mMaj7 included: minor-major 7 has no slot, and its triad is minor.
        return (MINOR7 if extension in {7, 9, 11, 13} and "maj" not in stripped else MINOR), False
    if stripped.startswith(("maj", "ma")):
        return (MAJOR7 if extension in {7, 9, 11, 13} else MAJOR), False
    if extension in {7, 9, 11, 13}:
        return DOMINANT7, False
    # 5 (power chord), 6, add9, or something unrecognized: the root is the
    # load-bearing fact and major is the honest default (§12.2 maps `5` → major).
    return MAJOR, False


def _is_minorish(text: str) -> bool:
    """Whether the token opens with a minor marker. `-` counts; `maj`/`ma` do not
    (they start with "m" but are major), and neither does `m7b5`, handled above."""
    if text.startswith(("min", "mi", "-")) and not text.startswith("mi7b5"):
        return True
    return text.startswith("m") and not text.startswith(("maj", "ma"))


def _highest_extension(text: str) -> int | None:
    """The largest chord-tone number in the token (``13`` in "13#11"), or None."""
    numbers = [int(n) for n in re.findall(r"\d+", text)]
    tones = [n for n in numbers if n in {5, 6, 7, 9, 11, 13}]
    return max(tones) if tones else None


def normalize(label: str) -> tuple[int, str, bool] | None:
    """Any chord label an engine produces → ``(root_pc, quality, exact)``, or
    **None for no-chord**.

    Handles the three shapes that actually turn up:

    - Harte, which BTC/Chordino speak natively — ``C:maj``, ``F#:min7``,
      ``G:maj/3``, ``N``, ``X``;
    - plain symbols — ``Cm7``, ``Bbmaj7``, ``Dsus4``;
    - slash chords in either — ``G/B``, ``C:maj/5``. **The bass note is
      discarded** (§12.2: `G/B` → `G`). This is a real loss and the right one:
      the app voices a chord from its root, so a bass it cannot play is better
      dropped than misread.

    The 2026-08-18 audit suggested keeping inversions (F34), on the grounds
    that BTC's large vocabulary emits them and the bass note is part of what was
    played. The obstacle is not this function: the container has no
    field for a bass note, and the app builds its voicing from the root, so a
    payload saying `C/E` would either fail to parse or sound a plain C under a
    label promising otherwise. Adding it is a container change first (§12.2) and
    a parser change second, which is why the loss is recorded here — in
    ``exactRatio``, which counts every discarded bass — rather than papered over.
    """
    text = label.strip()
    if text.lower() in NO_CHORD_LABELS:
        return None

    head = text.split("/", 1)[0].strip()   # drop the slash bass
    if ":" in head:
        root_text, _, quality_text = head.partition(":")
    else:
        # Symbolic: the root is the letter plus any accidentals glued to it.
        index = 1
        while index < len(head) and head[index] in _ACCIDENTALS:
            index += 1
        root_text, quality_text = head[:index], head[index:]

    if not root_text or root_text[0].upper() not in _LETTER_PC:
        return None
    root = pitch_class(root_text)
    if root is None:
        return None
    quality, exact = normalize_quality(quality_text)
    # A discarded bass note is lost information too, even when the quality was
    # read exactly — say so, so the counter upstream isn't optimistic.
    return root, quality, exact and "/" not in text


def normalize_name(label: str, *, flats: bool = False) -> str | None:
    """``normalize`` straight to an app-parseable name. None means no-chord."""
    parsed = normalize(label)
    if parsed is None:
        return None
    root, quality, _ = parsed
    return render(root, quality, flats=flats)
