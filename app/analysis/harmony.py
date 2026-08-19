"""§20.1 — the theory primitives the smart layer votes with.

Everything above this module works in chord *symbols*: a `(root_pc, quality)`
pair either equals another one or it doesn't. That binary is what made structure
detection brittle — one misheard chord and a verse stops being the verse it is a
repeat of (see `form.py`) — and it is what makes a consensus vote dangerous,
because "these two disagree" says nothing about *how*.

So this module answers the one question the rest of §20 keeps asking:

    when two readings of the same bar disagree, is that the engine mishearing,
    or is that the music changing?

**Harmonic distance is the discriminator.** C and Am share two of their three
notes; so do C and Em, C and Cm, C and Csus4. Those are exactly the confusions a
chord recognizer makes, because they are the confusions the *signal* supports —
a mix where the third is buried genuinely is ambiguous between C and Am. C and
F# share nothing, and no recognizer arrives at one from the other by accident.
So a disagreement between near neighbours is evidence of noise, and a
disagreement between distant chords is evidence of music. `consensus.py` is
allowed to overwrite the first and forbidden from touching the second.

That rule is doing real safety work. `structure._merge_similar` already
documents where the alternative leads: showing the player a chord that is not
being played, invisible to `lint` and `lint_sync` alike because the chart is
perfectly self-consistent and wrong.

The scale tables are here for the other two jobs — spelling a key that is modal
rather than major/minor (`keyfinder.py`), and breaking a tied vote toward the
chord that belongs to the key (`consensus.py`). **Modes are internal only**: the
container knows `major` and `minor` and nothing else (§12.2), so the projection
back down happens at the wire and is written out in `keyfinder.project`.
"""

from __future__ import annotations

from functools import lru_cache

from ..chords import (
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
)

# Semitones above the root, for each of the eleven qualities the app can sound.
# The whole point of spelling these out is that harmonic distance can then be
# measured on *notes* rather than on labels, which is the only level at which
# "C and Am are nearly the same sound" is expressible.
INTERVALS: dict[str, tuple[int, ...]] = {
    MAJOR: (0, 4, 7),
    MINOR: (0, 3, 7),
    DOMINANT7: (0, 4, 7, 10),
    MAJOR7: (0, 4, 7, 11),
    MINOR7: (0, 3, 7, 10),
    DIMINISHED: (0, 3, 6),
    DIMINISHED7: (0, 3, 6, 9),
    HALF_DIM7: (0, 3, 6, 10),
    AUGMENTED: (0, 4, 8),
    SUS4: (0, 5, 7),
    SUS2: (0, 2, 7),
}

# Scale degrees for the seven modes, plus the two minor variants that actually
# appear in this material. Ionian/Aeolian are the container's major/minor; the
# other five exist because a modal song scored against only those two picks its
# *relative* — a G-F-C song read as C major rather than G mixolydian — and then
# spells and analyzes everything a fifth away from where the player is.
IONIAN = (0, 2, 4, 5, 7, 9, 11)
DORIAN = (0, 2, 3, 5, 7, 9, 10)
PHRYGIAN = (0, 1, 3, 5, 7, 8, 10)
LYDIAN = (0, 2, 4, 6, 7, 9, 11)
MIXOLYDIAN = (0, 2, 4, 5, 7, 9, 10)
AEOLIAN = (0, 2, 3, 5, 7, 8, 10)
LOCRIAN = (0, 1, 3, 5, 6, 8, 10)

# Harmonic minor's raised 7th, carried as part of Aeolian rather than as a mode
# of its own: the V and vii° of a minor key are everywhere in this repertoire,
# and a scorer that calls them out-of-key would fail every minor song.
HARMONIC_MINOR_EXTRA = 11

SCALES: dict[str, tuple[int, ...]] = {
    "ionian": IONIAN,
    "dorian": DORIAN,
    "phrygian": PHRYGIAN,
    "lydian": LYDIAN,
    "mixolydian": MIXOLYDIAN,
    "aeolian": AEOLIAN,
    "locrian": LOCRIAN,
}

# Which container mode each internal mode collapses to (§12.2). Decided by the
# third, which is the only thing `Key(tonic:mode:).spelling` uses it for.
MODE_PROJECTION: dict[str, str] = {
    "ionian": "major",
    "lydian": "major",
    "mixolydian": "major",
    "dorian": "minor",
    "phrygian": "minor",
    "aeolian": "minor",
    "locrian": "minor",
}

# Distance at or below which a disagreement reads as an engine confusion rather
# than as a chord change. 0.5 is the Jaccard score of two triads sharing two of
# three notes — the relative (C/Am), the mediant (C/Em), the parallel (C/Cm) and
# the suspension (C/Csus4). One step further out is C/F at 0.2, which is I
# against IV: a different chord, in every sense that matters to a player.
NEAR_MISS = 0.5

# Semitones above the tonic → numeral, named against the *major* scale whatever
# the mode is. That is the convention chord sheets actually use — a minor song's
# progression prints as "i bVI bIII bVII", not as "i VI III VII" — and it has the
# practical virtue of being total: every one of the twelve intervals has a name,
# including the harmonic-minor leading tone that a mode-relative naming would
# have to invent a spelling for.
_ROMAN = ("I", "bII", "II", "bIII", "III", "IV", "#IV", "V", "bVI", "VI", "bVII", "VII")


def pitch_classes(root_pc: int, quality: str) -> frozenset[int]:
    """The chord as a set of pitch classes. Unknown qualities read as a bare
    root, which is the conservative answer: it shares something with everything
    on the same root and nothing with anything else."""
    intervals = INTERVALS.get(quality, (0,))
    return frozenset((root_pc + i) % 12 for i in intervals)


def similarity(a: tuple[int, str], b: tuple[int, str]) -> float:
    """How alike two chords sound, in 0…1 — Jaccard over their pitch classes.

    Notes rather than labels, so the measure knows what the *engine* knew: two
    chords that share most of their spectrum are two readings a recognizer can
    plausibly slide between, whatever their names look like.
    """
    if a == b:
        return 1.0
    first, second = pitch_classes(*a), pitch_classes(*b)
    union = first | second
    if not union:
        return 0.0
    return len(first & second) / len(union)


# A quality → the triad underneath it. **Colour dropped, identity kept**: the
# seventh and the suspension go, the major/minor/diminished/augmented distinction
# stays. This is not `chords.simplify` and must not be confused with it —
# `_EASY` folds a diminished chord onto a minor triad because a beginner has to
# play *something*, which is a statement about the player. This is a statement
# about the recognizer: a seventh is the note BTC is least sure it heard, and a
# suspension is the third it could not find, so two readings that differ only
# there are two readings of one chord.
#
# The suspensions fold to major, which is the one arbitrary choice here — a
# sus chord has no third to read. It is the right arbitrary choice because the
# alternative (a token of their own) makes every `Csus4` heard in a song that
# plays `C` break the repeat it lands in, and that is the defect this exists to
# fix. Being wrong about a sus in a minor song costs one bar of a similarity
# score; being wrong about the form costs the whole chart.
_TRIAD: dict[str, str] = {
    MAJOR: MAJOR, MAJOR7: MAJOR, DOMINANT7: MAJOR, SUS2: MAJOR, SUS4: MAJOR,
    MINOR: MINOR, MINOR7: MINOR,
    DIMINISHED: DIMINISHED, DIMINISHED7: DIMINISHED, HALF_DIM7: DIMINISHED,
    AUGMENTED: AUGMENTED,
}


def triad(quality: str) -> str:
    """The triad a quality is built on. Unknown qualities are left alone, which
    keeps this total without inventing an answer for something it cannot read."""
    return _TRIAD.get(quality, quality)


def is_near_miss(a: tuple[int, str], b: tuple[int, str]) -> bool:
    """Whether a disagreement between these two is plausibly the engine's.

    The gate `consensus.py` will not vote across. Note it is deliberately *not*
    transitive and is not meant to be: it is a statement about one pair of
    readings of one bar, not an equivalence class of chords.
    """
    return similarity(a, b) >= NEAR_MISS


@lru_cache(maxsize=None)
def scale_pcs(tonic_pc: int, mode: str) -> frozenset[int]:
    """The mode's pitch classes, rooted at `tonic_pc`.

    Cached, and unbounded on purpose: the domain is 12 tonics × 7 modes = 84
    frozensets of at most 8 small ints, so the whole table is a rounding error in
    memory and is fully populated after one song. Uncached it rebuilt a set from a
    tuple on every call — and `diatonic_fit` calls it once per candidate chord
    inside both the consensus vote and the vocabulary consolidation, which are the
    hottest loops in the theory layer.
    """
    degrees = SCALES.get(mode, IONIAN)
    pcs = set((tonic_pc + d) % 12 for d in degrees)
    if mode == "aeolian":
        pcs.add((tonic_pc + HARMONIC_MINOR_EXTRA) % 12)
    return frozenset(pcs)


def diatonic_fit(root_pc: int, quality: str, tonic_pc: int, mode: str) -> float:
    """Share of the chord's notes that belong to the key, in 0…1.

    Graded rather than boolean because real songs are full of borrowed chords
    and secondary dominants, and a rule that called those errors would do more
    damage than the errors it caught. This is only ever used to break a tie
    between readings that are *already* near-misses of each other — never on its
    own, and never to reject a chord the engine was confident about.
    """
    notes = pitch_classes(root_pc, quality)
    if not notes:
        return 0.0
    return len(notes & scale_pcs(tonic_pc, mode)) / len(notes)


def roman(root_pc: int, quality: str, tonic_pc: int, mode: str) -> str:
    """The chord's scale degree, in the usual case-carrying notation.

    Read by `keyaudit.py`, which reports a key conflict as the two degrees in
    dispute ("C#(VI) vs C#m(vi)") — a musician can tell at a glance from that
    whether the audit found a secondary dominant or a misheard third, and cannot
    tell from the chord names alone. `mode` is accepted for symmetry with the
    rest of this module but does not change the spelling — see `_ROMAN`.

    Two neighbours of this function, `distance` and `is_tonic`, were removed in
    the 2026-08-18 audit: both were written for callers that never arrived, and a
    helper nothing calls is a helper nobody has checked.
    """
    numeral = _ROMAN[(root_pc - tonic_pc) % 12]
    minorish = quality in {MINOR, MINOR7, DIMINISHED, DIMINISHED7, HALF_DIM7}
    numeral = numeral.lower() if minorish else numeral
    if quality in {DIMINISHED, DIMINISHED7}:
        numeral += "°"
    elif quality == HALF_DIM7:
        numeral += "ø"
    elif quality == AUGMENTED:
        numeral += "+"
    return numeral


def is_dominant_of(root_pc: int, quality: str, tonic_pc: int) -> bool:
    """Whether this chord is the key's V (or V7) — the half-cadence test.

    A block that ends on the dominant is leaning on what comes next, which is
    the one structural cue that distinguishes a pre-chorus from a verse well
    enough to be worth printing on the player's screen (`form._assign_labels`,
    which gates it further — see the note there on why one repeated group is
    never enough).
    """
    return (root_pc - tonic_pc) % 12 == 7 and quality in {MAJOR, DOMINANT7, SUS4}

