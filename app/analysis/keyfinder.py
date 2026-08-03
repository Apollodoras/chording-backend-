"""Key detection from the chord track — because the container demands one.

``CompositionPayload`` requires ``tonic`` and ``mode``, and the app uses them for
exactly one thing: **note spelling** (``Key(tonic, mode)?.spelling``). Get the key
wrong and the song still plays; the player just reads "A#m" where every chord
sheet in the world prints "Bbm".

So this does not need to be a musicological key-finder over audio, and
deliberately isn't one. It scores the *chords we already decided on* — which are
the strongest key evidence there is, and free — against the 24 major/minor keys.
Working from chords rather than from chroma also means the key can never
contradict the chart, which is the failure that would actually confuse a player.

``mode`` is major/minor only: §12.2, the song container knows no church modes.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    spell,
)
from .types import GridSpan

# Scale degree (semitones above the tonic) → the qualities that belong there.
# Read as "what a chord sheet in this key actually prints", not as strict theory:
# V is listed with both `major` and `dominant7` because both are the dominant,
# and the minor key carries its harmonic-minor V alongside the natural v.
_MAJOR_DEGREES: dict[int, set[str]] = {
    0: {MAJOR, MAJOR7, SUS2, SUS4},
    2: {MINOR, MINOR7},
    4: {MINOR, MINOR7},
    5: {MAJOR, MAJOR7, SUS2, SUS4},
    7: {MAJOR, DOMINANT7, SUS4},
    9: {MINOR, MINOR7},
    11: {DIMINISHED, HALF_DIM7},
}

_MINOR_DEGREES: dict[int, set[str]] = {
    0: {MINOR, MINOR7, SUS2, SUS4},
    2: {DIMINISHED, HALF_DIM7},
    3: {MAJOR, MAJOR7},
    5: {MINOR, MINOR7},
    7: {MINOR, MINOR7, MAJOR, DOMINANT7},   # natural v and harmonic V
    8: {MAJOR, MAJOR7},
    10: {MAJOR, DOMINANT7},
    11: {DIMINISHED, DIMINISHED7},          # leading-tone chord, harmonic minor
}

# A root that belongs to the key but wears an unexpected quality is still strong
# evidence for the key — borrowed chords are normal, wrong roots are not.
_QUALITY_MATCH = 1.0
_ROOT_ONLY = 0.45
# The tonic chord opening or closing a song is the single most reliable cue there
# is, and it is what separates a key from its relative major/minor — those two
# share every diatonic chord, so without this they score identically.
_TONIC_ENDPOINT_BONUS = 0.35

_UNRESOLVED = {AUGMENTED}   # belongs to no diatonic degree; ignored rather than penalised


@dataclass(frozen=True)
class DetectedKey:
    tonic: str
    mode: str
    confidence: float


def detect_key(spans: list[GridSpan]) -> DetectedKey:
    """Best-fitting key for a chord track, with a confidence in 0…1.

    Confidence is the **margin** over the runner-up, not the winner's raw score:
    "this key fits 80% of the song" says little when a neighbouring key fits 79%,
    and the relative-major/minor pair is exactly that case. A near-tie reports low
    confidence, which is the truthful answer.
    """
    if not spans:
        return DetectedKey("C", "major", 0.0)

    total = sum(s.length_beats for s in spans) or 1
    scored: list[tuple[float, str, int]] = []
    for tonic_pc in range(12):
        for mode, degrees in (("major", _MAJOR_DEGREES), ("minor", _MINOR_DEGREES)):
            scored.append((_score(spans, tonic_pc, degrees) / total, mode, tonic_pc))
    scored.sort(reverse=True)

    best_score, best_mode, best_pc = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    # Scale the margin so a decisive win (≥ 0.25 clear) reads as ~1.0; the raw
    # margin between two keys is small even when the answer is obvious.
    confidence = max(0.0, min(1.0, (best_score - runner_up) * 4.0))

    flats = _prefers_flat_spelling(best_pc, best_mode)
    return DetectedKey(spell(best_pc, flats), best_mode, round(confidence, 3))


def _score(spans: list[GridSpan], tonic_pc: int, degrees: dict[int, set[str]]) -> float:
    total = 0.0
    for span in spans:
        degree = (span.root_pc - tonic_pc) % 12
        qualities = degrees.get(degree)
        if qualities is None:
            continue
        if span.quality in qualities:
            total += span.length_beats * _QUALITY_MATCH
        elif span.quality not in _UNRESOLVED:
            total += span.length_beats * _ROOT_ONLY

    for endpoint in (spans[0], spans[-1]):
        if endpoint.root_pc % 12 == tonic_pc:
            total += endpoint.length_beats * _TONIC_ENDPOINT_BONUS
    return total


def _prefers_flat_spelling(tonic_pc: int, mode: str) -> bool:
    """Which spelling of the *tonic itself* to emit.

    ``chords.prefers_flats`` answers the same question for a tonic already
    spelled; this one has only a pitch class, so it names the five flat keys (and
    the six flat minors) by number. F/Bb/Eb/Ab/Db major and d/g/c/f/bb/eb minor.
    """
    if mode == "minor":
        return tonic_pc in {2, 7, 0, 5, 10, 3}
    return tonic_pc in {5, 10, 3, 8, 1}
