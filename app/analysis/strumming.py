"""§14 — strumming patterns: what is recoverable from a mix, and what is invented.

The app *requires* a pattern (a section with no resolvable pattern is silently
dropped, §12.3), so this always produces one. What it must never do is present
the invented half as if it were measured, so the split is kept explicit here and
in every emitted pattern's `tags`:

| Dimension | Recoverable | How |
|---|---|---|
| onset positions | yes | fold this section's onsets onto one bar of the grid |
| subdivision (8ths vs 16ths) | yes | histogram onsets modulo the bar; keep the grid that explains them |
| accent | roughly | onset strength against the bar's own mean |
| **direction (down/up)** | **no — convention** | the alternating-hand rule, below |
| mute / percussive | not in a full mix | **never emitted** |

**Direction is a convention, not a measurement.** You cannot hear which way a
hand moved in a mixed recording. §14 states the rule as "an onset on a beat is a
downstroke, an onset on the '&' (or the second/fourth 16th) is an upstroke" — two
halves of one idea, and on a 16th grid they only agree under the reading taken
here: **strokes alternate from each beat**, so within a beat the even
subdivisions are down and the odd ones are up. On an 8th grid that gives D on the
beat and U on the "&"; on a 16th grid it gives D-U-D-U, with the 'e' and 'a' as
ups — exactly §14's parenthetical. It is also *correct* far more often than not,
because that is how the instrument is physically played.

**Don't over-fit.** Campfire draws the pattern as direction triangles under the
bar and the player strums through it. A 16-onset syncopated transcription of a
strummed acoustic is less playable — and less true to the song — than the
D-DU-UD-U everyone actually plays, so support thresholds here are deliberately
strict and the fallback is deliberately boring: a plain quarter-note downstroke
bar that plays beats a confident pattern that's wrong.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..payload import PATTERN_PREFIX, PatternPayload, Stroke, derived_uuid
from .axis import position_in
from .types import Onset

# Subdivisions worth testing: quarters, eighths, triplets, sixteenths (cells per
# quarter-note beat). Ordered coarsest-first so a tie resolves toward the simpler
# pattern, per "don't over-fit".
SUBDIVISIONS = (1, 2, 3, 4)

# A grid position is kept only if it carries an onset in at least this share of
# the section's bars. High on purpose: an onset that happens in a third of the
# bars is a fill, and putting a fill in the repeating pattern makes every bar
# wrong instead of one bar right.
SUPPORT_THRESHOLD = 0.5

# How close (in beats) an onset must sit to a grid cell to count as landing on it.
TOLERANCE_BEATS = 0.12

# Louder than this multiple of the bar's mean onset strength reads as an accent.
ACCENT_RATIO = 1.35

# Below this many strokes-worth of agreement the extraction is not trusted and
# the quarter-note fallback is used instead.
MIN_STROKES = 2

DOWN, UP = "down", "up"

# Tags every emitted pattern carries, so nobody downstream — or in six months —
# mistakes the direction assignment for detection.
CONVENTION_TAGS = ["yt", "extracted", "directions-by-convention"]


@dataclass(frozen=True)
class ExtractedPattern:
    pattern: PatternPayload
    # 0…1. Carried into `videoSync.patternConfidence` so the client can say the
    # strumming is a guess when it is one (§14).
    confidence: float
    # True when the quarter-note fallback was used — no usable onset support.
    is_fallback: bool = False


# Where `t_ms` falls on a beat axis, interpolating. Defined in `axis.py` so the
# chart, the anchors and the strumming extractor all read one implementation —
# three private copies of "which beat is this?" is what put the chart out of
# phase with its own recording in the first place.
beat_position = position_in


def fold_onsets(onsets: list[Onset], axis, *, bar_beats: float,
                first_beat: float, last_beat: float) -> list[tuple[float, float]]:
    """Onsets inside a beat range → (bar-local beat position, strength).

    "Folding" is the whole trick: every bar of a section is laid on top of every
    other, so a stroke played in all eight bars shows up as eight onsets at the
    same bar-local position and a one-off fill shows up as one.

    `axis` is a `BeatAxis` (a plain list of beat times is also accepted, which is
    what the unit tests hand it). Passing the axis matters: the bar-local
    position is taken modulo the bar, so folding against a beat list whose
    origin differs from the chart's would put every stroke on the wrong side of
    the beat.
    """
    locate = axis.position_at if hasattr(axis, "position_at") else \
        (lambda t: position_in(axis, t))
    folded: list[tuple[float, float]] = []
    for onset in onsets:
        position = locate(onset.t_ms)
        if position < first_beat - TOLERANCE_BEATS or position >= last_beat:
            continue
        local = (position - first_beat) % bar_beats
        folded.append((local, onset.strength))
    return folded


def choose_subdivision(positions: list[float]) -> int:
    """The coarsest grid that explains most of the onsets.

    Scored by the share of onsets landing within tolerance of a cell. Coarsest
    that comes within a small margin of the best wins, because 16ths explain
    everything 8ths explain (every 8th is also a 16th) and picking the finest
    grid would invent syncopation out of timing jitter.
    """
    if not positions:
        return 1
    scores: dict[int, float] = {}
    for subdivision in SUBDIVISIONS:
        cell = 1.0 / subdivision
        hits = sum(1 for p in positions if abs(p / cell - round(p / cell)) * cell <= TOLERANCE_BEATS)
        scores[subdivision] = hits / len(positions)
    best = max(scores.values())
    for subdivision in SUBDIVISIONS:
        if scores[subdivision] >= best - 0.1:
            return subdivision
    return 1


def direction_for(position: float, subdivision: int) -> str:
    """The alternating-hand rule — see the module docstring.

    Strokes alternate from each beat: even subdivisions within the beat are down,
    odd ones are up. This is a **convention**, not a measurement.
    """
    within_beat = position - int(position)
    cell = round(within_beat * subdivision)
    return DOWN if cell % 2 == 0 else UP


def extract(onsets_in_bar: list[tuple[float, float]], *, bar_beats: float, bars: int,
            tempo: int, name: str, time_signature: str = "4/4") -> ExtractedPattern:
    """Folded onsets → one bar of strokes (§14's method, end to end).

    `bars` is how many bars were folded together, and it is what turns a raw
    count into support: three onsets at the same position means everything across
    three bars and nothing across sixteen.

    `bar_beats` is quarter-note beats (the app's unit — ``n × 4/d``), so 6/8
    arrives here as 3.0 and its strokes land where 3/4's would. `time_signature`
    rides through unchanged so the emitted pattern names the meter the song
    actually uses.
    """
    if bars <= 0 or bar_beats <= 0:
        return fallback(bar_beats=max(1.0, bar_beats), tempo=tempo, name=name,
                        time_signature=time_signature)

    subdivision = choose_subdivision([p for p, _ in onsets_in_bar])
    cell = 1.0 / subdivision
    cells = int(round(bar_beats * subdivision))

    kept: list[tuple[float, float, float]] = []   # (position, support, mean strength)
    for index in range(cells):
        position = index * cell
        matched = [s for p, s in onsets_in_bar if abs(p - position) <= TOLERANCE_BEATS]
        support = len(matched) / bars
        if support >= SUPPORT_THRESHOLD:
            kept.append((position, min(1.0, support), sum(matched) / len(matched)))

    if len(kept) < MIN_STROKES:
        # Too thin or too noisy to be a pattern. A boring bar that plays is worth
        # more than a confident one that's wrong.
        return fallback(bar_beats=bar_beats, tempo=tempo, name=name,
                        time_signature=time_signature)

    mean_strength = sum(s for _, _, s in kept) / len(kept)
    strokes = [
        Stroke(
            beat=round(position, 4),
            direction=direction_for(position, subdivision),
            accent=strength >= mean_strength * ACCENT_RATIO,
        )
        for position, _, strength in kept
    ]
    # Confidence is the mean support of the strokes we kept: how reliably this
    # bar's shape actually repeated across the section.
    confidence = sum(support for _, support, _ in kept) / len(kept)
    pattern = _pattern(strokes, time_signature=time_signature, tempo=tempo, name=name)
    return ExtractedPattern(pattern=pattern, confidence=round(confidence, 3))


def fallback(*, bar_beats: float, tempo: int, name: str,
             time_signature: str = "4/4") -> ExtractedPattern:
    """A plain downstroke on every beat — §14's honest floor."""
    strokes = [Stroke(beat=float(b), direction=DOWN) for b in range(int(bar_beats))]
    pattern = _pattern(strokes, time_signature=time_signature, tempo=tempo,
                       name=name, extra_tags=["fallback"])
    return ExtractedPattern(pattern=pattern, confidence=0.0, is_fallback=True)


def _pattern(strokes: list[Stroke], *, time_signature: str, tempo: int, name: str,
             extra_tags: list[str] | None = None) -> PatternPayload:
    """Wrap strokes in a `PatternPayload` with a **content-addressed id**.

    Hashing the meter and the strokes means an unchanged groove compiles to an
    unchanged id, which is §12.5's "keep embedded pattern ids stable when their
    strokes are unchanged" — held by construction rather than by bookkeeping.
    Mo's `grid.py` does the same thing for the same reason.
    """
    body = f"{time_signature}|" + ";".join(
        f"{s.beat:.4f},{s.direction},{int(s.accent)}" for s in strokes
    )
    fingerprint = hashlib.sha1(body.encode("utf-8")).hexdigest()[:12]
    pattern_id = f"{PATTERN_PREFIX}{fingerprint}"
    # The strokes' own ids are derived too, so the whole pattern is a function of
    # its content — two runs over the same recording produce the same bytes.
    identified = [
        s.model_copy(update={"id": derived_uuid(pattern_id, "stroke", index)})
        for index, s in enumerate(strokes)
    ]
    return PatternPayload(
        id=pattern_id,
        name=name,
        timeSignature=time_signature,
        tempo=tempo,
        strokes=identified,
        tags=CONVENTION_TAGS + (extra_tags or []),
    )
