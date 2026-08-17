"""§14 — strumming patterns: what is recoverable from a mix, and what is invented.

The app *requires* a pattern (a section with no resolvable pattern is silently
dropped, §12.3), so this always produces one. What it must never do is present
the invented half as if it were measured, so the split is kept explicit here and
in every emitted pattern's `tags`:

| Dimension | Recoverable | How |
|---|---|---|
| onset positions | yes | fold this section's onsets onto one bar of the grid |
| subdivision (8ths vs 16ths) | yes | quantize onsets modulo the bar; keep the coarsest grid they sit on |
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

That was the stated intent for a long time before the thresholds delivered it.
Support is a share of *bars*, and on a full mix nearly every cell is supported —
the kit plays in every bar — so "keep everything over the threshold" kept
fifteen of sixteen cells on a recording whose guitar plays six. Three rules now
stand between the grid and the pattern, in this order:

1. **contrast** — a cell must reach a fraction of the bar's strongest cell, which
   is what tells a hand from a hi-hat when support cannot;
2. **a budget** — at most two strokes per beat, spent on the strongest and most
   metrically important cells, so a saturated bar reads as the skeleton
   underneath it;
3. **a vocabulary** — an extraction that comes within a cell of a strum people
   actually play is snapped onto it, tagged `snapped-to-idiom`, the same
   measure-then-snap discipline `vocabulary.SNAP_TO` follows for chords.

The user's acceptance criterion for all of this is worth keeping in the file:
*if a pattern doesn't repeat, it isn't a pattern.* The half of that this module
can enforce is here; the half about how many patterns a song gets to have is in
`model._patterns`.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import NamedTuple

from ..payload import PATTERN_PREFIX, PatternPayload, Stroke, derived_uuid
from .axis import position_in
from .types import Onset

log = logging.getLogger("chords.strumming")

# Subdivisions worth testing: quarters, eighths, triplets, sixteenths (cells per
# quarter-note beat). Ordered coarsest-first so a tie resolves toward the simpler
# pattern, per "don't over-fit".
SUBDIVISIONS = (1, 2, 3, 4)

# A grid position is kept only if it carries an onset in at least this share of
# the section's bars. High on purpose: an onset that happens in a third of the
# bars is a fill, and putting a fill in the repeating pattern makes every bar
# wrong instead of one bar right.
SUPPORT_THRESHOLD = 0.5

# ...and it must also hold up against the bar's *strongest* cell. Support alone
# has no opinion about a bar where everything is supported, and on a full mix
# that is the normal case: `LibrosaOnsetDetector` fires on the drum kit, a hi-hat
# playing 8ths is present in every bar, and so more than half the bars carry an
# onset near nearly every 16th cell. Measured on a clean 120 bpm grid, a 6-stroke
# D-DU-UD-U with a kit behind it came out as fifteen of the sixteen cells — the
# literal opposite of what this module's docstring promises.
#
# A groove is contrast: which cells are struck, against which are not. Compared
# with the bar's *mean* that test is self-defeating — six struck cells out of
# eight drag the mean up until the six can no longer clear it — so the reference
# is the peak, and a cell under this fraction of it is the kit, not the hand.
CONTRAST_RATIO = 0.45

# And a hard ceiling, in strokes per quarter-note beat — eight in 4/4. Nobody
# strums sixteen times a bar on the material this service accepts, and a pattern
# that says so is unplayable however well-supported each cell was.
MAX_STROKES_PER_BEAT = 2

# How close (in beats) an onset must sit to a grid cell to count as landing on it.
TOLERANCE_BEATS = 0.12

# How much *average* quantization error a coarser grid may cost before the finer
# one is worth taking (see `choose_subdivision`). 0.05 beats is 25 ms at 120 bpm
# — under the tolerance above, so a grid is never refined for timing jitter, and
# small enough that a real subdivision (which costs a quarter-beat per onset it
# cannot place) always clears it.
SUBDIVISION_MARGIN_BEATS = 0.05

# Louder than this multiple of the bar's mean onset strength reads as an accent.
ACCENT_RATIO = 1.35

# Below this many strokes-worth of agreement the extraction is not trusted and
# the quarter-note fallback is used instead.
MIN_STROKES = 2

# ...unless the one stroke is played this reliably. A chord struck on the downbeat
# of nine bars in ten is a real (if sparse) pattern — slow ballads are played that
# way — and replacing it with the four-downstroke fallback puts *more* strokes in
# the bar than the recording has.
SOLID_SUPPORT = 0.9

DOWN, UP = "down", "up"

# Tags every emitted pattern carries, so nobody downstream — or in six months —
# mistakes the direction assignment for detection.
CONVENTION_TAGS = ["yt", "extracted", "directions-by-convention"]

# ...and the tag a snapped pattern carries as well, for the same reason: the
# strokes it names were chosen from the library below, not measured off this
# recording. Exactly the discipline `vocabulary.SNAP_TO` follows on the chord
# side, and named the same way on purpose.
SNAPPED_TAG = "snapped-to-idiom"

# The strums people actually play, in bar-local quarter-note beats, keyed by the
# bar's length. Small and deliberately unambitious: this is a *vocabulary*, not a
# generator, and its job is to catch an extraction that came within one cell of
# something idiomatic — not to have an entry for every groove in the world.
IDIOMS: dict[float, tuple[tuple[str, tuple[float, ...]], ...]] = {
    4.0: (
        ("half notes", (0.0, 2.0)),
        ("quarters", (0.0, 1.0, 2.0, 3.0)),
        ("the campfire pattern", (0.0, 1.0, 1.5, 2.5, 3.0, 3.5)),
        ("offbeats", (0.5, 1.5, 2.5, 3.5)),
        ("eighths", (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)),
    ),
    3.0: (
        ("dotted half", (0.0,)),
        ("waltz quarters", (0.0, 1.0, 2.0)),
        ("waltz eighths", (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)),
        ("6/8 in two", (0.0, 1.5)),
    ),
    2.0: (
        ("halves", (0.0,)),
        ("quarters", (0.0, 1.0)),
        ("eighths", (0.0, 0.5, 1.0, 1.5)),
    ),
}

# How alike an extraction and a library entry must be before the entry is taken
# instead. High: a 16th feel of (0, .75, 1, 1.75, 2, 3) is two thirds of the way
# to plain quarters by this measure and it is **not** plain quarters, so anything
# looser would flatten real playing into the nearest cliché. What clears it is an
# extraction that missed or gained a single cell of something idiomatic, which is
# the case worth rescuing.
SNAP_SIMILARITY = 0.7

# How far apart two stroke positions may be and still be the same stroke, when
# an extraction is matched against the library.
SNAP_TOLERANCE_BEATS = 0.05


class FoldedOnset(NamedTuple):
    """One onset laid onto the pattern bar, with the bar it came from.

    The bar identity is not decoration: support is "in how many *bars* was this
    stroke played", and counting onsets instead lets one bar with a flam or a
    doubled drum hit stand in for two bars of evidence.
    """

    bar: int
    position: float     # bar-local, in quarter-note beats
    strength: float


class _Cell(NamedTuple):
    """One position on the pattern grid, with everything scored about it.

    `prominence` is support scaled by how loud the cell is against the bar's own
    mean onset — the number `_with_contrast` thresholds on. Support alone cannot
    tell a strum from a hi-hat, because on a full mix both are present in every
    bar; loudness can, and it is the same evidence `ACCENT_RATIO` already trusts.
    """

    position: float
    support: float
    strength: float
    prominence: float


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
                first_beat: float, last_beat: float) -> list[FoldedOnset]:
    """Onsets inside a beat range → `FoldedOnset`s on one bar of the grid.

    "Folding" is the whole trick: every bar of a section is laid on top of every
    other, so a stroke played in all eight bars shows up as eight onsets at the
    same bar-local position and a one-off fill shows up as one.

    `axis` is a `BeatAxis` (a plain list of beat times is also accepted, which is
    what the unit tests hand it). Passing the axis matters: the bar-local
    position is taken modulo the bar, so folding against a beat list whose
    origin differs from the chart's would put every stroke on the wrong side of
    the beat.

    **A downbeat stroke is rarely late and often early.** Taken modulo the bar,
    a hand that arrives 20 ms *ahead* of the "one" folds to ~3.98 rather than
    ~0.0 — the far end of the bar, supporting nothing. Nobody plays there; they
    played the downbeat of the *next* bar and got there first. So an onset within
    tolerance of the top of the bar is rolled forward onto that downbeat, which
    is both where it belongs on the grid and which bar it is evidence for. The
    bar index is global (it counts from the song's beat 0, not from
    `first_beat`), so pooling several occurrences of the same section keeps their
    bars distinct.
    """
    locate = axis.position_at if hasattr(axis, "position_at") else \
        (lambda t: position_in(axis, t))
    origin_bar = int(round(first_beat / bar_beats))
    # A stroke anticipating the downbeat that *ends* the range belongs to the
    # next section, not this one — it is evidence for a bar we are not folding.
    bar_limit = origin_bar + int(round((last_beat - first_beat) / bar_beats))
    folded: list[FoldedOnset] = []
    for onset in onsets:
        position = locate(onset.t_ms)
        if position < first_beat - TOLERANCE_BEATS or position >= last_beat:
            continue
        offset = position - first_beat
        bar = origin_bar + int(offset // bar_beats)
        local = offset % bar_beats
        if local > bar_beats - TOLERANCE_BEATS:
            local -= bar_beats          # a small negative: just shy of the "one"
            bar += 1
        if bar >= bar_limit:
            continue
        folded.append(FoldedOnset(bar=bar, position=local, strength=onset.strength))
    return folded


def choose_subdivision(positions: list[float]) -> int:
    """The coarsest grid the onsets sit on, scored by mean quantization error.

    Each grid is scored by how far, on average, an onset lies from its nearest
    cell — and coarsest-first wins, but only where "coarser" actually means
    "less detail about the same grid":

    - Against a grid it **nests inside** (every 8th is also a 16th) the coarser
      one wins unless the finer one is better by more than
      `SUBDIVISION_MARGIN_BEATS`. This is §14's "don't over-fit": without the
      bias, timing jitter alone would always favour the finest grid and invent
      syncopation out of it.
    - Against a grid it **doesn't** nest inside, the better fit simply wins.
      Triplets and 16ths are different feels, not two resolutions of one grid, and
      3 is only nominally coarser than 4 — a 16th sits just 1/12 of a beat off a
      triplet cell, so a margin measured in beats would hand every 16th-note
      pattern to the triplet grid.

    Mean error is also what makes this survive a full mix. Scored by *share of
    onsets explained*, one consistent hi-hat 16th per bar — a sixth of the
    onsets, and present in every drummed recording — was enough to carry the vote
    to 16ths, and since `direction_for` reads the grid, every "&" in the bar
    flipped from an upstroke to a downstroke. Averaged instead, that ghost costs
    a quarter-beat spread across the bar's strokes and changes nothing; two of
    them per bar, which is a real 16th feel, still wins.
    """
    if not positions:
        return 1
    scores: dict[int, float] = {}
    for subdivision in SUBDIVISIONS:
        cell = 1.0 / subdivision
        error = sum(abs(p / cell - round(p / cell)) * cell for p in positions)
        scores[subdivision] = error / len(positions)

    for subdivision in SUBDIVISIONS:
        finer = [s for s in SUBDIVISIONS if s > subdivision]
        nested = [s for s in finer if s % subdivision == 0]
        rivals = [s for s in finer if s % subdivision != 0]
        if any(scores[subdivision] > scores[s] + SUBDIVISION_MARGIN_BEATS for s in nested):
            continue
        if any(scores[subdivision] > scores[s] for s in rivals):
            continue
        return subdivision
    return SUBDIVISIONS[-1]


def direction_for(position: float, subdivision: int) -> str:
    """The alternating-hand rule — see the module docstring.

    Strokes alternate from each beat: even subdivisions within the beat are down,
    odd ones are up. This is a **convention**, not a measurement.
    """
    within_beat = position - int(position)
    cell = round(within_beat * subdivision)
    return DOWN if cell % 2 == 0 else UP


def extract(onsets_in_bar: list[FoldedOnset], *, bar_beats: float, bars: int,
            tempo: int, name: str, time_signature: str = "4/4") -> ExtractedPattern:
    """Folded onsets → one bar of strokes (§14's method, end to end).

    `bars` is how many bars were folded together, and it is what turns a raw
    count into support: a stroke played in three bars means everything across
    three bars and nothing across sixteen.

    `bar_beats` is quarter-note beats (the app's unit — ``n × 4/d``), so 6/8
    arrives here as 3.0 and its strokes land where 3/4's would. `time_signature`
    rides through unchanged so the emitted pattern names the meter the song
    actually uses.
    """
    if bars <= 0 or bar_beats <= 0:
        return fallback(bar_beats=max(1.0, bar_beats), tempo=tempo, name=name,
                        time_signature=time_signature)

    folded = [FoldedOnset(*onset) for onset in onsets_in_bar]
    subdivision = choose_subdivision([onset.position for onset in folded])
    cell = 1.0 / subdivision
    cells = int(round(bar_beats * subdivision))

    # The bar's own loudness scale, so `prominence` below is a ratio and not a
    # level. A cell struck in every bar by something the mix barely registers —
    # a hi-hat — is not a stroke of the pattern however reliable it is.
    loudness = [o.strength for o in folded]
    mean_onset_strength = sum(loudness) / len(loudness) if loudness else 1.0

    scored: list[_Cell] = []
    for index in range(cells):
        position = index * cell
        matched = [o for o in folded
                   if _distance_in_bar(o.position, position, bar_beats) <= TOLERANCE_BEATS]
        # Support is a share of *bars*, not of onsets: a cell struck twice in half
        # the bars is half-supported, however many onsets that is.
        support = min(1.0, len({o.bar for o in matched}) / bars)
        strength = sum(o.strength for o in matched) / len(matched) if matched else 0.0
        relative = strength / mean_onset_strength if mean_onset_strength else 0.0
        scored.append(_Cell(position=position, support=support, strength=strength,
                            prominence=support * relative))

    kept = _with_contrast(scored)
    if not _is_a_pattern(kept):
        # Too thin, too noisy, or too *uniform* to be a pattern. A boring bar
        # that plays is worth more than a confident one that's wrong.
        return fallback(bar_beats=bar_beats, tempo=tempo, name=name,
                        time_signature=time_signature)

    kept, snapped = snap_to_idiom(kept, bar_beats=bar_beats)

    # Re-read the grid off the strokes that survived, because that is the grid
    # the pattern is actually on and `direction_for` reads directions from it.
    # A bar whose 16ths were all struck chooses a 16th grid, keeps the eight-note
    # skeleton, and would otherwise have every "&" come out a downstroke — the
    # same failure the one-ghost-hi-hat guard exists to prevent, arriving from
    # the other side.
    subdivision = choose_subdivision([c.position for c in kept])

    mean_strength = sum(c.strength for c in kept) / len(kept)
    strokes = [
        Stroke(
            beat=round(c.position, 4),
            direction=direction_for(c.position, subdivision),
            accent=c.strength >= mean_strength * ACCENT_RATIO,
        )
        for c in kept
    ]
    # Confidence is the mean support of the strokes we kept: how reliably this
    # bar's shape actually repeated across the section.
    confidence = sum(c.support for c in kept) / len(kept)
    pattern = _pattern(strokes, time_signature=time_signature, tempo=tempo, name=name,
                       extra_tags=[SNAPPED_TAG] if snapped else None)
    return ExtractedPattern(pattern=pattern, confidence=round(confidence, 3))


def _with_contrast(scored: list[_Cell]) -> list[_Cell]:
    """The cells that are actually the pattern, out of every cell of the grid.

    Two rules on top of `SUPPORT_THRESHOLD`, and they answer different halves of
    the same complaint:

    **Contrast.** A cell has to reach `CONTRAST_RATIO` of the bar's strongest
    cell. That is what separates the hand from the kit when both are present in
    every bar and support therefore says nothing: on the measured case the six
    strums sit near the peak and the hi-hat's own cells sit at a third of it, so
    the six survive and the wall of eighths does not.

    **A ceiling.** Whatever survives, at most `MAX_STROKES_PER_BEAT` per beat,
    ranked by prominence and then by metrical weight — so when the budget bites
    it spends what it has on the downbeat and the beats before the offbeats.
    This is the rule that catches the genuinely saturated bar, where every cell
    is struck *equally* and contrast has nothing to discriminate with: sixteen
    equal cells become the eight-note skeleton underneath them, which is what
    the recording is actually playing.
    """
    if not scored:
        return []
    peak = max(c.prominence for c in scored)
    kept = [c for c in scored
            if c.support >= SUPPORT_THRESHOLD
            and c.prominence >= peak * CONTRAST_RATIO]

    budget = max(1, int(round(MAX_STROKES_PER_BEAT * _beats_in(scored))))
    if len(kept) > budget:
        kept = sorted(sorted(kept, key=lambda c: c.position),
                      key=lambda c: (-c.prominence, -_metrical_weight(c.position)))[:budget]
    return sorted(kept, key=lambda c: c.position)


def _beats_in(scored: list[_Cell]) -> float:
    """The bar's length in quarter-note beats, read back off the cell grid."""
    if len(scored) < 2:
        return float(len(scored))
    cell = scored[1].position - scored[0].position
    return len(scored) * cell if cell > 0 else float(len(scored))


def _metrical_weight(position: float) -> float:
    """How strong a beat this is — the tie-break when the budget bites.

    A downbeat outranks a beat, a beat outranks an "&", an "&" outranks an "e"
    or an "a". This is the one place a musical prior is applied rather than
    measured, and it only ever chooses *between* cells the evidence already
    likes equally.
    """
    if abs(position) < 1e-6:
        return 3.0
    if abs(position - round(position)) < 1e-6:
        return 2.0
    if abs(position * 2 - round(position * 2)) < 1e-6:
        return 1.0
    return 0.0


def snap_to_idiom(kept: list[_Cell], *, bar_beats: float) -> tuple[list[_Cell], bool]:
    """Pull an extraction onto the nearest strum people actually play (§14).

    The direct answer to "patterns should be more musical". The chord side
    already has this shape in `vocabulary.SNAP_TO`: measure first, then move the
    measurement onto what the repertoire actually contains — and only when the
    two are already close, so the snap is a *correction* and never an invention.
    An extraction that lands a cell short of D-DU-UD-U almost certainly is
    D-DU-UD-U with one stroke under the support threshold; an extraction that is
    nothing like anything in the library is left exactly as measured.

    A snapped position keeps the support and strength of the extracted cell it
    came from, so accents stay a measurement even where the positions are not,
    and a position the library added but the recording did not play inherits the
    bar's mean — it is a stroke we are asserting, and its confidence should say
    so rather than borrow a neighbour's.
    """
    library = IDIOMS.get(round(bar_beats, 3))
    if not library or not kept:
        return kept, False

    positions = [c.position for c in kept]
    best_name, best_positions, best_score = "", (), 0.0
    for name, candidate in library:
        score = stroke_similarity(positions, candidate)
        # Ties go to the sparser entry: "don't over-fit" applies to the library
        # exactly as it applies to the grid.
        if (score, -len(candidate)) > (best_score, -len(best_positions)):
            best_name, best_positions, best_score = name, candidate, score

    if best_score < SNAP_SIMILARITY:
        return kept, False
    if best_score >= 1.0 and len(positions) == len(best_positions):
        return kept, False              # already idiomatic; nothing to snap
    log.info("strum snapped to %s (%.2f similar): %s → %s", best_name, best_score,
             [round(p, 3) for p in positions], list(best_positions))

    mean_support = sum(c.support for c in kept) / len(kept)
    mean_strength = sum(c.strength for c in kept) / len(kept)
    snapped: list[_Cell] = []
    for position in best_positions:
        source = min(kept, key=lambda c: abs(c.position - position))
        near = abs(source.position - position) <= SNAP_TOLERANCE_BEATS
        snapped.append(_Cell(
            position=position,
            support=source.support if near else mean_support,
            strength=source.strength if near else mean_strength,
            prominence=source.prominence if near else 0.0,
        ))
    return snapped, True


def stroke_similarity(a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]) -> float:
    """Jaccard over two stroke sets, matched with a tolerance rather than by
    equality — a cell at 1.4999 and one at 1.5 are the same stroke."""
    if not a or not b:
        return 0.0
    remaining = list(b)
    hits = 0
    for position in a:
        match = next((q for q in remaining
                      if abs(q - position) <= SNAP_TOLERANCE_BEATS), None)
        if match is not None:
            remaining.remove(match)
            hits += 1
    return hits / (len(a) + len(b) - hits)


def _distance_in_bar(position: float, cell: float, bar_beats: float) -> float:
    """How far an onset is from a cell, **around** the bar rather than along it.

    The bar is a loop, not a line: the last cell and the downbeat are adjacent,
    one grid step apart. Measured linearly, an onset a hair ahead of the "one"
    sits a whole bar away from it and no cell can claim it — which is exactly how
    the beat-1 stroke used to vanish from patterns extracted off real recordings
    while the extraction still reported full confidence in the rest.
    """
    direct = abs(position - cell)
    return min(direct, bar_beats - direct)


def _is_a_pattern(kept: list[_Cell]) -> bool:
    """Whether the kept cells are worth emitting instead of the fallback."""
    if len(kept) >= MIN_STROKES:
        return True
    # One stroke is a pattern only when it is played almost every bar; anything
    # less is a coincidence with nothing else in the bar to corroborate it.
    return len(kept) == 1 and kept[0].support >= SOLID_SUPPORT


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
