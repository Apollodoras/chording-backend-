"""§5.4 — turning raw engine output into something playable.

"Raw output is not a game chart." A frame-level chord recognizer produces
boundaries that wobble a few tens of milliseconds either side of the beat, a
scatter of 200 ms chords that are really one chord with a passing bass note, and
a vocabulary far wider than the app can sound. This module fixes all three, in
the order the handoff specifies, and it is **pure** — no audio, no engine, no
I/O — so every rule here is a unit test rather than a listening session.

The order matters and is §5.4's:

1. **quantize** boundaries to the nearest beat,
2. **merge** runs of the same chord,
3. **drop** spans shorter than a beat,
4. **fill** N/C gaps — *hold the previous chord* (§18: there is no rest primitive;
   a stroke always sounds a chord),
5. **confidence gate** (§5.4.6).

§5.5's difficulty simplification used to be step 5 of this chain, and it is gone.
The only reduction left is the one `normalize` cannot avoid — the app's grammar
is a closed vocabulary — and it is *counted* (`exact_ratio`) rather than chosen.
A chart that prints G where the record plays Gmaj7 is not an easier chart, it is
a wrong one.
"""

from __future__ import annotations

from dataclasses import replace

from ..chords import normalize
from .axis import BeatAxis
from .types import GridSpan, RawChordSpan

# §5.4.3's floor: "shorter than one beat, or < 250ms". Working in beats, one beat
# IS the floor, and it is the more honest of the two — at 60 bpm a 250 ms span is
# a quarter of a beat, at 180 bpm it is most of one.
MIN_SPAN_BEATS = 1


def quantize(spans: list[RawChordSpan], axis: BeatAxis) -> list[GridSpan]:
    """Snap every chord boundary to the nearest beat, in beat-index units.

    A chord change that lands 40 ms before the beat is the same musical event as
    one that lands on it, and the app draws strokes on the grid — so an
    unquantized boundary shows up as a chord that changes *between* two strokes,
    which reads as a glitch rather than as detail.

    The index is into the **`BeatAxis`**, not into whatever the tracker happened
    to emit: beat 0 is the song's first downbeat, so a chord landing on bar 3's
    downbeat lands on beat `3 · bar_beats` here and on anchor 3 in the sidecar.
    Those used to be two different origins (see `axis.py`), which put the whole
    chart out of phase with its own recording.

    Spans that collapse to zero length are dropped here rather than emitted as
    empty: a chord shorter than half a beat has no beat of its own to live on.
    """
    if len(axis.times_ms) < 2:
        return []

    out: list[GridSpan] = []
    for span in spans:
        parsed = normalize(span.label)
        if parsed is None:
            # No-chord. Kept out of the quantized list entirely; step 4 fills the
            # hole it leaves by extending its neighbour.
            continue
        root, quality, exact = parsed
        start = axis.beat_at(span.start_ms)
        end = axis.beat_at(span.end_ms)
        if end <= start:
            continue
        out.append(GridSpan(
            start_beat=start, length_beats=end - start,
            root_pc=root, quality=quality,
            confidence=span.confidence, exact=exact,
        ))
    return out


def merge(spans: list[GridSpan]) -> list[GridSpan]:
    """Collapse consecutive spans of the same chord into one.

    Also closes gaps *between* chords by extending the earlier one: after
    quantization two adjacent chords can end up a beat apart, and a hole in the
    harmony has no meaning in a container where every stroke sounds a chord.

    And it decides between two *different* chords quantization put on the same
    beat, which is the one case where this function has to choose rather than
    rearrange. See the branch itself for why the choice is "more evidence wins"
    rather than "first one seen".
    """
    if not spans:
        return []
    ordered = sorted(spans, key=lambda s: s.start_beat)
    out = [ordered[0]]
    for span in ordered[1:]:
        last = out[-1]
        same = span.root_pc == last.root_pc and span.quality == last.quality
        if same and span.start_beat <= last.end_beat:
            end = max(last.end_beat, span.end_beat)
            out[-1] = replace(
                last,
                length_beats=end - last.start_beat,
                # A merged span is only "exact" if every part of it was, and its
                # confidence is the weaker of the two — merging must not launder
                # a doubtful chord into a confident one.
                confidence=min(last.confidence, span.confidence),
                exact=last.exact and span.exact,
            )
        elif span.start_beat < last.end_beat:
            # Overlapping different chords: the later one wins from where it
            # starts, so the earlier is truncated rather than dropped.
            if span.start_beat > last.start_beat:
                out[-1] = replace(last, length_beats=span.start_beat - last.start_beat)
                out.append(span)
            else:
                # Same start beat, different chords — two engine readings of one
                # slot rather than two chords, because quantization put them
                # there: a 300 ms Am inside a bar of C rounds onto C's own
                # downbeat. Dropping the second silently kept whichever the sort
                # happened to reach first, so the chart could show the passing
                # chord and lose the one the bar is in. Keep the one with more
                # evidence.
                #
                # **Evidence is duration × confidence**, not duration and then
                # confidence. Ordering the pair lexicographically made length win
                # outright, so a four-beat span the engine hedged at 0.3 beat a
                # one-beat span it was sure of — and the second reading is worth
                # more than the first by any reading of what a posterior means.
                # This is `vocabulary.Reading.mass`, which is the same quantity
                # under the same argument; the tie-break behind it keeps the
                # result independent of the sort order (§16.5 wants byte-stable
                # output).
                mine = (span.length_beats * span.confidence, span.length_beats)
                theirs = (last.length_beats * last.confidence, last.length_beats)
                if mine > theirs:
                    out[-1] = span
        else:
            if span.start_beat > last.end_beat:
                # Gap — hold the previous chord across it (§18).
                out[-1] = replace(last, length_beats=span.start_beat - last.start_beat)
            out.append(span)
    return out


def drop_short(spans: list[GridSpan], min_beats: int = MIN_SPAN_BEATS) -> list[GridSpan]:
    """Remove spans below the floor, giving their time to the previous chord.

    Not "delete": the beats have to go *somewhere*, and handing them back to the
    neighbour is what makes a run of jitter read as one held chord. A too-short
    first span is absorbed forward instead, into whatever follows it.

    At the one-beat floor this is cleaning up jitter inside a held chord, so the
    previous chord is always the right home. There used to be a `to_longer` mode
    that gave each short span to its *longer* neighbour instead, because at
    `easy`'s one-*bar* floor the rule was doing something else entirely —
    deciding which of two real chords a beginner sees, where always reaching left
    swallowed the `| IV V |` cadence bar that resolves a phrase. That floor was
    the difficulty tiers' and went with them; at one beat there is no second real
    chord to choose between.
    """
    if not spans:
        return []
    kept: list[GridSpan] = []
    pending_beats = 0
    for span in spans:
        if span.length_beats < min_beats:
            if kept:
                kept[-1] = replace(kept[-1],
                                   length_beats=kept[-1].length_beats + span.length_beats)
            else:
                pending_beats += span.length_beats
            continue
        if pending_beats and not kept:
            span = replace(span, start_beat=span.start_beat - pending_beats,
                           length_beats=span.length_beats + pending_beats)
            pending_beats = 0
        kept.append(span)
    return kept


def hold_through_gaps(spans: list[GridSpan], total_beats: int | None = None) -> list[GridSpan]:
    """Make the chord timeline contiguous from beat 0 to the end.

    §18's answer to N/C, applied to the two holes quantization leaves: a gap in
    the middle (the previous chord is held across it) and a gap at the start
    (the first chord is pulled back to beat 0). There is no rest primitive in the
    container, so the alternative to holding is a section with no chord — and a
    section with empty chords is **silently dropped** by the importer.

    For a genuinely long instrumental gap the handoff's other option is a section
    of its own with a sparse pattern; that is a structure decision and lives in
    `structure.py`, not here.

    **What the fill costs, and how much of it has been paid back** (F33). The
    2026-08-18 audit's suggestion was to keep N.C. internally and fill only at
    compile time, which would buy three things: a key detected on what the band
    played, no phantom chords in a break, and sections with no harmony able to
    say so. The first is now had for free — `keyfinder` reads `process(...,
    fill=False)` and never sees a held chord (see `process`). The other two need
    an N.C. that survives all the way to `compile`, and that is a change to what a
    `GridSpan` *is* rather than to where this function is called: every consumer
    between here and the wire assumes a contiguous timeline (`bars_from_spans`,
    `form`, `consensus`, `canon`, `axis`), and §18 has no rest primitive to hand
    them. Worth doing, not worth doing halfway.
    """
    if not spans:
        return []
    ordered = sorted(spans, key=lambda s: s.start_beat)
    out: list[GridSpan] = []
    cursor = 0
    for span in ordered:
        if span.start_beat > cursor:
            if out:
                out[-1] = replace(out[-1], length_beats=span.start_beat - out[-1].start_beat)
            else:
                span = replace(span, start_beat=cursor,
                               length_beats=span.length_beats + (span.start_beat - cursor))
        out.append(span)
        cursor = out[-1].end_beat
    if total_beats is not None and cursor < total_beats:
        out[-1] = replace(out[-1], length_beats=total_beats - out[-1].start_beat)
    return out


def mean_confidence(spans: list[GridSpan]) -> float:
    """Track-level confidence, weighted by how long each chord is heard.

    Duration-weighted rather than a flat average because a track with one
    doubtful two-beat chord among thirty confident ones is a confident track, and
    a flat mean would say otherwise.
    """
    if not spans:
        return 0.0
    total = sum(s.length_beats for s in spans)
    if total <= 0:
        return 0.0
    return sum(s.confidence * s.length_beats for s in spans) / total


def exact_ratio(spans: list[GridSpan]) -> float:
    """How much of the track survived normalization without losing information.

    A low ratio is the honest signal that the chart is not really "what was
    played" on this recording — a jazz track reduced to triads and dominant 7ths
    is a different song, and this is the number that says so.

    Computed in `model.build` and carried out on the sidecar as
    `analysis.exactRatio`. Now that the chart is the only render, this measures
    the grammar's ceiling and nothing else — which is the only thing it was ever
    supposed to mean.
    """
    if not spans:
        return 0.0
    total = sum(s.length_beats for s in spans)
    if total <= 0:
        return 0.0
    return sum(s.length_beats for s in spans if s.exact) / total


def process(spans: list[RawChordSpan], axis: BeatAxis, *,
            total_beats: int | None = None,
            fill: bool = True) -> list[GridSpan]:
    """The whole §5.4 chain, in order. One analysis, one chart.

    `total_beats` defaults to the axis's own length, which is what makes the
    chart cover the **whole recording**: without it the harmony stopped at the
    last chord the engine was confident about, and the bars after that were
    simply dropped (§5.4's hold rule applied to the end of the song, not just to
    the gaps inside it).

    `fill=False` stops before step 4 — the chords the engine actually reported,
    with the silences still silent. Nothing playable comes out of it (§18 has no
    rest, and a section of empty bars is dropped by the importer), and it has
    exactly one caller: `keyfinder.detect_key`, which is measuring *what the song
    plays* rather than *what the chart shows*. Holding a chord across a
    forty-second instrumental break triples its weight in a duration-weighted bag
    of chords, and the key then partly describes the arrangement's gaps.
    """
    quantized = quantize(spans, axis)
    merged = merge(quantized)
    trimmed = drop_short(merged)
    if not fill:
        return trimmed
    return hold_through_gaps(
        trimmed, total_beats=axis.total_beats if total_beats is None else total_beats
    )
