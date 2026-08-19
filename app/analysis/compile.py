"""Sections + patterns + beat grid → the §13.1 envelope: a song and its sidecar.

This is where the analysis stops being analysis and becomes **the deliverable**
(§12): a ``CompositionPayload`` v2 the app already imports, plus the ``videoSync``
that lets a clock drive the cursor. Everything upstream of here is free to think
in chords and beats; everything from here on has to think in what
``ComposerService.import`` will accept.

Two encodings per section, and the choice is not cosmetic:

- **Flat** (``chordNames`` + ``beatsPerChord`` + ``patternID``) when the section's
  harmonic rhythm is uniform — every chord the same length, and that length
  dividing the bar. This is the form the app renders best and the form a shared
  song carries, and it is what makes ``repeats`` usable.
- **Bars** (``bars[].chordSpans[]``) otherwise. Fully general — mid-bar changes,
  a chord held across a barline — and §12.1 warns you will need it constantly on
  real recordings. Note the app **ignores ``repeats`` in bars mode**
  (``compileBars`` never sees it), so repeats are expanded here and ``repeats``
  is emitted as 1. Getting that wrong is a song that plays a quarter of its
  length with no error anywhere.

The one rule that ties this file to `app/sync.py`: **no ``tempoOverride`` and no
``timeSignatureOverride``, ever**, when a sidecar is being emitted. The client
derives its bar grid from a single song-level tempo, so an override silently
detaches the beat axis the anchors address. `lint_sync` enforces it; this module
simply never emits one.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..chords import prefers_flats, render
from ..payload import (
    Arrangement,
    Bar,
    BarRhythm,
    ChordSpan,
    CompositionPayload,
    PatternPayload,
    SongSection,
    bar_beats as parse_bar_beats,
    derived_uuid,
    song_id,
    total_beats,
)
from ..sync import Confidence, EngineVersions, TheoryReport, VideoSync, anchors_for
from .keyfinder import DetectedKey
from .strumming import ExtractedPattern
from .structure import BarChord, Section

_EPS = 1e-6


@dataclass(frozen=True)
class CompiledSong:
    """What one analysis compiles to."""

    payload: CompositionPayload
    sync: VideoSync | None


def compile_song(
    *,
    video_id: str,
    title: str,
    sections: list[Section],
    patterns: dict[str, ExtractedPattern],
    key: DetectedKey,
    tempo: int,
    time_signature: str,
) -> CompositionPayload:
    """Sections → a `CompositionPayload` v2.

    `patterns` is keyed by **repeat-group letter** (§20.3), not by section
    index. That is the musically correct key and it is also the one that makes
    the encoding compact: every occurrence of a verse is the same music, so it
    gets the same groove, and the content-addressed pattern id means the
    payload carries one pattern object rather than one per occurrence.
    """
    beats = parse_bar_beats(time_signature) or 4.0
    flats = prefers_flats(key.tonic, key.mode)

    payload_sections: list[SongSection] = []
    embedded: dict[str, PatternPayload] = {}

    song = song_id(video_id)
    for index, section in enumerate(sections):
        extracted = patterns.get(section.group)
        if extracted is None:
            # Unreachable: `model._patterns` emits one pattern per repeat group
            # and every section carries a group, with the quarter-note fallback
            # covering the case where there is nothing to extract from. Which is
            # exactly why it raises. Skipping the section instead — what this did
            # — compiles a song that is *short*: bars silently missing, `repeats`
            # still claiming the full count, the sidecar's anchors addressing
            # bars that no longer exist, and nothing anywhere to say so, because
            # both lints check the song against itself and a shorter song is
            # perfectly self-consistent. A future regression upstream should
            # break loudly here rather than ship a truncated chart.
            raise ValueError(
                f"section {index} belongs to repeat group {section.group!r}, "
                f"which has no pattern (have: {sorted(patterns)}) — every group "
                f"must carry one, including the fallback"
            )
        pattern = extracted.pattern
        embedded[pattern.id] = pattern
        payload_sections.append(
            _section(section, pattern_id=pattern.id, bar_beats=beats, flats=flats,
                     seed=f"{song}#{index}")
        )

    payload = CompositionPayload(
        version=2,
        id=song,
        title=title,
        tonic=key.tonic,
        mode=key.mode,
        tempo=tempo,
        timeSignature=time_signature,
        arrangement=Arrangement(sections=payload_sections),
        # Every referenced pattern MUST be embedded or the section is silently
        # dropped and the song plays short with no error (§12.3). There is no
        # bundled pattern catalog on the client any more.
        patterns=list(embedded.values()),
        progressions=[],
    )
    # The flat v1 summary (`chordNames`/`patternID`/`beatsPerChord`/`repeats`)
    # mirrors section 1 and is filled by `lint.repair`, which the pipeline runs
    # next — one implementation of that rule, shared with Mo.
    return payload


def _section(section: Section, *, pattern_id: str, bar_beats: float, flats: bool,
             seed: str) -> SongSection:
    flat = _as_flat(section, bar_beats=bar_beats, flats=flats)
    if flat is not None:
        chord_names, beats_per_chord = flat
        return SongSection(
            id=derived_uuid(seed, "section"),
            name=section.name,
            kindRaw=section.kind,
            chordNames=chord_names,
            patternID=pattern_id,
            beatsPerChord=beats_per_chord,
            repeats=section.repeats,
            tempoOverride=None,
            timeSignatureOverride=None,
            bars=None,
        )

    # Bars mode. `repeats` is expanded into explicit bars because the app's
    # `compileBars` never sees it — leaving repeats > 1 here plays one pass.
    expanded: list[list[BarChord]] = list(section.bars) * section.repeats
    return SongSection(
        id=derived_uuid(seed, "section"),
        name=section.name,
        kindRaw=section.kind,
        # Dead for compilation in bars mode (`compileBars` reads the bars, and
        # `importReport` takes its chord names from them too) — filled anyway,
        # with the first bar's chords, so that a section is never one field away
        # from the "empty chordNames ⇒ silently dropped" failure if a later
        # change drops `bars`.
        chordNames=[render(c.root_pc, c.quality, flats=flats) for c in expanded[0]] if expanded else [],
        patternID=pattern_id,
        beatsPerChord=int(bar_beats),
        repeats=1,
        tempoOverride=None,
        timeSignatureOverride=None,
        bars=[_bar(bar, flats=flats, seed=f"{seed}#bar{index}")
              for index, bar in enumerate(expanded)],
    )


def _as_flat(section: Section, *, bar_beats: float, flats: bool) -> tuple[list[str], int] | None:
    """The flat recipe for this section, or None when flat mode would play it wrong.

    **Flat mode requires one chord per bar**, and the reason is in the app's
    compiler rather than in taste. `Compiler.compile` tiles the strumming pattern
    **per chord slot**, restarting it at every chord: with `beatsPerChord` equal
    to the pattern's bar the groove plays once per chord, but with a
    `beatsPerChord` of 2 in a 4/4 bar the pattern is *truncated to its first two
    beats and restarted mid-bar* — silently, with no warning anywhere. So a
    mid-bar chord change cannot be said in flat mode at all; it needs `bars`,
    whose compiler (`notesForBar`) lays the section's pattern over the bar once
    and lets each stroke sound whichever chord is active at its beat.

    A chord held across bars stays flat: it is simply repeated, one entry per
    bar, which is the same encoding a shared song uses.
    """
    if not section.bars:
        return None

    names: list[str] = []
    for bar in section.bars:
        if len(bar) != 1:
            return None   # a mid-bar change — bars mode
        chord = bar[0]
        if abs(chord.start_beat) > _EPS or abs(chord.length_beats - bar_beats) > _EPS:
            return None   # doesn't fill its bar — bars mode
        names.append(render(chord.root_pc, chord.quality, flats=flats))

    if abs(bar_beats - round(bar_beats)) > _EPS:
        # A meter whose bar isn't a whole number of quarter-note beats can't be a
        # `beatsPerChord`, which is an Int on the wire.
        return None
    return names, int(round(bar_beats))


def _bar(bar: list[BarChord], *, flats: bool, seed: str) -> Bar:
    return Bar(
        id=derived_uuid(seed),
        chordSpans=[
            ChordSpan(
                chordName=render(chord.root_pc, chord.quality, flats=flats),
                startBeat=chord.start_beat,
                lengthBeats=chord.length_beats,
            )
            for chord in bar
        ],
        # Every bar inherits the section's pattern. Per-bar variation exists in
        # the model (`bars[].rhythm.custom`) but §14 is explicit that it "should
        # be the exception" — a per-bar transcription of a strummed acoustic is
        # less playable, and less true to the song, than the groove everyone
        # actually plays.
        rhythm=BarRhythm(kind="inherit"),
    )


def _within_duration(downbeats_ms: list[int], duration_ms: int, bpm: float) -> list[int]:
    """Keep anchors inside the recording, pinning a rounding overshoot to the end.

    Truncating by bar count is not the same as truncating by the clock, and the
    difference is measured in milliseconds: a tracker working from decoded audio
    can place the final downbeat a few ms after the duration YouTube's metadata
    reports. `lint_sync` then rejects the sidecar — correctly, by its own rule —
    and the song loses video sync over a **6 ms** disagreement between two
    different measurements of the same recording. (Observed: five of the
    benchmark's synthetic tracks, all by 6–20 ms.)

    So an overshoot within one beat is treated as what it is — the two clocks
    rounding differently — and pinned to the end. Anything beyond that is a real
    disagreement about where the song ends, and those anchors are dropped:
    inventing a position for them is how a cursor walks off a chart.
    """
    if not downbeats_ms:
        return downbeats_ms
    tolerance = int(60_000 / bpm) if bpm and bpm > 0 else 500

    kept: list[int] = []
    for time_ms in downbeats_ms:
        if time_ms <= duration_ms:
            kept.append(time_ms)
        elif time_ms - duration_ms <= tolerance and (not kept or kept[-1] < duration_ms):
            kept.append(duration_ms)
            break
        else:
            break
    return kept


def build_sync(
    *,
    video_id: str,
    duration_ms: int,
    downbeats_ms: list[int],
    bar_beats: float,
    time_signature: str,
    bpm: float,
    tempo_confidence: float,
    engine_chords: str,
    engine_beats: str,
    analyzed_at: str,
    low_confidence: bool,
    pattern_confidence: float | None,
    total_bars: int,
    offset_ms: int | None = 0,
    analysis: TheoryReport | None = None,
) -> VideoSync:
    """The §13 sidecar for a compiled song.

    Anchored one per downbeat (§13.2: "every downbeat is ideal — cheap, and it
    makes the interpolation exact rather than approximate"), and truncated to the
    song's own length: the recording may run past the last bar the chart
    covers — an outro fade, applause, a tacked-on second song — and anchors past
    the end would address bars that don't exist.
    """
    usable = downbeats_ms[:max(1, total_bars + 1)] if total_bars else downbeats_ms
    usable = _within_duration(usable, duration_ms, bpm)
    return VideoSync(
        videoId=video_id,
        durationMs=duration_ms,
        offsetMs=offset_ms,
        # Clamped, not asserted. Both are engine-reported and the container's
        # domain is 0…1; a tracker handing back 1.02 is a bug in the tracker, and
        # the sidecar's job is not to lose the whole beat map over it. `lint_sync`
        # still calls an out-of-range value fatal, which is now a statement about
        # *this* function having failed rather than about the recording.
        tempo=Confidence(bpm=round(bpm, 2), confidence=round(_unit(tempo_confidence), 3)),
        timeSignature=time_signature,
        lowConfidence=low_confidence,
        patternConfidence=None if pattern_confidence is None else _unit(pattern_confidence),
        beatAnchors=anchors_for(usable, bar_beats=bar_beats),
        engine=EngineVersions(chords=engine_chords, beats=engine_beats),
        analyzedAt=analyzed_at,
        analysis=analysis,
    )


def _unit(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def fit_sync(sync: VideoSync, payload: CompositionPayload) -> VideoSync:
    """The same sidecar, trimmed to what **this** chart actually covers.

    The anchors are measured on the **recording** and the chart is measured in
    bars, and the two can disagree about where the song ends: `model.impose` may
    expand or merge across a barline, so the chart can come out shorter than the
    grid the anchors were built from. The anchors past its last beat then address
    bars that do not exist, and the cursor walks off the end of the chart.

    That used to cost the song its sidecar entirely — `lint_sync` reported the
    overrun and §13.3 withheld. It is a trim, so it is now trimmed: drop the
    anchors beyond the chart's final barline and keep the map for everything the
    chart does cover. Nothing else about the sidecar is touched; a sidecar that
    already fits is returned unchanged rather than copied.

    The last anchor kept is the one *at* the song's final beat — it marks the
    closing boundary of the last bar, which is why an N-bar song ships N+1
    anchors and why the bound is inclusive.
    """
    song_length = total_beats(payload)
    if song_length <= 0:
        return sync
    kept = [a for a in sync.beatAnchors if a.songBeat <= song_length + _FIT_EPS]
    if len(kept) == len(sync.beatAnchors):
        return sync
    # Fewer than two anchors is not a map, and `lint_sync` says so as a fatal.
    # Handing the untrimmed sidecar back instead would trade one reported problem
    # for a worse unreported one, so the trim stands and the lint decides.
    return sync.model_copy(update={"beatAnchors": kept})


_FIT_EPS = 1e-6
