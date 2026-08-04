"""The whole job, in order — §5's pipeline with §2's invariants around it.

    probe ──► gate (blocklist, length cap, kill switch)
          └─► scratch dir ──► decode ──► beats ──► chords ──► onsets
                                  │
                                  └─► rm -rf audio   (guaranteed, every path)
                                          │
              post-process ──► sections ──► patterns ──► compile ──► lint
                                          │
                                    { song, videoSync }

Two structural decisions worth stating, because they are what make this testable
at all:

**Everything after `decode` is pure.** The DSP stages hand back plain data
(`BeatGrid`, `RawChordSpan`, `Onset`) and the rest of the function is arithmetic
over it. So `analyze` can be driven end to end with fake engines — which is how
the test suite exercises quantization, structure, difficulty tiers and the §13.2
anchor invariant without any audio, any model weights, or any network.

**The gate runs before the fetch, always.** A blocked video, a too-long video and
a disabled feature are all decided from `probe` metadata. §2's blast-radius
argument is only true if the service never downloads what it wasn't allowed to
touch, and "check afterwards" would make it false.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from ..chords import DIFFICULTIES, NORMAL
from ..errors import AnalysisError, FeatureDisabled, VideoBlocked, VideoTooLong
from ..lint import lint, lint_sync, repair
from ..payload import CompositionPayload
from ..sync import VideoSync
from . import postprocess
from .axis import build_axis
from .compile import build_sync, compile_song
from .keyfinder import detect_key
from .scratch import scratch
from .strumming import ExtractedPattern, extract, fallback, fold_onsets
from .structure import Section, bars_from_spans, segment
from .types import BeatGrid, EngineInfo, Onset, RawChordSpan, VideoMeta

log = logging.getLogger("chords.pipeline")


class LintFailure(AnalysisError):
    """The compiled song would warn on import.

    §12.4 is "never return a payload that would warn", and Mo makes that
    structural by having no code path that returns an unlinted song. Same here:
    the only way out of `analyze` runs through the lint, and a failure is an
    error rather than a warning attached to a song we ship anyway.
    """

    def __init__(self, problems: list[str]):
        super().__init__("That video didn’t produce a song we could play.")
        self.problems = problems


@dataclass(frozen=True)
class AnalysisOutcome:
    """One analysis, at every difficulty, plus what the store needs to file it."""

    meta: VideoMeta
    songs: dict[str, dict]          # difficulty → CompositionPayload wire dict
    sync: VideoSync | None          # shared across difficulties; None when weak
    low_confidence: bool
    engine_chords: str
    engine_beats: str
    analyzed_at: str
    duration_ms: int


def gate(meta: VideoMeta, *, settings, store) -> None:
    """Everything that can refuse a video before a byte is fetched.

    Order matters only in what the player is told: blocked outranks too-long,
    because "we can't analyze this video" is the more complete answer.
    """
    if not settings.analysis_enabled:
        raise FeatureDisabled()
    if store.is_blocked(video_id=meta.video_id, channel_id=meta.channel_id):
        raise VideoBlocked()
    if meta.duration_s > settings.max_video_seconds:
        raise VideoTooLong(settings.max_video_seconds)


def analyze(
    *,
    video_id: str,
    settings,
    store,
    source,
    chord_engine,
    beat_tracker,
    onset_detector=None,
    progress=None,
) -> AnalysisOutcome:
    """Run one video end to end. Raises `AnalysisError` for anything a player
    should be told about; anything else is a bug and becomes a 500.

    `progress` is an optional `(status, fraction) -> None` the job row follows,
    so `GET /v1/analyze/{jobId}` can say `fetching` rather than a spinner.
    """
    report = progress or (lambda status, fraction: None)
    started = time.monotonic()

    meta = source.probe(video_id)
    gate(meta, settings=settings, store=store)

    report("fetching", 0.1)
    # The ONLY block in the service where audio exists. It is destroyed on every
    # exit path, including the exception ones — see scratch.py.
    with scratch(settings.scratch_root, label=video_id) as workdir:
        pcm, sample_rate = source.decode(video_id, workdir)

        report("analyzing", 0.35)
        # Beats first, then chords, then quantize chords to the grid: "a rhythm
        # game needs chord changes that land ON beats — raw frame-level chord
        # output looks jittery and plays badly" (§5). Campfire isn't a rhythm
        # game, but a stroke lands on a stroke, so the requirement is the same.
        grid: BeatGrid = beat_tracker.track(pcm, sample_rate)
        report("analyzing", 0.6)
        raw: list[RawChordSpan] = chord_engine.analyze(pcm, sample_rate)
        onsets: list[Onset] = onset_detector.detect(pcm, sample_rate) if onset_detector else []

    # From here on there is no audio anywhere in the process.
    report("analyzing", 0.85)
    elapsed = time.monotonic() - started
    log.info("analysis of %s finished in %.1fs (%d chord spans, %d beats)",
             video_id, elapsed, len(raw), len(grid.beats_ms))

    return assemble(
        meta=meta, grid=grid, raw=raw, onsets=onsets, settings=settings,
        chords_engine=EngineInfo(chord_engine.name, chord_engine.version),
        beats_engine=EngineInfo(beat_tracker.name, beat_tracker.version),
    )


def assemble(
    *,
    meta: VideoMeta,
    grid: BeatGrid,
    raw: list[RawChordSpan],
    onsets: list[Onset],
    settings,
    chords_engine: EngineInfo,
    beats_engine: EngineInfo,
) -> AnalysisOutcome:
    """Features → songs. **Pure** — this is the half the tests drive directly."""
    analyzed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    duration_ms = int(round(meta.duration_s * 1000))

    # ONE beat axis, built once and shared by everything downstream: the chart
    # quantizes against it, the bars are sliced out of it, the onsets fold onto
    # it, and the sidecar's anchors are read off it. Three modules used to derive
    # this independently and disagree about where beat 0 was — see `axis.py`.
    axis = build_axis(grid)
    if axis is None or not axis.is_usable or not grid.is_usable:
        raise AnalysisError(
            "That track’s rhythm wasn’t clear enough to chart — try a video with a steadier beat."
        )

    beats = float(axis.bar_beats)
    bar_beats = axis.bar_beats
    tempo = int(round(grid.bpm))

    # --- §5.4, once per tier -------------------------------------------------
    reference = postprocess.process(raw, axis, difficulty=NORMAL)
    if not reference:
        raise AnalysisError("No chords could be read from that video.")

    confidence = postprocess.mean_confidence(reference)
    low_confidence = confidence < settings.confidence_floor or grid.confidence < settings.confidence_floor

    key = detect_key(reference)

    songs: dict[str, dict] = {}
    total_bars = 0
    pattern_confidence: float | None = None

    for difficulty in DIFFICULTIES:
        spans = postprocess.process(raw, axis, difficulty=difficulty)
        if not spans:
            continue
        bars = bars_from_spans(spans, bar_beats)
        if not bars:
            continue
        sections = segment(bars)
        patterns = _patterns_for(
            sections, onsets=onsets, axis=axis, bar_beats=beats, tempo=tempo,
        )
        # The anchor list has to cover the LONGEST tier, since one sidecar serves
        # whichever the player asked for.
        total_bars = max(total_bars, sum(s.total_bars for s in sections))
        if difficulty == NORMAL:
            supported = [p.confidence for p in patterns.values() if not p.is_fallback]
            pattern_confidence = round(sum(supported) / len(supported), 3) if supported else 0.0

        payload = compile_song(
            video_id=meta.video_id,
            title=meta.title,
            sections=sections,
            patterns=patterns,
            key=key,
            tempo=tempo,
            time_signature=grid.time_signature,
            # One Library row per difficulty, so all three can coexist without
            # `import` upserting one over another (§12.5).
            difficulty=difficulty,
        )
        repair(payload)
        problems = lint(payload)
        if problems:
            log.warning("lint rejected %s@%s: %s", meta.video_id, difficulty, problems)
            raise LintFailure(problems)
        songs[difficulty] = payload.wire_dict()

    if not songs:
        raise AnalysisError("No chords could be read from that video.")

    # --- §13, the sidecar ----------------------------------------------------
    sync: VideoSync | None = None
    if not low_confidence:
        sync = build_sync(
            video_id=meta.video_id,
            duration_ms=duration_ms,
            # The axis's downbeats, not the tracker's: `times_ms[k · bar_beats]`
            # IS the time of the chart's bar k, so the anchor and the bar it
            # addresses cannot disagree.
            downbeats_ms=axis.downbeats_ms,
            bar_beats=beats,
            time_signature=grid.time_signature,
            bpm=grid.bpm,
            tempo_confidence=grid.confidence,
            engine_chords=str(chords_engine),
            engine_beats=str(beats_engine),
            analyzed_at=analyzed_at,
            low_confidence=low_confidence,
            pattern_confidence=pattern_confidence,
            total_bars=total_bars,
        )
        # §13.3 — degrade honestly. If the anchors and the chart disagree, the
        # cursor would walk off the song, and a self-paced song that's right
        # beats a video-synced one that's wrong. Drop the sidecar, keep the song.
        #
        # Checked against EVERY difficulty, not just the reference: one sidecar
        # is returned alongside whichever tier the player asked for, so it has to
        # be true of all of them. The tiers share a bar grid and normally agree —
        # this catches the case where simplification shortened one of them.
        for tier, wire in songs.items():
            problems = lint_sync(CompositionPayload.model_validate(wire), sync)
            if problems:
                log.warning("sidecar withheld for %s (%s tier): %s", meta.video_id, tier, problems)
                sync = None
                break

    return AnalysisOutcome(
        meta=meta,
        songs=songs,
        sync=sync,
        low_confidence=low_confidence,
        engine_chords=str(chords_engine),
        engine_beats=str(beats_engine),
        analyzed_at=analyzed_at,
        duration_ms=duration_ms,
    )


def _patterns_for(sections: list[Section], *, onsets: list[Onset], axis,
                  bar_beats: float, tempo: int) -> dict[int, ExtractedPattern]:
    """One pattern per section (§14).

    With no onset detector at all, every section gets the quarter-note fallback —
    which is exactly the intended behaviour: the app requires a pattern, and a
    boring one that plays is worth more than no song.
    """
    out: dict[int, ExtractedPattern] = {}
    for index, section in enumerate(sections):
        name = f"{section.name or section.kind.title()} strum"
        time_signature = f"{axis.bar_beats}/4"
        if not onsets:
            out[index] = fallback(bar_beats=bar_beats, tempo=tempo, name=name,
                                  time_signature=time_signature)
            continue
        first_beat = section.start_bar * bar_beats
        last_beat = first_beat + section.total_bars * bar_beats
        # Folded onto the same axis the chart uses, so a stroke the extractor
        # calls "the downbeat" is the beat the chart calls bar-start too.
        folded = fold_onsets(onsets, axis, bar_beats=bar_beats,
                             first_beat=first_beat, last_beat=last_beat)
        out[index] = extract(folded, bar_beats=bar_beats, bars=section.total_bars,
                             tempo=tempo, name=name, time_signature=time_signature)
    return out
