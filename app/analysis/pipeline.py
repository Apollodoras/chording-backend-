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
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..chords import DIFFICULTIES
from ..errors import (
    AnalysisError,
    FeatureDisabled,
    TempoUnreadable,
    VideoBlocked,
    VideoTooLong,
)
from ..lint import TEMPO_MAX, TEMPO_MIN, lint, lint_sync, repair
from ..payload import CompositionPayload
from ..sync import DownbeatRepair, TheoryReport, VideoSync
from . import model as song_model
from .compile import build_sync, compile_song
from .scratch import scratch
from .tuning import CONCERT_PITCH, Tuning
from . import tuning as tuning_probe
from .types import BeatGrid, EnergyCurve, EngineInfo, Onset, RawChordSpan, VideoMeta

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
    sync: VideoSync | None          # the sidecar itself; None when none was built
    low_confidence: bool
    engine_chords: str
    engine_beats: str
    analyzed_at: str
    duration_ms: int
    # §20's provenance, always present — on the outcome as well as on the
    # sidecar, because a song whose sidecar was withheld (§13.3) still has an
    # analysis worth reporting, and the benchmark still has to be able to read
    # what the theory layer did to it.
    theory: TheoryReport = field(default_factory=TheoryReport)
    # **Why** the song is low-confidence, when it is. Four different things set
    # that flag and they call for four different responses — a weak chord read
    # is not a broken bar grid is not a tempo an octave out — and until this
    # existed the only signal anywhere was a single boolean on the stored row.
    # "This song lost video sync and nobody can say what for" is not a state an
    # operator should have to be in. Not on the wire: the player is told the
    # sync is unavailable, which is all it can act on.
    low_confidence_reasons: tuple[str, ...] = ()
    # **Which difficulties the sidecar is actually true of.**
    #
    # The sidecar is one object — it describes the *recording*, so its anchors,
    # tempo and bar grid are the same whichever tier the player asked for. What is
    # per-tier is whether the tier's chart still *agrees* with it: `model.impose`
    # can drop a section from one tier, and `lint_sync` then rejects that pairing.
    #
    # That check used to withhold the sidecar from **every** tier on the first
    # failure, so an `easy` render coming out a section short cost `hard` and
    # `normal` their video sync too — and the log line named only the tier that
    # failed, so the two that were fine looked untouched. Video sync is the
    # product's differentiator, and it should not be all-or-nothing across renders
    # that are allowed to differ by design. The store is already keyed on
    # `(video_id, difficulty)`, which is exactly the granularity this needs.
    #
    # Empty when no sidecar was built at all (low confidence, §13.3).
    sync_tiers: frozenset[str] = frozenset()

    def sync_for(self, difficulty: str) -> VideoSync | None:
        """The sidecar to store against one difficulty — `None` when this tier's
        chart and the sidecar disagree."""
        return self.sync if difficulty in self.sync_tiers else None


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
    structure_probe=None,
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

        # BEFORE anything names a chord: what does this recording call A?
        # A constant-Q filter bank is laid out from A440, and a record that
        # is not at A440 puts every partial between two bins — which comes
        # out as the right progression in the wrong key, with high
        # confidence and nothing downstream able to see it. Measured once,
        # here, because it is a property of the recording and every stage
        # that transforms audio has to be told the same answer.
        pitch = tuning_probe.estimate(pcm, sample_rate)

        report("analyzing", 0.35)
        # Beats first, then chords, then quantize chords to the grid: "a rhythm
        # game needs chord changes that land ON beats — raw frame-level chord
        # output looks jittery and plays badly" (§5). Campfire isn't a rhythm
        # game, but a stroke lands on a stroke, so the requirement is the same.
        grid: BeatGrid = beat_tracker.track(pcm, sample_rate)
        report("analyzing", 0.6)
        raw: list[RawChordSpan] = chord_engine.analyze(pcm, sample_rate, tuning=pitch.correction)
        onsets: list[Onset] = onset_detector.detect(pcm, sample_rate) if onset_detector else []
        # §20.7 — a loudness envelope, the only evidence that can tell a chorus
        # from a verse. A scalar per hop, and the last thing to touch the audio.
        #
        # Failing here must NOT fail the analysis. Nothing the chart is built
        # from depends on this: without it every section is `Part N`, which §15
        # calls the honest answer when the segmentation cannot tell. Losing a
        # song because the loudness probe choked would be trading the whole
        # deliverable for a label, so the probe is allowed to fail and the
        # analysis carries on exactly as it does on a build with no probe at all.
        energy: EnergyCurve | None = None
        if structure_probe is not None:
            try:
                energy = structure_probe.probe(pcm, sample_rate)
            except Exception:
                log.warning("structure probe failed for %s — sections will be unnamed",
                            video_id, exc_info=True)

    # From here on there is no audio anywhere in the process.
    report("analyzing", 0.85)
    elapsed = time.monotonic() - started
    log.info("analysis of %s finished in %.1fs (%d chord spans, %d beats, "
             "tuning %+.0f cents%s)",
             video_id, elapsed, len(raw), len(grid.beats_ms), pitch.cents,
             " — CORRECTED" if pitch.ambiguous else "")

    return assemble(
        meta=meta, grid=grid, raw=raw, onsets=onsets, energy=energy, settings=settings,
        tuning=pitch,
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
    energy: EnergyCurve | None = None,
    tuning: Tuning = CONCERT_PITCH,
) -> AnalysisOutcome:
    """Features → songs. **Pure** — this is the half the tests drive directly."""
    analyzed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    duration_ms = int(round(meta.duration_s * 1000))

    # ONE musical model, built once from the richest tier: the meter is
    # reconciled against the harmony, ONE beat axis is laid (the chart quantizes
    # against it, the bars are sliced out of it, the onsets fold onto it and the
    # anchors are read off it — see `axis.py`), the form is found, the repeats
    # are voted into line, and one groove is extracted per repeat group.
    # Each difficulty below is a *render* of this, not another analysis (§20.6).
    model = song_model.build(
        grid=grid, raw=raw, onsets=onsets, energy=energy,
        vote=getattr(settings, "theory_consensus", True),
        weigh=getattr(settings, "theory_belief", True),
        consolidate=getattr(settings, "theory_vocabulary", True),
        correct_octave=getattr(settings, "theory_tempo_octave", False),
        form_canonical=getattr(settings, "theory_form", True),
    )
    if model is None or not grid.is_usable:
        raise AnalysisError(
            "That track’s rhythm wasn’t clear enough to chart — try a video with a steadier beat."
        )

    axis = model.axis
    beats = float(axis.bar_beats)
    tempo = model.meter.tempo
    key = model.key
    time_signature = model.meter.time_signature

    # §13.3, on the one diagnosis the analysis makes and used to swallow. A tempo
    # the analysis calls implausible is an octave error far more often than it is
    # a real reading, and the two ways that lands are worth telling apart:
    #
    #   outside the container's range   there is no song to ship. Say what was
    #                                   read, rather than the generic "that video
    #                                   didn't produce a song we could play" the
    #                                   lint would have raised three lines later.
    #   inside it, still implausible    the song plays, but the axis it plays on
    #                                   is the thing in doubt — so it ships
    #                                   low-confidence, which withholds the
    #                                   sidecar and tells the player.
    if not (TEMPO_MIN <= tempo <= TEMPO_MAX):
        log.warning("tempo %d bpm is outside %d–%d for %s (octave suspect: %s)",
                    tempo, TEMPO_MIN, TEMPO_MAX, meta.video_id,
                    model.meter.tempo_octave_suspect)
        raise TempoUnreadable(tempo, TEMPO_MIN, TEMPO_MAX)

    # `downbeats.unreliable` is the fourth way in, and it is the one §20.2a
    # added: a song whose bars disagreed with their own mode more often than the
    # repair's ceiling may genuinely not be in the meter we are charting it in.
    # The repair still ran — a self-consistent grid is worth having either way —
    # but the sidecar is a claim about *this recording's* timeline, and that is
    # the claim in doubt. §13.3's honest degradation, on new evidence.
    reasons: list[str] = []
    if model.confidence < settings.confidence_floor:
        reasons.append(f"chords {model.confidence:.2f} < floor {settings.confidence_floor:.2f}")
    if grid.confidence < settings.confidence_floor:
        reasons.append(f"beat grid {grid.confidence:.2f} < floor {settings.confidence_floor:.2f}")
    if model.meter.tempo_octave_suspect:
        reasons.append(f"tempo {tempo} may be an octave out")
    if model.meter.downbeats.unreliable:
        reasons.append(
            f"bars disagreed with the song's own {model.meter.downbeats.bar_beats}-beat "
            f"bar more than the repair's ceiling"
        )
    low_confidence = bool(reasons)
    if low_confidence:
        log.warning("%s ships low-confidence (no sidecar): %s", meta.video_id,
                    "; ".join(reasons))
    pattern_confidence: float | None = model.pattern_confidence
    total_bars = model.total_bars
    theory = TheoryReport(
        scale=key.scale,
        sections=len(model.sections),
        groups=len(model.groups),
        rewrittenBars=model.consensus.rewritten_bars,
        contestedBars=model.consensus.contested_bars,
        weighedBars=model.consensus.weighed_bars,
        snappedSpans=model.vocabulary.snapped_spans,
        absorbedIslands=model.vocabulary.absorbed_islands,
        canonicalBars=model.canon.canonical_bars,
        settledBars=model.canon.settled_bars,
        heldBeats=model.canon.held_beats,
        splitBars=model.canon.split_bars,
        phaseShift=model.meter.phase_shift,
        meterSource=model.meter.meter_source,
        tempoOctaveSuspect=model.meter.tempo_octave_suspect,
        tempoOctaveShift=model.meter.tempo_octave_shift,
        irregularBars=model.meter.downbeats.irregular_bars,
        downbeatsRepaired=DownbeatRepair(
            dropped=model.meter.downbeats.dropped,
            inserted=model.meter.downbeats.inserted,
        ),
        exactRatio=round(model.exact_ratio, 3),
        tuningCents=round(tuning.cents, 1),
        tuningAmbiguous=tuning.ambiguous,
    )

    songs: dict[str, dict] = {}
    for difficulty in DIFFICULTIES:
        sections = song_model.render(model, raw, difficulty)
        if not sections:
            continue

        payload = compile_song(
            video_id=meta.video_id,
            title=meta.title,
            sections=sections,
            patterns=model.patterns,
            key=key,
            tempo=tempo,
            time_signature=time_signature,
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
    sync_tiers: set[str] = set()
    if not low_confidence:
        sync = build_sync(
            video_id=meta.video_id,
            duration_ms=duration_ms,
            # The axis's downbeats, not the tracker's: `times_ms[k · bar_beats]`
            # IS the time of the chart's bar k, so the anchor and the bar it
            # addresses cannot disagree.
            downbeats_ms=axis.downbeats_ms,
            bar_beats=beats,
            # The model's meter, not the tracker's: if §20.2 arbitrated the
            # meter, the payload carries the arbitrated one and a sidecar naming
            # the other would put the anchors on a different bar grid than the
            # chart. They have to be the same string.
            time_signature=time_signature,
            # The **model's** tempo, not the tracker's. Same argument as the
            # `time_signature` above it: the payload carries the reconciled
            # number, the client derives one bar grid from it, and a sidecar
            # quoting a different tempo would describe a different grid than the
            # chart it is shipped beside.
            bpm=float(tempo),
            tempo_confidence=grid.confidence,
            engine_chords=str(chords_engine),
            engine_beats=str(beats_engine),
            analyzed_at=analyzed_at,
            low_confidence=low_confidence,
            pattern_confidence=pattern_confidence,
            total_bars=total_bars,
            analysis=theory,
        )
        # §13.3 — degrade honestly. If the anchors and the chart disagree, the
        # cursor would walk off the song, and a self-paced song that's right
        # beats a video-synced one that's wrong. Drop the sidecar, keep the song.
        #
        # Checked against EVERY difficulty, not just the reference: one sidecar is
        # returned alongside whichever tier the player asked for, so it has to be
        # true of the tier being served. The tiers share a bar grid and normally
        # agree — this catches the case where simplification shortened one of them.
        #
        # **Per tier, not all-or-nothing.** This used to `break` on the first
        # failure and withhold the sidecar everywhere, which punished two renders
        # for a third one's shape; `impose` is allowed to drop a section per tier,
        # so that case is by design rather than a symptom. Each tier now keeps or
        # loses video sync on its own evidence, and the log says which did what.
        for tier, wire in sorted(songs.items()):
            problems = lint_sync(CompositionPayload.model_validate(wire), sync)
            if problems:
                log.warning("sidecar withheld for %s (%s tier): %s",
                            meta.video_id, tier, problems)
            else:
                sync_tiers.add(tier)
        if not sync_tiers:
            # No tier accepted it, so there is no sidecar to report either. Keeps
            # "`sync is None`" meaning exactly what it always meant: this analysis
            # has no usable video sync.
            sync = None

    return AnalysisOutcome(
        meta=meta,
        songs=songs,
        sync=sync,
        low_confidence=low_confidence,
        engine_chords=str(chords_engine),
        engine_beats=str(beats_engine),
        analyzed_at=analyzed_at,
        duration_ms=duration_ms,
        theory=theory,
        low_confidence_reasons=tuple(reasons),
        sync_tiers=frozenset(sync_tiers),
    )
