"""§8 step 2's harness: benchmark the candidate engines and report.

What it scores:

- **beat trackers** — F-measure against ground-truth beats and downbeats at the
  standard ±70 ms tolerance, plus tempo error. Downbeats are scored separately
  and matter more here than beats do: §13.2's anchors are downbeats, and a
  tracker that finds the pulse but not the "one" produces a sidecar that walks
  the cursor off the song.
- **chord engines** — per-beat accuracy after normalization into the app's
  grammar (§12.2), which is the only accuracy that means anything downstream: an
  engine that nails `Cmaj9` and one that says `Cmaj7` score identically, because
  the app plays the same chord for both. Root-only accuracy is reported beside
  it, because the gap between them is a different problem from getting the chord
  wrong — it is the engine hearing the right harmony and the wrong quality.
- **the whole pipeline** — whether the emitted song lints clean, whether the
  sidecar survives `lint_sync`, and **what the player actually sees**. An engine
  pairing that produces beautiful numbers and an unusable song has not won
  anything.

  That last one (`delivered`) is the column to read, and it is deliberately
  separate from the chord-engine score above. Everything else here measures a
  *component* against ground truth; `delivered` measures the **deliverable** —
  it reconstructs "what chord is on screen at video millisecond t" from
  `(song, videoSync)` the way the client does, after quantization, structure and
  `repeats` have all had their say. The two can diverge a long way, and when they
  do it is this one that describes the product.

Two corpora, reported separately and never averaged together:

- the **synthetic** specimens from `synth.py`, with exact ground truth, which
  prove the plumbing and nothing else;
- the **real** tracks from `fetch_corpus.py` — Isophonics annotations against
  the recordings — which are the only evidence about a dense real mix, and the
  only numbers the engine choice should turn on.

Usage:
    python bench/synth.py                              # synthetic set
    python bench/fetch_corpus.py --annotations …       # real set
    python bench/run_bench.py                          # score what is registered
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import time
import wave
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.analysis import engines, postprocess  # noqa: E402
from app.analysis import tuning as tuning_probe  # noqa: E402
from app.analysis.axis import build_axis  # noqa: E402
from app.analysis.downbeats import modal_bar_beats  # noqa: E402
from app.analysis.downbeats import repair as repair_downbeats  # noqa: E402
from app.analysis.meter import reconcile  # noqa: E402
from app.analysis.pipeline import assemble  # noqa: E402
from app.analysis.strumming import extract, fold_onsets  # noqa: E402
from app.analysis.types import BeatGrid, EngineInfo, RawChordSpan, VideoMeta  # noqa: E402
from app.chords import (  # noqa: E402
    MAJOR,
    MINOR,
    normalize,
    prefers_flats,
    render,
)
from app.config import Settings  # noqa: E402
from app.lint import lint, lint_sync  # noqa: E402
from app.payload import CompositionPayload  # noqa: E402
from app.sync import chord_at_video_ms  # noqa: E402

AUDIO = ROOT / "bench" / "audio"

# Sampling step for the delivered-accuracy metric. Fine enough that a one-beat
# phase error cannot hide between samples at any tempo this service accepts.
FRAME_STEP_MS = 50

# The MIREX-standard beat-tracking tolerance. Not arbitrary: it is roughly the
# window inside which a listener hears two attacks as simultaneous.
TOLERANCE_MS = 70


@dataclass
class Case:
    name: str
    pcm: object
    sample_rate: int
    truth: dict

    @property
    def is_real(self) -> bool:
        return bool(self.truth.get("source"))


def load_cases() -> list[Case]:
    cases: list[Case] = []
    for wav_path in sorted(AUDIO.glob("*.wav")):
        truth_path = AUDIO / f"{wav_path.stem}.truth.json"
        if not truth_path.exists():
            print(f"! {wav_path.name} has no ground truth — skipped")
            continue
        pcm, rate = read_wav(wav_path)
        cases.append(Case(wav_path.stem, pcm, rate, json.loads(truth_path.read_text())))
    return cases


def read_wav(path: Path):
    """Mono float PCM. Uses numpy when it is installed (the worker image has it),
    and a plain list otherwise, so the harness runs before the audio extra
    does."""
    import struct

    with wave.open(str(path)) as source:
        rate = source.getframerate()
        frames = source.readframes(source.getnframes())
    samples = struct.unpack(f"<{len(frames) // 2}h", frames)
    try:
        import numpy as np

        return np.asarray(samples, dtype="float32") / 32768.0, rate
    except ImportError:
        return [s / 32768.0 for s in samples], rate


# --- memoized engine runs ----------------------------------------------------
#
# Every stage below wants the same engine's output on the same track, and the
# end-to-end matrix wants each one once per *pairing*. Recomputing was costing
# more than everything else here put together (BTC and beat_this are seconds per
# track, times four pairings, times fifteen tracks), so each run happens once.

_CHORD_RUNS: dict[tuple[str, str], tuple[list[RawChordSpan], float]] = {}
_BEAT_RUNS: dict[tuple[str, str], tuple[BeatGrid, float]] = {}


def chords_for(name: str, case: Case) -> tuple[list[RawChordSpan], float]:
    key = (name, case.name)
    if key not in _CHORD_RUNS:
        engine = engines._CHORD_ENGINES[name]()
        started = time.monotonic()
        # The recording's own pitch reference, measured the same way the
        # pipeline measures it. Without this the bench scores every real track as
        # though it were at A440 — which is exactly the assumption that put 82%
        # of Mary Jane's Last Dance a semitone out, and a benchmark that cannot
        # exercise the fix cannot show it working or catch it regressing.
        spans = engine.analyze(case.pcm, case.sample_rate,
                               tuning=tuning_probe.estimate(case.pcm, case.sample_rate).correction)
        _CHORD_RUNS[key] = (spans, time.monotonic() - started)
    return _CHORD_RUNS[key]


def grid_for(name: str, case: Case) -> tuple[BeatGrid, float]:
    key = (name, case.name)
    if key not in _BEAT_RUNS:
        tracker = engines._BEAT_TRACKERS[name]()
        started = time.monotonic()
        grid = tracker.track(case.pcm, case.sample_rate)
        _BEAT_RUNS[key] = (grid, time.monotonic() - started)
    return _BEAT_RUNS[key]


# --- scoring ----------------------------------------------------------------

def f_measure(detected: list[int], expected: list[int], tolerance_ms: int = TOLERANCE_MS):
    """Standard beat-tracking F-measure: each expected time may be matched once."""
    if not expected:
        return 0.0, 0.0, 0.0
    unmatched = sorted(detected)
    hits = 0
    for target in expected:
        for index, candidate in enumerate(unmatched):
            if abs(candidate - target) <= tolerance_ms:
                hits += 1
                unmatched.pop(index)
                break
    precision = hits / len(detected) if detected else 0.0
    recall = hits / len(expected)
    f = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return f, precision, recall


def _chord_score(spans: list[RawChordSpan], truth: dict, *, root_only: bool) -> float:
    """Share of the track (by time) whose chord we would end up **playing**
    correctly — i.e. compared after normalization into the app's grammar.

    With `root_only`, quality is ignored. The gap between the two is the engine
    hearing the right harmony and the wrong colour, which post-processing can
    sometimes recover and a wrong root never can.
    """
    expected = [(c["startMs"], c["endMs"], c["name"]) for c in truth["chords"]]
    total_ms = sum(end - start for start, end, _ in expected)
    if not total_ms:
        return 0.0

    correct = 0
    for start, end, name in expected:
        target = normalize(name)
        if target is None:            # `N`/`X` — nothing to be right about
            continue
        for span in spans:
            overlap = min(end, span.end_ms) - max(start, span.start_ms)
            if overlap <= 0:
                continue
            parsed = normalize(span.label)
            if parsed is None:
                continue
            if root_only:
                hit = parsed[0] == target[0]
            else:
                hit = render(parsed[0], parsed[1]) == render(target[0], target[1])
            if hit:
                correct += overlap
    return correct / total_ms


def chord_accuracy(spans: list[RawChordSpan], truth: dict) -> float:
    return _chord_score(spans, truth, root_only=False)


def root_accuracy(spans: list[RawChordSpan], truth: dict) -> float:
    return _chord_score(spans, truth, root_only=True)


def delivered_accuracy(payload: CompositionPayload, sync, truth: dict) -> float:
    """Share of the track whose chord the **player actually sees** correctly.

    Everything above scores an *engine*: raw spans, straight against ground
    truth, before quantization, structure and `repeats` have touched them. This scores the *deliverable* — it reconstructs "what chord is
    on screen at video millisecond t" from `(song, videoSync)` exactly as the
    client does (`app/sync.py`) and compares that.

    The two numbers answer different questions and can diverge a long way. A
    perfect engine still scores badly here if the chart is laid onto the wrong
    beat axis, and that failure is invisible to `lint`, to `lint_sync` and to
    every other number this harness prints — all of which check the song against
    itself rather than against the recording.

    Returns NaN when there is no sidecar: a self-paced song has no map from video
    time to song beat, so the question doesn't apply and averaging a zero in
    would libel a pairing that had no anchors to be graded on.
    """
    if sync is None or not sync.beatAnchors:
        return float("nan")

    flats = prefers_flats(payload.tonic, payload.mode)
    total_ms = correct_ms = 0
    for chord in truth["chords"]:
        target = normalize(chord["name"])
        if target is None:            # `N`/`X` — nothing to be right about
            continue
        expected = render(target[0], target[1], flats=flats)
        start, end = int(chord["startMs"]), int(chord["endMs"])
        for t in range(start, end, FRAME_STEP_MS):
            total_ms += FRAME_STEP_MS
            if chord_at_video_ms(payload, sync, t) == expected:
                correct_ms += FRAME_STEP_MS
    return correct_ms / total_ms if total_ms else float("nan")


def grid_from(truth: dict) -> BeatGrid:
    return BeatGrid(
        beats_ms=truth["beats_ms"], downbeats_ms=truth["downbeats_ms"],
        bpm=float(truth["tempo"]), confidence=1.0,
        time_signature=truth["time_signature"],
    )


# --- reporting ---------------------------------------------------------------

@dataclass
class Tally:
    """Per-engine scores, kept split by corpus so they are never averaged."""

    real: dict[str, list[float]] = field(default_factory=dict)
    synthetic: dict[str, list[float]] = field(default_factory=dict)

    def add(self, case: Case, **metrics: float) -> None:
        bucket = self.real if case.is_real else self.synthetic
        for metric, value in metrics.items():
            bucket.setdefault(metric, []).append(value)

    def mean(self, corpus: str, metric: str) -> float:
        values = (self.real if corpus == "real" else self.synthetic).get(metric, [])
        return statistics.mean(values) if values else float("nan")


def _split(cases: list[Case]) -> tuple[list[Case], list[Case]]:
    return [c for c in cases if c.is_real], [c for c in cases if not c.is_real]


# --- the runs ---------------------------------------------------------------

def bar_regularity(grid: BeatGrid) -> tuple[float, float]:
    """Share of bars that are the song's own modal length — before, after repair.

    The number nothing measured until the beat audit, and the one the player was
    complaining about: on the stored catalog it ran from 58% to 92%, while every
    gate in the repo was green. `after` is what the chart is actually built on,
    since `meter.reconcile` runs the repair before `build_axis` sees the grid.
    """
    def share(g: BeatGrid) -> float:
        mode = modal_bar_beats(g.beats_ms, g.downbeats_ms)
        if mode is None:
            return 0.0
        beats = sorted({int(t) for t in g.beats_ms})
        downbeats = sorted({int(t) for t in g.downbeats_ms})
        counts = [sum(1 for t in beats if start <= t < end)
                  for start, end in zip(downbeats, downbeats[1:])]
        return counts.count(mode) / len(counts) if counts else 0.0

    repaired, _ = repair_downbeats(grid)
    return share(grid), share(repaired)


def bench_beats(cases: list[Case]) -> dict[str, Tally]:
    names = sorted(engines._BEAT_TRACKERS)
    if not names:
        print("no beat trackers registered — see app/analysis/engines.py\n")
        return {}
    print("BEAT TRACKERS")
    # `bars` is the column the beat audit added, and it is a different question
    # from downbeat F: F asks whether each downbeat is near a true one, and a
    # tracker can score well on it while emitting bars of wildly different
    # lengths — an extra downbeat one beat into a real bar costs F almost
    # nothing and costs the player a tempo change. This is the share of bars
    # that are the song's own modal length, before the repair and after it.
    print(f"{'engine':<12}{'track':<22}{'beat F':>8}{'downbeat F':>12}"
          f"{'bpm err':>9}{'meter':>8}{'bars':>13}{'sec':>7}")
    tallies = {name: Tally() for name in names}
    for name in names:
        for case in cases:
            grid, elapsed = grid_for(name, case)
            beat_f, _, _ = f_measure(grid.beats_ms, case.truth["beats_ms"])
            down_f, _, _ = f_measure(grid.downbeats_ms, case.truth["downbeats_ms"])
            bpm_error = abs(grid.bpm - case.truth["tempo"])
            meter_ok = grid.time_signature == case.truth["time_signature"]
            before, after = bar_regularity(grid)
            tallies[name].add(case, beat_f=beat_f, down_f=down_f,
                              bpm_err=bpm_error, meter=float(meter_ok),
                              bars_before=before, bars_after=after,
                              seconds=elapsed)
            print(f"{name:<12}{case.name:<22}{beat_f:>8.3f}{down_f:>12.3f}"
                  f"{bpm_error:>9.1f}{('ok' if meter_ok else grid.time_signature):>8}"
                  f"{f'{before:.2f}→{after:.2f}':>13}{elapsed:>7.1f}")
    print()
    return tallies


def bench_chords(cases: list[Case]) -> dict[str, Tally]:
    names = sorted(engines._CHORD_ENGINES)
    if not names:
        print("no chord engines registered — see app/analysis/engines.py\n")
        return {}
    print("CHORD ENGINES  (accuracy measured AFTER normalization — §12.2)")
    print(f"{'engine':<12}{'track':<22}{'accuracy':>10}{'root only':>11}"
          f"{'spans':>8}{'sec':>7}")
    tallies = {name: Tally() for name in names}
    for name in names:
        for case in cases:
            spans, elapsed = chords_for(name, case)
            accuracy = chord_accuracy(spans, case.truth)
            roots = root_accuracy(spans, case.truth)
            tallies[name].add(case, accuracy=accuracy, root=roots, seconds=elapsed)
            print(f"{name:<12}{case.name:<22}{accuracy:>10.3f}{roots:>11.3f}"
                  f"{len(spans):>8}{elapsed:>7.1f}")
    print()
    return tallies


def bench_pipeline(cases: list[Case]) -> None:
    """The number that actually decides: does the pairing produce a song the app
    will play, a sidecar that agrees with it, **and the right chord on screen**?

    `clean`/`synced` are self-consistency counts — they say the song is
    well-formed, not that it is true. `delivered` is the accuracy the player
    experiences, and it is the column to read: a pairing can be 15/15 clean,
    15/15 synced, and still be showing the wrong chord for most of every song.
    """
    chord_names = sorted(engines._CHORD_ENGINES)
    beat_names = sorted(engines._BEAT_TRACKERS)
    if not (chord_names and beat_names):
        return

    settings = Settings(scratch_root="/tmp/chords-scratch")
    print("END TO END  (does the pairing produce a song the app will play — and play right?)")
    print(f"{'pairing':<24}{'clean':>7}{'synced':>8}{'delivered':>11}{'failed':>8}  notes")
    for chord_name in chord_names:
        for beat_name in beat_names:
            clean = synced = failed = 0
            delivered: list[float] = []
            reasons: dict[str, int] = {}
            for case in cases:
                grid, _ = grid_for(beat_name, case)
                raw, _ = chords_for(chord_name, case)
                meta = VideoMeta(video_id="bench0000000", title=case.name,
                                 duration_s=case.truth["duration_ms"] / 1000.0)
                try:
                    outcome = assemble(meta=meta, grid=grid, raw=raw, onsets=[],
                                       settings=settings,
                                       chords_engine=EngineInfo(chord_name, "bench"),
                                       beats_engine=EngineInfo(beat_name, "bench"))
                except Exception as exc:
                    failed += 1
                    reasons[type(exc).__name__] = reasons.get(type(exc).__name__, 0) + 1
                    continue
                payload = CompositionPayload.model_validate(outcome.song)
                if not lint(payload):
                    clean += 1
                # Counts a *spotless* sidecar — advisory problems and fatal ones
                # both fail it — because that is what this column has always
                # measured, and the service shipping the advisory ones is a
                # separate decision from the bench reporting them.
                if outcome.sync is not None and not lint_sync(payload, outcome.sync):
                    synced += 1
                # One chart, scored against the whole grammar. This used to read
                # `hard` here and `normal` above, because `normal` folded
                # diminished and augmented onto their nearest playable triad and
                # scoring *that* against a truth containing those chords charged
                # the pipeline for a reduction it had been asked to make —
                # Michelle read 0.812 at `normal` and 0.952 at `hard` for exactly
                # that reason. With the tiers gone, what the chart loses is
                # pipeline error and nothing else.
                accuracy = delivered_accuracy(payload, outcome.sync, case.truth)
                if accuracy == accuracy:      # not NaN — a sidecar was emitted
                    delivered.append(accuracy)
            note = ", ".join(f"{k}×{v}" for k, v in sorted(reasons.items())) or "-"
            mean_delivered = f"{statistics.mean(delivered):.3f}" if delivered else "-"
            print(f"{chord_name + '+' + beat_name:<24}{clean:>4}/{len(cases):<2}"
                  f"{synced:>5}/{len(cases):<2}{mean_delivered:>11}{failed:>8}  {note}")
    print()


def truth_spans(truth: dict) -> list[RawChordSpan]:
    """Ground truth, shaped as an engine's output — a *perfect* engine's.

    Confidence 1.0 on every span, which is not a detail: §20.4's third gate
    forbids the consensus vote from overruling a bar that was believed as much
    as the winner, so a perfect engine makes the whole layer a provable no-op.
    That is what the truth-as-engines run below is checking.
    """
    return [RawChordSpan(start_ms=int(c["startMs"]), end_ms=int(c["endMs"]),
                         label=c["name"], confidence=1.0)
            for c in truth["chords"]]


def _analyze(case: Case, grid: BeatGrid, raw: list[RawChordSpan], *,
             vote: bool, vocab: bool = False):
    settings = Settings(scratch_root="/tmp/chords-scratch", theory_consensus=vote,
                        theory_vocabulary=vocab)
    meta = VideoMeta(video_id="bench0000000", title=case.name,
                     duration_s=case.truth["duration_ms"] / 1000.0)
    return assemble(meta=meta, grid=grid, raw=raw, onsets=[], settings=settings,
                    chords_engine=EngineInfo("bench", "1"),
                    beats_engine=EngineInfo("bench", "1"))


def _delivered(outcome, truth: dict) -> float:
    return delivered_accuracy(
        CompositionPayload.model_validate(outcome.song), outcome.sync, truth)


def bench_theory(cases: list[Case]) -> None:
    """§20's two-sided test — and it has to pass on *both* sides to ship.

    A layer that edits the chords an engine reported cannot be judged by one
    number, because the two ways it can be wrong pull in opposite directions:

    **Run A — ground truth as both engines.** Every span arrives correct and
    fully believed, so there is nothing to fix and anything either layer changes
    it changes *away from the truth*. The requirement is that every edit count is
    zero and every `delivered` column is identical. This is the regression guard,
    and it is the run that would have caught the alignment defect: it isolates the
    pipeline's own arithmetic from any engine's mistakes.

    **Run B — the deployed engines.** Now the spans are wrong in the way real
    spans are wrong, and each layer has to earn its place. The requirement is
    that `delivered` goes *up*. If it does not, the layer is not paying for the
    risk it carries and turning it off is the correct posture.

    Three delivered columns rather than two, because the two correcting layers
    answer with different evidence and a song can be helped by one and not at all
    by the other: `off` is neither, `cons` is §20.4's vote alone, `both` adds
    §20.8's vocabulary. The per-track rows matter more than the mean — nine tracks
    cannot resolve a half-point effect, which is what `--noise` exists for.

    Nothing here averages the truth run with the engine run. They answer
    different questions.
    """
    print("THEORY LAYER  (§20 — the two correcting layers, off vs on)")
    print(f"{'run':<8}{'track':<22}{'off':>7}{'cons':>7}{'both':>7}{'delta':>8}"
          f"{'rewrit':>7}{'snap':>6}{'isle':>6}{'contest':>8}{'key':>15}")

    runs: list[tuple[str, list[tuple[Case, BeatGrid, list[RawChordSpan]]]]] = []
    runs.append(("truth", [(c, grid_from(c.truth), truth_spans(c.truth)) for c in cases]))

    chord_name = Settings().chord_engine
    beat_name = Settings().beat_tracker
    if chord_name in engines._CHORD_ENGINES and beat_name in engines._BEAT_TRACKERS:
        runs.append((f"{chord_name}", [(c, grid_for(beat_name, c)[0],
                                        chords_for(chord_name, c)[0]) for c in cases]))

    for label, prepared in runs:
        # Real and synthetic kept apart, per this module's own rule: the synthetic
        # specimens prove the plumbing and nothing else, and a mean over both
        # dilutes the only evidence about a dense mix. Averaged together they read
        # 0.822 → 0.827 where the real corpus reads 0.796 → 0.803, and the second
        # pair is the one that describes the product.
        columns: dict[str, list[float]] = {"off": [], "cons": [], "both": []}
        real_columns: dict[str, list[float]] = {"off": [], "cons": [], "both": []}
        edits = {"rewritten": 0, "snapped": 0, "islands": 0}
        for case, grid, raw in prepared:
            try:
                outcomes = {
                    "off": _analyze(case, grid, raw, vote=False, vocab=False),
                    "cons": _analyze(case, grid, raw, vote=True, vocab=False),
                    "both": _analyze(case, grid, raw, vote=True, vocab=True),
                }
            except Exception as exc:
                print(f"{label:<8}{case.name:<22}{'—':>7}{'—':>7}{'—':>7}"
                      f"{'':>8}{'':>7}{'':>6}{'':>6}{'':>8}  {type(exc).__name__}")
                continue
            delivered = {name: _delivered(outcome, case.truth)
                         for name, outcome in outcomes.items()}
            report = outcomes["both"].theory
            edits["rewritten"] += report.rewrittenBars
            edits["snapped"] += report.snappedSpans
            edits["islands"] += report.absorbedIslands
            tonic = CompositionPayload.model_validate(outcomes["both"].song).tonic
            print(f"{label:<8}{case.name:<22}{delivered['off']:>7.3f}"
                  f"{delivered['cons']:>7.3f}{delivered['both']:>7.3f}"
                  f"{delivered['both'] - delivered['off']:>+8.3f}"
                  f"{report.rewrittenBars:>7}{report.snappedSpans:>6}"
                  f"{report.absorbedIslands:>6}{report.contestedBars:>8}"
                  f"{tonic + ' ' + report.scale:>15}")
            if delivered["off"] == delivered["off"]:
                for name, value in delivered.items():
                    columns[name].append(value)
                    if case.is_real:
                        real_columns[name].append(value)

        for name, table in (("REAL MEAN", real_columns), ("ALL MEAN", columns)):
            if not table["off"]:
                continue
            means = {key: statistics.mean(values) for key, values in table.items()}
            # The verdict is read off the real corpus only, for the same reason the
            # means are split: a synthetic specimen the layer cannot touch drags
            # every gain toward zero and would make a real win look marginal.
            verdict = (_verdict(label, means["off"], means["both"], sum(edits.values()))
                       if table is real_columns else "")
            print(f"{label:<8}{name:<22}{means['off']:>7.3f}{means['cons']:>7.3f}"
                  f"{means['both']:>7.3f}{means['both'] - means['off']:>+8.3f}"
                  f"{edits['rewritten']:>7}{edits['snapped']:>6}{edits['islands']:>6}"
                  f"{'':>8}  {verdict}")
        print()


# A gain smaller than this is not worth the risk of a layer that edits chords:
# on nine tracks it is one or two songs moving, which is noise, and the cost of
# being wrong is a chart that confidently shows a chord nobody played.
MATERIAL_GAIN = 0.005


def _verdict(label: str, off: float, on: float, rewritten: int) -> str:
    """State plainly whether the run met its requirement.

    Deliberately harder to please than "the mean went up". A layer with this
    much downside has to clear a margin, not a sign — and a mean that improves
    while individual tracks regress is exactly the shape a small corpus produces
    by chance.
    """
    if label == "truth":
        return ("PASS — no-op on perfect input" if rewritten == 0 and on >= off - 1e-9
                else "FAIL — edited a correct chart")
    gain = on - off
    if gain >= MATERIAL_GAIN:
        return "PASS — the correcting layers earn their place"
    if gain > 0:
        return f"MARGINAL (+{gain:.3f}) — real but within noise on {label}; read the per-track rows"
    return "no gain — ship with the two theory flags off"


# --- §20.8's two measurements ------------------------------------------------
#
# The theory layer's problem as a *measurement* problem: the real corpus is nine
# tracks, and the population any quality rule is allowed to touch is a few dozen
# spans inside them. That is far too little to resolve a half-point effect, and
# `bench_theory`'s per-track rows are the only honest read of it. So §20.8 is
# measured two further ways, and both are here rather than in a notebook because a
# rule whose justification cannot be re-run is a rule nobody can revisit.

def bench_strum(cases: list[Case]) -> None:
    """§14 — the strumming pattern, scored against the strum that was played.

    The one part of the pipeline that had ground truth recorded (`synth.Truth`
    carries `pattern_beats`) and **nothing scoring against it**. Every other
    stage here is measured; the grooves were judged by reading them, which is how
    a real regression — grooves collapsing into straight eighths on any track
    with drums on it — sat in the tree behind a green suite.

    Three columns, because a pattern can be wrong in two directions and one
    number hides which:

    - `F` — F-measure of the extracted stroke positions against the played ones.
      The headline, and the only column that is not gameable on its own.
    - `extra` — strokes emitted that nobody played, as a share of those played.
      This is the reported defect's own number: a groove that collapses into
      eighths scores high here while `F` stays deceptively respectable, because
      the strokes the player *did* strike are all still in there.
    - `miss` — the opposite failure, and the one an over-eager fix causes. A rule
      that suppresses the drummer by suppressing everything drives `extra` to
      zero and `miss` through the roof, and it has not bought anything.

    Synthetic only, deliberately. The real corpus has Isophonics *chord*
    annotations and no strum transcription, so there is nothing on those tracks
    to score against — and inventing one by ear is exactly the impressionism this
    is replacing. What the kit specimens can answer is the question the change
    turns on: whether the extractor is reading the guitar or the drummer.
    """
    scored = [c for c in cases if not c.is_real and c.truth.get("pattern_beats")]
    if not scored:
        print("STRUMMING — no synthetic specimens with a pattern to score against.\n")
        return

    print("STRUMMING  (§14 — extracted strokes vs the strum that was played)")
    print(f"{'track':<24}{'played':>8}{'got':>6}{'F':>7}{'extra':>7}{'miss':>7}"
          f"{'conf':>7}  pattern")

    totals: list[tuple[float, float, float]] = []
    detector = engines.build_onset_detector(Settings())
    if detector is None:
        print("  ! no onset detector registered — install the `audio` extra\n")
        return

    for case in scored:
        truth = case.truth
        played = [round(float(b), 3) for b in truth["pattern_beats"]]
        bar_beats = float(truth["time_signature"].split("/")[0])
        grid = grid_from(truth)
        axis = build_axis(grid)
        if axis is None:
            continue
        onsets = detector.detect(case.pcm, case.sample_rate)
        folded = fold_onsets(onsets, axis, bar_beats=bar_beats, first_beat=0.0,
                             last_beat=float(len(truth["beats_ms"]) - 1))
        got = extract(folded, bar_beats=bar_beats,
                      bars=max(1, len(truth["downbeats_ms"]) - 1),
                      tempo=int(round(truth["tempo"])), name="bench",
                      time_signature=truth["time_signature"])
        positions = [round(s.beat, 3) for s in got.pattern.strokes]

        hits = _matched(positions, played)
        precision = hits / len(positions) if positions else 0.0
        recall = hits / len(played) if played else 0.0
        f = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        extra = (len(positions) - hits) / len(played) if played else 0.0
        missed = (len(played) - hits) / len(played) if played else 0.0
        totals.append((f, extra, missed))

        shown = " ".join(f"{p:g}" for p in positions) or "—"
        print(f"{case.name:<24}{len(played):>8}{len(positions):>6}{f:>7.3f}"
              f"{extra:>7.2f}{missed:>7.2f}{got.confidence:>7.2f}  {shown}"
              + ("  [fallback]" if got.is_fallback else ""))

    if totals:
        print(f"{'MEAN':<24}{'':>8}{'':>6}"
              f"{statistics.mean(f for f, _, _ in totals):>7.3f}"
              f"{statistics.mean(e for _, e, _ in totals):>7.2f}"
              f"{statistics.mean(m for _, _, m in totals):>7.2f}")
    print()


def _matched(got: list[float], played: list[float], tolerance: float = 0.06) -> int:
    """How many extracted strokes land on a stroke that was played. Greedy and
    one-for-one, so emitting a cluster of strokes around one real one counts
    once — otherwise a denser answer would score better for being denser."""
    remaining = list(played)
    hits = 0
    for position in got:
        match = next((p for p in remaining if abs(p - position) <= tolerance), None)
        if match is not None:
            remaining.remove(match)
            hits += 1
    return hits


def bench_calibration(cases: list[Case]) -> None:
    """Given what the engine says, what does the record actually play?

    This is the table `analysis/vocabulary.py`'s `SNAP_TO` is built from, and it
    is the whole reason that table is not simply "anything `is_near_miss`
    admits". Near-miss says two chords are close enough for a recognizer to slide
    between them. It says nothing about **which direction it slides**, and that is
    the only fact that decides whether flattening a doubtful reading pays.

    Read the `as read` and `as triad` columns as a wager. A `dominant7` this engine
    reports is the plain major triad about twice as often as it is a seventh, so
    flattening a doubtful one is a bet at 2:1 on. A `major7` is *never* the plain
    triad, so the same edit there can only lose — which is what it did to "Let It
    Be"'s opening Fmaj7 while the rule was still generic.
    """
    chord_name, beat_name = Settings().chord_engine, Settings().beat_tracker
    if chord_name not in engines._CHORD_ENGINES or beat_name not in engines._BEAT_TRACKERS:
        print("CALIBRATION — needs the configured engines installed; skipped.\n")
        return

    print("CALIBRATION  (given the engine says X, what does the record play?)")
    stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {"spans": 0.0, "beats": 0.0, "conf": 0.0,
                 "as_read": 0.0, "as_triad": 0.0, "root_wrong": 0.0})
    for case in cases:
        if not case.is_real:
            continue
        grid, _ = grid_for(beat_name, case)
        raw, _ = chords_for(chord_name, case)
        axis = build_axis(reconcile(grid, raw).grid)
        if axis is None:
            continue
        for span in postprocess.process(raw, axis):
            played = _truth_chord_at(case.truth, axis, span)
            if played is None:
                continue
            entry = stats[span.quality]
            entry["spans"] += 1
            entry["beats"] += span.length_beats
            entry["conf"] += span.confidence * span.length_beats
            if played[0] != span.root_pc:
                entry["root_wrong"] += span.length_beats
            elif played[1] == span.quality:
                entry["as_read"] += span.length_beats
            elif played[1] == _plain_triad(span.quality):
                entry["as_triad"] += span.length_beats
    print(f"{'engine says':<16}{'spans':>7}{'beats':>7}{'conf':>7}"
          f"{'as read':>9}{'as triad':>10}{'root wrong':>12}{'other':>8}")
    for quality, entry in sorted(stats.items(), key=lambda kv: -kv[1]["beats"]):
        beats = entry["beats"] or 1.0
        other = beats - entry["as_read"] - entry["as_triad"] - entry["root_wrong"]
        print(f"{quality:<16}{entry['spans']:>7.0f}{beats:>7.0f}{entry['conf'] / beats:>7.2f}"
              f"{entry['as_read'] / beats:>9.2f}{entry['as_triad'] / beats:>10.2f}"
              f"{entry['root_wrong'] / beats:>12.2f}{other / beats:>8.2f}")
    print()


def _plain_triad(quality: str) -> str:
    return MINOR if quality in {"minor", "minor7", "diminished", "diminished7",
                                "halfDiminished7"} else MAJOR


def _truth_chord_at(truth: dict, axis, span) -> tuple[int, str] | None:
    """The chord the record plays for most of a quantized span's stretch."""
    times = axis.times_ms
    start = times[min(span.start_beat, len(times) - 1)]
    end = times[min(span.end_beat, len(times) - 1)]
    best, best_overlap = None, 0
    for chord in truth["chords"]:
        overlap = min(end, chord["endMs"]) - max(start, chord["startMs"])
        if overlap > best_overlap:
            parsed = normalize(chord["name"])
            if parsed is not None:
                best, best_overlap = (parsed[0], parsed[1]), overlap
    return best


# The engine's mistakes, as measured rather than as imagined: for each quality the
# record plays, what this engine reports instead, how often, and how sure it
# sounds when it does. Produced by `--calibration` (the conditional form of the
# table it prints), beats-weighted over the nine real tracks, entries below 1%
# folded into the correct reading.
#
# `rootN` means "the same quality, N semitones off" — the mistakes no rule in
# §20.8 can touch, and they are in the model precisely for that reason. A noise
# benchmark containing only the errors the layer is good at is not a measurement.
#
# The seventh rows matter for the same reason and are easy to leave out, which
# would quietly rig the whole run. A song's *genuine* sevenths have to be in the
# injected corpus, because the population §20.8 is most dangerous to is a real
# extension the engine heard correctly and **hedged on** — and this engine hedges
# on all of them (`CORRECT_CONFIDENCE`). Without these rows every seventh in the
# corpus arrives fully believed, the confidence gate closes on all of them, and
# the benchmark cannot see the one kind of damage the real corpus actually caught.
NOISE_MODEL: dict[str, tuple[tuple[str, float, float], ...]] = {
    MAJOR: (
        ("dominant7", 0.032, 0.60),      # the spurious seventh — the biggest bucket
        ("root-5", 0.030, 0.77),         # heard the IV/V instead
        ("root+5", 0.018, 0.85),
        ("root+2", 0.014, 0.75),
        ("root-3", 0.011, 0.65),         # the relative minor, a third down
        ("minor", 0.007, 0.71),          # the third the mix buried
    ),
    MINOR: (
        ("root+3", 0.051, 0.81),         # the relative major
        ("minor7", 0.047, 0.54),         # the spurious seventh again
        ("root-4", 0.046, 0.69),
        ("root-5", 0.016, 0.69),
        ("root+5", 0.014, 0.81),
        ("major", 0.010, 0.70),          # the third, the other way
    ),
    "dominant7": (
        ("major", 0.363, 0.64),          # the seventh dropped — nothing here can add it back
        ("root-3", 0.047, 0.62),
        ("root-5", 0.028, 0.35),
        ("root-2", 0.026, 0.67),
    ),
    "minor7": (
        ("minor", 0.372, 0.60),
        ("dominant7", 0.217, 0.65),
        ("root+3", 0.038, 0.71),
        ("root-5", 0.026, 0.72),
    ),
    "major7": (
        ("major", 0.730, 0.70),
        ("root-3", 0.206, 0.62),
        ("root+4", 0.048, 0.70),
    ),
}

# What the engine sounds like when it is **right**, per quality — measured, and
# the spread matters more than the mean. A flat 1.0 would make every confidence
# gate in §20 trivially open, which is the one way to rig this test; and a flat
# high value for the sevenths would do the same thing more subtly, since a
# correctly-heard seventh this engine was sure of is not a case any rule here has
# to survive. It is the hedged ones that are the test.
CORRECT_CONFIDENCE: dict[str, tuple[float, float]] = {
    MAJOR: (0.72, 0.95),                 # measured mean 0.86
    MINOR: (0.62, 0.92),                 # 0.78
    "dominant7": (0.48, 0.80),           # 0.64 — In My Life's A7 lives here
    "minor7": (0.40, 0.72),              # 0.55
    "major7": (0.36, 0.62),              # 0.48 — Let It Be's Fmaj7
    "diminished7": (0.16, 0.34),         # 0.24, and right 90% of the time
}
DEFAULT_CORRECT_CONFIDENCE = (0.55, 0.85)


def corrupt(truth: dict, rng: random.Random) -> list[RawChordSpan]:
    """Ground truth, mistaken the way this engine is measured to mistake it.

    One draw per annotated chord, from `NOISE_MODEL`. The confidence travels with
    the mistake, because that is the correlation the whole theory layer runs on:
    the engine is measurably less sure when it is wrong, and every gate in §20.4
    and §20.8 is built on being able to see that.
    """
    spans: list[RawChordSpan] = []
    for chord in truth["chords"]:
        parsed = normalize(chord["name"])
        if parsed is None:
            continue
        root, quality = parsed[0], parsed[1]
        confidence = rng.uniform(*CORRECT_CONFIDENCE.get(
            quality, DEFAULT_CORRECT_CONFIDENCE))
        roll = rng.random()
        for said, probability, said_confidence in NOISE_MODEL.get(quality, ()):
            if roll < probability:
                if said.startswith("root"):
                    root = (root + int(said[4:])) % 12
                else:
                    quality = said
                # Jittered around the measured mean, so the run is not decided by
                # one number sitting a hair either side of a threshold.
                confidence = min(0.99, max(0.05, rng.gauss(said_confidence, 0.08)))
                break
            roll -= probability
        spans.append(RawChordSpan(start_ms=int(chord["startMs"]), end_ms=int(chord["endMs"]),
                                  label=render(root, quality), confidence=confidence))
    return spans


def bench_noise(cases: list[Case], seeds: int = 12) -> None:
    """The measurement with enough noise in it to see the layers work.

    Ground truth supplies the harmony, the timing and the form; `corrupt` supplies
    the engine's *measured* mistakes, at the measured rates and confidences. That
    combination is what the real corpus cannot offer: the same nine songs, with
    hundreds of injected errors instead of a few dozen, and every error's correct
    answer known exactly.

    Three columns, because a layer that edits chords has to be judged on both
    sides at once and a single mean hides one of them:

        in       delivered accuracy of the corrupted chart, layers off
        out      the same chart with the layers on
        fixed    share of the *injected* errors the layers removed
        broke    share of the *correct* chords the layers destroyed

    `broke` is the column that would condemn this. A layer that fixes a third of
    the noise and breaks a twentieth of the music is not worth having, however
    well the mean reads — which is why the two are never summed here.

    **What this run cannot tell you**, and the reason `bench_theory`'s nine
    per-track rows stay the column of record: the noise is drawn independently per
    chord, so it cannot reproduce a mistake the engine makes *identically* in every
    pass of a section — which is real, common, and the thing that defeats the vote.
    It also inherits whatever the model leaves out. That is not a small caveat: the
    first version of this model had no rows for the sevenths at all, so every
    genuine seventh arrived fully believed, no confidence gate could open on one,
    and the run was structurally incapable of seeing the damage the real corpus
    caught on "In My Life". A synthetic benchmark answers exactly the question its
    noise model asks.
    """
    print(f"NOISE INJECTION  (truth + the engine's measured mistakes, {seeds} seeds)")
    print(f"{'layers':<22}{'in':>7}{'out':>7}{'delta':>8}{'fixed':>8}{'broke':>8}")

    real = [case for case in cases if case.is_real]
    if not real:
        print("  no real tracks — the noise model is measured against them.\n")
        return

    modes = (("consensus", True, False), ("vocabulary", False, True),
             ("both", True, True))
    for label, vote, vocab in modes:
        rows: list[tuple[float, float, float, float]] = []
        for seed in range(seeds):
            for case in real:
                # Seeded from the *string*, not from `hash()`: Python randomizes
                # string hashing per process, so `hash((name, seed))` drew a
                # different corpus on every run and the printed numbers moved by
                # ±0.005 between two runs of the same code. A benchmark whose
                # answer depends on the process it ran in cannot be quoted.
                rng = random.Random(f"{case.name}:{seed}")
                raw = corrupt(case.truth, rng)
                grid = grid_from(case.truth)
                try:
                    before = _analyze(case, grid, raw, vote=False, vocab=False)
                    after = _analyze(case, grid, raw, vote=vote, vocab=vocab)
                except Exception:
                    continue
                clean, dirty = _per_beat(before, case.truth), _per_beat(after, case.truth)
                if clean is None or dirty is None:
                    continue
                fixed = sum(1 for a, b in zip(clean, dirty) if not a and b)
                broke = sum(1 for a, b in zip(clean, dirty) if a and not b)
                wrong = sum(1 for a in clean if not a) or 1
                right = sum(1 for a in clean if a) or 1
                rows.append((sum(clean) / len(clean), sum(dirty) / len(dirty),
                             fixed / wrong, broke / right))
        if not rows:
            continue
        means = [statistics.mean(column) for column in zip(*rows)]
        print(f"{label:<22}{means[0]:>7.3f}{means[1]:>7.3f}{means[1] - means[0]:>+8.3f}"
              f"{means[2]:>8.3f}{means[3]:>8.3f}")
    print()


def _per_beat(outcome, truth: dict) -> list[bool] | None:
    """Right/wrong at every sampled millisecond of the delivered chart.

    The same reconstruction `delivered_accuracy` scores, kept as the vector rather
    than the mean, so the two sides of an edit can be counted separately.
    """
    if outcome.sync is None or not outcome.sync.beatAnchors:
        return None
    payload = CompositionPayload.model_validate(outcome.song)
    flats = prefers_flats(payload.tonic, payload.mode)
    out: list[bool] = []
    for chord in truth["chords"]:
        target = normalize(chord["name"])
        if target is None:
            continue
        expected = render(target[0], target[1], flats=flats)
        for t in range(int(chord["startMs"]), int(chord["endMs"]), FRAME_STEP_MS):
            out.append(chord_at_video_ms(payload, outcome.sync, t) == expected)
    return out or None


def summarise(chord_tallies: dict[str, Tally], beat_tallies: dict[str, Tally],
              real: list[Case], synthetic: list[Case]) -> None:
    """The table the decision is actually made from."""
    print("=" * 78)
    print(f"SUMMARY — {len(real)} real track(s), {len(synthetic)} synthetic")
    print("Real-mix numbers are the ones that matter; synthetic proves plumbing.\n")

    if chord_tallies:
        print(f"{'chord engine':<14}{'REAL acc':>10}{'root':>8}{'SYNTH acc':>11}"
              f"{'s/track':>9}")
        for name, tally in sorted(chord_tallies.items(),
                                  key=lambda kv: -kv[1].mean("real", "accuracy")):
            print(f"{name:<14}{tally.mean('real', 'accuracy'):>10.3f}"
                  f"{tally.mean('real', 'root'):>8.3f}"
                  f"{tally.mean('synthetic', 'accuracy'):>11.3f}"
                  f"{tally.mean('real', 'seconds'):>9.1f}")
        print()

    if beat_tallies:
        print(f"{'beat tracker':<14}{'REAL beat F':>13}{'REAL down F':>13}"
              f"{'bpm err':>9}{'meter':>8}{'bars raw':>10}{'repaired':>10}"
              f"{'s/track':>9}")
        for name, tally in sorted(beat_tallies.items(),
                                  key=lambda kv: -kv[1].mean("real", "down_f")):
            print(f"{name:<14}{tally.mean('real', 'beat_f'):>13.3f}"
                  f"{tally.mean('real', 'down_f'):>13.3f}"
                  f"{tally.mean('real', 'bpm_err'):>9.1f}"
                  f"{tally.mean('real', 'meter'):>8.2f}"
                  f"{tally.mean('real', 'bars_before'):>10.2f}"
                  f"{tally.mean('real', 'bars_after'):>10.2f}"
                  f"{tally.mean('real', 'seconds'):>9.1f}")
        print()


def main() -> int:
    cases = load_cases()
    if not cases:
        print("No benchmark audio. Run `python bench/synth.py` first, or build the "
              "real corpus with `python bench/fetch_corpus.py --annotations …`.")
        return 1
    real, synthetic = _split(cases)
    print(f"{len(cases)} track(s): {len(real)} real, {len(synthetic)} synthetic\n")

    if not engines._CHORD_ENGINES and not engines._BEAT_TRACKERS:
        print("No engines are registered. Install the `audio` extra and see\n"
              "app/analysis/engines.py for what each adapter needs.\n")
        return 0

    if not real:
        print("! No real tracks — synthetic audio cannot tell you how these engines\n"
              "  behave on a dense mix, which is what the choice hinges on.\n")

    if "--theory" in sys.argv:
        # §20 on its own: the two-sided consensus test, without paying for a
        # full engine sweep. `truth` needs no engines at all.
        bench_theory(cases)
        return 0

    if "--noise" in sys.argv:
        # §20.8's measurement: injected mistakes, so the population is big enough
        # to resolve. Needs no engines either — the noise model *is* the engine.
        bench_noise(cases)
        return 0

    if "--strum" in sys.argv:
        # §14 on its own: the extraction against the strum that was played.
        # Needs the onset detector but neither the chord engine nor the tracker,
        # since the grid comes from the ground truth.
        bench_strum(cases)
        return 0

    if "--calibration" in sys.argv:
        # Where `vocabulary.SNAP_TO` and `NOISE_MODEL` come from. Needs the real
        # engines, since it is a measurement *of* them.
        bench_calibration(cases)
        return 0

    chord_tallies = bench_chords(cases)
    beat_tallies = bench_beats(cases)
    bench_pipeline(cases)
    bench_theory(cases)
    bench_noise(cases)
    bench_calibration(cases)
    summarise(chord_tallies, beat_tallies, real, synthetic)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
