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
- **the whole pipeline** — whether the emitted song lints clean and whether the
  sidecar survives `lint_sync`. An engine pairing that produces beautiful numbers
  and an unusable song has not won anything.

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
import statistics
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.analysis import engines  # noqa: E402
from app.analysis.pipeline import assemble  # noqa: E402
from app.analysis.postprocess import quantize  # noqa: E402
from app.analysis.types import BeatGrid, EngineInfo, RawChordSpan, VideoMeta  # noqa: E402
from app.chords import NORMAL, normalize, render  # noqa: E402
from app.config import Settings  # noqa: E402
from app.lint import lint, lint_sync  # noqa: E402
from app.payload import CompositionPayload  # noqa: E402

AUDIO = ROOT / "bench" / "audio"

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
        spans = engine.analyze(case.pcm, case.sample_rate)
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

def bench_beats(cases: list[Case]) -> dict[str, Tally]:
    names = sorted(engines._BEAT_TRACKERS)
    if not names:
        print("no beat trackers registered — see app/analysis/engines.py\n")
        return {}
    print("BEAT TRACKERS")
    print(f"{'engine':<12}{'track':<22}{'beat F':>8}{'downbeat F':>12}"
          f"{'bpm err':>9}{'meter':>8}{'sec':>7}")
    tallies = {name: Tally() for name in names}
    for name in names:
        for case in cases:
            grid, elapsed = grid_for(name, case)
            beat_f, _, _ = f_measure(grid.beats_ms, case.truth["beats_ms"])
            down_f, _, _ = f_measure(grid.downbeats_ms, case.truth["downbeats_ms"])
            bpm_error = abs(grid.bpm - case.truth["tempo"])
            meter_ok = grid.time_signature == case.truth["time_signature"]
            tallies[name].add(case, beat_f=beat_f, down_f=down_f,
                              bpm_err=bpm_error, meter=float(meter_ok),
                              seconds=elapsed)
            print(f"{name:<12}{case.name:<22}{beat_f:>8.3f}{down_f:>12.3f}"
                  f"{bpm_error:>9.1f}{('ok' if meter_ok else grid.time_signature):>8}"
                  f"{elapsed:>7.1f}")
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
            quantize(spans, grid_from(case.truth))
            print(f"{name:<12}{case.name:<22}{accuracy:>10.3f}{roots:>11.3f}"
                  f"{len(spans):>8}{elapsed:>7.1f}")
    print()
    return tallies


def bench_pipeline(cases: list[Case]) -> None:
    """The number that actually decides: does the pairing produce a song the app
    will play, and a sidecar that agrees with it?"""
    chord_names = sorted(engines._CHORD_ENGINES)
    beat_names = sorted(engines._BEAT_TRACKERS)
    if not (chord_names and beat_names):
        return

    settings = Settings(scratch_root="/tmp/chords-scratch")
    print("END TO END  (does the pairing produce a song the app will play?)")
    print(f"{'pairing':<24}{'clean':>7}{'synced':>8}{'failed':>8}  notes")
    for chord_name in chord_names:
        for beat_name in beat_names:
            clean = synced = failed = 0
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
                payload = CompositionPayload.model_validate(outcome.songs[NORMAL])
                if not lint(payload):
                    clean += 1
                if outcome.sync is not None and not lint_sync(payload, outcome.sync):
                    synced += 1
            note = ", ".join(f"{k}×{v}" for k, v in sorted(reasons.items())) or "-"
            print(f"{chord_name + '+' + beat_name:<24}{clean:>4}/{len(cases):<2}"
                  f"{synced:>5}/{len(cases):<2}{failed:>8}  {note}")
    print()


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
              f"{'bpm err':>9}{'meter':>8}{'s/track':>9}")
        for name, tally in sorted(beat_tallies.items(),
                                  key=lambda kv: -kv[1].mean("real", "down_f")):
            print(f"{name:<14}{tally.mean('real', 'beat_f'):>13.3f}"
                  f"{tally.mean('real', 'down_f'):>13.3f}"
                  f"{tally.mean('real', 'bpm_err'):>9.1f}"
                  f"{tally.mean('real', 'meter'):>8.2f}"
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

    chord_tallies = bench_chords(cases)
    beat_tallies = bench_beats(cases)
    bench_pipeline(cases)
    summarise(chord_tallies, beat_tallies, real, synthetic)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
