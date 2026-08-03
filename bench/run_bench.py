"""§8 step 2's harness: benchmark the candidate engines, then **report and let
the owner choose**.

The handoff is explicit that this decision is not the backend's to make, so this
script deliberately does not pick a winner — it prints a table and stops.

What it scores:

- **beat trackers** — F-measure against ground-truth beats and downbeats at the
  standard ±70 ms tolerance, plus tempo error. Downbeats are scored separately
  and matter more here than beats do: §13.2's anchors are downbeats, and a
  tracker that finds the pulse but not the "one" produces a sidecar that walks
  the cursor off the song.
- **chord engines** — per-beat accuracy after normalization into the app's
  grammar (§12.2), which is the only accuracy that means anything downstream: an
  engine that nails `Cmaj9` and one that says `Cmaj7` score identically, because
  the app plays the same chord for both.
- **the whole pipeline** — whether the emitted song lints clean and whether the
  sidecar survives `lint_sync`. An engine pairing that produces beautiful numbers
  and an unusable song has not won anything.

Usage:
    python bench/synth.py          # render the synthetic set (exact ground truth)
    python bench/run_bench.py      # score whatever engines are registered

With no engines registered it says so and exits — which is the current state, and
the point: nothing is chosen until the numbers exist.
"""

from __future__ import annotations

import json
import math
import sys
import wave
from dataclasses import dataclass
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


def load_cases() -> list[Case]:
    cases: list[Case] = []
    for wav_path in sorted(AUDIO.glob("*.wav")):
        truth_path = wav_path.with_suffix("").with_suffix(".truth.json")
        if not truth_path.exists():
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


def chord_accuracy(spans: list[RawChordSpan], truth: dict) -> float:
    """Share of the track (by time) whose chord we would end up **playing**
    correctly — i.e. compared after normalization into the app's grammar."""
    expected: list[tuple[int, int, str]] = [
        (c["startMs"], c["endMs"], c["name"]) for c in truth["chords"]
    ]
    total_ms = sum(end - start for start, end, _ in expected)
    if not total_ms:
        return 0.0

    correct = 0
    for start, end, name in expected:
        for span in spans:
            overlap = min(end, span.end_ms) - max(start, span.start_ms)
            if overlap <= 0:
                continue
            parsed = normalize(span.label)
            if parsed is None:
                continue
            root, quality, _ = parsed
            if render(root, quality) == render(*normalize(name)[:2]):
                correct += overlap
    return correct / total_ms


def grid_from(truth: dict) -> BeatGrid:
    return BeatGrid(
        beats_ms=truth["beats_ms"], downbeats_ms=truth["downbeats_ms"],
        bpm=float(truth["tempo"]), confidence=1.0,
        time_signature=truth["time_signature"],
    )


# --- the runs ---------------------------------------------------------------

def bench_beats(cases: list[Case]) -> None:
    names = sorted(engines._BEAT_TRACKERS)
    if not names:
        print("no beat trackers registered — see app/analysis/engines.py\n")
        return
    print("BEAT TRACKERS")
    print(f"{'engine':<18}{'track':<26}{'beat F':>8}{'downbeat F':>12}{'bpm err':>9}")
    for name in names:
        tracker = engines._BEAT_TRACKERS[name]()
        for case in cases:
            grid = tracker.track(case.pcm, case.sample_rate)
            beat_f, _, _ = f_measure(grid.beats_ms, case.truth["beats_ms"])
            down_f, _, _ = f_measure(grid.downbeats_ms, case.truth["downbeats_ms"])
            bpm_error = abs(grid.bpm - case.truth["tempo"])
            print(f"{name:<18}{case.name:<26}{beat_f:>8.3f}{down_f:>12.3f}{bpm_error:>9.1f}")
    print()


def bench_chords(cases: list[Case]) -> None:
    names = sorted(engines._CHORD_ENGINES)
    if not names:
        print("no chord engines registered — see app/analysis/engines.py\n")
        return
    print("CHORD ENGINES  (accuracy measured AFTER normalization — §12.2)")
    print(f"{'engine':<18}{'track':<26}{'accuracy':>10}{'spans':>8}")
    for name in names:
        engine = engines._CHORD_ENGINES[name]()
        for case in cases:
            spans = engine.analyze(case.pcm, case.sample_rate)
            accuracy = chord_accuracy(spans, case.truth)
            quantized = quantize(spans, grid_from(case.truth))
            print(f"{name:<18}{case.name:<26}{accuracy:>10.3f}{len(quantized):>8}")
    print()


def bench_pipeline(cases: list[Case]) -> None:
    """The number that actually decides: does the pairing produce a song the app
    will play, and a sidecar that agrees with it?"""
    chord_names = sorted(engines._CHORD_ENGINES)
    beat_names = sorted(engines._BEAT_TRACKERS)
    if not (chord_names and beat_names):
        return

    settings = Settings(scratch_root="/tmp/chords-scratch")
    print("END TO END")
    print(f"{'pairing':<34}{'track':<26}{'lints':>7}{'sync':>7}{'sections':>10}")
    for chord_name in chord_names:
        for beat_name in beat_names:
            engine = engines._CHORD_ENGINES[chord_name]()
            tracker = engines._BEAT_TRACKERS[beat_name]()
            for case in cases:
                grid = tracker.track(case.pcm, case.sample_rate)
                raw = engine.analyze(case.pcm, case.sample_rate)
                meta = VideoMeta(video_id="bench0000000", title=case.name,
                                 duration_s=case.truth["duration_ms"] / 1000.0)
                try:
                    outcome = assemble(meta=meta, grid=grid, raw=raw, onsets=[],
                                       settings=settings,
                                       chords_engine=EngineInfo(chord_name, "bench"),
                                       beats_engine=EngineInfo(beat_name, "bench"))
                except Exception as exc:
                    print(f"{chord_name + '+' + beat_name:<34}{case.name:<26}"
                          f"{'FAIL':>7}{'-':>7}  {type(exc).__name__}: {exc}")
                    continue
                payload = CompositionPayload.model_validate(outcome.songs[NORMAL])
                clean = not lint(payload)
                synced = outcome.sync is not None and not lint_sync(payload, outcome.sync)
                sections = len(payload.arrangement.sections)
                print(f"{chord_name + '+' + beat_name:<34}{case.name:<26}"
                      f"{'yes' if clean else 'NO':>7}{'yes' if synced else 'no':>7}"
                      f"{sections:>10}")
    print()


def main() -> int:
    cases = load_cases()
    if not cases:
        print("No benchmark audio. Run `python bench/synth.py` first, or drop real "
              "tracks (plus <name>.truth.json) into bench/audio/.")
        return 1
    print(f"{len(cases)} track(s): {', '.join(c.name for c in cases)}\n")

    if not engines._CHORD_ENGINES and not engines._BEAT_TRACKERS:
        print("No engines are registered yet — which is the current, deliberate state.\n"
              "§8 step 2 says: benchmark 2+ chord engines and 2+ beat trackers, then\n"
              "report and let the owner choose. Register candidates in\n"
              "app/analysis/engines.py, add their dependencies to the `audio` extra and\n"
              "to modal_app.py's worker image, then run this again.\n")
        return 0

    bench_beats(cases)
    bench_chords(cases)
    bench_pipeline(cases)
    print("Numbers only — the engine choice is the owner's (§8 step 2).")
    print("Note: synthetic tracks prove the plumbing, not real-mix accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
