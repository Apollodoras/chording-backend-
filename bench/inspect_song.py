"""What the engines actually said, laid on the measured bar grid.

    python bench/inspect_song.py creep --bars 40

Written to answer one question while the reference charts were being drafted:
**where does the harmony change**. A published chord chart is aligned to lyrics,
not to barlines, and the memory of this repo already records what that costs —
Mary Jane's Last Dance was charted `| Am | G D |` from a lyric tab, the real
music is `| Am G | D Am |`, and grading against the mistake read as an engine
defect.

So the harmonic *rhythm* is read off the recording and only the chord *names*
come from the published chart. That is not circular: this prints the raw engine
output before any of the layers under test have touched it, and it is consulted
for **when** something changes, not for what it changed to. The beat grid is the
tracker's, which is graded by `run_bench.py`'s sync gate rather than by any chart.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.lab import load_features, songbook  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("slug")
    parser.add_argument("--bars", type=int, default=48)
    parser.add_argument("--from-bar", type=int, default=0)
    parser.add_argument("--width", type=int, default=8, help="bars per printed line")
    args = parser.parse_args()

    entry = songbook()[args.slug]
    cached = load_features(args.slug)
    grid = cached["grid"]
    beats, downbeats = grid["beats_ms"], grid["downbeats_ms"]
    raw = cached["raw"]

    print(f"{entry['title']} — {entry['artist']}   {grid['bpm']:.1f} bpm  "
          f"{grid['time_signature']}  beat-confidence {grid['confidence']:.2f}  "
          f"tuning {cached['tuning']['semitones'] * 100:+.0f} cents")
    print(f"{len(beats)} beats, {len(downbeats)} downbeats "
          f"({len(beats) / max(1, len(downbeats)):.2f} beats per bar), "
          f"{len(raw)} raw chord spans\n")

    # One cell per bar, holding whatever the engine had sounding over it. The
    # engine's own label, unnormalized — this is a view of its output, not of
    # the chart the chart-builder would make from it.
    edges = downbeats + [beats[-1] if beats else 0]
    cells: list[str] = []
    for index in range(len(downbeats)):
        start, end = edges[index], edges[index + 1]
        inside = [s for s in raw if s["end_ms"] > start and s["start_ms"] < end]
        names, seen = [], None
        for span in inside:
            label = span["label"].replace(":maj", "").replace(":min", "m")
            if label != seen:
                names.append(label)
                seen = label
        cells.append(" ".join(names) if names else "-")

    stop = min(len(cells), args.from_bar + args.bars)
    for start in range(args.from_bar, stop, args.width):
        row = cells[start:start + args.width]
        stamp = downbeats[start] / 1000.0
        print(f"{start:>4} {int(stamp // 60):>3d}:{stamp % 60:04.1f}  "
              + " ".join(f"| {cell:<11}" for cell in row) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
