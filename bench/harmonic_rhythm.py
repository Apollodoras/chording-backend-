"""Where the harmony changes, in bars — the measurement a reference chart is written from.

    python bench/harmonic_rhythm.py three-little-birds

A published chord chart is aligned to *lyrics*. A reference chart has to be
aligned to *barlines*, and this repo has already paid for the difference once:
Mary Jane's Last Dance was charted `| Am | G D |` off a lyric tab, the music is
`| Am G | D Am |`, and grading against the mistake read as an engine defect.

So the bar length is not assumed and not taken from the beat tracker either —
it is **fitted**. Every candidate bar length is scored by how close the song's
own chord durations come to whole numbers of it, which is a property of the
recording that no chart, tracker or engine gets a vote on. The winner is
reported with the fit, and the durations are printed in bars so a section can be
counted off by hand.

What is *not* derived here is the chord names: those come from published charts,
because the engine's labels are the thing under test. This measures **when**,
never **what** — and the two songs where the engine's labels are wrong (a major
where the record plays a minor) are exactly the ones where that separation
earns its keep.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.lab import load_features, songbook  # noqa: E402

# Chords shorter than this are passing tones or engine flicker, not structure.
FLOOR_MS = 350


def merged_roots(raw: list[dict]) -> list[tuple[float, float, str]]:
    """Adjacent spans sharing a root, joined. Quality is dropped on purpose:
    `Em` following `Em7` is one chord held, and counting it as two would put a
    change where the music has none."""
    out: list[list] = []
    for span in raw:
        root = span["label"].split(":")[0]
        if root in ("N", "X"):
            root = "N.C."
        if out and out[-1][2] == root:
            out[-1][1] = span["end_ms"]
        else:
            out.append([span["start_ms"], span["end_ms"], root])
    return [(a / 1000.0, b / 1000.0, r) for a, b, r in out if b - a >= FLOOR_MS]


def fit_bar(spans: list[tuple[float, float, str]], beat_ms: float) -> tuple[float, float]:
    """The bar length that best explains the chord durations, and its error.

    Searched over whole numbers of the tracker's beat *and* its halves and
    doubles, because the tracker's own octave is one of the things in doubt: a
    tempo read twice too fast puts every chord on two bars instead of one, and
    that is a defect this corpus has to be able to see rather than adopt.
    """
    best, best_error = beat_ms * 4 / 1000.0, 1e9
    for beats in (2, 3, 4, 6, 8, 12, 16):
        for scale in (0.5, 1.0, 2.0):
            bar = beat_ms * beats * scale / 1000.0
            if bar < 0.8 or bar > 8.0:
                continue
            error, weight = 0.0, 0.0
            for start, end, root in spans:
                if root == "N.C.":
                    continue
                held = (end - start) / bar
                if held < 0.4:
                    continue
                error += abs(held - round(held)) * held
                weight += held
            if weight <= 0:
                continue
            error /= weight
            if error < best_error - 1e-9:
                best, best_error = bar, error
    return best, best_error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("slug")
    parser.add_argument("--bar", type=float, help="override the fitted bar length, in seconds")
    parser.add_argument("--per-line", type=int, default=8)
    args = parser.parse_args()

    entry = songbook()[args.slug]
    cached = load_features(args.slug)
    beats = cached["grid"]["beats_ms"]
    beat_ms = (beats[-1] - beats[0]) / max(1, len(beats) - 1)
    spans = merged_roots(cached["raw"])

    bar, error = fit_bar(spans, beat_ms)
    if args.bar:
        bar = args.bar
    origin = next((s for s, _, r in spans if r != "N.C."), 0.0)

    print(f"{entry['title']} — {entry['artist']}")
    print(f"  tracker beat {beat_ms:.0f} ms ({60000 / beat_ms:.1f} bpm), "
          f"fitted bar {bar:.3f}s = {bar / (beat_ms / 1000):.2f} tracker beats "
          f"→ {240 / bar:.1f} bpm in 4/4   (fit error {error:.3f})")
    print(f"  first chord at {origin:.2f}s; "
          f"{(spans[-1][1] - origin) / bar:.0f} bars of music\n")

    cells: list[str] = []
    for start, end, root in spans:
        held = max(1, round((end - start) / bar))
        at = (start - origin) / bar
        cells.append(f"{round(at):>4} {root:<6}{held:>3}")
    for index in range(0, len(cells), 4):
        print("   ".join(cells[index:index + 4]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
