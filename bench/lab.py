"""The edit → measure loop, on this machine, in seconds.

    python bench/lab.py fetch                  # audio for every song in the songbook
    python bench/lab.py features               # engines once per song, cached
    python bench/lab.py grade                  # assemble + score against the charts
    python bench/lab.py grade --slug creep --diff --render

Why this exists, when `scripts/analyze_once.py` already runs the real pipeline:
because `analyze_once` re-fetches from YouTube and re-runs a transformer on a CPU
every time a line of `form.py` changes, and tuning structure against ten songs
means running it hundreds of times. That is not a loop anyone iterates in.

The split that makes it fast is one the pipeline already made for its own tests:
**everything after `decode` is pure** (`pipeline.assemble`). So the expensive,
non-deterministic half — fetch, tuning probe, beat tracker, chord engine, onsets,
loudness — runs **once per song** and its output is cached as JSON. The half
being worked on runs from that cache, and the whole ten-song corpus regrades in
about a second.

Three properties this is careful about, because a harness that lies is worse
than no harness:

**It runs the real code.** `features` calls the same engines the worker builds
(`engines.build_*`), through the same `postprocess`-facing types; `grade` calls
`pipeline.assemble` itself, not a reimplementation of it. What is cached is the
engine *output*, which is the same data the worker would hand to `assemble`.

**The cache is keyed on the engines that filled it.** `chords`/`beats` names and
versions are written into the file and checked on load, so upgrading BTC does not
silently grade new code against old features.

**No audio is kept beyond the fetch.** The downloaded media is decoded to samples
and deleted in the same call (`_features`); `bench/cache/` holds JSON only. §2.1
is a property of this repo, not only of the worker.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.analysis.types import (  # noqa: E402
    BeatGrid, EnergyCurve, EngineInfo, Onset, RawChordSpan, VideoMeta,
)

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
SONGBOOK = ROOT / "songbook.json"
REFERENCE = ROOT / "reference"
SAMPLE_RATE = 22050

# Bumped when the cached shape changes. A stale file is refused, never guessed at.
CACHE_VERSION = 3


def songbook() -> dict[str, dict]:
    return json.loads(SONGBOOK.read_text())


# --------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------

def audio_path(slug: str) -> Path:
    return CACHE / "audio" / f"{slug}.m4a"


def fetch(slug: str, entry: dict, *, force: bool = False) -> Path:
    """Download one song's audio into the cache.

    Deliberately *not* `YtDlpSource.decode`: that class is built to fetch into a
    scratch directory that is destroyed on the way out, which is right for the
    service and wrong for a bench that wants to re-derive features without
    re-downloading. Same yt-dlp invocation, different lifetime.
    """
    target = audio_path(slug)
    if target.exists() and not force:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={entry['videoId']}"
    result = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "--no-warnings", "--no-playlist",
         "--extractor-args", "youtube:player_client=android",
         "-f", "worstaudio[abr>=64]/bestaudio/best",
         "-o", str(target.with_suffix(".%(ext)s")), url],
        capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"{slug}: fetch failed — {result.stderr.strip()[:300]}")
    found = sorted(target.parent.glob(f"{slug}.*"))
    if not found:
        raise RuntimeError(f"{slug}: fetch produced no file")
    if found[0] != target:
        found[0].rename(target)
    return target


def decode(path: Path):
    """Media file → mono float32 at 22.05 kHz. `ytdlp_source.decode`'s second
    half, over a file that already exists."""
    import numpy as np
    import wave

    wav = path.with_suffix(".decoded.wav")
    result = subprocess.run(
        [os.environ.get("CHORDS_FFMPEG", "ffmpeg"), "-y", "-loglevel", "error",
         "-i", str(path), "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le",
         str(wav)], capture_output=True, text=True, timeout=300)
    if result.returncode != 0 or not wav.is_file():
        raise RuntimeError(f"decode failed: {result.stderr.strip()[:300]}")
    try:
        with wave.open(str(wav)) as source:
            frames = source.readframes(source.getnframes())
    finally:
        wav.unlink(missing_ok=True)
    return np.frombuffer(frames, dtype="<i2").astype("float32") / 32768.0, SAMPLE_RATE


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------

def features_path(slug: str) -> Path:
    return CACHE / "features" / f"{slug}.json"


def build_features(slug: str, entry: dict, *, force: bool = False) -> dict:
    """Run every engine once and cache what they said.

    This is the whole non-deterministic, minutes-long half of the pipeline. The
    audio is decoded, measured and dropped inside this function.
    """
    path = features_path(slug)
    if path.exists() and not force:
        cached = json.loads(path.read_text())
        if cached.get("cacheVersion") == CACHE_VERSION:
            return cached

    from app.analysis import engines, tuning as tuning_probe
    from app.config import load_settings

    settings = load_settings()
    engines.register_builtins()
    chord_engine = engines.build_chord_engine(settings)
    beat_tracker = engines.build_beat_tracker(settings)
    onset_detector = engines.build_onset_detector(settings)
    structure_probe = engines.build_structure_probe(settings)

    media = fetch(slug, entry)
    started = time.monotonic()
    pcm, sample_rate = decode(media)

    pitch = tuning_probe.estimate(pcm, sample_rate)
    grid = beat_tracker.track(pcm, sample_rate)
    raw = chord_engine.analyze(pcm, sample_rate, tuning=pitch.correction)
    onsets = onset_detector.detect(pcm, sample_rate) if onset_detector else []
    energy = structure_probe.probe(pcm, sample_rate) if structure_probe else None
    del pcm

    payload = {
        "cacheVersion": CACHE_VERSION,
        "slug": slug,
        "videoId": entry["videoId"],
        "title": entry.get("title", slug),
        "durationS": round(len(grid.beats_ms) and (grid.beats_ms[-1] / 1000.0) or 0.0, 2),
        "engines": {"chords": str(EngineInfo(chord_engine.name, chord_engine.version)),
                    "beats": str(EngineInfo(beat_tracker.name, beat_tracker.version))},
        "tuning": {"semitones": pitch.semitones},
        "grid": asdict(grid),
        "raw": [asdict(s) for s in raw],
        "onsets": [asdict(o) for o in onsets],
        "energy": asdict(energy) if energy is not None else None,
        "elapsedS": round(time.monotonic() - started, 1),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return payload


def load_features(slug: str) -> dict:
    path = features_path(slug)
    if not path.exists():
        raise SystemExit(f"{slug}: no cached features — run `python bench/lab.py features`")
    cached = json.loads(path.read_text())
    if cached.get("cacheVersion") != CACHE_VERSION:
        raise SystemExit(f"{slug}: cache is version {cached.get('cacheVersion')}, "
                         f"expected {CACHE_VERSION} — re-run `features --force`")
    return cached


# --------------------------------------------------------------------------
# assembling
# --------------------------------------------------------------------------

def assemble(slug: str, entry: dict, cached: dict | None = None):
    """Cached features → an `AnalysisOutcome`, through the real `assemble`."""
    from app.analysis.pipeline import assemble as real_assemble
    from app.analysis.tuning import Tuning
    from app.config import load_settings

    cached = cached or load_features(slug)
    grid = BeatGrid(**cached["grid"])
    raw = [RawChordSpan(**s) for s in cached["raw"]]
    onsets = [Onset(**o) for o in cached["onsets"]]
    energy = EnergyCurve(**cached["energy"]) if cached.get("energy") else None
    duration_s = (grid.beats_ms[-1] / 1000.0) if grid.beats_ms else 0.0
    meta = VideoMeta(video_id=entry["videoId"], title=cached.get("title", slug),
                     duration_s=duration_s, channel_id="")
    chords_name, chords_version = cached["engines"]["chords"].split("@", 1)
    beats_name, beats_version = cached["engines"]["beats"].split("@", 1)
    return real_assemble(
        meta=meta, grid=grid, raw=raw, onsets=onsets, energy=energy,
        settings=load_settings(), tuning=Tuning(cached["tuning"]["semitones"]),
        chords_engine=EngineInfo(chords_name, chords_version),
        beats_engine=EngineInfo(beats_name, beats_version),
    )


def system_chart(slug: str, entry: dict, cached: dict | None = None):
    from app.chords import NORMAL
    from bench.chartref import chart_from_payload

    outcome = assemble(slug, entry, cached)
    song = outcome.songs.get(NORMAL) or next(iter(outcome.songs.values()))
    chart = chart_from_payload(song, title=entry.get("title", slug),
                               artist=entry.get("artist", ""))
    return chart, outcome


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def _selected(args) -> dict[str, dict]:
    book = songbook()
    if args.slug:
        missing = [s for s in args.slug if s not in book]
        if missing:
            raise SystemExit(f"not in the songbook: {', '.join(missing)}")
        return {s: book[s] for s in args.slug}
    return book


def cmd_fetch(args) -> int:
    for slug, entry in _selected(args).items():
        path = fetch(slug, entry, force=args.force)
        print(f"{slug:<28} {path.stat().st_size // 1024:>7} KiB  {entry['videoId']}")
    return 0


def cmd_features(args) -> int:
    for slug, entry in _selected(args).items():
        cached = build_features(slug, entry, force=args.force)
        print(f"{slug:<28} {len(cached['raw']):>5} spans  "
              f"{len(cached['grid']['beats_ms']):>5} beats  "
              f"{cached['grid']['bpm']:>6.1f} bpm  "
              f"{cached['tuning']['semitones'] * 100:+6.1f} cents  "
              f"({cached['elapsedS']}s)")
    return 0


def cmd_chart(args) -> int:
    from bench.chartref import render_chart

    for slug, entry in _selected(args).items():
        chart, _ = system_chart(slug, entry)
        print(render_chart(chart, wrap=args.width))
    return 0


def cmd_grade(args) -> int:
    from bench.grade_chart import diff_bars, grade, report
    from bench.chartref import load_chart, render_chart

    rows = []
    for slug, entry in _selected(args).items():
        path = REFERENCE / f"{slug}.chart"
        if not path.exists():
            print(f"{slug}: no reference chart at {path}")
            continue
        reference = load_chart(path)
        try:
            chart, outcome = system_chart(slug, entry)
        except Exception as error:                      # noqa: BLE001
            print(f"{slug:<28} FAILED: {type(error).__name__}: {error}")
            rows.append((slug, None))
            continue
        result = grade(reference, chart)
        rows.append((slug, result))
        if args.verbose or args.slug:
            print(report(result))
            if outcome.low_confidence:
                print(f"  low confidence: {'; '.join(outcome.low_confidence_reasons)}")
            if args.render:
                print()
                print(render_chart(chart, wrap=args.width))
            if args.diff:
                print()
                print(diff_bars(reference, chart, limit=args.limit))
            print()

    print(f"\n{'song':<28} {'root':>6} {'triad':>6} {'form':>6} {'bars':>11} "
          f"{'vocab':>9} {'key':>10}")
    scored = [r for _, r in rows if r]
    for slug, result in rows:
        if result is None:
            print(f"{slug:<28} {'—':>6} {'—':>6} {'—':>6}")
            continue
        s = result["scores"]
        print(f"{slug:<28} {s['root']:>6.3f} {s['triad']:>6.3f} {s['form']:>6.3f} "
              f"{result['bars']['system']:>5}/{result['bars']['reference']:<5} "
              f"{result['vocabulary']['systemCount']:>4}/{result['vocabulary']['referenceCount']:<4} "
              f"{'ok' if result['key']['tonicMatch'] else 'WRONG':>10}")
    if scored:
        print(f"{'MEAN':<28} "
              f"{sum(r['scores']['root'] for r in scored) / len(scored):>6.3f} "
              f"{sum(r['scores']['triad'] for r in scored) / len(scored):>6.3f} "
              f"{sum(r['scores']['form'] for r in scored) / len(scored):>6.3f}"
              f"    {len(scored)}/{len(rows)} scored")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("fetch", cmd_fetch), ("features", cmd_features),
                          ("chart", cmd_chart), ("grade", cmd_grade)):
        child = sub.add_parser(name)
        child.add_argument("--slug", action="append")
        child.add_argument("--force", action="store_true")
        child.add_argument("--diff", action="store_true")
        child.add_argument("--render", action="store_true")
        child.add_argument("--verbose", "-v", action="store_true")
        child.add_argument("--limit", type=int, default=64)
        child.add_argument("--width", type=int, default=4,
                           help="bars per printed line (display only; see render_chart)")
        child.set_defaults(handler=handler)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
