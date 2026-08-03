"""Build the *real-audio* half of the benchmark — §8 step 2's missing ingredient.

`synth.py` renders six specimens with exact ground truth and is honest that it
cannot tell you how two engines behave on a dense real mix. This script supplies
the tracks that can, by pairing:

- **Isophonics' Beatles annotations** (Centre for Digital Music, QMUL) — 180
  songs with time-aligned chord labels in Chris Harte's syntax, plus beat files
  whose second column marks the position in the bar, so `1` is a downbeat. That
  gives ground truth for *both* things being benchmarked, from an annotation set
  the MIR community has checked for twenty years.
- **the audio**, fetched the same way the service fetches it.

    python bench/fetch_corpus.py --annotations ~/…/iso   # build
    python bench/run_bench.py                            # score

Nothing this writes is committed: `bench/audio/` is gitignored in full. The
annotations are CC BY-SA-NC and the recordings are obviously not ours, so both
stay on the machine that ran the benchmark. Re-running reproduces the corpus from
`bench/corpus.json`, which pins the resolved video id per track so a later run
scores the *same* audio rather than whatever search returns that day.

--------------------------------------------------------------------------------
The alignment problem, and why this script is more than a downloader
--------------------------------------------------------------------------------

Isophonics timed its annotations against **specific 1987 CD issues**. YouTube
serves 2009/2015 remasters. Those are usually the same tape transfer at the same
speed — but not always, and an upload can carry leading silence besides. A
constant offset of even 300 ms would wreck a chord score and rank the engines by
who happens to lag, so:

1. **Estimate the offset.** The annotated beats should land on audible attacks,
   so shifting the beat grid across librosa's onset-strength envelope and taking
   the offset that maximises the envelope's value *at* the beats recovers it.
   Engine-independent — it never asks a candidate where the beats are — and
   applied identically to every candidate, so it cannot favour one.
2. **Detect drift.** The same estimate is run over the first and last third
   separately. A speed mismatch (a different master, a pitch-corrected upload)
   shows up as the two disagreeing; more than `--max-drift-ms` apart and the
   track is **rejected**, because no constant offset can fix it.
3. **Report the confidence.** A weak peak means the alignment is a guess, and a
   guessed alignment is worse than a missing track. Below `--min-confidence` the
   track is rejected too.

Rejections are printed with the reason. A smaller corpus that is correctly
aligned beats a larger one that is quietly 200 ms out.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

AUDIO = ROOT / "bench" / "audio"
MANIFEST = ROOT / "bench" / "corpus.json"

SAMPLE_RATE = 22050          # what §5.1 decodes to; the benchmark must match
HOP = 512                    # ~23 ms frames for the onset envelope


@dataclass(frozen=True)
class Track:
    """One benchmark specimen.

    `album`/`stem` locate the Isophonics annotation; `query` is the search that
    finds the recording. `note` says what this track is *for* — the corpus is
    chosen to span the axes the engine choice turns on (harmonic complexity,
    meter, tempo), not to be a Beatles playlist.
    """

    slug: str
    album: str
    stem: str
    query: str
    note: str


# Chosen from a profile of all 180 annotated songs: complexity (share of time on
# anything that isn't a plain triad) from 0.00 to 0.82, tempo 66–178 bpm, 3/4 and
# irregular bars alongside 4/4, vocabulary 3–27 distinct chords. Weighted toward
# the guitar-and-a-voice end, because that is what the app is for.
CORPUS: tuple[Track, ...] = (
    Track("love-me-do", "01_-_Please_Please_Me", "08_-_Love_Me_Do",
          "The Beatles Love Me Do Remastered 2009",
          "3 chords, no extensions — the floor. Any engine must get this."),
    Track("twist-and-shout", "01_-_Please_Please_Me", "14_-_Twist_And_Shout",
          "The Beatles Twist and Shout Remastered 2009",
          "3-chord rock, loud mix, 126bpm."),
    Track("a-hard-days-night", "03_-_A_Hard_Day's_Night", "01_-_A_Hard_Day's_Night",
          "The Beatles A Hard Day's Night Remastered 2009",
          "The famously ambiguous opening chord — a known engine disagreement."),
    Track("norwegian-wood", "06_-_Rubber_Soul", "02_-_Norwegian_Wood_(This_Bird_Has_Flown)",
          "The Beatles Norwegian Wood This Bird Has Flown Remastered 2009",
          "3/4 at 178bpm — the meter case. Downbeat tracking is the whole test."),
    Track("michelle", "06_-_Rubber_Soul", "07_-_Michelle",
          "The Beatles Michelle Remastered 2009",
          "Chromatic, 52% non-triad — where a template engine should fall apart."),
    Track("in-my-life", "06_-_Rubber_Soul", "11_-_In_My_Life",
          "The Beatles In My Life Remastered 2009",
          "Mid-complexity diatonic pop at 103bpm."),
    Track("taxman", "07_-_Revolver", "01_-_Taxman",
          "The Beatles Taxman Remastered 2009",
          "82% non-triad, dominant-7 heavy on a tiny vocabulary — extension stress."),
    Track("yellow-submarine", "07_-_Revolver", "06_-_Yellow_Submarine",
          "The Beatles Yellow Submarine Remastered 2009",
          "Simple singalong, but a cluttered mix with sound effects."),
    Track("ob-la-di", "10CD1_-_The_Beatles", "CD1_-_04_-_Ob-La-Di,_Ob-La-Da",
          "The Beatles Ob-La-Di Ob-La-Da Remastered 2009",
          "Flat key (Bb) — catches engines biased toward guitar-friendly sharps."),
    Track("something", "11_-_Abbey_Road", "02_-_Something",
          "The Beatles Something Remastered 2009",
          "27 distinct chords at 66bpm — the hardest common-repertoire case."),
    Track("here-comes-the-sun", "11_-_Abbey_Road", "07_-_Here_Comes_The_Sun",
          "The Beatles Here Comes The Sun Remastered 2009",
          "Guitar staple with irregular bars (only 73% of bars are 4)."),
    Track("let-it-be", "12_-_Let_It_Be", "06_-_Let_It_Be",
          "The Beatles Let It Be Remastered 2009",
          "The campfire song, literally: C G Am F at 70bpm."),
)


# --- Isophonics parsing ------------------------------------------------------

def read_lab(path: Path) -> list[tuple[float, float, str]]:
    """`start end label`, whitespace delimited (WaveSurfer .lab)."""
    spans = []
    for line in path.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            spans.append((float(parts[0]), float(parts[1]), " ".join(parts[2:])))
        except ValueError:
            continue
    return spans


def read_beats(path: Path) -> list[tuple[float, int]]:
    """`time position-in-bar`, where position 1 is a downbeat.

    A handful of files carry a non-numeric position for an unclear bar; those
    lines are dropped rather than guessed at.
    """
    out = []
    for line in path.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            out.append((float(parts[0]), int(float(parts[1]))))
        except ValueError:
            continue
    return out


def modal_meter(beats: list[tuple[float, int]]) -> tuple[int, float]:
    """The most common number of beats between downbeats, and how dominant it is.

    Mode rather than max: an annotation with one 5-beat bar in an otherwise 4/4
    song is a 4/4 song, and taking the maximum would call it 5/4.
    """
    downs = [t for t, pos in beats if pos == 1]
    if len(downs) < 5:
        return 4, 0.0
    counts = []
    for start, end in zip(downs, downs[1:]):
        n = sum(1 for t, _ in beats if start <= t < end)
        if 1 < n <= 13:
            counts.append(n)
    if not counts:
        return 4, 0.0
    tally = Counter(counts)
    top, hits = tally.most_common(1)[0]
    return top, hits / len(counts)


# --- alignment ---------------------------------------------------------------

# Semitones above the root per quality, for turning a chord label back into the
# pitch classes it sounds. Only needs to be good enough to correlate against a
# chromagram, so extensions collapse onto their triad plus the characteristic
# tone.
_QUALITY_PCS = {
    "major": (0, 4, 7), "minor": (0, 3, 7), "dominant7": (0, 4, 7, 10),
    "major7": (0, 4, 7, 11), "minor7": (0, 3, 7, 10), "diminished": (0, 3, 6),
    "diminished7": (0, 3, 6, 9), "halfDiminished7": (0, 3, 6, 10),
    "augmented": (0, 4, 8), "sus4": (0, 5, 7), "sus2": (0, 2, 7),
}


def _label_pitch_classes(label: str) -> tuple[int, ...]:
    """Harte label → the pitch classes it sounds, via the app's own normalizer."""
    from app.chords import normalize

    parsed = normalize(label)
    if parsed is None:
        return ()
    root, quality = parsed[0], parsed[1]
    return tuple((root + i) % 12 for i in _QUALITY_PCS.get(quality, (0, 4, 7)))


def onset_envelope(pcm, sample_rate: int):
    import librosa
    return librosa.onset.onset_strength(y=pcm, sr=sample_rate, hop_length=HOP)


def _chroma_lag(pcm, sample_rate: int, spans: list[tuple[float, float, str]],
                search_s: float, hop: int = 2048,
                center_s: float = 0.0) -> tuple[float, float]:
    """Coarse offset, by correlating the audio's chroma against the annotation's.

    **Why not onsets.** The obvious approach — slide the annotated beat grid
    until it sits on audible attacks — is periodic at the beat period, so a shift
    of exactly one beat scores almost as well as the truth. The search then picks
    an arbitrary alias, and two halves of the same song can pick *different*
    ones, which reads as drift that isn't there. (That is not hypothetical: it is
    what the first version of this script reported.)

    A chord progression has no such periodicity — `C G Am F` at lag 0 and at lag
    one-beat are genuinely different harmonies — so correlating the *harmony*
    gives one unambiguous peak. Resolution is only the hop (~93 ms); the caller
    refines from there.
    """
    import librosa
    import numpy as np

    chroma = librosa.feature.chroma_cqt(y=pcm, sr=sample_rate, hop_length=hop)
    chroma = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9)
    frame_s = hop / sample_rate
    frames = chroma.shape[1]

    reference = np.zeros((12, frames), dtype="float32")
    for start, end, label in spans:
        pcs = _label_pitch_classes(label)
        if not pcs:
            continue
        first = max(0, int(start / frame_s))
        last = min(frames, int(end / frame_s))
        if last <= first:
            continue
        for pc in pcs:
            reference[pc, first:last] = 1.0
    reference = reference / (np.linalg.norm(reference, axis=0, keepdims=True) + 1e-9)

    centre = int(round(center_s / frame_s))
    span = int(search_s / frame_s)
    lags = list(range(centre - span, centre + span + 1))
    scores = []
    for lag in lags:
        if lag >= 0:
            a, b = chroma[:, lag:], reference[:, : frames - lag]
        else:
            a, b = chroma[:, : frames + lag], reference[:, -lag:]
        width = min(a.shape[1], b.shape[1])
        scores.append(float((a[:, :width] * b[:, :width]).sum() / width) if width else 0.0)

    best = max(range(len(scores)), key=lambda i: scores[i])
    # Confidence as the peak against the landscape *away* from it. Comparing to
    # the median punishes a broad-but-correct peak; comparing to the far field
    # asks the question that matters — is this lag distinctly better than an
    # unrelated one?
    far = [s for i, s in enumerate(scores) if abs(lags[i] - lags[best]) * frame_s > 1.0]
    baseline = (sum(far) / len(far)) if far else (statistics.median(scores) or 1e-9)
    return lags[best] * frame_s, scores[best] / (baseline or 1e-9)


def _refine_with_onsets(envelope, beat_times: list[float], frame_s: float,
                        around_s: float, window_s: float,
                        step_s: float = 0.01) -> float:
    """Sharpen a coarse offset to ~10 ms by sitting the beats on the attacks.

    Safe here precisely because it is bounded: the window is under half a beat,
    so the aliasing that makes this useless as a global search cannot reach the
    neighbouring period.
    """
    if not beat_times:
        return around_s
    limit = len(envelope)
    steps = int(2 * window_s / step_s) + 1
    best_offset, best_score = around_s, -1.0
    for i in range(steps):
        offset = around_s - window_s + i * step_s
        total, counted = 0.0, 0
        for t in beat_times:
            frame = int(round((t + offset) / frame_s))
            if 0 <= frame < limit:
                total += float(envelope[frame])
                counted += 1
        if counted < 0.6 * len(beat_times):
            continue
        score = total / counted
        if score > best_score:
            best_offset, best_score = offset, score
    return best_offset


def _region_lag(pcm, sample_rate: int, spans: list[tuple[float, float, str]],
                start_s: float, end_s: float, search_s: float,
                center_s: float = 0.0) -> tuple[float, float] | None:
    """`_chroma_lag` restricted to one stretch of the song.

    The audio slice and the annotation slice are cut from the same origin, so the
    lag it returns is directly comparable with any other region's.
    """
    pad = search_s + abs(center_s)
    origin = max(0.0, start_s - pad)
    segment = pcm[int(origin * sample_rate): int((end_s + pad) * sample_rate)]
    if len(segment) < 2 * sample_rate:
        return None
    local = [(max(0.0, s - origin), e - origin, label)
             for s, e, label in spans if e > start_s and s < end_s]
    if len(local) < 3:
        return None
    # Finer hop than the global pass: the search band is narrow, so the extra
    # frames are cheap, and lag quantisation is what limits the fit's residual.
    return _chroma_lag(segment, sample_rate, local, search_s, hop=1024,
                       center_s=center_s)


@dataclass(frozen=True)
class Alignment:
    """How the annotation's clock maps onto this recording's.

    `at(t) = t + offset_s + rate_error * t` — a shift *and* a stretch, because
    the two clocks differ in both. Remasters are usually the same transfer at the
    same speed, but "usually" isn't "always", and a 0.1% speed difference is
    inaudible while putting the last chorus a quarter-second out.
    """

    offset_s: float
    rate_error: float
    confidence: float
    residual_ms: float
    drift_ms: float

    def at(self, t: float) -> float:
        return t + self.offset_s + self.rate_error * t


def align(pcm, sample_rate: int, spans: list[tuple[float, float, str]],
          beat_times: list[float], search_s: float, windows: int = 6) -> Alignment:
    """Fit the annotation clock to the recording clock.

    1. One global chroma correlation finds the right neighbourhood (and, being
       harmonic rather than rhythmic, the right beat-period alias).
    2. `windows` local correlations, each searching only a narrow band around
       that, give lag-versus-time samples.
    3. A least-squares line through them is the shift and the stretch. Its
       **residual** is the honesty check: a good fit means one clock really does
       map onto the other linearly, and a bad one means this is the wrong
       recording, or an edit, and no correction will save it.
    4. The intercept is finally sharpened against the onset envelope, inside a
       window too narrow to reach a neighbouring beat.
    """
    duration = spans[-1][1] if spans else 0.0
    coarse, confidence = _chroma_lag(pcm, sample_rate, spans, search_s)

    samples: list[tuple[float, float]] = []
    for index in range(windows):
        start = duration * index / windows
        end = duration * (index + 1) / windows
        found = _region_lag(pcm, sample_rate, spans, start, end,
                            search_s=0.75, center_s=coarse)
        if found is not None:
            samples.append(((start + end) / 2, found[0]))

    offset, rate_error, residual_ms = coarse, 0.0, float("inf")
    if len(samples) >= 4:
        n = len(samples)
        mean_t = sum(t for t, _ in samples) / n
        mean_lag = sum(l for _, l in samples) / n
        variance = sum((t - mean_t) ** 2 for t, _ in samples)
        if variance > 0:
            rate_error = sum((t - mean_t) * (l - mean_lag) for t, l in samples) / variance
            offset = mean_lag - rate_error * mean_t
        residual_ms = 1000 * (
            sum((l - (offset + rate_error * t)) ** 2 for t, l in samples) / n
        ) ** 0.5

    drift_ms = abs(rate_error) * duration * 1000

    # Sharpen the intercept only: the stretch is already fixed by the fit, and
    # the onset envelope has nothing to say about it.
    deltas = [b - a for a, b in zip(beat_times, beat_times[1:]) if 0.1 < b - a < 2.0]
    beat_period = statistics.median(deltas) if deltas else 0.5
    envelope = onset_envelope(pcm, sample_rate)
    warped = [t + rate_error * t for t in beat_times]
    offset = _refine_with_onsets(envelope, warped, HOP / sample_rate,
                                 around_s=offset, window_s=0.4 * beat_period)

    return Alignment(offset, rate_error, confidence, residual_ms, drift_ms)


# --- fetch -------------------------------------------------------------------

def resolve_video(query: str, expected_s: float, tolerance_s: float) -> dict | None:
    """Search, then take the closest duration match — preferring official uploads.

    Duration matching is doing real work here: it rejects live takes, covers,
    edits and the "full album" uploads that a title search otherwise returns, all
    of which would be annotated-to-the-wrong-recording.
    """
    result = subprocess.run(
        [sys.executable, "-m", "yt_dlp", f"ytsearch8:{query}", "--flat-playlist",
         "--dump-json", "--no-warnings"],
        capture_output=True, text=True, timeout=180,
    )
    candidates = []
    for line in result.stdout.splitlines():
        try:
            info = json.loads(line)
        except json.JSONDecodeError:
            continue
        duration = info.get("duration")
        if not duration:
            continue
        delta = abs(duration - expected_s)
        if delta > tolerance_s:
            continue
        channel = (info.get("channel") or "")
        official = channel.strip().lower() in {"the beatles", "the beatles - topic"}
        candidates.append((not official, delta, info))
    if not candidates:
        return None
    candidates.sort(key=lambda c: (c[0], c[1]))
    return candidates[0][2]


def download_wav(video_id: str, destination: Path) -> None:
    """Fetch and decode to the sample rate the service uses.

    Deliberately the same shape as the production path (yt-dlp fetches, ffmpeg
    decodes): benchmarking a different decode than the one that will run in
    production would measure the wrong thing.
    """
    with tempfile.TemporaryDirectory(prefix="bench-fetch-") as tmp:
        media = Path(tmp) / "audio"
        subprocess.run(
            [sys.executable, "-m", "yt_dlp", "-f", "bestaudio", "--no-playlist",
             "--no-warnings", "-o", str(media) + ".%(ext)s",
             f"https://www.youtube.com/watch?v={video_id}"],
            check=True, capture_output=True, text=True, timeout=900,
        )
        downloaded = next(p for p in Path(tmp).iterdir() if p.name.startswith("audio"))
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(downloaded),
             "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(destination)],
            check=True, capture_output=True, text=True, timeout=600,
        )


# --- truth -------------------------------------------------------------------

def build_truth(track: Track, annotations: Path, alignment: Alignment,
                duration_s: float) -> dict:
    """Isophonics annotations → the harness's `<name>.truth.json`, shifted.

    `N` (no chord) spans are dropped rather than emitted: the app has no rest
    chord (§5.4 holds the previous one), the scorer would crash trying to
    normalize the label, and scoring silence tells you nothing about an engine.
    """
    chord_path = annotations / "chordlab" / "The Beatles" / track.album / f"{track.stem}.lab"
    beat_path = annotations / "beat" / "The Beatles" / track.album / f"{track.stem}.txt"

    spans = read_lab(chord_path)
    beats = read_beats(beat_path) if beat_path.exists() else []

    def shift(t: float) -> int:
        return int(round(alignment.at(t) * 1000))

    limit_ms = int(round(duration_s * 1000))
    chords = [
        {"startMs": shift(start), "endMs": shift(end), "name": label}
        for start, end, label in spans
        if label != "N" and shift(end) > 0 and shift(start) < limit_ms
    ]
    beats_ms = [shift(t) for t, _ in beats if 0 <= shift(t) < limit_ms]
    downbeats_ms = [shift(t) for t, pos in beats if pos == 1 and 0 <= shift(t) < limit_ms]

    meter, meter_confidence = modal_meter(beats)
    deltas = [b - a for (a, _), (b, _) in zip(beats, beats[1:]) if 0.1 < b - a < 2.0]
    tempo = 60.0 / statistics.median(deltas) if deltas else 0.0

    return {
        "name": track.slug,
        "source": "isophonics-beatles",
        "note": track.note,
        "duration_ms": limit_ms,
        "tempo": round(tempo, 2),
        "time_signature": f"{meter}/4",
        "meter_confidence": round(meter_confidence, 3),
        "alignment_offset_ms": int(round(alignment.offset_s * 1000)),
        "alignment_rate_error": round(alignment.rate_error, 6),
        "alignment_residual_ms": round(alignment.residual_ms, 1),
        "beats_ms": beats_ms,
        "downbeats_ms": downbeats_ms,
        "chords": chords,
    }


# --- driver ------------------------------------------------------------------

@dataclass
class Outcome:
    slug: str
    status: str
    detail: str = ""
    fields: dict = field(default_factory=dict)


def process(track: Track, annotations: Path, manifest: dict, args) -> Outcome:
    import soundfile as sf

    chord_path = annotations / "chordlab" / "The Beatles" / track.album / f"{track.stem}.lab"
    if not chord_path.exists():
        return Outcome(track.slug, "no-annotation", str(chord_path))

    spans = read_lab(chord_path)
    if not spans:
        return Outcome(track.slug, "empty-annotation")
    annotated_s = spans[-1][1]

    wav = AUDIO / f"{track.slug}.wav"
    pinned = manifest.get(track.slug, {}).get("videoId")

    if not wav.exists() or args.refetch:
        video_id = pinned
        if video_id is None:
            info = resolve_video(track.query, annotated_s, args.duration_tolerance)
            if info is None:
                return Outcome(track.slug, "no-match",
                               f"no result within {args.duration_tolerance}s of {annotated_s:.0f}s")
            video_id = info["id"]
        try:
            download_wav(video_id, wav)
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or "").strip().splitlines()
            return Outcome(track.slug, "fetch-failed", detail[-1] if detail else str(exc))
        except subprocess.TimeoutExpired:
            return Outcome(track.slug, "fetch-timeout")
        manifest.setdefault(track.slug, {})["videoId"] = video_id

    pcm, rate = sf.read(str(wav), dtype="float32")
    duration_s = len(pcm) / rate
    if abs(duration_s - annotated_s) > args.duration_tolerance:
        return Outcome(track.slug, "duration-mismatch",
                       f"audio {duration_s:.1f}s vs annotation {annotated_s:.1f}s")

    beat_path = annotations / "beat" / "The Beatles" / track.album / f"{track.stem}.txt"
    if not beat_path.exists():
        return Outcome(track.slug, "no-beats")
    beat_times = [t for t, _ in read_beats(beat_path)]
    if len(beat_times) < 20:
        return Outcome(track.slug, "too-few-beats", str(len(beat_times)))

    alignment = align(pcm, rate, spans, beat_times, args.search_s)

    if alignment.confidence < args.min_confidence:
        return Outcome(track.slug, "weak-alignment",
                       f"peak only {alignment.confidence:.2f}x the far field")
    # The residual, not the drift, is the reject criterion: drift that the linear
    # fit *explains* is corrected, and drift it cannot explain is what makes a
    # track unusable.
    if alignment.residual_ms > args.max_residual_ms:
        return Outcome(track.slug, "nonlinear-alignment",
                       f"residual {alignment.residual_ms:.0f}ms after fit")
    if abs(alignment.rate_error) > args.max_rate_error:
        return Outcome(track.slug, "speed-mismatch",
                       f"{alignment.rate_error*100:+.2f}% — a different recording, not a master")

    truth = build_truth(track, annotations, alignment, duration_s)
    if len(truth["chords"]) < 4 or len(truth["downbeats_ms"]) < 4:
        return Outcome(track.slug, "thin-truth")
    (AUDIO / f"{track.slug}.truth.json").write_text(json.dumps(truth, indent=2) + "\n")

    manifest.setdefault(track.slug, {}).update({
        "album": track.album, "stem": track.stem, "note": track.note,
        "offsetMs": truth["alignment_offset_ms"],
        "rateError": truth["alignment_rate_error"],
        "residualMs": truth["alignment_residual_ms"],
        "alignmentConfidence": round(alignment.confidence, 2),
        "driftMs": round(alignment.drift_ms),
        "timeSignature": truth["time_signature"],
        "tempo": truth["tempo"],
        "chords": len(truth["chords"]),
    })
    return Outcome(track.slug, "ok", "", {
        "offset": truth["alignment_offset_ms"], "conf": alignment.confidence,
        "drift": alignment.drift_ms, "residual": alignment.residual_ms,
        "meter": truth["time_signature"], "chords": len(truth["chords"]),
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--annotations", required=True, type=Path,
                        help="Isophonics Beatles annotation root (contains chordlab/ and beat/)")
    parser.add_argument("--only", nargs="*", help="slugs to build (default: all)")
    parser.add_argument("--refetch", action="store_true", help="re-download even if the wav exists")
    parser.add_argument("--search-s", type=float, default=5.0,
                        help="offset search half-window, seconds")
    parser.add_argument("--duration-tolerance", type=float, default=4.0)
    # Deliberately weak. A harmonically static song (a drone, a two-chord vamp)
    # has a genuinely broad global peak while still being perfectly aligned, so
    # the real evidence is the residual below: six local measurements landing on
    # one straight line is far harder to fake than one tall peak.
    parser.add_argument("--min-confidence", type=float, default=1.10,
                        help="reject when the alignment peak is this flat vs the far field")
    parser.add_argument("--max-residual-ms", type=float, default=120.0,
                        help="reject when a shift+stretch cannot explain the lag samples")
    parser.add_argument("--max-rate-error", type=float, default=0.01,
                        help="reject a speed difference beyond this (1%% = a different recording)")
    args = parser.parse_args()

    if not (args.annotations / "chordlab").is_dir():
        print(f"! {args.annotations} has no chordlab/ — point --annotations at the "
              f"extracted Isophonics 'The Beatles Annotations' root.")
        return 2

    AUDIO.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg") is None:
        print("! ffmpeg not found on PATH — needed to decode to 22.05 kHz mono.")
        return 2

    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    tracks = [t for t in CORPUS if not args.only or t.slug in args.only]

    outcomes = []
    for index, track in enumerate(tracks, 1):
        print(f"[{index}/{len(tracks)}] {track.slug} … ", end="", flush=True)
        outcome = process(track, args.annotations, manifest, args)
        outcomes.append(outcome)
        if outcome.status == "ok":
            f = outcome.fields
            print(f"ok  offset {f['offset']:+6d}ms  peak {f['conf']:.2f}x  "
                  f"drift {f['drift']:4.0f}ms corrected (residual {f['residual']:3.0f}ms)  "
                  f"{f['meter']:>5}  {f['chords']:3d} chords")
        else:
            print(f"REJECTED ({outcome.status}) {outcome.detail}")

    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    kept = [o for o in outcomes if o.status == "ok"]
    print(f"\n{len(kept)}/{len(outcomes)} usable → bench/audio/, pinned in {MANIFEST.name}")
    if len(outcomes) - len(kept):
        print("Rejected tracks are excluded on purpose: a misaligned specimen "
              "ranks engines by who lags, not by who is right.")
    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
