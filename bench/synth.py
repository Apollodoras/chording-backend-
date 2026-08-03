"""Synthetic test audio with **exact** ground truth — §8 step 2's test material.

Renders a known progression, at a known tempo, strummed on a known pattern, into
a WAV plus a JSON file saying precisely where every beat, downbeat, chord and
onset is. That makes the benchmark scoreable rather than impressionistic: a beat
tracker is right or wrong to the millisecond, and a chord engine is right or
wrong per bar.

**What this can and cannot tell you**, stated plainly because it decides how much
weight the numbers deserve:

- It *can* prove the plumbing — that a tracker's output lands on the grid the
  quantizer expects, that an engine's labels survive normalization, that
  onsets fold onto the right subdivisions, that the emitted song lints clean.
- It *cannot* tell you how BTC and Chordino behave on a dense real mix, which is
  exactly where they diverge and exactly what the engine choice hinges on. Real
  tracks are still needed before that call.

No external dependencies: plucked-string tones are additive sine partials with an
exponential decay, written through the stdlib `wave` module. That keeps it
license-clean, deterministic, and runnable before any engine is installed.
"""

from __future__ import annotations

import json
import math
import struct
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.chords import normalize  # noqa: E402

SAMPLE_RATE = 22050          # what §5.1 decodes to
A4_HZ = 440.0
A4_MIDI = 69

# Semitones above the root for each quality this generator can voice. Enough to
# tell a chord engine's answer apart; not a voicing model.
_INTERVALS = {
    "major": (0, 4, 7), "minor": (0, 3, 7),
    "dominant7": (0, 4, 7, 10), "major7": (0, 4, 7, 11), "minor7": (0, 3, 7, 10),
    "diminished": (0, 3, 6), "diminished7": (0, 3, 6, 9), "halfDiminished7": (0, 3, 6, 10),
    "augmented": (0, 4, 8), "sus4": (0, 5, 7), "sus2": (0, 2, 7),
}

# A strummed guitar's partials, roughly: fundamental strongest, odd harmonics
# present, everything rolling off. Enough spectral content for a chroma-based
# engine to have something to work with.
_PARTIALS = ((1, 1.0), (2, 0.45), (3, 0.30), (4, 0.18), (5, 0.12), (6, 0.07))

DDUUDU = (0.0, 1.0, 1.5, 2.5, 3.0, 3.5)
QUARTERS = (0.0, 1.0, 2.0, 3.0)


@dataclass
class Spec:
    """One song to render."""

    name: str
    chords: list[str]                       # one per bar, app-grammar names
    tempo: int = 120
    bar_beats: int = 4
    repeats: int = 4
    pattern: tuple[float, ...] = DDUUDU
    octave_root: int = 52                   # E3-ish — where a guitar actually sits
    noise: float = 0.0                      # 0…1, adds broadband hiss


@dataclass
class Truth:
    """Everything the benchmark scores against, in the pipeline's own units."""

    name: str
    tempo: int
    time_signature: str
    duration_ms: int
    beats_ms: list[int] = field(default_factory=list)
    downbeats_ms: list[int] = field(default_factory=list)
    chords: list[dict] = field(default_factory=list)     # {startMs, endMs, name}
    onsets_ms: list[int] = field(default_factory=list)
    pattern_beats: list[float] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, sort_keys=True) + "\n"


def midi_hz(note: int) -> float:
    return A4_HZ * (2.0 ** ((note - A4_MIDI) / 12.0))


def voice(name: str, octave_root: int) -> list[int]:
    """A chord name → MIDI notes, in a fixed low-ish register."""
    parsed = normalize(name)
    if parsed is None:
        raise ValueError(f"{name!r} is not a chord")
    root_pc, quality, _ = parsed
    root = octave_root + ((root_pc - octave_root) % 12)
    return [root + interval for interval in _INTERVALS[quality]]


def render(spec: Spec) -> tuple[list[float], Truth]:
    """Render one spec to a mono float buffer plus its ground truth."""
    seconds_per_beat = 60.0 / spec.tempo
    bars = spec.chords * spec.repeats
    total_beats = len(bars) * spec.bar_beats
    total_samples = int(total_beats * seconds_per_beat * SAMPLE_RATE) + SAMPLE_RATE
    buffer = [0.0] * total_samples

    truth = Truth(
        name=spec.name, tempo=spec.tempo,
        time_signature=f"{spec.bar_beats}/4",
        duration_ms=int(total_beats * seconds_per_beat * 1000),
        pattern_beats=list(spec.pattern),
    )
    truth.beats_ms = [int(b * seconds_per_beat * 1000) for b in range(total_beats + 1)]
    truth.downbeats_ms = [int(b * seconds_per_beat * 1000)
                          for b in range(0, total_beats + 1, spec.bar_beats)]

    for bar_index, chord_name in enumerate(bars):
        bar_start_beat = bar_index * spec.bar_beats
        notes = voice(chord_name, spec.octave_root)
        truth.chords.append({
            "startMs": int(bar_start_beat * seconds_per_beat * 1000),
            "endMs": int((bar_start_beat + spec.bar_beats) * seconds_per_beat * 1000),
            "name": chord_name,
        })
        for offset in spec.pattern:
            beat = bar_start_beat + offset
            start_ms = int(beat * seconds_per_beat * 1000)
            truth.onsets_ms.append(start_ms)
            # An upstroke rakes high-to-low, so its lowest string speaks last —
            # audible as a slightly later, slightly quieter attack.
            upstroke = round((offset - int(offset)) * 2) % 2 == 1
            _pluck(buffer, notes, start_ms, seconds_per_beat, upstroke=upstroke)

    _normalize(buffer)
    if spec.noise > 0:
        _add_noise(buffer, spec.noise)
    return buffer, truth


def _pluck(buffer: list[float], notes: list[int], start_ms: int,
           seconds_per_beat: float, *, upstroke: bool) -> None:
    """One strum: each string struck in turn, a few milliseconds apart.

    The stagger is not decoration — it is what makes the attack read as a strum
    rather than a block chord, and it is the same micro-stagger the app's own
    rake uses (CLAUDE.md §6.2).
    """
    ordered = sorted(notes, reverse=True) if upstroke else sorted(notes)
    gain = 0.7 if upstroke else 1.0
    decay_s = max(0.35, seconds_per_beat * 1.6)
    for index, note in enumerate(ordered):
        offset_samples = int((start_ms / 1000.0 + index * 0.008) * SAMPLE_RATE)
        _tone(buffer, midi_hz(note), offset_samples, decay_s, gain)


def _tone(buffer: list[float], hz: float, start: int, decay_s: float, gain: float) -> None:
    length = int(decay_s * SAMPLE_RATE)
    if start >= len(buffer):
        return
    length = min(length, len(buffer) - start)
    two_pi_f = 2.0 * math.pi * hz / SAMPLE_RATE
    for n in range(length):
        envelope = math.exp(-3.5 * n / (decay_s * SAMPLE_RATE))
        sample = 0.0
        for harmonic, weight in _PARTIALS:
            if hz * harmonic >= SAMPLE_RATE / 2:      # above Nyquist: would alias
                break
            sample += weight * math.sin(two_pi_f * harmonic * n)
        buffer[start + n] += gain * envelope * sample * 0.08


def _normalize(buffer: list[float]) -> None:
    peak = max((abs(x) for x in buffer), default=0.0)
    if peak > 0:
        scale = 0.89 / peak
        for i in range(len(buffer)):
            buffer[i] *= scale


def _add_noise(buffer: list[float], amount: float) -> None:
    """Deterministic pseudo-noise — a fixed LCG, so a "noisy" fixture is the same
    noisy fixture on every machine."""
    state = 0x2545F491
    for i in range(len(buffer)):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        buffer[i] += amount * ((state / 0x3FFFFFFF) - 1.0) * 0.05


def write_wav(path: Path, buffer: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        frames = b"".join(struct.pack("<h", int(max(-1.0, min(1.0, x)) * 32767))
                          for x in buffer)
        out.writeframes(frames)


# The benchmark set. Each one isolates something the engines are actually being
# compared on, rather than being "some music".
SPECS = [
    # The baseline: a folk progression anybody would recognise.
    Spec(name="folk-g-d-em-c", chords=["G", "D", "Em", "C"]),
    # Sevenths — where an engine that only knows triads shows itself, and where
    # the `normal` vs `hard` tiers actually differ.
    Spec(name="sevenths-ii-v-i", chords=["Dm7", "G7", "Cmaj7", "Cmaj7"], tempo=96),
    # A minor key with a flat spelling, to catch a key-finder that reports the
    # relative major (the classic failure) and a speller that emits "A#m".
    Spec(name="minor-bb", chords=["Bbm", "Gb", "Db", "Ab"], tempo=84),
    # 3/4, because meter detection is the thing most likely to be silently wrong
    # and most damaging when it is: a wrong bar length breaks §13.2's axis.
    Spec(name="waltz-3-4", chords=["C", "Am", "F", "G"], tempo=132,
         bar_beats=3, pattern=(0.0, 1.0, 2.0)),
    # Two chords per bar: the case that forces bars mode rather than flat.
    Spec(name="fast-changes", chords=["C", "G", "Am", "F"], tempo=144,
         pattern=QUARTERS, repeats=6),
    # The same song under noise — a crude stand-in for a dense mix, and the one
    # place this set gestures at what only real tracks can answer.
    Spec(name="folk-g-d-em-c-noisy", chords=["G", "D", "Em", "C"], noise=0.6),
]


def main() -> int:
    out_dir = ROOT / "bench" / "audio"
    for spec in SPECS:
        buffer, truth = render(spec)
        write_wav(out_dir / f"{spec.name}.wav", buffer)
        (out_dir / f"{spec.name}.truth.json").write_text(truth.to_json())
        print(f"rendered {spec.name}.wav  ({truth.duration_ms / 1000:.1f}s, "
              f"{len(truth.chords)} bars, {len(truth.onsets_ms)} onsets)")
    print(f"\n→ {out_dir}")
    print("These prove the plumbing. The engine choice still needs real tracks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
