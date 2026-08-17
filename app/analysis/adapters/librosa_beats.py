"""librosa beat tracking and onset detection — the always-available option.

librosa finds the pulse well and has no opinion about the bar, which is exactly
the wrong shape for this service: §13.2's anchors *are* downbeats, and a tracker
that nails every beat while guessing the "one" produces a sidecar that walks the
cursor off the song. So the beat tracker here is librosa's, and the downbeat
half is written on top of it.

**How the bar is recovered.** Given beats and no bar, two things have to be
chosen: the meter, and which beat is the first of the bar. Both are scored the
same way — a downbeat is where a bar's worth of evidence lands:

- **attack strength** — bars usually start with one, and
- **harmonic change** — a chord change is far likelier on a downbeat than
  mid-bar, so chroma flux is the second, independent vote.

Both are z-scored across all beats so neither can dominate by unit, then every
`(meter, phase)` candidate is scored by its mean. Comparing *means* is what makes
3 and 4 comparable at all: a sum would hand it to whichever meter produced more
candidates. 4 wins ties, because it usually should.

This is a heuristic and is meant to be beaten. Its purpose in the benchmark is to
put a number on what a dedicated downbeat model is worth — if `beat_this` can't
beat this, it isn't earning a torch dependency in the worker image.
"""

from __future__ import annotations

from ..types import PCM, BeatGrid, Onset

HOP = 512               # ~23 ms — onset-rate, unlike the chord engine's hop
_METERS = (4, 3)        # what to consider; 4 first so it wins ties


class LibrosaBeatTracker:
    """`BeatTracker` — librosa's pulse, plus a scored bar hypothesis."""

    name = "librosa"
    version = "1.0"

    def track(self, pcm: PCM, sr: int) -> BeatGrid:
        import librosa
        import numpy as np

        envelope = librosa.onset.onset_strength(y=pcm, sr=sr, hop_length=HOP)
        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=envelope, sr=sr, hop_length=HOP, units="frames"
        )
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP)
        if len(beat_times) < 4:
            return BeatGrid(beats_ms=[], downbeats_ms=[], bpm=float(np.atleast_1d(tempo)[0]),
                            confidence=0.0)

        chroma = librosa.feature.chroma_cqt(y=pcm, sr=sr, hop_length=HOP)
        meter, phase, margin = _choose_bar(np, envelope, chroma, beat_frames)

        beats_ms = [int(round(t * 1000)) for t in beat_times]
        downbeats_ms = beats_ms[phase::meter]

        # Confidence from how regular the pulse is: the standard deviation of the
        # inter-beat interval over its mean. A steady song lands near 0.
        intervals = np.diff(beat_times)
        regularity = 1.0 - min(1.0, float(np.std(intervals) / (np.mean(intervals) + 1e-9)))
        confidence = max(0.0, min(1.0, 0.6 * regularity + 0.4 * margin))

        return BeatGrid(
            beats_ms=beats_ms,
            downbeats_ms=downbeats_ms,
            bpm=float(np.atleast_1d(tempo)[0]),
            confidence=confidence,
            time_signature=f"{meter}/4",
        )


def _choose_bar(np, envelope, chroma, beat_frames) -> tuple[int, int, float]:
    """Pick `(meter, phase)`, and report how clearly it won.

    The margin is the winner over the runner-up *within the same meter* — i.e.
    "is this the right one of the four beats", which is the question a wrong
    answer gets wrong. Comparing against the best rival meter instead would call
    a confident 3/4 song ambiguous just because 4/4 also scored well.
    """
    frames = len(envelope)
    safe = [min(int(f), frames - 1) for f in beat_frames]

    attack = _zscore(np, np.asarray([envelope[f] for f in safe], dtype="float64"))

    # Chroma flux at each beat: how much the harmony moved going into it.
    flux = np.zeros(len(safe))
    for index, frame in enumerate(safe):
        if index == 0:
            continue
        previous = chroma[:, safe[index - 1]:frame]
        current = chroma[:, frame:safe[index + 1]] if index + 1 < len(safe) else chroma[:, frame:]
        if previous.size and current.size:
            flux[index] = float(np.linalg.norm(
                current.mean(axis=1) - previous.mean(axis=1)))
    flux = _zscore(np, flux)

    evidence = attack + flux
    best = (None, None, -1e9, -1e9)     # meter, phase, score, runner_up
    for meter in _METERS:
        scored = []
        for phase in range(meter):
            picks = evidence[phase::meter]
            scored.append((float(picks.mean()) if len(picks) else -1e9, phase))
        scored.sort(reverse=True)
        (top, phase), (second, _) = scored[0], scored[1]
        if top > best[2]:
            best = (meter, phase, top, second)

    meter, phase, top, second = best
    margin = max(0.0, min(1.0, (top - second) / 2.0))
    return meter, phase, margin


def _zscore(np, values):
    spread = float(values.std())
    if spread < 1e-9:
        return np.zeros_like(values)
    return (values - float(values.mean())) / spread


class HarmonicOnsetDetector:
    """`OnsetDetector` — §14's raw material, read off the **harmonic** component.

    The default, and the answer to "the strumming patterns aren't rhythmic". The
    full-mix detector below asks "did anything strike here", and on a produced
    record the answer is yes on every eighth, because that is what a hi-hat is
    for. §14 then folds those onsets onto one bar, every cell clears the support
    threshold, and the groove the player actually played comes out as
    machine-gun eighths. Measured on `bench/run_bench.py --strum`, which renders
    a D-DU-UD-U strum under a kit: the full-mix detector emits all eight eighths
    on every kit specimen, including a beat the guitar demonstrably never
    touched.

    So the question is changed from "did anything strike here" to **"was a chord
    struck here"**, and HPSS is what separates them. A strum re-excites the
    strings, so its energy is pitched and survives into the harmonic component; a
    kick, a snare and a hat are broadband transients and do not. On the same
    specimens the drum-only cells come back with support **0.00** rather than
    1.00 — not attenuated, *absent* — which is what makes the support number
    downstream mean "the player struck this in every bar" again.

    **`margin=1.0` is measured, not chosen for being the default.** The
    separation gets *worse* above it, and fast: at 2.0 the loud-kit specimen
    starts losing real strokes, and by 3.0 every specimen picks up spurious
    onsets around 2.667 and 3.667 — the envelope has been smeared enough that
    peak-picking finds artifacts of the separation rather than the playing.
    Running the onset envelope over a CQT of the harmonic part (tried, on the
    theory that pitched attacks would stand out further) is worse still, for the
    same reason.

    What this cannot do is separate a guitar from a piano, an organ or a voice —
    they are all harmonic, and all of them are supposed to be here anyway, since
    §14 is transcribing *the song's* rhythm and not one instrument's. The claim
    is only the one the measurement supports: the drums stop voting.

    **It costs about 11 seconds more than the full-mix detector on a four-minute
    track** (13.3s against 2.0s, measured at 22.05 kHz), because HPSS is a pair
    of median filters over the whole spectrogram. That is affordable against
    `dsp_reserve_s`, and it is worth naming rather than discovering: this is the
    most expensive thing in §14 by an order of magnitude, and a shorter deadline
    is the first place it would show up.
    """

    name = "harmonic"
    version = "1.0"

    # librosa's own default is 1.0; named here because it is a measurement and
    # not an inherited default — see the class docstring for what 2.0 and 3.0 do.
    margin = 1.0

    def detect(self, pcm: PCM, sr: int) -> list[Onset]:
        import librosa

        harmonic = librosa.effects.harmonic(pcm, margin=self.margin)
        envelope = librosa.onset.onset_strength(y=harmonic, sr=sr, hop_length=HOP)
        return _peaks(envelope, sr)


class LibrosaOnsetDetector:
    """`OnsetDetector` on the **full mix** — every attack, whatever made it.

    Kept registered and selectable (`CHORDS_ONSET_DETECTOR=librosa`) rather than
    deleted, because it is the right answer for a recording that *is* one
    instrument and it is the baseline the harmonic detector is measured against.
    It is not the default: on anything with drums on it, this is the detector
    that hands §14 a wall of eighths.

    Separate from the beat tracker because they answer different questions: the
    grid says where the pulse is, this says where a hand actually struck. A song
    can have a confident grid and unreadable strumming.
    """

    name = "librosa"
    version = "1.0"

    def detect(self, pcm: PCM, sr: int) -> list[Onset]:
        import librosa

        envelope = librosa.onset.onset_strength(y=pcm, sr=sr, hop_length=HOP)
        return _peaks(envelope, sr)


def _peaks(envelope, sr: int) -> list[Onset]:
    """An onset-strength envelope → the attacks in it."""
    import librosa

    # `backtrack=False` on purpose. Backtracking rolls each detection back to the
    # preceding energy minimum, which is what you want when slicing audio into
    # samples and wrong when you want the time a hand struck: it moves every
    # onset earlier, by an amount that depends on the attack's shape. §14 folds
    # these onto a bar, so a systematic early bias is not noise that averages out
    # — it walks the whole pattern off the grid, and the downbeat (the one cell
    # with a bar boundary in front of it) off first.
    frames = librosa.onset.onset_detect(
        onset_envelope=envelope, sr=sr, hop_length=HOP, backtrack=False
    )
    if len(frames) == 0:
        return []
    times = librosa.frames_to_time(frames, sr=sr, hop_length=HOP)
    peak = float(envelope.max()) or 1.0
    return [
        Onset(t_ms=int(round(t * 1000)),
              strength=float(envelope[min(int(f), len(envelope) - 1)] / peak))
        for f, t in zip(frames, times)
    ]
