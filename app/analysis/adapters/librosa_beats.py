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

from ..types import FULL, LOW, MID, PCM, BeatGrid, Onset

HOP = 512               # ~23 ms — onset-rate, unlike the chord engine's hop
_METERS = (4, 3)        # what to consider; 4 first so it wins ties

# Where the bass stops and the chord starts, for §14.1's band label.
#
# **Measured, on the synthetic oom-pah in `bench/synth.py`** (a root octave at C2
# on beats 1 and 3, a C-major right hand at C4–G4 on 2 and 4), comparing each
# band's own peak-normalized onset strength at the detected frames:
#
#     split    bass strokes (low/mid)      chord strokes (low/mid)
#     180 Hz   median  90, min  5.2        median 0.04, max 0.09
#     250 Hz   median 202, min  8.1        median 0.07, max 0.16
#     320 Hz   median 184, min  8.9        median 0.34, max 0.39
#
# 250 Hz separates them by the widest margin. 320 leaks the bass's upper partials
# into the chordal band — a third of the chord strokes' energy arrives as "low" —
# and 180 sits below the fundamental of a guitar's own low strings, so an ordinary
# strum starts reading as mid-only. The figure is also where the conventional
# bass/low-mid boundary sits, which is not a coincidence: it is under the root of
# a piano's left hand and over the fundamental of most bass notes.
BAND_SPLIT_HZ = 250.0

# How hard a band has to be struck, **against its own typical attack on this
# recording**, to count as struck here at all. Both bands struck ⇒ `FULL`.
#
# **The obvious rule is wrong and the measurement is what says so.** The first
# cut asked which band was louder — a ratio between the two — and it separates a
# bass note from a chord stab beautifully (a factor of 8 the one way, 0.16 the
# other). It then labels an ordinary strummed guitar `MID`, on every specimen in
# `bench/synth.py`: a chord voiced from E3 up puts most of its attack energy
# above 250 Hz, so "which band is louder" has a definite answer for a stroke
# where the honest answer is *both*. A rule that calls a plain strum a
# right-hand-only stroke would hand a piano a song with no bass in it.
#
# So each band is asked about on its own, and against itself: *relative to how
# hard this band is struck when it is struck on this recording, was it struck
# here?* That is the same self-relative discipline `strumming.py` already uses
# everywhere — support as a share of bars, prominence against the bar's own mean,
# contrast against the bar's strongest cell — and it is what makes the label
# survive a mix where one band is simply quieter than the other.
#
# Measured end to end, as the share of a specimen's onsets landing on the right
# label (`bench/synth.py`, plus a synthetic oom-pah — a root octave at C2 on
# beats 1 and 3, a C-major right hand at C4–G4 on 2 and 4):
#
#     presence   oom-pah (want low/mid)   folk strum (want full)
#     0.35       30/36                    43/54
#     0.50       30/36                    37/54
#     0.65       31/36                    32/54
#
# 0.35 is the best of them on both questions at once, and the per-onset noise it
# leaves is not what ships: `strumming._band_of` takes a majority across every
# bar that struck a cell, so a stroke has to be mislabelled in most of its bars
# before the pattern says so.
BAND_PRESENCE = 0.35

# librosa's own default for a mel spectrogram; named because `_bands`
# converts a frequency to a *mel bin index* against it, and the two have to
# agree or the split lands somewhere nobody chose.
_MEL_BANDS = 128


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
        # Labelled off the *harmonic* component, the same signal the onsets were
        # found in — a kick drum is bass-band energy that no hand played, and
        # labelling against the full mix would hand every one of them to the left
        # hand.
        return _peaks(envelope, sr, signal=harmonic)


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
        return _peaks(envelope, sr, signal=pcm)


def _peaks(envelope, sr: int, *, signal=None) -> list[Onset]:
    """An onset-strength envelope → the attacks in it.

    `signal` is the audio the envelope was computed from. Given one, each attack
    is also labelled with the band it arrived in (§14.1); without one every
    attack is `FULL`, which is what an unlabelled onset has always meant.

    **The onset times and strengths do not depend on the labelling**, and that is
    the point of doing it this way rather than peak-picking each band separately.
    Every threshold in `strumming.py` was tuned against the onsets this function
    already returned; a second detector running on a third of the spectrum would
    move all of them at once, for a label that is meant to be additive. So the
    attacks are found exactly as before and the bands only get a vote on *what*
    was struck, never on *whether*.
    """
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
    bands = _bands(signal, sr, frames)
    return [
        Onset(t_ms=int(round(t * 1000)),
              strength=float(envelope[min(int(f), len(envelope) - 1)] / peak),
              band=band)
        for f, t, band in zip(frames, times, bands)
    ]


def _bands(signal, sr: int, frames) -> list[str]:
    """Which band each already-detected attack arrived in (§14.1).

    Two onset-strength envelopes over the same STFT — one below `BAND_SPLIT_HZ`,
    one above — read at the frames the detector already chose.

    **Each band is normalized by its own peak before they are compared.** Without
    that, the label would be a report on the mix's balance rather than on the
    playing: a record with a loud bass guitar would call every attack `LOW` and a
    thin one would call every attack `MID`, in both cases without a single note
    having been played differently. Normalized, the question becomes the one worth
    asking — *relative to how hard this band is ever hit on this recording, was it
    hit here?* — which is the same self-relative discipline `strength` and
    `ACCENT_RATIO` already use.
    """
    if signal is None or len(frames) == 0:
        return [FULL] * len(frames)
    import librosa
    import numpy as np

    mels = librosa.mel_frequencies(n_mels=_MEL_BANDS, fmin=0, fmax=sr / 2)
    split = int(np.searchsorted(mels, BAND_SPLIT_HZ))
    # A split at either end leaves one band empty, and an empty band cannot
    # out-shout anything — so say so rather than emit a column of one label.
    if split <= 0 or split >= _MEL_BANDS:
        return [FULL] * len(frames)

    envelopes = librosa.onset.onset_strength_multi(
        y=signal, sr=sr, hop_length=HOP, n_mels=_MEL_BANDS,
        channels=[0, split, _MEL_BANDS],
    )
    low, mid = envelopes[0], envelopes[1]
    indices = [min(int(f), len(low) - 1, len(mid) - 1) for f in frames]
    at_low = [float(low[i]) for i in indices]
    at_mid = [float(mid[i]) for i in indices]

    # Each band's own typical attack: the median of what it does at the moments
    # something was struck. Not the peak — one loud bass note would then set the
    # bar for every other stroke on the record — and not the mean, which the same
    # loud note drags. The median is what "normally" means here.
    reference_low = _median(at_low) or 1.0
    reference_mid = _median(at_mid) or 1.0

    out: list[str] = []
    for bass, chordal in zip(at_low, at_mid):
        struck_low = bass >= BAND_PRESENCE * reference_low
        struck_mid = chordal >= BAND_PRESENCE * reference_mid
        if struck_low and struck_mid:
            out.append(FULL)
        elif struck_low:
            out.append(LOW)
        elif struck_mid:
            out.append(MID)
        else:
            # Quieter than usual in both bands — a ghost stroke. It was played,
            # so it keeps its onset; nothing is claimed about how.
            out.append(FULL)
    return out


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0
