"""What pitch this recording calls A — measured once, before any chord is named.

A constant-Q transform lays its filters on a fixed grid derived from A440. That
is a *choice about the recording* dressed up as a constant, and it is wrong for
any record that is not at concert pitch: analogue masters drift, tape machines
run fast, bands tune to the piano in the room, and a 1993 rock record has no
obligation to agree with a tuning fork.

When the recording disagrees, every partial lands between two bins. Which of the
two wins is then decided by whatever noise is nearest, and the answer flips
mid-song — so the chart comes out as the correct progression **transposed by a
semitone**, internally consistent, in a plausible key, with high confidence. It
is the single most expensive kind of wrong this pipeline can be, because nothing
downstream can see it: `lint` and `lint_sync` check the song against itself, the
consensus vote finds every repeat agreeing (they are all shifted the same way),
and the key finder happily names the wrong key it was handed.

Measured on Mary Jane's Last Dance: **48 cents sharp**, and 82% of the delivered
chart a semitone above the record. Every reference recording the benchmark had
was a DAW backing track, a modern cover, or a digital remaster — all at A440 by
construction — so ~33 graded songs could not produce the defect.

## Why the estimate is taken at semitone resolution

`librosa.estimate_tuning(bins_per_octave=N)` returns a deviation in fractions of
one *N-per-octave bin*, wrapped into [-0.5, 0.5). The bins_per_octave passed here
therefore decides **what the recording is snapped to**, and that is the whole
decision:

- at `bins_per_octave=24` (the CQT's own grid) the answer wraps at ±25 cents, so
  a 48-cent-sharp record reads as "2 cents flat" — true at quarter-tone
  resolution and useless, because it would align the recording's notes onto the
  *odd*, quarter-tone bins, which is not what the model was trained to see;
- at `bins_per_octave=12` the answer wraps at ±50 cents, so the same record reads
  as "48 cents sharp" and snaps to the nearest **semitone**, which is what a
  musician means by being out of tune.

The result is converted into the CQT's bin units at the call site, because that
is the unit `librosa.cqt(tuning=...)` wants — see `adapters/btc.py`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .types import PCM

log = logging.getLogger("chords.tuning")

# Beyond this, the recording is close enough to halfway between two semitones
# that which one its partials land on is a coin flip rather than a measurement.
# This is the threshold at which the correction is APPLIED, and the reason it is
# not simply "always" is measured rather than assumed.
#
# On the nine real tracks of `bench/` — Isophonics Beatles recordings, offsets of
# 4 to 35 cents — correcting every one of them cost 2.5 points of chord accuracy
# (0.808 → 0.783) while leaving root accuracy untouched at 0.888. That is not
# noise and it has a mechanism: BTC was trained on real recordings, which carry
# exactly this spread, so a few tens of cents is *inside* the distribution it
# learned and nudging the CQT grid moves the input off it. The model is better
# left alone there.
#
# What the model cannot survive is the boundary. At 35 cents (Here Comes The Sun)
# it is fine; at 48 cents (Mary Jane's Last Dance) every partial sits halfway
# between two bins, the assignment goes to whichever noise is nearest, and 82% of
# the chart comes out a semitone high. So the correction earns its place exactly
# where the uncorrected reading is undecidable, and nowhere else.
#
# The measurement is always taken and always published; only its *application* is
# gated. An audit wanting to know which recordings are off concert pitch reads
# `tuningCents`, which is the truth whether or not anything was done about it.
AMBIGUOUS_SEMITONES = 0.42

# Below this the recording is at concert pitch for all practical purposes and the
# correction is reported as zero, so the common case reads as "nothing to see"
# rather than as a number that changes in the fourth decimal between runs.
NEGLIGIBLE_SEMITONES = 0.02

# How much audio `settle_sign` listens to. Thirty seconds from the middle of the
# recording, which is one verse and one chorus of anything in this repertoire and
# costs the chord engine under a second. Swept at 20 / 30 / 45 s on Mary Jane's
# Last Dance: the margin between the two signs is 0.016 / 0.021 / 0.008, so 30 is
# both the widest separation and the middle of the range — a shorter clip has
# less harmony in it, and a longer one dilutes the difference with material the
# model finds easy either way.
SIGN_EXCERPT_S = 30.0


@dataclass(frozen=True)
class Tuning:
    """Where this recording's A sits, relative to 440 Hz."""

    # Semitones, in [-0.5, 0.5). Positive means the record is sharp.
    semitones: float

    @property
    def cents(self) -> float:
        return self.semitones * 100.0

    @property
    def ambiguous(self) -> bool:
        """Near enough to a quarter tone that the semitone it snaps to is a
        coin flip. A reason to lower confidence, never a reason to refuse."""
        return abs(self.semitones) >= AMBIGUOUS_SEMITONES

    @property
    def negligible(self) -> bool:
        return abs(self.semitones) < NEGLIGIBLE_SEMITONES

    @property
    def correction(self) -> "Tuning":
        """The shift to actually hand a transform — which is not the same thing
        as the shift that was measured.

        Zero unless the reading is ambiguous. See `AMBIGUOUS_SEMITONES`: below
        the boundary the recording is unambiguously on its semitone and the model
        reads it correctly, so moving the grid only costs accuracy; at the
        boundary nothing else can decide it. Separating "what we measured" from
        "what we did about it" is the whole point — the first is a fact about the
        recording and belongs in the report, the second is a policy about a model
        and belongs here.
        """
        return self if self.ambiguous else CONCERT_PITCH

    @property
    def other_sign(self) -> "Tuning":
        """The same measurement read the other way round.

        `estimate` wraps into [-0.5, 0.5), so a recording 48 cents sharp and one
        52 cents flat are *the same reading* — the wrap point falls between them
        and which side it lands on is decided by a hair of noise. This is that
        other reading, and it is only ever a meaningful question when
        `ambiguous` is true.
        """
        return Tuning(round(self.semitones - 1.0 if self.semitones > 0
                            else self.semitones + 1.0, 4))

    def bins(self, bins_per_octave: int) -> float:
        """The same offset in units of one bin of a `bins_per_octave` grid —
        what `librosa.cqt(tuning=...)` and `librosa.chroma_cqt(tuning=...)` take."""
        return self.semitones * (bins_per_octave / 12.0)


CONCERT_PITCH = Tuning(0.0)


def estimate(pcm: PCM, sr: int) -> Tuning:
    """Measure the recording's tuning. Never raises — a failure is concert pitch.

    Deliberately forgiving: a tuning estimate that throws would take down an
    analysis that used to succeed, and the pre-existing behaviour (assume A440)
    is exactly what `CONCERT_PITCH` means. So a broken estimate degrades to the
    status quo instead of to an error.
    """
    try:
        import librosa
        import numpy as np

        audio = np.asarray(pcm, dtype="float32")
        if audio.size < sr:  # under a second of audio has no tuning to find
            return CONCERT_PITCH
        # bins_per_octave=12 — see the module docstring; this is the line that
        # decides the snap target is a semitone and not a quarter tone.
        semitones = float(librosa.estimate_tuning(y=audio, sr=sr, bins_per_octave=12))
    except Exception:
        log.warning("tuning estimate failed — assuming concert pitch", exc_info=True)
        return CONCERT_PITCH

    if semitones != semitones:  # NaN
        return CONCERT_PITCH
    if abs(semitones) < NEGLIGIBLE_SEMITONES:
        return CONCERT_PITCH
    tuning = Tuning(round(semitones, 4))
    log.info("recording is %+.0f cents from A440%s", tuning.cents,
             " (ambiguous — within a hair of a quarter tone)" if tuning.ambiguous else "")
    return tuning


# --- the sign of an ambiguous reading, and why nothing here decides it ------
#
# `estimate` wraps into [-0.5, 0.5), so at the boundary "+48 cents sharp" and
# "−52 cents flat" are the same reading of the same audio and which side it lands
# on is decided by a hair of noise. That matters more than anything else in this
# module: measured on Mary Jane's Last Dance, correcting by +0.48 scores 0.380
# root against its reference chart and correcting by −0.52 scores **0.000** —
# every chord in the song a semitone from where it belongs, in a chart that is
# perfectly self-consistent and confident. Getting the sign wrong is worse than
# not correcting at all (0.112, with the transposition sweep finding the whole
# chart intact one semitone down).
#
# **The obvious tie-break does not work, and this is the measurement that says
# so.** The audit of 2026-08-18 recommended analysing a short excerpt under both
# signs and keeping whichever the chord engine believed more. Implemented and
# run over the two ambiguous recordings in the bench:
#
#   Mary Jane's Last Dance   +0.48 → 0.833   −0.52 → 0.812   picks the right one
#   Smooth Criminal          +0.47 → 0.654   −0.53 → 0.757   picks the WRONG one
#
# and the whole track rather than an excerpt does not rescue it (+0.47 → 0.781
# against −0.53 → 0.819, still backwards). End to end the rule left Mary Jane
# exactly where it was — the measured sign was already the right one — and took
# Smooth Criminal from 0.330 root to **0.012**.
#
# The mechanism is worth keeping, because it is why no confidence-shaped rule
# will work here: Smooth Criminal's harmony is a bass line with no third
# sounding, so the model is guessing at the quality throughout, and a grid shifted
# off the recording lets it guess *more decisively*. Softmax confidence measures
# how peaked the posterior is, not how right it is, and the two come apart
# exactly where the audio is ambiguous — which is the only place this question is
# ever asked.
#
# So the ambiguous reading is applied as measured, and `ambiguous` stays a
# confidence flag rather than a decision. A rule that settles this needs evidence
# from outside the recognizer — the key finder's own margin under each sign, or
# an interval-consistency test on the CQT itself — and neither is written.
