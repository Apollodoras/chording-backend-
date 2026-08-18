"""The pitch reference: what happens to a recording that is not at A440.

The defect these pin is the most expensive one this pipeline has shipped, and
the reason is that it is invisible from inside. A recording 48 cents sharp puts
every partial halfway between two CQT bins; the model picks whichever the noise
favours, and the answer is stable *within* a passage and flips *between* them —
so the chart comes out as the right progression, transposed, in a coherent key,
internally consistent, with high confidence. `lint` cannot see it (the song
agrees with itself), the consensus vote cannot see it (every repeat is shifted
the same way, so they all agree), and the key finder cannot see it (it is handed
the shifted chords and dutifully names the shifted key).

Measured on the real case — Tom Petty, Mary Jane's Last Dance, official video —
the recording is 48 cents sharp and 82% of the delivered chart was a semitone
above the record.

The one thing here that is genuinely easy to get backwards is the **direction**
of the correction, so it is asserted against synthesised audio whose offset is
known by construction rather than against a fixture whose answer was copied from
a previous run.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.analysis import tuning as tuning_probe
from app.analysis.tuning import AMBIGUOUS_SEMITONES, CONCERT_PITCH, Tuning

librosa = pytest.importorskip("librosa")

SR = 22050


def _bed(seconds: float = 20.0, cents: float = 0.0) -> np.ndarray:
    """A chord bed in the mid register, detuned by a known number of cents.

    Detuning is applied to the *frequencies*, not by resampling, so tempo and
    length are unchanged and pitch is the only thing the estimator can be
    responding to.

    C4-F4-G4-E4 rather than the C2-F2-G2-E2 bed `test_adapters` uses, and the
    difference is load-bearing: `librosa.estimate_tuning` runs `piptrack`, whose
    default analysis floor is 150 Hz. Fundamentals below that are measured only
    through their harmonics, and the estimate degrades until it wraps — a bed at
    +45 cents reads as -50. That is a property of the fixture and not of the
    recording: real music has ample energy above 150 Hz, and the two songs this
    module exists for measure cleanly (+48 and +15 cents, stable across windows).
    """
    ratio = 2.0 ** (cents / 1200.0)
    t = np.arange(int(SR * seconds)) / SR
    signal = np.zeros_like(t)
    for index, root in enumerate((261.63, 349.23, 392.00, 329.63)):  # C4 F4 G4 E4
        segment = (t >= index * 5) & (t < (index + 1) * 5)
        for harmonic, amplitude in ((1, 1.0), (2, 0.5), (3, 0.3), (4, 0.2)):
            signal[segment] += amplitude * np.sin(
                2 * np.pi * root * ratio * harmonic * t[segment])
    return (signal / np.abs(signal).max()).astype("float32")


# -- the measurement --------------------------------------------------------

def test_a_recording_at_concert_pitch_reports_no_offset():
    """Within a couple of cents, which is the fixture's own measurement bias —
    a synthesised bed of pure harmonics is not exactly what `piptrack` models."""
    assert tuning_probe.estimate(_bed(), SR).cents == pytest.approx(0.0, abs=5.0)


@pytest.mark.parametrize("cents", [-45.0, -30.0, -15.0, 15.0, 30.0, 45.0])
def test_the_offset_is_measured_with_the_right_sign_and_size(cents):
    """Sharp reads positive, flat reads negative, within a few cents."""
    measured = tuning_probe.estimate(_bed(cents=cents), SR)
    assert measured.cents == pytest.approx(cents, abs=8.0)


def test_an_offset_near_a_quarter_tone_is_flagged_ambiguous():
    """Not refused — flagged. Some semitone still has to be chosen, and the
    measured one beats assuming zero; but which one it is *is* a coin flip, and
    that is a real reason for the chart to carry less confidence."""
    assert Tuning(0.48).ambiguous is True
    assert Tuning(-0.45).ambiguous is True
    assert Tuning(0.1).ambiguous is False
    assert AMBIGUOUS_SEMITONES < 0.5


def test_a_broken_estimate_degrades_to_concert_pitch_rather_than_failing():
    """The pre-existing behaviour was "assume A440". A tuning probe that raised
    would take down analyses that used to succeed, which is a worse trade than
    silently doing what the code did before this module existed."""
    assert tuning_probe.estimate(np.zeros(4, dtype="float32"), SR) == CONCERT_PITCH
    assert tuning_probe.estimate(object(), SR) == CONCERT_PITCH


# -- the units, which are the other easy mistake ----------------------------

def test_bins_convert_from_semitones_to_the_grid_being_asked_about():
    """`librosa.cqt(tuning=...)` counts in bins of ITS OWN grid, so half a
    semitone is 0.5 bins at 12 per octave and 1.0 bins at 24. Getting this wrong
    is a silent factor-of-two in the correction."""
    assert Tuning(0.5).bins(12) == pytest.approx(0.5)
    assert Tuning(0.5).bins(24) == pytest.approx(1.0)
    assert Tuning(-0.25).bins(24) == pytest.approx(-0.5)


# -- the correction actually cancelling the offset --------------------------

def _peak_bin(features, times, low: float, high: float) -> int:
    """Which CQT bin the energy sits in, over one sustained note."""
    frames = [i for i, t in enumerate(times) if low < t < high]
    assert frames, "no frames in that window"
    return int(np.argmax(features[frames].mean(axis=0)))


def test_the_correction_puts_a_detuned_bed_back_on_the_in_tune_bins():
    """The end-to-end claim, and the one that pins the DIRECTION of the shift.

    Asserted on bin *positions* rather than on a distance between whole spectra.
    The spectra are 144 log-magnitude bins of which four carry the signal, so an
    L1 distance over all of them is dominated by the noise floor and moves only
    ~10% for a correction that is in fact exact. Where the peak lands is the
    thing the model reads, and it is unambiguous: at 45 cents sharp, every note
    is one 24-per-octave bin — a quarter tone — above where it belongs, and the
    correction has to put all four back.
    """
    from app.analysis.adapters.btc import _features

    in_tune, in_tune_times = _features(np, _bed(cents=0.0))
    sharp = _bed(cents=45.0)
    measured = tuning_probe.estimate(sharp, SR)
    corrected, corrected_times = _features(np, sharp, measured)
    uncorrected, uncorrected_times = _features(np, sharp, CONCERT_PITCH)

    # C4, F4, G4, E4 — the bed plays one per five seconds.
    for low, high in ((1.0, 4.0), (6.0, 9.0), (11.0, 14.0), (16.0, 19.0)):
        want = _peak_bin(in_tune, in_tune_times, low, high)
        assert _peak_bin(corrected, corrected_times, low, high) == want
        # And the test has to be able to fail: uncorrected must land elsewhere,
        # or this would pass with the fix reverted.
        assert _peak_bin(uncorrected, uncorrected_times, low, high) == want + 1


def test_the_shift_is_one_measurement_for_the_whole_recording():
    """Per-block estimation is the failure this parameter exists to remove: a
    pitch reference that can wander mid-song reproduces the original defect in a
    subtler form. `_features` is handed one `Tuning` and must apply that one to
    every block, so two halves of a detuned bed must land on the same bins."""
    from app.analysis.adapters.btc import _CHUNK_S, _features

    sharp = _bed(seconds=20.0, cents=45.0)
    measured = tuning_probe.estimate(sharp, SR)
    features, times = _features(np, sharp, measured)

    # Same note, two different blocks: C4 runs 0–5 s, so pick frames either side
    # of the 10 s block boundary from within one sustained pitch region.
    first = [i for i, t in enumerate(times) if 1.0 < t < 4.0]
    second = [i for i, t in enumerate(times) if 11.0 < t < 14.0]
    assert first and second and _CHUNK_S == 10.0

    # Peak bin of the strongest partial must agree across the block boundary.
    peak_first = int(np.argmax(features[first].mean(axis=0)))
    peak_second_region = features[second].mean(axis=0)
    # (11–14 s is F4/G4 territory, so compare stability of the *grid* rather than
    # the pitch: the bin spacing between the two peaks must be a whole number of
    # semitones — 2 bins at 24 per octave — which is only true if both blocks
    # were transformed on the same shifted grid.)
    peak_second = int(np.argmax(peak_second_region))
    assert (peak_second - peak_first) % 2 == 0


# -- when the correction is applied at all ----------------------------------

def test_a_small_offset_is_measured_and_deliberately_not_corrected():
    """The gate, and the measurement behind it.

    Correcting every real recording in `bench/` — Beatles masters at 4 to 35
    cents — cost 2.5 points of chord accuracy while leaving root accuracy exactly
    where it was. BTC was trained on real records, so that spread is inside its
    distribution and moving the grid moves the input off it. The correction is
    for the boundary case only.
    """
    for cents in (4.0, 15.0, 35.0, -25.0):
        measured = Tuning(cents / 100.0)
        assert measured.correction == CONCERT_PITCH
        # ...and the measurement itself survives for the report.
        assert measured.cents == pytest.approx(cents)


def test_an_ambiguous_offset_is_corrected():
    """48 cents is Mary Jane's Last Dance, where not correcting cost 82% of the
    chart. 42 is the boundary itself, which counts as ambiguous."""
    for cents in (48.0, 42.0, -45.0):
        measured = Tuning(cents / 100.0)
        assert measured.correction == measured


def test_the_gate_and_the_flag_are_the_same_decision():
    """`tuningAmbiguous` in the report means exactly "the correction fired", so
    the two must never be able to disagree."""
    for hundredths in range(-50, 50):
        measured = Tuning(hundredths / 100.0)
        assert (measured.correction != CONCERT_PITCH) == measured.ambiguous or measured.negligible
