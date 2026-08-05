"""§20.7 — a loudness envelope, which is all the audio layer gets to say about
section *names*.

RMS over short frames, normalized within the track. Deliberately the dullest
possible feature, for three reasons.

**It is enough.** §15's rule is "the repeated high-energy segment is the
chorus", and a chorus is louder and denser than the verse it follows in
essentially all of this repertoire. Anything more elaborate (timbral
self-similarity, `msaf`'s novelty curves) would be answering a question §20
already answers from the chords — *where* the sections are — rather than the one
that is actually open, which is what to call them.

**It is cheap.** One pass of RMS over a decoded track is milliseconds, on a
worker that is already paying for a chord model and a beat tracker.

**It carries nothing.** Normalizing within the track means the number is
relative, so it says "this part is louder than that part" and cannot say how
loud the recording is, what is in it, or anything that could be inverted back
toward audio — see the note on `EnergyCurve` in `types.py`.

Per-track normalization has one more virtue worth stating: it makes the measure
immune to mastering. A quiet remaster and a loud one produce the same curve, so
the section labels do not depend on which upload of a song was analyzed.
"""

from __future__ import annotations

from ..types import PCM, EnergyCurve

# ~46 ms at 22.05 kHz. Fine enough that a bar at any tempo this service accepts
# contains several frames, coarse enough that the curve is about sections rather
# than about individual strums.
HOP = 1024
FRAME = 2048


class LibrosaEnergyProbe:
    """`StructureProbe` — RMS loudness, normalized within the track."""

    name = "librosa_rms"
    version = "1"

    def probe(self, pcm: PCM, sr: int) -> EnergyCurve:
        import librosa
        import numpy as np

        samples = np.asarray(pcm, dtype="float32")
        if samples.size < FRAME:
            return EnergyCurve(hop_ms=0.0, values=[])

        rms = librosa.feature.rms(y=samples, frame_length=FRAME, hop_length=HOP)[0]
        # Loudness, not power: the ear is closer to logarithmic, and on a linear
        # scale one loud bridge flattens every verse/chorus difference in the
        # rest of the song into indistinguishable noise near zero.
        loudness = librosa.amplitude_to_db(np.maximum(rms, 1e-8), ref=1.0)

        low, high = float(loudness.min()), float(loudness.max())
        span = high - low
        # A track with no dynamic range at all gets a flat curve rather than a
        # division by zero — and a flat curve correctly names nothing a chorus,
        # because nothing in it is louder than anything else.
        values = ([0.0] * len(loudness) if span < 1e-6
                  else ((loudness - low) / span).tolist())

        # Not rounded to a whole millisecond: at 22.05 kHz the hop is 46.44 ms,
        # and calling it 46 walks the curve ~1% out of phase with the song — two
        # seconds adrift by the third minute, which is most of a bar. See the
        # note on `EnergyCurve.hop_ms`.
        hop_ms = 1000.0 * HOP / sr if sr else 0.0
        return EnergyCurve(hop_ms=hop_ms, values=[float(v) for v in values])
