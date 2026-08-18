"""A chroma-template chord engine — the floor, and the control.

Not a candidate to win. It exists because a benchmark with only strong entrants
cannot tell you whether the strong entrants are strong: if a template matcher
built from a dozen lines of textbook DSP scores within a few points of a
transformer, the transformer is not earning its dependency, its download, or its
cold start. Every number the real candidates post is read against this one.

It is also the corpus's independent check. `bench/fetch_corpus.py` aligns
annotations to a recording it did not produce, and a *wrong* alignment is the one
failure that quietly flatters or punishes every engine equally. Running this
engine against an aligned track answers that: chance is about 4% (one of ~24
plausible chords), so a track scoring near chance is misaligned, not difficult.

Method, deliberately ordinary: CQT chroma → per-frame cosine similarity against
48 chord templates → Viterbi smoothing with a self-transition bonus → merge runs
into spans. The Viterbi step is the only non-obvious part, and it earns its place
by removing the single-frame flicker that would otherwise turn one chord into
nine spans and wreck the quantizer downstream.
"""

from __future__ import annotations

from ..tuning import CONCERT_PITCH, Tuning
from ..types import PCM, RawChordSpan

# Sharps throughout; `app.chords.normalize` maps enharmonics, so the choice only
# affects what a debug print looks like.
_PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# Intervals per quality, and the Harte suffix to emit. Four qualities rather than
# the app's full grammar on purpose: a template matcher cannot reliably tell a
# maj9 from a maj7, and inventing distinctions it cannot support would make the
# comparison against real engines flattering rather than honest.
_TEMPLATES: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("maj", (0, 4, 7)),
    ("min", (0, 3, 7)),
    ("7", (0, 4, 7, 10)),
    ("min7", (0, 3, 7, 10)),
)

HOP = 2048          # ~93 ms at 22.05 kHz — chord-rate, not onset-rate

# Expected chord length, in frames (~2 s). Sets the stay-put prior: a chord is
# expected to persist, and P(stay) = 1 - 1/length is what "expected to" means.
_CHORD_FRAMES = 21.0

# Emission sharpness. Cosine similarity is **not** a likelihood — its log spans
# maybe 0.4 nats between a good and a bad match, while the switch penalty is
# nearly 7, so raw similarities lose every argument with the prior and the
# decoder emits one chord for the whole song. (Measured, not theorised: the first
# version of this file returned three spans for "A Hard Day's Night" and scored
# 0.05.) Raising the scores to a power before the log restores the balance.
# Swept over the corpus at 8/16/24/40: 0.524, 0.531, 0.527, 0.518 — a broad
# optimum, which is the reassuring shape. 16 it is.
_SHARPNESS = 16.0

# Smoothing window for the chroma, in frames (~0.5 s). Chord identity is a
# property of a bar, not of a 93 ms frame.
_SMOOTH_FRAMES = 5


class ChromaTemplateEngine:
    """`ChordEngine` — CQT chroma matched against fixed templates."""

    name = "chroma"
    version = "1.0"

    def analyze(self, pcm: PCM, sr: int, *,
                tuning: Tuning | None = None) -> list[RawChordSpan]:
        import librosa
        import numpy as np

        # Harmonic component only: a snare hitting under a held chord is broadband
        # energy that lands in every pitch class, which is noise to a template
        # matcher. Cheaper than it looks, and worth several points.
        harmonic = librosa.effects.harmonic(pcm, margin=2.0)
        # Same pitch-reference correction as the BTC adapter, for the same
        # reason: `chroma_cqt` estimates its own tuning when handed None, and
        # estimating it *here* would let the answer differ from the one the
        # rest of the analysis was told. One measurement, passed down.
        chroma = librosa.feature.chroma_cqt(
            y=harmonic, sr=sr, hop_length=HOP,
            tuning=(tuning or CONCERT_PITCH).bins(12))
        if chroma.shape[1] == 0:
            return []

        # Moving average over ~0.5 s. (An earlier version used librosa's
        # `nn_filter`, which is a source-separation trick, quadratic in frames,
        # and cost 58 s on a three-minute track for no accuracy.)
        if chroma.shape[1] >= _SMOOTH_FRAMES:
            kernel = np.ones(_SMOOTH_FRAMES) / _SMOOTH_FRAMES
            chroma = np.vstack([np.convolve(row, kernel, mode="same") for row in chroma])
        chroma = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9)

        labels, templates = _build_templates(np)
        # (n_templates, n_frames) cosine similarity, clipped to keep log() finite.
        scores = np.clip(templates @ chroma, 1e-6, None)

        path = _viterbi(np, scores)
        frame_s = HOP / sr
        return _spans_from_path(labels, path, scores, frame_s)


def _build_templates(np):
    """48 unit-norm pitch-class templates, plus their Harte labels."""
    labels: list[str] = []
    rows = []
    for root in range(12):
        for suffix, intervals in _TEMPLATES:
            vector = np.zeros(12, dtype="float32")
            for interval in intervals:
                vector[(root + interval) % 12] = 1.0
            rows.append(vector / np.linalg.norm(vector))
            labels.append(f"{_PITCH_NAMES[root]}:{suffix}")
    return labels, np.vstack(rows)


def _viterbi(np, scores):
    """Most likely label sequence under a uniform stay-or-switch transition.

    A plain per-frame argmax flickers: two chords sharing three notes trade
    places frame to frame, and the span list downstream becomes noise. One
    self-transition bonus fixes it without pretending to model harmony.
    """
    n_states, n_frames = scores.shape
    log_emission = _SHARPNESS * np.log(scores)
    self_transition = 1.0 - 1.0 / _CHORD_FRAMES
    stay = np.log(self_transition)
    switch = np.log((1.0 - self_transition) / (n_states - 1))

    delta = log_emission[:, 0].copy()
    backpointers = np.zeros((n_states, n_frames), dtype="int32")
    for t in range(1, n_frames):
        # Either stay in state i, or arrive from the best other state.
        best_other = int(np.argmax(delta))
        via_switch = delta[best_other] + switch
        stay_scores = delta + stay
        take_stay = stay_scores >= via_switch
        backpointers[:, t] = np.where(take_stay, np.arange(n_states), best_other)
        delta = np.where(take_stay, stay_scores, via_switch) + log_emission[:, t]
        delta -= delta.max()        # renormalise; only differences matter

    path = np.zeros(n_frames, dtype="int32")
    path[-1] = int(np.argmax(delta))
    for t in range(n_frames - 1, 0, -1):
        path[t - 1] = backpointers[path[t], t]
    return path


def _spans_from_path(labels, path, scores, frame_s: float) -> list[RawChordSpan]:
    """Runs of one label → one span, with mean similarity as confidence."""
    spans: list[RawChordSpan] = []
    start = 0
    for index in range(1, len(path) + 1):
        if index < len(path) and path[index] == path[start]:
            continue
        state = int(path[start])
        confidence = float(scores[state, start:index].mean())
        spans.append(RawChordSpan(
            start_ms=int(round(start * frame_s * 1000)),
            end_ms=int(round(index * frame_s * 1000)),
            label=labels[state],
            confidence=min(1.0, confidence),
        ))
        start = index
    return spans
