"""BTC — "A Bi-directional Transformer for Musical Chord Recognition" (ISMIR 2019).

Jonggwon Park et al., MIT-licensed, https://github.com/jayg996/BTC-ISMIR19.
The handoff names it (§5.2/§5.3) as the strongest candidate on pop/rock, and it
ships its own pretrained weights in-repo, so there is no training step and no
model-hosting question — only a checkout.

**Why this is an adapter over a checkout rather than a dependency.** BTC is
research code, not a package: no `setup.py`, no PyPI release, imports that assume
its own directory is on `sys.path`. Vendoring it into this repo would mean
carrying 12 MB of weights plus someone else's transformer through every review
here. So the worker image clones it and points `CHORDS_BTC_ROOT` at the result,
and this module is the only place that knows any of that.

Two details worth stating, both of which change the numbers if you get them
wrong:

- **The features must match training exactly.** CQT with 144 bins at 24 per
  octave, hop 2048, at 22.05 kHz — framed in 10-second blocks, because that is
  how the training set was cut and one block is exactly one inference window.
  Then `log(|CQT| + 1e-6)`, standardised by the mean and std stored *in the
  checkpoint*. Skipping that standardisation does not error; it just returns
  confident nonsense.
- **Its native sample rate is already ours.** §5.1 decodes to 22.05 kHz, which is
  what BTC trained on, so nothing is resampled in the normal path.

Output is Harte (`C:maj`, `F#:min7`, `N`, `X`), which is what `postprocess`
already normalizes — the adapter does no translation of its own.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import sys
import threading
from pathlib import Path

from ..tuning import CONCERT_PITCH, Tuning
from ..types import PCM, RawChordSpan

log = logging.getLogger("chords.engines.btc")

_ROOT_ENV = "CHORDS_BTC_ROOT"
_SAMPLE_RATE = 22050
_CHUNK_S = 10.0          # config.mp3.inst_len — the training crop length
_TIMESTEP = 108          # config.model.timestep — frames per inference window
_N_BINS = 144
_BINS_PER_OCTAVE = 24
_HOP = 2048
# librosa.cqt's default lowest bin — C1. Named because the analysis window below
# is derived from it, and the derivation is the whole point.
_FMIN_HZ = 32.70319566257483

# How long a chord is expected to last, in frames — the decoder's stay-put prior.
# One frame is `_HOP / _SAMPLE_RATE` ≈ 93 ms, so 21 frames is about two seconds,
# which is a bar at most tempos in this repertoire.
#
# **Measured, and the shape is the reassuring one.** Swept over the ten-song
# chart corpus from the cached posteriors (8 / 15 / 21 / 30 / 45 frames): root
# 0.856 / 0.857 / 0.857 / 0.857 / 0.857 against the argmax's 0.853. A broad flat
# optimum rather than a peak, so the value is not perched on the edge of the
# measurement, and every setting beats no decoder at all.
_DECODE_CHORD_FRAMES = 21.0

# Posteriors below this are clipped before the log, so a class the model gave
# exactly zero cannot make the whole path -inf.
_DECODE_FLOOR = 1e-6

# How much of each inference window the next one repeats. A quarter, so every
# frame is inside at least one window that saw ~40 frames of context on each side
# — see `BtcEngine._posteriors`. Costs a third more forward passes, which on a
# three-minute track is a few seconds against the CQT's own cost.
_WINDOW_OVERLAP = 0.25

# sha256 of `test/btc_model_large_voca.pt` in the BTC checkout at the commit
# `modal_app.BTC_COMMIT` pins (2682317b), measured from the vendored copy.
#
# The commit pin already says *which* checkpoint; this says the bytes arrived
# intact. It matters because `_load_checkpoint` has a `weights_only=False`
# fallback, and that flag is "run whatever code this pickle asks for". A pinned
# clone makes that defensible and does not make it *verified* — a clone is a
# network fetch, and the fallback would happily execute a substituted file. With
# the hash, the unsafe path is only reachable for a file we recognise.
#
# `CHORDS_BTC_CHECKPOINT_SHA256` overrides it, for an operator deliberately
# running different weights.
BTC_CHECKPOINT_SHA256 = "1673d23f8f9a55ae7f9e8b80a51da616debb22675b8d8b67ea6ce0ef37b0ab51"


def _context_frames() -> int:
    """How many whole hops of real audio each block needs on either side.

    A constant-Q filter is long at the bottom: at 24 bins per octave the Q factor
    is ~34, so the lowest bin's window is `Q · sr / fmin` ≈ 23 000 samples — just
    over a second. `librosa.cqt(..., center=True)` pads whatever it is handed, so a
    block transformed **in isolation** has roughly half that window of padding
    rather than audio at each end: ~0.52 s of each 10 s block, about 10% of it, and
    concentrated in the low bins where the chord *root* lives.

    Handing the transform a little of the neighbouring audio and then trimming back
    to the block's own frames costs one extra hop-length of CQT per edge and makes
    every frame a frame of the recording. Derived rather than hardcoded so it stays
    correct if the bin layout changes; `+1` is slack for the rounding in librosa's
    own filter sizing.
    """
    q = 1.0 / (2.0 ** (1.0 / _BINS_PER_OCTAVE) - 1.0)
    window_samples = math.ceil(q * _SAMPLE_RATE / _FMIN_HZ)
    return math.ceil((window_samples / 2.0) / _HOP) + 1


class BtcUnavailable(RuntimeError):
    """The checkout or its weights are missing. Raised at build time, not mid-job."""


def _btc_root() -> Path:
    configured = os.environ.get(_ROOT_ENV)
    if configured:
        root = Path(configured).expanduser()
    else:
        root = Path(__file__).resolve().parents[3] / "vendor" / "BTC-ISMIR19"
    if not (root / "btc_model.py").is_file():
        raise BtcUnavailable(
            f"BTC checkout not found at {root}. Clone "
            f"https://github.com/jayg996/BTC-ISMIR19 and set {_ROOT_ENV}."
        )
    return root


def _restore_numpy_aliases() -> None:
    """Put back the `np.float`-style aliases NumPy 1.24 removed.

    BTC is 2019 code and uses `np.float` in its positional-encoding helper. The
    aliases were always plain builtins — `np.float is float` — so restoring them
    changes no behaviour for anyone; it just lets six-year-old research code
    import. Done here, at the moment of use, rather than by editing the checkout:
    a patched checkout is a fix that exists only on the machine that applied it,
    and the next person to clone gets the traceback we already solved.
    """
    import numpy as np

    # Only the ones BTC actually uses. `np.str` and `np.complex` are deliberately
    # not in this list: NumPy 1.26 emits a FutureWarning merely for *asking*
    # whether they exist, so probing them would spray warnings through every
    # analysis to fix a call nobody makes.
    for alias, builtin in (("float", float), ("int", int), ("bool", bool),
                           ("object", object)):
        try:
            if not hasattr(np, alias):
                setattr(np, alias, builtin)
        except AttributeError:      # numpy raises rather than returning False
            setattr(np, alias, builtin)


def _checkpoint_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_pinned(path: Path) -> bool:
    """Whether these are the bytes we pinned. Never raises — a hashing failure
    must read as "not verified", which only ever *withholds* a privilege."""
    expected = os.environ.get("CHORDS_BTC_CHECKPOINT_SHA256") or BTC_CHECKPOINT_SHA256
    try:
        actual = _checkpoint_digest(path)
    except OSError:
        log.warning("could not hash the BTC checkpoint at %s", path, exc_info=True)
        return False
    if actual == expected:
        return True
    log.warning("BTC checkpoint at %s hashes to %s, not the pinned %s — the safe "
                "loader is still allowed, the arbitrary-code fallback is not. Set "
                "CHORDS_BTC_CHECKPOINT_SHA256 if these weights are intended.",
                path, actual, expected)
    return False


def _load_checkpoint(torch, path: Path):
    """`torch.load` BTC's weights across PyTorch's 2.6 default change.

    2.6 flipped `weights_only` from False to True. BTC's 2019 checkpoint stores
    its feature mean and std as **numpy scalars** beside the state dict, and
    numpy scalars are not on torch's default allowlist — so the load raises
    `UnpicklingError` on torch >= 2.6 and succeeds on everything older. That is
    the same local-vs-deployed split as `mir_eval` and CUDA-torchaudio: a laptop
    pinned to an older torch analyzes perfectly while the image fails every job
    in the chord stage, behind a health check that is still green.

    Allowlisting the one type it needs is preferred over `weights_only=False`.
    The checkpoint *is* trusted — a commit-pinned clone, baked in at build time —
    but trusted is not a reason to grant a pickle arbitrary-code rights when what
    it actually wants is a numpy scalar. The fallback exists because the
    allowlist API is itself only present from torch 2.4, and this adapter has to
    run on the older torch a local checkout may have.

    **Registering the object alone is not enough, and silently was not.** torch
    keys its allowlist on `f"{obj.__module__}.{obj.__qualname__}"`, and under
    numpy 2 `np.core` is a shim whose members report the *new* home: the
    reconstructor registers as `numpy._core.multiarray.scalar` while the 2019
    pickle asks for `numpy.core.multiarray.scalar`. The names never match, so
    every single engine build fell through to the `weights_only=False` fallback
    below — working, but logging a warning and a traceback each time, and
    granting exactly the pickle rights this function exists to withhold.
    Verified in the deployed image: torch 2.13.0+cpu, numpy 2.0.2.

    torch >= 2.7 takes `(callable, "dotted.path")` to register under an explicit
    name, which is the supported way to say "this object, under the legacy path".
    """
    pinned = _is_pinned(path)

    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is None:          # torch < 2.4 — weights_only is off anyway
        if not pinned:
            raise BtcUnavailable(
                f"This torch has no safe-globals API, so loading {path.name} means "
                f"unpickling it with full code rights — and it is not the pinned "
                f"checkpoint. Refusing."
            )
        return torch.load(str(path), map_location="cpu")

    import numpy as np

    allowed: list = [np.dtype, np.ndarray]
    # Scalar dtypes are pickled by their concrete class (Float64DType, …).
    allowed.extend(getattr(np.dtypes, name) for name in dir(getattr(np, "dtypes", ()))
                   if name.endswith("DType"))

    scalar = getattr(getattr(np, "core", None), "multiarray", None)
    scalar = getattr(scalar, "scalar", None)
    if scalar is None:
        scalar = getattr(getattr(np, "_core", None), "multiarray", None)
        scalar = getattr(scalar, "scalar", None)

    allowed = [obj for obj in allowed if obj is not None]
    if scalar is not None:
        allowed.append(scalar)

    add_safe_globals(allowed)
    if scalar is not None:
        # Additionally under the name the checkpoint actually pickles. Guarded
        # because the tuple form is torch >= 2.7; on older torch the plain
        # registration above is all that is available, and the fallback catches
        # the rest.
        try:
            add_safe_globals([(scalar, "numpy.core.multiarray.scalar")])
        except (TypeError, ValueError):
            log.debug("this torch does not accept named safe globals; "
                      "relying on the weights_only=False fallback if needed")

    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception:
        # Deliberately broad and deliberately last: a checkpoint that needs a
        # global we did not anticipate must still load, because the alternative
        # is a deployment whose only chord engine is dead. Logged at warning so
        # the weakening is visible rather than silent.
        #
        # **Gated on the hash.** This branch grants the pickle arbitrary-code
        # rights, and "it came from a commit-pinned clone" is an argument about
        # provenance, not a check. With the pin verified it is the file we
        # reviewed; without, the honest answer is to fail the engine — a dead
        # chord engine is a clean 503, and running unrecognised code to avoid one
        # is not a trade worth making.
        if not pinned:
            raise BtcUnavailable(
                f"The BTC checkpoint at {path} could not be loaded safely and does "
                f"not match the pinned hash, so it will not be unpickled with code "
                f"rights. Re-pull the checkout, or set "
                f"CHORDS_BTC_CHECKPOINT_SHA256 if these weights are intended."
            ) from None
        log.warning("BTC checkpoint needs an unlisted pickle global; loading with "
                    "weights_only=False (hash-verified, commit-pinned checkout)",
                    exc_info=True)
        return torch.load(str(path), map_location="cpu", weights_only=False)


class BtcEngine:
    """`ChordEngine` — BTC large-vocabulary (170 labels).

    The large-vocabulary head is the right one here even though the app's grammar
    is much smaller (§12.2): asking for `maj/min` only would throw away the
    sevenths and slash chords *before* normalization gets to decide what to do
    with them, and the chart is supposed to state what was played.
    """

    name = "btc"
    version = "ismir19-large-voca"

    def __init__(self) -> None:
        self._root = _btc_root()
        self._model = None
        self._mean = None
        self._std = None
        self._labels = None
        # One instance is shared for the life of the process now
        # (`engines._cached`), so two concurrent jobs can arrive here together.
        # Loading twice is the exact cost the cache exists to remove.
        self._load_lock = threading.Lock()

    # -- lazy load ---------------------------------------------------------

    def _load(self):
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is None:
                self._load_locked()

    def _load_locked(self):
        import torch

        root = str(self._root)
        if root not in sys.path:
            # Appended, not prepended: BTC's package names (`utils`, `btc_model`)
            # are generic enough that shadowing something of ours would be a very
            # confusing bug to find.
            sys.path.append(root)

        _restore_numpy_aliases()

        from btc_model import BTC_model                       # noqa: E402
        from utils.mir_eval_modules import idx2voca_chord     # noqa: E402

        checkpoint_path = self._root / "test" / "btc_model_large_voca.pt"
        if not checkpoint_path.is_file():
            raise BtcUnavailable(f"BTC weights missing: {checkpoint_path}")

        # Their own `run_config.yaml`, read directly. BTC's `HParams.load` calls
        # `yaml.load(f)` without a Loader, which PyYAML 6 refuses — patching the
        # checkout would fix it invisibly and only on this machine, so the config
        # is read here instead and the checkout stays pristine.
        import yaml

        with (self._root / "run_config.yaml").open() as handle:
            config = yaml.safe_load(handle)["model"]

        config["num_chords"] = 170        # the large-vocabulary head
        # Logits instead of a bare argmax, so the span carries a real confidence
        # rather than a hardcoded 1.0 that would make the §5.4 confidence gate a
        # no-op for this engine.
        config["probs_out"] = True

        model = BTC_model(config=config)
        checkpoint = _load_checkpoint(torch, checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        model.eval()

        self._model = model
        self._mean = float(checkpoint["mean"])
        self._std = float(checkpoint["std"])
        self._labels = idx2voca_chord()

    # -- inference ---------------------------------------------------------

    def analyze(self, pcm: PCM, sr: int, *,
                tuning: Tuning | None = None) -> list[RawChordSpan]:
        import numpy as np
        import torch

        self._load()

        if sr != _SAMPLE_RATE:
            import librosa
            pcm = librosa.resample(np.asarray(pcm, dtype="float32"),
                                   orig_sr=sr, target_sr=_SAMPLE_RATE)

        features, times_s = _features(np, pcm, tuning or CONCERT_PITCH)
        if features.shape[0] == 0:
            return []
        features = (features - self._mean) / self._std

        probabilities = self._posteriors(np, torch, features)
        if probabilities.shape[0] == 0:
            return []

        predictions = _viterbi(np, probabilities)
        # On what this number is and is not (F4): it is a **probability**, not a
        # calibrated one. Max-softmax over 170 classes is famously overconfident,
        # and §20 applies 0.02-wide margins to it — decisions finer than the
        # signal can carry. Two things were done about that and a third was not:
        # the span now publishes the posterior of the label it actually claims
        # (below), which is at least the right quantity; and the decoder means
        # that label is chosen with the whole sequence in view rather than one
        # frame. Temperature-scaling it against a labelled set is the real fix,
        # it belongs in `bench/run_bench.py --calibration` where the labelled set
        # already is, and it cannot be judged by itself — every threshold in the
        # theory layer was tuned against *this* distribution, so recalibrating
        # means re-sweeping them in the same change.
        # The probability of **what the chart claims**, not of whatever the frame
        # liked best. Where the decoder overrules a one-frame argmax the two
        # differ, and it is the first that this span is entitled to publish: a
        # span labelled `Bm` may not carry the confidence the model had in `Bm7`.
        # It is also the more honest number for §5.4.6's gate and for §20's
        # margins, which read confidence as "how much do we believe this label".
        confidences = probabilities[np.arange(probabilities.shape[0]), predictions]

        return _spans(self._labels, predictions.tolist(), confidences.tolist(), times_s)

    def _posteriors(self, np, torch, features):
        """Per-frame 170-class posteriors for the whole track.

        BTC is a fixed-window transformer: `_TIMESTEP` frames go in, `_TIMESTEP`
        frames of logits come out, and it has no state between calls. Tiling the
        song with non-overlapping windows therefore gives every window boundary a
        frame that saw no context on one side — a chord straddling a join is read
        from half its evidence, and the model's own attention cannot reach across.

        So the windows **overlap by a quarter** and their posteriors are combined
        with a triangular weight, heaviest at each window's centre and tapering to
        nothing at its edges. Every frame is then read mostly from the window that
        saw the most context around it, and no frame is decided by a window that
        only just contains it. The last window is padded with zeros as before,
        because the recording has to end somewhere, but its padded frames are
        dropped rather than blended in.

        **Measured: chords flat, form +0.008, cost nil.** Over the chart corpus
        root and triad come out identical to three decimals and the form score
        goes 0.747 → 0.755 (Viva La Vida 0.667 → 1.000, Wonderwall +0.016 root,
        Smooth Criminal −0.023 and it is the corpus's known outlier). The reason
        the chord columns do not move is worth keeping: `_features` already gives
        every block `_context_frames()` hops of its neighbours, so the *transform*
        never saw a boundary — only the transformer's attention did, and a
        ten-second window is long against a chord.

        A third more forward passes buys that, and it is free in practice: the
        whole eleven-song feature build went 493 s to 481 s, because the CQT
        rather than the transformer is what the time goes on. Kept for the form,
        and because "no frame is decided by a window that only just contains it"
        is the property one wants when the next engine is faster.
        """
        n_frames = features.shape[0]
        if n_frames == 0:
            return np.zeros((0, 170), dtype="float32")
        stride = max(1, int(_TIMESTEP * (1.0 - _WINDOW_OVERLAP)))
        # Frame k's weight inside a window: 1 at the centre, tapering to ~0 at the
        # edges. Never exactly 0, so a frame covered by exactly one window (the
        # first and last few) still gets a usable number.
        offsets = np.arange(_TIMESTEP, dtype="float32")
        weight = 1.0 - np.abs(offsets - (_TIMESTEP - 1) / 2.0) / (_TIMESTEP / 2.0)
        weight = np.maximum(weight, 1e-3)

        total = np.zeros((n_frames, 170), dtype="float32")
        mass = np.zeros((n_frames, 1), dtype="float32")
        starts = list(range(0, max(1, n_frames - _TIMESTEP + 1), stride))
        if starts[-1] + _TIMESTEP < n_frames:
            starts.append(n_frames - _TIMESTEP)
        if n_frames < _TIMESTEP:
            starts = [0]

        with torch.no_grad():
            for start in starts:
                block = features[start:start + _TIMESTEP]
                used = block.shape[0]
                if used < _TIMESTEP:
                    block = np.pad(block, ((0, _TIMESTEP - used), (0, 0)),
                                   mode="constant")
                tensor = torch.tensor(block, dtype=torch.float32).unsqueeze(0)
                encoded, _ = self._model.self_attn_layers(tensor)
                logits = self._model.output_layer(encoded)
                window = torch.softmax(logits, dim=-1).squeeze(0).numpy()[:used]
                total[start:start + used] += window * weight[:used, None]
                mass[start:start + used] += weight[:used, None]

        return total / np.maximum(mass, 1e-6)


def _features(np, pcm, tuning: Tuning = CONCERT_PITCH):
    """log-CQT, framed in 10-second blocks. Returns `(features, frame_times_s)`.

    **The blocking is BTC's framing, not a way of transforming the audio.** BTC was
    trained on 10-second crops and its `timestep` is 108 frames, which at hop 2048
    is exactly one 10-second block — so a block is an inference window, and the
    block boundaries have to stay where they are.

    What must *not* follow from that is transforming each block in isolation, which
    is what this did. `librosa.cqt(..., center=True)` pads whatever it is handed, so
    every block's first and last ~0.52 s were computed from zero-padding instead of
    from the recording — 10% of every block, for the whole song, worst in the low
    bins where the root is. Ten seconds of good evidence, half a second of
    fabricated evidence, repeat.

    So each block is transformed with `_context_frames()` hops of its neighbours
    included and then **trimmed back to its own frames**. The framing is unchanged,
    the frame count is unchanged, and every frame now sees audio. The very start of
    the song is the one place padding remains, because there is nothing before it —
    which is also what a whole-song CQT would do there.

    The times come back alongside the features because they are **piecewise**, and
    that is the second defect here. Frame *j* of block *b* is centred at
    `b · 10 s + j · hop / sr`; the old code used a single `10.0 / 108` per frame,
    which gets each block's origin right and then runs 0.287 ms/frame slow inside
    it — a 30.7 ms sawtooth against the beat grid, resetting every block. (Using the
    true hop *globally* is worse, not better: 108 hops span 10.031 s, so it would
    drift a full second every hundred blocks.) Both terms are exact here.
    """
    import librosa

    audio = np.asarray(pcm, dtype="float32")
    # In the CQT's own bin units, which is what `cqt(tuning=...)` takes. Computed
    # once: the pitch reference is a property of the recording, not of a block,
    # and re-estimating per block would let it wander mid-song — the exact
    # failure this parameter exists to remove.
    tuning_bins = tuning.bins(_BINS_PER_OCTAVE)
    step = int(_SAMPLE_RATE * _CHUNK_S)
    context = _context_frames() * _HOP
    hop_s = _HOP / _SAMPLE_RATE

    blocks: list = []
    times: list[float] = []
    for start in range(0, len(audio), step):
        block_len = min(step, len(audio) - start)
        if block_len <= 0:
            continue
        low = max(0, start - context)
        high = min(len(audio), start + block_len + context)
        spectrum = librosa.cqt(audio[low:high], sr=_SAMPLE_RATE, n_bins=_N_BINS,
                               bins_per_octave=_BINS_PER_OCTAVE, hop_length=_HOP,
                               tuning=tuning_bins)
        # `center=True` centres frame k of this slice at sample `low + k · hop`, and
        # `start - low` is a whole number of hops by construction — so the block's
        # own frames start at exactly this offset and there is no resampling of the
        # time base.
        offset = (start - low) // _HOP
        frames = min(1 + block_len // _HOP, spectrum.shape[1] - offset)
        if frames <= 0:
            continue
        blocks.append(spectrum[:, offset:offset + frames])
        block_origin_s = start / _SAMPLE_RATE
        times.extend(block_origin_s + j * hop_s for j in range(frames))

    if not blocks:
        return np.zeros((0, _N_BINS), dtype="float32"), []
    spectrum = np.concatenate(blocks, axis=1)
    return np.log(np.abs(spectrum) + 1e-6).T.astype("float32"), times


def _viterbi(np, probabilities):
    """The most likely label *sequence*, rather than the best label per frame.

    BTC emits a full 170-class posterior for every 93 ms frame, and this adapter
    used to keep the argmax of each one and throw the distribution away. That is
    the whole of the "engine flicker" the theory layer downstream spends three
    modules cleaning up: two chords sharing three notes — `Bm` and `Bm7`, `C` and
    `Cm`, a root and its relative — trade places frame to frame whenever the
    posterior is close, and each trade becomes a span. Nothing later can tell
    that noise from music, because by then the distribution that would have said
    "these were 0.31 against 0.29" is gone.

    So the decision is taken while the evidence is still here, with the one
    assumption that is true of every song in this repertoire: **a chord persists**.
    The transition model is uniform stay-or-switch — `P(stay) = 1 - 1/L` for an
    expected chord length of `_DECODE_CHORD_FRAMES` frames, the rest shared out
    over the other 169 labels — which is deliberately not a model of harmony. It
    knows nothing about which chord follows which, and it should not: a
    transition table learned on someone else's corpus would quietly impose that
    corpus's key habits on every recording this service sees.

    Uniform switching also makes the recursion linear rather than quadratic in
    the label count. The best predecessor of state *j* is either *j* itself or
    the globally best state, so one `argmax` per frame serves all 170 — 170×
    cheaper than the textbook form, and the reason this costs a few milliseconds
    against a transformer's forty seconds.

    `chroma.py` has had this since it was written, which is the awkward part of
    the finding: the *fallback* engine was the one that decoded, and the primary
    engine was the one that did not.

    **What it is worth, measured** on the ten-song chart corpus, decoding cached
    posteriors so the transformer ran once: root 0.853 → 0.857, triad 0.846 →
    0.849, and the emitted vocabulary falls from 71 distinct chords to 64 across
    the ten songs, with a quarter of the spans gone (1624 → 1235). The accuracy
    gain is smaller than the span reduction suggests, and the reason is worth
    recording rather than explaining away: `postprocess.quantize` snaps every
    boundary to a beat and `drop_short` removes anything under one beat, so most
    single-frame flicker was already being absorbed downstream. What the decoder
    adds on top of that is the flicker *long enough to survive quantization* —
    and the cleaner span list every later stage reasons about.

    **Beat-synchronous decoding was tried here and is measurably worse.** Pooling
    the posteriors between beat times and running this same recursion over beats
    (the stronger form the audit recommended) scores root 0.809 at four beats per
    chord and 0.799 at eight, against 0.857 here — a loss of five points. It buys
    form (0.688 → 0.742) and a smaller vocabulary, because pooling plus a
    bar-length prior drags the chart toward fewer, longer chords; but it pays for
    that by flattening every change that lands off the beat the pooling window
    owns, and this pipeline already has a layer whose job is to state the form
    (§21, `canon.py`) and which does it without giving up chords. Decoding at
    frame resolution and quantizing afterwards keeps both.
    """
    n_frames, n_states = probabilities.shape
    log_emission = np.log(np.clip(probabilities, _DECODE_FLOOR, None))
    stay = np.log(1.0 - 1.0 / _DECODE_CHORD_FRAMES)
    switch = np.log((1.0 / _DECODE_CHORD_FRAMES) / (n_states - 1))

    states = np.arange(n_states)
    delta = log_emission[0].copy()
    backpointers = np.zeros((n_frames, n_states), dtype="int32")
    for t in range(1, n_frames):
        best_other = int(np.argmax(delta))
        via_switch = delta[best_other] + switch
        stay_scores = delta + stay
        take_stay = stay_scores >= via_switch
        backpointers[t] = np.where(take_stay, states, best_other)
        delta = np.where(take_stay, stay_scores, via_switch) + log_emission[t]
        delta -= delta.max()        # renormalise; only differences matter

    path = np.zeros(n_frames, dtype="int32")
    path[-1] = int(np.argmax(delta))
    for t in range(n_frames - 1, 0, -1):
        path[t - 1] = backpointers[t, path[t]]
    return path


def _spans(labels, predictions, confidences, times_s) -> list[RawChordSpan]:
    """Runs of one predicted index → one span, mean probability as confidence.

    Times come from `times_s` rather than from `index × frame_length`: the frame
    grid is piecewise (see `_features`), so there is no single frame length that is
    right everywhere in the song.
    """
    if not times_s:
        return []
    hop_s = _HOP / _SAMPLE_RATE

    def at(index: int) -> float:
        if index < len(times_s):
            return times_s[index]
        # One frame past the end — the close of the final span.
        return times_s[-1] + hop_s

    spans: list[RawChordSpan] = []
    start = 0
    for index in range(1, len(predictions) + 1):
        if index < len(predictions) and predictions[index] == predictions[start]:
            continue
        window = confidences[start:index]
        spans.append(RawChordSpan(
            start_ms=int(round(at(start) * 1000)),
            end_ms=int(round(at(index) * 1000)),
            label=labels[predictions[start]],
            confidence=float(sum(window) / len(window)) if window else 0.0,
        ))
        start = index
    return spans
