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
  octave, hop 2048, at 22.05 kHz — computed in 10-second chunks, because that is
  how the training set was cut and `librosa.cqt` is not invariant to the length
  of what you hand it. Then `log(|CQT| + 1e-6)`, standardised by the mean and std
  stored *in the checkpoint*. Skipping that standardisation does not error; it
  just returns confident nonsense.
- **Its native sample rate is already ours.** §5.1 decodes to 22.05 kHz, which is
  what BTC trained on, so nothing is resampled in the normal path.

Output is Harte (`C:maj`, `F#:min7`, `N`, `X`), which is what `postprocess`
already normalizes — the adapter does no translation of its own.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from ..types import PCM, RawChordSpan

log = logging.getLogger("chords.engines.btc")

_ROOT_ENV = "CHORDS_BTC_ROOT"
_SAMPLE_RATE = 22050
_CHUNK_S = 10.0          # config.mp3.inst_len — the training crop length
_TIMESTEP = 108          # config.model.timestep — frames per inference window
_N_BINS = 144
_BINS_PER_OCTAVE = 24
_HOP = 2048


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
    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is None:          # torch < 2.4 — weights_only is off anyway
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
        log.warning("BTC checkpoint needs an unlisted pickle global; loading with "
                    "weights_only=False (source is the commit-pinned checkout)",
                    exc_info=True)
        return torch.load(str(path), map_location="cpu", weights_only=False)


class BtcEngine:
    """`ChordEngine` — BTC large-vocabulary (170 labels).

    The large-vocabulary head is the right one here even though the app's grammar
    is much smaller (§12.2): asking for `maj/min` only would throw away the
    sevenths and slash chords *before* normalization gets to decide what to do
    with them, and §5.5's `hard` tier exists precisely to keep some of that.
    """

    name = "btc"
    version = "ismir19-large-voca"

    def __init__(self) -> None:
        self._root = _btc_root()
        self._model = None
        self._mean = None
        self._std = None
        self._labels = None

    # -- lazy load ---------------------------------------------------------

    def _load(self):
        if self._model is not None:
            return
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

    def analyze(self, pcm: PCM, sr: int) -> list[RawChordSpan]:
        import numpy as np
        import torch

        self._load()

        if sr != _SAMPLE_RATE:
            import librosa
            pcm = librosa.resample(np.asarray(pcm, dtype="float32"),
                                   orig_sr=sr, target_sr=_SAMPLE_RATE)

        features = _features(np, pcm)
        if features.shape[0] == 0:
            return []
        features = (features - self._mean) / self._std

        pad = (-features.shape[0]) % _TIMESTEP
        if pad:
            features = np.pad(features, ((0, pad), (0, 0)), mode="constant")
        windows = features.shape[0] // _TIMESTEP

        # `_CHUNK_S / _TIMESTEP` rather than `hop / sr`: BTC's own timing derives
        # from the crop length, and the two differ by ~0.3 ms per frame — which
        # is nothing per frame and about a third of a second by the end of a
        # song, i.e. exactly the sort of slow drift §13.2's anchors would show.
        frame_s = _CHUNK_S / _TIMESTEP

        predictions: list[int] = []
        confidences: list[float] = []
        with torch.no_grad():
            tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            for index in range(windows):
                window = tensor[:, _TIMESTEP * index: _TIMESTEP * (index + 1), :]
                encoded, _ = self._model.self_attn_layers(window)
                logits = self._model.output_layer(encoded)
                probabilities = torch.softmax(logits, dim=-1).squeeze(0)
                best = torch.max(probabilities, dim=-1)
                predictions.extend(best.indices.tolist())
                confidences.extend(best.values.tolist())

        # Trim the frames that only exist because of the padding.
        if pad:
            predictions = predictions[:-pad]
            confidences = confidences[:-pad]

        return _spans(self._labels, predictions, confidences, frame_s)


def _features(np, pcm):
    """log-CQT in 10-second chunks — the training-time recipe, reproduced.

    Chunking is not an optimisation and removing it changes the answer: BTC was
    trained on 10-second crops, and `librosa.cqt` on a whole song produces
    slightly different low-frequency bins than on the crops it was fitted to.
    """
    import librosa

    audio = np.asarray(pcm, dtype="float32")
    step = int(_SAMPLE_RATE * _CHUNK_S)
    blocks = []
    for start in range(0, len(audio), step):
        block = audio[start:start + step]
        if len(block) == 0:
            continue
        blocks.append(librosa.cqt(block, sr=_SAMPLE_RATE, n_bins=_N_BINS,
                                  bins_per_octave=_BINS_PER_OCTAVE, hop_length=_HOP))
    if not blocks:
        return np.zeros((0, _N_BINS), dtype="float32")
    spectrum = np.concatenate(blocks, axis=1)
    return np.log(np.abs(spectrum) + 1e-6).T.astype("float32")


def _spans(labels, predictions, confidences, frame_s: float) -> list[RawChordSpan]:
    """Runs of one predicted index → one span, mean probability as confidence."""
    spans: list[RawChordSpan] = []
    start = 0
    for index in range(1, len(predictions) + 1):
        if index < len(predictions) and predictions[index] == predictions[start]:
            continue
        window = confidences[start:index]
        spans.append(RawChordSpan(
            start_ms=int(round(start * frame_s * 1000)),
            end_ms=int(round(index * frame_s * 1000)),
            label=labels[predictions[start]],
            confidence=float(sum(window) / len(window)) if window else 0.0,
        ))
        start = index
    return spans
