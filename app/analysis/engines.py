"""The engine registry — and the reason it is empty.

§8 step 2: *"Benchmark 2+ chord engines and 2+ beat trackers on a handful of
tracks the user picks. **Report results and let them choose** before committing."*
That choice has not been made, so **no engine is registered here yet** and
`build` answers with a clean "unavailable" rather than quietly picking one. A
default engine chosen by the person who happened to write the module is exactly
the commitment the handoff asks not to make.

What this file provides today is the seam that makes the choice cheap when it
comes: an adapter registers itself under a name, `CHORDS_CHORD_ENGINE` /
`CHORDS_BEAT_TRACKER` select it, and `name@version` is persisted on every stored
map (§5.3) so upgrading one engine invalidates only the caches it produced.

Adding an engine is three steps and no changes to anything upstream:

1. write the adapter (`analyze(pcm, sr) -> list[RawChordSpan]`, Harte or symbolic
   labels — `postprocess` normalizes them),
2. `register_chord_engine("btc", lambda: BtcEngine())`,
3. add its dependency to the `engines-btc` extra in `pyproject.toml` and to the
   worker image in `modal_app.py` — **never** to the API image (§4).

Candidates, from §5.2/§5.3, with what the benchmark has to settle:

- chords: **BTC** (strongest on pop/rock; needs a GPU — available per-function on
  Modal, but a GPU cold start per job may cost more latency than Chordino's
  accuracy gap costs quality, and this pipeline is already async),
  **Chordino/NNLS-Chroma** (predictable, no GPU), **autochord** (weaker).
- beats: **madmom** (the accuracy benchmark, effectively unmaintained, breaks on
  NumPy 2.x — hence the `<2` pin), **beat_this**, **BeatNet**,
  **librosa.beat.beat_track** (fast, weaker on downbeats).
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from typing import Callable

from ..errors import AnalysisError, CODE_FEATURE_DISABLED
from .types import BeatTracker, ChordEngine, OnsetDetector, StructureProbe

log = logging.getLogger("chords.engines")

ChordEngineFactory = Callable[[], ChordEngine]
BeatTrackerFactory = Callable[[], BeatTracker]
OnsetDetectorFactory = Callable[[], OnsetDetector]
StructureProbeFactory = Callable[[], StructureProbe]

_CHORD_ENGINES: dict[str, ChordEngineFactory] = {}
_BEAT_TRACKERS: dict[str, BeatTrackerFactory] = {}
_ONSET_DETECTORS: dict[str, OnsetDetectorFactory] = {}
_STRUCTURE_PROBES: dict[str, StructureProbeFactory] = {}


def register_chord_engine(name: str, factory: ChordEngineFactory) -> None:
    _CHORD_ENGINES[name] = factory


def register_beat_tracker(name: str, factory: BeatTrackerFactory) -> None:
    _BEAT_TRACKERS[name] = factory


def register_onset_detector(name: str, factory: OnsetDetectorFactory) -> None:
    _ONSET_DETECTORS[name] = factory


def register_structure_probe(name: str, factory: StructureProbeFactory) -> None:
    _STRUCTURE_PROBES[name] = factory


# --- built-in adapters -------------------------------------------------------
#
# `name -> (module, class, required top-level imports)`. Registration checks the
# requirements with `find_spec`, which does **not** import them: the API image
# has none of this installed (§4) and must not be made to try. The adapter module
# itself imports its dependency inside `analyze`/`track`, so even a registered
# engine costs nothing until it is built.

_BUILTIN_CHORD_ENGINES = {
    "chroma": (".adapters.chroma", "ChromaTemplateEngine", ("librosa", "numpy")),
    "chordino": (".adapters.chordino", "ChordinoEngine", ("vamp", "numpy")),
    "btc": (".adapters.btc", "BtcEngine", ("torch", "librosa")),
}

_BUILTIN_BEAT_TRACKERS = {
    "librosa": (".adapters.librosa_beats", "LibrosaBeatTracker", ("librosa", "numpy")),
    "beat_this": (".adapters.beat_this_tracker", "BeatThisTracker", ("beat_this", "torch")),
    "madmom": (".adapters.madmom_beats", "MadmomBeatTracker", ("madmom", "numpy")),
}

_BUILTIN_ONSET_DETECTORS = {
    "librosa": (".adapters.librosa_beats", "LibrosaOnsetDetector", ("librosa", "numpy")),
}

_BUILTIN_STRUCTURE_PROBES = {
    "librosa": (".adapters.librosa_energy", "LibrosaEnergyProbe", ("librosa", "numpy")),
}


def _installed(requirements: tuple[str, ...]) -> bool:
    for requirement in requirements:
        try:
            if importlib.util.find_spec(requirement) is None:
                return False
        except (ImportError, ValueError):
            return False
    return True


def _lazy(module: str, attribute: str):
    def factory():
        loaded = importlib.import_module(module, package=__package__)
        return getattr(loaded, attribute)()
    return factory


def register_builtins() -> None:
    """Register every adapter whose dependency is present in *this* image.

    Called at import so `/healthz` and `is_ready` describe what actually built —
    the same principle the health check follows everywhere else. Idempotent.
    """
    for registry, table, register in (
        (_CHORD_ENGINES, _BUILTIN_CHORD_ENGINES, register_chord_engine),
        (_BEAT_TRACKERS, _BUILTIN_BEAT_TRACKERS, register_beat_tracker),
        (_ONSET_DETECTORS, _BUILTIN_ONSET_DETECTORS, register_onset_detector),
        (_STRUCTURE_PROBES, _BUILTIN_STRUCTURE_PROBES, register_structure_probe),
    ):
        for name, (module, attribute, requirements) in table.items():
            if name in registry or not _installed(requirements):
                continue
            # The adapter module itself must exist too. Without this, an
            # unwritten adapter registers on the strength of its dependency
            # being installed, and `available()` — which /healthz publishes as
            # fact — advertises an engine that cannot be built.
            try:
                if importlib.util.find_spec(module, package=__package__) is None:
                    continue
            except (ImportError, ValueError):
                continue
            register(name, _lazy(module, attribute))


def available() -> dict[str, list[str]]:
    """What this build can actually run. Reported by `/healthz`, so a deployment
    that shipped without its engines says so rather than 500ing on first use."""
    return {
        "chords": sorted(_CHORD_ENGINES),
        "beats": sorted(_BEAT_TRACKERS),
        "onsets": sorted(_ONSET_DETECTORS),
        "structure": sorted(_STRUCTURE_PROBES),
    }


class EngineUnavailable(AnalysisError):
    """No engine is registered or configured — analysis cannot run at all.

    Deliberately the same 503 + `feature_disabled` shape as the kill switch: from
    the player's side "we haven't chosen an engine yet" and "we turned it off"
    are the same fact, and the client already renders it.
    """

    def __init__(self, message: str):
        super().__init__(message, CODE_FEATURE_DISABLED, status=503)


def _pick(kind: str, registry: dict, configured: str | None):
    if not registry:
        raise EngineUnavailable(
            f"Chord analysis isn’t available yet — no {kind} engine is installed in this build."
        )
    if configured is None:
        raise EngineUnavailable(
            f"Chord analysis isn’t available yet — no {kind} engine has been selected."
        )
    factory = registry.get(configured)
    if factory is None:
        raise EngineUnavailable(
            f"Chord analysis isn’t available yet — {kind} engine {configured!r} isn’t "
            f"installed (have: {', '.join(sorted(registry)) or 'none'})."
        )
    return factory()


def build_chord_engine(settings) -> ChordEngine:
    return _pick("chord", _CHORD_ENGINES, settings.chord_engine)


def build_beat_tracker(settings) -> BeatTracker:
    return _pick("beat", _BEAT_TRACKERS, settings.beat_tracker)


def build_onset_detector(settings) -> OnsetDetector | None:
    """Optional: without one, §14 falls back to a quarter-note downstroke bar,
    which is a song that plays rather than a song that fails."""
    if not _ONSET_DETECTORS:
        return None
    name = settings.beat_tracker if settings.beat_tracker in _ONSET_DETECTORS else None
    if name is None:
        name = sorted(_ONSET_DETECTORS)[0]
    return _ONSET_DETECTORS[name]()


def build_structure_probe(settings) -> StructureProbe | None:
    """Optional, like the onset detector, and for the same reason: without one
    every section is `Part N` (§15's honest fallback) rather than the analysis
    failing. Nothing the chart is built from depends on it."""
    if not _STRUCTURE_PROBES or not getattr(settings, "structure_probe", True):
        return None
    return _STRUCTURE_PROBES[sorted(_STRUCTURE_PROBES)[0]]()


def is_ready(settings) -> bool:
    """Whether an analysis could run right now. Used by `/healthz` and by the
    route that decides between 202-and-a-job and a clean 503."""
    return bool(
        _CHORD_ENGINES and _BEAT_TRACKERS
        and settings.chord_engine in _CHORD_ENGINES
        and settings.beat_tracker in _BEAT_TRACKERS
    )


register_builtins()
