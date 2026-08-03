"""The engine adapters, on audio the test generates itself.

These do not need `bench/audio/` (which is gitignored) or any download: a chord
is a handful of sine partials, and that is enough to assert the things an adapter
can get wrong on its own — does it return spans in milliseconds, are they
ordered and non-overlapping, is the label in a vocabulary `normalize` accepts, is
the confidence a real number rather than a hardcoded 1.0.

They are **not** accuracy tests. Accuracy is `bench/run_bench.py`'s job, against
real recordings with real annotations; a sine-wave C major proves plumbing.

Each skips when its dependency is absent, so the suite still runs in an image
with no audio stack — which is the API container, and CI.
"""

from __future__ import annotations

import math

import pytest

from app.chords import normalize

librosa = pytest.importorskip("librosa", reason="the chroma/librosa adapters need librosa")
np = pytest.importorskip("numpy")

SAMPLE_RATE = 22050

# Semitones above the root, and the roots of a I–IV–V–I in C.
_MAJOR = (0, 4, 7)
_PROGRESSION = (0, 5, 7, 0)          # C, F, G, C
_CHORD_SECONDS = 2.0
_PARTIALS = ((1, 1.0), (2, 0.5), (3, 0.3), (4, 0.15))


def tone(midi: int, seconds: float, rate: int = SAMPLE_RATE):
    """One plucked-ish note: partials under an exponential decay."""
    t = np.arange(int(seconds * rate), dtype="float32") / rate
    hz = 440.0 * (2.0 ** ((midi - 69) / 12.0))
    wave = np.zeros_like(t)
    for harmonic, amplitude in _PARTIALS:
        wave += amplitude * np.sin(2 * math.pi * hz * harmonic * t)
    return wave * np.exp(-1.5 * t)


def progression(rate: int = SAMPLE_RATE):
    """I–IV–V–I in C, each chord re-struck on every beat at 120 bpm."""
    beat = 0.5
    audio = []
    for root in _PROGRESSION:
        chord = np.zeros(int(_CHORD_SECONDS * rate), dtype="float32")
        for beat_index in range(int(_CHORD_SECONDS / beat)):
            start = int(beat_index * beat * rate)
            strike = np.zeros_like(chord)
            for interval in _MAJOR:
                note = tone(60 + root + interval, _CHORD_SECONDS - beat_index * beat, rate)
                strike[start:start + len(note)] += note[:len(strike) - start]
            chord += strike
        audio.append(chord)
    out = np.concatenate(audio)
    return (out / (np.abs(out).max() + 1e-9)).astype("float32")


@pytest.fixture(scope="module")
def audio():
    return progression()


# --- the chroma template engine ---------------------------------------------

def test_chroma_returns_well_formed_spans(audio):
    from app.analysis.adapters.chroma import ChromaTemplateEngine

    spans = ChromaTemplateEngine().analyze(audio, SAMPLE_RATE)

    assert spans, "no chords at all from a four-chord progression"
    for span in spans:
        assert span.end_ms > span.start_ms
        assert 0.0 <= span.confidence <= 1.0
        assert normalize(span.label) is not None, f"unparseable label {span.label!r}"
    for earlier, later in zip(spans, spans[1:]):
        assert earlier.end_ms == later.start_ms, "spans must tile without gaps"


def test_chroma_does_not_collapse_the_whole_track_into_one_chord(audio):
    """The regression that made this engine worthless: with the emission scores
    on the wrong scale the Viterbi switch penalty won every argument and a
    three-minute song came back as three spans."""
    from app.analysis.adapters.chroma import ChromaTemplateEngine

    spans = ChromaTemplateEngine().analyze(audio, SAMPLE_RATE)
    assert len(spans) >= 3, f"over-smoothed: {[s.label for s in spans]}"


def test_chroma_hears_the_tonic(audio):
    """Loosest possible accuracy assertion — C occupies more of a I–IV–V–I than
    anything else, and an engine that cannot find that is broken, not merely
    weak."""
    from app.analysis.adapters.chroma import ChromaTemplateEngine

    spans = ChromaTemplateEngine().analyze(audio, SAMPLE_RATE)
    time_by_root: dict[int, int] = {}
    for span in spans:
        parsed = normalize(span.label)
        if parsed:
            time_by_root[parsed[0]] = time_by_root.get(parsed[0], 0) + span.duration_ms

    assert time_by_root, "nothing normalized"
    assert max(time_by_root, key=time_by_root.get) == 0, "C should dominate a I–IV–V–I in C"


def test_chroma_survives_silence():
    from app.analysis.adapters.chroma import ChromaTemplateEngine

    spans = ChromaTemplateEngine().analyze(np.zeros(SAMPLE_RATE, dtype="float32"),
                                           SAMPLE_RATE)
    assert isinstance(spans, list)


# --- the librosa beat tracker ------------------------------------------------

def test_librosa_tracker_returns_a_usable_grid(audio):
    from app.analysis.adapters.librosa_beats import LibrosaBeatTracker

    grid = LibrosaBeatTracker().track(audio, SAMPLE_RATE)

    assert grid.is_usable
    assert grid.beats_ms == sorted(grid.beats_ms)
    assert set(grid.downbeats_ms) <= set(grid.beats_ms), \
        "every downbeat must be one of the beats — §13.2 anchors are downbeats"
    assert 0.0 <= grid.confidence <= 1.0
    assert grid.time_signature in ("4/4", "3/4")


def test_librosa_tracker_is_honest_about_unusable_audio():
    """Two beats is the floor for a grid, and a grid that cannot express a bar
    must say so rather than hand back a confident empty one."""
    from app.analysis.adapters.librosa_beats import LibrosaBeatTracker

    grid = LibrosaBeatTracker().track(np.zeros(SAMPLE_RATE // 2, dtype="float32"),
                                      SAMPLE_RATE)
    assert not grid.is_usable or grid.confidence == 0.0


def test_librosa_onsets_land_inside_the_track(audio):
    from app.analysis.adapters.librosa_beats import LibrosaOnsetDetector

    onsets = LibrosaOnsetDetector().detect(audio, SAMPLE_RATE)
    duration_ms = int(len(audio) / SAMPLE_RATE * 1000)

    assert onsets
    assert all(0 <= o.t_ms <= duration_ms for o in onsets)
    assert all(o.strength >= 0.0 for o in onsets)


# --- the registry ------------------------------------------------------------

def test_the_registry_only_advertises_what_it_can_build():
    """`/healthz` publishes `available()` as fact. An engine listed there whose
    adapter or dependency is missing turns the health check into the thing it
    exists to prevent."""
    from app.analysis import engines

    for name in engines.available()["chords"]:
        assert engines._CHORD_ENGINES[name]() is not None
    for name in engines.available()["beats"]:
        assert engines._BEAT_TRACKERS[name]() is not None


def test_registering_builtins_twice_is_harmless():
    from app.analysis import engines

    before = engines.available()
    engines.register_builtins()
    assert engines.available() == before
