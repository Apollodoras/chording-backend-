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


# --- BTC's feature extractor -------------------------------------------------
#
# The model itself needs a 12 MB checkout and torch, so these test `_features`
# directly. It is the right seam anyway: what was wrong was arithmetic about
# windows and frame times, not anything the transformer does with them.

def _chord_bed(seconds: float = 35.0) -> "np.ndarray":
    """Low-register chords — fundamentals in the bottom two CQT octaves, which is
    where the defect lived and where a chord recognizer reads the root."""
    sr = 22050
    t = np.arange(int(sr * seconds)) / sr
    signal = np.zeros_like(t)
    for index, root in enumerate((65.41, 87.31, 98.00, 82.41)):      # C2 F2 G2 E2
        segment = (t >= index * 8) & (t < (index + 1) * 8)
        for harmonic, amplitude in ((1, 1.0), (2, 0.5), (3, 0.3), (4, 0.2)):
            signal[segment] += amplitude * np.sin(2 * np.pi * root * harmonic * t[segment])
    return (signal / np.abs(signal).max()).astype("float32")


def _isolated_blocks(audio):
    """What `_features` used to do: `librosa.cqt` per 10-second block, concatenated.

    Kept here as the thing being improved on, so the assertion below is a
    comparison rather than a bare threshold nobody can calibrate.
    """
    from app.analysis.adapters.btc import (
        _BINS_PER_OCTAVE,
        _CHUNK_S,
        _HOP,
        _N_BINS,
        _SAMPLE_RATE,
    )

    step = int(_SAMPLE_RATE * _CHUNK_S)
    blocks = []
    for start in range(0, len(audio), step):
        block = audio[start:start + step]
        if len(block) == 0:
            continue
        blocks.append(librosa.cqt(block, sr=_SAMPLE_RATE, n_bins=_N_BINS,
                                  bins_per_octave=_BINS_PER_OCTAVE, hop_length=_HOP))
    return np.log(np.abs(np.concatenate(blocks, axis=1)) + 1e-6).T


def test_block_edges_see_audio_rather_than_padding():
    """The defect, measured.

    BTC's 10-second blocks are its *framing* — one block is exactly one inference
    window — but each block was also **transformed in isolation**, and
    `librosa.cqt(center=True)` pads whatever it is handed. At 24 bins per octave the
    lowest bin's analysis window is ~1.04 s, so roughly 0.52 s at each block edge was
    computed from zero-padding instead of from the recording: about 10% of every
    block, for the whole song, worst in the bass where the root lives.

    Scored against a single CQT over the whole signal, which is the ground truth for
    "what does this audio actually look like". Block interiors were always close to
    it; the edges were not, and now are.
    """
    from app.analysis.adapters.btc import (
        _BINS_PER_OCTAVE,
        _CHUNK_S,
        _HOP,
        _N_BINS,
        _SAMPLE_RATE,
        _context_frames,
        _features,
    )

    audio = _chord_bed()
    whole = np.log(np.abs(librosa.cqt(
        audio, sr=_SAMPLE_RATE, n_bins=_N_BINS,
        bins_per_octave=_BINS_PER_OCTAVE, hop_length=_HOP)) + 1e-6).T

    fixed, _times = _features(np, audio)
    old = _isolated_blocks(audio)
    frames = min(len(whole), len(fixed), len(old))

    per_block = 1 + int(_SAMPLE_RATE * _CHUNK_S) // _HOP
    context = _context_frames()
    # Frames within a context window of a block seam. The song's own start is
    # excluded: there is genuinely no audio before it, so padding there is what a
    # whole-song CQT does too.
    edges = [i for i in range(frames)
             if not (i // per_block == 0 and i % per_block < context)
             and (i % per_block < context or i % per_block >= per_block - context)]
    interiors = [i for i in range(frames) if i not in set(edges)]
    bass = slice(0, 48)          # the bottom two octaves

    def error(features, rows):
        return float(np.abs(features[rows][:, bass] - whole[rows][:, bass]).mean())

    old_edge, new_edge = error(old, edges), error(fixed, edges)
    interior = error(fixed, interiors)

    assert new_edge < old_edge / 3, (
        f"block edges still disagree with the recording: {new_edge:.4f} vs the old "
        f"{old_edge:.4f}"
    )
    # The real bar: an edge frame should be no worse than an interior one. Anything
    # else means there is still a seam.
    assert new_edge < interior * 1.5, (
        f"edge error {new_edge:.4f} is still worse than interior error {interior:.4f}"
    )


def test_the_framing_is_unchanged_by_the_context():
    """The context is trimmed away again, so BTC still sees its own windows.

    A block has to be exactly `timestep` frames — 108 at hop 2048 — because that is
    what one inference window is. Handing the transform more audio must not change
    the frame count, or the fix would have quietly re-cut the training-time framing.
    """
    from app.analysis.adapters.btc import _CHUNK_S, _HOP, _SAMPLE_RATE, _TIMESTEP, _features

    features, times = _features(np, _chord_bed(seconds=30.0))

    assert 1 + int(_SAMPLE_RATE * _CHUNK_S) // _HOP == _TIMESTEP
    assert len(features) == len(times) == 3 * _TIMESTEP


def test_frame_times_are_exact_at_both_ends_of_every_block():
    """The 30.7 ms sawtooth.

    Frame *j* of block *b* is centred at `b · 10 s + j · hop / sr`. The old code used
    a single `10.0 / 108` per frame, which gets each block's origin right and then
    runs 0.287 ms/frame slow inside it — an error growing to 30.7 ms by the end of
    every block and resetting at the next, against the beat grid the chart quantizes
    to.

    Using the true hop *globally* is worse rather than better: 108 hops span 10.031 s,
    so it would drift a full second every hundred blocks. Both terms have to be
    right, which is why the times are built where the block structure is known
    instead of derived from a frame index.
    """
    from app.analysis.adapters.btc import _CHUNK_S, _HOP, _SAMPLE_RATE, _TIMESTEP, _features

    _features_out, times = _features(np, _chord_bed(seconds=30.0))

    hop_s = _HOP / _SAMPLE_RATE
    truth = [(i // _TIMESTEP) * _CHUNK_S + (i % _TIMESTEP) * hop_s for i in range(len(times))]
    naive = [i * (_CHUNK_S / _TIMESTEP) for i in range(len(times))]

    assert max(abs(a - b) for a, b in zip(times, truth)) < 1e-9
    # And the old scheme really was off by the amount claimed, so this test is
    # measuring a fix rather than restating a tautology.
    assert 0.030 < max(abs(a - b) for a, b in zip(naive, truth)) < 0.031


def test_silence_shorter_than_one_hop_is_not_a_crash():
    """`_features` is reached with whatever `decode` produced, and a near-empty
    decode is a real outcome (a video that is 0.1 s of nothing)."""
    from app.analysis.adapters.btc import _features

    features, times = _features(np, np.zeros(64, dtype="float32"))
    assert len(features) == len(times)


def test_spans_close_on_the_last_frame_rather_than_off_the_end():
    """A span's end is the *next* frame's time, and the final span has no next
    frame. Reading past the end used to be impossible only because the times were
    computed arithmetically; now they are a list, so the boundary is real code."""
    from app.analysis.adapters.btc import _HOP, _SAMPLE_RATE, _spans

    times = [0.0, 0.1, 0.2]
    spans = _spans(["C:maj", "G:maj"], [0, 0, 1], [0.9, 0.9, 0.8], times)

    assert [(s.start_ms, s.end_ms) for s in spans] == [
        (0, 200), (200, int(round((0.2 + _HOP / _SAMPLE_RATE) * 1000))),
    ]
    assert _spans(["C:maj"], [], [], []) == []


# --- the engine cache --------------------------------------------------------

def test_an_engine_is_built_once_and_reused():
    """`run_job` builds all four engines per job, and `_lazy` used to construct a
    fresh adapter each time — whose `_load()` caches on `self`, so a new instance
    meant re-reading BTC's 12 MB checkpoint, rebuilding the transformer, re-deriving
    the 170-label vocabulary and reconstructing Beat This!'s model. Modal reuses warm
    containers across inputs, so that was seconds of waste on every job in a
    container that had already done all of it."""
    from app.analysis import engines
    from app.config import Settings

    built = []

    class Counted:
        name, version = "counted", "1"

        def __init__(self):
            built.append(1)

        def analyze(self, pcm, sr, *, tuning=None):
            return []

    engines.register_chord_engine("counted", Counted)
    try:
        settings = Settings(chord_engine="counted")
        first = engines.build_chord_engine(settings)
        second = engines.build_chord_engine(settings)

        assert first is second
        assert len(built) == 1
    finally:
        engines._CHORD_ENGINES.pop("counted", None)
        engines.reset_engine_cache()


def test_re_registering_a_name_drops_the_instance_it_cached():
    """Otherwise a cache turns into a correctness bug in the least expected place:
    a suite that registers a fake under a name it has used before would keep
    exercising the *previous* fake."""
    from app.analysis import engines
    from app.config import Settings

    class First:
        name, version = "swap", "1"

    class Second:
        name, version = "swap", "2"

    engines.register_chord_engine("swap", First)
    try:
        assert engines.build_chord_engine(Settings(chord_engine="swap")).version == "1"
        engines.register_chord_engine("swap", Second)
        assert engines.build_chord_engine(Settings(chord_engine="swap")).version == "2"
    finally:
        engines._CHORD_ENGINES.pop("swap", None)
        engines.reset_engine_cache()


def test_the_structure_probe_knob_is_a_real_setting():
    """It was read through `getattr(settings, "structure_probe", True)` against a
    `Settings` that had no such field, so the knob could never be anything but on —
    and a `getattr` default reads exactly like a working setting at the call site."""
    from dataclasses import fields

    from app.analysis import engines
    from app.config import Settings

    assert "structure_probe" in {f.name for f in fields(Settings)}
    assert engines.build_structure_probe(Settings(structure_probe=False)) is None


# --- the checkpoint is pinned by hash, not only by commit --------------------

_CHECKPOINT = (
    __import__("pathlib").Path(__file__).resolve().parents[1]
    / "vendor" / "BTC-ISMIR19" / "test" / "btc_model_large_voca.pt"
)


@pytest.mark.skipif(not _CHECKPOINT.is_file(),
                    reason="needs the BTC checkout's weights")
def test_the_pinned_hash_matches_the_checkpoint_we_reviewed():
    """`BTC_CHECKPOINT_SHA256` has to actually be this file's hash.

    A pin that does not match is worse than no pin: it silently disables the
    `weights_only=False` fallback on a healthy deployment, and the only chord engine
    dies with a message about hashes.
    """
    from app.analysis.adapters.btc import BTC_CHECKPOINT_SHA256, _is_pinned

    assert _is_pinned(_CHECKPOINT)
    assert len(BTC_CHECKPOINT_SHA256) == 64


def test_an_unrecognised_checkpoint_is_not_granted_code_rights(tmp_path, caplog):
    """`weights_only=False` is "run whatever code this pickle asks for".

    The commit pin says *which* checkpoint; it does not say the bytes arrived intact
    — a clone is a network fetch, and the fallback would happily execute a
    substituted file. Provenance is an argument, not a check.
    """
    from app.analysis.adapters.btc import _is_pinned

    impostor = tmp_path / "btc_model_large_voca.pt"
    impostor.write_bytes(b"not the weights")

    assert not _is_pinned(impostor)
    # And it says so loudly, naming the override for an operator who meant it.
    assert any("CHORDS_BTC_CHECKPOINT_SHA256" in record.message
               for record in caplog.records)


def test_an_operator_can_pin_different_weights(tmp_path, monkeypatch):
    """Running other weights is legitimate — a fine-tune, a newer release. It just
    has to be stated rather than assumed."""
    import hashlib

    from app.analysis.adapters.btc import _is_pinned

    other = tmp_path / "other.pt"
    other.write_bytes(b"deliberately different")
    monkeypatch.setenv("CHORDS_BTC_CHECKPOINT_SHA256",
                       hashlib.sha256(b"deliberately different").hexdigest())

    assert _is_pinned(other)


def test_an_unhashable_file_reads_as_unverified(tmp_path):
    """Hashing failure must withhold a privilege, never grant one."""
    from app.analysis.adapters.btc import _is_pinned

    assert not _is_pinned(tmp_path / "does-not-exist.pt")


# --- BTC's decoder ------------------------------------------------------------

def test_the_decoder_removes_a_one_frame_flicker():
    """F2, and the owner's first symptom at its source.

    Twenty frames of `Bm` with one frame in the middle where `Bm7` wins the
    argmax by a hair. Per-frame, that is three spans and a chord change the song
    does not have; decoded, it is one chord, because a single frame is not enough
    evidence to pay the transition cost.
    """
    import numpy as np
    from app.analysis.adapters.btc import _viterbi

    probabilities = np.full((20, 4), 0.01, dtype="float32")
    probabilities[:, 0] = 0.9                # "Bm" everywhere
    probabilities[10, 0], probabilities[10, 1] = 0.45, 0.46   # ...except one frame

    assert probabilities.argmax(axis=1)[10] == 1, "the argmax really does flicker"
    assert set(_viterbi(np, probabilities).tolist()) == {0}, "and the decoder does not"


def test_the_decoder_still_hears_a_real_chord_change():
    """The other half: a change the evidence supports has to survive. Ten frames
    of one chord and ten of another is a bar of each at this frame rate, and no
    stay-put prior worth having flattens that."""
    import numpy as np
    from app.analysis.adapters.btc import _viterbi

    probabilities = np.full((20, 4), 0.01, dtype="float32")
    probabilities[:10, 0] = 0.9
    probabilities[10:, 1] = 0.9

    path = _viterbi(np, probabilities).tolist()
    assert path == [0] * 10 + [1] * 10


# --- §14.1: which band an attack arrived in ---------------------------------
#
# These are plumbing tests in the sense the module docstring means: a sine-wave
# bass note is not a recording. What they can prove is the thing the label is
# actually built on — that two attacks an octave and a half apart end up on
# opposite sides of the split, and that one attack covering both ends up on
# neither.

def _struck(events, seconds: float, rate: int = SAMPLE_RATE):
    """`events` = [(start_seconds, [midi notes])] → one mono buffer."""
    out = np.zeros(int(seconds * rate), dtype="float32")
    for start, notes in events:
        at = int(start * rate)
        for note in notes:
            wave = tone(note, seconds - start, rate)
            room = min(len(wave), len(out) - at)
            out[at:at + room] += wave[:room]
    return (out / (np.abs(out).max() + 1e-9)).astype("float32")


def test_a_bass_note_and_a_chord_over_it_are_labelled_apart():
    """An oom-pah: a root octave on beats 1 and 3, a right hand on 2 and 4."""
    from app.analysis.adapters.librosa_beats import HarmonicOnsetDetector

    beat, bars = 0.5, 4
    events = []
    for bar in range(bars):
        base = bar * 4 * beat
        for offset in (0.0, 2.0):
            events.append((base + offset * beat, [36, 48]))          # C2 + C3
        for offset in (1.0, 3.0):
            events.append((base + offset * beat, [72, 76, 79]))      # C5 E5 G5
    onsets = HarmonicOnsetDetector().detect(_struck(events, bars * 4 * beat + 1.0),
                                            SAMPLE_RATE)
    bands = [o.band for o in onsets]
    assert "low" in bands and "mid" in bands
    # Every labelled attack is on the side it was played on: nothing in the bass
    # band came back chordal, and nothing in the right hand came back as bass.
    for onset in onsets:
        played_bass = round((onset.t_ms / 1000.0) / beat) % 2 == 0
        assert onset.band in ("low" if played_bass else "mid", "full"), onset


def test_a_full_range_strum_is_not_labelled_as_either_band():
    """The commonest stroke there is, and the one that must stay `full`: a chord
    voiced across the split moves both bands, so neither owns it."""
    from app.analysis.adapters.librosa_beats import HarmonicOnsetDetector

    # Struck a second apart. `tone` decays slowly enough that strikes half a
    # second apart overlap into a second detection each, and a decay tail loses
    # its bass before its treble — so a denser stimulus measures the shape of the
    # test signal rather than the shape of the rule.
    events = [(float(i), [40, 47, 52, 56, 59, 64]) for i in range(8)]    # E2…E4
    onsets = HarmonicOnsetDetector().detect(_struck(events, 9.0), SAMPLE_RATE)
    assert len(onsets) == len(events)
    assert all(o.band == "full" for o in onsets)


def test_a_detector_with_no_signal_to_look_at_claims_nothing():
    """`_bands` is additive by construction — without audio to split, every
    attack keeps the `full` that an unlabelled onset has always meant."""
    from app.analysis.adapters.librosa_beats import _bands

    assert _bands(None, SAMPLE_RATE, [10, 20, 30]) == ["full"] * 3
