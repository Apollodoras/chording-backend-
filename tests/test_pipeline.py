"""The whole job, end to end, with fake engines.

This is the test the pipeline's structure was designed for: everything after
`decode` is pure, so a known chord track plus a known beat grid produces a
known song, and the assertions can be musical ("the first section is G–D–Em–C
played four times") rather than structural.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from app.analysis.pipeline import LintFailure, analyze, assemble
from app.analysis.scratch import assert_clean
from app.analysis.types import BeatGrid, EngineInfo, RawChordSpan, VideoMeta
from app.errors import (
    CODE_TEMPO_UNREADABLE,
    AnalysisError,
    TempoUnreadable,
    VideoBlocked,
    VideoTooLong,
)
from app.lint import lint, lint_sync
from app.payload import CompositionPayload
from app.store import BLOCK_CHANNEL, BLOCK_VIDEO
from tests.conftest import (
    FakeBeatTracker,
    FakeChordEngine,
    FakeOnsetDetector,
    FakeSource,
    known_chords,
    known_grid,
    known_meta,
    known_onsets,
    recording,
)


def _assemble(rec, settings):
    """The pure half, on a `conftest.recording` — for the cases that are about
    what the *features* say rather than about the job around them."""
    return assemble(meta=rec.meta, grid=rec.grid, raw=rec.chords, onsets=[],
                    settings=settings,
                    chords_engine=EngineInfo("fake-chords", "1.0"),
                    beats_engine=EngineInfo("fake-beats", "1.0"))


def run(settings, store, **overrides):
    source = overrides.pop("source", None) or FakeSource()
    return analyze(
        video_id="dQw4w9WgXcQ",
        settings=settings,
        store=store,
        source=source,
        chord_engine=overrides.pop("chord_engine", None) or FakeChordEngine(),
        beat_tracker=overrides.pop("beat_tracker", None) or FakeBeatTracker(),
        onset_detector=overrides.pop("onset_detector", FakeOnsetDetector()),
        **overrides,
    )


# --- the happy path ---------------------------------------------------------

def test_a_known_song_produces_one_chart(settings, store):
    """§5.5 used to make three of these — easy/normal/hard — and the client
    picked. The chart states what was played, so there is nothing to pick."""
    outcome = run(settings, store)
    assert isinstance(outcome.song, dict) and outcome.song


def test_the_song_is_the_progression_that_went_in(settings, store):
    outcome = run(settings, store)
    payload = CompositionPayload.model_validate(outcome.song)
    assert payload.arrangement.sections[0].chordNames == ["G", "D", "Em", "C"]
    assert payload.tonic == "G" and payload.mode == "major"
    assert payload.tempo == 120
    assert payload.timeSignature == "4/4"


def test_the_emitted_song_lints_clean(settings, store):
    """§12.4 made structural: there is no code path out of `analyze` that skips
    the lint, so this is asserting the guarantee rather than the output."""
    outcome = run(settings, store)
    assert lint(CompositionPayload.model_validate(outcome.song)) == []


def test_the_sidecar_agrees_with_the_chart(settings, store):
    outcome = run(settings, store)
    assert outcome.sync is not None
    assert lint_sync(CompositionPayload.model_validate(outcome.song), outcome.sync) == []


def test_the_strumming_pattern_is_the_one_that_was_played(settings, store):
    outcome = run(settings, store)
    payload = CompositionPayload.model_validate(outcome.song)
    strokes = payload.patterns[0].strokes
    assert [s.beat for s in strokes] == [0.0, 1.0, 1.5, 2.5, 3.0, 3.5]
    assert [s.direction for s in strokes] == ["down", "down", "up", "up", "down", "up"]


def test_with_no_onset_detector_the_song_still_plays(settings, store):
    """The app *requires* a pattern — a section without one is silently dropped.
    So no onsets means a quarter-note bar, not a failed job."""
    outcome = run(settings, store, onset_detector=None)
    payload = CompositionPayload.model_validate(outcome.song)
    assert [s.beat for s in payload.patterns[0].strokes] == [0.0, 1.0, 2.0, 3.0]


def test_the_sidecar_says_how_much_of_the_chart_is_really_there(settings, store):
    """The chart promises what was played, and on a track the container cannot
    spell it is quietly a fiction: `Gmaj9` ships as `Gmaj7`, plays fine, lints
    clean and reports high confidence. `postprocess.exact_ratio` was written for
    exactly that and then computed nowhere in production — so the one signal that
    could say so said nothing.
    """
    outcome = run(settings, store)
    assert outcome.theory.exactRatio == 1.0
    assert outcome.sync.analysis.exactRatio == 1.0

    ninths = [replace(span, label=span.label.replace(":maj", ":maj9"))
              for span in known_chords()]
    reduced = run(settings, store, chord_engine=FakeChordEngine(spans=ninths))
    assert reduced.theory.exactRatio == 0.25, "only the one chord that wasn't a 9th"
    assert lint(CompositionPayload.model_validate(reduced.song)) == []


# --- §2.1 ------------------------------------------------------------------

def test_no_audio_survives_a_successful_job(settings, store):
    run(settings, store)
    assert_clean(settings.scratch_root)


def test_no_audio_survives_a_failed_job(settings, store):
    """The failure path is the one that matters."""
    class Exploding(FakeChordEngine):
        def analyze(self, pcm, sr, *, tuning=None):
            raise RuntimeError("engine exploded")

    with pytest.raises(RuntimeError, match="engine exploded"):
        run(settings, store, chord_engine=Exploding())
    assert_clean(settings.scratch_root)


# --- the gate ---------------------------------------------------------------

def test_a_blocked_video_is_refused_before_anything_is_fetched(settings, store):
    """§2's blast-radius argument is only true if the service never downloads
    what it wasn't allowed to touch."""
    store.block(BLOCK_VIDEO, "dQw4w9WgXcQ", reason="DMCA", actor="agent")
    source = FakeSource()
    with pytest.raises(VideoBlocked):
        run(settings, store, source=source)
    assert source.decoded == []


def test_a_blocked_channel_stops_its_videos_too(settings, store):
    store.block(BLOCK_CHANNEL, "UCtest", reason="label request", actor="agent")
    source = FakeSource()
    with pytest.raises(VideoBlocked):
        run(settings, store, source=source)
    assert source.decoded == []


def test_a_video_over_the_cap_is_refused_before_it_is_fetched(settings, store):
    """§18: cap at 10 minutes. Songs, not DJ sets."""
    long_meta = VideoMeta(video_id="dQw4w9WgXcQ", title="DJ set",
                          duration_s=settings.max_video_seconds + 1, channel_id="UCtest")
    source = FakeSource(meta=long_meta)
    with pytest.raises(VideoTooLong):
        run(settings, store, source=source)
    assert source.decoded == []


def test_the_kill_switch_stops_new_analysis(settings, store):
    """§3: one config flag, no deploy."""
    from dataclasses import replace

    from app.errors import FeatureDisabled

    source = FakeSource()
    with pytest.raises(FeatureDisabled):
        run(replace(settings, analysis_enabled=False), store, source=source)
    assert source.decoded == []


# --- degrading honestly -----------------------------------------------------

def test_a_weak_beat_grid_still_ships_the_sidecar_and_says_it_is_weak(settings, store):
    """The §13.3 amendment, and the defect that forced it.

    The old rule was "a self-paced campfire song that's right beats a
    video-synced one that's wrong", and it withheld the sidecar below the
    confidence floor. What that actually did was take *playing along with the
    recording* — the whole product — away from songs whose beat map was measured
    on that very recording, and hand back the same chart with a metronome. It
    repaired nothing: the chart is no better self-paced.

    So the sidecar ships, carrying `lowConfidence: true`, and the client is the
    one that says so.
    """
    weak = FakeBeatTracker(grid=known_grid(confidence=0.2))
    outcome = run(settings, store, beat_tracker=weak)
    assert outcome.low_confidence
    assert outcome.song, "the song still lands in the Library"
    assert outcome.sync is not None, "a weak reading is still a reading"
    assert outcome.sync.lowConfidence is True


def test_low_chord_confidence_does_not_cost_the_song_its_video(settings, store):
    """Chord confidence is a claim about the *harmony*; the sidecar is a claim
    about the *timeline*. A weak read of the first has never been evidence
    against the second — the beat grid is tracked independently — and withholding
    on it conflated the two."""
    engine = FakeChordEngine(spans=known_chords(confidence=0.1))
    outcome = run(settings, store, chord_engine=engine)
    assert outcome.low_confidence
    assert outcome.sync is not None and outcome.sync.lowConfidence is True


def test_a_suspect_tempo_is_repaired_rather_than_only_flagged(settings, store):
    """206 bpm is outside what this repertoire plausibly is (55–200) and inside
    what the container accepts (40–220), so it used to ship flagged and uncorrected.

    Since the 2026-08-18 audit the repair runs by default, and 206 is exactly the
    reading it is for: a tracker counting a 103 bpm song in eighths. The song
    ships at 103, confident, on a grid the correction rebuilt.
    """
    fast = recording(ms_per_beat=291)   # 206.2 bpm
    outcome = _assemble(fast, settings)
    assert outcome.song, "the song still lands in the Library"
    assert outcome.theory.tempoOctaveShift == -1
    assert not outcome.theory.tempoOctaveSuspect
    assert not outcome.low_confidence


def test_a_suspect_tempo_still_ships_flagged_with_the_repair_off(settings, store):
    """The supported posture, and the behaviour before the audit.

    The anchors are the reason this is safe to ship rather than the reason to
    withhold: they are absolute timestamps taken off the recording, so the cursor
    follows the video whether or not the *name* we gave that tempo is an octave
    out. What a suspect octave changes is how the bars are drawn, and that is
    what `lowConfidence` is for.
    """
    fast = recording(ms_per_beat=291)   # 206.2 bpm
    outcome = _assemble(fast, replace(settings, theory_tempo_octave=False))
    assert outcome.song, "the song still lands in the Library"
    assert outcome.low_confidence
    assert outcome.sync is not None and outcome.sync.lowConfidence is True
    assert outcome.theory.tempoOctaveSuspect
    assert outcome.theory.tempoOctaveShift == 0


def test_a_tempo_the_container_cannot_carry_says_so_instead_of_failing_the_lint(settings, store):
    """The defect this replaces: `meter` flagged the octave, `lint` refused the
    tempo, and the player was told "that video didn't produce a song we could
    play" — the one sentence that names none of what the pipeline had already
    worked out.

    Reachable now only with the repair off, since 235 is exactly one octave from
    a plausible 118 and the repair takes it there. What it still guards is the
    tempo the repair *cannot* reach, which is where this error belongs.
    """
    too_fast = recording(ms_per_beat=255)   # 235.3 bpm
    with pytest.raises(TempoUnreadable) as caught:
        _assemble(too_fast, replace(settings, theory_tempo_octave=False))
    assert "235" in caught.value.message
    assert caught.value.code == CODE_TEMPO_UNREADABLE
    assert not isinstance(caught.value, LintFailure)


def test_the_octave_correction_repairs_a_song_that_had_no_song_at_all(settings, store):
    """§20.2, on by default since the 2026-08-18 audit (F24).

    It rewrites every bar line in the song, which is why it was off — but the only
    recordings it can fire on are the ones `assemble` otherwise refuses outright,
    so what it risks is measured against nothing rather than against a song that
    was working. The user paid a quota charge for that refusal."""
    too_fast = recording(ms_per_beat=255)
    outcome = _assemble(too_fast, settings)
    payload = CompositionPayload.model_validate(outcome.song)
    assert payload.tempo == 118            # 235.3 / 2
    assert outcome.theory.tempoOctaveShift == -1
    assert not outcome.theory.tempoOctaveSuspect
    assert lint(payload) == []


def test_an_unusable_grid_fails_with_something_the_player_can_read(settings, store):
    tracker = FakeBeatTracker(grid=BeatGrid(beats_ms=[0], downbeats_ms=[0],
                                            bpm=120.0, confidence=0.9))
    with pytest.raises(AnalysisError) as caught:
        run(settings, store, beat_tracker=tracker)
    assert "rhythm" in caught.value.message


def test_a_track_with_no_readable_chords_fails_rather_than_shipping_nothing(settings, store):
    engine = FakeChordEngine(spans=[RawChordSpan(start_ms=0, end_ms=32000, label="N")])
    with pytest.raises(AnalysisError, match="No chords"):
        run(settings, store, chord_engine=engine)


# --- N/C --------------------------------------------------------------------

def test_a_no_chord_gap_is_held_rather_than_left_empty(settings, store):
    """§18: hold the previous chord. A section with empty chords is silently
    dropped by the importer."""
    spans = [
        RawChordSpan(start_ms=0, end_ms=8000, label="G:maj"),
        RawChordSpan(start_ms=8000, end_ms=16000, label="N"),
        RawChordSpan(start_ms=16000, end_ms=32000, label="C:maj"),
    ]
    outcome = run(settings, store, chord_engine=FakeChordEngine(spans=spans))
    payload = CompositionPayload.model_validate(outcome.song)
    for section in payload.arrangement.sections:
        assert section.chordNames or section.bars


# --- determinism ------------------------------------------------------------

def test_the_same_recording_produces_byte_identical_songs(settings, store):
    """Analysis is deterministic, and so is its output — including the ids.

    §12.5 needs the song id and the pattern ids stable, and this goes further:
    section, bar and stroke ids are derived from content too. That makes "did the
    analysis change?" answerable by diffing, keeps the §16.5 contract fixtures
    from churning in git, and means a re-analysis genuinely *replaces* a Library
    row rather than replacing it with a differently-identified equivalent.
    """
    first = run(settings, store)
    second = run(settings, store)
    assert first.song == second.song


def test_a_changed_groove_gets_a_new_pattern_id_but_the_song_keeps_its_own(settings, store):
    quiet = run(settings, store, onset_detector=None)
    strummed = run(settings, store)
    assert quiet.song["id"] == strummed.song["id"]
    assert quiet.song["patternID"] != strummed.song["patternID"]


# --- the pure half ----------------------------------------------------------

def test_assemble_needs_no_audio_engine_or_network(settings):
    """The property that makes all of the above runnable in CI before an engine
    has even been chosen."""
    outcome = assemble(
        meta=known_meta(), grid=known_grid(), raw=known_chords(), onsets=known_onsets(),
        settings=settings,
        chords_engine=EngineInfo("btc", "1.2.0"),
        beats_engine=EngineInfo("beat_this", "0.3.1"),
    )
    assert outcome.engine_chords == "btc@1.2.0"
    assert outcome.engine_beats == "beat_this@0.3.1"
    assert outcome.sync.engine.chords == "btc@1.2.0"


# --- the sidecar ------------------------------------------------------------

def test_an_advisory_problem_ships_the_sidecar_flagged_rather_than_withholding_it(
        settings, store, monkeypatch):
    """The change the fatal/advisory split exists for.

    An advisory problem — a ragged bar, a tail short of the last anchor, a
    section that is not whole bars — used to be indistinguishable from a fatal
    one, and *any* of them withheld the sidecar. Those are the songs that reached
    players with no "play with the video" option at all.

    Now the map ships and says it is imperfect, which is the only version of this
    the player can act on.
    """
    from app.analysis import pipeline
    from app.lint import SyncProblems

    monkeypatch.setattr(pipeline, "lint_sync_problems",
                        lambda payload, sync: SyncProblems(advisory=("ragged bars",)))
    outcome = run(settings, store)

    assert outcome.sync is not None
    assert outcome.sync.lowConfidence is True, "shipped, and honest about it"


def test_only_a_fatal_leaves_a_song_without_video_sync(settings, store, monkeypatch):
    """`sync is None` still means "this analysis has no usable video sync" — it
    is just far harder to reach than it was, because only anchors that cannot be
    interpolated get there. The *song* is unaffected either way."""
    from app.analysis import pipeline
    from app.lint import SyncProblems

    monkeypatch.setattr(pipeline, "lint_sync_problems",
                        lambda payload, sync: SyncProblems(fatal=("nope",)))
    outcome = run(settings, store)

    assert outcome.sync is None
    assert outcome.song, "the chart still ships"


def test_a_low_confidence_song_still_ships_video_sync(settings, store):
    """The rule the owner set: a song that came from a recording plays *with* the
    recording. Confidence decides what the client is told, never whether the
    sidecar exists."""
    weak = replace(settings, confidence_floor=0.99)
    outcome = run(weak, store)

    assert outcome.low_confidence
    assert outcome.sync is not None
    assert outcome.sync.lowConfidence is True


def test_a_song_with_an_unusable_map_is_filed_without_a_sidecar(settings, store, monkeypatch):
    """End to end through `run_job`, because the decision is only worth anything
    if the store write honours it."""
    from app.analysis import pipeline
    from app import jobs
    from app.jobs import run_job

    from app.lint import SyncProblems

    monkeypatch.setattr(pipeline, "lint_sync_problems",
                        lambda payload, sync: SyncProblems(fatal=("beatAnchors is empty",)))
    # `run_job` builds its engines from the registry, which this module does not
    # populate — the fakes are passed straight to `analyze` everywhere else here.
    monkeypatch.setattr(jobs.engines, "build_chord_engine", lambda _s: FakeChordEngine())
    monkeypatch.setattr(jobs.engines, "build_beat_tracker", lambda _s: FakeBeatTracker())
    monkeypatch.setattr(jobs.engines, "build_onset_detector", lambda _s: FakeOnsetDetector())
    monkeypatch.setattr(jobs.engines, "build_structure_probe", lambda _s: None)

    store.create_job(job_id="j1", uid="u1", video_id="dQw4w9WgXcQ")
    run_job(job_id="j1", video_id="dQw4w9WgXcQ", uid="u1",
            settings=settings, store=store, source=FakeSource())

    filed = store.get_map("dQw4w9WgXcQ")
    assert filed is not None and filed.sync is None


# --- a dirty chart is not a chart -------------------------------------------

def test_a_song_that_would_warn_on_import_fails_rather_than_shipping():
    """§12.4 made structural. There used to be three renders here and a dirty one
    was *withheld* so its clean siblings could ship; with one chart there is no
    sibling to spare, so nothing to ship and the lint's own words are the most
    useful thing to raise."""
    from app.analysis import pipeline as pipeline_module
    from app.config import Settings

    rec = recording()
    real_lint = pipeline_module.lint
    pipeline_module.lint = lambda payload: ["the chart is dirty"]
    try:
        with pytest.raises(LintFailure) as caught:
            _assemble(rec, Settings(scratch_root="/tmp/chords-scratch"))
    finally:
        pipeline_module.lint = real_lint
    assert caught.value.problems == ["the chart is dirty"]
