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
from app.chords import EASY, HARD, NORMAL
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

def test_a_known_song_produces_all_three_difficulties(settings, store):
    """§5.5: compute once, store all three, let the client pick. Never
    re-analyze per difficulty."""
    outcome = run(settings, store)
    assert sorted(outcome.songs) == [EASY, HARD, NORMAL]


def test_the_song_is_the_progression_that_went_in(settings, store):
    outcome = run(settings, store)
    payload = CompositionPayload.model_validate(outcome.songs[NORMAL])
    assert payload.arrangement.sections[0].chordNames == ["G", "D", "Em", "C"]
    assert payload.tonic == "G" and payload.mode == "major"
    assert payload.tempo == 120
    assert payload.timeSignature == "4/4"


def test_every_emitted_song_lints_clean(settings, store):
    """§12.4 made structural: there is no code path out of `analyze` that skips
    the lint, so this is asserting the guarantee rather than the output."""
    outcome = run(settings, store)
    for wire in outcome.songs.values():
        assert lint(CompositionPayload.model_validate(wire)) == []


def test_the_sidecar_agrees_with_every_tier(settings, store):
    outcome = run(settings, store)
    assert outcome.sync is not None
    for wire in outcome.songs.values():
        assert lint_sync(CompositionPayload.model_validate(wire), outcome.sync) == []


def test_the_strumming_pattern_is_the_one_that_was_played(settings, store):
    outcome = run(settings, store)
    payload = CompositionPayload.model_validate(outcome.songs[NORMAL])
    strokes = payload.patterns[0].strokes
    assert [s.beat for s in strokes] == [0.0, 1.0, 1.5, 2.5, 3.0, 3.5]
    assert [s.direction for s in strokes] == ["down", "down", "up", "up", "down", "up"]


def test_with_no_onset_detector_the_song_still_plays(settings, store):
    """The app *requires* a pattern — a section without one is silently dropped.
    So no onsets means a quarter-note bar, not a failed job."""
    outcome = run(settings, store, onset_detector=None)
    payload = CompositionPayload.model_validate(outcome.songs[NORMAL])
    assert [s.beat for s in payload.patterns[0].strokes] == [0.0, 1.0, 2.0, 3.0]


def test_the_easy_tier_has_no_more_chord_changes_than_the_hard_one(settings, store):
    outcome = run(settings, store)
    easy = CompositionPayload.model_validate(outcome.songs[EASY])
    hard = CompositionPayload.model_validate(outcome.songs[HARD])
    assert len(easy.arrangement.sections) <= len(hard.arrangement.sections)


def test_the_sidecar_says_how_much_of_the_hard_tier_is_really_there(settings, store):
    """`hard` promises "the full detected quality", and on a track the container
    cannot spell it is quietly a fiction: `Gmaj9` ships as `Gmaj7`, plays fine,
    lints clean and reports high confidence. `postprocess.exact_ratio` was
    written for exactly that and then computed nowhere in production — so the one
    signal that could say so said nothing.
    """
    outcome = run(settings, store)
    assert outcome.theory.exactRatio == 1.0
    assert outcome.sync.analysis.exactRatio == 1.0

    ninths = [replace(span, label=span.label.replace(":maj", ":maj9"))
              for span in known_chords()]
    reduced = run(settings, store, chord_engine=FakeChordEngine(spans=ninths))
    assert reduced.theory.exactRatio == 0.25, "only the one chord that wasn't a 9th"
    assert lint(CompositionPayload.model_validate(reduced.songs[HARD])) == []


# --- §2.1 ------------------------------------------------------------------

def test_no_audio_survives_a_successful_job(settings, store):
    run(settings, store)
    assert_clean(settings.scratch_root)


def test_no_audio_survives_a_failed_job(settings, store):
    """The failure path is the one that matters."""
    class Exploding(FakeChordEngine):
        def analyze(self, pcm, sr):
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

def test_a_weak_beat_grid_withholds_the_sidecar_but_keeps_the_song(settings, store):
    """§13.3: "a self-paced campfire song that's right beats a video-synced one
    that's wrong — the player has no scoring to protect, so the only cost of bad
    sync is that it feels broken"."""
    weak = FakeBeatTracker(grid=known_grid(confidence=0.2))
    outcome = run(settings, store, beat_tracker=weak)
    assert outcome.sync is None
    assert outcome.low_confidence
    assert outcome.songs, "the song still lands in the Library"


def test_low_chord_confidence_also_withholds_the_sidecar(settings, store):
    engine = FakeChordEngine(spans=known_chords(confidence=0.1))
    outcome = run(settings, store, chord_engine=engine)
    assert outcome.sync is None and outcome.low_confidence


def test_a_suspect_tempo_the_container_can_still_carry_ships_low_confidence(settings, store):
    """206 bpm is outside what this repertoire plausibly is and inside what the
    container accepts. So the song ships — but the axis it plays on is the thing
    in doubt, and the sidecar is what would put a cursor on that axis."""
    fast = recording(ms_per_beat=291)   # 206.2 bpm
    outcome = _assemble(fast, settings)
    assert outcome.songs, "the song still lands in the Library"
    assert outcome.low_confidence
    assert outcome.sync is None
    assert outcome.theory.tempoOctaveSuspect
    assert outcome.theory.tempoOctaveShift == 0


def test_a_tempo_the_container_cannot_carry_says_so_instead_of_failing_the_lint(settings, store):
    """The defect this replaces: `meter` flagged the octave, `lint` refused the
    tempo, and the player was told "that video didn't produce a song we could
    play" — the one sentence that names none of what the pipeline had already
    worked out."""
    too_fast = recording(ms_per_beat=255)   # 235.3 bpm
    with pytest.raises(TempoUnreadable) as caught:
        _assemble(too_fast, settings)
    assert "235" in caught.value.message
    assert caught.value.code == CODE_TEMPO_UNREADABLE
    assert not isinstance(caught.value, LintFailure)


def test_the_octave_correction_is_off_until_it_is_asked_for(settings, store):
    """It rewrites every bar line in the song and the corpus has no track that
    triggers it, so there is no measurement to turn it on with. With it on, the
    same recording that had no song at all is a song at half the tempo."""
    too_fast = recording(ms_per_beat=255)
    outcome = _assemble(too_fast, replace(settings, theory_tempo_octave=True))
    payload = CompositionPayload.model_validate(outcome.songs[HARD])
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
    payload = CompositionPayload.model_validate(outcome.songs[NORMAL])
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
    for tier in first.songs:
        assert first.songs[tier] == second.songs[tier]


def test_a_changed_groove_gets_a_new_pattern_id_but_the_song_keeps_its_own(settings, store):
    quiet = run(settings, store, onset_detector=None)
    strummed = run(settings, store)
    assert quiet.songs[NORMAL]["id"] == strummed.songs[NORMAL]["id"]
    assert quiet.songs[NORMAL]["patternID"] != strummed.songs[NORMAL]["patternID"]


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


# --- the sidecar is per difficulty -------------------------------------------

def test_the_sidecar_records_which_tiers_it_is_true_of(settings, store):
    """One sidecar object, and a set of the tiers whose chart agrees with it.

    The sidecar describes the *recording* — anchors, tempo, bar grid — so it is the
    same whichever tier the player asked for. What is per-tier is whether that
    tier's chart still lines up with it, and the store is already keyed on
    `(video_id, difficulty)`.
    """
    outcome = run(settings, store)

    assert outcome.sync is not None
    assert outcome.sync_tiers == frozenset(outcome.songs)
    for tier in outcome.songs:
        assert outcome.sync_for(tier) is outcome.sync


def test_one_disagreeing_tier_no_longer_costs_the_others_their_sync(settings, store,
                                                                    monkeypatch):
    """The defect.

    `impose` is allowed to drop a section from one tier — `easy`'s
    drop-passing-chords rule can merge across a barline — so a tier disagreeing with
    the sidecar is by design rather than a symptom. The check used to `break` on the
    first failure and withhold the sidecar from **every** tier, punishing two renders
    for a third one's shape, and the log line named only the tier that failed so the
    two that were fine looked untouched.

    Video sync is the product's differentiator; it should not be all-or-nothing
    across renders that are allowed to differ.
    """
    from app.analysis import pipeline

    real_lint_sync = pipeline.lint_sync

    def only_easy_disagrees(payload, sync):
        # The tier is in the song id (`yt:<videoId>:<difficulty>`, §12.5) — the
        # payload has no difficulty field of its own.
        if payload.id.endswith(f":{EASY}"):
            return ["bar count disagrees with the sidecar"]
        return real_lint_sync(payload, sync)

    monkeypatch.setattr(pipeline, "lint_sync", only_easy_disagrees)
    outcome = run(settings, store)

    assert EASY not in outcome.sync_tiers
    assert {NORMAL, HARD} <= outcome.sync_tiers
    assert outcome.sync is not None, \
        "the sidecar is still true of two tiers, so it is still worth having"
    assert outcome.sync_for(EASY) is None


def test_no_tier_agreeing_still_means_no_sidecar_at_all(settings, store, monkeypatch):
    """`sync is None` has to keep meaning exactly what it always meant — this
    analysis has no usable video sync — or every reader of it becomes subtly
    optimistic."""
    from app.analysis import pipeline

    monkeypatch.setattr(pipeline, "lint_sync", lambda payload, sync: ["nope"])
    outcome = run(settings, store)

    assert outcome.sync is None
    assert outcome.sync_tiers == frozenset()


def test_a_low_confidence_song_has_no_sync_tiers(settings, store):
    """§13.3: no sidecar is built at all, so there is nothing for any tier to
    agree with."""
    weak = replace(settings, confidence_floor=0.99)
    outcome = run(weak, store)

    assert outcome.sync is None
    assert outcome.sync_tiers == frozenset()


def test_only_the_agreeing_tiers_are_filed_with_a_sidecar(settings, store, monkeypatch):
    """End to end through `run_job`, because the per-tier decision is only worth
    anything if the store write honours it."""
    from app.analysis import pipeline
    from app import jobs
    from app.jobs import run_job

    real_lint_sync = pipeline.lint_sync
    monkeypatch.setattr(
        pipeline, "lint_sync",
        lambda payload, sync: (["disagrees"] if payload.id.endswith(f":{EASY}")
                               else real_lint_sync(payload, sync)))
    # `run_job` builds its engines from the registry, which this module does not
    # populate — the fakes are passed straight to `analyze` everywhere else here.
    monkeypatch.setattr(jobs.engines, "build_chord_engine", lambda _s: FakeChordEngine())
    monkeypatch.setattr(jobs.engines, "build_beat_tracker", lambda _s: FakeBeatTracker())
    monkeypatch.setattr(jobs.engines, "build_onset_detector", lambda _s: FakeOnsetDetector())
    monkeypatch.setattr(jobs.engines, "build_structure_probe", lambda _s: None)

    store.create_job(job_id="j1", uid="u1", video_id="dQw4w9WgXcQ", difficulty=NORMAL)
    run_job(job_id="j1", video_id="dQw4w9WgXcQ", difficulty=NORMAL, uid="u1",
            settings=settings, store=store, source=FakeSource())

    assert store.get_map("dQw4w9WgXcQ", EASY).sync is None
    assert store.get_map("dQw4w9WgXcQ", NORMAL).sync is not None
    assert store.get_map("dQw4w9WgXcQ", HARD).sync is not None
