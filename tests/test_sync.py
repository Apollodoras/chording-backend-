"""§13 — the sidecar, and the invariant that keeps the cursor on the song.

This file is the one that earns its keep the day something drifts. The §13.2
invariant — `songBeat` here must be the same axis the compiled chart produces,
so bar *n* starts at `songBeat = n × barBeats` — has no runtime symptom on the
backend at all: the payload is valid, the sidecar is well-formed, and the failure
appears only on a phone, as a cursor walking away from the recording partway
through a song. The handoff says "assert it in a test". This is that test.

`song_beat_at` is the client's interpolation reimplemented here, so the
assertions are about what the *client* will compute, not about what we stored.
"""

from __future__ import annotations

import pytest

from app.analysis.compile import compile_song
from app.analysis.keyfinder import DetectedKey
from app.analysis.postprocess import process
from app.analysis.strumming import fallback
from app.analysis.form import segment
from app.analysis.structure import bars_from_spans
from app.chords import NORMAL
from app.lint import lint, lint_sync, repair, section_beats, total_beats
from app.payload import CompositionPayload, bar_beats
from app.sync import BeatAnchor, Confidence, EngineVersions, VideoSync, anchors_for, song_beat_at
from tests.conftest import (
    BAR_BEATS,
    MS_PER_BEAT,
    known_axis,
    known_chords,
    known_downbeats,
)


def make_sync(**overrides) -> VideoSync:
    base = dict(
        videoId="dQw4w9WgXcQ",
        durationMs=32_000,
        offsetMs=0,
        tempo=Confidence(bpm=120.0, confidence=0.95),
        timeSignature="4/4",
        lowConfidence=False,
        beatAnchors=anchors_for(known_downbeats(), bar_beats=4),
        engine=EngineVersions(chords="fake@1", beats="fake@1"),
        analyzedAt="2026-08-03T10:00:00Z",
    )
    base.update(overrides)
    return VideoSync(**base)


def known_song() -> CompositionPayload:
    spans = process(known_chords(), known_axis(), difficulty=NORMAL)
    sections = segment(bars_from_spans(spans, BAR_BEATS))
    patterns = {s.group: fallback(bar_beats=4, tempo=120, name="Verse strum")
                for s in sections}
    payload = compile_song(
        video_id="dQw4w9WgXcQ", title="Known Song", sections=sections,
        patterns=patterns, key=DetectedKey("G", "major", 0.9),
        tempo=120, time_signature="4/4", difficulty=NORMAL,
    )
    repair(payload)
    assert lint(payload) == []
    return payload


# --- anchors ----------------------------------------------------------------

def test_anchors_land_on_bar_boundaries_of_the_charts_own_axis():
    """§13.2, stated directly: bar *n* of the payload starts at
    `songBeat = n × barBeats`."""
    anchors = anchors_for(known_downbeats(), bar_beats=4)
    assert [a.songBeat for a in anchors[:4]] == [0, 4, 8, 12]
    assert [a.tMs for a in anchors[:4]] == [0, 2000, 4000, 6000]


def test_an_intro_the_analysis_discarded_shifts_the_axis_not_the_clock():
    anchors = anchors_for([4000, 6000], bar_beats=4, first_bar_index=2)
    assert [(a.songBeat, a.tMs) for a in anchors] == [(8, 4000), (12, 6000)]


# --- the client's interpolation ---------------------------------------------

def test_interpolation_is_exact_on_the_anchors():
    anchors = anchors_for(known_downbeats(), bar_beats=4)
    for anchor in anchors:
        assert song_beat_at(anchors, anchor.tMs) == pytest.approx(anchor.songBeat)


def test_interpolation_is_linear_between_anchors():
    anchors = anchors_for(known_downbeats(), bar_beats=4)
    assert song_beat_at(anchors, MS_PER_BEAT) == pytest.approx(1.0)
    assert song_beat_at(anchors, 3000) == pytest.approx(6.0)


def test_anchors_absorb_tempo_drift_a_single_bpm_cannot():
    """The reason the sidecar is an anchor *list* and not a bpm: a recording that
    slows down is still exactly addressable."""
    anchors = [
        BeatAnchor(songBeat=0, tMs=0),
        BeatAnchor(songBeat=4, tMs=2000),     # 120 bpm
        BeatAnchor(songBeat=8, tMs=4400),     # 100 bpm
    ]
    assert song_beat_at(anchors, 1000) == pytest.approx(2.0)
    assert song_beat_at(anchors, 3200) == pytest.approx(6.0)


def test_the_cursor_keeps_moving_past_the_last_anchor():
    """A video that keeps playing past the last anchor should keep producing
    beats rather than freezing the cursor."""
    anchors = [BeatAnchor(songBeat=0, tMs=0), BeatAnchor(songBeat=4, tMs=2000)]
    assert song_beat_at(anchors, 4000) == pytest.approx(8.0)


def test_a_single_anchor_holds_rather_than_guessing_a_tempo():
    anchors = [BeatAnchor(songBeat=0, tMs=500)]
    assert song_beat_at(anchors, 90_000) == 0.0


# --- the invariant, end to end ----------------------------------------------

def test_every_anchor_addresses_a_bar_the_compiled_chart_actually_has():
    """The whole point. If the section layout and the anchor list disagree, the
    cursor walks off the song."""
    payload = known_song()
    sync = make_sync()
    assert lint_sync(payload, sync) == []

    beats = bar_beats(payload.timeSignature)
    song_length = total_beats(payload)
    for anchor in sync.beatAnchors:
        assert anchor.songBeat % beats == 0
        assert anchor.songBeat <= song_length


def test_section_lengths_are_whole_bars():
    payload = known_song()
    beats = bar_beats(payload.timeSignature)
    for section in payload.arrangement.sections:
        assert section_beats(section, beats) % beats == 0


# --- lint_sync refuses the things that would drift --------------------------

def test_a_tempo_override_is_refused_on_a_synced_song():
    """The client derives its bar grid from a SINGLE song-level tempo
    (`JamSongSheet.from`), so an override detaches the axis the anchors address.
    Tempo drift belongs in the anchors, which is what they are for."""
    payload = known_song()
    payload.arrangement.sections[0].tempoOverride = 90
    problems = lint_sync(payload, make_sync())
    assert any("tempoOverride" in p for p in problems)


def test_a_time_signature_override_is_refused_for_the_same_reason():
    payload = known_song()
    payload.arrangement.sections[0].timeSignatureOverride = "3/4"
    assert any("timeSignatureOverride" in p for p in lint_sync(payload, make_sync()))


def test_a_section_that_is_not_whole_bars_is_refused():
    """Sections concatenate, so a section that isn't a whole number of bars
    slides every later bar off the recording's downbeats — silently, and further
    with each section."""
    payload = known_song()
    section = payload.arrangement.sections[0]
    section.chordNames = ["G", "D", "Em"]
    section.beatsPerChord = 3
    section.repeats = 1                      # 3 × 3 × 1 = 9 beats, not a whole 4/4 bar
    assert any("whole number" in p for p in lint_sync(payload, make_sync()))


def test_anchors_must_increase():
    payload = known_song()
    sync = make_sync(beatAnchors=[
        BeatAnchor(songBeat=0, tMs=0),
        BeatAnchor(songBeat=4, tMs=2000),
        BeatAnchor(songBeat=8, tMs=1500),
    ])
    assert any("strictly increase in tMs" in p for p in lint_sync(payload, sync))


def test_an_anchor_off_a_bar_boundary_is_refused():
    payload = known_song()
    sync = make_sync(beatAnchors=[
        BeatAnchor(songBeat=0, tMs=0),
        BeatAnchor(songBeat=3, tMs=2000),
    ])
    assert any("bar boundary" in p for p in lint_sync(payload, sync))


def test_a_meter_disagreement_between_song_and_sidecar_is_refused():
    payload = known_song()
    assert any("timeSignature" in p for p in lint_sync(payload, make_sync(timeSignature="3/4")))


def test_anchors_past_the_end_of_the_video_are_refused():
    payload = known_song()
    assert any("past the end" in p for p in lint_sync(payload, make_sync(durationMs=1000)))


def test_empty_anchors_are_refused():
    payload = known_song()
    assert any("beatAnchors is empty" in p for p in lint_sync(payload, make_sync(beatAnchors=[])))


def test_anchors_that_stop_short_of_the_song_are_refused():
    """Short coverage isn't fatal — the client extrapolates — but it means the
    tail runs on an assumed tempo, which is exactly what anchors exist to
    avoid."""
    payload = known_song()
    sync = make_sync(beatAnchors=anchors_for(known_downbeats()[:3], bar_beats=4))
    assert any("tail would run on extrapolated tempo" in p for p in lint_sync(payload, sync))
