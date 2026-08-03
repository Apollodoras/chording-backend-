"""The compiler — flat vs bars, and the reasons the choice is not cosmetic.

The rule under test is one the app's compiler imposes rather than one anybody
chose: `Compiler.compile` tiles the strumming pattern **per chord slot**, so a
`beatsPerChord` smaller than the pattern's bar truncates the groove and restarts
it mid-bar, silently. That makes flat mode unusable for a mid-bar chord change,
and bars mode (`notesForBar`, which lays the pattern over the bar once and lets
each stroke sound whichever chord is active) the only correct encoding for it.
"""

from __future__ import annotations

from app.analysis.compile import compile_song
from app.analysis.keyfinder import DetectedKey
from app.analysis.strumming import fallback
from app.analysis.structure import BarChord, Section
from app.chords import EASY, MAJOR, MINOR, NORMAL
from app.lint import lint, repair
from app.payload import CompositionPayload


def bar(*chords) -> list[BarChord]:
    return list(chords)


def whole(root, quality=MAJOR) -> BarChord:
    return BarChord(root_pc=root, quality=quality, start_beat=0.0, length_beats=4.0)


def half(root, start, quality=MAJOR) -> BarChord:
    return BarChord(root_pc=root, quality=quality, start_beat=float(start), length_beats=2.0)


def build(sections: list[Section], **kwargs) -> CompositionPayload:
    patterns = {i: fallback(bar_beats=4, tempo=120, name=f"Section {i} strum")
                for i in range(len(sections))}
    payload = compile_song(
        video_id="dQw4w9WgXcQ", title="Known Song", sections=sections,
        patterns=patterns, key=DetectedKey("G", "major", 0.9),
        tempo=120, time_signature="4/4", **kwargs,
    )
    repair(payload)
    return payload


# --- flat mode --------------------------------------------------------------

def test_one_chord_per_bar_compiles_flat():
    section = Section(kind="verse", bars=[bar(whole(7)), bar(whole(2)),
                                          bar(whole(4, MINOR)), bar(whole(0))],
                      repeats=4)
    payload = build([section])
    compiled = payload.arrangement.sections[0]
    assert compiled.bars is None
    assert compiled.chordNames == ["G", "D", "Em", "C"]
    assert compiled.beatsPerChord == 4
    assert compiled.repeats == 4


def test_a_chord_held_across_bars_is_repeated_rather_than_lengthened():
    """`beatsPerChord` stays equal to the bar, and the chord appears once per bar
    — the same encoding a shared song uses, and the one that keeps the pattern
    tiling correctly."""
    section = Section(kind="verse", bars=[bar(whole(7)), bar(whole(7)), bar(whole(0)), bar(whole(0))])
    compiled = build([section]).arrangement.sections[0]
    assert compiled.bars is None
    assert compiled.chordNames == ["G", "G", "C", "C"]


# --- bars mode --------------------------------------------------------------

def test_a_mid_bar_change_compiles_to_bars():
    """Flat mode cannot say this: a `beatsPerChord` of 2 would truncate the
    4-beat pattern to its first two beats and restart it mid-bar."""
    section = Section(kind="verse", bars=[bar(half(7, 0), half(2, 2)),
                                          bar(whole(0)), bar(whole(0)), bar(whole(0))])
    compiled = build([section]).arrangement.sections[0]
    assert compiled.bars is not None
    assert [(s.chordName, s.startBeat, s.lengthBeats) for s in compiled.bars[0].chordSpans] == [
        ("G", 0.0, 2.0), ("D", 2.0, 2.0),
    ]


def test_bars_mode_expands_repeats_because_the_app_ignores_them():
    """`compileBars` never sees `repeats`. Leaving it above 1 plays a quarter of
    the section with no error anywhere."""
    section = Section(kind="verse",
                      bars=[bar(half(7, 0), half(2, 2)), bar(whole(0)), bar(whole(0)), bar(whole(0))],
                      repeats=3)
    compiled = build([section]).arrangement.sections[0]
    assert compiled.repeats == 1
    assert len(compiled.bars) == 12


def test_bars_mode_still_carries_chord_names():
    """Dead for compilation, filled anyway — so a section is never one field away
    from the "empty chordNames ⇒ silently dropped" failure."""
    section = Section(kind="verse", bars=[bar(half(7, 0), half(2, 2)), bar(whole(0)),
                                          bar(whole(0)), bar(whole(0))])
    compiled = build([section]).arrangement.sections[0]
    assert compiled.chordNames


def test_every_bar_span_fits_inside_its_bar():
    section = Section(kind="verse", bars=[bar(half(7, 0), half(2, 2)), bar(whole(0)),
                                          bar(whole(0)), bar(whole(0))])
    compiled = build([section]).arrangement.sections[0]
    for b in compiled.bars:
        for chord_span in b.chordSpans:
            assert chord_span.startBeat + chord_span.lengthBeats <= 4 + 1e-9


# --- the container's rules --------------------------------------------------

def test_a_compiled_song_lints_clean():
    """§12.4: never return a payload that would warn."""
    section = Section(kind="verse", bars=[bar(whole(7)), bar(whole(2)),
                                          bar(whole(4, MINOR)), bar(whole(0))], repeats=2)
    assert lint(build([section])) == []


def test_every_referenced_pattern_is_embedded():
    """§12.3: an unembedded reference means the section is **silently dropped**
    and the song plays short with no error. There is no bundled pattern catalog
    on the client any more."""
    sections = [
        Section(kind="verse", bars=[bar(whole(7)), bar(whole(2)), bar(whole(4, MINOR)), bar(whole(0))]),
        Section(kind="chorus", bars=[bar(whole(0)), bar(whole(0)), bar(whole(7)), bar(whole(7))]),
    ]
    payload = build(sections)
    embedded = {p.id for p in payload.patterns}
    for section in payload.arrangement.sections:
        assert section.patternID in embedded


def test_a_synced_song_never_carries_a_tempo_or_meter_override():
    """The client's bar grid is computed on one song-level tempo, so an override
    would detach the axis the sidecar's anchors address (§13.2)."""
    sections = [
        Section(kind="verse", bars=[bar(whole(7)), bar(whole(2)), bar(whole(4, MINOR)), bar(whole(0))]),
        Section(kind="chorus", bars=[bar(whole(0)), bar(whole(0)), bar(whole(7)), bar(whole(7))]),
    ]
    for section in build(sections).arrangement.sections:
        assert section.tempoOverride is None
        assert section.timeSignatureOverride is None


def test_the_flat_summary_mirrors_section_one():
    """§12.4.4 — the Library row's metadata reads these."""
    sections = [
        Section(kind="verse", bars=[bar(whole(7)), bar(whole(2)), bar(whole(4, MINOR)), bar(whole(0))],
                repeats=2),
        Section(kind="chorus", bars=[bar(whole(0)), bar(whole(0)), bar(whole(7)), bar(whole(7))]),
    ]
    payload = build(sections)
    first = payload.arrangement.sections[0]
    assert payload.chordNames == first.chordNames
    assert payload.patternID == first.patternID
    assert payload.beatsPerChord == first.beatsPerChord
    assert payload.repeats == first.repeats


# --- ids --------------------------------------------------------------------

def test_the_song_id_is_deterministic_so_re_analysis_replaces_the_library_row():
    """§12.5: `id` is the idempotency key — `import` upserts on it."""
    section = Section(kind="verse", bars=[bar(whole(7)), bar(whole(2)),
                                          bar(whole(4, MINOR)), bar(whole(0))])
    first = build([section])
    second = build([section])
    assert first.id == second.id == "yt:dQw4w9WgXcQ"


def test_each_difficulty_gets_its_own_row():
    """So all three tiers can coexist without `import` upserting one over
    another."""
    section = Section(kind="verse", bars=[bar(whole(7)), bar(whole(2)),
                                          bar(whole(4, MINOR)), bar(whole(0))])
    assert build([section], difficulty=EASY).id == "yt:dQw4w9WgXcQ:easy"
    assert build([section], difficulty=NORMAL).id == "yt:dQw4w9WgXcQ:normal"


def test_song_ids_are_namespaced_away_from_mo():
    section = Section(kind="verse", bars=[bar(whole(7)), bar(whole(2)),
                                          bar(whole(4, MINOR)), bar(whole(0))])
    assert build([section]).id.startswith("yt:")


# --- spelling ---------------------------------------------------------------

def test_chords_are_spelled_for_the_songs_key():
    """The player reads these off the campfire bands: "Bb" in an F-major song is
    right where "A#" is wrong."""
    section = Section(kind="verse", bars=[bar(whole(5)), bar(whole(10)),
                                          bar(whole(0)), bar(whole(5))])
    patterns = {0: fallback(bar_beats=4, tempo=120, name="Verse strum")}
    payload = compile_song(
        video_id="x" * 11, title="Flat Song", sections=[section], patterns=patterns,
        key=DetectedKey("F", "major", 0.9), tempo=120, time_signature="4/4",
    )
    assert payload.arrangement.sections[0].chordNames == ["F", "Bb", "C", "F"]
