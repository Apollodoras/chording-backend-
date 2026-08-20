"""The payload lint — the app's importer, ported (§12.4).

Every rule here mirrors something `ComposerService.importReport` checks or
something the app's compiler does silently. The silent ones are the reason the
lint exists at all: an unparseable chord, an unembedded pattern id or an empty
section produces **no error on device** — the section is dropped, the song plays
short, and the player sees a song that is quietly wrong.

`repair` handles the bookkeeping a server can do without musical judgment, so the
lint never fails on something that was only ever a missing UUID.
"""

from __future__ import annotations

from app.analysis import meter
from app.lint import lint, repair
from app.payload import (
    PATTERN_TEMPO_MAX,
    PATTERN_TEMPO_MIN,
    PLAUSIBLE_TEMPO_MAX,
    PLAUSIBLE_TEMPO_MIN,
    TEMPO_MAX,
    TEMPO_MIN,
    Arrangement,
    Bar,
    BarRhythm,
    ChordSpan,
    CompositionPayload,
    PatternPayload,
    SongSection,
    Stroke,
)


def pattern(pattern_id="yt:pat-test") -> PatternPayload:
    return PatternPayload(
        id=pattern_id, name="Test strum", timeSignature="4/4", tempo=120,
        strokes=[Stroke(id="s1", beat=0.0), Stroke(id="s2", beat=2.0)],
    )


def song(**overrides) -> CompositionPayload:
    base = dict(
        version=2, id="yt:dQw4w9WgXcQ", title="Known Song",
        tonic="G", mode="major", tempo=120, timeSignature="4/4",
        arrangement=Arrangement(sections=[SongSection(
            id="sec1", name="", kindRaw="verse",
            chordNames=["G", "D", "Em", "C"], patternID="yt:pat-test",
            beatsPerChord=4, repeats=2,
        )]),
        patterns=[pattern()],
    )
    base.update(overrides)
    payload = CompositionPayload(**base)
    repair(payload)
    return payload


def test_a_good_song_lints_clean():
    assert lint(song()) == []


# --- the silent failures ----------------------------------------------------

def test_an_unreadable_chord_is_caught():
    """On device this is an import *warning* and a chord that doesn't sound as
    intended — §12.4's job is to make that list empty."""
    payload = song()
    payload.arrangement.sections[0].chordNames = ["G", "Gmaj9", "Em"]
    assert any("Gmaj9" in p for p in lint(payload))


def test_an_unembedded_pattern_is_caught():
    """§12.3: the section is **silently dropped** and the song plays short with
    no error. There is no bundled pattern catalog on the client any more."""
    payload = song(patterns=[])
    problems = lint(payload)
    assert any("not defined in" in p for p in problems)


def test_a_per_bar_pattern_override_must_also_be_embedded():
    payload = song()
    payload.arrangement.sections[0].bars = [Bar(
        id="b1", chordSpans=[ChordSpan(chordName="G", startBeat=0, lengthBeats=4)],
        rhythm=BarRhythm(kind="pattern", patternID="yt:pat-missing"),
    )]
    assert any("yt:pat-missing" in p for p in lint(payload))


def test_a_section_with_no_pattern_is_caught():
    payload = song()
    payload.arrangement.sections[0].patternID = ""
    assert any("no strumming pattern" in p for p in lint(payload))


def test_an_empty_section_is_caught():
    """§18: "Never emit a section with empty chords — it's silently dropped"."""
    payload = song()
    payload.arrangement.sections[0].chordNames = []
    assert any("empty section" in p for p in lint(payload))


def test_a_stroke_outside_its_bar_is_caught():
    """The app silently drops out-of-range strokes."""
    payload = song(patterns=[PatternPayload(
        id="yt:pat-test", name="Test strum", timeSignature="4/4", tempo=120,
        strokes=[Stroke(id="s1", beat=0.0), Stroke(id="s2", beat=4.5)],
    )])
    assert any("outside one bar" in p for p in lint(payload))


def test_a_chord_span_running_past_its_bar_is_caught():
    payload = song()
    payload.arrangement.sections[0].bars = [Bar(
        id="b1",
        chordSpans=[ChordSpan(chordName="G", startBeat=2, lengthBeats=4)],
        rhythm=BarRhythm(kind="inherit"),
    )]
    assert any("runs past the bar" in p for p in lint(payload))


# --- vocabularies -----------------------------------------------------------

def test_mode_is_major_or_minor_only():
    """§12.2 — the song container knows no church modes (the jam room does)."""
    assert any("major" in p for p in lint(song(mode="dorian")))


def test_an_unknown_section_kind_is_caught():
    payload = song()
    payload.arrangement.sections[0].kindRaw = "breakdown"
    assert any("kindRaw" in p for p in lint(payload))


def test_an_absurd_tempo_is_caught():
    assert any("outside the sane" in p for p in lint(song(tempo=400)))


def test_the_three_tempo_ranges_nest():
    """Three ranges in three files, answering three questions, with nothing
    tying them together — and the combination refused songs: the analysis called
    230 bpm suspect and left it alone, and this lint then killed the song for it.

    The ordering is the contract. A tempo the analysis is happy with must be one
    the container accepts, and a song tempo the container accepts must be legal
    on the pattern that carries it — patterns take the song's own tempo, so an
    inversion here is a song that lints clean and a pattern that does not.
    """
    assert PATTERN_TEMPO_MIN <= TEMPO_MIN <= PLAUSIBLE_TEMPO_MIN
    assert PLAUSIBLE_TEMPO_MAX <= TEMPO_MAX <= PATTERN_TEMPO_MAX
    assert (meter.TEMPO_MIN, meter.TEMPO_MAX) == (PLAUSIBLE_TEMPO_MIN, PLAUSIBLE_TEMPO_MAX)


def test_a_song_at_the_edge_of_the_container_range_carries_a_legal_pattern():
    """The nesting, exercised rather than asserted: the pattern is emitted at the
    song's tempo, so the widest range has to cover the one lint enforces."""
    for tempo in (TEMPO_MIN, TEMPO_MAX):
        payload = song(tempo=tempo, patterns=[pattern()])
        payload.patterns[0].tempo = tempo
        assert lint(payload) == []


def test_duplicate_pattern_ids_are_caught():
    payload = song(patterns=[pattern(), pattern()])
    assert any("unique" in p for p in lint(payload))


# --- repair -----------------------------------------------------------------

def test_repair_fills_the_ids_swift_requires():
    """Swift's synthesized `Codable` requires every non-optional field present —
    a `SongSection` missing `id` fails `CompositionPayload.from(jsonData:)`
    outright, and the player sees "the answer couldn't be read"."""
    payload = CompositionPayload(
        version=2, id="", title="X", tonic="G", mode="major", tempo=120,
        timeSignature="4/4",
        arrangement=Arrangement(sections=[SongSection(
            id="", chordNames=["G"], patternID="yt:pat-test", beatsPerChord=4)]),
        patterns=[PatternPayload(id="", name="P", strokes=[Stroke(id="", beat=0.0)])],
    )
    repair(payload)
    assert payload.id.startswith("yt:")
    assert payload.arrangement.sections[0].id
    assert payload.patterns[0].id.startswith("yt:pat-")
    assert payload.patterns[0].strokes[0].id


def test_repair_mirrors_the_flat_summary_from_section_one():
    """The Library list and the tiny-share path read these."""
    payload = song()
    payload.chordNames = ["wrong"]
    repair(payload)
    assert payload.chordNames == ["G", "D", "Em", "C"]
    assert payload.beatsPerChord == 4
    assert payload.repeats == 2


def test_repair_can_force_the_id_so_re_analysis_replaces_the_row():
    payload = song()
    repair(payload, force_id="yt:dQw4w9WgXcQ:easy")
    assert payload.id == "yt:dQw4w9WgXcQ:easy"


# --- the wire ---------------------------------------------------------------

def test_the_wire_dict_keeps_every_field_swift_needs():
    payload = song()
    wire = payload.wire_dict()
    for key in ("version", "id", "title", "tonic", "mode", "tempo", "timeSignature",
                "chordNames", "patternID", "beatsPerChord", "repeats",
                "arrangement", "patterns", "progressions"):
        assert key in wire, key
    stroke = wire["patterns"][0]["strokes"][0]
    for key in ("id", "beat", "direction", "accent", "msOffset"):
        assert key in stroke, key


def test_a_bar_rhythm_serializes_as_swifts_enum_union():
    """`BarRhythm` is a Swift enum with associated values, so its JSON is the
    synthesized union encoding — get it wrong and the payload doesn't decode."""
    payload = song()
    payload.arrangement.sections[0].bars = [
        Bar(id="b1", chordSpans=[ChordSpan(chordName="G", startBeat=0, lengthBeats=4)],
            rhythm=BarRhythm(kind="inherit")),
        Bar(id="b2", chordSpans=[ChordSpan(chordName="C", startBeat=0, lengthBeats=4)],
            rhythm=BarRhythm(kind="pattern", patternID="yt:pat-test")),
        Bar(id="b3", chordSpans=[ChordSpan(chordName="D", startBeat=0, lengthBeats=4)],
            rhythm=BarRhythm(kind="custom", strokes=[Stroke(id="s", beat=0.0)])),
    ]
    bars = payload.wire_dict()["arrangement"]["sections"][0]["bars"]
    assert bars[0]["rhythm"] == {"inherit": {}}
    assert bars[1]["rhythm"] == {"pattern": {"_0": "yt:pat-test"}}
    assert bars[2]["rhythm"]["custom"]["_0"][0]["beat"] == 0.0


# --- §14.1 bands -------------------------------------------------------------

def _with_band(band):
    return song(patterns=[PatternPayload(
        id="yt:pat-test", name="Test strum", timeSignature="4/4", tempo=120,
        strokes=[Stroke(id="s1", beat=0.0, band=band), Stroke(id="s2", beat=2.0)],
    )])


def test_a_stroke_may_name_the_band_it_was_struck_in():
    for band in ("low", "mid"):
        assert not any("band" in p for p in lint(_with_band(band))), band


def test_a_stroke_with_an_invented_band_is_caught():
    assert any("has band" in p for p in lint(_with_band("bass")))


def test_full_is_not_a_transmissible_band():
    """`FULL` is a real answer and an absent field is how it travels (§14.1).

    Saying it out loud would be a claim the client has no rule for — and, worse,
    would change the bytes of every strummed song for no information at all.
    """
    assert any("has band" in p for p in lint(_with_band("full")))
