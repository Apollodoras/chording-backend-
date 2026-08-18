"""§20.6 — one analysis, three renders.

The claim under test is that a song has **one** structure and the difficulty
tier only changes which chord names are printed inside it. Before §20 the whole
of §5.4, §15 and §14 ran once per tier, so the three songs were related only by
having come from the same recording — and `easy`'s simplification can merge two
bars into identical ones, which changes what the segmenter collapses. The
pipeline had to re-check `lint_sync` against every tier to catch the fallout.

That check is still there and still passes. These tests assert the stronger
thing it could never establish: that the tiers agree **by construction**.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from app.analysis import model as song_model
from app.analysis.types import BeatGrid, EnergyCurve, RawChordSpan
from app.chords import DIFFICULTIES, EASY, HARD
from app.chords import render as render_name
from app.errors import AnalysisError

BEAT_MS = 500
BAR_MS = BEAT_MS * 4


def _grid(bars: int = 16) -> BeatGrid:
    beats = [i * BEAT_MS for i in range(bars * 4 + 1)]
    return BeatGrid(beats_ms=beats, downbeats_ms=beats[::4], bpm=120.0,
                    confidence=0.9, time_signature="4/4")


def _chords(bars: int = 16, *, names=("C", "G", "Am", "F")) -> list[RawChordSpan]:
    return [RawChordSpan(start_ms=i * BAR_MS, end_ms=(i + 1) * BAR_MS,
                         label=names[i % len(names)], confidence=0.9)
            for i in range(bars)]


def _model(**kwargs):
    return song_model.build(grid=_grid(), raw=_chords(), onsets=[], **kwargs)


# --- the model ---------------------------------------------------------------

def test_a_model_is_built_from_a_steady_grid_and_readable_chords():
    model = _model()
    assert model is not None
    assert model.sections and model.groups
    assert model.meter.bar_beats == 4
    assert model.axis.bar_count == 16


def test_no_usable_rhythm_returns_none_rather_than_raising():
    """The caller turns this into an honest refusal (§13.3). It is a different
    failure from having no chords, and the player is told a different thing."""
    thin = BeatGrid(beats_ms=[0, 500], downbeats_ms=[0], bpm=120.0,
                    confidence=0.9, time_signature="4/4")
    assert song_model.build(grid=thin, raw=_chords(), onsets=[]) is None


def test_no_readable_chords_raises_rather_than_blaming_the_rhythm():
    """"That track's beat wasn't clear enough" is wrong and confusing on a
    perfectly steady recording the chord engine simply heard as silence."""
    silent = [RawChordSpan(start_ms=0, end_ms=32000, label="N")]
    with pytest.raises(AnalysisError, match="No chords"):
        song_model.build(grid=_grid(), raw=silent, onsets=[])


# --- the tiers are renders, not analyses -------------------------------------

def test_every_tier_shares_the_models_section_boundaries():
    model = _model()
    expected = [(s.start_bar, s.total_bars) for s in model.sections]
    for difficulty in DIFFICULTIES:
        rendered = song_model.render(model, _chords(), difficulty)
        assert [(s.start_bar, s.total_bars) for s in rendered] == expected, difficulty


def test_every_tier_shares_the_models_repeat_groups():
    model = _model()
    expected = [s.group for s in model.sections]
    for difficulty in DIFFICULTIES:
        assert [s.group for s in song_model.render(model, _chords(), difficulty)] == expected


def test_the_tiers_tile_the_song_identically():
    """One sidecar serves whichever tier the player asked for, so a tier that
    tiled differently would be addressed by anchors meant for another."""
    model = _model()
    for difficulty in DIFFICULTIES:
        cursor = 0
        for section in song_model.render(model, _chords(), difficulty):
            assert section.start_bar == cursor
            cursor += section.total_bars
        assert cursor == model.total_bars


def test_simplification_changes_the_names_not_the_shape():
    """`easy` really does drop passing chords shorter than a bar — duration work
    that cannot be done by renaming qualities in place — but the boundaries come
    from the model and do not move."""
    raw = _chords(names=("Cmaj7", "G7", "Am7", "Fmaj7"))
    model = song_model.build(grid=_grid(), raw=raw, onsets=[])
    hard = song_model.render(model, raw, HARD)
    easy = song_model.render(model, raw, EASY)
    assert [(s.start_bar, s.total_bars) for s in hard] == [(s.start_bar, s.total_bars) for s in easy]

    hard_qualities = {c.quality for s in hard for bar in s.bars for c in bar}
    easy_qualities = {c.quality for s in easy for bar in s.bars for c in bar}
    assert hard_qualities != easy_qualities, "the tiers really do differ in vocabulary"
    assert easy_qualities <= {"major", "minor"}


# --- patterns ----------------------------------------------------------------

def test_one_pattern_per_repeat_group():
    model = _model()
    assert set(model.patterns) == {g.label for g in model.groups}


def test_sections_of_one_group_share_a_groove():
    """They are the same music, which is also why the pooled extraction had more
    evidence to work from."""
    model = _model()
    for section in model.sections:
        assert section.group in model.patterns


def test_a_pattern_names_the_meter_the_song_is_actually_in():
    """`bar_beats` is *quarter-note* beats, so 6/8 arrives as 3.0 — and building
    the pattern's signature back out of it as f"{bar_beats}/4" shipped every 6/8
    song a chart in 6/8 whose patterns claimed 3/4. Same bar length, so nothing
    played wrong and nothing failed the lint; the label was simply a lie, and
    `strumming`'s docstring promises the opposite."""
    beats = [i * BEAT_MS for i in range(16 * 3 + 1)]
    grid = BeatGrid(beats_ms=beats, downbeats_ms=beats[::3], bpm=120.0,
                    confidence=0.9, time_signature="6/8")
    raw = [RawChordSpan(start_ms=i * BEAT_MS * 3, end_ms=(i + 1) * BEAT_MS * 3,
                        label=("C", "G", "Am", "F")[i % 4], confidence=0.9)
           for i in range(16)]
    model = song_model.build(grid=grid, raw=raw, onsets=[])
    assert model is not None
    assert model.meter.time_signature == "6/8"
    assert model.axis.bar_beats == 3          # the bar is still three quarter-beats
    assert {p.pattern.timeSignature for p in model.patterns.values()} == {"6/8"}


# --- energy ------------------------------------------------------------------

def test_a_loudness_curve_becomes_one_number_per_bar():
    """Averaged between the axis's own downbeats, so bar k's energy is measured
    over exactly the span bar k occupies — the same correspondence the anchors
    publish."""
    model = _model()
    curve = EnergyCurve(hop_ms=50, values=[0.0] * 320 + [1.0] * 320)
    per_bar = song_model.per_bar_energy(curve, model.axis)
    assert per_bar is not None and len(per_bar) == model.axis.bar_count
    assert per_bar[0] < per_bar[-1]


def test_no_probe_means_no_energy_and_that_is_a_supported_configuration():
    model = _model()
    assert song_model.per_bar_energy(None, model.axis) is None
    assert all(s.kind == "custom" for s in model.sections), "§15's honest fallback"


def test_an_empty_curve_is_treated_as_no_curve():
    model = _model()
    assert song_model.per_bar_energy(EnergyCurve(hop_ms=50, values=[]), model.axis) is None
    assert song_model.per_bar_energy(EnergyCurve(hop_ms=0, values=[1.0]), model.axis) is None


def test_the_hop_is_a_real_duration_and_not_a_whole_millisecond():
    """1024 samples at 22.05 kHz is 46.44 ms, and rounding it to 46 is a ~1%
    *rate* error, not a rounding error. Three minutes in, the window read for a
    bar sat two seconds — most of a bar — from the bar it claimed to measure, so
    the verse/chorus comparison the curve exists for was taken over the wrong
    music. Asserted at four minutes, where the drift is most of a bar."""
    hop = 1000.0 * 1024 / 22050
    assert hop != round(hop)

    four_minutes_ms = 240_000
    # One frame per hop for four minutes, silent except for a one-second burst
    # starting at exactly four minutes in.
    frames = int(four_minutes_ms / hop) + 100
    values = [0.0] * frames
    burst_first = int(four_minutes_ms / hop)
    for i in range(burst_first, burst_first + int(1000 / hop)):
        values[i] = 1.0

    honest = EnergyCurve(hop_ms=hop, values=values)
    rounded = EnergyCurve(hop_ms=float(round(hop)), values=values)
    assert honest.mean_between(four_minutes_ms, four_minutes_ms + 1000) > 0.9
    assert rounded.mean_between(four_minutes_ms, four_minutes_ms + 1000) < 0.1, \
        "the rounded hop reads a window two seconds away and finds silence"


# --- a render is a read of the model, not a write to it ----------------------

def _sevenths(last: str = "Cmaj7") -> list[RawChordSpan]:
    """Gmaj7 D7 Em7 Cmaj7 ×4, with the final bar heard as `last`.

    Sevenths because that is what makes a tier render *visible* in the model: at
    `easy` every one of them flattens to a triad, so a render that wrote back to
    the model would leave the reference structure carrying easy's chords.
    """
    labels = ["G:maj7", "D:7", "E:min7", "C:maj7"] * 4
    labels[-1] = last
    return [RawChordSpan(start_ms=i * BAR_MS, end_ms=(i + 1) * BAR_MS, label=label,
                         confidence=0.2 if i == 15 and last != "Cmaj7" else 0.9)
            for i, label in enumerate(labels)]


def _voted_model():
    """A model whose vote actually fired — the last bar of verse 4 misheard as
    Am7, doubtfully, which is the case `consensus` exists for."""
    raw = _sevenths(last="A:min7")
    model = song_model.build(grid=_grid(), raw=raw, onsets=[])
    assert model.consensus.rewritten_bars == 1, "the fixture has to reach the vote"
    return model, raw


def _provenance(groups):
    return [(g.label, g.rewritten_bars, g.contested_bars,
             [[(c.root_pc, c.quality) for c in bar] for bar in g.canonical])
            for g in groups]


def test_rendering_a_tier_does_not_edit_the_model_it_renders():
    """`render` replays the vote per tier, and the vote writes its provenance
    onto the `RepeatGroup` objects — which are the model's own. So after
    compiling three tiers the model described whichever one compiled last: at
    `easy`, a canonical progression of plain triads for a song whose reference
    tier is all sevenths. The wire never saw it (the sidecar snapshots earlier),
    the benchmark and the logs did."""
    model, raw = _voted_model()
    before = (_provenance(model.groups), _provenance(model.vote_groups))
    for difficulty in DIFFICULTIES:
        song_model.render(model, raw, difficulty)
    assert (_provenance(model.groups), _provenance(model.vote_groups)) == before


def test_the_reference_tier_is_the_model_itself():
    """`hard` is the tier the model was built at, so its render has to come back
    with the model's own bars. It used to hold by luck: the vote was taken over
    the first pass's groups at build time and over the *second* pass's on every
    render, so the reference tier was re-deciding what `build` had decided."""
    model, raw = _voted_model()
    rendered = song_model.render(model, raw, HARD)
    assert _shape(rendered) == _shape(model.sections)


def _shape(sections):
    return [(s.group, s.kind, s.repeats,
             [[(c.root_pc, c.quality, c.start_beat, c.length_beats) for c in bar]
              for bar in s.bars])
            for s in sections]


# --- the key, and the small circle it used to sit in -------------------------

def test_the_key_is_read_off_the_chart_the_vote_left_behind(monkeypatch):
    """Key detection runs before the vote because the vote's diatonic tie-break
    needs it — so the key is upstream of edits made partly on its own authority.
    Reading it again off the corrected bars breaks that circle, and it is free:
    the chords are already in hand."""
    seen: list[set] = []
    real = song_model.detect_key

    def spy(spans):
        seen.append({(s.root_pc, s.quality) for s in spans})
        return real(spans)

    monkeypatch.setattr(song_model, "detect_key", spy)
    model, _ = _voted_model()

    assert len(seen) == 2, "once for the vote's tie-break, once on what it produced"
    assert (9, "minor7") in seen[0], "the engine's mistake was in the first reading"
    assert (9, "minor7") not in seen[1], "and not in the chart the key came from"
    assert model.key.tonic == "G"


def test_a_render_replays_the_vote_with_the_key_the_vote_used(monkeypatch):
    """The other half of "a render reproduces the reference vote".

    `build` votes with the key read off the engine's own chords, then re-reads
    the key from the bars the vote left behind — so `model.key` is deliberately
    *not* the key the vote was taken with. Replaying with it made the tier
    renders take a different vote from the one the model was built on, exactly
    the way voting over the second pass's groups used to: the diatonic tie-break
    consumes the tonic, so any bar the two readings disagree about could settle
    the other way. Both are guarded on the same `consensus.touched`, so the case
    never arose without the fix being needed.

    The re-read is forced here rather than found: on the fixtures available a
    corrected bar or two never moves the reading, which is why this was latent
    and not a visible wrong chart. The property is what is being pinned.
    """
    votes: list[tuple[int, str]] = []
    real_apply = song_model.consensus.apply

    def spy_apply(bars, groups, *, bar_beats, tonic_pc=0, mode="ionian", record=True,
                  weigh=True):
        votes.append((tonic_pc, mode))
        return real_apply(bars, groups, bar_beats=bar_beats, tonic_pc=tonic_pc,
                          mode=mode, record=record, weigh=weigh)

    real_detect = song_model.detect_key
    readings: list[int] = []

    def shifting_detect(spans):
        found = real_detect(spans)
        readings.append(found.tonic_pc)
        # The second call is C3's post-vote re-read. Move it somewhere else so
        # "which key did the replay use" has an observable answer at all.
        if len(readings) == 2:
            return replace(found, tonic_pc=(found.tonic_pc + 5) % 12)
        return found

    monkeypatch.setattr(song_model.consensus, "apply", spy_apply)
    monkeypatch.setattr(song_model, "detect_key", shifting_detect)

    model, raw = _voted_model()
    # Stated against `readings[0]` — the pre-vote reading `build` voted with —
    # rather than against `model.vote_key`, so that this fails on the behaviour
    # and not merely on the absence of the field that fixes it.
    assert len(readings) == 2, "the fixture has to reach the re-read"
    assert model.key.tonic_pc != readings[0], "and the re-read has to differ"

    song_model.render(model, raw, HARD)

    assert len(votes) == 2, "one vote in build, one in the render"
    assert votes[0] == votes[1], \
        "the render has to replay the vote build took, not take a new one"


def test_a_vote_that_changed_nothing_does_not_re_read_the_key(monkeypatch):
    """Re-running it there could differ only in the trailing partial bar
    `bars_from_spans` drops — a change with no reason behind it."""
    calls = []
    real = song_model.detect_key
    monkeypatch.setattr(song_model, "detect_key",
                        lambda spans: (calls.append(1), real(spans))[1])
    model = song_model.build(grid=_grid(), raw=_sevenths(), onsets=[])
    assert not model.consensus.touched
    assert len(calls) == 1


# --- §20.8's cleanup, and its place in the order -----------------------------

def _noisy_chords(passes: int = 8) -> list[RawChordSpan]:
    """Am–F–C–G, played `passes` times, with the tonic misheard as `Am7` in one
    pass and as `A` in another — the shape the layer was reported for, and one no
    vote can settle: the passes disagree two ways, so there is no majority.

    Eight passes rather than four because the rule wants the song to contradict
    the reading *overwhelmingly* (`vocabulary.MASS_DOMINANCE`), and four bars of a
    root is not a song's worth of evidence about it. That is the intended
    behaviour and the thing worth knowing about the layer: on a very short song it
    declines to speak.
    """
    labels = ["A:min", "F:maj", "C:maj", "G:maj"] * passes
    labels[4], labels[8] = "A:min7", "A:maj"
    return [RawChordSpan(start_ms=i * BAR_MS, end_ms=(i + 1) * BAR_MS, label=label,
                         confidence=0.5 if i in (4, 8) else 0.9)
            for i, label in enumerate(labels)]


def test_the_cleanup_runs_before_the_bars_are_cut():
    """Both readings of the tonic are corrected, and the correction happens while
    the timeline is still spans — which is what lets `form` cluster identical
    passes and the vote find nothing left to do."""
    model = song_model.build(grid=_grid(32), raw=_noisy_chords(), onsets=[])
    qualities = {c.quality for s in model.sections for bar in s.bars for c in bar
                 if c.root_pc == 9}
    assert model.vocabulary.snapped_spans == 2
    assert qualities == {"minor"}, "the song plays Am, and now so does the chart"


def test_a_tier_render_replays_the_cleanup():
    """`hard` is the tier the model was built at, so its render has to come back
    with the model's own bars — the same discipline as the vote replay, and it
    needs the same stored key (`seed_key`) to be a replay rather than a new
    decision."""
    raw = _noisy_chords()
    model = song_model.build(grid=_grid(32), raw=raw, onsets=[])
    assert _shape(song_model.render(model, raw, HARD)) == _shape(model.sections)


def test_every_tier_gets_the_cleanup_not_only_the_one_that_needed_it():
    """A tier's spans are not the reference tier's, so "did it change anything at
    `hard`" is the wrong question to gate the replay on. `easy` has to be cleaned
    too, or one tier ships noise the others do not."""
    raw = _noisy_chords()
    model = song_model.build(grid=_grid(32), raw=raw, onsets=[])
    for difficulty in DIFFICULTIES:
        rendered = song_model.render(model, raw, difficulty)
        tonic = {c.quality for s in rendered for bar in s.bars for c in bar
                 if c.root_pc == 9}
        assert tonic == {"minor"}, difficulty


def test_the_cleanup_can_be_turned_off():
    """`CHORDS_THEORY_VOCABULARY=off`, for the same reason the vote has a switch:
    this edits chords the engine reported, and a posture that can only be judged
    by measurement has to be reversible without a deploy.

    Every *other* stage that can remove this noise is off too. There are four of
    them now and they overlap on exactly this input, so a test that left one
    running would pass whatever `consolidate` did — which is the failure mode
    that makes a switch look tested when it is not.
    """
    model = song_model.build(grid=_grid(32), raw=_noisy_chords(), onsets=[],
                             consolidate=False, vote=False, form_canonical=False)
    assert not model.vocabulary.touched
    qualities = {c.quality for s in model.sections for bar in s.bars for c in bar
                 if c.root_pc == 9}
    assert qualities == {"minor", "minor7", "major"}, "the noise is still there"


# --- how much of `hard` is real ----------------------------------------------

def test_the_model_measures_how_much_of_the_reference_tier_survived_intact():
    """`Cmaj9` is not a chord the container can carry, so it ships as `Cmaj7` —
    playable, and not what was heard. One bar in four here, and the number is the
    only thing in the analysis that can say the `hard` tier is a fiction."""
    raw = _sevenths()
    assert song_model.build(grid=_grid(), raw=raw, onsets=[]).exact_ratio == 1.0

    reduced = [replace(span, label="C:maj9") if i % 4 == 3 else span
               for i, span in enumerate(raw)]
    assert song_model.build(grid=_grid(), raw=reduced, onsets=[]).exact_ratio == 0.75


# --- one groove, two groups --------------------------------------------------

def test_a_groove_two_groups_share_is_named_for_both():
    """A pattern's id is content-addressed — meter and strokes, not the name —
    so two groups that strum alike compile to **one** embedded pattern. That is
    the right encoding; what was wrong is that the name came from whichever
    group was written last, so the player saw "Part 1 strum" on the pattern the
    second section points at."""
    labels = ["G", "D", "Em", "C"] * 2 + ["C", "F", "C", "G"] * 2
    raw = [RawChordSpan(start_ms=i * BAR_MS, end_ms=(i + 1) * BAR_MS, label=label,
                        confidence=0.9) for i, label in enumerate(labels)]
    model = song_model.build(grid=_grid(), raw=raw, onsets=[])

    assert len({g.label for g in model.groups}) == 2, "two groups"
    ids = {p.pattern.id for p in model.patterns.values()}
    assert len(ids) == 1, "and with no onsets, one shared quarter-note groove"
    assert {p.pattern.name for p in model.patterns.values()} == {"Part 1 & Part 2 strum"}


def test_a_groove_the_whole_song_plays_is_named_for_no_section():
    """Three groups and one quarter-note fallback between them — which is what
    every song analyzed without an onset detector looks like. At that point the
    groove belongs to the song rather than to any section of it, and a list of
    three names has stopped being a name."""
    labels = ["G", "D", "Em", "C"] * 2 + ["C", "F", "C", "G"] * 2 + ["Am", "F", "G", "Am"] * 2
    raw = [RawChordSpan(start_ms=i * BAR_MS, end_ms=(i + 1) * BAR_MS, label=label,
                        confidence=0.9) for i, label in enumerate(labels)]
    model = song_model.build(grid=_grid(len(labels)), raw=raw, onsets=[])

    assert len(model.patterns) >= 3
    assert len({p.pattern.id for p in model.patterns.values()}) == 1
    assert {p.pattern.name for p in model.patterns.values()} == {"Strum"}


# --- the consensus switch ----------------------------------------------------

def test_the_vote_can_be_turned_off():
    """`CHORDS_THEORY_CONSENSUS=off` is a supported posture: this is the one part
    of §20 that edits chords the engine reported, and the honest way to ship
    something judged only by measurement is to be able to turn it off."""
    model = song_model.build(grid=_grid(), raw=_chords(), onsets=[], vote=False)
    assert model.consensus.rewritten_bars == 0
    assert model.consensus.groups_voted == 0


# --- "if a pattern doesn't repeat, it isn't a pattern" ------------------------
#
# The user's phrasing, kept verbatim as the acceptance criterion. Measured on the
# stored catalog, one extraction per repeat group emitted 17 patterns for Let It
# Be — every one of them `repeats: 1` — and 23 for Assima. Clustering those by
# what they actually contain collapses them to 3 and 6, which is the order of
# magnitude a song has.

def _onsets(per_bar, *, strong=1.5):
    """One list of onsets from a `{bar: positions}` map — bar-local beats."""
    from app.analysis.types import Onset
    out = []
    for bar, positions in per_bar.items():
        for position in positions:
            out.append(Onset(t_ms=int(bar * BAR_MS + position * BEAT_MS),
                             strength=strong if position == 0.0 else 1.0))
    return sorted(out, key=lambda o: o.t_ms)


DDUUDU = (0.0, 1.0, 1.5, 2.5, 3.0, 3.5)


def test_a_section_that_never_repeats_does_not_get_a_groove_of_its_own():
    """`RepeatGroup.is_repeat` has existed since §20.4 and was never consulted,
    and this is the question it was written for: a group with one occurrence has
    one section's worth of onsets behind it, and emitting it as a distinct
    pattern tells the player the song has two strums when it has one."""
    labels = ["C", "G", "Am", "F"] * 2 + ["D", "A", "Bm", "E"] + ["C", "G", "Am", "F"] * 2
    raw = [RawChordSpan(start_ms=i * BAR_MS, end_ms=(i + 1) * BAR_MS, label=label,
                        confidence=0.9) for i, label in enumerate(labels)]
    # The verse strums D-DU-UD-U; the section that plays once plays something
    # else, and its extraction is the one that must not survive on its own.
    once = (0.0, 0.75, 1.75, 2.25, 3.25)
    onsets = _onsets({bar: once if 8 <= bar < 12 else DDUUDU
                      for bar in range(len(labels))})

    model = song_model.build(grid=_grid(len(labels)), raw=raw, onsets=onsets)
    assert model is not None
    assert any(not g.is_repeat for g in model.groups), "a group that plays once"
    assert len({p.pattern.id for p in model.patterns.values()}) == 1, \
        "one song, one groove — the section that plays once inherits it"


def test_a_group_that_repeats_keeps_the_groove_it_measured():
    """The rule cuts one way only. A song whose sections genuinely repeat and
    genuinely differ is allowed to say so — this is not "collapse everything"."""
    labels = ["C", "G", "Am", "F"] * 4 + ["D", "A", "Bm", "E"] * 4
    raw = [RawChordSpan(start_ms=i * BAR_MS, end_ms=(i + 1) * BAR_MS, label=label,
                        confidence=0.9) for i, label in enumerate(labels)]
    eighths = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)
    onsets = _onsets({bar: (0.0, 2.0) if bar >= 16 else eighths
                      for bar in range(len(labels))})

    model = song_model.build(grid=_grid(len(labels)), raw=raw, onsets=onsets)
    assert model is not None
    assert len([g for g in model.groups if g.is_repeat]) >= 2, "two repeating groups"
    assert len({p.pattern.id for p in model.patterns.values()}) >= 2, \
        "and two grooves, because the song really does play two"


def test_two_measurements_of_one_groove_are_one_groove():
    """Content addressing collapses grooves that are byte-identical. Two
    extractions of the same strum differing by one 16th are not byte-identical,
    and nothing in the pipeline noticed — measured on the stored songs, the mean
    pairwise similarity of a song's "different" patterns was about 0.5."""
    from app.analysis.model import _consolidate
    from app.analysis.strumming import extract

    dduudu = extract([(bar, p, 1.0) for bar in range(16) for p in
                      (0.0, 0.75, 1.0, 1.75, 2.0, 3.0)],
                     bar_beats=4.0, bars=16, tempo=120, name="Verse strum")
    nearly = extract([(bar, p, 1.0) for bar in range(16) for p in
                      (0.0, 0.75, 1.0, 1.75, 2.0, 2.75, 3.0)],
                     bar_beats=4.0, bars=16, tempo=120, name="Chorus strum")
    assert dduudu.pattern.id != nearly.pattern.id, "different bytes, before"

    merged = _consolidate({"A": dduudu, "B": nearly}, {"A": 32, "B": 8})
    assert merged["B"].pattern.id == dduudu.pattern.id, \
        "and one groove after — the one with more bars behind it wins"


def test_grooves_that_are_genuinely_different_are_not_merged():
    from app.analysis.model import _consolidate
    from app.analysis.strumming import extract

    quarters = extract([(bar, p, 1.0) for bar in range(16) for p in (0.0, 1.0, 2.0, 3.0)],
                       bar_beats=4.0, bars=16, tempo=120, name="Verse strum")
    offbeats = extract([(bar, p, 1.0) for bar in range(16) for p in (0.5, 1.5, 2.5, 3.5)],
                       bar_beats=4.0, bars=16, tempo=120, name="Chorus strum")

    merged = _consolidate({"A": quarters, "B": offbeats}, {"A": 32, "B": 32})
    assert merged["B"].pattern.id == offbeats.pattern.id


# --- the reported defect, end to end -----------------------------------------

def test_a_four_chord_song_heard_with_variants_charts_as_four_chords():
    """The complaint §20.9 and `form.folded` were built for, through the whole
    model rather than a unit at a time.

    `Em G D C`, a four-bar loop played eight times, fed in the way BTC actually
    hears such a thing: the Em as `Em7` in half the passes and once as a bare
    `E`, one `G` as `G7`, one `C` as `Csus4` — every variant hedged, which is the
    engine telling us it could not hear the note it added.

    Two failures used to come out of this, and they are one wobble wearing two
    hats: the chart carried six or seven distinct chords instead of four, and the
    eight passes collapsed into a *four*-fold repeat of an eight-bar section
    because a lag of 8 explained the alternating seventh better than a lag of 4.
    """
    heard = [
        [("E:min", .78), ("G", .80), ("D", .82), ("C", .84)],
        [("E:min7", .62), ("G", .80), ("D", .82), ("C", .84)],
        [("E:min", .74), ("G:7", .61), ("D", .82), ("C", .84)],
        [("E:min7", .66), ("G", .80), ("D", .82), ("C", .84)],
        [("E", .58), ("G", .80), ("D", .82), ("C:sus4", .55)],
        [("E:min7", .64), ("G:7", .59), ("D", .82), ("C", .84)],
        [("E:min", .76), ("G", .80), ("D", .82), ("C", .84)],
        [("E:min7", .63), ("G", .80), ("D", .82), ("C", .84)],
    ]
    raw = [RawChordSpan(start_ms=i * BAR_MS, end_ms=(i + 1) * BAR_MS,
                        label=label, confidence=confidence)
           for i, (label, confidence) in enumerate(x for row in heard for x in row)]

    model = song_model.build(grid=_grid(bars=32), raw=raw, onsets=[])
    assert model is not None

    for difficulty in DIFFICULTIES:
        sections = song_model.render(model, raw, difficulty)
        names = {render_name(c.root_pc, c.quality)
                 for s in sections for b in s.bars for c in b}
        assert names == {"Em", "G", "D", "C"}, f"{difficulty} tier carried {sorted(names)}"

    assert [s.repeats for s in model.sections] == [8], "the eight passes collapsed"
    assert model.total_bars == 32
