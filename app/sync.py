"""The video-sync sidecar (handoff §13) — and the invariant that makes it work.

A ``CompositionPayload`` is a **relative** grid: tempo + meter + bars. A real
recording is **absolute** and drifts. The sidecar is the map between them, and it
travels *beside* the song rather than inside it, so the payload stays
byte-identical to what ``ComposerService.import`` expects and stays shareable
as-is.

All times are **integer milliseconds** from video start. Float drift over four
minutes is real, and the client's own clock is milliseconds.

---

**The load-bearing invariant (§13.2), and where it came from.**

The client addresses its cursor in *song beats* (``JamSongSheet.dueBeat(of:)`` =
``barIndex × barBeats + beatOffset``), and a video clock is in milliseconds, so
``beatAnchors`` is the interpolation table between the two::

    songBeat = piecewise_linear(beatAnchors, videoCurrentTimeMs - offsetMs)

That only works if ``songBeat`` here is the *same axis* the compiled chart
produces — bar *n* of the payload must start at ``songBeat = n × barBeats``.

Reading the client settles one thing the handoff did not spell out.
``JamSongSheet.from(chart:barBeats:secondsPerBeat:)`` derives every stroke's bar
as ``floor(beat / barBeats)`` **on a single song-level tempo and meter**; its own
comment says per-section tempo/meter overrides make that "approximate ... only
the bar grouping can drift". Approximate bar grouping is fine for a self-paced
campfire song and fatal for a video-synced one, because the anchors would be
addressing beats the cursor never lands on.

So: **a song that carries a sidecar must have one uniform grid** — no
``tempoOverride``, no ``timeSignatureOverride``, anywhere. Tempo drift belongs
entirely in the anchors, which is precisely what they are for. ``app/lint.py``
enforces this, and it is the rule to reach for first if a synced song ever walks
off its own chart.
"""

from __future__ import annotations

from bisect import bisect_right
from typing import Optional

from pydantic import BaseModel, Field

from .payload import CompositionPayload, bar_beats, section_beats

SOURCE_YOUTUBE = "youtube"

_EPS = 1e-9


class Confidence(BaseModel):
    """A measured value and how much to believe it."""

    bpm: float
    confidence: float


class BeatAnchor(BaseModel):
    """One (song beat ↔ video millisecond) correspondence.

    ``songBeat`` is float because a downbeat can land on a fractional beat of the
    song's grid once an offset is applied; ``tMs`` is integer because it is a
    clock reading.
    """

    songBeat: float
    tMs: int


class EngineVersions(BaseModel):
    """What produced this map — ``name@version``, e.g. ``btc@1.2.0``.

    Persisted on every stored map (§5.3) so a cache can be invalidated
    *selectively* when one engine is upgraded, rather than dropped wholesale.
    """

    chords: str
    beats: str


class DownbeatRepair(BaseModel):
    """What §20.2a changed about the tracker's downbeat sequence.

    Both counts are edits nothing downstream can see — the anchors that ship are
    the repaired ones, and a repaired grid is *more* self-consistent than the one
    it replaced, so no check anywhere would notice. Which is precisely why they
    are published rather than logged: a song reporting forty dropped downbeats
    had a beat tracker that fired forty bar lines the music does not have, and
    that is worth knowing about the recording even though the chart is now right.
    """

    dropped: int = 0
    inserted: int = 0


class TheoryReport(BaseModel):
    """What the §20 layer concluded, and what it changed to get there.

    Carried on the sidecar rather than in the payload for two reasons. The
    payload is the container the app already imports and must stay
    byte-identical to what any other client decodes (§12); and these are facts
    about the *analysis*, not about the song — the same distinction that put
    ``engine`` and ``analyzedAt`` here rather than in the composition.

    ``rewrittenBars`` is the one to watch. It counts bars where a repeated
    section's majority overruled a single occurrence — edits the player cannot
    see and neither lint can check (`consensus.py`). A song reporting a high
    count is one whose verses the engine heard inconsistently; a song reporting
    a high ``contestedBars`` is one whose verses genuinely differ. Both are
    worth knowing and neither is visible anywhere else.

    ``scale`` is the modal answer the container has no field for: a song the
    payload calls "G major" may really be G mixolydian (§20.5), and this is
    where that survives.

    ``exactRatio`` is the other one worth watching, and it says something no
    other number here does: how much of the track survived §5.4's normalization
    without losing information. A low value means the chart is a
    *fiction* on this recording — a jazz standard reduced to triads and dominant
    sevenths is a different song, and it is a different song that lints clean,
    plays fine and reports high confidence. This is the only field that can say
    so.
    """

    scale: str = "ionian"
    sections: int = 0
    groups: int = 0
    rewrittenBars: int = 0
    contestedBars: int = 0
    # §20.9's share of ``rewrittenBars`` — slots the count could not decide and
    # belief could (a tie, or a plurality pointing the other way from the
    # confidence). Published separately because it is the number that says
    # whether the second reduction is earning its place, and because it cannot be
    # inferred from the other two: switching it off moves bars from here into
    # ``contestedBars`` rather than changing any total.
    weighedBars: int = 0
    # §20.8's two counts, and they are the ones that move on a song whose
    # sections do *not* repeat enough for the vote to speak: spans pulled onto the
    # quality the song plays on that root everywhere else, and brief near-misses
    # absorbed back into the chord they interrupted (`vocabulary.py`). Like
    # ``rewrittenBars`` these are edits nothing downstream can see, which is why
    # they are published rather than logged.
    snappedSpans: int = 0
    absorbedIslands: int = 0
    # §20.10's two, and they answer a question nothing here could ask before:
    # does the finished chart sit in the key the payload names? ``keyConflicts``
    # lists every root the song reads two incompatible ways ("C# vs C#m") whether
    # or not anything was done about it, and ``resolvedSpans`` counts the spans
    # moved onto the key's own reading. A song with conflicts and no resolutions
    # is one the audit found something in and declined to touch — the state an
    # operator most needs to be able to see, and the one that used to be
    # invisible (`keyaudit.py`).
    resolvedSpans: int = 0
    keyConflicts: tuple[str, ...] = ()
    # §20.5b — where the windowed key reading changed, as lines a person can read
    # ("bar 96: E major -> F major"). Detection only; the container carries one
    # key and nothing acts on this. Empty on a single-key song, which is nearly
    # all of them — and non-empty is the one signal that distinguishes "the engine
    # misheard the last chorus" from "the last chorus is a tone up", which are
    # indistinguishable in every other field here.
    modulations: tuple[str, ...] = ()
    # §21's two counts. ``canonicalBars`` is bars brought into line with their own
    # section's progression, and it is emphatically *not* comparable with
    # ``rewrittenBars`` above: that one counts places the engine is thought to
    # have misheard, this one counts places the chart now states the song's form
    # rather than one pass of it. The other two say where the agreement ran out:
    # ``heldBeats`` is beats the occurrences tied on, where the bar held the chord
    # it had already settled on, and ``splitBars`` is slots where nothing settled
    # and every occurrence kept its own reading. A song with a high ``splitBars``
    # is one whose sections genuinely vary, and one this layer correctly declined
    # to speak for.
    canonicalBars: int = 0
    # §21b — bars given the chord that held most of them, because the song was
    # measured to change chord a bar at a time. Zero on a song whose harmony moves
    # faster than its bar (Wonderwall changes twice a bar and must not be touched),
    # which is a finding about the song rather than a stage that did not run.
    settledBars: int = 0
    heldBeats: int = 0
    splitBars: int = 0
    # Beats the downbeat grid was rotated by against the tracker's own answer
    # (§20.2). Non-zero means the harmony and the pulse disagreed about where
    # the bar started, and the harmony won.
    phaseShift: int = 0
    meterSource: str = "tracker"
    # §20.2a. `irregularBars` counts bars still disagreeing with the song's own
    # modal bar **after** the repair — which is exactly the set `axis.py`
    # resamples, so it is also "how many bars of this chart are evenly spread
    # rather than tracked". Zero on a clean grid, a handful on a song with real
    # metric irregularity in it (Here Comes The Sun), and a large number on a
    # song whose meter we have probably got wrong.
    irregularBars: int = 0
    downbeatsRepaired: DownbeatRepair = Field(default_factory=DownbeatRepair)
    # The tempo is outside the range this repertoire plausibly occupies, *after*
    # any correction — so a song reporting it is one whose beat grid may be an
    # octave out, and it is flagged `lowConfidence` for that reason alone.
    tempoOctaveSuspect: bool = False
    # Octaves the beat grid was moved by to get there: -1 halved, +1 doubled.
    # Always 0 unless `CHORDS_THEORY_TEMPO_OCTAVE` is on.
    tempoOctaveShift: int = 0
    # Share of the chart, by duration, whose chord quality reached the
    # container intact. **Optional and defaulting to None**, which reads as "not
    # measured" — the honest answer for every sidecar written before this field
    # existed, and one that neither 1.0 ("all exact") nor 0.0 ("none of it") can
    # tell the truth about.
    exactRatio: Optional[float] = None
    # How far this recording sits from A440, in cents, and whether that offset
    # was close enough to a quarter tone that the semitone it snapped to was a
    # coin flip. Published because it is the one number that distinguishes "the
    # engine misheard the harmony" from "the engine heard it in the wrong key" —
    # and a chart that is a semitone out is otherwise perfectly self-consistent,
    # so no other field here can betray it. Zero on any recording at concert
    # pitch, which is most of them.
    tuningCents: float = 0.0
    tuningAmbiguous: bool = False


class VideoSync(BaseModel):
    source: str = SOURCE_YOUTUBE
    videoId: str
    durationMs: int
    # Nullable, admin-settable per video (§6 as amended). The app has NO latency
    # calibration any more — it was deleted with the scoring it corrected — so a
    # player-facing early/late nudge (§17.7) is the only correction path that
    # exists, and this is the value it nudges.
    offsetMs: Optional[int] = 0
    tempo: Confidence
    timeSignature: str
    lowConfidence: bool = False
    # §14: the strumming pattern's direction assignment is a *convention*, not a
    # measurement, and support can be thin. Carried so the client can say so.
    patternConfidence: Optional[float] = None
    beatAnchors: list[BeatAnchor] = Field(default_factory=list)
    engine: EngineVersions
    analyzedAt: str
    # §20's provenance. Optional so every sidecar written before the theory
    # layer existed still validates, and additive so the client — whose
    # `VideoSync` is a synthesized `Codable` and therefore ignores keys it does
    # not know — decodes it unchanged.
    analysis: Optional[TheoryReport] = None


class AnalyzeResponse(BaseModel):
    """The §13.1 envelope: the song and its sidecar, side by side.

    ``videoSync`` is ``None`` only when the analysis produced no usable map from
    video time to song beat at all — no anchors, or anchors that cannot be
    interpolated. It is **not** withheld for a weak reading any more: a song that
    came from a recording should play with that recording, and a doubtful one
    says so through ``lowConfidence`` rather than by having its video taken away
    (§13.3 as amended). See `analysis/pipeline.py`.
    """

    song: dict
    videoSync: Optional[VideoSync] = None


def anchors_for(
    downbeat_ms: list[int],
    *,
    bar_beats: float,
    first_bar_index: int = 0,
) -> list[BeatAnchor]:
    """Downbeat times → anchors on the payload's beat axis.

    One anchor per downbeat, which is what §13.2 asks for ("every downbeat is
    ideal — cheap, and it makes the interpolation exact rather than
    approximate"): bar *n* starts at ``songBeat = n × barBeats``, exactly the
    axis ``dueBeat`` walks.

    ``first_bar_index`` exists because the song may not start at the recording's
    first downbeat — an intro the analysis discarded, or a pickup. It shifts the
    beat axis without touching the timestamps.
    """
    return [
        BeatAnchor(songBeat=(first_bar_index + i) * bar_beats, tMs=int(round(t)))
        for i, t in enumerate(downbeat_ms)
    ]


def song_beat_at(anchors: list[BeatAnchor], t_ms: float) -> float:
    """The client's interpolation, in Python — the reference implementation the
    tests assert the anchors against.

    Piecewise-linear inside the table; **linear extrapolation** off either end
    using the nearest segment's slope, so a video that keeps playing past the
    last anchor keeps producing beats rather than freezing the cursor. A single
    anchor has no slope to extrapolate along, so it holds.
    """
    if not anchors:
        return 0.0
    if len(anchors) == 1:
        return anchors[0].songBeat

    times = [a.tMs for a in anchors]
    index = bisect_right(times, t_ms) - 1
    # Clamp to a real segment: -1 (before the first) uses the first segment's
    # slope, and the last anchor uses the final one.
    index = max(0, min(index, len(anchors) - 2))
    left, right = anchors[index], anchors[index + 1]

    span_ms = right.tMs - left.tMs
    if span_ms <= 0:
        return left.songBeat
    ratio = (t_ms - left.tMs) / span_ms
    return left.songBeat + ratio * (right.songBeat - left.songBeat)


# ---------------------------------------------------------------------------
# The other half of the reference implementation: what the player actually SEES
# ---------------------------------------------------------------------------
#
# ``song_beat_at`` maps a video clock onto the song's beat axis. That is only
# half of what the client does — the other half is reading the chord off the
# chart at that beat, and until both halves existed in Python there was no way
# to ask the only question that matters for a chords-over-a-recording product:
#
#     at video millisecond t, does the app show the chord that is playing?
#
# Everything the service validates today (``lint``, ``lint_sync``) checks the
# song's *internal* consistency, which a chart can satisfy while being uniformly
# wrong against the recording. These two functions are what let a test and the
# benchmark score the deliverable instead of the engine.


def chord_at_song_beat(payload: CompositionPayload, beat: float) -> str | None:
    """The chord name the compiled chart sounds at ``beat``, or None past the end.

    Walks the arrangement the way the app's compiler does: sections concatenate
    on one uniform grid, a bars-mode section indexes its bars directly (and
    ignores ``repeats``, as ``compileBars`` does), and a flat section tiles its
    ``chordNames`` at ``beatsPerChord`` and loops through ``repeats``.
    """
    beats_per_bar = bar_beats(payload.timeSignature) or 4.0
    sections = payload.arrangement.sections if payload.arrangement else []

    if beat < -_EPS:
        # Before the song's first downbeat. The client extrapolates backwards off
        # the first anchor, so this is reachable whenever a recording opens with
        # a pickup or an intro the chart doesn't cover — and the honest answer is
        # that there is no chord yet, not the first one wrapped around.
        return None

    cursor = 0.0
    for section in sections:
        length = section_beats(section, beats_per_bar)
        if beat < cursor + length - _EPS:
            local = beat - cursor
            if section.bars is not None:
                index = int(local // beats_per_bar)
                if not (0 <= index < len(section.bars)):
                    return None
                in_bar = local - index * beats_per_bar
                spans = sorted(section.bars[index].chordSpans, key=lambda s: s.startBeat)
                for span in spans:
                    if span.startBeat - _EPS <= in_bar < span.startBeat + span.lengthBeats - _EPS:
                        return span.chordName
                # Between spans (a bar the compiler leaves partly uncovered): the
                # last chord to have started is the one still sounding.
                earlier = [s for s in spans if s.startBeat - _EPS <= in_bar]
                return earlier[-1].chordName if earlier else None
            if not section.chordNames:
                return None
            per_chord = max(1, section.beatsPerChord)
            one_pass = len(section.chordNames) * per_chord
            index = int((local % one_pass) // per_chord)
            return section.chordNames[index]
        cursor += length
    return None


def chord_at_video_ms(payload: CompositionPayload, sync: "VideoSync", t_ms: float) -> str | None:
    """What the player sees at video millisecond ``t_ms`` — both halves together.

    This is the composition the client performs every frame, and the function a
    test or the benchmark should score against ground truth. ``offsetMs`` is
    applied here exactly as §13.2 specifies it.
    """
    if not sync.beatAnchors:
        return None
    return chord_at_song_beat(payload, song_beat_at(sync.beatAnchors, t_ms - (sync.offsetMs or 0)))
