"""Server-side song validation — the backend's port of the app importer's
``ComposerService.importReport`` plus the handoff §12.4 sanity checks.

**Ported from the Mo backend's ``app/lint.py``** (chord-backend-handoff.md §12.4:
"Port the app's own importer check and run it server-side. Never return a payload
that would warn. Mo does this as a lint-in-the-loop tool ... do the same"), with
``yt:`` id prefixes and one section this service needs and Mo does not:
``lint_sync``, the §13.2 beat-axis invariant.

Two layers:

- ``repair`` fixes the *mechanical* bookkeeping deterministically (ids, version,
  the flat v1 summary, stroke ordering, the revise id contract) so Mo never
  burns a correction attempt on things a server can do.
- ``lint`` then validates the repaired payload and returns human-readable
  problems. Inside the agent loop these come back to Mo as the ``submit_song``
  tool result; the endpoint has **no path** that returns an unlinted song.

Self-containment is a hard rule here (owner decision, plan rev 2): there is no
bundled pattern catalog — every referenced ``patternID`` must resolve against
the payload's own embedded ``patterns``.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from .chords import is_valid_chord, pitch_class
from .payload import (
    MODES,
    PATTERN_PREFIX,
    PATTERN_TEMPO_MAX,
    PATTERN_TEMPO_MIN,
    PROGRESSION_PREFIX,
    SECTION_KINDS,
    SONG_PREFIX,
    STROKE_DIRECTIONS,
    STROKE_BANDS,
    TEMPO_MAX,
    TEMPO_MIN,
    CompositionPayload,
    PatternPayload,
    SongSection,
    Stroke,
    bar_beats,
    new_uuid,
    referenced_pattern_ids,
    # Defined in payload.py (they describe the container's shape, not the
    # validation) and re-exported here, where callers have always found them.
    section_beats,
    total_beats,
)
from .sync import VideoSync

_EPS = 1e-6

# `TEMPO_MIN`/`TEMPO_MAX` are imported above rather than defined here — callers
# still find them on this module, but the number now lives next to the two other
# tempo ranges it has to nest with (see `payload.py`).

# How far one bar's anchor gap may sit from the song's own modal bar before it
# counts as a different length bar. A quarter is wide: the failures this catches
# are half-bars and double-bars, and the width is what keeps rubato, a ritardando
# and a live drummer out of it.
ANCHOR_GAP_TOLERANCE = 0.25
# ...and how many bars may be off before the map is not worth publishing. A real
# song is allowed the odd inserted bar; a song where one bar in five is a
# different length has a beat map, not a bar grid.
ANCHOR_GAP_SHARE = 0.10
# How close two gaps must be to be "the same bar length" when the mode is taken.
ANCHOR_GAP_CLUSTER = 0.10
# Below this many bars there is no mode to speak of, and the check stands down.
ANCHOR_GAP_MIN_BARS = 8


# ---------------------------------------------------------------------------
# Repair — deterministic bookkeeping, before lint
# ---------------------------------------------------------------------------

def repair(payload: CompositionPayload, *, force_id: str | None = None) -> CompositionPayload:
    """Fix what a server can fix without musical judgment. Mutates and returns
    the payload.

    - ``version`` → 2 (Mo always emits the self-contained container).
    - ``id`` → the revise contract's id when given (replace-not-duplicate),
      else keep Mo's, else mint ``mo:<uuid>``.
    - Blank ids anywhere in the tree → minted UUIDs (Swift requires them).
    - Strokes sorted by beat (the app's adapter sorts anyway; sorted output
      reads honestly).
    - Flat v1 summary fields rewritten from section 1 (the Library row's
      metadata — the handoff's "mirror section 1" rule, done here so Mo can't
      get it wrong).
    """
    payload.version = 2
    if force_id is not None:
        payload.id = force_id
    elif not payload.id.strip():
        payload.id = f"{SONG_PREFIX}{new_uuid()}"

    for pattern in payload.patterns:
        if not pattern.id.strip():
            pattern.id = f"{PATTERN_PREFIX}{new_uuid().lower()}"
        pattern.strokes.sort(key=lambda s: s.beat)
        _fill_stroke_ids(pattern.strokes)

    for progression in payload.progressions:
        if not progression.id.strip():
            progression.id = f"{PROGRESSION_PREFIX}{new_uuid().lower()}"

    if payload.arrangement is not None:
        for section in payload.arrangement.sections:
            if not section.id.strip():
                section.id = new_uuid()
            for bar in section.bars or []:
                if not bar.id.strip():
                    bar.id = new_uuid()
                if bar.rhythm.kind == "custom" and bar.rhythm.strokes:
                    bar.rhythm.strokes.sort(key=lambda s: s.beat)
                    _fill_stroke_ids(bar.rhythm.strokes)
        _mirror_flat_summary(payload)

    return payload


def _fill_stroke_ids(strokes: list[Stroke]) -> None:
    for stroke in strokes:
        if not stroke.id.strip():
            stroke.id = new_uuid()


def _mirror_flat_summary(payload: CompositionPayload) -> None:
    """Flat v1 fields = section 1's recipe (what the Library list renders)."""
    sections = payload.arrangement.sections if payload.arrangement else []
    if not sections:
        return
    first = sections[0]
    if first.bars is not None:
        # Bars mode: summarize as the consecutive-deduped chord walk.
        names: list[str] = []
        for bar in first.bars:
            for span in bar.chordSpans:
                if not names or names[-1] != span.chordName:
                    names.append(span.chordName)
        payload.chordNames = names
    else:
        payload.chordNames = list(first.chordNames)
    payload.patternID = first.patternID
    payload.beatsPerChord = first.beatsPerChord
    payload.repeats = first.repeats


# ---------------------------------------------------------------------------
# Lint
# ---------------------------------------------------------------------------

def lint(payload: CompositionPayload) -> list[str]:
    """Every problem that would make the song warn, mis-decode, or silently
    play wrong on the device. Empty list == safe to return to the player."""
    problems: list[str] = []
    add = problems.append

    # --- Song-level metadata -------------------------------------------------
    if payload.version != 2:
        add(f'version must be 2 (got {payload.version})')
    if not payload.id.strip():
        add("id must be a non-empty unique string")
    if not payload.title.strip():
        add("title must not be empty")
    if payload.mode not in MODES:
        add(f'mode must be "major" or "minor" (got "{payload.mode}")')
    if pitch_class(payload.tonic) is None:
        add(f'tonic "{payload.tonic}" is not a note name (letter A–G plus optional #/b)')
    if not (TEMPO_MIN <= payload.tempo <= TEMPO_MAX):
        add(f"tempo {payload.tempo} is outside the sane {TEMPO_MIN}–{TEMPO_MAX} BPM range")
    song_bar_beats = bar_beats(payload.timeSignature)
    if song_bar_beats is None:
        add(f'timeSignature "{payload.timeSignature}" is not an "n/d" meter string')

    # --- Structure ------------------------------------------------------------
    sections = payload.arrangement.sections if payload.arrangement else []
    if not sections:
        add("the song needs an arrangement with at least one section")
    elif not any(s.is_playable for s in sections):
        add("no section is playable (each needs chords + a pattern, or bars with chord spans)")

    embedded_ids = [p.id for p in payload.patterns]
    embedded = set(embedded_ids)
    if len(embedded) != len(embedded_ids):
        dupes = sorted({i for i in embedded_ids if embedded_ids.count(i) > 1})
        add(f"embedded pattern ids must be unique (duplicated: {', '.join(dupes)})")

    # --- Sections ---------------------------------------------------------------
    for section in sections:
        _lint_section(section, payload, song_bar_beats, add)

    # --- Self-containment: every referenced pattern is embedded -----------------
    missing = referenced_pattern_ids(payload.arrangement) - embedded
    for pattern_id in sorted(missing):
        add(
            f'pattern "{pattern_id}" is referenced but not defined in "patterns" — '
            f"there is no pattern library; embed every pattern you reference"
        )

    # --- Embedded patterns -------------------------------------------------------
    for pattern in payload.patterns:
        _lint_pattern(pattern, add)

    # --- Progressions (optional garnish — light checks) --------------------------
    for progression in payload.progressions:
        if progression.mode not in MODES:
            add(f'progression "{progression.name or progression.id}": mode must be "major" or "minor"')
        for name in progression.chordNames:
            if not is_valid_chord(name):
                add(f'progression "{progression.name or progression.id}": couldn’t read chord "{name}"')

    return problems


# ---------------------------------------------------------------------------
# The sidecar's own lint (§13.2) — this service only; Mo has no sidecar
# ---------------------------------------------------------------------------

def lint_sync(payload: CompositionPayload, sync: VideoSync) -> list[str]:
    """Every §13.2 problem with a sidecar, fatal and advisory together.

    Kept as the flat list it has always been, because that is what the bench and
    the contract fixtures assert on ("this pairing is clean"). The pipeline reads
    `lint_sync_problems` instead, which is the same walk with the severities kept
    apart — see that function for why the distinction had to exist.
    """
    return list(lint_sync_problems(payload, sync).all)


@dataclass(frozen=True)
class SyncProblems:
    """A sidecar's problems, split by what they cost the player.

    **fatal** — the sidecar is not a usable map from video time to song beat, so
    handing it to the client produces a cursor that does something nonsensical:
    no anchors to interpolate between, an axis that runs backwards, a bar grid
    the client cannot derive. There is nothing to ship.

    **advisory** — the map works; it is imperfect. The tail runs on extrapolated
    tempo, a bar or two is a different length from its neighbours, the chart's
    sections are not whole bars. The song plays against the recording and the
    cursor tracks it; some of it is less exact than we would like.

    The split exists because the two used to be one list, and *any* entry in it
    withheld the sidecar (§13.3). That rule is what made "play with the video"
    disappear from songs that had a perfectly serviceable beat map with a ragged
    tail — and withholding does not repair a chart, it only removes the video.
    An imperfect sidecar shipped `lowConfidence: true` tells the player the truth
    and still lets them play along; withholding it tells them nothing and takes
    the product's whole differentiator away. See `analysis/pipeline.py`.
    """

    fatal: tuple[str, ...] = ()
    advisory: tuple[str, ...] = ()

    @property
    def all(self) -> tuple[str, ...]:
        return self.fatal + self.advisory


def lint_sync_problems(payload: CompositionPayload, sync: VideoSync) -> SyncProblems:
    """Everything that would let the cursor walk off the song (§13.2), by severity.

    Run only when a sidecar is actually being returned. A song with
    ``videoSync: None`` plays self-paced and none of this applies — and after the
    §13.3 amendment that is a state only `fatal` can produce.
    """
    fatal: list[str] = []
    advisory: list[str] = []
    stop = fatal.append          # the map is unusable
    warn = advisory.append       # the map is usable and imperfect

    beats = bar_beats(payload.timeSignature)
    if beats is None:
        stop(f'videoSync: the song\'s timeSignature "{payload.timeSignature}" is not an "n/d" meter string')
        return SyncProblems(tuple(fatal), tuple(advisory))
    if sync.timeSignature != payload.timeSignature:
        stop(
            f'videoSync: timeSignature "{sync.timeSignature}" disagrees with the song\'s '
            f'"{payload.timeSignature}" — they address the same bars'
        )

    sections = payload.arrangement.sections if payload.arrangement else []

    # --- One uniform grid ----------------------------------------------------
    # The client derives bar index as floor(beat / barBeats) on a SINGLE
    # song-level tempo and meter (JamSongSheet.from), so an override makes the
    # beat axis the anchors address stop existing. Fatal for exactly that reason:
    # not "less accurate", but "there is no axis to be accurate on".
    for section in sections:
        if section.tempoOverride is not None:
            stop(
                f"{section.display_name}: a video-synced song may not carry tempoOverride — "
                f"the client's bar grid is computed on one song-level tempo, so tempo drift "
                f"belongs in beatAnchors instead"
            )
        if section.timeSignatureOverride is not None:
            stop(
                f"{section.display_name}: a video-synced song may not carry "
                f"timeSignatureOverride — same reason as tempoOverride"
            )

    # --- Whole bars ----------------------------------------------------------
    # Sections concatenate, so a section that isn't a whole number of bars slides
    # every later bar off the recording's downbeats — silently, and further with
    # each section.
    #
    # Advisory, and the reasoning is the one that runs through this whole split:
    # the chart is equally wrong played self-paced, so withholding the sidecar
    # fixes nothing and costs the player the recording. The drift is real and it
    # is reported; `lowConfidence` carries it to the client.
    for section in sections:
        length = section_beats(section, beats)
        if abs(length / beats - round(length / beats)) > _EPS:
            warn(
                f"{section.display_name}: is {length:g} beats, which is not a whole number of "
                f"{beats:g}-beat bars — every later bar would drift off the recording's downbeats"
            )

    # --- The anchors themselves ---------------------------------------------
    anchors = sync.beatAnchors
    if not anchors:
        stop("videoSync: beatAnchors is empty — without it there is no map from video time to song beat")
        return SyncProblems(tuple(fatal), tuple(advisory))
    if len(anchors) < 2:
        stop("videoSync: a single anchor cannot express tempo — emit one per bar (every downbeat is ideal)")

    # Both directions of monotonicity are fatal: `song_beat_at` interpolates
    # between neighbouring anchors, so a repeated or reversed tMs divides by a
    # non-positive span and a reversed songBeat runs the cursor backwards
    # through the chart. Neither is "imprecise"; both are broken.
    for previous, current in zip(anchors, anchors[1:]):
        if current.tMs <= previous.tMs:
            stop(f"videoSync: beatAnchors must strictly increase in tMs (saw {previous.tMs} then {current.tMs})")
            break
    for previous, current in zip(anchors, anchors[1:]):
        if current.songBeat <= previous.songBeat:
            stop(
                f"videoSync: beatAnchors must strictly increase in songBeat "
                f"(saw {previous.songBeat} then {current.songBeat})"
            )
            break

    # --- Bars that are about as long as the other bars ----------------------
    # Every check in this function until now compared the song against *itself*,
    # and a chart that is uniformly wrong about its own bar lengths is perfectly
    # self-consistent: the anchors increase, they land on bar boundaries, they
    # cover the song, and bar 9 is still half the length of bar 8. That is the
    # blindness that let "beat-map anomalies" and "it changes tempo mid song"
    # reach the player — the client reads its cursor speed off these gaps, so
    # anchors a half-bar apart *are* a tempo change as far as it is concerned.
    #
    # Two things about how this is measured, both load-bearing:
    #
    #   compare against the modal gap, not the payload tempo. The tempo is
    #   derived from beat intervals and is right even when the bars are wrong, so
    #   comparing to it would pass exactly the songs that are broken.
    #
    #   withhold on a share, not on one bar. Real recordings have rubato and real
    #   songs have the occasional inserted bar; a single-bar trigger would flag
    #   songs that are fine.
    #
    # Advisory since the §13.3 amendment. The anchors still track the recording —
    # they were *measured* on it — so a song with ragged bars plays against the
    # video better than it plays against a metronome, which is the only other
    # thing on offer. It ships flagged.
    _lint_anchor_gaps(anchors, warn)

    # Anchors are downbeats, so each must land on a bar boundary of the axis the
    # chart produces — that IS the §13.2 invariant, stated as a check. Advisory:
    # an anchor a fraction of a beat off a barline still interpolates, and the
    # error it introduces is bounded by that fraction.
    for anchor in anchors:
        bars_in = anchor.songBeat / beats
        if abs(bars_in - round(bars_in)) > _EPS:
            warn(
                f"videoSync: anchor at songBeat {anchor.songBeat:g} is not on a bar boundary "
                f"({beats:g} beats/bar) — anchors mark downbeats"
            )
            break

    if any(a.tMs < 0 for a in anchors):
        stop("videoSync: anchor timestamps are milliseconds from video start and cannot be negative")
    if sync.durationMs <= 0:
        stop(f"videoSync: durationMs {sync.durationMs} must be positive")
    elif anchors[-1].tMs > sync.durationMs:
        warn(
            f"videoSync: the last anchor ({anchors[-1].tMs} ms) is past the end of the video "
            f"({sync.durationMs} ms)"
        )

    # --- Coverage ------------------------------------------------------------
    # The client extrapolates past the last anchor, so short coverage is not
    # fatal — but it means the tail of the song is running on an assumed tempo,
    # which is exactly what the anchors exist to avoid.
    song_length = total_beats(payload)
    if song_length > 0 and anchors[-1].songBeat + beats < song_length - _EPS:
        warn(
            f"videoSync: anchors stop at songBeat {anchors[-1].songBeat:g} but the song is "
            f"{song_length:g} beats — the tail would run on extrapolated tempo"
        )
    # And the other end of the same rule. An anchor past the song's last beat
    # addresses a bar that does not exist: the map claims the recording is still
    # playing chart at a point where the chart has run out. `song_length` itself
    # is fine and expected — that anchor marks the *end* boundary of the final
    # bar, which is why a song of N bars ships N+1 anchors.
    #
    # This is the check a dropped section trips, and it is now **repaired rather
    # than reported**: `fit_sync` trims the overrun before the sidecar
    # is ever linted (see `analysis/compile.py`). Reaching it here means the trim
    # did not, so it stays advisory — an overrun tail is extrapolation the client
    # already survives, not a broken map.
    if song_length > 0 and anchors[-1].songBeat > song_length + _EPS:
        warn(
            f"videoSync: anchors run to songBeat {anchors[-1].songBeat:g} but the song ends at "
            f"{song_length:g} beats — those anchors address bars that do not exist"
        )

    # Out-of-range confidences are a *bug*, not a weak reading — `build_sync`
    # clamps both — so they stay fatal: a container field outside its own domain
    # is the one thing here that says the producer, not the recording, is wrong.
    if not (0.0 <= sync.tempo.confidence <= 1.0):
        stop(f"videoSync: tempo confidence {sync.tempo.confidence} must be between 0 and 1")
    if sync.patternConfidence is not None and not (0.0 <= sync.patternConfidence <= 1.0):
        stop(f"videoSync: patternConfidence {sync.patternConfidence} must be between 0 and 1")

    return SyncProblems(tuple(fatal), tuple(advisory))


def _lint_anchor_gaps(anchors: list, add) -> None:
    """Consecutive anchors must sit about one modal bar apart. See `lint_sync`.

    Silent on short songs: a mode taken over five bars is not a mode, and a
    32-second demo with one inserted bar is not the failure this is looking for.
    """
    gaps = [b.tMs - a.tMs for a, b in zip(anchors, anchors[1:])]
    if len(gaps) < ANCHOR_GAP_MIN_BARS:
        return
    modal = _modal_gap(gaps)
    if modal is None or modal <= 0:
        return

    off = [g for g in gaps if abs(g - modal) > modal * ANCHOR_GAP_TOLERANCE]
    share = len(off) / len(gaps)
    if share > ANCHOR_GAP_SHARE:
        worst = max(off, key=lambda g: abs(g - modal))
        add(
            f"videoSync: {len(off)} of {len(gaps)} bars ({share:.0%}) are more than "
            f"{ANCHOR_GAP_TOLERANCE:.0%} away from this song's own bar of {modal} ms "
            f"(worst: {worst} ms) — the beat map has bars the recording does not, "
            f"and the cursor would change speed at each one"
        )


def _modal_gap(gaps: list[int]) -> int | None:
    """The song's own bar length in milliseconds, as a **mode** with a tolerance.

    A plain mode is useless on millisecond timings — no two bars are the same
    length to the millisecond — and a median is dragged by the very anomalies
    this is looking for: a song with 40% half-bars can have a median between the
    two. So each gap is scored by how many other gaps sit within
    `ANCHOR_GAP_CLUSTER` of it, and the winner's cluster is summarized by its
    median. That is the length most of this song's bars actually are.
    """
    if not gaps:
        return None
    best: list[int] = []
    for candidate in gaps:
        cluster = [g for g in gaps if abs(g - candidate) <= candidate * ANCHOR_GAP_CLUSTER]
        if len(cluster) > len(best):
            best = cluster
    return int(median(best)) if best else None


def _lint_section(
    section: SongSection,
    payload: CompositionPayload,
    song_bar_beats: float | None,
    add,
) -> None:
    label = section.display_name

    if section.kindRaw not in SECTION_KINDS:
        add(f'{label}: kindRaw "{section.kindRaw}" is not one of {sorted(SECTION_KINDS)}')
    if section.beatsPerChord < 1:
        add(f"{label}: beatsPerChord must be ≥ 1")
    if section.repeats < 1:
        add(f"{label}: repeats must be ≥ 1")
    if section.tempoOverride is not None and not (TEMPO_MIN <= section.tempoOverride <= TEMPO_MAX):
        add(f"{label}: tempoOverride {section.tempoOverride} is outside {TEMPO_MIN}–{TEMPO_MAX} BPM")

    effective_bar_beats = song_bar_beats
    if section.timeSignatureOverride is not None:
        effective_bar_beats = bar_beats(section.timeSignatureOverride)
        if effective_bar_beats is None:
            add(f'{label}: timeSignatureOverride "{section.timeSignatureOverride}" is not an "n/d" meter string')

    # Harmony lane — every chord name must parse (mirrors importReport).
    if section.bars is not None:
        chord_names = [span.chordName for bar in section.bars for span in bar.chordSpans]
    else:
        chord_names = section.chordNames
    for name in chord_names:
        if not is_valid_chord(name):
            add(
                f'{label}: couldn’t read chord "{name}" — use the accepted grammar '
                f"(root A–G + #/b + one of: m, 7, maj7, m7, dim, dim7, m7b5, aug, sus2, sus4); "
                f"simplify slash chords and extensions"
            )

    # Rhythm lane — mirrors importReport's pattern-needs walk.
    if section.bars is not None:
        if not any(bar.chordSpans for bar in section.bars):
            add(f"{label}: has bars but no chord spans — the app silently skips it; give it harmony or delete it")
        for index, bar in enumerate(section.bars, start=1):
            _lint_bar(label, index, bar, effective_bar_beats, add)
        if any(bar.rhythm.kind == "inherit" for bar in section.bars) and not section.patternID:
            add(f'{label}: a bar inherits the section pattern but patternID is empty')
    elif section.chordNames:
        if not section.patternID:
            add(f"{label}: no strumming pattern — the app silently skips this section")
    else:
        add(f"{label}: empty section (no chords, no bars) — the app silently skips it; fill it or delete it")


def _lint_bar(label: str, index: int, bar, effective_bar_beats: float | None, add) -> None:
    where = f"{label}, bar {index}"
    for span in bar.chordSpans:
        if span.startBeat < -_EPS or span.lengthBeats <= _EPS:
            add(f"{where}: chord span for \"{span.chordName}\" must start ≥ 0 with positive length")
        elif effective_bar_beats is not None and span.startBeat + span.lengthBeats > effective_bar_beats + _EPS:
            add(
                f'{where}: chord span for "{span.chordName}" runs past the bar '
                f"({span.startBeat} + {span.lengthBeats} > {effective_bar_beats} beats)"
            )
    if bar.rhythm.kind == "pattern" and not (bar.rhythm.patternID or "").strip():
        add(f"{where}: rhythm pattern override has an empty id")
    if bar.rhythm.kind == "custom":
        _lint_strokes(where, bar.rhythm.strokes or [], effective_bar_beats, add)
        if not (bar.rhythm.strokes or []):
            add(f"{where}: custom rhythm has no strokes")


def _lint_pattern(pattern: PatternPayload, add) -> None:
    label = f'pattern "{pattern.name or pattern.id}"'
    if not pattern.name.strip():
        add(f"{label}: give the pattern a short human-readable name")
    beats = bar_beats(pattern.timeSignature)
    if beats is None:
        add(f'{label}: timeSignature "{pattern.timeSignature}" is not an "n/d" meter string')
    if not pattern.strokes:
        add(f"{label}: has no strokes — a pattern is one bar of strokes")
    if not (PATTERN_TEMPO_MIN <= pattern.tempo <= PATTERN_TEMPO_MAX):
        add(f"{label}: suggested tempo {pattern.tempo} looks wrong "
            f"(expected {PATTERN_TEMPO_MIN}–{PATTERN_TEMPO_MAX} BPM)")
    _lint_strokes(label, pattern.strokes, beats, add)


def _lint_strokes(label: str, strokes: list[Stroke], beats_in_bar: float | None, add) -> None:
    seen_beats: set[float] = set()
    for stroke in strokes:
        if stroke.direction not in STROKE_DIRECTIONS:
            add(f'{label}: stroke direction "{stroke.direction}" must be one of down/up/bass/mute')
        if stroke.beat < -_EPS:
            add(f"{label}: stroke beat {stroke.beat} is negative — beats are 0-indexed from the bar start")
        elif beats_in_bar is not None and stroke.beat >= beats_in_bar - _EPS:
            add(
                f"{label}: stroke at beat {stroke.beat} falls outside one bar "
                f"(0 ≤ beat < {beats_in_bar}; beats are 0-indexed, so beat 0 is the downbeat) — "
                f"the app silently drops out-of-range strokes"
            )
        if stroke.strings is not None:
            if not stroke.strings:
                add(f"{label}: stroke at beat {stroke.beat} has an empty strings list — omit the field for a full strum")
            elif any(not (1 <= lane <= 6) for lane in stroke.strings):
                add(f"{label}: stroke at beat {stroke.beat} names string lanes outside 1–6 (1 = high E)")
        if stroke.band is not None and stroke.band not in STROKE_BANDS:
            # "full" is caught by this too, and deliberately: it is a real band
            # but not a transmissible one — the wire spells it as an absent field
            # (§14.1), and a payload that says it out loud would read as a claim
            # the client has no rule for.
            add(f'{label}: stroke at beat {stroke.beat} has band "{stroke.band}" — '
                f"must be one of {'/'.join(STROKE_BANDS)}, or omitted for a full-range stroke")
        key = round(stroke.beat, 6)
        if key in seen_beats:
            add(f"{label}: two strokes share beat {stroke.beat} — merge them (one stroke per onset)")
        seen_beats.add(key)
