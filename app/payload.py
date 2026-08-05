"""The wire format — a Pydantic mirror of the app's ``CompositionPayload`` v2
container and its ``Arrangement`` tree.

**Ported from the Mo backend's ``app/payload.py``, deliberately unchanged in
substance** (handoff §12: emit the container the app already imports, and the
song plays with zero client changes; §16: mirror Mo rather than invent). Only
the id prefixes differ — ``yt:`` here, ``mo:`` there — so the two services
cannot collide in the Library's id space. Keeping this a near-copy is the point:
the app's importer is the judge of both, and two divergent mirrors of one Swift
type is how a wire format rots.

Field names are deliberately the *wire* names (camelCase), so a model dump IS
the JSON the app decodes — no alias layer to forget.

Two Swift facts drive the shape (see MIDI_Tab_Game/Services/ComposerService.swift
and Models/Arrangement.swift):

1. **Swift's synthesized ``Codable`` requires every non-optional field present**,
   even ones with defaults. A ``SongSection`` missing ``id``/``name``/``kindRaw``,
   or a ``Stroke`` missing ``id``/``accent``/``msOffset``, fails
   ``CompositionPayload.from(jsonData:)`` outright and the player sees
   "Mo's answer couldn't be read." Serialization here therefore always emits
   those fields (with server-minted UUIDs when Mo omitted them), and only
   Optionals are dropped (``exclude_none`` — matching Swift's encodeIfPresent).

2. ``BarRhythm`` is a Swift enum with associated values, so its JSON is the
   synthesized union encoding — exactly one of::

       {"inherit": {}}
       {"pattern": {"_0": "<pattern id>"}}
       {"custom":  {"_0": [<Stroke>, ...]}}
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_serializer, model_validator

SECTION_KINDS = {"intro", "verse", "preChorus", "chorus", "bridge", "solo", "outro", "custom"}
STROKE_DIRECTIONS = {"down", "up", "bass", "mute"}
MODES = {"major", "minor"}

# --- The three tempo ranges, in one place because they have to nest ----------
#
# They were written in three files with nothing tying them together, and the
# combination hard-killed songs: `meter.py` called 230 bpm a suspect octave and
# left it alone, `lint.py` refused it, and the player got "that video didn't
# produce a song we could play". Three ranges is right — they answer three
# different questions — but the ordering between them is a contract:
#
#   PLAUSIBLE  55–200   what this repertoire actually occupies. Outside it a
#                       reading is more likely an octave error than a tempo, and
#                       `analysis/meter.py` says so.
#   TEMPO      40–220   what the container accepts. Outside it `lint` refuses
#                       the song, so this is the range that decides whether a
#                       song exists at all.
#   PATTERN    30–300   a pattern's *suggested* tempo, which is advisory — the
#                       pattern plays at the song's tempo, whatever it says.
#
# PATTERN ⊇ TEMPO ⊇ PLAUSIBLE, and every step matters: a tempo the analysis is
# happy with must be one lint accepts, and a song tempo lint accepts must be
# legal on the pattern that carries it (patterns take the song's own tempo).
# `tests/test_lint.py` pins the nesting.
PLAUSIBLE_TEMPO_MIN, PLAUSIBLE_TEMPO_MAX = 55, 200
TEMPO_MIN, TEMPO_MAX = 40, 220
PATTERN_TEMPO_MIN, PATTERN_TEMPO_MAX = 30, 300

# Id namespace (handoff §12.3/§12.5). Mo uses "mo:"; this service uses "yt:", so
# the two backends can never mint the same Library id — `import` upserts on it.
SONG_PREFIX = "yt:"
PATTERN_PREFIX = "yt:pat-"
PROGRESSION_PREFIX = "yt:prog-"


def song_id(video_id: str, difficulty: str | None = None) -> str:
    """The idempotency key (§12.5): deterministic, so re-analyzing a video
    **replaces** its Library row rather than duplicating it.

    The optional difficulty suffix is what lets one video produce several rows
    without them colliding (``yt:<videoId>:easy``).
    """
    base = f"{SONG_PREFIX}{video_id}"
    return f"{base}:{difficulty}" if difficulty else base

_KIND_DISPLAY = {
    "intro": "Intro", "verse": "Verse", "preChorus": "Pre-Chorus", "chorus": "Chorus",
    "bridge": "Bridge", "solo": "Solo", "outro": "Outro", "custom": "Section",
}


def new_uuid() -> str:
    """Swift's ``UUID().uuidString`` is uppercase; match it for byte-parity."""
    return str(uuid.uuid4()).upper()


# A fixed namespace for content-derived ids. Any constant would do; this one is
# just a UUID minted once and pinned, so the derivation is reproducible across
# processes and machines.
_ID_NAMESPACE = uuid.UUID("6f3f1f2a-9c4a-4a3e-9a1e-5b7b1f0a2c31")


def derived_uuid(*parts: object) -> str:
    """A UUID that is a function of its inputs rather than of the clock.

    Analysis is deterministic, and its output should be too: re-analyzing a video
    must **replace** its Library row rather than duplicate it (§12.5), and a
    payload whose section and stroke ids churn on every run makes that harder to
    reason about, makes the §16.5 contract fixtures churn in git, and turns "did
    the analysis change?" into a question you can't answer by diffing.

    Shaped and cased like ``UUID().uuidString`` so Swift decodes it as any other
    UUID — nothing on the client knows or cares that it was derived.
    """
    return str(uuid.uuid5(_ID_NAMESPACE, "|".join(str(p) for p in parts))).upper()


def bar_beats(time_signature: str) -> float | None:
    """Strict "n/d" → quarter-note beats per bar ("6/8" → 3.0). None if malformed.

    (The app's ``Arrangement.barBeats`` falls back to 4 on junk; the lint wants
    to *catch* junk, so this parser returns None instead.)
    """
    parts = time_signature.split("/")
    if len(parts) != 2:
        return None
    try:
        n, d = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if n <= 0 or d <= 0:
        return None
    return n * 4.0 / d


class Stroke(BaseModel):
    id: str = Field(default_factory=new_uuid)
    beat: float
    direction: str = "down"
    accent: bool = False
    msOffset: int = 0
    strings: Optional[list[int]] = None


class ChordSpan(BaseModel):
    chordName: str
    startBeat: float
    lengthBeats: float


class BarRhythm(BaseModel):
    """The Swift enum, held internally as (kind, patternID?, strokes?)."""

    kind: Literal["inherit", "pattern", "custom"] = "inherit"
    patternID: Optional[str] = None
    strokes: Optional[list[Stroke]] = None

    @model_validator(mode="before")
    @classmethod
    def _from_wire(cls, value: Any) -> Any:
        if isinstance(value, dict) and "kind" not in value:
            if "inherit" in value:
                return {"kind": "inherit"}
            if "pattern" in value:
                inner = value["pattern"]
                return {"kind": "pattern", "patternID": inner.get("_0") if isinstance(inner, dict) else None}
            if "custom" in value:
                inner = value["custom"]
                return {"kind": "custom", "strokes": inner.get("_0") if isinstance(inner, dict) else None}
            raise ValueError(
                'rhythm must be one of {"inherit": {}}, {"pattern": {"_0": "<id>"}}, '
                '{"custom": {"_0": [strokes]}}'
            )
        return value

    @model_serializer
    def _to_wire(self) -> dict[str, Any]:
        if self.kind == "inherit":
            return {"inherit": {}}
        if self.kind == "pattern":
            return {"pattern": {"_0": self.patternID or ""}}
        return {"custom": {"_0": [s.model_dump(exclude_none=True) for s in (self.strokes or [])]}}


class Bar(BaseModel):
    id: str = Field(default_factory=new_uuid)
    chordSpans: list[ChordSpan] = Field(default_factory=list)
    rhythm: BarRhythm = Field(default_factory=BarRhythm)


class SongSection(BaseModel):
    id: str = Field(default_factory=new_uuid)
    name: str = ""
    kindRaw: str = "verse"
    chordNames: list[str] = Field(default_factory=list)
    patternID: str = ""
    beatsPerChord: int = 4
    repeats: int = 1
    tempoOverride: Optional[int] = None
    timeSignatureOverride: Optional[str] = None
    bars: Optional[list[Bar]] = None

    @property
    def display_name(self) -> str:
        """Mirror of ``SongSection.displayName`` — lint messages use it so Mo's
        errors read the way the player's import warnings would."""
        trimmed = self.name.strip()
        if trimmed:
            return trimmed
        return _KIND_DISPLAY.get(self.kindRaw, "Section")

    @property
    def is_playable(self) -> bool:
        if self.bars is not None:
            return any(bar.chordSpans for bar in self.bars)
        return bool(self.chordNames) and bool(self.patternID)


class Arrangement(BaseModel):
    sections: list[SongSection] = Field(default_factory=list)


class PatternPayload(BaseModel):
    id: str = Field(default_factory=lambda: f"{PATTERN_PREFIX}{uuid.uuid4()}")
    name: str = ""
    timeSignature: str = "4/4"
    tempo: int = 90
    strokes: list[Stroke] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class ProgressionPayload(BaseModel):
    id: str = Field(default_factory=lambda: f"{PROGRESSION_PREFIX}{uuid.uuid4()}")
    name: str = ""
    tonic: str = "C"
    mode: str = "major"
    chordNames: list[str] = Field(default_factory=list)
    beatsPerChord: int = 4
    tags: list[str] = Field(default_factory=list)


class CompositionPayload(BaseModel):
    # Tolerant decode, like the app's custom init(from:): absent version ⇒ 1,
    # absent patterns/progressions ⇒ [] — so a revise call carrying an older
    # payload still parses.
    version: int = 1
    id: str = ""
    title: str = ""
    tonic: str = "C"
    mode: str = "major"
    tempo: int = 90
    timeSignature: str = "4/4"
    chordNames: list[str] = Field(default_factory=list)
    patternID: str = ""
    beatsPerChord: int = 4
    repeats: int = 1
    arrangement: Optional[Arrangement] = None
    patterns: list[PatternPayload] = Field(default_factory=list)
    progressions: list[ProgressionPayload] = Field(default_factory=list)

    def wire_dict(self) -> dict[str, Any]:
        """The JSON object the app decodes. ``exclude_none`` mirrors Swift's
        encodeIfPresent-for-Optionals; everything else is always present."""
        return self.model_dump(exclude_none=True)

    def wire_json(self) -> str:
        return json.dumps(self.wire_dict(), ensure_ascii=False, sort_keys=True, indent=2)


def section_beats(section: SongSection, bar_beats_: float) -> float:
    """A section's length in quarter-note beats, mirroring the app's
    ``SongSection.lengthBeats(barBeats:)``.

    Note the asymmetry, which is the app's and not a mistake here: **bars mode
    counts whole bars and ignores ``repeats``** (``compileBars`` never sees it),
    while flat mode multiplies the recipe out.

    Lives here rather than in ``lint.py`` because it is a property of the
    container's shape, not of the validation — ``sync.py`` needs it to walk the
    chart, and ``lint.py`` imports ``sync``.
    """
    if section.bars is not None:
        return len(section.bars) * bar_beats_
    return len(section.chordNames) * max(1, section.beatsPerChord) * max(1, section.repeats)


def total_beats(payload: CompositionPayload) -> float:
    """The whole song's length on the beat axis the sidecar's anchors address."""
    beats = bar_beats(payload.timeSignature) or 4.0
    sections = payload.arrangement.sections if payload.arrangement else []
    return sum(section_beats(s, beats) for s in sections)


def section_duration_profile(section: SongSection) -> list[tuple[str, float]]:
    """The section's harmonic rhythm for ONE pass: (chord, beats-held), merging
    consecutive identical chords — representation-independent, so the flat
    ``chordNames``/``beatsPerChord`` form and the ``bars``/``chordSpans`` form
    collapse to the same profile. This is what pins down "G lasts 2 beats,
    Am lasts 4" (the harmonic-rhythm truth lint can't judge but tests/evals can
    assert against a known song)."""
    if section.bars is not None:
        spans = [(sp.chordName, float(sp.lengthBeats))
                 for bar in section.bars
                 for sp in sorted(bar.chordSpans, key=lambda s: s.startBeat)]
    else:
        spans = [(n, float(section.beatsPerChord)) for n in section.chordNames]
    merged: list[list] = []
    for name, beats in spans:
        if merged and merged[-1][0] == name:
            merged[-1][1] += beats
        else:
            merged.append([name, beats])
    return [(n, b) for n, b in merged]


def referenced_pattern_ids(arrangement: Arrangement | None) -> set[str]:
    """Mirror of ``CompositionPayload.referencedPatternIDs(in:)`` — every id the
    song's sections + per-bar ``.pattern`` overrides point at."""
    ids: set[str] = set()
    if arrangement is None:
        return ids
    for section in arrangement.sections:
        if section.patternID:
            ids.add(section.patternID)
        for bar in section.bars or []:
            if bar.rhythm.kind == "pattern" and bar.rhythm.patternID:
                ids.add(bar.rhythm.patternID)
    return ids
