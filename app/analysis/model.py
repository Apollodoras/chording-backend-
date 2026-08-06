"""§20.6 — the song model, and why the difficulty tiers are renders of it.

Before §20 the pipeline ran the whole of §5.4, §15 and §14 **once per
difficulty**: three passes of quantize → bars → segment → patterns, from the
same raw spans, producing three songs that were only *related* by having come
from the same recording. That is one analysis too few and two structures too
many, and it had a consequence the pipeline already had to defend against —
`easy`'s simplification can merge two bars into identical ones, which changes
what the segmenter collapses, so the three tiers could legitimately disagree
about where the sections are. One sidecar serves whichever tier the player asked
for, so `assemble` re-checked `lint_sync` against every tier to catch it.

A song has one structure. It is in one key, at one tempo, in one meter, with one
form; the difficulty tier changes which chord names are printed inside that
structure and nothing else. So the structure is built **once**, from the richest
tier, and each difficulty is a *render* of it:

    build()   raw spans ─► meter ─► axis ─► chords@hard ─► form ─► consensus
                                                       └─► key, patterns
                                                              │
    render()  ───────────────────────────────────────────────┴─► one tier's
                                                                 sections

Boundaries, repeat groups and patterns are now identical across tiers by
construction rather than by inspection, which is a stronger statement than the
cross-tier `lint_sync` check could ever make — and that check stays anyway,
because it costs nothing and it is the kind of thing that should not be removed
on the strength of an argument.

**Why the form is discovered twice.** `form.detect` runs, `consensus.apply`
votes, and then `form.detect` runs again on the corrected bars. That is not
belt-and-braces: before the vote, four passes of a verse that the engine heard
four slightly different ways cannot collapse with `repeats`, because `repeats`
demands they be identical (§15, and `form._sections_from` on why merely-similar
must never use it). After the vote they often *are* identical, and the second
pass is what turns that into the compact encoding the container wants. The first
pass exists to find the groups; the second to encode them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

from ..chords import HARD
from ..errors import AnalysisError
from . import consensus, form, postprocess, vocabulary
from .axis import BeatAxis, build_axis
from .consensus import ConsensusReport
from .keyfinder import DetectedKey, detect_key
from .meter import Meter, reconcile
from .strumming import ExtractedPattern, fallback
from .structure import BarChord, Section, bars_from_spans, spans_from_bars
from .types import BeatGrid, EnergyCurve, Onset, RawChordSpan
from .vocabulary import VocabularyReport

log = logging.getLogger("chords.model")

# The tier the model is built at. The richest one, because everything structural
# is easier to see before simplification has flattened distinctions — two
# sections that differ only in a 7th are two sections at `hard` and one at
# `easy`, and the structure should be the one the recording has.
REFERENCE = HARD


@dataclass(frozen=True)
class SongModel:
    """Everything the analysis concluded about the song, before it is a payload.

    This is the object the §20 layer exists to produce. `compile.py` turns it
    into the container the app imports; the sidecar reports its provenance; the
    benchmark scores it.
    """

    meter: Meter
    axis: BeatAxis
    key: DetectedKey
    sections: list[Section]
    # The **encoding** pass's groups: re-found on the corrected bars, which is
    # what lets identical occurrences collapse with `repeats`. The vote's own
    # provenance is not on these — it is on `vote_groups` below, and in
    # aggregate on `consensus`.
    groups: list[form.RepeatGroup]
    # One pattern per repeat group, keyed by rehearsal letter. Two sections of
    # the same group share a groove because they are the same music, which is
    # also why the pooled extraction had more evidence to work from.
    patterns: dict[str, ExtractedPattern] = field(default_factory=dict)
    consensus: ConsensusReport = field(default_factory=ConsensusReport)
    # §20.8's edits — what the song's own vocabulary corrected before the vote
    # ever ran. Separate from `consensus` because the two answer with different
    # evidence (the rest of the song, versus the same bar in another pass) and a
    # song can be helped a lot by one and not at all by the other.
    vocabulary: VocabularyReport = field(default_factory=VocabularyReport)
    # Whether the cleanup **ran**, which is not the same as whether it changed
    # anything. `render` needs the first: a tier's spans are not the reference
    # tier's, so a stage that found nothing to do at `hard` can still have work at
    # `easy`, and skipping it there would leave one tier holding noise the others
    # do not. Default False so a hand-assembled model renders exactly as it did
    # before this stage existed.
    consolidated: bool = False
    confidence: float = 0.0
    total_bars: int = 0
    # How much of the reference tier survived normalization exactly (§5.4) —
    # `postprocess.exact_ratio`. Low means `hard` is not really "the full
    # detected quality" on this recording, and the sidecar reports it.
    exact_ratio: float = 1.0
    # The groups the vote was taken over: the **first** pass's, whose block
    # boundaries are the ones `consensus.apply` scored, and which therefore
    # carry what it did. Voting over `groups` instead would be a different vote,
    # and a tier render has to reproduce the reference one rather than take its
    # own — so `render` uses these.
    vote_groups: list[form.RepeatGroup] = field(default_factory=list)
    # And the key it was taken with, for exactly the same reason. The vote's
    # diatonic tie-break consumes a key, and it consumed the **pre-vote**
    # reading — `key` above is the post-vote one, re-read off the corrected bars
    # so the song reports the key its chart actually settled on. Replaying the
    # vote with that one would be taking a different vote, not reproducing the
    # reference: the tie-break would break the other way on any bar the two keys
    # disagree about. Stored rather than recomputed because the reading that
    # matters here is a historical fact about a decision already made.
    vote_key: DetectedKey | None = None
    # And the key §20.8's consolidation ran with, which is one reading earlier
    # still: taken off the engine's own spans, before either the cleanup or the
    # vote had touched them. `render` replays the cleanup with it for exactly the
    # reason it replays the vote with `vote_key` — a tier is a render of the
    # reference decisions, not a fresh chance to decide differently.
    seed_key: DetectedKey | None = None

    @property
    def bar_beats(self) -> int:
        return self.axis.bar_beats

    @property
    def pattern_confidence(self) -> float:
        supported = [p.confidence for p in self.patterns.values() if not p.is_fallback]
        return round(sum(supported) / len(supported), 3) if supported else 0.0


def per_bar_energy(curve: EnergyCurve | None, axis: BeatAxis) -> list[float] | None:
    """A loudness curve in milliseconds → one number per **bar** of the chart.

    Averaged between the axis's own downbeats rather than on a fixed time grid,
    so bar *k*'s energy is measured over exactly the span bar *k* occupies —
    the same correspondence the anchors publish. Anything else would compare a
    section's loudness against a window that drifts out of phase with it over
    the course of a song.
    """
    if curve is None or not curve.values or curve.hop_ms <= 0:
        return None
    downbeats = axis.downbeats_ms
    if len(downbeats) < 2:
        return None
    return [curve.mean_between(start, end)
            for start, end in zip(downbeats, downbeats[1:])]


def build(*, grid: BeatGrid, raw: list[RawChordSpan], onsets: list[Onset],
          energy: EnergyCurve | None = None, vote: bool = True,
          consolidate: bool = True,
          correct_octave: bool = False) -> SongModel | None:
    """Features → the song model. **Pure**, and the half worth testing directly.

    Returns None when there is not enough *rhythm* to lay bars over, and raises
    when there is rhythm but no readable *harmony*. Two different failures with
    two different things to tell the player, and they are worth keeping apart:
    "that track's beat wasn't clear enough" is wrong and confusing on a
    perfectly steady recording the chord engine simply heard as silence.
    """
    meter = reconcile(grid, raw, correct_octave=correct_octave)
    axis = build_axis(meter.grid)
    if axis is None or not axis.is_usable:
        return None

    spans = postprocess.process(raw, axis, difficulty=REFERENCE)

    # §20.8, before anything is cut into bars and before the vote. The order is
    # not incidental: the vote reasons about a bar as a *unit* — one disagreement
    # anywhere in it contests the whole bar — so a two-beat wobble that the song
    # itself contradicts is better removed while the timeline is still spans.
    # Cleaning first also means the form detection that follows is looking at the
    # song rather than at the engine's stutter, and it is `form` that decides
    # which bars the vote gets to compare at all.
    seed_key = detect_key(spans)
    vocab = VocabularyReport()
    if consolidate and spans:
        spans, vocab = vocabulary.consolidate(
            spans, tonic_pc=seed_key.tonic_pc, mode=seed_key.scale,
            bar_beats=axis.bar_beats,
        )

    bars = bars_from_spans(spans, axis.bar_beats) if spans else []
    if not bars:
        raise AnalysisError("No chords could be read from that video.")

    # Re-read only if the cleanup moved something, on the same principle the vote
    # follows below: with nothing changed, this is the reading `seed_key` already
    # is, and re-running it could differ only for reasons that are not about the
    # song.
    key = detect_key(spans) if vocab.touched else seed_key
    bar_beats = float(axis.bar_beats)
    bar_energy = per_bar_energy(energy, axis)

    # First pass: find the groups. Fuzzy on purpose — this has to work on the
    # engine's mistakes, since removing them is what it is for.
    _, vote_groups = form.detect(bars, bar_beats=bar_beats, energy=bar_energy,
                                 tonic_pc=key.tonic_pc)

    report = ConsensusReport()
    # Captured before the re-read below can replace it: this is the key the vote
    # is about to be taken with, and `render` has to replay it with the same one.
    vote_key = key
    if vote:
        bars, report = consensus.apply(
            bars, vote_groups, bar_beats=bar_beats,
            tonic_pc=key.tonic_pc, mode=key.scale,
        )
        if report.touched:
            # The key was detected from the chords the engine reported, and the
            # vote then used it as its diatonic tie-break — so the key is
            # upstream of edits made partly on its own authority. Reading it
            # again off the corrected bars breaks that small circle, and it is
            # free: the chords are already in hand.
            #
            # Only when the vote actually changed a bar. Otherwise the input is
            # the input to the first detection, and re-running could differ only
            # in the trailing partial bar `bars_from_spans` drops — a change
            # with no reason behind it.
            key = detect_key(spans_from_bars(bars, axis.bar_beats))

    # Second pass: encode them. Occurrences the vote brought into line can now
    # collapse with `repeats`, which they could not before.
    sections, groups = form.detect(bars, bar_beats=bar_beats, energy=bar_energy,
                                   tonic_pc=key.tonic_pc)

    # The song's own meter, not `f"{bar_beats}/4"`: `bar_beats` is quarter-note
    # beats, so synthesising a signature from it labelled every 6/8 song's
    # patterns "3/4" — the same bar length, and a meter the song is not in.
    patterns = _patterns(groups, sections, onsets=onsets, axis=axis,
                         bar_beats=bar_beats, tempo=meter.tempo,
                         time_signature=meter.time_signature)

    return SongModel(
        meter=meter, axis=axis, key=key, sections=sections, groups=groups,
        patterns=patterns, consensus=report, vocabulary=vocab,
        consolidated=consolidate,
        confidence=postprocess.mean_confidence(spans),
        total_bars=sum(s.total_bars for s in sections),
        exact_ratio=postprocess.exact_ratio(spans),
        vote_groups=vote_groups, vote_key=vote_key, seed_key=seed_key,
    )


def render(model: SongModel, raw: list[RawChordSpan], difficulty: str) -> list[Section]:
    """One difficulty tier's sections, on the model's structure.

    The chords are re-derived for the tier — `easy` really does drop passing
    chords shorter than a bar, and that is duration work that cannot be done by
    renaming qualities in place — but the *boundaries* come from the model, so
    every tier tiles the song the same way and the one sidecar addresses all of
    them.

    The vote is replayed over the model's **`vote_groups`**, with the model's
    **`vote_key`**, and told not to record what it did. All three matter and none
    is cosmetic. Voting over `model.groups` (the encoding pass) meant the
    reference tier's render took a subtly different vote from the one `build`
    took, so "hard is the model" held by luck. Voting with `model.key` was the
    same mistake wearing the other hat: that is the key re-read *after* the vote,
    so on any song where the correction moved the reading, the replay's diatonic
    tie-break broke differently from the original's — and the guard on both is
    the same `consensus.touched`, so the two conditions never came apart.
    Recording, finally, would leave the model's groups carrying whichever tier
    compiled last instead of the reference vote, which is what the benchmark and
    the logs read.
    """
    spans = postprocess.process(raw, model.axis, difficulty=difficulty)
    if not spans:
        return []
    if model.consolidated:
        # Replayed with the model's `seed_key` for the same reason the vote is
        # replayed with `vote_key`: this is a render of decisions already taken.
        # `or model.key` for a model assembled by hand rather than by `build`.
        seed_key = model.seed_key or model.key
        spans, _ = vocabulary.consolidate(
            spans, tonic_pc=seed_key.tonic_pc, mode=seed_key.scale,
            bar_beats=model.axis.bar_beats,
        )
    bars = bars_from_spans(spans, model.axis.bar_beats)
    if not bars:
        return []
    if model.consensus.touched:
        # `or model.key` for a model assembled by hand rather than by `build` —
        # a test fixture, mostly. Before the re-read existed the two were the
        # same object, so it is also the answer that was always right.
        vote_key = model.vote_key or model.key
        bars, _ = consensus.apply(
            bars, model.vote_groups, bar_beats=float(model.axis.bar_beats),
            tonic_pc=vote_key.tonic_pc, mode=vote_key.scale,
            record=False,
        )
    return impose(model.sections, bars)


def impose(sections: list[Section], bars: list[list[BarChord]]) -> list[Section]:
    """Re-read the model's sections out of a differently-rendered bar list.

    The boundaries are the model's and do not move. What can change is whether a
    section's passes are still identical: simplification is applied to the whole
    timeline, and `easy`'s drop-passing-chords rule can merge across a barline,
    so two passes that agreed at `hard` may not at `easy`. When that happens the
    section is expanded to explicit bars rather than keeping a `repeats` that
    would replay the wrong pass — the same rule, and the same reason, as
    `form._sections_from`.
    """
    out: list[Section] = []
    for section in sections:
        length = len(section.bars)
        start = section.start_bar
        passes = [bars[start + i * length:start + (i + 1) * length]
                  for i in range(section.repeats)]
        passes = [p for p in passes if len(p) == length]
        if not passes:
            continue
        signatures = {tuple(tuple((c.root_pc, c.quality) for c in bar) for bar in p)
                      for p in passes}
        if len(signatures) == 1 and len(passes) == section.repeats:
            rendered = Section(kind=section.kind, name=section.name, bars=passes[0],
                               repeats=section.repeats, start_bar=start,
                               group=section.group)
        else:
            flat = [bar for one_pass in passes for bar in one_pass]
            rendered = Section(kind=section.kind, name=section.name, bars=flat,
                               repeats=1, start_bar=start, group=section.group)
        out.append(rendered)

    cursor = 0
    for section in out:
        section.start_bar = cursor
        cursor += section.total_bars
    return out


def _patterns(groups: list[form.RepeatGroup], sections: list[Section], *,
              onsets: list[Onset], axis: BeatAxis, bar_beats: float, tempo: int,
              time_signature: str) -> dict[str, ExtractedPattern]:
    """One pattern per repeat group (§14, pooled per §20.4).

    With no onset detector at all every group gets the quarter-note fallback,
    which is the intended behaviour: the app requires a pattern, and a boring one
    that plays is worth more than no song.
    """
    names = {}
    for section in sections:
        names.setdefault(section.group, section.name or section.kind.title())

    out: dict[str, ExtractedPattern] = {}
    for group in groups:
        name = f"{names.get(group.label, group.label)} strum"
        if not onsets:
            out[group.label] = fallback(bar_beats=bar_beats, tempo=tempo, name=name,
                                        time_signature=time_signature)
            continue
        out[group.label] = consensus.pattern_for_group(
            group, onsets=onsets, axis=axis, bar_beats=bar_beats, tempo=tempo,
            name=name, time_signature=time_signature,
        )
    return _rename_shared_grooves(out, names)


def _rename_shared_grooves(patterns: dict[str, ExtractedPattern],
                           names: dict[str, str]) -> dict[str, ExtractedPattern]:
    """Give a groove two groups both play a name that is true of both.

    A pattern's id is **content-addressed** — meter plus strokes, and
    deliberately not the name (§12.5, so an unchanged groove keeps its id). Two
    groups that strum the same way therefore hash to the same id and compile to
    **one** embedded `PatternPayload`, which is the right encoding and the whole
    reason the id is a hash. What was wrong is the name that object ended up
    with: `compile` writes them into a dict keyed by id, so the last group
    written won, and the player saw "Verse strum" on the pattern the chorus
    section points at. Nothing plays wrong; the label is simply not this song's.

    Renaming is safe precisely because the name is not in the hash: the id, and
    so the wire, is unchanged. Two sharers are named for both; beyond two the
    groove is the song's rather than any section's, and a list of names has
    stopped being a name.
    """
    sharers: dict[str, list[str]] = {}
    for label, extracted in patterns.items():
        sharers.setdefault(extracted.pattern.id, []).append(label)

    out: dict[str, ExtractedPattern] = {}
    for label, extracted in patterns.items():
        group_labels = sharers[extracted.pattern.id]
        if len(group_labels) < 2:
            out[label] = extracted
            continue
        distinct = list(dict.fromkeys(names.get(g, g) for g in group_labels))
        shared = f"{' & '.join(distinct)} strum" if len(distinct) <= 2 else "Strum"
        out[label] = replace(extracted,
                             pattern=extracted.pattern.model_copy(update={"name": shared}))
    return out
