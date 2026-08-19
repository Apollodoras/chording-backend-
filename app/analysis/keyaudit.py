"""§20.10 — the chart against its own key, once, at the end.

The one question nothing else in this pipeline asks. `keyfinder.py` finds a key
*from* the chords and then the key is used for exactly two things: spelling on
the wire, and breaking ties inside `consensus`/`canon`/`vocabulary`. Nothing ever
turns around and asks whether the finished chart is a thing that could be in that
key. So a song comes out with `C` and `Cm` both functioning as its tonic — which
is not a close call about a borrowed chord, it is two different songs — and every
check in the building passes it, because `lint` compares the song against itself
and the song is perfectly self-consistent.

That is the owner's second symptom, and this module is the layer that can see it.

Diatonic-or-not is the wrong test
---------------------------------

The obvious rule — reject anything outside the scale — would be much worse than
the disease. Borrowed chords are *ordinary* in this repertoire: bVII and iv in a
major key are in half the folk canon, the I7 that leans into IV is in the other
half, and secondary dominants are everywhere. `harmony.diatonic_fit` is
deliberately graded rather than boolean for exactly this reason, and every use of
it elsewhere is a tie-break that is never allowed to reject a chord on its own.

So the test here is three-valued, and the middle value is the whole design:

- **diatonic** — the chord belongs to the mode (`keyfinder.degrees`, which is
  built from the scale rather than hand-written, and already carries the
  harmonic-minor V and vii° that a bare aeolian table would call foreign);
- **borrowed** — outside the scale and *idiomatic anyway*, from the table below.
  A chord that is borrowed is left alone as firmly as a diatonic one;
- **foreign** — neither. Only a foreign chord is a candidate for anything.

And even a foreign chord is not an error by itself. What this module acts on is
narrower: a **conflict**, meaning one root the song plays two ways, where one
reading is at home in the key and the other is foreign to it, and the two differ
in *colour alone*. That shape — `C#` against `C#m` in E major, `F#` against `F#m`
in A major, `E7` against `E` in E — is a recognizer that could not hear a third
or invented a seventh, not a modulation and not a borrowing. A modulation moves
several roots at once and a borrowing has one reading, not two.

Why `vocabulary.snap` cannot do this
------------------------------------

It is the right module and the wrong evidence. §20.8 decides by **mass**: the
song's dominant reading of a root wins if it is 6× the minority's, the minority
holds under 15% of the root's evidence, and it turns up on at most two occasions.
Every one of those is a statement about how *often* the engine said something,
and a systematic mishearing satisfies none of them — Don't Stop Believin's `C#`
is not a two-occasion wobble, it is what BTC hears in that passage every time the
passage comes round. §20.8's own docstring says as much about the case it cannot
reach.

The key is evidence that did not come from counting the engine's outputs, which
is precisely why it can speak where counting cannot. It is also weaker evidence,
so it is fenced accordingly: same root, colour only, a move `vocabulary.SNAP_TO`
allows, believed less on average, and never a chord the borrowings table
protects.

What it may not do
------------------

Move a root — the load-bearing fact (§12.2) — or touch a quality
`vocabulary.NEVER_SNAPPED` lists, or act where *both* readings are at home. It
also stops at reporting when it cannot repair: `KeyAuditReport.conflicts` names
every disagreement it found, repaired or not, and that string is carried to the
sidecar. A conflict this module declines to touch is one an operator should be
able to read, and before this existed there was nowhere for it to be written down.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from ..chords import (
    DIMINISHED,
    DIMINISHED7,
    DOMINANT7,
    MAJOR,
    MINOR,
    MINOR7,
    prefers_flats_for,
    spell,
)
from . import harmony
from .keyfinder import degrees
from .postprocess import merge
from .types import GridSpan
from .vocabulary import CONFIDENCE_MARGIN, NEVER_SNAPPED, SNAP_TO

log = logging.getLogger("chords.keyaudit")

# The most of a root's evidence a foreign reading may hold and still be treated
# as a mishearing. Above it the song is telling us something — either that the
# chord is real, or that the **key** is wrong — and either way the answer is to
# report the conflict rather than to edit the chart into agreement with it.
#
# Measured, and the corpus separates cleanly on this one number (share of the
# root's duration × confidence held by the out-of-key reading):
#
#   Creep              Cm  against C    35.5%   the song's own chord; E-minor is
#                                              a misread key and this is the
#                                              signal that says so
#   Mary Jane          A   against Am   23.5%   both are in the song
#   Don't Stop         C#  against C#m  40.4%   a whole passage misheard, not a
#                                              wobble — and its belief points the
#                                              other way too
#   Someone Like You   F#  against F#m  14.1%   the mishearing this layer is for
#   Don't Stop         G#  against G#m   7.5%   likewise
#   Don't Stop         A7  against A    11.1%   likewise
#
# 0.20 sits in the gap between 14.1 and 23.5 rather than on the edge of either.
# `vocabulary.MINORITY_SHARE` is the same idea at 0.15 and is deliberately not
# reused: that gate guards an edit made on *mass alone*, and this one has the key
# standing behind it, so it can afford to be a little wider.
MINORITY_SHARE = 0.20

DIATONIC = "diatonic"
BORROWED = "borrowed"
FOREIGN = "foreign"

# Chords outside the scale that this repertoire plays anyway, by degree above the
# tonic. Everything here is protected from repair as firmly as a diatonic chord
# — the point of the table is to be the reason a rule *declines*.
#
# The entries are the ones with names, which is the discipline that keeps this
# from becoming "allow everything":
#
#   major   I7      the tonic seventh leaning into IV — "In My Life" is four
#                   brief, doubtful A7s in A, and they are the song. This is the
#                   single most important row in the table: `vocabulary.py`
#                   records that exact case as the hardest it had to settle, and
#                   a key rule without this row would flatten all four.
#           bVII    the mixolydian cadence, everywhere in folk and rock
#           iv      the minor plagal
#           bVI/bIII the modal-interchange pair
#           II/III/VI as majors or dominants: secondary dominants of V, vi and ii
#           vii°    the leading-tone diminished
#   minor   IV      the dorian inflection
#           bII     the Neapolitan
#           I       the picardy third, and the parallel-major tonic
#           bVII7   the backdoor dominant
#
# Keyed by the *internal* mode names (`harmony.SCALES`), not by the container's
# major/minor, because that is what the caller has and because dorian and
# mixolydian songs borrow differently from ionian ones.
_MAJORISH = {
    0: frozenset({DOMINANT7}),
    2: frozenset({DOMINANT7}),
    3: frozenset({MAJOR}),
    4: frozenset({DOMINANT7}),
    5: frozenset({MINOR, MINOR7}),
    8: frozenset({MAJOR}),
    9: frozenset({DOMINANT7}),
    10: frozenset({MAJOR, DOMINANT7}),
    11: frozenset({DIMINISHED, DIMINISHED7}),
}
_MINORISH = {
    0: frozenset({MAJOR}),
    1: frozenset({MAJOR}),
    5: frozenset({MAJOR}),
    9: frozenset({MAJOR}),
    10: frozenset({DOMINANT7}),
}
BORROWINGS: dict[str, dict[int, frozenset[str]]] = {
    "ionian": _MAJORISH,
    "lydian": _MAJORISH,
    "mixolydian": _MAJORISH,
    "aeolian": _MINORISH,
    "dorian": _MINORISH,
    "phrygian": _MINORISH,
    "locrian": _MINORISH,
}


@dataclass(frozen=True)
class KeyAuditReport:
    """What the audit found, and what it did about it.

    `conflicts` is every disagreement seen, whether or not it was repaired, as
    readable pairs (`"C# vs C#m"`). It is the number the owner's second symptom
    would have shown up in, and nothing published it before.
    """

    resolved_spans: int = 0
    conflicts: tuple[str, ...] = ()

    @property
    def touched(self) -> bool:
        return bool(self.resolved_spans)


def standing(root_pc: int, quality: str, tonic_pc: int, mode: str) -> str:
    """`DIATONIC`, `BORROWED` or `FOREIGN` — where this chord sits in the key."""
    degree = (root_pc - tonic_pc) % 12
    if quality in degrees(mode).get(degree, frozenset()):
        return DIATONIC
    if quality in BORROWINGS.get(mode, {}).get(degree, frozenset()):
        return BORROWED
    return FOREIGN


def _at_home(root_pc: int, quality: str, tonic_pc: int, mode: str) -> bool:
    return standing(root_pc, quality, tonic_pc, mode) != FOREIGN


def resolve(spans: list[GridSpan], *, tonic_pc: int, mode: str
            ) -> tuple[list[GridSpan], KeyAuditReport]:
    """Settle the roots the song reads two incompatible ways.

    One pass over the whole track, after `vocabulary.consolidate` and on the same
    timeline — spans, before anything is cut into bars, for the same reason §20.8
    runs there: a reading the song contradicts is better removed while the
    timeline is still chords rather than after it has been sliced into bars the
    vote will judge as units.
    """
    if not spans:
        return [], KeyAuditReport()

    # root → quality → (beats, belief × beats). Duration-weighted, like §20.8's
    # profile: a chord heard briefly at 0.3 is not the same evidence as one heard
    # for a bar at 0.9, and airtime alone would say it nearly is.
    beats: dict[tuple[int, str], float] = {}
    weighted: dict[tuple[int, str], float] = {}
    for span in spans:
        if span.length_beats <= 0:
            continue
        key = (span.root_pc % 12, span.quality)
        beats[key] = beats.get(key, 0.0) + span.length_beats
        weighted[key] = weighted.get(key, 0.0) + span.length_beats * span.confidence

    by_root: dict[int, list[str]] = {}
    for root, quality in beats:
        by_root.setdefault(root, []).append(quality)

    verdicts: dict[tuple[int, str], tuple[int, str]] = {}
    conflicts: list[str] = []
    for root, qualities in sorted(by_root.items()):
        if len(qualities) < 2:
            continue
        home = [q for q in qualities if _at_home(root, q, tonic_pc, mode)]
        away = [q for q in qualities if not _at_home(root, q, tonic_pc, mode)]
        if not home or not away:
            continue
        # The song's own answer among the readings that are at home in the key —
        # by mass, so a diatonic chord the song barely plays cannot capture the
        # root away from one it plays constantly.
        winner = max(home, key=lambda q: (weighted[(root, q)], q))
        winner_belief = weighted[(root, winner)] / beats[(root, winner)]
        total = sum(weighted[(root, q)] for q in qualities)
        for quality in sorted(away):
            conflicts.append(_conflict(root, quality, winner, tonic_pc, mode))
            if quality in NEVER_SNAPPED:
                continue
            if winner not in SNAP_TO.get(quality, ()):
                continue
            if not harmony.is_near_miss((root, quality), (root, winner)):
                continue
            if total <= 0 or weighted[(root, quality)] / total > MINORITY_SHARE:
                # **Too much of this root to be a mishearing.** The gate that
                # keeps the layer from acting on its own worst case: when the key
                # itself is wrong, the chart's real chords are the ones that look
                # foreign, and they look foreign *a lot*. Creep is exactly that —
                # it is in G and plays G B C Cm, the key finder returns E minor
                # (in which the real `Cm` is bvi and foreign while a misheard `C`
                # is bVI and at home), and `Cm` holds a third of everything that
                # root ever sounds. Without this the audit deletes the chord the
                # song is known for and reports high confidence in the result.
                #
                # And note what `keyfinder.DetectedKey.confidence` cannot do
                # here, because it is the obvious gate and it is the wrong one:
                # Creep's **wrong** key is the most confident reading in the whole
                # corpus (0.305, against 0.004–0.070 for the nine keys that are
                # right), because that number is a margin over the runner-up and
                # a wrong answer can win by a mile. Mass on the disputed root is
                # the signal that actually separates these cases.
                #
                # The conflict is still reported. A chart that keeps its chords
                # and a sidecar line saying the key is in doubt is the outcome
                # wanted here, and it is the one an operator can act on.
                continue
            belief = weighted[(root, quality)] / beats[(root, quality)]
            if belief >= winner_belief - CONFIDENCE_MARGIN:
                # Believed as much as the key's own answer. The gate every
                # correcting layer here carries, and the one that keeps this a
                # no-op on flat-confidence input: ground truth arrives at 1.0 on
                # every span, so no reading is ever less believed than another
                # and nothing is ever rewritten.
                continue
            verdicts[(root, quality)] = (root, winner)

    if not verdicts:
        return list(spans), KeyAuditReport(conflicts=tuple(sorted(set(conflicts))))

    out: list[GridSpan] = []
    resolved = 0
    for span in spans:
        found = verdicts.get((span.root_pc % 12, span.quality))
        if found is None:
            out.append(span)
            continue
        # `exact` goes with it, exactly as in `vocabulary.snap`: this is no longer
        # the quality the engine reported, so the span cannot claim to have
        # reached the container intact.
        out.append(replace(span, quality=found[1], exact=False))
        resolved += 1

    if resolved:
        log.info("key audit: %d span(s) moved onto the key's own reading (%s)",
                 resolved, "; ".join(sorted(set(conflicts))))
    return merge(out), KeyAuditReport(resolved_spans=resolved,
                                      conflicts=tuple(sorted(set(conflicts))))


def _conflict(root: int, foreign: str, home: str, tonic_pc: int, mode: str) -> str:
    """One conflict, as a line a musician can act on: `"C#(VI) vs C#m(vi)"`.

    The degrees are what make it readable. `C# vs C#m` alone does not say which
    of the two the key expects, and the whole content of the finding is that one
    of them is at home and the other is not — so the numeral goes in beside the
    name. It also distinguishes the two ways this can be wrong at a glance: a
    conflict on the tonic degree usually means the *key* is wrong (Creep reports
    `Cm(i) vs C(I)` because it is really in G), and a conflict on vi or iii
    usually means a third was misheard.
    """
    flats = prefers_flats_for(tonic_pc, harmony.MODE_PROJECTION.get(mode, "major"))
    name = spell(root, flats)
    return (f"{name}{_suffix(foreign)}({harmony.roman(root, foreign, tonic_pc, mode)}) vs "
            f"{name}{_suffix(home)}({harmony.roman(root, home, tonic_pc, mode)})")


def _suffix(quality: str) -> str:
    """The chord-symbol suffix, for the report's benefit only — this is the one
    place in the module that is about being read rather than about being right."""
    return {MAJOR: "", MINOR: "m", DOMINANT7: "7", MINOR7: "m7",
            DIMINISHED: "dim", DIMINISHED7: "dim7"}.get(quality, quality)
