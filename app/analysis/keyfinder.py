"""§20.5 — key detection: modal inside, major/minor on the wire.

``CompositionPayload`` requires ``tonic`` and ``mode``, and the app uses them for
exactly one thing: **note spelling** (``Key(tonic, mode)?.spelling``). Get the key
wrong and the song still plays; the player just reads "A#m" where every chord
sheet in the world prints "Bbm".

So this does not need to be a musicological key-finder over audio, and
deliberately isn't one. It scores the *chords we already decided on* — the
strongest key evidence there is, and free — against every mode of every tonic.
Working from chords rather than from chroma also means the key can never
contradict the chart, which is the failure that would actually confuse a player.

**Why the modes are here, given the container has none.** The obvious reading of
§12.2 is that church modes are wasted work: the wire has `major` and `minor`, so
why score more. Because the mode is how you find the **tonic**, and the tonic is
what spelling keys off. Scored against major and minor only, `G F C G` comes out
as *A minor* — every chord is diatonic to it, so it ties with C major and beats G
major (whose F♯ the song never plays), and the strongest tonic cue there is —
the song starts and ends on G — is outvoted by membership. Add mixolydian and G
wins outright, because now the song is diatonic *and* begins and ends at home.
The mode is then projected away (`harmony.MODE_PROJECTION`) and the tonic, which
is the part that mattered, survives.

**And why only four of them.** The same argument that adds mixolydian caps the
list well short of seven — see `KEY_MODES`. Modes of one collection are
indistinguishable by membership, so every mode admitted that does not really
occur as a key is a pure liability.

What the modes do **not** buy is the consensus tie-break, and it is worth saying
so rather than implying otherwise: all seven modes of one collection share a
pitch-class set, so `harmony.diatonic_fit` returns the same number whichever of
them is chosen. The tie-break reads the collection; only the spelling reads the
tonic.

Modulation is deliberately not **acted on**. The container carries one key for
the whole song, so a detected key change could not be expressed on the wire; the
duration-weighted global answer is the honest one for a single-key container.

What changed in the 2026-08-18 audit is that it is now *detected and reported*
(`track`). A truck-driver modulation — the last chorus up a semitone or a tone —
is one of the commonest structures in this repertoire, and when it happens the
global key is wrong for the end of the song: the chart mixes two transpositions,
which the player sees as impossible alterations, and every diatonic tie-break in
`consensus`, `canon`, `vocabulary` and `keyaudit` is being taken against the
wrong key for that stretch. None of that is visible in a single `tonic`/`mode`
pair. `TheoryReport.modulations` is where it becomes visible, and it costs one
extra pass over the spans.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from statistics import median

from ..chords import (
    AUGMENTED,
    DIMINISHED,
    DIMINISHED7,
    DOMINANT7,
    HALF_DIM7,
    MAJOR,
    MAJOR7,
    MINOR,
    MINOR7,
    SUS2,
    SUS4,
    prefers_flats_for,
    spell,
)
from . import harmony
from .types import GridSpan

# A root that belongs to the key but wears an unexpected quality is still strong
# evidence for the key — borrowed chords are normal, wrong roots are not.
_QUALITY_MATCH = 1.0
_ROOT_ONLY = 0.45
# ...but "unexpected" covers two very different disagreements, and scoring them
# alike is what lets a key survive being contradicted about its own thirds. A
# `G7` where the mode wants `G` is a seventh the recognizer heard, and the triad
# under it still says the key is right; a `Gm` where the mode wants `G` says the
# opposite, and a song full of those is not in this key at all. The first keeps
# `_ROOT_ONLY`; the second gets this, which is the term the 2026-08-18 audit
# (F10) asked for.
#
# Swept over the eleven-song chart corpus at 0.45 (the old flat value) / 0.30 /
# 0.20 / 0.10 / 0.00, counting correct tonics: see `ANALYSIS-AUDIT.md`. The value
# below is the middle of the band that scores highest.
_WRONG_THIRD = 0.20
# The tonic chord opening or closing a song is the single most reliable cue there
# is, and it is what separates a key from its relative major/minor — those two
# share every diatonic chord, so without this they score identically.
_TONIC_ENDPOINT_BONUS = 0.35

# How much a key is rewarded for its tonic chord **actually sounding**, as a share
# of the song's duration spent on that root.
#
# This is the term that decides a key against its relative, and it had to be added
# because the endpoint bonus above — written for exactly that job — is not big
# enough to do it. The two share a diatonic collection, so membership scores them
# identically and the whole decision falls to one capped bonus worth under a
# hundredth of the total. Measured on the ten-song chart corpus, three of ten keys
# came out as the relative of the right one: Creep as E minor, Don't Stop
# Believin' as C# minor, Wonderwall as A major. Creep does not contain a single E
# chord.
#
# That last fact is the rule in one line: **a song in E minor plays an E minor
# chord.** A key whose tonic never sounds is not a key the song is in, however
# well its scale fits, and a relative pair is precisely the case where the scale
# cannot say anything else.
#
# The **cap** is doing more of the work than the weight, and that is the point:
# the claim being made is "the tonic sounds a normal amount", not "the tonic
# sounds a lot". A share of 0.15 is about what a four-chord song spends on any one
# of its chords, so a key clears this term completely or fails it, and a song that
# parks on one chord gains nothing extra for it.
#
# Swept over the chart corpus (weight × cap, counting correct tonics): every
# weight from 0.15 to 0.6 at a cap of 0.15 scores 9/10, and *every* setting with a
# looser cap scores 8 or fewer — an uncapped term hands the key to whichever chord
# the song plays most, which on Viva La Vida is the vi and on Someone Like You is
# the vi as well. 0.25 sits in the middle of the flat band, so the value is not
# perched on the edge of the measurement.
#
# The one it does not fix is Creep, where G major and C major score *identically*
# on every term here — same collection, both tonics sounded (`Cm` shares C's
# root), and the endpoint bonus split one each. What decides that song is that
# every one of its eleven passes begins on G, which is positional evidence about
# the *form* and not something a duration-weighted bag of chords can hold.
_TONIC_MASS_WEIGHT = 0.25
_TONIC_MASS_CAP = 0.15

_UNRESOLVED = {AUGMENTED}   # belongs to no diatonic degree; ignored rather than penalised

# The modes worth *scoring a key against*, which is a much shorter list than the
# modes worth being able to name (`harmony.SCALES` still defines all seven, and
# `diatonic_fit` still reads any of them).
#
# The reason is a property of the problem rather than a taste: the seven modes of
# one collection contain **exactly the same notes**, so scale membership cannot
# tell them apart at all, and the tonic gets decided by whatever tiny residue is
# left over. Measured on the real corpus, "A Hard Day's Night" came out F lydian
# ahead of G mixolydian by 0.009 — six candidates within 0.02, all of them the
# same seven notes, the winner picked by noise.
#
# Every mode admitted here is therefore a mode that has to earn its place by
# occurring as a **key** in this repertoire. Ionian, aeolian, mixolydian and
# dorian do, constantly. Lydian, phrygian and locrian essentially never do — a
# lydian *passage* is common, a lydian song is not — so including them added
# three more ways to pick a wrong tonic and no way to be right that the other
# four did not already cover. Dropping them takes the corpus from 8/9 correct
# tonics to 9/9.
KEY_MODES = ("ionian", "aeolian", "mixolydian", "dorian")

# Tie-break prior, applied only between modes that scored equally. Ordered by how
# often the mode actually turns up, so a two-chord song that fits all four
# equally well is called the likeliest one rather than whichever sorted first.
# Small enough that it can never outweigh real evidence.
_MODE_PRIOR = {"ionian": 0.006, "aeolian": 0.005, "mixolydian": 0.004, "dorian": 0.003}

_TRIAD_QUALITY = {(4, 7): MAJOR, (3, 7): MINOR, (3, 6): DIMINISHED, (4, 8): AUGMENTED}
_SEVENTH_QUALITY = {
    (MAJOR, 11): MAJOR7, (MAJOR, 10): DOMINANT7,
    (MINOR, 10): MINOR7, (MINOR, 11): MINOR7,
    (DIMINISHED, 10): HALF_DIM7, (DIMINISHED, 9): DIMINISHED7,
}


@dataclass(frozen=True)
class DetectedKey:
    """The key, as the wire needs it and as the model knows it.

    ``tonic``/``mode`` are the projected pair the payload carries. ``tonic_pc``
    and ``scale`` are the internal answer — the scale is one of
    `harmony.SCALES`, and it is what a diagnostic or a roman-numeral read-out
    should use.
    """

    tonic: str
    mode: str
    confidence: float
    tonic_pc: int = 0
    scale: str = "ionian"

    @property
    def is_modal(self) -> bool:
        """True when the song is in a mode the container cannot name — worth
        reporting, because "G major" is what the player will be told about a
        song that is really G mixolydian."""
        return self.scale not in ("ionian", "aeolian")


def degrees(mode: str) -> dict[int, frozenset[str]]:
    """The mode's degree → qualities table, for callers outside this module.

    Public because `keyaudit.py` has to ask the same question this module asks —
    "does this chord belong to the key" — and building a second table to answer
    it is how two tables drift apart (F13's lesson, one module over). One rule,
    one table, both readers."""
    return _degrees(mode)


@lru_cache(maxsize=None)
def _degrees(mode: str) -> dict[int, frozenset[str]]:
    """Scale degree (semitones above the tonic) → the qualities that belong there.

    Built from the scale rather than hand-written, so all seven modes are
    described by one rule instead of seven tables that could drift apart. Each
    degree carries its diatonic triad, its diatonic seventh, and — for major and
    minor degrees — the two suspensions, which are idiomatic anywhere and carry
    no third to contradict the mode.
    """
    scale = list(harmony.SCALES.get(mode, harmony.IONIAN))
    table: dict[int, set[str]] = {}
    for index, root in enumerate(scale):
        third = (scale[(index + 2) % 7] - root) % 12
        fifth = (scale[(index + 4) % 7] - root) % 12
        seventh = (scale[(index + 6) % 7] - root) % 12
        triad = _TRIAD_QUALITY.get((third, fifth))
        if triad is None:
            continue
        qualities = {triad}
        seventh_quality = _SEVENTH_QUALITY.get((triad, seventh))
        if seventh_quality:
            qualities.add(seventh_quality)
        if triad in (MAJOR, MINOR):
            qualities |= {SUS2, SUS4}
        table[root] = qualities

    if mode == "aeolian":
        # Harmonic minor, which is not a mode but is everywhere in this
        # repertoire: the major V and the leading-tone diminished. A minor-key
        # scorer without these calls half its songs' dominants foreign.
        table.setdefault(7, set()).update({MAJOR, DOMINANT7})
        table.setdefault(11, set()).update({DIMINISHED, DIMINISHED7})
    return {degree: frozenset(qualities) for degree, qualities in table.items()}


def detect_key(spans: list[GridSpan]) -> DetectedKey:
    """Best-fitting key for a chord track, with a confidence in 0…1.

    Three things are scored: how well the song's chords sit in the key
    (`_score`), how much of the song is spent **on the tonic chord**
    (`_TONIC_MASS_WEIGHT` — the term that decides a key against its relative,
    since the two share a scale and nothing else can), and a small prior over
    modes to break exact ties deterministically.

    Confidence is the **margin** over the runner-up *in a different projected
    key*, not over the runner-up outright: the seven modes of one collection are
    near-synonyms here, and two of them disagreeing about which is G mixolydian
    and which is D dorian says nothing about how sure we are of the spelling.
    A near-tie between genuinely different answers reports low confidence, which
    is the truthful thing to do.
    """
    if not spans:
        return DetectedKey("C", "major", 0.0, tonic_pc=0, scale="ionian")

    total = sum(s.length_beats for s in spans) or 1
    # The endpoint bonus's cap, computed once. It is a property of the chord track,
    # not of the candidate key, and `_score` runs 48 times — so recomputing a median
    # over every span inside it did the same sort 48 times per key detection, and
    # `detect_key` itself is called up to four times per analysis.
    typical = median(s.length_beats for s in spans)
    # How much of the song sounds on each root, for the tonic-mass term below.
    # Computed once here rather than inside `_score`, which runs 48 times.
    #
    # Every span is capped at the median length, for the same reason the endpoint
    # bonus is: this has to mean "the tonic sounds a normal amount", and raw
    # duration lets one dragged chord answer the question by itself. A track
    # ending on a 400-beat A minor is not in A minor because of that chord, and
    # uncapped mass says it is — which is the failure
    # `test_the_tonic_bonus_is_not_swamped_by_a_long_final_chord` was already
    # guarding against for the other positional term.
    mass: dict[int, float] = {}
    heard_total = 0.0
    for span in spans:
        weight = min(span.length_beats, typical)
        mass[span.root_pc % 12] = mass.get(span.root_pc % 12, 0.0) + weight
        heard_total += weight
    heard_total = heard_total or 1.0
    scored: list[tuple[float, str, int]] = []
    for tonic_pc in range(12):
        for mode in KEY_MODES:
            fit = _score(spans, tonic_pc, _degrees(mode), typical) / total
            heard = min(mass.get(tonic_pc, 0.0) / heard_total, _TONIC_MASS_CAP)
            score = fit + heard * _TONIC_MASS_WEIGHT + _MODE_PRIOR[mode]
            scored.append((score, mode, tonic_pc))
    scored.sort(key=lambda item: (-item[0], item[2], item[1]))

    best_score, best_mode, best_pc = scored[0]
    projected = harmony.MODE_PROJECTION[best_mode]

    runner_up = 0.0
    for score, mode, tonic_pc in scored[1:]:
        if (tonic_pc, harmony.MODE_PROJECTION[mode]) != (best_pc, projected):
            runner_up = score
            break

    # Scale the margin so a decisive win (≥ 0.25 clear) reads as ~1.0; the raw
    # margin between two keys is small even when the answer is obvious.
    confidence = max(0.0, min(1.0, (best_score - runner_up) * 4.0))
    flats = _prefers_flat_spelling(best_pc, projected)
    return DetectedKey(
        tonic=spell(best_pc, flats), mode=projected, confidence=round(confidence, 3),
        tonic_pc=best_pc, scale=best_mode,
    )


def _score(spans: list[GridSpan], tonic_pc: int, degrees: dict[int, frozenset[str]],
           typical: float) -> float:
    """This key's score for the chord track. `typical` is the median span length,
    passed in because it does not depend on the key being scored (see
    `detect_key`)."""
    total = 0.0
    for span in spans:
        degree = (span.root_pc - tonic_pc) % 12
        qualities = degrees.get(degree)
        if qualities is None:
            continue
        if span.quality in qualities:
            total += span.length_beats * _QUALITY_MATCH
        elif span.quality not in _UNRESOLVED:
            mine = harmony.triad(span.quality)
            agrees = any(harmony.triad(quality) == mine for quality in qualities)
            total += span.length_beats * (_ROOT_ONLY if agrees else _WRONG_THIRD)

    # The endpoint bonus is a claim about *position* — the song comes to rest on
    # its tonic — so it is weighted like one typical chord rather than by however
    # long that particular chord was held. Uncapped, a song ending on a chord
    # held for five bars gives its final root a bonus several times the size of
    # any evidence the rest of the song can offer, and the key becomes a fact
    # about the outro. Capping at the median span is what makes it "the last
    # chord, counted once".
    for endpoint in (spans[0], spans[-1]):
        if endpoint.root_pc % 12 == tonic_pc:
            total += min(endpoint.length_beats, typical) * _TONIC_ENDPOINT_BONUS
    return total


def _prefers_flat_spelling(tonic_pc: int, mode: str) -> bool:
    """Which spelling of the *tonic itself* to emit.

    Delegated to `chords.prefers_flats_for`, which is the same key-signature
    table `chords.prefers_flats` reads for a tonic that is already spelled. It
    used to be a second copy of the twelve numbers here, and two tables that have
    to agree and are not the same table only ever agree until one of them is
    edited.
    """
    return prefers_flats_for(tonic_pc, mode)


# --- modulation -------------------------------------------------------------

# How much song each windowed reading is taken over. Thirty-two bars at four
# beats: long enough that a window contains a section rather than a phrase (a
# key detected over eight bars is a statement about one progression), short
# enough that a final-chorus modulation occupies more than one of them.
WINDOW_BEATS = 128.0

# How many consecutive windows must agree on the new key before a change is
# reported. Two, which is the smallest number that is evidence: one window
# disagreeing with its neighbours is a bridge that tonicizes something, or a
# window that happened to land across a section boundary, and neither is a
# modulation.
MODULATION_WINDOWS = 2


def collection(key: DetectedKey) -> frozenset[int]:
    """The key's seven notes. What a modulation actually moves.

    The plain scale, without aeolian's borrowed leading tone — that note is a
    fact about how minor keys are harmonized, not about which collection the
    music is in, and including it would make a minor key differ from its relative
    major by one pitch class and read as a modulation.
    """
    degrees = harmony.SCALES.get(key.scale, harmony.IONIAN)
    return frozenset((key.tonic_pc + degree) % 12 for degree in degrees)


def track(spans: list[GridSpan], *, window_beats: float = WINDOW_BEATS
          ) -> tuple[tuple[float, DetectedKey], ...]:
    """Where the song changes key, as (beat, key) — the first entry is where it
    starts.

    Detection only: nothing downstream acts on this, by design (see the module
    docstring). A single entry means one key for the whole song, which is the
    ordinary case and the one the container is built for.

    The window reading is the same `detect_key` the song gets, over the spans
    that start inside the window — so a change is reported only when a stretch of
    the song, read on its own terms, comes out somewhere else. Runs shorter than
    `MODULATION_WINDOWS` are folded back into their neighbour rather than
    reported, because a bridge that tonicizes the relative minor for eight bars is
    not a modulation and a chart that said so would be crying wolf on half the
    repertoire.
    """
    if not spans:
        return ()
    end = max(span.end_beat for span in spans)
    if end <= window_beats:
        return ((0.0, detect_key(spans)),)

    readings: list[tuple[float, DetectedKey]] = []
    start = 0.0
    while start < end:
        window = [s for s in spans if start <= s.start_beat < start + window_beats]
        if window:
            readings.append((start, detect_key(window)))
        start += window_beats
    if not readings:
        return ((0.0, detect_key(spans)),)

    # Runs of windows that agree on the **collection**, not on the tonic. A
    # modulation moves the notes; B major and G# minor are the same seven notes
    # and disagreeing about which is home is what this module's whole first half
    # is about, not a key change. Reading it any other way makes I'm Yours report
    # "bar 64: B major -> G# minor", which is the relative-key wobble the tonic
    # terms already fight over and nothing a listener would call a modulation.
    runs: list[list[tuple[float, DetectedKey]]] = [[readings[0]]]
    for at, key in readings[1:]:
        if collection(key) == collection(runs[-1][-1][1]):
            runs[-1].append((at, key))
        else:
            runs.append([(at, key)])

    out: list[tuple[float, DetectedKey]] = []
    for run in runs:
        if out and len(run) < MODULATION_WINDOWS:
            continue                      # too short to be a key change
        if out and collection(run[0][1]) == collection(out[-1][1]):
            continue                      # back where it was — one run, interrupted
        out.append(run[0])
    if not out:
        out.append(readings[0])
    out[0] = (0.0, out[0][1])
    return tuple(out)


def describe_modulations(changes: tuple[tuple[float, DetectedKey], ...],
                         bar_beats: float = 4.0) -> tuple[str, ...]:
    """The changes as lines a person can read: `"bar 96: E major -> F major"`."""
    if len(changes) < 2:
        return ()
    out = []
    for (_, previous), (at, key) in zip(changes, changes[1:]):
        bar = int(at // bar_beats) if bar_beats else 0
        out.append(f"bar {bar}: {previous.tonic} {previous.mode} -> {key.tonic} {key.mode}")
    return tuple(out)
