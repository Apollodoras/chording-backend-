"""§20.2a — repair the downbeat sequence before anything is built on it.

`beat_this` emits `(beats, downbeats)` and, until this module existed, nothing
between the tracker and the sidecar ever asked whether a bar was about as long
as the other bars. `meter.reconcile` corrected the downbeat *phase* and the
tempo *octave*; `axis.build_axis` then defined **every consecutive pair of
downbeats to be exactly one bar** and resampled whatever it found in between.
That resampling is a deliberate concession to §13.2's single-meter requirement
and it is right for genuinely irregular bars — Here Comes The Sun has 11/8 and
15/8 bars inside a 4/4 song — but the same path silently absorbed *tracker
error*, which is one to two orders of magnitude more common.

The cost was measured, and it is not subtle. On the four stored catalog songs
between 8% and 42% of bars disagreed with the song's own modal bar, in both
directions: spurious downbeats a beat into a real bar (a half-bar and then a
short one) and dropped downbeats (a bar of double length). Both reach the
player as the same two complaints — "beat-map anomalies" and "it changes tempo
mid song" — because the client reads its cursor speed off the anchors, and
anchors a half-bar apart say the song doubled. Worse, a bar added or removed
shifts every later bar against the music, and `form._layout` searches for **one
global phase**: a phase that changes mid-song cannot be fitted, so block
similarity collapses and the song is emitted as seventeen sections that each
repeat once. One corrupted downbeat in thirty-two bars is enough to turn a
single eight-times-repeating section into four unrelated ones.

**The framing that makes this tractable: trust the tracker's beats, not its
downbeats.** The beat sequence is reliable — regularity is near 1.0 on every
song measured — and every anomaly observed lands on a real beat (the gaps are
integer numbers of beats). So the repair never moves a time. It only chooses
**which of the tracker's own beats are bar starts**, which is why it composes
with everything downstream: `axis` still publishes the tracker's timings, and
the phase vote that runs immediately after still rotates a grid made of the
same beats.

**Bounded and reported, never silent.** Real irregular bars exist and this must
not flatten them, so: a gap is only repaired when it is close to a whole number
of modal bars, a song whose bars disagree with their own mode more than
`IRREGULAR_CEILING` of the time is marked `unreliable` rather than forced into a
meter it may not be in, and everything the repair did rides out on
`TheoryReport` — the same provenance rule `consensus` and `vocabulary` follow.
"""

from __future__ import annotations

import logging
import statistics
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass

from .types import BeatGrid

log = logging.getLogger("chords.downbeats")

# A downbeat arriving sooner than this fraction of a bar after the last accepted
# one is spurious. Two thirds, roughly: on a 4/4 song that drops downbeats at 1,
# 2 and 2.5 beats — the observed failure — and keeps a bar that a tracker
# genuinely read as 3/4 inside a 4/4 song, which is a musical event and not this
# module's business.
SPURIOUS_RATIO = 0.65

# How far a multi-bar gap may sit from a whole number of bars and still be read
# as "n−1 downbeats went missing" rather than as one long irregular bar.
GAP_TOLERANCE = 0.25

# Above this share of bars disagreeing with the song's own modal bar, the song
# may genuinely not be in that meter — Anti Nowhere League's So What sits at
# 42% — and forcing it would be exactly the confident mistake §20 exists to
# avoid. The repair still runs (a self-consistent grid is worth having either
# way) but the song is flagged, and §13.3 withholds the sidecar rather than
# shipping a video sync nobody checked.
IRREGULAR_CEILING = 0.35

# Fewer bars than this and there is no mode worth trusting: three gaps can be
# 4, 4, 2 with the 2 being the song, and "repairing" it would invent a bar.
MIN_BARS = 4

# A downbeat is expected to *be* one of the tracker's beats. It is allowed to
# miss by a fraction of a beat (rounding through milliseconds, mostly), but a
# grid whose downbeats mostly do not sit on its beats is not one this repair
# understands, and it declines rather than guessing.
ON_BEAT_TOLERANCE = 0.25
ON_BEAT_SHARE = 0.9


@dataclass(frozen=True)
class DownbeatReport:
    """What the repair found, and what it did about it.

    Carried on `Meter` and published on the sidecar. `total_bars` and
    `irregular_bars` are counted **after** the repair, so a healthy song reports
    zero and a song still holding real 3/4 bars inside its 4/4 reports them —
    which is the honest answer, since those bars are exactly what `axis` is
    about to resample.
    """

    # Modal bar length, in beats, as measured off the tracker's own grid. 0 when
    # the repair declined to run.
    bar_beats: int = 0
    dropped: int = 0
    inserted: int = 0
    irregular_bars: int = 0
    total_bars: int = 0
    # The share of bars that disagreed with the mode **before** the repair
    # exceeded `IRREGULAR_CEILING`. Not "the repair failed" — it is "this grid
    # was too far gone for the result to be trusted without a second opinion".
    unreliable: bool = False
    ran: bool = False

    @property
    def touched(self) -> bool:
        return bool(self.dropped or self.inserted)

    @property
    def irregular_ratio(self) -> float:
        return self.irregular_bars / self.total_bars if self.total_bars else 0.0


def repair(grid: BeatGrid, *, bar_beats: int | None = None) -> tuple[BeatGrid, DownbeatReport]:
    """A tracker's grid → the same beats, with the bar starts chosen sensibly.

    Returns the grid **unchanged** whenever the evidence does not clearly say
    otherwise: too few bars to hold a mode, downbeats that do not sit on the
    beats, or a repair that would leave too little grid to build an axis on.

    `bar_beats` is the meter's own answer and is used only to break a tie
    between two equally common bar lengths — the mode is measured from the grid,
    because the grid is what the tracker actually produced.
    """
    beats = sorted({int(t) for t in grid.beats_ms})
    downbeats = sorted({int(t) for t in grid.downbeats_ms})
    if len(beats) < 2 or len(downbeats) < MIN_BARS + 1:
        return grid, DownbeatReport()

    indices = _beat_indices(beats, downbeats)
    if indices is None or len(indices) < MIN_BARS + 1:
        return grid, DownbeatReport()

    gaps = [b - a for a, b in zip(indices, indices[1:])]
    mode = _modal_bar(gaps, indices, hint=bar_beats)
    if mode is None:
        return grid, DownbeatReport()

    before = sum(1 for g in gaps if g != mode) / len(gaps)

    kept, dropped, inserted = _walk(indices, mode)
    if len(kept) < 3:
        # Nothing left to build bars on. The tracker's answer, wrong as it may
        # be, is still more grid than this.
        return grid, DownbeatReport(bar_beats=mode, unreliable=before > IRREGULAR_CEILING)

    after = [b - a for a, b in zip(kept, kept[1:])]
    report = DownbeatReport(
        bar_beats=mode,
        dropped=dropped,
        inserted=inserted,
        irregular_bars=sum(1 for g in after if g != mode),
        total_bars=len(after),
        unreliable=before > IRREGULAR_CEILING,
        ran=True,
    )

    if report.touched:
        log.info("downbeats repaired: %d spurious dropped, %d missing inserted; "
                 "bars disagreeing with the %d-beat mode %.0f%% → %.0f%%",
                 dropped, inserted, mode, before * 100, report.irregular_ratio * 100)
    if report.unreliable:
        log.warning("%.0f%% of bars disagreed with the %d-beat mode before repair — "
                    "the song may not be in that meter", before * 100, mode)

    repaired = BeatGrid(
        beats_ms=grid.beats_ms,
        downbeats_ms=[beats[i] for i in kept],
        bpm=grid.bpm,
        confidence=grid.confidence,
        time_signature=grid.time_signature,
    )
    return repaired, report


def modal_bar_beats(beats_ms: list[int], downbeats_ms: list[int], *,
                    hint: int | None = None) -> int | None:
    """The song's bar length in beats, for a caller holding only the two lists.

    Exported for `beat_this_tracker._meter`, which has to answer the same
    question one stage earlier and used to answer it with a plain
    `statistics.mode`. That disagrees with this module on exactly the songs that
    matter: a grid with a spurious downbeat in half its bars has more 3-beat gaps
    in it than 4-beat ones, so the tracker would report 3/4, the repair would
    conclude 4, and `axis` would then lay three-beat bars over four-beat music.
    One estimator, one answer.
    """
    beats = sorted({int(t) for t in beats_ms})
    downbeats = sorted({int(t) for t in downbeats_ms})
    if len(beats) < 2 or len(downbeats) < 3:
        return None
    indices = _beat_indices(beats, downbeats)
    if indices is None or len(indices) < 3:
        return None
    gaps = [b - a for a, b in zip(indices, indices[1:])]
    return _modal_bar(gaps, indices, hint=hint)


def _beat_indices(beats: list[int], downbeats: list[int]) -> list[int] | None:
    """Each downbeat as an index into the beat list, or None if they don't fit.

    The repair's whole premise is that the downbeats are a *selection* of the
    beats and the selection is what went wrong. A grid where that is not true —
    a tracker whose two outputs were computed independently — is one this module
    has no business editing, so it says so instead of snapping times together.
    """
    intervals = [b - a for a, b in zip(beats, beats[1:]) if b > a]
    if not intervals:
        return None
    tolerance = statistics.median(intervals) * ON_BEAT_TOLERANCE

    found: list[int] = []
    for t in downbeats:
        index = bisect_left(beats, t)
        candidates = [i for i in (index - 1, index) if 0 <= i < len(beats)]
        nearest = min(candidates, key=lambda i: abs(beats[i] - t))
        if abs(beats[nearest] - t) <= tolerance:
            found.append(nearest)

    if len(found) < len(downbeats) * ON_BEAT_SHARE:
        log.info("downbeat repair declined: only %d of %d downbeats sit on the beat grid",
                 len(found), len(downbeats))
        return None
    return sorted(set(found))


def _modal_bar(gaps: list[int], indices: list[int], *, hint: int | None) -> int | None:
    """The song's own bar length, in beats — **mode, not median**.

    A heavy tail of half-bars drags a median (So What's would read 3), and the
    whole point of the estimate is to be unmoved by exactly the anomalies it is
    about to remove.

    But raw frequency is not enough either, because the commonest failure
    manufactures its own runner-up: an extra downbeat one beat into a bar turns
    one `4` into a `1` and a `3`, so a song corrupted in half its bars can have
    more 3s in it than 4s while still being, unmistakably, in four.

    So the candidates are scored by **how much of the song each one already
    explains** — bars of that length, times that length, in beats. The `3`s
    above account for three beats each and the `4`s for four, and four wins on a
    song that is in four however many spurious downbeats it carries. Ties break
    toward the meter's own answer and then toward the shorter bar.

    **Not by how tidy the repair comes out**, which was the first thing tried
    and is a trap: the walk can *insert* downbeats, so a sub-multiple always
    scores perfectly — halve every bar and every bar agrees with the half. It
    picked a 2-beat bar for Sweet Home Alabama, doubled the song's bar count,
    and was caught only by the ceiling below. A candidate has to be judged on
    the downbeats the tracker actually produced, not on what it would look like
    after this module had rewritten them.
    """
    counts = Counter(g for g in gaps if g >= 2)
    if not counts:
        return None
    candidates = [g for g, _ in counts.most_common(4)]
    if hint and hint >= 2 and hint not in candidates:
        candidates.append(hint)

    def explains(candidate: int) -> tuple[int, int, int]:
        return (-counts[candidate] * candidate,
                0 if candidate == hint else 1,
                candidate)

    return min(candidates, key=explains)


def _walk(indices: list[int], mode: int) -> tuple[list[int], int, int]:
    """Choose the bar starts, in beat-index space.

    Each gap is measured from the **last accepted** downbeat rather than from
    the raw predecessor, and that is what makes the common failure fall out in
    one pass: a spurious downbeat one beat into a real bar shows up as a `1`
    followed by a `3`, and once the `1` is dropped the `3` is measured from the
    bar's real start and reads as a clean `4`.
    """
    kept = [indices[0]]
    dropped = inserted = 0

    for index in indices[1:]:
        gap = index - kept[-1]
        if gap <= 0:
            continue
        if gap < SPURIOUS_RATIO * mode:
            dropped += 1
            continue
        bars = int(round(gap / mode))
        if bars >= 2 and abs(gap / bars - mode) <= GAP_TOLERANCE * mode:
            # n−1 downbeats went missing. They are put back on the tracker's own
            # intervening beats, evenly, so a dropped downbeat inside a bar of
            # rubato lands where the beats say rather than where arithmetic on
            # the tempo would.
            start, step = kept[-1], gap / bars
            for step_index in range(1, bars):
                kept.append(start + int(round(step_index * step)))
            inserted += bars - 1
        kept.append(index)

    return kept, dropped, inserted
