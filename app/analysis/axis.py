"""The song's beat axis — one origin, owned in one place.

Everything downstream of the beat tracker addresses music in **song beats**: the
chart quantizes chords to them, `structure` slices bars out of them, and §13.2's
anchors map them back to video milliseconds. That only works if all three agree
about where beat 0 is and how wide a bar is, and for a long time they did not:

- `postprocess.quantize` indexed into `BeatGrid.beats_ms`, so the chart's bar 0
  began at the first **beat**;
- `sync.anchors_for` put `songBeat 0` at the first **downbeat**.

Nothing reconciled them, so every song whose first downbeat was not its first
beat — a pickup, a count-in, an intro fill, i.e. most real recordings — was laid
onto a chart phase-shifted against its own recording. Measured on the real
corpus with ground truth as both engines, that cost 23 points of accuracy before
any engine had made a mistake, and neither `lint` nor `lint_sync` could see it:
both check the song against *itself*, and a uniformly shifted chart is perfectly
self-consistent.

So the mapping stops being implicit. `BeatAxis.times_ms[i]` is the video time of
song beat `i`, **by construction**:

    beat 0        IS the first downbeat
    bar k         IS beats [k·bar_beats, (k+1)·bar_beats)
    bar k starts  at times_ms[k · bar_beats], which IS downbeat k

The three consumers now share one object instead of three assumptions, and the
invariant is a property of the constructor rather than something a test has to
keep rediscovering.

**Irregular bars.** A tracker on real music reports bars that are not all the
same length — Here Comes The Sun has 11/8 and 15/8 bars inside a 4/4 song, and
that alone cost it 47 points even after the phase was fixed. The container has
no way to say so (§13.2 requires one uniform grid, because the client derives
bar index as `floor(beat / barBeats)` on a single song-level meter). So a bar
whose beat count disagrees with the meter is **resampled**: its `bar_beats`
chart beats are spread evenly across the bar's real duration. The bar still
starts and ends on the real downbeats — which is what the anchors address, and
what the player sees — and only the beat subdivisions inside it are nudged.
Error stays bounded within one bar instead of accumulating over the song.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass

from ..payload import bar_beats as parse_bar_beats
from .types import BeatGrid


@dataclass(frozen=True)
class BeatAxis:
    """Song beat index → video millisecond, and the bar grid laid over it.

    ``times_ms`` has one entry per beat plus a final boundary, so a song of *n*
    bars has ``n · bar_beats + 1`` entries and ``times_ms[-1]`` is the end of the
    last bar.
    """

    times_ms: list[int]
    bar_beats: int

    @property
    def bar_count(self) -> int:
        return max(0, (len(self.times_ms) - 1) // self.bar_beats)

    @property
    def total_beats(self) -> int:
        return self.bar_count * self.bar_beats

    @property
    def downbeats_ms(self) -> list[int]:
        """The bar boundaries, one per bar plus the closing one.

        These are the sidecar's anchors, and because they are read off the same
        list the chart is quantized against, ``anchor[k]`` is the time of chart
        beat ``k · bar_beats`` — no longer a coincidence.
        """
        return [self.times_ms[i * self.bar_beats]
                for i in range(self.bar_count + 1)
                if i * self.bar_beats < len(self.times_ms)]

    @property
    def is_usable(self) -> bool:
        """At least two bars. One bar cannot express a tempo for the anchors to
        interpolate along, and §13.2 needs bars."""
        return self.bar_count >= 2 and len(self.times_ms) >= self.bar_beats + 1

    def beat_at(self, t_ms: float) -> int:
        """Index of the beat nearest ``t_ms``, clamped to the axis.

        Clamping rather than extrapolating: a chord starting before the first
        downbeat belongs to beat 0, and one running past the last belongs to the
        last. Inventing beats outside the axis would put chords on a timeline the
        anchors do not cover.
        """
        times = self.times_ms
        if not times:
            return 0
        index = bisect_left(times, t_ms)
        if index <= 0:
            return 0
        if index >= len(times):
            return len(times) - 1
        before, after = times[index - 1], times[index]
        return index if (after - t_ms) < (t_ms - before) else index - 1

    def position_at(self, t_ms: float) -> float:
        """Fractional beat position, for folding onsets onto the bar (§14)."""
        return position_in(self.times_ms, t_ms)


def position_in(times_ms: list[int], t_ms: float) -> float:
    """Where `t_ms` falls on a beat list, interpolating between entries.

    Fractional by necessity: an onset on the "&" sits halfway between beat 4 and
    beat 5, and §14's whole extraction is about where inside a beat things land.
    Linear inside a beat, and extrapolated along the nearest interval off either
    end, so an onset just before beat 0 reads as slightly negative rather than
    snapping to 0 and inventing a downbeat stroke.

    A free function rather than only a method because `strumming` needs it over a
    plain list too — but there is exactly one implementation, which is the point
    of this module.
    """
    if len(times_ms) < 2:
        return 0.0
    index = bisect_left(times_ms, t_ms)
    if index <= 0:
        span = times_ms[1] - times_ms[0]
        return (t_ms - times_ms[0]) / span if span > 0 else 0.0
    if index >= len(times_ms):
        span = times_ms[-1] - times_ms[-2]
        return (len(times_ms) - 1) + ((t_ms - times_ms[-1]) / span if span > 0 else 0.0)
    left, right = times_ms[index - 1], times_ms[index]
    span = right - left
    return (index - 1) + ((t_ms - left) / span if span > 0 else 0.0)


def build_axis(grid: BeatGrid) -> BeatAxis | None:
    """A `BeatGrid` as the tracker reported it → the axis the song is built on.

    Returns None when there is not enough grid to lay bars over — fewer than two
    downbeats, or a meter that isn't a whole number of beats. The caller degrades
    honestly rather than guessing (§13.3).
    """
    beats = parse_bar_beats(grid.time_signature)
    if beats is None:
        return None
    bar_beats = int(round(beats))
    if bar_beats < 1 or abs(beats - bar_beats) > 1e-6:
        # A meter whose bar isn't a whole number of quarter-note beats (7/8) has
        # no integer beat grid for the chart to sit on.
        return None

    downbeats = sorted(set(int(t) for t in grid.downbeats_ms))
    if len(downbeats) < 2:
        return None
    tracked = sorted(set(int(t) for t in grid.beats_ms))

    times: list[int] = []
    for start, end in zip(downbeats, downbeats[1:]):
        times.extend(_bar_beats_between(tracked, start, end, bar_beats))
    times.append(downbeats[-1])

    return BeatAxis(times_ms=times, bar_beats=bar_beats)


def _bar_beats_between(tracked: list[int], start: int, end: int, bar_beats: int) -> list[int]:
    """The `bar_beats` beat times of one bar, starting at `start` (exclusive of
    `end`, which is the next bar's downbeat).

    Uses the tracker's own beats when it found exactly the right number of them —
    that preserves any expressive unevenness inside the bar. Otherwise the bar is
    resampled evenly, which is the concession the container's single-meter
    requirement forces (see the module docstring). Either way the bar begins at
    `start`, so the downbeats — the only times the anchors publish — are always
    the tracker's own.
    """
    if end <= start:
        return [start]
    inner = [t for t in tracked if start < t < end]
    if len(inner) == bar_beats - 1:
        return [start] + inner
    span = end - start
    return [start + int(round(span * i / bar_beats)) for i in range(bar_beats)]
