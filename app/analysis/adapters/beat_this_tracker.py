"""Beat This! — CPJKU, ISMIR 2024. https://github.com/CPJKU/beat_this

A joint beat *and downbeat* model, which is the distinction that matters here:
§13.2's anchors are downbeats, so a tracker that finds the pulse and guesses the
bar is solving three quarters of the problem. librosa needs a hand-written
heuristic to place the "one" (see `librosa_beats.py`); this predicts it directly.

Run with `dbn=False`. The alternative postprocessor is madmom's DBN, which drags
in a package that does not build on Python 3.11 without a fork — a real cost for
a refinement, and the minimal postprocessor is what the paper reports anyway.

The checkpoint (`final0`) is fetched from the project's release on first use and
cached by torch. That is fine on a laptop and **wrong in a worker**: a container
that downloads model weights on its first request has turned a cold start into a
network dependency. The worker image should bake the cache in — see the note in
`modal_app.py`.
"""

from __future__ import annotations

import logging
import statistics

from ..downbeats import modal_bar_beats
from ..types import PCM, BeatGrid

log = logging.getLogger("chords.engines.beat_this")

_CHECKPOINT = "final0"


class BeatThisTracker:
    """`BeatTracker` — predicted beats and downbeats, meter read off the bars."""

    name = "beat_this"
    version = "ismir24-final0"

    def __init__(self) -> None:
        self._tracker = None

    def _load(self):
        if self._tracker is None:
            from beat_this.inference import Audio2Beats

            # CPU: this deployment has no GPU by default, and the model is small
            # enough that a GPU would mostly buy cold-start latency (§18).
            self._tracker = Audio2Beats(checkpoint_path=_CHECKPOINT, device="cpu",
                                        dbn=False)
        return self._tracker

    def track(self, pcm: PCM, sr: int) -> BeatGrid:
        import numpy as np

        tracker = self._load()
        beats, downbeats = tracker(np.asarray(pcm, dtype="float32"), sr)

        beats_ms = [int(round(float(t) * 1000)) for t in beats]
        downbeats_ms = [int(round(float(t) * 1000)) for t in downbeats]
        if len(beats_ms) < 2:
            return BeatGrid(beats_ms=beats_ms, downbeats_ms=downbeats_ms,
                            bpm=0.0, confidence=0.0)

        intervals = [b - a for a, b in zip(beats_ms, beats_ms[1:]) if b > a]
        median_interval = statistics.median(intervals) if intervals else 0
        bpm = 60000.0 / median_interval if median_interval else 0.0

        meter, meter_agreement = _meter(beats_ms, downbeats_ms)

        # Two independent things have to hold for the grid to be trustworthy: a
        # steady pulse, and bars that are consistently the same length. A song
        # can have one without the other, and only the pair is worth anchoring
        # a video cursor to.
        #
        # **Multiplied, not averaged**, and that is a fix rather than a taste.
        # `regularity` is computed from beat intervals and is near 1.0 on
        # essentially every song; `meter_agreement` is the bar-length half. Under
        # the old mean, a song with a flawless pulse and *zero* bar agreement
        # scored 0.5·1.0 + 0.5·0.0 = 0.5, and `pipeline.assemble` tests
        # `< confidence_floor` against a floor of 0.5 — so a song whose bars were
        # entirely wrong could not be flagged by this path at all, and shipped a
        # sidecar. A product is the truth: no bars, no confidence.
        spread = statistics.pstdev(intervals) / median_interval if len(intervals) > 1 and median_interval else 1.0
        regularity = max(0.0, 1.0 - min(1.0, spread))
        confidence = max(0.0, min(1.0, regularity * meter_agreement))

        return BeatGrid(
            beats_ms=beats_ms,
            downbeats_ms=downbeats_ms,
            bpm=bpm,
            confidence=confidence,
            time_signature=f"{meter}/4",
        )


def _meter(beats_ms: list[int], downbeats_ms: list[int]) -> tuple[int, float]:
    """Beats per bar, and the share of bars that agree.

    Counted from the beats actually falling in each bar rather than from a
    duration ratio, so a tempo change inside the song doesn't invent a meter
    change. The agreement figure is what feeds confidence: a song whose bars are
    4,4,4,3,4,5 is not a 4/4 song we should be anchoring to.

    **Every bar counts against the agreement.** The sample used to be filtered
    to `1 < counted <= 13`, which threw away the strongest evidence a grid is
    broken: single-beat "bars" — a spurious downbeat fired one beat into a real
    one. So What has 37 of them across 179 bars and shipped at confidence 0.842
    with 42% of its bars malformed, because none of the 37 ever entered the
    denominator. They are in it now.

    **The meter itself is `downbeats.modal_bar_beats`.** A plain mode over the
    same sample answers the question badly in the one case it is being asked
    about: a spurious downbeat turns a 4 into a 1 and a 3, so a grid corrupted
    in half its bars holds more 3s than 4s and elects 3/4. §20.2a's estimator
    tries each candidate and keeps whichever leaves the fewest bars disagreeing
    with themselves, and it is the same function the repair downstream uses — so
    the meter this reports and the bars that repair chooses cannot diverge.
    """
    if len(downbeats_ms) < 3:
        return 4, 0.0
    counts = [sum(1 for t in beats_ms if start <= t < end)
              for start, end in zip(downbeats_ms, downbeats_ms[1:])]
    mode = modal_bar_beats(beats_ms, downbeats_ms)
    if mode is None:
        plausible = [c for c in counts if 1 < c <= 13]
        if not plausible:
            return 4, 0.0
        mode = statistics.mode(plausible)
    return mode, counts.count(mode) / len(counts)
