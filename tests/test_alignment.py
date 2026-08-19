"""Does the app show the right chord at the right moment in the recording?

This is the only question a chords-over-a-recording product is finally judged on,
and until this module existed nothing in the repo asked it. `lint` and `lint_sync`
both check the song's *internal* consistency — that its sections are whole bars,
that its anchors rise, that every pattern resolves — and a chart can satisfy all
of that while being uniformly wrong against the recording it was made from.

The measurement is the client's own two steps, run in Python
(`app/sync.py`: `song_beat_at` → `chord_at_song_beat`):

    video ms ──► songBeat ──► the chord the compiled chart sounds there

scored against the ground truth the fake engines were given. Both engines are
**exact** here — the grid and the chord spans are the truth, not an estimate — so
any error these tests find is the pipeline's own arithmetic and not an engine's
mistake. That is the point: it separates "BTC misheard the chord" (a quality
problem, measured in `bench/`) from "we put the right chord in the wrong place"
(a correctness problem, measured here).

Two metrics, because they fail differently:

- **frame accuracy** — sampled on a fixed ms grid across the whole song. This is
  what a listener experiences as "the chords are right".
- **downbeat accuracy** — sampled just after each bar starts. This is where the
  player's eye actually is, and a phase error shows up here first and hardest.
"""

from __future__ import annotations

import pytest

from app.analysis.pipeline import assemble
from app.analysis.types import EngineInfo
from app.chords import normalize, prefers_flats, render
from app.config import Settings
from app.lint import lint, lint_sync
from app.payload import CompositionPayload
from app.sync import chord_at_video_ms

from tests.conftest import recording

FRAME_STEP_MS = 50

_ENGINES = {"chords_engine": EngineInfo("exact", "1.0"),
            "beats_engine": EngineInfo("exact", "1.0")}


def _analyze(rec, **settings):
    outcome = assemble(meta=rec.meta, grid=rec.grid, raw=rec.chords, onsets=[],
                       settings=Settings(**settings), **_ENGINES)
    payload = CompositionPayload.model_validate(outcome.song)
    return payload, outcome.sync


def _expected(payload: CompositionPayload, label: str) -> str:
    """Ground truth in the same vocabulary the chart emits — an engine label is
    Harte (`G:maj`) and the chart renders the app's grammar (`G`), so comparing
    them raw would report a mismatch on every single frame."""
    root, quality, _ = normalize(label)
    return render(root, quality, flats=prefers_flats(payload.tonic, payload.mode))


def score(rec, payload: CompositionPayload, sync) -> tuple[float, float]:
    """(frame accuracy, downbeat accuracy) for one analyzed recording."""
    hits = total = 0
    start_ms, end_ms = rec.truth[0][0], rec.truth[-1][1]
    t = start_ms
    while t < end_ms:
        label = rec.label_at(t)
        if label is not None:
            total += 1
            if chord_at_video_ms(payload, sync, t) == _expected(payload, label):
                hits += 1
        t += FRAME_STEP_MS
    frame = hits / total if total else 0.0

    on_downbeat = sum(
        1 for bar_start, _, label in rec.truth
        # +1 ms rather than exactly on the boundary: at the instant of a change
        # either answer is defensible, and the assertion should be about the bar,
        # not about a tie-break.
        if chord_at_video_ms(payload, sync, bar_start + 1) == _expected(payload, label)
    )
    return frame, on_downbeat / len(rec.truth)


# ---------------------------------------------------------------------------
# The case that already worked
# ---------------------------------------------------------------------------

def test_song_starting_on_a_downbeat_is_exact():
    """`pickup_beats=0` — beat 0 is downbeat 0, so the chart's origin and the
    sidecar's origin coincide by luck rather than by construction. This is the
    geometry every other fixture in the suite uses."""
    rec = recording(pickup_beats=0)
    payload, sync = _analyze(rec)

    assert sync is not None, "a clean grid must produce a sidecar"
    frame, downbeat = score(rec, payload, sync)
    assert frame == 1.0
    assert downbeat == 1.0


# ---------------------------------------------------------------------------
# The case real recordings actually present
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pickup", [1, 2, 3])
def test_pickup_does_not_shift_the_chart(pickup):
    """A song whose first downbeat is not its first beat must still line up.

    This is the normal case, not an edge case: a beat tracker reports beats from
    the first audible pulse, and songs routinely open with a pickup or a fill.
    Held by `BeatAxis`, which defines beat 0 *as* the first downbeat — so the
    chart and the anchors cannot start from different places.
    """
    rec = recording(pickup_beats=pickup)
    payload, sync = _analyze(rec)

    assert sync is not None
    frame, downbeat = score(rec, payload, sync)
    assert downbeat == 1.0, f"pickup={pickup}: only {downbeat:.0%} of bars show the right chord"
    assert frame == 1.0


def test_pickup_at_realistic_tempo_and_length():
    rec = recording(
        progression=["C:maj", "G:maj", "A:min", "F:maj",
                     "C:maj", "G:maj", "F:maj", "G:maj"],
        bars=32, pickup_beats=1, ms_per_beat=545, first_beat_ms=120,
    )
    payload, sync = _analyze(rec)

    assert sync is not None
    frame, downbeat = score(rec, payload, sync)
    assert downbeat == 1.0
    assert frame == 1.0


def test_an_odd_bar_does_not_derail_the_bars_after_it():
    """A song that drops in a longer bar must not lose everything downstream.

    Here Comes The Sun is the corpus case: 11/8 and 15/8 bars inside a 4/4 song.
    The container cannot say that (§13.2 wants one uniform grid), so `BeatAxis`
    resamples the odd bar onto the song's meter — the bar still *starts and
    ends* on the tracker's real downbeats, which is what the anchors publish and
    what the player sees, and the error stays inside that one bar instead of
    shifting every bar after it.
    """
    rec = recording(bars=16, pickup_beats=0, odd_bars={5: 6, 11: 3})
    payload, sync = _analyze(rec)

    assert sync is not None
    frame, downbeat = score(rec, payload, sync)
    # The two odd bars are allowed to be imperfect inside themselves; every other
    # bar must be exactly right, which is what "the error does not propagate"
    # means in practice.
    assert downbeat == 1.0, f"only {downbeat:.0%} of bars start on the right chord"
    assert frame >= 0.95


# ---------------------------------------------------------------------------
# Structure must not invent harmony
# ---------------------------------------------------------------------------

def _varied():
    """Four passes of C G Am F, with bar 12's F played as Dm."""
    return recording(
        progression=["C:maj", "G:maj", "A:min", "F:maj",
                     "C:maj", "G:maj", "A:min", "F:maj",
                     "C:maj", "G:maj", "A:min", "D:min",   # <- the odd bar
                     "C:maj", "G:maj", "A:min", "F:maj"],
        bars=16, pickup_beats=0,
    )


def test_a_varied_repeat_keeps_its_own_chords():
    """Phase held clean, so the only thing under test is the structure pass.

    Bar 12 is a Dm where every other pass plays F. `_merge_similar` may still
    merge the two blocks into one section — that is what §15 wants on the rail —
    but it may not replay pass one's chords over pass two's bars.

    **Run with §21 off**, because §21 is a *different* claim about this same bar
    and the two have to be measured apart. The rule under test here is that
    section merging is not allowed to substitute one pass's chords for another's
    — a structural mistake, made silently, with no evidence consulted. §21 makes
    the opposite kind of move deliberately and on evidence (see the test below),
    and leaving it on here would mean this test passed or failed for a reason
    that has nothing to do with the merge.
    """
    rec = _varied()
    payload, sync = _analyze(rec, theory_form=False)

    assert sync is not None
    frame, downbeat = score(rec, payload, sync)
    assert downbeat == 1.0
    assert frame == 1.0


def test_the_form_layer_flattens_a_one_off_variation():
    """§21's cost, pinned — the trade it makes, stated as a number.

    Same recording, §21 on. Three passes of the section play F in bar 12 and one
    plays Dm; the form layer writes the section's own progression over all four,
    so the chart shows F there and **one bar in sixteen is not what the recording
    plays**. That is the whole of the price, it is paid on this exact shape of
    input, and it is not a bug — a chart is a claim about the song, and a song
    that plays its verse four times does not have four verses.

    What buys it is in `bench/`, not here: on real recordings the same rule turns
    Creep from eighty-eight distinct bars with a ten-chord vocabulary into
    `| G | G | B | B | C | C | Cm | Cm |` played eleven times with four. The
    per-pass variation that this test calls music is, on a recording, almost
    always the recognizer.

    Both directions are supported postures (`CHORDS_THEORY_FORM`), which is why
    this test and the one above can both stand.
    """
    rec = _varied()
    payload, sync = _analyze(rec)

    assert sync is not None
    frame, downbeat = score(rec, payload, sync)
    assert downbeat == 15 / 16, "exactly the odd bar, and nothing else, is flattened"
    assert frame >= 0.9


def test_contrasting_sections_survive_the_structure_pass():
    """The control for the test above: blocks that differ enough not to be merged
    are reproduced exactly, which is what makes the failure above a threshold
    problem rather than a wholesale one."""
    rec = recording(
        progression=["C:maj", "G:maj", "A:min", "F:maj", "C:maj", "G:maj", "F:maj", "G:maj",
                     "C:maj", "G:maj", "A:min", "F:maj", "C:maj", "G:maj", "F:maj", "G:maj",
                     "F:maj", "C:maj", "G:maj", "A:min", "F:maj", "C:maj", "D:min", "G:maj",
                     "C:maj", "G:maj", "A:min", "F:maj", "C:maj", "G:maj", "F:maj", "G:maj"],
        bars=32, pickup_beats=0,
    )
    payload, sync = _analyze(rec)

    assert sync is not None
    frame, downbeat = score(rec, payload, sync)
    assert frame == 1.0
    assert downbeat == 1.0


# ---------------------------------------------------------------------------
# What the existing validation can and cannot see
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pickup", [0, 1, 2, 3])
def test_lint_passes_regardless_of_alignment(pickup):
    """Documents the gap rather than the defect.

    Every misaligned chart above lints clean **and** ships a sidecar, so §13.3's
    "degrade honestly" never fires: it can only catch anchors that disagree with
    the chart's shape, never a chart that disagrees with the music. After Phase 1
    this test should still pass — it is not asserting that lint is wrong, only
    that lint is not the thing protecting this property.
    """
    rec = recording(pickup_beats=pickup)
    payload, sync = _analyze(rec)

    assert lint(payload) == []
    assert sync is not None
    assert lint_sync(payload, sync) == []
