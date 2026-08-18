"""Score a system chart against a reference chart — at four resolutions, plus a
transposition sweep that needs no reference at all.

    python bench/grade_chart.py --reference bench/reference/so-what.chart \
                                --payload /tmp/so-what.json

The resolutions exist because "does it match" is not one question:

- **root** — right root, **for how much of the bar**. The number to read first: a
  chart with every root right and some qualities wrong is a different and much
  smaller problem than one that has drifted.

  Scored over the bar's *duration*, not over its first chord, and the difference
  is not academic. Grading `bar.head` alone means half of `| Am G |` is never
  looked at — so a system that reads Smooth Criminal's `| Am G | F E |` a chord
  late scores every bar wrong on a head comparison and half of every bar right in
  fact, and neither number tells you which. Several chords in one cell divide that
  cell evenly (`chartref`'s stated convention), so the comparison is over
  `_SLOTS` equal slices per bar and a bar holding one chord is simply twelve
  identical slices.
- **triad** — root plus major/minor, over the same slices, on the bars that
  aligned. Quality beyond the third is not graded, and not because it is
  unimportant: `§12.2` already ruled that the app plays `Cmaj9` and `Cmaj7`
  identically, so scoring them apart would measure a difference the product does
  not have.
- **bars** — did the chart put its changes where the music puts them. Reported as
  alignment coverage plus the raw bar counts, because a chart that is right about
  every chord and 30 bars too long has a real defect that the root score hides.
- **form** — the repetition pattern (A A B A B C), compared as a string. Section
  *labels* are reported and never scored: whether the engine says "Interlude" or
  "Verse 3" is close to unfalsifiable, and `form.py` already prefers an honest
  `Part N` to a confident guess.

And the one that carries the most weight per line of code:

**The transposition sweep.** The system chart is rotated through all twelve
semitones and re-scored. If some rotation beats rotation 0 by a wide margin, the
chart is *right and in the wrong key* — a pitch-reference bug, not a harmony
bug. That distinction is invisible to every other metric here, it is the defect
that cost Mary Jane's Last Dance 82% of its chart, and — this is the point — it
does not need the reference to be correct in absolute terms. The same sweep run
against a chart's own most-likely key works on songs nothing has ground truth
for, which is what lets it run over a live catalogue.

Two more, reported rather than scored, because they are what "musical" means
operationally:

- **vocabulary** — how many distinct chords. Real songs have 3–6; a chart with 15
  is transcribing noise as harmony.
- **changes** — how many chord changes. Same idea in the time domain.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chords import parse_chord  # noqa: E402
from bench.chartref import (  # noqa: E402
    POWER, Bar, Chart, chart_from_payload, load_chart, render_chart,
)

_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_POWER_RE = re.compile(r"^([A-G][#b♯♭]?)5$")

# Alignment weights. A mismatch is free and a gap is not, so the alignment stays
# dense: the question being asked is "which reference bar does this system bar
# correspond to", and answering it with a gap whenever the chord is wrong would
# flatter every score by only ever aligning the bars that already agree.
_MATCH = 1.0
_MISMATCH = 0.0
_GAP = -0.25

# Slices per bar for the duration-weighted comparison. Twelve divides by 2, 3, 4
# and 6, so every cell a reference chart can legally hold — two chords, three,
# four, six — splits into whole slices and none of them is rounded away.
_SLOTS = 12


def _root_quality(name: str) -> tuple[int | None, str]:
    """(pitch class, quality) for a chart cell, power chords included."""
    if name in ("N.C.", "NC"):
        return None, "none"
    power = _POWER_RE.match(name)
    if power:
        parsed = parse_chord(power.group(1))
        return (parsed[0] if parsed else None), POWER
    parsed = parse_chord(name)
    if parsed is None:
        return None, "none"
    return parsed[0], parsed[1]


def _is_minorish(quality: str) -> bool:
    return quality in ("minor", "minor7", "diminished", "diminished7", "halfDiminished7")


def _triad_agrees(reference: str, system: str) -> bool:
    """Do these two chords agree at major/minor resolution?

    The power-chord rule is the interesting case. A fifth has no third, so the
    recording does not say major or minor — but the *convention* is unambiguous
    (Ultimate Guitar prints `A`, and `A` is the only one of the two the app's
    grammar can even express). So a reference `A5` accepts `A` and rejects `Am`:
    inventing a minor third against a rock power chord is the specific defect,
    and calling it unscoreable would let it through.
    """
    ref_pc, ref_q = _root_quality(reference)
    sys_pc, sys_q = _root_quality(system)
    if ref_pc is None or sys_pc is None or ref_pc != sys_pc:
        return False
    if ref_q == POWER:
        return not _is_minorish(sys_q)
    return _is_minorish(ref_q) == _is_minorish(sys_q)


def _pc(bar: Bar) -> int | None:
    return _root_quality(bar.head)[0]


def _slices(bar: Bar) -> list[str]:
    """A bar as `_SLOTS` equal slices of chord name. `| Am G |` is six of each."""
    names = bar.chords or ("N.C.",)
    return [names[min(len(names) - 1, index * len(names) // _SLOTS)]
            for index in range(_SLOTS)]


def _shared(reference: Bar, system: Bar, agrees) -> int:
    """How many of a bar's slices the two charts agree on, under `agrees`."""
    return sum(1 for a, b in zip(_slices(reference), _slices(system)) if agrees(a, b))


def _same_root(reference: str, system: str) -> bool:
    ref_pc, _ = _root_quality(reference)
    sys_pc, _ = _root_quality(system)
    return ref_pc is not None and ref_pc == sys_pc


def _transpose_bar(bar: Bar, semitones: int) -> Bar:
    out = []
    for name in bar.chords:
        pc, quality = _root_quality(name)
        if pc is None:
            out.append(name)
            continue
        moved = _NAMES[(pc + semitones) % 12]
        if quality == POWER:
            out.append(f"{moved}5")
        else:
            out.append(re.sub(r"^[A-G][#b♯♭]?", moved, name))
    return Bar(tuple(out))


@dataclass
class Alignment:
    pairs: list[tuple[int | None, int | None]]

    @property
    def matched(self) -> list[tuple[int, int]]:
        return [(a, b) for a, b in self.pairs if a is not None and b is not None]


def align(reference: list[Bar], system: list[Bar]) -> Alignment:
    """Needleman-Wunsch over bar roots.

    Gap-tolerant on purpose: a system chart that finds an extra two-bar intro is
    not wrong about the song, and scoring at a single best offset would punish it
    for the whole length of the chart. What a fixed offset *cannot* express is an
    insertion in the middle, which is exactly what a section boundary error looks
    like.
    """
    n, m = len(reference), len(system)
    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    back = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0] = score[i - 1][0] + _GAP
        back[i][0] = 1
    for j in range(1, m + 1):
        score[0][j] = score[0][j - 1] + _GAP
        back[0][j] = 2
    for i in range(1, n + 1):
        ref_pc = _pc(reference[i - 1])
        for j in range(1, m + 1):
            same = ref_pc is not None and ref_pc == _pc(system[j - 1])
            diag = score[i - 1][j - 1] + (_MATCH if same else _MISMATCH)
            up = score[i - 1][j] + _GAP
            left = score[i][j - 1] + _GAP
            best = max(diag, up, left)
            score[i][j] = best
            back[i][j] = 0 if best == diag else (1 if best == up else 2)

    pairs: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and back[i][j] == 0:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or back[i][j] == 1):
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    pairs.reverse()
    return Alignment(pairs)


def _root_score(reference: list[Bar], system: list[Bar]) -> tuple[float, Alignment]:
    """Share of the reference's *duration* whose root the system gets right.

    A reference bar the alignment could not place contributes its slices as
    misses, so dropping a section costs exactly the bars that were dropped.
    """
    if not reference:
        return 0.0, Alignment([])
    alignment = align(reference, system)
    hits = sum(_shared(reference[a], system[b], _same_root) for a, b in alignment.matched)
    return hits / (len(reference) * _SLOTS), alignment


def grade(reference: Chart, system: Chart) -> dict:
    ref_bars, sys_bars = reference.bars, system.bars
    root, alignment = _root_score(ref_bars, sys_bars)

    aligned = alignment.matched
    triad_hits = sum(_shared(ref_bars[a], sys_bars[b], _triad_agrees) for a, b in aligned)
    triad = triad_hits / (len(ref_bars) * _SLOTS) if ref_bars else 0.0

    # The sweep. Rotation 0 is the plain score; the winner tells us whether the
    # chart is wrong or merely displaced.
    sweep = []
    for k in range(12):
        moved = [_transpose_bar(bar, k) for bar in sys_bars]
        sweep.append(round(_root_score(ref_bars, moved)[0], 4))
    best_k = max(range(12), key=lambda k: sweep[k])

    def changes(bars: list[Bar]) -> int:
        heads = [b.head for b in bars]
        return sum(1 for x, y in zip(heads, heads[1:]) if x != y)

    return {
        "title": reference.title,
        "scores": {
            "root": round(root, 4),
            "triad": round(triad, 4),
            "barCoverage": round(len(aligned) / len(ref_bars), 4) if ref_bars else 0.0,
            "form": _form_score(reference, system),
        },
        "transposition": {
            "bestRotation": best_k,
            "bestScore": sweep[best_k],
            "atZero": sweep[0],
            # The signature of a pitch-reference bug: some other rotation wins,
            # and wins by a margin no ordinary chord error could produce.
            "displaced": bool(best_k != 0 and sweep[best_k] - sweep[0] > 0.2),
            "sweep": sweep,
        },
        "bars": {"reference": len(ref_bars), "system": len(sys_bars)},
        "vocabulary": {
            "reference": sorted(reference.vocabulary),
            "system": sorted(system.vocabulary),
            "referenceCount": len(reference.vocabulary),
            "systemCount": len(system.vocabulary),
        },
        "changes": {"reference": changes(ref_bars), "system": changes(sys_bars)},
        "key": {
            "reference": f"{reference.tonic} {reference.mode}",
            "system": f"{system.tonic} {system.mode}",
            "tonicMatch": _same_tonic(reference.tonic, system.tonic),
        },
        "tempo": {
            "reference": reference.tempo, "system": system.tempo,
            "ratio": round(system.tempo / reference.tempo, 3)
            if reference.tempo and system.tempo else None,
        },
        "form": {"reference": "".join(_collapsed(reference.form)),
                 "system": "".join(_collapsed(system.form)),
                 "referenceLabels": [s.label for s in reference.sections],
                 "systemLabels": [s.label for s in system.sections]},
    }


def _same_tonic(a: str, b: str) -> bool:
    pa, pb = parse_chord(a or "?"), parse_chord(b or "?")
    return bool(pa and pb and pa[0] == pb[0])


def _form_score(reference: Chart, system: Chart) -> float:
    """How much of the reference's repetition pattern survives, as a ratio of
    longest-common-subsequence to reference length. Labels are not consulted.

    **Runs are collapsed first**, and that is the difference between measuring
    form and measuring a labelling choice. Creep is one eight-bar progression
    played eleven times; a reference that writes it out as intro / verse / chorus
    / verse / chorus / bridge / chorus / outro has the form string `AAAAAAAA`,
    and a system that says "one section, eleven times" has `A`. Those describe the
    *same music*, and the raw LCS scores the agreement at 0.125 — so the metric
    would report a total failure on a song the analysis got exactly right, purely
    because the two disagree about where a repeat is allowed to be split.

    Collapsing `AAAAAAAA` and `A` both to `A` scores that 1.0. It costs nothing
    that matters: contrast is what form *is*, so `ABABAC` survives collapsing
    unchanged, and a system that flattens a real ABABAC song into one section
    still scores 1/6. What stops being penalized is only the thing that was never
    a musical claim.
    """
    a, b = _collapsed(reference.form), _collapsed(system.form)
    if not a:
        return 0.0
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            table[i][j] = (table[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1]
                           else max(table[i - 1][j], table[i][j - 1]))
    return round(table[len(a)][len(b)] / len(a), 4)


def _collapsed(form: list[str]) -> list[str]:
    """`A A B A A` → `A B A`. Consecutive repeats of one letter are one entry."""
    return [letter for index, letter in enumerate(form)
            if index == 0 or letter != form[index - 1]]


def diff_bars(reference: Chart, system: Chart, limit: int = 40) -> str:
    """Side by side, so a failing grade is readable rather than inferred."""
    ref_bars, sys_bars = reference.bars, system.bars
    alignment = align(ref_bars, sys_bars)
    lines = [f"{'#':>4}  {'reference':<14} {'system':<14}  "]
    shown = 0
    for a, b in alignment.pairs:
        if shown >= limit:
            lines.append(f"      … {len(alignment.pairs) - shown} more bars")
            break
        left = " ".join(ref_bars[a].chords) if a is not None else "—"
        right = " ".join(sys_bars[b].chords) if b is not None else "—"
        if a is None or b is None:
            ok = "XX"
        else:
            share = _shared(ref_bars[a], sys_bars[b], _triad_agrees) / _SLOTS
            ok = "ok" if share > 0.99 else (f"{share:.0%}" if share else "XX")
        index = str(a + 1) if a is not None else "-"
        lines.append(f"{index:>4}  {left:<14} {right:<14}  {ok}")
        shown += 1
    return "\n".join(lines)


def report(result: dict) -> str:
    s, t = result["scores"], result["transposition"]
    out = [
        f"{result['title']}",
        f"  root  {s['root']:.3f}   triad {s['triad']:.3f}   "
        f"bars {s['barCoverage']:.3f}   form {s['form']:.3f}",
        f"  bars      reference {result['bars']['reference']}  "
        f"system {result['bars']['system']}",
        f"  vocabulary reference {result['vocabulary']['referenceCount']} "
        f"{result['vocabulary']['reference']}",
        f"             system    {result['vocabulary']['systemCount']} "
        f"{result['vocabulary']['system']}",
        f"  changes   reference {result['changes']['reference']}  "
        f"system {result['changes']['system']}",
        f"  key       reference {result['key']['reference']!r}  "
        f"system {result['key']['system']!r}  tonic match: {result['key']['tonicMatch']}",
        f"  tempo     reference {result['tempo']['reference']}  "
        f"system {result['tempo']['system']}  ratio {result['tempo']['ratio']}",
        f"  form      reference {result['form']['reference']}  "
        f"system {result['form']['system']}",
    ]
    if t["displaced"]:
        out.append(f"  ** TRANSPOSED: rotation +{t['bestRotation']} scores "
                   f"{t['bestScore']:.3f} against {t['atZero']:.3f} at concert pitch. "
                   f"The chart is right and in the wrong key.")
    else:
        out.append(f"  transposition best rotation +{t['bestRotation']} "
                   f"({t['bestScore']:.3f}) vs {t['atZero']:.3f} at zero")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--payload", required=True,
                        help="JSON with a `song` key, or a bare CompositionPayload")
    parser.add_argument("--diff", action="store_true", help="print the bar-by-bar diff")
    parser.add_argument("--render", action="store_true", help="print the system chart")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    reference = load_chart(args.reference)
    raw = json.loads(Path(args.payload).read_text())
    song = raw.get("song", raw)
    system = chart_from_payload(song)

    result = grade(reference, system)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(report(result))
        if args.render:
            print()
            print(render_chart(system))
        if args.diff:
            print()
            print(diff_bars(reference, system))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
