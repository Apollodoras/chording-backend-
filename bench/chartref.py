"""The reference-chart format: what a correct answer looks like, written by hand.

    Title: The Silence
    Artist: Manchester Orchestra
    Key: E minor (Dorian)
    Tempo: 75
    Time: 4/4

    [Intro]
    | Em   | D    | A    | Em   |
    | Em   | D    | A    | Em   |

    [Chorus 1]
    | G    | D    | A    | Em   |

This is **not** an output format and it carries no timestamps. The system's own
output stays what it is — a chart plus a `videoSync` sidecar timed to the
recording — and the sync gate stays `bench/run_bench.py`'s `delivered_accuracy`,
which is the only thing that can catch a chart laid onto the wrong beat origin.
What this format is for is teaching and grading the *musical* answer: which
chords, in which bars, in which sections, how many times.

Three things it is deliberate about.

**One line is one phrase.** `| Em | D | A | Em |` is a four-bar phrase, and a
section is a stack of them. Phrase is the unit the ear actually repeats, and it
is the unit a consensus vote should pool evidence over — so writing charts this
way is not cosmetic, it is the structure the analysis is supposed to find.

**Bars-per-phrase and chords-per-bar are derived, never authored.** A summary
line saying "4 bars per phrase, 1 chord per bar" can disagree with the pipes
underneath it; a derived one cannot. It also means an irregular song is writable:
`| Cm F | Bb |` is a two-bar phrase with two chords in its first bar, and
`Here Comes The Sun` (only 73% of its bars in 4) needs no special case.

**Key is two machine fields.** "E minor (Dorian)" is two different claims, and
the parenthetical is the specific one — `Em D A G` contains the C# of the A
major, so the mode really is Dorian and "minor" is the loose usage. A grader
cannot guess which half was meant, so the parenthetical wins and the result is
stored as `tonic` + `mode`.

Cell conventions, all of them standard leadsheet:

- `%` or an empty cell repeats the previous bar.
- `N.C.` is no chord.
- Several chords in one cell divide that bar: `| Cm F |`.
- `#` starts a comment; blank lines are ignored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from app.chords import normalize_name

# A power chord has no third, so "is it major or minor" is a question the
# recording does not answer. The app's grammar (a port of the client's
# `ChordSymbol`) has no `5` quality, so this is a name the reference may use and
# the system may never emit — see `POWER` handling in `grade_chart.py`.
POWER = "power"

_HEADER = re.compile(r"^([A-Za-z ]+):\s*(.*)$")
_SECTION = re.compile(r"^\[(.+)\]\s*$")
_TEMPO = re.compile(r"([0-9]+(?:\.[0-9]+)?)")
_KEY = re.compile(
    r"^\s*([A-G][#b♯♭]?)\s*"
    r"(?:\((?P<paren>[A-Za-z]+)[^)]*\)|(?P<plain>[A-Za-z]+))?"
    r"(?:\s*\((?P<paren2>[A-Za-z]+)[^)]*\))?\s*$"
)

# The mode names a chart may use. `minor`/`major` are the loose ones and stay
# legal; the specific ones are what a careful entry says.
MODES = {
    "major": "major", "maj": "major", "ionian": "ionian",
    "minor": "minor", "min": "minor", "m": "minor", "aeolian": "aeolian",
    "dorian": "dorian", "phrygian": "phrygian", "lydian": "lydian",
    "mixolydian": "mixolydian", "locrian": "locrian",
}

# Leading word of a section label → its structural kind. "Verse 1" and "Verse 2"
# are the same kind and different labels, and the distinction matters: form is
# graded on kinds, labels are only reported (whether the engine says "Interlude"
# or "Verse 3" is close to unfalsifiable, and `form.py` already has an honest
# `Part N` fallback for exactly that reason).
_KINDS = {
    "intro": "intro", "verse": "verse", "chorus": "chorus", "prechorus": "prechorus",
    "pre": "prechorus", "bridge": "bridge", "interlude": "interlude",
    "instrumental": "interlude", "solo": "solo", "outro": "outro", "coda": "outro",
    "refrain": "chorus", "hook": "chorus", "breakdown": "bridge", "part": "part",
}


class ChartError(ValueError):
    """The chart could not be read. Always names the line."""


@dataclass(frozen=True)
class Bar:
    """One bar: the chords sounding in it, in order. Never empty — a bar with no
    chord carries the single name `N.C.`."""

    chords: tuple[str, ...]

    @property
    def head(self) -> str:
        """The chord the bar *is*, for grading at bar resolution."""
        return self.chords[0]


@dataclass(frozen=True)
class Section:
    label: str
    kind: str
    phrases: tuple[tuple[Bar, ...], ...]

    @property
    def bars(self) -> tuple[Bar, ...]:
        return tuple(bar for phrase in self.phrases for bar in phrase)

    @property
    def phrase_count(self) -> int:
        return len(self.phrases)

    @property
    def bars_per_phrase(self) -> list[int]:
        return [len(p) for p in self.phrases]

    @property
    def signature(self) -> tuple[str, ...]:
        """The section's harmonic identity — its first phrase, as chord heads.

        Used to letter the form (A A B A B C). The *first phrase* rather than
        every bar, because two verses of different lengths are still the same
        music and a whole-section comparison would call them different.
        """
        return tuple(bar.head for bar in self.phrases[0]) if self.phrases else ()


@dataclass
class Chart:
    title: str = ""
    artist: str = ""
    tonic: str = ""
    mode: str = ""
    tempo: float | None = None
    time_signature: str = "4/4"
    sections: list[Section] = field(default_factory=list)
    source: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def bars(self) -> list[Bar]:
        return [bar for section in self.sections for bar in section.bars]

    @property
    def vocabulary(self) -> set[str]:
        return {c for bar in self.bars for c in bar.chords if c != "N.C."}

    @property
    def form(self) -> list[str]:
        """The repetition pattern, as letters: A A B A B C.

        Computed from harmonic content, not from the labels, so a chart and a
        system payload can be compared on form even when one of them calls
        everything `Part N`.
        """
        letters: dict[tuple[str, ...], str] = {}
        out = []
        for section in self.sections:
            key = section.signature
            if key not in letters:
                letters[key] = chr(ord("A") + len(letters)) if len(letters) < 26 else "?"
            out.append(letters[key])
        return out


def _split_cells(line: str) -> list[str]:
    """`| Em | D |` → `["Em", "D"]`.

    A leading and trailing pipe are conventional and optional; what is not
    optional is that cells are *between* pipes, so `Em | D` is two bars and not
    one bar named "Em | D".
    """
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    return [cell.strip() for cell in body.split("|")]


def _parse_cell(cell: str, previous: Bar | None, lineno: int) -> Bar:
    if not cell or cell == "%":
        if previous is None:
            raise ChartError(f"line {lineno}: '%' with no previous bar to repeat")
        return previous
    names = []
    for token in cell.split():
        if token.upper() in ("N.C.", "NC", "N.C"):
            names.append("N.C.")
            continue
        # `A5` is legal in a reference and illegal in the app's grammar, so it is
        # recognised here rather than pushed through `normalize_name` (which
        # would silently make it a major triad and erase the distinction the
        # entry was written to record).
        if re.fullmatch(r"[A-G][#b♯♭]?5", token):
            names.append(token)
            continue
        name = normalize_name(token)
        if name is None:
            raise ChartError(f"line {lineno}: {token!r} is not a chord this grammar reads")
        names.append(name)
    if not names:
        raise ChartError(f"line {lineno}: empty bar")
    return Bar(tuple(names))


def _parse_key(value: str, lineno: int) -> tuple[str, str]:
    match = _KEY.match(value)
    if not match:
        raise ChartError(f"line {lineno}: cannot read key {value!r}")
    tonic = match.group(1).replace("♯", "#").replace("♭", "b")
    # The parenthetical wins over the bare word. "E minor (Dorian)" is a loose
    # claim beside a specific one, and the specific one is the gradeable half.
    raw = match.group("paren") or match.group("paren2") or match.group("plain") or "major"
    mode = MODES.get(raw.strip().lower())
    if mode is None:
        raise ChartError(f"line {lineno}: unknown mode {raw!r}")
    return tonic, mode


def _kind_of(label: str) -> str:
    word = re.split(r"[\s/\-]+", label.strip().lower())[0]
    return _KINDS.get(word, "part")


def parse_chart(text: str, *, source: str = "") -> Chart:
    """Read the format. Raises `ChartError` naming the offending line."""
    chart = Chart(source=source)
    label: str | None = None
    phrases: list[tuple[Bar, ...]] = []
    previous: Bar | None = None
    seen_section = False

    def close():
        nonlocal label, phrases
        if label is not None:
            chart.sections.append(
                Section(label=label, kind=_kind_of(label), phrases=tuple(phrases)))
        label, phrases = None, []

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        section = _SECTION.match(line.strip())
        if section:
            close()
            label = section.group(1).strip()
            seen_section = True
            previous = None
            continue

        if "|" in line:
            if label is None:
                raise ChartError(f"line {lineno}: bars before any [Section]")
            bars: list[Bar] = []
            for cell in _split_cells(line):
                bar = _parse_cell(cell, previous, lineno)
                bars.append(bar)
                previous = bar
            if bars:
                phrases.append(tuple(bars))
            continue

        header = _HEADER.match(line.strip())
        if header and not seen_section:
            name, value = header.group(1).strip().lower(), header.group(2).strip()
            if name == "title":
                chart.title = value
            elif name == "artist":
                chart.artist = value
            elif name == "key":
                chart.tonic, chart.mode = _parse_key(value, lineno)
            elif name == "tempo":
                found = _TEMPO.search(value)
                if not found:
                    raise ChartError(f"line {lineno}: cannot read tempo {value!r}")
                chart.tempo = float(found.group(1))
            elif name in ("time", "time signature", "meter"):
                chart.time_signature = value
            elif name == "note":
                chart.notes.append(value)
            else:
                raise ChartError(f"line {lineno}: unknown header {name!r}")
            continue

        raise ChartError(f"line {lineno}: cannot read {line.strip()!r}")

    close()
    if not chart.sections:
        raise ChartError("chart has no sections")
    return chart


def load_chart(path: str | Path) -> Chart:
    path = Path(path)
    return parse_chart(path.read_text(), source=path.name)


# --------------------------------------------------------------------------
# The other direction: a system payload, in the same shape
# --------------------------------------------------------------------------

def chart_from_payload(song: dict, *, title: str = "", artist: str = "") -> Chart:
    """Read a `CompositionPayload` v2 back into a `Chart`, so system output and
    reference can be compared — and *printed* — as the same kind of object.

    The two encodings `compile.py` emits both have to be handled, and reading
    only one of them is a mistake this repo has already made twice (see
    `seed_catalog._chart`): a section with `bars` carries them explicitly, and a
    section without is the **flat** form, where `chordNames` is the whole
    section at `beatsPerChord` beats each. Treating the flat form as empty
    reports a dropped section; treating its `chordNames` as one bar reports a
    12-bar blues as one chord.

    Phrases are *not* in the payload — there is no phrase concept in v2 — so a
    section becomes one phrase of all its bars unless `repeats` says otherwise.
    That is the honest reading, and the gap it exposes is the point.
    """
    beats_per_bar = _beats_per_bar(song.get("timeSignature") or "4/4")
    sections: list[Section] = []

    for wire in ((song.get("arrangement") or {}).get("sections")) or []:
        if not isinstance(wire, dict):
            continue
        repeats = max(1, int(wire.get("repeats") or 1))
        bars = _bars_from_section(wire, beats_per_bar)
        if not bars:
            continue
        label = wire.get("name") or wire.get("kindRaw") or "Part"
        # `repeats` is the payload's own statement that this music plays N times,
        # so it is expanded into N phrases rather than silently collapsed — the
        # bar count has to match the reference's, and the reference writes the
        # repeats out.
        phrase = tuple(bars)
        sections.append(Section(label=str(label), kind=_kind_of(str(label)),
                                phrases=tuple([phrase] * repeats)))

    return Chart(
        title=title or str(song.get("title") or ""),
        artist=artist,
        tonic=str(song.get("tonic") or ""),
        mode=str(song.get("mode") or ""),
        tempo=float(song["tempo"]) if song.get("tempo") else None,
        time_signature=str(song.get("timeSignature") or "4/4"),
        sections=sections,
        source="payload",
    )


def _beats_per_bar(time_signature: str) -> int:
    try:
        return max(1, int(str(time_signature).split("/")[0]))
    except (ValueError, IndexError):
        return 4


def _bars_from_section(wire: dict, beats_per_bar: int) -> list[Bar]:
    explicit = wire.get("bars") or []
    if explicit:
        out = []
        for bar in explicit:
            if not isinstance(bar, dict):
                continue
            names = [s["chordName"] for s in (bar.get("chordSpans") or [])
                     if isinstance(s, dict) and s.get("chordName")]
            out.append(Bar(tuple(names) if names else ("N.C.",)))
        return out

    names = [n for n in (wire.get("chordNames") or []) if n]
    if not names:
        return []
    per_chord = int(wire.get("beatsPerChord") or beats_per_bar)
    if per_chord >= beats_per_bar:
        # One chord per bar (or held longer — still one bar each here, which is
        # what the flat encoding means).
        return [Bar((n,)) for n in names]
    # Several chords to a bar: pack them.
    per_bar = max(1, beats_per_bar // per_chord)
    return [Bar(tuple(names[i:i + per_bar])) for i in range(0, len(names), per_bar)]


# --------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------

def render_chart(chart: Chart, *, width: int = 6) -> str:
    """Back to the text format. Round-trips `parse_chart`.

    This is what makes a failing grade *readable*: reference and output printed
    in one shape, so the difference is visible rather than inferred from a score.
    """
    out = []
    if chart.title:
        out.append(f"Title: {chart.title}")
    if chart.artist:
        out.append(f"Artist: {chart.artist}")
    if chart.tonic:
        out.append(f"Key: {chart.tonic} {chart.mode}".rstrip())
    if chart.tempo:
        out.append(f"Tempo: {chart.tempo:g}")
    out.append(f"Time: {chart.time_signature}")

    for section in chart.sections:
        out.append("")
        out.append(f"[{section.label}]")
        for phrase in section.phrases:
            cells = " ".join(f"| {' '.join(bar.chords):<{width}}" for bar in phrase)
            out.append(f"{cells} |")
    return "\n".join(out) + "\n"


def describe(chart: Chart) -> str:
    """The bullet summary, derived from the pipes rather than authored beside
    them — the shape the owner's first sketch asked for."""
    out = [f"Title: {chart.title}", f"Artist: {chart.artist}",
           f"Key: {chart.tonic} {chart.mode}",
           f"Tempo: ~{chart.tempo:g} BPM" if chart.tempo else "Tempo: unknown",
           f"Time Signature: {chart.time_signature}", ""]
    for section in chart.sections:
        per_phrase = chart_bars_summary(section.bars_per_phrase)
        chords = sorted({len(bar.chords) for bar in section.bars})
        out.append(f"{section.label}")
        out.append(f"  Number of phrases: {section.phrase_count}")
        out.append(f"  Bars per phrase: {per_phrase}")
        out.append(f"  Chords per bar: {', '.join(str(c) for c in chords)}"
                   f"  ({' | '.join(' '.join(b.chords) for b in section.phrases[0])})"
                   if section.phrases else "")
    return "\n".join(out) + "\n"


def chart_bars_summary(counts: list[int]) -> str:
    unique = sorted(set(counts))
    return str(unique[0]) if len(unique) == 1 else ", ".join(str(c) for c in unique)
