"""Emit the wire bytes the app's own importer will judge (§16.5).

Mo's best idea, taken verbatim: the backend writes its serializer's **real
output** as JSON fixtures, and the iOS test suite walks them through the actual
importer (`CompositionPayload.from(jsonData:)` → `ComposerService.import` →
`campfireSheet`) asserting decode + **zero warnings**. That makes the app's own
importer the final judge of backend output, in CI, with no API key and no
network.

Two sources, both real:

- ``fixtures/valid/*.json`` — hand-written canonical payloads, round-tripped
  through `CompositionPayload` so what lands in `emitted/` is the serializer's
  bytes rather than the hand JSON.
- **the pipeline itself**, run over the known song with fake engines. This is the
  more valuable half: it is the actual shape an analysis produces — bars mode,
  extracted patterns, `yt:` ids — and it needs no audio.

Run:  python tests/emit_reference_fixtures.py

Then, on the app side, point `CHORDS_BACKEND_FIXTURES` at `tests/fixtures`
(mirroring Mo's `MO_BACKEND_FIXTURES`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.analysis.pipeline import assemble  # noqa: E402
from app.analysis.types import EngineInfo  # noqa: E402
from app.config import Settings  # noqa: E402
from app.lint import lint  # noqa: E402
from app.payload import CompositionPayload  # noqa: E402
from tests.conftest import (  # noqa: E402
    known_chords,
    known_grid,
    known_meta,
    known_onsets,
)

VALID = ROOT / "tests" / "fixtures" / "valid"
EMITTED = ROOT / "tests" / "fixtures" / "emitted"

# The sidecar carries `analyzedAt`, which is a wall clock and therefore the one
# field that makes these files differ on every run. Pinned here so a diff in
# `emitted/` means **the analysis changed** — which is the entire value of
# committing them. The importer only cares that the field is present and well
# formed, never what instant it names, so freezing it costs the contract nothing.
REFERENCE_ANALYZED_AT = "2026-01-01T00:00:00Z"


def emit_hand_written() -> int:
    problems_found = 0
    for source in sorted(VALID.glob("*.json")):
        payload = CompositionPayload.model_validate_json(source.read_text())
        problems = lint(payload)
        if problems:
            print(f"! {source.name} does not lint clean: {problems}")
            problems_found += 1
            continue
        (EMITTED / source.name).write_text(payload.wire_json() + "\n")
        print(f"emitted {source.name}")
    return problems_found


def emit_from_pipeline() -> int:
    """The real thing: what an analysis actually produces, minus the audio."""
    settings = Settings(scratch_root="/tmp/chords-scratch")
    outcome = assemble(
        meta=known_meta(), grid=known_grid(), raw=known_chords(), onsets=known_onsets(),
        settings=settings,
        chords_engine=EngineInfo("reference", "1.0.0"),
        beats_engine=EngineInfo("reference", "1.0.0"),
    )
    name = "pipeline-known-song.json"
    (EMITTED / name).write_text(json.dumps(outcome.song, ensure_ascii=False,
                                           sort_keys=True, indent=2) + "\n")
    print(f"emitted {name}")

    # The sidecar rides alongside rather than inside the payload, so it gets its
    # own file — the app's contract test must be able to read a payload that is
    # byte-identical to what the importer expects.
    if outcome.sync is not None:
        sidecar = EMITTED / "pipeline-known-song-videosync.json"
        stable = outcome.sync.model_copy(update={"analyzedAt": REFERENCE_ANALYZED_AT})
        sidecar.write_text(json.dumps(stable.model_dump(), ensure_ascii=False,
                                      sort_keys=True, indent=2) + "\n")
        print(f"emitted {sidecar.name}")
    return 0


def main() -> int:
    EMITTED.mkdir(parents=True, exist_ok=True)
    return emit_hand_written() + emit_from_pipeline()


if __name__ == "__main__":
    raise SystemExit(main())
