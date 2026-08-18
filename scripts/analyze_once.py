"""Analyze one video with the LOCAL working tree and print the payload. Stores nothing.

    modal run scripts/analyze_once.py --video-id aowSGxim_O8 > /tmp/mary.json

The gap this fills sits between `real_song_check.py` ("did the pipeline run at
all") and `seed_catalog.py` ("is the chart right, and put it in the catalog").
While the analysis is being changed, the catalog is a cache to be dropped rather
than an asset to be filled — so this runs the real worker image, on real audio,
through the real pipeline, and hands the payload back **without touching the
store**. Nothing a user could ever be served is created by running it.

`modal run` mounts the local `app/`, so what this measures is the working tree
rather than whatever is deployed. That is the point: it closes the edit → measure
loop without a deploy in the middle of it.
"""

from __future__ import annotations

import json

import modal

from modal_app import worker_image, worker_secrets

app = modal.App("rosetta-dechorder-analyze-once")
image = worker_image.add_local_python_source("modal_app")


@app.function(image=image, secrets=worker_secrets, timeout=1800, memory=4096)
def run(video_id: str) -> str:
    from app.analysis import engines
    from app.analysis.fetch import build_source
    from app.analysis.pipeline import analyze
    from app.chords import NORMAL
    from app.config import load_settings
    from app.store import build_store

    settings = load_settings()
    engines.register_builtins()

    # A throwaway SQLite in this container's own filesystem. `analyze` needs a
    # store only for `gate()`'s blocklist check, and this one dies with the call
    # — so the "stores nothing" claim holds against the real store, not against
    # a mock that happens not to be wired up.
    store = build_store(settings)

    outcome = analyze(
        video_id=video_id,
        settings=settings,
        store=store,
        source=build_source(settings),
        chord_engine=engines.build_chord_engine(settings),
        beat_tracker=engines.build_beat_tracker(settings),
        onset_detector=engines.build_onset_detector(settings),
        structure_probe=engines.build_structure_probe(settings),
    )

    song = outcome.songs.get(NORMAL) if isinstance(outcome.songs, dict) else None
    sync = outcome.sync
    return json.dumps({
        "videoId": video_id,
        "song": song,
        "sync": json.loads(sync.model_dump_json()) if sync is not None else None,
        "lowConfidence": outcome.low_confidence,
    })


@app.local_entrypoint()
def main(video_id: str):
    print(run.remote(video_id))
