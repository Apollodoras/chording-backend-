"""Delete stored analyses from the deployed store. Dry run unless told otherwise.

    modal run scripts/sweep_catalog.py                      # list, delete nothing
    modal run scripts/sweep_catalog.py --confirm            # purge the catalog
    modal run scripts/sweep_catalog.py --confirm --include-uploads

Every row in `chord_maps` is output of *this* pipeline, so a chart stored before
a correctness fix is not stale data, it is a wrong answer being served. Until the
analysis is trusted, the catalog is a cache to be dropped rather than an asset to
be migrated — and this script is the drop, kept in the repo because it is
expected to run again after the next fix.

Three things it is deliberate about:

**It purges through `store.purge`,** not with a `DELETE` of its own. That is the
§3 takedown primitive: it cascades to `jobs` and `job_followers`, returns the row
counts so the cascade can be *verified* rather than assumed, and writes an
`audit_log` entry per video. The audit table is never touched — the record of a
deletion has to outlive the thing deleted, and that is as true of a sweep as of a
DMCA purge.

**Uploads are excluded by default.** A row with an `owner_uid` is somebody's own
recording, content-addressed and readable only by them (`ChordMap.is_private`).
It is not catalog material and never was — `list_catalog` filters it out — so
"sweep the catalog" does not reach it without `--include-uploads`. The analysis
is just as obsolete; whose data it is, is the difference.

**The dry run is the default and prints the whole target.** A sweep that reports
"deleted 14" without ever saying which 14 is the one you cannot check afterwards.
"""

from __future__ import annotations

import json

import modal

from modal_app import api_image, api_secrets

app = modal.App("rosetta-dechorder-sweep")

# `api_image` is the container that already talks to the store, and this needs
# nothing else — no ffmpeg, no engines. `modal_app` itself is mounted for the
# same reason `seed_catalog.py` mounts it: this module imports it.
sweep_image = api_image.add_local_python_source("modal_app")


@app.function(image=sweep_image, secrets=api_secrets, timeout=600)
def sweep(confirm: bool, include_uploads: bool, actor: str, reason: str) -> str:
    from app.config import load_settings
    from app.store import build_store

    store = build_store(load_settings())

    # Straight SQL rather than `list_catalog`, which is the *player's* view: it
    # hides uploads and paginates. What has to be enumerated here is every row
    # that exists, including the ones the catalog would never show.
    with store._cursor() as cur:
        cur.execute(
            "SELECT video_id, difficulty, title, owner_uid, analyzed_at, "
            "       engine_chords, engine_beats, low_confidence "
            "FROM chord_maps ORDER BY analyzed_at DESC"
        )
        rows = [
            {
                "videoId": r[0], "difficulty": r[1], "title": r[2],
                "ownerUid": r[3], "analyzedAt": r[4],
                "engines": f"{r[5]} / {r[6]}", "lowConfidence": bool(r[7]),
            }
            for r in cur.fetchall()
        ]

    public = [r for r in rows if not r["ownerUid"]]
    private = [r for r in rows if r["ownerUid"]]
    targets = rows if include_uploads else public

    report = {
        "confirm": confirm,
        "includeUploads": include_uploads,
        "totalRows": len(rows),
        "catalogRows": len(public),
        "uploadRows": len(private),
        "targeted": len(targets),
        "rows": targets,
        "spared": [] if include_uploads else private,
    }

    if not confirm:
        report["result"] = "DRY RUN — nothing deleted"
        return json.dumps(report)

    # One purge per *video*, not per row: `purge` deletes every difficulty of a
    # video in one statement, so calling it per row would report a second pass
    # that matched nothing and read as a failed cascade.
    totals = {"maps": 0, "jobs": 0, "videos": 0}
    for video_id in dict.fromkeys(r["videoId"] for r in targets):
        counts = store.purge(video_id, actor=actor, reason=reason)
        totals["maps"] += counts["maps"]
        totals["jobs"] += counts["jobs"]
        totals["videos"] += 1

    with store._cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM chord_maps")
        remaining = cur.fetchone()[0]

    report["deleted"] = totals
    report["remainingRows"] = remaining
    report["catalogVersion"] = store.catalog_version()
    report["result"] = "purged"
    return json.dumps(report)


@app.local_entrypoint()
def main(confirm: bool = False, include_uploads: bool = False,
         actor: str = "sweep", reason: str = "obsolete analyses — pre-fix pipeline output"):
    report = json.loads(sweep.remote(confirm, include_uploads, actor, reason))

    print(f"\n{report['totalRows']} stored analyses "
          f"({report['catalogRows']} catalog, {report['uploadRows']} upload)")
    print(f"targeting {report['targeted']}\n")
    for row in report["rows"]:
        flag = " [lowConfidence]" if row["lowConfidence"] else ""
        owner = " [upload]" if row["ownerUid"] else ""
        print(f"  {row['analyzedAt'][:19]}  {row['videoId']:<12} {row['difficulty']:<7} "
              f"{(row['title'] or '')[:58]}{owner}{flag}")
    for row in report["spared"]:
        print(f"  SPARED (upload)  {row['videoId']:<20} {(row['title'] or '')[:50]}")

    print()
    if report["result"] == "DRY RUN — nothing deleted":
        print("DRY RUN — nothing deleted. Re-run with --confirm to purge.")
    else:
        d = report["deleted"]
        print(f"purged {d['videos']} videos: {d['maps']} maps, {d['jobs']} jobs")
        print(f"remaining rows in chord_maps: {report['remainingRows']}")
        print(f"catalog version now: {report['catalogVersion']}")
