"""Real videos, real audio, the deployed worker image.

    modal run scripts/real_song_check.py                     # the default four
    modal run scripts/real_song_check.py --songs canon-fingerstyle
    modal run scripts/real_song_check.py --songs all
    modal run scripts/real_song_check.py --songs isophonics   # see the note below

`scripts/smoke.py` proves the API container's config. `scripts/worker_check.py`
proves the worker's engines run — but on *synthesized* audio, with the fetch
stage stubbed out entirely. Neither touches the thing that had never once
succeeded here: YouTube serving audio to a Modal datacentre IP.

So this is the third gate, and the only one that runs the whole path —
`probe` → `gate` → `decode` → beats → chords → onsets → `assemble` — on a real
recording, inside the deployed image, with the real worker secret.

## Why not the Isophonics corpus

`bench/corpus.json` has YouTube ids with real ground truth, which would be the
obvious thing to check against. They are all unfetchable from Modal: every one
of the official Beatles uploads answers a datacentre IP with the bot check, in
every player client, **with or without cookies**. Ordinary uploads — covers,
backing tracks, small channels — resolve from the same IP in the same second, so
this is not a blanket IP ban but per-video enforcement on label-owned music.
They are kept under `--songs isophonics` so the situation stays checkable, not
because they are expected to pass.

The default set is therefore chosen for *verifiability by eye* instead: a 12-bar
blues is E/A/B and nothing else, and Canon in D is a fixed eight-chord cycle. If
BTC returns those, it is working on real audio. `expect` below is a set of chord
roots that should dominate the result — deliberately weaker than the §8 accuracy
gate, which is the bench's job, on local audio, where alignment is controlled.
"""

import json

import modal

from modal_app import worker_image, worker_secrets

app = modal.App("rosetta-dechorder-realsong")

# Same reason as `worker_check.py`: the container entrypoint imports this module,
# whose first act is to import `modal_app`, which `worker_image` does not mount.
check_image = worker_image.add_local_python_source("modal_app")

SONGS = {
    # -- fetchable, and chosen so the chord output can be judged by eye --------
    "blues-in-e": {
        "videoId": "36X3wecT2z8",
        "title": "Blues in E (90bpm) : Backing track",
        "expect": {"E", "A", "B"},
        "note": "12-bar blues. Three chords, fixed order — the cleanest possible read.",
    },
    "canon-fingerstyle": {
        "videoId": "85Sqw6FTxm4",
        "title": "Canon in D - Pachelbel (Fingerstyle Guitar Cover)",
        "expect": {"D", "A", "B", "F#", "G"},
        "note": "D A Bm F#m G D G A, looped. Solo guitar, so no mix to hide behind.",
    },
    "canon-rock": {
        "videoId": "2xjJXT0C0X4",
        "title": "Canon Rock",
        "expect": {"D", "A", "B", "F#", "G"},
        "note": "Same cycle under a distorted band mix — the harder version of the above.",
    },
    "hallelujah-cover": {
        "videoId": "eG-ZGPikL6I",
        "title": "Hallelujah by Leonard Cohen - Noah Guthrie Cover",
        "expect": {"C", "A", "F", "G", "E"},
        "note": "Vocal + guitar. Tests a real arrangement rather than an instrumental.",
    },
    # -- the Isophonics ids, kept for the record. All bot-checked. -------------
    "iso-let-it-be": {"videoId": "CGj85pVzRJs", "title": "Let It Be (Isophonics)",
                      "expect": {"C", "G", "A", "F"}, "isophonics": True,
                      "truth": {"tempo": 69.85, "meter": "4/4", "chords": 159}},
    "iso-norwegian-wood": {"videoId": "Y_V6y1ZCg_8", "title": "Norwegian Wood (Isophonics)",
                           "expect": {"E", "D", "A"}, "isophonics": True,
                           "truth": {"tempo": 178.04, "meter": "3/4", "chords": 43}},
    "iso-michelle": {"videoId": "WoBLi5eE-wY", "title": "Michelle (Isophonics)",
                     "expect": {"F", "C", "B", "D"}, "isophonics": True,
                     "truth": {"tempo": 117.42, "meter": "4/4", "chords": 92}},
}

DEFAULT_SONGS = ("blues-in-e", "canon-fingerstyle", "canon-rock", "hallelujah-cover")
ISOPHONICS = tuple(name for name, song in SONGS.items() if song.get("isophonics"))


@app.function(image=check_image, secrets=worker_secrets, timeout=1800, memory=4096)
def analyze_one(name: str) -> dict:
    """One real video, all the way through, inside the deployed worker image."""
    import os
    import time
    import traceback

    from app.config import load_settings
    from app.analysis import engines, pipeline
    from app.analysis.fetch import build_source
    from app.analysis.scratch import assert_clean
    from app.store import build_store

    song = SONGS[name]
    video_id = song["videoId"]
    report: dict = {"song": name, "videoId": video_id, "wanted": song["title"]}

    settings = load_settings()
    report["cookiesPresent"] = bool(os.environ.get("CHORDS_YTDLP_COOKIES_CONTENT"))

    engines.register_builtins()
    store = build_store(settings)
    source = build_source(settings)
    if source is None:
        report["ERROR"] = "no fetch source in this image (yt-dlp or ffmpeg missing)"
        return report

    stages: list[str] = []

    started = time.monotonic()
    try:
        # Split out from `analyze` so a bot check is attributable to `probe`
        # rather than surfacing as a generic failure of the whole pipeline.
        probe_started = time.monotonic()
        meta = source.probe(video_id)
        report["probeSeconds"] = round(time.monotonic() - probe_started, 1)
        report["title"] = meta.title
        report["durationS"] = round(meta.duration_s, 1)
        report["fetch"] = "ok"
    except Exception as exc:  # noqa: BLE001 — reporting, not handling
        report["fetch"] = "FAILED"
        report["ERROR"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()[-800:]
        return report

    try:
        outcome = pipeline.analyze(
            video_id=video_id,
            settings=settings,
            store=store,
            source=source,
            chord_engine=engines.build_chord_engine(settings),
            beat_tracker=engines.build_beat_tracker(settings),
            onset_detector=engines.build_onset_detector(settings),
            progress=lambda status, fraction: stages.append(f"{status}@{fraction:.2f}"),
        )
    except Exception as exc:  # noqa: BLE001
        report["analysis"] = "FAILED"
        report["ERROR"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()[-800:]
        return report

    report["analysis"] = "ok"
    report["wallSeconds"] = round(time.monotonic() - started, 1)
    report["realtimeRatio"] = round(report["wallSeconds"] / max(meta.duration_s, 1), 2)
    report["stages"] = stages
    report["engines"] = {"chords": outcome.engine_chords, "beats": outcome.engine_beats}
    report["lowConfidence"] = outcome.low_confidence
    report["difficulties"] = sorted(outcome.songs)
    report["hasSync"] = outcome.sync is not None

    # Read everything below off the wire payload the client would actually
    # receive, not off internal state, so a field that fails to serialize shows
    # up here rather than on someone's phone.
    wire = outcome.songs.get("intermediate") or outcome.songs[sorted(outcome.songs)[0]]
    report["payloadKeys"] = sorted(wire)
    report["payloadBytes"] = len(json.dumps(wire))

    # CompositionPayload v2, and reading it wrong is easy in two different ways.
    # There is no flat span list, so `wire["chords"]` finds nothing. And in bars
    # mode a section's `chordNames` is deliberately **only the first bar** — the
    # v1 fallback that `compile.py` fills so a section is never one field from
    # the "empty chordNames ⇒ silently dropped" failure. Reading it as the whole
    # progression reported `1 chord` for a 12-bar blues that the engine had in
    # fact called correctly. The chords are in `bars` when bars mode is on, and
    # in `chordNames` only when it is not.
    sections = ((wire.get("arrangement") or {}).get("sections")) or []
    labels: list[str] = []
    bars_mode = False
    for section in sections:
        if not isinstance(section, dict):
            continue
        bars = section.get("bars") or []
        if bars:
            bars_mode = True
            for bar in bars:
                if not isinstance(bar, dict):
                    continue
                for span in bar.get("chordSpans") or []:
                    if isinstance(span, dict) and span.get("chordName"):
                        labels.append(span["chordName"])
        else:
            labels.extend(n for n in (section.get("chordNames") or []) if n)

    report["sectionCount"] = len(sections)
    report["sectionNames"] = [s.get("name") or s.get("id") or "?"
                              for s in sections if isinstance(s, dict)][:8]
    report["barsMode"] = bars_mode
    report["barCount"] = sum(len(s.get("bars") or []) for s in sections if isinstance(s, dict))

    # The song-level flat summary. Always present, always the whole song, so it
    # is the honest cross-check on the walk above.
    flat = [n for n in (wire.get("chordNames") or []) if n]
    report["flatChordNames"] = flat[:20]
    report["flatChordCount"] = len(flat)
    if not labels:
        labels = flat

    report["chordCount"] = len(labels)
    report["firstChords"] = labels[:16]

    # Root = the note name only, so "Em", "E", "E7" and "E/G#" all count as E.
    # These are *rendered* names ("Em"), not engine labels ("E:min"), so
    # splitting on ":" leaves the quality attached and reports "Em" and "E" as
    # two different roots — which made a blues that is E/A/B look like it had
    # nothing in common with E/A/B.
    import re

    def root_of(name: str) -> str:
        match = re.match(r"^([A-G][#b]?)", name)
        return match.group(1) if match else name

    roots = [root_of(l.split("/")[0]) for l in labels]
    distinct = {}
    for root in roots:
        distinct[root] = distinct.get(root, 0) + 1
    report["rootHistogram"] = dict(sorted(distinct.items(), key=lambda kv: -kv[1])[:8])

    expect = song["expect"]
    in_expected = sum(count for root, count in distinct.items() if root in expect)
    report["expectedRoots"] = sorted(expect)
    report["shareInExpectedRoots"] = round(in_expected / len(roots), 3) if roots else 0.0

    for key in ("bpm", "tempo", "timeSignature", "meter", "key", "keyName"):
        if key in wire:
            report[key] = wire[key]
    if song.get("truth"):
        report["truth"] = song["truth"]

    assert_clean(settings.scratch_root)
    report["scratchClean"] = True
    return report


def _verdict(reports: list[dict]) -> int:
    """Fatal: the pipeline did not produce a playable result. Reported but not
    fatal: which chords it found. This gate answers "does the deployment work on
    real audio", and conflating that with accuracy would make it fire every time
    a cover was in an unexpected key."""
    failures, notes = [], []

    for report in reports:
        name = report["song"]
        if report.get("ERROR"):
            failures.append(f"{name}: {report['ERROR']}")
            continue
        if not report.get("chordCount"):
            failures.append(f"{name}: analysis returned no chords")
            continue

        share = report.get("shareInExpectedRoots", 0)
        verdict = "strong" if share >= 0.7 else "partial" if share >= 0.4 else "WEAK"
        notes.append(f"{name}: {report['chordCount']} chords, "
                     f"{share:.0%} on expected roots {report['expectedRoots']} — {verdict}")

        bpm = report.get("bpm") or report.get("tempo")
        if bpm:
            line = f"{name}: {bpm:.1f} bpm"
            truth = (report.get("truth") or {}).get("tempo")
            if truth:
                ratio = bpm / truth
                if 0.95 <= ratio <= 1.05:
                    line += f" — matches truth {truth:.1f}"
                elif 0.45 <= ratio <= 0.55 or 1.9 <= ratio <= 2.1:
                    line += f" — {ratio:.1f}x truth {truth:.1f}, half/double time"
                else:
                    line += f" — vs truth {truth:.1f}"
            notes.append(line)

        if report.get("realtimeRatio"):
            notes.append(f"{name}: {report['wallSeconds']}s wall for "
                         f"{report['durationS']}s audio ({report['realtimeRatio']}x realtime)")
        if report.get("lowConfidence"):
            notes.append(f"{name}: flagged lowConfidence")
        if not report.get("hasSync"):
            notes.append(f"{name}: no sync sidecar (beat grid too weak to align)")

    for note in notes:
        print(f"[ note ] {note}")
    for failure in failures:
        print(f"[ FAIL ] {failure}")

    if failures:
        print(f"\nFAILED — {len(failures)} of {len(reports)} song(s)")
        return 1
    print(f"\nAll {len(reports)} song(s) analyzed end to end on real audio")
    return 0


@app.local_entrypoint()
def main(songs: str = ",".join(DEFAULT_SONGS), attempts: int = 6):
    """`attempts` is not flakiness-papering, it is the measured shape of the
    problem: YouTube's bot check on Modal is **per egress IP**, and a fan-out of
    10 containers across 10 distinct IPs resolved the same three ordinary
    uploads on 2 of them — 20%, identically with and without cookies. One
    container is therefore a coin toss with bad odds; six independent ones make
    it ~74%. Each attempt lands on its own container, so it is its own IP.

    An attempt that meets the bot check returns after the probe, before any
    audio is fetched, so the losing attempts cost seconds rather than a decode.
    """
    import sys

    if songs == "all":
        names = sorted(SONGS)
    elif songs == "isophonics":
        names = list(ISOPHONICS)
    else:
        names = [s.strip() for s in songs.split(",") if s.strip()]

    unknown = [n for n in names if n not in SONGS]
    if unknown:
        sys.exit(f"unknown song(s): {', '.join(unknown)}\nknown: {', '.join(sorted(SONGS))}")

    print(f"Analyzing {len(names)} real video(s) in the deployed worker image, "
          f"up to {attempts} attempts each (one container, hence one IP, per "
          f"attempt):\n  {', '.join(names)}\n")

    jobs = [name for name in names for _ in range(attempts)]
    results = list(analyze_one.map(jobs))

    # First success per song; failing that, the last failure, so the report says
    # what went wrong rather than going quiet.
    best: dict[str, dict] = {}
    blocked: dict[str, int] = {name: 0 for name in names}
    for report in results:
        name = report["song"]
        if report.get("fetch") == "FAILED":
            blocked[name] += 1
        if report.get("analysis") == "ok":
            # A success always displaces a failure, whatever order they land in.
            # Doing this with `setdefault` alone silently keeps whichever report
            # arrived first, which is usually a bot check.
            if best.get(name, {}).get("analysis") != "ok":
                best[name] = report
        else:
            best.setdefault(name, report)

    reports = [best[name] for name in names]
    for report in reports:
        report["attemptsBlocked"] = f"{blocked[report['song']]}/{attempts}"

    print(json.dumps(reports, indent=2, default=str))
    reached = sum(1 for r in reports if r.get("analysis") == "ok")
    print(f"\nfetch reached YouTube for {reached}/{len(reports)} song(s) "
          f"within {attempts} attempts")
    sys.exit(_verdict(reports))
