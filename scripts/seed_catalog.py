"""Seed the catalog with known songs, then check the charts against ground truth.

    modal run scripts/seed_catalog.py                      # the whole seed set
    modal run scripts/seed_catalog.py --songs zombie,hey-joe
    modal run scripts/seed_catalog.py --songs controls     # the two already-verified ids

`scripts/real_song_check.py` is the deploy gate: it asks "did the pipeline run at
all on real audio", and its `expect` is deliberately a weak set of chord *roots*.
This is a different question, and a harder one: **is the chart right?** So it

1. runs the real `run_job`, which means the maps land in `chord_maps` exactly as
   a user's analysis would — the catalog is genuinely seeded, not simulated;
2. reads each chart back **out of the store**, not off in-memory state, so what
   is graded is what the client would be served;
3. grades it against ground truth written down per song below — key, meter,
   tempo, the chord vocabulary, and the progression.

## Where the ground truth comes from

Published chord transcriptions of each recording, at *sounding* pitch. That last
word is the whole trap: half of these are played with a capo, and the chart is
named for the shape, not the sound. Wonderwall is "Em7 G Dsus4 A7sus4" in every
songbook and comes out of a spectrogram as F#m A E B, because the capo is on 2.
Grading the shapes would fail a correct engine on all five chords.

`vocabulary` is the set of chords the song is built from; `share` measures how
much of the chart lands inside it, which is the number to read. `progression` is
the defining cycle, checked as an ordered subsequence of the chart so that a
song whose loop is right still passes when the engine hears an extra passing
chord. `tempo` is the recording's, and half/double time is reported as such
rather than failed — a tracker that hears 140 in a 70bpm ballad is not wrong
about the music, only about which layer of it to call the beat.

Covers, backing tracks and karaoke uploads, deliberately: the Isophonics ids are
label-owned and answer a datacentre IP with the bot check (see
`real_song_check.py`), and these resolve. It also means the *arrangement* may not
match the famous one — a cover can transpose, drop a bridge or vamp an outro —
so a key mismatch is reported with the cover's own key rather than treated as
proof the engine is broken.
"""

import json
import re

import modal

from modal_app import worker_image, worker_secrets

app = modal.App("rosetta-dechorder-seed")

# The container entrypoint imports this module, which imports `modal_app` —
# which `worker_image` does not mount. Same reason as `real_song_check.py`.
seed_image = worker_image.add_local_python_source("modal_app")

# The uid every seeded job is filed under. Not a real account: `jobs.uid` has no
# foreign key, and the point is that a seeded row is identifiable later — for a
# re-seed, or for telling "we put this here" apart from "a user asked for this".
SEED_UID = "seed:catalog"

SEEDS = {
    # -- three chords or fewer: if these are wrong, nothing else matters -------
    "sweet-home-alabama": {
        "videoId": "b14W-_3QsaA",
        "title": "Sweet Home Alabama [D] 97bpm — guitar backing track",
        "truth": {
            "key": "D",
            "meter": "4/4",
            "tempo": 97.0,
            "vocabulary": ["D", "C", "G"],
            "progression": ["D", "C", "G"],
            "note": "D–C–G, over and over, start to finish. The floor of the test set.",
        },
    },
    "hey-joe": {
        "videoId": "Di-x2luOibk",
        "title": "Hey Joe backing track, 60bpm",
        "truth": {
            "key": "E",
            "meter": "4/4",
            "tempo": 60.0,
            "vocabulary": ["C", "G", "D", "A", "E"],
            "progression": ["C", "G", "D", "A", "E"],
            "note": "A cycle of fourths, C to E, looped for the whole song. The "
                    "uploader put the progression in the title, which is why this "
                    "id was picked over a cover.",
        },
    },
    "knockin-heavens-door": {
        "videoId": "dTxOcc85h8s",
        "title": "Knockin' on Heaven's Door — backing track (GN'R arrangement)",
        "truth": {
            "key": "G",
            "meter": "4/4",
            "tempo": 74.0,
            "vocabulary": ["G", "D", "Am", "C"],
            "progression": ["G", "D", "Am", "G", "D", "C"],
            "note": "Two phrases that alternate: G D Am / G D C. The Am-vs-C swap "
                    "in bar 3 is the only thing to get right, and it is exactly the "
                    "kind of relative-minor confusion a template engine loses.",
        },
    },
    "zombie": {
        "videoId": "-oOo4imobAo",
        "title": "The Cranberries — Zombie, acoustic guitar cover",
        "truth": {
            "key": "Bm",
            "meter": "4/4",
            "tempo": 84.0,
            "vocabulary": ["Bm", "G", "D", "A"],
            "progression": ["Bm", "G", "D", "A"],
            "note": "One four-chord loop for the entire song, verse and chorus "
                    "alike. Structure is therefore the interesting output here, "
                    "not harmony: the sections have to be found from dynamics, "
                    "because the chords never change.\n"
                    "THIS COVER IS UP A FIFTH. The Cranberries record is Em–C–G–D "
                    "and every published chart says so; this upload plays it "
                    "Bm–G–D–A. Graded as Em it scores 50% and looks like the "
                    "engine failed, which is what it did say here first. The "
                    "recording settles it independently of anything in this repo: "
                    "a Krumhansl correlation over a CQT chroma puts C as the "
                    "*least* present pitch class of the twelve (0.042), and Em "
                    "spends a quarter of the song on C. The chart was right and "
                    "the truth entry was wrong — which is the failure mode this "
                    "whole file has to be most careful about, since a cover is "
                    "under no obligation to be in the original's key.",
        },
    },
    # -- four or five chords, with a real verse/chorus distinction -------------
    "stand-by-me": {
        "videoId": "isN_guhOaXc",
        "title": "Stand By Me — Ben E. King — backing track in A",
        "truth": {
            "key": "A",
            "meter": "4/4",
            "tempo": 118.0,
            "vocabulary": ["A", "F#m", "D", "E"],
            "progression": ["A", "F#m", "D", "E", "A"],
            "note": "The 50s doo-wop turnaround, I–vi–IV–V, two bars each. F#m is "
                    "the one at risk: it shares two notes with A.",
        },
    },
    "house-of-the-rising-sun": {
        "videoId": "QZQcGxATa9g",
        "title": "The Animals — House of the Rising Sun — acoustic cover (Kfir Ochaion)",
        "truth": {
            "key": "Am",
            "meter": "6/8",
            "tempo": 78.0,
            "vocabulary": ["Am", "C", "D", "F", "E"],
            "progression": ["Am", "C", "D", "F", "Am", "C", "E"],
            "note": "The meter case, and the reason it is in the set: this is 6/8 "
                    "(or a triplet 12/8 read of 4/4), not 4/4. A tracker that hears "
                    "it in four will still get every chord right and lay the bars "
                    "out wrong — which is a failure the chord score alone cannot see.\n"
                    "In the event it does not get that far: this is the one song in "
                    "the set that produces NO chart at all. The beat tracker locks "
                    "onto the eighth-note triplets rather than the dotted-quarter "
                    "pulse, reports 231 BPM, and the 40–220 guard rejects the "
                    "analysis as `tempo_unreadable`. 231 is very close to 3x the "
                    "77 BPM this is actually in, so the tracker is hearing the "
                    "subdivision, not a different song — and `tempoOctaveShift` "
                    "already exists for the 2x case. A compound-meter recording is "
                    "therefore not slightly wrong here, it is unanalyzable, and "
                    "that is the finding this entry exists to keep visible.",
        },
    },
    "wish-you-were-here": {
        "videoId": "M7Q-SwUqNo4",
        "title": "Pink Floyd — Wish You Were Here — acoustic cover",
        "truth": {
            "key": "G",
            "meter": "4/4",
            "tempo": 60.0,
            "vocabulary": ["Em", "G", "A", "C", "D", "Am"],
            "progression": ["Em", "G", "Em", "G", "Em", "A", "Em", "A", "G"],
            "note": "Intro/verse is the Em–G–Em–A figure; the chorus is C D Am G "
                    "and then C D G. Two genuinely different sections with "
                    "different vocabularies — so this is the honest test of "
                    "whether form clustering is finding anything.",
        },
    },
    "wonderwall": {
        "videoId": "iJQ-8eMoBqU",
        "title": "Oasis — Wonderwall — acoustic cover (Dave Winkler)",
        "truth": {
            "key": "F#m",
            "meter": "4/4",
            "tempo": 87.0,
            "vocabulary": ["F#m", "A", "E", "B", "D"],
            "progression": ["F#m", "A", "E", "B"],
            "note": "CAPO 2. Every chart on the internet says Em7 G Dsus4 A7sus4; "
                    "that is the shape. The sound is F#m A E B, and the sound is "
                    "what a spectrogram gets. If this comes back as Em/G/D/A the "
                    "engine is not wrong — the cover is playing without the capo.",
        },
    },
    # -- the hard end: seven chords, a key change of mode, and jazz ------------
    "hotel-california": {
        "videoId": "vjYG8Y8C1vs",
        "title": "Hotel California — guitar backing track (vocals, bass, drums)",
        "truth": {
            "key": "Bm",
            "meter": "4/4",
            "tempo": 75.0,
            "vocabulary": ["Bm", "F#", "A", "E", "G", "D", "Em"],
            "progression": ["Bm", "F#", "A", "E", "G", "D", "Em", "F#"],
            "note": "Eight chords, one per bar, never repeating within the cycle — "
                    "so there is no vamp for the engine to coast on. F# major (not "
                    "minor) in a B minor song is the harmonic-minor V that a "
                    "diatonic key model will try to talk it out of.",
        },
    },
    "autumn-leaves": {
        "videoId": "05le0tbWLYU",
        "title": "Autumn Leaves — backing track, Bb, 112bpm",
        "truth": {
            "key": "Bb",
            "meter": "4/4",
            "tempo": 112.0,
            "vocabulary": ["Cm", "F", "Bb", "Eb", "Am", "D", "Gm"],
            "progression": ["Cm", "F", "Bb", "Eb", "Am", "D", "Gm"],
            "note": "Jazz, and the only seventh-chord song here: ii–V–I in Bb "
                    "then ii°–V–i in Gm. Everything is a 7th or a m7b5, so this is "
                    "where §12.2's normalization into the app's grammar earns its "
                    "keep — Am7b5 has to survive as something playable and still "
                    "read as A.",
        },
    },
    # -- controls: already known to fetch and already judged by eye ------------
    "blues-in-e": {
        "videoId": "36X3wecT2z8",
        "control": True,
        "title": "Blues in E (90bpm) — backing track",
        "truth": {
            "key": "E",
            "meter": "4/4",
            "tempo": 90.0,
            "vocabulary": ["E", "A", "B"],
            "progression": ["E", "A", "E", "B", "A", "E"],
            "note": "12-bar blues. Carried over from `real_song_check.py`, where it "
                    "is known to fetch — so if this one fails, the failure is the "
                    "deployment and not the seed set.\n"
                    "It is also the one song here the pipeline gets WRONG, and the "
                    "grading resolution is what exposes it: the roots are perfect "
                    "(E, A and B account for every chord) while the qualities flip "
                    "between major and minor on the same root — E and Em both "
                    "appear, in a song that has no minor chord in it. Measured on "
                    "the recording, the major third wins on all three (G# 0.080 vs "
                    "G 0.071; C# 0.064 vs C 0.061; D# 0.085 vs D 0.057), so these "
                    "are errors and not an unusual arrangement. The margins say "
                    "why: a blues plays the minor third *as a blue note* over a "
                    "dominant chord, so the pitch that distinguishes E from Em is "
                    "genuinely in the audio. Root accuracy alone would score this "
                    "100% and report nothing.",
        },
    },
    "canon-in-d": {
        "videoId": "85Sqw6FTxm4",
        "control": True,
        "title": "Canon in D — Pachelbel — fingerstyle guitar cover",
        "truth": {
            "key": "D",
            "meter": "4/4",
            "tempo": 64.0,
            "vocabulary": ["D", "A", "Bm", "F#m", "G", "Em"],
            "progression": ["D", "A", "Bm", "F#m", "G", "D", "G", "A"],
            "note": "A fixed eight-chord ground bass, repeated for the whole piece, "
                    "solo guitar with no mix to hide behind. Also carried over from "
                    "`real_song_check.py`.",
        },
    },
}

CONTROLS = tuple(name for name, s in SEEDS.items() if s.get("control"))
DEFAULT = tuple(SEEDS)


# --------------------------------------------------------------------------
# Reading a chart back out of the store
# --------------------------------------------------------------------------

def _root(name: str) -> str:
    """Note name only: E, Em, E7 and E/G# all collapse to E."""
    match = re.match(r"^([A-G][#b]?)", name.split("/")[0])
    return match.group(1) if match else name


def _triad(name: str) -> str:
    """Root plus major/minor only — the grading resolution.

    Quality beyond the third is thrown away on purpose. §12.2 already decided
    that `Cmaj9` and `Cmaj7` are the same chord to this app because the player
    voices them identically, so grading them apart would score a difference the
    product does not have. Major/minor is kept because that difference is
    audible in every arrangement, and confusing a chord with its relative minor
    is the specific mistake worth catching.
    """
    head = name.split("/")[0]
    match = re.match(r"^([A-G][#b]?)(.*)$", head)
    if not match:
        return head
    root, rest = match.groups()
    minor = rest.startswith(("m", "min")) and not rest.startswith("maj")
    dim = rest.startswith(("dim", "°"))
    return f"{root}m" if (minor or dim) else root


def _dedupe(seq):
    """Collapse runs. A chord held for four bars is one chord in a progression."""
    out = []
    for item in seq:
        if not out or out[-1] != item:
            out.append(item)
    return out


def _chart(song: dict) -> dict:
    """Pull the playable chart out of a CompositionPayload v2.

    Two ways to read this wrong, both of which have already produced a wrong
    number in this repo (see `real_song_check.py`): there is no flat span list at
    the top level, and in bars mode a section's `chordNames` holds **only the
    first bar** — it is the v1 fallback `compile.py` fills so that no section is
    ever one field away from being silently dropped by the importer. Reading it
    as the section's whole progression reported "1 chord" for a correct 12-bar
    blues.
    """
    sections_wire = ((song.get("arrangement") or {}).get("sections")) or []
    sections, flat = [], []
    for section in sections_wire:
        if not isinstance(section, dict):
            continue
        labels = []
        bars = section.get("bars") or []
        if bars:
            for bar in bars:
                if not isinstance(bar, dict):
                    continue
                for span in bar.get("chordSpans") or []:
                    if isinstance(span, dict) and span.get("chordName"):
                        labels.append(span["chordName"])
        else:
            labels = [n for n in (section.get("chordNames") or []) if n]
        flat.extend(labels)
        # `kindRaw` before `id`: a section that carries no `name` is not
        # anonymous, it is *unnamed*, and its structural role still lives in
        # `kindRaw` (verse/chorus/…). Falling straight through to the UUID
        # prints a hex string where the form should be.
        sections.append({
            "name": section.get("name") or section.get("kindRaw") or section.get("id") or "?",
            "bars": len(bars),
            "chords": len(labels),
            "progression": _dedupe(_triad(l) for l in labels)[:16],
        })

    # `chordNames` at song level is always the whole song, so it is the honest
    # cross-check on the walk above rather than a substitute for it.
    song_flat = [n for n in (song.get("chordNames") or []) if n]
    if not flat:
        flat = song_flat

    triads = [_triad(l) for l in flat]
    histogram = {}
    for triad in triads:
        histogram[triad] = histogram.get(triad, 0) + 1

    return {
        "sections": sections,
        "sectionCount": len(sections),
        "chordCount": len(flat),
        "distinctChords": sorted(set(flat)),
        "triadHistogram": dict(sorted(histogram.items(), key=lambda kv: -kv[1])),
        "sequence": _dedupe(triads),
        "firstChords": flat[:24],
        "songLevelFlatCount": len(song_flat),
    }


def _contains_cycle(sequence, cycle) -> bool:
    """Is `cycle` an ordered subsequence of `sequence`, somewhere?

    Subsequence rather than a contiguous run, because a correct chart routinely
    has a passing chord or a held anticipation between two chords of the cycle,
    and demanding adjacency would fail it for being more detailed than the
    songbook. The cycle still has to appear **in order** and inside one pass, so
    this cannot be satisfied by a chart that merely contains the right chords.
    """
    if not cycle:
        return False
    for start in range(len(sequence)):
        index = 0
        for item in sequence[start:]:
            if item == cycle[index]:
                index += 1
                if index == len(cycle):
                    return True
        # Only the first alignment can succeed for a given start, so no early
        # break here would help; the loop above already consumed the tail.
    return False


# --------------------------------------------------------------------------
# One song, in the deployed worker image, on the real store
# --------------------------------------------------------------------------

@app.function(image=seed_image, secrets=worker_secrets, timeout=1800, memory=4096)
def seed_one(name: str) -> dict:
    import time
    import traceback
    import uuid

    from app.chords import DIFFICULTIES, NORMAL
    from app.config import load_settings
    from app.analysis import engines
    from app.analysis.fetch import build_source
    from app.analysis.scratch import assert_clean
    from app.jobs import OUTCOME_READY, run_job
    from app.store import build_store

    seed = SEEDS[name]
    video_id = seed["videoId"]
    report = {"song": name, "videoId": video_id, "wanted": seed["title"]}

    settings = load_settings()
    engines.register_builtins()
    store = build_store(settings)
    source = build_source(settings)
    if source is None:
        report["ERROR"] = "no fetch source in this image (yt-dlp or ffmpeg missing)"
        return report

    # `normal`, not "intermediate" — the tiers are `easy`/`normal`/`hard`
    # (`app.chords.DIFFICULTIES`). Naming a tier that does not exist stores the
    # analysis fine and then reads back nothing, which is how this script first
    # reported "job reported ready but no row landed in chord_maps" against a
    # run that had in fact filed all three rows.
    job_id = f"seed-{uuid.uuid4().hex[:12]}"
    store.create_job(job_id=job_id, uid=SEED_UID, video_id=video_id,
                     difficulty=NORMAL)

    started = time.monotonic()
    try:
        # `may_retry_elsewhere=False`: this attempt is the last one *this*
        # container will make, and the fan-out in `main` is what places the next
        # one on a fresh IP. Passing True here would leave the job row alive and
        # hand the retry to a worker spawn we are not waiting on.
        outcome = run_job(job_id=job_id, video_id=video_id, difficulty=NORMAL,
                          uid=SEED_UID, settings=settings, store=store, source=source,
                          may_retry_elsewhere=False)
    except Exception as exc:  # noqa: BLE001 — reporting, not handling
        report["ERROR"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()[-800:]
        return report

    report["wallSeconds"] = round(time.monotonic() - started, 1)
    report["outcome"] = outcome
    job = store.get_job(job_id)
    if outcome != OUTCOME_READY:
        report["ERROR"] = f"{outcome}: {(job.error_code if job else '?')} " \
                          f"{(job.error_message if job else '')}".strip()
        return report

    # Read the graded chart back out of `chord_maps` rather than off `outcome`.
    # The point of the exercise is that the catalog now holds something correct,
    # and the only way to claim that is to look at what the catalog holds.
    stored = store.get_map(video_id, NORMAL)
    if stored is None:
        report["ERROR"] = "job reported ready but no row landed in chord_maps"
        return report

    report["stored"] = True
    report["title"] = stored.title
    report["durationS"] = round(stored.duration_ms / 1000, 1)
    report["realtimeRatio"] = round(report["wallSeconds"] / max(stored.duration_ms / 1000, 1), 2)
    report["engines"] = {"chords": stored.engine_chords, "beats": stored.engine_beats}
    report["lowConfidence"] = stored.low_confidence
    report["hasSync"] = stored.sync is not None
    report["difficultiesStored"] = [d for d in DIFFICULTIES
                                    if store.get_map(video_id, d) is not None]

    song = stored.song
    report["chart"] = _chart(song)
    # `tonic` + `mode`, not `key`: CompositionPayload v2 carries the two
    # separately because the app's transposition needs the tonic on its own.
    for key in ("tonic", "mode", "tempo", "timeSignature", "beatsPerChord", "repeats"):
        if key in song:
            report[key] = song[key]
    report["key"] = f"{song.get('tonic', '?')}{'m' if song.get('mode') == 'minor' else ''}"

    assert_clean(settings.scratch_root)
    report["scratchClean"] = True
    return report


# --------------------------------------------------------------------------
# Grading
# --------------------------------------------------------------------------

def _grade(report: dict) -> dict:
    truth = SEEDS[report["song"]]["truth"]
    chart = report["chart"]
    grade = {}

    vocabulary = [_triad(c) for c in truth["vocabulary"]]
    histogram = chart["triadHistogram"]
    total = sum(histogram.values()) or 1
    inside = sum(count for triad, count in histogram.items() if triad in vocabulary)
    grade["share"] = round(inside / total, 3)
    grade["missing"] = [t for t in vocabulary if t not in histogram]
    grade["extra"] = sorted(t for t in histogram if t not in vocabulary)

    cycle = [_triad(c) for c in truth["progression"]]
    grade["progressionFound"] = _contains_cycle(chart["sequence"], cycle)
    grade["progressionWanted"] = cycle

    # Compared as a triad so "F#" + mode minor and the truth's "F#m" agree, and
    # so an enharmonic spelling is the only thing left that can fail this.
    reported_key = report.get("key")
    grade["key"] = {"got": reported_key, "want": truth["key"],
                    "match": bool(reported_key) and _triad(str(reported_key)) == _triad(truth["key"])}

    reported_meter = report.get("timeSignature")
    grade["meter"] = {"got": reported_meter, "want": truth["meter"],
                      "match": str(reported_meter) == truth["meter"]}

    bpm = report.get("tempo")
    tempo = {"got": round(bpm, 1) if bpm else None, "want": truth["tempo"]}
    if bpm:
        ratio = bpm / truth["tempo"]
        if 0.95 <= ratio <= 1.05:
            tempo["verdict"] = "match"
        elif 0.45 <= ratio <= 0.55 or 1.9 <= ratio <= 2.1:
            tempo["verdict"] = f"half/double time ({ratio:.2f}x)"
        else:
            tempo["verdict"] = f"off ({ratio:.2f}x)"
    else:
        tempo["verdict"] = "not reported"
    grade["tempo"] = tempo

    # One headline per song. The share is the spine of it: a chart can find the
    # cycle once and be noise everywhere else, and a share that ignores order
    # can be perfect on a chart with the chords in the wrong sequence, so the
    # verdict deliberately needs both.
    if grade["share"] >= 0.85 and grade["progressionFound"]:
        grade["verdict"] = "CORRECT"
    elif grade["share"] >= 0.7:
        grade["verdict"] = "MOSTLY"
    elif grade["share"] >= 0.45:
        grade["verdict"] = "PARTIAL"
    else:
        grade["verdict"] = "WRONG"
    return grade


def _report(reports: list[dict]) -> int:
    seeded, failed = [], []
    for report in reports:
        if report.get("ERROR"):
            failed.append(report)
            continue
        report["grade"] = _grade(report)
        seeded.append(report)

    print("\n" + "=" * 78)
    print("SEEDED CATALOG — chart vs. ground truth")
    print("=" * 78)
    for report in seeded:
        grade, chart = report["grade"], report["chart"]
        print(f"\n{report['song']}  ({report['videoId']})  — {grade['verdict']}")
        print(f"  stored     : {', '.join(report['difficultiesStored'])} "
              f"| sync {'yes' if report['hasSync'] else 'NO'}"
              f"{' | lowConfidence' if report['lowConfidence'] else ''}")
        print(f"  key        : {grade['key']['got']} (want {grade['key']['want']}) "
              f"{'ok' if grade['key']['match'] else 'MISMATCH'}")
        print(f"  meter      : {grade['meter']['got']} (want {grade['meter']['want']}) "
              f"{'ok' if grade['meter']['match'] else 'MISMATCH'}")
        print(f"  tempo      : {grade['tempo']['got']} (want {grade['tempo']['want']}) "
              f"— {grade['tempo']['verdict']}")
        print(f"  vocabulary : {grade['share']:.0%} of {chart['chordCount']} chords "
              f"inside {SEEDS[report['song']]['truth']['vocabulary']}")
        if grade["missing"]:
            print(f"               never found: {grade['missing']}")
        if grade["extra"]:
            print(f"               unexpected : {grade['extra'][:8]}")
        print(f"  progression: {'FOUND' if grade['progressionFound'] else 'not found'} "
              f"{grade['progressionWanted']}")
        print(f"  sections   : {chart['sectionCount']} — "
              f"{[s['name'] for s in chart['sections']][:8]}")
        for section in chart["sections"][:6]:
            print(f"      {section['name']:<12} {section['bars']:>3} bars  "
                  f"{'-'.join(section['progression'][:12])}")

    for report in failed:
        print(f"\n{report['song']}  ({report['videoId']})  — FAILED\n  {report['ERROR']}")

    print("\n" + "-" * 78)
    correct = sum(1 for r in seeded if r["grade"]["verdict"] == "CORRECT")
    mostly = sum(1 for r in seeded if r["grade"]["verdict"] == "MOSTLY")
    print(f"{len(seeded)}/{len(reports)} seeded into chord_maps; "
          f"{correct} CORRECT, {mostly} MOSTLY, "
          f"{len(seeded) - correct - mostly} below that, {len(failed)} never analyzed")
    return 0 if seeded else 1


@app.local_entrypoint()
def main(songs: str = ",".join(DEFAULT), attempts: int = 3):
    """`attempts` is the bot check, not flakiness — see `real_song_check.py`'s
    entrypoint for the measurement. Every id here is an ordinary upload rather
    than label-owned, which is the easy case, so the budget is lower than that
    gate's six. An attempt that meets the bot check returns after the probe,
    before any audio moves, so a losing attempt costs seconds.
    """
    import sys

    if songs == "all":
        names = list(SEEDS)
    elif songs == "controls":
        names = list(CONTROLS)
    else:
        names = [s.strip() for s in songs.split(",") if s.strip()]

    unknown = [n for n in names if n not in SEEDS]
    if unknown:
        sys.exit(f"unknown song(s): {', '.join(unknown)}\nknown: {', '.join(SEEDS)}")

    print(f"Seeding {len(names)} song(s) into the catalog, up to {attempts} "
          f"attempts each (one container, hence one egress IP, per attempt):\n"
          f"  {', '.join(names)}\n")

    jobs = [name for name in names for _ in range(attempts)]
    results = list(seed_one.map(jobs))

    best: dict[str, dict] = {}
    for report in results:
        name = report["song"]
        if report.get("stored"):
            if not best.get(name, {}).get("stored"):
                best[name] = report
        else:
            best.setdefault(name, report)

    reports = [best[name] for name in names]
    print(json.dumps(reports, indent=2, default=str))
    sys.exit(_report(reports))
