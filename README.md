# Chord analysis for Rosetta GP

Turns a YouTube video id into a **playable song the app already knows how to
import** — a `CompositionPayload` v2, plus an optional `videoSync` sidecar that
lets a video clock drive campfire's cursor.

The spec is [`chord-backend-handoff.md`](chord-backend-handoff.md) in this repo.
Sections **11–19** (the 2026-08-03 amendment) supersede 1–9 where they conflict;
§2's invariants and §3's operational surface are unchanged and binding. Section
references throughout the code point there.

Sibling repos: `~/Projects/MIDI_Tab_Game` (the iOS app, `rosetta` branch) and
`~/Projects/Mo-Rosetta-GP` (the existing Mo backend, whose conventions this one
mirrors deliberately — §16).

---

## Status

**Working end to end, in the deployed shape, and now measured on the thing that
actually matters.** A YouTube id (or an uploaded file) goes in; a linted
`CompositionPayload` v2 and a `videoSync` sidecar come out. 764 tests, green, no
audio and no network required to run them.

That last clause is new and it was the important one. Every number this repo
reported until 2026-08-03 described a **component** — BTC's accuracy, beat_this's
downbeat F, whether the song lints clean. None of them described the
**deliverable**: *does the app show the right chord at the right moment in the
recording?* When that was finally measured, with ground truth fed in as both
engines so any error had to be the pipeline's own, the answer was **0.768** — a
23-point loss before any engine had made a mistake, on charts that all linted
clean and all shipped a sidecar. It is **0.939** now. See
[Measuring the deliverable](#measuring-the-deliverable-not-the-engine).

### The accompaniment's two hands (§14.1)

A piano accompaniment is a left hand and a right hand doing different things on
different beats. The app now has a piano in it, so the analysis had to stop
describing every song as if a hand were sweeping six strings.

It turned out to be a small change, because **the extraction was already
instrument-neutral**: onset positions, subdivision and accent are facts about the
song's rhythm. Exactly one emitted field was guitar-shaped — `direction` — and
§14 has always said that one is a convention rather than a measurement. What was
missing was a dimension nobody was reading: **which band the attack arrived in.**
A bass note and the chord over it are an octave and a half apart, and a split at
250 Hz finds them.

So a stroke can now say `low` or `mid`; saying neither means both, and travels as
an absent field. The client decides what that *means* — a `low` stroke is a bass
note to a guitar and a left hand to a piano — which matters because the catalog is
shared and **a song is analyzed once**. A piano-specific analysis would double
every song and fragment the catalog by instrument.

Three constants were measured, not chosen, against a new `oom-pah` specimen in
`bench/synth.py` (every previous specimen strums a chord, so none of them could
ask this question):

- the split is **250 Hz** — at 320 a third of the chord's energy reads as bass, at
  180 a strum starts reading as chord-only;
- presence is judged **per band against that band's own typical attack**, not as a
  ratio between the two. The ratio test is the obvious rule and it is wrong: it
  labels every ordinary strum `mid`, because a chord voiced from E3 up really does
  put most of its energy above the split;
- **a bar's bands survive only if the bar actually splits.** `mid` means nothing
  without a `low` to mean it against, and a song whose bass is merely quiet would
  otherwise report that it has no left hand.

Finding it also surfaced a defect in §14 proper: contrast compared every cell
against the loudest cell **in the bar**, which deletes a bass note quieter than
the chord over it. On the oom-pah that emitted the two chord stabs as the entire
pattern. Contrast is measured within a band now; on unbanded material the two are
the same number, so nothing about a strummed song moved — including its
content-addressed id, which a test pins against the pre-§14.1 hash.

The rights posture — what is stored, what is not, which Chordify surfaces are
deliberately not cloned, and the two rules the **client** repo has to keep so
App Review 5.2.3 stays a non-event — is [`RIGHTS.md`](RIGHTS.md).

| | |
|---|---|
| §3 operational surface | ✅ kill switch, per-video + per-channel blocklist, admin block/purge/offset, append-only audit log, verified purge cascade |
| §12 `CompositionPayload` v2 | ✅ emitter + the app's importer lint, ported from Mo |
| §12.2 chord normalization | ✅ Harte + symbolic → the app's closed grammar |
| §5.4 post-processing | ✅ quantize → merge → drop → hold N/C → confidence gate |
| §5.5 difficulty tiers | ⛔ **removed** — the chart states what was played ([below](#the-chart-states-what-was-played-55-withdrawn)) |
| §15 sections | ✅ superseded by §20.3 — fuzzy, global, phase-aligned repeat groups |
| §14 strumming patterns | ✅ fold/histogram/convention + quarter-note fallback, pooled per repeat group (§20.4) |
| §14.1 bands (bass vs chordal) | ✅ per-onset band split at 250 Hz → `Stroke.band`, so one analysis serves a strummed guitar **and** a tapped piano ([below](#the-accompaniments-two-hands-141)) |
| §13 `videoSync` sidecar | ✅ beat anchors + the §13.2 invariant, enforced by lint — **shipped on every song from a recording** ([§13.3 amended](#every-song-plays-with-its-recording-133-amended)) |
| §16 API | ✅ Mo-shaped: Firebase bearer, `{message, code}` errors, job-id + poll |
| §16.5 contract fixtures | ✅ emitted and byte-stable; the app-side test is a small follow-up (below) |
| §5.1 fetch + decode | ✅ yt-dlp + ffmpeg, bounded, behind the §4 seam — **plus an upload path** with no YouTube-terms exposure ([`RIGHTS.md`](RIGHTS.md)) |
| egress | ✅ **`CHORDS_YTDLP_PROXY` is live** (IPRoyal residential, rotating) — verified clearing the bot check on the first attempt, twice, on real audio ([below](#the-bot-check-is-per-ip-and-cookies-do-not-fix-it)) |
| the beat axis | ✅ one origin for chart, bars and anchors (`axis.py`) — the defect that cost 23 points |
| §20 theory layer | ✅ meter reconciled against the harmony, repeat groups, gated consensus, the song's own vocabulary (§20.8), modal key, one model rendered once |
| §21 two-sided benchmark | ✅ both correcting layers are **provable no-ops** on perfect input, measured separately on real engines, and on injected noise where the population can resolve |
| §5.2/§5.3 engines | ✅ **BTC + Beat This!**, benchmarked against real recordings (below) |
| §4 two-container shape | ✅ the API delegates to the worker; `tests/test_deployment.py` covers what `modal_app.py` relies on |
| CI | ✅ suite, Postgres, fixture stability, and a test that the API image cannot touch audio |
| Deploy gate | ✅ `scripts/smoke.py` — `/healthz` audit, one real analysis, and a proof that the cache hit is free |
| Seeded catalog | ⚠️ `scripts/seed_catalog.py` — 12 known songs graded against published transcriptions: 9 correct, 1 mostly, **1 wrong, 1 unanalyzable** ([below](#what-the-seeded-catalog-says)) |
| Service audit (2026-08-17) | ✅ four ship blockers, four performance findings, two audio-accuracy findings and fourteen smaller ones — all fixed, with the measurements ([below](#the-service-audit-2026-08-17)) |
| Fetch (2026-08-17) | ✅ `player_client=android` — yt-dlp's default returned 403 on the bytes for *every* video; the suite could not see it ([below](#the-media-url-403-and-the-client-that-is-served)) |

One real analysis, start to finish, on this machine:

```
POST /v1/analyze {"videoId": "QDYfEBY9NM4"}     → 202 queued
  fetching 10% → analyzing 35% → 60% → ready     75 s total
'Let It Be (Remastered 2009)'  70 bpm  4/4  key=C major  10 sections
  Part 1: 12 bars, chords ['Am', 'C', 'F', 'G']
lint: clean     lint_sync: clean (70 anchors)
cache hit: 200 inline, quota unchanged (§16.4)
scratch root: empty
```

---

## Run it

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp .env.example .env
.venv/bin/python -m pytest              # 511 tests, ~17s, no network, no audio stack
.venv/bin/uvicorn app.main:app --reload
```

That gets you the API and the cache. To actually **analyze** something you need
the audio stack and the engines, which are deliberately a separate install — the
API container must not be able to fetch or decode at all (§4):

```bash
.venv/bin/pip install -e ".[dev,engines]"
git clone https://github.com/jayg996/BTC-ISMIR19 vendor/BTC-ISMIR19
```

BTC is a git checkout rather than a dependency because it is research code with
no package; it ships its own weights. `vendor/` is the default location, or set
`CHORDS_BTC_ROOT`. ffmpeg must be on `PATH`.

```bash
curl -s localhost:8000/healthz | python3 -m json.tool
```

`/healthz` reports what actually **built**, not what config asked for — auth
mode, store backend, kill-switch state, whether a fetch source exists, which
engines are installed. A green health check that hides a dead authenticator is
the failure the Mo backend learned from.

It describes **the container answering it**, which on the deployed two-container
shape is the API image — correctly no fetch and no engines (§4). `canAnalyze`
comes from the *runner*, which answers from the deployment's shape, so the page
reads `canAnalyze: true` beside `engines: {}`; the `runner` field names which
runner said so, because otherwise that looks like the endpoint contradicting
itself rather than one page describing two containers. What it genuinely could not
see was a **worker image that built without an engine** — the API comes up, healthz
stays green, and every job fails one poll later in the chord stage.
`GET /v1/admin/worker` closes that: admin-gated, because it costs a worker cold
start, and a liveness probe that starts a container per check is a different kind
of problem.

---

## Measuring the deliverable, not the engine

Every other number in this file scores a *component* against ground truth. This
one reconstructs **what the player sees** — `videoCurrentTimeMs → songBeat → the
chord the compiled chart sounds there`, which is exactly the composition the
client performs every frame (`app/sync.py`: `song_beat_at` →
`chord_at_song_beat`) — and scores that. It is the `delivered` column in
`bench/run_bench.py` and the whole of `tests/test_alignment.py`.

Run with **ground truth as both engines**, so every point lost is the pipeline's
own arithmetic and not BTC mishearing a chord:

| | real corpus |
|---|---|
| engine score (raw spans vs truth) | 1.000 |
| delivered, before | **0.768** |
| delivered, now | **0.939** |

Three defects, none of which `lint`, `lint_sync` or any prior number could see —
all three produce a chart that is perfectly self-consistent and wrong:

**The chart and the sidecar used different origins.** `postprocess.quantize`
indexed into `BeatGrid.beats_ms`, so bar 0 began at the first *beat*;
`sync.anchors_for` put `songBeat 0` at the first *downbeat*. Nothing reconciled
them, and `anchors_for` even takes a `first_bar_index` for precisely this and was
never passed one. Any song with a pickup — most real recordings — was uniformly
phase-shifted against itself. `Something` scored 0.501, `Here Comes The Sun`
0.300. The fix is `app/analysis/axis.py`: one `BeatAxis`, built once, where beat
0 **is** the first downbeat and bar *k* **is** beats `[k·B, (k+1)·B)` by
construction. Three modules now share one object instead of three assumptions.

**Irregular bars derailed everything after them.** `bars_from_spans` assumed a
downbeat every `bar_beats` beats. Here Comes The Sun has 11/8 and 15/8 bars
inside a 4/4 song, so re-indexing alone only took it 0.300 → 0.531. The container
cannot express a meter change (§13.2 needs one uniform grid), so `BeatAxis`
resamples an odd bar onto the song's meter: it still starts and ends on the
tracker's real downbeats — which is what the anchors publish — and the error
stays inside that one bar instead of accumulating. 0.900.

**Structure substituted chords it hadn't heard.** `_merge_similar` folded a block
into its neighbour at 0.75 similarity and bumped `repeats`, replaying pass one
over pass two's bars — and one differing bar in a four-bar unit is *exactly*
0.75. Merging the section is right (§15 wants it, the player reads the rail);
merging the harmony is not. Identical blocks still collapse to `repeats`, which
is lossless; merely-similar ones now keep both passes' bars.

**Why 303 green tests missed all of it:** every fixture in `tests/conftest.py`
had `downbeats_ms[0] == beats_ms[0] == 0` — the one geometry that needs no
reconciliation. `conftest.recording(pickup_beats=…, odd_bars=…)` exists now so
the normal case is reachable, and `tests/test_alignment.py` asserts frame *and*
downbeat accuracy across it.

The residual ~0.06 is quantization to the beat grid: a chord that changes off the
beat cannot be said in a container whose unit is a stroke. Chordify quantizes to
beats too. `delivered` is scored on the one chart the analysis produces. It used
to have to name a tier here: `normal` folded diminished and augmented onto their
nearest playable triad (§5.5), so scoring it charged the pipeline for a reduction
it had been asked to make and read Michelle as 0.812 instead of 0.952.

---

## Every song plays with its recording (§13.3, amended)

**The report.** Some songs arrived in the app with no "play with the video"
option at all. That is the product; a chart with no recording behind it is a
songbook page.

**The cause was ours and it was deliberate.** §13.3 said "degrade honestly": when
the analysis was not confident, the service withheld the `videoSync` sidecar and
returned the song alone, to play self-paced. Four things triggered it — chord
confidence under the floor, beat-grid confidence under the floor, a tempo that
might be an octave out, bars that disagreed with their own meter — and any one
`lint_sync` complaint, of any severity, did it too.

The reasoning was that a wrong sync is worse than no sync. It does not survive
contact with what withholding actually does:

- **It repairs nothing.** The chart handed back is the same chart. If the
  analysis misheard the harmony, it misheard it self-paced as well.
- **The anchors were measured on that recording.** They are absolute timestamps
  taken off the audio the player is about to press play on. A low chord
  confidence is a statement about the *harmony*; the beat grid is tracked
  separately, and one has never been evidence against the other.
- **It is permanent.** The sidecar-less row is a cache hit, so every later
  request for that song serves the same missing feature, forever.
- **The client was already told.** `lowConfidence` exists precisely so the app
  can caveat a doubtful reading. Withholding says less, not more.

**The rule now.** A song that came from a recording ships a sidecar. The only
thing that can still take it away is a `lint_sync` **fatal** — anchors that are
not a map at all: none of them, fewer than two, an order that runs backwards, or
a chart whose per-section tempo overrides mean the client cannot derive a bar
grid to address. Everything else is **advisory**: the map works and is imperfect,
so it ships carrying `lowConfidence: true` and the client says so.

Three pieces of machinery, in `app/lint.py`, `app/analysis/compile.py` and
`app/analysis/pipeline.py`:

| | |
|---|---|
| `SyncProblems` | `lint_sync`'s findings split into `fatal` and `advisory`. Only `fatal` withholds. |
| `fit_sync` | trims the anchors to the chart's own last barline. `impose` merging across a barline leaves anchors for bars the chart no longer has; that is a **trim**, so it is trimmed rather than reported and punished. |

The confidence floor still exists and still means what it meant. It sets
`lowConfidence`; it no longer decides whether the feature exists.

**Stored rows do not fix themselves.** A map filed before this change has
`sync_json IS NULL` and will keep serving no video option until it is
re-analyzed. That is a one-command sweep, narrow by design — it purges only the
affected rows, and each comes back the first time anyone opens the song:

```bash
modal run scripts/sweep_catalog.py --missing-sync            # list them
modal run scripts/sweep_catalog.py --missing-sync --confirm  # purge; they re-analyze on demand
```

**No video is treated differently from any other video.** Worth stating because
it was the first suspicion: nothing in the fetch, the gate, the pipeline or the
store reads a video's channel, its uploader, or whether it is an official label
upload. `probe` records `channel_id` for exactly one purpose — §3's blocklist is
per-channel — and the only other per-video branch in the service is the length
cap. Whatever made the missing feature *look* like it followed a category of
video ran through the confidence floor, which is the only thing in the system
that could have varied by recording — and it is the thing this change removed
from that decision. If a pattern survives the fix, it is a beat-tracking
question, and it will be visible in `lowConfidence` rather than in a feature
that is not there.

---

## The chart states what was played (§5.5, withdrawn)

**The ruling.** Asked whether a reference corpus should be stratified by harmonic
difficulty, the owner answered that it should not: *"The harmonic difficulty
isn't for us! It's for 'real guitar' softwares where people execute the chords.
For us the chord is just a band to swipe on."* So the easy/normal/hard tiers are
gone — not deferred to a later simplification layer, removed.

**Why the argument for them never applied here.** §5.5 defined `easy` as "collapse
to triads, major/minor only; drop passing chords shorter than a bar". Every word
of that is about what a beginner's hands can do. The Rosetta client never renders
a fingering: a chord is a band the player swipes, and `Gmaj7` costs exactly what
`G` costs to play. Collapsing the one onto the other bought a beginner nothing
and destroyed information the analysis had just worked to earn — the whole of
§20's theory layer exists to get that seventh right.

It had already half-collapsed under its own weight. §12.2 established that `hard`
could not mean "full detected quality" — the ceiling is what the app's
`ChordSymbol(name:)` parses — so the tiers were re-scoped to three shapes inside
that grammar, and `_NORMAL` then mapped `DOMINANT7 → DOMINANT7`, `MINOR7 →
MINOR7`, `SUS4 → SUS4`. The default tier reduced nothing and rendered identically
to `hard`. That read like a bug and was in fact the design telling the truth.

**What came out.**

| | |
|---|---|
| `chords.simplify`, `_EASY`, `_NORMAL`, `DIFFICULTIES` | the whole tier vocabulary. `normalize` stays, and is now the only reduction in the codebase — a ceiling rather than a choice, and one that *counts itself* in `exactRatio`. |
| `postprocess.simplify_tier` | step 5 of the §5.4 chain. The chain is quantize → merge → drop → hold N/C → confidence gate, and it never renames a chord. |
| `drop_short(to_longer=…)` | existed only for `easy`'s one-*bar* floor, where the rule was choosing between two real chords rather than cleaning jitter. At the one-beat floor there is no second chord to choose. |
| `model.render(…, difficulty)` | one render. The replay discipline it was built for — `vote_groups`, `vote_key`, `seed_key`, `canon_rounds`, every `record=False` — **stays**, because "a render reads the model's decisions rather than retaking them" is still worth holding by construction. |
| `AnalysisOutcome.songs` / `.syncs` | two dicts keyed by difficulty become `song` and `sync`. `sync_for` and `sync_tiers` are gone; `sync` is a plain field. |
| `chord_maps (video_id, difficulty)` | the primary key is `video_id`. One recording, one row. |
| `jobs.difficulty`, the `difficulty` request field, the catalog card's `difficulty` | gone from the wire and from the job. |
| `song_id(video_id, difficulty)` | `yt:<videoId>`, with no suffix. The suffix existed so three tiers could hold three Library rows without `import` upserting one over another. |

**The store migration is hand-written and lossy on purpose.** `difficulty` was
half of `chord_maps`' primary key, so the table is rebuilt rather than altered —
and rebuilding forces the question of which of a video's three rows survives. It
keeps **`hard`**: that was the reference render, built at the full grammar, and
the only one of the three that ever claimed to state what was played. `easy` and
`normal` were reductions, and a reduction is what this change exists to stop
shipping. See `Store._drop_difficulty`.

**One thing got simpler as a side effect.** `list_catalog` collapsed rows with a
`ROW_NUMBER() OVER (PARTITION BY video_id …)` subquery because a video analyzed
at two difficulties had two rows. One row per video is the primary key now, so
the listing is a plain indexed walk of `chord_maps_catalog` with `LIMIT`/`OFFSET`.

**What this does not change.** Nothing about accuracy: the reference tier was
already what every number in this README was measured on, so the benchmark reads
the same chart it always read. What it removes is two thirds of the compile work
per job, a schema that could file three answers to one question, and a standing
invitation to fix engine noise by hiding it behind a tier — which the owner had
already ruled out once, on different grounds, when a per-tier chord budget was
proposed for "the song has more chords than it should". The noise got fixed at
the source instead (§20.8, §20.10).

---

## The music-theory layer (§20), and what it actually bought

A song repeats itself. Almost every section here is one progression played
several times, in one key and one meter — while a chord recognizer makes
**independent** mistakes, so it can hear verse 3 differently from verses 1 and 2
on a nearly identical recording. §20 is the layer that knows this: it reconciles
the meter against the harmony, finds the song's repeat groups, and lets a
group's occurrences vote their engine noise out.

That last part edits chords the engine reported, which makes it the most
dangerous code in the service — `lint` and `lint_sync` both check the song
against *itself*, so a chart made uniformly self-consistent is exactly what they
cannot complain about. It is the same shape as the alignment defect: 23 points
gone behind three hundred green tests.

So overwriting requires **three independent gates** (§20.4): a decisive vote (at
least two occurrences reading the slot identically, and no other reading agreed by
as many), a harmonically *near* disagreement (C↔Am is a mishearing, C↔F is a chord
change), and a dissenter that was believed **less** than the winner. The third
gate is what makes the design testable — ground truth arrives at a flat confidence
of 1.0, so **on perfect input consensus is provably a no-op**, from the
construction rather than from luck.

### The vote's blind spot, and §20.8

The vote can only speak where a section **repeats** and its passes **disagree**,
and that turns out to be narrower than it sounds. A section that occurs twice is
one reading against one reading. A section that occurs once — an intro, a bridge,
a tag — has nothing to compare against. And a mistake the engine made in *every*
pass leaves nothing to disagree with, because a recognizer's errors are only
independent when the audio differs.

That blind spot is not a residue; it is the ordinary case, and it is what a user
sees. Reported against a song whose chart is Ebm–Db–Ab throughout: one bar showed
`Ebm7` and another showed `Eb`, neither of them in the recording, and both of them
in a four-pass verse where the engine had misheard *two* passes — which is exactly
the shape a two-thirds share files as "contested" and ships unchanged.

**`app/analysis/vocabulary.py` answers those with different evidence: the rest of
the song.** A song has a small chord vocabulary, measured over minutes, so a
brief and doubtful reading of a root that the song contradicts everywhere else is
the engine, not the arrangement. Two rules, both bounded to the same root, and the
edit is always a *spelling* correction rather than a new chord:

- an **island** — a hole punched in a held chord (Ebm | Eb | Ebm) — is filled;
- a **minority reading** is pulled onto the quality the song plays on that root.

Which moves are allowed is **measured, not reasoned** — `python bench/run_bench.py
--calibration` prints the table it comes from. Near-miss says two chords are close
enough for a recognizer to slide between them; it says nothing about which
direction it slides, and that is the only fact that decides whether an edit pays:

| this engine reports | the record plays it | plays the plain triad | wrong root |
|---|---|---|---|
| `dominant7` | 0.31 | **0.60** | 0.01 |
| `minor7` | 0.30 | **0.39** | 0.30 |
| `major7` | 0.33 | 0.00 | 0.67 |
| `augmented` | 0.12 | 0.00 | 0.88 |
| `diminished7` | **0.90** | 0.00 | 0.10 |

A reported dominant 7th is the plain major twice as often as it is a seventh, so
flattening a doubtful one is a bet at 2:1 on. A major 7th is *never* the plain
triad when it is wrong, so the same edit there can only lose — which is what it
did to Let It Be's opening `Fmaj7` while the rule was still generic. `major7`,
`augmented`, `diminished` and `diminished7` are excluded on that evidence.

The gate that settled the hardest case is **recurrence**. In My Life plays A for
seventy beats and A7 for nine, in four brief doubtful passes: every measure of
*amount* calls those noise, and they are the song. Something reports G7 twice, just
as briefly, and both times the record plays a plain G. Nothing about the amount of
evidence separates them — what does is how many separate occasions carry the
reading, so a reading appearing more than twice is treated as the arrangement.

`python bench/run_bench.py --theory` runs it twice, because the two ways it can
be wrong pull in opposite directions:

| run (nine real tracks) | layers off | consensus | + vocabulary | edits |
|---|---|---|---|---|
| ground truth as both engines | **0.939** | **0.939** | **0.939** | 0 |
| BTC + Beat This! | **0.796** | **0.800** | **0.803** | 16 bars, 12 spans |
| BTC + Beat This!, pre-§20 commit | **0.796** | — | — | — |

Nine tracks cannot resolve an effect this size, so §20.8 added a harness that
can: `python bench/run_bench.py --noise` injects the engine's *measured* mistakes
into ground truth, over the same nine songs, and counts both sides of every edit
separately — `fixed` is the share of injected errors removed, `broke` the share of
correct chords destroyed. Twelve seeds:

| layers | in | out | fixed | broke |
|---|---|---|---|---|
| consensus | 0.797 | 0.808 | 0.070 | 0.003 |
| vocabulary | 0.797 | 0.810 | 0.100 | 0.009 |
| both | 0.797 | 0.815 | **0.138** | **0.011** |

Twelve errors removed for every one introduced, and the two layers are nearly
additive (0.070 + 0.100 ≈ 0.138) — which is the design claim holding up, since they
answer with different evidence and so fix different mistakes. The numbers are never
summed into one score: a layer that fixed a third of the noise and broke a
twentieth of the music would not be worth having however well its mean read.

Read honestly, because the temptation is to read it the other way:

- **The architecture is delivered-neutral.** Meter reconciliation, the new form
  detection, the model/render split and the modal keyfinder together move the
  number by nothing. What they bought is coherence and provenance — repeats
  collapse, the render agrees with the model by construction, sections carry
  group identity, and the sidecar reports what was changed — not accuracy.
- **Consensus is a marginal win**: +0.003, with Michelle up 0.028 and Let It Be
  *down* 0.014. Real, but within noise on nine tracks. That is why
  `CHORDS_THEORY_CONSENSUS` exists, and why the harness prints MARGINAL rather
  than PASS below half a point.
- **The vocabulary layer is marginal on the corpus and decisive on the defect it
  was written for**: +0.003 delivered, no track regressing, and 10.0% of injected
  noise removed against 0.9% of correct chords damaged. On the reported song it is
  the difference between a chart with `Ebm7` and `Eb` in it and one that reads
  Ebm–Db–Ab throughout. `CHORDS_THEORY_VOCABULARY` turns it off.
- **Key detection is a wash**: 5/9 exact tonics before and after. The modal fix
  is still right on its own terms (`G F C G` was called *A minor*; it is G
  mixolydian), and so is capping the tonic-endpoint bonus.

`CHORDS_THEORY_FORM` is the fourth flag and the only one **on** by default that
edits the chart's shape — see [§21](#21--the-chart-states-the-songs-form), which
is a claim about the song's form rather than about the engine's mistakes.

There is a third flag beside those two, off by default:
**`CHORDS_THEORY_TEMPO_OCTAVE`**
lets §20.2 halve or double a beat grid whose tempo reads an octave out, instead
of only reporting it. It is off for the opposite reason to consensus — not
measured-and-marginal but **unmeasured**, because no track in the corpus triggers
it. With it off a suspect tempo the container can still carry ships
`lowConfidence`, and one it cannot fails with a message that names the reading
rather than the generic "didn't produce a song we could play"
([`PIPELINE-AUDIT.md`](PIPELINE-AUDIT.md) D1).

---

## §21 — the chart states the song's form

The complaint: *"clearly our system doesn't know that a verse (or any section) is
the same across the whole song"*. It was right. Creep is four chords over eight
bars for its whole length, and the pipeline emitted **88 distinct bars in one
unbroken section with a ten-chord vocabulary** — every wobble the recognizer made
on every pass preserved as though it were music.

Everything needed to fix it was already there. `form.py` finds the repeat groups
and `consensus.py` votes over them; what was missing is the step between "these
eleven blocks are the same music" and "so print them the same way". §20.6 says
the second `form.detect` pass exists because *"after the vote they often are
identical, and the second pass is what turns that into the compact encoding"* —
and on a real recording they never were, so `repeats` only ever fired on
synthetic input.

`app/analysis/canon.py` is that step, and it is **not** a correction layer. §20.4
asks whether the engine misheard a bar and gates every overwrite on three
conditions; this asks what the group **agrees** on and writes that everywhere. The
distinction matters because the gates are why §20.4 stays silent on the ordinary
case: Creep's eighth bar is heard `Cm` on six passes, `Cm D#` on three and `Cm F`
on two, nothing there is a near miss of anything, so all eleven readings ship.
That is the right answer to "did the engine mishear bar 8 of pass 7" and the
wrong answer to "what does this song play in bar 8".

Two rules, both gated on a measurement of the song rather than on a threshold:

- **`settle_to_bars`** — in a song whose harmony moves a bar at a time, a change
  the engine put a beat early belongs on the barline. §5.4 already argues this at
  beat resolution ("a chord change that lands 40 ms before the beat is the same
  musical event as one that lands on it"); the argument does not stop at the
  beat. The gate is `harmonic_unit`, the duration-weighted median chord length:
  4 beats or more and the song settles, 2 beats and it does not. Wonderwall and
  Smooth Criminal measure 2 and are left alone, which is correct — `| F#m7 A |`
  is a real two-chord bar and settling it would delete half the song.
- **`canonicalize`** — every occurrence of a repeat group plays the group's
  progression, decided **beat by beat** across the occurrences. Beat resolution
  rather than whole-bar, because occurrences disagree about *where in the bar* a
  change happens at least as often as about what the chord is: eight passes of
  `| G |` and four of `| G D |` are not two competing bars, they are twelve
  passes agreeing on beats 1–3.

What it costs, stated plainly: an occurrence that genuinely differs is flattened
onto the others. `tests/test_alignment.py` pins both sides — a one-off Dm in bar
12 of four passes survives with `CHORDS_THEORY_FORM=off` and is flattened with it
on, at a cost of exactly one bar in sixteen. Three things bound it: only groups
that already cohere (`MIN_COHESION`), only beats with a strict plurality (a tie
holds its neighbour, or the bar is declined outright), and never a chord no
occurrence played.

### What it is worth

Measured on the ten-song chart corpus (`bench/lab.py grade`, and
`CHORDS_THEORY_FORM=off` for the other row):

| | root | triad | form | vocabulary emitted |
|---|---|---|---|---|
| §21 off | 0.804 | 0.788 | 0.674 | Creep 10, Wonderwall 9, Smooth Criminal 15 |
| §21 on | **0.853** | **0.846** | 0.688 | Creep **4**, Wonderwall 8, Smooth Criminal 13 |

**+0.049 root and +0.058 triad**, and eight of the ten songs improve. That is a
much larger effect than any of §20's three correcting layers (each worth about
+0.003), and the reason is worth stating because it is not "voting harder": those
layers ask whether a *bar* was misheard and are gated so they mostly cannot
answer, while this one asks what the *section* plays and always can.

Two of the four largest gains come from `settle_to_bars` rather than from the
vote — Viva La Vida +0.144 and Don't Stop Believin' +0.091 — and the mechanism is
worth knowing: a bar holding the tail of one chord and the head of the next
matches *neither* reference bar, so a change a beat early costs the root twice.
Creep (+0.069) is the vote's own case, and it is the one where the shape matters
more than the number: it now compiles as one eight-bar section with `repeats: 11`
and a four-chord vocabulary, which is the chart, and Zombie as `| C | G | D | Em |`
twenty times over.

Smooth Criminal is the only song that goes down (−0.013), and it is the corpus's
known outlier for reasons that have nothing to do with this layer.

It runs to a **fixed point**, and that is where the last of the root goes. Each
round ends by re-finding the form on the bars it just rewrote, and `form.detect`
re-derives the period and the phase to do it — so the blocks it returns are
often not the ones just made to agree, and after one round Wonderwall's chorus
is two readings again and prints as one flat run of thirty bars. The corpus
converges in three rounds; a fourth is a no-op on all ten songs. Against a single
round that is form +0.014 and root −0.004.

### Two changes that did not ship, and why

Both were tried on the corpus and reverted, which is the only way to tell this
kind of change from an improvement:

- **Best-match clustering.** `form.cluster` joins a block to the *first* group it
  matches, not the best one, which looks obviously wrong. Fixing it moves form
  +0.012 and root −0.008 — Three Little Birds loses five points because a block
  reassigned to a better-matching group is then made to agree with it. A change
  that raises one number and lowers another has bought nothing.
- **Tempo-octave correction from harmonic evidence.** Country Roads is tracked at
  167 bpm against a true 82, and `harmonic_unit` sees it (its chords last 8 of the
  tracker's beats). Halving the grid moves every bar line and every anchor in the
  song, and the corpus has one track that would trigger it — so the diagnosis is
  reported and `CHORDS_THEORY_TEMPO_OCTAVE` stays off, exactly as §20.2 already
  argues.

### The corpus this was measured on

Ten popular songs with hand-written reference charts in `bench/reference/`, and
a cached-features harness (`bench/lab.py`) that regrades all ten in about a
second. Root scores after §21, best to worst: Creep 1.000, I'm Yours 0.985, Viva
La Vida 0.971, Zombie 0.943, Someone Like You 0.885, Three Little Birds 0.882,
Wonderwall 0.852, Country Roads 0.840, Don't Stop Believin' 0.838, Smooth
Criminal 0.330. Nine of ten keys are exact.

Smooth Criminal is the outlier and it is the owner's own failing case, on the
9:26 short film: the first minute is dialogue, the beat tracker emits **no beats
at all** across a 51-second stretch of it and reports 0.00 confidence, and the
harmony is a bass line with no instrument sounding a third — so a recognizer
hearing Bm and C where the bass walks through B and C is hearing the recording
correctly and charting it wrongly. Creep's key is the other known miss: G major
and C major score identically on every term the key finder has, because `Cm`
shares C's root and the endpoint bonus splits one each. Both chords spell the
same either way, so nothing the player sees changes.

---

Anyone extending this should assume the remaining accuracy is in the engines, and
in §20.2's downbeat-phase check on tracks where the tracker is genuinely wrong —
not in voting harder. The one exception the calibration table points at is the
**relative-major/minor confusion**: 5.1% of minor chords come back a third up,
which is the largest single bucket of engine error left, and it is deliberately out
of §20.8's scope because both chords are usually in the same song's vocabulary. It
needs bar-position evidence, which means it belongs to the vote.

---

## The engine choice (§8 step 2), and the evidence for it

**Chosen: BTC for chords, Beat This! for beats, librosa for onsets.** Defaults in
`app/config.py`; override with `CHORDS_CHORD_ENGINE` / `CHORDS_BEAT_TRACKER`.

### The corpus

The handoff's instruction was to benchmark on real tracks, and the honest problem
with that is ground truth: a chord chart off the web has no timestamps, so it
cannot score anything. **Isophonics** (Centre for Digital Music, QMUL) publishes
180 Beatles songs with time-aligned chord labels in Harte syntax *and* beat files
whose second column marks position-in-bar — ground truth for both halves of the
benchmark, from an annotation set the MIR community has checked for twenty years.

`bench/fetch_corpus.py` pairs those annotations with the recordings, fetched the
same way the service fetches. Twelve tracks were chosen by profiling all 180 for
the axes the choice turns on — harmonic complexity 0.00→0.82, tempo 66→178, 3/4
and irregular bars alongside 4/4, vocabulary 3→27 chords. **Nine survived**; the
rejects are logged with a reason, because a misaligned specimen ranks engines by
who lags rather than who is right.

Alignment is the interesting part. Isophonics timed its annotations to 1987 CD
issues; YouTube serves remasters. So the script fits a **shift and a stretch**:
one chroma correlation to find the neighbourhood, six local ones to sample lag
against time, least squares through them, and the *residual* as the accept
criterion. Let It Be needed +1.5 s of leading silence and 350 ms of speed
correction, fitted to a 25 ms residual. (The first version correlated onsets
instead, which is periodic at the beat and made two halves of one song lock onto
different beat-period aliases — reported as drift that wasn't there. Harmony is
not periodic; that is why the fit uses it.)

```bash
.venv/bin/python bench/synth.py                          # synthetic set
.venv/bin/python bench/fetch_corpus.py --annotations …   # real set
.venv/bin/python bench/run_bench.py
```

Nothing it writes is committed — `bench/audio/` is gitignored, the annotations
are CC BY-SA-NC and the recordings obviously aren't ours. `bench/corpus.json`
pins the resolved video id per track so a later run scores the *same* audio.

### The numbers

Nine real tracks. Chord accuracy is measured **after** normalization into the
app's grammar (§12.2) — the only accuracy that means anything downstream, since
an engine that says `Cmaj9` and one that says `Cmaj7` play identically here.

| chord engine | real acc | root only | synthetic | s/track |
|---|---|---|---|---|
| **btc** | **0.808** | 0.888 | 0.876 | 1.7 |
| chroma (control) | 0.531 | 0.702 | 0.895 | 8.8 |

| beat tracker | beat F | **downbeat F** | bpm err | s/track |
|---|---|---|---|---|
| **beat_this** | **0.918** | **0.893** | 1.2 | 36.8 |
| librosa | 0.720 | 0.486 | 17.3 | 5.3 |

Three things worth reading off that table:

**The control earns its place.** A chroma template matcher *beats BTC on the
synthetic set* (0.895 vs 0.876) and loses by 27 points on real recordings. That
is `synth.py`'s own warning, measured: synthetic audio proves plumbing and says
nothing about a dense mix. Had the benchmark stopped at synthetic, it would have
recommended the wrong engine.

**Downbeats are the whole beat-tracker margin.** The two are much closer on beats
(0.918 vs 0.720) than on downbeats (0.893 vs 0.486), and §13.2's anchors *are*
downbeats. librosa's 17.3 bpm mean error is octave errors — it heard Let It Be at
143 bpm instead of 70.

**A higher sidecar rate is not better.** In the end-to-end matrix librosa
produced a sidecar for 15/15 tracks and beat_this for 14/15 — while being wrong
about downbeats half the time. `lint_sync` can only catch anchors that disagree
with the *chart*; it cannot catch a grid that is internally consistent and
disagrees with the *music*. Choosing on that number would have picked the worse
tracker for the reason it was worse.

Cost: ~10 s of DSP for a 3-minute song on CPU, against a 450 s job deadline. BTC
on a GPU was not pursued — §18 anticipated that a GPU cold start per job may cost
more latency than the accuracy gap costs quality, and at these numbers there is
nothing to buy. (The `s/track` figures above are lower than the ones first
recorded here because the engines are now built once per container rather than per
job; see below.)

#### The feature extractor's block edges (fixed 2026-08-17)

BTC's 10-second blocks are its **framing** — `timestep` is 108 frames, which at hop
2048 is exactly one 10-second block, so a block *is* one inference window. What did
not follow from that, and was being done anyway, is transforming each block in
isolation. `librosa.cqt(..., center=True)` pads whatever it is handed, and at 24
bins per octave the lowest bin's analysis window is ~1.04 s — so roughly 0.52 s at
each block edge was computed from zero-padding rather than from the recording.
About 10% of every block, for the whole song, worst in the bass where the root is.

Each block now gets seven hops of its neighbours' audio and is trimmed back to its
own frames: same framing, same 108 frames, every frame real audio. Against a single
CQT over the whole signal as ground truth, mean absolute error in the bottom two
octaves:

| | block edges | block interiors |
|---|---|---|
| isolated blocks | 1.73 | 0.25 |
| with context, trimmed | **0.24** | 0.25 |

Frame *times* were wrong in a second, separate way: frame *j* of block *b* is
centred at `b · 10 s + j · hop / sr`, and the code used a single `10.0 / 108` per
frame — right about each block's origin, then 0.287 ms/frame slow inside it, a
**30.7 ms sawtooth** against the beat grid that reset every block. (Using the true
hop *globally* is worse, not better: 108 hops span 10.031 s, so it would drift a
full second every hundred blocks.) Both terms are now exact, which is why the times
are built where the block structure is known instead of derived from a frame index.

**And the honest part: downstream this is worth very little.** Raw chord accuracy
on the real corpus goes 0.805 → 0.808, with 8 of 9 tracks non-negative;
*delivered* accuracy — the chord actually on screen — does not move at all
(0.793 → 0.794). BTC's self-attention runs over all 108 frames of a window, so it
absorbs a couple of degraded frames at each edge, and the chord at a block seam is
usually the same chord as on either side of it. The defect was real and is
measurable in feature space; the fix is strictly more correct and slightly *faster*
(12.1 s vs 13.6 s over the corpus), and it is not an accuracy win. Keeping it is
about the frame grid being true — the sawtooth is an alignment error, and alignment
is what §13.2's anchors are for.

### Adding another candidate

Three steps, none of them upstream:

1. write the adapter in `app/analysis/adapters/` (`analyze(pcm, sr) ->
   list[RawChordSpan]`; emit Harte or symbolic labels, post-processing
   normalizes them),
2. add a row to `_BUILTIN_CHORD_ENGINES` in `app/analysis/engines.py` naming its
   required imports — registration is conditional on them, so an engine only
   advertises itself in an image that can actually build it,
3. add the dependency to an `engines-*` extra **and** to `modal_app.py`'s
   *worker* image — never the API image (§4).

Two candidates named in §5.2/§5.3 were not benchmarked, for reasons rather than
preference. **Chordino/NNLS-Chroma**: its binaries live on soundsoftware.ac.uk,
which was refusing connections throughout this work; the adapter slot is wired
(`engines.py` knows the name) and needs only the plugin. **madmom**: upstream
still does not import on Python 3.11, and the maintained CPJKU fork exists mainly
to serve `beat_this` — which won on its own merits anyway.

---

## How it fits together

```
POST /v1/analyze          {videoId}  ──┐   cache hit? ──► 200 {song, videoSync}
POST /v1/analyze/upload   {file}     ──┘        (free — §16.4, id is a content hash)
                 └─► 202 {jobId} ──► worker (its own container, own image)
                                       │
                       probe ──► gate (blocklist · 10-min cap · kill switch)
                                       │   nothing fetched until this passes
       scratch dir ──► decode ──► beats ──► chords ──► onsets ──► energy
                                       └─► rm -rf audio  (every exit path)
                                       │
                       §20.2 meter reconciled against the harmony
                       build_axis ──► ONE beat axis (chart · bars · anchors)
                                       │
                       §5.4 post-process ──► §20.3 form ──► §20.4 consensus
                                       │        (repeat groups)   (gated vote)
                       §20.6 ONE model ──► ONE render: what was played
                                       │
                       §12 compile ──► lint ──► Postgres: chord_maps
GET /v1/analyze/{jobId} ──► status / {song, videoSync}

GET /v1/catalog          ──► chord_maps, newest first ──► the app's home screen
GET /v1/catalog/version  ──► "<count>:<newest analyzed_at>"
```

**The catalog is what makes the app's home screen exist** (added 2026-08-06). It
lists every analyzed song, newest first, so a player who has analyzed nothing
still has something to play — the cold-start problem solved by sharing, since
every row is a cache hit and costs no quota, no egress and no wait. Four things
about it are load-bearing:

- **Uploads are excluded** (`owner_uid IS NULL`, added 2026-08-17). Uploaded audio
  lands in the same `chord_maps` table as a fetched video, and nothing
  distinguished the two — so one player's private rehearsal recording appeared on
  every other player's home screen, under their own filename, with their chart,
  key and tempo. The row was functionally broken as well as private:
  `embeddable: true` beside a `videoId` of `up_<hash>`, which no YouTube player
  can resolve. `GET /v1/maps/{id}` had the same hole and now answers 404 for a
  row the caller does not own; an uploader gets their own analysis back through
  `POST /v1/analyze/upload`, where possession of the bytes is what authorizes the
  read. `catalog_version()` counts the same public rows, so a private upload no
  longer wakes every client to re-fetch a list that did not change.
- **Blocked videos and channels are excluded in SQL**, not filtered afterwards.
  §3's takedown has to hold on a listing as firmly as on `GET /v1/maps/{id}`; a
  blocked video that vanishes from the detail route but still sits on a home
  screen is a takedown that did not happen.
- **One row per video, collapsed and paged in SQL.** A video analyzed at three
  difficulties is three rows in `chord_maps` and one *song* in the catalog, so the
  listing collapses to the newest per video id — with `ROW_NUMBER() OVER
  (PARTITION BY video_id)`, and then `LIMIT`/`OFFSET`.
- **Readable without signing in** (`_principal_browsing`). Every other route needs
  an identity because it spends quota or starts work; this one only reads rows
  that already exist, and refusing it would empty the home screen for exactly the
  person it is for — someone deciding whether to sign up at all. Anonymous callers
  still get an IP-keyed uid, so both rate limiters apply.

`/v1/catalog/version` is the cheap half: the client polls it and pulls the list
only when the token moves, so "a song someone else added shows up" costs one short
string rather than the whole catalog. Artwork is the video's own poster, which the
client derives from the id.

**The five scalars a card prints live in their own columns** (`song_id`, `artist`,
`tempo`, `tonic`, `mode`, `chord_names`), written by `put_map` from the payload it
is already storing. They used to be read *through* `song_json`, which meant the
collapse and the page happened in Python: every row in the table fetched, every
`CompositionPayload` decoded, deduped in memory, and then sixty of them returned.
So a catalog hit cost time linear in the size of the catalog forever, and `offset`
bought nothing at all — page 30 cost exactly what page 1 did. Measured on this
machine:

| maps | old | new |
|---|---|---|
| 500, first page | 44 ms | 4.5 ms |
| 2000, first page | 249 ms | 15 ms |
| 2000, `offset=1900` | 257 ms | 17 ms |

The duplication is safe because `put_map` is the only writer and
`store._catalog_scalars` is the only place that reads the payload into them; a
database written before the columns existed is backfilled once, on open, and
flagged so a container start after that is one indexed count rather than a table
scan.

Two input paths, and the difference is legal rather than technical: `/v1/analyze`
fetches a YouTube recording (which §2 concedes contravenes the API terms as
written), `/v1/analyze/upload` takes audio the player already has and carries no
such exposure. Everything downstream of `decode` is identical, and the upload
path is what §3's kill switch degrades *to* rather than degrading to nothing.
See [`RIGHTS.md`](RIGHTS.md).

| Module | What it owns |
|---|---|
| `app/payload.py` | `CompositionPayload` v2 — a near-copy of Mo's, `yt:` ids |
| `app/chords.py` | the app's grammar (ported) + §12.2 normalization — the only reduction there is |
| `app/lint.py` | the importer's checks (ported) + `lint_sync`, the §13.2 invariant |
| `app/sync.py` | the sidecar, anchors, and the client's interpolation in Python |
| `app/store.py` | maps, jobs, blocklist, audit log, quota, limiter — two backends |
| `app/analysis/` | the pipeline; `scratch.py` is §2.1 in code |
| `app/analysis/axis.py` | **one** beat axis — chart, bars and anchors share an origin by construction |
| `app/analysis/harmony.py` | §20.1 — harmonic distance: is a disagreement a mishearing or a chord change? |
| `app/analysis/meter.py` | §20.2 — the harmony's second opinion on where the bar starts |
| `app/analysis/form.py` | §20.3 — repeat groups, found fuzzily and globally (supersedes §15's segmentation) |
| `app/analysis/consensus.py` | §20.4 — the three-gate vote: the same bar in another pass |
| `app/analysis/vocabulary.py` | §20.8 — the song's own chord vocabulary, for the bars the vote cannot reach |
| `app/analysis/model.py` | §20.6 — the song model; `build` decides and `render` reads those decisions back |
| `app/analysis/ytdlp_source.py` | the only code that ever holds audio — worker image only |
| `app/analysis/file_source.py` | the upload path: player-supplied audio, no YouTube-terms exposure |
| `app/analysis/adapters/` | one file per engine; nothing else imports a model |
| `app/main.py` | the HTTP surface, shaped like Mo's |
| `modal_app.py` | two functions, **two images** — that split *is* §4's isolation |
| `bench/fetch_corpus.py` | annotations + recordings → a scoreable corpus |
| `scripts/smoke.py` | the deploy gate — audits a live `/healthz`, then analyzes one video for real |
| `scripts/real_song_check.py` | *does* the pipeline run on real audio — the third gate |
| `scripts/seed_catalog.py` | *is the chart right* — seeds `chord_maps` and grades it against published transcriptions |
| `scripts/secret_check.py` | what each Modal secret really carries — key names, the proxy's shape, and the time budget re-derived from the values |
| `scripts/worker_check.py` | *did the worker image build an engine* — the gap a green `/healthz` cannot see |
| `scripts/admin.py` | §3's takedown CLI, from a laptop |

---

## The invariants, and where each one lives

§2 says these are architectural, not preferences: *"if a refactor breaks one of
these, the refactor is wrong."*

**Audio is never persisted** (§2.1). `app/analysis/scratch.py` refuses a scratch
root under `$HOME`, inside the working tree, or on any non-ephemeral mount;
cleans up on every exit path including exceptions; and verifies the directory is
gone rather than assuming. `tests/test_scratch.py` and `tests/test_pipeline.py`
assert the scratch root is empty after success **and** after failure. On Modal
the container-level half is that the worker mounts **no Volume** — see
`modal_app.py`.

**Only the derived map is stored** (§2.2). There is no column in `chord_maps`,
and no field on any type in `app/analysis/types.py`, that could hold PCM, a
spectrogram, a chroma matrix, or a path to a decoded file. Two tests assert the
absence, so "let's cache the chroma to make re-analysis cheap" has to add a
column and argue with them first.

**The chord map is not a product** (§2.3). There is no export endpoint, no chord
sheet, no public read path. `/v1/maps/{videoId}` requires the same Firebase
bearer as everything else and returns the same envelope gameplay consumes.

**Chord symbols only** (§2.4). No lyrics field exists anywhere. Section names come
from structure, never from text — `app/analysis/structure.py` names sections
`Part 1`/`Part 2` when it can't justify `verse`/`chorus`.

**Never paywall playback** (§2.6). The quota gates *analysis*. Cache hits are free
and never charged, so a song already in the Library always plays.

**The kill switch is one flag** (§3). `CHORDS_ANALYSIS_ENABLED=0`, read per
request, no deploy. Cached maps keep serving; only new jobs stop.

---

## Takedowns (§3)

```bash
export CHORDS_BASE_URL=https://…  CHORDS_ADMIN_TOKEN=…
python scripts/admin.py block --video dQw4w9WgXcQ --reason "DMCA 2026-08-03"
python scripts/admin.py audit
```

`block` blocks **and purges** in one request, in that order — so even if the
purge half fails, nothing is served from the moment the request lands. It prints
the row counts, because the handoff asks you to verify the cascade actually
cascaded. Channel blocks work the same way and also stop videos nobody has
analyzed yet. The audit log is append-only and `purge` deliberately doesn't touch
it: the record of a takedown outlives what it took down.

**Still the owner's task:** registering a DMCA agent (§18). Ask before any public
exposure.

---

## Cross-language contract test (§16.5)

Mo's best idea, taken verbatim: the backend writes its serializer's real output
as fixtures, and **the app's own importer is the final judge**, in CI, with no API
key.

```bash
.venv/bin/python tests/emit_reference_fixtures.py
```

Writes `tests/fixtures/emitted/` — two hand-written payloads round-tripped
through the serializer, plus the pipeline's own output for the known song at all
three difficulties and its sidecar. Output is **byte-stable across runs**: every
id (song, pattern, section, bar, stroke) is derived from content rather than from
the clock, so a diff means the analysis changed. CI regenerates them and fails on
any diff.

That claim used to be false, quietly: the sidecar carries `analyzedAt`, a wall
clock, so one of the six files differed on every run and a real change would have
been invisible in the noise. It is pinned in the emitter now — the importer cares
that the field is present and well formed, never what instant it names.

**App-side follow-up (small, not done here):** add a `ChordsBackendContractTests`
beside `MoBackendContractTests.swift`, pointed at these fixtures via
`CHORDS_BACKEND_FIXTURES` (mirroring `MO_BACKEND_FIXTURES`), walking each through
`CompositionPayload.from(jsonData:)` → `ComposerService.import` → `campfireSheet`
and asserting decode + **zero warnings**. Left for the app repo deliberately —
adding a file there means a pbxproj round-trip, which `ROSETTA_GP.md` §7 flags as
hazardous and wants diffed line by line.

---

## Two things the next session should know

**A finding from the client source, not in the handoff.**
`JamSongSheet.from(chart:barBeats:secondsPerBeat:)` derives each stroke's bar as
`floor(beat / barBeats)` on a **single song-level tempo and meter** — its own
comment says per-section overrides make bar grouping "approximate". Approximate
is fine for a self-paced campfire song and fatal for a video-synced one, because
the anchors would address beats the cursor never lands on. So a song carrying a
sidecar must have **one uniform grid**: no `tempoOverride`, no
`timeSignatureOverride`, and every section a whole number of bars. Tempo drift
lives in `beatAnchors`, which is what they are for. `lint_sync` enforces it; the
compiler never emits one. If a synced song ever walks off its own chart, look
here first.

**Flat vs bars mode is forced by the app's compiler, not by taste.**
`Compiler.compile` tiles the strumming pattern **per chord slot**, restarting it
at every chord — so a `beatsPerChord` smaller than the pattern's bar truncates
the groove and restarts it mid-bar, silently. Flat mode therefore requires one
chord per bar; anything else (a mid-bar change) must use `bars`, whose compiler
lays the pattern over the bar once and lets each stroke sound whichever chord is
active. And in bars mode the app **ignores `repeats`**, so the compiler expands
them and emits `repeats: 1` — leaving it above 1 plays a quarter of the section
with no error anywhere.

---

## Deploying, and what only breaks in the deployed shape

This deploys as its **own Modal app**, `rosetta-dechorder` — not as functions
added to Mo's `main` app. Same §19.2 argument that splits the two containers
below, one level up: a bad build, a deploy or a rollback here must not be able to
take Mo down with it. (Modal app names are `[a-zA-Z0-9-_.]+`, so "Rosetta
Dechorder" lands as that slug.)

```bash
CHORDS_SCALE_OUT=1 modal deploy modal_app.py                 # 1 ⇒ chords-secrets holds a Postgres DSN
CHORDS_BASE_URL=https://…modal.run python scripts/smoke.py   # the API container
modal run scripts/worker_check.py                            # the worker image
modal run scripts/secret_check.py                            # both secrets
```

`CHORDS_SCALE_OUT` is the operator asserting what is in `chords-secrets`, so the
API can be allowed more than one container (SQLite on a network volume tolerates
exactly one writer; Postgres does not care). It used to be spelled as the DSN
itself — which put a live database password in shell history to communicate one
bit, and predictably got skipped, leaving the deployment pinned to one container
while `/healthz` reported `"store": "postgres"` and looked perfectly healthy. The
old spelling still works so existing runbooks do not silently re-pin. `modal
deploy` prints which shape it chose; that line in the deploy output is the only
place the answer exists.

**Both halves, because neither can see the other.** `smoke.py` talks HTTP to the
API container; §4 gives the worker its own image, and the only channel between
them is a job row — so no HTTP check can tell you whether BTC's weights load.
`worker_check.py` runs the engines on synthesized audio inside the deployed
worker image and exits non-zero if a chord engine or beat tracker cannot run.

Real credentials — the service-account key, the DSN, the admin token, i.e. the
contents of both Modal Secrets — live in `security/`, which is gitignored as a
whole directory. `security/README.md` says what each file is and how to rotate
it. Nothing there is needed for the tests or a local run.

`scripts/smoke.py` is the deploy gate. It reads `/healthz` and checks the answers
against what a *production* deployment must look like, then — given
`CHORDS_ID_TOKEN` and `--video <id>` — runs one real analysis end to end and asks
for the same video again, because the only way to know a cache hit is free
(§16.4, §2.6) is to read the quota before and after. Exit code 0 means all of it
passed.

### The three that a laptop cannot show you

Local dev runs **one** process: the container that answers the request also owns
the engines and the audio stack. Modal runs **two**, and that is the whole point
(§4) — so a question like "can I fetch and decode?" has a different answer on
each, and code that asks it of the wrong one is correct locally and wrong in
production. All three of these were live in the first deployable build.

**1. The API container is not the thing that analyzes.** `POST /v1/analyze` chose
between a 202 and a 503 by asking whether *this* container had a fetch source and
registered engines. The API image is built with neither, deliberately — so every
uncached analysis answered `503 feature_disabled` while the worker sat there able
to do the job, and `/v1/me` reported `analysisEnabled: false`, which is the flag
the app uses to hide the affordance entirely. The capability now belongs to the
`JobRunner` (`can_analyze`), because the runner is what actually does the work:
`RemoteJobRunner` dispatches to a worker and says yes, the inline runner still
answers for itself. `/healthz` reports both — `fetch`/`engines` describe this
container, `canAnalyze` describes the service.

**2. `run_job` writes a terminal status on every exit path it controls — and a
SIGKILL is not one of them.** Modal's `timeout=300`, an OOM at the memory cap, or
a reclaim leaves the job row mid-flight forever. The player polls `analyzing`
until they give up, and worse, `active_job_for` keeps handing that dead id to
*everyone else* asking for the same video: one killed worker made one video
permanently un-analyzable, and `prune_jobs` could not help, since it only
collects rows that already reached a terminal status. Non-terminal rows now carry
a **lease** (15 minutes, comfortably above the worker's own timeout). Past it the
job is presumed dead: failed, so the poller gets an answer and the pruner gets a
row, and refunded, because the player got nothing for it.

**3. The cookie setting could not be configured where it was needed.**
`CHORDS_YTDLP_COOKIES` was a *path* — and Modal delivers secrets as environment
variables, while the worker mounts no Volume, so nothing in that container could
place a file for it to point at. `CHORDS_YTDLP_COOKIES_CONTENT` takes the file's
contents instead and materializes it 0600, once per process. The bot check is
also no longer just another entry in the "video unavailable" list: to the player
it is the same calm outcome, but to an operator it is the opposite of a private
video — nothing is wrong with that video — so it logs at ERROR level.

This was written up at the time as fixing "the bot-check escape hatch". It was
not one, and [measurement later showed why](#the-bot-check-is-per-ip-and-cookies-do-not-fix-it):
cookies make no difference to the bot check. The setting is still right to
support, for age-restricted and members-only video; it is just not the answer to
the failure this section thought it was.

A fourth, smaller: a dispatch that *fails* (Modal refusing a spawn) charged the
player for a job nothing would run, and then that stranded row blocked its video
via the same path as #2. It now fails the job, refunds, and answers 503.

None of this was visible to the test suite, because nothing could import
`modal_app.py` — `modal` is not a dependency of this package. `tests/test_deployment.py`
asserts the properties that file relies on, on the classes it uses.

The worker image builds the engines in: CPU torch (explicitly — the default wheel
carries the whole CUDA runtime for hardware this deployment doesn't have), the
BTC checkout and `beat_this`, both **pinned to a commit** rather than a branch so
two deploys of identical code can't install different models. Beat This!'s
checkpoint is downloaded at *build* time; left to run time it would be fetched on
the first request of every cold container, turning a cold start into a dependency
on someone else's file server inside a job that already has a timeout.

### The three that only the worker image can show you

The same lesson one level in. These are not about the two *containers* but about
the two *machines*: a laptop and a Debian image resolve different wheels, so an
engine that runs perfectly here can be dead there. All three shipped through a
**fully green `smoke.py`**, and all three failed in the chord or beat stage —
behind `canAnalyze: true`, `engines: {chords: [btc, …]}` and `isReady: true`,
because registration checks that a dependency and an adapter module *exist*, not
that the engine *runs*. `scripts/worker_check.py` exists to close exactly this
gap, and is what found them.

**1. `beat_this` pulled the CUDA `torchaudio`.** It requires torchaudio and does
not pin it, so resolving it from the default index got the CUDA wheel — which
links `libcudart.so.13` and cannot import on a CPU image. It surfaced at the
checkpoint bake, several layers from the line that caused it. torchaudio is now
named in the *same* `pip_install` as torch, so both resolve from the CPU index as
the matched pair they have to be. macOS wheels have no CUDA variant, which is why
no local run can reproduce it.

**2. BTC needs `mir_eval`, and nothing in this repo says so.** Its
`utils/mir_eval_modules` — where `idx2voca_chord` and the 170-label vocabulary
come from — imports it. That is a dependency of the *checkout*, invisible to a
grep of this repo, and present on any machine that has run the bench. The image
now installs it, and the build **imports BTC and asserts the vocabulary is 170
labels**, because `test -f` proves the files arrived, not that they load.

**3. PyTorch 2.6 flipped `torch.load(weights_only=)` to `True`.** BTC's 2019
checkpoint stores its feature mean and std as numpy scalars, which are not on
torch's default allowlist — so it raises `UnpicklingError` on torch ≥ 2.6 and
loads fine on anything older. Old torch locally, new torch in the image.
`adapters/btc.py` now allowlists the numpy types it actually needs rather than
switching `weights_only` off wholesale: the checkpoint is trusted (a
commit-pinned clone, baked at build time), but trusted is not a reason to grant a
pickle arbitrary-code rights.

A note on the build-time gate that came out of #2: prefer failing the *build*
over failing every job. A missing engine dependency is indistinguishable from a
healthy deployment until someone analyzes a video, and while YouTube's bot check
stands, nobody can.

Two Modal Secrets, and the split is §19.2 applied one level further in:
`chords-secrets` (API — Firebase, admin token, DSN) and
`chords-worker-secrets` (worker — **no auth credentials at all**, since it never
authenticates anyone). `CHORDS_SCALE_OUT=1` at deploy time is what lifts the
single-container pin; forgetting it leaves a SQLite deployment correctly pinned
rather than silently losing writes.

What is *in* the worker secret is invisible from outside — `/healthz` reports the
API container's own posture, and on this two-container shape that is correctly
"no fetch, no engines, no egress". `scripts/secret_check.py` asks the worker
instead, and prints key names and the proxy's scheme and host, never a value.

Keep this deployment's credentials separate from Mo's (§19.2). Same Firebase
project for identity, different service-account key: Mo never touches a
recording, and the blast-radius argument says keep it that way.

### The bot check is per-IP, and cookies do not fix it

This is the one place where the obvious mitigation is the wrong one, so it is
worth stating with the numbers rather than as an opinion.

Cookies were supplied — a real Netscape `cookies.txt` from a signed-in browser,
in `CHORDS_YTDLP_COOKIES_CONTENT` on the worker secret — and then measured. Ten
containers, ten distinct Modal egress IPs, three ordinary uploads each:

| | resolved | |
|---|---|---|
| without cookies | 6 / 30 | 20% |
| with cookies | 6 / 30 | 20% |

Identical. Two of the ten IPs resolved everything; the other eight resolved
nothing. A second sweep some minutes later got 0/15. The check keys on **IP
reputation, not on the account**, so a session credential buys nothing — and a
real credential you keep for no benefit is worse than one you never stored. The
cookies were removed from the secret and the local copy deleted.

Three consequences worth building on:

1. **Retrying is the mitigation, and production does it now** (2026-08-04). Each
   attempt lands on its own container and so its own IP, and a losing attempt
   returns after `probe`, before any audio is fetched, so it costs seconds and no
   bandwidth. `analysis_worker` raises `EgressBlocked` — a `VideoUnavailable`
   subclass, so the wire contract is untouched and only the worker can tell the
   difference — then retires its own container with
   `modal.experimental.stop_fetching_inputs()` and re-spawns the job.
   **The self-retirement is load-bearing:** without it a warm blocked container
   keeps taking the retry and drawing the same dead IP. Between attempts the job
   row is left alive, un-refunded and back at `queued`, so a polling client sees
   "still working" and never a failure that un-fails
   (`run_job(..., may_retry_elsewhere=True)`; the last attempt passes `False`, so
   the failure still lands somewhere).

   Verified live: nine blocked containers, then one through, and a real map
   written — about five minutes, which is why the iOS client's poll deadline had
   to move 240 s → 480 s.

   **The player client is not a lever *against the bot check*, and was measured
   before being dismissed.** Six containers × eight clients (`default`,
   `web_safari`, `mweb`, `android`, `ios`, `tv`, `tv_embedded`, `web_embedded`):
   the one clean IP served all eight, the five blocked IPs refused all eight. The
   check is on the address, so no client can move it.

   That sentence used to end "so a `player_client` knob was deliberately **not**
   added — it would only look like a lever", and the second half of that did not
   survive [the 403 below](#the-media-url-403-and-the-client-that-is-served),
   which is a different failure reachable from the same place and which the client
   is the *only* thing that moves. The knob exists now
   (`CHORDS_YTDLP_PLAYER_CLIENT`). Both facts are true at once, and the useful
   lesson is that "measured and dismissed" was recorded against a conclusion
   broader than the measurement.

2. **The fix is egress, and the retry budget now says so** (2026-08-04). Set
   **`CHORDS_YTDLP_PROXY`** to a *rotating residential* pool and the first
   attempt is meant to work. `egress_attempt_budget()` reads
   `YtDlpSource.egress` and spends accordingly:

   | egress | budget | what a retry is |
   |---|---|---|
   | `direct` | `MAX_EGRESS_ATTEMPTS_DIRECT = 12` | the mitigation — twelve draws at ~1/6 gets a job to ~89% |
   | `proxy` | `MAX_EGRESS_ATTEMPTS_PROXIED = 3` | insurance against a pool miss; twelve would just be eleven cold starts on the way to the same failure |

   Both adjectives on "rotating residential" are load-bearing. A **static
   datacentre** address — which is what a cloud provider's static-egress feature
   sells you, `modal.Proxy` included — is the exact fingerprint the check looks
   for, and is strictly worse than the unproxied default: it trades a 1-in-6
   draw for a permanent no. The cost is bounded by the cache, which is the part
   that makes it affordable: a map is stored per `videoId` and shared across
   every user, so a proxied fetch is paid once per *song ever* — a few MB at
   `-f worstaudio[abr>=64]`, roughly a cent at 2026 pricing — and nothing after.

   `/healthz` reports `egress` — `"direct"`, `"proxy"`, or `null` on a container
   with no audio stack — **which on the deployed shape means the API container
   always answers `null`.** That is the honest answer, not a gap: fetching lives
   in the worker, the API image has `source=None` by design (§4), and the proxy
   credential is deliberately not in a container that has no use for it. So
   `scripts/smoke.py` warns when it *can* see an unproxied deployment (a local
   or single-container run) and prints "not visible from the API container"
   otherwise, rather than marking a correctly-configured deployment red. Ask the
   thing that knows, or say you cannot see — the same rule that made `canAnalyze`
   ask the runner instead of this container's own capabilities.

   Unproxied is a real deployment; it just means the success rate tracks
   Google's datacentre policy rather than anything in this repo.

   **Rotation has to be per *fetch*, not per *request* — a third load-bearing
   word** (2026-08-05). "Randomize IP" is the setting a rotating pool ships with,
   and it rotates on every HTTP request. yt-dlp makes several per fetch, and a
   `googlevideo` media URL is **bound to the IP that resolved it**, so the player
   response arrives on one address and the bytes are requested from another:
   **HTTP 403**, every time, while `probe` — which needs no such consistency —
   succeeds and makes the job look healthy right up to the download. Measured in
   the worker image, download stage, per attempt:

   | video | per-request rotation | sticky session |
   |---|---|---|
   | `8ui9umU0C2g` (official label upload) | **0/3** — 403, 403, bot | 2/3 |
   | `36X3wecT2z8` blues-in-e | 2/3 | 3/3 |
   | `85Sqw6FTxm4` canon-fingerstyle | 2/3 | 3/3 |

   `ytdlp_source._sticky_proxy` therefore pins a fresh
   `_session-<id>_lifetime-10m` per **analysis** — probe and fetch inherit one
   session, a new draw between analyses, so `EgressBlocked` still has somewhere to
   retry into. Do **not** pin a session in `CHORDS_YTDLP_PROXY` itself — that is
   one address for every job forever, i.e. the static-datacentre failure above.
   `real_song_check.py`'s `label-owned-fetch` entry exists to fail if this
   regresses, and did exactly that on 2026-08-17 — for the other reason, below.

   The reading recorded here at the time was that "the binding is enforced on
   label-owned content and slack on ordinary uploads". The table is real and it is
   the measurement that motivated the sticky session, but that generalisation was
   drawn from three videos and it is **wrong**: re-measured on 2026-08-17, the
   ordinary uploads fail on exactly the same clients as the label-owned one. What
   the sticky session fixes is the *rotation*, for everyone.

3. <a name="the-media-url-403-and-the-client-that-is-served"></a>**And the
   client the media URL is resolved through — which broke every fetch**
   (2026-08-17). A sticky session is necessary and stopped being sufficient. Every
   video in the corpus came back `MediaUrlRefused`: probe served, formats listed,
   403 on the bytes, from the *same* address that had just resolved them
   (verified: three requests through one pinned session, one IP, three times).

   The variable that moved it was the **player client**. Measured in the worker
   image, one sticky session per attempt, on an ordinary upload, an official label
   upload and a Beatles remaster:

   | player client | `36X3wecT2z8` | `8ui9umU0C2g` | `CGj85pVzRJs` |
   |---|---|---|---|
   | `default` (`android vr`) | 403 | bot check | bot check |
   | **`android`** | **ok (fmt 18)** | **ok (fmt 18)** | **ok (fmt 18)** |
   | `android_music` | 403 | 403 | 403 |
   | `android_creator` | 403 | 403 | 403 |
   | `tv_embedded` | 403 | 403 | 403 |
   | `web`, `web_embedded` | no format available | no format available | no format available |
   | `tv` | DRM protected | — | — |

   Every client that offers the audio-only `140` stream is refused the bytes, and
   the one client that is served offers only progressive `18`. So the fetch now
   downloads ~11 MB of video for a four-minute song and throws the picture away in
   ffmpeg — paid once per song ever, because the map is cached and shared, and
   strictly better than the alternative of not fetching at all. `_PLAYER_CLIENT`
   in `ytdlp_source` carries the table; `CHORDS_YTDLP_PLAYER_CLIENT` overrides it,
   because the one certainty is that this moves again.

   It was **not** a stale yt-dlp: the worker had 2026.07.04 and `pip install -U`
   in the same container returned the same version. Nothing in this repo changed
   to cause it and nothing in this repo could have caught it — no unit test
   reaches YouTube, so the 656-test suite, `/healthz` and `smoke.py`'s cache-hit
   proof were all green while every real analysis failed.

4. **Label-owned music is not a separate wall.** This section used to say the
   opposite — "every official Beatles upload in `bench/corpus.json` is refused in
   every player client", offered as a second, categorical obstacle — and the
   2026-08-17 re-measurement retires it. Under the residential pool and the
   `android` client, `--songs isophonics` analyzes **2 of 3** Beatles remasters
   end to end, with the tempo landing on the Isophonics ground truth:

   | | chords | tempo | truth |
   |---|---|---|---|
   | Let It Be | 150, 98% on C/G/Am/F | 70.0 bpm | 69.85 |
   | Michelle | 111, 64% on expected roots | 115.0 bpm | 117.42 |
   | Norwegian Wood | bot check on both attempts — retryable, not a wall | | |

   Why the old claim looked solid: it was measured unproxied, where the bot check
   made every attempt fail, and "the Beatles are blocked" is a more satisfying
   story than "this IP is blocked". The three ids are ordinary videos to YouTube's
   delivery layer. What is *not* retired is the rights posture — nothing about
   fetchability changes what [`RIGHTS.md`](RIGHTS.md) says about what may be
   stored or shipped, and the Isophonics ids stay out of the seeded catalog for
   that reason and not for this one.

   One artefact of the old era is worth keeping: on the single occasion cookies
   cleared the bot check, YouTube answered with storyboard images (`sb0`–`sb3`,
   mhtml) and no audio format — the PO-token/SABR path. A PO-token provider is
   still **not** the missing piece, and it is worth not spending a week finding
   that out: yt-dlp's own provider documentation states that passing PO tokens no
   longer clears the bot check in the majority of cases. Egress plus the right
   client is the stable lever.

`ytdlp_source` still accepts both cookie settings. They are the right mechanism
for age-restricted and members-only video, and worth revisiting if egress ever
stops being a datacentre IP — they are just not an answer to a bot check.

### What real audio actually produced

`scripts/real_song_check.py` is the third gate: `probe` → `gate` → `decode` →
beats → chords → onsets → `assemble`, on real recordings, in the deployed image.

| | blues-in-e | canon-rock |
|---|---|---|
| chords | 112 bars, 164 spans | 23 sections, 221 bars |
| roots | E:89 B:40 A:35 — **100%** on E/A/B | **100%** within D/A/Bm/F#m/G |
| tempo | 91 bpm (upload says 90) | 200 bpm, 4/4 |
| sync sidecar | yes | yes |
| speed | 302 s audio in 64 s (0.21× realtime) | 321 s in 68 s |

A 12-bar blues came back as exactly E, A and B; Canon Rock's cycle survived a
distorted band mix.

This section used to add that the `Em`/`E` and `Am`/`A` split in the blues was
"the blue third being genuinely ambiguous, not an error". Half of that was
right and the conclusion was not. Measured on the recording — a CQT chroma,
outside this repo's engines entirely — the **major** third wins on all three
roots: G# 0.080 vs G 0.071, C# 0.064 vs C 0.061, D# 0.085 vs D 0.057. So the
minor readings are errors. The blue note is the *explanation* for them (the
pitch that separates E from Em really is in the audio, which is why the margins
are thin), not a licence for them. See [below](#what-the-seeded-catalog-says).

Note also that until this was written the gate had been reading the **easy**
tier while its code said `intermediate` — a tier that did not exist, whose `or`
fallback silently resolved to `sorted(...)[0]`. Every root histogram this gate
printed before then described the simplified chart. (The tiers have since been
removed outright, so there is no longer a wrong one to read.)

The gate also surfaced a real limit. **Solo fingerstyle guitar degrades**: no
percussion, so the beat grid is too weak to align, and the result comes back
`lowConfidence: true` with `hasSync: false` and a flat two-chord stub instead of
an arrangement. That is the system reporting its own weakness rather than
lying — the behaviour we want — but sparse instrumental audio is a real weak
spot, and §8's numbers come from full-band recordings.

---

## What the seeded catalog says

`real_song_check.py` asks *did the pipeline run*. `scripts/seed_catalog.py` asks
the harder question — **is the chart right?** It runs the real `run_job`, so the
maps land in `chord_maps` exactly as a user's analysis would; it then reads each
chart back **out of the store** and grades it against published transcriptions
at sounding pitch.

```bash
modal run scripts/seed_catalog.py                  # the whole seed set
modal run scripts/seed_catalog.py --songs zombie   # one of them
```

Twelve songs, chosen to climb from three chords to seven, with a compound-meter
case and a jazz case at the top. Eleven analyzed; the twelfth produced no chart
at all.

| Song | Verdict | Key | Tempo | In vocabulary | Cycle found |
|---|---|---|---|---|---|
| Hey Joe | correct | Am *(want E)* | 60 ✓ | 100% | C–G–D–A–E ✓ |
| Stand By Me | correct | A ✓ | 120/118 ✓ | 100% | A–F#m–D–E ✓ |
| Wonderwall | correct | F#m ✓ | 88/87 ✓ | 100% | F#m–A–E–B ✓ |
| Sweet Home Alabama | correct | G *(want D)* | 97 ✓ | 98% | D–C–G ✓ |
| Canon in D | correct | D ✓ | 77/64 | 99% | D–A–Bm–F#m–G–D–G–A ✓ |
| Wish You Were Here | correct | G ✓ | 2× | 96% | Em–G–Em–A–G ✓ |
| Knockin' on Heaven's Door | correct | G ✓ | 67/74 | 93% | G–D–Am / G–D–C ✓ |
| Hotel California | correct | Bm ✓ | 73/75 ✓ | 93% | Bm–F#–A–E–G–D–Em–F# ✓ |
| Autumn Leaves | correct | Gm *(want Bb)* | 111/112 ✓ | 89% | Cm–F–Bb–Eb–Am–D–Gm ✓ |
| Zombie | mostly | Bm ✓ | 83/84 ✓ | 83% | Bm–G–D–A ✓ |
| Blues in E | **partial** | Am *(want E)* | 91/90 ✓ | 58% | E–A–E–B–A–E ✓ |
| House of the Rising Sun | **no chart** | — | — | — | — |

Structure holds up where a song has any. Wonderwall alternates verse (F#m–A–E–B)
against chorus (D–E–F#m); Hotel California separates its verse from the
G–D–Em–F# chorus; Wish You Were Here splits the Em–G figure from the C–D–Am–G
refrain. Canon in D came back `lowConfidence` with **no sidecar**, which is the
solo-fingerstyle weakness above, reporting itself correctly.

### Three defects it found

**Compound meter is unanalyzable, not merely inaccurate.** House of the Rising
Sun (6/8) produced nothing: the beat tracker locked onto the eighth-note
triplets, reported 231 BPM, and the 40–220 guard rejected the job as
`tempo_unreadable`. 231 is ≈3× the 77 BPM it is actually in, so the tracker is
hearing the subdivision rather than a different song — and `tempoOctaveShift`
already exists for the 2× case. The 3× case has no path.

**Dominant harmony reads as minor.** The blues is the one wrong chart, and the
grading resolution is what exposes it: the **roots are perfect** — E, A and B
account for every chord — while the qualities flip between major and minor on
the same root, in a song with no minor chord in it. Root-only accuracy scores
this 100% and reports nothing, which is exactly how it survived the gate above.

**The key model is weakest where the chords are right.** Hey Joe scored 100% on
chords and was labelled Am; the blues was labelled Am. Sweet Home Alabama's "G
vs D" and Autumn Leaves' "Gm vs Bb" are defensible conventions (Mixolydian, and
the relative-major pair) — the first two are not.

### And one thing it got wrong about itself

Zombie first scored 50% and read like an engine failure. It was not: that cover
is **up a fifth** (Bm–G–D–A, not the record's Em–C–G–D). The recording settles it
independently of anything in this repo — C is the *least* present pitch class of
the twelve, and Zombie in Em spends a quarter of its length on C. The truth
entry was wrong, not the chart.

That is the failure mode this file has to be most careful about, and the reason
every entry in it carries its reasoning: a cover is under no obligation to be in
the original's key, so a "wrong" chart is always two hypotheses, not one.

---

## The analysis audit (2026-08-18)

An external review of the chord-analysis half — engines, theory layer, structure
— traced back from four symptoms the owner reported: sevenths in almost every
song, major and minor sharing a key, `A` and `A#` in one chart, and several
distinct "verses" where the song has one. Thirty-four findings, written up with a
verdict against each in [`ANALYSIS-AUDIT.md`](ANALYSIS-AUDIT.md).

What it moved, on the ten-song chart corpus (`python bench/lab.py grade`):

| | root | triad | form | distinct chords |
|---|---|---|---|---|
| before | 0.853 | 0.846 | 0.688 | 71 |
| after | 0.854 | 0.849 | **0.755** | **61** |

and on `easy` — then the tier a beginner was shown, since removed — root
0.761 → **0.791**.

**Read the chord count next to the chord score.** The roots were mostly right
already; what the owner was seeing is ten spurious chords across ten songs —
Someone Like You reported eleven distinct chords against a reference of five, and
now reports six. The three changes that did it are a **Viterbi decoder on BTC's
posteriors** (which were being arg-maxed away frame by frame), a **key-consistency
audit** (§20.10, `keyaudit.py` — the only layer whose evidence did not come from
counting engine output, and so the only one that can settle a *systematic*
mishearing), and **per-occurrence gates in the consensus vote**, which used to
abandon a whole slot the moment one occurrence was too wrong to fix.

Six of the review's recommendations were implemented and then **reverted on
measurement**, and that is the part worth keeping: beat-synchronous decoding costs
five points of root, novelty-based segmentation costs eleven, and a
confidence-based tuning-sign check takes one song from 0.335 to 0.012. Which is
why the first thing built was the ruler — `python bench/lab.py layers` now scores
every song once per posture, engine-only through each theory layer, so "is this
the engine or the theory layer?" is a question with an answer per song:

```
posture                  root  triad   form
engine only             0.809  0.789  0.702
+ vocabulary  §20.8     0.809  0.789  0.702
+ key audit   §20.10    0.809  0.793  0.713
+ consensus   §20.4     0.821  0.812  0.713
+ belief      §20.9     0.821  0.812  0.713
+ form        §21       0.854  0.849  0.755
```

Most of the remaining error is the engine's.

## The service audit (2026-08-17)

A read of the whole service — HTTP surface, store, deployment shape, and the two
adapters — rather than of one subsystem. Twenty-four findings, all fixed. What is
worth recording is not the list but which of them **no existing test or number
could have caught**, and why.

### The four that would have shipped

**Uploaded audio was public.** `list_catalog` selected every row in `chord_maps`
with no source filter, and uploads are written to that same table — so one
player's private recording appeared on every other player's home screen, under
their own filename, with their chart, key and tempo. `GET /v1/maps/up_<hash>`
served it to anyone with the hash. `is_upload_id()` had been written for exactly
this question and was called nowhere. The reason 547 green tests missed it: the
suite had **one identity** in it, so "somebody else" was not expressible. The fix
is an `owner_uid` column and `owner_uid IS NULL` in the two queries; the tests
needed a two-identity authenticator before they could fail.

**On Modal with SQLite, no analysis could ever complete.** The worker calls
`build_store` in its own container. The `chords-data` Volume is mounted on the API
function only and `db_path` is relative — so a SQLite worker opens a *new* database
on a disk that dies with the call, writes `analyzing`, `ready` and the finished map
into it, and the API never sees any of them. Every job sits at `queued` until the
900 s lease reaper fails it. Both containers report a green `/healthz` throughout,
because individually each one is fine. The `MAX_CONTAINERS = 1` pin does not cover
this and never could: the worker is a separate container whether or not the API is
pinned. `build_store(role=ROLE_WORKER)` now refuses, loudly, naming the remedy.

**Two players, one video, and the second gets a job id they can never poll.**
`active_job_for` joins the second caller onto the first's in-flight analysis —
correctly, since decoding the same recording twice produces the same answer — but
the job row carries one `uid` and the poll route refused anyone else. So the second
player was told, forever, that the analysis had expired; retrying returned the same
dead id, and one player asking first made that video permanently un-analyzable for
everybody else. A `job_followers` row is the smallest fix that does not turn the id
into the credential: you may poll a job because you *joined* it. Joining is free,
on the same reasoning that makes a cache hit free (§16.4).

**Rate-limit 429s were invisible to browsers.** `add_middleware` *prepends*, so the
middleware installed last runs first and sits outermost — and the limiter was
installed after CORS, which put it *outside* `CORSMiddleware`. Its 429 left the app
without ever passing through the CORS layer, so it carried no
`Access-Control-Allow-Origin`, and a browser cannot read an opaque response: the
`Retry-After` header and the `rate_limited` code were both being set and both
unreachable from the web client, which saw a network error. Native clients ignore
CORS entirely, which is why it survived. Swapping the two install calls fixes it;
`OPTIONS` is now also exempt, because a preflight is the browser's request rather
than the player's and counting them halves a web client's real budget.

### The performance findings, measured

| | old | new |
|---|---|---|
| `/v1/catalog` first page, 2000 maps | 249 ms | **15 ms** |
| `/v1/catalog` at `offset=1900` | 257 ms | **17 ms** |
| engine construction per job (warm container) | 837 ms | **~0 ms** |
| `assemble()` per real track | 359 ms | **275 ms** |
| `detect_key()` on a 400-span track | 3.5 ms | **2.5 ms** |
| peak memory per upload body | 2.00× | **1.13×** |

The catalog was reading and JSON-decoding the entire table on every request to
return sixty rows, so `offset` bought nothing — page 30 cost what page 1 did. Both
engines were being rebuilt per job (`_lazy` constructs a fresh adapter, whose
`_load()` caches on `self`), so BTC's checkpoint and Beat This!'s model were
re-read on every analysis in a container that had already loaded them. The upload
path buffered each body twice — `b"".join(chunks)` while the chunk list is still
live — and that one is only *known* fixed because the test measures peak
allocation: the obvious `bytearray` rewrite was still 2.00×, because `bytes(buffer)`
is the same second copy under a different name.

### The timeout chain, which did not add up

Four numbers in three files described one job's wall clock and contradicted each
other in every direction: stage ceilings summing to 405 s, a container killed at
300 s, a job deadline claiming 180 s, and a 900 s lease. The failure that produced
is the worst shape available — a *successful but slow* fetch SIGKILLed with no
terminal status written, so the player watched a spinner until the reaper noticed
fifteen minutes later. They are now ordered, and `tests/test_deployment.py` asserts
the ordering rather than the comment describing it:

    probe + fetch + decode + dsp_reserve  ≤  job deadline (450)
                                          <  worker timeout (600)
                                          <  job lease (900)

Read outward, each layer is the backstop for the one inside it. And a deadline
breach is now refundable — it used to raise a bare `AnalysisError`, whose
`analysis_failed` code is deliberately *outside* `REFUNDABLE_CODES`, so players
paid a daily analysis for our own capacity planning.

### Two findings where the honest answer is "smaller than it looks"

The BTC feature extractor's block-edge corruption is real, measurable, and worth
about **+0.003** raw chord accuracy and **nothing** delivered — see
[the feature extractor](#the-feature-extractors-block-edges-fixed-2026-08-17) for
the numbers and why a transformer over 108 frames absorbs it.

Per-difficulty video sync was a **latent** fix rather than a measured one, and
the tiers it applied to have since been removed. The sidecar used to be withheld
from every tier on the first tier that disagreed with it, so one `easy` render
coming out a section short cost `hard` and `normal` their video sync too — and
the log named only the failing tier, so the two that were fine looked untouched.
On the benchmark corpus the disagreement was always a property of the *recording*
(all three tiers failed together), so the sidecar count was 13/15 before and
after. What survives of it is `fit_sync`, which trims the anchors to the chart
rather than reporting the overrun as a disagreement.

### Still open

**`probe` and `_fetch` remain two yt-dlp invocations.** They must stay two, because
§3's blocklist is per-channel and §18's cap is a rejection, so both have to be
decidable before a byte of audio moves. What *was* fixed is the more consequential
half: they now share one sticky egress session, so the address that cleared the bot
check is the one that downloads. The old comment claimed two independent draws were
better; that is true when either will do and false when you need both —
`1 − p²` against `1 − p`, which at a measured `p ≈ 0.2` is the difference between a
96% and an 80% chance of a job hitting the check. Reusing the probe's info JSON with
`--load-info-json` would remove the second extraction entirely, and was **not** done:
a googlevideo URL is bound to the resolving IP and time-limited, so the failure mode
is a 403 inside the fetch stage with no budget left to retry — in the part of the
system that is least testable from here and has the longest history of subtle egress
bugs. Worth doing against a live deployment, not blind.

**The Postgres-specific tests are unverified.** The limiter's count-then-insert race
is real on Postgres under READ COMMITTED and is fixed with a transaction-scoped
advisory lock (`_serialize_rate_key`), but there is no Postgres on this machine, so
`tests/test_store_postgres.py` — including the threaded test that would catch it —
skips. CI supplies one; that is where those five new tests first run.

---

## What is still owed, and by whom

Nothing in the backend's own scope is open. What remains needs an account, a key,
or a lawyer — every item below is console work, and none of it is code.

**To get it deployed:**

1. **Modal account** — `pip install modal && modal setup`.
2. **A Postgres DSN.** Modal does not host one; any managed provider works (Neon
   and Supabase both have a free tier that fits this). Nothing needs creating in
   it — `_migrate()` builds the schema on first connect.
3. **A Firebase service-account key**, from the **same project as Mo** for
   identity but a **different key** (§19.2). Mo never touches a recording, and
   the blast-radius argument says keep it that way.
4. **An admin token** — any long random string; it is the §3 takedown credential.
   `python -c "import secrets; print(secrets.token_urlsafe(32))"`.
5. **The two Modal Secrets**, in the dashboard (`modal secret create --force`
   replaces the *whole* secret and would drop keys):
   - `chords-secrets` → `FIREBASE_PROJECT_ID`, `FIREBASE_SERVICE_ACCOUNT_JSON`,
     `CHORDS_REQUIRE_AUTH=1`, `CHORDS_ADMIN_TOKEN`, `CHORDS_DATABASE_URL`,
     `CHORDS_RATE_LIMIT_IP_PER_MIN` (60 is a reasonable start),
     `CHORDS_RATE_LIMIT_PER_MIN` (10). `CHORDS_RATE_LIMIT_POLL_PER_MIN` is
     deliberately **not** on this list: job-status polls need their own, much
     larger budget, and defaulting it in code rather than in a hand-edited
     secret is what makes that true of every deployment rather than of the ones
     somebody remembered to update.
   - `chords-worker-secrets` → `CHORDS_DATABASE_URL` **and no auth credentials at
     all**. The worker authenticates nobody.
   - `CHORDS_DEV_TOKEN` must appear in neither — `CHORDS_REQUIRE_AUTH` refuses to
     start if it is set.
6. **Deploy and gate it:**
   ```bash
   CHORDS_SCALE_OUT=1 modal deploy modal_app.py
   CHORDS_BASE_URL=https://…modal.run python scripts/smoke.py
   ```
7. **One real analysis**, with a Firebase ID token for a verified account:
   ```bash
   CHORDS_ID_TOKEN=… python scripts/smoke.py --video QDYfEBY9NM4
   ```
   Expect the bot check here rather than later — see below.
8. **A daily quota number.** `CHORDS_DAILY_QUOTA` defaults to 10, which is a
   placeholder, not a recommendation: a fetch, a decode and two neural models of
   worker CPU per analysis is what it is spending. The DSP half is ~10 s per track
   now that the engines are built once per container rather than once per job; the
   fetch is the variable part.

**Not deployment, but still owed:**

| | |
|---|---|
| Register a DMCA agent (§18) — blocks public exposure, not development | owner |
| Media IP lawyer review (§10) | owner |
| §19.1: whether Phase 2 reverses the "no backing track" canon | owner |
| `ChordsBackendContractTests` in the app repo (§16.5, above) | app repo |

**The two failures to expect first.** YouTube answers datacentre IPs with a bot
check far more often than residential ones; when it happens the worker logs it at
ERROR level, distinctly from a video being private. Cookies do **not** fix it —
see [the section below](#the-bot-check-is-per-ip-and-cookies-do-not-fix-it) —
and the honest mitigation today is retrying on a fresh container, because each
one is a fresh IP. And the worker's 300 s
Modal timeout against ~47 s of DSP leaves room for a 10-minute video but not for
much retrying; if fetches get slow, that budget is the thing to watch — a job
that blows it is now reaped and refunded rather than left in flight, but it is
still a job the player didn't get.

## Standing note (§10)

The user was told, correctly, to have a media IP lawyer review this architecture
before launch. If the build starts drifting toward storing audio, exposing chord
data, or adding lyrics/tab, **stop and flag it** rather than quietly implementing
it. Those aren't features with a legal footnote — they're the difference between
this feature and the one that got sued.

§19's two owner calls are still open: whether Phase 2 (a record playing under the
player's strums) reverses the "no backing track" canon, and the lawyer review
above. Phase 1 — a video becoming a Library song the player plays self-paced —
sidesteps the first entirely, which is why the handoff says build it first.

Worth restating now that the fetch stage exists rather than being a seam: §2
concedes openly that fetching and decoding audio separated from video
contravenes the YouTube API terms as written, and the owner is knowingly
accepting that risk. The mitigation is blast radius — `ytdlp_source.py` is
imported only by the worker, its dependencies are not in the base install, and
CI asserts the API surface cannot import them. That argument only holds while
those three remain true.
