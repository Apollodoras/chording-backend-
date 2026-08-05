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
`CompositionPayload` v2 and a `videoSync` sidecar come out. 414 tests, green, no
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

The rights posture — what is stored, what is not, which Chordify surfaces are
deliberately not cloned, and the two rules the **client** repo has to keep so
App Review 5.2.3 stays a non-event — is [`RIGHTS.md`](RIGHTS.md).

| | |
|---|---|
| §3 operational surface | ✅ kill switch, per-video + per-channel blocklist, admin block/purge/offset, append-only audit log, verified purge cascade |
| §12 `CompositionPayload` v2 | ✅ emitter + the app's importer lint, ported from Mo |
| §12.2 chord normalization | ✅ Harte + symbolic → the app's closed grammar |
| §5.4 post-processing | ✅ quantize → merge → drop → hold N/C → simplify → confidence gate |
| §5.5 difficulty tiers | ✅ re-scoped against the grammar ceiling (§12.2) |
| §15 sections | ✅ superseded by §20.3 — fuzzy, global, phase-aligned repeat groups |
| §14 strumming patterns | ✅ fold/histogram/convention + quarter-note fallback, pooled per repeat group (§20.4) |
| §13 `videoSync` sidecar | ✅ beat anchors + the §13.2 invariant, enforced by lint |
| §16 API | ✅ Mo-shaped: Firebase bearer, `{message, code}` errors, job-id + poll |
| §16.5 contract fixtures | ✅ emitted and byte-stable; the app-side test is a small follow-up (below) |
| §5.1 fetch + decode | ✅ yt-dlp + ffmpeg, bounded, behind the §4 seam — **plus an upload path** with no YouTube-terms exposure ([`RIGHTS.md`](RIGHTS.md)) |
| egress | ✅ **`CHORDS_YTDLP_PROXY` is live** (IPRoyal residential, rotating) — verified clearing the bot check on the first attempt, twice, on real audio ([below](#the-bot-check-is-per-ip-and-cookies-do-not-fix-it)) |
| the beat axis | ✅ one origin for chart, bars and anchors (`axis.py`) — the defect that cost 23 points |
| §20 theory layer | ✅ meter reconciled against the harmony, repeat groups, gated consensus, modal key, one model rendered per tier |
| §21 two-sided benchmark | ✅ consensus is a **provable no-op** on perfect input, and measured separately on real engines |
| §5.2/§5.3 engines | ✅ **BTC + Beat This!**, benchmarked against real recordings (below) |
| §4 two-container shape | ✅ the API delegates to the worker; `tests/test_deployment.py` covers what `modal_app.py` relies on |
| CI | ✅ suite, Postgres, fixture stability, and a test that the API image cannot touch audio |
| Deploy gate | ✅ `scripts/smoke.py` — `/healthz` audit, one real analysis, and a proof that the cache hit is free |

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
.venv/bin/python -m pytest              # 414 tests, ~14s, no network, no audio stack
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
beats too. Note that `delivered` is scored on the **`hard`** tier — `normal`
deliberately folds diminished and augmented onto their nearest playable triad
(§5.5), and charging the pipeline for a reduction it was asked to make reads
Michelle as 0.812 instead of 0.952.

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

So overwriting requires **three independent gates** (§20.4): two-thirds
agreement, a harmonically *near* disagreement (C↔Am is a mishearing, C↔F is a
chord change), and a dissenter that was believed **less** than the winner. The
third gate is what makes the design testable — ground truth arrives at a flat
confidence of 1.0, so **on perfect input consensus is provably a no-op**, from
the construction rather than from luck.

`python bench/run_bench.py --theory` runs it twice, because the two ways it can
be wrong pull in opposite directions:

| run | consensus off | on | bars rewritten |
|---|---|---|---|
| ground truth as both engines | **0.939** | **0.939** | 0 |
| BTC + Beat This! | **0.796** | **0.799** | 15 |
| BTC + Beat This!, pre-§20 commit | **0.796** | — | — |

Read honestly, because the temptation is to read it the other way:

- **The architecture is delivered-neutral.** Meter reconciliation, the new form
  detection, the model/render split and the modal keyfinder together move the
  number by nothing. What they bought is coherence and provenance — repeats
  collapse, the three tiers agree by construction, sections carry group
  identity, and the sidecar reports what was changed — not accuracy.
- **Consensus is a marginal win**: +0.003, with Michelle up 0.028 and Let It Be
  *down* 0.014. Real, but within noise on nine tracks. That is why
  `CHORDS_THEORY_CONSENSUS` exists, and why the harness prints MARGINAL rather
  than PASS below half a point.
- **Key detection is a wash**: 5/9 exact tonics before and after. The modal fix
  is still right on its own terms (`G F C G` was called *A minor*; it is G
  mixolydian), and so is capping the tonic-endpoint bonus.

Anyone extending this should assume the remaining accuracy is in the engines,
and in §20.2's downbeat-phase check on tracks where the tracker is genuinely
wrong — not in voting harder.

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
| **btc** | **0.805** | 0.886 | 0.876 | 10.3 |
| chroma (control) | 0.531 | 0.702 | 0.895 | 30.5 |

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

Cost: ~47 s of DSP for a 3-minute song on CPU, against a 180 s job deadline. BTC
on a GPU was not pursued — §18 anticipated that a GPU cold start per job may cost
more latency than the accuracy gap costs quality, and at 10 s/track on CPU there
is nothing to buy.

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
                       §20.6 ONE model ──► rendered once per difficulty
                                       │
                       §12 compile ──► lint ──► Postgres: chord_maps
GET /v1/analyze/{jobId} ──► status / {song, videoSync}
```

Two input paths, and the difference is legal rather than technical: `/v1/analyze`
fetches a YouTube recording (which §2 concedes contravenes the API terms as
written), `/v1/analyze/upload` takes audio the player already has and carries no
such exposure. Everything downstream of `decode` is identical, and the upload
path is what §3's kill switch degrades *to* rather than degrading to nothing.
See [`RIGHTS.md`](RIGHTS.md).

| Module | What it owns |
|---|---|
| `app/payload.py` | `CompositionPayload` v2 — a near-copy of Mo's, `yt:` ids |
| `app/chords.py` | the app's grammar (ported) + §12.2 normalization + §5.5 tiers |
| `app/lint.py` | the importer's checks (ported) + `lint_sync`, the §13.2 invariant |
| `app/sync.py` | the sidecar, anchors, and the client's interpolation in Python |
| `app/store.py` | maps, jobs, blocklist, audit log, quota, limiter — two backends |
| `app/analysis/` | the pipeline; `scratch.py` is §2.1 in code |
| `app/analysis/axis.py` | **one** beat axis — chart, bars and anchors share an origin by construction |
| `app/analysis/harmony.py` | §20.1 — harmonic distance: is a disagreement a mishearing or a chord change? |
| `app/analysis/meter.py` | §20.2 — the harmony's second opinion on where the bar starts |
| `app/analysis/form.py` | §20.3 — repeat groups, found fuzzily and globally (supersedes §15's segmentation) |
| `app/analysis/consensus.py` | §20.4 — the three-gate vote; the only code that edits a chord the engine reported |
| `app/analysis/model.py` | §20.6 — the song model; the tiers are renders of it, not separate analyses |
| `app/analysis/ytdlp_source.py` | the only code that ever holds audio — worker image only |
| `app/analysis/file_source.py` | the upload path: player-supplied audio, no YouTube-terms exposure |
| `app/analysis/adapters/` | one file per engine; nothing else imports a model |
| `app/main.py` | the HTTP surface, shaped like Mo's |
| `modal_app.py` | two functions, **two images** — that split *is* §4's isolation |
| `bench/fetch_corpus.py` | annotations + recordings → a scoreable corpus |
| `scripts/smoke.py` | the deploy gate — audits a live `/healthz`, then analyzes one video for real |
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
CHORDS_DATABASE_URL="$(the DSN in chords-secrets)" modal deploy modal_app.py
CHORDS_BASE_URL=https://…modal.run python scripts/smoke.py   # the API container
modal run scripts/worker_check.py                            # the worker image
```

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
authenticates anyone). Passing `CHORDS_DATABASE_URL` at deploy time is what lifts
the single-container pin; forgetting it leaves a SQLite deployment correctly
pinned rather than silently losing writes.

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

   **The player client is not a lever, and was measured before being dismissed.**
   Six containers × eight clients (`default`, `web_safari`, `mweb`, `android`,
   `ios`, `tv`, `tv_embedded`, `web_embedded`): the one clean IP served all eight,
   the five blocked IPs refused all eight. So a `player_client` knob was
   deliberately **not** added — it would only look like a lever.

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

3. **Label-owned music is a second, separate wall.** Every official Beatles
   upload in `bench/corpus.json` is refused in every player client. On the one
   occasion cookies *did* clear the bot check, YouTube answered with storyboard
   images (`sb0`–`sb3`, mhtml) and no audio format at all — the PO-token/SABR
   path, which needs a token provider, not a credential. The corpus ids are kept
   under `--songs isophonics` so the situation stays checkable.

   A PO-token provider is **not** the missing piece, and it is worth not
   spending a week finding that out: yt-dlp's own provider documentation now
   states that passing PO tokens no longer clears the bot check in the majority
   of cases. It is a moving target maintained against a service that keeps
   changing it. Egress is the stable lever.

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
distorted band mix. The `Em`/`E` and `Am`/`A` split in the blues is the blue
third being genuinely ambiguous, not an error.

The gate also surfaced a real limit. **Solo fingerstyle guitar degrades**: no
percussion, so the beat grid is too weak to align, and the result comes back
`lowConfidence: true` with `hasSync: false` and a flat two-chord stub instead of
an arrangement. That is the system reporting its own weakness rather than
lying — the behaviour we want — but sparse instrumental audio is a real weak
spot, and §8's numbers come from full-band recordings.

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
     `CHORDS_RATE_LIMIT_PER_MIN` (10).
   - `chords-worker-secrets` → `CHORDS_DATABASE_URL` **and no auth credentials at
     all**. The worker authenticates nobody.
   - `CHORDS_DEV_TOKEN` must appear in neither — `CHORDS_REQUIRE_AUTH` refuses to
     start if it is set.
6. **Deploy and gate it:**
   ```bash
   CHORDS_DATABASE_URL="$(the DSN)" modal deploy modal_app.py
   CHORDS_BASE_URL=https://…modal.run python scripts/smoke.py
   ```
7. **One real analysis**, with a Firebase ID token for a verified account:
   ```bash
   CHORDS_ID_TOKEN=… python scripts/smoke.py --video QDYfEBY9NM4
   ```
   Expect the bot check here rather than later — see below.
8. **A daily quota number.** `CHORDS_DAILY_QUOTA` defaults to 10, which is a
   placeholder, not a recommendation: ~47 s of worker CPU per analysis is what it
   is spending.

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
