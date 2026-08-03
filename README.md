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

**Working end to end.** A YouTube id goes in; a linted `CompositionPayload` v2
and a `videoSync` sidecar come out. 286 tests, green, no audio and no network
required to run them.

| | |
|---|---|
| §3 operational surface | ✅ kill switch, per-video + per-channel blocklist, admin block/purge/offset, append-only audit log, verified purge cascade |
| §12 `CompositionPayload` v2 | ✅ emitter + the app's importer lint, ported from Mo |
| §12.2 chord normalization | ✅ Harte + symbolic → the app's closed grammar |
| §5.4 post-processing | ✅ quantize → merge → drop → hold N/C → simplify → confidence gate |
| §5.5 difficulty tiers | ✅ re-scoped against the grammar ceiling (§12.2) |
| §15 sections | ✅ repetition-based, whole bars, honest `Part N` fallback |
| §14 strumming patterns | ✅ fold/histogram/convention + quarter-note fallback |
| §13 `videoSync` sidecar | ✅ beat anchors + the §13.2 invariant, enforced by lint |
| §16 API | ✅ Mo-shaped: Firebase bearer, `{message, code}` errors, job-id + poll |
| §16.5 contract fixtures | ✅ emitted and byte-stable; the app-side test is a small follow-up (below) |
| §5.1 fetch + decode | ✅ yt-dlp + ffmpeg, bounded, behind the §4 seam |
| §5.2/§5.3 engines | ✅ **BTC + Beat This!**, benchmarked against real recordings (below) |
| CI | ✅ suite, Postgres, fixture stability, and a test that the API image cannot touch audio |

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
.venv/bin/python -m pytest              # 286 tests, ~30s, no network, no audio stack
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
POST /v1/analyze ──► cache hit? ──► 200 {song, videoSync}      (free — §16.4)
                 └─► 202 {jobId} ──► worker (its own container, own image)
                                       │
                       probe ──► gate (blocklist · 10-min cap · kill switch)
                                       │   nothing fetched until this passes
                       scratch dir ──► decode ──► beats ──► chords ──► onsets
                                       └─► rm -rf audio  (every exit path)
                                       │
                       §5.4 post-process ──► §15 sections ──► §14 patterns
                                       │
                       §12 compile ──► lint ──► Postgres: chord_maps
GET /v1/analyze/{jobId} ──► status / {song, videoSync}
```

| Module | What it owns |
|---|---|
| `app/payload.py` | `CompositionPayload` v2 — a near-copy of Mo's, `yt:` ids |
| `app/chords.py` | the app's grammar (ported) + §12.2 normalization + §5.5 tiers |
| `app/lint.py` | the importer's checks (ported) + `lint_sync`, the §13.2 invariant |
| `app/sync.py` | the sidecar, anchors, and the client's interpolation in Python |
| `app/store.py` | maps, jobs, blocklist, audit log, quota, limiter — two backends |
| `app/analysis/` | the pipeline; `scratch.py` is §2.1 in code |
| `app/analysis/ytdlp_source.py` | the only code that ever holds audio — worker image only |
| `app/analysis/adapters/` | one file per engine; nothing else imports a model |
| `app/main.py` | the HTTP surface, shaped like Mo's |
| `modal_app.py` | two functions, **two images** — that split *is* §4's isolation |
| `bench/fetch_corpus.py` | annotations + recordings → a scoreable corpus |

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

## Deploy

```bash
CHORDS_DATABASE_URL="$(the DSN in chords-secrets)" modal deploy modal_app.py
```

The worker image builds the engines in: CPU torch (explicitly — the default wheel
carries the whole CUDA runtime for hardware this deployment doesn't have), the
BTC checkout and `beat_this`, both **pinned to a commit** rather than a branch so
two deploys of identical code can't install different models. Beat This!'s
checkpoint is downloaded at *build* time; left to run time it would be fetched on
the first request of every cold container, turning a cold start into a dependency
on someone else's file server inside a job that already has a timeout.

Two Modal Secrets, and the split is §19.2 applied one level further in:
`chords-secrets` (API — Firebase, admin token, DSN) and
`chords-worker-secrets` (worker — **no auth credentials at all**, since it never
authenticates anyone). Passing `CHORDS_DATABASE_URL` at deploy time is what lifts
the single-container pin; forgetting it leaves a SQLite deployment correctly
pinned rather than silently losing writes.

Keep this deployment's credentials separate from Mo's (§19.2). Same Firebase
project for identity, different service-account key: Mo never touches a
recording, and the blast-radius argument says keep it that way.

---

## What is still owed, and by whom

Nothing in the backend's own scope is open. What remains needs an account, a key,
or a lawyer:

| | |
|---|---|
| Modal account, Postgres DSN, Firebase service-account key (**separate from Mo's** — §19.2), admin token → the two Modal Secrets | owner |
| First deploy, then `/healthz` against it | owner (needs the above) |
| Register a DMCA agent (§18) — blocks public exposure, not development | owner |
| Media IP lawyer review (§10) | owner |
| §19.1: whether Phase 2 reverses the "no backing track" canon | owner |
| `ChordsBackendContractTests` in the app repo (§16.5, below) | app repo |

Two things a deploy will surface that a laptop cannot. **YouTube answers
datacentre IPs with a bot check far more often than residential ones** — the
fetch stage understands `CHORDS_YTDLP_COOKIES` for exactly this, and it is the
most likely first failure in production. And the worker's 300 s Modal timeout
against ~47 s of DSP leaves room for a 10-minute video but not for much
retrying; if fetches get slow, that budget is the thing to watch.

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
