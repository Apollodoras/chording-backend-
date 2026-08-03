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

**Built: milestone 1 and every layer that doesn't need a DSP engine.** 251 tests,
all green, no audio and no network required to run them.

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
| §16.5 contract fixtures | ✅ emitted; the app-side test is a small follow-up (below) |
| §5.1 fetch + decode | ⬜ **seam only** — §8 step 4 |
| §5.2/§5.3 engines | ⬜ **none chosen** — §8 step 2 is the owner's call (below) |

---

## Run it

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp .env.example .env
.venv/bin/python -m pytest              # 251 tests, ~2s, no network
.venv/bin/uvicorn app.main:app --reload
```

```bash
curl -s localhost:8000/healthz | python3 -m json.tool
```

`/healthz` reports what actually **built**, not what config asked for — auth
mode, store backend, kill-switch state, whether a fetch source exists, which
engines are installed. A green health check that hides a dead authenticator is
the failure the Mo backend learned from.

---

## The two things still owed

### 1. Choose the engines (§8 step 2)

Nothing is chosen, deliberately: the handoff says *"benchmark 2+ chord engines
and 2+ beat trackers … report results and let them choose before committing."*
So `app/analysis/engines.py` is an empty registry, and an unconfigured engine
answers a clean 503 rather than guessing.

```bash
.venv/bin/python bench/synth.py       # render ground-truth audio (~1 min)
.venv/bin/python bench/run_bench.py   # score whatever is registered
```

`bench/synth.py` renders six specimens with **exact** ground truth — a folk
progression, a ii–V–I in sevenths, a flat-key minor, a 3/4 waltz, two-chords-per-
bar, and a noisy variant. It is honest about its limits: it proves the plumbing
(that a tracker's output lands where the quantizer expects, that labels survive
normalization, that the emitted song lints clean), and it **cannot** tell you how
BTC and Chordino behave on a dense real mix — which is where they diverge and
what the choice actually hinges on. Drop real tracks plus a `<name>.truth.json`
into `bench/audio/` when you have them.

Adding a candidate is three steps, none of them upstream:

1. write the adapter (`analyze(pcm, sr) -> list[RawChordSpan]`; emit Harte or
   symbolic labels, post-processing normalizes them),
2. `register_chord_engine("btc", …)` in `app/analysis/engines.py`,
3. add the dependency to the `audio` extra **and** to `modal_app.py`'s *worker*
   image — never the API image (§4).

Candidates and the trade-offs to settle, from §5.2/§5.3/§18: **BTC** (strongest
on pop/rock, GPU — available per-function on Modal, but a GPU cold start per job
may cost more latency than Chordino's accuracy gap costs quality, and this
pipeline is already async) vs **Chordino/NNLS-Chroma** (predictable, no GPU);
**madmom** (the accuracy benchmark, unmaintained, needs NumPy < 2) vs
**beat_this** / **BeatNet** vs **librosa** (fast, weak on downbeats). Downbeat
accuracy matters more than beat accuracy here — §13.2's anchors *are* downbeats.

### 2. Wire the fetch stage (§8 step 4)

`app/analysis/fetch.py` defines the seam and `build_source` returns `None` until
an implementation exists. `probe` (metadata, no media) is separate from `decode`
on purpose: the blocklist and the 10-minute cap must both be decidable **before**
a byte is fetched, or the service downloads recordings it was never allowed to
touch.

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
| `app/main.py` | the HTTP surface, shaped like Mo's |
| `modal_app.py` | two functions, **two images** — that split *is* §4's isolation |

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
the clock, so a diff means the analysis changed.

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
