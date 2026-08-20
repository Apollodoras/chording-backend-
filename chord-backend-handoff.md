# Handoff: Chord Analysis Backend

**To:** future me, working in a code interface
**From:** me, in chat, after a conversation about the legal shape of this feature
**Status:** greenfield backend. Client already exists. Wiring is the last step.

---

> ## ⚠️ AMENDMENT — 2026-08-03: the client contract (read §11 first)
>
> This doc was written from chat, before anyone read the client. **Sections 1, 6,
> 7 and 9 contain factual errors about the app** — they describe *Strum*, the
> scored falling-bands rhythm game, which is **deleted**. The live app is
> **Rosetta GP**: jam-first, **unscored**, and its play surface is **campfire**.
> Every "confirm this" and "ask the user" in the original text has now been
> answered from the source. Sections **11–19** below are the amendment and they
> **supersede** the originals where they conflict:
>
> | § | What it settles |
> |---|---|
> | **11** | What the client actually is (corrects §1, §7 "Scoring") |
> | **12** | **The deliverable is `CompositionPayload` v2** — supersedes §6's proposed schema |
> | **13** | The video-sync sidecar (the part §6 was reaching for) |
> | **14** | Strumming-pattern extraction — what the app can accept |
> | **15** | Sections / song structure |
> | **16** | API shape, auth, errors — mirror the existing Mo backend (supersedes §7) |
> | **17** | What the **frontend** must build, and the phasing that de-risks it |
> | **18** | Answers to every question in §9 |
> | **19** | Two canon collisions the owner must rule on before you ship |
>
> The invariants in §2 and the operational surface in §3 are **unchanged and still
> binding**. So is §10.
>
> **Repo:** its own, as instructed — a third sibling beside
> `~/Projects/MIDI_Tab_Game` (the iOS app) and `~/Projects/Mo-Rosetta-GP` (the
> existing Mo backend). Suggested: `~/Projects/Chords-Rosetta-GP`.

---

## 1. What we're building

~~The user has a shipped mobile rhythm game. Chords fall as bands down the screen; the player swipes each band on the beat and gets scored on timing.~~ **Corrected — see §11.** The app is **Rosetta GP**, an *instrument*, not a rhythm game: chords are **horizontal bands** the player strums across six string lanes, the play surface is **campfire** (self-paced follow-along), and **there is no scoring anywhere in the product**. Today's charts come from **Mo Rosetta**, an existing AI-agent backend that writes song recipes (and from shared songs) — not from backing tracks.

The new feature: the player picks a YouTube video, the app generates a **playable song** from it, and that song lands **in the Library** alongside Mo's songs. Playing it *against* the embedded video is the richer second step (§13/§17), not the first.

Your job is the backend that turns a video ID into a song the app can import. You are not building the play surface or the renderer — those exist. **You are also not building a scoring system; there isn't one.**

### Why a backend at all

We settled this in conversation. The reasoning, so you don't relitigate it:

- The YouTube embedded player is sandboxed on every platform. **There is no audio buffer to tap.** On-device analysis of the stream is not merely awkward, it's unavailable.
- The play surface needs **lookahead** — campfire pre-glows the *next* chord's band so the hand can move there, and the bar lane draws strokes ahead of the cursor. Causal real-time detection fundamentally cannot provide this. (The original wording said "a falling-bands game"; the app is not one — §11 — but the requirement is unchanged, and stronger for Phase 1, where the **whole song must exist as a Library row** before a note is played.)
- Therefore: pre-computed analysis, server side, cached by video ID.

Do not propose a mic-based on-device path as the primary architecture. It fails with headphones, fails in noisy rooms, and still can't do lookahead.

---

## 2. Non-negotiable invariants

These are architectural, not preferences. Every design decision below exists to preserve them. If a refactor breaks one of these, the refactor is wrong.

1. **Audio is never persisted.** Fetch → decode → analyze → destroy, inside one worker process. No disk writes outside tmpfs, no object storage, no cache of decoded PCM, no "temporary" debug dumps that survive the request. Enforce it in code, not in a comment.
2. **Only the derived map is stored.** Chord symbols, timestamps, beat grid, key, tempo, confidences. Nothing from which audio could be reconstructed. No spectrograms, no chroma matrices in durable storage.
3. **The chord map is not a product.** It is consumed by gameplay only. No public API, no export endpoint, no PDF/chord-sheet generation, no share-the-chart feature. The moment the transcription itself becomes the deliverable, we're standing where the tab sites stood when the publishers came after them.
4. **Chord symbols only.** No lyrics, no melody line, no tablature, no standard notation. Ever. This is the single biggest thing keeping the feature defensible.
5. **The YouTube player stays visible and unobstructed.** Client-side concern, but it constrains the API: never design anything that assumes we own the full screen or control playback chrome. This is why Chordify's layout looks the way it does.
6. **Never paywall the playback.** Gate scoring, progression, difficulty modes, history — never access to the video itself.

### Things that will actually bite us

Realistic failure modes here are a Google API ban or a takedown, not a lawsuit. Build for those:

- Fetching and decoding audio separated from video contravenes the YouTube API terms as written. The user is knowingly accepting that risk. Do not paper over it; do keep the blast radius small.
- Above modest quota, Google runs a compliance audit. Assume this pipeline does not survive close inspection. Keep the kill switch (below) genuinely one-flag.

---

## 3. Required operational surface

Build these in the first milestone, not later. They're cheap now and impossible to retrofit under pressure.

- **Kill switch.** One config flag disables new analysis jobs and returns a clean "feature unavailable" to clients. Must not require a deploy.
- **Blocklist.** Per-video-ID and per-channel-ID. A takedown request must be satisfiable in minutes: block the ID, purge its cached map, done.
- **Takedown intake.** A DMCA agent must be registered (user's task, remind them). Backend side: an admin endpoint that blocks + purges, and an append-only audit log of who blocked what and when.
- **Purge job.** Given a video ID, delete the map and all references. Verify it actually cascades.

---

## 4. Architecture

```
client ──POST /analyze {videoId}──► API (FastAPI)
                                     │
                                     ├─ cache hit? ──► return map immediately
                                     │
                                     └─ enqueue job ──► Redis queue
                                                          │
                                                    analysis worker (isolated container)
                                                          │  fetch → decode (tmpfs)
                                                          │  beats → chroma → chords
                                                          │  post-process → simplify
                                                          │  **rm -rf audio**
                                                          ▼
                                                    Postgres: chord_maps
client ──GET /analyze/{jobId}──► status / result
```

**Worker isolation matters.** Separate container, separate image, no credentials beyond what it needs, tmpfs-only writable layer, memory-capped, hard timeout. If the audio-handling code is compromised or buggy, it should be unable to write anywhere durable. Set `--read-only` with an explicit tmpfs mount for scratch.

### Proposed stack

Assumptions — flag to the user if they conflict with existing infra:

- Python 3.11, FastAPI, Uvicorn
- Postgres (maps, blocklist, audit log)
- Redis + RQ (Celery is fine if they already run it; RQ is less machinery for this)
- Docker Compose for local, whatever they use for deploy
- ffmpeg for decode

---

## 5. Analysis pipeline

Order matters: beats first, then chords, then quantize chords to the beat grid. A rhythm game needs chord changes that land *on* beats — raw frame-level chord output looks jittery and plays badly.

### 5.1 Fetch + decode

`yt-dlp` → ffmpeg → mono 22.05 kHz float32 into a tmpfs path. Delete in a `finally` block, and have the worker verify the scratch dir is empty before exiting. Hard timeout on the whole job (suggest 180s); kill and clean on breach.

### 5.2 Beat and downbeat tracking

Needed for grid quantization, for the game's lane timing, and for tempo display.

- `madmom` (`RNNDownBeatProcessor` + `DBNDownBeatTrackingProcessor`) is still the accuracy benchmark, **but it is effectively unmaintained** — last release is old and it breaks on NumPy 2.x. If you use it, pin NumPy < 2 in the worker image and isolate it there.
- Modern alternatives worth benchmarking first: `beat_this`, `BeatNet`. `librosa.beat.beat_track` is the fallback — fast, weaker on downbeats.

Output: beat times, downbeat times, tempo, time signature guess.

### 5.3 Chord recognition

Benchmark at least two before committing. Options, roughly in order of expected quality:

- **BTC** (Bi-directional Transformer for Chord Recognition) — pretrained weights are available; strongest on typical pop/rock.
- **Chordino / NNLS-Chroma** (Vamp plugin, via the `chord-extractor` wrapper or `sonic-annotator` CLI) — classical baseline, very predictable, no GPU.
- **autochord** — pure Python, easy, weaker.

Build an adapter interface so the engine is swappable:

```python
class ChordEngine(Protocol):
    name: str
    version: str
    def analyze(self, pcm: np.ndarray, sr: int) -> list[RawChordSpan]: ...
```

Persist `engine_name` + `engine_version` on every stored map so caches can be invalidated selectively when we upgrade.

### 5.4 Post-processing (this is where playability is won)

Raw output is not a game chart. Apply, in order:

1. **Quantize** chord boundaries to the nearest beat.
2. **Merge** runs of the same chord.
3. **Drop** spans shorter than a threshold (suggest: shorter than one beat, or < 250ms).
4. **Fill** N/C (no-chord) gaps — either hold the previous chord or emit an explicit rest lane; **ask the user which the client expects.**
5. **Simplify by difficulty** — see below.
6. **Confidence gate** — if mean confidence across the track is below threshold, return the map flagged `low_confidence` so the client can warn the player. Some material (dense mixes, heavy distortion, solo-heavy tracks) simply doesn't analyze; say so rather than shipping garbage.

### 5.5 Difficulty simplification

Backend-side, and a genuinely good feature. Same analysis, three renderings:

- `easy` — collapse to triads, major/minor only; drop passing chords shorter than a bar
- `normal` — triads + 7ths
- `hard` — full detected quality, including extensions and slash chords

Compute once, store all three, let the client pick. Don't re-analyze per difficulty.

---

## 6. Data contract

⚠️ ~~**This schema is proposed, not agreed.**~~ **SUPERSEDED BY §12 + §13 — do not
implement the schema below.** The instinct was right ("adapting the backend to the
client is far cheaper than touching a shipped game") and the client's format has now
been read: it is **`CompositionPayload` v2**, an already-shipped, already-tested,
self-contained song container with its own importer, lint and cross-language contract
test. Emit that. The block below survives only because §13's sidecar borrows its good
ideas — integer ms, `beatIndex`, the pre-rendered `label`, `offsetMs`.

```jsonc
{
  "videoId": "abc123",
  "schemaVersion": 1,
  "engine": { "name": "btc", "version": "1.2.0" },
  "analyzedAt": "2026-08-03T10:00:00Z",
  "durationMs": 214000,
  "tempo": { "bpm": 128.02, "confidence": 0.91 },
  "timeSignature": "4/4",
  "key": { "tonic": "G", "mode": "major", "confidence": 0.78 },
  "lowConfidence": false,
  "beats": [
    { "tMs": 512, "index": 0, "downbeat": true }
  ],
  "charts": {
    "normal": [
      {
        "tMs": 512,
        "durationMs": 1875,
        "beatIndex": 0,
        "root": "G",
        "quality": "maj",
        "bass": null,
        "label": "G",
        "confidence": 0.88
      }
    ]
  }
}
```

Notes:

- **All times are milliseconds from video start**, integer. No floats for timing — rounding drift across a 4-minute track is real.
- `beatIndex` lets the client snap lanes to the grid without recomputing.
- Keep `root` / `quality` / `bass` structured *and* `label` pre-rendered. The client shouldn't be building display strings, and we shouldn't be guessing its notation preferences.

### Sync offset — do not skip this

The video's audio may not start at t=0, and the player's own **output latency**
(especially Bluetooth — plausibly hundreds of ms) will drift the chart against what
they hear. ~~The game is already scoring swipes against a clock, so the client likely
has calibration; **ask**.~~ **Answered: there is no scoring and no calibration.** The
app *had* a tap-to-beat input-latency calibration; it was **deleted** (it only ever
corrected a scoring delta, and scoring is gone). So nothing on the client corrects for
latency today.

The instruction below stands and matters *more* than it did: **carry a nullable
`offsetMs`**, expose the admin endpoint to set it per video, and expect the client to
add a manual nudge control (§17) — a player-facing "the chart is early/late" slider is
now the only correction path that exists.

---

## 7. API

```
POST   /v1/analyze                 { videoId, difficulty? } → { jobId, status, map? }
GET    /v1/analyze/{jobId}         → { status, progress, map?, error? }
GET    /v1/maps/{videoId}          → cached map or 404
POST   /v1/admin/block             { videoId | channelId, reason } → purges + blocks
DELETE /v1/admin/maps/{videoId}    → purge
GET    /healthz
```

- `POST /v1/analyze` returns the map inline on cache hit (the common case once warm). Otherwise `202` + job ID, client polls.
- Statuses: `queued`, `fetching`, `analyzing`, `ready`, `failed`, `blocked`, `unavailable`.
- Real error codes the client must handle: video blocked, video unavailable/private/region-locked, too long (cap at ~10 min), analysis failed, low confidence, feature disabled by kill switch.
- Per-user rate limit on new analyses. Cache hits shouldn't count against it.

### Scoring

~~Keep swipe scoring **on the client**…~~ **Moot — the product has no scoring at
all** (§11). No deltas, no judgement tiers, no stars, no XP, no leaderboards; the
whole scored layer was deleted along with the campaign. The conclusion still holds in
the only form that matters: **never design a round-trip that needs the player's
timing.** Don't build score submission, and don't ask for it — "would you like
leaderboards later" is a product question the owner has already answered no to.

See §16 for the API shape that replaces this section.

---

## 8. Build order

1. **Skeleton + invariants.** FastAPI, Postgres, Redis, worker container with read-only FS + tmpfs. Blocklist, purge, kill switch, audit log. Health check. Nothing musical yet.
2. **Pipeline spike.** Offline script, local audio files only. Benchmark 2+ chord engines and 2+ beat trackers on a handful of tracks the user picks. **Report results and let them choose** before committing.
3. **Post-processing + difficulty tiers.** Tune against the same test set. This is where the feel lives — expect iteration.
4. **Wire the fetch stage in.** Add decode + guaranteed cleanup. Write the test asserting scratch is empty after every job, including failure paths.
5. **API + caching + job lifecycle.**
6. **Reconcile schema with the client, then wire.** Last, as the user specified.

---

## 9. Ask the user before building

> **All seven are ANSWERED in §18** — from the client source, not from a
> conversation. Read that instead of asking again. Only the DMCA-agent
> registration and the §19 canon calls are genuinely still owner-owned.

Don't guess on these:

- The client's existing chart format — field names, timing units, how it represents chord quality and rests.
- Does the client already do latency calibration? What does it expect from us regarding offset?
- N/C handling: hold previous chord, or explicit rest?
- Existing infra: what language/hosting/DB do they already run? The stack above is a default, not a requirement.
- Deploy target and whether GPU is available (decides BTC vs. Chordino).
- Max video length cap.
- Is the DMCA agent registered yet?

## 10. Standing note

The user was told, correctly, to have a media IP lawyer review the backend architecture before launch. If the build starts drifting toward storing audio, exposing chord data, or adding lyrics/tab, stop and flag it rather than quietly implementing it. Those aren't features with a legal footnote — they're the difference between this feature and the one that got sued.

---
---

# AMENDMENT (2026-08-03) — the client contract

*Everything below was read out of the app's source, not recalled from
conversation. Where it contradicts §§1–9, it wins.*

---

## 11. What the client actually is

The app is **Rosetta GP** (`rosetta` branch of `~/Projects/MIDI_Tab_Game`; bundle
id `com.themuseicon.RosettaGP`). The doc's opening description belongs to **Strum**,
its predecessor, frozen on `main` — the falling-bands highway, the timing windows and
the whole scoring model were **deleted**, not hidden.

**Read these two files before writing code** (they are the product canon, in order):
`ROSETTA_GP.md` §§1–4, then `MO_BACKEND_HANDOFF.md` §3 (the wire format, in full).
`CLAUDE.md` is Strum-era and describes many deleted systems — trust it only where
ROSETTA_GP.md §4 says it still applies.

### The one-line pitch

The player **plays a guitar on their phone**. Six vertical string lanes; swiping
across them rakes the strings and **the player's swipes make every sound** (a bundled
steel-string SoundFont). There is no backing track anywhere in the product today —
which is exactly why this feature needs a decision (§19).

### The play surface: campfire

There is exactly one way to play a song, and it is **not** a rhythm game:

- The song is a **`JamSongSheet`** — a flat list of **strokes**, each carrying its
  chord, direction (down/up), bar index, beat-offset-within-bar, accent, and section
  label. (`MIDI_Tab_Game/Models/JamSongSheet.swift`.)
- The song's **distinct chords become horizontal bands**. The current stroke's chord
  **glows**; the next chord pre-glows. Strumming the glowing band **advances the
  cursor by one stroke** — so a chord that spans four strokes stays lit while the
  player strums through it.
- **Self-paced.** No clock, no due times, no windows, no misses. The song moves when
  the hand moves, and loops at the end.
- **Unscored.** No `GameLogic`, no stars, no XP, no results screen. Nothing to report
  to a server, ever.

### There is already a clock-driven cursor — and it is your integration point

Campfire has an **opt-in** "play with the pulse" mode where a clock, not the hand,
walks the cursor (`JamViewModel.advanceCursorOnPulse`). The whole mechanism is three
lines:

```swift
songBeat += delta * pulse.beatsPerSecond              // elapsed song beats
guard let i = sheet.strokeIndex(atBeat: songBeat) else { … }   // pure, looping
sheet.setPosition(i)
```

**A video-synced mode is the same code with a different clock**: replace the pulse
accumulator with `songBeat = beatMap(videoCurrentTimeMs)`. That is why §13 asks for a
beat-anchor map rather than a re-derived chart — it makes the client change small and
keeps the cursor from drifting against the recording.

### The Library

SwiftData `Composition` rows. **One import door** for every song the app didn't
author — shared links, Mo's output, and (soon) yours:

```
CompositionPayload  →  ComposerService.import(payload)  →  Library row
                                                        →  campfireSheet(for:)  →  play
```

`import` returns an **`ImportReport`** with per-section warnings, which the app shows
the player ("Verse: strumming pattern not included"). Your lint's job is to make that
list empty (§12.4).

### There is already a backend, and you should look like it

**Mo Rosetta** (`~/Projects/Mo-Rosetta-GP`) is a live FastAPI service on Modal that
writes songs for this same app, through this same import door. It has already solved
auth, quota, error shape, deployment, and cross-language contract testing. **Copy its
conventions** (§16) rather than inventing parallel ones — the client's HTTP layer,
error enum and account plumbing already exist and can be reused nearly verbatim.

The two backends stay **separate repos** (the user's instruction, and good hygiene
here: Mo never touches audio, and §2's blast-radius argument says keep it that way).

---

## 12. The deliverable: `CompositionPayload` v2

**This supersedes §6.** Do not design a chord-map schema. Emit the container the app
already imports, and the song plays with **zero client changes**.

Authoritative spec: **`MO_BACKEND_HANDOFF.md` §3** (`~/Projects/MIDI_Tab_Game/`) —
read it in full; it is exact, it is what Mo's serializer targets, and it documents
the traps. Reference implementations: `Services/ComposerService.swift`
(`CompositionPayload`, `import`, `importReport`), `Models/Arrangement.swift`,
`Models/BlockPayloads.swift`.

### 12.1 Shape

```jsonc
{
  "version": 2,
  "id": "yt:dQw4w9WgXcQ",          // stable per video — see §12.5
  "title": "…",
  "tonic": "G",                     // A–G + optional #/b
  "mode": "major",                  // "major" | "minor" ONLY
  "tempo": 128,                     // integer BPM, quarter-note pulse
  "timeSignature": "4/4",

  // flat summary — REQUIRED even with an arrangement; mirror section 1
  "chordNames": ["G","D","Em","C"],
  "patternID": "yt:pat-…",
  "beatsPerChord": 4,
  "repeats": 1,

  "arrangement": { "sections": [ … ] },   // the real song
  "patterns":    [ … PatternPayload … ],  // MUST embed every pattern referenced
  "progressions": []                       // optional, [] is fine
}
```

A **section** is `{ id, name, kindRaw, chordNames, patternID, beatsPerChord,
repeats, tempoOverride, timeSignatureOverride, bars }`. `kindRaw` ∈ `intro | verse |
preChorus | chorus | bridge | solo | outro | custom`. For per-bar harmony (chords
changing mid-bar — you will need this constantly on real recordings) use
`bars[].chordSpans[]` with bar-local `startBeat` / `lengthBeats`.

### 12.2 Closed vocabularies — exceed them and the chord silently misfires

**Chord names** are parsed strictly by `ChordSymbol(name:)`. Root `A–G` + optional
`#`/`b`/`♯`/`♭` + exactly one of: *(none)*/`maj`/`major` · `m`/`min`/`mi`/`-` ·
`7`/`dom7` · `maj7`/`ma7`/`major7` · `m7`/`min7`/`mi7`/`-7` · `dim`/`°`/`o` ·
`dim7`/`°7`/`o7` · `m7b5`/`ø`/`halfdim` · `aug`/`+` · `sus`/`sus4`/`sus2`.

**Not supported — normalize before emitting** (a chord recognizer will hand you these
constantly): slash chords (`G/B` → `G`), extensions (`9`/`11`/`13`/`add9`/`6`/`m6` →
nearest 7th or triad), power chords (`5` → major), altered dominants (`7#5` → `7`).
**This is where §5.5's `easy`/`normal`/`hard` tiers pay off** — but note `hard` cannot
mean "full detected quality" here: the ceiling is this grammar. Redefine the tiers
against it.

**Mode** is `major`/`minor` only — the song container knows no church modes (the jam
room does; songs don't).

### 12.3 Strokes and patterns

```jsonc
{ "id":"uuid", "beat":0.5, "direction":"down", "accent":false, "msOffset":0, "strings":null }
```

- **`beat` is 0-INDEXED** (0 = downbeat, 0.5 = its "&") and must fit inside **one
  bar**. The app's *bundled* pattern files are 1-indexed — irrelevant to you, but it
  is the trap that bites anyone reading them for reference.
- `direction` ∈ `down` | `up` | `bass` | `mute`. `strings: null` = whole chord.
- **Every `patternID` a section references must be embedded** in top-level
  `patterns[]`, or **the section is silently dropped and the song plays short with no
  error**. There is no bundled pattern catalog on the client any more — it was
  deleted. Prefix ids `yt:pat-…` (Mo uses `mo:pat-…`).

### 12.4 Lint before returning — this is not optional

Port the app's own importer check (`ComposerService.importReport`, ~40 lines) and run
it server-side. **Never return a payload that would warn.** Mo does this as a
lint-in-the-loop tool inside its generation harness, so there is no code path that can
emit an unvalidated song; do the same.

1. Every `chordName` in every section and every bar `chordSpan` parses under §12.2.
2. Every referenced `patternID` (section-level *and* per-bar `.pattern(id)`
   overrides) resolves against the embedded definitions.
3. At least one playable section; no section with empty `chordNames` or empty
   `patternID`.
4. Flat summary fields mirror section 1. Tempo in 40–220. Stroke beats inside one bar.

### 12.5 Ids

`id` is the **idempotency key** — `import` upserts on it, so re-analyzing a video must
**replace** its Library row, not duplicate it. Use a deterministic `yt:<videoId>` (add
a suffix if you ever let one video produce several difficulty rows: `yt:<videoId>:easy`).
Keep embedded pattern ids stable when their strokes are unchanged.

---

## 13. Video sync — the sidecar

A `CompositionPayload` is a **relative** grid: tempo + meter + bars. A real recording
is **absolute** and drifts. So the payload alone can seed the Library and play
self-paced (§17 Phase 1), but it cannot follow a video. That is what §6's schema was
actually reaching for, and it survives as a **sidecar**, not as the song.

### 13.1 Response envelope

Return both, side by side — the payload stays byte-identical to what the importer
expects, and stays shareable as-is:

```jsonc
{
  "song": { …CompositionPayload v2… },
  "videoSync": {
    "source": "youtube",
    "videoId": "dQw4w9WgXcQ",
    "durationMs": 214000,
    "offsetMs": 0,                 // nullable; admin-settable per video (§6)
    "tempo": { "bpm": 128.02, "confidence": 0.91 },
    "timeSignature": "4/4",
    "lowConfidence": false,
    "beatAnchors": [               // THE load-bearing field — see 13.2
      { "songBeat": 0,   "tMs": 512 },
      { "songBeat": 32,  "tMs": 15512 },
      { "songBeat": 64,  "tMs": 30498 }
    ],
    "engine": { "chords": "btc@1.2.0", "beats": "beat_this@0.3.1" },
    "analyzedAt": "2026-08-03T10:00:00Z"
  }
}
```

All times **integer milliseconds** from video start (the original doc's instinct —
float drift over four minutes is real). Keep `beats[]` out of the client response if
you store it; the client needs only the anchors.

### 13.2 `beatAnchors` — why this shape

The client's cursor is addressed in **song beats** (`JamSongSheet.strokeIndex(atBeat:)`,
`dueBeat(of:)`). The video clock is in **milliseconds**. The anchor list is the map
between them, and it absorbs tempo drift that a single BPM cannot:

```
songBeat = piecewiseLinearInterpolate(beatAnchors, videoCurrentTimeMs - offsetMs)
```

Emit an anchor at least **every bar** (every downbeat is ideal — cheap, and it makes
the interpolation exact rather than approximate). The invariant that makes this work:
**`songBeat` here must be the same beat axis the compiled chart produces** — i.e.
bar *n* of the payload starts at `songBeat = n × barBeats`. If your section layout and
your anchor list disagree, the cursor walks off the song. Assert it in a test.

### 13.3 Degrading honestly

If beat tracking is weak (`lowConfidence`), **return the song with `videoSync: null`**
rather than a fake grid. A self-paced campfire song that's right beats a video-synced
one that's wrong — the player has no scoring to protect, so the only cost of bad sync
is that it feels broken.

---

## 14. Strumming patterns — yes, extract them, within limits

The user's ask ("probably the strumming patterns too") is right, and the app *requires*
one anyway: a section with no resolvable pattern is dropped (§12.3). So you always
emit at least one.

**What the app can represent:** one bar of strokes at arbitrary beat offsets, each
`down`/`up`/`bass`/`mute`, with an accent flag. That's it. No dynamics curve, no
per-string picking (the field exists; don't use it), no swing parameter.

**What is honestly recoverable from audio, and what isn't:**

| Dimension | Recoverable? | How |
|---|---|---|
| **Onset positions** | Yes | onset detection (`librosa.onset`, or madmom's) folded onto the beat grid |
| **Subdivision** (8ths vs 16ths) | Yes | quantize onsets modulo the bar; pick the coarsest grid they sit on |
| **Accent** | Roughly | onset strength relative to the bar's mean |
| **Band** (bass vs chordal) | **Yes** | split the onset envelope at 250 Hz — see §14.1 |
| **Direction (down/up)** | **No — infer by convention** | see below |
| **Mute / percussive** | Not reliably in a full mix | **don't emit `mute`** |

**Direction is a convention, not a measurement.** You cannot hear which way a hand
moved in a mixed recording. Use the alternating-hand rule every teacher writes: an
onset on a beat is a **down**stroke, an onset on the "&" (or the second/fourth 16th)
is an **up**stroke. State this in the pattern's `name`/`tags` so nobody later mistakes
it for detection. It is also *correct* far more often than not, because that is how
the instrument is physically played.

**Method that works:** per section, fold that section's onsets onto one bar of the
beat grid → score candidate subdivisions → keep positions above a support
threshold → assign directions by the rule above → emit one `PatternPayload` per
distinct section pattern. If support is too thin or too noisy, **fall back to a plain
quarter-note downstroke bar** — a boring pattern that plays is worth more than a
confident one that's wrong. Carry a `patternConfidence` in `videoSync` so the client
can say so.

Three details in that sentence are load-bearing, and each one shipped a defect
before it was got right (`app/analysis/strumming.py` carries the full reasoning):

- **The bar is a loop, not a line.** A hand is rarely late to the "one" and often
  early, so folding modulo the bar puts downbeat strokes at the *far end* of it.
  Match cells around the bar and roll an anticipating onset forward onto the
  downbeat it was reaching for, or the beat-1 stroke silently vanishes from
  patterns extracted off real recordings while confidence in the rest stays 1.0.
  For the same reason, run onset detection with **`backtrack=False`**: backtracking
  is a slicing feature and biases every onset early.
- **Score subdivisions by average error, not by how many onsets "fit".** Every 8th
  is also a 16th, so a fit count always leans finer, and one consistent hi-hat 16th
  per bar — i.e. every drummed recording — is enough to carry the grid to 16ths.
  Since direction is read off the grid, that flips every "&" in the bar from an
  upstroke to a downstroke.
- **Support is a share of bars, not of onsets.** Otherwise one bar with a flam
  stands in for two bars of evidence.

**Don't over-fit.** Campfire shows the pattern as direction triangles under the bar and
the player strums through it; a 16-onset syncopated transcription of a strummed
acoustic is less playable — and less *true to the song* — than the D-DU-UD-U everyone
actually plays. Per-bar variation exists in the model (`bars[].rhythm.custom`) but
should be the exception.

### 14.1 Bands — the accompaniment's two hands (2026-08-19)

**The ask.** The app grew a second instrument: a piano, played by tapping rather
than strumming (`ROSETTA_GP.md` §3.3). A piano accompaniment is not a strum — it
is a **left hand and a right hand doing different things on different beats**, and
the owner's direction is that the analysis should serve that rather than assume a
guitar.

**What actually changes, and it is smaller than it sounds.** Everything §14 above
extracts is already instrument-neutral: onset positions, subdivision, accent are
facts about the *song's* rhythm, not about a guitar. Exactly one field in the
emitted pattern is guitar-shaped — `direction` — and §14 already says it is a
convention rather than a measurement. So this is not a second extraction: it is
the same one, told to stop throwing away a dimension it was already measuring
over.

**The dimension is the band.** A bass note and a chord over it are separated by an
octave and a half, and a band split finds them where no amount of processing can
find which way a hand moved. Each detected attack is labelled:

| `band` | Meaning |
|---|---|
| `low` | the attack is in the bass band alone |
| `mid` | in the chordal band alone |
| *absent* | both — a strum, a block chord, or nothing was decided |

`Stroke.band` carries it. **Only `low` and `mid` go on the wire**; "both" travels
as an absent field, so a song that is simply strummed serializes to exactly the
bytes it did before this existed, and its content-addressed id (§12.5) is
unchanged. The lint rejects any other value, including a literal `"full"`.

**The reading belongs to the client, not here.** A `low` stroke is a bass note to
a guitar (boom-chick, the `bass` stroke kind the app has always had and the
analysis has never once emitted) and a left hand to a piano. The backend says
what was struck; the app says who struck it. That split is not tidiness — **the
catalog is shared and a song is analyzed once**, so one payload has to serve both
instruments or the catalog fragments by instrument and every song is analyzed
twice.

**Three things were measured rather than chosen** (`bench/synth.py` gained an
`oom-pah` specimen — a root octave on 1 and 3, a right hand on 2 and 4 — because
every previous specimen strums a chord and so could not ask this question at all):

- **The split is 250 Hz.** At 320 a third of the chord's energy arrives as bass;
  at 180 an ordinary strum starts reading as chord-only.
- **Presence is judged per band, against that band's own typical attack** — not
  as a ratio between the two. The ratio test separates a bass note from a chord
  stab beautifully *and* labels every ordinary strum `mid`, because a chord voiced
  from E3 up genuinely puts most of its energy above the split. A rule that calls
  a plain strum right-hand-only would hand a piano a song with no bass in it.
- **A bar's bands survive only if the bar actually splits** — something in the
  bass band alone, and something else not. `mid` means nothing without a `low` to
  mean it against: a song whose bass is merely quiet would otherwise report that
  it has no left hand. This is also what makes the whole feature a no-op on
  strummed material, including the contrast rule.

**One dependency fell out of it, and it is the interesting part.** §14's contrast
rule compares every cell against the loudest cell in the bar — which deletes a
bass note that is quieter than the chord over it. Measured on the oom-pah, that
rule emitted *the two chord stabs as the whole pattern* and dropped both bass
strokes. Contrast is now measured **within a band**. On unbanded material every
cell shares one band and the reference is the bar's peak exactly as before, so
the rule generalizes without moving.

**The client half landed 2026-08-19** (app `ROSETTA_GP.md` §3.4), so the field is no
longer inert: `StrokeBand` is parsed tolerantly (an unknown value reads as both hands),
each voicer marks which of *its* notes are the low band, and the compiler joins the two
by equality — a guitar plays a `low` stroke as its bass string, a piano as its
left-hand octave, and its audio engine releases per hand so a right-hand stab leaves
the bass ringing. `tests/fixtures/valid/banded-oom-pah.json` is a banded payload for
the app's own contract test, which asserts the chart **splits into hands** rather than
merely that the field decoded. Emitting it left every existing fixture byte-identical,
which is this section's back-compat claim proved rather than asserted.

**Not done, and blocked here rather than skipped:** carrying the *actual bass
pitch* (slash chords — `C/G`), which would be the other half of a real left hand.
`app/chords.py` discards the slash bass deliberately, and the 2026-08-18 audit
already considered keeping inversions (F34) and declined: **the app's chord
grammar has no field for a bass note** and voices every chord from its root, so a
bass emitted here is a chord the client cannot parse. That one needs a client
change first (`ChordSymbol` gains a bass), and the loss stays counted in
`exactRatio` rather than papered over.

---

## 15. Sections

Section labels are **player-visible** — campfire's header prints "Verse · bar 2 of 8"
and the rail marks section boundaries. So structure is worth getting roughly right,
and it is also the cheapest place to be wrong-but-harmless.

- Segment structurally (repetition/novelty — `msaf`, or `librosa.segment` +
  self-similarity), then map segments to sections.
- Naming: if you can identify the repeated high-energy segment, calling it `chorus`
  and the others `verse` is defensible. If you can't, use `kindRaw: "custom"` with
  `name: "Part 1"`, `"Part 2"`. **Never** name a section from lyrics — you must not
  have lyrics at all (§2.4).
- Practical floor: one section per structural segment, minimum ~4 bars. Merging
  near-identical adjacent segments reads better than a 14-section song.
  *(Amended by §20.3: the floor yields when honouring it would flatten a
  `repeats > 1` neighbour, and the tail-joins-previous rule moved to after
  clustering. Both destroyed repeat groups on ordinary input.)*
- `repeats` is your friend: a 4-bar progression played 4× is one section with
  `repeats: 4`, not 16 bars of explicit chords.

---

## 16. API — mirror the Mo backend (supersedes §7)

The client's HTTP layer already exists for Mo and will be **copied, not designed**
(`Services/RemoteMoRosettaService.swift`, `MoRosettaService.swift`,
`MoAccountService.swift`, `MoMeService.swift`). Match these and the client work is a
day; diverge and it's a week.

### 16.1 Endpoints

```
POST   /v1/analyze              { videoId | url, difficulty? }  → { song, videoSync } | { jobId, status }
GET    /v1/analyze/{jobId}      → { status, progress, song?, videoSync?, message? }
GET    /v1/maps/{videoId}       → cached result or 404
GET    /v1/me                   → identity + quota   (mirror Mo's shape exactly)
GET    /healthz                 → { "status": "ok", … }
POST   /v1/admin/block          { videoId | channelId, reason }  → purge + block
DELETE /v1/admin/maps/{videoId} → purge
```

Statuses: `queued`, `fetching`, `analyzing`, `ready`, `failed`, `blocked`,
`unavailable`. Cache hit returns inline (`200`); otherwise `202` + `jobId`.

**Job model:** unlike Mo (which is a long synchronous request the client awaits for up
to 120 s), analysis has a fetch+decode+DSP tail that deserves **job-id + poll** — the
original §7 was right. The client wraps it either way; put the polling inside one
service class, exactly as `RemoteMoRosettaService` isolates its transport.

### 16.2 Auth — reuse Firebase, do not invent

`Authorization: Bearer <Firebase ID token>`, verified with `firebase-admin` into a
`uid`. Same Firebase project as Mo, so **the player is already signed in** and the
client's token provider is reused as-is. Notes from Mo's build that will save you a
day:

- The ID token **expires hourly** — fetch per request, never cache above the seam.
- `MO_DEV_TOKEN`-style dev bypass is deliberately unset in production.
- The token is read from **off the main actor** on the client. (Mo crashed on exactly
  this; see ROSETTA_GP.md §3's isolation gotcha.)

### 16.3 Errors — the client already renders these

Return `{"message": "…"}` with a human-readable line. The client's error enum
(`MoRosettaError`) already maps and has **copy written** for: `401` → not signed in /
session unverified, `403` → email unverified, `429` → daily quota exhausted *or*
out-of-credits (distinguished by an `insufficient_credits` marker), other non-2xx →
server with your message, `URLError` → offline. Mirror those semantics and the new
client gets its whole failure UI for free.

Your own additional cases all ride the generic path with a good message, but the
client will want to distinguish a few: **video blocked**, **unavailable / private /
region-locked**, **too long**, **analysis failed**, **low confidence**, **feature
disabled** (§3's kill switch). Give each a stable machine-readable `code` alongside
`message`.

### 16.4 Metering

Analysis costs real money per call, like Mo does. The account/credit machinery
(plans, allowance, credits ledger, Stripe + StoreKit) already exists — see
`ROSETTA_ACCOUNTS.md` in the app repo. **Owner decision (§19):** does an analysis
spend from the *same* pool as a Mo song, or its own? Default until told otherwise:
**meter it, own daily quota, cache hits free** (§7's instinct — a cached map costs
nothing, so it must not cost the player anything either).

### 16.5 Contract testing — steal Mo's best idea

Mo emits its serializer's real output as JSON fixtures; the **iOS test suite walks
them through the actual importer** (`MoBackendContractTests` → `CompositionPayload
.from(jsonData:)` → `ComposerService.import` → `campfireSheet`) and asserts decode +
**zero warnings**. The app's own importer is the final judge of backend output, in CI,
with no API key needed. Do the same: write fixtures to `tests/fixtures/emitted/`, and
the app side adds a sibling test pointed at them via an env var
(`CHORDS_BACKEND_FIXTURES`, mirroring `MO_BACKEND_FIXTURES`).

---

## 17. What the frontend must build

Not "wiring" — this is real client work, and it's worth knowing the size before
committing to §8's build order. Listed cheapest-first, and **the phasing matters**:

### Phase 1 — Library only (no video playback). Small.
The player pastes/picks a video → backend returns a song → **it lands in the Library
and plays campfire, self-paced**, exactly like a Mo song. Client work:

1. `ChordAnalysisService` protocol + `RemoteChordAnalysisService` — a near-copy of
   `MoRosettaService` / `RemoteMoRosettaService`, plus poll.
2. An entry point + state machine + result card — clone `AskMoView` /
   `MoSessionViewModel` (brief → working → result → failure); the states, copy and
   quota handling already exist.
3. Nothing else. Import, Library row, campfire, sharing all work unchanged.

**This ships the user's stated ask** ("the app will use them for the library") with no
new play surface, no canon collision (§19), and no sync problem. Do it first.

### Phase 2 — Play against the video. Substantially bigger.
4. **Video surface** — an embedded YouTube player that stays visible and unobstructed
   (§2.5), composed with the six string lanes and the chord bands. This is a new
   screen layout, not a variant of the jam room.
5. **Video-clocked campfire** — swap the pulse accumulator for
   `songBeat = beatMap(videoCurrentTimeMs - offsetMs)` (§11, §13.2). The cursor
   machinery itself is already there and tested.
6. **Persistence** — `Composition` needs the sidecar. Follow the existing pattern:
   one `Codable` blob column (`arrangementData` / `bundledPatternsData` do exactly
   this), nullable, so every existing row migrates losslessly.
7. **Offset nudge** — a player-facing early/late slider, persisted per song. This is
   now the *only* latency correction in the app (§6, amended), and Bluetooth users
   will need it.
8. **Rights/UX guardrails** — never paywall playback (§2.6); handle blocked and
   unavailable videos as calm states, not errors.

### Both phases
9. The contract test of §16.5.

---

## 18. Answers to §9

| Question | Answer |
|---|---|
| **Client chart format** | **`CompositionPayload` v2** — §12. Not a chord map. Integer BPM + `n/d` meter, chords as strict `ChordSymbol` names (§12.2), rhythm as 0-indexed strokes in one bar, sections as an `Arrangement`. Spec: `MO_BACKEND_HANDOFF.md` §3. |
| **Latency calibration?** | **No — deleted.** It existed only to correct a scoring delta, and there is no scoring. Carry `offsetMs`; the client will add a manual nudge (§17.7). |
| **N/C handling** | **Hold the previous chord.** There is no rest primitive — a stroke always sounds a chord, and a section is chords × pattern. For a genuinely long instrumental gap (> ~2 bars), the honest options are: hold, or make it its own section with the previous chord and a sparse pattern. **Never emit a section with empty chords — it's silently dropped.** |
| **Existing infra** | Python 3.11 + FastAPI + `anthropic`, deployed on **Modal**, **Firebase Auth**, **Postgres** (with SQLite in dev), Stripe + StoreKit for billing. The stack in §4 is a good match — but check Modal against your worker-isolation needs (§2/§4) before assuming Redis+RQ is the right queue there; Modal has its own function/queue primitives and its own container isolation story. |
| **GPU?** | Available on Modal, per-function. So **BTC is viable** — but benchmark it against Chordino honestly (§5.3): a GPU cold start per job may cost more latency than Chordino's accuracy gap costs quality, and this pipeline is already async. |
| **Max video length** | **Cap at 10 minutes** (§7's own suggestion, adopted). Songs, not DJ sets. Reject longer with a clear message. |
| **DMCA agent registered?** | **Unknown — still the owner's task** (§3). Ask before any public exposure. |

---

## 19. Two calls the owner must make before this ships

Neither is a technical question; both change what gets built.

### 19.1 A video playing under the strums is a backing track

The product's founding audio law — `CLAUDE.md` §6.1, kept as live canon by
`ROSETTA_GP.md` §4 — is **"the player IS the sound source; there is no backing
track."** §13 of the same doc lists "backing track / playback audio" as an
**explicitly rejected decision, never to re-propose**. Even the metronome is
constrained by this: the pulse's flagship state is *silent* precisely so the player
stays the only sound, and the UI says so out loud ("The pulse is a reference, not a
backing track").

**Phase 2 of this feature plays a record while the player strums over it.** That is
the rejected thing, by the doc's own words. It may well be the right call — "play
along with the record" is how humans actually learn songs, and the player is still
making all the *guitar* sound — but it is a canon reversal and it is the owner's to
make, not the backend's. **Phase 1 (§17) sidesteps it entirely**, which is another
reason to build that first.

### 19.2 The legal shape is different from Mo's, in one specific way

Mo's copyright position is clean because Mo **never touches a recording**: it
retrieves *facts* (progression, tempo, structure) from public sources and emits a
recipe. §2's invariants aim this feature at the same position — chord symbols only,
audio never persisted, the map never a product — and that reasoning holds.

The difference worth flagging to the lawyer §10 already calls for: **this pipeline
downloads and decodes the recording itself**, and §2 concedes the YouTube-terms
problem openly. Keep the two backends in separate repos with separate credentials
(as instructed), keep §3's kill switch genuinely one-flag, and **do not let the
chord-analysis service inherit Mo's deployment or Mo's blast radius**.

Also inherited from Mo, non-negotiably: **no lyrics, ever, in any field.** The app's
campfire rail has no lyrics line by deliberate design.

---

# AMENDMENT (2026-08-04) — the music-theory layer

**Supersedes §15 where they conflict.** §5.4, §12, §13 and §14 are unchanged and
still binding; this amendment sits *between* the engines and the compiler and
changes what is handed to §12, not what §12 emits.

## 20. Why a theory layer at all

Everything up to §19 treats the engines' output as the truth and the chart as a
tidied copy of it. That is the right default and it is also the reason the
service leaves accuracy on the table, because it ignores the single largest
piece of free evidence a song contains: **a song repeats itself.**

Almost every section of almost every song in this repertoire is one progression
played several times, with one strumming pattern, in one key, in one meter. A
chord recognizer, by contrast, makes **independent** mistakes — it can hear the
third verse differently from the first two even though the recording is nearly
identical. So four passes of one verse are four noisy readings of one signal,
and the disagreements between them are mostly the engine, not the music.

Two things follow, and the second is the dangerous one:

- structure that is **found** from the chords is destroyed by chord errors, and
  §15's segmentation compared bars by exact equality, so one misheard chord made
  a verse a different section;
- structure that is **imposed** on the chords can silently overwrite real music.

§20 is the attempt to get the first without the second.

### 20.1 The rule everything else hangs off

When two readings of the same bar disagree, **harmonic distance says whether
that is the engine or the music** (`app/analysis/harmony.py`).

C and Am share two of three notes. So do C and Em, C and Cm, C and Csus4. Those
are exactly the confusions a recognizer makes, because they are the confusions
the *signal* supports — in a dense mix the third really is ambiguous. C and F
share one note; C and F♯ share none, and no engine arrives at one from the other
by accident.

So a near-miss disagreement is evidence of noise and a distant one is evidence
of music. This is the whole safety argument, and it is why the layer is allowed
to touch anything at all.

### 20.2 Timing (`meter.py`)

The beat tracker is a *rhythmic* witness making a partly *harmonic* claim. Chord
changes overwhelmingly land on barlines, so the residue of chord-change
positions modulo the bar is a second opinion on where the "one" is.

A rotation is applied only when the tracker's own share of chord-change mass is
poor **and** some rotation is decisively better. Beat This! scores a downbeat F
of 0.893 on the real corpus; casually second-guessing it would lose more than it
won. Meter is arbitrated only when the tracker reports low confidence. **Tempo
octave errors are reported and never corrected** — that rewrites what a beat is,
and there is no clean harmonic evidence for it.

This is the same *class* of defect the alignment work fixed (`axis.py`): a chart
laid out of phase with its own recording scored 0.768 with a perfect engine, and
no lint could see it.

### 20.3 Form (`form.py`, supersedes §15's segmentation)

Bars are compared by **sound** (`harmony.similarity`), blocks are matched
**globally** rather than only against their neighbours, and the block grid's
**phase** is chosen by how much repetition it exposes — a song opening with a
two-bar intro used to put every boundary two bars out.

The output is a set of **repeat groups** with rehearsal letters, and §15's
`Part N` fallback is unchanged. `repeats` is still only used for *identical*
passes, never merely similar ones.

Two of §15's rules had to move, because both were written for a pipeline with no
clustering step in it and both destroyed the groups on ordinary input
(`app/analysis/form.py` carries the reasoning, `PIPELINE-AUDIT.md` the evidence):

- **The tail no longer joins the previous block.** That rule is right at *section*
  level and wrong before clustering: a four-bar verse with a two-bar tag stuck on
  it is a six-bar block, and blocks of unequal length score 0, so the song's last
  verse stopped being an occurrence of the verse — no consensus vote, onsets not
  pooled into the verse's strum pattern, drawn on the rail as different music.
  Since songs rarely end on an exact multiple of their own period, that was the
  last section of most songs. The tail is now its own block and the floor is
  applied afterwards.
- **The ~4-bar floor yields to `repeats`.** Absorbing a runt into a neighbour
  played 4× means flattening `repeats: 4` and pulling the runt's bars into
  another group's section — which turned `2-bar intro + verse ×4` into one
  eighteen-bar section carrying the *intro's* strum pattern. When the neighbour
  is a collapsed repeat the runt now stays a short section of its own. The floor
  is a preference; `repeats` is an encoding the client reads, and lint imposes no
  minimum section length.

`CANDIDATE_UNITS` also gained **12** — without it the twelve-bar blues, which is
core repertoire, was chopped at period 4 and its I/IV/V phrases scattered across
groups that are not sections — and 6, for six-bar phrases and the 12/8 blues.

Section *kinds* now include `bridge` (a group the song plays once, mid-song) and
`preChorus` (a repeated non-chorus group ending on the key's V immediately before
the chorus — `harmony.is_dominant_of`, which existed for this and was called by
nothing). The pre-chorus cue is gated on the song having **two or more** repeated
non-chorus groups, because a lone verse ending on V to lead into the chorus is
the most ordinary thing in this repertoire. `solo` stays unreachable: telling a
solo from a verse is a question about timbre, and one loudness scalar per hop
cannot answer it.

### 20.4 Consensus (`consensus.py`) — the dangerous part

Per repeat group, per bar slot, the occurrences vote. Overwriting requires
**three independent gates**, all of which must hold:

1. **support** — at least ⅔ of occurrences agree (so 1-of-2 never votes);
2. **near-miss** — every dissenter is harmonically close to the winner (§20.1);
3. **confidence** — the dissenter was believed *less* than the winner.

Gate 3 does the real work: it is the only one consulting evidence from outside
the repetition. An engine that was confident about the F in verse 3 is telling
us the F is there.

It also gives the layer a property worth stating: **on perfect input, consensus
is provably a no-op.** Ground truth arrives at a flat confidence of 1.0, so no
dissenter is ever less believed than a winner, so nothing is ever rewritten.

Where the gates do not all hold the slot is **contested**: nothing is rewritten,
for any occurrence, and the count is reported. Strumming patterns are pooled
across every bar of every occurrence of a group — the unambiguous half, since a
pattern was always a per-section average and this only enlarges its sample.

### 20.5 Key (`keyfinder.py`)

Scored against **four modes** — ionian, aeolian, mixolydian, dorian — and
projected to the container's major/minor at the wire.

The modes are there to find the **tonic**, which is what spelling keys off.
`G F C G` scored against major and minor only comes out A minor, because every
chord is diatonic to it and the start-and-end-on-G cue is outvoted by
membership. Only four modes, because modes of one collection contain the same
notes and cannot be told apart by membership at all — so every mode admitted
that does not really occur as a key is a pure liability. Lydian, phrygian and
locrian cost a correct tonic on the real corpus and bought nothing.

Modulation is deliberately **not** modelled: the container carries one key, so a
detected key change could be neither expressed nor acted on.

### 20.6 One model, three renders (`model.py`)

The structure is built **once**, at `hard`, and each difficulty is a render of
it. Boundaries, repeat groups and patterns are therefore identical across tiers
*by construction* rather than by inspection. (The cross-tier `lint_sync` check
stays anyway — it costs nothing and is not the kind of thing to remove on the
strength of an argument.)

### 20.7 The audio layer's one new output (`StructureProbe`)

§15 says the repeated high-energy segment can defensibly be called the chorus.
Nothing ever measured energy, so **every song shipped as `Part 1…N`** and that
half of §15 was dead code. A loudness envelope — one normalized scalar per
~46 ms hop — is now the only new thing crossing the audio boundary. It is not
audio, cannot be inverted into audio, carries no melodic, harmonic or lyrical
content, and is never persisted; §2.1 is unaffected. A probe failure **must not
fail the analysis**: without it, sections are `Part N`, which §15 calls the
honest answer.

### 20.8 The song's own vocabulary (`vocabulary.py`) — added 2026-08-05

§20.4's vote can only speak where a section **repeats** and its passes
**disagree**, and that is narrower than §20 assumed. A section that occurs twice
is one reading against one; a section that occurs once — intro, bridge, tag — has
nothing to compare against; and a mistake the engine made in *every* pass leaves
nothing to disagree with, because errors are only independent when the audio
differs. Reported from use, on a song whose chart is Ebm–Db–Ab throughout: one bar
showed `Ebm7`, another `Eb`, neither of them played, in a four-pass verse where two
passes had been misheard — which the two-thirds share filed as `contested` and
shipped unchanged.

So there is a second corrector, using a different kind of evidence: **the rest of
the song.** A song has a small chord vocabulary, and that is a fact measured over
minutes rather than over one bar. Two rules, on the span timeline, before bars are
cut and before the vote:

- **islands** — a brief, doubtful near-miss flanked by *the same chord on both
  sides and on the same root* is filled in. §5.4's `drop_short` is this rule with a
  one-beat floor and no harmonic test;
- **snap** — a minority reading of a root is pulled onto the quality the song plays
  on that root, when seven conditions hold: the move is one `SNAP_TO` allows, the
  chords are near neighbours, the reading appears on no more than two separate
  occasions, the song's reading is overwhelmingly the other one, this reading holds
  a sliver of the root's evidence, it was believed less than the song's answer, and
  the move is not away from the key.

**The root is never moved**, so every edit is a spelling correction. The
relative-major/minor confusion is therefore out of scope and stays there: both
chords are usually in the same song's vocabulary, so only bar-position evidence
can settle it, and that is §20.4's.

**`SNAP_TO` is measured, not reasoned**, and this is the part to preserve.
`is_near_miss` says two chords are confusable; it does not say which direction the
engine errs, which is the only fact that decides whether an edit pays.
`bench/run_bench.py --calibration` prints the table: a reported `dominant7` is the
plain triad 60% of the time and a seventh 31%, so flattening a doubtful one pays;
a reported `major7` is the plain triad *never*, so the same edit only loses.
`major7`, `augmented`, `diminished` and `diminished7` are excluded on that
evidence.

Like §20.4 it is **a provable no-op on perfect input** (the confidence condition),
and like §20.4 it has a flag: `CHORDS_THEORY_VOCABULARY`.

## 21. How this is measured, and what it actually bought

A layer that edits the chords an engine reported cannot be judged by one number,
because the two ways it can be wrong pull in opposite directions. So
`bench/run_bench.py --theory` runs it twice:

- **Run A — ground truth as both engines.** Nothing to fix, so anything it
  changes it changes away from the truth. Requirement: zero rewrites and an
  unchanged `delivered`. This is the regression guard.
- **Run B — the deployed engines.** Requirement: `delivered` goes *up*.

Measured on the nine-track real corpus (2026-08-04, and again 2026-08-05 with
§20.8 — `--theory` now prints three delivered columns and keeps the real and
synthetic means apart, which it previously did not):

| | off | consensus | + vocabulary |
|---|---|---|---|
| Run A, ground truth as both engines | **0.939** | **0.939** | **0.939** |
| Run B, the deployed engines | **0.796** | **0.800** | **0.803** |
| Run B at the pre-§20 commit | **0.796** | — | — |

Nine tracks cannot resolve an effect that size, which §20.8 had to solve before it
could be judged at all. `--noise` injects the engine's *measured* mistakes into
ground truth over the same nine songs and counts both sides of every edit — 12
seeds, `fixed` = share of injected errors removed, `broke` = share of correct
chords destroyed:

| layers | in | out | fixed | broke |
|---|---|---|---|---|
| consensus | 0.797 | 0.808 | 0.070 | 0.003 |
| vocabulary | 0.797 | 0.810 | 0.100 | 0.009 |
| both | 0.797 | 0.815 | **0.138** | **0.011** |

The two are nearly additive, which is the design claim holding: they answer with
different evidence and so fix different mistakes. They are never summed into one
score.

Read honestly:

- **The architecture is delivered-neutral.** The meter reconciliation, the new
  form detection, the model/render split and the modal keyfinder together move
  the number by nothing on either run. What they bought is structure and
  provenance, not accuracy.
- **Consensus is a marginal win.** +0.003 on the mean, with Michelle up 0.028
  and Let It Be *down* 0.014. That is real but within noise on nine tracks,
  which is why `CHORDS_THEORY_CONSENSUS` exists and why the benchmark prints
  MARGINAL rather than PASS below half a point.
- **§20.8 is the same size on the corpus and much larger on injected noise**:
  +0.003 delivered with no track regressing, and 10% of injected errors removed
  against 0.9% of correct chords damaged. On the song it was reported for it is the
  difference between a chart carrying `Ebm7` and `Eb` and one that reads
  Ebm–Db–Ab throughout.
- **Key detection is a wash**: 5/9 exact tonics before and after, with the
  mixolydian fix and the endpoint cap both correct on their own terms.

The honest summary is that §20 makes the song *more coherent* — repeats collapse,
tiers agree, sections carry group identity, the sidecar reports what was
changed — without yet making it much *more accurate*. Anyone extending this
should assume the accuracy win, if there is one, is in the engines and in
§20.2's phase check on tracks where the tracker is genuinely wrong, not in
voting harder.

**Amended 2026-08-05.** That last sentence held for a year of corpus numbers and
was still incomplete, because the corpus could not see what a user could: the
noise that survives §20.4 is not a residue but the ordinary case, and it is
visible on screen. The place to look for the next win is therefore neither the
engines nor harder voting but **the largest remaining measured confusion** — the
relative major/minor, 5.1% of minor chords coming back a third up. It needs
bar-position evidence, so it belongs to §20.4, and it is the one edit §20.8 is
forbidden from making.
