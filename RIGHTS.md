# Rights posture

What this service does with other people's recordings, why each choice was made,
and which lines are not up for negotiation. Written down because the reasoning is
load-bearing: several of the invariants below look like missing features until
you know what they are protecting.

This is an engineering document, not legal advice. Handoff §10's instruction —
have a media IP lawyer review the architecture before launch — is still open and
still the owner's.

---

## The two questions, kept apart

They get conflated constantly, and the answers are very different.

### 1. Copyright in the output — the strong position

The service stores **chord symbols, timestamps, a beat grid, a key and a tempo**.
It does not store, transmit, or make reconstructable any part of a recording.
Chord progressions are close to uncopyrightable fact; the value of the analysis
is in the timing and the arrangement into something playable, which is ours.

This position is not a lucky accident of the current code — it is maintained by
four invariants (handoff §2), each enforced somewhere you can point at:

| Invariant | Enforced by |
|---|---|
| §2.1 Audio is never persisted | `app/analysis/scratch.py` refuses a durable scratch root, cleans on every exit path including exceptions, and verifies rather than assumes; the Modal worker mounts **no Volume**; `tests/test_scratch.py` and `tests/test_pipeline.py` assert the root is empty after success *and* after failure |
| §2.2 Only the derived map is stored | No column in `chord_maps` and no field in `app/analysis/types.py` can hold PCM, a spectrogram, a chroma matrix, or a path; two tests assert the absence |
| §2.3 The map is not a product | No export endpoint, no chord sheet, no public read path; `/v1/maps/{videoId}` requires the same Firebase bearer as everything else |
| §2.4 Chord symbols only | No lyrics field exists anywhere; section names come from structure, never from text (`app/analysis/structure.py` names sections `Part 1`/`Part 2` when it cannot justify `verse`/`chorus`) |

**If a change would break one of these, the change is wrong** — that is §2's own
wording, and it is the reason this file exists rather than a comment.

### 2. Terms of service on the input — the weak position

Fetching and decoding audio separated from its video **contravenes YouTube's API
terms as written.** The handoff concedes this openly (§2, §19.2) and the owner
has knowingly accepted the risk. Nothing here papers over it.

The realistic failure modes are a Google API ban or a takedown request, not a
lawsuit, so the mitigations are built for those:

- **Blast radius.** `ytdlp_source.py` is imported only by the worker, its
  dependencies are not in the base install, and CI asserts the API surface cannot
  import them. Separate Modal app, separate credentials, separate secrets from
  Mo — a problem here cannot reach the other backend.
- **The kill switch is genuinely one flag.** `CHORDS_ANALYSIS_ENABLED=0`, read
  per request, no deploy. Cached maps keep serving; only new jobs stop.
- **Takedowns in minutes.** Per-video and per-channel blocklist, block-then-purge
  in one request, verified cascade, append-only audit log that deliberately
  outlives what it took down (`scripts/admin.py`).

---

## The upload path exists because of the above

`POST /v1/analyze/upload` analyzes audio the player already has
(`app/analysis/file_source.py`). It is not a convenience feature:

- it carries **no YouTube-terms exposure at all** — question 2 simply does not
  arise, and question 1's answer is unchanged because every §2 invariant applies
  identically;
- it is what the kill switch degrades **to**. Before it existed, throwing the
  switch — or YouTube's bot check hardening past what cookies can solve — took
  the whole feature offline. Now it takes the YouTube half offline.

`/healthz` reports `canAnalyze` and `canAcceptUploads` separately for this
reason: they are different capabilities with different exposure, and an operator
needs to see which one is up.

Uploaded bytes live in worker memory for one job, are written only into the
scratch directory, and are never stored — §2.1 is not weakened. The id is a
content hash, so the *derived map* caches and the audio does not.

---

## Where "make it work like Chordify" stops

The product target is Chordify parity for the **player**. Two of Chordify's
surfaces are deliberately not cloned, and this is the section to re-read when
someone proposes them:

**A public catalogue, search, and indexable song pages.** Chordify's millions of
publicly-readable chord pages are its biggest surface and precisely what makes it
a target. §2.3 forbids a public read path. The entire copyright position in
question 1 rests on the transcription never becoming the deliverable — a
crawlable page *is* the transcription as deliverable.

**Export — MIDI, PDF, chord sheets.** Same clause, same reasoning. The moment the
chart is a file the player takes away, this feature is standing where the tab
sites stood when the publishers came after them.

Everything else Chordify does — transpose, capo, simplify, loop, playback speed,
sections, chord diagrams, a per-beat grid — is either already supported by the
payload or is client-side work, and none of it touches this posture.

---

## Still owed, and by whom

| | |
|---|---|
| Register a DMCA agent (§3, §18) — blocks public exposure, not development | **owner** |
| Media IP lawyer review of the architecture (§10) | **owner** |
| §19.1: whether Phase 2 (a record playing under the player's strums) reverses the "no backing track" canon | **owner** |

Ask before any public exposure. Until the DMCA agent is registered, the takedown
machinery in `scripts/admin.py` has no front door attached to it.
