# Rights posture

What this service does with other people's recordings, why each choice was made,
and which lines are not up for negotiation. Written down because the reasoning is
load-bearing: several of the invariants below look like missing features until
you know what they are protecting.

This is an engineering document, not legal advice. Handoff §10's instruction —
have a media IP lawyer review the architecture before launch — is still open and
still the owner's.

---

## The three questions, kept apart

They get conflated constantly, and the answers are very different. This file used
to ask two; the third was added on 2026-08-04 after a survey of how the
comparable products actually operate, along with a split inside the second that
had been quietly costing us the strongest argument we have.

The survey's one-line finding, because it reframes everything below: **every
platform in this category splits playback from analysis, and only playback is
legal.** Chordify and Positive Grid's Spark both stream through the YouTube
embed — the rightsholder-sanctioned player, where Google serves the bytes, ads
run and Content ID pays out — and both obtain analysis audio by a route their
terms do not permit, because no licence for that route is on sale to anyone.
Positive Grid's own help centre says out loud that Spark uses YouTube as its song
source *because of copyright restrictions on music streaming*. We are not
choosing a worse posture than the incumbents. We are choosing the same one, with
a smaller output surface (below) and the takedown machinery already built.

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

### 2. Getting to the audio — the weak position

Two exposures, and this section used to run them together. They are not the same
shape, they do not have the same answer, and — the part that matters — **the
upload path and the YouTube path score differently on the first one.** Collapsing
them lost us that distinction.

#### 2a. The transient copy made in order to analyze

Before any chord is emitted, a recording is decoded into memory. That is a
reproduction, and it needs its own justification independent of question 1's
argument about the output. (Section numbers in this file are the handoff's;
"question 1/2/3" are this document's own.)

The justification is that this is non-expressive, intermediate copying: nothing
of the recording reaches a user, nothing substitutes for it in any market, and
the copy is gone when the worker dies (§2.1, and it is enforced, not asserted).
In the US that is the well-trodden fair-use line — Sega v. Accolade, HathiTrust,
Google Books, and more recently the AI-training decisions. In the EU it is the
DSM Directive's Article 4 text-and-data-mining exception, which is available to
commercial operators and is written for exactly this: automated analysis to
generate information about patterns.

**Article 4 has two conditions, and this is where the two paths separate.** It
requires *lawful access* to the work, and it yields to a machine-readable
reservation of rights by the rightsholder.

| | lawful access | reservation |
|---|---|---|
| `POST /v1/analyze/upload` | the user has it, and supplies the file | none reaches us |
| `POST /v1/analyze` (YouTube) | contested — see 2b | YouTube's terms are plausibly one |

So the upload path is not merely "the same thing with less ToS risk". It is the
path on which the copying itself has a clean affirmative defence in both major
jurisdictions. The YouTube path relies on §1's output argument plus the
practical mitigations below, and does not get to lean on Article 4 with any
confidence. That is a stronger reason to grow the upload path than convenience
was, and it is the reason `/healthz` reports `canAcceptUploads` separately.

#### 2b. Terms of service, and the anti-circumvention question

Fetching and decoding audio separated from its video **contravenes YouTube's API
terms as written** — they prohibit separating or isolating the audio component,
and prohibit downloading or caching copies without written approval. The handoff
concedes this openly (§2, §19.2) and the owner has knowingly accepted the risk.
Nothing here papers over it.

**What was missing here until 2026-08-04: this is not only a contract problem.**
There is a live anti-circumvention theory, and it is worse in Europe than at
home. The RIAA's 2020 DMCA §1201 takedown of youtube-dl argued that YouTube's
signature cipher is a technical protection measure; EFF pushed back, GitHub
reinstated the project, and US law on the point is unsettled. Germany went the
other way — OLG Hamburg upheld a ruling against Uberspace for *hosting*
youtube-dl.org, holding the cipher an effective technical protection measure. In
the EU, circumvention liability for this act is live law and it attaches to the
operator, not only to the end user.

This does not change the decision; it changes the *size* of the downside, and
therefore what the kill switch is for. It is worth the owner putting in front of
the §10 lawyer explicitly rather than as part of "we know the ToS says no".

The realistic failure modes are still a Google API ban or a takedown request
long before a lawsuit, so the mitigations are built for those:

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

### 3. Distribution — the App Store, which is nobody's copyright problem

The exposure with the shortest fuse, and it was not in this document at all until
2026-08-04. **Apple App Review Guideline 5.2.3** bars apps that save, convert, or
download media from third-party sources — YouTube is named in the text — without
explicit authorization from those sources, and reviewers ask for the paperwork.
We do not have the paperwork and cannot get it.

Chordify's and Spark's iOS apps live comfortably under this rule because the
download is server-side and invisible to a reviewer: the app embeds a player and
displays chords. That is the whole trick, and it converts into two engineering
rules that are **binding on the client repo, not this one**:

1. **No fetch or extraction code ships in the app binary.** Not yt-dlp, not a
   stream resolver, not a "just for debugging" URL builder. This is also why the
   idea of moving the fetch to the device — attractive because it solves the
   egress problem completely, the user's own IP being residential by definition —
   is not on the table. It trades a 5.2.3 rejection for an egress bill, and the
   youtube-dl/Uberspace ruling in 2b is a reminder that *distributing the tool*
   is what draws liability in the first place.
2. **Playback goes through the official YouTube iOS player** (the IFrame API in a
   `WKWebView`), never a resolved `googlevideo.com` URL. A reviewer who sees the
   app pull a media stream directly gets us rejected on a ground we cannot argue.

Neither rule costs anything today — the app already plays via the embed and has
never fetched — which is precisely why they should be written down before someone
proposes an optimisation that breaks one.

---

## The upload path exists because of the above

`POST /v1/analyze/upload` analyzes audio the player already has
(`app/analysis/file_source.py`). It is not a convenience feature:

- it carries **no YouTube-terms exposure at all** — 2b simply does not arise, and
  question 1's answer is unchanged because every §2 invariant applies
  identically;
- it is the only path where **2a's affirmative defence is clean**: the user has
  lawful access to their own file and no reservation of rights reaches us, so the
  transient copy is squarely inside Article 4 rather than arguing about whether
  it is;
- it raises no question 3 at all — importing a file the user already has is not
  downloading media from a third-party source;
- it is what the kill switch degrades **to**. Before it existed, throwing the
  switch — or YouTube's bot check, which is per egress IP and which cookies were
  measured not to solve — took the whole feature offline. Now it takes the
  YouTube half offline.

`/healthz` reports `canAnalyze` and `canAcceptUploads` separately for this
reason: they are different capabilities with different exposure, and an operator
needs to see which one is up.

**This is the path to grow.** Three of the three questions answer better here
than on the YouTube path, and the widening is all client-side work that needs
nothing from this repo: Files and iCloud import, AirDrop, and the user's own
non-DRM library via `MPMediaQuery` + `AVAssetExportSession` (Apple Music streams
are FairPlay — check `asset.hasProtectedContent` and fall back rather than
failing). Every track that arrives this way is one that never had to be fetched,
never spent an egress draw, and never needed 2b's argument.

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
sites stood when the publishers came after them. That is not a metaphor: the
publishers did come, the tab archives did go down, and Ultimate Guitar's survival
runs through a Harry Fox licence covering lyrics, title search and tablature
display. A licence is what that surface costs. We are not buying one, so we are
not building the surface.

**And the third rail, which is neither of the above: lyrics.** §2.4 already
forbids a lyrics field anywhere. The reason belongs here — lyrics are the one
element in this space with no "uncopyrightable building block" argument at all,
they are what every licensing regime is actually built to charge for, and adding
them converts a defensible chord service into an unlicensed lyrics service that
happens to show chords.

Everything else Chordify does — transpose, capo, simplify, loop, playback speed,
sections, chord diagrams, a per-beat grid — is either already supported by the
payload or is client-side work, and none of it touches this posture.

---

## Egress is a rights decision wearing an ops costume

It reads as a reliability problem — YouTube's bot check refuses roughly five in
six Modal IPs, measured — and the code treats it as one (`CHORDS_YTDLP_PROXY`,
and a retry budget that shrinks when it is set, `modal_app.egress_attempt_budget`).
It belongs in this file anyway, because two of the available answers are rights
decisions and only one of them is the ops one:

- **A rotating residential proxy** — the ops answer, and the recommended one. It
  changes nothing about questions 1, 2 or 3; it buys a working first attempt.
  Bounded by the cache: a chord map is per videoId and shared across all users,
  so this is paid once per *song ever*, on the order of a cent, and never again.
- **A static datacentre IP**, including a cloud provider's static-egress feature.
  Not a cheaper version of the above — a worse version of doing nothing. One
  address fetching audio all day is the fingerprint the check exists to find, and
  once flagged there is no fresh draw to retry into.
- **Moving the fetch to the device.** Solves egress completely and is ruled out by
  question 3. See rule 1 there.
- **A third-party "YouTube audio API" vendor.** Does not move the exposure in 2b
  anywhere — the act is still done on our behalf, for our product — and it costs
  the audit trail that every §2 invariant is currently enforced against. No.

Worth stating plainly so nobody spends another week looking: **there is no
licensed full-track audio source for popular music at consumer scale.** 7digital
(now MassiveMusic/Songtradr) licenses catalogue to *playback* services, not PCM
to analysis ones. Spotify withdrew preview URLs from new apps in late 2024, and
Apple's and Deezer's previews are thirty seconds — useless for a chart. That
absence is why every product in this category is where it is, and why the upload
path matters more than it looks.

---

## Still owed, and by whom

| | |
|---|---|
| Register a DMCA agent (§3, §18) — blocks public exposure, not development | **owner** |
| Media IP lawyer review of the architecture (§10) — now with 2b's anti-circumvention question raised explicitly, not folded into "the ToS says no" | **owner** |
| §19.1: whether Phase 2 (a record playing under the player's strums) reverses the "no backing track" canon | **owner** |
| Buy egress, or decide not to. `CHORDS_YTDLP_PROXY`, rotating residential. Unset is a legitimate deployment and `scripts/smoke.py` warns rather than fails — but it means the success rate tracks Google's datacentre policy rather than anything in this repo | **owner** (it costs money) |
| The two question-3 rules, enforced in the client repo — no fetch code in the binary, playback via the official player only | **client repo** |
| Widen the upload path on iOS (Files, iCloud, AirDrop, non-DRM library) | **client repo** |

Ask before any public exposure. Until the DMCA agent is registered, the takedown
machinery in `scripts/admin.py` has no front door attached to it.
