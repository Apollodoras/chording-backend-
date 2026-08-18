# Pipeline audit — findings and fix progress

An audit of the chords/strumming pipeline (`model` → `meter`/`axis` → `postprocess`
→ `form` → `consensus` → `strumming` → `compile`/`lint`/`sync`) against §14/§15/§20
of the handoff. Every finding below was **reproduced against the real code** before
it was written down, and every fix carries a test that fails without it.

The architecture came out sound — one model, three renders, one beat axis, gated
consensus. What the audit found is a class of defect the test suite is blind to by
construction: **the fixtures are all exact.** Exact grid positions, exact bar
counts, songs that begin on bar 0 and end on a unit boundary. Real input is
jittered, carries an intro, ends on a tag, and has a drummer in it. Every
high-severity finding here is green in 414 tests and wrong on a recording.

Severity is "how much of the chart the player sees is wrong", not how deep the bug
is. `HIGH` means the user notices.

## Status

| Phase | What | State |
|---|---|---|
| 1 | Strumming correctness (A1–A6) | **Done** — `26b4d84`, on `main` |
| 2 | Structure (B1–B5) | **Done** — branch `structure-audit` |
| 3 | Meter/tempo (D1–D2) | **Done** — branch `structure-audit` |
| 4 | Hygiene (C1–C3, E1–E4) | **Done** — branch `structure-audit` |
| 5 | The service half (F1–F4) | **Done** — branch `structure-audit` |
| 6 | The theory layer's blind spot (G1–G3) | **Done** |
| — | Seeded catalog (H1–H3) | **Open** — found, characterised, not fixed |

Tests: 414 at the start of the audit → 423 after Phase 1 → 432 after Phase 2 →
443 after Phase 3 → 458 after Phase 4 → 466 after Phase 5 → 511 after Phase 6.

Phase 5 is a **second audit**, run over the half the first one never looked at:
`main` / `jobs` / `store` / `auth` / `modal_app` and the upload path. The
pipeline findings above were re-checked first and all hold.

Phase 6 is a **reported defect** rather than an audit: chord noise the theory
layer was supposed to remove and doesn't. It is the first phase whose findings
came from someone using the app, and the first where the fix needed a new
measurement before it could be judged at all.

H1–H3 are **open**, and listed here rather than folded into a phase because they
were not found by reading code. They came from seeding the catalog with twelve
known songs and grading the emitted charts against published transcriptions
(`scripts/seed_catalog.py`). Nine came back correct; the three below did not.
None has a fix yet — they are written up because a characterised defect is worth
more than an unrecorded one, and because two of them are invisible to every
metric this repo already collects.

---

## Phase 1 — strumming correctness ✅

Shipped in `26b4d84`. 414 → 423 tests.

| # | Sev | Finding | State |
|---|---|---|---|
| A1 | HIGH | **The downbeat stroke vanished.** `extract` matched cells with `abs(p - position)`, but the bar is a loop: an onset 20 ms *ahead* of the "one" folds to ~3.98, where no cell can claim it. The beat-1 stroke silently disappeared from every pattern extracted off a real recording — and confidence in what was left stayed 1.0. | ✅ `_distance_in_bar` measures around the bar; `fold_onsets` rolls an anticipating onset forward onto the downbeat, and onto the bar it is evidence for |
| A2 | HIGH | **Onsets were backtracked**, i.e. rolled back to the preceding energy minimum — a slicing feature, wrong for timing, and biased early by an attack-dependent amount. That is exactly the direction that triggers A1, and folding does not average it out. | ✅ `backtrack=False` |
| A3 | HIGH | **Directions flipped down-for-up.** `choose_subdivision` scored the *share of onsets explained*, and every 8th is also a 16th, so the count always leans finer — one consistent hi-hat 16th per bar was enough to carry the grid to 16ths. `direction_for` reads the grid, so every "&" in the bar flipped to a downstroke. | ✅ scored by mean quantization error, coarsest-first only against grids it nests inside |
| A4 | MED | **Support counted onsets, not bars**, so one bar with a flam stood in for two bars of evidence. | ✅ `FoldedOnset` carries its bar; support counts distinct bars |
| A5 | LOW | Patterns took `f"{bar_beats}/4"` as their meter, so every 6/8 song shipped patterns labelled **"3/4"** — same bar length, wrong meter, and the opposite of what `strumming`'s docstring promises. | ✅ patterns take the song's own `time_signature` |
| A6 | LOW | `MIN_STROKES = 2` replaced a legitimate once-per-bar ballad strum with the four-downstroke fallback — *more* strokes than the recording has. | ✅ one stroke stands if played in ≥90% of bars (`SOLID_SUPPORT`) |

Test gap closed: strumming fixtures now jitter the onsets, bias them early, add a
ghost 16th, and pin the 6/8 label.

---

## Phase 2 — structure ✅

The two high-severity structural findings are the same shape as A1: an earlier
stage gets the right answer and a later stage throws it away.

| # | Sev | Finding | State |
|---|---|---|---|
| B1 | HIGH | **A short intro destroyed the song's form.** Reproduced on the most common shape in the repertoire — 2-bar intro + verse ×4 — which collapsed into **one flat 18-bar section carrying the intro's group**. So the chart got the *intro's* strum pattern (2 bars of evidence, usually the fallback), the verse group's pooled 16-bar pattern was computed and never referenced, `repeats: 4` was gone, and the rail showed one undifferentiated blob. `_absorb_runts` took `out[0].group = head.group` on a head merge, and expanded a `repeats > 1` host on any merge. | ✅ absorption always keeps the **host's** group, and never expands a collapsed repeat — the runt stays a short section instead |
| B2 | MED | **The final occurrence was expelled from its repeat group.** `_chunk` merged a short tail into the last block *before* clustering, so that block no longer matched the group's length and `block_similarity` returned 0. Verse ×4 + 2-bar tag → group A = `[0, 4, 8]`, and the last verse became a group of its own: no consensus correction, onsets not pooled, labelled as different music. Real songs rarely end on an exact unit multiple, so this hit the last section of most songs. | ✅ the tail is left as its own block; the ~4-bar floor is applied afterwards at section level, where it belongs |
| B3 | MED | **`CANDIDATE_UNITS` had no 12.** The 12-bar blues — core campfire repertoire — was chopped at period 4 and its I/IV/V phrases scattered across groups. Reproduced: 3 choruses of blues came out `A B B A B B A B B` instead of one group played 3×. 6 was missing too (6-bar phrases, 12/8 blues). | ✅ now `(4, 8, 12, 2, 6, 16)`; blues comes out as one 12-bar group played 3× |
| B4 | MED | **The labelling vocabulary was unfinished, and part of it was dead code.** `_assign_labels` only ever emitted chorus/verse/intro/outro; `preChorus`, `bridge` and `solo` are in the container's `SECTION_KINDS` and were never produced. `harmony.is_dominant_of` exists precisely for the pre-chorus half-cadence cue and its docstring points at `form.label` — **a function that does not exist**. | ✅ wired: unique mid-song group → `bridge`; repeated non-chorus group ending on V before the chorus → `preChorus`, guarded on the song having ≥2 repeated non-chorus groups. `solo` documented as unreachable — it is a timbre question |
| B5 | LOW | **The energy curve drifted ~1% against the song.** `hop_ms` was rounded to an integer (46.44 → 46), so by minute 3–4 the window read for a bar was ~2 s — most of a bar — away from the bar it claimed to measure, quietly degrading the one comparison the probe exists to make. | ✅ `hop_ms` is a float |

Test gaps closed: a section-level intro test asserting `sections` (group and
repeats, not only `groups` — the old test passed throughout the bug); a
verse×N+tag fixture asserting the last occurrence stays in-group; a 12-bar blues
fixture; bridge/pre-chorus labelling including the case the guard must *not*
fire on; a four-minute energy-drift fixture.

**One change was tried and reverted**, and it is worth recording because it looks
obviously right. Since a runt block matches nothing, scoring the phase search on
full-length blocks only seems like removing noise. It is not: it hands the search
a way to cheat, by shifting the grid until the song's one misheard bar lands in
the tail runt and then scoring a perfect 1.0 on what is left. `test_consensus.py`
caught it. The runt's 0 is real information about the offset and is scored.

### Measured

`bench/run_bench.py --theory`, 15 tracks, at `26b4d84` and again with Phase 2:

```
                        consensus off      consensus on      sections
truth  MEAN            0.963 → 0.963      0.963 → 0.963      no-op holds
btc    MEAN            0.822 → 0.822      0.825 → 0.825      19 bars rewritten (was 16)

btc    here-comes-the-sun          0.788  →  0.791   sects 14 → 15
btc    ob-la-di                    0.845  →  0.847   sects 16 → 18
btc    twist-and-shout             0.896  →  0.902   sects  3 →  5
btc    let-it-be                   0.807  →  0.792   sects  4 →  7
```

Read honestly: **delivered-neutral on chord accuracy** — the mean does not move,
and consensus-off is byte-identical everywhere, which is the expected result
since none of this touches chord recognition. What moved is the *structure*: every
affected track gained sections, because intros and tags are no longer swallowed,
and the vote is now taken over groups that are actually the same music. Three
tracks gained, Let It Be lost 0.015, and the consensus no-op on perfect input
still holds exactly.

Like §20 before it, then, what this bought is coherence and provenance rather than
accuracy — the difference being that here the coherence *is* the deliverable, since
the section rail and the strum pattern attached to each group are what the player
reads.

**One §15 rule was deliberately relaxed.** The ~4-bar floor no longer applies when
honouring it would flatten a `repeats > 1` neighbour. The floor is a §15
preference; `repeats` is an encoding both the container and the player's rail
read, and lint imposes no minimum section length. A standalone two-bar intro is a
much smaller lie than an eighteen-bar section claiming to be one piece of music.
`test_structure.py` now pins both halves of that rule.

---

## Phase 3 — meter / tempo ✅

| # | Sev | Finding | State |
|---|---|---|---|
| D1 | MED | **A suspect tempo octave is detected, reported… and then hard-kills the song.** `meter.py` flags bpm outside 55–200 as a suspect octave and deliberately never corrects it; `lint.py` makes tempo outside 40–220 a fatal `LintFailure`. So a tracker reading 230 bpm for a 115 bpm song produces no song at all, with the generic "didn't produce a song we could play" — while the pipeline is holding the diagnosis (`tempo_octave_suspect=True`). The three tempo ranges in the codebase (55–200, 40–220, patterns' 30–300) are not consistent with each other. | ✅ the ranges are declared together and nest; the pipeline now degrades honestly on both sides of the container's range; the octave rewrite itself ships behind `CHORDS_THEORY_TEMPO_OCTAVE`, off |
| D2 | LOW | `_rotate` rebuilds downbeats as `beats[start::bar_beats]`, which assumes a metrically uniform beat list — one inserted or dropped beat shifts every later downbeat, discarding the downbeat-aware tracker's ability to survive irregular bars. Rare (the phase gate is strict), but it corrupts the whole tail of the grid when it fires. | ✅ each downbeat moves relative to **itself**; the head is extended back over the music the forward rotation would leave in no bar, which is the coverage the slice had |

**D1, in the three pieces it turned out to be.**

*The ranges now nest, in one place.* `payload.py` declares all three with the
ordering as the contract — plausible (55–200) ⊂ container (40–220) ⊂ pattern
(30–300) — and `lint.py` and `meter.py` read them from there. The nesting is not
decoration: patterns are emitted at the song's own tempo, so an inversion is a
song that lints clean carrying a pattern that does not, and the innermost range
being inside the container's is what makes "implausible" a warning rather than a
death sentence.

*The pipeline degrades honestly, on both sides.* A tempo the container can still
carry but the analysis calls implausible now ships `low_confidence` — the song
lands in the Library and the sidecar is withheld, because the axis is precisely
what is in doubt. A tempo outside the container's range raises `TempoUnreadable`,
which names the reading and says what it usually means, instead of arriving three
lines later as `LintFailure`'s "that video didn't produce a song we could play".
Both are reachable in `tests/test_pipeline.py`; before the change the second case
was reproduced as exactly that opaque lint failure.

*The rewrite is staged, not shipped.* `Settings.theory_tempo_octave`
(`CHORDS_THEORY_TEMPO_OCTAVE`) halves or doubles the whole grid — beats **and**
downbeats, so two of the tracker's bars become one, walked bar by bar rather than
sliced for the same reason D2 exists. It is off by default and for a different
reason than `theory_consensus`: not "measured and marginal" but **unmeasured**.
Only a single octave is ever applied, and only when it lands inside the plausible
band; 460 bpm halves to 230, which is still not a tempo, so it is declined and
stays suspect. That is the difference between an octave error and a tracker that
failed.

### Measured

`bench/run_bench.py --theory`, at Phase 2 and again with Phase 3: the `truth` run
is **byte-identical**, 0.963 → 0.963, every track unchanged, and the emitted
contract fixtures differ by exactly one additive field (`tempoOctaveShift: 0`).

That is the expected result and it is worth stating plainly rather than dressing
up: **no track in the corpus has a suspect tempo** — every ground-truth tempo is
between 66 and 179 bpm — and the phase gate does not fire on a ground-truth grid,
so neither D1 nor D2 has anything to change here. The corpus cannot price this
phase. What it can do is prove Phase 3 broke nothing, which is what it did.

The `btc` + Beat This! row could not be re-run: `beat_this` is not installed in
this environment, and that is the run that would exercise the two cases — a
tracker whose reported bpm is an octave out, and a phase rotation firing on a
grid with irregular bars. Both fixes are therefore pinned by fixtures rather than
by the benchmark, and D2's fixture was checked to **fail against the old
`_rotate`** (it puts every bar line a beat off the music).

**The one thing not done, deliberately.** `_octave` is not consulted about *what*
the octave error is — it trusts `BeatGrid.bpm` and rescales by 2. A tracker that
reports 3× (counting the eighths of a 6/8 bar) is out of its reach, and the
harmonic evidence that would settle it is the same chord-changes-on-barlines
histogram the phase check uses. That is a real extension and it needs a track
that exhibits the failure before it is worth writing.

---

## Phase 4 — hygiene ✅

Small, and mostly about failure modes lint cannot see.

| # | Sev | Finding | State |
|---|---|---|---|
| C1 | LOW | `render()` re-runs `consensus.apply` per tier, and `apply` writes `rewritten_bars`/`contested_bars`/`canonical` onto the **shared** `RepeatGroup` objects — so `model.groups` ends up carrying whichever tier rendered last, not the reference vote. The wire is fine (the sidecar snapshots earlier); benchmarks and logs read tier-polluted numbers. Also, the build-time vote used first-pass groups while tier renders use second-pass groups, so "hard = reference" holds by luck rather than by construction. | ✅ a render is a **read**: `apply(record=False)`, so nothing is written back, and the model carries `vote_groups` — the pass the vote was actually taken over — for renders to replay |
| C2 | LOW | `postprocess.exact_ratio` is the promised "the hard tier is a fiction on this track" signal — promised again in `GridSpan`'s docstring — and is computed **nowhere in production**. Surface it in `TheoryReport` or delete the promise. | ✅ computed on the reference tier in `model.build`, carried as `SongModel.exact_ratio`, reported as the sidecar's `analysis.exactRatio` |
| C3 | LOW | Key is detected *before* the vote, yet the vote's diatonic tie-break uses that key. Mildly circular; re-detecting after voting is free and strictly cleaner. | ✅ re-read off the corrected bars (`structure.spans_from_bars`), and only when the vote actually rewrote something |
| E1 | LOW | `compile_song` silently `continue`s a section whose group has no pattern — a silently shorter song. Unreachable today, which is exactly why it should raise rather than hide a future regression. | ✅ raises `ValueError` naming the group |
| E2 | LOW | `lint_sync` checks anchors stopping *short* of the chart but never anchors running *past* its end. With E1, a dropped section would ship anchors addressing bars that do not exist. | ✅ both ends of the coverage rule now, with the final barline (`songBeat == length`) explicitly not "past" |
| E3 | LOW | `postprocess.merge` silently drops the second of two different chords quantized to the same start beat; it should keep the longer or more confident one. | ✅ longer wins, confidence breaks the tie |
| E4 | LOW | Two groups whose grooves hash to the same content-addressed id share one `PatternPayload` and the last name wins — "Verse strum" shown where "Chorus strum" was meant. Cosmetic. | ✅ named for both ("Verse & Chorus strum"), in `model._patterns` where the names are, so the id — which hashes meter and strokes, not the name — is untouched |

**Where C1's two halves ended up.** The write-back is the reproducible one:
a model built on sevenths, rendered at `easy`, came back describing itself with
plain triads — `test_rendering_a_tier_does_not_edit_the_model_it_renders` fails
against the old code. The "hard = reference" half is pinned as a property rather
than as a reproduction, and that is worth saying plainly: on every fixture tried,
the two form passes agree, so voting over the wrong one produced the same answer.
It held by luck and now holds by construction, which is the whole claim.

**E4 was not fixed where it shows.** The collision appears in `compile`, which
keys embedded patterns by id — but the name is decided in `model._patterns`,
where the group names still exist, so that is where it is fixed. Renaming is safe
for exactly the reason the bug exists: §12.5's id is a hash of the meter and the
strokes and deliberately not of the name, so the wire is byte-identical. Beyond
two sharers the pattern is named "Strum" — a groove the whole song plays belongs
to no section, and a list of names has stopped being a name.

### Measured

`bench/run_bench.py --theory`, at Phase 3 and again with Phase 4: the `truth` run
is **unchanged**, 0.963 → 0.963, and consensus is still a provable no-op on
perfect input (0 rewritten bars on all 15 tracks).

That is not a hopeful reading of an unchanged number — on this corpus Phase 4
*cannot* move that run, and it is worth writing down why. Every behavioural change
in it is either downstream of the wire (C2, E1, E2, E4) or gated on the vote
having rewritten a bar (C1's replay, C3's re-read), which on ground truth never
happens. That leaves E3, the one change that can alter a chord on any input — and
instrumenting `merge` across the corpus counts **zero** coincident-start
collisions in the truth run, so it never fires. What the benchmark can say here is
that Phase 4 broke nothing, and it says it.

The `btc` + Beat This! row could not be re-run, for the same reason as in Phase 3:
neither engine is installed in this environment. E3 is the finding that row would
price, since quantization only puts two chords on one beat when the chords arrive
off-grid — which is what a real engine's output is and what ground truth is not.
It is pinned by fixtures instead.

---

## Phase 5 — the service half ✅

A second audit, over the code the first four phases never touched: the HTTP
service, the job seam, the store, auth, the Modal deployment and the upload
path. The pipeline was re-checked first — suite green, emitted fixtures
byte-stable, the API image still provably audio-free — and every Phase 1–4
finding still holds.

The defects here are a different shape from the pipeline's. Those were "the
analysis is confidently wrong about the music"; these are **"the service is
confidently wrong about what it just did"** — a promise in a docstring that the
code no longer keeps. All four were reproduced before being written down, and
each carries a test that fails without its fix.

| # | Sev | Finding | State |
|---|---|---|---|
| F1 | MED | **A job that analyzed cleanly and could not be *filed* hung.** `run_job`'s docstring promises it never raises, because it runs detached from any request — "every exit path writes a terminal status". Every *failure* path did. The success path did not: the `put_map` loop and the final `update_job` sat outside the try, so a dropped connection or a serialization error escaped, leaving the row at `analyzing` with progress 0.05, no message, and the charge spent. Under `ThreadJobRunner` there was not even a log line — the exception lands in a `Future` nobody reads. | ✅ persistence is inside the try; a store failure writes `failed` and refunds. `_finish_failed` now cannot raise either, so the promise holds on *every* path |
| F2 | LOW–MED | **A tier render replayed the vote with the wrong key.** C1 established "a render reproduces the reference vote" by replaying over `vote_groups`. C3, in the same phase, started re-reading the key *after* the vote — so `model.key` is the post-vote reading, and `render` was replaying with it while `build` had voted with the pre-vote one. The vote's diatonic tie-break consumes the tonic, so on any song where the correction moves the reading, the replay breaks a contested bar the other way. Both are guarded on the same `consensus.touched`, so the two conditions never came apart. | ✅ the model carries `vote_key` — the reading the vote was taken with — and `render` replays with it, exactly as it replays over `vote_groups` |
| F3 | LOW | **An upload's decode was measured, not bounded.** `gate()` refuses a long file on the duration **ffprobe claimed**, and a claim is not a measurement. ffmpeg then ran with no `-t`: a container understating its real length decoded unbounded into the scratch dir — tmpfs, so RAM on the worker — and was read into memory a second time by `_read_wav`. On a container with a 4 GB cap that is a dead worker rather than a clean 400. | ✅ `-t` bounds the decode itself, at `_DECODE_CEILING` — deliberately **above** `_LENGTH_TOLERANCE`, so an over-length file is still refused instead of truncated into legality |
| F4 | LOW | **No lint gate.** ruff had been run locally (there is a cache) but was not in the `dev` extra and had no CI job, so its findings were nobody's. Eight of them, all cosmetic — one in production code (a dead `import numpy as np` in `librosa_beats.detect`), the rest unused imports and an unread local in tests and bench. | ✅ ruff in `[dev]`, a `Lint` step ahead of `Test` in CI, config in `pyproject.toml`, and the eight fixed |

**F2 is a consistency fix, not a repaired chart, and that distinction is the
honest one.** The inconsistency is structural and certain — the two calls are
passed different keys, and the code says so. What could not be shown is it
changing an actual song: a sweep of 25 fixtures where the vote fired found no
case where re-reading the key moved the reading at all, because one or two
corrected bars out of sixteen do not shift a key detection. So the test forces
the re-read rather than finding it, and pins the property instead of a
reproduction. That is the same footing C1's own second half ended up on, and for
the same reason — which is worth noticing, since F2 *is* the gap C1 left.

**On F1's severity.** The lease reaper (`_JOB_LEASE_S`, 900 s) already collects
a row abandoned mid-flight and refunds it, so this was never permanent: the
player saw fifteen minutes of `analyzing` and then a terminal answer. What the
fix buys is answering **now**, for a job that had in fact succeeded, and — the
part that mattered more — a log line where there had been silence.

### Also looked at, and left alone

- **The per-IP limiter reads the rightmost `X-Forwarded-For` entry.** Correct
  behind exactly one trusted proxy, which is the deployed shape, and the safe
  direction to be wrong in (over-limiting, not a free bypass). Worth revisiting
  only if a second proxy is ever put in front.
- **`hit_rate_limit` is not atomic across connections on Postgres.** Two
  concurrent requests can both read under the limit and both insert, admitting
  `limit + 1`. It is a burst limiter and the daily quota is the real budget;
  making it atomic costs a lock on the hot path for one request.
- **Both rate limits default to `0` (off).** Documented that way, and the README
  tells the deployer to set them in the Modal secret. A default that is on would
  be a limiter tuned for nobody's traffic.

---

## Phase 6 — the theory layer's blind spot ✅

The first phase that started with a **report from using the app** rather than with
a read of the code:

> the chord analysis has "noise" like some wrong chords — The Silence by
> Manchester Orchestra has only Ebm and the app displays Ebm7 and Eb — the purpose
> of the intelligent layer is to avoid this problem, and obviously it's not
> working.

Both symptoms reproduce, and the reproduction is the finding. The song is
Ebm–Db–Ab–Ebm (Eb dorian; the Ab major rather than Ab minor is what makes it
dorian). Fed a verse played four times with the tonic misheard as `Ebm7` in one
pass and `Eb` in another — a 12% per-bar error rate, which is *better* than BTC
manages on the corpus — the chart shipped both mistakes at every difficulty, and
the sidecar reported the slot as `contested`, meaning "this song's verses
genuinely differ".

| # | Sev | Finding | State |
|---|---|---|---|
| G1 | HIGH | **The vote was defeated by the ordinary noise rate.** Gate 1 required two-thirds of a group's occurrences to agree *exactly*. Two mistakes in four passes is a 2-of-4 plurality, which fails that, so the slot was contested and **both** mistakes shipped — and each additional bad reading pushed the share further down, so the repetition that is supposed to be the evidence counted against it. Two errors in four passes is not a corner case; at BTC's measured per-bar rate it is the common case. | ✅ gate 1 is now a **plurality with a floor**: at least `MIN_AGREEING` (2) occurrences reading the slot identically, and no other reading agreed by as many. Gates 2 and 3 are unchanged and still applied to every dissenter individually, so one confident or harmonically distant reading still contests the whole slot |
| G2 | HIGH | **The vote was the only corrector, and it can only speak where a section repeats *and* its passes disagree.** Three ordinary situations fell straight through: a section that occurs twice (one reading against one), a section that occurs once (intro, bridge, tag), and a mistake the engine made identically in every pass — errors are only independent when the audio differs. Nothing anywhere in the pipeline consulted **the song's own chord vocabulary**, which is the evidence a musician would use and which is available over minutes rather than over one bar. | ✅ new `app/analysis/vocabulary.py` (§20.8), run before the bars are cut: islands filled, minority readings of a root snapped onto the one the song plays, seven gates, same root always, `CHORDS_THEORY_VOCABULARY` to turn it off |
| G3 | MED | **The layer could not be judged.** `--theory` is the only harness that scores it, and the population any quality rule may touch is a few dozen spans across nine tracks — far too little to resolve a half-point effect, and enough that one track's idiomatic sevenths swamp the mean. So "is this rule right?" had no answer, in either direction. | ✅ two new bench modes. `--calibration` measures the engine's actual confusions, which is what `SNAP_TO` is built from; `--noise` injects those measured mistakes into ground truth, so the same nine songs carry hundreds of errors whose correct answers are known |

### Measured

`bench/run_bench.py --theory`, delivered accuracy, as the harness now prints it —
three columns, because the two correcting layers answer with different evidence:

```
run     track                     off   cons   both   delta rewrit  snap  isle
truth   REAL MEAN               0.939  0.939  0.939  +0.000      0     0     0
btc     REAL MEAN               0.796  0.800  0.803  +0.007     16    10     2
btc     ALL MEAN                0.822  0.826  0.827  +0.005     16    10     2
```

The truth run is exactly a no-op — the property both layers are built to have, and
it holds by construction rather than by luck (every gate that can open needs a
*gap* in confidence, and ground truth has none).

On the engine run, **no track regresses against consensus-only**, and the three
that move go up: Something +0.015, Here Comes The Sun +0.005, Norwegian Wood
+0.002. Read precisely: off → both clears `MATERIAL_GAIN` and the harness prints
PASS, but §20.8's own marginal contribution over the vote is +0.003 — *below* that
bar, and reported as marginal rather than dressed up. Which is also why G3
mattered: nine tracks cannot resolve +0.003 in either direction.

Two mean rows now, real and synthetic split, which this harness's own docstring
always required and `bench_theory` was quietly not doing — it averaged the two
corpora together. Correcting it moves the printed numbers (0.822 → 0.796 for the
same run) without changing any analysis.

`--noise`, 12 seeds × 9 tracks, which is where the population is big enough to
resolve. `fixed` is the share of *injected* errors removed; `broke` the share of
*correct* chords destroyed. They are never summed:

```
layers          in     out    delta   fixed   broke
consensus      0.797  0.808  +0.010   0.070   0.003
vocabulary     0.797  0.810  +0.012   0.100   0.009
both           0.797  0.815  +0.017   0.138   0.011
```

The two layers are close to additive (7.0% + 10.0% ≈ 13.8%), which is the design
claim holding up: they answer with different evidence and therefore fix different
mistakes. Twelve errors removed for every one introduced.

The run is **seeded from the track name**, not from `hash()`, and that was a bug
worth recording: Python randomizes string hashing per process, so two runs of
identical code drew different corpora and printed numbers ±0.005 apart. A
benchmark whose answer depends on which process it ran in cannot be quoted, and
this one was being quoted.

### What the corpus overruled

Four things that looked right and were wrong, each caught by measurement rather
than by argument. They are the reason `SNAP_TO` is a table of measured moves
rather than "anything `harmony.is_near_miss` admits":

- **A generic near-miss rule cost accuracy.** Near-miss says two chords are close
  enough for a recognizer to slide between them; it says nothing about *which
  direction it slides*, which is the only fact that decides whether an edit pays.
  Flattening every doubtful seventh took In My Life down 0.031 (four real,
  hedged A7s) and Let It Be down 0.003 (the opening Fmaj7). Measured, a reported
  `dominant7` is the plain triad 60% of the time and a seventh 31% — worth doing —
  while a reported `major7` is the plain triad **never**. `major7`, `augmented`,
  `diminished` and `diminished7` are excluded on that evidence.
- **Duration could not tell vocabulary from noise.** In My Life plays A for
  seventy beats and A7 for nine, doubtfully, in four passes; Something reports G7
  twice, just as briefly and doubtfully, and the record plays plain G both times.
  No measure of *amount* separates them. What does is **occasions**:
  `MAX_OCCASIONS = 2`. Swept — 1, 2, 3, 4, unlimited — the damage all lives at 4
  and above, which is where In My Life's A7 enters. Counting every edit on the nine
  tracks: at ≤ 2, six corrected, one damaged, four neutral; unlimited, seven
  corrected, **five** damaged, eight neutral.
- **Islands could not be allowed to cross roots.** Fm | Caug | Fm looks like a
  hole in a held chord, and Michelle's augmented chord was *right*: an augmented
  triad is one set of notes under three names, so a rule reasoning about labels
  cannot tell Caug from Eaug. Same root only, and the edit is a spelling
  correction rather than a new chord.
- **The strict mass gates are free.** `MASS_DOMINANCE` (6×) and `MINORITY_SHARE`
  (0.15) were swept down to 2× / 0.35 on both harnesses: the delivered mean does
  not move at any setting, and the real corpus prefers the strict end. So they
  stay strict, and the cost of that is stated rather than hidden — on a very short
  song, or a root the song only plays a few times, the rule declines to speak.

### The one thing left alone, deliberately

The **relative-major/minor confusion** — hearing Gb where the song plays Ebm — is
the largest single bucket of engine error in the corpus (5.1% of minor chords come
back a third up) and is out of scope for §20.8. Both chords are usually in the
same song's vocabulary; they are in The Silence, whose chorus opens on Gb. Mass
cannot tell which one belongs in a given bar, so deciding it needs the same bar in
another pass — `consensus`'s evidence, not this module's. Guessing it from mass
would put a chord nobody played into the chart, which is what the whole layer
exists to avoid.

### And a caveat on the new harness

`--noise` draws its mistakes independently per chord, so it **cannot** reproduce a
mistake the engine makes identically in every pass — real, common, and the thing
that defeats the vote. It also inherits whatever its model leaves out, and that is
not hypothetical: the first version of the model had no rows for the sevenths, so
every genuine seventh arrived fully believed, no confidence gate could open on
one, and the run was structurally incapable of seeing the damage the real corpus
had already caught on In My Life. `broke` read 0.004 instead of 0.011. A synthetic
benchmark answers exactly the question its noise model asks — which is the same
lesson as the exact fixtures at the top of this document, one level up.

---

## Seeded catalog — H1–H3 ⚠️ open

Not an audit and not a bug report: twelve songs with published transcriptions,
run through the deployed worker image, the emitted chart graded against the
transcription. `scripts/seed_catalog.py` carries the ground truth and the
reasoning for each entry. Nine of twelve came back correct on key, meter, chord
vocabulary and the defining cycle. These three did not.

Severity keeps the meaning it has above — how much of what the player sees is
wrong.

### H1 — compound meter is unanalyzable, not inaccurate · HIGH

House of the Rising Sun (6/8) produces **no chart at all**. The beat tracker
locks onto the eighth-note triplets instead of the dotted-quarter pulse, reports
231 BPM, and `meter`'s 40–220 guard rejects the whole analysis as
`tempo_unreadable`.

231 is ≈3× the 77 BPM the recording is actually in, so this is not a tracker
that lost the song — it is a tracker that found the subdivision and called it the
beat. `tempoOctaveSuspect`/`tempoOctaveShift` already exist for exactly this
shape of error at 2×; the ternary case has no path through them, and the guard
fires before anything downstream could reconcile it against the harmony.

Worth noting what it costs: a 6/8 song is not *slightly* wrong here, it is a
failed job with a user-facing error. Every other defect in this document
degrades the chart. This one withholds it.

### H2 — dominant harmony reads as minor · HIGH

The 12-bar blues in E is the one wrong chart in the set, and the way it is wrong
is the finding. The **roots are perfect** — E, A and B account for every chord
in the song. The **qualities flip**, on the same root, in a song that contains no
minor chord: `E` 64 spans against `Em` 35, `A` 11 against `Am` 30, `B` 19
against `Bm` 3.

Checked against the recording rather than against our own engine — a CQT chroma,
averaged over the track — the major third wins on all three roots, and narrowly:

| root | major third | minor third |
|---|---|---|
| E | G# 0.080 | G 0.071 |
| A | C# 0.064 | C 0.061 |
| B | D# 0.085 | D 0.057 |

So the minor readings are errors, and the thin margins say why: a blues plays the
minor third *as a blue note* over a dominant chord, so the pitch that
distinguishes E from Em is genuinely present in the audio. The engine is not
hallucinating, it is resolving a real ambiguity the wrong way, bar by bar,
inconsistently.

Two things follow, and the second is the uncomfortable one:

- §20.8's vocabulary rule cannot reach this. Both readings are *in* the song's
  own vocabulary by mass — that is precisely the situation the module declines
  to speak in, for the same reason it declines on relative major/minor.
- **Root-only accuracy scores this 100%.** It is reported beside per-beat
  accuracy in `bench/run_bench.py` specifically as the "right harmony, wrong
  quality" signal, and here it reports nothing at all. `real_song_check.py`
  grades roots only, passed this song, and the README recorded the major/minor
  split as "genuinely ambiguous, not an error" — which is how a wrong chart sat
  behind a green gate and a written-down explanation.

### H3 — the key model is weakest where the chords are right · MEDIUM

Hey Joe: 100% of chords inside C–G–D–A–E, the cycle found in order, and the key
reported as **Am**. The blues: roots perfect, key reported as **Am**. Both are
songs whose chords are all major and whose tonal centre is unambiguous by ear.

Two other key mismatches in the set are *not* defects and are recorded here so
the ratio is not overstated: Sweet Home Alabama reported G against a truth of D
(D Mixolydian and G major are the same seven notes — the uploader's own title
says "in G (D Mixolydian)"), and Autumn Leaves reported Gm against a truth of Bb
(the relative pair; the tune resolves to Gm). A key label that picks the wrong
member of a relative or modal pair is a convention disagreement. Calling a
five-major-chord song A minor is not.

### The method's own failure mode, recorded

Zombie first graded 50% and read like an engine failure. It was not. The cover
used is **up a fifth** from the record — Bm–G–D–A where every published chart
says Em–C–G–D — and the truth entry, not the chart, was wrong.

The recording settles it independently of anything in this repo: over a CQT
chroma, **C is the least present of the twelve pitch classes** (0.042), and
Zombie in Em spends a quarter of its length on C.

This is the standing hazard of grading against covers, and the reason the ids in
`seed_catalog.py` are ordinary uploads in the first place. (That reason was
originally "the originals cannot be fetched"; since 2026-08-17 they can, and the
reason is now the rights posture — see `seed_catalog.py`'s own note.) A chart that disagrees with
the songbook is always two hypotheses — the engine misheard, or the performance
is not in the songbook's key — and the second one has to be excluded with a
measurement before the first is written down as a defect.
