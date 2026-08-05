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
| 3 | Meter/tempo (D1–D2) | Not started |
| 4 | Hygiene (C1–C3, E1–E4) | Not started |

Tests: 414 at the start of the audit → 423 after Phase 1 → 432 after Phase 2.

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

## Phase 3 — meter / tempo ⏳

| # | Sev | Finding | State |
|---|---|---|---|
| D1 | MED | **A suspect tempo octave is detected, reported… and then hard-kills the song.** `meter.py` flags bpm outside 55–200 as a suspect octave and deliberately never corrects it; `lint.py` makes tempo outside 40–220 a fatal `LintFailure`. So a tracker reading 230 bpm for a 115 bpm song produces no song at all, with the generic "didn't produce a song we could play" — while the pipeline is holding the diagnosis (`tempo_octave_suspect=True`). The three tempo ranges in the codebase (55–200, 40–220, patterns' 30–300) are not consistent with each other. | ⏳ |
| D2 | LOW | `_rotate` rebuilds downbeats as `beats[start::bar_beats]`, which assumes a metrically uniform beat list — one inserted or dropped beat shifts every later downbeat, discarding the downbeat-aware tracker's ability to survive irregular bars. Rare (the phase gate is strict), but it corrupts the whole tail of the grid when it fires. | ⏳ |

Minimum viable D1 is *degrade honestly*: `tempo_octave_suspect` + out-of-lint-range
becomes low confidence and a specific player-facing error, not an opaque
`LintFailure`. Actually halving the grid rewrites the axis and needs the
benchmark's verdict first — stage it behind a flag like `theory_consensus`.

---

## Phase 4 — hygiene ⏳

Small, and mostly about failure modes lint cannot see.

| # | Sev | Finding | State |
|---|---|---|---|
| C1 | LOW | `render()` re-runs `consensus.apply` per tier, and `apply` writes `rewritten_bars`/`contested_bars`/`canonical` onto the **shared** `RepeatGroup` objects — so `model.groups` ends up carrying whichever tier rendered last, not the reference vote. The wire is fine (the sidecar snapshots earlier); benchmarks and logs read tier-polluted numbers. Also, the build-time vote used first-pass groups while tier renders use second-pass groups, so "hard = reference" holds by luck rather than by construction. | ⏳ |
| C2 | LOW | `postprocess.exact_ratio` is the promised "the hard tier is a fiction on this track" signal — promised again in `GridSpan`'s docstring — and is computed **nowhere in production**. Surface it in `TheoryReport` or delete the promise. | ⏳ |
| C3 | LOW | Key is detected *before* the vote, yet the vote's diatonic tie-break uses that key. Mildly circular; re-detecting after voting is free and strictly cleaner. | ⏳ |
| E1 | LOW | `compile_song` silently `continue`s a section whose group has no pattern — a silently shorter song. Unreachable today, which is exactly why it should raise rather than hide a future regression. | ⏳ |
| E2 | LOW | `lint_sync` checks anchors stopping *short* of the chart but never anchors running *past* its end. With E1, a dropped section would ship anchors addressing bars that do not exist. | ⏳ |
| E3 | LOW | `postprocess.merge` silently drops the second of two different chords quantized to the same start beat; it should keep the longer or more confident one. | ⏳ |
| E4 | LOW | Two groups whose grooves hash to the same content-addressed id share one `PatternPayload` and the last name wins — "Verse strum" shown where "Chorus strum" was meant. Cosmetic. | ⏳ |
