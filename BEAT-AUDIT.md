# The beat pipeline — audit, and the plan to fix it

## Status

| Phase | What | State |
|---|---|---|
| 1 | Repair the downbeat sequence (`downbeats.py`), tracker meter + confidence | **Done** |
| 2 | Musical patterns: contrast, budget, idiom vocabulary, consolidation, the repeat rule | **Done** |
| 3 | `lint_sync` bar-length check, `TheoryReport` provenance, broken-grid fixtures, graded beat map | **Done** |
| — | Deployed and run on real audio | **Done** — and it found a bug the whole synthetic suite could not (below) |
| — | The structure probe's one look at a fresh run's logs | **Done** — it runs; sections come back `verse`/`chorus`/`bridge`, not `Part N` |
| — | The form's block sizes | **Open** — 64- and 72-bar "verses" on Canon Rock; the bar grid is right and the *blocks* are not |

Tests: 519 at the start → **547**, and the new ones are the grids the suite never
had (spurious downbeat, dropped downbeat, genuine irregular bar, and the
sub-multiple trap below).

## What the deployment said, and the bug it found

Deployed to `rosetta-dechorder`, three gates green, then run on real songs. The
first run was wrong in a way no synthetic fixture had caught, and the log said so
in one line:

    98% of bars disagreed with the 2-beat mode before repair

**A 2-beat bar, on a 4/4 song at 97 bpm.** The cause was the candidate-scoring
rule in `_modal_bar`: it tried each candidate through the walk and kept whichever
left the fewest bars disagreeing afterwards. But the walk can *insert*, so a
sub-multiple always scores perfectly — halve every bar and every bar agrees with
the half. Deployed, that inserted 93 downbeats into Blues in E and 228 into the
Hallelujah cover, doubled their bar counts, fragmented the form into 28 and 49
sections, and dragged `meter_agreement` down with it (Blues in E's grid
confidence read 0.26 — computed against a mode that was itself the bug). Only the
`IRREGULAR_CEILING` flag stood between that and the player.

Candidates are now scored on the downbeats the tracker **actually produced** —
bars of that length times that length, in beats — so a song in four wins on
coverage however many spurious downbeats it carries. `test_downbeats.py` pins it.

After the fix, on real audio:

| song | before | after |
|---|---|---|
| Hallelujah cover | 49 sections, +228 downbeats inserted, 2-beat "bar" | 10 sections, +2 inserted, 6-beat bar (it is a 6/8 cover) |
| Canon Rock | — | **sidecar published**; 19 irregular bars of 241 (7.9%), one strum for all 12 sections |
| Hey Joe | — | **sidecar published**; 0 of 56 bars off the song's own 4000 ms bar |

Canon Rock is the case Finding 4 was written about: a distorted band mix with a
full kit, which used to produce fifteen of sixteen 16th cells. It produces
`0d 1d 2d 3d` — four strokes, snapped to quarters, and every one of its twelve
sections points at it.

**Hallelujah is the one still withheld**, and correctly: `beat_this` reads its
triplet 8ths as the pulse (214 bpm), so the meter arbitrates to 3/4 while the
repair measures a 6-beat bar. That is H1 from `PIPELINE-AUDIT.md` — the ternary
gap, which has no octave path — not this audit's defect. The chart still ships;
only the video sync is withheld.

**Fetch, not analysis, is what limits a run like this.** Nine of sixteen attempts
never reached the audio: five seed ids now answer `video_unavailable` (gone or
region-locked from the proxy's exit) and Blues in E was bot-checked 3/3. Nothing
to do with this work, but it is why the sample is four songs and not twelve.

Two things to know before reading the plan below as if it were still pending:

**The confidence change shipped.** `regularity * meter_agreement`, per Phase 1.
That was flagged below as a product call — whether a song with an unreadable bar
grid is allowed to lose video sync — and it was taken the way the audit
recommends, because the alternative is a sidecar nobody can vouch for. It is one
line in `beat_this_tracker.track` if it needs reversing. Measured on the four
songs that got through: two ship a sidecar (Canon Rock at grid confidence 0.566,
Hey Joe), one is withheld for the ternary reason above, and one was never
re-reached after the fix. `pipeline.assemble` now records **why** on the outcome
(`low_confidence_reasons`) and logs it — before this there was one boolean and no
way to tell four different diagnoses apart.

**The meter estimator is shared now.** A plain mode over bar lengths elects 3/4
on a grid where half the bars carry a spurious downbeat (a `4` becomes a `1` and
a `3`), so the tracker and the repair would have disagreed about the meter on
exactly the songs the repair exists for. `downbeats.modal_bar_beats` tries each
candidate and keeps whichever leaves the fewest bars disagreeing with themselves;
`beat_this_tracker._meter` calls it, so there is one answer.

---

Three complaints came in from the player side:

1. **"Beat-map anomalies"** — segments 2–2.6× the length they should be (Knockin'
   8/93, Sweet Home Alabama 6/122, worst cases late in the song).
2. **"It sometimes changes the tempo mid song."**
3. **"Patterns should be more musical. If a pattern doesn't repeat then it's not
   a pattern. We should have one to few patterns that repeat through the song."**

They are not three problems. **(1) and (2) are the same defect seen from two
angles, and that defect is also the largest single cause of (3).** There is a
second, genuinely independent cause of (3) in the pattern extractor. Both are
live in the current tree and both are demonstrated below rather than argued.

---

## What was measured

Two sources of evidence, and it matters which is which.

**Stored analyses** (`chords.sqlite3`, four songs, `beat_this@ismir24-final0` +
`btc@ismir19-large-voca`). These rows are from **2026-08-04 ~02:00 PDT**, which
is *before* the whole §20 theory layer landed (`5350a45`, 19:29 the same day).
So their **section and pattern counts are stale** and must be re-measured on a
fresh run. What is *not* stale is the anchor timing: `axis.downbeats_ms` passes
the tracker's downbeats through verbatim, then and now, and nothing between the
tracker and the sidecar repairs them.

**Live runs against the current tree**, driving `song_model.build` directly with
synthetic grids. This is where the causal claims come from.

### The anchors, as shipped

Anchor gaps, expressed in beats of the song's own modal bar:

| song | gap histogram (beats) | bars exactly one bar long |
|---|---|---|
| Let It Be | `2.0:14  2.5:1  3.5:1  4.0:59  4.5:2` | 59/77 — **77%** |
| Assima | `1.0:3  2.0:12  4.0:83` | 83/98 — **85%** |
| Born To Be Wild | `1.0:4  3.0:4  4.0:119  4.5:2` | 119/129 — **92%** |
| Anti Nowhere League – So What | `1.0:37  2.0:4  3.0:30  3.5:1  4.0:104  5.0:2  7.0:1` | 104/179 — **58%** |

Two things to read off this.

**The anomalies are mostly *short*, not long.** The client reported 2.60× and
2.17× gaps — missing downbeats. Locally the dominant failure is the opposite:
half-bars and one-beat "bars", i.e. **spurious** downbeats. So What's signature
is unmistakable — 37 one-beat gaps and 30 three-beat gaps, which is one extra
downbeat fired one beat into a real bar, thirty-odd times. Both directions are
the same defect: nothing checks.

**The implied local tempo swings by an octave.** Let It Be reports 70 bpm and
its anchors imply a local tempo of up to 144.6; Assima reports 86 and implies up
to 352.9. That *is* complaint (2). The payload carries one tempo — it always
did — but the client reads its cursor speed off the anchors, and the anchors say
the song doubles.

---

## Finding 1 — nothing between the tracker and the sidecar validates a bar

`beat_this` emits `(beats, downbeats)`. From there:

- **`adapters/beat_this_tracker.py`** passes both through untouched. It *counts*
  beats per bar in `_meter()` and reports the disagreement as
  `meter_agreement` — so the information exists — but it repairs nothing.
- **`meter.reconcile`** corrects the downbeat *phase* (§20.2's vote) and,
  optionally, the tempo *octave*. Neither touches bar **length**. A grid whose
  bars are 4,4,1,3,4,4 is rotated and rescaled as faithfully as a clean one.
- **`axis.build_axis`** then does this:

  ```python
  for start, end in zip(downbeats, downbeats[1:]):
      times.extend(_bar_beats_between(tracked, start, end, bar_beats))
  ```

  **Every consecutive pair of downbeats is defined to be exactly one bar.** If
  the pair is half a bar apart, `_bar_beats_between` finds the wrong number of
  inner beats and resamples: `bar_beats` chart beats spread evenly across half a
  bar's duration.

That resampling is the module's deliberate concession to §13.2's single-meter
requirement, and the docstring defends it — correctly — for *genuinely* irregular
bars ("Here Comes The Sun has 11/8 and 15/8 bars inside a 4/4 song"). The defect
is that the same code path silently absorbs **tracker error**, which is one to
two orders of magnitude more common than real metric irregularity.

Live repro against the current tree — a perfect 120 bpm, 16-bar 4/4 grid, with
one downbeat corrupted:

```
clean (16 bars)          anchor gaps all 2000 ms; implied 120 bpm throughout
one SPURIOUS downbeat    gaps [..., 1000, 1000, ...]; implied 240 bpm for 8 beats
one DROPPED downbeat     gaps [..., 4000, ...];       implied  60 bpm for 4 beats
```

That is complaints (1) and (2), reproduced from a single bad downbeat, with no
audio involved.

## Finding 2 — one bad downbeat costs the song its form, and therefore its patterns

This is the bridge to complaint (3), and it is the finding I did not expect to
be this sharp.

A spurious downbeat **adds a bar to the chart**; a dropped one **removes one**.
Every bar after that point is shifted by one against the music. `form._layout`
searches for the block phase that exposes the most repetition — but it searches
for **one global phase**. A phase that changes mid-song cannot be fitted, so
block similarity collapses, `cluster` puts nearly every block in its own group,
and `model._patterns` emits one pattern per group.

Current tree, 32 bars of `G D Em C` played eight times, with a clean D-DU-UD-U
strum on every bar:

| grid | bars | sections | groups | section repeats |
|---|---|---|---|---|
| clean | 32 | 1 | 1 | `[8]` |
| one spurious downbeat at bar 10 | 33 | **3** | **3** | `[1, 5, 1]` |
| one dropped downbeat at bar 10 | 31 | **4** | **4** | `[2, 1, 4, 1]` |

One corrupted downbeat in thirty-two bars turns a single eight-times-repeating
section into three or four unrelated ones. At the **15–20% bad-bar rate actually
measured**, the form dissolves completely — which is exactly what the stored
songs show: Let It Be as 17 sections, all `repeats: 1`, with 17 patterns;
Assima as 23 and 23.

The user's phrasing is precise and worth keeping as the acceptance criterion:
**if a pattern doesn't repeat, it isn't a pattern.** Today a non-repeating
"pattern" is the *expected* output of a broken bar grid, and nothing anywhere
objects.

The corroborating fingerprint is in Let It Be's own chart. Bars 0–3 are
`C/G | Am/F | C/G | F/C`; bars 16–19 are `F/C | C/G | Am/F | C/G` — the same
four-bar cycle, rotated by one bar. The harmony is right (BTC did its job); the
bar grid slipped a bar somewhere in between.

## Finding 3 — confidence is structurally unable to see any of this

`beat_this_tracker.track`:

```python
confidence = max(0.0, min(1.0, 0.5 * regularity + 0.5 * meter_agreement))
```

`regularity` is computed from **beat** intervals, which are reliable.
`meter_agreement` is the bar-length half. So a song with a flawless pulse and
*zero* bar agreement scores `0.5 * 1.0 + 0.5 * 0.0 = 0.5`, and
`pipeline.assemble` tests `grid.confidence < settings.confidence_floor` with a
floor of **0.5**. `0.5 < 0.5` is false.

**A song whose bars are entirely wrong cannot be flagged low-confidence by this
path.** It ships with a sidecar.

It is worse than that, because `_meter()` discards its own worst evidence:

```python
if 1 < counted <= 13:
    counts.append(counted)
```

Single-beat bars are dropped from the sample entirely — so So What's 37
one-beat bars never entered the denominator, and it shipped at confidence
**0.842** with 42% of its bars malformed.

`lint_sync` cannot see it either. It checks that anchors increase strictly, land
on bar boundaries, stay inside the duration, and cover the song. **It never
checks that a bar is about as long as the other bars.** This is the same class
of blindness `axis.py` and `consensus.py` already document: every existing check
compares the song against *itself*, and a chart that is uniformly wrong about
its own bar lengths is perfectly self-consistent.

## Finding 4 — the pattern extractor has no musical prior (independent of the above)

This one survives a perfect beat grid, so it must be fixed separately.

`strumming.extract` keeps **every** grid cell whose bar-support clears
`SUPPORT_THRESHOLD = 0.5`, with no cap and no contrast requirement. On a full
mix, `LibrosaOnsetDetector` fires on the drum kit, and a hi-hat playing 8ths
plus a kick/snare pattern is present in *every* bar — so more than half the bars
carry an onset near nearly every 16th cell, and nearly every cell is kept.

Current tree, clean 120 bpm grid, guitar playing a 6-stroke D-DU-UD-U with human
jitter:

```
guitar only   6 strokes  DDUUDU            @ [0.0, 1.0, 1.5, 2.5, 3.0, 3.5]
with drums   15 strokes  DUDUDUDUDDUDUDU   @ [0.0, 0.25, 0.5, ... 3.5, 3.75]
```

Fifteen of sixteen 16th cells. That matches the stored songs exactly (Let It Be's
patterns run 3–14 strokes, several at 12–14) and it is the literal opposite of
what `strumming.py`'s own docstring promises: *"a 16-onset syncopated
transcription of a strummed acoustic is less playable — and less true to the
song — than the D-DU-UD-U everyone actually plays."* The intent is documented;
the thresholds do not deliver it.

`choose_subdivision` is not the culprit and should not be blamed for it — it
scores by mean quantization error specifically so a stray 16th cannot carry the
vote, and that reasoning is sound. The gap is downstream: **once the grid is
chosen, nothing limits how many of its cells become strokes.**

## Finding 5 — patterns are never consolidated at song level

`model._patterns` emits one `ExtractedPattern` **per repeat group**, and:

- `group.is_repeat` exists and is **never consulted**. A group with one
  occurrence gets its own extracted pattern on the strength of one section's
  onsets.
- `_rename_shared_grooves` collapses patterns only when their **content-addressed
  ids are byte-identical**. Two grooves that differ by one 16th are two patterns.

So there is no mechanism anywhere that says "this song has one strum." Measured
on the stored songs, clustering the emitted patterns by Jaccard over their cell
sets:

| song | patterns | @0.5 | @0.6 | @0.7 | mean pairwise similarity |
|---|---|---|---|---|---|
| Let It Be | 17 | **3** | 5 | 8 | 0.52 |
| Assima | 23 | **6** | 9 | 15 | 0.47 |
| Born To Be Wild | 7 | **2** | 3 | 4 | 0.52 |

The patterns are already mostly the same groove measured slightly differently.
Nothing in the pipeline notices.

---

## The plan

Three phases. Phase 1 is the one that matters most and should land first; Phase 2
is independently useful and can proceed in parallel; Phase 3 is what stops all of
this from silently regressing.

### Phase 1 — repair the downbeat sequence before anything consumes it

**New module `app/analysis/downbeats.py`, called from `meter.reconcile` before
the phase vote** (the vote assumes the bars are right, so it must run after the
repair), and therefore before `build_axis` — so the chart, the bars and the
anchors all move together and there is still exactly one origin.

The key change in framing: **trust the tracker's beats, not its downbeats.** The
beat sequence is reliable (regularity is near 1.0 on every song measured); the
downbeat sequence is what fails. And every anomaly measured lands on a real beat
(gaps are integer numbers of beats: 1.0, 2.0, 3.0), so the repair can be stated
as *choosing which beats are bar starts* rather than as moving times around.

1. Map each tracker downbeat to a beat index (`meter._nearest_beat` already does
   this).
2. Estimate the modal bar length in **beats** — mode, not median, because a heavy
   tail of half-bars drags a median.
3. Walk the downbeats in beat-index space:
   - a downbeat less than ~0.65 bars after the last accepted one is **spurious** —
     drop it;
   - a gap of ≈ *n* bars (*n* ≥ 2, within ~25%) is **n−1 missing downbeats** —
     insert them on the tracker's own intervening beats.
4. Report `dropped` / `inserted` / final irregular-bar count.

Prototyped against the stored anchors — and note this is the *weak* version,
working from millisecond gaps only, because the stored rows don't carry the beat
list; the real implementation has beat indices and will do better:

| song | before | after | edits |
|---|---|---|---|
| Let It Be | 81% | **100%** | −8 spurious |
| Assima | 85% | **97%** | −10 spurious |
| Born To Be Wild | 94% | **98%** | −4 spurious |
| So What | 59% | **87%** | −38 spurious, +1 missing |

**Keep the escape hatch that `axis.py` and `meter.py` were right to build.** Real
irregular bars exist. So the repair must be *bounded and reported*, not silent:

- If the share of bars disagreeing with the mode exceeds a ceiling (~35% —
  So What at 42% is precisely the case to watch), the song may genuinely not be
  in that meter. Do not force it: flag low-confidence and let §13.3 withhold the
  sidecar, which is the existing honest-degradation path.
- Everything the repair does goes into `TheoryReport` (below) so it is provenance
  rather than a silent rewrite — the same rule `consensus` and `vocabulary`
  already follow.

**Also fix, in the same phase** (both are small and both are load-bearing):

- `beat_this_tracker._meter()` — stop discarding single-beat bars from the
  agreement sample. They are the strongest evidence of a broken grid and they are
  currently the only evidence thrown away.
- `beat_this_tracker.track()` — `confidence` must be able to reach below the 0.5
  floor on bar evidence alone. Multiplying the two halves rather than averaging
  them (`regularity * meter_agreement`) makes a song with perfect beats and no
  bars score 0, which is the truth. This will move confidence on real songs, so
  it needs a benchmark pass before it ships.

**Expected effect on complaint (3):** most of it. Restoring the bar grid restores
the global phase, which restores clustering, which collapses 17 sections back
toward the handful the song actually has — and pooled extraction over a *real*
repeat group has 4–16× the evidence, which is the difference between
`SUPPORT_THRESHOLD` meaning something and it being a coin toss.

### Phase 2 — make patterns musical, and make them few

Independent of Phase 1. Four changes, in increasing order of ambition.

**2a. A stroke budget with a contrast requirement.** In `strumming.extract`,
after cells are scored: keep a cell only if its support clears the threshold
**and** stands above the bar's mean cell support by a margin; then cap the result
at ~2 strokes per beat (8 in 4/4). A groove is defined by *contrast* between
struck and unstruck cells — "an onset near everything" carries no pattern, and
should read as the 8th-note or quarter-note skeleton underneath, not as 15
strokes. This alone turns the drum case above from 15 strokes back toward 6.

**2b. A rhythm vocabulary — the direct answer to "more musical".** The chord side
already has this shape in `vocabulary.py`: measure, then snap to what the song
actually plays. Do the same for rhythm against a small library of idiomatic
strums:

```
D  D  D  D            quarters                    0, 1, 2, 3
D  -  D  U  -  U D U  the campfire pattern        0, 1, 1.5, 2.5, 3, 3.5
D  U  D  U            eighths                     0, .5, 1, 1.5, 2, 2.5, 3, 3.5
D  -  -  -  D  -  -   half notes                  0, 2
D  D  U  -  U  D  U   folk variant                0, 1, 1.5, 2.5, 3, 3.5
                      (plus the 3/4 and 6/8 forms)
```

Snap the extracted cell set to the nearest library entry when the distance is
below a threshold; otherwise keep the extraction verbatim. Tag the result
(`snapped-to-idiom`) exactly as `directions-by-convention` is tagged, so nobody
downstream mistakes the snap for a measurement. This is the same
measure-then-snap discipline as `vocabulary.SNAP_TO`, and it is why that
precedent is worth following rather than inventing a new one.

**2c. Song-level consolidation.** After per-group extraction, cluster the
patterns by Jaccard over their cell sets and keep one representative per cluster
(weighted by how many bars stand behind each). Content-addressing already
collapses *identical* grooves; this collapses *near-identical* ones, which is
what real extraction actually produces. Measured target: 17 → 3, 23 → 6, 7 → 2.

**2d. "If it doesn't repeat, it's not a pattern" — as a rule.** A repeat group
with a single occurrence does not earn its own extracted pattern; it inherits the
song's dominant one. `RepeatGroup.is_repeat` already exists for exactly this
question and is currently unused. Same for a group whose extraction fell back:
inherit the song's dominant groove rather than emit a second, different fallback.

Target for the deliverable: **one to three patterns per song**, each backed by a
group that actually repeats.

### Phase 3 — make the failure visible, so it cannot come back

**A new `lint_sync` check.** Consecutive anchor gaps must sit within a tolerance
of the song's own modal gap. Two design notes that matter:

- Compare against the **modal anchor gap**, not against the payload tempo. The
  payload tempo is derived from beat intervals and is right even when the bars
  are wrong, so comparing to it would pass the exact songs that are broken.
- Withhold on a **share**, not on a single bar. Real recordings have rubato and
  real songs have the occasional inserted bar; a single-bar trigger would
  withhold sidecars from songs that are fine. Something like ">10% of bars off
  modal" is the shape, calibrated against the corpus.

This is the check that would have caught every case in this document.

**Two new `TheoryReport` fields**, carried on the sidecar like the rest of §20's
provenance:

- `irregularBars` — how many bars disagreed with the mode after repair;
- `downbeatsRepaired` — dropped/inserted counts.

Then `bench/run_bench.py` and `scripts/seed_catalog.py` can *grade* the beat map
instead of grading only chords, which is the only way this stays fixed.

**Fixtures with broken grids.** Every beat grid in the test suite is
`beats[::4]` — `conftest.known_downbeats`, `test_model`, `test_meter`,
`test_pipeline`, all of them. A perfectly regular downbeat sequence appears in
414 green tests and an irregular one appears in none, which is exactly why none
of this was visible. Add three: a spurious downbeat, a dropped one, and a
*genuine* irregular bar that the repair must **not** flatten.

---

## Ordering, and who does what

Phase 1 first — it is the root cause of two complaints and most of the third, and
Phase 2's thresholds should be tuned against a *correct* bar grid rather than
re-tuned afterwards. Phase 2a and 2c are cheap and independently valuable and can
land alongside. Phase 3's lint check should land with Phase 1 so the repair's
effect is measurable immediately.

**Mine:** all of it — the repair module, the tracker fixes, the pattern work, the
lint check, the fixtures, and the benchmark pass that says what each change
bought.

**Yours:** one decision and one run. The decision is whether the confidence
change in Phase 1 is allowed to reduce the number of songs that ship with a
sidecar — a song with a genuinely unreadable bar grid *should* lose video sync,
but that trades a wrong sync for no sync and it is a product call, not a code
one. The run is re-analyzing the four catalog songs after Phase 1 so the
before/after is measured on real audio rather than on synthetic grids; the stored
rows in `chords.sqlite3` predate the §20 layer entirely and cannot serve as the
baseline.

---

## One thing to re-check, not yet a finding

All four stored analyses have `analysis: null` and every section named `Part N`
with `kind: custom`, which is the no-energy-hint path in `form._assign_labels`.
That is fully explained by the rows predating the energy probe (`5350a45`,
committed ~17 hours after the last of them), so it is **not** evidence that the
structure probe is broken today. But it does mean nobody has confirmed the probe
runs in the worker — and if it doesn't, `pipeline.analyze` swallows the failure
by design and every song silently loses verse/chorus naming. Worth one look at a
fresh run's logs.
