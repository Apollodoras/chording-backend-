# The analysis system — audit, and the plan to fix it

An external review of the chord-analysis half of the service (`app/analysis/**`
plus the scripts around it, ~9,200 LOC), traced back from four symptoms the owner
reported to the code that produces them. Published 2026-08-18; recorded here in
full, with a verdict of its own against every finding.

The four symptoms, in the owner's words:

1. **sevenths in almost every song** — `Bm` and `Bm7` side by side in one chart;
2. **major and minor in the same key** — `C` and `Cm` both functioning as tonic;
3. **unlikely alterations** — `A` and `A#` in one chart;
4. **structure that doesn't follow norms** — several distinct "Verses" where the
   song has one verse played three times.

The architecture came out sound, and the review says so: raw frame-level chords
are quantized onto a repaired beat grid, a key is detected, the vocabulary is
consolidated, repeating sections are found, and the repetitions vote each other
into consistency. What it found is a different failure class from the one
`PIPELINE-AUDIT.md` found: **the safety gates are tuned so conservatively that
they cannot fix the errors they were built for**, plus two genuine inversions.

## How this document differs from the review

Every finding below is recorded as written, and then answered. The reviewer did
not have the measurement history — `bench/lab.py`, `bench/run_bench.py
--calibration/--noise` and the memory of what each constant cost when it was
moved — so a handful of the recommendations are for behaviour that was already
tried and is already known to lose accuracy. Those are marked **Measured
otherwise** with the number, and they are not implemented; implementing them
would be trading a measured result for a plausible argument.

The rest were implemented and measured. The ruler is `bench/lab.py grade` over
the ten-song chart corpus (`bench/reference/*.chart`), which regrades in about a
second from cached engine features, plus `bench/run_bench.py --noise` for the
two-sided count that the corpus is too small to resolve.

**The numbers**, on the ten-song chart corpus (`bench/lab.py grade`), which is
the row every earlier measurement in this repo was taken on:

| | root | triad | form | tonics | distinct chords | tests |
|---|---|---|---|---|---|---|
| before (`6b6a0d7`) | 0.853 | 0.846 | 0.688 | 9/10 | 71 | 717 |
| after | **0.854** | **0.849** | **0.755** | 9/10 | **61** | 745 |

and on the `easy` tier, which is the one a beginner is shown:

| | root | triad |
|---|---|---|
| before | 0.761 | 0.732 |
| after | **0.789** | **0.772** |

Read the chord columns and the vocabulary count together, because that is where
the reported symptoms live. Root barely moves — the roots were mostly right —
while ten spurious chords leave the corpus's charts: Someone Like You goes from
eleven distinct chords against a reference of five to six, Don't Stop Believin'
from ten to eight, Smooth Criminal from thirteen to ten. Form is the other
visible column, +0.067.

An eleventh song, **Mary Jane's Last Dance**, was added to the corpus during this
work (`bench/songbook.json`). It is the only recording in the book that is not at
concert pitch — 48 cents sharp — and it is what settled F1. It is reported
separately rather than folded into the mean above, so the ten-song numbers stay
comparable with everything already written down.

## Status

| Phase | What | State |
|---|---|---|
| 0 | The ruler — per-layer attribution (F32) | **Done** — `bench/lab.py layers` |
| 1 | The engine's decoder (F1, F2, F3, F4) | **Done** — F1 and F4 partly, and the measurements say why |
| 2 | Vocabulary — "sevenths everywhere" (F5, F6, F7, F8) | **Done** — F6/F7 answered by §20.10 instead |
| 3 | Key — "major and minor in one key" (F9, F10, F11, F12, F13) | **Done** |
| 4 | Consensus & canon (F14, F15, F16, F17) | **Done** — F16/F17 measured and declined |
| 5 | Form — "several verses" (F18, F19, F20, F21, F22, F23) | **Done** — F19 measured and declined |
| 6 | Meter, pipeline, hygiene (F24–F31, F33, F34) | **Done** — F34 blocked by the container |

Tests: 717 → **745**.

### What the layers are each worth now

`python bench/lab.py layers` — the per-song, per-posture attribution the audit
asked for in F32, which is what makes every row below a measured delta rather
than an argument:

```
posture                  root  triad   form  tonics   Δroot
engine only             0.809  0.789  0.702    9/10
+ vocabulary  §20.8     0.809  0.789  0.702    9/10  +0.000
+ key audit   §20.10    0.809  0.793  0.713    9/10  +0.000
+ consensus   §20.4     0.821  0.812  0.713    9/10  +0.012
+ belief      §20.9     0.821  0.812  0.713    9/10  +0.000
+ form        §21       0.854  0.849  0.755    9/10  +0.033
```

The engine's own chart is 0.809 and the whole theory layer is worth +0.045 on top
of it. That is the honest shape of this system and it answers the question the
audit opened with: **most of the remaining error is the engine's**, and the
layers above it are worth what they are worth. The two rows that read +0.000 are
not idle — §20.8 and §20.10 move `triad` and the chord count rather than the
root, because what they fix is a chord's *colour*.

---

## The pipeline, as reviewed — and as it stands now

```
as reviewed
source → tuning → beats → chords (BTC, argmax) → quantize/merge → key
       → vocabulary → form → consensus → canon ×4 → patterns → compile + lint + sync

now
source → tuning → beats → chords (BTC, Viterbi over overlapped windows)
       → quantize/merge → key (pre-fill) → vocabulary → KEY AUDIT §20.10
       → form → consensus (per-occurrence) → canon ×4 → patterns
       → compile (per-tier lint) + sync
```

One new stage, `keyaudit.py`. Everything else in the diff is a rule inside a
stage that was already there.

## Symptom map

| Symptom | Findings |
|---|---|
| "7ths in almost every song — Bm and Bm7 side by side" | F2 (no decoder — flicker arrives raw), F6 (`snap` can never fire), F7 (`SNAP_TO` is one-directional) |
| "major and minor in the same key — C and Cm" | F9 (no key-consistency audit), F6, F10 (wrong-quality chords still earn key credit) |
| "unlikely alterations — A and A# in one chart" | F1 (tuning gate), F8 (`absorb_islands` can't move a root), F14 (the vote aborts on gross outliers), F11 (no modulation) |
| "several distinct verses" | F18 (`period` prefers the smallest divisor), F19 (one fixed block size), F20 (near-duplicate groups never merge; everything falls through to "verse") |

This is the **review's** map, kept as written. In three cases out of four the
finding that actually fixed the symptom is not the one predicted here — see
[What actually answered each symptom](#what-actually-answered-each-symptom) at the
end, and the verdict under each finding.

---

## A · Signal & engine layer

### F1 — the tuning-correction gate is inverted · CRITICAL
`app/analysis/tuning.py:33` · `pipeline.py:102,117`

`Tuning.correction` returns `self if self.ambiguous else CONCERT_PITCH`.
"Ambiguous" means |deviation| ≥ 0.42 semitones — the near-quarter-tone zone where
librosa's estimate is least trustworthy, its sign able to flip. So a recording
reliably 25¢ sharp gets no correction (its energy sits between CQT bins, the
engine flickers between adjacent roots), while a near-quarter-tone recording gets
a correction whose wrong sign transposes every label by a semitone.

**Recommended:** always pass the estimated correction; keep `ambiguous` as a
confidence flag; for |dev| ≥ 0.42 analyze a short excerpt with both signs and keep
the higher-confidence one.

**Verdict: the gate is not inverted, and the recommended tie-break is
measurably harmful. Neither shipped; the measurement did.**

The gate is the shape it is because correcting *every* real bench track cost 2.5
points of chord accuracy (0.808 → 0.783) when it was measured on 2026-08-17. BTC
is trained on real records, which carry a spread of tens of cents, so nudging the
CQT grid inside that spread moves the input off the distribution the model
learned. The correction earns its place only where the uncorrected reading is
undecidable — which is what ≥ 0.42 means.

Mary Jane's Last Dance was added to the corpus to test the rest of the finding,
and it proves the *danger* the finding describes at the same time as it kills the
proposed fix:

| correction | root | mean chord confidence | transposition sweep |
|---|---|---|---|
| none (A440) | 0.112 | 0.695 | the whole chart intact **one semitone down** |
| +0.48 — measured, and what ships | **0.380** | 0.773 | at zero, not displaced |
| −0.52 — the other sign | **0.000** | 0.756 | displaced |

So the sign genuinely decides the whole chart. And the recommended tie-break —
ask the engine which correction it believes more — was implemented and measured:
it picks correctly on Mary Jane (whose measured sign was already right) and
**incorrectly on Smooth Criminal** (0.654 for the right sign against 0.757 for
the wrong one; the whole track rather than an excerpt does not rescue it,
0.781 against 0.819), taking that song from 0.335 root to **0.012**.

The mechanism is why no confidence-shaped rule will work: Smooth Criminal's
harmony is a bass line with no third sounding, so the model is guessing
throughout, and a grid shifted off the recording lets it guess *more decisively*.
Softmax confidence measures how peaked a posterior is, not how right it is, and
the two come apart exactly where the audio is ambiguous — which is the only place
this question is ever asked. All of it is recorded in `tuning.py`, where the next
person to have this idea will look.

### F2 — BTC output is a per-frame argmax with no decoder · CRITICAL
`app/analysis/adapters/btc.py:218-233`

The adapter sets `probs_out = True`, computes the full 170-class posterior, then
keeps only the argmax per frame and throws the distribution away. No median
filter, no HMM, no Viterbi — while the *fallback* chroma engine has one. Every
one-frame flicker becomes a span the theory layer must clean up afterwards.

**Recommended:** decode the posteriors. At minimum a Viterbi with a self-transition
prior; the stronger version pools posteriors between beats and decodes over beats
with a change penalty smaller at bar boundaries.

**Verdict: the frame-level decoder shipped. The beat-synchronous one — the
audit's stronger recommendation — is measurably worse.**

`btc._viterbi` decodes the full posterior sequence under a uniform stay-or-switch
transition (`_DECODE_CHORD_FRAMES = 21`, swept 8/15/21/30/45 to a broad flat
optimum), and each span now publishes the posterior of the label it actually
claims rather than the frame's argmax.

Worth, measured from cached posteriors so the transformer ran once: root 0.853 →
0.857, triad 0.846 → 0.849, spans 1624 → 1235, distinct chords 71 → 64. Smaller
than the span reduction suggests, and the reason is worth stating rather than
explaining away: `quantize` snaps every boundary to a beat and `drop_short`
removes anything under one beat, so most single-frame flicker was already being
absorbed downstream. What the decoder adds is the flicker long enough to survive
quantization — and a cleaner span list for every stage above it.

| decoder | root | triad | form | distinct chords |
|---|---|---|---|---|
| argmax (before) | 0.853 | 0.846 | 0.688 | 71 |
| median filter, width 9 | 0.858 | 0.850 | 0.688 | 70 |
| **frame Viterbi (ships)** | **0.857** | 0.849 | 0.688 | 64 |
| beat-sync, 4 beats/chord | 0.809 | 0.806 | 0.742 | 57 |
| beat-sync, 8 beats/chord | 0.799 | 0.796 | 0.772 | 55 |

Beat-synchronous decoding buys form and a smaller vocabulary by dragging the
chart toward fewer, longer chords, and pays five points of root for it. This
pipeline already has a layer whose job is to state the form (§21, `canon.py`) and
which does it without giving up chords.

### F3 — inference windows are non-overlapping 10-second chunks · MAJOR
`app/analysis/adapters/btc.py:212-227,247-262`

Features are split into 108-frame windows with hard boundaries; the transformer
never sees across a join, so chords straddling a boundary flicker there. The final
window is zero-padded, biasing its tail.

**Fixed.** `BtcEngine._posteriors` overlaps windows by a quarter and combines
them with a triangular weight, so no frame is decided by a window that only just
contains it; padded tail frames are dropped rather than blended in.

Measured: root and triad identical to three decimals, form 0.747 → **0.755**
(Viva La Vida 0.667 → 1.000, Wonderwall +0.016 root, Smooth Criminal −0.023 and
it is the corpus's known outlier). A third more forward passes, and free in
practice — the eleven-song feature build went 493 s → 481 s, because the CQT
rather than the transformer is where the time goes. The chord columns do not move
because `_features` already gave every block its neighbours' *audio*; only the
attention ever saw a boundary.

### F4 — confidence is raw softmax, treated as calibrated · MAJOR
`btc.py:224-227` · `vocabulary.py:42` · `consensus.py:17`

Span confidence is the mean per-frame max-softmax over 170 classes. The theory
layer applies 0.02-wide margins to it.

**Partly fixed, and the rest is a project rather than a fix.** The number is
now the posterior of the label the span *claims* rather than the frame's argmax —
the right quantity, and it differs wherever the decoder overrules a one-frame
peak. It is still not calibrated. Temperature scaling belongs in
`bench/run_bench.py --calibration`, where the labelled set already is, and it
cannot be judged on its own: every threshold in the theory layer was tuned
against this distribution, so recalibrating means re-sweeping them in the same
change. Stated in `btc.py` rather than half-done.

### F5 — the chroma fallback has a different, tiny vocabulary · NOTE
`app/analysis/adapters/chroma.py:8-13`

maj/min/7/min7 only — no dim, sus or maj7, so any threshold tuned on BTC
misbehaves on it.

**Documented at the table**, with what it implies: `SNAP_TO`'s rows for the
missing qualities are unreachable on this engine, `exactRatio` reads high because
nothing is ever reduced, and `keyaudit`'s conflicts can only ever be
major-against-minor. Every number in this repo is BTC's number.

## B · Vocabulary consolidation

### F6 — `snap()`'s gates are conjunctive and nearly unsatisfiable · CRITICAL
`app/analysis/vocabulary.py:117-125` (constants at 36-42)

A span is relabelled only if all of: near-miss, ≤ 2 occurrences, 6× mass
dominance, ≤ 15% share, believed less, and not away from the key. A song read
60/40 Bm/Bm7 satisfies none, so both spellings ship.

**Recommended:** replace the per-span gate with a **per-root quality election** —
duration-weighted dominant quality per root within a triad class, relabelling the
rest unless the minority is concentrated in one section.

**Verdict: the gates are not loosened — they are measured — and the symptom is
answered by a new layer instead (§20.10, F9).**

Each gate the finding lists is a measurement, not a guess. `MAX_OCCASIONS = 2`
settles the hardest case in the module: "In My Life" plays A for seventy beats and
A7 for nine, in four brief doubtful passes, and those nine beats *are the song*
— every duration-based measure calls them noise and only counting occasions
separates them from "Something"'s two spurious G7s. At ≤ 2 the module corrects six
spans and damages one; with no limit it corrects seven and damages **five**.
`MASS_DOMINANCE` and `MINORITY_SHARE` are the same argument in duration.

What is true is the audit's diagnosis of the *gap*: a systematic mishearing — BTC
reading `C#m` as `C#` every time that passage comes round — satisfies none of
them, and it is exactly the case the owner reported. But mass cannot answer it,
because the mishearing inflates its own mass. The evidence that can is the **key**,
which did not come from counting engine output at all. See F9 — that layer removes
the `F#`/`F#m`, `C#`/`C#m`, `G#`/`G#m` and `E7`/`E` pairs the finding is about.

### F7 — `SNAP_TO` is asymmetric, 7th → triad only · MAJOR
`app/analysis/vocabulary.py:25-32`

When the seventh is the *majority* reading, the minority triad spans can never be
folded into it, guaranteeing the mixed spelling.

**Verdict: not made bidirectional. The asymmetry is the measurement.**

`SNAP_TO` is a *measured* table, not a reading of `is_near_miss`
(`bench/run_bench.py --calibration`, nine real tracks, beats-weighted): a reported
`dominant7` is the plain major 60% of the time and a real seventh 31%, so
flattening a doubtful one is a bet at 2:1 on. There is no measurement anywhere
saying the reverse move pays, and the repo already records that a *generic*
near-miss snapping rule — which is what symmetry amounts to — **loses** accuracy:
near-miss says two chords are confusable, and says nothing about which direction
the engine errs in, which is the only fact that decides whether an edit pays.

The product question the finding raises underneath — should a chart prefer plain
triads unless the seventh is strongly evidenced — is answered, and the answer is
yes: that *is* what the asymmetric table encodes, and it is why the corpus's chord
count fell from 71 to 61 without the table changing.

### F8 — `absorb_islands` can never fix a wrong-root island · CRITICAL
`app/analysis/vocabulary.py:147` · `harmony.py:73-80`

The absorber requires `island.root_pc == before.root_pc`. The classic engine
error — a short A# between two long A spans — is skipped, and cannot be caught
anywhere else either, because `harmony.similarity(A, A#)` is 0 and every
near-miss gate in the codebase is closed to semitone errors by construction.

**Fixed** — `vocabulary.absorb_semitone_islands`, a rule of its own rather than
a relaxation of the existing one. Flanks identical, island a semitone away, brief
relative to them, believed less, and neither side a quality `NEVER_SNAPPED`
protects (that last one is "Michelle" arriving by a second road: an augmented
triad is one set of notes under three names, so "a semitone from it" is not a
well-formed statement).

The finding's diagnosis is exactly right and worth restating: `harmony.similarity(A,
A#)` is **0**, because the two triads share no pitch class, so every near-miss gate
in the codebase reads a semitone slip as the furthest thing from a mishearing
there is. The harmonic test therefore had to be *replaced* rather than relaxed.

Measured: **inert on this corpus.** Of 209 identically-flanked triples in the
eleven songs, three are a semitone away and none is brief enough to qualify. The
rule is right and the corpus does not contain its case — which is worth saying
plainly rather than claiming a win. `tests/test_vocabulary.py` carries it.

### F9 — no whole-song key-consistency audit · CRITICAL
gap — `diatonic_fit` appears only as a tiebreaker

Nothing ever checks the finished chart against the detected key. A chart holding
both C and Cm as functional tonics passes every lint.

**Recommended:** a final harmony audit with a small prior table of common
borrowings per mode; repair or flag out-of-key chords that have an in-key
parallel; at minimum surface "key conflicts: C vs Cm" in the `TheoryReport`.

**Fixed, and it is the layer that answers two of the four symptoms** —
`app/analysis/keyaudit.py` (§20.10), on by default, `CHORDS_THEORY_KEY_AUDIT=0`
to turn it off.

Three-valued, exactly as recommended: **diatonic** (from `keyfinder.degrees`, so
there is one scale table and not two), **borrowed** (a small named table — I7,
bVII, iv, bVI/bIII, secondary dominants, vii°, and the minor-key equivalents), and
**foreign**. Only a *conflict* is acted on: one root the song reads two ways, one
at home in the key and one not, differing in colour alone.

Two gates were added by measurement rather than by design:

- **secondary dominants are `DOMINANT7` only, never the bare triad.** With bare
  majors admitted, `C#` in E major reads as V/ii and is protected — which
  protects precisely the chords the layer was reported for.
- **the foreign reading may hold at most `MINORITY_SHARE = 0.20`** of the root's
  evidence. Above that the song is telling us the *key* is wrong, not the chord.
  Creep is the case: it is in G, plays G B C Cm, the key finder returns E minor,
  and from there the real `Cm` is foreign and holds 35.5% of that root. Without
  this gate the audit deletes the chord the song is known for. The corpus
  separates cleanly on this one number — 35.5% and 23.5% for the two chords that
  must survive, 14.1% / 11.1% / 7.5% for the three that must not.

Note what does **not** work as a gate, since it is the obvious one:
`DetectedKey.confidence`. Creep's *wrong* key is the most confident reading in the
whole corpus (0.305, against 0.004–0.070 for the nine that are right), because
that number is a margin over the runner-up and a wrong answer can win by a mile.

Every conflict is reported whether or not it is repaired —
`TheoryReport.keyConflicts`, as `"C#(VI) vs C#m(vi)"`, with the degrees named
because the chord pair alone does not say which one the key expects. A song with
conflicts and no resolutions is one the audit found something in and declined to
touch, which is the state an operator most needs to see and the one that was
invisible before.

## C · Key detection

### F10 — wrong-quality chords still earn 45% key credit · MAJOR
`app/analysis/keyfinder.py:25,122-138`

`_ROOT_ONLY = 0.45` for a chord whose root is diatonic but whose quality
contradicts the mode, against mode priors of 0.003–0.006.

**Fixed, and measurably inert.** `_WRONG_THIRD = 0.20` now separates "the
recognizer heard a seventh, and the triad under it still agrees with the key" from
"the triad contradicts the mode", which the flat `_ROOT_ONLY = 0.45` scored alike.

Swept at 0.45 / 0.30 / 0.20 / 0.10 / 0.00: **identical on every column**, 9/10
tonics throughout. The corpus does not contain a song whose key is decided by a
wrong third. Kept because the two disagreements are different facts and scoring
them alike is a bug waiting for its song, and reported as unmeasured rather than
as a win.

### F11 — a single global key; modulations are invisible · MAJOR
`keyfinder.py:83-119` · `model.py:81-113`

The truck-driver modulation is one of the commonest structures in this
repertoire, and today it produces a chart mixing two transpositions.

**Fixed as far as the container allows: detected and reported, never acted on.**
`keyfinder.track` reads the key over 32-bar windows and `TheoryReport.modulations`
publishes the changes as lines a person can read (`"bar 96: E major -> F major"`).

One design note that cost a false positive on the first run: windows are compared
by **collection**, not by tonic. I'm Yours reported `"bar 64: B major -> G# minor"`
— one set of seven notes read two ways, which is the relative-key argument the
first half of `keyfinder` is entirely about, and nothing a listener would call a
modulation. A modulation moves the *notes*. With that fixed, the eleven-song
corpus reports no modulations at all, and a synthetic final-chorus-up-a-tone is
detected (`tests/test_keyfinder.py`).

Acting on it — windowed Viterbi over keys, local diatonic gates — is deliberately
not done: the container carries one key, so a repair driven by a second one would
be a change nothing on the wire could express.

### F12 — the key is detected after gaps are back-filled · MINOR
`model.py:79-81` · `postprocess.py:86-103`

`hold_through_gaps` stretches chords over silence before `detect_key` runs, so
intros and breaks inflate whatever chord preceded them.

**Fixed** — `postprocess.process(..., fill=False)` stops before the hold step,
and `model.build` detects the seed key on that. Same quantization, same merging,
same floor; the silences are simply still silent. A forty-second instrumental
break no longer triples the weight of whatever chord preceded it.

Measure-neutral on the corpus (9/10 tonics either way), which is expected — the
corpus's songs do not have long harmony-free stretches, and the one that does
(Smooth Criminal, 51 seconds of dialogue) has a wrong key for a different reason.

### F13 — duplicated flat-spelling logic · NOTE
`keyfinder.py:141-144` duplicates `app/chords.py:91-99`

**Fixed** — the key-signature table lives once, in `chords.prefers_flats_for`,
and `keyfinder._prefers_flat_spelling` delegates to it. Two tables that have to
agree and are not the same table only ever agree until one of them is edited.

## D · Consensus & canon

### F14 — voting aborts on gross outliers instead of fixing them · CRITICAL
`app/analysis/consensus.py:109-117,146-161`

If any losing bar is less than 50% similar to the winner, `_vote` returns `None`
— abandoning the whole slot, including near-miss losers it could have fixed. An
A# bar among three A verses has similarity 0, so the more wrong a bar is, the
more protected it is.

**Recommended:** decide per loser, not per slot.

**Fixed** — both gates are now asked of each losing occurrence rather than of
the slot, and the finding's description of the inversion is exact: an `A#` bar
among three `A` verses scores similarity **0** against the winner, so it failed
gate 2 outright, aborted the whole vote, and the near-miss losers beside it
shipped too. The more wrong one occurrence was, the more protection it bought for
every other.

**No new licence** — every occurrence that is rewritten still clears both gates
individually, exactly as before. What changed is only that failing them no longer
speaks for anybody else. A slot where some occurrences were corrected and one held
its own is now reported as **contested and rewritten**, which is a state the
report could not previously express.

Measured on `bench/run_bench.py --noise`, which is where a change this size can
be resolved at all (the corpus cannot): the vote's two-sided count goes from
`fixed` 0.038 / `broke` 0.007 to **`fixed` 0.060 / `broke` 0.010**, delta +0.003 →
+0.007. More than half again as many corrections for a third more damage, which is
the shape a rule has to have before it is worth keeping. And
`bench/run_bench.py --theory` still reports +0.000 on every ground-truth track, so
the no-op-on-perfect-input property is intact.

The audit's further suggestion — overwrite a gross outlier outright when the
majority is strong — was **not** taken. That is new licence, it is the exact move
`axis.py` records as having cost this service 23 points of delivered accuracy
behind three hundred green tests, and the corpus is too small to license it.

### F15 — `settle_to_bars` flattens genuine split bars everywhere · MAJOR
`app/analysis/canon.py:190-227`

One global test flips the whole song into bar rhythm, and then every bar where one
chord holds ≥ 75% is flattened — including the cadence bar with a real IV–V split.

**Fixed** — `canon.settle_to_bars` takes the repeat groups and leaves a split
alone when most of that slot's sibling occurrences are also split. The finding's
reasoning is right and is what the code now says: one bar cannot tell an
anticipation from a cadence, because `| IV V |` at the end of a phrase reads
exactly like `| IV |` with the next chord pushed early. The other passes of the
same slot can.

Measure-neutral on the corpus — no slot there is a corroborated split that was
being flattened — and pinned by a test that fails without it.

### F16 — canon can synthesize hybrid bars no occurrence played · MINOR
`canon.py:90-150`

**Measured and declined.** Voting per beat *can* assemble a bar no occurrence
played; every chord in it is one some occurrence played in that bar, but the
sequence can be new. Counted over the corpus: **3 slots out of 1744**, 0.2%.

Constraining the output to a whole candidate bar would give up the property §21
was chosen for — that occurrences disagreeing about *where* a change falls are
settled beat by beat rather than by a plurality of composite objects — which is
where its +0.049 root came from. Recorded in `canon._agreed`.

### F17 — four canon rounds × re-detected form = hard-to-verify state · MINOR
`model.py:118-145`

**Declined, on the measurement §21 already carries.** Against a single round,
running to a fixed point is form +0.014 and root −0.004, and the mechanism is in
`canon.py`: each round re-finds the form on the bars it just rewrote, so after one
round the occurrences that had just been made identical are two readings again.
Wonderwall is the case.

The finding's premise — that this is hard to verify — is fair, and the loop is
where F20's "groups are never re-merged" turns out to be false (see F20). Recorded
on `model.CANON_ROUNDS`.

## E · Form & section labeling

### F18 — `period()` prefers the smallest divisor · CRITICAL
`app/analysis/form.py:101-121`, `PERIOD_MARGIN = 0.05`

Pop's strong 4-bar inner loop puts lag-4 within 5% of lag-8 even when the section
is 8 or 16 bars, so the song is chunked into 4-bar blocks and an 8-bar verse whose
halves differ becomes alternating groups.

**Fixed, in the opposite direction to the one recommended, and the corpus is
emphatic about it.**

Preferring the *longest* candidate within the margin is a **no-op on all ten
songs**: the divisibility test already there prevents the case the finding
describes, and `folded()` (triad-level comparison) prevents the wobble that used
to cause it. Traced: Creep scores 0.176 at lag 4 against 0.872 at lag 8 — lag 4 is
nowhere near winning.

Widening the margin the *other* way is what pays. Swept (root / form): 0.05 →
0.857 / 0.713, a flat band from 0.08 to 0.15 at ~0.854 / **0.747**, 0.18 → 0.853 /
0.732. `PERIOD_MARGIN` is now **0.12**, the middle of the band.

Three and a half points of form for three thousandths of root, and the trade is
the right way round for what the constant decides: the shorter unit is what lets
`_sections_from` collapse a repeat into `repeats`, so a wider margin is not
"accept a worse period", it is "prefer the compact statement of the same period
more often" — which in this repertoire is most songs.

### F19 — one fixed block size cannot represent real song forms · CRITICAL
`form.py:124-158`

4-bar intro + 8-bar verse + 16-bar chorus has no valid representation.

**Recommended:** boundary detection on a bar-level self-similarity matrix (Foote
novelty, or laplacian segmentation), then cluster the variable-length segments.

**Implemented, measured, and off** — `form.novelty` / `form.novelty_blocks`,
behind `form.NOVELTY = False`, with a test that runs it so the measurement stays
re-runnable.

| segmentation | root | triad | form |
|---|---|---|---|
| fixed grid (ships) | **0.857** | **0.852** | **0.713** |
| Foote novelty | 0.846 | 0.835 | 0.610 |

Worse on every axis, and the way it loses says why: Creep goes from 1.000 root and
four chords to 0.949 and **nine**. The song is one eight-bar loop for its whole
length, the fixed grid at the measured period lands on it exactly, and the novelty
curve peaks on the chord changes *inside* the loop — because at bar resolution,
over harmony, that is what a checkerboard kernel sees. Foote novelty segments
audio features well, where timbre and density change at a boundary and not within
one; this repertoire's sections are the same four chords at different volumes.

So the finding's diagnosis is right — a 4-bar intro, an 8-bar verse and a 16-bar
chorus genuinely have no representation in the fixed grid — and its remedy is
wrong for this feature. The untried version is novelty over the **energy** curve,
which the structure probe already produces and which nothing but the chorus label
reads today.

### F20 — near-duplicate groups never merge; everything falls through to "verse" · CRITICAL
`form.py:161-185,279-317`

Greedy clustering with the first block as the fixed representative and a hard 0.75
threshold; no re-merge after canon smooths the noise; and the label assigner's
final `else` is `verse`.

**Partly fixed, and one third of the finding is factually wrong about this
tree.**

(a) *Groups are never re-merged after canon.* **They are.** `model.build` runs to
a fixed point and every round re-runs `form.detect`, which re-clusters from
scratch over the bars the previous round just made agree. Traced: Wonderwall 9 → 8
groups, Someone Like You 11 → 10, Mary Jane 8 → 7 → 6, Smooth Criminal 26 → 22.
Recorded on `model.CANON_ROUNDS`.

(b) *Greedy clustering with a fixed representative.* Best-match clustering was
tried before this audit and rejected on measurement: form +0.012, root −0.008 —
a block reassigned to a better-matching group is then made to agree with it (§21),
so a better cluster produces a worse chart. Left as it is, and `form.cluster` says
so.

(c) *Everything falls through to "verse".* **Fixed by F21** — labelling now works
from structure alone and no longer needs a loudness probe to say anything at all.

The symptom underneath remains real: ten or eleven groups on songs with five to
seven sections. What the measurements above say is that it is not caused by any of
the three mechanisms named, and the honest next step is the energy-feature
segmentation in F19 rather than more threshold movement.

### F21 — chorus = loudest repeated group; naming collapses without energy · MAJOR
`form.py:289-299`

**Fixed** — `form._chorus` scores chorus-ness from four signals rather than
one: loudness where a probe ran, how often the group comes round, whether its
occurrences are interleaved with other material, and whether it opens the song
(the heaviest structural term, because it is what decides `A B A` — where the
other two both point at `A`, which opens, recurs and is interleaved, and is the
verse).

The single point of failure is gone with it: a build with no structure probe used
to name every section `Part N`, and now returns the same labels the energy-fed
answer gives on the same fixture. `custom`/`Part N` is now reachable only when
*nothing in the song repeats*, which is the case where it was always the honest
answer.

Not measurable on the chart corpus — the form score reads section *content*, not
labels — so this is judged by argument and by tests, and it is stated that way.

### F22 — runt absorption destroys repeat structure · MINOR
`form.py:255-269` · `structure.py:7`

**Fixed in the narrow form the corpus supports.** A runt is no longer absorbed
into a host that is a clean occurrence of a **repeating** group, because an 18-bar
section is not an occurrence of a 16-bar group: `block_similarity` scores unequal
lengths 0, so that verse silently drops out of the vote and out of §21.

The unrestricted version — protect every clean host, including groups with one
occurrence — costs Smooth Criminal a quarter of its form (0.750 → 0.500) and gains
nothing anywhere, because a lone section is not made worse by two more bars.
Restricted, the corpus is unchanged to four decimals and the failure is closed.

### F23 — `solo` is never assigned; `preChorus` almost never fires · NOTE
`app/payload.py:9` · `form.py:311-331`

**Documented, not implemented.** `solo` is unreachable because telling a solo
from a verse is a question about *timbre*, and one loudness scalar per hop is
everything that crosses §2.1's audio boundary. The fix is a second scalar out of
the structure probe, not a rule in `form.py`. `preChorus` is rare because its
guard demands two repeated non-chorus groups *and* a half cadence *and* a chorus
after it — and songs satisfying all three are genuinely rare, which is the guard
working.

## F · Meter & tempo

### F24 — tempo-octave correction exists but is off by default · MAJOR
`app/config.py:43` · `pipeline.py:161-165` · `meter.py:219-229`

`assemble()` hard-fails with `TempoUnreadable` outside 40–220 BPM while the
halve/double machinery that would repair it sits behind a flag defaulting off.

**Fixed** — `theory_tempo_octave` defaults to **on**.

The reason it was off is worth keeping: the correction rewrites the beat grid, so
every bar line and every anchor moves, and there was no measurement to turn it on
with. There is now, and it is the shape that settles this kind of question. The
correction fires only on a tempo outside 40–220 BPM; no track in the corpus is;
turning it on is a **no-op to four decimal places on all eleven**. So the risk of
moving every anchor is bounded to songs that today do not ship at all —
`assemble` raises `TempoUnreadable` and the user pays a quota charge for a failure
the halve/double machinery beside it could have repaired.

A 235 BPM reading is now a song at 118; a 206 BPM one is a song at 103 rather than
a low-confidence chart at 206. Both pinned in `tests/test_pipeline.py`, with the
old behaviour still pinned under `theory_tempo_octave=False`.

### F25 — sidecar BPM is the rounded integer, not the measured value · MINOR
`pipeline.py:244` · `meter.py:114-124`

**Fixed** — the sidecar carries `meter._measured_bpm`'s two-decimal value
instead of the container's rounded integer. Same claim about the same grid, at the
precision it was measured at; up to half a BPM was being thrown away, which over a
four-minute song is a bar and a half.

### F26 — meter arbitration covers 4/4 vs 3/4 only · NOTE
`meter.py:23,281-300`

**Documented.** The trackers emit `n/4` and nothing else, so 6/8 and 12/8 are
approximated. Survivable for a strumming app — §14's idiom set includes "6/8 in
two", and the anchors are absolute times off the recording either way — and fixing
it needs a tracker that reports compound meter, not a rule in `meter.py`.

## G · Pipeline & robustness

### F27 — a lint failure on one difficulty tier discards all tiers · MAJOR
`pipeline.py:228-232`

The sidecar path immediately below already has the right pattern.

**Fixed** — a dirty tier is withheld and its siblings ship, which is the shape
the sidecar loop immediately below it already had. §12.4's rule ("never return a
payload that would warn") is satisfied exactly as it was by raising; what changed
is that an `easy` render merging itself into a malformed bar no longer takes
`normal` and `hard` down with it. The job still fails when no tier survives, which
is the case the fatal was really guarding.

> **Superseded (2026-08-19).** The difficulty tiers were removed; there is one
> chart, so there is no sibling to spare and a lint failure is fatal again.

### F28 — empty bars are silently dropped, shifting every index after them · MINOR
`structure.py:63`

**Fixed** — a bar no chord reaches is now *held* (§18's own rule) rather than
filtered out, so bar *k* stays at index *k* whatever happens upstream. It cannot
happen today, because `hold_through_gaps` runs first; the point is that this was
an invariant enforced in a different module, and the failure if it ever stopped
holding is silent and total — every section, `start_bar` and anchor moving
relative to the chart while every self-consistency check still passes.

### F29 — the EASY tier always merges short chords leftward · MINOR
`postprocess.py:66-83,106-114`

**Fixed, and it is the largest single win in this audit for the tier a beginner
actually sees.** At `easy`'s one-bar floor, a too-short span now goes to its
**longer** neighbour rather than always leftward — the same "more evidence wins"
rule `merge` uses one function up.

| easy tier | root | triad |
|---|---|---|
| before | 0.761 | 0.732 |
| after | **0.791** | **0.777** |

Viva La Vida alone goes 0.732 → 0.949. Form goes the other way (0.822 → 0.780) and
that is the trade, stated honestly: an over-flattened chart has a shorter form
string and the collapsed-LCS metric rewards it for that. `easy` exists to show a
beginner the right chord.

> **Superseded (2026-08-19).** The difficulty tiers were removed, and with them
> the one-bar floor this rule was written for. `drop_short` is back to its
> one-beat jitter floor, where there is no second real chord to choose between,
> so `to_longer` went too. The numbers above measured a chart that is no longer
> produced.

### F30 — overlap tiebreak prefers longer over more confident · NOTE
`postprocess.py:57`

**Fixed** — two readings quantized onto the same beat are now decided by
duration × confidence rather than by duration and then confidence, so a four-beat
span the engine hedged at 0.3 no longer beats a one-beat span it was sure of. This
is `vocabulary.Reading.mass`, the same quantity under the same argument.
Measure-neutral on the corpus; it is a rare path, and it was wrong.

### F31 — dead code and phantom adapters · NOTE
`harmony.py:83-84,119-137` · `engines.py:45-55`

**Fixed** — `harmony.distance` and `harmony.is_tonic` are gone (both written for
callers that never arrived); `harmony.roman` stays and now has one, because
`keyaudit` reports a conflict as the two degrees in dispute. `chordino` and
`madmom` are marked `# planned` at the table, with a note that
`register_builtins` checks the module exists before registering, so neither can
reach `/healthz` by accident.

## H · Evaluation

### F32 — no quantitative accuracy measurement exists · CRITICAL
`scripts/real_song_check.py` — 8 songs, root-set assertions only

**Recommended:** an offline `mir_eval` harness scoring **two charts per song** —
the raw engine output and the post-theory chart — so every stage's contribution is
attributable and each fix becomes a measured delta.

**Done, and it is Step 0 exactly as recommended — but the finding's premise was
stale.** `bench/run_bench.py` already scores delivered accuracy against
timestamped Isophonics annotations, and `bench/lab.py` already grades whole charts
against hand-written references in about a second.

What genuinely did not exist is **per-layer attribution**, and that is what was
built: `python bench/lab.py layers` scores every song once per posture — engine
only, then each theory layer added in pipeline order — so "sometimes it's the
engines, sometimes the theory layer" is now a question the repo answers per song.
Every verdict in this document is a delta measured with it.

```
posture                  root  triad   form
engine only             0.809  0.789  0.702
+ vocabulary  §20.8     0.809  0.789  0.702
+ key audit   §20.10    0.809  0.793  0.713
+ consensus   §20.4     0.821  0.812  0.713
+ belief      §20.9     0.821  0.812  0.713
+ form        §21       0.854  0.849  0.755
```

`mir_eval` itself is not adopted: the chart corpus deliberately carries no
timestamps (it grades harmony and form, and says so), and the timestamped corpus
already has `delivered_accuracy`. Two rulers, neither substituting for the other.

### F33 — enhancement: honest no-chord handling · NOTE

**Half done, and the half that was cheap.** The key is now detected on
`process(..., fill=False)` — the chords the engine reported, with the silences
still silent (F12). Keeping N.C. all the way to `compile` is the other half and is
a change to what a `GridSpan` *is*: every consumer between `postprocess` and the
wire assumes a contiguous timeline, and §18 has no rest primitive to hand them.
Worth doing; not worth doing halfway. Recorded on `hold_through_gaps`.

### F34 — enhancement: keep inversions at the hard tier · NOTE

**Declined — the obstacle is the container, not the parser.** There is no field
for a bass note in `CompositionPayload`, and the app voices a chord from its root,
so a payload saying `C/E` would either fail to parse or sound a plain C under a
label promising otherwise. Adding it is a §12.2 change first. The loss is already
counted rather than hidden: every discarded bass sets `exact=False` and shows up
in `exactRatio`. Recorded on `chords.normalize`.

---

## What actually answered each symptom

Worth setting down next to the symptom map at the top, because in three cases out
of four the finding that fixed it is not the finding the map predicted.

**"7ths in almost every song — Bm and Bm7 side by side."** Not F6/F7 — those gates
are measured and loosening them loses accuracy. What removed them is **F9's key
audit** (`E7` against `E` in E major is a conflict on the tonic degree, and the
seventh is the minority reading) working on top of **F2's decoder**, which stops
the seventh being reported in the first place on the passes where it was a
one-frame peak. Corpus vocabulary 71 distinct chords → 61.

**"Major and minor in the same key — C and Cm."** F9, as predicted, and it needed
two gates the finding did not name: secondary dominants restricted to sevenths,
and a 20% ceiling on how much of a root the foreign reading may hold. The second
is what stops the layer deleting Creep's `Cm` when the *key* is what is wrong.
Everything it declines to repair is now reported (`keyConflicts`), which is how a
wrong key becomes visible at all.

**"Unlikely alterations — A and A#."** F8's rule is written and is **inert on this
corpus** — three semitone islands in eleven songs, none brief enough to qualify.
What the corpus does contain is the *other* mechanism the map named: F1's tuning
case, where Mary Jane's Last Dance uncorrected is the entire chart a semitone
down. That gate was already right. F14 (per-loser voting) is the third route and
is now open.

**"Several distinct verses."** Not F19 — novelty segmentation is worse on every
axis here. What paid is **F18 measured in the opposite direction** (a *wider*
period margin, +0.034 form), **F3**'s overlapping windows (+0.008), and **F21**,
which gives a song labels from structure when there is no loudness probe instead
of calling everything `Part N`. Form 0.688 → 0.755.

## The one process note worth keeping

Six of the thirty-four recommendations were implemented and then **reverted or
left off on measurement**: F1's sign check (destroys a song), F2's beat-sync
decoder (−0.048 root), F6/F7's loosened snap gates (measured before this audit),
F19's novelty segmentation (−0.011 root, −0.103 form), F22's unrestricted runt
guard (−0.25 form on the song it touches), F16 and F17 (0.2% of slots; a measured
+0.014 form the other way).

That is not a criticism of the review — every one of those was a well-argued
reading of the code, and three of them are what a textbook would recommend. It is
the reason Step 0 was Step 0. A recommendation that sounds right and measures
wrong is indistinguishable from one that sounds right and measures right until
somebody runs it, and the whole value of `bench/lab.py layers` is that running it
now costs about a second.

## What the review credits, and what must survive any fix

* **Deterministic replay** — recording theory decisions in the model and
  replaying rather than retaking them (`model.render`, `record=False`), with the
  `seed_key`/`vote_key`/`canon_key` bookkeeping. (Written for the per-difficulty
  renders; kept when they were removed, because a render reading the model's
  decisions is worth holding by construction either way.)
* **Triad folding for structure** — `form.folded()`, so 7th flicker cannot
  fragment segmentation.
* **Downbeat repair** — the drop-spurious/insert-missing walk in `downbeats.py`
  with mode-based bar length and honest "unreliable" reporting.
* **Phase rotation from harmony** — `meter._phase`, correctly gated.
* **Fatal-vs-advisory sidecar linting** — shipping a low-confidence sidecar
  instead of failing.
* **Operational hygiene** — scratch containment with prefix verification,
  SHA-pinned checkpoint loading with a safe-unpickle path, refundable quota
  charges, self-explaining egress diagnostics.

Audit of the analysis system only; tests, data and git history were out of scope.
Line numbers refer to the tree as reviewed on 2026-08-18.
