# Where the strong baselines fail, round 4 (2026-10-03): diagnostics from existing predictions

This round uses existing predictions only. No model was trained. It answers three questions set after round 3.
Where do the strong baselines fail together? How many independent recordings are involved, and how many points do
those failures cost? Which testable mechanism could reach the original goal of 1 point of segment F1@50 above the
strongest previous configuration? The paper stays paused. Every number here is a diagnostic. None of it is a model
result.

- **Systems.** The main systems are E0 with B3 tuned and E0 with ASRF (seeds 0, 1 and 2). The same systems on E1 are
  the check. The official EchoViewCLIP system with B3 on its posteriors (B7 with B3) is a fixed system, reported on
  both halves.
- **Units.** A unit is one reference segment of a multi-fragment stream: a run of fragments with the same class. A
  unit is wrong for a system when the segment F1@50 matching gives it no predicted segment of its class. The
  selection half has 502 units from 231 source clips, and the calibration half has 518 from 213.
- **Clips and sessions.** A source clip is one labelled EV9V recording. A session is the protocol's split unit, the
  clip id without its last part. The streams reuse clips, so units are also counted by clip and session. A repeated
  clip is never counted as a new case.
- **Own samples.** These are the unit's own 10 Hz samples in the stream, with the same frames and contrast variant.
- **Causes.** Each wrong unit gets the first cause that applies. In a label error, a predicted segment covers the unit
  at IoU 0.5 or more but has another class. In a missed cut, a view change at the unit's edge has no predicted change
  within 1 s. In an extra cut, an unmatched predicted change lies inside the unit. A boundary shift is everything else.
- **Categories.** In a stable common error, B3 and all three ASRF seeds get the unit wrong. In a common error, B3 and
  at least two ASRF seeds do. B3-only and ASRF-only errors are defined the same way. The full B3 × ASRF-seed table is
  in `failure_tables.md`.
- **Validation only.** Both halves have been used during development. The test split was not read.

## Answers

### 1. Where the strong baselines fail together

- On E0, 54 of the 1,020 units are common errors and 37 are stable common errors. B3 gets 76 units wrong in total,
  and only 4 of them are B3-only. ASRF-only errors number 16.
- The common errors are A4C/A5C errors. Of the 54, 38 are A5C segments given A4C. Another 13 are A4C segments, and
  11 of those sit next to an A5C segment and lose the change between them. PLAX, PSAX and SC4C units fail in at most
  3% of cases for B3 and ASRF.
- At reference view changes, B3 and ASRF both miss 37% of the A4C/A5C changes. For every other class pair they miss
  at most 5%. B7 with B3 misses 14% of the A4C/A5C changes and shifts another 7%.
- B7 with B3 gets 27 of the 54 common errors wrong as well, 21 of the 32 on the selection half.
- For most common errors, the visual evidence itself does not support the label. For 37 of the 54, the frozen
  classifier's mean over the unit's own samples picks another class. Only 17 have own-sample evidence for the right
  class, which the systems then lose.
- E1 shows the same picture, with 43 common errors from 25 clips.

### 2. How many recordings, and how many points

- The 54 common errors come from 29 source clips in 24 sessions. Ten clips carry 38 of them, and nine of those ten are
  A5C clips. A single A5C clip is in 11 of the common errors and in 21 units in all.
- For 28 of the 54, the classifier also assigns the whole source clip to another class, with the same contrast
  variant. B3 and ASRF also misclassify several of these clips as single-view recordings.
- Repairing the 54 units ideally would raise E0 B3 from 0.934 to 0.986 on the selection half and from 0.934 to 0.973 on
  the calibration half. ASRF would gain 4.5 to 5.1 points and 3.1 to 4.0 points.
- Repairing only the units of the five most affected clips would raise E0 B3 by 5.5 points on the selection half (to
  0.989) and by 1.8 on the calibration half.
- A handful of clips therefore decides most of the selection-half gap. It also means a 1-point difference on that half
  can come from one or two clips.

### 3. Which mechanism could reach +1

- **Segmentation alone has little room.** Suppose every wrong unit gets its reference interval, labelled by the
  classifier's own evidence. E0 B3 then rises by 1.2 points on the selection half and 2.7 on the calibration half.
  Keeping the predicted intervals and correcting only the class raises it by 4.6 and 4.0.
- **The class decision fails in a specific way.** In misread A5C units the A5C evidence is intermittent. In 63% of
  them on E0 (85% on E1), the 90th percentile of p(A5C) exceeds 0.8, while the median sample stays below 0.5. A4C
  units rarely look like this (2% on E0, 5% on E1). In case 1 below, p(A5C) peaks about once per second. Averaging over
  the segment, as B3's labelling and the whole-clip check do, removes these peaks.
- **A fixed rule does not use that evidence well.** A no-training probe relabels a predicted A4C segment as A5C when
  its 90th percentile of p(A5C) passes a threshold chosen on the calibration half.
  - For E0 B3 it raises segment F1@50 from 0.934 to 0.966 on the selection half and to 0.945 on the calibration half.
    It also lowers native macro-F1 and does not meet the native floor.
  - For ASRF it costs native macro-F1 heavily. For E1 it changes little.
- **Only some bases could pass both halves even ideally.** The calibration target is 0.982. With class-only repairs,
  E0 B3 reaches only 0.974, so a mechanism built on it cannot pass that half. E1 ASRF reaches 0.970 to 0.973 on the
  selection half and 0.987 to 0.989 on the calibration half, so both of its targets are within reach. E0 ASRF seeds
  1 and 2 also clear both.
- **The testable mechanism with room** is a segment-level A4C/A5C decision. It reads the distribution of frame-level
  evidence inside each predicted segment instead of its mean. A segmentation mechanism that keeps short segments could
  add about 1 point on the selection half at best, even if it were perfect.

### Decision after review (2026-10-03)

No echo expert is available, so the clip review below will not take place. The user decided as follows.
- **Labels stay.** Every main result keeps all clips and their public labels.
- **The rule's verdict is not a finding.** The "data" verdict says the scores depend heavily on a few clips. It does
  not show that those clips are mislabelled. A whole clip read as another class by its mean probability may itself be
  the aggregation problem that section 7 describes.
- **Sensitivity analysis.** A separate analysis removes the ten clips listed in
  `configs/experiments/evidence_pooling_v1.yaml`, with the same list for every system and no re-tuning. It only shows
  how much the conclusions depend on those clips. It can never decide a winner.
- **Next round.** Round 5 tests whether keeping intermittent high-score evidence improves the A4C/A5C decision without
  new A4C errors. It first checks that the success criterion is reachable with the intervals it would use.

The recommendation as first written is kept below for the record.

### Recommended next step (one), as first written

The pre-declared rule selects **data first**. The five most affected clips carry 54% of the common errors, and 52%
have a whole source clip read as another class. Both thresholds are passed on E0 and E1 and for both category
definitions. One caveat applies. The whole-clip check uses the mean, which the intermittent evidence shows to be
biased against A5C. The concentration in a few clips does not depend on that.

- **The step itself.** Check the ten concentrated clips before any training.
  - For each clip, prepare its whole-clip probability trace and frames at its A5C peaks and A4C troughs, in the
    contrast variant used.
  - A named reviewer marks each clip as "A5C, outflow visible only part of the cycle", "looks like A4C throughout" or
    "cannot be determined". AI readings do not count as labels.
  - Fix in advance how marked clips enter evaluation. Results are reported with the original labels and without the
    marked clips, and labels are never changed silently.
- **If most clips stay A5C,** the single small training experiment follows. It trains a segment-level A4C/A5C decision
  on the training bank only, reading the frame-level evidence inside fixed predicted segments. Its base should be one
  whose class-only ceiling clears both targets (E1 ASRF). Protocol and success criterion stay as before.
- **If most clips look like A4C throughout,** no model change is indicated yet. The evaluation is corrected first, and
  this inventory is rerun on it.

## 1. What was fixed before the tables

`configs/experiments/failure_analysis_v1.yaml` was written before any failure table was computed. It fixes:
- the systems and units;
- the categories, including the full B3 × ASRF-seed table;
- the cause rules, with predicted changes matched one-to-one to reference changes within 1 s;
- the per-unit checks and the length bins;
- the controls: right units of the same class and length;
- the condition for the neighbour-swap check;
- the three repairs used in the simulations;
- the decision rule.

The inventory reproduced every system's reported segment counts exactly before anything else was computed. The
script stops otherwise. A fresh ASRF decode also had to match the stored predictions.

Two analyses were added after the inventory and are marked as such: the intermittent-evidence statistics with their
probe (section 7), and three extra repair tables (section 6).

## 2. Error inventory

E0, both halves (units / source clips / sessions):

| Group | Units | Clips | Sessions | Also wrong for B7 with B3 |
|---|---:|---:|---:|---:|
| All units | 1,020 | 444 | 71 | |
| Wrong for B3 | 76 | 45 | 33 | |
| Wrong for ASRF, seeds 0 / 1 / 2 | 92 / 59 / 56 | 51 / 36 / 30 | 35 / 28 / 23 | |
| Wrong for B7 with B3 | 60 | 39 | 30 | |
| Stable common (B3 and all three seeds) | 37 | 20 | 17 | 24 |
| Common (B3 and at least two seeds) | 54 | 29 | 24 | 27 |
| B3 only (no seed wrong) | 4 | 6 | 6 | 0 |
| ASRF only (at least two seeds, B3 right) | 16 | 16 | 14 | 4 |

Per half, the common errors number 32 on the selection half (16 clips, 11 sessions) and 22 on the calibration half (13
clips, 13 sessions). On E1 there are 43 common errors from 25 clips in 20 sessions, and B7 with B3 also gets 39 of them
wrong. The full tables are in `failure_tables.md`, including every cell of B3 right or wrong by the number of ASRF
seeds wrong.

**Causes.** No common error comes from an extra cut or a pure boundary shift.
- Under B3, 36 of the 54 common errors are label errors and 18 are missed cuts.
- Under ASRF, pooled over the wrong seeds, 92 are label errors and 53 are missed cuts.
- The ASRF-only errors are mostly missed cuts (21 of 32 pooled).

## 3. Recognition or segmentation

E0 common errors (54 units) against matched controls. Controls are units that B3 and all ASRF seeds get right,
re-weighted to the same class × length cells.

| Check | Question | Common errors | Controls |
|---|---|---:|---:|
| Whole source clip vs the unit | Is the whole clip right? | 48% | 94% |
| | Whole clip right but the unit's own samples wrong? | 17% | 0% |
| Reference interval | Is the mean over the unit's own samples right? | 31% | 100% |
| Alone vs in the stream | B3 right on the unit alone / in the stream | 30% / 15% | 100% / 100% |
| | ASRF right on the unit alone / in the stream (seed mean) | 36% / 24% | 100% / 100% |
| Predicted limits and classes | Causes under B3 | label 36, missed cut 18 | |

The units split three ways:
- in 17 the own samples are right, and the systems lose correct evidence;
- in 9 the whole clip is right but the unit's own samples are not, as the user warned can happen;
- in 28 both the unit and the whole clip are read as another class.

The last two groups (37 units) **cannot be determined** from the fragment label alone. The model may be wrong, or the
label may not fit those frames, and no expert has reviewed them. For stable common errors the split is 8 / 7 / 22.

The one-system errors differ. B3-only and ASRF-only units all have right own-sample evidence (4 of 4, 16 of 16). Those
are pure segmentation failures of one system. ASRF swallows short segments there, mostly through missed cuts. They are
few.

## 4. Representative cases

Figures are in `cases/` (`cases/cases.md` lists them). Each shows the E0 classifier's per-sample probabilities, the
reference and the outputs of B3, the three ASRF seeds and B7 with B3, the checks and five frames of the unit. Frames are
from EV9V (CC BY 4.0).

| Case | Category, cause | Unit | Own-sample mean | Reading |
|---|---|---|---|---|
| 1 | stable common, label | A5C, 60 samples, two clips | A4C 0.71 | Cannot be determined. p(A5C) peaks about once per second. |
| 2 | stable common, label | A5C, 12 samples | A4C 0.76 | Cannot be determined. |
| 3 | B3 only, missed cut | PSAX, 39 samples | PSAX 0.82 | Segmentation failure of B3. |
| 4 | B3 only, missed cut | A5C, 20 samples | A5C 0.80 | Segmentation failure of B3. |
| 5 | ASRF only, missed cut | A5C, 18 samples | A5C 0.87 | Two ASRF seeds merge it into the A4C neighbour. |
| 6 | ASRF only, missed cut | A4C, 11 samples | A4C 0.98 | Short segment swallowed by ASRF. |
| 7 | common, missed cut | A5C, 12 samples | A4C 0.73 | Cannot be determined. |

## 5. Failure rates, neighbours and the paired swap check

**By class.** On E0, A5C units fail in 24% of cases for B3 and 23% for ASRF. A4C units fail in 9% and 8%. The other
classes fail in at most 3%. **By length**, failure peaks at 11–15 samples (12%). 10-sample units fail less often
(4–5%), so length alone does not explain the failures. The tables by class × length and by neighbour class are in
`failure_tables.md`.

**Neighbour condition.** It held for one pair only. A4C units with an A5C neighbour fail in 21% of cases, against 3%
for A4C units without one, after length standardisation. That covers 20 wrong units from 19 clips. A5C units fail
about equally often whatever their neighbours (19–30%). Their failures come from the units themselves.

**Paired swap check (E0, both halves).**
- **Design.** The checked units are 65 A4C units next to an A5C unit, plus 152 A4C controls without one that every main
  system gets right. Each unit's samples stay fixed while only its neighbours change.
- **Donors.** Each change uses blocks from other clips of the same half, in the neighbour's contrast variant, with three
  draws per unit. B3 has no attention, so none of this can be an attention effect.
- **Effect on the controls.** Replacing a neighbour with another clip of the same class keeps 99–100% of the controls
  matched. Replacing it with an A5C block keeps only 58% for B3 and 61–78% for the ASRF seeds.
- **The inserted A5C block decides it.** When the classifier reads the inserted A5C block as A5C, 69% of controls stay
  matched for B3 and 75–92% for ASRF. When it reads the block as another class, which happened in 146 of 456 draws, only
  35% and 32–47% stay matched.
- **The checked units.** A4C units that B3 got wrong become right in 100% of cases when isolated or given A4C
  neighbours, because no A4C/A5C change remains. With their A5C neighbour swapped for another clip's A5C, they are
  matched in 33% of draws. Units B3 got right stay matched in only 70%.

Two effects therefore combine at A4C/A5C changes.
- When the A5C side looks like A4C to the classifier, both systems merge it with its A4C neighbour. That is a
  recognition problem showing up as a missed cut.
- When the A5C side does look like A5C, B3 still loses about a third of the A4C units, and ASRF 8–25%. That is a
  boundary weakness for this similar pair. In B3 it can only come from its window-mean divergence test or its
  threshold, which were not measured separately here.

## 6. What the failures cost (diagnostic repairs)

The largest cause over the main systems is the label error (157 wrong units, against 126 missed cuts). For E0 B3, with
target 0.944 on the selection half and 0.982 on the calibration half:

| Repair (diagnostic only) | Selection | Calibration |
|---|---:|---:|
| Now | 0.934 | 0.934 |
| Label errors, ideal (reference class on the reference interval) | 0.988 | 0.978 |
| Label errors, class only (predicted intervals kept) | 0.980 | 0.974 |
| Label errors, segmentation only (reference interval, classifier's evidence) | 0.941 | 0.948 |
| Missed cuts, ideal / segmentation only / class only (added) | 0.945 / 0.940 / 0.947 | 0.949 / 0.947 / 0.941 |
| All wrong units, segmentation only (added) | 0.946 | 0.961 |
| All wrong units, class only (added) | 0.980 | 0.974 |
| Units of the five most affected clips, ideal (added) | 0.989 | 0.952 |

Every system and half is listed in `failure_tables.md`. These repairs act on predicted segments, and a repaired
segment can cover more than its unit. They are bounds for this kind of fix, not achievable scores. A category column
there repairs only the units that the row's system gets wrong. Some are therefore empty by definition, such as the
B3-only column in ASRF rows.

## 7. Added after the inventory: intermittent A5C evidence and a probe

The case figures showed A5C probability peaking about once per second inside misread A5C units. The statistics below
were computed afterwards, so they are not pre-declared.

| Units (E0) | n | Median of the mean p(A5C) | Median 90th percentile of p(A5C) | Share with 90th percentile above 0.8 |
|---|---:|---:|---:|---:|
| A5C, common errors | 38 | 0.33 | 0.92 | 63% |
| A5C, right for every main system | 138 | 0.91 | 1.00 | 100% |
| A4C, right for every main system | 197 | 0.03 | 0.07 | 2% |
| A4C next to A5C, right for every main system | 45 | 0.02 | 0.06 | 2% |

On E1 the same shares are 85% for the misread A5C units and 5% for A4C units.

**The probe** (`a5c_evidence_probe.md`) is decoding only.
- **Rule.** A predicted A4C segment becomes A5C when its 90th percentile of p(A5C) reaches τ. τ is chosen on the
  calibration half by the shared rule and then applied unchanged to the selection half.
- **E0 B3.** Segment F1@50 rises to 0.966 on the selection half and 0.945 on the calibration half. Native macro-F1
  falls slightly (0.965 to 0.962, and 0.966 to 0.961), and no τ met the native floor.
- **E0 ASRF.** The fallback chose τ = 0.5. Native macro-F1 then drops to 0.905 on the selection half.
- **E1.** τ = 0.95 changes segment F1@50 by −0.3 to +0.6 points.

The evidence is there, but a fixed threshold trades it against A4C clips that show some A5C-like frames.

## 8. Decision rule

| Inputs of the rule (added: robustness) | Units | In the five most affected clips | Whole clip read as another class | Own samples read as another class |
|---|---:|---:|---:|---:|
| E0, common (the rule's evidence) | 54 | 54% | 52% | 69% |
| E0, stable common | 37 | 62% | 59% | 78% |
| E1, common | 43 | 63% | 70% | 60% |
| E1, stable common | 30 | 77% | 77% | 67% |

The first rule that holds is "data". It holds in every row. The "visual" rule would also hold, but it comes second.
The verdict means the scores depend heavily on a few clips. It does not establish that any clip is mislabelled. No
expert has looked at them, and the "whole clip" input uses the same mean pooling that section 7 calls into question.

## 9. Limitations

- Both validation halves have been used during development, so these are development diagnostics. The test split was
  not read.
- Labels are clip-level labels carried into every sample. No expert has reviewed any clip. Units whose own samples the
  classifier reads as another class cannot be resolved here.
- The streams reuse clips. One clip appears in 24 units. Counts of units overstate independent evidence, which is why
  clips and sessions are reported.
- The repairs are idealised. Section 7 was added after the inventory, and its probe threshold was chosen on the
  calibration half, which is development data like the selection half.
- The "whole clip" check and B3's labels both use the mean probability, which section 7 shows to be biased against
  intermittent A5C evidence.

## 10. Files and runs

- **Configuration:** `configs/experiments/failure_analysis_v1.yaml`.
- **Code:**
  - `echo_routing/evaluate/failure_units.py`, with tests in `tests/test_failure_units.py`;
  - `scripts/round4_failure_inventory.py`;
  - `scripts/round4_failure_report.py`;
  - `scripts/round4_neighbour_swap.py` and `scripts/summarize_neighbour_swap.py`;
  - `scripts/round4_case_figures.py`;
  - `scripts/round4_a5c_pooling_probe.py`.
- **Runs** (server, `~/echo-view-routing/runs/`):
  - inventory `20261003-014011-md4_failure_inventory-9f564d88-s0`;
  - swap check `20261003-015334-md4_neighbour_swap-9f564d88-s0`;
  - probe `20261003-020021-md4_a5c_pooling_probe-9f564d88-s0`;
  - report `runs/md_round4/report/` and figures `runs/md_round4/cases/`;
  - spec `runs/md_round4/spec.json`.
- **In this folder:**
  - `failure_tables.md` and `failure_report.json`: all tables;
  - `units.csv`: one row per unit, input and half, with every check and outcome;
  - `inventory.json`: reproduced counts and native errors;
  - `neighbour_swap.md` and `neighbour_swap.json`;
  - `a5c_evidence_probe.md` and `a5c_evidence_probe.json`;
  - `cases/`.
