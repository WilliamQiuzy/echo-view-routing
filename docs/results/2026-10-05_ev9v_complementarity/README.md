# EV9V complementarity of E0 and B7, and one fixed equal-weight fusion (2026-10-05)

This round asked three things. Do two existing recognition systems make different, usable errors? Does a fixed
fusion of their probabilities improve the full system? If it does, can one E1 + ASRF student inherit the gain? The plan
is `docs/superpowers/plans/2026-10-05-ev9v-complementarity-distillation.md`. The definitions, inputs and the start
conditions for training were committed before any result (`configs/experiments/ev9v_complementarity_v1.yaml`, commit
988ed11). One code fix followed before any task-2 result existed (36502be). It gave the calibration grid the
calibration half only and changed no rule.

Only the two EV9V validation halves were read. The test split stayed sealed. No EchoNet-Pediatric data, cache or
weight was used. Both halves have been used in earlier development, so everything here is exploratory.

## Answers

1. **Where the systems complement each other.** The complementarity runs in both directions and differs by half.
   On the selection half, 26 reference segments are right only for E0 + B3 and 17 only for B7 + B3. On the
   calibration half the counts are 6 and 31. A few recordings carry many of these segments.
2. **Whether the fixed fusion improves the full system.** It does not. Equal-weight averaging followed by B3 lands
   between the two teachers on each half. It is 2.1 points of segment F1@50 below the strongest previous system on
   the selection half and 0.8 points below on the calibration half. On the calibration half it is also below the B7
   control that uses the same budget, and no setting of the B3 grid lets it reach that control.
3. **Whether anything was trained.** No. Start conditions 1, 2 and 3 were not met. Only condition 4, the consistency
   check, was met. Task 3 did not start, and no training-bank teacher output was generated.
4. **Whether a single model inherits the gain.** This was not tested. No student was trained, and the fusion showed
   no gain to inherit.
5. **Whether the original goal is exceeded.** No. Nothing in this round beats the strongest previous system on
   either half.

Under the plan, the round stops here. This result closes only this equal-weight fusion and the conditional
distillation that depended on it. It does not show that other fusion rules fail. No weight search and no gate
training followed.

## Terms

- **E0** is the v3 ResNet-18 frame classifier with its own head. It gives one five-class probability per 10 Hz sample.
- **B7** is the EchoViewCLIP stage-1 video model, trained by this project with the official code. It is applied with
  16-frame windows of 1.6 s every 0.5 s. Every sample takes the output of the window whose centre is nearest.
- **B3** is the change-point decoder. It cuts where the averaged probabilities on the two sides of a point diverge.
  Each resulting segment then takes the class with the highest mean probability.
- **q** is the fused probability, 0.5 × p(E0) + 0.5 × p(B7) at every sample.
- A **unit** is one reference segment of a constructed multi-fragment stream. Streams reuse recordings, so units
  are never counted as independent cases. A unit is right for a system when segment F1@50 matching gives it a
  predicted segment of its own class.
- A **recording** is one labelled EV9V clip. A **session** is the protocol's split unit, the clip id without its last
  part. A **native clip** is one whole recording, read as a single view; it is right when the majority of its decoded
  labels is its label.
- **Segment F1@50** matches predicted to reference segments of the same class at overlap 0.5 or more and pools the
  counts over streams. **Boundary recall** is the share of reference view changes with a predicted change within
  0.5 s. **Native macro-F1** is the macro-F1 of the native clips' majority labels over the five classes.

## 1. Inputs and checks (task 1)

- **Hashes.** Every pinned input matched its hash. The digest over the E0 cache files and the digest over the 1,013
  cached B7 files matched as well.
- **Alignment.** All 1,013 validation sequences align sample by sample. These are 521 native clips, 252 pair streams
  and 240 multi-fragment streams. At every sample, E0's cached sample and B7's window-plan reference name the same
  clip, the same actual frame and the same contrast variant, and the lengths agree. No sample comes from a test clip.
- **Class map.** B7's cached five-class rows equal the frozen nine-to-five map of its stored nine-code output. That
  map drops the PMPALA code. The dropped mass averages 2.5% of a sample on the selection half and 3.3% on the
  calibration half. It exceeds 0.5 in 0.7% and 1.8% of samples, and it is largest on PSAX samples (4.4% and 8.6% on
  average). As the plan's checks require, the five mapped columns were divided by their sum for task 2. The frozen
  B7 + B3 was reproduced on the stored rows.
- **Windows run along the stream.** B7's windows are cut from each input stream as a whole, not reset at joins. In the
  multi-fragment streams, 1,834 of 3,032 windows on the selection half and 1,798 of 3,048 on the calibration half
  contain frames from more than one fragment.
- **Spot re-run.** The unchanged runner was re-run on twelve fixed streams, two from each validation bank. Its output
  was identical to the cache, with a largest difference of 0. The old cache therefore meets the plan's windowing
  rule, and nothing had to be re-inferred.
- **Reproduction.** E0 + B3 as frozen (0.9344 / 0.9340) and B7 + B3 as frozen (0.8798 / 0.9721) were reproduced
  exactly. The values are segment F1@50 on the selection and calibration halves. The match covers both systems'
  reported metrics and their segment counts.
- **Coverage.** B7 predictions exist for the two validation halves only. No training-bank or test predictions
  exist, and none were made, because task 3 did not start.
- **No equivalent earlier experiment.** The 2026-09-28 segment-level fusion of MS-TCN and EchoViewCLIP took its
  boundaries from MS-TCN and ran on the old ladder streams. It used caches that read frames in name order, and its
  numbers are withdrawn (`docs/results/2026-09-30_frame_order_fix/`). It was therefore not cited as a result.

## 2. Where E0 + B3 and B7 + B3 differ (task 1)

Both systems are the frozen originals. Each cell gives units, then recordings and sessions in brackets. A unit can
hold several recordings, so the recording counts can exceed the unit counts.

| Half | Both right | Only E0 right | Only B7 right | Both wrong |
|---|---:|---:|---:|---:|
| Selection (502 units) | 437 (229 / 36) | 26 (17 / 13) | 17 (12 / 11) | 22 (11 / 9) |
| Calibration (518 units) | 475 (212 / 35) | 6 (8 / 8) | 31 (21 / 18) | 6 (7 / 7) |

| Half | Both right | Only E0 right | Only B7 right | Both wrong |
|---|---:|---:|---:|---:|
| Selection (284 native clips) | 278 | 3 | 1 | 2 |
| Calibration (237 native clips) | 232 | 0 | 4 | 1 |

**A few recordings carry the one-sided units.**
- On the selection half, one SC4C recording is in 12 of the 26 units that only E0 + B3 gets right. B7 reads it as
  PLAX.
- Also on the selection half, two A5C recordings are in 10 of the 17 units that only B7 + B3 gets right.
- On the calibration half, the 31 units right only for B7 + B3 come from 21 recordings in 18 sessions. Of them, 18
  are A5C, 7 A4C, 5 PSAX and 1 PLAX.

**Some differences are class disagreements, and some are segmentation alone.** For every one-sided unit, the
table compares the two teachers' classes over the unit's reference interval, taking the class of each teacher's mean
probability there. When the classes differ, the teachers disagree on the class. When they agree, the full systems
differ only in their segmentation. The reference interval comes from the labels, so this is a diagnostic only.

| Half | Cell | Units | Teachers disagree on the class | Same class over the interval |
|---|---|---:|---:|---:|
| Selection | Only E0 right | 26 | 18 (E0's class right in all 18) | 8 (right in all) |
| Selection | Only B7 right | 17 | 8 (B7's class right in 7) | 9 (right in 8) |
| Selection | Both wrong | 22 | 2 | 20 (wrong in 14) |
| Calibration | Only E0 right | 6 | 3 (E0's class right in all 3) | 3 (right in all) |
| Calibration | Only B7 right | 31 | 12 (B7's class right in all 12) | 19 (right in all) |
| Calibration | Both wrong | 6 | 1 | 5 (wrong in all) |

The two teachers' class opinions differ in 18 + 8 units on the selection half and 3 + 12 on the calibration half.
In 14 of the 22 selection units that both systems get wrong, both teachers choose the same wrong class over the
interval.

The table reads B7 from its renormalised rows, the B7 teacher of task 2. The frozen B7 + B3 decoded the stored rows
instead, so the classes were also recomputed on those rows after the run. One unit changes. It is a selection A5C unit
that only B7 + B3 gets right (`multi_selection_0008#1`). On the stored rows its B7 class is A5C rather than A4C, so it
moves from "same class" to "teachers disagree". The selection row for "only B7 right" then reads 9 disagreements,
with B7's class right in 8, and 8 units with the same class, right in all 8. No native clip changes.

**The round-4 clue, checked in both directions.** This audit's outcomes agree unit by unit with the round-4
inventory.
- Of the 54 units that E0 + B3 and at least two E0 ASRF seeds get wrong, B7 + B3 gets 27 right. These are 11 of 32 on
  the selection half and 16 of 22 on the calibration half, from 15 recordings.
- In the other direction, E0 + B3 and at least two ASRF seeds get 928 units right. B7 + B3 gets 28 of them wrong
  (22 on the selection half, 6 on the calibration half), from 20 recordings.

These counts select, per unit, the system that happens to be right. That uses the labels, so it is not a
deployable system and never replaces task 2.

## 3. The fixed fusion (task 2)

The fused probability was fixed in advance, q = 0.5 × p(E0) + 0.5 × p(B7) at every sample. No weight, temperature,
class coefficient or post-processing was searched. The B3 grid was the same for E0, B7 and q (40 settings), with each
threshold set as a quantile of the system's own scores on the calibration native clips. On the calibration half only,
a setting had to reach native macro-F1 ≥ 0.980 and boundary recall ≥ 0.95. Among those settings, the highest segment
F1@50 won. The chosen setting was then applied once to the selection half. The frozen originals stay as separate
rows.

- **E0 control.** No setting is feasible (0 of 40). Under B3, E0's native macro-F1 on the calibration half reaches at
  most 0.9744, below 0.980. As fixed in advance, it is marked infeasible and gets no fallback setting.
- **B7 control.** 28 of 40 settings are feasible. The rule chose a half-window of 9 samples and the 0.80 quantile.
- **q.** 26 of 40 settings are feasible. The rule chose a half-window of 11 samples and the 0.999 quantile.

The table gives selection and calibration values in that order. "A5C → A4C units" counts A5C units whose segment is
read as A4C.

| System | Native macro-F1 | Native clips A4C → A5C / A5C → A4C | Segment F1@50 | Boundary recall | Native cuts per clip | Units wrong | A5C → A4C units |
|---|---|---|---|---|---|---|---|
| E0 + B3, frozen original | 0.9650 / 0.9660 | 0 / 3 and 0 / 4 | **0.9344** / 0.9340 | 0.961 / 0.955 | 0.014 / 0.034 | 39 / 37 | 25 / 23 |
| B7 + B3, frozen original | 0.9566 / 0.9927 | 1 / 2 and 0 / 1 | 0.8798 / **0.9721** | 0.955 / 0.975 | 0.063 / 0.042 | 48 / 12 | 17 / 4 |
| E0 + B3, common rule | infeasible | | | | | | |
| B7 + B3, common rule | 0.9566 / 0.9856 | 1 / 2 and 1 / 1 | 0.8658 / 0.9732 | 0.961 / 0.977 | 0.092 / 0.063 | 47 / 10 | 16 / 4 |
| q + B3, common rule | 0.9650 / 0.9850 | 0 / 3 and 0 / 2 | 0.9138 / 0.9643 | 0.955 / 0.977 | 0.011 / 0.013 | 41 / 18 | 21 / 16 |

Bold marks the strongest previous system on each half, the reference of the start conditions.

Paired differences use the existing bootstrap with 2,000 draws. It resamples native clips by session and streams by
stream. The streams reuse recordings, so the stream draws are not independent and these intervals understate the
uncertainty. The teachers and the fusion are fixed systems, each decoded once. They have no seeds.

| Difference | Half | Segment F1@50 | Native macro-F1 | Boundary recall | Units fixed / newly wrong |
|---|---|---|---|---|---|
| q − B7 control | Selection | +0.0479 (+0.0238 to +0.0756) | +0.0084 (−0.0327 to +0.0534) | −0.0052 (−0.0216 to +0.0106) | 18 / 12 |
| q − B7 control | Calibration | −0.0089 (−0.0241 to +0.0057) | −0.0006 (−0.0253 to +0.0218) | 0.0000 (−0.0173 to +0.0167) | 4 / 12 |
| q − E0 frozen | Selection | −0.0206 (−0.0430 to −0.0003) | 0.0000 | −0.0052 (−0.0226 to +0.0082) | 10 / 12 |
| q − E0 frozen | Calibration | +0.0303 (+0.0138 to +0.0482) | +0.0190 (0.0000 to +0.0451) | +0.0226 (+0.0051 to +0.0435) | 20 / 1 |
| q − B7 frozen | Selection | +0.0339 (+0.0129 to +0.0583) | +0.0084 (−0.0327 to +0.0534) | 0.0000 (−0.0168 to +0.0178) | 17 / 10 |
| q − B7 frozen | Calibration | −0.0078 (−0.0240 to +0.0086) | −0.0077 (−0.0292 to 0.0000) | +0.0025 (−0.0159 to +0.0221) | 6 / 12 |
| B7 control − B7 frozen | Selection | −0.0140 (−0.0293 to −0.0001) | 0.0000 | +0.0052 (0.0000 to +0.0133) | 4 / 3 |
| B7 control − B7 frozen | Calibration | +0.0010 (−0.0066 to +0.0113) | −0.0070 (−0.0248 to 0.0000) | +0.0025 (−0.0056 to +0.0149) | 3 / 1 |

**Where q gains and where it loses.**
- **Selection half, against the B7 control (+4.8 points).** Most of the gain comes from fewer predicted segments.
  The predicted segments fall from 549 to 507, while matched units rise only from 455 to 461. One recording, the SC4C
  clip that B7 reads as PLAX, is in 10 of the 18 fixed units. Without the units of that recording, q fixes 8 units
  and newly gets 12 wrong. Boundary recall does not improve (0.955 against 0.961).
- **Calibration half, against the B7 control (−0.9 points).** q newly gets 12 units wrong and fixes 4. In 11 of the 12,
  q reads an A5C unit as A4C where the B7 control reads it right. These units come from 7 recordings. Over the whole
  calibration grid, q's best segment F1@50 is 0.9643, below the B7 control's 0.9732. On the calibration half, no
  other choice of setting would have closed this gap.
- **Against E0 + B3 as frozen.** On the calibration half q fixes 20 units and newly gets 1 wrong (+3.0 points), spread
  over 17 recordings. On the selection half it fixes 10 and newly gets 12 wrong (−2.1 points), with predicted
  segments up from 489 to 507.
- **Fewer cuts inside native clips.** q makes the fewest internal cuts of all rows. It cuts 3 times on each half,
  against 26 and 15 times for the B7 control and 4 and 8 times for E0 + B3 as frozen. These are fewer false cuts.
  They are not a gain in finding view changes, since boundary recall does not rise.

The per-unit and per-clip lists of fixed and newly wrong items reproduce every total exactly. This holds for the
matched, predicted and reference segments and for the correct native clips (`changes.csv`, `fusion_units.csv`,
`fusion_natives.csv`).

## 4. Start conditions for training (task 3)

All four conditions were fixed before the computation. They decide whether training is worth its compute. They are not a
test of whether complementary information exists.

| Condition | Requirement | Result | Met |
|---|---|---|---|
| 1 | q + B3 at least 0.005 above the strongest previous system on both halves | −0.0206 (0.9138 against 0.9344) and −0.0078 (0.9643 against 0.9721) | no |
| 2 | At least 0.005 above each feasible same-budget control on both halves | B7 control +0.0479 and −0.0089; the E0 control is infeasible | no |
| 3 | Boundary recall ≥ 0.95 on both halves; native macro-F1 ≥ 0.973 (selection) and ≥ 0.980 (calibration) | Recall 0.955 and 0.977; native 0.9650 (below 0.973) and 0.9850 | no |
| 4 | Per-source lists of fixed and newly wrong items reproduce the totals | All comparisons consistent | yes |

Training therefore did not start. The later calibration system at 0.9739 (geometry G1 ASRF, three-seed mean) is a
separate sensitivity reference. q's calibration value is 0.0096 below it. That reference changes neither the start
conditions nor the final goal.

## 5. Cost

No model was trained. The B7 cache already existed for both validation halves. The spot re-run of twelve streams
took 17 s on the L40S. The audit ran in about 17 s. The fusion run took about 4 minutes on the CPU, including its
120 calibration decodings and the bootstrap.

Running cost per 10 Hz sample, from earlier measurements on the same L40S at batch size one, is listed per system.
The measurements used different timing set-ups, so their sum is only indicative. Nothing was re-timed.
- **Fused teachers.** B7 takes about 12.0–12.3 ms per sample, of which about 6 ms is preprocessing
  (`docs/results/2026-10-02_model_development_round2/timing_b7.json`). E0 takes about 2.9 ms, of which 2.64 ms is CPU
  frame preparation and 0.22 ms the forward pass (round 1). q + B3 would therefore cost about 15 ms per sample.
- **Single student.** An E1 + ASRF student, had one been trained, takes about 3.3–3.6 ms per sample
  (`timing_e1_systems.json`, round 2).

## 6. What this round does not show

- **That the two systems carry no useful complementary information.** Task 1 shows errors in both directions, and
  only one fixed combination rule was tested.
- **That other fusion rules or a trained combiner would fail.** None was tried. The plan does not open them as
  follow-ups.
- **Independent evidence.** Units come from constructed streams that reuse recordings. A few recordings carry many of
  the units, as section 2 shows. Constructed streams are not real scans.
- **Held-out performance.** Both validation halves were used in earlier development. The B7 checkpoint was selected
  on EchoViewCLIP's historical validation split, which contains both halves, so B7's numbers here are optimistic.
  Both teachers were trained on the training bank, so they have no out-of-fold training-bank outputs.

## Files

- `audit.md` and `audit.json` hold the task-1 tables and checks.
- `audit_units.csv` and `audit_natives.csv` list, per unit and per native clip, the outcomes of the frozen E0 + B3 and
  B7 + B3, the outcome cell and each teacher's class over the reference interval.
- `fusion.md` and `fusion.json` hold the task-2 settings, metrics, paired differences, concentration by recording and
  the start conditions.
- `fusion_units.csv` and `fusion_natives.csv` list the per-item outcomes of every task-2 row.
- `changes.csv` lists the fixed and newly wrong items of every comparison.
- `grid_calibration.csv` holds the 40 calibration settings of E0, B7 and q.

The server runs are `runs/20261005-182146-ev9v_complementarity_audit-f6478e28-s0` for the audit,
`runs/20261005-183150-ev9v_complementarity_fusion-f6478e28-s0` for the fusion and `runs/ev9v_complementarity/b7_spot_check`
for the spot re-run. The logs are `logs/ev9v_audit.log`, `logs/ev9v_fusion.log` and `logs/ev9v_b7_spot.log`. The
round-4 `units.csv` was read from the committed copy, which is byte-identical to the round-4 run's file (sha256
9153814c…).
