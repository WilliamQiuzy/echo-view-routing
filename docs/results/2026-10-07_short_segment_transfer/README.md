# Short-segment discriminators on the validation halves: a development-set transfer check (2026-10-07)

This check asked whether the simple rule learned by the short-segment discriminators still works on recordings that
did not take part in fitting them.

**What the probe did.** In a decoded pattern X → Y → X, the probe's discriminators decide whether the middle run is a
correct short view segment (keep it) or a wrong prediction inside one view (remove it). Q0 reads the two classes,
lengths, L1 probabilities and B3 boundary scores. Q1 adds frozen L1 visual features and two cosine similarities.

**What this check did.**
- Both models were frozen and applied unchanged to the two validation halves, with the probe's original thresholds.
- Both halves served only for evaluation; neither chose any threshold.

**What kind of evidence this is.** This is a development-set transfer check. The validation recordings took no part
in fitting Q0 and Q1. They also took no part in training L1, which used only training recordings and kept its final
update. But the project has used them repeatedly, and E1's historical choices involved them. The candidate generator,
J0 with κ = 8, was itself chosen on the calibration half in the joint decoding round. So this is not an independent
validation and not an unseen-patient result. Nothing was integrated into the full system, and the test split stays
sealed.
- The plan is `docs/superpowers/plans/2026-10-07-short-segment-transfer.md`.
- The configuration, code and tests were committed before any validation candidate was built (e92edf4).
- An independent fact-check followed the results. A reporting step (`--diagnose`) then added the exact intervals that
  the configuration promised for small subsets, and counts of the recordings that seeds, sources and middles share.
  It reads the saved counts and candidate lists and rescores nothing.

## Answers

1. **What was done.**
   - **Rebuild.** The probe had not saved its fitted classifiers. Q0 and Q1 were rebuilt with the committed code, the
     original training library and the stored C, without any search. This is a rebuild of existing models, not new
     training.
     - The rebuilt models reproduce the probe exactly in all six seed-model pairs: thresholds, held-out AUC and every
       held-out rate.
     - The fitted pipelines were saved with their sha256.
   - **Candidates.** The same generator (J0, κ = 8), candidate pattern, category rules and features ran on both
     validation halves: native clips, pair streams and multi-fragment streams.
     - The inputs came from the same L1_s{k} feature store as the training library.
     - The primary threshold is the probe's 95% calibration threshold, and the 90% threshold is secondary.
2. **The A4C/A5C pairs, the key case.** Q0 keeps fewer correct middles than the rule asks for.
   - **Correct middles.** Each seed has 13 correct A4C/A5C middles over the two halves. They match the same 13 true
     units in every seed, which cover 14 recordings of their own view from 12 sessions.
     - Q0 keeps 8, 9 and 9 of them and wrongly removes 5, 4 and 4 (seeds 0, 1, 2).
     - On the selection half alone it keeps 3 of 6 in every seed.
   - **Wrong middles.** Q0 removes 13 of 14, 16 of 18 and 21 of 22 wrong A4C/A5C middles. These come from 11, 12 and
     16 recordings. Over the three seeds, all 54 of them come from 17 recordings in 15 sessions.
   - **Secondary point.** At the 90% threshold, Q0 keeps only 2, 4 and 4 of the 13 correct middles.
3. **Other class pairs.** Q0 keeps every correct middle of the other pairs. These number 119, 118 and 116 and come
   from 126, 127 and 123 recordings. Q0 removes 4 of 5, 4 of 5 and 3 of 4 wrong ones.
4. **Overall.**
   - **Q0.** Its AUC is 0.992, 0.987 and 0.992 over both halves. It keeps 96.2–96.9% of all correct middles and
     removes 87.0–92.3% of all wrong ones.
   - **What the overall numbers mix.** Most correct middles belong to the other pairs (119 of 132, 118 of 131 and 116
     of 129), and Q0 keeps all of them. Most wrong middles are A4C/A5C (14 of 19, 18 of 23 and 22 of 26). So 66–76%
     of the comparisons behind the overall AUC set a correct middle of another pair against a wrong A4C/A5C middle.
     Within the A4C/A5C pair, Q0's AUC is 0.94, 0.88 and 0.95.
5. **Candidates outside the binary task.** These are 25, 21 and 27 of 176, 175 and 182 candidates (12–15%), and
   16.5–21.8% on the selection half. In nearly all of them (23 of 25, 20 of 21 and 24 of 27), at least one outer run
   has a reference view other than the class the decoder gave it. They are scored and counted, but the rates and AUCs
   above leave them out.
6. **Sources and joins.**
   - **Correct middles.** Every correct middle lies in a multi-fragment stream at a view-change join. Inside a stream
     only such a join changes the reference view, and a true middle needs a change on each side.
   - **Wrong middles.** Native clips give 11, 12 and 16 of the 19, 23 and 26 wrong middles, from 10, 10 and 13
     clips. Of the wrong middles in streams, 4, 6 and 6 sit at a verified same-view join, 3, 4 and 4 lie away from
     any join, and 1, 1 and 0 sit at a view-change join.
   - **The same-view control.** At a verified same-view join, both sides of the join have the same reference view. Q0
     removes 3 of 4, 4 of 6 and 5 of 6 wrong middles there. These counts are too small to measure a splice effect.
7. **Q1 against Q0.** Q1 did not meet the rule for extra value, and no consistent difference from Q0 was detected.
   - **Removal.** Over both halves, Q1 − Q0 for the removal rate of wrong middles is 0.0, +8.7 and +3.8 points. No
     interval lies above zero.
   - **Keep.** Q1 keeps 1.5, 1.5 and 2.3 points fewer correct middles. Every interval includes zero.
   - **A4C/A5C.** Q1 removes every wrong A4C/A5C middle but keeps fewer correct ones (7, 7 and 8 of 13).

**Decision.** The fixed rule's case is "Q0 high overall but removes correct A4C/A5C middles". Under the plan, Q0 is
not suited to the key problem and is not integrated. Q1 did not meet the rule for extra value. Neither discriminator
meets the failure rule, so this check does not call for stopping them on transfer grounds.

**How firm the A4C/A5C result is.** Q0 keeps 8 or 9 of the 13 correct A4C/A5C middles, below the rule's keep rate of
0.90 in every seed. The 13 are the same true units in every seed, so the seeds add no new cases. With 13 cases the
uncertainty is wide.
- For 9 of 13, the exact 95% interval is 0.39–0.91, which includes 0.90. For 8 of 13 it is 0.32–0.86.
- These intervals treat the middles as independent. Their own-view recordings come from only 12 sessions, and one
  recording serves two of them, so the intervals are likely too narrow.

The thresholds of this rule (for example a 0.90 keep rate for A4C/A5C) are my operationalisation of the plan's
qualitative branches, fixed before the check (section 7). Every count is reported so they can be read differently.

## Terms

- **Correct middle (a):** the middle matches a true reference unit of its class with IoU of at least 0.5, and both
  outer runs' reference majority is the outer class.
- **Wrong middle (b):** every reference label inside the middle is the outer class, and both outer runs' reference
  majority is the outer class.
- **Outside the binary task (c):** everything else. These candidates are scored, but no rate or AUC includes them.
- **Kept:** the discriminator's score is at or above the threshold.
- **Wrongly removed:** a correct middle scored below the threshold.
- **Missed:** a wrong middle scored at or above the threshold.
- **Same-view join:** a recording join within 5 samples of the middle where the reference view is the same on both
  sides of every such join. At a **view-change join**, at least one join within 5 samples changes the view.
- **Pooled halves:** the selection and calibration halves together. They hold different sessions.

## 1. Rebuild of the frozen models

| Seed | Model | C | Threshold difference | AUC difference | Held-out rates | Reproduced |
|---|---|---|---|---|---|---|
| 0 | Q0 / Q1 | 10 / 1 | 0 / 0 | 0 / 0 | identical / identical | yes / yes |
| 1 | Q0 / Q1 | 10 / 0.1 | 0 / 0 | 0 / 0 | identical / identical | yes / yes |
| 2 | Q0 / Q1 | 10 / 1 | 0 / 0 | 0 / 0 | identical / identical | yes / yes |

The probe chose C by cross-validation that grouped stream candidates by constructed stream and native candidates by
session. A recording reused in several streams can cross those folds, so the probe's cross-validation scores may be
optimistic. This does not invalidate its held-out evaluation, which uses separate sessions. C was not re-chosen here.

## 2. A4C/A5C first (primary threshold)

The table gives counts per seed and half.

| Seed | Half | Correct: Q0 kept / removed | Correct: Q1 kept / removed | Correct: recordings / sessions | Wrong: Q0 removed / missed | Wrong: Q1 removed / missed | Wrong: recordings / sessions |
|---|---|---|---|---|---|---|---|
| 0 | selection | 3 / 3 | 2 / 4 | 9 / 9 | 9 / 0 | 9 / 0 | 7 / 6 |
| 0 | calibration | 5 / 2 | 5 / 2 | 9 / 7 | 4 / 1 | 5 / 0 | 4 / 4 |
| 0 | both | 8 / 5 | 7 / 6 | 18 / 16 | 13 / 1 | 14 / 0 | 11 / 10 |
| 1 | selection | 3 / 3 | 2 / 4 | 9 / 9 | 12 / 1 | 13 / 0 | 7 / 6 |
| 1 | calibration | 6 / 1 | 5 / 2 | 9 / 7 | 4 / 1 | 5 / 0 | 5 / 5 |
| 1 | both | 9 / 4 | 7 / 6 | 18 / 16 | 16 / 2 | 18 / 0 | 12 / 11 |
| 2 | selection | 3 / 3 | 3 / 3 | 9 / 9 | 16 / 0 | 16 / 0 | 11 / 9 |
| 2 | calibration | 6 / 1 | 5 / 2 | 9 / 7 | 5 / 1 | 6 / 0 | 5 / 5 |
| 2 | both | 9 / 4 | 8 / 5 | 18 / 16 | 21 / 1 | 22 / 0 | 16 / 14 |

- **Middles and recordings.** The table counts every recording under a middle. Five correct A4C/A5C middles cover two
  or three recordings, and in four of them the decoded middle runs past its true unit into a fragment of another view.
  So the 13 middles count 18 recordings in 16 sessions, though they cover only 14 recordings of their own view.
- **The seeds share recordings.** The three seeds decode the same validation recordings with different L1 models. The
  13 correct A4C/A5C middles match the same 13 true units in every seed. The 54 wrong A4C/A5C middles of the three
  seeds come from 17 recordings in 15 sessions.
- **Native clips and streams share recordings.** Every stream fragment is also a native clip of the same half. Of the
  wrong A4C/A5C middles in streams, 4 of 7, 8 of 10 and 6 of 9 lie on a recording that also gives a wrong A4C/A5C
  middle as a native clip.
- **AUC within the A4C/A5C pair, both halves.** Q0's is 0.940, 0.876 and 0.951, and Q1's is 0.951, 0.970 and 0.944.
- **Exact 95% intervals, both halves.** Q0 keeps 8 of 13 (0.32–0.86), 9 of 13 (0.39–0.91) and 9 of 13 (0.39–0.91),
  and removes 13 of 14 (0.66–1.00), 16 of 18 (0.65–0.99) and 21 of 22 (0.77–1.00). Section 5 explains why these
  are likely too narrow.

## 3. Other class pairs (primary threshold, both halves)

| Seed | Correct: Q0 kept / removed | Correct: Q1 kept / removed | Correct: recordings / sessions | Wrong: Q0 removed / missed | Wrong: Q1 removed / missed | Wrong: recordings / sessions |
|---|---|---|---|---|---|---|
| 0 | 119 / 0 | 118 / 1 | 126 / 51 | 4 / 1 | 3 / 2 | 5 / 4 |
| 1 | 118 / 0 | 118 / 0 | 127 / 52 | 4 / 1 | 4 / 1 | 5 / 4 |
| 2 | 116 / 0 | 114 / 2 | 123 / 51 | 3 / 1 | 3 / 1 | 4 / 3 |

**Overall AUC, both halves.** Q0's is 0.992, 0.987 and 0.992, and Q1's is 0.947, 0.966 and 0.963. Per half, Q0's AUC
is 0.975–0.988 on the selection half and 0.996–0.998 on the calibration half.

## 4. Candidates outside the binary task

| Seed | Half | Candidates | Outside | Share | An outer run not the outer class | A third class under the middle | Part of a true unit (IoU < 0.5) |
|---|---|---|---|---|---|---|---|
| 0 | selection | 87 | 19 | 21.8% | 17 | 1 | 1 |
| 0 | calibration | 89 | 6 | 6.7% | 6 | 0 | 0 |
| 1 | selection | 85 | 14 | 16.5% | 13 | 0 | 1 |
| 1 | calibration | 90 | 7 | 7.8% | 7 | 0 | 0 |
| 2 | selection | 92 | 20 | 21.7% | 18 | 1 | 1 |
| 2 | calibration | 90 | 7 | 7.8% | 6 | 1 | 0 |

- **How a reason is assigned.** The reasons are tested in the order of the columns, and the first that applies is
  recorded. No candidate was assigned the fourth reason, a change of reference view under the middle.
- **A view change under the middle.** One outside candidate per seed does have two reference views under its middle.
  One of its outer runs is also wrong, so it counts under the first reason. Seven correct middles per seed also cover
  two views, because the decoded middle runs past its true unit into a neighbouring fragment. The IoU rule allows
  this.
- **Coverage of the rates.** These candidates are scored, but no rate or AUC includes them. So the rates cover 85–88%
  of the candidates, not every possible error.

## 5. Sources and joins (diagnostics, both halves, primary threshold)

| Subset | Correct middles: Q0 kept / n (s0, s1, s2) | Wrong middles: Q0 removed / n (s0, s1, s2) | Wrong middles: Q1 removed / n |
|---|---|---|---|
| Native clips | none | 10/11, 11/12, 15/16 | 10/11, 12/12, 16/16 |
| Pair streams | none | 1/1, 2/2, 3/3 | 1/1, 2/2, 3/3 |
| Multi-fragment streams | 127/132, 127/131, 125/129 | 6/7, 7/9, 6/7 | 6/7, 8/9, 6/7 |
| No join | none | 13/14, 15/16, 19/20 | 13/14, 16/16, 20/20 |
| Verified same-view join | none | 3/4, 4/6, 5/6 | 3/4, 5/6, 5/6 |
| View-change join | 127/132, 127/131, 125/129 | 1/1, 1/1, 0/0 | 1/1, 1/1, 0/0 |
| Middle 10–40 samples | 102/107, 103/107, 100/104 | 7/8, 12/14, 11/12 | 7/8, 13/14, 11/12 |

- **Correct middles** occur only in multi-fragment streams. A pair stream has two fragments, so it cannot hold an
  X → Y → X pattern with a true middle.
- **The same-view control** holds 4, 6 and 6 wrong middles. The counts are too small to measure a splice effect.
- These are diagnostics, not a way to pick a winner. No extra streams were built to enlarge any subset.

**Exact 95% intervals for Q0's removal of wrong middles.** The config asked for intervals on small subsets. These are
Clopper–Pearson intervals. They treat the middles as independent, which they are not, so they are likely too narrow.

| Subset | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Native clips | 10/11 (0.59–1.00) | 11/12 (0.62–1.00) | 15/16 (0.70–1.00) |
| Pair streams | 1/1 (0.03–1.00) | 2/2 (0.16–1.00) | 3/3 (0.29–1.00) |
| Multi-fragment streams | 6/7 (0.42–1.00) | 7/9 (0.40–0.97) | 6/7 (0.42–1.00) |
| No join | 13/14 (0.66–1.00) | 15/16 (0.70–1.00) | 19/20 (0.75–1.00) |
| Verified same-view join | 3/4 (0.19–0.99) | 4/6 (0.22–0.96) | 5/6 (0.36–1.00) |
| View-change join | 1/1 (0.03–1.00) | 1/1 (0.03–1.00) | none |
| Middle 10–40 samples | 7/8 (0.47–1.00) | 12/14 (0.57–0.98) | 11/12 (0.62–1.00) |

Q0 keeps 127 of 132, 127 of 131 and 125 of 129 correct middles in multi-fragment streams (0.91–0.99, 0.92–0.99 and
0.92–0.99). `diagnostics.json` holds the intervals for both models and both kinds of middle.

## 6. Q1 against Q0 (both halves, primary threshold)

| Seed | Removal of wrong middles, Q1 − Q0 (points) | Keep of correct middles, Q1 − Q0 (points) | A4C/A5C removal, Q1 − Q0 |
|---|---|---|---|
| 0 | 0.0 [−14.3, +14.3] | −1.5 [−4.6, +1.4] | +7.1 |
| 1 | +8.7 [0.0, +21.4] | −1.5 [−4.5, +1.5] | +11.1 |
| 2 | +3.8 [0.0, +12.5] | −2.3 [−6.5, +1.6] | +4.5 |

The intervals resample streams for stream candidates and sessions for native candidates (2,000 draws). They are likely
too narrow, because streams reuse recordings that are also native clips, and the thresholds are held fixed.

## 7. Decision under the fixed rule

The plan's branches were made checkable before the check. Per seed the two halves are pooled, and "most seeds" means
at least two of three.

| Branch | Rule | Met |
|---|---|---|
| Both fail | in most seeds, AUC < 0.80, or keep rate < 0.90, or removal rate < 0.50 (for both models) | no |
| Q1 extra value | in every seed: Q1 − Q0 removal interval above zero, Q1 keep rate at most 2 points lower, A4C/A5C removal not lower | no |
| Q0 practical value | in most seeds: keep ≥ 0.90 and removal ≥ 0.50 overall, and the same for A4C/A5C | no (A4C/A5C keep 0.62–0.69) |
| Q0 high overall, removes correct A4C/A5C | in most seeds: AUC ≥ 0.95, keep ≥ 0.95, removal ≥ 0.80, and A4C/A5C keep < 0.90 | **yes** |

**Q0's pooled rates per seed.** Its AUC is 0.992, 0.987 and 0.992. Its overall keep rate is 0.962, 0.969 and 0.969,
and its overall removal rate is 0.895, 0.870 and 0.923. For A4C/A5C, its keep rate is 0.615, 0.692 and 0.692, and its
removal rate is 0.929, 0.889 and 0.955.

**Outcome.** Q0 is not suited to the current key problem and is not integrated into the full system. Q1 did not meet
the rule for extra value. This check does not show that either discriminator would beat B3 or ASRF in the full system.

## 8. What this does not show

- **Independent evidence.** Both halves have been used repeatedly in development, E1's historical choices involved
  them, and the generator's κ was chosen on the calibration half.
- **Robust A4C/A5C estimates.** Each seed has 13 correct and 14–22 wrong A4C/A5C middles. The correct ones match the
  same 13 true units in every seed, from 12 sessions. The wrong ones of all three seeds come from 17 recordings in 15
  sessions, and many stream errors repeat a recording that also errs as a native clip. So the seeds and sources do not
  add independent cases.
- **Whether visual features could help.** Q1 is the probe's model, trained on a library whose correct and wrong
  candidates differ in source, length, confidence and contrast editing. This check freezes it rather than redesigning
  it.
- **Errors the binary task cannot see.** The 12–15% of candidates outside it, mostly with a wrong outer run, enter no
  rate.

## 9. Cost

- **Rebuild.** Refitting six logistic regressions and checking them took about 12 seconds on the CPU.
- **Application.** Both halves for three seeds took 48 seconds on the CPU.
- **Reporting step.** `--diagnose` took about 4 seconds on the CPU.

## Files

- `rebuild.json` holds the reproduction check of every rebuilt model and the sha256 of its saved pipeline.
- `summary.json` holds, per seed and half, the tables of sections 2–7 at both thresholds, the AUCs, the Q1 − Q0
  bootstrap and the decision inputs and outcome.
- `report_tables.md` holds the generated tables.
- `diagnostics.json` holds the exact intervals and the counts of shared recordings in sections 2, 4 and 5.
- None of these files contains recording or session ids. The candidate lists and the pickled pipelines stay on the
  server, in `runs/short_segment_transfer_v1/`.
