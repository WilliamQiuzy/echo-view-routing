# A18: a three-model A08 probability teacher for one video student (2026-10-21)

**Question.** Can the equal-weight probability average of the three frozen A08 models turn their prediction differences
into a stronger single video model? The average is used as the KL target in A10 R's stable continuation recipe.
- The primary comparison is the student E against A10's frozen single-teacher continuation R. Both are one ordinary CLS
  video model plus B3.
- The three-model average is also evaluated as a system of its own, to separate whether the teacher carries a usable
  increment from whether the student learns it.
- Knowledge distillation and probability averaging are known methods, and no algorithmic contribution is claimed. R09's
  old fusion gave no gain, and its student was never trained.
- The development halves were reused for exploratory feedback. The test split and EchoNet-Pediatric were not read.
- Configuration: `configs/experiments/video_ensemble_teacher_v1.yaml` (1dc26f7), fixed before any A18 output.

**Design.**
- **Student E.** Each seed's A08 all_s{k} is loaded strictly into the unchanged TargetMViT. E follows A10 R exactly:
  A10's pinned 12-window batches (rows 0–1,349 of 1,500 stored), its optimiser layout (the unused head group kept), its
  schedule, its seeds and its dropout. The only change is the KL target.
- **R's target and E's target.** R's target is the student's own A08 start model. E's target is q, the average of the
  three A08 models' softmax probabilities on the same window request, computed stably and checked against direct
  averaging.
- **Teacher cache.** Each teacher's logits for every update and window were computed beforehand from the same loader
  batches, digest-checked, so no teacher runs during training.
- **Control R.** A10's frozen R_s0/R_s2 with their pinned weights, outputs and committed choices, reproduced row by row.
  It was not retrained.
- **Choices.**
  - Each E got one B3 choice on calibration, from the same 40 settings and feasibility rule. The ensemble got one choice
    the same way.
  - The choices were committed (65fc207) after the weights (964ded4) and before selection was read.
- **Checks passed.**
  - Cached against live teacher logits: difference 0.0.
  - q is non-negative, sums to 1 and matches direct averaging to within 2e-7.
  - The cached own-teacher path matches A10's live `train_step`. At updates 0–1 the parameter difference is 2e-9. After
    unfreezing it is 5.9e-6, against a repeat difference of 5.7e-6, with equal CUDA random states.
  - Three identical teachers reduce to one teacher.
  - The optimiser is bound correctly, encoder gradients appear after unfreezing, and an exact resume holds.
- **Training-window diagnostic** (descriptive only).
  - The three members disagree on the top class in 2.1–2.3% of the 16,200 training windows per seed (3.8–4.6% in the
    change group).
  - q's top class differs from the student's own teacher in only 0.8%.
  - So on training data the new target is very close to the old one.

**Seed 1.** After seeds 0 and 2, the fixed promotion rule against each seed's A08 + B3 was met:
- Mean +0.0033, half means +0.0046 and +0.0019, and no cell below −0.0007.
- Native −0.0018, boundary recall +0.0052, and 15 missed short true segments against Base's 22.
- Both seeds feasible on calibration.

So seed 1 was built in A18's namespace by the fixed rule.
- A new R_s1 used A10 R's recipe, with the cached own-teacher target verified above.
- E_s1 used the same three teachers.
- Their choices were committed (a22e543) before the seed-1 selection.

## Results (three seeds; six cells)

| System | Mean F | − Base | Selection FP (short) | FN, six cells (short) | Boundary recall / precision | Native macro-F1 |
|---|---|---|---|---|---|---|
| Base (A08 A + B3) | 0.9704 | – | 75 (33) | 74 (32) | 0.980 / 0.966 | 0.985 |
| frozen Ordered + B3 | 0.9680 | −0.0023 | 76 (40) | 87 (41) | 0.975 / 0.966 | 0.982 |
| R + B3 (single teacher) | 0.9713 | +0.0009 | 79 (43) | 64 (25) | 0.983 / 0.963 | 0.987 |
| **E + B3 (three-model teacher)** | 0.9718 | +0.0015 | 72 (34) | 67 (27) | 0.981 / 0.966 | 0.983 |
| Ensemble + B3 (three forwards, one system) | 0.9731 | +0.0027 | 69 (27) | 60 (24) | 0.986 / 0.967 | 0.986 |

The ensemble is one system per half, three video forwards per window. Its row repeats the same two half results under
each seed and is never counted as separate cells. Its count columns are those two results counted three times, so that
they compare with the per-seed sums.

| Cell | Base | R + B3 | E + B3 | E − R [interval] |
|---|---|---|---|---|
| Seed 0 selection | 0.9563 | 0.9566 | 0.9557 | −0.0009 [−0.0089, 0.0081] |
| Seed 0 calibration | 0.9827 | 0.9846 | 0.9866 | +0.0020 [−0.0018, 0.0060] |
| Seed 1 selection | 0.9545 | 0.9545 | 0.9511 | −0.0034 [−0.0186, 0.0112] |
| Seed 1 calibration | 0.9885 | 0.9885 | 0.9875 | −0.0010 [−0.0079, 0.0038] |
| Seed 2 selection | 0.9564 | 0.9606 | 0.9663 | +0.0058 [0.0, 0.0120] |
| Seed 2 calibration | 0.9837 | 0.9828 | 0.9837 | +0.0009 [0.0, 0.0029] |

- **E − R, the primary comparison.** Mean +0.0006. There is no consistent difference in either half for segment F1,
  native macro-F1, boundary recall, false or missed segments.
  - E − Base (+0.0015) and R − Base (+0.0009) are also not consistent.
  - The two-seed signal (+0.0033 against Base) did not hold with seed 1. In seed-1 selection, E's chosen decoder misses
    more segments (25 against R's 19).
- **Native recognition.** E's native macro-F1 is 0.983 against R's 0.987.
  - E makes one extra A5C read as A4C in seed-0 calibration.
  - R_s1 fixes both native errors in seed-1 calibration (native 1.000).
- **Sensitivity** (fixed in advance, not used for any decision).
  - With each seed's A08 B3 setting, E − R is −0.0049, +0.0039, +0.0175, −0.0009, +0.0048 and 0.0 (mean +0.0034).
  - Plain argmax gives 0.9435 for E against 0.9383 for R and 0.9384 for A08.
  - So part of seed 1's loss comes from E's calibration-chosen decoder point. These readouts do not change the verdict.
- **Against A14 P and A17 O-fixed.** These exist for seeds 0 and 2 only.
  - E − A14 P is −0.0083, +0.0011, +0.0099 and 0.0 (mean +0.0007).
  - E − A17 O-fixed is −0.0055, +0.0030, +0.0058 and 0.0 (mean +0.0008).
  - Neither is a consistent difference, so E is not shown to beat A14 P.

**The teacher itself.**

| Half | A08 all_s0 | A08 all_s1 | A08 all_s2 | Ensemble + B3 |
|---|---|---|---|---|
| Selection | 0.9563 | 0.9545 | 0.9564 | 0.9625 |
| Calibration | 0.9827 | 0.9885 | 0.9837 | 0.9837 |

- **Selection.** The average beats each member by +0.006 to +0.008. Only the all_s1 interval reaches a lower bound of
  0.0: +0.0080 [0.0, 0.0187].
- **Calibration.** It is equal to all_s2, +0.0010 against all_s0, and −0.0048 against all_s1.
- So the teacher carries a modest increment, mainly on selection, at three times the forward cost.

## Did the student learn the teacher? (after selection; descriptive)

- **No movement toward the average.**
  - Each start model, a member of the average, sits at a window L1 distance of 0.017–0.022 from it.
  - After continuation, both students moved further away, to 0.042–0.047 for R and 0.040–0.043 for E. E is only
    0.001–0.003 closer than R.
  - Top-class agreement with the average is 0.992–0.997 for Base, R and E alike.
- **The average does not fix shared errors.**
  - On selection, 15 of 502 reference segments are missed by all three members, and the average recovers none of them.
  - It keeps all 11 segments that one or two members match. On calibration it keeps 4 of 7 and recovers neither of the 2
    shared misses.
  - So its gain comes from not dropping segments that some member finds.
- **The students differ.**
  - E_s1 and E_s2 recover 3 and 2 of the shared selection misses.
  - But E_s1 also loses 9 segments that all three members matched.

## What this round shows

1. **The three-model teacher provides a modest usable signal.** It is ahead of each member on selection and about level
   on calibration. The gain comes from keeping segments some members find, not from fixing errors all members share. On
   training windows it differs little from each student's own teacher.
2. **No full-segmentation gain over the matched single-teacher control is established.** E − R is +0.0006 over three
   seeds, with no consistent difference in any metric. The two-seed promotion signal was not confirmed by seed 1.
3. **The student did not move toward the teacher.** Continued training moved both students about equally far from the
   average.
4. **The single model does not beat A14 P.** On the two seeds where both exist, it is +0.0007 with mixed cells.
   - The ensemble's mean (0.9731) is a three-forward system, not a single-model result.
   - A14 P remains the highest single-pipeline point estimate on seeds 0 and 2, and A08 A + B3 the working reference.
   - This shows no gain for this one recipe. It does not show that distillation is ineffective in general.

## Cost and files

- **GPU.** 2.08 of 4 h.
  - Teacher caches: 0.59 h, plus one failed 0.02 h start.
  - Checks: 0.02 h.
  - Four students (E_s0, E_s2, R_s1, E_s1): 1.01 h.
  - Evaluation: 0.42 h.
- **CPU.** 0.95 h of 3 at the final commit, mostly the agent's active wall clock. Following A10/A12, the loader
  workers inside GPU jobs are covered by the GPU wall time.
- **Deployment.**
  - The student is one model of 34,640,549 parameters with one forward per window. It needs about 124 s per selection
    half (7,666 windows) on one L40S.
  - The ensemble needs three forwards per window.
- **Records.**
  - `cache_index.json`, `training_diagnostic.json`, `checks.json` and `checks_seed1.json`.
  - `train/` (run records with weights sha), `choices.json` and `choices_seed1.json` (both committed before selection).
  - `screen_two_seeds.json` with `decision_two_seeds.json`, then `screen_three_seeds.json` and `diagnostic.json`.
- **Frozen models.** `models_frozen/video_ensemble_teacher_v1/{E_s0,E_s1,E_s2,R_s1}`.
- **Server only.** Teacher caches, development outputs and all per-sequence data.
