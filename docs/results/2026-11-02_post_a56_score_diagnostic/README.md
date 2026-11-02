# POST-A56-ROC — Retrospective low-FPR ranking diagnostic

**The old zero operating point was not the only limitation.** Even an optimistic label-assisted threshold on the repeatedly inspected133source requests reaches at most24/39A5C successes at≤5/94A4C errors among these frozen models. A56 Z/R reach23/22, below the old28 requirement. This retrospective upper envelope is not calibration, deployment, a rescued round or an A57 gate.

| Frozen score | Original TP / FP | AUC | MaxTP at FP≤0 | ≤2 | ≤5 | ≤10 | ≤20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| G-F | 16 / 4 | 0.807420 | 9 | 15 | 23 | 24 | 28 |
| G-C | 16 / 2 | 0.814512 | 9 | 16 | 21 | 25 | 27 |
| SF | 31 / 28 | 0.803601 | 2 | 2 | 10 | 23 | 30 |
| SC | 29 / 15 | 0.871795 | 5 | 16 | 24 | 28 | 31 |
| DF | 29 / 19 | 0.811238 | 1 | 2 | 15 | 27 | 29 |
| DC | 30 / 14 | 0.877523 | 5 | 18 | 23 | 29 | 31 |
| Z | 30 / 19 | 0.892799 | 12 | 22 | 23 | 26 | 31 |
| R | 30 / 20 | 0.879978 | 11 | 21 | 22 | 26 | 30 |

All historical strict-zero TP/FP/BA/AUC values and exact133-request label/patient/video order were reproduced before the new ROC arithmetic. A55 uses each scalar head's own score; A56 uses original-classifier raw q3−q2. Absolute score scales are not compared. Tied scores move together; ±infinity endpoints are retained and no fractional interpolation is used. Independent sklearn AUC and complete ROC budget recount agree. Attained FP and original-error patient concentration are in [summary.json](summary.json). Thresholds, raw scores, IDs and private curves remain server-only.

The count field named `admissible_thresholds` in the frozen result is a finite enumeration of candidate endpoints, including a duplicate all-negative origin (+infinity and maximum observed score). It must not be interpreted as infinitely many real-valued cutoffs or as strictly deduplicated prediction-state counts. The subsequent card clarification concerns this wording; the frozen counts/maxima were preserved and no ROC was rerun. This endpoint duplication does not change any maximumTP or attainedFP.

The cohort comprises39A5C requests from6patients/7videos and94A4C requests from10patients/11videos,13videos total. Requests are correlated and the10patients have been reused repeatedly. Searching all thresholds with these labels gives an optimistic, reference-assisted envelope, not held-out or clinical performance. Neither A55 nor A56 is reopened; no calibrator, new model, source, image decode or protected split was used. A57 chooses its own target-task rules inside its training/calibration partition.

Code/card actualGit freeze:5e3e723 before private arithmetic. Three focused tests cover strict ties, endpoints/budgets, known AUC and score-scale invariance. The successful worker used1.422175measured CPU-core seconds, GPU0, under20CPUminutes/100MB. One identity preflight stopped before scores opened because the already-frozen test file had not been transferred; copying that unchanged file repaired the preflight. Its CPU was not measured and is conservatively bounded by4seconds separately; no neural operation or ROC arithmetic occurred in that failed attempt. [verification.json](verification.json) preserves this disclosure.
