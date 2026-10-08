# A session-fold prediction library from encoders that did not train on the sessions (2026-10-07)

This round asked one question. When the visual model that makes the predictions never trained on a recording's
session, does the EV9V training split yield enough dispersed A4C/A5C errors, and enough true short segments that a
later corrector must keep?

**What kind of evidence this is.**
- Only the EV9V training split was used. The validation halves and the test split took no part in training, model
  selection or evaluation.
- The hold-out is by session, not by patient.
- The constructed streams are not real continuous scans, and their joins are artificial.
- No segment-level discriminator was trained. Building this library is not a model contribution.
- Public labels are unchanged, and no recording was filtered by its predictions.

**Order of work.**
- The plan is `docs/superpowers/plans/2026-10-07-session-fold-library.md`.
- The configuration, code and tests were committed before any fold was written or any model trained (764968f).
- The fold composition was committed before training (8ae3c5b).
- Table 1 (recognition) was committed before any short-segment candidate was built (f6189c1).
- An independent fact-check followed the results. A reporting step (`--diagnose`) then added the own-view counts of
  the correct middles from the saved candidate lists. Nothing was rebuilt.

## Answers

1. **What was done.**
   - **Folds.** The 3,364 training clips (502 sessions) were split into three session folds by a fixed rule. The rule
     was chosen among several by the fold composition alone, before any prediction existed. All five classes' clip
     counts differ by at most one between folds. Sessions with A4C number 134–136 per fold, and sessions with A5C
     45–48.
   - **Models.** Three B0 models (ResNet-18 with a five-class head) were trained from ImageNet, one per fold, each on
     the other two folds. They used the B0 v3 recipe for exactly 30 epochs, and the last weights were kept. No data
     was evaluated during training, and nothing was selected.
   - **Predictions.** Each model predicted only the fold it never trained on, at 10 Hz, with and without the gamma
     edit. So each of the 3,364 training clips now has a prediction from a model that did not see its session.
   - **Frozen.** The three models are frozen on the server as `models_frozen/session_fold_b0_v1/fold{0,1,2}`.
2. **Table 1, recognition.** The main A4C/A5C error is A5C read as A4C, and it is spread over many sessions.
   - **A5C read as A4C.** 90 of 217 A5C clips (41%) are voted A4C, in 66 sessions. At the sample level, 5,046 of
     12,102 A5C samples are read as A4C, and 182 of the 217 A5C clips (in 122 sessions) have at least one such sample.
   - **A4C read as A5C.** Only 6 of 858 A4C clips are voted A5C, in 6 sessions. At the sample level, 1,569 of 46,022
     A4C samples are read as A5C, spread over 235 clips in 169 sessions.
   - **Other classes.** The other classes are read almost perfectly. At the clip level 1,117 of 1,120 PLAX, all 974
     PSAX and 194 of 195 SC4C clips are correct.
3. **Table 2, the wrong A4C/A5C fragments.** Over the three folds there are 419 wrong A4C/A5C fragments, from 150
   recordings in 116 sessions. They are spread over many recordings.
   - When a fragment over k recordings counts 1/k for each, the largest share of any recording is 2.5%, and the
     effective number of recordings is 85.
   - Within one fold the largest share is 5.4–8.8%, and the effective number is 24–33.
   - Streams reuse recordings, so up to 12 wrong fragments involve the same recording.
   - Most (313) are a short A4C run inside an A5C reference region. The other 106 are a short A5C run inside A4C.
4. **Table 2, the correct A4C/A5C short segments.** There are only 42 over the three folds (21, 7 and 14 per fold).
   - They come from 49 recordings of their own view, in 43 sessions.
   - Half of them (21) also run into a neighbouring fragment, so counting every recording under them gives 66
     recordings in 57 sessions.
   - Over all class pairs there are 598 correct middles, from 639 recordings of their own view in 319 sessions.
5. **Table 2, where the correct and wrong A4C/A5C candidates sit.** This matches the limitation the plan asked to
   report. The tables do not test whether splice position is the main separator, and length and confidence differ
   too.
   - **Splice position.** Every correct middle lies in a constructed stream at a join where the view changes. This is
     expected from the construction. A native clip has one reference view, and inside a stream only a join between
     fragments of different views changes the reference view. The join test allows 5 samples on either side of the
     middle. Of the 419 wrong ones, 243 lie away from any join, 126 at a verified same-view join and 50 at a
     view-change join.
   - **Source.** All 42 correct middles come from streams. Of the wrong ones, 148 come from native clips and 271 from
     streams.
   - **Length.** The correct middles are longer, with a median of 21.5 samples (quartiles 17.25 and 32.25). The wrong
     ones have a median of 8 (quartiles 5 and 13.5).
   - **Confidence.** The mean probability of the decoded middle class has a median of 0.981 for correct middles and
     0.933 for wrong ones.
   - **Contrast.** The edits are balanced by design. About half of the native clips (1,713 of 3,364) and half of the
     stream fragments (4,402 of 8,875) were edited. Among the wrong A4C/A5C middles, 185 are unedited, 215 edited and
     19 span both. A middle can span both only in a stream.
6. **Many A4C/A5C candidates fall outside the binary task.** There are 443 of them, more than the 419 wrong ones.
   - In 434 of them, the reference majority of at least one outer run is not the decoded outer class.
   - Of these 434, 373 are decoded as A4C → A5C → A4C.
   - Over all pairs, 648 candidates are outside the task, 597 of them for this reason. They are counted but belong to
     neither category.

**What the tables mean for the plan's next step.** The A4C/A5C errors are not concentrated on a few recordings. The
correct A4C/A5C short segments are few, and every one of them sits at a view-change join of a constructed stream.
Only 50 of the 419 wrong fragments (12%) do. More streams would add correct middles only at further artificial
view-change joins. This round makes no decision; the plan leaves the next step to the lead.

## Terms

- A **clip** is one EV9V recording; the two words name the same unit here. A **native clip** is a clip on its own.
- A **constructed stream** joins 4–8 fragments of different clips of one fold. Each fragment is 10–40 samples long.
- The **gamma edit** (gamma090) applies a gamma of 0.9, which slightly brightens mid-tones.
- **J0** decodes each sequence into runs of one class. With **κ = 8**, each change of class costs 8 in log
  probability.
- A **candidate** is three consecutive decoded runs X → Y → X. Y is the **middle**, and the two X runs are the **outer
  runs**. A **reference unit** is a run of one reference view.
- A **correct middle (a)** matches a reference unit of its class with IoU of at least 0.5, and both outer runs'
  reference majority is the outer class. These are the true short segments a corrector must keep. Here they exist
  only at the artificial joins of constructed streams.
- In a **wrong middle (b)**, every reference label inside the middle is the outer class, and both outer runs'
  reference majority is the outer class. These are the wrong fragments.
- Every other candidate lies **outside the binary task (c)**.
- A **view-change join** is a recording join within 5 samples of the middle that changes the reference view. At a
  **verified same-view join**, every such join keeps the reference view.
- The **largest recording share** counts a candidate as 1/k for each of its k recordings. The **effective number of
  recordings** is 1 divided by the sum of the squared shares.
- The **confidence** of a middle is the mean probability of its decoded class over the middle.

## 1. Folds and models

| Fold | Sessions | Clips | A4C clips | A5C clips | Sessions with A4C | Sessions with A5C | Training clips | Final training loss | Training minutes |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 163 | 1,120 | 286 | 72 | 134 | 45 | 2,244 | 0.00009 | 15.2 |
| 1 | 171 | 1,122 | 286 | 72 | 135 | 46 | 2,242 | 0.00047 | 15.0 |
| 2 | 168 | 1,122 | 286 | 73 | 136 | 48 | 2,242 | 0.00032 | 14.9 |

- **Same start.** The three models start from the same initial weights (same initial-weight digest, seed 0). The
  folds are three training sets, not three seeds.
- **Fold rule.** scikit-learn's `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=20261008)` over the
  clips sorted by id, with the class as stratum and the session as group.
- **Provenance.** `models.json` holds each model's checkpoint sha256, feature-cache key and training log summary.

## 2. Table 1, recognition (unedited input, every 10 Hz sample)

| Fold | Clip accuracy | Clip macro-F1 | A5C clips voted A4C (sessions) | A5C recordings with an A4C sample (sessions) | A5C samples read as A4C | A4C clips voted A5C (sessions) | A4C recordings with an A5C sample (sessions) | A4C samples read as A5C |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.972 | 0.941 | 25 of 72 (19) | 60 (39) | 1,514 of 3,416 | 2 of 286 (2) | 72 (56) | 533 of 15,296 |
| 1 | 0.969 | 0.927 | 33 of 72 (23) | 60 (42) | 1,978 of 4,174 | 0 of 286 (0) | 82 (58) | 437 of 14,520 |
| 2 | 0.964 | 0.924 | 32 of 73 (24) | 62 (41) | 1,554 of 4,512 | 4 of 286 (4) | 81 (55) | 599 of 16,206 |
| All | 0.968 | 0.931 | 90 of 217 (66) | 182 (122) | 5,046 of 12,102 | 6 of 858 (6) | 235 (169) | 1,569 of 46,022 |

**Correct clips per class, all folds.** PLAX 1,117 of 1,120, PSAX 974 of 974, A4C 847 of 858, A5C 126 of 217 and SC4C
194 of 195. A clip is voted by the majority of its per-sample predictions, with ties going to the lower class index.

**Per-sample recall, all folds.** PLAX 0.995, PSAX 0.990, A4C 0.959, A5C 0.578 and SC4C 0.991.

## 3. Table 2, candidates (J0, κ = 8, both contrasts)

Each cell gives the count, then its distinct recordings and sessions.

| Fold | Pairs | Correct middles (a) | Wrong middles (b) | Outside the binary task (c) |
|---|---|---|---|---|
| 0 | A4C/A5C | 21 (33, 26) | 175 (54, 39) | 190 (66, 42) |
| 1 | A4C/A5C | 7 (14, 14) | 124 (51, 41) | 139 (61, 46) |
| 2 | A4C/A5C | 14 (19, 17) | 120 (45, 36) | 114 (60, 39) |
| All | A4C/A5C | 42 (66, 57) | 419 (150, 116) | 443 (187, 127) |
| 0 | all pairs | 195 (216, 107) | 204 (67, 46) | 256 (126, 79) |
| 1 | all pairs | 189 (222, 114) | 136 (56, 44) | 208 (130, 76) |
| 2 | all pairs | 214 (225, 107) | 145 (61, 44) | 184 (140, 85) |
| All | all pairs | 598 (663, 328) | 485 (184, 134) | 648 (396, 240) |

The recording counts include every recording under a middle, including neighbouring fragments that a decoded middle
runs into. So correct middles can count more recordings than candidates. `diagnostics.json` gives the recordings of
the middle's own view.

**How the A4C/A5C categories differ (all folds).**

| Category | Native / stream | No join / same-view join / view-change join | Contrast unedited / edited / both | Middle length, quartiles | Confidence, quartiles | Largest recording share | Effective recordings |
|---|---|---|---|---|---|---|---|
| Correct (a), 42 | 0 / 42 | 0 / 0 / 42 | 15 / 18 / 9 | 17.25, 21.5, 32.25 | 0.923, 0.981, 0.998 | 4.8% | 49.7 |
| Wrong (b), 419 | 148 / 271 | 243 / 126 / 50 | 185 / 215 / 19 | 5, 8, 13.5 | 0.864, 0.933, 0.983 | 2.5% | 84.6 |
| Outside (c), 443 | 129 / 314 | 188 / 96 / 159 | 173 / 232 / 38 | 6, 11, 20 | 0.834, 0.914, 0.980 | 2.7% | 90.3 |

- **Recurring recordings.** The wrong A4C/A5C fragments in streams reuse recordings. Of the 118 recordings behind
  them, 71 also give a wrong fragment as a native clip. At most 12 wrong fragments involve the same recording.
- **Over all pairs.** The correct middles are long and confident, with a median length of 24 samples and a median
  confidence of 1.0. The wrong ones have a median length of 8 and a median confidence of 0.932.
- **Why candidates fall outside the task (all pairs).** 597 have an outer run whose reference view is not the decoded
  outer class. 42 have a third class under the middle, 5 a reference change under the middle, and 4 cover less than
  half of a true unit.

## 4. What this does not show

- **Errors of the system's own model.** These predictions come from B0 models trained from ImageNet on two thirds of
  the training split. They read 126 of 217 A5C clips correctly. Under the same per-sample argmax reading, L1 read 11,
  10 and 11 of the 11 A5C clips of the selection half and 17, 16 and 18 of the 18 of the calibration half in the
  fan224 round. The data differ, but the errors of these models may differ from the errors L1 makes on new recordings.
- **Unseen patients.** The hold-out is by session.
- **Real scans.** The streams are spliced from separate recordings. Their joins, and the view changes at them, are
  artificial.
- **Independent cases from repetition.** Streams reuse recordings, and the gamma edit is an augmentation. Recordings
  and sessions, not candidates, are the units.
- **That a corrector would work.** Nothing was trained on this library.

## 5. Cost

- **Training.** Each fold took about 15 minutes on the GPU (30 epochs of about 30 seconds).
- **Candidates.** Building the streams and candidates of all three folds took 16 seconds on the CPU.

## Files

- `fold_balance.json` holds the fold composition (counts only).
- `models.json` holds, per fold, the checkpoint sha256, the feature-cache key, the training summary and the number of
  predicted clips.
- `recognition.json` holds table 1 with the full confusion matrices at the sample and clip level.
- `composition.json` holds table 2 per fold and over all folds, for all pairs, A4C/A5C pairs and other pairs, with the
  class pairs and the reasons for candidates outside the task.
- `diagnostics.json` holds the own-view counts of the correct middles, added after the fact-check.
- `report_tables.md` holds the generated tables.
- None of these files contains recording or session ids.
- **Server only.** The session-to-fold map, the streams and the candidate lists stay on the server, in
  `runs/session_fold_library_v1/`. The per-sample probabilities and features stay in the feature cache, one cache per
  model.
