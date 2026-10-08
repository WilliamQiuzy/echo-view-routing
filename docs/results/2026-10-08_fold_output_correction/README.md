# Fold-output correction: ASRF trained on in-sample or session-held-out predictions (2026-10-08)

This round asked whether a temporal model recognises and segments differently when it trains on a visual encoder's
held-out predictions instead of its in-sample predictions. The same official ASRF recipe was trained on five-class
probabilities from two sources, with three seeds each.
- **T-in** read predictions from fold encoders that had trained on the recordings' sessions.
- **T-out** read predictions from fold encoders that had not.
- **P-out** is a simple control. It is a per-frame linear calibrator trained on the T-out inputs.

**What kind of evidence this is.**
- Training used the EV9V training split only.
- The development halves had been used before in this project, so the main evaluation on them is not an independent
  validation.
- The hold-out is by session, not by patient.
- Constructed streams are spliced from separate recordings and are not real scans.
- Native clips carry the public clip label, not expert frame labels.
- The result covers these B0 inputs and this ASRF recipe only. It says nothing directly about L1 or real continuous
  scans.
- The test split and EchoNet-Pediatric were not read.

**Order of work.**
- The plan is `docs/superpowers/plans/2026-10-08-fold-output-correction.md`.
- 8fdb68e committed the configuration, code and tests before the T-in inputs, the batch plans or any score existed.
  The T-out inputs and the B0 v3 caches came from earlier rounds.
- c6f8825 committed the preparation record before training.
- 0d43ad3 fixed two smoke-check problems before any training run (section 7).
- 8fdb68e already held the evaluation code. 4c1a289 added a CPU dry run on training data and a check that no
  recording or session id enters the summary. 090c872 added the review fixes (section 7). Both were committed while
  the models trained and before any evaluation.
- All nine runs finished, their final weights were fixed, and the development halves were then evaluated once.

## Answers

1. **What was done.**
   - **Inputs.** T-in read fold k through model (k + 1) % 3, which trained on fold k. T-out and P-out read fold k
     through model k, which did not. Every training recording's fold was checked against these rules. The fold
     models' training lists were matched to their run records by clip and session counts.
   - **Training.** Each arm used the session-fold library's 1,500 constructed streams and 1,500 native windows per
     epoch. The three arms saw the same sequences in every update. T-in and T-out differed only in the
     probabilities, and P-out read the T-out probabilities with its own model.
   - **Budget.** Each arm ran three seeds for 50 epochs of 375 updates, and the final weights were kept. Nothing was
     evaluated during training.
   - **Evaluation.** The main evaluation fed the frozen B0 v3 to every model. The three fold encoders fed every model
     in a separate auxiliary evaluation, which cannot change the main result.
   - **Why the auxiliary evaluation.** The plan added it before any training. B0 v3 is its run's epoch-0 checkpoint,
     the weights after one training epoch. The three fold encoders are the final weights of 30-epoch runs, and they
     produced the training inputs. Transfer to the main system can then be looked at separately from performance on
     inputs made by the same recipe as the training inputs. The auxiliary evaluation selects no encoder.
2. **The decision.** On the main encoder, T-out against T-in shows no consistent difference in segment F1@50 in either
   half.
   - On the selection half, T-out − T-in is −0.003, −0.015 and +0.002 (seeds 0, 1, 2). Every interval includes zero.
   - On the calibration half, it is −0.017, −0.063 and −0.038. Only seed 1's interval excludes zero, below zero.
   - Under the fixed table, the gain is not supported in either half. The round ends, and nothing further is searched.
3. **Against the two simple controls (main encoder).**
   - **P-out + fixed B3.** T-out is consistently better on the calibration half only (+0.060 to +0.105). On the
     selection half the difference is +0.016 to +0.035, and seed 1's interval includes zero.
   - **Uncalibrated B0 v3 + fixed B3.** T-out is consistently worse on the selection half only (−0.041 to −0.058). On
     the calibration half it is −0.022 to −0.065, and seed 0's interval includes zero.
   - The uncalibrated baseline has the highest segment F1 point estimates of the four systems here, 0.932 on the
     selection half and 0.928 on the calibration half.
4. **What T-out changes, against T-in (main encoder).**
   - **A5C read as A4C.** T-out fixes 3–4 such clips in each half and adds none.
   - **A4C read as A5C.** T-out adds 8–12 such clips on the selection half (in 7–9 sessions) and 2–4 on the
     calibration half, and fixes none.
   - **Segments and view changes (selection half).** 31–35 reference segments are matched by T-out only, from 12–14
     recordings in 11–12 sessions. 54–60 are matched by T-in only, from 46–51 recordings in 22–23 sessions. 4–6 true
     view changes are found by T-out only and 22–24 by T-in only. Streams reuse recordings, so these counts are not
     independent cases.
   - **Boundary recall.** T-out's boundary recall is 0.915–0.935, below T-in's 0.966–0.971.
   - **P-out.** It reads A4C as A5C in 36–44 clips per half. Its A4C recall on native clips is 0.49–0.52 on the
     selection half and 0.39–0.41 on the calibration half.
5. **Constraints and the goal.** T-out fails both system constraints in every seed and half.
   - Its boundary recall is 0.915–0.935, below 0.95.
   - Its native macro-F1 is 0.901–0.922 on the selection half and 0.950–0.962 on the calibration half, below 0.973
     and 0.980.
   - Its mean segment F1@50 is 0.884 and 0.885, which is 0.061 and 0.097 below the goal of 0.944 and 0.982.
6. **The auxiliary encoders (cannot change the main result).**
   - **Against T-in.** With fold encoders 1 and 2, T-out is consistently better in both halves (+0.046 to +0.073).
     With fold encoder 0 it is consistently better on the calibration half only (+0.049 to +0.057).
   - **Against P-out.** With fold encoders 1 and 2, no consistent difference is detected in either half. P-out's
     point estimate is ahead of T-out in all twelve of their seed and half cells, by 0.002 to 0.030. Two of these
     twelve intervals exclude zero, both with fold encoder 1 on the selection half. With fold encoder 0, T-out is
     consistently better on the calibration half only.
   - **Reading.** These are the same 30-epoch fold encoders that produced the training inputs, while B0 v3 is an
     epoch-0 checkpoint. A consistent gain over T-in appears with all three of them, in both halves with two and on
     the calibration half with fold encoder 0. It does not appear with B0 v3. A consistent gain over P-out appears
     only with fold encoder 0 on the calibration half. These results do not identify which property of the inputs
     matters.
   - **Units.** Three encoders and three seeds are not nine independent cases.
7. **The training inputs.**
   - **In-sample.** The fold encoders' in-sample predictions are nearly perfect, with 0.017–0.027% per-sample error and
     every clip correct.
   - **Held-out.** Their held-out predictions err on 3.6–4.2% of samples.
   - **Training loss.** In the last epoch, T-in's total training loss is about 0.03 and T-out's about 0.17. P-out's
     cross entropy ends near 0.19. No epoch was chosen from these curves.

## Terms

- The **paired bootstrap** resamples native clips by session and streams by whole stream, with the same draws for both
  systems (2,000 draws, seed 20261002). It treats streams that share recordings as independent, so its intervals are
  approximate.
- A difference is **consistent** in a half when all three seeds' 95% intervals exclude zero on the same side.
  Otherwise no consistent difference is detected, which does not mean the systems are equal.
- The **fixed B3** finds Jensen-Shannon peaks with half-window 9, threshold 0.31007912515352554 and NMS radius 5,
  keeps a peak only where the label changes, and gives each interval its mean-probability argmax. It is the registry's
  audited setting. P-out and the uncalibrated baseline use it.
- The **raw argmax** is each system's per-sample argmax without decoding. It is reported beside every full system.
- **Fixed** and **new** errors compare two systems on the same clips. A fixed error is wrong in the second system and
  right in the first. A new error is wrong in the first system and right in the second.
- A **reference segment** is a run of one true view in a multi-view stream. It is matched when a predicted segment of
  the same view has an IoU of at least 0.5 with it. Matching is one-to-one, highest IoU first.

## 1. Main evaluation (frozen B0 v3)

Each cell gives the selection half, then the calibration half.

| System | Segment F1@50 | Boundary recall | Boundary precision | Native macro-F1 | Cuts per clip | A5C read as A4C (clips) | A4C read as A5C (clips) | Raw argmax segment F1 |
|---|---|---|---|---|---|---|---|---|
| B0 v3 + fixed B3 | 0.932 / 0.928 | 0.963 / 0.967 | 0.979 / 0.941 | 0.965 / 0.957 | 0.014 / 0.051 | 3 / 5 | 0 / 0 | 0.528 / 0.577 |
| T-in seed 0 | 0.889 / 0.923 | 0.966 / 0.970 | 0.885 / 0.921 | 0.965 / 0.966 | 0.056 / 0.084 | 3 / 4 | 0 / 0 | 0.875 / 0.929 |
| T-in seed 1 | 0.888 / 0.927 | 0.969 / 0.967 | 0.909 / 0.928 | 0.941 / 0.966 | 0.053 / 0.059 | 4 / 4 | 0 / 0 | 0.878 / 0.928 |
| T-in seed 2 | 0.889 / 0.924 | 0.971 / 0.970 | 0.909 / 0.917 | 0.949 / 0.966 | 0.063 / 0.110 | 4 / 4 | 0 / 0 | 0.878 / 0.925 |
| T-out seed 0 | 0.886 / 0.907 | 0.919 / 0.935 | 0.964 / 0.954 | 0.901 / 0.962 | 0.021 / 0.021 | 0 / 0 | 12 / 2 | 0.860 / 0.878 |
| T-out seed 1 | 0.874 / 0.864 | 0.924 / 0.915 | 0.962 / 0.955 | 0.906 / 0.950 | 0.011 / 0.030 | 0 / 0 | 11 / 4 | 0.844 / 0.830 |
| T-out seed 2 | 0.891 / 0.885 | 0.919 / 0.925 | 0.972 / 0.966 | 0.922 / 0.962 | 0.004 / 0.013 | 0 / 0 | 8 / 2 | 0.852 / 0.851 |
| P-out seed 0 | 0.851 / 0.802 | 0.903 / 0.894 | 0.997 / 0.997 | 0.796 / 0.809 | 0.000 / 0.004 | 0 / 0 | 44 / 37 | 0.449 / 0.434 |
| P-out seed 1 | 0.857 / 0.804 | 0.903 / 0.894 | 0.997 / 0.997 | 0.800 / 0.814 | 0.000 / 0.004 | 0 / 0 | 43 / 36 | 0.445 / 0.431 |
| P-out seed 2 | 0.860 / 0.806 | 0.903 / 0.894 | 0.997 / 0.997 | 0.803 / 0.814 | 0.000 / 0.004 | 0 / 0 | 42 / 36 | 0.443 / 0.428 |

The B0 v3 + fixed B3 row is this round's own baseline. It differs from the archived E0 + B3 row (section 5), which uses
its own tuned B3.

**Two-view pair streams (supplementary).** The plan keeps these out of every decision. Each cell gives the
selection half, then the calibration half.

| System | Pair segment F1@50 | Pair boundary recall |
|---|---|---|
| B0 v3 + fixed B3 | 0.981 / 0.948 | 1.000 / 0.897 |
| T-in seed 0 | 0.974 / 0.968 | 1.000 / 0.966 |
| T-in seed 1 | 0.969 / 0.957 | 1.000 / 0.914 |
| T-in seed 2 | 0.969 / 0.965 | 1.000 / 0.931 |
| T-out seed 0 | 0.936 / 0.915 | 0.986 / 0.879 |
| T-out seed 1 | 0.912 / 0.871 | 0.986 / 0.914 |
| T-out seed 2 | 0.938 / 0.936 | 0.986 / 0.914 |
| P-out seed 0 | 0.839 / 0.776 | 0.986 / 0.862 |
| P-out seed 1 | 0.848 / 0.796 | 0.986 / 0.862 |
| P-out seed 2 | 0.848 / 0.802 | 0.986 / 0.862 |

## 2. Paired comparisons (main encoder, segment F1@50)

| Comparison | Half | Seed 0 | Seed 1 | Seed 2 | Verdict |
|---|---|---|---|---|---|
| T-out − T-in | selection | −0.003 [−0.047, +0.044] | −0.015 [−0.061, +0.033] | +0.002 [−0.042, +0.046] | no consistent difference detected |
| T-out − T-in | calibration | −0.017 [−0.049, +0.017] | −0.063 [−0.102, −0.022] | −0.038 [−0.076, +0.002] | no consistent difference detected |
| T-out − (P-out + B3) | selection | +0.035 [+0.010, +0.060] | +0.016 [−0.011, +0.043] | +0.032 [+0.008, +0.056] | no consistent difference detected |
| T-out − (P-out + B3) | calibration | +0.105 [+0.077, +0.135] | +0.060 [+0.034, +0.088] | +0.079 [+0.051, +0.109] | consistent positive |
| T-out − (B0 v3 + B3) | selection | −0.045 [−0.082, −0.009] | −0.058 [−0.097, −0.020] | −0.041 [−0.076, −0.005] | consistent negative |
| T-out − (B0 v3 + B3) | calibration | −0.022 [−0.055, +0.013] | −0.065 [−0.103, −0.026] | −0.043 [−0.081, −0.004] | no consistent difference detected |

## 3. Fixed and new errors, T-out against T-in (main encoder)

Each cell gives the count, then its distinct recordings and sessions where they apply.

| Seed | Half | A5C read as A4C, fixed / new | A4C read as A5C, fixed / new | Segments matched by T-out only | Segments matched by T-in only | View changes found by T-out only | View changes found by T-in only |
|---|---|---|---|---|---|---|---|
| 0 | selection | 3 / 0 | 0 / 12 (9 sessions) | 31 (14, 12) | 56 (48, 23) | 6 (9, 8) | 24 (29, 20) |
| 0 | calibration | 4 / 0 | 0 / 2 (2 sessions) | 17 (12, 11) | 39 (34, 20) | 3 (6, 5) | 17 (25, 18) |
| 1 | selection | 4 / 0 | 0 / 11 (9 sessions) | 35 (13, 11) | 60 (51, 23) | 5 (7, 6) | 22 (27, 19) |
| 1 | calibration | 4 / 0 | 0 / 4 (4 sessions) | 16 (10, 10) | 64 (52, 24) | 2 (4, 3) | 23 (31, 19) |
| 2 | selection | 4 / 0 | 0 / 8 (7 sessions) | 34 (12, 11) | 54 (46, 22) | 4 (5, 4) | 24 (29, 20) |
| 2 | calibration | 4 / 0 | 0 / 2 (2 sessions) | 14 (9, 9) | 52 (45, 23) | 1 (2, 2) | 19 (27, 18) |

**The frozen visual head against each system (native clips, main encoder).**
- **T-out.** It gets wrong 10–14 clips that B0 v3's own per-sample argmax gets right on the selection half (7–9
  sessions), and 4–6 on the calibration half. The reverse happens for 3 clips on the selection half and 2 on the
  calibration half in every seed.
- **T-in.** It gets wrong 0–2 such clips per half, and none in reverse.
- **P-out.** It gets wrong 43–45 such clips on the selection half (19 sessions) and 36–37 on the calibration half.

## 4. Auxiliary evaluation (each fold encoder on both halves)

These tables describe how the results depend on the input encoder. They select nothing and cannot reverse the main
result.

| Encoder | System | Segment F1@50, selection | Segment F1@50, calibration |
|---|---|---|---|
| fold 0 | encoder + fixed B3 | 0.860 | 0.895 |
| fold 0 | T-in (seeds 0, 1, 2) | 0.858, 0.869, 0.870 | 0.899, 0.905, 0.901 |
| fold 0 | T-out (seeds 0, 1, 2) | 0.886, 0.897, 0.895 | 0.949, 0.953, 0.957 |
| fold 0 | P-out (seeds 0, 1, 2) | 0.896, 0.896, 0.896 | 0.923, 0.923, 0.927 |
| fold 1 | encoder + fixed B3 | 0.859 | 0.863 |
| fold 1 | T-in (seeds 0, 1, 2) | 0.847, 0.844, 0.847 | 0.864, 0.866, 0.867 |
| fold 1 | T-out (seeds 0, 1, 2) | 0.893, 0.891, 0.899 | 0.918, 0.919, 0.917 |
| fold 1 | P-out (seeds 0, 1, 2) | 0.921, 0.921, 0.921 | 0.921, 0.921, 0.921 |
| fold 2 | encoder + fixed B3 | 0.836 | 0.843 |
| fold 2 | T-in (seeds 0, 1, 2) | 0.831, 0.849, 0.848 | 0.833, 0.836, 0.831 |
| fold 2 | T-out (seeds 0, 1, 2) | 0.890, 0.903, 0.897 | 0.906, 0.904, 0.898 |
| fold 2 | P-out (seeds 0, 1, 2) | 0.903, 0.905, 0.905 | 0.914, 0.910, 0.910 |

| Encoder | T-out − T-in, selection | T-out − T-in, calibration | T-out − P-out, selection | T-out − P-out, calibration |
|---|---|---|---|---|
| fold 0 | no consistent difference detected | consistent positive | no consistent difference detected | consistent positive |
| fold 1 | consistent positive | consistent positive | no consistent difference detected | no consistent difference detected |
| fold 2 | consistent positive | consistent positive | no consistent difference detected | no consistent difference detected |

T-out is also consistently better than the encoder's own fixed-B3 baseline with fold encoders 1 and 2 in both halves,
and with fold encoder 0 on the calibration half.
`summary.json` holds every auxiliary row, interval and change table.

## 5. Reference rows (archived, reproduced with their own settings)

| System | Segment F1@50 (selection / calibration) | Boundary recall | Native macro-F1 |
|---|---|---|---|
| E0 + B3 (historical strongest) | 0.934 / 0.934 | 0.961 / 0.955 | 0.965 / 0.966 |
| E1 + B3 | 0.894 / 0.959 | 0.955 / 0.977 | 0.978 / 0.986 |
| B7 + B3 | 0.880 / 0.972 | 0.955 / 0.975 | 0.957 / 0.993 |
| E1 + ASRF, seeds 0, 1, 2 | 0.910, 0.909, 0.907 / 0.976, 0.964, 0.964 | 0.942–0.953 / 0.972–0.977 | 0.948–0.966 / 0.975–0.993 |

Every row reproduced its archived counts.

## 6. Decision under the fixed rule

The configuration fixed this table in 8fdb68e, before any score.

| Observation | Conclusion that may be written | Action |
|---|---|---|
| The primary comparison is not consistent positive in both halves. | Not supported in both halves. Each half is reported as positive, negative or uncertain. | The round ends, with no automatic search. |
| Consistent positive in both halves, but not better than both simple controls. | Held-out inputs help relative to T-in, but are not shown better than the simple controls. | The round ends, with no new-model claim. |
| Better than T-in and both simple controls in both halves, but a system constraint fails. | A comparative gain that is not usable under the current standard. | The round ends, and the failed constraint is named. |
| Comparisons and constraints met, but not the final goal. | A local success with B0. The final goal is not reached. | The lead decides whether to start a transfer experiment to a stronger encoder. |
| Comparisons, constraints and the final goal all met. | The goal is reached under this development protocol. Independent validation is still needed. | The models are frozen and the results committed. Opening the test split is decided separately. |

The auxiliary results never reverse this table. This round went through it as follows.

| Step | Result |
|---|---|
| Primary comparison consistent positive in both halves? | No. No consistent difference was detected in either half. |
| Better than T-in and both simple controls in both halves? | No. |
| Constraints met (boundary recall ≥ 0.95, native macro-F1 ≥ 0.973 / 0.980, every seed and half)? | No. |
| Goal met (mean segment F1@50 ≥ 0.944409687184662 / 0.9821421709894332)? | No. The means are 0.884 and 0.885. |
| Case | The primary comparison is not consistent positive in both halves. |
| Action | The round ends, and no further search is started. |

The calibration sensitivity check (0.9738935784747763 + 0.01) is also missed, by 0.099.

## 7. Deviations from the plan and process checks

- **P-out padding (0d43ad3, before training).** The committed P-out loss ignored padding only when its label was
  −100. The smoke check's padding label reached the GPU kernel and failed. The loss now applies the valid mask
  explicitly, as the plan specifies.
- **Smoke padding tolerance (0d43ad3, before training).** The committed smoke check required an exactly unchanged T-in
  loss when the padding changed. In the first GPU smoke run the change was 1.19e-7, the same size as the difference
  between two identical loss computations. The check now uses the smoke test's 1e-6 tolerance and records the
  same-input difference. The rerun after the fix overwrote the first run's output, so the server keeps only the
  rerun (padding change 0.0, same-input difference 1.19e-7).
- **Evaluation code (4c1a289 and 090c872, before evaluation).** 4c1a289 added a CPU dry run on training data and the
  id check. A code review then led to four changes in 090c872, none of which alters a metric or a rule. Results are
  written before the reference and freezing steps. The auxiliary extraction checks the class order. The cache-content
  digests inside the run identities are computed once per cache set instead of once per run. Outputs are written
  atomically.
- **Class weights.** As the configuration fixes, the class weights and the boundary pos_weight were computed once per
  seed from that seed's epoch-0 labels. The three arms share them.
- **B0 v3 identity.** The preparation script checks its checkpoint digest (beb36d80df31…), epoch (0) and class
  order. A separate read-only check before training found every training and development clip in its caches. That
  check is not part of the committed scripts.
- **The old resume bug (fixed in 1122a68).** After a resume, the old B0 trainer kept the optimizer of a discarded
  model, so the loaded weights were not trained further. Their BatchNorm statistics could still change.
  - The deployed B0 v3 weights are their run's epoch-0 checkpoint. They were written before any point from which that
    run could have resumed, so the bug cannot have changed them.
  - Whether the bug touched that run's later epochs, and so which epoch was selected, is not confirmed.
  - The three logged runs of the old B0 trainer (a smoke run, the first B0 run and the v3 run) start at epoch 0 and
    never repeat an epoch. A resumed run continues the epoch count, so these logs cannot show whether a resume
    happened. Command records are incomplete, so whether any of these runs was ever resumed is not confirmed.
- **Tests.** The plan's local command (`python -m pytest tests/test_fold_output_correction.py
  tests/test_session_fold_library.py -q`, project `.venv`) gives 25 passed and 1 skipped. The skipped test needs the
  pinned ASRF checkout, which exists only on the server. On the server, these two files and the resume test passed
  before the input extraction (27 tests, none skipped).
- **Commands record.** The plan recommends recording the commands in the YAML and in this README. They are listed
  under Commands below. The YAML was not changed after training, because the run identities include its round
  section.
- **Nothing chosen after results.** No epoch, seed, hyperparameter, threshold or decoding setting was added or chosen
  after training started.

## 8. What this does not show

- **That per-frame calibration cannot help.** P-out is one specific calibrator. Its large A4C → A5C shift on B0 v3 does
  not show that every calibration method fails.
- **What causes the error shift on B0 v3 inputs.** With B0 v3 inputs, the held-out-trained models remove A5C → A4C
  errors and add A4C → A5C errors, and no consistent gain over T-in was detected. This round does not test what
  causes that.
- **A causal effect of memorisation.** T-in and T-out read inputs from different trained models, so the comparison is
  between two input-generating procedures. It is not a switch inside one set of weights.
- **Independent or patient-level evidence.** The development halves have been used before, and the hold-out is by
  session.
- **Anything about L1 or real scans.** All inputs here come from B0 encoders, and the streams are spliced.

## 9. Cost

- **Input extraction.** The T-in caches took about 6.5 minutes on the GPU and the auxiliary caches about 2.8 minutes,
  from log start to last write.
- **Training.** Each ASRF run took about 22 minutes on the GPU and each P-out run about 0.6 minutes, nine runs in all.
- **Evaluation.** The main and auxiliary evaluation, the references and the freezing took 10.4 minutes.

## Commands

Run on the server in the project `.venv`, one GPU job at a time, with `ECHO_GPU_MEM_FRACTION=0.6`.

```bash
C=configs/experiments/fold_output_correction_v1.yaml; R=docs/results/2026-10-08_fold_output_correction
python scripts/prepare_fold_output_correction.py --config $C --out $R --extract      # T-in caches (GPU)
python scripts/prepare_fold_output_correction.py --config $C --out $R --check        # preparation.json
python scripts/train_fold_output_correction.py --config $C --arm T-in --seed 0 --smoke   # also T-out and P-out
python scripts/train_fold_output_correction.py --config $C --queue                   # the nine runs, in order
python scripts/report_fold_output_correction.py --config $C --out $R --dry-run       # CPU, training data only
python scripts/report_fold_output_correction.py --config $C --out $R --extract-aux   # auxiliary caches (GPU)
python scripts/report_fold_output_correction.py --config $C --out $R --evaluate      # the single evaluation, freezing
python scripts/plot_fold_output_correction_curves.py runs/fold_output_correction_v1/train $R/training_curves.png
```

## Files

- `preparation.json` holds the input identities, the training-list counts and digests, cache alignment, the batch plans
  with their class weights and the input description (counts and digests only).
- `summary.json` holds every system row and raw-argmax row per encoder, seed and half, every comparison with its
  bootstrap and change tables, the verdicts, the decision, the reference rows and the frozen models.
- `report_tables.md` holds the generated tables, and `training_curves.png` the training losses.
- None of these files contains recording or session ids.
- **Server only.** The plans, initial states, run manifests and caches stay on the server. The nine final models are
  frozen there as `models_frozen/fold_output_correction_v1/{T-in,T-out,P-out}_s{0,1,2}`, and per-sample outputs can be
  regenerated from them.
