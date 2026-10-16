# FACT on frozen video features under a revised launch gate (A06-R1, 2026-10-15 to 2026-10-16)

A06 v1 stopped at its overfit smoke check before any formal training
([stop record](../2026-10-15_fact_video_feature/README.md)). Its task sheet required every matched token to predict its
segment's view, and 4 of the 9 single-segment check sequences kept no active token. After seeing that result, the
project lead amended this one launch requirement. Under A06-R1, a single-segment smoke sequence may name its segment
through the all-null fallback of the pinned official decoder, provided its official output is exact. That fallback is
part of the official FACT code: when every token predicts the null class, `Block._eval` returns the frame branch's
argmax. Every multi-segment token requirement stayed as it was. This is a post-hoc procedure amendment made after a
result was seen, and it is reported as one. It does not explain why those tokens stayed null, and it does not turn the
v1 stop into a pass; the v1 records are unchanged. The model, the inputs, the initial states, the three seeds, the
18,750 updates, the loss, the learning rate, the official decoding, every comparison and every target are those of v1.

**The result.**
- Three FACT heads were trained and frozen. On the two reused development halves, the primary comparison, FACT minus
  A01 ASRF in multi-stream segment F1@50, showed no consistent difference in either half.
- In the selection half the per-seed differences were +0.024, +0.009 and −0.002. In the calibration half they were
  −0.018, −0.016 and −0.009.
- In the calibration half, FACT was consistently below A01 Linear + B3 and frozen Ordered + B3 in multi-stream segment
  F1@50 and missed more reference segments than both.
- The goal was not met. FACT's calibration mean was 0.9605 against the minimum of 0.9821, and four seed-and-half cells
  failed a recall or native floor.
- The fixed readings that apply are one_half_or_local_effect (through point estimates only), trade_off,
  uncertain_or_no_qualifying_gain and end.

The task sheet is `docs/superpowers/plans/2026-10-15-fact-video-feature-gate-amendment.md` (0053bda). The configuration
is `configs/experiments/fact_video_feature_v2.yaml` (7bd5249, sha256 d3841a09). The original A06 task sheet and its
configuration stay as they were (`docs/superpowers/plans/2026-10-15-fact-video-feature-training.md`,
`configs/experiments/fact_video_feature_v1.yaml`).

## What was trained

- **Three FACT heads.** Each is the official GTEA FACT (pinned checkout 7bd81bda, 12 action tokens, 10,215,424
  parameters) on the frozen V01 Ordered 512-d video features from A01's caches. The token count is int(1.5 x 8), where 8
  is the longest training transcript.
- **The video encoder was not updated.** No feature was re-extracted, and no encoder weight entered an optimiser.
- **The recipe was v1's.** Each seed started from a bit-identical copy of F02's untrained FACT state (52ba4673, 048357d7
  and 7bb0b685). It saw A01's exact batches (every update's digest equals A01's plan) and ran Adam at 1e-4, multiplied by
  0.1 after 11,719 updates, for 18,750 updates. The official augmentation, matching and loss weights were unchanged. The
  final weights were kept, with no epoch choice and no early stop.
- **Decoding.** FACT's output is the official output of its last block on an all-zero label sequence. It has no B3, no
  threshold and no search. Its frame-branch argmax is reported only as a diagnostic readout of the same weights.
- **Controls.** Every other system is a historical frozen control with its stored setting, reproduced exactly against
  A01's stored rows before use. They are A01's ASRF, Linear + B3 and MS-TCN on the same features and frozen Ordered + B3
  (V01's saved probabilities). E0, E1 and B7 + B3 and F02's FACT are stored rows from A01's summary. They use image
  features as input, so they replace the whole input source and are not controlled comparisons.
- `fact_tokens.png` sketches how the tokens and the frame branch work together.

## What kind of evidence this is

- **Reused data.** The two development halves (selection and calibration) have been used and inspected in many earlier
  rounds. Every number here is a result on that reused development data, not an independent validation.
- **Grouping.** The hold-out is by session, not by patient. Constructed streams are spliced from separate recordings and
  are not continuous scans. A few recordings recur in them, so their units are not independent cases. The three seeds
  share the same cases.
- **Labels.** Native clips carry the public recording label, not expert frame-by-frame labels. No label was changed.
- **Untouched data.** The test split and EchoNet-Pediatric were not read.
- **Scope.** The results belong to this revised execution. They are not a pass of the pre-registered v1 plan. Any
  difference belongs to the whole FACT recipe (model, optimiser and augmentation), never to cross-attention or to one loss
  term alone.

## Order of work

All times are UTC.

- **2026-10-15, before any code.** 0053bda (14:44) holds the task sheet. 7bd5249 (14:49) committed the configuration.
- **Tests and code.** bfd1dc0 (15:07) committed the tests, which failed at import. 4ae1d3c (15:16) committed the code.
- **Review.** An independent implementation review passed it with two should-fix items. 4ffeb92 (15:37) fixed them, and
  the reviewer verified the fixes.
- **First revalidation.** The CPU preparation ran from 15:42:34 to 15:44:20 and the GPU check from 15:44:55 to 15:45:55.
  Both passed, and e8c4233 (15:46) committed the launch record.
- **A refused launch.** At 15:46:48 the queue refused that record before any run started (see the deviations).
  b355046 (15:51) and 1636ada (15:59) fixed the revalidation script, and the reviewer verified both. The second
  verification, from 15:59:20 to 16:00:48, overlapped the second preparation and ended before the second check.
- **Second revalidation.** The preparation ran again from 15:59:54 to 16:01:36. Its preparation.json is byte-identical
  to the first, and revalidation_prepare.json differs only in its time stamp and the script's sha256. The check ran
  again from 16:02:03 to 16:03:03 and passed. 9fced8a (16:03) committed the launch record.
- **Training.** The queue started at 16:03:49. FACT_s0 ran from 16:03:53 to 18:51:22, FACT_s1 from 18:51:26 to 21:38:16
  and FACT_s2 from 21:38:21 to 00:25:05 on 2026-10-16. The queue then froze the three heads.
- **2026-10-16, evaluation.** The training-stream fit ran at 00:26, the calibration stage at 00:28 and its dry run at
  00:29. The round's tests passed again at 00:32. cf3963e (00:34) committed the evaluation manifest. The selection half
  was opened once, at 00:51, and the stage finished at 00:54.

## The revised gate and the checks behind the launch

**The original failure is kept.** The v1 check (`checks.json`, sha256 9061032e, run from 06:08:23 to 06:28:43 on
2026-10-15) still records passed = false. The new launch record states original_gate_passed = false. Recomputing v1's
requirements from its stored outcome gives exactly one failed requirement, matched_classes_right.

**The new gate.** The task sheet's predicate was applied, unchanged, to the 16 rows of v1's final update-2,000 smoke
outcome. No other check point was read.
- All 16 rows have an exact transcript, segment F1@50 = 1 and one-to-one matching.
- Rows 0, 4 and 6 (native crops) and row 10 (a constructed stream) have one reference segment, no active token and an
  exact official output. They pass through the official fallback.
- The other 5 single-segment rows have 1, 1, 12, 1 and 1 active tokens, and each matched token is right.
- The 7 multi-segment streams (3 to 6 segments) each have one active token per segment, and every matched token is right.
- In row 3 (one segment, 12 active tokens) the 11 unmatched tokens are all active. In every other row the unmatched
  tokens are null.
- A unit test drives the pinned `Block._eval` with synthetic logits. With all tokens null it returns the frame-branch
  argmax, and with active tokens it applies the official fusion. This checks the decoder branch only. No new fallback
  logits of the trained model were measured.

**Inherited from the immutable v1 evidence.** Isolation and the final smoke rows were not rerun. Each was verified
field by field in the v1 check record above. The gradient record and the strict loads were verified there and also
repeated under A06-R1, with the same three parameters and 326 tensors and 10,215,424 parameters per initial state.
- Isolation under five label sequences at the initial and the overfit state: 216 sequences and 864 comparisons each, with
  no changed prediction, logit or adapter decode.
- The gradient record at the initial state. The same three parameters receive no gradient from the official total:
  `block_list.0.action_branch.layers.0.norm1.weight`, `...self_attn.in_proj_weight` and `...self_attn.out_proj.weight`
  of the input block's first action-branch layer. F02 recorded the same three. No other parameter lacks a gradient.
- The final smoke rows, the requirements recomputed from them and the strict loads of the three initial states.

**Run again under A06-R1.**
- **Sources.** A new CPU preparation re-verified every source, pin, cache hash, the official checkout, the token rule,
  every batch of all 50 epochs per seed and the initial states. It equals v1's preparation record in every field except
  the namespace paths, the task sheet, the configuration and the code digest.
- **Code.** The training code that produced the v1 evidence was exported from git at 3d2ea49. Its 11 files digest to
  ae7531732dd0, the digest recorded by v1's preparation and check. Every training file is byte-identical to that code
  except the round's own script. In that script only the namespace, identity, launch-record, cumulative-ledger and
  monitor-logging functions that the configuration lists changed. The training loop, the batches, the optimiser, the
  schedule, the resume logic and the decoding are AST-identical.
- **GPU checks (seed 0).** The 6-update repeat and the 3 + 3 resume through a saved file differ by 0.0. The padding check
  on the first formal batch (4 streams and 4 native crops) covers 831 padded positions, as in v1, with no change in loss
  or eval-mode prediction. The optimiser matches the configuration and is bound to the model. The gradient record shows
  the same three parameters. Timing measured 0.639 s and then 0.638 s per update.
- **Budget.** The projection used the larger of the old and the new values: v1's 3.371 hours per run and the new 0.113
  hours of post-training work. With the 1.25-hour reserve, the cumulative projection before the queue was 11.848 of 12
  GPU hours. The queue repeated it before each run: 11.268 before FACT_s1 and 10.678 before FACT_s2. The status file
  keeps only the last projection, and the earlier one recomputes from the ledger.

## Training

- **Runs.** Each run completed 18,750 updates without interruption, in 2.791, 2.780 and 2.779 GPU hours. The final
  weights (911a9a64, ff2c6637 and 369f7fc8) were frozen with `scripts/freeze_model.py` in
  `models_frozen/fact_video_feature_v2/FACT_s{0,1,2}`.
- **The 16 check sequences.** This monitor uses training data only and selects nothing. After the last epoch, segment
  F1@50 was 0.910, 0.908 and 0.920, with 14 of 16 exact transcripts in each seed.
- **Tokens on the check sequences.** These counts were added by A06-R1 as logging only, from the monitor's own forwards.
  - After the first epoch of FACT_s0, 8 of the 9 single-segment sequences had no active token.
  - After the last epoch, no single-segment sequence was all-null in any seed, and 8 of 9 single-segment matched tokens
    were right in each seed.
  - In the multi-segment streams, 33, 33 and 32 of the 33 matched tokens were right.
- **Losses.** The last epoch's mean training loss (augmented batches) was 1.730, 1.721 and 1.723. On the check
  sequences, the token-to-frame cross-attention term (f2a) was the largest term, near 1.30.
- **All 1,500 training streams per seed.** The official decoding gave the results below.
  - Segment F1@50 was 0.973, 0.970 and 0.971, frame accuracy 0.956, 0.955 and 0.955, and 1,340, 1,331 and 1,333 exact
    transcripts.
  - The missed reference segments numbered 169, 189 and 188. Most were short (91, 97 and 102 of at most 16 samples) or
    A4C and A5C (152, 174 and 175).
  - The number of active tokens equalled the number of segments in 1,190, 1,203 and 1,251 streams.
  - This locates under-fitting only. It added no updates and no budget.
- `training_curves.png` shows the curves.

## Development results

The tables give seeds 0, 1 and 2 and then their mean. Multi-stream metrics cover the 120 constructed multi-recording
streams per half, and the pair metric covers the constructed two-recording streams. Native macro-F1 and native cuts per
clip cover the single-view native clips. The same-class cut rate is the share of same-view joins at which a system
wrongly starts a new segment.

**Selection half** (541 sequences)

| System | Multi segment F1@50 | Multi boundary F1 | Multi boundary recall | Native macro-F1 | Native cuts per clip | Same-class cut rate | Pair segment F1@50 |
|---|---|---|---|---|---|---|---|
| FACT | 0.961 / 0.957 / 0.949 (0.9557) | 0.962 / 0.951 / 0.949 (0.9540) | 0.958 / 0.948 / 0.942 (0.9494) | 0.966 / 0.987 / 0.998 (0.9834) | 0.000 / 0.004 / 0.000 (0.0012) | 0.004 / 0.000 / 0.004 (0.0029) | 0.966 / 0.969 / 0.957 (0.9639) |
| A01 ASRF | 0.937 / 0.948 / 0.951 (0.9449) | 0.961 / 0.963 / 0.961 (0.9619) | 0.948 / 0.948 / 0.945 (0.9468) | 0.987 / 0.977 / 0.987 (0.9836) | 0.014 / 0.004 / 0.000 (0.0059) | 0.035 / 0.030 / 0.030 (0.0319) | 0.971 / 0.961 / 0.971 (0.9676) |
| A01 Linear + B3 | 0.947 / 0.957 / 0.939 (0.9476) | 0.954 / 0.962 / 0.940 (0.9521) | 0.955 / 0.953 / 0.971 (0.9599) | 0.979 / 0.979 / 0.979 (0.9786) | 0.046 / 0.039 / 0.085 (0.0563) | 0.026 / 0.009 / 0.030 (0.0217) | 0.969 / 0.952 / 0.957 (0.9593) |
| frozen Ordered + B3 | 0.938 / 0.960 / 0.964 (0.9544) | 0.953 / 0.964 / 0.966 (0.9610) | 0.958 / 0.969 / 0.974 (0.9668) | 0.979 / 0.979 / 0.970 (0.9758) | 0.042 / 0.039 / 0.046 (0.0423) | 0.017 / 0.004 / 0.004 (0.0087) | 0.962 / 0.959 / 0.957 (0.9592) |
| A01 MS-TCN | 0.920 / 0.953 / 0.943 (0.9387) | 0.941 / 0.957 / 0.948 (0.9489) | 0.966 / 0.963 / 0.958 (0.9625) | 0.963 / 0.977 / 0.975 (0.9717) | 0.074 / 0.042 / 0.056 (0.0575) | 0.035 / 0.017 / 0.013 (0.0217) | 0.966 / 0.962 / 0.971 (0.9664) |
| FACT frame branch (diagnostic) | 0.944 / 0.958 / 0.950 (0.9510) | 0.943 / 0.958 / 0.951 (0.9505) | 0.948 / 0.961 / 0.955 (0.9546) | 0.998 / 0.998 / 0.998 (0.9976) | 0.004 / 0.000 / 0.000 (0.0012) | 0.004 / 0.000 / 0.009 (0.0043) | 0.971 / 0.962 / 0.959 (0.9639) |

**Calibration half** (472 sequences)

| System | Multi segment F1@50 | Multi boundary F1 | Multi boundary recall | Native macro-F1 | Native cuts per clip | Same-class cut rate | Pair segment F1@50 |
|---|---|---|---|---|---|---|---|
| FACT | 0.962 / 0.958 / 0.961 (0.9605) | 0.966 / 0.971 / 0.964 (0.9671) | 0.955 / 0.967 / 0.955 (0.9590) | 0.978 / 0.986 / 0.993 (0.9854) | 0.008 / 0.004 / 0.008 (0.0070) | 0.020 / 0.015 / 0.020 (0.0182) | 0.982 / 0.968 / 0.968 (0.9728) |
| A01 ASRF | 0.980 / 0.975 / 0.970 (0.9747) | 0.981 / 0.982 / 0.971 (0.9780) | 0.970 / 0.972 / 0.957 (0.9665) | 0.978 / 0.972 / 0.993 (0.9811) | 0.008 / 0.008 / 0.008 (0.0084) | 0.015 / 0.010 / 0.015 (0.0133) | 0.980 / 0.968 / 0.974 (0.9736) |
| A01 Linear + B3 | 0.977 / 0.978 / 0.973 (0.9760) | 0.975 / 0.976 / 0.963 (0.9713) | 0.975 / 0.980 / 0.972 (0.9757) | 0.986 / 0.986 / 0.986 (0.9861) | 0.013 / 0.017 / 0.042 (0.0239) | 0.020 / 0.015 / 0.030 (0.0216) | 0.971 / 0.971 / 0.971 (0.9709) |
| frozen Ordered + B3 | 0.982 / 0.984 / 0.980 (0.9817) | 0.979 / 0.984 / 0.979 (0.9804) | 0.982 / 0.987 / 0.980 (0.9832) | 0.993 / 0.986 / 0.986 (0.9884) | 0.025 / 0.017 / 0.013 (0.0183) | 0.025 / 0.025 / 0.025 (0.0249) | 0.982 / 0.971 / 0.971 (0.9747) |
| A01 MS-TCN | 0.967 / 0.974 / 0.963 (0.9679) | 0.962 / 0.976 / 0.961 (0.9666) | 0.962 / 0.977 / 0.970 (0.9698) | 0.978 / 0.978 / 0.978 (0.9780) | 0.038 / 0.051 / 0.055 (0.0478) | 0.025 / 0.025 / 0.035 (0.0282) | 0.974 / 0.957 / 0.982 (0.9709) |
| FACT frame branch (diagnostic) | 0.966 / 0.966 / 0.963 (0.9652) | 0.968 / 0.969 / 0.967 (0.9681) | 0.965 / 0.970 / 0.962 (0.9657) | 0.986 / 0.986 / 0.986 (0.9856) | 0.013 / 0.008 / 0.008 (0.0098) | 0.015 / 0.020 / 0.020 (0.0182) | 0.982 / 0.971 / 0.974 (0.9757) |

The stored rows from A01's summary, in multi-stream segment F1@50 for selection and calibration, are E0 + B3 0.934 and
0.934, E1 + B3 0.894 and 0.959, B7 + B3 0.880 and 0.972, and F02's FACT on image features 0.906, 0.898 and 0.911 and
0.967, 0.961 and 0.965.

**Paired differences in multi segment F1@50.** Each cell is FACT minus the control, with a 95% bootstrap interval
(2,000 draws, seed 20261002, resampling sessions for native clips and streams for constructed streams).

| Comparison | Half | Seed 0 | Seed 1 | Seed 2 | Three seeds |
|---|---|---|---|---|---|
| FACT − A01 ASRF (primary) | selection | +0.0245 [+0.0071, +0.0431] | +0.0095 [−0.0106, +0.0321] | −0.0016 [−0.0195, +0.0165] | no consistent difference detected |
| FACT − A01 ASRF (primary) | calibration | −0.0175 [−0.0334, −0.0036] | −0.0163 [−0.0355, +0.0013] | −0.0087 [−0.0224, +0.0046] | no consistent difference detected |
| FACT − A01 Linear + B3 | selection | +0.0138 [−0.0018, +0.0304] | +0.0002 [−0.0131, +0.0127] | +0.0102 [−0.0099, +0.0318] | no consistent difference detected |
| FACT − A01 Linear + B3 | calibration | −0.0148 [−0.0304, −0.0011] | −0.0195 [−0.0348, −0.0068] | −0.0121 [−0.0264, −0.0003] | consistent negative |
| FACT − frozen Ordered + B3 | selection | +0.0225 [+0.0015, +0.0438] | −0.0033 [−0.0177, +0.0118] | −0.0154 [−0.0331, +0.0006] | no consistent difference detected |
| FACT − frozen Ordered + B3 | calibration | −0.0197 [−0.0340, −0.0073] | −0.0253 [−0.0410, −0.0114] | −0.0187 [−0.0326, −0.0067] | consistent negative |
| FACT − A01 MS-TCN | selection | +0.0411 [+0.0190, +0.0665] | +0.0036 [−0.0146, +0.0222] | +0.0063 [−0.0136, +0.0273] | no consistent difference detected |
| FACT − A01 MS-TCN | calibration | −0.0052 [−0.0204, +0.0090] | −0.0156 [−0.0340, +0.0016] | −0.0015 [−0.0170, +0.0146] | no consistent difference detected |

A difference is consistent only when all three seeds' intervals lie strictly on one side of zero within one half.
Anything else is reported as no consistent difference detected, which does not mean the systems are equal.

**Every consistent result** across the six metrics and the two counts.
- In the calibration half, FACT was below A01 Linear + B3 in multi segment F1@50 and had more missed reference segments
  in multi streams (FN +12, +13 and +14).
- In the calibration half, FACT was below frozen Ordered + B3 in multi segment F1@50, in multi boundary F1 (by 0.013,
  0.013 and 0.014) and in multi boundary recall (by 0.028, 0.020 and 0.025). It also had more missed reference segments
  in multi streams (FN +16, +16 and +14).
- In the selection half, FACT made fewer cuts inside native clips than frozen Ordered + B3 (−0.042, −0.035 and −0.046
  per clip) and than A01 MS-TCN (−0.074, −0.039 and −0.056 per clip).
- Nothing was consistently different from A01 ASRF.

**The goal was not met.**
- FACT's three-seed means were 0.9557 in selection (minimum 0.9444, met) and 0.9605 in calibration (minimum 0.9821, not
  met; sensitivity value 0.9839).
- Four seed-and-half cells failed a floor. Native macro-F1 was 0.9656 in seed 0 selection (floor 0.973) and 0.9780 in
  seed 0 calibration (floor 0.980). Boundary recall was 0.9476 in seed 1 selection and 0.9424 in seed 2 selection (floor
  0.95).
- For reference, frozen Ordered + B3 averaged 0.9544 and 0.9817 and also missed the calibration minimum, by 0.0004.

## Readings

These are the fixed, non-exclusive readings, applied once.

- **complete_system_progress does not hold.** It would need consistent positive differences against A01 ASRF, A01
  Linear + B3 and frozen Ordered + B3 in both halves, all floors and the goal.
- **learned_system_gain_without_strong_reference_win does not hold.** FACT was not consistently above A01 ASRF or A01
  Linear + B3 in either half.
- **one_half_or_local_effect applies through point estimates only.** No F1 comparison was consistent positive. Pooled
  over seeds, FACT recovered more reference segments than it lost in these items.

  | Control | Half | Population | Segments | Recovered | Lost | Reference segments per seed |
  |---|---|---|---|---|---|---|
  | A01 ASRF | selection | multi | all | 46 | 18 | 502 |
  | A01 ASRF | selection | multi | at most 16 samples | 16 | 8 | 141 |
  | A01 ASRF | selection | pairs | all | 9 | 7 | 207 |
  | A01 ASRF | selection | pairs | at most 16 samples | 4 | 0 | 59 |
  | A01 ASRF | selection | native | all | 4 | 3 | 284 |
  | A01 ASRF | calibration | native | all | 3 | 1 | 237 |
  | A01 ASRF | calibration | pairs | all | 3 | 2 | 173 |
  | A01 Linear + B3 | selection | native | all | 16 | 3 | 284 |
  | A01 Linear + B3 | calibration | native | all | 5 | 4 | 237 |
  | frozen Ordered + B3 | selection | native | all | 9 | 3 | 284 |

  Almost all of these counts rest on fewer than 10 sessions per seed. The largest, against A01 ASRF in selection multi
  streams, has its recovered segments in 11, 12 and 8 sessions and its lost segments in 5, 4 and 7 sessions in seeds 0,
  1 and 2. These are point estimates, not consistent evidence.
- **trade_off applies.** In every seed and half, against both A01 Linear + B3 and frozen Ordered + B3, FACT recovered
  reference segments while it also had new unmatched predicted segments (19 to 30 per cell). The recovered counts are
  small (1 to 18 per cell, 1 to 6 in calibration), and in every cell each population's recovered segments come from
  fewer than 10 sessions. FACT also had a lower native macro-F1 point estimate or a floor failure in 10 of these 12
  cells. This describes coexistence. It needs no cross-seed
  significance and proves no cause. The cells, their counts and their FP and native macro-F1 intervals are in
  `summary.json` under `decision`.
- **uncertain_or_no_qualifying_gain applies.** The primary comparison's direction was positive in two of three seeds in
  selection and negative in all three in calibration, with no consistent difference in either half.
- **end applies.** After this report, the official GTEA FACT, video-feature and budget combination closes. No extra
  seed, Breakfast configuration, more tokens or layers, encoder fine-tuning, loss, optimiser, epoch or decoder search,
  reference boundaries, boundary subdivision, teacher, gate or fusion follows automatically.

## Where the halves differ

These counts describe the outcomes. They do not establish a cause.

- **Calibration multi streams.** Pooled over seeds, FACT recovered 0 and lost 46 reference segments against frozen
  Ordered + B3 (the lost ones in 13, 12 and 11 sessions). Against A01 Linear + B3 it recovered 3 and lost 42 (recovered
  in 3, 0 and 0 sessions, lost in 11, 10 and 11). Against A01 ASRF it recovered 15 and lost 33 (recovered in 5, 11 and 5
  sessions, lost in 12, 12 and 7). Counts on fewer than 10 sessions are small samples. 21 of the 33 lost against ASRF
  had at most 16 samples. FACT matched 119 of the 134 short reference segments in each seed.
- **Selection multi streams.** Against A01 ASRF, FACT recovered 46 and lost 18 reference segments (recovered in 11, 12
  and 8 sessions, lost in 5, 4 and 7, which is a small sample). 16 of the recovered and 8 of the lost had at most 16
  samples.
- **A4C/A5C X-Y-X middles.** Each half has only 7 per seed, so these counts are small and rest on fewer than 10
  sessions. FACT matched 3, 1 and 1 of the 7 in selection and 4, 4 and 4 in calibration. A01 ASRF matched 3, 3 and 2 and
  5, 6 and 5. A01 Linear + B3 matched 3, 3 and 5 and 7, 7 and 7. Frozen Ordered + B3 matched 5, 3 and 5 and 7, 7 and 7.
  A01 MS-TCN matched 3, 3 and 2 and 6, 6 and 5.
- **Tokens on development multi streams.** The active tokens equalled the reference segments in 81 to 90 of 120 streams
  per seed and half. There were more tokens than segments in 24 to 32 streams and fewer in 4 to 12.
- **Native A4C and A5C confusions.** FACT read A4C as A5C in 0 or 1 clips and A5C as A4C in 0 to 2 clips per seed and
  half. These counts rest on very few clips.

## Cost

The budget is cumulative for v1 and R1: 12 GPU hours and 4 CPU hours.
- **v1.** Its pinned ledger holds 0.339 GPU and 0.900 CPU hours (9 lines, sha256 7e90dc53).
- **R1 before the fact-check.** Its ledger held 8.451 GPU and 0.575 CPU hours in 27 lines.
  - The two revalidation checks took 0.017 GPU hours each.
  - The three runs took 8.351 GPU hours.
  - The training-stream fit, the calibration stage, the dry run and the selection took 0.067 GPU hours together.
  - The CPU hours include the server tests, two preparations, the 12-second identity diagnostic, the queue overhead and
    freeze, the evaluation stages and 0.432 hours of independent review and verification.
- **The fact-check.** It added 0.014 GPU hours for its re-inference. Its wall clock, 0.358 hours, and its helper agent's,
  0.305 hours, were booked separately as CPU hours.
- **Total.** R1's ledger holds 8.465 GPU and 1.238 CPU hours in 30 lines. With v1, the round used 8.804 of 12 GPU hours
  and 2.138 of 4 CPU hours, before the final id scan.

## Deviations and disclosures

- **The first launch record was refused.**
  - The queue compares the run identities in the launch record with its own, and at 15:46:48 they differed. No run
    started, and the refusal cost 0.00003 CPU hours.
  - The cause is that the official FACT constructor fills the Bu and BU block sizes of the head's configuration in place,
    and the run identity hashes that configuration. The revalidation check had computed its identities after building
    models. The queue, every run, the freeze, the training-stream fit and the report compute them from a fresh head.
  - A 12-second CPU replay on training data located the difference. The revalidation script now uses a fresh head
    (b355046), and a regression test runs the queue against such a record. The training code was not changed.
  - The preparation and the check were then run again (1636ada requires one script version for both). The second check
    used 0.0166 GPU hours, and the two checks together used 0.033 of the 0.05 hours allowed. v1's check record carries
    the same latent difference, but v1 never reached its queue.
- **The manifest proof was made by a stand-alone runner.**
  - After the local machine slept, its Python environment's torch files were offline iCloud placeholders, and importing
    the report module stalled.
  - The git-object proof was therefore made with the round's own `git_proof` and `verify_proof` functions, loaded from
    `scripts/report_segment_objective.py` without the module's other imports, and written in the same format.
  - The server verified it with the round's code before it opened the selection half.
- **The training-stream fit ran before the calibration stage.** The task sheet lists it with the evaluation. It reads
  training data only.
- **The monitor gained token counts.** These were added by A06-R1, are logging only and are listed in the configuration.
  The function was reviewed, and a test shows the same loss values and RNG state as before.

## Files

- `README.md`, this report.
- `revalidation_prepare.json` and `revalidation.json`, the revalidation records. The second is the launch record, with
  the gate, the inherited and the new checks kept apart.
- `preparation.json`, the new preparation.
- `train_fit.json`, the decoding of all training streams.
- `evaluation_manifest.json`, committed before the selection half was opened.
- `summary.json` and `report_tables.md`, every system row, comparison, verdict, accounting item, goal and reading.
- `training_curves.png` and `fact_tokens.png`.
- `code_review.md` and `fact_check.md`, the independent review and fact-check.

The frozen weights are in `models_frozen/fact_video_feature_v2/`. Per-sequence outputs and ids stay on the server.
