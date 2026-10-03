# Keeping intermittent A5C evidence, round 5 (2026-10-03)

Round 4 found that in many A5C segments the frozen classifier gives strong A5C scores on only some frames, and the
mean then reads A4C. This round tests one question set by the user. Does an A4C/A5C decision that keeps that
intermittent evidence improve classification without new A4C errors?

No echo expert is available. Every main result therefore keeps all clips and their public labels. Nothing here says
whether the high-score frames show a real outflow tract or whether any label is wrong. The paper stays paused.

- **Step 1** checks that the success criterion is reachable at all with the spans a decision would use. It uses only
  existing models.
- **Step 2** trains one small A4C/A5C classifier on the training split, with the encoder, frame classifier and temporal
  model frozen. It ran because step 1 held.
- **Validation only.** Both halves are development data. The test split was not read.

## Answer

1. **The target is reachable only on E1 ASRF's candidate spans.**
   - **Candidate spans.** These are the spans between the boundary head's accepted peaks, before ASRF's class vote
     merges equal neighbours. The boundary head already proposes 96–98% of the view changes on the selection half
     and 99% on the calibration half.
   - **The ideal decision.** An ideal A4C/A5C decision on those spans would reach a mean segment F1@50 of 0.959 and
     0.995, against targets of 0.944 and 0.982. It would also meet recall 0.95 and the native floor in every seed.
   - **The final segments fail.** On ASRF's final segments, seed 0 keeps a recall of 0.940 on the selection half, and
     relabelling cannot add cuts.
   - **New setting.** Candidate spans are therefore a new decoding setting, and all three rules below use exactly the
     same spans.
2. **Reading the distribution adds nothing over reading the mean.**
   - The classifier on distribution statistics and the classifier on mean probabilities show no consistent difference
     on either half: −0.5 to +0.4 points of segment F1@50 per seed.
   - They make almost the same decisions. At most one segment per seed and half differs in outcome.
   - Neither shows a consistent difference from the plain mean rule.
3. **On those spans, every rule loses against ASRF's own output.** All three rules are consistently lower than ASRF's
   default on the selection half, by 2.3 to 5.4 points. On the calibration half no consistent difference is detected.
   Seed 0 drops by 4.5 to 5.4 points there, and seeds 1 and 2 by 0.1 to 0.8.
4. **A few A5C segments come back, but false pieces multiply in both directions.**
   - Against ASRF's default, 3 to 5 A5C segments per seed and half become right. They come from 3 to 5 clips, so the
     gain is not concentrated in one or two.
   - 0 to 4 A4C segments become wrong.
   - The larger effect is new false pieces inside correct segments.
     - **A4C pieces inside A5C segments.** On the selection half these rise from 8, 18 and 19 (seeds 0, 1 and 2) to 26
       to 36.
     - **A5C pieces inside A4C segments.** These rise from 2–9 to 10–16.
   - Observation: within a recording, the classifier's A5C score fluctuates from sample to sample, and short spans cut
     from one A5C segment fall on both sides of the decision. Whether the fluctuation follows the cardiac cycle was not
     tested. That remains a hypothesis.
5. **No rule passes the success test.** On all clips the rules reach 0.873 to 0.876 on the selection half and 0.946 to
   0.952 on the calibration half. They also stay below the native floor on the selection half.
6. **Without the ten sensitivity clips, the conclusions do not change.** The distribution and mean classifiers still
   show no consistent difference. Their scores rise with ASRF's (to 0.936–0.940 and 0.969–0.976), and the rules show
   no consistent difference from ASRF's default there. This analysis removes 43 of the 120 streams in each half, so it
   shows dependence only.

**Reading.** Observations, hypotheses and conclusions are kept apart here.
- **Experimental conclusion.** In this form, a decision that reads statistics of the five class probabilities does not
  improve classification. It also adds A4C/A5C false pieces.
- **Observation (training data).** On the training split, the pattern appears in both classes: 9% of A4C units and 6%
  of A5C units have a mean p(A5C) below 0.5 with a 90th percentile above 0.8. The statistics beyond the mean raise
  training accuracy by only 0.1 to 0.3 points.
- **Observation (short spans).** The probability scores fluctuate within a recording, so a rule on short spans
  labels pieces of one segment differently.
- **Not shown.** The decider read only five class probabilities. It never read the 512-dimensional visual features,
  so this round does not show that those features lack usable information. Round 2 already fine-tuned the last ViT
  block, with frame-level and boundary losses. A span-level training objective has not been tested.

## Decision after this round (2026-10-03)

The user ended the probability-aggregation route. The next round is a paired visual experiment on actual short spans,
described in `configs/experiments/span_classifier_v1.yaml`. One model uses frozen visual features and trains only a
small head. The other also fine-tunes the last ViT block, its final normalisation and its projection. Both start from
the same weights and see the same spans, frames and batches.

## 1. What was fixed in advance

`configs/experiments/evidence_pooling_v1.yaml` holds everything.
- **Before any round-5 number:** the ten sensitivity clips and their rule, the spans, the oracle and the reachability
  test.
- **After step 1 held, before any training:** the decision rules, the features, the classifier, the training spans,
  the weights, the seeds and the report.

No setting was chosen on validation data. The 0.95 recall requirement was not relaxed. Each run first checked that
ASRF's decode reproduces its stored predictions exactly.

## 2. Step 1: reachability

The oracle relabels every span that ASRF voted A4C or A5C with its reference majority class, when that is A4C or A5C.
Targets are 0.944 (selection) and 0.982 (calibration), and the floors for E1 are 0.973 and 0.980.

| Base, spans | Half | Cut recall, s0 / s1 / s2 | Oracle segment F1@50 (mean) | Oracle recall, min | Oracle native F1, min | Reachable |
|---|---|---|---|---:|---:|---|
| E1 ASRF, final | selection | 0.942 / 0.950 / 0.953 | 0.952 | 0.940 | 0.989 | no |
| E1 ASRF, final | calibration | 0.977 / 0.972 / 0.977 | 0.989 | 0.972 | 0.997 | yes |
| E1 ASRF, candidate | selection | 0.979 / 0.963 / 0.969 | 0.959 | 0.961 | 0.989 | yes |
| E1 ASRF, candidate | calibration | 0.995 / 0.992 / 0.992 | 0.995 | 0.992 | 0.997 | yes |
| E0 ASRF, candidate | selection | 0.948 / 0.961 / 0.982 | 0.982 | 0.942 | 1.000 | no |
| E0 ASRF, candidate | calibration | 0.970 / 0.970 / 0.987 | 0.984 | 0.965 | 0.994 | yes |

E1 ASRF with candidate spans is the only ASRF base that passes on both halves. B3 was reported for reference only; E1
B3 passes with a margin of 0.3 points on the selection half, and E0 B3 does not pass on the calibration half
(`reachability.md`).

## 3. Sensitivity of the baselines (the ten clips removed)

| System | Selection, all → without | Calibration, all → without |
|---|---|---|
| E0 B3 tuned | 0.934 → 0.987 | 0.934 → 0.973 |
| E0 ASRF, s0 / s1 / s2 | 0.912 / 0.946 / 0.924 → 0.981 / 0.984 / 0.990 | 0.936 / 0.960 / 0.960 → 0.971 / 0.986 / 0.994 |
| E1 B3 tuned | 0.894 → 0.945 | 0.959 → 0.976 |
| E1 ASRF, s0 / s1 / s2 | 0.910 / 0.909 / 0.907 → 0.939 / 0.968 / 0.967 | 0.976 / 0.964 / 0.964 → 0.988 / 0.985 / 0.987 |
| B7 with B3 | 0.880 → 0.940 | 0.972 → 0.987 |

Segment F1@50 is shown. Recall and native macro-F1 are in `reachability.md`. The ten clips leave the native set, and
every stream with a fragment from them is removed: 43 of 120 streams in each half, 3 and 7 native clips, and 3 and 6
pairs. Nothing was re-tuned. These numbers show how much the scores depend on those clips. They decide nothing.

## 4. Step 2: the decision rules

All three rules act only on spans that ASRF voted A4C or A5C, and they choose in either direction. Every other span
keeps ASRF's vote, and equal neighbours merge afterwards. The evidence is the frozen E1 probe's per-sample class
probabilities inside the span.
- **mean** compares the mean p(A4C) with the mean p(A5C). This is the original mean-probability decision.
- **trained_mean** is a small classifier on the five mean class probabilities.
- **trained_distribution** is the same classifier on 17 fixed statistics. These are the five means; the 10th, 25th,
  50th, 75th and 90th percentiles and the maximum of p(A5C); the 10th, 50th and 90th percentiles of p(A4C); the spread
  of p(A5C); and the shares of samples above 0.5 and 0.8.

The classifier has one hidden layer of 32 units and trains for 300 full-batch epochs with a fixed cut-off of 0.5.
- **Training spans.** It uses about 6,350 training spans from 1,075 training clips. Every A4C or A5C reference segment of
  the training streams is used once exactly and once with its limits moved by up to 3 samples, and every A4C or A5C
  training clip counts as one span.
- **Weights.** Each source clip carries the same weight, and the two classes are balanced.
- **Seeds.** Three seeds were run, with decision seed k paired with ASRF seed k.

## 5. Step 2 results

| Rule | Selection: segment F1@50, s0 / s1 / s2 (mean) | Calibration: segment F1@50 (mean) | Recall, min (sel / cal) | Native F1, min (sel / cal) |
|---|---|---|---|---|
| ASRF default | 0.910 / 0.909 / 0.907 (0.909) | 0.976 / 0.964 / 0.964 (0.968) | 0.942 / 0.972 | 0.948 / 0.975 |
| mean | 0.856 / 0.883 / 0.879 (0.873) | 0.931 / 0.964 / 0.960 (0.952) | 0.958 / 0.990 | 0.959 / 0.983 |
| trained_mean | 0.863 / 0.883 / 0.880 (0.875) | 0.923 / 0.957 / 0.956 (0.946) | 0.958 / 0.990 | 0.951 / 0.983 |
| trained_distribution | 0.858 / 0.885 / 0.884 (0.876) | 0.922 / 0.960 / 0.956 (0.946) | 0.958 / 0.987 | 0.951 / 0.983 |

Paired per-seed contrasts of segment F1@50 (points; all clips; "none" means no consistent difference detected):

| Contrast | Selection | Calibration |
|---|---|---|
| trained_distribution − trained_mean | −0.5 / +0.3 / +0.4, none | −0.2 / +0.3 / 0.0, none |
| trained_distribution − mean | +0.1 / +0.2 / +0.5, none | −1.0 / −0.4 / −0.4, none |
| trained_mean − mean | +0.7 / 0.0 / +0.1, none | −0.8 / −0.6 / −0.4, none |
| each rule − ASRF default | −2.3 to −5.4, consistently lower | −0.1 to −5.4, none |
| each rule − strongest previous | −4.9 to −7.8, consistently lower | −0.8 to −5.0, none |

**False pieces.** These are unmatched predicted segments lying inside a reference segment of the other class.

| Rule | A5C pieces in A4C segments, selection / calibration (s0, s1, s2) | A4C pieces in A5C segments, selection / calibration | Predicted segments, selection |
|---|---|---|---|
| ASRF default | 9, 2, 2 / 6, 3, 3 | 8, 18, 19 / 2, 7, 8 | 498, 508, 510 |
| mean | 14, 10, 11 / 11, 5, 6 | 36, 30, 30 / 28, 9, 11 | 570, 542, 545 |
| trained_mean | 16, 10, 12 / 14, 9, 7 | 31, 28, 26 / 28, 9, 11 | 564, 538, 541 |
| trained_distribution | 13, 10, 10 / 14, 7, 7 | 35, 28, 27 / 30, 9, 11 | 566, 535, 539 |

Per-seed counts of A5C segments and native clips that turn right and of A4C ones that turn wrong are in
`decider_tables.md`. They are given against the mean rule and against ASRF's default, by segments, clips and
sessions. Results without the ten clips are there too.

## 6. Limitations

- Both halves are development data. Labels are public clip labels, and no expert has reviewed any clip.
- The frame evidence comes from one frozen classifier, the E1 probe. The training clips are in-sample for it. Their
  probabilities look like the validation ones (median mean p(A5C) of A5C clips 0.85 against 0.88), but the decider never
  saw out-of-sample probabilities in training.
- Candidate spans are one new decoding setting at ASRF's official boundary threshold. No other threshold was tried,
  by design.
- The bootstrap intervals resample sessions and streams but not repeated use of a clip across streams. They are
  optimistic, and the three seeds are not independent test sets.

## 7. Files and runs

- **Configuration:** `configs/experiments/evidence_pooling_v1.yaml`.
- **Code:**
  - `echo_routing/temporal/evidence_decider.py`, with tests in `tests/test_evidence_decider.py`;
  - `scripts/round5_reachability.py` and `scripts/summarize_reachability.py`;
  - `scripts/round5_evidence_decider.py` and `scripts/summarize_evidence_decider.py`.
- **Runs** (server):
  - reachability `runs/20261003-041639-md5_reachability-56ba86fd-s0`;
  - decider `runs/20261003-051405-md5_evidence_decider-4797336b-s0`. It holds the per-rule rows and every rule's
    predictions.
  - **Earlier runs.** Three earlier runs gave identical metrics, contrasts and false-piece counts: one before the
    false-piece counts were added, one before the span-constancy check of the code review, and one before ties
    between equally affected clips were broken by clip id. The tie fix changed only the share of the two most affected
    clips, in four rows with at most five segments; that share is fragile at such small counts.
- **In this folder:**
  - `reachability.md` and `reachability.json`;
  - `decider_tables.md` and `decider.json`.
