# Corrected EV9V reproduction and evaluation audit (2026-09-30)

This round follows `docs/research_plan.md` ("问题 A 第一轮后的决定"). It delivers three tables:

1. a corrected model comparison;
2. the effect of frame order and of the sampled images;
3. a component audit of the pooled two-window detector.

**Scope of the results.**

- All results come from the EV9V validation and test splits, which already informed earlier decisions, so they are
  exploratory.
- The constructed streams join real clips with artificial, abrupt cuts.
- Every frame-derived input comes from `cache_v2`, which reads frames in numeric order.

## 1. Corrected model comparison

All rows use the same ladder configuration as run `20260928-001145`, on `cache_v2`. The table is built from three runs:

| Rows | Run |
|---|---|
| B0–B4, B7, B8 | `20261001-034125-baselines-17047dcb-s0`, with bootstrap in its `reports/bootstrap.md` |
| B3, audited setting | `20261001-035221-baselines-02a18614-s0` |
| B6 STFM | runs pending (see below) |

| Method | How its setting was chosen | Clip macro-F1 | Frame acc | Frag/min | Boundary F1, pairs | Boundary F1, 4–8 fragments | Error at 5%, multi thresholds | Policy 2 coverage / error |
|---|---|---|---|---|---|---|---|---|
| B0 frame classifier (ResNet-18) | trained once on EV9V train | 0.944 | 0.950 | 19.85 | 0.521 | 0.487 | 0.061 | 0.759 / 0.026 |
| B1 smoothing | ladder default | 0.947 | 0.965 | 2.72 | 0.933 | 0.851 | 0.093 | 0.782 / 0.038 |
| B2 HMM | ladder default | 0.940 | 0.965 | 2.51 | 0.930 | 0.833 | 0.088 | 0.759 / 0.030 |
| B3 change points | ladder default (0.5 s, JS 0.15) | 0.943 | 0.961 | 3.57 | 0.909 | 0.847 | 0.080 | 0.766 / 0.031 |
| B3 change points, audited | §3, chosen on validation (0.9 s, JS 0.31) | 0.930 | 0.973 | 0.69 | 0.980 | 0.927 | 0.083 | 0.741 / 0.018 |
| B4 MS-TCN (official code) | authors' constants, retrained on `cache_v2` | 0.947 | 0.973 | 1.51 | 0.923 | 0.876 | 0.088 | 0.848 / 0.024 |
| B5 ASFormer (official code) | authors' constants; retraining started on `cache_v2` | running | | | | | | |
| B6 STFM (official code), windowed | name-order seed-100 model, and patched seed-100 model | running | | | | | | |
| B7 EchoViewCLIP stage 1, windowed | authors' recipe, trained on MP4 in true order | 0.972 | 0.976 | 0.78 | 0.967 | 0.899 | 0.057 | 0.586 / 0.044 |
| B8 EchoPrime, per frame | released weights | 0.943 | 0.956 | 15.32 | 0.535 | 0.522 | 0.092 | 0.836 / 0.024 |

**Terms.**

- **Frag/min** counts label changes per minute inside clips that carry one source label.
- **Boundary F1** counts a predicted change as correct within 0.5 s of an artificial join between different source
  labels.
- **The error columns** report the share of routed time that is wrong.

**Paired bootstrap against MS-TCN** (run `20261001-034125`; pairs resampled by stream, multi streams by cine):

| Method | Pairs: boundary F1 | Pairs: same-view split rate | 4–8 fragments: boundary F1 | 4–8 fragments: same-view split rate |
|---|---|---|---|---|
| B1 smoothing | +0.010 [−0.037, 0.060] | −0.008 [−0.028, 0.000] | −0.025 [−0.085, 0.047] | +0.045 [0.011, 0.083] |
| B2 HMM | +0.007 [−0.032, 0.044] | −0.008 [−0.028, 0.000] | −0.043 [−0.112, 0.032] | +0.050 [0.012, 0.091] |
| B3 default | −0.014 [−0.069, 0.041] | 0.000 [−0.025, 0.025] | −0.029 [−0.083, 0.040] | +0.064 [0.019, 0.113] |
| B7 EchoViewCLIP | +0.044 [−0.012, 0.105] | −0.008 [−0.028, 0.000] | +0.023 [−0.045, 0.102] | −0.015 [−0.049, 0.020] |

**STFM, official evaluation.** This is the authors' video-level test on 888 clips and nine codes, a different task
from the ladder.

| STFM seed 100 | Accuracy | Macro-F1 |
|---|---|---|
| Name order, as in our reproduction | 93.92 | 90.14 |
| Numeric order (`third_party/patches/stfm-numeric-frame-order.patch`) | 94.48 | 90.90 |

- The patched run stopped early at epoch 25.
- The model is frozen as `models_frozen/stfm_numeric/s100`; the retrained MS-TCN is frozen as
  `models_frozen/mstcn_official/v2_frameorder`.
- This is one paired seed. It shows the size of one change, not stability across seeds.

**Reading.**

- With frames in order and the same inputs, the official MS-TCN shows no clear advantage over smoothing, HMM or
  default change points on boundary F1; every interval includes zero.
- MS-TCN does still split same-view joins less often in 4–8-fragment streams.
- EchoViewCLIP, a video model trained on frames in true order, scores highest among the published models on clip
  macro-F1 and boundary F1, but routes the least at 5% error.
- B3 with the audited setting has the highest boundary F1 in the table and few cuts inside single-label clips.
  - Its setting was chosen on validation by the problem A rule, while the published models keep their authors'
    settings.
  - This comparison is not paired with the main run.
- Learned temporal models therefore do not hold a clear, consistent advantage on this corrected benchmark, but this
  rests on one training run per model.

## 2. Order and sampled images

Protocol: `scripts/order_set_audit.py`, output in `runs/_demo/order_set_audit/`.

- **Frame identity.** Each sample is identified by the true frame number of the image it was computed on.
- **Frozen sets.** For every fragment of every recipe and for every native clip, the legacy image set (S_L) and the
  corrected image set (S_N) are fixed first.
- **Orders.** Each set is then put in name order, in true order, or in one seeded random order per fragment.
- **Time axis.** A uniform position grid: each sample is one 0.1 s step whatever its true frame number. Joins stay in
  place, the tolerance is 5 samples, and rates use n × 0.1 s. This is not physical time.
- **Sanity checks.** The (legacy, name) condition reproduces the legacy ladder exactly, and (new, true) reproduces the
  corrected ladder exactly.

**How much the image sets differ.**

| Comparison | Units | Median Jaccard (10th–90th percentile) | Identical |
|---|---|---|---|
| Legacy vs corrected, constructed-stream fragments | 2,393 | 0.12 (0.00–0.22) | 0 |
| Legacy vs corrected, whole native clips | val + test clips | 0.21 (0.16–0.25) | |
| Legacy vs the earlier decomposition (whole clip sorted, then cut) | 2,393 fragments | 0.60 (0.00–0.93) | 203 |

The earlier decomposition therefore mixed order with image set, and its attribution is withdrawn
(`docs/results/2026-09-30_frame_order_fix/`).

**Results.**

| Images | Order | B0 clip macro-F1 | B0 frag/min | B0 boundary F1, pairs | B1 frag/min | B1 boundary F1, pairs | B1 boundary F1, multi | B1 policy 2 coverage |
|---|---|---|---|---|---|---|---|---|
| Legacy | name (the old ladder) | 0.957 | 23.1 | 0.499 | 2.36 | 0.918 | 0.852 | 0.815 |
| Legacy | true | 0.957 | 19.2 | 0.523 | 2.69 | 0.930 | 0.853 | 0.874 |
| Legacy | random | 0.957 | 36.8 | 0.395 | 1.45 | 0.979 | 0.891 | 0.901 |
| Corrected | name | 0.944 | 24.7 | 0.512 | 2.18 | 0.930 | 0.866 | 0.791 |
| Corrected | true (the corrected ladder) | 0.944 | 19.9 | 0.521 | 2.72 | 0.933 | 0.851 | 0.782 |
| Corrected | random | 0.944 | 37.1 | 0.417 | 1.33 | 0.952 | 0.890 | 0.925 |

Effects for all four methods are in `conditions.json`.

**Reading.**

- **B0's clip score depends only on the images.** It does not move with order, because it is a majority vote. It
  falls by 0.013 from the legacy to the corrected image set.
- **Order matters most for unsmoothed per-frame output.** True order removes about 4–5 B0 label changes per minute
  compared with name order, and shuffling adds about 17.
- **The smoothing baselines (B1–B3) move by about 0.02 or less in boundary F1 between name and true order.** Their
  policy 2 coverage falls mainly when the image set changes, by 0.09–0.12 at true order.
- **Shuffling within fragments makes the smoothing baselines look better.** It cuts 1.3–1.9 fewer times per minute,
  raises boundary F1 by 0.02–0.07 and raises policy 2 coverage by 0.11–0.14. Real within-clip temporal structure
  makes this proxy task harder, not easier. Evaluations on shuffled frames can therefore flatter temporal smoothing.
- **MS-TCN was not decomposed.** That needs retraining under this protocol, which waits until the protocol is agreed.

## 3. Pooled detector: component audit (fixed B0)

Script: `scripts/pooling_audit.py` (module `echo_routing/realvideo/two_window.py`, tests in
`tests/test_two_window.py`). Output: `runs/_demo/pooling_audit/`.

**Pipeline.** Every configuration uses B3's own decoding:

- suppression of weaker peaks within 0.5 s;
- the requirement that the windows before and after a peak name different views;
- each segment labelled by its mean probability.

With B3's settings, the module reproduces B3 exactly; this is tested.

**Configurations.** Each changes one component relative to the row above it.

**Operating point.** Validation recall of different-label joins ≥ 0.95 within 0.5 s, then the fewest extra cuts per
minute in native clips; frozen for test.

| Config | Change | Chosen setting | Test recall | Test extra/min, native | Test extra/min, constructed | Same-label join cuts, multi | Decode time, all settings |
|---|---|---|---|---|---|---|---|
| A0 B3 as configured | B3's raw threshold grid | 0.5 s, JS 0.3 | 0.954 | 1.44 | 1.00 | 0.099 | 1.9 s |
| A1a | thresholds at fixed quantiles of validation scores | q 0.998 | 0.940 | 0.79 | 0.56 | 0.074 | 2.5 s |
| A1b | window choice 0.5 / 0.9 / 1.8 s | 0.9 s, q 0.998 | 0.944 | 0.54 | 0.44 | 0.064 | 8.5 s |
| A2 | TV instead of JS (the period-free detector) | 0.9 s, q 0.998 | 0.944 | 0.67 | 0.54 | 0.089 | 7.0 s |
| A3 | window = K × estimated period, one quantile map | K 0.5, q 0.998 | 0.942 | 0.70 | 0.47 | 0.069 | 8.1 s + period estimation |
| A4 | separate quantile maps with and without a period | K 2, q 0.998 | 0.946 | 0.59 | 0.54 | 0.064 | 9.9 s + period estimation |
| Reference: MS-TCN (problem A) | | min run 9 | 0.946 | 0.61 | 0.44 | 0.094 | |

**Budgets and cost.**

- A1b–A4 each search 3 windows × 10 thresholds.
- Period estimation for A3 and A4 took 5.7 minutes of wall-clock time on 4 workers for all 2,041 streams, far more
  than any decoding step.
- The median localisation error was 0 samples for every configuration.

**Paired test differences** (native clips resampled by presumed patient group, recall by stream):

| Contrast | Extra cuts/min, native [95% CI] | Recall [95% CI] |
|---|---|---|
| A1a minus A0 (threshold parameterisation) | −0.658 [−0.927, −0.389] | −0.014 [−0.024, −0.005] |
| A1b minus A1a (window choice) | −0.245 [−0.457, −0.025] | +0.004 [−0.007, +0.015] |
| A2 minus A1b (TV instead of JS) | +0.129 [+0.040, +0.238] | 0.000 [−0.006, +0.006] |
| A3 minus A2 (period-adaptive window) | +0.026 [−0.123, +0.185] | −0.002 [−0.010, +0.004] |
| A4 minus A3 (two quantile maps) | −0.103 [−0.253, +0.039] | +0.004 [−0.004, +0.013] |
| A2 minus MS-TCN | +0.065 [−0.225, +0.340] | −0.002 [−0.022, +0.019] |

**Reading.**

- **Most of the earlier gap between the pooled detector and B3 came from the threshold grid.** B3's raw grid could not
  place its operating point near the 95% recall target on validation. The finer quantile grid picks a stricter point,
  but test recall is then 0.014 lower, so part of this difference is a trade of recall for fewer cuts, not a better
  detector. A single quantile map is monotone, so it only re-parameterises the threshold.
- **Choosing a fixed 0.9 s window helps a little.**
- **TV is slightly worse than B3's JS divergence.**
- **Neither the period-adaptive window nor the two-group calibration gives a clear further gain,** and estimating the
  period costs far more than decoding.
- **Conclusion.** B3, given a window choice and a threshold grid that can reach the operating region, reproduces the
  gain. The result is therefore recorded as a strong simple baseline (§1, "B3, audited"), not as a pooling method.

## Pending in this round

- B5 ASFormer: retraining on `cache_v2` with the authors' constants is running (about 12 hours); its row will be added.
- B6 STFM windowed: inference with the name-order seed-100 model and with the patched seed-100 model is running into
  `cache_v2/stfm_windowed/79f4a41af6e3_{frozen_s100,numeric_s100}`; its rows will be added.
