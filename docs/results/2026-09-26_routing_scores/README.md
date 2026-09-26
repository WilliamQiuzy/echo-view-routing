# Routing confidence and threshold rules (2026-09-26)

Run `20260926-193436-routing_scores-1b027c50-s0` (`scripts/routing_scores.py`, tables from
`scripts/report_routing_scores.py`). It follows `../2026-09-26_multi_calibration/`, which found that on the 4–8
fragment streams the 5% target fails and most routed error is confidently misread A5C.

## Setup

- **Segments are unchanged.** They come from each method's own decoded labels, exactly as in the ladder. Only the
  confidence of each segment and the threshold rule change.
- **Scores.**
  - `own_p10`: ladder default, the 10th percentile of the segment's class probability times agreement.
  - `own_mean`.
  - `ownT_p10`: own probabilities temperature-scaled on validation.
  - `frame_p10` and `frame_mean`: ResNet-18 (B0) frame probabilities inside the method's segments.
  - `min_own_frame`: the minimum of `own_p10` and `frame_p10`.
- **Threshold rules.** All are selected on the 120 multi-fragment validation streams (334 cines) and then frozen.
  - `point`: the ladder rule.
  - `ucb`: 95% upper bound of the validation risk under a bootstrap that resamples cines.
  - `point_class` and `ucb_class`: one threshold per predicted class, each meeting the target within its class.
- **Test data.** 120 multi-fragment test streams (405 cines). Error intervals resample cines, 1,000 draws. There are
  no patient ids, so the intervals are descriptive.
- **Selection.**
  - The score is chosen on validation only, as the lowest mean validation AURC over the 24 methods.
  - The rule is `ucb_class`, chosen for what it is designed to do: a conservative bound, per predicted class, that
    respects cine reuse.
  - One caveat: the MS-TCN smoke run, which showed every rule on test, was seen before the full run. The table
    "every score and rule" therefore reports all 24 combinations.

## Findings

1. **Frame evidence ranks errors about twice as well.**
   - Mean validation AURC is 0.019 for `frame_mean` against 0.038 for `own_p10`. Test AURC is 0.027 against 0.058.
   - Mean hindsight coverage at 1% rises from 0.22 to 0.49.
   - Temperature scaling changes nothing (0.0377). Rescaling the probabilities does not fix their ranking.
   - The learned temporal models smooth away the frame-level doubt that marks A5C segments.
2. **Per-class thresholds with a cine-level upper bound restore the 5% target.**
   - With `frame_mean` + `ucb_class`, test error stays at or below 5% for 24 of 24 methods (median coverage 0.68).
   - The ladder rule holds for 3 of 24.
   - Both parts matter. With `frame_mean`, the global upper bound holds for 8 of 24 and per-class point estimates
     for 13 of 24.
3. **The strongest temporal models keep most of the recording.** On 4–8 fragment test streams, coverage / error
   (interval), ladder rule versus new rule:

   | model | 5% target, ladder | 5% target, new | 1% target, ladder | 1% target, new |
   |---|---|---|---|---|
   | MS-TCN | 1.00 / 0.090 | 0.85 / 0.027 (0.009–0.053) | 0.11 / 0.041 | 0.71 / 0.004 (0.001–0.008) |
   | ASFormer | 1.00 / 0.087 | 0.83 / 0.026 (0.008–0.051) | 0.15 / 0.036 | 0.72 / 0.014 (0.004–0.028) |
   | ViT-Router | 0.86 / 0.102 | 0.82 / 0.028 (0.008–0.058) | 0.10 / 0.008 | 0.75 / 0.017 (0.005–0.034) |
   | B3 JS divergence | 1.00 / 0.089 | 0.88 / 0.036 (0.015–0.065) | 0.85 / 0.025 | 0.66 / 0.007 (0.000–0.016) |
   | EchoViewCLIP | 0.61 / 0.066 | 0.59 / 0.037 (0.028–0.047) | 0.06 / 0.007 | 0.26 / 0.003 (0.001–0.007) |
   | Echo-MViT + centre | 0.82 / 0.087 | 0.71 / 0.043 (0.031–0.056) | 0.00 / — | 0.35 / 0.008 (0.005–0.012) |

   - The A5C share of routed error falls from 77% to 14% for MS-TCN, from 79% to 23% for ASFormer and from 86% to
     37% for ViT-Router.
   - On the easy pair streams the cost is small. MS-TCN and ASFormer route 0.95 at an error of 0.2–0.3%.
4. **The 1% target is not yet reliable.**
   - It holds for 16 of 24 methods, at median coverage 0.35.
   - Several strong models land slightly above 1%, for example ASFormer 1.4% and ViT-Router 1.7%. Their intervals
     include 1%.
   - With 29 A5C cines in validation, the data cannot certify a 1% error.

## Implications

- **Routing policy.** Segment with the temporal model and score with frame evidence. Use per-class thresholds from a
  cine-resampled upper bound. This should become the ladder's second routing policy, reported next to the legacy
  rule. It is a concrete result for hypothesis H2: separating segmentation from confidence.
- **Clip-level A5C evidence (step b)** now mainly buys coverage. A5C errors are already deferred, so a stronger A5C
  signal would let more of that time be routed automatically.
- **1% target.** Reporting 1% needs more independent validation cines. Otherwise treat it as exploratory.

## Files

- `summary.md`: all tables (ranking quality, ladder versus new rule, every score and rule over all methods, the
  chosen policy for every method).
- `results.json`: every method, score, rule and target, with thresholds, validation and test coverage and error,
  intervals, pair-stream numbers and the A5C share of error.
