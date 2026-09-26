# Routing thresholds calibrated on multi-fragment streams (2026-09-26)

Ladder run `20260926-190831-baselines-1b027c50-s0`. It uses the same config, feature cache (79f4a41af6e3) and prediction
directories as the ViT-family run `20260925-130154`. B-file and B1–B3 were added.

## What changed

- **Second frozen policy.** Thresholds are now also selected on the 120-stream multi-fragment validation bank
  (4–8 fragments, class-balanced). Test routing is reported for every target (1%, 5%, 10%) under both validation
  banks (`routing_by["pair@t"]`, `routing_by["multi@t"]`).
- **Per-fragment count tables** (`fragments/<method>.<bank>.csv`) feed a paired bootstrap
  (`scripts/bootstrap_ladder.py`, 2,000 draws).
  - Pair streams use each cine once, so they are resampled by stream.
  - Multi-fragment streams reuse cines (405 cines, 701 fragments, up to 11 appearances), so they are resampled by
    cine.
  - There are no patient ids, so the intervals are descriptive.
- **Integer sample positions for pooled boundary matching.** Pooled matching now uses exact sample positions instead
  of seconds, so it agrees exactly with the missed and false-split counts.
- **Continuity check.** All 3,024 metric fields of the previous run are reproduced exactly. Only the note text changed.

## Findings

1. **The pair streams are saturated.**
   - Change F1 is 0.992 (0.979–1.000) for MS-TCN, 0.988 (0.967–1.000) for ViT-Router and 0.980 (0.950–1.000)
     for ASFormer.
   - All three route essentially everything, with errors of 0.3–0.8%.
   - No method can be separated from the others here.
2. **On the 4–8 fragment streams the 5% target does not hold, even with thresholds selected on matching
   multi-fragment validation streams.** Test error at the multi-bank threshold:

   | model | test error | test coverage |
   |---|---|---|
   | MS-TCN | 0.090 (0.045–0.143) | 0.999 |
   | ASFormer | 0.087 (0.043–0.139) | 0.998 |
   | ViT-Router | 0.102 (0.054–0.159) | 0.859 |
   | Echo-MViT + centre | 0.087 (0.059–0.121) | 0.817 |
   | STFM | 0.050 (0.035–0.070) | 0.605 |

   STFM meets the target, but only by routing 60% of the time. MS-TCN's validation error at its threshold was 0.046,
   so the failure is a shift from validation to test, not a selection bug.
3. **A5C causes most of the error.**
   - Routed error time in A5C fragments is 77% of the total for MS-TCN, 79% for ASFormer and 86% for ViT-Router.
     Their A5C error rates are 34%, 33% and 50%.
   - A4C accounts for most of the rest. PLAX and PSAX are near zero.
   - The validation split has 29 A5C cines and test has 40, and a few hard A5C cines recur across streams. This is
     why the intervals are wide.
4. **The confidence of the learned temporal models does not flag these errors.**
   - To reach 1% error on validation, MS-TCN, ASFormer and ViT-Router keep only 7–12% of the time.
   - The JS-divergence baseline B3, working on frame probabilities, keeps 89%.
   - On test at the 1% target, B3 routes 85% at 2.5% error, while MS-TCN routes 11% at 4.1%.
   - No method meets 1% on test with useful coverage.

## Implications for the next modelling step

- **The dominant failure is recognition, not boundary placement.** The temporal models confidently read A5C as A4C.
  P0–P2 (boundary head, balanced nuisance pairs, consistency) targets boundaries and nuisance edits, so it is
  unlikely to fix this on its own.
- **Cheap experiment 1: routing confidence.** All predictions already exist, so this runs on CPU only.
  - Score each segment of the temporal model with frame-level evidence instead of its own smoothed probabilities.
  - Add temperature or per-class calibration.
  - Select thresholds with an upper confidence bound on validation risk, resampled by cine, instead of the point
    estimate.
- **Cheap experiment 2: stronger A5C evidence.** Feed a clip-level recognizer into the stream model, either
  Echo-MViT + centre or EchoViewCLIP windows. On native clips Echo-MViT reaches A5C recall of about 0.95, against
  0.57 for ViT-Router.
- **Then P0–P2**, with the multi-fragment streams as the main constructed benchmark.

## Files

- `multi_calibration.md`: the three tables (multi routing under both threshold sources, paired differences to
  MS-TCN, pair streams).
- `bootstrap.md` / `bootstrap.json`: full bootstrap output for every endpoint and method.
- `ladder.md` / `ladder.csv`: the ladder with the new multi-fragment columns.
