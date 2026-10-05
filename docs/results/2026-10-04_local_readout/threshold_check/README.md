# Round 7 threshold check (2026-10-04): post-hoc decision calibration of P0 and P1

P1 ranked A4C/A5C spans better than P0 in round 7, but every decision used the fixed threshold 0.5. This check asks
whether a threshold chosen on the calibration half turns that ranking into a full-system gain on the selection half.
The project lead set the check after the round-7 results.

It trains nothing. All weights, ASRF's candidate spans and votes, the decided spans and the merge rule are frozen, and
only the span scores saved by round 7 are used. It is a post-hoc decision-calibration diagnostic. It does not restart
P2 and is not a model contribution. The rules were committed before any threshold was evaluated
(`configs/experiments/threshold_check_v1.yaml`, commit 33ce14b). The test split was not read, and the paper stays
paused.

## Answer

Under the pre-declared terms, **there is no gain**. Neither arm shows a calibration gain, and P1's better ranking does
not turn into better decisions. The search on these fixed spans and readout scores ends.

1. **Chosen thresholds.** Every arm and seed had feasible thresholds (10 to 13 of 19). The chosen ones swing between
   seeds. P0 took 0.35, 0.65 and 0.60, and P1 took 0.65, 0.35 and 0.75. On the calibration half, where they were
   chosen, they add 0.1 to 0.9 points of segment F1@50 over 0.5. That figure is optimistic.
2. **Selection half.** Each threshold was applied once. Against the same arm at 0.5, segment F1@50 changes by
   +1.8 / −0.5 / −0.4 points for P0 and −0.3 / +0.1 / −0.3 for P1, with no consistent difference. With both arms
   calibrated, P1 − P0 is −2.0 / +0.1 / −0.3 points. In seed 0 the interval lies below zero.
3. **Against the references, on the selection half.**
   - Calibrated P1 is consistently below ASRF's default (−2.8 to −4.3 points).
   - Calibrated P0 is below it in every seed (−2.3 to −2.9) but not consistently by the interval rule.
   - Both are consistently below the strongest previous configuration (−4.7 to −6.7 points).
   - Neither meets the final goal.
4. **What a threshold moves.** On the selection half, the chosen thresholds changed 63 A5C span decisions against
   0.5, and 61 of them lie in the four common clips. They also changed 43 A4C span decisions, none of them in those
   clips. Lowering the threshold sets A5C spans in the common clips right and turns A4C spans elsewhere wrong.
   - In seed 0, P0 at 0.35 sets 15 A5C spans right, all from the common clips (3 clips, 2 sessions). It turns 14 A4C
     spans wrong, from 9 clips in 8 sessions.
   - Raising the threshold does the reverse. In seed 0, P1 at 0.65 sets 12 A4C spans right, from 9 clips, and turns
     9 A5C spans wrong in the common clips.

The common clips are the source clips of the A5C spans that every round-7 readout reads as A4C in every seed on the
selection half. There are four, and all were kept. Their roughly 40 erroneous spans are reused material, not 40
independent cases. They neither show that the labels are wrong nor that the visual information is insufficient.

**What this does not show.** It tests one decision rule on fixed scores: a single threshold per arm and seed, chosen on
the calibration half. As decided before the check, no further search on these spans and scores follows.

## 1. Thresholds chosen on the calibration half

| Arm | Seed | Feasible thresholds | Chosen | Segment F1@50 at the chosen threshold (at 0.5) |
|---|---|---|---|---|
| P0 | 0 | 13 of 19 | 0.35 | 0.9333 (0.9242) |
| P0 | 1 | 11 of 19 | 0.65 | 0.9638 (0.9601) |
| P0 | 2 | 10 of 19 | 0.60 | 0.9602 (0.9527) |
| P1 | 0 | 11 of 19 | 0.65 | 0.9375 (0.9286) |
| P1 | 1 | 10 of 19 | 0.35 | 0.9649 (0.9620) |
| P1 | 2 | 12 of 19 | 0.75 | 0.9611 (0.9602) |

A threshold was feasible when native macro-F1 reached 0.980 and boundary recall 0.95 on the calibration half.

## 2. Selection half

| Arm | Seed | Threshold | Segment F1@50 (at 0.5) | Boundary recall (at 0.5) | Native macro-F1 (at 0.5) |
|---|---|---|---|---|---|
| P0 | 0 | 0.35 | 0.8872 (0.8690) | 0.974 (0.974) | 0.935 (0.951) |
| P0 | 1 | 0.65 | 0.8802 (0.8855) | 0.958 (0.958) | 0.957 (0.968) |
| P0 | 2 | 0.60 | 0.8787 (0.8827) | 0.961 (0.963) | 0.968 (0.968) |
| P1 | 0 | 0.65 | 0.8674 (0.8701) | 0.971 (0.971) | 0.978 (0.951) |
| P1 | 1 | 0.35 | 0.8810 (0.8802) | 0.958 (0.958) | 0.961 (0.968) |
| P1 | 2 | 0.75 | 0.8760 (0.8785) | 0.961 (0.963) | 0.966 (0.968) |

For reference, ASRF's default scores 0.910 / 0.909 / 0.907 on this half and the strongest previous configuration
0.9344.

## 3. Start conditions and checks

- The decoded validation spans equalled round 7's saved spans, span by span, in every seed and half.
- At 0.5, the recomputed P0 and P1 systems equalled round 7's stored predictions for every sequence.
- The selection half was scored only at the chosen threshold and at 0.5.
- The report was rerun once to fix a duplicated word in its labels. Its numbers were byte-identical.

## 4. Files

- `report.json` and `report_tables.md` hold every table, the paired intervals and the changes with their clips and
  sessions.
- `grid_<arm>_s<seed>_calibration.csv` holds the calibration-half metrics for every threshold of the grid.
- The code is `scripts/round7_threshold_check.py`, run from the config above.
