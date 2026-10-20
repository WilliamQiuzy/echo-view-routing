# A11: a sustained-switch readout with training hard negatives (2026-10-20)

**Question.** Can a switch head separate brief appearance changes from sustained view changes, keep A08's recognition
and short true segments, and cut fewer false segments than A10's video branch? This is a hypothesis check of a known
multi-task idea, run on the reused development halves for exploratory feedback. The test split and EchoNet-Pediatric
were not read. Configuration: `configs/experiments/video_sustained_switch_v1.yaml` (b9db5d9), fixed before stage one,
the mining and any training.

**Stage one: a decoding check on A10's cached outputs (CPU).** A10's head decoder lacked B3's left/right argmax check, so
the check was added to the same candidates, NMS and segment naming.
- **P.** The check changes nothing: identical rows at both thresholds and in both halves.
- **V, selection.** The check trades false cuts for missed true changes.
  - Seed 0 improves from 0.929 to 0.934 at the original threshold and to 0.938 at the recalibrated one. False segments
    fall from 53 to 32, while missed segments rise from 21 to 30.
  - Seed 2 falls from 0.928 to 0.906. Missed segments rise from 19 to 46, and boundary recall drops from 0.987 to 0.927.
- **V, calibration.** The check is worse for both seeds.
- **Recovered segments.** The check keeps 5 to 6 of the 6, and 6 of the 8, segments that V had recovered against A08 A.
- **Decision.** Under the fixed mapping, the mean F1 over the 8 cells falls from 0.9612 to 0.9565, so A11's decoder does
  not use the check.

**Design.**
- **Arms.** U and H, paired within each seed. Each starts from its seed's frozen A08 A, with the same weights as an
  eval-mode teacher, and trains with CE + KL + 1.0 × BCE for 1,350 updates. λ = 1.0 because the BCE/CE gradient ratio at
  the start was 0.02.
- **Head.** One central-edge head (33,921 parameters) reads the 4 adjacent windows of each group:
  - the endpoint probabilities and the two-sided mean probabilities;
  - unit-normalised embeddings with local and two-sided differences, their norms and cosine distances;
  - no layer normalisation.
- **Batches.** Each update has 12 windows: A07's change group, A10's same-label join group, and on odd updates a
  hard-negative pool group in place of the random native group.
- **Pool.** The pool was 1,952 groups, not the planned 2,700. The fixed cap of 4 groups per session was reached by the
  native draws, so only 602 same-label stream groups entered. The frozen A10 V heads scored the pool once, and the top
  1,350 groups form the hard set.
- **H against U.** U uses balanced BCE. H gives raw weight 3 to the top quartile of frozen scores within each source and
  rescales to U's negative mass; its positives equal U's.
- **Checks passed.** These covered edges, the weight masses, pairing (identical logits and RNG), gradients, the teacher
  receiving none, resume against a rerun, and A08 parity (0.0).

## Results (seeds 0 and 2; four cells)

Definitions:
- F is multi segment F1@50; the head-decoded systems are primary.
- FP and FN are false and missed multi-stream segments; "short" means at most 16 samples.

| System | Mean F | Mean F − A08 A | Selection FP (short) | FN (missed short true) | Boundary recall / precision | Native macro-F1 | A4C→A5C / A5C→A4C |
|---|---|---|---|---|---|---|---|
| Frozen Ordered + B3 | 0.9661 | −0.0037 | 54 (29) | 62 (29) | 0.974 / 0.965 | 0.9819 | 8 / 0 |
| A08 A + B3 | 0.9698 | – | 49 (22) | 51 (22) | 0.979 / 0.966 | 0.9839 | 6 / 1 |
| U head | 0.9659 | −0.0039 | 66 (37) | 53 (18) | 0.984 / 0.963 | 0.9880 | 5 / 0 |
| H head | 0.9686 | −0.0012 | 62 (35) | 46 (13) | 0.987 / 0.964 | 0.9862 | 5 / 1 |
| U + B3 (sensitivity) | 0.9603 | −0.0095 | 92 (67) | – | – | – | – |
| H + B3 (sensitivity) | 0.9609 | −0.0089 | 92 (68) | – | – | – | – |
| A10 V head (historical) | 0.9553 | −0.0145 | 109 (78) | 51 (10) | 0.986 / 0.936 | – | – |

**Per cell, H head against A08 A.**
- Seed 0 selection: +0.0079 (0.9643 against 0.9563).
- Seed 0 calibration: +0.0029.
- Seed 2 selection: −0.0127 (0.9437 against 0.9564; false segments 42 against 25).
- Seed 2 calibration: −0.0030.
- All four per-seed paired intervals include zero.

**Per cell, U head.** +0.0048, +0.0019, −0.0195 and −0.0029. The seed-2 selection interval lies below zero.

**H − U (heads).**
- Selection: +0.0031 [0.000, 0.007] and +0.0067 [0.000, 0.016].
- Calibration: +0.001 and 0.000.
- The direction favours H in both seeds, but with two seeds and intervals touching zero this is not established.

**Recovered and lost segments, H against A08 A (all four cells).** H recovers 16 reference segments (12 short) and loses
11 (3 short). It adds 44 unmatched predicted segments, 30 of them in seed 2 on selection, and removes 33.

**Fixed third-seed rule: not met**, so A11 closes with two seeds.
- **U.** No signal (−0.0039), and a selection reversal (−0.0073).
- **H.** No signal (−0.0012). Its no-reversal and no-harm parts pass.

## What this round shows

1. **The video head's false cuts can be reduced without losing short true segments.** Against A10's V head, H's
   selection false segments fall from 109 to 62, missed short true segments stay low (13, against 22 for A08 A),
   boundary recall stays high and native recognition is kept.
   - This is a whole-recipe effect, from the sustained readout, magnitude-preserving features, hard-pool groups and
     training data together. No single part is credited.
2. **It is still not better than A08 A, and it is seed-dependent.** Seed 0 is ahead of A08 A in both halves. Seed 2
   loses on selection.
   - Seed 2's own CLS outputs through B3 also degrade on selection: 64 false segments, 52 of them short, against 25 for
     A08 A.
   - So part of the seed-2 loss arises in the encoder continuation, not only in the head.
3. **Hard-negative weighting points the right way but is not established.** H ≥ U in every cell, with selection gains of
   +0.003 and +0.007 and intervals touching zero.
4. **The left/right decoding check is not the missing piece.** On A10's V outputs it trades false cuts for missed true
   changes. It is not used.

## Cost and files

- **GPU.** 1.898 h of 6: mining 0.331, check 0.021, four runs 1.281 and evaluation 0.265. Stage one was 11 seconds of
  CPU.
- **CPU.** Booked in `runs/video_sustained_switch_v1/ledger.jsonl`, including the agent's active wall clock.
- **Frozen models.** `models_frozen/video_sustained_switch_v1/{U,H}_s{0,2}`.
- **Records.**
  - `stage_one.json`.
  - `preparation.json`: the pool, scoring, hard set, central-edge counts per source and the U/H masses.
  - `checks.json`.
  - `screen.json`: rows, counts, choices with the head grids, comparisons, accounting, A10 historical rows and the
    third-seed decision.
- **Server only.** Per-window outputs and the mined scores stay on the server.
