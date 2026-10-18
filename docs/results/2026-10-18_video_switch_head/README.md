# A10: an explicit switch head on the A08 all-neighbour model (2026-10-18)

**Question.** A10 starts each seed from its frozen A08 A model. Can a learned decision about where to cut and where not
to cut beat the strong references while keeping A08's class decisions? This is a known multi-task idea; no new
algorithm is claimed. The development halves were reused for exploratory feedback, and the test split and
EchoNet-Pediatric were not read. Configuration: `configs/experiments/video_switch_head_v1.yaml`, fixed before any
output (4c38d13, 12c80c6).

**Design.**
- **Arms.** Three paired arms per seed start from that seed's frozen A08 all_s{k}. The same weights in eval mode are the
  teacher.
  - **R:** CE + KL(teacher ‖ student), decoded with B3.
  - **P:** R plus 0.5 × balanced BCE of a small switch head that reads the adjacent windows' class probabilities
    (545 parameters).
  - **V:** the same head width, but it also reads the adjacent 512-d embedding difference (16,929 parameters).
- **Batches.** Each update has 12 windows: A07's native group and its change-containing group, plus a group at a join of
  two same-label recordings. That join group is the hard negative.
- **Edges.** Per seed there are 1,546 positive edges, 11,857 negatives (about 1,590 of them join negatives) and 97
  masked edges.
- **Schedule.** 1,350 updates with full cosine decay. The schedule was shortened from 1,500 before any formal run,
  because the projection exceeded the screen budget.
- **Checks passed.** These covered edges, pairing (identical logits and RNG across arms), gradients, the teacher
  receiving none, resume against an identical rerun, and the A08 start parity (0.0).
- **Decoding.** R uses B3, with 40 settings. P and V use their head: edge scores feed the existing `boundary_decode`,
  which applies NMS 5, names segments by mean-probability argmax and merges adjacent same-class segments. The grid has
  8 quantiles of the model's own calibration native edge scores, and the choice is made once on calibration.

## Results (seeds 0 and 2; four cells = 2 seeds × 2 halves)

Definitions:
- F is multi segment F1@50.
- FP and FN are false and missed multi-stream segments; "short" means at most 16 samples.
- Join cuts are cuts at the 862 same-label joins of the four cells.

| System | Mean F | Mean F − A08 A | Selection FP (short) | FN (missed short true) | Mean boundary recall / precision | Native A4C→A5C / A5C→A4C | Join cuts |
|---|---|---|---|---|---|---|---|
| Frozen Ordered + B3 | 0.9661 | −0.0037 | 54 (29) | 62 (29) | 0.974 / 0.965 | 8 / 0 | 15 |
| A08 A + B3 | 0.9698 | – | 49 (22) | 51 (22) | 0.979 / 0.966 | 6 / 1 | 19 |
| R + B3 | 0.9712 | +0.0013 | 52 (30) | 43 (17) | 0.983 / 0.963 | 6 / 1 | 17 |
| P head | 0.9671 | −0.0027 | 68 (39) | 48 (16) | 0.983 / 0.959 | 6 / 1 | 20 |
| V head | 0.9553 | −0.0145 | 109 (78) | 51 (10) | 0.986 / 0.936 | 5 / 0 | 24 |
| P + B3 (sensitivity) | 0.9707 | +0.0010 | 55 (30) | 41 (16) | – | – | – |
| V + B3 (sensitivity) | 0.9692 | −0.0006 | 55 (30) | 48 (15) | – | – | – |

**Per half.**
- **Calibration.** Every new system is within about 0.003 of A08 A.
- **Selection.** Here the head decoders separate:
  - P loses 0.0024 and 0.0102 against A08 A.
  - V loses 0.0277 and 0.0284. Its per-seed paired intervals lie below zero, as do those of V − P (−0.025 and −0.018).
  - R − A08 A has no interval clearly away from zero in any cell.
- **Head thresholds.** Calibration chose the 0.995 quantile for P and the 0.97 quantile for V in both seeds.

**Fixed third-seed rule: not met by any arm**, so A10 closes with two seeds and no three-seed verdict.
- **R.** No signal: +0.0013 against the required +0.003.
- **P.** No signal, and a selection reversal: the selection mean is −0.0063.
- **V.** No signal, and a selection reversal: the selection mean is −0.028.

All three pass the no-harm part: native macro-F1, boundary recall and missed short true segments are not worse than
A08 A's.

## What this round shows

1. **The common continuation (R) keeps A08 A's level.** Mean F is +0.0013, with fewer missed segments (43 against 51)
   and as many false segments on selection (52 against 49). CE + KL with join-negative groups is a stable continuation,
   not a gain.
2. **Cutting from the same probabilities does not beat B3.** P's head is below B3 applied to the same weights' outputs
   (P + B3) in three of four cells. On selection it adds 13 false segments and 4 missed ones relative to P + B3. So converting
   window probabilities into cuts is not the missing piece in this test. Probability calibration itself (for example a
   temperature) was not tested and remains **undetermined**.
3. **The video embedding carries extra change evidence, but not enough specificity.**
   - V's head misses the fewest short true segments (10, against 16 for P and 22 for A08 A) and has the highest boundary
     recall.
   - On selection it makes 109 false segments, against 68 for P and 49 for A08 A. Most are short (78), and most are not
     at same-label joins (13 join cuts).
   - The calibration-chosen threshold did not transfer to selection, where calibration stayed level.
   - Whether a transferable working point or a more specific switch feature would turn this recall into a full-system
     gain is **undetermined**. No matched test was run, and no selection-side threshold was searched, by rule.
4. **Strong references.** No A10 arm beats A08 A. R is level, P is slightly below and V is clearly below on selection.

## Cost and files

- **GPU.** 2.377 h of 8, made up of three checks (0.066 h), six runs (1.913 h) and the screen evaluation (0.398 h).
- **CPU.** Booked in `runs/video_switch_head_v1/ledger.jsonl`, including the agent's active wall clock.
- **Frozen models.** `models_frozen/video_switch_head_v1/{R,P,V}_s{0,2}` (sha256 1db6ccd4, 12427b51, f3ee9417, 858c4e25,
  9bdaf93b and 76e29686).
- **Records.** `screen.json` (rows, counts, choices with the head grids, paired comparisons, accounting and the
  third-seed decision), `checks.json` and `preparation.json`. Per-window outputs and embeddings stay on the server.
