# A25: expected segment-error risk on short groups of training windows (2026-10-23)

**Question.** Starting from the strong A08 system, does a differentiable expected segment-error count on groups of
consecutive training windows reduce false short segments and isolated flips while keeping true short segments, and raise
full segmentation, against the identical training without it?
- **Arm.** Seg is A10 R's recipe exactly (A08 all_s{k} start, A10's pinned 12 windows per update in three groups of four
  consecutive request windows, CE + KL to the cached own A08, A10's optimiser, schedule and seeds), plus λ · L_seg.
- **Control.** A10's frozen R_s0/R_s2 are the same recipe with λ = 0. They were reused, not retrained, after a λ = 0
  replay of their logged training.
- **How this differs from earlier rounds.**
  - It is not a reweighted per-window CE.
  - It is not A05/A07's run-balanced CE or transcript-forward likelihood.
  - It is not A08's neighbour smoothing or A24's teacher-referenced difference relation.
  - It adds no module, data, teacher or decoder, and it does not repair A23.
  - The term charges the joint output pattern of four windows by the F1@50 matching rule.
- **A24's blind spot.** In A24's constant-offset example (teacher (0.8, 0.2), (0.55, 0.45), (0.8, 0.2), student offset
  (−0.1, +0.1)) the A24 relation is 0. The expected segment-error count changes from 1.884 to 2.221.
- **Data limits.** The check pool is not a new holdout, and the development halves are reused exploratory feedback. The
  test split and EchoNet-Pediatric were not read.
- Expected-loss (minimum Bayes risk) training is a known approach, and no algorithmic contribution is claimed.
- Configuration: `configs/experiments/video_segment_risk_v1.yaml` (bc84e5c), fixed before any A25 output.

**Design.**
- **Groups.** Each of the three groups of four windows is one sequence with consecutive centres. Clip-end repeats stay as
  positions.
- **Pads.** One fixed, correctly predicted position sits on each side of a group. It carries the true label 5 samples
  before the first centre or after the last one, in the same sequence, and is absent at a sequence end. Pads carry no
  gradient.
  - Left pads are missing in 11–12% of groups and right pads in 14–16%.
  - Within a group the sample steps are 0, 3, 4 or 5, so a pad is not always exactly the neighbouring request centre.
- **Cost.** For an output pattern of the four windows, the cost is the number of unmatched predicted runs plus unmatched
  true runs. Runs are matched one-to-one within the same class at IoU of at least 0.5, greedily by IoU.
  - An isolated flip inside a true run costs 2.
  - A boundary shifted by one window costs 0.
  - Erasing a true short run costs more than 0.
- **Term.** L_seg is the mean over the three groups of the exact expectation over all 625 patterns under the student's
  per-window softmax. It is differentiable through all four windows.
- **Loss.** CE + KL(own A08 ‖ student) + λ · L_seg, with λ = 1.0 fixed once.
- **Approximations.** Independent per-window draws stand in for B3's smoothing, hysteresis and NMS. Request windows
  (0.5 s) stand in for samples. Groups of four plus two pads stand in for whole sequences.
- **Order.**
  - The trainer and checks were committed in 15fb8e5 before training.
  - The weights, run records and evaluator were committed in 63102aa. Calibration was then started in a separate command.
  - The choices were committed alone in d2b18f4, with the git object's sha verified, before selection or the check pool
    was read.
  - The check pool was evaluated once.

**Checks passed** (`checks.json`).
- **Cost.** The cost equals echo_routing's `segment_f1` on 300 random patterns and on all 625 patterns of four padded
  groups, with no mismatch.
- **Cases.** An interior flip costs 2 against 0 for a boundary shift with the same one wrong window. An A–B–A output
  costs more than 0 under A–A–A truth and 0 under A–B–A truth.
- **Gradient.** Autograd agrees with finite differences within 7e-10.
- **λ = 0 replay.** The new code with λ = 0 reproduces A10 R_s0's logged losses over updates 0–136, including after
  unfreezing, within 1.9e-6, and R_s2's first five updates within 4e-8.
- **Training mechanics.**
  - Teacher parity is 0.0.
  - Resume (3 + 3 against 6) is within bounds.
  - Encoder gradients are 0 before the warm-up and non-zero after.
  - The optimiser is bound.
- **Size of the term.** At three probe updates from the start weights, λ · L_seg's gradient norm is 3.3–11.5, against
  0.6–5.0 for CE + KL (cosine 0.35–0.87). The term is larger at every probe and dominates the update direction.

**Training-set readout** (`training_summary.json`). On the training windows, Seg's mean CE is +0.004/+0.008 above R's.
Its KL to A08 is about three times R's (0.048/0.043 against 0.016/0.014). L_seg falls from 0.34/0.27 in the
frozen-encoder phase to 0.11/0.13 in the last 270 updates.

## Results (seeds 0 and 2)

| System (one forward per window) | Development mean | s0 sel, s0 cal, s2 sel, s2 cal | Check s0 / s2 (mean) |
|---|---|---|---|
| Base (A08 + B3) | 0.9698 | 0.9563, 0.9827, 0.9564, 0.9837 | 0.9603 / 0.9555 (0.9579) |
| Long10 R@A08 (control, λ = 0) | 0.9695 | 0.9559, 0.9827, 0.9567, 0.9828 | 0.9606 / 0.9578 (0.9592) |
| **Seg@A08** | 0.9518 | 0.9241, 0.9838, 0.9212, 0.9780 | 0.9572 / 0.9569 (0.9571) |
| Seg own-cal (sensitivity) | 0.9517 | 0.9096, 0.9828, 0.9363, 0.9780 | 0.9523 / 0.9599 (0.9561) |
| A14 P | 0.9724 | 0.9640, 0.9855, 0.9564, 0.9837 | not evaluated |
| A20 H@A08 | 0.9724 | 0.9577, 0.9866, 0.9644, 0.9808 | not evaluated |

**Fixed third-seed rule: not met**, so A25 closes with two seeds.

| Part | Seg@A08 | Required |
|---|---|---|
| Mean dF against Base | −0.0180 | ≥ +0.003 |
| Half means (selection / calibration) | −0.0338 / −0.0023 | each ≥ −0.005 |
| Lowest cell | −0.0353 (s2 selection) | ≥ −0.015 |
| Native macro-F1 difference | +0.0018 | ≥ −0.003 |
| Boundary recall difference | +0.0065 | ≥ −0.010 |
| Missed short true segments | 15 against Base's 22 | ≤ Base + 3 |
| Calibration row feasible | both seeds | each seed |

### 1. What does the risk term add?

| Comparison (multi segment F1@50) | s0 sel | s0 cal | s2 sel | s2 cal | Check s0 / s2 |
|---|---|---|---|---|---|
| Seg − R | −0.0318 [−0.0526, −0.0139] | +0.0010 | −0.0355 [−0.0570, −0.0167] | −0.0048 [−0.0107, 0.0000] | −0.0034 / −0.0009 |
| R − Base | −0.0004 | 0.0000 | +0.0003 | −0.0009 | +0.0003 / +0.0023 |
| Seg argmax − R argmax | −0.0002 | +0.0085 | −0.0101 | −0.0010 | not evaluated |
| Seg own-cal − R + B3 | −0.0470 [−0.0708, −0.0252] | −0.0018 | −0.0242 [−0.0441, −0.0069] | −0.0048 | – |

- On selection the risk term lowers full segmentation by 0.032–0.036 in both seeds, and both intervals exclude zero. On
  calibration and on the check pool the differences are small (−0.0048 to +0.0010).
- The common continuation (R − Base) is level, so the loss comes from the term.
- **Where the loss is.** In seed 0 selection, against R, Seg adds 31 unmatched predicted segments (23 short; 15 A4C and
  16 A5C, in 17 recordings of 12 sessions). It removes 4 and loses 7 matched reference segments (6 A5C).
- **Window decisions barely move.** Window accuracy stays near R's level (selection 0.9661/0.9674 against
  0.9665/0.9688), and the plain argmax rows differ little (−0.010 to +0.009). The loss appears when B3, at A08's numeric setting, reads
  Seg's probabilities. Seg's own calibration choice does not recover it on selection.

### 2. Does it remove false short segments and keep true short ones?

| Count (short in brackets) | Base | R | Seg | Seg own-cal |
|---|---|---|---|---|
| Development false segments | 72 (39) | 84 (53) | 147 (107) | 143 (99) |
| Development missed true segments | 51 (22) | 41 (15) | 54 (15) | 58 (17) |
| Check false segments | 248 (153) | 263 (180) | 306 (227) | 304 (221) |
| Check missed true segments | 187 (79) | 160 (62) | 142 (50) | 154 (60) |
| Development same-class join cuts (of 862) | 19 | 21 | 24 | 26 |

| Window flips over both seeds (A24's fixed definitions) | Base | R | Seg |
|---|---|---|---|
| New pure flips, selection / calibration / check | 0 / 0 / 0 | 15 / 13 / 35 | 28 / 11 / 38 |
| Pure flips, selection / calibration / check | 166 / 59 / 289 | 162 / 65 / 268 | 163 / 55 / 260 |
| Wrong windows within 8 samples of a true change, check | 1,057 | 910 | 954 |
| Wrong windows farther away, check | 580 | 549 | 548 |

- **It does the opposite of the aim.** It roughly doubles false short segments on development against R (107 against
  53) and adds a quarter on the check pool (227 against 180).
- **Short true segments.** On development it keeps R's level of missed short true segments (15). On the check pool it
  misses fewer (50 against 62), with more false segments.
- **Flips.** New pure flips on development rise from 28 to 39, far fewer than the 54 extra false short segments. So most
  extra false segments are not plain argmax flips.
- **Recognition.** Native macro-F1 is unchanged or slightly higher on development (+0.0018). In seed 0 on the check
  pool it is 0.010 below R and 0.005 below Base. The A4C/A5C confusions equal Base's on development except one fewer
  A5C read as A4C in seed 2 calibration; on the check pool seed 0 has one more.

## What this round shows

1. **An exact expected segment-error count on four-window groups did not reduce false short segments.** It roughly
   doubled them under A08's decoder setting on selection. Full segmentation fell by 0.032–0.036 there and stayed near R
   elsewhere.
2. **It moved probabilities more than decisions.** Argmax flips and window accuracy stayed near R's. The term changed the
   probability profile that B3 reads, and Seg's own calibration did not repair this on selection. This round does not
   separate why: candidates are the dominant gradient, sharper or noisier probabilities, and the independence and
   group-of-four approximations.
3. **What is closed.** This configuration is closed: the padded four-window expected F1@50 error count, λ = 1, on A10's
   batches from A08.
   - It does not show that segment-level objectives cannot help.
   - It shows that this local, independence-based surrogate does not match what B3 does with the probabilities.
   - It is not a coefficient search, and none follows from this round.

## Cost and files

- **GPU.** 0.85 of 6 h.
  - Checks: 0.04 h.
  - Training: 0.51 h.
  - Calibration inference: 0.06 h.
  - Selection and check-pool inference: 0.23 h.
- **CPU.** About 0.7 of 4 h at the final commit.
  - This includes the decode/evaluation workers outside GPU time (0.20 h) and the agent's active wall clock booked by
    hand.
  - Waiting is not booked, and GPU loaders are not counted again.
- **Records.**
  - `checks.json` and `preparation.json` (15fb8e5).
  - `training_summary.json` and `train/` (63102aa).
  - `choices.json` (d2b18f4).
  - `screen.json`: rows, counts, comparisons, accounting, window groups, flips and cost.
- **Frozen weights.** `models_frozen/video_segment_risk_v1/Seg_s{0,2}` on the server (final.pt sha aa4a8f82d92b and
  cbc6b9e379a8). A10's frozen R arms were reused, not retrained.
- **Server only.** Group and pad tables, frames, logits, per-sequence predictions and all identifiers.
