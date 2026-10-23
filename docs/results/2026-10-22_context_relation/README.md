# A24: a teacher-referenced neighbour relation on top of the A23 context interventions (2026-10-22)

**Question.** A23's other-view context interventions made window predictions robust near view changes. But they added
scattered flips inside stable segments, which B3 turned into false short segments. Does a training term on the relation
between neighbouring predictions stop those flips and raise full segmentation, while the local recognition is kept? The
term asks the student's change between consecutive training windows to follow the frozen own A08's change there.
- **Arms.** OtherRel and SameRel are A23's Other and Same recipes exactly, plus the relation term.
- **Zero-relation references.** A23's frozen Other and Same are reused unchanged.
- **What each comparison tests.**
  - OtherRel − SameRel tests the intervention under the term.
  - Each new arm minus its A23 arm tests the term itself.
  - The difference of those two tests their interaction.
  - Every arm is also compared with A08 + B3 (Base), A14 P and A20 H.
- **How this differs from A08.** A08 used MS-TCN's truncated squared log-probability penalty towards zero change, on all
  or label-masked neighbours. It was closed after post-A08 showed that penalty dominated the gradients. A24's target
  change is A08's own change on the same pair, so switches A08 makes cost nothing. It uses no labels, mask, truncation
  or log-probabilities, and only truly consecutive original windows.
- **How this differs from KL.** It is not a restatement of the per-window KL. A constant offset from the teacher costs
  nothing; only changes in the residual between neighbours are penalised.
- **Data limits.** The check pool is not a new holdout, and the development halves are reused exploratory feedback. The
  test split and EchoNet-Pediatric were not read.
- This is a conventional consistency term, and no algorithmic contribution is claimed.
- Configuration: `configs/experiments/video_context_relation_v1.yaml` (a1dddc5), fixed before any A24 output.

**Design.**
- **Relation term.** Rel = mean over 9 pairs of 0.5 · Σ_k |(p_i − p_{i−1})_k − (q_i − q_{i−1})_k|.
  - p is the student's softmax. q is the softmax of the cached own A08 teacher on the same original windows (A18's
    cache), fully detached. Both student endpoints receive gradient.
  - Pairs are consecutive windows inside each of the 12 originals' three groups (native, change-containing constructed,
    same-label join). Each group is one sequence with consecutive centres, giving 3 pairs per group and 9 per update.
  - There are no pairs across groups and none on the copies, whose donors are independent. Clip-end repeats contribute
    0.
- **Loss.** A23's loss (CE + KL on the originals, CE on the copies) plus λ · Rel, with λ = 1.0 fixed once.
- **Everything else is A23's.** Manifests (pinned by sha), the 24-window update, the optimiser, schedule, seeds,
  initialisation, batch order and 1,350 updates are unchanged. The all-protocol-train frame-path table is used.
- **Order.**
  - The weights, run records and evaluator were committed in 412056a at 23:33:41 UTC. Calibration was then started in a
    separate command at 23:33:51.
  - The choices were committed alone in bd42d95, with the git object's sha verified, before selection or the check pool
    was read.
  - The check pool was evaluated once.

**Checks passed** (`checks.json`).
- **Relation cases.** On small examples the term is 0 when p = q, 0 for a constant offset from q, and 0 when the student
  follows a teacher switch. It is positive for a new student jump, and exactly 1 of 9 pairs is penalised when the
  student removes a teacher switch. It is never paired across groups. The torch value equals an independent loop
  restatement.
- **Groups.** Every group is one sequence with consecutive centres in both seeds.
- **λ = 0 replay.** The new code with λ = 0 reproduces A23 Other_s0's logged losses over updates 0–136, including after
  unfreezing, within 6.0e-6.
- **Training mechanics.**
  - Teacher parity is 0.0, and forced OtherRel = SameRel inputs reproduce exactly.
  - Resume (3 + 3 against 6) is within bounds.
  - Encoder gradients are 0 before the warm-up and non-zero after. The optimiser is bound.
- **Size of the term.** At the start the term's gradient norm is 2.2–2.9, against 6.7–17.4 for A23's loss (cosine
  0.16–0.66). It does not dominate the way A08's penalty did. Rel is about 0.02–0.06 per update.
- **Failed launches.** The first check was stopped by hand because one synthetic case had a wrong threshold (0.013 GPU h,
  no result used). A relaunch hit out-of-memory while the old process was still exiting (0.001 GPU h). Both are booked.

**Training-set readout** (`training_summary.json`). OtherRel's Rel stays at about 0.06, against SameRel's 0.024.
OtherRel's original-window CE gap to SameRel is +0.010/+0.003, smaller than A23's Other − Same gap of +0.017/+0.011.

## Results

Multi segment F1@50. Development has four cells (2 seeds × 2 halves); the check pool has one cell per seed.

| System (one forward per window) | Development mean | s0 sel, s0 cal, s2 sel, s2 cal | Check s0 / s2 (mean) |
|---|---|---|---|
| Base (A08 + B3) | 0.9698 | 0.9563, 0.9827, 0.9564, 0.9837 | 0.9603 / 0.9555 (0.9579) |
| A23 Other@A08 (λ = 0) | 0.9545 | 0.9392, 0.9630, 0.9442, 0.9714 | 0.9458 / 0.9487 (0.9473) |
| **OtherRel@A08** | 0.9562 | 0.9390, 0.9658, 0.9448, 0.9751 | 0.9472 / 0.9503 (0.9488) |
| A23 Same@A08 (λ = 0) | 0.9659 | 0.9540, 0.9790, 0.9519, 0.9789 | 0.9603 / 0.9582 (0.9593) |
| **SameRel@A08** | 0.9676 | 0.9567, 0.9790, 0.9557, 0.9789 | 0.9589 / 0.9565 (0.9577) |
| A14 P | 0.9724 | 0.9640, 0.9855, 0.9564, 0.9837 | not evaluated |
| A20 H@A08 | 0.9724 | 0.9577, 0.9866, 0.9644, 0.9808 | not evaluated |

**Promotion rule: not met for either arm.**

| Part | OtherRel@A08 | SameRel@A08 | Required |
|---|---|---|---|
| Mean dF against Base | −0.0136 | −0.0022 | ≥ +0.003 |
| Half means (selection / calibration) | −0.0145 / −0.0127 | −0.0002 / −0.0042 | each ≥ −0.005 |
| Lowest cell | −0.0173 (s0 selection) | −0.0048 (s2 calibration) | ≥ −0.015 |
| Native macro-F1 difference | −0.0011 | −0.0017 | ≥ −0.003 |
| Boundary recall difference | +0.0124 | +0.0052 | ≥ −0.010 |
| Missed short true segments | 12 against Base's 22 | 19 against 22 | ≤ Base + 3 |
| Calibration row feasible | s2 fails (native 0.9788) | both seeds | each seed |

So no seed 1 was trained. A24 closes with two seeds and no coefficient, loss, input, teacher or schedule variant.

### 1. What does the relation term add?

| Comparison | s0 sel | s0 cal | s2 sel | s2 cal | Development mean | Check s0 / s2 |
|---|---|---|---|---|---|---|
| OtherRel − A23 Other | −0.0002 | +0.0029 [0.0000, +0.0083] | +0.0006 | +0.0037 [+0.0009, +0.0077] | +0.0017 | +0.0014 / +0.0016 |
| SameRel − A23 Same | +0.0027 | 0.0 | +0.0038 | 0.0 | +0.0016 | −0.0014 / −0.0017 |

- **With other-view copies, a small consistent-direction gain.** Five of the six cells are positive and one is at
  −0.0002. Only the s2 calibration interval excludes zero.
- **How the gain arises.** The term removes part of the extra pure-segment flips. Over both seeds, new pure flips (pure
  same-label neighbours that flip where A08 does not) fall from 149/118/300 to 128/107/238 (selection, calibration,
  check). Short false segments fall from 113 to 98 on development and from 314 to 290 on check, and boundary precision
  rises from 0.933 to 0.940 on development.
- **Robustness kept.** Most of A23's window robustness survives. On check, OtherRel fixes 342/343 A08-wrong windows
  (A23 Other 359/356) and breaks 134/155 (158/188). Mixed-window accuracy is 0.949/0.955 (0.951/0.956).
- **With same-view copies, no consistent effect.** Development is +0.0016 and check −0.0015, so there is no common
  relation effect. The interaction is +0.0001 on development and +0.0030 on check.

### 2. Does the intervention now help under the relation term?

**No.** OtherRel − SameRel is −0.0177 [−0.0342, −0.0036], −0.0131 [−0.0278, −0.0018], −0.0108 and −0.0038 on
development (mean −0.0114). On check it is −0.0117 [−0.0180, −0.0057] and −0.0062. Every cell is negative, and three
intervals exclude zero.
- Against Base the term recovered about a tenth of A23 Other's loss (−0.0153 to −0.0136). The gap to the matched arm
  is unchanged (−0.0115 to −0.0114), because SameRel moved by the same amount on development.
- OtherRel still makes 98 short false segments on development, against SameRel's 49 and Base's 39. On check it makes
  290, against 178 and 153.

### 3. Does it keep stable segments and true short segments while removing false ones?

| Count (short in brackets) | Base | A23 Other | OtherRel | A23 Same | SameRel |
|---|---|---|---|---|---|
| Development missed true segments | 51 (22) | 45 (10) | 48 (12) | 45 (17) | 46 (19) |
| Development false segments | 72 (39) | 145 (113) | 134 (98) | 95 (59) | 87 (49) |
| Check missed true segments | 187 (79) | 149 (42) | 152 (44) | 156 (61) | 172 (66) |
| Check false segments | 248 (153) | 406 (314) | 386 (290) | 267 (184) | 266 (178) |

| Window flips over both seeds (fixed definitions) | Base | A23 Other | OtherRel | A23 Same | SameRel |
|---|---|---|---|---|---|
| New pure flips, selection / calibration / check | 0 / 0 / 0 | 149 / 118 / 300 | 128 / 107 / 238 | 24 / 16 / 41 | 18 / 11 / 40 |
| Wrong windows within 8 samples of a true change, check | 1,057 | 559 | 579 | 921 | 1,018 |
| Wrong windows farther away, check | 580 | 709 | 662 | 537 | 535 |

- **Errors move.** The other-view arms cut errors near true changes by about half (1,057 to 559–579) but add errors far
  from changes (580 to 662–709). The relation term moves only part of the far errors back.
- **The A08-referenced term barely touches flips that A08 never makes.** It is trained on 9 original pairs per update,
  while the flips appear on held-out windows where the other-view training made the model trust local evidence. A
  term constrained only on soft probabilities over training pairs did not stop the argmax flips at evaluation.
- **SameRel against Base.** It is close to Base (development −0.0022, check −0.0002) and below A14 P and A20 H
  (−0.0049 and −0.0048).

## What this round shows

1. **A teacher-referenced relation term is a small partial repair.** With other-view interventions it moves full
   segmentation by about +0.0017 on development and +0.0015 on check, in a consistent direction, by removing 9–21% of
   the new pure-segment flips. With same-view copies it has no consistent effect.
2. **It does not make the intervention useful.** OtherRel stays clearly below SameRel and Base in every cell. The
   window robustness is still bought with scattered errors away from view changes.
3. **What is closed.** This configuration is closed: the A08-referenced half-L1 relation on 9 original pairs, λ = 1, on
   A23's batches.
   - It does not show that output-structure training cannot work.
   - It shows that this soft, training-pair-only constraint is too weak to cancel the flips the intervention creates.
     The flips are argmax changes on held-out pure windows.
   - It is not a parameter search, and none follows from this round.

## Cost and files

- **GPU.** 2.75 of 6 h.
  - Checks: 0.09 h, including 0.014 h for the two failed launches.
  - Training: 2.07 h.
  - Calibration inference: 0.13 h.
  - Selection and check-pool inference: 0.46 h.
- **CPU.** 0.59 of 4 h at the final commit.
  - This includes the decode/evaluation workers outside GPU time (0.23 h) and the agent's active wall clock booked by
    hand.
  - Waiting is not booked, and GPU loaders are not counted again.
- **Records.**
  - `checks.json` (66fd989).
  - `training_summary.json` and `train/` (412056a).
  - `choices.json` (bd42d95).
  - `screen.json`: rows, counts, comparisons, accounting, window groups, flips and cost.
- **Frozen weights.** `models_frozen/video_context_relation_v1/{OtherRel,SameRel}_s{0,2}` on the server (final.pt sha
  a0bf09795b0d, 0d5f1d609b13, 38a31cb14455 and a3f313bad179). A23's frozen arms were reused, not retrained.
- **Server only.** Manifests (A23's donor mappings), frames, logits, per-sequence predictions and all identifiers.
