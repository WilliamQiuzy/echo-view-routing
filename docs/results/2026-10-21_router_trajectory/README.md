# A17: does the local prediction trajectory help the router beyond summary features? (2026-10-21)

**Question.** A15's P router reads 52 summary features of a zone and its neighbours. It cannot see the full sequence of
predictions inside the zone, so it may not tell a brief flicker from a sustained change. With A08 + B3 kept as the
default and only local replacements learned, does the within-region time order of a candidate's prediction trajectory
add decision evidence?
- The two arms have the same capacity. They differ only in that order: true time order (O) against a fixed shuffle
  inside each region (S).
- The development halves were reused for exploratory feedback.
- The router-check set is unseen by the routers, but it has been read repeatedly and A08/A13 saw its recordings.
- The test split and EchoNet-Pediatric were not read.
- Configuration: `configs/experiments/video_router_trajectory_v1.yaml` (cc3115e), fixed before any A17 output.

**History.** Temporal models (R01/A01), probability statistics and local readouts (R05/R07), sustained-switch heads
(A11/A13) and pooled visual features (A14) were tried before. The new condition is candidate-conditioned trajectories
over the fixed A08 + B3 / A13 H edit zones, with A15's segment-utility label. Convolutions and routing are known methods,
and no contribution is claimed.

**Design.**
- **Fixed sources.** A15's pinned pool, masks, candidate order, cached A08 window outputs and the Base and Alternative
  labels are all pinned by sha. Base is A08 A + B3 and Alternative is A13 H, both frozen. Zones and the replacement
  action are unchanged.
- **Labels.** A15's binary label returns: a zone is positive when replacing it alone strictly raises its sequence's
  segment F1@50. Zero and harmful zones are ordinary negatives, under balanced BCE with large-P's fit counts. No A16
  utility weights are used.
- **Trajectory input.** For each zone, the trajectory covers the whole zone plus up to 16 samples on each side, clamped
  to the sequence. Each 10 Hz sample has 18 channels:
  - A08's 5 class probabilities. These are the stride-5 window softmax mapped to samples by nearest centre, so
    neighbouring samples can repeat one window.
  - Base and Alternative one-hots (5 each).
  - A region one-hot (left, zone, right).
- **No reference information.** No reference, join, true length or identifier is an input. Changing the references
  leaves tensors, zones, features and scores unchanged.
- **S, the control.** Rows are permuted once inside each region with a key built only from positions. Whole rows move,
  regions never mix, and the 52 features are not recomputed. So S keeps those features' coarse timing statistics and
  the left/zone/right context. O − S therefore tests only the fine-grained within-region order.
- **Model, identical for both arms.**
  - Two width-16 convolutions (kernel 5) with padding zeroed after each layer.
  - Masked means over left, zone and right (48 values), joined with the normalised 52 features.
  - Then Linear(100, 16), GELU and Linear(16, 1): 4,385 parameters.
  - Same initialisation, normalisation (equal to A15 large-P's), batches and schedule. 1,350 updates on one CPU thread,
    about 10–12 s per router, final weights.
- **Two rules per arm.**
  - **fixed** accepts a zone when its score is above 0.5. It is the primary rule.
  - **cal** uses A15's calibration grid and feasibility rule.
- **Order.** Router records were committed (39338de) before calibration, and choices (f700ea4) before any selection
  output was read.
- **Checks passed.**
  - Probabilities rebuilt from A15's cache reproduce A15's Base, Alternative, zones and 52 features bit for bit.
  - O and S share region members and channel sums; 1,346 regions were actually reordered.
  - Reference isolation, and padding, empty-neighbour and long-zone invariance (difference 0.0).
  - 4,385 parameters, identical starts and batches, a gradient on zero-label zones, router-only optimisation and an
    exact resume.
  - No-op equals Base and accept-all equals Alternative.

**Training data.** 596 (seed 0) and 665 (seed 2) zones. Trajectories have 22–98 samples (median 37), and zones have 2–66
(median 5).

**Router check.** These sessions are unseen by the routers. They were logged only, and nothing was chosen from them.
"Check F1" is the pooled segment F1@50 over the 600 check streams.

| Router at the fixed rule | Accepted (beneficial / zero / harmful) | Check F1 | Short FP / FN |
|---|---|---|---|
| Seed 0 O | 36 (20 / 6 / 10) | 0.9622 | 48 / 47 |
| Seed 0 S | 32 (20 / 3 / 9) | 0.9628 | 48 / 46 |
| Seed 2 O | 42 (20 / 8 / 14) | 0.9575 | 80 / 35 |
| Seed 2 S | 40 (19 / 6 / 15) | 0.9575 | 82 / 37 |

On the check streams, Base scores 0.9603 and 0.9555 and A16 C-fixed 0.9616 and 0.9579.

## Results (seeds 0 and 2; four cells)

| System | Mean F | − Base | − Alternative | − A14 P | Selection FP (short) | FN, four cells (short) | Boundary recall / precision |
|---|---|---|---|---|---|---|---|
| Base (A08 A + B3) | 0.9698 | – | −0.0015 | −0.0026 | 49 (22) | 51 (22) | 0.979 / 0.966 |
| Alternative (A13 H) | 0.9712 | +0.0015 | – | −0.0012 | 44 (18) | 54 (25) | 0.978 / 0.973 |
| A14 P | 0.9724 | +0.0026 | +0.0012 | – | 41 (16) | 52 (23) | 0.979 / 0.974 |
| A15 large-P (= A16 C-cal) | 0.9705 | +0.0007 | −0.0008 | −0.0019 | 47 (19) | 53 (24) | 0.978 / 0.969 |
| A16 C-fixed | 0.9715 | +0.0017 | +0.0002 | −0.0009 | 44 (13) | 52 (24) | 0.979 / 0.971 |
| A16 U-fixed | 0.9717 | +0.0019 | +0.0005 | −0.0007 | 43 (13) | 52 (26) | 0.979 / 0.972 |
| **O-fixed** | 0.9722 | +0.0024 | +0.0010 | −0.0002 | 44 (19) | 49 (22) | 0.980 / 0.971 |
| S-fixed | 0.9722 | +0.0024 | +0.0010 | −0.0002 | 44 (19) | 49 (22) | 0.980 / 0.971 |
| O-cal | 0.9702 | +0.0004 | −0.0010 | −0.0022 | 47 (19) | 53 (24) | 0.978 / 0.969 |
| S-cal | 0.9712 | +0.0014 | 0.0000 | −0.0012 | 44 (18) | 53 (25) | 0.978 / 0.971 |

| Cell | Base | Alternative | A14 P | A16 C-fixed | O-fixed = S-fixed | O-cal | S-cal |
|---|---|---|---|---|---|---|---|
| Seed 0 selection | 0.9563 | 0.9639 | 0.9640 | 0.9591 | 0.9611 | 0.9572 | 0.9611 |
| Seed 0 calibration | 0.9827 | 0.9826 | 0.9855 | 0.9836 | 0.9836 | 0.9836 | 0.9836 |
| Seed 2 selection | 0.9564 | 0.9558 | 0.9564 | 0.9605 | 0.9606 | 0.9564 | 0.9564 |
| Seed 2 calibration | 0.9837 | 0.9827 | 0.9837 | 0.9827 | 0.9837 | 0.9837 | 0.9837 |

- **Accepted zones.** O-fixed accepts 10, 7, 15 and 8 zones in the four cells, and S-fixed 11, 7, 16 and 10. O-cal
  chose 0.9 and no-op, and S-cal 0.5 and no-op.
- **Reproduction.** Base, Alternative, A14 P, A15 large-P, A16 C-fixed and A16 U-fixed reproduce their stored rows
  exactly.
- **Native recognition.**
  - O-fixed's native macro-F1 and A4C/A5C counts equal Base's in every cell.
  - S-fixed additionally accepts a native zone in seed-2 calibration. That zone corrects one A5C read as A4C (native
    macro-F1 0.9929 against 0.9856), so S's mean native difference is +0.0018.

**Comparisons** (paired bootstrap intervals; the four cells in the order of the table).
- **O-fixed − S-fixed, the primary comparison.** 0.0 [0.0, 0.0] in all four cells. The two arms produce identical
  multi-stream segmentations.
- **O-cal − S-cal.** −0.0039 [−0.0093, 0.0] on seed-0 selection, and 0 elsewhere.
- **Same weights, fixed − cal.**
  - O: +0.0039 [0.0, 0.0093], 0.0, +0.0041 [−0.0044, 0.0138] and 0.0.
  - S: equal in seed 0, +0.0041 on seed-2 selection.
- **O-fixed against the references.**
  - Base: +0.0048 [−0.0002, 0.011], +0.0009, +0.0041 [−0.0044, 0.0138] and 0.0.
  - A16 C-fixed: +0.0020 [−0.0019, 0.0071], 0.0, +0.0001 and +0.0010 [0.0, 0.0031].
  - Alternative: −0.0027 [−0.0095, 0.0024], +0.0010, +0.0047 and +0.0010.
  - A14 P: −0.0029 [−0.0077, 0.0011], −0.0019 [−0.005, 0.0001], +0.0041 and 0.0.

**Fixed promotion rule: not met.** A17 closes with two seeds, and no seed 1 was built.
- **O-fixed.** Mean +0.0024 against the required +0.003. Its other parts pass: half means +0.0044 and +0.0004, no cell
  below 0, native difference 0, boundary recall +0.0007, and 22 missed short true segments against Base's 22. Both seeds
  are feasible.
- **O-cal.** Mean +0.0004.
- **S** cannot qualify. It is also +0.0024 (fixed) and +0.0014 (cal).

## Reference-assisted diagnostic (after selection; not a system)

Each development zone was classed by the change in its sequence's segment F1 when it alone is replaced (A15's training
label). Only multi streams enter the primary metric. The last column gives each system's net multi counts against Base,
re-matched on whole sequences.

| Cell | Multi zones, beneficial / zero / harmful | O-fixed (= S-fixed) accepts beneficial / harmful | A16 C-fixed | A14 P | Net dTP / dFP, O-fixed (C-fixed) |
|---|---|---|---|---|---|
| Seed 0 selection | 9 / 22 / 4 | 5 / 1 | 4 / 1 | 8 / 1 | 0 / −5 (−1 / −4) |
| Seed 0 calibration | 3 / 37 / 4 | 3 / 2 | 3 / 2 | 3 / 1 | −2 / −3 (−2 / −3) |
| Seed 2 selection | 8 / 23 / 6 | 7 / 4 | 6 / 2 | no-op | +4 / 0 (+3 / −1) |
| Seed 2 calibration | 1 / 22 / 2 | 1 / 1 (and 2 zero) | 1 / 2 | no-op | 0 / 0 (−1 / 0) |

- **Harmful edits are not fewer.** Over the four cells, O-fixed accepts 8 individually harmful multi zones, against 7 for
  C-fixed. It also accepts 16 beneficial ones, against 14.
- **The net counts are better.** O-fixed ends +2 matches over the four cells against C-fixed's −1. So singly harmful zones
  do not add up to whole-sequence losses in the same way.
- **Ranking.** The AUC of beneficial against other zones is about equal for O and S: 0.90 / 0.90, 0.98 / 0.97,
  0.86 / 0.86 and 0.93 / 0.95.
- **Natives and pairs** are reported separately in `diagnostic.json`.

**A16 supplement (read-only, after A17's selection).** A16 recorded two short true segments in seed-2 selection that
A16 U-fixed missed and A16 C-fixed kept. Undoing each U-accepted zone singly shows two different cases.
- **The first was never damaged by U.** U accepted no zone of that sequence, so U equals Base there. The segment was
  recovered by the one zone that C accepted and U's score kept below 0.5. This is a missed beneficial acceptance.
- **The second is single-zone damage.** The only zone U accepted there (6 samples, not accepted by C) caused it.
  Undoing that zone alone restores the match without losing another.
- Nothing from this supplement fed into A17's rules or features, and no repaired system was scored.

## What this round shows

1. **The ordered trajectory did not beat the same-capacity shuffled input.** O-fixed and S-fixed produce identical
   multi-stream segmentations in every cell, and their rankings are about equal. Within-region prediction order adds
   nothing here. This covers this permutation and readout only.
2. **The common input gain is real but small, and it is not attributable to order.**
   - Both trajectory routers at the fixed point are above A16 C-fixed (+0.0008) and A15 large-P (+0.0018).
   - They keep Base's level of missed short true segments (22, against 24 for C-fixed and 26 for U-fixed), and miss 49
     segments in all against Base's 51.
   - Because S matches O, this gain belongs to the shared input, capacity or training change, not to time order.
   - It does not come from fewer harmful edits: O-fixed accepts slightly more of them, and the whole-sequence counts
     combine better.
3. **No new system beats A14 P.** O-fixed is above both sources (+0.0024 against Base, +0.0010 against Alternative). It
   is 0.0002 below A14 P, below it in both seed-0 cells and above it only on seed-2 selection.
   - A14 P remains the highest point estimate, and A08 A + B3 the working reference.
   - The calibrated rule again lost what the fixed point kept on the selection halves.
4. **This local-trajectory readout closes** under its fixed rule, with no follow-up variant. Local routing as a whole is
   not shown to be useless.

## Cost and files

- **GPU.** 0 of 1 h. Everything ran on CPU from A15's and A13's caches.
- **CPU.** 0.38 h of 3 at the final commit, in `runs/video_router_trajectory_v1/ledger.jsonl`, including the agent's
  active wall clock. The A15 and A16 ledgers were not touched, and the booking helper writes only to the A17 ledger.
- **Records.**
  - `preparation.json` and `checks.json`.
  - `routers/{O,S}_s{0,2}.json` (weights sha, timing and check logging, committed before calibration).
  - `choices.json` (committed before selection) and `check_log.json`.
  - `screen.json` (rows, counts, zones, comparisons, accounting and the promotion decision) and `diagnostic.json`
    (with the A16 supplement).
- **Server only.** Trajectory tensors, permutation keys, router weights and all per-sequence outputs.
