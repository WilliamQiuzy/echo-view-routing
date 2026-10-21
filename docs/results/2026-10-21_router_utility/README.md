# A16: does weighting zones by segment gains and losses help the local router? (2026-10-21)

**Question.** A15's binary label marks a zone positive when replacing it alone raises its sequence's segment F1. So
zero-utility and harmful zones share the negative class, and the sizes of gains and losses are ignored. Does training
the same router on a count-based utility, weighted by size, give a selection whose gains survive in full segmentation?
- The data are A15's pinned large-fit candidates, and the router is the same 1,729-parameter P router. The only trained
  difference between the two arms is the label and the per-zone loss weight.
- The development halves were reused for exploratory feedback.
- The router-check set is unseen by the routers but not new. A08 and A13 had seen its recordings, and A15's check
  summaries had been read.
- The test split and EchoNet-Pediatric were not read.
- Configuration: `configs/experiments/video_router_utility_v1.yaml` (466831b), fixed before any A16 output.

**Design.**
- **Fixed from A15.** Every input is pinned by sha: the pool manifest, the candidate order, the 52 P features, the cached
  A08 window outputs and the Base and Alternative predictions. Base is A08 A + B3 and Alternative is A13 H. Zones, the
  replacement action and the features are unchanged.
- **Utility.**
  - For each zone, dTP and dFP are the changes in matched and false segments when that zone alone is replaced, on the
    formal segment F1@50 matching.
  - F_fit is Base's pooled segment F1@50 over all 1,500 large-fit multi streams: 0.9582 in seed 0 and 0.9580 in seed 2.
  - The utility is u = 2 dTP − F_fit (dTP + dFP). With a set's own Base F, its sign is the sign of that replacement's
    F1 change. u is zero exactly when a zone changes neither count.
  - Inference never sees a reference. So u is only a training proxy, and every reported metric re-matches whole
    sequences.
- **Two arms.**
  - C (control) is A15 large-P: the binary label with balanced BCE. Retrained, it equals A15's frozen large-P bit for bit
    in weights, normalisation and scores.
  - U uses the label u > 0 with weight |u| divided by the mean |u| over the fit zones. Zero-utility zones get weight 0
    and so no gradient.
  - Both share the initialisation, normalisation, batches and schedule, and keep their final weights. Each trains on CPU
    in a few seconds.
- **Two rules per arm.**
  - **fixed** accepts a zone when its score is above 0.5. It was fixed in the configuration before any output.
  - **cal** uses A15's calibration grid and feasibility rule. C-cal reproduces A15 large-P exactly, in choices, rows
    and predictions, so the two are one row below.
- **Order.** The router records were committed (859811a) before calibration. The rules and choices were committed
  (f0dbb34) before any selection output was read.
- **Checks passed.**
  - The pins.
  - An independent recount of 122 zones per seed with the evaluator-checked matching, with 0 mismatches. A15's counting
    equals that matching on all 3,187 sequences.
  - Synthetic tests for removing a false segment, losing a true one and recovering a short one, each with the exact F1
    identity.
  - Weights from fit zones only, zero weight giving zero gradient, and costs moving scores the right way.
  - Identical starts, router-only optimisation, C's code path reproduced by the weighted function, an exact resume, and
    no-op and accept-all.

**Training labels.**

| | Seed 0 | Seed 2 |
|---|---|---|
| Large-fit zones | 429 | 483 |
| u > 0 / u = 0 / u < 0 | 94 / 290 / 45 | 82 / 324 / 77 |
| Weight mass on u > 0 / u < 0 | 284 / 145 | 222 / 261 |

- On multi streams, every zone's u sign equals its old label (positive, zero or harmful), in fit and check alike. So for
  multi zones U differs from C only in the weights, and in giving the zero-utility zones no gradient at all.
- The two labels differ only on native clips. There, a clip with no matched segment has F1 0 whatever changes: 6
  old-zero native zones become u > 0 in seed 0, and 8 become u < 0 in seed 2.

**Router check.** These sessions are unseen by the routers. They were logged only, and nothing was chosen from them.
Utility signs use F_fit. "Check F1" is the pooled segment F1@50 over the 600 check streams; Base scores 0.9603 and 0.9555.

| Router at the fixed rule | Accepted | u > 0 / 0 / < 0 | Accepted utility (+ / −) | Check F1 |
|---|---|---|---|---|
| Seed 0 C | 34 | 20 / 4 / 10 | 30.1 / −21.5 | 0.9616 |
| Seed 0 U | 138 | 20 / 108 / 10 | 35.3 / −19.4 | 0.9630 |
| Seed 2 C | 37 | 21 / 3 / 13 | 31.1 / −20.6 | 0.9579 |
| Seed 2 U | 111 | 19 / 83 / 9 | 29.0 / −12.9 | 0.9586 |

At the cal choices the results differ by seed.
- In seed 0, C (threshold 0.9) accepts 14 zones for 0.9604, and U (threshold 0.7) accepts 120 for 0.9627.
- In seed 2 both chose no-op.

## Results (seeds 0 and 2; four cells)

| System | Mean F | − Base | − Alternative | − A14 P | Selection FP (short) | FN, four cells (short) | Boundary recall / precision |
|---|---|---|---|---|---|---|---|
| Base (A08 A + B3) | 0.9698 | – | −0.0015 | −0.0026 | 49 (22) | 51 (22) | 0.979 / 0.966 |
| Alternative (A13 H) | 0.9712 | +0.0015 | – | −0.0012 | 44 (18) | 54 (25) | 0.978 / 0.973 |
| A14 P | 0.9724 | +0.0026 | +0.0012 | – | 41 (16) | 52 (23) | 0.979 / 0.974 |
| C-cal (= A15 large-P) | 0.9705 | +0.0007 | −0.0008 | −0.0019 | 47 (19) | 53 (24) | 0.978 / 0.969 |
| C-fixed | 0.9715 | +0.0017 | +0.0002 | −0.0009 | 44 (13) | 52 (24) | 0.979 / 0.971 |
| U-cal | 0.9710 | +0.0012 | −0.0003 | −0.0014 | 46 (19) | 52 (24) | 0.979 / 0.970 |
| U-fixed | 0.9717 | +0.0019 | +0.0005 | −0.0007 | 43 (13) | 52 (26) | 0.979 / 0.972 |

| Cell | Base | Alternative | A14 P | C-fixed | C-cal | U-fixed | U-cal |
|---|---|---|---|---|---|---|---|
| Seed 0 selection | 0.9563 | 0.9639 | 0.9640 | 0.9591 | 0.9572 | 0.9602 | 0.9592 |
| Seed 0 calibration | 0.9827 | 0.9826 | 0.9855 | 0.9836 | 0.9846 | 0.9836 | 0.9846 |
| Seed 2 selection | 0.9564 | 0.9558 | 0.9564 | 0.9605 | 0.9564 | 0.9604 | 0.9564 |
| Seed 2 calibration | 0.9837 | 0.9827 | 0.9837 | 0.9827 | 0.9837 | 0.9827 | 0.9837 |
| Accepted zones (four cells) | – | all | 14 / 6 / 0 / 0 | 9 / 7 / 13 / 6 | 3 / 5 / 0 / 0 | 36 / 43 / 33 / 27 | 26 / 38 / 0 / 0 |

- Base, Alternative, A14 P and A15 large-P reproduce their stored rows exactly.
- Native macro-F1 and both A4C/A5C error counts equal Base's for every router system in every cell.
- Both fixed rules meet the calibration feasibility rule in both seeds.

**Comparisons** (paired bootstrap intervals; the four cells in the order of the table).
- **U-fixed − C-fixed, the primary comparison.** +0.0010 [−0.0029, 0.0061], 0, −0.0001 [−0.0042, 0.0039], 0. The mean
  is +0.0002.
- **U-cal − C-cal.** +0.0020 [0.0, 0.0063] on seed-0 selection, and 0 in the other cells.
- **Same weights, fixed − cal.**
  - On selection, C gains +0.0019 [0.0, 0.0049] and +0.0040 [−0.0009, 0.0097], and U gains +0.0010 [0.0, 0.0029] and
    +0.0040 [−0.0019, 0.0096].
  - On calibration both lose 0.001 in both seeds.
- **U-fixed − Base.** On selection +0.0038 [−0.0011, 0.0098] and +0.0040 [−0.0019, 0.0096]. On calibration +0.0009
  and −0.0010.
- **U-fixed − A14 P.** −0.0038 and −0.0019 in seed 0, then +0.0040 and −0.0010 in seed 2.

**Fixed promotion rule: not met.** A16 closes with two seeds, and no seed 1 was built.
- **U-fixed.** Mean +0.0019 against the required +0.003. It also misses 26 short true segments, against Base's 22 and
  an allowance of 25. Its no-reversal and calibration-feasibility parts pass.
- **U-cal.** Mean +0.0012; its other parts pass.
- **C-fixed** is +0.0017. It is a control and cannot qualify.

## Reference-assisted diagnostic (after selection; not a system)

Each development zone was scored by its single-replacement counts and u with the seed's F_fit. Only multi streams enter
the primary metric.

| Cell | Multi zones, u > 0 / 0 / < 0 | C-fixed accepts u > 0 / < 0 | U-fixed accepts u > 0 / < 0 (zero) | U-cal accepts | A14 P accepts |
|---|---|---|---|---|---|
| Seed 0 selection | 9 / 22 / 4 | 4 / 1 | 4 / 1 (21) | 3 / 1 (17) | 8 / 1 |
| Seed 0 calibration | 3 / 37 / 4 | 3 / 2 | 3 / 2 (34) | 3 / 1 (31) | 3 / 1 |
| Seed 2 selection | 8 / 23 / 6 | 6 / 2 | 6 / 2 (16) | no-op | no-op |
| Seed 2 calibration | 1 / 22 / 2 | 1 / 2 | 1 / 2 (17) | no-op | no-op |

- **The same harmful zones get through.** In every cell, U-fixed accepts the same numbers of positive- and
  negative-utility multi zones as C-fixed. The accepted negative utility is −2.08, −1.04, −2.0 and −2.0 for U, against
  −2.08, −1.04, −1.92 and −2.0 for C.
- **What U adds are zero-utility zones.** It accepts 16–34 per cell, which singly change no segment count.
- **Net multi counts against Base, re-matched on whole sequences.**
  - U-fixed: dTP 0 and dFP −4 in seed-0 selection, +2 and −2 in seed-2 selection.
  - C-fixed: −1 and −4, then +3 and −1.
  - Both lose 1–2 matches on the calibration halves.
- **Natives and pairs are kept separate.** In seed-2 selection both fixed rules accept 2 negative-utility native zones,
  which add 4–5 false native segments. Native macro-F1 is unchanged.

## What this round shows

1. **Weighting by gain and loss size did not keep harmful edits out of the development halves.**
   - On multi streams the utility has the old label's sign for every zone, so U changes only the weights.
   - On the router-check pool, U accepted less negative utility (−19.4 against −21.5, and −12.9 against −20.6), and its
     check F1 was 0.0014 and 0.0007 higher.
   - On development, U-fixed let through as many harmful multi zones as C-fixed, and U-fixed − C-fixed is about zero.
2. **Zero-utility zones are left unconstrained.** Nothing in U's loss pushes their scores down, so U accepts many of
   them: 27–43 calibration zones against C-fixed's 6–7. They change no count singly. U-fixed's 26 missed short true
   segments (C-fixed: 24) come from seed-2 selection, where it loses 2 short matched segments that C-fixed keeps; which
   accepted zones cause this was not separated.
3. **The fixed point held up better on selection than the calibrated choice, but it is not established.**
   - On the same weights, fixed beats cal on both selection halves (+0.0010 to +0.0040). The cal rule's no-op in seed 2
     gave up a gain that the fixed point kept on selection.
   - But fixed is 0.001 below cal on both calibration halves, and 0.001 below Base on seed-2 calibration. It stays under
     the promotion line, and every interval includes zero.
   - This does not show that 0.5 transfers in general, or that selection can be dropped.
4. **No new system beats both sources and A14 P.**
   - U-fixed is above Base (+0.0019) and Alternative (+0.0005) but below A14 P (−0.0007), and U-cal is below Alternative.
   - In seed-2 selection U-fixed is above all three, by +0.004 to +0.0046 with intervals that include zero. In seed-0
     selection it stays below Alternative and A14 P (−0.0037 and −0.0038).
   - A14 P remains the highest point estimate, and A08 A + B3 the working reference. A paper-level algorithmic
     contribution is not established.

## Cost and files

- **GPU.** 0 of 1 h. Seeds 0 and 2 reused A15's and A13's caches, and no seed 1 was approved.
- **CPU.** 0.36 h of 3 at the final commit, in `runs/video_router_utility_v1/ledger.jsonl`, including the agent's
  active wall clock.
- **Records.**
  - `preparation.json` (F_fit, utilities, cross-counts and the recount) and `checks.json`.
  - `routers/{C,U}_s{0,2}.json` (weights sha and check logging, committed before calibration).
  - `choices.json` (rules, fixed calibration rows and cal choices, committed before selection) and `check_log.json`.
  - `screen.json` (rows, counts, zones, comparisons, accounting and the promotion decision) and `diagnostic.json`.
- **Server only.** The router weights, utility arrays and all per-sequence outputs.
- **A slip in this round.** One mistaken helper call appended an empty line (0 h) to A15's ledger. It was removed at
  once, restoring the file to its earlier 16 lines and totals (GPU 0.645 h, CPU 0.962 h).
