# A19: does choosing the decoder point on more training-source data help? (2026-10-21)

**Question.** A09-R1, A10, A11 and A16–A18 suggested that working points chosen on the reused calibration half may not
transfer. With the same nine frozen video models and the same 40 B3 settings, does choosing each
model's setting on A15's large training-source pool, instead of its small pool, give a full-segmentation gain that
transfers?
- No video encoder or new network was trained; only existing B3 parameters were chosen.
- The large pool is the pre-specified primary, the small pool is the matched coverage control, and the old committed
  choices and each seed's A08 setting are references outside the new choice.
- The router-check sessions are unseen by the choice. But their recordings trained the video models and were used in
  earlier exploration, so they are not a new holdout and not patient-independent. The development halves are reused
  exploratory feedback.
- The test split and EchoNet-Pediatric were not read.
- Configuration: `configs/experiments/video_b3_coverage_v1.yaml` (e60602b), fixed before any A19 output.

**Design.**
- **Models.** Nine frozen ordinary TargetMViT models: A08 all_s0/s1/s2, R_s0/R_s2 (A10), and R_s1 and E_s0/s1/s2 (A18).
  All are pinned by sha.
- **Pools.** A15's pinned pool, rebuilt exactly. Only whole sequences with their reference labels are used; no A15 zone,
  router label or utility.
  - Large fit: 1,500 multi streams and the actual 887 native clips.
  - Small fit: the fixed 300 and 200 inside it.
  - Check: 600 and 200 from the 100 sessions that share no session or recording with fit.
- **Inference.**
  - A15's verified A08 s0/s2 window logits were reused. Fresh online logits on 32 random windows matched them exactly
    (difference 0.0).
  - The other seven models had one frozen forward over all 67,829 pool windows (5-class logits only).
- **Choice.** report_center_supervision's b3_choose procedure over the A08 grid of 40 settings: half window 5–15,
  quantile 0.80–0.999, NMS radius 5.
  - Numeric thresholds come from each pool's own native scores.
  - Feasibility is native macro-F1 ≥ 0.980 and multi boundary recall ≥ 0.95. The highest multi segment F1@50 wins, and
    ties go to the lower order.
  - The pool has no pair streams, so its pair tables are empty.
  - An infeasible pool gets no fallback.
- **Order.** The 18 choices (9 models × 2 pools) were committed (e575edb; the git object's sha was verified) before the
  check pool or either development half was evaluated, once.
  - The first evaluation attempt stopped on a missing input field before any row or comparison was written. The same
    fixed evaluation was then rerun, and the failed time is booked.
- **Checks passed.**
  - The pool rebuild; small inside large; fit and check disjoint; no development clip.
  - Cache parity, and model states unchanged.
  - The selector reproduces all nine old committed B3 choices from their old calibration inputs. This only re-derives the
    old choices.
  - Thresholds are never recomputed on check or development data.
  - All old development rows of A08, R, E and A14 P reproduced.

**Choices.**
- **Large pool.** Feasible for every model, with 28–35 of 40 settings feasible. 7 of the 9 models chose half window 7
  and quantile 0.995.
- **Small pool.**
  - R_s1 is infeasible (0 of 40). A08_s1 has only 6 feasible settings and E_s1 21.
  - E_s0 chose quantile 0.97, a much lower threshold, which cuts far more often: 0.9160 on seed-0 selection.
- **Old choices, for reference.** The old development-calibration choices mostly used quantile 0.999.

## Results

Mean multi segment F1@50: development over six cells (3 seeds × 2 halves); check over three seeds (one pool each).

| Model | Large (primary) | Small | Old committed | A08 setting |
|---|---|---|---|---|
| A08, development | 0.9693 | 0.9687 | 0.9704 (= Base) | = old |
| A08, check | 0.9575 | 0.9564 | 0.9574 | = old |
| R, development | 0.9704 | seeds 0, 2 only (R_s1 infeasible) | 0.9713 | 0.9678 |
| R, check | 0.9609 | seeds 0, 2 only | 0.9598 | 0.9587 |
| E, development | **0.9730** | 0.9669 | 0.9718 | 0.9712 |
| E, check | 0.9600 | 0.9591 | 0.9585 | 0.9598 |

**Large − small, same weights** (paired intervals over constructed streams, exploratory).

| Model | Check s0 / s1 / s2 | Selection s0 / s1 / s2 | Calibration s0 / s1 / s2 |
|---|---|---|---|
| A08 | +0.0034 [0.0004, 0.0069] / −0.0001 / 0.0000 | +0.0038 / −0.0006 / −0.0046 | +0.0076 [0.0019, 0.0153] / −0.0009 / −0.0019 |
| E | +0.0055 [0.0013, 0.0104] / +0.0003 / −0.0030 [−0.0058, −0.0009] | +0.0414 [0.0219, 0.0641] / −0.0029 / −0.0067 [−0.0130, −0.0019] | +0.0056 / −0.0009 / 0.0 |
| R (s0, s2) | +0.0012 / – / −0.0011 | −0.0002 / – / −0.0057 [−0.0106, −0.0018] | +0.0009 / – / +0.0001 |

- **No consistent difference.** For A08 and E over three seeds, there is no consistent difference in any metric on the
  check pool or either half.
- **Where large helps and where it hurts.**
  - The large pool avoids the small pool's occasional extreme or infeasible choice: E_s0's low threshold and R_s1.
  - Elsewhere, especially seed 2, it is slightly worse.
- **The direction never holds.** Large − small never has the same direction across seeds on both the check pool and
  selection. So no gain is attributed to more selection data.

**Under the same large rule.**

| Comparison | Check s0 / s1 / s2 | Development cells (s0 sel, s0 cal, s1 sel, s1 cal, s2 sel, s2 cal) | Development mean |
|---|---|---|---|
| E − R | −0.0021 / −0.0001 / −0.0006 | −0.0001, +0.0020, +0.0106 [0.0017, 0.0224], 0.0, +0.0020, +0.0009 | +0.0026 |
| E − A08 | −0.0012 / +0.0038 / +0.0052 [0.0008, 0.0097] | +0.0002, +0.0029, +0.0078, −0.0009, +0.0104 [0.0005, 0.0232], +0.0019 | +0.0037 |
| E − Base (A08 old) | −0.0005 / +0.0038 / +0.0045 | +0.0011, +0.0029, +0.0078, −0.0009, +0.0051, 0.0 | +0.0027 |

- **E − R.** No consistent difference. On the check pool E is slightly below R in all three seeds. A18's sensitivity
  readout (+0.0034 at A08's setting) does not hold up as a model difference under one shared, larger-data rule.
- **E − Base.**
  - Boundary recall is consistently higher on the check pool. Segment F1 shows no consistent difference.
  - E-large misses 63 true segments over the six cells (26 short), against Base's 74 (32). It makes 69 false selection
    segments (33 short), against 75 (33).
  - Native macro-F1 is 0.9833 against 0.9848, with A4C/A5C errors at 9/2 against 8/2.
- **Candidate criterion: not met.**
  - Mean +0.0027 against the required +0.003, not rounded.
  - Its other parts pass: half means +0.0047 and +0.0007, lowest cell −0.0009, native −0.0015, boundary recall +0.0035,
    26 missed short true segments against 32, and every seed feasible on calibration.
  - The criterion would not by itself replace the E − R evidence.
- **E-large against A14 P** (seeds 0 and 2, the four cells both have): −0.0065, +0.0001, +0.0051 and 0.0, a mean of
  −0.0003. So no single pipeline is shown to beat A14 P.

## What this round shows

1. **More training-source data for the decoder choice gave no consistent transferable gain.** Large − small has no
   consistent sign on the check pool or the development halves. The large pool protects against a bad small-pool choice,
   but it does not lift segmentation overall.
2. **The E − R difference is within the working-point noise.** Under one shared large-pool rule, E is +0.0026 on
   development but slightly negative on the check pool in every seed. The decoder point moves results by about as much
   as the model change did (±0.002–0.004).
3. **No new strongest pipeline.** E-large keeps more true and short segments and has higher boundary recall than A08 +
   B3. But it is +0.0027, under the line, slightly lower in native recognition, and not above A14 P on the seeds both
   have.
4. **This configuration closes.** Choosing the decoder point on wider training-source data is closed. This does not
   show that calibration is useless.
   - Over A14–A19, routing, teacher targets and decoder points each moved results by about 0.002–0.004 without a
     consistent gain.
   - The remaining errors are ones the decoder cannot fix from these outputs: segments that every model misses (in A18,
     15 of 502 selection segments were missed by all three A08 models and kept missed by their average), short
     segments, and A4C/A5C confusion.
   - The next training motive is therefore to change what the video model predicts on those windows, not how its
     outputs are decoded or routed.

## Cost and files

- **GPU.** 2.26 of 4 h: the probe 0.12 h, frozen pool inference 2.14 h, and the parity check under 0.01 h.
- **CPU.** 1.96 h of 3 at the final commit.
  - This includes the decode workers outside GPU time, the 18 choices (0.61 h), the evaluation and its failed first
    attempt, and the agent's active wall clock.
  - A dated correction line removed 0.58 h of waiting that had been booked as active time while the choice job ran.
  - Loader workers inside GPU time are not counted again (A18's rule).
- **Records.**
  - `probe.json`, `checks.json` and `logits_index.json`.
  - `choices.json`, committed before evaluation.
  - `screen.json`: rows, counts, comparisons, accounting, verdicts and the candidate criterion.
- **Server only.** The pool logits, per-sequence predictions and all identifiers.
