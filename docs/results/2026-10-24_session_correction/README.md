# A31: session-held-out interval recognition correction (2026-10-24)

**Question.** Does error-focused interval sampling (E) give more net correction than uniform sampling (U) on source
sessions that took no part in this correction training, while keeping full segmentation?
- **Arms.**
  - F (frozen representation, head only) and T (also block 15, norm and projection) are matched controls.
  - All four arms are copies of the original A08 all_s{0,2}.
  - The original A08 + B3 cut path never changes.
- **Inputs.** Native windows, pooled-probability CE, eval mode, no KL.
- **Data limits.** Training data only: A15's large-fit sources. The original A08 trained on these recordings, so this
  checks the correction branch's transfer to sessions outside this fine-tune. It is not a holdout or a patient-level
  result. No development, check, test or Pediatric data was read.
- **Predecessors.** A20 (hard-position window replay, closed), A27–A29 (recognition branches on the full pool) and A30
  (91341d2: no direct conflicts, small groups fittable). The common learning rates are stronger than A29's (head 1e-3,
  representation 1e-4), so only E − U supports a sampling attribution.
- Configuration: `configs/experiments/video_session_correction_v1.yaml` (d6697c8). Task sheet
  `claude_prompts/2026-10-24-session-heldout-recognition-correction.md`.

**Design.**
- **Session folds.** Two folds of 200 sessions each (rule k = 0 met the coverage floor). All recordings of a session
  share a fold.
- **Pools.** Each fold was built by A15's rules inside its own clips: 600 multi streams and 740 / 747 natives (A5C and
  SC4C below the 200 cap, all eligible clips used). Request windows: 24,585 / 25,774.
- **Isolation verified.** 0 shared sessions, 0 shared recordings and 0 shared complete 16-reference + marker window
  identities (`split.json`).
- **Base and runs per seed.**
  - Base is A08 + B3 at the committed setting.
  - T runs are reference runs. P runs are Base runs with their majority reference label.
  - P error share: 3.2% / 4.3% (seed 0) and 3.1% / 3.9% (seed 2). The errors are mostly A4C/A5C.
- **Batches.**
  - Every update: 16 T plus 16 P. The T part is identical in all four arms.
  - U draws 16 uniform P runs (error share 3.1–4.3%). E draws 8 wrong plus 8 correct from the fit fold's static sets.
  - The U and E P batches differ in every update. F and T share each sampling arm's batches.
- **Training.** 1,350 updates per model, AdamW (head 1e-3, representation 1e-4), warm-up 135 then cosine, clip 1, final
  weights only.
  - The schedule was not shortened: the check projected 3.15 / 8 GPU h.
  - T trained through one verified frozen-prefix cache per seed (about 61 GB, shared across folds and arms, deleted after
    use).
  - 16 models in total.
- **Order.** Protocol d6697c8, trainer and split 5a64d06, checks and schedule 7222753. The seed-0 weights were committed
  in 5979f41 before seed 0's held-out readout, and the seed-2 weights in 1ba2989 before seed 2's.
- **Checks passed** (`checks.json`).
  - A full forward against the cached logits matches within 2.9e-6, with 0 argmax mismatches.
  - A15 parity holds on 6,583 identical native windows within 1.4e-6.
  - Zero-update F and T reproduce Base, and the merged sequences equal Base.
  - Pooled CE equals numpy within 2e-16. Microbatch gradients equal the whole batch within 1e-7.
  - With representation lr 0, T equals F within 6e-6. Resume is exact, the optimizer is bound, and the frozen prefix is
    unchanged.
  - The four structurally zero-gradient block-15 tensors are as in A30 (7,491,461 effective trainable).

## Results: all held-out P runs, out-of-fold over both folds

| Seed, arm | Base-wrong (fixed) | Base-correct (broken) | Net fixed | Fix rate | Damage rate | Run accuracy − Base [95% session bootstrap] |
|---|---|---|---|---|---|---|
| s0 F_U | 254 (30) | 6,534 (17) | **+13** | 0.118 | 0.26% | +0.0019 [−0.0006, +0.0045] |
| s0 F_E | 254 (97) | 6,534 (314) | −217 | 0.382 | 4.81% | −0.0320 [−0.0425, −0.0220] |
| s0 T_U | 254 (67) | 6,534 (58) | **+9** | 0.264 | 0.89% | +0.0013 [−0.0034, +0.0060] |
| s0 T_E | 254 (88) | 6,534 (113) | −25 | 0.346 | 1.73% | −0.0037 [−0.0114, +0.0034] |
| s2 F_U | 237 (27) | 6,505 (17) | **+10** | 0.114 | 0.26% | +0.0015 [−0.0011, +0.0042] |
| s2 F_E | 237 (99) | 6,505 (284) | −185 | 0.418 | 4.37% | −0.0274 [−0.0377, −0.0179] |
| s2 T_U | 237 (49) | 6,505 (56) | −7 | 0.207 | 0.86% | −0.0010 [−0.0055, +0.0032] |
| s2 T_E | 237 (70) | 6,505 (113) | −43 | 0.295 | 1.74% | −0.0064 [−0.0131, +0.0003] |

**Paired comparisons** (net fixed difference; run accuracy difference [95% session bootstrap]).

| Comparison | Seed 0 | Seed 2 |
|---|---|---|
| F_E − F_U (sampling, frozen) | −230; −0.0339 [−0.0433, −0.0248] | −195; −0.0289 [−0.0388, −0.0200] |
| T_E − T_U (sampling, adapted) | −34; −0.0050 [−0.0100, −0.0005] | −36; −0.0053 [−0.0097, −0.0011] |
| T_U − F_U (representation, uniform) | −4; −0.0006 [−0.0045, +0.0031] | −17; −0.0025 [−0.0065, +0.0012] |
| T_E − F_E (representation, error-focused) | +192; +0.0283 [+0.0211, +0.0358] | +142; +0.0211 [+0.0128, +0.0297] |

- **Bootstrap.** 2,000 draws over 398 held-out sessions. Each run is assigned to its majority-recording session, an
  approximation for runs that span sessions. Stream regeneration was not assessed, since there was one generation.
- **Direction.** Fixes are mostly A4C runs that Base read as A5C. Breaks are mostly correct A5C runs.
  - Example, seed 0 held-out fold 1: T_E fixed 46 of 70 A4C-read-as-A5C errors and 3 of 69 A5C-read-as-A4C errors. It
    broke 67 correct A5C runs.
  - So the trained branches mainly move the A4C/A5C boundary towards A4C, and E moves it furthest.

**Full sequences** (frozen B3 runs relabelled, merged, complete RLE scored; multi segment F1@50).

| Seed, held-out fold | Base | F_U | F_E | T_U | T_E |
|---|---|---|---|---|---|
| s0, fold 1 | 0.9499 | 0.9511 | 0.9179 | 0.9526 | 0.9460 |
| s0, fold 0 | 0.9639 | 0.9654 | 0.9366 | 0.9649 | 0.9585 |
| s2, fold 1 | 0.9508 | 0.9534 | 0.9250 | 0.9503 | 0.9408 |
| s2, fold 0 | 0.9649 | 0.9647 | 0.9388 | 0.9637 | 0.9633 |

- **E against U.**
  - E − U is negative in all eight sampling comparisons: F −0.026 to −0.033, T −0.0005 to −0.0096.
  - F_E loses 113–139 matched segments against F_U. Native macro-F1 falls to 0.934–0.940 against Base's 0.974–0.983.
- **U against Base.** F_U +0.0012, +0.0015, +0.0026, −0.0001; T_U +0.0027, +0.0010, −0.0004, −0.0011. U raises native
  macro-F1 (for example 0.9758 → 0.9810), mostly through fewer A4C read as A5C.
- **No valid segment intervals.** Streams share sessions, so no valid independent unit exists for segment-level
  intervals. The inherited stream bootstrap in `readout_s{0,2}.json` treats streams as independent and is not used for
  any claim here.

## Answers

1. **Error-focused sampling does not give net correction.** In both source seeds and both representations, E − U is
   negative: −195 to −230 runs for F and −34 to −36 for T, with every session-bootstrap interval excluding 0.
   - E roughly doubles to triples the fix rate (0.30–0.42 against 0.11–0.26), but it breaks many more correct runs
     (1.7–4.8% against 0.3–0.9%).
   - With errors at 3–4%, that damage exceeds the gains.
2. **No transferable F/T difference under uniform sampling.** T_U − F_U is −4 / −17 runs, with intervals including 0.
   - T memorises its fit fold far better: it fixes 79–111 of 105–145 training errors with at most 4 breaks, against F_U's
     14–23.
   - On held-out sessions it fixes more (49–67 against 27–30) but also breaks more (56–58 against 17).
   - Under E, T is much less harmful than F (+142 / +192) but still below Base.
3. **Full segmentation.** E damages it, F_E strongly (−0.026 to −0.033). U keeps it level, from −0.0011 to +0.0027
   against Base.
   - The changes are mostly A4C/A5C relabelling. A4C runs read as A5C get fixed, while correct A5C runs and their
     matched segments get lost.
4. **Recommendation: do not advance to a full-fit candidate and a development evaluation.** Neither net correction nor
   full segmentation has a reproducible gain.
   - The pattern is consistent across seeds, folds and representations: training on these labels moves the A4C/A5C
     boundary, and held-out sessions pay for every fix with a break.
   - **Next evidence.** Turn to a different kind of evidence on the A4C/A5C distinction before more training: label and
     visual separability of the A5C runs that Base reads as A4C and of the correct A5C runs that get broken. That means
     their per-recording concentration, structured cues such as LVOT visibility, or external A5C-labelled data.
   - **Not supported.** Another sampling, head or block variant is not supported by this round.
   - **What this does not show.** Lack of significance is not equivalence, and none of this shows that A4C/A5C cannot be
     separated.

## Cost and files

- **GPU.** 4.17 of 8 h.
  - A08 passes with prefix caches: 0.59 h.
  - T training, 8 models: 3.28 h.
  - Held-out readouts: 0.27 h.
  - Checks: 0.04 h.
  - No failed run.
- **CPU.** About 0.9 of 4 h, including the agent's active wall clock. F trained in about 2 s per model.
- **Records.**
  - `split.json`, `checks.json`, `schedule.json`, `base_s{0,2}.json`.
  - `train/` and `weights_s{0,2}.json`.
  - `readout_s{0,2}.json` and `report.json`.
- **Server only.** Fold pools, manifests, run lists, batch plans, cross-fit weights, held-out predictions and all
  identifiers. The prefix caches were deleted after use, and their records were kept.
