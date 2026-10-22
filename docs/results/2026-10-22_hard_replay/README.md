# A20: does replaying hard training positions improve the single CLS video model? (2026-10-22)

**Question.** A19 ended by pointing the next training motive at the video model's own predictions on the windows where
errors remain. A10's switch-dense 12-window batches are kept. Does adding a 4-window replay group drawn from training
positions where the frozen A08 classifies badly (H) give a stronger single CLS video model than the same replay drawn
uniformly (U)?
- Hard-example training is a known method. No algorithmic contribution is claimed.
- U against H is the matched comparison. Both use the same streams, short/non-short bucket, four-target class tuple,
  loss and recipe. U/H against each seed's A08 + B3 (Base) is the full-system test.
- A10 R and A18 R (12 windows, no replay) are background references, not the control.
- The check pool's sessions are unseen by the replay. But its recordings trained the video encoders, so it is not a new
  holdout. The development halves are reused exploratory feedback.
- The test split and EchoNet-Pediatric were not read.
- Configuration: `configs/experiments/video_hard_replay_v1.yaml` (5ea169f), fixed before any A20 output.

**Design.**
- **Start and model.** Per seed, A08 all_s{k} is strictly loaded into the unchanged TargetMViT (EchoPrime-initialised
  MViTv2-S, 16 RGB frames, target marker, 512-d projection, five-class CLS). Deployment is one CLS forward plus B3.
- **Batches.** Every update keeps A10's pinned 12 windows (VSH rows 0–1,349: 4 native, 4 change-containing constructed,
  4 same-label join) and adds one 4-window replay group, 16 windows in all. The 12 are never replaced.
- **Replay pool.** Only the 1,500 multi streams of A15's large-fit pool are used. No check, calibration, selection, test
  or Pediatric sequence enters it, and no development error list is used.
  - A candidate is 4 adjacent stride-5 windows of one stream, with their original 16-frame context, gamma, marker and
    frame references.
  - Only mixed groups count: at least one of the four windows' 16-frame reference labels holds two or more views. That
    leaves 22,724 of 34,686 groups.
  - A group is short when one of its four targets lies in a true label run of at most 16 samples. There are 5,938 short
    groups (from 987 streams) and 16,786 non-short groups (1,478 streams).
  - Difficulty is the seed's own A08 mean CE over the four targets. It comes from the pinned A08 pool logits and was
    computed once before training.
- **Sampling.** Each seed has one numpy stream (`default_rng([20261021, 20, seed])`), with three uniforms per update.
  - Even updates use the short bucket and odd updates the non-short bucket.
  - U picks a stream uniformly among those with candidates in the bucket, then one candidate uniformly in that stream.
  - H takes the candidates in the same stream and bucket that share U's four-target class tuple and sorts them by A08
    CE. It draws one uniformly from the top quarter. When the stratum has one candidate, H = U.
- **Loss and recipe.**
  - Both arms use all 16 windows: mean CE + mean KL(own frozen A08 ‖ student), weight 1, temperature 1.
  - A10 R's recipe: 1,350 updates and AdamW. The encoder is frozen for the first 135 updates, then follows warm-up and
    cosine at lr 1e-5. New layers use 1e-4, weight decay 1e-4 and clip 1.
  - Teacher logits are cached: the base 12 from A18's verified cache, the replay 4 from the A08 pool logits.
  - Only the final weights are kept.
- **Order.**
  - The weights and run records were committed (e1106ff) before any evaluation.
  - The sensitivity choices were committed (51a33d2; the git object's sha was verified) before the selection half or the
    check pool was read.
  - The check pool was evaluated once.
- **Checks passed before training.**
  - The 12 base windows equal A10's.
  - Cached and online teacher logits differ by at most 2.9e-6 over 64 windows, with the same argmax. The teacher is
    unchanged and has no gradient.
  - The sampling was recomputed independently, and every U/H constraint holds: same stream, bucket, parity and tuple;
    all mixed; H in the top quarter; large-fit only.
  - H forced to U reproduces U's loss exactly, with parameters within 4.8e-7.
  - The encoder is frozen before the warm-up and receives gradients after it. The optimiser groups are bound.
  - Resuming after 3 updates matches 6 straight updates: loss within 4.5e-7, and parameters within 6.0e-5 against a
    repeat difference of 6.0e-5.
  - Timing was 0.955 s per update, a projection of 2.15 of 4 GPU h.
- **Evaluation.**
  - The primary readout decodes every new model with its seed's committed A08 B3 numeric setting (U@A08, H@A08).
  - Sensitivity readouts are one own-calibration choice per model from the standard 40 settings, and plain argmax.
  - Old A08, frozen Ordered, A10 R and A14 P development rows reproduced. A19's A08 check rows reproduced.

**What the replay contained.**
- **H equals U in 83.4% (seed 0) and 83.9% (seed 2) of updates.** In 73–75% of updates the stratum (same stream, bucket
  and tuple) held one candidate.
- **On the remaining 224 and 217 updates, H's groups were harder.** Over all replay windows, H's mean A08 CE was 0.256
  against U's 0.215 for seed 0, and 0.246 against 0.215 for seed 2. The A08 error rate on replay windows was 8.5% against
  6.9% for seed 0, and 8.1% against 6.9% for seed 2.
- **Coverage was the same.** Each arm used 1,401–1,431 source recordings from 369–375 sessions, and no stream was reused
  more than 7 times.
- **Training-set readout** (`training_summary.json`, training windows only).
  - On the updates where H ≠ U, the student's replay CE was higher for H (0.311 for seed 0, 0.202 for seed 2) than for
    U (0.066 and 0.028).
  - On the updates where H = U, both arms had almost the same replay CE (0.235 for seed 0, 0.237–0.239 for seed 2).
  - The base-12 CE and accuracy did not differ between arms (CE 0.104, accuracy 0.968).

## Results

Multi segment F1@50. Development has four cells (2 seeds × 2 halves); the check pool has one cell per seed.

| System | Development mean | s0 sel, s0 cal, s2 sel, s2 cal | Check s0 / s2 (mean) |
|---|---|---|---|
| Base (A08 + B3) | 0.9698 | 0.9563, 0.9827, 0.9564, 0.9837 | 0.9603 / 0.9555 (0.9579) |
| **U@A08** | 0.9719 | 0.9577, 0.9866, 0.9625, 0.9808 | 0.9593 / 0.9585 (0.9589) |
| **H@A08** | 0.9724 | 0.9577, 0.9866, 0.9644, 0.9808 | 0.9597 / 0.9589 (0.9593) |
| R@A08 (A10 recipe, no replay; A18's committed rows) | 0.9695 | 0.9559, 0.9827, 0.9567, 0.9828 | not evaluated |
| R + B3 (its own committed choice) | 0.9711 | 0.9566, 0.9846, 0.9606, 0.9828 | not evaluated |
| A14 P | 0.9724 | 0.9640, 0.9855, 0.9564, 0.9837 | not evaluated |
| frozen Ordered + B3 | 0.9661 | 0.9385, 0.9817, 0.9644, 0.9797 | not evaluated |

**Promotion rule: not met for U or H.**

| Part | U@A08 | H@A08 | Required |
|---|---|---|---|
| Mean dF against Base | +0.0021 | +0.0026 | ≥ +0.003 |
| Half means (selection / calibration) | +0.0037 / +0.0005 | +0.0047 / +0.0005 | each ≥ −0.005 |
| Lowest cell | −0.0029 (s2 calibration) | −0.0029 (s2 calibration) | ≥ −0.015 |
| Native macro-F1 difference | −0.0017 | 0.0 | ≥ −0.003 |
| Boundary recall difference | +0.0058 | +0.0065 | ≥ −0.010 |
| Missed short true segments | 13 against Base's 22 | 13 against 22 | ≤ Base + 3 |
| Calibration row feasible | s2 fails (native 0.9788 < 0.980) | both seeds | each seed |

So no seed 1 was trained. A20 closes with two seeds and no ratio, CE weight, re-mining or length variant.

### 1. Did the predictions on hard windows improve?

**Partly. 14–18% of the windows that A08 got wrong became right, but most stayed confidently wrong. Most of the change
also appears without replay.** Window accuracy is the argmax of each window against its reference target. An
"A08-wrong" window is one where the seed's own A08 argmax is wrong.

Check pool, per seed (842 A08-wrong windows for s0, 795 for s2):

| Window readout | Base s0 | U s0 | H s0 | Base s2 | U s2 | H s2 |
|---|---|---|---|---|---|---|
| All windows accuracy | 0.9537 | 0.9592 | 0.9593 | 0.9563 | 0.9621 | 0.9623 |
| Mixed windows accuracy | 0.9084 | 0.9236 | 0.9239 | 0.9152 | 0.9314 | 0.9322 |
| Short-run windows accuracy | 0.9150 | 0.9334 | 0.9334 | 0.9199 | 0.9361 | 0.9383 |
| A08-wrong windows now right | – | 120 | 122 | – | 136 | 140 |
| A08-right windows now wrong | – | 19 | 20 | – | 30 | 30 |
| Mean CE on A08-wrong windows | 2.213 | 2.230 | 2.208 | 2.208 | 2.145 | 2.124 |

- U and H fix 14–18% of the A08-wrong windows and break 19–30 others.
- The mean CE over all A08-wrong windows stays at 2.12–2.23, against A08's 2.21. So these errors remain confident, and
  the extra supervision did not move most of them.
- **Post hoc background, development halves only** (`background_windows.json`; outside every rule; the check pool was
  not read again).
  - A10 R is the same recipe from the same start without replay. Over the four cells it already fixes 112 A08-wrong
    windows and breaks 38.
  - U fixes 130 and breaks 51, and H fixes 132 and breaks 48.
  - So the replay adds 18–20 fixed windows and 10–13 broken ones, a net of 5–10 windows. Most of the window gain comes
    with A10's recipe itself.

### 2. Did this become a full gain that keeps short true segments and removes false ones?

**No. The models keep more true segments but add false short segments, and the net gain stays under the line.**

| Count (short in brackets) | Base | R@A08 (no replay) | U@A08 | H@A08 | A14 P |
|---|---|---|---|---|---|
| Development missed true segments | 51 (22) | 41 (15) | 38 (13) | 38 (13) | 52 (23) |
| Development false segments | 72 (39) | 84 (53) | 77 (49) | 75 (47) | 60 (30) |
| Check missed true segments | 187 (79) | – | 161 (61) | 161 (59) | – |
| Check false segments | 248 (153) | – | 265 (175) | 261 (172) | – |

- **Recall improves.** Against Base, U and H miss fewer true segments, including short ones, on development and on the
  check pool. Boundary recall is higher in every cell where it moved (development mean +0.006; check +0.006 and
  +0.008).
- **Precision falls.** Both add false short segments: 47–49 against 39 on development, 172–175 against 153 on check.
  Boundary precision falls (development 0.9610/0.9629 against 0.9662; check by 0.0010–0.0044).
- **Net.** Development +0.0021 (U) and +0.0026 (H); check +0.0010 and +0.0014 (s0 −0.0010/−0.0006, s2 +0.0031/+0.0034).
  - No paired interval against Base excludes zero except H on s2 selection (+0.0080 [+0.0001, +0.0168]).
  - The s2 calibration cell (−0.0029 for both) has an interval ending at exactly 0.
- **Native recognition.** H keeps it unchanged. U_s2 drops calibration native macro-F1 to 0.9788, one more A4C read as
  A5C, which fails the feasibility part of the rule. On the check pool both raise seed-0 native macro-F1 from 0.985 to
  0.990.
- **Against the same recipe without replay at the same decoder (R@A08, background).** U/H are +0.0024/+0.0029 (cells
  +0.0018, +0.0039, +0.0058/+0.0077, −0.0019). They make fewer false segments (75–77 against 84) with similar recall.
  So replaying mixed windows offsets part of the recipe's precision cost. It does not take the system past A08 + B3 by
  the rule, or past A14 P (U −0.0005, H 0.0000 over the same four cells).

### 3. Is H better than the matched U?

**No.** H − U is 0.0, 0.0, +0.0019 and 0.0 over the development cells (mean +0.0005), and +0.0004 on the check pool for
each seed.
- In segment accounting, H and U differ by at most three segments in any category of any cell.
  - On development, H swaps one false segment for another in each seed-0 half, removes 2 short false segments on s2
    selection, and matches U on s2 calibration.
- At the window level they fix 120 against 122 and 136 against 140 check windows.
- The reason is visible in the replay. 84% of updates are identical, and the other 16% (harder for H by A08 CE) left no
  held-out trace.
- Hard-position training is not supported. The shared U/H change is a common replay effect, and mostly a recipe effect.

**Sensitivity readouts** (never used for the rule).
- H own-cal is close to H@A08: development 0.9719, check 0.9602.
- U_s2's own calibration choice (quantile 0.8, threshold about 7e-5) cuts far too often. It gives 0.9194 on selection
  with 68 false segments (54 short), and 0.9503 on check. So H own-cal − U own-cal (+0.0431 on s2 selection) is a
  decoder-choice artefact.
- Argmax readouts of U and H are above A08's argmax in three cells and equal to it on s2 calibration.

## What this round shows

1. **Hard windows improved only partly, and mostly not because of the replay.** U and H fix 14–18% of A08's wrong check
   windows, and the mean CE over those windows stays at 2.12–2.23 against A08's 2.21. On development, the same recipe without replay already
   makes most of these corrections.
2. **No full gain.** The models keep more true and short true segments but add false short segments. The full-system
   gain against A08 + B3 is +0.002–0.003 on development and +0.001 on the check pool, under the +0.003 line. U also
   fails feasibility on seed 2.
3. **H is not better than U.** Choosing replay positions by the current model's own CE adds nothing over uniform replay
   under these matched constraints. Most H updates were the same as U, and the rest left no held-out difference.
4. **How this decides the next training motive.**
   - Emphasis on training windows is removed from the list. That covers which windows are replayed, uniformly or by
     the current model's CE, under the same CE + KL labels.
   - Together with A14–A19 (routing, teacher targets, decoder points), every lever tried on the same labels and model
     family has moved full segmentation against A08 + B3 by about +0.001 to +0.003, without a consistent gain.
   - The windows that remain wrong stay confidently wrong under U, H and R. Recall gains have come with short false
     segments.
   - So the next training motive needs information that the current supervision lacks about those windows. Another way
     to weight, sample, route or decode the same supervision would not supply it.
   - This is a reading of A14–A20, not a proposal. It does not show that better labels or inputs would help.

## Cost and files

- **GPU.** 1.98 of 4 h.
  - Checks: 0.02 h, including 0.003 h for the first attempt, which stopped on an array-shape error before any update.
  - Four training runs: 1.36 h.
  - Calibration inference: 0.13 h.
  - Selection and check-pool inference: 0.47 h.
- **CPU.** 1.09 of 3 h at the final commit.
  - This includes preparation (0.23 h with its decode workers), the decode/evaluation workers outside GPU time
    (0.19 h), the training and post-hoc readouts, and the agent's active wall clock booked by hand.
  - Waiting for background jobs is not booked. Loader workers inside GPU training jobs are not counted again (A18/A19's
    rule).
- **Records.**
  - `preparation.json`, `checks.json`, `manifests_index.json`, `training_summary.json` and `train/` (run records and
    logs), committed before calibration (e1106ff).
  - `choices.json`, committed before selection (51a33d2).
  - `screen.json`: rows, segment counts, paired comparisons, segment accounting, window groups and the promotion rule.
  - `background_windows.json`: post hoc and outside every rule.
- **Frozen weights.** `models_frozen/video_hard_replay_v1/{U,H}_s{0,2}` on the server (final.pt sha 73cafe137a58,
  c45707ba43f1, 4efb1e2f3f33 and b5602a1f82b8).
- **Server only.** Replay manifests (mappings and scores), logits, per-sequence predictions and all identifiers.
