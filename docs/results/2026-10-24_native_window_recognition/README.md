# A29: interval recognition on A08's native request windows, frozen against adapted (2026-10-24)

**Question.** Keep A08's own 16-frame, 10 Hz request windows and the nearest-centre pooling of their probabilities inside
each Base run, so that a zero update is Base exactly. Does adapting the recognition branch's last block, norm and
projection (T) then classify the runs better than the frozen representation (F)? And does that improve full
segmentation?
- **Arms.** Both start from A08 all_s{k}'s original weights.
  - F trains only the copied classifier (2,565 parameters).
  - T also trains `encoder.blocks.15`, `encoder.norm` and `encoder.head.1` (7,497,989), through the real request windows.
  - Both run in eval mode, on A27's T/P runs, multiplicity weights and batches.
  - Both use the same pooled-probability CE, with no KL and no teacher.
- **Unchanged.** The original A08 + B3 path is never modified.
- **Scope.** T − F is the attribution for adaptation.
- **How this differs from A27 and A28.** A27 used the same pooled windows with a retention KL and a head only. A28
  adapted the same modules but on one evenly spaced clip per run, where Init was already below Base.
- **Data limits.** The development halves are reused exploratory feedback. The A15 check pool's recordings trained the
  encoder, so it is not a holdout. Test and Pediatric were not read.
- Configuration: `configs/experiments/video_native_window_recognition_v1.yaml` (2966d62). Task sheet
  `claude_prompts/2026-10-24-native-window-recognition-training.md`.

**Design.**
- **Windows.** A15's request windows exactly: stride-5 targets on the whole stream, 16 frames at 10 Hz, A08's marker,
  gamma and preprocessing. Samples map to their nearest centre, with no clamping to reference boundaries.
- **Runs.** A27's pinned lists. The training runs use 49,654 windows, 224–226 per update on average (at most 391). Every
  run keeps all its windows.
- **T's frozen prefix.**
  - For each seed, the input of the last block (393 tokens × 768, fp32, 60 GB) was cached once.
  - Every cached window was verified: cached prefix plus the original suffix reproduces A15's logits within 1.4e-6 and
    2.9e-6, with 0 argmax mismatches.
  - Each cache was deleted after its seed's training.
- **Training.** 1,350 updates. AdamW with weight decay 1e-4; classifier lr 1e-4, T representation lr 1e-5. Warm-up 135
  then cosine, clip 1, final weights only. Duplicate windows of a batch are computed once.
- **Monitoring** (training data only, fixed before training).
  - All Base-wrong P runs (274 / 256), plus a fixed 1,000 Base-correct P runs.
  - Recorded at updates 0, 270, …, 1,350.
  - Read as CE, true-class margin, fixed and broken, split by pure/mixed and single/multi-recording.
- **Deployment.** Each Base run takes the argmax of the arm's mean per-sample probability. Equal neighbours merge, and
  complete sequences are scored.
- **Order.**
  - Configuration 2966d62, then trainer and preparation a7b2b94, then checks 096438e.
  - Training: chain PID 719466; T_s0 ran as Python PID 719468, 1,350 real updates per run.
  - The final weights and cache records were committed in 8f7eb08 before evaluation.
  - The development halves and check pool were evaluated once.

**Checks passed** (`checks.json`).
- **Zero update.**
  - On all training P runs' windows, F and T logits equal the original q within 5.7e-6 and 4.3e-6, with exact argmax,
    and both arms' run labels equal Base.
  - F's zero-update path also reproduces Base on all 3,187 pool sequences and the four stored development rows.
  - T's zero update was checked on the training windows, not online on the development halves.
  - So Init = Base.
- **Objective.** Pooled CE equals numpy within 2e-16.
- **Microbatching.** Accumulated microbatch gradients equal the whole batch within 9.6e-8.
- **Forced equality.** With representation lr 0, T equals F within 3.2e-6, the representation is unchanged and the
  frozen prefix is unchanged.
- **Mechanics.** A 3 + 3 resume equals 6 exactly. Gradients reach only the trainable set, and the optimizer binds exactly
  those 7,497,989 parameters. The A08 file is unchanged.
- **Engineering faults before training** (booked as 0.014 GPU h, no result read).
  - The suffix was first called with the pre-pooling token grid (8, 56, 56) instead of block 15's (8, 7, 7).
  - The cache key hashed the whole trainer file, so that fix invalidated it.
  - The key now covers only the code that determines the cache content. The cache was re-stamped after a 64-window online
    spot check (relative token difference 3.0e-6, logits 1.1e-6), recorded in `prefix_cache_s0.json`.
  - The first spot check used an absolute token bound and failed on batch-size-dependent kernel differences.

## Results (seeds 0 and 2)

| System | s0 sel | s0 cal | s2 sel | s2 cal | Check s0 / s2 | Native cal s0 / s2 |
|---|---|---|---|---|---|---|
| Base (A08 + B3) = Init | 0.9563 | 0.9827 | 0.9564 | 0.9837 | 0.9603 / 0.9555 | 0.9929 / 0.9856 |
| F | 0.9544 | 0.9846 | 0.9514 | 0.9837 | 0.9605 / 0.9564 | 0.9929 / 0.9788 |
| T | 0.9544 | 0.9836 | 0.9514 | 0.9837 | 0.9605 / 0.9568 | 0.9929 / 0.9788 |
| A27 W / I, A28 Frozen / Adapted (stored, other inputs) | 0.9483–0.9563 | 0.9817–0.9827 | 0.9523–0.9543 | 0.9827–0.9837 | – | – |

**Fixed gate: not met by either arm**, so A29 closes with two seeds and no seed-1 run.

| Part | F | T | Required |
|---|---|---|---|
| Mean dF against Base | −0.0013 | −0.0015 | ≥ +0.003 |
| Half means (selection / calibration) | −0.0035 / +0.0010 | −0.0035 / +0.0005 | each ≥ −0.005 |
| Lowest cell | −0.0050 (s2 selection) | −0.0050 (s2 selection) | ≥ −0.015 |
| Native macro-F1 / boundary recall difference | −0.0017 / −0.0007 | −0.0017 / −0.0013 | ≥ −0.003 / ≥ −0.010 |
| Missed short true segments (four cells) | 21 against Base's 22 | 22 against 22 | ≤ Base + 3 |
| Calibration feasibility | fails seed 2 (native 0.9788) | fails seed 2 (native 0.9788) | each seed |

**T − F** (multi segment F1@50, 95% paired intervals). In seed 0 it is 0, −0.0010 [−0.0030, 0.0] (one lost segment) and 0;
in seed 2 it is 0, 0 and +0.0004 [0.0, +0.0012] on check. The two arms' final segmentations are nearly identical, with
no adaptation effect.

**Against Base** (both arms alike).
- Selection: −0.0020 (one lost segment) and −0.0050 (three lost).
- Calibration: 0 to +0.0019.
- Check: +0.0002 to +0.0013.
- Seed 2 calibration loses one native clip (A4C read as A5C, 0.9856 → 0.9788), and check native falls by one clip in both
  seeds.

**Labels of the fixed Base runs** (development and check).
- Accuracy equals Base's or moves by at most one run per cell, for example s0 calibration 0.9850 F against 0.9839.
- Fixed and broken are 0–4 each, in 0–6 recordings.

**Training-pool readout** (`training_summary.json`, monitor groups and full P list).

| Run | Base-wrong P runs | Fixed | Broken | CE on Base-wrong, start → end | Margin on Base-wrong | Margin on Base-correct (sample) |
|---|---|---|---|---|---|---|
| F s0 | 274 | 17 | 2 | 2.00 → 1.99 | −1.71 → −1.72 | 3.75 → 5.53 |
| T s0 | 274 | 21 | 1 | 2.00 → 1.92 | −1.71 → −1.64 | 3.75 → 5.83 |
| F s2 | 256 | 7 | 5 | 2.10 → 2.02 | −1.84 → −1.78 | 3.82 → 5.51 |
| T s2 | 256 | 10 | 5 | 2.10 → 1.94 | −1.84 → −1.68 | 3.82 → 5.81 |

- **Wrong runs barely move.** On the Base-wrong training runs, both arms leave CE near 2 and the true-class margin near
  −1.7. These are confident errors, and training fixes only 7–21 of them.
- **Correct runs become more confident.** The margin on Base-correct runs rises from about 3.8 to 5.5–5.8.
- **T against F.** T fixes 3–4 more wrong runs than F and lowers their CE slightly more. Most of the runs fixed in either
  arm are pure (12–15 of 17–21 in seed 0).
- **Not an established cause.** This is a description of the training runs, not a demonstrated cause.

## What this round shows

1. **The native window input holds Base exactly at zero update**, unlike A28's single clip. So the start point is not
   degraded.
2. **Adapting the last block, norm and projection did not beat the frozen representation.**
   - T − F is 0 in most cells, and the arms' final segmentations nearly coincide.
   - On its own training runs, T fixes only 3–4 more of the 256–274 confident errors than F.
3. **Neither arm improves full segmentation.**
   - Both lose 1–3 matched segments on selection and gain little on calibration and check.
   - Seed 2 loses one calibration native clip, which fails the feasibility floor.
   - With plain pooled CE, both mostly sharpen already-correct runs.
4. **What is closed.** This configuration is closed: native windows, nearest-centre pooling, pooled CE without KL, the
   head only against the last block, norm and projection, these learning rates.
   - It does not show that representation adaptation cannot help.
   - It does not establish why the confident errors stay. Candidates such as the training runs being A08's own training
     data, label or reference questions, or the update size were not separated.
   - No variant or seed-1 run follows.

## Cost and files

- **GPU.** 2.58 of 6 h.
  - Prefix caches: 0.59 h.
  - T training: 1.51 h, including the monitoring and full readouts read from the cache. The check projected 0.55 s per
    update; the wall time was 1.9–2.1 s per update with the readouts.
  - Checks: 0.12 h, including the failed launches.
  - Evaluation: 0.36 h (65,724 T windows online, 0.016 GPU s per window).
- **CPU.** About 1.2 of 3 h, including the agent's active wall clock. F trained in about 2 s per run on CPU.
- **Deployment cost.**
  - F adds one 512 × 5 layer to A08's existing embeddings, so it is effectively free.
  - T adds a second suffix per request window. A shared frozen prefix is possible, but this evaluation ran a full copy.
- **Records.**
  - `preparation.json` (a7b2b94) and `checks.json` (096438e).
  - `training_summary.json`, `train/` and `prefix_cache_s{0,2}.json` (8f7eb08).
  - `screen.json`: rows, counts, run tables, comparisons, accounting, gate and context.
- **Frozen.** `models_frozen/video_native_window_recognition_v1/{F,T}_s{0,2}`. Each holds the trainable tensors only,
  applied onto A08 all_s{k}: F 3e75b7016ecc and e941fe921a5f, T 8a0ffa98f85d and c96f2c104ff6.
- **Server only.** Monitor groups, cache-row lists, per-sequence predictions and all identifiers. The 60 GB prefix caches
  were deleted after use, and their records were kept.
