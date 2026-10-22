# A22: real intermediate frames against repeated frames in a short window (2026-10-22)

**Question.** The old 10 Hz input skips the raw JPEGs between samples: all protocol-train recordings play at 30 fps,
and same-source steps are 3 raw frames. Keep the same short time support, the same 16 input slots and the same video
model. Does inserting real intermediate source frames (Interleaved) beat repeating the already-sampled frames
(Repeated)? Can it keep short true segments while holding down the short false segments that A20/A21's continuations
added?
- Interleaved against Repeated is the matched comparison. The 8 old samples, the marker, the targets, the teacher and
  the recipe are identical; only 7 odd slots differ.
- A10 R (A18 R_s1 for seed 1) is reused unchanged as the Long10 control: 16 frames at 10 Hz over a 1.5 s support from
  the same A08 start, batches and CE + KL.
- Interleaved/Repeated against each seed's A08 + B3 (Base) is the full-system test.
- Seconds and Hz are the playback time axis, not a verified acquisition rate or cardiac phase. Any Interleaved −
  Repeated difference is first attributed to the extra observed frames, and appearance can explain it as well as motion.
- The check pool's sessions are unseen by A22's batches, but its recordings trained the A08 encoder, so it is not a new
  holdout. The development halves are reused exploratory feedback. The test split and EchoNet-Pediatric were not read.
- Configuration: `configs/experiments/video_temporal_density_v1.yaml` (9742eb2), fixed before any A22 output.

**How this differs from earlier work.**
- V01 showed that multiple frames help and that long cross-view context can mislead.
- V02 and A02–A04 changed readouts or supervision of the same frames.
- V03 made the request times denser (0.5 s to 0.1 s), not the frames inside a window.
- A22 changes the local raw frames the model sees and reopens no decoder, router, resampling or spatial branch.

**Design.**
- **Model.** Per seed, A08 all_s{k} is strictly loaded into the unchanged TargetMViT (34,640,549 parameters, one CLS
  forward per window). There is no new branch, head or token.
- **Windows.** For each original sequence (N samples at 10 Hz) and the old target c, a = window_frames(c, N, 8, 8)
  gives 8 old positions. That function's end shift and short-sequence repeats are kept.
  - The old support is 0.7 s (8 samples), against Long10's 1.5 s (16 samples).
  - **Repeated.** Slots 2i and 2i+1 both hold old ref[a_i].
  - **Interleaved.** Even slots hold the same old refs. Odd slot 2i+1 (i < 7) holds one real JPEG inside a_i's own
    source interval. Slot 15 repeats a_7 in both arms.
  - **Intermediate frame.** The left sample's own source gives its next planned 10 Hz sample, even across a fragment
    end. At the end of a source the raw frame count is used. The frame is mid = floor((f + next) / 2), the nearest frame
    with ties to the earlier, which is f + 1 for a step of 3. It carries the left fragment's gamma and is looked up
    through FramePaths' numeric-order list.
  - **No intermediate available.** If no raw frame lies inside the interval, or a_i = a_{i+1}, the odd slot repeats
    the left frame.
  - **Marker.** The marker comes only from planned times (2a_i, a_i + a_{i+1}, 2a_7): the middle slot whose
    coordinate equals 2c. It is the same in both arms.
  - **Unchanged.** Gamma, the 256/224 crop, normalisation and slot order are unchanged.
- **Outputs.** Outputs stay at the original stride-5 requests with nearest-centre mapping, and B3 is unchanged.
- **Data and teacher.** A10's pinned 12-window batches are used. The teacher is each seed's frozen A08 on the original
  16-frame 10 Hz window, from A18's verified cache. Loss is CE + KL(own A08 long ‖ student).
- **Recipe.** A18's cached-own step with only the window request replaced. 1,350 updates; the encoder and its
  projection run at 1e-5 and are frozen for the first 135 updates; marker and classifier 1e-4; the unused head group
  is kept for layout.
- **Order.**
  - The weights and run records were committed in 705b59c. That commit fell in the same second (15:53:45 UTC) as the
    launch of the calibration process, before any calibration data was loaded.
  - The sensitivity choices were committed in 133515b, with the git object's sha verified, before selection or the
    check pool was read.
  - The check pool was evaluated once.

**Inputs** (`preparation.json`, all 16,200 training windows per seed).
- **Rebuild.** The windows were rebuilt from the complete sequences. The Long10 rebuild equals A10's VSH windows,
  markers and targets for every window.
- **Source plans.** All 3,167 sources have a regular step-3 plan that matches their raw frame count, so every raw JPEG
  exists. 238,019 distinct frames were decoded with no failure and no fallback.
- **New frames.** Every window gains new frames, 6.98–6.99 on average. About 1.5% gain only 6, because a source ended
  with no raw frame left inside the interval.
- **Pixel difference.** An inserted frame differs from its left frame by 8.2 on average (0–255 scale). Only 0.26% are
  identical.
- **Same on development and check.** Every window gains new frames (6.98–6.99 on average). No intermediate frame failed
  to decode.
- **Label share.** In the short windows 96–98% of slots carry the target label, against 92–95% in the long windows.
- **Teacher error.** The teacher's error on the training windows is 3.3–3.7%, and 6.6–7.3% on change windows.

**Checks passed** (`checks.json`).
- **Synthetic windows.** Every case matched an independent loop implementation. Even slots and slot 15 were equal
  across arms, every intermediate stayed inside its left interval, and the markers were correct. The cases covered:
  - source boundaries and two fragments of one source
  - gamma changes and sequence ends
  - N < 8 with repeated positions
  - both raw-end cases (with and without a remaining raw frame)
  - steps of 2, 3 and 5
  - the fixed fallback rule
- **Real inputs.** On real batches the even slots and slot 15 are bit-equal across arms, odd slots differ, and markers
  and targets are equal.
- **Teacher.** The cached teacher equals online A08 over 72 windows, and its state is unchanged.
- **Long10 reproduction.** Run as Long10, the training path reproduces A18 R_s1's logged loss at every update 0–136,
  including after unfreezing, with a largest difference of 2.0e-6. It reproduces A10 R_s0's first loss within 1.5e-8.
  The same step run twice is identical, with parameters within 2.4e-7 after unfreezing.
- **Training mechanics.**
  - Forced Interleaved = Repeated inputs reproduce loss and parameters.
  - The optimiser is bound, and the encoder is frozen and then unfrozen.
  - Resume (3 + 3 against 6) is within bounds.
- **Timing.** 0.68 s per update.

**Training-set readout** (`training_summary.json`). Both arms see the same targets. Interleaved's training CE is lower by
0.0095 (seed 0) and 0.0076 (seed 2), mostly early: −0.041 while the encoder is frozen, and −0.003/−0.006 over the last
270 updates. It gets 33 and 13 more of 16,200 windows right.

## Results

Multi segment F1@50. Development has four cells (2 seeds × 2 halves); the check pool has one cell per seed.

| System (one forward per window) | Development mean | s0 sel, s0 cal, s2 sel, s2 cal | Check s0 / s2 (mean) |
|---|---|---|---|
| Base (A08 + B3) | 0.9698 | 0.9563, 0.9827, 0.9564, 0.9837 | 0.9603 / 0.9555 (0.9579) |
| Long10 R@A08 | 0.9695 | 0.9559, 0.9827, 0.9567, 0.9828 | 0.9606 / 0.9578 (0.9592) |
| **Repeated@A08** | 0.9386 | 0.9077, 0.9538, 0.9290, 0.9639 | 0.9409 / 0.9466 (0.9438) |
| **Interleaved@A08** | 0.9392 | 0.8951, 0.9659, 0.9300, 0.9658 | 0.9432 / 0.9483 (0.9457) |
| A14 P | 0.9724 | 0.9640, 0.9855, 0.9564, 0.9837 | not evaluated |
| A20 H@A08 | 0.9724 | 0.9577, 0.9866, 0.9644, 0.9808 | not evaluated |

**Promotion rule: not met for either arm.**

| Part | Repeated@A08 | Interleaved@A08 | Required |
|---|---|---|---|
| Mean dF against Base | −0.0312 | −0.0306 | ≥ +0.003 |
| Half means (selection / calibration) | −0.0380 / −0.0243 | −0.0438 / −0.0173 | each ≥ −0.005 |
| Lowest cell | −0.0486 (s0 selection) | −0.0612 (s0 selection) | ≥ −0.015 |
| Native macro-F1 difference | −0.0069 | +0.0079 | ≥ −0.003 |
| Boundary recall difference | +0.0117 | +0.0110 | ≥ −0.010 |
| Missed short true segments | 17 against Base's 22 | 15 against 22 | ≤ Base + 3 |
| Calibration row feasible | fails both seeds (native 0.9723 / 0.9788) | both seeds | each seed |

So no seed 1 was trained. A22 closes with two seeds and no time-length, rate, insertion-count, teacher or input-branch
variant.

### 1. Do real intermediate frames beat repeated frames over the same support?

**Partly. They help recognition, but they do not improve full segmentation.**

| Interleaved − Repeated | s0 selection | s0 calibration | s2 selection | s2 calibration | Check s0 / s2 |
|---|---|---|---|---|---|
| Multi segment F1@50 | −0.0126 [−0.0260, −0.0003] | +0.0121 [0.0000, +0.0257] | +0.0010 | +0.0020 | +0.0022 / +0.0017 |
| Native macro-F1 | +0.0380 | +0.0139 | 0.0 | +0.0073 | +0.0050 / 0.0 |

- **Native recognition.** Native is better or equal in all six cells.
  - The development native mean is 0.9919 against Repeated's 0.9771.
  - On both seeds' selection halves, Interleaved reads no A4C clip as A5C. Base reads 2 on each, and Repeated 3 and 0.
  - Repeated fails the native feasibility line on both calibration halves; Interleaved does not.
  - Training CE is also lower.
- **Full segmentation.**
  - The four development cells have no consistent sign, with a mean of +0.0006. The seed-0 selection cell is worse and
    its interval excludes zero.
  - The check pool shows +0.0020, with intervals that include zero.
  - The own-calibration sensitivity readout favours Interleaved (+0.0145 development, −0.0004/+0.0149 check). But it is
    never used for the rule, and Repeated_s2's own choice was infeasible.
- **Windows.** Window-level results are close between the arms. On the check pool, A08-wrong windows fixed: Interleaved
  270 and 268, Repeated 262 and 264. A08-right windows broken: 166 and 200 against 162 and 198.
- **Reading.** The extra raw frames carry recognition-relevant information, but it does not reach full segmentation.
  That information could be appearance as well as motion, so it is not claimed as motion or cardiac phase.

### 2. Does the short window improve cross-view and short segments, and what does it cost?

**It improves the windows near view changes but destabilises the rest, and B3 turns that into many short false
segments.**

| Count (short in brackets) | Base | Long10 R@A08 | Repeated@A08 | Interleaved@A08 |
|---|---|---|---|---|
| Development missed true segments | 51 (22) | 41 (15) | 61 (17) | 58 (15) |
| Development false segments | 72 (39) | 84 (53) | 198 (153) | 199 (155) |
| Check missed true segments | 187 (79) | 160 (62) | 161 (42) | 150 (42) |
| Check false segments | 248 (153) | 263 (180) | 432 (342) | 422 (331) |

- **Gains.** Mixed-window accuracy rises.
  - Check s0: 0.934 against Long10 R's 0.919 and Base's 0.908.
  - Calibration s0: 0.951 against 0.934 and 0.913.
  - The short windows fix 30–42% of A08-wrong windows, against 8–21% for Long10 R. Short true segments missed on the
    check pool fall to 42 from 62 (R) and 79 (Base).
  - This fits V01's note that long cross-view context can mislead the target.
- **Cost.**
  - Pure-window accuracy falls (check 0.970–0.972 against 0.976–0.978 for Base and R), and many more A08-right
    windows break (check 162–200 against R's 20–33).
  - After B3 at A08's setting this becomes about four times Base's false short segments on development (153–155
    against 39) and about twice on check (331–342 against 153). Boundary precision falls to 0.909–0.912 on development,
    against 0.958 (R) and 0.966 (Base).
  - Even each model's own calibration choice leaves development at 0.941/0.955 (Repeated/Interleaved), against Base's
    0.970.
- **Native recognition.** Interleaved improves native recognition (above); Repeated harms it.

### 3. Does the full system beat A08, Long10 R and A14 P at the same parameters and one forward?

**No.** Both arms keep A08's 34,640,549 parameters and one forward per window, with 0.0162 s per window measured over
103,364 evaluation passes. But full segmentation is far below every reference:

| Comparison | Development mean | Check mean |
|---|---|---|
| Interleaved@A08 − Base | −0.0306 | −0.0122 |
| Repeated@A08 − Base | −0.0312 | −0.0141 |
| Interleaved@A08 − Long10 R@A08 | −0.0303 | −0.0135 |
| Repeated@A08 − Long10 R@A08 | −0.0309 | −0.0155 |
| Interleaved@A08 − A14 P | −0.0332 | – |
| Interleaved@A08 − A20 H@A08 | −0.0332 | – |

Long10 R@A08 itself is at Base (−0.0003 on development, +0.0013 on check), as in A18/A20.

## What this round shows

1. **Real intermediate frames help recognition in a short window.** Interleaved is better than or equal to Repeated in
   native recognition in every cell, and it has no A4C→A5C errors on either selection half. It does not beat Repeated in full segmentation
   (development mean +0.0006, with one significantly negative cell; check +0.0020).
2. **The 0.7 s support is the larger effect, and it is harmful.**
   - Both short-window arms improve mixed and A08-wrong windows and miss fewer short true segments on the check pool.
   - But they destabilise pure windows. At the fixed decoder they produce two to four times the false short segments
     and lose 0.03 of full segmentation against A08 + B3 and Long10 R.
3. **No full gain at equal cost.** Parameters and forwards equal A08's, but the full system is well below A08, Long10
   R, A14 P and A20 H.
4. **What is closed.** This short-support configuration is closed: 8 old samples over 0.7 s in 16 slots, with the
   long-window teacher, these batches and this schedule.
   - It does not show that motion or intermediate frames are useless. The recognition gain points the other way.
   - It does not show that dense frames fail over the full 1.5 s support, or without the long-window KL teacher.
   - It does not show that other temporal samplings fail.
   - Repeated was not better, so the insertion condition did add something, but only to recognition.

## Cost and files

- **GPU.** 1.65 of 4 h.
  - Checks: 0.04 h.
  - Four training runs: 1.02 h.
  - Calibration inference: 0.13 h.
  - Selection and check-pool inference: 0.46 h.
- **CPU.** 1.13 of 3 h at the final commit.
  - This includes preparation (0.27 h with its decode workers), the decode/evaluation workers outside GPU time
    (0.33 h), the training readout, and the agent's active wall clock booked by hand.
  - Waiting for background jobs is not booked. Loader workers inside GPU training jobs are not counted again
    (A18–A21's rule).
- **Records.**
  - `preparation.json`, `checks.json` and `manifests_index.json` (42ecf23).
  - `training_summary.json` and `train/` (705b59c).
  - `choices.json` (133515b).
  - `screen.json`: rows, segment counts, comparisons, accounting, window groups, input statistics and cost.
- **Frozen weights.** `models_frozen/video_temporal_density_v1/{Repeated,Interleaved}_s{0,2}` on the server (final.pt
  sha 015a62091be3, bec1fd156201, d9172a22958b and 6349682708f1).
- **Server only.** The per-arm manifests (frame mappings), raw frames, logits, per-sequence predictions and all
  identifiers. No frame was exported.
