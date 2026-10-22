# A21: does extra spatial context beyond the centre crop improve full segmentation? (2026-10-22)

**Question.** The video models see a 224×224 centre crop of each frame. The crop is taken after the short side is resized
to 256 (`crop_uint8`). For the 320×240 recordings that is 57.48% of the frame. Keeping A08's original centre view, does
also showing the whole frame reduce recognition errors on high-loss windows? And does it give a full-segmentation gain
that keeps short true segments without adding short false ones?
- Full against Restricted isolates the extra visible area. Both have the same parameters, forwards, loss, teacher,
  batches, random layout and schedule. Only Full's second view can see pixels outside the old centre crop.
- Full/Restricted against each seed's A08 + B3 (Base) is the full-system test. A10 R is background, never the matched
  control.
- This is a conventional dual-view / residual-fusion test. No algorithmic contribution is claimed.
- Each window costs two encoder forwards at deployment, so no equal-compute or one-forward claim is made against A08.
- The extra pixels may be background, black borders or screen text. Any Full gain is first attributed to the extra
  visible RGB area, not to named anatomy.
- The check pool's sessions are unseen by A21's batches, but its recordings trained the A08 encoder, so it is not a new
  holdout. The development halves are reused exploratory feedback. The test split and EchoNet-Pediatric were not read.
- Configuration: `configs/experiments/video_spatial_context_v1.yaml` (5f995a3), fixed before any A21 output.

**How this differs from earlier work.**
- R13 added a target-logit residual on E1 image embeddings with no new spatial input. V02 and A02–A04 changed readouts
  or time-token supervision of the same video tokens.
- R15 and A12 replaced the input (fan crop, Pro processing). A12's short Pro adaptation had no gain. That showed
  neither that the crop drops cardiac structure nor that KL limited adaptation.
- A21 keeps the original centre branch and adds the whole frame beside it, with the original labels and no Pro or
  supplier algorithm.

**Design.**
- **Model.** Per seed, A08 all_s{k} is strictly loaded into the unchanged TargetMViT. Both views pass the same encoder
  and marker stem, one set of parameters registered once.
  - e_crop = embed(crop, marker), then e_aux = embed(aux, marker).
  - z = classifier(e_crop) + g([e_crop, e_aux]), with g = Linear(1024, 32) → GELU → Linear(32, 5).
  - g's first layer is drawn N(0, 0.02²) from a generator seeded 2100 + seed. Its last layer is all zero. So at the
    start the model's eval logits equal A08's exactly.
- **Pixels.** Each frame is decoded from V01's original JPEG path with the original gamma LUT. It is then resized as in
  `crop_uint8` (short side 256, PIL BILINEAR), giving the intermediate image R.
  - The crop is R's exact centre 224×224 and equals the old `load_frame_u8` bit for bit.
  - Full's aux is R fitted into 224×224: long side 224, PIL BILINEAR, centred on an RGB(29,28,29) canvas.
  - Restricted's aux first fills every pixel of R outside the centre rectangle with RGB(29,28,29), then applies the same
    fit. Masking happens before the resize, so no outside pixel leaks through interpolation.
- **Data and teacher.** A10's pinned 12-window batches (VSH rows 0–1,349) are used unchanged in both arms. The teacher
  is each seed's frozen A08 on the crop, from A18's verified cache.
- **Loss and recipe.**
  - Mean CE + mean KL(own A08 ‖ model), weight 1, temperature 1.
  - 1,350 updates and AdamW. The encoder and its 512-d projection run at 1e-5 and are frozen for the first 135 updates.
    The marker stem and classifier use 1e-4, and g uses 1e-3.
  - Weight decay 1e-4 and clip 1, with A10's warm-up and cosine schedule.
  - Both arms use the same seeds. The crop forward runs before the aux forward.
- **Order.**
  - The weights and run records were committed (8a94846) before any evaluation.
  - The sensitivity choices were committed (1822d84; the git object's sha was verified) before the selection half or the
    check pool was read.
  - The check pool was evaluated once.
- **Evaluation.**
  - The primary readout decodes each model with its seed's committed A08 B3 numeric setting (Full@A08,
    Restricted@A08).
  - Sensitivity readouts are one own-calibration choice per model from the standard 40 settings, and plain argmax.
  - Old A08, frozen Ordered, A10 R, A14 P and A20 U/H development rows reproduced. A19's A08 check rows reproduced.

**What the two views contained** (`preparation.json`, training manifests only, every fifth update of each seed).
- **Sizes.** Every sampled request comes from a 320×240 source (21,173–21,544 requests and 1,104–1,130 recordings per
  seed). Every R is 341×256.
- **Area.** The extra area outside the centre crop is 42.5% of R for every request.
- **Difference between arms.** Full and Restricted aux views differ in 32.5% of their 224×224 pixels. No request and
  no window has identical aux views, so the preparation condition was met.
- **Content.** About 77% of the extra-region pixels are near black (intensity at most 10). The per-frame pixel standard
  deviation there is about 53, so the region varies. This is a pixel description, not evidence of useful anatomy.

**Checks passed before training** (`checks.json`).
- **Synthetic geometry.** Seven R shapes, including odd padding (16/17, 17/18, 0/1, 40/41) and a square R, were tested.
  When everything outside the centre 224 was changed, the crop and the Restricted aux stayed identical bit for bit and
  the Full aux changed. Padding is the fill colour.
- **Frame handling.** Mixed-size windows are processed frame by frame, and gamma is applied before the resize.
- **Real requests.** 768 real requests give crops bit-identical to `load_frame_u8`, and the window digests equal A10's.
- **Start state.** At the start both arms' eval logits equal A08's exactly (difference 0.0). The cached teacher equals
  online A08 exactly over 72 windows; the teacher is unchanged and has no gradient.
- **Cross-arm RNG.** In train mode the crop logits and the CPU/CUDA random states are identical across arms after both
  forwards, while the aux inputs differ.
- **Gradients.**
  - At update 0, g's last layer receives gradient, and its first layer and the aux input receive exactly 0. This is
    expected with a zero last layer, and the encoder is frozen.
  - At update 1, g's first layer and the aux input receive gradient.
  - After the warm-up, the encoder receives gradient.
- **Sharing.** 34,673,514 parameters (A08's 34,640,549 plus g's 32,965), each registered once in the optimiser.
- **Reproducibility.** Forced Full = Restricted inputs reproduce loss and parameters exactly. Resuming after 3 updates
  matches 6 straight updates: loss within 1.9e-7, and parameters within 1.9e-5 against a repeat difference of 1.8e-5.
- **Timing.** 1.34 s per update, with a 35.8 GiB peak.

**Training-set readout** (`training_summary.json`, training windows only).
- Both arms see the same 12 windows every update. Their per-update training CE differs by −0.00017 (seed 0) and
  +0.00026 (seed 2) on average.
- Correct counts differ by 0 and −3 of 16,200 windows. The extra area did not help the model fit its own training
  windows.

## Results

Multi segment F1@50. Development has four cells (2 seeds × 2 halves); the check pool has one cell per seed.

| System | Development mean | s0 sel, s0 cal, s2 sel, s2 cal | Check s0 / s2 (mean) |
|---|---|---|---|
| Base (A08 + B3) | 0.9698 | 0.9563, 0.9827, 0.9564, 0.9837 | 0.9603 / 0.9555 (0.9579) |
| **Full@A08** | 0.9693 | 0.9587, 0.9856, 0.9569, 0.9761 | 0.9586 / 0.9595 (0.9590) |
| **Restricted@A08** | 0.9696 | 0.9607, 0.9846, 0.9569, 0.9761 | 0.9578 / 0.9594 (0.9586) |
| A20 H@A08 | 0.9724 | 0.9577, 0.9866, 0.9644, 0.9808 | not evaluated |
| A14 P | 0.9724 | 0.9640, 0.9855, 0.9564, 0.9837 | not evaluated |
| R + B3 (its own committed choice) | 0.9711 | 0.9566, 0.9846, 0.9606, 0.9828 | not evaluated |
| frozen Ordered + B3 | 0.9661 | 0.9385, 0.9817, 0.9644, 0.9797 | not evaluated |

**Promotion rule: not met for either arm.**

| Part | Full@A08 | Restricted@A08 | Required |
|---|---|---|---|
| Mean dF against Base | −0.0005 | −0.0002 | ≥ +0.003 |
| Half means (selection / calibration) | +0.0014 / −0.0023 | +0.0024 / −0.0028 | each ≥ −0.005 |
| Lowest cell | −0.0076 (s2 calibration) | −0.0076 (s2 calibration) | ≥ −0.015 |
| Native macro-F1 difference | −0.0017 | −0.0017 | ≥ −0.003 |
| Boundary recall difference | +0.0072 | +0.0065 | ≥ −0.010 |
| Missed short true segments | 13 against Base's 22 | 13 against 22 | ≤ Base + 3 |
| Calibration row feasible | s2 fails (native 0.9788 < 0.980) | s2 fails (native 0.9788) | each seed |

So no seed 1 was trained. A21 closes with two seeds and no input, crop, head, teacher or schedule variant.

### 1. Does the extra area carry transferable information?

**No evidence that it does.** Full − Restricted is the only comparison that isolates the extra visible area.

| Full − Restricted | s0 selection | s0 calibration | s2 selection | s2 calibration | Check s0 / s2 |
|---|---|---|---|---|---|
| Multi segment F1@50 | −0.0020 [−0.0061, 0.0000] | +0.0010 [0.0000, +0.0030] | 0.0 | 0.0 | +0.0007 / +0.0002 |

- The development mean is −0.0002 and the check mean +0.0005. Native macro-F1 and boundary recall are equal or nearly
  equal in every cell.
- On seed 2, the two arms' development segments are identical. On seed 0's development halves they differ by at most
  three segments in any accounting category.
- **Windows.** On the check pool, A08-wrong windows fixed: Full 121 and 129, Restricted 120 and 131. A08-right windows
  broken: 36 and 23 against 38 and 22. On the check pool, mixed- and short-window accuracy differ by at most 0.0006.
- **Development windows.** Full's mixed-window accuracy is 0.0007–0.0014 higher in three cells and equal in the
  fourth. Short-run accuracy differs by −0.0039 to +0.0020. These window differences do not become segment
  differences.
- **Training set.** As above, the matched training CE is the same.

### 2. Does Full beat the same-compute Restricted?

**No.** Full and Restricted have the same parameters and the same two forwards per window. Their full-system results
are within 0.0020 of each other in every cell, with no consistent sign across seeds or halves.

### 3. Is there a full gain that both recovers missed true segments and removes short false ones?

**No. Both arms keep more true segments but add false short ones, as in A20.**

| Count (short in brackets) | Base | Full@A08 | Restricted@A08 | A20 H@A08 | A14 P |
|---|---|---|---|---|---|
| Development missed true segments | 51 (22) | 40 (13) | 40 (13) | 38 (13) | 52 (23) |
| Development false segments | 72 (39) | 86 (55) | 85 (55) | 75 (47) | 60 (30) |
| Check missed true segments | 187 (79) | 158 (58) | 158 (58) | – | – |
| Check false segments | 248 (153) | 267 (183) | 272 (186) | – | – |

- **Recall improves.** Boundary recall is higher: development 0.9864/0.9858 against 0.9792, check +0.007 to +0.010.
- **Precision falls.** Boundary precision is lower: development 0.9581 against 0.9662, check −0.002 to −0.006. False
  short segments rise by 16 on development and by 30–33 on the check pool.
- **Net.** Development −0.0005 (Full) and −0.0002 (Restricted); check +0.0012 and +0.0007 (s0 −0.0018/−0.0025, s2
  +0.0041/+0.0039).
- **Seed 2 calibration is worse.** Both arms lose 0.0076 [−0.0164, −0.0009], an interval that excludes zero, with
  native 0.9788 (one more A4C read as A5C).
- **Against references.**
  - Both arms are below A20 H@A08 (−0.0031/−0.0028) and A14 P (−0.0031/−0.0028) over the same four cells.
  - They are below A10 R + B3 (−0.0018/−0.0016).
  - The shared change from Base is a common dual-view / residual / continuation effect, not an effect of the extra
    area.

### 4. Would a gain pay for the two forwards?

**There is no gain to pay for.**

| Cost (L40S, batch 32, eval) | Seconds per window | Peak memory |
|---|---|---|
| A08, one CLS forward | 0.0153 | 8.7 GiB |
| Full or Restricted, two encoder forwards (benchmark) | 0.0306 | 9.0 GiB |
| Full or Restricted, measured over 103,364 evaluation window passes | 0.0322 | 9.3 GiB |
| A18's three-model average (three A08 forwards, estimated) | 0.046 | – |

- Each two-view model has only 32,965 parameters more than A08 (0.1%), but it costs twice A08's encoder compute per
  window at deployment.
- Training took 1.34 s per update, against 0.71 s for A10's single-view recipe (A18's timing), with a 35.8 GiB peak.

**Sensitivity readouts** (never used for the rule).
- Own-calibration choices were the same setting for both arms of each seed: half window 7, quantile 0.999 for s0 and
  0.995 for s2.
- On seed 2 that choice cuts too often: selection 0.9300 with 56 false segments (43 short), and check 0.9548/0.9552.
- Full own-cal − Restricted own-cal is within ±0.003 everywhere. Argmax readouts are equal across arms except on s2
  calibration (0.9733 against 0.9752).

## What this round shows

1. **The extra area carries no information this design can use.** The whole frame shows the 42.5% of R that the
   centre crop leaves out, most of it near black. Full and Restricted are indistinguishable on the training set, the development
   halves and the check pool.
2. **Full does not beat the same-compute Restricted.** Every difference is within 0.002 and has no consistent sign.
3. **No full gain.** Both two-view models recall more true segments and short true segments. But they add more false
   short segments than A20's replay did, and they lose 0.0076 on seed-2 calibration. The system is at Base (−0.0005 /
   −0.0002 on development) and below A20 H and A14 P.
4. **Nothing to pay two forwards for.** The models double A08's encoder compute per window and give no gain.
5. **What is closed.** This dual-view residual configuration, with whole-frame auxiliary context at 224 and these
   batches, teacher and schedule, is closed.
   - It does not show that the image periphery never matters.
   - It does not show that the centre crop loses nothing for any model, resolution or task.
   - It does not show that other spatial designs (higher resolution, a learned ROI, or other fusion) would fail.
   - Any gain here would first have been attributed to the extra RGB area, not to anatomy. With no Full − Restricted
     difference, nothing is attributed.

## Cost and files

- **GPU.** 3.20 of 6 h.
  - Checks: 0.02 h.
  - Four training runs: 2.01 h.
  - Calibration inference: 0.25 h.
  - Selection and check-pool inference: 0.92 h.
- **CPU.** 1.15 of 4 h at the final commit.
  - This includes preparation (0.24 h with its decode workers), the decode/evaluation workers outside GPU time
    (0.35 h), the training readout, and the agent's active wall clock booked by hand.
  - Waiting for background jobs is not booked. Loader workers inside GPU training jobs are not counted again
    (A18–A20's rule).
  - A first preparation launch stopped before reading anything (missing configuration copy on the server) and wrote no
    ledger line.
- **Records.**
  - `preparation.json` and `checks.json` (afc5f60).
  - `training_summary.json` and `train/` (run records and logs), committed before calibration (8a94846).
  - `choices.json` (1822d84).
  - `screen.json`: rows, segment counts, paired comparisons, segment accounting, window groups, cost and the promotion
    rule.
- **Frozen weights.** `models_frozen/video_spatial_context_v1/{Full,Restricted}_s{0,2}` on the server (final.pt sha
  4fe3ea991c41, 9d45dec65ed1, 2e585fb657d4 and 527ee3ab7468).
- **Server only.** Raw frames, logits, per-sequence predictions and all identifiers. No frame was exported.
