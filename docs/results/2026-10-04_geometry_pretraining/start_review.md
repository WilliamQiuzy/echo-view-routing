# Geometry pretraining: start review (2026-10-04)

This review checks every start condition in `configs/experiments/geometry_pretraining_v1.yaml` before the G1 and G2
runs. On 2026-10-04 the project lead asked for five corrections and approved the launch once they passed their checks.
The corrections come first, then each start condition with its evidence, the model checks, the pair banks and the
contrast baseline. The round was launched at 05:40 UTC on 2026-10-04. No test-split data were used, and the paper stays
paused.

## The five corrections

1. **v14 and the aligned comparison.** The first check compared classifier scores of two different images. The
   pipeline rendered its fixed grid, and EchoSlicer rendered its own grid, extent, rotation and flips. The comparison
   was repeated at the same time point, with the same physical sample points, display orientation and grey processing,
   on raw images (`scripts/geometry_aligned_check.py`). Coordinates and pixels agree in all 18 volumes (next section).
   Correctness is now judged on coordinate and pixel agreement under aligned conditions. The classifier comparison is
   an auxiliary check made on aligned images. This one revision applies to every volume and is recorded in the config's
   `amendments`. v14's failure under the first rule stays on record. v14 was not passed because its score was higher.
2. **Extractor tolerance.** The tolerance is now 1e-4 on identical input tensors with the same weights in fp32. The
   new extractor meets it (9.5e-6). Its full pipeline differs from the cached E1 embeddings by up to 1.5e-4, and the
   cause was traced to GPU summation order (section on the extractor). The existing probe and decoding settings were
   reused without retuning and reproduce E1's predictions and metrics exactly.
3. **The question and the training details.** The question now reads: beyond same-plane temporal consistency, does
   adding regression of the tilt difference improve real 2D tasks? G2 − G1 tests the extra tilt supervision. G1 already
   uses one geometric relation, so G1 − G0 is not only an effect of more data. Sampling now draws a person first and
   then a volume. The InfoNCE masks are explicit: an anchor's partner appears once in the denominator, as its
   positive, and is never counted again as a negative. The first round has no flips, rotations or intensity changes.
   The tilt term alone sends gradients to every trainable encoder tensor.
4. **Contrast baseline on the model's input.** It is computed on the raw slices and on the final input (after
   standardisation, clipping, resizing and cropping), next to the zero predictor. On the final input it explains more
   of the tilt than on the raw slices (section on the baseline).
5. **Research signal and final goal.** The four interpretation rules screen for a research signal. "No consistent
   difference detected" is not evidence of no loss. The report gives G2 against G1 and G0 for classification and
   segmentation, every system against the native floors (0.973 and 0.980) and boundary recall ≥ 0.95, and every system
   against B3, ASRF and the strongest previous configuration. The final goal is the success test of the earlier rounds:
   a three-seed mean segment F1@50 at least 1 point above the strongest previous configuration on each half, with the
   thresholds met in every seed and half. The terms the report evaluates were written into the config before training.

## Start conditions

| # | Condition | Evidence | Holds |
|---|---|---|---|
| 1 | STACOM decodes, sectors fitted, timing recorded, one probe frame, inclusion list | All 15 volunteers decode, and the DICOM equals the VTK voxel for voxel. Sector fits have R² ≥ 0.9998. Volume rates are 17.8 to 21.1 per second. One probe-frame convention covers all, and all 15 are included (`docs/results/2026-10-03_stacom_verification/`). | yes |
| 2 | EchoSlicer volumes load, exclusions listed | All 29 load as spherical volumes. D_0 (median coverage 0%) and D_3 (10%) are excluded, and 27 remain. | yes |
| 3 | Unit tests; (a) EchoSlicer planes; (b) two STACOM paths; (c) masks inside the volume | 39 geometry tests pass. (a) Under the revised rule, sample points lie within 6e-14 mm of the pipeline's plane, pixel r is 0.994 to 1.000 and the best shift is 0 in 18 of 18; aligned classifier scores differ by at most 0.037. (b) r ≥ 0.998 and shift 0 in 15 of 15. (c) No mask pixel fell outside at random tilts, and the bank renderer stops on any outside pixel. | yes; (a) under the revised rule |
| 4 | Extractor within 1e-4 on identical tensors; full pipeline traced | 9.5e-6 on identical tensors. The full pipeline differs by up to 1.48e-4 (orig) and 1.14e-4 (gamma 0.9) over 238,572 frames each, traced to batch-shape summation order. | yes |
| 5 | Existing probe and decoding reproduce E1 without retuning | None of 477,144 probe predictions changes. The stored ASRF checkpoints decode every frame of both halves identically for seeds 0 to 2. All 111 B0 and B3 tables are identical. | yes |
| 6 | Pair banks with digests; held-out people fixed | The held-out bank and the seed-0 bank are rendered with digests. Seeds 1 and 2 render while seed 0 trains, and the round script starts a seed only after its bank's digest file exists. | yes for held-out and seed 0; checked per seed at use |
| 7 | One GPU run at a time, within 20 GB | The round script runs one GPU job at a time. Training peaked at 1.5 GB in the smoke run, and the device showed 2.2 GB during the first formal run. GPU memory is logged every 30 s (`logs/geometry_round_gpu.csv`). | yes |
| 8 | Approval | The project lead's decision of 2026-10-04. | yes |

## The extractor difference

The new extractor runs E1 in the main environment. On identical input tensors it matches the original encoder within
9.5e-6. From frames, its embeddings differ from the float32 raw cache by up to 1.48e-4 (orig) and 1.14e-4 (gamma 0.9).
The cause was traced in three steps (`scripts/encoder_tensor_parity.py`).
- Both environments crop the frames identically, since both use Pillow 12.3.0.
- On the 24 frames with the largest differences, the new and original models differ by 3.3e-5.
- The original model, re-run on those frames in a 24-frame batch, differs from the cache by 1.26e-4. Run alone, it
  differs by at most 2.1e-5. Inside the 256-frame batch of the original extraction, it reproduces the cache exactly.

The residual is therefore the order of GPU summation, which depends on the batch shape. It changes no prediction
(`scripts/geometry_g0_reproduce.py`).

## Model checks

- **Gradients.** On the first batch, the tilt term alone gives finite, non-zero gradients in all 15 trainable encoder
  tensors (smallest norm 1.0e-3), and so does InfoNCE alone (smallest norm 2.8e-2). No frozen tensor receives a
  gradient.
- **Identical batches.** In the smoke runs, G1 and G2 read bit-identical batches. In the formal round, the script
  compares the batch digests of G1 and G2 for each seed and stops if they differ.
- **G1's tilt term.** G1 computes the tilt loss only for its log, outside the graph. Its tilt head is neither trained
  nor shrunk by weight decay.
- **Precision.** TF32 is off in every process.

## Pair banks

Each bank stores 224 × 224 grey slices in their final form, the pair table and digests (`data/geometry/banks/`, server
only).

| Bank | Pairs | People | Pairs per person | same_plane / same_time / both | u8 digest (first 12) |
|---|---:|---|---|---|---|
| held-out (seed 100) | 4,000 | EchoSlicer C; STACOM v14, v15, v16 | 963 to 1,067 | 1,349 / 1,262 / 1,389 | 4a12cc304193 |
| seed 0 | 96,000 | 15 pretraining people | 6,313 to 6,539 | 31,931 / 31,828 / 32,241 | 2ce80ecacbdc |
| seed 1, seed 2 | 96,000 each | the same 15 | rendering at launch | | |

EchoSlicer person A's 15 volumes count as one person, so A contributes as many pairs as one STACOM volunteer.

## Contrast baseline

A ridge regression (λ = 1, fixed in advance) predicts d_alpha from each slice's mean, SD, sharpness and relative
sharpness and their differences. It is fitted on the seed-0 bank and scored on the held-out bank
(`runs/geometry/contrast_baseline.json`).

| Input | Signed d_alpha, MAE (R²) | Moving planes only, MAE (R²) | Tilt size, MAE (R²) |
|---|---|---|---|
| zero predictor | 6.59° (0.00) | 9.95° (0.00) | 6.59° |
| raw slices | 6.51° (0.07) | 9.41° (0.07) | 6.24° (0.00) |
| final input | 5.85° (0.21) | 8.26° (0.22) | 6.29° (0.00) |

The final input carries more of the signed tilt than the raw slices. Most of it comes from the slice mean. Fitted
alone, the mean explains R² 0.10 on the final input and 0.05 on the raw slices. Its correlation with the slice's own
tilt is −0.25 on the final input and −0.08 on the raw slices. Standardisation sets the mean and SD over the whole mask
before two later steps. Clipping at 0 removes the darkest values, and the centre crop keeps about the middle 82% of
the width and 88% of the depth. After them, the mean (61.3 ± 3.3 grey levels across held-out slices) and SD
(63.1 ± 2.6) follow the content near the centre of the slice. This check does not separate anatomy entering the plane from
acquisition effects.

The baseline is a control, not a start condition, so the round was not stopped. It sets the shortcut warning fixed in
the config: in any seed where the baseline's error on the final input (5.85°) is at most 1.1 times that of G2's tilt
head, the G2 result carries the warning. The head must therefore reach an error below 5.32° to avoid it.

## One more revision before launch

The report's self-test on G0 found that the AUC of clip-mean margins is at its ceiling for G0 (1.000 on selection,
0.998 on calibration). Each half has only 11 and 18 native A5C clips, and E1 already ranks them all above, or nearly
all above, the native A4C clips. Rule 2 could not detect a gain on that measure. Before any G1 or G2 result existed,
the separation used by the rules was set to the AUC over the clips' cached samples, which uses the same margin
before averaging. For G0 it is 0.915 on selection and 0.882 on calibration. The clip-mean AUC is still reported. The
revision applies to every arm and is in the config's `amendments`.

## What runs now

- G1 and G2 for seeds 0, 1 and 2, each 3,000 steps of 32 pairs, final weights (`scripts/geometry_round_run.sh`).
- For each encoder, the embeddings are extracted and the E1-recipe probe is trained. ASRF then runs with the same seed
  number, and so do B0 to B3.
- The report is `scripts/geometry_round_report.py`. Its self-test replaces G1 and G2 by G0. It reproduces the round-2
  numbers (ASRF 0.9087 and 0.9682, B3 0.8940 and 0.9589, strongest previous 0.9344 and 0.9721), and every arm
  difference is zero.
- A failed step stops the round at that step, as does a failed downstream job or a batch mismatch between G1 and G2.
  The log is `logs/geometry_round.log`.
