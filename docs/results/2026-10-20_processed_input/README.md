# A12: the Pro group's processed EV9V input on the strong video model (2026-10-20)

**Question.** Does the Pro group's ready-made processing give the current strong video model a full-segmentation gain?
If switching the input shifts the distribution, does adapting to it under identical training help? Only the student's
input source changed. The development halves were reused for exploratory feedback; this is the same EV9V data, not a
holdout. The test split and EchoNet-Pediatric were not read. Configuration: `configs/experiments/video_processed_input_v1.yaml`
(bcc6197), fixed before any mapping, inference or training.

**The input, from the provider's own records (read only).**
- **Pipeline.** Each video got one FastSAM fan analysis, from 30 equally spaced frames over the whole recording. The
  output is a square crop with a 5% margin and padding, resized to 224 with INTER_AREA and encoded with libx264 CRF 18.
  Fps and duration were audited by the provider.
- **Offline, whole-recording input.** The crop region uses the whole recording, so the information budget is not that
  of an online 16-frame window.
- **Multiple layouts.** 223 of the videos we needed have more than one layout segment.
- **Labels.** The provider's manifest carries a view-label column. Whether any processing step used it is unknown; no
  label-dependent parameter appears in its configuration.

**Mapping (verified).**
- Our `video_path` equals the provider's `input_path`.
- **Coverage.** Of 3,364 training and 521 development videos, 3,360 and 520 have a Pro output. The other 4 and 1 have no
  provider row (these 5 are the provider's analysis failures inside our allowlist) and fall back to the original input,
  by rule.
- **Identity frame mapping.** The provider's frame count, our `n_frames` and the decoded count agree for every mapped
  video.
- **Spot check.** 60 frames (20 training videos at their first, middle and last frames) all correlate best with the
  original frame at offset 0, against the original cropped with the provider's own geometry (median NCC 0.79).
- **Pixels.** Pro frames take the same gamma LUT and normalisation, with no extra crop or mask. The original branch keeps
  V01's exact loader.
- **Common support.** It is essentially the full task: 541 of 541 selection sequences and 468 of 472 calibration
  sequences have no fallback window.

**Design.**
- **Stage one.** Frozen A08 all_s0 and all_s2 on Pro pixels. Primary: B3 at A08's committed parameters. Secondary: one
  calibration choice on the 40-setting grid.
- **Stage two.** Paired arms Raw and Pro (student on original or Pro pixels), from each seed's A08, with A10 R's recipe
  (CE + KL, 1,350 updates). The teacher is always on the original pixels, so Pro's KL is cross-input distillation.
- **Checks passed.** These covered pairing (identical teacher logits and RNG), gradients, the teacher receiving none,
  resume against a rerun, and source parity (0.0).
- **Raw reproduces A10 R.** Its segment F1 values equal A10 R's in all four cells (0.9566, 0.9846, 0.9606 and 0.9828).

## Results (seeds 0 and 2; four cells; each system on its stated input, B3 chosen once per model and input on calibration)

| System | Mean F | Mean native macro-F1 | Selection FP (short) | FN (missed short true) | A4C→A5C / A5C→A4C |
|---|---|---|---|---|---|
| A08 A, original input (reference) | 0.9698 | 0.9839 | 49 (22) | 51 (22) | 6 / 1 |
| Frozen Ordered, original | 0.9661 | – | 54 (29) | 62 (29) | – |
| Raw (trained, original) | 0.9712 | 0.9839 | 52 (30) | 43 (17) | 6 / 1 |
| A08 A on Pro, A08 setting (stage one) | 0.9524 | 0.9706 | 59 (20) | 93 (47) | 8 / 5 |
| A08 A on Pro, recalibrated | 0.9443 | – | – | – | – |
| Pro (trained, Pro input) | 0.9461 | 0.9754 | 109 (75) | 72 (27) | 8 / 3 |
| Raw model on Pro input (sensitivity) | 0.9416 | 0.9702 | 106 (53) | 88 (39) | 7 / 6 |
| Pro model on original input (sensitivity) | 0.9633 | 0.9858 | 70 (51) | 55 (26) | 6 / 0 |

**Per-seed paired differences** (multi segment F1@50, 95% intervals; two seeds, so no verdict).
- **Trained Pro − trained Raw.** Below zero in all four cells:
  - selection −0.018 [−0.036, −0.001] and −0.055 [−0.082, −0.030];
  - calibration −0.014 [−0.026, −0.005] and −0.013 [−0.026, −0.004].
- **Stage one, A08 on Pro at A08's setting − A08.** −0.017 [−0.036, 0.001], −0.015, −0.024 and −0.013; the last three
  intervals lie below zero.
- **Pro model on the original − on Pro.** +0.021, +0.008, +0.029 and +0.009. The two selection intervals lie above zero; the two calibration intervals include it.
- **Recalibrating B3 does not rescue the Pro input.** In seed 2 no calibration setting met the native and recall floors,
  for frozen A08 on Pro, the Raw model on Pro and the Pro model alike.
- **A5C.** Native A5C recall falls on Pro input. In seed 0 on selection it goes from 1.00 to 0.82 for frozen A08 and to
  0.91 for the trained Pro model.

**Fixed rules: neither met**, so A12 closes with two seeds.
- **Third-seed rule.** Pro − Raw is −0.025 and Pro − A08 A is −0.024, with selection −0.034 against A08 A. Native falls by
  0.009, and missed short true segments are 27 against 22.
- **Stage-one rule.** Mean −0.017, native −0.013, boundary recall −0.022, missed short true segments 47 against 22.

## What this round shows

1. **The input change was tested, not left open.** Mapping, timing and coverage were verified, and the common support
   is essentially the full task. So the result is not an artefact of time alignment or of the fallback.
2. **There is no immediate gain.** The strong model is sensitive to this input: segment F1, boundary recall and A5C
   recognition fall, and many more short true segments are missed.
3. **Adaptation under this short recipe did not help.**
   - The Pro-trained model stays below the original-input systems.
   - It still works better on the original input, which suggests the cross-input KL to the original-input teacher and
     the short schedule kept it an original-input model.
   - Whether a longer adaptation, or one without the original-input teacher, would help is **not tested** here. This
     closes this short configuration only and does not show the Pro method is useless.
4. **The strong reference stands.** Raw reproduces A10 R exactly, and A08 A remains the strongest model.

## Cost and files

- **GPU.** 1.979 h of 6: checks 0.016, four runs 1.296, evaluation 0.535 and a booking correction of 0.133 for a failed
  evaluation attempt. Two evaluator faults (the fallback frame paths and the run-record test) were fixed before any
  result was read.
- **CPU.** Booked in `runs/video_processed_input_v1/ledger.jsonl`, including decoding worker time and the agent's active
  wall clock.
- **Frozen models.** `models_frozen/video_processed_input_v1/{Raw,Pro}_s{0,2}`.
- **Records.** `preparation.json` (mapping, fallback, spot check, cache), `checks.json` and `screen.json` (rows,
  common-support rows, counts, choices, comparisons and both rules).
- **Server only.** The Pro frame cache (21 GB) and every per-video mapping stay on the server.
