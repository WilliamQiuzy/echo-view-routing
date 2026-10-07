# Fan224 visual adaptation: full-sector input and ordinary adaptation, four arms x three seeds (2026-10-06)

This round asked whether a full-sector 224 x 224 input and ordinary visual adaptation improve the A4C/A5C evidence
that reaches B3. A 2 x 2 design separates the input from the training scope:

| Arm | Input | Trained |
|---|---|---|
| L0 | legacy E1 input (short side 256, centre crop 224) | five-class head |
| L1 | legacy | head + ViT block 11, ln_post, proj |
| F0 | fan224_v1 (fixed sector template, padded to 224) | five-class head |
| F1 | fan224_v1 | head + ViT block 11, ln_post, proj |

F1 was the pre-specified candidate. The plan is `docs/superpowers/plans/2026-10-06-fan224-visual-adaptation.md`.
The input contract, configuration and code were committed before training (f5ecadb). Only EV9V training recordings
entered training, and the two validation halves were read only after all twelve runs had ended. The test split and
EchoNet-Pediatric were not read, and the public labels are unchanged.

## Answers

1. **What was done.**
   - **Input.** A fixed full-sector template (fan224_v1) was fitted on training images and frozen after a QC of 100
     training recordings. All 415,228 training frames were exported as PNG (section 1).
   - **Training.** Twelve runs, L0/L1/F0/F1 x seeds 0, 1, 2. Every run starts from the frozen E1 encoder and the E1
     probe. The four arms of a seed read one saved batch plan, and each run has 1,500 updates (section 2).
   - **Evaluation.** The per-sample head was read on both validation halves, as argmax and with B3. The contrasts use
     the paired bootstrap. The unadapted E1 head on each input is reported as an input-sensitivity control.
2. **Where the changes come from.**
   - **The fan input lowers the full system in every seed and half.**
     - F0 − L0 is −2.1 to −2.8 points of segment F1@50 on the selection half and −3.4 to −4.1 on the calibration
       half.
     - F1 − L1 is −4.0 to −4.2 and −3.8 to −5.1.
     - All twelve intervals lie below zero.
     - No F model reaches the calibration native floor of 0.980 under any of the 40 B3 settings, so B3 has no
       feasible setting for any F model.
   - **Last-block fine-tuning shows a positive trend on the legacy input only.**
     - L1 − L0 is +1.0 to +2.0 points on the selection half and +0.6 to +1.1 on the calibration half. Only one of the
       six intervals lies above zero, so under the round's rule this is not a consistent improvement.
     - On the fan input, F1 − F0 shows no consistent difference (−0.3 to +0.7 points).
   - **The fine-tuning trend does not carry over to the fan input.** The interaction (F1 − F0) − (L1 − L0) is
     negative in all six cells (−0.3 to −2.1 points), and two intervals lie below zero.
   - **A4C and A5C.** With B3, the fan input loses A5C clips:
     - F1 reads 9, 9 and 10 of 11 A5C clips on the selection half, against 11 of 11 for L1 in every seed.
     - It reads 16 of 18 on the calibration half in every seed, against 17, 18 and 18 for L1.
     - It also misreads 2 to 4 of 61 A4C clips on the calibration half, against 1 or 2 for L1.
3. **Which metrics still miss the goal.** All of them for F1.
   - F1 + B3 averages 0.868 segment F1@50 on the selection half and 0.925 on the calibration half; the goal requires
     0.944 and 0.982.
   - No F1 seed has a feasible B3 setting.
   - F1's native macro-F1 with B3 is 0.966, 0.966 and 0.978 on the selection half (floor 0.973) and 0.969, 0.969 and
     0.956 on the calibration half (floor 0.980).
   - The best arm, L1 + B3, averages 0.909 / 0.969 and also misses the goal. Its seed 1 selection boundary recall is
     0.937, below 0.95.
   - No arm beats E0 + B3 on the selection half (0.934) or B7 + B3 on the calibration half (0.972).

Under the plan, the main question has a negative answer. F1 is better than neither L1 nor F0, and the candidate shows
no gain. The round stops here. The fan224_v1 input interface is kept. No learning-rate scan, larger trainable part,
temporal module or retrained ASRF was started.

L1's positive trend is a local result of ordinary fine-tuning on the old input; it was not the pre-specified
candidate. Whether it merits matched-feature ASRF or another test is the next decision, not part of this round.

## Terms

- **Points** are hundredths of a metric; −2.1 points of segment F1@50 means −0.021.
- **Argmax rows** label each 10 Hz sample with its most probable view, without a decoder.
- **B3 rows** run the change-point decoder on the same probabilities. The setting is chosen per model on the
  calibration half only: native macro-F1 at least 0.980 and boundary recall at least 0.95, then the highest segment
  F1@50. It is applied once to the selection half. When no setting qualifies, the fixed setting 9 / 0.95 is shown as a
  diagnostic and passes nothing.
- **Native clips** are whole single-view recordings. A clip is read correctly when the majority of its samples is.
- **Cuts per clip** count additional label switches within clips assigned a single reference label.
- **Consistent** means that all three seeds' 95% intervals exclude zero in the same direction on a half. An interval
  that contains zero does not mean the systems are equal.

## 1. The fan224_v1 input (Task 1)

**QC list and size scan.**
- The QC list holds 100 training recordings, 20 per class. Thirty are the pilot's recordings, and 14 per class were
  added from the remaining training recordings with seed 20261007.
- Every frame file of all 3,364 training recordings (619,524 frames) is 320 x 240 and not empty. The first, middle and
  last frame of every recording is not blank.

**Template.** One fixed sector for every recording and frame, fitted on the QC training images.
- The apex and both straight edges follow the region that changes over time in at least half of the QC recordings:
  apex (156.7, 31.2), left half-angle 42.5°.
- The right edge sits at 44.0°, 1.8° inside that region, to drop the depth ticks and focus marker drawn over it.
- The radius is 200 px. It stops 3–4 px short of the image's lower arc, which drops the depth-end tick at its centre.
- x ≤ 275 drops the vertical interface lines and the focal-zone marker at the lower right.
- The bounding box (254 x 200 px) is resized by 0.882 to 224 x 176 and padded with black to 224 x 224.
- The pilot template had apex (157, 26), ±46.5°, radius 207 and scale 0.749. Its sector edges carried depth ticks in
  nearly every recording.

**QC metrics** (all frames of each QC recording).
- **Constant bright pixels inside the template** beyond 25 px from the apex are pixels that stay bright and nearly
  unchanged over the whole recording, mostly drawn interface marks. They remain in 18 of 100 recordings, 95 px in
  total and at most 27 in one recording. The pilot template left 4,020 px in 98 recordings.
- **Changing pixels dropped.** Of the pixels inside the pilot sector that change over time, the template drops a
  median of 1.9% per recording (90th percentile 3.6%, maximum 4.4%). They lie in thin strips along the right edge,
  near the apex and in the lower right corner.

**Visual review.** The first, middle and last frame of all 100 QC recordings was viewed with the template outline,
next to the fan224 output.
- No obvious truncation of the imaged anatomy was found.
- Faint depth ticks remain inside the left edge in a few recordings.
- Doppler or M-mode cursor lines drawn inside the sector appear in frames of six recordings. The legacy crop contains
  them too.
- Bright near-field echoes at the apex are image content and were kept.
- One recording first looked like a steered sector; its activity outside the template edges was below the median, so
  it is not a different layout.

**Export.** Every training-split 10 Hz sample (207,614) was written in both variants (orig, gamma090) as 224 x 224 RGB
PNG: 415,228 frames, 13 GB, server only.
- The gamma LUT comes before masking and resizing.
- The manifest keeps the source frame, time stamp, frame-order version, source and output sha256 and the transform
  digest (619b8e21…). Its sha256 (c54a9143…) is pinned in the configuration.
- F models read these PNGs as stored. At evaluation the same transform is applied to the validation source frames.
  The smoke checks confirm that stored PNGs equal the transform of their source frames.

`input_qc.json` holds the transform specification, the size scan and the per-recording QC metrics under anonymous QC
numbers.

## 2. Training (Task 2)

**Recipe** (identical for the four arms).
- Each update holds 16 training recordings: a uniform class, then a uniform recording within that class. Each
  recording gives 4 spaced 10 Hz samples, for 64 frames per update.
- Each recording is drawn as orig or gamma090 with probability 0.5 each.
- The loss is the mean per-frame five-class cross-entropy with the recording label.
- AdamW: head 1e-3, encoder 1e-5, weight decay 1e-4; gradient clip 1.0.
- 100 linear warm-up updates, then cosine to 0, for 1,500 updates. fp32 with TF32 off. The final update is kept.
- Head-only arms freeze the whole encoder, ln_post and proj included.

**Checks.**
- The evaluation verified every run's identity and final checkpoint.
- Per seed, the four arms share the batch plan, the initial state and the per-update sample digests.
- L0 and L1 read identical pixels, as do F0 and F1; L and F read different pixels.
- The smoke checks passed for all four arms with the committed code:
  - encoder parity against the export reference 2.9 x 10⁻⁶, and against the E1 cache 9.7 x 10⁻⁴;
  - the PNG contract;
  - identical reruns and identical resume;
  - unchanged frozen weights;
  - non-zero gradients on all 17 trained tensors of L1 and F1.

**Runs.**
- Head-only runs took 172–174 s and last-block runs 213–214 s.
- Peak GPU memory was 947 MiB and 1,525 MiB.
- All twelve ran from 02:57 to 03:38 UTC on 7 October, 41 minutes.
- Mean training loss over the last 50 updates: L0 0.157–0.176, L1 0.138–0.158, F0 0.160–0.190, F1 0.139–0.170.

## 3. Results per model

Every cell gives the selection half, then the calibration half.

**With B3.**

| Model | B3 setting (half-window / quantile) | Segment F1@50 | Boundary recall | Boundary precision | Native macro-F1 | Cuts per clip |
|---|---|---|---|---|---|---|
| E1 on L, unadapted | feasible, 9 / 0.995 | 0.894 / 0.959 | 0.955 / 0.977 | 0.943 / 0.951 | 0.978 / 0.986 | 0.05 / 0.07 |
| E1 on F, unadapted | none feasible; diagnostic 9 / 0.95 | 0.871 / 0.926 | 0.969 / 0.990 | 0.879 / 0.891 | 0.966 / 0.971 | 0.18 / 0.25 |
| L0 s0 | feasible, 9 / 0.995 | 0.896 / 0.961 | 0.955 / 0.980 | 0.943 / 0.954 | 0.978 / 0.986 | 0.04 / 0.07 |
| L0 s1 | feasible, 9 / 0.995 | 0.896 / 0.962 | 0.955 / 0.982 | 0.943 / 0.949 | 0.978 / 0.983 | 0.05 / 0.08 |
| L0 s2 | feasible, 9 / 0.995 | 0.892 / 0.958 | 0.955 / 0.980 | 0.943 / 0.951 | 0.989 / 0.986 | 0.05 / 0.08 |
| L1 s0 | feasible, 7 / 0.995 | 0.908 / 0.971 | 0.961 / 0.977 | 0.961 / 0.973 | 0.989 / 0.986 | 0.10 / 0.10 |
| L1 s1 | feasible, 7 / 0.999 | 0.906 / 0.968 | 0.937 / 0.962 | 0.970 / 0.995 | 0.989 / 0.990 | 0.01 / 0.02 |
| L1 s2 | feasible, 7 / 0.995 | 0.912 / 0.969 | 0.961 / 0.982 | 0.961 / 0.968 | 0.989 / 0.983 | 0.09 / 0.13 |
| F0 s0 | none feasible; diagnostic 9 / 0.95 | 0.870 / 0.922 | 0.966 / 0.990 | 0.874 / 0.887 | 0.966 / 0.971 | 0.17 / 0.23 |
| F0 s1 | none feasible; diagnostic 9 / 0.95 | 0.868 / 0.921 | 0.966 / 0.990 | 0.870 / 0.883 | 0.966 / 0.971 | 0.18 / 0.24 |
| F0 s2 | none feasible; diagnostic 9 / 0.95 | 0.871 / 0.924 | 0.971 / 0.990 | 0.877 / 0.887 | 0.978 / 0.971 | 0.16 / 0.20 |
| F1 s0 | none feasible; diagnostic 9 / 0.95 | 0.867 / 0.920 | 0.969 / 0.987 | 0.864 / 0.879 | 0.966 / 0.969 | 0.16 / 0.22 |
| F1 s1 | none feasible; diagnostic 9 / 0.95 | 0.866 / 0.923 | 0.969 / 0.987 | 0.862 / 0.883 | 0.966 / 0.969 | 0.17 / 0.21 |
| F1 s2 | none feasible; diagnostic 9 / 0.95 | 0.870 / 0.931 | 0.969 / 0.987 | 0.864 / 0.889 | 0.978 / 0.956 | 0.16 / 0.22 |

**Without a decoder (argmax).** Per-sample labels fragment every model's output, so these rows are not systems.

| Model | Segment F1@50 | Native macro-F1 | Cuts per clip | Native sample error | Stream sample error |
|---|---|---|---|---|---|
| E1 on L, unadapted | 0.418 / 0.532 | 0.978 / 0.985 | 2.93 / 3.29 | 5.5% / 6.0% | 10.3% / 5.3% |
| E1 on F, unadapted | 0.446 / 0.545 | 0.978 / 0.971 | 3.20 / 3.65 | 6.1% / 6.6% | 10.2% / 5.3% |
| L0, three seeds | 0.438–0.447 / 0.528–0.534 | 0.978 / 0.986–0.993 | 2.83–3.06 / 3.26–3.40 | 5.3–5.8% / 5.9–6.2% | 9.8–10.2% / 5.3% |
| L1, three seeds | 0.453–0.471 / 0.542–0.579 | 0.978–0.989 / 0.978–0.986 | 2.65–3.28 / 3.01–3.49 | 5.1–6.2% / 5.5–6.3% | 9.0–9.2% / 4.9–5.1% |
| F0, three seeds | 0.448–0.465 / 0.549–0.555 | 0.966–0.978 / 0.971 | 2.94–3.23 / 3.43–3.60 | 5.6–5.9% / 6.2–6.5% | 10.0–10.1% / 5.2% |
| F1, three seeds | 0.441–0.462 / 0.559–0.577 | 0.966–0.978 / 0.969–0.976 | 2.83–3.22 / 3.14–3.46 | 5.2–5.9% / 6.0–6.4% | 9.6–10.0% / 5.0% |

- **The unadapted E1 head on the legacy input reproduces E1 + B3 exactly** (0.894 / 0.959).
- **On the fan input, the same head drops to 0.871 / 0.926.** Training the head (F0) or also the last block (F1) does
  not recover the loss.

**Native clips read correctly per class, with B3.** All models read PLAX 104 of 105 and 80 of 80, PSAX 68 of 68 on the
selection half, A4C 87 of 87 on the selection half and SC4C 12 of 13 and 7 of 7.

| Model | A5C, selection | A5C, calibration | A4C, calibration | PSAX, calibration |
|---|---|---|---|---|
| E1 on L, unadapted | 10/11 | 17/18 | 60/61 | 71/71 |
| E1 on F, unadapted | 9/11 | 16/18 | 59/61 | 71/71 |
| L0 s0 / s1 / s2 | 10, 10, 11 of 11 | 17, 17, 17 of 18 | 60, 60, 60 of 61 | 71, 70, 71 of 71 |
| L1 s0 / s1 / s2 | 11, 11, 11 of 11 | 17, 18, 18 of 18 | 60, 60, 59 of 61 | 71, 70, 70 of 71 |
| F0 s0 / s1 / s2 | 9, 9, 10 of 11 | 16, 16, 16 of 18 | 59, 59, 59 of 61 | 71, 71, 71 of 71 |
| F1 s0 / s1 / s2 | 9, 9, 10 of 11 | 16, 16, 16 of 18 | 59, 59, 57 of 61 | 70, 70, 70 of 71 |

## 4. Paired contrasts

The contrasts use a paired bootstrap of 2,000 draws (seed 20261002). Native clips are resampled by session and
streams by stream; stream intervals are approximate because streams reuse recordings. The interaction uses the same
draws for all four systems. Values are points of segment F1@50 with 95% intervals, selection half first.

| Contrast | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| F0 − L0, B3 | −2.6 (−4.5, −0.7) / −3.8 (−6.0, −1.7) | −2.8 (−4.8, −0.9) / −4.1 (−6.1, −2.3) | −2.1 (−4.1, −0.0) / −3.4 (−5.3, −1.6) |
| F1 − L1, B3 | −4.2 (−6.7, −1.6) / −5.1 (−7.3, −2.9) | −4.0 (−7.1, −1.0) / −4.4 (−7.4, −1.7) | −4.2 (−6.7, −1.8) / −3.8 (−5.5, −2.2) |
| F1 − F0, B3 | −0.3 (−1.6, +0.9) / −0.3 (−1.5, +1.0) | −0.2 (−1.4, +1.0) / +0.3 (−0.4, +1.1) | −0.0 (−1.3, +1.3) / +0.7 (−0.9, +2.6) |
| L1 − L0, B3 | +1.2 (−0.2, +2.8) / +1.0 (−0.0, +2.1) | +1.0 (−0.7, +2.7) / +0.6 (−1.2, +2.2) | +2.0 (+0.4, +3.8) / +1.1 (−0.1, +2.4) |
| (F1 − F0) − (L1 − L0), B3 | −1.5 (−3.5, +0.3) / −1.3 (−2.5, −0.1) | −1.2 (−3.4, +0.9) / −0.3 (−2.1, +1.6) | −2.1 (−4.1, −0.1) / −0.4 (−2.2, +1.5) |
| F0 − L0, argmax | +2.8 (+0.5, +5.0) / +2.4 (−0.1, +4.8) | +2.4 (+0.1, +4.7) / +2.7 (+0.2, +5.3) | +0.1 (−2.0, +2.4) / +1.5 (−1.1, +4.4) |
| F1 − L1, argmax | −0.2 (−2.4, +1.7) / +0.9 (−2.3, +4.0) | −0.9 (−3.0, +1.3) / −0.2 (−3.2, +2.5) | −1.2 (−3.3, +0.7) / +1.7 (−1.3, +4.7) |
| F1 − F0, argmax | −0.4 (−1.8, +1.1) / +2.2 (+0.4, +4.1) | −0.1 (−1.6, +1.4) / +2.2 (+0.4, +4.3) | −0.7 (−2.4, +1.0) / +1.0 (−1.3, +3.2) |
| L1 − L0, argmax | +2.6 (+1.3, +4.0) / +3.7 (+1.8, +5.7) | +3.2 (+2.0, +4.4) / +5.2 (+3.8, +6.6) | +0.6 (−1.5, +2.6) / +0.8 (−1.1, +2.6) |
| (F1 − F0) − (L1 − L0), argmax | −3.0 (−4.8, −1.2) / −1.5 (−4.2, +1.1) | −3.3 (−5.0, −1.5) / −3.0 (−5.2, −0.7) | −1.3 (−3.6, +1.0) / +0.2 (−2.2, +2.9) |

- **With B3, F0 − L0 and F1 − L1 are consistently negative on both halves.** All twelve intervals lie below zero.
  Seed 2's selection interval for F0 − L0 ends at −0.02 points.
- **The B3 differences describe whole systems.** Every F row uses the diagnostic setting 9 / 0.95, because no setting
  is feasible, while the L rows use their chosen stricter settings. The diagnostic setting proposes more boundaries:
  F rows have boundary precision 0.862–0.877 on the selection half against 0.943–0.970 for L.
- **At one common setting the fan input still trails, by less.** This is a description of the calibration half, where
  settings are chosen, and not a selection step. At 9 / 0.995, F0 reaches 0.942–0.948 against 0.958–0.962 for L0,
  and F1 reaches 0.951–0.954 against 0.964–0.967 for L1. At 9 / 0.95 the gap is about 1 point.
- **The native floor is out of reach for the fan input under any setting.** The best calibration native macro-F1 over
  all 40 settings is 0.969–0.971 for every F model, below the 0.980 floor. For L0 and L1 it is 0.986–0.997.
- **Without a decoder, the fan input does not lower stream segmentation.** F0 − L0 is +0.1 to +2.8 points. But every
  model's argmax segment F1@50 is 0.42–0.58, so the decoded rows are the ones that matter.
- **L1 − L0 is the only contrast with a positive tendency on every cell.** With B3, its intervals include zero except
  in seed 2 on the selection half. Without a decoder, its intervals lie above zero in seeds 0 and 1 on both halves.
- **Native macro-F1 with B3** is lower for F than for L in every seed and half. F1 − L1 is −1.1 to −2.2 points on the
  selection half and −1.7 to −2.8 on the calibration half, with session intervals that touch zero, except seed 2 on
  the calibration half (−5.9, −0.1).
- **Boundary recall with B3** is higher for F than for L. F1 − L1 is +0.5 to +3.1 points, because the diagnostic
  setting proposes more boundaries.

## 5. Where the changes come from

**Native clips fixed / newly wrong, with B3.** The first number counts clips the first system of the pair reads
correctly and the second misreads.

| Contrast | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| F0 − L0 | 0 / 1 and 0 / 2 | 0 / 1 and 1 / 2 | 0 / 1 and 0 / 2 |
| F1 − L1 | 0 / 2 and 0 / 3 | 0 / 2 and 1 / 4 | 0 / 1 and 1 / 5 |
| F1 − F0 | 0 / 0 and 0 / 1 | 0 / 0 and 0 / 1 | 0 / 0 and 0 / 3 |
| L1 − L0 | 1 / 0 and 0 / 0 | 1 / 0 and 1 / 0 | 0 / 0 and 1 / 2 |

Each cell gives the selection half, then the calibration half.

- **F1 against L1.** The new errors are mostly A5C clips read as A4C: 2, 2 and 1 on the selection half and 1, 2 and 2
  on the calibration half. There are also A4C clips read as A5C on the calibration half (1, 1 and 2).
- **L1 against L0.** The fixes are single A5C clips read as A4C by L0, in seeds 0 and 1 on the selection half and in
  seeds 1 and 2 on the calibration half. Seed 2 also newly misreads two calibration clips (one A4C read as A5C).
- **Stream segments.** For L1 − L0 with B3, the stream segments fixed / newly missed are 8 / 4,
  5 / 8 and 12 / 4 on the selection half, from 6 / 4, 6 / 9 and 9 / 4 source recordings. On the calibration half they
  are 5 / 4, 4 / 14 and 6 / 3, from 5 / 5, 6 / 14 and 6 / 4 recordings. Streams reuse recordings, so these segments
  are not independent cases.

## 6. Goal

| System | Mean segment F1@50 | Required | Margin | Calibration margin against 0.9739 + 0.01 |
|---|---|---|---|---|
| F1 + B3 (candidate) | 0.868 / 0.925 | 0.944 / 0.982 | −0.077 / −0.057 | −0.059 |
| F0 + B3 | 0.870 / 0.922 | 0.944 / 0.982 | −0.075 / −0.060 | −0.061 |
| L1 + B3 | 0.909 / 0.969 | 0.944 / 0.982 | −0.035 / −0.013 | −0.015 |
| L0 + B3 | 0.895 / 0.960 | 0.944 / 0.982 | −0.050 / −0.022 | −0.024 |

- **References**, reproduced from their frozen outputs: E0 + B3 0.934 / 0.934, E1 + B3 0.894 / 0.959, B7 + B3
  0.880 / 0.972, E1 + ASRF seeds 0 to 2 0.907–0.910 / 0.964–0.976.
- **F1 + B3** has no feasible B3 setting in any seed. Its selection native macro-F1 is below 0.973 in seeds 0 and 1,
  and its calibration native macro-F1 is below 0.980 in every seed.
- **L1 + B3** meets every per-seed floor except seed 1's selection boundary recall (0.937). Its means miss the goal on
  both halves.

## 7. Interpretation under the plan

- **Main question: is F1 better than both F0 and L1?** No. F1 − L1 is consistently negative with B3, and F1 − F0
  shows no consistent difference.
- **A fine-tuning gain shared by both inputs (L1 − L0 and F1 − F0).** Not shown. L1 − L0 is positive in every cell
  but consistent in none of the halves, and F1 − F0 shows nothing.
- **A fan-specific gain (the interaction).** Not shown. The interaction is negative, with two of six intervals below
  zero.
- **Simple adaptation on the new input (F0 − L0).** It is consistently worse than on the old input, so there is no
  reason to keep simple adaptation on fan224_v1 either.
- **Outcome.** The plan's first case applies: no gain from the candidate. The round stops, the input interface stays,
  and nothing is scanned or added. This does not show that every cropping method is ineffective.
- **L1.** Its tendency is reported as a local positive result of ordinary visual adaptation on the old input, short
  of the goal. On the selection half it rests on single A5C clips and a few stream segments.

## 8. What this does not show

- **That full-sector input cannot help.** One fixed template was tested with 1,500 updates of head-only or last-block
  adaptation from an encoder pretrained on centre-cropped images. The fan input shrinks the anatomy (scale 0.882
  against 1.067 for the legacy path) and adds lateral field. Neither effect is separated from the other.
- **Why the fan input loses A5C clips.** No analysis in this round localises it.
- **That text or interface marks cause any error.** The legacy crop already removes most peripheral text, and the
  template was not compared with a text-only mask as a training arm.
- **Independent evidence.** Both validation halves have been used in earlier development rounds, the three seeds share
  one validation set, constructed streams reuse recordings, and the counts of changed native clips are small.

## 9. Cost

- **Input QC.** Scanning 619,524 frame headers, plus all frames of the 100 QC recordings, took a few minutes on 8 CPU
  workers. The PNG export of 415,228 frames took about 12 minutes on 8 CPU workers.
- **Training.** Twelve runs in 41 minutes on one NVIDIA L40S (fp32, TF32 off). Head-only runs took 172–174 s and
  last-block runs 213–214 s.
- **Evaluation.**
  - Each of 14 models took 104–105 s, about 73 s of it for reading 40,566 validation frames.
  - The whole evaluation, with comparisons, references and freezing, took about 30 minutes.
  - The training-split features for the twelve models (415,228 frames each, plan Task 3) were started after the
    evaluation and go to the server cache only. They serve no result in this README.

## Files

- `summary.json` holds every model's rows, B3 grid and choice, the contrasts and interaction, the goals for all four
  arms, the references, the run records and the timing. It contains no recording ids.
- `report_tables.md` holds the generated tables.
- `input_qc.json` holds the input contract and its QC under anonymous QC numbers.
- The twelve models are frozen in `models_frozen/fan224_visual_adaptation_v1/{arm}_s{seed}` together with their batch
  plans.
- Per-sequence probabilities, per-recording details and the QC sheets stay on the server, in
  `runs/fan224_visual_adaptation_v1/`.
