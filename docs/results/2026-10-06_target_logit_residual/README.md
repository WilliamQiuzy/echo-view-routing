# Target-logit residual readout: six paired trainings against the three C models (2026-10-06)

This round tested one change to the C model's window readout. R adds the target frame's own frame-head logits to a
temporal correction that starts at zero. Z reads out the same zero-initialised correction alone. The three C models of
the previous round are the reference. The plan is `docs/superpowers/plans/2026-10-06-target-logit-residual-training.md`.
The configuration and pipeline were committed before training (9eed6d8). Only EV9V training data entered training,
and the two validation halves were read only after all six runs had ended. The test split and EchoNet-Pediatric were
not read, and the public labels are unchanged.

## Answers

1. **Which six models were trained.** Z and R, each with seeds 0, 1 and 2.
   - **What the new runs share with C.** Each seed starts from the C run's untrained initial state for that seed, with
     only the window output layer (`head.weight`, `head.bias`) set to zero. Z and R of one seed share this initial
     state. Every run reads the same windows in the same order as C, with C's loss, optimiser and budget of 3,750
     updates, and keeps its final epoch.
   - **How the three controls differ.** The three C models were not retrained. Their frozen weights were re-evaluated
     with the same code. Their window output layer started from the E1 probe, not from zero. R − Z therefore isolates
     the direct target path, while R − C changes both the path and the initialisation.
   - **Freezing.** The six new models are frozen in `models_frozen/target_logit_residual_v1/{arm}_s{seed}`.
2. **R − Z and R − C.**
   - **With B3, neither shows a consistent difference.** On the selection half, R − Z is +2.5 points of segment
     F1@50 in seed 1, with an interval above zero, and −1.1 and −1.7 in seeds 0 and 2. On the calibration half it is
     −0.7 to +0.5. R − C is −1.8 to +1.1 on the selection half and −1.4 to −0.3 on the calibration half. Five of
     those six R − C point estimates are negative, and every interval includes zero.
   - **Without a decoder, R is worse than both.** R's segment F1@50 is lower than Z's and C's in every seed and half,
     by 2.4 to 6.2 points, and all twelve intervals lie below zero.
3. **Whether frame-level denoising and whole-clip flips improved.** Against C, whole-clip flips improved on the
   calibration half only, and frame-level denoising got worse.
   - **Whole-clip flips, calibration half.** Without a decoder, R misreads fewer native clips than C in every seed
     (one or two fixed, none newly wrong). It reads 17 of 18 A5C clips in every seed against C's 16, and in seed 1 it
     also fixes an A4C clip that C read as A5C. The session intervals of the native macro-F1 difference have a lower
     bound of zero, so under this round's rule this is not a consistent improvement.
   - **Whole-clip flips, selection half.** The changes are mixed. In each seed R fixes two A4C clips that C read as
     A5C. It newly misreads one A5C clip in seeds 0 and 1, and five clips in seed 2 (three A4C, two A5C).
   - **Frame-level denoising.** R's raw output cuts native clips 1.7 to 3.2 times as often as C's (0.18–0.31
     against 0.08–0.12 cuts per clip). Its target-sample error in the constructed streams is slightly higher than C's
     and Z's.
4. **Whether the strong baselines are beaten with all constraints.** No. R + B3 averages 0.848 segment F1@50 on the
   selection half and 0.961 on the calibration half. The goal requires 0.944 and 0.982. The selection native floor
   of 0.973 is missed in every seed (0.940, 0.966, 0.914), and seed 1 also misses the selection boundary-recall
   floor (0.940). Z + B3 and C + B3 miss the goal as well.

Under the plan's reading this is case 3, a trade-off between recognition and temporal gains. The direct path protects
some whole-clip labels on the calibration half but brings back frame-level fragmentation. It does not solve the
problem. The round stops here. No other readout, gate, residual weight, encoder freezing, training length or
threshold search was tried.

## Terms

- **Final logits.** `h` is the temporal Transformer's CLS state after its LayerNorm. Z reads out `Δ = W h + b`. R
  reads out `a_c + Δ`, where `a_c` is the frame head's logits at the window's target sample. For both, `W` and `b`
  start at zero, and the frame head starts from the E1 probe and keeps training. C reads out `W h + b` with `W` and
  `b` started from the E1 probe.
- **Argmax rows** give each sample the most probable view of the window whose centre is nearest. Window centres are 5
  samples apart. No decoder is used.
- **B3 rows** run the change-point decoder on the same probabilities. For every model the B3 setting was chosen on
  the calibration half only. A setting qualifies when native macro-F1 is at least 0.980 and boundary recall at least
  0.95 there. The qualifying setting with the highest segment F1@50 is then applied once to the selection half. When
  no setting qualifies, the fixed setting 9 / 0.95 is shown as a diagnostic, and such rows pass nothing.
- **Native clips** are whole single-view recordings. A clip is read correctly when the majority of its samples is.
- **Cuts per clip** count label changes inside native clips. All of them are false.
- **Points** are hundredths of a metric. For example, +2.5 points of segment F1@50 means +0.025.

## 1. Results per model

Every cell gives the selection half, then the calibration half.

**With B3.**

| Model | B3 setting (half-window / quantile) | Segment F1@50 | Boundary recall | Boundary precision | Native macro-F1 | Cuts per clip |
|---|---|---|---|---|---|---|
| C s0 | none feasible; diagnostic 9 / 0.95 | 0.854 / 0.965 | 0.966 / 0.990 | 0.864 / 0.959 | 0.922 / 0.975 | 0.07 / 0.07 |
| C s1 | feasible, 11 / 0.80 | 0.857 / 0.974 | 0.969 / 0.992 | 0.858 / 0.956 | 0.948 / 0.985 | 0.09 / 0.08 |
| C s2 | none feasible; diagnostic 9 / 0.95 | 0.852 / 0.970 | 0.971 / 0.992 | 0.828 / 0.959 | 0.941 / 0.975 | 0.10 / 0.08 |
| Z s0 | feasible, 15 / 0.995 | 0.854 / 0.965 | 0.955 / 0.985 | 0.863 / 0.963 | 0.930 / 0.982 | 0.04 / 0.05 |
| Z s1 | none feasible; diagnostic 9 / 0.95 | 0.843 / 0.966 | 0.966 / 0.995 | 0.837 / 0.952 | 0.966 / 0.978 | 0.11 / 0.10 |
| Z s2 | none feasible; diagnostic 9 / 0.95 | 0.852 / 0.962 | 0.966 / 0.995 | 0.844 / 0.952 | 0.929 / 0.975 | 0.10 / 0.13 |
| R s0 | feasible, 9 / 0.99 | 0.842 / 0.957 | 0.961 / 0.987 | 0.830 / 0.942 | 0.940 / 0.982 | 0.09 / 0.11 |
| R s1 | feasible, 7 / 0.999 | 0.868 / 0.971 | 0.940 / 0.977 | 0.932 / 0.982 | 0.966 / 0.986 | 0.03 / 0.05 |
| R s2 | feasible, 5 / 0.99 | 0.835 / 0.955 | 0.958 / 0.982 | 0.838 / 0.942 | 0.914 / 0.986 | 0.11 / 0.20 |

**Without a decoder (argmax).**

| Model | Segment F1@50 | Boundary recall | Boundary precision | Native macro-F1 | Cuts per clip | Same-class cut rate |
|---|---|---|---|---|---|---|
| C s0 | 0.847 / 0.965 | 0.966 / 0.990 | 0.852 / 0.959 | 0.935 / 0.975 | 0.08 / 0.11 | 0.11 / 0.03 |
| C s1 | 0.846 / 0.974 | 0.969 / 0.992 | 0.833 / 0.956 | 0.948 / 0.978 | 0.12 / 0.10 | 0.12 / 0.02 |
| C s2 | 0.836 / 0.963 | 0.971 / 0.990 | 0.808 / 0.945 | 0.941 / 0.975 | 0.11 / 0.11 | 0.10 / 0.03 |
| Z s0 | 0.843 / 0.961 | 0.969 / 0.992 | 0.835 / 0.947 | 0.944 / 0.985 | 0.10 / 0.13 | 0.11 / 0.03 |
| Z s1 | 0.832 / 0.966 | 0.966 / 0.995 | 0.811 / 0.952 | 0.966 / 0.985 | 0.11 / 0.17 | 0.12 / 0.03 |
| Z s2 | 0.832 / 0.962 | 0.966 / 0.995 | 0.802 / 0.947 | 0.929 / 0.975 | 0.14 / 0.14 | 0.12 / 0.03 |
| R s0 | 0.795 / 0.937 | 0.966 / 0.990 | 0.766 / 0.900 | 0.940 / 0.983 | 0.18 / 0.24 | 0.15 / 0.05 |
| R s1 | 0.784 / 0.933 | 0.969 / 0.992 | 0.740 / 0.890 | 0.954 / 0.993 | 0.20 / 0.31 | 0.15 / 0.06 |
| R s2 | 0.807 / 0.911 | 0.969 / 0.990 | 0.782 / 0.857 | 0.914 / 0.983 | 0.20 / 0.30 | 0.14 / 0.08 |

Three-seed means with B3 are 0.848 / 0.961 for R, 0.849 / 0.964 for Z and 0.854 / 0.970 for C.

- **B3 feasibility.** For R, a B3 setting meets both calibration floors in all three seeds. For C and Z this happens in
  one seed each.
- **R seed 1.** Its selection result with B3 is the highest of the nine models (0.868). It uses the strictest
  threshold in the grid (quantile 0.999), and its selection boundary recall of 0.940 is below the floor.
- **C rows.** They reproduce the previous round's C results.

The reference systems reproduced exactly from their frozen outputs.

| System | Segment F1@50 | Boundary recall | Native macro-F1 |
|---|---|---|---|
| E0 + B3 | 0.934 / 0.934 | 0.961 / 0.955 | 0.965 / 0.966 |
| E1 + B3 | 0.894 / 0.959 | 0.955 / 0.977 | 0.978 / 0.986 |
| B7 + B3 | 0.880 / 0.972 | 0.955 / 0.975 | 0.957 / 0.993 |
| E1 + ASRF, seeds 0 to 2 | 0.907–0.910 / 0.964–0.976 | 0.942–0.953 / 0.972–0.977 | 0.948–0.966 / 0.975–0.993 |

## 2. Paired differences

Each pair was compared with a paired bootstrap of 2,000 draws (seed 20261002). Native clips are resampled by
session and streams by stream. The stream intervals are approximate because streams reuse recordings. Values are
points of segment F1@50 with 95% intervals, selection half first.

| Comparison | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| R − Z, B3 | −1.1 (−2.9, +0.5) / −0.7 (−1.8, +0.3) | +2.5 (+0.8, +4.3) / +0.5 (−0.9, +1.9) | −1.7 (−3.6, +0.2) / −0.7 (−2.0, +0.6) |
| R − Z, argmax | −4.8 (−7.0, −2.8) / −2.4 (−4.0, −1.1) | −4.8 (−7.1, −2.5) / −3.2 (−4.9, −1.8) | −2.4 (−4.8, −0.1) / −5.1 (−7.4, −3.0) |
| R − C, B3 | −1.2 (−2.6, +0.1) / −0.7 (−1.6, +0.0) | +1.1 (−0.6, +2.9) / −0.3 (−1.9, +1.2) | −1.8 (−4.2, +0.4) / −1.4 (−2.9, +0.0) |
| R − C, argmax | −5.2 (−7.7, −2.9) / −2.8 (−4.4, −1.4) | −6.2 (−8.8, −3.7) / −4.1 (−5.9, −2.4) | −2.9 (−5.4, −0.6) / −5.2 (−7.5, −3.2) |
| Z − C, B3 | −0.1 (−1.0, +1.0) / −0.0 (−1.3, +1.2) | −1.4 (−2.7, −0.2) / −0.8 (−1.6, −0.3) | −0.1 (−2.1, +1.8) / −0.7 (−1.7, +0.2) |
| Z − C, argmax | −0.4 (−1.5, +0.6) / −0.4 (−1.1, +0.3) | −1.4 (−2.9, −0.2) / −0.8 (−1.6, −0.3) | −0.4 (−1.9, +0.9) / −0.1 (−1.2, +1.2) |

- **With B3, no comparison is consistent across the three seeds.** R − Z lies above zero only in seed 1 on the
  selection half. Z − C lies below zero only in seed 1, on both halves. An interval that contains zero does not mean
  the systems are equal.
- **Without a decoder, R is consistently worse than Z and C.** All twelve intervals lie below zero.
- **B3 differences describe whole systems.** The B3 settings differ between models (section 1), so these differences
  cannot all be credited to the readout.
- **Other measures.** Without a decoder, R − C in cuts per clip is +0.08 to +0.21, with every interval above zero.
  R − Z is +0.06 to +0.16, with five of six intervals above zero. Boundary recall without a decoder differs by at most
  0.5 points between the arms. With B3, R's boundary recall is lower than C's in seed 1 (−2.9 / −1.5 points, both
  intervals below zero). `report_tables.md` and `summary.json` hold the boundary F1 and same-class cut differences.

**Native clips fixed / newly wrong.** The first number counts clips that the first model of the pair reads correctly
and the second misreads. The second number counts the reverse.

| Comparison | Seed | Argmax, selection | Argmax, calibration | B3, selection | B3, calibration |
|---|---|---|---|---|---|
| R − Z | 0 | 1 / 1 | 1 / 2 | 1 / 0 | 0 / 0 |
| R − Z | 1 | 0 / 1 | 1 / 0 | 0 / 0 | 1 / 0 |
| R − Z | 2 | 2 / 4 | 1 / 0 | 2 / 4 | 2 / 0 |
| R − C | 0 | 2 / 1 | 1 / 0 | 2 / 0 | 1 / 0 |
| R − C | 1 | 2 / 1 | 2 / 0 | 2 / 0 | 1 / 1 |
| R − C | 2 | 2 / 5 | 1 / 0 | 2 / 5 | 2 / 0 |
| Z − C | 0 | 1 / 0 | 2 / 0 | 1 / 0 | 1 / 0 |
| Z − C | 1 | 2 / 0 | 1 / 0 | 2 / 0 | 0 / 1 |
| Z − C | 2 | 0 / 1 | 0 / 0 | 0 / 1 | 0 / 0 |

**Native clips read correctly per class, without a decoder.** Only the classes that differ between models are shown.
All nine models read PLAX 104 of 105 and 80 of 80, PSAX 68 of 68 on the selection half, and SC4C 12 of 13 and 7 of 7.

| Model | A4C, selection | A5C, selection | A4C, calibration | A5C, calibration | PSAX, calibration |
|---|---|---|---|---|---|
| C s0 | 85/87 | 8/11 | 60/61 | 16/18 | 70/71 |
| C s1 | 85/87 | 9/11 | 60/61 | 16/18 | 71/71 |
| C s2 | 85/87 | 9/11 | 60/61 | 16/18 | 70/71 |
| Z s0 | 86/87 | 8/11 | 61/61 | 16/18 | 71/71 |
| Z s1 | 87/87 | 9/11 | 61/61 | 16/18 | 71/71 |
| Z s2 | 85/87 | 8/11 | 60/61 | 16/18 | 70/71 |
| R s0 | 87/87 | 7/11 | 60/61 | 17/18 | 70/71 |
| R s1 | 87/87 | 8/11 | 61/61 | 17/18 | 71/71 |
| R s2 | 84/87 | 7/11 | 60/61 | 17/18 | 70/71 |

- **R against C.** On the calibration half R fixes one or two clips per seed and breaks none. On the selection half it
  trades A4C and A5C clips (2 / 1, 2 / 1 and 2 / 5).
- **R against Z.** The changes are small and mixed, with 0 to 2 clips fixed and 0 to 4 newly wrong per seed and half.
- **Intervals.** Without a decoder, the native macro-F1 difference R − C is +0.4, +0.6 and −2.7 points on the
  selection half, with intervals across zero. On the calibration half it is +0.8, +1.5 and +0.8 points, with
  intervals whose lower bound is zero.

## 3. What the correction keeps and what it breaks

Every R model was also read with Δ set to zero, which leaves only the target frame's logits. This reading equals the
model's frame head read at the same window centres (the J-grid path of the diagnostic round) within 1.1 × 10⁻⁶. It is
a diagnostic, not a candidate. Values are the selection half, then the calibration half.

| Seed | Native macro-F1, Δ = 0 | Cuts per clip, Δ = 0 | Segment F1@50, Δ = 0 | Native clips fixed / newly wrong by the correction, selection | Same, calibration |
|---|---|---|---|---|---|
| 0 | 0.948 / 0.993 | 0.63 / 0.70 | 0.704 / 0.804 | 2 / 2 | 0 / 2 |
| 1 | 0.907 / 0.993 | 0.59 / 0.78 | 0.700 / 0.802 | 5 / 0 | 0 / 0 |
| 2 | 0.913 / 0.990 | 0.62 / 0.78 | 0.706 / 0.783 | 2 / 1 | 0 / 1 |

- **The correction removes most of the frame-level fragmentation.** Cuts fall from 0.59–0.78 per clip to 0.18–0.31.
  Segment F1@50 rises from 0.700–0.706 to 0.784–0.807 on the selection half and from 0.783–0.804 to 0.911–0.937 on
  the calibration half.
- **It leaves more fragmentation than C's readout**, which cuts 0.08–0.12 times per clip.
- **On the selection half it fixes at least as many native clips as it breaks** (2 / 2, 5 / 0 and 2 / 1).
- **On the calibration half it fixes none and newly breaks up to two clips** that the target logits alone read
  correctly. So the correction still turns some correct target-frame readings into errors, as the diagnostic round
  found for C.

## 4. Target-sample error and context

The target-sample error is the share of windows whose target sample is read wrongly. The recording-weighted figure in
brackets counts every source recording equally. Each cell gives the range over the three seeds.

**Selection half.**

| Arm | Native clips (3,444 windows) | Single-recording streams (1,868) | Cross-recording, one view (905) | Mixed views (1,449) |
|---|---|---|---|---|
| C | 2.2–2.4% (2.1–2.3%) | 7.9–9.1% (2.7%) | 3.5–4.9% (1.4–1.6%) | 6.9–7.7% (2.7–3.2%) |
| Z | 2.0–2.6% (1.9–2.5%) | 8.5–9.2% (2.7–2.9%) | 4.4–4.6% (1.5–1.7%) | 6.5–7.3% (2.3–3.0%) |
| R | 2.4–3.1% (2.2–3.1%) | 9.5–10.1% (3.2–3.6%) | 6.0–6.6% (1.9–2.2%) | 7.3–7.5% (2.6–2.7%) |

**Calibration half.**

| Arm | Native clips (2,950 windows) | Single-recording streams (1,833) | Cross-recording, one view (798) | Mixed views (1,440) |
|---|---|---|---|---|
| C | 2.2–2.6% (1.3–1.6%) | 0.9–1.2% (0.9–1.4%) | 1.3–1.5% (0.7–0.8%) | 2.5–3.8% (2.2–3.3%) |
| Z | 2.1–2.7% (1.2–1.6%) | 1.0–1.1% (0.9–1.0%) | 1.5–1.6% (0.9%) | 2.2–3.0% (1.9–2.6%) |
| R | 2.6–3.0% (1.7–2.1%) | 1.1–1.6% (0.8–1.3%) | 2.1–3.1% (1.3–2.5%) | 3.0–3.5% (2.8–3.0%) |

- **Per window, R is slightly worse.** In every seed, R's error is higher than C's and Z's for native clips,
  single-recording streams and cross-recording one-view windows, on both halves. In mixed-view windows the three arms
  are close.
- **Weighted by recording, the gaps are smaller.** R is higher in most groups but not in all.
- **R's native-clip errors are spread over more recordings.** They touch 18–20 recordings on the selection half and
  19–23 on the calibration half, against 12–15 and 8–9 for C.

**Context probe.** Each of 474 targets keeps the centre frames of its own window. The remaining frames come from a
recording of another session, either of the same view or of another view. The table gives the accuracy when another
view fills the rest of the window.

| Centre frames from the target | C | Z | R |
|---|---|---|---|
| 4 of 16 | 0.022–0.025 | 0.039–0.058 | 0.938–0.941 |
| 8 of 16 | 0.861–0.936 | 0.958–0.963 | 0.953–0.961 |
| 12 of 16 | 0.975–0.977 | 0.973–0.979 | 0.964–0.975 |
| 14 of 16 | 0.979–0.980 | 0.978–0.982 | 0.967–0.980 |

- **R keeps the target's own view when another view fills most of the window.** With 4 of 16 frames from the target,
  R is right 94% of the time, while C and Z are right in 2–6% of cases.
- **With 12 or 14 frames from the target, R is equal to or slightly below C and Z.** Its same-view accuracy is also
  slightly lower (0.975–0.989 against 0.983–1.000).
- **The probe shows that R's output follows the target frame**, as the direct path is designed to do. The round did
  not test whether this property causes R's extra cuts.

## 5. Goal

| System | Mean segment F1@50 | Required | Margin | Calibration margin against the 0.9739 sensitivity reference |
|---|---|---|---|---|
| R + B3 | 0.848 / 0.961 | 0.944 / 0.982 | −0.096 / −0.021 | −0.013 |
| Z + B3 | 0.849 / 0.964 | 0.944 / 0.982 | −0.095 / −0.018 | −0.010 |
| C + B3 | 0.854 / 0.970 | 0.944 / 0.982 | −0.090 / −0.013 | −0.004 |

The required values are the strongest reference per half plus 0.01. Those references are E0 + B3 on the selection
half (0.934) and B7 + B3 on the calibration half (0.972).

| Constraint for R + B3 | Floor | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|---|
| Boundary recall, selection | 0.95 | 0.961 | 0.940 | 0.958 |
| Native macro-F1, selection | 0.973 | 0.940 | 0.966 | 0.914 |
| Boundary recall, calibration | 0.95 | 0.987 | 0.977 | 0.982 |
| Native macro-F1, calibration | 0.980 | 0.982 | 0.986 | 0.986 |

The goal is not met.

## 6. Interpretation under the plan

- **Case 1 (R better than Z, but not better than C).** It does not apply. R is not consistently better than Z. With
  B3 one seed-half favours R, and without a decoder R is worse in every cell.
- **Case 2 (Z and R both better than C, with no consistent R − Z difference).** It does not apply. Neither Z nor R
  is better than C, and Z is below C in seed 1.
- **Case 3 (R's native recognition improves while target-time or segmentation results regress).** This case fits.
  On the calibration half R misreads fewer native clips than C in every seed, although the intervals touch zero.
  Without a decoder its segmentation is consistently worse, with more cuts and a slightly higher target-sample error.
  With B3 the full system is not better than C. This is a trade-off between recognition and temporal gains, and the
  problem is not solved.
- **Case 4 (R improves native recognition against C and Z and keeps the temporal gains).** It does not apply,
  because the temporal gains are not kept.

The goal is not met by R, Z or C. As the plan fixes in advance, the round stops after the six trainings.

## 7. What this does not show

- **That a direct target path cannot help in any form.** One form was tested. It is an unweighted sum of logits with
  a zero-initialised correction, trained jointly with four ViT blocks.
- **Why R's raw output fragments more than C's.** The Δ = 0 reading shows that R's target-frame logits fragment
  heavily by themselves and that the correction removes most but not all of it. No experiment in this round separates
  further causes.
- **Independent evidence.** Both validation halves were used in earlier development rounds. Constructed streams reuse
  recordings, so their intervals are approximate, and the three seeds share one validation set. The counts of changed
  native clips are small. The selection half has 11 A5C clips and the calibration half 18.

## 8. Cost

- **Training.** Each model took 1,708–1,713 s (about 28.5 minutes, 0.46 s per update) on one NVIDIA L40S, in fp32 with
  TF32 off. Peak allocated GPU memory was 7,566 MiB. The six runs ran in sequence from 16:59 to 19:51 UTC, 2.9 hours
  in total. Each finished in one pass without a resume.
- **Evaluation.** Each of the nine models took 106–110 s, about 75 s of which was feature extraction. The total was
  about 16 minutes. This covers the windows, the B3 grid, the census, the probe and the Δ = 0 reading.

## Files

- `summary.json` holds every model's rows, B3 grid and choice, census, probe and Δ = 0 diagnostic. It also holds the
  paired comparisons, the goals for R, Z and C, the references, the run records and the timing. It contains no
  recording ids.
- `report_tables.md` holds the generated tables.
- Per-recording details and the per-sequence probabilities of all nine models stay on the server in
  `runs/target_logit_residual_v1/report`.
