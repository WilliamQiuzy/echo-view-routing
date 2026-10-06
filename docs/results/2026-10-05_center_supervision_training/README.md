# Target-time supervision: nine paired video-model trainings on EV9V (2026-10-05)

This round trained one video model under three window targets. It asks whether telling the model which time point
to report helps temporal segmentation, while the model also learns native recording classification. The plan is
`docs/superpowers/plans/2026-10-05-ev9v-center-supervision-training.md`. Every parameter was fixed in
`configs/experiments/ev9v_center_supervision_v1.yaml` and the pipeline was committed before the formal runs (6fcc200).
Only EV9V training data entered training. The two validation halves served evaluation after all nine trainings had
ended. The test split was not read, EchoNet-Pediatric was not used and the public labels are unchanged.

## Answers

1. **Which nine models were trained.** Arms N, S and C, each with seeds 0, 1 and 2. All nine share one architecture,
   the E1 ViT-B/16 frame encoder with blocks 8-11, ln_post and proj trainable, followed by a two-layer temporal
   Transformer with positions relative to the target sample. Each model ran 3,750 updates of 8 windows. The final
   epoch is the model; nothing was chosen on validation. The nine models are frozen in
   `models_frozen/center_supervision_v1/{arm}_s{seed}` on the server.
2. **Whether C beats S.** C is ahead of S in every seed and half with B3, by 1.3 to 2.8 points of segment F1@50. The
   rule fixed in advance needs all three seeds' intervals above zero on a half, and it is not met on either half, so
   no consistent gain is detected. On windows that mix views, C errs at the target time half to two-thirds as often as S.
3. **Whether native classification improved or regressed.** It regressed against the E1 base. On the selection half
   the C models make 3 to 5 more native-clip errors than E1 + B3, mainly A5C clips read as A4C. Native macro-F1 is
   0.92 to 0.95 against the floor of 0.973. Between C and S no consistent native difference is detected.
4. **Whether the same complete system beats the strongest previous configuration.** No. C + B3 averages 0.854
   segment F1@50 on the selection half and 0.970 on the calibration half. The goal requires 0.944 and 0.982. The
   selection native floor is missed in every seed, and in seeds 0 and 2 no B3 setting meets the calibration
   constraints at all.

The round stops here, as the plan requires. No layers, losses or sweeps were added.

## Terms

- A **window** is the 16 frames that B7's rule takes around a target sample. They are 16 consecutive 10 Hz samples,
  clamped at the ends of a sequence. The **target sample** is the time point the window's output stands for. Each frame's
  **relative position** is its sample index minus the target's.
- **N** trains on native windows only, each from one recording, with the recording's label as the target.
- **S** trains on the same four native windows per update plus four windows that mix views. Its target is the share
  of each view among the window's frames.
- **C** trains on exactly the same windows as S. Its target is the view at the target sample.
- **C − S** is the main comparison. The two arms share images, frame order, initial weights, dropout seeds, batches and
  updates. Only the window target differs.
- **B3** is the change-point decoder applied to the window outputs. For every model its setting was chosen on the
  calibration half only, needing native macro-F1 ≥ 0.980 and boundary recall ≥ 0.95, then applied once to the
  selection half. When no setting qualified, a fixed diagnostic setting is shown. Such rows pass no criterion. The
  **argmax** rows give every sample the most probable view of its nearest window, with no decoder.

## 1. Training

| Item | Value |
|---|---|
| Training data | 3,364 training recordings and the protocol's 1,500 training streams (constructed sequences of training recordings) |
| Windows per update | 4 native (shared by N, S, C) + 4 mixed (S and C) or the target recording's own window (N) |
| Mixed windows available | 195 to 816 training recordings and about 15,000 windows per class |
| Budget | 10 epochs × 375 updates; AdamW, encoder 1e-5, head 1e-4, warm-up then cosine; fp32 |
| Time | about 28.5 minutes per model on one L40S (about 8 GB of GPU memory); 4.3 hours for all nine |

- **Checks before the formal runs.** Every check passed.
  - Encoder parity was 2.9 × 10⁻⁶ against the reference embeddings and 9.7 × 10⁻⁴ against the fp16 cache.
  - Exactly blocks 8-11, ln_post and proj are trainable, all with finite non-zero gradients, and the frozen weights
    were unchanged.
  - Same-seed reruns and a resume reproduced the weights to about 10⁻⁸.
  - S and C read one manifest file, and all three arms started from one initial state per seed.
- **Checks after training.** The report script verified all nine runs, their identities and final checkpoints, and the
  pairing. S and C had equal per-epoch input digests, and N had the same first four windows.
- **One deviation from the plan's order.** The window manifests were generated before the pipeline commit. The
  committed code regenerated them byte-identically for all three seeds.

## 2. Results per model

Each cell gives the selection half, then the calibration half.

| Model | B3 | Segment F1@50 (B3) | Boundary recall (B3) | Native macro-F1 | Segment F1@50 (argmax) |
|---|---|---|---|---|---|
| N s0 | diagnostic | 0.819 / 0.941 | 0.885 / 0.940 | 0.930 / 0.975 | 0.807 / 0.936 |
| N s1 | feasible | 0.820 / 0.965 | 0.908 / 0.957 | 0.930 / 0.985 | 0.812 / 0.962 |
| N s2 | diagnostic | 0.789 / 0.914 | 0.798 / 0.814 | 0.914 / 0.975 | 0.763 / 0.903 |
| S s0 | diagnostic | 0.838 / 0.949 | 0.927 / 0.955 | 0.922 / 0.978 | 0.832 / 0.971 |
| S s1 | feasible | 0.835 / 0.959 | 0.927 / 0.970 | 0.900 / 0.986 | 0.825 / 0.974 |
| S s2 | feasible | 0.824 / 0.957 | 0.911 / 0.970 | 0.907 / 0.985 | 0.826 / 0.970 |
| C s0 | diagnostic | 0.854 / 0.965 | 0.966 / 0.990 | 0.922 / 0.975 | 0.847 / 0.965 |
| C s1 | feasible | 0.857 / 0.974 | 0.969 / 0.992 | 0.948 / 0.985 | 0.846 / 0.974 |
| C s2 | diagnostic | 0.852 / 0.970 | 0.971 / 0.992 | 0.941 / 0.975 | 0.836 / 0.963 |

Three-seed means with B3 are 0.809 / 0.940 for N, 0.832 / 0.955 for S and 0.854 / 0.970 for C.

Reference systems, reproduced from their frozen outputs:

| System | Segment F1@50 | Boundary recall | Native macro-F1 |
|---|---|---|---|
| E0 + B3 (strongest previous on selection) | **0.934** / 0.934 | 0.961 / 0.955 | 0.965 / 0.966 |
| E1 + B3 | 0.894 / 0.959 | 0.956 / 0.977 | 0.978 / 0.986 |
| B7 + B3 (strongest previous on calibration) | 0.880 / **0.972** | 0.955 / 0.975 | 0.957 / 0.993 |
| E1 + ASRF, seeds 0 / 1 / 2 | 0.910, 0.909, 0.907 / 0.976, 0.964, 0.964 | 0.942–0.953 / 0.972–0.977 | 0.948–0.966 / 0.975–0.993 |

On the selection half every new model is below all four reference systems. On the calibration half C + B3 is close to
B7 + B3 and E1 + ASRF.

## 3. C − S

Paired bootstrap, 2,000 draws. Native clips are resampled by session and streams by stream. Streams reuse
recordings, so the stream intervals are approximate. Differences are in points.

| Seed, half | Segment F1@50, B3 | Boundary recall, B3 | Native macro-F1 | Segment F1@50, argmax | Boundary recall, argmax |
|---|---|---|---|---|---|
| s0 selection | +1.6 (−0.06 to +3.4) | +3.9 (+2.1 to +6.0) | 0.0 | +1.5 (+0.1 to +3.2) | +1.8 (+0.5 to +3.5) |
| s1 selection | +2.1 (+0.2 to +4.2) | +4.2 (+2.1 to +6.8) | +4.8 (−1.6 to +11.0) | +2.1 (+0.5 to +3.7) | +0.3 (−0.5 to +1.1) |
| s2 selection | +2.8 (+0.4 to +5.9) | +6.0 (+3.4 to +9.1) | +3.5 (−1.1 to +9.1) | +1.0 (−1.0 to +3.2) | +2.6 (+0.8 to +4.7) |
| s0 calibration | +1.6 (−0.2 to +3.6) | +3.5 (+1.7 to +5.6) | −0.3 (−0.9 to 0.0) | −0.7 (−1.7 to +0.2) | 0.0 (−0.7 to +0.8) |
| s1 calibration | +1.6 (+0.2 to +3.0) | +2.3 (+1.0 to +3.9) | −0.1 (−2.5 to +2.2) | 0.0 (−0.9 to +1.0) | +0.5 (0.0 to +1.3) |
| s2 calibration | +1.3 (−0.2 to +2.8) | +2.3 (+0.5 to +4.3) | −1.0 (−2.9 to 0.0) | −0.6 (−2.0 to +0.5) | +0.8 (−0.3 to +1.9) |

- **Segment F1@50 with B3.** Every point estimate favours C. The interval of seed 0 includes zero on both halves, and
  that of seed 2 on the calibration half. Under the rule, no consistent gain is detected on either half. This does not
  mean the two arms are equal.
- **Boundary recall with B3.** All six intervals lie above zero. In seeds 1 and 2 the B3 settings of C and S differ,
  because each was chosen per model or fell back to the diagnostic setting. Without a decoder, the argmax rows show a
  smaller recall difference that is not consistent across seeds. Part of the B3 recall gain may therefore come from
  the decoder settings.
- **Without a decoder** C leads on the selection half in all three seeds, with two intervals above zero. It does not
  lead on the calibration half.

## 4. What changed at the target time

**Window-type census.** This is the error at each window's target sample, in the windows B7 uses on the validation
streams, averaged over seeds. The recording-averaged error counts each recording once.

| Model | Mixed-view windows, selection | Mixed-view windows, calibration | One-recording windows (sel / cal) |
|---|---|---|---|
| N | 21.4% (13.5% by recording) | 17.0% (13.6%) | 5.0–5.6% / 1.6–2.5% |
| S | 12.2% (6.9%) | 6.5% (5.4%) | 4.8–6.1% / 1.7–1.9% |
| C | 7.3% (3.0%) | 3.3% (2.8%) | 4.3–4.7% / 1.7–2.1% |
| B7 (fixed, last round) | 15.0% (11.2%) | 11.7% (11.7%) | 4.2% / 1.5% |

In mixed windows, C's target-time error is 60% (selection) and 51% (calibration) of S's, and about a third and a fifth
of N's. Its errors in one-recording windows stay where they were.

**Paired context probe.** The probe reuses the 474 targets of the last round. It keeps a target recording's centre
frames and fills the rest of the window from a recording of the same view or of another view. The table gives the
accuracy with another view's context; with a same-view context every arm is near 99%. Values are for seeds 0, 1
and 2.

| Model | 8 of 16 frames from the target | 12 of 16 frames from the target |
|---|---|---|
| N | 0.48, 0.61, 0.56 | 0.95, 0.95, 0.90 |
| S | 0.57, 0.56, 0.59 | 0.98, 0.97, 0.98 |
| C | 0.86, 0.94, 0.89 | 0.98, 0.98, 0.98 |

- **At an even split C mostly keeps the target's view, while S and N mostly do not.** Unlike B7, the new head sees
  frame positions, and the C target uses them.
- **With 12 of 16 frames** C and S are equal within noise, and both are better than N.
- **With only 4 of 16 frames** every arm reports the context's view. Such short middle segments are shorter than any
  training fragment.

## 5. Native recognition

| Native clips, selection half | A5C recall | Clips with A5C read as A4C | SC4C read as PLAX |
|---|---|---|---|
| N, seeds 0-2 | 0.64 in each seed | 4, 4, 4 | 1, 1, 1 |
| S, seeds 0-2 | 0.64 in each seed | 4, 4, 4 | 1, 1, 1 |
| C, seeds 0-2 | 0.64, 0.82, 0.82 | 4, 2, 1 | 1, 1, 1 |
| E1 + B3 | 0.91 | 1 | 1 |
| B7 + B3 | 0.82 | 2 | 1 |

- **Against S.** C fixes 6 and 4 native clips on the selection half in seeds 1 and 2 and breaks 1 and 0, mostly
  A4C/A5C confusions. Seed 0 is unchanged. On the calibration half the changes are 0 to 1 fixed and 1 to 2 newly
  wrong per seed. No interval excludes zero.
- **Against E1 + B3, the frame-level system this encoder started from.**
  - On the selection half C fixes 0, 0 and 1 native clips and newly gets 5, 3 and 4 wrong. On the calibration half
    the changes are 0/2, 1/1 and 0/2.
  - Native recognition therefore regressed under this training recipe. The shared SC4C recording that the frame
    models read as PLAX stays wrong in every model.

## 6. Final goal

The same C + B3 would need a three-seed mean of at least 0.944 on the selection half and 0.982 on the calibration
half. In every seed and half it would also need boundary recall ≥ 0.95 and native macro-F1 ≥ 0.973 / 0.980.

- **Segment F1@50.** The means are 0.854 (−0.080 against 0.934) and 0.970 (−0.003 against 0.972).
- **Floors.** Recall is met everywhere (0.966 to 0.992). The selection native floor is missed in all seeds (0.922 to
  0.948). In seeds 0 and 2 B3 has no feasible setting, because calibration native macro-F1 is 0.975.
- **Sensitivity reference.** Against the later calibration system at 0.9739, C + B3 is 0.004 lower.

The goal is not met. The local improvement of C over S concerns the target time in mixed windows. It does not carry
the full system past the previous strongest configurations, and native recognition is below its E1 starting point.

## 7. Cost

- **Training.** About 1,700 s per model on one L40S (fp32), 4.3 hours in all.
- **Inference.** In the batched evaluation each model decoded and encoded 40,566 distinct frames for 71,503 validation
  samples in about 73 s. The window head took about 1.4 s, and B3 decoding with the metrics about 38 s. That is about
  1.5 ms per 10 Hz sample end to end, amortised over frames shared between overlapping windows. This is not a batch-1
  streaming latency and is not a speed claim.

## 8. What this round does not show

- **That target-time supervision cannot help.** C − S favours C in every cell, but no consistent gain is detected
  under the fixed rule. The round's recipe stops here, as planned.
- **Why native recognition regressed.** Unfreezing four blocks, the window budget and the soft or hard targets all
  changed together with the base. No ablation separates them.
- **Independent evidence.** Both validation halves were used in earlier development, and E1's upstream checkpoint was
  selected on the historical validation split. Constructed streams reuse recordings and are not real scans. The three
  seeds share one validation set.

## Files

- `report.json` holds every model's metrics for both rows and halves, its B3 grid and choice, the census, the probe,
  the C − S comparisons, the native changes, the references, the goal, the timing and the frozen paths.
- `report_tables.md` holds the generated tables.
- Server paths:
  - training runs `runs/center_supervision_v1/train/{arm}_s{seed}`;
  - manifests `runs/center_supervision_v1/manifests`;
  - frozen models `models_frozen/center_supervision_v1/{arm}_s{seed}`;
  - logs `logs/center_supervision_v1_*.log`.
