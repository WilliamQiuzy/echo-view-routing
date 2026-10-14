# Video token supervision round (2026-10-13)

This round asked whether class supervision of the video model's time tokens improves the view segmentation while it
keeps native recognition. Each seed's frozen Ordered model of the raw-video context round (V01) was trained further in
three paired arms. Every arm keeps the original CLS output, the 512-dimensional projection, the five-class head, the
marker, the 16-frame input and the stride-5 inference. During training, every arm also averages each of the 8 time
positions of the final normalised tokens over space and reads them with the shared head. The arms differ only in how
these 8 time logits are supervised:

- **CLS continuation** (`center_only`) gives them weight 0.
- **Aligned token supervision** (`aligned_dense`, the primary candidate) uses the classes at the true input time anchors.
- **Shuffled token supervision** (`shuffled_dense`) uses the same classes permuted within each window.

Inference uses only the CLS output with B3. No parameter is added. This tests a training objective on an existing
backbone (MViTv2); it is not a new architecture. Per-frame auxiliary supervision was tried before (R11, on E1 image
features before a temporal Transformer), but not on the MViT tokens after video interaction. The task sheet is
`docs/superpowers/plans/2026-10-13-video-dense-supervision.md`.

**What kind of evidence this is.**
- The development halves have been used in many earlier rounds, so this evaluation is not an independent validation.
- The hold-out is by session, not by patient.
- Constructed streams are spliced from separate recordings and are not real scans. They reuse recordings, so their
  units are not independent cases. The three seeds share the same cases.
- Native clips carry the public recording label, not expert frame labels. The auxiliary labels come from these public
  labels and the constructed streams, not from expert frame annotation. No label was changed.
- "Aligned" means the class of the frame at each token's anchor slot (the centre of the stem's three time slots). A
  review measurement found that this slot is not the frame that most drives the token (see "Terms").
- The anchor diagnostics read the time logits against reference labels. They are diagnostics, not systems or achievable
  segmentation scores.
- The test split and EchoNet-Pediatric were not read.

**Order of work.** All times are UTC on 2026-10-14.
- bad13a6 committed the task sheet and the pending index entry A02 at 02:04. 8ca5fc5 committed the fixed configuration
  before any code.
- 471f8bf committed the data tests, which failed at import, and 56f5279 the preparation. 3bc7c89 committed the model
  tests, which failed at import, and ef2c938 the model module.
- cd0d4c0 committed the training entry, the checks and the queue. Their tests were written before them but committed
  with them. fab6e6b committed the report tests, the fixed readings and the evaluation code before any development output.
- The preparation ran twice, at 02:17 and at 02:32. The second run is the formal record (see "Deviations").
- The first check passed from 02:35 to 02:40. An independent implementation review then passed the code with fixes
  (`code_review.md`). f120f94 made the fixes, and the second check passed from 03:05 to 03:10.
- d235c0b committed the review, preparation and check records. 940e4f8 kept the check's frame figure on the server.
- The nine trainings ran from 03:12 to 07:28, and the queue froze the nine final weights.
- The single evaluation ran from 07:29 to 08:29. The server was then unreachable until 15:05.
- An independent fact-check recomputed the results (see "Fact-check").

## Answers

Values are given for seeds 0, 1 and 2 in that order. A range "a to b" gives the smallest and the largest value over
seeds, and over halves where the sentence says so. A difference is consistent only when all three seeds' 95% intervals
exclude zero strictly on the same side. All systems use B3 unless the sentence says otherwise.

1. **Does aligned token supervision improve the segmentation over CLS continuation?** No consistent difference was
   detected in either half.
   - **The primary metric.** Aligned − CLS continuation multi-stream segment F1@50 is +0.011, −0.003 and +0.034 on
     selection and +0.003, +0.003 and +0.004 on calibration. Only seed 2's selection interval excludes zero, and every
     calibration interval contains zero.
   - **Other metrics.** None of the five other paired metrics shows a consistent difference either. They are native
     macro-F1, label switches per native clip, boundary recall, boundary F1 and cuts at same-view joins.
2. **Does it beat the strong reference or the shuffled control?** No consistent difference was detected against either.
   - **Aligned − frozen Ordered + B3.** The differences are +0.004, −0.010 and −0.003 on selection and +0.004, −0.004
     and 0.000 on calibration. Seed 1's selection interval lies below zero, but the others contain zero.
   - **Aligned − shuffled.** The differences are +0.004, +0.011 and +0.001 on selection and −0.001, +0.003 and −0.001 on
     calibration.
   - **The other within-round comparisons.** CLS continuation − frozen Ordered, shuffled − CLS continuation and aligned −
     historical Continued CLS show no consistent difference either. Neither do the direct argmax rows or the rows
     decoded with the seed's Ordered numeric B3 setting.
3. **Did the time tokens learn the aligned labels?** Yes, at the token level, as a diagnostic.
   - **Anchor accuracy.** On mixed development windows, the aligned arm's time logits name the reference class at 0.845
     to 0.857 of the anchors on selection and 0.860 to 0.879 on calibration. The shuffled arm reaches 0.695 to 0.715 and
     CLS continuation 0.650 to 0.699, over seeds and halves.
   - **View changes inside a window.** At 1,415 selection and 1,377 calibration changes of the reference class between
     consecutive anchors, the aligned arm gets both anchors right at 404 to 444 and 417 to 505 changes. The other two
     arms get both right at 4 to 21. These changes come from 238 recordings and 36 sessions on selection and 222
     recordings and 35 sessions on calibration. CLS continuation's calibration counts in seeds 0 and 2 (7 and 4 changes)
     rest on fewer than 10 sessions (8 and 6).
   - **What this means.** The auxiliary objective was learned and carries over to development windows at the token level.
     No consistent change in the CLS target decisions was detected (answer 4).
4. **Did the CLS target decisions on mixed windows change?** No consistent difference from CLS continuation was
   detected.
   - **Aligned − CLS continuation.** On all mixed windows, the differences in target accuracy are −0.009, −0.012 and
     −0.008 on selection and +0.009, −0.003 and +0.002 on calibration. On selection, two seeds' intervals lie below
     zero, but not all three, so no consistent difference was detected. On the windows where the requested view covers
     8 of 16 frames, the differences are −0.043, −0.064 and −0.053 on selection and 0.000, −0.082 and −0.068 on
     calibration, where two seeds' intervals lie below zero; again neither half is consistent.
   - **Aligned − shuffled.** No consistent difference was detected on any mixed subset.
   - **Continued training itself.** CLS continuation − frozen Ordered is consistent positive on all mixed windows in both
     halves (+0.015 to +0.033 over seeds and halves) and on the 8-of-16 windows (+0.096 to +0.219). Aligned − frozen
     Ordered is consistent positive on calibration only (+0.025 to +0.030). V02 found the same continued-training gain
     on mixed windows. In neither round was a consistent segment F1@50 difference against frozen Ordered + B3 detected.
5. **Native recognition.** The aligned arm keeps the native floors in every seed and half.
   - **Native macro-F1.** Aligned has 0.9786, 0.9877 and 0.9786 on selection and 0.9929, 0.9861 and 0.9929 on
     calibration.
   - **Changed clips.** Against CLS continuation, aligned gets one A4C clip right that CLS continuation reads as A5C, in
     seeds 1 and 2 on selection and seed 2 on calibration. It newly misreads none. With B3, no system of this round reads
     an A5C clip as A4C.
6. **Constraints and goal.** No system meets the goal.
   - **Three-seed means.** Aligned has 0.9515 and 0.9817, shuffled 0.9463 and 0.9814, CLS continuation 0.9374 and
     0.9786, and frozen Ordered + B3 0.9544 and 0.9817. The minimums are 0.9444 and 0.9821. Aligned misses the
     calibration minimum by 0.0004.
   - **Constraint cells.** Aligned and shuffled have a feasible B3 setting, boundary recall of at least 0.95 and native
     macro-F1 at the floor in every seed and half. CLS continuation fails two cells: seed 0's selection boundary recall
     (0.935) and seed 2's selection native macro-F1 (0.9701). Frozen Ordered + B3 fails one cell, seed 2's selection
     native macro-F1 (0.9701).
7. **Against weaker references.** Aligned is consistent positive against E1 + B3, Original FACT and the historical E1 +
   ASRF in both halves. It is consistent positive against E0 + B3 on calibration only, and against B7 + B3 and the
   paired ASRF on selection only. In segment F1@50 it shows no consistent difference against A01's Linear + B3, MS-TCN
   and ASRF; its boundary recall is consistent positive against A01 ASRF on calibration only. These
   gains over weaker systems do not show progress beyond the frozen Ordered + B3 reference, which the task sheet requires.
8. **The fixed decision.** Two readings apply: no gain and end.
   - **No gain.** Aligned − CLS continuation is consistent positive on neither all-mixed accuracy nor segment F1@50 in
     either half, and aligned − frozen Ordered + B3 is not consistent positive in either half. This configuration ends.
     Dense supervision in general is not rejected.
   - **End.** No teacher, boundary loss, gate, new head, lambda sweep, more epochs or seeds, or sampling change follows.
   - **Not applying.** Model progress, mechanism support, system only, shuffled only and trade-off do not apply. No half
     had a stream-sensitivity conflict. The test split stays sealed.
9. **Cost.** The round used 5.44 GPU hours of its 8-hour cap, including the fact-check.
   - **Checks.** The two checks took 0.083 hours each.
   - **Training.** The nine trainings took 4.27 hours, 0.473 to 0.475 hours each.
   - **Evaluation.** It took 0.99 hours. Inference took 16.3 to 16.4 ms per window for 14,687 windows per model.
   - **Fact-check.** Its re-inference took 0.011 GPU hours.

## Terms

- **The model.** Every arm runs the original TargetMViT forward, so its CLS logits are the model's output. A hook on the
  final normalisation stores a copy of its output, with gradients, and changes nothing. The 8 time positions of the
  final tokens (8 x 7 x 7 patch tokens) are averaged over space and read by the shared dropout, projection and head. In
  every arm these time logits are computed after the CLS logits, so all arms use the random number generator alike.
- **The loss** is (L_CLS + lambda L_time) / (1 + lambda). L_CLS is the CLS cross-entropy of the window's target and L_time
  the mean cross-entropy over the 8 anchors. Lambda is 0 for CLS continuation and 0.5 for the aligned and shuffled arms.
- **The anchors** are input slots 0, 2, ..., 14, the centres of the stem's time kernel (kernel 3, stride 2, padding 1).
  Each final time position has interacted with every other position, so it is not a local per-frame feature. The
  implementation review measured, on seed 0, that a token's anchor slot carries 5 to 7% of its input sensitivity,
  against 11 to 16% for the next slot (2j + 1). In about 53% of the mixed training windows, at least one anchor's label
  differs from the label of the slot after it. "Aligned" therefore means the anchor slot's class, not the class of the
  frame that most drives the token. The aligned and shuffled arms share this rule and the label multiset.
- **The shuffled labels** permute each window's 8 aligned labels once with a fixed per-window generator. The multiset
  stays the same. Native windows keep their labels, and 13,402 to 13,444 of the 15,000 mixed training windows per seed
  have changed labels.
- **Frozen Ordered + B3** is V01's saved output with its stored B3 settings. **Historical Continued CLS** is V02's saved
  output with its stored settings. Both reproduce their stored rows exactly.
- **The references** keep their own inputs and stored settings, and every one was reproduced before use.
  - E0 + B3, E1 + B3 and B7 + B3 are the image and window classifiers with B3.
  - Historical E1 + ASRF is the ASRF of model-development round 2.
  - Paired ASRF and Original FACT come from the paired FACT-ASRF round.
  - A01's Linear + B3, MS-TCN and ASRF are its frozen heads on its saved features.
  - G1 + ASRF supplies the calibration sensitivity target only.
- **B3** is chosen once per model on calibration from 40 settings. The choice needs native macro-F1 of at least 0.980
  and boundary recall of at least 0.95, takes the highest segment F1@50 and is then applied to selection. Every model of
  this round had a feasible setting. The **fixed rows** decode each arm with the seed's Ordered numeric setting
  (half-window 9) as a decoding sensitivity.
- **Mixed windows** are windows of constructed streams with at least two reference views among their 16 frames. Their
  target accuracy is the CLS argmax at the requested sample. Their paired bootstrap resamples the source sessions of the
  requested samples (primary) or whole streams (sensitivity).
- A **unit** is a reference segment of a constructed stream, matched at IoU 0.5 by a predicted segment of its class.
  **Segments of at most 16 samples** last at most 1.6 seconds. An **X-Y-X middle** is the middle segment of three
  consecutive reference segments whose outer two share a class.
- The **paired bootstrap** of the systems resamples native clips by session and streams by stream, with 2,000 draws,
  seed 20261002, and the same draws for both systems.

## 1. Systems

Each cell gives the selection half, then the calibration half.

| System (B3) | Seed | Segment F1@50 | Boundary recall | Boundary precision | Native macro-F1 | Label switches per native clip | Cuts at same-view joins | A4C read as A5C (clips) | A5C read as A4C (clips) |
|---|---|---|---|---|---|---|---|---|---|
| CLS continuation | 0 | 0.931 / 0.983 | 0.935 / 0.985 | 0.937 / 0.980 | 0.9786 / 0.9929 | 0.039 / 0.025 | 0.026 / 0.015 | 2 / 1 | 0 / 0 |
| CLS continuation | 1 | 0.954 / 0.977 | 0.974 / 0.990 | 0.942 / 0.975 | 0.9786 / 0.9861 | 0.049 / 0.025 | 0.009 / 0.025 | 2 / 2 | 0 / 0 |
| CLS continuation | 2 | 0.927 / 0.976 | 0.984 / 0.987 | 0.887 / 0.966 | 0.9701 / 0.9861 | 0.092 / 0.059 | 0.026 / 0.030 | 3 / 2 | 0 / 0 |
| Aligned token supervision | 0 | 0.942 / 0.986 | 0.958 / 0.995 | 0.953 / 0.978 | 0.9786 / 0.9929 | 0.042 / 0.017 | 0.022 / 0.025 | 2 / 1 | 0 / 0 |
| Aligned token supervision | 1 | 0.951 / 0.980 | 0.966 / 0.990 | 0.941 / 0.973 | 0.9877 / 0.9861 | 0.035 / 0.013 | 0.017 / 0.020 | 1 / 2 | 0 / 0 |
| Aligned token supervision | 2 | 0.962 / 0.980 | 0.976 / 0.985 | 0.954 / 0.982 | 0.9786 / 0.9929 | 0.060 / 0.017 | 0.022 / 0.020 | 2 / 1 | 0 / 0 |
| Shuffled token supervision | 0 | 0.939 / 0.987 | 0.958 / 0.995 | 0.943 / 0.975 | 0.9786 / 0.9929 | 0.042 / 0.017 | 0.022 / 0.025 | 2 / 1 | 0 / 0 |
| Shuffled token supervision | 1 | 0.940 / 0.977 | 0.979 / 0.990 | 0.908 / 0.966 | 0.9786 / 0.9861 | 0.067 / 0.046 | 0.022 / 0.025 | 2 / 2 | 0 / 0 |
| Shuffled token supervision | 2 | 0.960 / 0.981 | 0.969 / 0.985 | 0.959 / 0.985 | 0.9786 / 0.9929 | 0.049 / 0.017 | 0.017 / 0.015 | 2 / 1 | 0 / 0 |
| frozen Ordered + B3 | 0 | 0.938 / 0.982 | 0.958 / 0.982 | 0.948 / 0.975 | 0.9786 / 0.9929 | 0.042 / 0.025 | 0.017 / 0.025 | 2 / 1 | 0 / 0 |
| frozen Ordered + B3 | 1 | 0.960 / 0.984 | 0.969 / 0.987 | 0.959 / 0.980 | 0.9786 / 0.9861 | 0.039 / 0.017 | 0.004 / 0.025 | 2 / 2 | 0 / 0 |
| frozen Ordered + B3 | 2 | 0.964 / 0.980 | 0.974 / 0.980 | 0.959 / 0.977 | 0.9701 / 0.9861 | 0.046 / 0.013 | 0.004 / 0.025 | 3 / 2 | 0 / 0 |
| historical Continued CLS | 0 | 0.943 / 0.981 | 0.971 / 0.987 | 0.932 / 0.978 | 0.9786 / 0.9929 | 0.042 / 0.030 | 0.022 / 0.020 | 2 / 1 | 0 / 0 |
| historical Continued CLS | 1 | 0.955 / 0.976 | 0.974 / 0.987 | 0.944 / 0.970 | 0.9786 / 0.9861 | 0.067 / 0.021 | 0.013 / 0.030 | 2 / 2 | 0 / 0 |
| historical Continued CLS | 2 | 0.935 / 0.976 | 0.984 / 0.987 | 0.900 / 0.966 | 0.9701 / 0.9861 | 0.081 / 0.059 | 0.022 / 0.030 | 3 / 2 | 0 / 0 |

The B3 choices are half-window 5, 7 and 5 for CLS continuation and 7, 7 and 5 for the aligned and shuffled arms, at
quantile 0.999 except CLS continuation seed 2 (0.995) and shuffled seed 1 (0.99). Direct argmax and the seed's Ordered
numeric setting give these rows (diagnostic).

| System | Seed | Segment F1@50 | Boundary recall | Boundary precision | Native macro-F1 | Label switches per native clip | Cuts at same-view joins | A4C read as A5C (clips) | A5C read as A4C (clips) |
|---|---|---|---|---|---|---|---|---|---|
| CLS continuation, argmax | 0 | 0.893 / 0.964 | 0.974 / 0.987 | 0.838 / 0.945 | 0.9877 / 0.9929 | 0.116 / 0.118 | 0.052 / 0.035 | 1 / 1 | 0 / 0 |
| CLS continuation, argmax | 1 | 0.941 / 0.970 | 0.992 / 0.992 | 0.905 / 0.961 | 0.9877 / 0.9929 | 0.113 / 0.110 | 0.017 / 0.035 | 1 / 1 | 0 / 0 |
| CLS continuation, argmax | 2 | 0.906 / 0.971 | 0.984 / 0.990 | 0.855 / 0.954 | 0.9877 / 0.9788 | 0.120 / 0.127 | 0.030 / 0.035 | 1 / 2 | 0 / 1 |
| Aligned token supervision, argmax | 0 | 0.897 / 0.973 | 0.971 / 0.995 | 0.845 / 0.952 | 0.9877 / 0.9929 | 0.095 / 0.076 | 0.052 / 0.030 | 1 / 1 | 0 / 0 |
| Aligned token supervision, argmax | 1 | 0.914 / 0.970 | 0.987 / 0.992 | 0.855 / 0.950 | 0.9786 / 0.9796 | 0.099 / 0.076 | 0.035 / 0.035 | 2 / 3 | 0 / 0 |
| Aligned token supervision, argmax | 2 | 0.911 / 0.971 | 0.984 / 0.990 | 0.849 / 0.954 | 0.9877 / 0.9929 | 0.134 / 0.084 | 0.030 / 0.035 | 1 / 1 | 0 / 0 |
| Shuffled token supervision, argmax | 0 | 0.900 / 0.975 | 0.974 / 0.995 | 0.845 / 0.957 | 0.9877 / 0.9929 | 0.099 / 0.059 | 0.048 / 0.030 | 1 / 1 | 0 / 0 |
| Shuffled token supervision, argmax | 1 | 0.912 / 0.970 | 0.987 / 0.995 | 0.849 / 0.957 | 0.9786 / 0.9929 | 0.099 / 0.097 | 0.035 / 0.030 | 2 / 1 | 0 / 0 |
| Shuffled token supervision, argmax | 2 | 0.908 / 0.970 | 0.984 / 0.990 | 0.851 / 0.952 | 0.9786 / 0.9929 | 0.120 / 0.101 | 0.030 / 0.035 | 2 / 1 | 0 / 0 |
| CLS continuation, Ordered setting | 0 | 0.941 / 0.981 | 0.969 / 0.987 | 0.925 / 0.973 | 0.9786 / 0.9929 | 0.042 / 0.025 | 0.026 / 0.030 | 2 / 1 | 0 / 0 |
| CLS continuation, Ordered setting | 1 | 0.955 / 0.974 | 0.976 / 0.990 | 0.942 / 0.968 | 0.9786 / 0.9861 | 0.042 / 0.025 | 0.009 / 0.025 | 2 / 2 | 0 / 0 |
| CLS continuation, Ordered setting | 2 | 0.955 / 0.975 | 0.982 / 0.985 | 0.938 / 0.970 | 0.9786 / 0.9861 | 0.056 / 0.025 | 0.017 / 0.030 | 2 / 2 | 0 / 0 |
| Aligned token supervision, Ordered setting | 0 | 0.945 / 0.985 | 0.955 / 0.990 | 0.963 / 0.980 | 0.9786 / 0.9929 | 0.035 / 0.008 | 0.013 / 0.020 | 2 / 1 | 0 / 0 |
| Aligned token supervision, Ordered setting | 1 | 0.957 / 0.979 | 0.961 / 0.987 | 0.968 / 0.973 | 0.9877 / 0.9861 | 0.028 / 0.013 | 0.004 / 0.020 | 1 / 2 | 0 / 0 |
| Aligned token supervision, Ordered setting | 2 | 0.962 / 0.981 | 0.963 / 0.985 | 0.968 / 0.985 | 0.9786 / 0.9929 | 0.035 / 0.013 | 0.009 / 0.015 | 2 / 1 | 0 / 0 |
| Shuffled token supervision, Ordered setting | 0 | 0.943 / 0.984 | 0.953 / 0.990 | 0.955 / 0.978 | 0.9786 / 0.9929 | 0.025 / 0.013 | 0.013 / 0.025 | 2 / 1 | 0 / 0 |
| Shuffled token supervision, Ordered setting | 1 | 0.959 / 0.980 | 0.961 / 0.987 | 0.963 / 0.975 | 0.9877 / 0.9861 | 0.028 / 0.013 | 0.009 / 0.020 | 1 / 2 | 0 / 0 |
| Shuffled token supervision, Ordered setting | 2 | 0.962 / 0.979 | 0.963 / 0.982 | 0.968 / 0.982 | 0.9786 / 0.9929 | 0.039 / 0.013 | 0.009 / 0.020 | 2 / 1 | 0 / 0 |

References, each with its own inputs and stored settings.

| System | Seed | Segment F1@50 | Boundary recall | Boundary precision | Native macro-F1 | Label switches per native clip | Cuts at same-view joins | A4C read as A5C (clips) | A5C read as A4C (clips) |
|---|---|---|---|---|---|---|---|---|---|
| E0 + B3 | - | 0.934 / 0.934 | 0.961 / 0.955 | 0.995 / 0.969 | 0.9650 / 0.9660 | 0.014 / 0.034 | 0.004 / 0.020 | 0 / 0 | 3 / 4 |
| E1 + B3 | - | 0.894 / 0.959 | 0.955 / 0.977 | 0.943 / 0.951 | 0.9780 / 0.9856 | 0.046 / 0.068 | 0.039 / 0.020 | 0 / 1 | 1 / 1 |
| B7 + B3 | - | 0.880 / 0.972 | 0.955 / 0.975 | 0.890 / 0.963 | 0.9566 / 0.9927 | 0.063 / 0.042 | 0.057 / 0.035 | 1 / 0 | 2 / 1 |
| historical E1 + ASRF | 0 | 0.910 / 0.976 | 0.942 / 0.977 | 0.952 / 0.980 | 0.9477 / 0.9929 | 0.046 / 0.038 | 0.030 / 0.015 | 2 / 1 | 2 / 0 |
| historical E1 + ASRF | 1 | 0.909 / 0.964 | 0.950 / 0.972 | 0.936 / 0.970 | 0.9664 / 0.9753 | 0.007 / 0.030 | 0.057 / 0.030 | 0 / 1 | 2 / 2 |
| historical E1 + ASRF | 2 | 0.907 / 0.964 | 0.953 / 0.977 | 0.933 / 0.970 | 0.9664 / 0.9753 | 0.021 / 0.046 | 0.057 / 0.035 | 0 / 1 | 2 / 2 |
| paired ASRF | 0 | 0.878 / 0.972 | 0.935 / 0.987 | 0.927 / 0.975 | 0.9304 / 0.9856 | 0.011 / 0.004 | 0.065 / 0.035 | 1 / 1 | 4 / 1 |
| paired ASRF | 1 | 0.888 / 0.972 | 0.940 / 0.985 | 0.937 / 0.978 | 0.9537 / 0.9856 | 0.007 / 0.004 | 0.048 / 0.035 | 0 / 1 | 3 / 1 |
| paired ASRF | 2 | 0.882 / 0.966 | 0.942 / 0.982 | 0.942 / 0.980 | 0.9537 / 0.9856 | 0.011 / 0.004 | 0.065 / 0.030 | 0 / 1 | 3 / 1 |
| Original FACT | 0 | 0.906 / 0.967 | 0.948 / 0.982 | 0.926 / 0.968 | 0.9886 / 0.9856 | 0.000 / 0.034 | 0.052 / 0.025 | 0 / 1 | 0 / 1 |
| Original FACT | 1 | 0.898 / 0.961 | 0.950 / 0.975 | 0.928 / 0.968 | 0.9780 / 0.9856 | 0.000 / 0.030 | 0.043 / 0.030 | 0 / 1 | 1 / 1 |
| Original FACT | 2 | 0.911 / 0.965 | 0.940 / 0.972 | 0.925 / 0.972 | 0.9664 / 0.9929 | 0.000 / 0.034 | 0.035 / 0.025 | 0 / 1 | 2 / 0 |
| A01 Linear + B3 | 0 | 0.947 / 0.977 | 0.955 / 0.975 | 0.953 / 0.975 | 0.9786 / 0.9861 | 0.046 / 0.013 | 0.026 / 0.020 | 2 / 2 | 0 / 0 |
| A01 Linear + B3 | 1 | 0.957 / 0.978 | 0.953 / 0.980 | 0.971 / 0.973 | 0.9786 / 0.9861 | 0.039 / 0.017 | 0.009 / 0.015 | 2 / 2 | 0 / 0 |
| A01 Linear + B3 | 2 | 0.939 / 0.973 | 0.971 / 0.972 | 0.912 / 0.953 | 0.9786 / 0.9861 | 0.085 / 0.042 | 0.030 / 0.030 | 2 / 2 | 0 / 0 |
| A01 MS-TCN | 0 | 0.920 / 0.967 | 0.966 / 0.962 | 0.918 / 0.962 | 0.9626 / 0.9780 | 0.074 / 0.038 | 0.035 / 0.025 | 0 / 1 | 3 / 2 |
| A01 MS-TCN | 1 | 0.953 / 0.974 | 0.963 / 0.977 | 0.951 / 0.975 | 0.9771 / 0.9780 | 0.042 / 0.051 | 0.017 / 0.025 | 1 / 1 | 1 / 2 |
| A01 MS-TCN | 2 | 0.943 / 0.963 | 0.958 / 0.970 | 0.938 / 0.953 | 0.9753 / 0.9780 | 0.056 / 0.055 | 0.013 / 0.035 | 0 / 1 | 2 / 2 |
| A01 ASRF | 0 | 0.937 / 0.980 | 0.948 / 0.970 | 0.976 / 0.992 | 0.9869 / 0.9780 | 0.014 / 0.008 | 0.035 / 0.015 | 0 / 1 | 1 / 2 |
| A01 ASRF | 1 | 0.948 / 0.975 | 0.948 / 0.972 | 0.978 / 0.992 | 0.9771 / 0.9723 | 0.004 / 0.008 | 0.030 / 0.010 | 1 / 3 | 1 / 1 |
| A01 ASRF | 2 | 0.951 / 0.970 | 0.945 / 0.957 | 0.978 / 0.984 | 0.9869 / 0.9929 | 0.000 / 0.008 | 0.030 / 0.015 | 0 / 1 | 1 / 0 |

## 2. Comparisons

Paired differences with B3 and 95% intervals. Streams reuse recordings, so the intervals are approximate. "none" means
that no consistent difference was detected, which does not mean that the systems are equal.

| Comparison | Half | Segment F1@50 (seeds 0, 1, 2) | Verdict | Boundary recall (seeds 0, 1, 2) | Verdict | Other consistent differences |
|---|---|---|---|---|---|---|
| aligned − center_only | selection | +0.0111 [-0.0061, +0.0276]; -0.0031 [-0.0161, +0.0114]; +0.0342 [+0.0158, +0.0544] | none | +0.0236 [+0.0028, +0.0442]; -0.0079 [-0.0181, +0.0000]; -0.0079 [-0.0203, +0.0027] | none | none |
| aligned − center_only | calibration | +0.0030 [-0.0009, +0.0073]; +0.0029 [-0.0020, +0.0090]; +0.0037 [-0.0057, +0.0137] | none | +0.0101 [+0.0024, +0.0207]; +0.0000 [-0.0100, +0.0099]; -0.0025 [-0.0108, +0.0052] | none | none |
| aligned − Ordered | selection | +0.0039 [-0.0125, +0.0199]; -0.0096 [-0.0208, -0.0006]; -0.0029 [-0.0188, +0.0131] | none | +0.0000 [-0.0154, +0.0132]; -0.0026 [-0.0154, +0.0081]; +0.0026 [-0.0126, +0.0187] | none | none |
| aligned − Ordered | calibration | +0.0039 [-0.0010, +0.0096]; -0.0038 [-0.0106, +0.0020]; +0.0000 [-0.0074, +0.0067] | none | +0.0126 [+0.0026, +0.0243]; +0.0025 [-0.0052, +0.0106]; +0.0050 [-0.0049, +0.0155] | none | none |
| aligned − shuffled | selection | +0.0037 [+0.0000, +0.0084]; +0.0107 [-0.0039, +0.0252]; +0.0012 [-0.0106, +0.0131] | none | +0.0000 [-0.0079, +0.0075]; -0.0131 [-0.0282, -0.0026]; +0.0079 [+0.0000, +0.0213] | none | none |
| aligned − shuffled | calibration | -0.0010 [-0.0030, +0.0000]; +0.0028 [+0.0000, +0.0064]; -0.0009 [-0.0087, +0.0058] | none | +0.0000 [+0.0000, +0.0000]; +0.0000 [+0.0000, +0.0000]; +0.0000 [+0.0000, +0.0000] | none | none |
| center_only − Ordered | selection | -0.0073 [-0.0290, +0.0140]; -0.0065 [-0.0210, +0.0062]; -0.0370 [-0.0614, -0.0153] | none | -0.0236 [-0.0485, +0.0000]; +0.0052 [-0.0077, +0.0182]; +0.0105 [+0.0000, +0.0247] | none | none |
| center_only − Ordered | calibration | +0.0009 [-0.0031, +0.0052]; -0.0067 [-0.0147, +0.0001]; -0.0037 [-0.0124, +0.0031] | none | +0.0025 [-0.0051, +0.0124]; +0.0025 [-0.0052, +0.0120]; +0.0075 [+0.0000, +0.0177] | none | none |
| shuffled − center_only | selection | +0.0074 [-0.0091, +0.0238]; -0.0137 [-0.0273, -0.0022]; +0.0330 [+0.0132, +0.0539] | none | +0.0236 [+0.0026, +0.0456]; +0.0052 [-0.0055, +0.0186]; -0.0157 [-0.0328, -0.0025] | none | none |
| shuffled − center_only | calibration | +0.0039 [+0.0000, +0.0089]; +0.0001 [-0.0048, +0.0059]; +0.0046 [-0.0030, +0.0140] | none | +0.0101 [+0.0024, +0.0207]; +0.0000 [-0.0100, +0.0099]; -0.0025 [-0.0108, +0.0052] | none | none |
| aligned − Continued CLS | selection | -0.0008 [-0.0165, +0.0150]; -0.0040 [-0.0178, +0.0102]; +0.0269 [+0.0128, +0.0433] | none | -0.0131 [-0.0298, +0.0000]; -0.0079 [-0.0181, +0.0000]; -0.0079 [-0.0203, +0.0027] | none | none |
| aligned − Continued CLS | calibration | +0.0048 [+0.0010, +0.0107]; +0.0038 [-0.0009, +0.0097]; +0.0037 [-0.0052, +0.0137] | none | +0.0075 [+0.0000, +0.0172]; +0.0025 [-0.0052, +0.0106]; -0.0025 [-0.0119, +0.0051] | none | none |
| aligned − (E1 + B3) | selection | +0.0484 [+0.0185, +0.0796]; +0.0567 [+0.0256, +0.0881]; +0.0675 [+0.0378, +0.0993] | consistent positive | +0.0026 [-0.0190, +0.0248]; +0.0105 [-0.0108, +0.0338]; +0.0209 [+0.0000, +0.0417] | none | none |
| aligned − (E1 + B3) | calibration | +0.0267 [+0.0143, +0.0389]; +0.0209 [+0.0101, +0.0315]; +0.0208 [+0.0063, +0.0342] | consistent positive | +0.0176 [+0.0047, +0.0329]; +0.0126 [-0.0026, +0.0293]; +0.0075 [-0.0077, +0.0237] | none | native label switches consistent negative, boundary F1 consistent positive |
| aligned − (E0 + B3) | selection | +0.0079 [-0.0156, +0.0311]; +0.0163 [-0.0069, +0.0408]; +0.0271 [+0.0028, +0.0537] | none | -0.0026 [-0.0213, +0.0161]; +0.0052 [-0.0132, +0.0255]; +0.0157 [-0.0025, +0.0332] | none | none |
| aligned − (E0 + B3) | calibration | +0.0516 [+0.0308, +0.0747]; +0.0459 [+0.0244, +0.0702]; +0.0458 [+0.0240, +0.0703] | consistent positive | +0.0402 [+0.0190, +0.0623]; +0.0352 [+0.0129, +0.0585]; +0.0302 [+0.0075, +0.0536] | consistent positive | boundary F1 consistent positive |
| aligned − (B7 + B3) | selection | +0.0625 [+0.0353, +0.0894]; +0.0708 [+0.0423, +0.0995]; +0.0817 [+0.0550, +0.1094] | consistent positive | +0.0026 [-0.0190, +0.0247]; +0.0105 [-0.0104, +0.0325]; +0.0209 [+0.0027, +0.0411] | none | boundary F1 consistent positive, cuts at same-view joins consistent negative |
| aligned − (B7 + B3) | calibration | +0.0135 [+0.0028, +0.0264]; +0.0077 [-0.0039, +0.0211]; +0.0076 [-0.0066, +0.0226] | none | +0.0201 [+0.0074, +0.0365]; +0.0151 [+0.0000, +0.0320]; +0.0101 [-0.0074, +0.0287] | none | none |
| aligned − Original FACT | selection | +0.0361 [+0.0095, +0.0620]; +0.0524 [+0.0218, +0.0832]; +0.0506 [+0.0219, +0.0819] | consistent positive | +0.0105 [-0.0099, +0.0324]; +0.0157 [-0.0054, +0.0394]; +0.0366 [+0.0154, +0.0607] | none | native label switches consistent positive |
| aligned − Original FACT | calibration | +0.0182 [+0.0040, +0.0331]; +0.0193 [+0.0049, +0.0361]; +0.0145 [+0.0020, +0.0275] | consistent positive | +0.0126 [+0.0025, +0.0238]; +0.0151 [+0.0000, +0.0308]; +0.0126 [-0.0026, +0.0293] | none | none |
| aligned − paired ASRF | selection | +0.0645 [+0.0351, +0.0946]; +0.0631 [+0.0334, +0.0948]; +0.0790 [+0.0470, +0.1113] | consistent positive | +0.0236 [-0.0049, +0.0519]; +0.0262 [+0.0025, +0.0506]; +0.0340 [+0.0104, +0.0590] | none | cuts at same-view joins consistent negative |
| aligned − paired ASRF | calibration | +0.0135 [+0.0019, +0.0254]; +0.0078 [-0.0028, +0.0200]; +0.0135 [+0.0019, +0.0268] | none | +0.0075 [-0.0026, +0.0192]; +0.0050 [-0.0073, +0.0174]; +0.0025 [-0.0105, +0.0154] | none | none |
| aligned − historical E1 + ASRF | selection | +0.0323 [+0.0046, +0.0587]; +0.0418 [+0.0110, +0.0725]; +0.0544 [+0.0255, +0.0841] | consistent positive | +0.0157 [-0.0088, +0.0418]; +0.0157 [-0.0105, +0.0420]; +0.0236 [+0.0000, +0.0479] | none | none |
| aligned − historical E1 + ASRF | calibration | +0.0098 [+0.0010, +0.0195]; +0.0155 [+0.0029, +0.0295]; +0.0154 [+0.0011, +0.0308] | consistent positive | +0.0176 [+0.0052, +0.0308]; +0.0176 [+0.0024, +0.0357]; +0.0075 [-0.0050, +0.0210] | none | none |
| aligned − A01 ASRF | selection | +0.0058 [-0.0104, +0.0222]; +0.0031 [-0.0223, +0.0310]; +0.0109 [-0.0043, +0.0267] | none | +0.0105 [-0.0082, +0.0290]; +0.0183 [+0.0000, +0.0408]; +0.0314 [+0.0028, +0.0587] | none | none |
| aligned − A01 ASRF | calibration | +0.0061 [-0.0065, +0.0178]; +0.0052 [-0.0087, +0.0199]; +0.0100 [-0.0017, +0.0247] | none | +0.0251 [+0.0101, +0.0430]; +0.0176 [+0.0025, +0.0344]; +0.0276 [+0.0102, +0.0480] | consistent positive | none |
| aligned − A01 MS-TCN | selection | +0.0224 [+0.0067, +0.0396]; -0.0027 [-0.0186, +0.0132]; +0.0188 [+0.0020, +0.0367] | none | -0.0079 [-0.0256, +0.0099]; +0.0026 [-0.0163, +0.0211]; +0.0183 [-0.0027, +0.0399] | none | none |
| aligned − A01 MS-TCN | calibration | +0.0184 [+0.0062, +0.0328]; +0.0059 [-0.0066, +0.0199]; +0.0171 [+0.0010, +0.0344] | none | +0.0327 [+0.0131, +0.0543]; +0.0126 [+0.0000, +0.0258]; +0.0151 [-0.0024, +0.0353] | none | none |
| aligned − A01 Linear + B3 | selection | -0.0049 [-0.0165, +0.0048]; -0.0062 [-0.0197, +0.0053]; +0.0227 [+0.0107, +0.0376] | none | +0.0026 [-0.0108, +0.0161]; +0.0131 [+0.0027, +0.0240]; +0.0052 [-0.0080, +0.0203] | none | none |
| aligned − A01 Linear + B3 | calibration | +0.0088 [+0.0010, +0.0178]; +0.0020 [-0.0047, +0.0081]; +0.0066 [-0.0031, +0.0166] | none | +0.0201 [+0.0074, +0.0360]; +0.0101 [+0.0024, +0.0201]; +0.0126 [+0.0000, +0.0264] | none | none |

## 3. Mixed-window target accuracy

CLS argmax accuracy at the requested sample of mixed windows. Selection has 1,449 mixed windows from 238 recordings and
36 sessions, and calibration 1,440 from 222 recordings and 35 sessions. No window has the requested view on 7 or fewer
of its 16 frames. The 8-of-16 subset has 94 and 73 windows, and the A4C/A5C pair windows 111 and 127.

| System | Seed | All mixed windows | 8 of 16 | 9-11 of 16 | 12-15 of 16 | A4C/A5C pair windows |
|---|---|---|---|---|---|---|
| CLS continuation | 0 | 0.919 / 0.931 | 0.777 / 0.671 | 0.892 / 0.902 | 0.962 / 0.983 | 0.811 / 0.921 |
| CLS continuation | 1 | 0.938 / 0.947 | 0.819 / 0.849 | 0.909 / 0.916 | 0.978 / 0.983 | 0.883 / 0.929 |
| CLS continuation | 2 | 0.927 / 0.933 | 0.819 / 0.781 | 0.895 / 0.895 | 0.968 / 0.983 | 0.883 / 0.921 |
| Aligned token supervision | 0 | 0.910 / 0.940 | 0.734 / 0.671 | 0.877 / 0.918 | 0.962 / 0.988 | 0.793 / 0.921 |
| Aligned token supervision | 1 | 0.926 / 0.944 | 0.755 / 0.767 | 0.906 / 0.920 | 0.966 / 0.983 | 0.856 / 0.921 |
| Aligned token supervision | 2 | 0.919 / 0.935 | 0.766 / 0.712 | 0.892 / 0.906 | 0.963 / 0.985 | 0.865 / 0.921 |
| Shuffled token supervision | 0 | 0.914 / 0.940 | 0.723 / 0.671 | 0.887 / 0.918 | 0.962 / 0.988 | 0.793 / 0.921 |
| Shuffled token supervision | 1 | 0.924 / 0.940 | 0.766 / 0.712 | 0.898 / 0.915 | 0.967 / 0.986 | 0.856 / 0.929 |
| Shuffled token supervision | 2 | 0.919 / 0.931 | 0.755 / 0.712 | 0.893 / 0.895 | 0.963 / 0.985 | 0.865 / 0.913 |
| frozen Ordered | 0 | 0.903 / 0.915 | 0.628 / 0.575 | 0.871 / 0.875 | 0.967 / 0.986 | 0.829 / 0.913 |
| frozen Ordered | 1 | 0.909 / 0.914 | 0.702 / 0.630 | 0.866 / 0.867 | 0.973 / 0.985 | 0.847 / 0.890 |
| frozen Ordered | 2 | 0.912 / 0.910 | 0.681 / 0.644 | 0.877 / 0.856 | 0.971 / 0.985 | 0.883 / 0.890 |
| historical Continued CLS | 0 | 0.920 / 0.929 | 0.734 / 0.644 | 0.893 / 0.898 | 0.967 / 0.986 | 0.829 / 0.913 |
| historical Continued CLS | 1 | 0.934 / 0.941 | 0.809 / 0.822 | 0.904 / 0.909 | 0.977 / 0.982 | 0.874 / 0.921 |
| historical Continued CLS | 2 | 0.925 / 0.933 | 0.809 / 0.753 | 0.893 / 0.896 | 0.967 / 0.983 | 0.892 / 0.921 |

Paired differences on all mixed windows and on the 8-of-16 windows (session bootstrap; the stream bootstrap is the
sensitivity). `summary.json` holds every subset.

| Comparison | Windows | Half | Session bootstrap (seeds 0, 1, 2) | Stream bootstrap (seeds 0, 1, 2) | Verdict |
|---|---|---|---|---|---|
| aligned − center_only | all mixed | selection | -0.0090 [-0.0162, -0.0026]; -0.0117 [-0.0206, -0.0039]; -0.0076 [-0.0165, +0.0008] | -0.0090 [-0.0159, -0.0020]; -0.0117 [-0.0205, -0.0041]; -0.0076 [-0.0168, +0.0013] | none |
| aligned − center_only | all mixed | calibration | +0.0090 [+0.0014, +0.0163]; -0.0028 [-0.0095, +0.0035]; +0.0021 [-0.0059, +0.0097] | +0.0090 [+0.0026, +0.0174]; -0.0028 [-0.0103, +0.0050]; +0.0021 [-0.0051, +0.0095] | none |
| aligned − center_only | 8 of 16 | selection | -0.0426 [-0.0938, +0.0122]; -0.0638 [-0.1058, -0.0241]; -0.0532 [-0.1161, +0.0000] | -0.0426 [-0.1011, +0.0000]; -0.0638 [-0.1300, +0.0000]; -0.0532 [-0.1183, +0.0000] | none |
| aligned − center_only | 8 of 16 | calibration | +0.0000 [-0.0400, +0.0375]; -0.0822 [-0.1538, -0.0145]; -0.0685 [-0.1379, -0.0161] | +0.0000 [-0.0411, +0.0395]; -0.0822 [-0.1690, +0.0000]; -0.0685 [-0.1299, -0.0137] | none |
| aligned − shuffled | all mixed | selection | -0.0035 [-0.0087, +0.0021]; +0.0021 [-0.0056, +0.0090]; +0.0000 [-0.0044, +0.0040] | -0.0035 [-0.0087, +0.0015]; +0.0021 [-0.0048, +0.0090]; +0.0000 [-0.0058, +0.0059] | none |
| aligned − shuffled | all mixed | calibration | +0.0000 [-0.0044, +0.0041]; +0.0035 [-0.0014, +0.0078]; +0.0049 [+0.0008, +0.0098] | +0.0000 [-0.0041, +0.0040]; +0.0035 [-0.0020, +0.0088]; +0.0049 [+0.0013, +0.0088] | none |
| aligned − shuffled | 8 of 16 | selection | +0.0106 [-0.0238, +0.0556]; -0.0106 [-0.0460, +0.0227]; +0.0106 [-0.0253, +0.0465] | +0.0106 [-0.0227, +0.0486]; -0.0106 [-0.0550, +0.0345]; +0.0106 [-0.0217, +0.0482] | none |
| aligned − shuffled | 8 of 16 | calibration | +0.0000 [-0.0572, +0.0519]; +0.0548 [-0.0137, +0.1053]; +0.0000 [-0.0400, +0.0441] | +0.0000 [-0.0541, +0.0541]; +0.0548 [+0.0000, +0.1233]; +0.0000 [-0.0411, +0.0417] | none |
| aligned − Ordered | all mixed | selection | +0.0069 [-0.0086, +0.0190]; +0.0173 [+0.0046, +0.0276]; +0.0076 [-0.0055, +0.0180] | +0.0069 [-0.0021, +0.0172]; +0.0173 [+0.0067, +0.0277]; +0.0076 [-0.0035, +0.0189] | none |
| aligned − Ordered | all mixed | calibration | +0.0250 [+0.0135, +0.0364]; +0.0299 [+0.0177, +0.0414]; +0.0257 [+0.0146, +0.0374] | +0.0250 [+0.0152, +0.0366]; +0.0299 [+0.0198, +0.0414]; +0.0257 [+0.0169, +0.0352] | consistent positive |
| center_only − Ordered | all mixed | selection | +0.0159 [+0.0047, +0.0255]; +0.0290 [+0.0184, +0.0393]; +0.0152 [+0.0020, +0.0257] | +0.0159 [+0.0069, +0.0250]; +0.0290 [+0.0183, +0.0402]; +0.0152 [+0.0050, +0.0256] | consistent positive |
| center_only − Ordered | all mixed | calibration | +0.0160 [+0.0069, +0.0254]; +0.0326 [+0.0203, +0.0456]; +0.0236 [+0.0143, +0.0352] | +0.0160 [+0.0077, +0.0252]; +0.0326 [+0.0222, +0.0439]; +0.0236 [+0.0147, +0.0331] | consistent positive |
| center_only − Ordered | 8 of 16 | selection | +0.1489 [+0.0685, +0.2329]; +0.1170 [+0.0495, +0.1842]; +0.1383 [+0.0633, +0.2019] | +0.1489 [+0.0825, +0.2247]; +0.1170 [+0.0505, +0.1889]; +0.1383 [+0.0659, +0.2198] | consistent positive |
| center_only − Ordered | 8 of 16 | calibration | +0.0959 [+0.0234, +0.1772]; +0.2192 [+0.1154, +0.3250]; +0.1370 [+0.0617, +0.2295] | +0.0959 [+0.0282, +0.1690]; +0.2192 [+0.1250, +0.3195]; +0.1370 [+0.0667, +0.2162] | consistent positive |

## 4. Anchor diagnostics

The time logits against the reference labels at the 8 anchors of the development windows (diagnostics, not systems).
Each cell gives selection, then calibration. The last column counts the reference class changes between consecutive
anchors of mixed windows at which both anchors are right. A count resting on fewer than 10 sessions gives their number
in brackets.

| Arm | Seed | Anchor accuracy, mixed windows | Anchor accuracy, native windows | A4C anchor recall, mixed | A5C anchor recall, mixed | Changes with both anchors right |
|---|---|---|---|---|---|---|
| CLS continuation | 0 | 0.662 / 0.682 | 0.979 / 0.978 | 0.675 / 0.650 | 0.583 / 0.660 | 11 of 1415 / 7 of 1377 (8 sessions) |
| CLS continuation | 1 | 0.650 / 0.660 | 0.890 / 0.924 | 0.384 / 0.407 | 0.679 / 0.682 | 13 of 1415 / 20 of 1377 |
| CLS continuation | 2 | 0.674 / 0.699 | 0.983 / 0.978 | 0.727 / 0.716 | 0.498 / 0.582 | 10 of 1415 / 4 of 1377 (6 sessions) |
| Aligned token supervision | 0 | 0.845 / 0.864 | 0.981 / 0.983 | 0.857 / 0.832 | 0.773 / 0.865 | 421 of 1415 / 442 of 1377 |
| Aligned token supervision | 1 | 0.846 / 0.860 | 0.978 / 0.978 | 0.816 / 0.795 | 0.823 / 0.871 | 404 of 1415 / 417 of 1377 |
| Aligned token supervision | 2 | 0.857 / 0.879 | 0.976 / 0.980 | 0.844 / 0.844 | 0.803 / 0.879 | 444 of 1415 / 505 of 1377 |
| Shuffled token supervision | 0 | 0.695 / 0.715 | 0.982 / 0.985 | 0.709 / 0.695 | 0.603 / 0.698 | 12 of 1415 / 13 of 1377 |
| Shuffled token supervision | 1 | 0.702 / 0.715 | 0.979 / 0.984 | 0.678 / 0.682 | 0.651 / 0.719 | 16 of 1415 / 19 of 1377 |
| Shuffled token supervision | 2 | 0.696 / 0.714 | 0.979 / 0.983 | 0.684 / 0.677 | 0.625 / 0.698 | 15 of 1415 / 21 of 1377 |

## 5. Native clips

Fixed and new native errors with B3, per seed (s0; s1; s2).

| Comparison | Half | A4C read as A5C, fixed/new | A5C read as A4C, fixed/new | All native clips, fixed/new |
|---|---|---|---|---|
| aligned − center_only | selection | 0/0; 1/0; 1/0 | 0/0; 0/0; 0/0 | 0/0; 1/0; 1/0 |
| aligned − center_only | calibration | 0/0; 0/0; 1/0 | 0/0; 0/0; 0/0 | 0/0; 0/0; 1/0 |
| aligned − Ordered | selection | 0/0; 1/0; 1/0 | 0/0; 0/0; 0/0 | 0/0; 1/0; 1/0 |
| aligned − Ordered | calibration | 0/0; 0/0; 1/0 | 0/0; 0/0; 0/0 | 0/0; 0/0; 1/0 |
| aligned − shuffled | selection | 0/0; 2/1; 0/0 | 0/0; 0/0; 0/0 | 0/0; 2/1; 0/0 |
| aligned − shuffled | calibration | 0/0; 0/0; 0/0 | 0/0; 0/0; 0/0 | 0/0; 0/0; 0/0 |
| center_only − Ordered | selection | 0/0; 0/0; 1/1 | 0/0; 0/0; 0/0 | 0/0; 0/0; 1/1 |
| center_only − Ordered | calibration | 0/0; 0/0; 0/0 | 0/0; 0/0; 0/0 | 0/0; 0/0; 0/0 |

## 6. Short segments and X-Y-X middles

Matched reference segments of the multi streams with B3, and the mean absolute offset of matched boundaries.

| System (B3) | Seed | Segments of at most 16 samples | A4C or A5C among them | X-Y-X middles | A4C/A5C X-Y-X middles | Mean absolute boundary offset (samples) |
|---|---|---|---|---|---|---|
| CLS continuation | 0 | 133/141 / 129/134 | 59/65 / 65/67 | 62/69 / 80/80 | 4/7 / 7/7 | 1.44 / 1.37 |
| CLS continuation | 1 | 134/141 / 128/134 | 58/65 / 62/67 | 64/69 / 79/80 | 3/7 / 7/7 | 1.38 / 1.32 |
| CLS continuation | 2 | 140/141 / 128/134 | 64/65 / 63/67 | 66/69 / 79/80 | 6/7 / 7/7 | 1.45 / 1.35 |
| Aligned token supervision | 0 | 131/141 / 132/134 | 55/65 / 65/67 | 64/69 / 80/80 | 3/7 / 7/7 | 1.51 / 1.35 |
| Aligned token supervision | 1 | 132/141 / 129/134 | 56/65 / 63/67 | 64/69 / 79/80 | 3/7 / 7/7 | 1.43 / 1.33 |
| Aligned token supervision | 2 | 138/141 / 128/134 | 62/65 / 64/67 | 67/69 / 80/80 | 6/7 / 7/7 | 1.50 / 1.35 |
| Shuffled token supervision | 0 | 131/141 / 133/134 | 55/65 / 66/67 | 64/69 / 80/80 | 3/7 / 7/7 | 1.50 / 1.36 |
| Shuffled token supervision | 1 | 137/141 / 129/134 | 61/65 / 63/67 | 65/69 / 79/80 | 4/7 / 7/7 | 1.46 / 1.34 |
| Shuffled token supervision | 2 | 135/141 / 127/134 | 59/65 / 63/67 | 66/69 / 80/80 | 5/7 / 7/7 | 1.50 / 1.39 |
| frozen Ordered + B3 | 0 | 130/141 / 128/134 | 54/65 / 66/67 | 66/69 / 80/80 | 5/7 / 7/7 | 1.55 / 1.46 |
| frozen Ordered + B3 | 1 | 134/141 / 129/134 | 58/65 / 65/67 | 64/69 / 80/80 | 3/7 / 7/7 | 1.58 / 1.47 |
| frozen Ordered + B3 | 2 | 136/141 / 127/134 | 60/65 / 65/67 | 66/69 / 80/80 | 5/7 / 7/7 | 1.55 / 1.49 |

## 7. Goal

Three-seed means against the minimums of 0.9444 and 0.9821. The calibration sensitivity target, 0.9839, is the
G1 + ASRF calibration mean plus 0.01.

| System | Three-seed mean | Margin to the minimum | Calibration sensitivity margin | Constraint failures | Failing cells | Goal met |
|---|---|---|---|---|---|---|
| CLS continuation | 0.9374 / 0.9786 | −0.0070 / −0.0036 | −0.0053 | 2 | seed 0 selection (recall 0.935); seed 2 selection (native 0.9701) | no |
| Aligned token supervision | 0.9515 / 0.9817 | +0.0071 / −0.0004 | −0.0021 | 0 | none | no |
| Shuffled token supervision | 0.9463 / 0.9814 | +0.0019 / −0.0007 | −0.0024 | 0 | none | no |
| frozen Ordered + B3 | 0.9544 / 0.9817 | +0.0100 / −0.0004 | −0.0022 | 1 | seed 2 selection (native 0.9701) | no |
| historical Continued CLS | 0.9442 / 0.9776 | −0.0003 / −0.0045 | −0.0063 | 1 | seed 2 selection (native 0.9701) | no |

## 8. Training

`training_curves.png` shows the CLS loss, the time-token loss against each arm's own labels and the training-batch CLS
accuracy per epoch. The final weights after 3,750 updates were kept, and nothing was chosen from these curves.

- **Inputs and pairing.** Every arm of a seed read the same RV windows in the same order, and every decoded window
  matched its pixel digest. The check found identical logits and random number states across the three arms after an
  update, and identical aligned and shuffled losses and gradients on native windows.
- **Warm-up.** Every encoder was unchanged through the 375-update frozen warm-up. No run was interrupted or resumed.
- **Losses.** The CLS loss fell in every run, from 0.21 to 0.26 in the first epoch to 0.13 to 0.19 in the last. The
  aligned arm's time-token loss fell from 0.67 to 0.75 in the first epoch to 0.29 in the last. The shuffled arm's fell
  from 0.65 to 0.74 to 0.48 to 0.50. In CLS continuation, which gives it weight 0, it stayed between 0.68 and 0.81.
- **Training-batch time accuracy.** In the last epoch, the time logits named their own labels at 0.904 to 0.908 of the
  anchors in the aligned arm and 0.834 to 0.838 in the shuffled arm. In CLS continuation, they named the aligned labels
  at 0.768 to 0.782 of the anchors.
- **Parameters.** Every model has 34,640,549 parameters, the source's, in V01's two optimiser groups (34,623,872
  encoder and 16,677 marker and head parameters).
- **Determinism.** Training ran in fp32 with TF32 off and deterministic cuDNN. V01's recorded MViT max-pool backward has
  no deterministic CUDA kernel, so runs are not bit-reproducible after the encoder unfreezes. In the check, the 6-update
  and 3 + 3 runs differed by at most 1e-7 (9.7e-8) in loss and 4e-9 in parameters, and the unfreeze boundary stayed
  within V01's drift rule.

## What this round does not show

- **No new architecture.** The arms share the source model and differ only in a training objective.
- **No system gain.** No consistent segment F1@50 difference was detected between aligned token supervision and CLS
  continuation, the frozen Ordered reference or the shuffled control, and no system meets the goal.
- **Not a verdict on dense supervision in general.** Only this configuration ends: these anchors, this loss weight, this
  budget and CLS-only inference. The time logits were never used as a system.
- **Not why no change in the CLS decisions was detected.** The anchor diagnostics show that the time logits learned
  the aligned labels. This round does not show why no consistent change in the CLS decisions came with that, or
  whether the slightly lower selection mixed-window accuracy of both supervised arms comes from the auxiliary loss.
- **Not the effect of the anchor choice.** Another anchor slot (for example the more influential slot 2j + 1) was not
  tested, as the design fixed the stem centre.
- **No cardiac or probe explanation.** Score changes are not attributed to cardiac phase or probe motion.
- **Not clinical scanning.** Constructed streams and public recording labels are not real continuous scans or expert
  frame labels, and the development halves are not an independent validation.

## Deviations and launch record

1. **Training tests committed with the code.** The task sheet asks for failing tests before each implementation. The
   data and model tests were committed first and failed at import (471f8bf and 3bc7c89). The training tests were
   written before the training, check and queue scripts but committed with them in cd0d4c0. The report tests were
   written with the tables module and committed with it in fab6e6b, before any development output.
2. **A test correction.** Two identical multithreaded CPU backward passes differ by about 3e-10 in the
   relative-position gradients. The exact gradient-equality tests therefore run on one CPU thread, where repeated runs
   and the lambda-0 comparison are bit-identical (ef2c938).
3. **The first preparation hashed evaluation files.** The first preparation (02:17) also computed the digests of the
   saved development outputs of V01 and V02, A01's configuration and development features, and the result summaries.
   Nothing was parsed.
   cd0d4c0 restricted the preparation to the training inputs, and the run at 02:32 is the formal record. Its label files
   are identical to the first run's. The evaluation checks the reference pins after training.
4. **The review's fixes.** f120f94 made the queue charge killed attempts and refuse a second queue, made the trainer's
   bookkeeping start clean, and tightened one check. The trainer change altered the run identities, so the check ran a
   second time. Both checks are in the ledger.
5. **The timing estimate.** The check projected the cost from timed updates without the epoch saving. The nine runs took
   0.473 to 0.475 hours each against the projected 0.477 hours.
6. **Raw media kept on the server.** The check first wrote a figure with 16 frames of a training window into the results
   directory, and it entered an unpushed commit. Following AGENTS.md, that commit was rewritten without the figure, the
   figure was moved to the server's check directory (940e4f8), and the report draws a frame-free schematic
   (`token_supervision.png`).

## Fact-check

An independent fact-check on 2026-10-14 recomputed this round's results, mostly with its own code. It found no error in
the results. Before publication it corrected ten statements of this README and reworded five. The server was unreachable
from about 08:38 to 15:05 UTC, so the check ran in two parts, first on the local files and then on the server.

- **Integrity.**
  - All 33 pins of the configuration match.
  - In all nine runs, the frozen weights, run manifests, queue record and `summary.json` agree, and the run identity
    recomputes. Every run has 3,750 updates and no interruption or resume, and its 30,000 window digests were checked.
  - The three V01 sources are unchanged, and the server's code equals commit 940e4f8. The four test files pass on the
    CPU (54 tests).
- **Labels.** The label files were re-derived from the pinned window lists. The aligned labels, the 90,000 permutations,
  the shuffled labels, the pairing checks and the counts are identical. At least one anchor's label differs from the next
  slot's in 52.4 to 53.2% of the mixed training windows.
- **Systems.**
  - All 54 rows of the nine new models and their nine B3 choices reproduce exactly from the saved per-window outputs.
  - Independent code for B3, segment F1@50 and native macro-F1 agrees to the last digit.
  - Frozen Ordered and historical Continued CLS reproduce their stored rows, and every reference row equals its stored
    row.
- **Comparisons and readings.**
  - The 102 within-round paired bootstraps reproduce exactly, and the 36 B3 ones also with independent code.
  - The verdicts, the mixed-window accuracies, bootstraps and verdicts, the anchor diagnostics, the goal and the fixed
    readings (no gain and end) all reproduce.
- **Re-inference.**
  - The three frozen aligned models were rerun in the round's own batches on the first multi stream of each half (28
    and 25 windows). They reproduced the saved CLS and time logits bit for bit, and their CLS logits equal the plain
    forward.
  - In a smaller batch, the CLS logits stay identical and the time logits differ by at most 2.9e-6.
  - The job took 0.011 GPU hours, so the round has used 5.44 of its 8 hours. Its ledger note says 32 windows per half,
    but the job ran all 28 and 25 windows of these streams.
- **Ids and frames.** No recording, session or stream id appears in any file of this directory, and neither figure
  contains a frame.
- **Corrections before publication.**
  - The CLS-continuation time-loss range (0.68 to 0.81) and the shuffled arm's first-epoch time loss (0.65 to 0.74).
  - The check's largest loss difference (9.7e-8) and the inference time per window (16.3 to 16.4 ms).
  - Two anchor counts that rest on fewer than 10 sessions.
  - The scope of the A01 sentence, a row label in section 3, the check's commit and the first preparation's hashed files.
  - Five sentences that read a difference that was not detected as no effect.

## Commands

Run on the server in the project `.venv`, one GPU job at a time, with `ECHO_GPU_MEM_FRACTION=0.6`.

```bash
C=configs/experiments/video_dense_supervision_v1.yaml
python scripts/prepare_video_dense_supervision.py --config $C           # pins, geometry, SC/RV pairing, label files (CPU)
python scripts/train_video_dense_supervision.py --config $C --check     # pairing, parity, arm pairing, smokes, resume, boundary, timing, budget
python scripts/queue_video_dense_supervision.py --config $C             # nine trainings, then freezing
python scripts/report_video_dense_supervision.py --config $C --evaluate
python -m pytest tests/test_video_dense_supervision_data.py tests/test_video_dense_supervision_model.py tests/test_video_dense_supervision_training.py tests/test_video_dense_supervision_report.py -q
```

## Provenance

- **Code.**
  - The model module is at ef2c938, the preparation at cd0d4c0 and the training entry and queue at f120f94.
  - The passing check ran at f120f94; 940e4f8 later changed only where it writes its frame figure. The configuration
    is at 8ca5fc5.
  - The evaluation is at 940e4f8, with the tables module at fab6e6b.
- **Sources.**
  - The configuration pins the task sheet, the V01 and V02 configurations, and the three V01 Ordered final weights.
  - It also pins V01's RV and the SC window lists with their indexes.
  - For the references, it pins the saved Ordered and Continued CLS outputs, A01's configuration and features, and the
    V01, V02 and A01 summaries.
  - Every digest was checked before use. The V01 sources were unchanged after training.
- **Models.** The nine final weights are frozen in `models_frozen/video_dense_supervision_v1/{center_only,aligned_dense,shuffled_dense}_s{0,1,2}`
  with their run manifests, label files and configuration. Their digests are in `summary.json`.
- **Server outputs.**
  - `runs/video_dense_supervision_v1/manifests/` holds the label files.
  - `runs/video_dense_supervision_v1/evaluation/{selection,calibration}/` holds every model's per-window CLS and time
    logits and per-sample probabilities.
  - `runs/video_dense_supervision_v1/checks/` holds the check record and its frame figure.
  - Recording, session and stream ids stay on the server.

## Files

- `README.md` is this file. `code_review.md` is the implementation review, `preparation.json` the formal preparation
  record and `checks.json` the second passed check.
- `summary.json` holds every system row, the paired comparisons and verdicts, the mixed-window and anchor analyses, the
  units, the goal, the decision, the reference pins and the costs. `report_tables.md` holds the generated tables.
- `training_curves.png` shows the training curves, and `token_supervision.png` the frame-free schematic.
- None of these files contains a recording, session or stream id or a frame.
