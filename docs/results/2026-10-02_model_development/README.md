# Model development, round 1 (2026-10-02)

*Wording corrected on 2026-10-02 after review. "Parity" is replaced by "no consistent difference detected", the
head timings are no longer read as a speed result, and per-sample and native-clip claims are kept apart.*

This round follows `docs/superpowers/plans/2026-10-02-model-development.md`. The goal was a new temporal model that
outputs a view label for every sample of a continuous stream and beats the existing strong baselines under a fair
comparison. The model tested is "boundary-controlled context aggregation". It learns local classification and view
changes together. It uses neighbouring samples while it predicts the same view, and it limits mixing across a
predicted change. Everything below is on the EV9V validation split. The test split was not used.

**Terms used below.**

- A *native clip* is one whole recorded clip. A *constructed stream* joins 4–8 *fragments* (pieces of 1–4 s from
  different clips). A *join* is where two fragments meet. A *view change* is a join between fragments of different
  classes.
- *Native macro-F1* scores one label per native clip, taken as the majority of the decoded sample labels.
- *Segment F1@50* counts a predicted segment as correct when it has the true class and overlaps a true segment with
  intersection over union of at least 0.5. It is pooled over the 120 multi-fragment streams of a half. It is the
  primary metric.
- *Boundary recall* is the share of true view changes matched one-to-one by a predicted label change within 0.5 s.
  The plan requires at least 0.95.
- *Cuts per clip* counts label changes inside native clips. The *same-class cut rate* is the share of same-class
  joins at which a change is predicted within 0.5 s.
- The validation split is divided by recording session into a *selection half*, where checkpoints were chosen, and
  a *calibration half*, where decoding settings were tuned. A *seed* is the training seed of a temporal head. The
  encoder is the same for every row.
- Intervals are paired 95% bootstrap intervals with 2,000 draws. Native clips are resampled by recording session
  and streams by stream. A contrast is *consistent* when all three seeds' intervals exclude zero on the same side.
  "No consistent difference detected" is not evidence that two methods are equal.

## Answer

1. **The new model as specified (M4) does not beat the strong baselines.** With its default per-sample decoding it
   is below ASRF on segment F1@50 in every seed and on both halves. The differences are −3.7 to −8.9 points on the
   calibration half and −7.0 to −13.2 points on the selection half, and every interval lies below zero. M4 fails
   the plan's expansion gate.
2. **The predicted-boundary path term works as designed, but only matters for per-sample decisions.** Against the
   identical model without it (M3), M4 raises segment F1 on the selection half in all three seeds, by +2.7 to +5.2
   points with every interval above zero. On the calibration half the gains are +2.4 to +3.4 points, and only one of
   the three intervals excludes zero. Once any segment-level decoder is applied, the path term makes no consistent
   difference.
3. **The error analysis points to one problem.** On constructed streams in the calibration half (seed 0), M4 gets
   fewer fragments entirely wrong than the segment-level baselines: 17 against 36 for ASRF and 31 for the audited
   B3. This is a constructed-stream result; M4 is not better at native-clip classification. Its losses come from short runs of wrong labels at view changes and inside fragments, which split segments in
   two.
4. **One fix was trained for that problem: segment pooling (M5).** M5 averages the class evidence over each
   predicted segment, weighted by the predicted boundary probabilities. It raises segment F1 over M4 in every seed
   and on both halves (+5.5 to +13.3 points), and it cuts internal cuts by a factor of 7 to 10. The price is 1.8–3.4
   points of boundary recall on the selection half.
5. **No consistent difference was detected between M5 and the strongest baseline, and M5 does not beat it.** This
   is not evidence that the two are equivalent. Over three seeds M5 averages 0.949
   segment F1 on the calibration half and 0.917 on the selection half. ASRF averages 0.952 and 0.927. The audited and
   tuned B3 decoders on the encoder's probabilities reach 0.928–0.934 on both halves. No contrast between M5 and ASRF
   is consistent across seeds. The round therefore meets neither the 0.5-point expansion gate nor the 1-point
   target. Following the plan, Task 5 (stronger encoder, joint fine-tuning) was not started.

### Gains and losses against the strongest baseline

ASRF (M1) is the strongest learned baseline on both halves. On the selection half the tuned B3 decoder is stronger
still. The table compares the best new-model variant, M5, with both. Values are means over three seeds with their
range, and B3 does not depend on the seed.

| Calibration half | native macro-F1 | segment F1@50 | boundary recall | boundary F1 | cuts per clip | same-class cut rate |
|---|---:|---:|---:|---:|---:|---:|
| M5 (new model + pooling) | 0.969 [0.966, 0.972] | 0.949 [0.945, 0.954] | 0.973 [0.970, 0.975] | 0.968 | 0.037 | 0.025 |
| M1 ASRF | 0.972 [0.970, 0.975] | 0.952 [0.936, 0.960] | 0.964 [0.950, 0.977] | 0.973 | 0.004 | 0.020 |
| B3 tuned | 0.966 | 0.934 | 0.955 | 0.962 | 0.034 | 0.020 |

| Selection half | native macro-F1 | segment F1@50 | boundary recall | boundary F1 | cuts per clip | same-class cut rate |
|---|---:|---:|---:|---:|---:|---:|
| M5 (new model + pooling) | 0.964 [0.963, 0.965] | 0.917 [0.907, 0.925] | 0.951 [0.942, 0.955] | 0.947 | 0.025 | 0.035 |
| M1 ASRF | 0.966 [0.965, 0.968] | 0.927 [0.912, 0.946] | 0.952 [0.937, 0.966] | 0.963 | 0.008 | 0.014 |
| B3 tuned | 0.965 | 0.934 | 0.961 | 0.977 | 0.014 | 0.004 |

What M5 gains:

- On constructed streams only, it labels more samples correctly. This is a per-sample result. It does not carry
  over to native-clip classification, where M5 is not better (see below). In seed 0, 2.9% of stream samples are wrong for M5
  against 4.9% for ASRF on the calibration half, and 6.5% against 9.3% on the selection half. M5 corrects 384 and
  507 stream samples that ASRF gets wrong and breaks 80 and 87.
- Its boundary recall on the calibration half is higher in two of three seeds (+2.5, +0.5 and −0.3 points; not
  consistent).
- It is consistently better than the adapted official MS-TCN (+2.7 to +5.3 points of segment F1 on both halves in
  every seed). M4 decoded by its own boundary segments is also consistently better than the tuned HMM.
- It is smaller: 380k parameters against 1.19M. The head timings in section 7 (0.007 ms per sample against 0.137 ms)
  are not a speed comparison. M5 was decoded in batches while ASRF ran one sequence at a time in its official
  decoding, and neither figure includes the encoder. A speed claim needs the same batching and full-system timing.

What M5 loses:

- It leaves more internal cuts: 0.037 per clip against 0.004 on the calibration half (higher in every seed, with two
  of three intervals above zero).
- On the selection half its boundary recall (0.942–0.955) falls below 0.95 in one seed, and its same-class cut rate
  is higher.
- On the selection half's native clips it is worse per sample. In seed 0 it breaks 126 native samples that ASRF gets
  right and fixes 29. One PLAX clip is read as PSAX.
- Clip-level errors are unchanged. On each half both methods misread the identical three A5C clips as A4C.

## Status

| Item | State |
|---|---|
| Task 1, protocol, metrics and the sampling check | Done. The persistent-worker defect was confirmed with real worker batches and fixed. |
| Task 2, v3 encoder and shared features | Done. One encoder seed (seed 0), features for train and validation only, online check passed. |
| Task 3, new model, functional checks, training | Done. 33 tests pass, the 16-stream overfit check passed, M4 trained with seeds 0–2. |
| Task 4, M0–M4 matrix | Done with seeds 0–2. B1, HMM and B3 were tuned on the calibration half. The learning-rate retry (1e-3) was not used, because every row peaked early rather than under-training. |
| Task 4, fixed-gate ablation | Not run. Its condition (an M4 increment on validation) was not met. |
| Task 4, feature-masking step for M1 and M4 | Done for seed 0 only. Effects were small and inconsistent between halves, so masking was not adopted. |
| One evidenced fix: segment pooling | Done. M5 and M3p trained with seeds 0–2. |
| Task 5, stronger encoder and joint fine-tuning | Not started. The plan starts it only after M4 shows a useful trend against the strongest same-encoder comparator. |
| Task 6, this report, weights and curves | Done. |

## 1. Data, protocol and encoder

**Protocol.** `scripts/model_development_protocol.py` wrote
`runs/20261002-051140-model_development_protocol-fa45e7f4-s0/protocol.json` (SHA-256 `a897483b776e8cff…`). The
validation split's 72 recording sessions were halved with seed 20261002, and the first draw already covered all five
classes in both halves.

| Part | Sessions | Clips | PLAX / PSAX / A4C / A5C / SC4C clips | Streams |
|---|---:|---:|---|---|
| Selection half | 36 | 284 | 105 / 68 / 87 / 11 / 13 | 137 pairs, 120 multi-fragment |
| Calibration half | 36 | 237 | 80 / 71 / 61 / 18 / 7 | 115 pairs, 120 multi-fragment |
| Training | — | 3,364 | 1,120 / 974 / 858 / 217 / 195 | 1,500 multi-fragment (digest `228062c5f9acef9d`) |

Each half's streams are built only from that half's sessions. The training bank is the one used by earlier MS-TCN
training, and its digest was checked again at the start of every run.

**Sampling defect.** `EpochFrameDataset.resample()` changed the sample list in the main process only. With
persistent DataLoader workers, every epoch of the original B0 read epoch 0's frames. A test that reads real worker
batches reproduces the defect, and the training loader now restarts its workers each epoch. In the v3 run, the
sample-order digests of epochs 0 and 1 differ (47,334 and 46,895 distinct samples).

**Encoder v3.** It is a ResNet-18 with ImageNet initialisation, trained at 224 pixels on 16 frames per clip per epoch.
Training used AdamW at lr 3e-4. It was selected on the selection half by 10 Hz clip majority vote. Epoch 0 was best
(clip macro-F1 0.965), and early stopping ended the run at epoch 6. The checkpoint's SHA-256 is `beb36d80df31…`, which
is also the feature-cache key. Features for train and validation were extracted for the unedited frames and the
gamma 0.9 edit (3,885 clips each). Every protocol clip is cached with the protocol's sample count, and no test row
exists. A short validation clip re-extracted online matched the cache on frame indices and on every argmax. The
largest probability difference was 0.0007, from bf16 batching.

The encoder's own frame-majority clip macro-F1 is 0.965 on the selection half and 0.982 on the calibration half. These
values set the native-F1 floors of the selection rules (0.960 and 0.977).

## 2. Models compared

All temporal heads read the same frozen v3 features and the same training batches. Each update reads 4 constructed
training streams and 4 native windows of at most 256 samples, re-drawn each epoch. Each epoch has 375 updates. All
heads share AdamW at lr 3e-4 with weight decay 1e-4, 2 warm-up epochs then cosine decay, gradient clipping at 1, at
most 50 epochs and patience 8. The checkpoint is chosen on the selection half. The rule takes the highest
multi-fragment segment F1 among epochs within 0.5 points of the encoder's native macro-F1, and breaks ties within
0.2 points by fewer cuts.

| Row | Model | Training loss | Default decoding | Parameters |
|---|---|---|---|---:|
| M0 | Official MS-TCN (pinned `33ed91c`), 4 stages × 10 layers × 64 | the authors' CE + 0.15 clamped T-MSE per stage | per-sample argmax | 695,764 |
| M1 | Official ASRF (pinned `9623f1e`), 4 + 4 stages | the authors' weighted CE + Gaussian-similarity TMSE, boundary BCE with pos_weight, λ = 0.1 | the authors' refinement with boundaries at 0.5 | 1,191,768 |
| M2 | New model without the path term and without boundary loss | stream CE + 0.25 native CE + 0.2 local CE + 0.2 clip NLL | argmax | 380,464 |
| M3 | M2 + boundary loss (attention does not read b) | + 0.2 boundary loss | argmax | 380,464 |
| M4 | Full new model: attention reads the predicted boundaries | as M3 | argmax | 380,464 |
| M3p, M5 | M3 and M4 with segment pooling of the output (section 6) | as M3 | argmax | 380,464 |

The new model projects each sample to 128 dimensions and passes it through three residual dilated convolutions
(dilations 1, 2 and 4). A boundary head reads adjacent local representations and predicts b[t], the probability of a
change between samples t−1 and t. Four attention heads look ±20 samples, with a learned relative position bias. In
M4 each score from t to j also gets the sum of log(1 − b) over the transitions between them. A learned gate mixes the
local and context logits.

M2, M3 and M4 build the same modules in the same order, so one seed gives identical initial weights. M2's boundary
head is never trained and never used. The official classes and losses are imported unchanged from the pinned
checkouts. What was adapted for this task is the shared data, optimiser, update budget and checkpoint rule.

**Checks before training.** 33 tests pass: 20 in `tests/test_boundary_context.py` and 13 in the sampling, protocol
and extraction test files. Together they cover the plan's checklist:

- output shapes, padding invariance and finite outputs at T = 1;
- boundary targets on a hand-made stream;
- weak negatives weighing exactly 0.1 of strong ones;
- the path term against an explicit sum;
- b near 0 reproducing plain attention, and b near 1 cutting the attention mass across a change to below 1/1000 of
  its plain-attention value;
- predictions that ignore recipe metadata;
- identical initial weights across M2–M4;
- the batch plan, the masking augmentation, the metrics on a hand-made example and the selection rules;
- both official heads training and decoding.

On 16 training streams, M4's classification loss fell from 1.50 to below 1e-4 and its boundary loss from 0.77 to below
1e-4. Training accuracy reached 1.0. The mean predicted b was 0.99999 at true changes and 0.00002 at same-class joins.

## 3. Results across seeds

Default decodings, means over seeds 0–2 with the range across seeds. B1 is omitted from the table. Under the native-F1
floor, its best setting barely smooths (window 3, minimum run 1), so it reaches only 0.74 and 0.64.

| System | Calibration native F1 | Calibration segment F1@50 | Calibration recall | Calibration cuts/clip | Selection segment F1@50 | Selection recall | Selection cuts/clip |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 encoder argmax | 0.982 | 0.577 | 0.990 | 2.688 | 0.528 | 0.995 | 1.616 |
| B2 HMM, tuned | 0.966 | 0.925 | 0.982 | 0.165 | 0.867 | 0.966 | 0.067 |
| B3 change points, tuned | 0.966 | 0.934 | 0.955 | 0.034 | 0.934 | 0.961 | 0.014 |
| B3 change points, audited | 0.957 | 0.928 | 0.967 | 0.051 | 0.932 | 0.963 | 0.014 |
| M0 MS-TCN | 0.969 | 0.916 [0.914, 0.918] | 0.964 | 0.195 | 0.876 [0.872, 0.884] | 0.965 | 0.067 |
| M1 ASRF | 0.972 | **0.952** [0.936, 0.960] | 0.964 | 0.004 | 0.927 [0.912, 0.946] | 0.952 | 0.008 |
| M2 plain attention | 0.968 | 0.858 [0.848, 0.866] | 0.982 | 0.515 | 0.794 [0.772, 0.807] | 0.972 | 0.195 |
| M3 + boundary loss | 0.968 | 0.855 [0.836, 0.875] | 0.983 | 0.533 | 0.785 [0.766, 0.800] | 0.973 | 0.180 |
| M4 full new model | 0.967 | 0.882 [0.870, 0.899] | 0.985 | 0.394 | 0.822 [0.792, 0.842] | 0.976 | 0.177 |
| M3p = M3 + pooling | 0.972 | 0.944 [0.934, 0.957] | 0.970 | 0.044 | 0.915 [0.892, 0.928] | 0.955 | 0.040 |
| M5 = M4 + pooling | 0.969 | 0.949 [0.945, 0.954] | 0.973 | 0.037 | 0.917 [0.907, 0.925] | 0.951 | 0.025 |

The calibration half is the main comparison for learned rows, because no checkpoint was chosen on it. It favours the
simple decoders, which were tuned on it. The selection half favours the learned rows, whose checkpoints were chosen
there. Even so, B3 is at least as good as every learned row on the selection half.

**Contrasts across seeds** (segment F1@50 unless stated). Full intervals are in `seeds.md`.

| Contrast | Calibration half | Selection half |
|---|---|---|
| Path term, per-sample decoding (M4 − M3) | +2.4, +2.4, +3.4; not consistent | +5.2, +3.0, +2.7; consistent |
| Path term + boundary loss (M4 − M2) | +3.3, +1.6, +2.2; not consistent | +3.5, +2.9, +2.0; consistent |
| Boundary loss alone (M3 − M2) | +0.9, −0.8, −1.2; not consistent | −1.7, −0.1, −0.7; not consistent |
| Path term under B3 decoding of posteriors (M4 − M3) | −0.4, −0.1, −0.3; not consistent | −1.7, +1.1, +3.8; not consistent |
| Path term with pooling (M5 − M3p) | +2.0, −1.2, +0.7; not consistent | +1.5, −1.0, −0.0; not consistent |
| Segment pooling (M5 − M4) | +5.5, +6.8, +7.9; consistent | +6.5, +8.8, +13.3; consistent |
| M4 − ASRF | −3.7, −8.3, −8.9; consistently lower | −7.0, −11.6, −13.2; consistently lower |
| M5 − ASRF | +1.8, −1.5, −1.1; not consistent | −0.5, −2.8, +0.2; not consistent |
| M5 − MS-TCN | +3.9, +2.7, +3.5; consistent | +3.4, +3.5, +5.3; consistent |
| M5 − B3 tuned | +2.0, +1.1, +1.5; not consistent | −2.7, −1.6, −0.9; not consistent |

Segment pooling also lowers cuts per clip in every seed on both halves. It costs boundary recall consistently on the
selection half (−2.1, −3.4 and −1.8 points) but not on the calibration half.

## 4. What the new model fixes and what it breaks

Error runs are stretches of consecutive wrong samples. The counts below are for seed 0 on the calibration half's
multi-fragment streams. A *timing* run starts or ends within 5 samples of a view change, and a *whole-fragment*
run covers at least 80% of a fragment. *Interior* runs are the rest, split at 3 samples.

| Row | Whole fragment wrong | Timing runs | Interior runs ≤ 3 | Interior runs > 3 | Wrong samples |
|---|---:|---:|---:|---:|---:|
| B0 encoder argmax | 6 | 116 | 154 | 34 | 6.1% |
| B3 audited | 31 | 15 | 0 | 3 | 5.2% |
| M1 ASRF | 36 | 3 | 0 | 1 | 4.9% |
| M4 | 17 | 24 | 15 | 6 | 3.2% |

- **Fixed.** M4 removes most of the encoder's short flickers. Error runs inside native clips drop from 311 to 47,
  and runs of at most 3 samples from 241 to 23. Against B0, M4 corrects 451 native and 509 stream samples and breaks
  180 and 83. This answers the plan's question for short runs: they are corrected, while 98% of true changes are kept.
- **Not fixed.** Runs of 10 or more samples inside native clips are not reduced (14 for B0, 15 for M4). Clip-level
  A4C/A5C errors are shared by every method. On the calibration half the methods misread 2–5 of the 18 A5C clips,
  and two of those clips defeat every method. On the selection half all ten methods misread the same 3 of 11.
- **Broken by per-sample decisions.** M4's remaining runs are short and scattered, so each one splits a segment.
  ASRF and B3 decide one label per segment. They leave almost no short runs, but they get whole fragments wrong about
  twice as often.
- **After pooling.** M5 keeps M4's advantage on fragments (2.9% wrong stream samples) and removes most short runs. In
  seed 0 it has 5 error runs of at most 3 samples on the streams, against 29 for M4. Its native clips have 8 error
  runs, against 47 for M4 and 5 for ASRF.
- **PLAX/PSAX.** On the calibration half M5 has no PLAX/PSAX frame confusions, against 96 frames for ASRF and 99 for
  B3. On the selection half the order reverses: 91 frames for M5 and 0 for ASRF. Of these, 85 are one whole PLAX clip
  that M5 reads as PSAX.

## 5. Decoders on learned posteriors

To separate the head from its decision rule, `scripts/posterior_decoders.py` decodes each head's class posteriors with
the HMM and B3 grids, and with *boundary segments*. Boundary segments cut where the head's own b peaks above a
threshold and label each segment by its mean posterior. Each family's setting is chosen on the calibration half by
the shared rule.

With these decoders every attention head reaches 0.946–0.957 on the calibration half (means over seeds). M5 with B3
reaches 0.957 [0.955, 0.959] and M3p 0.956. ASRF with boundary segments reaches 0.954 and MS-TCN with B3 0.946. On the
selection half the same systems reach 0.91–0.93, level with B3 on the encoder probabilities (0.932–0.934).

Under any shared segment decoder, M4's path term shows no consistent effect. The gain from the context heads over the
encoder comes from better posteriors combined with segment-level decisions. It does not depend on the path term
specifically.

## 6. The fix: segment pooling

The diagnosis in section 4 shows that the decision rule is what fragments M4's output, not its classification. The
one change made in response is a parameter-free pooling of the output. The output at t is the mean of the gated
log-probabilities over all samples j of the sequence. Each j is weighted by the product of (1 − b) along the path from
t to j, which is the softmax over j of the same path term. Inside a predicted segment the weights are nearly uniform.
Across a confident predicted change they fall to about 1e-4. The layer is the trainable, soft counterpart of decoding
by boundary segments. b now also gets gradient from the classification loss through the pooling.

Pooling passed the same functional checks: padding invariance, segment averages on hand-made boundaries and the
16-stream overfit check. M3p applies the same pooling without the path term in attention.

## 7. Training curves and cost

`training_curves_seed0.png` plots the training loss and the selection-half curves of every seed-0 row.
`training_curves_M4_s0.png` and `training_curves_M5_s0.png` add each run's loss terms, boundary positive rate and gate
mean. Every run's own curve, per-epoch log, predictions and digests are in its run directory (section 9).

All learned rows reach their selected checkpoint early (epoch 0–14), and validation then drifts while training loss
keeps falling. The selection-half native macro-F1 jumps between 0.951 and 0.965 as one A5C clip flips. This makes the
native floor of the checkpoint rule coarse.

| Row (seed 0) | Selected epoch | Updates | Peak training GPU memory | Training time | Head decoding per sample |
|---|---:|---:|---:|---:|---:|
| M0 MS-TCN | 0 | 3,375 | 78 MB | 90 s | 0.004 ms |
| M1 ASRF | 1 | 3,750 | 108 MB | 280 s | 0.137 ms |
| M4 | 7 | 6,000 | 629 MB | 69 s | 0.003 ms |
| M5 | 10 | 7,125 | 629 MB | 145 s | 0.007 ms |

Head times are GPU decoding of a whole half, batched, except ASRF, which decodes one sequence at a time as in its
official evaluation. The shared encoder dominates the full system's cost. In the 2026-10-01 measurement of the same
ResNet-18, reading and letterboxing a frame took 2.64 ms per sample on the CPU and the forward pass 0.22 ms on the GPU.
All rows share these costs.

## 8. Deviations and caveats

- **Native floor and a post-hoc fallback order.** On the calibration half no temporal decoder reaches the native-F1
  floor (0.977). With 7 SC4C and 18 A5C clips, one clip moves macro-F1 by 1.4–2.8 points. The plan did not say which
  setting to report when none qualifies, and the order used here was set after seeing seed 0. It departs from strict
  pre-registration. The fallback now keeps the recall requirement first, then the native floor, then neither. It is
  applied to every method alike and was not changed again for seeds 1–2.
  - Under the earlier order (floor first, then anything), tuned B3 reported a setting with 0.936 segment F1 and recall
    0.945 on the calibration half. The current order gives 0.934 and recall 0.955. MS-TCN with B3 (seed 0) moved from
    recall 0.942 to a setting that meets 0.95.
  - Rows chosen by this fallback should be read with that in mind. These are tuned B2 and B3 and every decoder on
    learned posteriors. The default-decoding rows of the learned heads have no decoding settings and do not use it.
- **Added after seeing results.** The boundary-segment grid (thresholds 0.3–0.9, NMS radius 5) was written into the
  decoder script before its first run on seed 0. It was moved into the configuration before seeds 1–2. Seeds 1–2, the
  pooled rows and the M5 contrasts were all added after seed 0's results were known. The M5 contrasts pair M5 with
  every reference row rather than a chosen subset.
- **A bug in this round's own tooling.** Two decoder diagnostics started in the same second and wrote into one run
  directory, and seed 2's summary overwrote seed 1's. Seed 1 was rerun under a distinct name, and run names now carry
  a tag.
- **Feature masking** (seed 0 only) changed calibration segment F1 by +1.3 points for both M4 and ASRF. On the
  selection half it changed M4 by −1.0 and ASRF by +0.6. It is reported but not adopted.
- **Proxy labels.** Stream labels are inherited from the source clips, and native frames carry their clip's label.
  Neither is an expert frame label.
- **Bootstrap.** Multi-fragment streams reuse clips across streams, so stream-level intervals are somewhat optimistic.
- **Selection half reused.** Checkpoints were chosen on the selection half, so its learned-row numbers are optimistic.
- **Single encoder.** All rows use one encoder seed. Differences between heads are differences given this encoder.
- **Test split unused.** The test split was not touched. A final comparison on it would still be exploratory,
  because earlier rounds used it.

## 9. Weights, runs and code

**Frozen weights.** These are read-only directories on the server, each with a MANIFEST.json.

| Directory | Contents | best.pt SHA-256 |
|---|---|---|
| `models_frozen/b0_frame_encoder/v3_s0` | encoder v3, train log, run and feature manifests | `beb36d80df31…` |
| `models_frozen/boundary_context_m4/round1_s0`, `_s1`, `_s2` | M4 weights, curves, metrics, manifests | `2bb901bfc37e…`, `649b57027e2d…`, `6eaa43c95dae…` |
| `models_frozen/boundary_context_m5/round1_s0`, `_s1`, `_s2` | M5 weights, curves, metrics, manifests | `27fd87cf7228…`, `29ad338df68a…`, `fd6b43dbed00…` |

**Runs.** These are on the server. Each run directory has `run_manifest.json`, `validation_metrics.json`,
`train_log.jsonl`, `training_curves.png`, predictions for both halves, per-row tables and content digests.

- Queue status files: `runs/md_round1/status_s{0,1,2}.json` for M0–M4 and `status_pool_s{0,1,2}.json` for M5 and M3p.
- Feature masking: `status_s0_aug.json`.
- Simple decoders: `runs/20261002-061914-md_simple_decoders-8e834aa6-s0`.
- Decoders on posteriors: `runs/20261002-0618{12,27}-md_posterior_decoders-…`, `…062442…_s1…` and `…062444…_pool…`.
- Feature manifest: `runs/20261002-053334-feature_manifest-6941483d-s0/feature_manifest.json`.

**Files here.**

- `tables_seed0.md`: both halves, every row, costs and paired intervals.
- `seeds.md` and `seeds.json`: three-seed summaries and contrasts.
- `digests_seed0.json`: checkpoint and prediction digests.
- The training-curve figures.

**Code.**

- Model: `echo_routing/temporal/models/boundary_context.py`.
- Losses: `echo_routing/temporal/losses/boundary_context.py`.
- Data, heads, training loop and run wrapper: `echo_routing/temporal/dev_data.py`, `dev_heads.py`, `dev_train.py` and
  `dev_run.py`.
- Metrics and selection rules: `echo_routing/evaluate/dev_metrics.py`.
- Entry scripts: `scripts/train_boundary_context.py`, `run_mstcn_adapted.py`, `run_asrf_official.py`,
  `calibrate_simple_decoders.py`, `posterior_decoders.py`, `model_development_queue.py`, `report_model_development.py`
  and `report_model_development_seeds.py`.
- Configuration: `configs/experiments/model_development_v1.yaml` and `boundary_context_v1.yaml`.

## 10. What the evidence suggests next

These options are not executed. The choice changes the plan's gate, so it is the project owner's decision.

- **Stronger visual input.** Several different heads land near 0.95 segment F1 on the calibration half. This is
  not a demonstrated ceiling. Their remaining shared errors are whole A4C/A5C clips and fragments. That supports
  checking the visual representation, but it does not rule out a mismatch between whole-clip labels and local
  evidence. Testing it means Task 5's EchoViewCLIP features with every comparator upgraded too. The plan starts
  Task 5 only after a trend against the strongest comparator, and that trend was not seen.
- **Better use of the boundary head.** Within the temporal head, the open question is whether the boundary
  information can beat ASRF's segment voting outright, not just match it. One option is training the boundary head
  directly for the pooled output's decision. Another is longer constructed streams. Either would be a new,
  separately tested change.
