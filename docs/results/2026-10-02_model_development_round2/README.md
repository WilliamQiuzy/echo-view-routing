# Model development, round 2 (2026-10-02): visual input and joint training

This round follows `docs/superpowers/plans/2026-10-02-model-development-round2.md`. It asks three questions. How much
does a stronger visual input change each temporal head? Does the new structure keep an advantage of its own? Does a
paired, end-to-end fine-tuning of the visual encoder add more? Terms are as in the round-1 report
(`docs/results/2026-10-02_model_development/README.md`).

- **E0 and E1.** E0 is the round-1 visual input: the v3 ResNet-18 and its own classification head. E1 is the frozen
  EchoViewCLIP stage-1 image encoder with a new five-class linear probe. E0 and E1 are two complete visual-input
  configurations, including their preprocessing. They are not a change of weights alone.
- **Methods.** ASRF is the official ASRF trained with the shared recipe (M1). M3p has boundary supervision and
  boundary-controlled output pooling. M5 is M3p with the path term in attention. B3 is the change-point decoder tuned
  on the calibration half. B7 is the official EchoViewCLIP stage 1 applied with a sliding window, the full-system
  baseline.
- **Decodings.** Every learned head is reported twice, with its default decoding (argmax, or ASRF's official
  boundary refinement) and with the same B3 grid on its own posteriors. The two are separate rows; no row mixes them.
- **Validation only.** Everything here is on the two validation halves. Both halves have been used during
  development, so every result is an exploratory development result. The test split was not read.

## Answer

1. **The stronger visual input improves clip recognition, but that does not carry into temporal segmentation.**
   - E1's frame-level classifier raises native macro-F1 on both halves (selection 0.965 to 0.978) and fixes two of the
     three A5C clips that every E0 method missed.
   - It also flickers more from sample to sample. Given the reference segmentation, it gets as many selection-half
     segments wrong as E0 (23 of 502).
   - Every temporal head moves the same way: up 0.4–4.0 points of segment F1@50 on the calibration half, down 0.2–4.5
     on the selection half. No E1 − E0 contrast is consistent across seeds.
   - Because all methods move together, none of this belongs to the new structure.
   - The earlier view that recognition had no room left is withdrawn. The open problem is keeping the visual
     classification intact in a continuous segmented output.
2. **Joint fine-tuning does not consistently improve either system.** Paired with identical frames and starting
   weights, fine-tuning the last ViT block changes segment F1@50 by −1.9 to +1.3 points for M5 and −0.8 to +1.2 for
   ASRF. No difference is consistent across seeds on either half.
3. **The new structure has no advantage left against ASRF or B3.**
   - Fine-tuned M5 against fine-tuned ASRF: no consistent difference, with five of six point estimates below zero.
   - Frozen E1 M5 against ASRF: all six point estimates below zero.
   - Against B3 tuned on the same E1 probabilities: no consistent difference.
   - The attention path term (M5 − M3p) shows no consistent difference, so it is not counted as a contribution.
4. **Nothing beats the strongest previous configuration.**
   - Selection half: the strongest is B3 tuned on the v3 probabilities (0.934). Fine-tuned M5 is consistently lower,
     by 3.4–5.2 points.
   - Calibration half: the strongest is the official EchoViewCLIP with B3 on its posteriors (0.972). There is no
     consistent difference (−0.8 to +0.2).
   - The final development goals remain unmet.
5. **M5 loses part of the classification inside its own segments; ASRF does not.** On frozen E1 runs, relabelling
   M5's own segments with the probe's mean probability raises segment F1@50 in all six seed-half cells (+0.3 to +1.4
   points; three of six intervals exclude zero). For ASRF it changes nothing. This supports studying a split where
   the classifier labels and the temporal module only segments, for M5 and not for ASRF. It does not identify where
   M5 loses the information.

## 1. What was fixed before training

- The round-1 protocol is reused unchanged: `protocol.json` with SHA-256 `a897483b…`. That covers the two session-level
  validation halves and their banks, the 1,500-stream training bank, labels, 10 Hz frame indices and the gamma 0.9
  variant. The new configuration `configs/experiments/model_development_v2.yaml` records the parent protocol's digest.
  It was written before any E1 training.
- The checkpoint rule, the native-F1 floor (each visual input's own frame-majority macro-F1 minus 0.005), the recall
  requirement (0.95) and the decoding grids are unchanged from round 1.
- The fallback order for decoding settings was fixed before this round: both constraints, then recall only, then the
  floor only, then neither. Rows chosen by a fallback stay flagged and never count toward the final goals.
- The queue now records a run identity for every job. It is a digest of the experiment, config, protocol, feature
  manifest, job, seed and learning rate. A finished job is skipped only when the identity matches, and a status file
  of another run is refused. Run directories can no longer collide when two jobs start in the same second.
- Training refuses a feature cache that lacks a protocol clip or holds a test row. The input dimension is read from
  the feature manifest.
- 55 tests pass on the server, 22 of them new this round. They cover the guards above, the error-attribution helpers,
  the E1 cache contract and the joint-training checks in section 6.

## 2. The E1 visual input

**Encoder.** The encoder is the frozen EchoViewCLIP stage-1 checkpoint, `models_frozen/echoviewclip_stage1/v1/best.pth`
(SHA-256 `a74c7b23…`), loaded through the authors' model factory at the pinned commit `f6a0d86`. This project trained
it on the EV9V train split with nine classes and selected it on the historical validation split. It is therefore not
an encoder that has never seen this project's development feedback.

**Layer.** The feature is `model.image_encoder(image)`: the CLIP ViT-B/16 class token after `ln_post` and `proj`,
512-d. It is taken before ViFiCLIP's temporal mean and attention pooling and its outer projector.

**Checks on extraction.**
- Only the cached text-prompt tensor was excluded when loading. No visual key was missing or unexpected (86.2M encoder
  parameters).
- The same frame embeds identically alone, inside a batch and after reordering (difference at most 9.5e-7), so the
  layer works on one image at a time.
- An online re-extraction of a short clip matched the cache to within 6.7e-6.
- The frame indices were copied from the round-1 v3 cache for both variants. That covers 3,885 clips (train and
  validation) and no test clip.

**Probe.** A five-class linear probe was trained on the frozen features:
- AdamW, lr 1e-3, weight decay 1e-4, batch 256, at most 20 epochs, patience 5, seed 0;
- per epoch, 3,364 clips drawn class-balanced with replacement (as for v3) and 16 cached samples each.

It was selected at epoch 2 on the selection half's native macro-F1 (0.978). The calibration half never chose it.

**Cache.** The formal cache is `cache_v2/features/16efc85be645`. Its feature manifest
`runs/20261002-142832-e1_probe-38bbcf7c-s0/feature_manifest.json` (SHA-256 `56f59166…`) binds the visual checkpoint,
layer, preprocessing and probe digests.

**A bug in this round's own code.** The first E1 cache index listed the clips in protocol order instead of the v3
order. The training bank is drawn in index order, so the rebuilt bank's digest no longer matched the protocol, and
all first E1 jobs stopped at that check before training. The index is now written in v3 order and checked at cache
time. The rerun reproduced the identical probe (same SHA-256). The failed status files are kept under
`runs/md_round2/*attempt1_bank_order_bug.json`.

**The two frame-level classifiers compared.** Both read every 10 Hz sample.

| Measure | E0 selection | E1 selection | E0 calibration | E1 calibration |
|---|---:|---:|---:|---:|
| Native macro-F1 (clip majority) | 0.965 | 0.978 | 0.982 | 0.985 |
| A5C clips right / all | 8/11 | 10/11 | 16/18 | 16/18 |
| PLAX and SC4C clips right / all | 105/105, 13/13 | 104/105, 12/13 | 80/80, 7/7 | 80/80, 7/7 |
| Wrong samples on multi-fragment streams | 8.7% | 10.3% | 6.1% | 5.3% |
| Label changes per native clip under argmax | 1.62 | 2.93 | 2.69 | 3.29 |
| ORACLE_DIAGNOSTIC: wrong reference segments | 23/502 | 23/502 | 18/518 | 5/518 |

E1 classifies native clips better on both halves. It fixes two of the three A5C clips that every E0 method missed,
but it adds one wrong PLAX clip and one wrong SC4C clip on the selection half. It also flickers more from sample to
sample.

## 3. Error attribution (ORACLE_DIAGNOSTIC)

`scripts/model_development_oracle_diagnostic.py` asks how many errors remain when the reference segmentation is given.
Each multi-fragment stream is cut at its reference class changes (same-class joins are not boundaries). A method's
class probabilities are averaged inside each reference segment, and the segment takes the argmax. This is a
diagnostic, not a deployable result and not a performance bound. The labels are proxy labels from the source clips,
and normal inference never sees these boundaries. Full tables are in `oracle_E0.md` and `oracle_E1.md`.

- **On E1, the temporal heads' posteriors classify given segments worse than the frame-level classifier.** With the
  reference segmentation given, the E1 probe gets fewer segments wrong than every E1 temporal head on both halves.
  This does not hold on E0. There, ASRF seeds 1 and 2 get fewer segments wrong than the E0 head on both halves
  (selection 3.4% and 4.4% against 4.6%; calibration 2.9% and 3.3% against 3.5%).

  | Wrong reference segments | E0 selection | E1 selection | E0 calibration | E1 calibration |
  |---|---:|---:|---:|---:|
  | Frame-level classifier (E0 head / E1 probe) | 4.6% | 4.6% | 3.5% | 1.0% |
  | ASRF, seeds 0/1/2 | 6.2 / 3.4 / 4.4% | 7.0 / 5.8 / 5.8% | 4.8 / 2.9 / 3.3% | 1.9 / 2.5 / 2.5% |
  | M3p | 5.0 / 5.8 / 4.8% | 6.4 / 8.0 / 7.0% | 4.3 / 4.1 / 4.6% | 2.5 / 3.5 / 2.9% |
  | M5 (pooled output) | 5.2 / 6.6 / 5.4% | 6.6 / 6.4 / 8.2% | 3.9 / 4.8 / 4.1% | 2.5 / 3.1 / 3.5% |

  M5's probabilities before pooling (a diagnostic field only) do slightly better than its pooled output, by 0.2–1.6
  points, but still worse than the frame-level classifier.

- **This diagnostic does not establish where the gains of normal inference come from.** Normal inference has no
  reference segmentation. The oracle comparison holds the segmentation fixed, so it only measures how well each
  set of posteriors labels given segments. Section 3a tests the practical version of that question on the models'
  own predicted segments.
- **Errors concentrate on a few source clips.** The frame-level classifier's 23 wrong selection-half segments involve
  only 6 source clips, which appear 28–29 times across streams. Repeated streams are not independent problems.
- **On the selection half the stronger visual input gave no segment-level gain.** Even with the segmentation given,
  E1 is wrong on as many reference segments as E0 (23 of 502).

## 3a. Keeping the visual classification on the models' own segments (frozen E1 runs)

`scripts/model_development_relabel_diagnostic.py` holds each model's own predicted segments fixed. These are the runs
of its default output: ASRF's official refinement, or M5's pooled argmax. No reference boundary is used. Each
interval is labelled two ways on exactly the same intervals: by the model's own mean class probability ("own"), and
by the mean E1 probe probability ("visual").

For M5, "own" is identical to the default output by construction. Its labels are the argmax of the same
probabilities, and an interval is a run where that argmax is constant. The M5 comparison is therefore default
against visual. For ASRF, "own" uses the frame posteriors before the official refinement, so it can differ from
ASRF's default output. Step 1 compares the interval labels. Step 2 writes the labels per sample,
so adjacent intervals with the same label merge under one rule, and scores them. Full tables are in `relabel.md`.

**Step 1: classification on the fixed intervals (multi-fragment streams).** Counts are wrong intervals, own then
visual, per seed.

| Run | Selection half | Calibration half | Visual fixes / breaks (selection) |
|---|---|---|---|
| M5 | 45→39, 39→33, 39→34 | 14→12, 18→15, 14→11 | 6/0, 8/2, 7/2 |
| ASRF | 32→31, 33→34, 36→35 | 12→12, 15→13, 16→14 | 8/7, 1/2, 3/2 |

**Step 2: after merging.** Segment F1@50, visual minus own, in points, with 95% bootstrap intervals.

| Run | Selection half | Calibration half |
|---|---|---|
| M5 | +0.8 [+0.3, +1.6], +1.1 [−0.0, +2.3], +1.4 [+0.4, +2.5] | +0.3 [−0.4, +1.1], +0.9 [+0.2, +1.7], +0.6 [−0.3, +1.5] |
| ASRF | +0.0 [−1.5, +1.4], −0.1 [−0.9, +0.7], +0.0 [−0.9, +0.9] | −0.0 [−0.9, +0.9], +0.4 [+0.0, +1.0], +0.4 [+0.0, +1.0] |

- **For M5, the probe's labels on M5's own segments help.** All six differences are positive, and three of the six
  intervals exclude zero, so this is not consistent across seeds by the three-seed rule. Relabelling never adds a
  boundary; it can only merge adjacent intervals that receive the same label, so recall can only fall. For M5,
  boundary recall did not change in any seed-half cell; for ASRF it moved by at most 0.2 points. Native macro-F1 rises
  or stays (selection seed 2: 0.932 to 0.968), and internal cuts fall.
- **For ASRF, the probe's labels change nothing measurable.**
- **Relabelling does not put M5 ahead of ASRF.** With visual labels, M5 averages 0.907 on the selection half and
  0.970 on the calibration half. ASRF's own output averages 0.909 and 0.968.
- **What this supports.** M5 loses some of the visual classifier's information when it labels its own segments.
  Letting the classifier label and the temporal module only segment is worth studying for M5, but not for ASRF. It
  does not show where the loss arises: the feature transform, mixing across boundaries or the output pooling.

**Case A: an A5C clip that E1 fixes** (`case_A_e1_fixed_a5c.png`). This is selection clip
`2022-04-29_18-53-17_006_320x240_1`, only 9 samples long.
- The E0 head reads 5 of 9 samples as A4C, so its clip label is wrong.
- The E1 probe reads 5 of 9 as A5C. It fixes the clip, but by one sample.
- M5 seeds 0 and 1 pool the clip to A5C, which is correct. Seed 2 reads A4C.
- ASRF reads A4C in all three seeds, so the E1 fix does not survive ASRF.

Of the three A5C clips that every E0 method missed on the selection half, E1 fixes two (this one and
`2022-07-20_14-54-55_320x240_1`).

**Case B: a fragment the probe labels right and M5 gets wrong** (`case_B_probe_right_m5_wrong.png`). This is the
A4C fragment at samples 126–136 of `multi_selection_0004` (source clip `2022-03-28_15-40-18_320x240_30`), which M5
seed 0 reads as A5C.
- The probe favours A4C (about 0.8) apart from a short A5C rise at the end.
- M5's boundary head puts b ≈ 1 at both joins, and the segmentation is right.
- M5's probabilities before pooling already favour A5C over the whole fragment (0.6–0.8), and pooling only flattens
  them.

**Correction (after review).** An earlier version said this case excluded cross-boundary mixing. That claim is
withdrawn. In `echo_routing/temporal/models/boundary_context.py`, three dilated convolutions run before the boundary
head and before attention is restricted. Their receptive field is 15 samples, 7 on each side, and the fragment is
only 10 samples long. Neighbouring fragments can therefore reach the fragment's features before any boundary takes
effect. The loss happens before pooling, but its source (the convolutions, the attention or the gate) is not
identified. The round-3 neighbour-swap check tests this directly.
Across the three M5 seeds and both halves there are 64 fragments where the probe is right and M5 is wrong on at least
80% of the fragment. Counted once per source recording, they come from 25 distinct source clips.

## 4. Frozen-feature matrix: E1 against E0, three seeds

Default decodings, segment F1@50 (higher is better) per seed. Full per-seed tables with per-class right/all,
boundary F1, cuts per minute, same-class cut rate and stream sample error are in `round2_tables.md`.

| System | E0 selection | E1 selection | E0 calibration | E1 calibration |
|---|---|---|---|---|
| ASRF | 0.912 / 0.946 / 0.924 | 0.910 / 0.909 / 0.907 | 0.936 / 0.960 / 0.960 | 0.976 / 0.964 / 0.964 |
| M3p | 0.892 / 0.928 / 0.926 | 0.887 / 0.889 / 0.881 | 0.934 / 0.957 / 0.942 | 0.971 / 0.967 / 0.962 |
| M5 | 0.907 / 0.919 / 0.925 | 0.886 / 0.906 / 0.896 | 0.954 / 0.945 / 0.949 | 0.969 / 0.961 / 0.961 |
| B3 tuned on the frame-level probabilities | 0.934 | 0.894 | 0.934 | 0.959 |
| B7, official EchoViewCLIP, windowed | — | 0.842 | — | 0.957 |
| B7 with B3 on its posteriors | — | 0.880 | — | 0.972 |

| System | Native macro-F1, E1 selection | Native macro-F1, E1 calibration | Recall, E1 selection | Recall, E1 calibration |
|---|---|---|---|---|
| ASRF | 0.948 / 0.966 / 0.966 | 0.993 / 0.975 / 0.975 | 0.942 / 0.950 / 0.953 | 0.977 / 0.972 / 0.977 |
| M3p | 0.978 / 0.976 / 0.978 | 0.978 / 0.983 / 0.975 | 0.942 / 0.937 / 0.950 | 0.977 / 0.967 / 0.980 |
| M5 | 0.978 / 0.968 / 0.932 | 0.986 / 0.983 / 0.977 | 0.945 / 0.950 / 0.929 | 0.972 / 0.965 / 0.957 |
| B7 (default) | 0.957 | 0.986 | 0.961 | 0.975 |

**Contrasts.** Paired per seed, segment F1@50 in points, with the seed category of the 95% bootstrap intervals.
Native clips are resampled by session and streams by stream. Clip reuse across streams is not modelled, so the
intervals are optimistic.

| Contrast | Calibration half | Selection half |
|---|---|---|
| E1 − E0, ASRF | +4.0, +0.4, +0.5: none | −0.2, −3.7, −1.7: none |
| E1 − E0, M3p | +3.7, +1.0, +1.9: none | −0.6, −4.0, −4.5: none |
| E1 − E0, M5 | +1.5, +1.6, +1.2: none | −2.1, −1.3, −3.0: none |
| E1 − E0, M3p with B3 decoding | +1.9, +1.1, +1.3: none | −3.9, −3.4, −3.4: consistently lower |
| E1 − E0, frame-level argmax | −4.5: lower | −10.9: lower |
| Path term on E1 (M5 − M3p) | −0.2, −0.6, −0.0: none | −0.1, +1.7, +1.5: none |
| Path term on E0 (M5 − M3p) | +2.0, −1.2, +0.7: none | +1.5, −1.0, −0.0: none |
| E1: M5 − ASRF | −0.7, −0.3, −0.3: none | −2.4, −0.3, −1.1: none |
| E1: M3p − ASRF | −0.5, +0.3, −0.3: none | −2.3, −2.0, −2.6: consistently lower |
| E1: M5 − B3 tuned | +1.0, +0.3, +0.2: none | −0.8, +1.2, +0.2: none |

"None" means no consistent difference detected, which is not evidence of equivalence. Native macro-F1 and recall
contrasts are in `round2_tables.md`.

What these show:

- **The visual input moves every method together.** E1 raises calibration-half segment F1 for every head and lowers
  it on the selection half. No E1 − E0 contrast is consistent across seeds except two that are consistently lower on
  the selection half. Because all methods move together, none of the change belongs to the new structure.
- **Recall drops on the selection half.** With E1, most selection-half rows fall below 0.95 boundary recall
  (0.929–0.953). Every E1 row stays at or above 0.957 on the calibration half.
- **The path term adds nothing measurable.** As the plan requires, M3p is kept as the comparator. Without a
  consistent M5 − M3p difference, the attention path term is not counted as a contribution.
- **The new model does not lead ASRF on E1.** Every M5 − ASRF point estimate is below zero, and none is consistent.
  Against B3 tuned on the same E1 probabilities there is no consistent difference either.
- **No system meets the final goals on both halves.** That includes the official full EchoViewCLIP (B7). It reaches
  0.957 on the calibration half but only 0.842 on the selection half, below every learned E1 head.

## 5. Start condition for joint fine-tuning (Task 5b)

The plan fixed the condition in advance as (A or B) and C. `scripts/report_model_development_round2.py` evaluates it
mechanically.

- **A holds.** The E1 probe's native macro-F1 is strictly higher than the E0 head's on both halves: 0.978 against
  0.965 on the selection half, 0.985 against 0.982 on the calibration half.
- **B does not hold.** The frame-level oracle segment error drops on the calibration half (0.0097 against 0.0347) but
  ties on the selection half (0.0458 against 0.0458).
- **C holds.** Each listed default decoding keeps calibration recall at or above 0.95 in all three seeds. The
  unrounded minimums are 0.972 for ASRF, 0.967 for M3p and 0.957 for M5.

So the paired joint fine-tuning was started. The fixed candidate rule compares the three-seed mean selection-half
segment F1 under default decoding: M5 0.8959 against M3p 0.8854. The gap of 0.0105 is at least 0.005, so the
candidate is M5.

## 6. Paired joint fine-tuning (Task 5b)

**Setup.**
- *Starting point.* Each run starts from the frozen E1 visual weights and from the E1 head of the same seed.
- *Encoder.* The authors' CLIP VisionTransformer class from the pinned checkout, loaded strictly with the image
  encoder exported from the frozen stage-1 checkpoint (`visual_encoder_fp32.pt`, SHA-256 `c9f87aa2…`).
- *Trainable part.* The last ViT block, `ln_post` and `proj` (7.48M parameters). The other eleven blocks are frozen
  and run without autograd.
- *Optimiser.* AdamW with backbone lr 1e-5, head lr 1e-4 and weight decay 1e-4; two warm-up epochs, then cosine
  decay; gradient clipping at 1; at most 10 epochs with patience 3.
- *Batches.* The round-1 batch plan: 4 training streams and 4 native windows per update, 375 updates per epoch. Each
  update decodes exactly the frames of its streams and windows from the source JPEGs, with the E1 preprocessing.
- *Loss.* It is computed on the whole batch. The trainable part of the encoder runs in checkpointed chunks, and tests
  show the gradients equal those of an unchunked whole-batch pass for both heads (fp32, dropout off). No gradient
  accumulation was needed.
- *ASRF's similarity loss.* Its similarity index is detached from the encoder; the official code also never
  backpropagates through it. This changes nothing for frozen features.
- *Selection.* Checkpoints are chosen on the selection half, as in round 1. B3 is fitted on the calibration half only
  after training.

**Checks.**
- In one-update sanity runs for both systems, all 15 trainable tensors got finite, non-zero gradients and changed.
  Every frozen weight stayed bit-identical.
- The two systems of each seed start from the same visual weights and decode identical frames: the same first-update
  digest, and the same per-epoch frame digests in every epoch they share.
- Frozen layers were still unchanged at the end of all six runs.
- **TF32 finding.** With TF32 enabled, torch 2.14 drifts from the E1 features by up to 4e-3; without it the match is
  3e-6. All joint and timing runs use full fp32.

**Cost.** About 11–12 minutes per epoch on the L40S. The runs stopped after 4–8 epochs, at 0.7–1.4 GPU hours each and
5.9 hours in total.

**Results, default decodings, per seed.**

| System | Selection segF1@50 | Selection recall | Selection native F1 | Calibration segF1@50 | Calibration recall | Calibration native F1 | Calibration cuts/clip | Chosen epoch (rule) |
|---|---|---|---|---|---|---|---|---|
| M5 frozen E1 | 0.886 / 0.906 / 0.896 | 0.945 / 0.950 / 0.929 | 0.978 / 0.968 / 0.932 | 0.969 / 0.961 / 0.961 | 0.972 / 0.965 / 0.957 | 0.986 / 0.983 / 0.977 | 0.080 / 0.063 / 0.080 | — |
| M5 joint | 0.882 / 0.887 / 0.901 | 0.948 / 0.942 / 0.948 | 0.978 / 0.978 / 0.957 | 0.964 / 0.974 / 0.964 | 0.980 / 0.985 / 0.977 | 0.975 / 0.983 / 0.983 | 0.068 / 0.055 / 0.055 | 1 ok, 2 ok, 0 floor unmet |
| ASRF frozen E1 | 0.910 / 0.909 / 0.907 | 0.942 / 0.950 / 0.953 | 0.948 / 0.966 / 0.966 | 0.976 / 0.964 / 0.964 | 0.977 / 0.972 / 0.977 | 0.993 / 0.975 / 0.975 | 0.038 / 0.030 / 0.046 | — |
| ASRF joint | 0.907 / 0.903 / 0.900 | 0.929 / 0.950 / 0.937 | 0.945 / 0.957 / 0.966 | 0.972 / 0.976 / 0.970 | 0.980 / 0.982 / 0.985 | 0.990 / 0.986 / 0.978 | 0.034 / 0.008 / 0.017 | 0, 4, 3, all floor unmet |

All three joint ASRF checkpoints and joint M5 seed 2 miss the selection-half native-F1 floor (0.973). By the rule
they are the highest-segment-F1 epochs, kept and flagged. Joint M5's selection-half recall stays below 0.95 in every
seed.

**The four questions.** Segment F1@50 differences per seed, in points, with 95% bootstrap intervals.

| Question | Calibration half | Selection half |
|---|---|---|
| Does joint fine-tuning improve M5? (joint − frozen) | −0.5 [−1.4, +0.2], +1.3 [+0.1, +2.4], +0.3 [−1.1, +1.7]: none | −0.3 [−1.8, +1.1], −1.9 [−3.5, −0.4], +0.5 [−1.1, +2.0]: none |
| Does it improve ASRF? (joint − frozen) | −0.4 [−0.9, +0.0], +1.2 [+0.2, +2.2], +0.6 [−0.2, +1.5]: none | −0.3 [−1.3, +0.9], −0.6 [−2.2, +0.8], −0.8 [−1.5, −0.2]: none |
| Does fine-tuned M5 beat fine-tuned ASRF? | −0.8 [−2.1, +0.5], −0.2 [−1.2, +0.9], −0.6 [−1.8, +0.5]: none | −2.5 [−4.8, −0.3], −1.5 [−3.3, +0.0], +0.1 [−1.8, +2.1]: none |
| Does it beat the strongest previous configuration on this half? | B7 with B3 (0.972): −0.8 [−2.3, +0.8], +0.2 [−1.2, +1.7], −0.8 [−2.4, +0.8]: none | B3 tuned on v3 (0.934): −5.2 [−8.2, −2.3], −4.7 [−7.6, −1.9], −3.4 [−6.0, −0.6]: consistently lower |

"Strongest previous configuration" was fixed in code before the joint results were compared. Per half, it is the
system with the highest three-seed mean segment F1@50 among all non-joint systems of rounds 1–2 on this protocol:
every head under both decodings, tuned B1/B2/B3 on either visual input, and B7 with and without B3. Joint M5 against
B3 tuned on the same E1 probabilities shows no consistent difference. Joint M5 against plain B7 is consistently higher
on the selection half (+4.0, +4.5, +5.8), but plain B7 is not the strongest configuration.

## 7. Same-hardware timing

The timing uses one L40S. Each sequence is processed alone (batch = 1), after warm-up, with CUDA synchronisation, in
fp32. The sequences are the same 60 selection-half sequences in three length groups for every system. CPU
preprocessing runs in one process. Times are milliseconds per 10 Hz sample, from `timing_e1_systems.json` and
`timing_b7.json`.

| Sequence length (samples) | 1–60 | 61–120 | 121–400 |
|---|---:|---:|---:|
| E1 preprocessing (CPU) | 1.52 | 1.55 | 1.56 |
| E1 encoder forward (GPU) | 1.48 | 1.54 | 1.61 |
| M5 head with decoding | 0.19 | 0.09 | 0.05 |
| ASRF head with official refinement | 0.58 | 0.25 | 0.13 |
| **Full system: E1 + M5** | **3.20** | **3.18** | **3.21** |
| **Full system: E1 + ASRF** | **3.59** | **3.35** | **3.30** |
| Full system: E1 probe + B3 | 3.02 | 3.12 | 3.19 |
| **Full system: official EchoViewCLIP, windowed (B7)** | **12.25** | **12.00** | **12.24** |
| B7 preprocessing / model | 6.19 / 6.07 | 6.10 / 5.90 | 6.36 / 5.88 |

- **Encoder and preprocessing dominate.** At batch = 1 they take most of the time. The M5 and ASRF heads differ by
  0.1–0.4 ms per sample, which is 3–11% of the full system.
- **Head throughput (batched, listed separately).** The M5 head takes 0.004–0.061 ms per sample when all sequences
  of a length group go through it at once. ASRF's official decoding always runs one sequence at a time, so its
  "batched" figure (0.046–0.235) loops over sequences without GPU batching. It is not a batched-throughput number.
- **B7 is costlier as officially applied.** Its 16-frame windows overlap, so each frame is decoded and encoded about
  three times. B7 also runs in its own environment (torch 1.13). The difference is therefore a measured system cost of
  the official application, not an architectural speed-up.
- **Round-1 claim withdrawn.** The "20 times" figure from round 1 compared batched and unbatched decoding. It is
  superseded by this table.

## 8. Deviations, bugs and caveats

- **Bugs in this round's own code.**
  - The E1 cache index order (section 2) was caught by the bank-digest check before any training.
  - Two jobs could share a run directory when started in the same second. Run directories are now unique.
  - TF32 numerics are disabled for the encoder (section 6).
- **Disclosed post-hoc choices.**
  - The fallback order fixed before this round was itself set after seeing round-1 seed 0 (round-1 report, section
    8).
  - The relabelling diagnostic (section 3a) and the "strongest previous" comparison were added on request after the
    frozen-feature results were known. Both use fixed definitions applied to every row.
- **Fallback rows.** Several rows use fallback decoding settings or checkpoints that miss the native floor. They are
  flagged in `round2_tables.md` and `round2.json`, and none counts toward a goal.
- **Same validation data throughout.** Both validation halves have been used for development since round 1. The
  calibration half chose the decoders and the selection half chose the checkpoints, so neither is an unseen
  confirmation set. The test split was not read.
- **Intervals are optimistic.** Multi-fragment streams reuse clips, and that dependence is not modelled. Seeds share
  the same pretrained visual weights and data, so three seeds are not three independent replications.
- **Fragile A5C fixes.** E1's A5C fixes are small in number and fragile: the clip in case A is fixed by one sample
  out of nine.
- **E1 is not an external encoder.** This project trained it on EV9V and selected it on the historical validation
  split.

## 9. Weights, runs and code

**Frozen this round.** These are read-only directories on the server, each with a MANIFEST.json. Round-1 versions
are untouched.

| Directory | Contents | Checkpoint SHA-256 (seeds 0/1/2) |
|---|---|---|
| `models_frozen/e1_echoviewclip_probe/v1` | probe, feature manifest, extraction metadata, exported encoder | probe `41f2033d…`, encoder `c9f87aa2…` |
| `models_frozen/asrf_adapted/round2_e1_s{0,1,2}` | ASRF on frozen E1 | `ed1214c8…`, `beaa6a7e…`, `a599d40c…` |
| `models_frozen/boundary_context_m3p/round2_e1_s{0,1,2}` | M3p on frozen E1 | `38940fc0…`, `365be346…`, `823b3f45…` |
| `models_frozen/boundary_context_m5/round2_e1_s{0,1,2}` | M5 on frozen E1 | `6d88bc5d…`, `ec21b674…`, `147faac8…` |
| `models_frozen/joint_visual_m5/round2_s{0,1,2}` | joint M5: trainable visual part and head | `97a8aa6e…`, `e99e3fdb…`, `70a5c08d…` |
| `models_frozen/joint_visual_asrf/round2_s{0,1,2}` | joint ASRF: trainable visual part and head | `6c57f791…`, `7564197b…`, `355db2a5…` |

**Runs on the server.**
- E1 matrix: `runs/md_round2/e1_s{0,1,2}_status.json`. Joint runs: `runs/md_round2/joint_status.json`.
- Reports: `runs/md_round2/report_final_v2/`, from `report_spec_final.json`. It is a rerun of `report_final/` after
  review: identical contrasts and gate, with the raw argmax removed from the strongest-previous pool, where it
  never won. Timing: `runs/md_round2/timing/`.
- Diagnostics: oracle `runs/20261002-224210-…E0…` and `…224209-…E1…` (reruns with error runs split at joins; oracle
  numbers unchanged from the first runs `…140922…` and `…145334…`); relabelling `…160508-md_relabel_diagnostic…`.
- Official EchoViewCLIP on the protocol halves: `cache_v2/echoviewclip_windowed/beb36d80df31_protocol_a897483b776e`.

**Files here.**
- `round2_tables.md` and `round2.json`: every system and seed, with per-class right/all and all contrasts.
- `oracle_E0.md`, `oracle_E1.md` and `relabel.md`: the two diagnostics.
- `case_A_e1_fixed_a5c.png` and `case_B_probe_right_m5_wrong.png`: the two cases.
- `timing_e1_systems.json` and `timing_b7.json`: the timing runs.

**Code.**
- Extraction and probe: `scripts/extract_echoviewclip_features.py` and `scripts/train_feature_probe.py`.
- Joint training: `echo_routing/temporal/joint_visual.py`, `scripts/train_joint_visual_temporal.py` and
  `scripts/joint_finetune_queue.py`.
- Diagnostics: `scripts/model_development_oracle_diagnostic.py` and `scripts/model_development_relabel_diagnostic.py`.
- Reporting and timing: `scripts/report_model_development_round2.py`, `scripts/time_systems.py` and
  `scripts/time_echoviewclip_windowed.py`.
- Configuration: `configs/experiments/model_development_v2.yaml`.
- Updated: `scripts/model_development_queue.py` (run identity), `scripts/build_window_plans.py` (`--protocol`),
  `scripts/posterior_decoders.py` (prediction directories, per-source native floor) and `scripts/_common.py`
  (unique run directories).

## 10. Status of the final goals, and what the evidence supports next

The goals are unchanged from the round-1 plan, and none is met.
- **Segment F1@50.** Not met. No candidate exceeds the strongest baseline by 1 point on segment F1@50.
- **Native classification.** Not shown. No candidate beats the strongest full system on native classification in a
  consistent way.
- **Cuts at equal recall.** Not met. Joint M5 has fewer internal cuts than plain B7, but not fewer than B7 with B3 on
  the calibration half, and its selection-half recall is below 0.95 in every seed.

What this round establishes, as evidence for the next decision:

- **Recognition gains need a temporal output that preserves them.** A better clip classifier (E1) did not give better
  segmentation, and both temporal heads blurred part of its classification.
- **For M5, the loss is inside its own segments.** Relabelling M5's segments with the visual classifier recovers
  0.3–1.4 points of segment F1@50 in every seed. The cases show the loss happening before pooling. Mixing from
  neighbouring fragments through the convolutions is not excluded (see the correction under case B).
- **Two possible next experiments.** One is a decoder that takes segment extents from the temporal model and labels
  from the frozen classifier. The other is a head trained to preserve the classifier's posterior inside predicted
  segments. Either is a new, separately tested change, and the strongest-baseline bar stays as it is.
- **The paper stays paused.**
