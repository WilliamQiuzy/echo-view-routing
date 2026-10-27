# A40: masked-region HOG prediction as an auxiliary target in full-encoder A08 continuation (2026-10-27 UTC)

**Question.** In the same full-encoder classification continuation of A08, does adding a label-free target improve
A5C correction in a way that transfers to held-out sessions, while keeping correct A4C / A5C and the complete
segmentation? The target is the histogram of oriented gradients (HOG) of masked regions, computed from the raw pixels.
- **Predecessors.**
  - A31's last-block adaptation fit the training folds better but gave no held-out session net gain. Its error
    emphasis mostly moved the A4C / A5C boundary.
  - A34 and A36–A39 changed the aggregation or a frozen descriptor. None established a target correction that protects
    correct A5C; A39's two heads each recovered S1 0 / 103.
  - D04 already trained temporal InfoNCE / tilt regression (self-supervised) and A09 a supervised contrast. V02 changed
    the patch readout end to end. A40 is not a first self-supervision, first full-encoder training or first use of
    patches.
- **The changed condition.** The training target now comes from the pixels and has a dense gradient path. A HOG target
  from the raw pixels of masked regions trains the encoder's non-CLS outputs. H is compared with R: the same
  full-encoder continuation, inputs and classification CE, without the auxiliary term.
  - The masked-HOG idea is borrowed from [MaskFeat (Wei et al., CVPR 2022)](https://arxiv.org/abs/2112.09133) for a
    short joint adaptation. A40 is not a reproduction of that paper and not a new algorithm.
  - HOG is a local distribution of edge orientations. It is not the LVOT, an anatomical label or an expert judgement.
- **Scope.** Source seed 0 (A08 all_s0) on A31's two session folds; four models (R and H per fit fold). No development,
  check, test or Pediatric data. A08 trained on these recordings, so the folds hold out only this continuation's
  sessions. Labels are EV9V clip codes.
- **Records.** Configuration `configs/experiments/video_masked_feature_recognition_v1.yaml` (2c69e75, committed before
  any HOG statistic, lambda, forward or training). Code 9cb0878, with a check-only fix 893bc9e. Task sheet
  `claude_prompts/2026-10-26-masked-feature-a5c-training.md`.

## Design

- **Arms.** Both copy A08 all_s0 and train everything from update 0: the MViTv2-S trunk, the 768 → 512 projection,
  the marker stem, the classifier and an auxiliary decoder used only in training.
  - R: loss = CE + 0 × L_HOG. The masked forward and its backward still run.
  - H: loss = CE + λ × L_HOG.
  - Same batches, pixels, masks, decoder initialisation, model RNG start, optimiser, schedule and train-mode dropout /
    stochastic depth. The coefficient is the only difference.
- **Batches.** 900 real updates of 8 windows. 4 are adjacent request centres of one native sequence. The other 4 are
  adjacent centres of one constructed stream, with a label change among the four (A07's group rules, fit pool only,
  stateless draws). CE is the equal-weight mean over the 8 centre clip codes.
  - This differs from A31's 32-run pooled CE. Only H − R isolates the auxiliary target.
- **Inputs.** A31's original request windows: stride-5 centres, 16 samples at 10 Hz (about 1.5 s), 224 centre crop,
  gamma, A08's marker.
- **Auxiliary target.**
  - **HOG.** Computed from the clean uint8 frames before masking, and detached. Gray = RGB mean / 255. Centred
    replicate differences give gradients, binned into 9 unsigned orientation bins with linear circular votes. Each
    8 × 8 cell is L2-normalised, and a 32 × 32 block gives 4 × 4 cells × 9 = 144 dims.
  - **Masking.** Per window, 25 of the 7 × 7 blocks are masked, the same on all 16 slots. The normalised RGB is set to
    0 there; the marker is untouched.
  - **Prediction.** The masked forward's `encoder.norm` output is captured with autograd. Non-CLS tokens are read
    time-major then y, x; token time t predicts slot 2t. The decoder is Linear(768, 144). L_HOG is the MSE over masked
    blocks only.
- **λ.** Fixed once per fit fold from that fold's first 4 training batches before formal training:
  0.25 × median(‖g_CE‖ / ‖g_HOG‖) over the shared trunk (the encoder minus its projection, plus the marker stem). No
  step was taken, and the RNG was restored. λ was committed before training and never tuned.
- **Optimisation.** AdamW: encoder (including the projection) lr 1e-5; marker stem, classifier and decoder lr 1e-4;
  weight decay 1e-4. Linear warm-up over 90 updates, then cosine to 900; clip 1; FP32 with TF32 off. Final weights only.
- **Readout.** The unchanged Base P runs (frozen A08 + B3 cuts) take the argmax of the multiplicity-weighted mean
  softmax of the new model. Equal neighbours merge, and complete sequences are scored with the original evaluator.
  Fit f is read on fold 1 − f, once, after the weights were committed.

**Checks** (`checks.json`; all passed on the second attempt).
- **First attempt.** It stopped on a check-code defect: the token-order check hooked `conv_proj`, and
  `TargetMViT.embed` refuses that hook. The check was fixed (893bc9e, check only) and the 0.0016 GPU h is booked.
- **HOG.** The vectorised target equals an independent float64 pixel loop to within 6.4e-7.
  - Constant images give 0. A horizontal ramp votes only bins 0 and 8, and a vertical ramp only bin 4.
  - Token time t takes slot 2t. The target has no autograd path.
- **Mask.** The mask is exactly the 25 drawn blocks on all slots and channels; everything else is unchanged and the
  marker is untouched.
- **Tokens.** First-stage tokens equal the `conv_proj` (+ marker) output flattened as T, H, W (difference 0). The
  final grid is 8 × 7 × 7 and the token reshape equals an explicit loop.
- **Hook.** It keeps autograd and leaves e224 and q unchanged. It is removed after an error.
- **Init.** On the training-batch path, Init equals A31's cached q to within 1.9e-6 (0 on 32 random windows). Init =
  Base on every P run, sequence and A32 Base row. The readout code run on Init (z = q) also reproduced A39's Base cells.
- **Batches.** Groups follow A07's grid. All 900 × 2 draws are valid, and both arms' rows, pixels, masks and initial
  states are identical.
- **Gradients.** Accumulated backward equals the joint backward (9.3e-10), and a 4 + 4 microbatch split differs by
  1.9e-7.
  - The aux loss alone gives finite, nonzero gradients to the marker stem, `conv_proj`, block 0, block 15 and the
    decoder. Its only zero-gradient tensors are the projection and the classifier.
  - A30's four block-15 tensors (pool_q, rel_pos h / t / w) get no gradient from CE alone, as in A30, but do get one
    from the aux loss.
- **Training mechanics.**
  - All 34,640,549 + 110,736 parameters are trainable and the optimiser binds exactly.
  - Against the fixed limits (loss 1e-4, parameters 1e-5): the 6-update repeat gave 3.7e-8 / 2.3e-6, forced λ = 0 H
    against R 1.3e-7 / 1.8e-6, and 3 + 3 resume (decoder, optimiser, RNG) against 6 gave 6.0e-8 / 2.0e-6.
  - MViT's MaxPool3d backward stays nondeterministic (warnings kept).
- **Code.** All 140 loaded modules match the committed code.

**λ (fit only, committed in 4d52ab4 before training).**
- λ = 0.02834 (fit fold 0) and 0.02840 (fit fold 1). At the start, the HOG gradient on the shared trunk (8.7–18.3) was
  larger than CE's (0.20–6.35), with per-batch ratios 0.013–0.35 (median 0.113 / 0.114).
- Every value was finite and nonzero, the parameters were unchanged and the RNG was restored.

**Training (fit-only monitors).** Each of the four models ran 900 real AdamW steps (about 0.23 GPU h, 13.3 GB peak).
None resumed.
- **H learned the auxiliary target.** On the fit batches of updates 0–3 (eval), masked-HOG MSE fell from 0.0889 /
  0.0933 to 0.0363 / 0.0369 under H. R, with its decoder untrained, stayed at 0.0897 / 0.0939.
- **The classification fit was identical.** On those batches CE went 0.0868 → 0.0448 (fit fold 0) and 0.0692 → 0.0128
  (fit fold 1) for both arms, to four decimals.
- **The auxiliary's share of the trunk update was small.** At every 50th update, λ‖g_HOG‖ on the trunk was 0.03–0.50,
  against ‖g_CE‖ 0.01–16.6. The median ratio was 0.031 / 0.116. The joint gradient (mean norm 3.2–4.7 over the first and
  last 90 updates) was then clipped to 1.
- **Weights.** Both continuations moved the trunk 0.23% (relative L2) from A08. In the frozen weights, H − R is 7% of
  that drift in the trunk, 0.9–1.4% in the marker stem and 0.2% in the projection and classifier.

## Results (held-out folds; OOF = both folds summed)

**All held-out P runs.**
- Fold 0 has 3,384 runs with 109 Base errors, and fold 1 has 3,404 with 145.
- Correct A5C runs are 561 / 589 and correct A4C runs 714 / 699.
- A32's S1 runs (true A5C read as A4C by Base) are 34 / 69, from 12 / 26 A5C-coded source recordings in 12 / 21
  sessions.

| Model | Fixed / broken / net (f0; f1; OOF) | A4C read as A5C fixed | S1 fixed | Correct A5C kept | Correct A4C kept | Segment F1@50 (f0 / f1) | Native macro-F1 | Boundary R (f0 / f1) | Boundary P (f0 / f1) | FP / short FP | FN / missed short |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Base (= Init) | 0 / 0 / 0 | 0 | 0 / 103 | 1,150 / 1,150 | 1,413 / 1,413 | 0.9639 / 0.9499 | 0.9776 / 0.9758 | 0.9676 / 0.9681 | 0.9556 / 0.9420 | 262 / 178 | 183 / 76 |
| R (CE only) | 4/4/+0; 17/16/+1; **21/20/+1** | 20 | 1 | 1,137 | 1,409 | 0.9637 / 0.9495 | 0.9776 / 0.9772 | 0.9671 / 0.9645 | 0.9565 / 0.9456 | 254 / 162 | 193 / 77 |
| H (CE + λ L_HOG) | 4/4/+0; 17/16/+1; **21/20/+1** | 20 | 1 | 1,137 | 1,409 | 0.9637 / 0.9495 | 0.9776 / 0.9772 | 0.9671 / 0.9645 | 0.9565 / 0.9456 | 254 / 162 | 193 / 77 |

FP, FN and their short parts are summed over the two folds.

**H − R, the matched comparison, is exactly 0 in every reported aggregate of both folds.** That covers fixed, broken,
the change table by direction, per-class run accuracy, confusion, S1, kept, segment / native / boundary scores, FP / FN
and deleted boundaries. H and R are different models (frozen sha256 b313d8af / 59d6a544 and ec496a9f / 098166c0). The
readout publishes aggregates only, so identical per-run labels are not separately verified. H − Base = R − Base:

| Contrast, fold | Net runs | S1 fixed | Correct A5C / A4C kept | Segment dF | Native macro-F1 | Boundary R | Boundary P | FP / short FP | FN / missed short |
|---|---|---|---|---|---|---|---|---|---|
| H − R, 0 and 1 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| H − Base, 0 | +0 | +1 | +0 / −3 | −0.0002 | +0.0000 | −0.0005 | +0.0009 | −1 / −2 | +2 / +1 |
| H − Base, 1 | +1 | +0 | −13 / −1 | −0.0003 | +0.0014 | −0.0036 | +0.0036 | −7 / −14 | +8 / +0 |

- **Direction of the changes (OOF, the same for R and H).**
  - Fixes: 20 true A4C runs that Base read as A5C, and 1 S1 run (fold 0, from 1 A5C-coded recording).
  - Breaks: 13 correct A5C into A4C, 3 A4C into A5C, 1 A4C into SC4C, 2 PSAX into SC4C and 1 PSAX into A5C.
  - Over all 1,255 true A5C runs, recall is 0.9163 for Base and 0.9068 for R / H (read as A4C: 103 against 115). Over
    all 1,554 true A4C runs it is 0.9093 against 0.9196 (read as A5C: 135 against 118).
  - A32's S2 runs kept as A5C are 84 / 84 and 144 / 157.
- **Sources.** Fixes come from 23 recordings in 22 sessions and breaks from 28 / 26 (a run counts all its source
  recordings). These session counts are descriptive. Sessions are not confirmed patients, and constructed streams
  share sources.
- **Initial cuts and final boundaries.** The cuts never move (2,044 / 2,057 internal Base boundaries). Relabel and
  merge delete 3 / 24 of them.

**Source-2 rule (fixed before any output): not met.**

| Part | Requirement | Result | Met |
|---|---|---|---|
| Target increment | H − R S1 fixed ≥ +1 per fold and ≥ +5 in total | 0 / 0; total 0 (H 1, R 1) | No |
| A5C signal | H S1 fixed ≥ 11 of 103, from ≥ 3 A5C-coded recordings and ≥ 1 per fold | 1; 1 recording (1 / 0) | No |
| Retention | correct A5C ≥ 1,127 of 1,150 and ≥ R − 3; correct A4C ≥ 1,385 of 1,413 and ≥ R − 3 | 1,137 (R 1,137); 1,409 (R 1,409) | Yes |
| Net | H − Base ≥ 0 and H − R ≥ 0 per fold | +0 / +1; 0 / 0 | Yes |
| System | segment dF (H − Base) ≥ 0 per fold | −0.0002 / −0.0003 | No |
| | segment dF (H − R) ≥ 0; native ≥ −0.003; boundary recall ≥ −0.005 per fold; missed short ≤ Base + 3 | 0 / 0; +0.0000 / +0.0014; −0.0005 / −0.0036; 77 against 76 (limit 79) | Yes |

## Answers

1. **At this coefficient, the masked-HOG target produces no measured held-out difference.** H learned to predict
   masked HOG (fit-batch MSE 0.089 → 0.036, against R's unchanged 0.090), and its trunk ends measurably different from
   R's. But H − R is 0 in every reported run and sequence aggregate of both folds, including S1 (1 and 1 of 103). The
   main question is answered negatively for this configuration.
2. **Why it was small.** λ was fixed by the pre-set rule, 0.25 × the median CE / HOG trunk-gradient ratio of four
   fit batches (≈ 0.028). During training the scaled auxiliary gradient was a median 3% / 12% of the CE gradient on the
   trunk before clipping. This describes what the fixed rule produced. It is not evidence that a larger coefficient
   would or would not help, and no other coefficient was run.
3. **The full-encoder CE continuation itself (R) does not pass the protections either.** It moves the A4C / A5C
   boundary in the familiar direction: 20 A4C-read-as-A5C fixes against 13 correct A5C broken into A4C. It recovers 1
   of 103 S1 runs, has net +0 / +1 and segment dF −0.0002 / −0.0003 against Base, and misses one more short true
   segment.
   - This resembles A31's finding that more trainable capacity fits better without a held-out net gain. A40's batch /
     CE differ from A31, so the R − Base pattern is not attributed to either design.
4. **Closing scope.**
   - Under the fixed rule this configuration ends at source seed 0. There is no source 2 and no mask, target, λ,
     schedule, layer or input variant.
   - The result covers only this masked-HOG auxiliary in a 900-update full-encoder continuation of A08, with this
     batch / CE and the fixed λ rule. It shows no difference at this coefficient on one seed; that is not equivalence.
     It does not disprove self-supervised, spatial or video-model routes. A08 + B3 stays the working reference.
5. **Cost of deployment.** The new branch is a second full A08-size encoder: 16.2 ms GPU per window, against 16.5 ms in
   Base's A31 pass. That doubles the encoder cost, and the 2.3 ms decode is shared. Parameters are 34.6M (Base) plus
   34.6M; the decoder is used in training only.

## Cost, timeline and files

- **GPU 1.44 of 6 h (ledger):**
  - check attempt 1 (failed, 0.0016 h) and attempt 2 (0.032 h);
  - λ (0.004 h);
  - four trainings (0.913 h: 0.227 / 0.230 / 0.227 / 0.230);
  - the readout (0.490 h).
  - A source-2 round would have been projected at a further 1.69 h, well within the cap; it was not triggered.
- **CPU 0.60 of 3 h, as of 2026-10-27 05:58 UTC:**
  - the jobs (Init dry run and freeze): 0.004 h;
  - the agent's active wall clock: 0.591 h in three spans. Waiting during training (04:28:30–05:21:21) and the readout
    (05:24:00–05:51:43) is excluded.
  - All of it is in `runs/video_masked_feature_recognition_v1/ledger.jsonl` on the server. A short closing span after
    the result commit is appended there.
- **New disk:** 532 MB of 3 GB (four 139 MB final states, decoders and logs). No pixel, target or prefix cache was
  written, and no resume file remains. Free space is 146 GB.
- **Timeline (UTC, 2026-10-27).**
  - Dispatch 04:01, configuration 04:09 (2c69e75), code 04:21 (9cb0878) and check fix 04:23 (893bc9e).
  - Checks complete 04:24:40, λ committed 04:26 (4d52ab4), training 04:26–05:21.
  - Weights committed 05:21:48 (a74281d), freeze 05:22:06, readout 05:22–05:51, checked summary 05:58.
  - Codex's review file had no blocking finding before training, while polling, or before the freeze. No helper agents
    were used.
- **Records.** `checks.json`, `lambda_s0.json`, `weights_s0.json` (manifests with fit-only monitors, drift, gradient
  splits, rows / masks sha256) and `readout_s0.json`.
- **Code.** `scripts/train_masked_feature.py`, `echo_routing/temporal/masked_feature.py` and
  `tests/test_masked_feature.py` (10 tests).
- **Server only.** Trained weights, decoders, per-update logs, job logs and all identifiers. The readout kept
  per-window logits and per-run labels in memory only; no per-run prediction file was saved.
