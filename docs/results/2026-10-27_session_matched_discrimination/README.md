# A42: same-session A4C / A5C margin ranking against cross-session rewiring in full A08 continuation (2026-10-27 UTC)

**Question.** In a full continuation of A08, does comparing A4C and A5C windows from the *same recording session* give
more transferable A5C corrections than the identical training windows with their A4C / A5C pairs rewired *across*
sessions?
- **Predecessors.**
  - A31's last-block correction with error emphasis mostly moved the A4C / A5C boundary, with no held-out net gain. A31
    supplies the two session folds used here.
  - A09 / A09-R1 trained supervised contrast with cross-session *same-class* positives on global embeddings (closed).
    R08 found that fitted E1 clip embeddings cluster by view rather than by session, in sample, and trained nothing. No
    predecessor established a session shortcut.
  - A34 and A39 (frozen descriptors / heads) recovered 0 of the S1 runs. A40, the closest design (full A08 continuation,
    900 updates, A07's 8-window batch), recovered 1 / 103 with and without its auxiliary target. A41's EchoJEPA branch
    fixed 12 / 103 but broke 220 correct runs.
- **The changed condition.** Only the *association* of four A4C and four A5C endpoint windows inside a pairwise margin
  ranking term. S pairs each A5C window with the A4C window of its own session. C pairs it with the A4C window of the
  next chosen session (cyclic). Images, labels, CE, exposure and computation are identical in the two arms.
  - This is not A09's SupCon, a frozen local readout, A41's LoRA, error-priority sampling or a new anatomical label.
  - **Hypothesis, not a finding:** a same-session pair might focus the supervision on differences between similar
    acquisitions. A session is not a verified patient or acquisition-geometry match. Clip codes are weak reference
    labels, not LVOT visibility or expert truth.
- **Scope.** Source seed 0 (A08 all_s0) on A31's two session folds; four models (C and S per fit fold). No development,
  check, test or Pediatric data. A08 trained on these recordings, so the folds hold out only this continuation's
  sessions. Labels are EV9V clip codes.
- **Records.** Configuration `configs/experiments/video_session_matched_discrimination_v1.yaml` (b84a78c, committed
  before any pool, pixel, loss or model computation; amended before training in 70362d4 and fba4eaf, see Checks). Code
  f5a4155 + 70362d4 + fba4eaf. Task sheet `claude_prompts/2026-10-27-session-matched-a5c-training.md`.

## Design

- **Native pair bank** (per fit pool; kinds, recording ids and clip codes only). A session qualifies if the pool holds
  at least one native A4C and one native A5C recording of it. Sources that appear only inside constructed streams are
  not candidates. The bank was asserted before any pixel: fold 0 has 35 sessions (55 A4C / 48 A5C recordings, 77
  distinct recording pairs) among 740 native recordings, fold 1 has 44 (72 / 54, 89) among 747. Selection never used
  a Base mistake, an S1 / harm set, a held-out result or a threshold.
- **Batch.** 16 windows per update, the same in both arms and in this order:
  - slots 0–7: A40's batch rules (4 adjacent native centres, then 4 adjacent constructed centres with a label change;
    A07 group rules), from RNG [20261027, 42, s, f, u, 1];
  - slots 8–11: one A4C window from each of 4 distinct bank sessions; slots 12–15: one A5C window from the same 4
    sessions in the same order (RNG [20261027, 42, s, f, u, 2]: sessions, then per session an A4C recording, an A5C
    recording and one original stride-5 centre of each). Duplicates with slots 0–7 are kept (7 / 6 of 7,200 extra slots).
- **Loss.** d(x) = logit_A5C − logit_A4C. L = mean CE over all 16 centre clip codes + 0.5 × mean_i softplus(−(d5_i −
  d4_partner(i))). S: partner(i) = i (same session). C: partner(i) = (i + 1) mod 4, so every C edge is cross-session.
  Coefficient 0.5 and temperature 1 were fixed by the task; nothing was searched. All 5 classes stay in the CE.
- **Model and optimisation.** Each arm is a bit-identical copy of A08 all_s0 with every parameter trainable from update
  0 (34,640,549). Eval mode throughout (dropout / stochastic depth off), gradients on. AdamW: encoder including the 512
  projection lr 1e-5 (397 tensors), marker stem and classifier lr 1e-4 (3 tensors); weight decay 1e-4; linear warm-up
  90 then cosine to 900; global clip 1. FP32, TF32 off, `use_deterministic_algorithms(True, warn_only=False)`.
  Microbatch 1: 16 separate forwards, differentiable logits, one backward and one step. 900 real updates per model,
  order C_f0, S_f0, C_f1, S_f1; final weights only. Model RNG 2026102742 + 10 s + f.
  - **Deterministic skip pool (engineering).** torchvision MViT's residual `MaxPool3d` (blocks 1, 3, 14) has no
    deterministic CUDA backward and raises under `warn_only=False`. Training copies use a gather max pool: indices from
    `max_pool3d(return_indices=True)` under no_grad, values gathered. Outputs are bitwise identical and the state dict
    is unchanged; the readout uses the original modules.
- **Readout.** The unchanged Base P runs (frozen A08 + B3 cuts, original nearest-centre multiplicities) take the argmax
  of the multiplicity-weighted mean softmax of the new model; equal neighbours merge (RLE); complete sequences are
  scored with the original evaluator. Fit f is read on fold 1 − f, once, after the weights were committed.

**Checks** (`checks.json`, attempt 3; earlier attempts kept).
- **Attempt 1 failed one gate, and that original criterion did not pass.** The FP32 microbatch-1 against batch 8 + 8
  gradient differed by 2.52e-5 relative (2.70e-6 of 0.107), above the fixed 1e-5. A check-only diagnosis on the same
  batch (`diagnose_numerics.json`) gave 1.5e-15 for the same comparison in FP64 (no measurable batch coupling on that
  checked fit batch; not a proof for every input) and 7.8e-5 between
  each FP32 grouping and FP64 (FP32 round-off through 16 blocks). Before any training, the configuration was amended
  (70362d4): the coupling gate is evaluated in FP64 at the unchanged 1e-5. The FP32 value is recorded and is not
  called a pass.
- **Attempt 2 was not valid.** It reported success, but an inline comment had swallowed the skip-pool gate (its value,
  2.8e-7, was computed but not enforced), and the production FP32 gradient was not gated. Codex flagged both before
  training. Fix fba4eaf (configuration amended): each gate is its own entry, the enforced gate list is written out, a
  missing required gate fails, and the formal 16-graph FP32 gradient is gated against an independent per-window chain
  rule (J^T dL/dz) at 1e-5.
- **Attempt 3: all 25 enforced gates pass, none missing.**
  - **Plan.** Both banks equal the asserted counts. All 900 × 2 plans: slots 0–7 follow A07's rules, slots 8–15 are
    native A4C / A5C windows of 4 distinct bank sessions, every S edge is same-session and every C edge cross-session,
    and the endpoint / window / CE multisets are identical between arms. The 7,200 extra slots per fold cover 894 /
    1,277 distinct windows from 103 / 126 recordings and all 35 / 44 bank sessions (session draws 85–123 / 62–102).
    No session crosses folds.
  - **Init.** The training copy equals the pinned A08 bitwise; its online microbatch-1 fit-window logits equal A31's
    cached q within 1.9e-6 (fit-only online parity). The patched forward equals the original bitwise. Separately, Base
    P runs, sequences and A32 rows were reproduced exactly from the cached q (no held-out pixel was read for Init), and
    the readout code on Init reproduced A32's rows and A40's Base cells.
  - **Loss.** Torch CE / rank / total and their logit gradients equal an independent float64 loop (≤ 1.6e-16), for
    both partner maps and at margins of ±1e3. At zero logits CE = ln 5 (denominator 16) and rank = ln 2 (total ln 5 +
    0.5 ln 2). The margin is antisymmetric under swapping A4C / A5C and the rank is shift-invariant. The rank-only
    gradient reaches the marker stem, `conv_proj`, blocks 0 and 15, the 512 projection and only the A4C / A5C classifier
    rows.
  - **Gradients.** FP64 grouping 1.5e-15; production FP32 16-graph against the per-window chain rule 2.1e-7; gather
    against original max-pool backward 2.1e-7 (`warn_only` restored to False). Only A30's four block-15 tensors
    (pool_q, rel_pos h / t / w) get no gradient from CE + rank.
  - **Mechanics.** Optimiser binds exactly; 6 real steps move every group (blocks 386 / 390 tensors, `conv_proj`,
    position / norm, projection, marker stem, classifier). 6-update repeat, forced C partner = identity against S, and
    3 + 3 resume against 6 are all 0 / 0 (limits 1e-4 / 1e-5). C differs from S after 6 updates (5.6e-4). The gradient
    monitor leaves the update unchanged. Readout pooling and relabelling are unchanged with every reference label
    replaced by a constant.
  - **Code.** All 142 loaded modules equal their committed blobs.
  - **Cost.** 1.68 s per real update (25.4 GB peak); source-0 projection 3.16 of 4 GPU h including the 0.5 h reserve.

**Training (fit-only monitors; `weights_s0.json`, committed in aaab220 before the readout).** Each of the four models
ran 900 real AdamW steps (about 0.41 GPU h each, 25.8 GB peak), from the same A08 state, with no resume. Final sha256:
C_f0 61e38779, S_f0 91cf5e41, C_f1 c231f467, S_f1 cf515a6e.
- **The rank term was almost always satisfied.** On the training pairs, pair accuracy was 0.986–0.994 over the first 90
  updates and 0.997–1.000 over the last 90. At every 50th update the scaled rank gradient on the shared trunk was
  0–1.03 against CE's 0.03–10.6, with a median ratio of 0.0000 / 0.0000 / 0.0003 / 0.0005 (C_f0 / S_f0 / C_f1 / S_f1).
  The joint gradient (mean norm 2.1–3.4) was clipped to 1.
- **Each arm fit its own pairing.** Over the last 90 updates the arm's own rank loss was 0.0001 (C) and 0.001–0.003
  (S); same-session pairs stayed the harder ones for both models. On the fit batches of updates 0–3 (eval), CE went
  0.090 → 0.007–0.008 (fit fold 0) and 0.137 → 0.011–0.012 (fit fold 1), and both pairings' rank losses fell below
  1e-5.
- **Weights.** Both arms moved the encoder trunk 0.31–0.33% (relative L2) from A08, the projection 0.72–0.75%, the
  marker stem 26–31% and the classifier 17%, almost identically between C and S. Only A30's four
  block-15 tensors never move.

## Results (held-out folds; OOF = both folds summed)

**All held-out P runs.**
- Fold 0 has 3,384 runs with 109 Base errors, and fold 1 has 3,404 with 145.
- Correct A5C runs are 561 / 589 and correct A4C runs 714 / 699.
- A32's S1 runs (true A5C read as A4C by Base) are 34 / 69, from 12 / 26 A5C-coded source recordings in 12 / 21
  sessions.
- Base was reproduced exactly: A32's sequence rows and A40's Base cells, in both folds.

| Model | Fixed / broken / net (f0; f1; OOF) | A4C read as A5C fixed | S1 fixed (f0 / f1) | Correct A5C kept | Correct A4C kept | Segment F1@50 (f0 / f1) | Native macro-F1 | Boundary R (f0 / f1) | Boundary P (f0 / f1) | FP / short FP | FN / missed short |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Base (= Init) | 0 / 0 / 0 | 0 | 0 / 103 | 1,150 / 1,150 | 1,413 / 1,413 | 0.9639 / 0.9499 | 0.9776 / 0.9758 | 0.9676 / 0.9681 | 0.9556 / 0.9420 | 262 / 178 | 183 / 76 |
| C (cross-session pairs) | 13/9/+4; 31/47/−16; **44/56/−12** | 40 | 3 (3 / 0) | 1,105 | 1,405 | 0.9667 / 0.9443 | 0.9832 / 0.9786 | 0.9666 / 0.9584 | 0.9627 / 0.9535 | 237 / 134 | 219 / 88 |
| S (same-session pairs) | 14/8/+6; 32/48/−16; **46/56/−10** | 41 | 4 (4 / 0) | 1,105 | 1,405 | 0.9665 / 0.9447 | 0.9832 / 0.9786 | 0.9666 / 0.9589 | 0.9623 / 0.9540 | 237 / 135 | 218 / 87 |

FP, FN and their short parts are summed over the two folds.

**S − C, the matched comparison, is at most two runs per fold.** The two arms' run labels differ in 2 of 3,384 and
2 of 3,404 runs (from the private cache).

| Contrast, fold | Net runs | S1 fixed | Correct A5C / A4C kept | Segment dF | Native macro-F1 | Boundary R | Boundary P | FP / short FP | FN / missed short |
|---|---|---|---|---|---|---|---|---|---|
| S − C, 0 | +2 | +1 | +1 / 0 | −0.0002 | 0.0000 | 0.0000 | −0.0005 | +1 / +1 | 0 / 0 |
| S − C, 1 | 0 | 0 | −1 / 0 | +0.0004 | 0.0000 | +0.0005 | +0.0005 | −1 / 0 | −1 / −1 |
| S − Base, 0 | +6 | +4 | −1 / −6 | +0.0026 | +0.0057 | −0.0010 | +0.0067 | −15 / −14 | +1 / +2 |
| S − Base, 1 | −16 | 0 | −44 / −2 | −0.0051 | +0.0028 | −0.0092 | +0.0120 | −10 / −29 | +34 / +9 |
| C − Base, 0 | +4 | +3 | −2 / −6 | +0.0028 | +0.0057 | −0.0010 | +0.0072 | −16 / −15 | +1 / +2 |
| C − Base, 1 | −16 | 0 | −43 / −2 | −0.0055 | +0.0028 | −0.0098 | +0.0115 | −9 / −29 | +35 / +10 |

- **Direction of the changes (OOF).**
  - Fixes (S / C): 41 / 40 true A4C runs that Base read as A5C, 4 / 3 S1 runs and 1 PLAX run.
  - Breaks (both 56): 44 correct A5C into A4C (S 1 + 43, C 2 + 42), 1 A5C into PSAX, 6 correct A4C into A5C, 2 A4C
    into SC4C, and 3 others among PLAX / PSAX / SC4C.
  - Over all 1,255 true A5C runs, recall is 0.9163 for Base against 0.8837 (S) and 0.8829 (C); read as A4C 103 against
    143 / 144. Over all 1,554 true A4C runs it is 0.9093 against 0.9305 / 0.9299; read as A5C 135 against 100 / 100.
  - A32's S2 runs kept as A5C are 83 / 84 and 125 / 157 for S (C 82 / 84 and 125 / 157).
  - Fit fold 0's models, read on held-out fold 1, cause most of the damage: 43 / 42 correct A5C into A4C there.
- **Sources.** S's 4 S1 fixes come from 3 A5C-coded recordings in 3 sessions, all in held-out fold 0 (C: 3 / 3 / 3,
  also fold 0). All fixes come from 44 / 43 recordings in 39 / 38 sessions, and all breaks from 53 / 54 recordings in
  41 / 42 sessions (S / C; a run counts every source recording in it). These counts are descriptive: sessions are not
  confirmed patients, and constructed streams share sources.
- **Initial cuts and final boundaries.** The cuts never move (2,044 / 2,057 internal Base boundaries). Relabel and
  merge delete 17 / 61 of them for S and 18 / 61 for C.

**Source-2 rule (fixed before any output): not met.** A source-2 repetition would also not have fitted the remaining
shared budget (about 2.1 GPU h needed against 1.69 h left).

| Part | Requirement | Result | Met |
|---|---|---|---|
| Target increment | S − C S1 fixed ≥ +1 per fold and ≥ +5 in total | +1 / 0; total +1 (S 4, C 3) | No |
| A5C signal | S S1 fixed ≥ 11 of 103, from ≥ 3 A5C-coded recordings and ≥ 1 per fold | 4; 3 recordings (3 / 0) | No |
| Retention, A5C | correct A5C ≥ 1,127 of 1,150 and ≥ C − 3 | 1,105 (C 1,105) | No |
| Retention, A4C | correct A4C ≥ 1,385 of 1,413 and ≥ C − 3 | 1,405 (C 1,405) | Yes |
| Net | S − Base ≥ 0 and S − C ≥ 0 per fold | +6 / −16; +2 / 0 | No |
| System | segment dF (S − Base) ≥ 0 and (S − C) ≥ 0 per fold | +0.0026 / −0.0051; −0.0002 / +0.0004 | No |
| | native (S − Base) ≥ −0.003 per fold | +0.0057 / +0.0028 | Yes |
| | boundary recall (S − Base) ≥ −0.005 per fold | −0.0010 / −0.0092 | No |
| | missed short true ≤ Base + 3 (OOF) | 87 against 76 (limit 79) | No |

## Answers

1. **Same-session pairing made no material difference.** S and C differ in 2 runs per held-out fold. S − C is +1 / 0
   in S1 fixes (4 against 3 of 103), +2 / 0 net, and −0.0002 / +0.0004 in segment F1. The main question is answered
   negatively for this configuration: same-session association did not give a more transferable A5C correction than
   the identical endpoints rewired across sessions.
2. **Why the arms barely differ.** The pairwise term was nearly satisfied from the start. Base already ranked almost
   every training A5C window above its paired A4C window, in either pairing (pair accuracy about 0.99). So the scaled
   rank gradient was a median ≤ 0.05% of CE's on the trunk, and the two arms received almost the same updates. This
   describes the fixed configuration (coefficient 0.5, logistic margin on clip codes). It is not evidence that another
   weight, margin or pair rule would or would not help, and none was run.
3. **The shared continuation moves the A4C / A5C boundary against A5C, as in A31 / A40.** Against the target, S fixes
   4 of 103 S1 runs, while all fixes are 46 and all breaks 56, 45 of them correct A5C runs (44 into A4C). Most fixes
   are A4C runs read as A5C (41 / 40). Correct A5C falls from 1,150 to 1,105 (floor 1,127),
   missed short true segments rise from 76 to 87 / 88, and fold 1 loses segment F1 (−0.0051 / −0.0055) and boundary
   recall. Fold 0 gains segment F1 (+0.0026 / +0.0028) with fewer false positives. Native macro-F1 rises in both folds
   (+0.0057 / +0.0028), but the two-fold mean segment F1 falls from 0.9569 (Base) to 0.9556 (S, −0.0013) and 0.9555 (C).
   - Both arms differ from A40's CE-only continuation in two ways: 8 extra paired native A4C / A5C windows inside the
     common CE (A4C / A5C are 69–70% of the CE windows), and the rank term. The shared shift is therefore not attributed
     to either component.
4. **Closing scope.**
   - Under the fixed rule this configuration closes at source seed 0. There is no source 2 and no alternative rank / CE
     weight, pair bank, crop, hard mining, encoder or schedule.
   - The result covers only same-session vs cyclically rewired logistic margin ranking (coefficient 0.5) on clip codes,
     added to CE in a 900-update full A08 continuation with this 16-window batch, at one source seed. It is not
     equivalence, and it neither shows nor rules out a session shortcut. Nothing here shows that pairing learned the
     LVOT or removed acquisition effects. A08 was trained on these recordings, so the folds hold out only this
     continuation's sessions, and the labels are clip codes, not expert truth. No source 2, development / test /
     Pediatric evaluation or A43 follows. A08 + B3 stays the working reference.
5. **Cost of deployment.** The new copy is a second full A08-size encoder: 16.3 ms GPU per window, against 16.5 ms in
   Base's A31 pass, so encoder cost doubles; the 2.4 ms decode is shared. Parameters are 34.6M (Base) plus 34.6M.

## Verification

- `readout_s0.json` (sha256 18e66cd1…) equals the server copy. The OOF fixed / broken equal the per-fold sums, and
  the rule entries were recomputed from the cells.
- On the server, the private cache (`readout_cache_s0.npz`, sha256 b4ad5a5e…: per-window logits and per-run
  labels) reproduces every cached run label from the logits through the original pooling. It also reproduces every
  published fixed / broken / S1 / kept count (CPU, 0.001 h).
- Freeze ordering: weights committed 00:41:57 (aaab220), freeze 00:42:18 and readout start 00:42:19 (UTC). Original
  A08 s0 / s2 files unchanged.
- Codex independently re-scored the private cache (134 assertions, 0 discrepancies, incl. all 20,364 Base / C / S run
  predictions, rebuilt sequences, F1 / FP / FN counts and the 13 gate booleans). That audit and Codex's monitoring
  ran outside this round's ledger and are not booked as worker CPU.

## Cost, timeline and files

- **GPU 2.31 of 4 h (ledger):**
  - the engineering probe (random tensors, 0.002 h);
  - check attempt 1 (failed, 0.051 h), the numerics diagnosis (0.008 h), attempt 2 (not valid, 0.056 h) and attempt 3
    (0.056 h);
  - four trainings (1.638 h: 0.408 / 0.410 / 0.410 / 0.410);
  - the readout (0.495 h).
- **CPU (agent active wall clock plus server CPU jobs):** 0.64 of 3 h, as of 2026-10-28 01:16:15 UTC.
  - Server CPU jobs (Init dry run, freeze, cache verification): 0.006 h.
  - Agent active wall clock: 0.636 h in seven spans across two CLI processes (pid 40800, then 87732 after the
    resume). Waiting during training and the readout (remote-sleep polls) and the gap before the resume are excluded.
  - All of it is in `runs/video_session_matched_discrimination_v1/ledger.jsonl` on the server. A short closing span
    after the result commit is appended there. Codex's feasibility checks, audits and monitoring are outside it.
- **New disk:** 533 MiB (about 0.56 GB) of 3 GB: four final states of 138.6 MB each, update logs and the 2.2 MB
  private cache. No pixel, prefix,
  token or target cache was written, and no resume file remains. Free space is 143 GB.
- **Timeline (UTC, 2026-10-27 / 28).**
  - Dispatch 22:28, configuration 22:35 (b84a78c), code 22:43 (f5a4155).
  - Check attempt 1 failed 22:46, diagnosis 22:50, repair 22:52 (70362d4), attempt 2 22:56, Codex's two blocking
    findings 22:57, gate fix 22:58 (fba4eaf), attempt 3 passed 23:02:04 (committed af26bad).
  - Training 23:02:51–00:41:14 (pid 2551592). The Claude CLI was resumed once (23:06:56, pid 87732) while the single
    training queue continued untouched.
  - Weights committed 00:41:57 (aaab220), freeze 00:42:18, readout 00:42:19–01:12:06 (pid 2579819), verified summary
    01:13.
  - Codex's review file had no blocking finding before training (after the fba4eaf fix), during training or before
    the freeze. No helper agents were used.
- **Records.** `checks.json` (attempt 3), `checks_attempt1.json`, `checks_attempt2.json`, `diagnose_numerics.json`,
  `weights_s0.json` (manifests with fit-only monitors, drift, gradient splits, plan sha256) and `readout_s0.json`.
- **Code.** `scripts/train_session_matched_discrimination.py`, `echo_routing/temporal/session_pairs.py`,
  `scripts/diagnose_session_matched_numerics.py` and `tests/test_session_pairs.py` (14 tests).
- **Server only.** Trained weights, per-update logs, job logs, the private readout cache and all identifiers.
