# Within-View Output Regularization Implementation Plan — A08

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. The owner authorizes the implementation, checks, reviews, six bounded trainings, staged evaluation, fact-check and round-only commits without waiting for their return.

**Goal:** Test whether penalizing changes between neighbouring predictions with the same training label reduces false fragments without suppressing real view changes, beyond ordinary smoothing and the existing plain-CE control.

**Architecture:** Keep the V01 Ordered TargetMViT, RGB preprocessing, marker, CLS classifier and inference interface. Reuse A07's exact training-window manifests and frozen plain-CE F controls. Train two new three-seed continuations from V01: A smooths all neighbouring targets; M smooths only equal-label neighbours. Both use plain CE plus a fixed truncated log-probability penalty. No new parameters or temporal decoder.

**Tech Stack:** Existing PyTorch MViTv2-S, 16-frame 224×224 inputs, pinned A07 manifests/optimizer/state, a pure loss module, existing calibration/evaluation/proof machinery; English aggregate README.

---

## 1. Owner decision and the new question

Close A07 as reported: no qualifying F/B/O objective gain. Do not extend its transcript coefficient, four-target grouping, epochs or seed search. Preserve every earlier closed configuration.

A07's result raises a different testable question. Its continued models recover some reference segments but produce more unmatched predictions, especially selection seed 2; the gap remains with direct argmax and a common decoder. Native recognition floors are met. These are observations, not proof that missing regularization caused the errors or that all errors are short false cuts. M is a hypothesis to test, not an established cure.

**Primary question:** does releasing a fixed smoothness penalty at labelled view changes improve the complete system, compared with applying that penalty everywhere? **Practical comparison:** does M improve over A07 F and the frozen V01 reference, rather than merely undoing harm from blind smoothing?

Read AGENTS.md, CLAUDE.md and `docs/EXPERIMENT_INDEX.md`, then V01, A01, A04, A05, A06-R1 and A07's corrected reports. Distinguish this change:

- A01 put MS-TCN/ASRF, including their smoothing terms, on **frozen video features**. A08 applies an output penalty to the raw-video model and updates its encoder; it adds no temporal head.
- A04 examined time-token occurrences of the same nominal frame across different contexts. A08 uses ordinary **CLS outputs for different neighbouring request times**, with a reference-defined training mask. It does not claim A04 proved runtime output instability or reopen a same-frame consistency proposal.
- A05/A07 tested run balancing and transcript order. A08 removes those terms and tests prediction regularization, holding A07 sampling and optimization fixed. A07 F is reused as an already executed control, not renamed or improved.
- R03/R13 and V02/V03's boundaries, residuals, patch readouts, density and fixed-output searches remain closed. No such addition is authorized.

The truncated log-probability penalty comes from [MS-TCN](https://arxiv.org/abs/1903.01945); the [authors' implementation](https://github.com/yabufarha/ms-tcn/blob/master/model.py) uses coefficient 0.15, truncation 16 and a detached previous output. The reference-label mask and per-group normalization below are this task's explicit adaptation. This is an established regularization idea on a different trainable representation, not an architectural novelty claim or an exact MS-TCN reproduction.

## 2. Immutable sources and reused F control

Repository: `/Users/williamqiu/Desktop/Harvard/Capstone/echo-view-routing`; server: `echo-l40s:/home/william/echo-view-routing`.

Pin before any new output or loss study:

| File | SHA256 |
|---|---|
| `configs/experiments/video_sequence_objective_v1.yaml` | `f8d50e07666dc1cb4ab26a85d4c45749f2241a650be041715c3505fb8698f390` |
| `docs/results/2026-10-15_video_sequence_objective/preparation.json` | `745eef911a8361a9fc285d43c659e74ac69bf79d50651009003ef775d7abed66` |
| `docs/results/2026-10-15_video_sequence_objective/summary.json` | `7c25e1b66652e4a0df99b82ccda2d100111cd14d9fb534edf10f676e342e3079` |
| `docs/results/2026-10-15_video_sequence_objective/choices.json` | `b4aa8fdca51343a8ed08d29e2079030cf28d6333570044222d51da181897dffa` |
| `scripts/train_video_sequence_objective.py` | `1d86871b71f1ea6ba4db894aec1d49a7f0a0f4be9bb633b690df45e77d8ee44d` |
| `scripts/prepare_video_sequence_objective.py` | `508512bd4ddbc649e12901070fccd32de03078aeae91e5e8121cf9b65c0302c0` |
| `scripts/report_video_sequence_objective.py` | `9bcea1749b4abc20d94187d617f84ff6ae69cbb91cdc9e6f4a3902421b4b8a12` |

Read the existing `runs/video_sequence_objective_v1/manifests/VSO_s{0,1,2}.npz` and index, never regenerate, edit or replace them. Their SHA256 values are respectively `0d5653cd807521bcbd7125ff602772e8289c832759ce5071816df24b0ef6c38c`, `d92b5a05fc21c16e3d1e5934074c640afd04b04b8b2e7cbcf5992fffbf83299a`, `2310dd5e9be40864d1f054065334e4d8309518efe50b330b5ef82275c0db2dfd`. Check all epoch/array digests against A07 preparation. Reuse its training allowlist, source pixel digests, class order and the original V01 seed-specific **start** weights pinned in A07's config. Do not start from an A07-trained F/B/O model.

F is A07's already-trained plain-CE continuation, with final states at `models_frozen/video_sequence_objective_v1/frame_s{0,1,2}/final.pt` and SHA256:

- s0 `6fbc987cd15c3bb6af4f8326ca2f7e6ae411d4da02e25bcdd5c93abccb2ce210`
- s1 `958878ed541707e1c0539a555b4026f4c6a8ce2e2a94f155f27d419da14e3361`
- s2 `0333fc15a193fe82a50d82d9f3a974e04d85570bc2e2ab9c95acd0e1cca6c3bd`

Pin their complete identities/manifests from A07 summary and their saved per-window NPZ identities/hashes from server files. Reproduce F's stored probabilities, chosen setting, direct-argmax/common-setting predictions and reported metrics exactly in the appropriate evaluation phase. F is not retrained or retuned. If any source is missing/incompatible, stop dependent work with evidence; do not invent a replacement control or silently recompute different outputs.

No development arrays/labels/media or archived development probability contents are read before the six new runs freeze. File hashes and source aggregate reports can be checked earlier. Training metadata are filtered by A07's allowlist before labels or paths. Keep numeric frame order, whole-stream context, 16 frames, 224×224 and EchoPrime normalization. Same-view joins do not create a label change; no recording/session identity is a model input. Test/Pediatric remain unread.

## 3. Exact arms and equations

Each update is A07's unchanged eight windows, with one native group and one change-containing constructed group. The model forwards once, returning logits z reshaped to `(2,4,5)`. This grouping is for loss computation only. Use A07's exact plain-CE value/gradient path (`segment_objective.objective('frame', z, runs, ones(5))`), called L_CE, without importing run-balanced or transcript terms into the new loss.

For each group n, adjacent t=1,2,3 and class c, define:

`ell = log_softmax(z, dim=-1).to(float64)`

`d[n,t,c] = clamp((ell[n,t,c] - stop_gradient(ell[n,t-1,c]))**2, min=0, max=16)`.

- **A (all-neighbour regularization):** `S_A = mean_n sum_(t,c) d[n,t,c] / (3*5)`.
- **M (equal-label regularization, candidate):** `m[n,t] = 1{label[n,t] == label[n,t-1]}`; `S_M = mean_n [sum_(t,c) m[n,t]*d[n,t,c] / (5*sum_t m[n,t])]`, with the bracket **zero** if that group has no selected edge.
- **Objectives:** `L_A = L_CE + 0.15*S_A`; `L_M = L_CE + 0.15*S_M`. F remains the stored L_CE baseline. No division by 1.15, learned coefficient, sweep, label smoothing or auxiliary term.

Equal group normalization keeps the native group's regularization weight the same in A and M; it also reweights selected edges within a mixed group. Therefore M−A tests **this mask-and-normalization rule**, not a pure position intervention of equal per-edge weight. Count edge masks and zero-edge groups before training; report counts without using them to choose a different rule. Repeat centres remain as A07 and their number is disclosed, not called extra cases.

For zero-edge safety, divide by `5 * mask_count.clamp_min(1)` before zeroing empty groups; never evaluate a zero denominator inside an unselected `torch.where` branch. Such a branch can still create invalid backward values.

The detached previous output is intentional: the regularizer has no gradient through its earlier endpoint; CE still trains every output. Do not replace it with symmetric KL, reverse the direction or penalize a boundary with a negative-distance term. Training labels supply the mask only. At inference use the ordinary single-window CLS output, with **no mask, label, boundary, transcript, group or flag input**.

M equals A on an all-equal-label group. An all-zero mask contributes exact zero and no regularizer gradient. Both A/M regularizers and CE are computed/logged from the same forward in both arms; no extra dropout forward or RNG use. At lambda zero the new training update must match A07 F's update path. No new parameters, gate, teacher, FACT/ASRF or modified decoder.

## 4. Training and budget

- Six **new** runs: all_s0, masked_s0, all_s1, masked_s1, all_s2, masked_s2. Three stored F controls are reused. Each new run strictly loads its seed's V01 Ordered final state, starts a fresh optimizer and makes **3,750 updates**, keeping final weights only.
- Exact A07 manifests, pixel/epoch digests, batch order, two AdamW groups, lr 1e-5 encoder / 1e-4 marker+classifier, weight decay 1e-4, betas .9/.999, eps 1e-8, clip 1, 375-update frozen-encoder warm-up then inherited cosine/unfreeze. Autograd remains through the frozen encoder to the marker. Preserve all original dropout/stochastic depth and fp32/TF32-off settings.
- Reset identical model CPU/CUDA RNG per seed/arm independently of data RNG; save complete resume states. Retain A07's state restoration and disclosed max-pool CUDA backward nondeterminism policy. Do not claim exact rerun weights or extend tolerances from final performance.
- **Cap: 6 GPU h and 3 CPU execution h**, including checks, failed attempts, training, evaluation and fact-check. Before launch, spent + six measured run projections + 1.25 GPU h evaluation/fact-check reserve + .25 GPU h overhead must fit. Measure 50 complete unfrozen RGB updates per new arm after five warm-up updates; include real input loading and all loss/monitor work. At each run use spent + remaining runs at max(measured, observed cost) + remaining reserves, without double counting. No new run starts if the cap cannot fit; report actual evidence instead of shortening, dropping a control, changing precision or seeking approval for routine work.
- No full F retraining, epoch selection, early stopping, seed/lambda/grid/window/sampling changes or automatic extension.

## 5. Implementation tasks

Create only: `configs/experiments/video_within_view_v1.yaml`; `echo_routing/temporal/losses/within_view.py`; `scripts/train_video_within_view.py`; `scripts/report_video_within_view.py`; `tests/test_video_within_view_losses.py`, `tests/test_video_within_view_pipeline.py`; and `docs/results/2026-10-16_video_within_view/`. Private outputs live under `runs/video_within_view_v1/`, weights under `models_frozen/video_within_view_v1/{all,masked}_s{0,1,2}/`. Reuse pinned A07 helpers without editing their files, config, results or identities.

### Task 1 — Sources, config and tests

- [ ] Commit the config, exact equations and decision rules before new masks/counts or output analysis. Write failing meaningful tests, then implement the pure loss and a read-only source/manifest adapter. Reuse the original manifest arrays, not a newly sampled plan; check per-seed digests and pixel validation.
- [ ] Test native and mixed groups, X-Y-X, same-class joins, zero/all-one masks, clipping and finite-gradient behavior. Explicitly test earlier-endpoint detach on supplied independent log-probability tensors; CE must still give the earlier logits gradients. Float64 finite differences apply to smooth, unclipped current-endpoint values with the previous endpoint held fixed, **not** a naive finite-difference test across a stop-gradient operation.
- [ ] Demonstrate lambda=0 value/logit-gradient identity to A07 frame loss on pure and mixed batches, plus identical before-update logits and RNG consumption for A/M. Check that changing labels with fixed RGB/marker never changes eval forward outputs. Pin source function/module digests and all new run identities before launch.

### Task 2 — Engineering checks, review and six trainings

- [ ] Strict source load/zero-update training-window parity ≤1e-4 and equal argmax; lambda-zero single-update and 3+3/six-update comparison with A07 F's original update code and identical initialized states/RNG/optimizer. Use A07's documented tolerances/state checks. This establishes a compatible implemented control path; it does not claim a bitwise full F rerun.
- [ ] Check encoder immutability during warm-up, optimizer binding, unfreeze-transition resume, finite regularizer gradients through encoder/projection/marker/head after unfreezing, and hook cleanup. Save 100-update engineering smoke curves on fixed training inputs per arm; no perfect-fit/token-activation performance gate.
- [ ] Independently review math, detach/clipping, input pairing, control reuse, isolation/barrier, identities and budget. Repair substantive code issues and rerun affected checks. Commit check/review records, then **launch without waiting for the owner** if within scope/budget.
- [ ] Verify actual trainer/queue PID, log and update count before saying training started. Freeze every new final weight/manifest/hash before evaluation; resume only matching complete identities.

### Task 3 — Staged evaluation

- [ ] Inherit A07's tested half-filtered metadata and frame reader, calibration-only first. Never use an eager reader of both halves. Check source V01 calibration parity and reproduce frozen V01 and stored F calibration rows exactly; read only the permitted half of saved outputs. Saved controls stay under their original identities, not new A08 runs.
- [ ] For each of the six new models, use the unchanged calibration B3 grid h=[5,7,9,11,15], q=[.80,.85,.90,.95,.97,.99,.995,.999], NMS=5; require native macro-F1≥.980 and multi recall≥.95; maximize multi F1@50 with original grid-order ties. Infeasible is failed, with 9/.95 diagnostic only. F retains its A07 choices. Save all choices and source/control provenance; dry run on calibration, commit choices and verifiable Git-object proof before selection. Open selection once only. No selection grid or chosen best seed.
- [ ] **Primary M−A** multi segment F1@50, per seed and half. Mandatory **M−F, A−F, M−frozen Ordered, A−frozen Ordered**. Both practical reference comparisons are essential. Report own calibrated B3, direct argmax and the original V01 seed's fixed numerical B3 settings separately; do not choose from sensitivities.
- [ ] Use inherited 2,000-draw paired bootstrap seed20261002, native/session and stream approximations, and strict three-seed same-side exclusion of zero within each half. CIs that touch zero are not consistent differences, and absence of consistency is not equality. Every new comparison is exploratory on repeatedly reused development halves.
- [ ] Report native/pair/multi separately; segment TP/FP/FN, recovered/newly lost reference segments and extra predictions versus A/F/V01; ≤16-sample and A4C/A5C X-Y-X middles; boundary P/R/F1, same-view-join cuts, native switches, native macro-F1/per-class P/R and A4C↔A5C fixed/new counts. Deduplicate by source recordings/sessions, flag <10 sessions and shared seed cases. Show training same-label/different-label neighbour penalties separately. A smaller smoothness loss or fewer label switches alone is not success; fewer false fragments must not be bought by swallowing real segments.

### Task 4 — Fixed readings, fact-check and closure

- [ ] **Full success:** M consistently improves multi F1@50 over A, F and frozen V01 in both halves; satisfies all per-seed native/recall/feasibility floors and original mean targets. This is a development-protocol success only.
- [ ] **Mask-specific signal:** M−A and M−F are consistent positive in at least one half, neither comparison consistent negative in the other, and M has no floor failure. If it fails the comparison with V01 or the mean targets, keep that limitation. Close this fixed recipe regardless.
- [ ] **Smoothing-only signal:** A−F and M−F qualify in the same half, M−A has no consistent advantage there, and neither candidate has consistent opposite-half loss or floor failure. Credit common regularization, not the mask. **Release of harm only:** M−A qualifies but M−F does not; do not call recovery from A's damage a practical gain. Report these nonexclusive readings per half rather than hiding them in a catch-all.
- [ ] **Trade-off:** explicitly flag candidate/half pairs with recovered segments or lower FP accompanied by any floor failure, consistent recall/F1 loss, or more lost true segments than recovered. Raw small counts remain descriptive; a flag does not establish a causal trade-off. **No qualifying gain:** if neither mask-specific nor smoothing-only nor full success qualifies, state it and end. Always report consistent negative results separately.
- [ ] Targets unchanged: three-seed mean multi F1@50≥.944409687184662 selection / ≥.9821421709894332 calibration; every seed/half boundary recall≥.95, native macro-F1≥.973/.980 and feasible calibration decoder. Also list calibration sensitivity target .9838935784747763. Meeting old targets without beating the stronger frozen reference is not a new winner.
- [ ] Independent fact-check recomputes saved-output metrics/CIs/decisions, verifies source/control identities and gradient/mask tests, inspects private-ID exclusion and samples frozen-model inference within budget. Save superseded artifacts for corrections. English README, tables/summary, curves, source/check/review/fact-check records and commands; update A08 pending to executed with actual closed scope/commits. Commit only this round, no push.

## 6. Authority and protected state

This task authorizes all six new trainings and the whole evaluation/report pipeline once engineering checks pass; no new permission question is needed. Implementer choices may resolve routine engineering details but may not change these scientific rules or invent a new efficacy gate. On real source, math, isolation, budget or unresolvable implementation failure, preserve evidence and stop dependent steps. No automatic follow-on task, lambda search or extra models.

Original A07 and all closed records remain immutable. No test or Pediatric access, public relabeling, paper resumption, raw-media/private-ID export, unrelated cleanup or push. Preserve the eight staged theory files and all other dirty work. Use the repaired `.venv` outside iCloud and retain its backup. Source sessions are not verified patients; constructed joins are not real continuous scans; native labels are public clip labels rather than expert frame annotations.

Report actual preparation/check/launch/update/freeze/calibration/selection/fact-check states. This task sheet is not evidence that any A08 code, training or result exists.

## 7. Plan review

2026-10-16 (America/New_York): one separate read-only reviewer inspected the complete plan, seven local source hashes, A07 aggregate manifest/F checkpoint identities, math, control compatibility and evaluation barrier. Verdict: **Approved; no required fixes**. Its safe-denominator reminder is included in section 3. This approval is for the design, not implementation or model efficacy. The lead separately confirmed through read-only SSH that the three F checkpoints, three manifests and their calibration/selection output files exist; no output arrays were opened. Claude must verify full source identities and engineering checks before training.
