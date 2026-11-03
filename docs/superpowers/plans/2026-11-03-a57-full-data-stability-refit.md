# A57 full-data refit and within-view stability Implementation Plan

> **For agentic workers:** Use `executing-plans` inline under the existing sole `training_execution` owner. Keep one GPU queue and the existing A57 workspace. A focused read-only reviewer checks this card; no new permission is needed for authorized preparation, checks, execution, repairs or closure. Never modify an active frozen producer/reader dependency.

**Goal:** Test whether greater authorized task-training coverage improves dense EchoJEPA2.1 recognition, and whether a matched training-time stability term reduces excess short false segments while retaining real short segments.

**Architecture:** Freshly initialize the admitted final-layer EchoJEPA2.1/768D + original two-stage 128-channel TCN for two matched full-data models. R-CE uses the original two-stage dense CE; R-ST uses the same CE plus a fixed, label-protected temporal stability penalty. Full five-class prediction, 64-position geometry and raw inference are unchanged.

**Tech Stack:** Existing pinned PyTorch/CUDA/BF16 encoder, FP32 loss, NumPy private manifests/caches, existing source loader, durable checkpoints, complete-axis evaluator and paired analysis.

## Authorization, predecessors and interpretation

The user explicitly requested diagnosis and targeted new training on 2026-11-03, after seeing the first outer A57 results. This is an **adaptive exploratory stage informed by reused outer-development feedback**; it is not selection-independent confirmation. Keep original A57 and the prospectively frozen four-depth MD card/config/results intact. No outer labels enter these new training pools. Test/sealedtest, Pediatric, Unity training and the21quarantined source patients remain excluded.

- A57 `381946a` froze32,000 actual updates and17outer logical rows. J−F establishes adaptation over that same weak frozen representation, not superiority over A08. The original core reader completes all its declared rows and audits unchanged.
- A57 first-seed outer selection has4/11A5C clips calledA4C; A08-s0 has0/11. Raw J short≤16FN5/141 versus7/141 forA08 coexists with shortFP39 versus9. These are distinct recognition and fragmentation deficits; stability alone need not correct a persistently wrong clip.
- A57 task FIT has2736clips/402sessions; the628/100 inner sources were excluded from its updates. Actual A08 V01 initialization manifests expose3330–3338distinct protocoltrain sources; later continuation manifests expose3018–3044, not every allowlisted3364clip in every run. Hash/publish the worker's verified aggregate source-exposure record. Coverage is a plausible contributor, not an established cause or an equalization of pretraining histories.
- A08 `docs/results/2026-10-16_video_within_view/README.md` already tested all/equal-label MS-TCN smoothing on the old window-CLS MViT. `docs/results/2026-10-17_post_a08_loss_diagnostic/README.md` found substantial early penalty-gradient dominance. This established loss is not new methodology. The changed condition is the dense JEPA/TCN graph, tubelet geometry and final refit coverage. R-ST−R-CE isolates the added regularizer; R-CE−old J is a refit package including source coverage, bank and sampling changes.

## 1. Freeze cache-only diagnosis

Files: create `scripts/diagnose_dense_bottlenecks.py` and public aggregate evidence under `docs/results/2026-11-02_dense_target_sequence/bottleneck_diagnosis/`; retain private case-level output on server only.

- [ ] Recount both seeds' fixed6000 raw/inner-B3 and original A08-s0 on both complete outer halves: five-class confusions/per-class F1/recall, native paired correct/wrong four cells, multi segment/matched/FP/FN, boundary P/R, same-class joins and short≤16 plus≤10 denominators.
- [ ] From existing probabilities/labels only, separate label-stable interior errors/false switches from true-change neighborhoods (fixed±5original positions; report interior and near-boundary denominators). Native whole-clip misclassification and within-clip flicker remain separate. Report margins/entropy only where cached probabilities support them; do not invent new forwards or anatomical explanations.
- [ ] Record FIT/inner/outer class and session counts, actual per-class training coverage, fixed training/inner curves and selected-checkpoint versus fixed6000 distinction. Training loss alone cannot prove generalization or diagnose overfitting.
- [ ] Record actual diagnostic PID/CPU/wall/exit and zero new neural forwards. No per-case IDs in public artifacts. The diagnosis motivates this fixed test, not another threshold search.

## 2. New full-data input contract

Create `scripts/prepare_dense_refit.py`, `configs/experiments/dense_target_refit_v1.yaml`, and private inputs in `runs/a57_dense_sequence_v1/full_refit_v1`; public identities go in `docs/results/2026-11-02_dense_target_sequence/full_refit/`.

- [ ] Verify the native union is exactly the authorized3364protocoltrain clips/502sessions, disjoint from outer/protected sources. The former628inner sources now become **training data**: never label their subsequent readouts validation, use them for checkpoint selection, or claim out-of-fold evidence.
- [ ] Rebuild a1500stream constructed bank from the full authorized training pool using the unchanged construction distribution, recipe seed2026110259. Verify all64-position/padding/overlap footprints. Preserve class/session/source counts and canonical metadata/plan SHA before model forwards. Existing raw frames remain shared, without duplicating a frame cache.
- [ ] Fix model/head/sampling seed2026110257 for this one matched pair; build6000steps with the original five-class native round-robin shuffled cycles and alternating uniform/change-centered constructed starts, one native+one constructed chunk per step. The two arms read byte-identical plans/inputs/labels/masks. Preserve complete class coverage and distinguish repeated exposures from distinct sources.
- [ ] Source: original admitted target_encoder SHA`deb0cd98ea40ef59d21eb883e1deb517a9bcca614047e710d9fad136c750348c`, strict158keys. Fresh source initialization, no trained A57/A08 teacher or warm start. Same149active video tensors and9frozen image/unused-norm tensors as final-layer A57, same56head tensors and initial head state.

## 3. Exact objective and focused checks

Create `echo_routing/temporal/dense_refit.py`, `tests/test_dense_refit.py` and `scripts/check_dense_refit.py`. Reuse immutable original geometry/CE; do not edit `dense_sequence.py` or running readers.

For logits of either stage `z[B,5,32]`, set FP32 `ell=log_softmax(z,dim=1)`. For edge j→j+1, E[b,j] is true only if original positions2j,2j+1,2j+2,2j+3 are **all valid and all have the same training label**. Thus mixed-label tubelets, real changes and padding are excluded. Define:

`d[b,c,j]=clamp((ell[b,c,j+1]-ell[b,c,j].detach())**2,max=16)`

`S_stage=mean_b(sum_(c,j)(E[b,j]*d[b,c,j]) / (5*max(1,sum_j E[b,j])))`

An edge-free chunk contributes exactly zero; divide safely before masking. `S=mean_stage(S_stage)` for both stages. `R-CE=dense_ce(stages,labels,mask)`; `R-ST=dense_ce(...)+0.15*S`. The coefficient0.15, squared-difference cap16, earlier-endpoint detach and both-stage/per-chunk normalization are fixed here, without a sweep. Coefficient0 uses the original CE path exactly. Labels select loss terms only, never model/inference inputs.

- [ ] Observe focused RED tests before implementing: identical outputs give zero; only later endpoint has penalty gradient; all4original labels/valid positions gate each edge; a cross-label/mixed/padded edge has zero penalty; edge-free chunks remain finite/zero; independent formula and coefficient0 CE/logit gradients match; both stages and chunk normalization match the formula.
- [ ] Bounded real FIT check after source/code/data/config freeze: actual2×64 batch, initial R-CE/R-ST logits equal; lambda0 optimizer step equals parent CE; original149active video+56head gradients/moments finite and9inactive tensors unchanged/unbound; same4-step repeat and disk2+2 resume within existing tolerances. Do not repeat the entire predecessor source qualification.
- [ ] During this bounded check measure CE versus0.15S scalar and logit-gradient norms/cosine from the same forward, plus clamp/eligible-edge counts. This discloses scale/conflict, not an automatic coefficient-adjustment rule. Failures/ambiguous updates retain their actual charges; never silently retry them.
- [ ] Time20different real FIT updates for the new objective and one atomic full optimizer write. Include checks in a complete actual cost forecast before formal launch. Existing unchanged loader/geometry checks are reused.

## 4. Fixed training and shared admission

Create `scripts/train_dense_refit.py` and `scripts/supervise_dense_refit.py`; use the original A57 queue lock/global process ledger. Update progress with real PID and advancing committed optimizer steps.

Both arms run6000updates regardless of their observed development scores, with the original headlr1e-3, bodylr3e-5, AdamW(.9,.999)/eps1e-8/wd.05, clip1, warm200/full6000cosine, BF16 encoder/FP32head-loss, bodydropout0/head-dropout.1, identical inputs and common initial head. No augmentation, architecture, class-weight, schedule or learning-rate change is bundled into this test.

- [ ] Before the first formal update, freeze a complete forecast including all completed/remaining core rows, core audits/aggregation, the prospectively preserved MD obligations, both new6000 arms, checks/cold starts/checkpoint writes, new fixed6000 outer inference/cache/paired analyses and≥2GPUh closure reserve. Caps remain shared24GPU-job-wallh/72actualCPU-coreh/40GBadded peak/48hwall; origin1793648044.1596537 and deadline1793820844.1596537. Include CPU prep/failures, not coordinator wall as worker CPU.
- [ ] Preserve sufficient model storage by retaining latest full optimizer/RNG state every100steps and fixed2000/6000weights only. There is **no every500 full inner evaluation or checkpoint selection** for these full-data models. Preserve all physical updates/pending ambiguity/replay accounting and immutable milestone identities.
- [ ] If the complete pair plus prior obligations cannot fit, preserve the prepared card and report that budget constraint; do not silently shorten6000, drop the CE control, choose a tier from performance or extend caps. Routine resource-safe ordering of fully authorized stages is allowed; original core full evaluation stays first.
- [ ] Log per-step CE/S/weighted-S, eligible/clamped edge counts, head/body gradient norms and actual exposure; these replace unavailable validation curves, not generalization evidence. Finite healthy runs proceed without repeated approvals.

## 5. Frozen final evaluation and continuation

Create `scripts/report_dense_refit.py`; reuse independently reviewed complete-axis readers/metrics/paired bootstrap and existing frozen outer input identity. Use unique feature/config/data/model identities in cache keys and clean atomic sequence-prefix resume; pending ambiguity stops for audit.

- [ ] Audit both6000models/optimizers/physical logs/source/config/input identities before outer inference. Reproduce the149active/9inactive/56head role contract and fixed schedule; pin exact158source keys, expected head shapes and optimizer binding.
- [ ] Evaluate **fixed6000 only**, both reused outer halves. Primary raw comparison is R-ST−R-CE; both are also compared to old fixed6000 J and fixed A08-s0 (show all3A08seeds' range). The only B3 supplement transfers the unchanged pre-stage parent seed2026110257 J-high6000 inner-B3 parameters verbatim, hash pinned before new outputs; label it transferred, not recalibrated. No fit/choice on former inner data or outer predictions.
- [ ] Publish the full metric vector and paired differences/conditional intervals: five-class macro-F1 and confusions (especially both A4C/A5C directions), segmentF1@50, boundary P/R, short≤16/≤10 true/predicted/missed/spurious counts and recovered/lost segments. Use existing2000 paired stream/session resamples, seed2026110257; session is not patient, fixed constructed banks share sources.
- [ ] Reduction in false fragments is useful only alongside true-boundary/short-segment retention and recognition. A stable wrong A5C clip remains a recognition failure. Keep every negative result; no automatic smoothing-strength escalation or source133 gate.
- [ ] Update the experiment index and public result report with adaptive timing, tested causal contrasts, limitations, actual shared resource/process exits and residual independent-validation gap. Sync public aggregates to main while preserving8staged theory files/unrelated edits; no website change/push/external message. Continue already-authorized preserved MD according to measured budget, then close the full bounded phase and delete only its monitor when all admitted work/closure is done.
