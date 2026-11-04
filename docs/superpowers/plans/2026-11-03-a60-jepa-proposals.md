# A60 JEPA proposals with anchored risk prediction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans task-by-task. Preserve one server executor, `training_execution`; independent reviewers never launch server jobs. Steps use checkbox syntax. User has explicitly authorized implementation, repairs, necessary checks, complete admitted experiments, round-only commits and selective public integration without ordinary confirmation. This replaces A59's noautomaticA60 instruction for A60 only. No automatic A61.

**Goal:** Test whether JEPA-originated label proposals recover short segments/changes unavailable to A59, while learned frame/false-short risk prediction and a frozen A08 anchor retain useful recognition.

**Architecture:** Freeze original-stride5 A08+B3 as default. Union raw-A08 disagreement actions with JEPA constant-class run actions; train matched three-output local scorers on complete-axis action effects. At most one action is accepted; compare old-domain, expanded-domain and expanded-with-risk-veto policies using identical weights and an observable majority guard.

**Tech Stack:** Existing Python/NumPy/PyTorch FP32 CPU heads, frozen qualified A08 and EchoJEPA2.1 producers, existing IoU50 matcher/temporal metrics, actual-receipt resource supervisor, JSON/CSV plus standard scientific plots. Existing venvs only.

## 1. Why this round differs

Authoritative history: `docs/EXPERIMENT_INDEX.md` A57/A57-MD/A57-R/A58/A59 and their linked corrected reports. A57 jointly learned temporal heads beat their matched frozen controls but remained below A08, with recognition errors and excess internal fragments. A58's joint structured decoding harmed native recognition; its short stress gains did not establish a stable complete-task improvement. A59 retained the anchor but could only restore raw-A08 labels. Main guarded oracle upper gains were only 0.215/0.195 percentage points; A59 fresh calibration union recovery was 0/78 even without the guard, within its declared raw-supported single-action domain. Its small short-FN improvements came with more false short segments. These are completed negative results, not invalidated runs or a rejection of all JEPA models.

A60 changes **which labels/actions are available** and **what action harm is directly supervised**, rather than adding another seed/threshold to A59. Hypotheses: (H1) expanded JEPA proposals increase recoverable true short/change coverage; (H2) JEPA probability features help rank the expanded actions beyond an A08 feature control; (H3) explicit frame-error and false-short predictions reduce false repairs while retaining some expanded-domain benefit. All may fail. Routing itself was already studied in A14–17 and is not claimed as algorithmic novelty. A paper contribution requires stable matched capability gains plus independent evidence still absent here.

The risk veto below is an empirical predictor, **not** finite-sample/conformal risk control. [Learn then Test](https://arxiv.org/abs/2110.01052v5) supplies a distinct calibrated hypothesis-testing framework; [Conformal Risk Control](https://arxiv.org/abs/2208.02814v4) controls expected monotone loss. Neither guarantee is obtained by predicting harms and comparing to zero. Reused OPEN halves, clustered source reuse and clip-label propagation remain limitations; no new calibration guarantee is asserted.

## 2. Scope, roles and identities

Only worktree `/Users/williamqiu/.config/superpowers/worktrees/echo-view-routing/a60-jepa-proposals`, branch `a60-jepa-proposals`, parent `9a4e4be3d22db884fede1e31e19127d72618921a`. Birth epoch is the actual filesystem birth in `docs/results/2026-11-03_jepa_proposals/workspace_birth_and_preservation.json`; it does not replace the original phase origin/deadline. Server `ssh echo-l40s`, project `/home/william/echo-view-routing`, new namespace `runs/a60_jepa_proposals_v1`. Older namespaces and their source/config/cache files are read-only. Never import an old CLI whose module-level paths write A59; reuse qualified pure library functions only. Model identity and source-cache identity use separate filenames.

Producer A08-s0: `models_frozen/video_within_view_v1/all_s0/final.pt`, SHA256 `9b144925f32fac7d725d13231a132abeb6635fa44b6fe0dbcec7f5b59d035bbc`, all 400 keys strict/frozen. Preserve original 16 frames/10Hz/marker/gamma/256 resize/224 center crop, FP32/TF32off/eval and original stride5 centres. Raw five-class argmax is R_A. B is original B3 (half_window7, threshold .19660949291632382, nms_radius5, quantile .999), unchanged.

Producer JEPA: `runs/a57_dense_sequence_v1/seed2026110257/J-high/step6000.pt`, SHA256 `2fa26f2b3ffbffd4d8481c6ac1ff2190572f18648e13f13dbf37333a17bdd985`, all 214 keys strict/frozen; original 64-position/stride32/positive-triangle stitch, BF16 encoder/FP32 head. R_J is full-axis argmax of these five-class probabilities. Preserve qualified official2.1 source layout and preprocessing. No retraining either producer, deleting keys, interpolating dense2 as original5, or new source search by performance.

Reuse exact A59 FIT2736 native/402sessions +1500 mixed and inner628 native/100sessions +300 mixed, complete source-footprint separation. Qualify A59 original5 A08 and JEPA caches by complete original-axis lengths, full source refs/gamma/order/role/manifest/checkpoint/SHA bindings. A59 FIT/inner all-context cache already exists; do not redo forward simply because this round has new heads. FIT normalizers/labels/targets never read inner/outer. A08 historically saw inner in earlier work, so no full-model OOF claim. No external test/sealedtest/Pediatric/Unity training/21 quarantined patients. IDs/raw/weights/per-case arrays stay server-side. No human labels available; do not invent them or turn that into an approval blocker.

## 3. Frozen inference action grammar

Use only full unpadded original-axis B, p_A and p_J. No native/kind/source-ID/true join/GT/teacher or decoded metric inputs to inference.

1. **Old actions:** exactly A59 maximal contiguous components of R_A != B, replacing that component by its R_A labels. They may contain several raw class runs.
2. **JEPA actions:** for every maximal constant-class run [s,e) of R_J with class c, replace B[s:e] by constant c. Skip a no-change action. No confidence/length threshold, extension, merge or GT boundary snap.
3. For either action, remove only unchanged prefix and suffix from its represented interval (the edited complete label axis is unchanged). Keep any unchanged interior positions. Canonical key is `(s,e,tuple(replacement_labels))`; deduplicate exact same edited actions, retaining a set of old/JEPA provenance flags solely for domain ablation/reporting, not a feature. Sort by that key. Each candidate must change at least one position. Report newly available classes/axes beyond the old domain.
4. Compute edited full-axis occupancy-majority using `np.bincount(...,minlength=5).argmax()` with fixed class-order ties. G-eligible means this equals B's majority. Apply the same observable rule on **every** sequence, before training/scoring. Store rejected candidates for domain diagnostics. No new hard length/cut limit. No native-only switch.
5. All policies execute at most one G-eligible action. Among admitted actions with predicted utility strictly >0, pick highest utility, then canonical key; otherwise return B. Do not combine actions or retry/rescore after applying one.

G mechanically preserves the primary native majority-class prediction and its confusion/macroF1, including B's mistakes. It does **not** preserve all frames, cut count, short FP or clinical safety. Whole-axis/long replacements and native harm must be measured. One action is not one boundary change or an edit-distance guarantee.

## 4. Features, targets and matched training

Common 29 features: inherit A59's region log lengths3, B-in-action class histogram5, **proposed replacement** class histogram5, immediate B left/right onehots10, availability2, observable change in full-axis predicted segment count1 =26; append `log1p(number_changed_positions)`, changed/action length and changed/full length =3. All are available without truth. Region flanks remain16 original positions.

Each source block has A59's 36 region summaries plus four features over changed positions: mean/min/max of `p[t,new[t]]-p[t,B[t]]` and mean of `log(max(p[t,new[t]],1e-8))-log(max(p[t,B[t]],1e-8))`. Thus 40 per block. A control is 29+A40+duplicate A40; J is 29+A40+J40; **109 inputs**. Both already require JEPA to generate the expanded candidate geometry/classes. J−A isolates added posterior features conditional on the same JEPA proposal set; A is not operationally an A08-only system. Per-arm FIT-only population mean/std floor1e-6, no inner/outer adaptation.

For each G-eligible action compute complete-axis truth-assisted targets on the server:

- `u = 2*(M_after-M_before) - F_fit*(P_after-P_before)` using full five-class IoU50 class-aware one-to-one counts. `F_fit=2*M_base/(P_base+T_base)` pooled over the unchanged1500 FIT multi bank only. Recompute and bind; expected A59 value is .8917446 but never silently force it if cache identity differs. This is the exact single-action marginal F1 numerator at that FIT operating point for multi; native uses the same count proxy, not a native macroF1 surrogate.
- `dE = count(edited != truth)-count(B != truth)` in original positions.
- `dFP16 = spurious_predicted_short_segments(edited,truth,<=16)-same(B,truth)` using the identical full matcher, not an isolated interval count or unsigned loss.

Target scales are `max(sqrt(mean(target_k**2)),1)` across all G-eligible FIT candidates, one shared vector for A/J, with **no centering**. Divide targets by these scales, equal mean MSE over three outputs. Inverse-scale before decoding/reporting, preserving zero/sign. No coefficient/scale-floor/threshold search. MLP109→32 GELU→3, **3619 parameters**, output layer zero initialized, same initial full weights per seed across arms. Four models: A/J × seeds2026110360 and2026110361, each6000 steps. AdamW lr1e-3, betas(.9,.999), eps1e-8, wd.01; warm100/cos6000, FP32 clip1. Batch64 candidate draws =32native+32constructed. Reuse A59's declared empty-kind fallback only if one eligible pool is empty; if both are empty, report structural impossibility, do not invent data. Shared deterministic plan per seed, arm-independent canonical candidate indices and identical exposure. FIT candidate rows are never duplicated under a new statistical identity.

latest every100 includes full model/optimizer/moments/RNG/normalizer/scales/plan/cursor/actual update counters, durable atomic file+directory fsync. Weight-only2000/4000/6000. At these steps run all inner G policies for diagnosis only (12 checkpoint records ×3policy metrics=36 sets), no selection/early stop. Fixed6000 is the sole primary checkpoint. Formal24000 and all engineering updates are separate physical counters. Freeze code/input/config/data identities before actual model work, weights+inner+fixed endpoints in actual Git before outer outputs.

## 5. Same-weight inference controls and full endpoints

Each model decodes:

| Policy | Domain | Additional admission |
|---|---|---|
| O | G-eligible old actions | predicted u>0 |
| E | G-eligible old∪JEPA actions | predicted u>0 |
| R | same expanded G domain | predicted u>0 **and** predicted dE<=0 **and** predicted dFP16<=0 |

All use the same three-output model weights; R ranks eligible actions by u, not a new coefficient. Zero thresholds are fixed by the semantics of signed harm. O/E compares inference availability conditional on training on the expanded domain, not an isolated candidate-training causal effect or A59 reproduction. R−E measures the learned veto's actual recovery/harm tradeoff; predicting zero harm is not guaranteeing it.

Four outer tables, each with selection and calibration halves:

1. Original main native/multi/pair task and labels10..40.
2. A58 duration-stress (A59 old_stress), retrospective reused feedback.
3. A59 confirmation, now **retrospective**, not new held-out confirmation.
4. New A60 construction confirmation:120multi streams per half, recipe frozen before the first head result. Use existing outer native sources with A59 recipe (4..8fragments, uniform classes, same-class probability.35, gamma .90 probability.5, lengths4..min(80,n), no repeated cine within stream), new domain-separated seeds `[2026110360,60,2]`/`[2026110360,60,3]`. Freeze complete recipes/source footprints/labels and disjoint halves. Actual across-join forward for both producers, never concatenate native logits. Construction requires authorized class metadata and creates labels; disclose this construction-only access. No head target/metric/selection access to these labels before weight freeze.

12 policy rows ×2halves ×4tables =96 model metric rows. References: main five/half =10 (three original A08 seeds+B3, raw s0, rawJEPA); other tables three/half =6 each (A08s0+B3/raws0/rawJEPA). Total **124 metric rows**, not70/72. All references/source identities qualified, no reference head retraining.

Predeclare **144** paired5000-bootstrap comparisons: nine per seed/half/table =J−A for eachO/E/R (3); E−O withinA andJ (2); R−E withinA andJ (2); R−Base withinA andJ (2). Use seed2026110360; whole source-session units for native, whole streams for multi and whole pairs for pair; preserve undefined-ratio counts/null, shared-source conditional/exploratory limits. Do not count constructed streams as independent patients. No selecting winning seed/half/table.

All metrics use complete five-class axes: native occupancy-majority/class-order ties/confusion/macro/perclass/A4C↔A5C; native frame errors/accuracy and internal cuts; multi/pair segmentF1@50 counts and boundaryP/R; true-short<=10/<=16 FN denominators and spurious-short FP; same-class join cuts; recovery **and** harm vsBase. Report candidate/G-vetoed/risk-vetoed/accepted counts, score/target errors and missed-beneficial vsaccepted-harmful candidates; newly JEPA-supported recoveries; proposal and accepted lengths, changed counts, interval/full fractions, whole-axis incidence and added cuts **for every logical policy**, no late omission. Retain all training curves/exposure and complete inference resources including producers, not just tiny head timing.

Only after all124 system metric rows/predictions are frozen, compute **32 exact one-action oracle cells** =old/expanded ×U/G ×4tables ×2halves. Include no-op; use the inherited exact fractional pooled-IoU50 procedure, with complete-action independent counts. Raw old-domain oracle must reproduce A59 where input/table identical. Report absolute/count ceilings, marginal/union recoverability of Base misses and G loss; union is not jointly attainable. Oracle uses labels, is not a deployable model, and cannot change A60 candidates/thresholds/models. Ground-truth dE/dFP16 at vetoed candidates are diagnosis, never an extra inference oracle. Do not assert the one-action domain exhausts all useful JEPA information.

## 6. Budget, admission and stopping

Not a budget reset. A59 final `resources_final.json` SHA `e04d513c2c67ef62abc3caccab50e2c3fa3165b9e4528facc56cb9e87399b8eb` carries GPU18.3349858850903h, measuredCPU18.253671176388888h, CPUaccounting19.003671176388888h including original separately described .75h allowances, tracked-path conservative upper34541577626B. Inherited hard caps:24GPU-jobwall h,72workerCPU-core h,40,000,000,000B, original origin1793648044.1596537 and **earlier deadline1793820844.1596537**. Starting remainder5.6650141149097GPUh,52.99632882361111CPUaccountingh,5,458,422,374B. Keep2GPUh unspent closing reserve within that quota. No new shorter arbitrary cap; use these remaining shared limits.

Only existing machine/services; no purchase/contact. Actual receipt measurement includes preparation, failures, local/new independent tests, caches, checks, four full heads, inner, all124rows/144CIs/32oracle, checkpoints, validation, plots, integration and closure. Coordination wall/unmetered tool overhead stays separately disclosed; no microbudget masquerading as workerCPU. Existing unknown orphan OS/CPUtail/overlap limitations persist. Forecast/reserve are not actual spending.

Before any server launch inspect live PID/startticks/queue lock/claims/GPU/apps. Same queue, launch-intent durable before Popen, reload SHA-bound actual spent ledger under lock, ENOENT-safe size scan, failclosed cleanup with parent-wait OS status. Never infer exit0 from a complete manifest. A CPU stage sets `CUDA_VISIBLE_DEVICES=''` **before importing torch**, verifies all model/optimizer tensorsCPU and no CUDA context; unexpected context occupies/charges its full wall exactly once. Preserve actual failed receipts.

Full measured admission covers already-spent + all remaining work +atomic/checkpoint/cache peaks +2GPUh reserve, bounded by every shared cap/deadline. Qualify existing caches by identity, not repeat old source qualification/forwards. New local graph checks must verify proposal semantics/alignment, all intended model gradients, finite actual64candidate updates, exactAdam recipe, deterministic repeat/disk-resume/pending refusal, actual durableIO throughput. Actual checker small panel degeneracy must be diagnosed, not removed/hidden or called formal failure. Forecast CPU/head/metric cost from real fullFIT candidate volume and actual durable updates, include worst sequence/candidate lengths and full output coverage. Whole matrix is admitted, not a pilot-only forecast.

Proceed through all admitted arms and confirmations regardless preliminary performance. No new LR/B3/threshold/coefficient/arm/search, no inner checkpoint selection, no cache rerun by model score. Stop for actual cap/deadline, verified incompatible/source unavailability or irrecoverable engineering/external block; finish honest partial accounting/exit/artifact closure. Routine correctable implementation defects are already authorized. All accepted reporting/auditing/sync must finish before deleting only A60's monitor; no automaticA61.

## 7. File map and bounded implementation tasks

Reuse frozen pure `anchor_features` region summaries, `anchor_edits` label/count/majority helpers, `anchor_metrics`, `anchor_cache`, source binding, exact oracle arithmetic and qualified producer functions. Do not change their old APIs or old manifests. New files under `echo_routing/temporal/proposal_{actions,features,training,data,resources,metrics,readout_plan,oracle}.py` separate pure responsibilities. New stage CLIs under `scripts/{prepare,check,admit,train,audit,cache,readout,aggregate,oracle,close}_jepa_proposals.py` and `scripts/supervise_jepa_proposals.py`; add narrowly necessary helpers rather than one monolith or wrappers with hardcoded old namespace. Test files `tests/test_proposal_{actions,features,training,data,resources,readout_plan,oracle}.py`. Public folder `docs/results/2026-11-03_jepa_proposals/` contains replan, running README, all config/freeze/admission/audit/receipt/summary tables/standard plots. Raw rows remain private server namespace.

### Task 1: Snapshot and source/cache reuse qualification

- [ ] Read-only live server PID/lock/GPU/A59-closed receipt and inventory; bind source/cache role/ref/axis/gamma/labels manifests, exact prior terminal SHA, no old writes.
- [ ] Freeze new confirmation recipes before any head outcome; source construction-only class access explicitly recorded. Write complete endpoint124/144/32 machine manifest.
- [ ] Commit only A60 card/config/public preparation files. Root obtains independent whole-plan review, fixes factual/spec gaps before modeling; no approval pause after completion.

### Task 2: Proposal grammar and feature/target semantics

- [ ] Write failing synthetic tests in action/feature files: a JEPA class absent from RA creates a nonempty new action; canonical no-op trim/dedup; old inclusion; arbitrary long and1position runs; ties; G preserves majority but can harm frames/cuts; no forbidden inference argument; signed full-sequence FP/frame targets.
- [ ] Run the new focused tests with existing venv and capture measured SELF+waitedCHILDREN CPU plus RED exit, not model evidence.
- [ ] Implement exact grammar/features/109 dimensions/scales/complete matcher targets as sections3–4, no hidden filter. Run GREEN, verify common/A blocks equal across arms, commit only new files.

### Task 3: Head lifecycle and role-pure preparation

- [ ] Write RED checks for3619params/three outputs/zero no-op/exact optimizer/defaults and shared init, FIT-only scaling, both canonical seed plans with no alias-key folding, actual counters/Adam/RNG resume and corrupted cursor/partial pending refusal, atomic file+directory durability.
- [ ] Implement data/plan/trainer/inner diagnostics with actual updates/IO receipts. Repeat and2+2disk-resume on representative nondegenerate cachedFIT rows; gradients finite/non-None and each active tensor liveness explicit. Test all intended parameters, not mere Adam decay. Fixtures never enter formal data.
- [ ] Focused GREEN tests and independent spec then quality review of only new graph; preserve original failures/repair costs. No repetition of inherited GPU/library checks.

### Task 4: Full measured forecast and actual formal admission

- [ ] Implement new namespace supervisor/resource inheritance from A59 final totals (not its older A58 baseline), exact-once newlocal receipts and locked spend reload, launch/pending/exittruth tests using cheap subprocesses.
- [ ] Meter actual fullFIT preparation, all action distributions/G coverage/Ffit/normalizer/targets and real batch64+durableIO head throughput; confirm noCUDAcontext. No outer-label utility/metric access.
- [ ] Freeze actual source/data/code Git objects and SHA-bound measured allmatrix forecast. Once all caps fit, directly start four6000head jobs in the single CPU queue; record real PID/starttick and increasing Adam steps. Engineer updates separate from formal24000.

### Task 5: Complete training audit and freeze before outer

- [ ] Finish all four fixed6000 and12innerdiagnostics/36policymetricsets. Independently recount actual trace/optimizer/RNG/exposure/common init and inner stored-label metrics, bind all milestone weights/normalizers/scales/plans.
- [ ] Commit complete weights/inner/model manifest and124/144/32 endpoint plan **before** new outer outputs. Check root actual Git object and serverSHA identity once for this new stage, no broad repeated library audit.

### Task 6: Fixed full readout, paired intervals and oracle

- [ ] Qualify reused main/stress/A59confirmation sources; admit and perform only missing A60confirmation true across-join source forward. Bind full input manifest/journal/no missing/duplicate/changedSHA axes tests. No source/model identity filename collision.
- [ ] Execute all124metricrows (including length/fraction/cuts/harm descriptors), freeze readouts, independent stored-private-label recount; run144×5000pairedCI with predetermined units/seed/null accounting.
- [ ] Then32oraclecells, verify old-domain reproduction and exact fractional synthetic exhaustive/degenerate tests. Preserve all negative/undefined outputs. Ground-truthoracle never selects model/candidate threshold.

### Task 7: Scientific closure, plots, accounting and selective integration

- [ ] Export standard standalone plots for training/matchedsegment and shortFN-vsFP/frameharm/proposalcoverage; inspect rendering. Full report question→attempt→results→limits→next evidence, separate proposal coverage, learned ranking and learned risk-veto efficacy. Main/fresh/retrospective tables separate; no unproved anatomy or independent clinical/patient claims.
- [ ] Independent final124/144/32 scope/denominator/source/label/restoreharm/length/resource/exit audit. Model/optimizer files remain server; public exact allowlist excludesIDs/percase/raw/weights. Actual costs include all failures/tests/checks and legacy uncertainty; final worker/supervisor absence/GPU release/empty claims/available lock directly verified.
- [ ] Selectively sync public exact manifest + authorized source/test files to main, append index only preserving user's prefix. Protect all unrelated edits and8stagedtheory diffSHA `a9cab9907c011551149b26d9331f86d0f99776eff29f022362849b64effd5ff8`; private before-inventory in `/tmp/a60-main-preservation-before.json`, no whole-index replacement or broad staging. No webpage/push/external contact.
- [ ] Commit only A60 files/records, verify copied SHA/count and existing-change guard, then delete only capstone-a60 monitor after every accepted stage completed or hard-limit/external honest termination. Report remaining independent annotation/clinical evidence gap. No automaticA61.
