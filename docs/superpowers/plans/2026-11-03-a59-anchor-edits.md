# A59 Anchor-Preserving Local Edits Implementation Plan

> **For agentic workers:** Use the existing `training_execution` as the sole implementer/server executor, and one fresh read-only plan reviewer. Follow test-driven-development and executing-plans in this isolated worktree; do not spawn another execution queue. Routine repairs, checks, measured admission, training and frozen evaluation are already authorized by the user's 2026-11-03 A59 instruction, without another confirmation.

**Goal:** Preserve the frozen A08 baseline's stable recording-level recognition while testing whether JEPA evidence can identify useful raw-A08-supported repairs of short segments and true changes.

**Architecture:** Freeze original-stride5 A08-s0+B3. Build deterministic disagreement components between its raw and B3 labels. Train matched A08-only / A08+JEPA small utility regressors; each predicts the value of one local replacement, with at most one replacement per complete sequence. Decode the same weights with and without a baseline-majority protection constraint. JEPA does not supply replacement classes. The constraint preserves the exact occupancy-majority class for every input sequence, without knowing whether it is native, constructed or correct.

**Tech Stack:** Existing Python/PyTorch/NumPy and qualified source/metric libraries. CPU regressors, existing L40S only for missing frozen-producer caches. No new paid service, website change, push or contact with others.

## 1. Authorization and inherited limits

The user explicitly asks to replan A59 from A57/A58, focus on retaining A08 recognition and JEPA short/change increments, and continue within remaining budget without waiting. This supersedes A58's no-automatic-A59 restriction for this named phase only. A57/A58 models/results remain immutable. No automatic A60 or architecture/coefficient search follows A59.

- Worktree `/Users/williamqiu/.config/superpowers/worktrees/echo-view-routing/a59-anchor-edits`, branch `a59-anchor-edits`, parent `c1356cbb362adbe176bba8ed211ac708322723a8`. Origin is its actual birth epoch in the public guard, not a budget reset.
- Server `ssh echo-l40s`, project `/home/william/echo-view-routing`; new namespace `runs/a59_anchor_edits_v1`. Only `training_execution` mutates server work. Root/reviewer may perform read-only checks. Never write A57/A58 run artifacts or their frozen dependencies.
- Inherit cumulative **24 GPU-job wall h / 72 measured worker CPU-core h / 40GB tracked-path upper** and original deadline **1793820844.1596537**, origin1793648044.1596537. A58 closed GPU17.31542736033599h; measured CPU receipt sum17.2137378675h, with separate completed allowances bringing conservative CPU accounting to17.9637378675h; tracked upper33624492578B. Retain orphan OS/CPU-tail and historical overlap limitations.
- Thus before A59 costs: GPU remaining6.68457263966401h, CPU accounting remaining54.0362621325h, tracked upper remaining6375507422B. Keep **2 GPU h of closing reserve** within this remainder. Forecast all new preparation/failures/caches/checks/four trained heads/inner/main/old-stress/new-stress/readouts/bootstrap/checkpoints/audits/closure before formal admission. Coordination/unmetered metadata bookkeeping is separate, never called worker CPU. Actual receipts replace forecasts; unused reserve is not spent cost.
- Current connection observation: first live check was idle/no GPU apps; later minimal SSH requests timed out before authentication. Continue local implementation while resolving connection, but do not claim training or current server queue state from the earlier snapshot. Formal execution needs real PID/advancing optimizer counters. No new server purchase or automatic restart is authorized by a timeout.
- Preserve main HEAD, eight staged theory files (binary diff SHAa9cab9907c011551149b26d9331f86d0f99776eff29f022362849b64effd5ff8), unrelated edits, website and push restrictions. Old test/sealedtest, Pediatric, Unity training and21quarantined patients remain excluded. IDs/media/weights/per-case arrays stay server-side. User has no annotation resources; no invented labels/clinical validation.

## 2. Why this is a new, bounded condition

Read index and linked corrected records before changing code. A57 completed68000updates, not only core32000: backbone adaptation helped J−F but did not beat A08; full-data refit and four-depth did not resolve the complete target. A58 completed48000updates/72rows/132contrasts (aggregate34002ea, terminal7155ff2). D2−D1 selection segment differences−.046334/−.043495 had negative conditional intervals, while calibration+.001198/+.001263 was uncertain. Frozen A08 unaries were not enough: A5C selection11/11 became7/11. D0g=0 had similar native/true-short harm, so explicit duration alone is not the explanation.

A58's duration-stress calibration is a real scoped positive: D2−A08 segment+3.26/+3.16pp, true-short FN23/159 versus54/159. Selection gain was uncertain, native harm persisted. This motivates a separate selective repair decision, not another unconstrained complete-label decoder. We do not infer anatomy or latent-feature superiority from posterior trajectories.

**A14–17 already tried local routing.** A14 chose between A08+B3 and A13 switch outputs; A15 enlarged session-disjoint pools; A16 used count utility/weighted BCE; A17 tested ordered vs shuffled local trajectories. Their small/mixed gains and failures remain. A25's four-window differentiable segment-risk continuation also failed. A59 does not claim routing/utility learning is new. Substantive differences here are: raw A08 supplies all replacement labels; JEPA is only a matched evidence input; one accepted edit removes simultaneous-edit utility mismatch; explicit majority protection is evaluated as the same-weight inference ablation; and training construction covers the short-duration support implicated by A58. Exact scalar utility regression replaces threshold/calibration search. These are testable design changes, not established publication novelty.

## 3. Sources and data roles

- A08-s0 `models_frozen/video_within_view_v1/all_s0/final.pt`, SHA9b144925f32fac7d725d13231a132abeb6635fa44b6fe0dbcec7f5b59d035bbc, complete400keys. Keep original16frame/10Hz/marker/gamma/shortside256-centercrop224/FP32/TF32off/eval and **original stride5 centres** exactly. B3 fixed: half_window7, threshold.19660949291632382, nms_radius5, threshold_quantile.999. No new B3 selection.
- JEPA A57 core seed2026110257 J-high step6000 SHA2fa26f2b3ffbffd4d8481c6ac1ff2190572f18648e13f13dbf37333a17bdd985, complete214keys. Keep original64chunk/stride32/positive-triangle stitch, BF16 encoder/FP32 head and full5class probabilities. Freeze all producer parameters; no new latent arm or source choice.
- Reuse A58 FIT2736/402sessions and inner628/100sessions native groups. Do **not** train on inner. A08 historically saw some inner sources; neither the whole producer nor reused development is OOF/independent-patient evidence.
- Generate **new1500FIT and300inner constructed streams** from their own groups with bank seed2026110359, existing4..8fragment/class-uniform/p_same_next.35/gamma.90 probability.5 recipe, but length uniform4..min(80,n), then start uniform0..n−length. No repeated cine within a stream; reuse across streams allowed. Do not select/retry recipes for desirable labels/results. Full16/64/padding/context footprints remain group-disjoint.
- Main evaluation is unchanged old OPEN selection/calibration native284/237, multi120/120, pairs137/115. Include the A58 fixed duration-stress bank as **retrospective reused stress**, plus a **new fixed construction-confirmation bank** of120streams per outer half using the same4..80 recipe and seed2026110359. Freeze both new outer recipes/input identities before any head result; no label/metric read of fresh confirmation before all4heads/inner records/weights freeze. It is a new construction draw from previously used sources, not new patients or an independent clinical holdout.
- A58 original-stride5 caches exist for old main and old stress, not full FIT/inner. Never interpolate dense-stride2 posteriors into an alleged original5 cache. Exact prior full-context cache reuse requires refs/order/gamma/window/axis/sourceSHA equality. New constructed streams must actually forward through joins, never concatenate native logits. Qualify reusable native JEPA records from A58; forward new joined banks/fresh confirmation only where absent.
- Reuse qualified source libraries, not A58 CLIs with hardcoded namespace paths. Source loader reads only allowlisted video_id/frames_dir metadata and materializes only allowed media. Transient ID/path parsing is disclosed; no false claim the global index file was never opened.

## 4. Fixed candidate/action/guard contract

For one complete probability axis pA,pJ of lengthT, derive `R=argmax(pA)` and `B=original_B3(pA)` (class-order ties0..4). No GT, sourceID, stream kind, joins or source cut boundaries enter inference.

Candidates are exactly **maximal contiguous components of R!=B**, with no merging, maximum/minimum length threshold or expanded boundary grid. A candidate action replaces B[s:e] by R[s:e]. Empty disagreements give no-op. Candidate order is increasing(s,e). Unchanged agreement positions always retain B. No JEPA class is inserted without raw A08 support.

For a training sequence, compute Base matched segmentsM and predicted segment countP with the unchanged class-aware one-to-one IoU50 matcher. For each candidate applied alone, obtainΔM,ΔP. Let F_fit be Base's pooled multi segmentF1 on the new1500FITbank, frozen before head training. Target utility is **u=2ΔM−F_fit×ΔP**. For edits of the FIT multi bank, this is the exact numerator sign of that single edit's pooled-F1 change at the FIT operating point. Native candidates use the same fixed count cost as a stability proxy; it is not a native macroF1 surrogate or an exact pooled-multi change for a native recording. ΔP is observable and is also a model input. Targets/label matching are server-only training preparation; inference sees no utility/labels. All whole-sequence metrics are re-matched after inference. A14/A16 count utilities are acknowledged; A59's one-edit limit makes actual applied action match the single-action supervision, without claiming general multi-edit optimality.

Regressor predicts signed utility q. Each policy considers only q>0, ranks by descendingq then(s,e), and accepts **at most one** action:
- **U:** highest positive candidate, otherwise no-op.
- **G:** highest positive candidate whose edited full-axis `np.bincount(...,minlength=5).argmax()` equals the same quantity for B, otherwise no-op. Apply this rule to **every** sequence; there is no native flag. Do not rescore after editing or admit further patches.

G guarantees equality of the recording occupancy-majority class (and thus native confusion/macroF1 under the existing metric) to Base, including ties. This is a **mechanical preservation property**, not new learned recognition, patient safety or guaranteed frame/segment quality. It also preserves Base's wrong recording class and may veto useful repairs. Report native frame accuracy/internal cuts and all vetoed/gained/lost segments so the constraint cannot hide harmful short bursts. One edit per full offline sequence limits repair coverage/latency; disclose it.

## 5. Exact matched feature/model/training contract

For each candidate, regions are L=[max(0,s−16),s), Z=[s,e), R=[e,min(T,e+16)). For each source and each region concatenate: mean5probabilities; mean5absolute adjacent probability differences (zero for fewer than2positions); mean entropy in nats; mean top1−top2 probability margin. Empty regions give zeros. Each source contributes3×12=36features.

Common26features: log1p of L/Z/R lengths (3); mean Base one-hots over Z (5); mean raw-A08 one-hots over Z (5); Base class immediately left/right as5one-hots (10, zero at absent edge); two L/R availability flags; observableΔP (1). Exact feature dimension is **98**. FIT-only per-feature mean/std (std floor1e−6) is frozen, no label/inner/outer fitting. Input blocks:
- **A:** common + A08 + duplicate A0836block.
- **J:** same common + same A08 + JEPA36block.

Both heads are `Linear(98,32) -> GELU -> Linear(32,1)`, **3201parameters**, identical seeded initial tensors. Zero final weight/bias gives q=0 and exact no-op for both policies. First-layer zero gradients at the initial zero output are expected; verify later liveness/moments/drift. No trained unary scaling or producer gradients.

Train **2feature heads ×2seeds2026110359/2026110360 =4physical models**, each6000AdamWupdates, lr1e−3, wd.01, betas(.9,.999), eps1e−8, warm100/cos6000, FP32, gradientclip1, MSE(q,u). Batch32native candidates +32constructed candidates, drawn uniformly with replacement from each kind's complete eligible FIT candidate list; shared draw plan and initialization across A/J. Loss is mean of the two kind means. If one kind has no candidate, use the nonempty kind alone and record the exact plan/mask; if both empty, document the genuine no-action domain instead of pretending training. Log all eligible/ineligible sources/sessions and actual candidate/position exposure; not every forwarded source is an optimizer exposure.

Set CUDA_VISIBLE_DEVICES empty **before torch import** for CPU jobs; verify cuda is uninitialized and no NVIDIA process appears. If unexpected context occupation occurs, count full occupied wall once and enforce budget rather than claiming GPU0 from a raw flag. Latest every100 has full Adam/RNG/draw cursor/normalizer/identity; immutable weights at2000/4000/6000. Inner whole-task readout at those points is diagnostic **only**, no checkpoint/threshold/LR/coefficient selection. All4models finish6000 unless engineering/hard-budget/external failure. Freeze complete weights/inner/identity in actual Git before fresh outer results.

Four logical policies per seed are A-U,A-G,J-U,J-G. G/U share exact model weights, not duplicate training. Primary **J-G−A-G** and J-G−Base at fixed6000; same-weight J-G−J-U/A-G−A-U quantify guard cost. Report all systems, seeds, halves and tables, not only the best. No score threshold search: q>0 and one edit fixed throughout.

## 6. Fixed evaluation and interpretation

Evaluate main, reused A58stress, and fresh construction-confirmation. The stress tables reuse identical main native/pair predictions; only multi changes. Baselines are originalA08-s0+B3, rawA08-s0 and rawJEPA; original threeA08 main references reproduce historical rows. A58 full results remain historical context, no whole-system architecture causal claim. No new B3/postprocessing or copied oracle prediction is a system.

For all tables report full5class native macroF1/confusion/recall, frame accuracy/internal cuts, multi/pair segmentF1@50, boundaryP/R, same-label cuts, true-shortFN and false-shortFP at10/16positions, A4C↔A5C, recovered/lost true segments, candidate counts/lengths, score sign, accepted/vetoed actions, and source/exposure/inference/CPU/GPU/storage costs. G's native-majority equality must be exact; do not call its zero difference empirical evidence of generalization. Include A5C11/18 denominators and all other classes.

Predeclare paired contrasts for eachseed/half/table: J-G−A-G; J-U−A-U; J-G−J-U; A-G−A-U; each of A-U,A-G,J-U,J-G−Base. Eight contrasts ×2seeds×2halves×3tables=**96comparisons**. Each uses5000paired bootstrap replicates, seed2026110359, whole36native sessions/whole120fixed streams/wholepairs. Shared-source bank/uncertain patient identity/reused development/multiplicity caveats stay explicit; undefined ratio replicates counted and null where all undefined. CIs crossing0 are not equivalence. Full matching/count arithmetic independently recounted from committed labels.

After all frozen readouts, run a pre-frozen, cache-only label-assisted action-space diagnostic. For each multi stream enumerate Base and each single candidate action (separately U-domain and G-domain), recording exact M/P/T. Compute the **exact constrained pooled-F1 upper bound** over these per-sequence choices using finite fractional maximization: initialize fraction a/b=0/1; per sequence maximize integer 2*M*b−a*(P+T), tie to no-op then(s,e); set a_new=2*sum(M), b_new=sum(P+T); stop when a_new*b==a*b_new exactly. Positive denominators and finite options make the monotone procedure terminate; verify against exhaustive Cartesian choices on small examples. This only bounds the declared one-edit, raw-supported action domain. Also report baseline-missed true-unit coverage across individual candidate actions (union is not jointly reachable), vetoed useful actions and false/native harm. Oracle choices use labels and are never supplied to a system, gate, threshold or later policy in this matrix. Freeze diagnostic arithmetic/inputs before running; no oracle-guided candidate/model search.

Success requires actual short/change improvements with no hidden native-frame/cut harm and repeatable effects across bothseeds/halves/main/freshstress. G's mechanically retained macroF1 alone is not success. Regardless outcome, preserve all negatives/adaptivity. New training lengths were chosen after A58stress feedback; fresh recipes are prospectively frozen construction only. No annotation/clinical/independent-patient claim follows.

## 7. Files and execution tasks

All new files are round-specific; do not edit old qualified libraries to make checks pass.

### Task1: data/source/config identity and measured readiness
Create `configs/experiments/anchor_edits_v1.json`, `scripts/anchor_common.py`, `scripts/prepare_anchor_edits.py`, `scripts/cache_anchor_edits.py`; public `docs/results/2026-11-03_anchor_edits/` and private newnamespace.
- [ ] Record clean parent/birth/main8staged guard and inherited19focused baseline checks.
- [ ] Confirm live queue/PID/GPU/old terminal/cache pins on reconnection, no duplicate job. Keep connection failures honest; local code progresses independently.
- [ ] Freeze new1500/300banks,120+120confirmation recipes and complete source/role identities; verify all footprints and metadata columns. Qualify native/cache reuse exactly.
- [ ] Strict-load400/214keys once against pinned source; reuse prior functional checks and verify new original5/axis/source graph parity on overlapping authorized refs. Benchmark representative/longest actual FIT A08windows, JEPAchunks and actual CPU64candidate backward/update/readout/checkpoint path.
- [ ] Freeze a full measured forecast for all executed+remaining stages,96bootstrap comparisons/audits/closures/storage/earlierdeadline and2GPUhreserve before formal admission. No model-performance-based arm reduction or hidden retry.

### Task2: deterministic action and protection graph (TDD)
Create `echo_routing/temporal/anchor_edits.py`, `anchor_features.py`, `tests/test_anchor_edits.py`, `tests/test_anchor_features.py`.
- [ ] First failing tests for components/no merging/all5classes/empty axes rejection/no-op/raw-only/agreement positions/atmostone/ties/guard-veto fallback; then minimal implementation and passing tests.
- [ ] Enumerate small5class examples proving G majority invariance under ties and showing native macro protection can still permit harmful frame cuts. Include correcting Base-wrong class veto, no GT/kind/source/truejoin input, padding/region boundaries and long zones.
- [ ] Check u sign/full pooledF1 identity on exhaustive small candidate edits with formal matcher, removing false short/lost true/recovered short andΔP, and no simultaneous edits. Validate exact98feature order, empty regions and source-block swap/no labels in features.

### Task3: matched utility models/training lifecycle (TDD)
Create `echo_routing/temporal/anchor_training.py`, `anchor_resources.py`, `scripts/train_anchor_edits.py`, `scripts/supervise_anchor_edits.py`, `tests/test_anchor_training.py`, `tests/test_anchor_resources.py`.
- [ ] Test3201counts/common initialization/zero-output exact no-op and frozen producer roles; finite gradient and later first-layer liveness, exact optimizer coverage/step counters/recipe rejection, shared seeded kind-balanced draw plans and empty-kind behavior.
- [ ] Test repeat and2+2disk resume against4, cursor/Adam mismatch rejection, atomic cache/checkpoint claim recovery, complete SHA binding and no ambiguous silent replay.
- [ ] CPU and GPU occupation measured separately, inherited closed receipts retained, CPU allowances separated, one execution lock/actual PID/start times and all failures charged. Do not import hardcoded A58namespace CLIs for writes.
- [ ] Formal launch only after newgraph checks, immutable actualGit code/data/config/source freeze and measured allmatrix admission. Record advancing actual Adam steps for each of4heads.
- [ ] Execute all4×6000,3inner points each, curves/loss/native/constructed/gradient/LR/exposure logs; weights and all12inner rows freeze before outer.

### Task4: full readout and independent aggregate
Create `scripts/readout_anchor_edits.py`, `scripts/report_anchor_edits.py`, `tests/test_anchor_readout.py` (reuse verified metric/paired helpers without mutation).
- [ ] Freeze readout/model/policy identities, reproduce A08all3 main and zero-op/guard invariance. Evaluate all8logical systems across main/oldstress/freshstress; exact native/pair reuse and checkpoint dedup disclosed.
- [ ] Run all96paired comparisons, action/short/boundary/recovery/native harm tables, full curves and input/resource provenance. Independently recount from full original-axis label journals and pin source SHA/Git objects.
- [ ] Run only predeclared cache-only action-space diagnostics after systems freeze; retain local positives and all negative results. No post-result threshold/model/candidate search.

### Task5: scientific/resource/integration closure
- [ ] Write readable findings/figures distinguishing design guarantees, trained effects, negative/diagnostic outcomes and publication gaps; visual QA plots, hashes tie to the sealed aggregate.
- [ ] Verify actual optimizer/model retention, all current contracts/weights, all recorded worker/supervisor/guard exits, GPUidle/queue/pending0, measured costs/allowances/storage upper/earlierdeadline. Unknown OS status remains null; never infer it from manifest success.
- [ ] Update only A59 index entry; selectively sync exact public code/config/tests/card/aggregates into main while preserving8staged bytes and unrelated changes. No push/website change.
- [ ] A59monitor stays quiet during healthy work, notifies meaningful results/failure/required action/full closure; remove only after full admitted matrix/audit/public sync or honest full-phase hard-cap/external termination. Do not recreate A58 or modify other monitors. No automaticA60.
