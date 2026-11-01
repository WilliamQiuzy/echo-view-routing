# PW-T1-C — completed clinical-purpose representation transfer

**Completed and independently audited. All six models trained for600real updates. The tested clinical-purpose representation does not supply protected A5C recovery.** Correct-purpose L fixes2/103A5C errors and damages18/1413previously correct A4C runs; original COCO G fixes4/damages26, shuffled-purpose S fixes3/damages25. Every arm lowers complete-segmentation F1 in both folds relative to Base.

## Question and substantive change

The question was whether the image representation learned from PW-P1's correct clinical acquisition-purpose supervision adds target A5C/A4C discrimination beyond both its same-image shuffled-label control and original COCO initialization. This changes upstream supervision, rather than the downstream loss, threshold or deployment gate.

PW-P1 provided a small exploratory source-task signal: L10/12source-heldout images correct/AUC.944444 versus S6/12/.333333, on19fit/6heldout paired exams. Those weak acquisition-purpose labels are not expert visible-LVOT/A5C truth. IMG-D1's fixed image-perturbation diagnosis had not demonstrated a safe border/OCR/grayscale/static-frame correction. A43–A48's corrected scoped negative results remain in EXPERIMENT_INDEX. The original PW-T1 stopped before formal training at its old resource gate; **that stop and its artifacts remain unchanged**. This is the newly user-authorized execution continuation of the same unresolved comparison, with an additive, prospective resource extension.

A50-PW's25/80formal source quantity failure and A49's0/109global annotation binding remain unchanged. No pool expansion, source-model retraining, pseudo anatomy labels or external-test training was introduced.

## Fixed matched execution

G/S/L use the exact PW-P1 ResNet50-FPN: spatial means of four256-channel maps concatenated to1024features, fresh1025-parameter scalar head at weight0/bias−2. All28selected layer4/FPN/head parameter tensors genuinely receive gradients and change; earlier body/FrozenBN tensors remain exact G. Inputs are each original request's marked RGB frame, original gamma/short-side256/centre224crop, then `/255` and ImageNet normalization. This image branch differs from A08's16-frame video encoder; the causal comparison here is **L versus matched G/S**, not a claim that initialization alone explains differences from earlier architectures.

Original source0session folds and A44-C600×4-run plans:34/69fit-positive runs,714/699protected-negative runs,1/3other eligible runs excluded from fit only. Every original membership and multiplicity weight is retained. Original class-balanced BCE on complete weighted run means, AdamW/backbone1e-5/head1e-3,60warmup/cosine600,clip1,FP32/TF32off,oneCPUthread. Every occurrence has two true full model forwards and one backward, including zero coefficients. No frozen-feature/head-only substitute, checkpoint selection or threshold adjustment.

Execution: G0,S0,L0,G1,S1,L1, **3600updates**,90918occurrences,181836training forwards/90918backwards. Full-fit Init/final scores for every model; after all six complete-model identities were committed and frozen, one opposite-fold pass on5030/5437eligible original request rows/arm (**31401scores**). Raw score cells were sealed before any aggregation or evaluator. No cell was repeated. No new dev/check/test/Pediatric/A26runner/Unity or external-test data was read.

## Once-opposite-fold target result

Historical training-source folds, not a new independent clinical test. Denominators:103true-A5C/Base-A4C errors and1413correct Base-A4C runs. Full F1 is the original complete constructed-multi-fragment segment F1@IoU50, averaged over the two folds.

| Arm | A5C recovered /103 | Correct A4C damaged /1413 | Net fixes | Full F1 mean | Delta from Base | Short misses |
|---|---:|---:|---:|---:|---:|---:|
| Base A08+B3 |0|0|0|0.956885|0|76|
| G original COCO |4|26|−22|0.954975|−0.001910|86|
| S shuffled-purpose |3|25|−22|0.954587|−0.002298|87|
| L correct-purpose |2|18|−16|0.955189|−0.001696|85|

| Heldout fold | Base F1 | G recovered / harmed; F1 | S recovered / harmed; F1 | L recovered / harmed; F1 |
|---|---:|---|---|---|
|0|0.963907|1/12;0.962705|1/10;0.962319|1/8;0.962333|
|1|0.949864|3/14;0.947245|2/15;0.946856|1/10;0.948044|

L's native macro-F1 deltas are0and−0.006682; boundary-recall deltas0and−0.002055. L merges away12Base boundaries overall but raises short misses76→85 and lowers full F1 in both folds. Fewer boundaries therefore do not establish improved segmentation. All original non-A4C per-sample labels are preserved structurally; that is not learned protection.

Every arm fails the predeclared absolute package gate. L misses the11-A5C recovery and3source-recording requirements (only2recoveries/2sources), exceeds the14-A4C harm allowance, has negative net fixes, fails full/native F1 and short-miss conditions. Boundary/structural checks pass. L also fails the separately predeclared contribution criterion against each control. Exact, unrounded values and every gate are in `comparison.json` and `readout.json`.

## What the purpose supervision contributed

There are **zero L-only A5C recoveries** relative to either control: L's two fixed runs are shared with G and S. G has2additional fixes; S has1.

Protection is not identical at the individual-run level. Against G,13harms are shared,13are G-only and5are L-only. Against S,15are shared,10are S-only and3are L-only. Thus L has fewer total harms while introducing some errors the controls avoid. Aggregate totals alone would conceal these differences.

The fit models demonstrate actual learning but not the desired transfer. All arms recover34/34fit positives in fold0and63/69in fold1. Fit A4C harms are G6/11,S5/8,L4/13. Good fit recovery and PW-P1's small source-task signal do not establish heldout A5C recovery. Under this exact input/architecture/600-step protocol, correct-purpose initialization supplies no exclusive recovered A5C and no usable protected-recovery package. This does not disprove every clinical-purpose or anatomical-supervision approach; local anatomical truth, patient-level validity and source-to-target correspondence remain unverified.

For historical context on the same103/1413denominators, A44-C fixed12/damaged22 and A44-N26/76; their full F1 means were0.956122/0.952001, versus Base0.956885. These are background comparisons across different representations, not the matched source-initialization contrast in this round. The working Base is retained; this fixed transfer configuration is closed, without source2, deployment or a post-readout loss/gate/seed rescue.

## Independent verification and provenance

-Original input/plan/source/code metadata are actual git objects in faa1d60before any continuation model check. Entire original Base/C/N evaluator and cached weighted-mean parity pass before formal training. Fit-only forced-equal/repeat6/resume3+3checks are bitwise; production gradient/reference checks and8targeted synthetic tests pass.
-`independent_models.json` reconstructs **all six full models** from immutable shared G plus **every mutable tensor**, verifies canonical full-state/frozen-state/head/source hashes, all3600journals/optimizer counters/plans/occurrence rows/physical forward-backward counts, nonzero gradients and parameter drift. Every eligible run membership is independently rebuilt from nearest-centre ties and multiplicities. Source recordings and sessions are disjoint across folds; verified patient disjointness is not claimed.18fixed fit-only reconstruction forwards have maximum absolute error4.95911e-5(<the1e-4check tolerance);0new heldout forwards.
-All six audited complete-model manifests were committed in **f687a3f**, then sealed in **b5fe484**, before any opposite-fold forward. Raw cell receipts bind model hash/fit fold/input identity and prevent duplicate inference. Score cache SHA: `e258175951cf63ad49627c66a39357c8da8c7e1766dcb697a05aecdff7329b3b`.
-`independent_final.json` recomputes all1520eligible runs, full membership-weighted means, strict-zero one-way per-sample labels, original full evaluator/counts/boundaries, true-class source attribution, exact G/S/L fix/harm overlaps and every absolute/contribution gate from the one raw cache. It performs0new model forwards.

See the [continuation card](../../superpowers/plans/2026-11-01-pw-purpose-transfer-continuation.md), original immutable [PW-T1 scientific contract](../../superpowers/plans/2026-10-31-pw-purpose-transfer.md), `protocol.json`, `models_manifest.json`, freezes, process receipts, independent audits and resource/closure verification. Private images/identifiers/per-case scores/journals/model tensors stay server-only; public outputs contain aggregates and SHA provenance.

## Resources and closure

Actual formal training wall2451.895sec, once-only opposite-fold wall293.001sec. Including bootstrap and fit/model checks, new GPU-job wall2799.764sec(**46.663min**); CPU evaluator/final audit are separately charged. This continuation inherits the old actual CPUupper5.185769h/GPU.349524h ledger and adds the new explicitly bounded3CPUupperh/2GPUh/600MBretained allowance; nothing resets the past. Actual owner/reviewer/source/check/failure/monitoring/audit/closing charges and the simultaneous storage upper bound are in `resources.json`; explicit empty compute waits cost0. The final actual closing tail is retained in `runs/pw_purpose_transfer_live_v1/final_resource_receipt.json`.

Only owned code/report/index changes are integrated, with eight staged theory and other dirty/external-test work preserved. No push or website change. Final verification records public server/Mac/git hashes, unchanged sources/Base, owned process closure, worktree cleanup and retirement of this round's monitor; the independent upstream Issue4 monitor remains separate.
