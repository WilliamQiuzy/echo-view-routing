# A48 valve-localization spatial adaptation implementation plan

> **For agentic workers:** Execute inline with executing-plans and test-driven-development. One fresh-context advisory card reviewer; Codex remains sole implementation/execution owner, no Claude/other queue. User authorized conditional direct A48 execution and independent audit.

**Goal:** Test whether an actually box-supervised aortic-valve spatial prior, adapted through its last convolutional stage and full feature pyramid, provides transferable A5C discrimination while protecting correct A4C and complete segmentation.

**Architecture:** R starts from AoVdetector’s ResNet50FPN trained with manually drawn aortic-valve boxes; G starts from the same architecture’s official COCO_V1 state used to initialize that detector. Both read identical original marker RGB plus frozen original A08 window embedding512. Layer4 and every FPN lateral/output convolution are genuinely trained; earlier body and FrozenBN buffers remain fixed. Pool all four spatial pyramid outputs to14×14, concatenate1024 channels and fit the same new spatial/scalar repair head. No box proposals, NMS, detector classes/scores, confidence rule or additional deployment gate is used. R−G tests a named localization-trained state, not a uniquely isolated causal effect of anatomy versus all upstream data/training differences.

**Tech Stack:** Installed torch/torchvision ResNet50FPN, FrozenBatchNorm2d eps0 matching official COCO_V1; pinned licensed weights, original A31/A44 loaders and evaluator. FP32 deterministic two-pass whole-run gradients. No upstream unlicensed code is copied or executed.

## History and new discriminative evidence

A32 found no mapping/marker/member defect and no verified per-frame LVOT truth; the existing clip labels cannot locate when the outflow tract appears. A43 max MIL2fix/4harm versus D25/112, A44 C12/22 versus N26/76, A45 reconstruction veto0/1, A46 motion58/161 and A47 frozen PanEcho clinical-report features5/15 versus ImageNet4/14 all fail the safe full-sequence package. More positive editing or fewer edits alone was insufficient. A35 view logits and A39/A40/A41/A36 negatives remain scoped to their actual configurations.

The missing ingredient is a testable local image-to-structure signal rather than another scalar boundary shift. That is a hypothesis, not a diagnosis of every residual error. A47’s study-level measurement/report state did not supply verified local anatomy. A48 uses an upstream objective that explicitly localizes the aortic valve by image boxes, includes apical3CH, and adapts the multi-scale spatial representation on the actual repair task. This differs both from global report supervision and from frozen feature-head fitting. Shared valve structure motivates transfer to A5C, but the source has no A5C/A4C validation; no AV detection, attention or anatomical visibility is assumed on EV9V before the result.

Direct new operator-site training was investigated first under the separate qualification card. EchoXFlow’s fixed static B-mode sector carries an LVOT PW sampling pose, but the decoded array’s raw SHA differs from data_sha256 and the published hash definition remains unresolved. That route is **not admitted**, no model was trained on it, and neither the spectral trace nor operator pose was invented into an expert A5C/absence label. Its failure/traffic/process receipts remain in the A48 ledger and source report. The present model choice precedes every A48 model forward and every new EV9V outcome.

## Exact sources and source eligibility

R HF revisiond35108887c9bcd432d852da103e684db6c2fd969, model_avdetector.pth165770983bytes, SHA25620951944cad682ea8dfa8ca9635ac12411123453b6f4b9c6702fa958974b8557. Weights CC-BY-NC-SA4.0, noncommercial research adaptation, no deployment/service/redistribution. Source facts inspected at GitHubcb59a4b0398db42997dcad87cdebf99980e87c3f: manual two-click boxes, PLAX/PSAX/3CH classes and COCO_V1 initialization; source repository license not established, so its scripts, preprocessing and unsafe eval/pickle paths are not reused. Published92/98 accuracies and the EchoAVC clinical paper are not evidence of this branch’s A5C transfer or localization on our data. Original source masked/resized images and disease population differ; upstream exact overlap/validation reliability remain uncertain.

G is torchvision FasterRCNN_ResNet50_FPN_Weights.COCO_V1, official258fb6c6 URL. Acquire/inspect safely with weights_only=True. Exact full classifier shapes R4/G91 are checked but unused. Strictly extract every backbone tensor, including FrozenBN statistics. COCO’s FPNv1 keys are converted only by the exact torchvision documented v1→v2 alias mapping (16 expected renames); no new value, missing-key fill, random tensor or unexplained alias allowed. This initial strict-name failure is preserved, and canonical state must equal the official full-model loader tensor for tensor. Require actual differences in layer4 and FPN, not changes to an unused classifier. Keep full raw weights for official parity and actual SHA receipts; all sources/code/config/manifests are real git objects before first source-backed/EV9V model forward (synthetic unit tests excluded).

## Inputs, representation and trainable parameters

Source0 first, original A31 two session folds. Base-A4C eligible749/771, request5030/5437, fit positives34/69, correctA4C negatives714/699, other1/3 excluded only from fit and retained in complete readout. Keep every original16-frame request, marker, crop/gamma, duration/multiplicity/nearest-centre mapping and fixed600 positive+3negative plans. A08 appearance512 is unchanged and frozen. New image is exactly the original marker slot’s clean224RGB from the original loader; pixel references are selected only by original Base eligibility, never a new score or target label. Deduplicate immutable pixels by full source/frame/gamma identity in RAM only; still forward every actual window occurrence in both passes, even repeated/zero-coefficient ones.

Use the official GeneralizedRCNNTransform eval path: originalRGB /255, ImageNet channel normalization, min_size800/max_size1333 and size_divisible32, no random scale. This is resampling the same input, not adding raw resolution or a new crop. Validate against the full official detector transform and backbone after strict source load. No clinical claim of reproducing upstream masking/domain is made.

Trainable spatial head:

```
P2,P3,P4,P5 -> each adaptive_avg_pool14×14 -> concat1024
per-token LayerNorm1024 -> Conv2d1024→8,k1 -> GELU -> Flatten1568 -> Linear1568→128 -> GELU
appearance LayerNorm512
concat640 -> Linear640→128 -> GELU -> Linear128→1
```

294281 head parameters (verify). Source R/G backbone initializations differ; fresh head initialization is identical per source/fold, seed2026102948+10*source+fold. Last weight0/bias−2 gives Init=Base. Enable only body.layer4.* and fpn.* parameters plus every head parameter; earlier body and all FrozenBN buffers remain exactly unchanged from their named source. Every FPN output block participates through its corresponding scale. The extra parameter-free pooledP6 is unused. Eval/no dropout throughout, no batch coupling.

## Fixed objective and real gradients

Exactly A44C: each of600 updates uses one positive and three correctA4C negative whole Base runs, complete original members and weights. r_j=sum(v*s); loss=softplus(−r_pos)+mean(softplus(r_neg)). No changed margin, multiplier, risk loss, threshold, sampling plan or best-epoch selection. AdamW FP32, betas(.9,.999),eps1e−8,wd1e−4; layer4+FPN lr1e−5, head lr1e−3;60 warmup/cosine600, global clip1. TF32 off, strict deterministic algorithms.

Two genuine forwards over every occurrence: first no-grad scalar collection, autograd through exact whole-run scalar loss to get each score coefficient, second real raw-image autograd in microbatches4 accumulating J-transpose-v, including coefficient0 occurrences. No frozen/cached backbone-feature or cached-head substitution. Preserve identical computation and occurrence ordering across R/G. Upstream first-step gradients are structurally zero from the zero scorer, then must become live. Track all selected parameter groups, real gradient and final drift separately from frozen tensors.

## Sequential engineering and execution gates

- [ ] **Source/data preparation:** complete safe source manifests, mandatory source tensor migration/parity, all original row/reference/marker/membership hashes and session isolation; exact 600 plans/all negative coverage. Commit source/config/code/check identities before first source-backed/EV9V model forward (synthetic unit tests excluded).
- [ ] **Core/tests:** failing synthetic tests before reusable code: source aliases strictness; full4-scale use; parameter flags/count/Init; weighted objective; two-pass vs retained graph including zero coefficient; legal deployment and repeat/resume. Files echo_routing/temporal/valve_localized.py and tests/test_valve_localized.py.
- [ ] **Production check:** real cached-RAM pixels match original loader, transform/full official source backbone parity, InitBase, full real minimum-occurrence retained graph vs production two-pass gradients, formula/coefficients independently FP64, batch invariance, repeated6, forced-equal R/G source path,3+3 optimizer resume, every selected trainable tensor, each FPN lateral/output block, layer4 and head has observed finite nonzero gradients after initialization, and all frozen tensors unchanged; AdamW drift alone does not qualify. Maximum occurrence production step has real backward, peakGPU<40GB. Measured full source0 forecast plus actual prior task usage must fit GPU6h/CPU3h/new3GB, including audit/closing; failures retained. No formal train before all critical checks pass.
- [ ] **Formal source0:** one queue, R/G×fitfold0/1, each600 actual updates. Log actual PID/device/update/all occurrence row and coefficient counts, gradient groups, lr and optimizer counters; atomic own checkpoints only for continuation, no re-training a completed model or label-driven repair.
- [ ] **Commit/freeze:** four full model/head/source/A08/config/code/input/plan/check/optimizer SHA manifests, actual local git objects before freeze and once opposite-fold readout. Server has no project.git; root verifies every loaded repository/config blob against local git and public/source SHA before recording freeze.
- [ ] **Once readout:** only original BaseA4C full weighted mean scalar r>0 becomes A5C; other labels copied, r0 stays; RLE may merge and harm true boundaries. Save each arm/fold private window scores before evaluation for cache-only recovery. No detector threshold, box veto or new runtime gate. Original nonA4C invariance is structural, not learned safety.
- [ ] **Independent audit:** all4×600 actual optimizer/input logs; every trainable tensor and every frozen source tensor; full original marker refs and nearest-centre sample expansion, FP64 means, one-way predictions, true A5C source identities/counts; exact paired R/G fix/harm overlap; original evaluator full F1/native/boundary/short counts and all gates. Rebuild functional head and explicit4-scale pooling/normalization on a fixed first/middle/last private subset with independently constructed official backbone, no complete readout repeat. Server/local/actualgit public aggregate SHA and final resources/no own jobs verified.

## Scientific decisions, not extra runtime rules

**Absolute package, either arm:** find>=ceil(.10*S1) with>=3 true A5C-coded source recordings and>=1 per fold; harm<=floor(.01*correctA4C); per-fold net>=0 and OOF net>=5; fullF1>=Base eachfold and mean delta>=.001; native delta>=−.003 and boundary recall delta>=−.005 eachfold; total short true misses<=Base+3; nonA4C labels unchanged, not no-op. Use unrounded metrics and actual source-specific denominators: S1 means original Base-A4C runs whose majority reference is A5C (103 for source0), protected correctA4C is1413 for source0.

**Separate named pretrained-state increment:** R finds>=G eachfold and+5OOF; R harms<=G eachfold; OOF netR−G>=5; fullF1 R>=G eachfold; absoluteR passes. Do not confuse this with the absolute package or make an anatomy-causal claim.

**Source2:** only if either source0 absolute package passes and measured shared-budget forecast permits both arms, run both R/G source2 under the exact same card,600 and original source-specific Base/denominators. Otherwise close at0. No new initialization/threshold/epoch/loss or extra variant, no automatic A49.

## Budget, privacy and closure

Shared whole A48 GPU6h/CPU3h/new3GB beginning04:30UTC includes all earlier source research/failed direct-data qualification/acquisition, imports/bootstrap, checks, actual training/readout/audit/closing, and owner/reviewer scope. Record process CPU separately from active-owner/reviewer proxy; empty waiting is not compute, no future compute charge. R/G raw/checkpoint/model copies and peak own resume files count; RAM-only immutable pixels create no disk-cache budget exception. All private frames/references/IDs/per-case outputs stay server-only. No dev/check/test/Pediatric/A26 runner/Unity training, expert/model-generated pseudo labels, teammate actions, push or outside messaging. Preserve8staged theory and other dirty. Round-only actual commits and A48 index/result report after verified closure, with corrected source failure and narrow claim retained.

Implementation runner scripts/train_valve_localized_a5c.py, auditor scripts/audit_valve_localized_a5c.py, config configs/experiments/video_valve_localized_a5c_v1.yaml, server runs/video_valve_localized_a5c_v1, report docs/results/2026-10-29_valve_localized_a5c. Reuse old mathematical/read-only loader/evaluator helpers; never call their output-writing stages or overwrite old artifacts. Use existing isolated-worktree integration pattern and avoid repeated Mac historical dataless reads.
