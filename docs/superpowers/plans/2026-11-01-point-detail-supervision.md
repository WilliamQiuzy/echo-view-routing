# A52 point-frame detail supervision implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:executing-plans inline. One read-only plan/code reviewer; one executing owner and GPU queue.

**Goal:** Test whether precise frame-level view labels can teach a spatial encoder to recover A5C while retaining correct A4C, then conditionally test target transfer against unchanged A08+B3.

**Architecture:** A directly supervised, fully adapted ResNet50-FPN sees only the labelled marker frame. H retains its complete field of view at a 448 canvas; R first reduces the same canvas to224 and enlarges it back. Same model, pixels before the detail operation, labels, exposure, initialization, optimizer and computation. No learned gate, anatomy pseudo-label, changed loss or threshold sweep.

**Tech Stack:** pinned existing COCO ResNet50-FPN, PyTorch FP32 deterministic CUDA, Pillow bilinear RGB, existing durable A51 update/readout machinery, server-only private images and scores.

## Evidence and changed condition

A43–A48/PW-T1-C recovered A5C with unacceptable A4C regressions. A51's valid limited702-marker pilot trained L/W600each: manual L source fit564/569 but diagnostic102/133; A5C31/39, A4C false positives23/94 versus original A08 G16/39 and4/94. L gained15 A5C classifications and newly damaged19 previously correct A4C windows; L versus W adds2 positives but changes8/6 different A4C mistakes. This supports a transfer/overfitting problem, not a proven anatomical mechanism. Mixed/unknown context336/702 can disagree with point supervision. No A51 target training followed either failed predeclared source gate.

A36 already compared native320 versus reduced224 detail on **frozen A08 video embeddings and residual heads**; its high-detail increment failed. A21 added full-field224 video information. A48 adapted layer4/FPN with valve-box source state; PW-T1-C adapted layer4/FPN on weak19-pair acquisition-purpose supervision. A52 instead trains **every spatial convolution/FPN parameter on qualified direct marked-frame view labels**, omits unlabelled surrounding frames, and matches detail at448. This distinct supervision/representation combination is a hypothesis, not evidence that resolution or an LVOT feature will help. No claim of unseen clinical validation or anatomical localization.

## Fixed source and input contract

- Same EchoUNAL release5e853ea4bb3eee8bb103fa44ad6ddb13fc63a5eb, CC-BY-NC4.0 research scope, existing68 selected AVIs. No new download, quarantine, pool expansion or protected project test read.
- Same fixed702 chronology-qualified markers:569fit/133diagnostic,36/10patient identities; manual A4C/A5C330/239fit and94/39diagnostic. Five original ambiguous FIT markers remain UNKNOWN/excluded, no replacement. Original707 binding claim remains failed. Reusing the133 A51 diagnostic markers is explicitly exploratory, not a fresh holdout.
- Decode original released AVI ordinal `centre`, explicitly require `centre == sealed frames[8]`, verify full AVI SHA and exact native PTS against sealed chronology; raw point RGB SHA bound to request, patient, fold and manual label. Audit raw and processed H/R hashes. Retain within-partition same-label duplicate rows and report unique-image counts; abort cross-patient/partition collisions or conflicting labels without silently dropping or reselecting frames.
- H: RGB, aspect-preserving resize to long side448, integer rounded dimensions, centered black padding448square. R: H bilinear224square then bilinear448square, same canvas. No crop, color jitter, masking, augmentation or source gamma. Normalization /255 and ImageNet(.485,.456,.406)/(.229,.224,.225). FPN P2–P5 global means concatenate1024, scalar linear1025 head W0,bias−2. FrozenBatchNorm buffers remain unchanged; **all** backbone/FPN/head parameters require gradients. No detector proposals/heads, confidence gates or attention.
- Fit-only geometry control uses raw width,height,fps,decodedN, point-image mean/std/chroma and aspect; fixed standardized logistic C1, constant fit prior. Report historical A51 G/geometry separately; do not substitute video/point controls. Point duplicate eligibility checked before model preparation.
-600updates×8requests,4perTRUEclass, cycling deterministic seed2026110151 covers all569; same immutable plan for H/R. Source BCE mean, AdamW backbone1e−5/head1e−3 wd1e−4 betas.9/.999 eps1e−8 warm60cosine600 clip1, deterministic eval FrozenBN FP32/TF32off. Each occurrence real forward/backward, even zero trunk gradient at first update. Final state only, no best checkpoint selection.

## Frozen decisions before new outcomes

For each H/R source classifier at strict score>0, require BA>=.80, both class recalls>=.70, correct-A4C FPR<=historical A51 original A08 G4/94+.02, AUC>=.85 and >=historical G AUC+.03, BA>=new point geometry+.10 and constant+.10. No adjusted threshold. Absolute source condition and detail increment are separate: H−R BA>=.05, no extra A4C false positives, AUC>=R+.03. Passing the absolute condition in either arm permits BOTH target arms if the complete remaining resource forecast fits; detail increment alone never permits unsafe transfer. All source arms and controls reported regardless of verdict. If neither absolute passes, close A52 without target training, gate revision, extra seed or rescue variant.

Only if qualified: reset each source head0/−2 and train both H/R on both original A31 fit folds using unchanged A44-C600plans, weighted full membership mean/BCE/two real passes with per-occurrence J-transpose-v including zeros. Point image is exactly the original request marker's raw numeric JPEG reference and rounded gamma LUT, followed by A52 spatial pipeline; changed crop/norm explicitly disclosed, no claim of A08 pixel parity. Preserve all native positives34/69, correct A4C negatives714/699, other1/3readout only, eligible5030/5437 rows. H−R target contrasts the complete detail recipe during source/target training and target inference, not source pretraining alone. A common gain over A08 is a package gain, without a matched unpretrained point comparator. No B3, threshold, original Base prediction or non-A4C label changes. No additional generic target comparator; compare matched H/R and frozen A08, old A44-C as historical background only.

Target absolute acceptance unchanged:>=11/103A5C fixes from>=3recordings and>=1/fold;<=14/1413correctA4C harms; OOFnet>=5 and positive eachfold; complete segmentation F1 mean delta>=.003 and positive eachfold; native macroF1delta>=−.002 each; boundary recall delta>=−.005 each; short misses<=79 vsBase76. Detail benefit separate:>=5extra fixes,noextra harms,positive fullF1difference bothfolds. Source scores do not establish these outcomes. No source2/deployment/automatic next version.

## Files and execution tasks

### Task1 — data and scientific contract
- [ ] Add `echo_routing/temporal/point_detail.py` with letterboxing/detail/normalization, full-body model and source gates; tests `tests/test_point_detail.py` first fail then pass on geometry, detail difference, invalid inputs and strict protection failures.
- [ ] Add `scripts/train_point_detail.py` prepare stage: verify sealed metadata/AVIs/PTS, cache only702 point inputs, audit cross-patient/fold raw hashes and conflicts, fit-only geometry, immutable balanced true-label plan, code/config/input SHAs.
- [ ] Independently audit every point binding and fixed source partition with direct CPU reconstruction before source training. Audit report contains aggregate counts/hashes only on Mac.

### Task2 — source mechanics, once-only models and freeze
- [ ] Fit-only checks: independent BCE and derivative; exact optimizer parameter coverage; all-parameter finite nonzero gradients/drift after zero-head startup; microbatch versus full-graph gradients; H forced to R gives identical updates; repeat;3+3resume; full model bundle reconstruction; unchanged BNbuffers/Base. Record actual source timings, full two-arm/conditional-target time/storage forecast and remaining resources.
- [ ] One source supervisor/worker OS flock; durable pending update before each step; fsynced journal/state, refuse uncertain resume. Retain complete final optimizer/RNG through independent audit. No duplicate queues.
- [ ] Commit exact source code/input/checks and verify actual Mac git objects BEFORE prelaunch freeze and2×600 source training. Verify actual GPU PID/device/update, notify formal start once.
- [ ] Independently audit both600-step journals, optimizer states and all live tensor changes/frozen BN. Commit complete full-model/head/base/code/input/plan SHAs as actual Git objects, THEN freeze. Once-only per-row H/R133readout with durable pending guards; reuse historical G scores without rerunning G.
- [ ] Independent cache-only metrics/AUC/gates/paired changes and raw score/input hashes, no second model readout. Report fit/generalization discrepancy without attributing model features to anatomy.

### Task3 — conditional target, closure
- [ ] If source condition and full remaining resources pass, implement/check/commit/freeze all target code/inputs before4×600; independently audit4complete models then actual Git freeze before single opposite-fold raw readout. Recompute membership, means, one-way labels, source coverage, paired fixes/harms/fullF1/native/boundary/short/gates from private raw cache. Otherwise explicitly record target untrained/unread.
- [ ] Close A51 and A52 in own reports/index; preserve original invalid/full-source failures, all negative results and primary A08; integrate own changes preserving main dirty index and eight staged theory files, no push. Final no-own-worker check and actual cumulative resource receipt.

## Shared finite resources and stopping

Same A51 allowance, **no reset**: additional CPUupper6h/GPUjobwall4h/newpeak5.5GB above inherited7.0900909796CPUh/1.1272365705GPUh/2,925,834,156bytes. At A52 planning snapshot A51newCPU2.9641h/GPU.34302h/retained2.08325GB; conservative past peak2.22662GB. Track actual root active wall proxy/reviewer/failures/bootstrap/childCPU/closing; explicit empty wait0, do not precharge future. Reserve.35CPUh/.15GPUh/200MB before each formal launch. Source completeCNNstate≈108MB, optimizer≈216MB/cell; keep finalstateaudit, only remove safe redundant intermediate optimizer after independent acceptance if necessary, never weights/logs. Six possible newcells≈2.59GB (including duplicate model inside optimizer checkpoint) + atomic.33GB + pointcache.10GB, plusprior2.08GB≈5.10GB<5.5GB forecast; measure actual. If any runtime/provenance/remaining-budget condition fails, record and stop the relevant stage, never reduce600 or omit an arm to claim a comparison. All source media/IDs/per-case data remain server-only.

**Predecessor paired evidence:** cache-only `pilot/paired_learning_analysis.json` independently establishes L/G A5C16shared/15L-only/0G-only and A4C4shared/19L-only/0G-only; the newly gained/damaged wording above uses these overlaps, not marginals.

**Pre-model bookkeeping repair:** prepare completed immutable point cache/binding/plan but failed code hashing on a nonexistent dynamic Python `_classes.py` module. Preserve attempt1 CPU/failure; do not re-decode/replace rows. After independent binding, a separate `seal` stage publishes the manifest using extant fixed inputs and filters nonexistent module paths. No scientific change or model forward. Finalization recovery uses the actual retained optimizer path and anchors every model manifest to the current fixed input SHA.
