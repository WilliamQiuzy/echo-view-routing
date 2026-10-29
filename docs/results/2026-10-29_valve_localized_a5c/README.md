# A48 — valve-localized spatial adaptation

**Completed and independently audited, negative at source0.** Human-box localization pretraining R shows a descriptive ranking signal over its actual COCO initialization G, but it does not deliver safe A5C repair: R fixes14/103 target runs and harms50/1413 originally correct A4C runs; G fixes10 and harms22. Complete segment F1 falls for both. Both absolute packages and the separate R−G contribution gate fail; source2 is not opened. No A49, threshold scan, loss variant or new gate follows automatically. Original A08 + B3 remains the working reference.

## Research question and evidence gap

A32 found no window/marker/class-map/input-support defect and no verified per-frame LVOT truth. Clip codes and constructed streams cannot locate when the outflow tract is visible. A43–A47 did not yield a safe full-sequence package. This identifies an unresolved need for image-dependent local discrimination; it does **not** prove that missing LVOT visibility uniquely causes every residual error.

A48's substantive change is upstream human **aortic-valve box supervision**, plus live adaptation of layer4 and the entire spatial feature pyramid. A47 used study-level clinical-report pretraining and froze its image encoder. A46 supplied displacement; A45 supplied reconstruction support; A44 changed risk optimization. A48 uses none of their added losses, veto rules or confidence gates. Its matched R/G comparison tests a named localization-trained state, including all upstream data/training differences, rather than isolating anatomy supervision as the sole cause.

## Public source and admission

R is [AoVdetector at HF revision d35108887c9bcd432d852da103e684db6c2fd969](https://huggingface.co/CardiovascularImagingVHIR/AoVdetector/tree/d35108887c9bcd432d852da103e684db6c2fd969), released under CC-BY-NC-SA4.0 for noncommercial research adaptation. The [pinned source facts](https://github.com/CardiovascularImagingVallHebron/AoVdetector/tree/cb59a4b0398db42997dcad87cdebf99980e87c3f) establish manually drawn boxes, PLAX/PSAX/3CH classes, and COCO_V1 initialization. That repository's code license was not established; its scripts are not copied or executed. Licensed torchvision constructs and loads the network. There are no detector proposals, boxes, NMS or detector-class/confidence outputs in A48 deployment. The source does not validate A5C-versus-A4C discrimination, target-frame AV/LVOT localization, patient-disjoint source accuracy or clinical transfer. Its original masking/domain are not reproduced.

The full R file165770983bytes matches published LFS SHA20951944cad682ea8dfa8ca9635ac12411123453b6f4b9c6702fa958974b8557. G is the actual official COCO_V1 source, full SHA258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384. All281 canonical backbone state tensors load exactly;58 differ between R/G, including layer4 and FPN. COCO's16 FPNv1→v2 key aliases are the documented torchvision migration, with no value changes or missing-key filling. Original four-class/91-class detector heads are checked and unused. FrozenBN eps0 matches the official full COCO loader. See sources.json and the preserved first source-name failure.

The separate bounded [EchoXFlow](https://huggingface.co/datasets/Ahus-AIM/EchoXFlow) operator-site route was **not admitted**: the fixed raw B-mode array's SHA differed from published data_sha256 and the hash definition/read-path ambiguity remained unresolved. This is not a claim of dataset corruption. No EchoXFlow pixel, PW trace or operator site trained A48; no expert A5C presence/absence label was invented. Its failed qualification, source acquisition, vendor bytes and CPU work remain in this round's shared budget. The corresponding qualification/acquisition plans and scripts are preserved.

## Fixed experiment

[Full card](../../superpowers/plans/2026-10-29-valve-localized-a5c.md), [config](../../../configs/experiments/video_valve_localized_a5c_v1.yaml). Source0, original A31 session folds,749/771 original Base-A4C runs and5030/5437 unique eligible request rows. Fit positives34/69, protected correct-A4C negatives714/699; other1/3 runs excluded only from fit. Every original request, marker, crop/gamma, duration/multiplicity, nearest-centre membership and fixed600 A44-C plans remain unchanged. Original A08 appearance512 is frozen. Clean original marker RGB224 is resized/normalized by the official deterministic800 transform; this resampling adds no new raw detail.

Each of P2–P5 is pooled to14×14 and concatenated to1024channels, then read by the same294281-parameter spatial/appearance head. Train body.layer4, every FPN lateral/output block and the head:18580873 trainable parameters,27093577 total. Earlier body/FrozenBN state stays bitwise fixed. Head weight0/bias−2 gives Init=Base. R/G have identical fresh heads per fold, complete weighted-mean run scores, original balanced A44-C BCE, AdamW,600 updates, warmup/cosine and r>0 one-way deployment. No threshold selection or best epoch. Two real passes over every occurrence, including repeated/zero-gradient occurrences; the second pass performs real raw-image J-transpose-v. Immutable pixels are deduplicated only in RAM, with no backbone-feature/head cache replacing training.

All four models completed600 actual optimizer updates:14778 occurrences per f0 arm,15528 per f1 arm,60612 total and121224 real marker forwards across both training passes. Every selected tensor has observed finite nonzero gradients after the structural-zero initialization step and final drift; all frozen tensors and the original Base SHA remain unchanged. CPU hard limit4500seconds was stricter than the4700-second pre-authorization bound; actual formal process CPU was0.619380h.

## Results: same source0 outcome denominators

Counts are original Base-A4C target A5C fixes / originally correct A4C harms. These are descriptive cross-round outcomes on the same exploratory folds, not matched cross-round causal effects.

| Model | A5C fixed /103 | A4C harmed /1413 | Net | Complete segment F1 mean | Missed short true segments |
|---|---:|---:|---:|---:|---:|
| Base | 0 | 0 | 0 | .956885277 | 76 |
| A43 M | 2 | 4 | −2 | .956306238 | 76 |
| A43 D | 25 | 112 | −87 | .944178842 | 114 |
| A44 C | 12 | 22 | −10 | .956122151 | 86 |
| A44 N | 26 | 76 | −50 | .952000617 | 104 |
| A45 N+A = N+U | 0 | 1 | −1 | .956692264 | 76 |
| A46 F | 58 | 161 | −103 | .947259188 | 131 |
| A47 PanEcho P | 5 | 15 | −10 | .956157652 | 80 |
| A47 generic G | 4 | 14 | −10 | .956350170 | 80 |
| **A48 R: valve-localized** | **14** | **50** | **−36** | **.954397164** | **91** |
| **A48 G: COCO initialization** | **10** | **22** | **−12** | **.956127686** | **84** |

| Heldout fold | Base F1 | R fixes / harms | R F1 | G fixes / harms | G F1 |
|---|---:|---:|---:|---:|---:|
| 0 | .963906582 | 10 /36 | .959456838 | 5 /14 | .963063237 |
| 1 | .949863972 | 4 /14 | .949337490 | 5 /8 | .949192135 |

R deletes45 Base boundaries and G27; short misses rise by15/8. Boundary-recall deltas R−Base are−.004550/−.002569 and G−Base−.001517/−.001028: passing this individual tolerance does not mean boundaries are unharmed. Native macro F1 R is.965141/.967508 versus Base.977557/.975832; G.979593/.970803. Original non-A4C labels are bitwise preserved **by the deployment rule**, not by learned protection.

Independent per-case R/G comparison:

| Outcome | Shared | R only | G only |
|---|---:|---:|---:|
| A5C fix | 3 | 11 | 7 |
| Correct-A4C harm | 8 | 42 | 14 |

There are75 run-label disagreements. R adds11 fixes but loses7 G fixes; it adds42 harms while avoiding14 G harms. Its net difference from G is−24. Equal aggregate counts in earlier rounds are not evidence of equal cases. R's14 fixes cover11 true-A5C source recordings across the session-isolated folds; G's10 cover7. Recording/session counts are not patient counts.

**There is a limited ranking signal.** Target-A5C versus protected-A4C AUROC is R.820728/.744065 and G.713503/.719807. Both R values are higher. Thus this named localization-trained state carries some descriptive transferable ordering signal under this screen; it is incorrect to summarize A48 as proving “no added information.” It still fails to turn that signal into safe corrections under the predeclared rule. This is not significance testing, proof of AV/LVOT localization, or evidence that a different threshold would satisfy the full package.

The R absolute package passes target recovery/source diversity but fails protection (50 harms; at most14), net, complete F1, native preservation and short-segment gates. G fails target recovery (10; at least11), protection and the other joint requirements. R−G fails every predeclared contribution requirement: not more fixes in each fold, only+4 OOF fixes, more harms,−24net and no F1 advantage in both folds. Both absolute gates are false, so **source2 is closed without training it**.

## Independent verification and provenance

Source/code153 actual git objects were verified before the first source-backed forward. b63250b is the initial implementation;07a9013 records deterministic equivalent pooling;bae7068 commits passed production checks. Production two-pass versus retained-graph gradient relative error6.2434e−8, formula error1.7373e−7 and coefficient error1.4001e−8; repeated6, forced-equal source paths and3+3 resume are bitwise identical. All four official source/transform maps match exactly, peak allocated GPU4.505GB.

The complete model/head/Base/source/config/code/input/plan/check/optimizer manifests are actual git objects in **d3e1725be0a93472d7681157200e82eacd4ddf1a**; root verified157 bound files. Explicit head files are derived from the completed full checkpoints without a forward or training. Their bytes/tensors were independently matched. Freeze was06:14:37UTC, and once opposite-fold readout began06:14:51UTC, after the manifest commit at06:14:07UTC. Each arm/fold window-score cache was saved before evaluator work. No complete inference was repeated.

The final independent auditor reconstructs all original marker references and session isolation, expands all1520 eligible memberships for both arms, computes FP64 duration means and r>0 labels independently, recounts true source recordings, rebuilds complete sequences/F1/native/boundaries/short misses, recomputes every gate and exact fix/harm overlaps, and checks all4×600 optimizer steps/actual occurrence, weight and slot hashes, every selected tensor's gradients/drift and frozen/old-Base state. Fresh fixed first/middle/last window subsets use independent normalization/resize/four-scale pooling and a functional head. Maximum fresh-score difference1.026e−5 (<1e−4), final functional-head gradient errors≤1.86e−16. Independent results exactly match readout aggregates and private cache SHA49350becc6351e9a6ae75144bf15755fe3a699d9701d70268ca7169eeccfc370.

Two engineering failures are retained and charged. The first production backward failed because CUDA adaptive_avg_pool2d backward lacks strict determinism; separable normalized indicator matrices implement identical floor/ceil means, with FP64 value/gradient parity1e−12, while keeping strict determinism. The first audit and its diagnostic rerun stopped because the independent logaddexp reference omitted PyTorch softplus's input>20 linear branch. Head outputs matched exactly, but the loss differed6.39e−10, violating the original1e−10 relative gradient gate. The audit-only6717f34 correction and boundary/extreme-input tests reproduce that branch without relaxing the gradient threshold; deviation from mathematical softplus remains bounded4.2e−9 for the whole loss. The first clamp-at-boundary unit implementation also failed and was corrected before GPU audit. **Training/weights/once-readout remain unchanged.** The original frozen audit identity remains in d3e1725; the corrected auditor's actual git identity/SHA is separate. Only fixed label-free parity subsets were rerun; failed logs remain server-side.

See preparation_s0.json, sources.json, checks_s0.json, weights_s0.json, frozen_s0.json, readout_s0.json, independent_audit_s0.json, verification.json and resources.json. Private media, identifiers, detailed scores and per-case outputs remain on the server. The original eight staged theory files and all unrelated dirty files are preserved; no push.

## Scope and stopping decision

This is a repeated training-source session-fold exploratory screen on constructed streams, not a new untouched test set or clinical continuous scanning. It does not disprove localization supervision or pretrained representations. It closes this released AV-box state + live layer4/full-FPN + fixed whole-run repair recipe at source0.

The most specific unresolved supervision gap is **target-relevant local contrast**: the source teaches where an aortic valve is in PLAX/PSAX/3CH, rather than validating outflow continuity/visibility versus confusing correct A4C within these repair intervals. That is a next-evidence hypothesis, not a demonstrated unique cause. R's ranking gain warrants keeping the source result; its substantial A4C harm rules out claiming the objective was solved. A future direction must demonstrate new target-relevant supervision/evidence and legal access before another run. This round does not substitute a new loss, confidence gate or post-hoc threshold for that evidence.

Resource accounting and terminal scope are in resources.json, with the final measured bookkeeping tail in [the runtime receipt](../../../runs/video_valve_localized_a5c_v1/final_resource_receipt.json). The main-worktree focused suite passed22 tests after integration. All source inspection, qualification failures, actual formal/check/audit/readout stages, process bootstrap, root/reviewer proxies and actual closing work are included; empty monitor waiting is excluded. No source2, no running A48 model process, no automatic A49.
