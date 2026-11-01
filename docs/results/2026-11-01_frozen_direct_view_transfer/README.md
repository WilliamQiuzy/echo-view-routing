# A53: frozen direct-view representation transfer

Completed and independently audited **negative**. A08 remains the working reference. The precise-label L representation did not improve aggregate recovery/protection or mean complete F1 over either matched-source W or original-feature G readout. Individual fold comparisons vary.

## Question and substantive change

A51's precise manual source labels improved limited same-recording score ranking, but its source classifier failed A4C protection. A52's finer spatial detail also failed. POST-A52-D independently found L same-video AUC .88034 versus W .70940 and original G .71795 on 117 correlated pairs in five videos. That observation motivated discarding the unsafe source classifiers and testing their **frozen 512-dimensional features**. It did not establish anatomical evidence or target transfer. A53 changes the transferred representation and resets the classifier; it does not revise A51/A52's failed gates.

G is [original A08 G512, zero512]; L is [G512, frozen A51 precise-label L512]; W is [G512, frozen A51 directory-label W512]. All trunks, marker encoders, projections and old heads stay frozen. Six new FP64 linear heads completed 600 CPU updates each under the same original A44-C plans. G has 513 active parameters and 512 structural zeros; L/W have 1025 active parameters. L versus G has a capacity difference; L versus W is the matched source-label comparison. **These are six readout heads, not six full-backbone training runs.**

Original 16-frame requests, A08 crop/marker/gamma/normalization, all duration and repeated-request memberships remain intact. Eligible rows 5030/5437 come only from the original Base mapping. Fit lists are 34/69 positive A5C runs, 714/699 correct A4C negatives, and 1/3 other eligible runs excluded from training but retained in complete readout. Scalers use only plan-referenced fit rows. The original source0 recording/session folds have been used repeatedly; this is not fresh patient-held-out clinical evidence.

## Complete opposite-fold result

Fixes and harms are changes relative to the unchanged original A08 labels. G below is a newly trained head on A08 features, not the original A08 classifier. F1 is the mean complete-stream segment F1@0.5 across the two original folds, not a pooled patient estimate.

| Arm | A5C found /103 | Correct A4C harmed /1413 | Net fixes minus harms | Complete F1 mean | Short segments missed | Deleted original cuts |
|---|---:|---:|---:|---:|---:|---:|
| Original A08 Base | 0 | 0 | 0 | .956885 | 76 | 0 |
| G: original features, reset head | 67 | 230 | -163 | .940271 | 149 | 209 |
| L: G + precise-label features | 60 | 231 | -171 | .939075 | 145 | 191 |
| W: G + directory-label features | 63 | 231 | -168 | .939526 | 146 | 200 |

| Held-out fold | Arm | A5C fixes | A4C harms | Complete F1 | Native macro F1 | Boundary recall | Short misses |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | Base | 0 | 0 | 0.963906582 | 0.977556578 | 0.967644085 | 41 |
| 0 | G | 23 | 101 | 0.946257573 | 0.951893569 | 0.950960566 | 81 |
| 0 | L | 17 | 101 | 0.944368534 | 0.950256048 | 0.952477250 | 81 |
| 0 | W | 19 | 100 | 0.944346807 | 0.951893569 | 0.951971689 | 82 |
| 1 | Base | 0 | 0 | 0.949863972 | 0.975831638 | 0.968139774 | 35 |
| 1 | G | 44 | 129 | 0.934283452 | 0.922913521 | 0.948612539 | 68 |
| 1 | L | 43 | 130 | 0.933781380 | 0.920098512 | 0.950668037 | 64 |
| 1 | W | 44 | 131 | 0.934705184 | 0.920098512 | 0.949640288 | 64 |

All arms pass only the A5C recovery and structural-label-copy components. All fail correct-A4C harm, net gain, complete F1, native recognition, boundary and short-segment components. The absolute harm cap was 14/1413; observed harms were 230–231. The separate L−G and L−W contribution gates also fail. Non-A4C labels being copied exactly is a deployment rule, not learned protection; RLE still deletes true boundaries and short segments. Fewer cuts are not success.

Exact paired sets were independently reconstructed. L/W share 59 fixes, with 1 L-only and 4 W-only; they share 221 harms, with 10 unique harms each. L/G share 59 fixes, with 1 L-only and 8 G-only; they share 212 harms, with 19 L-only and 18 G-only. Equal aggregate harm counts do not imply equal examples. L fixes span 31 original true-A5C recordings; G 32 and W 31, not confirmed independent patients.

## What the learning check shows

Post-readout, fit-only arithmetic was examined without changing any head, cutoff, feature or gate. The positive first and three-negative plan labels were verified explicitly for every step; the plan ordering is correct. Fit AUCs are .934–.953, yet all six heads already harm 91–101 correct A4C fit runs at the predeclared zero cutoff. L fit0 finds 32/34 while harming 96/714; fit1 finds 59/69 while harming 94/699. Thus the protection problem is already present in the fitted operating rule; it cannot be attributed solely to held-out overfitting. High ranking AUC and falling batch loss do not establish safe A4C specificity. This descriptive check does not uniquely distinguish feature overlap, noisy run labels or objective/operating-point mismatch and is not permission to choose a new threshold after seeing outcomes.

A53 provides no evidence that this frozen precise-label supplement supplies the missing target-specific discriminant at the tested readout. It does not disprove all representations, all manual supervision, or all temporal supervision. Stronger supervision must be visibly bound to the relevant image content and qualified before another model is justified.

## Execution, freeze order and independent verification

- Actual extractor/input/code Git object 7aeb30dc08bde7fe3e8977258e56fc05502f5b57 precedes frozen L/W extraction. Formal extraction performs 20934 window forwards; fit checks add 60 GPU forwards and four independent CPU forwards. Held-out covariates were encoded before head fitting without labels or fitted scalers; this is real held-out backbone inference and is counted.
- Feature/scaler manifests were committed as 525e4db39aacbb3f0aa85a4bee0922b5204b1170 before head fitting. Full membership multiplicities and fit-only scalers were independently recounted. Fresh fit embedding errors are at most 2.51e-6; original G parity error 1.68e-6. Explicit per-window versus pooled linear gradients agree within 1.34e-15; repeat, forced-identical-feature and disk 3+3 resume checks pass.
- All six heads, optimizer states, 600-step journals, source/Base/config/code/input/plan/scaler hashes and the complete manifest were committed as 845981ed99d8e77c4cbb965aca990b4a71725923. Freeze epoch 1793572830.025024 precedes production scoring start 1793572833.119332. All 171 referenced Git objects were verified, with zero new opposite-fold classifier scores before freeze.
- A separate NumPy AdamW implementation replayed all 3600 updates and final states; maximum head error 4.45e-16. Backbones have zero optimizer updates. Formal training has 14400 pooled-run logits and 3600 backward batches.
- One production readout produces 31401 window scores. The prospectively declared independent audit reconstructs the same 31401 logits algebraically from sealed cached features and frozen heads, without fresh neural backbone forwards. It independently recovers repeated nearest-frame memberships, duration means, one-way labels, source counts, paired sets, original-reference evaluator F1/native/boundary/short counts, and all gates. This second arithmetic pass is not reported as zero classifier computation. The later fit-only descriptive check adds 4548 pooled-run score reconstructions.
- Raw private readout SHA256: d061f27ae3aee1ae792205cb97941f4daa0a56498333a40e7e2a58fecee330e0. Original Base SHA256: 9b144925f32fac7d725d13231a132abeb6635fa44b6fe0dbcec7f5b59d035bbc. Exact six head and source hashes remain in the manifests; media, IDs, tensors, weights and per-case outputs stay on the server.

Engineering failures before formal extraction remain in prelaunch history and resource receipts: JSON tuple/list identity comparison, a stale-code check blocked before any encoder forward, and local compile/test repairs. Scientific inputs, plans, stopping gates and formal extraction were not changed to rescue outcomes.

## Closed scope and next evidence boundary

No source2, deployment, threshold rescue, alternate loss, seed expansion or next model was launched. A08 remains unchanged. The parallel cache-only dense-label audit qualified only 38 fit /15 diagnostic informative requests, below fixed 80/20; it launched zero models. The bounded public-source refresh did not establish a new reusable adult Bmode LVOT image-supervision contract. EchoXFlow issue4 still had no maintainer comment at the recorded query. These are scoped source-qualification results, not proof that no such resource exists.

All A51/A52 inherited costs, POST audit, source searches, failed preparation/checks, full child bootstrap, frozen extraction, training, independent replay, root/reviewer active work and closing share the same finite CPU6h/GPU4h/new-peak5.5GB phase; no budget reset. See resources.json and the final runtime receipt for measured totals and cutoffs. No protected data, teammate files or website were changed; no push or external contact was made.

Artifacts: [fixed card](../../superpowers/plans/2026-11-01-frozen-direct-view-evidence-transfer.md), input_manifest.json, checks.json, structural_checks.json, extract_freeze.json, feature_manifest.json, fit_design_manifest.json, independent_features.json, head_train_freeze.json, heads_manifest.json, complete_heads_freeze_manifest.json, independent_heads.json, heads_freeze.json, raw_complete.json, readout.json, independent_final.json, posthoc_fit_learning.json, verification.json and resources.json.
