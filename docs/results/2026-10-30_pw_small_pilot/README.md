# PW-P1 — small-sample clinical-purpose learning diagnostic

**Completed, independently checked; an exploratory positive source-task signal.** Correct-purpose L and within-exam shuffled-label S each completed600 real SGD updates. The original A50-PW80/64/16 formal source-quantity gate remains failed and unchanged. This diagnostic has no EV9V readout, A5C fix/harm or complete-segmentation result.

The user challenged why no training was started. The original quantity threshold was a conservative formal-protocol decision, not a mathematical inability to train. This separate, explicitly small-sample diagnostic tests actual learnability with the25already qualified pairs; it does not retroactively pass the old card or add source examples.

## Changed condition and fixed comparison

New raw source images and operator-recorded acquisition-purpose labels (LVOT PW versus mitral-inflow PW) were untrained in A50-PW. Both arms use the same COCO_V1 ResNet50-FPN initialization, live layer4/all FPN, frozen early body/FrozenBN, deterministic224 inputs,1025-parameter zero binary head, optimizer and600four-image batches. S swaps labels in exactly9of19fit exams while preserving both classes within each exam. No cached-feature training, detector output, new loss, overlay/gate/spectral/model-predicted truth or deployment veto is used.

Data:19fit examinations/38images (provider folds1–4),6heldout examinations/12images (fold0). Each examination contributes one image of each purpose. Provider patient grouping retained; independent patient IDs/counts unavailable. These are weak clinical-purpose targets, not expert LVOT visibility, A5C or A4C labels.

## Once-heldout source result

| Arm/control | Correct /12 | AUROC | Within-exam ranking points /6 |
|---|---:|---:|---:|
| Correct-purpose L | 10/12 | 0.944444 | 6/6 |
| Shuffled-purpose S | 6/12 | 0.333333 | 2/6 |
| Geometry-only ridge | 4/12 | 0.375000 | 2.5/6 |
| Constant | 6/12 | 0.500000 | 3/6 |

L adds5correct heldout images and loses1that S gets correct (shared5). Both arms fit their own training labels38/38; S true-purpose fit accuracy20/38 reflects its deliberate label flips, not failure to optimize its assigned task. L ranks the LVOT-purpose image above its paired mitral-purpose image in all6heldout exams; S does so2/6. Geometry ties receive half credit.

**Interpretation:** L passes the predeclared descriptive signposts: AUROC>=.75, >=.10over S/geometry, >=8/12correct and>=4/6pair-order points. This supports some image-dependent clinical-purpose information under this specific split/seed. Six heldout exams cannot establish a stable population effect, local anatomical recognition, clinical generalization or protection of correct A4C. Scanner/preset/image-acquisition cues beyond the six geometry fields remain possible. No significance or independent-patient claim is made.

The source result warrants keeping this supervision hypothesis. It does not establish transfer to the original103A5C/1413protected-A4C repair problem. This card unconditionally stops before EV9V; no source2, seed/loss/threshold rescue or automatic next version. A target-domain transfer comparison would need its own prospectively specified matched card and sufficient remaining resources.

## Execution and independent verification

All input/source/plan/control/code hashes are actual git objects in3ba5309 before any source-backed forward; both final model manifests are committed inaecbea9 before the once-only heldout source readout. Original source artifacts/cards are untouched. Corrected-source work is a separate diagnostic, not a rewrite of A50-PW.

Fit-only repeated2-update checks are bitwise equal. Each arm has600logged updates, identical input occurrences/initial state/plan; all28selected parameter tensors have nonzero gradients and change, while frozen tensors remain exact. Independent audit recounts data membership/shuffling/plan/all1200steps, full weights and frozen drift, geometry fit and every scalar metric. Functional head max-score error1.52e-6, fresh fixed fit-subset error0, independent FP64 head-gradient max error1.46e-17. Audit performs no additional heldout model forward. All per-case scores, features, raw images, IDs and models stay server-only.

See [card](../../superpowers/plans/2026-10-30-pw-small-pilot.md), input_freeze.json, models_manifest.json, readout.json, independent_audit.json, verification.json and resources.json. The network initialization is the already publisher-hash-verified COCO G backbone4ae0006f..., not A48 trained repair weights. Source images remain CC-BY-NC-SA4.0 noncommercial-research assets; native decoding/ownership checks were completed in A50-PW.

## Resources and closure

Actual matched training wall20.766sec; total source-check/train/readout/audit GPU-process wall upper0.010121h, measured process CPU36.720sec. These are small source-task models, not full video repair training. Current cumulative shared CPU upper2.758705h includes earlier source/contact/review/owner work, under the original3h cap; GPU6h/storage3GB caps remain. Historical new-storage peak upper2.219GB remains conservative, with this later diagnostic namespace <.5GB. Actual closing-tail accounting is in runs/pw_small_pilot_v1/final_resource_receipt.json. No waits/future compute are charged.

Only own files/index row are integrated; eight staged theory and other dirty/external-test work are preserved. No push or forbidden dataset reads. Own processes and temporary worktree are closed after verification; Issue4 watching remains independent and cannot create a second queue.
