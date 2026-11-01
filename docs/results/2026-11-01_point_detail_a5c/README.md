# A52 direct point-frame detail representation

**Completed and independently audited, negative. A08+B3 remains the working reference.** Both source models completed600real updates. Neither source protection nor detail-increment condition passed; no target model or target103/1413/fullF1readout was started.

This round trains every convolution/FPN parameter on the same qualified EchoUNAL marked-frame labels as A51, while removing unlabelled16-frame context from the supervised input. H sees the full-field448image; R sees the same canvas reduced224and enlarged448. Both use identical ResNet50FPN/1025-parameter scalar head, true-class-balanced569image600×8plan, initial COCO state and BCE/optimizer/compute. No crop detector, loss change or threshold optimization.

A36 tested frozen A08video features with residual heads, A48 tested adapted valve-box source state/layer4, and PW-T1-C used weak19-pair purpose supervision. This is a distinct direct point-supervision/all-body representation hypothesis, not a rerun or a claim that an LVOT anatomical feature has been learned. H−R isolates the detail recipe; a common gain versus A08 would be a package gain. Conditional target H−R changes detail in source learning, target learning and inference, not source pretraining alone.

Exact fixed source702points,569fit/133diagnostic,36/10patients, raw/H/R each702unique; zero cross-patient/partition duplicates/conflicting labels. All68selectedAVI SHA/full nativePTS and all702raw→H/R/slot8point bindings independently recomputed. Original707qualification failed5FITmarkers and is not reopened. Source133diagnostic images were already used by A51, so this is exploratory repeated-source evaluation. No anatomy truth, original-publisher-test replication or new clinical holdout claim.

Current fit-only checks passed independent BCE/bias gradients, nonzero-head full/microCNNgradient error2.98e−8, every71trainable tensor finite/nonzero gradients, zero-head startup, all7groups later nonzero, frozen BNbuffers, repeat/forced-identicalinput/disk3+3resume exact. Complete source code/input/checks actual Git81946aa precedes prelaunchfreeze. Original A08 remains unchanged. The prepare-path hashing failure retained allfixedinputs/no model, charged83.91CPU seconds; manifest seal repaired metadata bookkeeping with independent unchanged-input verification.

Source absolute A4C protection/BA/AUC/geometry criteria and separate H−Rdetail criterion fixed before A52 outcomes. Only an absolute pass permits bothtargetarms on unchanged A44-C600plans, if complete remaining shared resources fit. Otherwise no target103/1413orfullF1readout and no rescue variant. See card `docs/superpowers/plans/2026-11-01-point-detail-supervision.md`, input/checks/independent_binding/prelaunch_projection/review/freeze JSON.


## Executed result and independent verification

| Source model | A5C correct /39 | A4C false positives /94 | Balanced accuracy | AUC | Training correct /569 |
|---|---:|---:|---:|---:|---:|
| Historical original A08 G, A51once cache |16|4|.683852|.807420|not refit|
| H full-field448detail |27|24|.718494|.745772|565|
| R reduced224→448detail |23|21|.683170|.757229|565|
| New point-only geometry |20|19|.655346|.712220|fit-onlycontrol|
| Fit-prior constant |0|0|.500000|.500000|notmodel|

These are **public-source point classification counts**, not target103A5C recovery/1413correctA4C harm or a segmentation result. The133diagnostic rows are reused exploratory source data,10patients. Threshold is strict0; no threshold tuning, source re-selection or best-checkpoint choice.

Paired source comparisons, computed from saved raw scores:

- H/G A5C11shared,16H-only,5G-only; A4C errors0shared,24H-only,4G-only. Thus H newly corrects16previously missed A5C points but loses5previously correct ones; newly misclassifies24previously correct A4C points while fixing4old Gerrors. Net total correct−9, not safe recovery.
- R/G A5C11shared,12R-only,5G-only; A4C errors0shared,21R-only,4G-only. Net total correct−10.
- H/R A5C23shared,4H-only,0R-only; A4C errors20shared,4H-only,1R-only. More detail gives4extraA5C correct, creates4differentA4C mistakes and avoids1; net correct+1, AUC lower by.011457. This is not a protected detail benefit or proof that anatomy was recognized.

Fit BCE H.020065/R.016688, both565/569fit correct. Strong fit learning remains insufficient for source generalization. A51precise video model had31/39A5C and23/94A4C false positives; A52does not improve that trade either. These cross-round whole-system comparisons differ in architecture/input, while H−Ris matched. The experiments narrow the usable evidence: direct labels and extra image detail in this limited source do not establish a safe deployable discriminator. They do not prove that no LVOT cue exists, nor identify which anatomical or spurious visual feature causes the errors.

Checks/independent audits: both600-stepjournals,71live tensors each with nonzero gradient checks/drift/nonzero final optimizer second moments, every optimizer state at600, exact groups/hypers/final cosineLR/finite float32moments/nonnegative second moments, unchanged FrozenBN and original COCO/A08base digests. Full model/optimizer/RNG retained. CPU fresh independent torchvisionFPN+head evaluation on4fixed FIT inputs agrees with production GPUfitcache within1.53e−5;0additional diagnostic forwards. Source GPUtraining job276.106secwall/268.201CPU. Actual full-model/head/base/code/input/plan provenance actualGit **dd669dec16d13186d39386d104c395b43262045c** before freeze and the only266diagnosticforwards. Per-row complete/progress/rawscorecellSHA and order verified. Independent cache-only metrics/AUC/geometry/constant/paired changes/bothsourcegates match exactly.

Raw-score-cache SHA256 `376225fb0735c50b345a4ad7726f83989a34d6925f1cc394d89df7bc69ce0a33`. Models manifest `d236c2ea78c5f4ee2e971bef7f54f615a0a8c6be2c4929d4e6c52fcd36f34e25`; allfullweights/head/optimizer/base/code/inputSHA in `complete_freeze_manifest.json`. Complete models remain server-only. Source readout16.391secwall/14.985CPU. The saved133historicalG scores from the A51 three-classifier399forwardreadout were reused without rerunning G.

## Decision and scope

Both H/R fail source BA.80, A5C recall.70, originalG A4C FPR+.02, AUCfloor/increment and geometryBA+.10. H−R fails allthree detail conditions. **Target transfer is untrained/unread, not a measured target learning failure or a new no-opF1 result.** No source2, extra seed, replacement labels/frames, threshold rescue or automatic next version. Working A08 model/decoder/predictions unchanged.

A51+A52 comprise four real600step source models/2400formalupdates, not target models. Current goal—exceeding A08 while recovering A5C safely—remains unmet. Candidate next evidence includes domain-relevant matched hard A4C negatives and more localized supervision for a transferable discriminative cue; this comparison did not establish anatomical necessity or isolate the feature causing errors. Another loss, cutoff or larger image alone has no supporting result here. Expert-free public supervision was actually tried, not assumed unavailable, but this fixed source did not supply a qualified safe transfer signal. Any future acquisition or training needs a distinct prospective evidence contract; these outcomes are not permission to tune against the reused diagnostic cohort.

`resources.json` records the same finite A51/A52phase allowance and inherited ledger, actual failures/checks/training/audits/bootstrap/root/reviewer/closing; no reset. No protected project data/other-owner task/website was changed. Public aggregates and code are archived on Mac/server; private media/IDs/scores/weights remain server-only. See `verification.json` for final SHA/process/Git/preservation closure.
