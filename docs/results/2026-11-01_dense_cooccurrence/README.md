# A54 dense self-supervised local co-occurrence source screen

Completed and independently audited. This source-only pilot is not an EV9V recovery, correct-A4C protection or full-segmentation result. A08 remains unchanged.

## Question and changed representation

A53's precise-label global512 supplement failed even on fit specificity. IMG-D1 had shown image-interior sensitivity in both true and harmful A5C decisions, without anatomical identification; A51/A52's global video and point-detail source heads failed source safety. A54 tests a different representation: official general-image self-supervised DINOv2-S/14 normalized patch features, rather than another loss, threshold, gate or A08 token pool. It learns two 913-parameter CPU readout heads; the 21M-parameter external backbone is completely frozen. These are not full-backbone fine-tuning runs and there is no claim of LVOT recognition.

Exactly the original A51 verified569fit/133diagnostic manually labelled marker frames and36/10patient allocation are retained. Only marker slot8 of each sealed original16-frame224crop is used, with ImageNet normalization appropriate to DINO, replacing A08's EchoPrime normalization. This is not a comparison to Helena's preprocessing. No new source media, replacement cases, relabelled UNKNOWNs or expert labels were obtained. The diagnostic patients have been repeatedly inspected; source efficacy is exploratory, not fresh clinical validation. Generic DINO pretraining overlap is unknown.

C and S both share the same384-dimensional patch mean and a fixed orthonormal384→32 projection. C retains local projected-channel covariance; S independently circularly shifts each channel over256patches before covariance, preserving each marginal, mean and diagonal variance while disrupting cross-channel patch associations. Each reads the same912-dimensional feature length with the same913trainable parameters, initialization, fit-only scaling and original600×8exposure plan. This contrast concerns channel co-occurrence; it does not establish anatomical location or organized tissue shape. Both use mean BCE, AdamWlr.001/wd.0001/warm60cos600/clip1. No deployment rule was trained.

## Source diagnostic outcome

| Frozen representation readout | A5C correct /39 | A4C false positives /94 | Balanced accuracy | AUC |
|---|---:|---:|---:|---:|
| Historical original A08G |16|4|.683852|.807420|
| C |13|13|0.597518|0.647027|
| S |12|21|0.542144|0.598200|

C absolute source gate: **False**. C−S association gate: **False**. Combined source qualification: **False**. Advancement requires both, never absolute alone. Absolute requirements were BA≥.80, AUC≥max(.85,originalG+.03),≤5/94A4Cerrors,≥28/39A5Ccorrect, and BA≥the original fixed A51 cached geometry+.10. Separate association requires≥.03BA,≥4extraA5Ccorrect and no additionalA4Cerrors. Initial20floor was prospectively raised to28before intake/anyforward, preserving the earlier.70recall requirement. Prior A51/A52/A53 gates and negative outcomes remain unchanged.

Exact paired source decisions: {"correct_A5C": {"shared": 12, "C_only": 1, "S_only": 0}, "A4C_errors": {"shared": 13, "C_only": 0, "S_only": 8}}. Equal counts are not presumed equal examples. No new target103/1413/F1 score was read.

## Execution and independent checks

Official [DINOv2 repository](https://github.com/facebookresearch/dinov2) README, LICENSE and model card were fetched at the pinned commit before the official weight download. The captured README/LICENSE state Apache2 for code/model weights; this is a recorded source contract, not a clinical or legal guarantee. Exact metadata/archive/weight/code digests are in source.json; private weights were strictly loaded with no missing keys. All images, IDs, tokens, model tensors and per-row scores remain server-only.

Upstream commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`; complete weight SHA256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`. Actual source/input/code Git3729dd6 preceded four fixedFITchecks and702frozen neural source forwards. These include133diagnostic **backbone feature extractions** before head fitting, with no fitted normalizer or classifier scoring; they are counted as real model inference. IdenticalFITrepeat/strictload/frozenparameters pass.

A code review arrived after extraction had exited successfully. Its real durability gap is preserved rather than erased: original per-row mmap.flush lacked explicit fsync and NaN initialization. No crash, ambiguous pending row or repeat occurred. The complete702×256×384tensor was independently checked finite with done702/pendingNone, then fully fsynced. Original extractor code/input freeze is retained in execution_history and execution_patch.json. This round does not claim those original per-row fsync barriers existed. The completed cache is additionally verified by exact SHA and independent all702descriptor/scaler reconstruction. No encoder was rerun to hide the issue.

Before any formal head update, optimizer checkpoint/journal fsync and complete readout identity checks were corrected and pinned in9e403bf. The original source/feature manifests remain immutable; a separately committed execution patch links old and corrected code, without changes to pixels, projection, plan, labels, source weights or gates. Both600-stepheads and optimizer moments/journals were independently replayed with NumPy on fit data only. All1200real updates complete; disk3+3resume, forced-identical and independentBCEgradient checks pass. Bothheads, allSHAchains and independent audit actualGitc23f19c precede the headsfreeze and only266source classifier scores. The readout verifies head/feature bytes, source/feature manifest chain and current matching independent-head audit before arithmetic.

The final cache-only auditor reconstructs all702C/S descriptors using independent direct token arithmetic, recomputes fit-only scaling, repeats all1200AdamWupdates and complete optimizer states, then algebraically reconstructs the266diagnostic scores, manual-label counts/AUC, exact paired sets and both gates. It performs zero fresh neural forwards. Two independent replay passes total2400replayed optimizer steps/19200fit logits; descriptive fit-score checks add2276fit logits; computational preflight adds18headupdates/152fitlogits. Formal training has9600fitlogits. These are accurately distinguished from the266production and266independent diagnostic scores.

Raw diagnostic cache SHA256 `a1f2c5f880879521326812072bc95e60337ea738e2b5011b4cd7d9c02fbe300e`. Complete private-head hashes are preserved in heads_manifest.json, head freeze actualGit `c23f19c6fb81147a34710e91cb2d5023ac1d2c18` at epoch1793592717.7123706. Independent final audit passed.

## Decision and closure

No source2, loss/cutoff rescue, target model or target readout was launched. If qualification passes, it establishes only a reason to review a future bounded target card; if it fails, this exact source configuration closes. Neither outcome proves or disproves all self-supervised patch representations. No current result exceeds A08 or resolves target A4C/A5C protection.

All A51/A52/POST/A53 inherited charges remain in the same finite CPU6upperh/GPU4jobwallh/newpeak5.5GB ledger. Source intake, engineering gap checks, bootstrap, extraction, training, both audits, reviewers/root and actual closing are charged, between-turn idle excluded. See resources.json and runtime final_resource_receipt.json for actual cutoffs. No future costs are precharged, no budget reset. No protected project dev/check/test/Pediatric/fetal/A26runner/Unitytraining, other-owner files, external messages, pushes or website changes.

Artifacts: source.json, input_freeze.json, execution_patch.json/execution_history, features_manifest.json, features_freeze.json, checks.json, heads_manifest.json, independent_heads.json, heads_freeze.json, readout.json, independent_final.json, resources.json, verification.json; [fixed card](../../superpowers/plans/2026-11-01-dense-selfsupervised-cooccurrence.md).
