# PW-T1 — clinical-purpose representation transfer qualification

**Closed before formal target training at the shared remaining-resource gate.** This is not a negative learning result. There are0formal target models,0target600-update runs and0opposite-fold score reads. No new A5C fix/harm or complete-segmentation result exists.

## Why this question was different

The [image-dependence diagnostic](../2026-10-31_a5c_image_dependence/README.md) supplies no convincing shortcut-cleaning fix. The already completed [PW-P1](../2026-10-30_pw_small_pilot/README.md) instead added new operator-recorded clinical-purpose image supervision: correct-purpose L had10/12heldout source images correct/AUROC.944444, versus same-image shuffled-purpose S6/12/.333333. That is a small exploratory source-task signal, not visible-LVOT truth or evidence of A5C transfer.

The prospectively reviewed [PW-T1 card](../../superpowers/plans/2026-10-31-pw-purpose-transfer.md) tests whether this **learned upstream representation** adds A5C/A4C discrimination beyond **both** S and original COCO G. The G arm matters because L>S alone could reflect damage from shuffled-label pretraining. It does not introduce another repair loss, score gate, threshold, nuisance mask or source pool. A50-PW's25/80formal source quantity failure remains unchanged; its fixed pool was not widened, and PW-P1 was not retrained.

## Fixed proposed comparison and completed preparation

G/S/L each use the exact PW-P1 ResNet50-FPN four-scale global-mean architecture, fresh zero-weight/bias−2target head, live layer4/all FPN/head and frozen early body/FrozenBN. Original A44-C run-balanced loss,600update plans,optimizer/schedule/FP32 and genuine two-pass per-occurrence forward/backward remain fixed. All original marked frames use the original short256/centre224RGB crop followed by ImageNet normalization; no feature-cache training substitute.

Two folds require six complete target models,90918training occurrences,181836real training forwards/90918backwards,62802full-fit Init/final marker scores,31401once-opposite-fold scores. Fit denominators remain34/69positive and714/699protected negative runs;1/3other eligible runs remain in complete readout. Marker pixels, all membership/weights and original plan hashes were prepared and bound before any target forward. This target comparison would reuse historically inspected training-source folds, not a fresh clinical test.

Complete models were specified as shared immutable G plus **every** mutable layer4/FPN/head tensor, with exact frozen-state and canonical full-state SHA reconstruction. Six redundant copies of unchanged weights were not required; a head-only model would be invalid. No final model bundle was produced.

## Fit-only production checks actually run

-18real G-initialized fit-only optimizer updates:6original,6forced-identical-source repeat,3+3compact-save/resume. The final states are bitwise equal.
-Live two-pass production gradients versus a retained graph with nonzero head: relative maximum error5.780708365690849e-8. All selected tensors see nonzero gradients after the expected zero-head structural first step.
-Init scores−2on the12fixed fit rows; fixed first/middle/last original marker pixels and independently calculated ImageNet normalization agree. No heldout score or checkpoint selection.
-Four steady update timings cover18–27occurrences:0.3519–0.5256sec, upper0.0198804sec/occurrence. Measured inference0.00817353sec/row;73150468mutable tensor bytes/model; compact checkpoint write0.1709sec. These are real model timings, not CLI liveness.

An initial metadata attempt lacked the historical public source manifest on the server; its exact local committed artifact was copied. The first fit-check attempt stopped before any model forward/update because the driver called the existing learning-rate API with the wrong argument shape. It was repaired to the unchanged60/600schedule and covered by a synthetic production-step regression. All failures/bootstrap remain charged. Details: `engineering_recovery.json`. A harmless read-only NumPy conversion warning occurred; the input is copied to the device and normalized without in-place writes.

## Whole-comparison launch decision

`launch_projection.json` records the prospective arithmetic; forecasts are **not** charged as executed work.

| Quantity at launch decision | Value |
|---|---:|
| New GPU-job wall already spent, all diagnostics/checks/failure |20.3642min|
| New GPU-job allowance remaining |39.6358min|
| Complete six-model/full-fit/readout forecast,20%margin+6minverification reserve |59.7137min|
| New active CPU upper already spent, owner/reviewer/children |2.0614h|
| Projected further CPU upper, including.25hclosing+.15howner reserve |1.2950h|
| New CPU upper cap |3h|
| Projected six-model retained artifacts, including diagnostic/tool allowance |498.9MB|
| Projected cumulative historical+new peak |2.937GB|

Storage fits; GPU and CPU do not. Even the training plus fit/readout timing extrapolation before bootstrap/margins/reserve exceeds the remaining GPU allowance. The exact card therefore stops the **entire** formal comparison. No G arm/fold/member/step was dropped, precision changed, head substituted or partial target result reported. The unexecuted train/readout code is retained as a prospective implementation, not a validated end-to-end training pipeline or an authorized launch after closure.

This stop leaves the representation-transfer hypothesis **unresolved**. It does not justify another weak-label/loss/gate variant or promoting PW-P1's source-task accuracy as A5C success. The completed contribution of this work is the independently verified image-dependence diagnosis and the measured, scoped transfer-qualification record.

Input/code actual git freezes550072eand scheduler-repair2d3c29bprecede the corresponding checks. `input_freeze.json`, `input_manifest.json`, `checks.json`, actual process receipts, `launch_projection.json` and final `verification.json`/`resources.json` preserve the executed scope. Original source L/S/G and A08 weights remain unchanged; all private images, IDs, model tensors and per-case data remain on the server. No source2, external-test evaluation, deployment, website modification or push.
