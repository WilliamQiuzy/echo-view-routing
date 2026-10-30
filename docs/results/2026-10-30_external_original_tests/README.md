# A50 — Frozen A08 on obtainable original test sources

**Completed 2026-10-30. Evaluation only: no new training, seed selection, threshold tuning or test-driven model promotion.** All 2,470 admitted source sequences were evaluated for each of the three previously frozen A08 seeds. STFM was independently run on its original 888-cine native test. The aggregate result files and the independent recount govern exact values.

## Question and outcome

Does the working reference retain its performance on other models' original test data? The source audit covers 13 model groups. Only STFM's obtainable original cine-view test has a compatible shared five-class endpoint; it is EV9V, which this project already used. **This round obtains no new independent five-class or A4C/A5C cine benchmark.** The other executable public sources provide single-view source-label checks, with different original targets. Unavailable truth/pixels and incompatible tasks are recorded separately in `availability.json` and the webpage, never assigned a score.

| Shared original EV9V test, 800 cines / 125 sessions | Accuracy | Five-class macro-F1 |
|---|---:|---:|
| A08 + frozen B3, s0/s1/s2 score mean | 98.0417% | 95.3701% |
| STFM native 10-key rule, old frozen numeric-order s100 weights | 97.0000% | 96.5339% |
| STFM prior added once, same old weights | 96.8750% | 96.4849% |

A08 has higher accuracy and lower macro-F1. Against the native STFM rule, the paired session-bootstrap mean difference is **+1.0417 percentage points accuracy, 95% interval [-0.3163, +2.5050]**, and **-1.1639 points macro-F1, [-3.1129, +0.6530]**. Both intervals cross zero. There is no overall superiority claim. All three seed scores are shown; their average is not an ensemble or extra independent cases. STFM's full nine-code baseline-only accuracy is 94.4820% on 888; A08 has no nine-code output. These STFM weights were not retrained under the newer official recipe.

| Auxiliary official source test | N | A08 fixed-B3 three-seed mean | Descriptive 95% interval |
|---|---:|---:|---:|
| EchoNet-Dynamic, source A4C | 1,277 | 94.7794% | [93.6309%, 95.8235%] |
| EchoNet-LVH, source PLAX | 343 distinct recordings | 95.7240% | [93.9747%, 97.2789%] |
| CAMUS testing 4CH half-cycle, nominal 4CH→A4C | 50 patients | 73.3333% | [62.0000%, 84.0000%] |

These are source-code recall/agreement checks. They cannot be compared with EF MAE, wall-thickness regression, segmentation Dice, or full five-class accuracy. CAMUS 4CH may include A5C: an A5C prediction is not necessarily a clinical error. Its 0.18–0.51 s half-cycles also offer much less temporal support than ordinary A08 windows. Dynamic/LVH patient relationships and upstream pretraining overlap are unknown. Zero cuts on LVH/CAMUS do not establish boundary detection. Existing A26 native/pair/multi results are displayed separately as historical constructed-stream context, with their original seeds and denominators.

## Frozen method and data handling

- Working reference: A08 all-neighbour-smoothed MViTv2-S, saved all_s0/all_s1/all_s2 states, their original five-class heads and frozen B3 decoder. FP32 evaluation, TF32 off, batch 16, L40S memory fraction 0.6. No new model fit.
- Fixed output order: PLAX, PSAX, A4C, A5C, SC4C. Unsupported baseline predictions remain errors; the four raw PSAX probabilities are combined before shared-family classification. Per-recording labels receive one vote each, irrespective of duration.
- Use official test membership. EV9V retains the 800 compatible five-family cines from the original 888-code test; the 88 PMPALA cines are excluded only from the shared endpoint. STFM still scores all 888 for its nine-code endpoint. LVH repeats from measurement rows are deduplicated by recording. No failed case is dropped.
- Native numeric JPEG order / full video decoding / lossless CAMUS NIfTI uint8 conversion. Sample nearest source frames at 10 Hz without extending recording duration. Reuse original 16-frame padded windows, target-position marker, short-side 256 / centre 224 crop and EchoPrime normalization. Request centres every five 10-Hz samples; each request is a window prediction for a target time, not an external API call. Original endpoint windows and marker locations match the training evaluator exactly.
- Per-case private caches retain logits, centres, argmax and B3 labels for all seeds. Majority vote over the full predicted 10-Hz sequence yields each native classification. Macro-F1 is reported only on the complete five-class shared support. Full confusions, recalls, durations and internal cuts are retained in `native_results.json`.
- No Pediatric source was newly opened, trained on or exported. No cases/media/recording identifiers are hosted.

## Verification and prospective source repair

Protocol/config/weights were committed and hashed before target inference (`cfcf088`, `2b9b904`; STFM `3467dcf`). The first queue finished EV9V and Dynamic, then stopped at an LVH CSV/container rate mismatch **before any LVH forward**. All 343 videos were decoded with PTS; all were CFR and frame counts matched. A source-timing amendment (`5bf6d9c`) prospectively used the qualified 25–84 fps container rate for LVH, preserving the original CSV field. EV9V/Dynamic retain explicit v1 bindings; LVH/CAMUS use v2. No output, weight or decoder was selected after seeing LVH results. Both development-logit parity checks passed (maximum absolute error < 2.9e-6, identical argmax). See preparation/checks v1/v2 and `timing_amendment.json`.

An independently authored NumPy/standard-library auditor imported no production decoder or metric. It verified immutable config/code/weight/source-metadata hashes, complete membership, all 2,470 native caches, 103,122 sampled timepoints per seed, full independent label sequences and aggregate metrics, both STFM fusion conditions and the historical A26 native recount. It bootstrapped 2,000 grouped samples with fixed RNG: EV9V sessions, CAMUS patients, otherwise recordings. Paired differences use identical sampled membership for both systems and seeds. Final audit passed. Public source hashes are a count plus digest; individual source paths remain private.

Audit failures are retained: missing local input-audit receipt, then a 1.82e-12 duration summation discrepancy. Independent `math.fsum` resolved the numerical reference without relaxing the 1e-12 comparison; labels/scores were unchanged. The successful audit was rerun after removing source-path keys from public serialization. Four CPU audit attempts and the original GPU-queue failure remain charged in `resources.json`.

Limits: audit rechecks metadata hashes, source size/mtime, container timing and CAMUS headers; it does not reload all source pixels or establish raw-media byte hashes. Only declared modules are frozen, not every transitive environment dependency. Baseline caches lack per-case sampling receipts. These caveats are included in the downloadable verification. Repeated test use, weak clip labels, session-not-patient grouping and constructed transitions still constrain interpretation.

## Resources, presentation and closure

Three GPU jobs consumed **1,423.90 s (23.73 min) elapsed**, including the initial source failure, and **875.72 s (14.60 min) measured forward time**. Those clocks are not rental billing. Process CPU and all audit receipts, 80.66 s LVH PTS qualification and 4.92 s CAMUS pixel qualification are separately retained. Run files total 11,789,234 bytes, excluding existing datasets/weights and webpage copies. Agent/source-audit/development overhead is not fully metered; no total dollar charge is inferred. No new dataset download was required.

The [published owner-private report](https://echo-routing-external-tests-wq-20261030.qiuzy-william2019.chatgpt.site) is deployed successfully (see `publication.json`). The current aggregate-only local page is `demo/index.html`, built by `scripts/build_external_results.py`; JSON/CSV are downloadable. Packaging refuses incomplete denominators, a failed/stale audit, or per-source metadata paths. Desktop/mobile rendering and no-runtime-error/download checks are recorded in `ui_validation.json`. Seven obsolete tracked generated demo files/builders are retired with recoverable Git blob hashes; scientific results, annotation tools and pre-existing dirty/staged/untracked work remain intact. See `demo/legacy/` for public-landing cleanup and publication receipts.

Scope closed: all admitted obtainable compatible source tests have complete results and an independent recount. Other reviewed models have an evidence-backed access/compatibility disposition. No independent five-class generalization, natural-scan segmentation, safe A5C recovery or replacement working-reference gate is passed by this evaluation.
