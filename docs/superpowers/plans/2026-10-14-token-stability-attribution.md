# Token Stability and Segmentation Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The owner authorizes all routine work within this task, including independent implementation review and fact-checking, without waiting for their return.

**Goal:** Explain where A03's mixed-window classification gains and segmentation losses occur, and measure whether identical nominal frame evidence receives conflicting time-token predictions in overlapping windows.

**Architecture:** A04 is a retrospective, CPU-only diagnosis of frozen A02/A03 outputs. It reproduces the actual CLS/token systems, then uses four explicitly reference-conditioned readout combinations at one fixed decoder to separate mixed-window changes from pure-window changes. A second analysis compares saved time logits at matching frame anchors across overlapping windows. Neither analysis is an inference policy, a trained model, a performance upper bound, or proof that a consistency loss will help.

**Tech Stack:** Existing Python/NumPy, allowlisted A03 metadata adapter, the existing evaluator and paired bootstrap; Markdown and aggregate JSON. No GPU or new inference.

---

## Question and authorization

Accept A03's `labels_learned_no_usable_gain + end` and keep its results/decisions unchanged. A03 improved all-mixed requested-view accuracy in both development halves, but did not improve class-aware segment F1. Token probabilities introduced additional native switches and some A5C mistakes. Those observations do **not** explain the lack of system gain.

The unresolved questions are:

1. Are the corrected mixed-window decisions mostly changes near already-detected boundaries, or do they recover otherwise-missed reference segments?
2. Do changes in pure windows add label errors or prediction fragments that offset any mixed-window benefit at a fixed decoder?
3. Can the same nominal frame, with the same local RGB neighborhood, get conflicting time-token predictions in overlapping windows? If so, how many independent source recordings/sessions are affected, and which RGB, marker and position roles differ?

This is a new attribution question, **not** another selector, fusion, B3 or sampling search. No training is authorized, including an overlap-consistency loss, teacher, gate, residual, classifier, new token head or longer continuation. Do not silently convert a reference-conditioned diagnostic into a deployable candidate. Complete this diagnosis and report the evidence for the next modeling decision.

### Predecessors and boundaries

- **A02:** aligned anchor-label training; system inference through CLS. Labels were learned; no consistent system gain. Closed.
- **A03:** fixed nearest supervised-anchor readout; local gain, no usable system gain. Closed. This task keeps the selector exactly unchanged.
- **V02:** replacement patch readouts and more training did not improve the complete system. Do not repeat them.
- **V03:** more requests improved offsets, without a consistent segment gain; many changed samples were near reference boundaries. It did not attribute A03's two-readout difference or compare matching anchors across windows.
- **A01/R03/R11/R16:** temporal heads, boundary-only models, image-frame auxiliary supervision and L1+ASRF have already been examined. A04 does not train another temporal smoother or infer that all temporal models fail.

These development predictions and their overall results have already been inspected. Definitions are fixed before **new diagnostic counts**, but this is exploratory retrospective analysis, not independent validation or a claim of untouched preregistration.

## Fixed source records

Repository root: `/Users/williamqiu/Desktop/Harvard/Capstone/echo-view-routing`.
Server root: `/home/william/echo-view-routing`, SSH alias `echo-l40s`.

Pin these exact current artifacts before implementation; mismatches stop the run and are reported rather than repaired by changing source results:

| Source | SHA256 |
|---|---|
| `configs/experiments/trained_token_readout_v1.yaml` | `c2e90d605316a2c7d9679728464dbf3a534a7eafc6ea0760a966025be72fc8b4` |
| `docs/results/2026-10-14_trained_token_readout/summary.json` | `1cc2936f05a50dbb084a25e9d6d98486883a5d881bfd08d3bb673aa0fa688cdc` |
| `docs/results/2026-10-14_trained_token_readout/README.md` | `4d72729291866683db99ebe7bd27f327af84d047e8a57ebfdd0dabb7a1be3cce` |

Inherit the source protocol, mapping `family5_v1`, class order, recipes, window sampling, A02 prediction identities/digests and A03 choices from the pinned config/summary. Use the three aligned seeds in both halves for attribution; the other two A02 arms are controls for anchor stability only. Read saved arrays; do not load or run any model. Final checkpoint hashing is optional if it merely repeats A03 provenance; output identity/digest checking is mandatory.

Use A03's allowlist metadata adapter, or narrowly factor out its pure helpers without changing its behavior. Never call `validation()`, `Inputs` or A02's full evaluation entry. Do not read training features, test/Pediatric data, raw images or non-allowlisted split labels. Both development halves are explicitly authorized for this diagnosis. Public clip labels remain reference labels, not expert-confirmed per-frame anatomy.

## Files and responsibilities

- Create `configs/experiments/token_stability_attribution_v1.yaml`: immutable definitions, source pins, decoder settings, budget, output paths and reporting rules.
- Create `scripts/token_stability_attribution.py`: provenance, metadata-only preparation, reproduction, four diagnostic arrays, run ledger and server-private case data; phased `--prepare`, `--check`, `--run` commands.
- Create `scripts/token_stability_attribution_tables.py`: pure counts, matching, paired differences, context-role summaries and Markdown rendering.
- Create `tests/test_token_stability_attribution.py` and `tests/test_token_stability_attribution_tables.py`: consequential selector/mapping/intervention/matching/aggregation checks.
- Create result directory `docs/results/2026-10-14_token_stability_attribution/` with English `README.md`, `summary.json`, `report_tables.md`, `preparation.json`, `code_review.md` and `fact_check.md`.
- Write private arrays and case lists only under server `runs/token_stability_attribution_v1/`.
- Update **only A04** in `docs/EXPERIMENT_INDEX.md` after completion. Preserve all closed predecessors and protected unrelated changes.

## Task 1: Freeze definitions and verify the source

- [ ] Read AGENTS.md, CLAUDE.md, experiment index and corrected A02/A03/V03 reports. Commit the config before new diagnostic counts.
- [ ] Write failing tests for wrong prediction identity, reordered sequence/centre/marker arrays, wrong label mapping, selector parity and forbidden split access. Then implement the source loader.
- [ ] Reconstruct each half's metadata with the existing adapter; check saved `seq`, `centre`, `marker`, logits shape, finite values and source SHA. Never relax tolerances because a check failed.
- [ ] Reproduce all aligned A03 actual B3 rows and the same-CLS-number sensitivity rows, per seed and half, before using them. Reproduce the CLS actual rows from A02. Counts and predictions must match exactly; float probability parity follows A03's recorded computation.
- [ ] Record digest-only preparation and code/environment identity. Do not print private identifiers or include them in committed outputs.

## Task 2: Account for the actual system changes

For each seed/half, compare aligned token at its **already chosen A03 decoder** with aligned CLS at its **already chosen A02 decoder**. This is a complete-system difference, not attribution to the readout alone. Also compute the same comparison at the CLS's fixed numeric decoder in Task 3.

- [ ] Use the existing evaluator's class-aware one-to-one segment matching at IoU 0.50. Expose the exact pairs following `scripts/report_fan224_visual.py:matched_true`: descending `(temporal_iou, prediction_index, reference_index)`, greedily accepting unused prediction/reference indices. Verify matched counts against the evaluator. Report TP, FP, FN, total predicted/reference segments and exact segment F1, not only a mean per-window accuracy. Keep native, pair-stream and multi-stream tables separate; multi streams remain the primary segment population and are never replaced by pooled constructed counts.
- [ ] Count reference segments newly matched/lost, and predicted segments newly unmatched/removed, per class, native versus constructed, length <=16 versus longer, and A4C/A5C X-Y-X middles. Reference segment keys remain server-private.
- [ ] For every changed requested-window correctness, report distance from the request to the nearest reference view-change boundary in sample bins **0–5**, **6–10**, **>10**, and **no reference boundary**. Report fixed/new errors and source/session counts in each bin. These are distances of requests, not inferred proof of a recovered boundary.
- [ ] Report reference segments containing corrected-window requests: already matched by CLS, newly matched by token, matched by neither, or lost under token. A segment can contain both fixed and newly wrong windows; retain this overlap and do not sum it as mutually exclusive evidence.
- [ ] Account separately for windows whose argmax does not change but whose probabilities do; they can change B3 scores. Do not call all decoder changes classification changes.
- [ ] Report raw argmax and B3 native label-switch counts without calling a switch an expert-confirmed false cut. Group cases by source recording/session, not by seed or duplicated stream as extra patients.

## Task 3: Reference-conditioned four-cell diagnosis at a fixed decoder

Let `M[i]` be true only when the 16 input slots of window i contain at least two reference classes. It is defined from the fixed constructed labels. Native windows are pure. Keep the nearest-request mapping, selector, all windows and probabilities unchanged.

Create four window-probability arrays, **then** map each to the same 10 Hz samples. Do not apply a per-sample mask after mapping:

```python
def diagnostic_probs(p_cls, p_token, mixed, mixed_source, pure_source):
    # Sources are literal 0=CLS, 1=token. This mask uses reference labels: diagnosis only.
    sources = np.where(mixed, mixed_source, pure_source)
    return np.where(sources[:, None] == 1, p_token, p_cls)

# Name order = mixed source, pure source.
CC = diagnostic_probs(p_cls, p_token, M, 0, 0)
TC = diagnostic_probs(p_cls, p_token, M, 1, 0)
CT = diagnostic_probs(p_cls, p_token, M, 0, 1)
TT = diagnostic_probs(p_cls, p_token, M, 1, 1)
```

- [ ] Tests: CC equals the original CLS probabilities; TT equals A03 token probabilities; empty mixed/pure subsets yield the relevant endpoint; a window serving several samples uses one source consistently; inputs are never mutated.
- [ ] Decode all four with that seed's **A02 aligned CLS stored numeric B3 parameters**, including the stored numerical threshold and NMS. Do not refit quantiles, tune settings, choose a decoder or apply the token's newly selected parameters to these four cells. CC/TT reproduce A03's fixed-CLS-number sensitivity outputs.
- [ ] Recompute the same four cells with direct argmax as a prespecified sensitivity, without a second decoder search.
- [ ] Report full TP/FP/FN, class-aware F1, native macro-F1, boundary P/R/F1, native switches and same-view join cuts. Report paired F1 changes `TC−CC`, `CT−CC`, `TT−CC`, plus interaction `TT−TC−CT+CC`. This algebra is an exact metric contrast; it is not an additive causal decomposition of independent medical mechanisms.
- [ ] Fixed bootstrap: 2,000 draws, seed 20261002, same draws within each contrast; constructed streams resampled by whole stream, native metrics by session. Three-seed consistency requires all three intervals strictly exclude zero on the same side in each half. Report all six seed/half cells and raw counts. Few source sessions and repeated clips limit inference.
- [ ] Never show a reference-conditioned result as a model score in the main model-selection table. No numerical goal/constraint passing claim is allowed for TC/CT. The reference mask is not available in deployment; these cells are not attainable results or an upper bound on a proposed model.

## Task 4: Same-anchor stability across overlapping windows

This is a secondary representation check. Time tokens represent a receptive field and have interacted with the rest of the window; an anchor is not a pure single-frame embedding.

- [ ] For each sequence and nominal sample index t, collect observations `(window row, centre, marker slot, anchor j, s[2*j], time_logits[j])` across the saved eight anchors. Never compare different sequences/contrast variants or different physical samples as if they were the same.
- [ ] Deduplicate repeated/clamped copies of the same t inside one window: prefer the anchor closest to that window's marker, then lower j. Compare only distinct window centres. If no eligible pairs exist, report zero support; do not issue new requests to manufacture pairs.
- [ ] Main stability subset additionally requires both nominal anchor slots to have the complete three-sample local RGB neighborhood (2j−1, 2j, 2j+1), and the two windows' sequence-sample-index triples to match exactly. Exclude padding/edge cases from this main subset; list the broader nominal-frame subset separately. Record the three target-marker bits at these slots for each observation. Report equal-marker-triple and changed-marker-triple support/rates separately. Only RGB matching **and marker-triple equality** justify describing complete local stem inputs as identical: `marker_stem` also contributes to the stem. Check the RGB and marker stem configurations before assigning this neighborhood; if they differ from the documented 3-frame geometry, stop and report the discrepancy.
- [ ] Count sequence-sample occurrence groups, observations and unordered pairs; use each occurrence group total weight 1, splitting that weight equally over its pairs. These are not distinct physical frames across reused streams. If reporting distinct physical frames, deduplicate by source recording/frame/contrast identity. Report how many pairs include the **A03-selected output anchor** of at least one window, how many include both, and how many include neither. The stride-5/even-anchor geometry may leave little or no selected-output support; explicitly report it, rather than treating side-anchor sensitivity as the cause of A03 output switches.
- [ ] Compute argmax disagreement and JS divergence, both-correct/one-correct/both-wrong against the same t's reference label, per arm/seed/half. Keep native, pair-stream and multi-stream strata separate; both-pure versus at-least-one-mixed contexts are prespecified descriptive subsets. Report source/session counts, anchor indices, marker-relative positions and marker triples, centre distances and reference-boundary distances.
- [ ] Read corresponding saved logits for center_only and shuffled_dense as controls; do not compare different requests to CLS as if CLS predicted the same t. If the same frame has conflicting time-token predictions, context, token position and target marker all changed together. It is **joint window-context/position/marker sensitivity**, not isolated cross-boundary contamination or heart-phase causation.
- [ ] No statistical superiority claim is required for this diagnostic. Do not turn thousands of pairs from reused frames into thousands of independent cases. Group-weighted rates and source counts suffice; uncertainty for small subsets must be stated.

## Task 5: Review, run once, fact-check and close

- [ ] Run `python -m pytest tests/test_token_stability_attribution.py tests/test_token_stability_attribution_tables.py -q`, plus relevant A03 metadata/source tests. Test real failure modes and known synthetic streams; do not write a large mirror of the implementation.
- [ ] Obtain one independent implementation review. Fix blocking and consequential issues before producing diagnostic output; record all repairs and identity changes. Routine fixes within this plan are authorized.
- [ ] Commit implementation and checks, then run the source checks and the one complete diagnosis on the server. Suggested interface: `python scripts/token_stability_attribution.py --config configs/experiments/token_stability_attribution_v1.yaml --prepare`, then `--check`, then `--run`. All outputs have atomic run identities; no partial output is called complete.
- [ ] Independent fact-check must reconstruct TP/FP/FN/F1 and the fixed-decoder four-cell endpoints/contrasts from private saved arrays, verify distance bins and group weights, and spot-check matching anchor/triplet metadata with a separately written calculation. No model inference is needed.
- [ ] Write an English README that answers the three questions with counts, source units, intervals where prescribed and limits. Distinguish observations from hypotheses. In particular, say whether new correct mixed requests coincide with recovered reference segments, whether TC/CT contrasts show a pure-window trade-off at this fixed decoder, and whether stability pairs actually cover selected output anchors.
- [ ] Commit only this round's authorized files. Replace A04 pending with its scoped diagnostic result and source links. Do not change A02/A03 numerical rows, choices, rules or closed status. No push.

## Fixed report readings and stopping scope

These are nonexclusive diagnostic descriptions, not a gate claiming a trainable mechanism:

- **Local correction without segment recovery:** corrected mixed requests mainly sit in reference segments already matched by CLS; report the actual proportion and proximity bins. Never assume this merely from being within 0.5 seconds.
- **Mixed-versus-pure trade-off:** TC−CC and CT−CC have opposite directions at the fixed decoder; give the per-seed intervals and counts. Use “consistent” only if the existing three-seed rule is met in a half. The reference mask does not establish a learnable routing policy.
- **Anchor sensitivity observed:** identical nominal frame/local RGB neighborhoods receive different anchor argmax predictions; report equal-marker versus changed-marker support and changed roles. Complete local stem equality requires marker-triple equality too. If matching selected-output support is absent, the runtime cause remains unresolved.
- **Unresolved:** contradictory directions, few independent sources, differing working points or absent overlap support prevent attribution. No forced favorable branch.
- **End:** the configuration closes after the report. No training, readout fusion, shifted requests, interpolation, threshold search, new loss or changed labels follows automatically. A training task would need its own model-level hypothesis, matched controls and budget; this diagnosis cannot demonstrate that a consistency loss improves segmentation.

Budget: cumulative **2 CPU wall-hours of executed checks, diagnosis, tests and fact-check probes**, including failed/superseded runs; **0 GPU hours**. Coding and human waiting time are not CPU-stage execution time. Record a single server/local runtime ledger. If the cap or source availability prevents completion, preserve the partial stage and report it; do not remove controls, invent outputs or quietly increase the budget.

Keep test sealed, Pediatric unread, public labels unchanged, raw media/private IDs on the server, the eight staged theory files and unrelated working-tree changes intact. No deletion/cleanup task is authorized. Historical EV9V test use, repeated development use, session-versus-patient ambiguity and artificial-join limitations remain disclosed. The current strongest frozen Ordered reference and all numerical success goals remain unchanged; **A04 itself cannot meet or fail a model-development goal because it trains and proposes no deployable model**.

## Plan review record

Independently reviewed on 2026-10-14. First review identified an overstatement: equal RGB neighborhoods do not imply equal full stem inputs, because the learned marker stem also contributes. The plan now records marker triples, separates equal/changed-marker support, and requires marker equality for full local-stem equality. It also fixes the exact greedy matching order, separates native/pair/multi metric populations, and distinguishes sequence-sample occurrence groups from distinct physical source frames. Full re-review: **Approved**, no remaining blocking issues. This is plan readiness, not execution or a diagnostic result.
