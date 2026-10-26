# Residual full-segmentation errors of Base on the training-source pools (2026-10-26)

**What this is.** A cached-output diagnostic (namespace `post_a35_residual_error_inventory_v1`; scope dae4a16). It is
not an A-numbered experiment.
- **Nothing new was run.** No model inference, pixel decoding, training, threshold scan or decoder change.
- **Data not read.** No development, check, test or Pediatric file, and not the A26 runner.
- **Question.** Beyond A4C/A5C confusion, which residual errors of the current Base (A08 + B3, source seeds 0 / 2) have
  real counts and source coverage on A31's two training-source pools?
- **What it is not.** A08 trained on these recordings, so this is an inventory against EV9V references and
  construction rules. It is not a holdout or a clinical truth, and it confirms no label error.

**Method.**
- **Reproduction.** Each of the four seed × held-out fold cells was rescored with the original evaluator (one-to-one
  same-class IoU ≥ 0.5 matching) and matches A31 / A32 / A35 exactly.
- **Units.** Reference units are runs of same-class fragments, and every unit keeps all its sources.
- **Forms.** Each unmatched unit gets one fixed form from `failure_units.unit_outcomes`, with its original priority and
  tolerances (iou 0.5, a change matched within 10 samples = 1 s, hit 5):
  1. label: a predicted segment of another class overlaps it at IoU ≥ 0.5;
  2. missed cut: an edge change has no matched predicted change;
  3. extra cut: an unmatched predicted change lies inside;
  4. boundary shift: everything else.
- **Checks.** Unit counts reconcile with the evaluator's TP / FP / FN in every cell, and the forms sum to FN. No unit
  lost a same-class IoU ≥ 0.5 candidate to the one-to-one matching.
- **Reading the counts.** Forms describe morphology, not causes. Counts are summed over four cells that share sources
  (the two seeds use the same pools), so they are cell-instances, not patients.
- **Previous form evidence.** Codex's catalog diagnostic (2026-10-23, old development caches) found 22 label / 26
  missed cut / 2 extra cut / 1 boundary shift. That was read here as a document only.

## Table 1: full error table per cell (multi streams unless marked native)

| Cell | Segment F1@50 | TP / FP / FN | Short FP | Missed short true / short true | Native macro-F1 | False cuts at same-class joins | Native units / unmatched / extra predicted segments |
|---|---|---|---|---|---|---|---|
| s0, fold 0 | 0.9639 | 2,497 / 106 / 81 | 73 | 41 / 718 | 0.9776 | 24 / 1,022 | 740 / 14 / 55 |
| s0, fold 1 | 0.9499 | 2,444 / 156 / 102 | 105 | 35 / 626 | 0.9758 | 23 / 1,098 | 747 / 21 / 78 |
| s2, fold 0 | 0.9649 | 2,498 / 102 / 80 | 68 | 42 / 718 | 0.9832 | 23 / 1,022 | 740 / 12 / 49 |
| s2, fold 1 | 0.9508 | 2,434 / 140 / 112 | 90 | 42 / 626 | 0.9736 | 19 / 1,098 | 747 / 21 / 65 |

**By class, summed over the four cells.** The class totals are from fold 0 / fold 1 (identical for both seeds); the
error counts are summed over the four cells.

| True class (FN) | Units per cell (fold 0 / fold 1) | FN, 4 cells | Of which short (≤ 16) | Predicted class (FP) | FP, 4 cells | Of which short |
|---|---|---|---|---|---|---|
| A4C | 531 / 517 | 166 | 73 | A4C | 243 | 162 |
| A5C | 512 / 526 | 173 | 69 | A5C | 238 | 159 |
| PLAX | 501 / 487 | 15 | 3 | PLAX | 8 | 6 |
| PSAX | 525 / 524 | 13 | 11 | PSAX | 9 | 8 |
| SC4C | 509 / 492 | 8 | 4 | SC4C | 6 | 1 |
| **All** | | **375** | 160 | | **504** | 336 |

- **FN.** A4C and A5C units are 8.1% unmatched (339 / 4,172 cell-instances), against 0.6% for PLAX / PSAX / SC4C
  (36 / 6,076).
  - The A4C/A5C FN come from 150 distinct recordings and 109 sessions.
  - The other-class FN come from 24 recordings and 23 sessions.
- **FP.** 95% of false segments are predicted A4C or A5C (481 / 504). Of these, 319 overlap most with a reference unit
  of the other class of the pair, and 154 with a unit of their own class.
- **Reference changes.** A4C ↔ A5C changes are missed 15.8% of the time (133 / 840). All other class changes are missed
  0.6% of the time (42 / 7,008).
- **Natives.** Unmatched natives are A4C 42 and A5C 24, against 2 PLAX. Extra predicted segments are A4C 106, A5C 132
  and 9 other.

## Table 2: forms × class of every unmatched multi unit (four cells summed)

| Form | Total | A4C | A5C | PLAX / PSAX / SC4C | Distinct recordings / sessions (union over cells) | True class top-ranked by unit-mean probability | Mean share of samples with argmax = true class |
|---|---|---|---|---|---|---|---|
| Label: A4C ↔ A5C | 121 | 66 | 55 | — | 52 / 43 | 14 / 121 (12%) | 0.21 |
| Label: other classes | 9 | 3 | 0 | 6 / 0 / 0 | 7 / 7 | 0 / 9 | 0.23 |
| Missed cut | 189 | 73 | 87 | 8 / 13 / 8 | 111 / 87 | 107 / 189 (57%) | 0.57 |
| Extra cut | 50 | 22 | 27 | 1 / 0 / 0 | 46 / 42 | 39 / 50 (78%) | 0.55 |
| Boundary shift | 6 | 2 | 4 | — | 5 / 5 | 5 / 6 | 0.63 |
| **Unmatched total** | **375** | 166 | 173 | 15 / 13 / 8 | | | |
| Matched (reference) | 9,873 | | | | | 99.9% | 0.94 |

**Missed reference changes and the existing probability evidence.** For each missed change, the per-sample argmax
majority was checked on each side.

| Change pair | Missed (4 cells) | Both sides' argmax already correct | One side wrong | Both wrong | Shorter side ≤ 16 when both correct |
|---|---|---|---|---|---|
| A4C ↔ A5C | 133 / 840 | 48 | 81 | 4 | 27 / 48 |
| All other pairs | 42 / 7,008 | 14 | 28 | 0 | 8 / 14 |

- **Not saved.** B3's candidate scores were not saved, so only the cuts it kept are known. Nothing here was re-decoded.
- **Reference-assisted.** "Both sides correct" uses reference boundaries. It does not show that a deployable decoder
  could keep these cuts without adding false short segments.
- **Not additive.** These forms are not separable gain budgets or mathematical limits.

## Decision

**No problem outside A4C/A5C ambiguity has enough coverage to support another training round.**
- **What the other classes contribute.** Most of their errors are missed cuts next to a short segment with one side
  misread. They come to:
  - 36 unmatched unit-instances over four cells (0.6% of their units; 24 recordings, 23 sessions);
  - 42 missed changes (0.6%);
  - 23 false segments.
  These are too few and too varied to define a new question. They overlap the short-segment and boundary work already
  closed: short-segment probes, A10 / A11 / A13 switch heads, A19 decoder points, A25 segment risk.
- **What remains is A4C/A5C.**
  - 90% of unmatched true segments and 95% of false segments are A4C or A5C.
  - Even the missed-cut and extra-cut forms are dominated by A4C ↔ A5C changes. A4C ↔ A5C changes are missed 15.8% of
    the time; others 0.6%.
  - Within those, 48 missed A4C ↔ A5C changes already have correct argmax majorities on both sides. That is a boundary
    observation, but it concerns the same A4C/A5C boundary.
  - Keeping more short A4C/A5C cuts is what the closed switch-head, router and segment-risk rounds tried, and they
    traded it for false short segments.
- **What is missing before new training is justified.**
  1. **For A4C/A5C.** Evidence that the clip codes and the per-frame view agree, or a supervision source that resolves
     them. A27–A35 showed that clip-code training (head or last block, interval or whole-recording bag, mean or
     power-mean, uniform or error-focused) and three frozen external recognisers do not establish the needed
     transferable correction.
  2. **For boundaries.** Deployable evidence, which no closed round found, that separates real short A4C ↔ A5C segments
     from false ones.
- **The one open question.** Do EV9V's A5C codes on the recordings where every model reads A4C match what an expert
  sees, including whether the LVOT / aortic valve is visible?
  - This needs a human reader. The blind pilot tool is ready, with 0 real records.
  - Model agreement cannot settle it, and no model round is started.
- **Working reference.** A08 + B3 (Base) stays the working reference.
- **Scope of the stop.** This stops this chain of automatic model variants. It does not declare the model family or
  weak supervision useless.

**A27–A35 in one paragraph.**
- **Training.** Recognition corrections on top of the unchanged A08 + B3 cuts were trained on clip codes: head-only or
  last-block branches, interval or native-window or whole-recording supervision, mean or power-mean aggregation,
  uniform or error-focused sampling, and denser real frames in the 1.5 s window. None established a transferable gain.
  Gains on A4C runs read as A5C were paid for with correct A5C runs.
- **Frozen recognisers.** EchoViewCLIP, EchoPrime and EchoFine did not recover the A5C runs that A08 reads as A4C.
- **Labels.** Expert review is what can address label and visibility questions. Model consensus cannot decide them.

## Cost and files

- **GPU 0 h. CPU 0.13 of 1 h:**
  - one failed attempt (a JSON-key error after computation, booked by hand);
  - the inventory and a class-group union;
  - the agent's active wall clock.
  - All of it is in `runs/post_a35_residual_error_inventory_v1/ledger.jsonl`.
- **Files.** `inventory.json` (aggregates per cell) and `scripts/inventory_residual_errors.py`.
- **Server only.** Per-unit forms and recording lists.
