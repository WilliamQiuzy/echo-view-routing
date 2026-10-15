# Segment-Balanced and Transcript-Forward Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The owner authorizes routine implementation, fixes, checks, independent reviews, formal training, evaluation and round-only commits within this fixed task without waiting for their return.

**Goal:** Test whether changing the training objective on the strong frozen video representation recovers class-aware reference segments, rather than only improving labels within already-matched segments.

**Architecture:** A05 has three matched five-class `Linear(512, 5)` heads on each seed's existing frozen V01 Ordered video features. F uses frame-weighted cross-entropy; B gives reference segments equal importance before the inherited class weighting; O adds a transcript-forward loss to B. All three start from the same source classifier, see identical batches and train for the same number of updates. Inference uses only the features, five-class probabilities and the existing simple change-point decoder; a transcript is never an inference input.

**Tech Stack:** Existing PyTorch/NumPy, A01's float32 feature caches and batch plans, log-space dynamic programming, existing B3/evaluator/bootstrap; English aggregate reports. No image/feature extraction or backbone fine-tuning.

---

## 1. Decision and hypothesis

Accept and preserve A04's diagnostic closure. A04 found that most corrected mixed requests were in reference segments already matched by CLS; it did not establish a pure-window trade-off or the cause of the actual selected-output switches. **Do not launch cross-window consistency training on that evidence.**

New question: with the representation and head structure fixed, do segment-balanced errors and an entire training sequence's class order produce better complete segmentation than frame-weighted errors? This is a training-objective comparison, not a claim that frame loss caused the remaining mistakes or that a transcript guarantees correct durations/IoU.

The O loss sums probabilities of all five-class frame paths that collapse to the training transcript. This encourages support for every transcript element, including X-Y-X middles, but can still give poor durations or boundary positions. B supplies the simple explanation/control: perhaps just balancing segment contributions is enough. F, B and O use the **same public labels**, with no new expert, anatomical, cardiac-phase or geometric supervision.

### Predecessors and substantive change

- **V01:** strong Ordered MViT/video representation and simple decoder; keep frozen as the strongest current reference.
- **A01:** these same frozen video features with frame-weighted linear/ASRF/MS-TCN heads; no consistent segment gain. A05 does not add another temporal architecture. The changed condition is the loss, with matched F/B controls; F repeats the A01 linear condition as an explicit replication/control.
- **A02/A03:** anchor-label training and direct time-token readout improved some local classification but not complete segmentation. A05 does not reopen anchor selection or those models' training.
- **A04:** exploratory attribution motivates evaluating recovered/lost reference segments. It does not prove the A05 hypothesis.
- **F02/F04:** FACT on E1 image features used segment-token matching/refinement; that differs from the present five-class linear model, frozen video input and forward training criterion. Existing FACT scores remain references, not evidence for or against this new loss.
- **R03, R13, R17 and S01–S05:** boundary separation, residuals, fixed-output decoding and fragment classification/calibration already have closed configurations. No such search is added here.

Sequence-probability training is an established idea: [CTC, Graves et al. 2006](https://www.cs.toronto.edu/~graves/icml_2006.pdf) sums alignment probabilities; [Li et al. 2019](https://arxiv.org/abs/1909.13155) uses transcript-related forward learning for action segmentation. Our five-symbol stay/advance graph has **no blank**, learned duration model, HMM, GRU, invalid-path discriminative term or transcript decoder. It is not a reproduction of either paper and must not be presented as a novel architecture or an already-supported paper idea.

## 2. Fixed data, sources and scope

Local repository: `/Users/williamqiu/Desktop/Harvard/Capstone/echo-view-routing`.
Server: `echo-l40s:/home/william/echo-view-routing`.

Pin these sources before implementation or new training-library counts:

| Source | SHA256 |
|---|---|
| `configs/experiments/video_feature_temporal_v1.yaml` | `3b065e7fe908ca7432ba0a407fd6b82432882c1d72dabd1d22b899b79d790690` |
| `docs/results/2026-10-12_video_feature_temporal/preparation.json` | `5f7d48db3a3b2f54f0788d9dc66a53dae2ff3dbf5105d9c1033b78a2f417d021` |
| `docs/results/2026-10-12_video_feature_temporal/summary.json` | `3a1601512bc423a6ad97b5c88ed29e8ebf949a6df802bc0a76fb6fc13c503acf` |
| `docs/results/2026-10-14_token_stability_attribution/README.md` | `52a81cefa4dee51ddea202fcc37536ecbb6dde7c4e788cce0a72c4d38c58957d` |
| `docs/results/2026-10-14_token_stability_attribution/summary.json` | `87a60003962995d6dc093b7de7804a89587a6d53532b484dac925875d1852639` |

Read AGENTS.md, CLAUDE.md and the experiment index first. Read the linked corrected A01/A04/V01 reports. Use `family5_v1` in its existing class order, numeric frame order, the fixed protocol/bank and session splits. A01's six train/development `ordered_s{0,1,2}.npz` caches and preparation record were confirmed present on the server before this plan; their **contents/identities must still pass the task's source checks**.

- Training: the same 3,364 native recordings and 1,500 constructed streams, 10 Hz sample features, nearest stride-5 window feature per sample, no re-encoding or normalization. A feature already contains 16-frame video context. Repeated feature rows are not independent new observations.
- Read only seed k's `ordered_s{k}` coordinates with seed k's head. Never mix the three feature coordinate systems.
- The A01 stored sequences/recipes provide labels and all same-view joins. A transcript is the run-length-collapsed **class** sequence: adjacent equal classes merge even across a recording join; X-Y-X keeps three states. No frame/recording/session/stream ID enters a model or a loss weight.
- Native crops are the same A01 crop slices; recompute segments/transcript **inside the crop**, with no out-of-crop label or length information.
- Only training allowlisted labels/features are read before all nine trainings finish and freeze. Development **hashes/pin paths only** may be checked earlier; no new development feature arrays, probabilities or labels are interpreted until then. Historical reports are known, so these remain reused development data, not independent validation.
- Source fingerprints, ordered model/source-classifier initial states, feature identities, sequence-file digests and epoch batch digests must agree with A01. Pin full cache hashes in the new preparation record; do not regenerate a missing cache or call a video encoder under this task.
- Test stays sealed, Pediatric unread, public labels unchanged. No raw media or private identifiers leave the server. Preserve the eight staged theory files and unrelated changes. No push or cleanup.

## 3. Exact model and objectives

All heads have **2,565 trainable parameters**. Copy A01's pinned linear **initial** state, itself a copy of that seed's V01 final source classifier. Do not start from an A01 trained head, an A02 model or a newly random initialization. Do not load a full video encoder into a training optimizer.

For sequence n, valid length T_n, logits z[t,c], p=softmax(z), reference runs R_n and transcript y[0:K_n], let alpha[c] be the exact A01 per-seed class weights from the pinned training plan. They are shared by all three arms, never re-estimated from development data. Let e[t]=-log p[t, reference[t]]. All sums below exclude padding.

**F — frame-weighted baseline:**

`L_F = sum_n sum_t alpha[reference[t]] * e[t] / sum_n sum_t alpha[reference[t]]`.

This is A01 linear's masked weighted CE. Use the same computation, optimizer and batches where possible; record any unavoidable numerical implementation difference, rather than calling a changed loss exact replication.

**B — segment-balanced control:**

`D_n = sum_(r in R_n) alpha[class(r)]`

`L_B = sum_n sum_(r in R_n) alpha[class(r)] * mean_(t in r) e[t] / sum_n D_n`.

A longer segment no longer has more total weight solely due to its number of samples. This is equal segment importance **before class weighting**, not equal patients, source recordings or independent cases. No class or reference segment is discarded.

**O — segment balancing plus transcript-forward candidate:**

For each valid sequence, sum over all positive-duration paths through its K states, starting at state 0, ending at K−1 and either staying or advancing exactly one state per sample:

`P(y | x) = sum_(valid paths k[0:T]) product_t p[t, y[k[t]]]`.

`q_n = -log P(y_n | x_n) / T_n`

`L_order = sum_n D_n * q_n / sum_n D_n`

`L_O = (L_B + 0.5 * L_order) / 1.5`.

The fixed coefficient 0.5 is an operational choice, not a tuned value. No sweep follows. Labels supply training losses only; `model(x)` and decoding receive no transcript, labels, joins, native/mixed flags or case identity. There is no sixth/blank class and no new head or parameter. The inferred five-class output and B3 interface are identical across arms.

### Forward recursion and efficient implementation

The reference graph can be evaluated without an O(T) sequence of GPU kernel launches. For finite `(T,5)` log probabilities and adjacent-distinct transcript y, use the following log-cumulative-sum recursion (CPU float64 reference):

```python
def transcript_nll(logp, y):
    T, C = logp.shape
    K = len(y)
    assert C == 5 and 1 <= K <= T
    assert all(y[k] != y[k - 1] for k in range(1, K))
    prev = logp[:, y[0]].cumsum(0)
    for k in range(1, K):
        pref = logp[:, y[k]].cumsum(0)
        entries = prev[k-1:T-1] - pref[k-1:T-1]
        valid_suffix = pref[k:] + torch.logcumsumexp(entries, dim=0)
        prev = torch.cat((logp.new_full((k,), -torch.inf), valid_suffix))
    return -prev[-1] / T
```

Every logcumsumexp input suffix starts at a reachable state; never differentiate an all-negative-infinity prefix. Do not use probability products, allow zero-length states, silently use `zero_infinity`, collapse nonadjacent repeats, or substitute Viterbi/max for the forward sum. Formal head weights/feature arithmetic remain fp32/TF32 off as A01; compute the objective accumulations and forward recursion in float64 from the same fp32 log-softmax values in **all arms**. This keeps the normalization/numerics matched; it may cause small differences from historical A01 F, which must be reported. Returning the loss in float64 still propagates gradients to the fp32 head. No precision change is selected after results.

The sequence loss does not optimize IoU or post-B3 segment F1 directly. It can improve transcript ordering without improving localization. That is a possible negative result, not a reason to reinterpret the goal.

## 4. Files and task sequence

Create these focused files; do not edit old experiment outputs, configs or official model libraries:

- `configs/experiments/segment_objective_v1.yaml`: all sources, equations, training/evaluation rules, budget and identities.
- `echo_routing/temporal/losses/segment_objective.py`: pure F/B/forward losses, reference run extraction and masks.
- `scripts/train_segment_objective.py`: source reuse, head construction, prepare/check/queue/run and resume, training-only monitors and one ledger.
- `scripts/report_segment_objective.py`: metadata-only development loading, final-head probabilities, calibration choices and one selection evaluation.
- `scripts/segment_objective_tables.py`: pure system/matching/paired/decision summaries.
- `tests/test_segment_objective_losses.py`, `tests/test_segment_objective_pipeline.py`: meaningful mathematics/data-isolation/replication tests.
- `docs/results/2026-10-14_segment_objective/`: English README, summary.json, report_tables.md, preparation/check records, code_review.md, fact_check.md and a compact training-curve figure.
- Server-only `runs/segment_objective_v1/` and `models_frozen/segment_objective_v1/{frame,segment,order}_s{0,1,2}`.

### Task 1 — Fix the configuration and sources

- [ ] Commit the configuration before new library counts, code, head preparation or loss comparisons. Put the exact equations and nonexclusive decision rules in it; unknown source manifests are resolved by pinning A01 records, not by changing data.
- [ ] Write failing source/shape/mask tests, then implement read-only training-cache and initial-state loaders. Verify A01 feature identities, class order, sequence offsets, training allowlist, all epoch batches and source initializer digests; require five finite positive inherited class weights. Resolve every pin before using the cache.
- [ ] Save all three per-seed initial states as bit-identical arm copies; record parameter counts and parent identities. Three F runs are real matched controls, not an unverified reuse of historical models.
- [ ] Report training-only segment/transcript counts, length/class distributions, same-view joins and loss-mass distribution for F versus B; these are descriptive and do not trigger adaptive weighting or sample filtering.

### Task 2 — Prove the losses before training

- [ ] Write and run red tests before implementation; then compare the forward value and autograd gradients against independently enumerated positive-duration compositions on tiny T<=6 sequences, including X-Y-X. CPU float64 value/gradient tolerance 1e-9.
- [ ] Test K=1 reduces q to mean CE; T=K has one alignment; transcript probability <=1 and forward q <= fixed-reference unweighted mean CE within 1e-9. Test impossible/empty/adjacent-repeated states reject explicitly; do not silently skip samples.
- [ ] Check B and O have equal loss and gradients (1e-9 in CPU float64) on all-native batches of differing lengths/classes. They should: D weights make `L_order == L_B` there. F differs when duration weighting differs, intentionally.
- [ ] Check arbitrary changes in padded labels/features do not change any valid loss/output/gradient. Check F reproduces masked weighted CE; B is invariant to merely repeating identical-loss samples within a reference run while F changes that run's relative mass.
- [ ] Prove inference label isolation: changing labels/transcripts/joins/IDs while holding x and valid lengths fixed cannot change logits/probabilities/predictions. No transcript-constrained decoding is allowed at evaluation.
- [ ] Implement model/optimizer/RNG round-trip tests: six updates versus rerun and 3+3 resume, tolerance 1e-6 on formal fp32 parameters and losses. All three arms have identical initial predictions and exact batch digests; source parameters are never optimized.
- [ ] Run 100-update finite-gradient/updated-head smoke checks on fixed A01 training monitor sequences, without a fit score threshold or retuning. Time each arm on formal batches, including the order recursion. Independent implementation review must verify equations, DP/gradient tests, source/data isolation, batching, optimizer binding, choices barrier and budget. Fix consequential issues and recheck before starting.

### Task 3 — Run the nine bounded trainings without waiting for the owner

- [ ] Three seeds 0/1/2. A01's 50 epochs × 375 updates = **18,750 updates per head**, four complete bank streams plus four native crops per batch; same epoch/batch/crop plan in all arms. AdamW lr 3e-4, weight decay 1e-4, betas (0.9,0.999), eps 1e-8, grad clip 1, 750-update warmup then the same cosine schedule as A01. No dropout, feature normalization, augmentation or class-weight change.
- [ ] Run order: frame_s0, segment_s0, order_s0, then the same for seeds 1 and 2. Source embeddings and their extraction/model settings never change. Do not unfreeze the encoder or add a temporal network.
- [ ] Before launch, require spent + 3×each arm's measured projected training + reserve to fit the fixed budget. Count successful/failed smoke and formal attempts. Do not drop controls, shorten runs, add a seed or change a learning rate to fit the cap.
- [ ] Take final weights only. No new development feature/label reads or validation during training. Save per-epoch complete resume states and training-only loss components, reference-segment mean CE, native/multi training fit and normalized transcript edit distance. These monitors do not select epochs or trigger extensions.
- [ ] Freeze all nine final heads and manifests/digests before evaluation. An interruption resumes only under the same complete run identity, not merely a matching run directory name. Record actual process/update evidence before reporting training started.

### Task 4 — Calibrate, freeze choices, evaluate selection once

- [ ] After all nine freeze, load only allowlisted calibration cached features/labels. The safe metadata-only helpers are **A03's `scripts/trained_token_readout.py::half_index`, `meta_stream` and `half_sets`**, with their pinned source protocol/plans; use or narrowly factor them without changing A03 behavior. Do **not** call A01 `dev_features`, `report_center_supervision.validation`, `dev_run.Inputs` or an encoder-extraction fallback: those paths eagerly load other arrays/halves or run an encoder. Add access-guard tests that reject premature selection access and any encoder construction/forward.
- [ ] Build a separate read-only A01 feature-cache adapter with explicit half allowlists and offsets. A01's development NPZ concatenates both halves: never evaluate `np.load(...)["samples"]` on the full development array during calibration and filter afterward. Its `np.savez` members are uncompressed: validate ZIP_STORED, parse the NPY header/offset and use a read-only memory map or equivalent bounded seek to copy only the allowed row spans. IDs/lengths/offsets are metadata; expose no forbidden feature rows or global development labels. Reject an incompatible/compressed layout rather than falling back to eager loading or extraction. Test with canary selection spans that calibration never indexes; label metadata must also be filtered before values are read. Cache splitting/routing is not new model feature extraction and must not run a head on selection before the barrier.
- [ ] Pin A01 development metadata/cache identities and verify calibration source-classifier feature-to-logit parity (1e-4, argmax equal) against the archived calibration outputs. Reproduce calibration Frozen Ordered and A01 Linear/ASRF/MS-TCN archived rows only. The corresponding selection parity/reference reproduction is postponed until choices are committed; do not load selection archived probability arrays early merely because they already exist.
- [ ] For each final F/B/O head, select B3 **once on calibration** from h=[5,7,9,11,15], q=[.80,.85,.90,.95,.97,.99,.995,.999], NMS=5. Quantiles use that head's calibration native scores. Feasible requires native macro-F1>=.980 and multi boundary recall>=.95; maximize multi segment F1@50, exact ties to original grid order. Infeasible is failed; 9/.95 may appear as a diagnostic only and can never pass.
- [ ] Commit all choices and provenance before any new selection probability or metric is computed. Make this a tested launch barrier: verify the committed choices blob against the file and reject missing, uncommitted or changed choices/source/model/config identities; the server run requires that exact verified choices SHA and commit proof (synchronize needed Git objects/proof without pushing). Only then read selection's allowlisted feature/label spans and archived reference arrays, verify selection parity, and evaluate selection exactly once. Do not inspect a selection grid or choose a head/epoch/weight from it.
- [ ] Main comparison is **O−B multi segment F1@50**, per seed and half. Mandatory comparisons are B−F, O−F, O−Frozen Ordered+B3, B−Frozen Ordered+B3. Also list F versus historical A01 Linear and the strong reference to disclose replication/normalization differences. Other archived references may be shown as stored rows, never re-tuned.
- [ ] Report the same heads' raw argmax and their probabilities at the seed's **stored Frozen Ordered numerical B3 settings** as sensitivities. Do not select a result from these rows.
- [ ] For each actual system, report TP/FP/FN and F1 for native, pair and multi populations separately; reference segments recovered/lost versus B/F/Frozen Ordered, <=16-sample units and A4C/A5C X-Y-X middles; all with distinct source recordings/sessions. Use existing exact greedy class-aware IoU=.5 matching/tie rules. More matched references with many extra predictions is not a successful recovery.
- [ ] Report native macro-F1 and per-class precision/recall, A4C↔A5C fixed/new recordings, native switches, boundary P/R/F1 at .5s tolerance, same-view join cuts and normalized transcript edit distance (raw and B3). A better transcript or a changed probability calibration alone is not a segment result. Public labels and constructed boundaries retain their limitations.
- [ ] Paired bootstrap: 2,000 draws, seed 20261002, same draws per comparison; native by session and constructed streams by whole stream, approximate with reused sources. Bootstrap main F1, native macro-F1, boundary recall, FP/FN counts and native switches. Three-seed consistency means all three per-seed intervals strictly exclude zero on the same side within a half; otherwise no consistent difference detected, not equality. Seeds are not new cases.

### Task 5 — Fact-check, report and close

- [ ] Independent fact-check recomputes all system rows, primary/must-list intervals, targets/readings and source-unit counts from saved probabilities. Independently verify DP values/gradients on toy cases, one recorded formal batch's F/B/O losses and matching indices; verify label isolation and choices-before-selection. Do not call a queued review a passed review.
- [ ] English README: what trained, exact loss/inference distinction, actual run/check identities, all source pins, training curves, comparisons, recovered/lost source cases, constraints/goal, numerical/timing deviations and scope limits. Put the whole-video encoder unchanged and 2,565 trainable parameters prominently in the model description.
- [ ] Commit only this round's files, then replace A05 pending in the index with the verified result and links. No source/older choices/results are edited. No raw media, private IDs or per-sequence arrays in commits; these stay on the server.

## 5. Fixed targets, readings and stopping scope

Numerical goals stay unchanged: mean multi segment F1@50 >= **0.944409687184662 / 0.9821421709894332** for selection/calibration; calibration sensitivity **0.9838935784747763**; every seed/half requires a feasible B3, recall>=.95 and native macro-F1>=**.973/.980**. Use unrounded values. Passing the old goal alone is not superiority over the now stronger Frozen Ordered system.

Nonexclusive readings, fixed before implementation:

- **Complete system progress:** B or O has consistent positive F1 versus both F and Frozen Ordered in both halves, meets every constraint and old goal. Claim an order-objective contribution only if O−B is also consistent positive in both halves; otherwise report which candidate progressed without that claim.
- **Segment-balancing effect:** B−F F1 is consistent positive in both halves and B meets every constraint. Report reference/goal results separately. A standard weighting change is not automatically novel.
- **Order-objective effect:** O−B F1 is consistent positive in both halves and O meets every constraint. This is the effect of this extra training term under this recipe, not proof that all missed segments were caused by absent temporal order.
- **Transcript-only / local effect:** edit distance or some recovery counts improve but primary F1/constraints do not. Report the gain in its own metric; do not call it model success.
- **Trade-off:** recovered segments are accompanied by new FP, native classification loss or recall-floor failure; show counts/intervals without forcing causality. Better F1 below a mandatory floor is still not a qualifying system.
- **Uncertain/no qualifying gain:** primary consistency conditions are not met; preserve direction and uncertainty, including one-half-only effects. Do not equate confidence intervals containing zero with equality.
- **End:** after the one report, this fixed feature/loss/budget configuration closes. No larger encoder, nonlinear head, boundary head, teacher, gate, duration prior, transcript decoder, loss-weight/epoch/learning-rate search, more seeds, new feature extraction or automatic continuation follows.

## 6. Budget and autonomy

Cap **4 GPU hours** for all checks, nine trainings, head inference and GPU fact-checks; reserve **0.75 GPU hours** in launch projections for evaluation/review contingencies. Additional executed CPU-stage wall budget **3 hours**, including source checks, tests, bootstrap and CPU fact-checks. Record both separately in one append-only ledger; include failed/superseded attempts and avoid double-counting overlapping CPU/GPU stages. Human/coding wait is not execution time. Routine repairs and resume within the unchanged task are authorized; budget/source/model-scope changes are not.

Recommended commands after the matching code/config exist: `python scripts/train_segment_objective.py --config configs/experiments/segment_objective_v1.yaml --prepare`, `--check`, `--queue`; then `python scripts/report_segment_objective.py --config configs/experiments/segment_objective_v1.yaml --calibrate`, commit choices, and `--selection`. Exact executable commands/environment are recorded in the report. Checks passing and projected cost fitting the cap authorize immediate launch: **do not wait for the owner to return**. A source mismatch, failed check or insufficient budget must be located and reported, not fixed by weakening a scientific rule or silently dropping a group.

Respect other projects' processes and shared GPU limits. Preserve original source caches; all new outputs have new run identities. This task is a bounded test of an objective, not a promise of a positive result. The two development halves are repeatedly used; grouping is by session, not verified patient; constructed streams are not real scanning. All previous closed configurations and the currently strongest frozen reference remain unchanged.

## Plan review record

Independently reviewed on 2026-10-14. First review identified an unsafe adapter reference: A01's report uses eager loaders and an encoder fallback. Task 4 now names A03's metadata-only helpers, forbids those A01 paths, requires half-bounded access to the combined development NPZ, and tests the committed-choices barrier. Full re-review: **Approved**, no remaining blocking issues. All listed local source pins matched during review.

Before dispatch, the plan's recursion was also compared with separately enumerated positive-duration alignments on 25 synthetic CPU float64 cases, including X-Y-X: largest loss difference 8.9e-16, gradient difference 1.2e-16. This is a mathematical planning check, not formal implementation verification, a trained model or a data result. The formal implementation/review/check and nine runs remain pending until Claude executes them.
