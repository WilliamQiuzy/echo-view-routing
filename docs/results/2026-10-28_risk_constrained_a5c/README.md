# A44: a repair policy for Base-A4C runs trained on the fit fold's actual repair opportunities, risk-constrained (N) vs class-balanced BCE (C) (2026-10-28 UTC)

**Question.** Can an end-to-end visual repair policy recover Base-missed A5C runs while respecting an explicit budget on
damaging correct A4C runs? N (hinge false-promotion constraint on the protected correct-A4C runs with an adaptive dual
variable) against C (ordinary class-balanced binary cross-entropy) on identical deployed-unit training examples, inputs,
architecture, initialisation, 600 updates and compute. A common C / N gain over Base supports the shared package
(training on deployed-unit repair opportunities + the one-way rule); N − C alone addresses the constraint.

**Completed and independently checked (2026-10-28 UTC).** All four source-0 models completed their original
600 updates before the Codex takeover; no training was repeated. The four-weight manifest was committed as
`629ccb9b458181143844df1fea3d9a74b57e2f83`, inspected as actual git blobs, frozen at 19:20:31 UTC, then scored once
(fit f → held-out 1−f, PID 2891670, 19:20:48–19:32:26 UTC). Both arms fail the absolute package gate;
N also fails its final-fit hinge constraint and the separate N−C contribution gate. Source 2 was not triggered;
no further training or A45 was launched.

| Held-out result (runs, fixed original denominators) | C: balanced BCE | N: constrained primal-dual |
|---|---:|---:|
| A5C S1 recovered / 103 | 12 (6 / 6 by fold) | 26 (14 / 12) |
| Correct A4C harmed / 1,413 | 22 (20 / 2) | 76 (64 / 12) |
| Net fixed over every class | −10 (−14 / +4) | −50 (−50 / 0) |
| True-A5C recordings inside recovered runs | 9 (5 / 4) | 16 (9 / 7) |
| Complete multi-stream F1@50, fold 0 / 1 | 0.961456518 / 0.950787784 | 0.953660436 / 0.950340798 |
| Two-fold mean F1@50 | 0.956122151 | 0.952000617 |
| Mean ΔF1 from Base 0.956885277 | −0.000763126 | −0.004884660 |
| Missed short true segments (Base 76) | 86 | 104 |
| Original boundaries deleted by merging | 30 (21 / 9) | 75 (55 / 20) |

**The constraint did not provide protection in this run.** Independent per-run comparison found all 12 C fixes and
all 22 C harms also under N. N-only changes comprise 14 additional A5C fixes and 54 additional correct-A4C harms
(68 label disagreements), hence net N−C = −40. N has lower full F1 than C in **both** held-out folds. These are
paired private-cache counts, not an inference from aggregates. “Recovered” counts original P runs, not patients.


## Predecessors and the changed condition

- A43 trained a marker-time gated-attention witness on whole native A4C / A5C bags and acted on Base P runs by run max:
  D fixed 25 / 103 S1 but promoted 112 / 1,413 correct A4C (net −87) and worsened segmentation; M fixed 2 and broke 4.
  A42's same-session ranking fixed 4 (S) vs 3 (C) S1 runs and broke 56 correct runs. A31's error emphasis and
  A29 / A27's pooled five-class interval CE, A16's utility weighting / routers and A25's fixed expected-segment-risk
  penalty already exist. None is reopened; A44 does not invent interval pooling, selective correction, hard negatives,
  encoder adaptation or risk awareness.
- **Changed condition.** Training *directly on the fit fold's actual Base-A4C repair opportunities* (positives = S1 runs,
  protected negatives = correct A4C runs, each with its entire original P membership and multiplicity weights) with a
  false-promotion constraint on the protected runs and an adaptive dual variable, against a matched soft-cost classifier.
  The supervision unit and the optimisation problem change; A43's one-way r > 0 rule does not. Motivation only:
  Neyman–Pearson asymmetric error control; Cotter et al. AISTATS 2019 discuss the limits of nonconvex Lagrangian
  learning. A44 tests the simple empirical hinge-constrained primal–dual recipe below, not their proxy-Lagrangian
  algorithm or its guarantees; nonconvex stochastic optimisation on these weakly labelled recordings gives no held-out
  harm guarantee.
- **Scope.** Source seed 0 (A08 all_s0) on A31's two session folds; four models (C and N per fit fold). No development,
  check, test or Pediatric data; no A43 weights. A08 trained on these recordings, so the folds hold out only the new
  repair policy. Labels are EV9V clip codes (the run reference is the original clip-code majority, not anatomical or
  expert truth).
- **Records.** Configuration `configs/experiments/video_risk_constrained_a5c_v1.yaml` (696ac68, committed before any
  plan, model or loss computation; gate wording clarified 24c6bc8 before any check). Code 565bb95 + 24c6bc8 + 8fcdd5b.
  Checks 5139fed (attempt 1 kept, 8fcdd5b). Weights: `629ccb9` (complete manifest and provenance, verified before freeze/readout). Task sheet
  `claude_prompts/2026-10-28-risk-constrained-a5c-training.md` (sha256 25f668c7…).

## Design

- **Training units** (per fit pool, from A31's cached Base P runs and reference labels only). Eligible = original Base
  P-run label A4C. Positives = eligible with reference A5C (S1); protected negatives = eligible with reference A4C; the
  rare other-class eligible runs are excluded from binary training and kept in every readout. Lists in original P order,
  no purity / duration / source filter, no augmentation. Asserted before any pixel: 34 / 714 / 1 (fit 0) and 69 / 699 / 3
  (fit 1); eligible 749 / 771. Each run uses its *entire* original membership: unique nearest-centre request rows with
  weights v_j = multiplicity / total (sum 1); r = Σ v_j s_j, verified against the explicit repeated-sample mean (4.4e-16).
  Windows straddling run boundaries are neither cropped nor recentred; a row shared by two runs is two occurrences.
- **Plan** (RNG [20261028, 44, s, f, u] for the positive, one permutation [20261028, 44, s, f, 1000] for the negatives at
  indices (3u + 0..2) mod n). 600 updates of one positive + three protected runs; 14,778 / 15,528 window occurrences
  (at most 129 / 88 per update); all positives and all protected negatives drawn, none repeated within an update; plan
  sha256 as asserted on the card; the same plan for C and N.
- **Model.** A bit-identical A08 all_s0 copy runs its full original marker forward; the unchanged 512-d `embed` output
  feeds a new Linear(512, 1) head (513 parameters; weight 0, bias −2, so every initial score is −2 and Init = Base
  exactly). Trained from update 0: encoder trunk (lr 1e-5), marker stem (1e-4), the original 768 → 512 projection
  (1e-4) and the head (1e-3); AdamW wd 1e-4, warm-up 60 then cosine to 600, global clip 1; FP32, TF32 off,
  `warn_only=False`, A42's gather skip pools on training copies; eval mode. Only the unused five-class classifier is frozen.
- **Objectives** on the four run means of an update: A = softplus(−r_p), B = mean softplus(r_n), H = mean relu(1 + r_n).
  C = A + B. N = A + λ (H − 0.01) with λ constant in the primal step and λ ← clip(λ + 0.1 (H_pre − 0.01), 0, 100)
  afterwards (λ₀ = 1; no EMA, warm-up or final selection; saved in the resume file). relu(1 + r) bounds 1[r > 0], so a
  full-fit H ≤ 0.01 implies an empirical fit false-promotion fraction ≤ 0.01; the minibatch trajectory proves neither the
  final constraint nor held-out protection, so the full fit is scored separately at update 0 and 600.
- **Exact whole-membership gradient.** Pass 1 scores every occurrence of the four runs without a graph and forms dL/ds on
  a leaf vector through the weighted run means; pass 2 re-forwards every occurrence with autograd and back-propagates its
  coefficient (zero-gradient N windows included); one clip and one step per update; both arms run the same two forwards
  and one backward per occurrence.
- **One-way deployment.** On the original Base (frozen A08 + B3) P runs only Base-A4C runs are eligible (749 / 771);
  each run's unique rows (5,030 / 5,437 of 24,585 / 25,774) are scored, r = weighted mean; r > 0 relabels the run A5C.
  Every other label is copied; equal neighbours merge; complete sequences are scored with the original evaluator. Fit f
  is read on fold 1 − f, once, after the weights were committed.

## Checks (`checks.json`, attempt 2; attempt 1 kept as `checks_attempt1.json`)

- **Attempt 1 failed five check-only gates** (0.067 GPU h kept): the independent reference nearest map (A27's
  `nearest_rows`) is fold-local but was offset a second time on fold 1 (membership / deploy-mean references), the Init
  weighted mean was asserted bit-exact (−2 plus ~1e-15 float64 accumulation on 212 / 235 runs), the kink check ran in
  FP32 against FP64 expectations, and the low-clip dual fixture expected 0 where the recurrence correctly gives 0.004.
  The production two-pass gradient (C 1.9e-7, N 5.7e-8 relative), replay, repeat, forced-C, resume and real-movement
  gates had passed. Check-only repair 8fcdd5b (reference conventions asserted, FP64 tolerance 1e-12 with the exact
  head-score and exact Base-parity gates retained, float64 kink check, genuinely clipping fixture 0.0005 → 0 and the
  non-clipping 0.005 → 0.004 case, a CPU-only `cpucheck` stage that validated both folds without pixels). The training
  path, objectives, memberships and numerics are unchanged.
- **Attempt 2: all 38 enforced gates pass, none missing.**
  - **Inputs / lists / plan.** Pools, split, A08, q and P-run hashes as pinned; no recording or session crosses folds.
    Lists and eligible rows as asserted; memberships equal the repeated-sample loop on both folds (≤ 4.4e-16), weights
    sum to 1, rows unique and inside the fold, runs aligned with Base's sequences; plan sha256, explicit RNG, coverage
    and exposure as asserted. Update 0's frames equal an independent per-reference decode; markers equal A07's slots.
  - **Init.** The copy equals A08 bitwise; C and N start bit-identically; 513 head parameters at 0 / −2; every first-pass
    score exactly −2; online fit logits equal A31's cache within 1.4e-6; the patched forward equals the original
    bitwise; Init reproduces Base on every cached P run, sequence and A32 row (weighted means within 1.3e-15 of −2, all
    negative).
  - **Formula / dual.** Run means, A / B / H, C / N gradients (including zero gradients of inactive negative windows)
    equal an independent float64 loop (≤ 1e-8) for memberships of 1–129 occurrences and ±1e3 margins; the float64 kink
    check gives exact values at 1 + r = 0; the dual recurrence, clipping at 0 / 100 and the no-change case at H = α pass.
  - **Gradients.** With a check-only nonzero head weight, the production two-pass FP32 gradient equals one
    retained-graph backward over short runs (2–6 rows each) to 1.9e-7 (C) / 5.7e-8 (N); pass-1, replayed and retained
    scores agree exactly; gather vs original max-pool backward 2.3e-8.
  - **Mechanics.** Update 0: trunk, marker and projection gradients exactly 0 (zero head), head 21.0; updates 1–5
    growing nonzero gradients in all four groups (trunk 0.011 → 0.18 before clipping). The optimiser binds exactly the
    trainable tensors (400 state tensors). After 6 updates 376 / 395 trunk tensors, the marker stem, both projection
    tensors and the head moved; the 19 unchanged trunk tensors are 15 key-LayerNorm biases (round-off gradient
    ≤ 2.9e-8 of the trunk maximum; softmax invariance, A43's structural reason) and block 15's query-pool weight and
    three relative-position tables (exactly zero gradient, as A30 / A40 recorded). 6-update repeat, N's path forced to
    C's objective vs C, and 3 + 3 resume (λ restored) vs 6 are all 0 / 0; N differs from C (2.1e-4) with λ 0.999 → 0.994
    while H stayed 0.
  - **Cost.** 145 loaded modules equal their committed blobs; 0.137 s per occurrence (two passes), 2.5 GB peak,
    0.041 s per row for no-grad scoring; source-0 projection 3.98 of 6 GPU h (source 2 would need ≈ 3.45 h more).

## Actual training and final-fit constraint

The server update files contain contiguous u=0…599 for all four models: C/N fit0 each computed 14,778 window
occurrences and C/N fit1 each 15,528, totaling 60,612. Model/head/update/full-fit-cache hashes equal the committed
manifest. C/N share the exact plan, initial state and per-update occurrence count. Independent parameter comparison
against original A08 found movement in 391/395 trunk tensors, the marker tensor, both original projection tensors
and the scalar head in every model; the old five-class classifier is bit-identical. The four unchanged trunk tensors
are block15's structurally unused query-pool weight and relative-position tables. Saved tensors and loss/gradient
records are finite. Replay is zero; independent λ recurrence errors over all 600 updates are zero. No resume file
remains, and original A08 SHA remains `9b144925…035bbc`.

| Final fit only (not held-out efficacy) | Full protected-negative H | False promotions | Positive recovery | Final λ |
|---|---:|---:|---:|---:|
| C fit0 | 0 | 0 / 714 | 33 / 34 | 1 (inert) |
| N fit0 | 0.010329310118 | 2 / 714 | 34 / 34 | 8.320015481 |
| C fit1 | 0.103119757039 | 15 / 699 | 68 / 69 | 1 (inert) |
| N fit1 | 0.181141403206 | 35 / 699 | 68 / 69 | 13.484867912 |

Both N fits violate H≤0.01. Fit0's small excess is still a failure under the exact predeclared condition; fit1's
violation is substantial. Neither the multiplier nor the minibatch trajectory certifies final or held-out protection.
No tolerance, checkpoint, threshold, update count or objective was changed to repair this scientific result.


## Full readout and independent recount

The frozen models each scored only the unique original Base-A4C request rows, 5,030 / 5,437 per held-out fold
(20,934 policy forwards across both arms). The original Base table reproduces A32 and A43 exactly on both folds.
The original majority labels, fixed denominators, every P run (6,788 total), whole membership/multiplicity and
excluded-from-training other-class eligible runs are retained in the readout.

| Complete-sequence endpoint, fold0 / fold1 | Base | C | N |
|---|---|---|---|
| Multi segment F1@50 | .963906582 / .949863972 | .961456518 / .950787784 | .953660436 / .950340798 |
| Native macro F1 | .977556578 / .975831638 | .972028352 / .975831638 | .967144081 / .972660078 |
| Boundary recall | .967644085 / .968139774 | .964610718 / .968139774 | .958543984 / .967625899 |
| Boundary precision | .955566650 / .942000000 | .961209068 / .944360902 | .968335036 / .946706888 |
| Short FP | 73 / 105 | 66 / 101 | 59 / 102 |
| Missed short true | 41 / 35 | 51 / 35 | 67 / 37 |

Every originally non-A4C run label, including the 1,150 correct A5C runs, is bitwise retained by the one-way rule.
That is structural protection, not learned recognition. RLE merging still deletes 30 / 75 original boundaries under
C / N and short misses increase; fewer predicted cuts or short FPs do not establish improved segmentation.

`independent_audit.json` records a separate CPU implementation over the private score cache: directly reconstructed
nearest-centre maps (left ties), all-run unique rows and multiplicity weights, explicit repeated-sample means,
`math.fsum` weighted means, r>0 labels, full per-sample reconstruction, the **original** full evaluator, true-A5C
recording membership **inside** corrected runs, paired C/N fix/harm overlap and independent gate booleans. It does
not call the production weighted-mean, one-way or sequence-edit helpers and performs no model forward. Every
published row/count matches; errors are below 1e-10. It also independently rechecks the full-fit score caches,
all 4×600 logs, λ recurrences and final artifact hashes. Server score cache SHA:
`526ab49bc0b69af56b23f8a1757921695127b834e642761080b9f3f8bcf7a5ef`.
Private arrays, membership identifiers, per-case outputs and media remain on the server.

C passes the recovery/source condition but fails the correct-A4C floor (needs≥1,399 retained; keeps1,391),
fold0 and OOF net, fold0/mean segmentation, native floor and short-miss conditions. N fails those protection/utility
conditions as well as the boundary-recall condition and both full-fit H checks. The separate constraint-contribution
gate fails: additional A5C recall came with **more** A4C damage in both folds and lower F1. Source2 trigger=false.


## Interpretation, same-denominator A43 comparison and closed scope

| Same source0 held-out denominators | A5C recovered / 103 | Correct A4C harmed / 1,413 | Net | Mean full F1@50 | Short misses |
|---|---:|---:|---:|---:|---:|
| Base | 0 | 0 | 0 | .956885277 | 76 |
| A43 D | 25 | 112 | −87 | .944178842 | 114 |
| A43 M | 2 | 4 | −2 | .956306238 | 76 |
| A44 C | 12 | 22 | −10 | .956122151 | 86 |
| A44 N | 26 | 76 | −50 | .952000617 | 104 |

Compared with A43 D, N recovers one more A5C run with 36 fewer A4C harms, but still damages 76 correct A4C runs and
falls below Base in mean segmentation. C damages fewer A4C runs but also recovers fewer A5C; it is not a qualifying
full-system improvement. A43 M remains a low-recall result, not a solution. No cross-round per-case overlap is claimed.
The controlled causal contrast in **A44** is N−C: this fixed hinge/primal-dual recipe increased recall and harm,
rather than providing the intended risk protection. Common differences from A43 change both supervision unit and
readout/training package and cannot identify the contribution of any one component.

This closes only the fixed original-A08-embed scalar policy, original deployed-unit membership, 600-step C/N recipe,
source0 screen and r>0 rule. It does not disprove constrained learning generally, establish equivalence, identify LVOT,
or establish patient-independent/clinical continuous-scan performance. The two folds separate source sessions for the
new repair policy; A08 already trained on these recordings and majority clip codes are weak references. No extra seed,
threshold search, expert/pseudo supervision, dev/check/test/Pediatric read, A26 runner, Unity training or A45 was used.


## Resources and closeout

See `resources_closeout.json` and `ledger_closeout.jsonl` for the final actual-time accounting. GPU includes the failed
check, successful checks, all four training jobs/full-fit scores, one readout and process-bootstrap gaps. CPU preserves
prior active spans, corrects out the check wait, adds recovered active work between training polls and the Codex takeover,
independent cached-score audit and actual closing. Explicit idle waits and the Claude quota outage are excluded. Earlier
root research/review/heartbeat inspections were outside the worker ledger and not precisely timed; that scope is
explicitly disclosed, not recorded as zero. No future closing reserve is reported as actual use.

The one readout measured 608.239 GPU-compute seconds plus67.655 decode seconds; 29.055 ms per eligible row for one
policy, in addition to the original Base pass (A31 16.461 ms/window). Deployment adds an A08-size encoder plus513 head
parameters on eligible rows. Final server GPU is idle and no A44 model process remains. New A44 run/result artifacts
are about0.56GB, far below3GB; no persistent pixels or feature cache was created. Eight staged theory files and unrelated
dirty files were preserved; round-only commits, no push. The user authorized Codex to complete closeout after Claude's
monthly quota blocked the original worker; no weight, plan, training condition or trained result changed in the takeover.
