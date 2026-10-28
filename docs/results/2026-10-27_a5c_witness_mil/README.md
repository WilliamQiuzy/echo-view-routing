# A43: end-to-end A5C witness specialist with one-way correction, existential (M) vs dense (D) positive supervision (2026-10-28 UTC)

**Question.** Can a separately trained visual A5C witness specialist repair Base's A4C predictions without sacrificing
its existing A5C labels? And does existential bag supervision (M) beat forcing every window of an A5C recording
positive (D)?

**Answer at source seed 0: no on both counts. The round closes without source 2.**

| Held-out folds 0 / 1 (OOF) | Base | D (dense) | M (existential) |
|---|---|---|---|
| S1 fixed (true A5C read as A4C; 34 / 69 runs) | 0 | **16 / 9 (25 / 103)**, 16 true-A5C recordings | 1 / 1 (2 / 103), 2 recordings |
| Correct A4C promoted to A5C (of 714 / 699) | 0 | **95 / 17 (112)**; kept 1,301 / 1,413 | 4 / 0 (4); kept 1,409 / 1,413 |
| Net fixed runs | 0 | −79 / −8 (−87) | −3 / +1 (−2) |
| Correct A5C kept (protected by the rule) | 561 / 589 | 561 / 589 (1,150 / 1,150) | 561 / 589 (1,150 / 1,150) |
| Complete multi segment F1@50 | 0.9639 / 0.9499 | 0.9386 / 0.9498 (−0.0253 / −0.0001) | 0.9627 / 0.9499 (−0.0012 / 0.0000) |
| Native macro-F1 delta | | −0.0190 / −0.0116 | +0.0002 / 0.0000 |
| Boundary recall delta | | −0.0293 / −0.0005 | 0.0000 / 0.0000 |
| Missed short true segments (OOF) | 76 | 114 | 76 |
| Base boundaries deleted by merge | 0 | 102 / 15 | 2 / 0 |

- **Target fixes against A4C damage.** D fixed 25 of 103 targets (on the same folds A41's branches fixed 12 / 17 and
  A42's S 4). The same rule promoted 112 correct A4C runs, so net fixed is −87. M kept almost every
  correct A4C run but fixed only 2 targets.
- **Structural A5C preservation is not system protection.** Every originally non-A4C run label was unchanged in both
  arms, including all 1,150 correct A5C runs. That holds by the decision rule and is not a learned or empirical result.
  Even so, D's relabelled A4C runs merged into neighbouring segments: 117 Base boundaries were deleted, fold-0 segment
  F1 fell by 0.025, and missed short true segments rose from 76 to 114.
- **M − D.** The predeclared MIL signal fails in the opposite direction on S1: M fixed 15 / 8 fewer targets than D
  (needed ≥ +1 per fold and ≥ +5 OOF). M was better on net (+76 / +9), segment F1 (+0.024 / +0.0001) and correct A4C
  kept (+108), but only because it changed almost nothing (6 runs OOF against D's 138).
- **Absolute package gate (either arm).** D fails correct A4C (1,301 < 1,399), net (each fold and OOF), segment F1,
  native, boundary recall and missed short. M fails S1 (2 < 11 runs; 2 < 3 recordings), net (fold 0 −3; OOF −2 < +5)
  and the F1 rule (fold 0 −0.0012; mean −0.0006 < +0.001). Neither arm passed, so source 2 was not run. Results are
  aggregate and private per-case outputs stay on the server.

## Predecessors and the changed condition

- A42 fixed 4 / 103 S1 runs (3 in its rewired control) but broke 56 correct runs, 45 A5C; its pair term was nearly
  satisfied from update 0. A31's error emphasis / last-block changes mainly moved the A4C / A5C boundary. A40 (masked
  HOG) and A41 (new pretrained representation) gave no transferable A5C corrections with system protection. None is
  reopened.
- A34 already trained whole-native bags (mean / power means), as a copied five-class linear head on frozen global A08
  embeddings that relabelled every Base class. R07 learned spatial / temporal attention on frozen E1 tokens; A39 used
  fixed mean / max of frozen A08 non-CLS tokens; V02 trained patch readouts end to end. A43 is not the first MIL
  experiment, and attention, patches and encoder unfreezing are not individually new here.
- **Changed condition.** Binary A5C witness learning: a gated-attention readout of the 49 marker-time final tokens and
  the whole A08 trunk, trained end to end under native-recording supervision, applied *only to original Base-A4C runs*.
  M against D changes only the positive term (same architecture, images, negative loss, plan, compute and one-way
  deployment). An A5C-coded bag containing a usable witness is a hypothesis, not an expert annotation; an A4C code does
  not certify LVOT absence; attention is not anatomy. Method references for motivation only: Ilse et al. ICML 2018
  (gated attention MIL); Durand et al. CVPR 2017 (WILDCAT).
- **Scope.** Source seed 0 (A08 all_s0) on A31's two session folds; four models (D and M per fit fold). No development,
  check, test or Pediatric data. A08 trained on these recordings, so the folds hold out only the specialist's sessions.
  Labels are EV9V clip codes; expert records remain 0.
- **Records.** Configuration `configs/experiments/video_a5c_witness_mil_v1.yaml` (e839c34, committed before any bag,
  plan, pixel, loss or model computation; amended before training in 5eb5da3). Code 845f38a + 5eb5da3. Checks 0c4987b.
  Weights 5b95aa1 (frozen before the readout). Task sheet `claude_prompts/2026-10-27-a5c-witness-mil-training.md`
  (sha256 95f103d3…).

## Design

- **Native bags** (per fit pool; kinds, clip codes and the window grid only). Each native A4C / A5C recording is one bag
  of *all* its original stride-5 request windows (16 samples at 10 Hz, 224 crop, original gamma / normalisation /
  marker); no crop, cap or positive subsampling. Window weight = samples whose nearest centre it is / recording samples.
  Asserted before any pixel: 200 A4C / 67 A5C bags (pool 0), 200 / 75 (pool 1); mean windows 11.59 / 10.09 and
  12.00 / 12.37, maxima 102 / 100 and 72 / 58.
- **Plan.** Each update draws one A4C bag then one A5C bag uniformly (RNG [20261027, 43, s, f, u]), full bags, A4C
  first. The 600 draws were fixed before pixels and identical for D and M: 12,498 / 14,231 windows (up to 118 / 98 per
  update); 191 / 191 of 200 A4C and all 67 / 75 A5C recordings drawn. No mistake, S1, harm or session selection.
- **Specialist.** A bit-identical A08 all_s0 copy runs its full original marker forward; a hook on `encoder.norm`
  (removed in `finally`) keeps the graph of the 1 + 392 final tokens. The 49 spatial tokens at temporal index
  marker_slot // 2 feed gated attention (V, U 768 → 128 with bias; w_a 128 → 1 without) and a scalar scorer (768 → 1):
  197,761 parameters, scorer weight 0 / bias −2 at Init (every initial score −2, so Init = Base exactly). The unused 512
  projection and five-class classifier are frozen. Trained: encoder trunk (34,230,144; lr 1e-5), marker stem (14,112;
  1e-4), attention + scorer (1e-3); AdamW wd 1e-4, warm-up 60 then cosine to 600, global clip 1; FP32, TF32 off,
  `warn_only=False`, A42's gather skip pools on training copies; eval mode.
- **Losses.** A4C (both arms) Σ v_j softplus(s_j). A5C: D Σ v_j softplus(−s_j); M softplus(−max_j s_j). Total
  π4 · A4C + π5 · A5C with π the fit pool's native class shares (200 / 267, 67 / 267; 200 / 275, 75 / 275).
- **Exact whole-bag gradient.** Pass 1 scores every window of both bags without a graph and forms dL/ds on a leaf
  vector; pass 2 re-forwards *every* window with autograd and back-propagates its dL/ds (zero-gradient M windows
  included); one clip and one step per update. Both arms run the same two forwards and one backward per window.
- **One-way deployment.** On the original Base (frozen A08 + B3) P runs, only Base-A4C runs are eligible (749 / 771).
  Each run's unique original request rows (5,030 / 5,437 of 24,585 / 25,774) are scored; r = max score; r > 0 relabels
  the run A5C. Every other label is copied; equal neighbours merge; complete sequences are scored with the original
  evaluator. Fit f is read on fold 1 − f, once, after the weights were committed.

## Checks (`checks.json`, attempt 2; attempt 1 kept)

- **Attempt 1 failed one gate, `real_update`.** After 6 updates every group moved (380 / 395 trunk tensors, marker
  stem, all head tensors). 15 trunk tensors stayed bitwise unchanged: every key LayerNorm bias
  (`blocks.*.attn.pool_k.norm_act.0.bias`) except block 4's. In torchvision's MViT the key pool's `norm_act` is a single
  LayerNorm applied after the CLS token is re-attached, and pooled keys enter only through q·kᵀ before softmax (the
  relative-position terms depend on q). Its bias therefore shifts each query's logits by a constant: the exact gradient
  is 0 and only round-off reaches it (max |grad| ≤ 6.1e-12, ≤ 2.7e-11 of the trunk maximum). The gate had admitted only
  frozen or exactly-zero-gradient tensors. Check-only repair 5eb5da3 (configuration amended): an unchanged trainable
  tensor passes only with this structural reason and ≤ 1e-6 × the trunk maximum. The training path and every other
  gate and limit are unchanged. These biases are not reported as trained movement (four stay bitwise unchanged after
  600 M updates).
- **Attempt 2: all 37 enforced gates pass, none missing.**
  - **Bags / plan / pixels.** Counts and feasibility values as asserted; whole bags in pool order, one recording each;
    weights equal an independent nearest-centre loop; no recording or session crosses folds. Frames equal an
    independent per-reference decode; markers equal A07's marker slots.
  - **Init.** The copy equals A08 bitwise; D and M start bit-identically. Every first-pass score on update 0's 29 fit
    windows is exactly −2. The copy's online fit logits equal A31's cache within 1.4e-6; the patched forward equals the
    original bitwise. Init reproduces Base on every cached P run, sequence and table, and A32 / A42 Base rows (a CPU dry
    run of the readout gave the same; no held-out pixel).
  - **Tokens / attention / loss.** Tokens 1 × 393 × 768 on the network's 8 × 7 × 7 grid; selection equals an explicit
    loop; the time rule equals an independent nearest-time loop (odd slot 7 → index 3); the hook is removed after a
    forward and after an error. Attention / score equal a float64 loop (≤ 4.4e-16) and are invariant to token order.
    D / M losses and score gradients equal an independent float64 loop (≤ 2.8e-17) for bags of 1–102 windows, unequal
    weights, 2- and 3-way exact max ties and ±1e3 margins; one-instance M = D.
  - **Gradients.** With a check-only nonzero scorer weight, the production two-pass FP32 gradient equals one
    retained-graph backward over short fit bags (5 + 4 windows) to 1.6e-7 relative (D and M); pass-1, replayed and
    retained scores agree exactly; gather vs original max-pool backward 5.9e-10.
  - **Mechanics.** Update 0: trunk, marker and attention gradients exactly 0, scorer nonzero. Updates 1–5: all four
    groups nonzero (trunk 0.14 → 2.4 before clipping). The optimiser binds exactly the trainable tensors. 6-update
    repeat, M forced to the dense positive against D, and 3 + 3 resume against 6 are all 0 / 0; M differs from D.
  - **Code / cost.** All 143 loaded modules equal their committed blobs. 0.140 s per window (two passes), 2.5 GB peak;
    projection 3.21 of 6 GPU h.

## Training (fit-only; `weights_s0.json`, committed 5b95aa1 before the readout)

- Four models, 600 real two-pass updates each, no resume, pass-1 / replay score difference 0 in all 2,400 updates,
  2.3 GB peak. Final sha256: D_f0 6abf2f9a, M_f0 df84439a, D_f1 cced5478, M_f1 5b166131 (heads 645038c4 / ebb6dba8 /
  fe5fe669 / 0fa8a44b). Trunk drift 0.42–0.46% relative L2, marker stem 26–36%; scorer bias stayed −1.98 while its
  weight grew to norm 1.1–1.2.
- **Each arm fit its own objective.** Over the last 60 updates: D's A5C windows scored positive 99% (fit 0) / 99%
  (fit 1), its A4C windows 0.3% / 3.7% (A4C loss 0.010 / 0.162); M's A5C windows 73% / 58% and A4C 0.0% / 0.3%,
  with max-witness loss 0.07 / 0.10 and a bag-max margin of about 16 against D's 13–14. On the fixed fit monitoring
  bags (first 8 A4C / 8 A5C) at update 600 every A5C bag had r > 0 in all four models; D_f1 also flagged 1 of 8 A4C
  bags. Nothing was selected or retuned from these.

## Held-out detail

- Fold 0 has 3,384 P runs (109 Base errors), fold 1 3,404 (145). S1: 34 / 69 runs from 12 / 26 A5C-coded source
  recordings. Base reproduced A32's rows and A42's Base cells exactly in both folds.
- **Eligible-run scores.** Read on fold 0, D (from fit 1) gave r > 0 to 16 / 34 true-A5C and 95 / 714 true-A4C
  eligible runs (medians −2.0 and −7.2). Read on fold 1, D (from fit 0) gave 9 / 69 and 17 / 699, plus 1 / 3 other.
  M gave 1 / 34 and 4 / 714, then 1 / 69 and 0 / 699; its median r for true-A5C targets (−8.9 / −9.2) matched true A4C
  (−9.1 / −9.2). The witness M learned on native fit bags almost never fired on held-out Base-A4C runs.
- **Where D's changes landed.** All 138 changed runs were A4C → A5C. Fold 0: 16 fixed (9 short ≤ 16 samples), 95
  broken (77 in constructed streams, 18 native, 24 short); FN 81 → 186 and missed short 41 → 80. Fold 1: 9 fixed, 17
  broken, segment F1 −0.0001. S1 fixes came from 8 + 8 true-A5C recordings in 8 + 6 sessions (a run counts only its
  A5C-coded sources); broken runs touched 184 recordings in 131 sessions.
- The large fold-0 harm comes from D trained on fit fold 1, the model whose own fit A4C loss stayed highest. One source
  seed and two folds cannot separate a fold effect from a model effect.

## Interpretation and limits

- On these folds, dense positive supervision produced a witness that promotes many correct A4C runs alongside real
  targets. Existential supervision produced a separated fit witness that did not transfer to held-out Base-A4C runs.
  Neither is a usable one-way corrector, and MIL did not outperform dense supervision on target fixes. This is not
  equivalence and does not show that MIL, local tokens or a one-way rule cannot help under another design.
- Clip codes are weak labels; a positive bag is assumed, not shown, to contain an LVOT witness. Local tokens carry
  global receptive context, so attention is not anatomical localisation. A08 had already trained on these recordings.
  Training used whole native recordings, while deployment scores Base-predicted runs, often short or inside constructed
  streams: a bag-to-interval shift. Sessions are not verified patients. There is no expert truth.

## Cost (ledger `runs/video_a5c_witness_mil_v1/ledger.jsonl`, server; append-only)

- GPU 2.37 of 6 h. Stage / model timers 2.362 h: checks 0.106 (attempt 1 failed, 0.053), training 2.076 (0.48 / 0.48 /
  0.55 / 0.56), readout 0.181 (28.5 ms per eligible row per specialist; Base's pass 16.5 ms per window). Those timers
  start inside each process and exclude its imports, contexts, fold build and final publish: 5.6–7.7 s per process,
  0.0068 h in total, measured after closure from each launch log's birth time to its last output's modification time
  and ledgered separately. No source 2.
- CPU 1.01 of 3 h as of 05:59 UTC, metered from actual intervals only (idle training / readout polls are not charged):
  - wall-clock agent spans 1–6 (0.755 h) and CPU jobs (dry run, freeze, cache recount; 0.005 h);
  - active intervals recovered after closure from the session stream timestamps (tool result → next tool call):
    0.167 h inside the training-poll window 03:31:00–05:33:59 UTC, which span 3 had left unmetered (README draft
    37.8 s, recount script 35.6 s, 130 status / poll turns), and 0.018 h inside the readout-poll window;
  - 0.003 h of CPU-process bootstrap outside the dry-run / freeze timers, and this closeout's active span (the part
    after 05:59 UTC is in the ledger).
  - The wall-clock spans are not pure active time: span 2 includes 0.037 h of polling the two check jobs and span 1 a
    120 s stalled `git commit`; they stay as recorded. Charging both poll windows in full would add at most 2.05 h +
    0.17 h of mostly idle time; that is a loose upper bound, not usage.
- Codex's metadata feasibility, reviews and independent audits (including its final cached-score / full-sequence /
  gate audit: 93 assertions, 0 discrepancies) ran outside this ledger.
- New disk 0.54 GB (four final models + heads, logs, private readout cache); no resume or pixel cache left; 143 GB free.
- Files: `checks.json`, `checks_attempt1.json`, `weights_s0.json`, `readout_s0.json` (aggregate). The private row
  scores / run labels (`readout_cache_s0.npz`, sha256 in `readout_s0.json`) stay on the server; an independent
  explicit-loop recount from them matched every run-level total.
