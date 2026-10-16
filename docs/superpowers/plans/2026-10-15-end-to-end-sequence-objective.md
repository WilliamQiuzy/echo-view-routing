# End-to-End Short-Sequence Objective Implementation Plan — A07

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax. Claude is the intended executor. The owner authorizes implementation, repairs, reviews, checks, all nine bounded trainings, staged evaluation, fact-checking and round-only commits without waiting for their return.

**Goal:** Test whether segment and class-order supervision can improve complete view segmentation when its gradients update the strong video encoder itself, rather than only a linear head on frozen features.

**Architecture:** Keep V01 Ordered's TargetMViT, target marker, five-class CLS head and inference interface. Three matched continuations read identical groups of neighbouring target windows: F uses target-wise CE, B balances runs within each group, and O adds A05's no-blank transcript-forward loss to B. Unlike A05, gradients reach the RGB video encoder after the inherited warm-up. No new model parameters or downstream temporal head.

**Tech Stack:** Existing PyTorch/torchvision MViTv2-S, 16-frame 224×224 RGB pipeline, training-only sequence recipes, A05's tested losses, existing evaluator and change-point decoder. English aggregate README; private manifests and outputs on the server.

---

## 1. Owner decision, evidence and limits

A06-R1 is closed. Do not run Breakfast FACT, extend GTEA training or reinterpret its calibration losses as a gain. Keep V01 frozen Ordered + B3 as the strong reference.

The new question is narrow: **can a short sequence's segment/order objective change the video representation in a way that recovers missing class-aware segments without losing native recognition?** This is a hypothesis, not a diagnosed cause or a promised improvement.

Read AGENTS.md, CLAUDE.md, `docs/EXPERIMENT_INDEX.md` and these predecessors before implementation:

- **V01:** multi-frame input gives recognition and system gains over the E1 image route. Keep its successful RGB/CLS architecture and source checkpoints.
- **V02 and A02/A03:** continued training, replacement readouts and anchor-label supervision do not show qualifying segment gains. A07 does not replace the CLS readout or add time-token supervision.
- **V03:** denser requests improve boundary offsets without a consistent segment gain. A07 keeps stride 5; no density/interpolation search.
- **A01 and A06-R1:** adding conventional temporal heads or this FACT configuration to frozen video features did not beat the strong reference. A07 adds neither.
- **A04:** many corrected mixed-window labels were inside already-matched segments. Judge recovered/lost segments and FP/FN, not local accuracy alone. This does not prove an encoder loss is the cause or cure.
- **A05:** F/B/O trained only 2,565-parameter linear heads on frozen video features; neither B nor O had qualifying gains. A07 explicitly changes the trainable representation and uses short raw-video target sequences. It is **not** an exact A05 repetition or a clean frozen-versus-unfrozen causal comparison: sequence length, sampling and optimization also differ historically. Only the new matched arms isolate their objectives.

The forward-sum idea is established: [CTC, Graves et al. 2006](https://www.cs.toronto.edu/~graves/icml_2006.pdf) and [Li et al. 2019](https://arxiv.org/abs/1909.13155). The implemented graph remains A05's five-class, no-blank stay/advance graph, not standard CTC or a reproduction of those papers. No architectural novelty is claimed. A transcript loss does not directly optimize segment IoU, duration or post-decoder F1.

This is one new bounded comparison. It does not reopen earlier closed searches or authorize an automatic next round.

## 2. Sources and isolation

Local repository: `/Users/williamqiu/Desktop/Harvard/Capstone/echo-view-routing`.
Server: `echo-l40s:/home/william/echo-view-routing`.

Pin these files in the new config before generating a new training plan:

| Source | SHA256 |
|---|---|
| `configs/experiments/raw_video_context_v1.yaml` | `832694f4ea1ec4c7d1811b59c0a6803495645e9ff8be1db80fe28d2c049af485` |
| `configs/experiments/video_dense_supervision_v1.yaml` | `49f8782ad0d7702f65244dd6b637f2fc20dc5dd0aab40c2de4389d69f0eb1087` |
| `configs/experiments/video_feature_temporal_v1.yaml` | `3b065e7fe908ca7432ba0a407fd6b82432882c1d72dabd1d22b899b79d790690` |
| `docs/results/2026-10-10_raw_video_context/summary.json` | `a521d47e5a5618e7000fcd877ac58dc7ebc16f488a48b229f20bccfb18f333ab` |
| `docs/results/2026-10-14_segment_objective/summary.json` | `047f715780106ca1166fa2d4b2fe9ed364e6d6426ebf42c4f442e9db867dd29a` |
| `echo_routing/temporal/losses/segment_objective.py` | `93ae515735afdfb8b4b92780981559456516acdfd114f028ee90b7c92c479b85` |

Start all three arms of seed k from exactly that seed's **V01 Ordered final** state, including encoder/projection, marker and classifier. No reinitialization:

| Seed | Server-relative file | SHA256 |
|---|---|---|
| 0 | `models_frozen/raw_video_context_v1/ordered_s0/final.pt` | `80f13a42e628fe2647634b79e59d5b8486fee5c3e0ae45f70bcc667c41258505` |
| 1 | `models_frozen/raw_video_context_v1/ordered_s1/final.pt` | `74276323ababf78fcb578be28d3172f3c220b3b7e66c2ff96a2072f39210c26f` |
| 2 | `models_frozen/raw_video_context_v1/ordered_s2/final.pt` | `923eb3afb0d846b1361897dbb63bb5aecf22401ce4b305b44fe990d48edc7f2c` |

Use family5_v1, numeric frame order, the same 3,364 training recordings, A01's fixed 1,500 training-stream recipes and existing session split. Reconstruct training RGB references using the allowlisted sequence/recipe path in `scripts/prepare_video_feature_temporal.py` and `echo_routing/temporal/video_sequence_features.py`; do not run its feature-extraction stage. Pin the original recipe/protocol digests from A01. This round creates a new target-group manifest, not new recordings or streams.

Filter metadata by the training allowlist before reading labels or paths. Do not read development arrays, labels or media before all nine runs finish and freeze. Hashing known source files is allowed. Test remains sealed and Pediatric unread. No relabeling, excluded failure clips or development-case sampling.

Keep 224×224 V01 preprocessing exactly: the existing numeric frame references, gamma per recipe fragment, short-side-256/centre-crop-224 geometry and EchoPrime 0–255 mean/std. Assemble an entire stream's references **before** constructing its windows; a training chunk must not truncate context at its own ends. Do not revive fan224 or manufacture video transitions by interpolating images.

## 3. Fixed training groups: eight video windows per update

An ordered group contains **four neighbouring request centres** on the same stride-5 request grid used at inference. Each request gets its own ordinary 16-frame RGB window and target marker, through unchanged TargetMViT.forward. The eight windows form one normal batch; reshape their five-class CLS logits to `(2,4,5)` only for the loss.

Every update has one native group and one boundary-containing constructed-stream group: four native targets and four constructed targets. This matches V01's eight-window compute count and native/mixed target balance, but not its historical target sampling. A07 F is the necessary control for that new sampling.

Freeze the following sampling algorithm before counting its outputs:

1. Obtain centres/samples/markers from `video_sequence_features.windows(T)`. A native group is four consecutive grid centres. If a native sequence has fewer than four centres, repeat its last available centre to reach four; retain it and report repeated targets. Never form a constructed group with duplicate centres.
2. For a constructed sequence, enumerate consecutive four-centre groups whose **four sampled target labels contain at least one change**. A group is eligible on those four labels only. The loss transcript is their collapsed label sequence; do not insert a reference class that exists only between sampled targets. Report the training reference runs skipped by this sampling. These are short sampled sequences, not whole-stream transcripts.
3. Native selection: uniform over source training recordings, then uniform over their starts. Constructed selection: uniform over the fixed streams with at least one eligible start, then uniform over eligible starts. A same-view-only stream has no eligible start and remains in the source accounting; no replacement streams are created. Native groups supply the pure-window control. Report eligible/ineligible counts and source exposure, without using those counts as a new performance gate.
4. Use `numpy.default_rng([20261015, seed, update])`, drawing the native recording/start, then the eligible stream/start, in that order. Sort source lists in pinned protocol/bank order, and eligible starts by centre index. No rejection sampling based on model scores or development results.
5. The same seed's F/B/O use bit-identical references, targets, markers, pixels, group boundaries and batch order. All 3,750 manifests are prepared before training; save source/pixel/epoch digests and class/run/length/exposure counts on the server. Repeated sources/targets do not add independent cases.

This deliberately teaches short-sequence class structure; it does not model full-exam order, confirm natural view transitions, or create new clinical annotations.

## 4. Exact objectives and optimization

Let z be `(2,4,5)` CLS logits, p its softmax, e[n,t] the CE for each group's sampled target, and R_n its maximal equal-label runs. All class weights are **1**, as in V01's unweighted CE; do not import A05's class weights.

- **F (target CE):** `L_F = mean_(n,t) e[n,t]`.
- **B (run balanced):** `L_B = sum_n sum_(r in R_n) mean_(t in r) e[n,t] / sum_n |R_n|`.
- **O (candidate):** `q_n = transcript_nll(logp[n], transcript_n)` using the pinned A05 routine (already divided by T=4); `L_order = sum_n |R_n| * q_n / sum_n |R_n|`; `L_O = (L_B + 0.5 * L_order) / 1.5`.

Use `segment_objective.objective(arm, logits, runs, alpha=ones(5))` unchanged if it implements these formulas exactly. Its fp32 log-softmax followed by float64 accumulation applies to every arm. Runs merge adjacent equal labels; X-Y-X retains three states. All positive-duration stay/advance paths are summed, not maximized; no blank, zero-duration state or label-constrained inference.

All arms compute the same model forward once per update, and compute/log the same detached loss components without extra stochastic forwards. Only the chosen differentiable objective changes. On batches where both groups contain a single class, F, B and O must agree in value and logit gradients. The native part of a mixed batch can receive a different global normalization through the run count; report the equations rather than claiming equal per-recording weights across arms.

- Seeds 0/1/2; F, B, O per seed; nine continuations of **3,750 updates**, final weights only. No validation, epoch choice or training-set score gate.
- Use V01/A02's optimizer and schedule unchanged: AdamW, encoder lr 1e-5, marker/classifier lr 1e-4, weight decay 1e-4, betas .9/.999, eps 1e-8, clip 1; warm-up 375 updates and inherited cosine. Freeze encoder parameters/lr for those first 375 updates but keep autograd through it to the marker, then unfreeze every encoder parameter including the 512-d projection.
- Start a fresh optimizer and update count for each continuation. Preserve source dropout/stochastic depth, fp32/TF32-off and recorded MViT CUDA nondeterminism. Complete state/RNG restoration is mandatory; do not promise bitwise weights after nondeterministic pooling backward.
- Reset `torch.manual_seed(seed)` and all CUDA generators identically before each arm's first update, independently of the manifest RNG. From equal source and RNG state, test equal forward logits and RNG consumption before objective updates diverge. Save CPU/CUDA RNG state in every resume checkpoint.
- No time-token head, FACT/ASRF, teacher, distillation, contrastive term, residual, gate or new decoder. No parameter, coefficient, sampling or update-count sweep.

## 5. Files and execution

Create only the focused A07 files:

- `configs/experiments/video_sequence_objective_v1.yaml` — source pins, sampling/equations, all guards, comparisons, budget and reporting rules.
- `scripts/prepare_video_sequence_objective.py` — allowlisted training references, group manifests and checksums; no features or development loading.
- `scripts/train_video_sequence_objective.py` — strict source load, F/B/O updates, checks/queue/resume/ledger/freeze.
- `scripts/report_video_sequence_objective.py` — safe calibration, committed choices barrier, one selection evaluation and aggregate report.
- `tests/test_video_sequence_objective_data.py`, `tests/test_video_sequence_objective_training.py`, `tests/test_video_sequence_objective_report.py`.
- `docs/results/2026-10-15_video_sequence_objective/` — English README, tables/summary, curves, preparation/check/review/fact-check records.
- Server-only `runs/video_sequence_objective_v1/`, `models_frozen/video_sequence_objective_v1/{frame,segment,order}_s{0,1,2}/`.

### Task 1 — Freeze scope and manifests

- [ ] Commit the config and exact scientific rules before new plan counts or training; then write meaningful failing tests and implement preparation. Pin the reused helper source digests. Do not modify source experiments or official code.
- [ ] Test four-centre ordering, native repeats, X-Y-X, same-class recording joins, eligibility, whole-stream context, class/marker alignment and allowlist-before-label access. A label or transcript change with fixed RGB/marker cannot change forward logits.
- [ ] Reproduce manifests/digests twice per seed; verify same-seed F/B/O pairing and strict final source weights. Check zero-update predictions against the source forward on fixed training inputs, max logit error ≤1e-4 and identical argmax.

### Task 2 — Implementation checks and direct launch

- [ ] Reuse A05's brute-force DP and gradient tests; add T=4 tests, all-one-class value/gradient identity and an integration test proving O's order gradients reach the RGB encoder, marker and classifier after unfreezing. Test optimizer binding and frozen-layer immutability during warm-up.
- [ ] Compare six updates/repeat and 3+3 resume, including a state/schedule check across update 375's frozen-to-unfrozen transition as in A02. Use the recorded V01/A02 state-restoration and MViT nondeterminism policy; if it needs a drift allowance, derive/document it from paired reruns **before formal training**, not from final performance. Do not silently relax a failing mathematical/alignment check.
- [ ] Run 100-update finite-loss/finite-gradient checks per arm on fixed training groups. Save actual loss/fit curves; do **not** demand perfect segment fit or token activation. This is not an efficacy gate.
- [ ] Measure 50 full unfrozen updates per arm after warm-up, including actual RGB loading, all objectives and backward, with peak memory. One independent implementation review must check math, grouping, gradients, isolation, resume, evaluation barrier and budget. Fix substantive implementation issues and recheck, then commit records and start the queue without owner confirmation.
- [ ] Total cap: **8 GPU hours and 4 CPU execution hours**, including preparation/check failures, training, evaluation and fact-checking. Before launch require spent + 3×each measured arm's training projection + 1.25 GPU h for staged evaluation/fact-check + 0.25 GPU h overhead ≤8 GPU h. Before each later run use **spent + projected remaining runs + remaining reserves**, without double-counting completed runs; project with the larger of timing estimates and observed per-run cost. If it cannot fit, preserve the measured estimate and stop before the dependent run; do not cut an arm, shorten training, change precision or add budget.
- [ ] Run F_s0/B_s0/O_s0, then seed 1 and 2. Resume only complete matching identities, never by directory name alone. Verify actual PID/log/update progress before reporting training started. Freeze all nine final states/manifests before opening development data.

### Task 3 — Staged evaluation and decisive comparison

- [ ] Keep the **ordinary single-window CLS inference**: no knowledge of training groups, labels, reference boundaries/transcripts or native/mixed flags. Same stride-5 centres, nearest-centre 10 Hz mapping and decoder interface as V01.
- [ ] Use A03's half-filtered metadata reader (`trained_token_readout.half_index/meta_stream/half_sets`) and bounded RGB-loading pattern. Do not call older helpers that eagerly read both halves. Re-infer each seed's frozen V01 source on a fixed calibration subset for parity (≤1e-4, argmax equal), then reproduce its archived calibration row. Postpone selection/reference-array reads until the choices barrier.
- [ ] Select each arm/seed's existing B3 grid only on calibration: h=[5,7,9,11,15], q=[.80,.85,.90,.95,.97,.99,.995,.999], NMS=5. Feasible means native macro-F1≥.980 and multi boundary recall≥.95; maximize multi segment F1@50, ties by original grid order. Infeasible is failed; 9/.95 can be diagnostic only. Commit choices, source identities and a verifiable Git proof before any new selection inference/labels; then evaluate selection once. Never compute a selection grid.
- [ ] **Primary O−B:** multi segment F1@50, per seed and half. Mandatory B−F, O−F, O−frozen Ordered and B−frozen Ordered. Report F versus frozen Ordered separately so sampling/continuation effects are visible. Include archived A01 ASRF/Linear, A05 and A06-R1 rows without retuning or re-ranking their checkpoints.
- [ ] Same model outputs also get direct argmax and the source Ordered seed's fixed numerical B3 setting as sensitivities. Keep these separate from each arm's calibrated setting; no sensitivity chooses the winner.
- [ ] Use the inherited 2,000-draw paired bootstrap, seed 20261002, native/session and stream approximations and strict three-seed same-side exclusion-of-zero rule. Give individual seed CIs. No consistent difference is not equality. Shared source clips across streams and seeds remain dependent.
- [ ] Report native/pair/multi separately; TP/FP/FN; recovered/newly lost matched reference segments versus B/F/frozen Ordered; ≤16-sample units and A4C/A5C X-Y-X middles; native macro-F1/per-class P/R and A4C↔A5C flips; native switches, boundary P/R/F1, same-view join cuts and transcript edit distance. Deduplicate by source recording/session; flag counts from <10 sessions. Local target accuracy or improved edit distance alone is not a segment win.

### Task 4 — Interpret, verify, record and stop

- [ ] **Full success:** O−B, O−F and O−frozen Ordered multi segment F1@50 are consistent positive in both halves; O satisfies every seed/half feasibility, native and recall constraint and both original mean targets. This is evidence only on reused development data, not independent or clinical validation.
- [ ] **Objective signal without full success:** O−B and O−F are consistent positive in at least one half, not consistent negative in the other; distinguish this from a win over frozen Ordered. Still close this fixed configuration after reporting.
- [ ] **Balancing/continuation:** if B−F qualifies but O−B does not, credit balancing only; if F itself improves against the source while O−F does not, credit this sampling/continuation condition, not the order term. No causal attribution across historical A05 and A07.
- [ ] **Trade-off/no qualifying gain:** report rescued/lost segments, FP/FN, native/recall failures and local-only changes explicitly. Better window accuracy, a single favorable seed, a mean crossing a goal or a CI touching zero does not override the primary rules. All readings may coexist with descriptive observations; do not force a catch-all that hides a real unfavorable result.
- [ ] Original targets stay selection≥0.944409687184662 and calibration≥0.9821421709894332 (three-seed multi means), every seed/half recall≥.95 and native macro-F1≥.973/.980. Also report calibration sensitivity target .9838935784747763. Crossing the old targets alone is insufficient to claim improvement over the stronger V01 reference.
- [ ] Independently fact-check saved outputs, all main rows/CIs/decisions, source identities, grouping, gradient math and private-ID/media exclusion. Fix report bugs transparently and preserve superseded artifacts. Update A07 from pending to actual in the experiment index, with frozen identities, result commits, constraints and closed scope. Commit only round files; no push.

## 6. Authorization and protected state

The owner has authorized the full task above, including automatically proceeding from successful engineering checks to all nine trainings, calibration, committed-choice barrier, one selection evaluation and report. Do not stop merely to ask whether to train. Repair code within the fixed scientific scope, record changes and repeat affected engineering checks. Stop dependent work only for a real source/math/isolation/budget failure or an unresolvable implementation error; do not silently invent new scientific gates or fallback rules.

No automatic follow-on experiment is authorized. Test stays sealed; Pediatric unread; public labels unchanged; paper paused. IDs, raw frames and per-case outputs remain server-only. Preserve the eight staged theory files, license drafts and all unrelated dirty files. Do not clean their directories or move the repository. The repaired local `.venv` already points outside iCloud; use it as-is and keep its backup.

Report truthful stages: prepared, checks passed, actual training running, nine models frozen, calibrated, selection evaluated, fact-checked and committed. A written plan is not training evidence.

## 7. Plan review

2026-10-15 (America/New_York): a separate read-only plan reviewer inspected the predecessor aggregate reports, loss helper, source pins, grouping, initialization/optimizer and evaluation barrier. Verdict: **Approved; no required fixes**. Its three advisory clarifications are incorporated above: remaining-cost budget arithmetic, identical per-arm model RNG, and a resume check crossing the unfreeze boundary. A local CPU design check also confirmed equal pure-group F/B/O values and logit gradients, and the T=4 X-Y-X forward-sum probability against its three explicit paths (error 2.3e-16). These are plan/mathematics checks, not model training, source-checkpoint validation or an A07 result. Claude must still complete the formal implementation checks before launching.
