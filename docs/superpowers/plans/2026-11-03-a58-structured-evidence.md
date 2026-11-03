# A58 Structured Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. The existing `training_execution` worker is the sole server executor; use one focused read-only plan review. No additional GPU queue.

**Goal:** Test whether jointly trained sequence decoding can retain A08 recognition while using JEPA temporal evidence to recover true changes without creating false short segments.

**Architecture:** Freeze the existing A08 seed-0 and A57 core seed2026110257 J-high6000 visual producers. Train a conditional age-state sequence model with exact global normalization and exact MAP under the same score. A matched A08-only decoder isolates the value of JEPA evidence; a duration-neutral and JEPA-only arm isolate the duration package and producer choice.

**Tech Stack:** Existing Python/PyTorch/NumPy stack, existing allowlisted EV9V loaders and metric library, one existing L40S server. No purchased service or external data export.

## 1. Authorization, scope and budgets

The user's 2026-11-03 instruction explicitly asks for failure analysis, a more promising research plan aimed at paper-quality results, and continued experiments. This is a new bounded A58 phase after closed A57, not an automatic rescue of A57 or an amendment of its outcomes. Routine implementation, engineering repairs, source qualification, checks, admissible training, evaluation and round-only commits are authorized without another confirmation. No coefficient/threshold/architecture search or new clinical-data access is inferred.

- Sole local workspace: `/Users/williamqiu/.config/superpowers/worktrees/echo-view-routing/a58-structured-evidence`, from clean `8b3eb9f8080b3049b3c14cd744c06ecdb465e87a`.
- Server: `ssh echo-l40s`, project `/home/william/echo-view-routing`, namespace `runs/a58_structured_evidence_v1`. Only `training_execution` launches or mutates server work.
- A58 cap: **16 GPU-job wall hours, 48 actual worker CPU-core hours, 20 GB incremental storage peak, 36 wall hours**, including preparation, failed checks, prediction caches, all admitted models/readouts/audits and closure. Reserve **2 GPU-job hours** for final inference/verification. New phase origin is worktree birth **1793718754.473543**, deadline **1793848354.473543**. Record earlier A57 and previous costs separately; never erase them. Coordination wall/CPU bookkeeping is separate from worker CPU receipts and explicitly discloses unmetered overhead.
- Start no formal matrix until measured end-to-end forecast covers all four arms × two seeds, all caches/inner/outer/stress readouts/checkpoints/audits and reserves. Engineering timing may choose CPU or GPU once before any formal model result; no model-performance-based tier selection. If the full matrix cannot fit, stop before formal admission and document the measured limitation; do not silently shrink seeds or arms.
- No website edits, push, unrelated staged changes, contact with others, new paid services, restricted test/sealedtest/Pediatric/Unity training, or 21 quarantined source-patient access. Keep recording/session IDs, raw images, weights, individual outputs and annotation packets on the server. Public artifacts contain aggregate counts/hashes only.

## 2. What the prior evidence does and does not establish

Read `docs/EXPERIMENT_INDEX.md` and corrected linked reports. A01 temporal heads and A06-R1 FACT action/frame tokens were completed negative experiments; A05/A07 transcript-forward training still inferred with window probabilities+B3; A10/A11 switch heads and A14 local routing did not jointly normalize complete labels, cuts and durations. A25 used a four-window expected-error surrogate with B3 inference. Existing B2 HMM has sticky transitions, hence implicit geometric durations. A58 is a different combination of established methods, **not a claim that structured training/attention was never tried or that CRFs are new**.

A57 completed 12 models / 68000 updates. J−F replicated, but raw J, full-data R-CE/R-ST, and four-depth J-MD did not exceed A08 on the full task. On selection, core J A5C recall was7/11 and8/11 versus A08-s0 11/11; stable-interior errors854/705 versus235 and false short≤16 segments39/52 versus9. J had fewer errors near constructed changes (456/454 versus711), confounded by two-position tubelet versus original stride5 nearest-centre sampling. Full-data/stability did not resolve the tradeoff; four-depth improved native recognition but not full segmentation. These are tested configuration-level negatives, not disproof of JEPA or all anatomical/temporal approaches.

Unresolved: complementary time-local scores under matched sampling, joint train/inference sequence objective, anatomical visibility/label correctness, actual patient independence and real continuous-scan transfer. A08 originally saw many of A57's inner sources; its reported inner result cannot be independent validation. Learned durations from synthetic concatenations are benchmark properties, not physiological durations.

## 3. Fixed sources, data and prediction contract

Use exact frozen A08 seed0 weight/marker/preprocessing identities from A57's three-seed reference reproduction and A57 core seed2026110257 J-high fixed6000 (not inner-chosen or R-refit). Qualify exact file bytes/SHA and complete checkpoint keys before prediction. Keep A57 dependencies immutable; any new producer/cache code is A58-owned. Use the two-column allowlisted metadata view, never a broad global-column CSV materialization.

- Reuse A57 FIT2736 recordings/402 sessions, inner628/100 sessions and independently rebuilt1500/300 constructed banks; freeze all private manifests and full media/window footprints. New head uses FIT only. The chosen J producer used FIT only, but A08 historically used many inner sources; disclose mixed inherited exposure and OPEN development status.
- Fit a new head using complete native and constructed sequences. Never concatenate cached native logits at joins: original video windows/JEPA chunks must be forwarded on the actual constructed stream, with the identical deployed padding/stitching context.
- **A08 time-dense producer:** retain original16-frame/10Hz/224/marker/input transformation/weights exactly; change only centre stride5→2, nearest-centre assignment on the full original10Hz axis. Canonical `window_centres(n,2)` begins1,3,... with its original endpoint rule; do not substitute0,2,... . Qualification counts are181057FIT/37124inner/18509selection/16988calibration windows before the additional stress bank. Cache original stride5 references unchanged and time-dense raw plus unchanged originalB3 transfer so density alone is visible. Any ambiguity must be resolved from original code and frozen before forwarding.
- **JEPA producer:** reuse exact A57 64chunk/stride32/positive-triangle probability stitch, two-position tubelet expansion and complete five-class output. Cache only probabilities and exact original-axis validity, not truth-dependent features.
- On each axis `p[t,c]` is FP32, finite, nonnegative, normalized; use `log(max(p,1e-6))` as score, not an independent-observation likelihood. Producer caches bind source weights/code/data/preprocessing/axis contract and output bytes. No source IDs, true labels/cuts, teacher labels, Base boundaries or training membership enters inference.
- Native evaluation uses the actual historical rule in `echo_routing/evaluate/dev_metrics.py`: majority/occupancy of decoded per-position labels, ties by class order (`np.bincount.argmax`). Use exactly this rule for both references and new decoders; also report decoded per-position confusion and within-native switches. Do not call a frozen producer's unchanged accuracy the new system's native accuracy.

## 4. Prospectively frozen complementarity diagnosis

Before new arithmetic, commit this card and a diagnostic input manifest referencing existing OPEN A57 outer caches: original A08-s0 and core J seed2026110257 fixed6000, raw original-axis outputs and the already-fixed originalA08 B3. No new media/forwards, tuning or model selection.

Report full five-class position agreement/error overlap for native, multi interiors (distance>5 original positions from a constructed label change), and near changes; paired native correctness and segment recoveries/losses with fixed IoU50 matching; oracle choosing between existing predictions at each position is a label-assisted upper bound, **not a deployable model**. Include both A4C/A5C directions and ≤16/≤10 true-short FN and false-short FP. Low complementarity on reused development is a scoped warning, not an invented gate cancelling the admitted controlled matrix.

## 5. Exact score and four-arm matrix

For `T` valid original positions, five labels, define age `a[t]=min(run_length_at_t,129)` where129 denotes overflow beyond128. Every label sequence has **one** age sequence (start age1; stay increments/caps; changing class resets1). Every class can last one position or an arbitrarily long time. No forced cut, no hard minimum/maximum length, no adjacent same-class semantic segment. No extra latent segmentation multiplicity.

For probabilities `p`, define20D feature at `t>=1` as `[p[t],p[t-1],p[t]-p[t-1],abs(p[t]-p[t-1])]`. Concatenate two blocks to40D. Neural transition score `v[t] = Linear64to25(GELU(Linear40to64(x[t])))`, reshaped5×5, **including stay**. First layer seeded identically across arms; final layer zero. Duration score `g[c,a]` has5×129 entries, initialized0; no start-class prior. A path score is

`S(y|x)=sum_t unary[t,y[t]] + sum_{t=1..T-1} v[t,y[t-1],y[t]] + sum_{t: t==T-1 or y[t+1]!=y[t]} g[y[t],a[t]]`.

Duration is an end-of-run potential, applied at terminal too. Overflow self-loop does not repeatedly charge the end score. Partition `logZ=logsumexp_all_y S(y|x)` uses exact age-state forward recurrence; training `NLL=(logZ-S(truth))/T` averaged equally over one native and one constructed sequence. Inference uses Viterbi maximizing **the same score**, deterministic lexicographic tie handling. NLL is conditional structured likelihood, not differentiable F1. It can still damage correct A08 recognition.

| Arm | Fixed unary | Transition input blocks | Duration | Question |
|---|---|---|---|---|
| D0 | time-dense A08 logp | A08,A08 | g fixed0 | duration-neutral structured control |
| D1 | time-dense A08 logp | A08,A08 | learned g | duration package vsD0 |
| D2 | time-dense A08 logp | A08,JEPA | learned g | **primary D2−D1: added JEPA evidence** |
| D3 | JEPA logp | JEPA,JEPA | learned g | stand-alone producer under same decoder |

D1/D2 have identical parameter counts, architecture, head initialization, optimizer, plans and exposures; duplicate block keeps width/parameter count matched, not independent information. D0 has intentionally frozen g, disclose differing active count. At zero initialization all arms decode unary argmax; D0/D1/D2 must agree exactly for any common A08 inputs. No learned unary scale, extra B3, post-hoc merging, coefficient/LR/duration-cap search or outer-driven checkpoint choice.

## 6. Training, selection and evaluation

- Two head seeds **2026110358 and2026110359**; same seeded FIT plan across arms, each6000 AdamW steps, lr1e-3, wd.01, warm100/cos6000, FP32/clip1. Batch one full native+one full constructed stream; use valid masks for variable lengths. Native and constructed losses each normalized by their actual valid length, then averaged; no ground-truth-dependent sampling beyond the declared training supervision.
- Save latest complete model/Adam/RNG/plan cursor every100steps atomically; immutable weights at2000/4000/6000. Record optimizer-step progression and actual finite gradients/changed parameters; first-layer gradients are expected zero at initial zero-output step, must become active later. Frozen producers get no optimizer entries.
- Inner readout only2000/4000/6000; primary table fixed6000, separate inner-selected table by multi segmentF1@50, tie native macroF1, then fewer false-short≤16, then earlier step (numeric equality tolerance1e-12). No parameter/decoder search. A08 historical exposure to inner disclosed in every summary; former R-refit inner cannot be reused as validation for R.
- Only after complete weights/selection/source manifest actualGit freeze run once on unchanged OPEN outer selection/calibration, same native284/237, multi120/120, pairs137/115 and full10Hz axis. All3 originalA08 references reused as historical matched-axis context; primary matched learned-decoder contrast uses A08-s0 in all arms. No averaging over A08 seeds as a new arm.
- Fixed metrics: native five-class macroF1 and5×5 confusion, complete multi segmentF1@50, boundaryP/R using existing fixed matcher/tolerance, ≤16 and≤10 FN/FP denominators, A4C→A5C and reverse, native switch rates, recovered/lost true segments, exposure table, training/inner curves, inference time/VRAM/storage. Paired comparison uses5000bootstrap replicates seed2026110358 and95% descriptive intervals. Native resamples whole sessions with all their recordings; multi resamples whole fixed-bank streams; pair readouts resample whole fixed-bank pairs, preserving all positions in each unit and identical sampled units across models. Multi/pair intervals are conditional on the bank, with shared-source dependence explicitly unresolved; they are not patient-clustered population confidence intervals. Sessions are not confirmedpatients, and reused outer is not confirmatory holdout.
- **Prospective duration-shift stress:** additional fixed120 streams per outer half, same authorized sources, seed2026110358. Reuse original4..8fragments/class-uniform/p_same_next.35/gamma090edit_prob.5 recipe with reuse across streams but no repeated cine within one stream. Change support from10..40 to4..80 original positions (0.4–8s): filter eligibility n>=4, draw each length uniformly4..min(80,n), then start uniformly0..n−length. This is the original availability-truncated sampling mechanism, not independent requested4..80 then clipping. Report realized lengths and same-label RLE merges; no resampling until preferred result. Freeze the bank identities before first formal head result. This is an OPEN construction stress test, not new patients or clinical evidence. Evaluate fixed6000 only, keep main unchanged.
- A local gain requires D2−D1 segment gain without hidden native/short/boundary harm; a paper-quality result additionally needs robust effect against A08, both seeds/halves and duration shift, credible ablations, and genuinely independent expert/clinical evaluation. CIs crossing0 are uncertainty, not equivalence. Report all arms regardless outcome. No automatic next architecture after matrix closure.

## 7. Evidence and publication track

The user's goal is paper-quality science, not an unsupported publication-readiness declaration. Draft a concrete independent annotation protocol: blind model identity; expert frame-range view/visibility and transition uncertainty; agreement/adjudication; patient split verification; separate real continuous scans. Existing clip labels propagated to frames and concatenated joins do not establish anatomy/visibility. User availability for blinded annotation may be clarified while computation proceeds; no invented labels, restricted-data access or external contact. Keep any review packet server-only. No clinical/independent-patient claim can follow from A58 alone.

Established literature, not novelty claims: [semi-Markov CRFs](https://papers.nips.cc/paper_files/paper/2004/hash/eb06b9db06012a7a4179b8f3cb5384d3-Abstract.html) already jointly learn segment scores; [FACT](https://openaccess.thecvf.com/content/CVPR2024/html/Lu_FACT_Frame-Action_Cross-Attention_Temporal_Modeling_for_Efficient_Action_Segmentation_CVPR_2024_paper.html) already models frame/action relations. The current [official EchoJEPA source](https://github.com/bowang-lab/EchoJEPA/blob/main/README.md) recommends its2.0 ViT-L VMix checkpoint for view classification and2.1 for dense prediction. A41 already tested that2.0 VMix ViT-L with900-step LoRA; it is not a never-used checkpoint. A matched full-video-path/dense sequence assay using that functional source remains unexecuted, distinct from A41 and not another arm here. A58's age-capped end potentials are an efficient restricted duration model, not a full arbitrary-segment feature semi-CRF.

## 8. Implementation tasks and required checks

### Task1: sources, cache-only diagnosis and baseline
Files create `scripts/prepare_structured_evidence.py`, `scripts/diagnose_structured_complementarity.py`; public results `docs/results/2026-11-03_structured_evidence/`; private run manifests under new namespace. Read existing `echo_routing/temporal/dense_sequence.py`, A57 inference/reference/data utilities and their frozen receipts. Reuse source loaders/metric code without altering frozen prior dependency files.
- [ ] Freeze card and input identities in actualGit before new diagnosis arithmetic.
- [ ] Verify new worktree inherits tested source, run `/Users/williamqiu/Desktop/Harvard/Capstone/echo-view-routing/.venv/bin/python -m pytest tests/test_dense_sequence.py -q` (or identical qualified server environment); report any unavailable dependency rather than install a new environment gratuitously.
- [ ] Capture real server GPU/PID/disk inventory, A08/JEPA complete-key/sourceSHA qualifications and private allowlist/footprint counts; preserve main8 staged theory files. Historical native classification is occupancy/class-order ties, verified from implementation rather than assumed probability pooling.
- [ ] Execute once cache-only complementarity with CPU receipt; save aggregate output, denominator/matching oracle check and zero new forwards.

### Task2: exact structured model
Create `echo_routing/temporal/structured_evidence.py`, `tests/test_structured_evidence.py`, `scripts/check_structured_evidence.py`. Test first on2/3classes short enumerable sequences, then implement5class production API.
- [ ] Enumerate every label path forT<=6 and small agecap; independently compute scores/logZ/MAP/gradients. Require recurrence scores agree1e-6FP32 (use FP64 for numerical oracle1e-10), NLL>=0 within tolerance and gold score<=logZ.
- [ ] Test unique age paths, true length1segments, alternating labels, all-one-class T>128, terminal score, overflow charge only once, variable-length masks and empty/invalid inputs fail closed. All5classes available; no forced native cut.
- [ ] Test zero-init argmax equivalence, D1/D2 equal initialization/counts, a positive learned duration perturbation changes score as specified, no truth/source ID inference input, and future sequence context allowed (offline task, disclose latency).
- [ ] Test finite/autograd gradients against brute force, nonzero active parameters after updates, frozen producer roles, Adam coverage, exact repeat and checkpoint+RNG resume, corrupt-cache/identity rejection and atomic-write recovery.
- [ ] Explicitly protect all-unreachable age-state reductions against NaN backward; test finite gradients at lengths1/128/129/130. Verify global lexicographic MAP ties against enumerable tied paths rather than assuming local state-order ties produce the declared rule.
- [ ] Run focused tests `.../.venv/bin/python -m pytest tests/test_structured_evidence.py tests/test_dense_sequence.py -q`; no repeated whole-library audit for documentation-only changes.

### Task3: real producer timing and full admission
Create `scripts/cache_structured_evidence.py`, `scripts/admit_structured_evidence.py`, new frozen config under `configs/a58_structured_evidence_v1.json` (or repository-equivalent config directory, fixed in implementation identity).
- [ ] On allowlisted FIT sequences measure actual stride2 A08 and unchanged JEPA full-stream inference, identity/axis/cache-shape and constructed-join context checks. Measure actual complete full-sequence forward/backward/optimizer steps for longest supported stream and representative length mix, including exact logZ/MAP on CPU/GPU. These are engineering charges, not formal performance results.
- [ ] Bind two-column metadata, source/encoder/decode contracts, actualGit code, source SHA, plans, stress banks, complete CPU/GPU/storage forecast and2GPUhreserve; check remaining deadline. Freeze computing backend once from cost/stability before results.
- [ ] Complete caches with atomic manifests, verify no gap/duplicate axis and no source split crossing. Formal start needs real workerPID plus advancing optimizer-step evidence.

### Task4: matched training and frozen readout
Create `scripts/train_structured_evidence.py`, `scripts/supervise_structured_evidence.py`, `scripts/readout_structured_evidence.py`, `scripts/report_structured_evidence.py`.
- [ ] Execute all8runs under one queue, shared seeded plans; record every physical cost/failed attempt. Honest engineering retry only after documented deterministic bug fix; no ambiguous performance retries.
- [ ] Save exact weights/optimizer/RNG traces, verify sampled resume/repeat and parameter identities. Freeze fixed and inner-selected manifests before outer readout.
- [ ] Read out all predefined controls/matrix/stress, aggregate fullmetrics/pairedcomparisons and diagnostic recovered/lost segments; retain negative results and sampling/legacy exposure caveats.

### Task5: independent verification, publication-gap record and closure
- [ ] Focused independent audit of DP on enumerated paths, cache/model/data identities and aggregate metrics/denominators/paired contrasts; actual job exit/GPUidle evidence and receipts. Keep audit compute within cap.
- [ ] Update A58 result/index and annotation protocol with completed/running/negative/diagnostic distinctions. Selectively sync verified public A58 records into main without staging user files, rewriting other index rows, pushing or touching webpage.
- [ ] Save actual resource/terminal/public-file manifests. End A58 monitor only after all admitted work/verification/sync finishes or a real hard cap/external blocker ends the entire phase with honest closure. No automatic A59.
