# A46 local motion representation implementation plan

> **For agentic workers:** Use superpowers:executing-plans task by task. Codex is the sole execution owner, directly authorized by the lead on 2026-10-28. One read-only plan review is advisory, never another training owner.

**Goal:** Test whether explicit spatially organized pixel displacement supplies transferable A5C evidence while protecting correct A4C and complete segmentation.

**Architecture:** Frozen original A08 window embeddings provide appearance; a newly trained 3D convolutional encoder reads dense two-component optical flow over the same 16 original frames. A common learned fusion head produces a scalar run-repair score. F uses intact flow maps; S uses identical flow values with independently permuted spatial tiles, holding architecture, initialization, loss, batches, training and compute fixed. Neither arm vetoes an earlier policy.

**Tech Stack:** PyTorch / torchvision RAFT-small, NumPy, existing A31 pools and evaluator, server-only private arrays, local git as authority.

## Research contract (fixed before any formal training or held-out model scores)

### History and substantive difference

- A43 D:25/103 S1 recovered,112/1413 correct A4C harmed; M:2/103,4/1413. A44 C:12/103,22/1413; N:26/103,76/1413. A45 A/U:both0/103,1/1413. None passes the full segmentation/protection package.
- The post-A44 audit found correct dual arithmetic but final full-fit H=.010329/.181141 >.01, largely unresolved last-seen negatives and an upward score shift without consistent held-out ranking gain. It does not identify a unique anatomical cause.
- A45 reconstruction on frozen 512-dimensional features vetoed all26 N fixes and75/76 harms: rejection alone did not add target evidence.
- A09 SupCon, A42 pair-ranking and A44 risk loss changed objectives on RGB/clip codes. A40 trained masked HOG prediction, a static gradient reconstruction target; its auxiliary gain was zero. A22/A33 changed frame density. A36 tested frozen higher-resolution RGB descriptors. A39 pooled frozen patch tokens. A41 compared pretrained video states with LoRA; A43 learned local attention over RGB tokens. None tested an explicit learned displacement-field stream with a spatial-organization control.
- New evidence means a different representation and correspondence prior, not new human truth or information absent from the source frames. RAFT estimates displacement; flow or learned activations are not LVOT/valve labels, cardiac-phase truth, or verified tissue motion. Ultrasound speckle/domain shift may defeat this hypothesis.
- Expert pilot currently has0 real records; Unity remains evaluation-only. No expert/pseudo/anatomical labels are invented or used. A32/A35 recognizer outputs are not targets.
- Sources: [two-stream original](https://proceedings.neurips.cc/paper_files/paper/2014/file/ca007296a63f7d1721a2399d56363022-Paper.pdf), [RAFT](https://arxiv.org/abs/2003.12039), [torchvision flow contract](https://docs.pytorch.org/vision/0.22/auto_examples/others/plot_optical_flow.html). These establish methods, not A5C effectiveness.

### Data and immutable baseline

- Namespace `video_local_motion_a5c_v1`, source seed0 screen, A31 two opposite session folds. Source sessions are not confirmed patients. These are repeatedly inspected training-source pools, not an untouched test; A08 had prior training exposure to the recordings.
- Use exactly original Base-A4C P runs:749/771, needed rows5030/5437 from grids24585/25774. Fit positives34/69 and protected negatives714/699; other eligible1/3 excluded from fitting only, retained in complete readout. Identical original per-run membership, nearest-centre multiplicities/duration weights, marker, gamma,224 crop and request order.
- Appearance is original cached A08 512-d window embeddings, frozen intentionally. No A44/N weights or proposals are inputs. Pin original A08 model, pool, split, output and P-run files to A44's existing full SHA identities. Fold footprints and source isolation are reasserted before pixels. No dev/check/test/Pediatric/A26 runner/Unity training; no teammate work, no push.
- Only original eligible rows select flow inputs, without new-model or true-label selection. Build unlabeled flow fields once for both folds; no predictions or held-out fitting during preprocessing.

### Flow representation (label-free, fixed)

- Frozen `torchvision.models.optical_flow.raft_small`, exact `Raft_Small_Weights.C_T_V2`, URL `https://download.pytorch.org/models/raft_small_C_T_V2-01064c6d.pth`, trained on FlyingChairs+FlyingThings3D (official metadata). Save complete weight SHA and installed torch/torchvision/source SHA. No echo dataset in its stated training recipe; exact unrelated pretraining overlap is not assessed.
- For each original 16-slot224x224 RGB uint8 window, estimate15 forward fields between adjacent slots. Inputs scale to[-1,1] by official transforms, no resize/crop change, FP32, eval,12 recurrent updates, TF32 off. Output vectors are in224-image pixel units. Average pool4x4 to56x56 without rescaling vector units; divide by8, clamp[-4,4], store FP16 then restore FP32 identically in both arms. Record clipping/finiteness/nonzero statistics.
- End clamping/repeated-frame pairs and cross-recording/splice pairs have exact zero fields rather than artificial jumps. A valid pair must have the same source recording and gamma, and advance by the original adjacent source-frame increment; repeated/non-contiguous pairs are zeroed. Assert this from reference metadata, not class labels. Deduplicate by complete ordered reference pair, never by class/score. Public records contain counts/digests only; private pair identities stay server.
- S partitions56x56 into7x7 tiles of8x8. A deterministic permutation from SHA256 of `A46|fold|original-row` permutes49 tiles; the same permutation applies to both vector channels and all15 times. This preserves the per-time/channel value multiset and temporal trajectory inside each tile, while destroying original inter-tile neighborhoods and positions. F uses identity. Permutations have no label access. S may retain global/local within-tile evidence; beating S alone is insufficient.

### Trainable model and fixed learning rule

- Input per window:2x15x56x56 flow plus a marker channel with a single pair-time plane at `min(original_marker,14)`; frozen appearance512.
- Motion encoder: Conv3d(3,16,k=(3,5,5),s=(1,2,2),p=(1,2,2)), GroupNorm(4,16), SiLU; Conv3d16->32,k3,s(1,2,2),p1,GN8,SiLU; Conv3d32->64 likewise,GN8,SiLU; adaptive average pool(1,4,4), flatten1024,Linear1024->128,GELU.
- Appearance LayerNorm512; concatenate with motion128;Linear640->128,GELU,Linear128->1. Last weight0,bias-2: Init score-2, exact Base labels. New convolution/fusion/normalization parameters train; original A08 and RAFT frozen and checked unchanged. No dropout or BatchNorm running-state differences.
- F/S same seeded initialization `2026102846+fit_fold`, same600-update A44 C cyclic plans (1positive,3negative runs, full negative coverage), same complete window occurrences, no random data augmentation. Same balanced BCE `softplus(-r_pos)+mean softplus(r_neg)`, where each r is its full weighted mean of window scores. No dual/hinge, adaptive threshold, margin mining, early stopping or new gate.
- AdamW(all new parameters, lr1e-3,betas.9/.999,eps1e-8,wd1e-4),60linear warmup then cosine600, clip global L2 norm1; FP32 deterministic CUDA. Full-batch retained-graph autograd over all occurrence windows is intended (small CNN), no frozen CNN-feature/head substitute. If memory requires microbatching, exact two-pass J-transpose-v across all windows incl zero gradients, with numerical parity before formal training; no changed objective/compute between arms.
- Four actual600-step models F_f0,S_f0,F_f1,S_f1, same data/parameter counts and actual optimizer counters. Resume restores optimizer/update and exact input plan. No intermediate held-out scores or checkpoint selection. No automatic extra seed/variant in this bounded screen; a positive exploratory result is not confirmed generalization.

### Freeze, one readout, and gates

- Train finals only; save full SHA for model, head, original Base, RAFT, config, code, all inputs, pair maps, embedding arrays, plan and update logs. Commit actual local git manifest/rule objects before server freeze and one opposite-fold readout. Server has no.git. Verify server/local aggregate SHA and the git objects in commit, not just an untracked file.
- Score all original eligible rows once/model; full run weighted mean. Only original Base-A4C and r>0 changes toA5C, others copied bitwise; merge equal neighbors. A score0 staysBase. Correct non-A4C including A5C preservation is structural, not learned achievement. RLE may delete correct boundaries.
- Publish Base/F/S full run counts, S1 recovery/source attribution, correct-A4C harms, real-proportion native/full segment F1@50, boundary P/R, short true misses and deleted boundaries. Optional same-forward zero-motion ablation Z_F/Z_S is diagnostic only (not an alternative chosen package); predeclare and cache it in the same once readout. No threshold scan.
- Absolute package gate, each arm separately: S1>=11/103,>=1 per fold and>=3 source recordings total with>=1 each fold; harms<=14/1413; nonA4C invariant; net>=0 each fold and>=5 OOF; complete F1 nondecreasing each fold and mean gain>=.001; native macro-F1 loss<=.003 each fold; boundary recall loss<=.005 each fold; short misses<=Base+3 OOF; non-noop.
- Representation contribution gate separate from package: F fixes>=S in each fold and>=5 more OOF, harms<=S each fold, net F-S>=5 OOF, full F1>=S each fold; F absolute package must pass. Report overlaps F-only/S-only/shared fixes and harms from private cases, not aggregate equality. Compare A43D/M,A44C/N,A45 under same source0 denominators without rescoring old models.
- A failed screen closes exactly this optical-flow/cached-appearance/CNN recipe. No A47 or recovery retuning. Gates are engineering selection criteria on reused folds, no clinical risk guarantee.

### Checks, independence, resources

- Meaningful tests before implementation: spatial permutation multiset/time/vector binding, invalid/repeated pair zeros, marker binding, independently expressed weighted BCE/gradient, InitBase, legal one-way edits, complete input/plan coverage, optimizer+RNG resume, complete evaluator harmed-boundary example.
- Before training: input provenance/fold isolation; RAFT official-transform and batched/single parity; synthetic known translation/same-frame checks (loose sanity for pretrained model, not ultrasound truth); finite flow/clipping; full numerical production gradient vs independent retained/two-pass formulation; all CNN/fusion components genuinely update after zero-head first step; identical-arm forced-F path/repeat/resume; synthetic deployment boundary damage; memory/time/disk projection.
- Independent audit after readout reconstructs pair/window membership and preprocessing on a label-free subset, frozen means from original repeated samples, spatial permutations, final model scores from fresh explicit nn.Sequential modules, source attribution from original run intervals, per-case edits, original evaluator counts/F1/gates and F-S overlap. Audit input SHA and all model/old Base unchanged; no active jobs/GPU at closure. Audit need not run a second formal model readout: use retained private score cache plus a fixed small parity subset, clearly distinguished.
- Shared round cap GPU6h/activeCPU3h/new3GB; all flow preparation, failures, bootstrap, checks, training, scoring and closing count. Idle waits do not. Root work starts20:17:31UTC; count real active intervals, disclose audit/reviewer scope. Stop before budget overrun without publishing a partial result as completed. Source media/identifiers/private scores remain server. Protected eight staged theory files and other dirty files must not change.
- One Codex owner/queue. Heartbeat only meaningful start (once), blocker/failure/action or verified completion; otherwise quiet. Never spawn Claude/duplicate jobs. Delete heartbeat after complete closure, no A47.

## Implementation tasks

Pre-training engineering correction,2026-10-28 20:40UTC: attempt1 stopped before its first optimizer step because CUDA adaptive_avg_pool3d backward has no deterministic implementation. Keep strict determinism and the same adaptive(1,4,4) floor/ceil bins, implemented with explicit slice means and stacking. Independent FP64 forward/gradient parity is required before retry. No architecture parameters, training target, input, batch, gate, pooling geometry or formal output changed. Failed check/CPU/GPU time and original log are retained.

### Task1: contract and metadata
- [ ] Commit this protocol and YAML config; advisory read-only plan review and repairs before dependent work.
- [ ] On server prepare fixed pair maps/counts from eligible input references; full expected train lists/plans, frozen SHA and byte/time forecast. Refuse infeasible budget before flow extraction.

### Task2: representation and tests
Files: `echo_routing/temporal/local_motion.py`, `tests/test_local_motion.py`.
- [ ] Write failing synthetic critical tests, run `pytest tests/test_local_motion.py -q` and preserve expected RED evidence.
- [ ] Implement pair validation/permutation/model, weighted means/BCE and legal deployment, independent-reference tests; run focused tests plus A44/A45 baseline.
- [ ] Commit round-only code/config. Baseline already18tests pass; original warning preserved.

### Task3: single stage runner
File: `scripts/train_local_motion_a5c.py`.
- [ ] Implement prepare/flow/check/train/freeze/readout stages, once-only markers/resume, exact source SHA manifests and actual ledger/progress/PID.
- [ ] Run plan+flow checks, inspect raw tensor/device/update evidence; commit check aggregates.
- [ ] Train fixed four600-step models; monitor GPU/model PID/update, inspect final counters and all updated tensors; commit weights manifest.

### Task4: once readout and independent verification
- [ ] Verify manifest and rules are actual local git objects, freeze on server, then one opposite-fold scoring and complete tables.
- [ ] Independently reconstruct cached means/labels/membership/source attribution, overlap and full metrics; compare Base and closed A43–45 without new old-model inference.
- [ ] Publish negative or qualifying scoped result, exact resource ledger, final SHA comparison/no-process check; round-only commits/index row while preserving protected staged hash. Delete heartbeat only after closure.
