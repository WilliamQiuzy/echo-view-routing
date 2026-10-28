# A46: explicit local motion representation for A5C repair

**Completed and independently verified, negative.** The four fixed F/S models each completed600 actual optimizer updates. This changed the visual representation and correspondence prior, with a genuinely trained motion encoder; it did not add an A44 proposal veto or alter a risk-loss coefficient. It recovered substantially more clip-coded A5C runs while harming many more correct A4C runs. Neither arm passes the absolute protection/complete-segmentation package or the separate representation-contribution rule. No threshold change, new variant, extra seed or A47 follows.

| Policy | A5C recovered | Correct A4C harmed | Net | Mean complete multi F1@50 | Short true misses | Base boundaries deleted |
|---|---:|---:|---:|---:|---:|---:|
| Base | 0/103 | 0/1413 | +0 | 0.956885277 | 76 | 0 |
| F | 58/103 | 161/1413 | -103 | 0.947259188 | 131 | 156 |
| S | 66/103 | 189/1413 | -123 | 0.945804024 | 139 | 182 |
| Z_F | 33/103 | 104/1413 | -71 | 0.948433842 | 117 | 87 |
| Z_S | 66/103 | 191/1413 | -125 | 0.945308686 | 140 | 183 |

F (intact spatial motion) fixes58/103 S1 runs from29 actual A5C-coded source recordings (11/18 by fold), but breaks161/1413 correct A4C runs (89/72). Its mean complete F1 falls0.009626089 from Base, short misses rise76→131, and merges delete156 Base boundaries. The engineered harm limit was14, not161. Other original Base classes, including1150 correct A5C runs, remain unchanged by the one-way rule; this is structural preservation rather than learned protection. Higher boundary precision and fewer internal cuts do not compensate for lost true boundaries, short segments or native recognition.

S (spatially shuffled identical fields) fixes66 but harms189, net−123. F−S is−8 target fixes,−28 harms, net+20 and F1+0.001455164 on average, with F1 higher in both folds. This is less harmful than a failed control, not a successful package. The target-recovery increment condition fails as well as F's absolute gate.

## Why this was a different question

A43's RGB-token witness M found2/broke4; dense-positive D found25/broke112. A44 deployed-run C found12/broke22, and risk-constrained N26/76. The [post-A44 audit](../2026-10-28_post_a44_risk_diagnosis/README.md) verified its dual signs and all600 recurrences, but the final full-fit constraint was not satisfied and held-out scores moved toward promotion without consistent ranking improvement. A45's A4C-support reconstruction on frozen RGB embeddings found0/broke1 after vetoing every real N fix. These rounds did not deliver discriminative evidence sufficient for safe recovery.

Earlier A09/A42 changed contrast/ranking objectives; A40 added masked static HOG reconstruction; A22/A33 changed temporal sampling; A36 tested frozen high-resolution RGB; A39 pooled frozen patch tokens; A41 adapted pretrained video features with LoRA. A46 instead makes pixel displacement an explicit input modality and learns its spatiotemporal representation with a new CNN. The expert pilot record still supplies no authorized anatomical training labels; Unity remains evaluation-only. No expert or model-produced anatomy label was used.

The motivation is the [two-stream representation](https://proceedings.neurips.cc/paper_files/paper/2014/file/ca007296a63f7d1721a2399d56363022-Paper.pdf) and [RAFT correspondence model](https://arxiv.org/abs/2003.12039), implemented according to the [official torchvision contract](https://docs.pytorch.org/vision/0.22/auto_examples/others/plot_optical_flow.html). This is a representation/prior change using the same source frames, not proof of new independent information, LVOT visibility, valve anatomy or reliable tissue motion. Optical flow in ultrasound can reflect speckle/domain shift.

## Fixed representation, training and data

Source0, original A31 session folds and Base+A08+B3 P runs:749/771 eligible runs,5030/5437 original request rows from grids24585/25774. Fit positives34/69, protected negatives714/699; other eligible1/3 are excluded only from binary fitting and remain in complete evaluation. All original16-slot windows,224 crop/gamma, target markers, nearest-centre membership and duration/multiplicity weights remain. Only original Base eligibility selects input windows; no new score or reference label selects flow examples.

Frozen RAFT-small C_T_V2 (990162 parameters, FlyingChairs+FlyingThings3D training recipe) supplies15 two-component forward fields per original window. Exact full weight SHA01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27; installed torch2.14.0+cu130 / torchvision0.29.0+cu130 and source SHA in flow.json. Original224 RGB is scaled to[-1,1],12 recurrent steps, FP32/TF32 off. Average pool4x4 retains vector units in224-image pixels, divide8, clip±4, FP16 cache/FP32 CNN input.43543 distinct valid reference pairs are computed once. Repeats and cross-source/noncontiguous pairs are exact zero, avoiding artificial splice motion. Clipped element share is0.039%/0.024%; fields are finite and nonconstant. Deduplication uses complete ordered reference pairs, not labels.

F preserves spatial fields. S deterministically permutes49 spatial8x8 tiles of the56x56 maps per original row, sharing its permutation across both vector channels and all15 times. Values and within-tile temporal trajectories are retained, inter-tile neighborhoods/positions disrupted. The sham can retain global and within-tile evidence; F−S isolates spatial organization within this recipe, not total motion benefit. F/S have the same CNN forward/backward, window occurrence and update counts; S additionally incurs its input tile-reordering overhead, not extra learned computation.

The genuinely trained model has287457 parameters/20 state tensors: three3D convolutions3→16→32→64 with GroupNorm/SiLU, adaptive(1,4,4) mean bins, motion projection1024→128, LayerNorm on frozen original A08 appearance512, fused640→128→1 GELU head. A single pair-time marker channel binds the original target. The last weight is0/bias−2 at Init, exactly reproducing Base edits. Original A08 trunk/projection/classifier and RAFT are deliberately frozen inputs, not falsely described as updated; the new motion encoder, appearance normalization and fusion all receive real gradients and updates. No cached-CNN-feature/head substitute.

Same seeded starts and complete A44 C600 plans (one positive,three cyclic protected negatives; every negative covered), full run weighted-mean scalar BCE, AdamW lr.001/wd.0001,60 warmup then cosine600, clip1, FP32 strict deterministic CUDA. Full-batch retained-graph autograd fits memory. Each fold/arm uses14778/15528 actual window occurrences,60612 total. No validation checkpoint/epoch/threshold selection or risk multiplier. Sole Codex worker PID2925054 on L40S, formal stage20:50:57–20:51:43UTC; all four final optimizer receipts have600 steps in all20 tensors, all final tensors moved and all three component gradient groups are live.

## Complete-system results and case-level comparison

| Held-out fold | Policy | S1 fixed | Correct A4C harmed | Complete F1@50 | Native macro F1 | Boundary recall | Boundary precision | Short misses |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | Base | 0 | 0 | 0.963906582 | 0.977556578 | 0.967644085 | 0.955566650 | 41 |
| 0 | F | 24 | 89 | 0.948662893 | 0.958552519 | 0.954499494 | 0.970694087 | 77 |
| 0 | S | 24 | 89 | 0.948457634 | 0.958552519 | 0.953993933 | 0.970679012 | 80 |
| 0 | Z_F | 24 | 92 | 0.947286216 | 0.958552519 | 0.953993933 | 0.970679012 | 80 |
| 0 | Z_S | 24 | 90 | 0.947861746 | 0.958552519 | 0.953488372 | 0.970663922 | 80 |
| 1 | Base | 0 | 0 | 0.949863972 | 0.975831638 | 0.968139774 | 0.942000000 | 35 |
| 1 | F | 34 | 72 | 0.945855483 | 0.947947306 | 0.956320658 | 0.962752199 | 54 |
| 1 | S | 42 | 100 | 0.943150415 | 0.930028124 | 0.953751285 | 0.966666667 | 59 |
| 1 | Z_F | 9 | 12 | 0.949581468 | 0.976003688 | 0.967625899 | 0.945755902 | 37 |
| 1 | Z_S | 42 | 101 | 0.942755626 | 0.930028124 | 0.953751285 | 0.966666667 | 60 |

Independent paired cases, F/S:55 shared A5C fixes,3 F-only,11 S-only;144 shared A4C harms,17 F-only,45 S-only;76 total run-label disagreements. Equal fold0 counts are not equal cases: both harm89, of which86 are shared and3 unique to each. Complete metrics are computed on the actual resulting whole sequences and merges, not inferred from fix counts.

The predeclared zero-motion interventions Z_F/Z_S are diagnostics only, not alternative candidate packages. Removing F's motion retains33 A5C fixes, breaks104 A4C, and yields F1.948433842, still below Base. Actual F preserves all33 such fixes and adds25, but also adds61 harms while avoiding4 (net57 additional harms); this is not a clean safe-evidence gain. Effects are asymmetric: held-out fold0 has24 fixes with or without motion, while fold1 rises9→34 and harms12→72. Held-out ranking AUROC F vs Z_F is.85261 vs.84903 and.84039 vs.86436, so not a consistent ranking gain. Zero input is a distribution-shifted ablation, not an anatomy label or separate trained static control. S/Z_S make the same66 fixes and differ only on2 harms, verified per case.

Historical endpoint comparison uses the same source0 Base run denominators, without rescoring old models. F vs A44 C:11 shared fixes,47 F-only and1 C-only; every22 C harm is retained and139 more appear. F vs N:24 shared fixes,34 F-only and2 N-only;62 shared harms,99 F-only and14 N-only. Thus some recovered targets differ from A44's, but the new representation does not meet protection and complete-system requirements. This cross-round comparison is descriptive, not a matched architecture causal test.

## Freeze and independent verification

Fixed card3568f33; model implementation674d685, equivalent deterministic pooling20e0afc, check resource-forecast repairc2c3918; checks/preparationc196395. All152 loaded repository modules/parent configurations matched actual local git blobs before formal fit. Four-model/full head/base/config/code/input/pair-map/plan SHA manifest **14e453e27b9bbc3375baaebd028aba59f46a1720**, SHA455a5bf8ba4b5c2647d09d2a543805ae16be8cd1f649d33b2664f19da9abad6a, was committed and its actual blob verified20:56:45UTC before server freeze20:57:18 and exactly one opposite-fold readout20:57:24–20:57:40 (PID2928462). Server has no.git; local git is authority. See freeze_provenance.json.

26 focused tests pass (18 baseline+8 new critical tests), expected RED before implementation. The failed CUDA adaptive-pool check is preserved: attempt1 stopped before its first optimizer step because deterministic backward is unavailable. Exact floor/ceil bins were implemented using slice means; independent FP64 values/gradients agree, strict determinism retained. Attempt2 was interrupted during a redundant1200-fold metadata resource projection; reusing the already verified identical exposure means fixes that engineering cost only. Both failed stages are charged and their original logs/receipts retained; no formal retraining or scientific gate relaxation. Attempt3 passes: full production vs individual-window chain-rule relative gradient2.49e−7, score difference0, exact6-step repeat and3+3 optimizer resume0, actual peak1.04GB, live CNN/fusion gradients, frozen-flow checks and synthetic4-pixel translation median3.95. Tests do not establish anatomical flow accuracy.

The separate auditor (1e7b3da) independently reconstructs every pair/window mapping and all deterministic permutations; expands every run one original sample at a time under the nearest-centre map; reconstructs weighted means, legal labels, source attribution, whole sequences/merges, original full evaluator counts/F1, each gate and paired overlaps. It checks all600 logged input-row hashes, optimizer counters, independent initializations and all20 moved state tensors in every final. A separate functional convolution/GroupNorm/GELU/LayerNorm implementation verifies a fixed label-free first/middle/last-row subset (maximum score error9.54e−7);6 fixed pairs are freshly redecoded/re-estimated against the frozen field cache (maximum normalized error0.0004883). Full numerical/evaluator recount uses the single formal private score cache; this is not a second full model readout. No private media/recording/session ids or individual scores leave the server.

Readout.json, independent_audit.json, cache_comparison.json and weights.json are authoritative aggregate results. Original A08, A44 comparative cache, all field banks and final models are SHA-checked unchanged. Directory resolution reads only the whitelisted video_id/frames_dir manifest columns to locate allowed training-source frames; no forbidden split pixels, predictions or labels are used. No dev/check/test/Pediatric/A26 runner, Unity training or expert pseudo-labels; no push or teammate action. Eight staged theory files and other dirty user files are retained.

## Decision, scope and resources

Both absolute package gates fail on A4C harm, per-fold/OOF net, full F1, native recognition, boundary recall and short misses. F also fails the predeclared positive representation increment. No candidate is advanced and no threshold relaxation, extra seed or A47 is launched.

This closes this fixed RAFT-small/cached-A08/287457-parameter CNN/600-step recipe on source0. It does not refute motion representation learning generally. Clip codes are weak view labels, flow is not anatomical supervision, these folds have repeatedly informed design, A08 had training exposure to the source recordings, source sessions are not verified patients, and zero-flow ablation can be out of distribution. More recovered clip codes alone do not demonstrate reliable A5C evidence or clinical improvement.

Snapshot through 2026-10-28T21:16:17.149815+00:00: charged GPU wall **0.187643h/6**, CPU **1.171195h/3** (cap check **1.289250h** including the past reviewer-window upper bound), new server artifacts **0.570GB/3** plus<.01GB metadata allowance. A conservative all-job elapsed-span upper bound of.458160GPU h also covers unmetered shutdown tails; it includes idle and is not reported as spending. See resources.json and the ledger snapshot for actual measured stage/process CPU, all failed checks/flow preparation/training/readout/audit, bootstrap accounting and active-owner wall scope. Idle time is not charged as compute; unmetered process-tail/reviewer scope is disclosed separately with conservative budget bounds. Root terminal publication after the public cutoff is timed in the private final closing receipt. Aggregate files are checked server/local/git before closure; no active round jobs or GPU allocation remain. The heartbeat is removed on complete closure.
