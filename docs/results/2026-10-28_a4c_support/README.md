# A45: independent A4C support reconstruction as an A5C-edit veto

**Completed, independently checked, negative.** Four CPU support models each completed their original600 updates. No Claude, A44 retraining, new image inference, threshold sweep, source2 or A46. Fixed card9d178fe; codef865f36; checkscdcfff2. All four models, scalers, fixed plans, inner calibration rules and input/code identities were committed as actual git objects in **dd2595b** before freeze and one opposite-fold readout.

The independent NumPy full-network/data/rank/evaluator recount passed. Both the class-specific A4C model A and the nonspecific reconstruction control U veto all actual A5C recoveries. Their run labels are **bitwise identical**, verified per case (not inferred from aggregates). No primary or secondary absolute gate and no class-specific contribution gate passed.

| Policy on original Base runs | A5C recovered /103 | Correct A4C harmed /1413 | Net | Mean complete multi F1@50 | Short misses | Boundaries deleted |
|---|---:|---:|---:|---:|---:|---:|
| Base | 0 | 0 | +0 | 0.956885277 | 76 | 0 |
| N | 26 | 76 | -50 | 0.952000617 | 104 | 75 |
| N_A | 0 | 1 | -1 | 0.956692264 | 76 | 0 |
| N_U | 0 | 1 | -1 | 0.956692264 | 76 | 0 |
| C | 12 | 22 | -10 | 0.956122151 | 86 | 30 |
| C_A | 0 | 0 | +0 | 0.956885277 | 76 | 0 |
| C_U | 0 | 0 | +0 | 0.956885277 | 76 | 0 |

**Main result.** Against frozen A44 N, A vetoes75/76 correct-A4C harms but also all26 real A5C fixes:0 recovered,1 harmed, net−1. U makes the exact same single harmful edit; primary A−U is zero in both folds on every run label and full endpoint. With frozen C (the predeclared secondary comparison), both verifiers reject every proposed edit and reproduce Base exactly. This is abstention, not safe A5C recovery. Low harm without target recovery does not meet the question.

## Why this direction and what A44 diagnosis established

See [post-A44 diagnostic](../2026-10-28_post_a44_risk_diagnosis/README.md) and its `audit.json`. A44 loss/dual signs and all600 updates agree with the card; the final full-fit constraint nevertheless fails in both N models. In the last50 updates95–97% of negative-window score gradients are zero, with occasional strong active-negative gradients. Most final fit false promotions were already positive at their last encounter (N fit1:30/35); only2/35 were last seen≤−1. Thus drift after negative exposure contributes but does not explain most errors, and the diagnosis does not claim that a small total negative gradient is the sole cause.

Held-out C/N rank AUROC is.82184/.84661 versus.83173/.84425, while scores have correlation.98645/.97800 and a substantial positive shift toward promotion. C promotions are a subset of N promotions. The fixed finite nonconvex minibatch procedure did not enforce even the full empirical fit constraint, and changed the operating region without a consistent transferable ranking improvement. No clinical anatomy or label error is established. This motivated independent A4C-support verification rather than another multiplier/loss/learning-rate variant.

## Changed training problem and matched control

- Input is the complete original deployed P-run multiplicity-weighted mean of original A08 window embeddings (512 dimensions). This is deliberately frozen-representation training, not end-to-end pixel training or a new backbone. A08, original marker/crop/gamma/request support and every membership remain unchanged. The frozen A44 N proposal is primary; frozen C is a secondary comparison fixed before readout.
- A trains only on protected correct-A4C proper-fit runs, to reconstruct their standardized visual embeddings from Gaussian-noised inputs. U has the same architecture, initialization, noise,600 updates, optimizer/schedule and squared reconstruction loss, but learns nonspecific balanced A4C/A5C support. The first16 examples per update are common correctA4C; the last16 are a second A4C draw for A and an A5C draw for U. The comparison changes the learned support distribution, not a classification-loss coefficient.
- Model512→128→32→128→512, GELU after first3 layers,140064 parameters; all8 parameter tensors really updated. Clean-target denoising MSE, noiseσ=.1, AdamW lr1e−3/wd1e−4, cosine600, clip1, FP32/one CPU thread/eval mode. A common A4C-only proper-fit mean/std (floor1e−3) supplies normalization. No A5C classification gradient can push A toward A5C.
- Labels remain original majority clip codes. Not reconstructing like A4C is a necessary verifier condition, never proof of A5C or LVOT. Reconstruction models can also reconstruct unfamiliar samples; no general anomaly-detection guarantee is claimed.

## Inner isolation and calibration

| Outer fit fold | Proper-fit correct A4C | Proper-fit A5C (U only) | Held-in calibration correct A4C | Proper/calibration source sessions |
|---|---:|---:|---:|---:|
| 0 | 158 | 8 | 208 | 100 /100 |
| 1 | 226 | 17 | 160 | 100 /100 |

The split is fixed RNG over sorted source sessions. A run is usable for proper fitting/calibration only when **every source session in every16-frame original membership window** lies in that partition. Mixed-footprint runs are excluded only from inner fitting/calibration; all original749/771 eligible runs and all6,788 P runs remain in the outer readout. No session/ref/window footprint overlaps proper fitting and calibration. An independent implementation reproduced identities and exact plans; proper-fit/calibration IDs remain server-only.

Calibration uses protected-A4C reconstruction errors only, after600 updates. The fixedα=.01 rule chooses rank ceil(.99*(n+1)) and requires strictly greater error; p=(1+#calibration errors≥query)/(n+1), ties conservative. No positive or outer outcome selectedα, epoch, checkpoint or threshold. Correlated/reused runs and prior original A08 exposure mean the rank rule is an empirical safeguard, not a distribution-free clinical/run-wise protection guarantee.

| Model | Calibration n | Frozen threshold | Calibration outliers | Fit MSE init→final |
|---|---:|---:|---:|---:|
| A_f0 | 208 | 0.310488880 | 1 | 1.003582 → 0.026073 |
| U_f0 | 208 | 0.455981165 | 1 | 1.011511 → 0.042729 |
| A_f1 | 160 | 0.372942865 | 0 | 1.003702 → 0.029575 |
| U_f1 | 160 | 0.458738267 | 0 | 1.038590 → 0.046320 |

Fit reconstruction fell substantially in every model, so the zero target recovery is not evidence that training failed to execute. Per-model logs contain600 contiguous update records, finite losses/gradients, exact plan hashes and identical A/U noise hashes at every update; the optimizer asserted600 real steps in all8 tensors before final save. All final/resume identities and independent initialization/movement checks pass; no resume file remains.

## Frozen deployment and complete segmentation

Only an **original** Base-A4C run with frozen proposal r>0 and verifier error above its frozen threshold is changed to A5C. No reference labels/source identifiers enter deployment decisions. All other Base run labels are copied. A structural subset of N/C cannot itself establish learned protection, and RLE can still damage full segmentation. The same support models/calibration rules feed N and C; the secondary no-op cannot substitute for the failed primary comparison.

| Complete endpoint, fold0 /fold1 | Base | N+A =N+U | C+A =C+U |
|---|---|---|---|
| Multi F1@50 | 0.963906582 / 0.949863972 | 0.963520556 / 0.949863972 | 0.963906582 / 0.949863972 |
| Native macro F1 | 0.977556578 / 0.975831638 | 0.977556578 / 0.975831638 | 0.977556578 / 0.975831638 |
| Boundary recall | 0.967644085 / 0.968139774 | 0.967644085 / 0.968139774 | 0.967644085 / 0.968139774 |
| Boundary precision | 0.955566650 / 0.942000000 | 0.955566650 / 0.942000000 | 0.955566650 / 0.942000000 |

N+A has0/34 and0/69 S1 recoveries, one correct-A4C harm in fold0 and none in fold1. Its full F1 delta is−0.000386026/0; mean−0.000193013. Native/boundary endpoints and missed-short counts equal Base. All1,150 originally correct A5C and every non-A4C run label remain structurally unchanged. The primary policies fail recovery/source, net and full-segmentation signal conditions; the C policies also fail the explicit no-op condition. A−U has no class-specific increment.

## Independent verification and provenance

10 focused CPU unit tests passed after the expected missing-implementation red tests. Actual-data checks passed: full-footprint session separation/minimum counts; weighted run embedding vs explicit repeated-sample mean (≤3.6e−15);140064 parameters;8 real tensor movements; exact6-step repeat and3+3 optimizer/noise-RNG resume; forced identical input path equality; independent FP64 full-model MSE gradient expression error0. Plan advisory review approved the fixed card; no helper or Claude executed training.

The separate server auditor reconstructs all original memberships through a nearest-centre repeated-sample loop, full input footprints and inner partitions, proper normalizers and all600 batch identities. It independently runs the entire network using NumPy matrix products and erf GELU (not the training PyTorch forward), checks calibration/output errors and conservative rank/p-values, independently creates every run label, reassigns whole sequences and calls the original full evaluator. It independently counts true-A5C source recordings inside corrected runs, paired A/U fix/harm overlap, correct recoveries vetoed and every gate. Maximum calibration NumPy-vs-FP32 error is1.20e−7, with exact agreement on all deployment labels and every published endpoint. It independently rebuilds the architecture/initialization and verifies all8 final tensors moved and are finite in all4 models.

Readout is once, after model/rule manifest commitdd2595b and server freeze. No A44 inference was repeated; frozen C/N proposal caches were reused with SHA verification and original Base/C/N tables reproduced exactly. Server-only private cache SHA `20627bded4a1a158115f565a242e2c3c38ef769bcab47bdcb3858f35b449a279`; no private arrays, ids or media exported. `weights.json`, `freeze_provenance.json`, `checks.json`, `preparation.json`, `readout.json` and `independent_audit.json` are the aggregate evidence.

## Scope and decision

This closes the fixed source0 A08 frozen-embedding denoising-reconstruction support verifier andα=.01 veto. It does not show that one-class/generative learning generally cannot work, settle other representations or label correctness, establish equivalence, identify LVOT, or yield patient-independent/clinical evidence. These training-source folds have repeatedly informed research choices; A45 is an adaptive exploratory screen. Proper-fit A5C support for U is only8/17 runs, another limit.

No gate passed, no new candidate is advanced, no threshold relaxation or new variant follows. A08+ B3 remains the working reference. A43 M recovered2/broke4; A44 C12/22 and N26/76; A45 reduces edits toward no-op rather than solving safe A5C recovery.

## Resources and closeout

Snapshot through20:10:06UTC: **GPU0h/6; charged activeCPU0.446596h/3**, CPU cap check≤0.469374h including the bounded reviewer window; **~0.033GB/3** new artifacts. Terminal publication/verification after the cutoff is separately timed. See `resources.json` and `ledger.jsonl`: zero GPU; actual active CPU includes post-A44 diagnosis, planning/one advisory reviewer, implementation/tests/checks, four600-step CPU fits, calibration, one readout, independent full-model audit and closing. Import/bootstrap/SSH overhead is conservatively covered by active owner wall time; explicit idle waits are excluded and no future closing is precharged. Minor unused-test-import lint failure was fixed before formal training; no failed model job or scientific condition change. New artifacts are small (no persistent pixels or features from a new encoder). Final resource snapshot and terminal scope are recorded with actual timestamps.

Sole Codex execution in an isolated worktree, integrated by fast-forward with the8 staged theory files and unrelated dirty files unchanged. Round-only result/index commits, no push. Monitor deleted on complete closure; no automatic A46.

References are motivation and limitations, not reproduced algorithms: [Cotter2019](https://proceedings.mlr.press/v98/cotter19a.html), [Ruff2018](https://proceedings.mlr.press/v80/ruff18a.html), [Denouden2018](https://arxiv.org/abs/1812.02765).
