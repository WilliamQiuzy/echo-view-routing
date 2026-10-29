# A47 clinical multitask pretraining implementation plan

> **For agentic workers:** Execute inline with executing-plans and test-driven-development. A fresh-context advisory plan reviewer reads this card; Codex remains the only implementation/execution owner.

**Goal:** Test whether the released PanEcho clinical-multitask ConvNeXt-Tiny state supplies transferable A5C evidence beyond the same ImageNet architecture, under a fixed binary repair task protecting correct A4C and complete segmentation.

**Architecture:** Two deliberately frozen external image encoders supply final7x7 spatial maps from the exact original marker frame. Identical genuinely trained spatial/fusion models combine each map with the unchanged cached A08 window embedding512. P differs from G only in the named pretrained encoder state. This is a pretrained-representation comparison, not a claim to have LVOT labels, an anatomical detector, a new gate or a new loss.

**Tech stack:** Installed PyTorch/torchvision ConvNeXt-Tiny, pinned PanEcho state, existing A31/A44 data and original evaluator. No Claude, expert annotation, Unity training, model-output pseudo-label, dev/check/test/Pediatric/A26 read or push.

## Evidence, changed condition and limits

- Source intake plan/report: `2026-10-28-public-a5c-evidence-intake.md` and `docs/results/2026-10-28_public_a5c_evidence/sources.json`. EchoXFlow legally permits noncommercial research adaptation, but the inspected LVOT Trace points target spectral PW data. Static sampling-site metadata alone has not established a trustworthy frame-level B-mode target. No invented A4C absence label or EV9V A5C pseudo-label is used.
- A35: frozen EchoFine view outputs failed; P uses learned feature maps from human clinical measurement/report multitask training, not those outputs.
- A36/A39: A08 resolution/token pooling failed to add A5C evidence. A47 changes representation provenance and trains a spatial readout, not channel-max pooling of a CLS-trained projection.
- A40: masked-HOG auxiliary learned but no demonstrated correction increment. A41: EchoJEPA vs generic V-JEPA2 under900-update LoRA failed; these are self-supervised latent-prediction states, unlike PanEcho's39 clinical reporting tasks, including LVOT diameter and aortic-valve structure/stenosis.
- A43/A44: max/bag objectives and risk loss failed the target/harm trade-off. A45 reconstruction veto suppressed all target fixes. A46 added motion but found58/harmed161. Keep their scopes closed; do not reinterpret any as absence of anatomical evidence.
- This pretraining is not proved stronger on A5C in advance: the falsifiable P−G screen determines whether it transfers. P−G cannot isolate clinical anatomy from all differences in upstream data and training. Study-level measurement labels are not per-frame visibility labels. Source/pretraining exact overlap with EV9V is unknown. A31 folds are repeated training-source/session screens, not an untouched patient/clinical holdout.

## Sources fixed before any forward

PanEcho code836c46ef6bc4eace50b7dc36704ecef2d33affde, official `hubconf.py` and `src/models.py`: ConvNeXt-Tiny initialized from torchvision IMAGENET1K_V1 before clinical training; four-layer temporal model and39 task heads. Weight URL `https://github.com/CarDS-Yale/PanEcho/releases/download/v1.0/panecho.pt`, release asset505054604bytes. Weights CC-BY-NC-SA4.0; other repository materials AGPLv3. Keep this noncommercial research, attribution and upstream license notices; no public service or weight redistribution.

G uses installed torchvision `ConvNeXt_Tiny_Weights.IMAGENET1K_V1` (official983f1562 URL prefix). Acquire with no GPU forward; compute full SHA, strict image-encoder key/shape extraction and actual task-head identity, save complete source manifest and extracted frozen states. P extracts only `encoder.encoder.model.` with no random fill; G discards only its1000-class final linear weight/bias. Both full image encoders otherwise load strictly. Publish actual downloaded/extracted/code/torchvision SHA manifest and commit its real git blob before the first feature forward. Original large acquisition files may be removed only after that evidence is committed and verified to honor disk cap.

Use the final `.features` map before average pool and classifier LayerNorm. Do not apply a pooled-head normalization to patch tokens. Validate that official global average pool plus original classifier normalization reproduces each complete image encoder, and verify the selected map against an upstream read-only image-encoder hook. The temporal Transformer and all39 output heads are unused downstream, explicitly recorded.

## Input, model and optimization contract

Source0 first, original A31 two session folds. Original Base-A4C eligible runs749/771, need5030/5437 rows, fit positive34/69, protected correct-A4C714/699, other1/3 excluded only from fitting and included in full readout. Keep original16-frame windows, marker positions, crop/gamma, normalization, nearest-centre membership, multiplicities and full duration weights. Select new image inputs only from original Base eligibility; no P/G scores or reference label selects frames.

Resolve the exact reference at original marker slot (including fragment/gamma identity), decode the same clean224 RGB as the A08 loader, /255 then ImageNet channel normalization. P/G get identical RGB. Deduplicate only identical complete target-frame references; private reference maps bind every eligible row. Cache final768x7x7 maps in FP16 after FP32 deterministic encoder forwards; read as FP32. A08 appearance512 remains unchanged. Exact source2 Base row metadata may be prepared up front; union its eligible target references with source0 for a single label-free feature cache if the disk projection passes. Source2 outcome is never read unless its fixed gate fires.

Trainable model, identical initialization per source/fold in P/G:

```
per-token LayerNorm768 -> Conv2d768→16,k1 -> GELU -> Flatten784 -> Linear784→128 -> GELU
appearance LayerNorm512
concat640 -> Linear640→128 -> GELU -> Linear128→1
```

197521 parameters (verify actual binding), eval mode/no dropout, final weight0 and bias−2, so Init makes no edit and equals Base. Every trained parameter/group is bound to optimizer; record actual gradient and drift, structural zero coordinates separately. External encoders and A08 intentionally remain frozen; cached features are the specified experimental representation, not a claim that their trunks were fine-tuned.

Use exactly A44 C's600-update cyclic whole-run plan (one positive, three correct-A4C negatives), all negative runs covered, same rows and occurrences in both arms. For window scores s and complete weights v, r_j=sum(v*s) and L=softplus(−r_0)+mean_{j=1..3}softplus(r_j). Retained graph over every occurrence; independent individual-window J-transpose-v checks exact production gradient, including zero derivative occurrences. AdamW lr.001,betas(.9,.999),eps1e−8,wd.0001,60-step warmup then cosine600, clip1, FP32, TF32off, strict deterministic algorithms. No multiplier, margin, risk penalty, threshold/epoch/head search or best-checkpoint choice. Final600 only.

## Sequential stages and scientific decision

1. **Prepare/acquire/check.** Read-only source/data pins, all membership/marker/frame mappings, session isolation, plan hashes/exposure, InitBase; strict source loads and immutable manifests; meaningful tests with captured RED before core implementation. Repeat6, forced equal-feature P=G, optimizer3+3 resume, full-vs-individual chain rule, feature fresh/cache and official parity. No formal train until every critical gate passes. Forecast full shared GPU/CPU/disk including failures/bootstrap/audit before launch.
2. **Formal source0 training.** One queue, four actual600-step heads. Preserve failed checks; no formal retraining or labels-driven repair. Log PID/device/u/real optimizer counters and actual input-row hashes.
3. **Commit then freeze.** Full four-model/head/frozen-P/G/A08/config/code/input/plan/check SHA manifests are actual local git objects before server freeze and once opposite-fold score readout. Server has no project.git; Codex verifies its source blob locally against server file SHA and records times. Private score cache is authoritative for later recount; never repeat a full held-out forward for auditing.
4. **Deploy/readout.** Only original Base-A4C with full weighted-mean scalar r>0 becomes A5C; r=0 stays Base. All original non-A4C labels copied bitwise; no extra valve veto, confidence threshold or proposal gate. Equal neighbors merge. Structural preservation is not learned protection; evaluate complete sequences, boundaries and short segments.
5. **Absolute package for either arm:** target recovery>=ceil(.10*S1) with>=3 actual A5C-coded source recordings and>=1 per fold; correct-A4C harm<=floor(.01*correctA4C); per-fold net>=0 and OOF net>=5; complete F1 not below Base in either fold and mean delta>=.001; native macro-F1 delta>=−.003 and boundary-recall delta>=−.005 each fold; total short true misses<=Base+3; original non-A4C labels unchanged. Denominators/classes/original Base sources are independently reconstructed, not inferred from aggregate equality.
6. **Separate pretrained-state increment:** P target fixes>=G in both folds and OOF>=G+5, P harms<=G in both folds, OOF net P−G>=5, F1 P>=G both folds, and absolute P passes. This is distinct from either absolute package.
7. **Source2 conditional:** only if either source0 arm's full absolute package passes and the same shared budget forecast permits it, train/read both P/G for source2 under exactly this card, same600 plan and source-specific original Base/denominators. No seed/state/input/loss variant. Otherwise close at source0. No automatic A48.
8. **Independent audit and closure:** separate functional head/gradient calculations, original nearest-centre expanded memberships, FP64 weighted means, legal predictions, true source attribution, paired P/G fixes and harms (including unique/shared cases), original full evaluator/F1/boundary/short counts, gate recomputation,600 update/input hashes and frozen digests. Verify public aggregate SHA server/local/actual git blobs, final resource receipts and no remaining own jobs. Brief Chinese checked result versus G/Base and same-denominator A43–A46 descriptions; no causal LVOT claim.

## Shared resource cap

**Pre-forward release-identity correction:** official prose describes39 tasks, but the exact published v1.0 checkpoint contains40 named clinical heads, including LVOTDiam and AVStructure. Initial acquisition stopped before any forward at the original39-head assertion. Preserve that failed receipt; require the actual40-head identity and record every name. All task heads and the temporal Transformer remain unused. This corrects the source manifest contract before computation and changes neither encoder state nor model/input/loss/plan/deployment conditions.

GPU6h, CPU3h, new disk3GB for the entire search/intake/A47 task, including prior intake metadata/ranges, failures, startup, source2 if triggered, active owner/reviewer work and actual closing. Disclose unmetered initial source-search scope and separately bound it; no idle scheduled wait charged as compute and no future time precharged. Strict pre-feature map byte forecast; remove only re-downloadable original acquisition duplicates after committed SHA evidence if needed. Do not delete old round artifacts. Record process CPU separately from GPU stage wall and root active-wall proxy, with explicit as-of bounds.

## Implementation tasks/files

- [ ] **Core/tests:** `echo_routing/temporal/clinical_pretraining.py`, `tests/test_clinical_pretraining.py`. First test marker/reference selection, strict extraction, spatial head/Init, exact full membership loss/gradient, legal one-way deployment and repeat/resume/forced-equal. Observe RED; implement minimally; run focused tests, no unrelated suite.
- [ ] **Stages:** `scripts/train_clinical_pretrained_a5c.py`; config `configs/experiments/video_clinical_pretrained_a5c_v1.yaml`; sole namespace `runs/video_clinical_pretrained_a5c_v1`. Reuse verified A31/A44 loading/evaluator functions read-only, not their output writers. Actual external-source SHA before feature generation; checks before fit; actual manifest git commit before freeze/readout.
- [ ] **Audit/report:** `scripts/audit_clinical_pretrained_a5c.py`, aggregate result folder `docs/results/2026-10-28_clinical_pretrained_a5c/`. Private images/maps/identifiers/step and case outputs server only. Preserve unrelated dirty/staged content; round-only commits, no push. Update only A47 index entry after verified outcome.
