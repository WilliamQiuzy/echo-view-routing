# A5C image-dependence diagnostic implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. A single advisory plan reviewer is read-only; Codex is the sole implementation/execution owner.

**Goal:** Determine which observable input regions and preprocessing choices affect frozen A08/A44 C/N A5C decisions, especially harmful A4C promotions, without inventing anatomical truth or training another loss/gating variant.

**Architecture:** Preserve all prior models/readouts. Metadata-only selection precedes pixels; a committed public selection/model/code manifest precedes inference. One server-only diagnostic cache records original and paired interventions; a separate auditor independently reconstructs membership and numerical summaries. All raw frames, identifiers, maps and per-case results remain on echo-l40s.

**Tech Stack:** Existing A31/A44 Python loaders, PyTorch/torchvision/Pillow/NumPy; synthetic pytest checks; one bounded SSH-owned GPU process.

## Authority, history, data and resources

- User on 2026-10-31 authorized the proposed image-dependence audit, autonomous evidence-driven iteration and several hours of work without returning for confirmation. The supplied Site is read-only context, not a new training-data grant or permission to tune on external test sets.
- A32 (`2026-10-25_a5c_source_external_evidence`) established 103 source0 missed-A5C runs, concentrated sources, valid mapping/windows and no expert visibility truth. Its 24-case review package was unreviewed. A43 attention was not anatomical localization. A44 (`2026-10-28_risk_constrained_a5c`) C12/22 versus N26/76 fixes/harms and its post-hoc risk diagnosis analyzed scores/gradients, not image attribution. A48 localization-pretrained R14/50 did not protect A4C. PW-P1's 10/12 clinical-purpose source result does not identify anatomical cues and remains closed.
- Website identified through authorized Sites read: `appgprj_6ae41741b814819197c33307729a762b`, published v6. HTML SHA256 `80a166e57b127ecef8cbf4a92ce202b8f99e81a26be6b5a7f437d95a6d7e09da`, data.js SHA256 `0622938060f551944dd28ad4711aba52c29dd19e4739e80bc2a931c55543f48c`. Its published EV9V/STFM and EchoNet/CAMUS results are contextual aggregates; no underlying external test pixels, scripts or labels enter this audit. No Site changes/publication.
- Only immutable A31 training-pool session folds0/1, seed0, original A08 and four A44 final models/cache. No dev/check/test/Pediatric/A26 runner/Unity training, new source download, expert pseudo-label or teammate-task access. Outcome-conditioned exploratory diagnosis is NOT a fresh heldout evaluation or efficacy claim.
- Old cumulative resource ledger is retained at `runs/pw_small_pilot_v1/final_resource_receipt.json`: CPU upper2.9321053071189715h, GPU job-wall upper0.010121472941504586h, historical new-storage peak upper2218716939bytes. Its old3h CPU cap and closed results are not rewritten. This new several-hours authorization is an **additive** diagnostic allowance of at most3 active CPU-upper hours (including owner/reviewer/bootstrap/failures/audit/closing), at most1 GPU job-wall hour and at most600MB new persistent files. Cumulative CPU upper cap5.9321053071189715h, cumulative GPU cap remains6h, cumulative new-storage cap remains3GB. Empty waits0; no future charge; all new receipts show inherited plus increment. A15-minute conservative past preflight owner allowance will be included and clearly labeled if exact turn-start timing is unavailable. Stop before consuming audit/closing reserve.
- No second training executor/queue. Fresh ps/GPU inventory precedes launch. No training, checkpoint selection, threshold search or automatic deployment in this diagnostic. If a concrete intervention warrants subsequent training, write/review a new prospective matched card within remaining explicit resources and keep already inspected cases diagnostic; never claim its reused cases are independent confirmation.

## Fixed scientific contract

1. Metadata-only cohort: per original heldout fold, use N's frozen opposite-fold cache to define **A5C recovered**, **A5C missed**, **correct A4C harmed**, **correct A4C retained** among original Base-A4C P runs. Select up to8/group/fold by RNG `[20261031,51,fold,group]`, preferring distinct majority-source sessions, then distinct recordings, then remaining permutation entries. No image/quality/confidence-ranked selection, no replacement after seeing pixels. Record complete cohort denominators and each selection level; duplicates of sources across groups remain disclosed.
2. Preserve every selected run's original nearest-centre row membership and multiplicity weights. Use all its member windows for original and global interventions. Choose one spatial/temporal anchor per run by largest membership weight, ties smallest original row index; anchor maps are explicitly representative-window findings, not whole-run anatomy.
3. Load A08 and the appropriate fit(1-heldout) C/N finals strictly. Verify full source/model/head/base/input/cache SHAs. Public SHA manifests must be git objects before the first new model/pixel forward. No model gradients/optimizer steps are needed. Model state must remain exact.
4. Original-window outputs must reproduce the existing A31 logits (A08 A5C minus A4C) and A44 scalar cache on every selected unique row within1e-4 absolute. Recompute original weighted run scores and group membership independently. On a parity failure, stop affected inference; preserve failure; repair only an implementation defect, never retune the scientific condition.
5. Actual video input: original 16 request frames, target marker unchanged, original gamma, resize short side256 / centre crop224, EchoPrime 0-255 normalization. This is distinct from PW-P1's static-source square-resize/ImageNet pipeline. No claim of equivalence to an unidentified Helena version.
6. Global interventions fixed before outputs:16-pixel peripheral band mask and equal-area hash-permuted spatial control; chromatic-pixel proxy mask and equal-area control; static-bright-pixel proxy and equal-area control; grayscale; marker-frame repetition; full-frame aspect-preserving224 letterbox and matched-scale full-frame canvas with content outside the original crop footprint removed. Pixel masks are label-free and use the same spatial support over all16 slots. Original target marker is never masked or moved. Colored/static-bright regions are **proxies**, not verified text, calipers or anatomical masks. Report masked area and empty-mask frequency. Missing OCR is not negative evidence about overlays.
7. For each anchor/model:7x7 nonoverlapping32x32 spatial cells, occluded across all16 frames with two replacements (per-frame channel mean; local17x17 blur), plus four4-frame temporal-block mean replacements. Compare each model to itself. Compute a deterministic weight-permuted readout on the SAME embeddings as a limited head-randomization sanity control; no extra training. No cross-model raw-score scale comparison. Heatmaps are sensitivity, not proof of anatomical recognition or causal disease mechanism; replacement agreement/relative location/head-control differences are reported without semantic labels.
8. Raw-vs-processed contact sheets, masks and per-case maps/quality proxies are server-only. No automatic relabeling, dropping cases, or inferring LVOT presence. Provide interpretable aggregate crop retention, resolution/contrast/temporal-variation/chroma/static-bright proxies and stratified effects. These proxies do not certify clinical image quality.
9. Descriptive endpoints: baseline parity; per-group/fold/model weighted score shift and zero-crossing counts; spatial peripheral/interior sensitivity per equal-area cell; two-fill agreement; limited head-randomization map correlation; temporal-block effects; source/session dependence and preprocessing retained-area summaries. Publish fixed denominators, medians/quantiles and paired counts. No significance, patient-count, corrected full-F1 or safe-repair claim from this selected subset.

## Tasks

### Task1: pure intervention and cohort functions
**Create:** `echo_routing/temporal/image_dependence.py`; **Test:** `tests/test_image_dependence.py`.
- [ ] Write synthetic tests first for outcome masks, reproducible source/session-balanced selection, exact all-member duration means, disjoint grid coverage, equal-area controls, marker/time preservation, chroma/brightness empty masks, two replacement operators and crop-footprint letterbox geometry.
- [ ] Run focused tests and observe expected failures before implementing the new module.
- [ ] Implement deterministic NumPy/Pillow operators; run tests and commit.

### Task2: metadata preparation and provenance freeze
**Create:** `scripts/diagnose_a5c_image_dependence.py`; `docs/results/2026-10-31_a5c_image_dependence/`.
- [ ] Prepare only metadata and hashes on server, pin old-cache/model SHAs, select cases and original memberships; save server-only selection and public aggregate manifest.
- [ ] Independently validate frozen groups/counts/membership/source grouping before committing public input/model/code manifest.
- [ ] Save actual git-object freeze receipt on server before dependent pixels/inference. Source tree on server lacks.git; Mac is source of truth.

### Task3: bounded frozen-model diagnostic
- [ ] Decode actual inputs and retain private raw/processed contact sheets. Run parity and original predictions first; stop on mismatch.
- [ ] Execute fixed global/anchor interventions once per unique input/model, store private score cache before summaries. Atomic progress lists actual GPU PID/device/rows/stage, not just CLI ownership.
- [ ] Record measured process CPU, GPU-job wall, failures and disk deltas. Do not mutate code during running job. If backgrounded, attach one matching monitor; no duplicate launches.

### Task4: independent numerical audit and evidence decision
**Create:** `scripts/audit_a5c_image_dependence.py`; **Test:** `tests/test_image_dependence_audit.py`.
- [ ] Independent code reconstructs selection and weighted score effects, score flips, heatmap/control summaries and hashes from original private inputs/cache; no repeated complete model readout.
- [ ] Run synthetic counterexamples (wrong membership/area/weights/prediction signs rejected). Verify models/base unchanged and all scalar aggregate outputs match.
- [ ] Separate observed dependence, suspected shortcut and unresolved anatomy/label question. Do not force one unique explanation or train a blind variant when observations do not identify it.

### Task5: closure and authorized next action
- [ ] Write concise checked report, public manifests/audit/resources; add one diagnostic index row preserving every unrelated byte, eight staged theory files and other dirty paths. No push.
- [ ] Integrate own commits safely, verify server/Mac/git SHA mirror, no owned processes, remove temporary worktree/branch and diagnostic monitor when fully closed. Keep independent Issue4 watch within its remaining original allowance unless scope/deadline closes it.
- [ ] If evidence supports another substantive training comparison, specify it prospectively and execute within additive authorization/resources without waiting for the user. Otherwise report the exact evidence gap and completed diagnosis rather than invent a training success.
