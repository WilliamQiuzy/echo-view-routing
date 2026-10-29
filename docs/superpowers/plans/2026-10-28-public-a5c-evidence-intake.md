# Public A5C evidence intake implementation plan

> **For agentic workers:** Execute inline with the executing-plans skill. One fresh-context advisory plan reviewer checks this document; no second execution owner or training queue.

**Goal:** Determine whether an actually available, research-training-permitted anatomical resource adds a substantive A5C supervision/representation condition before authorizing computation under an A47 model card.

**Architecture:** Metadata-first source verification, followed by a bounded sample/schema audit of the strongest new resource. This intake is not A47 training and produces no EV9V model scores. A47 can start only after a separate fully specified, pre-output matched card and engineering gates pass.

**Tech stack:** Official source pages, pinned GitHub/Hugging Face metadata, Python standard-library download/JSON/tar inspection, existing repository result conventions. Sole owner Codex, per direct user instruction; no Claude dependency.

## Scope and predecessor evidence

User authorized a search without expert annotations, conditional A47 execution, monitoring and independent verification. AGENTS.md and the untracked docs/EXPERIMENT_WORKFLOW.md were read. Preserve eight staged theory files and all unrelated dirty files; use this external worktree and round-only commits, no push.

- A35: EchoFine's fine-grained view classifications did not safely recover S1. Predictions are not anatomy labels.
- A36/A39: frozen higher-resolution/global-local A08 descriptors did not add A5C recovery; neither rejects anatomical supervision.
- A40: masked HOG learned its auxiliary target without demonstrated target gain.
- A41: EchoJEPA vs V-JEPA2 with this 900-update LoRA recipe failed. A larger backbone name alone is insufficient.
- A43/A44: positive bag max and risk-loss changes did not solve target/false-promotion trade-offs; full-fit constraints were not satisfied in A44 N.
- A45: reconstruction veto removed all recovered A5C; do not add another gate on the old embedding.
- A46: optical-flow representation changed cases but F recovered58/harmed161, below Base complete segmentation. Motion is not anatomical supervision.
- The expert pilot has zero real records. Unity data remain evaluation-only under the existing repository decision; this intake does not revise it. No dev/check/test/Pediatric/A26 runner or private media/identifiers leave the server.

## Tasks

- [ ] **1. Verify source identity, license, availability and label semantics.** Save official URLs, immutable revisions where provided, retrieval UTC, text hashes and a concise license/use matrix in `docs/results/2026-10-28_public_a5c_evidence/`. Distinguish article/code/weights/data licenses. Do not infer that an ND license bans every internal training use. No messages to authors or external uploads.
- [ ] **2. Audit EchoXFlow's metadata on the server only.** Pin dataset7a36922880061791519a8f01553c15cad5df564f and code61f027e02118106379682d2cf286faa060130541. Metadata is705276151bytes, whole data archive479281111040bytes; never snapshot-download the entire dataset. Download the single metadata file only if its pinned manifest SHA and advertised size agree. Parse actual annotation array names/counts, recording-mode bindings and patient split provenance. Emit aggregate counts only.
- [ ] **3. Inspect a bounded schema sample.** Use an original-Doppler LVOT-trace annotation and associated sampling-site metadata, selected from metadata alone without EV9V scores. At most one examination archive, chosen by an explicit metadata rule, maximum100MB fetched by byte ranges (no entire oversized archive). Preflight uncompressed tar format and a usable bounded member-offset/header path. Verify returned Content-Range, requested lengths, tar boundaries and coordinate/time bindings. Do not search an unbounded compressed prefix. If valid range support or exact site identity cannot be established, stop this sample stage; do not substitute a model-classified Other/A5C view or spectral trace for a B-mode anatomical point. Store data only below server `runs/a47_public_evidence_intake_v1/`.
- [ ] **4. Check alternative pretrained states.** AoVdetector has CC-BY-NC-SA weights but its official training views are PLAX/PSAX/3CH; establish actual label/training provenance and whether any credible apical A5C transfer evidence exists. PanEcho has39 study-level tasks, not anatomical localization. EchoFM, EchoApex/EchoVLM and MedSapiens need distinct weight availability/license/task evidence; published scores alone are not obtainable training resources. No weight download or EV9V forward in this intake.
- [ ] **5. Independently verify the intake and decide.** Recompute counts from pinned metadata with a separate aggregate path; compare official license language and label unit against the matrix. Record each resource’s overlap with EV9V and protected sources separately from its internal grouping; unavailable overlap remains unknown and limits independence claims. Link each factual finding to its source. Select at most one substantive anatomical-supervision/pretraining direction, and document what it can and cannot identify. If the training input/target contract, permission, independent grouping or feasible computation is missing, record A47 not started and the exact missing evidence, without manufacturing labels or requesting a routine approval.
- [ ] **6. Conditional next card only.** If all evidence gates pass, write A47's architecture and matched control (same downstream data/compute/capacity), plus either its verified supervised anatomical target or a substantive documented pretrained-state contrast, input contract, training plan, model SHA/freeze-before-once-readout rule, full-segmentation and correct-A4C protection gates, independent audit and resource budget before execution. Otherwise close this intake with no A47 model or fabricated result. Do not automatically run A48.

## Resource and evidence gates

Intake cap: GPU0h; CPU1h of newly metered process CPU plus owner active wall from this card's creation (combined conservative accounting), download/storage1.1GB; sample range traffic <=100MB; no future time precharged. Earlier source-search effort is separately disclosed as an unmetered pre-card exploration span, not quietly omitted or claimed measured. A47, if justified, requires its own explicit shared cap before any model forward/training, including failed checks and closing audit. Network wait is separately recorded, not GPU time.

Before every dependent stage: inspect the preceding receipt, validate hashes/limits and record failed/partial attempts. File existence or a running CLI is not successful acquisition/model computation. Outcome of this intake may legitimately be no training start.

## Files and verification

New files: this plan; `docs/results/2026-10-28_public_a5c_evidence/README.md`, `sources.json`, `inventory.json`, `audit.json` and reproducible metadata inspection code if used. Public summaries contain no recording/session identifiers or per-case labels. The server contains source metadata and any permitted sample. Verify all copied aggregate SHAs against their actual git blobs before closing; record current head and preserved staged-diff SHA. No broad repository rewrites or unrelated tests.

## Pre-sample amendment (metadata only)

The pinned manifest contains36036 recording entries,728251 arrays and no LVOT-named array path; annotation paths are encoded, so zero name matches is not absence of LVOT annotations. The fixed schema-sample rule is now the lexicographically first examination with at least one recording containing both PW Doppler and B-mode; inspect consolidated metadata of its PW+B-mode recordings only. Stop at the first verified LVOT-measurement/site binding or at the original100MB/one-exam limit. No model view classifier or EV9V outcome selects this sample. Report sampling coverage rather than extrapolating its label availability to all examinations.

## Engineering amendment before sample attempt2

Attempt1 reached its300-request implementation ceiling after19660800bytes and retained its failed receipt. Exact HTTP206/range/USTAR checks passed. The physical sample remains the same one examination and total sample traffic remains<=100MB: attempt2 gets<=80MB,1200requests,900seconds and persists fetched pages. Only its PW+B-mode recordings with measurement arrays are requested, selected using the already downloaded metadata, not a view or outcome. An extra512-byte range preflight is also counted. No model condition, scientific gate or forbidden data changes.
