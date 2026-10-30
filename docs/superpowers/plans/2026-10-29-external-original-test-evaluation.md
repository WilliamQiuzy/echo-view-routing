# External Original-Test Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Execute every obtainable, task-compatible original-test comparison for the frozen strongest cardiac system, publish concise evidence tables, and retire misleading presentation entrypoints.

**Architecture:** One execution owner controls GPU inference. Read-only helpers audit official source availability and old presentation dependencies. Freeze input definitions, all three A08 states, decoder numbers, adapters and metrics before new test inference. Original-paper endpoints, shared-label endpoints, auxiliary source-shift checks and unavailable/incompatible datasets remain separate. Public artifacts contain aggregates only.

**Tech Stack:** Existing PyTorch MViTv2-S, NumPy/Pandas/Pillow/OpenCV, pytest; static aggregate-only HTML/CSS; Sites private hosting.

---

## Authorization, predecessors and fixed scientific scope

User explicitly authorizes acquisition of obtainable external-model original test sets, running the strongest existing model, reliable comparisons, a concise multi-table webpage, and cleanup of obsolete demos, without further questions. This is new evaluation work, not an automatic continuation of A49 or an optimization experiment.

Predecessors: A08 (working reference, all-neighbour MViTv2-S plus saved B3); A26 (historical EV9V same-benchmark comparison, not external confirmation); A31-A48 (no promoted replacement); A49 (LVOT source-binding qualification failed before training). Preserve their frozen outputs, decisions and private raw data. Latest STFM code and historical pinned code are distinct conditions. Frozen A08 has five outputs; no invented nine-/38-/58-class or EF/action/localization score.

**Model selection:** all_s0, all_s1, all_s2 from A08; never choose a seed on new data. Individual models and mean reported; no new ensemble selection. Weights and B3 numbers pinned by SHA before inference.

**Data:** original EV9V test, same official split, five-family shared support (800/888); full nine-code STFM original endpoint is a separate baseline-only check. Original external test data with five-family cine truth are admitted after source review. EchoNet-Dynamic official TEST A4C, EchoNet-LVH official test PLAX and CAMUS official testing A4C are useful auxiliary source-shift checks, not substitutes for EF/wall-thickness/quality endpoints or balanced five-family tests. No newly fabricated image-to-video input or action-to-cardiac mapping. No Pediatric training/derived export; no private identifiers in reports or hosting. No contacting authors.

**Readout:** fixed per-video majority over 10 Hz predictions, raw argmax and frozen B3 separately; supported class recalls, full five-class confusion, native false cuts and durations. Macro-F1 only where the complete shared label support is appropriate. Source checks with one true class get class recall and its denominator, not a misleading five-class ranking. Exact decoded frame counts and timing; no failed-case dropping. Bootstrap by patient when known, otherwise session or recording with explicit limitation. Source overlap unknown stays unknown.

**Resources:** initial shared evaluation envelope 12 GPU h, 12 active CPU h, 8 GB new persistent storage, one GPU queue, memory fraction 0.6. Time actual jobs/checks/failures and retain receipts. Stage acquisition and measure a label-free timing subset before scheduling full work. A resource limit is a pending condition, never an unavailable/incompatible disposition or a completion claim; revise a resource forecast prospectively if required by the authorized scope. Do not train a new head or tune thresholds to rescue outcomes. Acquisition checks stop on unavailable pixels/truth or an incompatible target; record the exact missing prerequisite rather than substituting pseudo-labels.

## Task 1: Availability and preparation

**Files:** create `configs/experiments/external_original_test_v1.yaml`, `docs/results/2026-10-30_external_original_tests/availability.json`; private server manifests under `runs/external_original_test_v1/private/`.

- [x] Audit each external model against official source/test access and target compatibility. Starting roster: STFM, EchoViewCLIP, EchoPrime, ASFormer, MS-TCN, ASRF, FACT, EchoFine/TTE47, EchoJEPA/V-JEPA, PanEcho and AoVdetector; also record intake-only representation models and resources separately.
- [x] Pin source versions, all model SHA256s, native split membership and decoder numbers.
- [x] Commit protocol/config before new target inference; log access and preparation receipts.

## Task 2: Reliable inference adapter

**Files:** create `echo_routing/evaluate/external_native.py`, `scripts/external_original_tests.py`, `tests/test_external_native.py`.

- [x] Write failing tests for FPS-aware time sampling, exact target positions and short windows, unsupported-label error accounting, per-video weighting, and refusing incompatible source/cache identities.
- [x] Run focused tests and confirm missing implementation failure.
- [x] Implement native JPEG/video/NIfTI adapters with strict source timing, original crop/normalization and a label-free forward path.
- [x] Verify online A08 against stored development logits; verify frozen state, batch parity, timestamps and original decoder parity. Run all focused tests.
- [x] Freeze code/config identity, run all admitted sets for all three seeds through one resumable queue; no target-dependent variant.

## Task 3: Baseline endpoints and independent verification

**Files:** report JSON/CSV/README and checks under `docs/results/2026-10-30_external_original_tests/`; private predictions and memberships remain server-side.

- [x] Evaluate original-native STFM and available official cardiac classifiers on the exact same common labels; separately preserve original task endpoints and code versions.
- [x] Recount metrics from frozen per-case predictions with an independent implementation and fixed manifest cardinalities; verify exclusions and failures, uncertainty units and source overlap disclosures.
- [x] Include existing A26 native/pair/multi, class recalls and short-segment errors as historical context, with exact seed/population labels.

## Task 4: Presentation and cleanup

**Files:** aggregate-only builder/result page, new standalone private Site source, demo legacy notices/entrypoints, cleanup manifest.

- [x] Build concise tables directly from verified aggregate JSON: original-test availability, completed shared-task scores, auxiliary source-shift scores, native/pair/multi context, timing and limitations.
- [x] Remove dependency on unavailable case/video bundles; include source links and downloadable aggregate CSV/JSON only.
- [x] Archive an enumerated set of clean tracked obsolete generated demo pages and builders by dependency, with traceable paths; preserve annotation tooling, original reports and ALL pre-existing untracked files and unrelated dirty/staged work.
- [x] Validate figures/denominators/links, desktop/mobile rendering, no runtime errors, and no per-case identifiers/media.
- [x] Publish private Site, verify deployment and current local entrypoint, update A50/index and final resources. Complete only after every obtainable compatible comparison has a verified outcome and all other models have an evidence-backed access/compatibility disposition.

## Closure — 2026-10-30

Completed A50: 2,470 source sequences × three fixed A08 seeds and STFM original888; independent full-label/metric/hash recount passed, 2,000 grouped bootstrap draws, 26 relevant tests, desktop/mobile browser checks. No new independent five-class/A4C-A5C cine test was obtained. Prospective all343-source-PTS LVH timing repair is recorded separately; all failures remain retained. The owner-private Site is deployed successfully, and both old public entrypoints now serve retirement notices (HTTP200, exact marker verified). See the A50 README, resources, ui_validation and publication JSON. Seven clean tracked generated demo files/builders are retired; pre-existing dirty/staged/untracked work is preserved. Source audit/development/agent overhead is not fully metered and is explicitly excluded from complete-resource-cost claims.
