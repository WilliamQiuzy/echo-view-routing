# Paired FACT and ASRF training on frozen E1 features (2026-10-08)

**Status (2026-10-09).** The configuration, code, preparation and smoke checks are done. One requirement of the FACT
overfit check was not met at its fixed length, so the queue has not started and the lead's decision is awaited
(section "Checks before the queue"). No formal training has started.

This round trains the official FACT (Lu and Elhamifar, CVPR 2024) and the official ASRF (Ishikawa et al., WACV 2021) on
the same frozen E1 512-d per-frame features, three seeds each. Both systems then go through one evaluation on the two
development halves, beside B3 and the archived references. Both models are prior work. Training them is not a
contribution of this project. The comparison is between two complete trained systems with their own optimisers, so a
difference cannot be attributed to cross-attention or to the matching loss alone.

## The fixed settings

The lead fixed four decisions (`docs/superpowers/plans/2026-10-08-fact-asrf-training.md`). The configuration is
`configs/experiments/fact_asrf_paired_v1.yaml`.

| Item | FACT | ASRF |
|---|---|---|
| Input | frozen E1 512-d features (short side 256, centre crop 224, numeric frame order, 10 Hz) | the same |
| Training data | 4 bank streams and 4 native windows (at most 256 samples) per update, from one plan per seed | the same plan, update by update |
| Seeds | 0, 1, 2 | 0, 1, 2 |
| Updates | 18,750 (50 epochs of 375) | 18,750 |
| Model | official FACT, `configs/default.py` + `gtea.yaml`, 12 tokens, `trans=false`, one-to-one matching | official ASRF, 512 input channels |
| Optimiser | Adam, lr 1e-4, betas (0.9, 0.999), eps 1e-8, no weight decay, clip 10 | AdamW, lr 3e-4, weight decay 1e-4, clip 1 |
| Learning rate | 1e-4 for the first 11,719 updates, then 1e-5; no warm-up | 750 warm-up updates, then cosine to 0 |
| Determinism | fp32, TF32 off, cuDNN deterministic, deterministic algorithms | fp32, TF32 off, cuDNN deterministic |
| Checkpoint | the weights after update 18,750 | the same |
| Decoding | official output (frame-branch weight 0.1), no search | official boundary refinement at 0.5, no search |

- **The GTEA base.** The help of the official `utils/gen_config.py` names it for small datasets. It has three blocks,
  128-wide branches and its own dropout and masking. This is a resource choice made in advance. It is not evidence
  that GTEA works better, and it is not a pure parameter-count contrast with the Breakfast base.
- **The decay step.** 11,719 is `ceil(18750 x 250 / 400)`, the official GTEA ratio of epoch 250 to 400 applied to this
  round's budget. It is an adaptation, not the paper's training recipe.
- **ASRF's smoothing weight.** The Gaussian-similarity term enters at weight 1, because the official `train.py`
  passes `gstmse_weight=True`. The 0.15 written in the L1 + ASRF configuration was a recording error. That report now
  carries a dated note, and its models are unchanged.
- **Determinism.** ASRF's cross-entropy has no deterministic CUDA kernel, so its training keeps the project's earlier
  setting. Both arms are decoded with deterministic algorithms.
- **Budget.** The six formal runs may use at most 13 GPU hours. The queue checks the projection before every run.

## Order of work

1. The configuration, the three entries (`scripts/train_fact_asrf.py`, `scripts/queue_fact_asrf.py`,
   `scripts/report_fact_asrf.py`), their tests and the task sheet are committed before any preparation or smoke check.
2. `--prepare` writes the seed plans and initial states and reuses the integration stage's alignment.
3. The smoke checks run on the GTEA FACT and on ASRF. After an independent implementation review, the checks are
   committed and the queue starts.
4. After all six runs, the final weights are frozen and the report entry evaluates once.

## Checks before the queue

- **Preparation.** It ran twice with identical files. Seeds 0 to 2 have plans of 18,750 updates, and the six initial
  states hold 10,215,424 parameters (FACT) and 1,191,768 (ASRF). All 56,250 plan batches were unpadded and matched
  their sequences. The integration stage's alignment was reused, because the protocol, the feature manifest, the
  cache key and the bank digest are unchanged (`preparation.json`).
- **Tests.** All 25 tests ran on the server, with none skipped.
- **ASRF smoke (current code, stamped).** It passed. The same-state rerun differs by 2.4e-7 in the loss, and the
  3 + 3 resume by 7.5e-9 in the weights. Padding labels change nothing. The optimiser is AdamW as configured, and no
  parameter lacks a gradient. One update takes 0.09 to 0.11 s (the second measurement shared the GPU), or 0.47 to
  0.57 h per run (`smoke_ASRF.json`).
- **FACT smoke (GTEA, code of 5863c6a).** Every check passed except one requirement of the overfit (`smoke_FACT.json`).
  - The same-state rerun and the 3 + 3 resume differ by 0. Padding labels change nothing, and the unpadded inputs
    equal the sequences. The optimiser is Adam as configured.
  - Inference label isolation found no change in 864 comparisons on the initial model and 864 on the overfit model.
  - At initialisation, only the three structural parameters of the input block's first token self-attention lack a
    gradient. After the overfit they have one, because the GTEA base uses attention dropout.
  - One update takes 0.596 s on average (median 0.589, 90th percentile 0.674, slowest 0.749) with 1,251 MB at peak,
    or 3.11 h per run.
  - **The requirement not met.** After the fixed 1,000 updates, all 16 transcripts were right, segment F1@50 was 1 for
    all 16, and the matching was one to one. In 2 of the 16 sequences, both with a single reference segment, all 12
    tokens predicted null. The official output fell back to the frame branch and was right, but the matched token's
    class was null.
- **Diagnostic (bc45d88, committed before it ran).** The same overfit was replayed with identical settings. At update
  1,000 it reproduced the smoke's loss exactly, and it continued to 3,000 updates. Every requirement was met from update
  1,500 onward (`fact_overfit_extension.json`). The failure at 1,000 updates is therefore too few updates for this small
  check, not an implementation error.
- **Implementation review.** It found no critical or high issue. Its medium note, that the queue trusted smoke reports
  without a stamp, was fixed in a06d037. The ASRF smoke was rerun on that code. The FACT smoke is rerun once the lead
  decides.
- **Shared GPU.** Since 01:18 server time, a preprocessing job of another project (FastSAM, in
  `/home/william/hli-echo-segmentation`) has shared the GPU. Timings after that may be slower.
- **Budget.** The estimate is 3 x 3.11 h + 3 x 0.47 to 0.57 h, about 10.7 to 11.0 h, within the 13 h cap. The queue
  checks the projection again before every run.
