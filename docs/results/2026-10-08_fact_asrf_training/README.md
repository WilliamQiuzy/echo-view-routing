# Paired FACT and ASRF training on frozen E1 features (2026-10-08)

**Status.** The configuration and code are fixed. Preparation, smoke checks, training and the single evaluation follow,
and their results will be added here. No training has started at this point.

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
