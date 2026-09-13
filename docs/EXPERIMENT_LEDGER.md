# Experiment ledger (append-only)

| Date | run_id | Script / config | Seed | Result summary | Notes |
|---|---|---|---|---|---|
| 2026-09-13 | 20260913-033227-b0_frame_encoder_smoke-bb4839da-s0 | `train_frame_encoder.py -c experiments/b0_frame_encoder_smoke.yaml` | 0 | 2 epochs × 400 cines; val cine macro-F1 0.812 (ep0), 0.755 (ep1); ~14 s/epoch | pipeline validation only; ckpt_hash b291f81334f7 used for the smoke ladder |
| 2026-09-13 | 20260913-033411-b0_frame_encoder-b072725b-s0 | `train_frame_encoder.py -c experiments/b0_frame_encoder.yaml` | 0 | full run, 30 ep max, patience 6; ep0 val cine macro-F1 0.977, 69 s/epoch | shared GPU with co-tenant (~4–6 GB) |
| 2026-09-13 | 20260913-034701-baselines-6a612902-s0 | `run_baselines.py` on smoke ckpt b291f81334f7 | 0 | pipeline validation only; exposed cell-imbalance and threshold-selection bugs (fixed in a03f57c..); numbers not reportable | duplicate concurrent run 034659 from a stale process, ignore |
| 2026-09-13 | 20260913-035909-baselines-6a612902-s0 | `run_baselines.py -c experiments/baselines.yaml --ckpt-hash 79f4a41af6e3` | 0 | test: B0 cine macro-F1 0.957 / 23.1 frag/min / boundary F1@0.5s 0.50; B1 0.949 / 2.4 / 0.918; B2 0.946 / 2.4 / 0.906; B3 0.946 / 3.6 / 0.913; B-file cine acc 0.975, 5%-policy coverage 0.998 risk 0.019 | 240 balanced test streams (60/cell); first reportable ladder |
