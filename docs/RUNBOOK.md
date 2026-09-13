# RUNBOOK — from raw EV9V to the baseline ladder

All server commands run from `/home/william/echo-view-routing` via `remote/run.sh <job> <script> [args]`
(nohup + log + pid) or interactively via `remote/shell.sh`.

| Step | Command (Mac side) | Device | Output |
|---|---|---|---|
| 0 | `remote/sync_code.sh` | — | code on server |
| 1 | one-time: `remote/bootstrap.sh` (venv, deps, STFM clone, EV9V download + extract) | net/disk | `data/ev9v/{raw,Videos,Images}` |
| 2 | `remote/run.sh manifest scripts/build_manifest.py` | CPU | `data/manifests/ev9v_native.csv` |
| 3 | `remote/run.sh train_b0 scripts/train_frame_encoder.py -c configs/experiments/b0_frame_encoder.yaml` | GPU | `checkpoints/<run>/{last,best}.pt` |
| 3s | smoke: `... -c configs/experiments/b0_frame_encoder_smoke.yaml` | GPU | 2 epochs, 400 cines/epoch |
| 4 | `remote/run.sh feats scripts/extract_features.py --ckpt checkpoints/<run>/best.pt --splits train,validation,test` | GPU | `cache/features/<hash>/{orig,gamma090,gamma110}/` |
| 5 | `remote/run.sh ladder scripts/run_baselines.py -c configs/experiments/baselines.yaml --ckpt-hash <hash>` | CPU | `runs/<run>/metrics/*.json`, `reports/ladder.md` |
| 6 | `remote/run.sh b4 scripts/train_mstcn.py -c configs/experiments/b4_mstcn.yaml --ckpt-hash <hash>` | GPU | B4 checkpoint |
| 7 | `remote/run.sh stfm_data scripts/prepare_stfm_data.py -c configs/experiments/b6_stfm.yaml` then `remote/run.sh stfm scripts/run_stfm.py -c configs/experiments/b6_stfm.yaml` | CPU / GPU | STFM run (B6) |
| 8 | `remote/pull_results.sh` | — | `runs/` metrics + reports on the Mac |

Resume a preempted encoder run: add `--resume checkpoints/<run>/last.pt --run-dir runs/<run>`.
Check progress: `remote/status.sh <job>`. Stop one of our jobs: `remote/kill.sh <job>`.
