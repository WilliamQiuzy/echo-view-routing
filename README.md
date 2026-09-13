# echo-view-routing

Selective temporal routing of echocardiography video (Harvard capstone, public-data study).
Given a recording, output intervals with start/end, view label, accept/defer decision and confidence,
and distinguish **semantic view changes** (A4C → PLAX) from **nuisance discontinuities**
(same-view joins across patients, brightness edits). See `docs/ARCHITECTURE.md` for the design and
`docs/RUNBOOK.md` for the exact commands.

## Layout (short)

| Path | Purpose |
|---|---|
| `echo_routing/` | library: `ingest`, `audit`, `features`, `compose`, `temporal`, `evaluate` (+ `config`, `manifest`, `errors`) |
| `scripts/` | thin CLIs: build manifest, train encoder, extract features, run baselines, MS-TCN, STFM adapter |
| `configs/` | `base.yaml`, `labels.yaml` (frozen label ontology), `experiments/*.yaml` |
| `remote/` | Mac → Nebius helpers: `sync_code.sh`, `run.sh`, `status.sh`, `pull_results.sh`, `kill.sh`, `shell.sh` |
| `tests/` | pytest; `make test` runs the CPU suite (markers `gpu`/`data` are server-only) |
| `docs/` | architecture, runbook, label-mapping notes, experiment ledger |
| `demo/` | Mac-side demo (`demo_cli.py`) + a few sample videos under `demo/samples/` (gitignored) |
| `data/ runs/ checkpoints/ cache/ logs/` | server-only, gitignored |

## Data

EV9V (`bgx666/EV9V`, CC-BY-4.0): 5,138 cines, nine raw view codes, official patient-level splits
(3,683 / 567 / 888). Primary task is the coarse five-family routing task
(PLAX, PSAX, A4C, A5C, SC4C; PMPALA excluded until its code expansion is resolved), which leaves
3,364 / 521 / 800 cines. The raw code is preserved in every manifest row. Data lives only on the server.

## Baseline ladder (proposal §7.1)

| ID | Method | Status |
|---|---|---|
| B-file | native-file mean probability, whole-file accept/defer | implemented |
| B0 | ResNet-18 frame classifier, per-frame argmax | implemented |
| B1 | probability moving average + hysteresis | implemented |
| B2 | sticky HMM, Viterbi over posteriors | implemented |
| B3 | JS divergence between left/right windows + persistent label change | implemented |
| B4 | MS-TCN on frozen features | implemented (compact re-implementation) |
| B6 | STFM official code (EV9V authors) via subprocess adapter | adapter implemented; vendored at pinned commit |
| P0–P2 | proposed TCN + semantic boundary head | not started |

Everything downstream of feature extraction runs on CPU from the feature cache.

## Quickstart

```bash
cp .env.example .env            # fill in server alias / paths
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -e ".[dev]"
make test                        # CPU unit tests
remote/sync_code.sh              # push code to the server
remote/run.sh manifest scripts/build_manifest.py
remote/run.sh train_b0 scripts/train_frame_encoder.py -c configs/experiments/b0_frame_encoder.yaml
remote/status.sh train_b0
```

## Compute

One Nebius L40S (46 GB) shared with an unrelated project. Our processes cap GPU memory with
`ECHO_GPU_MEM_FRACTION` and live entirely under `/home/william/echo-view-routing`.
