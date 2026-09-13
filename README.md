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
| `docs/` | architecture, runbook, label-mapping notes, experiment ledger; `docs/proposal/` holds the research proposal (EN) and discussion brief (ZH) |
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
| B6 | STFM official code (EV9V authors) via subprocess adapter | run (adaptation: 15-epoch cap) |
| P0–P2 | proposed TCN + semantic boundary head | not started |

Everything downstream of feature extraction runs on CPU from the feature cache.

## Current results (2026-09-13, encoder `79f4a41af6e3`, family5 task)

Test split: 800 native cines and 240 balanced constructed two-fragment streams; thresholds selected on the
validation split at a 5% contamination target and frozen. Full table and figure: `docs/results/2026-09-13_ladder/`.

| method | cine macro-F1 | frame acc | fragments/min | boundary F1@0.5s | false split same/none · same/edit | missed diff/none · diff/edit | coverage@5% | achieved risk |
|---|---|---|---|---|---|---|---|---|
| B0 argmax | 0.957 | 0.951 | 23.1 | 0.499 | 0.033 · 0.133 | 0.000 · 0.000 | 0.967 | 0.007 |
| B1 smoothing | 0.949 | 0.968 | 2.4 | 0.918 | 0.000 · 0.017 | 0.017 · 0.017 | 0.999 | 0.014 |
| B2 HMM | 0.946 | 0.967 | 2.4 | 0.906 | 0.000 · 0.000 | 0.000 · 0.000 | 0.996 | 0.009 |
| B3 JS divergence | 0.946 | 0.963 | 3.6 | 0.913 | 0.000 · 0.017 | 0.000 · 0.000 | 0.989 | 0.010 |
| B4 MS-TCN | 0.568 | 0.915 | 0.7 | 0.779 | 0.000 · 0.017 | 0.067 · 0.117 | 0.020 | 0.000 |

B-file (per-file mean probability): cine accuracy 0.975; at the 5% target coverage 0.998 with achieved risk 0.019.
**B4 and B6 are pipeline checks, not valid baselines yet.** B4 is our compact MS-TCN re-implementation trained on an unbalanced
600-stream bank; B6 is the official STFM code cut to 15 of the authors' 100 epochs (test acc 0.939, macro-F1 0.904 on its nine-code task).
Both must be rerun under the authors' protocols (official code, full recipe, matched against published numbers) before being cited. See `docs/EXPERIMENT_LEDGER.md`.

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
