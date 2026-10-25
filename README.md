# echo-view-routing

**Live demo (GitHub Pages):** https://williamqiuzy.github.io/echo-view-routing/ — five published baselines on 27 real EV9V streams. Rebuild and publish with `demo/publish_pages.sh`.

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
| `tests/` | pytest; `make test` runs the CPU suite (markers `gpu`/`data` are server-only) |
| `docs/` | architecture, runbook, label-mapping notes, experiment ledger, pre-registered experiment plans (`docs/superpowers/plans/`), results |
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
| B4 | official MS-TCN (yabufarha/ms-tcn, pinned) on frozen features | run under the authors' recipe |
| B6 | official STFM (bgx666/stfm, pinned) via subprocess adapter | authors' recipe; multi-seed comparison to the paper in progress |
| P0–P2 | proposed TCN + semantic boundary head | not started |

Everything downstream of feature extraction runs on CPU from the feature cache.

## Current results (2026-09-17): five published baselines on the same EV9V test data

> **Pending recomputation (2026-09-30).** The numbers below were computed with EV9V frames read in name order (0.jpg, 1.jpg, 10.jpg, …), which shuffles frames within each clip. They will be replaced after the rerun from `cache_v2`. The core methods (B0–B4) are already recomputed in `docs/results/2026-09-30_frame_order_fix/`; with frames in order, MS-TCN no longer finds view changes clearly better than simple smoothing.

Full tables: `docs/results/2026-09-17_five_baselines/five_baselines.md`. Test split: 800 untouched clips (five-family task),
240 two-fragment streams and 120 multi-fragment streams; thresholds chosen on validation at a 5% contamination target and frozen.

| model | clip macro-F1 | frame acc | fragments/min | boundary F1@0.5 s | false split same/edit | missed diff/none | coverage@5% | achieved risk |
|---|---|---|---|---|---|---|---|---|
| STFM (official, sliding window) | 0.960 | 0.981 | 0.5 | 0.963 | 0.000 | 0.033 | 0.996 | 0.028 |
| EchoViewCLIP (official, sliding window) | 0.972 | 0.979 | 0.7 | 0.975 | 0.000 | 0.017 | 0.992 | 0.025 |
| EchoPrime (released weights, per frame) | 0.945 | 0.955 | 19.3 | 0.507 | 0.117 | 0.000 | 0.974 | 0.007 |
| MS-TCN (official) | 0.956 | 0.977 | 1.0 | 0.992 | 0.017 | 0.000 | 1.000 | 0.008 |
| ASFormer (official) | 0.959 | 0.980 | 0.8 | 0.980 | 0.017 | 0.000 | 1.000 | 0.007 |

Reproduction protocol and per-model verification: `docs/REPRODUCTION.md`; frozen checkpoint manifests with SHA-256 hashes: `docs/models_frozen/` (the weights themselves are available on request).

## Compute

One cloud NVIDIA L40S (46 GB) shared with another project; our processes cap GPU memory with
`ECHO_GPU_MEM_FRACTION`. The helper scripts that sync code and launch jobs on that server are not part of
the public repository.
