# ARCHITECTURE — Selective Temporal Routing of Echocardiography Video

Status: design frozen for implementation. Version 1.0 · 2026-09-12 · Python package `echo_routing`.
Scope: builds the baseline ladder (B-file, B0–B4, B6) and the scaffolding required for P0–P2 on EV9V, public data only.

**Non-goals for v1.0:** DICOM/MIMIC ingestion (interfaces reserved, unimplemented), natural-transition arm, TTE47/TMED-2 external arm, B5 (ASFormer), conformal risk control. Each has a reserved module path so later work does not reshape the tree.

## 0. Implementation status (v0.1 scaffold, 2026-09-13)

The v0.1 scaffold implements the subset needed for the first baseline demo. Deviations from the v1.0 design below are deliberate simplifications, each reversible without moving files:

| Area | v1.0 design | v0.1 as built | Why |
|---|---|---|---|
| Manifest format | parquet | CSV (`data/manifests/ev9v_native.csv`) | inspectable, diff-able; parquet writer is a one-line switch |
| Label map | `configs/label_map_v1.csv` + `class_index_v1.yaml` | `configs/labels.yaml` (mapping + both conflicting expansions + class order) | one file, same content |
| Class index (family5) | A4C 0, A5C 1, PLAX 2, PSAX 3, SC4C 4 | **PLAX 0, PSAX 1, A4C 2, A5C 3, SC4C 4** (proposal §4.2 listing order) | frozen in `configs/labels.yaml`; raw9 order follows STFM's `INDEXOFLABEL` |
| Perturbations | gamma090, gamma110, reenc_crf28, resample_15hz | gamma090, gamma110 (feature-cache variants) | re-encode/resample deferred |
| Recipe bank | 4–8 fragments, 10–45 s, five banks | two-fragment pair recipes, four cells, one bank per split | enough for the four-cell figure; multi-fragment builder is the next step |
| Correspondence audit / phash | in manifest | not yet | scheduled after the demo |
| Tests | `tests/unit|integration|server` | flat `tests/` with `gpu`/`data` markers | will be split once server tests exist |
| Remote scripts | `common.sh`, `kill.sh`, `bootstrap.sh` | `common.sh`, `kill.sh`, `sync_code.sh`, `run.sh`, `status.sh`, `pull_results.sh`, `shell.sh` | `bootstrap.sh` documents the manual server setup |

Implemented modules: `config`, `errors`, `manifest`, `audit/label_map`, `ingest/frames`, `features/{sampling,dataset,encoder,train_encoder,extract,cache}`, `compose/{recipe_builder,stream_assembler}`, `temporal/{segments,decode_select}`, `temporal/baselines/{b0,b_file,b1,b2,b3}`, `evaluate/metrics_*`. See `docs/RUNBOOK.md` for the executable sequence.

---

## 1. Repository layout

Local: `/Users/williamqiu/Desktop/Harvard/Capstone/echo-view-routing` · Server: `/home/william/echo-view-routing` (identical relative layout; `data/ runs/ checkpoints/ logs/ .venv/` populated only on the server).

```
echo-view-routing/
├── README.md                         Project overview, quickstart, provenance statement
├── pyproject.toml                    Package metadata, deps, pytest/ruff config (Python 3.12 target)
├── .env.example                      All ECHO_* variables with placeholder values (COMMITTED)
├── .env                              Real values, machine-local (GITIGNORED)
├── .gitignore                        Ignores data/ runs/ checkpoints/ logs/ cache/ .venv/ .env demo/samples/*
├── Makefile                          Local shortcuts: make test, make sync, make status
│
├── configs/
│   ├── base.yaml                     Single source of truth: paths, sampling, encoder, temporal, policy, seeds
│   ├── labels.yaml                   Frozen 9-code -> 5-family mapping + both conflicting expansions + class order
│   └── experiments/
│       ├── b0_frame_encoder.yaml     Encoder training run (proposal §9.2: AdamW 3e-4, wd 1e-4, bs 64, 30 ep)
│       ├── b0_frame_encoder_smoke.yaml  Short smoke run for pipeline validation
│       ├── baselines.yaml            B1/B2/B3 decoding parameters + policy targets
│       ├── b4_mstcn.yaml             MS-TCN on frozen features
│       └── b6_stfm.yaml              STFM adapter: vendored config + data-layout mapping
│
├── echo_routing/                     See section 3
├── scripts/                          Thin argparse CLIs; no logic
│   ├── build_manifest.py  train_frame_encoder.py  extract_features.py  gpu_probe.py
│   ├── run_baselines.py  train_mstcn.py  prepare_stfm_data.py  run_stfm.py  report.py
│   └── make_demo_bundle.py
│
├── remote/
│   ├── common.sh                     Loads .env, validates required vars, defines ssh wrapper
│   ├── bootstrap.sh                  One-time server setup (root, venv, deps, third_party clone)
│   ├── sync_code.sh                  rsync Mac -> server (code only; server data protected)
│   ├── run.sh                        nohup a remote job; writes log + pid
│   ├── status.sh                     Our jobs, GPU, disk, tail of a log
│   ├── pull_results.sh               rsync server -> Mac (metrics/reports/small artifacts only)
│   ├── kill.sh                       Terminate one of OUR jobs by name (refuses foreign pids)
│   └── shell.sh                      Interactive shell in the project venv
│
├── third_party/
│   ├── README.md                     Vendoring policy: clone-only, never edited, pinned commit recorded
│   └── stfm/                         clone of github.com/bgx666/stfm (pinned; server-side, gitignored)
│
├── tests/                            pytest; default selection excludes `gpu` and `data` markers
├── docs/                             ARCHITECTURE.md, RUNBOOK.md, LABEL_MAPPING.md, EXPERIMENT_LEDGER.md
├── demo/
│   ├── demo_cli.py                   Mac-only: renders ladder table + four-cell figure from pulled artifacts
│   ├── samples/                      3–5 EV9V MP4s (gitignored)
│   └── artifacts/                    Pulled metrics/tables used by the demo (gitignored)
│
├── data/                             SERVER ONLY, gitignored
│   ├── ev9v/raw/                     *_labeled.txt, Videos.tar, Images.tar, dataset card
│   ├── ev9v/Videos/{video_id}.mp4    Extracted MP4s (canonical timing source)
│   ├── ev9v/Images/{video_id}/frame_XXXXXX.jpg   Extracted JPEGs (canonical pixel source)
│   ├── stfm/EchoData/                Symlink farm matching STFM's expected layout
│   └── manifests/                    ev9v_native.csv, recipes/*.json
├── cache/features/{ckpt_hash}/{variant}/{video_id}.npz   feat + prob + logit per cine
├── runs/{run_id}/                    config.resolved.yaml, metrics/, intervals/, reports/
├── checkpoints/{run_id}/             last.pt, best.pt
└── logs/{job_name}.{log,pid}         nohup outputs from remote/run.sh
```

---

## 2. Key architectural decisions

**ADR-0001 — Constructed streams are composed in feature/logit space, not pixel space.** Frame features come from a frame-wise frozen encoder, so concatenating cached per-frame feature rows is bit-identical to encoding a rendered concatenation. Nuisance edits are applied at the native-cine level and cached once per `variant`; composition is pure metadata.
- Pros: recipe banks cost seconds; thousands of recipes at zero decode cost; exactly reproducible from a seed; no rendered corpus.
- Cons: invalid for models with pixel-level temporal operators (STFM). Rendering exists only for B6 and the demo.

**ADR-0002 — Two pixel sources with an explicit correspondence audit.** MP4 (`Videos.tar`) is authoritative for dimensions, decoded frame count, timing and duration. JPEG (`Images.tar`) is the default pixel source for training/feature extraction and is required by STFM. Because the release removed "invalid extracted frames", the manifest will record `n_frames_video`, `n_frames_images` and a `correspondence_status`; mismatched cines are flagged and excluded from the primary recipe bank until reviewed.

**ADR-0003 — Everything downstream of feature extraction is CPU-only and cache-driven.** B-file, B0, B1, B2, B3, policy search, all metrics and all reports consume `cache/features/`. Only encoder training, feature extraction, MS-TCN/TCN training and STFM touch the GPU. The demo is reproducible on the Mac from a small pulled cache.

---

## 3. Module decomposition (`echo_routing/`)

Conventions: every public function validates inputs at the boundary and raises a typed error from `echo_routing/errors.py` (`ConfigError`, `LabelError`, `ManifestError`, `FrameError`, `CacheMissError`, `RecipeError`). Dataclasses are `frozen=True`; updates use `dataclasses.replace`. No module writes outside paths derived from `Paths`.

### 3.0 Core — `config.py`, `errors.py`, `manifest.py`
```python
Paths(repo_root, data_root, ev9v_root, runs_root, checkpoints_root, cache_root, logs_root)
load_paths(repo_root=None, env=None) -> Paths           # ECHO_* overrides, repo-relative defaults
load_config(*yaml_paths, overrides=None) -> dict          # deep merge; `extends:` supported
gpu_memory_fraction(default=0.6) -> float                 # ECHO_GPU_MEM_FRACTION, validated
build_manifest(raw_dir, ontology, images_root, videos_root) -> DataFrame
save_manifest / load_manifest / task_subset(df, task) / summarize(df)
```
Must NOT: read the filesystem beyond given paths; decode video; infer patient identity from filenames.

### 3.1 `ingest/` — decoding and pixel access
`frames.py`: `list_frame_paths`, `letterbox`, `apply_gamma`, `load_frame`, `VideoRecord`. Planned: `probe.py` (PyAV probe of MP4 timing), `stfm_layout.py`, `dicom.py` (stub).
Must NOT: apply label mapping, write to `cache/`.

### 3.2 `audit/` — label and duplicate audit
`label_map.py`: `LabelOntology`, `load_ontology`. Planned: `split_audit.py`, `duplicates.py` (phash).
Must NOT: silently drop rows, "fix" a conflicting expansion, or delete PMPALA from the manifest (it is retained with `family5_label=None`).

### 3.3 `features/` — frame sampling and feature extraction
`sampling.py` (`spaced_indices`, `stride_indices`, `balanced_video_weights`), `dataset.py` (`EpochFrameDataset`, `FixedFrameDataset`), `encoder.py` (`FrameEncoder`, checkpoint I/O, `checkpoint_hash`), `train_encoder.py` (`TrainConfig`, `train`), `extract.py` (`extract_split`), `cache.py` (`VideoFeatures`, npz I/O, index).
Class-balanced sampling operates over cines, not frames. `train` writes `last.pt` after every epoch and resumes from it.
Must NOT: build recipes; write features without the checkpoint hash in the path.

### 3.4 `compose/` — composition and perturbation
`recipe_builder.py` (`Pool`, `Fragment`, `Recipe`, `make_pair_recipes`, dict round-trip), `stream_assembler.py` (`Stream`, `render`).
Invariants (unit-tested): all fragments of a recipe come from one official split; the test bank uses each native cine at most once; the four cells are equinumerous; a same-view join is recorded as a physical join but not a semantic boundary.
Must NOT: stretch or loop a short cine; expose join positions to any predictor.

### 3.5 `temporal/` — temporal prediction and selection
`segments.py` (`Segment`, `labels_to_segments`, `boundaries_from_labels`, `non_max_suppression`), `decode_select.py` (`segment_score`, `route`, `select_threshold`, `RoutedInterval`), `baselines/` one file per rung, `baselines/registry.py` (`decode(method, prob, cfg) -> labels`), `models/mstcn.py`.
Must NOT: read ground-truth labels or join positions at inference; reset context at joins; tune any threshold on test streams; modify `third_party/`.

### 3.6 `evaluate/` — evaluation and native-file export
`metrics_classification.py`, `metrics_stability.py`, `metrics_boundary.py`, `metrics_segmentation.py`, `metrics_routing.py`, `ladder.py` (runs one method over native + constructed settings), `report.py` (tables). Planned: `export.py` (`export_native_intervals` splits any interval spanning a file join).
Primary metrics: `known_view_routing_coverage` and `known_view_routing_risk`. Risk is `None` (never 0.0) when nothing is accepted.
Must NOT: choose thresholds, run models, write into `cache/`.

---

## 4. Data contracts

### 4.1 Native-recording manifest — `data/manifests/ev9v_native.csv`
One row per EV9V video (5,138). Columns: `dataset, dataset_revision, split, video_id, raw_label, mapping_version, family5_label (nullable), family5_index (nullable), raw9_index, n_frames, fps_playback, duration_s, frames_dir, video_path`. Planned additions: `n_frames_video, correspondence_status, phash_*`. Validation: unique `video_id`; no cross-split id; family5 counts must equal **3,364 / 521 / 800**; `patient_id` is never inferred.

### 4.2 Label mapping — `configs/labels.yaml`
`mapping_version: family5_v1`; PLHLA→PLAX (provisional); PASA, PMASA, PMVLSA, PPMLSA→PSAX (broad family, level unresolved); A4C, A5C, SC4C identity; PMPALA→null (conflicting expansion). Both published expansions are stored verbatim.

### 4.3 Recipe bank — `data/manifests/recipes/{split}_{bank}.json`
List of `Recipe{recipe_id, cell, fragments[{video_id, label, variant, start, end}]}`; `start/end` index the cached sample arrays. Semantic boundaries are derived at assembly time and consumed only by `evaluate/`.

### 4.4 Feature cache — `cache/features/{ckpt_hash}/{variant}/{video_id}.npz`
Arrays `frame_idx int32[N]`, `t_sec float32[N]`, `feat float16[N,512]`, `prob float32[N,C]`, `logit float32[N,C]`; `index.csv` (video_id, split, raw_label, label_index, n_samples, path) and `meta.json` per variant. ≈160 MB per variant for the whole dataset at 10 Hz.

### 4.5 Model output intervals — `runs/{run_id}/intervals/{setting}.csv`
`stream_id, start_sample, end_sample, start_s, end_s, view, score, decision, method_id, policy_tau`. Native exports never span two source files.

### 4.6 Metrics JSON — `runs/{run_id}/metrics/{method}.json`
`{"method_id", "task", "ckpt_hash", "policy": {"tau", "target_risk", "selected_on": "validation"}, "native": {classification, stability}, "constructed": {"cells": {...}, "boundary": {...}, "routing": {"coverage", "risk"}}}`.

---

## 5. Baseline execution pipeline

| # | Command | Device | Consumes | Produces |
|---|---|---|---|---|
| 1 | download + extract (see `remote/bootstrap.sh`) | net/disk | HF `bgx666/EV9V` | `data/ev9v/{raw,Videos,Images}` |
| 2 | `scripts/build_manifest.py` | CPU | step 1 | `data/manifests/ev9v_native.csv` |
| 3 | `scripts/train_frame_encoder.py -c configs/experiments/b0_frame_encoder.yaml` | **GPU** | step 2 | `checkpoints/{run}/{last,best}.pt` |
| 4 | `scripts/extract_features.py --ckpt ... --variants orig,gamma090,gamma110` | **GPU** | step 3 | `cache/features/{hash}/{variant}/` |
| 5 | `scripts/run_baselines.py --ckpt-hash ... --methods b_file,b0,b1,b2,b3` | CPU | step 4 | `runs/{run}/metrics/*.json`, `reports/ladder.md` |
| 6 | `scripts/train_mstcn.py` then `run_baselines.py --methods b4` | **GPU** | step 4 | B4 checkpoint + metrics |
| 7 | `scripts/prepare_stfm_data.py` + `scripts/run_stfm.py` | CPU/**GPU** | step 1 | STFM run dir (B6) |
| 8 | `scripts/report.py --runs runs/*` | CPU | steps 5–7 | `runs/_reports/baseline_ladder.md` |

Every step is idempotent: extraction skips existing files, training resumes from `last.pt`, feature extraction skips cached cines, baselines are pure functions of the cache.

---

## 6. Remote workflow

`remote/common.sh` sources `.env`, asserts `ECHO_SSH_ALIAS`, `ECHO_REMOTE_ROOT`, `ECHO_REMOTE_PYTHON`, refuses a root that overlaps `ECHO_CO_TENANT_ROOT`, and defines `rssh`.

| Script | Behaviour |
|---|---|
| `sync_code.sh` | `rsync -az --delete` with excludes for `data/ runs/ checkpoints/ cache/ logs/ .venv/ .git/ .env third_party/stfm/ demo/samples/`; excluded paths are protected from deletion on the server. |
| `run.sh <job> <script> [args]` | `nohup $ECHO_REMOTE_PYTHON -u <script> … > logs/<job>.log & echo $! > logs/<job>.pid`; refuses to start if the job is alive. Exports `ECHO_GPU_MEM_FRACTION`. |
| `status.sh [job]` | GPU summary incl. co-tenant processes, liveness of our pids, disk usage, tail of a log. |
| `pull_results.sh` | rsync `runs/` → local, restricted to json/csv/md/yaml/png/svg/txt. |
| `kill.sh <job>` | Kills only the recorded pid and only if its command line lives under `ECHO_REMOTE_ROOT`. |

Preemptible-instance rules: checkpoint after every epoch (write-to-temp + replace planned); every script resumable; downloads use `curl -C -`; per-video atomic feature writes.

---

## 7. Config strategy

`configs/base.yaml` holds every tunable with a default; experiment YAMLs use `extends: ../base.yaml` and override a few keys; CLI `--set key=value` overrides last. The merged config is written to `runs/{run_id}/config.resolved.yaml`. `.env` is loaded once (`ECHO_*`), real environment variables win. One seed per run; recipe banks take their own seed. Every GPU entry point calls `torch.cuda.set_per_process_memory_fraction(ECHO_GPU_MEM_FRACTION)` (default 0.6 ≈ 27 GB of 46 GB) so the co-tenant job keeps headroom.

---

## 8. Testing strategy

Target 80 % line coverage on `echo_routing/` excluding training loops and GPU extraction. Markers: `gpu`, `data`. Default selection excludes both, so `make test` is the full Mac-runnable suite.

| Tier | Examples |
|---|---|
| Pure unit | sampling spacing; HMM Viterbi vs brute force; JSD symmetry/bounds; hysteresis; NMS; boundary/segment F1 vs hand-computed matches; coverage/risk edge cases (zero accepted ⇒ risk `None`) |
| Text fixtures | split-file parsing rejects 1- and 3-token lines and cross-split duplicates; label map validation |
| Synthetic arrays | recipe invariants (cells, no reuse, semantic vs physical joins); assembly equals concatenation; every decoder returns a label per sample |
| Server only (`gpu`/`data`) | real manifest reproduces 3,683/567/888 and 3,364/521/800; one encoder step decreases loss under the VRAM cap; STFM adapter parses real outputs |

TDD order: metrics → manifest → recipes → decode/select → baselines → training loops (smoke-tested only).

---

## 9. Isolation checklist vs. `qwen-agent-training`

- [x] All our paths under `/home/william/echo-view-routing`; `remote/common.sh` refuses an overlapping root.
- [x] Separate venv `/home/william/echo-view-routing/.venv` (Python 3.12); never install into the other venv.
- [x] rsync `--delete` scoped to our root with server-only dirs excluded (protected).
- [x] Our logs/pids in `$ECHO_REMOTE_ROOT/logs/`; `kill.sh` refuses pids whose command line is outside our root.
- [x] `status.sh` shows the co-tenant's VRAM; `ECHO_GPU_MEM_FRACTION` caps ours. No `nvidia-smi --gpu-reset`, no MPS/MIG changes.
- [x] No writes to `~/.bashrc`, `~/.profile`; no changes to the other project's files.
- [ ] `HF_HOME` under our data root (set when `huggingface_hub` downloads are used).
- [x] System packages installed via apt (ffmpeg, tmux) are additive and do not affect the running co-tenant process.
- [x] No ports; TensorBoard, if used, binds 127.0.0.1 over an SSH tunnel.

---

## 10. Demo plan

Shown from `demo/demo_cli.py` reading only `demo/artifacts/` pulled by `remote/pull_results.sh` — no live SSH during the meeting.
1. Data and ontology reality check: 3,683/567/888 raw, nine codes, the card-vs-README expansion conflict, 3,364/521/800 primary counts, PMPALA retained but excluded.
2. Baseline ladder table: B-file (separate operational table), B0, B1, B2, B3 (+B4/B6 when available): native macro-F1 / balanced accuracy, fragments per minute (proxy), boundary F1 @ 0.25/0.5/1.0 s, false splits per same-view join by edit condition, coverage/risk at the frozen validation-selected 5 % policy.
3. The four-cell figure: false semantic splits on same-view joins vs recall of different-view boundaries, split by edit condition.
4. Risk–coverage curves with the frozen operating point marked and achieved-vs-target contamination stated.
5. One qualitative timeline on a demo sample.
Fallbacks: (a) smoke encoder if the full run is unfinished, labelled as such; (b) ImageNet-initialised untrained head — pipeline validation only; (c) synthetic corpus with the architecture; (d) B6 shown as vendored commit + adapter plan. No results cell is ever filled with an expected number.

---

## 11. Open decisions and risks

| # | Open question | Recommended default |
|---|---|---|
| 1 | EV9V cines are short (median 116 frames ≈ 3.9 s); 15–45 s streams need many fragments. | Pair recipes now; 4–8 fragment recipes next; never stretch or loop; report realized durations. |
| 2 | PLHLA→PLAX and the four-code PSAX family are provisional. | Ship `family5_v1`, keep raw codes everywhere, nine-code table as sensitivity endpoint; ask EV9V authors. |
| 3 | Images vs Videos as pixel source; invalid frames removed. | `pixel_source=images`; MP4 authoritative for timing; correspondence audit before the primary bank is frozen. |
| 4 | No patient identifiers. | Cine-level bootstrap only; caveat in every metrics JSON. |
| 5 | Near-duplicate pixels across splits. | phash audit before model evaluation; report, do not silently drop. |
| 6 | Preemption during training. | Per-epoch checkpoints + resume; smoke config always available. |
| 7 | Shared GPU OOM. | Memory fraction cap; fail fast if free VRAM is insufficient. |
| 8 | STFM may not match released behaviour. | Vendor unmodified at a pinned commit; label divergences as adaptation. |
| 9 | B3 may already match B4/P-series. | Report it as a finding; do not add capacity to beat it. |
| 10 | Contamination target unreachable. | `select_threshold` returns `None`; JSON records target unmet; never report zero coverage as perfect purity. |
| 11 | Feature-space composition validity. | Frame-wise encoders only; STFM is evaluated on rendered pixels. |
| 12 | Mac (Python 3.14, no GPU). | Mac runs tests, demo, `remote/*.sh`; GPU-only imports are guarded. |
