# Changelog

Notable changes to this project. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the versions follow [Semantic Versioning](https://semver.org/).

## [1.0.0rc1] - 2026-11-17

The repository becomes a package.

### Added

- `echoview`: inference with the published ensemble `C2_ens_v1`. Commands `selfcheck`, `classify`, `segment`, `weights` and `mimic-table`, and a Python API.
- Weights on the Hugging Face Hub in safetensors, checked against their SHA-256 on every load.
- `echoview evaluate`: the EV9V development evaluation. It reproduces the research code's numbers exactly.
- `echoview-cache` and `echoview-train`: the frame cache and the three training stages, with their recipes in `configs/`.
- Definition files that name EV9V recordings by a hash of their folder name: the development benchmark, the list of recordings and the sampling plans.
- Parity tests against stored outputs of the research code: windows, decoder, metrics, scorecard and decoder rule.
- A Docker image, `ghcr.io/williamqiuzy/echo-view-routing`, with a CPU-only variant, built from the `Dockerfile` for x86-64 and ARM. `docs/DOCKER.md` shows how to run it.
- `requirements.txt` and `requirements-dev.txt`: the pinned versions the project was checked with. A test keeps them inside the ranges of `pyproject.toml`.
- A release workflow: a version tag publishes the wheel, the source archive and the Docker images.
- Fixes from a code review before the first push: a recipe file's values are checked like command-line values; a failed download always ends with advice instead of a traceback; `echoview evaluate` checks for SciPy and the weights before its long part; the frame cache file carries a checksum; the nearest-window table no longer grows with the square of the stream length; whether a system is the reference follows from the checksums alone.

- `echoview-train --checkpoint-every N --resume` for stage 3: a run on a machine that can be switched off continues from its last checkpoint and ends where an uncut run ends, up to the difference between two uncut runs.
- The decoder-setting search of `echoview evaluate` computes the decoder's scores once per half window instead of once per threshold. The scoring part is three times faster and every number is unchanged.

### Checked on a GPU (L40S, 2026-11-18 to 2026-11-20)

- Stage 3 against the research trainer, 100 updates from the same start: the package differs from the research trainer no more than two runs of the research trainer differ.
- Stages 1 and 2 against the published training logs of seed 0, 750 updates: the same pixels (hashes of 90,000 windows), the same accuracy up to one window in 3,000, losses equal in five to eight digits. Two runs of the loop differ by as much during warm-up.
- `echoview evaluate` on the command line: 0.9594 and 0.9837 for the ensemble, as published.
- A stage-3 run killed after update 60 of 100 and continued from its checkpoint (2026-11-20): it ends as far from the uncut run as two uncut runs end from each other.

### Fixed

- Stages 1 and 2 computed some learning rates with the last binary digit different from the published runs (the cosine's fraction was multiplied by pi in another order). A test now pins the rates stored in the published log.
- The training document said a rerun of stages 1 and 2 gives the published log number for number. It does not, and cannot: one GPU kernel of the backward pass has no deterministic form. The document now gives the measured differences.

### Not yet in this release

- The EV9V test benchmark and the MIMIC-IV-Echo evaluations.
- A full rerun of stage 3 with this package and its evaluation.
- Checkpoints for stages 1 and 2.
