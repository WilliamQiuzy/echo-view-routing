# Changelog

Notable changes to this project. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Removed

- The Docker images of versions 1.0.0 and 1.1.0 were withdrawn from the registry: they held Pillow 12.3.0. The releases themselves and their wheels stay.

### Checked on a GPU (L40S, 2026-11-20)

- A full rerun of stage 3: the three seeds trained again with this package come out within 0.004 of the published seeds on both development halves, and their ensemble, decoded with the published setting, within 0.003. The rule's own choice of the decoder setting did not repeat for the retrained ensemble; `docs/TRAINING.md` gives the numbers and what follows from them.
- The test that asserts the published development numbers on the real dataset and the published weights passes.
- Stages 1 and 2, two runs each of 750 updates with the final code: the training accuracy equals the published log's in both periods, and the mean loss differs from it by at most 2.7e-5, which is of the size of the gap between the two runs.

## [1.1.1] - 2026-11-20

### Security

- Pillow 13.0.0 instead of 12.3.0 in the pinned requirements and in the Docker images. Pillow before 13.0 has published vulnerabilities in several decoders. On Python 3.11 and newer the package now requires Pillow 13.0 or newer. `SECURITY.md` says what this means on Python 3.10, for which Pillow 13 does not exist.
- The frames are the same: under 12.3.0 and 13.0.0 the package decodes 374 JPEG frames, with and without a gamma, to identical pixels.

## [1.1.0] - 2026-11-20

### Added

- `echoview evaluate --benchmark test`: the project's frozen EV9V test benchmark, decoded once with the system's fixed setting, with 95% bootstrap intervals. Its definition file, `ev9v_test.json.gz`, names recordings by key, as the development file does.
- `echoview evaluate --save-logits FILE` keeps the window logits of a run, so that its numbers can be looked into without running the models again.

### Checked on a GPU (L40S, 2026-11-20)

- The test evaluation with the published weights gives the row the research code stored for this system: 0.9089 on multi streams, 98.1% and 0.956 on recordings. Every number and both ends of every interval are equal to the last digit.
- The frames this package decodes for the six sampling plans of stages 1 and 2 give the pixel hashes the research trainer stored, for all 180,000 windows.

### Not yet in this release

- The MIMIC-IV-Echo evaluations.
- A full rerun of stage 3 with this package and its evaluation.
- Checkpoints for stages 1 and 2.

## [1.0.0] - 2026-11-20

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
