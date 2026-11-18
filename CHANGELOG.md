# Changelog

Notable changes to this project. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the versions follow [Semantic Versioning](https://semver.org/).

## [1.0.0rc1] - 2026-11-17

The repository becomes a package. The research log that was here before is kept on the branch `research-archive`.

### Added

- `echoview`: inference with the published ensemble `C2_ens_v1`. Commands `selfcheck`, `classify`, `segment`, `weights` and `mimic-table`, and a Python API.
- Weights on the Hugging Face Hub in safetensors, checked against their SHA-256 on every load.
- `echoview evaluate`: the EV9V development evaluation. It reproduces the research code's numbers exactly.
- `echoview-cache` and `echoview-train`: the frame cache and the three training stages, with their recipes in `configs/`.
- Definition files that name EV9V recordings by a hash of their folder name: the development benchmark, the list of recordings and the sampling plans.
- Parity tests against stored outputs of the research code: windows, decoder, metrics, scorecard and decoder rule.
- Fixes from an independent review before the first push: a recipe file's values are checked like command-line values; a failed download always ends with advice instead of a traceback; `echoview evaluate` checks for SciPy and the weights before its long part; the frame cache file carries a checksum; the nearest-window table no longer grows with the square of the stream length; whether a system is the reference follows from the checksums alone.

### Not yet in this release

- The EV9V test benchmark and the MIMIC-IV-Echo evaluations.
- The GPU runs that compare this package's training with the published training.
