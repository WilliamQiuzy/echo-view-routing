# Changelog

Notable changes to this project. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Every release from 1.2.1 on is archived on Zenodo. The README and `CITATION.cff` carry the DOI that stands for all versions, 10.5281/zenodo.24485965.
- `python -m echoview.evaluation.comparison` carries out the plan of `docs/COMPARISON.md`: `inventory` lists stored outputs, `settings` chooses decoder settings on the calibration half, and `score` builds every row and scores it with intervals by stream and by examination. Before the test benchmark is scored, the stored outputs must give every published number.
- `docs/COMPARISON.md`: the plan of a fuller comparison with the four other approaches on the test benchmark, written before anything in it is computed. It fixes the numbers for every system, the rows that give the other systems the project's decoder, the intervals and the wording of the result. Stage A of the plan (what is stored, the decoder settings chosen on the calibration half, and the selection half as a trial) is in its section 9 and in `docs/comparison/`.

- `docs/COMPARISON.md`, section 11: an addendum that answers the two comparisons the first read left open. STFM and ASFormer are run on the development benchmark, get a decoder setting by the same rule, and the test benchmark is read a second time for these two rows only. `score --earlier` compares a later output with an earlier one, number by number, and writes nothing when one differs; a read of the test benchmark now needs the first read's output.
- `docs/FOLLOWUP.md` and `python -m echoview.evaluation.followup`: three checks on development data of changes that need no training (a denser read-out, an offset on the A5C logit, EchoViewCLIP's probabilities mixed in), with the rules for calling one a candidate fixed before the numbers.
- `Ensemble.window_logits`, `probabilities_from_logits` and `window_plan` take an optional `stride`, the number of samples between two outputs. The default stays 5 and the published system is unchanged.

### Read once (2026-11-24)

- The frozen EV9V test benchmark, for the comparison with the four other approaches as planned in `docs/COMPARISON.md`: stored outputs only, every system as published and, where a setting could be chosen on development data, with the project's decoder. EchoRouter has the highest segment F1 on multi streams as published (0.909). With the same decoder, EchoViewCLIP scores 0.884 and is not shown to differ from EchoRouter; EchoRouter is ahead of EchoPrime's classifier; for STFM and ASFormer the question could not be answered. EchoViewCLIP stays the best row on single recordings. The README's comparison section now shows this table.

### Fixed

- The comparison table of the README gave EchoPrime's macro-F1 on single recordings as 0.944. It is 0.943 (0.9435).

### Changed

- The README and `docs/DOCKER.md` say how long the self-check takes in a container without a GPU on an Apple-silicon laptop: about twenty minutes when it was measured with the 1.2.1 image.

## [1.2.1] - 2026-11-23

### Added

- `echoview --version` prints the version of the package and the name of the published system.
- `python -m echoview.evaluation.uncertain_study confirm` scores the released uncertain rule on the test benchmark and chooses nothing. It reads stored window logits, stops unless they give the benchmark's four published numbers, and never replaces its output. `docs/UNCERTAIN.md`, section 9, is the plan of that read, committed before it, and its result.

### Changed

- `CITATION.cff` carries the name EchoRouter in its title and lists Helena Li by that name.
- The quick start of the README says which Python the package needs.

### Read once (2026-11-20)

- The frozen EV9V test benchmark with the released uncertain rule, on the stored model outputs of the published ensemble: 94.5% of the samples keep a view and 2.0% of them are wrong, against 3.6% when every sample is named. The rule's tests for whole pieces are confirmed there. The gain is smaller than on the development half, and in the multi streams a plain confidence threshold leaves fewer wrong samples. The README and the model card give these numbers beside the development ones.

## [1.2.0] - 2026-11-20

### Added

- **The uncertain output.** `echoview segment --uncertain` leaves the weak parts of a stream unnamed: the 0.3 s before and after every cut, and a part whose view hangs on a weak cut or on a narrow margin between two views. Such parts come back as pieces called "uncertain". Every other sample keeps the view it has without the option, and the default output is unchanged. In Python: `system.segment(frames, echoview.uncertain.released())`. The rule was chosen and checked for the ensemble of three seeds, and the command says so when it is used with another model.
- **How the rule was found.** `docs/UNCERTAIN.md` is the plan of the study, committed before any result, followed by the choice on the EV9V calibration half and the report on the selection half. There, 92.1% of the samples keep a view and 1.0% of them are wrong, against 3.0% when every sample is named. The check passed by a small margin, and the study does not show that the rule beats a plain threshold on the model's confidence. The rule is in `echoview.reference.UNCERTAIN`.
- `echoview.evaluation.selective` scores the labelled part and the uncertain part of an output separately. `python -m echoview.evaluation.uncertain_study` runs the two steps of the study. Its output files are in `docs/uncertain/`.
- The system has a name: **EchoRouter**. `C2_ens_v1` stays the identifier of the published weights, and the weights do not change. `echoview.MODEL_NAME` holds the name.
- `tools/render_demo.py` renders a stream with the view the system gives every 0.1 s under it, as an animated WebP, a GIF or an MP4. The README opens with two streams of public EV9V recordings, run with the uncertain output.
- The README compares the system with four published approaches on the EV9V test benchmark.

### Removed

- The Docker images of versions 1.0.0 and 1.1.0 were withdrawn from the registry: they held Pillow 12.3.0. The releases themselves and their wheels stay.

### Checked on a GPU (L40S, 2026-11-20)

- The study of the uncertain output: the window logits of the published ensemble for all 1,085 development sequences, then the two commands of the study.
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
