# echo-view-routing

[![CI](https://github.com/WilliamQiuzy/echo-view-routing/actions/workflows/ci.yml/badge.svg)](https://github.com/WilliamQiuzy/echo-view-routing/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

Echocardiography view recognition and view segmentation for video.

The model reads a recording and names its view: **PLAX, PSAX, A4C, A5C or SC4C**. For a longer stream, such as several recordings of one examination joined together, it also says where one view ends and the next begins. This repository holds the code to run the model, to evaluate it and to train it. The weights are on the Hugging Face Hub.

```text
recording ──► 10 frames per second, grey, 224 × 224
          ──► window model: one prediction every 0.5 s from 16 frames (three seeds, averaged)
          ──► one view per recording, or a change-point decoder that cuts a stream into views
```

## Status

| Part | State |
|---|---|
| Inference (`echoview classify`, `segment`, Python API) | ready, checked against the research code |
| EV9V development evaluation (`echoview evaluate`) | ready, reproduces the research numbers exactly |
| Training, stage 3 (`echoview-train`) | code and tests are here; the end-to-end rehearsal on a GPU is still to run |
| Training, stages 1 and 2 | not ported yet; the stage-2 models are provided as start weights |
| EV9V test benchmark and MIMIC-IV-Echo evaluation | not ported yet; the numbers below come from the research code |
| Weights | private on the Hugging Face Hub, see [Licence](#licence-and-acknowledgements) |

## Quick start

```bash
git clone https://github.com/WilliamQiuzy/echo-view-routing
cd echo-view-routing
python -m venv .venv && source .venv/bin/activate
pip install -e .

hf auth login            # the weights repository is private for now
echoview selfcheck       # downloads the weights and five sample clips, then compares with stored results
```

The self-check takes seconds on a GPU and about three minutes on a laptop CPU. It ends with `SELF-CHECK PASSED`.

```bash
# one view per recording; a folder is searched for DICOM and video files
echoview classify /path/to/recordings --out views.csv

# join recordings in the given order into one stream and cut it into views
echoview segment clip1.dcm clip2.dcm clip3.dcm --out segments.csv --samples samples.csv
```

```python
import echoview

system = echoview.load()                             # the three published seeds as one ensemble
recording = echoview.read_recording("clip.mp4")      # DICOM cine, video file or folder of frames
result = system.classify(recording.frames)
print(result.view, result.probability, result.seeds_agree)

for start, end, view, probability in system.segment(recording.frames).segments:
    print(f"{start:6.1f} - {end:6.1f} s  {view}  {probability:.2f}")
```

Options for every command:

- `--weights DIR` uses a local folder instead of a download. The environment variable `ECHOVIEW_WEIGHTS` does the same.
- `--device cpu|cuda|cuda:1|mps`. The default is an NVIDIA GPU if PyTorch sees one, else the CPU.
- `--seeds 0` runs one seed instead of three. It is three times faster and no longer the reference system.
- `--fps` gives the frame rate of inputs that carry none. A folder of frame images always needs it.

## How it works

- **Input.** Each recording is brought to 10 frames per second. Each frame is resized so that its short side is 256 pixels, cropped to the central 224 × 224 and reduced to grey levels.
- **Window model.** One prediction every 0.5 s. Each prediction reads 16 consecutive frames around the requested time. The network is torchvision's MViTv2-S video encoder with a 512-d projection, a second input stem that marks the requested frame, and a linear five-class head.
- **Ensemble.** Three training seeds of the same network. Their probabilities are averaged.
- **Recording view.** The mean of the window probabilities. The largest of the five wins.
- **Stream segmentation.** Every 0.1 s sample takes the probability of the nearest window. A change-point decoder cuts where the probabilities before and after a sample differ and the two sides name different views.

## Results

All numbers are for the published ensemble, `C2_ens_v1`.

| Data | What was measured | Result |
|---|---|---|
| EV9V development halves, streams built from labelled recordings | segment F1 at 50% overlap, selection half and calibration half | 0.959 and 0.984 |
| EV9V frozen test benchmark, 120 streams of 4 to 8 fragments | segment F1 at 50% overlap | 0.909 (95% CI 0.883 to 0.934) |
| EV9V frozen test benchmark, 800 single recordings | accuracy and macro-F1 | 98.1% and 0.956 |
| MIMIC-IV-Echo, 9,341 recordings of 1,001 patients (GE machines) | same view as another classifier's table | 95.8% (95.2 to 96.3) |
| MIMIC-IV-Echo, 1,139 streams of 988 of those patients | segment F1 at 50% overlap against that table | 0.882 (0.870 to 0.894) |

The MIMIC rows are agreement between two models, not accuracy: MIMIC-IV-Echo has no reader's view labels. The [model card](MODEL_CARD.md) says how to read each row and what the model cannot do.

## Reproduce the evaluation

The development evaluation needs the public EV9V dataset (`bgx666/EV9V` on the Hugging Face Hub, CC BY 4.0) and a GPU.

```bash
pip install -e ".[evaluation]"
echoview evaluate --images /data/EV9V/Images --out report.json
```

It prints one line per system. The line of the ensemble must read `selection F1@50 0.9594  calibration F1@50 0.9837`. See [docs/EVALUATION.md](docs/EVALUATION.md) for the benchmark, the metrics and the decoder rule.

## Train

Stage 3 is the stage that produced the published seeds. It continues a stage-2 model for 1,500 updates with input augmentation.

```bash
echoview-cache --images /data/EV9V/Images --out /data/ev9v_cache           # 18 GB, once
echoview-train --config configs/stage3.json --seed 0 --cache /data/ev9v_cache \
    --init /path/to/weights/training/stage2_s0.pt --out runs/C2_s0
```

One run needs a GPU with 32 GB of memory. See [docs/TRAINING.md](docs/TRAINING.md) for the three stages and what a rerun can and cannot reproduce.

## Repository layout

| Path | Content |
|---|---|
| `src/echoview/` | the package: model, readers, windows, decoder, ensemble, command line |
| `src/echoview/evaluation/` | the EV9V development benchmark, the metrics and the decoder rule |
| `src/echoview/training/` | frame cache, sampling plans, augmentation and the training loop |
| `src/echoview/data/` | the benchmark definition, the list of recordings and the sampling plans |
| `configs/` | training recipes |
| `docs/` | data, evaluation and training in detail |
| `tests/` | unit tests and parity tests against the research code |
| `tools/` | maintainers' scripts: building the Hub folder, exporting definitions from the research code |

## Data

- **EV9V** is public. The package ships no EV9V file. Its definition files name a recording by a hash of its folder name, so they work with your own copy of the dataset. See [docs/DATA.md](docs/DATA.md).
- **MIMIC-IV-Echo** is restricted-access data. Nothing derived from single MIMIC recordings is in this repository. `echoview mimic-table` runs the model on your own credentialed copy.
- **Sample clips.** The five clips of the self-check are EV9V recordings. They are published with the weights.

## Limits

- **Five views only.** The model has no "other" class. A recording of any other view still gets one of the five names, often with a high probability.
- **A4C against A5C is its weakest distinction.**
- **One training site and vendor.** Robustness to another vendor's recordings comes from input augmentation and was measured as agreement with another classifier.
- **Research use only.** This is not a medical device and it is not validated for clinical use.

## Licence and acknowledgements

- **Code.** MIT, see [LICENSE](LICENSE).
- **Weights.** The encoder was initialised from EchoPrime (Vukadinovic, He, Ouyang and colleagues, Cedars-Sinai Medical Center). EchoPrime's Academic Software License prohibits transferring derivatives of it, and these weights are such a derivative. They are private until that question is settled. The licence asks publications to include: "The Software used in this research was created by Milos Vukadinovic, Bryan He, and David Ouyang of Cedars-Sinai Medical Center. © 2024 Cedars-Sinai Medical Center."
- **Data.** EV9V is distributed under CC BY 4.0 by its authors.
- **Architecture.** torchvision's `mvit_v2_s` (BSD 3-Clause).

## Citation

A paper is in preparation. Until it is out, please cite this repository. GitHub's "Cite this repository" button reads [CITATION.cff](CITATION.cff).

## History

This repository was a research log before it became this package. The earlier code, with the reports of every experiment, is kept on the branch `research-archive`.
