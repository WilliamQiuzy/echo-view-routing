# EchoRouter

[![CI](https://github.com/WilliamQiuzy/echo-view-routing/actions/workflows/ci.yml/badge.svg)](https://github.com/WilliamQiuzy/echo-view-routing/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/WilliamQiuzy/echo-view-routing)](https://github.com/WilliamQiuzy/echo-view-routing/releases)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24485965.svg)](https://doi.org/10.5281/zenodo.24485965)
[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Weights on the Hugging Face Hub](https://img.shields.io/badge/weights-Hugging%20Face-yellow.svg)](https://huggingface.co/WilliamQiu123/echo-view-routing)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

EchoRouter reads echocardiography video and names the view: **PLAX, PSAX, A4C, A5C or SC4C**. For a longer stream, such as several recordings of one examination joined together, it also says where one view ends and the next begins. Where its evidence is weak, most often right at a change of view, it can answer **uncertain** instead of naming a view. This repository holds the code to run it, to evaluate it and to train it. The weights are on the [Hugging Face Hub](https://huggingface.co/WilliamQiu123/echo-view-routing).

<p>
  <img src="docs/assets/exam_stream.webp" width="49%" alt="The recordings of one examination joined in their order: the frames the model reads, the view it gives every 0.1 s with the uncertain parts hatched, and the reference below">
  <img src="docs/assets/constructed_stream.webp" width="49%" alt="A stream built from fragments of several recordings: the frames the model reads, the view it gives every 0.1 s with the uncertain parts hatched, and the reference below">
</p>

Two streams of public EV9V recordings, run with `--uncertain`. The left one joins the recordings of one examination in their order (29 s). The right one is built from fragments of several recordings (20 s). The wide lane is EchoRouter's answer for every 0.1 s, hatched where it declines. The narrow lane is the reference.

- **What to look for.** Every named sample of both streams is right. The hatched parts sit at the changes of view, 0.3 s to each side. In the left stream the 1.2 s of A5C between two A4C parts is left unnamed as a whole. The system's own two cuts there are 1.0 s apart, and what their bands leave between them is shorter than the 0.5 s a named part must have.
- **How they were picked.** By hand. Both come from the half of the development benchmark that the rule was not chosen on. Of its 156 multi and examination-order streams, 118 have no wrong named sample, and the two were picked among those.
- **Your own recordings.** `python tools/render_demo.py clip1.mp4 clip2.mp4 --uncertain --out demo.webp` makes such a picture.

```text
recording ──► 10 frames per second, grey, 224 × 224
          ──► window model: one prediction every 0.5 s from 16 frames (three seeds, averaged)
          ──► one view per recording, or a change-point decoder that cuts a stream into views
          ──► on request: the weak parts of a stream are marked uncertain instead of named
```

## How it compares

On the project's frozen EV9V test benchmark we scored four approaches from the literature beside EchoRouter, on the same recordings and the same streams: each with its own way of decoding ("as published") and with EchoRouter's decoder on its outputs. We ran all of them ourselves, three of them from our own training of the authors' code. The rows, the numbers and the wording were fixed before this read of the benchmark ([docs/COMPARISON.md](docs/COMPARISON.md)). It was not the first read: earlier numbers of six of the rows were known, and the plan says so. Bold is the best value of a column, by the value alone.

| Row | Segment F1 at 50%, multi | Segment F1 at 50%, pairs | Edit | Piece midpoint | Wrong samples | Cut precision | Cut recall | Cut distance (s) | Recordings: accuracy | Macro-F1 | A5C recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EchoRouter | **0.909** | 0.985 | **91.4** | 93.7% | 9.4% | 0.934 | 0.937 | 0.153 | 98.1% | 0.956 | **95.0%** |
| EchoViewCLIP | 0.867 | 0.989 | 89.0 | 92.9% | 11.2% | 0.851 | 0.953 | 0.183 | **98.9%** | **0.972** | 92.5% |
| STFM | 0.859 | 0.971 | 88.1 | 92.9% | 12.0% | 0.812 | 0.879 | 0.252 | 98.2% | 0.960 | 87.5% |
| ASFormer | 0.856 | 0.981 | 87.2 | 91.0% | **9.0%** | 0.869 | 0.942 | 0.013 | 98.2% | 0.956 | 77.5% |
| EchoPrime per frame | 0.496 | 0.743 | 54.9 | 89.6% | 12.1% | 0.357 | **0.974** | 0.017 | 97.9% | 0.943 | 67.5% |
| EchoViewCLIP + decoder | 0.884 | **0.993** | 90.1 | 92.9% | 11.1% | 0.877 | 0.955 | 0.184 | 98.8% | 0.969 | 90.0% |
| STFM + decoder | 0.865 | 0.972 | 88.4 | 91.7% | 12.8% | 0.808 | 0.844 | 0.283 | 98.1% | 0.957 | 87.5% |
| ASFormer + decoder | 0.883 | 0.988 | 88.3 | 89.9% | 9.9% | **0.961** | 0.918 | 0.008 | 98.2% | 0.956 | 77.5% |
| EchoPrime + decoder | 0.851 | 0.988 | 85.0 | 88.3% | 11.9% | 0.945 | 0.900 | **0.005** | 97.9% | 0.943 | 67.5% |
| EchoRouter, no decoder | 0.874 | 0.975 | 89.5 | **94.0%** | 9.1% | 0.848 | 0.955 | 0.163 | 98.4% | 0.960 | 92.5% |

"Multi" streams join four to eight pieces of recordings, "pairs" two. Piece midpoint is the share of pieces whose middle sample gets the right view. Cut distance is how far a cut lies from the view change it belongs to.

- **Streams, as published.** EchoRouter has the highest segment F1 on multi streams, 0.909 (95% interval 0.872 to 0.944, with whole examinations drawn). The four others score 0.851 to 0.867. For EchoPrime's classifier that is with a decoder setting of its own from October; per frame it scores 0.496.
- **Streams, with the same decoder.** This is the comparison that was named in advance, with 98.75% intervals.
  - EchoViewCLIP with EchoRouter's decoder scores 0.884. EchoRouter is not shown to differ from it: the difference is 0.025 (−0.010 to 0.063). Of the published gap of 0.042, 0.025 remains.
  - ASFormer with the decoder scores 0.883. EchoRouter is not shown to differ from it either: the difference is 0.026 (−0.008 to 0.066).
  - EchoRouter is ahead of STFM with the decoder (0.865) by 0.044 (0.006 to 0.083).
  - EchoRouter is ahead of EchoPrime's classifier with the decoder by 0.058 (0.010 to 0.109).
  - STFM and ASFormer were answered by a second read of the benchmark on the same day, for these two rows only. It was decided after the first result was known, planned in the same document, and could change no other number.
- **Where EchoRouter has the best value:** segment F1 on multi streams, the Edit score and A5C. It gets the middle of 82.8% of the A5C pieces right, the others 57.0% to 79.5%. Against EchoViewCLIP and against STFM that difference is not shown: its intervals include zero. The price is A4C. EchoRouter reads A4C as A5C in 10 of the 231 single recordings of A4C, the others in 1 to 8.
- **Where it does not:**
  - Single recordings: EchoViewCLIP is the best row, 98.9% against 98.1%.
  - Pair streams: all rows but one are close, and EchoViewCLIP with the decoder is highest.
  - Exact cuts: ASFormer and EchoPrime's classifier give an output for every sample, and their cuts are at most 0.02 s off on average. EchoRouter is read every 0.5 s and is 0.15 s off on average; a perfect system on that grid would be 0.12 s off. Reading it every 0.1 s brings its cuts to 0.085 s on this benchmark, against 0.153 s, at a cost: it reads 784 of the 800 single recordings right where the published system reads 785. One recording is a difference that this read cannot tell from none. That option is off by default ([docs/FOLLOWUP.md](docs/FOLLOWUP.md), section 9).
  - Precision of the cuts: ASFormer with the decoder has the highest, 0.961 against 0.934.
  - Wrong samples: ASFormer has the fewest, 9.0% against 9.4%.
- **How to read the table.** We ran all five. STFM, ASFormer and EchoViewCLIP were trained by us on EV9V's training recordings, and EchoPrime's classifier is the released one. Each of the four is one training run. Only STFM's training could be checked against its authors' numbers: the released command comes within the spread of the paper's row for that configuration and stays below the paper's main result ([docs/COMPARISON.md](docs/COMPARISON.md), section 12). The benchmark is ours. Its streams are built from pieces of recordings, it is not the task these methods were published for, and it is not an untouched test set: the project scored models on these recordings and streams many times before it sealed them ([docs/EVALUATION.md](docs/EVALUATION.md), "The history of the test benchmark"). On development data EchoRouter's lead over EchoViewCLIP with the decoder was larger (0.094); the test benchmark did not bear that out.

## Quick start

```bash
git clone https://github.com/WilliamQiuzy/echo-view-routing
cd echo-view-routing
python -m venv .venv && source .venv/bin/activate
pip install -e .

echoview selfcheck       # downloads the weights and five sample clips, then compares with stored results
```

The package needs Python 3.10 or newer. The self-check takes seconds on a GPU and about three minutes on a laptop CPU. It ends with `SELF-CHECK PASSED`.

`pip install -e .` accepts any recent PyTorch. To get the exact versions the project was checked with (Python 3.12), install the pinned requirements first: `pip install -r requirements.txt && pip install --no-deps -e .`

**With Docker**, nothing is installed on the host:

```bash
docker run --rm -v echoview-cache:/cache ghcr.io/williamqiuzy/echo-view-routing:latest echoview selfcheck
```

Add `--gpus all` on a machine with an NVIDIA GPU. Without one, a container is slower than an installed package: on an Apple-silicon laptop the self-check took about twenty minutes in the container. [docs/DOCKER.md](docs/DOCKER.md) shows how to mount recordings and where results go.

```bash
# one view per recording; a folder is searched for DICOM and video files
echoview classify /path/to/recordings --out views.csv

# join recordings in the given order into one stream and cut it into views
echoview segment clip1.dcm clip2.dcm clip3.dcm --out segments.csv --samples samples.csv

# the same, and leave the weak parts unnamed: they come back as pieces called "uncertain"
echoview segment clip1.dcm clip2.dcm clip3.dcm --uncertain --out segments.csv

# read the models every 0.1 s instead of every 0.5 s: about five times the work. On the test benchmark the cuts
# lie nearer to the view changes (0.085 s against 0.153 s), and one single recording more of 800 is misread
# (docs/FOLLOWUP.md, section 9)
echoview segment clip1.dcm clip2.dcm clip3.dcm --read-every 0.1 --out segments.csv
```

```python
import echoview

system = echoview.load()                             # the three published seeds as one ensemble
recording = echoview.read_recording("clip.mp4")      # DICOM cine, video file or folder of frames
result = system.classify(recording.frames)
print(result.view, result.probability, result.seeds_agree)

for start, end, view, probability in system.segment(recording.frames).segments:
    print(f"{start:6.1f} - {end:6.1f} s  {view}  {probability:.2f}")

declining = system.segment(recording.frames, echoview.uncertain.released())   # with the uncertain output
print(declining.uncertain.mean())                                             # the share of samples left unnamed
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
- **A denser read-out, on request.** `--read-every 0.2` or `0.1` asks the window model for a prediction every 0.2 s or 0.1 s and decodes with a setting chosen for that step. This is not the published system. It was chosen on development data. One read of the test benchmark found nearer cuts at a cost for both steps: the cuts lie 0.122 s and 0.085 s from the view changes where the published system's lie 0.153 s away, segment F1 is not lower in value, and 784 of the 800 single recordings are read right where the published system reads 785. The macro-F1 of the single recordings is lower by 0.004 (95% interval -0.015 to +0.005) and by 0.003 (-0.011 to 0.000), so the read does not show whether that cost is real ([docs/FOLLOWUP.md](docs/FOLLOWUP.md), section 9). It does not combine with `--uncertain`.
- **Uncertain output, on request.** The decoder runs as above. Then the 0.3 s before and after every cut are marked uncertain, and so is a part whose view hangs on a weak cut or on a narrow margin between two views. Every other sample keeps the view it has without the option.
- **Speed.** On one NVIDIA L40S the published system works through a stream of 60 s in 4.8 s, which is 12.6 times as fast as the stream plays, and PyTorch holds 2.6 GiB of GPU memory for it. A read-out every 0.2 s takes 11.9 s and one every 0.1 s takes 24.2 s. Reading and decoding the files comes on top ([docs/EVALUATION.md](docs/EVALUATION.md), "Speed").

## Results

All numbers are for the published weights (identifier `C2_ens_v1`).

| Data | What was measured | Result |
|---|---|---|
| EV9V development halves, streams built from labelled recordings | segment F1 at 50% overlap, selection half and calibration half | 0.959 and 0.984 |
| EV9V frozen test benchmark, 120 streams of 4 to 8 fragments | segment F1 at 50% overlap | 0.909 (95% interval 0.883 to 0.934 with streams drawn, 0.872 to 0.944 with examinations drawn) |
| EV9V frozen test benchmark, 800 single recordings | accuracy and macro-F1 of the view that most decoded samples carry | 98.1% and 0.956 |
| EV9V development, selection half, 577 sequences, with `--uncertain` | samples that keep a view, and wrong samples among them | 92.1%, and 1.0% (0.6 to 1.5) against 3.0% without the option |
| EV9V frozen test benchmark, 1,160 sequences, with `--uncertain` | samples that keep a view, and wrong samples among them | 94.5%, and 2.0% (1.4 to 2.8) against 3.6% without the option |
| MIMIC-IV-Echo, 9,341 recordings of 1,001 patients (GE machines) | same view as another classifier's table | 95.8% (95.2 to 96.3) |
| MIMIC-IV-Echo, 1,139 streams of 988 of those patients | segment F1 at 50% overlap against that table | 0.882 (0.870 to 0.894) |

- **The MIMIC rows are agreement between two models, not accuracy.** MIMIC-IV-Echo has no reader's view labels.
- **Two intervals for one number.** The streams of the benchmark reuse recordings, so the interval with examinations drawn is the safer one to read ([docs/EVALUATION.md](docs/EVALUATION.md), "Two sets of intervals").
- **Single recordings.** The benchmark gives a recording the view that most of its decoded samples carry, as `echoview segment` does. `echoview classify` takes the largest mean probability. On the development halves the first rule misreads 5 of 521 recordings and the second 3. On the test benchmark `classify`'s rule was not scored.
- **Not an untouched test set.** The test recordings were used often in the project's early development.

The [model card](MODEL_CARD.md) says how to read each row and what the model cannot do. [docs/EVALUATION.md](docs/EVALUATION.md) has the confusion table of the 800 recordings, the model's calibration on development data, how the system was chosen and the history of the test benchmark.

## The uncertain output

By default EchoRouter names a view for every 0.1 s of a stream, also where two views meet and where a part looks like two views at once. With `--uncertain` it declines there. The rule was chosen on one half of the EV9V development benchmark and checked on the other half. The project's frozen test benchmark was then read once with the finished rule.

| | Samples that keep a view | Wrong among them |
|---|---|---|
| Development half the rule was not chosen on (577 sequences), every sample named | 100% | 3.0% |
| The same, with `--uncertain` | 92.1% | 1.0% |
| Test benchmark (1,160 sequences), every sample named | 100% | 3.6% |
| The same, with `--uncertain` | 94.5% | 2.0% |

- **What becomes uncertain.** Of the samples the default output has wrong, 68% on the development half and 48% on the test benchmark. Of those it has right, 6% and 4%. In both, 92% to 93% of the true changes of view have an uncertain sample beside them.
- **What it costs.** A part of about a second between two cuts is left unnamed as a whole. A5C loses most: 74% of its samples keep the name A5C, against 89% and 84% without the option.
- **What it is not.** Uncertain does not mean "none of the five views". The model cannot tell that. And the rule is not shown to beat a plain threshold on the model's confidence. A threshold that declines as many samples leaves 1.1% wrong on the development half and 1.8% on the test benchmark, and in the test benchmark's multi streams it is clearly ahead, with 3.2% against 4.5%.
- **Where it holds.** On EV9V streams joined from recordings, for the ensemble of three seeds. A continuous examination was not measured. The sequences of each benchmark share their recordings, so the intervals in the results table are too narrow as statements about new recordings.

[docs/UNCERTAIN.md](docs/UNCERTAIN.md) has the plan, which was committed before the results, the choice, the full two-part report and the read of the test benchmark.

## Reproduce the evaluation

Both evaluations need the public EV9V dataset (`bgx666/EV9V` on the Hugging Face Hub, CC BY 4.0) and a GPU. Run beside the research code on the same weights, they give its numbers exactly.

```bash
pip install -e ".[evaluation]"
echoview evaluate --images /data/EV9V/Images --out report.json
```

It prints one line per system. The line of the ensemble must read `selection F1@50 0.9594  calibration F1@50 0.9837`.

The frozen test benchmark is run with `echoview evaluate --benchmark test --images /data/EV9V/Images`. Its first line must read `multi streams   segment F1@50 0.9089 (95% interval 0.883 to 0.934)`.

See [docs/EVALUATION.md](docs/EVALUATION.md) for the benchmarks, the metrics, the decoder rule and the study of the uncertain output.

## Train

The model was trained in three stages, each with its recipe in `configs/`. Stage 3 is the stage that produced the published seeds. It continues a stage-2 model for 1,500 updates with input augmentation.

The training code was checked against the research trainer and the published training logs. In a full rerun of the three seeds with this package, each seed lands within 0.004 of its published seed on both development halves, and the ensemble within 0.003 when it is decoded with the published setting. With a decoder setting chosen again by the rule, the rerun ensemble scores 0.943 on the selection half against the published 0.959, so a retrained ensemble should keep the published setting ([docs/TRAINING.md](docs/TRAINING.md)).

```bash
echoview-cache --images /data/EV9V/Images --out /data/ev9v_cache           # 18 GB, once
echoview-train --config configs/stage3.json --seed 0 --cache /data/ev9v_cache --init stage2 --out runs/C2_s0
```

`--init stage2` downloads the stage-2 model of the seed from the weights repository and checks it.

Stage 3 needs a GPU with 32 GB of memory. On a machine that can be switched off, add `--checkpoint-every 100 --resume`: the run then continues from its last checkpoint. See [docs/TRAINING.md](docs/TRAINING.md) for all three stages and what a rerun can and cannot reproduce.

## Repository layout

| Path | Content |
|---|---|
| `src/echoview/` | the package: model, readers, windows, decoder, ensemble, command line |
| `src/echoview/evaluation/` | the EV9V development and test benchmarks, the metrics, the intervals and the decoder rule |
| `src/echoview/training/` | frame cache, sampling plans, augmentation and the training loop |
| `src/echoview/data/` | the benchmark definitions, the list of recordings and the sampling plans |
| `configs/` | training recipes |
| `docs/` | data, evaluation and training in detail |
| `tests/` | unit tests and parity tests against the research code |
| `tools/` | `render_demo.py` for the animations above, and maintainers' scripts: building the Hub folder, exporting definitions from the research code |
| `requirements.txt`, `requirements-dev.txt` | the pinned versions the project was checked with |
| `Dockerfile` | the image: the package on top of the pinned requirements |

## Data

- **EV9V** is public. The package ships no EV9V file. Its definition files name a recording by a hash of its folder name, so they work with your own copy of the dataset. See [docs/DATA.md](docs/DATA.md).
- **MIMIC-IV-Echo** is restricted-access data. Nothing derived from single MIMIC recordings is in this repository. `echoview mimic-table` runs the model on your own credentialed copy.
- **Sample clips.** The five clips of the self-check are EV9V recordings. They are published with the weights.

## Limits

- **Five views only.** The model has no "other" class. A recording of any other view still gets one of the five names, often with a high probability. The uncertain output does not change this: it marks weak evidence between the five views.
- **A4C against A5C is its weakest distinction.**
- **Cuts on a 0.5 s grid.** The published system gives one prediction every 0.5 s, so a cut can be a few tenths of a second off. `--read-every 0.1` reads more often: on the test benchmark its cuts are 0.085 s off instead of 0.153 s, at about five times the work, and one single recording more of 800 is misread.
- **One training site and vendor.** All training recordings come from one hospital and from Philips machines. On another vendor's recordings the system was measured only as agreement with another classifier, never against a reader. That agreement rose with training stage 3, which adds input augmentation and 1,500 updates; the two were not told apart.
- **Streams are joins.** Every stream of the benchmarks is built by joining pieces of recordings. The pieces are 1 to 4 s long, and about half of those in the multi streams are shown with changed grey levels. A view change there is a hard cut. Streams that join the recordings of one examination in their order were scored on development data only (segment F1 0.946 and 0.959), and no continuous examination was measured.
- **The labels are the dataset's.** EV9V's views were named by its authors' readers, licensed attending physicians. How often they agreed is not published, and the dataset describes its patients by an age range only, so nothing here is reported by sex, body size, disease or image quality.
- **Small classes.** The test benchmark has 40 single recordings of A5C and 35 of SC4C. One recording moves their recall by 2.5 and 2.9 points.
- **Confidence.** The model's probabilities were checked for calibration on development data only, where three recordings are wrong in all ([docs/EVALUATION.md](docs/EVALUATION.md)).
- **Research use only.** This is not a medical device and it is not validated for clinical use.

## Licence and acknowledgements

- **Code.** MIT, see [LICENSE](LICENSE).
- **Weights.** Not MIT. The encoder was initialised from EchoPrime (Vukadinovic, He, Ouyang and colleagues, Cedars-Sinai Medical Center), which is released under an [Academic Software License](https://github.com/echonet/EchoPrime/blob/main/LICENSE) for academic and nonprofit research. The weights are a derivative of EchoPrime and are offered under the terms of that licence: research use only, no commercial use, and please do not redistribute them. The licence asks publications to include: "The Software used in this research was created by Milos Vukadinovic, Bryan He, and David Ouyang of Cedars-Sinai Medical Center. © 2024 Cedars-Sinai Medical Center."
- **Data.** EV9V is distributed under CC BY 4.0 by its authors.
- **Architecture.** torchvision's `mvit_v2_s` (BSD 3-Clause).

## Citation

A paper is in preparation. Until it is out, please cite this repository. GitHub's "Cite this repository" button reads [CITATION.cff](CITATION.cff).

Every release from 1.2.1 on is archived on Zenodo. The DOI [10.5281/zenodo.24485965](https://doi.org/10.5281/zenodo.24485965) stands for all versions and leads to the newest one. Each release also has a DOI of its own on that page; cite it when the exact version matters.
