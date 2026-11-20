---
license: other
license_name: cedars-sinai-academic-software-license
license_link: https://github.com/echonet/EchoPrime/blob/main/LICENSE
library_name: pytorch
pipeline_tag: video-classification
tags:
  - echocardiography
  - ultrasound
  - view-classification
  - temporal-segmentation
  - medical-imaging
datasets:
  - bgx666/EV9V
---

# echo-view-routing: the C2 ensemble (`C2_ens_v1`)

This model reads an echocardiography recording and names its view: PLAX, PSAX, A4C, A5C or SC4C. For a longer stream, such as several recordings of one examination joined together, it also says where one view ends and the next begins.

The code is at <https://github.com/WilliamQiuzy/echo-view-routing>. This repository holds the weights.

> **Licence.** The weights were fine-tuned from the EchoPrime video encoder and fall under EchoPrime's Academic Software License: academic and nonprofit research only. See [Licence](#licence).

## Use it

```bash
pip install "git+https://github.com/WilliamQiuzy/echo-view-routing"
echoview selfcheck       # downloads the weights and five sample clips, then compares with stored results
echoview classify /path/to/recordings --out views.csv
echoview segment clip1.dcm clip2.dcm clip3.dcm --out segments.csv
```

```python
import echoview

system = echoview.load()
recording = echoview.read_recording("clip.mp4")      # DICOM cine, video file or folder of frames
print(system.classify(recording.frames))             # view, probability, agreement of the three seeds
print(system.segment(recording.frames).segments)     # (start_s, end_s, view, mean probability)
```

The code checks every weight file against its SHA-256 before it calls the result the reference system.

## What the model is

- **Input.** Each recording is brought to 10 frames per second. Each frame is resized so that its short side is 256 pixels, cropped to the central 224 × 224 and reduced to grey levels. There is no masking and no intensity correction.
- **Window model.** One prediction every 0.5 s. Each prediction reads 16 consecutive frames (1.6 s) around the requested time. The network is torchvision's MViTv2-S video encoder with a 512-d projection, a second input stem that marks the requested frame, and a linear five-class head. It has 34.6 million parameters.
- **Ensemble.** Three training seeds of the same network. Their probabilities are averaged.
- **Recording view.** The mean of the window probabilities. The largest of the five wins.
- **Stream segmentation.** Every 0.1 s sample takes the probability of the nearest window. A change-point decoder cuts where the probabilities before and after a sample differ and the two sides name different views. The difference is a Jensen-Shannon divergence over 0.7 s on each side, and the threshold is 0.1738. The setting is in `decoder.json`.

## Training

- **Data.** The training split of the public EV9V dataset only: 3,364 recordings from one hospital, made on Philips machines.
- **Three stages.** The encoder started from the released EchoPrime video encoder. Stage 1 trained the window model on EV9V with cross-entropy. Stage 2 continued with an added smoothing penalty between neighbouring windows (the truncated log-probability smoothing of MS-TCN, weight 0.15). Stage 3 continued for 1,500 updates with the same objective and strong input augmentation: zoom, shift, rotation, grey-level maps, gain, blur, noise and borders. Stage 3 is what makes the model usable on another vendor's recordings.
- **No MIMIC training.** No MIMIC-IV-Echo recording, labelled or not, was used to train the network. MIMIC recordings were used, with another classifier's labels, for two choices: those of 372 patients to choose between training recipes, and those of 866 further patients when the decision was made to run the three seeds as one ensemble.

## Results

All numbers are for this ensemble as it is in this repository.

| Data | What was measured | Result |
|---|---|---|
| EV9V development halves, streams built from labelled recordings | segment F1 at 50% overlap, selection half and calibration half | 0.959 and 0.984 |
| EV9V frozen test benchmark, 120 streams of 4 to 8 fragments | segment F1 at 50% overlap | 0.909 (95% CI 0.883 to 0.934) |
| EV9V frozen test benchmark, 800 single recordings | accuracy and macro-F1 | 98.1% and 0.956 |
| MIMIC-IV-Echo, 9,341 recordings of 1,001 patients (GE machines) | same view as another classifier's table, mean over its classes PLAX_LV, PSAX_non_zAV and A4C_general | 95.8% (95.2 to 96.3) |
| MIMIC-IV-Echo, 1,139 streams of 988 of those patients | segment F1 at 50% overlap against that table | 0.882 (0.870 to 0.894) |

Segment F1 at 50% overlap counts a predicted segment as right when it overlaps a reference segment of the same view with an intersection over union of at least 0.5.

How to read these numbers:

- **The MIMIC rows are agreement between two models, not accuracy.** MIMIC-IV-Echo has no reader's view labels. The reference there is the output of another view classifier, which has no A5C and no SC4C class. When an A5C answer is counted as A4C, the stream F1 is 0.935 (0.925 to 0.943).
- **The MIMIC patients of these two rows played no part in forming this system.** The system and its decoder were fixed before these patients were read, once. A larger ensemble of six models was read on the same patients as a second system and scored 0.009 higher on streams. The project kept the three-seed ensemble.
- **The EV9V test benchmark is not untouched.** Earlier rounds of the project were evaluated on it.
- **The augmentation of stage 3 is what the MIMIC agreement rests on.** On 8,033 recordings of 866 other MIMIC patients, the stage-2 model agreed with the other classifier on 84.7% and single stage-3 seeds on 94.6%. On the EV9V test benchmark the two show no consistent difference.
- **What this code reproduces.** `echoview evaluate` reproduces the development row exactly. On 2026-11-17 it was run beside the research code on the same weights: the logits differed by less than 6e-6 and every metric was identical. `echoview evaluate --benchmark test` runs the test benchmark: on 2026-11-20 its evaluation was run on the published weights and gave the numbers of the two test-benchmark rows, with their intervals, to the last digit. The MIMIC rows were produced by the research code and are not part of this package.

## Limits

- **Five views only.** The model has no "other" class. A recording of any other view (A2C, A3C, subcostal IVC, Doppler and so on) still gets one of the five names, often with a high probability. Its probability does not tell an in-scope recording from an out-of-scope one. Filter by another method first if the input is a whole study.
- **A4C against A5C is its weakest distinction** on EV9V, where both views have reference labels.
- **One training site and vendor.** Robustness to GE recordings comes from input augmentation. It was measured only as agreement with another classifier, not against a reader.
- **Research use only.** This is not a medical device and it is not validated for clinical use.

## Files

| Path | Content |
|---|---|
| `C2_s0/`, `C2_s1/`, `C2_s2/` | one folder per training seed |
| `C2_s*/model.safetensors` | the weights; the code loads these |
| `C2_s*/final.pt` | the frozen training output, a plain PyTorch state dict; `model.safetensors` holds the same tensors bit for bit |
| `C2_s*/config.json`, `train_log.jsonl` | the stage-3 training configuration and its log |
| `C2_s*/decoder.json` | the decoder setting of that seed when it is run alone |
| `decoder.json` | the decoder setting of the ensemble |
| `config.json` | the system in one file: classes, members, input, architecture |
| `examples/` | five public EV9V clips, their labelled views and the stored results of the self-check |
| `training/stage2_s*.pt` | the stage-2 models, the start weights of training stage 3 |
| `MANIFEST.json` | SHA-256 and size of every file |

## Licence

- **Weights.** The encoder was initialised from EchoPrime (Vukadinovic, He, Ouyang and colleagues, Cedars-Sinai Medical Center). EchoPrime is released under an [Academic Software License](https://github.com/echonet/EchoPrime/blob/main/LICENSE) for academic and nonprofit research. These weights are a derivative of EchoPrime and are offered under the terms of that licence: research use only, no commercial use, and please do not redistribute them. The licence asks publications to include: "The Software used in this research was created by Milos Vukadinovic, Bryan He, and David Ouyang of Cedars-Sinai Medical Center. © 2024 Cedars-Sinai Medical Center."
- **Code.** The code in the GitHub repository is under the MIT licence.
- **Sample clips.** The clips in `examples/` are from EV9V (`bgx666/EV9V` on the Hugging Face Hub), distributed under CC BY 4.0. Their file names are the dataset's own. EV9V's code PLHLA is the parasternal long axis (PLAX) and PMASA a parasternal short axis (PSAX).
- **Architecture.** The network is torchvision's `mvit_v2_s` (BSD 3-Clause).

## Citation

A paper is in preparation. Until it is out, please cite the code repository (see its `CITATION.cff`) and EV9V.
