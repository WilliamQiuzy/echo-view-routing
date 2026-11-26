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

# EchoRouter

EchoRouter reads an echocardiography recording and names its view: PLAX, PSAX, A4C, A5C or SC4C. For a longer stream, such as several recordings of one examination joined together, it also says where one view ends and the next begins. On request it leaves the weak parts of a stream unnamed and calls them uncertain.

The code is at <https://github.com/WilliamQiuzy/echo-view-routing>. This repository holds the weights: three training seeds that run as one ensemble. In the project's records the system is called `C2_ens_v1`.

> **Licence.** The weights were fine-tuned from the EchoPrime video encoder and fall under EchoPrime's Academic Software License: academic and nonprofit research only. See [Licence](#licence).

## Use it

```bash
pip install "git+https://github.com/WilliamQiuzy/echo-view-routing"
echoview selfcheck       # downloads the weights and five sample clips, then compares with stored results
echoview classify /path/to/recordings --out views.csv
echoview segment clip1.dcm clip2.dcm clip3.dcm --out segments.csv
echoview segment clip1.dcm clip2.dcm clip3.dcm --uncertain --out segments.csv   # weak parts come back as "uncertain"
echoview segment clip1.dcm clip2.dcm clip3.dcm --read-every 0.1 --out segments.csv   # five times the work: nearer cuts at a cost
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
- **A denser read-out, on request.** `--read-every 0.2` or `0.1` asks the window model for a prediction every 0.2 s or 0.1 s and decodes with a setting chosen for that step on development data. This is not the published system. One read of the test benchmark found nearer cuts at a cost: the cuts lie 0.122 s and 0.085 s from the view changes instead of 0.153 s, and 784 of the 800 single recordings are read right instead of 785, a difference that the read cannot tell from none. The code repository's `docs/FOLLOWUP.md`, section 9, has the plan and the result.

## Training

- **Data.** The training split of the public EV9V dataset only: 3,364 recordings of five views. Its authors describe the dataset as routine examinations of adults aged 20 to 68 at the First Affiliated Hospital of Chengdu Medical College, made on Philips CVX and EPIQ 7C machines, with the views named by licensed attending physicians. They give no sex, no diagnosis and no measure of how often the readers agreed. The code repository's `docs/DATA.md` has what is known.
- **Three stages.** The encoder started from the released EchoPrime video encoder. Stage 1 trained the window model on EV9V with cross-entropy. Stage 2 continued with an added smoothing penalty between neighbouring windows (the truncated log-probability smoothing of MS-TCN, weight 0.15). Stage 3 continued for 1,500 updates with the same objective and strong input augmentation: zoom, shift, rotation, grey-level maps, gain, blur, noise and borders. After stage 3 the model agrees far more often with another classifier on another vendor's recordings (see below). Whether that comes from the augmentation or from the added updates was not told apart.
- **No MIMIC training.** No MIMIC-IV-Echo recording, labelled or not, was used to train the network. MIMIC recordings were used, with another classifier's labels, for two choices between trained systems. 3,105 recordings of 369 patients were the yardstick of the rule for the training recipe, where in the end one candidate alone was eligible on EV9V. They were also a step of the rule that chose to run the three seeds as one ensemble; the last step of that rule was the recall of the four-chamber view on CAMUS. The recordings of 866 further patients had been read before that decision and were not in its rule. The code repository's `docs/EVALUATION.md`, "How the system was chosen", has both rules.

## Results

All numbers are for this ensemble as it is in this repository.

| Data | What was measured | Result |
|---|---|---|
| EV9V development halves, streams built from labelled recordings | segment F1 at 50% overlap, selection half and calibration half | 0.959 and 0.984 |
| EV9V frozen test benchmark, 120 streams of 4 to 8 fragments | segment F1 at 50% overlap | 0.909 (95% interval 0.883 to 0.934 with streams drawn, 0.872 to 0.944 with examinations drawn) |
| EV9V frozen test benchmark, 800 single recordings | accuracy and macro-F1 of the view that most decoded samples carry | 98.1% and 0.956 |
| EV9V development, selection half, 577 sequences, with `--uncertain` | samples that keep a view, and wrong samples among them | 92.1%, and 1.0% (0.6 to 1.5) against 3.0% without the option |
| EV9V frozen test benchmark, 1,160 sequences, with `--uncertain` | samples that keep a view, and wrong samples among them | 94.5%, and 2.0% (1.4 to 2.8) against 3.6% without the option |
| MIMIC-IV-Echo, 9,341 recordings of 1,001 patients (GE machines) | same view as another classifier's table, mean over its classes PLAX_LV, PSAX_non_zAV and A4C_general | 95.8% (95.2 to 96.3) |
| MIMIC-IV-Echo, 1,139 streams of 988 of those patients | segment F1 at 50% overlap against that table | 0.882 (0.870 to 0.894) |

Segment F1 at 50% overlap counts a predicted segment as right when it overlaps a reference segment of the same view with an intersection over union of at least 0.5.

How to read these numbers:

- **The MIMIC rows are agreement between two models, not accuracy.** MIMIC-IV-Echo has no reader's view labels. The reference there is the output of another view classifier, which has no A5C and no SC4C class. When an A5C answer is counted as A4C, the stream F1 is 0.935 (0.925 to 0.943).
- **The MIMIC patients of these two rows played no part in forming this system.** The system and its decoder were fixed before these patients were read, once. A larger ensemble of six models was read on the same patients as a second system and scored 0.009 higher on streams. The project kept the three-seed ensemble.
- **The EV9V test benchmark is not untouched.** Models were scored on its recordings many times in the project's early development, and the published model descends from one of them. Since the system was published the benchmark was read for the uncertain output (2026-11-20), twice for a comparison with four other approaches (2026-11-24), once for the denser read-out (2026-11-25) and once for what `classify` returns (2026-11-26). The code repository's `docs/EVALUATION.md` lists the uses that the project's records show. Read the numbers as a description of the system on these recordings; they are likely to be optimistic for new ones.
- **Two intervals for one number.** The streams of the benchmark reuse recordings, so the interval with whole examinations drawn is the safer one to read.
- **Single recordings.** The benchmark gives a recording the view that most of its decoded samples carry, as `echoview segment` does. `echoview classify` takes the largest mean probability. On the development halves the first rule misreads 5 of 521 recordings and the second 3. On the test benchmark the first misreads 15 of 800 and the second 14. One read that was planned beforehand does not show the two rules to differ there (`docs/EVALUATION.md` in the code repository, "One read of the test benchmark for `classify`").
- **Confidence.** On the development halves the expected calibration error of `classify`'s probability is 0.030 and 0.034, with a median probability of 0.98 and three wrong recordings in all. That is too few to say whether a low probability marks a wrong answer. On the test benchmark, in one read that was planned beforehand, the calibration error is 0.025: the mean probability is 0.958 and 786 of the 800 recordings are right (98.25%), so the system is less sure than it is right. The 14 wrong recordings there have a median probability of 0.65 and the right ones 0.98, and a recording that is read right has the higher probability in 97% of the pairs with one that is read wrong. That rests on 14 wrong recordings of one hospital. The probability was not measured on another vendor's recordings.
- **The uncertain output.** Its rule was chosen on the calibration half of the development benchmark and passed its check on the selection half by a small margin. One later read of the frozen test benchmark, planned before it was made, confirmed the rule's tests for whole pieces. It makes the named part cleaner at a price: 5% to 8% of the samples get no name, parts of about a second between two cuts go missing, and every view is named on fewer of its samples (A5C on 74% instead of 84% to 89%). The intervals in the table are over sequences that share their recordings, so they are too narrow as statements about new recordings. The rule is not shown to beat a plain threshold on the model's confidence: on the test benchmark's multi streams a threshold that declines as many samples leaves fewer wrong ones (3.2% against 4.5%). The code repository's `docs/UNCERTAIN.md` has the plan and the full report.
- **The MIMIC agreement came with stage 3.** On 8,033 recordings of 866 other MIMIC patients, single stage-2 seeds agreed with the other classifier's table on 84.7% (83.7 to 85.8) and single stage-3 seeds on 94.6% (94.0 to 95.3). Each number is the mean over three seeds of the agreement averaged over the table's classes PLAX_LV, PSAX_non_zAV and A4C_general, as in the table above, and not a share of the 8,033 recordings. Stage 3 adds input augmentation and 1,500 updates, and no run tells the two apart. On the multi streams of the EV9V test benchmark the stage-2 seeds score 0.916 and the stage-3 seeds 0.912 on average, and the paired difference of every seed has an interval that includes zero. The ensemble scores 0.909, which is 0.003 below the mean of its members and above one of them.
- **What this code reproduces.** `echoview evaluate` reproduces the development row exactly. On 2026-11-17 it was run beside the research code on the same weights: the logits differed by less than 6e-6 and every metric was identical. `echoview evaluate --benchmark test` runs the test benchmark: on 2026-11-20 its evaluation was run on the published weights and gave the numbers of the two test-benchmark rows, with their intervals, to the last digit. The MIMIC rows were produced by the research code and are not part of this package.

## Limits

- **Five views only.** The model has no "other" class. A recording of any other view (A2C, A3C, subcostal IVC, Doppler and so on) still gets one of the five names, often with a high probability. Its probability does not tell an in-scope recording from an out-of-scope one. Filter by another method first if the input is a whole study. The uncertain output does not help here: it marks weak evidence between the five views and says nothing about other views.
- **A4C against A5C is its weakest distinction** on EV9V, where both views have reference labels. Twelve of the fifteen single recordings that the test benchmark has wrong are between these two views.
- **Cuts on a 0.5 s grid.** The published system gives one prediction every 0.5 s, so a cut can be a few tenths of a second off: 0.153 s on average on the test benchmark, where a perfect system on that grid would be 0.121 s off.
- **One training site and vendor.** All training recordings come from one hospital and from Philips machines. On GE recordings the system was measured only as agreement with another classifier, never against a reader.
- **Streams are joins.** Every stream of the benchmarks is built by joining pieces of recordings. A view change there is a hard cut. No continuous examination was measured.
- **The labels are the dataset's.** How often its readers agreed is not published, and its patients are described by an age range only. Nothing here is reported by sex, body size, disease or image quality.
- **Small classes.** The test benchmark has 40 single recordings of A5C and 35 of SC4C.
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
