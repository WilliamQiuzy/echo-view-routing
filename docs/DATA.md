# Data

## EV9V

EV9V is a public dataset of echocardiography recordings with view labels (`bgx666/EV9V` on the Hugging Face Hub, CC BY 4.0). Its authors describe it in Gou and colleagues, "Spatio-Temporal Fusion Model for Standard View Classification of Echocardiographic Videos", arXiv:2606.17437. The network was trained on its training split and on nothing else. Other data entered two choices between trained systems; see "MIMIC-IV-Echo" and "Other datasets" below.

**What its authors say about it.** This is the dataset's card as it read on 2026-11-25. Beyond the counts, this project could check none of it.

- **Where and when.** Routine clinical examinations at the First Affiliated Hospital of Chengdu Medical College, between January 2020 and January 2023, on two ultrasound systems: the Philips CVX and the Philips EPIQ 7C.
- **Who.** 703 adults aged 20 to 68 who came for a routine cardiac assessment or for screening for suspected heart disease. The card gives no sex, no diagnosis and no other description of the patients, and no rule by which a patient or a recording was taken in or left out.
- **How much.** 5,138 recordings with 910,579 frames in nine views: 3,683 recordings of 504 patients for training, 567 of 72 for validation and 888 of 127 for testing. The card says that the split is by patient.
- **Who named the views.** Several attending physicians labelled each recording independently and blind to each other. Associate chief physicians or senior experts decided the cases on which they differed, and senior experts checked the result once more. The card gives neither the number of readers nor how often they agreed.
- **Ethics.** Approved by the ethics committee of that hospital (approval number 2026CYFYIRB-BA-067), with all identifying information removed. The card does not mention consent.

**What this project uses and found.**

- **Five views.** 3,364 training recordings, 521 development recordings and 800 test recordings of PLAX, PSAX, A4C, A5C and SC4C. A recording of the test benchmark lasts 0.9 to 53.5 s, and half of them at most 3.9 s.
- **Dates in the names.** The names of the training and validation recordings carry dates from March to July 2022, on 55 days. That is a shorter time than the card gives. Whether a date in a name is the day of the examination is not said anywhere. The test list was not opened for this count.
- **Labels.** EV9V has nine view codes. PLHLA counts as PLAX. The four parasternal short-axis codes (PASA, PMVLSA, PPMLSA and PMASA) count as PSAX. A4C, A5C and SC4C keep their names. Recordings with the ninth code, PMPALA, are not used.
- **Layout.** The dataset ships each recording as a folder of JPEG frames, `Images/<recording>/<frame number>.jpg`, at 30 frames per second and 320 × 240 pixels. This project reads every third frame, which gives 10 frames per second.
- **Download.** Get `Images` from the dataset's page on the Hugging Face Hub and unpack it. Every command of this package takes the path of that folder.

## The definition files in this package

Four kinds of file say which frames are read. All are in `src/echoview/data/`.

| File | What it defines |
|---|---|
| `ev9v_dev.json.gz` | the development benchmark: the sequences of both halves, fragment by fragment |
| `ev9v_test.json.gz` | the frozen test benchmark: its sequences, fragment by fragment |
| `ev9v_recordings.json.gz` | the recordings of the training frame cache: split, label and length |
| `plans/groups_s*.npz`, `plans/windows_s*.npz` | the sampling plans: which windows the training updates draw from |

**No file names a recording.** A recording is named by a key: the first 16 hex digits of the SHA-256 of its EV9V folder name. The code hashes the folder names of your copy of EV9V and finds every recording that way (`echoview.evaluation.ev9v.recording_key`). If a recording is missing from your copy, the command stops and says how many are missing.

## Splits

- **The dataset's own split is kept.** Training recordings come from EV9V's training split, development recordings from its validation split and test recordings from its test split.
- **Sessions.** EV9V has no patient numbers. A recording's name ends in a running number, and the name without that number is called a session here. The documents of this project also call it an examination. Counted that way the three splits have 504, 72 and 127 sessions, which equal the card's numbers of patients. The card does not say how the names relate to patients, so it is not established that a session is one patient.
- **No session is in two splits.**
  - Training and validation share no session, no time stamp and no recording. This was counted on 2026-11-25 from the dataset's two lists.
  - The test split shares no session and no recording with either. This was counted on 2026-10-23, when the test benchmark was built.
  - The tests of this package check that no recording is in two of the definition files.
- **Sessions with one of the five views.** 502, 72 and 125. Two training sessions and two test sessions hold recordings of the ninth view only.
- **Training.** 3,364 recordings. Only these are read by the training code.
- **Development, two halves.** 284 recordings in the `selection` half and 237 in the `calibration` half. The halves were split by session, 36 in each, so they share no session and no recording. The decoder setting is chosen on `calibration`. The development numbers are read from `selection`.
- **Test.** 800 recordings from 125 examinations. `echoview evaluate --benchmark test` reads them, and so do the planned reads that [UNCERTAIN.md](UNCERTAIN.md), [COMPARISON.md](COMPARISON.md), [FOLLOWUP.md](FOLLOWUP.md) and [EVALUATION.md](EVALUATION.md) report. They share no recording with the training and development data, and nothing in training or in the choice of the decoder setting reads them. They were read often before the project sealed them; [EVALUATION.md](EVALUATION.md), "The history of the test benchmark", lists the uses that the project's records show.

## How the streams are built

EV9V holds single-view recordings, so streams with view changes are built by joining fragments of recordings. The rules below are those of the project's research code, which wrote the definition files once. The files hold its whole output, and the counts here were taken from them.

- **A fragment** is a run of 1.0 to 4.0 s from one recording. Its length is drawn uniformly from 10 to 40 samples, or up to the recording's length where that is shorter, and its place in the recording uniformly. A recording shorter than 1.0 s gives no fragment.
- **Pair streams** join two fragments.
  - The recordings are shuffled and dealt into four groups: the same view or two different views, each with and without a change of grey levels on the second fragment. A recording is used once.
  - The test benchmark has 60 streams in each group: 120 joins with a view change and 120 without.
  - The views are not balanced. The pair streams of the test benchmark hold four A5C fragments and no SC4C fragment.
- **Multi streams** join four to eight fragments, the number drawn uniformly.
  - The first view is drawn uniformly from the five. Each next fragment keeps the view with probability 0.35 and otherwise takes one of the four other views, uniformly.
  - Its recording is drawn uniformly among the recordings of that view that are not yet in the stream.
  - A recording is never twice in one stream, but it comes back in other streams. The 701 fragments of the test benchmark's multi streams come from 405 recordings, and one recording is used 11 times. They have 379 joins with a view change and 202 without.
- **The change of grey levels.** Part of the fragments are shown with a gamma of 0.9: a grey level `i` becomes `255 × (i / 255)^0.9` before the frame is resized.
  - In multi streams each fragment gets it with probability 0.5: 362 of the 701 in the test benchmark. In pair streams the second fragment of two of the four groups gets it: 120 of 480. Whole recordings and examination streams are never changed.
  - The design asks two things of a system. A change of appearance without a change of view should not make it cut. And a change of view should not be found by the change of appearance alone.
  - It is a plain change of brightness. It is not a model of how ultrasound images differ between machines.
- **Examination streams** (development only) join all recordings of one session that show one of the five views, whole, in the order of the running numbers of their names. That order is taken to be the order of acquisition; the dataset does not say so. 33 of the 449 joins skip a recording of the ninth view.
- **Seeds.** Every draw comes from a generator with a fixed seed: one for each kind of stream of the test benchmark and of each half of the development benchmark.
- **What the streams are not.** A join is a hard cut between two recordings, nearly always of two sessions. In an examination the probe moves from one view to the next, and no recording of this dataset shows that.

## The development benchmark

| Kind | What it is | Selection | Calibration |
|---|---|---|---|
| `native` | one whole recording | 284 | 237 |
| `pairs` | two fragments joined: the same view or two different views | 137 | 115 |
| `multi` | four to eight fragments joined | 120 | 120 |
| `exam` | every recording of one examination, in the order of acquisition | 36 | 36 |

A join of two fragments of the same view is a splice without a view change. The metric `same_class_cut_rate` counts how often a system cuts there.

## The test benchmark

The project's frozen test benchmark is built in the same way from EV9V's test recordings: 800 whole recordings, 240 pair streams and 120 multi streams. It has no examination streams. Each recording carries the number of its examination, because the intervals of the recording-level numbers are computed by drawing whole examinations. See [EVALUATION.md](EVALUATION.md) for how it is run and how to read its numbers.

## MIMIC-IV-Echo

MIMIC-IV-Echo is restricted-access data from PhysioNet. It was never used to train the model.

- **It entered two choices between trained systems.** Agreement with another classifier's view table on the recordings of 369 patients was the yardstick of the rule for the training recipe, and a step of the rule that chose the three-seed ensemble. [EVALUATION.md](EVALUATION.md), "How the system was chosen", has the rules.

- **Nothing of it is here.** No recording, frame, identifier or per-recording output of MIMIC-IV-Echo is in this repository or in the weights repository. Only aggregate numbers are reported.
- **Its view labels are not a reader's.** The view table used for comparison is another classifier's output. A comparison with it is agreement between two models.
- **Running the model on your copy.** `echoview mimic-table` reads the DICOM files of your own credentialed copy and writes a table in the format of the existing view table.

## Other datasets

EchoNet-Dynamic, EchoNet-LVH, EchoNet-Pediatric and CAMUS were read during development, for evaluation and never for training. One of them entered a choice: the recall of the four-chamber view on CAMUS was the last step of the rule that chose the three-seed ensemble ([EVALUATION.md](EVALUATION.md), "How the system was chosen"). No file of them and nothing derived from single recordings of them is here.
