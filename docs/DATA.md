# Data

## EV9V

EV9V is a public dataset of echocardiography recordings with view labels (`bgx666/EV9V` on the Hugging Face Hub, CC BY 4.0). Everything in this repository was trained and developed on it.

- **What this project uses.** 3,364 training recordings and 521 development recordings of five views: PLAX, PSAX, A4C, A5C and SC4C.
- **Labels.** EV9V has nine view codes. PLHLA counts as PLAX. The four parasternal short-axis codes (PASA, PMVLSA, PPMLSA and PMASA) count as PSAX. A4C, A5C and SC4C keep their names. Recordings with the ninth code, PMPALA, are not used.
- **Layout.** The dataset ships each recording as a folder of JPEG frames, `Images/<recording>/<frame number>.jpg`, at 30 frames per second and 320 × 240 pixels. This project reads every third frame, which gives 10 frames per second.
- **Download.** Get `Images` from the dataset's page on the Hugging Face Hub and unpack it. Every command of this package takes the path of that folder.

## The definition files in this package

Three kinds of file say which frames are read. All are in `src/echoview/data/`.

| File | What it defines |
|---|---|
| `ev9v_dev.json.gz` | the development benchmark: the sequences of both halves, fragment by fragment |
| `ev9v_recordings.json.gz` | the recordings of the training frame cache: split, label and length |
| `plans/groups_s*.npz`, `plans/windows_s*.npz` | the sampling plans: which windows the training updates draw from |

**No file names a recording.** A recording is named by a key: the first 16 hex digits of the SHA-256 of its EV9V folder name. The code hashes the folder names of your copy of EV9V and finds every recording that way (`echoview.evaluation.ev9v.recording_key`). If a recording is missing from your copy, the command stops and says how many are missing.

## Splits

- **Training.** 3,364 recordings. Only these are read by the training code.
- **Development, two halves.** 284 recordings in the `selection` half and 237 in the `calibration` half. The halves were split by recording session, so they share no session and no recording. The decoder setting is chosen on `calibration`. The development numbers are read from `selection`.
- **Test.** EV9V's test recordings are not named in any file of this package and are never read by it.

## The development benchmark

EV9V holds single-view recordings, so streams with view changes are built by joining fragments of recordings.

| Kind | What it is | Selection | Calibration |
|---|---|---|---|
| `native` | one whole recording | 284 | 237 |
| `pairs` | two fragments joined: the same view or two different views | 137 | 115 |
| `multi` | four to eight fragments joined | 120 | 120 |
| `exam` | every recording of one examination, in the order of acquisition | 36 | 36 |

A join of two fragments of the same view is a splice without a view change. The metric `same_class_cut_rate` counts how often a system cuts there.

## MIMIC-IV-Echo

MIMIC-IV-Echo is restricted-access data from PhysioNet. It was never used to train the model.

- **Nothing of it is here.** No recording, frame, identifier or per-recording output of MIMIC-IV-Echo is in this repository or in the weights repository. Only aggregate numbers are reported.
- **Its view labels are not a reader's.** The view table used for comparison is another classifier's output. A comparison with it is agreement between two models.
- **Running the model on your copy.** `echoview mimic-table` reads the DICOM files of your own credentialed copy and writes a table in the format of the existing view table.

## Other datasets

EchoNet-Dynamic, EchoNet-LVH, EchoNet-Pediatric and CAMUS were used for evaluation only during the project. No file of them and nothing derived from single recordings of them is here.
