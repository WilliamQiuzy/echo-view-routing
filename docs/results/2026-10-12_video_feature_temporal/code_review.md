# Implementation review of the video feature temporal round

An independent reviewer read commit 74e135d and its parents back to the task sheet commit 19092dd before the training
checks. The reviewer compared the code with the task sheet
(`docs/superpowers/plans/2026-10-12-video-feature-temporal-training.md`) and the fixed configuration
(`configs/experiments/video_feature_temporal_v1.yaml`). The reviewer ran the tests and read the running preparation's log,
plans and initial states on the server without disturbing the job. No file was edited by the reviewer.

## Verdict

The review passed the code, with no critical finding. The running feature extraction did not need to stop or be repeated.

## Findings and what was done

1. **High: the parity gate came after saving.** The preparation saved the training features first and checked the source
   parity afterwards. If parity had failed, a rerun under the same feature identity would have reused the saved files
   without checking them again. 32bf8af moves the gate before saving. The feature-building functions are unchanged, so
   the running extraction keeps its feature identity, and its own parity result stays a real check.
2. **Medium: the server's configuration was older.** The server's copy of the configuration lacked the reading
   conditions of 14e359c. Only the decision section differed, and no code that reads it had run on the server. The
   configuration was synchronised with the training and evaluation code before the checks.
3. **Low: the queue takes the official commits from the configuration.** The training entry still checks the pinned
   commit of each official repository before every run, so a drift would stop a run. No change was made.
4. **Low: two unused names in the feature tests.** 32bf8af removes them.

## What the reviewer confirmed

- **Assembly before encoding.** Constructed streams take their per-sample references from the recipe, checked against
  `windowed.stream_plan`, before any window is taken. No recording is encoded alone and spliced afterwards. Native crops
  are slices of the whole recording's features.
- **Float32 throughout.** The batch adapter refuses features that are not float32, and no fp16 path of the old helpers is
  used in training or in the development evaluation.
- **No mixed coordinates.** Each seed's heads read only that seed's encoder features.
- **Label isolation.**
  - Padding labels become the ignore value and padding features 0 before every loss.
  - Inference outputs do not change when labels, joins and ids do.
  - The index read parses labels for training rows only.
- **Batches and weights.**
  - On the server, all 18,750 updates of every seed equal the paired ASRF plans.
  - The class weights and pos_weight are equal too.
  - The nine initial states exist.
- **Budget.** Every job writes its GPU hours to the ledger, also on failure. The queue projects the budget before every
  run.
- **Isolation.** No code reads the test split or EchoNet-Pediatric, and every write stays in this round's directories.
- **Readings.** The decision code matches the configuration's conditions.
