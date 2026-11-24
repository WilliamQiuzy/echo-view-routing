# Evaluation

Two evaluations run on EV9V: the development evaluation, on which the decoder setting is chosen, and the frozen test benchmark, on which nothing is chosen.

## Run it

```bash
pip install -e ".[evaluation]"
echoview evaluate --images /data/EV9V/Images --out report.json --cache /data/echoview_cache
```

The command needs the EV9V `Images` folder (see [DATA.md](DATA.md)) and a GPU. It decodes about 40,000 frames, runs the three seeds over about 21,000 windows and writes the full report. `--cache` keeps the decoded frames (2 GB) for the next run. On one L40S the first run takes about 18 minutes.

It prints one line per system:

```text
C2_s0                        selection F1@50 0.9584  calibration F1@50 0.9856  recording errors 1/1  decoder h7 thr 0.1860 (feasible)
C2_s1                        selection F1@50 0.9476  calibration F1@50 0.9827  recording errors 1/1  decoder h5 thr 0.1297 (feasible)
C2_s2                        selection F1@50 0.9483  calibration F1@50 0.9778  recording errors 1/2  decoder h5 thr 0.1371 (feasible)
ensemble                     selection F1@50 0.9594  calibration F1@50 0.9837  recording errors 1/2  decoder h7 thr 0.1738 (feasible)
ensemble_published_setting   selection F1@50 0.9594  calibration F1@50 0.9837  recording errors 1/2  decoder h7 thr 0.1738 (fixed)
```

To evaluate seeds you trained yourself, put them in a folder as `C2_s0/final.pt`, `C2_s1/final.pt` and `C2_s2/final.pt`, and add `--weights <folder> --unverified`.

## What is evaluated

- **Each seed alone.** Its decoder setting is chosen on the calibration half by the rule below and applied unchanged to the selection half.
- **The ensemble.** The three seeds' probabilities are averaged. The setting is chosen in the same way.
- **The ensemble with its published setting.** The setting that ships with the model. It gives the same numbers as the rule's choice.

The evaluation uses plain float32 arithmetic on the GPU. TF32 is switched off, because it changes logits in the fourth decimal.

## Metrics

**Recordings.** A recording's view is the majority of its decoded sample labels.

- `native_macro_f1` is the mean F1 over the five views of that decision.
- `native_internal_cuts_per_clip` counts label changes inside one recording. There should be none.
- The `native` block of the report gives the recording-level decision from the mean window probability, its confusion table and its calibration error.

**Streams.**

- `segment_f1_50_multi`. Predicted and reference segments of the same view are matched one to one when their temporal intersection over union is at least 0.5, best match first. F1 is 2 × matched / (predicted + reference), pooled over all streams. This is the headline number.
- `boundary_recall_multi`, `boundary_precision_multi`, `boundary_f1_multi`. Predicted label changes are matched one to one to reference view changes at most 0.5 s away, nearest first.
- `same_class_cut_rate_multi`. Of the joins where two fragments of the same view meet, the share with a predicted change within 0.5 s. A model that cuts at every splice instead of at view changes scores high here.

**Where the errors are.** The blocks `streams_multi` and `streams_exam` describe the decoded streams further (`echoview/evaluation/scorecard.py`).

- `A1_piece`. Is the middle sample of every source recording given its view?
- `A1_segment`. Does the middle sample of every returned segment carry the segment's view?
- `wrong_split`. Every wrong sample falls in one of four classes: `edge` (the cut is a little off), `spurious` (an extra segment), `absorbed` (a piece swallowed by its neighbour) and `misread` (a whole piece given another view).
- `A4_cuts`. The signed time offsets of matched cuts.
- `A5_purity`, `A5_recovery`. Per view: the share of returned samples that are right, and the share of reference samples that are returned.
- `A6_f1`, `A6_edit`. Segment F1 at overlaps 0.10, 0.25 and 0.50 and the Edit score, as defined by MS-TCN.

## The decoder and its rule

The decoder scores every sample by the Jensen-Shannon divergence between the mean probabilities of the `half_window` samples before it and after it. A cut is a local peak of the score at or above the threshold at which the two sides also name different views.

The setting is chosen on the calibration half:

1. The grid is five half windows (5, 7, 9, 11, 15 samples) by eight thresholds. A threshold is a quantile (0.80 to 0.999) of the scores inside the single-view recordings of the calibration half.
2. A setting is feasible when the recording-level macro-F1 stays at least 0.980 and the boundary recall on multi streams at least 0.95.
3. Among the feasible settings the highest segment F1@50 on multi streams wins. An equal score goes to the earlier grid position.
4. When nothing is feasible, the report says so and shows the numbers at a fixed diagnostic setting.

## The test benchmark

```bash
echoview evaluate --benchmark test --images /data/EV9V/Images --out test_report.json
```

The project's frozen test benchmark is built from EV9V's test split. It holds 800 single recordings from 125 examinations, 240 streams of two fragments and 120 streams of four to eight fragments. None of its recordings is in the training or the development data. The definition file is `src/echoview/data/ev9v_test.json.gz`.

Nothing is chosen on it. The loaded system is decoded once with the setting that ships with it, and the command prints three lines:

```text
multi streams   segment F1@50 0.9089 (95% interval 0.883 to 0.934)
pair streams    segment F1@50 0.9847 (95% interval 0.972 to 0.996)
recordings      accuracy 0.9812 (95% interval 0.972 to 0.990)  macro-F1 0.9561 (95% interval 0.932 to 0.976)
```

- **Metrics.** The same as in the development evaluation: segment F1 at 50% overlap pooled over the streams, and the recording-level decision by the majority of the decoded sample labels.
- **Intervals.** A percentile bootstrap with 2,000 draws and a fixed seed (`echoview/evaluation/bootstrap.py`). Whole units are drawn again: examinations for the recording-level numbers, streams for segment F1.
- **This is not an untouched test set.** Earlier rounds of the project were evaluated on this benchmark, and the numbers of single seeds on it were known when the ensemble was chosen. The ensemble itself was chosen by a rule that reads development data only. Read the numbers as a description of the system.
- **The run needs** the EV9V `Images` folder and a GPU. It decodes about 55,000 frames and runs the three seeds over about 15,000 windows.

## Checked against the research code

On 2026-11-17 this evaluation was run beside the research code on the same frozen weights, on one L40S.

| Check | Result |
|---|---|
| Window logits, 6 comparisons of about 9,900 to 11,000 windows each | largest difference 5.2e-6, no window changed its view |
| Decoder choice of every seed and of the ensemble | identical |
| Every decoded metric, both halves, 3 seeds and the ensemble | identical |
| The ensemble with its published setting against the rule's choice | identical numbers |

One difference is known. The published threshold of the ensemble, 0.17379, was computed on another GPU with TF32 and compiled code. Plain float32 gives 0.17383. Decoding with either gives the same numbers on both halves.

The tests of this repository repeat the comparison without a GPU: `tests/test_evaluation_parity.py` feeds stored predictions for the whole benchmark to the metrics, the scorecard and the decoder rule, and requires the research code's stored outputs.

**The test benchmark.** On 2026-11-20 the test evaluation was run on one L40S with the published weights. The research code had stored this system's row on 2026-11-13. Every number of the new run equals it to the last digit, with both ends of its interval:

| Number | This package and the research code |
|---|---|
| Segment F1@50, multi streams | 0.908908908908909 (0.8833483675956257 to 0.9335977211135994) |
| Segment F1@50, pair streams | 0.9847009735744089 (0.9724487962630256 to 0.9957449817930121) |
| Accuracy, recordings | 0.98125 (0.9717355423890712 to 0.9899333074769822) |
| Macro-F1, recordings | 0.9561411223630938 (0.9324449865632222 to 0.9757512258723523) |

- **How it was run.** The machine was a preemptible one that was reclaimed every few minutes that night. The evaluation was therefore run with the functions the command calls (`Ensemble.window_logits` for every sequence, then `run.evaluate_test_benchmark`), in 24 pieces that are written to disk as they finish. Nothing was scored before all pieces existed. Two earlier starts were cut off before any model output existed.
- **One evaluation.** The project keeps a record of every read of its test lists, frames and labels. This run is in it. Nothing was tuned on its output and it was not repeated.
- **The intervals** are also tested without a GPU: `tests/test_evaluation_test_benchmark.py` feeds made-up counts to the bootstrap and requires the intervals the research code's bootstrap gave them.

## The uncertain output

`echoview segment --uncertain` lets the system leave weak samples unnamed. Its rule came out of a study on the development benchmark whose plan was committed before any result: [UNCERTAIN.md](UNCERTAIN.md). The rule was chosen on the calibration half and checked on the selection half. The test benchmark was not read for it.

```bash
echoview evaluate --images /data/EV9V/Images --save-logits dev_logits.npz --batch 32
python -m echoview.evaluation.uncertain_study choose --logits dev_logits.npz --out choice.json
python -m echoview.evaluation.uncertain_study report --logits dev_logits.npz --choice choice.json --out report.json
```

- **What the commands do.** The first keeps the window logits of every development sequence. `choose` reads the calibration half only and writes the band, the counts of all 300 settings and the candidate. `report` reads the selection half only and writes the two-part report of UNCERTAIN.md, section 5, with the decision. Both run on the CPU. When the study ran, each took about half a minute.
- **A later read of the test benchmark** with the released rule is planned and reported in UNCERTAIN.md, section 9. Its command is `python -m echoview.evaluation.uncertain_study confirm --logits test_logits.npz --out test_report.json`. The file holds the window logits of exactly the benchmark's sequences, in the format of `echoview evaluate --benchmark test --save-logits`, and the command stops unless they give the benchmark's published numbers.
- **The files of the study** are in `docs/uncertain/`: `choice.json` and `report.json`, as the commands wrote them. They hold pooled counts only.
- **The measures** are in `echoview.evaluation.selective`. They score the labelled part and the uncertain part separately, and they put every number that can only improve when more is declined beside what it cost.

## The comparison with other approaches

[COMPARISON.md](COMPARISON.md) holds a fuller comparison on the test benchmark: the numbers of this page and some more for every system, and rows that give the other systems the project's decoder. Its plan was committed before the read, and its result follows the plan. `python -m echoview.evaluation.comparison` is the command, and its outputs are in `docs/comparison/`. Its section 11 is an addendum that answered the comparisons with STFM and ASFormer by a second read for two rows.

[FOLLOWUP.md](FOLLOWUP.md) asks on development data only whether three changes that need no training are worth confirming: a denser read-out, an offset between A4C and A5C, and EchoViewCLIP's probabilities mixed in. `python -m echoview.evaluation.followup` is the command, and its output is in `docs/followup/`. Its section 9 plans one read of the test benchmark that would confirm the denser read-out; `followup logits --test` and `followup confirm` are the commands for it.

## Not here yet

The MIMIC-IV-Echo evaluations of the model card were produced by the research code. They are not part of this package: MIMIC-IV-Echo is restricted-access data, and its reference there is another classifier's table.
