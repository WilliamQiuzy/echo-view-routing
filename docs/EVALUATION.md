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
- **This is not an untouched test set.** Earlier rounds of the project were evaluated on this benchmark: its streams are the stream banks of the project's early development, and the four other approaches were scored on them then. The numbers of single seeds on it were known when the ensemble was chosen. The ensemble itself was chosen by a rule that reads development data only: the EV9V development halves, MIMIC-IV-Echo recordings of development patients with another classifier's labels, and CAMUS. No number of the test benchmark is in that rule. Read the numbers as a description of the system. "How the system was chosen" and "The history of the test benchmark" below give the details.
- **The report of that run** is [`evaluation/test_report.json`](evaluation/test_report.json), as the evaluation of 2026-11-20 returned it. That evaluation was run in pieces, with the functions that the command calls (see below). The file holds pooled numbers only.
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
- **One evaluation.** Since the benchmark was frozen on 2026-10-23, the project keeps a record of the reads of its test lists, frames and labels. This run is in it. Nothing was tuned on its output and it was not repeated.
- **The intervals** are also tested without a GPU: `tests/test_evaluation_test_benchmark.py` feeds made-up counts to the bootstrap and requires the intervals the research code's bootstrap gave them.

## Single recordings: two rules, the confusion table and the model's confidence

**Two rules give a recording its view.** The benchmark's numbers use the majority of the decoded sample labels. That is what `echoview segment` returns for one recording. `echoview classify` takes the largest mean window probability instead.

| | Recordings | Wrong by the majority of the decoded samples | Wrong by the mean probability (`classify`) |
|---|---|---|---|
| Development, calibration half | 237 | 2 | 2 |
| Development, selection half | 284 | 3 | 1 |
| Test benchmark | 800 | 15 | not measured |

On the selection half `classify`'s rule has two wrong recordings fewer. For that rule the test benchmark was not scored; that would be one more read of it.

**The 800 single recordings of the test benchmark,** by the majority rule ([`evaluation/test_report.json`](evaluation/test_report.json)). A row is the reference view, a column the view that was given.

| | PLAX | PSAX | A4C | A5C | SC4C | Right |
|---|---|---|---|---|---|---|
| PLAX | 275 | 0 | 0 | 0 | 0 | 100% |
| PSAX | 0 | 219 | 0 | 0 | 0 | 100% |
| A4C | 0 | 0 | 219 | 10 | 2 | 94.8% |
| A5C | 0 | 0 | 2 | 38 | 0 | 95.0% |
| SC4C | 0 | 0 | 0 | 1 | 34 | 97.1% |

Twelve of the fifteen errors are between A4C and A5C. A5C has 40 recordings and SC4C has 35, so one recording moves their rows by 2.5 and 2.9 points.

**Confidence.** The development report ([`evaluation/development_report.json`](evaluation/development_report.json), as `echoview evaluate` wrote it for the published weights) holds an expected calibration error for `classify`'s rule. The recordings are put into ten bins of equal width by the probability of the view they are given. In each bin the mean probability is set against the share that is right, and the gaps are averaged with the bins' sizes as weights.

- It is 0.030 on the calibration half and 0.034 on the selection half.
- The median probability is 0.98 on both halves, and 99.2% and 99.6% of the recordings are right. The model is a little less sure than it is right.
- Three recordings are wrong in all. That is too few to say whether a low probability marks a wrong answer.
- Calibration was not measured on the test benchmark, and not on recordings of another hospital or machine, where a probability may mean something else.
- A recording of a view that the model does not know still gets one of the five names, often with a high probability.

**Streams in examination order.** The development benchmark also joins all recordings of one examination in their order (36 streams in each half, see [DATA.md](DATA.md)). The published system's segment F1 at 50% is 0.946 on the calibration half and 0.959 on the selection half, and 3.5% and 1.9% of the samples get a wrong view. These streams are joins of whole recordings: how the probe moves from one view to the next was not recorded. The test benchmark has no such streams.

## How the system was chosen

These are the last three choices that led to the published system. The family of the window model and its first two training stages were chosen in earlier rounds: on the development halves and, before 2026-10-02, with numbers of the test split in view (next section). None of the three choices below read the test benchmark.

- **The training recipe of stage 3** (2026-11-12). Eight candidate systems were trained or put together.
  - A candidate was eligible when its mean segment F1 at 50% on the multi streams of both EV9V development halves was at most 0.005 below that of the stage-2 models, and it misread at most two more single recordings per half.
  - Among the eligible ones, the highest agreement with another classifier's view table on MIMIC-IV-Echo development recordings was to win: 3,105 recordings of 369 patients.
  - One candidate was eligible. It is the published recipe.
- **Three seeds as one ensemble** (2026-11-13). A rule that was committed beforehand compared eight candidates, single models and ensembles.
  - Its steps were: eligibility on the EV9V development halves as above, then segment F1 on MIMIC development streams, then agreement on MIMIC development recordings, then the recall of the four-chamber view on CAMUS.
  - The ensemble of the three seeds and one other candidate were tied until the last step: 98.0% against 97.2% on CAMUS, with an A5C answer counted as four-chamber.
  - The project's own report calls that last step inside the noise: reruns of the ensemble gave 97.2% to 98.8% there. The rule's preference between the two is not a robust ranking.
  - The owner of the project then decided for the ensemble.
- **The decoder setting** is the one that the rule above gives on the calibration half. The round's study of decoder settings also read MIMIC development streams, and it kept this setting.

What was known then and is not in the rules:

- **The single seeds on the test benchmark.** The stage-2 seeds and the stage-3 seeds had been scored on it earlier on 2026-11-13: 0.916 and 0.912 on the multi streams, as means of three seeds. The ensemble's own row, 0.909, was computed after the decision. It is 0.003 below the mean of its three members and above one of them, and stage 3 is 0.004 below stage 2. No interval of these differences excludes zero.
- **The ensemble on further MIMIC patients.** Its agreement on the recordings of 866 patients outside the development set had been read before the choice. Those patients were not in the rule.
- **What MIMIC agreement rewards.** The reference there is another classifier's output, with classes for three of the five views. A recipe that is chosen for agreement with it is chosen to agree with that classifier.

## The history of the test benchmark

The test recordings are EV9V's test split. This table holds the uses of them that the project's records show.

| When | What was read | Why | A choice afterwards? |
|---|---|---|---|
| 2026-09-13 to 2026-10-01 | Scores on the test split for a ladder of baselines, STFM, EchoViewCLIP, EchoPrime's classifier, MS-TCN and ASFormer, and for a family of vision transformers that includes the MViTv2-S started from EchoPrime's encoder. Streams built from test recordings, scored with thresholds and fusions; the pair and multi streams of the later benchmark are these streams. Single A5C test recordings, looked at case by case | early development, before the split was sealed | Yes. Rankings of model families, thresholds and that error analysis informed development, and the published model descends from that MViTv2-S. The runs are in the project's experiment ledger, but no access record was kept then, so single reads cannot be counted |
| 2026-10-02 | The split is sealed. Development moves to the two halves of the validation split | | |
| 2026-10-17 | Nothing. A written check asked whether the split could confirm a result | | It was judged not eligible, because of the uses above |
| 2026-10-23 | The test benchmark is frozen as it stands: the 800 recordings and the stream banks of early development. Two stage-2 seeds and one other system of the project are run on it. The four other approaches are scored from their stored outputs, and their rows equal those of early development. Ten of its sequences were put on a preview page of the project, all of them picked afterwards by the form of their errors. For the check of that page, three of the approaches were run again on two sequences | a first benchmark of frozen systems | No tuning. The stage-2 model stayed the working reference |
| 2026-10-30 | The three stage-2 seeds on the 800 recordings, and STFM on all 888. The design of the pair and multi streams was counted again from the test lists and labels | a comparison on STFM's own test | None |
| 2026-11-03 and 2026-11-04 | By mistake, two label values and one pooled confusion table of the test split were printed during other work. Lists that hold test rows were read into memory and dropped | | None. Both are recorded as slips |
| 2026-11-13 | The three stage-2 seeds and the three stage-3 seeds. The forward passes were made twice, because the first ones used another arithmetic mode and failed their reproduction check. Later that day the ensemble, for which the six single models were run a third time | whether stage 3 costs anything on EV9V; then the ensemble's row | The recipe had been chosen before, on development data. The ensemble was decided before its row was computed, with the single seeds' numbers known |
| 2026-11-20 | The benchmark's definition is exported for this package: its lists and reference views, no model output. The published weights are run with this package, in three starts of which two were cut off before any output. Then one read with the uncertain rule | the published numbers; UNCERTAIN.md, section 9 | None |
| 2026-11-24 | Two reads of stored outputs | COMPARISON.md, sections 10 and 11 | None. The second read was decided with the first result known |
| 2026-11-24 and 2026-11-25 | Forward passes without labels, then one read | FOLLOWUP.md, section 9 | None. The option stayed off by default |

- **The access records.** The read of 2026-10-23 and every read since 2026-11-20 are in an access record on the project's server. The runs of 2026-11-13 were made on a rented machine and have records of their own. The records are kept with the data and are not public, because they name files of those machines.
- **What this means for the numbers.** They describe the published system on recordings that its development had seen in aggregate many times before the split was sealed. They are likely to be optimistic for new recordings, by an amount that these data cannot give. The project has no result against readers' labels on recordings that no step of its development has touched.

## Two sets of intervals

The same number can carry two intervals in these documents. Both are percentile intervals from a bootstrap with a fixed seed.

- **`echoview evaluate --benchmark test`** draws 2,000 times: streams for the numbers about streams, examinations for the numbers about recordings. For segment F1 on the multi streams it gives 0.883 to 0.934.
- **The comparison and the follow-up** draw 10,000 times and take every number about streams twice: over streams (0.883 to 0.933) and over the 125 examinations (0.872 to 0.944). They report the wider one. The streams reuse recordings, so an interval over streams is too narrow for anything that depends on few recordings.

Where one interval stands beside the 0.909, the wider one is the safer one to read. Streams and examinations cross, so neither unit holds all the variation ([FOLLOWUP.md](FOLLOWUP.md), section 9). Every interval is for these recordings and this way of building streams. None says how the system does at another hospital.

## Speed

`python tools/measure_speed.py` times `Ensemble.segment` for the published read-out and for the two denser ones.

- **The stream** is the five sample clips of the self-check, joined and repeated to the length that is asked for. Every window costs the same forward pass whatever it shows.
- **What is timed** is the window model of the three seeds on every window, and the decoder, on frames that are already decoded. Reading and decoding the files is not timed. It is the same work for every step.
- **How.** Every step runs once before it is timed. The number is the median of the timed runs, with the fastest and the slowest beside it.

| Machine | Stream | Read-out | Windows | Seconds, median (fastest to slowest) | Times as fast as the stream plays | Times the published step |
|---|---|---|---|---|---|---|
| NVIDIA L40S, 8 windows per forward pass, 5 runs | 60 s | every 0.5 s (published) | 120 | 4.75 (4.75 to 4.77) | 12.6 | 1.0 |
| | | every 0.2 s | 300 | 11.95 (11.92 to 11.96) | 5.0 | 2.5 |
| | | every 0.1 s | 600 | 24.15 (24.02 to 24.20) | 2.5 | 5.1 |

- **Memory.** On the L40S PyTorch held at most 2.6 GiB of GPU memory for the 60 s stream, with its frames on the GPU. That is PyTorch's own count of what it allocated; `nvidia-smi` shows more.
- **A larger batch is no faster.** With 32 windows per forward pass the three steps take 5.01, 12.56 and 24.99 s and hold 8.9 GiB.
- **The order of the steps.** The three steps were timed one after the other, in the order of the table, with PyTorch's default settings, as `echoview segment` runs. A window cost 39.6, 39.8 and 40.3 ms, so the last ratio is 5.1 where the windows alone give 5.0.
- **Windows.** A long stream has 2.5 and 5 times the windows of the published step. The 1,160 sequences of the test benchmark are short and have 2.4 and 4.9 times as many.
- **One machine, one measurement,** made on 2026-11-25 with PyTorch 2.14. The files are in `docs/speed/`. A laptop was not measured in this way; the README gives the times of the self-check on one.

## What others can repeat

With a copy of EV9V and a GPU:

- `echoview evaluate` gives the development numbers, and `echoview evaluate --benchmark test` the four numbers of the test benchmark with their intervals. Both were checked against the research code, as said above.
- `python -m echoview.evaluation.uncertain_study choose` and `report` repeat the study of the uncertain output on development data, and `confirm` repeats its read of the test benchmark from logits of one's own, as long as they give the published numbers.
- `python -m echoview.evaluation.followup logits` and `study` repeat the development study of FOLLOWUP.md. Its third question also needs EchoViewCLIP's outputs, which this package cannot produce.
- Training is in [TRAINING.md](TRAINING.md).

Without any data, the tables of UNCERTAIN.md, COMPARISON.md and FOLLOWUP.md can be computed again from the files in `docs/uncertain/`, `docs/comparison/` and `docs/followup/`. The tests of this repository do that.

What others cannot repeat as it stands:

- **The reads of the test benchmark for the comparison and for the denser read-out** (`comparison score --benchmark test` and `followup confirm`). These commands take only files of model outputs whose SHA-256 the code names. The files hold outputs for single recordings and are not published, and outputs that are computed again on another machine differ in the sixth decimal. The lock keeps the project from reading the benchmark again through these commands. It is on the commands, not on the functions behind them: `echoview.evaluation.followup.confirm` scores outputs of one's own.
- **The four other approaches.** Their outputs come from the project's research code. This package holds neither their code nor their weights. [COMPARISON.md](COMPARISON.md), section 12, says where each came from.
- **The MIMIC-IV-Echo numbers,** see "Not here yet".

## The uncertain output

`echoview segment --uncertain` lets the system leave weak samples unnamed. Its rule came out of a study on the development benchmark whose plan was committed before any result: [UNCERTAIN.md](UNCERTAIN.md). The rule was chosen on the calibration half and checked on the selection half. That study did not read the test benchmark; one later read of it is described below.

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

[FOLLOWUP.md](FOLLOWUP.md) asks on development data only whether three changes that need no training are worth confirming: a denser read-out, an offset between A4C and A5C, and EchoViewCLIP's probabilities mixed in. `python -m echoview.evaluation.followup` is the command, and its output is in `docs/followup/`. Its section 9 is the one read of the test benchmark for the denser read-out, made on 2026-11-25: both steps have nearer cuts at a cost. `followup logits --test` and `followup confirm` are its commands, its output is `docs/followup/confirmation.json`, and `confirm` can now only reproduce that file.

## Not here yet

The MIMIC-IV-Echo evaluations of the model card were produced by the research code. They are not part of this package: MIMIC-IV-Echo is restricted-access data, and its reference there is another classifier's table.
