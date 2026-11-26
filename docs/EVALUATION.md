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

## One read of the test benchmark for `classify`

**Status: planned on 2026-11-26 (UTC). Not read yet.** Everything in this section down to "The result" was written and committed before any of its numbers existed.

The section above says that `echoview classify` does not use the rule of the benchmark's published numbers, and that its own rule has no number on the test benchmark. This section prepares the one read that gives it one. On 2026-11-25 the owner of the project was asked whether the test benchmark may be read once more for `classify`'s rule, and said yes. This sentence was committed before the read.

### What is scored

What `classify` returns for a recording, and nothing else: the view with the largest mean window probability, the five probabilities, and whether the three seeds name the same view.

- **The function** is `echoview.pipeline.classification`. `Ensemble.classify` calls it on the window logits of the recording.
- **The logits** are the stored window logits of the published system for the 800 single recordings of the test benchmark: the file with SHA-256 `c5df435ef3022dd55f711c4efa72c74f19aad7f40193f179902c57dca97e288d`, which gave the published numbers. No model runs and no image is read.
- **No number about streams is reported.** `classify` is a command for one recording. The read decodes the streams only to hold the published numbers, as step 3 says.

### What is known now

- **The majority rule on these recordings.** 785 of the 800 are right, the macro-F1 is 0.956, and the confusion table is the one above. Without the decoder, with the majority of the likeliest views of the samples, 787 are right, with A4C read as A5C in 7 recordings and A5C read as A4C in 3 ([COMPARISON.md](COMPARISON.md), section 10).
- **`classify`'s rule on development data.** It has 2 and 1 wrong recordings on the two halves where the majority rule has 2 and 3. The calibration error is 0.030 and 0.034, and the median probability is 0.98.
- **No number by `classify`'s rule on the test benchmark** is in the project's records for the published system. Two records of single-recording numbers on these recordings were looked at for this plan, those of 2026-10-30 and of 2026-11-13. Both use the majority of the decoded samples.
- **The test benchmark has been read before,** see "The history of the test benchmark". Of the reads since the system was published, the one for the uncertain output scored a threshold on the probability of single samples of these recordings ([UNCERTAIN.md](UNCERTAIN.md), section 9). None scored the mean probability of a recording.

### What is run, in this order

1. **This plan and the code are public** before the read. The code is `echoview/evaluation/classify_read.py`. It is tested and reviewed before it runs on the test benchmark.
2. **A trial on development data.** `classify_read trial` runs the same functions on the stored logits of both development halves. The development report holds this rule's numbers for the published system: the wrong recordings, the confusion table, the calibration error and the median probability. They must come out: the counts exactly, and the two numbers made of probabilities within 0.00005, because logits of another run of the same models differ in the sixth decimal. The trial reads no file of the test benchmark. The read takes the trial's output and does not start without it.
3. **The read** is one run of `classify_read read`.
   - It is entered in the access record of the test benchmark before it starts.
   - From the stored logits it first computes the published system's rows. The four published numbers must come out. The rows "EchoRouter" and "Perfect on the 0.5 s grid" must equal those of the comparison's first read (`comparison/test.json`, SHA-256 `4f00000524df0581ee7f093980135a206c9180f51d15457524acb6547afe45c0`) number by number, intervals included. When one differs, nothing is written, and `classify`'s rule has been applied to no recording.
   - Then it scores what `classify` returns and writes one file.
4. **No second read.** Once the output exists, its digest goes into the code. From then on the command stops unless it is given that file and its own result equals it.
5. **When a run goes wrong.** A run that stops before it writes its file shows no number and is not the read. Starting it again is a decision, and the access record says so. An output that is later found to be wrong stays published with a correction beside it, and the benchmark is not read again.

### The numbers and the intervals

- **The view.** Accuracy, macro-F1, the recall of every view, the recordings of A4C read as A5C and of A5C read as A4C, and the confusion table. They are given for `classify`'s rule and, from the same run, for the majority rule. The code is the comparison's, so the majority rule's numbers are those already published.
- **The two rules against each other.** `classify`'s rule minus the majority rule for the accuracy, the macro-F1, the recall of A4C and of A5C, and the recordings of A4C read as A5C and of A5C read as A4C, with the same draws for both. The number of recordings that only one of the two rules reads right, with an exact two-sided sign test of that split.
- **The probability.** The probability of a recording is that of the view it is given.
  - The expected calibration error in ten bins of equal width, as in the development report, with every bin's number of recordings, mean probability and share that is right.
  - The same gaps over ten parts of equal count: the recordings in the order of their probability, 80 in each part. Recordings of one probability stay in the benchmark's order.
  - The mean probability minus the share that is right. Above zero the system is surer than it is right, below zero less sure.
  - The Brier score: the squared distance of the five probabilities from the reference view, summed over the views and averaged over the recordings. It runs from 0 to 2. And the log loss: the mean negative logarithm of the probability of the reference view, where a probability below 10⁻¹² counts as 10⁻¹².
  - The median probability of all recordings, of the right ones and of the wrong ones.
  - The separation: the chance that a recording that is read right has a higher probability than one that is read wrong, where a tie counts half. That is the area under the ROC curve of the probability taken as a sign of a right answer.
- **The seeds.** In how many recordings the three seeds do not name the same view, how many of those are wrong, and how many wrong recordings the seeds agree on.
- **Intervals.** Percentile intervals at 95% from 10,000 draws of whole examinations with seed 20261123.
  - The numbers about the view and the differences between the rules have the comparison's draws. A count of 5 or fewer has no interval.
  - The mean probability minus the share that is right, the Brier score, the log loss and the separation share draws of their own, from a generator started with the same seed.
  - A draw of examinations without a wrong recording, or without a right one, gives no separation, and the output says how many such draws there were.
  - The calibration errors, the bins, the medians and the seeds' counts have no interval. A calibration error cannot be negative, and a resample overstates it; the mean probability minus the share that is right has an interval in its place.
- **What the numbers rest on.** The output also holds the number of recordings and of wrong ones, how many examinations are drawn and how many of them hold a wrong recording, the mean probability, the share that is right, the edges of the bins and the right recordings in each, and for the majority rule the label changes inside a recording, a number of the comparison. It names the digests of its three input files and the checks that held.

### How the result will be worded

These rules are fixed now.

1. **One word, for the view.** On the test benchmark `classify`'s rule is called "more accurate than the majority rule" or "less accurate than the majority rule" when two things hold. The 95% interval of the paired difference in accuracy lies wholly on one side of zero, where an end at exactly zero is not on one side. And the exact sign test of the recordings that only one rule reads right gives p below 0.05. Otherwise it is "not shown to differ from the majority rule", and both values are given.
2. **Everything else is a description.** The calibration errors, the bins, the Brier score, the log loss, the separation and the seeds get no word such as "calibrated" or "reliable". The numbers are given with what they rest on.
3. **Nothing changes, whatever the numbers.** `classify` keeps its rule and `segment` keeps its own. No weight and no setting changes. The published numbers of the benchmark stay those of the majority rule, because every row of the comparison uses it. No threshold on the probability is chosen or proposed from this read.
4. **Where it goes.** The table of the section above gets its missing cell. The README and the model card say for `classify` what this read found, where they now give the majority rule's number only. The output goes to `docs/evaluation/classify_test.json` and the trial's to `docs/evaluation/classify_trial.json`, whatever they hold.
5. **One read.** The test benchmark is not read again for this question.

### What is expected, and the limits

- **Expected.** Between 11 and 16 wrong recordings of the 800, most likely 12 to 14: the rule is close to the majority of the likeliest views without the decoder, which has 13. The two rules should differ in five recordings at most, and the word should be "not shown to differ". The calibration error should be near the 0.03 of the development halves, with the system a little less sure than it is right. The separation should be well above 0.5, because most wrong recordings are A4C against A5C, where the probabilities are lower. With about 13 wrong recordings its interval will be wide.
- **Few wrong recordings.** Every number about wrong answers rests on about a dozen recordings, most of them A4C or A5C, and those recordings lie in fewer examinations than there are recordings. An interval that draws examinations is then uneven and may be too narrow.
- **The bins of equal width are mostly empty.** With a median probability near 0.98 nearly all recordings fall into the last bin. That calibration error then says little more than the gap between the mean probability and the share that is right. The parts of equal count and the two scores are there for that reason.
- **The same benchmark.** The limits of "The history of the test benchmark" hold for this read as for every other. The recordings are not untouched, and they come from one hospital and two machines of one maker. A probability may mean something else elsewhere.
- **The rule, on the benchmark's frames.** The read applies the function that `classify` calls to logits of the benchmark's frames. It does not run the command on the 800 folders, so the command's own reading of a file is not part of it.
- **Recordings of one examination are not independent.** The intervals draw whole examinations. The exact test counts every recording on its own.

**Reviewed before it was committed.** The code and this plan were reviewed on 2026-11-26, before the plan was public and before anything ran. The review found no error in a number. Four things changed with it, and they are in the text above.

- The word needs the exact test beside the interval. With the interval alone, four recordings against none would have got a word at p = 0.125.
- The probability gets the parts of equal count, the mean probability minus the share that is right, the Brier score and the log loss. They are fixed here, and not after the read.
- The read takes the trial's output.
- A run that stops before it writes is not the read.

### The result

Not read yet.

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
