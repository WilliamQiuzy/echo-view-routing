# What the parts add, and what the system stands: studies on development data

**Status: planned on 2026-11-26 (UTC). Part 1 was run on the same day and its text corrected once (section 9); part 2 has run and its results are not written yet; part 3 was planned after part 1 was read (section 10) and is not run yet.** Everything in this document down to "Results" was written and committed before any of its numbers existed, and section 10 before any number of part 3.

Nothing here reads the test benchmark, trains a model or changes the published system.

## 1. Why

On 2026-11-26 the public documents were read as a reviewer of a journal would read them, and a list was made of the experiments that such a reviewer would miss. This document takes the ones that need no new data, no training and no read of the test benchmark:

- what each seed adds to the ensemble;
- whether a simpler decoder would do what the change-point decoder does, and how much the decoder depends on its settings;
- whether one number would repair the probabilities, which say less than the system knows ([EVALUATION.md](EVALUATION.md), "One read of the test benchmark for `classify`");
- how the system stands changed images and changed timing, and what stage 3 of the training adds there;
- what the system does with a view it does not know.

The list also holds things that this document cannot give: recordings of other hospitals read by experts, continuous examinations with reference cuts, several training runs of the other approaches, the same recipe from another starting point. Section 8 names them.

## 2. What was known when this was written

- **The ensemble on development data.** The seeds alone reach 0.958, 0.948 and 0.948 on the multi streams of the selection half, and the ensemble 0.959 ([EVALUATION.md](EVALUATION.md)). The project's research records also hold the three pairs of seeds: 0.958, 0.957 and 0.942.
- **The decoder.** On the test benchmark it adds 0.035 to EchoRouter's own probabilities ([COMPARISON.md](COMPARISON.md), section 10). The research records hold a study of 240 settings for the stage-3 models. No other way of smoothing was ever tried on the published system's outputs.
- **The probabilities.** On development data and on the test benchmark the system is less sure than it is right, by 0.03 and 0.025.
- **Changed images.** The research records hold one test of single seeds on the 521 development recordings under 14 changes of grey levels, blur and zoom. Stage-2 and stage-3 seeds both read 99.0% to 99.8% of the recordings right under every change, and the records' own reading is that these changes do not separate the systems. No stream was scored there. Noise, compression, a lower resolution, a mirrored or a rotated image, and any change of timing were not tried on the published models.
- **A view the system does not know.** For the 46 recordings of EV9V's ninth view (PMPALA) in the validation split, the research records give an AUROC of 0.575 (0.485 to 0.642) for the published system's largest probability against the 521 recordings of the five views. An earlier model read all 46 as PSAX.

## 3. Part 1: from stored outputs

The input is the stored window logits of the published system for both development halves, the file that gives the published development numbers. No model runs. `python -m echoview.evaluation.ablations` is the command.

**Every choice is made on the calibration half and reported on the selection half.**

### The size of the ensemble

Every subset of the three seeds is a system of its own: three single seeds, three pairs, and the ensemble. Each gets the decoder setting that the project's rule gives it on the calibration half ([EVALUATION.md](EVALUATION.md), "The decoder and its rule"). Each is given minus the ensemble with the same draws. For `classify`'s rule the wrong recordings and the calibration error are given too. This is a description and gets no word.

### Other decoders

The ensemble's per-sample probabilities are fixed. Five other ways turn them into a view for every sample:

| Name | What it does | Its one number, and the values tried |
|---|---|---|
| no decoder | the likeliest view of every sample | none |
| mode filter | a sample takes the most frequent likeliest view among the samples around it; on a tie it keeps its own view when that is among the most frequent | the samples in that window: 3, 5, 9, 13, 17, 21, 25, 31 |
| mean filter | the likeliest view of the probabilities summed over the samples around a sample | the samples in that window: 3, 5, 9, 13, 17, 21, 25, 31 |
| minimum duration | the likeliest view of every sample, and then a piece that is too short takes the view of its longer neighbour, the shortest piece first | the samples a piece must last: 3, 5, 8, 10, 15, 20, 30 |
| Viterbi path | the path of views with the largest sum of log probabilities, where every change of view costs a fixed amount | that cost: 0.5, 1, 2, 4, 8, 16, 32, 64 |

- **How the number is chosen.** By the project's own rule, on the calibration half: among the values that keep the recording-level macro-F1 at 0.980 and the boundary recall on multi streams at 0.95, the one with the highest segment F1 at 50% on the multi streams, the earlier value on a tie. When no value keeps the two floors they are dropped for that decoder, and the table says so.
- **The word.** Each of the five is compared with the released decoder at its published setting, on the selection half, by segment F1 at 50% on the multi streams. The released decoder is "ahead of" another one when the interval of the difference lies wholly above zero, "behind" it when wholly below, and otherwise "not shown to differ from" it. Five comparisons are made, so this interval is taken at 99%. Every other number is a description with a 95% interval.
- **The decoder without one of its parts.** The released decoder makes a cut only where the two sides of a peak name different views. One more row drops that test: every peak of the score is a cut. Its setting is chosen by the rule over the decoder's own 40 settings. The row is described and gets no word.

### The decoder's settings

The released decoder is run at all 40 settings of its grid: five half windows by eight thresholds, the thresholds taken from the calibration half as the rule takes them. For every setting the three numbers of the rule are given on both halves. The summary says how many settings come within 0.01 and within 0.02 of the published setting on the selection half, above or below it, how many lie above it, and gives the lowest and the highest value among the settings that keep the floors.

**The halves exchanged.** The rule is also applied the other way round: it chooses on the selection half, and the calibration half reports. This is beside the published numbers and replaces none.

### One temperature

Every logit of the three seeds is divided by one number before the probabilities are formed. A number below 1 makes the system surer.

- **Fitted** on the single recordings of the calibration half: of the values 0.30, 0.35, and so on up to 2.00, the one with the lowest log loss of `classify`'s five probabilities, and of equal ones the nearest to 1.
- **Reported** on the single recordings of the selection half, before and after: wrong recordings, the two calibration errors, the mean probability minus the share that is right, the Brier score and the log loss, as in the read for `classify`.
- **The word.** The temperature is a "candidate" when it is not 1, the log loss on the selection half is lower with a 95% interval of the paired difference wholly below zero, and no more recordings are wrong. Otherwise it is "not supported on development data".
- **What it leaves out.** The decoder reads the same probabilities, and its setting was chosen for them as they are. Streams are not scored with a temperature here.

## 4. Part 2: forward passes on development data

The published models are run on the selection half: 284 single recordings, 137 pair streams and 120 multi streams. `python -m echoview.evaluation.stress` is the command.

### The conditions

Every sequence is shown once as it is and once under each condition. The decoder keeps its published setting. Nothing is chosen.

| Kind | Conditions |
|---|---|
| Grey levels | gamma 0.5, 0.7, 1.5 and 2.0; 40 grey levels darker and 40 brighter; contrast 0.5 and 1.5 about the middle grey |
| Degradation | Gaussian noise with a spread of 10 and of 25 grey levels; speckle (every pixel times a number with a spread of 0.25 about 1); Gaussian blur of 1 and of 2 pixels; JPEG at quality 30 and 10; the frame made 112 and 64 pixels wide and brought back to 224 |
| Geometry | mirrored left to right; the middle of the frame enlarged 1.2 times; the frame made 0.8 times its size and set in black; rotated by 10 and by 20 degrees about the middle |
| Time | still frame: every window shows its centre frame sixteen times; shuffled frames: every window shows its own sixteen frames in a random order; played backwards; half the frame rate: five frames a second, each shown twice |

- An image condition changes every frame of a sequence in the same way, after the frame has been cropped and resized as always. Noise is drawn anew for every frame, from a generator that is fixed for the condition and the sequence.
- At half the frame rate a sample shows the frame of the even sample before it, so a cut can lie 0.1 s late by construction. With shuffled frames the marker of the window stays with the centre frame.
- Stage 3 trained with a zoom of 0.74 to 1.35, rotations of up to 12 degrees, a gamma of 0.61 to 1.65, additive noise with a spread of up to 5 grey levels and speckle of up to 0.15, among other changes, and no stage mirrored a frame. Several conditions lie outside these ranges on purpose.
- The unchanged run must give the stored outputs again, to 0.001 in every probability, and the published number of the selection half.

### Stage 3 against stage 2

Stage 3 of the training adds changes of grey levels, geometry and image quality ([TRAINING.md](TRAINING.md)). The three published stage-2 models, from which stage 3 starts, are run as one ensemble under the same conditions.

- Their decoder setting is the one that the rule gives them on the calibration half, from their own outputs for it. When no setting keeps the floors they are dropped, and the table says so.
- For every condition the published system minus the stage-2 ensemble is given, with the same draws.
- Stage 3 also trained for 1,500 further updates. What is measured is stage 3 as a whole against its starting point. It does not separate the augmentation from the further updates.

### A view the system does not know

EV9V has a ninth view, PMPALA, that the system was never given. Its recordings in the validation split and in the training split are read as `echoview classify --fps 30` reads a folder of frames. The training code reads recordings of the five views only, so it has read none of them.

- **Reported for each of the two lists:** how many recordings get each of the five names, the quartiles of the probability, in how many the seeds differ.
- **One reader for both kinds.** The 521 development recordings of the five views are read in the same way, as a folder of frames, so that the two kinds are compared on one way of reading. How far that reading lies from the benchmark's own frames is reported: the largest difference in a probability, and how many recordings get another view.
- **Told apart or not.** Three numbers say how sure the five probabilities are: the largest probability, its margin over the second, and the negative entropy. For each, the chance that a recording of the five views is surer than a recording of the ninth view, where a tie counts half. 0.5 means that the number does not tell them apart.

### How part 2 is read

For every condition, every number is given minus its value on the unchanged frames, with a 95% interval from the same draws. For the stage-2 ensemble that is its own run on the unchanged frames. For `classify`'s rule the wrong recordings, the mean probability and the recordings in which the seeds differ are given as counts. These are many intervals and none is corrected for their number. Part 2 gets no word: it describes.

## 5. The numbers and the intervals

- **Numbers.** For multi streams: segment F1 at 50%, the Edit score, wrong samples, precision and recall of the cuts, the mean distance of the matched cuts, cuts at joins of the same view, returned segments that match nothing and reference segments that are missed. For pair streams: segment F1 at 50%. For single recordings: accuracy, macro-F1, A4C read as A5C and A5C read as A4C, label changes inside a recording. They are those of [COMPARISON.md](COMPARISON.md), section 5, from the same code.
- **Intervals.** Percentile intervals from 10,000 draws with seed 20261123. Streams are drawn for the numbers about streams and recordings for the numbers about recordings.
- **They are too narrow.** The development benchmark does not name the session of a recording, so sessions are not drawn, and each half has only 36 of them. The streams also share recordings. An interval here says less than it seems to about new recordings.

## 6. What follows, whatever comes out

- **Nothing changes in the published system.** No weight, no decoder setting, no rule.
- **A decoder or a temperature that comes out well is a candidate and no more.** The released decoder, its grid and its floors were developed on these streams, and so was the model, so development numbers favour what exists: from the selection half to the test benchmark EchoRouter lost 0.05 where another system lost nothing ([COMPARISON.md](COMPARISON.md), section 10). A change would need its own plan and other data to confirm it. On the test benchmark that would be one more read, which the owner of the project would have to allow by name.
- **A weakness under a condition is reported as a limit.** Training against it would make another model, with a plan of its own.
- **Everything is published,** in `docs/ablations/`, whatever it shows.

## 7. What is expected

- **The ensemble.** The three seeds together at or above every pair and every single seed on the selection half, by 0.01 or less. With `classify`'s rule no visible difference: there are one to three wrong recordings.
- **Other decoders.** The released decoder has the highest value. The Viterbi path and the two filters come within 0.03 of it, and for at least one of them the difference is not shown. No decoder at all is clearly behind. The row without the view test cuts more often at joins of the same view.
- **The settings.** Most of the 40 settings lie within 0.02 of the published one on the selection half. With the halves exchanged the rule may choose another setting.
- **The temperature.** A value below 1, and a lower log loss: a candidate.
- **Conditions.** Small losses under the changes of grey levels, which stage 3 trained with. Larger ones under strong noise, 64 pixels and JPEG at quality 10. A mirrored image is the open one: no training stage mirrored. With a still frame the single recordings should be read nearly as well and the streams worse. Shuffled frames should cost little, because earlier training arms of the project found no value in the order of the frames.
- **Stage 3 against stage 2.** Stage 3 ahead under changes of grey levels and image quality, and no difference on the unchanged frames.
- **The ninth view.** Nearly all of its recordings named PSAX, most with a probability above 0.9, and a separation near 0.6: the probability does not tell.

## 8. The limits

- **Development data.** The model, the decoder and its rule were developed on these halves. Everything here describes the system where it is at home.
- **One hospital, built streams.** The recordings come from one hospital and two machines of one maker, and the streams are joined from fragments ([DATA.md](DATA.md)).
- **The conditions are made up.** A gamma, a blur or a rotation is not another machine, another operator or another patient. They show where the system is brittle, and say little about how often that matters.
- **One unknown view.** PMPALA is one view of many that an examination holds, and it is a parasternal view close to PSAX.
- **What this document cannot give,** and a reviewer would still ask for:
  - recordings of other hospitals and machines with views named by readers, and how often readers agree;
  - continuous examinations with reference cuts;
  - more than one training run of each of the other approaches, and the same recipe started from other weights than EchoPrime's;
  - stage 3 with the same further updates and no augmentation.

## 9. Results

### Part 1: from stored outputs

**Run on 2026-11-26 at 03:33 UTC with the code of commit `fb674f6`, which is this plan.** The output of the command is [`ablations/stored.json`](ablations/stored.json). The stored logits gave the published number of the selection half, 0.9594, before anything else was computed. Every number below is for the selection half unless a line names the calibration half. Intervals are at 95% unless a line says 99%. Streams are drawn for the numbers about streams and recordings for the numbers about recordings.

**The text of part 1 was corrected on 2026-11-26, after an independent check of it against the file.** No number was wrong. Some sentences said more than the numbers carry, and numbers that the plan had promised stood only in the file. The end of part 1 lists what changed.

**The size of the ensemble.**

| System | Decoder setting by the rule | Segment F1 at 50%, multi streams | Minus the three seeds together | Pair streams | Recordings right of 284, majority rule | Wrong by `classify`'s rule | Calibration error, `classify`'s rule |
|---|---|---|---|---|---|---|---|
| seed 0 | half window 7, threshold 0.186 | 0.958 [0.935, 0.978] | -0.001 [-0.005, +0.002] | 0.971 | 281 | 1 | 0.034 |
| seed 1 | half window 5, threshold 0.130 | 0.948 [0.923, 0.969] | -0.012 [-0.023, -0.003] | 0.978 | 281 | 1 | 0.032 |
| seed 2 | half window 5, threshold 0.137 | 0.948 [0.924, 0.970] | -0.011 [-0.023, -0.001] | 0.974 | 282 | 1 | 0.037 |
| seeds 0+1 | half window 7, threshold 0.179 | 0.958 [0.935, 0.977] | -0.002 [-0.005, +0.000] | 0.971 | 281 | 1 | 0.033 |
| seeds 0+2 | half window 5, threshold 0.139 | 0.955 [0.932, 0.976] | -0.004 [-0.013, +0.003] | 0.966 | 282 | 1 | 0.034 |
| seeds 1+2 | half window 5, threshold 0.133 | 0.948 [0.924, 0.970] | -0.011 [-0.022, -0.002] | 0.959 | 282 | 1 | 0.034 |
| seeds 0+1+2 | half window 7, threshold 0.174 | 0.959 [0.937, 0.979] |  | 0.971 | 281 | 1 | 0.034 |

Each smaller system minus the three seeds together, for every number of section 5, with 95% intervals. A count of confused recordings has no interval.

| Minus the three seeds together | seed 0 | seed 1 | seed 2 | seeds 0+1 | seeds 0+2 | seeds 1+2 |
|---|---|---|---|---|---|---|
| Segment F1 at 50%, multi streams | -0.001 [-0.005, +0.002] | -0.012 [-0.023, -0.003] | -0.011 [-0.023, -0.001] | -0.002 [-0.005, +0.000] | -0.004 [-0.013, +0.003] | -0.011 [-0.022, -0.002] |
| Edit | -0.2 [-0.9, +0.3] | -1.5 [-2.9, -0.3] | -1.0 [-2.4, +0.2] | -0.4 [-1.0, +0.0] | -0.5 [-1.8, +0.6] | -1.0 [-2.2, +0.1] |
| Wrong samples, percentage points | +0.2 [-0.0, +0.4] | +0.6 [-0.1, +1.4] | +0.6 [+0.0, +1.3] | +0.1 [-0.1, +0.5] | +0.3 [-0.1, +0.9] | +0.6 [+0.1, +1.3] |
| Cut precision | -0.003 [-0.012, +0.005] | -0.013 [-0.032, +0.002] | -0.006 [-0.022, +0.009] | -0.005 [-0.013, +0.000] | +0.002 [-0.012, +0.014] | -0.006 [-0.021, +0.007] |
| Cut recall | -0.005 [-0.013, +0.000] | -0.013 [-0.028, -0.003] | -0.018 [-0.035, -0.005] | +0.000 [+0.000, +0.000] | -0.008 [-0.018, +0.000] | -0.013 [-0.027, -0.003] |
| Mean distance of matched cuts, seconds | +0.007 [+0.002, +0.014] | -0.003 [-0.009, +0.003] | -0.004 [-0.008, +0.000] | -0.001 [-0.004, +0.002] | -0.000 [-0.003, +0.003] | -0.003 [-0.006, -0.000] |
| Cuts at joins of the same view, percentage points | +0.0 [+0.0, +0.0] | +0.9 [+0.0, +2.7] | +0.9 [+0.0, +2.7] | +0.0 [+0.0, +0.0] | +0.9 [+0.0, +2.7] | +0.9 [+0.0, +2.7] |
| Segments that match nothing | +0 [-3, +3] | +6 [+0, +13] | +3 [-3, +10] | +2 [+0, +5] | +0 [-5, +6] | +4 [-2, +11] |
| Segments missed | +1 [+0, +3] | +6 [+1, +14] | +8 [+2, +16] | +0 [+0, +0] | +4 [+0, +9] | +7 [+1, +15] |
| Segment F1 at 50%, pair streams | +0.000 [+0.000, +0.000] | +0.007 [-0.007, +0.029] | +0.002 [+0.000, +0.007] | +0.000 [+0.000, +0.000] | -0.005 [-0.022, +0.007] | -0.012 [-0.031, +0.000] |
| Accuracy, single recordings | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.004 [+0.000, +0.011] | +0.000 [+0.000, +0.000] | +0.004 [+0.000, +0.011] | +0.004 [+0.000, +0.011] |
| Macro-F1, single recordings | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.009 [+0.000, +0.032] | +0.000 [+0.000, +0.000] | +0.009 [+0.000, +0.032] | +0.009 [+0.000, +0.032] |
| A4C read as A5C, recordings | +0 | +0 | -1 | +0 | -1 | -1 |
| A5C read as A4C, recordings | +0 | +0 | +0 | +0 | +0 | +0 |
| Label changes per recording | -0.014 [-0.035, +0.000] | +0.014 [+0.000, +0.032] | +0.004 [-0.018, +0.025] | +0.000 [+0.000, +0.000] | -0.004 [-0.028, +0.021] | +0.007 [+0.000, +0.018] |

- **On the multi streams the three seeds together have the highest segment F1.** The mean over the single seeds is 0.951, over the pairs 0.954, and the three together reach 0.959. Against seed 1, seed 2 and their pair the interval of the difference lies below zero. Against seed 0 and the two pairs that hold it, it does not: seed 0 alone is 0.001 below the three together, with an interval from -0.005 to +0.002. These are six intervals at 95%, and none is corrected for their number.
- **On pair streams, and on single recordings by the majority rule, they are not the highest.** On pair streams seed 1 (0.978) and seed 2 (0.974) are above the three together (0.971). By the majority rule the three together read 281 of the 284 single recordings right, and seed 2 and the two pairs that hold it read 282. No interval of these differences excludes zero.
- **Of the other numbers on the multi streams** the three together are alone the best on the Edit score and on wrong samples. They are level with the best on cut recall, segments missed, segments that match nothing and cuts at joins of the same view. They are not the best on cut precision and on the distance of the matched cuts. The table above has every difference. Its 78 intervals are at 95%, and none is corrected for their number.
- **The rule gives the three seeds the published setting:** half window 7, a threshold that agrees with the published one to 0.0001, and the same value for every number. Every smaller system gets another threshold, and four of the six another half window.
- **By `classify`'s rule the size makes no visible difference.** Every system has one wrong recording of 284, and the calibration errors lie between 0.032 and 0.037.
- **Against what was expected.** Section 7 expected the three together at or above every smaller system, by 0.01 or less. On the multi streams they are above each, by 0.012 at most. On pair streams and by the majority rule they are not above each.
- **Two numbers of the plan come out differently.** Section 2 quoted 0.957 and 0.942 for two of the pairs from the research records. From the published logits and by the rule they are 0.955 and 0.948. The numbers of the records have no file in this repository, and why they differ was not looked into.

**Other decoders.** The released decoder minus each other one, for segment F1 at 50% on the multi streams. The interval that decides is at 99%.

| Decoder | Its number | Segment F1 at 50%, multi streams | Released minus this | 99% interval | The released decoder is | Edit | Cut precision | Cut recall | Cuts at joins of the same view | Segments that match nothing | Segments missed | Label changes per recording |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| released change-point decoder | half window 7, threshold 0.174 | 0.959 | | | | 96.5 | 0.959 | 0.976 | 1.3% | 24 | 17 | 0.049 |
| no decoder | none | 0.906 | +0.054 | [+0.022, +0.089] | ahead of | 91.6 | 0.858 | 0.984 | 3.9% | 78 | 22 | 0.102 |
| mode filter | 17 | 0.923 | +0.037 | [+0.008, +0.069] | ahead of | 93.3 | 0.896 | 0.971 | 2.6% | 56 | 24 | 0.077 |
| mean filter | 3 | 0.908 | +0.052 | [+0.022, +0.087] | ahead of | 91.6 | 0.861 | 0.987 | 3.5% | 77 | 21 | 0.102 |
| minimum duration | 8 | 0.921 | +0.039 | [+0.010, +0.072] | ahead of | 93.0 | 0.885 | 0.950 | 2.6% | 55 | 27 | 0.077 |
| Viterbi path | 8 | 0.953 | +0.007 | [-0.009, +0.026] | not shown to differ from | 96.1 | 0.944 | 0.969 | 0.9% | 29 | 19 | 0.039 |
| change points without the view test | half window 7, threshold 0.174 | 0.959 | +0.000 |  |  | 96.5 | 0.959 | 0.976 | 1.3% | 24 | 17 | 0.049 |

The other numbers of section 5 for the same decoders:

| Decoder | Segment F1 at 50%, pair streams | Wrong samples | Mean distance of matched cuts | Recordings right of 284, majority rule | Macro-F1 | A4C read as A5C | A5C read as A4C |
|---|---|---|---|---|---|---|---|
| released change-point decoder | 0.971 | 6.1% | 0.153 s | 281 | 0.979 | 2 | 0 |
| no decoder | 0.957 | 7.0% | 0.157 s | 282 | 0.988 | 1 | 0 |
| mode filter | 0.971 | 7.0% | 0.165 s | 282 | 0.988 | 1 | 0 |
| mean filter | 0.953 | 6.5% | 0.139 s | 282 | 0.988 | 1 | 0 |
| minimum duration | 0.971 | 7.2% | 0.166 s | 282 | 0.988 | 1 | 0 |
| Viterbi path | 0.969 | 6.2% | 0.160 s | 281 | 0.979 | 2 | 0 |
| change points without the view test | 0.971 | 6.1% | 0.153 s | 281 | 0.979 | 2 | 0 |

The released decoder minus each other one, for every number, with 95% intervals. These are descriptions, and only the first table decides a word. A count of confused recordings has no interval. The 65 intervals are not corrected for their number. The 95% interval of every value of the two tables above is in the file.

| Released minus | no decoder | mode filter | mean filter | minimum duration | Viterbi path |
|---|---|---|---|---|---|
| Segment F1 at 50%, multi streams | +0.054 [+0.029, +0.080] | +0.037 [+0.014, +0.061] | +0.052 [+0.027, +0.078] | +0.039 [+0.016, +0.064] | +0.007 [-0.006, +0.021] |
| Edit | +4.9 [+2.5, +7.5] | +3.2 [+1.1, +5.6] | +4.9 [+2.5, +7.5] | +3.5 [+1.3, +5.9] | +0.4 [-0.8, +1.7] |
| Wrong samples, percentage points | -0.9 [-1.8, -0.2] | -0.9 [-1.7, -0.2] | -0.4 [-1.2, +0.3] | -1.1 [-1.9, -0.3] | -0.2 [-0.7, +0.3] |
| Cut precision | +0.100 [+0.065, +0.138] | +0.063 [+0.030, +0.098] | +0.098 [+0.062, +0.137] | +0.074 [+0.041, +0.109] | +0.015 [+0.003, +0.029] |
| Cut recall | -0.008 [-0.021, +0.000] | +0.005 [-0.008, +0.018] | -0.010 [-0.024, +0.000] | +0.026 [+0.008, +0.046] | +0.008 [-0.003, +0.022] |
| Mean distance of matched cuts, seconds | -0.004 [-0.009, +0.001] | -0.012 [-0.019, -0.005] | +0.014 [+0.007, +0.020] | -0.013 [-0.021, -0.005] | -0.007 [-0.013, -0.001] |
| Cuts at joins of the same view, percentage points | -2.6 [-5.1, -0.5] | -1.3 [-3.5, +0.0] | -2.2 [-4.6, -0.4] | -1.3 [-3.5, +0.0] | +0.4 [+0.0, +1.4] |
| Segments that match nothing | -54 [-77, -33] | -32 [-52, -14] | -53 [-76, -32] | -31 [-51, -13] | -5 [-12, +2] |
| Segments missed | -5 [-13, +4] | -7 [-15, +1] | -4 [-12, +5] | -10 [-20, -1] | -2 [-11, +6] |
| Segment F1 at 50%, pair streams | +0.014 [+0.000, +0.038] | +0.000 [+0.000, +0.000] | +0.019 [+0.000, +0.043] | +0.000 [+0.000, +0.000] | +0.002 [+0.000, +0.008] |
| Accuracy, single recordings | -0.004 [-0.011, +0.000] | -0.004 [-0.011, +0.000] | -0.004 [-0.011, +0.000] | -0.004 [-0.011, +0.000] | +0.000 [+0.000, +0.000] |
| Macro-F1, single recordings | -0.009 [-0.032, +0.000] | -0.009 [-0.032, +0.000] | -0.009 [-0.032, +0.000] | -0.009 [-0.032, +0.000] | +0.000 [+0.000, +0.000] |
| A4C read as A5C, recordings | +1 | +1 | +1 | +1 | +0 |
| A5C read as A4C, recordings | +0 | +0 | +0 | +0 | +0 |
| Label changes per recording | -0.053 [-0.109, -0.014] | -0.028 [-0.053, -0.007] | -0.053 [-0.109, -0.014] | -0.028 [-0.053, -0.007] | +0.011 [-0.018, +0.039] |

- **The released decoder is ahead of no decoder, of both filters and of the minimum duration.** It is 0.037 to 0.054 above them, and every 99% interval lies above zero.
- **It is not shown to differ from a Viterbi path.** A path with one number, a cost of 8 for every change of view, reaches 0.953 where the released decoder reaches 0.959. The difference is 0.007, and its 99% interval runs from -0.009 to +0.026. This does not show that the two are equal: the interval allows the released decoder to be 0.026 better or 0.009 worse.
- **Two of the other numbers separate the two, both for the released decoder.** The precision of its cuts is higher by 0.015, with an interval from +0.003 to +0.029. Its matched cuts lie nearer to the reference cuts by 0.007 s, and the interval of that difference in distance runs from -0.013 to -0.001. The Viterbi path has fewer cuts at joins of the same view (2 against 3 of 230 joins) and fewer label changes inside single recordings (11 against 14 over the 284), and neither interval excludes zero.
- **On single recordings the released decoder is not the best.** By the majority rule the released decoder and the Viterbi path read 281 of the 284 recordings right, with A4C read as A5C twice. No decoder, the two filters and the minimum duration read 282, with A4C read as A5C once. The difference is one recording, -0.004 [-0.011, +0.000] in accuracy.
- **Its matched cuts are not the nearest.** They lie 0.153 s from the reference cuts on average. Those of the mean filter lie at 0.139 s, nearer by 0.014 s (+0.014 [+0.007, +0.020]). Those of the mode filter, the minimum duration and the Viterbi path lie farther, at 0.165 s, 0.166 s and 0.160 s, and each of these intervals excludes zero. Without a decoder they lie at 0.157 s, and that interval includes zero. Every decoder matches cuts of its own, so these distances are not over the same cuts.
- **The released decoder has the fewest wrong samples,** 6.1%, against 6.2% for the Viterbi path and 6.5% to 7.2% for the others. The interval of the difference excludes zero against no decoder, the mode filter and the minimum duration.
- **On pair streams the mode filter and the minimum duration equal the released decoder,** at 0.971. The Viterbi path reaches 0.969, no decoder 0.957 and the mean filter 0.953. No interval lies wholly above zero.
- **Without the view test no number changes.** Without the test the rule chooses the same setting, every number of the tables comes out the same, and every interval of a difference is [0, 0]. This says little about the test. It drops a peak when the windows on its two sides name the same view, and every number here is computed from the views of the samples. A dropped peak can show in them only when the two pieces that it would have separated take different views. So the numbers stay the same when the test drops no peak, and they stay the same when the pieces on the two sides of every dropped peak take the same view. They do not say which of the two it is. The file does not hold how many peaks the test drops. Section 7 expected more cuts at joins of the same view without the test. With numbers of this kind that could hardly have shown.
- **How the numbers were chosen.** For every other decoder some value kept the two floors on the calibration half: 6 of 8 for the mode filter, the mean filter and the Viterbi path, and 4 of 7 for the minimum duration. No floor was dropped. The window of the mean filter, 3, is the smallest that was tried. On the calibration half, where the numbers were chosen, the decoders lie closer together than on the selection half: the released decoder 0.984, the mode filter 0.983, the minimum duration 0.983, the Viterbi path 0.981 and the mean filter 0.979.
- **Against what was expected.** The released decoder has the highest segment F1 on the multi streams, and doing without a decoder is clearly behind, both as expected. The Viterbi path came within 0.03 of the released decoder, as expected. The two filters did not: they are 0.037 and 0.052 below.

**The decoder's settings.** Segment F1 at 50% on the multi streams of the selection half, at every setting. A row is a half window in samples. A column is the quantile of the scores inside single recordings of the calibration half that gives the threshold.

| Half window | 0.80 | 0.85 | 0.90 | 0.95 | 0.97 | 0.99 | 0.995 | 0.999 |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.904 | 0.904 | 0.904 | 0.904 | 0.904 | 0.903 | 0.912 | 0.947 |
| 7 | 0.913 | 0.913 | 0.913 | 0.913 | 0.913 | 0.914 | 0.920 | 0.959 |
| 9 | 0.916 | 0.916 | 0.916 | 0.916 | 0.916 | 0.921 | 0.926 | 0.958 |
| 11 | 0.918 | 0.918 | 0.918 | 0.918 | 0.918 | 0.921 | 0.941 | 0.953 |
| 15 | 0.916 | 0.916 | 0.916 | 0.916 | 0.916 | 0.934 | 0.942 | 0.941 |

- **The decoder depends on its threshold.** The published setting, half window 7 at the 0.999 quantile, has the highest value of the 40. Three of the 40 settings lie within 0.01 of that value, the published one among them, and seven within 0.02. None lies above it. The lowest is 0.903 and the median 0.916. Section 7 expected most settings within 0.02. That was wrong.
- **At the five lower quantiles the decoder is below the mode filter and the minimum duration.** There every half window stays between 0.904 and 0.918, where the minimum duration reaches 0.921 and the mode filter 0.923. At the 0.99 and 0.995 quantiles four of the ten settings are above both, with 0.926 to 0.942.
- **The published threshold is the highest on the grid.** The grid does not show what a still higher threshold would do.
- **All 40 settings keep the two floors on the calibration half,** so there the rule decides by segment F1 alone. On the selection half 7 of the 40 fall below the floor of 0.980 for the macro-F1 of single recordings. The published setting is one of them, with 0.979, and so are five of the six other settings within 0.02 of it. None falls below the floor of 0.95 for the boundary recall. The table at the end of part 1 has the three numbers of the rule for every setting on both halves.
- **The halves exchanged.** Chosen on the selection half, the rule takes half window 5 with a threshold of 0.098, the 0.995 quantile of the scores of the selection half. This is not a cell of the table above, whose thresholds come from the calibration half: its cell for half window 5 and 0.995 has a threshold of 0.054. On the calibration half the exchanged choice gives 0.981, where the published setting gives 0.984. The file does not hold what the exchanged choice reaches on the selection half, nor how many settings kept the floors there.

**One temperature.** The fit on the calibration half chose 0.30, the lowest value that was tried. The log loss there falls from 0.041 at a temperature of 1 to 0.016 at 0.5, and stays near 0.015 below that. The intervals of this part draw recordings.

| The 284 single recordings of the selection half | Wrong | Mean probability | Right | Mean probability minus the share that is right | Calibration error, equal width | Calibration error, equal count | Brier score | Log loss |
|---|---|---|---|---|---|---|---|---|
| as published | 1 | 0.963 | 283 | -0.034 [-0.042, -0.025] | 0.034 | 0.034 | 0.011 | 0.043 |
| with the temperature | 1 | 0.989 | 283 | -0.007 [-0.014, +0.000] | 0.012 | 0.007 | 0.008 | 0.015 |

- **The temperature is a candidate.** The log loss with the temperature minus the log loss as published is -0.028 [-0.032, -0.024], the same difference for the Brier score is -0.004 [-0.006, -0.001], and no more recordings are wrong.
- **It closes most of the gap.** The system is less sure than it is right by 0.034 as published, and by 0.007 with the temperature.
- **Two reasons for care.** The chosen value is the end of the values tried, so the fit did not find a best one. And one recording of 284 is wrong here, so making every answer surer costs almost nothing. On the test benchmark 14 of 800 recordings are wrong by `classify`'s rule, two of them with a probability above 0.9 ([EVALUATION.md](EVALUATION.md)). Whether the same temperature helps there is not known, and by section 6 that would take one more planned read.

**All 40 settings of the decoder.** The three numbers of the rule on both halves: segment F1 at 50% on the multi streams, the macro-F1 of single recordings, and the boundary recall on the multi streams. The thresholds come from the calibration half.

| Half window | Quantile | Threshold | Calibration half: segment F1 | Macro-F1 | Boundary recall | Selection half: segment F1 | Macro-F1 | Boundary recall |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.8 | 0.00005 | 0.975 | 0.993 | 0.982 | 0.904 | 0.988 | 0.963 |
| 5 | 0.85 | 0.00007 | 0.975 | 0.993 | 0.982 | 0.904 | 0.988 | 0.963 |
| 5 | 0.9 | 0.00015 | 0.975 | 0.993 | 0.982 | 0.904 | 0.988 | 0.963 |
| 5 | 0.95 | 0.00081 | 0.975 | 0.993 | 0.982 | 0.904 | 0.988 | 0.963 |
| 5 | 0.97 | 0.00289 | 0.975 | 0.993 | 0.982 | 0.904 | 0.988 | 0.963 |
| 5 | 0.99 | 0.02520 | 0.978 | 0.993 | 0.982 | 0.903 | 0.988 | 0.961 |
| 5 | 0.995 | 0.05421 | 0.979 | 0.993 | 0.982 | 0.912 | 0.988 | 0.961 |
| 5 | 0.999 | 0.14120 | 0.981 | 0.993 | 0.980 | 0.947 | 0.979 | 0.955 |
| 7 | 0.8 | 0.00006 | 0.977 | 0.993 | 0.990 | 0.913 | 0.988 | 0.984 |
| 7 | 0.85 | 0.00010 | 0.977 | 0.993 | 0.990 | 0.913 | 0.988 | 0.984 |
| 7 | 0.9 | 0.00021 | 0.977 | 0.993 | 0.990 | 0.913 | 0.988 | 0.984 |
| 7 | 0.95 | 0.00120 | 0.977 | 0.993 | 0.990 | 0.913 | 0.988 | 0.984 |
| 7 | 0.97 | 0.00468 | 0.977 | 0.993 | 0.990 | 0.913 | 0.988 | 0.984 |
| 7 | 0.99 | 0.04173 | 0.977 | 0.993 | 0.990 | 0.914 | 0.988 | 0.982 |
| 7 | 0.995 | 0.08171 | 0.978 | 0.993 | 0.990 | 0.920 | 0.988 | 0.982 |
| 7 | 0.999 | 0.17383 | 0.984 | 0.986 | 0.990 | 0.959 | 0.979 | 0.976 |
| 9 | 0.8 | 0.00007 | 0.975 | 0.993 | 0.990 | 0.916 | 0.988 | 0.984 |
| 9 | 0.85 | 0.00012 | 0.975 | 0.993 | 0.990 | 0.916 | 0.988 | 0.984 |
| 9 | 0.9 | 0.00028 | 0.975 | 0.993 | 0.990 | 0.916 | 0.988 | 0.984 |
| 9 | 0.95 | 0.00157 | 0.975 | 0.993 | 0.990 | 0.916 | 0.988 | 0.984 |
| 9 | 0.97 | 0.00656 | 0.975 | 0.993 | 0.990 | 0.916 | 0.988 | 0.984 |
| 9 | 0.99 | 0.05326 | 0.976 | 0.993 | 0.990 | 0.921 | 0.988 | 0.982 |
| 9 | 0.995 | 0.10344 | 0.976 | 0.993 | 0.990 | 0.926 | 0.988 | 0.982 |
| 9 | 0.999 | 0.22197 | 0.981 | 0.986 | 0.985 | 0.958 | 0.979 | 0.976 |
| 11 | 0.8 | 0.00008 | 0.973 | 0.993 | 0.990 | 0.918 | 0.988 | 0.984 |
| 11 | 0.85 | 0.00014 | 0.973 | 0.993 | 0.990 | 0.918 | 0.988 | 0.984 |
| 11 | 0.9 | 0.00034 | 0.973 | 0.993 | 0.990 | 0.918 | 0.988 | 0.984 |
| 11 | 0.95 | 0.00198 | 0.973 | 0.993 | 0.990 | 0.918 | 0.988 | 0.984 |
| 11 | 0.97 | 0.00876 | 0.973 | 0.993 | 0.990 | 0.918 | 0.988 | 0.984 |
| 11 | 0.99 | 0.06257 | 0.973 | 0.993 | 0.990 | 0.921 | 0.988 | 0.982 |
| 11 | 0.995 | 0.12294 | 0.975 | 0.993 | 0.990 | 0.941 | 0.977 | 0.979 |
| 11 | 0.999 | 0.23574 | 0.975 | 0.986 | 0.985 | 0.953 | 0.979 | 0.974 |
| 15 | 0.8 | 0.00010 | 0.975 | 0.993 | 0.990 | 0.916 | 0.988 | 0.982 |
| 15 | 0.85 | 0.00017 | 0.975 | 0.993 | 0.990 | 0.916 | 0.988 | 0.982 |
| 15 | 0.9 | 0.00046 | 0.975 | 0.993 | 0.990 | 0.916 | 0.988 | 0.982 |
| 15 | 0.95 | 0.00259 | 0.975 | 0.993 | 0.990 | 0.916 | 0.988 | 0.982 |
| 15 | 0.97 | 0.01125 | 0.975 | 0.993 | 0.990 | 0.916 | 0.988 | 0.982 |
| 15 | 0.99 | 0.07830 | 0.975 | 0.993 | 0.990 | 0.934 | 0.977 | 0.982 |
| 15 | 0.995 | 0.13027 | 0.973 | 0.993 | 0.987 | 0.942 | 0.979 | 0.979 |
| 15 | 0.999 | 0.28204 | 0.968 | 0.986 | 0.967 | 0.941 | 0.998 | 0.958 |

**What the correction of 2026-11-26 changed.**

- **The lower thresholds.** A headline said that below the highest quantile the decoder is no better than a filter. Four settings at the 0.99 and 0.995 quantiles are above the mode filter and the minimum duration. At the five lower quantiles the decoder is below those two, and 20 of its 25 settings there are above the mean filter.
- **The floors.** The sentence that the floors do not narrow the choice was true for the calibration half only. The selection half is now given.
- **The exchanged choice** was given by its quantile alone, which reads as a cell of the table. Its threshold is now given.
- **The ensemble.** The text used the words of a comparison, which the plan had not given this part, and named only the number on which the three seeds together are highest. It now says on which numbers they are not, and that the expected 0.01 was passed.
- **The view test.** The row without it was read as showing that the test does nothing, and as holding in every draw. The file holds the ends of the intervals, and the row shows that no number changes, which a dropped peak could change only when the pieces on its two sides take different views.
- **The Viterbi path.** Two of its rates were compared without their counts and intervals. The released decoder was said to have the higher precision and recall of the cuts. That is now said of the precision, because the interval for the recall includes zero.
- **Two readings against the released decoder are new.** By the majority rule it reads one single recording fewer right than four simpler decoders, and the matched cuts of the mean filter lie nearer. The first text had neither.
- **The numbers of the plan.** Two numbers of section 2 were called corrected. They come out differently, and the numbers of the records were not checked.
- **How the other decoders were chosen.** The window of the mean filter is the smallest that was tried, and the calibration half hardly separates the decoders. Both are now said.
- **The published setting of the three seeds.** The threshold of the rule agrees with the published one to 0.0001. The first text called them the same.
- **The intervals.** The first text said that every interval draws streams. Those about recordings draw recordings, the temperature's among them. The difference of the Brier score now has its interval, and both differences of the temperature are written with their sign.
- **Added tables.** The numbers of section 5 that the first tables left out, every difference of the decoders and of the subsets of the seeds with its interval, the three numbers of the rule for all 40 settings on both halves, and the calibration error of every subset of the seeds.
- **Part 2.** When the results of part 1 were committed, at 03:48 UTC, this document said of part 2 "Not run yet". Part 2 had been started at 03:36 UTC.

### Part 2: forward passes

Started on 2026-11-26 at 03:36 UTC and interrupted at 07:43 UTC, when the server stopped answering. It went on at 20:47 UTC and ended at 22:19 UTC. Its results are not written yet: when this line was committed, no number of its output had been looked at.

## 10. Part 3: what part 1 left open

**Status: planned on 2026-11-26 (UTC), after part 1 was read and its text corrected, and before any number of part 3 existed. Not run yet.**

Part 1 ended with five things that its file does not show. Part 3 takes them. The input is the same file of stored logits, no model runs, the test benchmark is not read, and nothing changes in the published system. `python -m echoview.evaluation.open_points` is the command. As in part 1, what is chosen is chosen on the calibration half and reported on the selection half. The one exception is the exchange of the halves, which is there to show the other direction.

**This plan was written with the results of part 1 known, and that weakens what part 3 can show.**

- The selection half has been looked at for all 40 settings of the decoder's grid, for five other decoders, for seven subsets of the seeds and for the temperature 0.30.
- That higher thresholds are worth a look was itself read off the selection half, where the value still rose with the threshold.
- 40 of the 55 settings below and 35 of the 40 temperatures are those of part 1, and their numbers are known.
- One open point of part 1 is not taken: why two numbers of the research records differ from the published logits.

**When no setting keeps the floors.** Wherever the rule chooses in part 3 and no setting keeps its two floors, the floors are dropped. The setting with the highest segment F1 takes their place, the earlier of equals, and the output says so.

### Thresholds above the grid

The published threshold is the highest of its grid: the 0.999 quantile of the scores inside single recordings of the calibration half. Three higher quantiles are added for every half window: 0.9995, 0.9999 and 1, which is the largest score inside a single recording. That gives 15 more settings, 55 in all.

- **These thresholds rest on few scores.** The single recordings of the calibration half give 14,086 scores for a half window. The 0.9995 quantile is about the eighth largest of them, the 0.9999 quantile lies between the second and the third largest, and quantile 1 is the largest. Neighbouring samples share an event, so these few scores may come from one to three recordings.
- **Given for every setting:** the three numbers of the rule on both halves.
- **The rule over the 55.** The project's rule chooses on the calibration half, with its two floors and with the earlier setting on a tie. The added settings of a half window come after its own eight.
- **The word.** When the rule still takes the published setting, the word is "the published setting stays". Otherwise its choice is compared with the published setting on the selection half, with the same draws. The choice is a "candidate" when the 95% interval of the difference in segment F1 at 50% on the multi streams lies wholly above zero and the accuracy on single recordings, by the majority rule, is not lower as a number. Otherwise it is "not supported on development data".
- **When the choice is the last threshold.** If the rule takes a setting at quantile 1, the word is given as above, the result says that the choice lies at the end of the thresholds tried, and no higher threshold is tried.

### The halves exchanged, in full

Part 1 gave the exchanged choice and one number for it. Here the thresholds of the grid are taken from the selection half, as the rule takes them when it chooses there. All 40 settings are given on both halves, with how many keep the floors on the selection half, and which setting is the highest there whatever the floors. This describes and gets no word.

### What the view test drops

At the published setting, for both halves, and for single recordings, pair streams and multi streams apart: the peaks of the score, how many of them the view test drops and in how many streams, and in how many streams and samples the decoder without the test gives another view. This describes and gets no word.

### The other decoders at every value of their number

Part 1 gave each other decoder at the one value that the rule chose for it. Here the mode filter, the mean filter, the minimum duration and the Viterbi path are given at every value of section 3, with the three numbers of the rule on both halves. The summary is that of the decoder's own settings: on the selection half, how many values come within 0.01 and within 0.02 of the segment F1 of the chosen value, how many lie above it, and the lowest and the highest. A grid here has 7 or 8 values where the released decoder's has 40, so the counts are not shares of the same thing. The highest value of another decoder on the selection half was not chosen on the calibration half, and it is not set against the released decoder. This describes and gets no word.

### Lower temperatures

Part 1 chose 0.30, the lowest value that was tried. Five lower values are added: 0.05, 0.10, 0.15, 0.20 and 0.25. The fit, what is reported and the word are those of section 3, by the same function.

- **The word is about `classify`'s probabilities only.** Part 1 gave "candidate" for 0.30 on these recordings, so the word is likely here as well and says little that is new. Whether a temperature could serve the streams is what the paragraph below describes.
- **When the fit takes 0.05.** Then it has again taken the lowest value tried. The word is given as above, the result says that the fit found no best value, and no lower value is tried: near zero every probability becomes 0 or 1.
- **The log loss of one recording is capped.** A probability counts as at least 1e-12, so one recording adds at most 27.6. At a low temperature a recording that is read wrong can reach that cap.

**The streams under the temperature.** The decoder reads the same probabilities. With every logit divided by the fitted temperature, the multi streams, the pair streams and the single recordings of the selection half are scored twice: with the published decoder setting, and with the setting that the rule gives the changed probabilities on the calibration half, from its grid of 40. Both are given minus the published system, with the same draws. Sharp probabilities can make the thresholds of that grid fall together, and the output says how many of the 40 are distinct and how many are zero. This describes and gets no word.

### What is expected of part 3

- **Higher thresholds.** At least one of the 15 added settings has a higher segment F1 on the multi streams of the selection half than the published setting. The rule takes the published setting or a setting at the 0.9995 quantile. The word is not "candidate".
- **The halves exchanged.** Fewer than 40 settings keep the floors on the selection half, and the setting with the highest segment F1 there is not one of them.
- **The view test.** On the selection half no sample changes. On each half the test drops fewer than 10 peaks over all streams.
- **The other decoders.** For the Viterbi path at least 4 of its 8 costs, the chosen one among them, lie within 0.02 of the chosen one's segment F1 on the selection half.
- **Lower temperatures.** The fit takes 0.25 or a lower value. Between the fitted value and 0.30 the log loss on the calibration half differs by less than 0.002. With the published decoder setting, segment F1 on the multi streams is lower under the temperature than as published. With the rule's own setting it is at most 0.01 lower.

### What follows from part 3

Section 6 holds for part 3 as well. Nothing changes in the published system, a candidate needs its own plan and other data, and everything is published whatever it shows.

## 11. Results of part 3

Not run yet.
