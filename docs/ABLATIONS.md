# What the parts add, and what the system stands: studies on development data

**Status: planned on 2026-11-26 (UTC). Parts 1 and 2 were run on the same day (section 9; the text of part 1 was corrected once). Part 3 was planned after part 1 was read (section 10) and run the same day (section 11).** Everything in this document down to "Results" was written and committed before any of its numbers existed, and section 10 before any number of part 3.

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

**Run on 2026-11-26 from 03:36 to 07:43 UTC and from 20:47 to 22:19 UTC with the code of commit `fb674f6`, which is the plan.** The server stopped in between, and a condition whose file existed was not computed again. The output of the report is [`ablations/forward.json`](ablations/forward.json). The unchanged frames, run again, gave the stored outputs to less than 0.000001 in every probability and the published number of the selection half, 0.9594.

The streams and the single recordings are those of the selection half. The recordings of the ninth view are two lists of their own, and they are set against the 521 single recordings of both halves. Intervals are at 95% and draw streams for the numbers about streams and recordings for the numbers about recordings. They are many, and none is corrected for their number. An end printed as +0.000 or -0.000 can be a number just off zero; the file has every digit. Part 2 describes and gets no word.

**The published system under every condition.**

| Condition | Segment F1 at 50%, multi streams | Minus the unchanged frames | Pair streams, minus unchanged | Recordings right of 284, majority rule | Wrong by `classify`'s rule | Mean probability | Recordings in which the seeds differ |
|---|---|---|---|---|---|---|---|
| unchanged | 0.959 |  |  | 281 | 1 | 0.963 | 0 |
| gamma 0.5 | 0.935 | -0.025 [-0.041, -0.010] | -0.000 [-0.007, +0.007] | 281 | 2 | 0.962 | 1 |
| gamma 0.7 | 0.952 | -0.008 [-0.016, -0.001] | -0.000 [-0.007, +0.007] | 281 | 1 | 0.963 | 1 |
| gamma 1.5 | 0.959 | +0.000 [-0.005, +0.006] | -0.002 [-0.008, +0.000] | 281 | 1 | 0.963 | 0 |
| gamma 2.0 | 0.949 | -0.011 [-0.024, -0.000] | -0.002 [-0.008, +0.000] | 280 | 2 | 0.962 | 2 |
| darker by 40 | 0.964 | +0.005 [-0.007, +0.016] | -0.000 [-0.007, +0.007] | 281 | 2 | 0.961 | 1 |
| brighter by 40 | 0.962 | +0.003 [-0.007, +0.014] | -0.002 [-0.008, +0.000] | 281 | 1 | 0.955 | 1 |
| contrast 0.5 | 0.964 | +0.005 [-0.012, +0.022] | -0.008 [-0.029, +0.009] | 280 | 3 | 0.949 | 1 |
| contrast 1.5 | 0.958 | -0.001 [-0.008, +0.006] | -0.003 [-0.011, +0.005] | 282 | 1 | 0.961 | 1 |
| noise 10 | 0.964 | +0.004 [-0.010, +0.018] | -0.007 [-0.026, +0.007] | 281 | 1 | 0.956 | 1 |
| noise 25 | 0.922 | -0.037 [-0.063, -0.012] | -0.023 [-0.049, -0.000] | 274 | 9 | 0.890 | 6 |
| speckle 0.25 | 0.965 | +0.005 [-0.012, +0.023] | -0.007 [-0.029, +0.009] | 277 | 8 | 0.936 | 2 |
| blur 1 pixel | 0.940 | -0.020 [-0.033, -0.008] | -0.007 [-0.022, +0.000] | 280 | 1 | 0.963 | 0 |
| blur 2 pixels | 0.928 | -0.032 [-0.049, -0.016] | -0.012 [-0.030, +0.000] | 281 | 1 | 0.958 | 1 |
| JPEG quality 30 | 0.945 | -0.015 [-0.030, -0.003] | +0.002 [+0.000, +0.007] | 281 | 1 | 0.964 | 0 |
| JPEG quality 10 | 0.937 | -0.023 [-0.039, -0.009] | -0.002 [-0.008, +0.000] | 281 | 1 | 0.964 | 0 |
| 112 pixels | 0.938 | -0.022 [-0.036, -0.010] | -0.007 [-0.022, +0.000] | 281 | 1 | 0.963 | 0 |
| 64 pixels | 0.930 | -0.030 [-0.046, -0.015] | -0.012 [-0.030, +0.000] | 281 | 1 | 0.960 | 0 |
| mirrored | 0.730 | -0.230 [-0.270, -0.192] | -0.093 [-0.138, -0.053] | 264 | 18 | 0.890 | 11 |
| zoomed in 1.2 | 0.964 | +0.005 [-0.001, +0.013] | +0.000 [+0.000, +0.000] | 280 | 2 | 0.962 | 1 |
| zoomed out 0.8 | 0.926 | -0.033 [-0.049, -0.019] | +0.002 [+0.000, +0.007] | 281 | 2 | 0.962 | 2 |
| rotated 10 degrees | 0.954 | -0.006 [-0.017, +0.002] | -0.009 [-0.029, +0.000] | 283 | 1 | 0.961 | 0 |
| rotated 20 degrees | 0.940 | -0.020 [-0.038, -0.002] | -0.029 [-0.059, -0.005] | 279 | 3 | 0.949 | 5 |
| still frame | 0.824 | -0.136 [-0.174, -0.099] | -0.084 [-0.133, -0.040] | 279 | 4 | 0.881 | 7 |
| shuffled frames | 0.933 | -0.026 [-0.044, -0.010] | -0.005 [-0.014, +0.000] | 281 | 2 | 0.964 | 1 |
| played backwards | 0.927 | -0.032 [-0.054, -0.011] | +0.007 [+0.000, +0.019] | 282 | 1 | 0.964 | 1 |
| half the frame rate | 0.913 | -0.047 [-0.068, -0.027] | -0.003 [-0.025, +0.019] | 284 | 1 | 0.959 | 1 |

- **A mirrored image is the largest loss on the multi streams.** Segment F1 falls from 0.959 to 0.730, which is -0.230 [-0.270, -0.192]; the next largest loss is 0.136. By the majority rule 20 of the 284 single recordings are read wrong, where 3 are wrong on the unchanged frames, and by `classify`'s rule 18, where 1 is. Their macro-F1 falls from 0.979 to 0.793. No training stage mirrored a frame; part 2 does not test whether that is the cause. The stage-2 ensemble loses 0.214 under it (-0.214 [-0.251, -0.177]).
- **Windows that show one frame are the second largest.** When every window shows its centre frame sixteen times, the multi streams lose 0.136 (-0.136 [-0.174, -0.099]). Of the single recordings 279 of 284 are right by the majority rule, where 281 are on the unchanged frames, and 4 are wrong by `classify`'s rule, where 1 is. Inside the single recordings the decoded view changes 103 times, where it changes 14 times, the mean probability falls from 0.963 to 0.881, and the seeds differ in 7 recordings, where they differ in 0.
- **Shuffled frames, frames played backwards and half the frame rate each cost the multi streams something.** They cost 0.026, 0.032 and 0.047, and each interval lies below zero. Only for the shuffled frames is the order the one thing that changes, with the marker staying on the centre frame (section 4). A stream played backwards also has its windows at other places, and half the frame rate puts a cut up to 0.1 s late by construction. For the pair streams no interval of the three lies below zero. By the majority rule no more single recordings are read wrong than on the unchanged frames. By `classify`'s rule 2 are wrong under the shuffled frames, where 1 is wrong on the unchanged frames and under the other two.
- **Grey levels.** The interval lies below zero for gamma 0.5 and gamma 0.7, and the larger of the two losses is 0.025. For gamma 2.0 it ends at zero to three decimals (-0.011 [-0.024, -0.000]). For the other five conditions of grey levels it includes zero.
- **Degradation.** Noise with a spread of 25 grey levels, blur of 1 and of 2 pixels, JPEG at quality 30 and 10, and 112 and 64 pixels cost 0.015 to 0.037, each with an interval below zero. For noise with a spread of 10 and for speckle the interval includes zero. For the single recordings the strong noise and the speckle are harder than the multi streams show: `classify` reads 9 and 8 wrong under them.
- **Geometry other than mirroring.** The interval lies below zero for zoomed out 0.8 and rotated 20 degrees (0.033 and 0.020), and includes zero for zoomed in 1.2 and rotated 10 degrees.
- **For segment F1 on the multi streams no interval lies above zero.** Of the 26 conditions, 16 have an interval below zero, one has an interval that ends at zero to three decimals, and 9 have one that includes zero. Under 6 conditions the value is above that of the unchanged frames, by 0.003 to 0.005. After the mirrored image and the still frame the largest loss is 0.047. Other numbers do have intervals on the better side under a condition, 9 in all: for example fewer segments that match nothing under noise with a spread of 10 and under speckle (the tables of every number).

**The stage-2 ensemble under every condition.** Its decoder setting is the one that the rule gives it on the calibration half: half window 7, threshold 0.185, the 0.999 quantile. The floors were kept.

| Condition | Segment F1 at 50%, multi streams | Minus the unchanged frames | Pair streams, minus unchanged | Recordings right of 284, majority rule | Wrong by `classify`'s rule | Mean probability | Recordings in which the seeds differ |
|---|---|---|---|---|---|---|---|
| unchanged | 0.962 |  |  | 281 | 1 | 0.963 | 0 |
| gamma 0.5 | 0.934 | -0.029 [-0.045, -0.015] | -0.005 [-0.019, +0.007] | 282 | 1 | 0.953 | 1 |
| gamma 0.7 | 0.946 | -0.017 [-0.030, -0.006] | -0.000 [-0.007, +0.007] | 281 | 1 | 0.963 | 1 |
| gamma 1.5 | 0.964 | +0.002 [-0.007, +0.012] | +0.007 [-0.007, +0.029] | 281 | 1 | 0.961 | 2 |
| gamma 2.0 | 0.953 | -0.009 [-0.023, +0.004] | +0.004 [-0.010, +0.027] | 281 | 2 | 0.958 | 3 |
| darker by 40 | 0.951 | -0.011 [-0.029, +0.005] | -0.012 [-0.039, +0.016] | 280 | 2 | 0.950 | 2 |
| brighter by 40 | 0.953 | -0.009 [-0.030, +0.012] | -0.010 [-0.040, +0.019] | 279 | 5 | 0.905 | 5 |
| contrast 0.5 | 0.918 | -0.044 [-0.070, -0.019] | -0.025 [-0.058, +0.001] | 277 | 7 | 0.882 | 7 |
| contrast 1.5 | 0.957 | -0.005 [-0.021, +0.010] | -0.008 [-0.027, +0.014] | 279 | 3 | 0.947 | 3 |
| noise 10 | 0.941 | -0.022 [-0.042, -0.002] | -0.008 [-0.034, +0.020] | 279 | 5 | 0.916 | 3 |
| noise 25 | 0.609 | -0.353 [-0.410, -0.299] | -0.171 [-0.223, -0.121] | 263 | 20 | 0.749 | 41 |
| speckle 0.25 | 0.869 | -0.094 [-0.134, -0.056] | -0.047 [-0.083, -0.015] | 270 | 14 | 0.863 | 10 |
| blur 1 pixel | 0.939 | -0.023 [-0.039, -0.009] | -0.012 [-0.031, +0.002] | 280 | 1 | 0.949 | 0 |
| blur 2 pixels | 0.674 | -0.289 [-0.339, -0.240] | -0.114 [-0.164, -0.067] | 274 | 7 | 0.783 | 13 |
| JPEG quality 30 | 0.947 | -0.016 [-0.028, -0.006] | +0.007 [-0.007, +0.029] | 281 | 1 | 0.964 | 0 |
| JPEG quality 10 | 0.936 | -0.027 [-0.043, -0.013] | +0.012 [-0.008, +0.039] | 280 | 1 | 0.963 | 0 |
| 112 pixels | 0.933 | -0.029 [-0.048, -0.012] | -0.012 [-0.031, +0.002] | 281 | 1 | 0.934 | 0 |
| 64 pixels | 0.751 | -0.211 [-0.256, -0.168] | -0.065 [-0.106, -0.029] | 278 | 3 | 0.814 | 10 |
| mirrored | 0.749 | -0.214 [-0.251, -0.177] | -0.097 [-0.140, -0.058] | 264 | 18 | 0.874 | 21 |
| zoomed in 1.2 | 0.957 | -0.005 [-0.016, +0.004] | +0.007 [-0.010, +0.030] | 280 | 2 | 0.958 | 1 |
| zoomed out 0.8 | 0.924 | -0.038 [-0.057, -0.021] | +0.004 [-0.010, +0.028] | 280 | 2 | 0.958 | 1 |
| rotated 10 degrees | 0.958 | -0.004 [-0.015, +0.005] | +0.000 [+0.000, +0.000] | 281 | 1 | 0.958 | 0 |
| rotated 20 degrees | 0.930 | -0.032 [-0.053, -0.012] | -0.015 [-0.037, +0.002] | 278 | 5 | 0.937 | 8 |
| still frame | 0.805 | -0.157 [-0.197, -0.121] | -0.072 [-0.118, -0.028] | 269 | 12 | 0.756 | 20 |
| shuffled frames | 0.931 | -0.031 [-0.051, -0.013] | -0.002 [-0.015, +0.009] | 281 | 1 | 0.962 | 1 |
| played backwards | 0.935 | -0.028 [-0.047, -0.010] | +0.000 [-0.007, +0.007] | 283 | 1 | 0.964 | 2 |
| half the frame rate | 0.898 | -0.065 [-0.089, -0.042] | -0.010 [-0.037, +0.018] | 282 | 1 | 0.942 | 0 |

**The published system minus the stage-2 ensemble,** with the same draws.

| Condition | Segment F1 at 50%, multi streams | Pair streams | Accuracy, single recordings | Wrong samples, percentage points |
|---|---|---|---|---|
| unchanged | -0.003 [-0.009, +0.003] | +0.007 [-0.007, +0.029] | +0.000 [+0.000, +0.000] | +0.2 [-0.2, +0.7] |
| gamma 0.5 | +0.001 [-0.013, +0.016] | +0.012 [+0.000, +0.030] | -0.004 [-0.011, +0.000] | -0.6 [-1.2, +0.1] |
| gamma 0.7 | +0.006 [-0.007, +0.021] | +0.007 [-0.007, +0.029] | +0.000 [+0.000, +0.000] | -0.5 [-1.2, +0.2] |
| gamma 1.5 | -0.005 [-0.014, +0.004] | -0.002 [-0.010, +0.005] | +0.000 [+0.000, +0.000] | +0.3 [-0.4, +1.1] |
| gamma 2.0 | -0.005 [-0.019, +0.008] | +0.000 [-0.010, +0.010] | -0.004 [-0.018, +0.007] | +0.5 [-0.5, +1.5] |
| darker by 40 | +0.013 [+0.002, +0.027] | +0.019 [+0.004, +0.041] | +0.004 [+0.000, +0.011] | -0.9 [-1.5, -0.4] |
| brighter by 40 | +0.009 [-0.011, +0.029] | +0.015 [-0.002, +0.037] | +0.007 [+0.000, +0.018] | -1.0 [-2.1, +0.0] |
| contrast 0.5 | +0.046 [+0.023, +0.070] | +0.025 [+0.007, +0.048] | +0.011 [+0.000, +0.025] | -3.6 [-5.1, -2.2] |
| contrast 1.5 | +0.001 [-0.013, +0.016] | +0.012 [-0.016, +0.038] | +0.011 [+0.000, +0.025] | -0.3 [-1.2, +0.6] |
| noise 10 | +0.023 [+0.009, +0.039] | +0.007 [-0.015, +0.031] | +0.007 [+0.000, +0.018] | -1.6 [-2.5, -0.7] |
| noise 25 | +0.313 [+0.258, +0.371] | +0.155 [+0.108, +0.204] | +0.039 [+0.018, +0.063] | -21.4 [-25.0, -18.0] |
| speckle 0.25 | +0.096 [+0.064, +0.132] | +0.047 [+0.020, +0.077] | +0.025 [+0.007, +0.042] | -6.9 [-9.3, -4.7] |
| blur 1 pixel | +0.000 [-0.010, +0.012] | +0.012 [-0.005, +0.036] | +0.000 [+0.000, +0.000] | -0.8 [-1.7, -0.0] |
| blur 2 pixels | +0.254 [+0.206, +0.305] | +0.109 [+0.066, +0.155] | +0.025 [+0.007, +0.042] | -16.9 [-20.6, -13.4] |
| JPEG quality 30 | -0.002 [-0.015, +0.011] | +0.002 [-0.005, +0.012] | +0.000 [+0.000, +0.000] | -0.6 [-1.5, +0.2] |
| JPEG quality 10 | +0.001 [-0.013, +0.015] | -0.007 [-0.029, +0.007] | +0.004 [+0.000, +0.011] | -0.5 [-1.4, +0.3] |
| 112 pixels | +0.005 [-0.007, +0.018] | +0.012 [-0.005, +0.036] | +0.000 [+0.000, +0.000] | -1.0 [-2.2, -0.1] |
| 64 pixels | +0.179 [+0.137, +0.221] | +0.060 [+0.027, +0.098] | +0.011 [+0.000, +0.025] | -10.9 [-13.4, -8.3] |
| mirrored | -0.019 [-0.044, +0.005] | +0.011 [-0.015, +0.038] | +0.000 [-0.021, +0.018] | +1.1 [-0.8, +3.0] |
| zoomed in 1.2 | +0.007 [-0.003, +0.019] | +0.000 [-0.010, +0.010] | +0.000 [+0.000, +0.000] | -0.9 [-2.0, -0.0] |
| zoomed out 0.8 | +0.002 [-0.008, +0.012] | +0.005 [+0.000, +0.013] | +0.004 [+0.000, +0.011] | -0.2 [-0.8, +0.5] |
| rotated 10 degrees | -0.005 [-0.014, +0.005] | -0.002 [-0.007, +0.000] | +0.007 [+0.000, +0.018] | -0.1 [-0.6, +0.4] |
| rotated 20 degrees | +0.009 [-0.003, +0.022] | -0.007 [-0.016, +0.000] | +0.004 [+0.000, +0.011] | -0.4 [-1.4, +0.7] |
| still frame | +0.018 [-0.019, +0.059] | -0.005 [-0.042, +0.031] | +0.035 [+0.014, +0.060] | -5.5 [-7.9, -3.4] |
| shuffled frames | +0.002 [-0.010, +0.015] | +0.005 [-0.007, +0.021] | +0.000 [-0.011, +0.011] | +0.4 [-0.5, +1.6] |
| played backwards | -0.007 [-0.019, +0.002] | +0.014 [+0.000, +0.037] | -0.004 [-0.011, +0.000] | +0.1 [-0.3, +0.7] |
| half the frame rate | +0.015 [+0.000, +0.031] | +0.015 [+0.000, +0.034] | +0.007 [+0.000, +0.018] | -0.8 [-1.7, +0.1] |

- **On the unchanged frames the stage-2 ensemble has the higher value.** It reaches 0.962 where the published system reaches 0.959: -0.003 [-0.009, +0.003], an interval that includes zero.
- **Under five conditions the published system's segment F1 is the higher by 0.046 to 0.313.** Noise with a spread of 25 grey levels +0.313 [+0.258, +0.371], blur of 2 pixels +0.254 [+0.206, +0.305], 64 pixels +0.179 [+0.137, +0.221], speckle +0.096 [+0.064, +0.132] and a contrast of 0.5 +0.046 [+0.023, +0.070]. There the stage-2 ensemble falls to 0.609, 0.674, 0.751, 0.869 and 0.918.
- **Two more intervals lie above zero:** noise 10 (+0.023 [+0.009, +0.039]) and darker by 40 (+0.013 [+0.002, +0.027]). For half the frame rate the interval begins at zero to three decimals (+0.015 [+0.000, +0.031]).
- **For segment F1 on the multi streams no interval lies below zero.** Under the other 18 conditions it includes zero, the mirrored image among them. Other numbers do have intervals on the side of the stage-2 ensemble, 9 in all. Under the mirrored image and under the still frame the published system has the lower cut precision, more segments that match nothing and more label changes inside single recordings (the last table of part 2).
- **What this does not say.** Stage 3 is 1,500 further updates and its input changes together. Part 2 cannot tell the two apart (section 4).

**A view the system does not know.** The recordings of EV9V's ninth view, read as `echoview classify --fps 30` reads a folder.

| System | List | Recordings | Named PLAX | PSAX | A4C | A5C | SC4C | Probability: lowest | quarter | median | three quarters | highest | Above 0.9 | Seeds differ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| published | validation | 46 | 0 | 46 | 0 | 0 | 0 | 0.855 | 0.974 | 0.978 | 0.981 | 0.984 | 44 | 0 |
| published | training | 319 | 0 | 318 | 1 | 0 | 0 | 0.495 | 0.977 | 0.980 | 0.981 | 0.984 | 310 | 1 |
| stage-2 ensemble | validation | 46 | 0 | 46 | 0 | 0 | 0 | 0.840 | 0.966 | 0.973 | 0.979 | 0.983 | 43 | 0 |
| stage-2 ensemble | training | 319 | 0 | 318 | 1 | 0 | 0 | 0.615 | 0.973 | 0.977 | 0.980 | 0.984 | 311 | 0 |

The chance that a recording of the five views is surer than a recording of the ninth view, for three numbers. 0.5 means that the number does not tell them apart. The 521 single recordings of the five views were read by the same reader. The intervals draw recordings.

| System | List | The largest probability | Its margin over the second | The negative entropy |
|---|---|---|---|---|
| published | validation | 0.575 [0.506, 0.646] | 0.568 [0.499, 0.639] | 0.578 [0.509, 0.648] |
| published | training | 0.510 [0.472, 0.550] | 0.506 [0.467, 0.545] | 0.512 [0.473, 0.551] |
| stage-2 ensemble | validation | 0.685 [0.612, 0.755] | 0.678 [0.605, 0.750] | 0.688 [0.616, 0.758] |
| stage-2 ensemble | training | 0.613 [0.575, 0.650] | 0.608 [0.570, 0.646] | 0.616 [0.579, 0.653] |

- **The published system names the ninth view PSAX.** All 46 recordings of the validation split and 318 of the 319 of the training split get that name, and one is named A4C. 44 of the 46 and 310 of the 319 have a probability above 0.9, with a median of 0.978 and 0.980. The recordings of the five views have a median of 0.980.
- **The probability tells little.** For the validation list the chance is 0.575 [0.506, 0.646] and for the larger training list 0.510 [0.472, 0.550]. The margin and the negative entropy give nearly the same numbers. On the training list every interval includes 0.5. On the validation list that of the margin does, and the other two lie just above it. Section 2 quoted the 0.575 from the research records with an interval of 0.485 to 0.642, which includes 0.5. The two intervals are not the same, and this document does not hold how the records drew theirs.
- **For the stage-2 ensemble the chance is** 0.685 [0.612, 0.755] and 0.613 [0.575, 0.650]. Both intervals lie above 0.5.
- **One reader for both kinds.** The 521 single recordings of the five views, read as a folder, get the view that the benchmark's own frames give them in every case, and no probability differs by more than 0.000001.

**Every number of section 5.** The tables above show a part. Here is every number: for the published system and for the stage-2 ensemble each condition minus the unchanged frames, and then the published system minus the stage-2 ensemble. A count of confused recordings has an interval only where both counts are above 5.

The published system, each condition minus the unchanged frames:

| Condition | Segment F1, multi | Edit | Wrong samples, points | Cut precision | Cut recall | Cut distance, s | Cuts at same-view joins, points | Segments that match nothing | Segments missed | Segment F1, pairs | Accuracy | Macro-F1 | A4C as A5C | A5C as A4C | Label changes per recording |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gamma 0.5 | -0.025 [-0.041, -0.010] | -1.6 [-2.9, -0.4] | +1.2 [+0.4, +2.2] | -0.018 [-0.037, -0.000] | -0.010 [-0.024, +0.000] | +0.004 [-0.003, +0.012] | +0.9 [+0.0, +2.7] | +14 [+5, +25] | +11 [+4, +20] | -0.000 [-0.007, +0.007] | +0.000 [-0.011, +0.011] | -0.001 [-0.033, +0.028] | -1 | +1 | +0.000 [-0.021, +0.021] |
| gamma 0.7 | -0.008 [-0.016, -0.001] | -0.6 [-1.4, +0.1] | +0.4 [-0.1, +1.0] | -0.010 [-0.022, +0.002] | +0.000 [+0.000, +0.000] | -0.001 [-0.005, +0.004] | +0.0 [+0.0, +0.0] | +6 [+0, +13] | +2 [+0, +5] | -0.000 [-0.007, +0.007] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0 | +0 | +0.004 [-0.018, +0.025] |
| gamma 1.5 | +0.000 [-0.005, +0.006] | -0.1 [-0.8, +0.5] | +0.1 [-0.4, +0.6] | -0.005 [-0.019, +0.005] | -0.005 [-0.013, +0.000] | +0.002 [-0.003, +0.007] | +0.4 [+0.0, +1.4] | +0 [-4, +4] | +0 [-3, +3] | -0.002 [-0.008, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0 | +0 | -0.021 [-0.049, +0.000] |
| gamma 2.0 | -0.011 [-0.024, -0.000] | -1.4 [-2.8, -0.1] | +0.6 [-0.0, +1.3] | -0.010 [-0.026, +0.002] | -0.013 [-0.028, -0.003] | +0.002 [-0.004, +0.008] | +0.4 [+0.0, +1.4] | +5 [-1, +12] | +6 [+0, +14] | -0.002 [-0.008, +0.000] | -0.004 [-0.018, +0.007] | -0.011 [-0.047, +0.021] | +0 | +1 | -0.007 [-0.032, +0.011] |
| darker by 40 | +0.005 [-0.007, +0.016] | +0.2 [-0.7, +1.1] | -0.4 [-1.1, +0.1] | +0.004 [-0.005, +0.015] | -0.013 [-0.028, -0.003] | +0.004 [-0.002, +0.011] | -0.4 [-1.4, +0.0] | -6 [-12, -1] | +1 [-5, +9] | -0.000 [-0.007, +0.007] | +0.000 [-0.011, +0.011] | +0.000 [-0.027, +0.026] | +0 | +0 | -0.018 [-0.046, +0.004] |
| brighter by 40 | +0.003 [-0.007, +0.014] | +0.2 [-1.1, +1.6] | -0.2 [-1.0, +0.6] | -0.000 [-0.016, +0.015] | -0.008 [-0.020, +0.003] | +0.012 [+0.004, +0.020] | -0.4 [-1.4, +0.0] | -3 [-11, +4] | +0 [-5, +5] | -0.002 [-0.008, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0 | +0 | -0.007 [-0.032, +0.011] |
| contrast 0.5 | +0.005 [-0.012, +0.022] | +0.3 [-1.3, +1.9] | -0.1 [-1.0, +0.7] | -0.003 [-0.022, +0.016] | -0.016 [-0.036, +0.003] | +0.022 [+0.011, +0.033] | +0.0 [-1.3, +1.3] | -5 [-14, +3] | +0 [-11, +11] | -0.008 [-0.029, +0.009] | -0.004 [-0.011, +0.000] | -0.009 [-0.030, +0.000] | +1 | +0 | -0.021 [-0.046, +0.000] |
| contrast 1.5 | -0.001 [-0.008, +0.006] | -0.0 [-0.8, +0.8] | +0.3 [-0.3, +0.8] | -0.008 [-0.021, +0.005] | -0.005 [-0.013, +0.000] | +0.008 [+0.001, +0.016] | +0.4 [+0.0, +1.4] | +1 [-4, +6] | +0 [-3, +3] | -0.003 [-0.011, +0.005] | +0.004 [+0.000, +0.011] | +0.009 [+0.000, +0.032] | -1 | +0 | -0.014 [-0.042, +0.007] |
| noise 10 | +0.004 [-0.010, +0.018] | +0.3 [-1.0, +1.7] | -0.3 [-1.1, +0.4] | +0.015 [+0.002, +0.029] | -0.018 [-0.034, -0.005] | +0.010 [+0.002, +0.019] | -0.4 [-1.4, +0.0] | -9 [-17, -2] | +4 [-4, +13] | -0.007 [-0.026, +0.007] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0 | +0 | -0.025 [-0.053, -0.004] |
| noise 25 | -0.037 [-0.063, -0.012] | -4.6 [-7.0, -2.2] | +3.1 [+1.7, +4.6] | -0.021 [-0.048, +0.007] | -0.102 [-0.134, -0.071] | +0.055 [+0.038, +0.073] | +0.9 [-1.6, +3.3] | +1 [-12, +13] | +34 [+19, +50] | -0.023 [-0.049, -0.000] | -0.025 [-0.042, -0.007] | -0.050 [-0.090, -0.018] | +7 | +0 | -0.039 [-0.067, -0.014] |
| speckle 0.25 | +0.005 [-0.012, +0.023] | -0.1 [-2.0, +1.7] | +0.1 [-1.0, +1.2] | +0.020 [+0.002, +0.039] | -0.021 [-0.040, -0.003] | +0.007 [-0.004, +0.018] | +0.0 [-1.7, +1.8] | -11 [-21, -2] | +5 [-5, +15] | -0.007 [-0.029, +0.009] | -0.014 [-0.028, -0.004] | -0.031 [-0.067, -0.007] | +4 | +0 | -0.014 [-0.039, +0.007] |
| blur 1 pixel | -0.020 [-0.033, -0.008] | -0.9 [-1.7, -0.3] | +1.0 [+0.3, +1.8] | -0.012 [-0.027, +0.002] | -0.003 [-0.012, +0.005] | -0.001 [-0.007, +0.006] | +0.4 [+0.0, +1.4] | +12 [+5, +21] | +8 [+3, +15] | -0.007 [-0.022, +0.000] | -0.004 [-0.011, +0.000] | -0.011 [-0.038, +0.000] | +0 | +1 | +0.004 [+0.000, +0.011] |
| blur 2 pixels | -0.032 [-0.049, -0.016] | -2.2 [-3.6, -1.0] | +2.4 [+1.1, +3.9] | -0.015 [-0.033, +0.002] | -0.010 [-0.024, +0.000] | +0.002 [-0.005, +0.009] | +0.4 [+0.0, +1.4] | +17 [+8, +28] | +15 [+8, +24] | -0.012 [-0.030, +0.000] | +0.000 [-0.011, +0.011] | -0.001 [-0.032, +0.027] | -1 | +1 | +0.007 [-0.014, +0.028] |
| JPEG quality 30 | -0.015 [-0.030, -0.003] | -0.9 [-2.1, +0.1] | +0.4 [-0.1, +1.1] | -0.015 [-0.035, +0.003] | -0.003 [-0.012, +0.005] | -0.002 [-0.006, +0.002] | +0.9 [+0.0, +2.3] | +10 [+1, +21] | +5 [+1, +11] | +0.002 [+0.000, +0.007] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0 | +0 | -0.004 [-0.021, +0.011] |
| JPEG quality 10 | -0.023 [-0.039, -0.009] | -1.8 [-3.3, -0.6] | +1.5 [+0.4, +2.9] | -0.012 [-0.033, +0.007] | -0.005 [-0.016, +0.005] | -0.004 [-0.009, +0.000] | +0.9 [+0.0, +2.3] | +13 [+4, +24] | +10 [+4, +17] | -0.002 [-0.008, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0 | +0 | -0.011 [-0.035, +0.007] |
| 112 pixels | -0.022 [-0.036, -0.010] | -1.1 [-2.0, -0.1] | +1.2 [+0.3, +2.3] | -0.008 [-0.023, +0.008] | -0.008 [-0.020, +0.003] | -0.002 [-0.008, +0.005] | +0.4 [+0.0, +1.4] | +11 [+4, +19] | +11 [+4, +19] | -0.007 [-0.022, +0.000] | +0.000 [-0.011, +0.011] | -0.001 [-0.032, +0.027] | -1 | +1 | +0.000 [-0.011, +0.011] |
| 64 pixels | -0.030 [-0.046, -0.015] | -2.0 [-3.3, -0.8] | +2.2 [+0.9, +3.6] | -0.015 [-0.033, +0.002] | -0.010 [-0.024, +0.000] | +0.000 [-0.006, +0.007] | +0.4 [+0.0, +1.4] | +16 [+7, +26] | +14 [+7, +22] | -0.012 [-0.030, +0.000] | +0.000 [-0.011, +0.011] | -0.001 [-0.032, +0.027] | -1 | +1 | +0.000 [-0.018, +0.014] |
| mirrored | -0.230 [-0.270, -0.192] | -21.8 [-25.3, -18.5] | +17.7 [+14.5, +21.1] | -0.093 [-0.129, -0.057] | -0.131 [-0.172, -0.091] | +0.042 [+0.026, +0.059] | +6.1 [+2.7, +9.8] | +106 [+85, +127] | +122 [+100, +146] | -0.093 [-0.138, -0.053] | -0.060 [-0.092, -0.032] | -0.186 [-0.270, -0.112] | +0 | +3 | +0.018 [-0.007, +0.042] |
| zoomed in 1.2 | +0.005 [-0.001, +0.013] | +0.3 [-0.4, +1.1] | -0.5 [-1.1, -0.0] | +0.005 [-0.005, +0.015] | +0.003 [+0.000, +0.008] | -0.001 [-0.006, +0.004] | -0.4 [-1.4, +0.0] | -3 [-9, +2] | -2 [-5, +0] | +0.000 [+0.000, +0.000] | -0.004 [-0.011, +0.000] | -0.009 [-0.030, +0.000] | +1 | +0 | -0.011 [-0.039, +0.018] |
| zoomed out 0.8 | -0.033 [-0.049, -0.019] | -2.5 [-4.0, -1.3] | +2.5 [+1.2, +4.1] | -0.022 [-0.043, -0.003] | -0.013 [-0.027, +0.000] | +0.005 [-0.004, +0.015] | +0.9 [-0.8, +2.7] | +19 [+10, +29] | +15 [+8, +24] | +0.002 [+0.000, +0.007] | +0.000 [-0.014, +0.014] | -0.003 [-0.050, +0.041] | -2 | +2 | -0.007 [-0.028, +0.011] |
| rotated 10 degrees | -0.006 [-0.017, +0.002] | -0.1 [-0.9, +0.8] | +0.2 [-0.3, +0.6] | +0.000 [-0.010, +0.010] | +0.000 [-0.010, +0.011] | +0.003 [-0.004, +0.010] | +0.0 [+0.0, +0.0] | +3 [-1, +9] | +3 [-1, +9] | -0.009 [-0.029, +0.000] | +0.007 [+0.000, +0.018] | +0.019 [+0.000, +0.053] | -2 | +0 | -0.007 [-0.025, +0.007] |
| rotated 20 degrees | -0.020 [-0.038, -0.002] | -2.2 [-3.8, -0.6] | +1.8 [+0.7, +3.0] | -0.020 [-0.043, -0.000] | -0.016 [-0.033, +0.000] | +0.016 [+0.006, +0.026] | +0.4 [-0.9, +2.0] | +11 [+1, +22] | +9 [-1, +19] | -0.029 [-0.059, -0.005] | -0.007 [-0.021, +0.007] | -0.016 [-0.053, +0.016] | +2 | +0 | -0.011 [-0.032, +0.007] |
| still frame | -0.136 [-0.174, -0.099] | -11.7 [-15.2, -8.3] | +5.3 [+3.4, +7.3] | -0.185 [-0.232, -0.138] | -0.008 [-0.028, +0.011] | -0.022 [-0.032, -0.011] | +8.7 [+4.7, +13.1] | +121 [+87, +158] | +32 [+20, +45] | -0.084 [-0.133, -0.040] | -0.007 [-0.025, +0.011] | -0.008 [-0.054, +0.036] | -2 | +2 | +0.313 [+0.204, +0.440] |
| shuffled frames | -0.026 [-0.044, -0.010] | -2.1 [-4.0, -0.5] | +2.3 [+1.0, +3.8] | -0.042 [-0.068, -0.017] | -0.021 [-0.042, -0.003] | +0.026 [+0.013, +0.040] | +0.9 [-0.9, +3.2] | +18 [+8, +30] | +9 [+0, +18] | -0.005 [-0.014, +0.000] | +0.000 [-0.014, +0.014] | -0.003 [-0.050, +0.041] | -2 | +2 | +0.014 [-0.004, +0.039] |
| played backwards | -0.032 [-0.054, -0.011] | -2.7 [-4.4, -1.2] | +1.2 [+0.2, +2.3] | -0.049 [-0.076, -0.024] | +0.003 [-0.008, +0.014] | -0.000 [-0.016, +0.015] | +3.0 [+0.9, +5.7] | +28 [+13, +44] | +6 [-2, +14] | +0.007 [+0.000, +0.019] | +0.004 [-0.007, +0.014] | +0.008 [-0.027, +0.046] | -2 | +1 | -0.007 [-0.032, +0.018] |
| half the frame rate | -0.047 [-0.068, -0.027] | -4.0 [-6.2, -2.0] | +3.5 [+2.0, +5.2] | -0.036 [-0.061, -0.012] | -0.039 [-0.067, -0.016] | +0.029 [+0.015, +0.043] | +2.2 [+0.4, +4.6] | +23 [+11, +36] | +24 [+12, +38] | -0.003 [-0.025, +0.019] | +0.011 [+0.000, +0.025] | +0.021 [+0.000, +0.055] | -2 | +0 | +0.000 [-0.021, +0.021] |

The stage-2 ensemble, each condition minus the unchanged frames:

| Condition | Segment F1, multi | Edit | Wrong samples, points | Cut precision | Cut recall | Cut distance, s | Cuts at same-view joins, points | Segments that match nothing | Segments missed | Segment F1, pairs | Accuracy | Macro-F1 | A4C as A5C | A5C as A4C | Label changes per recording |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gamma 0.5 | -0.029 [-0.045, -0.015] | -2.8 [-4.3, -1.4] | +2.0 [+1.2, +2.9] | -0.026 [-0.049, -0.006] | -0.034 [-0.052, -0.018] | +0.014 [+0.006, +0.023] | +0.0 [+0.0, +0.0] | +13 [+4, +24] | +16 [+9, +24] | -0.005 [-0.019, +0.007] | +0.004 [+0.000, +0.011] | +0.009 [+0.000, +0.033] | -1 | +0 | +0.004 [-0.011, +0.021] |
| gamma 0.7 | -0.017 [-0.030, -0.006] | -1.4 [-2.6, -0.5] | +1.0 [+0.4, +1.8] | -0.013 [-0.026, -0.000] | -0.010 [-0.022, -0.002] | +0.005 [-0.003, +0.012] | +0.0 [+0.0, +0.0] | +9 [+2, +17] | +8 [+3, +15] | -0.000 [-0.007, +0.007] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0 | +0 | +0.007 [+0.000, +0.021] |
| gamma 1.5 | +0.002 [-0.007, +0.012] | +0.2 [-0.8, +1.3] | -0.0 [-0.7, +0.6] | +0.007 [-0.005, +0.021] | -0.008 [-0.018, +0.000] | +0.002 [-0.003, +0.007] | -0.9 [-2.2, +0.0] | -4 [-11, +2] | +2 [-3, +7] | +0.007 [-0.007, +0.029] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0 | +0 | -0.011 [-0.028, +0.000] |
| gamma 2.0 | -0.009 [-0.023, +0.004] | -0.5 [-1.7, +0.8] | +0.3 [-0.5, +1.1] | -0.003 [-0.021, +0.014] | -0.016 [-0.031, -0.003] | +0.005 [-0.003, +0.013] | -0.9 [-2.2, +0.0] | +2 [-7, +10] | +7 [+0, +16] | +0.004 [-0.010, +0.027] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0 | +0 | -0.018 [-0.042, +0.000] |
| darker by 40 | -0.011 [-0.029, +0.005] | -0.8 [-2.4, +0.9] | +0.7 [-0.2, +1.5] | -0.007 [-0.025, +0.013] | -0.039 [-0.061, -0.020] | +0.018 [+0.006, +0.031] | +0.0 [-1.7, +1.7] | -1 [-10, +8] | +12 [+3, +23] | -0.012 [-0.039, +0.016] | -0.004 [-0.011, +0.000] | -0.009 [-0.030, +0.000] | +1 | +0 | -0.007 [-0.021, +0.000] |
| brighter by 40 | -0.009 [-0.030, +0.012] | -1.4 [-3.2, +0.3] | +1.0 [-0.1, +2.2] | +0.014 [-0.008, +0.037] | -0.047 [-0.076, -0.020] | +0.028 [+0.014, +0.041] | -0.9 [-2.7, +0.9] | -8 [-19, +2] | +16 [+3, +30] | -0.010 [-0.040, +0.019] | -0.007 [-0.018, +0.000] | -0.016 [-0.045, +0.000] | +2 | +0 | -0.007 [-0.028, +0.011] |
| contrast 0.5 | -0.044 [-0.070, -0.019] | -5.0 [-7.3, -2.7] | +3.6 [+2.2, +5.2] | -0.002 [-0.027, +0.024] | -0.110 [-0.147, -0.074] | +0.032 [+0.017, +0.047] | -0.9 [-3.0, +1.3] | -1 [-13, +11] | +42 [+25, +60] | -0.025 [-0.058, +0.001] | -0.014 [-0.028, -0.004] | -0.031 [-0.066, -0.007] | +4 | +0 | -0.021 [-0.046, +0.000] |
| contrast 1.5 | -0.005 [-0.021, +0.010] | -0.8 [-2.3, +0.8] | +0.8 [-0.1, +1.6] | -0.006 [-0.026, +0.014] | -0.024 [-0.043, -0.008] | +0.022 [+0.009, +0.034] | +0.0 [-1.7, +1.7] | -1 [-10, +8] | +6 [-2, +16] | -0.008 [-0.027, +0.014] | -0.007 [-0.018, +0.000] | -0.016 [-0.045, +0.000] | +2 | +0 | +0.004 [-0.018, +0.025] |
| noise 10 | -0.022 [-0.042, -0.002] | -2.0 [-4.1, +0.1] | +1.5 [+0.3, +2.5] | +0.013 [-0.008, +0.035] | -0.063 [-0.090, -0.038] | +0.010 [+0.001, +0.019] | +0.0 [-1.7, +1.8] | -5 [-15, +4] | +25 [+13, +39] | -0.008 [-0.034, +0.020] | -0.007 [-0.018, +0.000] | -0.016 [-0.045, +0.000] | +2 | +0 | -0.039 [-0.077, -0.011] |
| noise 25 | -0.353 [-0.410, -0.299] | -35.5 [-39.9, -31.2] | +24.7 [+21.3, +28.4] | -0.016 [-0.054, +0.020] | -0.505 [-0.559, -0.453] | +0.061 [+0.037, +0.085] | -0.9 [-3.0, +1.3] | +41 [+23, +59] | +239 [+206, +273] | -0.171 [-0.223, -0.121] | -0.063 [-0.092, -0.035] | -0.120 [-0.178, -0.072] | +15 | +0 | -0.035 [-0.077, -0.004] |
| speckle 0.25 | -0.094 [-0.134, -0.056] | -10.7 [-14.3, -7.2] | +7.2 [+4.7, +9.8] | +0.013 [-0.009, +0.035] | -0.160 [-0.206, -0.115] | +0.028 [+0.013, +0.044] | +0.0 [-2.1, +2.1] | +9 [-5, +23] | +77 [+53, +103] | -0.047 [-0.083, -0.015] | -0.039 [-0.063, -0.018] | -0.072 [-0.119, -0.035] | +9 | +0 | -0.035 [-0.077, -0.004] |
| blur 1 pixel | -0.023 [-0.039, -0.009] | -1.4 [-2.8, +0.0] | +1.9 [+0.9, +3.1] | -0.003 [-0.022, +0.016] | -0.016 [-0.031, -0.003] | +0.010 [-0.000, +0.021] | +0.0 [-1.3, +1.3] | +9 [+0, +18] | +14 [+6, +23] | -0.012 [-0.031, +0.002] | -0.004 [-0.011, +0.000] | -0.011 [-0.038, +0.000] | +0 | +1 | +0.007 [-0.007, +0.025] |
| blur 2 pixels | -0.289 [-0.339, -0.240] | -29.1 [-33.3, -25.0] | +19.5 [+15.9, +23.2] | +0.010 [-0.019, +0.038] | -0.411 [-0.463, -0.358] | +0.015 [-0.003, +0.034] | -0.4 [-2.3, +1.4] | +36 [+20, +52] | +202 [+170, +235] | -0.114 [-0.164, -0.067] | -0.025 [-0.042, -0.007] | -0.048 [-0.099, -0.013] | +0 | +1 | -0.004 [-0.025, +0.014] |
| JPEG quality 30 | -0.016 [-0.028, -0.006] | -1.1 [-2.1, -0.3] | +1.2 [+0.5, +2.0] | -0.005 [-0.015, +0.004] | -0.010 [-0.022, -0.002] | +0.004 [-0.001, +0.010] | +0.4 [+0.0, +1.4] | +7 [+2, +14] | +9 [+3, +16] | +0.007 [-0.007, +0.029] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0 | +0 | +0.000 [+0.000, +0.000] |
| JPEG quality 10 | -0.027 [-0.043, -0.013] | -2.4 [-4.2, -0.6] | +2.2 [+0.9, +3.7] | -0.008 [-0.030, +0.015] | -0.010 [-0.023, +0.000] | -0.002 [-0.008, +0.004] | +0.4 [-0.9, +2.0] | +13 [+3, +24] | +14 [+7, +22] | +0.012 [-0.008, +0.039] | -0.004 [-0.011, +0.000] | -0.011 [-0.038, +0.000] | +0 | +1 | +0.000 [-0.021, +0.021] |
| 112 pixels | -0.029 [-0.048, -0.012] | -2.1 [-4.0, -0.3] | +2.4 [+1.1, +4.0] | -0.001 [-0.022, +0.020] | -0.029 [-0.049, -0.011] | +0.011 [+0.000, +0.023] | +0.0 [-1.7, +1.8] | +9 [-1, +19] | +20 [+10, +31] | -0.012 [-0.031, +0.002] | +0.000 [-0.011, +0.011] | -0.001 [-0.032, +0.027] | -1 | +1 | +0.004 [-0.014, +0.025] |
| 64 pixels | -0.211 [-0.256, -0.168] | -20.6 [-24.4, -16.8] | +13.2 [+10.7, +15.8] | +0.007 [-0.018, +0.033] | -0.304 [-0.352, -0.255] | +0.011 [-0.005, +0.028] | -0.4 [-1.9, +0.9] | +30 [+17, +44] | +153 [+125, +183] | -0.065 [-0.106, -0.029] | -0.011 [-0.025, +0.000] | -0.023 [-0.059, +0.000] | +0 | +1 | -0.004 [-0.025, +0.014] |
| mirrored | -0.214 [-0.251, -0.177] | -20.7 [-24.3, -17.2] | +16.8 [+13.7, +20.1] | -0.055 [-0.089, -0.020] | -0.162 [-0.205, -0.120] | +0.044 [+0.026, +0.063] | +2.6 [+0.0, +5.7] | +80 [+62, +98] | +125 [+103, +148] | -0.097 [-0.140, -0.058] | -0.060 [-0.088, -0.032] | -0.153 [-0.231, -0.086] | +0 | +3 | -0.007 [-0.042, +0.021] |
| zoomed in 1.2 | -0.005 [-0.016, +0.004] | +0.2 [-1.0, +1.6] | +0.6 [-0.2, +1.6] | +0.012 [-0.005, +0.031] | -0.005 [-0.016, +0.005] | +0.004 [-0.003, +0.012] | -0.4 [-1.4, +0.0] | -1 [-8, +6] | +6 [+1, +12] | +0.007 [-0.010, +0.030] | -0.004 [-0.011, +0.000] | -0.009 [-0.030, +0.000] | +1 | +0 | +0.000 [-0.028, +0.028] |
| zoomed out 0.8 | -0.038 [-0.057, -0.021] | -2.9 [-4.7, -1.1] | +2.9 [+1.6, +4.5] | -0.016 [-0.042, +0.009] | -0.031 [-0.050, -0.014] | +0.007 [-0.002, +0.016] | +0.4 [-1.3, +2.4] | +16 [+5, +28] | +22 [+13, +32] | +0.004 [-0.010, +0.028] | -0.004 [-0.018, +0.007] | -0.013 [-0.055, +0.022] | -1 | +2 | -0.004 [-0.021, +0.014] |
| rotated 10 degrees | -0.004 [-0.015, +0.005] | +0.0 [-1.1, +1.3] | +0.4 [-0.2, +1.1] | +0.005 [-0.008, +0.020] | -0.005 [-0.016, +0.005] | +0.003 [-0.003, +0.009] | -0.4 [-1.4, +0.0] | +0 [-7, +7] | +4 [-1, +10] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0 | +0 | +0.004 [-0.011, +0.021] |
| rotated 20 degrees | -0.032 [-0.053, -0.012] | -3.1 [-5.0, -1.3] | +2.4 [+1.3, +3.6] | -0.019 [-0.043, +0.006] | -0.034 [-0.057, -0.013] | +0.020 [+0.007, +0.034] | +0.0 [-1.7, +1.8] | +13 [+1, +26] | +19 [+8, +30] | -0.015 [-0.037, +0.002] | -0.011 [-0.028, +0.004] | -0.025 [-0.066, +0.010] | +2 | +0 | -0.007 [-0.028, +0.014] |
| still frame | -0.157 [-0.197, -0.121] | -15.1 [-18.8, -11.6] | +11.0 [+8.3, +14.0] | -0.092 [-0.128, -0.057] | -0.107 [-0.147, -0.071] | -0.024 [-0.038, -0.009] | +6.1 [+3.0, +9.5] | +76 [+55, +98] | +82 [+61, +105] | -0.072 [-0.118, -0.028] | -0.042 [-0.070, -0.018] | -0.057 [-0.124, -0.002] | -2 | +2 | +0.144 [+0.085, +0.208] |
| shuffled frames | -0.031 [-0.051, -0.013] | -2.2 [-3.5, -0.9] | +2.1 [+1.2, +3.1] | -0.040 [-0.064, -0.016] | -0.026 [-0.045, -0.008] | +0.026 [+0.013, +0.039] | +0.4 [-0.9, +2.0] | +19 [+7, +32] | +13 [+4, +22] | -0.002 [-0.015, +0.009] | +0.000 [-0.011, +0.011] | -0.001 [-0.032, +0.027] | -1 | +1 | +0.025 [+0.004, +0.049] |
| played backwards | -0.028 [-0.047, -0.010] | -3.1 [-4.6, -1.6] | +1.2 [+0.4, +2.1] | -0.040 [-0.064, -0.016] | +0.000 [-0.013, +0.013] | -0.001 [-0.016, +0.013] | +2.2 [+0.4, +4.2] | +23 [+10, +38] | +6 [-1, +13] | +0.000 [-0.007, +0.007] | +0.007 [+0.000, +0.018] | +0.019 [+0.000, +0.053] | -2 | +0 | +0.004 [-0.025, +0.032] |
| half the frame rate | -0.065 [-0.089, -0.042] | -5.6 [-7.9, -3.5] | +4.4 [+2.9, +6.2] | -0.033 [-0.055, -0.012] | -0.068 [-0.100, -0.039] | +0.033 [+0.018, +0.048] | +1.7 [+0.4, +3.6] | +25 [+14, +37] | +39 [+24, +56] | -0.010 [-0.037, +0.018] | +0.004 [+0.000, +0.011] | +0.009 [+0.000, +0.032] | -1 | +0 | +0.007 [-0.007, +0.021] |

The published system minus the stage-2 ensemble:

| Condition | Segment F1, multi | Edit | Wrong samples, points | Cut precision | Cut recall | Cut distance, s | Cuts at same-view joins, points | Segments that match nothing | Segments missed | Segment F1, pairs | Accuracy | Macro-F1 | A4C as A5C | A5C as A4C | Label changes per recording |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| unchanged | -0.003 [-0.009, +0.003] | -0.3 [-1.4, +0.8] | +0.2 [-0.2, +0.7] | -0.000 [-0.010, +0.013] | -0.003 [-0.008, +0.000] | -0.007 [-0.012, -0.002] | -0.4 [-1.4, +0.0] | +1 [-4, +6] | +2 [+0, +5] | +0.007 [-0.007, +0.029] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0 | +0 | +0.007 [+0.000, +0.021] |
| gamma 0.5 | +0.001 [-0.013, +0.016] | +0.9 [-0.4, +2.3] | -0.6 [-1.2, +0.1] | +0.009 [-0.011, +0.029] | +0.021 [+0.003, +0.040] | -0.017 [-0.025, -0.009] | +0.4 [+0.0, +1.4] | +2 [-8, +11] | -3 [-10, +6] | +0.012 [+0.000, +0.030] | -0.004 [-0.011, +0.000] | -0.011 [-0.039, +0.000] | +0 | +1 | +0.004 [-0.018, +0.025] |
| gamma 0.7 | +0.006 [-0.007, +0.021] | +0.6 [-0.5, +1.8] | -0.5 [-1.2, +0.2] | +0.003 [-0.012, +0.020] | +0.008 [+0.000, +0.018] | -0.012 [-0.019, -0.005] | -0.4 [-1.4, +0.0] | -2 [-12, +7] | -4 [-11, +2] | +0.007 [-0.007, +0.029] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0 | +0 | +0.004 [-0.018, +0.025] |
| gamma 1.5 | -0.005 [-0.014, +0.004] | -0.6 [-1.5, +0.2] | +0.3 [-0.4, +1.1] | -0.012 [-0.024, -0.003] | +0.000 [-0.011, +0.010] | -0.007 [-0.013, -0.000] | +0.9 [+0.0, +2.3] | +5 [+0, +11] | +0 [-5, +5] | -0.002 [-0.010, +0.005] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0 | +0 | -0.004 [-0.021, +0.011] |
| gamma 2.0 | -0.005 [-0.019, +0.008] | -1.2 [-2.8, +0.2] | +0.5 [-0.5, +1.5] | -0.007 [-0.022, +0.007] | +0.000 [-0.011, +0.010] | -0.010 [-0.018, -0.003] | +0.9 [+0.0, +2.3] | +4 [-4, +13] | +1 [-5, +8] | +0.000 [-0.010, +0.010] | -0.004 [-0.018, +0.007] | -0.011 [-0.047, +0.021] | +0 | +1 | +0.018 [+0.004, +0.039] |
| darker by 40 | +0.013 [+0.002, +0.027] | +0.7 [-0.3, +1.8] | -0.9 [-1.5, -0.4] | +0.011 [+0.000, +0.024] | +0.024 [+0.008, +0.041] | -0.021 [-0.031, -0.011] | -0.9 [-2.2, +0.0] | -4 [-10, +1] | -9 [-18, -2] | +0.019 [+0.004, +0.041] | +0.004 [+0.000, +0.011] | +0.009 [+0.000, +0.030] | -1 | +0 | -0.004 [-0.021, +0.011] |
| brighter by 40 | +0.009 [-0.011, +0.029] | +1.3 [-0.6, +3.3] | -1.0 [-2.1, +0.0] | -0.014 [-0.034, +0.004] | +0.037 [+0.011, +0.063] | -0.022 [-0.035, -0.009] | +0.0 [-1.3, +1.3] | +6 [-2, +15] | -14 [-28, -1] | +0.015 [-0.002, +0.037] | +0.007 [+0.000, +0.018] | +0.016 [+0.000, +0.045] | -2 | +0 | +0.007 [-0.007, +0.021] |
| contrast 0.5 | +0.046 [+0.023, +0.070] | +5.0 [+2.6, +7.5] | -3.6 [-5.1, -2.2] | -0.001 [-0.019, +0.017] | +0.092 [+0.058, +0.128] | -0.017 [-0.029, -0.004] | +0.4 [-0.9, +1.9] | -3 [-12, +7] | -40 [-57, -25] | +0.025 [+0.007, +0.048] | +0.011 [+0.000, +0.025] | +0.022 [+0.000, +0.052] | -3 | +0 | +0.007 [-0.007, +0.025] |
| contrast 1.5 | +0.001 [-0.013, +0.016] | +0.4 [-0.8, +1.8] | -0.3 [-1.2, +0.6] | -0.002 [-0.017, +0.013] | +0.016 [-0.003, +0.036] | -0.021 [-0.032, -0.010] | +0.0 [-1.7, +1.8] | +3 [-4, +10] | -4 [-14, +5] | +0.012 [-0.016, +0.038] | +0.011 [+0.000, +0.025] | +0.026 [+0.000, +0.062] | -3 | +0 | -0.011 [-0.035, +0.007] |
| noise 10 | +0.023 [+0.009, +0.039] | +2.0 [+0.6, +3.6] | -1.6 [-2.5, -0.7] | +0.001 [-0.013, +0.015] | +0.042 [+0.021, +0.064] | -0.006 [-0.015, +0.002] | -0.9 [-2.3, +0.0] | -3 [-9, +3] | -19 [-31, -9] | +0.007 [-0.015, +0.031] | +0.007 [+0.000, +0.018] | +0.016 [+0.000, +0.045] | -2 | +0 | +0.021 [+0.000, +0.056] |
| noise 25 | +0.313 [+0.258, +0.371] | +30.6 [+26.3, +35.1] | -21.4 [-25.0, -18.0] | -0.005 [-0.030, +0.022] | +0.401 [+0.344, +0.460] | -0.012 [-0.031, +0.006] | +1.3 [+0.0, +3.0] | -39 [-54, -25] | -203 [-238, -171] | +0.155 [+0.108, +0.204] | +0.039 [+0.018, +0.063] | +0.070 [+0.031, +0.120] | -8 [-14, -3] | +0 | +0.004 [-0.014, +0.025] |
| speckle 0.25 | +0.096 [+0.064, +0.132] | +10.3 [+6.9, +13.8] | -6.9 [-9.3, -4.7] | +0.007 [-0.009, +0.024] | +0.136 [+0.094, +0.181] | -0.028 [-0.042, -0.015] | -0.4 [-2.0, +0.9] | -19 [-29, -9] | -70 [-94, -48] | +0.047 [+0.020, +0.077] | +0.025 [+0.007, +0.042] | +0.041 [+0.014, +0.076] | -5 [-10, -1] | +0 | +0.028 [+0.000, +0.067] |
| blur 1 pixel | +0.000 [-0.010, +0.012] | +0.1 [-0.9, +1.2] | -0.8 [-1.7, -0.0] | -0.009 [-0.024, +0.005] | +0.010 [+0.000, +0.023] | -0.017 [-0.028, -0.007] | +0.0 [+0.0, +0.0] | +4 [-3, +12] | -4 [-11, +2] | +0.012 [-0.005, +0.036] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0 | +0 | +0.004 [-0.018, +0.025] |
| blur 2 pixels | +0.254 [+0.206, +0.305] | +26.6 [+22.5, +30.7] | -16.9 [-20.6, -13.4] | -0.025 [-0.051, +0.001] | +0.398 [+0.347, +0.449] | -0.021 [-0.038, -0.003] | +0.4 [-1.4, +2.3] | -18 [-35, -1] | -185 [-218, -154] | +0.109 [+0.066, +0.155] | +0.025 [+0.007, +0.042] | +0.047 [+0.012, +0.095] | -1 | +0 | +0.018 [-0.004, +0.042] |
| JPEG quality 30 | -0.002 [-0.015, +0.011] | -0.1 [-1.3, +1.3] | -0.6 [-1.5, +0.2] | -0.009 [-0.028, +0.010] | +0.005 [+0.000, +0.013] | -0.013 [-0.021, -0.005] | +0.0 [-1.3, +1.3] | +4 [-5, +14] | -2 [-7, +3] | +0.002 [-0.005, +0.012] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0 | +0 | +0.004 [+0.000, +0.011] |
| JPEG quality 10 | +0.001 [-0.013, +0.015] | +0.2 [-0.5, +1.1] | -0.5 [-1.4, +0.3] | -0.005 [-0.020, +0.010] | +0.003 [-0.005, +0.012] | -0.009 [-0.016, -0.002] | +0.0 [+0.0, +0.0] | +1 [-8, +10] | -2 [-9, +5] | -0.007 [-0.029, +0.007] | +0.004 [+0.000, +0.011] | +0.011 [+0.000, +0.038] | +0 | -1 | -0.004 [-0.025, +0.018] |
| 112 pixels | +0.005 [-0.007, +0.018] | +0.8 [-0.8, +2.4] | -1.0 [-2.2, -0.1] | -0.007 [-0.021, +0.008] | +0.018 [+0.003, +0.036] | -0.020 [-0.031, -0.010] | +0.0 [-1.3, +1.3] | +3 [-4, +10] | -7 [-16, +0] | +0.012 [-0.005, +0.036] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0 | +0 | +0.004 [-0.018, +0.025] |
| 64 pixels | +0.179 [+0.137, +0.221] | +18.3 [+14.6, +22.2] | -10.9 [-13.4, -8.3] | -0.023 [-0.046, +0.001] | +0.291 [+0.243, +0.337] | -0.018 [-0.034, -0.001] | +0.4 [-0.9, +1.9] | -13 [-26, +0] | -137 [-166, -110] | +0.060 [+0.027, +0.098] | +0.011 [+0.000, +0.025] | +0.021 [+0.000, +0.054] | -1 | +0 | +0.011 [-0.007, +0.032] |
| mirrored | -0.019 [-0.044, +0.005] | -1.4 [-3.6, +0.7] | +1.1 [-0.8, +3.0] | -0.038 [-0.066, -0.014] | +0.029 [+0.009, +0.050] | -0.009 [-0.023, +0.004] | +3.0 [+0.9, +5.8] | +27 [+13, +42] | -1 [-14, +12] | +0.011 [-0.015, +0.038] | +0.000 [-0.021, +0.018] | -0.033 [-0.093, +0.020] | +0 | +0 | +0.032 [+0.004, +0.070] |
| zoomed in 1.2 | +0.007 [-0.003, +0.019] | -0.2 [-1.2, +0.7] | -0.9 [-2.0, -0.0] | -0.007 [-0.023, +0.008] | +0.005 [+0.000, +0.013] | -0.012 [-0.020, -0.004] | -0.4 [-1.4, +0.0] | -1 [-9, +6] | -6 [-12, -1] | +0.000 [-0.010, +0.010] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0 | +0 | -0.004 [-0.021, +0.011] |
| zoomed out 0.8 | +0.002 [-0.008, +0.012] | +0.0 [-0.9, +1.0] | -0.2 [-0.8, +0.5] | -0.006 [-0.022, +0.009] | +0.016 [+0.005, +0.029] | -0.009 [-0.018, -0.001] | +0.0 [-1.3, +1.3] | +4 [-3, +11] | -5 [-11, +0] | +0.005 [+0.000, +0.013] | +0.004 [+0.000, +0.011] | +0.010 [+0.000, +0.035] | -1 | +0 | +0.004 [-0.018, +0.025] |
| rotated 10 degrees | -0.005 [-0.014, +0.005] | -0.4 [-1.4, +0.5] | -0.1 [-0.6, +0.4] | -0.005 [-0.018, +0.007] | +0.003 [-0.005, +0.012] | -0.007 [-0.013, -0.002] | +0.0 [+0.0, +0.0] | +4 [-2, +10] | +1 [-3, +6] | -0.002 [-0.007, +0.000] | +0.007 [+0.000, +0.018] | +0.019 [+0.000, +0.053] | -2 | +0 | -0.004 [-0.025, +0.014] |
| rotated 20 degrees | +0.009 [-0.003, +0.022] | +0.6 [-0.6, +1.9] | -0.4 [-1.4, +0.7] | -0.001 [-0.018, +0.015] | +0.016 [+0.000, +0.033] | -0.011 [-0.021, -0.002] | +0.0 [-1.3, +1.3] | -1 [-9, +6] | -8 [-15, -1] | -0.007 [-0.016, +0.000] | +0.004 [+0.000, +0.011] | +0.008 [+0.000, +0.030] | +0 | +0 | +0.004 [+0.000, +0.011] |
| still frame | +0.018 [-0.019, +0.059] | +3.1 [-0.7, +7.0] | -5.5 [-7.9, -3.4] | -0.093 [-0.133, -0.052] | +0.097 [+0.067, +0.131] | -0.005 [-0.014, +0.004] | +2.2 [-1.3, +5.6] | +46 [+16, +77] | -48 [-69, -29] | -0.005 [-0.042, +0.031] | +0.035 [+0.014, +0.060] | +0.049 [+0.015, +0.099] | +0 | +0 | +0.176 [+0.077, +0.292] |
| shuffled frames | +0.002 [-0.010, +0.015] | -0.3 [-1.7, +1.0] | +0.4 [-0.5, +1.6] | -0.002 [-0.017, +0.012] | +0.003 [-0.016, +0.021] | -0.007 [-0.016, +0.003] | +0.0 [+0.0, +0.0] | +0 [-7, +6] | -2 [-10, +5] | +0.005 [-0.007, +0.021] | +0.000 [-0.011, +0.011] | -0.002 [-0.037, +0.030] | -1 | +1 | -0.004 [-0.025, +0.018] |
| played backwards | -0.007 [-0.019, +0.002] | +0.0 [-0.7, +0.8] | +0.1 [-0.3, +0.7] | -0.009 [-0.022, +0.002] | +0.000 [-0.008, +0.008] | -0.006 [-0.011, -0.000] | +0.4 [+0.0, +1.4] | +6 [-1, +15] | +2 [-2, +6] | +0.014 [+0.000, +0.037] | -0.004 [-0.011, +0.000] | -0.011 [-0.039, +0.000] | +0 | +1 | -0.004 [-0.025, +0.014] |
| half the frame rate | +0.015 [+0.000, +0.031] | +1.3 [-0.0, +2.7] | -0.8 [-1.7, +0.1] | -0.003 [-0.020, +0.014] | +0.026 [+0.010, +0.046] | -0.011 [-0.020, -0.002] | +0.0 [-1.3, +1.3] | -1 [-9, +8] | -13 [-23, -5] | +0.015 [+0.000, +0.034] | +0.007 [+0.000, +0.018] | +0.012 [+0.000, +0.038] | -1 | +0 | +0.000 [-0.011, +0.011] |

**Against what was expected** (section 7).

- **Grey levels.** Small losses were expected. The largest is 0.025, the mean over the eight conditions is 0.004, and for 5 of the eight the interval includes zero.
- **Strong noise, 64 pixels and JPEG at quality 10.** Larger losses than under grey levels were expected. They cost 0.037, 0.030 and 0.023, each with an interval below zero. A gamma of 0.5 alone costs 0.025, more than the JPEG.
- **The mirrored image** was named as the open one. It is the largest loss on the multi streams.
- **The still frame.** Single recordings read nearly as well and streams read worse were expected. The streams lose 0.136. By the majority rule 2 more recordings are wrong, and by `classify`'s rule 3 more.
- **Shuffled frames** were expected to cost little. They cost 0.026, about what a gamma of 0.5 costs, with an interval below zero.
- **Stage 3 against stage 2.** The published system was expected to have the higher value under changes of grey levels and of image quality. Its interval lies above zero under 5 of the 9 conditions of degradation and under 2 of the 8 of grey levels. No difference on the unchanged frames was expected, and none is shown.
- **The ninth view.** Nearly all recordings named PSAX and most with a probability above 0.9 were expected, and that is so. A separation near 0.6 was expected. It is 0.575 on the validation list and 0.510 on the training list.

**What follows.** By section 6 a weakness under a condition is reported as a limit. The model card and the README now name two: mirrored images, and windows that show one frame. Training against a mirrored image would make another model, with a plan of its own.

## 10. Part 3: what part 1 left open

**Status: planned on 2026-11-26 (UTC), after part 1 was read and its text corrected, and before any number of part 3 existed. Run on the same day (section 11).**

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

**Run on 2026-11-26 from 22:24 to 22:27 UTC with the code of commit `7892e4d`, which holds the plan of section 10.** The output of the command is [`ablations/open.json`](ablations/open.json). The stored logits are those of part 1 and gave the published number of the selection half, 0.9594. The 40 settings and the 35 temperatures of part 1 came out again to the last digit. Intervals are at 95%.

**Thresholds above the grid.** The 0.999 quantile of the grid and the three added quantiles, for every half window.

| Half window | Quantile | Threshold | Calibration half: segment F1 | Macro-F1 | Boundary recall | Selection half: segment F1 | Macro-F1 | Boundary recall |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.999 | 0.141 | 0.981 | 0.993 | 0.980 | 0.947 | 0.979 | 0.955 |
| 5 | 0.9995 | 0.159 | 0.982 | 0.986 | 0.977 | 0.945 | 0.979 | 0.953 |
| 5 | 0.9999 | 0.248 | 0.941 | 0.986 | 0.894 | 0.889 | 0.979 | 0.859 |
| 5 | 1 | 0.344 | 0.803 | 0.986 | 0.686 | 0.785 | 0.998 | 0.675 |
| 7 | 0.999 | 0.174 | 0.984 | 0.986 | 0.990 | 0.959 | 0.979 | 0.976 |
| 7 | 0.9995 | 0.212 | 0.982 | 0.986 | 0.980 | 0.956 | 0.979 | 0.969 |
| 7 | 0.9999 | 0.284 | 0.973 | 0.986 | 0.962 | 0.922 | 0.988 | 0.911 |
| 7 | 1 | 0.387 | 0.835 | 0.986 | 0.729 | 0.807 | 0.998 | 0.712 |
| 9 | 0.999 | 0.222 | 0.981 | 0.986 | 0.985 | 0.958 | 0.979 | 0.976 |
| 9 | 0.9995 | 0.258 | 0.982 | 0.986 | 0.980 | 0.949 | 0.988 | 0.953 |
| 9 | 0.9999 | 0.325 | 0.976 | 0.986 | 0.967 | 0.933 | 0.988 | 0.927 |
| 9 | 1 | 0.412 | 0.869 | 0.986 | 0.781 | 0.826 | 0.998 | 0.741 |
| 11 | 0.999 | 0.236 | 0.975 | 0.986 | 0.985 | 0.953 | 0.979 | 0.974 |
| 11 | 0.9995 | 0.299 | 0.981 | 0.986 | 0.980 | 0.950 | 0.988 | 0.955 |
| 11 | 0.9999 | 0.341 | 0.980 | 0.986 | 0.972 | 0.946 | 0.988 | 0.937 |
| 11 | 1 | 0.415 | 0.920 | 0.986 | 0.849 | 0.865 | 0.998 | 0.801 |
| 15 | 0.999 | 0.282 | 0.968 | 0.986 | 0.967 | 0.941 | 0.998 | 0.958 |
| 15 | 0.9995 | 0.345 | 0.964 | 0.986 | 0.950 | 0.938 | 0.998 | 0.921 |
| 15 | 0.9999 | 0.391 | 0.958 | 0.986 | 0.930 | 0.930 | 0.998 | 0.895 |
| 15 | 1 | 0.396 | 0.952 | 0.986 | 0.920 | 0.925 | 0.998 | 0.890 |

- **The published setting stays.** Over the 55 settings the rule takes half window 7 at the 0.999 quantile again. 47 of the 55 keep the floors on the calibration half, and the 8 that do not are all added ones.
- **No added setting is above the published one, on either half.** On the selection half the highest of the 15 is 0.956, at half window 7 and the 0.9995 quantile, where the published setting has 0.959. For every half window each step above the 0.999 quantile lowers the value on the selection half.
- **At quantile 1 the decoder misses cuts.** There the boundary recall on the selection half is 0.675 to 0.890.
- **What this answers.** Part 1 could not say what a higher threshold would do. No added setting is above the published one on either half. On the calibration half the first step up is a little higher than the 0.999 quantile for half windows 5, 9 and 11, and still below the published setting. This was read on the two halves on which the decoder was developed.

**The halves exchanged, in full.** The 40 settings with thresholds from the selection half, where the rule chooses in this exchange.

| Half window | Quantile | Threshold | Selection half: segment F1 | Macro-F1 | Boundary recall | Keeps the floors | Calibration half: segment F1 | Macro-F1 | Boundary recall |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 0.8 | 0.00005 | 0.904 | 0.988 | 0.963 | yes | 0.975 | 0.993 | 0.982 |
| 5 | 0.85 | 0.00009 | 0.904 | 0.988 | 0.963 | yes | 0.975 | 0.993 | 0.982 |
| 5 | 0.9 | 0.00020 | 0.904 | 0.988 | 0.963 | yes | 0.975 | 0.993 | 0.982 |
| 5 | 0.95 | 0.00172 | 0.904 | 0.988 | 0.963 | yes | 0.975 | 0.993 | 0.982 |
| 5 | 0.97 | 0.00701 | 0.903 | 0.988 | 0.961 | yes | 0.975 | 0.993 | 0.982 |
| 5 | 0.99 | 0.05461 | 0.912 | 0.988 | 0.961 | yes | 0.979 | 0.993 | 0.982 |
| 5 | 0.995 | 0.09789 | 0.941 | 0.988 | 0.961 | yes | 0.981 | 0.993 | 0.982 |
| 5 | 0.999 | 0.21550 | 0.917 | 0.979 | 0.903 | no | 0.967 | 0.986 | 0.947 |
| 7 | 0.8 | 0.00007 | 0.913 | 0.988 | 0.984 | yes | 0.977 | 0.993 | 0.990 |
| 7 | 0.85 | 0.00011 | 0.913 | 0.988 | 0.984 | yes | 0.977 | 0.993 | 0.990 |
| 7 | 0.9 | 0.00029 | 0.913 | 0.988 | 0.984 | yes | 0.977 | 0.993 | 0.990 |
| 7 | 0.95 | 0.00269 | 0.913 | 0.988 | 0.984 | yes | 0.977 | 0.993 | 0.990 |
| 7 | 0.97 | 0.01115 | 0.913 | 0.988 | 0.984 | yes | 0.977 | 0.993 | 0.990 |
| 7 | 0.99 | 0.08681 | 0.923 | 0.988 | 0.982 | yes | 0.979 | 0.993 | 0.990 |
| 7 | 0.995 | 0.14384 | 0.949 | 0.979 | 0.976 | no | 0.982 | 0.986 | 0.990 |
| 7 | 0.999 | 0.26559 | 0.944 | 0.988 | 0.937 | no | 0.977 | 0.986 | 0.970 |
| 9 | 0.8 | 0.00008 | 0.916 | 0.988 | 0.984 | yes | 0.975 | 0.993 | 0.990 |
| 9 | 0.85 | 0.00014 | 0.916 | 0.988 | 0.984 | yes | 0.975 | 0.993 | 0.990 |
| 9 | 0.9 | 0.00042 | 0.916 | 0.988 | 0.984 | yes | 0.975 | 0.993 | 0.990 |
| 9 | 0.95 | 0.00398 | 0.916 | 0.988 | 0.984 | yes | 0.975 | 0.993 | 0.990 |
| 9 | 0.97 | 0.01627 | 0.916 | 0.988 | 0.984 | yes | 0.975 | 0.993 | 0.990 |
| 9 | 0.99 | 0.11301 | 0.928 | 0.988 | 0.979 | yes | 0.977 | 0.993 | 0.990 |
| 9 | 0.995 | 0.17814 | 0.948 | 0.979 | 0.976 | no | 0.980 | 0.986 | 0.990 |
| 9 | 0.999 | 0.30850 | 0.950 | 0.988 | 0.945 | no | 0.983 | 0.986 | 0.980 |
| 11 | 0.8 | 0.00009 | 0.918 | 0.988 | 0.984 | yes | 0.973 | 0.993 | 0.990 |
| 11 | 0.85 | 0.00017 | 0.918 | 0.988 | 0.984 | yes | 0.973 | 0.993 | 0.990 |
| 11 | 0.9 | 0.00053 | 0.918 | 0.988 | 0.984 | yes | 0.973 | 0.993 | 0.990 |
| 11 | 0.95 | 0.00534 | 0.918 | 0.988 | 0.984 | yes | 0.973 | 0.993 | 0.990 |
| 11 | 0.97 | 0.02137 | 0.918 | 0.988 | 0.984 | yes | 0.973 | 0.993 | 0.990 |
| 11 | 0.99 | 0.13077 | 0.941 | 0.968 | 0.979 | no | 0.975 | 0.993 | 0.990 |
| 11 | 0.995 | 0.19997 | 0.954 | 0.979 | 0.976 | no | 0.974 | 0.986 | 0.990 |
| 11 | 0.999 | 0.33659 | 0.944 | 0.988 | 0.937 | no | 0.980 | 0.986 | 0.972 |
| 15 | 0.8 | 0.00011 | 0.916 | 0.988 | 0.982 | yes | 0.975 | 0.993 | 0.990 |
| 15 | 0.85 | 0.00021 | 0.916 | 0.988 | 0.982 | yes | 0.975 | 0.993 | 0.990 |
| 15 | 0.9 | 0.00066 | 0.916 | 0.988 | 0.982 | yes | 0.975 | 0.993 | 0.990 |
| 15 | 0.95 | 0.00864 | 0.916 | 0.988 | 0.982 | yes | 0.975 | 0.993 | 0.990 |
| 15 | 0.97 | 0.03024 | 0.918 | 0.988 | 0.982 | yes | 0.975 | 0.993 | 0.990 |
| 15 | 0.99 | 0.14987 | 0.941 | 0.979 | 0.976 | no | 0.973 | 0.993 | 0.987 |
| 15 | 0.995 | 0.21879 | 0.940 | 0.979 | 0.971 | no | 0.971 | 0.986 | 0.987 |
| 15 | 0.999 | 0.37694 | 0.933 | 0.998 | 0.903 | no | 0.964 | 0.986 | 0.942 |

- **29 of the 40 settings keep the floors on the selection half.** The rule takes half window 5 with a threshold of 0.098, as part 1 reported. On the selection half that setting has 0.941, and on the calibration half 0.981.
- **The setting with the highest value on the selection half does not keep the floors.** It is half window 11 with a threshold of 0.200: segment F1 0.954, and a macro-F1 of single recordings of 0.979, below the floor of 0.980.
- **What this answers.** 8 settings have a higher value on the selection half than the rule's choice, and each has a higher threshold. None keeps the floors there: 5 fall below the floor for the macro-F1 of single recordings and 3 below the floor for the boundary recall. So with the halves exchanged it is the two floors that send the rule to a lower threshold. Part 1 could only suggest that.

**What the view test drops,** at the published setting.

| Half | Kind | Streams | Peaks | Dropped by the view test | Streams with a dropped peak | Streams in which a sample changes | Samples that change |
|---|---|---|---|---|---|---|---|
| calibration | single recordings | 237 | 4 | 0 | 0 | 0 | 0 |
| calibration | pair streams | 115 | 59 | 4 | 4 | 1 | 12 |
| calibration | multi streams | 120 | 415 | 10 | 10 | 0 | 0 |
| selection | single recordings | 284 | 15 | 0 | 0 | 0 | 0 |
| selection | pair streams | 137 | 81 | 6 | 6 | 0 | 0 |
| selection | multi streams | 120 | 407 | 13 | 13 | 0 | 0 |

- **The test drops peaks, and almost never a sample changes.** On the selection half it drops 19 of 503 peaks and on the calibration half 14 of 478. None is inside a single recording.
- **On the selection half no sample changes,** which is why part 1 saw every number unchanged. On the calibration half 1 pair stream changes, in 12 samples.
- **What this answers.** The test is not idle: it removes about one peak in thirty. A removed peak has the same view on both sides, and the pieces beside it then take the same view anyway, so the views of the samples hardly ever depend on the test.

**The other decoders at every value of their number.**

| Decoder | Its number | Calibration half: segment F1 | Macro-F1 | Boundary recall | Selection half: segment F1 | Macro-F1 | Boundary recall | Chosen |
|---|---|---|---|---|---|---|---|---|
| mode filter | 3 | 0.979 | 0.993 | 0.990 | 0.906 | 0.988 | 0.984 |  |
| mode filter | 5 | 0.979 | 0.993 | 0.990 | 0.906 | 0.988 | 0.984 |  |
| mode filter | 9 | 0.979 | 0.993 | 0.990 | 0.906 | 0.988 | 0.984 |  |
| mode filter | 13 | 0.977 | 0.993 | 0.990 | 0.909 | 0.988 | 0.982 |  |
| mode filter | 17 | 0.983 | 0.993 | 0.987 | 0.923 | 0.988 | 0.971 | yes |
| mode filter | 21 | 0.969 | 0.993 | 0.957 | 0.930 | 0.988 | 0.919 |  |
| mode filter | 25 | 0.948 | 0.993 | 0.945 | 0.904 | 0.988 | 0.893 |  |
| mode filter | 31 | 0.849 | 0.993 | 0.739 | 0.842 | 0.988 | 0.762 |  |
| mean filter | 3 | 0.979 | 0.993 | 0.992 | 0.908 | 0.988 | 0.987 | yes |
| mean filter | 5 | 0.977 | 0.993 | 0.992 | 0.906 | 0.988 | 0.987 |  |
| mean filter | 9 | 0.978 | 0.993 | 0.992 | 0.913 | 0.988 | 0.982 |  |
| mean filter | 13 | 0.977 | 0.993 | 0.990 | 0.918 | 0.998 | 0.982 |  |
| mean filter | 17 | 0.975 | 0.993 | 0.990 | 0.924 | 0.998 | 0.974 |  |
| mean filter | 21 | 0.961 | 0.993 | 0.970 | 0.930 | 0.998 | 0.948 |  |
| mean filter | 25 | 0.941 | 0.993 | 0.930 | 0.914 | 0.998 | 0.908 |  |
| mean filter | 31 | 0.873 | 0.993 | 0.789 | 0.848 | 0.998 | 0.791 |  |
| minimum duration | 3 | 0.979 | 0.993 | 0.990 | 0.906 | 0.988 | 0.984 |  |
| minimum duration | 5 | 0.979 | 0.993 | 0.990 | 0.906 | 0.988 | 0.984 |  |
| minimum duration | 8 | 0.983 | 0.993 | 0.985 | 0.921 | 0.988 | 0.950 | yes |
| minimum duration | 10 | 0.977 | 0.993 | 0.970 | 0.912 | 0.988 | 0.935 |  |
| minimum duration | 15 | 0.887 | 0.993 | 0.774 | 0.841 | 0.988 | 0.723 |  |
| minimum duration | 20 | 0.741 | 0.993 | 0.568 | 0.725 | 0.988 | 0.531 |  |
| minimum duration | 30 | 0.512 | 0.993 | 0.322 | 0.490 | 0.988 | 0.275 |  |
| Viterbi path | 0.5 | 0.979 | 0.993 | 0.990 | 0.906 | 0.988 | 0.982 |  |
| Viterbi path | 1 | 0.979 | 0.993 | 0.990 | 0.911 | 0.988 | 0.982 |  |
| Viterbi path | 2 | 0.980 | 0.993 | 0.990 | 0.914 | 0.988 | 0.979 |  |
| Viterbi path | 4 | 0.979 | 0.993 | 0.987 | 0.920 | 0.988 | 0.979 |  |
| Viterbi path | 8 | 0.981 | 0.993 | 0.985 | 0.953 | 0.979 | 0.969 | yes |
| Viterbi path | 16 | 0.980 | 0.993 | 0.970 | 0.954 | 0.998 | 0.940 |  |
| Viterbi path | 32 | 0.902 | 0.993 | 0.832 | 0.895 | 0.998 | 0.825 |  |
| Viterbi path | 64 | 0.710 | 0.986 | 0.543 | 0.670 | 0.998 | 0.492 |  |

On the selection half, around the value that the rule chose:

| Decoder | Values | Chosen | Segment F1 of the chosen | Lowest | Highest | Within 0.01 of the chosen | Within 0.02 | Above the chosen |
|---|---|---|---|---|---|---|---|---|
| mode filter | 8 | 17 | 0.923 | 0.842 | 0.930 | 2 | 7 | 1 |
| mean filter | 8 | 3 | 0.908 | 0.848 | 0.930 | 4 | 6 | 5 |
| minimum duration | 7 | 8 | 0.921 | 0.490 | 0.921 | 2 | 4 | 0 |
| Viterbi path | 8 | 8 | 0.953 | 0.670 | 0.954 | 2 | 2 | 1 |

- **Two of the Viterbi path's eight costs lie within 0.02 of the chosen one:** 8 and 16, with 0.953 and 0.954. A cost of 4 gives 0.920 and a cost of 32 gives 0.895. Its grid doubles the cost at every step, which is coarse.
- **For the two filters a larger window does better on the selection half than the chosen one.** At 21 samples the mean filter has 0.930 and the mode filter 0.930, where the chosen windows have 0.908 and 0.923. These values were not chosen on the calibration half, and by the plan they are not set against the released decoder.
- **The minimum duration has no value above the chosen one,** and 4 of its 7 values lie within 0.02 of it.

**Lower temperatures.**

| Temperature | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.50 | 1.00 |
|---|---|---|---|---|---|---|---|---|
| Log loss, calibration half | 0.0138 | 0.0141 | 0.0144 | 0.0146 | 0.0148 | 0.0150 | 0.0158 | 0.0411 |

| The 284 single recordings of the selection half | Wrong | Mean probability | Right | Mean probability minus the share that is right | Calibration error, equal width | Calibration error, equal count | Brier score | Log loss |
|---|---|---|---|---|---|---|---|---|
| as published | 1 | 0.963 | 283 | -0.034 [-0.042, -0.025] | 0.034 | 0.034 | 0.011 | 0.043 |
| with a temperature of 0.05 | 1 | 0.990 | 283 | -0.006 [-0.013, +0.001] | 0.011 | 0.006 | 0.008 | 0.014 |

The streams of the selection half under the fitted temperature:

| System | Decoder setting | Segment F1 at 50%, multi streams | Minus as published | Pair streams | Wrong samples | Cut precision | Cut recall | Segments that match nothing | Recordings right of 284, majority rule |
|---|---|---|---|---|---|---|---|---|---|
| as published | half window 7, threshold 0.174 | 0.959 | | 0.971 | 6.1% | 0.959 | 0.976 | 24 | 281 |
| the temperature, the published setting | half window 7, threshold 0.174 | 0.906 | -0.053 [-0.079, -0.029] | 0.957 | 7.1% | 0.862 | 0.982 | 76 | 282 |
| the temperature, the setting by the rule | half window 5, threshold 0.000 | 0.907 | -0.053 [-0.079, -0.028] | 0.957 | 7.0% | 0.860 | 0.984 | 77 | 282 |

Every number of section 5, under the temperature minus as published, with the same draws. A count of confused recordings has no interval.

| Under the temperature minus as published | With the published decoder setting | With the setting by the rule |
|---|---|---|
| Segment F1 at 50%, multi streams | -0.053 [-0.079, -0.029] | -0.053 [-0.079, -0.028] |
| Edit | -5.0 [-7.7, -2.6] | -4.9 [-7.6, -2.5] |
| Wrong samples, percentage points | +1.0 [+0.2, +1.8] | +0.9 [+0.1, +1.7] |
| Cut precision | -0.097 [-0.135, -0.061] | -0.098 [-0.137, -0.063] |
| Cut recall | +0.005 [+0.000, +0.013] | +0.008 [+0.000, +0.021] |
| Mean distance of matched cuts, seconds | +0.000 [-0.005, +0.006] | -0.000 [-0.005, +0.005] |
| Cuts at joins of the same view, percentage points | +2.6 [+0.5, +5.1] | +2.6 [+0.5, +5.1] |
| Segments that match nothing | +52 [+31, +75] | +53 [+32, +76] |
| Segments missed | +6 [-2, +14] | +5 [-4, +13] |
| Segment F1 at 50%, pair streams | -0.014 [-0.038, +0.000] | -0.014 [-0.038, +0.000] |
| Accuracy, single recordings | +0.004 [+0.000, +0.011] | +0.004 [+0.000, +0.011] |
| Macro-F1, single recordings | +0.009 [+0.000, +0.032] | +0.009 [+0.000, +0.032] |
| A4C read as A5C, recordings | -1 | -1 |
| A5C read as A4C, recordings | +0 | +0 |
| Label changes per recording | +0.039 [+0.011, +0.070] | +0.053 [+0.014, +0.109] |

- **The fit takes 0.05, the lowest value tried again.** On the calibration half the log loss falls with every step down, from 0.0150 at 0.30 to 0.0138 at 0.05. The fit found no best value, and by the plan no lower value is tried.
- **The word is "candidate", as in part 1, and it says little.** With a temperature of 0.05 the log loss of the selection half is 0.014 (-0.029 [-0.035, -0.024] against as published), where 0.30 gave 0.015. The same difference for the Brier score is -0.004 [-0.007, -0.000]. One recording of 284 is wrong either way.
- **At 0.05 the probability is close to a count of votes.** A logit is multiplied by 20 before the probabilities are formed, so a window gives its likeliest view a probability near 1 unless two views are almost level. The probability of a recording is then close to the share of its windows and seeds that name the view. The median probability is 1.000. The mean probability is 0.006 below the share of right recordings, where the published one is 0.034 below.
- **The streams lose under the temperature, with either decoder setting.** With the published setting segment F1 on the multi streams is 0.906 (-0.053 [-0.079, -0.029]), and the segments that match nothing rise from 24 to 76. With the rule's own setting it is 0.907 (-0.053 [-0.079, -0.028]).
- **Sharp probabilities make thresholds of the rule's grid fall together.** Of its 40 thresholds 15 are zero, and the 40 hold 21 distinct values. 35 of the 40 settings keep the floors, and the rule takes half window 5 with a threshold of zero. With either setting the result is about that of no decoder at all in part 1: 0.906 and 0.907 against 0.906.
- **What this answers.** At the fitted temperature the decoder loses what it works with. The streams were scored at this one temperature only. What a milder one does to them, such as the 0.30 of part 1, was not measured.

**Against what was expected** (section 10).

- **Higher thresholds: wrong in one point.** An added setting above the published one was expected on the selection half. There is none. The rule keeps the published setting and the word is not "candidate", both as expected.
- **The halves exchanged: as expected.**
- **The view test: wrong in one point.** Fewer than 10 dropped peaks were expected on each half. There are 14 and 19. No sample changes on the selection half, as expected.
- **The other decoders: wrong.** At least 4 of the Viterbi path's 8 costs were expected within 0.02 of the chosen one. There are 2.
- **Lower temperatures: wrong in one point.** With the rule's own setting the streams were expected to lose 0.01 at most. They lose 0.053. The fitted value, the flat log loss and the loss with the published setting came out as expected.

**What follows.** By section 6 nothing changes in the published system. Part 3 names no candidate for the decoder. For `classify`'s probability the word is "candidate" for a temperature of 0.30 in part 1 and for 0.05 here, by the same rule, and the fit has no best value.
