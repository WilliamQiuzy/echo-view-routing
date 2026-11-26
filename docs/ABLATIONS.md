# What the parts add, and what the system stands: studies on development data

**Status: planned on 2026-11-26 (UTC). No result yet.** Everything in this document down to "Results" was written and committed before any of its numbers existed.

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

No result yet.
