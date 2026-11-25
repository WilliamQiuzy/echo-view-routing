# After the comparison: three checks on development data

**Status: done on 2026-11-24; the read of section 9 was made on 2026-11-25.** A denser read-out is a candidate: on development data its cuts lie nearer to the view changes and nothing else is lost. It is offered as an option that is off by default. One read of the test benchmark found nearer cuts at a cost for both of its steps: the cuts are nearer, segment F1 is not lower, and one single recording more of 800 is misread (section 9). The offset between A4C and A5C and the mix with EchoViewCLIP are not supported on development data. Sections 1 to 7 are the plan. They were committed before any of their numbers existed, as commits `c1e6252` and `a98fccc`. The results are in section 8. The test benchmark was not read for sections 1 to 8. Section 9 is the plan of one read of it for the denser read-out, and the result of that read.

Words used here are those of [COMPARISON.md](COMPARISON.md): a **piece** is a run of samples of the reference that come from one recording, a **segment** is a run of samples to which a system gives one view, a **cut** is a sample whose view differs from the view of the sample before it, and a **stream** is several recordings joined into one sequence. A **sample** is 0.1 s.

## 1. Why

The comparison on the test benchmark ([COMPARISON.md](COMPARISON.md), section 10) found three things that are against EchoRouter.

- **Its cuts are no more exact than its 0.5 s grid.** 52.5% of the view changes have a cut within 0.1 s. Two systems that give an output for every sample have about 90%.
- **It reads A4C as A5C** in 10 of the 800 single recordings. The other systems do so in 1 to 7.
- **EchoViewCLIP reads single recordings better** (98.9% against 98.1%), and ten recordings are read right only by EchoViewCLIP.

Rule 8 of that plan says what may follow: a change to the system is chosen on development data and needs other data to confirm it. This document does the first half. It asks, on the development benchmark only, whether three changes that need no training are worth confirming. It changes nothing in the published system.

## 2. What was known when this was written

- **The test numbers** of [COMPARISON.md](COMPARISON.md), section 10.
- **The development numbers** of its section 9. On the selection half EchoRouter scores 0.959 on multi streams, reads 98.9% of the 284 single recordings right and has an A5C recall of 100%. So EchoRouter makes few errors on development data, and the weaknesses of the test benchmark may not show there at all. That would itself be a result.
- **Earlier rounds.** In October 2026 the project tried a denser read-out on earlier models. Segment F1 did not rise. The distance of the cuts was not looked at then.

## 3. The three questions

### Q1. Does a denser read-out place the cuts better?

EchoRouter's model reads 16 samples (1.6 s) and gives the view of one marked sample. The published system asks it once every 5 samples, and every sample takes the output nearest to it. The model can be asked more often.

| Arm | What it is | Forward passes |
|---|---|---|
| every 0.5 s | the published system | 1× |
| every 0.5 s, interpolated | the same outputs; between two outputs each seed's probabilities are interpolated linearly | 1× |
| every 0.2 s | one output every second sample | 2.5× |
| every 0.1 s | one output for every sample | 5× |

Two reference rows stand beside the arms: a perfect system that is read every 0.5 s, and one that is read every 0.2 s. At 0.2 s every second sample lies midway between two outputs. It takes the earlier one, as everywhere in the package, so a perfect system on that grid is still 0.05 s off on average. At 0.1 s every sample has an output of its own.

Each arm is decoded twice. **With a setting of its own:** the rule of [EVALUATION.md](EVALUATION.md) on the arm's probabilities for the calibration half, the same rule, grid and floors that gave the published setting. This is the like-for-like row, and the sentence rests on it. **With the released numbers carried over** (half window 7, threshold 0.1738, radius 5): shown as a description.

### Q2. Can one number trade A4C against A5C?

One number `b` is added to the A5C logit of every output of every seed. Nothing else changes; the decoder keeps its released setting. `b` runs from −3 to +3 in steps of 0.5. The `b` with the highest macro-F1 over the single recordings of the calibration half is chosen. Among equal values the one nearest to zero wins, and of two equally near the negative one.

### Q3. Does EchoViewCLIP's evidence add to EchoRouter's?

For every sample the two systems' probabilities are mixed: (1 − `w`) of EchoRouter's and `w` of EchoViewCLIP's. EchoViewCLIP's rows sum to less than 1, because the codes outside the five views were dropped; for the mix they are rescaled to sum to 1. `w` runs from 0 to 0.5 in steps of 0.1. Each `w` gets a decoder setting of its own by the rule on the calibration half. The `w` with the highest segment F1 at 50% on the calibration half's multi streams is chosen, among the values of `w` for which a setting meets the floors. Among equal values the smaller `w` wins.

This is not a change that could ship as it is: it needs a second model. It asks whether a second source of evidence would be worth building in.

## 4. The data and the forward passes

- **The development benchmark only:** 472 sequences in the calibration half and 541 in the selection half (521 single recordings, 252 pair streams, 240 multi streams). Everything is chosen on the calibration half and reported on the selection half.
- **Q1 needs forward passes.** The three published seeds give an output for every sample of the 1,013 sequences (71,503 samples). The outputs of the three sparser arms are taken from these. **The check:** every logit must be a finite number, at the published positions the new logits must agree with the stored development logits to 0.001, and the two arms at 0.5 s are always computed from the stored logits.
- **Q2 needs none.** It uses the stored development logits.
- **Q3 needs none.** It uses the stored development logits and EchoViewCLIP's stored outputs for the development benchmark ([COMPARISON.md](COMPARISON.md), section 9).
- **The code** is `echoview/evaluation/followup.py`. It is committed and tested before it runs. For Q1 the package gets an optional read-out step; its default stays 5 samples.

## 5. The numbers and the intervals

Every arm gets every number of [COMPARISON.md](COMPARISON.md), section 5, on the selection half, from the same code. Differences are an arm minus the published arm, with the same draws for both: percentile intervals at 95% from 10,000 draws with a fixed seed. The development benchmark names no examinations, so streams are drawn for the numbers about streams and recordings for the numbers about single recordings. These intervals are narrower than intervals over examinations would be.

## 6. How the result will be worded

These rules are fixed now. Each question ends with one of two words: the change is a **candidate**, or it is **not supported on development data**.

1. **Q1.** An arm is a candidate when, with a setting of its own on the multi streams of the selection half, (a) its cuts are nearer: the mean distance of the matched cuts is lower than the published arm's and the interval of the difference lies wholly below zero; (b) its segment F1 at 50% is not lower than the published arm's by more than 0.005; and (c) its macro-F1 on single recordings is not lower.
2. **Q2.** The offset is a candidate when the chosen `b` is not zero and, on the selection half, the macro-F1 of the single recordings is higher than the published system's with an interval wholly above zero, and the segment F1 at 50% on multi streams is not lower by more than 0.005.
3. **Q3.** The mix is a candidate when the chosen `w` is above zero and, on the selection half, the segment F1 at 50% on multi streams is higher than the published system's with an interval wholly above zero, and the macro-F1 on single recordings is not lower.
4. **A candidate is not a result.** It was chosen and seen on development data that the project has used many times. It changes nothing in the published system. It would need other data to confirm it: one read of the test benchmark for that question alone, with the owner's authorisation, or new labelled recordings. A candidate of Q1 may be offered as an option that is off by default and says that it is not confirmed.
5. **Everything is published.** Every arm and number goes into `docs/followup/`, whatever the word.

## 7. What is expected, and the limits

**Expected.**

- **Q1.** The denser arms place their cuts nearer: a perfect system on the 0.5 s grid is 0.12 s off on average. Segment F1 should change little, as in October.
- **Q2.** `b` stays at or near zero. On the selection half EchoRouter reads 281 of 284 single recordings right, so there is little to gain there, and the weakness seen on the test benchmark cannot be tuned on these recordings.
- **Q3.** No gain. On the selection half EchoViewCLIP with the decoder is far behind EchoRouter (0.866 against 0.959).

**Limits.**

- The development benchmark was used to choose EchoRouter's recipe, its seeds' checkpoints, the decoder's grid and its floors. EchoRouter's numbers on it are optimistic: it loses 0.05 from the selection half to the test benchmark.
- Q1 has three arms against the published one. With three chances, a candidate by chance is more likely than with one. That is one reason why a candidate needs confirmation.
- EchoViewCLIP was stopped early on the validation recordings that form the calibration half. A `w` chosen there is chosen on data that favour EchoViewCLIP.
- The streams are built by joining pieces of labelled recordings. They are not examinations.

**Changes to this plan, made before anything was scored.** The code was reviewed on 2026-11-24, before it ran on any data. The review found no wrong number. It led to three additions above: the two reference rows of Q1 and the note on samples that lie midway between two outputs, the demand that every logit is finite, and the sentence that both arms at 0.5 s come from the stored logits. No rule of section 6 changed.

## 8. Results

**Done on 2026-11-24 with the code of commit `a98fccc`. The test benchmark was not read.** The output of the command is [`followup/followup.json`](followup/followup.json), SHA-256 `cd9117720be355b0278c2e7c620735c8000900069178714bf360e1e3e6084c24`. The forward passes for Q1 ran from 07:05 to 07:55 UTC. At the published positions the new logits differ from the stored ones by 0.000005 at most, over all 1,013 sequences. The stored logits give the published 0.959 on the selection half.

### Q1. A denser read-out: two arms are candidates

The settings, chosen on the calibration half:

| Arm | Forward passes | Feasible settings of 40 | Half window | Quantile | Calibration half: segment F1 at 50% | Word |
|---|---|---|---|---|---|---|
| every 0.5 s | 1.0× | 40 | 7 | 0.999 | 0.984 | the published system |
| every 0.5 s, interpolated | 1.0× | 40 | 7 | 0.999 | 0.983 | not supported on development data |
| every 0.2 s | 2.4× | 40 | 5 | 0.999 | 0.982 | candidate |
| every 0.1 s | 4.9× | 40 | 5 | 0.999 | 0.984 | candidate |

The selection half, each arm with the setting of its own. Stream numbers are for the multi streams unless a column says pairs.

| Arm | Segment F1 at 50%, multi | pairs | Cut recall within 0.1 s | 0.2 s | Cut precision (0.5 s) | Cut recall (0.5 s) | Cut distance (s) | Wrong samples | of them edge | Piece midpoint | Segments that match nothing | Segments missed | Recordings: accuracy | Macro-F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| every 0.5 s (published) | 0.959 | 0.971 | 51.0% | 83.2% | 0.959 | 0.976 | 0.153 | 6.1% | 3.7% | 97.8% | 24 | 17 | 98.9% | 0.979 |
| every 0.5 s, interpolated | 0.953 | 0.971 | 57.3% | 86.6% | 0.969 | 0.966 | 0.134 | 5.7% | 3.1% | 97.4% | 23 | 24 | 98.9% | 0.979 |
| every 0.2 s | 0.957 | 0.976 | 72.8% | 88.7% | 0.966 | 0.974 | 0.109 | 5.0% | 2.6% | 97.5% | 23 | 20 | 99.3% | 0.988 |
| every 0.1 s | 0.961 | 0.969 | 81.9% | 92.1% | 0.974 | 0.976 | 0.071 | 4.0% | 1.7% | 97.5% | 20 | 19 | 98.9% | 0.979 |

An arm minus the published system, with 95% intervals:

| Arm minus the published system | Cut distance (s) | Segment F1 at 50%, multi | pairs | Cut recall within 0.1 s | Wrong samples | Macro-F1 of recordings |
|---|---|---|---|---|---|---|
| every 0.5 s, interpolated | -0.019 [-0.025, -0.012] | -0.006 [-0.016, +0.002] | -0.000 [-0.007, +0.007] | +6.3 [+3.2, +9.4] | -0.3 [-0.7, +0.0] | +0.000 [+0.000, +0.000] |
| every 0.2 s | -0.044 [-0.056, -0.032] | -0.002 [-0.014, +0.009] | +0.005 [+0.000, +0.015] | +21.7 [+16.1, +27.4] | -1.0 [-1.9, -0.2] | +0.009 [+0.000, +0.032] |
| every 0.1 s | -0.082 [-0.093, -0.071] | +0.002 [-0.009, +0.012] | -0.002 [-0.008, +0.000] | +30.9 [+25.3, +36.3] | -2.0 [-3.0, -1.1] | +0.000 [+0.000, +0.000] |

- **Every 0.1 s is a candidate.** Its cuts are 0.071 s from the view changes on average, where the published system's are 0.153 s away. 81.9% of the view changes have a cut within 0.1 s, against 51.0%. Its wrong samples fall from 6.1% to 4.0%, and the fall is at the edges of the pieces (3.7% to 1.7%). Segment F1 at 50% and the macro-F1 of single recordings do not change. It reads 4.9 times as many windows.
- **Every 0.2 s is a candidate.** Its cuts are 0.109 s away, and 72.8% of the view changes have a cut within 0.1 s. Segment F1 is lower by 0.002, inside the margin. It reads 2.4 times as many windows.
- **Interpolating the published outputs is not supported.** The cuts come nearer by 0.019 s, but segment F1 falls by 0.006, more than the margin of 0.005.
- **What does not change.** The pieces that are read as another view as a whole stay as they were (1.6% of the samples for the published system, 1.8% at 0.1 s), and so does the reading of A5C pieces (92.9% in all three). A denser read-out moves cuts. It does not change what the model takes a piece for. Segment F1 did not rise in October either.
- **The grid is not all.** A perfect system that is read every 0.5 s is 0.120 s off on average, and one that is read every 0.2 s is 0.050 s off. Read for every sample, EchoRouter is still 0.071 s off, so about half of the published system's 0.153 s was the grid and half is the model.
- **The setting matters little at 0.1 s.** With the released numbers carried over, the arm at 0.1 s scores the same (0.959 and 0.071 s). At 0.2 s the setting of its own is the better one (0.957 against 0.948):

| Arm, with the released numbers | Segment F1 at 50%, multi | pairs | Cut recall within 0.1 s | 0.2 s | Cut precision (0.5 s) | Cut recall (0.5 s) | Cut distance (s) | Wrong samples | of them edge | Piece midpoint | Segments that match nothing | Segments missed | Recordings: accuracy | Macro-F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| every 0.5 s, interpolated | 0.954 | 0.966 | 57.3% | 86.4% | 0.979 | 0.958 | 0.133 | 5.8% | 3.1% | 97.1% | 19 | 27 | 98.9% | 0.979 |
| every 0.2 s | 0.948 | 0.971 | 72.0% | 88.2% | 0.954 | 0.971 | 0.111 | 5.6% | 2.6% | 97.1% | 30 | 23 | 99.3% | 0.988 |
| every 0.1 s | 0.959 | 0.967 | 81.4% | 92.4% | 0.964 | 0.976 | 0.071 | 4.1% | 1.7% | 97.5% | 23 | 18 | 98.9% | 0.979 |

**What follows from Q1.** By rule 4 of section 6 the two candidates are offered as an option that is off by default: `echoview segment --read-every 0.2` or `0.1`, and `Ensemble.segment(frames, stride=2)` or `stride=1` in Python. The option decodes with the settings of the table above, says on every run that it is not the published system, writes its step into the version column of its output, and does not combine with `--uncertain`, which was studied with the published read-out only. The published system and every published number stay as they are. The option was checked against the study on 55 sequences of the selection half (12 multi streams, 14 pair streams, 29 single recordings): at each of the three steps `Ensemble.segment` returns the views that the study decoded, for all 55. Confirming the option would take one read of the test benchmark for this question alone: forward passes of the published models for every sample of the benchmark, with the owner's authorisation. That read has not been asked for. (It was asked for later: section 9 plans and reports it.)

### Q2. An offset between A4C and A5C: not supported

The calibration half chooses `b` = +0.5: the macro-F1 of its single recordings rises from 0.986 to 0.993, because one more A5C recording is read right. On the selection half this offset changes no single recording (macro-F1 +0.000) and leaves segment F1 where it was (+0.000 [−0.007, +0.007]). The whole curve, for description:

**Calibration half**

| Offset on A5C | Recordings: accuracy | Macro-F1 | A4C read as A5C | A5C read as A4C | A4C recall | A5C recall | Multi streams: segment F1 at 50% | Piece midpoint, A4C | Piece midpoint, A5C | Wrong samples |
|---|---|---|---|---|---|---|---|---|---|---|
| -3.0 | 98.7% | 0.977 | 0 | 3 | 100.0% | 83.3% | 0.928 | 98.8% | 71.5% | 8.5% |
| -2.5 | 98.7% | 0.977 | 0 | 3 | 100.0% | 83.3% | 0.959 | 98.8% | 82.5% | 6.8% |
| -2.0 | 99.2% | 0.985 | 0 | 2 | 100.0% | 88.9% | 0.976 | 98.8% | 88.3% | 5.5% |
| -1.5 | 99.2% | 0.985 | 0 | 2 | 100.0% | 88.9% | 0.981 | 98.8% | 90.5% | 5.2% |
| -1.0 | 98.7% | 0.978 | 1 | 2 | 98.4% | 88.9% | 0.981 | 98.1% | 92.0% | 5.1% |
| -0.5 | 98.7% | 0.978 | 1 | 2 | 98.4% | 88.9% | 0.983 | 98.1% | 94.2% | 4.9% |
| +0.0 | 99.2% | 0.986 | 1 | 1 | 98.4% | 94.4% | 0.984 | 96.9% | 97.1% | 4.6% |
| +0.5 | 99.6% | 0.993 | 1 | 0 | 98.4% | 100.0% | 0.980 | 95.7% | 97.8% | 4.8% |
| +1.0 | 99.2% | 0.986 | 2 | 0 | 96.7% | 100.0% | 0.979 | 95.7% | 97.8% | 4.8% |
| +1.5 | 99.2% | 0.986 | 2 | 0 | 96.7% | 100.0% | 0.975 | 93.8% | 97.8% | 5.2% |
| +2.0 | 98.7% | 0.980 | 3 | 0 | 95.1% | 100.0% | 0.969 | 92.6% | 97.8% | 5.6% |
| +2.5 | 97.5% | 0.961 | 6 | 0 | 90.2% | 100.0% | 0.958 | 87.7% | 99.3% | 6.5% |
| +3.0 | 97.0% | 0.955 | 7 | 0 | 88.5% | 100.0% | 0.939 | 82.7% | 100.0% | 8.1% |

**Selection half**

| Offset on A5C | Recordings: accuracy | Macro-F1 | A4C read as A5C | A5C read as A4C | A4C recall | A5C recall | Multi streams: segment F1 at 50% | Piece midpoint, A4C | Piece midpoint, A5C | Wrong samples |
|---|---|---|---|---|---|---|---|---|---|---|
| -3.0 | 97.9% | 0.933 | 0 | 5 | 100.0% | 54.5% | 0.871 | 100.0% | 45.2% | 16.0% |
| -2.5 | 98.2% | 0.949 | 0 | 4 | 100.0% | 63.6% | 0.888 | 100.0% | 52.4% | 14.1% |
| -2.0 | 98.9% | 0.975 | 0 | 2 | 100.0% | 81.8% | 0.923 | 100.0% | 73.2% | 10.0% |
| -1.5 | 98.9% | 0.975 | 0 | 2 | 100.0% | 81.8% | 0.939 | 99.3% | 83.3% | 8.4% |
| -1.0 | 98.9% | 0.975 | 0 | 2 | 100.0% | 81.8% | 0.937 | 98.0% | 85.7% | 8.0% |
| -0.5 | 98.9% | 0.979 | 2 | 0 | 97.7% | 100.0% | 0.948 | 98.0% | 89.9% | 6.5% |
| +0.0 | 98.9% | 0.979 | 2 | 0 | 97.7% | 100.0% | 0.959 | 97.4% | 92.9% | 6.1% |
| +0.5 | 98.9% | 0.979 | 2 | 0 | 97.7% | 100.0% | 0.960 | 96.1% | 94.6% | 6.0% |
| +1.0 | 98.2% | 0.962 | 4 | 0 | 95.4% | 100.0% | 0.966 | 94.1% | 97.6% | 5.6% |
| +1.5 | 97.9% | 0.955 | 5 | 0 | 94.3% | 100.0% | 0.967 | 92.8% | 98.8% | 5.7% |
| +2.0 | 97.5% | 0.948 | 6 | 0 | 93.1% | 100.0% | 0.968 | 92.8% | 98.2% | 6.0% |
| +2.5 | 97.5% | 0.948 | 6 | 0 | 93.1% | 100.0% | 0.959 | 87.5% | 99.4% | 6.8% |
| +3.0 | 96.5% | 0.929 | 9 | 0 | 89.7% | 100.0% | 0.941 | 78.9% | 100.0% | 8.6% |

- **There is nothing to tune on these recordings.** Without an offset EchoRouter makes two errors between A4C and A5C among the single recordings of each half.
- **The development data point the other way than the test benchmark.** On the test benchmark EchoRouter reads A4C as A5C in 10 single recordings and A5C as A4C in 2, which would ask for a negative offset. The calibration half asks for a positive one. A negative offset costs A5C on development data at once: at −1.0 the selection half loses 2 of its 11 A5C recordings.
- **Single recordings and streams pull apart.** On the selection half's streams a positive offset reads the A5C pieces better and the A4C pieces worse (at +2.0: 98.2% and 92.8%, from 92.9% and 97.4%). On the test benchmark, too, the A5C pieces of the streams are the weaker ones (82.8% against 92.9% for A4C), while the single recordings err towards A5C. One number cannot serve both.

### Q3. EchoViewCLIP's probabilities mixed in: not supported

The calibration half chooses `w` = 0.2 (segment F1 0.987 against 0.984). On the selection half the mix is worse than EchoRouter alone: 0.934 against 0.959, a difference of −0.026 [−0.042, −0.011]. The single recordings do not change.

| Weight of EchoViewCLIP | Feasible settings of 40 | Half window | Quantile | Calibration half: segment F1 at 50% | Selection half: segment F1 at 50%, multi | Recordings: accuracy | Macro-F1 | A5C recall | Piece midpoint, A5C | Cut distance (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 40 | 7 | 0.999 | 0.984 | 0.959 | 98.9% | 0.979 | 100.0% | 92.9% | 0.153 |
| 0.1 | 40 | 7 | 0.999 | 0.986 | 0.959 | 98.9% | 0.979 | 100.0% | 92.9% | 0.152 |
| 0.2 | 40 | 5 | 0.999 | 0.987 | 0.934 | 98.9% | 0.979 | 100.0% | 88.7% | 0.152 |
| 0.3 | 40 | 7 | 0.999 | 0.986 | 0.944 | 98.9% | 0.979 | 100.0% | 89.9% | 0.152 |
| 0.4 | 40 | 7 | 0.999 | 0.982 | 0.929 | 98.9% | 0.979 | 100.0% | 86.3% | 0.151 |
| 0.5 | 39 | 11 | 0.999 | 0.978 | 0.926 | 98.6% | 0.968 | 90.9% | 85.7% | 0.150 |

- **The calibration half favours EchoViewCLIP**, as section 7 warned: it was stopped early on these recordings. On the selection half every weight above 0.1 lowers segment F1.
- **The ten recordings stay out of reach.** On the test benchmark ten single recordings are read right only by EchoViewCLIP. Whether a mix would recover them cannot be seen on development data, where EchoRouter reads 281 of 284 single recordings right and the mix changes none.

### What the three say together

- **One change is worth confirming:** the denser read-out. It answers the first of the three findings of section 1, and only that one.
- **The other two findings cannot be worked on with this development benchmark.** EchoRouter makes too few errors on it, and the errors it makes there are not the ones of the test benchmark. Work on A4C against A5C, or on single recordings, needs labelled recordings that show those errors: a development set that the system was not chosen on, or readings by an expert.
- **Nothing here is a result about the published system.** Every number of this section is a development number.

## 9. Confirming the denser read-out on the test benchmark

**Status: planned on 2026-11-24 and read on 2026-11-25. Both steps have nearer cuts at a cost.** Everything in this section down to "The result" was written and committed before any of its numbers existed.

Rule 4 of section 6 says what confirms a candidate: one read of the test benchmark for that question alone, with the owner's authorisation. This section prepares that read for the two candidates of Q1. The owner asked for the preparation on 2026-11-24. The read itself is made only once the owner has authorised it by name, and that is recorded here when it is given. An earlier version of this paragraph said that the authorisation had been given; that was written too early. On 2026-11-25 the owner was asked whether this one read may be made, and said yes. That was after the plan, the reviewed code and the result of the forward passes were public. This sentence was committed before the read.

### What is confirmed

The option as version 1.3.0 ships it, and nothing else. The three published models are read every 0.2 s and every 0.1 s, and each step is decoded with the setting that `echoview/reference.py` holds for it: half window 5 and radius 5 for both, with thresholds 0.1414 and 0.1528. These are the settings of the first table of section 8. Nothing is chosen on the test benchmark. The interpolated arm and the rows with the released numbers carried over are not scored: they are not offered, and this read answers one question.

### What is known now

- **The published system on the test benchmark** ([COMPARISON.md](COMPARISON.md), section 10). On the 120 multi streams its segment F1 at 50% is 0.909, its cuts are 0.153 s from the view changes on average, 52.5% of the 379 view changes have a cut within 0.1 s, and 9.4% of the samples are wrong. It reads 98.1% of the 800 single recordings right. A perfect system on the 0.5 s grid is 0.121 s off.
- **The development numbers** of section 8.
- **The question came from this benchmark.** Finding 1 of section 1 was seen on it. The settings did not come from it: they were chosen on the calibration half, and the published models have never been read at a denser step on a recording of the test benchmark.
- **The test benchmark has been read before:** for its published numbers, for the uncertain output, for the comparison and for the comparison's addendum. None of these reads used a denser read-out.

### What is run, in this order

1. **This plan is public** before any code for it runs.
2. **The code** is in `echoview/evaluation/followup.py`: `logits --test` writes the outputs for every sample of the test benchmark, and `confirm` makes the read. Both are tested and reviewed before they run on the test benchmark.
3. **Forward passes.** The three published models give an output for every sample of the 1,160 sequences of the test benchmark: 800 single recordings, 240 pair streams and 120 multi streams, 72,185 samples. The published system reads 14,871 windows for them, the step of 0.2 s reads 35,791 and the step of 0.1 s reads 72,185. The outputs at 0.2 s are taken from those at 0.1 s, as in section 4. This step reads the images of the test benchmark and no label.
4. **The check, without labels.** Every logit must be a finite number. At the published positions the new logits must agree to 0.001 with the stored test logits, the file with SHA-256 `c5df435ef3022dd55f711c4efa72c74f19aad7f40193f179902c57dca97e288d` that gave the published numbers. The digest of the new file and the result of this check are committed before the read.
5. **The read** is one run of `confirm`.
   - It is entered in the access record of the test benchmark before it starts.
   - From the stored logits it first computes the published system's rows. The four published numbers must come out. The rows "EchoRouter" and "Perfect on the 0.5 s grid" must equal those of the comparison's first read (`comparison/test.json`, SHA-256 `4f00000524df0581ee7f093980135a206c9180f51d15457524acb6547afe45c0`) number by number, intervals included. When one differs, nothing is written.
   - Then it scores the two arms and writes one file.
6. **No second read.** Once the output exists, its digest goes into the code. From then on `confirm` stops unless it is given that file and its own result equals it.

### The numbers and the intervals

- **The rows:** the published system, every 0.2 s, every 0.1 s, and for reference a perfect system that is read every 0.5 s and one that is read every 0.2 s.
- **Every row gets every number of [COMPARISON.md](COMPARISON.md), section 5,** from the same code, on the multi streams, the pair streams and the single recordings.
- **Differences** are an arm minus the published system with the same draws: 10,000 draws with seed 20261123, as in the comparison. The test benchmark names its examinations, so every interval is taken twice, over streams and over the 125 examinations. A description gives the wider of the two at 95%.

### How the result will be worded

These rules are fixed now. They are the three conditions of rule 1 of section 6, taken on the test benchmark.

1. **(a) Nearer cuts.** On the multi streams the mean distance of the matched cuts is lower than the published system's, and the interval of the difference lies wholly below zero. Two arms are tested, so this interval is taken at 97.5%. It is the hull of the interval over streams and the interval over examinations, as for the named comparisons of the comparison: both must lie below zero.
2. **(b) Segment F1 kept.** The arm's segment F1 at 50% on the multi streams is not lower than the published system's by more than 0.005.
3. **(c) Single recordings kept.** The arm's macro-F1 on the single recordings is not lower than the published system's.
4. **The words.** An arm is **confirmed on the test benchmark** when (a), (b) and (c) hold. It has **nearer cuts at a cost** when (a) holds and (b) or (c) does not; the cost is then stated with its interval. It is **not confirmed on the test benchmark** when (a) does not hold.
5. **What follows, whatever the word.** The published system and every published number stay as they are. The default read-out stays 0.5 s: the uncertain output and every published number rest on it, and a denser read-out reads 2.4 or 4.9 times as many windows. The option's note and the README then say for each step what the read found, in the words of rule 4, where they now say "not confirmed". Making a denser read-out the default would be a new decision of the owner. The test benchmark is not read again for this question.
6. **Everything is published** in `docs/followup/`, whatever the word.

### What is expected, and the limits

- **Expected.** The published system's cuts are as far off on the test benchmark as on the selection half (0.153 s on both). The denser arms should come about as much nearer as there: to about 0.07 s at 0.1 s and 0.11 s at 0.2 s. Segment F1 should move by less than 0.01 either way. Among 800 single recordings one or two may be read differently, in either direction.
- **(b) and (c) are rules on single values.** A move of two or three segments, or of one recording, can decide them. That is why the word "nearer cuts at a cost" exists, and why the cost is given with its interval.
- **The question came from this benchmark,** as said above. An arm that is confirmed here is confirmed for settings this benchmark did not choose, on a weakness this benchmark showed.
- **(a) is about matched cuts.** The distance is averaged over the view changes that have a cut within 1 s. An arm that finds more view changes averages over more of them. The share of view changes with a match and the shares with a cut within 0.1 s to 1 s are reported beside it.
- **The view changes are joins.** The streams are built by joining pieces of labelled recordings, so a view change is a hard cut between two recordings and is known to the sample. In an examination the probe moves from one view to the next, and no reference is exact to 0.1 s there. Nearer cuts on this benchmark say that the system places a cut more exactly when the change is sharp.

**Changes to this plan, made before any forward pass on the test benchmark.** The code was reviewed on 2026-11-24, before it ran on any data of the test benchmark. The review found nothing that could make a number or a word wrong. It led to four additions. The code now names the digest of the file with the new outputs, and `confirm` takes no other file; until that digest is committed, the command does not run. The output also holds, for each arm, how many single recordings only the arm or only the published system reads right: a number of section 5 of the comparison that the first draft of the code left out. An output file now appears whole or not at all. And the run on the server refuses code that is not a commit, writes its line into the access record before the read or stops, and does not start a read a second time without a decision. No rule of this section changed.

### The forward passes

**Done with the code of commit `282be13`, from 23:21 UTC on 2026-11-24 to 00:13 UTC on 2026-11-25. No label was read and nothing was scored.** The step is entered in the access record of the test benchmark. Its check is [`followup/dense_test_check.json`](followup/dense_test_check.json).

- **The file.** One output of each of the three published models for every sample of the 1,160 sequences: 72,185 samples. Its SHA-256 is `d1a23a82577b9af9fbade07c4f40aca64e081647c0e40d9af057e63f8bc3e500`, and the code now names it: `confirm` takes this file and no other.
- **The check.** Every logit is a finite number. At the 14,871 published positions the new logits differ from the stored test logits by 0.000005 at most, where the plan allows 0.001.
- **What the steps read.** 35,791 windows at 0.2 s and 72,185 at 0.1 s, against 14,871 for the published system.

### The result

**Read on 2026-11-25 at 06:18 UTC with the code of commit `b15ff57`. Both steps have nearer cuts at a cost.** The owner authorised the read on that day. The sentence that records it is that commit, made at 06:18:03 UTC. The read ran from 06:18:42 to 06:18:58 UTC, and its line in the access record of the test benchmark carries the time 06:18:42. Its output is [`followup/confirmation.json`](followup/confirmation.json), SHA-256 `87bd05802365859c63c9279cb6aa6d02b2a4ee1d32a71434d6e3c49538a0f5a2`. The code now names that file, so `confirm` can only reproduce it from here on.

**The checks before the scoring held.** The stored logits give the four published numbers: 0.909 and 0.985 for the segment F1 at 50% of the multi and the pair streams, 98.1% and 0.956 for the accuracy and the macro-F1 of the single recordings. The rows "EchoRouter" and "Perfect on the 0.5 s grid" equal those of the comparison's first read number by number, intervals included.

The three conditions and the word for each step. The intervals of (a) are at 97.5%, over streams and over examinations; those of (b) and (c) are at 95%.

| Step | Windows read | Cut distance minus the published system (s) | 97.5% over streams | 97.5% over examinations | (a) nearer cuts | Segment F1 at 50% minus the published system | (b) kept | Macro-F1 of single recordings minus the published system | (c) kept | Word |
|---|---|---|---|---|---|---|---|---|---|---|
| every 0.2 s | 2.4× | -0.031 | [-0.044, -0.017] | [-0.045, -0.017] | yes | +0.005 [-0.009, +0.018] | yes | -0.004 [-0.015, +0.005] | no | **nearer cuts at a cost** |
| every 0.1 s | 4.9× | -0.068 | [-0.080, -0.056] | [-0.083, -0.054] | yes | +0.012 [+0.000, +0.023] | yes | -0.003 [-0.011, +0.000] | no | **nearer cuts at a cost** |

Every row. Stream numbers are for the multi streams unless a column says pairs.

| Row | Segment F1 at 50%, multi | pairs | Cut recall within 0.1 s | 0.2 s | Cut precision (0.5 s) | Cut recall (0.5 s) | Cut distance (s) | View changes with a matched cut | Wrong samples | of them edge | Piece midpoint | Segments that match nothing | Segments missed | Recordings: accuracy | Macro-F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| every 0.5 s (published) | 0.909 | 0.985 | 52.5% | 79.2% | 0.934 | 0.937 | 0.153 | 95.0% | 9.4% | 3.2% | 93.7% | 46 | 45 | 98.1% | 0.956 |
| every 0.2 s | 0.914 | 0.986 | 68.1% | 82.6% | 0.952 | 0.947 | 0.122 | 95.5% | 8.3% | 2.6% | 94.2% | 42 | 44 | 98.0% | 0.952 |
| every 0.1 s | 0.921 | 0.986 | 74.9% | 88.7% | 0.957 | 0.950 | 0.085 | 95.8% | 7.3% | 1.8% | 94.4% | 38 | 41 | 98.0% | 0.953 |
| a perfect system read every 0.5 s | 1.000 | 1.000 | 60.4% | 100.0% | 1.000 | 1.000 | 0.121 | 100.0% | 3.0% | 3.0% | 100.0% | 0 | 0 | 100.0% | 1.000 |
| a perfect system read every 0.2 s | 1.000 | 1.000 | 100.0% | 100.0% | 1.000 | 1.000 | 0.053 | 100.0% | 1.3% | 1.3% | 100.0% | 0 | 0 | 100.0% | 1.000 |

An arm minus the published system, with 95% intervals. Shares are in percentage points.

| Arm minus the published system | Cut distance (s) | Segment F1 at 50%, multi | pairs | Cut recall within 0.1 s | within 0.2 s | Wrong samples | of them edge | Piece midpoint | Accuracy of recordings | Macro-F1 of recordings |
|---|---|---|---|---|---|---|---|---|---|---|
| every 0.2 s | -0.031 [-0.043, -0.019] | +0.005 [-0.009, +0.018] | +0.001 [+0.000, +0.004] | +15.6 [+9.8, +21.4] | +3.4 [-0.7, +7.6] | -1.1 [-1.8, -0.3] | -0.7 [-1.0, -0.3] | +0.4 [-0.4, +1.4] | -0.1 [-0.6, +0.3] | -0.004 [-0.015, +0.005] |
| every 0.1 s | -0.068 [-0.081, -0.055] | +0.012 [+0.000, +0.023] | +0.001 [+0.000, +0.004] | +22.4 [+16.9, +28.1] | +9.5 [+5.6, +13.7] | -2.1 [-2.9, -1.5] | -1.5 [-1.8, -1.2] | +0.7 [-0.1, +1.7] | -0.1 [-0.4, +0.0] | -0.003 [-0.011, +0.000] |

- **Every 0.1 s has nearer cuts at a cost.**
  - (a) holds. Its cuts are 0.085 s from the view changes on average, where the published system's are 0.153 s away. The difference is -0.068 s, and at 97.5% its interval runs from -0.083 to -0.054 over streams and examinations together. 74.9% of the 379 view changes have a cut within 0.1 s, against 52.5%.
  - (b) holds. Segment F1 at 50% is 0.921 against 0.909. Wrong samples fall from 9.4% to 7.3%, and most of the fall is at the edges of the pieces (3.2% to 1.8%). The segments that match nothing fall from 46 to 38.
  - (c) does not hold. The macro-F1 of the single recordings is 0.953 against 0.956: -0.003, with the interval [-0.011, +0.000]. This is one recording of 800. The published system reads it right and this step does not, and no recording goes the other way.
- **Every 0.2 s has nearer cuts at a cost.**
  - (a) holds. Its cuts are 0.122 s away: -0.031 s, with the interval from -0.045 to -0.017 at 97.5%. 68.1% of the view changes have a cut within 0.1 s.
  - (b) holds. Segment F1 at 50% is 0.914 against 0.909.
  - (c) does not hold. The macro-F1 of the single recordings is 0.952 against 0.956: -0.004, with the interval [-0.015, +0.005]. Two recordings are read right only by the published system and one only by this step.
- **The cost is A5C read as A4C, in single recordings.** The benchmark has 40 single recordings of A5C. The published system reads 38 of them right, the step of 0.2 s reads 36 and the step of 0.1 s reads 37; the others are read as A4C. The step of 0.2 s reads one more of the 231 recordings of A4C right (220 against 219). A4C against A5C is the weakness that the comparison already names, and in single recordings a denser read-out does not help it. On the multi streams it is the other way at 0.1 s: more samples of both views are read right (A5C 81.5% against 78.9%, A4C 92.9% against 90.0%, both intervals above zero), while the middles of their pieces are not shown to change.
- **The cost is one recording, and the rule is on the value.** Both intervals of (c) reach zero and go down to -0.015 and -0.011. So the read shows neither that a denser read-out reads single recordings worse nor that it reads them as well. Rule 3 asks that the value is not lower, and it is lower: at both steps one single recording more of 800 is misread. The plan said that one recording can decide this rule. The word is the one that the rules give.
- **The gain is smaller than on development data.** The plan expected about 0.07 s at 0.1 s and about 0.11 s at 0.2 s. The read gives 0.085 s and 0.122 s. The cuts come nearer by 0.068 s and 0.031 s, where the selection half gave 0.082 s and 0.044 s. Segment F1 at 0.1 s is higher by 0.012, a little more than the 0.01 that the plan expected. That is a description: the plan tested no gain in segment F1.
- **The grid is not all.** A perfect system that is read every 0.5 s is 0.121 s off on average, and one that is read every 0.2 s is 0.053 s off. Read for every sample, EchoRouter is still 0.085 s off, and that part is not the grid's.
- **What is not shown to change.** The pieces that are read as another view as a whole are 4.3% of the samples for the published system, 4.2% at 0.2 s and 4.0% at 0.1 s, and the intervals of both differences include zero. A denser read-out moves cuts. The read does not show that it changes what the model takes a piece for.
- **The distances are averaged over nearly the same number of view changes.** 360 of the 379 view changes have a matched cut for the published system, 362 at 0.2 s and 363 at 0.1 s.

**What follows.** By rule 5 the published system and every published number stay as they are, and the default read-out stays 0.5 s. The option's note, its help text and the README now say what this read found, where they said "not confirmed". The test benchmark is not read again for this question. Making a denser read-out the default would be a new decision of the owner. The limits of the plan hold for this result: the question came from this benchmark, and its view changes are joins of recordings.
