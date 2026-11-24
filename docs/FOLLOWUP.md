# After the comparison: three checks on development data

**Status: done on 2026-11-24.** A denser read-out is a candidate: on development data its cuts lie nearer to the view changes and nothing else is lost. It is offered as an option that is off by default, and it is not confirmed on the test benchmark. The offset between A4C and A5C and the mix with EchoViewCLIP are not supported on development data. Sections 1 to 7 are the plan. They were committed before any of their numbers existed, as commits `c1e6252` and `a98fccc`. The results are in section 8. The test benchmark was not read for this document.

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

**What follows from Q1.** By rule 4 of section 6 the two candidates are offered as an option that is off by default: `echoview segment --read-every 0.2` or `0.1`, and `Ensemble.segment(frames, stride=2)` or `stride=1` in Python. The option decodes with the settings of the table above, says on every run that it is not the published system, writes its step into the version column of its output, and does not combine with `--uncertain`, which was studied with the published read-out only. The published system and every published number stay as they are. The option was checked against the study on 55 sequences of the selection half (12 multi streams, 14 pair streams, 29 single recordings): at each of the three steps `Ensemble.segment` returns the views that the study decoded, for all 55. Confirming the option would take one read of the test benchmark for this question alone: forward passes of the published models for every sample of the benchmark, with the owner's authorisation. That read has not been asked for.

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
