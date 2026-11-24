# After the comparison: three checks on development data

**Status: plan, written on 2026-11-24.** Sections 1 to 7 were committed before any of their numbers existed. The results will go into section 8. The test benchmark is not read for this document.

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

Not yet.
