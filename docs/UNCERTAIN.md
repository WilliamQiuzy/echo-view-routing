# The "uncertain" output

**Status: the study is done, and its rule is released.** `echoview segment --uncertain` applies it: a band of 0.3 s on each side of every cut, `f_low = 2`, `f_high = 1.25`, `m = 0.1`, seeds off. Sections 1 to 6 are the plan. They were written, reviewed and committed before any result of the study existed. Sections 7 and 8 hold the results, in the order they were produced. Section 9 is a later read of the test benchmark with the released rule. Its plan was committed before the read. The read confirmed the tests for whole pieces. It also showed a smaller gain than on the development half, and a confidence threshold that does better in the multi streams.

Words used here: a **piece** is a run of samples of the reference that come from one recording. A **segment** is a run of samples to which the system gives one view. A **cut** is a sample whose view, as the system gives it, differs from the view of the sample before it.

## 1. The idea

The decoder cuts a stream into views and gives every 0.1 s sample one of five views. It always commits: where two views meet, the cut is put at one sample, and a piece that looks half like A4C and half like A5C still gets one of the two names.

The proposal is to let the system decline. Where the evidence is weak, a sample is marked **uncertain** and gets no view.

- **Uncertain is not "other".** It does not mean "this is none of the five views". The model cannot tell that: its probabilities do not separate in-scope from out-of-scope recordings (see the model card). Uncertain means "this system cannot decide here between the views it knows".
- **Nothing else changes.** A sample that is not uncertain keeps the view the published decoder gives it. With the option off, the output is the published output.

## 2. The rule

The published decoder runs first, with its published setting (half window 7, threshold 0.1738, radius 5). Four tests then mark samples as uncertain. A sample is uncertain when any test marks it.

| Test | What it marks | Parameter |
|---|---|---|
| **Band** | the `k` samples before every cut and the `k` samples from the cut on | `k`, in samples |
| **Weak cut** | the samples whose view changes when the stream is decoded again with a lower threshold (`threshold / f_low`) or a higher one (`threshold × f_high`) | `f_low`, `f_high` (1 switches a side off) |
| **Margin** | every sample of a segment whose mean probability does not favour one view clearly: the largest minus the second largest is below `m` | `m` |
| **Seeds** | every sample of a segment that one of the seeds, by its own mean probability, would name differently | on or off |

- **What the tests are for.** The band covers the position of a cut, which the system knows only to a few tenths of a second: it decides once every 0.5 s from 1.6 s of video. The other three are aimed at whole pieces: a short segment that exists only because of a weak cut, or a segment the model reads as two views at once.
- **How the weak-cut test behaves.** A lower threshold only adds cuts and a higher one only removes them. When a cut between two long segments is removed, the two become one segment, and the side that loses its name is marked as a whole. Which side loses depends on the lengths and the probabilities of both. The marked set does not grow steadily with `f`.
- **Tidying.** A labelled run shorter than 5 samples (0.5 s) that has an uncertain sample next to it becomes uncertain too. This is fixed, not tuned.
- **No band at the ends of a stream.** The first and the last sample of a stream are not cuts.

## 3. What is compared

Every setting is compared, at the same share of labelled samples, with:

1. **The published system**: nothing declined.
2. **Chance**: the same number of samples declined at random. Its error is the published system's error.
3. **The ideal**: the same number of samples declined, wrong ones first.
4. **The band alone**, for `k` from 0 to 8.
5. **Low confidence**: the same number of samples declined, those with the smallest largest-probability first. This is the usual way to let a classifier abstain.

## 4. How the setting is chosen

Only the calibration half of the EV9V development benchmark is used for choices. The selection half is read in a second step, by a second command, after the choice of the first step has been committed.

**Step 1, the band.** `k` is the smallest value from 0 to 8 for which at least 90% of the matched view changes of the calibration half lie at most `k` samples from their cut. View changes and cuts are matched one to one, at most 1 s apart, in the multi streams and the examination-order streams (the matching of the existing scorecard). The band then holds the true change in 90% of the matched cases. The report gives the share for `k − 1`, `k` and `k + 1`. If no `k` up to 8 reaches 90%, the study stops there. Section 6 says that this step is expected to give 3.

**Step 2, the tests for whole pieces.** With that `k`, every combination of `f_low` and `f_high` in {1, 1.25, 1.5, 2, 3}, `m` in {0, 0.1, 0.2, 0.3, 0.4, 0.5} and seeds off or on is tried: 300 settings.

- **What is counted.** Only the pieces whose middle sample the band alone leaves labelled are counted, over all calibration sequences. For each setting, `D_wrong` is the number of these pieces that the published system reads wrong at the middle and the setting declines there, and `D_right` the number it reads right and the setting declines.
- **The score.** `S = D_wrong − D_right / 5`. One wrong piece taken away is worth giving up five right ones. About 2% of the pieces are wrong, so a test that declines at random gives up about fifty right pieces for one wrong one.
- **The candidate** is the setting with the largest `S`. A tie goes to the fewer uncertain samples, then to the earlier setting in the order above.
- **The bar.** The candidate goes on to step 3 only if `S` is above 0 and `D_wrong` is at least 4 pieces that come from at least 3 different recordings. Otherwise the three tests are not adopted, and the band alone is the result.

**Step 3, the selection half.** One command scores: the published output, the band alone for `k` from 0 to 8, the low-confidence reference, the candidate of step 2 if there is one, and each of its tests switched off in turn.

- **The test of the candidate.** `S` is computed on the selection half in the same way. Its distribution comes from a bootstrap that draws whole recordings again (2,000 draws): every piece of a recording goes with it. The three tests are part of the released rule only if `S` is above 0 and the 5th percentile of that distribution is above 0. Otherwise the released rule is the band alone.
- **Why recordings.** A recording appears in several sequences: whole as a single recording, once in an examination-order stream, and as fragments in several multi streams. Its pieces are not independent. Every count of pieces is therefore given with the number of different recordings behind it.

**What these choices are.** 90%, the factor 5, the bar and the 5th percentile are judgements, made before the results. The report shows `D_wrong`, `D_right` and `S` of all 300 settings, so that a reader can judge another choice.

## 5. How the result is evaluated

The report has two parts, one for each kind of output, by kind of sequence: single recordings, pair streams, multi streams and examination-order streams. Every share comes with its counts. A number that can only improve when more is declined is printed beside what it cost and beside its value under chance.

**Part A, the labelled part: what the system commits to.**

| Number | Meaning |
|---|---|
| Coverage | the share of samples that keep a view |
| Error among labelled samples | of the samples that keep a view, the share with a wrong one; per view as purity and recovery |
| The same at equal coverage | the error of each reference of section 3 when it declines as many samples |
| Pieces by their middle sample | right or wrong by the published answer, kept or declined: four counts, each with its number of recordings |
| Pieces without a labelled sample | reference pieces of which nothing is left |
| Segments by their middle sample | every labelled run the system returns is right or wrong at its middle |

**Part B, the uncertain part: what the system declines.**

| Number | Meaning |
|---|---|
| Size | the share of samples that are uncertain; the number of uncertain intervals and their lengths |
| The table of samples | wrong or right by the published answer, declined or kept: four counts. From them: the error among the labelled samples, and the error a forced answer would have on the declined ones |
| Where it falls | uncertain samples within 0.5 s of a true view change, against those inside one view; uncertain intervals with no view change within 0.5 s |
| Single recordings | recordings with an uncertain sample, split by whether the published system reads the recording right. Declining a rightly read recording is a false alarm; declining a misread one is a catch |
| Same-view joins | of the joins where two fragments of one view meet, the share with an uncertain sample within 0.5 s |
| Errors declined | of the published system's wrong samples, the share now uncertain: in total, by kind (edge, spurious, absorbed, misread) and for A4C against A5C. Beside it: the share of all samples that is uncertain, which is what chance would catch |
| Changes located | of the true view changes, the share with an uncertain sample on at least one side |
| Beyond the band | of the samples the three tests add to the band, the share the published system has wrong, against the error left after the band alone |
| Each test | the samples the rule would lose if one test were switched off, and the share of them that is wrong |

**Intervals.** The numbers about samples get a 95% interval from a bootstrap over sequences. In the multi streams this interval is conditional on the recordings: the streams reuse them. The numbers about pieces get theirs from a bootstrap over recordings. Counts of 5 or fewer are given as counts, without an interval.

**How a result is worded.**

- A rule that passes holds for EV9V streams joined from recordings, where a view changes from one sample to the next. A continuous examination has no such splices, and nothing here measures it.
- A rule that fails is "not shown", with the number of wrong pieces and recordings it had to work with and the smallest effect that could have been seen.
- The numbers of the calibration half for a setting chosen there are marked as chosen on that half.

## 6. What was known when this was written

- The development report of the published system on both halves, including: the share of wrong samples by kind (in the multi streams 60% to 75% are edge errors), the offsets of matched cuts (median 0.1 s, 90% within 0.3 s), and the purity per view with 0.5 s removed at both ends of every segment. So step 1 is expected to give `k = 3`, and the effect of a band on the purity of the selection half was roughly known.
- That the remaining wrong pieces of both halves are mostly A5C read as A4C or the reverse. The three tests for whole pieces were designed with that in mind. The selection half therefore checks their parameters, not the idea behind them.
- Not known: any number about the three tests for whole pieces.
- Not read for this study: the test benchmark and MIMIC-IV-Echo.
- An independent review of this plan, made before any result, led to: the score and the bar of step 2 in place of "fewest wrong pieces", the counting of pieces beyond the band only, recordings as the unit for pieces, the low-confidence reference, and the removal of a summary number that could not tell a large uncertain part from a small one.
- An independent review of the code, made on 2026-11-20 before either command had run, found places where the code did less than this plan says or where the plan left a choice open. They were settled as follows, and nothing else in the plan changed:
  - A tie in `S` is decided on whole numbers, `5 × D_wrong − D_right`, so that rounding cannot break it.
  - A single recording counts as read right when the majority of its samples have the right view. This is the scorecard's rule for a recording.
  - Every number about samples has its interval. A number whose numerator or denominator is 5 or fewer has none.
  - The low-confidence reference has an interval from the same draws, and so does the rule's error minus the reference's error.
  - The share of pieces read right at the middle has its interval from the bootstrap over recordings.
  - Neither command replaces a file that exists unless it is given `--force`: running the first command again does not overwrite its choice.

## 7. Step 1 and step 2: the choice on the calibration half

Run on 2026-11-20 with the code of commit `5c85d02`, on the window logits of the published ensemble for the 508 sequences of the calibration half. The output of the command is in this repository, unchanged: [`uncertain/choice.json`](uncertain/choice.json), SHA-256 `61cb9e90f857a63ddcc9df3f85ea97bca76f376a39338eb5ff81144325db6484`. It holds the counts of all 300 settings. The selection half was not read.

**Step 1 gives `k = 3`.** The multi streams and the examination-order streams of the calibration half have 490 view changes. 484 of them are matched with a cut.

| `k` | 2 | 3 | 4 |
|---|---|---|---|
| Matched changes at most `k` samples from their cut | 86.8% | 96.1% | 99.0% |

The band is therefore the 0.3 s before every cut and the 0.3 s from the cut on. Section 6 expected this value.

**Step 2 gives a candidate, and the candidate meets the bar.**

| Setting | `f_low` | `f_high` | `m` | Seeds | `D_wrong` | `D_right` | `S` | Uncertain samples, of 48,539 |
|---|---|---|---|---|---|---|---|---|
| Candidate | 2 | 1.25 | 0.1 | off | 10 pieces from 6 recordings | 7 | 8.6 | 4,046 (8.3%) |
| Band alone | 1 | 1 | 0 | off | 0 | 0 | 0 | 3,508 (7.2%) |

- **The bar.** `S` is above 0, and the 10 wrong pieces come from 6 recordings. The bar asks for 4 pieces from 3 recordings. The candidate goes on to step 3.
- **What it had to work with.** Beyond the band, the published system reads 19 pieces wrong at the middle and 1,355 right. The candidate declines 10 of the 19 and 7 of the 1,355.
- **Ties.** `f_low = 3` gives the same counts as `f_low = 2`, and the earlier setting was taken. Six more settings reach 8.6, with 11 wrong and 12 right pieces. They mark more samples.
- **Which test does the work.** The lower threshold alone (`f_low = 2`) declines 8 wrong pieces and 1 right one, which gives 7.8. The margin alone reaches 5.8 at most (`m = 0.5`, 10 wrong and 21 right). The seeds test alone declines 1 wrong piece and 4 right ones. A higher threshold alone scores below 0 from `f_high = 1.5` on, and at `f_high = 2` it declines 3 wrong pieces and 78 right ones. In the candidate, the higher threshold and the margin add 2 wrong pieces and 6 right ones to what the lower threshold declines.
- **All settings.** 177 of the 300 have `S` above 0.

**The candidate on the calibration half.** These numbers come from the half on which the setting was chosen. They describe the choice and say little about how the rule does on other data.

| All 508 sequences | Samples that keep a view | Wrong among them |
|---|---|---|
| Published output | 100% | 3.38% |
| Band alone | 92.8% | 1.85% |
| Candidate | 91.7% | 1.18% |
| Low confidence, as many samples declined as by the candidate | 91.7% | 0.83% |

- **The low-confidence reference is ahead here.** Declining the samples with the smallest largest-probability leaves fewer wrong samples than the candidate does at the same coverage. The same holds against the band alone (0.95% against 1.85%). Step 3 gives this comparison on the selection half with its interval. It is not part of the decision of section 4, and section 8 reports it whatever it shows.
- **Pieces.** Read at the middle sample, 98.4% of the 1,423 pieces are right in the published output and 99.3% of those the candidate leaves labelled. 45 pieces have no labelled sample left, against 35 with the band alone.

## 8. Step 3: the selection half

Run on 2026-11-20 with the code of commit `75e9419`, after the choice of section 7 had been committed and pushed. The output of the command is [`uncertain/report.json`](uncertain/report.json), SHA-256 `a363092dc54ca206dd2117e9634e1c8e7fd51ce6ae970ca198b9a0dbfb1e9dc3`, unchanged. The selection half has 577 sequences with 53,922 samples, and 1,574 pieces from 284 recordings. All numbers are for the reference ensemble of three seeds.

**How to read the intervals.** They are 95% intervals, given in square brackets. The numbers about samples have theirs from a bootstrap over sequences. All 577 sequences are built from the same 284 recordings, so these intervals take the recordings as given and are too narrow as a statement about new recordings. The numbers about pieces have theirs from a bootstrap over recordings. A count of pieces is followed by its number of recordings in round brackets.

### The decision

**The three tests are adopted, by a small margin.**

| | Wrong pieces declined beyond the band | Right pieces declined beyond the band | `S` | 5th to 95th percentile of `S` |
|---|---|---|---|---|
| Candidate | 6, from 5 recordings | 12, from 7 recordings | 3.6 | 0.4 to 7.2 |

`S` is above 0, and so is its 5th percentile. By the rule of section 4 the released rule is the candidate: `k = 3`, `f_low = 2`, `f_high = 1.25`, `m = 0.1`, seeds off. Beyond the band, the published system reads 20 pieces wrong (11) and 1,493 right (284). The decision rests on those 20 pieces.

### Part A, the labelled part

| All 577 sequences | Samples that keep a view | Wrong among them | Low confidence at the same coverage | Rule minus low confidence |
|---|---|---|---|---|
| Published output | 100% | 3.04% [2.49, 3.63] | | |
| Band alone | 93.1% [92.2, 93.9] | 1.30% [0.83, 1.81] | 1.23% | +0.06 points [−0.22, +0.36] |
| Released rule | 92.1% [91.1, 93.0] | 1.04% [0.59, 1.54] | 1.09% | −0.06 points [−0.27, +0.20] |

At the same coverage, chance leaves 3.04% wrong and the ideal leaves none.

| Released rule, by kind | Sequences | Samples that keep a view | Wrong in the published output | Wrong among the labelled | Low confidence at the same coverage | Rule minus low confidence |
|---|---|---|---|---|---|---|
| Single recordings | 284 | 98.8% (16,438 of 16,635) | 1.38% | 0.85% [0.08, 1.80] | 1.01% | −0.16 points [−0.45, +0.04] |
| Pair streams | 137 | 91.5% (5,208 of 5,691) | 3.16% | 0.94% [0.21, 1.89] | 1.17% | −0.23 points [−1.06, +0.64] |
| Multi streams | 120 | 81.7% (12,220 of 14,961) | 6.06% | 1.70% [0.94, 2.63] | 1.29% | +0.41 points [−0.35, +1.24] |
| Examination-order streams | 36 | 95.1% (15,818 of 16,635) | 1.92% | 0.75% [0.01, 1.66] | 0.97% | −0.21 points [−0.42, −0.02] |

Purity is the share of the samples given a view that have this view in the reference. Recovery is the share of a view's reference samples that are given this view, so an uncertain sample counts against it.

| | PLAX | PSAX | A4C | A5C | SC4C |
|---|---|---|---|---|---|
| Purity, published output | 98.7% | 97.3% | 96.5% | 88.5% | 97.8% |
| Purity, released rule | 99.9% | 99.1% | 98.6% | 93.8% | 99.9% |
| Recovery, published output | 98.5% | 98.4% | 96.2% | 89.3% | 96.6% |
| Recovery, released rule | 95.1% | 93.6% | 90.3% | 74.3% | 86.6% |

Pieces are read at their middle sample.

| | Right, kept | Right, declined | Wrong, kept | Wrong, declined | Right among the kept | Pieces with no labelled sample |
|---|---|---|---|---|---|---|
| Published output | 1,545 (284) | 0 | 29 (13) | 0 | 98.2% [96.7, 99.3] | 0 |
| Band alone | 1,493 (284) | 52 (36) | 20 (11) | 9 (5) | 98.7% [97.4, 99.6] | 33 (24) |
| Released rule | 1,481 (283) | 64 (40) | 14 (8) | 15 (9) | 99.1% [98.1, 99.7] | 40 (28) |

| Released rule, by kind | Pieces | Right, kept | Right, declined | Wrong, kept | Wrong, declined | Pieces with no labelled sample | `S`, with its 5th to 95th percentile |
|---|---|---|---|---|---|---|---|
| Single recordings | 284 (284) | 278 (278) | 0 | 1 (1) | 5 (5) | 0 | 2.0 [0.0, 4.0] |
| Pair streams | 274 (274) | 268 (268) | 3 (3) | 2 (2) | 1 (1) | 0 | −0.2 [−0.6, 0.0] |
| Multi streams | 732 (231) | 661 (226) | 55 (35) | 9 (4) | 7 (7) | 35 (25) | 1.0 [−1.0, 3.4] |
| Examination-order streams | 284 (284) | 274 (274) | 6 (6) | 2 (2) | 2 (2) | 5 (5) | 0.8 [−0.4, 2.8] |

Of the labelled runs the system returns, 97.8% are right at their middle (1,096 of 1,121) [96.7, 98.7]. In the published output it is 96.8% (1,132 of 1,170) [95.7, 97.8].

### Part B, the uncertain part of the released rule

| Number | Value |
|---|---|
| Size | 7.9% of the samples are uncertain [7.0, 8.9]: 4,238 of 53,922, in 558 intervals. The median interval is 0.6 s long, 90% are at most 1.5 s, the longest is 4.0 s. |
| The table of samples | Declined: 1,121 wrong and 3,117 right. Kept: 516 wrong and 49,168 right. A forced answer would be wrong on 26.5% of the declined samples [24.1, 28.8], against 1.04% of the kept ones. |
| Where it falls | 78.9% of the uncertain samples lie within 0.5 s of a true view change [73.9, 83.6]. 45 of the 558 intervals have no view change within 0.5 s. |
| Single recordings | Of the 281 recordings the published system reads right, 7 get an uncertain sample (2.5%) [0.8, 4.5]. All 3 misread recordings get one. |
| Same-view joins | 14 of 437 have an uncertain sample within 0.5 s (3.2%) [1.7, 4.9]. |
| Errors declined | 68.5% of the published system's wrong samples are now uncertain [58.8, 78.8]. Chance would catch 7.9%. By kind: edge 770 of 808, spurious 111 of 261, absorbed 33 of 75, misread 207 of 493. A4C against A5C: 369 of 756. |
| Changes located | 522 of the 560 true view changes have an uncertain sample on at least one side (93.2%) [91.2, 95.2]. |
| Beyond the band | The three tests add 509 samples to the band. 134 of them are wrong (26.3%) [13.6, 40.7], against 1.30% left after the band alone. |
| Each test | Without the lower threshold the rule loses 350 samples, 49 of them wrong. Without the higher threshold it loses 120, 54 of them wrong. Without the margin it loses 9, 1 of them wrong. |

| Released rule, by kind | Uncertain samples | Uncertain intervals | Wrong samples now uncertain | Right samples now uncertain | True view changes with an uncertain sample beside them |
|---|---|---|---|---|---|
| Single recordings | 197 of 16,635 (1.2%) | 15 | 90 of 230 | 107 of 16,405 | no changes |
| Pair streams | 483 of 5,691 (8.5%) | 73 | 131 of 180 | 352 of 5,511 | 66 of 70 |
| Multi streams | 2,741 of 14,961 (18.3%) | 359 | 699 of 907 | 2,042 of 14,054 | 353 of 382 |
| Examination-order streams | 817 of 16,635 (4.9%) | 111 | 201 of 320 | 616 of 16,315 | 103 of 108 |

### The band at every width

| `k` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| Samples that keep a view | 100% | 97.8% | 95.6% | 93.1% | 91.0% | 88.9% | 86.4% | 84.6% | 82.4% |
| Wrong among them | 3.04% | 2.13% | 1.58% | 1.30% | 1.16% | 1.06% | 0.94% | 0.89% | 0.80% |

### How to read it

- **What the rule buys.** With the rule on, 7.9% of the samples of these streams get no view, and the share of wrong samples among the rest falls from 3.04% to 1.04%. Most of this is the band. It takes the error to 1.30% and costs 6.9% of the samples. The three tests take it to 1.04% for another 0.9 points. A wider band does less for more: `k = 4` reaches 1.16% and keeps 91.0%.
- **The pass is narrow.** The three tests were adopted on 6 wrong pieces from 5 recordings against 12 right ones, and the 5th percentile of `S` is 0.4. By kind of sequence `S` is 2.0 for single recordings, −0.2 for pair streams, 1.0 for multi streams and 0.8 for examination-order streams. No kind has a 5th percentile above 0 by itself.
- **This study does not show that the rule beats a confidence threshold.** Declining the same number of samples by lowest confidence leaves 1.09% wrong against the rule's 1.04%, and the interval of the difference includes 0. By kind, the rule is ahead in three and behind in the multi streams. Only the interval of the examination-order streams excludes 0, in the rule's favour (−0.21 points [−0.42, −0.02]). The reference was given the number of samples to decline on this half. As a rule of its own it would need a threshold, which was not chosen or tested here. What the rule adds is the form of its answer: whole intervals at the cuts and whole weak pieces.
- **What it costs.** The recovery of every view falls, and most for A5C, from 89.3% to 74.3%. A short piece between two cuts can be lost as a whole: 40 of the 1,574 pieces (28) keep no labelled sample. 7 of the 281 rightly read single recordings get an uncertain part.
- **What is left.** 14 wrong pieces from 8 recordings keep their wrong view, and half of the A4C against A5C errors stay labelled.
- **Where it holds.** The rule passed on EV9V streams joined from recordings, where a view changes from one sample to the next. A continuous examination has no such splices, and nothing here measures it. The test benchmark and MIMIC-IV-Echo were not read with the rule for this study. The test benchmark was read with it once afterwards (section 9).
- **Uncertain is still not "other".** Nothing in this study is about views outside the five.

## 9. One read of the test benchmark

**Status: read once, on 2026-11-20.** Everything in this section down to "The result" was written and committed before the read, as commit `d7cfbe0`.

The owner of the project authorised one read of its frozen EV9V test benchmark with the released rule, on 2026-11-20. The development study above did not read it.

- **What is read.** The window logits of the published ensemble for the 1,160 sequences of the test benchmark: 800 single recordings, 240 pair streams and 120 multi streams, 72,185 samples. They were stored by the evaluation of 2026-11-20 that reproduced the benchmark's published numbers (EVALUATION.md, "The test benchmark"), as one file with SHA-256 `c5df435ef3022dd55f711c4efa72c74f19aad7f40193f179902c57dca97e288d`. No model runs again. Before it scores anything, the command checks that the file holds exactly the benchmark's 1,160 arrays and that they give its four published numbers: the segment F1 of the multi streams and of the pair streams, and the accuracy and the macro-F1 of the single recordings. It stops if one of them differs. It writes the file's hash into its output.
- **What is computed.** One command, `python -m echoview.evaluation.uncertain_study confirm`, scores the published output, the band alone and the released rule with the two-part report of section 5, and the band at every width from 0 to 8. The benchmark has no examination-order streams. The command does not replace an output file that exists.
- **Nothing is chosen.** The rule is the released one, read from the package. Whatever the read shows, the released rule stays as it is, and the result is reported beside the development numbers in the README and the model card.
- **How the result is worded.** The test is the one of step 3: `S` over the pieces beyond the band, with its distribution from a bootstrap over recordings (2,000 draws). If `S` and its 5th percentile are above 0, the three tests for whole pieces are "confirmed on the test benchmark". If not, they are "not confirmed on the test benchmark". The band has no test of its own. For the released width, its coverage and its error among the labelled samples are reported with their intervals, and so is the comparison with the low-confidence reference. The other widths are given without intervals, as in section 8.
- **What is expected.** From the selection half: about 92% of the samples keep a view over all sequences and about 82% in the multi streams, and the share of wrong samples among them is about a third of the published output's. The check on the selection half passed by a small margin, so a failure of the `S` test here would be no surprise.
- **Limits known beforehand.** The test benchmark is not untouched: earlier rounds of the project were evaluated on it, though never with a rule for uncertain samples. The intervals are those of the development study, over sequences for the numbers about samples and over recordings for the numbers about pieces. The benchmark's own published intervals draw examinations for the single recordings, which this report does not do. The read is entered in the project's record of test reads.

### The result

Read once on 2026-11-20 at 22:33 UTC with the code of commit `d7cfbe0`. The output of the command is [`uncertain/test_report.json`](uncertain/test_report.json), SHA-256 `e35f745754ac55a25de3624efc3a0a3bf7efdb89c7ad453629548d2fe633d4dd`, unchanged. The logits gave the benchmark's four published numbers. The benchmark has 1,981 pieces from 800 recordings. Intervals and brackets are as in section 8.

**The three tests for whole pieces are confirmed on the test benchmark.**

| | Wrong pieces declined beyond the band | Right pieces declined beyond the band | `S` | 5th to 95th percentile of `S` |
|---|---|---|---|---|
| Released rule | 13 (11) | 13 (12) | 10.4 | 4.2 to 17.2 |

Beyond the band the published system reads 56 pieces wrong here (31). By kind, `S` is 1.6 [−1.0, 4.8] for the single recordings, −0.2 [−0.6, 0.0] for the pair streams and 9.0 [3.8, 15.0] for the multi streams.

**The labelled part.**

| All 1,160 sequences | Samples that keep a view | Wrong among them | Low confidence at the same coverage | Rule minus low confidence |
|---|---|---|---|---|
| Published output | 100% | 3.63% [2.85, 4.48] | | |
| Band alone | 95.5% [94.9, 96.1] | 2.46% [1.73, 3.29] | 2.07% | +0.39 points [+0.10, +0.67] |
| Released rule | 94.5% [93.8, 95.2] | 2.00% [1.35, 2.77] | 1.76% | +0.24 points [−0.02, +0.50] |

| Released rule, by kind | Sequences | Samples that keep a view | Wrong in the published output | Wrong among the labelled | Low confidence at the same coverage | Rule minus low confidence |
|---|---|---|---|---|---|---|
| Single recordings | 800 | 98.8% (45,968 of 46,507) | 1.92% | 1.53% [0.73, 2.55] | 1.66% | −0.13 points [−0.31, +0.04] |
| Pair streams | 240 | 92.9% (9,649 of 10,391) | 2.77% | 0.98% [0.35, 1.78] | 0.58% | +0.40 points [−0.15, +1.13] |
| Multi streams | 120 | 82.5% (12,605 of 15,287) | 9.41% | 4.47% [3.05, 6.07] | 3.17% | +1.30 points [+0.34, +2.19] |

| | PLAX | PSAX | A4C | A5C | SC4C |
|---|---|---|---|---|---|
| Purity, published output | 99.3% | 99.1% | 94.7% | 77.0% | 95.6% |
| Purity, released rule | 99.9% | 100.0% | 97.0% | 81.6% | 97.8% |
| Recovery, published output | 99.4% | 98.9% | 93.1% | 84.2% | 90.7% |
| Recovery, released rule | 97.4% | 96.1% | 87.8% | 73.6% | 84.3% |

| Pieces, read at the middle | Right, kept | Right, declined | Wrong, kept | Wrong, declined | Right among the kept | Pieces with no labelled sample |
|---|---|---|---|---|---|---|
| Published output | 1,917 (789) | 0 | 64 (32) | 0 | 96.8% [95.3, 98.0] | 0 |
| Band alone | 1,881 (789) | 36 (32) | 56 (31) | 8 (5) | 97.1% [95.9, 98.3] | 24 (22) |
| Released rule | 1,868 (787) | 49 (42) | 43 (25) | 21 (14) | 97.7% [96.6, 98.8] | 36 (30) |

| Released rule, by kind | Pieces | Right, kept | Right, declined | Wrong, kept | Wrong, declined | Pieces with no labelled sample | `S`, with its 5th to 95th percentile |
|---|---|---|---|---|---|---|---|
| Single recordings | 800 (800) | 778 (778) | 8 (8) | 10 (10) | 4 (4) | 3 (3) | 1.6 [−1.0, 4.8] |
| Pair streams | 480 (480) | 471 (471) | 3 (3) | 6 (6) | 0 | 0 | −0.2 [−0.6, 0.0] |
| Multi streams | 701 (405) | 619 (386) | 38 (34) | 27 (15) | 17 (12) | 33 (29) | 9.0 [3.8, 15.0] |

Of the labelled runs the system returns, 96.6% are right at their middle (1,583 of 1,638) [95.7, 97.6]. In the published output it is 96.0% (1,611 of 1,678) [95.0, 96.9].

**The uncertain part of the released rule.**

| Number | Value |
|---|---|
| Size | 5.5% of the samples are uncertain [4.8, 6.2]: 3,963 of 72,185, in 504 intervals. The median interval is 0.6 s long, 90% are at most 1.5 s, the longest is 6.0 s. |
| The table of samples | Declined: 1,257 wrong and 2,706 right. Kept: 1,363 wrong and 66,859 right. A forced answer would be wrong on 31.7% of the declined samples [28.3, 35.1], against 2.00% of the kept ones. |
| Where it falls | 73.4% of the uncertain samples lie within 0.5 s of a true view change [67.1, 78.7]. 53 of the 504 intervals have no view change within 0.5 s. |
| Single recordings | Of the 785 recordings the published system reads right, 20 get an uncertain sample (2.5%) [1.5, 3.7]. Of the 15 misread recordings, 7 get one. |
| Same-view joins | 16 of 322 have an uncertain sample within 0.5 s (5.0%) [2.6, 7.5]. |
| Errors declined | 48.0% of the published system's wrong samples are now uncertain [39.6, 56.9]. Chance would catch 5.5%. By kind: edge 622 of 657, spurious 150 of 391, absorbed 99 of 233, misread 386 of 1,339. A4C against A5C: 599 of 1,532. |
| Changes located | 459 of the 499 true view changes have an uncertain sample on at least one side (92.0%) [89.5, 94.3]. |
| Beyond the band | The three tests add 731 samples to the band. 331 of them are wrong (45.3%) [29.2, 60.4], against 2.46% left after the band alone. |
| Each test | Without the lower threshold the rule loses 429 samples, 170 of them wrong. Without the higher threshold it loses 134, 46 of them wrong. Without the margin it loses 121, 88 of them wrong. |

| Released rule, by kind | Uncertain samples | Uncertain intervals | Wrong samples now uncertain | Right samples now uncertain | True view changes with an uncertain sample beside them |
|---|---|---|---|---|---|
| Single recordings | 539 of 46,507 (1.2%) | 36 | 189 of 893 | 350 of 45,614 | no changes |
| Pair streams | 742 of 10,391 (7.1%) | 121 | 193 of 288 | 549 of 10,103 | 112 of 120 |
| Multi streams | 2,682 of 15,287 (17.5%) | 347 | 875 of 1,439 | 1,807 of 13,848 | 347 of 379 |

| Band of width `k` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| Samples that keep a view | 100% | 98.5% | 97.1% | 95.5% | 94.2% | 92.8% | 91.1% | 89.9% | 88.5% |
| Wrong among them | 3.63% | 3.03% | 2.66% | 2.46% | 2.36% | 2.26% | 2.16% | 2.11% | 2.03% |

**How to read it.**

- **Confirmed, and by a wider margin than on the selection half.** The rule declines 13 wrong pieces from 11 recordings against 13 right ones, and the 5th percentile of `S` is 4.2. Nearly all of it comes from the multi streams.
- **The gain is smaller than expected.** The plan expected the share of wrong samples among the labelled to fall to about a third of the published output's. It falls to 55% of it, from 3.63% to 2.00%. Half of this benchmark's wrong samples belong to misread pieces (1,339 of 2,620, against 493 of 1,637 on the selection half), and the rule leaves 71% of those labelled. The coverage is 94.5% where about 92% was expected: 64% of the benchmark's samples are single recordings, which have no cut. In the multi streams it is 82.5%, as expected.
- **A confidence threshold does better here in the multi streams.** At the same coverage the low-confidence reference leaves 1.76% wrong against the rule's 2.00% over all sequences, and 3.17% against 4.47% in the multi streams, where the interval of the difference excludes 0 (+1.30 points [+0.34, +2.19]). Against the band alone it is ahead over all sequences (+0.39 points [+0.10, +0.67]). The reference was again given the number of samples to decline, and a threshold of its own was not chosen or tested. On this benchmark the evidence on this count is still against the rule: whoever wants the fewest wrong samples for a given share of unnamed ones should expect a confidence threshold to do at least as well.
- **What it costs.** The recovery of A5C falls from 84.2% to 73.6%. 36 of the 1,981 pieces (30) keep no labelled sample. 20 of the 785 rightly read single recordings get an uncertain part.
- **What is left.** 43 wrong pieces from 25 recordings keep their wrong view, and 61% of the A4C against A5C errors stay labelled.
- **The limits named in the plan apply.** The benchmark is not untouched, and the intervals take the recordings as given.
