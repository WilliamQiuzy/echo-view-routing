# The comparison with other approaches, row by row

**Status: done. The test benchmark was read once, on 2026-11-24 (section 10).** With the project's decoder, EchoViewCLIP is not shown to differ from EchoRouter on streams; EchoRouter is ahead of EchoPrime's classifier; for STFM and ASFormer the question could not be answered. Sections 1 to 8 are the plan. They were written, reviewed and committed before either stage, as commit `26571f5`. Stage A touches no test data: it lists the stored outputs, chooses decoder settings on development data and tries the code on the development benchmark. Its results go into section 9 and are committed. Stage B is one read of the test benchmark. Its results are in section 10.

Words used here: a **piece** is a run of samples of the reference that come from one recording. A **segment** is a run of samples to which a system gives one view. A **cut** is a sample whose view, as a system gives it, differs from the view of the sample before it. A **stream** is several recordings joined into one sequence.

## 1. Why

The README compares EchoRouter with four other approaches by three numbers: segment F1 at 50% overlap on streams, and accuracy and macro-F1 on single recordings. That is too little to judge by.

- **One threshold of one matching rule.** A returned segment counts as right when it has the reference segment's view and the two overlap by at least half of their union. A segment with one end off by a third of its length still counts. The number does not say how exact the cuts are, and it does not separate a piece that is read as another view from a cut that is a little off.
- **The decoding was not the same for all.** EchoRouter's probabilities go through the project's decoder. Of the four others, only EchoPrime's classifier got that decoder. EchoViewCLIP, STFM and ASFormer were decoded by their authors' test rule: the likeliest view of every sample of their output (ASFormer's output is already that of a temporal model). The table does not show the others with the same decoder.
- **The published ensemble was never compared pair by pair.** The paired intervals in the README are for two single stage-2 models.
- **Macro-F1 rests on few recordings.** Of the 800 single recordings, 40 are A5C and 35 are SC4C.

This plan fixes a fuller table, the rows that give the others the same decoder, the intervals and the wording of the result, before any of it is computed.

## 2. What was known when this was written

The plan is not blind. These numbers existed before it.

- **From the run of 2026-10-23** on the test benchmark (the four other systems, and two single stage-2 models with the decoder, which score 0.919 each):

  | Test benchmark | Single stage-2 models | The four others |
  |---|---|---|
  | Multi streams: returned segments that match nothing | 41 | 64 to 92 |
  | Multi streams: reference segments that are missed | 40 | 47 to 82 |
  | Multi streams: precision of the cuts (within 0.5 s) | 0.93 | 0.81 to 0.94; EchoPrime with the decoder is the highest |
  | Multi streams: recall of the view changes | 0.93 | 0.88 to 0.95; EchoViewCLIP and ASFormer are higher |
  | Multi streams: cuts at joins of two pieces of the same view | 5% and 6% | 6% to 10% |
  | Pair streams: segment F1 at 50% | 0.978 and 0.979 | 0.971 to 0.989; no lead |
  | Single recordings: label changes inside a recording, per recording | 0.03 | 0.04 to 0.14 |
  | Single recordings: macro-F1 against EchoViewCLIP | −0.024 [−0.044, −0.008] and −0.013 [−0.035, +0.003] | |
  | Single recordings: A4C read as A5C | 9 and 12 | 1 to 7 |
  | Single recordings: A5C read as A4C | 2 and 3 | 3 to 13 |
  | Single recordings: A5C recall | 37 and 38 of 40 | 27 to 37 of 40 |

  EchoViewCLIP's sliding window returns 92 segments that match nothing, 62 of them at most 1.6 s long. EchoPrime's classifier scores 0.496 per frame and 0.851 with the decoder.
- **EchoViewCLIP with the project's decoder, on development data** (2026-10-05). With the setting that the rule of section 4 chose for it (half window 9, quantile 0.80; 28 of 40 settings were feasible), it scored 0.866 on the multi streams of the selection half and 0.973 on those of the calibration half. EchoRouter scores 0.959 and 0.984 there. On development data the same decoder leaves EchoViewCLIP well behind on the selection half. This row was never computed on the test benchmark.
- **From the read of 2026-11-20** ([UNCERTAIN.md](UNCERTAIN.md), section 9): over all 1,160 sequences EchoRouter gives 3.6% of the samples a wrong view, about half of those samples lie in pieces that are read wrong as a whole, and the middle sample is right for 96.8% of the pieces.
- **The seeds.** The numbers of single seeds on the test benchmark were known when the ensemble was chosen ([EVALUATION.md](EVALUATION.md)). The ensemble's 0.909 is below the 0.919 of the single stage-2 models.
- **The published rows** of the README's table.

## 3. What is read

Stored outputs only. No model runs for this plan, unless stage A says otherwise (section 4, row D).

- **EchoRouter.** The window logits of the published ensemble for the 1,160 sequences of the test benchmark, stored on 2026-11-20, with its three seeds: one file, SHA-256 `c5df435ef3022dd55f711c4efa72c74f19aad7f40193f179902c57dca97e288d`.
- **The four other systems.** Their stored outputs for the same 1,160 sequences: one row of five view probabilities per 0.1 s sample. The research code wrote them in October 2026, and the run of 2026-10-23 read them. Stage A records a digest of every folder.
  - EchoViewCLIP and STFM are read every 0.5 s on a window of 1.6 s, and a sample takes the output of the nearest window, as with EchoRouter. ASFormer and EchoPrime's classifier give an output for every sample.
  - The rows of EchoViewCLIP, STFM and EchoPrime sum to less than 1, because the codes outside the five views were dropped. They are used as stored and are not rescaled, as in October.
- **For decoder settings only:** each system's stored outputs for the calibration half of the development benchmark, where they exist. Stage A lists what exists. EchoViewCLIP's are known to exist.

**The check before anything is scored.** The stored outputs must give the published numbers of every row to half a unit of the sixth decimal:

| Row | Segment F1 at 50%, multi | Segment F1 at 50%, pairs | Accuracy | Macro-F1 |
|---|---|---|---|---|
| EchoRouter | 0.908909 | 0.984701 | 0.981250 | 0.956141 |
| EchoViewCLIP, sliding window | 0.866731 | 0.988920 | 0.988750 | 0.971936 |
| STFM | 0.859086 | 0.970954 | 0.982500 | 0.959627 |
| ASFormer | 0.856311 | 0.980716 | 0.982500 | 0.955852 |
| EchoPrime's classifier with the decoder | 0.851020 | 0.987586 | 0.978750 | 0.943459 |
| EchoPrime's classifier per frame | 0.495768 | 0.742734 | 0.978750 | 0.943459 |

When one number differs, the command stops, nothing is scored, and this plan is amended with the reason before another attempt. The command never replaces an output file. Its output names every file it read with its SHA-256 and holds pooled counts only: no recording is named. (The README printed EchoPrime's macro-F1 as 0.944. It is 0.943 and was corrected on 2026-11-23.)

**The benchmark.** The multi streams hold 701 pieces from 405 recordings and 379 view changes in 15,287 samples. Each stream mixes recordings of 3 to 8 examinations. The pieces per view are 149 PLAX, 140 PSAX, 154 A4C, 151 A5C and 107 SC4C, but the A5C pieces come from 39 recordings and the SC4C pieces from 34. The pair streams hold 480 pieces and 120 view changes; four of their pieces are A5C and none is SC4C. A piece is 1.0 to 4.0 s long, and a third of the pieces are at most 1.6 s long. The 800 single recordings come from 125 examinations, the 40 A5C ones from 29.

## 4. The rows

**A. As published.** EchoRouter (the ensemble with the decoder at its released setting). EchoViewCLIP, STFM and ASFormer by the likeliest view of every sample. EchoPrime's classifier with the decoder at the setting chosen for it on validation data in October, and per frame.

**D. With the project's decoder.** EchoViewCLIP, STFM and ASFormer, each with the decoder at a setting of its own.

- The setting is chosen in stage A by the rule of [EVALUATION.md](EVALUATION.md) ("The decoder and its rule") on the system's own outputs for the calibration half: the half, the grid and the rule that gave EchoRouter its setting. The grid is not extended for any system.
- When no setting meets the rule's two floors for a system, the floors are dropped for it: the setting with the highest segment F1 at 50% on the calibration half's multi streams is taken, and the table says so.
- When a system's outputs for the calibration half are not stored, it has no row D, and its question of section 7 is "not answered". Stage A says so. It may also ask the owner for forward passes of that system on the calibration half only; without that, nothing is run.
- For EchoPrime's classifier, the published row with the decoder is its row D.

**C. The released numbers carried over** (half window 7, threshold 0.1738, radius 5) to every other system. This is not like for like: the threshold was a quantile of EchoRouter's own scores. The row is shown as a sensitivity row. No sentence of section 7 rests on it.

**N. EchoRouter without the decoder:** the likeliest view of every sample of the ensemble's probabilities.

**S. The seeds.** Each of EchoRouter's three seeds alone, with the setting the rule chooses for that seed on the calibration half (stage A, from the stored development logits). The other systems are single models, the published system is three.

**P. A perfect system read every 0.5 s:** the reference labels taken at the window centres and given to the nearest samples. EchoRouter, EchoViewCLIP and STFM cannot do better than this row.

## 5. The numbers

Every row gets every number. "↑" means higher is better, "↓" lower.

**Streams**, for the 120 multi streams and the 240 pair streams apart:

| Number | What it says |
|---|---|
| Segment F1 at 50%, the project's matching ↑, overall and per view | the published headline |
| Segment F1 at 10%, 25%, 50% ↑ and the Edit score ↑, as MS-TCN defines them | the usual set of the action-segmentation literature; Edit looks at the order of the segments only |
| Returned segments that match nothing ↓, reference segments that are missed ↓, each split at 1.6 s | where the F1 is lost |
| Segments returned per reference segment (1 is right) | above 1: a system cuts too often |
| Wrong samples ↓ | the share of 0.1 s samples with a wrong view; row P shows what the 0.5 s grid alone costs |
| Piece midpoint accuracy ↑, overall, by view and by piece length (at most 1.6 s, longer) | is the middle sample of every piece given its view? This separates a piece read as another view from a cut that is a little off |
| Segment midpoint accuracy ↑ | does the middle sample of every returned segment carry the right view? |
| Wrong samples by kind | every wrong sample is `edge` (the cut is a little off), `spurious` (an extra segment), `absorbed` (a piece swallowed by its neighbour) or `misread` (a whole piece given another view) |
| Cuts: precision ↑ and recall ↑ within 0.5 s; recall within 0.1, 0.2, 0.3 and 1.0 s | cuts and view changes are matched one to one, nearest first (`metrics.py`) |
| Cuts: of those matched within 1.0 s, the mean distance in seconds ↓, with the matched share and the share that is late | matched one to one, most matches first, then least distance (`scorecard.py`) |
| Cuts at joins of two pieces of the same view ↓ | a system that cuts at every splice scores high |
| Per view: the share of a view's samples that are returned ↑, and the share of the samples returned as a view that are right ↑ | |

**Single recordings** (800): accuracy ↑, macro-F1 ↑, recall per view with the number of recordings, A4C read as A5C and A5C read as A4C, label changes inside one recording ↓, and against EchoRouter the number of recordings that only one of the two reads right. A recording's view is the majority of its decoded samples, for every system.

A count of 5 or fewer is given without an interval. SC4C has no piece in the pair streams and A5C has four.

**The code.** The numbers come from `echoview/evaluation/metrics.py` and `echoview/evaluation/scorecard.py`. New code is needed for the per-view segment F1, the split at 1.6 s, segments per reference segment, cut precision on pair streams, cut recall at other distances, the recordings only one system reads right, and all intervals of section 6. It is committed, reviewed and tried on the development benchmark in stage A, before the read.

## 6. The intervals

- **Draws.** Percentile intervals from 10,000 draws with a fixed seed.
- **Two units for the numbers about streams.** (a) The streams are drawn, as in the benchmark's published intervals. (b) The 125 examinations are drawn: every reference segment, returned segment, match, piece and cut is counted for the examination of the recording at its middle sample (for a cut, at the cut). The streams reuse recordings (151 A5C pieces come from 39 recordings), so (a) is too narrow for anything that depends on a few recordings. Both are computed, and **the wider one is reported and decides**.
- **Single recordings.** Examinations are drawn.
- **A difference.** EchoRouter minus another row, with the same draws for both.
- **Conditional.** All intervals are for these recordings and this way of building streams. They say nothing about other hospitals or machines.
- **Four comparisons are named in advance.** Segment F1 at 50% with the project's matching on multi streams, EchoRouter minus each of the four others with the project's decoder (row D). Because there are four, their 98.75% intervals decide. Where a system has no row D, its comparison is "not answered". Every other number in the table is a description, with a 95% interval.

## 7. How the result will be worded

These rules are fixed now.

1. **The named comparisons.** EchoRouter is "ahead of" a system when the 98.75% interval of the difference lies wholly above zero, "behind" it when wholly below, and otherwise "not shown to differ from" it. EchoRouter is called ahead of a system on streams only when, in addition, its paired 95% interval against that system's published row lies above zero.
2. **The gap in two parts.** For each of the four, the published gap is split: (EchoRouter minus the system with the decoder) plus (the system with the decoder minus the system as published). Each part is given with its paired 95% interval, in the form "of the published gap of G, R [low, high] remains when the system has the project's decoder". No sentence says where a lead comes from on the strength of an interval that holds zero.
3. **One sentence for all four** only when the four named comparisons give the same word. Otherwise the README lists them one by one.
4. **The decoder on EchoRouter.** Row N against the published row is reported in the same form: what the decoder adds to EchoRouter's own probabilities.
5. **The seeds.** The named comparisons are also computed for each seed alone. When a seed alone would get another word than the ensemble, the README says that the result depends on the seed.
6. **The best row.** For every number with a direction, the README names the row with the best value among A, D and N, and names rows with equal values together. This is by the value alone and is not a claim about a difference. The words of rule 1 are used for the named comparisons only.
7. **Everything is published.** Every row and number goes into this file, and the command's output into `docs/comparison/`. The README shows a subset fixed here, for rows A, D and N: segment F1 at 50% on multi streams and on pair streams, the Edit score, piece midpoint accuracy, wrong samples, precision and recall of the cuts, the mean distance of the cuts, and for single recordings accuracy, macro-F1 and A5C recall.
8. **Nothing is tuned on this read.** No weight, decoder setting or rule of EchoRouter changes on account of it. A number that is against EchoRouter is reported as it is. A change to the system would be chosen on development data and would need other data to confirm it.
9. **One read.** Stage B runs once.

## 8. What is expected, and the limits

**Expected.**

- EchoRouter is above the four as published. That is known.
- EchoViewCLIP with the project's decoder stays clearly below EchoRouter on multi streams, as on the development selection half (0.87 against 0.96). For STFM and ASFormer it is open. ASFormer is a temporal model already, so the decoder should change it little.
- On single recordings EchoViewCLIP is above EchoRouter. That is known. Whether it is also above by piece midpoint accuracy in streams is open.
- ASFormer and EchoPrime's classifier give an output for every sample and may place their cuts more exactly than the three systems that are read every 0.5 s.

**Limits known beforehand.**

- The benchmark is the project's own. Its streams are built by joining pieces of labelled recordings. They are not examinations, and the test benchmark has no streams in examination order. EchoRouter's model, the decoder's grid and its two floors were developed on streams built this way.
- It is not an untouched test set. Earlier rounds of the project were evaluated on it, and the numbers of section 2 were known.
- STFM, ASFormer and EchoViewCLIP were trained by us with their authors' recipes on EV9V's training recordings. They may do better in their authors' hands. How close our copies come to their authors' reported numbers on the authors' own protocols is in the project's research records, not in this package; it will be quoted beside the table where it was measured, and the table will say where it was not.
- STFM and EchoViewCLIP were stopped early on the validation recordings that form the calibration half. Their numbers on that half are optimistic, and a setting chosen there is chosen on data they have partly seen.
- Whether the data that EchoPrime and CLIP were pretrained on overlap with EV9V's test recordings is unknown.
- The reference view of a sample is the label of its recording. No reader marked single frames.
- A recording's view is decided by the majority of its samples for every system, not by each author's own rule for a clip.
- A plain smoothing filter as a third decoder was considered and left out: its width would be one more setting to choose.
- The read needs the owner's authorisation, as every read of the test benchmark does. It is recorded in section 9 when given.

## 9. Stage A

**Done on 2026-11-24 with the code of commit `377fb01`. No test output was scored.** The outputs of the three commands are in [`comparison/`](comparison/): `inventory_test.json`, `inventory_dev.json`, `settings.json` and `selection.json`. The owner authorised both stages on 2026-11-23.

### Changes to the plan, made before stage B

The code was reviewed before it ran. The review led to five changes. None of them loosens a rule.

1. **The word of a named comparison needs both units.** Section 6 says that the wider of the two intervals decides. Where the two intervals do not lie inside one another, the wider one can lie above zero while the other holds zero. The word now rests on the hull of the two: from the lower of the two lower bounds to the higher of the two upper bounds. Both intervals are still reported. This can only make "ahead of" and "behind" rarer.
2. **A difference has no interval when either of its two numbers rests on a count of 5 or fewer**, as a single number has none.
3. **The Edit score has the interval over streams only.** It is a score per stream and cannot be counted per examination.
4. **What the decoder adds to EchoPrime's classifier** is measured against its row per frame, because its published row already has the decoder.
5. **Before the test benchmark is read, the command checks what the plan fixes:** 10,000 draws, the four stores, the logits file by its SHA-256, a setting for each of the three seeds, and for every other system either a setting or the note that it has no development outputs. When one is missing, nothing is read.

### The stored outputs

| Stored outputs | Files | Test benchmark | Calibration half | Selection half | Rows sum to | Digest of the test files (first 16) |
|---|---|---|---|---|---|---|
| EchoViewCLIP, test folder | 2,068 | 1,160 of 1,160 | 237 of 472 | 284 of 541 | 0.07 to 1.00 | `a53bcc7d6a6426f7` |
| STFM, test folder | 2,068 | 1,160 of 1,160 | 237 of 472 | 284 of 541 | 0.22 to 0.98 | `00e549a82b3459d7` |
| ASFormer, test folder | 2,041 | 1,160 of 1,160 | 237 of 472 | 284 of 541 | 1.00 to 1.00 | `c3af7fdf80b775a2` |
| EchoPrime, test folder | 2,068 | 1,160 of 1,160 | 237 of 472 | 284 of 541 | 0.01 to 1.00 | `fc8c652c3e4207c6` |
| EchoViewCLIP, development folder | 1,013 | 0 of 1,160 | 472 of 472 | 541 of 541 | 0.07 to 1.00 | calibration `486f8c3d4e932c4f`, selection `d95b4aedb7719e21` |

- Every file holds the array `prob` with one row of five views per sample. No file has another shape. The digests are in full in the two inventory files.
- The test folders also hold the single recordings of the development halves, but none of their streams.
- **STFM and ASFormer have no stored outputs for the streams of the calibration half.** By section 4 they get no row D, and their named comparison is "not answered". No forward pass was asked for. Their row C is shown, as planned, and carries no sentence.
- EchoViewCLIP has outputs for both development halves and gets a row D. For EchoPrime's classifier the published row with the decoder is its row D.

### The decoder settings, chosen on the calibration half

| System | Feasible settings of 40 | Half window | Quantile | Threshold | Calibration half: segment F1 at 50% | Macro-F1 of recordings | Recall of the view changes |
|---|---|---|---|---|---|---|---|
| EchoRouter, the ensemble | 40 | 7 | 0.999 | 0.1738 | 0.984 | 0.986 | 0.990 |
| EchoRouter seed 0 | 40 | 7 | 0.999 | 0.1860 | 0.986 | 0.986 | 0.990 |
| EchoRouter seed 1 | 37 | 5 | 0.999 | 0.1297 | 0.983 | 0.986 | 0.987 |
| EchoRouter seed 2 | 36 | 5 | 0.999 | 0.1371 | 0.978 | 0.986 | 0.982 |
| EchoViewCLIP | 28 | 9 | 0.8 | 0.0007 | 0.973 | 0.986 | 0.980 |

- **The ensemble.** The rule gives the released grid position: half window 7 at the quantile 0.999. On these logits the threshold is 0.17383, and the released one is 0.17379. The stored logits were computed on another machine than the one the released number came from. The check accepts a threshold within 0.001 at the same grid position.
- **EchoViewCLIP.** 28 of the 40 settings meet the floors, and the rule takes half window 9 at the quantile 0.80. This is the choice, and these are the numbers, of the project's record of 2026-10-05 (section 2). Reading the stores, decoding them and applying the rule here give what the research code gave then.

### The selection half, as a trial

These are development numbers. The benchmark names no examinations there, so the intervals are over streams only.

| Row | Segment F1 at 50%, multi | Segment F1 at 50%, pairs | Edit | Piece midpoint | Wrong samples | Cut precision | Cut recall | Cut distance (s) | Recordings: accuracy | Macro-F1 | A5C recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EchoRouter | 0.959 | 0.971 | 96.5 | 97.8% | 6.1% | 0.959 | 0.976 | 0.15 | 98.9% | 0.979 | 100.0% |
| EchoViewCLIP | 0.842 | 0.941 | 86.9 | 92.8% | 12.0% | 0.827 | 0.961 | 0.19 | 98.2% | 0.957 | 81.8% |
| EchoViewCLIP + decoder | 0.866 | 0.967 | 88.6 | 92.8% | 11.7% | 0.853 | 0.958 | 0.19 | 98.2% | 0.957 | 81.8% |
| EchoRouter, no decoder | 0.906 | 0.957 | 91.6 | 97.1% | 7.0% | 0.858 | 0.984 | 0.16 | 99.3% | 0.988 | 100.0% |
| EchoViewCLIP + released numbers | 0.866 | 0.946 | 87.6 | 90.7% | 12.7% | 0.942 | 0.885 | 0.18 | 98.2% | 0.957 | 81.8% |
| EchoRouter seed 0 | 0.958 | 0.971 | 96.3 | 97.7% | 6.3% | 0.956 | 0.971 | 0.16 | 98.9% | 0.979 | 100.0% |
| EchoRouter seed 1 | 0.948 | 0.978 | 95.0 | 97.1% | 6.7% | 0.946 | 0.963 | 0.15 | 98.9% | 0.979 | 100.0% |
| EchoRouter seed 2 | 0.948 | 0.974 | 95.5 | 96.7% | 6.6% | 0.953 | 0.958 | 0.15 | 99.3% | 0.988 | 100.0% |
| Perfect on the 0.5 s grid | 1.000 | 1.000 | 100.0 | 100.0% | 3.1% | 1.000 | 1.000 | 0.12 | 100.0% | 1.000 | 100.0% |

- **The named comparison on development data.** EchoRouter minus EchoViewCLIP with the decoder is 0.094 [0.063, 0.127]; the 98.75% interval is [0.055, 0.135]. Of the gap of 0.117 [0.081, 0.155] to EchoViewCLIP as published, 0.094 remains when it has the project's decoder. Each seed alone is also ahead of EchoViewCLIP with the decoder.
- **What the decoder adds.** 0.023 [0.007, 0.041] to EchoViewCLIP and 0.054 [0.029, 0.080] to EchoRouter, on multi streams. EchoRouter without the decoder (0.906) is above EchoViewCLIP with it (0.866).
- **The 0.5 s grid.** A perfect system that is read every 0.5 s has 3.1% wrong samples and cuts that are 0.12 s off on average.

## 10. Stage B: the result

Read once on 2026-11-24 at 02:55 UTC with the code of commit `377fb01`. The output of the command is [`comparison/test.json`](comparison/test.json), SHA-256 `4f00000524df0581ee7f093980135a206c9180f51d15457524acb6547afe45c0`, unchanged. Before anything was scored, the stored outputs gave the published numbers of all six rows of section 3 to the sixth decimal. Intervals are 95% unless said otherwise, from 10,000 draws. Where a number has an interval by stream and one by examination, the wider one is shown.

### The four named comparisons

Segment F1 at 50% with the project's matching, multi streams. EchoRouter scores 0.909 [0.872, 0.944].

| System | Its row with the decoder | EchoRouter minus that row | 98.75% interval by stream | 98.75% interval by examination | Word |
|---|---|---|---|---|---|
| EchoViewCLIP | 0.884 | 0.025 | [-0.005, 0.061] | [-0.010, 0.063] | not shown to differ from |
| STFM | none | – | – | – | not answered |
| ASFormer | none | – | – | – | not answered |
| EchoPrime | 0.851 | 0.058 | [0.026, 0.093] | [0.010, 0.109] | ahead of |

- **EchoViewCLIP.** With the project's decoder it scores 0.884. EchoRouter is not shown to differ from it: the difference is 0.025, and both 98.75% intervals hold zero.
- **EchoPrime's classifier.** EchoRouter is ahead of it: 0.058, and both intervals lie above zero.
- **STFM and ASFormer.** Not answered. They have no stored outputs for the streams of the calibration half, so no decoder setting could be chosen for them by the rule (section 9).
- **The seeds.** Each seed alone scores 0.913, 0.916 and 0.908 and gets the same four words as the ensemble. The result does not depend on the seed.

### The gap in two parts

- **EchoViewCLIP.** Of the published gap of 0.042 [0.009, 0.076], 0.025 [-0.003, 0.054] remains when it has the project's decoder. The decoder adds 0.017 [0.003, 0.040] to it.
- **STFM.** The published gap is 0.050 [0.017, 0.084]. There is no row with the decoder.
- **ASFormer.** The published gap is 0.053 [0.012, 0.094]. There is no row with the decoder.
- **EchoPrime's classifier.** Its published row has the decoder, so the published gap is what remains: 0.058 [0.018, 0.098]. Against its row per frame the decoder adds 0.355 [0.289, 0.406].
- **EchoRouter itself.** The decoder adds 0.035 [0.010, 0.063] on multi streams and 0.009 [-0.000, 0.020] on pair streams.

### Every row

Bold marks the best value among the rows as published, with the project's decoder, and EchoRouter without its decoder. It is by the value alone and claims no difference.

| Row | What it is | Segment F1 at 50%, multi | Segment F1 at 50%, pairs | Edit | Piece midpoint | Wrong samples | Cut precision | Cut recall | Cut distance (s) | Recordings: accuracy | Macro-F1 | A5C recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EchoRouter | as published | **0.909** | 0.985 | **91.4** | 93.7% | 9.4% | 0.934 | 0.937 | 0.153 | 98.1% | 0.956 | **95.0%** |
| EchoViewCLIP | as published | 0.867 | 0.989 | 89.0 | 92.9% | 11.2% | 0.851 | 0.953 | 0.183 | **98.9%** | **0.972** | 92.5% |
| STFM | as published | 0.859 | 0.971 | 88.1 | 92.9% | 12.0% | 0.812 | 0.879 | 0.252 | 98.2% | 0.960 | 87.5% |
| ASFormer | as published | 0.856 | 0.981 | 87.2 | 91.0% | **9.0%** | 0.869 | 0.942 | 0.013 | 98.2% | 0.956 | 77.5% |
| EchoPrime per frame | as published | 0.496 | 0.743 | 54.9 | 89.6% | 12.1% | 0.357 | **0.974** | 0.017 | 97.9% | 0.943 | 67.5% |
| EchoViewCLIP + decoder | with the project's decoder | 0.884 | **0.993** | 90.1 | 92.9% | 11.1% | 0.877 | 0.955 | 0.184 | 98.8% | 0.969 | 90.0% |
| EchoPrime + decoder | with the project's decoder | 0.851 | 0.988 | 85.0 | 88.3% | 11.9% | **0.945** | 0.900 | **0.005** | 97.9% | 0.943 | 67.5% |
| EchoRouter, no decoder | without the decoder | 0.874 | 0.975 | 89.5 | **94.0%** | 9.1% | 0.848 | 0.955 | 0.163 | 98.4% | 0.960 | 92.5% |
| EchoViewCLIP + released numbers | released numbers carried over | 0.867 | 0.957 | 86.0 | 89.6% | 13.4% | 0.922 | 0.868 | 0.184 | 98.6% | 0.966 | 87.5% |
| STFM + released numbers | released numbers carried over | 0.100 | 0.753 | 27.4 | 44.8% | 50.3% | 1.000 | 0.003 | 0.500 | 98.4% | 0.963 | 90.0% |
| ASFormer + released numbers | released numbers carried over | 0.883 | 0.985 | 89.4 | 91.0% | 9.0% | 0.932 | 0.934 | 0.013 | 98.2% | 0.956 | 77.5% |
| EchoPrime + released numbers | released numbers carried over | 0.829 | 0.966 | 84.9 | 89.6% | 11.0% | 0.816 | 0.958 | 0.016 | 97.6% | 0.937 | 65.0% |
| EchoRouter seed 0 | one seed | 0.913 | 0.985 | 91.8 | 93.9% | 9.5% | 0.941 | 0.934 | 0.162 | 98.1% | 0.955 | 92.5% |
| EchoRouter seed 1 | one seed | 0.916 | 0.981 | 91.7 | 93.9% | 9.2% | 0.942 | 0.937 | 0.148 | 98.0% | 0.953 | 92.5% |
| EchoRouter seed 2 | one seed | 0.908 | 0.976 | 91.6 | 93.9% | 9.5% | 0.934 | 0.934 | 0.156 | 98.2% | 0.959 | 95.0% |
| Perfect on the 0.5 s grid | reference | 1.000 | 1.000 | 100.0 | 100.0% | 3.0% | 1.000 | 1.000 | 0.121 | 100.0% | 1.000 | 100.0% |

- **The rows with the released numbers carried over** carry no sentence, as planned. For STFM the row shows why: the threshold was a quantile of EchoRouter's own scores, and on STFM's outputs it lets almost no cut through.
- **The segment F1 of every row, with both intervals:**

| Row | Segment F1 at 50%, multi | 95% interval by stream | 95% interval by examination |
|---|---|---|---|
| EchoRouter | 0.909 | [0.883, 0.933] | [0.872, 0.944] |
| EchoViewCLIP | 0.867 | [0.830, 0.900] | [0.805, 0.925] |
| STFM | 0.859 | [0.827, 0.889] | [0.802, 0.913] |
| ASFormer | 0.856 | [0.820, 0.891] | [0.793, 0.918] |
| EchoPrime per frame | 0.496 | [0.449, 0.545] | [0.405, 0.604] |
| EchoViewCLIP + decoder | 0.884 | [0.851, 0.914] | [0.827, 0.936] |
| EchoPrime + decoder | 0.851 | [0.818, 0.882] | [0.796, 0.904] |
| EchoRouter, no decoder | 0.874 | [0.840, 0.906] | [0.818, 0.926] |
| EchoViewCLIP + released numbers | 0.867 | [0.837, 0.895] | [0.818, 0.913] |
| STFM + released numbers | 0.100 | [0.068, 0.134] | [0.069, 0.129] |
| ASFormer + released numbers | 0.883 | [0.853, 0.911] | [0.824, 0.937] |
| EchoPrime + released numbers | 0.829 | [0.794, 0.863] | [0.765, 0.891] |
| EchoRouter seed 0 | 0.913 | [0.888, 0.936] | [0.876, 0.947] |
| EchoRouter seed 1 | 0.916 | [0.891, 0.939] | [0.880, 0.948] |
| EchoRouter seed 2 | 0.908 | [0.881, 0.933] | [0.869, 0.944] |
| Perfect on the 0.5 s grid | 1.000 | [1.000, 1.000] | [0.993, 1.007] |

  With examinations as the unit, a match and its returned segment can fall into different examinations, so a resample can score a little above 1. The row of the perfect system shows the size of this: 0.007.

### Where segments are lost (multi streams)

| Row | MS-TCN F1 at 10% | at 25% | at 50% | Returned segments that match nothing (at most 1.6 s) | Reference segments missed (at most 1.6 s) | Segments per reference segment | Cuts at joins of the same view |
|---|---|---|---|---|---|---|---|
| EchoRouter | 0.939 | 0.939 | 0.909 | 46 (22) | 45 (15) | 1.00 | 4.0% |
| EchoViewCLIP | 0.920 | 0.911 | 0.867 | 92 (62) | 47 (11) | 1.09 | 8.4% |
| STFM | 0.915 | 0.910 | 0.859 | 88 (60) | 57 (21) | 1.06 | 8.9% |
| ASFormer | 0.895 | 0.891 | 0.856 | 90 (48) | 58 (16) | 1.06 | 9.9% |
| EchoPrime per frame | 0.584 | 0.561 | 0.496 | 745 (696) | 89 (15) | 2.31 | 28.7% |
| EchoViewCLIP + decoder | 0.928 | 0.924 | 0.884 | 77 (45) | 43 (9) | 1.07 | 8.4% |
| EchoPrime + decoder | 0.894 | 0.888 | 0.851 | 64 (17) | 82 (23) | 0.96 | 5.9% |
| EchoRouter, no decoder | 0.929 | 0.920 | 0.874 | 90 (67) | 42 (10) | 1.10 | 8.4% |

### Wrong samples and pieces (multi streams)

| Row | Wrong samples | edge | spurious | absorbed | misread | Piece midpoint, pieces of at most 1.6 s | longer pieces | Segment midpoint |
|---|---|---|---|---|---|---|---|---|
| EchoRouter | 9.4% | 3.2% | 0.5% | 1.4% | 4.3% | 94.4% | 93.4% | 92.8% |
| EchoViewCLIP | 11.2% | 3.9% | 1.1% | 1.2% | 5.0% | 94.4% | 92.1% | 89.5% |
| STFM | 12.0% | 5.0% | 0.9% | 1.6% | 4.5% | 91.8% | 93.4% | 89.2% |
| ASFormer | 9.0% | 0.2% | 0.3% | 1.2% | 7.3% | 91.8% | 90.6% | 86.8% |
| EchoPrime per frame | 12.1% | 0.2% | 4.4% | 0.2% | 7.4% | 93.1% | 87.9% | 63.6% |
| EchoViewCLIP + decoder | 11.1% | 3.8% | 1.0% | 1.1% | 5.2% | 94.4% | 92.1% | 90.4% |
| EchoPrime + decoder | 11.9% | 0.1% | 0.3% | 2.4% | 9.1% | 91.3% | 86.8% | 88.6% |
| EchoRouter, no decoder | 9.1% | 3.4% | 1.0% | 1.0% | 3.7% | 95.2% | 93.4% | 90.9% |
| Perfect on the 0.5 s grid | 3.0% | 3.0% | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% | 100.0% |

### Cuts (multi streams)

| Row | Precision within 0.5 s | Recall within 0.1 s | 0.2 s | 0.3 s | 0.5 s | 1.0 s | Mean distance (s) | Late |
|---|---|---|---|---|---|---|---|---|
| EchoRouter | 0.934 | 52.5% | 79.2% | 90.0% | 93.7% | 95.0% | 0.153 | 46.4% |
| EchoViewCLIP | 0.851 | 47.0% | 68.3% | 85.2% | 95.3% | 96.6% | 0.183 | 51.4% |
| STFM | 0.812 | 34.3% | 55.7% | 68.3% | 87.9% | 96.0% | 0.252 | 56.3% |
| ASFormer | 0.869 | 91.6% | 93.1% | 93.7% | 94.2% | 94.7% | 0.013 | 2.5% |
| EchoPrime per frame | 0.357 | 94.5% | 95.8% | 96.8% | 97.4% | 98.4% | 0.017 | 2.4% |
| EchoViewCLIP + decoder | 0.877 | 47.2% | 70.4% | 85.2% | 95.5% | 97.1% | 0.184 | 51.1% |
| EchoPrime + decoder | 0.945 | 88.7% | 89.4% | 89.7% | 90.0% | 90.0% | 0.005 | 1.5% |
| EchoRouter, no decoder | 0.848 | 52.8% | 79.7% | 91.0% | 95.5% | 97.9% | 0.163 | 46.9% |
| Perfect on the 0.5 s grid | 1.000 | 60.4% | 100.0% | 100.0% | 100.0% | 100.0% | 0.121 | 37.7% |

### By view (multi streams)

| Row | Piece midpoint, PLAX | Piece midpoint, PSAX | Piece midpoint, A4C | Piece midpoint, A5C | Piece midpoint, SC4C | Segment F1, PLAX | Segment F1, PSAX | Segment F1, A4C | Segment F1, A5C | Segment F1, SC4C |
|---|---|---|---|---|---|---|---|---|---|---|
| EchoRouter | 99.3% | 100.0% | 92.9% | 82.8% | 94.4% | 0.986 | 1.000 | 0.816 | 0.811 | 0.957 |
| EchoViewCLIP | 100.0% | 100.0% | 94.2% | 76.8% | 94.4% | 1.000 | 1.000 | 0.745 | 0.701 | 0.957 |
| STFM | 100.0% | 99.3% | 91.6% | 79.5% | 95.3% | 0.981 | 0.925 | 0.754 | 0.709 | 0.976 |
| ASFormer | 100.0% | 100.0% | 94.8% | 66.9% | 95.3% | 1.000 | 0.985 | 0.731 | 0.680 | 0.929 |
| EchoPrime per frame | 98.0% | 99.3% | 96.1% | 66.2% | 88.8% | 0.751 | 0.660 | 0.372 | 0.236 | 0.809 |
| EchoViewCLIP + decoder | 100.0% | 100.0% | 94.2% | 76.8% | 94.4% | 1.000 | 1.000 | 0.774 | 0.741 | 0.957 |
| EchoPrime + decoder | 100.0% | 97.9% | 98.1% | 57.0% | 89.7% | 0.986 | 0.984 | 0.710 | 0.667 | 0.938 |
| EchoRouter, no decoder | 99.3% | 100.0% | 92.9% | 83.4% | 95.3% | 0.986 | 0.990 | 0.787 | 0.709 | 0.963 |

### Pair streams

| Row | Segment F1 at 50% | Edit | Piece midpoint | Wrong samples | Cut precision | Cut recall | Segments that match nothing | Segments missed |
|---|---|---|---|---|---|---|---|---|
| EchoRouter | 0.985 | 98.6 | 98.8% | 2.8% | 0.992 | 0.983 | 5 | 6 |
| EchoViewCLIP | 0.989 | 99.2 | 99.4% | 2.6% | 0.959 | 0.975 | 5 | 3 |
| STFM | 0.971 | 97.9 | 98.3% | 3.7% | 0.886 | 0.908 | 12 | 9 |
| ASFormer | 0.981 | 99.1 | 99.2% | 0.7% | 0.952 | 1.000 | 10 | 4 |
| EchoPrime per frame | 0.743 | 81.6 | 97.1% | 2.2% | 0.365 | 1.000 | 224 | 15 |
| EchoViewCLIP + decoder | 0.993 | 99.2 | 99.4% | 2.6% | 0.967 | 0.975 | 3 | 2 |
| EchoPrime + decoder | 0.988 | 98.9 | 99.6% | 0.5% | 0.952 | 0.992 | 7 | 2 |
| EchoRouter, no decoder | 0.975 | 98.6 | 99.0% | 2.7% | 0.944 | 0.992 | 12 | 6 |

### Single recordings

| Row | Accuracy | Macro-F1 | Recall PLAX | PSAX | A4C | A5C | SC4C | A4C read as A5C | A5C read as A4C | Label changes per recording | Read right only by EchoRouter and only by this row |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EchoRouter | 98.1% [97.1, 99.0] | 0.956 [0.929, 0.975] | 100.0% | 100.0% | 94.8% | 95.0% | 97.1% | 10 | 2 | 0.02 | – |
| EchoViewCLIP | 98.9% [98.0, 99.6] | 0.972 [0.952, 0.988] | 100.0% | 100.0% | 97.8% | 92.5% | 97.1% | 5 | 3 | 0.10 | 4 and 10 |
| STFM | 98.2% [97.3, 99.1] | 0.960 [0.933, 0.978] | 100.0% | 100.0% | 96.5% | 87.5% | 97.1% | 7 | 4 | 0.13 | 6 and 7 |
| ASFormer | 98.2% [97.1, 99.2] | 0.956 [0.929, 0.978] | 100.0% | 99.5% | 98.7% | 77.5% | 97.1% | 2 | 9 | 0.14 | 8 and 9 |
| EchoPrime per frame | 97.9% [96.2, 99.1] | 0.943 [0.907, 0.975] | 100.0% | 99.5% | 99.6% | 67.5% | 94.3% | 1 | 13 | 1.62 | 14 and 12 |
| EchoViewCLIP + decoder | 98.8% [97.9, 99.5] | 0.969 [0.948, 0.986] | 100.0% | 100.0% | 97.8% | 90.0% | 97.1% | 5 | 4 | 0.09 | 4 and 9 |
| EchoPrime + decoder | 97.9% [96.5, 99.0] | 0.943 [0.914, 0.971] | 100.0% | 99.5% | 99.6% | 67.5% | 94.3% | 1 | 13 | 0.04 | 13 and 11 |
| EchoRouter, no decoder | 98.4% [97.4, 99.2] | 0.960 [0.935, 0.979] | 100.0% | 100.0% | 96.1% | 92.5% | 97.1% | 7 | 3 | 0.10 | 1 and 3 |

### How to read it

- **With the same decoder, the lead over EchoViewCLIP is not shown.** As published, EchoRouter is 0.042 above EchoViewCLIP on multi streams. Given the project's decoder, EchoViewCLIP gains 0.017, and the 0.025 that remains lies within what resampling these streams and examinations can produce. Against EchoPrime's classifier with the decoder the lead holds. For STFM and ASFormer the question is open.
- **What section 8 expected was wrong on this point.** It expected EchoViewCLIP with the decoder to stay clearly below EchoRouter, as on the development selection half (0.866 against 0.959). On the test benchmark EchoViewCLIP with the decoder scores 0.884 and EchoRouter 0.909. EchoRouter loses 0.05 from the selection half to the test benchmark, EchoViewCLIP with the decoder loses nothing. EchoRouter's model, its decoder's grid and its floors were developed on the development streams, and its development numbers are optimistic to that extent.
- **What EchoRouter does differently is A5C.** The middle sample of an A5C piece is right for 82.8% of the pieces, against 57.0% to 79.5% for the other systems, as published or with the decoder, and its segment F1 for A5C is 0.811 against 0.741 at most. It also returns the fewest segments that match nothing (46, the others 64 to 92 and 745 per frame) and cuts least often at a join of two pieces of the same view (4.0%). The price is A4C: it reads A4C as A5C in 10 single recordings, the others in 1 to 7.
- **Its cuts are as exact as a 0.5 s grid allows, and no more.** 52.5% of the view changes have a cut within 0.1 s; a perfect system on that grid would have 60.4%. ASFormer and EchoPrime's classifier, which give an output for every sample, have about 90%, and their cuts are 0.01 s off on average where EchoRouter's are 0.15 s off.
- **Of EchoRouter's 9.4% wrong samples**, 3.2 points are edges, which is what the grid alone costs (3.0%), and 4.3 points lie in pieces that are read as another view as a whole. ASFormer has the fewest wrong samples (9.0%), almost all of them in pieces read wrong as a whole (7.3 points).
- **Single recordings.** EchoViewCLIP is the best row: 98.9% against EchoRouter's 98.1%, a difference of −0.8 points [−1.7, +0.1], and a macro-F1 of 0.972 against 0.956, −0.016 [−0.038, +0.003]. Four recordings are read right only by EchoRouter and ten only by EchoViewCLIP. EchoRouter has the highest A5C recall, 38 of 40.
- **Pair streams.** No row stands out. EchoViewCLIP with the decoder has the highest value (0.993); EchoRouter has 0.985.
- **The limits of section 8 stand.** The benchmark is the project's own and is not untouched, its streams are built and are not examinations, three of the four other systems were trained by us, and A5C rests on 39 recordings.
