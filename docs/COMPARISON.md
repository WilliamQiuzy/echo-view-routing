# The comparison with other approaches, row by row

**Status: stage A is done (section 9). Stage B, the read of the test benchmark, has not run.** Sections 1 to 8 are the plan. They were written, reviewed and committed before either stage, as commit `26571f5`. Stage A touches no test data: it lists the stored outputs, chooses decoder settings on development data and tries the code on the development benchmark. Its results go into section 9 and are committed. Stage B is one read of the test benchmark. Its results are added after section 9.

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
