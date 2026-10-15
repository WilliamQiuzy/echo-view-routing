# Segment-balanced and transcript-forward training (2026-10-14)

This round changed only the training objective of one small head on fixed video features. It asked whether
segment-balanced errors, or a loss over each training sequence's whole class order, give better complete segmentation
than frame-weighted errors.

**What was trained.**
- Each head is one `Linear(512, 5)` with **2,565 trainable parameters**. It reads the frozen V01 Ordered 512-d video
  features from A01's caches.
- **The whole video encoder was unchanged.** No feature was re-extracted, and no encoder weight entered an optimiser.
- Each of three seeds trained three heads, so nine in all. Each head started from a bit-identical copy of A01's linear
  initial state, which is the seed's V01 source classifier.
- The three heads of a seed saw A01's exact batches (every update's digest equals A01's plan) and ran A01's optimiser and
  schedule for 18,750 updates. The final weights were kept.

**The three arms.**
- **F (frame-weighted).** A01's masked class-weighted cross-entropy. Every sample counts, weighted by its class. F
  repeats A01's linear condition as a matched control.
- **B (segment-balanced).** Every reference run (a maximal stretch of one class) contributes its class weight times its
  mean cross-entropy. A long run therefore no longer outweighs a short one just by having more samples.
- **O (B plus a transcript term).** (B + 0.5 × L_order) / 1.5. L_order is the negative log-probability, per sample, of
  the sequence's run-length-collapsed class order. That probability sums over every way of placing the classes in that
  order with positive durations, with no blank class and no maximum over paths.

**Training versus inference.** Labels, runs and transcripts entered the training losses only. At inference every head
gives per-sample softmax probabilities from the features alone. They are decoded with the simple change-point decoder
B3, chosen once per head on calibration. No transcript, label, join or id reaches a head or the decoder.

The task sheet is `docs/superpowers/plans/2026-10-14-segment-objective-training.md` (a13276e). The fixed configuration is
`configs/experiments/segment_objective_v1.yaml`.

**What kind of evidence this is.**
- **Reused data.** The two development halves (selection and calibration) have been used and inspected in many earlier
  rounds. This is a matched comparison on reused development data, not an independent validation.
- **Grouping.** The hold-out is by session, not by patient. Constructed streams are spliced from separate recordings and
  are not real scans. They reuse recordings, so their units are not independent cases. The three seeds share the same
  cases.
- **Labels.** Native clips carry the public recording label. No label was changed.
- **Untouched data.** The test split and EchoNet-Pediatric were not read.
- **Novelty.** Segment weighting and sequence-probability training are established ideas. This round is a bounded test
  of one recipe, not a new architecture.

**Order of work.** All times are UTC on 2026-10-15.
- **Before any code.** a13276e committed the task sheet at 00:47. 6088a5b committed the configuration at 01:00.
- **Tests and code.** 787bb42 committed the tests at 01:11, and they failed at import. 2fb4ffd committed the code at
  01:24. The first preparation and check ran at 01:26.
- **Review.** An independent implementation review passed the code with fixes (`code_review.md`). c7e8e70 made the fixes
  at 01:56. The preparation and check ran again at 01:57, and the reviewer verified the fixes.
- **Training.** The nine trainings ran from 02:05:54 to 02:27:09, one after another, and were then frozen.
- **Calibration.** It ran from 02:27 to 02:32, and a dry run of the selection code followed. The dry run used the
  calibration half for both halves.
- **Barrier.** eaff3e1 committed the calibration choices with their provenance at 02:37:47. A git-object proof of that
  commit was made at 02:37:57.
- **Selection.** The selection half was opened once, at 02:38:26, and evaluated by 02:42:58.
- **Fact-check.** An independent fact-check then recomputed the results (see "Fact-check").

## Answers

Values are given for seeds 0, 1 and 2 in that order, on the selection half and then the calibration half. The main
metric is multi-stream segment F1@50 with each head's B3 setting. A difference is consistent only when all three seeds'
95% intervals exclude zero strictly on the same side in a half. Otherwise it is "no consistent difference detected",
which is not the same as equality.

### 1. The primary comparison O − B shows no consistent difference

- **Segment F1@50.** O − B is +0.0031, +0.0019 and −0.0010 on selection, and +0.0010, −0.0001 and +0.0010 on
  calibration. Every interval includes zero. Two calibration intervals start exactly at zero.
- **Other metrics.** No consistent difference was detected in native macro-F1, multi boundary recall, multi
  false-positive (FP) or false-negative (FN) segment counts, or native switches per clip.
- **Reference segments.** Over the three seeds, on multi streams, O recovered 4 reference segments that B missed and
  lost 2 that B matched on selection. On calibration it recovered 2 and lost 3. Each per-seed count rests on at most 3
  sessions, fewer than 10.
- **Conclusion.** Under this recipe, adding the transcript-forward term to the segment-balanced loss gave no consistent
  segment difference.

### 2. Segment balancing (B − F) and O − F show no consistent difference; seed 2 differs at its own B3 settings

- **B − F.** It is +0.0008, −0.0019 and +0.0232 on selection, and +0.0000, +0.0010 and +0.0074 on calibration. Only
  seed 2's selection interval [+0.0090, +0.0413] lies above zero.
- **O − F.** It is +0.0040, +0.0000 and +0.0222 on selection, and +0.0010, +0.0009 and +0.0084 on calibration. Seed 2's
  intervals lie above zero in both halves. Seeds 0 and 1 include zero.
- **Seed 2's FP segments.** The seed-2 gap comes with fewer multi-stream FP segments. B − F is −27 [−45, −13] on
  selection and −10 [−17, −4] on calibration.
- **The decoder settings in seed 2.**
  - F's seed-2 calibration choice was half-window 9 with threshold quantile 0.995. Only 3 of the 40 settings were
    feasible for it. B's choice was half-window 11 with quantile 0.999, from 38 feasible settings.
  - On selection, F's seed-2 boundary precision is 0.912 and B's is 0.971.
- **At common decoders.** Every head was also decoded with the seed's stored frozen Ordered setting and with direct
  argmax (section 7).
  - At the stored setting, seed 2's B − F is −0.0011 [−0.0033, +0.0000] on selection and −0.0001 [−0.0043, +0.0050] on
    calibration.
  - With direct argmax, seed 2's B − F is +0.0079 [+0.0008, +0.0153] on selection and +0.0037 [−0.0046, +0.0133] on
    calibration. O − F is +0.0069 [+0.0018, +0.0124] on selection.
- **Reading.** At each head's own B3 setting, the seed-2 gap coincides with F's seed-2 setting (3 of 40 feasible,
  boundary precision 0.912). At the common stored setting the gap is near zero. With direct argmax, seed 2 still favours
  B and O on selection. No cause is established.

### 3. Against frozen Ordered + B3, no consistent difference; lower calibration means

- **O − frozen Ordered.** It is +0.0128, −0.0034 and −0.0034 on selection, and −0.0039, −0.0049 and +0.0019 on
  calibration.
- **B − frozen Ordered.** It is +0.0096, −0.0054 and −0.0024 on selection, and −0.0049, −0.0048 and +0.0009 on
  calibration.
- **Intervals.** Only B's seed-1 calibration interval [−0.0102, −0.0009] excludes zero. No difference is consistent.
- **Means.** The three-seed means are 0.9550 / 0.9788 for B and 0.9564 / 0.9794 for O, against 0.9544 / 0.9817 for
  frozen Ordered + B3. Both calibration means are below the reference's.
- **Reference segments (multi streams, summed over seeds).**
  - On selection, O recovered 10 and lost 19 reference segments, B recovered 7 and lost 18, and F recovered 8 and lost
    16. On calibration the counts are 1 and 9 for O, 1 and 8 for B, and 2 and 9 for F. Most of these are A4C or A5C
    segments.
  - No head recovered an A4C/A5C X-Y-X middle against the reference. Each head lost 2 such middles, in seed 0 on
    selection.
  - Each per-seed count rests on fewer than 10 sessions (0 to 7).
- **FP segments.** On selection, B (25, 20 and 17) and O (25, 18 and 17) have fewer multi-stream FP segments than frozen
  Ordered (33, 22 and 21) in every seed. F has fewer in seeds 0 and 1 (27 and 18) but more in seed 2 (44). O − frozen
  Ordered FP is −8, −4 and −4 on selection, with every interval including zero.

### 4. Native recognition: one clip per seed breaks the floor for B and O

- **F.** It keeps native macro-F1 0.9786 on selection in every seed and has no constraint failure.
- **The extra misread clips.**
  - B misreads one more A4C clip than F on selection in seeds 1 and 2. In seed 1 it is read as A5C, and in seed 2 as
    SC4C.
  - O reads one more A4C clip as A5C than F on selection in seeds 0 and 1.
- **Floor failures.** Each of these single clips lowers native macro-F1 to 0.9701 or 0.9700, below the 0.973 selection
  floor. So B fails the floor in seeds 1 and 2, and O fails it in seeds 0 and 1. Each failure rests on one clip and one
  session.
- **Calibration.** On calibration no head fails a floor. B's seed 2 reads one A4C clip fewer as A5C than F.
- **The bootstrap.** No native macro-F1 difference is consistent.

### 5. Transcripts, boundaries and switches

- **Transcript distance.** The normalised transcript edit distance of multi streams with B3 compares the predicted and
  the reference class orders. On selection it is 0.059, 0.039 and 0.043 for O, against 0.059, 0.041 and 0.042 for B and
  0.062, 0.042 and 0.088 for F. On calibration it is 0.031, 0.029 and 0.027 for O, against 0.032, 0.028 and 0.028 for B
  and 0.032, 0.030 and 0.042 for F.
  - O is lower than B in 4 of the 6 seed–half cells.
  - B is lower than F in 5 of 6 and equal in one.
  - Frozen Ordered's calibration values (0.028, 0.021 and 0.030) are the lowest in two of three seeds.
- **Boundary recall.** Every head reaches at least 0.9529 in every seed and half, so the recall floor holds.
- **Native switches.** Counting label switches inside native clips, B has 16, 10 and 11 and O has 16, 8 and 13 on
  selection, against 13, 11 and 24 for F. On calibration B and O have 3 or 4 per seed, against 3, 4 and 10 for F. No
  switch difference is consistent. These counts come from 3 to 10 clips and 3 to 8 sessions per seed and half, fewer
  than 10 sessions.

### 6. F reproduces A01 Linear

- **Probabilities and weights.** F's probabilities differ from A01 Linear's by at most 1.04e-6 on either half, with no
  argmax change. Its final weights differ by at most 8.2e-8.
- **Decoded outputs.** Every decoded output and every metric equals A01 Linear's in all seeds and halves.
- **B3 choices.** F's calibration choices equal A01's in status, setting and feasible count (23, 6 and 3). The chosen
  threshold values differ by at most 2.0e-8.
- **Numerics.** F accumulates its loss in float64 from fp32 log-softmax values, while A01 used fp32 cross-entropy. On a
  formal batch the losses differ by 1.5e-8 and the gradients by 2.2e-8. This left no trace in the decoded outputs.

### 7. Sensitivities

These rows show the same heads with other decoders. Nothing is chosen from them.
- **Direct argmax.**
  - O − B is +0.0010, +0.0079 and −0.0010 on selection, and −0.0054, +0.0000 and −0.0066 on calibration.
  - B − F is +0.0017, +0.0083 and +0.0079, then +0.0100, +0.0055 and +0.0037.
  - O − F is +0.0027, +0.0161 and +0.0069, then +0.0046, +0.0055 and −0.0029.
  - None is consistent.
- **The seed's stored frozen Ordered setting.** This is the same numeric B3 setting for all three heads of a seed.
  - O − B is −0.0019, +0.0043 and +0.0043, then +0.0020, +0.0050 and +0.0050.
  - B − F is +0.0031, −0.0019 and −0.0011, then +0.0040, −0.0011 and −0.0001.
  - O − F is +0.0012, +0.0024 and +0.0032, then +0.0059, +0.0039 and +0.0049.
  - None is consistent. Several of these intervals start exactly at zero.

### 8. Goal and the fixed readings

- **The goal.** It requires three-seed means of at least 0.944409687184662 (selection) and 0.9821421709894332
  (calibration), and the floors in every seed and half.
  - Both B and O pass the selection mean and miss the calibration mean. Their calibration sensitivity margins are
    −0.0051 and −0.0045.
  - Both also fail the native floor in two selection cells (section 4).
  - The goal is not met. F (0.9476 / 0.9760) and frozen Ordered + B3 (0.9544 / 0.9817, one native-floor failure) do
    not meet it either.
- **Applied readings: uncertain or no qualifying gain, and end.**
  - **Not applied: complete system progress, segment balancing effect and order objective effect.** No primary or
    mandatory comparison is consistent, and B and O each fail a constraint.
  - **Not applied: transcript-only or local effect.** Its operational test was that O − B or B − F lowers the multi edit
    distance in every seed and half, or recovers more multi segments than it loses in both halves. Neither held.
  - **Not applied: trade-off.** Its operational test was a pair with recovered multi segments together with
    consistently more FP segments, a consistently lower native macro-F1 or a recall-floor failure in that half. None
    held.
- **Scope of the operational tests.** The two tests above were fixed in code before selection and passed the review. The
  fact-check found them narrower than the task sheet's wording in two places. Both patterns are reported here
  descriptively. They do not change the applied readings or the conclusion.
  - **Transcript distance of O − F.** O − F, a mandatory comparison, has a lower multi edit distance with B3 in all six
    seed–half cells. On selection it is 0.0586 against 0.0623, 0.0389 against 0.0420 and 0.0426 against 0.0876. On
    calibration it is 0.0308 against 0.0325, 0.0293 against 0.0304 and 0.0267 against 0.0419. The differences are small
    in seeds 0 and 1, the seed-2 difference coincides with F's seed-2 setting, and no interval was computed. Read under
    the sheet's wording, this is a transcript-only or local effect, not a system result.
  - **Native-floor failures beside recoveries.** The trade-off test counted a single-seed recall-floor failure but not a
    single-seed native-floor failure. On selection, B and O recover multi segments against F (5 and 8 over the seeds,
    while losing 8 and 9) in the same half where each fails the native floor in two seeds through one A4C clip. Read
    under the sheet's wording, this matches its pattern of segment changes beside a native classification loss below a
    floor, though not consistently across seeds.
- **What "end" means.** This fixed feature, loss and budget configuration closes. No larger encoder, nonlinear head,
  boundary head, teacher, gate, duration prior, transcript decoder, loss-weight, epoch or learning-rate search, extra
  seed or new extraction follows.

### 9. Cost

- **GPU.** 0.392 of 4 hours: checks 0.012 h, the nine trainings 0.345 h, and calibration, dry run and selection
  inference 0.035 h.
- **Extra CPU.** 0.323 of 3 hours. This covers the tests, preparations, review, calibration grids, dry run, selection
  bootstrap, queue overhead and the fact-check's 0.04 hours.
- The ledger is `runs/segment_objective_v1/ledger.jsonl` on the server. It has 27 lines. They include the reviewer's and
  the fact-checker's hand-recorded command time and, last, a final id scan of the result files.

## Terms

- **Native clip.** One whole public recording with one view label.
- **Pair stream and multi stream.** Constructed sequences that splice fragments of two, or of several, recordings.
- **Run.** A maximal stretch of one class inside a sequence. Adjacent fragments of the same view form one run.
- **Transcript.** The sequence's run classes in order. An X-Y-X pattern keeps three entries.
- **B3.** The simple change-point decoder (half-window, threshold, NMS radius 5). It cuts where the probability change
  between two windows peaks and names each segment by its mean probabilities. Each head's setting was chosen once on
  calibration from the 40-setting grid. A setting is feasible with native macro-F1 at least 0.980 and multi boundary
  recall at least 0.95; the highest segment F1@50 wins.
- **Segment F1@50.** Predicted and reference segments are matched one to one by class, with temporal IoU at least 0.5,
  using the evaluator's greedy order. F1 is 2 × matched / (predicted + reference), pooled over multi streams.
- **Recovered and lost.** Seen from system a against system b, a reference segment is recovered when only a matches it,
  and lost when only b matches it.
- **FP and FN segments.** Unmatched predicted segments, and unmatched reference segments.
- **Boundary recall.** The share of reference view changes with a predicted change within 0.5 s, matched one to one.
- **Native macro-F1.** The class-averaged F1 of native clips, each named by its majority decoded label.
- **Normalised transcript edit distance.** The Levenshtein distance between the predicted and the reference transcript,
  divided by the reference transcript's length.
- **Bootstrap.** 2,000 draws with seed 20261002, resampling native clips by session and streams by whole stream. The
  same draws serve every comparison.

## Main system rows (B3)

Each cell gives seeds 0, 1 and 2.

| System | Half | Segment F1@50 | Native macro-F1 | Boundary recall | Boundary precision |
|---|---|---|---|---|---|
| F + B3 | selection | 0.9473, 0.9569, 0.9388 | 0.9786, 0.9786, 0.9786 | 0.9555, 0.9529, 0.9712 | 0.9530, 0.9707, 0.9115 |
| B + B3 | selection | 0.9481, 0.9550, 0.9620 | 0.9786, 0.9701, 0.9700 | 0.9529, 0.9529, 0.9607 | 0.9579, 0.9655, 0.9709 |
| O + B3 | selection | 0.9512, 0.9569, 0.9610 | 0.9701, 0.9701, 0.9786 | 0.9555, 0.9529, 0.9607 | 0.9530, 0.9707, 0.9735 |
| Frozen Ordered + B3 | selection | 0.9385, 0.9603, 0.9644 | 0.9786, 0.9786, 0.9701 | 0.9581, 0.9686, 0.9738 | 0.9482, 0.9585, 0.9588 |
| F + B3 | calibration | 0.9768, 0.9779, 0.9732 | 0.9861, 0.9861, 0.9861 | 0.9749, 0.9799, 0.9724 | 0.9749, 0.9726, 0.9532 |
| B + B3 | calibration | 0.9768, 0.9789, 0.9806 | 0.9861, 0.9861, 0.9929 | 0.9774, 0.9849, 0.9698 | 0.9774, 0.9703, 0.9797 |
| O + B3 | calibration | 0.9778, 0.9788, 0.9816 | 0.9861, 0.9861, 0.9861 | 0.9799, 0.9774, 0.9724 | 0.9774, 0.9774, 0.9797 |
| Frozen Ordered + B3 | calibration | 0.9817, 0.9836, 0.9797 | 0.9929, 0.9861, 0.9861 | 0.9824, 0.9874, 0.9799 | 0.9751, 0.9800, 0.9774 |

A01 Linear + B3 equals F + B3 in every cell. The full rows (pair streams, switches, same-view join cuts, per-class
precision and recall, argmax and stored-setting rows, A01 MS-TCN and ASRF) are in `report_tables.md`.

**Multi-stream TP / FP / FN with B3** (reference segments: 502 on selection, 518 on calibration).

| System | Selection, seeds 0 / 1 / 2 | Calibration, seeds 0 / 1 / 2 |
|---|---|---|
| F + B3 | 476/27/26, 477/18/25, 483/44/19 | 506/12/12, 508/13/10, 508/18/10 |
| B + B3 | 475/25/27, 477/20/25, 481/17/21 | 506/12/12, 510/14/8, 506/8/12 |
| O + B3 | 478/25/24, 477/18/25, 480/17/22 | 507/12/11, 507/11/11, 507/8/11 |
| Frozen Ordered + B3 | 473/33/29, 484/22/18, 487/21/15 | 510/11/8, 511/10/7, 508/11/10 |

## Paired comparisons of segment F1@50 (B3)

Each cell gives the difference and its 95% bootstrap interval for seeds 0, 1 and 2. All verdicts are "no consistent
difference detected". The same holds for the other five bootstrapped metrics and for the argmax and stored-setting
rows. All 78 decoding, comparison and metric combinations agree, in both halves.

| Comparison | Selection | Calibration |
|---|---|---|
| O − B (primary) | +0.0031 [−0.0019, +0.0105]; +0.0019 [−0.0021, +0.0061]; −0.0010 [−0.0053, +0.0030] | +0.0010 [+0.0000, +0.0031]; −0.0001 [−0.0039, +0.0037]; +0.0010 [+0.0000, +0.0030] |
| B − F | +0.0008 [−0.0031, +0.0052]; −0.0019 [−0.0093, +0.0048]; +0.0232 [+0.0090, +0.0413] | +0.0000 [+0.0000, +0.0000]; +0.0010 [−0.0019, +0.0044]; +0.0074 [−0.0001, +0.0161] |
| O − F | +0.0040 [−0.0021, +0.0121]; +0.0000 [−0.0070, +0.0060]; +0.0222 [+0.0086, +0.0390] | +0.0010 [+0.0000, +0.0031]; +0.0009 [−0.0021, +0.0039]; +0.0084 [+0.0013, +0.0168] |
| O − frozen Ordered | +0.0128 [−0.0023, +0.0292]; −0.0034 [−0.0126, +0.0052]; −0.0034 [−0.0133, +0.0059] | −0.0039 [−0.0109, +0.0020]; −0.0049 [−0.0109, +0.0000]; +0.0019 [−0.0019, +0.0058] |
| B − frozen Ordered | +0.0096 [−0.0051, +0.0252]; −0.0054 [−0.0159, +0.0034]; −0.0024 [−0.0118, +0.0060] | −0.0049 [−0.0121, +0.0011]; −0.0048 [−0.0102, −0.0009]; +0.0009 [−0.0031, +0.0052] |
| F − A01 Linear | 0 in every seed | 0 in every seed |
| F − frozen Ordered | +0.0088 [−0.0058, +0.0231]; −0.0034 [−0.0128, +0.0052]; −0.0256 [−0.0446, −0.0093] | −0.0049 [−0.0121, +0.0011]; −0.0058 [−0.0118, −0.0001]; −0.0066 [−0.0144, +0.0000] |

## Training

All nine runs completed 18,750 updates without interruption. They took 0.019 to 0.022 GPU hours for F and B and 0.073
to 0.074 hours for O, whose transcript term adds a per-sequence forward recursion.

The monitors below are training-data fit only. They selected nothing.
- **Last-epoch batch losses.** F's mean training loss was 0.157, 0.153 and 0.155. B's was 0.197, 0.198 and 0.197. O's
  total was 0.158, 0.159 and 0.157, with a transcript term of 0.077, 0.076 and 0.075.
- **The 16 fixed training check sequences** (8 native crops and 8 bank streams).
  - The transcript term L_order fell only slightly in every arm, from 0.218 to 0.232 after the first epoch to 0.203 to
    0.221 after the last.
  - Argmax accuracy on the 8 check streams stayed near 0.946.
  - Argmax accuracy on the 8 native check crops ended at 0.79 to 0.84 in every arm. A01's linear run shows the same
    pattern, and its smallest check-sequence accuracy was 0.0. F's overall check fit, 0.8715 in seed 0, equals A01's.

`training_curves.png` shows these monitors over training.

## What this round does not show

- **No cause is shown.** The round does not show that frame weighting caused the remaining segment errors, or that a
  sequence-level loss cannot help with another representation, head or decoder. It tested one fixed recipe.
- **Seed 2 is not explained.** At each head's own B3 setting, its B − F and O − F gaps coincide with F's seed-2 setting.
  At the common stored setting they are near zero, while direct argmax still favours B and O on selection. They are
  attributed neither to segment balancing nor to the decoder.
- **Small counts.** The recovered, lost and native counts rest on fewer than 10 sessions per seed and half. They are
  descriptive.
- **No generalisation claim.** Both development halves have been reused many times, constructed streams are not real
  scans, and grouping is by session, not by patient.
- **No novelty claim.** Segment balancing and the forward-sum transcript loss are established ideas. Nothing here is
  claimed as a new architecture.

## Deviations and launch record

- **Configuration wording.** The configuration changed once, in wording only, after the review (c7e8e70). The numerics
  sentence now states that the plan's class weights are float32-exact. The sha256 moved from b53bf634 to 3f698094. No
  model, data, metric, threshold or budget changed, and the run identities were made afterwards.
- **Tests after the failing commit.** c7e8e70 added three tests (label access, selection openings, and one ledger line
  per killed run). It also extended two existing tests: the encoder scan now covers the training files, and the readings
  test has the order-claim case.
- **The dry run.** The task sheet does not name one, but the configuration fixed it before any code (6088a5b,
  `evaluation.dry_run`), so it is not a deviation. It ran the complete selection pipeline with calibration standing in
  for both halves, before the choices were committed. It read no selection data, and its outputs stayed on the server.
- **Launch.** The first status check, at 02:06:44, saw the queue and the first run process alive. The run held 646 MiB of
  GPU memory and had logged 13,125 of 18,750 updates. These figures come from the live process list and log at that
  time and were not kept as a separate record. The run's own timing (35 epochs in 47 s of training time, 50 s after its
  start) is consistent with them.
- **Booking.** About 4.2 to 4.4 seconds per run (imports and data loading before the run's own ledger line starts),
  about 38 seconds over the nine runs, are booked as queue overhead, not GPU.

## Fact-check

An independent fact-checker, a separate agent, recomputed the round from the saved outputs with its own code between
02:51 and 03:14 UTC (`fact_check.md`). Every value it recomputed matched: 0 mismatches in about 17,000.

**What it covered.**
- the system rows of all 13 systems;
- the 1,872 bootstrap values and the 156 verdicts;
- the goal records and the recovered and lost accounting;
- the B3 thresholds and choices;
- the run identities and the git-object proof;
- the loss on 3,249 toy cases against path enumeration, and one formal batch against an independent forward-backward;
- an epoch-0 replay of each arm (within 4.5e-9 of the recorded epoch-mean loss);
- label isolation.

It agreed with the applied readings. It found one medium factual error and one medium wording problem in the README
(FP segments against frozen Ordered; the seed-2 wording). It also found two places where the readings' operational
tests are narrower than the task sheet's wording, and four low or nit-level errors. All were corrected or disclosed
before publication, and `fact_check.md` lists each one.

## Commands

```
python scripts/train_segment_objective.py --config configs/experiments/segment_objective_v1.yaml --prepare
ECHO_GPU_MEM_FRACTION=0.6 python scripts/train_segment_objective.py --config configs/experiments/segment_objective_v1.yaml --check
ECHO_GPU_MEM_FRACTION=0.6 python scripts/train_segment_objective.py --config configs/experiments/segment_objective_v1.yaml --queue
ECHO_GPU_MEM_FRACTION=0.6 python scripts/report_segment_objective.py --config configs/experiments/segment_objective_v1.yaml --calibrate
ECHO_GPU_MEM_FRACTION=0.6 python scripts/report_segment_objective.py --config configs/experiments/segment_objective_v1.yaml --dry-run
python scripts/report_segment_objective.py --config configs/experiments/segment_objective_v1.yaml --proof --choices-commit eaff3e1096646b2ee341043f9f8e78bfb9916e5b   # local
ECHO_GPU_MEM_FRACTION=0.6 python scripts/report_segment_objective.py --config configs/experiments/segment_objective_v1.yaml --selection \
    --choices-sha256 0b5034da72a43b18da7b1d521561ba9d495e717fcec9ea3fec279972186e28d2 --choices-commit eaff3e1096646b2ee341043f9f8e78bfb9916e5b
```

The server ran Python 3.12.3, NumPy 2.5.2, pandas 3.0.5, SciPy 1.18.1, torch 2.14.0+cu130, torchvision 0.29.0+cu130,
CUDA 13.0 and cuDNN 92400, from `/home/william/echo-view-routing/.venv`.

## Provenance

- **Source pins.** These match the task sheet:
  - A01 configuration 3b065e7f;
  - A01 preparation record 5f7d48db (the server copy is identical);
  - A01 summary 3a160151;
  - A04 README 52a81cef and summary 87a60003;
  - A03 configuration c2e90d60, for the metadata helpers' protocol, clip index and window plans.
- **A01 records.** The plans (s0 767401a2, s1 746180b8, s2 723d5b0f), linear initial states (s0 711730a2, s1 0860cbba,
  s2 0a51430a), training feature identities, sequence file and frozen heads matched A01's records. The initial states
  equal the V01 source classifiers tensor for tensor.
- **Caches.** The training and development cache sha256 values are in `preparation.json`. The development caches were
  only hashed before the freeze.
- **Code and configuration.** The configuration sha256 is 3f698094 (c7e8e70). The training code digest is c77bce94, and
  the report code digest is 26120990. Both are recorded in `choices.json` and `summary.json`.
- **Runs.** Every run identity, final weight sha256 and frozen path is in `summary.json`. The frozen heads are
  `models_frozen/segment_objective_v1/{frame,segment,order}_s{0,1,2}`.
- **Choices.** The choice record's sha256 is 0b5034da at commit eaff3e1. The proof covers the commit, the four trees on
  the path and blob 66546ed1, and the selection stage verified it on the server, which has no git. The selection half was
  opened once.
- **Parity.** The source classifier on the cached window features matched V01's saved logits within 8.11e-6 at most,
  with equal argmax, on 7,666 selection and 7,021 calibration windows per seed.
- **Reproduction.** A01's 18 stored rows per half were reproduced exactly: frozen Ordered + B3 and argmax, A01 Linear +
  B3 and argmax, MS-TCN and ASRF, for three seeds.

## Files

- `README.md`. This report.
- `summary.json`. Every system row, comparison, verdict, accounting entry, goal, reading input, replication value,
  parity, identity and cost.
- `report_tables.md`. The generated tables.
- `choices.json`. The calibration choices with each head's 40-setting grid and the provenance. Committed before
  selection.
- `preparation.json` and `checks.json`. The preparation and check records made by the reviewed code.
- `code_review.md`. The implementation review and its verification.
- `fact_check.md`. The independent fact-check.
- `training_curves.png`. Training-data monitors.

Per-sequence probabilities, decodings and case lists stay on the server under `runs/segment_objective_v1/selection/`.
