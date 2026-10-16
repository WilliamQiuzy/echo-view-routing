# End-to-end sequence objectives on the Ordered video model (2026-10-15/16)

This round changed only the training objective of a short continuation of the V01 Ordered video model. Unlike A05, the
gradients reached the whole video encoder. It asked whether a run-balanced loss, or that loss plus a term over each
group's class order, gives better complete segmentation than plain cross-entropy when the encoder trains on groups of
four neighbouring windows.

**What was trained.**
- The model is V01's Ordered TargetMViT. It is torchvision's MViTv2-S with its 512-d projection, V01's target-marker
  stem and a five-class head, **34,640,549 parameters** in all. No parameter was added.
- Each of three seeds continued its own frozen V01 Ordered final weights three times, once per arm, so there are nine
  runs. Each run made 3,750 updates of 8 windows and kept its final weights. There was no validation, epoch choice or
  training-score gate.
- V01's optimiser and schedule were reused unchanged, with a fresh optimiser for each run. AdamW used a learning rate of
  1e-4 for the marker stem and the classifier and 1e-5 for the encoder, including the projection. Weight decay was
  1e-4, the gradient norm was clipped at 1, and both rates followed V01's warm-up and cosine factor.
- For the first 375 updates the encoder was frozen, and gradients flowed through it to the marker stem. From the 376th
  update on, every parameter trained.
- The three runs of a seed read the same manifest, so they saw the same windows, targets, markers, pixels and batch
  order. They also started from the same model random state.
- V01 itself trained each model for 3,750 updates of 4 native and 4 mixed-view windows with plain cross-entropy. F keeps
  that loss but continues from V01's final weights on grouped windows. F − frozen Ordered therefore shows this sampling
  and continuation condition, not an objective.

**The groups.** Each update holds two groups of four windows. A window is V01's input: 16 frames centred on a grid
centre, with the marker on the centre's slot. Grid centres lie every 5 samples along the whole sequence, as in V01.
- **The native group** takes four consecutive grid centres of one training clip. All four targets share the clip's
  public label. A clip with fewer than four centres repeats its last one.
- **The constructed group** takes four consecutive grid centres of one constructed training stream from A01's bank. A
  start is eligible only when the four sampled labels contain at least one change.
- **The draw.** `default_rng([20261015, seed, update])` drew the native clip, its start, the eligible stream and its
  start, in that order. No model score or development result entered the draw.
- **Counts.** The 3,364 training clips give 33,049 native starts, and 230 clips needed a repeated centre. 1,478 of the
  1,500 bank streams have an eligible start, with 14,153 eligible starts in all. In each seed's 3,750 constructed
  groups, 3,631 to 3,640 sampled orders have two runs and 110 to 119 have three runs. Of the three-run orders, 17, 26 and
  20 are X-Y-X with two classes, so 102, 84 and 91 hold three classes. Every reference run of the training streams is at
  least 10 samples long, so no run fell between two sampled centres.

**The three arms.** All three use A05's pinned loss module, unchanged, on the update's 2 × 4 logits with equal class
weights.
- **F (plain cross-entropy).** The mean cross-entropy of the eight windows. This is V01's own loss.
- **B (run-balanced cross-entropy).** Each group's sampled targets form runs of equal labels. Every run contributes its
  mean cross-entropy, and the runs of both groups are averaged. A native group is one run, so it weighs as much as one
  run of the constructed group.
- **O (B plus a transcript term).** (B + 0.5 × L_order) / 1.5. L_order is the negative log-probability, per window, of
  the group's run-collapsed class order. That probability sums over every way of placing the classes in that order over
  the four windows with positive durations, with no blank class.
- On a batch whose two groups each hold one class, the three losses are equal in value and gradient. Every constructed
  group holds a change, so the arms differ in every update.

**Training versus inference.** Labels, runs and orders entered the training losses only. At inference every model gives
V01's ordinary single-window CLS probabilities at the same stride-5 centres, mapped to samples by the nearest centre. No
group, label, order, join or id reaches a model. The probabilities are decoded with the simple change-point decoder B3,
whose setting was chosen once per model on calibration.

The task sheet is `docs/superpowers/plans/2026-10-15-end-to-end-sequence-objective.md` (5110b12). The fixed configuration
is `configs/experiments/video_sequence_objective_v1.yaml` (1a76407).

**What kind of evidence this is.**
- **Reused data.** The two development halves (selection and calibration) have been used and inspected in many earlier
  rounds. This is a matched comparison on reused development data, not an independent validation.
- **Grouping.** The hold-out is by session, not by patient. Selection holds 284 native clips from 36 sessions, 137 pair
  streams and 120 multi streams. Calibration holds 237 native clips from 36 sessions, 115 pair streams and 120 multi
  streams. Constructed streams are spliced from separate recordings and are not real scans. They reuse recordings, so
  their units are not independent cases. The three seeds share the same cases.
- **Labels.** Native clips carry the public recording label. No label was changed.
- **Untouched data.** The test split and EchoNet-Pediatric were not read.
- **Novelty.** Run balancing and sequence-probability training are established ideas. This round is a bounded test of
  one recipe, not a new architecture.

**Order of work.** All times are UTC on 2026-10-16.
- **Before any code.** 5110b12 committed the task sheet at 02:21. 1a76407 committed the configuration at 02:32.
- **Tests and code.** 3b92fe8 committed the tests at 02:40, and they failed at import. 095320c committed the code at
  02:53. It also corrected one test assertion to the configuration's readings (no qualifying gain does not apply
  together with balancing only) and removed two unused imports. Its commit message records the fix.
- **Preparation and check.** The preparation ran from 02:54:59 to 02:58:14 and the GPU check from 02:59:06 to 03:04:20.
  Both passed.
- **Review.** An independent implementation review passed the code with no blocking or should-fix finding
  (`code_review.md`). f39c4b3 committed the records and the test the reviewer suggested at 03:18.
- **Training.** The queue was launched at 03:19:12, the time of the launch command. The nine runs ran one after another
  from 03:19:15 to 07:35:19 and were frozen at 07:35:25.
- **Calibration.** It ran from 07:37 to 07:59. The dry run that followed stopped on a report-code error, which was fixed
  in two commits (dcae2cd and 5ec611c; see "Deviations"). Calibration ran again from 08:11 to 08:15 with the fixed code,
  and the dry run passed from 08:16 to 08:20. The dry run used the calibration half for both halves.
- **Barrier.** a96dfc50 committed the calibration choices with their provenance at 08:21:26. A git-object proof of that
  commit was made at 08:21:40.
- **Selection.** The selection half was opened once, at 08:22:11, and evaluated by 08:45:23.
- **Fact-check.** An independent fact-check then recomputed the results (see "Fact-check").

## Answers

Values are given for seeds 0, 1 and 2 in that order, on the selection half and then the calibration half. The main
metric is multi-stream segment F1@50 with each model's B3 setting. A difference is consistent only when all three
seeds' 95% intervals exclude zero strictly on the same side in a half. Otherwise it is "no consistent difference
detected", which is not the same as equality. None of the 216 verdicts (three decoders, six comparisons and six
bootstrapped metrics, in two halves) is consistent. Two further bootstrapped metrics, multi boundary F1 and the
same-view cut rate, carry no verdict, and none of their 72 interval sets is consistent either.

### 1. No consistent difference was detected for the primary comparison O − B

- **Segment F1@50.** O − B is +0.0046, +0.0001 and +0.0010 on selection, and +0.0009, +0.0000 and −0.0019 on
  calibration. Only seed 0's selection interval [+0.0010, +0.0090] excludes zero. Seed 1's calibration outputs give a
  difference of exactly zero, and seed 2's calibration interval ends at zero.
- **Other metrics.** No consistent difference was detected in native macro-F1, multi boundary recall, multi
  false-positive (FP) or false-negative (FN) segment counts, or native switches per clip.
- **Reference segments.** Over the three seeds, on multi streams, O recovered 2 reference segments that B missed and lost
  none on selection. On calibration it recovered none and lost 2. Each per-seed count rests on at most 1 session, fewer
  than 10.
- **Conclusion.** Under this recipe, no consistent multi segment F1@50 difference between O and B was detected.

### 2. No consistent difference was detected for run balancing (B − F) or O − F

- **B − F.** It is −0.0037, +0.0020 and +0.0026 on selection, and −0.0009, +0.0000 and +0.0010 on calibration. Only
  seed 0's selection interval [−0.0074, −0.0009] excludes zero, below it.
- **O − F.** It is +0.0009, +0.0021 and +0.0036 on selection, and +0.0000, +0.0000 and −0.0009 on calibration. Seed 0's
  selection interval [+0.0000, +0.0029] starts at zero, so it does not exclude zero strictly. No interval excludes zero.
- **Reference segments (multi streams, summed over seeds).** B recovered 3 and lost 2 against F on selection, and
  recovered 2 and lost none on calibration. O recovered 4 and lost 1 against F on selection, and recovered 1 and lost 1
  on calibration. Each per-seed count rests on at most 2 sessions.
- **Native clips.** B and F get the same native clips right and wrong in both halves. O reads one more A4C clip as A5C than B
  and F in seed 0 on calibration (native macro-F1 0.9861 against 0.9929). This rests on one clip and one session.

### 3. No consistent difference was detected against frozen Ordered + B3; the continued models differ by seed and cut more often

- **Segment F1@50.**
  - O − frozen Ordered is +0.0156, −0.0075 and −0.0433 on selection, and −0.0038, +0.0049 and −0.0036 on calibration.
  - B − frozen Ordered is +0.0109, −0.0076 and −0.0444, then −0.0047, +0.0049 and −0.0017.
  - F − frozen Ordered is +0.0146, −0.0095 and −0.0470, then −0.0038, +0.0049 and −0.0027.
  - On selection, seed 0's intervals lie above zero for O and F, and seed 2's lie below zero for all three arms. Every
    calibration interval includes zero. The signs differ between seeds, so no difference is consistent.
- **Means.** The three-seed means are 0.9404 / 0.9812 for F, 0.9407 / 0.9812 for B and 0.9426 / 0.9809 for O, against
  0.9544 / 0.9817 for frozen Ordered + B3. All three are below the reference in both halves.
- **Reference segments (multi streams).**
  - On selection every arm recovered 15, 4 and 7 reference segments per seed that frozen Ordered missed, from 8, 4 and 5
    sessions. Most are A4C or A5C segments (14, 3 and 6). O lost 0, 3 and 10, B lost 0, 4 and 11, and F lost 0, 5 and
    11, all A4C or A5C segments.
  - On calibration O recovered 3, 5 and 6 and lost 4, 0 and 2. B recovered 3, 5 and 6 and lost 3, 0 and 1. F recovered
    3, 5 and 5 and lost 4, 0 and 1.
  - Of the 26 segments recovered on selection, 14 have at most 16 samples. Every arm recovered 2 A4C/A5C X-Y-X middles
    on selection and lost none. On calibration no arm recovered or lost such a middle.
  - Each seed's multi reference holds 502 segments on selection, 141 of them with at most 16 samples and 7 A4C/A5C
    X-Y-X middles. On calibration it holds 518, with 134 and 7.
  - Every per-seed count rests on fewer than 10 sessions (0 to 8).
- **Extra cuts.** The continued models predict more change points.
  - Their multi boundary recall is higher in every seed and half. For O it is +0.0236, +0.0105 and +0.0131 on selection
    and +0.0050, +0.0075 and +0.0151 on calibration.
  - Their boundary precision is lower on selection: 0.9352, 0.9444 and 0.8788 for O against 0.9482, 0.9585 and 0.9588.
  - O adds 18, 18 and 55 unmatched predicted segments on selection and removes 18, 9 and 11. On calibration it adds 8,
    3 and 12 and removes 5, 3 and 4. O − frozen Ordered in multi FP segments is +0, +9 and +44 on selection and +3, +0
    and +8 on calibration. Seed 0's selection interval and seed 0's and seed 1's calibration intervals include zero, so
    this is not consistent.
  - The same holds for B and F within a few segments.
  - **Sessions.** On selection these added and removed segments come from 10 to 17 sessions per seed. On calibration
    O's added segments come from 10, 2 and 12 sessions and its removed ones from 7, 5 and 8. B's and F's calibration
    counts come from 2 to 11 sessions. Most calibration counts therefore rest on fewer than 10 sessions.

### 4. Seed 2 on selection

- **The decoder settings.** On calibration, seed 2's three models chose half-window 5 with threshold quantile 0.995
  (thresholds 0.122, 0.134 and 0.129 for F, B and O; 34, 29 and 34 of the 40 settings were feasible). Frozen Ordered's
  stored seed-2 setting is half-window 9 with threshold 0.308. Seeds 0 and 1 chose quantile 0.999 at half-window 7 to 11.
- **What happens on selection.** Seed 2's continued models produce 65 to 68 multi FP segments against frozen Ordered's 21.
  Their boundary precision is 0.875 to 0.881 against 0.959. Their multi transcript edit distance is 0.116 to 0.123
  against 0.042.
- **Other decoders.** The selection gap remains at the common stored setting (F, B and O − frozen Ordered: −0.0236,
  −0.0246 and −0.0208) and with direct argmax (−0.0224, −0.0207 and −0.0215). All six intervals lie below zero.
- **Calibration.** On calibration, where the settings were chosen, seed 2 shows no segment F1@50 gap. Its B3 differences
  are −0.0027, −0.0017 and −0.0036 for F, B and O, and every interval includes zero. Its continued models still cut
  more there with B3 (multi FP segments 18, 18 and 19 against 11; native switches 15, 14 and 16 against 3). At the stored
  setting they cut slightly more (15 to 16 FP segments against 11, with intervals starting at zero), and with argmax they
  cut less (24 against 30).
- **Reading.** Seed 2's continued models cut more often on selection under all three decoders, and with B3 also on
  calibration, where it cost no segment F1@50. No cause is established.

### 5. Native recognition

- **The arms.** F, B and O have native macro-F1 0.9786 on selection in every seed, and 0.9861 to 0.9929 on calibration.
  No arm fails a floor in any seed or half. No native macro-F1 difference is consistent.
- **Against frozen Ordered.**
  - In seed 2 on selection, frozen Ordered reads one A4C clip as A5C that all three continued models read correctly.
    Frozen Ordered's only floor failure (0.9701, below the 0.973 selection floor) is in this cell, where the continued
    models have 0.9786.
  - In seed 1 on calibration, all three arms read correctly one A4C clip that frozen Ordered reads as A5C.
  - In seed 0 on calibration, O reads one more A4C clip as A5C than frozen Ordered, B and F.
  - Each of these rests on one clip and one session.
- **Error types.** On selection, A5C precision is 0.846 for every arm in every seed (2 A4C clips read as A5C), against
  0.846, 0.846 and 0.786 for frozen Ordered. No B3 or argmax row reads an A5C clip as A4C in either half. Only the stored-
  setting rows of seed 2 do, one clip each on calibration. With B3 the only other native error, one PLAX clip read as
  PSAX on selection, is shared by every system. Per-class precision and recall are in `summary.json`.

### 6. Transcripts, switches and same-view joins

- **Transcript distance.** The normalised multi transcript edit distance with B3 is as follows.
  - O: 0.0627, 0.0484 and 0.1204 on selection, and 0.0248, 0.0159 and 0.0294 on calibration.
  - B: 0.0731, 0.0484 and 0.1158, then 0.0290, 0.0159 and 0.0273.
  - F: 0.0648, 0.0537 and 0.1234, then 0.0297, 0.0159 and 0.0301.
  - Frozen Ordered: 0.0645, 0.0438 and 0.0424, then 0.0279, 0.0206 and 0.0295.
  - O is lower than B in 2 of the 6 seed–half cells, equal in 2 and higher in 2. O is lower than F in 5 cells and
    equal in 1. No interval was computed for these distances, and a lower distance alone is not a segment win.
- **Native switches.** Counting label switches inside native clips with B3, seed 2's continued models have 25, 22 and 24
  (F, B and O) on selection against 13 for frozen Ordered, and 15, 14 and 16 on calibration against 3. In seeds 0 and 1
  the counts are close to frozen Ordered's (10 to 16 against 12 and 11 on selection, 4 or 5 against 6 and 4 on
  calibration). These counts come from 2 to 11 clips and 2 to 8 sessions per seed and half, fewer than 10 sessions.
- **Same-view joins.** Some multi streams join two fragments of the same view. On selection the continued models cut 5,
  3 and 7 of these 230 joins, against 4, 1 and 1 for frozen Ordered. On calibration they cut 4 to 7, 3 and 7 of 201,
  against 5, 5 and 5. The fact-check counted the sessions behind these cuts. They lie in 2 to 10 sessions per system,
  seed and half, fewer than 10 in 20 of the 24 cells, so they are descriptive.

### 7. Sensitivities

These rows show the same models with other decoders. Nothing is chosen from them.
- **Direct argmax.**
  - O − B is +0.0000, −0.0027 and −0.0008 on selection, and +0.0000, +0.0019 and +0.0000 on calibration.
  - B − F is +0.0022, +0.0044 and +0.0017, then +0.0000 in every seed.
  - O − F is +0.0022, +0.0017 and +0.0008, then +0.0000, +0.0019 and +0.0000.
  - O − frozen Ordered argmax is +0.0226, −0.0228 and −0.0215, then −0.0010, +0.0160 and +0.0104.
- **The seed's stored frozen Ordered setting.** This is the same numeric B3 setting for every model of a seed.
  - O − B is −0.0009, −0.0001 and +0.0038, then +0.0000, +0.0000 and −0.0019.
  - B − F is −0.0009, −0.0028 and −0.0009, then −0.0047, +0.0010 and +0.0000.
  - O − F is −0.0018, −0.0029 and +0.0029, then −0.0047, +0.0010 and −0.0019.
  - O − frozen Ordered + B3 is +0.0100, −0.0150 and −0.0208, then −0.0103, +0.0049 and −0.0018.
- None of these is consistent. Several intervals start or end exactly at zero.

### 8. Goal and the fixed readings

- **The goal.** It requires three-seed means of at least 0.944409687184662 (selection) and 0.9821421709894332
  (calibration), and the floors in every seed and half.
  - O misses both means, by 0.0018 on selection and 0.0013 on calibration. Its margin to the calibration sensitivity
    target 0.9838935784747763 is −0.0030. O meets every floor.
  - F (0.9404 / 0.9812) and B (0.9407 / 0.9812) miss both means. Frozen Ordered + B3 (0.9544 / 0.9817) passes the
    selection mean, misses the calibration mean and fails the native floor once (seed 2, selection).
  - The goal is not met.
- **Applied readings: no qualifying gain, and end.**
  - **Not applied: full success and objective signal.** O − B, O − F and O − frozen Ordered are not consistent positive
    in any half.
  - **Not applied: balancing only and continuation only.** Neither B − F nor F − frozen Ordered is consistent positive
    in any half.
  - **Not applied: trade-off.** Its fixed test needs a comparison and half where the candidate recovers multi segments
    together with consistently more multi FP segments, a consistently lower native macro-F1, or a floor failure of the
    candidate. No cell meets it. The extra FP segments against frozen Ordered (section 3) are not consistent, and F, B
    and O fail no floor.
  - **Not applied: consistent losses.** No primary, mandatory or F − frozen Ordered comparison is consistent negative in
    a half. Seed 2's selection losses against frozen Ordered (sections 3 and 4) are single-seed results and are
    reported here descriptively.
  - The primary direction per half is "no consistent difference detected" on selection and on calibration.
- **What "end" means.** This fixed configuration closes after this report. No extra seed, loss, coefficient, sampling,
  epoch, decoder or readout search, teacher, gate or automatic follow-on follows.

### 9. Cost

- **GPU.** 5.000 of 8 hours before the fact-check: the check 0.087 h, the nine trainings 4.266 h, and calibration, dry
  runs and selection inference 0.647 h. The latter includes 0.012 h for the failed dry run and the failed calibration
  rerun.
- **CPU.** 0.698 of 4 hours before the fact-check. This covers the tests, the preparation, the review (0.297 h), the
  probe, the queue overhead, the calibration grids, the dry runs and the selection bootstrap.
- **Fact-check.** 0.018 GPU hours for one 66-second re-inference job, and 0.917 CPU hours. The CPU part is the
  fact-checker's own wall clock (0.408 hours besides its GPU job, and 0.016 hours for one follow-up answer) and its two
  read-only helper agents (0.209 and 0.284 hours), each booked separately. Probes that checked its findings add 0.004
  hours.
- **In all.** 5.018 of 8 GPU hours and 1.619 of 4 CPU hours, before a last id scan of the result files.
- The ledger is `runs/video_sequence_objective_v1/ledger.jsonl` on the server.

## Terms

- **Native clip.** One whole public recording with one view label.
- **Pair stream and multi stream.** Constructed sequences that splice fragments of two, or of several, recordings.
- **Run.** A maximal stretch of one class inside a sequence. Adjacent fragments of the same view form one run.
- **Transcript.** The sequence's run classes in order. An X-Y-X pattern keeps three entries.
- **B3.** The simple change-point decoder (half-window, threshold, NMS radius 5). It cuts where the probability change
  between two windows peaks and names each segment by its mean probabilities. Each model's setting was chosen once on
  calibration from the 40-setting grid. A setting is feasible with native macro-F1 at least 0.980 and multi boundary
  recall at least 0.95; the highest segment F1@50 wins, with ties broken by grid order.
- **Segment F1@50.** Predicted and reference segments are matched one to one by class, with temporal IoU at least 0.5,
  using the evaluator's greedy order. F1 is 2 × matched / (predicted + reference), pooled over multi streams.
- **Recovered and lost.** Seen from system a against system b, a reference segment is recovered when only a matches it,
  and lost when only b matches it.
- **FP and FN segments.** Unmatched predicted segments, and unmatched reference segments.
- **Boundary recall and precision.** The shares of reference view changes, and of predicted changes, matched one to one
  within 0.5 s.
- **Native macro-F1.** The class-averaged F1 of native clips, each named by its majority decoded label.
- **Normalised transcript edit distance.** The Levenshtein distance between the predicted and the reference transcript,
  divided by the reference transcript's length.
- **Bootstrap.** 2,000 draws with seed 20261002, resampling native clips by session and streams by whole stream. The
  same draws serve every comparison.

## Main system rows (B3)

Each cell gives seeds 0, 1 and 2.

| System | Half | Segment F1@50 | Native macro-F1 | Boundary recall | Boundary precision |
|---|---|---|---|---|---|
| F + B3 | selection | 0.9531, 0.9508, 0.9174 | 0.9786, 0.9786, 0.9786 | 0.9817, 0.9738, 0.9869 | 0.9328, 0.9442, 0.8747 |
| B + B3 | selection | 0.9494, 0.9528, 0.9200 | 0.9786, 0.9786, 0.9786 | 0.9817, 0.9764, 0.9869 | 0.9236, 0.9467, 0.8808 |
| O + B3 | selection | 0.9541, 0.9528, 0.9210 | 0.9786, 0.9786, 0.9786 | 0.9817, 0.9791, 0.9869 | 0.9352, 0.9444, 0.8788 |
| Frozen Ordered + B3 | selection | 0.9385, 0.9603, 0.9644 | 0.9786, 0.9786, 0.9701 | 0.9581, 0.9686, 0.9738 | 0.9482, 0.9585, 0.9588 |
| F + B3 | calibration | 0.9779, 0.9885, 0.9771 | 0.9929, 0.9929, 0.9861 | 0.9874, 0.9950, 0.9925 | 0.9752, 0.9754, 0.9634 |
| B + B3 | calibration | 0.9770, 0.9885, 0.9781 | 0.9929, 0.9929, 0.9861 | 0.9874, 0.9950, 0.9950 | 0.9680, 0.9754, 0.9635 |
| O + B3 | calibration | 0.9779, 0.9885, 0.9762 | 0.9861, 0.9929, 0.9861 | 0.9874, 0.9950, 0.9950 | 0.9752, 0.9754, 0.9635 |
| Frozen Ordered + B3 | calibration | 0.9817, 0.9836, 0.9797 | 0.9929, 0.9861, 0.9861 | 0.9824, 0.9874, 0.9799 | 0.9751, 0.9800, 0.9774 |

Pair-stream segment F1@50 is 0.9551 to 0.9642 on selection and 0.9769 to 0.9884 on calibration for F, B and O, against
0.9569 to 0.9615 and 0.9708 to 0.9825 for frozen Ordered + B3. The full rows (pair streams, switches, same-view join
cuts, A4C/A5C confusions, argmax and stored-setting rows, archived A01, A05 and A06-R1 rows) are in `report_tables.md`.

**Multi-stream TP / FP / FN with B3** (reference segments: 502 on selection, 518 on calibration).

| System | Selection, seeds 0 / 1 / 2 | Calibration, seeds 0 / 1 / 2 |
|---|---|---|
| F + B3 | 488/34/14, 483/31/19, 483/68/19 | 509/14/9, 516/10/2, 512/18/6 |
| B + B3 | 488/38/14, 484/30/18, 483/65/19 | 510/16/8, 516/10/2, 513/18/5 |
| O + B3 | 488/33/14, 485/31/17, 484/65/18 | 509/14/9, 516/10/2, 512/19/6 |
| Frozen Ordered + B3 | 473/33/29, 484/22/18, 487/21/15 | 510/11/8, 511/10/7, 508/11/10 |

**Archived rows** are listed as stored, with no retuning or paired comparison. Their segment F1@50 for seeds 0, 1 and 2
is as follows.
- A01 ASRF: 0.9366, 0.9476 and 0.9506 on selection, and 0.9796, 0.9747 and 0.9698 on calibration.
- A01 Linear + B3, which equals A05 F + B3: 0.9473, 0.9569 and 0.9388, then 0.9768, 0.9779 and 0.9732.
- A05 B + B3: 0.9481, 0.9550 and 0.9620, then 0.9768, 0.9789 and 0.9806.
- A05 O + B3: 0.9512, 0.9569 and 0.9610, then 0.9778, 0.9788 and 0.9816.
- A06-R1 FACT: 0.9610, 0.9570 and 0.9489, then 0.9620, 0.9584 and 0.9611.

These rounds trained other heads on frozen features. A07 differs from them in representation, sampling and
optimisation, so no causal comparison between them is made.

## Paired comparisons of segment F1@50 (B3)

Each cell gives the difference and its 95% bootstrap interval for seeds 0, 1 and 2. All verdicts are "no consistent
difference detected".

| Comparison | Selection | Calibration |
|---|---|---|
| O − B (primary) | +0.0046 [+0.0010, +0.0090]; +0.0001 [−0.0028, +0.0031]; +0.0010 [−0.0052, +0.0088] | +0.0009 [−0.0001, +0.0030]; +0.0000 [+0.0000, +0.0000]; −0.0019 [−0.0059, +0.0000] |
| B − F | −0.0037 [−0.0074, −0.0009]; +0.0020 [−0.0071, +0.0110]; +0.0026 [−0.0017, +0.0076] | −0.0009 [−0.0040, +0.0019]; +0.0000 [+0.0000, +0.0000]; +0.0010 [+0.0000, +0.0031] |
| O − F | +0.0009 [+0.0000, +0.0029]; +0.0021 [−0.0068, +0.0111]; +0.0036 [−0.0009, +0.0108] | +0.0000 [−0.0028, +0.0028]; +0.0000 [+0.0000, +0.0000]; −0.0009 [−0.0058, +0.0029] |
| O − frozen Ordered | +0.0156 [+0.0026, +0.0304]; −0.0075 [−0.0206, +0.0033]; −0.0433 [−0.0693, −0.0196] | −0.0038 [−0.0121, +0.0039]; +0.0049 [−0.0028, +0.0130]; −0.0036 [−0.0122, +0.0042] |
| B − frozen Ordered | +0.0109 [−0.0024, +0.0261]; −0.0076 [−0.0205, +0.0032]; −0.0444 [−0.0709, −0.0206] | −0.0047 [−0.0127, +0.0022]; +0.0049 [−0.0028, +0.0130]; −0.0017 [−0.0093, +0.0058] |
| F − frozen Ordered | +0.0146 [+0.0019, +0.0294]; −0.0095 [−0.0233, +0.0013]; −0.0470 [−0.0740, −0.0224] | −0.0038 [−0.0123, +0.0039]; +0.0049 [−0.0028, +0.0130]; −0.0027 [−0.0097, +0.0041] |

## Training

All nine runs completed 3,750 updates from update 0, with no interruption or resume. Each took 0.473 to 0.476 GPU hours,
with a peak of 12,380 MiB allocated by torch. The check measured 0.459 to 0.466 seconds per unfrozen update. Including
the faster warm-up epoch, the runs averaged about 0.455 seconds per update.

The monitors below are training-data fit only. They selected nothing.
- **Last-epoch means** (the last 375 updates). F's loss was 0.191, 0.108 and 0.150. B's was 0.239, 0.134 and 0.201.
  O's total was 0.201, 0.120 and 0.179, with a transcript term L_order of 0.134, 0.092 and 0.138.
- **Target accuracy.** In the last epoch, 92.8 to 93.0% of the constructed groups' targets were right in seed 0, 96.0
  to 96.1% in seed 1 and 93.5 to 93.6% in seed 2. In the first epoch it was 90.0 to 91.3%.
- **Exact orders.** The constructed groups' argmax order equalled the sampled order in 87.2% of the last epoch's
  updates in seed 0, 92.0 to 92.3% in seed 1 and 85.6 to 85.9% in seed 2.
- **Gradient norms.** Before clipping, the per-epoch mean gradient norm was 2.35 to 9.43 over all runs and epochs, above
  the clip threshold of 1.
- **The arms move together.** The three runs of a seed see the same batches, and their epoch means rise and fall
  together. The epoch-to-epoch swings are larger than the gaps between arms.

`training_curves.png` shows L_order, L_F and the constructed-group target accuracy per epoch. The grey line marks update
375, where the encoder was unfrozen. The figure's title says "dashed: encoder unfrozen"; that refers to the grey line,
while the dashed curves are seed 1. `smoke_curves.png` shows the 100-update check runs.

## What this round does not show

- **No cause is shown.** The round does not show that plain cross-entropy caused the remaining segment errors, or that a
  sequence-level loss cannot help with another representation, group size, decoder or schedule. It tested one fixed
  recipe.
- **Seed 2 is not explained.** Its continued models cut more often on selection under every decoder. On calibration
  they also cut more with B3 (multi FP segments 18, 18 and 19 against 11; native switches 15, 14 and 16 against 3),
  slightly more at the stored setting (15 to 16 against 11, with intervals starting at zero) and not with argmax (24
  against 30), without a segment F1@50 gap. Neither the objective nor the decoder is credited or blamed for it.
- **Small counts.** The recovered, lost, native, switch and join-cut counts rest on fewer than 10 sessions in most seed
  and half cells, and so do most calibration counts of unmatched predicted segments. They are descriptive.
- **No comparison with A05.** A05 trained linear heads on frozen features with other sequences and sampling. Its rows
  are listed, not compared.
- **No generalisation claim.** Both development halves have been reused many times, constructed streams are not real
  scans, and grouping is by session, not by patient.
- **No novelty claim.** Run balancing and the forward-sum transcript loss are established ideas.

## Deviations and launch record

- **Two report fixes after the dry run.**
  - The first dry run (08:01 to 08:05) stopped with KeyError `a05_summary`. A05's summary is pinned under
    `source.a05_summary`, while the report looked for every archived summary under `source.archived_rows`. dcae2cd adds
    `archived_pin`, which reads both, and a test of every archived pin.
  - dcae2cd placed the new function between two methods of the report's context class. The method after it,
    `v01_output`, became dead code, and the calibration rerun at 08:08 stopped with AttributeError before any model
    ran. Its ledger line is labelled GPU inference and holds 0.003 hours of set-up time. 5ec611c moved the function
    below the class and added a test of every reader method.
  - Against 095320c the report differs only by `archived_pin` and its use. The training code, its digest (1733e5e6) and
    the run identities did not change.
  - **The superseded choices.** The first calibration's choice record (sha256 c7763b22, report code 6dbea9c8) was never
    committed, proven or used. It is kept on the server under
    `runs/video_sequence_objective_v1/calibration/superseded_report_code_6dbea9c8c205/`. The committed record (b4aa8fdc,
    report code 1894d3b4) differs from it only in the report-code digest and the frame-decoding time. Every choice, grid
    row, parity value and reproduced row is identical.
  - The failed dry run and the failed calibration rerun are booked with ok = false (0.012 GPU and 0.058 CPU hours),
    besides the test line below.
- **The review's test.** f39c4b3 added the test the reviewer suggested, for a nonzero count of skipped reference runs.
  It changed only the tests.
- **Server test runs.** Two server test runs at 03:16 gave no usable summary line. The first, at 03:16:13, had its
  output cut. The second, at 03:16:36, printed none because the repository's pytest options are quieter. A rerun at
  03:17:04 exited 0 with 51 tests. The first two are booked in one ok = false line (0.007 CPU hours).
- **The review's ledger line.** It was booked by hand with 0.297 hours. Its finished field (03:18:09) is the booking
  time, not the review's end.
- **Launch.** Status checks at 03:19 and 03:20 saw the queue process, the first run's process and six loader workers
  alive, with the run's log at update 0 of 3,750. At 03:24:51 nvidia-smi showed 15,077 MiB of GPU memory in use at 96%
  utilisation, and the run had logged 750 of 3,750 updates in two epoch lines. These figures come from the live
  process list and log at those times and were not kept as a separate record.

## Fact-check

An independent fact-checker, a separate agent, checked the round with its own code from 08:55 to 09:20 UTC on
2026-10-16 (`fact_check.md`). That was after the single selection evaluation, against this README's draft. Two
read-only helper agents worked inside that window, and the fact-checker re-checked each of their findings itself.

**What it covered.**
- the commit chain, the pins, the failing tests at 3b92fe8 and the server copies of the round's files;
- the run identities (63 comparisons) and each run's updates, learning rates and weights;
- all 3,750 updates of every seed's manifest, re-derived with its own sampler (90,000 windows, 0 mismatches);
- the loss on 336 cases against explicit path enumeration (largest differences 4.9e-16 in value and 1.7e-15 in
  gradient), and a GPU replay of update 0 of each arm, which matched the check record within 6.9e-17;
- 17,037 of the 17,039 values of `summary.json`, recomputed or compared with their sources, with 0 mismatches. The
  two frame-decoding times cannot be recomputed;
- a GPU re-inference of all nine models on 60 sequences (2,511 windows; largest logit difference 7.6e-6, equal
  argmax);
- the barrier (its own walk of the git objects, the proof and the single opening) and the nine choice grids (4,680
  values, 0 mismatches);
- the ledger and the budget projections;
- 1,085 development ids, none of which appears in the result files.

**Verdict.** Pass with corrections. It found no numeric error. It found two medium and nine low wording or flagging
issues and several nits, all corrected before publication. `fact_check.md` lists each one.

## Commands

```
python scripts/prepare_video_sequence_objective.py --config configs/experiments/video_sequence_objective_v1.yaml
ECHO_GPU_MEM_FRACTION=0.6 python scripts/train_video_sequence_objective.py --config configs/experiments/video_sequence_objective_v1.yaml --check
ECHO_GPU_MEM_FRACTION=0.6 python scripts/train_video_sequence_objective.py --config configs/experiments/video_sequence_objective_v1.yaml --queue
ECHO_GPU_MEM_FRACTION=0.6 python scripts/report_video_sequence_objective.py --config configs/experiments/video_sequence_objective_v1.yaml --calibrate
ECHO_GPU_MEM_FRACTION=0.6 python scripts/report_video_sequence_objective.py --config configs/experiments/video_sequence_objective_v1.yaml --dry-run
python scripts/report_video_sequence_objective.py --config configs/experiments/video_sequence_objective_v1.yaml --proof --choices-commit a96dfc50e03058cff8750f3d43d3617cb5f4ff94   # local
ECHO_GPU_MEM_FRACTION=0.6 python scripts/report_video_sequence_objective.py --config configs/experiments/video_sequence_objective_v1.yaml --selection \
    --choices-sha256 b4aa8fdca51343a8ed08d29e2079030cf28d6333570044222d51da181897dffa --choices-commit a96dfc50e03058cff8750f3d43d3617cb5f4ff94
```

The server ran Python 3.12.3, NumPy 2.5.2, pandas 3.0.5, SciPy 1.18.1, torch 2.14.0+cu130, torchvision 0.29.0+cu130,
CUDA 13.0 and cuDNN 92400, from `/home/william/echo-view-routing/.venv`.

## Provenance

- **Source pins.** The configuration pins V01's configuration and summary, the dense-target and A01 configurations, the
  A05 summary and loss module, A03's configuration for the metadata helpers, the archived A01 and A06-R1 summaries, the
  protocol, V01's saved outputs and 18 reused helper files. Every pin was checked before use.
- **Source weights.** V01 Ordered final weights `models_frozen/raw_video_context_v1/ordered_s{0,1,2}/final.pt` (80f13a42,
  74276323 and 923eb3af). Each continuation loaded them strictly.
- **Manifests.** VSO_s0 0d5653cd, VSO_s1 d92b5a05 and VSO_s2 2310dd5e. Each was built twice with identical results.
- **Code and configuration.** The configuration sha256 is f8d50e07 (1a76407), unchanged since. The training code digest
  is 1733e5e6, and the report code digest is 1894d3b4 (5ec611c). Both are recorded in `choices.json` and `summary.json`.
- **Runs.** Every run identity, final weight sha256 and frozen path is in `summary.json`. The frozen models are
  `models_frozen/video_sequence_objective_v1/{frame,segment,order}_s{0,1,2}`, read-only.
- **Choices.** The choice record's sha256 is b4aa8fdc at commit a96dfc50. The proof covers the commit, the trees on the
  path and blob 8490a018. The selection stage verified it on the server, which has no git. The selection half was opened
  once.
- **Parity.** Re-inferring each seed's V01 source model on the first 64 calibration windows matched V01's saved logits
  exactly (largest difference 0.0), with equal argmax.
- **Reproduction.** V01's stored argmax and B3 rows of frozen Ordered were reproduced on both halves for all three
  seeds. The window plans matched V01's saved outputs.
- **Halves.** Selection has 541 sequences, 7,666 windows and 21,359 frames. Calibration has 472 sequences, 7,021 windows
  and 19,207 frames.

## Files

- `README.md`. This report.
- `summary.json`. Every system row, comparison, verdict, accounting entry, goal, reading input, parity, identity and cost.
- `report_tables.md`. The generated tables. In three of its tables the row keys contain "|" characters, which shift
  the Markdown cells. The values are complete in `summary.json`.
- `choices.json`. The calibration choices with each model's 40-setting grid and the provenance. Committed before
  selection.
- `preparation.json` and `checks.json`. The preparation and check records made by the reviewed code.
- `code_review.md`. The implementation review and its follow-up.
- `fact_check.md`. The independent fact-check.
- `training_curves.png` and `smoke_curves.png`. Training-data monitors.

Per-sequence probabilities, decodings and case lists stay on the server under `runs/video_sequence_objective_v1/`.
