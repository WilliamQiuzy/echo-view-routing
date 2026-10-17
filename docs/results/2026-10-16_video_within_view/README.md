# Within-view output regularisation on the Ordered video model (2026-10-16)

This round asked whether penalising changes between neighbouring outputs that share a training label reduces false
fragments without suppressing real view changes. It compared that masked penalty with the same penalty on every
neighbour pair and with A07's plain cross-entropy control.

**What was trained.**
- The model is V01's Ordered TargetMViT. It is torchvision's MViTv2-S with its 512-d projection, V01's target-marker
  stem and a five-class head, **34,640,549 parameters** in all. No parameter was added.
- Two new arms were trained, three seeds each, so there are six new runs. Every run started from its seed's frozen V01
  Ordered weights, never from an A07-trained model. Each made 3,750 updates of 8 windows and kept its final weights.
  There was no validation, epoch choice or training-score gate.
- Every run read A07's unchanged training manifest of its seed, so the six runs and A07's F saw the same windows,
  targets, markers, pixels and batch order. They also started from the same model random state.
- V01's optimiser and schedule were reused unchanged, as in A07. AdamW used a learning rate of 1e-4 for the marker stem
  and the classifier and 1e-5 for the encoder, including the projection. Weight decay was 1e-4, the gradient norm was
  clipped at 1 and both rates followed V01's warm-up and cosine factor. For the first 375 updates the encoder was frozen.
- **The control F is not new.** It is A07's three plain cross-entropy continuations, frozen under their A07 identities.
  They were not retrained, retuned or re-chosen. Their saved outputs, decoder choices and rows were reused and
  reproduced exactly.

**The groups.** Each update holds A07's two groups of four windows. A window is V01's input: 16 frames centred on a grid
centre, with the marker on the centre's slot. Grid centres lie every 5 samples.
- **The native group** takes four consecutive grid centres of one training clip, so its four labels are equal.
- **The constructed group** takes four consecutive grid centres of one constructed training stream, with at least one
  label change among them.

**The two arms.** Both add a fixed penalty to A07 F's plain cross-entropy, computed from the same forward.
- **The penalty.** Between each window and the previous one in its group, and for each class, it takes the squared
  difference of the log-probabilities, capped at 16. The previous window's log-probability is held fixed (detached), so
  the penalty pulls each output towards its predecessor and not the reverse. The coefficient is 0.15. Coefficient,
  cap and detach follow MS-TCN's smoothing loss.
- **A (all neighbours).** The penalty averaged over all three neighbour pairs and five classes of each group, then over
  the two groups.
- **M (equal-label neighbours).** The penalty only on neighbour pairs whose training labels are equal, averaged per group
  over its selected pairs and classes, then over the two groups. A group without such a pair would contribute zero.
- **What M − A tests.** Every native group has three equal-label pairs, so A and M treat it identically. They differ
  only in the constructed groups. There M drops the pairs that cross a label change and gives each kept pair a larger
  share of the group's weight. M − A therefore tests this mask-and-normalisation rule, not equal per-pair weights.
- **Training labels only.** Labels choose the pairs during training. At inference every model gives V01's ordinary
  single-window CLS output, with no label, mask, boundary or group input.

The task sheet is `docs/superpowers/plans/2026-10-16-video-within-view-regularization.md` (6cbb67b). The fixed
configuration is `configs/experiments/video_within_view_v1.yaml` (19ff07a).

**What kind of evidence this is.**
- **Reused data.** The two development halves (selection and calibration) have been used and inspected in many earlier
  rounds. This is a matched comparison on reused development data, not an independent validation.
- **Grouping.** The hold-out is by session, not by patient. Constructed streams are spliced from separate recordings and
  are not real scans. They reuse recordings, so their units are not independent cases. The three seeds share the same
  cases.
- **Labels.** Native clips carry the public recording label. No label was changed.
- **Untouched data.** The test split and EchoNet-Pediatric were not read.
- **Novelty.** The penalty is MS-TCN's established smoothing loss. The label mask and the per-group normalisation are
  this task's adaptation. No new architecture is claimed.

**Order of work.** All times are UTC on 2026-10-16.
- **Before any code.** 6cbb67b committed the task sheet at 12:39. 19ff07a committed the configuration at 12:50.
- **Tests and code.** 042b7ef committed the tests at 12:55, and they failed at import. 170017c committed the code at
  13:05.
- **Preparation and check.** The preparation ran from 13:06:46 to 13:08:18 and the GPU check from 13:08:57 to 13:13:03.
  Both passed.
- **Review.** An independent implementation review passed the code (`code_review.md`). 39bb154 committed the records at
  13:27.
- **Training.** The queue was launched at 13:27:17, the time of the launch command; its first record is at 13:27:21. The
  six runs ran one after another from 13:27:21 to 16:18:14 and were frozen at 16:18:20.
- **Report fix.** While the first run trained, eb937a0 added the review's shared-case counts to the report at 13:29.
  This was before calibration, and it changed no run identity.
- **Calibration.** It ran from 16:20:56 to 16:37:11. The dry run that followed, from 16:38 to 16:44, used the
  calibration half for both halves and passed.
- **Barrier.** c99bb6af committed the calibration choices with their provenance at 16:45:02. A git-object proof of that
  commit was made immediately after.
- **Selection.** The selection half was opened once, at 16:45:43, and evaluated by 17:05:52.
- **Fact-check.** A first independent fact-check stalled with the agent infrastructure at 18:16:54. A second attempt
  recomputed the results on 2026-10-17 from 14:17 to 14:35 (see "Fact-check").

## Answers

Values are given for seeds 0, 1 and 2 in that order, on the selection half and then the calibration half. The main
metric is multi-stream segment F1@50 with each model's B3 setting. A difference is consistent only when all three
seeds' 95% intervals exclude zero strictly on the same side in a half. Otherwise it is "no consistent difference
detected", which is not the same as equality. None of the 30 segment-F1 verdicts (three decoders, five comparisons, two
halves) is consistent.

### 1. No consistent difference was detected for the primary comparison M − A

- **Segment F1@50.** M − A is +0.0034, −0.0008 and +0.0079 on selection, and −0.0057, +0.0000 and −0.0088 on
  calibration. Every selection interval includes zero. On calibration, seed 0's interval [−0.0118, −0.0010] and seed
  2's [−0.0185, −0.0019] lie below zero, and seed 1's includes zero.
- **The halves pull apart.** On selection, M's multi boundary recall is higher than A's (+0.0183, +0.0105 and +0.0105;
  seeds 0 and 1 above zero) and M has fewer FN segments (−7, −2 and −4). On calibration, M has more FP segments (+4, +1
  and +1) and, in seed 2, more FN segments (+8) and lower boundary recall (−0.0126). None of these is consistent.
- **Reference segments (multi streams).**
  - On selection, M recovered 7, 2 and 8 reference segments that A missed and lost 0, 0 and 4. The 17 recoveries are
    11 distinct segments, 6 of them recovered in two seeds. 15 of the 17 are A4C or A5C segments, and 11 have at most
    16 samples.
  - On calibration, M recovered 0, 1 and 0 and lost 2, 0 and 8. The 10 losses are 9 distinct segments, and 9 of the 10
    are A4C or A5C segments.
  - Every per-seed count rests on at most 8 sessions, fewer than 10.
- **Conclusion.** Under this recipe, no consistent multi segment F1@50 difference between M and A was detected.

### 2. Against F, no consistent difference; on selection in seed 2, neither new arm shows F's over-segmentation

- **M − F.** It is +0.0066, +0.0029 and +0.0470 on selection, and −0.0009, +0.0000 and −0.0022 on calibration. Only
  seed 2's selection interval [+0.0251, +0.0710] excludes zero.
- **A − F.** It is +0.0032, +0.0037 and +0.0391 on selection, and +0.0048, −0.0000 and +0.0066 on calibration.
  - On selection, only seed 2's interval [+0.0180, +0.0632] excludes zero.
  - On calibration, seed 0's interval [+0.0009, +0.0105] and seed 2's [+0.0017, +0.0129] lie above zero. Seed 1's
    [−0.0029, +0.0028] includes zero, so the difference is not consistent.
- **F's seed 2.** A07 reported that F's seed 2 over-segments on selection. Here F + B3 has 68 multi FP segments in seed
  2, against 25 for A and 21 for M. Seed 2's segment F1@50 is 0.9174 for F, 0.9564 for A and 0.9644 for M.
- **Fewer FP segments, fewer matched boundaries.** A has fewer multi FP segments than F in every seed and half (−10, −5
  and −43 on selection; −3, −1 and −6 on calibration), but none of this is consistent. On selection, A's boundary
  recall is lower than F's (−0.0157, −0.0079 and −0.0183). Seeds 0 and 2 lie below zero and seed 1's interval ends at
  zero.
- **Reference segments (multi streams).**
  - A against F on selection: recovered 1, 1 and 8 and lost 7, 2 and 8. The 17 losses are 12 distinct segments, 5 of
    them lost in two seeds, and 12 have at most 16 samples. On calibration A recovered 2, 0 and 2 and lost 0, 1 and 1.
  - M against F on selection: recovered 1, 2 and 8 and lost 0, 1 and 4. On calibration it recovered 0, 0 and 1 and lost
    0, 0 and 8, all 8 in seed 2.
  - Every per-seed count rests on at most 7 sessions.

### 3. Against frozen Ordered + B3, no consistent difference

- **M − frozen Ordered.** It is +0.0213, −0.0066 and +0.0000 on selection, and −0.0047, +0.0049 and −0.0048 on
  calibration. Only seed 0's selection interval [+0.0081, +0.0370] excludes zero.
- **A − frozen Ordered.** It is +0.0179, −0.0059 and −0.0079 on selection, and +0.0010, +0.0048 and +0.0040 on
  calibration. Only seed 0's selection interval [+0.0064, +0.0318] excludes zero. Seed 2's calibration interval
  [+0.0000, +0.0089] starts at zero.
- **Means.** The three-seed means are 0.9557 / 0.9850 for A and 0.9593 / 0.9801 for M, against 0.9544 / 0.9817 for
  frozen Ordered + B3 and 0.9404 / 0.9812 for F.
- **Reference segments (multi streams).**
  - M on selection recovered 16, 4 and 7 segments that frozen Ordered missed and lost 0, 4 and 7. The 27 recoveries are
    16 distinct segments, 4 of them recovered in all three seeds. On calibration M recovered 3, 5 and 4 and lost 4, 0
    and 7.
  - A on selection recovered 9, 4 and 3 and lost 0, 6 and 7. On calibration it recovered 3, 5 and 5 and lost 2, 1
    and 0.
  - Every per-seed count rests on fewer than 10 sessions (0 to 9).

### 4. The goal: A meets it, which is not a new winner

- **A.** Its three-seed means are 0.9557 on selection and 0.9850 on calibration, above both targets (0.944409687184662
  and 0.9821421709894332). They are also above the calibration sensitivity target 0.9838935784747763, by 0.0011. Every
  seed and half has a feasible decoder, boundary recall of at least 0.95 and native macro-F1 at or above the floor (0.973
  on selection, 0.980 on calibration). **A meets the original goal.** No earlier system in the experiment index met every goal and floor.
- **Why this is not a new winner.** A − frozen Ordered shows no consistent difference in either half. The task sheet
  says that meeting the old targets without beating the stronger frozen reference is not a new winner. A is also the
  comparison arm, not the candidate, and the fixed readings do not credit it (section 8).
- **M** has the highest selection mean of the four (0.9593) but misses the calibration mean by 0.0020. It meets every
  floor.
- **F** (0.9404 / 0.9812) misses both means. **Frozen Ordered + B3** (0.9544 / 0.9817) passes the selection mean, misses
  the calibration mean and fails the native floor once (seed 2, selection).

### 5. Native recognition

- **Floors.** Neither A nor M fails a floor in any seed or half. On selection both have native macro-F1 0.9786, 0.9877
  and 0.9786. On calibration A has 0.9929, 0.9856 and 0.9856 and M has 0.9929, 0.9929 and 0.9861. No native macro-F1
  difference is consistent.
- **A5C read as A4C.** With B3, A reads one A5C clip as A4C on calibration in seeds 1 and 2. It is the same clip in both
  seeds. M reads it correctly. No other system at its own B3 setting reads an A5C clip as A4C. At the stored Ordered
  setting, A, M and F each read one in seed 2 on calibration.
- **A4C read as A5C.** On selection A and M read 2, 1 and 2 A4C clips as A5C, against 2, 2 and 3 for frozen Ordered and
  2, 2 and 2 for F. In seed 2, A and M read correctly one A4C clip that frozen Ordered reads as A5C. That clip is behind
  frozen Ordered's only floor failure (0.9701), and A and M have 0.9786 there. In seed 1, A and M also read correctly one
  A4C clip that frozen Ordered reads as A5C, and one that F reads as A5C.
- Each of these changes rests on one clip and one session.

### 6. Transcripts, switches and same-view joins

- **Transcript distance.** The normalised multi transcript edit distance with B3 is as follows.
  - A: 0.0422, 0.0495 and 0.0489 on selection, and 0.0248, 0.0142 and 0.0199 on calibration.
  - M: 0.0472, 0.0444 and 0.0352, then 0.0248, 0.0159 and 0.0254.
  - F: 0.0648, 0.0537 and 0.1234, then 0.0297, 0.0159 and 0.0301.
  - Frozen Ordered: 0.0645, 0.0438 and 0.0424, then 0.0279, 0.0206 and 0.0295.
  - No interval was computed for these distances, and a lower distance alone is not a segment win.
- **Native switches.** Counting label switches inside native clips with B3, A has 11, 12 and 12 on selection and M 10,
  9 and 8, against 16, 10 and 25 for F and 12, 11 and 13 for frozen Ordered. On calibration A has 3, 4 and 4, M 4, 3
  and 5, F 5, 4 and 15 and frozen Ordered 6, 4 and 3. These come from 2 to 11 clips and 2 to 8 sessions per seed and
  half. Fewer switches alone are not a success.
- **Same-view joins.** On selection A cuts 5, 4 and 3 of the 230 same-view joins and M 4, 4 and 3, against 5, 3 and 7
  for F and 4, 1 and 1 for frozen Ordered. On calibration A cuts 5, 4 and 6 of 201, M 6, 3 and 4, F 4, 3 and 7 and
  frozen Ordered 5, 5 and 5.

### 7. Sensitivities

These rows show the same models with other decoders. Nothing is chosen from them.
- **The seed's stored frozen Ordered setting.** This is the same numeric B3 setting for every model of a seed.
  - M − A is +0.0118, +0.0083 and +0.0031 on selection, and −0.0028, +0.0000 and −0.0039 on calibration. Seeds 0 and 1
    lie above zero on selection; seed 2 includes zero.
  - M − F is +0.0150, +0.0108 and +0.0253, then +0.0018, −0.0000 and −0.0011. A − F is +0.0033, +0.0026 and +0.0222,
    then +0.0046, −0.0001 and +0.0027.
  - Five other results at this setting are consistent, all on selection. A − F has lower multi boundary recall, which
    is unfavourable to A. A − F and M − F have fewer multi FP segments and fewer native cuts per clip, which favours the
    new arms.
- **Direct argmax.**
  - M − A is +0.0073, −0.0130 and +0.0067 on selection, and −0.0111, +0.0010 and −0.0085 on calibration.
  - M − F is −0.0045, −0.0058 and +0.0036, then −0.0045, +0.0029 and −0.0020. A − F is −0.0118, +0.0073 and −0.0031,
    then +0.0066, +0.0019 and +0.0065.
  - No difference is consistent.

### 8. The fixed readings

- **Applied: trade-off, consistent negatives, no qualifying gain, and end.**
- **Not applied: full success and the mask-specific signal.** M − A and M − F are not consistent positive in any half.
- **Not applied: the smoothing-only signal.** A − F and M − F are never both consistent positive in the same half. A − F
  comes closest on calibration, with two of three intervals above zero.
- **Not applied: release of harm only.** M − A does not qualify.
- **Trade-off flags.** These are descriptive, not causal. Each flagged cell pairs a recovered segment or fewer FP
  segments with more lost than recovered reference segments.
  - M − A on calibration: 1 recovered and 10 lost, with 38 multi FP segments for M against 32 for A.
  - M − F on calibration: 1 recovered and 8 lost, with 38 FP segments against 42.
  - A − F on selection: 10 recovered and 17 lost, with 75 FP segments against 133.
  - The counts are small and rest on fewer than 10 sessions per seed.
- **Consistent negatives.** The five consistent results at the stored setting (section 7) are listed. One is
  unfavourable: A − F's lower boundary recall on selection. The other four are fewer FP segments or native cuts. The
  reading applies when any consistent negative exists, and every listed item carries a flag that says whether it
  disfavours the candidate.
- **What "end" means.** This fixed recipe closes after this report. No coefficient, truncation, mask, epoch, seed,
  sampling, decoder or readout search and no automatic follow-on is authorised.

### 9. Cost

- **GPU.** 3.392 of 6 hours before the fact-check: the check 0.068 h, the six trainings 2.847 h, and calibration, dry
  run and selection inference 0.476 h.
- **CPU.** 0.590 of 3 hours before the fact-check. This covers the tests, the preparation, the review (0.317 h), the
  queue overhead, the calibration grids, the dry run and the selection bootstrap.
- **Fact-check.** The first attempt (one agent and two helpers) stalled with the agent infrastructure at 18:16:54 UTC on
  2026-10-16 and was stopped by the harness. Its three wall clocks (1.082, 1.036 and 1.043 hours, of which 0.745 hours
  were commands) are booked as 3.161 CPU hours, marked not ok. The second attempt (one agent, no helpers) ran for 0.303
  hours on 2026-10-17, including one 44-second GPU job (0.012 GPU hours). A probe that confirmed the determinism
  explanation (see "Fact-check") took 10 GPU seconds.
- **The CPU cap.** The configuration books every agent's wall clock as CPU hours. Under that rule the stalled attempt
  brought the ledger to 3.751 CPU hours, over the 3-hour cap. On 2026-10-17 the user kept the rule and authorised up to
  1.5 CPU hours beyond the cap for one lean fact-check agent. This was taken as a ceiling of 4.5 hours.
- **In all.** 3.407 of 6 GPU hours and 4.043 CPU hours, 1.043 over the original cap and within the authorised ceiling,
  before a last id scan of the result files.
- The ledger is `runs/video_within_view_v1/ledger.jsonl` on the server. No stage of the round failed. The three lines of
  the stalled fact-check attempt are booked as not ok.

## Terms

- **Native clip.** One whole public recording with one view label.
- **Pair stream and multi stream.** Constructed sequences that splice fragments of two, or of several, recordings.
- **Run.** A maximal stretch of one class inside a sequence. Adjacent fragments of the same view form one run.
- **B3.** The simple change-point decoder (half-window, threshold, NMS radius 5). Each new model's setting was chosen
  once on calibration from the 40-setting grid. A setting is feasible with native macro-F1 at least 0.980 and multi
  boundary recall at least 0.95; the highest segment F1@50 wins, with ties broken by grid order. F keeps its A07 setting.
- **Segment F1@50.** Predicted and reference segments are matched one to one by class, with temporal IoU at least 0.5.
  F1 is 2 × matched / (predicted + reference), pooled over multi streams.
- **Recovered and lost.** Seen from system a against system b, a reference segment is recovered when only a matches it,
  and lost when only b matches it.
- **FP and FN segments.** Unmatched predicted segments, and unmatched reference segments.
- **Boundary recall and precision.** The shares of reference view changes, and of predicted changes, matched one to one
  within 0.5 s.
- **Native macro-F1.** The class-averaged F1 of native clips, each named by its majority decoded label.
- **Shared cases.** The three seeds read the same development cases. A pooled count over the seeds can count one
  reference segment up to three times, so the distinct segments and their seed counts are given beside it.
- **Bootstrap.** 2,000 draws with seed 20261002, resampling native clips by session and streams by whole stream. The
  same draws serve every comparison.

## Main system rows (B3)

Each cell gives seeds 0, 1 and 2.

| System | Half | Segment F1@50 | Native macro-F1 | Boundary recall | Boundary precision |
|---|---|---|---|---|---|
| A + B3 | selection | 0.9563, 0.9545, 0.9564 | 0.9786, 0.9877, 0.9786 | 0.9660, 0.9660, 0.9686 | 0.9560, 0.9510, 0.9536 |
| M + B3 | selection | 0.9598, 0.9537, 0.9644 | 0.9786, 0.9877, 0.9786 | 0.9843, 0.9764, 0.9791 | 0.9471, 0.9491, 0.9639 |
| F + B3 | selection | 0.9531, 0.9508, 0.9174 | 0.9786, 0.9786, 0.9786 | 0.9817, 0.9738, 0.9869 | 0.9328, 0.9442, 0.8747 |
| Frozen Ordered + B3 | selection | 0.9385, 0.9603, 0.9644 | 0.9786, 0.9786, 0.9701 | 0.9581, 0.9686, 0.9738 | 0.9482, 0.9585, 0.9588 |
| A + B3 | calibration | 0.9827, 0.9885, 0.9837 | 0.9929, 0.9856, 0.9856 | 0.9899, 0.9950, 0.9925 | 0.9801, 0.9802, 0.9753 |
| M + B3 | calibration | 0.9770, 0.9885, 0.9749 | 0.9929, 0.9929, 0.9861 | 0.9874, 0.9950, 0.9799 | 0.9728, 0.9754, 0.9799 |
| F + B3 | calibration | 0.9779, 0.9885, 0.9771 | 0.9929, 0.9929, 0.9861 | 0.9874, 0.9950, 0.9925 | 0.9752, 0.9754, 0.9634 |
| Frozen Ordered + B3 | calibration | 0.9817, 0.9836, 0.9797 | 0.9929, 0.9861, 0.9861 | 0.9824, 0.9874, 0.9799 | 0.9751, 0.9800, 0.9774 |

F's rows equal A07's stored rows exactly, and frozen Ordered's equal V01's. Pair-stream segment F1@50 is 0.9617 to
0.9808 on selection and 0.9709 to 0.9826 on calibration for A, and 0.9663 to 0.9712 and 0.9797 to 0.9855 for M. The
full rows (pair streams, switches, same-view join cuts, A4C/A5C confusions, argmax and stored-setting rows, shared
cases and training penalties) are in `report_tables.md`.

**Multi-stream TP / FP / FN with B3** (reference segments: 502 on selection, 518 on calibration).

| System | Selection, seeds 0 / 1 / 2 | Calibration, seeds 0 / 1 / 2 |
|---|---|---|
| A + B3 | 482/24/20, 482/26/20, 483/25/19 | 511/11/7, 515/9/3, 513/12/5 |
| M + B3 | 489/28/13, 484/29/18, 487/21/15 | 509/15/9, 516/10/2, 505/13/13 |
| F + B3 | 488/34/14, 483/31/19, 483/68/19 | 509/14/9, 516/10/2, 512/18/6 |
| Frozen Ordered + B3 | 473/33/29, 484/22/18, 487/21/15 | 510/11/8, 511/10/7, 508/11/10 |

## Paired comparisons of segment F1@50 (B3)

Each cell gives the difference and its 95% bootstrap interval for seeds 0, 1 and 2. All verdicts are "no consistent
difference detected".

| Comparison | Selection | Calibration |
|---|---|---|
| M − A (primary) | +0.0034 [−0.0054, +0.0147]; −0.0008 [−0.0078, +0.0063]; +0.0079 [−0.0037, +0.0209] | −0.0057 [−0.0118, −0.0010]; +0.0000 [−0.0028, +0.0029]; −0.0088 [−0.0185, −0.0019] |
| M − F | +0.0066 [−0.0010, +0.0173]; +0.0029 [−0.0068, +0.0144]; +0.0470 [+0.0251, +0.0710] | −0.0009 [−0.0041, +0.0019]; +0.0000 [+0.0000, +0.0000]; −0.0022 [−0.0122, +0.0061] |
| A − F | +0.0032 [−0.0104, +0.0166]; +0.0037 [−0.0091, +0.0171]; +0.0391 [+0.0180, +0.0632] | +0.0048 [+0.0009, +0.0105]; −0.0000 [−0.0029, +0.0028]; +0.0066 [+0.0017, +0.0129] |
| M − frozen Ordered | +0.0213 [+0.0081, +0.0370]; −0.0066 [−0.0196, +0.0036]; +0.0000 [−0.0134, +0.0146] | −0.0047 [−0.0128, +0.0021]; +0.0049 [−0.0028, +0.0130]; −0.0048 [−0.0157, +0.0047] |
| A − frozen Ordered | +0.0179 [+0.0064, +0.0318]; −0.0059 [−0.0203, +0.0054]; −0.0079 [−0.0224, +0.0042] | +0.0010 [−0.0053, +0.0070]; +0.0048 [−0.0037, +0.0137]; +0.0040 [+0.0000, +0.0089] |

## Training

All six runs completed 3,750 updates from update 0, with no interruption or resume. Each took 0.473 to 0.476 GPU hours.
The check measured 0.461 seconds per unfrozen update for each arm, with a peak of 12,376 MiB allocated by torch.

The monitors below are training-data fit only. They selected nothing. Both arms log both penalties.
- **The penalty's scale.** At the first update of the check, 0.15 × S was about 0.95 to 0.98, against a cross-entropy
  of 0.115. Most of the penalty comes from changes in the log-probabilities of non-target classes between neighbouring
  windows. The coefficient is fixed by the task sheet.
- **First epoch.** In both arms the equal-label penalty per edge averaged 2.16 to 2.36 and the different-label penalty
  7.18 to 7.55.
- **Last epoch** (the last 375 updates).
  - The equal-label penalty fell to 0.63 to 0.70 under A and 0.57 to 0.66 under M.
  - The different-label penalty fell from 7.18 to 7.52 to 4.04 to 4.28 under A, and less, from 7.25 to 7.55 to 5.23 to
    5.89, under M, which does not penalise those edges.
  - The cross-entropy was 0.166, 0.137 and 0.160 under A and 0.160, 0.106 and 0.139 under M.
  - Inside the constructed groups, the predicted label changed between neighbours with equal targets 81, 57 and 72
    times under A and 75, 32 and 51 times under M. This counts the last epoch's 375 updates per run.
- No group lacked a selected edge in any update. Every native group had three selected edges.

`training_curves.png` shows these monitors over training. The grey line marks update 375, where the encoder was
unfrozen. `smoke_curves.png` shows the 100-update check runs.

## What this round does not show

- **No cause is shown.** The round does not show that missing regularisation caused F's or frozen Ordered's errors, or
  that the mask cannot help with another coefficient, representation or decoder. It tested one fixed recipe.
- **A is not established as better than frozen Ordered.** A meets the old goal, but no consistent difference from
  frozen Ordered was detected.
- **The penalty's scale is not tuned.** At the start it is about eight times the cross-entropy, mostly from non-target
  classes. M − A is read under this fixed scale only.
- **Small counts.** The recovered, lost, native and switch counts rest on fewer than 10 sessions per seed and half, there
  are at most 7 join cuts per seed and half, and the three seeds share the same cases. They are descriptive.
- **No generalisation claim.** Both development halves have been reused many times, constructed streams are not real
  scans, and grouping is by session, not by patient.
- **No novelty claim.** The penalty is MS-TCN's smoothing loss. Only the label mask and the per-group normalisation are
  this task's adaptation.

## Deviations and launch record

- **The review's report fix.** The independent review found that the report did not count the seeds' shared cases.
  eb937a0 added these counts and tests of two report helpers while the first run trained, before calibration. The
  report is not part of the training code digest (56224ab7), so the run identities did not change.
- **The coefficient-0 multi-update checks.** The six-update and 3+3 comparisons with A07's F update run inside the
  warm-up. There the new parameters move by at most about 5.6e-6, close to the 1e-5 tolerance, so these checks have
  little power. The single update was bitwise identical. The tolerances come from the task sheet.
- **The stalled fact-check and the CPU overrun.** The first fact-check attempt stalled with the agent infrastructure,
  and its booking took the CPU ledger over the 3-hour cap. The user authorised the overrun for one lean second attempt
  (see "Cost"). The second attempt verified and re-ran the first attempt's independent scripts and finished their
  comparison.
- **Launch.** At 13:27:21 the queue process and the first run's process were alive. The run's log started at update 0
  of 3,750. By 13:33:04 it had logged 750 updates in two epoch lines, and nvidia-smi showed 15,077 MiB of GPU memory in
  use at 93% utilisation. These figures come from the live process list and log at those times and were not kept as a
  separate record.

## Fact-check

An independent fact-checker, a separate agent, checked the round with its own code (`fact_check.md`). The first attempt,
on 2026-10-16 with two helper agents, stalled with the agent infrastructure before finishing. The second attempt, one
agent without helpers, ran from 14:17 to 14:35 UTC on 2026-10-17 against this README's draft. It read, verified and
re-ran the first attempt's independent scripts and finished their comparison.

**What it covered.**
- the commit chain, the pins, the server copies of the round's files and A07's unchanged files;
- the six run identities, their training records, learning rates, start weights and frozen weights;
- the edge masks and repeated native targets, recounted from A07's manifests;
- 16,910 of the 16,918 values of `summary.json`, recomputed or compared with their sources, with 0 mismatches. The
  eight values not compared are output labels and decoding times;
- about 750 numbers of this README, with 0 numeric mismatches;
- the barrier (its own walk of the git objects, the proof and the single opening) and the six choice grids (3,120
  values, 0 mismatches);
- the loss on 600 random and adversarial cases against its own implementation (largest difference 8.9e-16 in value);
- a GPU re-inference of the six models on 12 sequences per half, with equal argmax everywhere. Its logits differed by
  up to 0.0016, because its job ran with the default TF32 setting. A probe under the round's determinism settings
  then reproduced the cached logits of all six models exactly (0.0) on 64 calibration windows;
- the ledger and the budget projections;
- 1,095 development ids, none of which appears in the result files.

**Not done.** It did not replay update 0 on the GPU, recompute the training penalties from forwards or run the failing
tests of 042b7ef, which it checked statically. `fact_check.md` lists each item.

**Verdict.** Pass with corrections. It found no numeric error. Four low and three nit-level wording issues were
corrected before publication.

## Commands

```
python scripts/train_video_within_view.py --config configs/experiments/video_within_view_v1.yaml --prepare
ECHO_GPU_MEM_FRACTION=0.6 python scripts/train_video_within_view.py --config configs/experiments/video_within_view_v1.yaml --check
ECHO_GPU_MEM_FRACTION=0.6 python scripts/train_video_within_view.py --config configs/experiments/video_within_view_v1.yaml --queue
ECHO_GPU_MEM_FRACTION=0.6 python scripts/report_video_within_view.py --config configs/experiments/video_within_view_v1.yaml --calibrate
ECHO_GPU_MEM_FRACTION=0.6 python scripts/report_video_within_view.py --config configs/experiments/video_within_view_v1.yaml --dry-run
python scripts/report_video_within_view.py --config configs/experiments/video_within_view_v1.yaml --proof --choices-commit c99bb6af45e3175878816d53641a1db719573220   # local
ECHO_GPU_MEM_FRACTION=0.6 python scripts/report_video_within_view.py --config configs/experiments/video_within_view_v1.yaml --selection \
    --choices-sha256 b2a731e5f4393187ee9954f710821ac04938652f0a7d00e0381ad0b8b21cf27e --choices-commit c99bb6af45e3175878816d53641a1db719573220
```

The server ran Python 3.12.3, NumPy 2.5.2, pandas 3.0.5, SciPy 1.18.1, torch 2.14.0+cu130, torchvision 0.29.0+cu130,
CUDA 13.0 and cuDNN 92400, from `/home/william/echo-view-routing/.venv`.

## Provenance

- **Sources.** The configuration pins A07's configuration, preparation, summary, choices and three scripts, A05's loss
  module, A07's three manifests and their index, and F's identities, weights, run records and saved outputs. The V01
  start weights, V01's saved outputs and summary, the protocol, A03's metadata configuration and 18 helper files are the
  pins of A07's configuration. Every file was checked against its pin before use.
- **Manifests.** VSO_s0 0d5653cd, VSO_s1 d92b5a05 and VSO_s2 2310dd5e. Their file, array and epoch digests equal A07's
  index and preparation, and every one of the 90,000 window pixel digests was recomputed from the training frames.
- **Start weights.** V01 Ordered `models_frozen/raw_video_context_v1/ordered_s{0,1,2}/final.pt` (80f13a42, 74276323 and
  923eb3af).
- **The F control.** A07's frozen F runs (6fbc987c, 958878ed and 0333fc15). Their saved outputs were opened only after
  the six freezes, one half at a time. The probabilities recomputed from their stored logits equal the stored ones, and
  their rows equal A07's.
- **Code and configuration.** The configuration sha256 is f671b2e5 (19ff07a), unchanged since. The training code digest
  is 56224ab7 (170017c), and the report code digest is 8548d580 (eb937a0). Both are recorded in `choices.json` and
  `summary.json`.
- **Runs.** The final weights are 9b144925 (A s0), d97a7132 (M s0), c8b09eca (A s1), d3d2443a (M s1), f95fed7b (A s2) and
  3df1cd68 (M s2). Every run identity and frozen path is in `summary.json`. The frozen models are
  `models_frozen/video_within_view_v1/{all,masked}_s{0,1,2}`.
- **Choices.** The choice record's sha256 is b2a731e5 at commit c99bb6af. The proof covers the commit, the trees on the
  path and blob 924efbc9, and the selection stage verified it on the server, which has no git. The selection half was
  opened once.
- **Parity.** Re-inferring each seed's V01 source and F model on the first 64 calibration windows matched their saved
  logits exactly (largest difference 0.0), with equal argmax.
- **Reproduction.** V01's stored argmax and B3 rows of frozen Ordered, and F's B3, argmax and stored-setting rows, were
  reproduced exactly on both halves for all three seeds.
- **Halves.** Selection has 541 sequences, 7,666 windows and 21,359 frames. Calibration has 472 sequences, 7,021 windows
  and 19,207 frames.

## Files

- `README.md`. This report.
- `summary.json`. Every system row, comparison, verdict, accounting entry, shared-case count, goal, reading input,
  training penalty, parity, identity and cost.
- `report_tables.md`. The generated tables.
- `choices.json`. The six calibration choices with each model's 40-setting grid, F's A07 choices and the provenance.
  Committed before selection.
- `preparation.json` and `checks.json`. The preparation and check records made by the reviewed code.
- `code_review.md`. The implementation review and its follow-up.
- `fact_check.md`. The independent fact-check.
- `training_curves.png` and `smoke_curves.png`. Training-data monitors.

Per-sequence probabilities, decodings and case lists stay on the server under `runs/video_within_view_v1/`.
