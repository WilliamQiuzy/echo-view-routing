# Video feature temporal round (2026-10-12)

This round asked whether a learned temporal model, trained on fixed video features, recovers view segments that the
simple change-point decoder loses, while it still meets the native recognition and segmentation requirements. The three
frozen Ordered context models of the raw-video context round gave a 512-dimensional feature for every stride-5 window.
Every 10 Hz sample took the feature of its nearest window. Three heads were trained per seed on exactly the same
features and batches. They are a linear class read-out with the change-point decoder B3 (Linear + B3), the official
MS-TCN and the official ASRF, which was fixed as the primary candidate. The video models were never updated. ASRF and
MS-TCN are existing methods, and no new architecture is claimed. The task sheet is
`docs/superpowers/plans/2026-10-12-video-feature-temporal-training.md`.

**What kind of evidence this is.**
- The development halves have been used in many earlier rounds, so this evaluation is not an independent validation.
- The hold-out is by session, not by patient.
- Constructed streams are spliced from separate recordings and are not real scans. They reuse recordings, so their
  units are not independent cases. The three seeds share the same cases.
- Native clips carry the public recording label, not expert frame labels. No label was changed.
- The test split and EchoNet-Pediatric were not read.
- Reference intervals serve the error diagnostics only. The diagnostic counts are not scores that any system could be
  expected to reach.

**Order of work.**
- 19092dd committed the task sheet and the fixed configuration before any code. 1427324 committed the feature tests,
  which failed at import, and b47b986 the feature module and the preparation.
- 43b0d74 committed the training entry, its checks and the queue. Its tests were written together with this code, not
  before it (see "Deviations").
- 1926b67 committed the report tests, which failed at import. 14e359c made the reading conditions explicit, and 74e135d
  committed the evaluation code. All of this happened before any development output existed.
- The preparation ran from 02:29 to 03:40 UTC on 2026-10-13. It encoded the training sequences and wrote the batch
  plans and initial states. It started right after b47b986, so 43b0d74, 1926b67, 14e359c, 74e135d and 32bf8af were
  committed while it ran. 14e359c changed only the reading conditions, so the preparation's inputs were not affected.
- An independent implementation review read 74e135d while the preparation ran (`code_review.md`). 32bf8af fixed its
  high-severity finding and one low-severity item.
- The server could not be reached from about 03:10 UTC, and it restarted at about 14:19 UTC. The preparation had
  finished at 03:40 UTC with all its records, so nothing was repeated.
- The first check failed at 14:22 UTC (see "Deviations"). 6a1bd89 fixed the check, and the second check passed at
  20:46 UTC. c4f5590 committed the review, preparation and check records.
- The nine trainings ran from 20:47 to 23:19 UTC, and the queue then froze the nine final weights.
- The single evaluation ran from 23:20 to 23:43 UTC.
- An independent fact-check recomputed the results (see "Fact-check").

## Answers

Values are given for seeds 0, 1 and 2 in that order. A range "a to b" gives the smallest and the largest value over
seeds, and over halves where the sentence says so. A difference is consistent only when all three seeds' 95% intervals
exclude zero strictly on the same side. Frozen Ordered + B3 is the strong reference that the task sheet requires to be
kept.

1. **Does ASRF beat Linear + B3?** No consistent difference was detected in either half.
   - **The primary metric.** ASRF − (Linear + B3) multi-stream segment F1@50 is −0.011, −0.009 and +0.012 on
     selection and +0.003, −0.003 and −0.003 on calibration. Every 95% interval contains zero.
   - **Other metrics.** None of the five other paired metrics shows a consistent difference either. They are native
     macro-F1, label switches per native clip, boundary recall, boundary F1 and cuts at same-view joins.
2. **Does a temporal head beat the strong reference?** No. Neither temporal head shows a consistent segment F1@50
   difference from frozen Ordered + B3 in either half, and all twelve point estimates lie below zero.
   - **ASRF − (frozen Ordered + B3).** The differences are −0.002, −0.013 and −0.014 on selection and −0.002, −0.009
     and −0.010 on calibration. Every interval contains zero.
   - **MS-TCN − (frozen Ordered + B3).** The differences are −0.019, −0.007 and −0.022 on selection and −0.015,
     −0.010 and −0.017 on calibration. The intervals of seed 2 on selection and of seeds 0 and 2 on calibration lie
     below zero, but not those of all three seeds.
   - **(Linear + B3) − (frozen Ordered + B3).** The retrained read-out shows no consistent difference either. The
     differences are +0.009, −0.003 and −0.026 on selection and −0.005, −0.006 and −0.007 on calibration.
   - **ASRF − MS-TCN.** No consistent segment difference was detected (+0.017, −0.006 and +0.008 on selection and
     +0.012, +0.001 and +0.007 on calibration). MS-TCN does not lead in either half.
   - **Label switches.** The only consistent differences among these systems concern label switches inside native
     clips. On selection, ASRF has fewer of them than frozen Ordered + B3 and than MS-TCN.
3. **How does ASRF's output differ?** In every seed and half, ASRF places fewer boundaries than frozen Ordered + B3
   and matches fewer view changes. The boundary recall difference is not consistent by the round's rule.
   - **Boundaries.** Its boundary precision is 0.976 to 0.978 on selection and 0.984 to 0.992 on calibration, against
     0.948 to 0.959 and 0.975 to 0.980 for frozen Ordered + B3. Its boundary recall is 0.945 to 0.948 on selection,
     below the 0.95 floor in every seed, and 0.957 to 0.972 on calibration, against 0.958 to 0.974 and 0.980 to 0.987.
     ASRF − (frozen Ordered + B3) boundary recall is negative in all six seeds and halves, but it is not consistent.
     On calibration, the intervals of seeds 1 and 2 lie below zero, and the interval of seed 0 ends exactly at zero.
   - **Units.** ASRF leaves 37, 32 and 31 of 502 selection units and 15, 17 and 21 of 518 calibration units
     unmatched, against 29, 18 and 15 and 8, 7 and 10 for frozen Ordered + B3. Its main cause is missed cuts (23, 25
     and 23 on selection and 11, 11 and 15 on calibration, against 15, 14 and 11 and 5, 3 and 6). ASRF has no extra
     cut or boundary shift, and it leaves fewer unmatched predicted segments (26, 20 and 18 and 6, 9 and 10, against
     33, 22 and 21 and 11, 10 and 11).
   - **Short segments.** ASRF matches 124 to 128 of 141 selection segments of at most 16 samples and 122 to 127 of 134
     on calibration, against 130 to 136 and 127 to 129 for frozen Ordered + B3. It matches 2 to 3 of 7 A4C/A5C X-Y-X
     middles on selection and 5 to 6 of 7 on calibration, against 3 to 5 and 7 of 7.
   - **Native label switches.** ASRF switches labels 0.014, 0.004 and 0.000 times per native clip on selection and
     0.008 times in each seed on calibration. Frozen Ordered + B3 switches 0.042, 0.039 and 0.046 and 0.025, 0.017 and
     0.013 times.
4. **The trade-off.** The fixed trade-off reading applies to ASRF and to MS-TCN in both halves.
   - **Cases gained.** Counted per seed and summed over seeds, ASRF gets 20 selection cases and 6 calibration cases
     right that frozen Ordered + B3 of the same seed gets wrong. A case is a matched unit or a correct native clip.
     MS-TCN gets 24 and 9.
   - **Cases lost.** Counted the same way, ASRF newly misreads 3 selection and 5 calibration native clips. It misses 13,
     20 and 19 selection units and 9, 11 and 12 calibration units that frozen Ordered + B3 matched. MS-TCN newly
     misreads 7 and 6 native clips and misses 12, 8 and 12 and 10, 7 and 11 such units. Linear + B3 misses 4, 7 and 5
     and 5, 3 and 1.
   - **Native direction.** Both temporal heads fix some A4C clips that frozen Ordered + B3 reads as A5C, but they read
     some A5C clips as A4C, which neither Linear + B3 nor frozen Ordered + B3 does. Over the three seeds, ASRF reads 2
     selection and 2 calibration A5C recordings as A4C, and MS-TCN 3 and 4. ASRF's A5C recall is 0.909 in every seed
     on selection (10 of 11 clips) and 0.889, 0.944 and 1.000 on calibration (of 18 clips). It is 1.000 everywhere for
     Linear + B3 and frozen Ordered + B3. This round does not show why.
   - **Wrong native recordings.** Over the three seeds, ASRF misreads 4 selection recordings and 5 calibration
     recordings, against 4 and 2 for frozen Ordered + B3. The pairs are listed in section 3.
5. **Does ASRF recover the reference's decoding losses?** Only a few. The frozen diagnostic list holds the units that
   frozen Ordered + B3 leaves unmatched while its own argmax majority names the reference class. There are 16, 11 and 9
   on selection and 3, 1 and 2 on calibration. ASRF matches 4, 4 and 2 of them on selection and none on calibration.
   MS-TCN matches 6, 3 and 2 and 0, 0 and 2, and Linear + B3 matches 3, 0 and 1 and 0, 0 and 1. In every seed and
   half, each temporal head misses more units that the reference matched (answer 4) than it recovers from the list.
6. **Where do ASRF's errors arise?** These readings are diagnostics. They use the reference intervals, and they are
   not exclusive.
   - **Edges.** On selection, the unmatched units have 62, 54 and 55 reference edges. ASRF proposed a boundary within
     5 samples at 42, 43 and 43 of them, and its refinement merged 10, 19 and 19 of those. It proposed none at 20, 11
     and 12 edges. On calibration, it proposed 19, 20 and 28 of 26, 28 and 34 edges, merged 6, 6 and 14 and missed 7,
     8 and 6.
   - **Class branch and refinement.** For 14 unmatched units in every seed on selection and 4, 7 and 8 on
     calibration, the class branch names the reference class but the refined output fails the unit. For 12, 11 and 11
     and 8, 5 and 6 units, the source classifier names the reference class but the class branch does not.
   - **Native clips.** ASRF gets 2, 3 and 2 selection clips and 3, 4 and 1 calibration clips wrong. In 1, 1 and 1 and
     2, 3 and 0 of them, the source classifier's majority was right but the class branch's was wrong. In one selection
     clip of seed 1, the class branch was right and the refined output wrong.
   - **What these readings mean.** They locate where the output fails. They do not show why ASRF merges these edges or
     misreads these classes.
7. **Does Linear + B3 make the same fixes?** Only a minority of them. Of the 26 cases that ASRF fixes against frozen
   Ordered + B3, counted per seed and summed over seeds and halves, Linear + B3 of the same seed also fixes 7 (share
   0.27). For MS-TCN it is 10 of 33 (0.30). The linear-explains reading needs a share above 0.5, so it does not apply.
   This does not turn the temporal heads' fixes into a gain, because their losses are larger (answer 4).
8. **Against weaker references.** ASRF beats some weaker references on selection only.
   - **ASRF − paired ASRF.** The paired ASRF of the FACT-ASRF round used the same batch plans and the same initial
     weights in all three seeds, on E1 image features. ASRF − paired ASRF is consistent positive on selection (+0.059,
     +0.060 and +0.068) and not consistent on calibration (+0.007, +0.003 and +0.004). The input replacement changes
     the encoder, its pretraining, its temporal context and the numeric precision together, so no single cause is
     named.
   - **Other references.** ASRF − (E1 + B3) and ASRF − Original FACT are consistent positive on selection only. ASRF −
     historical E1 + ASRF shows no consistent difference in either half.
   - **What this means.** These gains over weaker inputs do not show progress beyond the frozen Ordered + B3 reference,
     which the task sheet requires.
9. **The goal.** No system meets the goal.
   - **Three-seed means.** ASRF has 0.9449 and 0.9747, Linear + B3 0.9476 and 0.9760, MS-TCN 0.9387 and 0.9679, and
     frozen Ordered + B3 0.9544 and 0.9817. The minimums are 0.9444 and 0.9821.
   - **Constraints.** ASRF fails five constraint cells. Its selection boundary recall is below 0.95 in every seed
     (0.948, 0.948 and 0.945), and its calibration native macro-F1 is below 0.980 in seeds 0 and 1 (0.9780 and
     0.9723). MS-TCN fails four cells, and frozen Ordered + B3 one. Linear + B3 fails none, and its B3 setting was
     feasible in every seed, but its calibration mean is below the minimum.
10. **The fixed decision.** Three readings apply.
    - **Trade-off.** Both temporal heads fix some units and native clips against frozen Ordered + B3, but they newly
      misread native clips, and their segment F1@50 difference is not consistent positive. This configuration ends.
      No source-logit residual or gate is added, and no official threshold is retuned.
    - **No gain.** ASRF − (Linear + B3) and ASRF − (frozen Ordered + B3) are consistent positive in neither half. This
      video-feature temporal recipe ends. This does not reject temporal models in general. MS-TCN does not lead in
      either half, so no MS-TCN reading is added.
    - **End.** The round ends at this report.

    The system-gain, selection-only and linear-explains readings do not apply. The test split stays sealed.
11. **Cost.** The round used 4.13 GPU hours of its 8-hour cap before the fact-check.
    - **Preparation.** It took 1.18 hours, mostly to encode 81,710 training windows with each of the three encoders.
    - **Checks.** The failed check took 0.014 hours and the passed one 0.023 hours.
    - **Training.** The nine trainings took 2.53 hours. Linear runs took 0.016 to 0.022 hours, MS-TCN runs 0.17 to
      0.25 hours and ASRF runs 0.48 to 0.74 hours.
    - **Evaluation.** It took 0.38 hours. About 13 minutes of it encoded the 14,687 development windows with each
      encoder.
    - **Fact-check.** It took 0.049 GPU hours, so the round used 4.18 of its 8 hours.

## Terms

- **The input.** Each frozen Ordered encoder gives `model.embed(16 frames, marker)`, a 512-dimensional feature, at every
  stride-5 window centre. Every 10 Hz sample takes the feature of its nearest centre. A constructed stream is assembled
  from its recipe first and then encoded as a whole, so windows cross its joins. A native clip is encoded as a whole
  recording, and training crops are slices of these features.
- **Linear + B3** is a `Linear(512, 5)` read-out at every sample. It starts as a copy of the seed's frozen source
  classifier and is trained with ASRF's class weights. Its probabilities are decoded with B3, whose setting was chosen
  once on calibration from the 40-setting grid. The choice needs native macro-F1 of at least 0.980 and boundary recall
  of at least 0.95, and then takes the highest segment F1@50. The chosen settings are half-window 7 at quantile 0.999
  for seeds 0 and 1 and half-window 9 at quantile 0.995 for seed 2. 23, 6 and 3 settings were feasible.
- **MS-TCN** is the official model with 4 stages, 10 layers and 64 channels on the 512-dimensional input. It is
  trained with cross-entropy plus 0.15 times the smoothing loss. Its output is the last stage's argmax.
- **ASRF** is the official model with 64 features, 10 layers and 4 stages in each branch. It is trained with weighted
  cross-entropy, the Gaussian-similarity smoothing loss at weight 1 and 0.1 times the boundary loss. Its output is the
  official refinement with boundaries at threshold 0.5. The argmax of its class (frame) branch is a diagnostic.
- **Frozen Ordered + B3** is the raw-video context round's own output. It uses that round's saved per-sample
  probabilities and stored B3 settings, and every stored row was reproduced exactly.
- **The references** keep their own inputs and stored settings, and every one was reproduced before use. For E0 + B3,
  E1 + B3, B7 + B3 and historical E1 + ASRF, the check compares the multi-stream segment counts with their stored row
  files. For paired ASRF and Original FACT, it compares every stored metric.
  - E0 + B3, E1 + B3 and B7 + B3 are the image and window classifiers with B3.
  - Historical E1 + ASRF is the ASRF trained in model-development round 2 on E1 features, from its saved outputs.
  - Paired ASRF and Original FACT come from the paired FACT-ASRF round on E1 features.
  - G1 + ASRF supplies the calibration sensitivity target only.
- A **unit** is a reference segment of a constructed stream. A unit is **matched** when a predicted segment of its
  class overlaps it with IoU of at least 0.5 under the one-to-one matching of segment F1@50. An unmatched unit gets the
  first cause that applies, in the order label, missed cut, extra cut, boundary shift.
- **Segments of at most 16 samples** are reference segments of multi streams lasting at most 1.6 seconds. An **X-Y-X
  middle** is the middle segment of three consecutive reference segments whose outer two share a class that differs
  from the middle one. In an A4C/A5C X-Y-X middle, the two classes are A4C and A5C.
- A **label switch** is a change of the output label inside a native clip. **Cuts at same-view joins** give the share
  of joins between two clips of the same view that have a predicted change within 5 samples.
- A **fixed** case of system A against system B is a unit that A matches and B does not, or a native clip that A gets
  right and B gets wrong. A **new** error is the reverse.
- The rule of the **frozen diagnostic list** was fixed before any output of this round. The list depends only on the
  source round's saved outputs, and the evaluation computed it. It holds, per seed and half, the units that frozen
  Ordered + B3 leaves unmatched while its own argmax majority over the unit names the reference class. It was never
  used for sampling, training or a setting.
- **ASRF readings** of an unmatched unit use the reference edges. An edge is proposed when the official boundary peak
  rule finds a peak within 5 samples of it. A proposed edge is merged when the refined output has no label change
  within 5 samples of it.
- The **paired bootstrap** resamples native clips by session and streams by stream, with 2,000 draws, seed 20261002,
  and the same draws for both systems.

## 1. Systems

Each cell gives the selection half, then the calibration half.

| System | Seed | Segment F1@50 | Boundary recall | Boundary precision | Native macro-F1 | Label switches per native clip | Cuts at same-view joins | A4C read as A5C (clips) | A5C read as A4C (clips) |
|---|---|---|---|---|---|---|---|---|---|
| Linear + B3 | 0 | 0.947 / 0.977 | 0.955 / 0.975 | 0.953 / 0.975 | 0.9786 / 0.9861 | 0.046 / 0.013 | 0.026 / 0.020 | 2 / 2 | 0 / 0 |
| Linear + B3 | 1 | 0.957 / 0.978 | 0.953 / 0.980 | 0.971 / 0.973 | 0.9786 / 0.9861 | 0.039 / 0.017 | 0.009 / 0.015 | 2 / 2 | 0 / 0 |
| Linear + B3 | 2 | 0.939 / 0.973 | 0.971 / 0.972 | 0.912 / 0.953 | 0.9786 / 0.9861 | 0.085 / 0.042 | 0.030 / 0.030 | 2 / 2 | 0 / 0 |
| MS-TCN | 0 | 0.920 / 0.967 | 0.966 / 0.962 | 0.918 / 0.962 | 0.9626 / 0.9780 | 0.074 / 0.038 | 0.035 / 0.025 | 0 / 1 | 3 / 2 |
| MS-TCN | 1 | 0.953 / 0.974 | 0.963 / 0.977 | 0.951 / 0.975 | 0.9771 / 0.9780 | 0.042 / 0.051 | 0.017 / 0.025 | 1 / 1 | 1 / 2 |
| MS-TCN | 2 | 0.943 / 0.963 | 0.958 / 0.970 | 0.938 / 0.953 | 0.9753 / 0.9780 | 0.056 / 0.055 | 0.013 / 0.035 | 0 / 1 | 2 / 2 |
| ASRF | 0 | 0.937 / 0.980 | 0.948 / 0.970 | 0.976 / 0.992 | 0.9869 / 0.9780 | 0.014 / 0.008 | 0.035 / 0.015 | 0 / 1 | 1 / 2 |
| ASRF | 1 | 0.948 / 0.975 | 0.948 / 0.972 | 0.978 / 0.992 | 0.9771 / 0.9723 | 0.004 / 0.008 | 0.030 / 0.010 | 1 / 3 | 1 / 1 |
| ASRF | 2 | 0.951 / 0.970 | 0.945 / 0.957 | 0.978 / 0.984 | 0.9869 / 0.9929 | 0.000 / 0.008 | 0.030 / 0.015 | 0 / 1 | 1 / 0 |
| frozen Ordered + B3 | 0 | 0.938 / 0.982 | 0.958 / 0.982 | 0.948 / 0.975 | 0.9786 / 0.9929 | 0.042 / 0.025 | 0.017 / 0.025 | 2 / 1 | 0 / 0 |
| frozen Ordered + B3 | 1 | 0.960 / 0.984 | 0.969 / 0.987 | 0.959 / 0.980 | 0.9786 / 0.9861 | 0.039 / 0.017 | 0.004 / 0.025 | 2 / 2 | 0 / 0 |
| frozen Ordered + B3 | 2 | 0.964 / 0.980 | 0.974 / 0.980 | 0.959 / 0.977 | 0.9701 / 0.9861 | 0.046 / 0.013 | 0.004 / 0.025 | 3 / 2 | 0 / 0 |

Direct outputs without the systems' own decoding (diagnostic).

| System | Seed | Segment F1@50 | Boundary recall | Boundary precision | Native macro-F1 | Label switches per native clip | Cuts at same-view joins | A4C read as A5C (clips) | A5C read as A4C (clips) |
|---|---|---|---|---|---|---|---|---|---|
| Linear argmax | 0 | 0.901 / 0.941 | 0.974 / 0.980 | 0.834 / 0.909 | 0.9701 / 0.9861 | 0.134 / 0.118 | 0.039 / 0.040 | 3 / 2 | 0 / 0 |
| Linear argmax | 1 | 0.907 / 0.958 | 0.979 / 0.985 | 0.858 / 0.938 | 0.9701 / 0.9732 | 0.155 / 0.080 | 0.035 / 0.040 | 3 / 4 | 0 / 0 |
| Linear argmax | 2 | 0.908 / 0.950 | 0.982 / 0.975 | 0.841 / 0.917 | 0.9786 / 0.9796 | 0.162 / 0.131 | 0.035 / 0.040 | 2 / 3 | 0 / 0 |
| ASRF frame branch | 0 | 0.919 / 0.969 | 0.953 / 0.962 | 0.919 / 0.967 | 0.9869 / 0.9780 | 0.063 / 0.038 | 0.030 / 0.025 | 0 / 1 | 1 / 2 |
| ASRF frame branch | 1 | 0.934 / 0.974 | 0.950 / 0.977 | 0.914 / 0.975 | 0.9869 / 0.9647 | 0.049 / 0.059 | 0.022 / 0.020 | 0 / 3 | 1 / 2 |
| ASRF frame branch | 2 | 0.937 / 0.966 | 0.958 / 0.962 | 0.929 / 0.965 | 0.9869 / 0.9929 | 0.074 / 0.051 | 0.026 / 0.020 | 0 / 1 | 1 / 0 |
| frozen Ordered argmax | 0 | 0.893 / 0.961 | 0.979 / 0.987 | 0.846 / 0.942 | 0.9701 / 0.9861 | 0.095 / 0.093 | 0.043 / 0.035 | 3 / 2 | 0 / 0 |
| frozen Ordered argmax | 1 | 0.925 / 0.961 | 0.984 / 0.990 | 0.872 / 0.945 | 0.9877 / 0.9796 | 0.113 / 0.101 | 0.030 / 0.040 | 1 / 3 | 0 / 0 |
| frozen Ordered argmax | 2 | 0.926 / 0.960 | 0.984 / 0.982 | 0.874 / 0.940 | 0.9701 / 0.9861 | 0.116 / 0.097 | 0.026 / 0.035 | 3 / 2 | 0 / 0 |

References, each with its own inputs and stored settings.

| System | Seed | Segment F1@50 | Boundary recall | Boundary precision | Native macro-F1 | Label switches per native clip | Cuts at same-view joins | A4C read as A5C (clips) | A5C read as A4C (clips) |
|---|---|---|---|---|---|---|---|---|---|
| E0 + B3 | - | 0.934 / 0.934 | 0.961 / 0.955 | 0.995 / 0.969 | 0.9650 / 0.9660 | 0.014 / 0.034 | 0.004 / 0.020 | 0 / 0 | 3 / 4 |
| E1 + B3 | - | 0.894 / 0.959 | 0.955 / 0.977 | 0.943 / 0.951 | 0.9780 / 0.9856 | 0.046 / 0.068 | 0.039 / 0.020 | 0 / 1 | 1 / 1 |
| B7 + B3 | - | 0.880 / 0.972 | 0.955 / 0.975 | 0.890 / 0.963 | 0.9566 / 0.9927 | 0.063 / 0.042 | 0.057 / 0.035 | 1 / 0 | 2 / 1 |
| historical E1 + ASRF | 0 | 0.910 / 0.976 | 0.942 / 0.977 | 0.952 / 0.980 | 0.9477 / 0.9929 | 0.046 / 0.038 | 0.030 / 0.015 | 2 / 1 | 2 / 0 |
| historical E1 + ASRF | 1 | 0.909 / 0.964 | 0.950 / 0.972 | 0.936 / 0.970 | 0.9664 / 0.9753 | 0.007 / 0.030 | 0.057 / 0.030 | 0 / 1 | 2 / 2 |
| historical E1 + ASRF | 2 | 0.907 / 0.964 | 0.953 / 0.977 | 0.933 / 0.970 | 0.9664 / 0.9753 | 0.021 / 0.046 | 0.057 / 0.035 | 0 / 1 | 2 / 2 |
| paired ASRF | 0 | 0.878 / 0.972 | 0.935 / 0.987 | 0.927 / 0.975 | 0.9304 / 0.9856 | 0.011 / 0.004 | 0.065 / 0.035 | 1 / 1 | 4 / 1 |
| paired ASRF | 1 | 0.888 / 0.972 | 0.940 / 0.985 | 0.937 / 0.978 | 0.9537 / 0.9856 | 0.007 / 0.004 | 0.048 / 0.035 | 0 / 1 | 3 / 1 |
| paired ASRF | 2 | 0.882 / 0.966 | 0.942 / 0.982 | 0.942 / 0.980 | 0.9537 / 0.9856 | 0.011 / 0.004 | 0.065 / 0.030 | 0 / 1 | 3 / 1 |
| Original FACT | 0 | 0.906 / 0.967 | 0.948 / 0.982 | 0.926 / 0.968 | 0.9886 / 0.9856 | 0.000 / 0.034 | 0.052 / 0.025 | 0 / 1 | 0 / 1 |
| Original FACT | 1 | 0.898 / 0.961 | 0.950 / 0.975 | 0.928 / 0.968 | 0.9780 / 0.9856 | 0.000 / 0.030 | 0.043 / 0.030 | 0 / 1 | 1 / 1 |
| Original FACT | 2 | 0.911 / 0.965 | 0.940 / 0.972 | 0.925 / 0.972 | 0.9664 / 0.9929 | 0.000 / 0.034 | 0.035 / 0.025 | 0 / 1 | 2 / 0 |

## 2. Comparisons

Paired differences with 95% intervals. Streams reuse recordings, so the intervals are approximate. "none" means that
no consistent difference was detected, which does not mean that the systems are equal.

| Comparison | Half | Segment F1@50 (seeds 0, 1, 2) | Verdict | Boundary recall (seeds 0, 1, 2) | Verdict | Other consistent differences |
|---|---|---|---|---|---|---|
| ASRF − (Linear + B3) | selection | −0.0107 [−0.0273, +0.0055]; −0.0093 [−0.0340, +0.0128]; +0.0118 [−0.0067, +0.0303] | none | −0.0079 [−0.0262, +0.0107]; −0.0052 [−0.0252, +0.0131]; −0.0262 [−0.0522, −0.0026] | none | none |
| ASRF − (Linear + B3) | calibration | +0.0027 [−0.0088, +0.0143]; −0.0032 [−0.0179, +0.0102]; −0.0034 [−0.0212, +0.0112] | none | −0.0050 [−0.0209, +0.0100]; −0.0075 [−0.0233, +0.0052]; −0.0151 [−0.0374, +0.0027] | none | none |
| ASRF − (frozen Ordered + B3) | selection | −0.0019 [−0.0221, +0.0181]; −0.0127 [−0.0400, +0.0104]; −0.0138 [−0.0328, +0.0037] | none | −0.0105 [−0.0310, +0.0103]; −0.0209 [−0.0433, +0.0000]; −0.0288 [−0.0548, −0.0028] | none | native label switches consistent negative |
| ASRF − (frozen Ordered + B3) | calibration | −0.0022 [−0.0132, +0.0100]; −0.0089 [−0.0225, +0.0037]; −0.0100 [−0.0255, +0.0028] | none | −0.0126 [−0.0282, +0.0000]; −0.0151 [−0.0299, −0.0026]; −0.0226 [−0.0426, −0.0073] | none | none |
| MS-TCN − (frozen Ordered + B3) | selection | −0.0186 [−0.0417, +0.0037]; −0.0069 [−0.0242, +0.0092]; −0.0217 [−0.0420, −0.0032] | none | +0.0079 [−0.0107, +0.0260]; −0.0052 [−0.0227, +0.0109]; −0.0157 [−0.0349, +0.0026] | none | none |
| MS-TCN − (frozen Ordered + B3) | calibration | −0.0145 [−0.0286, −0.0029]; −0.0097 [−0.0230, +0.0026]; −0.0171 [−0.0345, −0.0008] | none | −0.0201 [−0.0375, −0.0050]; −0.0101 [−0.0207, −0.0025]; −0.0101 [−0.0298, +0.0074] | none | none |
| (Linear + B3) − (frozen Ordered + B3) | selection | +0.0088 [−0.0058, +0.0231]; −0.0034 [−0.0128, +0.0052]; −0.0256 [−0.0446, −0.0093] | none | −0.0026 [−0.0184, +0.0125]; −0.0157 [−0.0287, −0.0052]; −0.0026 [−0.0082, +0.0000] | none | none |
| (Linear + B3) − (frozen Ordered + B3) | calibration | −0.0049 [−0.0121, +0.0011]; −0.0058 [−0.0118, −0.0001]; −0.0066 [−0.0144, +0.0000] | none | −0.0075 [−0.0184, +0.0026]; −0.0075 [−0.0170, +0.0000]; −0.0075 [−0.0166, +0.0000] | none | none |
| ASRF − MS-TCN | selection | +0.0166 [−0.0031, +0.0391]; −0.0058 [−0.0250, +0.0132]; +0.0079 [−0.0058, +0.0220] | none | −0.0183 [−0.0371, +0.0000]; −0.0157 [−0.0311, −0.0026]; −0.0131 [−0.0288, +0.0000] | none | native label switches consistent negative |
| ASRF − MS-TCN | calibration | +0.0124 [−0.0011, +0.0272]; +0.0007 [−0.0099, +0.0106]; +0.0071 [−0.0038, +0.0192] | none | +0.0075 [−0.0072, +0.0227]; −0.0050 [−0.0216, +0.0095]; −0.0126 [−0.0278, +0.0000] | none | none |
| MS-TCN − (Linear + B3) | selection | −0.0273 [−0.0467, −0.0106]; −0.0035 [−0.0208, +0.0135]; +0.0039 [−0.0147, +0.0226] | none | +0.0105 [−0.0028, +0.0262]; +0.0105 [−0.0075, +0.0291]; −0.0131 [−0.0317, +0.0028] | none | none |
| MS-TCN − (Linear + B3) | calibration | −0.0097 [−0.0242, +0.0028]; −0.0039 [−0.0185, +0.0093]; −0.0106 [−0.0277, +0.0047] | none | −0.0126 [−0.0282, +0.0000]; −0.0025 [−0.0119, +0.0051]; −0.0025 [−0.0231, +0.0157] | none | none |
| ASRF − paired ASRF | selection | +0.0587 [+0.0285, +0.0901]; +0.0600 [+0.0271, +0.0960]; +0.0681 [+0.0366, +0.1003] | consistent positive | +0.0131 [−0.0179, +0.0462]; +0.0079 [−0.0183, +0.0346]; +0.0026 [−0.0267, +0.0336] | none | none |
| ASRF − paired ASRF | calibration | +0.0074 [−0.0064, +0.0217]; +0.0026 [−0.0105, +0.0153]; +0.0035 [−0.0144, +0.0199] | none | −0.0176 [−0.0367, −0.0024]; −0.0126 [−0.0308, +0.0049]; −0.0251 [−0.0492, −0.0048] | none | none |
| ASRF − (E1 + B3) | selection | +0.0426 [+0.0109, +0.0747]; +0.0536 [+0.0165, +0.0898]; +0.0566 [+0.0246, +0.0891] | consistent positive | −0.0079 [−0.0319, +0.0183]; −0.0079 [−0.0350, +0.0207]; −0.0105 [−0.0399, +0.0192] | none | none |
| ASRF − (E1 + B3) | calibration | +0.0206 [+0.0034, +0.0389]; +0.0158 [−0.0021, +0.0329]; +0.0108 [−0.0094, +0.0282] | none | −0.0075 [−0.0293, +0.0128]; −0.0050 [−0.0235, +0.0124]; −0.0201 [−0.0445, +0.0026] | none | native label switches consistent negative |
| ASRF − Original FACT | selection | +0.0303 [+0.0028, +0.0585]; +0.0493 [+0.0168, +0.0827]; +0.0397 [+0.0089, +0.0723] | consistent positive | +0.0000 [−0.0254, +0.0269]; −0.0026 [−0.0308, +0.0286]; +0.0052 [−0.0251, +0.0380] | none | boundary F1 consistent positive |
| ASRF − Original FACT | calibration | +0.0122 [−0.0070, +0.0325]; +0.0142 [−0.0031, +0.0331]; +0.0045 [−0.0135, +0.0208] | none | −0.0126 [−0.0308, +0.0048]; −0.0025 [−0.0179, +0.0122]; −0.0151 [−0.0371, +0.0053] | none | none |
| ASRF − historical E1 + ASRF | selection | +0.0266 [−0.0009, +0.0546]; +0.0387 [+0.0048, +0.0740]; +0.0434 [+0.0135, +0.0752] | none | +0.0052 [−0.0231, +0.0364]; −0.0026 [−0.0336, +0.0286]; −0.0079 [−0.0359, +0.0215] | none | none |
| ASRF − historical E1 + ASRF | calibration | +0.0037 [−0.0097, +0.0179]; +0.0104 [−0.0031, +0.0229]; +0.0054 [−0.0136, +0.0250] | none | −0.0075 [−0.0267, +0.0098]; +0.0000 [−0.0147, +0.0136]; −0.0201 [−0.0436, +0.0000] | none | none |

## 3. Native clips

Selection has 284 native clips (105 PLAX, 68 PSAX, 87 A4C, 11 A5C and 13 SC4C). Calibration has 237 (80, 71, 61, 18
and 7). Each cell gives selection, then calibration.

| System | Seed | Wrong native clips | PLAX recall | PSAX recall | A4C recall | A5C recall | SC4C recall |
|---|---|---|---|---|---|---|---|
| Linear + B3 | 0 | 3 / 2 | 0.990 / 1.000 | 1.000 / 1.000 | 0.977 / 0.967 | 1.000 / 1.000 | 1.000 / 1.000 |
| Linear + B3 | 1 | 3 / 2 | 0.990 / 1.000 | 1.000 / 1.000 | 0.977 / 0.967 | 1.000 / 1.000 | 1.000 / 1.000 |
| Linear + B3 | 2 | 3 / 2 | 0.990 / 1.000 | 1.000 / 1.000 | 0.977 / 0.967 | 1.000 / 1.000 | 1.000 / 1.000 |
| MS-TCN | 0 | 4 / 3 | 0.990 / 1.000 | 1.000 / 1.000 | 1.000 / 0.984 | 0.727 / 0.889 | 1.000 / 1.000 |
| MS-TCN | 1 | 3 / 3 | 0.990 / 1.000 | 1.000 / 1.000 | 0.989 / 0.984 | 0.909 / 0.889 | 1.000 / 1.000 |
| MS-TCN | 2 | 3 / 3 | 0.990 / 1.000 | 1.000 / 1.000 | 1.000 / 0.984 | 0.818 / 0.889 | 1.000 / 1.000 |
| ASRF | 0 | 2 / 3 | 0.990 / 1.000 | 1.000 / 1.000 | 1.000 / 0.984 | 0.909 / 0.889 | 1.000 / 1.000 |
| ASRF | 1 | 3 / 4 | 0.990 / 1.000 | 1.000 / 1.000 | 0.989 / 0.951 | 0.909 / 0.944 | 1.000 / 1.000 |
| ASRF | 2 | 2 / 1 | 0.990 / 1.000 | 1.000 / 1.000 | 1.000 / 0.984 | 0.909 / 1.000 | 1.000 / 1.000 |
| frozen Ordered + B3 | 0 | 3 / 1 | 0.990 / 1.000 | 1.000 / 1.000 | 0.977 / 0.984 | 1.000 / 1.000 | 1.000 / 1.000 |
| frozen Ordered + B3 | 1 | 3 / 2 | 0.990 / 1.000 | 1.000 / 1.000 | 0.977 / 0.967 | 1.000 / 1.000 | 1.000 / 1.000 |
| frozen Ordered + B3 | 2 | 4 / 2 | 0.990 / 1.000 | 1.000 / 1.000 | 0.966 / 0.967 | 1.000 / 1.000 | 1.000 / 1.000 |

Fixed and new native errors against frozen Ordered + B3 and against Linear + B3, per seed (s0; s1; s2).

| Comparison | Half | A4C read as A5C, fixed/new | A5C read as A4C, fixed/new | All native clips, fixed/new |
|---|---|---|---|---|
| ASRF − (frozen Ordered + B3) | selection | 2/0; 1/0; 3/0 | 0/1; 0/1; 0/1 | 2/1; 1/1; 3/1 |
| ASRF − (frozen Ordered + B3) | calibration | 0/0; 1/2; 1/0 | 0/2; 0/1; 0/0 | 0/2; 1/3; 1/0 |
| MS-TCN − (frozen Ordered + B3) | selection | 2/0; 2/1; 3/0 | 0/3; 0/1; 0/2 | 2/3; 2/2; 3/2 |
| MS-TCN − (frozen Ordered + B3) | calibration | 0/0; 1/0; 1/0 | 0/2; 0/2; 0/2 | 0/2; 1/2; 1/2 |
| (Linear + B3) − (frozen Ordered + B3) | selection | 0/0; 0/0; 1/0 | 0/0; 0/0; 0/0 | 0/0; 0/0; 1/0 |
| (Linear + B3) − (frozen Ordered + B3) | calibration | 0/1; 0/0; 0/0 | 0/0; 0/0; 0/0 | 0/1; 0/0; 0/0 |
| ASRF − (Linear + B3) | selection | 2/0; 1/0; 2/0 | 0/1; 0/1; 0/1 | 2/1; 1/1; 2/1 |
| ASRF − (Linear + B3) | calibration | 1/0; 1/2; 1/0 | 0/2; 0/1; 0/0 | 1/2; 1/3; 1/0 |

The union over the three seeds, by distinct recordings. A seed is not an extra case.

| System | Half | Wrong recordings | By class pair: recordings (sessions) |
|---|---|---|---|
| Linear + B3 | selection | 3 | A4C read as A5C 2 (2); PLAX read as PSAX 1 (1) |
| Linear + B3 | calibration | 3 | A4C read as A5C 3 (3) |
| MS-TCN | selection | 5 | A4C read as A5C 1 (1); A5C read as A4C 3 (3); PLAX read as PSAX 1 (1) |
| MS-TCN | calibration | 5 | A4C read as A5C 1 (1); A5C read as A4C 4 (4) |
| ASRF | selection | 4 | A4C read as A5C 1 (1); A5C read as A4C 2 (2); PLAX read as PSAX 1 (1) |
| ASRF | calibration | 5 | A4C read as A5C 3 (2); A5C read as A4C 2 (2) |
| frozen Ordered + B3 | selection | 4 | A4C read as A5C 3 (3); PLAX read as PSAX 1 (1) |
| frozen Ordered + B3 | calibration | 2 | A4C read as A5C 2 (2) |

## 4. Reference units of multi streams

Unmatched units by cause, with each system's own decoding. Selection has 502 units and calibration 518. Each cell
gives selection, then calibration. `summary.json` adds the distinct recordings of every cause.

| System | Seed | Unmatched units | Label | Missed cut | Extra cut | Boundary shift | Unmatched predicted segments |
|---|---|---|---|---|---|---|---|
| Linear + B3 | 0 | 26 / 12 | 9 / 5 | 16 / 7 | 1 / 0 | 0 / 0 | 27 / 12 |
| Linear + B3 | 1 | 25 / 10 | 4 / 5 | 20 / 5 | 1 / 0 | 0 / 0 | 18 / 13 |
| Linear + B3 | 2 | 19 / 10 | 5 / 5 | 11 / 5 | 3 / 0 | 0 / 0 | 44 / 18 |
| MS-TCN | 0 | 31 / 17 | 11 / 7 | 15 / 10 | 5 / 0 | 0 / 0 | 51 / 17 |
| MS-TCN | 1 | 21 / 13 | 3 / 5 | 16 / 5 | 1 / 1 | 1 / 2 | 26 / 14 |
| MS-TCN | 2 | 25 / 16 | 6 / 6 | 17 / 8 | 2 / 1 | 0 / 1 | 33 / 23 |
| ASRF | 0 | 37 / 15 | 14 / 4 | 23 / 11 | 0 / 0 | 0 / 0 | 26 / 6 |
| ASRF | 1 | 32 / 17 | 7 / 6 | 25 / 11 | 0 / 0 | 0 / 0 | 20 / 9 |
| ASRF | 2 | 31 / 21 | 8 / 6 | 23 / 15 | 0 / 0 | 0 / 0 | 18 / 10 |
| frozen Ordered + B3 | 0 | 29 / 8 | 11 / 3 | 15 / 5 | 2 / 0 | 1 / 0 | 33 / 11 |
| frozen Ordered + B3 | 1 | 18 / 7 | 3 / 4 | 14 / 3 | 1 / 0 | 0 / 0 | 22 / 10 |
| frozen Ordered + B3 | 2 | 15 / 10 | 4 / 4 | 11 / 6 | 0 / 0 | 0 / 0 | 21 / 11 |

## 5. Short segments and X-Y-X middles

Matched reference segments of the multi streams, out of all such segments. Each cell gives selection, then
calibration.

| System | Seed | Segments of at most 16 samples | A4C or A5C among them | X-Y-X middles | A4C/A5C X-Y-X middles |
|---|---|---|---|---|---|
| Linear + B3 | 0 | 131/141 / 126/134 | 55/65 / 63/67 | 64/69 / 80/80 | 3/7 / 7/7 |
| Linear + B3 | 1 | 129/141 / 127/134 | 53/65 / 63/67 | 64/69 / 79/80 | 3/7 / 7/7 |
| Linear + B3 | 2 | 135/141 / 126/134 | 59/65 / 64/67 | 66/69 / 79/80 | 5/7 / 7/7 |
| MS-TCN | 0 | 130/141 / 124/134 | 56/65 / 61/67 | 64/69 / 78/80 | 3/7 / 6/7 |
| MS-TCN | 1 | 131/141 / 126/134 | 55/65 / 62/67 | 63/69 / 78/80 | 3/7 / 6/7 |
| MS-TCN | 2 | 128/141 / 126/134 | 53/65 / 63/67 | 63/69 / 77/80 | 2/7 / 5/7 |
| ASRF | 0 | 128/141 / 124/134 | 52/65 / 62/67 | 63/69 / 77/80 | 3/7 / 5/7 |
| ASRF | 1 | 126/141 / 127/134 | 52/65 / 63/67 | 63/69 / 78/80 | 3/7 / 6/7 |
| ASRF | 2 | 124/141 / 122/134 | 51/65 / 60/67 | 62/69 / 78/80 | 2/7 / 5/7 |
| ASRF frame branch | 0 | 129/141 / 123/134 | 53/65 / 61/67 | 63/69 / 77/80 | 3/7 / 5/7 |
| ASRF frame branch | 1 | 127/141 / 129/134 | 52/65 / 65/67 | 63/69 / 78/80 | 3/7 / 6/7 |
| ASRF frame branch | 2 | 128/141 / 123/134 | 53/65 / 61/67 | 62/69 / 78/80 | 1/7 / 5/7 |
| frozen Ordered + B3 | 0 | 130/141 / 128/134 | 54/65 / 66/67 | 66/69 / 80/80 | 5/7 / 7/7 |
| frozen Ordered + B3 | 1 | 134/141 / 129/134 | 58/65 / 65/67 | 64/69 / 80/80 | 3/7 / 7/7 |
| frozen Ordered + B3 | 2 | 136/141 / 127/134 | 60/65 / 65/67 | 66/69 / 80/80 | 5/7 / 7/7 |

## 6. The frozen diagnostic list and the overlap with Linear + B3

Units of the frozen list that each new system matches, and units that frozen Ordered + B3 matched but the new system
misses. Each cell gives selection, then calibration. `summary.json` adds the distinct recordings and sessions.

| Seed | Listed units | Fixed by Linear + B3 | Fixed by MS-TCN | Fixed by ASRF | New misses, Linear + B3 | New misses, MS-TCN | New misses, ASRF |
|---|---|---|---|---|---|---|---|
| 0 | 16 / 3 | 3 / 0 | 6 / 0 | 4 / 0 | 4 / 5 | 12 / 10 | 13 / 9 |
| 1 | 11 / 1 | 0 / 0 | 3 / 0 | 4 / 0 | 7 / 3 | 8 / 7 | 20 / 11 |
| 2 | 9 / 2 | 1 / 1 | 2 / 2 | 2 / 0 | 5 / 1 | 12 / 11 | 19 / 12 |

Cases each temporal head fixes against frozen Ordered + B3, and how many of them Linear + B3 of the same seed also
fixes. A case is a unit or a native clip.

| Seed | Half | ASRF fixes (also by Linear + B3) | MS-TCN fixes (also by Linear + B3) |
|---|---|---|---|
| 0 | selection | 7 (4) | 12 (6) |
| 0 | calibration | 2 (1) | 1 (1) |
| 1 | selection | 7 (0) | 7 (0) |
| 1 | calibration | 2 (0) | 2 (0) |
| 2 | selection | 6 (2) | 5 (2) |
| 2 | calibration | 2 (0) | 6 (1) |
| pooled | both | 26 (7), share 0.27 | 33 (10), share 0.30 |

## 7. ASRF readings

Readings of ASRF's unmatched units and wrong native clips. The readings are not exclusive. Unit cells give units, then
the distinct recordings in brackets.

| Seed | Half | Unmatched units (recordings, sessions) | Their reference edges | Source right, class branch wrong | Class branch right, refinement wrong | Edge not proposed | Proposed but merged | Wrong native clips | Native: source right, class branch wrong | Native: class branch right, refinement wrong |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | selection | 37 (26, 16) | 62: 42 proposed, 10 of them merged, 20 not proposed | 12 (9) | 14 (17) | 18 (15) | 10 (11) | 2 | 1 | 0 |
| 0 | calibration | 15 (17, 13) | 26: 19 proposed, 6 of them merged, 7 not proposed | 8 (12) | 4 (4) | 7 (9) | 5 (5) | 3 | 2 | 0 |
| 1 | selection | 32 (27, 14) | 54: 43 proposed, 19 of them merged, 11 not proposed | 11 (10) | 14 (16) | 10 (11) | 18 (17) | 3 | 1 | 1 |
| 1 | calibration | 17 (19, 16) | 28: 20 proposed, 6 of them merged, 8 not proposed | 5 (9) | 7 (7) | 8 (10) | 5 (6) | 4 | 3 | 0 |
| 2 | selection | 31 (25, 13) | 55: 43 proposed, 19 of them merged, 12 not proposed | 11 (8) | 14 (16) | 10 (11) | 17 (14) | 2 | 1 | 0 |
| 2 | calibration | 21 (22, 15) | 34: 28 proposed, 14 of them merged, 6 not proposed | 6 (7) | 8 (11) | 6 (7) | 12 (15) | 1 | 0 | 0 |

## 8. Goal

Three-seed means against the minimums of 0.9444 and 0.9821. The calibration sensitivity target, 0.9839, is the
G1 + ASRF calibration mean plus 0.01. A constraint cell fails when boundary recall is below 0.95 or native macro-F1 is
below its floor (0.973 on selection and 0.980 on calibration).

| System | Three-seed mean | Margin to the minimum | Calibration sensitivity margin | Constraint failures | Failing cells | Goal met |
|---|---|---|---|---|---|---|
| Linear + B3 | 0.9476 / 0.9760 | +0.0032 / −0.0062 | −0.0079 | 0 | none | no |
| MS-TCN | 0.9387 / 0.9679 | −0.0057 / −0.0142 | −0.0160 | 4 | seed 0 selection (native 0.9626); calibration in all seeds (native 0.9780) | no |
| ASRF | 0.9449 / 0.9747 | +0.0005 / −0.0075 | −0.0092 | 5 | selection in all seeds (recall 0.948, 0.948 and 0.945); calibration in seeds 0 and 1 (native 0.9780 and 0.9723) | no |
| frozen Ordered + B3 | 0.9544 / 0.9817 | +0.0100 / −0.0004 | −0.0022 | 1 | seed 2 selection (native 0.9701) | no |

## 9. Training

`training_curves.png` shows the training loss and the fit on 16 training sequences per epoch. The final weights after
18,750 updates were kept, and nothing was chosen from these curves.

- **Inputs and batches.** Every epoch's batches of every run matched the seed's plan, which equals the paired ASRF
  plan in all 18,750 updates. The check also compared the three heads' collated batches bit for bit in the first and
  the last epoch (750 updates per seed). The class weights and the boundary weight equal the paired round's.
- **Initial states.** Each ASRF initial state equals the paired round's ASRF state of the seed, tensor for tensor, and
  was reused. Linear starts as a copy of the source classifier. MS-TCN was built after seeding.
- **Parameters.** Linear has 2,565 parameters, MS-TCN 695,764 and ASRF 1,191,768.
- **Losses.** The Linear loss starts near 0.19 and stays between 0.15 and 0.17 from the third epoch. The ASRF loss
  falls to 0.074 to 0.078. The MS-TCN total loss is lowest in epochs 17, 28 and 16, counting from 1, and then rises to
  0.66, 0.60 and 0.99. From that lowest point to the end, its cross-entropy falls from 0.34, 0.21 and 0.36 to 0.19,
  0.15 and 0.19. Its weighted smoothing term rises from 0.09, 0.09 and 0.08 to 0.47, 0.44 and 0.80, so the rise of the
  total comes from this term.
- **Fit monitor.** The fit is the mean, over 8 training clips and 8 training streams, of each sequence's share of
  correct samples. Its minimum is 0 in every run, and in all nine runs it comes from the same 27-sample training clip
  labelled PLAX. Every seed's frozen source classifier reads that clip entirely as A4C or A5C before any training, and
  every head keeps a wrong reading. With the source classifiers, the other 15 sequences scored 0.77 to 1.00 at the
  start. This is a training-set observation, and it changed nothing.
- **Determinism.** Training ran in fp32 with TF32 off and deterministic cuDNN. The official 2-d cross-entropy of ASRF
  has no deterministic CUDA kernel, so ASRF runs are not bit-reproducible on the GPU. In the check, a 6-update rerun
  and a 3 + 3 resume differed from the original by at most 2.6e-7 in loss and parameters.

## What this round does not show

- **No new model.** The heads are existing methods on frozen features. ASRF − MS-TCN is not an ablation of a boundary
  loss.
- **No system gain.** Neither temporal head differs consistently from Linear + B3 or from frozen Ordered + B3 in
  segment F1@50, and all their point estimates against frozen Ordered + B3 are negative. No system meets the goal.
- **Not a verdict on temporal models in general.** Only this recipe ends. It used frozen Ordered features at stride 5,
  these two heads with their official losses and settings, these batches and this budget.
- **Not why ASRF misses cuts or why A5C clips are read as A4C.** The readings locate the failures. They do not show
  why the refinement merges proposed edges, why some edges get no boundary peak, or why the temporal heads read some
  A5C clips as A4C.
- **Not how the training features matter.** The encoders were trained on the same training recordings, so the heads
  learned from features of recordings that the encoders had already fitted. This round does not test whether that
  matters.
- **No cardiac or probe explanation.** Score changes over a clip are not attributed to cardiac phase or probe motion.
- **Not clinical scanning.** Constructed streams and public recording labels are not real continuous scans or expert
  frame labels, and the development halves are not an independent validation.

## Deviations and launch record

1. **Training tests written with the code.** The task sheet asks for tests before each implementation. The feature
   and report tests were committed first and failed at import (1427324 and 1926b67). The training tests were written
   together with the training entry and committed with it in 43b0d74.
2. **The review's high-severity finding.** The preparation saved the training features before it checked their
   parity with the source models. 32bf8af moved the check before the saving. The preparation that produced the
   features still ran the older order. Its parity check passed (largest logit difference 7.3e-6, every argmax equal),
   so the saved features are the checked ones. The feature-building code did not change, so their identity held.
3. **The first check failed.** On the GPU, replacing the padding labels changed the ASRF loss by 2.4e-7. The
   normalised inputs were identical, and the official 2-d cross-entropy is recorded as having no deterministic CUDA
   kernel. 6a1bd89 runs this exact test on the CPU, where the difference is 0, and records the GPU difference. The
   second check passed. Both checks are in the ledger.
4. **The server restart.** The server could not be reached from about 03:10 UTC, and it restarted at about 14:19 UTC.
   The preparation still finished at 03:40 UTC and wrote all its records, so it was not affected. Nothing was
   repeated.
5. **Run times.** The check projected 0.75 hours per ASRF run, and the runs took 0.48 to 0.74 hours. The records do
   not show why the ASRF run times differ. The queue projected the budget again before every run, and every projection
   stayed under the cap.
6. **A figure label.** After the evaluation, the schematic `feature_flow.png` was found to call its three-fragment
   stream "two fragments". The label was corrected in the report code, and both figures were redrawn from the same run
   records on the CPU. No number changed.

## Fact-check

An independent fact-check on 2026-10-14 (UTC) recomputed the results on the server, mostly with its own code. It
changed no file of the round. It found no numerical error, and all 709 cells of the README's tables equal
`summary.json`.

- **Integrity.**
  - All 34 digests pinned in the configuration match their files, including the three source Ordered models.
  - The nine frozen final weights match their run manifests, `checks.json` and `summary.json`. Every run has 18,750
    updates and no interruption.
  - The server's code and configuration equal the committed versions, except the report script. That script differs
    only by the schematic's label, which the result commit adds.
  - The three test files pass on the server (29 tests).
- **Inputs.**
  - All saved development window features reproduce the source round's window logits (largest difference 8.1e-6,
    every argmax equal). Every sample carries its nearest centre's feature.
  - A GPU re-encoding of the first multi stream of each half with the seed 0 and 2 encoders matched the saved features
    exactly. These streams have 28 and 25 windows.
  - The training features keep their identity under the current code.
- **Systems and comparisons.**
  - The nine frozen heads were run again on the saved features. All 66 system rows equal `summary.json` in every
    field.
  - The B3 settings of Linear + B3 were chosen again on calibration, with the same result.
  - Frozen Ordered + B3 reproduces the source round's stored rows exactly.
  - The fact-checker's own segment F1@50 and native macro-F1 agree with the repository's on all 56 rows with
    predictions.
  - All 60 paired bootstraps and all 120 verdicts reproduce. The readings are again trade-off, no gain and end.
  - Decoding MS-TCN one sequence at a time instead of in padded batches changes one sample and no reported number.
- **Diagnostics.**
  - The unit causes, the frozen list, the ASRF readings, the overlap with Linear + B3, the short-segment counts and the
    native error pairs reproduce, with distinct recordings and sessions.
  - An independent implementation of the unit causes agrees on all 2,880 stream outcomes.
- **Training records.**
  - The batch digests of all updates equal the plans and the paired ASRF plans.
  - The initial states, the class weights, the fit-monitor statements and the loss statements of section 9 were
    confirmed on the CPU.
- **Cost and ids.**
  - The fact-check used 0.049 GPU hours in two short jobs, and the round's ledger now sums to 4.18 of 8 hours.
  - No recording, session or stream id appears in any file of this directory.
- **Corrections.**
  - The fact-check corrected the wording of answer 3, the definitions of the frozen list and of the fit, and the epoch
    numbering in section 9.
  - It also corrected the chronology of the preparation and removed an unrecorded statement about server load from
    deviation 5.
  - It added precision to the reference check and to the boundary-recall intervals.
  - The statement that the server could not be reached from about 03:10 UTC rests on this session's connection
    attempts, not on a record of the round.

## Commands

Run on the server in the project `.venv`, one GPU job at a time, with `ECHO_GPU_MEM_FRACTION=0.6`.

```bash
C=configs/experiments/video_feature_temporal_v1.yaml
python scripts/prepare_video_feature_temporal.py --config $C        # training features, plans, initial states, parity, timing
python scripts/train_video_feature_temporal.py --config $C --check  # pairing, labels, smoke, reproducibility, resume, timing, budget
python scripts/queue_video_feature_temporal.py --config $C          # nine trainings, then freezing
python scripts/report_video_feature_temporal.py --config $C --evaluate
python -m pytest tests/test_video_sequence_features.py tests/test_video_feature_temporal_training.py tests/test_video_feature_temporal_report.py -q
```

## Provenance

- **Code.**
  - The feature module is at b47b986, the preparation at 32bf8af and the queue at 43b0d74.
  - The training entry is at 6a1bd89, and the configuration at 14e359c.
  - The evaluation code is at 74e135d. The result commit only changes the schematic's label.
- **Sources.**
  - The configuration pins the task sheet and the dense round's closing record.
  - It also pins the source round's configuration, summary and README, and its three frozen Ordered models with their
    run manifests, identities and saved stride-5 outputs.
  - The protocol, the E1 feature manifest, the frame manifest, B7's window plans, the EchoPrime checkpoint and the
    environment are pinned too.
  - So are the paired round's configuration, summary, plans and ASRF initial states.
  - Every digest was checked before use. The source models were unchanged after both extractions.
- **Models.** The nine final weights are frozen in `models_frozen/video_feature_temporal_v1/{linear,mstcn,asrf}_s{0,1,2}`
  with their run manifests, plans and configuration. Their digests are in `summary.json`.
- **Server outputs.**
  - `runs/video_feature_temporal_v1/features/{train,development}/ordered_s{0,1,2}.npz` hold the float32 window and
    sample features.
  - `runs/video_feature_temporal_v1/{plans,inits,train}` hold the batch plans, the initial states and the training
    logs.
  - Recording, session and stream ids stay on the server.

## Files

- `README.md` is this file. `code_review.md` is the implementation review, `preparation.json` the preparation record
  and `checks.json` the passed check record.
- `summary.json` holds every system row, the paired comparisons and verdicts, the goal, the Linear + B3 settings, the
  frozen list, the ASRF readings, the overlap, the unit causes, the short-segment counts, the native errors, the
  decision, the parity, the runs and the costs. `report_tables.md` holds the generated tables.
- `training_curves.png` shows the training curves, and `feature_flow.png` the input schematic.
- None of these files contains a recording, session or stream id.
