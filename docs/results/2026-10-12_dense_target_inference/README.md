# Dense target-time inference round (2026-10-12)

The video models of the raw-video context round answer which view the probe shows at a requested 10 Hz sample. That
round requested an answer at every fifth sample and gave each remaining sample the answer of its nearest request. This
round checked whether that sparse request causes an avoidable loss. The three frozen Ordered context models were asked
at every sample (Dense), with the same weights, inputs and B3 numbers. At the old request positions, their saved
answers were reused. The saved sparse answers were also interpolated linearly between requests (Interpolated). Sparse
is the source round's own output. Nothing was trained, and
no threshold was chosen again. This is an implementation check of the same trained models, not a new model. The task
sheet is `docs/superpowers/plans/2026-10-12-dense-target-inference.md`.

**What kind of evidence this is.**
- The three frozen models and their saved outputs come from the raw-video context round. Every Sparse number that the
  source round stored is reproduced exactly.
- The development halves have been used in many earlier rounds, so this evaluation is not an independent validation.
- The hold-out is by session, not by patient.
- Constructed streams are spliced from separate recordings and are not real scans. They reuse recordings, so their
  units are not independent cases. The three seeds share the same cases.
- Native clips carry the public recording label, not expert frame labels. No label was changed.
- The test split and EchoNet-Pediatric were not read.
- Reference intervals serve the error diagnostics only. The diagnostic counts are not scores that any system could be
  expected to reach.

**Order of work.**
- c584846 committed the task sheet and the fixed configuration before any code. 186146b committed the tests, which
  failed at import, and refined three readings of the configuration before any output existed.
- ff65af1 committed the code. A first check passed. A CPU dry run of the report path then set all three conditions to
  the known Sparse outputs, so no Dense or Interpolated output existed. It showed a table header that broke its markdown
  columns.
- An independent implementation review passed ff65af1 with no critical or high-severity finding
  (`code_review.md`). 2d181ab fixed the header and the review's two low-severity items.
- The check of 2d181ab started at 07:19 UTC and passed, and 6176838 committed its record and the review record.
- The run went from 07:21 to 08:10 UTC on 2026-10-12. It inferred 170,448 new windows and then computed every result
  once.
- An independent fact-check recomputed the results (see "Fact-check").

## Answers

Values are given for seeds 0, 1 and 2 in that order. A range "a to b" gives the smallest and the largest value, over
seeds and, where the sentence says so, over halves. A difference is consistent only when all three seeds' 95% intervals
exclude zero strictly on the same side. All systems below use B3 with each seed's inherited numbers unless they are
called direct argmax.

1. **Does a request at every sample improve the segmentation?** No consistent effect was detected.
   - **The primary metric.** Dense − Sparse multi-stream segment F1@50 is +0.004, +0.004 and +0.005 on selection and
     +0.000, −0.005 and −0.003 on calibration. Every 95% interval contains zero. None of the five other paired metrics
     (native macro-F1, label switches per native clip, boundary recall, boundary F1 and cuts at same-view joins) shows a
     consistent difference either.
   - **Boundary placement.** The changes that Dense finds lie closer to the reference. Their mean absolute offset is
     1.02 to 1.14 samples over seeds and halves, against 1.46 to 1.58 for Sparse. The 90th percentile is 2 samples
     (0.2 s) against 3 (0.3 s). The share of exact matches is 0.24 to 0.30 against 0.20 to 0.22, over seeds and
     halves. Dense matches about as many changes (371 to 374 on selection and 390 to 392 on calibration, against 366 to
     372 and 390 to 393). Offsets within the 0.5 s tolerance count the same for boundary recall, and these offsets
     carry no interval.
   - **The 10 Hz axis.** On multi streams, the Dense argmax is right where the Sparse argmax is wrong at 331 to 393
     samples per seed and half, and the reverse holds at 163 to 218 samples, out of about 15,000. Both are wrong at
     458 to 742 samples per seed and half. With B3, per-sample accuracy on multi streams is 0.943 to 0.960 for Dense
     against 0.930 to 0.951 for Sparse, over seeds and halves. These per-sample changes do not reach the segment
     metric, and they are not called a system gain.
2. **Do new observations or the mapping make the difference?** Neither leads to a consistent segmentation gain.
   - **Interpolated − Sparse.** Interpolation changes no observation, only the values between requests. With the
     inherited numbers it lowers boundary recall consistently in both halves (−0.020 to −0.083 over seeds and halves).
     It leaves 12 to 45 multi-stream units with a missed cut per seed and half, against 3 to 15 for Sparse. Its seed 2
     misses the calibration recall floor with the fixed numbers (0.897), so that seed's B3 row fails. Its segment F1@50
     shows no consistent difference (+0.004, −0.005 and −0.017 on selection and −0.009, −0.009 and −0.046 on
     calibration). Seed 2's intervals lie below zero in both halves, but the other seeds' do not.
   - **Dense − Interpolated.** Dense has consistently higher boundary recall than Interpolated in both halves (+0.015
     to +0.083 over seeds and halves), but its segment F1@50 shows no consistent difference (+0.001 to +0.043 over
     seeds and halves). Here seed 2's intervals lie above zero in both halves, but the other seeds' do not.
   - **What this means.** The thresholds were chosen in the source round on the stepwise Sparse outputs. With the same
     numbers, the interpolated outputs miss more view changes; this round does not test why. Because Dense − Sparse
     shows no consistent gain, there is no gain to divide between new observations and the mapping. By the fixed
     reading, neither equivalence of Dense and Interpolated nor a gain wholly from new observations is claimed.
3. **Where do the remaining errors arise?**
   - **Native clips.** The native clips that B3 gets wrong are few, and most are A4C read as A5C. Over the three seeds,
     Dense gets 5 selection recordings from 4 sessions wrong (4 A4C read as A5C and 1 PLAX read as PSAX) and 2
     calibration recordings from 2 sessions (both A4C read as A5C). Sparse gets 4 selection recordings (3 A4C read as
     A5C and 1 PLAX read as PSAX) and the same 2 calibration recordings wrong. Dense fixes one A4C clip of seed 1 and
     newly misreads one of seed 0 on selection.
   - **The stage of the native errors.** In every condition and half, more than half of the wrong recordings are in the
     recognition stage, where the condition's own argmax majority and the Dense one are both wrong. For Dense on
     selection, 4 of the 5 wrong recordings are in the recognition stage and 2 in the decoding stage, because one
     recording is wrong at different stages in different seeds. For Dense, in each seed, one wrong selection clip has
     the right class by its mean probability. This is localisation only. It does not show that the visual features of
     these clips carry no information, and it triggers no training.
   - **Multi-stream units.** Dense leaves 24, 16 and 14 of 502 selection units and 7, 9 and 11 of 518 calibration units
     unmatched, against 29, 18 and 15 and 8, 7 and 10 for Sparse. On selection the main causes are missed cuts (13, 12
     and 9) and label errors (11, 4 and 4). On calibration they are label errors (3, 5 and 4) and missed cuts (3, 4 and
     6).
   - **Readings of the unmatched Dense units.** The readings are not exclusive. On selection, 11, 5 and 7 units have a
     Dense argmax majority that is not the reference class (raw classification), and 13, 11 and 7 have the right
     majority while the decoded output fails them (decoding loss). On calibration, raw classification covers 7, 7 and 9
     of the 7, 9 and 11 unmatched units, and decoding loss covers 0, 2 and 2. Interval mismatch applies to no selection
     unit and to 2, 1 and 1 calibration units. Sampling position cannot apply to Dense itself; for Sparse it covers 1,
     2 and 0 selection units and no calibration unit.
   - **Hit and near.** A change matched within 1 s but beyond 0.5 s never counts as a hit. Units that are matched but
     have such an edge number 2, 2 and 0 on selection and 3, 3 and 2 on calibration for Dense, against 7, 4 and 4 and 2,
     2 and 2 for Sparse.
   - **What sampling changes and what stays unexplained.** The request at every sample moves the found boundaries closer
     and changes per-sample labels on multi streams. The fact-check found 82 to 93% of these changed samples within 5
     samples of a reference view change. No paired metric changes consistently. Dense adds one constraint failure,
     seed 0's selection native floor, from one newly misread A4C clip. Most remaining native errors and most remaining
     calibration unit errors have a wrong Dense argmax majority, so denser requests to the same model do not reach them.
     On selection, about half or more of the unmatched units have a right Dense majority. These are mostly missed cuts,
     a decoding loss that Sparse and Dense share, and this round does not show why B3 misses them.
4. **The direct argmax rows.** The Dense argmax switches labels more often inside native clips (0.135 to 0.232 per clip
   over seeds and halves, against 0.093 to 0.116 for the Sparse argmax). Without B3, Dense − Sparse segment F1@50 is
   consistently negative in both halves (−0.040 to −0.094 over seeds and halves), and so is boundary F1. With the same
   B3 numbers, Dense has 0.017 to 0.046 label switches per native clip, like Sparse's 0.013 to 0.046, over seeds and
   halves. These rows are secondary.
5. **The goal.** No condition meets the goal.
   - **Three-seed means.** Sparse has 0.9544 and 0.9817, Dense 0.9588 and 0.9792, and Interpolated 0.9484 and 0.9603,
     against minimums of 0.9444 and 0.9821.
   - **Constraints.** Dense misses the selection native floor in seeds 0 and 2 (0.9701 against 0.973), and Sparse in
     seed 2. Interpolated fails four constraints, including its infeasible seed 2. With the fixed numbers, the
     calibration rule holds for every Sparse and Dense seed.
6. **The fixed decision.** Four readings apply.
   - **Offset only.** Dense has the smaller mean offset in every seed and half, while no half has a consistent positive
     Dense − Sparse difference in segment F1@50 or native macro-F1. So this is finer localisation only, and not evidence
     that the remaining loss comes from the request grid. This configuration ends without any stride, interpolation or
     decoding search.
   - **Interpolation close.** In both halves, Dense − Interpolated segment F1@50 is not consistent positive. Neither an
     equivalence nor a gain wholly from new observations is claimed, and the primary condition stays Dense − Sparse.
   - **Native localisation.** The recognition stage holds in every condition and half.
   - **End.** The round ends at this report.

   The gain readings, the goal reading and the cost reading do not apply. The test split stays sealed.
7. **Cost.** The round used 0.87 GPU hours of its own 4-hour cap.
   - **Checks.** The two check runs took 0.017 hours each.
   - **Run.** The run took 0.83 hours. Inference of the new windows took 0.76 hours, at 16.0 ms per window in batches
     of 32 windows. Frame decoding took 23 seconds, and loading and the report about 4 minutes.
   - **Fact-check.** Its GPU job took 18.5 seconds.
   - **Speed.** A request at every sample needs about five times the windows of the stride-5 request. The 16.0 ms per
     window is batch throughput on one GPU, not a streaming latency.

## Terms

- **Sparse**, **Interpolated** and **Dense** give every 10 Hz sample one output. Sparse takes the nearest stride-5
  request (ties to the earlier one). Interpolated takes the linear interpolation of the two neighbouring requests' class
  probabilities, and holds the end values outside them. Dense requests a 16-frame window at the sample itself. At the
  old request positions the three are identical.
- **B3** is the change-point decoder. Every condition uses each seed's numbers from the source round (half-window 9,
  suppression radius 5, thresholds 0.29269, 0.28451 and 0.30845). No quantile was recomputed.
- A **unit** is a reference segment of a constructed stream. A unit is **matched** when a predicted segment of its class
  overlaps it with IoU of at least 0.5 under the one-to-one matching of segment F1@50. An unmatched unit gets the first
  cause that applies, in the order label, missed cut, extra cut, boundary shift.
- **Hit** means a predicted change within 5 samples (0.5 s) of a reference change, the main boundary tolerance. **Near**
  means within 10 samples (1 s), a diagnostic range only.
- The four **readings** of an unmatched unit are not exclusive, and a wrong native clip has the first three.
  - Raw classification means that the Dense argmax majority over the reference interval is not the reference class.
  - Sampling position means that the condition's own argmax majority is wrong while the Dense one is right.
  - Decoding loss means that the condition's own argmax majority is right but the decoded output fails.
  - Interval mismatch means a boundary shift, or an edge matched only beyond hit.
- The **native stage** of a wrong clip is decoding when its own argmax majority is right, sampling when that majority is
  wrong but the Dense one is right, and recognition when both are wrong.
- The **paired bootstrap** resamples native clips by session and streams by stream, with 2,000 draws, seed 20261002, and
  the same draws for both systems.

## 1. Systems

Each cell gives the selection half, then the calibration half. The Sparse rows equal the source round's stored rows.

| System (B3, fixed numbers) | Seed | Segment F1@50 | Boundary recall | Boundary precision | Native macro-F1 | Label switches per native clip | Cuts at same-view joins | 10 Hz accuracy, multi streams | A4C read as A5C (clips) | Fixed numbers feasible |
|---|---|---|---|---|---|---|---|---|---|---|
| Sparse | 0 | 0.938 / 0.982 | 0.958 / 0.982 | 0.948 / 0.975 | 0.9786 / 0.9929 | 0.042 / 0.025 | 0.017 / 0.025 | 0.930 / 0.950 | 2 / 1 | yes |
| Sparse | 1 | 0.960 / 0.984 | 0.969 / 0.987 | 0.959 / 0.980 | 0.9786 / 0.9861 | 0.039 / 0.017 | 0.004 / 0.025 | 0.943 / 0.951 | 2 / 2 | yes |
| Sparse | 2 | 0.964 / 0.980 | 0.974 / 0.980 | 0.959 / 0.977 | 0.9701 / 0.9861 | 0.046 / 0.013 | 0.004 / 0.025 | 0.943 / 0.947 | 3 / 2 | yes |
| Interpolated | 0 | 0.942 / 0.973 | 0.932 / 0.952 | 0.981 / 0.992 | 0.9786 / 0.9929 | 0.021 / 0.008 | 0.013 / 0.010 | 0.924 / 0.942 | 2 / 1 | yes |
| Interpolated | 1 | 0.956 / 0.975 | 0.948 / 0.967 | 0.973 / 0.987 | 0.9877 / 0.9861 | 0.032 / 0.008 | 0.009 / 0.010 | 0.943 / 0.948 | 1 / 2 | yes |
| Interpolated | 2 | 0.947 / 0.934 | 0.903 / 0.897 | 0.989 / 0.997 | 0.9701 / 0.9861 | 0.035 / 0.013 | 0.004 / 0.005 | 0.922 / 0.919 | 3 / 2 | no |
| Dense | 0 | 0.943 / 0.982 | 0.971 / 0.985 | 0.946 / 0.973 | 0.9701 / 0.9929 | 0.042 / 0.021 | 0.022 / 0.025 | 0.943 / 0.958 | 3 / 1 | yes |
| Dense | 1 | 0.964 / 0.979 | 0.974 / 0.982 | 0.964 / 0.973 | 0.9877 / 0.9861 | 0.039 / 0.021 | 0.009 / 0.020 | 0.958 / 0.960 | 1 / 2 | yes |
| Dense | 2 | 0.969 / 0.977 | 0.979 / 0.980 | 0.971 / 0.975 | 0.9701 / 0.9861 | 0.046 / 0.017 | 0.009 / 0.025 | 0.957 / 0.957 | 3 / 2 | yes |

Direct argmax, without B3 (secondary).

| System (direct argmax) | Seed | Segment F1@50 | Boundary recall | Boundary precision | Native macro-F1 | Label switches per native clip | 10 Hz accuracy, multi streams |
|---|---|---|---|---|---|---|---|
| Sparse | 0 | 0.893 / 0.961 | 0.979 / 0.987 | 0.846 / 0.942 | 0.9701 / 0.9861 | 0.095 / 0.093 | 0.926 / 0.947 |
| Sparse | 1 | 0.925 / 0.961 | 0.984 / 0.990 | 0.872 / 0.945 | 0.9877 / 0.9796 | 0.113 / 0.101 | 0.937 / 0.945 |
| Sparse | 2 | 0.926 / 0.960 | 0.984 / 0.982 | 0.874 / 0.940 | 0.9701 / 0.9861 | 0.116 / 0.097 | 0.937 / 0.943 |
| Interpolated | 0 | 0.898 / 0.961 | 0.976 / 0.985 | 0.842 / 0.940 | 0.9701 / 0.9861 | 0.095 / 0.093 | 0.930 / 0.950 |
| Interpolated | 1 | 0.931 / 0.961 | 0.990 / 0.987 | 0.877 / 0.942 | 0.9877 / 0.9796 | 0.113 / 0.101 | 0.945 / 0.950 |
| Interpolated | 2 | 0.927 / 0.959 | 0.987 / 0.980 | 0.875 / 0.935 | 0.9701 / 0.9861 | 0.116 / 0.097 | 0.942 / 0.947 |
| Dense | 0 | 0.853 / 0.919 | 0.987 / 0.987 | 0.756 / 0.845 | 0.9701 / 0.9861 | 0.176 / 0.135 | 0.940 / 0.955 |
| Dense | 1 | 0.831 / 0.912 | 0.992 / 0.990 | 0.698 / 0.835 | 0.9877 / 0.9861 | 0.208 / 0.232 | 0.951 / 0.958 |
| Dense | 2 | 0.855 / 0.912 | 0.992 / 0.990 | 0.740 / 0.831 | 0.9786 / 0.9861 | 0.187 / 0.224 | 0.949 / 0.954 |

## 2. Comparisons

Paired differences with 95% intervals. Streams reuse recordings, so the intervals are approximate. "none" means that
no consistent difference was detected, which does not mean that the systems are equal.

| Comparison | Half | Segment F1@50 (seeds 0, 1, 2) | Verdict | Boundary recall (seeds 0, 1, 2) | Verdict | Other consistent differences |
|---|---|---|---|---|---|---|
| B3: Dense − Sparse | selection | +0.0043 [−0.0066, +0.0177]; +0.0040 [−0.0020, +0.0108]; +0.0049 [−0.0010, +0.0119] | none | +0.0131 [+0.0028, +0.0250]; +0.0052 [−0.0050, +0.0160]; +0.0052 [−0.0050, +0.0158] | none | none |
| B3: Dense − Sparse | calibration | +0.0000 [−0.0038, +0.0039]; −0.0048 [−0.0122, +0.0000]; −0.0029 [−0.0074, +0.0010] | none | +0.0025 [−0.0051, +0.0124]; −0.0050 [−0.0127, +0.0000]; +0.0000 [−0.0074, +0.0075] | none | none |
| B3: Dense − Interpolated | selection | +0.0007 [−0.0146, +0.0178]; +0.0086 [−0.0015, +0.0190]; +0.0217 [+0.0096, +0.0347] | none | +0.0393 [+0.0165, +0.0663]; +0.0262 [+0.0126, +0.0427]; +0.0759 [+0.0499, +0.1035] | consistent positive | none |
| B3: Dense − Interpolated | calibration | +0.0092 [−0.0024, +0.0235]; +0.0041 [−0.0038, +0.0157]; +0.0431 [+0.0214, +0.0668] | none | +0.0327 [+0.0131, +0.0563]; +0.0151 [+0.0025, +0.0326]; +0.0829 [+0.0543, +0.1147] | consistent positive | none |
| B3: Interpolated − Sparse | selection | +0.0036 [−0.0131, +0.0196]; −0.0046 [−0.0134, +0.0036]; −0.0169 [−0.0306, −0.0027] | none | −0.0262 [−0.0494, −0.0059]; −0.0209 [−0.0356, −0.0081]; −0.0707 [−0.0976, −0.0452] | consistent negative | none |
| B3: Interpolated − Sparse | calibration | −0.0092 [−0.0233, +0.0019]; −0.0089 [−0.0227, +0.0008]; −0.0460 [−0.0702, −0.0243] | none | −0.0302 [−0.0538, −0.0118]; −0.0201 [−0.0424, −0.0048]; −0.0829 [−0.1147, −0.0538] | consistent negative | none |
| argmax: Dense − Sparse | selection | −0.0400 [−0.0621, −0.0183]; −0.0941 [−0.1300, −0.0614]; −0.0705 [−0.0991, −0.0435] | consistent negative | +0.0079 [−0.0027, +0.0199]; +0.0079 [−0.0027, +0.0192]; +0.0079 [+0.0000, +0.0183] | none | native cuts per clip consistent positive, multi boundary F1 consistent negative |
| argmax: Dense − Sparse | calibration | −0.0418 [−0.0643, −0.0215]; −0.0494 [−0.0766, −0.0242]; −0.0483 [−0.0705, −0.0279] | consistent negative | +0.0000 [−0.0073, +0.0076]; +0.0000 [−0.0074, +0.0074]; +0.0075 [+0.0000, +0.0175] | none | multi boundary F1 consistent negative |
| argmax: Dense − Interpolated | selection | −0.0448 [−0.0659, −0.0243]; −0.0998 [−0.1354, −0.0682]; −0.0715 [−0.0990, −0.0457] | consistent negative | +0.0105 [+0.0025, +0.0213]; +0.0026 [+0.0000, +0.0080]; +0.0052 [+0.0000, +0.0134] | none | native cuts per clip consistent positive, multi boundary F1 consistent negative |
| argmax: Dense − Interpolated | calibration | −0.0418 [−0.0643, −0.0215]; −0.0494 [−0.0766, −0.0242]; −0.0474 [−0.0700, −0.0272] | consistent negative | +0.0025 [+0.0000, +0.0080]; +0.0025 [+0.0000, +0.0078]; +0.0101 [+0.0024, +0.0206] | none | multi boundary F1 consistent negative |
| argmax: Interpolated − Sparse | selection | +0.0048 [−0.0009, +0.0122]; +0.0057 [−0.0036, +0.0169]; +0.0010 [−0.0026, +0.0057] | none | −0.0026 [−0.0122, +0.0055]; +0.0052 [−0.0052, +0.0162]; +0.0026 [+0.0000, +0.0082] | none | none |
| argmax: Interpolated − Sparse | calibration | +0.0000 [+0.0000, +0.0000]; +0.0000 [+0.0000, +0.0000]; −0.0009 [−0.0028, +0.0000] | none | −0.0025 [−0.0078, +0.0000]; −0.0025 [−0.0078, +0.0000]; −0.0025 [−0.0078, +0.0000] | none | none |

## 3. Matched boundary offsets

Predicted changes matched one to one to reference changes within 5 samples, with B3.

| System (B3) | Seed | Matched changes | Mean absolute offset (samples) | Median | 90th percentile | Mean absolute offset (s) | Exact share |
|---|---|---|---|---|---|---|---|
| Sparse | 0 | 366 / 391 | 1.55 / 1.46 | 1 / 1 | 3 / 3 | 0.155 / 0.146 | 0.20 / 0.22 |
| Sparse | 1 | 370 / 393 | 1.58 / 1.47 | 1 / 1 | 3 / 3 | 0.158 / 0.147 | 0.20 / 0.22 |
| Sparse | 2 | 372 / 390 | 1.55 / 1.49 | 1 / 1 | 3 / 3 | 0.155 / 0.149 | 0.20 / 0.21 |
| Interpolated | 0 | 356 / 379 | 1.47 / 1.43 | 1 / 1 | 3 / 3 | 0.147 / 0.143 | 0.20 / 0.21 |
| Interpolated | 1 | 362 / 385 | 1.46 / 1.38 | 1 / 1 | 3 / 3 | 0.146 / 0.138 | 0.22 / 0.22 |
| Interpolated | 2 | 345 / 357 | 1.49 / 1.42 | 1 / 1 | 3 / 3 | 0.149 / 0.142 | 0.21 / 0.21 |
| Dense | 0 | 371 / 392 | 1.10 / 1.14 | 1 / 1 | 2 / 2 | 0.110 / 0.114 | 0.30 / 0.24 |
| Dense | 1 | 372 / 391 | 1.04 / 1.02 | 1 / 1 | 2 / 2 | 0.104 / 0.102 | 0.30 / 0.27 |
| Dense | 2 | 374 / 390 | 1.12 / 1.10 | 1 / 1 | 2 / 2 | 0.112 / 0.110 | 0.26 / 0.26 |

## 4. Reference units of multi streams

Unmatched units by cause and by reading, with B3. Each cell gives the selection half, then the calibration half. The
readings are not exclusive. `summary.json` adds the distinct recordings and sessions of every count, the pair streams
and the flags.

| System (B3) | Seed | Wrong units (of 502 / 518) | Label | Missed cut | Extra cut | Boundary shift | Raw classification | Sampling position | Decoding loss | Interval mismatch | Matched units with an edge only within near | Unmatched predicted segments |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Sparse | 0 | 29 / 8 | 11 / 3 | 15 / 5 | 2 / 0 | 1 / 0 | 12 / 7 | 1 / 0 | 16 / 3 | 1 / 0 | 7 / 2 | 33 / 11 |
| Sparse | 1 | 18 / 7 | 3 / 4 | 14 / 3 | 1 / 0 | 0 / 0 | 5 / 6 | 2 / 0 | 11 / 1 | 0 / 0 | 4 / 2 | 22 / 10 |
| Sparse | 2 | 15 / 10 | 4 / 4 | 11 / 6 | 0 / 0 | 0 / 0 | 6 / 9 | 0 / 0 | 9 / 2 | 0 / 0 | 4 / 2 | 21 / 11 |
| Interpolated | 0 | 38 / 22 | 12 / 3 | 26 / 19 | 0 / 0 | 0 / 0 | 12 / 7 | 1 / 0 | 26 / 15 | 1 / 0 | 3 / 0 | 19 / 6 |
| Interpolated | 1 | 27 / 17 | 3 / 5 | 21 / 12 | 1 / 0 | 2 / 0 | 6 / 8 | 1 / 0 | 20 / 10 | 2 / 0 | 5 / 2 | 17 / 9 |
| Interpolated | 2 | 42 / 53 | 6 / 8 | 36 / 45 | 0 / 0 | 0 / 0 | 7 / 9 | 0 / 0 | 36 / 45 | 0 / 0 | 4 / 0 | 9 / 13 |
| Dense | 0 | 24 / 7 | 11 / 3 | 13 / 3 | 0 / 0 | 0 / 1 | 11 / 7 | 0 / 0 | 13 / 0 | 0 / 2 | 2 / 3 | 34 / 12 |
| Dense | 1 | 16 / 9 | 4 / 5 | 12 / 4 | 0 / 0 | 0 / 0 | 5 / 7 | 0 / 0 | 11 / 2 | 0 / 1 | 2 / 3 | 20 / 13 |
| Dense | 2 | 14 / 11 | 4 / 4 | 9 / 6 | 1 / 0 | 0 / 1 | 7 / 9 | 0 / 0 | 7 / 2 | 0 / 1 | 0 / 2 | 17 / 13 |

## 5. Native clips

Per seed, with B3. "Raw" is the condition's own argmax majority and "decode" the B3 majority, which is the scored label.

| System | Seed | B3-wrong clips | A4C read as A5C | Raw right, decode wrong | Both wrong | Raw wrong, decode right | Mean label right among the wrong | Stage decoding | Stage sampling | Stage recognition |
|---|---|---|---|---|---|---|---|---|---|---|
| Sparse | 0 | 3 / 1 | 2 / 1 | 0 / 0 | 3 / 1 | 1 / 1 | 1 / 0 | 0 / 0 | 0 / 0 | 3 / 1 |
| Sparse | 1 | 3 / 2 | 2 / 2 | 1 / 0 | 2 / 2 | 0 / 1 | 2 / 0 | 1 / 0 | 0 / 0 | 2 / 2 |
| Sparse | 2 | 4 / 2 | 3 / 2 | 0 / 0 | 4 / 2 | 0 / 0 | 1 / 0 | 0 / 0 | 1 / 0 | 3 / 2 |
| Interpolated | 0 | 3 / 1 | 2 / 1 | 0 / 0 | 3 / 1 | 1 / 1 | 1 / 0 | 0 / 0 | 0 / 0 | 3 / 1 |
| Interpolated | 1 | 2 / 2 | 1 / 2 | 1 / 0 | 1 / 2 | 1 / 1 | 1 / 0 | 1 / 0 | 0 / 0 | 1 / 2 |
| Interpolated | 2 | 4 / 2 | 3 / 2 | 0 / 0 | 4 / 2 | 0 / 0 | 1 / 0 | 0 / 0 | 1 / 0 | 3 / 2 |
| Dense | 0 | 4 / 1 | 3 / 1 | 0 / 0 | 4 / 1 | 0 / 1 | 1 / 0 | 0 / 0 | 0 / 0 | 4 / 1 |
| Dense | 1 | 2 / 2 | 1 / 2 | 1 / 0 | 1 / 2 | 1 / 0 | 1 / 0 | 1 / 0 | 0 / 0 | 1 / 2 |
| Dense | 2 | 4 / 2 | 3 / 2 | 1 / 0 | 3 / 2 | 0 / 0 | 1 / 0 | 1 / 0 | 0 / 0 | 3 / 2 |

The union over the three seeds, by distinct recordings.

| System | Half | B3-wrong recordings (sessions) | By class pair: recordings (sessions) | Decoding | Sampling | Recognition | Fixed reading |
|---|---|---|---|---|---|---|---|
| Sparse | selection | 4 (4) | A4C read as A5C 3 (3); PLAX read as PSAX 1 (1) | 1 | 1 | 3 | recognition |
| Sparse | calibration | 2 (2) | A4C read as A5C 2 (2) | 0 | 0 | 2 | recognition |
| Interpolated | selection | 4 (4) | A4C read as A5C 3 (3); PLAX read as PSAX 1 (1) | 1 | 1 | 3 | recognition |
| Interpolated | calibration | 2 (2) | A4C read as A5C 2 (2) | 0 | 0 | 2 | recognition |
| Dense | selection | 5 (4) | A4C read as A5C 4 (4); PLAX read as PSAX 1 (1) | 2 | 0 | 4 | recognition |
| Dense | calibration | 2 (2) | A4C read as A5C 2 (2) | 0 | 0 | 2 | recognition |

## 6. The 10 Hz axis

Samples by sequence kind, comparing the Dense argmax with the Sparse argmax. `summary.json` adds the other two pairs and
the distinct recordings and sessions.

| Seed | Sequences | Samples | Dense right, Sparse wrong | Sparse right, Dense wrong | Both wrong |
|---|---|---|---|---|---|
| 0 | native | 16635 / 14323 | 22 / 12 | 15 / 18 | 296 / 182 |
| 0 | pairs | 5691 / 4883 | 62 / 50 | 41 / 28 | 173 / 137 |
| 0 | multi | 14961 / 15010 | 368 / 331 | 163 / 218 | 742 / 458 |
| 1 | native | 16635 / 14323 | 26 / 26 | 27 / 22 | 298 / 173 |
| 1 | pairs | 5691 / 4883 | 69 / 49 | 45 / 27 | 175 / 150 |
| 1 | multi | 14961 / 15010 | 393 / 354 | 182 / 168 | 544 / 469 |
| 2 | native | 16635 / 14323 | 25 / 22 | 23 / 28 | 343 / 225 |
| 2 | pairs | 5691 / 4883 | 68 / 55 | 36 / 24 | 183 / 135 |
| 2 | multi | 14961 / 15010 | 349 / 343 | 175 / 182 | 592 / 510 |

## 7. Goal

Three-seed means with B3 against the minimums of 0.9444 and 0.9821. The calibration sensitivity target, 0.9839, is the
G1 + ASRF calibration mean plus 0.01.

| System (B3) | Three-seed mean | Margin to the minimum | Calibration sensitivity margin | Constraint failures | Goal met |
|---|---|---|---|---|---|
| Sparse | 0.9544 / 0.9817 | +0.0100 / −0.0004 | −0.0022 | 1 | no |
| Interpolated | 0.9484 / 0.9603 | +0.0040 / −0.0218 | −0.0236 | 4 | no |
| Dense | 0.9588 / 0.9792 | +0.0144 / −0.0030 | −0.0047 | 2 | no |

## What this round does not show

- **No new model.** Dense uses the same trained weights and the same decoder numbers. It is an implementation check of
  request density, not a new architecture or an order mechanism.
- **No system gain.** Dense does not differ consistently from Sparse in any paired metric with B3. Interpolated has
  consistently lower boundary recall than Sparse. No condition meets the goal.
- **Not the cause of the remaining loss.** Smaller offsets with unchanged segments do not show that the remaining loss
  comes from the request grid. Most remaining native errors and most calibration unit errors have a wrong Dense argmax
  majority. On selection, about half or more of the unmatched units have a right Dense majority. The round does not show
  why the model misreads these A4C clips or why B3 misses the selection cuts that the raw output supports.
- **Not why interpolation loses recall.** The fixed thresholds came from stepwise outputs. Whether the interpolated
  outputs give change scores below these thresholds was not tested.
- **No cardiac or probe explanation.** Score changes over a clip are not attributed to cardiac phase or probe motion.
- **Not clinical scanning.** Constructed streams and public recording labels are not real continuous scans or expert
  frame labels, and the development halves are not an independent validation.

## Deviations and launch record

1. **Readings refined before any output.** 186146b added a reading for a segmentation gain without the constraints,
   made the offset-only reading require no consistent positive segment F1@50 or native macro-F1 in either half, and
   split the native localisation into decoding, sampling and recognition. This happened before any dense output existed.
2. **The parity windows.** The task sheet asks for 16 old windows chosen in sequence order. The configuration fixed them
   before any code as the first 8 old windows of native clips and the first 8 of multi streams on the selection half,
   for every seed.
3. **A dry run of the report path.** It was not in the task sheet. It ran on the CPU with all three conditions set to
   the known Sparse outputs and 20 bootstrap draws, wrote only to a scratch directory, and was deleted. It found the
   table header fault fixed in 2d181ab.
4. **Two check runs.** The first check started at 06:48 UTC and passed. The fixes of 2d181ab changed the report code and
   so the run identity, and the check was repeated, starting at 07:19 UTC. Both are in the ledger.
5. **A cost field name.** The field `report_seconds` in `summary.json` holds the seconds from the start of the run to
   the end of the report (2,977 s), not the report alone.

## Fact-check

An independent fact-check recomputed the round from the saved per-sample outputs, mostly with its own code. Every
number in `summary.json`, `report_tables.md`, `checks.json` and the README tables matched. Its corrections concerned
wording only, and this README carries them.

- **Shared observations.** At the old request positions, the dense logits equal the source round's saved logits bit for
  bit, and the three conditions are identical. Sparse equals the saved per-sample probabilities, Interpolated equals an
  independent rebuild, and Dense equals the float32 softmax of the saved logits.
- **Recomputed exactly.**
  - All 36 system rows with 16 fields each. The Sparse rows equal the source round's stored rows.
  - All 36 paired bootstraps with six metrics each, and all 72 verdicts.
  - The offsets, the unit tables with hit 5 and near 10, the native stages and their unions, the 10 Hz table, the goal
    with its feasibility recheck, the case counts and the fixed readings.
- **Thresholds.** The inherited B3 numbers equal the source round's chosen settings. No code recomputes a quantile or
  searches a setting.
- **Re-decoded.** The first 32 new windows of the first multi stream of each half, for seeds 0 and 2, gave logits
  bit-identical to the saved ones. The job took 18.5 seconds of GPU time and is in the ledger.
- **Integrity.** All pinned files match, and the frozen files are unchanged. The round's code reads neither the test
  split nor EchoNet-Pediatric, and it wrote only to its own directories. The server code equals the committed code, and
  all 41 tests pass.
- **Added observation.** Of the multi-stream samples where the Dense and Sparse argmax differ, 82 to 93% lie within 5
  samples of a reference view change, per seed and half.
- **Ids.** No recording, session or stream id appears in any result file.

## Commands

Run on the server in the project `.venv`, one GPU job at a time, with `ECHO_GPU_MEM_FRACTION=0.6`.

```bash
C=configs/experiments/dense_target_inference_v1.yaml
python scripts/dense_target_inference.py --config $C --check   # sources, Sparse reproduction, alignment, parity, timing, budget
python scripts/dense_target_inference.py --config $C --run     # every new window of three seeds and two halves, then the report
python -m pytest tests/test_dense_target_inference.py tests/test_dense_target_inference_report.py -q
```

## Provenance

- **Code.** The entry script and the report module are at 2d181ab. The configuration is at 186146b.
- **Sources.** The configuration pins the task sheet, the source round's configuration, summary and README, the three
  frozen Ordered models with their run manifests and identities, their saved stride-5 outputs, the protocol, the E1
  feature manifest, B7's window plans, the EchoPrime checkpoint and the server's environment. Every digest was checked
  before use.
- **Models.** The three models stay frozen in `models_frozen/raw_video_context_v1/ordered_s{0,1,2}`. Their weights and
  in-memory state were unchanged after inference. This round adds no weights.
- **Server outputs.** `runs/dense_target_inference_v1/inference/{selection,calibration}/ordered_s{0,1,2}.npz` hold the
  logits of every centre, whether each was reused or new, and the per-sample probabilities of the three conditions.
  `runs/dense_target_inference_v1/cases/` holds 41 case figures, each showing all three conditions. They cover 16
  native-floor cases from 5 recordings and 4 sessions, and 10 native and 52 unit cases on calibration from 13
  recordings and 11 sessions. Recording, session and stream ids stay on the server.

## Files

- `README.md` is this file. `code_review.md` is the implementation review, and `checks.json` the passed check record.
- `summary.json` holds every system row, the paired comparisons and verdicts, the offsets, the unit, native and 10 Hz
  tables with their distinct recordings and sessions, the goal, the decision, the case counts and the costs.
  `report_tables.md` holds the generated tables.
- None of these files contains a recording, session or stream id.
