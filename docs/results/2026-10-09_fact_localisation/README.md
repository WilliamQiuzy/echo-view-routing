# FACT short-segment localisation on the training sequences (2026-10-09)

This round did no training and reports no development-set result, and conditions B and C of its intervention use the
reference partition, so they are diagnostics and never model results. It decoded the frozen FACT, ASRF and E1 of the
paired FACT-ASRF round (`docs/results/2026-10-08_fact_asrf_training`) on that round's own training sequences, to find
where FACT loses short reference runs. The task sheet is `docs/superpowers/plans/2026-10-09-fact-localisation.md`.

**What kind of evidence this is.**
- The sequences are each seed's last training epoch (epoch 49). That is the 1,500 constructed bank streams and the
  1,500 native windows that the epoch actually drew. FACT and ASRF trained on exactly these sequences, so every number
  here is a fit to training data, not a sign of generalisation.
- The three seeds share the same 1,500 streams. The seeds are three models on the same cases, not three samples.
  Native windows differ by seed, with 2,798 distinct recordings in 4,500 windows.
- Constructed streams are spliced from separate recordings and are not real scans. They reuse recordings, so their runs
  are not independent cases. Every count below gives its distinct recordings and sessions.
- A native window holds at most 256 samples of a clip, not the whole recording. Its label is the public clip label.
- The cache index also lists the rows of other splits. For every row, only the id, the split and the sample count were
  read, to filter the rows. The class label was read for the kept training rows only. No other index column and no
  feature of another split was read. No development half, test file or EchoNet-Pediatric file was read.
- **Correction.** The first run of this round read every index column except the class label, for every row. That
  included the raw view name and the file path of the other splits' rows. They were never used. The fact-check found
  this, so the read was narrowed and every step was repeated. The repeated run gave the same records (section 1).
- All results are post hoc, on frozen models. Several summary counts were added after the first run (sections 3 and 4).
  They come from the same saved records.

**Order of work.**
- 89bac43 committed the task sheet, the configuration, the code and its tests before anything ran.
- An independent code review passed the code with one medium and one low finding. 1f09d90 fixed both and added the
  review record (`code_review.md`).
- `--prepare` and `--check` passed, and `--run` decoded every sequence under run identity 58bb82909520492c. Each
  stream went through six FACT passes and each native window through four. 2b7a00d and 9aafb99 then added summary
  tables that read the saved records.
- An independent fact-check recounted every number and found no wrong one. It re-decoded 600 sequences and re-ran B and
  C on 60 streams with its own code, all bit for bit. It also found the broader index read and some wording errors.
- c062220 narrowed the index read and added three summary counts. `--prepare`, `--check` and `--run` were repeated
  under run identity 1bdea50968984e60. All 9,000 per-sequence records equal those of the first run. Every number in
  this README comes from the repeated run.

## Answers

Values are given for seeds 0, 1 and 2 in that order. A range "a to b" gives the smallest and the largest of the three
seeds.

1. **The 16 monitor sequences do not represent the whole training library, and FACT does not fit the library fully
   either.**
   - **The 16.** On the 16 fixed training sequences, FACT's mean per-sequence segment F1@50 is 0.955, 0.960 and 0.934,
     with 13, 12 and 12 exact transcripts. These values reproduce the last monitor record of each run exactly.
   - **The library.** On all 1,500 streams of the last epoch, the mean per-sequence F1@50 is 0.972, 0.971 and 0.969, and
     1,340, 1,328 and 1,326 transcripts are exact. The 16 hold five native windows and eleven streams chosen for
     same-view joins, A4C → A5C → A4C runs, repeated views and the longest transcripts, so their mix differs from the
     library and their mean is lower. The two averages describe different sets and are not compared further.
   - **What is left.** FACT's official output misses 182, 192 and 212 of the 6,463 reference runs in the streams. ASRF
     misses 19, 14 and 14 on the same sequences.
   - **Which runs.** Of the missed runs, 88 to 100 have at most 16 samples, 88 to 103 have 17 to 40 samples, 6 to 9
     have 41 to 80 samples, and none is longer. Among FACT's wrong stream samples, A5C read as A4C makes up 43% to 49%,
     and A4C read as A5C makes up 36% to 42%.
   - **Sources.** The missed runs come from 148 to 169 recordings in 117 to 127 sessions per seed. 141 runs, from 111
     recordings in 91 sessions, are missed in all three seeds.
   - **Native windows.** The official output matches 1,491 to 1,493 of the 1,500 windows. 8 to 12 windows contain a
     label change.
   - **Not shown.** These counts do not show that longer training would close the gap. This round split no loss and
     searched no optimiser.
2. **Most short runs that FACT misses are already unmatched before the U block, and runs lost at the u block are
   almost never matched again.**
   - **Where they were first lost.** The official output misses 88, 91 and 100 of the 1,726 short runs. Section 3
     traces each through E1's argmax, the i block, the u block, the U block's frame branch and the official output.
     30 to 34 were already unmatched in E1's argmax. 35 to 37 were first lost at the i block, 14 to 21 at the u block,
     0 to 1 at the U frame branch, and 5 to 7 only at the official output.
   - **Recovery.** E1's per-frame argmax misses 109 short runs, mostly because it flickers inside them. The i block
     matches 70 to 71 of these, but it loses 40 to 45 short runs that E1's argmax had matched. Of the short runs first
     lost at the u block, the official output matches 0 to 1 again.
   - **After the u block.** The U frame branch loses 3 to 8 short runs that the u block matched and recovers 7 to 8.
     Against the U frame branch, the official output loses 12 to 15 and recovers 6 to 9.
   - **A4C/A5C middles.** FACT's official output matches 62, 65 and 61 of the 99 A4C/A5C X → Y → X middles. ASRF matches
     96 to 97, and E1's argmax 72. Of FACT's 34 to 38 misses, 13 were unmatched in E1's argmax, 14 to 15 were first lost
     at the i block, 5 to 7 at the u block, none at the U frame branch, and 2 to 4 only at the official output.
   - **What this allows.** Most misses arise before the U block, so the U block cannot be their only cause. A run
     matched before U and lost after it is a localisation finding, not a shown mechanism.
3. **Reference positions recovered more than the same number of uniform cuts, in all three seeds.**
   - **Totals.** Against A, B raises the official output's pooled segment F1@50 by 0.0057, 0.0055 and 0.0059, and C
     by 0.0034 to 0.0035. B − C is +0.0024, +0.0021 and +0.0025 in segment F1@50 and +0.012 to +0.015 in boundary
     recall.
   - **Short runs.** Against A, B fixes 43, 37 and 44 short runs and newly misses 5, 6 and 7. C fixes 26, 24 and 27
     and newly misses 5, 7 and 11. Against C, B fixes 20, 22 and 26 and newly misses 3, 8 and 5.
   - **A4C/A5C middles.** B matches 78, 79 and 77 of 99, against 62, 65 and 61 under A and 66, 69 and 64 under C.
     Against C, B fixes 12, 10 and 13 and newly misses none.
   - **Which runs B recovers.** 85 to 93 runs per seed are the minority in every interval of the partition entering U.
     The official output matches none of them under A, 40, 42 and 36 under B, and 27, 30 and 24 under C. In this run
     the official output never changed its label inside a partition interval of a stream.
   - **The frame branch.** The U block's frame branch changes by at most 0.0015 in segment F1@50 under B or C.
   - **What this allows.** B − C compares reference positions with uniform positions at the same local number of
     added intervals. The frozen network recovers some training runs when it is given reference positions. This does
     not show that these positions can be predicted from the images. It is not an achievable gain, not an upper bound
     of the model, and not evidence about the goal of 0.944 and 0.982. A, B and C also change the sequence that the GRU
     reads, so the effect cannot be assigned to the averaging or the attention alone. Even under B, FACT matches 1,663
     to 1,676 short runs and 77 to 79 A4C/A5C middles, against 1,719 to 1,723 and 96 to 97 for ASRF without any
     reference information.

## Terms

- A **reference run** is a maximal run of one reference view in a sequence. In a constructed stream, two neighbouring
  fragments of the same view form one run. A native window is one run.
- A run is **matched** when a predicted run of the same view overlaps it with an IoU of at least 0.5, one to one. This
  is the rule of segment F1@50. A run that is not matched is a **miss**.
- **Pooled segment F1@50** is 2 × matched runs / (predicted runs + reference runs), summed over all sequences of one
  kind. The training monitor instead averaged the F1 of each of its 16 sequences. That per-sequence mean is reported
  beside the pooled value.
- **Boundary recall and precision** match predicted label changes one to one to reference changes within 0.5 s
  (5 samples).
- A **short** run has at most 16 samples (1.6 s), as in the paired round.
- An **X → Y → X middle** is a reference run whose two neighbours share a view that differs from its own. In an
  **A4C/A5C middle**, X and Y are A4C and A5C.
- The **readouts** are the following. E1 argmax is the per-frame argmax of the cached E1 probe probabilities, with no
  smoothing. FACT i, u and U are the argmax of each block's frame-branch class probabilities. The FACT official output
  is the official decode of the last block. ASRF last stage is the argmax of ASRF's last stage, and the ASRF official
  output is its refinement.
- The **partition entering U** is the set of runs of the u block's argmax. The U block averages the frame features
  inside each interval, then runs a GRU, cross-attention and its frame branch.
- **Conditions A, B and C** change only that partition. A keeps it. B adds every reference view change that lies
  strictly inside an interval of A. C adds, in each interval of A, as many cuts as B adds there, evenly spaced. The
  weights, the inputs to U and the rest of the decode stay the same.
- In "x − y", a **fixed** run is matched under x but not under y, and a **new** miss is matched under y but not
  under x.
- A run's **own share** is the largest share of one partition interval that the run fills. A run whose own share is
  below one half is the minority in every interval it touches.

## 1. Data and checks

- **Rows.** The cache index has 3,885 rows. The protocol's training allowlist keeps 3,364 of them, and the other 521
  were dropped before any label was parsed. The training bank rebuilt from the kept rows has the protocol's digest.
- **Batches.** In each seed, all 375 batches of epoch 49 equal the saved plan. The streams are the 1,500 bank streams,
  each once. The 1,500 native windows come from 1,500 distinct recordings (442, 456 and 456 sessions), with 9 to 256
  samples (median 40 to 41.5).
- **Units.** The streams hold 6,463 reference runs from 2,873 recordings in 499 sessions. 1,726 runs are short, 873
  are X → Y → X middles, and 99 are A4C/A5C middles.
- **Check on the 16 fixed sequences.** The last monitor values of all six runs were reproduced exactly. For each FACT
  seed, every equality check held on all 16 sequences, and B and C were also executed on the five native windows. The
  weights did not change.
- **Full run.** In both runs, every check held on all 3,000 sequences of each seed. FACT's official forward, the
  ordinary pass and condition A gave identical tensors, and a repeated A reproduced the first. B and C left every input
  to the U block unchanged. They were executed on all 1,500 streams. A native window has one reference label, so its B
  and C partitions equal A's, and its A output was reused.
- **Repeat.** The repeated run's 9,000 per-sequence records equal the first run's, field by field.
- **Readouts.** In every sequence, the argmax of each block's class probabilities equals the argmax of its logits.
- **Cost.** `summary.json` counts 0.239 GPU hours for the repeated run (the check took 13 s and the three seeds 271 to
  280 s each). The first run counted 0.240 hours, its two later summary passes about 20 s each and the fact-check's
  decoding about 46 s. So the round used about 0.51 of its 1 GPU hour.

## 2. Fit of the whole library (question 1)

Each cell gives seeds 0 / 1 / 2. E1 reads the cache, so it is the same in every seed on the streams. The native
windows differ by seed.

**Constructed streams (1,500 per seed).**

| Readout | Frame error | Segment F1@50 (pooled) | Per-sequence F1, mean | Sequences at F1 = 1 | Matched reference runs (of 6,463) | Exact transcripts (of 1,500) | Boundary recall | Boundary precision |
|---|---|---|---|---|---|---|---|---|
| E1 argmax | 0.065 / 0.065 / 0.065 | 0.518 / 0.518 / 0.518 | 0.600 / 0.600 / 0.600 | 255 / 255 / 255 | 5672 / 5672 / 5672 | 255 / 255 / 255 | 0.997 / 0.997 / 0.997 | 0.355 / 0.355 / 0.355 |
| FACT i block | 0.029 / 0.030 / 0.029 | 0.954 / 0.953 / 0.952 | 0.959 / 0.959 / 0.958 | 1225 / 1240 / 1235 | 6275 / 6265 / 6263 | 1259 / 1272 / 1271 | 0.968 / 0.966 / 0.967 | 0.925 / 0.924 / 0.923 |
| FACT u block | 0.027 / 0.027 / 0.027 | 0.966 / 0.968 / 0.967 | 0.969 / 0.969 / 0.969 | 1287 / 1292 / 1285 | 6278 / 6281 / 6271 | 1318 / 1320 / 1317 | 0.967 / 0.967 / 0.965 | 0.954 / 0.956 / 0.956 |
| FACT U block (frame branch) | 0.027 / 0.027 / 0.029 | 0.969 / 0.971 / 0.970 | 0.971 / 0.972 / 0.970 | 1298 / 1300 / 1294 | 6288 / 6288 / 6272 | 1324 / 1323 / 1324 | 0.968 / 0.968 / 0.965 | 0.958 / 0.962 / 0.962 |
| FACT official output | 0.027 / 0.027 / 0.030 | 0.971 / 0.970 / 0.968 | 0.972 / 0.971 / 0.969 | 1306 / 1302 / 1295 | 6281 / 6271 / 6251 | 1340 / 1328 / 1326 | 0.967 / 0.966 / 0.964 | 0.964 / 0.966 / 0.966 |
| ASRF last stage | 0.002 / 0.002 / 0.002 | 0.996 / 0.997 / 0.997 | 0.997 / 0.997 / 0.997 | 1478 / 1482 / 1482 | 6445 / 6451 / 6447 | 1480 / 1482 / 1482 | 0.997 / 0.998 / 0.997 | 0.995 / 0.995 / 0.995 |
| ASRF official output | 0.002 / 0.002 / 0.002 | 0.997 / 0.998 / 0.998 | 0.998 / 0.998 / 0.998 | 1483 / 1487 / 1489 | 6444 / 6449 / 6449 | 1484 / 1488 / 1489 | 0.997 / 0.997 / 0.997 | 0.998 / 0.998 / 0.998 |

**Native windows (1,500 per seed).** A window has no reference change, so every predicted change is a disagreement
with the clip label.

| Readout | Frame error | Matched windows (of 1,500) | Windows with a label change | Label changes |
|---|---|---|---|---|
| E1 argmax | 0.0492 / 0.0471 / 0.0477 | 1273 / 1290 / 1289 | 570 / 557 / 558 | 3960 / 3894 / 3883 |
| FACT i block | 0.0101 / 0.0084 / 0.0113 | 1483 / 1487 / 1481 | 51 / 41 / 38 | 121 / 94 / 101 |
| FACT u block | 0.0047 / 0.0045 / 0.0071 | 1492 / 1493 / 1490 | 12 / 9 / 14 | 20 / 11 / 21 |
| FACT U block (frame branch) | 0.0044 / 0.0044 / 0.0068 | 1493 / 1493 / 1491 | 12 / 7 / 12 | 18 / 9 / 21 |
| FACT official output | 0.0050 / 0.0045 / 0.0073 | 1492 / 1493 / 1491 | 12 / 8 / 11 | 16 / 10 / 16 |
| ASRF last stage | 0.0021 / 0.0024 / 0.0033 | 1497 / 1496 / 1494 | 8 / 8 / 13 | 18 / 18 / 24 |
| ASRF official output | 0.0018 / 0.0024 / 0.0023 | 1497 / 1496 / 1497 | 4 / 4 / 4 | 6 / 4 / 4 |

- **E1 flickers.** E1's per-frame argmax predicts 15,421 runs for the 6,463 reference runs of the streams. That is why
  its pooled F1 is low while its frame error is 0.065.
- **Error types.** In the streams, A5C read as A4C and A4C read as A5C make up 83% to 85% of FACT's wrong samples
  (4,308 to 4,898 of 5,192 to 5,761). For ASRF they make up 87% to 90% of far fewer wrong samples (351 to 422). Every
  other type has at most 132 wrong samples for FACT and 43 for ASRF.
- **Lengths of FACT's missed stream runs.** 14 to 16 have 1 to 10 samples, 73 to 86 have 11 to 16, 88 to 103 have
  17 to 40, 6 to 9 have 41 to 80, and none is longer.
- **Across seeds.** The streams are the same in every seed, so a run can be followed across seeds. 141 runs (111
  recordings, 91 sessions) are missed by FACT's official output in all three seeds, and 257 (206 recordings, 149
  sessions) in at least one. For ASRF the counts are 12 (11 recordings, 10 sessions) and 19 (17 recordings, 15
  sessions).

## 3. Where short runs are lost (question 2)

**Matched reference runs by group.**

| Readout | Short runs (of 1,726) | X → Y → X middles (of 873) | A4C/A5C X → Y → X middles (of 99) | Native windows (of 1,500) |
|---|---|---|---|---|
| E1 argmax | 1617 / 1617 / 1617 | 772 / 772 / 772 | 72 / 72 / 72 | 1273 / 1290 / 1289 |
| FACT i block | 1648 / 1642 / 1643 | 827 / 822 / 824 | 69 / 67 / 68 | 1483 / 1487 / 1481 |
| FACT u block | 1639 / 1640 / 1632 | 822 / 827 / 822 | 64 / 69 / 65 | 1492 / 1493 / 1490 |
| FACT U block (frame branch) | 1644 / 1644 / 1632 | 827 / 830 / 822 | 68 / 71 / 65 | 1493 / 1493 / 1491 |
| FACT official output | 1638 / 1635 / 1626 | 823 / 825 / 821 | 62 / 65 / 61 | 1492 / 1493 / 1491 |
| ASRF last stage | 1720 / 1723 / 1720 | 869 / 870 / 869 | 97 / 98 / 97 | 1497 / 1496 / 1494 |
| ASRF official output | 1719 / 1723 / 1722 | 868 / 869 / 869 | 96 / 97 / 97 | 1497 / 1496 / 1497 |

**Runs missed by FACT's official output, by the earliest readout where they were unmatched.** The chain is E1 argmax,
the i block, the u block, the U frame branch and the official output. A run first unmatched at one readout may be
matched again at a later one and lost once more at the end.

| Earliest unmatched at | Short runs | A4C/A5C middles | All stream runs |
|---|---|---|---|
| E1 argmax | 30 / 34 / 34 | 13 / 13 / 13 | 110 / 116 / 129 |
| FACT i block | 35 / 36 / 37 | 15 / 14 / 15 | 46 / 47 / 47 |
| FACT u block | 18 / 14 / 21 | 7 / 5 / 6 | 20 / 18 / 23 |
| FACT U block (frame branch) | 0 / 0 / 1 | 0 / 0 / 0 | 0 / 0 / 1 |
| Only the official output | 5 / 7 / 7 | 2 / 2 / 4 | 6 / 11 / 12 |
| **Missed by the official output** | **88 / 91 / 100** | **37 / 34 / 38** | **182 / 192 / 212** |

**Recovery after a first loss (short runs).** Of the runs first unmatched at each readout, the official output matches
these numbers again. At E1 argmax it matches 79, 75 and 75 of 109. At the i block it matches 5, 9 and 7 of 40, 45 and
44. At the u block it matches 0, 0 and 1 of 18, 14 and 22. At the U frame branch it matches 1, 2 and 4 of 1, 2 and 5.

**Lost and recovered between readouts (short runs).**

| Readouts | Lost | Recovered |
|---|---|---|
| E1 argmax → FACT i block | 40 / 45 / 44 | 71 / 70 / 70 |
| FACT i block → u block | 20 / 15 / 22 | 11 / 13 / 11 |
| FACT u block → U frame branch | 3 / 3 / 8 | 8 / 7 / 8 |
| FACT U frame branch → official output | 12 / 15 / 15 | 6 / 6 / 9 |
| E1 argmax → FACT official output | 58 / 57 / 66 | 79 / 75 / 75 |
| E1 argmax → ASRF official output | 4 / 1 / 2 | 106 / 107 / 107 |

For A4C/A5C middles, the i block loses 17 to 18 middles that E1's argmax matched and recovers 13 to 14. The official
output loses 6 to 8 middles that the U frame branch matched and recovers 0 to 4. `report_tables.md` lists every pair
of readouts for each of these groups.

**What the misses look like at the official output.** These descriptors are not causes, and one run can carry several.
- **No predicted run at IoU 0.5.** 75 to 85 of the 88 to 100 missed short runs have no predicted run of any view at
  IoU 0.5. The rest are covered at IoU 0.5 or more by a run of another view.
- **Shape of the same-view prediction.** For 46 to 56 missed short runs, no sample inside the run is predicted as its
  own view. For 21 to 26 a longer run of the same view covers it, for 8 to 14 a shorter one lies inside it, and 6 to
  10 overlap one of its ends.
- **No competition.** No same-view run at IoU 0.5 was taken by another reference run. This cannot happen here,
  because a predicted run cannot reach IoU 0.5 with two reference runs of its view.
- **Views inside.** The view predicted on most samples of a missed short run is A5C for 42 to 50 runs and A4C for 16
  to 17. For 28 to 33 runs it is the run's own view, but the same-view prediction overlaps the run with an IoU below
  0.5.
- **Tokens.** For 77, 79 and 86 of the missed short runs, a non-null U token of the run's own view was active in the
  sequence. For only 56, 53 and 59 did the sequence hold at least as many such tokens as reference runs of that view.
  So a token of the run's view was often present, but these counts do not show that one was free for the run.

## 4. The partition entering U

**Intervals.** The partition entering U has 6,510 to 6,532 intervals in the streams. 5,447 to 5,541 cover one
reference run, 920 to 986 cover two, 53 to 82 cover three, and 5 cover four or more. Of the 978 to 1,063 intervals
that mix views, the median share of the smaller views is 0.067 to 0.069. Most mixed intervals therefore cross a
reference change by only a few samples.

**Exact alignment (fixed before the run).** A run has "own intervals" when both of its ends are partition boundaries.

| Group | Own interval | Own intervals, cut inside | Inside a larger interval | Spans intervals, an end inside one |
|---|---|---|---|---|
| Short runs (1,726) | 1111 / 1152 / 1116 | 3 / 3 / 0 | 437 / 396 / 428 | 175 / 175 / 182 |
| A4C/A5C middles (99) | 2 / 2 / 1 | 0 / 0 / 0 | 44 / 46 / 45 | 53 / 51 / 53 |

Both ends of an A4C/A5C middle are partition boundaries in only 1 to 2 of the 99 middles, and 18, 25 and 22 of their
198 inner ends are (9% to 13%). Of the 541 changes between A4C and A5C in the streams, 87, 97 and 83 (15% to 18%) lie
on a partition boundary, against 77% to 79% of all 4,963 reference changes. These counts were added after the first
run. The categories of the table put a run inside an own-view interval that is a few samples too long together with a
run absorbed by a neighbour. The next table separates the two.

**Own share (added after the first run).** The official output matched these runs under A, B and C. The last column
also counts the runs the U frame branch matched.

| Group | Minority in every interval | Their recordings / sessions | Official output A / B / C | U frame branch A / B / C |
|---|---|---|---|---|
| All stream runs | 85 / 93 / 93 | 78/70, 85/75, 86/76 | 0/40/27, 0/42/30, 0/36/24 | 2/6/4, 3/4/3, 2/2/0 |
| Short runs | 58 / 63 / 67 | 55/51, 60/55, 63/57 | 0/26/14, 0/27/16, 0/28/16 | 2/3/3, 3/4/3, 2/2/0 |
| A4C/A5C middles | 28 / 26 / 28 | 27/27, 25/25, 27/27 | 0/14/8, 0/10/5, 0/8/2 | 0/1/1, 2/1/0, 1/1/0 |

- **Every absorbed middle is an A4C/A5C middle.** The X → Y → X middles that are the minority in every interval are
  exactly the A4C/A5C middles of the last row.
- **Runs that fill the majority of an interval.** The official output matches 6,281, 6,271 and 6,251 of 6,370 to 6,378
  such runs under A. B adds 22 to 34 and C adds 7 to 9. So about half of A's misses (97 to 119) are runs that do fill
  the majority of an interval.
- **Labels inside an interval.** The official output did not change its label inside any of the 6,510 to 6,532 stream
  intervals. Inside the 1,511 to 1,521 native intervals it changed once (seed 2). The official decode gives every frame
  of one interval the same token and mixes in the frame branch at weight 0.1. That mix moved the label of one whole
  stream interval (seed 1) off its token's class, and of no other stream interval. The U frame branch
  changed its label inside 493, 569 and 670 stream intervals. This describes the frozen models on these sequences. It
  does not prove that a merged run can never be recovered.
- **Tokens.** The number of non-null U tokens equals the number of reference runs in 1,178, 1,125 and 1,188 of the
  1,500 streams. It is larger in 264 to 337 streams and smaller in 28 to 48. For the i and u blocks it is equal in 895
  to 1,010 streams. On 5 to 9 native windows every final token was null, and the decode fell back to the frame branch.

## 5. The intervention at the last U block (question 3)

B adds 1,141, 1,041 and 1,145 cuts in 768, 725 and 769 of the 1,500 streams. C adds the same number in each interval.
Native windows get no cut.

**Official output on the constructed streams.**

| Condition | Segment F1@50 (pooled) | Boundary recall | Boundary precision | Matched reference runs (of 6,463) |
|---|---|---|---|---|
| A | 0.9707 / 0.9704 / 0.9679 | 0.9666 / 0.9655 / 0.9637 | 0.9636 / 0.9657 / 0.9657 | 6281 / 6271 / 6251 |
| B | 0.9764 / 0.9758 / 0.9739 | 0.9825 / 0.9803 / 0.9794 | 0.9696 / 0.9682 / 0.9687 | 6343 / 6337 / 6321 |
| C | 0.9741 / 0.9738 / 0.9714 | 0.9680 / 0.9682 / 0.9641 | 0.9602 / 0.9616 / 0.9618 | 6315 / 6310 / 6284 |
| B − A | +0.0057 / +0.0055 / +0.0059 | +0.0159 / +0.0147 / +0.0157 | +0.0059 / +0.0024 / +0.0030 | |
| C − A | +0.0034 / +0.0034 / +0.0035 | +0.0014 / +0.0026 / +0.0004 | −0.0034 / −0.0042 / −0.0039 | |
| B − C | +0.0024 / +0.0021 / +0.0025 | +0.0145 / +0.0121 / +0.0153 | +0.0094 / +0.0066 / +0.0069 | |

**U frame branch.** B − A is +0.0007, +0.0006 and −0.0003 in segment F1@50, C − A is −0.0005, −0.0001 and −0.0015,
and B − C is +0.0012, +0.0007 and +0.0013.

**Matched, fixed and new runs of the official output by group.** Subsets report only matched runs and fixed and new
runs. They have no F1 or precision, because a predicted run cannot be assigned to a subset.

| Group | Matched under A | B | C | B − A fixed / new | C − A fixed / new | B − C fixed / new |
|---|---|---|---|---|---|---|
| All runs (6,463) | 6281 / 6271 / 6251 | 6343 / 6337 / 6321 | 6315 / 6310 / 6284 | 74/12, 72/6, 79/9 | 49/15, 53/14, 55/22 | 36/8, 36/9, 48/11 |
| Short runs (1,726) | 1638 / 1635 / 1626 | 1676 / 1666 / 1663 | 1659 / 1652 / 1642 | 43/5, 37/6, 44/7 | 26/5, 24/7, 27/11 | 20/3, 22/8, 26/5 |
| X → Y → X middles (873) | 823 / 825 / 821 | 838 / 837 / 836 | 824 / 829 / 821 | 18/3, 14/2, 17/2 | 9/8, 8/4, 8/8 | 14/0, 11/3, 16/1 |
| A4C/A5C middles (99) | 62 / 65 / 61 | 78 / 79 / 77 | 66 / 69 / 64 | 18/2, 14/0, 16/0 | 9/5, 7/3, 8/5 | 12/0, 10/0, 13/0 |

**Sources of the changes.**
- **Short runs, B − A.** The fixed runs come from 43, 37 and 43 recordings in 40, 37 and 40 sessions. The new misses
  come from 5 to 7 recordings.
- **Short runs, B − C.** The fixed runs come from 20, 22 and 26 recordings in 20, 21 and 25 sessions. The new misses
  come from 3, 8 and 5 recordings.
- **A4C/A5C middles, B − C.** The fixed middles come from 12, 10 and 13 recordings, each in its own session.

**Reading.**
- **Direction.** All three seeds point the same way. B matches more runs than A and C in every group. C matches more
  runs than A overall and has a higher pooled F1, but its boundary precision is lower in every seed. For X → Y → X
  middles, C − A is +1, +4 and 0 runs.
- **Absorbed runs.** Much of B's extra recovery is in runs that were the minority in every interval entering U. Under
  B the official output matches 36 to 42 of them, against 24 to 30 under C (section 4).
- **The frame branch.** The partition changes the official output far more than the frame branch. The decode points
  to the per-interval token assignment as the route, but this route was not measured on its own.
- **Not shown.** These results do not show that the reference positions can be predicted from the images. They are
  not an achievable gain or a bound on the model, and they cannot be assigned to the averaging or the attention alone
  (section 6).

## 6. What this round does not show

- **No generalisation.** Every number is a fit to the models' own training sequences. Nothing here is a development-set
  result, and nothing here changes the paired round's decision.
- **Not a model.** B and C read the reference partition, so they cannot run on a new video. Their outputs are
  diagnostics and must not be reported as model results.
- **Not the goal.** Even under B, FACT stays below ASRF on short runs and A4C/A5C middles on the same sequences. So B is
  no evidence that FACT could reach the goal of 0.944 and 0.982.
- **Averaging against attention.** A, B and C change the sequence of segment features that the GRU and the attention
  read, so the effect cannot be assigned to the averaging, the GRU or the attention alone.
- **Training time.** The library's misfit does not show that longer training would close it. No loss was split and no
  optimiser was searched.
- **Next step.** A training method that does not read reference boundaries, with a fixed-subdivision control, would be
  the lead's decision. This round starts no training.

## Commands

Run on the server in the project `.venv`, one GPU job at a time, with `ECHO_GPU_MEM_FRACTION=0.6`.

```bash
C=configs/experiments/fact_localisation_v1.yaml
python -m pytest tests/test_fact_localisation.py -q
python scripts/diagnose_fact_localisation.py --config $C --prepare   # allowlist, bank digest, epoch-49 batches, counts
python scripts/diagnose_fact_localisation.py --config $C --check     # the 16 fixed sequences, every equality check, budget
python scripts/diagnose_fact_localisation.py --config $C --run       # every sequence; reuses finished seeds; summary
```

## Provenance

- **Code.** The reported run used the code at c062220, under run identity 1bdea50968984e60. The identity covers the
  configuration, the inputs, a fixed list of imported code files, the source of every inference function and their
  constants. The first run used the code at 1f09d90 (identity 58bb82909520492c), and its summary code at 9aafb99.
- **Configuration.** `configs/experiments/fact_localisation_v1.yaml` pins the paired round's configuration (5caffd56),
  summary (32c2665f), monitor set (5fb823f0), three plans and six frozen final weights. Every digest was checked before
  use.
- **Server records.** The per-sequence records are in `runs/fact_localisation_v1/seed{0,1,2}_sequences.jsonl.gz`
  (sha256 prefixes d026f6e3034e, c60eb9305fcb and 9879d824fe62). Each holds the reference labels, the source
  fragments, every readout, the tokens, the partition entering U, the final assignment and the B and C outputs.
  `check.json` holds the check. The first run's records and check are kept in
  `runs/fact_localisation_v1/superseded_58bb8290/`. Recording and session ids stay on the server.

## Files

- `README.md` is this file. `code_review.md` is the record of the code review before the first run, with a note on the
  index read.
- `preparation.json` holds the row filtering, the batch checks and the counts with sources.
- `summary.json` holds every table of this README and more. That includes every group, the miss descriptors, the
  transitions between every pair of readouts, the error types, the lengths, the tokens and the across-seed counts.
- `report_tables.md` holds the generated tables.
- None of these files contains a recording or session id.
