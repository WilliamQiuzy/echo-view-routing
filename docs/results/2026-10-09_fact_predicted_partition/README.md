# Predicted-partition FACT (2026-10-09 to 10-10)

This round retrained the official FACT (Lu and Elhamifar, CVPR 2024) twice per seed on the frozen E1 features. Each
time, one more step was added in the last U block. The Uniform group cut every official run every 8 samples. The
Predicted group added the cuts of a small boundary head, trained beforehand and frozen. Both groups were then
evaluated once on the two development halves, beside the paired round's FACT ("Original FACT"), its ASRF, the simple
references and a Boundary head + frame classifier control. FACT is prior work. The refinement is an extension of it,
and this round claimed no gain or novelty in advance. The task sheet is
`docs/superpowers/plans/2026-10-09-fact-predicted-partition.md`.

**What kind of evidence this is.**
- The heads and both FACT groups trained on the EV9V training split only, through the allowlist read of the cache
  index. Nothing was chosen from the development halves during training.
- The development halves have been used in many earlier rounds, so this evaluation is not an independent validation.
- The hold-out is by session, not by patient.
- Constructed streams are spliced from separate recordings and are not real scans. They reuse recordings, so their
  units are not independent cases. The three seeds share the same cases.
- Native clips carry the public clip label, not expert frame labels.
- The test split and EchoNet-Pediatric were not read.

**Order of work.**
- 26069b5 committed the task sheet, the configuration, the code and its tests before any head or FACT metric existed.
- An independent implementation review passed the code with one medium and one low finding, both in the decision
  table. d814ff6 fixed both and added the review record (`code_review.md`).
- `--prepare`, the head training, the head freezing and both smoke checks passed, and the queue then started the six
  formal runs. 72624ce committed the check records a minute after the queue started, before any formal output existed.
- After all six runs had finished, the evaluation decoded every system once.
- An independent fact-check recounted the results from the saved per-sequence outputs and found every number right.
  It also re-decoded 900 sequences bit for bit. Its wording corrections are in this README.

## Answers

Values are given for seeds 0, 1 and 2 in that order. A range "a to b" gives the smallest and the largest of the three
seeds. "Selection" and "calibration" are the two development halves.

1. **The decision.** The readings of the decision table were fixed in the configuration before any evaluation.
   - **Predicted against Uniform (the primary comparison).** No consistent difference was detected in either half,
     and the seeds point in different directions. In segment F1@50, Predicted − Uniform is −0.010, +0.009 and +0.016
     on selection and −0.001, +0.004 and −0.004 on calibration. Every interval includes zero.
   - **The row that applies.** Only row 4 applies: the result is uncertain. So no threshold, epoch, grid or seed may
     be changed or picked to rescue it. Learned boundaries cannot be claimed to add value over a fixed grid.
   - **Against Original FACT.** Neither group consistently beats Original FACT in either half. Uniform − Original
     FACT is negative in all six seed-half cells (−0.022 to −0.0005), and one interval excludes zero (seed 2,
     selection). Predicted − Original FACT is negative in four of the six cells, and one interval excludes zero (seed
     0, selection).
   - **The goal.** Neither group reaches it. The three-seed means are 0.891 and 0.962 for Uniform and 0.896 and 0.962
     for Predicted. The goal is 0.944 and 0.982. Original FACT has 0.905 and 0.964.
   - **Constraints.** Uniform meets every constraint. Predicted fails one cell: seed 1 has native macro-F1 0.966 on
     selection, against the floor of 0.973.
2. **Short runs.** As point estimates, the new groups keep a few more short runs than Original FACT in 9 of the 12
   seed-half cells. Uniform ties Original FACT in seed 2 in both halves, and Predicted keeps one fewer in seed 2 on
   calibration.
   - **Short runs (at most 16 samples).** On selection, Uniform keeps 128 to 131 of 141 and Predicted 129 to 131,
     against 127 to 128 for Original FACT, 123 to 125 for ASRF and 129 for E1 + B3. On calibration they keep 125 to
     129 and 124 to 128 of 134, against 123 to 125 for Original FACT, 125 to 128 for ASRF and 125 for E1 + B3.
   - **A4C/A5C X → Y → X middles.** There are only 7 per half. On selection Uniform keeps 3 to 4, Predicted 4 to 6,
     Original FACT 1 to 3, ASRF 3 to 4 and E1 + B3 5. On calibration they keep 6 to 7, 4 to 7, 5 to 7, 7 and 7.
   - **These counts are small.** The short runs a FACT system misses come from 4 to 11 recordings per seed and half
     (ASRF's and the control's from 6 to 15), and the counts decide nothing.
3. **Extra cuts.** The new groups cut more often where the reference does not change, and this outweighs their small
   gains in short runs.
   - **Boundary precision.** On selection it is 0.885 to 0.901 for Uniform and 0.901 to 0.913 for Predicted, against
     0.925 to 0.928 for Original FACT. Boundary recall rises a little, to 0.950 to 0.958 against 0.940 to 0.950.
   - **Cuts at same-view joins.** Uniform cuts 0.065 to 0.096 of these joins on selection, against 0.035 to 0.052 for
     Original FACT. Predicted cuts 0.052 to 0.070.
   - **Native clips.** Label switches inside native clips stay rare for both groups (0 to 0.007 per clip on
     selection, 0.021 to 0.034 on calibration).
4. **Where the official output's misses arise.** This was tracked through E1's argmax, FACT's i, u and U blocks and
   the official output, on the development multi streams.
   - **Most misses were already unmatched in E1's argmax.** E1's per-frame argmax flickers, and FACT's first block
     repairs most of its misses. Of the runs that the official output still misses, most were first unmatched at E1
     in 17 of the 18 system-seed-half cells (exactly half in the last, Predicted seed 2 on calibration). Only 5 to 10
     per seed on selection and 1 to 3 on calibration were first lost at the i or u block.
   - **The fixed reading of row 3.** Row 3 asks whether most misses were first lost at the i or u block. That did not
     hold for any group, so row 3 does not apply under the fixed reading. Under this reading Predicted did lose fewer
     runs first at i or u than Original FACT in every seed and half (6, 7 and 6 against 10 on selection; 1, 2 and 1
     against 3 on calibration), but it has no full-system gain.
   - **Other readings (added after the evaluation).** These readings were not fixed beforehand, so they do not replace
     the decision above, and they disagree with each other.
     - *First unmatched at E1, i or u.* Most misses fall here in every seed and half for both new groups, and neither
       group reduces these counts in every seed and half against Original FACT. Under this reading row 3's conditions
       would hold. It also counts runs that E1 missed, i and u matched, and U or the official output lost again
       (Uniform 4, 6 and 8 on selection and 0, 2 and 3 on calibration; Predicted 9, 3 and 4 and 3, 4 and 3).
     - *Already missed by the u block's argmax, the U block's own input.* The independent fact-check counted this from
       the saved per-sequence outputs. Predicted then misses only 7 of 14 runs this way in seed 0 and 6 of 16 in seed
       2 on calibration, which is not "most". Under this reading row 3's conditions would not hold for Predicted.
5. **The position control (frozen Predicted models).** The head's cuts were moved to even positions inside each
   official run, keeping their number.
   - **No stable direction.** Head positions minus even positions in segment F1@50 is −0.007, +0.019 and +0.010 on
     selection and −0.003, −0.001 and +0.001 on calibration.
   - **Checks.** The weights and every input to the U block stayed unchanged in all 1,013 sequences per seed, and the
     normal output was reproduced afterwards.
6. **The boundary head and its control.**
   - **The head on the development streams.** Its cuts find 0.93 to 0.94 of the view changes on selection and 0.96 on
     calibration, with precision 0.80 to 0.82 and 0.85 to 0.89. It cuts 0.12 to 0.15 of the same-view joins on
     selection and proposes 0.21 to 0.32 cuts per native clip.
   - **Boundary head + frame classifier.** This control names each head interval by the mean frozen E1 probabilities.
     It reaches 0.880 to 0.895 on selection and 0.956 on calibration. Predicted − control is positive in all six
     cells (+0.002 to +0.017), but every interval includes zero, so no consistent difference was detected.
7. **Cost.** The round used 17.8 GPU hours of its 24-hour cap.
   - **Training.** The six formal runs took 2.78 to 2.83 hours each, 16.8 hours in all.
   - **Heads and checks.** The three heads took about 1.4 minutes each, the two smoke checks 0.35 hours each and the
     evaluation 0.19 hours.
   - **Model size.** The head adds 110,785 parameters to FACT's 10,215,424.
   - **Decoding.** It takes 0.018 to 0.020 s per sequence for all three FACT systems.
   - **Partition sizes.** Uniform averages 17.5 to 17.6 intervals per stream in the U block, Predicted 5.6 to 5.9 and
     Original FACT 4.3 to 4.4.

## Terms

- A **difference is consistent** in a half when all three seeds' 95% intervals exclude zero strictly on the same side.
  Otherwise no consistent difference is detected, which does not mean the systems are equal.
- The **paired bootstrap** resamples native clips by session and streams by whole stream, with the same draws for both
  systems (2,000 draws, seed 20261002). Streams share recordings, so the intervals are approximate.
- A **reference run** is a run of one view in a multi stream. It is **matched** when a predicted run of the same view
  has an IoU of at least 0.5 with it, one to one. A **short** run has at most 16 samples.
- An **X → Y → X middle** is a reference run whose two neighbours share a view that differs from its own.
- The **partition entering U** is the set of runs of the u block's argmax. Uniform adds cuts at s + 8, s + 16, and so
  on, inside every run [s, e). Predicted adds the frozen head's cuts inside every run. Only positions enter the
  partition, never a reference label.
- The **boundary head** reads [x_(t−1), x_t, x_t − x_(t−1)] of the raw E1 features (a linear layer and two
  convolutions). It was trained on the training split for 7,500 updates and then frozen. It cuts where its sigmoid is
  at least 0.5, after non-maximum suppression with radius 2.
- The **first unmatched readout** of a missed run is the earliest of E1's argmax, the i, u and U blocks' frame-branch
  argmax and the official output at which it is unmatched. The configuration's text names i, u, U and the official
  output; the code, committed before any metric, also puts E1 first, as the localisation round did. Without E1, row 3
  still would not apply: for Predicted on calibration, only 7 of 14 misses in seed 0 and 6 of 16 in seed 2 would
  fall at i or u.

## 1. Systems

Each cell gives the selection half, then the calibration half. Original FACT, ASRF and the three references reproduce
the paired round's rows exactly.

| System | Segment F1@50 | Boundary recall | Boundary precision | Native macro-F1 | Label switches per native clip | Cuts at same-view joins | A5C read as A4C (clips) | A4C read as A5C (clips) |
|---|---|---|---|---|---|---|---|---|
| Original FACT s0 | 0.906 / 0.967 | 0.948 / 0.982 | 0.926 / 0.968 | 0.989 / 0.986 | 0.000 / 0.034 | 0.052 / 0.025 | 0 / 1 | 0 / 1 |
| Original FACT s1 | 0.898 / 0.961 | 0.950 / 0.975 | 0.928 / 0.968 | 0.978 / 0.986 | 0.000 / 0.030 | 0.043 / 0.030 | 1 / 1 | 0 / 1 |
| Original FACT s2 | 0.911 / 0.965 | 0.940 / 0.972 | 0.925 / 0.972 | 0.966 / 0.993 | 0.000 / 0.034 | 0.035 / 0.025 | 2 / 0 | 0 / 1 |
| Uniform s0 | 0.893 / 0.965 | 0.958 / 0.980 | 0.901 / 0.944 | 0.978 / 0.993 | 0.007 / 0.021 | 0.074 / 0.025 | 1 / 0 | 0 / 1 |
| Uniform s1 | 0.890 / 0.960 | 0.950 / 0.975 | 0.885 / 0.942 | 0.989 / 0.983 | 0.000 / 0.021 | 0.096 / 0.040 | 0 / 1 | 0 / 1 |
| Uniform s2 | 0.889 / 0.962 | 0.955 / 0.967 | 0.899 / 0.953 | 0.989 / 0.986 | 0.000 / 0.030 | 0.065 / 0.035 | 0 / 1 | 0 / 1 |
| Predicted s0 | 0.883 / 0.964 | 0.958 / 0.987 | 0.901 / 0.963 | 0.978 / 0.983 | 0.000 / 0.025 | 0.070 / 0.025 | 1 / 1 | 0 / 1 |
| Predicted s1 | 0.899 / 0.964 | 0.958 / 0.987 | 0.901 / 0.954 | 0.966 / 0.983 | 0.000 / 0.030 | 0.057 / 0.015 | 2 / 1 | 0 / 1 |
| Predicted s2 | 0.905 / 0.958 | 0.958 / 0.977 | 0.913 / 0.949 | 0.978 / 0.993 | 0.000 / 0.034 | 0.052 / 0.035 | 1 / 0 | 0 / 1 |
| ASRF s0 | 0.878 / 0.972 | 0.935 / 0.987 | 0.927 / 0.975 | 0.930 / 0.986 | 0.011 / 0.004 | 0.065 / 0.035 | 4 / 1 | 1 / 1 |
| ASRF s1 | 0.888 / 0.972 | 0.940 / 0.985 | 0.937 / 0.978 | 0.954 / 0.986 | 0.007 / 0.004 | 0.048 / 0.035 | 3 / 1 | 0 / 1 |
| ASRF s2 | 0.882 / 0.966 | 0.942 / 0.982 | 0.942 / 0.980 | 0.954 / 0.986 | 0.011 / 0.004 | 0.065 / 0.030 | 3 / 1 | 0 / 1 |
| Boundary head + frame classifier s0 | 0.880 / 0.956 | 0.927 / 0.955 | 0.910 / 0.967 | 0.978 / 0.986 | 0.077 / 0.080 | 0.057 / 0.025 | 1 / 1 | 0 / 1 |
| Boundary head + frame classifier s1 | 0.895 / 0.956 | 0.935 / 0.957 | 0.930 / 0.962 | 0.978 / 0.986 | 0.095 / 0.084 | 0.052 / 0.020 | 1 / 1 | 0 / 1 |
| Boundary head + frame classifier s2 | 0.888 / 0.956 | 0.929 / 0.957 | 0.929 / 0.957 | 0.978 / 0.986 | 0.077 / 0.080 | 0.043 / 0.030 | 1 / 1 | 0 / 1 |
| E1 + B3 | 0.894 / 0.959 | 0.955 / 0.977 | 0.943 / 0.951 | 0.978 / 0.986 | 0.046 / 0.068 | 0.039 / 0.020 | 1 / 1 | 0 / 1 |
| E0 + B3 (historical strongest on selection) | 0.934 / 0.934 | 0.961 / 0.955 | 0.995 / 0.969 | 0.965 / 0.966 | 0.014 / 0.034 | 0.004 / 0.020 | 3 / 4 | 0 / 0 |
| B7 + B3 (official EchoViewCLIP posteriors) | 0.880 / 0.972 | 0.955 / 0.975 | 0.890 / 0.963 | 0.957 / 0.993 | 0.063 / 0.042 | 0.057 / 0.035 | 2 / 1 | 1 / 0 |

G1 + ASRF, the higher calibration reference, has 0.904, 0.920 and 0.915 on selection and 0.972, 0.974 and 0.976 on
calibration (stored rows).

**The U block's frame branch.** Read on its own, Predicted's frame branch reaches 0.904, 0.921 and 0.918 on selection
and 0.971, 0.961 and 0.968 on calibration. That is higher than its official output on selection. This readout was not a
decision system, and its intervals were not computed.

## 2. Comparisons

Segment F1@50 differences with 95% intervals, for seeds 0, 1 and 2.

| Comparison | Selection | Calibration | Verdict (selection / calibration) |
|---|---|---|---|
| Predicted − Uniform | −0.0097 [−0.035, +0.016]; +0.0093 [−0.014, +0.035]; +0.0161 [−0.005, +0.039] | −0.0011 [−0.015, +0.013]; +0.0038 [−0.015, +0.021]; −0.0036 [−0.021, +0.014] | none / none |
| Predicted − Original FACT | −0.0230 [−0.045, −0.002]; +0.0005 [−0.020, +0.023]; −0.0057 [−0.024, +0.014] | −0.0037 [−0.019, +0.013]; +0.0033 [−0.018, +0.024]; −0.0072 [−0.027, +0.011] | none / none |
| Uniform − Original FACT | −0.0132 [−0.033, +0.004]; −0.0088 [−0.031, +0.015]; −0.0218 [−0.043, −0.001] | −0.0026 [−0.018, +0.013]; −0.0005 [−0.015, +0.015]; −0.0036 [−0.014, +0.006] | none / none |
| Predicted − ASRF | +0.0054 [−0.018, +0.029]; +0.0113 [−0.013, +0.037]; +0.0227 [+0.003, +0.045] | −0.0085 [−0.023, +0.006]; −0.0083 [−0.024, +0.007]; −0.0082 [−0.027, +0.010] | none / none |
| Uniform − ASRF | +0.0151 [−0.009, +0.040]; +0.0020 [−0.020, +0.025]; +0.0066 [−0.018, +0.030] | −0.0073 [−0.017, +0.002]; −0.0121 [−0.025, +0.001]; −0.0046 [−0.016, +0.007] | none / none |
| Predicted − (E1 + B3) | −0.0107 [−0.035, +0.014]; +0.0049 [−0.018, +0.029]; +0.0112 [−0.010, +0.036] | +0.0047 [−0.009, +0.018]; +0.0049 [−0.015, +0.022]; −0.0009 [−0.017, +0.014] | none / none |
| Predicted − (E0 + B3) | −0.0511 [−0.083, −0.020]; −0.0356 [−0.067, −0.004]; −0.0292 [−0.057, −0.000] | +0.0297 [+0.007, +0.053]; +0.0298 [+0.008, +0.053]; +0.0240 [−0.001, +0.048] | consistent negative / none |
| Predicted − (B7 + B3) | +0.0034 [−0.022, +0.029]; +0.0190 [−0.001, +0.039]; +0.0253 [+0.005, +0.048] | −0.0085 [−0.026, +0.010]; −0.0083 [−0.027, +0.011]; −0.0141 [−0.033, +0.006] | none / none |
| Predicted − (Boundary head + frame classifier) | +0.0030 [−0.022, +0.029]; +0.0042 [−0.024, +0.035]; +0.0167 [−0.007, +0.044] | +0.0073 [−0.009, +0.026]; +0.0083 [−0.010, +0.027]; +0.0024 [−0.010, +0.016] | none / none |

`summary.json` also holds every comparison's other bootstrap metrics, native changes, segment changes and boundary
changes with their sources.

## 3. Reference units

Matched runs of the multi streams for seeds 0, 1 and 2, selection then calibration. Subsets have no F1 or precision,
because a predicted run cannot be assigned to a subset.

| System | Short runs (141 / 134) | Short A4C or A5C runs (65 / 67) | X → Y → X middles (69 / 80) | A4C/A5C middles (7 / 7) |
|---|---|---|---|---|
| Original FACT | 127,127,128 / 124,123,125 | 54,56,55 / 58,57,59 | 60,60,59 / 79,77,78 | 1,3,3 / 7,5,6 |
| Uniform | 131,129,128 / 129,128,125 | 58,56,55 / 62,61,59 | 63,63,61 / 80,79,78 | 4,4,3 / 7,7,6 |
| Predicted | 131,131,129 / 125,128,124 | 58,58,56 / 59,61,58 | 63,65,65 / 78,79,76 | 4,6,5 / 6,7,4 |
| ASRF | 125,123,125 / 127,128,125 | 53,51,54 / 61,61,60 | 60,60,60 / 79,79,79 | 4,3,4 / 7,7,7 |
| Boundary head + frame classifier | 122,125,123 / 124,124,124 | 51,53,52 / 58,58,58 | 60,63,60 / 77,77,77 | 2,4,2 / 5,5,5 |
| E1 + B3 | 129 / 125 | 56 / 59 | 63 / 80 | 5 / 7 |

The short runs missed by Original FACT come from 9 to 11 recordings on selection and 6 to 8 on calibration. For Uniform
the counts are 8 to 10 and 4 to 8, and for Predicted 8 to 9 and 5 to 9. `cases_a4c_a5c_middles.png` shows every A4C/A5C
middle of the development multi streams, selected by its reference labels alone. It shows the reference, E1's argmax and
every system's official output for every seed. The configuration asked for stage tracks in this figure; they are in
`summary.json` instead (section 4).

## 4. Where the misses arise

Reference runs of the development multi streams that the official output misses, by the first unmatched readout (E1,
i, u, U, official output only), for seeds 0, 1 and 2.

| Half | System | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|---|
| Selection | Original FACT | 30/8/2/0/3 (43) | 34/10/0/0/3 (47) | 29/8/2/0/3 (42) |
| Selection | Uniform | 34/6/2/1/0 (43) | 33/5/0/1/4 (43) | 34/7/2/0/2 (45) |
| Selection | Predicted | 36/6/0/2/4 (48) | 31/7/0/2/0 (40) | 31/6/0/0/2 (39) |
| Calibration | Original FACT | 9/2/1/1/1 (14) | 10/1/2/1/5 (19) | 10/2/1/0/5 (18) |
| Calibration | Uniform | 9/1/1/0/0 (11) | 9/2/0/1/2 (14) | 11/1/2/1/2 (17) |
| Calibration | Predicted | 9/1/0/1/3 (14) | 10/1/1/0/0 (12) | 8/1/0/3/4 (16) |

- **E1 dominates.** E1's argmax misses 92 of the 502 selection runs and 59 of the 518 calibration runs. FACT's
  official output matches 56 to 63 and 48 to 51 of these again. In 17 of the 18 cells, most of its remaining misses
  are runs that E1 had already missed (exactly half in Predicted seed 2 on calibration).
- **Few runs are lost at i or u.** For Original FACT 10 runs per seed on selection and 3 on calibration were first
  lost there. For Uniform 5 to 9 and 2 to 3, and for Predicted 6 to 7 and 1 to 2.
- **On the 16 fixed training sequences.** The same tracking is in `summary.json` under "training monitor".

## 5. Partition, position control and the head

- **Intervals in the U block.** Per development stream, Uniform averages 17.5 to 17.6 intervals, Predicted 5.6 to 5.9
  and Original FACT 4.3 to 4.4. Per native clip they average 7.8 to 8.0, 1.2 to 1.3 and 1.0.
- **Position control.** In segment F1@50 the differences are given in answer 5. Head positions recovered 10, 13 and 12
  runs that even positions missed on selection, and missed 10, 2 and 4 that even positions recovered. On calibration
  these counts are 2, 5 and 7 against 0, 3 and 4.
- **Head cuts on the development streams.** Their precision against view changes is 0.80, 0.81 and 0.82 on selection
  and 0.89, 0.88 and 0.85 on calibration. Their recall is 0.93, 0.94 and 0.93 on selection and 0.96 in every seed on
  calibration. The head's cuts are counted apart from the label switches of the systems' outputs.
- **Head training.** Each head trained for 7,500 updates in about 1.4 minutes, on the plan's epochs 0 to 19. Its
  report used its seed's last training epoch (epoch 49), which holds the same 1,500 bank streams. There it found 0.996
  of the view changes with precision 0.868 to 0.880 (`head_s*.json`). That is a fit to training data.

## 6. What this round does not show

- **No value of learned boundaries.** Predicted did not consistently beat Uniform, so this round gives no evidence that
  learning where to cut adds value over a fixed grid.
- **No finer-partition benefit.** Neither refinement consistently beat Original FACT. Their point estimates are mostly
  lower, mainly through extra cuts.
- **Short runs.** The small gains in kept short runs are point estimates over about 10 recordings, and they decide
  nothing.
- **No mechanism.** The groups differ from Original FACT in their whole training. A difference cannot be assigned to
  the averaging, the GRU, the attention or the partition alone.
- **Not the goal.** Neither group reaches the goal or beats the strong references. The development halves are reused,
  so these numbers are no independent validation, and the test split stays sealed.
- **Next step.** Whether to end this route is the lead's decision. The fixed decision is "uncertain" (row 4). The
  readings of row 3 added after the evaluation disagree (answer 4), so they settle nothing.

## Commands

Run on the server in the project `.venv`, one GPU job at a time, with `ECHO_GPU_MEM_FRACTION=0.6`.

```bash
C=configs/experiments/fact_predicted_partition_v1.yaml
python scripts/train_fact_predicted_partition.py --config $C --prepare
python scripts/train_fact_predicted_partition.py --config $C --train-head --seed 0      # and 1, 2
python scripts/train_fact_predicted_partition.py --config $C --freeze-heads
python scripts/train_fact_predicted_partition.py --config $C --smoke --group Uniform --seed 0   # and Predicted
python scripts/queue_fact_predicted_partition.py --config $C                            # the six formal runs
python scripts/report_fact_predicted_partition.py --config $C --evaluate                # the single evaluation
python -m pytest tests/test_fact_predicted_partition.py tests/test_fact_predicted_partition_report.py -q
```

## Provenance

- **Code.** The training code is at 26069b5, and the decision readings at d814ff6.
- **Configuration.** `configs/experiments/fact_predicted_partition_v1.yaml` pins the paired round's configuration,
  summary, plans, initial states and frozen final weights. Every digest was checked before use.
- **Frozen weights.** On the server, the heads are in `models_frozen/fact_predicted_partition_v1/head_s{0,1,2}`, with
  sha256 prefixes 9d9ac27fb426, 5e34c3275b45 and 00b080f34bb6. The final FACT weights are in
  `models_frozen/fact_predicted_partition_v1/{Uniform,Predicted}_s{0,1,2}`.
- **Run identities and digests.** `summary.json` holds every run's identity and final sha256.
- **Server outputs.** `runs/fact_predicted_partition_v1/evaluation/` holds the per-sequence outputs on both halves.
  For the three FACT systems they are the official output, the U frame branch's probabilities, the i, u and U argmax,
  the partitions and (Predicted) the head's cuts. For ASRF they are both outputs and the probabilities, and for the
  control the output and the head's cuts. Recording and session ids stay on the server.

## Files

- `README.md` is this file. `code_review.md` is the record of the implementation review before training.
- `preparation.json`, `head_s*.json` and `smoke_*.json` hold the checks before the queue.
- `summary.json` holds every system row, every comparison with its bootstrap and change tables, the verdicts, the
  decision, the reference units, the stage tracking, the position control, the interval counts, the head cuts, the
  references and the costs.
- `report_tables.md` holds the generated tables. `training_curves.png` shows the training curves, and
  `cases_a4c_a5c_middles.png` the case figure.
- None of these files contains a recording or session id.
