# Paired FACT and ASRF training on frozen E1 features (2026-10-08 to 10-09)

This round trained the official FACT (Lu and Elhamifar, CVPR 2024) and the official ASRF (Ishikawa et al., WACV 2021)
on the same frozen E1 512-d per-frame features, three seeds each. It then evaluated them once on the two development
halves, beside B3 and the archived references. Both models are prior work, and training them is not a contribution of
this project. The comparison is between two complete trained systems with their own optimisers and augmentation. A
difference therefore cannot be attributed to cross-attention or to the matching loss alone.

**What kind of evidence this is.**
- Training used the EV9V training split only. Nothing was chosen from the development halves during training.
- The development halves have been used in many earlier rounds, so this evaluation is not an independent validation.
- The hold-out is by session, not by patient.
- Constructed streams are spliced from separate recordings and are not real scans. They reuse recordings, so their
  units are not independent cases. The three seeds share the same cases.
- Native clips carry the public clip label, not expert frame labels.
- The test split and EchoNet-Pediatric were not read.

**Order of work.**
- 5863c6a committed the task sheet, the configuration, the three entries and their tests before any preparation.
- The preparation and smoke checks followed. bc45d88 added a diagnostic before it ran, and a06d037 fixed the
  implementation review's note.
- a086346 lengthened the FACT smoke's overfit check to 2,000 updates by the user's decision (section 7). 708ccf1
  committed the passing smoke checks.
- The queue then ran the six formal runs once. bee551c changed the report code after a second review, before any
  evaluation (section 7).
- After all six runs finished, the report entry verified, froze and evaluated them once.

## Answers

1. **The decision.** FACT − ASRF is not consistent positive in both halves, so row 1 of the fixed table applies. The
   comparison of this GTEA configuration ends. Segment-token methods in general are not rejected, and the Breakfast base
   is not run automatically.
   - **Selection half (uncertain, positive direction).** FACT − ASRF in segment F1@50 is +0.028, +0.011 and +0.028
     (seeds 0, 1, 2). Seeds 0 and 2 exclude zero, and seed 1 does not.
   - **Calibration half (uncertain, negative direction).** It is −0.005, −0.012 and −0.001, and every interval includes
     zero.
2. **Totals (question 1).**
   - **Segment F1@50.** The three-seed means are 0.905 (selection) and 0.964 (calibration) for FACT, against 0.883 and
     0.970 for the paired ASRF. E1 + B3 has 0.894 and 0.959.
   - **Against E1 + B3.** FACT − (E1 + B3) is +0.004 to +0.017 on selection and +0.002 to +0.008 on calibration. Every
     interval includes zero.
   - **Boundaries.** Boundary recall changes little against ASRF, and no seed's interval excludes zero in either half.
     FACT's boundary precision is a little lower, 0.925 to 0.928 against 0.927 to 0.942 on selection, and 0.968 to
     0.972 against 0.975 to 0.980 on calibration.
   - **Native macro-F1.** FACT has 0.966 to 0.989 on selection, against 0.930 to 0.954 for ASRF. Only seed 0's interval
     excludes zero (+0.058 [+0.012, +0.124]). On calibration, seeds 0 and 1 equal ASRF exactly (0.9856), and seed 2
     has 0.993.
   - **Pair streams.** On two-view pair streams, FACT's segment F1@50 is 0.971 to 0.983 on selection and 0.953 to 0.980
     on calibration, against 0.964 to 0.969 and 0.971 to 0.980 for ASRF. Its pair boundary recall on calibration is
     lower, 0.897 to 0.931 against 0.931 to 0.948. Pair streams are reported beside the multi streams and decide
     nothing.
3. **Short segments (question 2).** The counts are small and the measures decide nothing.
   - **Short reference segments (at most 16 samples).** On selection, FACT keeps 127 to 128 of 141, against 123 to 125
     for ASRF and 129 for E1 + B3. On calibration it keeps 123 to 125 of 134, against 125 to 128 for ASRF and 125 for
     E1 + B3.
   - **Short A4C or A5C segments.** FACT keeps 54 to 56 of 65 on selection (ASRF 51 to 54, E1 + B3 56) and 57 to 59 of
     67 on calibration (ASRF 60 to 61, E1 + B3 59).
   - **A4C/A5C X → Y → X middles.** There are only 7 per half. FACT keeps 1 to 3 on selection and 5 to 7 on calibration.
     ASRF keeps 3 to 4 and 7, and E1 + B3 keeps 5 and 7.
   - **Why FACT misses.** On selection, most of FACT's missed short segments have no predicted segment at IoU 0.5 (8 to
     9 of 13 to 14). On calibration the two reasons are about even, with 4 to 6 wrong-view misses and 5 misses without a
     segment. Every missed A4C/A5C middle of FACT had no segment at IoU 0.5, none a wrong view. Section 3 gives the
     source recordings and sessions.
4. **False cuts (question 3).**
   - **Label switches inside native clips.** FACT has none on selection, against 0.007 to 0.011 per clip for ASRF. On
     calibration it has 0.030 to 0.034, against 0.004 for ASRF. The calibration difference is +0.025 to +0.030 per clip,
     and two of three intervals exclude zero.
   - **These are not confirmed errors.** Native labels are clip-level labels, so a switch is a disagreement with that
     label, not an error an expert confirmed.
   - **Cuts at same-view joins.** FACT cuts at 0.035 to 0.052 of these joins on selection and 0.025 to 0.030 on
     calibration, against 0.048 to 0.065 and 0.030 to 0.035 for ASRF. No interval excludes zero.
   - **No clear trade.** No consistent gain in boundary recall was detected in either half, so there is no sign that
     FACT bought recall with extra cuts. On calibration it shows slightly more native switches and slightly lower
     boundary precision.
5. **A4C and A5C (question 4).** FACT did not lose A5C recognition against ASRF.
   - **Against ASRF.** On selection FACT fixes 4, 2 and 1 clips of A5C read as A4C (3, 2 and 1 sessions) and adds none.
     It fixes 1 clip of A4C read as A5C (seed 0) and adds none. On calibration it fixes 1 (seed 2) and adds none.
   - **Against E1 + B3.** Seed 0 fixes 1 A5C clip on selection. Seed 2 adds 1 on selection (1 session) and fixes 1 on
     calibration.
6. **FACT's frame branch (question 5).** The official output and the frame-branch argmax differ by at most 0.010 in
   segment F1@50. The official output switches less inside native clips on selection (0 against 0.004 to 0.007). This is
   a readout of the trained system. The frame branch was trained together with the tokens, so it is not an ablation
   without tokens.
7. **Tokens (question 6).** Each half has 120 multi streams.
   - **Token counts.** The non-null tokens equal the reference segment count in 73 to 82 streams on selection and 81
     to 92 on calibration. There are more tokens than segments in 25 to 37 streams and fewer in 3 to 10. The mean excess
     is +0.18 to +0.31 tokens.
   - **Separate intervals.** One token covered separate intervals in 1 to 6 outputs per half and seed, out of 472 to
     541 sequences. So the 12 tokens do not cap the number of output segments.
8. **Constraints and the goal.** FACT fails the constraints in two seed-half cells.
   - **The failures.** Seed 0 on selection has boundary recall 0.948. Seed 2 on selection has recall 0.940 and native
     macro-F1 0.966. Every calibration cell meets them.
   - **The goal.** The means of 0.905 and 0.964 miss the goal of 0.944 and 0.982 by 0.039 and 0.018. The calibration
     sensitivity target of 0.984 is missed by 0.020.
9. **Training and cost.**
   - **Training time.** The six runs took 10.10 GPU hours of training and monitoring (10.13 h of wall clock with saves
     and start-up), within the 13 h cap. FACT took 2.80 to 3.17 h per run and ASRF 0.38 to 0.41 h.
   - **Size and decoding.** FACT has 10,215,424 parameters and ASRF 1,191,768. FACT decodes in 0.018 to 0.021 s per
     sequence on the same GPU, against 0.008 to 0.011 s for ASRF.
   - **Training curves.** Over the last 15 epochs, FACT's training loss changed by 0.3% to 0.6% net and fluctuated
     within 1.5% to 1.7%. Its fit on the 16 fixed training sequences stayed between segment F1@50 0.90 and 0.96 (11 to
     13 exact transcripts). ASRF's loss was still falling slowly as its learning rate reached zero, and it fitted all 16
     sequences. The final weights were kept in every run.

## Terms

- A **difference is consistent** in a half when all three seeds' 95% intervals exclude zero strictly on the same side.
  Otherwise no consistent difference is detected, which does not mean the systems are equal.
- The **paired bootstrap** resamples native clips by session and streams by whole stream, with the same draws for both
  systems (2,000 draws, seed 20261002). Streams share recordings, so the intervals are approximate.
- A **reference segment** is a run of one view in a multi stream. It is **matched** when a predicted segment of the same
  view has an IoU of at least 0.5 with it, one to one. A **short** segment has at most 16 samples, the lower quartile
  of the training bank's segment lengths.
- An **X → Y → X middle** is a reference run whose two neighbours share a view that differs from its own.
- A **label switch** is a change of the predicted label inside a native clip.
- A **non-null token** is a token whose class is a view, not null.

## 1. Systems

Each cell gives the selection half, then the calibration half. Recompute with the command in the Commands section.

| System | Segment F1@50 | Boundary recall | Boundary precision | Native macro-F1 | Native label switches per clip | Cuts at same-view joins | A5C read as A4C (clips) | A4C read as A5C (clips) |
|---|---|---|---|---|---|---|---|---|
| FACT seed 0 | 0.906 / 0.967 | 0.948 / 0.982 | 0.926 / 0.968 | 0.989 / 0.986 | 0.000 / 0.034 | 0.052 / 0.025 | 0 / 1 | 0 / 1 |
| FACT seed 1 | 0.898 / 0.961 | 0.950 / 0.975 | 0.928 / 0.968 | 0.978 / 0.986 | 0.000 / 0.030 | 0.043 / 0.030 | 1 / 1 | 0 / 1 |
| FACT seed 2 | 0.911 / 0.965 | 0.940 / 0.972 | 0.925 / 0.972 | 0.966 / 0.993 | 0.000 / 0.034 | 0.035 / 0.025 | 2 / 0 | 0 / 1 |
| ASRF seed 0 | 0.878 / 0.972 | 0.935 / 0.987 | 0.927 / 0.975 | 0.930 / 0.986 | 0.011 / 0.004 | 0.065 / 0.035 | 4 / 1 | 1 / 1 |
| ASRF seed 1 | 0.888 / 0.972 | 0.940 / 0.985 | 0.937 / 0.978 | 0.954 / 0.986 | 0.007 / 0.004 | 0.048 / 0.035 | 3 / 1 | 0 / 1 |
| ASRF seed 2 | 0.882 / 0.966 | 0.942 / 0.982 | 0.942 / 0.980 | 0.954 / 0.986 | 0.011 / 0.004 | 0.065 / 0.030 | 3 / 1 | 0 / 1 |
| E1 + B3 | 0.894 / 0.959 | 0.955 / 0.977 | 0.943 / 0.951 | 0.978 / 0.986 | 0.046 / 0.068 | 0.039 / 0.020 | 1 / 1 | 0 / 1 |
| E0 + B3 (historical strongest on selection) | 0.934 / 0.934 | 0.961 / 0.955 | 0.995 / 0.969 | 0.965 / 0.966 | 0.014 / 0.034 | 0.004 / 0.020 | 3 / 4 | 0 / 0 |
| B7 + B3 | 0.880 / 0.972 | 0.955 / 0.975 | 0.890 / 0.963 | 0.957 / 0.993 | 0.063 / 0.042 | 0.057 / 0.035 | 2 / 1 | 1 / 0 |

**Two-view pair streams (reported beside the multi streams; they decide nothing).**

| System | Pair segment F1@50 | Pair boundary recall |
|---|---|---|
| FACT seed 0 | 0.971 / 0.977 | 0.986 / 0.931 |
| FACT seed 1 | 0.971 / 0.980 | 0.986 / 0.931 |
| FACT seed 2 | 0.983 / 0.953 | 0.986 / 0.897 |
| ASRF seed 0 | 0.964 / 0.974 | 0.986 / 0.948 |
| ASRF seed 1 | 0.969 / 0.971 | 0.971 / 0.931 |
| ASRF seed 2 | 0.969 / 0.980 | 0.986 / 0.948 |
| E1 + B3 | 0.957 / 0.963 | 0.986 / 0.931 |
| E0 + B3 | 0.993 / 0.959 | 1.000 / 0.897 |
| B7 + B3 | 0.967 / 0.965 | 1.000 / 0.879 |

**Archived references, reproduced with their stored settings.** E0 + B3, E1 + B3, B7 + B3 and the historical E1 + ASRF
reproduced their archived counts. The historical E1 + ASRF has 0.910, 0.909 and 0.907 on selection and 0.976, 0.964
and 0.964 on calibration. It chose its checkpoint on the selection half, as
`configs/experiments/l1_asrf_integration_v1.yaml` records, so its selection numbers are optimistic. G1 +
ASRF of the geometry round has 0.904, 0.920 and 0.915 on selection and 0.972, 0.974 and 0.976 on calibration. Its
calibration mean reproduces 0.9738935784747763.

## 2. Paired comparisons (segment F1@50)

| Comparison | Half | Seed 0 | Seed 1 | Seed 2 | Verdict |
|---|---|---|---|---|---|
| FACT − ASRF | selection | +0.028 [+0.006, +0.053] | +0.011 [−0.014, +0.038] | +0.028 [+0.004, +0.054] | no consistent difference detected |
| FACT − ASRF | calibration | −0.005 [−0.021, +0.011] | −0.012 [−0.029, +0.004] | −0.001 [−0.013, +0.011] | no consistent difference detected |
| FACT − (E1 + B3) | selection | +0.012 [−0.007, +0.033] | +0.004 [−0.018, +0.027] | +0.017 [−0.006, +0.041] | no consistent difference detected |
| FACT − (E1 + B3) | calibration | +0.008 [−0.009, +0.024] | +0.002 [−0.015, +0.017] | +0.006 [−0.009, +0.021] | no consistent difference detected |
| FACT − (E0 + B3) | selection | −0.028 [−0.055, +0.000] | −0.036 [−0.064, −0.008] | −0.024 [−0.052, +0.006] | no consistent difference detected |
| FACT − (E0 + B3) | calibration | +0.033 [+0.007, +0.059] | +0.027 [+0.003, +0.051] | +0.031 [+0.007, +0.057] | consistent positive |
| FACT − (B7 + B3) | selection | +0.026 [+0.006, +0.048] | +0.018 [−0.003, +0.040] | +0.031 [+0.008, +0.056] | no consistent difference detected |
| FACT − (B7 + B3) | calibration | −0.005 [−0.023, +0.014] | −0.012 [−0.030, +0.006] | −0.007 [−0.022, +0.010] | no consistent difference detected |

The upper bound of seed 0's FACT − (E0 + B3) interval on selection is +0.0000126, so it includes zero. `summary.json`
also holds the paired differences in native macro-F1, boundary recall, boundary F1, native switches and cuts at
same-view joins.

## 3. Short segments and X → Y → X middles

Each cell gives matched over all units, on selection and then calibration.

| System | Short segments | Short A4C or A5C segments | X → Y → X middles | A4C/A5C X → Y → X middles |
|---|---|---|---|---|
| FACT seed 0 | 127/141 / 124/134 | 54/65 / 58/67 | 60/69 / 79/80 | 1/7 / 7/7 |
| FACT seed 1 | 127/141 / 123/134 | 56/65 / 57/67 | 60/69 / 77/80 | 3/7 / 5/7 |
| FACT seed 2 | 128/141 / 125/134 | 55/65 / 59/67 | 59/69 / 78/80 | 3/7 / 6/7 |
| ASRF seed 0 | 125/141 / 127/134 | 53/65 / 61/67 | 60/69 / 79/80 | 4/7 / 7/7 |
| ASRF seed 1 | 123/141 / 128/134 | 51/65 / 61/67 | 60/69 / 79/80 | 3/7 / 7/7 |
| ASRF seed 2 | 125/141 / 125/134 | 54/65 / 60/67 | 60/69 / 79/80 | 4/7 / 7/7 |
| E1 + B3 | 129/141 / 125/134 | 56/65 / 59/67 | 63/69 / 80/80 | 5/7 / 7/7 |

The misses are split below. Each cell gives the wrong-view misses + the misses without a segment at IoU 0.5, then the
distinct source recordings and sessions of all misses in brackets. No unit was missed because another reference took
its match.

| System | Short segments | Short A4C or A5C segments | X → Y → X middles | A4C/A5C X → Y → X middles |
|---|---|---|---|---|
| FACT seed 0 | 5+9 (11, 9) / 5+5 (6, 5) | 2+9 (10, 8) / 4+5 (5, 4) | 1+8 (10, 10) / 1+0 (1, 1) | 0+6 (7, 7) / 0+0 (0, 0) |
| FACT seed 1 | 6+8 (11, 9) / 6+5 (8, 6) | 2+7 (8, 7) / 5+5 (7, 5) | 2+7 (9, 9) / 1+2 (3, 3) | 0+4 (4, 4) / 0+2 (2, 2) |
| FACT seed 2 | 5+8 (9, 8) / 4+5 (7, 5) | 2+8 (8, 7) / 4+4 (6, 4) | 1+9 (12, 12) / 0+2 (2, 2) | 0+4 (4, 4) / 0+1 (1, 1) |
| ASRF seed 0 | 6+10 (14, 9) / 5+2 (7, 6) | 3+9 (12, 8) / 4+2 (6, 5) | 4+5 (9, 8) / 1+0 (3, 3) | 0+3 (4, 4) / 0+0 (0, 0) |
| ASRF seed 1 | 6+12 (15, 10) / 4+2 (6, 5) | 3+11 (13, 9) / 4+2 (6, 5) | 3+6 (10, 9) / 1+0 (3, 3) | 0+4 (5, 5) / 0+0 (0, 0) |
| ASRF seed 2 | 6+10 (14, 10) / 5+4 (9, 7) | 3+8 (11, 8) / 4+3 (7, 6) | 4+5 (9, 8) / 1+0 (3, 3) | 0+3 (4, 4) / 0+0 (0, 0) |
| E1 + B3 | 6+6 (10, 7) / 5+4 (8, 6) | 3+6 (9, 6) / 4+4 (7, 5) | 2+4 (6, 5) / 0+0 (0, 0) | 0+2 (2, 2) / 0+0 (0, 0) |

## 4. Fixed and new errors, FACT against ASRF and E1 + B3

In the first two columns, each count is followed by its sessions in brackets. In the other columns, each count is
followed by its distinct recordings and sessions.

| Comparison | Seed | Half | A5C → A4C fixed / new | A4C → A5C fixed / new | Segments matched by FACT only | Segments matched by the other only | View changes found by FACT only | View changes found by the other only |
|---|---|---|---|---|---|---|---|---|
| FACT − ASRF | 0 | selection | 4 (3) / 0 (0) | 1 (1) / 0 (0) | 23 (17, 13) | 6 (8, 6) | 11 (18, 15) | 6 (9, 8) |
| FACT − ASRF | 0 | calibration | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 5 (8, 8) | 7 (6, 5) | 1 (2, 2) | 3 (6, 5) |
| FACT − ASRF | 1 | selection | 2 (2) / 0 (0) | 0 (0) / 0 (0) | 20 (19, 13) | 11 (14, 12) | 10 (16, 13) | 6 (9, 8) |
| FACT − ASRF | 1 | calibration | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 6 (8, 8) | 12 (10, 7) | 2 (4, 4) | 6 (9, 7) |
| FACT − ASRF | 2 | selection | 1 (1) / 0 (0) | 0 (0) / 0 (0) | 25 (22, 14) | 8 (12, 10) | 10 (16, 11) | 11 (16, 14) |
| FACT − ASRF | 2 | calibration | 1 (1) / 0 (0) | 0 (0) / 0 (0) | 7 (8, 6) | 8 (7, 5) | 3 (6, 5) | 7 (11, 7) |
| FACT − (E1 + B3) | 0 | selection | 1 (1) / 0 (0) | 0 (0) / 0 (0) | 15 (12, 10) | 7 (9, 8) | 3 (5, 4) | 6 (9, 6) |
| FACT − (E1 + B3) | 0 | calibration | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 10 (13, 12) | 8 (7, 7) | 6 (12, 10) | 4 (8, 7) |
| FACT − (E1 + B3) | 1 | selection | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 14 (12, 9) | 10 (13, 10) | 4 (6, 5) | 6 (10, 8) |
| FACT − (E1 + B3) | 1 | calibration | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 8 (10, 9) | 11 (10, 7) | 5 (10, 9) | 6 (10, 8) |
| FACT − (E1 + B3) | 2 | selection | 0 (0) / 1 (1) | 0 (0) / 0 (0) | 18 (12, 10) | 9 (12, 11) | 3 (5, 5) | 9 (15, 12) |
| FACT − (E1 + B3) | 2 | calibration | 1 (1) / 0 (0) | 0 (0) / 0 (0) | 9 (11, 10) | 11 (10, 8) | 6 (12, 10) | 8 (13, 9) |

## 5. FACT's readouts and tokens

| FACT seed | Official output segment F1@50 | Frame-branch argmax segment F1@50 | Official native macro-F1 | Frame-branch native macro-F1 | Official switches per clip | Frame-branch switches per clip |
|---|---|---|---|---|---|---|
| 0 | 0.906 / 0.967 | 0.900 / 0.962 | 0.989 / 0.986 | 0.978 / 0.986 | 0.000 / 0.034 | 0.007 / 0.025 |
| 1 | 0.898 / 0.961 | 0.905 / 0.961 | 0.978 / 0.986 | 0.966 / 0.986 | 0.000 / 0.030 | 0.004 / 0.030 |
| 2 | 0.911 / 0.965 | 0.901 / 0.967 | 0.966 / 0.993 | 0.978 / 0.993 | 0.000 / 0.034 | 0.007 / 0.042 |

| FACT seed | Half | Multi streams | Tokens = segments | Fewer tokens | More tokens | Mean tokens − segments | Outputs with a token on separate intervals |
|---|---|---|---|---|---|---|---|
| 0 | selection | 120 | 77 | 10 | 33 | +0.25 | 4 of 541 |
| 0 | calibration | 120 | 81 | 5 | 34 | +0.31 | 2 of 472 |
| 1 | selection | 120 | 73 | 10 | 37 | +0.29 | 6 of 541 |
| 1 | calibration | 120 | 81 | 7 | 32 | +0.27 | 6 of 472 |
| 2 | selection | 120 | 82 | 10 | 28 | +0.18 | 5 of 541 |
| 2 | calibration | 120 | 92 | 3 | 25 | +0.21 | 1 of 472 |

## 6. Decision under the fixed rule

| Step | Result |
|---|---|
| FACT − ASRF consistent positive in both halves? | No. No consistent difference was detected in either half. |
| FACT − (E1 + B3) consistent positive in both halves? | No. No consistent difference was detected in either half. |
| Constraints met in every FACT seed and half (boundary recall ≥ 0.95, native macro-F1 ≥ 0.973 / 0.980)? | No. Seed 0 on selection (recall 0.948) and seed 2 on selection (recall 0.940, native 0.966) fail. |
| Goal met (FACT means ≥ 0.944409687184662 / 0.9821421709894332)? | No. The means are 0.9051 and 0.9644. |
| Case | Row 1. FACT − ASRF is not consistent positive in both halves. |
| Action | The comparison of this GTEA configuration ends. Each half is reported as positive, negative or uncertain. Segment-token methods in general are not rejected, and Breakfast is not run automatically. |

The calibration sensitivity target (0.9838935784747763) is missed by 0.0195.

## 7. Deviations and limits

- **The smoke's overfit length.** The task sheet fixed 1,000 updates. At that length, 2 of the 6 single-segment
  sequences among the 16 still had only null tokens, though the output was right through the frame-branch fallback. The
  diagnostic (bc45d88) replayed the check exactly and met every requirement from update 1,500 on. The sheet only names
  the auto-start condition, that every check passes. So the user chose how to proceed, and the check was lengthened to
  2,000 updates (a086346). No training setting changed.
- **The report code.** A second review before the evaluation found that results were written only at the end. It also
  found that every system would have been decoded under deterministic algorithms, a setting in which ASRF had never
  been trained, smoke-tested or decoded. bee551c made the run write after each stage and decode each arm under its own
  training setting. Both changes came before any evaluation. The report code is outside the run
  identities.
- **The shared GPU.** A preprocessing job of another project (FastSAM, in `/home/william/hli-echo-segmentation`)
  shared the GPU during the smoke reruns, the training and the evaluation. It may have slowed the measured times.
  Sharing a GPU changes timing, not the arithmetic of a run. FACT's runs are bit-reproducible under deterministic
  algorithms. ASRF trains without them, and its same-state reruns already differ by about 2e-7 in the loss, so its runs
  are reproducible to that level, shared GPU or not.
- **Comparability.** This round's ASRF keeps its final weights. The historical E1 + ASRF chose its checkpoint on the
  selection half, so the two are not directly comparable on that half.
- **What this does not show.**
  - It does not show that FACT is worse or no better than ASRF. On selection, all three point estimates favour FACT,
    and two of the three intervals exclude zero.
  - It does not show which part of FACT matters. The arms differ in model, optimiser and augmentation.
  - It does not show anything about the Breakfast base, other token counts or other matching modes, because they were
    not run.
  - It does not show behaviour on real continuous scans or on patients outside these sessions.

## 8. The fixed settings

The lead fixed four decisions (`docs/superpowers/plans/2026-10-08-fact-asrf-training.md`). The configuration is
`configs/experiments/fact_asrf_paired_v1.yaml`.

| Item | FACT | ASRF |
|---|---|---|
| Input | frozen E1 512-d features (short side 256, centre crop 224, numeric frame order, 10 Hz) | the same |
| Training data | 4 bank streams and 4 native windows (at most 256 samples) per update, from one plan per seed | the same plan, update by update |
| Seeds | 0, 1, 2 | 0, 1, 2 |
| Updates | 18,750 (50 epochs of 375) | 18,750 |
| Model | official FACT, `configs/default.py` + `gtea.yaml`, 12 tokens, `trans=false`, one-to-one matching | official ASRF, 512 input channels |
| Optimiser | Adam, lr 1e-4, betas (0.9, 0.999), eps 1e-8, no weight decay, clip 10 | AdamW, lr 3e-4, weight decay 1e-4, clip 1 |
| Learning rate | 1e-4 for the first 11,719 updates, then 1e-5; no warm-up | 750 warm-up updates, then cosine to 0 |
| Determinism | fp32, TF32 off, cuDNN deterministic, deterministic algorithms | fp32, TF32 off, cuDNN deterministic |
| Checkpoint | the weights after update 18,750 | the same |
| Decoding | official output (frame-branch weight 0.1), no search | official boundary refinement at 0.5, no search |

- **The GTEA base.** The help of the official `utils/gen_config.py` names it for small datasets. It has three blocks,
  128-wide branches and its own dropout and masking. This is a resource choice made in advance. It is not evidence
  that GTEA works better, and it is not a pure parameter-count contrast with the Breakfast base.
- **The decay step.** 11,719 is `ceil(18750 x 250 / 400)`, the official GTEA ratio of epoch 250 to 400 applied to this
  round's budget. It is an adaptation, not the paper's training recipe.
- **ASRF's smoothing weight.** The Gaussian-similarity term enters at weight 1, because the official `train.py`
  passes `gstmse_weight=True`. The 0.15 written in the L1 + ASRF configuration was a recording error. That report now
  carries a dated note, and its models are unchanged.
- **Determinism.** ASRF's cross-entropy has no deterministic CUDA kernel, so its training keeps the project's earlier
  setting. Each arm is also decoded under its own training setting (bee551c).
- **Budget.** The six formal runs may use at most 13 GPU hours. The queue checks the projection before every run.

## 9. Checks before the queue

- **Preparation.** It ran twice with identical files. Seeds 0 to 2 have plans of 18,750 updates, and the six initial
  states hold 10,215,424 parameters (FACT) and 1,191,768 (ASRF). All 56,250 plan batches were unpadded and matched
  their sequences. The integration stage's alignment was reused, because the protocol, the feature manifest, the
  cache key and the bank digest are unchanged (`preparation.json`).
- **Tests.** All 25 tests of that time passed on the server, with none skipped. bee551c added a 26th, and the
  fact-check reran all 26 on the server, all passing.
- **ASRF smoke.** It ran three times and passed each time. The first run (code 5863c6a, unstamped) is in the server
  log only. The second (code a06d037, stamped) is in git at eac230b. `smoke_ASRF.json` now holds the third, described
  under the rerun below. The same-state rerun differs by 2.4e-7 in the loss, and the 3 + 3 resume by 7.5e-9 to 1.5e-8
  in the weights. Padding labels change nothing. The optimiser is AdamW as configured, and no parameter lacks a
  gradient. One update took 0.089 to 0.107 s, or 0.47 to 0.57 h per run.
- **FACT smoke (GTEA, code of 5863c6a).** Every check passed except one requirement of the overfit. Its record is
  `smoke_FACT.json` at eac230b; the current file holds the rerun.
  - The same-state rerun and the 3 + 3 resume differ by 0. Padding labels change nothing, and the unpadded inputs
    equal the sequences. The optimiser is Adam as configured.
  - Inference label isolation found no change in 864 comparisons on the initial model and 864 on the overfit model.
  - At initialisation, only the three structural parameters of the input block's first token self-attention lack a
    gradient. After the overfit, the query and key rows of its projection have gradients of about 1e-6 to 5e-6, and
    the value rows still have none. The integration stage's analysis points to attention dropout, which the GTEA base
    uses. This round did not test it.
  - One update takes 0.596 s on average (median 0.589, 90th percentile 0.674, slowest 0.749) with 1,251 MB at peak,
    or 3.11 h per run.
  - **The requirement not met.** After the fixed 1,000 updates, all 16 transcripts were right, segment F1@50 was 1 for
    all 16, and the matching was one to one. In 2 of the 16 sequences, both with a single reference segment, all 12
    tokens predicted null. The official output fell back to the frame branch and was right, but the matched token's
    class was null.
- **Diagnostic (bc45d88, committed before it ran).** The same overfit was replayed with identical settings. At update
  1,000 it reproduced the smoke's loss exactly, and it continued to 3,000 updates. Every requirement was met from update
  1,500 onward (`fact_overfit_extension.json`). The failure at 1,000 updates is therefore too few updates for this small
  check, not an implementation error.
- **Implementation review.** It found no critical or high issue. Its medium note, that the queue trusted smoke reports
  without a stamp, was fixed in a06d037.
- **The decision (a086346).** The task sheet only names the auto-start condition, that every check passes. So the user
  chose how to proceed. The FACT overfit check was lengthened from 1,000 to 2,000 updates. No training setting, plan or
  initial state changed, and the run identities do not cover the smoke settings.
- **Smoke rerun (code a06d037, config a086346, both reports stamped).** Both passed (`smoke_FACT.json`,
  `smoke_ASRF.json`).
  - **FACT.** The rerun and the 3 + 3 resume differ by 0, and padding labels change nothing. Isolation again found no
    change in 864 comparisons on each model. After 2,000 updates every requirement was met. All 16 transcripts were
    right, segment F1@50 was 1 for all 16, the matching was one to one, the non-null tokens equalled the segments, and
    every matched token had its segment's view. The loss at update 2,000 equals the diagnostic's at the same update.
    One update took 0.666 s on average (median 0.656, 90th percentile 0.764, slowest 0.874) on the shared GPU, or
    3.48 h per run.
  - **ASRF.** The rerun differs by 2.4e-7 in the loss and the resume by 1.5e-8 in the weights. One update took 0.097 s,
    or 0.51 h per run.
- **Shared GPU.** Since 01:18 server time, a preprocessing job of another project (FastSAM, in
  `/home/william/hli-echo-segmentation`) has shared the GPU. Timings after that may be slower.
- **Budget.** With the rerun's timings the projection is 11.96 h (3 x 3.476 h + 3 x 0.510 h), within the 13 h cap.
  The queue checks the projection again before every run, using the finished runs' actual hours.

## Commands

Run on the server in the project `.venv`, one GPU job at a time, with `ECHO_GPU_MEM_FRACTION=0.6`.

```bash
C=configs/experiments/fact_asrf_paired_v1.yaml
python scripts/train_fact_asrf.py --config $C --prepare                    # plans, initial states, checks
python scripts/train_fact_asrf.py --config $C --smoke --arm FACT --seed 0  # and --arm ASRF
python scripts/fact_overfit_extension.py --config $C --updates 3000        # the diagnostic
python scripts/queue_fact_asrf.py --config $C                              # the six formal runs
python scripts/report_fact_asrf.py --config $C --evaluate                  # the single evaluation; every table here
python -m pytest tests/test_fact_integration.py tests/test_fact_asrf_training.py tests/test_fact_asrf_report.py -q
```

## Provenance

- **Configuration.** `configs/experiments/fact_asrf_paired_v1.yaml` at a086346.
- **Final models.** They are frozen on the server in `models_frozen/fact_asrf_paired_v1/{FACT,ASRF}_s{0,1,2}`. The
  final weights' sha256 prefixes are as follows.
  - For FACT seeds 0, 1 and 2 they are 7422d286933d, c793447bd97f and e3387203a161.
  - For ASRF seeds 0, 1 and 2 they are b7d25d305ae1, 1665b7e11627 and 4199cecee651.
  - `summary.json` holds the full digests and run identities.
- **Plans.** Both arms of a seed trained on the same plan (de3bf27c5949, 41668f307581 and 6e9cbba70df8 for seeds 0, 1
  and 2), each from its own initial state.

## Files

- `summary.json` holds every system row (official output and, for FACT and ASRF, the frame-branch argmax), every
  comparison with its bootstrap and change tables, the verdicts, the decision, the mechanism and token tables, the
  references, the costs and the run digests.
- `report_tables.md` holds the generated tables, and `training_curves.png` the training curves.
- `preparation.json`, `smoke_FACT.json`, `smoke_ASRF.json` and `fact_overfit_extension.json` hold the checks before the
  queue.
- None of these files contains recording or session ids. Ids and the frozen weights stay on the server. Per-sequence
  predictions were not saved. They can be regenerated exactly from the frozen weights, as the fact-check did.
