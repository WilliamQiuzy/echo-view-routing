# Short-segment pair thresholds: tables

## Thresholds (library calibration part)

| seed | model | original | pair | pair vs original | cal correct A4C/A5C kept (pair) | keeps all | keep lower bound | cal correct rec/sess | cal wrong removed (orig / pair) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | Q0 | 0.6818 | 0.0011 | lower | 13/13 | True | 0.7940 | 20/17 | 17/18 / 4/18 |
| 0 | Q1 | 0.8268 | 0.0019 | lower | 13/13 | True | 0.7940 | 20/17 | 17/18 / 6/18 |
| 1 | Q0 | 0.5684 | 0.0044 | lower | 14/14 | True | 0.8070 | 21/17 | 22/22 / 8/22 |
| 1 | Q1 | 0.5874 | 0.0270 | lower | 14/14 | True | 0.8070 | 21/17 | 21/22 / 8/22 |
| 2 | Q0 | 0.6280 | 0.0005 | lower | 11/11 | True | 0.7620 | 16/14 | 20/20 / 5/20 |
| 2 | Q1 | 0.8511 | 0.0049 | lower | 11/11 | True | 0.7620 | 16/14 | 20/20 / 8/20 |

## A4C/A5C under both rules

| seed | half | model | rule | correct kept / removed | correct rec / sess | wrong removed / missed | wrong rec / sess |
|---|---|---|---|---|---|---|---|
| 0 | selection | Q0 | original | 3 / 3 | 9 / 9 | 9 / 0 | 7 / 6 |
| 0 | selection | Q0 | pair | 6 / 0 | 9 / 9 | 1 / 8 | 7 / 6 |
| 0 | selection | Q1 | original | 2 / 4 | 9 / 9 | 9 / 0 | 7 / 6 |
| 0 | selection | Q1 | pair | 6 / 0 | 9 / 9 | 1 / 8 | 7 / 6 |
| 0 | calibration | Q0 | original | 5 / 2 | 9 / 7 | 4 / 1 | 4 / 4 |
| 0 | calibration | Q0 | pair | 7 / 0 | 9 / 7 | 0 / 5 | 4 / 4 |
| 0 | calibration | Q1 | original | 5 / 2 | 9 / 7 | 5 / 0 | 4 / 4 |
| 0 | calibration | Q1 | pair | 7 / 0 | 9 / 7 | 0 / 5 | 4 / 4 |
| 0 | pooled halves | Q0 | original | 8 / 5 | 18 / 16 | 13 / 1 | 11 / 10 |
| 0 | pooled halves | Q0 | pair | 13 / 0 | 18 / 16 | 1 / 13 | 11 / 10 |
| 0 | pooled halves | Q1 | original | 7 / 6 | 18 / 16 | 14 / 0 | 11 / 10 |
| 0 | pooled halves | Q1 | pair | 13 / 0 | 18 / 16 | 1 / 13 | 11 / 10 |
| 1 | selection | Q0 | original | 3 / 3 | 9 / 9 | 12 / 1 | 7 / 6 |
| 1 | selection | Q0 | pair | 6 / 0 | 9 / 9 | 1 / 12 | 7 / 6 |
| 1 | selection | Q1 | original | 2 / 4 | 9 / 9 | 13 / 0 | 7 / 6 |
| 1 | selection | Q1 | pair | 6 / 0 | 9 / 9 | 7 / 6 | 7 / 6 |
| 1 | calibration | Q0 | original | 6 / 1 | 9 / 7 | 4 / 1 | 5 / 5 |
| 1 | calibration | Q0 | pair | 7 / 0 | 9 / 7 | 1 / 4 | 5 / 5 |
| 1 | calibration | Q1 | original | 5 / 2 | 9 / 7 | 5 / 0 | 5 / 5 |
| 1 | calibration | Q1 | pair | 7 / 0 | 9 / 7 | 1 / 4 | 5 / 5 |
| 1 | pooled halves | Q0 | original | 9 / 4 | 18 / 16 | 16 / 2 | 12 / 11 |
| 1 | pooled halves | Q0 | pair | 13 / 0 | 18 / 16 | 2 / 16 | 12 / 11 |
| 1 | pooled halves | Q1 | original | 7 / 6 | 18 / 16 | 18 / 0 | 12 / 11 |
| 1 | pooled halves | Q1 | pair | 13 / 0 | 18 / 16 | 8 / 10 | 12 / 11 |
| 2 | selection | Q0 | original | 3 / 3 | 9 / 9 | 16 / 0 | 11 / 9 |
| 2 | selection | Q0 | pair | 6 / 0 | 9 / 9 | 4 / 12 | 11 / 9 |
| 2 | selection | Q1 | original | 3 / 3 | 9 / 9 | 16 / 0 | 11 / 9 |
| 2 | selection | Q1 | pair | 6 / 0 | 9 / 9 | 12 / 4 | 11 / 9 |
| 2 | calibration | Q0 | original | 6 / 1 | 9 / 7 | 5 / 1 | 5 / 5 |
| 2 | calibration | Q0 | pair | 7 / 0 | 9 / 7 | 1 / 5 | 5 / 5 |
| 2 | calibration | Q1 | original | 5 / 2 | 9 / 7 | 6 / 0 | 5 / 5 |
| 2 | calibration | Q1 | pair | 7 / 0 | 9 / 7 | 0 / 6 | 5 / 5 |
| 2 | pooled halves | Q0 | original | 9 / 4 | 18 / 16 | 21 / 1 | 16 / 14 |
| 2 | pooled halves | Q0 | pair | 13 / 0 | 18 / 16 | 5 / 17 | 16 / 14 |
| 2 | pooled halves | Q1 | original | 8 / 5 | 18 / 16 | 22 / 0 | 16 / 14 |
| 2 | pooled halves | Q1 | pair | 13 / 0 | 18 / 16 | 12 / 10 | 16 / 14 |

## A4C/A5C decisions that change (pooled halves)

| seed | model | correct rescued | correct lost | wrong let through | wrong newly removed | outside, now kept | outside, now removed |
|---|---|---|---|---|---|---|---|
| 0 | Q0 | 5 (9 rec, 8 sess) | 0 (0 rec, 0 sess) | 12 (10 rec, 9 sess) | 0 (0 rec, 0 sess) | 10 (14 rec, 13 sess) | 0 (0 rec, 0 sess) |
| 0 | Q1 | 6 (9 rec, 9 sess) | 0 (0 rec, 0 sess) | 13 (10 rec, 9 sess) | 0 (0 rec, 0 sess) | 9 (13 rec, 12 sess) | 0 (0 rec, 0 sess) |
| 1 | Q0 | 4 (8 rec, 8 sess) | 0 (0 rec, 0 sess) | 14 (8 rec, 8 sess) | 0 (0 rec, 0 sess) | 4 (7 rec, 6 sess) | 0 (0 rec, 0 sess) |
| 1 | Q1 | 6 (9 rec, 9 sess) | 0 (0 rec, 0 sess) | 10 (11 rec, 10 sess) | 0 (0 rec, 0 sess) | 7 (11 rec, 10 sess) | 0 (0 rec, 0 sess) |
| 2 | Q0 | 4 (8 rec, 8 sess) | 0 (0 rec, 0 sess) | 16 (12 rec, 11 sess) | 0 (0 rec, 0 sess) | 8 (11 rec, 10 sess) | 0 (0 rec, 0 sess) |
| 2 | Q1 | 5 (7 rec, 7 sess) | 0 (0 rec, 0 sess) | 10 (8 rec, 8 sess) | 0 (0 rec, 0 sess) | 10 (13 rec, 11 sess) | 0 (0 rec, 0 sess) |

## Pooled rates

| rule | model | seed | K_aa | R_aa | K_all | R_all |
|---|---|---|---|---|---|---|
| original | Q0 | 0 | 0.6150 | 0.9290 | 0.9620 | 0.8950 |
| original | Q0 | 1 | 0.6920 | 0.8890 | 0.9690 | 0.8700 |
| original | Q0 | 2 | 0.6920 | 0.9550 | 0.9690 | 0.9230 |
| original | Q1 | 0 | 0.5380 | 1.0000 | 0.9470 | 0.8950 |
| original | Q1 | 1 | 0.5380 | 1.0000 | 0.9540 | 0.9570 |
| original | Q1 | 2 | 0.6150 | 1.0000 | 0.9460 | 0.9620 |
| pair | Q0 | 0 | 1.0000 | 0.0710 | 1.0000 | 0.2630 |
| pair | Q0 | 1 | 1.0000 | 0.1110 | 1.0000 | 0.2610 |
| pair | Q0 | 2 | 1.0000 | 0.2270 | 1.0000 | 0.3080 |
| pair | Q1 | 0 | 1.0000 | 0.0710 | 0.9920 | 0.2110 |
| pair | Q1 | 1 | 1.0000 | 0.4440 | 1.0000 | 0.5220 |
| pair | Q1 | 2 | 1.0000 | 0.5450 | 0.9840 | 0.5770 |

## Q1 - Q0 under the pair rule

| seed | pairs | removal delta | keep delta | removal ci95 | keep ci95 |
|---|---|---|---|---|---|
| 0 | all | -0.0526 | -0.0076 | [-0.16000000000000003, 0.0] | [-0.024590163934426257, 0.0] |
| 0 | A4C/A5C pairs | 0.0000 | 0.0000 | [0.0, 0.0] | [0.0, 0.0] |
| 1 | all | 0.2609 | 0.0000 | [0.09367897727272728, 0.461666666666666] | [0.0, 0.0] |
| 1 | A4C/A5C pairs | 0.3333 | 0.0000 | [0.125, 0.5625679347826084] | [0.0, 0.0] |
| 2 | all | 0.2692 | -0.0155 | [0.0625, 0.5] | [-0.04166666666666663, 0.0] |
| 2 | A4C/A5C pairs | 0.3182 | 0.0000 | [0.09090909090909094, 0.56] | [0.0, 0.0] |

## Correct A4C/A5C cases (original -> pair)

| case | half | direction | s0 Q0 | s0 Q1 | s1 Q0 | s1 Q1 | s2 Q0 | s2 Q1 |
|---|---|---|---|---|---|---|---|---|
| 1 | calibration | A4C -> A5C -> A4C | removed -> kept | removed -> kept | kept -> kept | removed -> kept | kept -> kept | removed -> kept |
| 2 | calibration | A5C -> A4C -> A5C | kept -> kept | kept -> kept | kept -> kept | kept -> kept | kept -> kept | kept -> kept |
| 3 | calibration | A5C -> A4C -> A5C | kept -> kept | kept -> kept | kept -> kept | kept -> kept | kept -> kept | kept -> kept |
| 4 | calibration | A4C -> A5C -> A4C | removed -> kept | kept -> kept | removed -> kept | kept -> kept | removed -> kept | kept -> kept |
| 5 | calibration | A4C -> A5C -> A4C | kept -> kept | kept -> kept | kept -> kept | kept -> kept | kept -> kept | kept -> kept |
| 6 | calibration | A5C -> A4C -> A5C | kept -> kept | kept -> kept | kept -> kept | kept -> kept | kept -> kept | kept -> kept |
| 7 | calibration | A4C -> A5C -> A4C | kept -> kept | removed -> kept | kept -> kept | removed -> kept | kept -> kept | removed -> kept |
| 8 | selection | A4C -> A5C -> A4C | kept -> kept | kept -> kept | kept -> kept | kept -> kept | kept -> kept | kept -> kept |
| 9 | selection | A5C -> A4C -> A5C | removed -> kept | removed -> kept | removed -> kept | removed -> kept | removed -> kept | removed -> kept |
| 10 | selection | A4C -> A5C -> A4C | removed -> kept | removed -> kept | removed -> kept | removed -> kept | removed -> kept | removed -> kept |
| 11 | selection | A5C -> A4C -> A5C | kept -> kept | removed -> kept | kept -> kept | removed -> kept | kept -> kept | removed -> kept |
| 12 | selection | A4C -> A5C -> A4C | kept -> kept | kept -> kept | kept -> kept | kept -> kept | kept -> kept | kept -> kept |
| 13 | selection | A5C -> A4C -> A5C | removed -> kept | removed -> kept | removed -> kept | removed -> kept | removed -> kept | kept -> kept |

## Changed candidates of the pair

| seed | model | half | category | now | direction | source | join | middle samples | score | case |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | Q0 | selection | outside | kept | A5C -> A4C -> A5C | native clips | none | 50 | 0.1290 | None |
| 0 | Q0 | selection | outside | kept | A5C -> A4C -> A5C | native clips | none | 11 | 0.0063 | None |
| 0 | Q0 | selection | wrong | kept | A4C -> A5C -> A4C | native clips | none | 6 | 0.0052 | None |
| 0 | Q0 | selection | wrong | kept | A4C -> A5C -> A4C | native clips | none | 8 | 0.0067 | None |
| 0 | Q0 | selection | outside | kept | A4C -> A5C -> A4C | native clips | none | 54 | 0.0029 | None |
| 0 | Q0 | selection | wrong | kept | A5C -> A4C -> A5C | native clips | none | 16 | 0.0408 | None |
| 0 | Q0 | selection | wrong | kept | A4C -> A5C -> A4C | pair streams | none | 8 | 0.0045 | None |
| 0 | Q0 | selection | outside | kept | A4C -> A5C -> A4C | multi streams | same-view | 70 | 0.0234 | None |
| 0 | Q0 | selection | wrong | kept | A4C -> A5C -> A4C | multi streams | none | 8 | 0.0066 | None |
| 0 | Q0 | selection | correct | kept | A5C -> A4C -> A5C | multi streams | view-change | 26 | 0.2680 | 9 |
| 0 | Q0 | selection | correct | kept | A4C -> A5C -> A4C | multi streams | view-change | 10 | 0.0337 | 10 |
| 0 | Q0 | selection | wrong | kept | A4C -> A5C -> A4C | multi streams | view-change | 8 | 0.0028 | None |
| 0 | Q0 | selection | wrong | kept | A4C -> A5C -> A4C | multi streams | none | 8 | 0.0258 | None |
| 0 | Q0 | selection | outside | kept | A5C -> A4C -> A5C | multi streams | view-change | 26 | 0.6653 | None |
| 0 | Q0 | selection | outside | kept | A5C -> A4C -> A5C | multi streams | none | 12 | 0.2359 | None |
| 0 | Q0 | selection | wrong | kept | A5C -> A4C -> A5C | multi streams | same-view | 40 | 0.2943 | None |
| 0 | Q0 | selection | correct | kept | A5C -> A4C -> A5C | multi streams | view-change | 12 | 0.1281 | 13 |
| 0 | Q0 | calibration | wrong | kept | A5C -> A4C -> A5C | native clips | none | 9 | 0.0037 | None |
| 0 | Q0 | calibration | outside | kept | A4C -> A5C -> A4C | native clips | none | 24 | 0.0030 | None |
| 0 | Q0 | calibration | wrong | kept | A4C -> A5C -> A4C | native clips | none | 7 | 0.0715 | None |
| 0 | Q0 | calibration | outside | kept | A5C -> A4C -> A5C | native clips | none | 48 | 0.6665 | None |
| 0 | Q0 | calibration | outside | kept | A5C -> A4C -> A5C | multi streams | view-change | 11 | 0.1582 | None |
| 0 | Q0 | calibration | correct | kept | A4C -> A5C -> A4C | multi streams | view-change | 10 | 0.5944 | 1 |
| 0 | Q0 | calibration | wrong | kept | A4C -> A5C -> A4C | multi streams | same-view | 28 | 0.0381 | None |
| 0 | Q0 | calibration | outside | kept | A5C -> A4C -> A5C | multi streams | same-view | 16 | 0.1907 | None |
| 0 | Q0 | calibration | correct | kept | A4C -> A5C -> A4C | multi streams | view-change | 15 | 0.5190 | 4 |
| 0 | Q0 | calibration | wrong | kept | A5C -> A4C -> A5C | multi streams | same-view | 16 | 0.0161 | None |
| 0 | Q1 | selection | outside | kept | A5C -> A4C -> A5C | native clips | none | 11 | 0.0068 | None |
| 0 | Q1 | selection | wrong | kept | A4C -> A5C -> A4C | native clips | none | 6 | 0.0058 | None |
| 0 | Q1 | selection | wrong | kept | A4C -> A5C -> A4C | native clips | none | 8 | 0.0021 | None |
| 0 | Q1 | selection | outside | kept | A4C -> A5C -> A4C | native clips | none | 54 | 0.1191 | None |
| 0 | Q1 | selection | wrong | kept | A5C -> A4C -> A5C | native clips | none | 16 | 0.5208 | None |
| 0 | Q1 | selection | wrong | kept | A4C -> A5C -> A4C | pair streams | none | 8 | 0.0023 | None |
| 0 | Q1 | selection | outside | kept | A4C -> A5C -> A4C | multi streams | same-view | 70 | 0.6612 | None |
| 0 | Q1 | selection | wrong | kept | A4C -> A5C -> A4C | multi streams | none | 8 | 0.0226 | None |
| 0 | Q1 | selection | correct | kept | A5C -> A4C -> A5C | multi streams | view-change | 26 | 0.1142 | 9 |
| 0 | Q1 | selection | correct | kept | A4C -> A5C -> A4C | multi streams | view-change | 10 | 0.0881 | 10 |
| 0 | Q1 | selection | wrong | kept | A4C -> A5C -> A4C | multi streams | view-change | 8 | 0.0023 | None |
| 0 | Q1 | selection | wrong | kept | A4C -> A5C -> A4C | multi streams | none | 8 | 0.0079 | None |
| 0 | Q1 | selection | outside | kept | A5C -> A4C -> A5C | multi streams | view-change | 26 | 0.4271 | None |
| 0 | Q1 | selection | correct | kept | A5C -> A4C -> A5C | multi streams | view-change | 14 | 0.8017 | 11 |
| 0 | Q1 | selection | outside | kept | A5C -> A4C -> A5C | multi streams | none | 12 | 0.2612 | None |
| 0 | Q1 | selection | wrong | kept | A5C -> A4C -> A5C | multi streams | same-view | 40 | 0.0025 | None |
| 0 | Q1 | selection | correct | kept | A5C -> A4C -> A5C | multi streams | view-change | 12 | 0.5700 | 13 |
| 0 | Q1 | calibration | wrong | kept | A5C -> A4C -> A5C | native clips | none | 9 | 0.0071 | None |
| 0 | Q1 | calibration | outside | kept | A4C -> A5C -> A4C | native clips | none | 24 | 0.0068 | None |
| 0 | Q1 | calibration | wrong | kept | A5C -> A4C -> A5C | native clips | none | 67 | 0.1029 | None |
| 0 | Q1 | calibration | wrong | kept | A4C -> A5C -> A4C | native clips | none | 7 | 0.0420 | None |
| 0 | Q1 | calibration | outside | kept | A5C -> A4C -> A5C | native clips | none | 48 | 0.0319 | None |
| 0 | Q1 | calibration | outside | kept | A5C -> A4C -> A5C | multi streams | view-change | 11 | 0.3022 | None |
| 0 | Q1 | calibration | correct | kept | A4C -> A5C -> A4C | multi streams | view-change | 10 | 0.2795 | 1 |
| 0 | Q1 | calibration | wrong | kept | A4C -> A5C -> A4C | multi streams | same-view | 28 | 0.0397 | None |
| 0 | Q1 | calibration | outside | kept | A5C -> A4C -> A5C | multi streams | same-view | 16 | 0.0184 | None |
| 0 | Q1 | calibration | correct | kept | A4C -> A5C -> A4C | multi streams | view-change | 15 | 0.0586 | 7 |
| 0 | Q1 | calibration | wrong | kept | A5C -> A4C -> A5C | multi streams | same-view | 16 | 0.2677 | None |
| 1 | Q0 | selection | wrong | kept | A4C -> A5C -> A4C | native clips | none | 8 | 0.0256 | None |
| 1 | Q0 | selection | wrong | kept | A5C -> A4C -> A5C | native clips | none | 38 | 0.4707 | None |
| 1 | Q0 | selection | wrong | kept | A4C -> A5C -> A4C | native clips | none | 16 | 0.0217 | None |
| 1 | Q0 | selection | wrong | kept | A5C -> A4C -> A5C | native clips | none | 16 | 0.0297 | None |
| 1 | Q0 | selection | wrong | kept | A4C -> A5C -> A4C | pair streams | none | 16 | 0.0408 | None |
| 1 | Q0 | selection | wrong | kept | A4C -> A5C -> A4C | pair streams | none | 8 | 0.0186 | None |
| 1 | Q0 | selection | outside | kept | A4C -> A5C -> A4C | multi streams | same-view | 70 | 0.0237 | None |
| 1 | Q0 | selection | wrong | kept | A4C -> A5C -> A4C | multi streams | none | 8 | 0.0177 | None |
| 1 | Q0 | selection | correct | kept | A5C -> A4C -> A5C | multi streams | view-change | 26 | 0.0979 | 9 |
| 1 | Q0 | selection | correct | kept | A4C -> A5C -> A4C | multi streams | view-change | 10 | 0.0278 | 10 |
| 1 | Q0 | selection | wrong | kept | A4C -> A5C -> A4C | multi streams | view-change | 8 | 0.0126 | None |
| 1 | Q0 | selection | wrong | kept | A4C -> A5C -> A4C | multi streams | none | 8 | 0.0509 | None |
| 1 | Q0 | selection | outside | kept | A5C -> A4C -> A5C | multi streams | view-change | 26 | 0.5253 | None |
| 1 | Q0 | selection | wrong | kept | A5C -> A4C -> A5C | multi streams | same-view | 27 | 0.4577 | None |
| 1 | Q0 | selection | outside | kept | A5C -> A4C -> A5C | multi streams | none | 13 | 0.0125 | None |
| 1 | Q0 | selection | wrong | kept | A5C -> A4C -> A5C | multi streams | same-view | 22 | 0.0914 | None |
| 1 | Q0 | selection | correct | kept | A5C -> A4C -> A5C | multi streams | view-change | 12 | 0.0240 | 13 |
| 1 | Q0 | calibration | wrong | kept | A4C -> A5C -> A4C | native clips | none | 7 | 0.0261 | None |
| 1 | Q0 | calibration | outside | kept | A5C -> A4C -> A5C | multi streams | view-change | 11 | 0.0252 | None |
| 1 | Q0 | calibration | wrong | kept | A4C -> A5C -> A4C | multi streams | same-view | 28 | 0.0432 | None |
| 1 | Q0 | calibration | correct | kept | A4C -> A5C -> A4C | multi streams | view-change | 15 | 0.4849 | 4 |
| 1 | Q0 | calibration | wrong | kept | A5C -> A4C -> A5C | multi streams | same-view | 24 | 0.0905 | None |
| 1 | Q1 | selection | outside | kept | A4C -> A5C -> A4C | native clips | none | 53 | 0.1525 | None |
| 1 | Q1 | selection | wrong | kept | A5C -> A4C -> A5C | native clips | none | 16 | 0.2082 | None |
| 1 | Q1 | selection | wrong | kept | A4C -> A5C -> A4C | pair streams | none | 16 | 0.0315 | None |
| 1 | Q1 | selection | outside | kept | A4C -> A5C -> A4C | multi streams | same-view | 70 | 0.2797 | None |
| 1 | Q1 | selection | wrong | kept | A4C -> A5C -> A4C | multi streams | none | 8 | 0.0493 | None |
| 1 | Q1 | selection | correct | kept | A5C -> A4C -> A5C | multi streams | view-change | 26 | 0.0876 | 9 |
| 1 | Q1 | selection | correct | kept | A4C -> A5C -> A4C | multi streams | view-change | 10 | 0.1395 | 10 |
| 1 | Q1 | selection | wrong | kept | A4C -> A5C -> A4C | multi streams | none | 8 | 0.0347 | None |
| 1 | Q1 | selection | outside | kept | A5C -> A4C -> A5C | multi streams | view-change | 26 | 0.2760 | None |
| 1 | Q1 | selection | wrong | kept | A5C -> A4C -> A5C | multi streams | same-view | 27 | 0.0748 | None |
| 1 | Q1 | selection | correct | kept | A5C -> A4C -> A5C | multi streams | view-change | 14 | 0.5395 | 11 |
| 1 | Q1 | selection | outside | kept | A5C -> A4C -> A5C | multi streams | none | 13 | 0.1051 | None |
| 1 | Q1 | selection | wrong | kept | A5C -> A4C -> A5C | multi streams | same-view | 40 | 0.0729 | None |
| 1 | Q1 | selection | correct | kept | A5C -> A4C -> A5C | multi streams | view-change | 12 | 0.3750 | 13 |
| 1 | Q1 | calibration | wrong | kept | A5C -> A4C -> A5C | native clips | none | 65 | 0.1391 | None |
| 1 | Q1 | calibration | wrong | kept | A4C -> A5C -> A4C | native clips | none | 7 | 0.0433 | None |
| 1 | Q1 | calibration | outside | kept | A5C -> A4C -> A5C | native clips | none | 48 | 0.0853 | None |
| 1 | Q1 | calibration | outside | kept | A5C -> A4C -> A5C | multi streams | view-change | 11 | 0.1127 | None |
| 1 | Q1 | calibration | correct | kept | A4C -> A5C -> A4C | multi streams | view-change | 10 | 0.4695 | 1 |
| 1 | Q1 | calibration | wrong | kept | A4C -> A5C -> A4C | multi streams | same-view | 28 | 0.0583 | None |
| 1 | Q1 | calibration | outside | kept | A5C -> A4C -> A5C | multi streams | view-change | 27 | 0.0591 | None |
| 1 | Q1 | calibration | correct | kept | A4C -> A5C -> A4C | multi streams | view-change | 15 | 0.1780 | 7 |
| 1 | Q1 | calibration | wrong | kept | A5C -> A4C -> A5C | multi streams | same-view | 24 | 0.2741 | None |
| 2 | Q0 | selection | wrong | kept | A4C -> A5C -> A4C | native clips | none | 66 | 0.0007 | None |
| 2 | Q0 | selection | outside | kept | A5C -> A4C -> A5C | native clips | none | 50 | 0.0921 | None |
| 2 | Q0 | selection | outside | kept | A5C -> A4C -> A5C | native clips | none | 10 | 0.0987 | None |
| 2 | Q0 | selection | wrong | kept | A4C -> A5C -> A4C | native clips | none | 7 | 0.0111 | None |
| 2 | Q0 | selection | wrong | kept | A4C -> A5C -> A4C | native clips | none | 9 | 0.0149 | None |
| 2 | Q0 | selection | wrong | kept | A5C -> A4C -> A5C | native clips | none | 37 | 0.1566 | None |
| 2 | Q0 | selection | wrong | kept | A4C -> A5C -> A4C | native clips | none | 17 | 0.0007 | None |
| 2 | Q0 | selection | wrong | kept | A5C -> A4C -> A5C | native clips | none | 16 | 0.0401 | None |
| 2 | Q0 | selection | wrong | kept | A4C -> A5C -> A4C | pair streams | none | 17 | 0.0037 | None |
| 2 | Q0 | selection | wrong | kept | A4C -> A5C -> A4C | pair streams | none | 9 | 0.0108 | None |
| 2 | Q0 | selection | outside | kept | A4C -> A5C -> A4C | multi streams | same-view | 72 | 0.0565 | None |
| 2 | Q0 | selection | wrong | kept | A4C -> A5C -> A4C | multi streams | none | 9 | 0.0132 | None |
| 2 | Q0 | selection | correct | kept | A5C -> A4C -> A5C | multi streams | view-change | 26 | 0.4274 | 9 |
| 2 | Q0 | selection | correct | kept | A4C -> A5C -> A4C | multi streams | view-change | 10 | 0.0164 | 10 |
| 2 | Q0 | selection | wrong | kept | A5C -> A4C -> A5C | multi streams | same-view | 14 | 0.0332 | None |
| 2 | Q0 | selection | wrong | kept | A4C -> A5C -> A4C | multi streams | none | 9 | 0.0479 | None |
| 2 | Q0 | selection | outside | kept | A5C -> A4C -> A5C | multi streams | none | 12 | 0.2050 | None |
| 2 | Q0 | selection | wrong | kept | A5C -> A4C -> A5C | multi streams | same-view | 40 | 0.3903 | None |
| 2 | Q0 | selection | correct | kept | A5C -> A4C -> A5C | multi streams | view-change | 12 | 0.0577 | 13 |
| 2 | Q0 | calibration | wrong | kept | A5C -> A4C -> A5C | native clips | none | 9 | 0.0053 | None |
| 2 | Q0 | calibration | outside | kept | A4C -> A5C -> A4C | native clips | none | 24 | 0.0017 | None |
| 2 | Q0 | calibration | wrong | kept | A4C -> A5C -> A4C | native clips | none | 7 | 0.0239 | None |
| 2 | Q0 | calibration | outside | kept | A5C -> A4C -> A5C | native clips | none | 48 | 0.5628 | None |
| 2 | Q0 | calibration | outside | kept | A5C -> A4C -> A5C | multi streams | view-change | 10 | 0.1018 | None |
| 2 | Q0 | calibration | wrong | kept | A4C -> A5C -> A4C | multi streams | same-view | 29 | 0.0203 | None |
| 2 | Q0 | calibration | outside | kept | A5C -> A4C -> A5C | multi streams | view-change | 26 | 0.3618 | None |
| 2 | Q0 | calibration | correct | kept | A4C -> A5C -> A4C | multi streams | view-change | 15 | 0.4297 | 4 |
| 2 | Q0 | calibration | wrong | kept | A5C -> A4C -> A5C | multi streams | same-view | 16 | 0.0330 | None |
| 2 | Q1 | selection | outside | kept | A5C -> A4C -> A5C | native clips | none | 289 | 0.0202 | None |
| 2 | Q1 | selection | outside | kept | A5C -> A4C -> A5C | native clips | none | 46 | 0.7164 | None |
| 2 | Q1 | selection | outside | kept | A5C -> A4C -> A5C | native clips | none | 10 | 0.0222 | None |
| 2 | Q1 | selection | wrong | kept | A4C -> A5C -> A4C | native clips | none | 7 | 0.0108 | None |
| 2 | Q1 | selection | wrong | kept | A5C -> A4C -> A5C | native clips | none | 16 | 0.4620 | None |
| 2 | Q1 | selection | outside | kept | A4C -> A5C -> A4C | multi streams | same-view | 72 | 0.6791 | None |
| 2 | Q1 | selection | wrong | kept | A4C -> A5C -> A4C | multi streams | none | 9 | 0.0082 | None |
| 2 | Q1 | selection | correct | kept | A5C -> A4C -> A5C | multi streams | view-change | 26 | 0.0871 | 9 |
| 2 | Q1 | selection | correct | kept | A4C -> A5C -> A4C | multi streams | view-change | 10 | 0.1088 | 10 |
| 2 | Q1 | selection | wrong | kept | A5C -> A4C -> A5C | multi streams | same-view | 14 | 0.8168 | None |
| 2 | Q1 | selection | outside | kept | A5C -> A4C -> A5C | multi streams | view-change | 25 | 0.5300 | None |
| 2 | Q1 | selection | correct | kept | A5C -> A4C -> A5C | multi streams | view-change | 14 | 0.7696 | 11 |
| 2 | Q1 | selection | outside | kept | A5C -> A4C -> A5C | multi streams | none | 12 | 0.2600 | None |
| 2 | Q1 | calibration | wrong | kept | A4C -> A5C -> A4C | native clips | none | 55 | 0.0054 | None |
| 2 | Q1 | calibration | wrong | kept | A5C -> A4C -> A5C | native clips | none | 9 | 0.0087 | None |
| 2 | Q1 | calibration | outside | kept | A4C -> A5C -> A4C | native clips | none | 24 | 0.0058 | None |
| 2 | Q1 | calibration | wrong | kept | A5C -> A4C -> A5C | native clips | none | 65 | 0.2274 | None |
| 2 | Q1 | calibration | wrong | kept | A4C -> A5C -> A4C | native clips | none | 7 | 0.0502 | None |
| 2 | Q1 | calibration | outside | kept | A5C -> A4C -> A5C | native clips | none | 48 | 0.0681 | None |
| 2 | Q1 | calibration | outside | kept | A5C -> A4C -> A5C | multi streams | view-change | 10 | 0.1411 | None |
| 2 | Q1 | calibration | correct | kept | A4C -> A5C -> A4C | multi streams | view-change | 10 | 0.2988 | 1 |
| 2 | Q1 | calibration | wrong | kept | A4C -> A5C -> A4C | multi streams | same-view | 29 | 0.0407 | None |
| 2 | Q1 | calibration | outside | kept | A5C -> A4C -> A5C | multi streams | view-change | 26 | 0.0448 | None |
| 2 | Q1 | calibration | correct | kept | A4C -> A5C -> A4C | multi streams | view-change | 15 | 0.1125 | 7 |
| 2 | Q1 | calibration | wrong | kept | A5C -> A4C -> A5C | multi streams | same-view | 16 | 0.3466 | None |

## Decision

```
{
 "cases": {
  "Q0": "keeps correct, lets wrong through",
  "Q1": "keeps correct, lets wrong through"
 },
 "meets": [],
 "outcome": "no model meets the standard under the pair rule: the threshold check of these two discriminators ends; no further thresholds",
 "original rule, for reference": {
  "Q0": "keep not reached",
  "Q1": "keep not reached"
 }
}
```
