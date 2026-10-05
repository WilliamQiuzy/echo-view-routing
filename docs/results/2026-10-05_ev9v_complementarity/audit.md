# Task 1 audit: E0 and B7 on the EV9V validation halves

Coverage: validation halves only (521 native clips, 252 pair streams, 240 multi-fragment streams); no training-bank or test predictions exist

## Reproduction of the frozen systems

| system|half | native_macro_f1 | segment_f1_50_multi | boundary_recall_multi | native_internal_cuts_per_clip | segments_matched_pred_true | max_abs_diff |
|---|---|---|---|---|---|---|
| E0 frozen|selection | 0.9650 | 0.9344 | 0.9607 | 0.0141 | [463, 489, 502] | 0.0000 |
| E0 frozen|calibration | 0.9660 | 0.9340 | 0.9548 | 0.0338 | [481, 512, 518] | 0.0000 |
| B7 frozen|selection | 0.9566 | 0.8798 | 0.9555 | 0.0634 | [454, 530, 502] | 0.0000 |
| B7 frozen|calibration | 0.9927 | 0.9721 | 0.9749 | 0.0422 | [506, 523, 518] | 0.0000 |

## B7 mass dropped by the frozen map (PMPALA)

| half | samples | mean | median | p99 | max | share_above_0.1 | share_above_0.5 |
|---|---|---|---|---|---|---|---|
| selection | 37287 | 0.0245 | 0.0125 | 0.2942 | 0.9263 | 0.0278 | 0.0070 |
| calibration | 34216 | 0.0335 | 0.0124 | 0.6795 | 0.8954 | 0.0454 | 0.0176 |

## B7 windows that span a join

| half/kind | windows | windows_spanning_a_join | samples | samples_given_a_spanning_window |
|---|---|---|---|---|
| selection/pairs | 1190 | 520 | 5691 | 2513 |
| selection/multi | 3032 | 1834 | 14961 | 9080 |
| calibration/pairs | 1023 | 440 | 4883 | 2131 |
| calibration/multi | 3048 | 1798 | 15010 | 8935 |

## Spot re-run of B7

max |difference| 0.00e+00 over 12 streams

## Units (reference segments; not independent cases)

| half | cell | items | recordings | sessions |
|---|---|---|---|---|
| calibration | both right | 475 | 212 | 35 |
| calibration | only E0 right | 6 | 8 | 8 |
| calibration | only B7 right | 31 | 21 | 18 |
| calibration | both wrong | 6 | 7 | 7 |
| selection | both right | 437 | 229 | 36 |
| selection | only E0 right | 26 | 17 | 13 |
| selection | only B7 right | 17 | 12 | 11 |
| selection | both wrong | 22 | 11 | 9 |

| half | class | cell | items | recordings | sessions |
|---|---|---|---|---|---|
| calibration | A4C | both right | 102 | 56 | 28 |
| calibration | A4C | only E0 right | 4 | 6 | 6 |
| calibration | A4C | only B7 right | 7 | 8 | 8 |
| calibration | A4C | both wrong | 1 | 1 | 1 |
| calibration | A5C | both right | 74 | 18 | 14 |
| calibration | A5C | only E0 right | 0 | 0 | 0 |
| calibration | A5C | only B7 right | 18 | 9 | 8 |
| calibration | A5C | both wrong | 5 | 6 | 6 |
| calibration | PLAX | both right | 100 | 68 | 27 |
| calibration | PLAX | only E0 right | 2 | 2 | 2 |
| calibration | PLAX | only B7 right | 1 | 1 | 1 |
| calibration | PLAX | both wrong | 0 | 0 | 0 |
| calibration | PSAX | both right | 97 | 63 | 24 |
| calibration | PSAX | only E0 right | 0 | 0 | 0 |
| calibration | PSAX | only B7 right | 5 | 3 | 3 |
| calibration | PSAX | both wrong | 0 | 0 | 0 |
| calibration | SC4C | both right | 102 | 7 | 6 |
| calibration | SC4C | only E0 right | 0 | 0 | 0 |
| calibration | SC4C | only B7 right | 0 | 0 | 0 |
| calibration | SC4C | both wrong | 0 | 0 | 0 |
| selection | A4C | both right | 93 | 69 | 28 |
| selection | A4C | only E0 right | 6 | 5 | 5 |
| selection | A4C | only B7 right | 5 | 5 | 5 |
| selection | A4C | both wrong | 7 | 7 | 5 |
| selection | A5C | both right | 77 | 10 | 9 |
| selection | A5C | only E0 right | 4 | 5 | 4 |
| selection | A5C | only B7 right | 10 | 4 | 4 |
| selection | A5C | both wrong | 15 | 4 | 4 |
| selection | PLAX | both right | 105 | 81 | 29 |
| selection | PLAX | only E0 right | 3 | 3 | 2 |
| selection | PLAX | only B7 right | 0 | 0 | 0 |
| selection | PLAX | both wrong | 0 | 0 | 0 |
| selection | PSAX | both right | 87 | 56 | 24 |
| selection | PSAX | only E0 right | 1 | 1 | 1 |
| selection | PSAX | only B7 right | 1 | 2 | 2 |
| selection | PSAX | both wrong | 0 | 0 | 0 |
| selection | SC4C | both right | 75 | 13 | 11 |
| selection | SC4C | only E0 right | 12 | 3 | 3 |
| selection | SC4C | only B7 right | 1 | 1 | 1 |
| selection | SC4C | both wrong | 0 | 0 | 0 |

## Native clips

| half | cell | items | recordings | sessions |
|---|---|---|---|---|
| calibration | both right | 232 | 232 | 36 |
| calibration | only E0 right | 0 | 0 | 0 |
| calibration | only B7 right | 4 | 4 | 4 |
| calibration | both wrong | 1 | 1 | 1 |
| selection | both right | 278 | 278 | 36 |
| selection | only E0 right | 3 | 3 | 3 |
| selection | only B7 right | 1 | 1 | 1 |
| selection | both wrong | 2 | 2 | 2 |

| half | class | cell | items | recordings | sessions |
|---|---|---|---|---|---|
| calibration | A4C | both right | 61 | 61 | 29 |
| calibration | A4C | only E0 right | 0 | 0 | 0 |
| calibration | A4C | only B7 right | 0 | 0 | 0 |
| calibration | A4C | both wrong | 0 | 0 | 0 |
| calibration | A5C | both right | 14 | 14 | 11 |
| calibration | A5C | only E0 right | 0 | 0 | 0 |
| calibration | A5C | only B7 right | 3 | 3 | 3 |
| calibration | A5C | both wrong | 1 | 1 | 1 |
| calibration | PLAX | both right | 80 | 80 | 28 |
| calibration | PLAX | only E0 right | 0 | 0 | 0 |
| calibration | PLAX | only B7 right | 0 | 0 | 0 |
| calibration | PLAX | both wrong | 0 | 0 | 0 |
| calibration | PSAX | both right | 70 | 70 | 24 |
| calibration | PSAX | only E0 right | 0 | 0 | 0 |
| calibration | PSAX | only B7 right | 1 | 1 | 1 |
| calibration | PSAX | both wrong | 0 | 0 | 0 |
| calibration | SC4C | both right | 7 | 7 | 6 |
| calibration | SC4C | only E0 right | 0 | 0 | 0 |
| calibration | SC4C | only B7 right | 0 | 0 | 0 |
| calibration | SC4C | both wrong | 0 | 0 | 0 |
| selection | A4C | both right | 86 | 86 | 29 |
| selection | A4C | only E0 right | 1 | 1 | 1 |
| selection | A4C | only B7 right | 0 | 0 | 0 |
| selection | A4C | both wrong | 0 | 0 | 0 |
| selection | A5C | both right | 8 | 8 | 8 |
| selection | A5C | only E0 right | 0 | 0 | 0 |
| selection | A5C | only B7 right | 1 | 1 | 1 |
| selection | A5C | both wrong | 2 | 2 | 2 |
| selection | PLAX | both right | 104 | 104 | 30 |
| selection | PLAX | only E0 right | 1 | 1 | 1 |
| selection | PLAX | only B7 right | 0 | 0 | 0 |
| selection | PLAX | both wrong | 0 | 0 | 0 |
| selection | PSAX | both right | 68 | 68 | 25 |
| selection | PSAX | only E0 right | 0 | 0 | 0 |
| selection | PSAX | only B7 right | 0 | 0 | 0 |
| selection | PSAX | both wrong | 0 | 0 | 0 |
| selection | SC4C | both right | 12 | 12 | 10 |
| selection | SC4C | only E0 right | 1 | 1 | 1 |
| selection | SC4C | only B7 right | 0 | 0 | 0 |
| selection | SC4C | both wrong | 0 | 0 | 0 |

## Recordings behind the one-sided units

| half|cell | items | recordings | items_involving_top1 | items_involving_top3 | items_involving_top5 |
|---|---|---|---|---|---|
| selection|only E0 right | 26 | 17 | 12 | 15 | 18 |
| selection|only B7 right | 17 | 12 | 8 | 11 | 13 |
| calibration|only E0 right | 6 | 8 | 1 | 3 | 5 |
| calibration|only B7 right | 31 | 21 | 6 | 11 | 15 |

## Teacher class over the reference interval

Units, selection

| cell | opinion | items | recordings | E0 class right | B7 class right |
|---|---|---|---|---|---|
| only E0 right | teachers disagree on class | 18 | 10 | 18 | 0 |
| only E0 right | same class opinion | 8 | 10 | 8 | 8 |
| only B7 right | teachers disagree on class | 8 | 3 | 1 | 7 |
| only B7 right | same class opinion | 9 | 11 | 8 | 8 |
| both wrong | teachers disagree on class | 2 | 2 | 1 | 1 |
| both wrong | same class opinion | 20 | 10 | 6 | 6 |

Native clips, selection

| cell | opinion | items | recordings | E0 class right | B7 class right |
|---|---|---|---|---|---|
| only E0 right | teachers disagree on class | 3 | 3 | 3 | 0 |
| only E0 right | same class opinion | 0 | 0 | 0 | 0 |
| only B7 right | teachers disagree on class | 1 | 1 | 0 | 1 |
| only B7 right | same class opinion | 0 | 0 | 0 | 0 |
| both wrong | teachers disagree on class | 0 | 0 | 0 | 0 |
| both wrong | same class opinion | 2 | 2 | 0 | 0 |

Units, calibration

| cell | opinion | items | recordings | E0 class right | B7 class right |
|---|---|---|---|---|---|
| only E0 right | teachers disagree on class | 3 | 3 | 3 | 0 |
| only E0 right | same class opinion | 3 | 5 | 3 | 3 |
| only B7 right | teachers disagree on class | 12 | 6 | 0 | 12 |
| only B7 right | same class opinion | 19 | 17 | 19 | 19 |
| both wrong | teachers disagree on class | 1 | 3 | 0 | 1 |
| both wrong | same class opinion | 5 | 5 | 0 | 0 |

Native clips, calibration

| cell | opinion | items | recordings | E0 class right | B7 class right |
|---|---|---|---|---|---|
| only E0 right | teachers disagree on class | 0 | 0 | 0 | 0 |
| only E0 right | same class opinion | 0 | 0 | 0 | 0 |
| only B7 right | teachers disagree on class | 2 | 2 | 0 | 2 |
| only B7 right | same class opinion | 2 | 2 | 2 | 2 |
| both wrong | teachers disagree on class | 0 | 0 | 0 | 0 |
| both wrong | same class opinion | 1 | 1 | 0 | 0 |

## Round-4 clue

| group|half | units | B7_B3_wrong | B7_B3_right | recordings | recordings_B7_right | recordings_B7_wrong |
|---|---|---|---|---|---|---|
| E0 common errors (B3 and >= 2 ASRF seeds wrong)|selection | 32 | 21 | 11 | 16 | 7 | 10 |
| E0 common errors (B3 and >= 2 ASRF seeds wrong)|calibration | 22 | 6 | 16 | 13 | 8 | 7 |
| E0 common errors (B3 and >= 2 ASRF seeds wrong)|both | 54 | 27 | 27 | 29 | 15 | 17 |
| E0 common right (B3 and >= 2 ASRF seeds right)|selection | 451 | 22 | 429 | 228 | 228 | 12 |
| E0 common right (B3 and >= 2 ASRF seeds right)|calibration | 477 | 6 | 471 | 212 | 212 | 8 |
| E0 common right (B3 and >= 2 ASRF seeds right)|both | 928 | 28 | 900 | 440 | 440 | 20 |

Outcomes agree with the round-4 inventory: True
