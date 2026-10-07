# Joint decoding switch-cost check (post hoc): tables

## Choices (calibration only)

| seed | system | status | lambda | kappa | at upper limit | feasible settings | calibration segment F1@50 |
|---|---|---|---|---|---|---|---|
| 0 | J0 | feasible | 0.0000 | 24.0000 | False | 9 of 13 | 0.9739 |
| 0 | J1 | feasible | 2.0000 | 24.0000 | False | 43 of 65 | 0.9769 |
| 1 | J0 | feasible | 0.0000 | 16.0000 | False | 7 of 13 | 0.9808 |
| 1 | J1 | feasible | 2.0000 | 32.0000 | False | 42 of 65 | 0.9846 |
| 2 | J0 | feasible | 0.0000 | 16.0000 | False | 9 of 13 | 0.9730 |
| 2 | J1 | feasible | 2.0000 | 24.0000 | False | 45 of 65 | 0.9749 |

## Systems

| seed | system | half | segment F1@50 | boundary P | boundary R | native macro-F1 | cuts per clip | A4C read as A5C | A5C read as A4C | SC4C read as PLAX | matched units | unmatched predicted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | B + ASRF | selection | 0.9022 | 0.9447 | 0.9398 | 0.9566 | 0.0352 | 1 | 2 | 1 | 452 | 48 |
| 0 | B + ASRF | calibration | 0.9777 | 0.9797 | 0.9724 | 0.9929 | 0.0253 | 1 | 0 | 0 | 505 | 10 |
| 0 | L1 + B3 | selection | 0.9084 | 0.9607 | 0.9607 | 0.9886 | 0.0951 | 0 | 0 | 1 | 456 | 46 |
| 0 | L1 + B3 | calibration | 0.9711 | 0.9725 | 0.9774 | 0.9856 | 0.1013 | 1 | 1 | 0 | 504 | 16 |
| 0 | J0 (kappa <= 8) | selection | 0.9042 | 0.9051 | 0.9738 | 0.9696 | 0.1056 | 2 | 0 | 1 | 467 | 64 |
| 0 | J0 (kappa <= 8) | calibration | 0.9696 | 0.9493 | 0.9874 | 0.9929 | 0.0759 | 1 | 0 | 0 | 510 | 24 |
| 0 | J0 | selection | 0.9188 | 0.9653 | 0.9476 | 0.9696 | 0.0106 | 2 | 0 | 1 | 458 | 37 |
| 0 | J0 | calibration | 0.9739 | 0.9848 | 0.9774 | 0.9903 | 0.0253 | 1 | 0 | 0 | 503 | 12 |
| 0 | J1 (kappa <= 8) | selection | 0.9033 | 0.9029 | 0.9738 | 0.9696 | 0.1092 | 2 | 0 | 1 | 467 | 65 |
| 0 | J1 (kappa <= 8) | calibration | 0.9715 | 0.9541 | 0.9925 | 0.9929 | 0.0759 | 1 | 0 | 0 | 511 | 23 |
| 0 | J1 | selection | 0.9272 | 0.9633 | 0.9607 | 0.9696 | 0.0176 | 2 | 0 | 1 | 465 | 36 |
| 0 | J1 | calibration | 0.9769 | 0.9825 | 0.9874 | 0.9903 | 0.0295 | 1 | 0 | 0 | 507 | 13 |
| 1 | B + ASRF | selection | 0.9067 | 0.9378 | 0.9476 | 0.9566 | 0.0211 | 1 | 2 | 1 | 457 | 49 |
| 1 | B + ASRF | calibration | 0.9778 | 0.9799 | 0.9824 | 0.9929 | 0.0422 | 1 | 0 | 0 | 507 | 12 |
| 1 | L1 + B3 | selection | 0.9062 | 0.9702 | 0.9372 | 0.9886 | 0.0070 | 0 | 0 | 1 | 449 | 40 |
| 1 | L1 + B3 | calibration | 0.9677 | 0.9948 | 0.9623 | 0.9903 | 0.0211 | 1 | 0 | 0 | 495 | 10 |
| 1 | J0 (kappa <= 8) | selection | 0.8940 | 0.8942 | 0.9738 | 0.9780 | 0.0951 | 0 | 1 | 1 | 464 | 72 |
| 1 | J0 (kappa <= 8) | calibration | 0.9658 | 0.9517 | 0.9899 | 0.9903 | 0.0759 | 1 | 0 | 0 | 508 | 26 |
| 1 | J0 | selection | 0.9145 | 0.9479 | 0.9529 | 0.9886 | 0.0176 | 0 | 0 | 1 | 460 | 44 |
| 1 | J0 | calibration | 0.9808 | 0.9801 | 0.9899 | 0.9903 | 0.0338 | 1 | 0 | 0 | 510 | 12 |
| 1 | J1 (kappa <= 8) | selection | 0.8940 | 0.8942 | 0.9738 | 0.9780 | 0.0951 | 0 | 1 | 1 | 464 | 72 |
| 1 | J1 (kappa <= 8) | calibration | 0.9658 | 0.9517 | 0.9899 | 0.9903 | 0.0759 | 1 | 0 | 0 | 508 | 26 |
| 1 | J1 | selection | 0.9268 | 0.9653 | 0.9476 | 0.9886 | 0.0106 | 0 | 0 | 1 | 462 | 33 |
| 1 | J1 | calibration | 0.9846 | 0.9849 | 0.9849 | 0.9903 | 0.0211 | 1 | 0 | 0 | 510 | 8 |
| 2 | B + ASRF | selection | 0.8963 | 0.9284 | 0.9503 | 0.9566 | 0.0141 | 1 | 2 | 1 | 454 | 57 |
| 2 | B + ASRF | calibration | 0.9730 | 0.9749 | 0.9749 | 0.9829 | 0.0338 | 1 | 1 | 0 | 504 | 14 |
| 2 | L1 + B3 | selection | 0.9124 | 0.9607 | 0.9607 | 0.9886 | 0.0915 | 0 | 0 | 1 | 458 | 44 |
| 2 | L1 + B3 | calibration | 0.9693 | 0.9678 | 0.9824 | 0.9835 | 0.1308 | 2 | 0 | 0 | 505 | 19 |
| 2 | J0 (kappa <= 8) | selection | 0.8986 | 0.8983 | 0.9712 | 0.9591 | 0.1197 | 2 | 1 | 1 | 465 | 68 |
| 2 | J0 (kappa <= 8) | calibration | 0.9677 | 0.9565 | 0.9950 | 0.9835 | 0.0844 | 2 | 0 | 0 | 509 | 25 |
| 2 | J0 | selection | 0.9174 | 0.9556 | 0.9581 | 0.9788 | 0.0282 | 1 | 0 | 1 | 461 | 42 |
| 2 | J0 | calibration | 0.9730 | 0.9800 | 0.9849 | 0.9835 | 0.0338 | 2 | 0 | 0 | 505 | 15 |
| 2 | J1 (kappa <= 8) | selection | 0.9012 | 0.9049 | 0.9712 | 0.9682 | 0.1127 | 1 | 1 | 1 | 465 | 65 |
| 2 | J1 (kappa <= 8) | calibration | 0.9695 | 0.9612 | 0.9950 | 0.9835 | 0.0886 | 2 | 0 | 0 | 509 | 23 |
| 2 | J1 | selection | 0.9169 | 0.9655 | 0.9529 | 0.9788 | 0.0141 | 1 | 0 | 1 | 458 | 39 |
| 2 | J1 | calibration | 0.9749 | 0.9874 | 0.9874 | 0.9835 | 0.0211 | 2 | 0 | 0 | 505 | 13 |

## Comparisons

| seed | comparison | half | d multi segment F1@50 (points) | d multi boundary recall (points) | d native macro-F1 (points) |
|---|---|---|---|---|---|
| 0 | J1 - (J0) | selection | +0.85 [+0.00, +2.01] | +1.31 [-0.25, +3.08] | +0.00 [+0.00, +0.00] |
| 0 | J1 - (J0) | calibration | +0.30 [-0.41, +1.25] | +1.01 [-0.24, +2.54] | +0.00 [+0.00, +0.00] |
| 0 | J1 - (L1 + B3) | selection | +1.89 [-0.08, +3.95] | +0.00 [-2.12, +1.85] | -1.90 [-4.98, +0.00] |
| 0 | J1 - (L1 + B3) | calibration | +0.58 [-0.48, +1.61] | +1.01 [+0.00, +2.28] | +0.47 [-0.76, +2.54] |
| 0 | J0 - (L1 + B3) | selection | +1.04 [-0.94, +3.05] | -1.31 [-3.70, +0.78] | -1.90 [-4.98, +0.00] |
| 0 | J0 - (L1 + B3) | calibration | +0.28 [-0.93, +1.40] | +0.00 [-1.84, +1.56] | +0.47 [-0.76, +2.54] |
| 0 | J1 - (B + ASRF) | selection | +2.50 [+0.41, +4.68] | +2.09 [+0.00, +4.38] | +1.30 [-2.27, +4.83] |
| 0 | J1 - (B + ASRF) | calibration | -0.09 [-1.20, +1.14] | +1.51 [+0.24, +3.16] | -0.27 [-0.92, +0.00] |
| 0 | J0 - (J0 (kappa <= 8)) | selection | +1.46 [-0.67, +3.62] | -2.62 [-5.04, -0.54] | +0.00 [-2.32, +2.93] |
| 0 | J0 - (J0 (kappa <= 8)) | calibration | +0.43 [-0.41, +1.26] | -1.01 [-2.38, +0.00] | -0.27 [-0.92, +0.00] |
| 0 | J1 - (J1 (kappa <= 8)) | selection | +2.39 [+0.26, +4.51] | -1.31 [-3.43, +0.51] | +0.00 [-2.32, +2.93] |
| 0 | J1 - (J1 (kappa <= 8)) | calibration | +0.54 [-0.33, +1.53] | -0.50 [-1.55, +0.50] | -0.27 [-0.92, +0.00] |
| 0 | J1 - (E0 + B3) | selection | -0.72 [-3.49, +2.16] | +0.00 [-2.10, +2.05] | +0.46 [-4.19, +4.91] |
| 0 | J1 - (E0 + B3) | calibration | +4.29 [+2.06, +6.74] | +3.27 [+1.31, +5.48] | +2.43 [-0.94, +6.19] |
| 0 | J1 - (B7 + B3) | selection | +4.74 [+2.25, +7.40] | +0.52 [-1.89, +2.60] | +1.30 [-2.27, +4.83] |
| 0 | J1 - (B7 + B3) | calibration | +0.47 [-0.69, +1.78] | +1.26 [-0.50, +3.13] | -0.24 [-2.56, +2.06] |
| 1 | J1 - (J0) | selection | +1.23 [+0.18, +2.52] | -0.52 [-1.34, +0.00] | +0.00 [+0.00, +0.00] |
| 1 | J1 - (J0) | calibration | +0.38 [-0.60, +1.60] | -0.50 [-1.28, +0.00] | +0.00 [+0.00, +0.00] |
| 1 | J1 - (L1 + B3) | selection | +2.06 [+0.35, +3.96] | +1.05 [-0.26, +2.75] | +0.00 [+0.00, +0.00] |
| 1 | J1 - (L1 + B3) | calibration | +1.68 [+0.31, +3.20] | +2.26 [+0.71, +4.13] | +0.00 [+0.00, +0.00] |
| 1 | J0 - (L1 + B3) | selection | +0.84 [-0.84, +2.69] | +1.57 [+0.27, +3.23] | +0.00 [+0.00, +0.00] |
| 1 | J0 - (L1 + B3) | calibration | +1.30 [-0.05, +2.73] | +2.76 [+1.19, +4.65] | +0.00 [+0.00, +0.00] |
| 1 | J1 - (B + ASRF) | selection | +2.00 [+0.16, +3.84] | +0.00 [-2.26, +2.14] | +3.20 [+0.00, +7.13] |
| 1 | J1 - (B + ASRF) | calibration | +0.67 [-0.58, +2.10] | +0.25 [-1.03, +1.54] | -0.27 [-0.92, +0.00] |
| 1 | J0 - (J0 (kappa <= 8)) | selection | +2.05 [-0.11, +4.24] | -2.09 [-4.29, -0.50] | +1.07 [+0.00, +3.19] |
| 1 | J0 - (J0 (kappa <= 8)) | calibration | +1.50 [+0.64, +2.64] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| 1 | J1 - (J1 (kappa <= 8)) | selection | +3.28 [+0.92, +5.75] | -2.62 [-4.84, -0.79] | +1.07 [+0.00, +3.19] |
| 1 | J1 - (J1 (kappa <= 8)) | calibration | +1.88 [+0.73, +3.20] | -0.50 [-1.28, +0.00] | +0.00 [+0.00, +0.00] |
| 1 | J1 - (E0 + B3) | selection | -0.76 [-3.06, +1.52] | -1.31 [-3.25, +0.51] | +2.36 [-1.83, +7.12] |
| 1 | J1 - (E0 + B3) | calibration | +5.06 [+3.01, +7.33] | +3.02 [+1.26, +5.04] | +2.43 [-0.94, +6.19] |
| 1 | J1 - (B7 + B3) | selection | +4.69 [+2.47, +7.08] | -0.79 [-3.14, +1.31] | +3.20 [+0.00, +7.13] |
| 1 | J1 - (B7 + B3) | calibration | +1.24 [-0.28, +2.98] | +1.01 [-0.76, +2.92] | -0.24 [-2.56, +2.06] |
| 2 | J1 - (J0) | selection | -0.05 [-0.97, +0.79] | -0.52 [-1.60, +0.51] | +0.00 [+0.00, +0.00] |
| 2 | J1 - (J0) | calibration | +0.19 [+0.00, +0.59] | +0.25 [+0.00, +0.79] | +0.00 [+0.00, +0.00] |
| 2 | J1 - (L1 + B3) | selection | +0.46 [-1.33, +2.22] | -0.79 [-3.17, +1.29] | -0.99 [-3.00, +0.00] |
| 2 | J1 - (L1 + B3) | calibration | +0.56 [-0.61, +1.61] | +0.50 [-0.78, +1.82] | +0.00 [+0.00, +0.00] |
| 2 | J0 - (L1 + B3) | selection | +0.51 [-1.16, +2.25] | -0.26 [-2.17, +1.57] | -0.99 [-3.00, +0.00] |
| 2 | J0 - (L1 + B3) | calibration | +0.37 [-0.71, +1.32] | +0.25 [-1.19, +1.64] | +0.00 [+0.00, +0.00] |
| 2 | J1 - (B + ASRF) | selection | +2.06 [-0.13, +4.12] | +0.26 [-1.91, +2.02] | +2.21 [+0.00, +5.45] |
| 2 | J1 - (B + ASRF) | calibration | +0.19 [-0.57, +1.06] | +1.26 [+0.00, +2.65] | +0.05 [-2.25, +2.18] |
| 2 | J0 - (J0 (kappa <= 8)) | selection | +1.89 [+0.31, +3.57] | -1.31 [-2.97, +0.00] | +1.97 [-1.57, +5.66] |
| 2 | J0 - (J0 (kappa <= 8)) | calibration | +0.53 [-0.22, +1.38] | -1.01 [-2.04, -0.24] | +0.00 [+0.00, +0.00] |
| 2 | J1 - (J1 (kappa <= 8)) | selection | +1.58 [-0.45, +3.56] | -1.83 [-4.05, +0.00] | +1.06 [-2.11, +4.97] |
| 2 | J1 - (J1 (kappa <= 8)) | calibration | +0.54 [-0.22, +1.44] | -0.75 [-1.90, +0.26] | +0.00 [+0.00, +0.00] |
| 2 | J1 - (E0 + B3) | selection | -1.75 [-4.38, +0.84] | -0.79 [-2.82, +1.13] | +1.38 [-2.44, +5.47] |
| 2 | J1 - (E0 + B3) | calibration | +4.09 [+1.77, +6.60] | +3.27 [+1.34, +5.42] | +1.75 [-2.12, +5.65] |
| 2 | J1 - (B7 + B3) | selection | +3.71 [+1.21, +6.30] | -0.26 [-2.49, +1.75] | +2.21 [+0.00, +5.45] |
| 2 | J1 - (B7 + B3) | calibration | +0.28 [-1.00, +1.70] | +1.26 [-0.51, +3.16] | -0.92 [-3.79, +1.65] |

## Gains and losses against the kappa <= 8 choices

| seed | system | half | found by old | kept | lost | found only now | recovered kept | extra changes old/new | unmatched predicted old/new | units fixed/lost |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | J0 | selection | 372 | 362 | 10 | 0 | 7 of 12 | 35/13 | 64/37 | 3/12 |
| 0 | J0 | calibration | 393 | 389 | 4 | 0 | 4 of 7 | 17/6 | 24/12 | 0/7 |
| 0 | J1 | selection | 372 | 365 | 7 | 2 | 9 of 12 | 37/14 | 65/36 | 6/8 |
| 0 | J1 | calibration | 395 | 392 | 3 | 1 | 6 of 8 | 16/7 | 23/13 | 1/5 |
| 1 | J0 | selection | 372 | 364 | 8 | 0 | 4 of 7 | 41/20 | 72/44 | 6/10 |
| 1 | J0 | calibration | 394 | 394 | 0 | 0 | 4 of 4 | 16/8 | 26/12 | 2/0 |
| 1 | J1 | selection | 372 | 362 | 10 | 0 | 4 of 7 | 41/13 | 72/33 | 8/10 |
| 1 | J1 | calibration | 394 | 392 | 2 | 0 | 3 of 4 | 16/6 | 26/8 | 3/1 |
| 2 | J0 | selection | 371 | 366 | 5 | 0 | 5 of 8 | 39/17 | 68/42 | 2/6 |
| 2 | J0 | calibration | 396 | 392 | 4 | 0 | 5 of 8 | 15/8 | 25/15 | 1/5 |
| 2 | J1 | selection | 371 | 363 | 8 | 1 | 4 of 8 | 37/13 | 65/39 | 3/10 |
| 2 | J1 | calibration | 396 | 392 | 4 | 1 | 5 of 8 | 14/5 | 23/13 | 1/5 |

## References

| system | half | segment F1@50 |
|---|---|---|
| E0 + B3 | selection | 0.9344 |
| E0 + B3 | calibration | 0.9340 |
| E1 + B3 | selection | 0.8940 |
| E1 + B3 | calibration | 0.9589 |
| B7 + B3 | selection | 0.8798 |
| B7 + B3 | calibration | 0.9721 |
| E1 + ASRF s0 | selection | 0.9100 |
| E1 + ASRF s0 | calibration | 0.9758 |
| E1 + ASRF s1 | selection | 0.9089 |
| E1 + ASRF s1 | calibration | 0.9643 |
| E1 + ASRF s2 | selection | 0.9071 |
| E1 + ASRF s2 | calibration | 0.9644 |

## Stable rule, goals and decision

```
{
 "stable": {
  "J1 - (J0)": {
   "stable": false,
   "selection_above_zero": [
    false,
    true,
    false
   ],
   "calibration_below_zero": [
    false,
    false,
    false
   ]
  },
  "J1 - (L1 + B3)": {
   "stable": false,
   "selection_above_zero": [
    false,
    true,
    false
   ],
   "calibration_below_zero": [
    false,
    false,
    false
   ]
  },
  "J0 - (L1 + B3)": {
   "stable": false,
   "selection_above_zero": [
    false,
    false,
    false
   ],
   "calibration_below_zero": [
    false,
    false,
    false
   ]
  },
  "J1 - (B + ASRF)": {
   "stable": false,
   "selection_above_zero": [
    true,
    true,
    false
   ],
   "calibration_below_zero": [
    false,
    false,
    false
   ]
  },
  "J0 - (J0 (kappa <= 8))": {
   "stable": false,
   "selection_above_zero": [
    false,
    false,
    true
   ],
   "calibration_below_zero": [
    false,
    false,
    false
   ]
  },
  "J1 - (J1 (kappa <= 8))": {
   "stable": false,
   "selection_above_zero": [
    true,
    true,
    false
   ],
   "calibration_below_zero": [
    false,
    false,
    false
   ]
  }
 },
 "goal_J0": {
  "mean_segment_f1": {
   "selection": 0.916894042198337,
   "calibration": 0.9758856050802734
  },
  "required": {
   "selection": 0.944409687184662,
   "calibration": 0.9821421709894332
  },
  "margin": {
   "selection": -0.027515644986324972,
   "calibration": -0.006256565909159795
  },
  "per_seed": [
   {
    "selection": {
     "feasible": true,
     "recall": 0.9476439790575916,
     "native": 0.9696431360186241
    },
    "calibration": {
     "feasible": true,
     "recall": 0.9773869346733668,
     "native": 0.9902810262914544
    }
   },
   {
    "selection": {
     "feasible": true,
     "recall": 0.9528795811518325,
     "native": 0.9886353840806397
    },
    "calibration": {
     "feasible": true,
     "recall": 0.9899497487437185,
     "native": 0.9902810262914544
    }
   },
   {
    "selection": {
     "feasible": true,
     "recall": 0.9581151832460733,
     "native": 0.9787836625425645
    },
    "calibration": {
     "feasible": true,
     "recall": 0.9849246231155779,
     "native": 0.9834796751360362
    }
   }
  ],
  "met": false,
  "sensitivity_calibration": -0.008007973394502876
 },
 "goal_J1": {
  "mean_segment_f1": {
   "selection": 0.923638534301697,
   "calibration": 0.9787793573920741
  },
  "required": {
   "selection": 0.944409687184662,
   "calibration": 0.9821421709894332
  },
  "margin": {
   "selection": -0.02077115288296505,
   "calibration": -0.0033628135973591267
  },
  "per_seed": [
   {
    "selection": {
     "feasible": true,
     "recall": 0.9607329842931938,
     "native": 0.9696431360186241
    },
    "calibration": {
     "feasible": true,
     "recall": 0.9874371859296482,
     "native": 0.9902810262914544
    }
   },
   {
    "selection": {
     "feasible": true,
     "recall": 0.9476439790575916,
     "native": 0.9886353840806397
    },
    "calibration": {
     "feasible": true,
     "recall": 0.9849246231155779,
     "native": 0.9902810262914544
    }
   },
   {
    "selection": {
     "feasible": true,
     "recall": 0.9528795811518325,
     "native": 0.9787836625425645
    },
    "calibration": {
     "feasible": true,
     "recall": 0.9874371859296482,
     "native": 0.9834796751360362
    }
   }
  ],
  "met": false,
  "sensitivity_calibration": -0.005114221082702208
 },
 "decision": {
  "case": "neither_improves",
  "action": "neither J0 nor J1 improves: the decoding search on these fixed outputs ends",
  "j0_improves": false,
  "j1_improves": false,
  "j1_stably_better_than_j0": false,
  "goal_met": {
   "J0": false,
   "J1": false
  }
 }
}
```
