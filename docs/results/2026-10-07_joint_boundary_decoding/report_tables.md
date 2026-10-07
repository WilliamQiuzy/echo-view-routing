# Joint boundary decoding: tables

## Choices (calibration only)

| seed | system | status | lambda | kappa | feasible settings | calibration segment F1@50 |
|---|---|---|---|---|---|---|
| 0 | J0 | feasible | 0.0000 | 8.0000 | 5 of 5 | 0.9696 |
| 0 | J1 | feasible | 0.5000 | 8.0000 | 22 of 25 | 0.9715 |
| 1 | J0 | feasible | 0.0000 | 8.0000 | 3 of 5 | 0.9658 |
| 1 | J1 | feasible | 0.0000 | 8.0000 | 21 of 25 | 0.9658 |
| 2 | J0 | feasible | 0.0000 | 8.0000 | 5 of 5 | 0.9677 |
| 2 | J1 | feasible | 0.2500 | 8.0000 | 25 of 25 | 0.9695 |

## Systems

| seed | system | half | segment F1@50 | boundary P | boundary R | native macro-F1 | cuts per clip | A4C read as A5C | A5C read as A4C | SC4C read as PLAX | A5C right |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | B + ASRF | selection | 0.9022 | 0.9447 | 0.9398 | 0.9566 | 0.0352 | 1 | 2 | 1 | 9/11 |
| 0 | B + ASRF | calibration | 0.9777 | 0.9797 | 0.9724 | 0.9929 | 0.0253 | 1 | 0 | 0 | 18/18 |
| 0 | L1 + B3 | selection | 0.9084 | 0.9607 | 0.9607 | 0.9886 | 0.0951 | 0 | 0 | 1 | 11/11 |
| 0 | L1 + B3 | calibration | 0.9711 | 0.9725 | 0.9774 | 0.9856 | 0.1013 | 1 | 1 | 0 | 17/18 |
| 0 | J0 | selection | 0.9042 | 0.9051 | 0.9738 | 0.9696 | 0.1056 | 2 | 0 | 1 | 11/11 |
| 0 | J0 | calibration | 0.9696 | 0.9493 | 0.9874 | 0.9929 | 0.0759 | 1 | 0 | 0 | 18/18 |
| 0 | J1 | selection | 0.9033 | 0.9029 | 0.9738 | 0.9696 | 0.1092 | 2 | 0 | 1 | 11/11 |
| 0 | J1 | calibration | 0.9715 | 0.9541 | 0.9925 | 0.9929 | 0.0759 | 1 | 0 | 0 | 18/18 |
| 1 | B + ASRF | selection | 0.9067 | 0.9378 | 0.9476 | 0.9566 | 0.0211 | 1 | 2 | 1 | 9/11 |
| 1 | B + ASRF | calibration | 0.9778 | 0.9799 | 0.9824 | 0.9929 | 0.0422 | 1 | 0 | 0 | 18/18 |
| 1 | L1 + B3 | selection | 0.9062 | 0.9702 | 0.9372 | 0.9886 | 0.0070 | 0 | 0 | 1 | 11/11 |
| 1 | L1 + B3 | calibration | 0.9677 | 0.9948 | 0.9623 | 0.9903 | 0.0211 | 1 | 0 | 0 | 18/18 |
| 1 | J0 | selection | 0.8940 | 0.8942 | 0.9738 | 0.9780 | 0.0951 | 0 | 1 | 1 | 10/11 |
| 1 | J0 | calibration | 0.9658 | 0.9517 | 0.9899 | 0.9903 | 0.0759 | 1 | 0 | 0 | 18/18 |
| 1 | J1 | selection | 0.8940 | 0.8942 | 0.9738 | 0.9780 | 0.0951 | 0 | 1 | 1 | 10/11 |
| 1 | J1 | calibration | 0.9658 | 0.9517 | 0.9899 | 0.9903 | 0.0759 | 1 | 0 | 0 | 18/18 |
| 2 | B + ASRF | selection | 0.8963 | 0.9284 | 0.9503 | 0.9566 | 0.0141 | 1 | 2 | 1 | 9/11 |
| 2 | B + ASRF | calibration | 0.9730 | 0.9749 | 0.9749 | 0.9829 | 0.0338 | 1 | 1 | 0 | 17/18 |
| 2 | L1 + B3 | selection | 0.9124 | 0.9607 | 0.9607 | 0.9886 | 0.0915 | 0 | 0 | 1 | 11/11 |
| 2 | L1 + B3 | calibration | 0.9693 | 0.9678 | 0.9824 | 0.9835 | 0.1308 | 2 | 0 | 0 | 18/18 |
| 2 | J0 | selection | 0.8986 | 0.8983 | 0.9712 | 0.9591 | 0.1197 | 2 | 1 | 1 | 10/11 |
| 2 | J0 | calibration | 0.9677 | 0.9565 | 0.9950 | 0.9835 | 0.0844 | 2 | 0 | 0 | 18/18 |
| 2 | J1 | selection | 0.9012 | 0.9049 | 0.9712 | 0.9682 | 0.1127 | 1 | 1 | 1 | 10/11 |
| 2 | J1 | calibration | 0.9695 | 0.9612 | 0.9950 | 0.9835 | 0.0886 | 2 | 0 | 0 | 18/18 |

## Comparisons

| seed | comparison | half | d multi segment F1@50 (points) | d multi boundary recall (points) | d native macro-F1 (points) |
|---|---|---|---|---|---|
| 0 | J1 - (J0) | selection | -0.09 [-1.04, +0.81] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| 0 | J1 - (J0) | calibration | +0.19 [-0.20, +0.69] | +0.50 [-0.27, +1.54] | +0.00 [+0.00, +0.00] |
| 0 | J1 - (L1 + B3) | selection | -0.51 [-2.21, +1.39] | +1.31 [+0.26, +2.58] | -1.90 [-5.57, +0.00] |
| 0 | J1 - (L1 + B3) | calibration | +0.04 [-0.94, +0.88] | +1.51 [+0.25, +2.89] | +0.73 [+0.00, +2.68] |
| 0 | J1 - (B + ASRF) | selection | +0.11 [-1.85, +2.08] | +3.40 [+1.30, +5.68] | +1.30 [-3.59, +5.54] |
| 0 | J1 - (B + ASRF) | calibration | -0.63 [-1.87, +0.59] | +2.01 [+0.71, +3.58] | +0.00 [+0.00, +0.00] |
| 0 | J0 - (L1 + B3) | selection | -0.42 [-2.08, +1.56] | +1.31 [+0.26, +2.58] | -1.90 [-5.57, +0.00] |
| 0 | J0 - (L1 + B3) | calibration | -0.15 [-1.14, +0.71] | +1.01 [-0.26, +2.34] | +0.73 [+0.00, +2.68] |
| 0 | J1 - (E0 + B3) | selection | -3.11 [-5.88, -0.12] | +1.31 [-0.52, +3.19] | +0.46 [-5.67, +5.58] |
| 0 | J1 - (E0 + B3) | calibration | +3.75 [+1.63, +6.02] | +3.77 [+1.66, +6.01] | +2.70 [-0.74, +6.49] |
| 0 | J1 - (B7 + B3) | selection | +2.34 [+0.43, +4.46] | +1.83 [+0.25, +3.66] | +1.30 [-3.59, +5.54] |
| 0 | J1 - (B7 + B3) | calibration | -0.07 [-1.38, +1.37] | +1.76 [+0.48, +3.40] | +0.03 [-2.26, +2.43] |
| 1 | J1 - (J0) | selection | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| 1 | J1 - (J0) | calibration | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| 1 | J1 - (L1 + B3) | selection | -1.21 [-3.69, +1.54] | +3.66 [+1.37, +6.23] | -1.07 [-3.19, +0.00] |
| 1 | J1 - (L1 + B3) | calibration | -0.20 [-1.93, +1.48] | +2.76 [+1.19, +4.65] | +0.00 [+0.00, +0.00] |
| 1 | J1 - (B + ASRF) | selection | -1.27 [-3.42, +0.80] | +2.62 [+0.79, +4.88] | +2.13 [+0.00, +6.06] |
| 1 | J1 - (B + ASRF) | calibration | -1.20 [-2.69, +0.11] | +0.75 [-0.52, +2.35] | -0.27 [-0.92, +0.00] |
| 1 | J0 - (L1 + B3) | selection | -1.21 [-3.69, +1.54] | +3.66 [+1.37, +6.23] | -1.07 [-3.19, +0.00] |
| 1 | J0 - (L1 + B3) | calibration | -0.20 [-1.93, +1.48] | +2.76 [+1.19, +4.65] | +0.00 [+0.00, +0.00] |
| 1 | J1 - (E0 + B3) | selection | -4.04 [-7.05, -0.96] | +1.31 [-0.52, +3.19] | +1.29 [-2.25, +6.15] |
| 1 | J1 - (E0 + B3) | calibration | +3.18 [+1.16, +5.38] | +3.52 [+1.72, +5.65] | +2.43 [-0.94, +6.19] |
| 1 | J1 - (B7 + B3) | selection | +1.42 [-0.55, +3.57] | +1.83 [+0.25, +3.66] | +2.13 [+0.00, +6.06] |
| 1 | J1 - (B7 + B3) | calibration | -0.64 [-2.08, +0.87] | +1.51 [+0.00, +3.30] | -0.24 [-2.56, +2.06] |
| 2 | J1 - (J0) | selection | +0.26 [+0.00, +0.60] | +0.00 [+0.00, +0.00] | +0.91 [+0.00, +2.86] |
| 2 | J1 - (J0) | calibration | +0.18 [+0.00, +0.46] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| 2 | J1 - (L1 + B3) | selection | -1.12 [-2.70, +0.60] | +1.05 [+0.25, +2.27] | -2.05 [-5.19, +0.00] |
| 2 | J1 - (L1 + B3) | calibration | +0.02 [-1.19, +1.07] | +1.26 [+0.00, +2.60] | +0.00 [+0.00, +0.00] |
| 2 | J1 - (B + ASRF) | selection | +0.48 [-1.50, +2.48] | +2.09 [+0.51, +3.94] | +1.15 [-2.51, +5.15] |
| 2 | J1 - (B + ASRF) | calibration | -0.34 [-1.14, +0.40] | +2.01 [+0.75, +3.41] | +0.05 [-2.25, +2.18] |
| 2 | J0 - (L1 + B3) | selection | -1.38 [-2.93, +0.32] | +1.05 [+0.25, +2.27] | -2.96 [-5.91, +0.00] |
| 2 | J0 - (L1 + B3) | calibration | -0.16 [-1.35, +0.88] | +1.26 [+0.00, +2.60] | +0.00 [+0.00, +0.00] |
| 2 | J1 - (E0 + B3) | selection | -3.32 [-6.16, -0.29] | +1.05 [-0.80, +3.04] | +0.31 [-4.51, +5.27] |
| 2 | J1 - (E0 + B3) | calibration | +3.55 [+1.29, +5.98] | +4.02 [+1.90, +6.23] | +1.75 [-2.12, +5.65] |
| 2 | J1 - (B7 + B3) | selection | +2.13 [+0.02, +4.32] | +1.57 [-0.25, +3.46] | +1.15 [-2.51, +5.15] |
| 2 | J1 - (B7 + B3) | calibration | -0.26 [-1.49, +0.95] | +2.01 [+0.74, +3.65] | -0.92 [-3.79, +1.65] |

## Diagnostics

| seed | system | half | three clips right | merged (B + ASRF) found | extra changes | native switches | correct evidence lost |
|---|---|---|---|---|---|---|---|
| 0 | B + ASRF | selection | 0 |  | 21 (16 rec) | 10 | 34 fr / 26 rec |
| 0 | B + ASRF | calibration |  |  | 8 (11 rec) | 6 | 12 fr / 11 rec |
| 0 | L1 + B3 | selection | 3 |  | 15 (9 rec) | 27 | 32 fr / 24 rec |
| 0 | L1 + B3 | calibration |  |  | 11 (10 rec) | 24 | 11 fr / 9 rec |
| 0 | J0 | selection | 3 | 12 of 14 | 35 (18 rec) | 30 | 20 fr / 17 rec |
| 0 | J0 | calibration |  | 7 of 10 | 17 (15 rec) | 18 | 4 fr / 4 rec |
| 0 | J1 | selection | 3 | 12 of 14 | 37 (18 rec) | 31 | 20 fr / 17 rec |
| 0 | J1 | calibration |  | 8 of 10 | 16 (15 rec) | 18 | 4 fr / 4 rec |
| 1 | B + ASRF | selection | 0 |  | 20 (11 rec) | 6 | 27 fr / 23 rec |
| 1 | B + ASRF | calibration |  |  | 7 (9 rec) | 10 | 10 fr / 8 rec |
| 1 | L1 + B3 | selection | 3 |  | 11 (9 rec) | 2 | 38 fr / 29 rec |
| 1 | L1 + B3 | calibration |  |  | 2 (1 rec) | 5 | 20 fr / 16 rec |
| 1 | J0 | selection | 2 | 7 of 10 | 41 (18 rec) | 27 | 22 fr / 17 rec |
| 1 | J0 | calibration |  | 4 of 5 | 16 (16 rec) | 18 | 5 fr / 5 rec |
| 1 | J1 | selection | 2 | 7 of 10 | 41 (18 rec) | 27 | 22 fr / 17 rec |
| 1 | J1 | calibration |  | 4 of 5 | 16 (16 rec) | 18 | 5 fr / 5 rec |
| 2 | B + ASRF | selection | 0 |  | 27 (18 rec) | 4 | 34 fr / 26 rec |
| 2 | B + ASRF | calibration |  |  | 10 (13 rec) | 8 | 11 fr / 10 rec |
| 2 | L1 + B3 | selection | 3 |  | 15 (9 rec) | 26 | 29 fr / 23 rec |
| 2 | L1 + B3 | calibration |  |  | 13 (11 rec) | 31 | 9 fr / 9 rec |
| 2 | J0 | selection | 2 | 8 of 13 | 39 (19 rec) | 34 | 20 fr / 17 rec |
| 2 | J0 | calibration |  | 8 of 9 | 15 (15 rec) | 20 | 5 fr / 5 rec |
| 2 | J1 | selection | 2 | 8 of 13 | 37 (19 rec) | 32 | 20 fr / 17 rec |
| 2 | J1 | calibration |  | 8 of 9 | 14 (14 rec) | 21 | 5 fr / 5 rec |

## References

| system | half | segment F1@50 | boundary R | native macro-F1 |
|---|---|---|---|---|
| E0 + B3 | selection | 0.9344 | 0.9607 | 0.9650 |
| E0 + B3 | calibration | 0.9340 | 0.9548 | 0.9660 |
| E1 + B3 | selection | 0.8940 | 0.9555 | 0.9780 |
| E1 + B3 | calibration | 0.9589 | 0.9774 | 0.9856 |
| B7 + B3 | selection | 0.8798 | 0.9555 | 0.9566 |
| B7 + B3 | calibration | 0.9721 | 0.9749 | 0.9927 |
| E1 + ASRF s0 | selection | 0.9100 | 0.9424 | 0.9477 |
| E1 + ASRF s0 | calibration | 0.9758 | 0.9774 | 0.9929 |
| E1 + ASRF s1 | selection | 0.9089 | 0.9503 | 0.9664 |
| E1 + ASRF s1 | calibration | 0.9643 | 0.9724 | 0.9753 |
| E1 + ASRF s2 | selection | 0.9071 | 0.9529 | 0.9664 |
| E1 + ASRF s2 | calibration | 0.9644 | 0.9774 | 0.9753 |

## Stable rule and goal

```
{
 "stable": {
  "J1 - (J0)": {
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
  "J1 - (L1 + B3)": {
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
    false,
    false,
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
   "selection": 0.8989134442323219,
   "calibration": 0.967680608365019
  },
  "required": {
   "selection": 0.944409687184662,
   "calibration": 0.9821421709894332
  },
  "margin": {
   "selection": -0.045496242952340116,
   "calibration": -0.014461562624414226
  },
  "per_seed": [
   {
    "selection": {
     "feasible": true,
     "recall": 0.9738219895287958,
     "native": 0.9696431360186241
    },
    "calibration": {
     "feasible": true,
     "recall": 0.9874371859296482,
     "native": 0.9929417020326111
    }
   },
   {
    "selection": {
     "feasible": true,
     "recall": 0.9738219895287958,
     "native": 0.977968717413973
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
     "recall": 0.9712041884816754,
     "native": 0.959080219466415
    },
    "calibration": {
     "feasible": true,
     "recall": 0.9949748743718593,
     "native": 0.9834796751360362
    }
   }
  ],
  "met": false,
  "sensitivity_calibration": -0.016212970109757308
 },
 "goal_J1": {
  "mean_segment_f1": {
   "selection": 0.8994926556033489,
   "calibration": 0.9689287223127527
  },
  "required": {
   "selection": 0.944409687184662,
   "calibration": 0.9821421709894332
  },
  "margin": {
   "selection": -0.044917031581313105,
   "calibration": -0.013213448676680506
  },
  "per_seed": [
   {
    "selection": {
     "feasible": true,
     "recall": 0.9738219895287958,
     "native": 0.9696431360186241
    },
    "calibration": {
     "feasible": true,
     "recall": 0.992462311557789,
     "native": 0.9929417020326111
    }
   },
   {
    "selection": {
     "feasible": true,
     "recall": 0.9738219895287958,
     "native": 0.977968717413973
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
     "recall": 0.9712041884816754,
     "native": 0.9681547153241088
    },
    "calibration": {
     "feasible": true,
     "recall": 0.9949748743718593,
     "native": 0.9834796751360362
    }
   }
  ],
  "met": false,
  "sensitivity_calibration": -0.014964856162023588
 }
}
```
