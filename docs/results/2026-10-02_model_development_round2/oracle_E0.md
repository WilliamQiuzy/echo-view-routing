# ORACLE_DIAGNOSTIC (E0): classification versus segmentation errors on multi-fragment streams

Reference segmentation is given only in the oracle columns. Native and stream columns use the default decoding.

## selection half

| Method | native macro-F1 | native clips wrong | A5C clips right / all | stream sample error | oracle wrong segments | oracle segment error | oracle sample error | oracle (pre-pooling) segment error | default runs: timing / whole / interior ≤3 / interior >3 | wrong source clips (occurrences) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| B0 (E0) | 0.965 | 3 | 8/11 | 0.0868 | 23/502 | 0.0458 | 0.0499 |  | 121 / 1 / 163 / 87 | 6 (28) |
| M1_s0 | 0.965 | 3 | 8/11 | 0.0927 | 31/502 | 0.0618 | 0.0721 |  | 5 / 59 / 0 / 2 | 10 (46) |
| M3p_s0 | 0.965 | 3 | 8/11 | 0.0704 | 25/502 | 0.0498 | 0.0507 | 0.0398 | 25 / 24 / 2 / 14 | 7 (30) |
| M5_s0 | 0.963 | 4 | 8/11 | 0.0646 | 26/502 | 0.0518 | 0.0536 | 0.0498 | 17 / 27 / 1 / 11 | 8 (33) |
| M1_s1 | 0.968 | 3 | 9/11 | 0.0473 | 17/502 | 0.0339 | 0.0348 |  | 3 / 29 / 0 / 0 | 5 (19) |
| M3p_s1 | 0.965 | 3 | 8/11 | 0.0609 | 29/502 | 0.0578 | 0.0543 | 0.0458 | 4 / 38 / 2 / 1 | 13 (36) |
| M5_s1 | 0.965 | 3 | 8/11 | 0.0698 | 33/502 | 0.0657 | 0.0649 | 0.0498 | 4 / 42 / 1 / 4 | 16 (42) |
| M1_s2 | 0.965 | 3 | 8/11 | 0.0682 | 22/502 | 0.0438 | 0.0466 |  | 55 / 33 / 0 / 5 | 8 (28) |
| M3p_s2 | 0.965 | 3 | 8/11 | 0.0593 | 24/502 | 0.0478 | 0.0422 | 0.0359 | 14 / 28 / 1 / 5 | 9 (28) |
| M5_s2 | 0.965 | 3 | 8/11 | 0.0616 | 27/502 | 0.0538 | 0.0508 | 0.0378 | 9 / 29 / 0 / 5 | 10 (32) |

## calibration half

| Method | native macro-F1 | native clips wrong | A5C clips right / all | stream sample error | oracle wrong segments | oracle segment error | oracle sample error | oracle (pre-pooling) segment error | default runs: timing / whole / interior ≤3 / interior >3 | wrong source clips (occurrences) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| B0 (E0) | 0.982 | 3 | 16/18 | 0.0606 | 18/518 | 0.0347 | 0.0257 |  | 116 / 4 / 157 / 36 | 11 (23) |
| M1_s0 | 0.971 | 5 | 15/18 | 0.0493 | 25/518 | 0.0483 | 0.0331 |  | 3 / 40 / 0 / 1 | 15 (29) |
| M3p_s0 | 0.964 | 6 | 15/18 | 0.0350 | 22/518 | 0.0425 | 0.0318 | 0.0309 | 21 / 21 / 1 / 7 | 15 (28) |
| M5_s0 | 0.970 | 4 | 15/18 | 0.0290 | 20/518 | 0.0386 | 0.0284 | 0.0290 | 10 / 24 / 1 / 2 | 13 (26) |
| M1_s1 | 0.975 | 4 | 16/18 | 0.0323 | 15/518 | 0.0290 | 0.0198 |  | 7 / 27 / 0 / 1 | 11 (18) |
| M3p_s1 | 0.979 | 4 | 16/18 | 0.0276 | 21/518 | 0.0405 | 0.0241 | 0.0309 | 3 / 26 / 1 / 0 | 16 (24) |
| M5_s1 | 0.966 | 6 | 16/18 | 0.0380 | 25/518 | 0.0483 | 0.0331 | 0.0386 | 8 / 29 / 1 / 4 | 16 (31) |
| M1_s2 | 0.970 | 4 | 15/18 | 0.0343 | 17/518 | 0.0328 | 0.0204 |  | 43 / 24 / 0 / 2 | 11 (20) |
| M3p_s2 | 0.972 | 5 | 16/18 | 0.0327 | 24/518 | 0.0463 | 0.0298 | 0.0347 | 9 / 24 / 0 / 3 | 14 (28) |
| M5_s2 | 0.972 | 5 | 16/18 | 0.0335 | 21/518 | 0.0405 | 0.0280 | 0.0290 | 9 / 24 / 0 / 4 | 14 (26) |

