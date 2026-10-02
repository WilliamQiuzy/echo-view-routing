# ORACLE_DIAGNOSTIC (E1): classification versus segmentation errors on multi-fragment streams

Reference segmentation is given only in the oracle columns. Native and stream columns use the default decoding.

## selection half

| Method | native macro-F1 | native clips wrong | A5C clips right / all | stream sample error | oracle wrong segments | oracle segment error | oracle sample error | oracle (pre-pooling) segment error | default runs: timing / whole / interior ≤3 / interior >3 | wrong source clips (occurrences) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| B0 (E1) | 0.978 | 3 | 10/11 | 0.1025 | 23/502 | 0.0458 | 0.0432 |  | 156 / 1 / 275 / 91 | 6 (29) |
| M1_s0 | 0.948 | 6 | 9/11 | 0.0632 | 35/502 | 0.0697 | 0.0502 |  | 22 / 38 / 1 / 1 | 16 (39) |
| M3p_s0 | 0.978 | 3 | 10/11 | 0.0727 | 32/502 | 0.0637 | 0.0608 | 0.0637 | 42 / 33 / 0 / 11 | 12 (42) |
| M5_s0 | 0.978 | 3 | 10/11 | 0.0736 | 33/502 | 0.0657 | 0.0614 | 0.0598 | 38 / 32 / 1 / 9 | 12 (42) |
| M1_s1 | 0.966 | 4 | 9/11 | 0.0742 | 29/502 | 0.0578 | 0.0577 |  | 21 / 40 / 3 / 3 | 10 (39) |
| M3p_s1 | 0.976 | 4 | 10/11 | 0.0755 | 40/502 | 0.0797 | 0.0745 | 0.0737 | 17 / 41 / 1 / 4 | 18 (52) |
| M5_s1 | 0.968 | 4 | 10/11 | 0.0680 | 32/502 | 0.0637 | 0.0557 | 0.0598 | 22 / 35 / 1 / 5 | 16 (40) |
| M1_s2 | 0.966 | 4 | 9/11 | 0.0719 | 29/502 | 0.0578 | 0.0584 |  | 21 / 38 / 2 / 3 | 10 (39) |
| M3p_s2 | 0.978 | 3 | 10/11 | 0.0783 | 35/502 | 0.0697 | 0.0661 | 0.0657 | 32 / 38 / 0 / 8 | 13 (46) |
| M5_s2 | 0.932 | 8 | 9/11 | 0.0669 | 41/502 | 0.0817 | 0.0673 | 0.0737 | 27 / 37 / 3 / 3 | 23 (50) |

## calibration half

| Method | native macro-F1 | native clips wrong | A5C clips right / all | stream sample error | oracle wrong segments | oracle segment error | oracle sample error | oracle (pre-pooling) segment error | default runs: timing / whole / interior ≤3 / interior >3 | wrong source clips (occurrences) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| B0 (E1) | 0.985 | 2 | 16/18 | 0.0528 | 5/518 | 0.0097 | 0.0057 |  | 146 / 2 / 197 / 24 | 5 (6) |
| M1_s0 | 0.993 | 1 | 18/18 | 0.0232 | 10/518 | 0.0193 | 0.0159 |  | 24 / 15 / 1 / 0 | 9 (14) |
| M3p_s0 | 0.978 | 3 | 16/18 | 0.0232 | 13/518 | 0.0251 | 0.0171 | 0.0232 | 20 / 17 / 1 / 1 | 11 (17) |
| M5_s0 | 0.986 | 2 | 17/18 | 0.0249 | 13/518 | 0.0251 | 0.0160 | 0.0193 | 22 / 17 / 0 / 0 | 11 (16) |
| M1_s1 | 0.975 | 4 | 16/18 | 0.0239 | 13/518 | 0.0251 | 0.0178 |  | 13 / 18 / 0 / 3 | 11 (17) |
| M3p_s1 | 0.983 | 3 | 17/18 | 0.0235 | 18/518 | 0.0347 | 0.0231 | 0.0212 | 19 / 15 / 1 / 0 | 16 (22) |
| M5_s1 | 0.983 | 3 | 18/18 | 0.0282 | 16/518 | 0.0309 | 0.0197 | 0.0251 | 25 / 17 / 0 / 0 | 13 (19) |
| M1_s2 | 0.975 | 4 | 16/18 | 0.0262 | 13/518 | 0.0251 | 0.0170 |  | 16 / 18 / 0 / 3 | 11 (17) |
| M3p_s2 | 0.975 | 4 | 16/18 | 0.0231 | 15/518 | 0.0290 | 0.0201 | 0.0251 | 15 / 15 / 3 / 2 | 12 (19) |
| M5_s2 | 0.977 | 4 | 18/18 | 0.0294 | 18/518 | 0.0347 | 0.0243 | 0.0251 | 27 / 17 / 2 / 0 | 16 (21) |

