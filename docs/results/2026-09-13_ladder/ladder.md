# Baseline ladder (merged) · encoder 79f4a41af6e3 · methods: b_file, b0, b1, b2, b3, b4

### Temporal routing ladder (test, frozen validation-selected policy)

| method | native_cine_macroF1 | native_cine_balAcc | native_frame_acc | frag_per_min | share_fragmented | boundaryF1@0.5s | falseSplit_same_none | falseSplit_same_edit | missed_diff_none | missed_diff_edit | tau@5% | coverage@5% | risk@5% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b0 | 0.957 | 0.962 | 0.951 | 23.136 | 0.307 | 0.499 | 0.033 | 0.133 | 0.000 | 0.000 | 0.461 | 0.967 | 0.007 |
| b1 | 0.949 | 0.960 | 0.968 | 2.363 | 0.086 | 0.918 | 0.000 | 0.017 | 0.017 | 0.017 | 0.000 | 0.999 | 0.014 |
| b2 | 0.946 | 0.956 | 0.967 | 2.378 | 0.084 | 0.906 | 0.000 | 0.000 | 0.000 | 0.000 | 0.032 | 0.996 | 0.009 |
| b3 | 0.946 | 0.955 | 0.963 | 3.550 | 0.129 | 0.913 | 0.000 | 0.017 | 0.000 | 0.000 | 0.026 | 0.989 | 0.010 |
| b4 | 0.568 | 0.598 | 0.915 | 0.715 | 0.062 | 0.779 | 0.000 | 0.017 | 0.067 | 0.117 | 0.982 | 0.020 | 0.000 |

### B-file operational comparator (native files; known file boundaries)

| target_risk | tau | status | val_coverage | val_risk | test_coverage | test_risk | test_cine_macroF1 |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.381 | ok | 1.000 | 0.018 | 0.998 | 0.019 | 0.938 |
| 0.01 | 0.590 | ok | 0.973 | 0.005 | 0.974 | 0.011 | 0.938 |
| 0.1 | 0.381 | ok | 1.000 | 0.018 | 0.998 | 0.019 | 0.938 |
