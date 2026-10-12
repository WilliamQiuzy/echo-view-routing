# Dense target inference round: tables

Recompute: `python scripts/dense_target_inference.py --config configs/experiments/dense_target_inference_v1.yaml --run` (after `--check`).
Each cell gives the selection half, then the calibration half. Every condition uses each seed's inherited B3 numbers.

## Counts and cost

| Half | Sequences (native, pairs, multi) | Samples | Old windows | New windows | Short sequences | Windows with the marker off slot 8 | Windows crossing a join or gamma |
|---|---|---|---|---|---|---|---|
| selection | 284, 137, 120 | 37287 | 7666 | 29621 | 28 | 8020 | 11549 |
| calibration | 237, 115, 120 | 34216 | 7021 | 27195 | 19 | 7011 | 11027 |

```
{
 "new_windows": 170448,
 "inference_seconds": {
  "s0|selection": 478.8266849517822,
  "s0|calibration": 440.98770403862,
  "s1|selection": 471.24430751800537,
  "s1|calibration": 432.5497770309448,
  "s2|selection": 471.44005370140076,
  "s2|calibration": 432.5716700553894
 },
 "ms_per_new_window": 16.002652992678954,
 "frame_decode_seconds": 22.6115825176239,
 "ledger_hours_before_run": 0.03405962275134193,
 "report_seconds": 2977.05690574646
}
```

## Systems with the fixed B3 numbers

| Condition | Seed | Segment F1@50 | Boundary recall | Boundary precision | Native macro-F1 | Label switches per native clip | Same-view join cuts | 10 Hz accuracy, multi | A4C read as A5C (clips) | A5C read as A4C (clips) | Fixed numbers feasible |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Sparse | 0 | 0.938 / 0.982 | 0.958 / 0.982 | 0.948 / 0.975 | 0.9786 / 0.9929 | 0.042 / 0.025 | 0.017 / 0.025 | 0.9296 / 0.9499 | 2 / 1 | 0 / 0 | True |
| Sparse | 1 | 0.960 / 0.984 | 0.969 / 0.987 | 0.959 / 0.980 | 0.9786 / 0.9861 | 0.039 / 0.017 | 0.004 / 0.025 | 0.9426 / 0.9510 | 2 / 2 | 0 / 0 | True |
| Sparse | 2 | 0.964 / 0.980 | 0.974 / 0.980 | 0.959 / 0.977 | 0.9701 / 0.9861 | 0.046 / 0.013 | 0.004 / 0.025 | 0.9427 / 0.9466 | 3 / 2 | 0 / 0 | True |
| Interpolated | 0 | 0.942 / 0.973 | 0.932 / 0.952 | 0.981 / 0.992 | 0.9786 / 0.9929 | 0.021 / 0.008 | 0.013 / 0.010 | 0.9243 / 0.9424 | 2 / 1 | 0 / 0 | True |
| Interpolated | 1 | 0.956 / 0.975 | 0.948 / 0.967 | 0.973 / 0.987 | 0.9877 / 0.9861 | 0.032 / 0.008 | 0.009 / 0.010 | 0.9430 / 0.9477 | 1 / 2 | 0 / 0 | True |
| Interpolated | 2 | 0.947 / 0.934 | 0.903 / 0.897 | 0.989 / 0.997 | 0.9701 / 0.9861 | 0.035 / 0.013 | 0.004 / 0.005 | 0.9215 / 0.9187 | 3 / 2 | 0 / 0 | False |
| Dense | 0 | 0.943 / 0.982 | 0.971 / 0.985 | 0.946 / 0.973 | 0.9701 / 0.9929 | 0.042 / 0.021 | 0.022 / 0.025 | 0.9432 / 0.9578 | 3 / 1 | 0 / 0 | True |
| Dense | 1 | 0.964 / 0.979 | 0.974 / 0.982 | 0.964 / 0.973 | 0.9877 / 0.9861 | 0.039 / 0.021 | 0.009 / 0.020 | 0.9579 / 0.9604 | 1 / 2 | 0 / 0 | True |
| Dense | 2 | 0.969 / 0.977 | 0.979 / 0.980 | 0.971 / 0.975 | 0.9701 / 0.9861 | 0.046 / 0.017 | 0.009 / 0.025 | 0.9574 / 0.9570 | 3 / 2 | 0 / 0 | True |

## Direct argmax

| Condition | Seed | Segment F1@50 | Boundary recall | Boundary precision | Native macro-F1 | Label switches per native clip | Same-view join cuts | 10 Hz accuracy, multi | A4C read as A5C (clips) | A5C read as A4C (clips) |
|---|---|---|---|---|---|---|---|---|---|---|
| Sparse | 0 | 0.893 / 0.961 | 0.979 / 0.987 | 0.846 / 0.942 | 0.9701 / 0.9861 | 0.095 / 0.093 | 0.043 / 0.035 | 0.9258 / 0.9474 | 3 / 2 | 0 / 0 |
| Sparse | 1 | 0.925 / 0.961 | 0.984 / 0.990 | 0.872 / 0.945 | 0.9877 / 0.9796 | 0.113 / 0.101 | 0.030 / 0.040 | 0.9374 / 0.9452 | 1 / 3 | 0 / 0 |
| Sparse | 2 | 0.926 / 0.960 | 0.984 / 0.982 | 0.874 / 0.940 | 0.9701 / 0.9861 | 0.116 / 0.097 | 0.026 / 0.035 | 0.9371 / 0.9432 | 3 / 2 | 0 / 0 |
| Interpolated | 0 | 0.898 / 0.961 | 0.976 / 0.985 | 0.842 / 0.940 | 0.9701 / 0.9861 | 0.095 / 0.093 | 0.048 / 0.035 | 0.9296 / 0.9502 | 3 / 2 | 0 / 0 |
| Interpolated | 1 | 0.931 / 0.961 | 0.990 / 0.987 | 0.877 / 0.942 | 0.9877 / 0.9796 | 0.113 / 0.101 | 0.017 / 0.035 | 0.9445 / 0.9496 | 1 / 3 | 0 / 0 |
| Interpolated | 2 | 0.927 / 0.959 | 0.987 / 0.980 | 0.875 / 0.935 | 0.9701 / 0.9861 | 0.116 / 0.097 | 0.022 / 0.040 | 0.9422 / 0.9474 | 3 / 2 | 0 / 0 |
| Dense | 0 | 0.853 / 0.919 | 0.987 / 0.987 | 0.756 / 0.845 | 0.9701 / 0.9861 | 0.176 / 0.135 | 0.057 / 0.035 | 0.9395 / 0.9550 | 3 / 2 | 0 / 0 |
| Dense | 1 | 0.831 / 0.912 | 0.992 / 0.990 | 0.698 / 0.835 | 0.9877 / 0.9861 | 0.208 / 0.232 | 0.030 / 0.040 | 0.9515 / 0.9576 | 1 / 2 | 0 / 0 |
| Dense | 2 | 0.855 / 0.912 | 0.992 / 0.990 | 0.740 / 0.831 | 0.9786 / 0.9861 | 0.187 / 0.224 | 0.035 / 0.050 | 0.9487 / 0.9539 | 2 / 2 | 0 / 0 |

## Comparisons

### With the fixed B3 numbers

| Comparison | Half | Segment F1@50 (seeds 0, 1, 2) | Verdict | native macro-F1 verdict | native cuts per clip verdict | multi boundary recall verdict | multi boundary F1 verdict | same-class cut rate verdict |
|---|---|---|---|---|---|---|---|---|
| Dense - Sparse | selection | +0.0043 [-0.0066, +0.0177]; +0.0040 [-0.0020, +0.0108]; +0.0049 [-0.0010, +0.0119] | no consistent difference detected | no consistent difference detected | no consistent difference detected | no consistent difference detected | no consistent difference detected | no consistent difference detected |
| Dense - Sparse | calibration | +0.0000 [-0.0038, +0.0039]; -0.0048 [-0.0122, +0.0000]; -0.0029 [-0.0074, +0.0010] | no consistent difference detected | no consistent difference detected | no consistent difference detected | no consistent difference detected | no consistent difference detected | no consistent difference detected |
| Dense - Interpolated | selection | +0.0007 [-0.0146, +0.0178]; +0.0086 [-0.0015, +0.0190]; +0.0217 [+0.0096, +0.0347] | no consistent difference detected | no consistent difference detected | no consistent difference detected | consistent positive | no consistent difference detected | no consistent difference detected |
| Dense - Interpolated | calibration | +0.0092 [-0.0024, +0.0235]; +0.0041 [-0.0038, +0.0157]; +0.0431 [+0.0214, +0.0668] | no consistent difference detected | no consistent difference detected | no consistent difference detected | consistent positive | no consistent difference detected | no consistent difference detected |
| Interpolated - Sparse | selection | +0.0036 [-0.0131, +0.0196]; -0.0046 [-0.0134, +0.0036]; -0.0169 [-0.0306, -0.0027] | no consistent difference detected | no consistent difference detected | no consistent difference detected | consistent negative | no consistent difference detected | no consistent difference detected |
| Interpolated - Sparse | calibration | -0.0092 [-0.0233, +0.0019]; -0.0089 [-0.0227, +0.0008]; -0.0460 [-0.0702, -0.0243] | no consistent difference detected | no consistent difference detected | no consistent difference detected | consistent negative | no consistent difference detected | no consistent difference detected |

### Direct argmax (secondary)

| Comparison | Half | Segment F1@50 (seeds 0, 1, 2) | Verdict | native macro-F1 verdict | native cuts per clip verdict | multi boundary recall verdict | multi boundary F1 verdict | same-class cut rate verdict |
|---|---|---|---|---|---|---|---|---|
| Dense - Sparse | selection | -0.0400 [-0.0621, -0.0183]; -0.0941 [-0.1300, -0.0614]; -0.0705 [-0.0991, -0.0435] | consistent negative | no consistent difference detected | consistent positive | no consistent difference detected | consistent negative | no consistent difference detected |
| Dense - Sparse | calibration | -0.0418 [-0.0643, -0.0215]; -0.0494 [-0.0766, -0.0242]; -0.0483 [-0.0705, -0.0279] | consistent negative | no consistent difference detected | no consistent difference detected | no consistent difference detected | consistent negative | no consistent difference detected |
| Dense - Interpolated | selection | -0.0448 [-0.0659, -0.0243]; -0.0998 [-0.1354, -0.0682]; -0.0715 [-0.0990, -0.0457] | consistent negative | no consistent difference detected | consistent positive | no consistent difference detected | consistent negative | no consistent difference detected |
| Dense - Interpolated | calibration | -0.0418 [-0.0643, -0.0215]; -0.0494 [-0.0766, -0.0242]; -0.0474 [-0.0700, -0.0272] | consistent negative | no consistent difference detected | no consistent difference detected | no consistent difference detected | consistent negative | no consistent difference detected |
| Interpolated - Sparse | selection | +0.0048 [-0.0009, +0.0122]; +0.0057 [-0.0036, +0.0169]; +0.0010 [-0.0026, +0.0057] | no consistent difference detected | no consistent difference detected | no consistent difference detected | no consistent difference detected | no consistent difference detected | no consistent difference detected |
| Interpolated - Sparse | calibration | +0.0000 [+0.0000, +0.0000]; +0.0000 [+0.0000, +0.0000]; -0.0009 [-0.0028, +0.0000] | no consistent difference detected | no consistent difference detected | no consistent difference detected | no consistent difference detected | no consistent difference detected | no consistent difference detected |

### Native clips fixed / newly wrong with B3 (sessions)

| Comparison | Half | Seed | A4C->A5C | A5C->A4C | SC4C->PLAX | all clips |
|---|---|---|---|---|---|---|
| Dense - Sparse | selection | 0 | 0 (0) / 1 (1) | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 0 (0) / 1 (1) |
| Dense - Sparse | selection | 1 | 1 (1) / 0 (0) | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 1 (1) / 0 (0) |
| Dense - Sparse | selection | 2 | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 0 (0) / 0 (0) |
| Dense - Sparse | calibration | 0 | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 0 (0) / 0 (0) |
| Dense - Sparse | calibration | 1 | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 0 (0) / 0 (0) |
| Dense - Sparse | calibration | 2 | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 0 (0) / 0 (0) |
| Dense - Interpolated | selection | 0 | 0 (0) / 1 (1) | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 0 (0) / 1 (1) |
| Dense - Interpolated | selection | 1 | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 0 (0) / 0 (0) |
| Dense - Interpolated | selection | 2 | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 0 (0) / 0 (0) |
| Dense - Interpolated | calibration | 0 | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 0 (0) / 0 (0) |
| Dense - Interpolated | calibration | 1 | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 0 (0) / 0 (0) |
| Dense - Interpolated | calibration | 2 | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 0 (0) / 0 (0) |
| Interpolated - Sparse | selection | 0 | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 0 (0) / 0 (0) |
| Interpolated - Sparse | selection | 1 | 1 (1) / 0 (0) | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 1 (1) / 0 (0) |
| Interpolated - Sparse | selection | 2 | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 0 (0) / 0 (0) |
| Interpolated - Sparse | calibration | 0 | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 0 (0) / 0 (0) |
| Interpolated - Sparse | calibration | 1 | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 0 (0) / 0 (0) |
| Interpolated - Sparse | calibration | 2 | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 0 (0) / 0 (0) | 0 (0) / 0 (0) |

### Multi-stream segments and changes with B3: a only / b only (recordings, sessions)

| Comparison | Half | Seed | Segments matched by a only | Segments matched by b only | Changes found by a only | Changes found by b only |
|---|---|---|---|---|---|---|
| Dense - Sparse | selection | 0 | 5 (5, 5) | 0 (0, 0) | 5 (10, 9) | 0 (0, 0) |
| Dense - Sparse | selection | 1 | 2 (2, 2) | 0 (0, 0) | 3 (6, 5) | 1 (2, 2) |
| Dense - Sparse | selection | 2 | 2 (2, 2) | 1 (1, 1) | 3 (6, 5) | 1 (2, 2) |
| Dense - Sparse | calibration | 0 | 1 (1, 1) | 0 (0, 0) | 2 (4, 4) | 1 (2, 2) |
| Dense - Sparse | calibration | 1 | 0 (0, 0) | 2 (2, 2) | 0 (0, 0) | 2 (4, 4) |
| Dense - Sparse | calibration | 2 | 0 (0, 0) | 1 (1, 1) | 1 (2, 2) | 1 (2, 1) |
| Dense - Interpolated | selection | 0 | 15 (14, 11) | 1 (2, 2) | 15 (19, 14) | 0 (0, 0) |
| Dense - Interpolated | selection | 1 | 11 (10, 6) | 0 (0, 0) | 10 (18, 12) | 0 (0, 0) |
| Dense - Interpolated | selection | 2 | 29 (28, 18) | 1 (2, 2) | 29 (49, 27) | 0 (0, 0) |
| Dense - Interpolated | calibration | 0 | 15 (15, 13) | 0 (0, 0) | 13 (24, 16) | 0 (0, 0) |
| Dense - Interpolated | calibration | 1 | 8 (8, 6) | 0 (0, 0) | 6 (10, 7) | 0 (0, 0) |
| Dense - Interpolated | calibration | 2 | 42 (42, 23) | 0 (0, 0) | 33 (53, 27) | 0 (0, 0) |
| Interpolated - Sparse | selection | 0 | 3 (3, 3) | 12 (13, 10) | 1 (2, 2) | 11 (17, 12) |
| Interpolated - Sparse | selection | 1 | 0 (0, 0) | 9 (9, 5) | 0 (0, 0) | 8 (15, 9) |
| Interpolated - Sparse | selection | 2 | 2 (3, 3) | 29 (28, 18) | 0 (0, 0) | 27 (46, 25) |
| Interpolated - Sparse | calibration | 0 | 0 (0, 0) | 14 (14, 12) | 0 (0, 0) | 12 (22, 16) |
| Interpolated - Sparse | calibration | 1 | 0 (0, 0) | 10 (10, 8) | 0 (0, 0) | 8 (13, 10) |
| Interpolated - Sparse | calibration | 2 | 0 (0, 0) | 43 (42, 23) | 0 (0, 0) | 33 (51, 27) |

## Matched boundary offsets (within 5 samples)

| Condition (B3) | Seed | Matched changes | Mean absolute offset (samples) | Median | 90th percentile | Mean absolute offset (s) | 90th percentile (s) | Signed mean | Exact share |
|---|---|---|---|---|---|---|---|---|---|
| Sparse | 0 | 366 / 391 | 1.5519 / 1.4552 | 1.000 / 1.000 | 3.000 / 3.000 | 0.1552 / 0.1455 | 0.300 / 0.300 | 0.1639 / 0.0691 | 0.199 / 0.220 |
| Sparse | 1 | 370 / 393 | 1.5757 / 1.4733 | 1.000 / 1.000 | 3.000 / 3.000 | 0.1576 / 0.1473 | 0.300 / 0.300 | -0.0838 / 0.0076 | 0.197 / 0.216 |
| Sparse | 2 | 372 / 390 | 1.5457 / 1.4949 | 1.000 / 1.000 | 3.000 / 3.000 | 0.1546 / 0.1495 | 0.300 / 0.300 | -0.0134 / 0.0179 | 0.204 / 0.213 |
| Interpolated | 0 | 356 / 379 | 1.4719 / 1.4301 | 1.000 / 1.000 | 3.000 / 3.000 | 0.1472 / 0.1430 | 0.300 / 0.300 | 0.1292 / 0.1161 | 0.197 / 0.206 |
| Interpolated | 1 | 362 / 385 | 1.4641 / 1.3844 | 1.000 / 1.000 | 3.000 / 3.000 | 0.1464 / 0.1384 | 0.300 / 0.300 | -0.0166 / -0.0286 | 0.215 / 0.223 |
| Interpolated | 2 | 345 / 357 | 1.4928 / 1.4202 | 1.000 / 1.000 | 3.000 / 3.000 | 0.1493 / 0.1420 | 0.300 / 0.300 | 0.0435 / 0.0644 | 0.209 / 0.213 |
| Dense | 0 | 371 / 392 | 1.0970 / 1.1352 | 1.000 / 1.000 | 2.000 / 2.000 | 0.1097 / 0.1135 | 0.200 / 0.200 | 0.1321 / 0.1352 | 0.299 / 0.240 |
| Dense | 1 | 372 / 391 | 1.0430 / 1.0179 | 1.000 / 1.000 | 2.000 / 2.000 | 0.1043 / 0.1018 | 0.200 / 0.200 | 0.0000 / 0.1483 | 0.296 / 0.274 |
| Dense | 2 | 374 / 390 | 1.1176 / 1.1000 | 1.000 / 1.000 | 2.000 / 2.000 | 0.1118 / 0.1100 | 0.200 / 0.200 | 0.0374 / 0.0897 | 0.262 / 0.264 |

## Short segments and X-Y-X middles (B3)

| Condition (B3) | Seed | short segments kept (141 / 134) | short A4C or A5C segments kept (65 / 67) | X-Y-X middles kept (69 / 80) | A4C/A5C X-Y-X middles kept (7 / 7) |
|---|---|---|---|---|---|
| Sparse | 0 | 130 / 128 | 54 / 66 | 66 / 80 | 5 / 7 |
| Sparse | 1 | 134 / 129 | 58 / 65 | 64 / 80 | 3 / 7 |
| Sparse | 2 | 136 / 127 | 60 / 65 | 66 / 80 | 5 / 7 |
| Interpolated | 0 | 123 / 120 | 48 / 59 | 62 / 78 | 1 / 6 |
| Interpolated | 1 | 127 / 123 | 53 / 61 | 64 / 77 | 3 / 6 |
| Interpolated | 2 | 119 / 109 | 49 / 53 | 66 / 73 | 5 / 6 |
| Dense | 0 | 132 / 129 | 56 / 66 | 66 / 80 | 5 / 7 |
| Dense | 1 | 134 / 127 | 58 / 63 | 64 / 79 | 3 / 7 |
| Dense | 2 | 136 / 126 | 60 / 64 | 66 / 80 | 5 / 7 |

## Reference units of multi streams (B3; hit 5, near 10)

| Condition (B3) | Seed | Wrong units | label (rec, ses) | missed_cut (rec, ses) | extra_cut (rec, ses) | boundary_shift (rec, ses) | Unmatched predicted segments (inside a unit) | Matched units with an edge only within near |
|---|---|---|---|---|---|---|---|---|
| Sparse | 0 | 29 of 502 / 8 of 518 | 11 (5, 4) / 3 (5, 5) | 15 (14, 11) / 5 (4, 4) | 2 (4, 4) / 0 (0, 0) | 1 (1, 1) / 0 (0, 0) | 33 (15) / 11 (7) | 7 / 2 |
| Sparse | 1 | 18 of 502 / 7 of 518 | 3 (2, 2) / 4 (6, 6) | 14 (15, 12) / 3 (3, 3) | 1 (2, 2) / 0 (0, 0) | 0 (0, 0) / 0 (0, 0) | 22 (8) / 10 (6) | 4 / 2 |
| Sparse | 2 | 15 of 502 / 10 of 518 | 4 (3, 3) / 4 (6, 6) | 11 (12, 10) / 6 (5, 5) | 0 (0, 0) / 0 (0, 0) | 0 (0, 0) / 0 (0, 0) | 21 (8) / 11 (6) | 4 / 2 |
| Interpolated | 0 | 38 of 502 / 22 of 518 | 12 (6, 6) / 3 (5, 5) | 26 (22, 13) / 19 (18, 14) | 0 (0, 0) / 0 (0, 0) | 0 (0, 0) / 0 (0, 0) | 19 (7) / 6 (3) | 3 / 0 |
| Interpolated | 1 | 27 of 502 / 17 of 518 | 3 (2, 2) / 5 (6, 6) | 21 (19, 13) / 12 (11, 8) | 1 (2, 2) / 0 (0, 0) | 2 (2, 2) / 0 (0, 0) | 17 (3) / 9 (3) | 5 / 2 |
| Interpolated | 2 | 42 of 502 / 53 of 518 | 6 (4, 4) / 8 (11, 9) | 36 (31, 20) / 45 (42, 22) | 0 (0, 0) / 0 (0, 0) | 0 (0, 0) / 0 (0, 0) | 9 (1) / 13 (2) | 4 / 0 |
| Dense | 0 | 24 of 502 / 7 of 518 | 11 (8, 7) / 3 (5, 5) | 13 (13, 11) / 3 (2, 2) | 0 (0, 0) / 0 (0, 0) | 0 (0, 0) / 1 (1, 1) | 34 (19) / 12 (10) | 2 / 3 |
| Dense | 1 | 16 of 502 / 9 of 518 | 4 (4, 4) / 5 (6, 6) | 12 (13, 11) / 4 (4, 4) | 0 (0, 0) / 0 (0, 0) | 0 (0, 0) / 0 (0, 0) | 20 (8) / 13 (8) | 2 / 3 |
| Dense | 2 | 14 of 502 / 11 of 518 | 4 (3, 3) / 4 (6, 6) | 9 (8, 8) / 6 (5, 5) | 1 (2, 2) / 0 (0, 0) | 0 (0, 0) / 1 (1, 1) | 17 (10) / 13 (8) | 0 / 2 |

Readings of wrong units, not exclusive: units (recordings, sessions).

| Condition (B3) | Seed | raw classification | sampling position | decoding loss | interval mismatch |
|---|---|---|---|---|---|
| Sparse | 0 | 12 (7, 6) / 7 (8, 8) | 1 (1, 1) / 0 (0, 0) | 16 (16, 12) / 3 (5, 4) | 1 (1, 1) / 0 (0, 0) |
| Sparse | 1 | 5 (4, 3) / 6 (7, 6) | 2 (2, 2) / 0 (0, 0) | 11 (13, 11) / 1 (3, 3) | 0 (0, 0) / 0 (0, 0) |
| Sparse | 2 | 6 (5, 5) / 9 (10, 9) | 0 (0, 0) / 0 (0, 0) | 9 (10, 9) / 2 (2, 2) | 0 (0, 0) / 0 (0, 0) |
| Interpolated | 0 | 12 (7, 6) / 7 (8, 8) | 1 (1, 1) / 0 (0, 0) | 26 (23, 15) / 15 (15, 13) | 1 (1, 1) / 0 (0, 0) |
| Interpolated | 1 | 6 (5, 4) / 8 (7, 6) | 1 (1, 1) / 0 (0, 0) | 20 (19, 13) / 10 (12, 10) | 2 (2, 2) / 0 (0, 0) |
| Interpolated | 2 | 7 (6, 6) / 9 (10, 9) | 0 (0, 0) / 0 (0, 0) | 36 (30, 20) / 45 (44, 24) | 0 (0, 0) / 0 (0, 0) |
| Dense | 0 | 11 (7, 6) / 7 (8, 8) | 0 (0, 0) / 0 (0, 0) | 13 (14, 11) / 0 (0, 0) | 0 (0, 0) / 2 (4, 4) |
| Dense | 1 | 5 (4, 3) / 7 (7, 6) | 0 (0, 0) / 0 (0, 0) | 11 (14, 12) / 2 (4, 4) | 0 (0, 0) / 1 (3, 3) |
| Dense | 2 | 7 (6, 6) / 9 (10, 9) | 0 (0, 0) / 0 (0, 0) | 7 (8, 8) / 2 (2, 2) | 0 (0, 0) / 1 (1, 1) |

Flags of wrong units, not exclusive: units.

| Condition (B3) | Seed | all edges within hit | an edge only within near | missed edge | extra cuts | own majority right | own mean right | dense majority right |
|---|---|---|---|---|---|---|---|---|
| Sparse | 0 | 11 / 3 | 1 / 0 | 17 / 5 | 6 / 2 | 16 / 3 | 17 / 0 | 17 / 1 |
| Sparse | 1 | 4 / 4 | 0 / 0 | 14 / 3 | 4 / 2 | 11 / 1 | 12 / 0 | 13 / 1 |
| Sparse | 2 | 4 / 3 | 0 / 0 | 11 / 7 | 3 / 1 | 9 / 2 | 9 / 1 | 9 / 1 |
| Interpolated | 0 | 9 / 2 | 1 / 0 | 29 / 20 | 2 / 0 | 26 / 15 | 26 / 14 | 26 / 15 |
| Interpolated | 1 | 5 / 4 | 1 / 0 | 21 / 13 | 4 / 0 | 20 / 10 | 21 / 8 | 21 / 9 |
| Interpolated | 2 | 4 / 3 | 0 / 0 | 38 / 50 | 0 / 0 | 36 / 45 | 36 / 44 | 35 / 44 |
| Dense | 0 | 9 / 3 | 0 / 1 | 15 / 3 | 7 / 2 | 13 / 0 | 13 / 0 | 13 / 0 |
| Dense | 1 | 3 / 4 | 0 / 1 | 13 / 4 | 4 / 3 | 11 / 2 | 11 / 1 | 11 / 2 |
| Dense | 2 | 5 / 4 | 0 / 0 | 9 / 7 | 3 / 2 | 7 / 2 | 7 / 2 | 7 / 2 |

## Reference units of pair streams (B3)

| Condition (B3) | Seed | Wrong units | label (rec, ses) | missed_cut (rec, ses) | extra_cut (rec, ses) | boundary_shift (rec, ses) | Unmatched predicted segments (inside a unit) | Matched units with an edge only within near |
|---|---|---|---|---|---|---|---|---|
| Sparse | 0 | 7 of 207 / 5 of 173 | 2 (3, 3) / 1 (1, 1) | 3 (3, 3) / 4 (4, 4) | 2 (2, 2) / 0 (0, 0) | 0 (0, 0) / 0 (0, 0) | 9 (8) / 1 (0) | 0 / 2 |
| Sparse | 1 | 7 of 207 / 6 of 173 | 2 (3, 3) / 3 (4, 4) | 3 (3, 3) / 3 (3, 3) | 2 (2, 2) / 0 (0, 0) | 0 (0, 0) / 0 (0, 0) | 10 (9) / 4 (3) | 0 / 2 |
| Sparse | 2 | 7 of 207 / 7 of 173 | 2 (3, 3) / 3 (4, 4) | 3 (3, 3) / 4 (4, 4) | 2 (2, 2) / 0 (0, 0) | 0 (0, 0) / 0 (0, 0) | 11 (10) / 3 (2) | 0 / 2 |
| Interpolated | 0 | 8 of 207 / 5 of 173 | 3 (4, 4) / 1 (1, 1) | 4 (4, 4) / 4 (4, 4) | 1 (1, 1) / 0 (0, 0) | 0 (0, 0) / 0 (0, 0) | 7 (5) / 1 (0) | 0 / 2 |
| Interpolated | 1 | 8 of 207 / 8 of 173 | 4 (5, 5) / 3 (4, 4) | 3 (3, 3) / 5 (5, 5) | 1 (1, 1) / 0 (0, 0) | 0 (0, 0) / 0 (0, 0) | 8 (7) / 4 (3) | 0 / 2 |
| Interpolated | 2 | 11 of 207 / 10 of 173 | 2 (3, 3) / 3 (4, 4) | 8 (8, 7) / 7 (7, 6) | 1 (1, 1) / 0 (0, 0) | 0 (0, 0) / 0 (0, 0) | 6 (6) / 3 (2) | 0 / 0 |
| Dense | 0 | 7 of 207 / 4 of 173 | 4 (5, 5) / 1 (1, 1) | 3 (3, 3) / 3 (3, 3) | 0 (0, 0) / 0 (0, 0) | 0 (0, 0) / 0 (0, 0) | 9 (8) / 1 (0) | 0 / 2 |
| Dense | 1 | 8 of 207 / 7 of 173 | 5 (6, 6) / 3 (4, 4) | 3 (3, 3) / 4 (4, 4) | 0 (0, 0) / 0 (0, 0) | 0 (0, 0) / 0 (0, 0) | 10 (8) / 4 (2) | 0 / 2 |
| Dense | 2 | 6 of 207 / 7 of 173 | 4 (5, 5) / 3 (4, 4) | 2 (2, 2) / 4 (4, 4) | 0 (0, 0) / 0 (0, 0) | 0 (0, 0) / 0 (0, 0) | 10 (10) / 3 (1) | 0 / 2 |

Readings of wrong units, not exclusive: units (recordings, sessions).

| Condition (B3) | Seed | raw classification | sampling position | decoding loss | interval mismatch |
|---|---|---|---|---|---|
| Sparse | 0 | 5 (6, 5) / 4 (4, 4) | 0 (0, 0) / 0 (0, 0) | 4 (4, 3) / 1 (1, 1) | 0 (0, 0) / 0 (0, 0) |
| Sparse | 1 | 5 (6, 5) / 5 (5, 5) | 0 (0, 0) / 1 (2, 2) | 4 (4, 3) / 0 (0, 0) | 0 (0, 0) / 0 (0, 0) |
| Sparse | 2 | 5 (6, 5) / 5 (5, 5) | 0 (0, 0) / 0 (0, 0) | 4 (4, 3) / 2 (3, 3) | 0 (0, 0) / 0 (0, 0) |
| Interpolated | 0 | 5 (6, 5) / 4 (4, 4) | 0 (0, 0) / 0 (0, 0) | 3 (3, 3) / 1 (1, 1) | 0 (0, 0) / 0 (0, 0) |
| Interpolated | 1 | 6 (7, 6) / 5 (5, 5) | 0 (0, 0) / 1 (2, 2) | 3 (3, 3) / 2 (2, 2) | 0 (0, 0) / 0 (0, 0) |
| Interpolated | 2 | 5 (6, 5) / 5 (5, 5) | 0 (0, 0) / 1 (2, 2) | 7 (7, 7) / 4 (4, 3) | 0 (0, 0) / 0 (0, 0) |
| Dense | 0 | 5 (6, 5) / 4 (4, 4) | 0 (0, 0) / 0 (0, 0) | 2 (2, 2) / 0 (0, 0) | 0 (0, 0) / 0 (0, 0) |
| Dense | 1 | 6 (7, 6) / 5 (5, 5) | 0 (0, 0) / 0 (0, 0) | 2 (2, 2) / 2 (3, 2) | 0 (0, 0) / 0 (0, 0) |
| Dense | 2 | 5 (6, 5) / 5 (5, 5) | 0 (0, 0) / 0 (0, 0) | 1 (1, 1) / 2 (3, 3) | 0 (0, 0) / 0 (0, 0) |

Flags of wrong units, not exclusive: units.

| Condition (B3) | Seed | all edges within hit | an edge only within near | missed edge | extra cuts | own majority right | own mean right | dense majority right |
|---|---|---|---|---|---|---|---|---|
| Sparse | 0 | 4 / 1 | 0 / 0 | 3 / 4 | 3 / 0 | 4 / 1 | 2 / 1 | 2 / 1 |
| Sparse | 1 | 4 / 3 | 0 / 0 | 3 / 3 | 3 / 0 | 4 / 0 | 3 / 0 | 2 / 1 |
| Sparse | 2 | 4 / 3 | 0 / 0 | 3 / 4 | 3 / 0 | 4 / 2 | 2 / 1 | 2 / 2 |
| Interpolated | 0 | 3 / 1 | 0 / 0 | 5 / 4 | 2 / 0 | 3 / 1 | 3 / 1 | 3 / 1 |
| Interpolated | 1 | 5 / 3 | 0 / 0 | 3 / 5 | 3 / 0 | 3 / 2 | 3 / 2 | 2 / 3 |
| Interpolated | 2 | 3 / 3 | 0 / 0 | 8 / 7 | 2 / 0 | 7 / 4 | 7 / 4 | 6 / 5 |
| Dense | 0 | 4 / 1 | 0 / 0 | 3 / 3 | 3 / 0 | 2 / 0 | 2 / 0 | 2 / 0 |
| Dense | 1 | 5 / 3 | 0 / 0 | 3 / 4 | 3 / 0 | 2 / 2 | 2 / 1 | 2 / 2 |
| Dense | 2 | 4 / 3 | 0 / 0 | 2 / 4 | 3 / 0 | 1 / 2 | 1 / 1 | 1 / 2 |

## Native clips

| Condition | Seed | B3-wrong clips (rec, ses) | A4C->A5C | A5C->A4C | Raw right, decode wrong | Both wrong | Raw wrong, decode right | Mean label right among wrong | Stage: decoding | Stage: sampling | Stage: recognition |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Sparse | 0 | 3 (3, 3) / 1 (1, 1) | 2 / 1 | 0 / 0 | 0 / 0 | 3 / 1 | 1 / 1 | 1 / 0 | 0 / 0 | 0 / 0 | 3 / 1 |
| Sparse | 1 | 3 (3, 3) / 2 (2, 2) | 2 / 2 | 0 / 0 | 1 / 0 | 2 / 2 | 0 / 1 | 2 / 0 | 1 / 0 | 0 / 0 | 2 / 2 |
| Sparse | 2 | 4 (4, 4) / 2 (2, 2) | 3 / 2 | 0 / 0 | 0 / 0 | 4 / 2 | 0 / 0 | 1 / 0 | 0 / 0 | 1 / 0 | 3 / 2 |
| Interpolated | 0 | 3 (3, 3) / 1 (1, 1) | 2 / 1 | 0 / 0 | 0 / 0 | 3 / 1 | 1 / 1 | 1 / 0 | 0 / 0 | 0 / 0 | 3 / 1 |
| Interpolated | 1 | 2 (2, 2) / 2 (2, 2) | 1 / 2 | 0 / 0 | 1 / 0 | 1 / 2 | 1 / 1 | 1 / 0 | 1 / 0 | 0 / 0 | 1 / 2 |
| Interpolated | 2 | 4 (4, 4) / 2 (2, 2) | 3 / 2 | 0 / 0 | 0 / 0 | 4 / 2 | 0 / 0 | 1 / 0 | 0 / 0 | 1 / 0 | 3 / 2 |
| Dense | 0 | 4 (4, 3) / 1 (1, 1) | 3 / 1 | 0 / 0 | 0 / 0 | 4 / 1 | 0 / 1 | 1 / 0 | 0 / 0 | 0 / 0 | 4 / 1 |
| Dense | 1 | 2 (2, 2) / 2 (2, 2) | 1 / 2 | 0 / 0 | 1 / 0 | 1 / 2 | 1 / 0 | 1 / 0 | 1 / 0 | 0 / 0 | 1 / 2 |
| Dense | 2 | 4 (4, 4) / 2 (2, 2) | 3 / 2 | 0 / 0 | 1 / 0 | 3 / 2 | 0 / 0 | 1 / 0 | 1 / 0 | 0 / 0 | 3 / 2 |

Union over the three seeds: distinct recordings (sessions).

| Condition | Half | B3-wrong recordings | By class pair | Stage: decoding | Stage: sampling | Stage: recognition | Localisation |
|---|---|---|---|---|---|---|---|
| Sparse | selection | 4 (4) | A4C->A5C 3 (3); PLAX->PSAX 1 (1) | 1 | 1 | 3 | recognition |
| Sparse | calibration | 2 (2) | A4C->A5C 2 (2) | 0 | 0 | 2 | recognition |
| Interpolated | selection | 4 (4) | A4C->A5C 3 (3); PLAX->PSAX 1 (1) | 1 | 1 | 3 | recognition |
| Interpolated | calibration | 2 (2) | A4C->A5C 2 (2) | 0 | 0 | 2 | recognition |
| Dense | selection | 5 (4) | A4C->A5C 4 (4); PLAX->PSAX 1 (1) | 2 | 0 | 4 | recognition |
| Dense | calibration | 2 (2) | A4C->A5C 2 (2) | 0 | 0 | 2 | recognition |

## The 10 Hz axis

| a - b | Seed | Sequences | Samples | a right, b wrong (rec, ses) | b right, a wrong (rec, ses) | Both wrong (rec, ses) |
|---|---|---|---|---|---|---|
| Dense - Sparse | 0 | native | 16635 / 14323 | 22 (9, 7) / 12 (4, 3) | 15 (5, 5) / 18 (7, 7) | 296 (12, 8) / 182 (9, 8) |
| Dense - Sparse | 0 | pairs | 5691 / 4883 | 62 (45, 22) / 50 (35, 20) | 41 (28, 18) / 28 (22, 15) | 173 (43, 19) / 137 (36, 23) |
| Dense - Sparse | 0 | multi | 14961 / 15010 | 368 (113, 34) / 331 (124, 32) | 163 (68, 32) / 218 (97, 31) | 742 (96, 35) / 458 (115, 29) |
| Dense - Sparse | 1 | native | 16635 / 14323 | 26 (10, 7) / 26 (6, 4) | 27 (11, 7) / 22 (7, 7) | 298 (13, 8) / 173 (9, 7) |
| Dense - Sparse | 1 | pairs | 5691 / 4883 | 69 (45, 22) / 49 (31, 18) | 45 (31, 17) / 27 (18, 13) | 175 (47, 22) / 150 (37, 24) |
| Dense - Sparse | 1 | multi | 14961 / 15010 | 393 (122, 34) / 354 (130, 33) | 182 (78, 32) / 168 (94, 30) | 544 (98, 33) / 469 (113, 31) |
| Dense - Sparse | 2 | native | 16635 / 14323 | 25 (8, 7) / 22 (8, 7) | 23 (7, 5) / 28 (9, 8) | 343 (15, 10) / 225 (11, 10) |
| Dense - Sparse | 2 | pairs | 5691 / 4883 | 68 (47, 23) / 55 (37, 21) | 36 (28, 15) / 24 (21, 16) | 183 (37, 17) / 135 (32, 22) |
| Dense - Sparse | 2 | multi | 14961 / 15010 | 349 (109, 34) / 343 (127, 32) | 175 (71, 32) / 182 (94, 30) | 592 (100, 32) / 510 (118, 32) |
| Dense - Interpolated | 0 | native | 16635 / 14323 | 11 (7, 6) / 9 (3, 3) | 14 (7, 6) / 18 (4, 4) | 297 (12, 8) / 182 (9, 8) |
| Dense - Interpolated | 0 | pairs | 5691 / 4883 | 45 (36, 17) / 32 (27, 19) | 33 (24, 15) / 29 (22, 15) | 181 (46, 20) / 136 (37, 23) |
| Dense - Interpolated | 0 | multi | 14961 / 15010 | 285 (103, 34) / 249 (114, 31) | 137 (67, 31) / 177 (89, 31) | 768 (101, 36) / 499 (119, 30) |
| Dense - Interpolated | 1 | native | 16635 / 14323 | 14 (8, 5) / 23 (6, 5) | 20 (10, 7) / 18 (7, 6) | 305 (13, 8) / 177 (9, 7) |
| Dense - Interpolated | 1 | pairs | 5691 / 4883 | 62 (43, 22) / 36 (26, 16) | 35 (28, 14) / 28 (19, 13) | 185 (49, 23) / 149 (39, 24) |
| Dense - Interpolated | 1 | multi | 14961 / 15010 | 276 (112, 34) / 268 (116, 31) | 172 (74, 32) / 149 (88, 29) | 554 (100, 34) / 488 (115, 31) |
| Dense - Interpolated | 2 | native | 16635 / 14323 | 25 (10, 8) / 17 (7, 6) | 22 (7, 6) / 19 (9, 8) | 344 (15, 10) / 234 (11, 10) |
| Dense - Interpolated | 2 | pairs | 5691 / 4883 | 52 (40, 21) / 41 (33, 21) | 29 (25, 15) / 23 (19, 15) | 190 (41, 19) / 136 (35, 22) |
| Dense - Interpolated | 2 | multi | 14961 / 15010 | 257 (103, 33) / 249 (113, 32) | 159 (72, 31) / 151 (85, 29) | 608 (104, 33) / 541 (121, 32) |
| Interpolated - Sparse | 0 | native | 16635 / 14323 | 17 (8, 6) / 12 (4, 3) | 7 (4, 4) / 9 (4, 4) | 301 (12, 8) / 182 (9, 8) |
| Interpolated - Sparse | 0 | pairs | 5691 / 4883 | 19 (14, 11) / 20 (16, 12) | 10 (8, 7) / 1 (1, 1) | 216 (60, 24) / 167 (48, 27) |
| Interpolated - Sparse | 0 | multi | 14961 / 15010 | 115 (53, 26) / 92 (50, 25) | 58 (21, 14) / 51 (35, 18) | 995 (130, 36) / 697 (141, 31) |
| Interpolated - Sparse | 1 | native | 16635 / 14323 | 19 (9, 7) / 10 (5, 3) | 14 (5, 3) / 11 (5, 5) | 305 (13, 8) / 189 (9, 7) |
| Interpolated - Sparse | 1 | pairs | 5691 / 4883 | 13 (10, 9) / 18 (15, 11) | 16 (9, 8) / 4 (4, 4) | 231 (64, 27) / 181 (50, 26) |
| Interpolated - Sparse | 1 | multi | 14961 / 15010 | 136 (59, 26) / 92 (53, 28) | 29 (18, 14) / 25 (20, 13) | 801 (136, 35) / 731 (142, 32) |
| Interpolated - Sparse | 2 | native | 16635 / 14323 | 11 (6, 6) / 8 (5, 5) | 12 (5, 4) / 12 (4, 4) | 357 (15, 10) / 239 (11, 10) |
| Interpolated - Sparse | 2 | pairs | 5691 / 4883 | 16 (13, 10) / 19 (16, 12) | 7 (5, 5) / 6 (5, 4) | 235 (61, 25) / 171 (49, 27) |
| Interpolated - Sparse | 2 | multi | 14961 / 15010 | 115 (53, 26) / 105 (55, 24) | 39 (19, 12) / 42 (27, 16) | 826 (128, 35) / 748 (146, 33) |

## Goal

| Condition | Three-seed mean (sel / cal) | Margin to the minimum | Calibration sensitivity margin | Fixed numbers feasible (seeds) | Constraint failures | Goal met |
|---|---|---|---|---|---|---|
| Sparse | 0.9544 / 0.9817 | +0.0100 / -0.0004 | -0.0022 | True True True | 1 | False |
| Interpolated | 0.9484 / 0.9603 | +0.0040 / -0.0218 | -0.0236 | True True False | 4 | False |
| Dense | 0.9588 / 0.9792 | +0.0144 / -0.0030 | -0.0047 | True True True | 2 | False |

## Cases (figures on the server)

```
{
 "native floor": {
  "cases": 16,
  "recordings": 5,
  "sessions": 4
 },
 "calibration": {
  "native clip cases": 10,
  "unit cases": 52,
  "recordings": 13,
  "sessions": 11
 },
 "note": "figures stay on the server; each case is one condition, seed and source"
}
```

## Decision

Applied: offset_only, interpolation_close, native_localisation, end

```
{
 "dense_system_gain": {
  "applies": false,
  "reading": "Dense - Sparse B3 segment F1@50 consistent positive in both halves and Dense meeting every constraint -> 'an implementation gain of denser target requests with the same trained model'; never a new architecture or an order mechanism"
 },
 "goal_met": {
  "applies": false,
  "reading": "only together with dense_system_gain and Dense's goal met -> 'the old development protocol goal is met'; the test split stays sealed"
 },
 "gain_without_constraints": {
  "applies": false,
  "reading": "Dense - Sparse B3 segment F1@50 consistent positive in both halves while Dense misses a constraint -> 'a segmentation gain that does not meet the constraints'; not a system gain"
 },
 "offset_only": {
  "applies": true,
  "reading": "Dense's mean absolute matched boundary offset (B3) below Sparse's in every seed and half, while no half has Dense - Sparse B3 segment F1@50 or native macro-F1 consistent positive -> 'finer localisation only; not evidence that the remaining loss comes from the grid'; this configuration ends without any stride, interpolation or decoding search"
 },
 "interpolation_close": {
  "halves": [
   "selection",
   "calibration"
  ],
  "applies": true,
  "reading": "in a half where Dense - Interpolated B3 segment F1@50 is not consistent positive -> neither equivalence nor a gain wholly from new observations is claimed; Dense and Interpolated are reported by cost and counts and the primary condition is not changed"
 },
 "dense_costs": {
  "halves": [],
  "applies": false,
  "reading": "Dense - Sparse with B3 consistent positive in native label switches per clip or same-class cut rate, or consistent negative in native macro-F1, in a half -> real gains and costs are both reported; Sparse stays the fixed reference; no threshold is retuned"
 },
 "native_localisation": {
  "by_condition_and_half": {
   "Sparse|selection": "recognition",
   "Sparse|calibration": "recognition",
   "Interpolated|selection": "recognition",
   "Interpolated|calibration": "recognition",
   "Dense|selection": "recognition",
   "Dense|calibration": "recognition"
  },
  "applies": true,
  "reading": "per half and condition, over the union of seeds by distinct recordings, each B3-wrong native clip falls in one stage by its own raw majority: decoding (the condition's argmax majority is right), sampling (it is wrong while the Dense argmax majority is right) or recognition (both are wrong); a recording wrong in several seeds counts in every stage it meets. More than half of the wrong recordings in decoding -> evidence for decoding research only; in recognition -> evidence for recognition research only; in sampling -> a sampling effect; otherwise mixed. Localisation only, never a trigger for training or a module; the report states which part sampling changes and which part stays unexplained"
 },
 "end": {
  "applies": true,
  "reading": "the round ends at the report: no training, dense temporal head, ASRF or FACT retraining, distillation, expert pseudo-labels or external data; the paper stays paused; the patch-readout and FACT-refinement configurations stay closed"
 }
}
```
