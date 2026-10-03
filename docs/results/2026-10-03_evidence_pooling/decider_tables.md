# A4C/A5C decision on E1 ASRF candidate spans (round 5, step 2)

Targets: selection 0.9444, calibration 0.9821; native floors: selection 0.9730, calibration 0.9800.

## Training (training split only)

- seed 0: 6347 spans (2914 A5C) from 1075 clips; weighted training accuracy trained_mean 0.966, trained_distribution 0.967
- seed 1: 6345 spans (2912 A5C) from 1075 clips; weighted training accuracy trained_mean 0.966, trained_distribution 0.968
- seed 2: 6348 spans (2914 A5C) from 1075 clips; weighted training accuracy trained_mean 0.965, trained_distribution 0.968

## All clips (main result)

| rule | half | segment F1@50 s0 / s1 / s2 (mean) | recall s0 / s1 / s2 | native macro-F1 s0 / s1 / s2 |
|---|---|---|---|---|
| default | selection | 0.910 / 0.909 / 0.907 (0.909) | 0.942 / 0.950 / 0.953 | 0.948 / 0.966 / 0.966 |
| default | calibration | 0.976 / 0.964 / 0.964 (0.968) | 0.977 / 0.972 / 0.977 | 0.993 / 0.975 / 0.975 |
| mean | selection | 0.856 / 0.883 / 0.879 (0.873) | 0.971 / 0.958 / 0.961 | 0.959 / 0.968 / 0.968 |
| mean | calibration | 0.931 / 0.964 / 0.960 (0.952) | 0.992 / 0.990 / 0.990 | 0.986 / 0.983 / 0.983 |
| trained_mean | selection | 0.863 / 0.883 / 0.880 (0.875) | 0.974 / 0.958 / 0.963 | 0.951 / 0.968 / 0.968 |
| trained_mean | calibration | 0.923 / 0.957 / 0.956 (0.946) | 0.992 / 0.990 / 0.990 | 0.986 / 0.983 / 0.983 |
| trained_distribution | selection | 0.858 / 0.885 / 0.884 (0.876) | 0.974 / 0.958 / 0.963 | 0.951 / 0.968 / 0.968 |
| trained_distribution | calibration | 0.922 / 0.960 / 0.956 (0.946) | 0.992 / 0.987 / 0.987 | 0.986 / 0.983 / 0.983 |

## Without the ten sensitivity clips (dependence only)

| rule | half | segment F1@50 s0 / s1 / s2 (mean) | recall s0 / s1 / s2 | native macro-F1 s0 / s1 / s2 |
|---|---|---|---|---|
| default | selection | 0.939 / 0.968 / 0.967 (0.958) | 0.945 / 0.975 / 0.966 | 0.950 / 0.974 / 0.974 |
| default | calibration | 0.988 / 0.985 / 0.987 (0.986) | 0.984 / 0.977 / 0.984 | 0.990 / 0.988 / 0.988 |
| mean | selection | 0.929 / 0.944 / 0.947 (0.940) | 0.983 / 0.971 / 0.979 | 0.964 / 0.976 / 0.976 |
| mean | calibration | 0.971 / 0.981 / 0.976 (0.976) | 0.992 / 0.988 / 0.988 | 0.990 / 0.988 / 0.988 |
| trained_mean | selection | 0.925 / 0.942 / 0.942 (0.936) | 0.983 / 0.971 / 0.979 | 0.954 / 0.976 / 0.976 |
| trained_mean | calibration | 0.956 / 0.975 / 0.976 (0.969) | 0.992 / 0.988 / 0.988 | 0.990 / 0.988 / 0.988 |
| trained_distribution | selection | 0.931 / 0.942 / 0.948 (0.940) | 0.983 / 0.971 / 0.979 | 0.954 / 0.976 / 0.976 |
| trained_distribution | calibration | 0.955 / 0.978 / 0.976 (0.970) | 0.992 / 0.988 / 0.988 | 0.990 / 0.988 / 0.988 |

## Success test (all clips)

| rule | half | mean segment F1@50 (target) | min recall | min native F1 (floor) | passes |
|---|---|---|---|---|---|
| default | selection | 0.909 (0.944) | 0.942 | 0.948 (0.973) | no |
| default | calibration | 0.968 (0.982) | 0.972 | 0.975 (0.980) | no |
| mean | selection | 0.873 (0.944) | 0.958 | 0.959 (0.973) | no |
| mean | calibration | 0.952 (0.982) | 0.990 | 0.983 (0.980) | no |
| trained_mean | selection | 0.875 (0.944) | 0.958 | 0.951 (0.973) | no |
| trained_mean | calibration | 0.946 (0.982) | 0.990 | 0.983 (0.980) | no |
| trained_distribution | selection | 0.876 (0.944) | 0.958 | 0.951 (0.973) | no |
| trained_distribution | calibration | 0.946 (0.982) | 0.987 | 0.983 (0.980) | no |

## Paired contrasts of segment F1@50 (points, 95% bootstrap intervals)

- trained_distribution - trained_mean, all, selection: s0 -0.5 [-2.0, +0.8]; s1 +0.3 [-0.5, +1.1]; s2 +0.4 [-0.2, +1.0] -> no consistent difference detected
- trained_distribution - trained_mean, all, calibration: s0 -0.2 [-0.7, +0.4]; s1 +0.3 [-0.2, +0.8]; s2 +0.0 [+0.0, +0.0] -> no consistent difference detected
- trained_distribution - mean, all, selection: s0 +0.1 [-0.3, +0.7]; s1 +0.2 [-0.5, +1.0]; s2 +0.5 [-0.2, +1.3] -> no consistent difference detected
- trained_distribution - mean, all, calibration: s0 -1.0 [-2.3, +0.1]; s1 -0.4 [-0.9, +0.0]; s2 -0.4 [-1.2, +0.0] -> no consistent difference detected
- trained_mean - mean, all, selection: s0 +0.7 [-0.6, +2.1]; s1 -0.0 [-0.6, +0.5]; s2 +0.1 [-0.6, +0.9] -> no consistent difference detected
- trained_mean - mean, all, calibration: s0 -0.8 [-2.2, +0.3]; s1 -0.6 [-1.4, +0.0]; s2 -0.4 [-1.2, +0.0] -> no consistent difference detected
- mean - default, all, selection: s0 -5.4 [-8.1, -2.7]; s1 -2.6 [-4.4, -0.9]; s2 -2.8 [-4.4, -1.3] -> consistent (lower)
- mean - default, all, calibration: s0 -4.5 [-6.5, -2.3]; s1 -0.1 [-1.3, +1.2]; s2 -0.4 [-1.6, +0.8] -> no consistent difference detected
- trained_mean - default, all, selection: s0 -4.7 [-7.3, -2.1]; s1 -2.6 [-4.3, -1.0]; s2 -2.7 [-4.2, -1.2] -> consistent (lower)
- trained_mean - default, all, calibration: s0 -5.3 [-7.5, -3.0]; s1 -0.7 [-2.1, +0.8]; s2 -0.8 [-2.2, +0.6] -> no consistent difference detected
- trained_distribution - default, all, selection: s0 -5.2 [-7.9, -2.5]; s1 -2.4 [-4.1, -0.8]; s2 -2.3 [-3.8, -0.9] -> consistent (lower)
- trained_distribution - default, all, calibration: s0 -5.4 [-7.7, -3.1]; s1 -0.4 [-1.7, +1.0]; s2 -0.8 [-2.2, +0.6] -> no consistent difference detected
- trained_distribution - trained_mean, without, selection: s0 +0.6 [+0.0, +1.5]; s1 -0.0 [-0.9, +0.9]; s2 +0.6 [+0.0, +1.5] -> no consistent difference detected
- trained_distribution - trained_mean, without, calibration: s0 -0.1 [-0.5, +0.0]; s1 +0.3 [+0.0, +0.9]; s2 +0.0 [+0.0, +0.0] -> no consistent difference detected
- trained_distribution - mean, without, selection: s0 +0.1 [-0.5, +0.8]; s1 -0.2 [-1.1, +0.7]; s2 +0.1 [-0.5, +0.9] -> no consistent difference detected
- trained_distribution - mean, without, calibration: s0 -1.6 [-3.4, -0.1]; s1 -0.3 [-0.9, +0.0]; s2 +0.0 [+0.0, +0.0] -> no consistent difference detected
- trained_mean - mean, without, selection: s0 -0.5 [-1.3, +0.0]; s1 -0.2 [-0.5, +0.0]; s2 -0.5 [-1.2, +0.0] -> no consistent difference detected
- trained_mean - mean, without, calibration: s0 -1.4 [-3.2, +0.0]; s1 -0.6 [-1.5, +0.0]; s2 +0.0 [+0.0, +0.0] -> no consistent difference detected
- mean - default, without, selection: s0 -1.0 [-3.3, +1.3]; s1 -2.4 [-4.7, -0.6]; s2 -2.0 [-3.7, -0.3] -> no consistent difference detected
- mean - default, without, calibration: s0 -1.7 [-3.1, -0.4]; s1 -0.4 [-1.7, +0.8]; s2 -1.0 [-2.4, +0.0] -> no consistent difference detected
- trained_mean - default, without, selection: s0 -1.4 [-3.6, +0.7]; s1 -2.6 [-4.9, -0.8]; s2 -2.4 [-4.2, -0.6] -> no consistent difference detected
- trained_mean - default, without, calibration: s0 -3.2 [-5.2, -1.2]; s1 -1.0 [-2.5, +0.3]; s2 -1.0 [-2.4, +0.0] -> no consistent difference detected
- trained_distribution - default, without, selection: s0 -0.8 [-3.0, +1.3]; s1 -2.6 [-4.9, -0.8]; s2 -1.8 [-3.4, -0.2] -> no consistent difference detected
- trained_distribution - default, without, calibration: s0 -3.3 [-5.3, -1.4]; s1 -0.7 [-2.2, +0.5]; s2 -1.0 [-2.4, +0.0] -> no consistent difference detected
- default - strongest previous, all, selection: s0 -2.4 [-5.2, +0.4]; s1 -2.5 [-5.2, +0.1]; s2 -2.7 [-5.3, -0.1] -> no consistent difference detected
- default - strongest previous, all, calibration: s0 +0.4 [-1.1, +1.8]; s1 -0.8 [-2.0, +0.5]; s2 -0.8 [-2.3, +0.7] -> no consistent difference detected
- mean - strongest previous, all, selection: s0 -7.8 [-11.5, -4.0]; s1 -5.1 [-8.2, -2.1]; s2 -5.6 [-8.8, -2.4] -> consistent (lower)
- mean - strongest previous, all, calibration: s0 -4.1 [-6.0, -2.2]; s1 -0.8 [-2.4, +0.7]; s2 -1.2 [-2.7, +0.3] -> no consistent difference detected
- trained_mean - strongest previous, all, selection: s0 -7.1 [-11.0, -3.3]; s1 -5.2 [-8.5, -2.1]; s2 -5.4 [-8.6, -2.2] -> consistent (lower)
- trained_mean - strongest previous, all, calibration: s0 -4.9 [-7.1, -2.6]; s1 -1.5 [-3.0, +0.1]; s2 -1.6 [-3.3, +0.0] -> no consistent difference detected
- trained_distribution - strongest previous, all, selection: s0 -7.7 [-11.3, -4.0]; s1 -4.9 [-8.1, -1.9]; s2 -5.1 [-8.3, -1.9] -> consistent (lower)
- trained_distribution - strongest previous, all, calibration: s0 -5.0 [-7.2, -2.8]; s1 -1.2 [-2.7, +0.3]; s2 -1.6 [-3.3, +0.0] -> no consistent difference detected

## A5C turning right and A4C turning wrong (units = reference segments; clips and sessions deduplicated)

| comparison | half | seed | stream units: A5C fixed | stream units: A4C broken | natives: A5C fixed | natives: A4C broken |
|---|---|---|---|---|---|---|
| mean vs default | selection | 0 | 4 (4 clips, 4 sessions; top 2 clips 50%) | 0 (0 clips, 0 sessions) | 1 (1 clips, 1 sessions; top 2 clips 100%) | 0 (0 clips, 0 sessions) |
| trained_mean vs default | selection | 0 | 5 (5 clips, 5 sessions; top 2 clips 60%) | 0 (0 clips, 0 sessions) | 1 (1 clips, 1 sessions; top 2 clips 100%) | 1 (1 clips, 1 sessions; top 2 clips 100%) |
| trained_mean vs mean | selection | 0 | 4 (5 clips, 4 sessions; top 2 clips 100%) | 3 (3 clips, 2 sessions; top 2 clips 67%) | 0 (0 clips, 0 sessions) | 1 (1 clips, 1 sessions; top 2 clips 100%) |
| trained_distribution vs default | selection | 0 | 4 (4 clips, 4 sessions; top 2 clips 50%) | 0 (0 clips, 0 sessions) | 1 (1 clips, 1 sessions; top 2 clips 100%) | 1 (1 clips, 1 sessions; top 2 clips 100%) |
| trained_distribution vs mean | selection | 0 | 0 (0 clips, 0 sessions) | 1 (1 clips, 1 sessions; top 2 clips 100%) | 0 (0 clips, 0 sessions) | 1 (1 clips, 1 sessions; top 2 clips 100%) |
| trained_distribution vs trained_mean | selection | 0 | 0 (0 clips, 0 sessions) | 0 (0 clips, 0 sessions) | 0 (0 clips, 0 sessions) | 0 (0 clips, 0 sessions) |
| mean vs default | calibration | 0 | 3 (3 clips, 3 sessions; top 2 clips 67%) | 0 (0 clips, 0 sessions) | 0 (0 clips, 0 sessions) | 1 (1 clips, 1 sessions; top 2 clips 100%) |
| trained_mean vs default | calibration | 0 | 3 (3 clips, 3 sessions; top 2 clips 67%) | 3 (5 clips, 5 sessions; top 2 clips 100%) | 0 (0 clips, 0 sessions) | 1 (1 clips, 1 sessions; top 2 clips 100%) |
| trained_mean vs mean | calibration | 0 | 1 (1 clips, 1 sessions; top 2 clips 100%) | 4 (6 clips, 6 sessions; top 2 clips 75%) | 0 (0 clips, 0 sessions) | 0 (0 clips, 0 sessions) |
| trained_distribution vs default | calibration | 0 | 3 (3 clips, 3 sessions; top 2 clips 67%) | 2 (3 clips, 3 sessions; top 2 clips 100%) | 0 (0 clips, 0 sessions) | 1 (1 clips, 1 sessions; top 2 clips 100%) |
| trained_distribution vs mean | calibration | 0 | 1 (1 clips, 1 sessions; top 2 clips 100%) | 3 (4 clips, 4 sessions; top 2 clips 67%) | 0 (0 clips, 0 sessions) | 0 (0 clips, 0 sessions) |
| trained_distribution vs trained_mean | calibration | 0 | 0 (0 clips, 0 sessions) | 0 (0 clips, 0 sessions) | 0 (0 clips, 0 sessions) | 0 (0 clips, 0 sessions) |
| mean vs default | selection | 1 | 4 (4 clips, 3 sessions; top 2 clips 75%) | 3 (3 clips, 3 sessions; top 2 clips 67%) | 1 (1 clips, 1 sessions; top 2 clips 100%) | 1 (1 clips, 1 sessions; top 2 clips 100%) |
| trained_mean vs default | selection | 1 | 4 (4 clips, 3 sessions; top 2 clips 75%) | 4 (3 clips, 3 sessions; top 2 clips 75%) | 1 (1 clips, 1 sessions; top 2 clips 100%) | 1 (1 clips, 1 sessions; top 2 clips 100%) |
| trained_mean vs mean | selection | 1 | 0 (0 clips, 0 sessions) | 2 (2 clips, 2 sessions; top 2 clips 100%) | 0 (0 clips, 0 sessions) | 0 (0 clips, 0 sessions) |
| trained_distribution vs default | selection | 1 | 4 (4 clips, 3 sessions; top 2 clips 75%) | 4 (3 clips, 3 sessions; top 2 clips 75%) | 1 (1 clips, 1 sessions; top 2 clips 100%) | 1 (1 clips, 1 sessions; top 2 clips 100%) |
| trained_distribution vs mean | selection | 1 | 0 (0 clips, 0 sessions) | 2 (2 clips, 2 sessions; top 2 clips 100%) | 0 (0 clips, 0 sessions) | 0 (0 clips, 0 sessions) |
| trained_distribution vs trained_mean | selection | 1 | 0 (0 clips, 0 sessions) | 1 (1 clips, 1 sessions; top 2 clips 100%) | 0 (0 clips, 0 sessions) | 0 (0 clips, 0 sessions) |
| mean vs default | calibration | 1 | 5 (3 clips, 3 sessions; top 2 clips 80%) | 1 (1 clips, 1 sessions; top 2 clips 100%) | 1 (1 clips, 1 sessions; top 2 clips 100%) | 0 (0 clips, 0 sessions) |
| trained_mean vs default | calibration | 1 | 5 (3 clips, 3 sessions; top 2 clips 80%) | 2 (1 clips, 1 sessions; top 2 clips 100%) | 1 (1 clips, 1 sessions; top 2 clips 100%) | 0 (0 clips, 0 sessions) |
| trained_mean vs mean | calibration | 1 | 0 (0 clips, 0 sessions) | 1 (1 clips, 1 sessions; top 2 clips 100%) | 0 (0 clips, 0 sessions) | 0 (0 clips, 0 sessions) |
| trained_distribution vs default | calibration | 1 | 5 (3 clips, 3 sessions; top 2 clips 80%) | 2 (1 clips, 1 sessions; top 2 clips 100%) | 1 (1 clips, 1 sessions; top 2 clips 100%) | 0 (0 clips, 0 sessions) |
| trained_distribution vs mean | calibration | 1 | 0 (0 clips, 0 sessions) | 1 (1 clips, 1 sessions; top 2 clips 100%) | 0 (0 clips, 0 sessions) | 0 (0 clips, 0 sessions) |
| trained_distribution vs trained_mean | calibration | 1 | 0 (0 clips, 0 sessions) | 0 (0 clips, 0 sessions) | 0 (0 clips, 0 sessions) | 0 (0 clips, 0 sessions) |
| mean vs default | selection | 2 | 3 (3 clips, 3 sessions; top 2 clips 67%) | 3 (2 clips, 2 sessions; top 2 clips 100%) | 1 (1 clips, 1 sessions; top 2 clips 100%) | 1 (1 clips, 1 sessions; top 2 clips 100%) |
| trained_mean vs default | selection | 2 | 4 (4 clips, 3 sessions; top 2 clips 75%) | 4 (2 clips, 2 sessions; top 2 clips 100%) | 1 (1 clips, 1 sessions; top 2 clips 100%) | 1 (1 clips, 1 sessions; top 2 clips 100%) |
| trained_mean vs mean | selection | 2 | 1 (2 clips, 2 sessions; top 2 clips 100%) | 2 (2 clips, 2 sessions; top 2 clips 100%) | 0 (0 clips, 0 sessions) | 0 (0 clips, 0 sessions) |
| trained_distribution vs default | selection | 2 | 4 (4 clips, 3 sessions; top 2 clips 75%) | 3 (2 clips, 2 sessions; top 2 clips 100%) | 1 (1 clips, 1 sessions; top 2 clips 100%) | 1 (1 clips, 1 sessions; top 2 clips 100%) |
| trained_distribution vs mean | selection | 2 | 1 (2 clips, 2 sessions; top 2 clips 100%) | 1 (1 clips, 1 sessions; top 2 clips 100%) | 0 (0 clips, 0 sessions) | 0 (0 clips, 0 sessions) |
| trained_distribution vs trained_mean | selection | 2 | 0 (0 clips, 0 sessions) | 0 (0 clips, 0 sessions) | 0 (0 clips, 0 sessions) | 0 (0 clips, 0 sessions) |
| mean vs default | calibration | 2 | 5 (3 clips, 3 sessions; top 2 clips 80%) | 1 (1 clips, 1 sessions; top 2 clips 100%) | 1 (1 clips, 1 sessions; top 2 clips 100%) | 0 (0 clips, 0 sessions) |
| trained_mean vs default | calibration | 2 | 5 (3 clips, 3 sessions; top 2 clips 80%) | 2 (3 clips, 3 sessions; top 2 clips 100%) | 1 (1 clips, 1 sessions; top 2 clips 100%) | 0 (0 clips, 0 sessions) |
| trained_mean vs mean | calibration | 2 | 0 (0 clips, 0 sessions) | 1 (2 clips, 2 sessions; top 2 clips 100%) | 0 (0 clips, 0 sessions) | 0 (0 clips, 0 sessions) |
| trained_distribution vs default | calibration | 2 | 5 (3 clips, 3 sessions; top 2 clips 80%) | 2 (3 clips, 3 sessions; top 2 clips 100%) | 1 (1 clips, 1 sessions; top 2 clips 100%) | 0 (0 clips, 0 sessions) |
| trained_distribution vs mean | calibration | 2 | 0 (0 clips, 0 sessions) | 1 (2 clips, 2 sessions; top 2 clips 100%) | 0 (0 clips, 0 sessions) | 0 (0 clips, 0 sessions) |
| trained_distribution vs trained_mean | calibration | 2 | 0 (0 clips, 0 sessions) | 0 (0 clips, 0 sessions) | 0 (0 clips, 0 sessions) | 0 (0 clips, 0 sessions) |

## False pieces: unmatched predicted A5C segments inside A4C units, and A4C inside A5C units

| rule | half | A5C inside A4C, s0 / s1 / s2 (clips) | A4C inside A5C, s0 / s1 / s2 (clips) | predicted segments s0 / s1 / s2 |
|---|---|---|---|---|
| default | selection | 9 (6) / 2 (2) / 2 (2) | 8 (5) / 18 (8) / 19 (8) | 498 / 508 / 510 |
| default | calibration | 6 (5) / 3 (4) / 3 (3) | 2 (4) / 7 (8) / 8 (8) | 517 / 519 / 521 |
| mean | selection | 14 (12) / 10 (9) / 11 (10) | 36 (9) / 30 (9) / 30 (9) | 570 / 542 / 545 |
| mean | calibration | 11 (14) / 5 (6) / 6 (7) | 28 (16) / 9 (10) / 11 (12) | 573 / 532 / 536 |
| trained_mean | selection | 16 (13) / 10 (9) / 12 (12) | 31 (9) / 28 (9) / 26 (9) | 564 / 538 / 541 |
| trained_mean | calibration | 14 (17) / 9 (10) / 7 (9) | 28 (16) / 9 (10) / 11 (12) | 576 / 537 / 538 |
| trained_distribution | selection | 13 (10) / 10 (7) / 10 (8) | 35 (9) / 28 (8) / 27 (8) | 566 / 535 / 539 |
| trained_distribution | calibration | 14 (17) / 7 (8) / 7 (9) | 30 (17) / 9 (10) / 11 (12) | 580 / 534 / 538 |
