# Round 6: span-level A4C/A5C training, V0 (frozen encoder) and V1 (last block trained)

Targets: selection 0.9444, calibration 0.9821; floors 0.973 and 0.980.

## Spans (per seed: training split / selection / calibration)

| seed | part | spans (A4C/A5C majority) | A5C | median length | p90 length | mixed share | vote differs from majority |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | train | 6120 | 2852 | 16 | 43 | 2.0% | 9.7% |
| 0 | selection | 556 | 197 | 17 | 42 | 2.7% | 10.4% |
| 0 | calibration | 530 | 205 | 16 | 41 | 1.7% | 12.3% |
| 1 | train | 4681 | 2076 | 21 | 55 | 2.4% | 8.1% |
| 1 | selection | 463 | 171 | 21 | 53 | 3.7% | 11.0% |
| 1 | calibration | 416 | 160 | 21 | 49 | 2.9% | 7.7% |
| 2 | train | 5251 | 2429 | 19 | 48 | 1.8% | 6.4% |
| 2 | selection | 514 | 202 | 18 | 47 | 2.9% | 10.9% |
| 2 | calibration | 447 | 178 | 20 | 47 | 2.0% | 10.1% |

## Span classification (validation spans with an A4C or A5C majority)

| half | rule | accuracy s0 / s1 / s2 | A4C recall | A5C recall | source clips mostly right |
|---|---|---|---|---|---|
| selection | vote | 0.896 / 0.890 / 0.891 | 0.875 / 0.983 / 0.965 | 0.934 / 0.731 / 0.777 | 97% / 97% / 97% |
| selection | mean | 0.840 / 0.860 / 0.858 | 0.894 / 0.925 / 0.926 | 0.741 / 0.749 / 0.752 | 96% / 98% / 97% |
| selection | V0 | 0.858 / 0.875 / 0.875 | 0.894 / 0.935 / 0.923 | 0.792 / 0.772 / 0.802 | 95% / 97% / 97% |
| selection | V1 | 0.883 / 0.901 / 0.883 | 0.900 / 0.942 / 0.923 | 0.853 / 0.830 / 0.822 | 97% / 98% / 96% |
| calibration | vote | 0.877 / 0.923 / 0.899 | 0.849 / 0.961 / 0.929 | 0.922 / 0.863 / 0.854 | 95% / 97% / 97% |
| calibration | mean | 0.843 / 0.923 / 0.902 | 0.898 / 0.957 / 0.941 | 0.756 / 0.869 / 0.843 | 97% / 99% / 97% |
| calibration | V0 | 0.843 / 0.921 / 0.897 | 0.883 / 0.941 / 0.918 | 0.780 / 0.887 / 0.865 | 97% / 97% / 96% |
| calibration | V1 | 0.874 / 0.928 / 0.906 | 0.892 / 0.941 / 0.922 | 0.844 / 0.906 / 0.882 | 95% / 97% / 97% |

## Systems: All clips (main result)

| half | system | segment F1@50 s0 / s1 / s2 (mean) | recall | native macro-F1 | predicted segments |
|---|---|---|---|---|---|
| selection | default | 0.910 / 0.909 / 0.907 (0.909) | 0.942 / 0.950 / 0.953 | 0.948 / 0.966 / 0.966 | 498 / 508 / 510 |
| selection | mean | 0.856 / 0.883 / 0.879 (0.873) | 0.971 / 0.958 / 0.961 | 0.959 / 0.968 / 0.968 | 570 / 542 / 545 |
| selection | V0 | 0.869 / 0.885 / 0.883 (0.879) | 0.974 / 0.958 / 0.963 | 0.951 / 0.968 / 0.968 | 559 / 537 / 538 |
| selection | V1 | 0.887 / 0.897 / 0.884 (0.889) | 0.971 / 0.958 / 0.963 | 0.968 / 0.978 / 0.978 | 547 / 533 / 545 |
| calibration | default | 0.976 / 0.964 / 0.964 (0.968) | 0.977 / 0.972 / 0.977 | 0.993 / 0.975 / 0.975 | 517 / 519 / 521 |
| calibration | mean | 0.931 / 0.964 / 0.960 (0.952) | 0.992 / 0.990 / 0.990 | 0.986 / 0.983 / 0.983 | 573 / 532 / 536 |
| calibration | V0 | 0.924 / 0.960 / 0.953 (0.946) | 0.992 / 0.990 / 0.987 | 0.986 / 0.983 / 0.983 | 577 / 534 / 540 |
| calibration | V1 | 0.943 / 0.967 / 0.964 (0.958) | 0.995 / 0.987 / 0.987 | 0.986 / 0.976 / 0.983 | 559 / 531 / 532 |

## Systems: Without the ten sensitivity clips (dependence only)

| half | system | segment F1@50 s0 / s1 / s2 (mean) | recall | native macro-F1 | predicted segments |
|---|---|---|---|---|---|
| selection | default | 0.939 / 0.968 / 0.967 (0.958) | 0.945 / 0.975 / 0.966 | 0.950 / 0.974 / 0.974 | 309 / 313 / 314 |
| selection | mean | 0.929 / 0.944 / 0.947 (0.940) | 0.983 / 0.971 / 0.979 | 0.964 / 0.976 / 0.976 | 337 / 325 / 327 |
| selection | V0 | 0.931 / 0.945 / 0.945 (0.940) | 0.983 / 0.971 / 0.979 | 0.954 / 0.976 / 0.976 | 334 / 322 / 326 |
| selection | V1 | 0.930 / 0.944 / 0.944 (0.939) | 0.975 / 0.971 / 0.979 | 0.976 / 0.974 / 0.989 | 330 / 323 / 327 |
| calibration | default | 0.988 / 0.985 / 0.987 (0.986) | 0.984 / 0.977 / 0.984 | 0.990 / 0.988 / 0.988 | 332 / 330 / 333 |
| calibration | mean | 0.971 / 0.981 / 0.976 (0.976) | 0.992 / 0.988 / 0.988 | 0.990 / 0.988 / 0.988 | 348 / 337 / 340 |
| calibration | V0 | 0.956 / 0.978 / 0.970 (0.968) | 0.992 / 0.988 / 0.988 | 0.990 / 0.988 / 0.988 | 352 / 337 / 342 |
| calibration | V1 | 0.971 / 0.982 / 0.979 (0.977) | 0.992 / 0.988 / 0.988 | 0.990 / 0.979 / 0.988 | 346 / 336 / 338 |

## Success test (all clips)

| system | half | mean (target) | min recall | min native F1 (floor) | passes |
|---|---|---|---|---|---|
| default | selection | 0.909 (0.944) | 0.942 | 0.948 (0.973) | no |
| default | calibration | 0.968 (0.982) | 0.972 | 0.975 (0.980) | no |
| mean | selection | 0.873 (0.944) | 0.958 | 0.959 (0.973) | no |
| mean | calibration | 0.952 (0.982) | 0.990 | 0.983 (0.980) | no |
| V0 | selection | 0.879 (0.944) | 0.958 | 0.951 (0.973) | no |
| V0 | calibration | 0.946 (0.982) | 0.987 | 0.983 (0.980) | no |
| V1 | selection | 0.889 (0.944) | 0.958 | 0.968 (0.973) | no |
| V1 | calibration | 0.958 (0.982) | 0.987 | 0.976 (0.980) | no |

## Paired contrasts of segment F1@50 (points per seed, 95% intervals)

- V1 - V0, all, selection: +1.8 [-0.9, +4.3] / +1.1 [-0.6, +2.9] / +0.2 [-2.0, +2.3], no consistent difference detected
- V1 - V0, all, calibration: +1.9 [+0.8, +3.1] / +0.7 [-0.2, +1.6] / +1.1 [-0.1, +2.5], no consistent difference detected
- V0 - mean, all, selection: +1.3 [+0.2, +2.6] / +0.2 [-0.3, +0.8] / +0.4 [-0.5, +1.2], no consistent difference detected
- V0 - mean, all, calibration: -0.7 [-2.0, +0.2] / -0.4 [-0.9, +0.0] / -0.7 [-1.9, +0.0], no consistent difference detected
- V1 - mean, all, selection: +3.0 [+0.3, +5.8] / +1.3 [-0.2, +3.0] / +0.6 [-1.4, +2.5], no consistent difference detected
- V1 - mean, all, calibration: +1.2 [-0.4, +2.6] / +0.3 [-0.6, +1.2] / +0.4 [-0.9, +1.6], no consistent difference detected
- V0 - default, all, selection: -4.1 [-6.6, -1.5] / -2.3 [-4.1, -0.8] / -2.4 [-4.0, -1.0], consistent (lower)
- V0 - default, all, calibration: -5.2 [-7.4, -2.9] / -0.4 [-1.7, +1.0] / -1.2 [-2.7, +0.4], no consistent difference detected
- V1 - default, all, selection: -2.3 [-5.0, +0.3] / -1.2 [-3.2, +0.6] / -2.3 [-4.5, -0.1], no consistent difference detected
- V1 - default, all, calibration: -3.2 [-5.5, -1.1] / +0.2 [-0.9, +1.5] / -0.1 [-1.5, +1.2], no consistent difference detected
- mean - default, all, selection: -5.4 [-8.1, -2.7] / -2.6 [-4.4, -0.9] / -2.8 [-4.4, -1.3], consistent (lower)
- mean - default, all, calibration: -4.5 [-6.5, -2.3] / -0.1 [-1.3, +1.2] / -0.4 [-1.6, +0.8], no consistent difference detected
- V1 - V0, without, selection: -0.0 [-1.8, +1.7] / -0.1 [-1.2, +0.8] / -0.1 [-1.7, +1.4], no consistent difference detected
- V1 - V0, without, calibration: +1.4 [+0.3, +3.1] / +0.4 [+0.0, +1.2] / +0.9 [-0.5, +2.6], no consistent difference detected
- V0 - mean, without, selection: +0.1 [-0.5, +0.8] / +0.1 [-0.5, +0.9] / -0.2 [-1.0, +0.6], no consistent difference detected
- V0 - mean, without, calibration: -1.4 [-3.2, +0.0] / -0.3 [-0.9, +0.0] / -0.6 [-1.8, +0.0], no consistent difference detected
- V1 - mean, without, selection: +0.1 [-1.8, +2.1] / -0.0 [-0.8, +0.6] / -0.3 [-1.8, +1.1], no consistent difference detected
- V1 - mean, without, calibration: -0.0 [-2.0, +1.8] / +0.1 [+0.0, +0.5] / +0.3 [-0.7, +1.5], no consistent difference detected
- V0 - default, without, selection: -0.8 [-3.0, +1.3] / -2.3 [-4.6, -0.5] / -2.1 [-3.8, -0.4], no consistent difference detected
- V0 - default, without, calibration: -3.2 [-5.2, -1.2] / -0.7 [-2.2, +0.5] / -1.6 [-3.5, -0.3], no consistent difference detected
- V1 - default, without, selection: -0.9 [-3.4, +1.6] / -2.5 [-4.9, -0.5] / -2.3 [-4.3, -0.2], no consistent difference detected
- V1 - default, without, calibration: -1.7 [-3.9, +0.4] / -0.3 [-1.6, +0.9] / -0.7 [-1.8, +0.1], no consistent difference detected
- mean - default, without, selection: -1.0 [-3.3, +1.3] / -2.4 [-4.7, -0.6] / -2.0 [-3.7, -0.3], no consistent difference detected
- mean - default, without, calibration: -1.7 [-3.1, -0.4] / -0.4 [-1.7, +0.8] / -1.0 [-2.4, +0.0], no consistent difference detected
- default - strongest previous, all, selection: -2.4 [-5.2, +0.4] / -2.5 [-5.2, +0.1] / -2.7 [-5.3, -0.1], no consistent difference detected
- default - strongest previous, all, calibration: +0.4 [-1.1, +1.8] / -0.8 [-2.0, +0.5] / -0.8 [-2.3, +0.7], no consistent difference detected
- mean - strongest previous, all, selection: -7.8 [-11.5, -4.0] / -5.1 [-8.2, -2.1] / -5.6 [-8.8, -2.4], consistent (lower)
- mean - strongest previous, all, calibration: -4.1 [-6.0, -2.2] / -0.8 [-2.4, +0.7] / -1.2 [-2.7, +0.3], no consistent difference detected
- V0 - strongest previous, all, selection: -6.5 [-10.2, -2.9] / -4.9 [-8.2, -1.9] / -5.2 [-8.4, -1.9], consistent (lower)
- V0 - strongest previous, all, calibration: -4.8 [-7.0, -2.6] / -1.2 [-2.7, +0.3] / -1.9 [-3.9, -0.1], no consistent difference detected
- V1 - strongest previous, all, selection: -4.8 [-8.1, -1.5] / -3.8 [-6.8, -1.0] / -5.0 [-8.2, -1.7], consistent (lower)
- V1 - strongest previous, all, calibration: -2.9 [-4.9, -1.0] / -0.6 [-1.9, +0.9] / -0.8 [-2.5, +0.7], no consistent difference detected

## A5C turning right and A4C turning wrong (stream units; native clips in brackets)

| comparison | half | A5C fixed s0 / s1 / s2 (clips) | A4C broken s0 / s1 / s2 (clips) |
|---|---|---|---|
| V0 vs default | selection | 5 (5) [1] / 4 (4) [1] / 4 (4) [1] | 0 (0) [1] / 3 (3) [1] / 4 (2) [1] |
| V0 vs default | calibration | 3 (3) [0] / 5 (3) [1] / 5 (3) [1] | 2 (3) [1] / 2 (1) [0] / 3 (5) [0] |
| V1 vs default | selection | 9 (6) [2] / 9 (4) [1] / 10 (5) [2] | 4 (5) [1] / 4 (3) [0] / 5 (5) [0] |
| V1 vs default | calibration | 3 (3) [0] / 6 (4) [1] / 6 (4) [1] | 2 (3) [0] / 1 (1) [1] / 2 (3) [0] |
| V0 vs mean | selection | 4 (5) [0] / 0 (0) [0] / 1 (2) [0] | 2 (2) [1] / 1 (1) [0] / 2 (2) [0] |
| V0 vs mean | calibration | 1 (1) [0] / 0 (0) [0] / 0 (0) [0] | 3 (4) [0] / 1 (1) [0] / 2 (4) [0] |
| V1 vs mean | selection | 15 (8) [1] / 7 (3) [1] / 9 (4) [1] | 4 (5) [1] / 2 (2) [0] / 3 (4) [0] |
| V1 vs mean | calibration | 2 (2) [0] / 2 (4) [0] / 2 (4) [0] | 2 (3) [1] / 0 (0) [1] / 2 (3) [0] |
| V1 vs V0 | selection | 12 (7) [1] / 7 (3) [1] / 8 (4) [1] | 4 (5) [1] / 2 (2) [0] / 3 (4) [0] |
| V1 vs V0 | calibration | 1 (1) [0] / 2 (4) [0] / 2 (4) [0] | 0 (0) [1] / 0 (0) [1] / 1 (1) [0] |

## False pieces (unmatched predicted segments inside a reference segment of the other class; clips in brackets)

| system | half | A5C inside A4C s0 / s1 / s2 | A4C inside A5C s0 / s1 / s2 |
|---|---|---|---|
| default | selection | 9 (6) / 2 (2) / 2 (2) | 8 (5) / 18 (8) / 19 (8) |
| default | calibration | 6 (5) / 3 (4) / 3 (3) | 2 (4) / 7 (8) / 8 (8) |
| mean | selection | 14 (12) / 10 (9) / 11 (10) | 36 (9) / 30 (9) / 30 (9) |
| mean | calibration | 11 (14) / 5 (6) / 6 (7) | 28 (16) / 9 (10) / 11 (12) |
| V0 | selection | 13 (10) / 9 (7) / 11 (10) | 30 (9) / 28 (8) / 25 (8) |
| V0 | calibration | 14 (17) / 7 (8) / 8 (10) | 28 (16) / 9 (10) / 11 (12) |
| V1 | selection | 15 (13) / 10 (9) / 13 (14) | 17 (10) / 21 (9) / 24 (9) |
| V1 | calibration | 11 (16) / 6 (8) / 7 (10) | 16 (14) / 6 (9) / 7 (10) |

## Training

- seed 0: V0 final loss 0.2759; V1 final loss 0.1503, 32 min; parity 2.9e-06 / 4.9e-04
- seed 1: V0 final loss 0.2355; V1 final loss 0.1494, 29 min; parity 2.9e-06 / 4.9e-04
- seed 2: V0 final loss 0.2452; V1 final loss 0.1485, 30 min; parity 2.9e-06 / 4.9e-04

## Pre-declared terms and reading

```
{
 "V0": {
  "classification_improves": false,
  "system_improves": false,
  "fragments_increase": true
 },
 "V1": {
  "classification_improves": false,
  "system_improves": false,
  "fragments_increase": true
 },
 "v1_beats_v0": false
}
```

Reading: **neither improves: the short-span classification route ends**
