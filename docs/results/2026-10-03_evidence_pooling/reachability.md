# Reachability of an A4C/A5C decision (round 5, step 1; diagnostic only)

Targets: selection 0.9444, calibration 0.9821. Floors: E0 selection 0.9600; E0 calibration 0.9773; E1 selection 0.9730; E1 calibration 0.9800.

| base | spans | half | candidate-cut recall per seed | oracle segment F1@50 per seed (mean) | oracle recall per seed | oracle native F1 per seed | reachable on this half |
|---|---|---|---|---|---|---|---|
| E0 ASRF | final | selection | 0.937 / 0.953 / 0.966 | 0.975 / 0.980 / 0.985 (0.980) | 0.937 / 0.953 / 0.966 | 1.000 / 1.000 / 1.000 | no |
| E0 ASRF | final | calibration | 0.950 / 0.965 / 0.977 | 0.971 / 0.981 / 0.994 (0.982) | 0.950 / 0.962 / 0.977 | 0.994 / 0.997 / 1.000 | no |
| E0 ASRF | candidate | selection | 0.948 / 0.961 / 0.982 | 0.976 / 0.981 / 0.991 (0.982) | 0.942 / 0.955 / 0.979 | 1.000 / 1.000 / 1.000 | no |
| E0 ASRF | candidate | calibration | 0.970 / 0.970 / 0.987 | 0.977 / 0.981 / 0.995 (0.984) | 0.965 / 0.965 / 0.987 | 0.994 / 0.997 / 1.000 | yes |
| E0 B3 (reference) | final | selection | 0.961 | 0.982 (0.982) | 0.958 | 1.000 | yes |
| E0 B3 (reference) | final | calibration | 0.955 | 0.975 (0.975) | 0.955 | 0.997 | no |
| E0 B3 (reference) | candidate | selection | 0.961 | 0.982 (0.982) | 0.958 | 1.000 | yes |
| E0 B3 (reference) | candidate | calibration | 0.960 | 0.975 (0.975) | 0.955 | 0.997 | no |
| E1 ASRF | final | selection | 0.942 / 0.950 / 0.953 | 0.947 / 0.955 / 0.956 (0.952) | 0.940 / 0.950 / 0.953 | 0.989 / 0.989 / 0.989 | no |
| E1 ASRF | final | calibration | 0.977 / 0.972 / 0.977 | 0.991 / 0.986 / 0.988 (0.989) | 0.975 / 0.972 / 0.977 | 1.000 / 0.997 / 0.997 | yes |
| E1 ASRF | candidate | selection | 0.979 / 0.963 / 0.969 | 0.960 / 0.958 / 0.959 (0.959) | 0.976 / 0.961 / 0.963 | 0.989 / 0.989 / 0.989 | yes |
| E1 ASRF | candidate | calibration | 0.995 / 0.992 / 0.992 | 0.999 / 0.993 / 0.993 (0.995) | 0.995 / 0.992 / 0.992 | 1.000 / 0.997 / 0.997 | yes |
| E1 B3 (reference) | final | selection | 0.955 | 0.947 (0.947) | 0.953 | 0.989 | yes |
| E1 B3 (reference) | final | calibration | 0.977 | 0.987 (0.987) | 0.977 | 1.000 | yes |
| E1 B3 (reference) | candidate | selection | 0.955 | 0.947 (0.947) | 0.953 | 0.989 | yes |
| E1 B3 (reference) | candidate | calibration | 0.977 | 0.987 (0.987) | 0.977 | 1.000 | yes |

**Reachable on both halves (pre-declared test):** E0 ASRF, final spans: no; E0 ASRF, candidate spans: no; E0 B3 (reference), final spans: no; E0 B3 (reference), candidate spans: no; E1 ASRF, final spans: no; E1 ASRF, candidate spans: yes; E1 B3 (reference), final spans: yes; E1 B3 (reference), candidate spans: yes

## Sensitivity: all clips against the ten clips removed (same list for every system, nothing re-tuned)

| system | half | segment F1@50 all -> without | recall all -> without | native F1 all -> without |
|---|---|---|---|---|
| E0 B3 | selection | 0.934 -> 0.987 | 0.961 -> 0.987 | 0.965 -> 0.986 |
| E0 ASRF_s0 | selection | 0.912 -> 0.981 | 0.937 -> 0.958 | 0.965 -> 0.986 |
| E0 ASRF_s1 | selection | 0.946 -> 0.984 | 0.953 -> 0.966 | 0.968 -> 0.973 |
| E0 ASRF_s2 | selection | 0.924 -> 0.990 | 0.966 -> 0.987 | 0.965 -> 0.986 |
| E0 B3 | calibration | 0.934 -> 0.973 | 0.955 -> 0.965 | 0.966 -> 0.990 |
| E0 ASRF_s0 | calibration | 0.936 -> 0.971 | 0.950 -> 0.953 | 0.971 -> 0.997 |
| E0 ASRF_s1 | calibration | 0.960 -> 0.986 | 0.965 -> 0.969 | 0.975 -> 0.990 |
| E0 ASRF_s2 | calibration | 0.960 -> 0.994 | 0.977 -> 0.988 | 0.970 -> 0.990 |
| E1 B3 | selection | 0.894 -> 0.945 | 0.955 -> 0.975 | 0.978 -> 0.989 |
| E1 ASRF_s0 | selection | 0.910 -> 0.939 | 0.942 -> 0.945 | 0.948 -> 0.950 |
| E1 ASRF_s1 | selection | 0.909 -> 0.968 | 0.950 -> 0.975 | 0.966 -> 0.974 |
| E1 ASRF_s2 | selection | 0.907 -> 0.967 | 0.953 -> 0.966 | 0.966 -> 0.974 |
| system B7_B3 | selection | 0.880 -> 0.940 | 0.955 -> 0.966 | 0.957 -> 0.961 |
| E1 B3 | calibration | 0.959 -> 0.976 | 0.977 -> 0.992 | 0.986 -> 0.990 |
| E1 ASRF_s0 | calibration | 0.976 -> 0.988 | 0.977 -> 0.984 | 0.993 -> 0.990 |
| E1 ASRF_s1 | calibration | 0.964 -> 0.985 | 0.972 -> 0.977 | 0.975 -> 0.988 |
| E1 ASRF_s2 | calibration | 0.964 -> 0.987 | 0.977 -> 0.984 | 0.975 -> 0.988 |
| system B7_B3 | calibration | 0.972 -> 0.987 | 0.975 -> 0.981 | 0.993 -> 1.000 |

Removed: E0 selection: natives 284->281, pairs 137->134, streams 120->77; E0 calibration: natives 237->230, pairs 115->109, streams 120->77
