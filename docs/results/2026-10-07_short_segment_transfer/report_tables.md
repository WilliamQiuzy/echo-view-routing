# Short-segment transfer check: tables

## A4C/A5C first (threshold 0.95)

| seed | half | model | pairs | correct kept / wrongly removed | correct rec / sess | wrong removed / missed | wrong rec / sess | AUC (all) |
|---|---|---|---|---|---|---|---|---|
| 0 | selection | Q0 | A4C/A5C pairs | 3 / 3 | 9 / 9 | 9 / 0 | 7 / 6 | 0.9866 |
| 0 | selection | Q0 | other pairs | 50 / 0 | 56 / 26 | 2 / 1 | 3 / 3 | 0.9866 |
| 0 | selection | Q1 | A4C/A5C pairs | 2 / 4 | 9 / 9 | 9 / 0 | 7 / 6 | 0.9226 |
| 0 | selection | Q1 | other pairs | 49 / 1 | 56 / 26 | 2 / 1 | 3 / 3 | 0.9226 |
| 0 | calibration | Q0 | A4C/A5C pairs | 5 / 2 | 9 / 7 | 4 / 1 | 4 / 4 | 0.9962 |
| 0 | calibration | Q0 | other pairs | 69 / 0 | 70 / 25 | 2 / 0 | 2 / 1 | 0.9962 |
| 0 | calibration | Q1 | A4C/A5C pairs | 5 / 2 | 9 / 7 | 5 / 0 | 4 / 4 | 0.9812 |
| 0 | calibration | Q1 | other pairs | 69 / 0 | 70 / 25 | 1 / 1 | 2 / 1 | 0.9812 |
| 0 | pooled halves | Q0 | A4C/A5C pairs | 8 / 5 | 18 / 16 | 13 / 1 | 11 / 10 | 0.9924 |
| 0 | pooled halves | Q0 | other pairs | 119 / 0 | 126 / 51 | 4 / 1 | 5 / 4 | 0.9924 |
| 0 | pooled halves | Q1 | A4C/A5C pairs | 7 / 6 | 18 / 16 | 14 / 0 | 11 / 10 | 0.9470 |
| 0 | pooled halves | Q1 | other pairs | 118 / 1 | 126 / 51 | 3 / 2 | 5 / 4 | 0.9470 |
| 1 | selection | Q0 | A4C/A5C pairs | 3 / 3 | 9 / 9 | 12 / 1 | 7 / 6 | 0.9750 |
| 1 | selection | Q0 | other pairs | 49 / 0 | 57 / 27 | 2 / 1 | 3 / 3 | 0.9750 |
| 1 | selection | Q1 | A4C/A5C pairs | 2 / 4 | 9 / 9 | 13 / 0 | 7 / 6 | 0.9534 |
| 1 | selection | Q1 | other pairs | 49 / 0 | 57 / 27 | 2 / 1 | 3 / 3 | 0.9534 |
| 1 | calibration | Q0 | A4C/A5C pairs | 6 / 1 | 9 / 7 | 4 / 1 | 5 / 5 | 0.9981 |
| 1 | calibration | Q0 | other pairs | 69 / 0 | 70 / 25 | 2 / 0 | 2 / 1 | 0.9981 |
| 1 | calibration | Q1 | A4C/A5C pairs | 5 / 2 | 9 / 7 | 5 / 0 | 5 / 5 | 0.9944 |
| 1 | calibration | Q1 | other pairs | 69 / 0 | 70 / 25 | 2 / 0 | 2 / 1 | 0.9944 |
| 1 | pooled halves | Q0 | A4C/A5C pairs | 9 / 4 | 18 / 16 | 16 / 2 | 12 / 11 | 0.9874 |
| 1 | pooled halves | Q0 | other pairs | 118 / 0 | 127 / 52 | 4 / 1 | 5 / 4 | 0.9874 |
| 1 | pooled halves | Q1 | A4C/A5C pairs | 7 / 6 | 18 / 16 | 18 / 0 | 12 / 11 | 0.9661 |
| 1 | pooled halves | Q1 | other pairs | 118 / 0 | 127 / 52 | 4 / 1 | 5 / 4 | 0.9661 |
| 2 | selection | Q0 | A4C/A5C pairs | 3 / 3 | 9 / 9 | 16 / 0 | 11 / 9 | 0.9877 |
| 2 | selection | Q0 | other pairs | 48 / 0 | 54 / 26 | 1 / 1 | 2 / 2 | 0.9877 |
| 2 | selection | Q1 | A4C/A5C pairs | 3 / 3 | 9 / 9 | 16 / 0 | 11 / 9 | 0.9465 |
| 2 | selection | Q1 | other pairs | 46 / 2 | 54 / 26 | 1 / 1 | 2 / 2 | 0.9465 |
| 2 | calibration | Q0 | A4C/A5C pairs | 6 / 1 | 9 / 7 | 5 / 1 | 5 / 5 | 0.9983 |
| 2 | calibration | Q0 | other pairs | 68 / 0 | 69 / 25 | 2 / 0 | 2 / 1 | 0.9983 |
| 2 | calibration | Q1 | A4C/A5C pairs | 5 / 2 | 9 / 7 | 6 / 0 | 5 / 5 | 0.9883 |
| 2 | calibration | Q1 | other pairs | 68 / 0 | 69 / 25 | 2 / 0 | 2 / 1 | 0.9883 |
| 2 | pooled halves | Q0 | A4C/A5C pairs | 9 / 4 | 18 / 16 | 21 / 1 | 16 / 14 | 0.9922 |
| 2 | pooled halves | Q0 | other pairs | 116 / 0 | 123 / 51 | 3 / 1 | 4 / 3 | 0.9922 |
| 2 | pooled halves | Q1 | A4C/A5C pairs | 8 / 5 | 18 / 16 | 22 / 0 | 16 / 14 | 0.9630 |
| 2 | pooled halves | Q1 | other pairs | 114 / 2 | 123 / 51 | 3 / 1 | 4 / 3 | 0.9630 |

## Outside the binary task

| seed | half | candidates | outside the binary task | share | sides not X | a third class under the middle | part of a Y unit (IoU < 0.5) | a reference change under the middle |
|---|---|---|---|---|---|---|---|---|
| 0 | selection | 87 | 19 | 0.2184 | 17 | 1 | 1 | 0 |
| 0 | calibration | 89 | 6 | 0.0674 | 6 | 0 | 0 | 0 |
| 0 | pooled halves | 176 | 25 | 0.1420 | 23 | 1 | 1 | 0 |
| 1 | selection | 85 | 14 | 0.1647 | 13 | 0 | 1 | 0 |
| 1 | calibration | 90 | 7 | 0.0778 | 7 | 0 | 0 | 0 |
| 1 | pooled halves | 175 | 21 | 0.1200 | 20 | 0 | 1 | 0 |
| 2 | selection | 92 | 20 | 0.2174 | 18 | 1 | 1 | 0 |
| 2 | calibration | 90 | 7 | 0.0778 | 6 | 1 | 0 | 0 |
| 2 | pooled halves | 182 | 27 | 0.1484 | 24 | 2 | 1 | 0 |

## Sources and joins (halves pooled)

| seed | model | subset | correct kept / n | wrong removed / n | wrong rec / sess |
|---|---|---|---|---|---|
| 0 | Q0 | native clips | 0 / 0 | 10 / 11 | 10 / 8 |
| 0 | Q0 | pair streams | 0 / 0 | 1 / 1 | 1 / 1 |
| 0 | Q0 | multi streams | 127 / 132 | 6 / 7 | 7 / 6 |
| 0 | Q0 | no join | 0 / 0 | 13 / 14 | 10 / 8 |
| 0 | Q0 | same-view join | 0 / 0 | 3 / 4 | 6 / 5 |
| 0 | Q0 | view-change join | 127 / 132 | 1 / 1 | 1 / 1 |
| 0 | Q0 | middle 10-40 samples | 102 / 107 | 7 / 8 | 10 / 9 |
| 0 | Q1 | native clips | 0 / 0 | 10 / 11 | 10 / 8 |
| 0 | Q1 | pair streams | 0 / 0 | 1 / 1 | 1 / 1 |
| 0 | Q1 | multi streams | 125 / 132 | 6 / 7 | 7 / 6 |
| 0 | Q1 | no join | 0 / 0 | 13 / 14 | 10 / 8 |
| 0 | Q1 | same-view join | 0 / 0 | 3 / 4 | 6 / 5 |
| 0 | Q1 | view-change join | 125 / 132 | 1 / 1 | 1 / 1 |
| 0 | Q1 | middle 10-40 samples | 100 / 107 | 7 / 8 | 10 / 9 |
| 1 | Q0 | native clips | 0 / 0 | 11 / 12 | 10 / 9 |
| 1 | Q0 | pair streams | 0 / 0 | 2 / 2 | 2 / 2 |
| 1 | Q0 | multi streams | 127 / 131 | 7 / 9 | 9 / 8 |
| 1 | Q0 | no join | 0 / 0 | 15 / 16 | 10 / 9 |
| 1 | Q0 | same-view join | 0 / 0 | 4 / 6 | 8 / 7 |
| 1 | Q0 | view-change join | 127 / 131 | 1 / 1 | 1 / 1 |
| 1 | Q0 | middle 10-40 samples | 103 / 107 | 12 / 14 | 13 / 12 |
| 1 | Q1 | native clips | 0 / 0 | 12 / 12 | 10 / 9 |
| 1 | Q1 | pair streams | 0 / 0 | 2 / 2 | 2 / 2 |
| 1 | Q1 | multi streams | 125 / 131 | 8 / 9 | 9 / 8 |
| 1 | Q1 | no join | 0 / 0 | 16 / 16 | 10 / 9 |
| 1 | Q1 | same-view join | 0 / 0 | 5 / 6 | 8 / 7 |
| 1 | Q1 | view-change join | 125 / 131 | 1 / 1 | 1 / 1 |
| 1 | Q1 | middle 10-40 samples | 101 / 107 | 13 / 14 | 13 / 12 |
| 2 | Q0 | native clips | 0 / 0 | 15 / 16 | 13 / 10 |
| 2 | Q0 | pair streams | 0 / 0 | 3 / 3 | 4 / 4 |
| 2 | Q0 | multi streams | 125 / 129 | 6 / 7 | 8 / 7 |
| 2 | Q0 | no join | 0 / 0 | 19 / 20 | 13 / 10 |
| 2 | Q0 | same-view join | 0 / 0 | 5 / 6 | 9 / 8 |
| 2 | Q0 | view-change join | 125 / 129 | 0 / 0 | 0 / 0 |
| 2 | Q0 | middle 10-40 samples | 100 / 104 | 11 / 12 | 11 / 10 |
| 2 | Q1 | native clips | 0 / 0 | 16 / 16 | 13 / 10 |
| 2 | Q1 | pair streams | 0 / 0 | 3 / 3 | 4 / 4 |
| 2 | Q1 | multi streams | 122 / 129 | 6 / 7 | 8 / 7 |
| 2 | Q1 | no join | 0 / 0 | 20 / 20 | 13 / 10 |
| 2 | Q1 | same-view join | 0 / 0 | 5 / 6 | 9 / 8 |
| 2 | Q1 | view-change join | 122 / 129 | 0 / 0 | 0 / 0 |
| 2 | Q1 | middle 10-40 samples | 97 / 104 | 11 / 12 | 11 / 10 |

## Decision

```
{
 "case": "Q0 high overall but removes correct A4C/A5C middles",
 "fails": {
  "Q0": false,
  "Q1": false
 },
 "Q0_practical": false,
 "Q0_high_but_aa": true,
 "Q1_extra": false
}
```
