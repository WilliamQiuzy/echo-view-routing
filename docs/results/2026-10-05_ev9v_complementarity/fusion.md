# Task 2: fixed equal-weight fusion

| system | status | params |
|---|---|---|
| E0 frozen | frozen original | {"half_window": 9, "threshold": 0.39023423383871153, "nms_radius": 5, "threshold_quantile": 0.999} |
| B7 frozen | frozen original | {"half_window": 11, "threshold": 0.0789225098201578, "nms_radius": 5, "threshold_quantile": 0.99} |
| E0 | infeasible | None |
| B7 | feasible | {"half_window": 9, "threshold": 0.0005778548352362705, "nms_radius": 5, "threshold_quantile": 0.8} |
| q | feasible | {"half_window": 11, "threshold": 0.22802094488504168, "nms_radius": 5, "threshold_quantile": 0.999} |

## Metrics

| system | status | half | native_macro_f1 | native_A4C_as_A5C | native_A5C_as_A4C | segment_f1_50_multi | boundary_recall_multi | native_internal_cuts_per_clip | units_wrong | units_A5C_as_A4C | units_A4C_as_A5C | extra_cuts_inside_units |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E0 frozen | frozen original | selection | 0.9650 | 0 | 3 | 0.9344 | 0.9607 | 0.0141 | 39 | 25 | 3 | 2 |
| E0 frozen | frozen original | calibration | 0.9660 | 0 | 4 | 0.9340 | 0.9548 | 0.0338 | 37 | 23 | 3 | 12 |
| B7 frozen | frozen original | selection | 0.9566 | 1 | 2 | 0.8798 | 0.9555 | 0.0634 | 48 | 17 | 5 | 40 |
| B7 frozen | frozen original | calibration | 0.9927 | 0 | 1 | 0.9721 | 0.9749 | 0.0422 | 12 | 4 | 1 | 10 |
| B7 | feasible | selection | 0.9566 | 1 | 2 | 0.8658 | 0.9607 | 0.0915 | 47 | 16 | 5 | 56 |
| B7 | feasible | calibration | 0.9856 | 1 | 1 | 0.9732 | 0.9774 | 0.0633 | 10 | 4 | 1 | 11 |
| q | feasible | selection | 0.9650 | 0 | 3 | 0.9138 | 0.9555 | 0.0106 | 41 | 21 | 4 | 21 |
| q | feasible | calibration | 0.9850 | 0 | 2 | 0.9643 | 0.9774 | 0.0127 | 18 | 16 | 1 | 9 |

## Paired differences (bootstrap 95% intervals; streams reuse recordings)

| comparison | half | d multi segment F1@50 | d native macro-F1 | d multi boundary recall | units fixed/new | recordings fixed/new | natives fixed/new | consistent |
|---|---|---|---|---|---|---|---|---|
| q - B7 | selection | +0.0479 [+0.0238, +0.0756] | +0.0084 [-0.0327, +0.0534] | -0.0052 [-0.0216, +0.0106] | 18/12 | 11/14 | 3/1 | True |
| q - B7 | calibration | -0.0089 [-0.0241, +0.0057] | -0.0006 [-0.0253, +0.0218] | +0.0000 [-0.0173, +0.0167] | 4/12 | 6/7 | 1/1 | True |
| q - E0 frozen | selection | -0.0206 [-0.0430, -0.0003] | +0.0000 [+0.0000, +0.0000] | -0.0052 [-0.0226, +0.0082] | 10/12 | 8/11 | 0/0 | True |
| q - E0 frozen | calibration | +0.0303 [+0.0138, +0.0482] | +0.0190 [+0.0000, +0.0451] | +0.0226 [+0.0051, +0.0435] | 20/1 | 17/1 | 3/0 | True |
| q - B7 frozen | selection | +0.0339 [+0.0129, +0.0583] | +0.0084 [-0.0327, +0.0534] | +0.0000 [-0.0168, +0.0178] | 17/10 | 10/11 | 3/1 | True |
| q - B7 frozen | calibration | -0.0078 [-0.0240, +0.0086] | -0.0077 [-0.0292, +0.0000] | +0.0025 [-0.0159, +0.0221] | 6/12 | 8/7 | 0/1 | True |
| B7 - B7 frozen | selection | -0.0140 [-0.0293, -0.0001] | +0.0000 [+0.0000, +0.0000] | +0.0052 [+0.0000, +0.0133] | 4/3 | 4/2 | 0/0 | True |
| B7 - B7 frozen | calibration | +0.0010 [-0.0066, +0.0113] | -0.0070 [-0.0248, +0.0000] | +0.0025 [-0.0056, +0.0149] | 3/1 | 3/1 | 0/1 | True |

## Start conditions of task 3

```
{
 "q_feasible": true,
 "margins_vs_references": {
  "selection": -0.02063367132737759,
  "calibration": -0.00782201669820859
 },
 "margins_vs_controls": {
  "B7": {
   "selection": 0.04793396067174682,
   "calibration": -0.008859922337127846
  }
 },
 "infeasible_controls": [
  "E0"
 ],
 "floors": {
  "selection": {
   "boundary_recall": 0.9554973821989529,
   "native_macro_f1": 0.9650312221231043
  },
  "calibration": {
   "boundary_recall": 0.9773869346733668,
   "native_macro_f1": 0.9850094876660341
  }
 },
 "met": {
  "1": false,
  "2": false,
  "3": false,
  "4": true
 },
 "start_training": false,
 "sensitivity_calibration": -0.009573424183551671
}
```
