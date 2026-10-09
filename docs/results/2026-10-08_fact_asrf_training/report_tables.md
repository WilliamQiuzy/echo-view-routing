# Paired FACT-ASRF round: tables

Recompute: `python scripts/report_fact_asrf.py --config configs/experiments/fact_asrf_paired_v1.yaml --evaluate`

## Systems

| system | half | segment F1@50 | boundary P | boundary R | native macro-F1 | cuts per clip | same-class cut rate | A5C read as A4C | A4C read as A5C | frame-branch segment F1@50 |
|---|---|---|---|---|---|---|---|---|---|---|
| FACT_s0 | selection | 0.9062 | 0.9258 | 0.9476 | 0.9886 | 0.0000 | 0.0522 | 0 | 0 | 0.9002 |
| FACT_s0 | calibration | 0.9674 | 0.9678 | 0.9824 | 0.9856 | 0.0338 | 0.0249 | 1 | 1 | 0.9618 |
| ASRF_s0 | selection | 0.8779 | 0.9273 | 0.9346 | 0.9304 | 0.0106 | 0.0652 | 4 | 1 | 0.8566 |
| ASRF_s0 | calibration | 0.9721 | 0.9752 | 0.9874 | 0.9856 | 0.0042 | 0.0348 | 1 | 1 | 0.9565 |
| FACT_s1 | selection | 0.8983 | 0.9284 | 0.9503 | 0.9780 | 0.0000 | 0.0435 | 1 | 0 | 0.9046 |
| FACT_s1 | calibration | 0.9605 | 0.9676 | 0.9749 | 0.9856 | 0.0295 | 0.0299 | 1 | 1 | 0.9608 |
| ASRF_s1 | selection | 0.8876 | 0.9373 | 0.9398 | 0.9537 | 0.0070 | 0.0478 | 3 | 0 | 0.8688 |
| ASRF_s1 | calibration | 0.9721 | 0.9776 | 0.9849 | 0.9856 | 0.0042 | 0.0348 | 1 | 1 | 0.9658 |
| FACT_s2 | selection | 0.9109 | 0.9253 | 0.9398 | 0.9664 | 0.0000 | 0.0348 | 2 | 0 | 0.9014 |
| FACT_s2 | calibration | 0.9653 | 0.9724 | 0.9724 | 0.9929 | 0.0338 | 0.0249 | 0 | 1 | 0.9674 |
| ASRF_s2 | selection | 0.8825 | 0.9424 | 0.9424 | 0.9537 | 0.0106 | 0.0652 | 3 | 0 | 0.8733 |
| ASRF_s2 | calibration | 0.9662 | 0.9799 | 0.9824 | 0.9856 | 0.0042 | 0.0299 | 1 | 1 | 0.9545 |
| E1 + B3 | selection | 0.8940 | 0.9432 | 0.9555 | 0.9780 | 0.0458 | 0.0391 | 1 | 0 | None |
| E1 + B3 | calibration | 0.9589 | 0.9511 | 0.9774 | 0.9856 | 0.0675 | 0.0199 | 1 | 1 | None |
| E0 + B3 | selection | 0.9344 | 0.9946 | 0.9607 | 0.9650 | 0.0141 | 0.0043 | 3 | 0 | None |
| E0 + B3 | calibration | 0.9340 | 0.9694 | 0.9548 | 0.9660 | 0.0338 | 0.0199 | 4 | 0 | None |
| B7 + B3 | selection | 0.8798 | 0.8902 | 0.9555 | 0.9566 | 0.0634 | 0.0565 | 2 | 1 | None |
| B7 + B3 | calibration | 0.9721 | 0.9628 | 0.9749 | 0.9927 | 0.0422 | 0.0348 | 1 | 0 | None |

## Comparisons

| comparison | d segment F1@50 | native fixed/new |
|---|---|---|
| s0|FACT - ASRF|selection | +0.0284 [+0.0061, +0.0528] | 5/0 |
| s0|FACT - ASRF|calibration | -0.0048 [-0.0214, +0.0107] | 0/0 |
| s1|FACT - ASRF|selection | +0.0108 [-0.0136, +0.0377] | 2/0 |
| s1|FACT - ASRF|calibration | -0.0115 [-0.0289, +0.0039] | 0/0 |
| s2|FACT - ASRF|selection | +0.0284 [+0.0040, +0.0536] | 1/0 |
| s2|FACT - ASRF|calibration | -0.0010 [-0.0131, +0.0107] | 1/0 |
| s0|FACT - (E1 + B3)|selection | +0.0123 [-0.0067, +0.0334] | 1/0 |
| s0|FACT - (E1 + B3)|calibration | +0.0084 [-0.0089, +0.0244] | 0/0 |
| s0|FACT - (E0 + B3)|selection | -0.0282 [-0.0546, +0.0000] | 3/2 |
| s0|FACT - (E0 + B3)|calibration | +0.0334 [+0.0073, +0.0587] | 4/1 |
| s0|FACT - (B7 + B3)|selection | +0.0264 [+0.0058, +0.0483] | 3/0 |
| s0|FACT - (B7 + B3)|calibration | -0.0048 [-0.0234, +0.0138] | 0/1 |
| s1|FACT - (E1 + B3)|selection | +0.0044 [-0.0176, +0.0271] | 0/0 |
| s1|FACT - (E1 + B3)|calibration | +0.0016 [-0.0150, +0.0174] | 0/0 |
| s1|FACT - (E0 + B3)|selection | -0.0361 [-0.0639, -0.0078] | 2/2 |
| s1|FACT - (E0 + B3)|calibration | +0.0266 [+0.0026, +0.0514] | 4/1 |
| s1|FACT - (B7 + B3)|selection | +0.0185 [-0.0031, +0.0402] | 2/0 |
| s1|FACT - (B7 + B3)|calibration | -0.0116 [-0.0300, +0.0058] | 0/1 |
| s2|FACT - (E1 + B3)|selection | +0.0169 [-0.0058, +0.0406] | 0/1 |
| s2|FACT - (E1 + B3)|calibration | +0.0063 [-0.0092, +0.0210] | 1/0 |
| s2|FACT - (E0 + B3)|selection | -0.0235 [-0.0524, +0.0056] | 1/2 |
| s2|FACT - (E0 + B3)|calibration | +0.0313 [+0.0074, +0.0574] | 5/1 |
| s2|FACT - (B7 + B3)|selection | +0.0310 [+0.0079, +0.0558] | 1/0 |
| s2|FACT - (B7 + B3)|calibration | -0.0069 [-0.0224, +0.0096] | 1/1 |

## Verdicts

```
{
 "FACT - ASRF": {
  "selection": "no consistent difference detected",
  "calibration": "no consistent difference detected"
 },
 "FACT - (E1 + B3)": {
  "selection": "no consistent difference detected",
  "calibration": "no consistent difference detected"
 },
 "FACT - (E0 + B3)": {
  "selection": "no consistent difference detected",
  "calibration": "consistent positive"
 },
 "FACT - (B7 + B3)": {
  "selection": "no consistent difference detected",
  "calibration": "no consistent difference detected"
 }
}
```

## Decision

```
{
 "primary": {
  "selection": "no consistent difference detected",
  "calibration": "no consistent difference detected"
 },
 "versus_E1_B3": {
  "selection": "no consistent difference detected",
  "calibration": "no consistent difference detected"
 },
 "consistent_over_ASRF_both_halves": false,
 "consistent_over_E1_B3_both_halves": false,
 "constraints_met": false,
 "constraint_failures": [
  {
   "seed": 0,
   "half": "selection",
   "boundary_recall": 0.9476439790575916,
   "native_macro_f1": 0.9886353840806397
  },
  {
   "seed": 2,
   "half": "selection",
   "boundary_recall": 0.9397905759162304,
   "native_macro_f1": 0.9663626568079124
  }
 ],
 "mean_segment_f1_FACT": {
  "selection": 0.9051440188441351,
  "calibration": 0.9643867954993187
 },
 "goal_met": false,
 "goal_margin": {
  "selection": -0.039265668340526916,
  "calibration": -0.017755375490114522
 },
 "calibration_sensitivity_margin": -0.019506782975457604,
 "observation": "FACT - ASRF is not consistent positive in both halves",
 "action": "the comparison of this GTEA configuration ends; each half is reported as positive, negative or uncertain; segment-token methods in general are not rejected and Breakfast is not run automatically"
}
```
