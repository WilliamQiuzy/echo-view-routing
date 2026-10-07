# L1 + ASRF round: tables

## Systems

| system | half | chosen | segment F1@50 | boundary P | boundary R | native macro-F1 | cuts per clip | A4C read as A5C | A5C read as A4C | SC4C read as PLAX |
|---|---|---|---|---|---|---|---|---|---|---|
| A_s0 + ASRF | selection | 6 (feasible) | 0.9058 | 0.9495 | 0.9346 | 0.9477 | 0.0141 | 2 | 2 | 1 |
| A_s0 + ASRF | calibration | 6 (feasible) | 0.9739 | 0.9873 | 0.9799 | 0.9829 | 0.0338 | 1 | 1 | 0 |
| B_s0 + ASRF | selection | 5 (feasible) | 0.9022 | 0.9447 | 0.9398 | 0.9566 | 0.0352 | 1 | 2 | 1 |
| B_s0 + ASRF | calibration | 5 (feasible) | 0.9777 | 0.9797 | 0.9724 | 0.9929 | 0.0253 | 1 | 0 | 0 |
| L1_s0 + B3 | selection |  | 0.9084 | 0.9607 | 0.9607 | 0.9886 | 0.0951 | 0 | 0 | 1 |
| L1_s0 + B3 | calibration |  | 0.9711 | 0.9725 | 0.9774 | 0.9856 | 0.1013 | 1 | 1 | 0 |
| A_s1 + ASRF | selection | 32 (feasible) | 0.8772 | 0.9175 | 0.9319 | 0.9397 | 0.0070 | 0 | 4 | 1 |
| A_s1 + ASRF | calibration | 32 (feasible) | 0.9729 | 0.9848 | 0.9799 | 0.9856 | 0.0042 | 1 | 1 | 0 |
| B_s1 + ASRF | selection | 3 (feasible) | 0.9067 | 0.9378 | 0.9476 | 0.9566 | 0.0211 | 1 | 2 | 1 |
| B_s1 + ASRF | calibration | 3 (feasible) | 0.9778 | 0.9799 | 0.9824 | 0.9929 | 0.0422 | 1 | 0 | 0 |
| L1_s1 + B3 | selection |  | 0.9062 | 0.9702 | 0.9372 | 0.9886 | 0.0070 | 0 | 0 | 1 |
| L1_s1 + B3 | calibration |  | 0.9677 | 0.9948 | 0.9623 | 0.9903 | 0.0211 | 1 | 0 | 0 |
| A_s2 + ASRF | selection | 28 (feasible) | 0.8922 | 0.9229 | 0.9398 | 0.9566 | 0.0141 | 1 | 2 | 1 |
| A_s2 + ASRF | calibration | 28 (feasible) | 0.9778 | 0.9849 | 0.9824 | 0.9929 | 0.0253 | 1 | 0 | 0 |
| B_s2 + ASRF | selection | 11 (feasible) | 0.8963 | 0.9284 | 0.9503 | 0.9566 | 0.0141 | 1 | 2 | 1 |
| B_s2 + ASRF | calibration | 11 (feasible) | 0.9730 | 0.9749 | 0.9749 | 0.9829 | 0.0338 | 1 | 1 | 0 |
| L1_s2 + B3 | selection |  | 0.9124 | 0.9607 | 0.9607 | 0.9886 | 0.0915 | 0 | 0 | 1 |
| L1_s2 + B3 | calibration |  | 0.9693 | 0.9678 | 0.9824 | 0.9835 | 0.1308 | 2 | 0 | 0 |

## Comparisons

| comparison | half | d multi segment F1@50 | d multi boundary recall | d native macro-F1 | native fixed/new |
|---|---|---|---|---|---|
| s0|B - A | selection | -0.0036 [-0.0186, +0.0116] | +0.0052 [-0.0052, +0.0157] | +0.0090 [+0.0000, +0.0285] | 1/0 |
| s0|B - A | calibration | +0.0039 [-0.0068, +0.0141] | -0.0075 [-0.0228, +0.0072] | +0.0100 [+0.0000, +0.0300] | 2/0 |
| s0|B - (L1 + B3) | selection | -0.0062 [-0.0289, +0.0156] | -0.0209 [-0.0439, +0.0000] | -0.0320 [-0.0713, +0.0000] | 0/3 |
| s0|B - (L1 + B3) | calibration | +0.0066 [-0.0077, +0.0210] | -0.0050 [-0.0238, +0.0129] | +0.0073 [+0.0000, +0.0268] | 1/0 |
| s1|B - A | selection | +0.0295 [+0.0047, +0.0539] | +0.0157 [-0.0107, +0.0443] | +0.0169 [-0.0242, +0.0671] | 2/1 |
| s1|B - A | calibration | +0.0049 [-0.0038, +0.0140] | +0.0025 [-0.0101, +0.0155] | +0.0073 [+0.0000, +0.0268] | 1/0 |
| s1|B - (L1 + B3) | selection | +0.0006 [-0.0218, +0.0238] | +0.0105 [-0.0146, +0.0363] | -0.0320 [-0.0713, +0.0000] | 0/3 |
| s1|B - (L1 + B3) | calibration | +0.0101 [-0.0104, +0.0304] | +0.0201 [-0.0025, +0.0424] | +0.0027 [+0.0000, +0.0092] | 1/0 |
| s2|B - A | selection | +0.0042 [-0.0128, +0.0223] | +0.0105 [-0.0100, +0.0331] | +0.0000 [+0.0000, +0.0000] | 0/0 |
| s2|B - A | calibration | -0.0048 [-0.0128, +0.0030] | -0.0075 [-0.0186, +0.0026] | -0.0100 [-0.0300, +0.0000] | 0/2 |
| s2|B - (L1 + B3) | selection | -0.0160 [-0.0371, +0.0044] | -0.0105 [-0.0312, +0.0103] | -0.0320 [-0.0713, +0.0000] | 0/3 |
| s2|B - (L1 + B3) | calibration | +0.0037 [-0.0104, +0.0167] | -0.0075 [-0.0235, +0.0077] | -0.0005 [-0.0218, +0.0225] | 1/1 |

## References

| system | half | segment F1@50 | boundary R | native macro-F1 |
|---|---|---|---|---|
| E0 + B3 | selection | 0.9344 | 0.9607 | 0.9650 |
| E0 + B3 | calibration | 0.9340 | 0.9548 | 0.9660 |
| E1 + B3 | selection | 0.8940 | 0.9555 | 0.9780 |
| E1 + B3 | calibration | 0.9589 | 0.9774 | 0.9856 |
| B7 + B3 | selection | 0.8798 | 0.9555 | 0.9566 |
| B7 + B3 | calibration | 0.9721 | 0.9749 | 0.9927 |
| E1 + ASRF s0 | selection | 0.9100 | 0.9424 | 0.9477 |
| E1 + ASRF s0 | calibration | 0.9758 | 0.9774 | 0.9929 |
| E1 + ASRF s1 | selection | 0.9089 | 0.9503 | 0.9664 |
| E1 + ASRF s1 | calibration | 0.9643 | 0.9724 | 0.9753 |
| E1 + ASRF s2 | selection | 0.9071 | 0.9529 | 0.9664 |
| E1 + ASRF s2 | calibration | 0.9644 | 0.9774 | 0.9753 |

## Goal (B + ASRF)

```
{
 "mean_segment_f1": {
  "selection": 0.9017630410843491,
  "calibration": 0.976176120856817
 },
 "required": {
  "selection": 0.944409687184662,
  "calibration": 0.9821421709894332
 },
 "margin": {
  "selection": -0.04264664610031288,
  "calibration": -0.005966050132616196
 },
 "per_seed": [
  {
   "selection": {
    "feasible": true,
    "recall": 0.9397905759162304,
    "native": 0.9566353840806396
   },
   "calibration": {
    "feasible": true,
    "recall": 0.9723618090452262,
    "native": 0.9929417020326111
   }
  },
  {
   "selection": {
    "feasible": true,
    "recall": 0.9476439790575916,
    "native": 0.9566353840806396
   },
   "calibration": {
    "feasible": true,
    "recall": 0.9824120603015075,
    "native": 0.9929417020326111
   }
  },
  {
   "selection": {
    "feasible": true,
    "recall": 0.9502617801047121,
    "native": 0.9566353840806396
   },
   "calibration": {
    "feasible": true,
    "recall": 0.9748743718592965,
    "native": 0.9829495246231419
   }
  }
 ],
 "met": false,
 "sensitivity_calibration": -0.007717457617959278
}
```
