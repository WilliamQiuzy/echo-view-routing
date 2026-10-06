# Target-time supervision: tables

## Models

| model | row | half | B3 status | segment F1@50 | boundary recall | boundary precision | native macro-F1 | cuts per clip | A4C read as A5C | A5C read as A4C | SC4C read as PLAX |
|---|---|---|---|---|---|---|---|---|---|---|---|
| N_s0 | b3 | selection | infeasible (diagnostic setting shown) | 0.8186 | 0.8848 | 0.8264 | 0.9304 | 0.0704 | 1 | 4 | 1 |
| N_s0 | b3 | calibration | infeasible (diagnostic setting shown) | 0.9407 | 0.9397 | 0.9167 | 0.9753 | 0.0549 | 1 | 2 | 0 |
| N_s0 | argmax | selection |  | 0.8073 | 0.8927 | 0.8100 | 0.9304 | 0.0775 | 1 | 4 | 1 |
| N_s0 | argmax | calibration |  | 0.9360 | 0.9372 | 0.9120 | 0.9753 | 0.0633 | 1 | 2 | 0 |
| S_s0 | b3 | selection | infeasible (diagnostic setting shown) | 0.8378 | 0.9267 | 0.8551 | 0.9218 | 0.0669 | 2 | 4 | 1 |
| S_s0 | b3 | calibration | infeasible (diagnostic setting shown) | 0.9490 | 0.9548 | 0.9476 | 0.9780 | 0.0675 | 1 | 2 | 0 |
| S_s0 | argmax | selection |  | 0.8321 | 0.9476 | 0.8498 | 0.9218 | 0.0810 | 2 | 4 | 1 |
| S_s0 | argmax | calibration |  | 0.9714 | 0.9899 | 0.9610 | 0.9780 | 0.0675 | 1 | 2 | 0 |
| C_s0 | b3 | selection | infeasible (diagnostic setting shown) | 0.8541 | 0.9660 | 0.8642 | 0.9218 | 0.0669 | 2 | 4 | 1 |
| C_s0 | b3 | calibration | infeasible (diagnostic setting shown) | 0.9647 | 0.9899 | 0.9586 | 0.9753 | 0.0717 | 1 | 2 | 0 |
| C_s0 | argmax | selection |  | 0.8474 | 0.9660 | 0.8522 | 0.9353 | 0.0810 | 2 | 3 | 1 |
| C_s0 | argmax | calibration |  | 0.9647 | 0.9899 | 0.9586 | 0.9753 | 0.1055 | 1 | 2 | 0 |
| N_s1 | b3 | selection | feasible | 0.8200 | 0.9084 | 0.8321 | 0.9304 | 0.0704 | 1 | 4 | 1 |
| N_s1 | b3 | calibration | feasible | 0.9654 | 0.9573 | 0.9478 | 0.9850 | 0.0549 | 0 | 2 | 0 |
| N_s1 | argmax | selection |  | 0.8115 | 0.9110 | 0.8227 | 0.9304 | 0.0704 | 1 | 4 | 1 |
| N_s1 | argmax | calibration |  | 0.9615 | 0.9573 | 0.9478 | 0.9850 | 0.0549 | 0 | 2 | 0 |
| S_s1 | b3 | selection | feasible | 0.8352 | 0.9267 | 0.8389 | 0.9000 | 0.0599 | 5 | 4 | 1 |
| S_s1 | b3 | calibration | feasible | 0.9588 | 0.9698 | 0.9531 | 0.9856 | 0.0844 | 1 | 1 | 0 |
| S_s1 | argmax | selection |  | 0.8254 | 0.9660 | 0.8218 | 0.9000 | 0.0669 | 5 | 4 | 1 |
| S_s1 | argmax | calibration |  | 0.9742 | 0.9874 | 0.9609 | 0.9780 | 0.0844 | 1 | 2 | 0 |
| C_s1 | b3 | selection | feasible | 0.8566 | 0.9686 | 0.8585 | 0.9477 | 0.0915 | 2 | 2 | 1 |
| C_s1 | b3 | calibration | feasible | 0.9743 | 0.9925 | 0.9564 | 0.9850 | 0.0802 | 0 | 2 | 0 |
| C_s1 | argmax | selection |  | 0.8462 | 0.9686 | 0.8333 | 0.9477 | 0.1162 | 2 | 2 | 1 |
| C_s1 | argmax | calibration |  | 0.9743 | 0.9925 | 0.9564 | 0.9780 | 0.0970 | 1 | 2 | 0 |
| N_s2 | b3 | selection | infeasible (diagnostic setting shown) | 0.7888 | 0.7984 | 0.7439 | 0.9140 | 0.0986 | 3 | 4 | 1 |
| N_s2 | b3 | calibration | infeasible (diagnostic setting shown) | 0.9135 | 0.8141 | 0.8060 | 0.9753 | 0.0633 | 1 | 2 | 0 |
| N_s2 | argmax | selection |  | 0.7633 | 0.8010 | 0.7051 | 0.9140 | 0.1232 | 3 | 4 | 1 |
| N_s2 | argmax | calibration |  | 0.9025 | 0.8116 | 0.7917 | 0.9753 | 0.0717 | 1 | 2 | 0 |
| S_s2 | b3 | selection | feasible | 0.8243 | 0.9110 | 0.8406 | 0.9067 | 0.0951 | 4 | 4 | 1 |
| S_s2 | b3 | calibration | feasible | 0.9569 | 0.9698 | 0.9531 | 0.9850 | 0.0886 | 0 | 2 | 0 |
| S_s2 | argmax | selection |  | 0.8262 | 0.9450 | 0.8376 | 0.9067 | 0.1303 | 4 | 4 | 1 |
| S_s2 | argmax | calibration |  | 0.9695 | 0.9824 | 0.9537 | 0.9780 | 0.1139 | 1 | 2 | 0 |
| C_s2 | b3 | selection | infeasible (diagnostic setting shown) | 0.8523 | 0.9712 | 0.8281 | 0.9414 | 0.0986 | 2 | 1 | 1 |
| C_s2 | b3 | calibration | infeasible (diagnostic setting shown) | 0.9695 | 0.9925 | 0.9587 | 0.9753 | 0.0844 | 1 | 2 | 0 |
| C_s2 | argmax | selection |  | 0.8363 | 0.9712 | 0.8083 | 0.9414 | 0.1056 | 2 | 1 | 1 |
| C_s2 | argmax | calibration |  | 0.9630 | 0.9899 | 0.9448 | 0.9753 | 0.1055 | 1 | 2 | 0 |

## References

| system | half | segment F1@50 | boundary recall | native macro-F1 | A4C read as A5C | A5C read as A4C | SC4C read as PLAX |
|---|---|---|---|---|---|---|---|
| E0 + B3 | selection | 0.9344 | 0.9607 | 0.9650 | 0 | 3 | 0 |
| E0 + B3 | calibration | 0.9340 | 0.9548 | 0.9660 | 0 | 4 | 0 |
| E1 + B3 | selection | 0.8940 | 0.9555 | 0.9780 | 0 | 1 | 1 |
| E1 + B3 | calibration | 0.9589 | 0.9774 | 0.9856 | 1 | 1 | 0 |
| B7 + B3 | selection | 0.8798 | 0.9555 | 0.9566 | 1 | 2 | 1 |
| B7 + B3 | calibration | 0.9721 | 0.9749 | 0.9927 | 0 | 1 | 0 |
| E1 + ASRF s0 | selection | 0.9100 | 0.9424 | 0.9477 | 2 | 2 | 1 |
| E1 + ASRF s0 | calibration | 0.9758 | 0.9774 | 0.9929 | 1 | 0 | 0 |
| E1 + ASRF s1 | selection | 0.9089 | 0.9503 | 0.9664 | 0 | 2 | 1 |
| E1 + ASRF s1 | calibration | 0.9643 | 0.9724 | 0.9753 | 1 | 2 | 0 |
| E1 + ASRF s2 | selection | 0.9071 | 0.9529 | 0.9664 | 0 | 2 | 1 |
| E1 + ASRF s2 | calibration | 0.9644 | 0.9774 | 0.9753 | 1 | 2 | 0 |

## C - S (paired bootstrap; streams reuse recordings)

| comparison | d multi segment F1@50 | d native macro-F1 | d multi boundary recall |
|---|---|---|---|
| s0|b3|selection | +0.0163 [-0.0006, +0.0339] | +0.0000 [+0.0000, +0.0000] | +0.0393 [+0.0206, +0.0599] |
| s0|b3|calibration | +0.0157 [-0.0024, +0.0362] | -0.0027 [-0.0092, +0.0000] | +0.0352 [+0.0171, +0.0564] |
| s0|argmax|selection | +0.0153 [+0.0009, +0.0321] | +0.0135 [+0.0000, +0.0449] | +0.0183 [+0.0052, +0.0349] |
| s0|argmax|calibration | -0.0066 [-0.0165, +0.0019] | -0.0027 [-0.0092, +0.0000] | +0.0000 [-0.0074, +0.0076] |
| s1|b3|selection | +0.0214 [+0.0022, +0.0422] | +0.0477 [-0.0159, +0.1104] | +0.0419 [+0.0205, +0.0682] |
| s1|b3|calibration | +0.0155 [+0.0021, +0.0303] | -0.0006 [-0.0253, +0.0218] | +0.0226 [+0.0099, +0.0387] |
| s1|argmax|selection | +0.0208 [+0.0047, +0.0373] | +0.0477 [-0.0159, +0.1104] | +0.0026 [-0.0054, +0.0110] |
| s1|argmax|calibration | +0.0001 [-0.0092, +0.0099] | +0.0000 [+0.0000, +0.0000] | +0.0050 [+0.0000, +0.0129] |
| s2|b3|selection | +0.0280 [+0.0037, +0.0587] | +0.0347 [-0.0114, +0.0912] | +0.0602 [+0.0342, +0.0914] |
| s2|b3|calibration | +0.0127 [-0.0016, +0.0281] | -0.0097 [-0.0286, +0.0000] | +0.0226 [+0.0050, +0.0431] |
| s2|argmax|selection | +0.0101 [-0.0099, +0.0323] | +0.0347 [-0.0114, +0.0912] | +0.0262 [+0.0084, +0.0471] |
| s2|argmax|calibration | -0.0064 [-0.0198, +0.0048] | -0.0027 [-0.0092, +0.0000] | +0.0075 [-0.0026, +0.0188] |

## Goal

```
{
 "N + B3": {
  "mean_segment_f1": {
   "selection": 0.8091338785372946,
   "calibration": 0.9398575770946708
  },
  "margin": {
   "selection": -0.12527580864736743,
   "calibration": -0.03228459389476246
  },
  "per_seed": [
   {
    "selection": {
     "feasible": false,
     "recall": 0.8848167539267016,
     "native": 0.930354087619147
    },
    "calibration": {
     "feasible": false,
     "recall": 0.9396984924623115,
     "native": 0.9753184183354981
    }
   },
   {
    "selection": {
     "feasible": true,
     "recall": 0.9083769633507853,
     "native": 0.930354087619147
    },
    "calibration": {
     "feasible": true,
     "recall": 0.957286432160804,
     "native": 0.9850094876660341
    }
   },
   {
    "selection": {
     "feasible": false,
     "recall": 0.7984293193717278,
     "native": 0.9139687174139729
    },
    "calibration": {
     "feasible": false,
     "recall": 0.8140703517587939,
     "native": 0.9753184183354981
    }
   }
  ],
  "met": false,
  "sensitivity_calibration": -0.03403600138010554
 },
 "S + B3": {
  "mean_segment_f1": {
   "selection": 0.8324704014359187,
   "calibration": 0.9548724696873085
  },
  "margin": {
   "selection": -0.10193928574874334,
   "calibration": -0.017269701302124685
  },
  "per_seed": [
   {
    "selection": {
     "feasible": false,
     "recall": 0.9267015706806283,
     "native": 0.9218172022624577
    },
    "calibration": {
     "feasible": false,
     "recall": 0.9547738693467337,
     "native": 0.977979094076655
    }
   },
   {
    "selection": {
     "feasible": true,
     "recall": 0.9267015706806283,
     "native": 0.8999698902379656
    },
    "calibration": {
     "feasible": true,
     "recall": 0.9698492462311558,
     "native": 0.9856102003642988
    }
   },
   {
    "selection": {
     "feasible": true,
     "recall": 0.9109947643979057,
     "native": 0.9067127090545164
    },
    "calibration": {
     "feasible": true,
     "recall": 0.9698492462311558,
     "native": 0.9850094876660341
    }
   }
  ],
  "met": false,
  "sensitivity_calibration": -0.019021108787467766
 },
 "C + B3": {
  "mean_segment_f1": {
   "selection": 0.8543611483380061,
   "calibration": 0.9695207676609202
  },
  "margin": {
   "selection": -0.08004853884665586,
   "calibration": -0.0026214033285130167
  },
  "per_seed": [
   {
    "selection": {
     "feasible": false,
     "recall": 0.9659685863874345,
     "native": 0.9218172022624577
    },
    "calibration": {
     "feasible": false,
     "recall": 0.9899497487437185,
     "native": 0.9753184183354981
    }
   },
   {
    "selection": {
     "feasible": true,
     "recall": 0.9685863874345549,
     "native": 0.947674046567578
    },
    "calibration": {
     "feasible": true,
     "recall": 0.992462311557789,
     "native": 0.9850094876660341
    }
   },
   {
    "selection": {
     "feasible": false,
     "recall": 0.9712041884816754,
     "native": 0.9414189242399024
    },
    "calibration": {
     "feasible": false,
     "recall": 0.992462311557789,
     "native": 0.9753184183354981
    }
   }
  ],
  "met": false,
  "sensitivity_calibration": -0.004372810813856098
 }
}
```
