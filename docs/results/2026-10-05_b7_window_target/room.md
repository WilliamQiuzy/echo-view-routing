# Step 4: room and gate

| decoding | variant | half | segment_f1_50_multi | boundary_recall_multi | native_macro_f1 | native_internal_cuts_per_clip | repaired_windows |
|---|---|---|---|---|---|---|---|
| frozen | original | selection | 0.8798 | 0.9555 | 0.9566 | 0.0634 | 0 |
| frozen | original | calibration | 0.9721 | 0.9749 | 0.9927 | 0.0422 | 0 |
| frozen | R1 | selection | 0.9070 | 0.9974 | 0.9566 | 0.0634 | 218 |
| frozen | R1 | calibration | 0.9857 | 1.0000 | 0.9927 | 0.0422 | 168 |
| frozen | R2 | selection | 1.0000 | 0.9974 | 1.0000 | 0.0000 | 466 |
| frozen | R2 | calibration | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 250 |
| common rule | original | selection | 0.8658 | 0.9607 | 0.9566 | 0.0915 | 0 |
| common rule | original | calibration | 0.9732 | 0.9774 | 0.9856 | 0.0633 | 0 |
| common rule | R1 | selection | 0.8924 | 1.0000 | 0.9566 | 0.0915 | 218 |
| common rule | R1 | calibration | 0.9857 | 1.0000 | 0.9856 | 0.0633 | 168 |
| common rule | R2 | selection | 1.0000 | 0.9974 | 1.0000 | 0.0000 | 466 |
| common rule | R2 | calibration | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 250 |

| half | windows | recordings | sessions |
|---|---|---|---|
| selection | 218 | 88 | 32 |
| calibration | 168 | 86 | 31 |

| pair | windows | recordings | sessions |
|---|---|---|---|
| A4C -> A5C | 28 | 17 | 12 |
| A4C -> PLAX | 38 | 31 | 19 |
| A4C -> PSAX | 14 | 12 | 10 |
| A4C -> SC4C | 10 | 9 | 7 |
| A5C -> A4C | 44 | 4 | 3 |
| A5C -> PLAX | 6 | 4 | 3 |
| A5C -> PSAX | 2 | 2 | 2 |
| A5C -> SC4C | 5 | 3 | 2 |
| PLAX -> A4C | 1 | 1 | 1 |
| PLAX -> A5C | 7 | 7 | 5 |
| PLAX -> PSAX | 4 | 4 | 4 |
| PLAX -> SC4C | 1 | 1 | 1 |
| PSAX -> A5C | 6 | 5 | 5 |
| PSAX -> PLAX | 15 | 12 | 12 |
| PSAX -> SC4C | 1 | 1 | 1 |
| SC4C -> A4C | 3 | 1 | 1 |
| SC4C -> A5C | 7 | 6 | 6 |
| SC4C -> PLAX | 25 | 1 | 1 |
| SC4C -> PSAX | 1 | 1 | 1 |

| pair | windows | recordings | sessions |
|---|---|---|---|
| A4C -> A5C | 21 | 15 | 12 |
| A4C -> PLAX | 32 | 22 | 16 |
| A4C -> PSAX | 15 | 12 | 10 |
| A4C -> SC4C | 16 | 15 | 12 |
| A5C -> A4C | 15 | 5 | 5 |
| A5C -> PLAX | 6 | 4 | 4 |
| A5C -> PSAX | 4 | 4 | 3 |
| A5C -> SC4C | 5 | 4 | 4 |
| PLAX -> A4C | 1 | 1 | 1 |
| PLAX -> A5C | 6 | 5 | 4 |
| PLAX -> PSAX | 2 | 2 | 2 |
| PLAX -> SC4C | 4 | 4 | 4 |
| PSAX -> A4C | 2 | 2 | 2 |
| PSAX -> A5C | 9 | 7 | 6 |
| PSAX -> PLAX | 11 | 11 | 11 |
| PSAX -> SC4C | 11 | 10 | 7 |
| SC4C -> A5C | 8 | 4 | 4 |

```
{
 "goal": {
  "frozen": {
   "segment_part": {
    "selection": false,
    "calibration": true
   },
   "floors_part": {
    "selection": false,
    "calibration": true
   },
   "met": false
  },
  "common rule": {
   "segment_part": {
    "selection": false,
    "calibration": true
   },
   "floors_part": {
    "selection": false,
    "calibration": true
   },
   "met": false
  }
 },
 "gate_met": false
}
```
