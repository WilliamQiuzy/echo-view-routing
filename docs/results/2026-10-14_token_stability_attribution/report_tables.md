# Token stability and segmentation attribution: tables

Recompute: `python scripts/token_stability_attribution.py --config configs/experiments/token_stability_attribution_v1.yaml --run`

Reference-conditioned cells (TC, CT) use reference labels to choose a readout per window. They are diagnostics, not deployable systems, attainable scores or upper bounds. Seeds are not extra cases; development halves are reused.

## Actual systems (token at A03's decoder against CLS at A02's decoder): segments

| seed | half | population | token tp | token fp | token fn | token f1 | cls tp | cls fp | cls fn | cls f1 | both | token only | cls only | neither |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0 | selection | native | 280 | 17 | 4 | 0.9639 | 280 | 16 | 4 | 0.9655 | 279 | 1 | 1 | 3 |
| s0 | selection | pairs | 204 | 7 | 3 | 0.9761 | 200 | 9 | 7 | 0.9615 | 200 | 4 | 0 | 3 |
| s0 | selection | multi | 477 | 40 | 25 | 0.9362 | 474 | 30 | 28 | 0.9423 | 469 | 8 | 5 | 20 |
| s1 | selection | native | 276 | 37 | 8 | 0.9246 | 280 | 14 | 4 | 0.9689 | 276 | 0 | 4 | 4 |
| s1 | selection | pairs | 202 | 16 | 5 | 0.9506 | 201 | 11 | 6 | 0.9594 | 201 | 1 | 0 | 5 |
| s1 | selection | multi | 487 | 64 | 15 | 0.9250 | 482 | 30 | 20 | 0.9507 | 474 | 13 | 8 | 7 |
| s2 | selection | native | 279 | 16 | 5 | 0.9637 | 279 | 22 | 5 | 0.9538 | 277 | 2 | 2 | 3 |
| s2 | selection | pairs | 202 | 10 | 5 | 0.9642 | 200 | 12 | 7 | 0.9547 | 200 | 2 | 0 | 5 |
| s2 | selection | multi | 480 | 25 | 22 | 0.9533 | 487 | 24 | 15 | 0.9615 | 478 | 2 | 9 | 13 |
| s0 | calibration | native | 235 | 5 | 2 | 0.9853 | 236 | 5 | 1 | 0.9874 | 235 | 0 | 1 | 1 |
| s0 | calibration | pairs | 170 | 3 | 3 | 0.9827 | 168 | 2 | 5 | 0.9796 | 166 | 4 | 2 | 1 |
| s0 | calibration | multi | 512 | 13 | 6 | 0.9818 | 514 | 11 | 4 | 0.9856 | 512 | 0 | 2 | 4 |
| s1 | calibration | native | 231 | 28 | 6 | 0.9315 | 235 | 5 | 2 | 0.9853 | 231 | 0 | 4 | 2 |
| s1 | calibration | pairs | 170 | 9 | 3 | 0.9659 | 168 | 4 | 5 | 0.9739 | 168 | 2 | 0 | 3 |
| s1 | calibration | multi | 509 | 27 | 9 | 0.9658 | 511 | 14 | 7 | 0.9799 | 507 | 2 | 4 | 5 |
| s2 | calibration | native | 235 | 5 | 2 | 0.9853 | 236 | 5 | 1 | 0.9874 | 235 | 0 | 1 | 1 |
| s2 | calibration | pairs | 169 | 4 | 4 | 0.9769 | 166 | 2 | 7 | 0.9736 | 164 | 5 | 2 | 2 |
| s2 | calibration | multi | 510 | 13 | 8 | 0.9798 | 508 | 11 | 10 | 0.9797 | 507 | 3 | 1 | 7 |

## Actual systems (token at A03's decoder against CLS at A02's decoder): changed segments

| seed | half | population | change | segments | by_class PLAX | by_class PSAX | by_class A4C | by_class A5C | by_class SC4C | short | long | xyx_a4c_a5c | recordings | sessions | few_units |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0 | selection | native | newly_matched | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 1 | True |
| s0 | selection | native | lost | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 1 | 1 | True |
| s0 | selection | native | new_unmatched_pred | 7 | 0 | 0 | 4 | 3 | 0 | 5 | 2 |  | 4 | 4 | True |
| s0 | selection | native | removed_unmatched_pred | 6 | 1 | 1 | 2 | 2 | 0 | 1 | 5 |  | 3 | 3 | True |
| s0 | selection | pairs | newly_matched | 4 | 1 | 1 | 1 | 1 | 0 | 3 | 1 | 0 | 4 | 3 | True |
| s0 | selection | pairs | lost | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | True |
| s0 | selection | pairs | new_unmatched_pred | 2 | 1 | 1 | 0 | 0 | 0 | 1 | 1 |  | 2 | 2 | True |
| s0 | selection | pairs | removed_unmatched_pred | 4 | 1 | 1 | 1 | 1 | 0 | 2 | 2 |  | 3 | 3 | True |
| s0 | selection | multi | newly_matched | 8 | 0 | 0 | 3 | 5 | 0 | 6 | 2 | 3 | 8 | 8 | True |
| s0 | selection | multi | lost | 5 | 0 | 0 | 2 | 3 | 0 | 0 | 5 | 0 | 6 | 5 | True |
| s0 | selection | multi | new_unmatched_pred | 26 | 0 | 1 | 15 | 10 | 0 | 17 | 9 |  | 20 | 14 | False |
| s0 | selection | multi | removed_unmatched_pred | 16 | 0 | 0 | 10 | 6 | 0 | 2 | 14 |  | 23 | 13 | False |
| s1 | selection | native | newly_matched | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | True |
| s1 | selection | native | lost | 4 | 0 | 0 | 2 | 2 | 0 | 1 | 3 | 0 | 4 | 4 | True |
| s1 | selection | native | new_unmatched_pred | 28 | 1 | 2 | 12 | 13 | 0 | 22 | 6 |  | 12 | 8 | True |
| s1 | selection | native | removed_unmatched_pred | 5 | 0 | 1 | 2 | 2 | 0 | 1 | 4 |  | 4 | 3 | True |
| s1 | selection | pairs | newly_matched | 1 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | True |
| s1 | selection | pairs | lost | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | True |
| s1 | selection | pairs | new_unmatched_pred | 10 | 1 | 1 | 4 | 4 | 0 | 9 | 1 |  | 7 | 7 | True |
| s1 | selection | pairs | removed_unmatched_pred | 5 | 0 | 1 | 1 | 3 | 0 | 3 | 2 |  | 4 | 4 | True |
| s1 | selection | multi | newly_matched | 13 | 0 | 0 | 6 | 7 | 0 | 9 | 4 | 4 | 11 | 9 | True |
| s1 | selection | multi | lost | 8 | 0 | 0 | 0 | 8 | 0 | 0 | 8 | 0 | 6 | 5 | True |
| s1 | selection | multi | new_unmatched_pred | 51 | 0 | 2 | 23 | 26 | 0 | 41 | 10 |  | 24 | 15 | False |
| s1 | selection | multi | removed_unmatched_pred | 17 | 0 | 0 | 11 | 6 | 0 | 9 | 8 |  | 23 | 14 | False |
| s2 | selection | native | newly_matched | 2 | 0 | 0 | 2 | 0 | 0 | 0 | 2 | 0 | 2 | 1 | True |
| s2 | selection | native | lost | 2 | 0 | 0 | 2 | 0 | 0 | 0 | 2 | 0 | 2 | 2 | True |
| s2 | selection | native | new_unmatched_pred | 6 | 0 | 0 | 3 | 2 | 1 | 2 | 4 |  | 4 | 4 | True |
| s2 | selection | native | removed_unmatched_pred | 12 | 2 | 2 | 5 | 3 | 0 | 8 | 4 |  | 4 | 3 | True |
| s2 | selection | pairs | newly_matched | 2 | 1 | 0 | 1 | 0 | 0 | 2 | 0 | 0 | 2 | 2 | True |
| s2 | selection | pairs | lost | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | True |
| s2 | selection | pairs | new_unmatched_pred | 5 | 1 | 1 | 2 | 1 | 0 | 4 | 1 |  | 4 | 4 | True |
| s2 | selection | pairs | removed_unmatched_pred | 7 | 1 | 1 | 2 | 3 | 0 | 4 | 3 |  | 6 | 5 | True |
| s2 | selection | multi | newly_matched | 2 | 0 | 0 | 0 | 2 | 0 | 1 | 1 | 0 | 3 | 3 | True |
| s2 | selection | multi | lost | 9 | 1 | 0 | 4 | 4 | 0 | 4 | 5 | 3 | 10 | 8 | True |
| s2 | selection | multi | new_unmatched_pred | 16 | 0 | 1 | 11 | 4 | 0 | 15 | 1 |  | 16 | 11 | False |
| s2 | selection | multi | removed_unmatched_pred | 15 | 0 | 0 | 7 | 8 | 0 | 7 | 8 |  | 14 | 10 | False |
| s0 | calibration | native | newly_matched | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | True |
| s0 | calibration | native | lost | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 1 | True |
| s0 | calibration | native | new_unmatched_pred | 2 | 0 | 0 | 1 | 1 | 0 | 0 | 2 |  | 1 | 1 | True |
| s0 | calibration | native | removed_unmatched_pred | 2 | 0 | 0 | 2 | 0 | 0 | 0 | 2 |  | 2 | 2 | True |
| s0 | calibration | pairs | newly_matched | 4 | 1 | 0 | 2 | 1 | 0 | 4 | 0 | 0 | 4 | 4 | True |
| s0 | calibration | pairs | lost | 2 | 0 | 0 | 1 | 1 | 0 | 0 | 2 | 0 | 2 | 2 | True |
| s0 | calibration | pairs | new_unmatched_pred | 2 | 0 | 0 | 1 | 1 | 0 | 1 | 1 |  | 3 | 3 | True |
| s0 | calibration | pairs | removed_unmatched_pred | 1 | 0 | 0 | 0 | 1 | 0 | 1 | 0 |  | 1 | 1 | True |
| s0 | calibration | multi | newly_matched | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | True |
| s0 | calibration | multi | lost | 2 | 0 | 0 | 1 | 1 | 0 | 1 | 1 | 0 | 2 | 2 | True |
| s0 | calibration | multi | new_unmatched_pred | 7 | 0 | 0 | 3 | 4 | 0 | 4 | 3 |  | 11 | 10 | False |
| s0 | calibration | multi | removed_unmatched_pred | 5 | 0 | 0 | 2 | 3 | 0 | 2 | 3 |  | 7 | 7 | True |
| s1 | calibration | native | newly_matched | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | True |
| s1 | calibration | native | lost | 4 | 0 | 0 | 2 | 2 | 0 | 0 | 4 | 0 | 4 | 3 | True |
| s1 | calibration | native | new_unmatched_pred | 25 | 0 | 0 | 12 | 12 | 1 | 17 | 8 |  | 9 | 6 | True |
| s1 | calibration | native | removed_unmatched_pred | 2 | 0 | 0 | 1 | 1 | 0 | 0 | 2 |  | 2 | 2 | True |
| s1 | calibration | pairs | newly_matched | 2 | 1 | 0 | 0 | 1 | 0 | 2 | 0 | 0 | 2 | 2 | True |
| s1 | calibration | pairs | lost | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | True |
| s1 | calibration | pairs | new_unmatched_pred | 7 | 0 | 0 | 3 | 4 | 0 | 6 | 1 |  | 5 | 5 | True |
| s1 | calibration | pairs | removed_unmatched_pred | 2 | 0 | 0 | 0 | 2 | 0 | 1 | 1 |  | 3 | 3 | True |
| s1 | calibration | multi | newly_matched | 2 | 1 | 0 | 0 | 1 | 0 | 2 | 0 | 0 | 2 | 2 | True |
| s1 | calibration | multi | lost | 4 | 0 | 0 | 2 | 2 | 0 | 1 | 3 | 0 | 5 | 5 | True |
| s1 | calibration | multi | new_unmatched_pred | 20 | 0 | 0 | 11 | 9 | 0 | 14 | 6 |  | 16 | 15 | False |
| s1 | calibration | multi | removed_unmatched_pred | 7 | 0 | 0 | 3 | 4 | 0 | 4 | 3 |  | 9 | 9 | True |
| s2 | calibration | native | newly_matched | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | True |
| s2 | calibration | native | lost | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 1 | True |
| s2 | calibration | native | new_unmatched_pred | 3 | 0 | 0 | 1 | 2 | 0 | 1 | 2 |  | 2 | 2 | True |
| s2 | calibration | native | removed_unmatched_pred | 3 | 0 | 0 | 2 | 1 | 0 | 1 | 2 |  | 2 | 2 | True |
| s2 | calibration | pairs | newly_matched | 5 | 1 | 0 | 2 | 2 | 0 | 5 | 0 | 0 | 5 | 5 | True |
| s2 | calibration | pairs | lost | 2 | 0 | 0 | 1 | 1 | 0 | 0 | 2 | 0 | 2 | 2 | True |
| s2 | calibration | pairs | new_unmatched_pred | 2 | 0 | 0 | 1 | 1 | 0 | 1 | 1 |  | 3 | 3 | True |
| s2 | calibration | pairs | removed_unmatched_pred | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  | 0 | 0 | True |
| s2 | calibration | multi | newly_matched | 3 | 3 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 3 | 3 | True |
| s2 | calibration | multi | lost | 1 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | True |
| s2 | calibration | multi | new_unmatched_pred | 8 | 0 | 0 | 4 | 4 | 0 | 6 | 2 |  | 11 | 11 | False |
| s2 | calibration | multi | removed_unmatched_pred | 6 | 0 | 0 | 2 | 4 | 0 | 4 | 2 |  | 11 | 8 | True |

## Actual systems (token at A03's decoder against CLS at A02's decoder): changed requested-window correctness by distance

| seed | half | population | windows | distance | fixed windows | fixed recordings | fixed sessions | fixed few_units | new windows | new recordings | new sessions | new few_units |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0 | selection | native | 3444 | none | 8 | 4 | 3 | True | 11 | 9 | 8 | True |
| s0 | selection | pairs | 166 | 0-5 | 4 | 4 | 3 | True | 1 | 1 | 1 | True |
| s0 | selection | pairs | 79 | 6-10 | 2 | 2 | 2 | True | 0 | 0 | 0 | True |
| s0 | selection | pairs | 24 | >10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s0 | selection | pairs | 62 | 6-10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s0 | selection | pairs | 259 | >10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s0 | selection | pairs | 600 | none | 2 | 1 | 1 | True | 4 | 4 | 4 | True |
| s0 | selection | multi | 916 | 0-5 | 43 | 29 | 15 | False | 6 | 6 | 5 | True |
| s0 | selection | multi | 242 | 6-10 | 1 | 1 | 1 | True | 1 | 1 | 1 | True |
| s0 | selection | multi | 22 | >10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s0 | selection | multi | 373 | 6-10 | 0 | 0 | 0 | True | 3 | 1 | 1 | True |
| s0 | selection | multi | 1460 | >10 | 2 | 1 | 1 | True | 12 | 5 | 4 | True |
| s0 | selection | multi | 19 | none | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s1 | selection | native | 3444 | none | 13 | 11 | 7 | True | 10 | 8 | 6 | True |
| s1 | selection | pairs | 166 | 0-5 | 4 | 4 | 4 | True | 0 | 0 | 0 | True |
| s1 | selection | pairs | 79 | 6-10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s1 | selection | pairs | 24 | >10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s1 | selection | pairs | 62 | 6-10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s1 | selection | pairs | 259 | >10 | 1 | 1 | 1 | True | 0 | 0 | 0 | True |
| s1 | selection | pairs | 600 | none | 2 | 1 | 1 | True | 4 | 4 | 3 | True |
| s1 | selection | multi | 916 | 0-5 | 38 | 26 | 13 | False | 8 | 8 | 7 | True |
| s1 | selection | multi | 242 | 6-10 | 2 | 2 | 2 | True | 0 | 0 | 0 | True |
| s1 | selection | multi | 22 | >10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s1 | selection | multi | 373 | 6-10 | 1 | 1 | 1 | True | 6 | 3 | 2 | True |
| s1 | selection | multi | 1460 | >10 | 9 | 7 | 6 | True | 16 | 4 | 4 | True |
| s1 | selection | multi | 19 | none | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s2 | selection | native | 3444 | none | 9 | 7 | 5 | True | 14 | 7 | 6 | True |
| s2 | selection | pairs | 166 | 0-5 | 4 | 4 | 3 | True | 2 | 2 | 2 | True |
| s2 | selection | pairs | 79 | 6-10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s2 | selection | pairs | 24 | >10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s2 | selection | pairs | 62 | 6-10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s2 | selection | pairs | 259 | >10 | 1 | 1 | 1 | True | 1 | 1 | 1 | True |
| s2 | selection | pairs | 600 | none | 3 | 1 | 1 | True | 5 | 4 | 4 | True |
| s2 | selection | multi | 916 | 0-5 | 38 | 33 | 17 | False | 11 | 9 | 8 | True |
| s2 | selection | multi | 242 | 6-10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s2 | selection | multi | 22 | >10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s2 | selection | multi | 373 | 6-10 | 1 | 1 | 1 | True | 1 | 1 | 1 | True |
| s2 | selection | multi | 1460 | >10 | 6 | 3 | 2 | True | 13 | 5 | 5 | True |
| s2 | selection | multi | 19 | none | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s0 | calibration | native | 2950 | none | 5 | 1 | 1 | True | 11 | 9 | 8 | True |
| s0 | calibration | pairs | 135 | 0-5 | 2 | 2 | 2 | True | 1 | 1 | 1 | True |
| s0 | calibration | pairs | 65 | 6-10 | 1 | 1 | 1 | True | 1 | 1 | 1 | True |
| s0 | calibration | pairs | 17 | >10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s0 | calibration | pairs | 54 | 6-10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s0 | calibration | pairs | 273 | >10 | 0 | 0 | 0 | True | 1 | 1 | 1 | True |
| s0 | calibration | pairs | 479 | none | 1 | 1 | 1 | True | 2 | 1 | 1 | True |
| s0 | calibration | multi | 945 | 0-5 | 31 | 28 | 17 | False | 8 | 7 | 6 | True |
| s0 | calibration | multi | 257 | 6-10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s0 | calibration | multi | 21 | >10 | 0 | 0 | 0 | True | 1 | 1 | 1 | True |
| s0 | calibration | multi | 391 | 6-10 | 0 | 0 | 0 | True | 5 | 3 | 3 | True |
| s0 | calibration | multi | 1434 | >10 | 0 | 0 | 0 | True | 7 | 4 | 4 | True |
| s1 | calibration | native | 2950 | none | 5 | 4 | 4 | True | 21 | 8 | 7 | True |
| s1 | calibration | pairs | 135 | 0-5 | 4 | 4 | 4 | True | 1 | 1 | 1 | True |
| s1 | calibration | pairs | 65 | 6-10 | 1 | 1 | 1 | True | 0 | 0 | 0 | True |
| s1 | calibration | pairs | 17 | >10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s1 | calibration | pairs | 54 | 6-10 | 0 | 0 | 0 | True | 1 | 1 | 1 | True |
| s1 | calibration | pairs | 273 | >10 | 0 | 0 | 0 | True | 1 | 1 | 1 | True |
| s1 | calibration | pairs | 479 | none | 1 | 1 | 1 | True | 2 | 1 | 1 | True |
| s1 | calibration | multi | 945 | 0-5 | 28 | 27 | 16 | False | 8 | 8 | 7 | True |
| s1 | calibration | multi | 257 | 6-10 | 0 | 0 | 0 | True | 2 | 2 | 2 | True |
| s1 | calibration | multi | 21 | >10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s1 | calibration | multi | 391 | 6-10 | 0 | 0 | 0 | True | 6 | 5 | 4 | True |
| s1 | calibration | multi | 1434 | >10 | 1 | 1 | 1 | True | 7 | 2 | 2 | True |
| s2 | calibration | native | 2950 | none | 2 | 1 | 1 | True | 15 | 8 | 8 | True |
| s2 | calibration | pairs | 135 | 0-5 | 6 | 6 | 6 | True | 0 | 0 | 0 | True |
| s2 | calibration | pairs | 65 | 6-10 | 2 | 2 | 2 | True | 1 | 1 | 1 | True |
| s2 | calibration | pairs | 17 | >10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s2 | calibration | pairs | 54 | 6-10 | 0 | 0 | 0 | True | 1 | 1 | 1 | True |
| s2 | calibration | pairs | 273 | >10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s2 | calibration | pairs | 479 | none | 0 | 0 | 0 | True | 2 | 2 | 2 | True |
| s2 | calibration | multi | 945 | 0-5 | 37 | 35 | 19 | False | 9 | 8 | 8 | True |
| s2 | calibration | multi | 257 | 6-10 | 2 | 2 | 2 | True | 0 | 0 | 0 | True |
| s2 | calibration | multi | 21 | >10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s2 | calibration | multi | 391 | 6-10 | 0 | 0 | 0 | True | 2 | 1 | 1 | True |
| s2 | calibration | multi | 1434 | >10 | 1 | 1 | 1 | True | 5 | 4 | 4 | True |

## Actual systems (token at A03's decoder against CLS at A02's decoder): reference segments holding changed requests

| seed | half | population | windows | fixed windows_by_status both | fixed windows_by_status token only | fixed windows_by_status cls only | fixed windows_by_status neither | fixed segments_by_status both | fixed segments_by_status token only | fixed segments_by_status cls only | fixed segments_by_status neither | fixed sessions_by_status both | fixed sessions_by_status token only | fixed sessions_by_status cls only | fixed sessions_by_status neither | fixed sessions | new windows_by_status both | new windows_by_status token only | new windows_by_status cls only | new windows_by_status neither | new segments_by_status both | new segments_by_status token only | new segments_by_status cls only | new segments_by_status neither | new sessions_by_status both | new sessions_by_status token only | new sessions_by_status cls only | new sessions_by_status neither | new sessions | segments_with_both |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0 | selection | pairs | mixed | 1 | 5 | 0 | 0 | 1 | 3 | 0 | 0 | 1 | 2 | 0 | 0 | 3 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 |
| s0 | selection | pairs | all | 1 | 5 | 0 | 2 | 1 | 3 | 0 | 1 | 1 | 2 | 0 | 1 | 3 | 4 | 0 | 0 | 1 | 4 | 0 | 0 | 1 | 4 | 0 | 0 | 1 | 5 | 1 |
| s0 | selection | multi | mixed | 35 | 4 | 0 | 5 | 33 | 3 | 0 | 4 | 13 | 2 | 0 | 3 | 15 | 6 | 1 | 0 | 0 | 6 | 1 | 0 | 0 | 4 | 1 | 0 | 0 | 5 | 0 |
| s0 | selection | multi | all | 36 | 4 | 1 | 5 | 34 | 3 | 1 | 4 | 14 | 2 | 1 | 3 | 15 | 17 | 1 | 1 | 3 | 16 | 1 | 1 | 3 | 7 | 1 | 1 | 1 | 8 | 2 |
| s1 | selection | pairs | mixed | 1 | 0 | 0 | 3 | 1 | 0 | 0 | 3 | 1 | 0 | 0 | 3 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| s1 | selection | pairs | all | 1 | 0 | 0 | 6 | 1 | 0 | 0 | 4 | 1 | 0 | 0 | 4 | 5 | 4 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 3 | 0 |
| s1 | selection | multi | mixed | 30 | 8 | 2 | 0 | 30 | 7 | 2 | 0 | 12 | 4 | 1 | 0 | 13 | 7 | 0 | 1 | 0 | 7 | 0 | 1 | 0 | 7 | 0 | 1 | 0 | 7 | 2 |
| s1 | selection | multi | all | 37 | 9 | 4 | 0 | 37 | 7 | 4 | 0 | 14 | 4 | 1 | 0 | 15 | 20 | 0 | 8 | 2 | 17 | 0 | 7 | 2 | 9 | 0 | 2 | 2 | 9 | 8 |
| s2 | selection | pairs | mixed | 1 | 1 | 0 | 2 | 1 | 1 | 0 | 2 | 1 | 1 | 0 | 2 | 3 | 2 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 2 | 0 |
| s2 | selection | pairs | all | 1 | 1 | 0 | 6 | 1 | 1 | 0 | 4 | 1 | 1 | 0 | 4 | 5 | 7 | 0 | 0 | 1 | 6 | 0 | 0 | 1 | 5 | 0 | 0 | 1 | 6 | 1 |
| s2 | selection | multi | mixed | 34 | 0 | 2 | 2 | 33 | 0 | 2 | 2 | 15 | 0 | 2 | 2 | 17 | 11 | 0 | 0 | 0 | 11 | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 8 | 0 |
| s2 | selection | multi | all | 39 | 1 | 2 | 3 | 38 | 1 | 2 | 3 | 15 | 1 | 2 | 2 | 17 | 23 | 1 | 1 | 0 | 19 | 1 | 1 | 0 | 11 | 1 | 1 | 0 | 11 | 3 |
| s0 | calibration | pairs | mixed | 1 | 2 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 2 | 1 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 2 | 0 |
| s0 | calibration | pairs | all | 2 | 2 | 0 | 0 | 2 | 1 | 0 | 0 | 2 | 1 | 0 | 0 | 3 | 4 | 1 | 0 | 0 | 3 | 1 | 0 | 0 | 3 | 1 | 0 | 0 | 4 | 0 |
| s0 | calibration | multi | mixed | 31 | 0 | 0 | 0 | 29 | 0 | 0 | 0 | 17 | 0 | 0 | 0 | 17 | 8 | 0 | 1 | 0 | 8 | 0 | 1 | 0 | 6 | 0 | 1 | 0 | 6 | 0 |
| s0 | calibration | multi | all | 31 | 0 | 0 | 0 | 29 | 0 | 0 | 0 | 17 | 0 | 0 | 0 | 17 | 18 | 0 | 1 | 2 | 17 | 0 | 1 | 1 | 9 | 0 | 1 | 2 | 10 | 3 |
| s1 | calibration | pairs | mixed | 3 | 2 | 0 | 0 | 3 | 1 | 0 | 0 | 3 | 1 | 0 | 0 | 4 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 |
| s1 | calibration | pairs | all | 3 | 2 | 0 | 1 | 3 | 1 | 0 | 1 | 3 | 1 | 0 | 1 | 5 | 5 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 3 | 0 |
| s1 | calibration | multi | mixed | 24 | 2 | 0 | 2 | 23 | 2 | 0 | 2 | 15 | 2 | 0 | 2 | 16 | 8 | 1 | 1 | 0 | 8 | 1 | 1 | 0 | 8 | 1 | 1 | 0 | 9 | 3 |
| s1 | calibration | multi | all | 25 | 2 | 0 | 2 | 24 | 2 | 0 | 2 | 16 | 2 | 0 | 2 | 17 | 15 | 1 | 5 | 2 | 14 | 1 | 3 | 2 | 10 | 1 | 2 | 2 | 11 | 4 |
| s2 | calibration | pairs | mixed | 3 | 5 | 0 | 0 | 3 | 3 | 0 | 0 | 3 | 3 | 0 | 0 | 6 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 |
| s2 | calibration | pairs | all | 3 | 5 | 0 | 0 | 3 | 3 | 0 | 0 | 3 | 3 | 0 | 0 | 6 | 1 | 0 | 1 | 2 | 1 | 0 | 1 | 1 | 1 | 0 | 1 | 2 | 4 | 0 |
| s2 | calibration | multi | mixed | 31 | 5 | 0 | 3 | 30 | 3 | 0 | 3 | 16 | 3 | 0 | 3 | 19 | 9 | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 8 | 1 |
| s2 | calibration | multi | all | 32 | 5 | 0 | 3 | 31 | 3 | 0 | 3 | 16 | 3 | 0 | 3 | 19 | 16 | 0 | 0 | 0 | 14 | 0 | 0 | 0 | 11 | 0 | 0 | 0 | 11 | 2 |

## Actual systems (token at A03's decoder against CLS at A02's decoder): probability-only changes

| seed | half | population | windows_same_argmax | windows_changed_argmax | tv_same_argmax n | tv_same_argmax mean | tv_same_argmax median | tv_same_argmax p90 | tv_changed_argmax n | tv_changed_argmax mean | tv_changed_argmax median | tv_changed_argmax p90 | decoded_changes samples | decoded_changes serving_window_argmax_changed | decoded_changes serving_window_argmax_same | decoded_changes sequences |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0 | selection | native | 3425 | 19 | 3425 | 0.0155 | 0.0064 | 0.0333 | 19 | 0.3478 | 0.3353 | 0.5325 | 69 | 19 | 50 | 6 |
| s0 | selection | pairs | 1177 | 13 | 1177 | 0.0221 | 0.0078 | 0.0521 | 13 | 0.4002 | 0.4191 | 0.6022 | 58 | 34 | 24 | 6 |
| s0 | selection | multi | 2962 | 70 | 2962 | 0.0283 | 0.0091 | 0.0755 | 70 | 0.3547 | 0.3266 | 0.5915 | 526 | 239 | 287 | 47 |
| s1 | selection | native | 3421 | 23 | 3421 | 0.0152 | 0.0054 | 0.0300 | 23 | 0.2419 | 0.1905 | 0.4459 | 131 | 56 | 75 | 12 |
| s1 | selection | pairs | 1179 | 11 | 1179 | 0.0220 | 0.0066 | 0.0466 | 11 | 0.3594 | 0.4344 | 0.4936 | 58 | 33 | 25 | 7 |
| s1 | selection | multi | 2950 | 82 | 2950 | 0.0285 | 0.0091 | 0.0750 | 82 | 0.3301 | 0.2919 | 0.5360 | 477 | 317 | 160 | 55 |
| s2 | selection | native | 3421 | 23 | 3421 | 0.0165 | 0.0067 | 0.0350 | 23 | 0.3592 | 0.3418 | 0.5347 | 242 | 64 | 178 | 8 |
| s2 | selection | pairs | 1174 | 16 | 1174 | 0.0203 | 0.0069 | 0.0405 | 16 | 0.4086 | 0.4112 | 0.5800 | 87 | 30 | 57 | 9 |
| s2 | selection | multi | 2960 | 72 | 2960 | 0.0255 | 0.0076 | 0.0685 | 72 | 0.3537 | 0.3251 | 0.5906 | 439 | 226 | 213 | 51 |
| s0 | calibration | native | 2934 | 16 | 2934 | 0.0160 | 0.0060 | 0.0261 | 16 | 0.2946 | 0.2864 | 0.4469 | 25 | 15 | 10 | 2 |
| s0 | calibration | pairs | 1014 | 9 | 1014 | 0.0177 | 0.0066 | 0.0358 | 9 | 0.3619 | 0.3155 | 0.7111 | 85 | 25 | 60 | 7 |
| s0 | calibration | multi | 2994 | 54 | 2994 | 0.0221 | 0.0073 | 0.0477 | 54 | 0.3914 | 0.3589 | 0.6021 | 258 | 189 | 69 | 36 |
| s1 | calibration | native | 2924 | 26 | 2924 | 0.0183 | 0.0054 | 0.0280 | 26 | 0.4018 | 0.4189 | 0.6280 | 199 | 95 | 104 | 9 |
| s1 | calibration | pairs | 1012 | 11 | 1012 | 0.0194 | 0.0062 | 0.0406 | 11 | 0.4214 | 0.4153 | 0.6590 | 65 | 50 | 15 | 9 |
| s1 | calibration | multi | 2996 | 52 | 2996 | 0.0233 | 0.0076 | 0.0514 | 52 | 0.3884 | 0.3685 | 0.6291 | 288 | 214 | 74 | 38 |
| s2 | calibration | native | 2933 | 17 | 2933 | 0.0209 | 0.0065 | 0.0331 | 17 | 0.3994 | 0.4348 | 0.5732 | 77 | 15 | 62 | 3 |
| s2 | calibration | pairs | 1011 | 12 | 1011 | 0.0188 | 0.0070 | 0.0359 | 12 | 0.4104 | 0.3546 | 0.7390 | 97 | 39 | 58 | 9 |
| s2 | calibration | multi | 2991 | 57 | 2991 | 0.0231 | 0.0066 | 0.0482 | 57 | 0.4286 | 0.4411 | 0.6691 | 265 | 198 | 67 | 43 |

## Actual systems (token at A03's decoder against CLS at A02's decoder): native label switches

| seed | half | system | decoding | switches | clips | clips_with_switches | recordings | sessions | few_units |
|---|---|---|---|---|---|---|---|---|---|
| s0 | selection | token | argmax | 42 | 284 | 16 | 16 | 11 | False |
| s0 | selection | token | b3 | 13 | 284 | 8 | 8 | 7 | True |
| s0 | selection | cls | argmax | 27 | 284 | 11 | 11 | 8 | True |
| s0 | selection | cls | b3 | 12 | 284 | 7 | 7 | 6 | True |
| s1 | selection | token | argmax | 45 | 284 | 16 | 16 | 10 | False |
| s1 | selection | token | b3 | 29 | 284 | 13 | 13 | 9 | True |
| s1 | selection | cls | argmax | 28 | 284 | 13 | 13 | 8 | True |
| s1 | selection | cls | b3 | 10 | 284 | 4 | 4 | 4 | True |
| s2 | selection | token | argmax | 46 | 284 | 18 | 18 | 12 | False |
| s2 | selection | token | b3 | 11 | 284 | 7 | 7 | 6 | True |
| s2 | selection | cls | argmax | 38 | 284 | 14 | 14 | 9 | True |
| s2 | selection | cls | b3 | 17 | 284 | 7 | 7 | 5 | True |
| s0 | calibration | token | argmax | 32 | 237 | 11 | 11 | 10 | False |
| s0 | calibration | token | b3 | 3 | 237 | 3 | 3 | 3 | True |
| s0 | calibration | cls | argmax | 18 | 237 | 6 | 6 | 6 | True |
| s0 | calibration | cls | b3 | 4 | 237 | 4 | 4 | 4 | True |
| s1 | calibration | token | argmax | 38 | 237 | 12 | 12 | 9 | True |
| s1 | calibration | token | b3 | 22 | 237 | 11 | 11 | 8 | True |
| s1 | calibration | cls | argmax | 18 | 237 | 6 | 6 | 5 | True |
| s1 | calibration | cls | b3 | 3 | 237 | 3 | 3 | 3 | True |
| s2 | calibration | token | argmax | 45 | 237 | 12 | 12 | 10 | False |
| s2 | calibration | token | b3 | 3 | 237 | 3 | 3 | 3 | True |
| s2 | calibration | cls | argmax | 20 | 237 | 7 | 7 | 7 | True |
| s2 | calibration | cls | b3 | 4 | 237 | 3 | 3 | 3 | True |

## Fixed decoder (TT against CC at A02's CLS setting): segments

| seed | half | population | token tp | token fp | token fn | token f1 | cls tp | cls fp | cls fn | cls f1 | both | token only | cls only | neither |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0 | selection | native | 280 | 17 | 4 | 0.9639 | 280 | 16 | 4 | 0.9655 | 279 | 1 | 1 | 3 |
| s0 | selection | pairs | 204 | 7 | 3 | 0.9761 | 200 | 9 | 7 | 0.9615 | 200 | 4 | 0 | 3 |
| s0 | selection | multi | 477 | 40 | 25 | 0.9362 | 474 | 30 | 28 | 0.9423 | 469 | 8 | 5 | 20 |
| s1 | selection | native | 280 | 15 | 4 | 0.9672 | 280 | 14 | 4 | 0.9689 | 280 | 0 | 0 | 4 |
| s1 | selection | pairs | 203 | 12 | 4 | 0.9621 | 201 | 11 | 6 | 0.9594 | 201 | 2 | 0 | 4 |
| s1 | selection | multi | 488 | 18 | 14 | 0.9683 | 482 | 30 | 20 | 0.9507 | 478 | 10 | 4 | 10 |
| s2 | selection | native | 277 | 25 | 7 | 0.9454 | 279 | 22 | 5 | 0.9538 | 275 | 2 | 4 | 3 |
| s2 | selection | pairs | 202 | 14 | 5 | 0.9551 | 200 | 12 | 7 | 0.9547 | 200 | 2 | 0 | 5 |
| s2 | selection | multi | 482 | 35 | 20 | 0.9460 | 487 | 24 | 15 | 0.9615 | 482 | 0 | 5 | 15 |
| s0 | calibration | native | 235 | 5 | 2 | 0.9853 | 236 | 5 | 1 | 0.9874 | 235 | 0 | 1 | 1 |
| s0 | calibration | pairs | 170 | 3 | 3 | 0.9827 | 168 | 2 | 5 | 0.9796 | 166 | 4 | 2 | 1 |
| s0 | calibration | multi | 511 | 13 | 7 | 0.9808 | 514 | 11 | 4 | 0.9856 | 511 | 0 | 3 | 4 |
| s1 | calibration | native | 233 | 8 | 4 | 0.9749 | 235 | 5 | 2 | 0.9853 | 233 | 0 | 2 | 2 |
| s1 | calibration | pairs | 168 | 6 | 5 | 0.9683 | 168 | 4 | 5 | 0.9739 | 166 | 2 | 2 | 3 |
| s1 | calibration | multi | 507 | 21 | 11 | 0.9694 | 511 | 14 | 7 | 0.9799 | 506 | 1 | 5 | 6 |
| s2 | calibration | native | 234 | 10 | 3 | 0.9730 | 236 | 5 | 1 | 0.9874 | 234 | 0 | 2 | 1 |
| s2 | calibration | pairs | 168 | 4 | 5 | 0.9739 | 166 | 2 | 7 | 0.9736 | 164 | 4 | 2 | 3 |
| s2 | calibration | multi | 506 | 13 | 12 | 0.9759 | 508 | 11 | 10 | 0.9797 | 503 | 3 | 5 | 7 |

## Fixed decoder (TT against CC at A02's CLS setting): changed segments

| seed | half | population | change | segments | by_class PLAX | by_class PSAX | by_class A4C | by_class A5C | by_class SC4C | short | long | xyx_a4c_a5c | recordings | sessions | few_units |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0 | selection | native | newly_matched | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 1 | True |
| s0 | selection | native | lost | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 1 | 1 | True |
| s0 | selection | native | new_unmatched_pred | 7 | 0 | 0 | 4 | 3 | 0 | 5 | 2 |  | 4 | 4 | True |
| s0 | selection | native | removed_unmatched_pred | 6 | 1 | 1 | 2 | 2 | 0 | 1 | 5 |  | 3 | 3 | True |
| s0 | selection | pairs | newly_matched | 4 | 1 | 1 | 1 | 1 | 0 | 3 | 1 | 0 | 4 | 3 | True |
| s0 | selection | pairs | lost | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | True |
| s0 | selection | pairs | new_unmatched_pred | 2 | 1 | 1 | 0 | 0 | 0 | 1 | 1 |  | 2 | 2 | True |
| s0 | selection | pairs | removed_unmatched_pred | 4 | 1 | 1 | 1 | 1 | 0 | 2 | 2 |  | 3 | 3 | True |
| s0 | selection | multi | newly_matched | 8 | 0 | 0 | 3 | 5 | 0 | 6 | 2 | 3 | 8 | 8 | True |
| s0 | selection | multi | lost | 5 | 0 | 0 | 2 | 3 | 0 | 0 | 5 | 0 | 6 | 5 | True |
| s0 | selection | multi | new_unmatched_pred | 26 | 0 | 1 | 15 | 10 | 0 | 17 | 9 |  | 20 | 14 | False |
| s0 | selection | multi | removed_unmatched_pred | 16 | 0 | 0 | 10 | 6 | 0 | 2 | 14 |  | 23 | 13 | False |
| s1 | selection | native | newly_matched | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | True |
| s1 | selection | native | lost | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | True |
| s1 | selection | native | new_unmatched_pred | 5 | 1 | 1 | 1 | 2 | 0 | 2 | 3 |  | 3 | 2 | True |
| s1 | selection | native | removed_unmatched_pred | 4 | 0 | 1 | 2 | 1 | 0 | 1 | 3 |  | 3 | 2 | True |
| s1 | selection | pairs | newly_matched | 2 | 0 | 0 | 1 | 1 | 0 | 2 | 0 | 0 | 2 | 2 | True |
| s1 | selection | pairs | lost | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | True |
| s1 | selection | pairs | new_unmatched_pred | 5 | 1 | 1 | 1 | 2 | 0 | 3 | 2 |  | 5 | 5 | True |
| s1 | selection | pairs | removed_unmatched_pred | 4 | 0 | 1 | 1 | 2 | 0 | 3 | 1 |  | 3 | 3 | True |
| s1 | selection | multi | newly_matched | 10 | 0 | 0 | 3 | 7 | 0 | 6 | 4 | 2 | 9 | 9 | True |
| s1 | selection | multi | lost | 4 | 0 | 0 | 3 | 1 | 0 | 1 | 3 | 0 | 4 | 4 | True |
| s1 | selection | multi | new_unmatched_pred | 9 | 0 | 2 | 5 | 2 | 0 | 5 | 4 |  | 12 | 10 | False |
| s1 | selection | multi | removed_unmatched_pred | 21 | 0 | 0 | 13 | 8 | 0 | 13 | 8 |  | 22 | 15 | False |
| s2 | selection | native | newly_matched | 2 | 0 | 0 | 2 | 0 | 0 | 0 | 2 | 0 | 2 | 1 | True |
| s2 | selection | native | lost | 4 | 0 | 0 | 3 | 1 | 0 | 0 | 4 | 0 | 4 | 4 | True |
| s2 | selection | native | new_unmatched_pred | 16 | 0 | 0 | 7 | 8 | 1 | 9 | 7 |  | 9 | 8 | True |
| s2 | selection | native | removed_unmatched_pred | 13 | 2 | 2 | 5 | 4 | 0 | 8 | 5 |  | 5 | 4 | True |
| s2 | selection | pairs | newly_matched | 2 | 1 | 0 | 1 | 0 | 0 | 2 | 0 | 0 | 2 | 2 | True |
| s2 | selection | pairs | lost | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | True |
| s2 | selection | pairs | new_unmatched_pred | 6 | 0 | 0 | 3 | 3 | 0 | 6 | 0 |  | 5 | 4 | True |
| s2 | selection | pairs | removed_unmatched_pred | 4 | 0 | 0 | 2 | 2 | 0 | 2 | 2 |  | 3 | 3 | True |
| s2 | selection | multi | newly_matched | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | True |
| s2 | selection | multi | lost | 5 | 0 | 1 | 2 | 2 | 0 | 1 | 4 | 0 | 5 | 5 | True |
| s2 | selection | multi | new_unmatched_pred | 19 | 0 | 1 | 13 | 5 | 0 | 18 | 1 |  | 11 | 8 | True |
| s2 | selection | multi | removed_unmatched_pred | 8 | 0 | 0 | 5 | 3 | 0 | 2 | 6 |  | 10 | 9 | True |
| s0 | calibration | native | newly_matched | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | True |
| s0 | calibration | native | lost | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 1 | True |
| s0 | calibration | native | new_unmatched_pred | 2 | 0 | 0 | 1 | 1 | 0 | 0 | 2 |  | 1 | 1 | True |
| s0 | calibration | native | removed_unmatched_pred | 2 | 0 | 0 | 2 | 0 | 0 | 0 | 2 |  | 2 | 2 | True |
| s0 | calibration | pairs | newly_matched | 4 | 1 | 0 | 2 | 1 | 0 | 4 | 0 | 0 | 4 | 4 | True |
| s0 | calibration | pairs | lost | 2 | 0 | 0 | 1 | 1 | 0 | 0 | 2 | 0 | 2 | 2 | True |
| s0 | calibration | pairs | new_unmatched_pred | 2 | 0 | 0 | 1 | 1 | 0 | 1 | 1 |  | 3 | 3 | True |
| s0 | calibration | pairs | removed_unmatched_pred | 1 | 0 | 0 | 0 | 1 | 0 | 1 | 0 |  | 1 | 1 | True |
| s0 | calibration | multi | newly_matched | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | True |
| s0 | calibration | multi | lost | 3 | 0 | 0 | 2 | 1 | 0 | 2 | 1 | 0 | 3 | 3 | True |
| s0 | calibration | multi | new_unmatched_pred | 7 | 0 | 0 | 3 | 4 | 0 | 4 | 3 |  | 11 | 10 | False |
| s0 | calibration | multi | removed_unmatched_pred | 5 | 0 | 0 | 2 | 3 | 0 | 2 | 3 |  | 7 | 7 | True |
| s1 | calibration | native | newly_matched | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | True |
| s1 | calibration | native | lost | 2 | 0 | 0 | 1 | 1 | 0 | 0 | 2 | 0 | 2 | 2 | True |
| s1 | calibration | native | new_unmatched_pred | 4 | 0 | 0 | 2 | 2 | 0 | 1 | 3 |  | 2 | 2 | True |
| s1 | calibration | native | removed_unmatched_pred | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |  | 1 | 1 | True |
| s1 | calibration | pairs | newly_matched | 2 | 1 | 0 | 0 | 1 | 0 | 2 | 0 | 0 | 2 | 2 | True |
| s1 | calibration | pairs | lost | 2 | 0 | 0 | 1 | 1 | 0 | 0 | 2 | 0 | 2 | 2 | True |
| s1 | calibration | pairs | new_unmatched_pred | 2 | 0 | 0 | 1 | 1 | 0 | 1 | 1 |  | 3 | 2 | True |
| s1 | calibration | pairs | removed_unmatched_pred | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  | 0 | 0 | True |
| s1 | calibration | multi | newly_matched | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | True |
| s1 | calibration | multi | lost | 5 | 1 | 0 | 1 | 3 | 0 | 2 | 3 | 0 | 7 | 7 | True |
| s1 | calibration | multi | new_unmatched_pred | 13 | 0 | 0 | 6 | 7 | 0 | 8 | 5 |  | 14 | 12 | False |
| s1 | calibration | multi | removed_unmatched_pred | 6 | 0 | 0 | 2 | 4 | 0 | 3 | 3 |  | 8 | 8 | True |
| s2 | calibration | native | newly_matched | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | True |
| s2 | calibration | native | lost | 2 | 0 | 0 | 0 | 2 | 0 | 0 | 2 | 0 | 2 | 2 | True |
| s2 | calibration | native | new_unmatched_pred | 8 | 0 | 0 | 3 | 5 | 0 | 3 | 5 |  | 3 | 3 | True |
| s2 | calibration | native | removed_unmatched_pred | 3 | 0 | 0 | 2 | 1 | 0 | 1 | 2 |  | 2 | 2 | True |
| s2 | calibration | pairs | newly_matched | 4 | 1 | 0 | 2 | 1 | 0 | 4 | 0 | 0 | 4 | 4 | True |
| s2 | calibration | pairs | lost | 2 | 0 | 0 | 1 | 1 | 0 | 0 | 2 | 0 | 2 | 2 | True |
| s2 | calibration | pairs | new_unmatched_pred | 2 | 0 | 0 | 1 | 1 | 0 | 1 | 1 |  | 3 | 3 | True |
| s2 | calibration | pairs | removed_unmatched_pred | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  | 0 | 0 | True |
| s2 | calibration | multi | newly_matched | 3 | 3 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 3 | 3 | True |
| s2 | calibration | multi | lost | 5 | 0 | 1 | 1 | 3 | 0 | 2 | 3 | 0 | 5 | 5 | True |
| s2 | calibration | multi | new_unmatched_pred | 8 | 0 | 0 | 3 | 5 | 0 | 4 | 4 |  | 12 | 12 | False |
| s2 | calibration | multi | removed_unmatched_pred | 6 | 0 | 0 | 2 | 4 | 0 | 4 | 2 |  | 11 | 8 | True |

## Fixed decoder (TT against CC at A02's CLS setting): changed requested-window correctness by distance

| seed | half | population | windows | distance | fixed windows | fixed recordings | fixed sessions | fixed few_units | new windows | new recordings | new sessions | new few_units |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0 | selection | native | 3444 | none | 8 | 4 | 3 | True | 11 | 9 | 8 | True |
| s0 | selection | pairs | 166 | 0-5 | 4 | 4 | 3 | True | 1 | 1 | 1 | True |
| s0 | selection | pairs | 79 | 6-10 | 2 | 2 | 2 | True | 0 | 0 | 0 | True |
| s0 | selection | pairs | 24 | >10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s0 | selection | pairs | 62 | 6-10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s0 | selection | pairs | 259 | >10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s0 | selection | pairs | 600 | none | 2 | 1 | 1 | True | 4 | 4 | 4 | True |
| s0 | selection | multi | 916 | 0-5 | 43 | 29 | 15 | False | 6 | 6 | 5 | True |
| s0 | selection | multi | 242 | 6-10 | 1 | 1 | 1 | True | 1 | 1 | 1 | True |
| s0 | selection | multi | 22 | >10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s0 | selection | multi | 373 | 6-10 | 0 | 0 | 0 | True | 3 | 1 | 1 | True |
| s0 | selection | multi | 1460 | >10 | 2 | 1 | 1 | True | 12 | 5 | 4 | True |
| s0 | selection | multi | 19 | none | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s1 | selection | native | 3444 | none | 13 | 11 | 7 | True | 10 | 8 | 6 | True |
| s1 | selection | pairs | 166 | 0-5 | 4 | 4 | 4 | True | 0 | 0 | 0 | True |
| s1 | selection | pairs | 79 | 6-10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s1 | selection | pairs | 24 | >10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s1 | selection | pairs | 62 | 6-10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s1 | selection | pairs | 259 | >10 | 1 | 1 | 1 | True | 0 | 0 | 0 | True |
| s1 | selection | pairs | 600 | none | 2 | 1 | 1 | True | 4 | 4 | 3 | True |
| s1 | selection | multi | 916 | 0-5 | 38 | 26 | 13 | False | 8 | 8 | 7 | True |
| s1 | selection | multi | 242 | 6-10 | 2 | 2 | 2 | True | 0 | 0 | 0 | True |
| s1 | selection | multi | 22 | >10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s1 | selection | multi | 373 | 6-10 | 1 | 1 | 1 | True | 6 | 3 | 2 | True |
| s1 | selection | multi | 1460 | >10 | 9 | 7 | 6 | True | 16 | 4 | 4 | True |
| s1 | selection | multi | 19 | none | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s2 | selection | native | 3444 | none | 9 | 7 | 5 | True | 14 | 7 | 6 | True |
| s2 | selection | pairs | 166 | 0-5 | 4 | 4 | 3 | True | 2 | 2 | 2 | True |
| s2 | selection | pairs | 79 | 6-10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s2 | selection | pairs | 24 | >10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s2 | selection | pairs | 62 | 6-10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s2 | selection | pairs | 259 | >10 | 1 | 1 | 1 | True | 1 | 1 | 1 | True |
| s2 | selection | pairs | 600 | none | 3 | 1 | 1 | True | 5 | 4 | 4 | True |
| s2 | selection | multi | 916 | 0-5 | 38 | 33 | 17 | False | 11 | 9 | 8 | True |
| s2 | selection | multi | 242 | 6-10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s2 | selection | multi | 22 | >10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s2 | selection | multi | 373 | 6-10 | 1 | 1 | 1 | True | 1 | 1 | 1 | True |
| s2 | selection | multi | 1460 | >10 | 6 | 3 | 2 | True | 13 | 5 | 5 | True |
| s2 | selection | multi | 19 | none | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s0 | calibration | native | 2950 | none | 5 | 1 | 1 | True | 11 | 9 | 8 | True |
| s0 | calibration | pairs | 135 | 0-5 | 2 | 2 | 2 | True | 1 | 1 | 1 | True |
| s0 | calibration | pairs | 65 | 6-10 | 1 | 1 | 1 | True | 1 | 1 | 1 | True |
| s0 | calibration | pairs | 17 | >10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s0 | calibration | pairs | 54 | 6-10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s0 | calibration | pairs | 273 | >10 | 0 | 0 | 0 | True | 1 | 1 | 1 | True |
| s0 | calibration | pairs | 479 | none | 1 | 1 | 1 | True | 2 | 1 | 1 | True |
| s0 | calibration | multi | 945 | 0-5 | 31 | 28 | 17 | False | 8 | 7 | 6 | True |
| s0 | calibration | multi | 257 | 6-10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s0 | calibration | multi | 21 | >10 | 0 | 0 | 0 | True | 1 | 1 | 1 | True |
| s0 | calibration | multi | 391 | 6-10 | 0 | 0 | 0 | True | 5 | 3 | 3 | True |
| s0 | calibration | multi | 1434 | >10 | 0 | 0 | 0 | True | 7 | 4 | 4 | True |
| s1 | calibration | native | 2950 | none | 5 | 4 | 4 | True | 21 | 8 | 7 | True |
| s1 | calibration | pairs | 135 | 0-5 | 4 | 4 | 4 | True | 1 | 1 | 1 | True |
| s1 | calibration | pairs | 65 | 6-10 | 1 | 1 | 1 | True | 0 | 0 | 0 | True |
| s1 | calibration | pairs | 17 | >10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s1 | calibration | pairs | 54 | 6-10 | 0 | 0 | 0 | True | 1 | 1 | 1 | True |
| s1 | calibration | pairs | 273 | >10 | 0 | 0 | 0 | True | 1 | 1 | 1 | True |
| s1 | calibration | pairs | 479 | none | 1 | 1 | 1 | True | 2 | 1 | 1 | True |
| s1 | calibration | multi | 945 | 0-5 | 28 | 27 | 16 | False | 8 | 8 | 7 | True |
| s1 | calibration | multi | 257 | 6-10 | 0 | 0 | 0 | True | 2 | 2 | 2 | True |
| s1 | calibration | multi | 21 | >10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s1 | calibration | multi | 391 | 6-10 | 0 | 0 | 0 | True | 6 | 5 | 4 | True |
| s1 | calibration | multi | 1434 | >10 | 1 | 1 | 1 | True | 7 | 2 | 2 | True |
| s2 | calibration | native | 2950 | none | 2 | 1 | 1 | True | 15 | 8 | 8 | True |
| s2 | calibration | pairs | 135 | 0-5 | 6 | 6 | 6 | True | 0 | 0 | 0 | True |
| s2 | calibration | pairs | 65 | 6-10 | 2 | 2 | 2 | True | 1 | 1 | 1 | True |
| s2 | calibration | pairs | 17 | >10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s2 | calibration | pairs | 54 | 6-10 | 0 | 0 | 0 | True | 1 | 1 | 1 | True |
| s2 | calibration | pairs | 273 | >10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s2 | calibration | pairs | 479 | none | 0 | 0 | 0 | True | 2 | 2 | 2 | True |
| s2 | calibration | multi | 945 | 0-5 | 37 | 35 | 19 | False | 9 | 8 | 8 | True |
| s2 | calibration | multi | 257 | 6-10 | 2 | 2 | 2 | True | 0 | 0 | 0 | True |
| s2 | calibration | multi | 21 | >10 | 0 | 0 | 0 | True | 0 | 0 | 0 | True |
| s2 | calibration | multi | 391 | 6-10 | 0 | 0 | 0 | True | 2 | 1 | 1 | True |
| s2 | calibration | multi | 1434 | >10 | 1 | 1 | 1 | True | 5 | 4 | 4 | True |

## Fixed decoder (TT against CC at A02's CLS setting): reference segments holding changed requests

| seed | half | population | windows | fixed windows_by_status both | fixed windows_by_status token only | fixed windows_by_status cls only | fixed windows_by_status neither | fixed segments_by_status both | fixed segments_by_status token only | fixed segments_by_status cls only | fixed segments_by_status neither | fixed sessions_by_status both | fixed sessions_by_status token only | fixed sessions_by_status cls only | fixed sessions_by_status neither | fixed sessions | new windows_by_status both | new windows_by_status token only | new windows_by_status cls only | new windows_by_status neither | new segments_by_status both | new segments_by_status token only | new segments_by_status cls only | new segments_by_status neither | new sessions_by_status both | new sessions_by_status token only | new sessions_by_status cls only | new sessions_by_status neither | new sessions | segments_with_both |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0 | selection | pairs | mixed | 1 | 5 | 0 | 0 | 1 | 3 | 0 | 0 | 1 | 2 | 0 | 0 | 3 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 |
| s0 | selection | pairs | all | 1 | 5 | 0 | 2 | 1 | 3 | 0 | 1 | 1 | 2 | 0 | 1 | 3 | 4 | 0 | 0 | 1 | 4 | 0 | 0 | 1 | 4 | 0 | 0 | 1 | 5 | 1 |
| s0 | selection | multi | mixed | 35 | 4 | 0 | 5 | 33 | 3 | 0 | 4 | 13 | 2 | 0 | 3 | 15 | 6 | 1 | 0 | 0 | 6 | 1 | 0 | 0 | 4 | 1 | 0 | 0 | 5 | 0 |
| s0 | selection | multi | all | 36 | 4 | 1 | 5 | 34 | 3 | 1 | 4 | 14 | 2 | 1 | 3 | 15 | 17 | 1 | 1 | 3 | 16 | 1 | 1 | 3 | 7 | 1 | 1 | 1 | 8 | 2 |
| s1 | selection | pairs | mixed | 1 | 1 | 0 | 2 | 1 | 1 | 0 | 2 | 1 | 1 | 0 | 2 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| s1 | selection | pairs | all | 1 | 1 | 0 | 5 | 1 | 1 | 0 | 3 | 1 | 1 | 0 | 3 | 5 | 4 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 3 | 0 |
| s1 | selection | multi | mixed | 31 | 6 | 1 | 2 | 31 | 5 | 1 | 2 | 12 | 3 | 1 | 2 | 13 | 7 | 0 | 1 | 0 | 7 | 0 | 1 | 0 | 7 | 0 | 1 | 0 | 7 | 2 |
| s1 | selection | multi | all | 40 | 6 | 1 | 3 | 40 | 5 | 1 | 2 | 14 | 3 | 1 | 2 | 15 | 26 | 1 | 2 | 1 | 23 | 1 | 1 | 1 | 9 | 1 | 1 | 1 | 9 | 8 |
| s2 | selection | pairs | mixed | 1 | 1 | 0 | 2 | 1 | 1 | 0 | 2 | 1 | 1 | 0 | 2 | 3 | 2 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 2 | 0 |
| s2 | selection | pairs | all | 1 | 1 | 0 | 6 | 1 | 1 | 0 | 4 | 1 | 1 | 0 | 4 | 5 | 7 | 0 | 0 | 1 | 6 | 0 | 0 | 1 | 5 | 0 | 0 | 1 | 6 | 1 |
| s2 | selection | multi | mixed | 36 | 0 | 0 | 2 | 35 | 0 | 0 | 2 | 16 | 0 | 0 | 2 | 17 | 10 | 0 | 1 | 0 | 10 | 0 | 1 | 0 | 7 | 0 | 1 | 0 | 8 | 0 |
| s2 | selection | multi | all | 41 | 0 | 0 | 4 | 40 | 0 | 0 | 4 | 16 | 0 | 0 | 2 | 17 | 22 | 0 | 2 | 1 | 18 | 0 | 2 | 1 | 10 | 0 | 2 | 1 | 11 | 3 |
| s0 | calibration | pairs | mixed | 1 | 2 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 2 | 1 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 2 | 0 |
| s0 | calibration | pairs | all | 2 | 2 | 0 | 0 | 2 | 1 | 0 | 0 | 2 | 1 | 0 | 0 | 3 | 4 | 1 | 0 | 0 | 3 | 1 | 0 | 0 | 3 | 1 | 0 | 0 | 4 | 0 |
| s0 | calibration | multi | mixed | 31 | 0 | 0 | 0 | 29 | 0 | 0 | 0 | 17 | 0 | 0 | 0 | 17 | 8 | 0 | 1 | 0 | 8 | 0 | 1 | 0 | 6 | 0 | 1 | 0 | 6 | 0 |
| s0 | calibration | multi | all | 31 | 0 | 0 | 0 | 29 | 0 | 0 | 0 | 17 | 0 | 0 | 0 | 17 | 18 | 0 | 1 | 2 | 17 | 0 | 1 | 1 | 9 | 0 | 1 | 2 | 10 | 3 |
| s1 | calibration | pairs | mixed | 3 | 2 | 0 | 0 | 3 | 1 | 0 | 0 | 3 | 1 | 0 | 0 | 4 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 |
| s1 | calibration | pairs | all | 3 | 2 | 0 | 1 | 3 | 1 | 0 | 1 | 3 | 1 | 0 | 1 | 5 | 4 | 0 | 1 | 0 | 3 | 0 | 1 | 0 | 3 | 0 | 1 | 0 | 3 | 0 |
| s1 | calibration | multi | mixed | 24 | 1 | 0 | 3 | 23 | 1 | 0 | 3 | 15 | 1 | 0 | 3 | 16 | 8 | 0 | 1 | 1 | 8 | 0 | 1 | 1 | 8 | 0 | 1 | 1 | 9 | 3 |
| s1 | calibration | multi | all | 25 | 1 | 0 | 3 | 24 | 1 | 0 | 3 | 16 | 1 | 0 | 3 | 17 | 17 | 0 | 3 | 3 | 15 | 0 | 2 | 3 | 10 | 0 | 2 | 3 | 11 | 4 |
| s2 | calibration | pairs | mixed | 3 | 5 | 0 | 0 | 3 | 3 | 0 | 0 | 3 | 3 | 0 | 0 | 6 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 |
| s2 | calibration | pairs | all | 3 | 5 | 0 | 0 | 3 | 3 | 0 | 0 | 3 | 3 | 0 | 0 | 6 | 1 | 0 | 1 | 2 | 1 | 0 | 1 | 1 | 1 | 0 | 1 | 2 | 4 | 0 |
| s2 | calibration | multi | mixed | 30 | 5 | 1 | 3 | 29 | 3 | 1 | 3 | 16 | 3 | 1 | 3 | 19 | 9 | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 8 | 1 |
| s2 | calibration | multi | all | 31 | 5 | 1 | 3 | 30 | 3 | 1 | 3 | 16 | 3 | 1 | 3 | 19 | 16 | 0 | 0 | 0 | 14 | 0 | 0 | 0 | 11 | 0 | 0 | 0 | 11 | 2 |

## Fixed decoder (TT against CC at A02's CLS setting): probability-only changes

| seed | half | population | windows_same_argmax | windows_changed_argmax | tv_same_argmax n | tv_same_argmax mean | tv_same_argmax median | tv_same_argmax p90 | tv_changed_argmax n | tv_changed_argmax mean | tv_changed_argmax median | tv_changed_argmax p90 | decoded_changes samples | decoded_changes serving_window_argmax_changed | decoded_changes serving_window_argmax_same | decoded_changes sequences |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0 | selection | native | 3425 | 19 | 3425 | 0.0155 | 0.0064 | 0.0333 | 19 | 0.3478 | 0.3353 | 0.5325 | 69 | 19 | 50 | 6 |
| s0 | selection | pairs | 1177 | 13 | 1177 | 0.0221 | 0.0078 | 0.0521 | 13 | 0.4002 | 0.4191 | 0.6022 | 58 | 34 | 24 | 6 |
| s0 | selection | multi | 2962 | 70 | 2962 | 0.0283 | 0.0091 | 0.0755 | 70 | 0.3547 | 0.3266 | 0.5915 | 526 | 239 | 287 | 47 |
| s1 | selection | native | 3421 | 23 | 3421 | 0.0152 | 0.0054 | 0.0300 | 23 | 0.2419 | 0.1905 | 0.4459 | 59 | 14 | 45 | 4 |
| s1 | selection | pairs | 1179 | 11 | 1179 | 0.0220 | 0.0066 | 0.0466 | 11 | 0.3594 | 0.4344 | 0.4936 | 76 | 30 | 46 | 6 |
| s1 | selection | multi | 2950 | 82 | 2950 | 0.0285 | 0.0091 | 0.0750 | 82 | 0.3301 | 0.2919 | 0.5360 | 426 | 232 | 194 | 47 |
| s2 | selection | native | 3421 | 23 | 3421 | 0.0165 | 0.0067 | 0.0350 | 23 | 0.3592 | 0.3418 | 0.5347 | 304 | 75 | 229 | 12 |
| s2 | selection | pairs | 1174 | 16 | 1174 | 0.0203 | 0.0069 | 0.0405 | 16 | 0.4086 | 0.4112 | 0.5800 | 67 | 35 | 32 | 8 |
| s2 | selection | multi | 2960 | 72 | 2960 | 0.0255 | 0.0076 | 0.0685 | 72 | 0.3537 | 0.3251 | 0.5906 | 361 | 235 | 126 | 45 |
| s0 | calibration | native | 2934 | 16 | 2934 | 0.0160 | 0.0060 | 0.0261 | 16 | 0.2946 | 0.2864 | 0.4469 | 25 | 15 | 10 | 2 |
| s0 | calibration | pairs | 1014 | 9 | 1014 | 0.0177 | 0.0066 | 0.0358 | 9 | 0.3619 | 0.3155 | 0.7111 | 85 | 25 | 60 | 7 |
| s0 | calibration | multi | 2994 | 54 | 2994 | 0.0221 | 0.0073 | 0.0477 | 54 | 0.3914 | 0.3589 | 0.6021 | 268 | 189 | 79 | 37 |
| s1 | calibration | native | 2924 | 26 | 2924 | 0.0183 | 0.0054 | 0.0280 | 26 | 0.4018 | 0.4189 | 0.6280 | 25 | 15 | 10 | 2 |
| s1 | calibration | pairs | 1012 | 11 | 1012 | 0.0194 | 0.0062 | 0.0406 | 11 | 0.4214 | 0.4153 | 0.6590 | 73 | 45 | 28 | 7 |
| s1 | calibration | multi | 2996 | 52 | 2996 | 0.0233 | 0.0076 | 0.0514 | 52 | 0.3884 | 0.3685 | 0.6291 | 245 | 190 | 55 | 36 |
| s2 | calibration | native | 2933 | 17 | 2933 | 0.0209 | 0.0065 | 0.0331 | 17 | 0.3994 | 0.4348 | 0.5732 | 147 | 20 | 127 | 3 |
| s2 | calibration | pairs | 1011 | 12 | 1011 | 0.0188 | 0.0070 | 0.0359 | 12 | 0.4104 | 0.3546 | 0.7390 | 84 | 39 | 45 | 8 |
| s2 | calibration | multi | 2991 | 57 | 2991 | 0.0231 | 0.0066 | 0.0482 | 57 | 0.4286 | 0.4411 | 0.6691 | 295 | 223 | 72 | 38 |

## Fixed decoder (TT against CC at A02's CLS setting): native label switches

| seed | half | system | decoding | switches | clips | clips_with_switches | recordings | sessions | few_units |
|---|---|---|---|---|---|---|---|---|---|
| s0 | selection | token | argmax | 42 | 284 | 16 | 16 | 11 | False |
| s0 | selection | token | b3 | 13 | 284 | 8 | 8 | 7 | True |
| s0 | selection | cls | argmax | 27 | 284 | 11 | 11 | 8 | True |
| s0 | selection | cls | b3 | 12 | 284 | 7 | 7 | 6 | True |
| s1 | selection | token | argmax | 45 | 284 | 16 | 16 | 10 | False |
| s1 | selection | token | b3 | 11 | 284 | 5 | 5 | 4 | True |
| s1 | selection | cls | argmax | 28 | 284 | 13 | 13 | 8 | True |
| s1 | selection | cls | b3 | 10 | 284 | 4 | 4 | 4 | True |
| s2 | selection | token | argmax | 46 | 284 | 18 | 18 | 12 | False |
| s2 | selection | token | b3 | 18 | 284 | 11 | 11 | 9 | True |
| s2 | selection | cls | argmax | 38 | 284 | 14 | 14 | 9 | True |
| s2 | selection | cls | b3 | 17 | 284 | 7 | 7 | 5 | True |
| s0 | calibration | token | argmax | 32 | 237 | 11 | 11 | 10 | False |
| s0 | calibration | token | b3 | 3 | 237 | 3 | 3 | 3 | True |
| s0 | calibration | cls | argmax | 18 | 237 | 6 | 6 | 6 | True |
| s0 | calibration | cls | b3 | 4 | 237 | 4 | 4 | 4 | True |
| s1 | calibration | token | argmax | 38 | 237 | 12 | 12 | 9 | True |
| s1 | calibration | token | b3 | 4 | 237 | 4 | 4 | 4 | True |
| s1 | calibration | cls | argmax | 18 | 237 | 6 | 6 | 5 | True |
| s1 | calibration | cls | b3 | 3 | 237 | 3 | 3 | 3 | True |
| s2 | calibration | token | argmax | 45 | 237 | 12 | 12 | 10 | False |
| s2 | calibration | token | b3 | 7 | 237 | 4 | 4 | 4 | True |
| s2 | calibration | cls | argmax | 20 | 237 | 7 | 7 | 7 | True |
| s2 | calibration | cls | b3 | 4 | 237 | 3 | 3 | 3 | True |

## Four reference-conditioned cells at the fixed decoder (diagnostics only)

| seed | half | decoding | cell | segment_f1_50_multi | native_macro_f1 | boundary_precision_multi | boundary_recall_multi | boundary_f1_multi | native_internal_cuts_per_clip | same_class_cut_rate_multi | native tp | native fp | native fn | pairs tp | pairs fp | pairs fn | multi tp | multi fp | multi fn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0 | selection | b3 | CC | 0.9423 | 0.9786 | 0.9531 | 0.9581 | 0.9556 | 0.0423 | 0.0217 | 280 | 16 | 4 | 200 | 9 | 7 | 474 | 30 | 28 |
| s0 | selection | b3 | TC | 0.9459 | 0.9786 | 0.9443 | 0.9764 | 0.9601 | 0.0423 | 0.0217 | 280 | 16 | 4 | 203 | 10 | 4 | 481 | 34 | 21 |
| s0 | selection | b3 | CT | 0.9365 | 0.9786 | 0.9482 | 0.9581 | 0.9531 | 0.0458 | 0.0217 | 280 | 17 | 4 | 201 | 7 | 6 | 472 | 34 | 30 |
| s0 | selection | b3 | TT | 0.9362 | 0.9786 | 0.9370 | 0.9738 | 0.9551 | 0.0458 | 0.0217 | 280 | 17 | 4 | 204 | 7 | 3 | 477 | 40 | 25 |
| s0 | selection | argmax | CC | 0.8973 | 0.9877 | 0.8451 | 0.9712 | 0.9038 | 0.0951 | 0.0522 | 278 | 33 | 6 | 201 | 13 | 6 | 476 | 83 | 26 |
| s0 | selection | argmax | TC | 0.8951 | 0.9877 | 0.8408 | 0.9817 | 0.9058 | 0.0951 | 0.0522 | 278 | 33 | 6 | 203 | 12 | 4 | 478 | 88 | 24 |
| s0 | selection | argmax | CT | 0.8852 | 0.9877 | 0.8263 | 0.9712 | 0.8929 | 0.1479 | 0.0565 | 276 | 50 | 8 | 200 | 20 | 7 | 474 | 95 | 28 |
| s0 | selection | argmax | TT | 0.8831 | 0.9877 | 0.8224 | 0.9817 | 0.8950 | 0.1479 | 0.0565 | 276 | 50 | 8 | 202 | 19 | 5 | 476 | 100 | 26 |
| s1 | selection | b3 | CC | 0.9507 | 0.9877 | 0.9413 | 0.9660 | 0.9535 | 0.0352 | 0.0174 | 280 | 14 | 4 | 201 | 11 | 6 | 482 | 30 | 20 |
| s1 | selection | b3 | TC | 0.9539 | 0.9877 | 0.9395 | 0.9764 | 0.9576 | 0.0352 | 0.0130 | 280 | 14 | 4 | 203 | 10 | 4 | 486 | 31 | 16 |
| s1 | selection | b3 | CT | 0.9571 | 0.9877 | 0.9659 | 0.9634 | 0.9646 | 0.0387 | 0.0087 | 280 | 15 | 4 | 201 | 13 | 6 | 480 | 21 | 22 |
| s1 | selection | b3 | TT | 0.9683 | 0.9877 | 0.9663 | 0.9764 | 0.9714 | 0.0387 | 0.0043 | 280 | 15 | 4 | 203 | 12 | 4 | 488 | 18 | 14 |
| s1 | selection | argmax | CC | 0.9144 | 0.9786 | 0.8549 | 0.9869 | 0.9162 | 0.0986 | 0.0348 | 276 | 36 | 8 | 202 | 15 | 5 | 486 | 75 | 16 |
| s1 | selection | argmax | TC | 0.9135 | 0.9786 | 0.8575 | 0.9921 | 0.9199 | 0.0986 | 0.0304 | 276 | 36 | 8 | 202 | 17 | 5 | 486 | 76 | 16 |
| s1 | selection | argmax | CT | 0.9047 | 0.9869 | 0.8415 | 0.9869 | 0.9084 | 0.1585 | 0.0304 | 274 | 55 | 10 | 202 | 21 | 5 | 484 | 84 | 18 |
| s1 | selection | argmax | TT | 0.9042 | 0.9869 | 0.8366 | 0.9921 | 0.9078 | 0.1585 | 0.0261 | 274 | 55 | 10 | 202 | 23 | 5 | 486 | 87 | 16 |
| s2 | selection | b3 | CC | 0.9615 | 0.9786 | 0.9540 | 0.9764 | 0.9651 | 0.0599 | 0.0217 | 279 | 22 | 5 | 200 | 12 | 7 | 487 | 24 | 15 |
| s2 | selection | b3 | TC | 0.9537 | 0.9786 | 0.9466 | 0.9738 | 0.9600 | 0.0599 | 0.0217 | 279 | 22 | 5 | 202 | 12 | 5 | 484 | 29 | 18 |
| s2 | selection | b3 | CT | 0.9538 | 0.9700 | 0.9443 | 0.9764 | 0.9601 | 0.0634 | 0.0174 | 277 | 25 | 7 | 200 | 14 | 7 | 485 | 30 | 17 |
| s2 | selection | b3 | TT | 0.9460 | 0.9700 | 0.9370 | 0.9738 | 0.9551 | 0.0634 | 0.0174 | 277 | 25 | 7 | 202 | 14 | 5 | 482 | 35 | 20 |
| s2 | selection | argmax | CC | 0.9108 | 0.9877 | 0.8488 | 0.9843 | 0.9115 | 0.1338 | 0.0304 | 276 | 46 | 8 | 202 | 19 | 5 | 485 | 78 | 17 |
| s2 | selection | argmax | TC | 0.9042 | 0.9877 | 0.8344 | 0.9895 | 0.9054 | 0.1338 | 0.0304 | 276 | 46 | 8 | 202 | 20 | 5 | 486 | 87 | 16 |
| s2 | selection | argmax | CT | 0.9009 | 0.9700 | 0.8393 | 0.9843 | 0.9060 | 0.1620 | 0.0435 | 274 | 56 | 10 | 201 | 25 | 6 | 482 | 86 | 20 |
| s2 | selection | argmax | TT | 0.8961 | 0.9700 | 0.8289 | 0.9895 | 0.9021 | 0.1620 | 0.0435 | 274 | 56 | 10 | 201 | 26 | 6 | 483 | 93 | 19 |
| s0 | calibration | b3 | CC | 0.9856 | 0.9929 | 0.9778 | 0.9950 | 0.9863 | 0.0169 | 0.0249 | 236 | 5 | 1 | 168 | 2 | 5 | 514 | 11 | 4 |
| s0 | calibration | b3 | TC | 0.9817 | 0.9929 | 0.9752 | 0.9874 | 0.9813 | 0.0169 | 0.0249 | 236 | 5 | 1 | 172 | 2 | 1 | 511 | 12 | 7 |
| s0 | calibration | b3 | CT | 0.9818 | 0.9856 | 0.9753 | 0.9925 | 0.9838 | 0.0127 | 0.0249 | 235 | 5 | 2 | 166 | 3 | 7 | 512 | 13 | 6 |
| s0 | calibration | b3 | TT | 0.9808 | 0.9856 | 0.9728 | 0.9874 | 0.9800 | 0.0127 | 0.0299 | 235 | 5 | 2 | 170 | 3 | 3 | 511 | 13 | 7 |
| s0 | calibration | argmax | CC | 0.9734 | 0.9929 | 0.9519 | 0.9950 | 0.9730 | 0.0759 | 0.0299 | 234 | 21 | 3 | 171 | 4 | 2 | 513 | 23 | 5 |
| s0 | calibration | argmax | TC | 0.9753 | 0.9929 | 0.9541 | 0.9925 | 0.9729 | 0.0759 | 0.0299 | 234 | 21 | 3 | 171 | 5 | 2 | 513 | 21 | 5 |
| s0 | calibration | argmax | CT | 0.9595 | 0.9861 | 0.9294 | 0.9925 | 0.9599 | 0.1350 | 0.0448 | 232 | 37 | 5 | 171 | 7 | 2 | 510 | 35 | 8 |
| s0 | calibration | argmax | TT | 0.9559 | 0.9861 | 0.9227 | 0.9899 | 0.9552 | 0.1350 | 0.0448 | 232 | 37 | 5 | 171 | 8 | 2 | 509 | 38 | 9 |
| s1 | calibration | b3 | CC | 0.9799 | 0.9861 | 0.9728 | 0.9899 | 0.9813 | 0.0127 | 0.0199 | 235 | 5 | 2 | 168 | 4 | 5 | 511 | 14 | 7 |
| s1 | calibration | b3 | TC | 0.9751 | 0.9861 | 0.9680 | 0.9874 | 0.9776 | 0.0127 | 0.0199 | 235 | 5 | 2 | 170 | 4 | 3 | 509 | 17 | 9 |
| s1 | calibration | b3 | CT | 0.9741 | 0.9723 | 0.9704 | 0.9874 | 0.9788 | 0.0169 | 0.0199 | 233 | 8 | 4 | 166 | 6 | 7 | 508 | 17 | 10 |
| s1 | calibration | b3 | TT | 0.9694 | 0.9723 | 0.9632 | 0.9874 | 0.9752 | 0.0169 | 0.0299 | 233 | 8 | 4 | 168 | 6 | 5 | 507 | 21 | 11 |
| s1 | calibration | argmax | CC | 0.9696 | 0.9796 | 0.9495 | 0.9925 | 0.9705 | 0.0759 | 0.0348 | 233 | 22 | 4 | 169 | 7 | 4 | 511 | 25 | 7 |
| s1 | calibration | argmax | TC | 0.9687 | 0.9796 | 0.9496 | 0.9950 | 0.9718 | 0.0759 | 0.0348 | 233 | 22 | 4 | 170 | 7 | 3 | 511 | 26 | 7 |
| s1 | calibration | argmax | CT | 0.9521 | 0.9861 | 0.9227 | 0.9899 | 0.9552 | 0.1603 | 0.0448 | 231 | 44 | 6 | 169 | 11 | 4 | 507 | 40 | 11 |
| s1 | calibration | argmax | TT | 0.9512 | 0.9861 | 0.9229 | 0.9925 | 0.9564 | 0.1603 | 0.0448 | 231 | 44 | 6 | 170 | 11 | 3 | 507 | 41 | 11 |
| s2 | calibration | b3 | CC | 0.9797 | 0.9929 | 0.9825 | 0.9849 | 0.9837 | 0.0169 | 0.0199 | 236 | 5 | 1 | 166 | 2 | 7 | 508 | 11 | 10 |
| s2 | calibration | b3 | TC | 0.9768 | 0.9929 | 0.9749 | 0.9749 | 0.9749 | 0.0169 | 0.0199 | 236 | 5 | 1 | 170 | 3 | 3 | 506 | 12 | 12 |
| s2 | calibration | b3 | CT | 0.9759 | 0.9929 | 0.9776 | 0.9849 | 0.9812 | 0.0295 | 0.0299 | 234 | 10 | 3 | 165 | 3 | 8 | 507 | 14 | 11 |
| s2 | calibration | b3 | TT | 0.9759 | 0.9929 | 0.9749 | 0.9774 | 0.9762 | 0.0295 | 0.0249 | 234 | 10 | 3 | 168 | 4 | 5 | 506 | 13 | 12 |
| s2 | calibration | argmax | CC | 0.9705 | 0.9929 | 0.9540 | 0.9899 | 0.9716 | 0.0844 | 0.0348 | 235 | 22 | 2 | 169 | 10 | 4 | 510 | 23 | 8 |
| s2 | calibration | argmax | TC | 0.9715 | 0.9929 | 0.9471 | 0.9899 | 0.9681 | 0.0844 | 0.0348 | 235 | 22 | 2 | 170 | 10 | 3 | 512 | 24 | 6 |
| s2 | calibration | argmax | CT | 0.9556 | 0.9929 | 0.9359 | 0.9899 | 0.9621 | 0.1899 | 0.0448 | 233 | 49 | 4 | 169 | 8 | 4 | 506 | 35 | 12 |
| s2 | calibration | argmax | TT | 0.9567 | 0.9929 | 0.9292 | 0.9899 | 0.9586 | 0.1899 | 0.0448 | 233 | 49 | 4 | 170 | 8 | 3 | 508 | 36 | 10 |

## Cell contrasts (paired bootstrap; native clips by session, streams by stream)

| decoding | seed | half | contrast | multi segment F1@50 | native macro-F1 | multi boundary recall | multi boundary F1 | native cuts per clip | same-class cut rate |
|---|---|---|---|---|---|---|---|---|---|
| b3 | s0 | selection | TC - CC | +0.0036 [-0.0085, +0.0182] | +0.0000 [+0.0000, +0.0000] | +0.0183 [+0.0027, +0.0356] | +0.0045 [-0.0089, +0.0193] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| b3 | s0 | selection | CT - CC | -0.0058 [-0.0189, +0.0068] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0025 [-0.0123, +0.0077] | +0.0035 [-0.0244, +0.0308] | +0.0000 [-0.0127, +0.0124] |
| b3 | s0 | selection | TT - CC | -0.0061 [-0.0241, +0.0131] | +0.0000 [+0.0000, +0.0000] | +0.0157 [+0.0025, +0.0329] | -0.0005 [-0.0165, +0.0167] | +0.0035 [-0.0244, +0.0308] | +0.0000 [-0.0127, +0.0124] |
| b3 | s0 | selection | TT - TC - CT + CC | -0.0039 [-0.0140, +0.0047] | +0.0000 [+0.0000, +0.0000] | -0.0026 [-0.0082, +0.0000] | -0.0025 [-0.0100, +0.0047] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| argmax | s0 | selection | TC - CC | -0.0021 [-0.0097, +0.0066] | +0.0000 [+0.0000, +0.0000] | +0.0105 [+0.0025, +0.0216] | +0.0020 [-0.0091, +0.0145] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| argmax | s0 | selection | CT - CC | -0.0121 [-0.0252, -0.0016] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0109 [-0.0226, +0.0000] | +0.0528 [+0.0153, +0.0912] | +0.0043 [-0.0091, +0.0205] |
| argmax | s0 | selection | TT - CC | -0.0141 [-0.0296, -0.0010] | +0.0000 [+0.0000, +0.0000] | +0.0105 [+0.0025, +0.0216] | -0.0088 [-0.0235, +0.0046] | +0.0528 [+0.0153, +0.0912] | +0.0043 [-0.0091, +0.0205] |
| argmax | s0 | selection | TT - TC - CT + CC | +0.0001 [-0.0001, +0.0003] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0001 [-0.0002, +0.0003] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| b3 | s1 | selection | TC - CC | +0.0032 [-0.0077, +0.0157] | +0.0000 [+0.0000, +0.0000] | +0.0105 [-0.0026, +0.0258] | +0.0041 [-0.0058, +0.0146] | +0.0000 [+0.0000, +0.0000] | -0.0043 [-0.0140, +0.0000] |
| b3 | s1 | selection | CT - CC | +0.0064 [-0.0043, +0.0187] | +0.0000 [+0.0000, +0.0000] | -0.0026 [-0.0082, +0.0000] | +0.0111 [+0.0025, +0.0215] | +0.0035 [-0.0091, +0.0144] | -0.0087 [-0.0222, +0.0000] |
| b3 | s1 | selection | TT - CC | +0.0176 [+0.0040, +0.0338] | +0.0000 [+0.0000, +0.0000] | +0.0105 [-0.0026, +0.0258] | +0.0179 [+0.0062, +0.0311] | +0.0035 [-0.0091, +0.0144] | -0.0130 [-0.0293, +0.0000] |
| b3 | s1 | selection | TT - TC - CT + CC | +0.0079 [+0.0010, +0.0170] | +0.0000 [+0.0000, +0.0000] | +0.0026 [+0.0000, +0.0082] | +0.0026 [-0.0035, +0.0090] | +0.0000 [+0.0000, +0.0000] | +0.0000 [-0.0000, +0.0000] |
| argmax | s1 | selection | TC - CC | -0.0009 [-0.0060, +0.0043] | +0.0000 [+0.0000, +0.0000] | +0.0052 [+0.0000, +0.0133] | +0.0037 [-0.0022, +0.0113] | +0.0000 [+0.0000, +0.0000] | -0.0043 [-0.0140, +0.0000] |
| argmax | s1 | selection | CT - CC | -0.0097 [-0.0290, +0.0083] | +0.0083 [-0.0246, +0.0495] | +0.0000 [+0.0000, +0.0000] | -0.0077 [-0.0198, +0.0034] | +0.0599 [+0.0000, +0.1184] | -0.0043 [-0.0205, +0.0091] |
| argmax | s1 | selection | TT - CC | -0.0102 [-0.0302, +0.0092] | +0.0083 [-0.0246, +0.0495] | +0.0052 [+0.0000, +0.0133] | -0.0084 [-0.0217, +0.0045] | +0.0599 [+0.0000, +0.1184] | -0.0087 [-0.0270, +0.0083] |
| argmax | s1 | selection | TT - TC - CT + CC | +0.0004 [-0.0069, +0.0076] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0044 [-0.0114, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0000 [-0.0000, +0.0000] |
| b3 | s2 | selection | TC - CC | -0.0078 [-0.0140, -0.0028] | +0.0000 [+0.0000, +0.0000] | -0.0026 [-0.0115, +0.0054] | -0.0051 [-0.0146, +0.0041] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| b3 | s2 | selection | CT - CC | -0.0077 [-0.0180, -0.0009] | -0.0086 [-0.0435, +0.0187] | +0.0000 [+0.0000, +0.0000] | -0.0050 [-0.0121, +0.0000] | +0.0035 [-0.0449, +0.0625] | -0.0043 [-0.0136, +0.0000] |
| b3 | s2 | selection | TT - CC | -0.0155 [-0.0277, -0.0064] | -0.0086 [-0.0435, +0.0187] | -0.0026 [-0.0115, +0.0054] | -0.0100 [-0.0216, +0.0002] | +0.0035 [-0.0449, +0.0625] | -0.0043 [-0.0136, +0.0000] |
| b3 | s2 | selection | TT - TC - CT + CC | +0.0000 [-0.0000, +0.0002] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [-0.0000, +0.0002] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| argmax | s2 | selection | TC - CC | -0.0066 [-0.0134, -0.0007] | +0.0000 [+0.0000, +0.0000] | +0.0052 [+0.0000, +0.0132] | -0.0061 [-0.0157, +0.0038] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| argmax | s2 | selection | CT - CC | -0.0099 [-0.0224, +0.0038] | -0.0177 [-0.0504, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0055 [-0.0153, +0.0044] | +0.0282 [-0.0176, +0.0951] | +0.0130 [-0.0047, +0.0312] |
| argmax | s2 | selection | TT - CC | -0.0147 [-0.0295, +0.0009] | -0.0177 [-0.0504, +0.0000] | +0.0052 [+0.0000, +0.0132] | -0.0094 [-0.0218, +0.0045] | +0.0282 [-0.0176, +0.0951] | +0.0130 [-0.0047, +0.0312] |
| argmax | s2 | selection | TT - TC - CT + CC | +0.0018 [-0.0000, +0.0054] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0022 [-0.0000, +0.0070] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| b3 | s0 | calibration | TC - CC | -0.0039 [-0.0097, +0.0009] | +0.0000 [+0.0000, +0.0000] | -0.0075 [-0.0172, +0.0000] | -0.0050 [-0.0126, +0.0012] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| b3 | s0 | calibration | CT - CC | -0.0038 [-0.0091, -0.0000] | -0.0073 [-0.0278, +0.0000] | -0.0025 [-0.0078, +0.0000] | -0.0025 [-0.0063, +0.0000] | -0.0042 [-0.0138, +0.0000] | +0.0000 [-0.0140, +0.0155] |
| b3 | s0 | calibration | TT - CC | -0.0048 [-0.0109, +0.0000] | -0.0073 [-0.0278, +0.0000] | -0.0075 [-0.0172, +0.0000] | -0.0063 [-0.0143, +0.0000] | -0.0042 [-0.0138, +0.0000] | +0.0050 [+0.0000, +0.0165] |
| b3 | s0 | calibration | TT - TC - CT + CC | +0.0029 [+0.0000, +0.0078] | +0.0000 [+0.0000, +0.0000] | +0.0025 [+0.0000, +0.0078] | +0.0013 [+0.0000, +0.0039] | +0.0000 [+0.0000, +0.0000] | +0.0050 [+0.0000, +0.0154] |
| argmax | s0 | calibration | TC - CC | +0.0019 [-0.0066, +0.0117] | +0.0000 [+0.0000, +0.0000] | -0.0025 [-0.0078, +0.0000] | -0.0001 [-0.0090, +0.0085] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| argmax | s0 | calibration | CT - CC | -0.0139 [-0.0290, -0.0026] | -0.0068 [-0.0191, +0.0000] | -0.0025 [-0.0078, +0.0000] | -0.0131 [-0.0262, -0.0012] | +0.0591 [+0.0218, +0.1055] | +0.0149 [+0.0000, +0.0359] |
| argmax | s0 | calibration | TT - CC | -0.0176 [-0.0339, -0.0049] | -0.0068 [-0.0191, +0.0000] | -0.0050 [-0.0127, +0.0000] | -0.0178 [-0.0328, -0.0046] | +0.0591 [+0.0218, +0.1055] | +0.0149 [+0.0000, +0.0359] |
| argmax | s0 | calibration | TT - TC - CT + CC | -0.0055 [-0.0149, +0.0001] | +0.0000 [+0.0000, +0.0000] | -0.0000 [-0.0000, +0.0000] | -0.0047 [-0.0120, +0.0002] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| b3 | s1 | calibration | TC - CC | -0.0048 [-0.0119, +0.0010] | +0.0000 [+0.0000, +0.0000] | -0.0025 [-0.0130, +0.0078] | -0.0037 [-0.0125, +0.0049] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| b3 | s1 | calibration | CT - CC | -0.0058 [-0.0161, +0.0009] | -0.0139 [-0.0382, +0.0000] | -0.0025 [-0.0078, +0.0000] | -0.0025 [-0.0090, +0.0025] | +0.0042 [+0.0000, +0.0138] | +0.0000 [-0.0198, +0.0219] |
| b3 | s1 | calibration | TT - CC | -0.0105 [-0.0225, -0.0010] | -0.0139 [-0.0382, +0.0000] | -0.0025 [-0.0130, +0.0078] | -0.0061 [-0.0165, +0.0038] | +0.0042 [+0.0000, +0.0138] | +0.0100 [+0.0000, +0.0273] |
| b3 | s1 | calibration | TT - TC - CT + CC | +0.0001 [-0.0027, +0.0028] | +0.0000 [+0.0000, +0.0000] | +0.0025 [+0.0000, +0.0078] | +0.0001 [-0.0036, +0.0037] | +0.0000 [+0.0000, +0.0000] | +0.0100 [+0.0000, +0.0256] |
| argmax | s1 | calibration | TC - CC | -0.0009 [-0.0045, +0.0019] | +0.0000 [+0.0000, +0.0000] | +0.0025 [+0.0000, +0.0078] | +0.0013 [-0.0047, +0.0082] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| argmax | s1 | calibration | CT - CC | -0.0175 [-0.0359, -0.0017] | +0.0066 [+0.0000, +0.0229] | -0.0025 [-0.0078, +0.0000] | -0.0154 [-0.0290, -0.0025] | +0.0844 [+0.0286, +0.1632] | +0.0100 [-0.0093, +0.0323] |
| argmax | s1 | calibration | TT - CC | -0.0184 [-0.0373, -0.0019] | +0.0066 [+0.0000, +0.0229] | +0.0000 [-0.0074, +0.0075] | -0.0141 [-0.0290, -0.0001] | +0.0844 [+0.0286, +0.1632] | +0.0100 [-0.0093, +0.0323] |
| argmax | s1 | calibration | TT - TC - CT + CC | +0.0000 [-0.0001, +0.0002] | +0.0000 [+0.0000, +0.0000] | +0.0000 [-0.0000, +0.0000] | +0.0000 [-0.0002, +0.0002] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| b3 | s2 | calibration | TC - CC | -0.0029 [-0.0105, +0.0050] | +0.0000 [+0.0000, +0.0000] | -0.0101 [-0.0263, +0.0050] | -0.0088 [-0.0194, +0.0012] | +0.0000 [+0.0000, +0.0000] | +0.0000 [-0.0152, +0.0152] |
| b3 | s2 | calibration | CT - CC | -0.0038 [-0.0088, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0025 [-0.0066, +0.0000] | +0.0127 [+0.0000, +0.0274] | +0.0100 [+0.0000, +0.0263] |
| b3 | s2 | calibration | TT - CC | -0.0039 [-0.0128, +0.0046] | +0.0000 [+0.0000, +0.0000] | -0.0075 [-0.0233, +0.0070] | -0.0075 [-0.0178, +0.0023] | +0.0127 [+0.0000, +0.0274] | +0.0050 [-0.0106, +0.0240] |
| b3 | s2 | calibration | TT - TC - CT + CC | +0.0029 [-0.0000, +0.0066] | +0.0000 [+0.0000, +0.0000] | +0.0025 [+0.0000, +0.0078] | +0.0037 [+0.0000, +0.0086] | +0.0000 [+0.0000, +0.0000] | -0.0050 [-0.0175, +0.0000] |
| argmax | s2 | calibration | TC - CC | +0.0010 [-0.0037, +0.0060] | +0.0000 [+0.0000, +0.0000] | +0.0000 [-0.0074, +0.0076] | -0.0036 [-0.0117, +0.0038] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| argmax | s2 | calibration | CT - CC | -0.0149 [-0.0295, -0.0037] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0095 [-0.0198, -0.0023] | +0.1055 [+0.0332, +0.2028] | +0.0100 [-0.0094, +0.0306] |
| argmax | s2 | calibration | TT - CC | -0.0138 [-0.0295, -0.0018] | +0.0000 [+0.0000, +0.0000] | +0.0000 [-0.0074, +0.0076] | -0.0130 [-0.0257, -0.0019] | +0.1055 [+0.0332, +0.2028] | +0.0100 [-0.0094, +0.0306] |
| argmax | s2 | calibration | TT - TC - CT + CC | +0.0000 [-0.0001, +0.0002] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0001 [-0.0000, +0.0003] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |

## Cell contrast verdicts over seeds

| decoding | contrast | metric | selection | calibration |
|---|---|---|---|---|
| b3 | TC - CC | multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| b3 | TC - CC | native macro-F1 | no consistent difference detected | no consistent difference detected |
| b3 | TC - CC | native cuts per clip | no consistent difference detected | no consistent difference detected |
| b3 | TC - CC | multi boundary recall | no consistent difference detected | no consistent difference detected |
| b3 | TC - CC | multi boundary F1 | no consistent difference detected | no consistent difference detected |
| b3 | TC - CC | same-class cut rate | no consistent difference detected | no consistent difference detected |
| b3 | CT - CC | multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| b3 | CT - CC | native macro-F1 | no consistent difference detected | no consistent difference detected |
| b3 | CT - CC | native cuts per clip | no consistent difference detected | no consistent difference detected |
| b3 | CT - CC | multi boundary recall | no consistent difference detected | no consistent difference detected |
| b3 | CT - CC | multi boundary F1 | no consistent difference detected | no consistent difference detected |
| b3 | CT - CC | same-class cut rate | no consistent difference detected | no consistent difference detected |
| b3 | TT - CC | multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| b3 | TT - CC | native macro-F1 | no consistent difference detected | no consistent difference detected |
| b3 | TT - CC | native cuts per clip | no consistent difference detected | no consistent difference detected |
| b3 | TT - CC | multi boundary recall | no consistent difference detected | no consistent difference detected |
| b3 | TT - CC | multi boundary F1 | no consistent difference detected | no consistent difference detected |
| b3 | TT - CC | same-class cut rate | no consistent difference detected | no consistent difference detected |
| b3 | TT - TC - CT + CC | multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| b3 | TT - TC - CT + CC | native macro-F1 | no consistent difference detected | no consistent difference detected |
| b3 | TT - TC - CT + CC | native cuts per clip | no consistent difference detected | no consistent difference detected |
| b3 | TT - TC - CT + CC | multi boundary recall | no consistent difference detected | no consistent difference detected |
| b3 | TT - TC - CT + CC | multi boundary F1 | no consistent difference detected | no consistent difference detected |
| b3 | TT - TC - CT + CC | same-class cut rate | no consistent difference detected | no consistent difference detected |
| argmax | TC - CC | multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| argmax | TC - CC | native macro-F1 | no consistent difference detected | no consistent difference detected |
| argmax | TC - CC | native cuts per clip | no consistent difference detected | no consistent difference detected |
| argmax | TC - CC | multi boundary recall | no consistent difference detected | no consistent difference detected |
| argmax | TC - CC | multi boundary F1 | no consistent difference detected | no consistent difference detected |
| argmax | TC - CC | same-class cut rate | no consistent difference detected | no consistent difference detected |
| argmax | CT - CC | multi segment F1@50 | no consistent difference detected | consistent negative |
| argmax | CT - CC | native macro-F1 | no consistent difference detected | no consistent difference detected |
| argmax | CT - CC | native cuts per clip | no consistent difference detected | consistent positive |
| argmax | CT - CC | multi boundary recall | no consistent difference detected | no consistent difference detected |
| argmax | CT - CC | multi boundary F1 | no consistent difference detected | consistent negative |
| argmax | CT - CC | same-class cut rate | no consistent difference detected | no consistent difference detected |
| argmax | TT - CC | multi segment F1@50 | no consistent difference detected | consistent negative |
| argmax | TT - CC | native macro-F1 | no consistent difference detected | no consistent difference detected |
| argmax | TT - CC | native cuts per clip | no consistent difference detected | consistent positive |
| argmax | TT - CC | multi boundary recall | no consistent difference detected | no consistent difference detected |
| argmax | TT - CC | multi boundary F1 | no consistent difference detected | consistent negative |
| argmax | TT - CC | same-class cut rate | no consistent difference detected | no consistent difference detected |
| argmax | TT - TC - CT + CC | multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| argmax | TT - TC - CT + CC | native macro-F1 | no consistent difference detected | no consistent difference detected |
| argmax | TT - TC - CT + CC | native cuts per clip | no consistent difference detected | no consistent difference detected |
| argmax | TT - TC - CT + CC | multi boundary recall | no consistent difference detected | no consistent difference detected |
| argmax | TT - TC - CT + CC | multi boundary F1 | no consistent difference detected | no consistent difference detected |
| argmax | TT - TC - CT + CC | same-class cut rate | no consistent difference detected | no consistent difference detected |

## Overlapping-anchor support

| half | subset | population | context | groups | observations | pairs | physical_frames | recordings | sessions | few_units | selected_output_anchor at least one | selected_output_anchor both | selected_output_anchor neither |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| selection | main | native | all | 7418 | 17311 | 12664 | 7418 | 284 | 36 | False | 2313 | 0 | 10351 |
| selection | main | native | both pure | 7418 | 17311 | 12664 | 7418 | 284 | 36 | False | 2313 | 0 | 10351 |
| selection | main | native | at least one mixed | 0 | 0 | 0 | 0 | 0 | 0 | True | 0 | 0 | 0 |
| selection | main | pairs | all | 2631 | 6541 | 5298 | 2631 | 274 | 36 | False | 1190 | 0 | 4108 |
| selection | main | pairs | both pure | 2005 | 4918 | 3879 | 2005 | 221 | 36 | False | 905 | 0 | 2974 |
| selection | main | pairs | at least one mixed | 750 | 1873 | 1419 | 750 | 140 | 33 | False | 285 | 0 | 1134 |
| selection | main | multi | all | 6279 | 13640 | 8468 | 4636 | 231 | 36 | False | 1005 | 0 | 7463 |
| selection | main | multi | both pure | 3028 | 6701 | 4327 | 2580 | 186 | 34 | False | 675 | 0 | 3652 |
| selection | main | multi | at least one mixed | 3392 | 7225 | 4141 | 2788 | 212 | 36 | False | 330 | 0 | 3811 |
| selection | main, equal marker | native | all | 6677 | 14681 | 9535 | 6677 | 284 | 36 | False | 0 | 0 | 9535 |
| selection | main, equal marker | native | both pure | 6677 | 14681 | 9535 | 6677 | 284 | 36 | False | 0 | 0 | 9535 |
| selection | main, equal marker | native | at least one mixed | 0 | 0 | 0 | 0 | 0 | 0 | True | 0 | 0 | 0 |
| selection | main, equal marker | pairs | all | 2259 | 5185 | 3680 | 2259 | 274 | 36 | False | 0 | 0 | 3680 |
| selection | main, equal marker | pairs | both pure | 1676 | 3812 | 2642 | 1676 | 221 | 36 | False | 0 | 0 | 2642 |
| selection | main, equal marker | pairs | at least one mixed | 655 | 1519 | 1038 | 655 | 140 | 33 | False | 0 | 0 | 1038 |
| selection | main, equal marker | multi | all | 5932 | 12433 | 7095 | 4441 | 231 | 36 | False | 0 | 0 | 7095 |
| selection | main, equal marker | multi | both pure | 2731 | 5792 | 3400 | 2361 | 186 | 34 | False | 0 | 0 | 3400 |
| selection | main, equal marker | multi | at least one mixed | 3283 | 6809 | 3695 | 2725 | 212 | 36 | False | 0 | 0 | 3695 |
| selection | main, changed marker | native | all | 1889 | 5018 | 3129 | 1889 | 284 | 36 | False | 2313 | 0 | 816 |
| selection | main, changed marker | native | both pure | 1889 | 5018 | 3129 | 1889 | 284 | 36 | False | 2313 | 0 | 816 |
| selection | main, changed marker | native | at least one mixed | 0 | 0 | 0 | 0 | 0 | 0 | True | 0 | 0 | 0 |
| selection | main, changed marker | pairs | all | 984 | 2602 | 1618 | 984 | 274 | 36 | False | 1190 | 0 | 428 |
| selection | main, changed marker | pairs | both pure | 777 | 2014 | 1237 | 777 | 221 | 36 | False | 905 | 0 | 332 |
| selection | main, changed marker | pairs | at least one mixed | 237 | 618 | 381 | 237 | 87 | 29 | False | 285 | 0 | 96 |
| selection | main, changed marker | multi | all | 860 | 2233 | 1373 | 791 | 141 | 34 | False | 1005 | 0 | 368 |
| selection | main, changed marker | multi | both pure | 612 | 1539 | 927 | 583 | 109 | 32 | False | 675 | 0 | 252 |
| selection | main, changed marker | multi | at least one mixed | 286 | 732 | 446 | 270 | 69 | 28 | False | 330 | 0 | 116 |
| selection | broad | native | all | 9571 | 22393 | 16722 | 9571 | 284 | 36 | False | 2417 | 0 | 14305 |
| selection | broad | native | both pure | 9571 | 22393 | 16722 | 9571 | 284 | 36 | False | 2417 | 0 | 14305 |
| selection | broad | native | at least one mixed | 0 | 0 | 0 | 0 | 0 | 0 | True | 0 | 0 | 0 |
| selection | broad | pairs | all | 3111 | 7945 | 6873 | 3111 | 274 | 36 | False | 1244 | 0 | 5629 |
| selection | broad | pairs | both pure | 2362 | 5928 | 4946 | 2362 | 221 | 36 | False | 943 | 0 | 4003 |
| selection | broad | pairs | at least one mixed | 931 | 2411 | 1927 | 931 | 140 | 33 | False | 301 | 0 | 1626 |
| selection | broad | multi | all | 8664 | 18817 | 11834 | 5937 | 231 | 36 | False | 1059 | 0 | 10775 |
| selection | broad | multi | both pure | 4032 | 8924 | 5833 | 3288 | 189 | 34 | False | 707 | 0 | 5126 |
| selection | broad | multi | at least one mixed | 4838 | 10340 | 6001 | 3734 | 213 | 36 | False | 352 | 0 | 5649 |
| calibration | main | native | all | 6316 | 14758 | 10867 | 6316 | 237 | 36 | False | 1956 | 0 | 8911 |
| calibration | main | native | both pure | 6316 | 14758 | 10867 | 6316 | 237 | 36 | False | 1956 | 0 | 8911 |
| calibration | main | native | at least one mixed | 0 | 0 | 0 | 0 | 0 | 0 | True | 0 | 0 | 0 |
| calibration | main | pairs | all | 2253 | 5580 | 4492 | 2253 | 230 | 36 | False | 986 | 0 | 3506 |
| calibration | main | pairs | both pure | 1732 | 4285 | 3439 | 1732 | 197 | 34 | False | 800 | 0 | 2639 |
| calibration | main | pairs | at least one mixed | 626 | 1509 | 1053 | 626 | 116 | 31 | False | 186 | 0 | 867 |
| calibration | main | multi | all | 6321 | 13722 | 8496 | 4660 | 213 | 35 | False | 1020 | 0 | 7476 |
| calibration | main | multi | both pure | 2894 | 6482 | 4288 | 2465 | 171 | 34 | False | 720 | 0 | 3568 |
| calibration | main | multi | at least one mixed | 3559 | 7508 | 4208 | 2909 | 202 | 34 | False | 300 | 0 | 3908 |
| calibration | main, equal marker | native | all | 5689 | 12514 | 8168 | 5689 | 237 | 36 | False | 0 | 0 | 8168 |
| calibration | main, equal marker | native | both pure | 5689 | 12514 | 8168 | 5689 | 237 | 36 | False | 0 | 0 | 8168 |
| calibration | main, equal marker | native | at least one mixed | 0 | 0 | 0 | 0 | 0 | 0 | True | 0 | 0 | 0 |
| calibration | main, equal marker | pairs | all | 1937 | 4447 | 3155 | 1937 | 230 | 36 | False | 0 | 0 | 3155 |
| calibration | main, equal marker | pairs | both pure | 1432 | 3296 | 2346 | 1432 | 197 | 34 | False | 0 | 0 | 2346 |
| calibration | main, equal marker | pairs | at least one mixed | 560 | 1265 | 809 | 560 | 116 | 31 | False | 0 | 0 | 809 |
| calibration | main, equal marker | multi | all | 5983 | 12530 | 7126 | 4450 | 213 | 35 | False | 0 | 0 | 7126 |
| calibration | main, equal marker | multi | both pure | 2600 | 5559 | 3324 | 2229 | 171 | 34 | False | 0 | 0 | 3324 |
| calibration | main, equal marker | multi | at least one mixed | 3460 | 7129 | 3802 | 2838 | 202 | 34 | False | 0 | 0 | 3802 |
| calibration | main, changed marker | native | all | 1617 | 4316 | 2699 | 1617 | 237 | 36 | False | 1956 | 0 | 743 |
| calibration | main, changed marker | native | both pure | 1617 | 4316 | 2699 | 1617 | 237 | 36 | False | 1956 | 0 | 743 |
| calibration | main, changed marker | native | at least one mixed | 0 | 0 | 0 | 0 | 0 | 0 | True | 0 | 0 | 0 |
| calibration | main, changed marker | pairs | all | 817 | 2154 | 1337 | 817 | 230 | 36 | False | 986 | 0 | 351 |
| calibration | main, changed marker | pairs | both pure | 689 | 1782 | 1093 | 689 | 197 | 34 | False | 800 | 0 | 293 |
| calibration | main, changed marker | pairs | at least one mixed | 160 | 404 | 244 | 160 | 58 | 29 | False | 186 | 0 | 58 |
| calibration | main, changed marker | multi | all | 854 | 2224 | 1370 | 814 | 136 | 32 | False | 1020 | 0 | 350 |
| calibration | main, changed marker | multi | both pure | 629 | 1593 | 964 | 611 | 107 | 32 | False | 720 | 0 | 244 |
| calibration | main, changed marker | multi | at least one mixed | 259 | 665 | 406 | 251 | 68 | 24 | False | 300 | 0 | 106 |
| calibration | broad | native | all | 8171 | 19134 | 14358 | 8171 | 237 | 36 | False | 2042 | 0 | 12316 |
| calibration | broad | native | both pure | 8171 | 19134 | 14358 | 8171 | 237 | 36 | False | 2042 | 0 | 12316 |
| calibration | broad | native | at least one mixed | 0 | 0 | 0 | 0 | 0 | 0 | True | 0 | 0 | 0 |
| calibration | broad | pairs | all | 2685 | 6811 | 5820 | 2685 | 230 | 36 | False | 1046 | 0 | 4774 |
| calibration | broad | pairs | both pure | 2035 | 5149 | 4356 | 2035 | 197 | 34 | False | 844 | 0 | 3512 |
| calibration | broad | pairs | at least one mixed | 804 | 2001 | 1464 | 804 | 116 | 31 | False | 202 | 0 | 1262 |
| calibration | broad | multi | all | 8725 | 18934 | 11872 | 5956 | 213 | 35 | False | 1080 | 0 | 10792 |
| calibration | broad | multi | both pure | 3819 | 8550 | 5714 | 3122 | 176 | 34 | False | 760 | 0 | 4954 |
| calibration | broad | multi | at least one mixed | 5110 | 10837 | 6158 | 3929 | 204 | 35 | False | 320 | 0 | 5838 |

## Main-subset distributions

| half | population | distribution | values |
|---|---|---|---|
| selection | native | anchor pairs | {"1-1": 596, "2-1": 122, "2-2": 596, "3-1": 576, "3-2": 122, "3-3": 596, "4-1": 120, "4-2": 576, "4-3": 122, "4-4": 596, "5-1": 106, "5-2": 120, "5-3": 576, "5-4": 122, "5-5": 596, "6-1": 1990, "6-2": 106, "6-3": 120, "6-4": 576, "6-5": 122, "6-6": 596, "7-1": 102, "7-2": 1990, "7-3": 106, "7-4": 120, "7-5": 576, "7-6": 122, "7-7": 596} |
| selection | native | marker-relative positions | {"-1": 990, "-10": 247, "-11": 455, "-12": 127, "-13": 411, "-2": 945, "-3": 825, "-4": 2917, "-5": 817, "-6": 2811, "-7": 615, "-8": 370, "-9": 612, "0": 1323, "1": 816, "10": 578, "11": 20, "12": 586, "2": 1344, "3": 677, "4": 3010, "5": 624, "6": 3038, "7": 582, "8": 568, "9": 20} |
| selection | native | marker triples | {"000/000": 9535, "000/001": 599, "000/010": 578, "000/100": 411, "001/000": 391, "010/000": 745, "100/000": 405} |
| selection | native | centre distance | {"10": 5834, "13": 205, "14": 28, "15": 356, "18": 10, "19": 65, "20": 42, "23": 10, "25": 7, "3": 399, "4": 455, "5": 4858, "8": 35, "9": 360} |
| selection | native | boundary distance | {"none": 12664} |
| selection | pairs | anchor pairs | {"1-1": 274, "2-1": 50, "2-2": 274, "3-1": 350, "3-2": 50, "3-3": 274, "4-1": 50, "4-2": 350, "4-3": 50, "4-4": 274, "5-1": 74, "5-2": 50, "5-3": 350, "5-4": 50, "5-5": 274, "6-1": 431, "6-2": 74, "6-3": 50, "6-4": 350, "6-5": 50, "6-6": 274, "7-1": 46, "7-2": 431, "7-3": 74, "7-4": 50, "7-5": 350, "7-6": 50, "7-7": 274} |
| selection | pairs | marker-relative positions | {"-1": 499, "-10": 134, "-11": 199, "-12": 88, "-13": 175, "-2": 512, "-3": 403, "-4": 924, "-5": 403, "-6": 874, "-7": 291, "-8": 190, "-9": 270, "0": 691, "1": 428, "10": 292, "12": 296, "2": 703, "3": 353, "4": 970, "5": 330, "6": 985, "7": 296, "8": 290} |
| selection | pairs | marker triples | {"000/000": 3680, "000/001": 299, "000/010": 312, "000/100": 224, "001/000": 200, "010/000": 379, "100/000": 204} |
| selection | pairs | centre distance | {"10": 1868, "13": 84, "14": 8, "15": 188, "18": 2, "19": 27, "20": 24, "23": 2, "25": 2, "3": 161, "4": 175, "5": 2607, "9": 150} |
| selection | pairs | boundary distance | {"0-5": 998, "6-10": 574, ">10": 1078, "none": 2648} |
| selection | multi | anchor pairs | {"1-1": 240, "2-1": 50, "2-2": 240, "3-1": 284, "3-2": 50, "3-3": 240, "4-1": 24, "4-2": 284, "4-3": 50, "4-4": 240, "5-1": 54, "5-2": 24, "5-3": 284, "5-4": 50, "5-5": 240, "6-1": 2380, "6-2": 54, "6-3": 24, "6-4": 284, "6-5": 50, "6-6": 240, "7-1": 50, "7-2": 2380, "7-3": 54, "7-4": 24, "7-5": 284, "7-6": 50, "7-7": 240} |
| selection | multi | marker-relative positions | {"-1": 432, "-10": 68, "-11": 196, "-12": 44, "-13": 153, "-2": 429, "-3": 337, "-4": 2792, "-5": 337, "-6": 2742, "-7": 229, "-8": 122, "-9": 271, "0": 573, "1": 368, "10": 240, "12": 240, "2": 605, "3": 287, "4": 2853, "5": 262, "6": 2876, "7": 240, "8": 240} |
| selection | multi | marker triples | {"000/000": 7095, "000/001": 265, "000/010": 262, "000/100": 201, "001/000": 167, "010/000": 311, "100/000": 167} |
| selection | multi | centre distance | {"10": 5531, "13": 48, "15": 174, "19": 25, "3": 84, "4": 175, "5": 2281, "9": 150} |
| selection | multi | boundary distance | {"0-5": 2531, "6-10": 1550, ">10": 4331, "none": 56} |
| calibration | native | anchor pairs | {"1-1": 491, "2-1": 94, "2-2": 491, "3-1": 526, "3-2": 94, "3-3": 491, "4-1": 98, "4-2": 526, "4-3": 94, "4-4": 491, "5-1": 90, "5-2": 98, "5-3": 526, "5-4": 94, "5-5": 491, "6-1": 1750, "6-2": 90, "6-3": 98, "6-4": 526, "6-5": 94, "6-6": 491, "7-1": 74, "7-2": 1750, "7-3": 90, "7-4": 98, "7-5": 526, "7-6": 94, "7-7": 491} |
| calibration | native | marker-relative positions | {"-1": 866, "-10": 195, "-11": 391, "-12": 117, "-13": 315, "-2": 800, "-3": 718, "-4": 2497, "-5": 712, "-6": 2415, "-7": 541, "-8": 286, "-9": 517, "0": 1090, "1": 743, "10": 494, "11": 24, "12": 503, "2": 1103, "3": 611, "4": 2587, "5": 569, "6": 2613, "7": 519, "8": 484, "9": 24} |
| calibration | native | marker triples | {"000/000": 8168, "000/001": 515, "000/010": 489, "000/100": 374, "001/000": 351, "010/000": 601, "100/000": 369} |
| calibration | native | centre distance | {"10": 5074, "13": 131, "14": 36, "15": 348, "18": 10, "19": 61, "20": 43, "23": 10, "25": 6, "3": 259, "4": 322, "5": 4272, "8": 35, "9": 260} |
| calibration | native | boundary distance | {"none": 10867} |
| calibration | pairs | anchor pairs | {"1-1": 230, "2-1": 52, "2-2": 230, "3-1": 276, "3-2": 52, "3-3": 230, "4-1": 44, "4-2": 276, "4-3": 52, "4-4": 230, "5-1": 68, "5-2": 44, "5-3": 276, "5-4": 52, "5-5": 230, "6-1": 378, "6-2": 68, "6-3": 44, "6-4": 276, "6-5": 52, "6-6": 230, "7-1": 54, "7-2": 378, "7-3": 68, "7-4": 44, "7-5": 276, "7-6": 52, "7-7": 230} |
| calibration | pairs | marker-relative positions | {"-1": 420, "-10": 89, "-11": 187, "-12": 50, "-13": 178, "-2": 415, "-3": 342, "-4": 799, "-5": 342, "-6": 747, "-7": 249, "-8": 149, "-9": 264, "0": 566, "1": 351, "10": 243, "12": 247, "2": 600, "3": 289, "4": 841, "5": 265, "6": 863, "7": 247, "8": 241} |
| calibration | pairs | marker triples | {"000/000": 3155, "000/001": 256, "000/010": 252, "000/100": 183, "001/000": 164, "010/000": 314, "100/000": 168} |
| calibration | pairs | centre distance | {"10": 1583, "13": 72, "14": 8, "15": 170, "18": 2, "19": 31, "20": 15, "23": 2, "25": 2, "3": 133, "4": 182, "5": 2136, "9": 156} |
| calibration | pairs | boundary distance | {"0-5": 676, "6-10": 460, ">10": 1168, "none": 2188} |
| calibration | multi | anchor pairs | {"1-1": 240, "2-1": 30, "2-2": 240, "3-1": 298, "3-2": 30, "3-3": 240, "4-1": 48, "4-2": 298, "4-3": 30, "4-4": 240, "5-1": 60, "5-2": 48, "5-3": 298, "5-4": 30, "5-5": 240, "6-1": 2372, "6-2": 60, "6-3": 48, "6-4": 298, "6-5": 30, "6-6": 240, "7-1": 30, "7-2": 2372, "7-3": 60, "7-4": 48, "7-5": 298, "7-6": 30, "7-7": 240} |
| calibration | multi | marker-relative positions | {"-1": 419, "-10": 106, "-11": 182, "-12": 58, "-13": 153, "-2": 433, "-3": 314, "-4": 2808, "-5": 314, "-6": 2778, "-7": 218, "-8": 166, "-9": 227, "0": 601, "1": 350, "10": 240, "12": 240, "2": 632, "3": 284, "4": 2860, "5": 269, "6": 2860, "7": 240, "8": 240} |
| calibration | multi | marker triples | {"000/000": 7126, "000/001": 255, "000/010": 269, "000/100": 186, "001/000": 164, "010/000": 332, "100/000": 164} |
| calibration | multi | centre distance | {"10": 5631, "13": 96, "15": 149, "19": 15, "3": 168, "4": 105, "5": 2242, "9": 90} |
| calibration | multi | boundary distance | {"0-5": 2504, "6-10": 1632, ">10": 4360} |

## Overlapping-anchor rates (group-weighted)

| readout | half | subset | population | context | pairs | disagreeing pairs | disagreement | js_mean | both correct | one correct | both wrong | disagreeing sessions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aligned_dense_s0 | selection | main | native | all | 12664 | 83 | 0.0094 | 0.0043 | 0.9766 | 0.0094 | 0.0140 | 10 |
| aligned_dense_s0 | selection | main | native | both pure | 12664 | 83 | 0.0094 | 0.0043 | 0.9766 | 0.0094 | 0.0140 | 10 |
| aligned_dense_s0 | selection | main | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| aligned_dense_s0 | selection | main | pairs | all | 5298 | 256 | 0.0444 | 0.0271 | 0.9339 | 0.0441 | 0.0221 | 27 |
| aligned_dense_s0 | selection | main | pairs | both pure | 3879 | 32 | 0.0077 | 0.0030 | 0.9791 | 0.0077 | 0.0132 | 5 |
| aligned_dense_s0 | selection | main | pairs | at least one mixed | 1419 | 224 | 0.1353 | 0.0876 | 0.8172 | 0.1340 | 0.0488 | 26 |
| aligned_dense_s0 | selection | main | multi | all | 8468 | 779 | 0.1069 | 0.0717 | 0.8615 | 0.1039 | 0.0345 | 36 |
| aligned_dense_s0 | selection | main | multi | both pure | 4327 | 86 | 0.0244 | 0.0102 | 0.9431 | 0.0244 | 0.0325 | 7 |
| aligned_dense_s0 | selection | main | multi | at least one mixed | 4141 | 693 | 0.1763 | 0.1237 | 0.7926 | 0.1708 | 0.0366 | 36 |
| aligned_dense_s0 | selection | main, equal marker | native | all | 9535 | 78 | 0.0105 | 0.0047 | 0.9748 | 0.0105 | 0.0147 | 10 |
| aligned_dense_s0 | selection | main, equal marker | native | both pure | 9535 | 78 | 0.0105 | 0.0047 | 0.9748 | 0.0105 | 0.0147 | 10 |
| aligned_dense_s0 | selection | main, equal marker | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| aligned_dense_s0 | selection | main, equal marker | pairs | all | 3680 | 219 | 0.0502 | 0.0309 | 0.9231 | 0.0498 | 0.0271 | 27 |
| aligned_dense_s0 | selection | main, equal marker | pairs | both pure | 2642 | 27 | 0.0094 | 0.0034 | 0.9777 | 0.0094 | 0.0128 | 5 |
| aligned_dense_s0 | selection | main, equal marker | pairs | at least one mixed | 1038 | 192 | 0.1491 | 0.0983 | 0.7887 | 0.1475 | 0.0637 | 26 |
| aligned_dense_s0 | selection | main, equal marker | multi | all | 7095 | 757 | 0.1124 | 0.0755 | 0.8558 | 0.1092 | 0.0350 | 36 |
| aligned_dense_s0 | selection | main, equal marker | multi | both pure | 3400 | 82 | 0.0271 | 0.0112 | 0.9430 | 0.0271 | 0.0299 | 7 |
| aligned_dense_s0 | selection | main, equal marker | multi | at least one mixed | 3695 | 675 | 0.1806 | 0.1273 | 0.7859 | 0.1749 | 0.0392 | 36 |
| aligned_dense_s0 | selection | main, changed marker | native | all | 3129 | 5 | 0.0013 | 0.0009 | 0.9878 | 0.0013 | 0.0109 | 3 |
| aligned_dense_s0 | selection | main, changed marker | native | both pure | 3129 | 5 | 0.0013 | 0.0009 | 0.9878 | 0.0013 | 0.0109 | 3 |
| aligned_dense_s0 | selection | main, changed marker | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| aligned_dense_s0 | selection | main, changed marker | pairs | all | 1618 | 37 | 0.0183 | 0.0088 | 0.9675 | 0.0183 | 0.0142 | 9 |
| aligned_dense_s0 | selection | main, changed marker | pairs | both pure | 1237 | 5 | 0.0032 | 0.0012 | 0.9826 | 0.0032 | 0.0142 | 2 |
| aligned_dense_s0 | selection | main, changed marker | pairs | at least one mixed | 381 | 32 | 0.0654 | 0.0328 | 0.9135 | 0.0654 | 0.0211 | 8 |
| aligned_dense_s0 | selection | main, changed marker | multi | all | 1373 | 22 | 0.0128 | 0.0070 | 0.9570 | 0.0128 | 0.0302 | 7 |
| aligned_dense_s0 | selection | main, changed marker | multi | both pure | 927 | 4 | 0.0033 | 0.0020 | 0.9575 | 0.0033 | 0.0392 | 1 |
| aligned_dense_s0 | selection | main, changed marker | multi | at least one mixed | 446 | 18 | 0.0315 | 0.0171 | 0.9545 | 0.0315 | 0.0140 | 7 |
| aligned_dense_s0 | selection | broad | native | all | 16722 | 133 | 0.0102 | 0.0061 | 0.9750 | 0.0102 | 0.0148 | 10 |
| aligned_dense_s0 | selection | broad | native | both pure | 16722 | 133 | 0.0102 | 0.0061 | 0.9750 | 0.0102 | 0.0148 | 10 |
| aligned_dense_s0 | selection | broad | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| aligned_dense_s0 | selection | broad | pairs | all | 6873 | 402 | 0.0506 | 0.0341 | 0.9268 | 0.0500 | 0.0231 | 31 |
| aligned_dense_s0 | selection | broad | pairs | both pure | 4946 | 53 | 0.0080 | 0.0044 | 0.9784 | 0.0080 | 0.0135 | 6 |
| aligned_dense_s0 | selection | broad | pairs | at least one mixed | 1927 | 349 | 0.1503 | 0.1055 | 0.8011 | 0.1485 | 0.0505 | 30 |
| aligned_dense_s0 | selection | broad | multi | all | 11834 | 1311 | 0.1212 | 0.0834 | 0.8476 | 0.1177 | 0.0347 | 36 |
| aligned_dense_s0 | selection | broad | multi | both pure | 5833 | 145 | 0.0279 | 0.0140 | 0.9406 | 0.0279 | 0.0315 | 8 |
| aligned_dense_s0 | selection | broad | multi | at least one mixed | 6001 | 1166 | 0.1945 | 0.1384 | 0.7742 | 0.1883 | 0.0375 | 36 |
| aligned_dense_s1 | selection | main | native | all | 12664 | 95 | 0.0097 | 0.0048 | 0.9741 | 0.0097 | 0.0162 | 9 |
| aligned_dense_s1 | selection | main | native | both pure | 12664 | 95 | 0.0097 | 0.0048 | 0.9741 | 0.0097 | 0.0162 | 9 |
| aligned_dense_s1 | selection | main | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| aligned_dense_s1 | selection | main | pairs | all | 5298 | 287 | 0.0474 | 0.0303 | 0.9296 | 0.0466 | 0.0237 | 28 |
| aligned_dense_s1 | selection | main | pairs | both pure | 3879 | 39 | 0.0090 | 0.0039 | 0.9779 | 0.0090 | 0.0131 | 6 |
| aligned_dense_s1 | selection | main | pairs | at least one mixed | 1419 | 248 | 0.1423 | 0.0964 | 0.8055 | 0.1396 | 0.0549 | 27 |
| aligned_dense_s1 | selection | main | multi | all | 8468 | 830 | 0.1146 | 0.0755 | 0.8674 | 0.1115 | 0.0211 | 35 |
| aligned_dense_s1 | selection | main | multi | both pure | 4327 | 89 | 0.0241 | 0.0091 | 0.9621 | 0.0241 | 0.0138 | 11 |
| aligned_dense_s1 | selection | main | multi | at least one mixed | 4141 | 741 | 0.1908 | 0.1318 | 0.7877 | 0.1849 | 0.0274 | 35 |
| aligned_dense_s1 | selection | main, equal marker | native | all | 9535 | 84 | 0.0104 | 0.0052 | 0.9725 | 0.0104 | 0.0172 | 8 |
| aligned_dense_s1 | selection | main, equal marker | native | both pure | 9535 | 84 | 0.0104 | 0.0052 | 0.9725 | 0.0104 | 0.0172 | 8 |
| aligned_dense_s1 | selection | main, equal marker | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| aligned_dense_s1 | selection | main, equal marker | pairs | all | 3680 | 241 | 0.0532 | 0.0348 | 0.9176 | 0.0523 | 0.0301 | 28 |
| aligned_dense_s1 | selection | main, equal marker | pairs | both pure | 2642 | 28 | 0.0089 | 0.0044 | 0.9765 | 0.0089 | 0.0145 | 5 |
| aligned_dense_s1 | selection | main, equal marker | pairs | at least one mixed | 1038 | 213 | 0.1606 | 0.1091 | 0.7729 | 0.1575 | 0.0696 | 27 |
| aligned_dense_s1 | selection | main, equal marker | multi | all | 7095 | 807 | 0.1208 | 0.0795 | 0.8613 | 0.1175 | 0.0213 | 35 |
| aligned_dense_s1 | selection | main, equal marker | multi | both pure | 3400 | 81 | 0.0267 | 0.0098 | 0.9618 | 0.0267 | 0.0115 | 11 |
| aligned_dense_s1 | selection | main, equal marker | multi | at least one mixed | 3695 | 726 | 0.1962 | 0.1356 | 0.7804 | 0.1901 | 0.0295 | 35 |
| aligned_dense_s1 | selection | main, changed marker | native | all | 3129 | 11 | 0.0037 | 0.0013 | 0.9846 | 0.0037 | 0.0116 | 5 |
| aligned_dense_s1 | selection | main, changed marker | native | both pure | 3129 | 11 | 0.0037 | 0.0013 | 0.9846 | 0.0037 | 0.0116 | 5 |
| aligned_dense_s1 | selection | main, changed marker | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| aligned_dense_s1 | selection | main, changed marker | pairs | all | 1618 | 46 | 0.0239 | 0.0104 | 0.9634 | 0.0239 | 0.0127 | 14 |
| aligned_dense_s1 | selection | main, changed marker | pairs | both pure | 1237 | 11 | 0.0084 | 0.0017 | 0.9820 | 0.0084 | 0.0097 | 4 |
| aligned_dense_s1 | selection | main, changed marker | pairs | at least one mixed | 381 | 35 | 0.0717 | 0.0380 | 0.8987 | 0.0717 | 0.0295 | 11 |
| aligned_dense_s1 | selection | main, changed marker | multi | all | 1373 | 23 | 0.0140 | 0.0090 | 0.9657 | 0.0140 | 0.0203 | 12 |
| aligned_dense_s1 | selection | main, changed marker | multi | both pure | 927 | 8 | 0.0074 | 0.0027 | 0.9690 | 0.0074 | 0.0237 | 5 |
| aligned_dense_s1 | selection | main, changed marker | multi | at least one mixed | 446 | 15 | 0.0262 | 0.0218 | 0.9633 | 0.0262 | 0.0105 | 8 |
| aligned_dense_s1 | selection | broad | native | all | 16722 | 161 | 0.0112 | 0.0074 | 0.9724 | 0.0112 | 0.0164 | 10 |
| aligned_dense_s1 | selection | broad | native | both pure | 16722 | 161 | 0.0112 | 0.0074 | 0.9724 | 0.0112 | 0.0164 | 10 |
| aligned_dense_s1 | selection | broad | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| aligned_dense_s1 | selection | broad | pairs | all | 6873 | 446 | 0.0535 | 0.0371 | 0.9226 | 0.0527 | 0.0247 | 31 |
| aligned_dense_s1 | selection | broad | pairs | both pure | 4946 | 63 | 0.0096 | 0.0059 | 0.9767 | 0.0096 | 0.0137 | 7 |
| aligned_dense_s1 | selection | broad | pairs | at least one mixed | 1927 | 383 | 0.1586 | 0.1123 | 0.7897 | 0.1557 | 0.0546 | 30 |
| aligned_dense_s1 | selection | broad | multi | all | 11834 | 1373 | 0.1286 | 0.0865 | 0.8545 | 0.1251 | 0.0204 | 35 |
| aligned_dense_s1 | selection | broad | multi | both pure | 5833 | 145 | 0.0274 | 0.0120 | 0.9599 | 0.0274 | 0.0128 | 12 |
| aligned_dense_s1 | selection | broad | multi | at least one mixed | 6001 | 1228 | 0.2083 | 0.1456 | 0.7715 | 0.2021 | 0.0264 | 35 |
| aligned_dense_s2 | selection | main | native | all | 12664 | 123 | 0.0121 | 0.0048 | 0.9711 | 0.0121 | 0.0167 | 12 |
| aligned_dense_s2 | selection | main | native | both pure | 12664 | 123 | 0.0121 | 0.0048 | 0.9711 | 0.0121 | 0.0167 | 12 |
| aligned_dense_s2 | selection | main | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| aligned_dense_s2 | selection | main | pairs | all | 5298 | 257 | 0.0464 | 0.0263 | 0.9306 | 0.0459 | 0.0235 | 28 |
| aligned_dense_s2 | selection | main | pairs | both pure | 3879 | 49 | 0.0126 | 0.0039 | 0.9730 | 0.0126 | 0.0144 | 7 |
| aligned_dense_s2 | selection | main | pairs | at least one mixed | 1419 | 208 | 0.1296 | 0.0824 | 0.8214 | 0.1279 | 0.0508 | 27 |
| aligned_dense_s2 | selection | main | multi | all | 8468 | 729 | 0.1002 | 0.0640 | 0.8753 | 0.0976 | 0.0271 | 35 |
| aligned_dense_s2 | selection | main | multi | both pure | 4327 | 100 | 0.0286 | 0.0086 | 0.9511 | 0.0286 | 0.0203 | 11 |
| aligned_dense_s2 | selection | main | multi | at least one mixed | 4141 | 629 | 0.1603 | 0.1109 | 0.8119 | 0.1554 | 0.0327 | 35 |
| aligned_dense_s2 | selection | main, equal marker | native | all | 9535 | 109 | 0.0136 | 0.0053 | 0.9694 | 0.0136 | 0.0170 | 12 |
| aligned_dense_s2 | selection | main, equal marker | native | both pure | 9535 | 109 | 0.0136 | 0.0053 | 0.9694 | 0.0136 | 0.0170 | 12 |
| aligned_dense_s2 | selection | main, equal marker | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| aligned_dense_s2 | selection | main, equal marker | pairs | all | 3680 | 222 | 0.0525 | 0.0299 | 0.9195 | 0.0519 | 0.0286 | 27 |
| aligned_dense_s2 | selection | main, equal marker | pairs | both pure | 2642 | 45 | 0.0143 | 0.0044 | 0.9711 | 0.0143 | 0.0146 | 7 |
| aligned_dense_s2 | selection | main, equal marker | pairs | at least one mixed | 1038 | 177 | 0.1450 | 0.0923 | 0.7927 | 0.1430 | 0.0643 | 26 |
| aligned_dense_s2 | selection | main, equal marker | multi | all | 7095 | 702 | 0.1049 | 0.0674 | 0.8703 | 0.1021 | 0.0276 | 35 |
| aligned_dense_s2 | selection | main, equal marker | multi | both pure | 3400 | 87 | 0.0299 | 0.0094 | 0.9509 | 0.0299 | 0.0192 | 11 |
| aligned_dense_s2 | selection | main, equal marker | multi | at least one mixed | 3695 | 615 | 0.1647 | 0.1141 | 0.8058 | 0.1597 | 0.0346 | 35 |
| aligned_dense_s2 | selection | main, changed marker | native | all | 3129 | 14 | 0.0042 | 0.0012 | 0.9786 | 0.0042 | 0.0172 | 7 |
| aligned_dense_s2 | selection | main, changed marker | native | both pure | 3129 | 14 | 0.0042 | 0.0012 | 0.9786 | 0.0042 | 0.0172 | 7 |
| aligned_dense_s2 | selection | main, changed marker | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| aligned_dense_s2 | selection | main, changed marker | pairs | all | 1618 | 35 | 0.0178 | 0.0084 | 0.9654 | 0.0178 | 0.0168 | 14 |
| aligned_dense_s2 | selection | main, changed marker | pairs | both pure | 1237 | 4 | 0.0032 | 0.0014 | 0.9833 | 0.0032 | 0.0135 | 2 |
| aligned_dense_s2 | selection | main, changed marker | pairs | at least one mixed | 381 | 31 | 0.0633 | 0.0307 | 0.9030 | 0.0633 | 0.0338 | 12 |
| aligned_dense_s2 | selection | main, changed marker | multi | all | 1373 | 27 | 0.0174 | 0.0063 | 0.9651 | 0.0174 | 0.0174 | 11 |
| aligned_dense_s2 | selection | main, changed marker | multi | both pure | 927 | 13 | 0.0139 | 0.0022 | 0.9649 | 0.0139 | 0.0212 | 5 |
| aligned_dense_s2 | selection | main, changed marker | multi | at least one mixed | 446 | 14 | 0.0262 | 0.0147 | 0.9668 | 0.0262 | 0.0070 | 8 |
| aligned_dense_s2 | selection | broad | native | all | 16722 | 188 | 0.0129 | 0.0068 | 0.9701 | 0.0129 | 0.0170 | 12 |
| aligned_dense_s2 | selection | broad | native | both pure | 16722 | 188 | 0.0129 | 0.0068 | 0.9701 | 0.0129 | 0.0170 | 12 |
| aligned_dense_s2 | selection | broad | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| aligned_dense_s2 | selection | broad | pairs | all | 6873 | 397 | 0.0525 | 0.0333 | 0.9235 | 0.0517 | 0.0248 | 31 |
| aligned_dense_s2 | selection | broad | pairs | both pure | 4946 | 69 | 0.0125 | 0.0053 | 0.9728 | 0.0125 | 0.0147 | 8 |
| aligned_dense_s2 | selection | broad | pairs | at least one mixed | 1927 | 328 | 0.1466 | 0.1003 | 0.8040 | 0.1437 | 0.0523 | 30 |
| aligned_dense_s2 | selection | broad | multi | all | 11834 | 1205 | 0.1114 | 0.0743 | 0.8642 | 0.1084 | 0.0274 | 35 |
| aligned_dense_s2 | selection | broad | multi | both pure | 5833 | 158 | 0.0311 | 0.0116 | 0.9489 | 0.0311 | 0.0200 | 11 |
| aligned_dense_s2 | selection | broad | multi | at least one mixed | 6001 | 1047 | 0.1743 | 0.1239 | 0.7978 | 0.1689 | 0.0332 | 35 |
| center_only_s0 | selection | main | native | all | 12664 | 115 | 0.0124 | 0.0118 | 0.9731 | 0.0120 | 0.0149 | 10 |
| center_only_s0 | selection | main | native | both pure | 12664 | 115 | 0.0124 | 0.0118 | 0.9731 | 0.0120 | 0.0149 | 10 |
| center_only_s0 | selection | main | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| center_only_s0 | selection | main | pairs | all | 5298 | 476 | 0.0910 | 0.0566 | 0.8775 | 0.0868 | 0.0357 | 32 |
| center_only_s0 | selection | main | pairs | both pure | 3879 | 24 | 0.0067 | 0.0099 | 0.9781 | 0.0067 | 0.0151 | 7 |
| center_only_s0 | selection | main | pairs | at least one mixed | 1419 | 452 | 0.3065 | 0.1746 | 0.6166 | 0.2919 | 0.0915 | 31 |
| center_only_s0 | selection | main | multi | all | 8468 | 1901 | 0.2601 | 0.1441 | 0.7129 | 0.2522 | 0.0350 | 36 |
| center_only_s0 | selection | main | multi | both pure | 4327 | 158 | 0.0456 | 0.0195 | 0.9393 | 0.0456 | 0.0151 | 9 |
| center_only_s0 | selection | main | multi | at least one mixed | 4141 | 1743 | 0.4427 | 0.2502 | 0.5187 | 0.4281 | 0.0531 | 36 |
| center_only_s0 | selection | main, equal marker | native | all | 9535 | 102 | 0.0135 | 0.0122 | 0.9722 | 0.0130 | 0.0147 | 10 |
| center_only_s0 | selection | main, equal marker | native | both pure | 9535 | 102 | 0.0135 | 0.0122 | 0.9722 | 0.0130 | 0.0147 | 10 |
| center_only_s0 | selection | main, equal marker | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| center_only_s0 | selection | main, equal marker | pairs | all | 3680 | 390 | 0.1057 | 0.0648 | 0.8574 | 0.1002 | 0.0423 | 32 |
| center_only_s0 | selection | main, equal marker | pairs | both pure | 2642 | 19 | 0.0075 | 0.0105 | 0.9786 | 0.0075 | 0.0139 | 7 |
| center_only_s0 | selection | main, equal marker | pairs | at least one mixed | 1038 | 371 | 0.3494 | 0.1985 | 0.5559 | 0.3306 | 0.1135 | 31 |
| center_only_s0 | selection | main, equal marker | multi | all | 7095 | 1792 | 0.2739 | 0.1518 | 0.6979 | 0.2657 | 0.0364 | 36 |
| center_only_s0 | selection | main, equal marker | multi | both pure | 3400 | 144 | 0.0498 | 0.0207 | 0.9376 | 0.0498 | 0.0126 | 8 |
| center_only_s0 | selection | main, equal marker | multi | at least one mixed | 3695 | 1648 | 0.4549 | 0.2576 | 0.5034 | 0.4401 | 0.0565 | 36 |
| center_only_s0 | selection | main, changed marker | native | all | 3129 | 13 | 0.0045 | 0.0064 | 0.9815 | 0.0045 | 0.0140 | 5 |
| center_only_s0 | selection | main, changed marker | native | both pure | 3129 | 13 | 0.0045 | 0.0064 | 0.9815 | 0.0045 | 0.0140 | 5 |
| center_only_s0 | selection | main, changed marker | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| center_only_s0 | selection | main, changed marker | pairs | all | 1618 | 86 | 0.0429 | 0.0239 | 0.9322 | 0.0419 | 0.0259 | 19 |
| center_only_s0 | selection | main, changed marker | pairs | both pure | 1237 | 5 | 0.0039 | 0.0070 | 0.9833 | 0.0039 | 0.0129 | 2 |
| center_only_s0 | selection | main, changed marker | pairs | at least one mixed | 381 | 81 | 0.1698 | 0.0796 | 0.7605 | 0.1656 | 0.0738 | 19 |
| center_only_s0 | selection | main, changed marker | multi | all | 1373 | 109 | 0.0674 | 0.0242 | 0.8971 | 0.0651 | 0.0378 | 20 |
| center_only_s0 | selection | main, changed marker | multi | both pure | 927 | 14 | 0.0131 | 0.0079 | 0.9567 | 0.0131 | 0.0302 | 4 |
| center_only_s0 | selection | main, changed marker | multi | at least one mixed | 446 | 95 | 0.1801 | 0.0588 | 0.7710 | 0.1731 | 0.0559 | 20 |
| center_only_s0 | selection | broad | native | all | 16722 | 173 | 0.0139 | 0.0134 | 0.9716 | 0.0135 | 0.0149 | 11 |
| center_only_s0 | selection | broad | native | both pure | 16722 | 173 | 0.0139 | 0.0134 | 0.9716 | 0.0135 | 0.0149 | 11 |
| center_only_s0 | selection | broad | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| center_only_s0 | selection | broad | pairs | all | 6873 | 752 | 0.1036 | 0.0653 | 0.8645 | 0.0986 | 0.0369 | 33 |
| center_only_s0 | selection | broad | pairs | both pure | 4946 | 43 | 0.0086 | 0.0114 | 0.9764 | 0.0086 | 0.0150 | 8 |
| center_only_s0 | selection | broad | pairs | at least one mixed | 1927 | 709 | 0.3387 | 0.1974 | 0.5863 | 0.3219 | 0.0918 | 32 |
| center_only_s0 | selection | broad | multi | all | 11834 | 2928 | 0.2773 | 0.1549 | 0.6972 | 0.2681 | 0.0348 | 36 |
| center_only_s0 | selection | broad | multi | both pure | 5833 | 243 | 0.0503 | 0.0215 | 0.9362 | 0.0503 | 0.0134 | 10 |
| center_only_s0 | selection | broad | multi | at least one mixed | 6001 | 2685 | 0.4579 | 0.2611 | 0.5058 | 0.4414 | 0.0529 | 36 |
| center_only_s1 | selection | main | native | all | 12664 | 709 | 0.0729 | 0.0141 | 0.8562 | 0.0673 | 0.0764 | 22 |
| center_only_s1 | selection | main | native | both pure | 12664 | 709 | 0.0729 | 0.0141 | 0.8562 | 0.0673 | 0.0764 | 22 |
| center_only_s1 | selection | main | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| center_only_s1 | selection | main | pairs | all | 5298 | 667 | 0.1264 | 0.0509 | 0.8039 | 0.1136 | 0.0825 | 35 |
| center_only_s1 | selection | main | pairs | both pure | 3879 | 181 | 0.0476 | 0.0123 | 0.8961 | 0.0448 | 0.0591 | 19 |
| center_only_s1 | selection | main | pairs | at least one mixed | 1419 | 486 | 0.3195 | 0.1475 | 0.5741 | 0.2822 | 0.1438 | 32 |
| center_only_s1 | selection | main | multi | all | 8468 | 1873 | 0.2532 | 0.1267 | 0.7083 | 0.2354 | 0.0563 | 36 |
| center_only_s1 | selection | main | multi | both pure | 4327 | 152 | 0.0348 | 0.0185 | 0.9343 | 0.0335 | 0.0321 | 18 |
| center_only_s1 | selection | main | multi | at least one mixed | 4141 | 1721 | 0.4399 | 0.2188 | 0.5160 | 0.4080 | 0.0760 | 36 |
| center_only_s1 | selection | main, equal marker | native | all | 9535 | 601 | 0.0762 | 0.0152 | 0.8493 | 0.0701 | 0.0807 | 21 |
| center_only_s1 | selection | main, equal marker | native | both pure | 9535 | 601 | 0.0762 | 0.0152 | 0.8493 | 0.0701 | 0.0807 | 21 |
| center_only_s1 | selection | main, equal marker | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| center_only_s1 | selection | main, equal marker | pairs | all | 3680 | 535 | 0.1410 | 0.0591 | 0.7805 | 0.1254 | 0.0941 | 35 |
| center_only_s1 | selection | main, equal marker | pairs | both pure | 2642 | 135 | 0.0507 | 0.0140 | 0.8914 | 0.0478 | 0.0608 | 18 |
| center_only_s1 | selection | main, equal marker | pairs | at least one mixed | 1038 | 400 | 0.3603 | 0.1695 | 0.5122 | 0.3139 | 0.1739 | 32 |
| center_only_s1 | selection | main, equal marker | multi | all | 7095 | 1757 | 0.2670 | 0.1338 | 0.6938 | 0.2476 | 0.0586 | 36 |
| center_only_s1 | selection | main, equal marker | multi | both pure | 3400 | 121 | 0.0368 | 0.0200 | 0.9323 | 0.0350 | 0.0327 | 18 |
| center_only_s1 | selection | main, equal marker | multi | at least one mixed | 3695 | 1636 | 0.4534 | 0.2257 | 0.5014 | 0.4199 | 0.0787 | 36 |
| center_only_s1 | selection | main, changed marker | native | all | 3129 | 108 | 0.0356 | 0.0053 | 0.9157 | 0.0356 | 0.0487 | 20 |
| center_only_s1 | selection | main, changed marker | native | both pure | 3129 | 108 | 0.0356 | 0.0053 | 0.9157 | 0.0356 | 0.0487 | 20 |
| center_only_s1 | selection | main, changed marker | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| center_only_s1 | selection | main, changed marker | pairs | all | 1618 | 132 | 0.0711 | 0.0175 | 0.8633 | 0.0676 | 0.0691 | 24 |
| center_only_s1 | selection | main, changed marker | pairs | both pure | 1237 | 46 | 0.0344 | 0.0059 | 0.9122 | 0.0325 | 0.0553 | 14 |
| center_only_s1 | selection | main, changed marker | pairs | at least one mixed | 381 | 86 | 0.1825 | 0.0547 | 0.7036 | 0.1741 | 0.1224 | 20 |
| center_only_s1 | selection | main, changed marker | multi | all | 1373 | 116 | 0.0709 | 0.0184 | 0.8674 | 0.0663 | 0.0663 | 23 |
| center_only_s1 | selection | main, changed marker | multi | both pure | 927 | 31 | 0.0294 | 0.0053 | 0.9191 | 0.0286 | 0.0523 | 7 |
| center_only_s1 | selection | main, changed marker | multi | at least one mixed | 446 | 85 | 0.1608 | 0.0455 | 0.7622 | 0.1486 | 0.0892 | 22 |
| center_only_s1 | selection | broad | native | all | 16722 | 953 | 0.0742 | 0.0148 | 0.8526 | 0.0676 | 0.0798 | 22 |
| center_only_s1 | selection | broad | native | both pure | 16722 | 953 | 0.0742 | 0.0148 | 0.8526 | 0.0676 | 0.0798 | 22 |
| center_only_s1 | selection | broad | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| center_only_s1 | selection | broad | pairs | all | 6873 | 974 | 0.1371 | 0.0577 | 0.7978 | 0.1248 | 0.0774 | 35 |
| center_only_s1 | selection | broad | pairs | both pure | 4946 | 245 | 0.0494 | 0.0134 | 0.8981 | 0.0470 | 0.0549 | 19 |
| center_only_s1 | selection | broad | pairs | at least one mixed | 1927 | 729 | 0.3431 | 0.1646 | 0.5601 | 0.3072 | 0.1327 | 33 |
| center_only_s1 | selection | broad | multi | all | 11834 | 2838 | 0.2666 | 0.1343 | 0.6982 | 0.2470 | 0.0548 | 36 |
| center_only_s1 | selection | broad | multi | both pure | 5833 | 205 | 0.0339 | 0.0192 | 0.9380 | 0.0331 | 0.0289 | 19 |
| center_only_s1 | selection | broad | multi | at least one mixed | 6001 | 2633 | 0.4523 | 0.2260 | 0.5073 | 0.4178 | 0.0749 | 36 |
| center_only_s2 | selection | main | native | all | 12664 | 131 | 0.0130 | 0.0093 | 0.9764 | 0.0130 | 0.0106 | 9 |
| center_only_s2 | selection | main | native | both pure | 12664 | 131 | 0.0130 | 0.0093 | 0.9764 | 0.0130 | 0.0106 | 9 |
| center_only_s2 | selection | main | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| center_only_s2 | selection | main | pairs | all | 5298 | 488 | 0.0931 | 0.0578 | 0.8851 | 0.0898 | 0.0252 | 33 |
| center_only_s2 | selection | main | pairs | both pure | 3879 | 43 | 0.0116 | 0.0069 | 0.9818 | 0.0116 | 0.0067 | 5 |
| center_only_s2 | selection | main | pairs | at least one mixed | 1419 | 445 | 0.2962 | 0.1855 | 0.6381 | 0.2847 | 0.0772 | 32 |
| center_only_s2 | selection | main | multi | all | 8468 | 1900 | 0.2611 | 0.1520 | 0.7002 | 0.2504 | 0.0493 | 36 |
| center_only_s2 | selection | main | multi | both pure | 4327 | 203 | 0.0556 | 0.0183 | 0.9079 | 0.0556 | 0.0365 | 8 |
| center_only_s2 | selection | main | multi | at least one mixed | 4141 | 1697 | 0.4346 | 0.2657 | 0.5244 | 0.4148 | 0.0608 | 36 |
| center_only_s2 | selection | main, equal marker | native | all | 9535 | 115 | 0.0141 | 0.0101 | 0.9759 | 0.0141 | 0.0101 | 9 |
| center_only_s2 | selection | main, equal marker | native | both pure | 9535 | 115 | 0.0141 | 0.0101 | 0.9759 | 0.0141 | 0.0101 | 9 |
| center_only_s2 | selection | main, equal marker | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| center_only_s2 | selection | main, equal marker | pairs | all | 3680 | 398 | 0.1096 | 0.0676 | 0.8616 | 0.1057 | 0.0327 | 33 |
| center_only_s2 | selection | main, equal marker | pairs | both pure | 2642 | 34 | 0.0140 | 0.0080 | 0.9802 | 0.0140 | 0.0058 | 5 |
| center_only_s2 | selection | main, equal marker | pairs | at least one mixed | 1038 | 364 | 0.3427 | 0.2139 | 0.5696 | 0.3295 | 0.1009 | 32 |
| center_only_s2 | selection | main, equal marker | multi | all | 7095 | 1795 | 0.2751 | 0.1606 | 0.6850 | 0.2641 | 0.0509 | 36 |
| center_only_s2 | selection | main, equal marker | multi | both pure | 3400 | 182 | 0.0603 | 0.0201 | 0.9047 | 0.0603 | 0.0350 | 8 |
| center_only_s2 | selection | main, equal marker | multi | at least one mixed | 3695 | 1613 | 0.4478 | 0.2741 | 0.5081 | 0.4278 | 0.0641 | 36 |
| center_only_s2 | selection | main, changed marker | native | all | 3129 | 16 | 0.0042 | 0.0028 | 0.9836 | 0.0042 | 0.0122 | 4 |
| center_only_s2 | selection | main, changed marker | native | both pure | 3129 | 16 | 0.0042 | 0.0028 | 0.9836 | 0.0042 | 0.0122 | 4 |
| center_only_s2 | selection | main, changed marker | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| center_only_s2 | selection | main, changed marker | pairs | all | 1618 | 90 | 0.0445 | 0.0220 | 0.9428 | 0.0445 | 0.0127 | 22 |
| center_only_s2 | selection | main, changed marker | pairs | both pure | 1237 | 9 | 0.0058 | 0.0029 | 0.9884 | 0.0058 | 0.0058 | 3 |
| center_only_s2 | selection | main, changed marker | pairs | at least one mixed | 381 | 81 | 0.1656 | 0.0829 | 0.7922 | 0.1656 | 0.0422 | 20 |
| center_only_s2 | selection | main, changed marker | multi | all | 1373 | 105 | 0.0651 | 0.0245 | 0.8942 | 0.0610 | 0.0448 | 19 |
| center_only_s2 | selection | main, changed marker | multi | both pure | 927 | 21 | 0.0212 | 0.0042 | 0.9371 | 0.0212 | 0.0417 | 4 |
| center_only_s2 | selection | main, changed marker | multi | at least one mixed | 446 | 84 | 0.1503 | 0.0659 | 0.8094 | 0.1381 | 0.0524 | 19 |
| center_only_s2 | selection | broad | native | all | 16722 | 202 | 0.0147 | 0.0106 | 0.9753 | 0.0147 | 0.0100 | 9 |
| center_only_s2 | selection | broad | native | both pure | 16722 | 202 | 0.0147 | 0.0106 | 0.9753 | 0.0147 | 0.0100 | 9 |
| center_only_s2 | selection | broad | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| center_only_s2 | selection | broad | pairs | all | 6873 | 758 | 0.1042 | 0.0664 | 0.8733 | 0.1005 | 0.0262 | 34 |
| center_only_s2 | selection | broad | pairs | both pure | 4946 | 67 | 0.0135 | 0.0080 | 0.9797 | 0.0135 | 0.0067 | 7 |
| center_only_s2 | selection | broad | pairs | at least one mixed | 1927 | 691 | 0.3232 | 0.2080 | 0.6130 | 0.3106 | 0.0765 | 33 |
| center_only_s2 | selection | broad | multi | all | 11834 | 2902 | 0.2770 | 0.1619 | 0.6866 | 0.2651 | 0.0483 | 36 |
| center_only_s2 | selection | broad | multi | both pure | 5833 | 300 | 0.0602 | 0.0203 | 0.9055 | 0.0602 | 0.0343 | 10 |
| center_only_s2 | selection | broad | multi | at least one mixed | 6001 | 2602 | 0.4478 | 0.2746 | 0.5137 | 0.4265 | 0.0598 | 36 |
| shuffled_dense_s0 | selection | main | native | all | 12664 | 107 | 0.0123 | 0.0058 | 0.9765 | 0.0123 | 0.0113 | 9 |
| shuffled_dense_s0 | selection | main | native | both pure | 12664 | 107 | 0.0123 | 0.0058 | 0.9765 | 0.0123 | 0.0113 | 9 |
| shuffled_dense_s0 | selection | main | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| shuffled_dense_s0 | selection | main | pairs | all | 5298 | 508 | 0.0914 | 0.0528 | 0.8865 | 0.0880 | 0.0255 | 32 |
| shuffled_dense_s0 | selection | main | pairs | both pure | 3879 | 31 | 0.0090 | 0.0045 | 0.9826 | 0.0090 | 0.0084 | 4 |
| shuffled_dense_s0 | selection | main | pairs | at least one mixed | 1419 | 477 | 0.2981 | 0.1752 | 0.6401 | 0.2861 | 0.0739 | 31 |
| shuffled_dense_s0 | selection | main | multi | all | 8468 | 1799 | 0.2474 | 0.1551 | 0.7277 | 0.2405 | 0.0318 | 36 |
| shuffled_dense_s0 | selection | main | multi | both pure | 4327 | 147 | 0.0397 | 0.0160 | 0.9399 | 0.0397 | 0.0204 | 6 |
| shuffled_dense_s0 | selection | main | multi | at least one mixed | 4141 | 1652 | 0.4233 | 0.2735 | 0.5469 | 0.4104 | 0.0427 | 36 |
| shuffled_dense_s0 | selection | main, equal marker | native | all | 9535 | 99 | 0.0136 | 0.0063 | 0.9754 | 0.0136 | 0.0109 | 9 |
| shuffled_dense_s0 | selection | main, equal marker | native | both pure | 9535 | 99 | 0.0136 | 0.0063 | 0.9754 | 0.0136 | 0.0109 | 9 |
| shuffled_dense_s0 | selection | main, equal marker | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| shuffled_dense_s0 | selection | main, equal marker | pairs | all | 3680 | 416 | 0.1081 | 0.0616 | 0.8629 | 0.1041 | 0.0330 | 32 |
| shuffled_dense_s0 | selection | main, equal marker | pairs | both pure | 2642 | 26 | 0.0109 | 0.0052 | 0.9816 | 0.0109 | 0.0075 | 4 |
| shuffled_dense_s0 | selection | main, equal marker | pairs | at least one mixed | 1038 | 390 | 0.3464 | 0.2007 | 0.5695 | 0.3327 | 0.0978 | 31 |
| shuffled_dense_s0 | selection | main, equal marker | multi | all | 7095 | 1710 | 0.2617 | 0.1639 | 0.7122 | 0.2545 | 0.0333 | 36 |
| shuffled_dense_s0 | selection | main, equal marker | multi | both pure | 3400 | 134 | 0.0442 | 0.0177 | 0.9380 | 0.0442 | 0.0178 | 6 |
| shuffled_dense_s0 | selection | main, equal marker | multi | at least one mixed | 3695 | 1576 | 0.4370 | 0.2820 | 0.5297 | 0.4238 | 0.0465 | 36 |
| shuffled_dense_s0 | selection | main, changed marker | native | all | 3129 | 8 | 0.0021 | 0.0016 | 0.9852 | 0.0021 | 0.0127 | 3 |
| shuffled_dense_s0 | selection | main, changed marker | native | both pure | 3129 | 8 | 0.0021 | 0.0016 | 0.9852 | 0.0021 | 0.0127 | 3 |
| shuffled_dense_s0 | selection | main, changed marker | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| shuffled_dense_s0 | selection | main, changed marker | pairs | all | 1618 | 92 | 0.0452 | 0.0181 | 0.9416 | 0.0452 | 0.0132 | 21 |
| shuffled_dense_s0 | selection | main, changed marker | pairs | both pure | 1237 | 5 | 0.0039 | 0.0018 | 0.9878 | 0.0039 | 0.0084 | 1 |
| shuffled_dense_s0 | selection | main, changed marker | pairs | at least one mixed | 381 | 87 | 0.1751 | 0.0707 | 0.7890 | 0.1751 | 0.0359 | 21 |
| shuffled_dense_s0 | selection | main, changed marker | multi | all | 1373 | 89 | 0.0517 | 0.0202 | 0.9180 | 0.0506 | 0.0314 | 20 |
| shuffled_dense_s0 | selection | main, changed marker | multi | both pure | 927 | 13 | 0.0106 | 0.0036 | 0.9551 | 0.0106 | 0.0343 | 3 |
| shuffled_dense_s0 | selection | main, changed marker | multi | at least one mixed | 446 | 76 | 0.1329 | 0.0544 | 0.8427 | 0.1294 | 0.0280 | 20 |
| shuffled_dense_s0 | selection | broad | native | all | 16722 | 172 | 0.0140 | 0.0066 | 0.9747 | 0.0140 | 0.0113 | 9 |
| shuffled_dense_s0 | selection | broad | native | both pure | 16722 | 172 | 0.0140 | 0.0066 | 0.9747 | 0.0140 | 0.0113 | 9 |
| shuffled_dense_s0 | selection | broad | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| shuffled_dense_s0 | selection | broad | pairs | all | 6873 | 775 | 0.1013 | 0.0611 | 0.8759 | 0.0974 | 0.0267 | 34 |
| shuffled_dense_s0 | selection | broad | pairs | both pure | 4946 | 48 | 0.0092 | 0.0051 | 0.9821 | 0.0092 | 0.0087 | 4 |
| shuffled_dense_s0 | selection | broad | pairs | at least one mixed | 1927 | 727 | 0.3243 | 0.1977 | 0.6151 | 0.3111 | 0.0738 | 33 |
| shuffled_dense_s0 | selection | broad | multi | all | 11834 | 2778 | 0.2641 | 0.1667 | 0.7136 | 0.2564 | 0.0300 | 36 |
| shuffled_dense_s0 | selection | broad | multi | both pure | 5833 | 229 | 0.0451 | 0.0179 | 0.9378 | 0.0451 | 0.0172 | 7 |
| shuffled_dense_s0 | selection | broad | multi | at least one mixed | 6001 | 2549 | 0.4373 | 0.2852 | 0.5352 | 0.4235 | 0.0413 | 36 |
| shuffled_dense_s1 | selection | main | native | all | 12664 | 106 | 0.0123 | 0.0061 | 0.9735 | 0.0123 | 0.0142 | 10 |
| shuffled_dense_s1 | selection | main | native | both pure | 12664 | 106 | 0.0123 | 0.0061 | 0.9735 | 0.0123 | 0.0142 | 10 |
| shuffled_dense_s1 | selection | main | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| shuffled_dense_s1 | selection | main | pairs | all | 5298 | 497 | 0.0914 | 0.0546 | 0.8838 | 0.0881 | 0.0281 | 32 |
| shuffled_dense_s1 | selection | main | pairs | both pure | 3879 | 40 | 0.0115 | 0.0051 | 0.9758 | 0.0115 | 0.0127 | 5 |
| shuffled_dense_s1 | selection | main | pairs | at least one mixed | 1419 | 457 | 0.2910 | 0.1798 | 0.6494 | 0.2794 | 0.0712 | 31 |
| shuffled_dense_s1 | selection | main | multi | all | 8468 | 1765 | 0.2421 | 0.1559 | 0.7433 | 0.2345 | 0.0222 | 36 |
| shuffled_dense_s1 | selection | main | multi | both pure | 4327 | 106 | 0.0293 | 0.0134 | 0.9617 | 0.0293 | 0.0090 | 9 |
| shuffled_dense_s1 | selection | main | multi | at least one mixed | 4141 | 1659 | 0.4228 | 0.2773 | 0.5577 | 0.4087 | 0.0337 | 36 |
| shuffled_dense_s1 | selection | main, equal marker | native | all | 9535 | 96 | 0.0136 | 0.0067 | 0.9721 | 0.0136 | 0.0143 | 10 |
| shuffled_dense_s1 | selection | main, equal marker | native | both pure | 9535 | 96 | 0.0136 | 0.0067 | 0.9721 | 0.0136 | 0.0143 | 10 |
| shuffled_dense_s1 | selection | main, equal marker | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| shuffled_dense_s1 | selection | main, equal marker | pairs | all | 3680 | 409 | 0.1069 | 0.0639 | 0.8616 | 0.1030 | 0.0354 | 32 |
| shuffled_dense_s1 | selection | main, equal marker | pairs | both pure | 2642 | 32 | 0.0137 | 0.0060 | 0.9750 | 0.0137 | 0.0112 | 5 |
| shuffled_dense_s1 | selection | main, equal marker | pairs | at least one mixed | 1038 | 377 | 0.3347 | 0.2066 | 0.5823 | 0.3215 | 0.0963 | 31 |
| shuffled_dense_s1 | selection | main, equal marker | multi | all | 7095 | 1676 | 0.2559 | 0.1647 | 0.7285 | 0.2479 | 0.0236 | 36 |
| shuffled_dense_s1 | selection | main, equal marker | multi | both pure | 3400 | 97 | 0.0321 | 0.0147 | 0.9612 | 0.0321 | 0.0067 | 9 |
| shuffled_dense_s1 | selection | main, equal marker | multi | at least one mixed | 3695 | 1579 | 0.4364 | 0.2860 | 0.5402 | 0.4220 | 0.0377 | 36 |
| shuffled_dense_s1 | selection | main, changed marker | native | all | 3129 | 10 | 0.0026 | 0.0016 | 0.9828 | 0.0026 | 0.0146 | 5 |
| shuffled_dense_s1 | selection | main, changed marker | native | both pure | 3129 | 10 | 0.0026 | 0.0016 | 0.9828 | 0.0026 | 0.0146 | 5 |
| shuffled_dense_s1 | selection | main, changed marker | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| shuffled_dense_s1 | selection | main, changed marker | pairs | all | 1618 | 88 | 0.0432 | 0.0178 | 0.9416 | 0.0432 | 0.0152 | 21 |
| shuffled_dense_s1 | selection | main, changed marker | pairs | both pure | 1237 | 8 | 0.0058 | 0.0017 | 0.9807 | 0.0058 | 0.0135 | 2 |
| shuffled_dense_s1 | selection | main, changed marker | pairs | at least one mixed | 381 | 80 | 0.1603 | 0.0695 | 0.8122 | 0.1603 | 0.0274 | 20 |
| shuffled_dense_s1 | selection | main, changed marker | multi | all | 1373 | 89 | 0.0523 | 0.0198 | 0.9291 | 0.0512 | 0.0198 | 20 |
| shuffled_dense_s1 | selection | main, changed marker | multi | both pure | 927 | 9 | 0.0082 | 0.0030 | 0.9706 | 0.0082 | 0.0212 | 2 |
| shuffled_dense_s1 | selection | main, changed marker | multi | at least one mixed | 446 | 80 | 0.1399 | 0.0542 | 0.8497 | 0.1364 | 0.0140 | 20 |
| shuffled_dense_s1 | selection | broad | native | all | 16722 | 170 | 0.0136 | 0.0068 | 0.9724 | 0.0136 | 0.0140 | 10 |
| shuffled_dense_s1 | selection | broad | native | both pure | 16722 | 170 | 0.0136 | 0.0068 | 0.9724 | 0.0136 | 0.0140 | 10 |
| shuffled_dense_s1 | selection | broad | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| shuffled_dense_s1 | selection | broad | pairs | all | 6873 | 765 | 0.1022 | 0.0631 | 0.8727 | 0.0984 | 0.0288 | 34 |
| shuffled_dense_s1 | selection | broad | pairs | both pure | 4946 | 59 | 0.0121 | 0.0058 | 0.9751 | 0.0121 | 0.0128 | 5 |
| shuffled_dense_s1 | selection | broad | pairs | at least one mixed | 1927 | 706 | 0.3192 | 0.2022 | 0.6230 | 0.3064 | 0.0706 | 33 |
| shuffled_dense_s1 | selection | broad | multi | all | 11834 | 2740 | 0.2593 | 0.1675 | 0.7284 | 0.2511 | 0.0205 | 36 |
| shuffled_dense_s1 | selection | broad | multi | both pure | 5833 | 173 | 0.0319 | 0.0148 | 0.9608 | 0.0319 | 0.0073 | 11 |
| shuffled_dense_s1 | selection | broad | multi | at least one mixed | 6001 | 2567 | 0.4396 | 0.2891 | 0.5437 | 0.4250 | 0.0314 | 36 |
| shuffled_dense_s2 | selection | main | native | all | 12664 | 118 | 0.0136 | 0.0064 | 0.9726 | 0.0136 | 0.0138 | 10 |
| shuffled_dense_s2 | selection | main | native | both pure | 12664 | 118 | 0.0136 | 0.0064 | 0.9726 | 0.0136 | 0.0138 | 10 |
| shuffled_dense_s2 | selection | main | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| shuffled_dense_s2 | selection | main | pairs | all | 5298 | 518 | 0.0932 | 0.0540 | 0.8823 | 0.0897 | 0.0279 | 32 |
| shuffled_dense_s2 | selection | main | pairs | both pure | 3879 | 39 | 0.0110 | 0.0051 | 0.9776 | 0.0110 | 0.0114 | 5 |
| shuffled_dense_s2 | selection | main | pairs | at least one mixed | 1419 | 479 | 0.2985 | 0.1772 | 0.6392 | 0.2865 | 0.0743 | 31 |
| shuffled_dense_s2 | selection | main | multi | all | 8468 | 1833 | 0.2503 | 0.1558 | 0.7321 | 0.2433 | 0.0246 | 36 |
| shuffled_dense_s2 | selection | main | multi | both pure | 4327 | 157 | 0.0430 | 0.0151 | 0.9463 | 0.0430 | 0.0107 | 10 |
| shuffled_dense_s2 | selection | main | multi | at least one mixed | 4141 | 1676 | 0.4265 | 0.2758 | 0.5491 | 0.4136 | 0.0373 | 36 |
| shuffled_dense_s2 | selection | main, equal marker | native | all | 9535 | 108 | 0.0150 | 0.0070 | 0.9712 | 0.0150 | 0.0138 | 10 |
| shuffled_dense_s2 | selection | main, equal marker | native | both pure | 9535 | 108 | 0.0150 | 0.0070 | 0.9712 | 0.0150 | 0.0138 | 10 |
| shuffled_dense_s2 | selection | main, equal marker | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| shuffled_dense_s2 | selection | main, equal marker | pairs | all | 3680 | 430 | 0.1101 | 0.0630 | 0.8589 | 0.1061 | 0.0349 | 32 |
| shuffled_dense_s2 | selection | main, equal marker | pairs | both pure | 2642 | 35 | 0.0131 | 0.0059 | 0.9762 | 0.0131 | 0.0106 | 5 |
| shuffled_dense_s2 | selection | main, equal marker | pairs | at least one mixed | 1038 | 395 | 0.3474 | 0.2035 | 0.5700 | 0.3337 | 0.0963 | 31 |
| shuffled_dense_s2 | selection | main, equal marker | multi | all | 7095 | 1734 | 0.2642 | 0.1646 | 0.7166 | 0.2569 | 0.0265 | 36 |
| shuffled_dense_s2 | selection | main, equal marker | multi | both pure | 3400 | 142 | 0.0471 | 0.0166 | 0.9448 | 0.0471 | 0.0081 | 10 |
| shuffled_dense_s2 | selection | main, equal marker | multi | at least one mixed | 3695 | 1592 | 0.4396 | 0.2844 | 0.5316 | 0.4264 | 0.0420 | 36 |
| shuffled_dense_s2 | selection | main, changed marker | native | all | 3129 | 10 | 0.0029 | 0.0021 | 0.9815 | 0.0029 | 0.0156 | 3 |
| shuffled_dense_s2 | selection | main, changed marker | native | both pure | 3129 | 10 | 0.0029 | 0.0021 | 0.9815 | 0.0029 | 0.0156 | 3 |
| shuffled_dense_s2 | selection | main, changed marker | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| shuffled_dense_s2 | selection | main, changed marker | pairs | all | 1618 | 88 | 0.0422 | 0.0176 | 0.9416 | 0.0422 | 0.0163 | 20 |
| shuffled_dense_s2 | selection | main, changed marker | pairs | both pure | 1237 | 4 | 0.0026 | 0.0019 | 0.9871 | 0.0026 | 0.0103 | 1 |
| shuffled_dense_s2 | selection | main, changed marker | pairs | at least one mixed | 381 | 84 | 0.1667 | 0.0679 | 0.7911 | 0.1667 | 0.0422 | 20 |
| shuffled_dense_s2 | selection | main, changed marker | multi | all | 1373 | 99 | 0.0587 | 0.0212 | 0.9233 | 0.0576 | 0.0192 | 20 |
| shuffled_dense_s2 | selection | main, changed marker | multi | both pure | 927 | 15 | 0.0139 | 0.0037 | 0.9608 | 0.0139 | 0.0253 | 4 |
| shuffled_dense_s2 | selection | main, changed marker | multi | at least one mixed | 446 | 84 | 0.1503 | 0.0572 | 0.8462 | 0.1469 | 0.0070 | 20 |
| shuffled_dense_s2 | selection | broad | native | all | 16722 | 193 | 0.0153 | 0.0071 | 0.9711 | 0.0153 | 0.0136 | 10 |
| shuffled_dense_s2 | selection | broad | native | both pure | 16722 | 193 | 0.0153 | 0.0071 | 0.9711 | 0.0153 | 0.0136 | 10 |
| shuffled_dense_s2 | selection | broad | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| shuffled_dense_s2 | selection | broad | pairs | all | 6873 | 795 | 0.1035 | 0.0623 | 0.8716 | 0.0996 | 0.0288 | 33 |
| shuffled_dense_s2 | selection | broad | pairs | both pure | 4946 | 62 | 0.0121 | 0.0058 | 0.9764 | 0.0121 | 0.0115 | 5 |
| shuffled_dense_s2 | selection | broad | pairs | at least one mixed | 1927 | 733 | 0.3240 | 0.1997 | 0.6155 | 0.3109 | 0.0736 | 33 |
| shuffled_dense_s2 | selection | broad | multi | all | 11834 | 2807 | 0.2655 | 0.1667 | 0.7185 | 0.2578 | 0.0237 | 36 |
| shuffled_dense_s2 | selection | broad | multi | both pure | 5833 | 238 | 0.0469 | 0.0166 | 0.9439 | 0.0469 | 0.0093 | 10 |
| shuffled_dense_s2 | selection | broad | multi | at least one mixed | 6001 | 2569 | 0.4389 | 0.2862 | 0.5387 | 0.4251 | 0.0362 | 36 |
| aligned_dense_s0 | calibration | main | native | all | 10867 | 66 | 0.0070 | 0.0034 | 0.9792 | 0.0070 | 0.0138 | 9 |
| aligned_dense_s0 | calibration | main | native | both pure | 10867 | 66 | 0.0070 | 0.0034 | 0.9792 | 0.0070 | 0.0138 | 9 |
| aligned_dense_s0 | calibration | main | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| aligned_dense_s0 | calibration | main | pairs | all | 4492 | 174 | 0.0468 | 0.0295 | 0.9410 | 0.0463 | 0.0127 | 29 |
| aligned_dense_s0 | calibration | main | pairs | both pure | 3439 | 21 | 0.0052 | 0.0016 | 0.9879 | 0.0052 | 0.0069 | 4 |
| aligned_dense_s0 | calibration | main | pairs | at least one mixed | 1053 | 153 | 0.1541 | 0.1020 | 0.8211 | 0.1525 | 0.0264 | 28 |
| aligned_dense_s0 | calibration | main | multi | all | 8496 | 717 | 0.0992 | 0.0727 | 0.8870 | 0.0984 | 0.0146 | 33 |
| aligned_dense_s0 | calibration | main | multi | both pure | 4288 | 35 | 0.0093 | 0.0053 | 0.9820 | 0.0093 | 0.0086 | 5 |
| aligned_dense_s0 | calibration | main | multi | at least one mixed | 4208 | 682 | 0.1688 | 0.1250 | 0.8138 | 0.1673 | 0.0189 | 33 |
| aligned_dense_s0 | calibration | main, equal marker | native | all | 8168 | 58 | 0.0077 | 0.0037 | 0.9786 | 0.0077 | 0.0136 | 8 |
| aligned_dense_s0 | calibration | main, equal marker | native | both pure | 8168 | 58 | 0.0077 | 0.0037 | 0.9786 | 0.0077 | 0.0136 | 8 |
| aligned_dense_s0 | calibration | main, equal marker | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| aligned_dense_s0 | calibration | main, equal marker | pairs | all | 3155 | 158 | 0.0541 | 0.0342 | 0.9323 | 0.0536 | 0.0141 | 29 |
| aligned_dense_s0 | calibration | main, equal marker | pairs | both pure | 2346 | 13 | 0.0063 | 0.0019 | 0.9870 | 0.0063 | 0.0068 | 4 |
| aligned_dense_s0 | calibration | main, equal marker | pairs | at least one mixed | 809 | 145 | 0.1712 | 0.1138 | 0.7989 | 0.1694 | 0.0317 | 28 |
| aligned_dense_s0 | calibration | main, equal marker | multi | all | 7126 | 686 | 0.1041 | 0.0765 | 0.8814 | 0.1033 | 0.0152 | 33 |
| aligned_dense_s0 | calibration | main, equal marker | multi | both pure | 3324 | 32 | 0.0108 | 0.0059 | 0.9806 | 0.0108 | 0.0086 | 5 |
| aligned_dense_s0 | calibration | main, equal marker | multi | at least one mixed | 3802 | 654 | 0.1720 | 0.1279 | 0.8094 | 0.1707 | 0.0199 | 33 |
| aligned_dense_s0 | calibration | main, changed marker | native | all | 2699 | 8 | 0.0024 | 0.0007 | 0.9821 | 0.0024 | 0.0156 | 4 |
| aligned_dense_s0 | calibration | main, changed marker | native | both pure | 2699 | 8 | 0.0024 | 0.0007 | 0.9821 | 0.0024 | 0.0156 | 4 |
| aligned_dense_s0 | calibration | main, changed marker | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| aligned_dense_s0 | calibration | main, changed marker | pairs | all | 1337 | 16 | 0.0098 | 0.0039 | 0.9780 | 0.0098 | 0.0122 | 8 |
| aligned_dense_s0 | calibration | main, changed marker | pairs | both pure | 1093 | 8 | 0.0065 | 0.0010 | 0.9840 | 0.0065 | 0.0094 | 3 |
| aligned_dense_s0 | calibration | main, changed marker | pairs | at least one mixed | 244 | 8 | 0.0219 | 0.0155 | 0.9563 | 0.0219 | 0.0219 | 5 |
| aligned_dense_s0 | calibration | main, changed marker | multi | all | 1370 | 31 | 0.0181 | 0.0092 | 0.9737 | 0.0170 | 0.0094 | 13 |
| aligned_dense_s0 | calibration | main, changed marker | multi | both pure | 964 | 3 | 0.0024 | 0.0009 | 0.9928 | 0.0024 | 0.0048 | 2 |
| aligned_dense_s0 | calibration | main, changed marker | multi | at least one mixed | 406 | 28 | 0.0541 | 0.0282 | 0.9305 | 0.0502 | 0.0193 | 12 |
| aligned_dense_s0 | calibration | broad | native | all | 14358 | 101 | 0.0078 | 0.0050 | 0.9789 | 0.0078 | 0.0133 | 9 |
| aligned_dense_s0 | calibration | broad | native | both pure | 14358 | 101 | 0.0078 | 0.0050 | 0.9789 | 0.0078 | 0.0133 | 9 |
| aligned_dense_s0 | calibration | broad | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| aligned_dense_s0 | calibration | broad | pairs | all | 5820 | 286 | 0.0545 | 0.0368 | 0.9331 | 0.0537 | 0.0131 | 30 |
| aligned_dense_s0 | calibration | broad | pairs | both pure | 4356 | 25 | 0.0051 | 0.0025 | 0.9887 | 0.0051 | 0.0062 | 4 |
| aligned_dense_s0 | calibration | broad | pairs | at least one mixed | 1464 | 261 | 0.1705 | 0.1188 | 0.8026 | 0.1680 | 0.0294 | 29 |
| aligned_dense_s0 | calibration | broad | multi | all | 11872 | 1201 | 0.1141 | 0.0846 | 0.8729 | 0.1130 | 0.0141 | 35 |
| aligned_dense_s0 | calibration | broad | multi | both pure | 5714 | 54 | 0.0110 | 0.0073 | 0.9799 | 0.0110 | 0.0091 | 6 |
| aligned_dense_s0 | calibration | broad | multi | at least one mixed | 6158 | 1147 | 0.1873 | 0.1396 | 0.7971 | 0.1855 | 0.0174 | 34 |
| aligned_dense_s1 | calibration | main | native | all | 10867 | 78 | 0.0094 | 0.0034 | 0.9742 | 0.0094 | 0.0164 | 12 |
| aligned_dense_s1 | calibration | main | native | both pure | 10867 | 78 | 0.0094 | 0.0034 | 0.9742 | 0.0094 | 0.0164 | 12 |
| aligned_dense_s1 | calibration | main | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| aligned_dense_s1 | calibration | main | pairs | all | 4492 | 181 | 0.0484 | 0.0318 | 0.9357 | 0.0478 | 0.0165 | 29 |
| aligned_dense_s1 | calibration | main | pairs | both pure | 3439 | 23 | 0.0056 | 0.0021 | 0.9827 | 0.0056 | 0.0117 | 5 |
| aligned_dense_s1 | calibration | main | pairs | at least one mixed | 1053 | 158 | 0.1590 | 0.1090 | 0.8163 | 0.1568 | 0.0269 | 28 |
| aligned_dense_s1 | calibration | main | multi | all | 8496 | 752 | 0.1058 | 0.0749 | 0.8807 | 0.1045 | 0.0148 | 33 |
| aligned_dense_s1 | calibration | main | multi | both pure | 4288 | 40 | 0.0111 | 0.0058 | 0.9798 | 0.0111 | 0.0091 | 5 |
| aligned_dense_s1 | calibration | main | multi | at least one mixed | 4208 | 712 | 0.1791 | 0.1285 | 0.8045 | 0.1767 | 0.0188 | 33 |
| aligned_dense_s1 | calibration | main, equal marker | native | all | 8168 | 69 | 0.0101 | 0.0037 | 0.9733 | 0.0101 | 0.0166 | 11 |
| aligned_dense_s1 | calibration | main, equal marker | native | both pure | 8168 | 69 | 0.0101 | 0.0037 | 0.9733 | 0.0101 | 0.0166 | 11 |
| aligned_dense_s1 | calibration | main, equal marker | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| aligned_dense_s1 | calibration | main, equal marker | pairs | all | 3155 | 164 | 0.0554 | 0.0368 | 0.9270 | 0.0547 | 0.0183 | 29 |
| aligned_dense_s1 | calibration | main, equal marker | pairs | both pure | 2346 | 14 | 0.0061 | 0.0025 | 0.9818 | 0.0061 | 0.0121 | 5 |
| aligned_dense_s1 | calibration | main, equal marker | pairs | at least one mixed | 809 | 150 | 0.1767 | 0.1215 | 0.7936 | 0.1742 | 0.0322 | 28 |
| aligned_dense_s1 | calibration | main, equal marker | multi | all | 7126 | 719 | 0.1110 | 0.0787 | 0.8750 | 0.1095 | 0.0155 | 33 |
| aligned_dense_s1 | calibration | main, equal marker | multi | both pure | 3324 | 36 | 0.0123 | 0.0064 | 0.9783 | 0.0123 | 0.0094 | 5 |
| aligned_dense_s1 | calibration | main, equal marker | multi | at least one mixed | 3802 | 683 | 0.1826 | 0.1314 | 0.8000 | 0.1801 | 0.0198 | 33 |
| aligned_dense_s1 | calibration | main, changed marker | native | all | 2699 | 9 | 0.0034 | 0.0012 | 0.9790 | 0.0034 | 0.0176 | 6 |
| aligned_dense_s1 | calibration | main, changed marker | native | both pure | 2699 | 9 | 0.0034 | 0.0012 | 0.9790 | 0.0034 | 0.0176 | 6 |
| aligned_dense_s1 | calibration | main, changed marker | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| aligned_dense_s1 | calibration | main, changed marker | pairs | all | 1337 | 17 | 0.0104 | 0.0048 | 0.9718 | 0.0104 | 0.0177 | 8 |
| aligned_dense_s1 | calibration | main, changed marker | pairs | both pure | 1093 | 9 | 0.0073 | 0.0017 | 0.9768 | 0.0073 | 0.0160 | 4 |
| aligned_dense_s1 | calibration | main, changed marker | pairs | at least one mixed | 244 | 8 | 0.0219 | 0.0172 | 0.9563 | 0.0219 | 0.0219 | 4 |
| aligned_dense_s1 | calibration | main, changed marker | multi | all | 1370 | 33 | 0.0193 | 0.0105 | 0.9707 | 0.0187 | 0.0105 | 11 |
| aligned_dense_s1 | calibration | main, changed marker | multi | both pure | 964 | 4 | 0.0032 | 0.0012 | 0.9905 | 0.0032 | 0.0064 | 2 |
| aligned_dense_s1 | calibration | main, changed marker | multi | at least one mixed | 406 | 29 | 0.0560 | 0.0320 | 0.9266 | 0.0541 | 0.0193 | 10 |
| aligned_dense_s1 | calibration | broad | native | all | 14358 | 141 | 0.0122 | 0.0057 | 0.9717 | 0.0122 | 0.0161 | 13 |
| aligned_dense_s1 | calibration | broad | native | both pure | 14358 | 141 | 0.0122 | 0.0057 | 0.9717 | 0.0122 | 0.0161 | 13 |
| aligned_dense_s1 | calibration | broad | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| aligned_dense_s1 | calibration | broad | pairs | all | 5820 | 298 | 0.0565 | 0.0393 | 0.9275 | 0.0556 | 0.0169 | 30 |
| aligned_dense_s1 | calibration | broad | pairs | both pure | 4356 | 34 | 0.0062 | 0.0035 | 0.9825 | 0.0062 | 0.0113 | 5 |
| aligned_dense_s1 | calibration | broad | pairs | at least one mixed | 1464 | 264 | 0.1752 | 0.1249 | 0.7988 | 0.1723 | 0.0290 | 29 |
| aligned_dense_s1 | calibration | broad | multi | all | 11872 | 1254 | 0.1195 | 0.0866 | 0.8676 | 0.1181 | 0.0143 | 34 |
| aligned_dense_s1 | calibration | broad | multi | both pure | 5714 | 59 | 0.0118 | 0.0081 | 0.9780 | 0.0118 | 0.0102 | 5 |
| aligned_dense_s1 | calibration | broad | multi | at least one mixed | 6158 | 1195 | 0.1961 | 0.1424 | 0.7896 | 0.1936 | 0.0168 | 34 |
| aligned_dense_s2 | calibration | main | native | all | 10867 | 71 | 0.0083 | 0.0034 | 0.9758 | 0.0083 | 0.0159 | 10 |
| aligned_dense_s2 | calibration | main | native | both pure | 10867 | 71 | 0.0083 | 0.0034 | 0.9758 | 0.0083 | 0.0159 | 10 |
| aligned_dense_s2 | calibration | main | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| aligned_dense_s2 | calibration | main | pairs | all | 4492 | 155 | 0.0411 | 0.0287 | 0.9448 | 0.0407 | 0.0145 | 26 |
| aligned_dense_s2 | calibration | main | pairs | both pure | 3439 | 18 | 0.0042 | 0.0018 | 0.9871 | 0.0042 | 0.0087 | 5 |
| aligned_dense_s2 | calibration | main | pairs | at least one mixed | 1053 | 137 | 0.1364 | 0.0985 | 0.8370 | 0.1348 | 0.0282 | 25 |
| aligned_dense_s2 | calibration | main | multi | all | 8496 | 588 | 0.0836 | 0.0626 | 0.9029 | 0.0826 | 0.0144 | 33 |
| aligned_dense_s2 | calibration | main | multi | both pure | 4288 | 36 | 0.0101 | 0.0049 | 0.9812 | 0.0101 | 0.0086 | 7 |
| aligned_dense_s2 | calibration | main | multi | at least one mixed | 4208 | 552 | 0.1403 | 0.1074 | 0.8428 | 0.1386 | 0.0186 | 33 |
| aligned_dense_s2 | calibration | main, equal marker | native | all | 8168 | 57 | 0.0088 | 0.0037 | 0.9751 | 0.0088 | 0.0161 | 9 |
| aligned_dense_s2 | calibration | main, equal marker | native | both pure | 8168 | 57 | 0.0088 | 0.0037 | 0.9751 | 0.0088 | 0.0161 | 9 |
| aligned_dense_s2 | calibration | main, equal marker | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| aligned_dense_s2 | calibration | main, equal marker | pairs | all | 3155 | 138 | 0.0477 | 0.0332 | 0.9366 | 0.0472 | 0.0163 | 26 |
| aligned_dense_s2 | calibration | main, equal marker | pairs | both pure | 2346 | 12 | 0.0056 | 0.0021 | 0.9858 | 0.0056 | 0.0086 | 5 |
| aligned_dense_s2 | calibration | main, equal marker | pairs | at least one mixed | 809 | 126 | 0.1508 | 0.1096 | 0.8167 | 0.1490 | 0.0343 | 25 |
| aligned_dense_s2 | calibration | main, equal marker | multi | all | 7126 | 568 | 0.0878 | 0.0659 | 0.8984 | 0.0868 | 0.0148 | 33 |
| aligned_dense_s2 | calibration | main, equal marker | multi | both pure | 3324 | 34 | 0.0115 | 0.0055 | 0.9795 | 0.0115 | 0.0090 | 7 |
| aligned_dense_s2 | calibration | main, equal marker | multi | at least one mixed | 3802 | 534 | 0.1432 | 0.1100 | 0.8396 | 0.1415 | 0.0189 | 33 |
| aligned_dense_s2 | calibration | main, changed marker | native | all | 2699 | 14 | 0.0040 | 0.0008 | 0.9805 | 0.0040 | 0.0155 | 6 |
| aligned_dense_s2 | calibration | main, changed marker | native | both pure | 2699 | 14 | 0.0040 | 0.0008 | 0.9805 | 0.0040 | 0.0155 | 6 |
| aligned_dense_s2 | calibration | main, changed marker | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| aligned_dense_s2 | calibration | main, changed marker | pairs | all | 1337 | 17 | 0.0098 | 0.0040 | 0.9755 | 0.0098 | 0.0147 | 10 |
| aligned_dense_s2 | calibration | main, changed marker | pairs | both pure | 1093 | 6 | 0.0044 | 0.0012 | 0.9826 | 0.0044 | 0.0131 | 4 |
| aligned_dense_s2 | calibration | main, changed marker | pairs | at least one mixed | 244 | 11 | 0.0312 | 0.0155 | 0.9500 | 0.0312 | 0.0187 | 7 |
| aligned_dense_s2 | calibration | main, changed marker | multi | all | 1370 | 20 | 0.0123 | 0.0072 | 0.9795 | 0.0123 | 0.0082 | 10 |
| aligned_dense_s2 | calibration | main, changed marker | multi | both pure | 964 | 2 | 0.0016 | 0.0012 | 0.9952 | 0.0016 | 0.0032 | 1 |
| aligned_dense_s2 | calibration | main, changed marker | multi | at least one mixed | 406 | 18 | 0.0367 | 0.0208 | 0.9440 | 0.0367 | 0.0193 | 9 |
| aligned_dense_s2 | calibration | broad | native | all | 14358 | 119 | 0.0101 | 0.0057 | 0.9740 | 0.0101 | 0.0159 | 10 |
| aligned_dense_s2 | calibration | broad | native | both pure | 14358 | 119 | 0.0101 | 0.0057 | 0.9740 | 0.0101 | 0.0159 | 10 |
| aligned_dense_s2 | calibration | broad | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| aligned_dense_s2 | calibration | broad | pairs | all | 5820 | 263 | 0.0493 | 0.0361 | 0.9355 | 0.0486 | 0.0159 | 29 |
| aligned_dense_s2 | calibration | broad | pairs | both pure | 4356 | 23 | 0.0048 | 0.0031 | 0.9867 | 0.0048 | 0.0085 | 5 |
| aligned_dense_s2 | calibration | broad | pairs | at least one mixed | 1464 | 240 | 0.1541 | 0.1150 | 0.8157 | 0.1516 | 0.0327 | 28 |
| aligned_dense_s2 | calibration | broad | multi | all | 11872 | 997 | 0.0959 | 0.0738 | 0.8902 | 0.0946 | 0.0152 | 35 |
| aligned_dense_s2 | calibration | broad | multi | both pure | 5714 | 48 | 0.0101 | 0.0072 | 0.9801 | 0.0101 | 0.0098 | 8 |
| aligned_dense_s2 | calibration | broad | multi | at least one mixed | 6158 | 949 | 0.1567 | 0.1211 | 0.8267 | 0.1546 | 0.0187 | 34 |
| center_only_s0 | calibration | main | native | all | 10867 | 128 | 0.0161 | 0.0107 | 0.9703 | 0.0159 | 0.0137 | 11 |
| center_only_s0 | calibration | main | native | both pure | 10867 | 128 | 0.0161 | 0.0107 | 0.9703 | 0.0159 | 0.0137 | 11 |
| center_only_s0 | calibration | main | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| center_only_s0 | calibration | main | pairs | all | 4492 | 319 | 0.0857 | 0.0555 | 0.8934 | 0.0832 | 0.0234 | 29 |
| center_only_s0 | calibration | main | pairs | both pure | 3439 | 22 | 0.0050 | 0.0065 | 0.9890 | 0.0050 | 0.0060 | 3 |
| center_only_s0 | calibration | main | pairs | at least one mixed | 1053 | 297 | 0.2952 | 0.1833 | 0.6461 | 0.2862 | 0.0677 | 28 |
| center_only_s0 | calibration | main | multi | all | 8496 | 1833 | 0.2571 | 0.1550 | 0.7290 | 0.2501 | 0.0209 | 34 |
| center_only_s0 | calibration | main | multi | both pure | 4288 | 53 | 0.0151 | 0.0147 | 0.9792 | 0.0151 | 0.0058 | 8 |
| center_only_s0 | calibration | main | multi | at least one mixed | 4208 | 1780 | 0.4452 | 0.2639 | 0.5347 | 0.4329 | 0.0324 | 34 |
| center_only_s0 | calibration | main, equal marker | native | all | 8168 | 115 | 0.0170 | 0.0112 | 0.9699 | 0.0167 | 0.0134 | 11 |
| center_only_s0 | calibration | main, equal marker | native | both pure | 8168 | 115 | 0.0170 | 0.0112 | 0.9699 | 0.0167 | 0.0134 | 11 |
| center_only_s0 | calibration | main, equal marker | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| center_only_s0 | calibration | main, equal marker | pairs | all | 3155 | 283 | 0.0991 | 0.0636 | 0.8776 | 0.0958 | 0.0266 | 29 |
| center_only_s0 | calibration | main, equal marker | pairs | both pure | 2346 | 14 | 0.0061 | 0.0068 | 0.9879 | 0.0061 | 0.0061 | 3 |
| center_only_s0 | calibration | main, equal marker | pairs | at least one mixed | 809 | 269 | 0.3279 | 0.2038 | 0.6067 | 0.3167 | 0.0766 | 28 |
| center_only_s0 | calibration | main, equal marker | multi | all | 7126 | 1742 | 0.2704 | 0.1631 | 0.7144 | 0.2636 | 0.0220 | 34 |
| center_only_s0 | calibration | main, equal marker | multi | both pure | 3324 | 48 | 0.0164 | 0.0155 | 0.9776 | 0.0164 | 0.0060 | 8 |
| center_only_s0 | calibration | main, equal marker | multi | at least one mixed | 3802 | 1694 | 0.4561 | 0.2708 | 0.5222 | 0.4442 | 0.0336 | 34 |
| center_only_s0 | calibration | main, changed marker | native | all | 2699 | 13 | 0.0055 | 0.0053 | 0.9799 | 0.0052 | 0.0149 | 5 |
| center_only_s0 | calibration | main, changed marker | native | both pure | 2699 | 13 | 0.0055 | 0.0053 | 0.9799 | 0.0052 | 0.0149 | 5 |
| center_only_s0 | calibration | main, changed marker | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| center_only_s0 | calibration | main, changed marker | pairs | all | 1337 | 36 | 0.0220 | 0.0117 | 0.9602 | 0.0208 | 0.0190 | 15 |
| center_only_s0 | calibration | main, changed marker | pairs | both pure | 1093 | 8 | 0.0058 | 0.0048 | 0.9862 | 0.0058 | 0.0080 | 2 |
| center_only_s0 | calibration | main, changed marker | pairs | at least one mixed | 244 | 28 | 0.0875 | 0.0409 | 0.8562 | 0.0813 | 0.0625 | 13 |
| center_only_s0 | calibration | main, changed marker | multi | all | 1370 | 91 | 0.0562 | 0.0254 | 0.9256 | 0.0509 | 0.0234 | 19 |
| center_only_s0 | calibration | main, changed marker | multi | both pure | 964 | 5 | 0.0048 | 0.0069 | 0.9921 | 0.0048 | 0.0032 | 2 |
| center_only_s0 | calibration | main, changed marker | multi | at least one mixed | 406 | 86 | 0.1737 | 0.0684 | 0.7741 | 0.1564 | 0.0695 | 18 |
| center_only_s0 | calibration | broad | native | all | 14358 | 185 | 0.0177 | 0.0124 | 0.9686 | 0.0174 | 0.0140 | 12 |
| center_only_s0 | calibration | broad | native | both pure | 14358 | 185 | 0.0177 | 0.0124 | 0.9686 | 0.0174 | 0.0140 | 12 |
| center_only_s0 | calibration | broad | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| center_only_s0 | calibration | broad | pairs | all | 5820 | 509 | 0.0993 | 0.0647 | 0.8797 | 0.0955 | 0.0249 | 30 |
| center_only_s0 | calibration | broad | pairs | both pure | 4356 | 27 | 0.0054 | 0.0073 | 0.9889 | 0.0054 | 0.0057 | 3 |
| center_only_s0 | calibration | broad | pairs | at least one mixed | 1464 | 482 | 0.3244 | 0.2030 | 0.6192 | 0.3109 | 0.0699 | 29 |
| center_only_s0 | calibration | broad | multi | all | 11872 | 2813 | 0.2743 | 0.1671 | 0.7121 | 0.2673 | 0.0206 | 34 |
| center_only_s0 | calibration | broad | multi | both pure | 5714 | 75 | 0.0160 | 0.0166 | 0.9780 | 0.0160 | 0.0060 | 9 |
| center_only_s0 | calibration | broad | multi | at least one mixed | 6158 | 2738 | 0.4585 | 0.2744 | 0.5227 | 0.4466 | 0.0307 | 34 |
| center_only_s1 | calibration | main | native | all | 10867 | 312 | 0.0304 | 0.0133 | 0.9088 | 0.0281 | 0.0631 | 20 |
| center_only_s1 | calibration | main | native | both pure | 10867 | 312 | 0.0304 | 0.0133 | 0.9088 | 0.0281 | 0.0631 | 20 |
| center_only_s1 | calibration | main | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| center_only_s1 | calibration | main | pairs | all | 4492 | 407 | 0.1100 | 0.0494 | 0.8293 | 0.1020 | 0.0687 | 32 |
| center_only_s1 | calibration | main | pairs | both pure | 3439 | 97 | 0.0285 | 0.0100 | 0.9212 | 0.0285 | 0.0503 | 13 |
| center_only_s1 | calibration | main | pairs | at least one mixed | 1053 | 310 | 0.3228 | 0.1516 | 0.5910 | 0.2942 | 0.1148 | 30 |
| center_only_s1 | calibration | main | multi | all | 8496 | 1995 | 0.2714 | 0.1323 | 0.6953 | 0.2541 | 0.0506 | 35 |
| center_only_s1 | calibration | main | multi | both pure | 4288 | 184 | 0.0447 | 0.0182 | 0.9275 | 0.0426 | 0.0299 | 15 |
| center_only_s1 | calibration | main | multi | at least one mixed | 4208 | 1811 | 0.4478 | 0.2207 | 0.5155 | 0.4188 | 0.0657 | 34 |
| center_only_s1 | calibration | main, equal marker | native | all | 8168 | 260 | 0.0318 | 0.0143 | 0.9062 | 0.0293 | 0.0645 | 17 |
| center_only_s1 | calibration | main, equal marker | native | both pure | 8168 | 260 | 0.0318 | 0.0143 | 0.9062 | 0.0293 | 0.0645 | 17 |
| center_only_s1 | calibration | main, equal marker | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| center_only_s1 | calibration | main, equal marker | pairs | all | 3155 | 347 | 0.1255 | 0.0568 | 0.8100 | 0.1159 | 0.0740 | 31 |
| center_only_s1 | calibration | main, equal marker | pairs | both pure | 2346 | 73 | 0.0326 | 0.0113 | 0.9131 | 0.0326 | 0.0544 | 13 |
| center_only_s1 | calibration | main, equal marker | pairs | at least one mixed | 809 | 274 | 0.3563 | 0.1685 | 0.5566 | 0.3230 | 0.1204 | 29 |
| center_only_s1 | calibration | main, equal marker | multi | all | 7126 | 1864 | 0.2843 | 0.1394 | 0.6810 | 0.2663 | 0.0527 | 35 |
| center_only_s1 | calibration | main, equal marker | multi | both pure | 3324 | 147 | 0.0453 | 0.0198 | 0.9242 | 0.0428 | 0.0331 | 15 |
| center_only_s1 | calibration | main, equal marker | multi | at least one mixed | 3802 | 1717 | 0.4594 | 0.2267 | 0.5036 | 0.4300 | 0.0664 | 34 |
| center_only_s1 | calibration | main, changed marker | native | all | 2699 | 52 | 0.0177 | 0.0051 | 0.9243 | 0.0177 | 0.0580 | 15 |
| center_only_s1 | calibration | main, changed marker | native | both pure | 2699 | 52 | 0.0177 | 0.0051 | 0.9243 | 0.0177 | 0.0580 | 15 |
| center_only_s1 | calibration | main, changed marker | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| center_only_s1 | calibration | main, changed marker | pairs | all | 1337 | 60 | 0.0398 | 0.0113 | 0.8905 | 0.0386 | 0.0710 | 20 |
| center_only_s1 | calibration | main, changed marker | pairs | both pure | 1093 | 24 | 0.0203 | 0.0054 | 0.9231 | 0.0203 | 0.0566 | 9 |
| center_only_s1 | calibration | main, changed marker | pairs | at least one mixed | 244 | 36 | 0.1187 | 0.0358 | 0.7594 | 0.1125 | 0.1281 | 13 |
| center_only_s1 | calibration | main, changed marker | multi | all | 1370 | 131 | 0.0826 | 0.0204 | 0.8741 | 0.0785 | 0.0474 | 21 |
| center_only_s1 | calibration | main, changed marker | multi | both pure | 964 | 37 | 0.0358 | 0.0054 | 0.9444 | 0.0350 | 0.0207 | 7 |
| center_only_s1 | calibration | main, changed marker | multi | at least one mixed | 406 | 94 | 0.1892 | 0.0548 | 0.7162 | 0.1776 | 0.1062 | 20 |
| center_only_s1 | calibration | broad | native | all | 14358 | 449 | 0.0327 | 0.0144 | 0.9077 | 0.0294 | 0.0629 | 20 |
| center_only_s1 | calibration | broad | native | both pure | 14358 | 449 | 0.0327 | 0.0144 | 0.9077 | 0.0294 | 0.0629 | 20 |
| center_only_s1 | calibration | broad | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| center_only_s1 | calibration | broad | pairs | all | 5820 | 627 | 0.1202 | 0.0566 | 0.8201 | 0.1111 | 0.0688 | 32 |
| center_only_s1 | calibration | broad | pairs | both pure | 4356 | 135 | 0.0287 | 0.0106 | 0.9231 | 0.0287 | 0.0482 | 14 |
| center_only_s1 | calibration | broad | pairs | at least one mixed | 1464 | 492 | 0.3424 | 0.1675 | 0.5730 | 0.3114 | 0.1156 | 30 |
| center_only_s1 | calibration | broad | multi | all | 11872 | 3003 | 0.2865 | 0.1407 | 0.6829 | 0.2684 | 0.0487 | 35 |
| center_only_s1 | calibration | broad | multi | both pure | 5714 | 253 | 0.0460 | 0.0197 | 0.9277 | 0.0438 | 0.0285 | 17 |
| center_only_s1 | calibration | broad | multi | at least one mixed | 6158 | 2750 | 0.4579 | 0.2270 | 0.5093 | 0.4286 | 0.0621 | 34 |
| center_only_s2 | calibration | main | native | all | 10867 | 75 | 0.0094 | 0.0081 | 0.9740 | 0.0094 | 0.0167 | 7 |
| center_only_s2 | calibration | main | native | both pure | 10867 | 75 | 0.0094 | 0.0081 | 0.9740 | 0.0094 | 0.0167 | 7 |
| center_only_s2 | calibration | main | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| center_only_s2 | calibration | main | pairs | all | 4492 | 340 | 0.0878 | 0.0563 | 0.8926 | 0.0864 | 0.0209 | 29 |
| center_only_s2 | calibration | main | pairs | both pure | 3439 | 20 | 0.0038 | 0.0043 | 0.9863 | 0.0038 | 0.0098 | 3 |
| center_only_s2 | calibration | main | pairs | at least one mixed | 1053 | 320 | 0.3066 | 0.1919 | 0.6500 | 0.3018 | 0.0482 | 28 |
| center_only_s2 | calibration | main | multi | all | 8496 | 1770 | 0.2468 | 0.1612 | 0.7396 | 0.2403 | 0.0201 | 34 |
| center_only_s2 | calibration | main | multi | both pure | 4288 | 39 | 0.0112 | 0.0126 | 0.9806 | 0.0112 | 0.0082 | 7 |
| center_only_s2 | calibration | main | multi | at least one mixed | 4208 | 1731 | 0.4301 | 0.2765 | 0.5523 | 0.4186 | 0.0291 | 34 |
| center_only_s2 | calibration | main, equal marker | native | all | 8168 | 68 | 0.0103 | 0.0088 | 0.9732 | 0.0103 | 0.0165 | 7 |
| center_only_s2 | calibration | main, equal marker | native | both pure | 8168 | 68 | 0.0103 | 0.0088 | 0.9732 | 0.0103 | 0.0165 | 7 |
| center_only_s2 | calibration | main, equal marker | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| center_only_s2 | calibration | main, equal marker | pairs | all | 3155 | 294 | 0.1011 | 0.0655 | 0.8761 | 0.0996 | 0.0243 | 29 |
| center_only_s2 | calibration | main, equal marker | pairs | both pure | 2346 | 14 | 0.0051 | 0.0050 | 0.9863 | 0.0051 | 0.0086 | 3 |
| center_only_s2 | calibration | main, equal marker | pairs | at least one mixed | 809 | 280 | 0.3383 | 0.2149 | 0.6049 | 0.3330 | 0.0622 | 28 |
| center_only_s2 | calibration | main, equal marker | multi | all | 7126 | 1683 | 0.2602 | 0.1701 | 0.7243 | 0.2535 | 0.0222 | 34 |
| center_only_s2 | calibration | main, equal marker | multi | both pure | 3324 | 38 | 0.0126 | 0.0137 | 0.9788 | 0.0126 | 0.0086 | 7 |
| center_only_s2 | calibration | main, equal marker | multi | at least one mixed | 3802 | 1645 | 0.4413 | 0.2842 | 0.5384 | 0.4297 | 0.0319 | 34 |
| center_only_s2 | calibration | main, changed marker | native | all | 2699 | 7 | 0.0025 | 0.0025 | 0.9852 | 0.0025 | 0.0124 | 3 |
| center_only_s2 | calibration | main, changed marker | native | both pure | 2699 | 7 | 0.0025 | 0.0025 | 0.9852 | 0.0025 | 0.0124 | 3 |
| center_only_s2 | calibration | main, changed marker | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| center_only_s2 | calibration | main, changed marker | pairs | all | 1337 | 46 | 0.0288 | 0.0110 | 0.9608 | 0.0288 | 0.0104 | 16 |
| center_only_s2 | calibration | main, changed marker | pairs | both pure | 1093 | 6 | 0.0044 | 0.0022 | 0.9840 | 0.0044 | 0.0116 | 2 |
| center_only_s2 | calibration | main, changed marker | pairs | at least one mixed | 244 | 40 | 0.1281 | 0.0470 | 0.8688 | 0.1281 | 0.0031 | 15 |
| center_only_s2 | calibration | main, changed marker | multi | all | 1370 | 87 | 0.0527 | 0.0247 | 0.9368 | 0.0515 | 0.0117 | 19 |
| center_only_s2 | calibration | main, changed marker | multi | both pure | 964 | 1 | 0.0008 | 0.0038 | 0.9936 | 0.0008 | 0.0056 | 1 |
| center_only_s2 | calibration | main, changed marker | multi | at least one mixed | 406 | 86 | 0.1718 | 0.0732 | 0.8069 | 0.1680 | 0.0251 | 18 |
| center_only_s2 | calibration | broad | native | all | 14358 | 120 | 0.0108 | 0.0094 | 0.9720 | 0.0108 | 0.0172 | 8 |
| center_only_s2 | calibration | broad | native | both pure | 14358 | 120 | 0.0108 | 0.0094 | 0.9720 | 0.0108 | 0.0172 | 8 |
| center_only_s2 | calibration | broad | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| center_only_s2 | calibration | broad | pairs | all | 5820 | 523 | 0.0988 | 0.0654 | 0.8798 | 0.0968 | 0.0234 | 30 |
| center_only_s2 | calibration | broad | pairs | both pure | 4356 | 24 | 0.0039 | 0.0049 | 0.9861 | 0.0039 | 0.0100 | 3 |
| center_only_s2 | calibration | broad | pairs | at least one mixed | 1464 | 499 | 0.3270 | 0.2111 | 0.6253 | 0.3205 | 0.0542 | 29 |
| center_only_s2 | calibration | broad | multi | all | 11872 | 2728 | 0.2645 | 0.1726 | 0.7221 | 0.2582 | 0.0197 | 34 |
| center_only_s2 | calibration | broad | multi | both pure | 5714 | 59 | 0.0127 | 0.0142 | 0.9786 | 0.0127 | 0.0087 | 7 |
| center_only_s2 | calibration | broad | multi | at least one mixed | 6158 | 2669 | 0.4442 | 0.2855 | 0.5394 | 0.4334 | 0.0272 | 34 |
| shuffled_dense_s0 | calibration | main | native | all | 10867 | 58 | 0.0064 | 0.0045 | 0.9812 | 0.0064 | 0.0123 | 6 |
| shuffled_dense_s0 | calibration | main | native | both pure | 10867 | 58 | 0.0064 | 0.0045 | 0.9812 | 0.0064 | 0.0123 | 6 |
| shuffled_dense_s0 | calibration | main | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| shuffled_dense_s0 | calibration | main | pairs | all | 4492 | 329 | 0.0832 | 0.0531 | 0.9016 | 0.0822 | 0.0161 | 29 |
| shuffled_dense_s0 | calibration | main | pairs | both pure | 3439 | 20 | 0.0038 | 0.0021 | 0.9900 | 0.0038 | 0.0062 | 2 |
| shuffled_dense_s0 | calibration | main | pairs | at least one mixed | 1053 | 309 | 0.2894 | 0.1862 | 0.6731 | 0.2858 | 0.0411 | 28 |
| shuffled_dense_s0 | calibration | main | multi | all | 8496 | 1726 | 0.2410 | 0.1632 | 0.7485 | 0.2383 | 0.0133 | 34 |
| shuffled_dense_s0 | calibration | main | multi | both pure | 4288 | 32 | 0.0088 | 0.0079 | 0.9855 | 0.0088 | 0.0058 | 5 |
| shuffled_dense_s0 | calibration | main | multi | at least one mixed | 4208 | 1694 | 0.4215 | 0.2840 | 0.5646 | 0.4165 | 0.0189 | 34 |
| shuffled_dense_s0 | calibration | main, equal marker | native | all | 8168 | 51 | 0.0072 | 0.0049 | 0.9812 | 0.0072 | 0.0116 | 6 |
| shuffled_dense_s0 | calibration | main, equal marker | native | both pure | 8168 | 51 | 0.0072 | 0.0049 | 0.9812 | 0.0072 | 0.0116 | 6 |
| shuffled_dense_s0 | calibration | main, equal marker | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| shuffled_dense_s0 | calibration | main, equal marker | pairs | all | 3155 | 282 | 0.0967 | 0.0617 | 0.8855 | 0.0955 | 0.0189 | 29 |
| shuffled_dense_s0 | calibration | main, equal marker | pairs | both pure | 2346 | 12 | 0.0047 | 0.0024 | 0.9893 | 0.0047 | 0.0061 | 2 |
| shuffled_dense_s0 | calibration | main, equal marker | pairs | at least one mixed | 809 | 270 | 0.3232 | 0.2079 | 0.6307 | 0.3193 | 0.0500 | 28 |
| shuffled_dense_s0 | calibration | main, equal marker | multi | all | 7126 | 1642 | 0.2544 | 0.1722 | 0.7330 | 0.2516 | 0.0154 | 34 |
| shuffled_dense_s0 | calibration | main, equal marker | multi | both pure | 3324 | 28 | 0.0097 | 0.0087 | 0.9842 | 0.0097 | 0.0060 | 5 |
| shuffled_dense_s0 | calibration | main, equal marker | multi | at least one mixed | 3802 | 1614 | 0.4331 | 0.2917 | 0.5497 | 0.4281 | 0.0221 | 34 |
| shuffled_dense_s0 | calibration | main, changed marker | native | all | 2699 | 7 | 0.0020 | 0.0012 | 0.9847 | 0.0020 | 0.0133 | 3 |
| shuffled_dense_s0 | calibration | main, changed marker | native | both pure | 2699 | 7 | 0.0020 | 0.0012 | 0.9847 | 0.0020 | 0.0133 | 3 |
| shuffled_dense_s0 | calibration | main, changed marker | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| shuffled_dense_s0 | calibration | main, changed marker | pairs | all | 1337 | 47 | 0.0282 | 0.0087 | 0.9608 | 0.0282 | 0.0110 | 15 |
| shuffled_dense_s0 | calibration | main, changed marker | pairs | both pure | 1093 | 8 | 0.0058 | 0.0018 | 0.9855 | 0.0058 | 0.0087 | 2 |
| shuffled_dense_s0 | calibration | main, changed marker | pairs | at least one mixed | 244 | 39 | 0.1187 | 0.0376 | 0.8625 | 0.1187 | 0.0187 | 13 |
| shuffled_dense_s0 | calibration | main, changed marker | multi | all | 1370 | 84 | 0.0492 | 0.0221 | 0.9444 | 0.0480 | 0.0076 | 18 |
| shuffled_dense_s0 | calibration | main, changed marker | multi | both pure | 964 | 4 | 0.0032 | 0.0020 | 0.9936 | 0.0032 | 0.0032 | 1 |
| shuffled_dense_s0 | calibration | main, changed marker | multi | at least one mixed | 406 | 80 | 0.1544 | 0.0685 | 0.8320 | 0.1506 | 0.0174 | 17 |
| shuffled_dense_s0 | calibration | broad | native | all | 14358 | 86 | 0.0072 | 0.0051 | 0.9807 | 0.0072 | 0.0121 | 6 |
| shuffled_dense_s0 | calibration | broad | native | both pure | 14358 | 86 | 0.0072 | 0.0051 | 0.9807 | 0.0072 | 0.0121 | 6 |
| shuffled_dense_s0 | calibration | broad | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| shuffled_dense_s0 | calibration | broad | pairs | all | 5820 | 518 | 0.0953 | 0.0623 | 0.8887 | 0.0937 | 0.0176 | 30 |
| shuffled_dense_s0 | calibration | broad | pairs | both pure | 4356 | 24 | 0.0039 | 0.0024 | 0.9902 | 0.0039 | 0.0059 | 2 |
| shuffled_dense_s0 | calibration | broad | pairs | at least one mixed | 1464 | 494 | 0.3149 | 0.2073 | 0.6453 | 0.3094 | 0.0452 | 29 |
| shuffled_dense_s0 | calibration | broad | multi | all | 11872 | 2668 | 0.2582 | 0.1756 | 0.7321 | 0.2551 | 0.0127 | 34 |
| shuffled_dense_s0 | calibration | broad | multi | both pure | 5714 | 50 | 0.0098 | 0.0088 | 0.9839 | 0.0098 | 0.0062 | 6 |
| shuffled_dense_s0 | calibration | broad | multi | at least one mixed | 6158 | 2618 | 0.4353 | 0.2947 | 0.5529 | 0.4301 | 0.0171 | 34 |
| shuffled_dense_s1 | calibration | main | native | all | 10867 | 58 | 0.0074 | 0.0043 | 0.9805 | 0.0074 | 0.0121 | 7 |
| shuffled_dense_s1 | calibration | main | native | both pure | 10867 | 58 | 0.0074 | 0.0043 | 0.9805 | 0.0074 | 0.0121 | 7 |
| shuffled_dense_s1 | calibration | main | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| shuffled_dense_s1 | calibration | main | pairs | all | 4492 | 316 | 0.0818 | 0.0541 | 0.8992 | 0.0806 | 0.0202 | 29 |
| shuffled_dense_s1 | calibration | main | pairs | both pure | 3439 | 15 | 0.0040 | 0.0029 | 0.9867 | 0.0040 | 0.0092 | 3 |
| shuffled_dense_s1 | calibration | main | pairs | at least one mixed | 1053 | 301 | 0.2837 | 0.1877 | 0.6734 | 0.2796 | 0.0470 | 28 |
| shuffled_dense_s1 | calibration | main | multi | all | 8496 | 1740 | 0.2421 | 0.1637 | 0.7463 | 0.2384 | 0.0154 | 34 |
| shuffled_dense_s1 | calibration | main | multi | both pure | 4288 | 37 | 0.0105 | 0.0082 | 0.9834 | 0.0105 | 0.0061 | 5 |
| shuffled_dense_s1 | calibration | main | multi | at least one mixed | 4208 | 1703 | 0.4222 | 0.2846 | 0.5621 | 0.4156 | 0.0223 | 34 |
| shuffled_dense_s1 | calibration | main, equal marker | native | all | 8168 | 54 | 0.0080 | 0.0048 | 0.9805 | 0.0080 | 0.0116 | 7 |
| shuffled_dense_s1 | calibration | main, equal marker | native | both pure | 8168 | 54 | 0.0080 | 0.0048 | 0.9805 | 0.0080 | 0.0116 | 7 |
| shuffled_dense_s1 | calibration | main, equal marker | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| shuffled_dense_s1 | calibration | main, equal marker | pairs | all | 3155 | 276 | 0.0945 | 0.0628 | 0.8827 | 0.0932 | 0.0241 | 29 |
| shuffled_dense_s1 | calibration | main, equal marker | pairs | both pure | 2346 | 10 | 0.0042 | 0.0034 | 0.9860 | 0.0042 | 0.0098 | 3 |
| shuffled_dense_s1 | calibration | main, equal marker | pairs | at least one mixed | 809 | 266 | 0.3169 | 0.2095 | 0.6293 | 0.3123 | 0.0584 | 28 |
| shuffled_dense_s1 | calibration | main, equal marker | multi | all | 7126 | 1651 | 0.2554 | 0.1727 | 0.7310 | 0.2515 | 0.0175 | 34 |
| shuffled_dense_s1 | calibration | main, equal marker | multi | both pure | 3324 | 33 | 0.0117 | 0.0091 | 0.9819 | 0.0117 | 0.0064 | 5 |
| shuffled_dense_s1 | calibration | main, equal marker | multi | at least one mixed | 3802 | 1618 | 0.4335 | 0.2923 | 0.5477 | 0.4269 | 0.0254 | 34 |
| shuffled_dense_s1 | calibration | main, changed marker | native | all | 2699 | 4 | 0.0015 | 0.0011 | 0.9814 | 0.0015 | 0.0170 | 2 |
| shuffled_dense_s1 | calibration | main, changed marker | native | both pure | 2699 | 4 | 0.0015 | 0.0011 | 0.9814 | 0.0015 | 0.0170 | 2 |
| shuffled_dense_s1 | calibration | main, changed marker | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| shuffled_dense_s1 | calibration | main, changed marker | pairs | all | 1337 | 40 | 0.0251 | 0.0091 | 0.9584 | 0.0251 | 0.0165 | 15 |
| shuffled_dense_s1 | calibration | main, changed marker | pairs | both pure | 1093 | 5 | 0.0044 | 0.0018 | 0.9826 | 0.0044 | 0.0131 | 2 |
| shuffled_dense_s1 | calibration | main, changed marker | pairs | at least one mixed | 244 | 35 | 0.1094 | 0.0397 | 0.8625 | 0.1094 | 0.0281 | 13 |
| shuffled_dense_s1 | calibration | main, changed marker | multi | all | 1370 | 89 | 0.0527 | 0.0230 | 0.9403 | 0.0515 | 0.0082 | 20 |
| shuffled_dense_s1 | calibration | main, changed marker | multi | both pure | 964 | 4 | 0.0032 | 0.0016 | 0.9936 | 0.0032 | 0.0032 | 1 |
| shuffled_dense_s1 | calibration | main, changed marker | multi | at least one mixed | 406 | 85 | 0.1660 | 0.0726 | 0.8185 | 0.1622 | 0.0193 | 19 |
| shuffled_dense_s1 | calibration | broad | native | all | 14358 | 89 | 0.0086 | 0.0049 | 0.9801 | 0.0086 | 0.0113 | 7 |
| shuffled_dense_s1 | calibration | broad | native | both pure | 14358 | 89 | 0.0086 | 0.0049 | 0.9801 | 0.0086 | 0.0113 | 7 |
| shuffled_dense_s1 | calibration | broad | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| shuffled_dense_s1 | calibration | broad | pairs | all | 5820 | 505 | 0.0944 | 0.0631 | 0.8856 | 0.0924 | 0.0220 | 30 |
| shuffled_dense_s1 | calibration | broad | pairs | both pure | 4356 | 18 | 0.0043 | 0.0032 | 0.9867 | 0.0043 | 0.0090 | 3 |
| shuffled_dense_s1 | calibration | broad | pairs | at least one mixed | 1464 | 487 | 0.3115 | 0.2078 | 0.6430 | 0.3050 | 0.0520 | 29 |
| shuffled_dense_s1 | calibration | broad | multi | all | 11872 | 2679 | 0.2590 | 0.1759 | 0.7303 | 0.2550 | 0.0147 | 35 |
| shuffled_dense_s1 | calibration | broad | multi | both pure | 5714 | 57 | 0.0123 | 0.0093 | 0.9812 | 0.0123 | 0.0065 | 6 |
| shuffled_dense_s1 | calibration | broad | multi | at least one mixed | 6158 | 2622 | 0.4349 | 0.2949 | 0.5516 | 0.4282 | 0.0202 | 34 |
| shuffled_dense_s2 | calibration | main | native | all | 10867 | 66 | 0.0084 | 0.0049 | 0.9788 | 0.0084 | 0.0129 | 7 |
| shuffled_dense_s2 | calibration | main | native | both pure | 10867 | 66 | 0.0084 | 0.0049 | 0.9788 | 0.0084 | 0.0129 | 7 |
| shuffled_dense_s2 | calibration | main | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| shuffled_dense_s2 | calibration | main | pairs | all | 4492 | 329 | 0.0844 | 0.0534 | 0.8990 | 0.0834 | 0.0176 | 29 |
| shuffled_dense_s2 | calibration | main | pairs | both pure | 3439 | 24 | 0.0062 | 0.0025 | 0.9877 | 0.0062 | 0.0062 | 4 |
| shuffled_dense_s2 | calibration | main | pairs | at least one mixed | 1053 | 305 | 0.2872 | 0.1862 | 0.6699 | 0.2837 | 0.0464 | 28 |
| shuffled_dense_s2 | calibration | main | multi | all | 8496 | 1732 | 0.2408 | 0.1620 | 0.7476 | 0.2360 | 0.0164 | 34 |
| shuffled_dense_s2 | calibration | main | multi | both pure | 4288 | 37 | 0.0105 | 0.0080 | 0.9827 | 0.0105 | 0.0068 | 6 |
| shuffled_dense_s2 | calibration | main | multi | at least one mixed | 4208 | 1695 | 0.4200 | 0.2818 | 0.5650 | 0.4114 | 0.0236 | 34 |
| shuffled_dense_s2 | calibration | main, equal marker | native | all | 8168 | 61 | 0.0093 | 0.0053 | 0.9783 | 0.0093 | 0.0123 | 7 |
| shuffled_dense_s2 | calibration | main, equal marker | native | both pure | 8168 | 61 | 0.0093 | 0.0053 | 0.9783 | 0.0093 | 0.0123 | 7 |
| shuffled_dense_s2 | calibration | main, equal marker | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| shuffled_dense_s2 | calibration | main, equal marker | pairs | all | 3155 | 284 | 0.0984 | 0.0620 | 0.8826 | 0.0973 | 0.0201 | 29 |
| shuffled_dense_s2 | calibration | main, equal marker | pairs | both pure | 2346 | 16 | 0.0074 | 0.0029 | 0.9865 | 0.0074 | 0.0061 | 4 |
| shuffled_dense_s2 | calibration | main, equal marker | pairs | at least one mixed | 809 | 268 | 0.3220 | 0.2079 | 0.6277 | 0.3181 | 0.0542 | 28 |
| shuffled_dense_s2 | calibration | main, equal marker | multi | all | 7126 | 1645 | 0.2542 | 0.1709 | 0.7325 | 0.2492 | 0.0183 | 34 |
| shuffled_dense_s2 | calibration | main, equal marker | multi | both pure | 3324 | 33 | 0.0117 | 0.0088 | 0.9812 | 0.0117 | 0.0072 | 6 |
| shuffled_dense_s2 | calibration | main, equal marker | multi | at least one mixed | 3802 | 1612 | 0.4314 | 0.2894 | 0.5510 | 0.4227 | 0.0263 | 34 |
| shuffled_dense_s2 | calibration | main, changed marker | native | all | 2699 | 5 | 0.0015 | 0.0014 | 0.9861 | 0.0015 | 0.0124 | 2 |
| shuffled_dense_s2 | calibration | main, changed marker | native | both pure | 2699 | 5 | 0.0015 | 0.0014 | 0.9861 | 0.0015 | 0.0124 | 2 |
| shuffled_dense_s2 | calibration | main, changed marker | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| shuffled_dense_s2 | calibration | main, changed marker | pairs | all | 1337 | 45 | 0.0269 | 0.0094 | 0.9614 | 0.0269 | 0.0116 | 14 |
| shuffled_dense_s2 | calibration | main, changed marker | pairs | both pure | 1093 | 8 | 0.0058 | 0.0019 | 0.9855 | 0.0058 | 0.0087 | 2 |
| shuffled_dense_s2 | calibration | main, changed marker | pairs | at least one mixed | 244 | 37 | 0.1125 | 0.0412 | 0.8656 | 0.1125 | 0.0219 | 12 |
| shuffled_dense_s2 | calibration | main, changed marker | multi | all | 1370 | 87 | 0.0515 | 0.0211 | 0.9397 | 0.0504 | 0.0100 | 19 |
| shuffled_dense_s2 | calibration | main, changed marker | multi | both pure | 964 | 4 | 0.0032 | 0.0021 | 0.9936 | 0.0032 | 0.0032 | 1 |
| shuffled_dense_s2 | calibration | main, changed marker | multi | at least one mixed | 406 | 83 | 0.1622 | 0.0654 | 0.8166 | 0.1583 | 0.0251 | 18 |
| shuffled_dense_s2 | calibration | broad | native | all | 14358 | 104 | 0.0094 | 0.0055 | 0.9779 | 0.0094 | 0.0127 | 7 |
| shuffled_dense_s2 | calibration | broad | native | both pure | 14358 | 104 | 0.0094 | 0.0055 | 0.9779 | 0.0094 | 0.0127 | 7 |
| shuffled_dense_s2 | calibration | broad | native | at least one mixed | 0 | 0 |  |  |  |  |  | 0 |
| shuffled_dense_s2 | calibration | broad | pairs | all | 5820 | 518 | 0.0965 | 0.0622 | 0.8854 | 0.0949 | 0.0197 | 30 |
| shuffled_dense_s2 | calibration | broad | pairs | both pure | 4356 | 30 | 0.0069 | 0.0028 | 0.9872 | 0.0069 | 0.0059 | 4 |
| shuffled_dense_s2 | calibration | broad | pairs | at least one mixed | 1464 | 488 | 0.3121 | 0.2057 | 0.6412 | 0.3066 | 0.0522 | 29 |
| shuffled_dense_s2 | calibration | broad | multi | all | 11872 | 2667 | 0.2578 | 0.1737 | 0.7314 | 0.2526 | 0.0160 | 34 |
| shuffled_dense_s2 | calibration | broad | multi | both pure | 5714 | 57 | 0.0117 | 0.0089 | 0.9811 | 0.0117 | 0.0073 | 6 |
| shuffled_dense_s2 | calibration | broad | multi | at least one mixed | 6158 | 2610 | 0.4333 | 0.2914 | 0.5537 | 0.4244 | 0.0219 | 34 |

## Disagreeing pairs over all populations

| readout | half | subset | pairs | disagreeing pairs | disagreeing sessions | sessions |
|---|---|---|---|---|---|---|
| aligned_dense_s0 | selection | main | 26430 | 1118 | 36 | 36 |
| aligned_dense_s0 | selection | main, equal marker | 20310 | 1054 | 36 | 36 |
| aligned_dense_s0 | selection | main, changed marker | 6120 | 64 | 17 | 36 |
| aligned_dense_s0 | selection | broad | 35429 | 1846 | 36 | 36 |
| aligned_dense_s1 | selection | main | 26430 | 1212 | 35 | 36 |
| aligned_dense_s1 | selection | main, equal marker | 20310 | 1132 | 35 | 36 |
| aligned_dense_s1 | selection | main, changed marker | 6120 | 80 | 21 | 36 |
| aligned_dense_s1 | selection | broad | 35429 | 1980 | 36 | 36 |
| aligned_dense_s2 | selection | main | 26430 | 1109 | 35 | 36 |
| aligned_dense_s2 | selection | main, equal marker | 20310 | 1033 | 35 | 36 |
| aligned_dense_s2 | selection | main, changed marker | 6120 | 76 | 22 | 36 |
| aligned_dense_s2 | selection | broad | 35429 | 1790 | 36 | 36 |
| center_only_s0 | selection | main | 26430 | 2492 | 36 | 36 |
| center_only_s0 | selection | main, equal marker | 20310 | 2284 | 36 | 36 |
| center_only_s0 | selection | main, changed marker | 6120 | 208 | 28 | 36 |
| center_only_s0 | selection | broad | 35429 | 3853 | 36 | 36 |
| center_only_s1 | selection | main | 26430 | 3249 | 36 | 36 |
| center_only_s1 | selection | main, equal marker | 20310 | 2893 | 36 | 36 |
| center_only_s1 | selection | main, changed marker | 6120 | 356 | 32 | 36 |
| center_only_s1 | selection | broad | 35429 | 4765 | 36 | 36 |
| center_only_s2 | selection | main | 26430 | 2519 | 36 | 36 |
| center_only_s2 | selection | main, equal marker | 20310 | 2308 | 36 | 36 |
| center_only_s2 | selection | main, changed marker | 6120 | 211 | 27 | 36 |
| center_only_s2 | selection | broad | 35429 | 3862 | 36 | 36 |
| shuffled_dense_s0 | selection | main | 26430 | 2414 | 36 | 36 |
| shuffled_dense_s0 | selection | main, equal marker | 20310 | 2225 | 36 | 36 |
| shuffled_dense_s0 | selection | main, changed marker | 6120 | 189 | 27 | 36 |
| shuffled_dense_s0 | selection | broad | 35429 | 3725 | 36 | 36 |
| shuffled_dense_s1 | selection | main | 26430 | 2368 | 36 | 36 |
| shuffled_dense_s1 | selection | main, equal marker | 20310 | 2181 | 36 | 36 |
| shuffled_dense_s1 | selection | main, changed marker | 6120 | 187 | 28 | 36 |
| shuffled_dense_s1 | selection | broad | 35429 | 3675 | 36 | 36 |
| shuffled_dense_s2 | selection | main | 26430 | 2469 | 36 | 36 |
| shuffled_dense_s2 | selection | main, equal marker | 20310 | 2272 | 36 | 36 |
| shuffled_dense_s2 | selection | main, changed marker | 6120 | 197 | 27 | 36 |
| shuffled_dense_s2 | selection | broad | 35429 | 3795 | 36 | 36 |
| aligned_dense_s0 | calibration | main | 23855 | 957 | 33 | 36 |
| aligned_dense_s0 | calibration | main, equal marker | 18449 | 902 | 33 | 36 |
| aligned_dense_s0 | calibration | main, changed marker | 5406 | 55 | 19 | 36 |
| aligned_dense_s0 | calibration | broad | 32050 | 1588 | 35 | 36 |
| aligned_dense_s1 | calibration | main | 23855 | 1011 | 34 | 36 |
| aligned_dense_s1 | calibration | main, equal marker | 18449 | 952 | 34 | 36 |
| aligned_dense_s1 | calibration | main, changed marker | 5406 | 59 | 19 | 36 |
| aligned_dense_s1 | calibration | broad | 32050 | 1693 | 35 | 36 |
| aligned_dense_s2 | calibration | main | 23855 | 814 | 33 | 36 |
| aligned_dense_s2 | calibration | main, equal marker | 18449 | 763 | 33 | 36 |
| aligned_dense_s2 | calibration | main, changed marker | 5406 | 51 | 19 | 36 |
| aligned_dense_s2 | calibration | broad | 32050 | 1379 | 35 | 36 |
| center_only_s0 | calibration | main | 23855 | 2280 | 34 | 36 |
| center_only_s0 | calibration | main, equal marker | 18449 | 2140 | 34 | 36 |
| center_only_s0 | calibration | main, changed marker | 5406 | 140 | 26 | 36 |
| center_only_s0 | calibration | broad | 32050 | 3507 | 34 | 36 |
| center_only_s1 | calibration | main | 23855 | 2714 | 35 | 36 |
| center_only_s1 | calibration | main, equal marker | 18449 | 2471 | 35 | 36 |
| center_only_s1 | calibration | main, changed marker | 5406 | 243 | 30 | 36 |
| center_only_s1 | calibration | broad | 32050 | 4079 | 35 | 36 |
| center_only_s2 | calibration | main | 23855 | 2185 | 34 | 36 |
| center_only_s2 | calibration | main, equal marker | 18449 | 2045 | 34 | 36 |
| center_only_s2 | calibration | main, changed marker | 5406 | 140 | 27 | 36 |
| center_only_s2 | calibration | broad | 32050 | 3371 | 34 | 36 |
| shuffled_dense_s0 | calibration | main | 23855 | 2113 | 34 | 36 |
| shuffled_dense_s0 | calibration | main, equal marker | 18449 | 1975 | 34 | 36 |
| shuffled_dense_s0 | calibration | main, changed marker | 5406 | 138 | 26 | 36 |
| shuffled_dense_s0 | calibration | broad | 32050 | 3272 | 34 | 36 |
| shuffled_dense_s1 | calibration | main | 23855 | 2114 | 34 | 36 |
| shuffled_dense_s1 | calibration | main, equal marker | 18449 | 1981 | 34 | 36 |
| shuffled_dense_s1 | calibration | main, changed marker | 5406 | 133 | 27 | 36 |
| shuffled_dense_s1 | calibration | broad | 32050 | 3273 | 35 | 36 |
| shuffled_dense_s2 | calibration | main | 23855 | 2127 | 34 | 36 |
| shuffled_dense_s2 | calibration | main, equal marker | 18449 | 1990 | 34 | 36 |
| shuffled_dense_s2 | calibration | main, changed marker | 5406 | 137 | 28 | 36 |
| shuffled_dense_s2 | calibration | broad | 32050 | 3289 | 34 | 36 |

## Roles of the disagreeing pairs

| readout | half | subset | population | anchor pairs | marker-relative positions | marker triples | centre distance | boundary distance | selected output anchor |
|---|---|---|---|---|---|---|---|---|---|
| aligned_dense_s0 | selection | main | native | {"2-2": 2, "3-1": 2, "3-3": 2, "4-2": 2, "4-3": 1, "5-5": 1, "6-1": 31, "6-3": 2, "6-4": 2, "7-2": 33, "7-3": 1, "7-4": 2, "7-5": 2} | {"-1": 1, "-11": 1, "-3": 2, "-4": 37, "-5": 1, "-6": 33, "-7": 1, "-9": 7, "0": 3, "1": 1, "10": 1, "12": 1, "2": 4, "4": 34, "5": 1, "6": 37, "7": 1} | {"000/000": 78, "000/010": 2, "001/000": 1, "010/000": 1, "100/000": 1} | {"10": 68, "13": 2, "15": 3, "4": 1, "5": 8, "9": 1} | {"none": 83} | {"at least one": 4, "both": 0, "neither": 79} |
| aligned_dense_s0 | selection | main | pairs | {"1-1": 2, "2-1": 2, "2-2": 1, "3-1": 6, "3-2": 2, "3-3": 1, "4-1": 2, "4-2": 9, "5-1": 10, "5-3": 26, "5-5": 2, "6-1": 44, "6-2": 12, "6-4": 22, "6-5": 2, "6-6": 1, "7-1": 10, "7-2": 45, "7-3": 20, "7-5": 33, "7-6": 3, "7-7": 1} | {"-1": 6, "-10": 9, "-11": 11, "-12": 6, "-13": 13, "-2": 23, "-3": 13, "-4": 52, "-5": 6, "-6": 45, "-7": 5, "-8": 20, "-9": 11, "0": 23, "1": 8, "10": 12, "12": 24, "2": 45, "3": 10, "4": 57, "5": 12, "6": 65, "7": 24, "8": 12} | {"000/000": 219, "000/001": 5, "000/010": 16, "000/100": 4, "001/000": 1, "010/000": 7, "100/000": 4} | {"10": 142, "13": 1, "15": 34, "18": 2, "19": 3, "20": 9, "23": 2, "25": 1, "4": 2, "5": 56, "9": 4} | {"0-5": 216, "6-10": 7, ">10": 6, "none": 27} | {"at least one": 29, "both": 0, "neither": 227} |
| aligned_dense_s0 | selection | main | multi | {"2-1": 2, "2-2": 4, "3-1": 12, "3-2": 1, "4-2": 8, "5-1": 8, "5-2": 1, "5-3": 16, "6-1": 323, "6-2": 7, "6-3": 6, "6-4": 14, "7-1": 14, "7-2": 325, "7-3": 14, "7-5": 23, "7-7": 1} | {"-1": 2, "-10": 2, "-11": 12, "-12": 4, "-13": 12, "-2": 25, "-3": 1, "-4": 332, "-5": 2, "-6": 330, "-7": 6, "-8": 5, "-9": 24, "0": 18, "1": 2, "10": 7, "12": 12, "2": 33, "3": 8, "4": 338, "5": 7, "6": 355, "7": 13, "8": 8} | {"000/000": 757, "000/010": 14, "001/000": 2, "010/000": 4, "100/000": 2} | {"10": 691, "13": 4, "15": 32, "19": 7, "3": 1, "4": 1, "5": 42, "9": 1} | {"0-5": 663, "6-10": 31, ">10": 85} | {"at least one": 20, "both": 0, "neither": 759} |
| aligned_dense_s0 | selection | main, equal marker | native | {"2-2": 2, "3-1": 1, "3-3": 2, "4-2": 1, "5-5": 1, "6-1": 31, "6-3": 2, "7-2": 33, "7-3": 1, "7-4": 2, "7-5": 2} | {"-11": 1, "-3": 2, "-4": 36, "-5": 1, "-6": 32, "-7": 1, "-9": 6, "12": 1, "2": 4, "4": 34, "6": 37, "7": 1} | {"000/000": 78} | {"10": 67, "13": 2, "15": 3, "4": 1, "5": 5} | {"none": 78} | {"at least one": 0, "both": 0, "neither": 78} |
| aligned_dense_s0 | selection | main, equal marker | pairs | {"1-1": 2, "2-1": 2, "2-2": 1, "3-1": 5, "3-2": 2, "3-3": 1, "4-2": 2, "5-1": 10, "5-3": 26, "5-5": 1, "6-1": 44, "6-2": 12, "6-4": 3, "6-5": 1, "7-1": 10, "7-2": 45, "7-3": 20, "7-5": 31, "7-6": 1} | {"-10": 5, "-11": 11, "-12": 6, "-13": 12, "-2": 23, "-3": 12, "-4": 49, "-5": 4, "-6": 44, "-7": 5, "-8": 20, "-9": 11, "10": 4, "12": 24, "2": 45, "3": 9, "4": 53, "5": 4, "6": 61, "7": 24, "8": 12} | {"000/000": 219} | {"10": 130, "15": 34, "18": 2, "19": 3, "20": 9, "23": 2, "25": 1, "5": 34, "9": 4} | {"0-5": 188, "6-10": 4, ">10": 5, "none": 22} | {"at least one": 0, "both": 0, "neither": 219} |
| aligned_dense_s0 | selection | main, equal marker | multi | {"2-1": 2, "2-2": 4, "3-1": 10, "3-2": 1, "4-2": 2, "5-1": 8, "5-2": 1, "5-3": 16, "6-1": 323, "6-2": 7, "6-3": 6, "7-1": 14, "7-2": 325, "7-3": 14, "7-5": 23, "7-7": 1} | {"-11": 12, "-12": 4, "-13": 12, "-2": 25, "-3": 1, "-4": 330, "-6": 328, "-7": 6, "-8": 5, "-9": 24, "12": 12, "2": 33, "3": 8, "4": 338, "6": 355, "7": 13, "8": 8} | {"000/000": 757} | {"10": 682, "13": 4, "15": 32, "19": 7, "3": 1, "4": 1, "5": 29, "9": 1} | {"0-5": 646, "6-10": 30, ">10": 81} | {"at least one": 0, "both": 0, "neither": 757} |
| aligned_dense_s0 | selection | main, changed marker | native | {"3-1": 1, "4-2": 1, "4-3": 1, "6-4": 2} | {"-1": 1, "-4": 1, "-6": 1, "-9": 1, "0": 3, "1": 1, "10": 1, "5": 1} | {"000/010": 2, "001/000": 1, "010/000": 1, "100/000": 1} | {"10": 1, "5": 3, "9": 1} | {"none": 5} | {"at least one": 4, "both": 0, "neither": 1} |
| aligned_dense_s0 | selection | main, changed marker | pairs | {"3-1": 1, "4-1": 2, "4-2": 7, "5-5": 1, "6-4": 19, "6-5": 1, "6-6": 1, "7-5": 2, "7-6": 2, "7-7": 1} | {"-1": 6, "-10": 4, "-13": 1, "-3": 1, "-4": 3, "-5": 2, "-6": 1, "0": 23, "1": 8, "10": 8, "3": 1, "4": 4, "5": 8, "6": 4} | {"000/001": 5, "000/010": 16, "000/100": 4, "001/000": 1, "010/000": 7, "100/000": 4} | {"10": 12, "13": 1, "4": 2, "5": 22} | {"0-5": 28, "6-10": 3, ">10": 1, "none": 5} | {"at least one": 29, "both": 0, "neither": 8} |
| aligned_dense_s0 | selection | main, changed marker | multi | {"3-1": 2, "4-2": 6, "6-4": 14} | {"-1": 2, "-10": 2, "-4": 2, "-5": 2, "-6": 2, "0": 18, "1": 2, "10": 7, "5": 7} | {"000/010": 14, "001/000": 2, "010/000": 4, "100/000": 2} | {"10": 9, "5": 13} | {"0-5": 17, "6-10": 1, ">10": 4} | {"at least one": 20, "both": 0, "neither": 2} |
| aligned_dense_s0 | selection | broad | native | {"2-0": 7, "2-2": 2, "3-1": 2, "3-3": 2, "4-2": 2, "4-3": 1, "5-0": 29, "5-5": 1, "6-0": 2, "6-1": 31, "6-3": 2, "6-4": 2, "7-0": 12, "7-2": 33, "7-3": 1, "7-4": 2, "7-5": 2} | {"-1": 1, "-11": 2, "-14": 2, "-15": 1, "-3": 5, "-4": 37, "-5": 1, "-6": 33, "-7": 1, "-8": 44, "-9": 9, "0": 3, "1": 1, "10": 1, "12": 5, "2": 37, "4": 36, "5": 1, "6": 41, "7": 5} | {"000/000": 128, "000/010": 2, "001/000": 1, "010/000": 1, "100/000": 1} | {"10": 101, "13": 2, "15": 10, "19": 1, "20": 6, "4": 1, "5": 11, "9": 1} | {"none": 133} | {"at least one": 4, "both": 0, "neither": 129} |
| aligned_dense_s0 | selection | broad | pairs | {"0-0": 3, "1-0": 8, "1-1": 2, "2-0": 7, "2-1": 2, "2-2": 1, "3-0": 4, "3-1": 6, "3-2": 2, "3-3": 1, "4-0": 12, "4-1": 2, "4-2": 9, "5-0": 49, "5-1": 10, "5-3": 26, "5-5": 2, "6-0": 10, "6-1": 44, "6-2": 12, "6-4": 22, "6-5": 2, "6-6": 1, "7-0": 53, "7-1": 10, "7-2": 45, "7-3": 20, "7-5": 33, "7-6": 3, "7-7": 1} | {"-1": 6, "-10": 14, "-11": 18, "-12": 10, "-13": 21, "-14": 13, "-15": 16, "-2": 27, "-3": 13, "-4": 59, "-5": 6, "-6": 53, "-7": 5, "-8": 101, "-9": 26, "0": 29, "1": 11, "10": 14, "12": 46, "2": 90, "3": 12, "4": 63, "5": 14, "6": 77, "7": 46, "8": 14} | {"000/000": 356, "000/001": 5, "000/010": 16, "000/100": 4, "001/000": 1, "010/000": 13, "100/000": 7} | {"10": 194, "13": 3, "15": 74, "18": 3, "19": 6, "20": 39, "23": 3, "25": 4, "4": 2, "5": 66, "9": 8} | {"0-5": 339, "6-10": 9, ">10": 12, "none": 42} | {"at least one": 35, "both": 0, "neither": 367} |
| aligned_dense_s0 | selection | broad | multi | {"0-0": 4, "1-0": 8, "2-0": 18, "2-1": 2, "2-2": 4, "3-0": 2, "3-1": 12, "3-2": 1, "4-0": 6, "4-2": 8, "5-0": 390, "5-1": 8, "5-2": 1, "5-3": 16, "6-0": 20, "6-1": 323, "6-2": 7, "6-3": 6, "6-4": 14, "7-0": 84, "7-1": 14, "7-2": 325, "7-3": 14, "7-5": 23, "7-7": 1} | {"-1": 2, "-10": 5, "-11": 28, "-12": 5, "-13": 17, "-14": 16, "-15": 20, "-2": 28, "-3": 4, "-4": 343, "-5": 2, "-6": 338, "-7": 7, "-8": 464, "-9": 38, "0": 24, "1": 2, "10": 7, "12": 45, "2": 427, "3": 8, "4": 358, "5": 7, "6": 372, "7": 47, "8": 8} | {"000/000": 1283, "000/010": 14, "001/000": 2, "010/000": 10, "100/000": 2} | {"10": 1090, "13": 5, "15": 92, "19": 17, "20": 42, "3": 1, "4": 3, "5": 56, "9": 5} | {"0-5": 1093, "6-10": 70, ">10": 148} | {"at least one": 26, "both": 0, "neither": 1285} |
| aligned_dense_s1 | selection | main | native | {"1-1": 4, "3-1": 4, "3-3": 2, "4-2": 4, "4-4": 2, "5-1": 1, "5-3": 2, "6-1": 31, "6-3": 1, "6-4": 4, "7-1": 2, "7-2": 28, "7-3": 2, "7-4": 2, "7-5": 6} | {"-1": 2, "-11": 2, "-13": 3, "-2": 4, "-4": 34, "-5": 3, "-6": 37, "-7": 3, "-8": 2, "-9": 2, "0": 7, "1": 2, "10": 2, "12": 4, "2": 7, "3": 1, "4": 34, "5": 2, "6": 34, "7": 4, "8": 1} | {"000/000": 84, "000/010": 4, "001/000": 2, "010/000": 3, "100/000": 2} | {"10": 72, "13": 1, "15": 2, "20": 1, "25": 1, "3": 1, "5": 17} | {"none": 95} | {"at least one": 9, "both": 0, "neither": 86} |
| aligned_dense_s1 | selection | main | pairs | {"1-1": 3, "2-1": 1, "2-2": 2, "3-1": 13, "3-2": 2, "3-3": 1, "4-1": 2, "4-2": 16, "4-4": 1, "5-1": 17, "5-3": 31, "5-4": 2, "5-5": 1, "6-1": 44, "6-2": 14, "6-4": 19, "6-6": 3, "7-1": 9, "7-2": 43, "7-3": 22, "7-5": 36, "7-6": 1, "7-7": 4} | {"-1": 9, "-10": 13, "-11": 12, "-12": 9, "-13": 14, "-2": 23, "-3": 18, "-4": 53, "-5": 8, "-6": 50, "-7": 10, "-8": 26, "-9": 12, "0": 27, "1": 10, "10": 12, "12": 24, "2": 55, "3": 12, "4": 59, "5": 15, "6": 65, "7": 27, "8": 11} | {"000/000": 241, "000/001": 5, "000/010": 15, "000/100": 5, "001/000": 4, "010/000": 12, "100/000": 5} | {"10": 151, "13": 1, "15": 40, "18": 2, "19": 2, "20": 10, "23": 2, "25": 1, "4": 3, "5": 73, "9": 2} | {"0-5": 245, "6-10": 2, ">10": 7, "none": 33} | {"at least one": 36, "both": 0, "neither": 251} |
| aligned_dense_s1 | selection | main | multi | {"1-1": 1, "2-1": 6, "3-1": 9, "3-2": 2, "3-3": 2, "4-2": 10, "5-1": 6, "5-2": 2, "5-3": 16, "6-1": 326, "6-2": 8, "6-3": 6, "6-4": 15, "6-6": 4, "7-1": 20, "7-2": 362, "7-3": 13, "7-4": 2, "7-5": 18, "7-7": 2} | {"-1": 2, "-10": 1, "-11": 10, "-12": 4, "-13": 16, "-2": 26, "-4": 382, "-5": 1, "-6": 332, "-7": 6, "-8": 4, "-9": 26, "0": 17, "1": 4, "10": 11, "12": 12, "2": 26, "3": 8, "4": 343, "5": 10, "6": 401, "7": 10, "8": 8} | {"000/000": 807, "000/001": 2, "000/010": 15, "010/000": 2, "100/000": 4} | {"10": 737, "13": 5, "15": 29, "19": 10, "5": 45, "9": 4} | {"0-5": 706, "6-10": 34, ">10": 90} | {"at least one": 19, "both": 0, "neither": 811} |
| aligned_dense_s1 | selection | main, equal marker | native | {"1-1": 2, "3-1": 2, "3-3": 2, "4-2": 2, "4-4": 1, "5-1": 1, "5-3": 2, "6-1": 31, "6-3": 1, "7-1": 2, "7-2": 28, "7-3": 2, "7-4": 2, "7-5": 6} | {"-11": 2, "-13": 3, "-2": 4, "-4": 32, "-6": 35, "-7": 3, "-8": 2, "-9": 2, "12": 4, "2": 7, "3": 1, "4": 34, "6": 34, "7": 4, "8": 1} | {"000/000": 84} | {"10": 70, "13": 1, "15": 2, "20": 1, "25": 1, "3": 1, "5": 8} | {"none": 84} | {"at least one": 0, "both": 0, "neither": 84} |
| aligned_dense_s1 | selection | main, equal marker | pairs | {"1-1": 3, "2-1": 1, "2-2": 2, "3-1": 10, "3-2": 2, "3-3": 1, "4-2": 3, "5-1": 17, "5-3": 31, "5-4": 2, "6-1": 44, "6-2": 14, "6-4": 2, "6-6": 2, "7-1": 9, "7-2": 43, "7-3": 22, "7-5": 32, "7-7": 1} | {"-10": 6, "-11": 12, "-12": 9, "-13": 13, "-2": 23, "-3": 17, "-4": 49, "-5": 4, "-6": 46, "-7": 10, "-8": 26, "-9": 12, "10": 6, "12": 24, "2": 55, "3": 11, "4": 55, "5": 6, "6": 60, "7": 27, "8": 11} | {"000/000": 241} | {"10": 138, "15": 40, "18": 2, "19": 2, "20": 10, "23": 2, "25": 1, "4": 1, "5": 43, "9": 2} | {"0-5": 211, "6-10": 2, ">10": 6, "none": 22} | {"at least one": 0, "both": 0, "neither": 241} |
| aligned_dense_s1 | selection | main, equal marker | multi | {"1-1": 1, "2-1": 6, "3-1": 9, "3-2": 2, "3-3": 1, "4-2": 4, "5-1": 6, "5-2": 2, "5-3": 16, "6-1": 326, "6-2": 8, "6-3": 6, "6-6": 3, "7-1": 20, "7-2": 362, "7-3": 13, "7-4": 2, "7-5": 18, "7-7": 2} | {"-11": 10, "-12": 4, "-13": 16, "-2": 26, "-4": 378, "-6": 332, "-7": 6, "-8": 4, "-9": 26, "10": 3, "12": 12, "2": 26, "3": 8, "4": 341, "5": 3, "6": 401, "7": 10, "8": 8} | {"000/000": 807} | {"10": 728, "13": 5, "15": 29, "19": 10, "5": 31, "9": 4} | {"0-5": 691, "6-10": 34, ">10": 82} | {"at least one": 0, "both": 0, "neither": 807} |
| aligned_dense_s1 | selection | main, changed marker | native | {"1-1": 2, "3-1": 2, "4-2": 2, "4-4": 1, "6-4": 4} | {"-1": 2, "-4": 2, "-5": 3, "-6": 2, "0": 7, "1": 2, "10": 2, "5": 2} | {"000/010": 4, "001/000": 2, "010/000": 3, "100/000": 2} | {"10": 2, "5": 9} | {"none": 11} | {"at least one": 9, "both": 0, "neither": 2} |
| aligned_dense_s1 | selection | main, changed marker | pairs | {"3-1": 3, "4-1": 2, "4-2": 13, "4-4": 1, "5-5": 1, "6-4": 17, "6-6": 1, "7-5": 4, "7-6": 1, "7-7": 3} | {"-1": 9, "-10": 7, "-13": 1, "-3": 1, "-4": 4, "-5": 4, "-6": 4, "0": 27, "1": 10, "10": 6, "3": 1, "4": 4, "5": 9, "6": 5} | {"000/001": 5, "000/010": 15, "000/100": 5, "001/000": 4, "010/000": 12, "100/000": 5} | {"10": 13, "13": 1, "4": 2, "5": 30} | {"0-5": 34, ">10": 1, "none": 11} | {"at least one": 36, "both": 0, "neither": 10} |
| aligned_dense_s1 | selection | main, changed marker | multi | {"3-3": 1, "4-2": 6, "6-4": 15, "6-6": 1} | {"-1": 2, "-10": 1, "-4": 4, "-5": 1, "0": 17, "1": 4, "10": 8, "4": 2, "5": 7} | {"000/001": 2, "000/010": 15, "010/000": 2, "100/000": 4} | {"10": 9, "5": 14} | {"0-5": 15, ">10": 8} | {"at least one": 19, "both": 0, "neither": 4} |
| aligned_dense_s1 | selection | broad | native | {"1-1": 4, "2-0": 12, "3-1": 4, "3-3": 2, "4-0": 2, "4-2": 4, "4-4": 2, "5-0": 34, "5-1": 1, "5-3": 2, "6-1": 31, "6-3": 1, "6-4": 4, "7-0": 18, "7-1": 2, "7-2": 28, "7-3": 2, "7-4": 2, "7-5": 6} | {"-1": 2, "-10": 1, "-11": 2, "-13": 3, "-14": 3, "-15": 1, "-2": 4, "-3": 5, "-4": 36, "-5": 3, "-6": 37, "-7": 3, "-8": 60, "-9": 5, "0": 9, "1": 2, "10": 2, "12": 11, "2": 46, "3": 1, "4": 34, "5": 2, "6": 38, "7": 11, "8": 1} | {"000/000": 148, "000/010": 4, "001/000": 2, "010/000": 5, "100/000": 2} | {"10": 113, "13": 1, "15": 12, "20": 10, "25": 1, "3": 1, "5": 23} | {"none": 161} | {"at least one": 11, "both": 0, "neither": 150} |
| aligned_dense_s1 | selection | broad | pairs | {"0-0": 3, "1-0": 2, "1-1": 3, "2-0": 11, "2-1": 1, "2-2": 2, "3-0": 6, "3-1": 13, "3-2": 2, "3-3": 1, "4-0": 16, "4-1": 2, "4-2": 16, "4-4": 1, "5-0": 52, "5-1": 17, "5-3": 31, "5-4": 2, "5-5": 1, "6-0": 8, "6-1": 44, "6-2": 14, "6-4": 19, "6-6": 3, "7-0": 61, "7-1": 9, "7-2": 43, "7-3": 22, "7-5": 36, "7-6": 1, "7-7": 4} | {"-1": 9, "-10": 19, "-11": 18, "-12": 14, "-13": 22, "-14": 13, "-15": 15, "-2": 29, "-3": 20, "-4": 60, "-5": 8, "-6": 52, "-7": 10, "-8": 120, "-9": 27, "0": 35, "1": 14, "10": 14, "12": 49, "2": 105, "3": 14, "4": 63, "5": 17, "6": 80, "7": 52, "8": 13} | {"000/000": 388, "000/001": 5, "000/010": 15, "000/100": 5, "001/000": 4, "010/000": 20, "100/000": 9} | {"10": 209, "13": 4, "15": 85, "18": 3, "19": 4, "20": 45, "23": 3, "25": 4, "4": 5, "5": 82, "9": 2} | {"0-5": 371, "6-10": 9, ">10": 16, "none": 50} | {"at least one": 44, "both": 0, "neither": 402} |
| aligned_dense_s1 | selection | broad | multi | {"1-0": 8, "1-1": 1, "2-0": 28, "2-1": 6, "3-0": 2, "3-1": 9, "3-2": 2, "3-3": 2, "4-0": 8, "4-2": 10, "5-0": 401, "5-1": 6, "5-2": 2, "5-3": 16, "6-0": 14, "6-1": 326, "6-2": 8, "6-3": 6, "6-4": 15, "6-6": 4, "7-0": 82, "7-1": 20, "7-2": 362, "7-3": 13, "7-4": 2, "7-5": 18, "7-7": 2} | {"-1": 2, "-10": 5, "-11": 21, "-12": 5, "-13": 22, "-14": 16, "-15": 16, "-2": 28, "-3": 7, "-4": 396, "-5": 1, "-6": 340, "-7": 6, "-8": 477, "-9": 42, "0": 25, "1": 4, "10": 11, "12": 45, "2": 434, "3": 8, "4": 357, "5": 10, "6": 419, "7": 41, "8": 8} | {"000/000": 1342, "000/001": 2, "000/010": 15, "010/000": 10, "100/000": 4} | {"10": 1151, "13": 6, "15": 86, "19": 17, "20": 42, "5": 63, "9": 8} | {"0-5": 1146, "6-10": 75, ">10": 152} | {"at least one": 27, "both": 0, "neither": 1346} |
| aligned_dense_s2 | selection | main | native | {"3-1": 11, "3-3": 2, "4-2": 7, "4-4": 2, "5-1": 2, "5-3": 5, "5-5": 1, "6-1": 34, "6-3": 2, "6-4": 1, "6-6": 2, "7-1": 4, "7-2": 38, "7-4": 2, "7-5": 8, "7-7": 2} | {"-1": 8, "-12": 1, "-13": 2, "-2": 6, "-4": 46, "-6": 47, "-7": 2, "-8": 3, "-9": 1, "0": 2, "1": 4, "10": 2, "12": 6, "2": 9, "3": 8, "4": 43, "5": 2, "6": 45, "7": 6, "8": 3} | {"000/000": 109, "000/001": 2, "000/010": 2, "000/100": 1, "001/000": 6, "100/000": 3} | {"10": 91, "13": 2, "15": 2, "20": 2, "25": 1, "4": 1, "5": 24} | {"none": 123} | {"at least one": 10, "both": 0, "neither": 113} |
| aligned_dense_s2 | selection | main | pairs | {"1-1": 1, "2-1": 1, "3-1": 8, "3-2": 2, "3-3": 1, "4-1": 2, "4-2": 12, "4-3": 2, "4-4": 1, "5-1": 8, "5-3": 24, "5-4": 2, "5-5": 2, "6-1": 45, "6-2": 10, "6-4": 16, "6-6": 5, "7-1": 10, "7-2": 49, "7-3": 16, "7-4": 4, "7-5": 34, "7-7": 2} | {"-1": 6, "-10": 7, "-11": 12, "-12": 2, "-13": 13, "-2": 20, "-3": 15, "-4": 53, "-5": 6, "-6": 49, "-7": 6, "-8": 18, "-9": 14, "0": 24, "1": 5, "10": 12, "12": 26, "2": 50, "3": 8, "4": 58, "5": 9, "6": 67, "7": 26, "8": 8} | {"000/000": 222, "000/001": 3, "000/010": 11, "000/100": 3, "001/000": 3, "010/000": 13, "100/000": 2} | {"10": 143, "13": 1, "14": 1, "15": 32, "18": 2, "19": 4, "20": 6, "23": 2, "25": 1, "4": 1, "5": 61, "9": 3} | {"0-5": 201, "6-10": 6, ">10": 8, "none": 42} | {"at least one": 30, "both": 0, "neither": 227} |
| aligned_dense_s2 | selection | main | multi | {"1-1": 1, "2-1": 2, "2-2": 1, "3-1": 18, "3-2": 1, "3-3": 2, "4-1": 2, "4-2": 8, "4-4": 1, "5-1": 6, "5-2": 2, "5-3": 14, "5-5": 1, "6-1": 287, "6-2": 8, "6-3": 4, "6-4": 9, "6-6": 1, "7-1": 14, "7-2": 312, "7-3": 12, "7-5": 22, "7-7": 1} | {"-1": 7, "-10": 3, "-11": 8, "-12": 5, "-13": 12, "-2": 25, "-3": 1, "-4": 320, "-5": 4, "-6": 299, "-7": 7, "-8": 4, "-9": 20, "0": 17, "1": 3, "10": 5, "12": 11, "2": 31, "3": 7, "4": 304, "5": 6, "6": 341, "7": 11, "8": 7} | {"000/000": 702, "000/001": 2, "000/010": 9, "000/100": 1, "001/000": 5, "010/000": 8, "100/000": 2} | {"10": 644, "13": 4, "15": 26, "19": 7, "4": 1, "5": 46, "9": 1} | {"0-5": 595, "6-10": 33, ">10": 101} | {"at least one": 24, "both": 0, "neither": 705} |
| aligned_dense_s2 | selection | main, equal marker | native | {"3-1": 5, "3-3": 1, "4-2": 4, "4-4": 1, "5-1": 2, "5-3": 5, "5-5": 1, "6-1": 34, "6-3": 2, "6-6": 2, "7-1": 4, "7-2": 38, "7-4": 2, "7-5": 8} | {"-12": 1, "-13": 2, "-2": 6, "-4": 43, "-6": 41, "-7": 2, "-8": 3, "-9": 1, "10": 1, "12": 6, "2": 9, "3": 7, "4": 42, "5": 1, "6": 44, "7": 6, "8": 3} | {"000/000": 109} | {"10": 90, "13": 2, "15": 2, "20": 2, "25": 1, "5": 12} | {"none": 109} | {"at least one": 0, "both": 0, "neither": 109} |
| aligned_dense_s2 | selection | main, equal marker | pairs | {"1-1": 1, "2-1": 1, "3-1": 6, "3-2": 2, "3-3": 1, "4-2": 2, "5-1": 8, "5-3": 24, "5-4": 2, "5-5": 1, "6-1": 45, "6-2": 10, "6-4": 2, "6-6": 5, "7-1": 10, "7-2": 49, "7-3": 16, "7-4": 4, "7-5": 31, "7-7": 2} | {"-10": 2, "-11": 12, "-12": 2, "-13": 12, "-2": 20, "-3": 15, "-4": 51, "-6": 46, "-7": 6, "-8": 18, "-9": 13, "10": 5, "12": 26, "2": 50, "3": 8, "4": 55, "5": 5, "6": 64, "7": 26, "8": 8} | {"000/000": 222} | {"10": 131, "14": 1, "15": 32, "18": 2, "19": 4, "20": 6, "23": 2, "25": 1, "4": 1, "5": 40, "9": 2} | {"0-5": 170, "6-10": 6, ">10": 8, "none": 38} | {"at least one": 0, "both": 0, "neither": 222} |
| aligned_dense_s2 | selection | main, equal marker | multi | {"2-1": 2, "2-2": 1, "3-1": 13, "3-2": 1, "4-2": 2, "5-1": 6, "5-2": 2, "5-3": 14, "5-5": 1, "6-1": 287, "6-2": 8, "6-3": 4, "6-6": 1, "7-1": 14, "7-2": 312, "7-3": 12, "7-5": 22} | {"-11": 8, "-12": 5, "-13": 11, "-2": 25, "-3": 1, "-4": 318, "-6": 294, "-7": 7, "-8": 4, "-9": 20, "10": 1, "12": 11, "2": 31, "3": 7, "4": 302, "5": 1, "6": 340, "7": 11, "8": 7} | {"000/000": 702} | {"10": 637, "13": 3, "15": 26, "19": 7, "4": 1, "5": 27, "9": 1} | {"0-5": 582, "6-10": 31, ">10": 89} | {"at least one": 0, "both": 0, "neither": 702} |
| aligned_dense_s2 | selection | main, changed marker | native | {"3-1": 6, "3-3": 1, "4-2": 3, "4-4": 1, "6-4": 1, "7-7": 2} | {"-1": 8, "-4": 3, "-6": 6, "0": 2, "1": 4, "10": 1, "3": 1, "4": 1, "5": 1, "6": 1} | {"000/001": 2, "000/010": 2, "000/100": 1, "001/000": 6, "100/000": 3} | {"10": 1, "4": 1, "5": 12} | {"none": 14} | {"at least one": 10, "both": 0, "neither": 4} |
| aligned_dense_s2 | selection | main, changed marker | pairs | {"3-1": 2, "4-1": 2, "4-2": 10, "4-3": 2, "4-4": 1, "5-5": 1, "6-4": 14, "7-5": 3} | {"-1": 6, "-10": 5, "-13": 1, "-4": 2, "-5": 6, "-6": 3, "-9": 1, "0": 24, "1": 5, "10": 7, "4": 3, "5": 4, "6": 3} | {"000/001": 3, "000/010": 11, "000/100": 3, "001/000": 3, "010/000": 13, "100/000": 2} | {"10": 12, "13": 1, "5": 21, "9": 1} | {"0-5": 31, "none": 4} | {"at least one": 30, "both": 0, "neither": 5} |
| aligned_dense_s2 | selection | main, changed marker | multi | {"1-1": 1, "3-1": 5, "3-3": 2, "4-1": 2, "4-2": 6, "4-4": 1, "6-4": 9, "7-7": 1} | {"-1": 7, "-10": 3, "-13": 1, "-4": 2, "-5": 4, "-6": 5, "0": 17, "1": 3, "10": 4, "4": 2, "5": 5, "6": 1} | {"000/001": 2, "000/010": 9, "000/100": 1, "001/000": 5, "010/000": 8, "100/000": 2} | {"10": 7, "13": 1, "5": 19} | {"0-5": 13, "6-10": 2, ">10": 12} | {"at least one": 24, "both": 0, "neither": 3} |
| aligned_dense_s2 | selection | broad | native | {"0-0": 2, "2-0": 8, "3-1": 11, "3-3": 2, "4-0": 5, "4-2": 7, "4-4": 2, "5-0": 32, "5-1": 2, "5-3": 5, "5-5": 1, "6-0": 2, "6-1": 34, "6-3": 2, "6-4": 1, "6-6": 2, "7-0": 16, "7-1": 4, "7-2": 38, "7-4": 2, "7-5": 8, "7-7": 2} | {"-1": 8, "-10": 2, "-12": 1, "-13": 2, "-14": 6, "-15": 3, "-2": 7, "-3": 2, "-4": 50, "-6": 47, "-7": 3, "-8": 51, "-9": 7, "0": 3, "1": 6, "10": 3, "12": 12, "2": 43, "3": 8, "4": 43, "5": 3, "6": 51, "7": 12, "8": 3} | {"000/000": 171, "000/001": 2, "000/010": 2, "000/100": 1, "001/000": 6, "010/000": 1, "100/000": 5} | {"10": 128, "13": 2, "15": 13, "20": 13, "25": 1, "4": 1, "5": 30} | {"none": 188} | {"at least one": 11, "both": 0, "neither": 177} |
| aligned_dense_s2 | selection | broad | pairs | {"0-0": 5, "1-1": 1, "2-0": 6, "2-1": 1, "3-0": 5, "3-1": 8, "3-2": 2, "3-3": 1, "4-0": 15, "4-1": 2, "4-2": 12, "4-3": 2, "4-4": 1, "5-0": 51, "5-1": 8, "5-3": 24, "5-4": 2, "5-5": 2, "6-0": 8, "6-1": 45, "6-2": 10, "6-4": 16, "6-6": 5, "7-0": 50, "7-1": 10, "7-2": 49, "7-3": 16, "7-4": 4, "7-5": 34, "7-7": 2} | {"-1": 6, "-10": 15, "-11": 14, "-12": 8, "-13": 19, "-14": 9, "-15": 16, "-2": 25, "-3": 16, "-4": 57, "-5": 6, "-6": 49, "-7": 6, "-8": 102, "-9": 28, "0": 33, "1": 8, "10": 14, "12": 46, "2": 98, "3": 10, "4": 62, "5": 11, "6": 79, "7": 47, "8": 10} | {"000/000": 350, "000/001": 3, "000/010": 11, "000/100": 3, "001/000": 3, "010/000": 22, "100/000": 5} | {"10": 197, "13": 3, "14": 1, "15": 73, "18": 3, "19": 6, "20": 34, "23": 3, "25": 3, "3": 1, "4": 1, "5": 69, "9": 3} | {"0-5": 317, "6-10": 8, ">10": 16, "none": 56} | {"at least one": 39, "both": 0, "neither": 358} |
| aligned_dense_s2 | selection | broad | multi | {"0-0": 4, "1-0": 5, "1-1": 1, "2-0": 15, "2-1": 2, "2-2": 1, "3-0": 2, "3-1": 18, "3-2": 1, "3-3": 2, "4-0": 6, "4-1": 2, "4-2": 8, "4-4": 1, "5-0": 344, "5-1": 6, "5-2": 2, "5-3": 14, "5-5": 1, "6-0": 17, "6-1": 287, "6-2": 8, "6-3": 4, "6-4": 9, "6-6": 1, "7-0": 83, "7-1": 14, "7-2": 312, "7-3": 12, "7-5": 22, "7-7": 1} | {"-1": 7, "-10": 6, "-11": 19, "-12": 7, "-13": 16, "-14": 16, "-15": 18, "-2": 27, "-3": 3, "-4": 331, "-5": 4, "-6": 304, "-7": 7, "-8": 416, "-9": 34, "0": 23, "1": 3, "10": 5, "12": 44, "2": 377, "3": 7, "4": 321, "5": 6, "6": 358, "7": 44, "8": 7} | {"000/000": 1172, "000/001": 2, "000/010": 9, "000/100": 1, "001/000": 5, "010/000": 14, "100/000": 2} | {"10": 997, "13": 5, "15": 81, "19": 16, "20": 42, "3": 1, "4": 2, "5": 57, "9": 4} | {"0-5": 983, "6-10": 60, ">10": 162} | {"at least one": 30, "both": 0, "neither": 1175} |
| center_only_s0 | selection | main | native | {"1-1": 2, "2-2": 3, "3-1": 3, "3-3": 5, "4-1": 1, "4-2": 1, "4-4": 5, "5-1": 1, "5-3": 2, "5-5": 3, "6-1": 36, "6-2": 1, "6-3": 1, "6-4": 1, "6-6": 4, "7-2": 39, "7-3": 1, "7-5": 2, "7-7": 4} | {"-1": 6, "-10": 1, "-11": 1, "-13": 4, "-2": 4, "-3": 2, "-4": 43, "-5": 3, "-6": 43, "-7": 2, "-8": 1, "-9": 4, "0": 5, "1": 2, "10": 3, "12": 3, "2": 7, "3": 1, "4": 45, "5": 2, "6": 44, "7": 1, "8": 3} | {"000/000": 102, "000/001": 4, "000/010": 1, "000/100": 2, "001/000": 2, "010/000": 4} | {"10": 87, "13": 2, "15": 1, "3": 4, "5": 21} | {"none": 115} | {"at least one": 11, "both": 0, "neither": 104} |
| center_only_s0 | selection | main | pairs | {"2-2": 2, "3-1": 42, "4-1": 4, "4-2": 41, "4-3": 2, "5-1": 22, "5-2": 4, "5-3": 36, "6-1": 90, "6-2": 20, "6-3": 4, "6-4": 38, "7-1": 14, "7-2": 93, "7-3": 22, "7-4": 4, "7-5": 36, "7-7": 2} | {"-1": 21, "-10": 19, "-11": 14, "-12": 13, "-13": 16, "-2": 38, "-3": 12, "-4": 122, "-5": 13, "-6": 122, "-7": 15, "-8": 25, "-9": 18, "0": 46, "1": 19, "10": 20, "12": 25, "2": 52, "3": 19, "4": 126, "5": 21, "6": 133, "7": 25, "8": 18} | {"000/000": 390, "000/001": 7, "000/010": 27, "000/100": 5, "001/000": 14, "010/000": 19, "100/000": 14} | {"10": 291, "13": 8, "15": 51, "18": 2, "19": 3, "20": 15, "23": 2, "25": 2, "4": 1, "5": 100, "9": 1} | {"0-5": 409, "6-10": 40, ">10": 9, "none": 18} | {"at least one": 67, "both": 0, "neither": 409} |
| center_only_s0 | selection | main | multi | {"1-1": 4, "2-1": 5, "2-2": 3, "3-1": 42, "3-2": 4, "3-3": 4, "4-1": 6, "4-2": 41, "4-3": 6, "4-4": 3, "5-1": 12, "5-2": 6, "5-3": 40, "5-4": 4, "5-5": 4, "6-1": 779, "6-2": 12, "6-3": 6, "6-4": 40, "6-5": 2, "6-6": 4, "7-1": 27, "7-2": 775, "7-3": 12, "7-4": 6, "7-5": 45, "7-6": 6, "7-7": 3} | {"-1": 28, "-10": 7, "-11": 23, "-12": 4, "-13": 27, "-2": 43, "-3": 12, "-4": 820, "-5": 10, "-6": 819, "-7": 10, "-8": 14, "-9": 39, "0": 56, "1": 25, "10": 19, "12": 20, "2": 72, "3": 20, "4": 826, "5": 21, "6": 850, "7": 20, "8": 17} | {"000/000": 1792, "000/001": 8, "000/010": 36, "000/100": 6, "001/000": 20, "010/000": 20, "100/000": 19} | {"10": 1668, "13": 12, "15": 52, "19": 14, "4": 2, "5": 139, "9": 14} | {"0-5": 1543, "6-10": 190, ">10": 168} | {"at least one": 84, "both": 0, "neither": 1817} |
| center_only_s0 | selection | main, equal marker | native | {"1-1": 2, "2-2": 3, "3-1": 1, "3-3": 4, "4-2": 1, "4-4": 3, "5-1": 1, "5-3": 2, "5-5": 2, "6-1": 36, "6-2": 1, "6-3": 1, "6-6": 2, "7-2": 39, "7-3": 1, "7-5": 2, "7-7": 1} | {"-10": 1, "-11": 1, "-13": 3, "-2": 4, "-3": 1, "-4": 43, "-5": 1, "-6": 41, "-7": 2, "-8": 1, "-9": 4, "10": 2, "12": 2, "2": 6, "3": 1, "4": 43, "5": 1, "6": 43, "7": 1, "8": 3} | {"000/000": 102} | {"10": 85, "13": 1, "15": 1, "3": 2, "5": 13} | {"none": 102} | {"at least one": 0, "both": 0, "neither": 102} |
| center_only_s0 | selection | main, equal marker | pairs | {"2-2": 2, "3-1": 28, "4-2": 14, "5-1": 22, "5-2": 4, "5-3": 36, "6-1": 90, "6-2": 20, "6-3": 4, "6-4": 6, "7-1": 14, "7-2": 93, "7-3": 22, "7-4": 4, "7-5": 31} | {"-10": 10, "-11": 14, "-12": 13, "-13": 14, "-2": 38, "-3": 12, "-4": 108, "-5": 6, "-6": 108, "-7": 15, "-8": 25, "-9": 17, "10": 7, "12": 25, "2": 52, "3": 18, "4": 120, "5": 7, "6": 128, "7": 25, "8": 18} | {"000/000": 390} | {"10": 269, "13": 6, "15": 51, "18": 2, "19": 3, "20": 15, "23": 2, "25": 2, "5": 40} | {"0-5": 333, "6-10": 35, ">10": 9, "none": 13} | {"at least one": 0, "both": 0, "neither": 390} |
| center_only_s0 | selection | main, equal marker | multi | {"1-1": 3, "2-1": 5, "2-2": 3, "3-1": 23, "3-2": 4, "3-3": 1, "4-2": 18, "5-1": 12, "5-2": 6, "5-3": 40, "5-4": 4, "5-5": 3, "6-1": 779, "6-2": 12, "6-3": 6, "6-4": 3, "6-5": 1, "6-6": 3, "7-1": 27, "7-2": 775, "7-3": 12, "7-4": 6, "7-5": 43, "7-6": 3} | {"-10": 1, "-11": 23, "-12": 4, "-13": 24, "-2": 43, "-3": 12, "-4": 801, "-5": 2, "-6": 799, "-7": 10, "-8": 14, "-9": 36, "10": 2, "12": 20, "2": 72, "3": 19, "4": 819, "5": 2, "6": 844, "7": 20, "8": 17} | {"000/000": 1792} | {"10": 1645, "13": 9, "15": 52, "19": 14, "4": 1, "5": 60, "9": 11} | {"0-5": 1463, "6-10": 176, ">10": 153} | {"at least one": 0, "both": 0, "neither": 1792} |
| center_only_s0 | selection | main, changed marker | native | {"3-1": 2, "3-3": 1, "4-1": 1, "4-4": 2, "5-5": 1, "6-4": 1, "6-6": 2, "7-7": 3} | {"-1": 6, "-13": 1, "-3": 1, "-5": 2, "-6": 2, "0": 5, "1": 2, "10": 1, "12": 1, "2": 1, "4": 2, "5": 1, "6": 1} | {"000/001": 4, "000/010": 1, "000/100": 2, "001/000": 2, "010/000": 4} | {"10": 2, "13": 1, "3": 2, "5": 8} | {"none": 13} | {"at least one": 11, "both": 0, "neither": 2} |
| center_only_s0 | selection | main, changed marker | pairs | {"3-1": 14, "4-1": 4, "4-2": 27, "4-3": 2, "6-4": 32, "7-5": 5, "7-7": 2} | {"-1": 21, "-10": 9, "-13": 2, "-4": 14, "-5": 7, "-6": 14, "-9": 1, "0": 46, "1": 19, "10": 13, "3": 1, "4": 6, "5": 14, "6": 5} | {"000/001": 7, "000/010": 27, "000/100": 5, "001/000": 14, "010/000": 19, "100/000": 14} | {"10": 22, "13": 2, "4": 1, "5": 60, "9": 1} | {"0-5": 76, "6-10": 5, "none": 5} | {"at least one": 67, "both": 0, "neither": 19} |
| center_only_s0 | selection | main, changed marker | multi | {"1-1": 1, "3-1": 19, "3-3": 3, "4-1": 6, "4-2": 23, "4-3": 6, "4-4": 3, "5-5": 1, "6-4": 37, "6-5": 1, "6-6": 1, "7-5": 2, "7-6": 3, "7-7": 3} | {"-1": 28, "-10": 6, "-13": 3, "-4": 19, "-5": 8, "-6": 20, "-9": 3, "0": 56, "1": 25, "10": 17, "3": 1, "4": 7, "5": 19, "6": 6} | {"000/001": 8, "000/010": 36, "000/100": 6, "001/000": 20, "010/000": 20, "100/000": 19} | {"10": 23, "13": 3, "4": 1, "5": 79, "9": 3} | {"0-5": 80, "6-10": 14, ">10": 15} | {"at least one": 84, "both": 0, "neither": 25} |
| center_only_s0 | selection | broad | native | {"0-0": 4, "1-1": 2, "2-0": 4, "2-2": 3, "3-0": 1, "3-1": 3, "3-3": 5, "4-0": 1, "4-1": 1, "4-2": 1, "4-4": 5, "5-0": 38, "5-1": 1, "5-3": 2, "5-5": 3, "6-1": 36, "6-2": 1, "6-3": 1, "6-4": 1, "6-6": 4, "7-0": 10, "7-2": 39, "7-3": 1, "7-5": 2, "7-7": 4} | {"-1": 6, "-10": 2, "-11": 1, "-12": 1, "-13": 4, "-14": 1, "-15": 4, "-2": 7, "-3": 4, "-4": 43, "-5": 3, "-6": 43, "-7": 4, "-8": 51, "-9": 5, "0": 6, "1": 2, "10": 3, "12": 7, "2": 47, "3": 1, "4": 45, "5": 2, "6": 46, "7": 5, "8": 3} | {"000/000": 159, "000/001": 4, "000/010": 1, "000/100": 2, "001/000": 2, "010/000": 5} | {"10": 127, "13": 3, "15": 7, "20": 5, "3": 5, "5": 26} | {"none": 173} | {"at least one": 12, "both": 0, "neither": 161} |
| center_only_s0 | selection | broad | pairs | {"0-0": 2, "2-0": 40, "2-2": 2, "3-0": 4, "3-1": 42, "4-0": 18, "4-1": 4, "4-2": 41, "4-3": 2, "5-0": 96, "5-1": 22, "5-2": 4, "5-3": 36, "6-0": 14, "6-1": 90, "6-2": 20, "6-3": 4, "6-4": 38, "7-0": 102, "7-1": 14, "7-2": 93, "7-3": 22, "7-4": 4, "7-5": 36, "7-7": 2} | {"-1": 21, "-10": 28, "-11": 17, "-12": 17, "-13": 31, "-14": 21, "-15": 16, "-2": 42, "-3": 27, "-4": 132, "-5": 13, "-6": 122, "-7": 15, "-8": 214, "-9": 39, "0": 56, "1": 23, "10": 24, "12": 65, "2": 159, "3": 21, "4": 132, "5": 25, "6": 159, "7": 65, "8": 20} | {"000/000": 652, "000/001": 7, "000/010": 27, "000/100": 5, "001/000": 14, "010/000": 29, "100/000": 18} | {"10": 406, "13": 10, "15": 119, "18": 3, "19": 6, "20": 73, "23": 3, "25": 8, "4": 1, "5": 122, "9": 1} | {"0-5": 603, "6-10": 93, ">10": 25, "none": 31} | {"at least one": 77, "both": 0, "neither": 675} |
| center_only_s0 | selection | broad | multi | {"0-0": 4, "1-0": 4, "1-1": 4, "2-0": 48, "2-1": 5, "2-2": 3, "3-0": 6, "3-1": 42, "3-2": 4, "3-3": 4, "4-0": 12, "4-1": 6, "4-2": 41, "4-3": 6, "4-4": 3, "5-0": 793, "5-1": 12, "5-2": 6, "5-3": 40, "5-4": 4, "5-5": 4, "6-0": 28, "6-1": 779, "6-2": 12, "6-3": 6, "6-4": 40, "6-5": 2, "6-6": 4, "7-0": 132, "7-1": 27, "7-2": 775, "7-3": 12, "7-4": 6, "7-5": 45, "7-6": 6, "7-7": 3} | {"-1": 28, "-10": 13, "-11": 39, "-12": 7, "-13": 39, "-14": 15, "-15": 25, "-2": 51, "-3": 33, "-4": 825, "-5": 10, "-6": 823, "-7": 12, "-8": 950, "-9": 53, "0": 68, "1": 25, "10": 19, "12": 75, "2": 887, "3": 20, "4": 854, "5": 21, "6": 872, "7": 75, "8": 17} | {"000/000": 2807, "000/001": 8, "000/010": 36, "000/100": 6, "001/000": 20, "010/000": 32, "100/000": 19} | {"10": 2484, "13": 15, "15": 149, "19": 28, "20": 66, "4": 2, "5": 168, "9": 16} | {"0-5": 2277, "6-10": 386, ">10": 265} | {"at least one": 96, "both": 0, "neither": 2832} |
| center_only_s1 | selection | main | native | {"1-1": 16, "2-1": 1, "2-2": 14, "3-1": 20, "3-2": 2, "3-3": 17, "4-1": 5, "4-2": 22, "4-4": 12, "5-1": 10, "5-2": 5, "5-3": 22, "5-5": 14, "6-1": 241, "6-2": 11, "6-3": 6, "6-4": 31, "6-5": 3, "6-6": 21, "7-1": 2, "7-2": 192, "7-3": 4, "7-4": 1, "7-5": 27, "7-6": 4, "7-7": 6} | {"-1": 24, "-10": 11, "-11": 20, "-12": 10, "-13": 10, "-2": 31, "-3": 25, "-4": 216, "-5": 21, "-6": 266, "-7": 15, "-8": 16, "-9": 12, "0": 64, "1": 20, "10": 24, "12": 11, "2": 49, "3": 14, "4": 285, "5": 24, "6": 225, "7": 11, "8": 14} | {"000/000": 601, "000/001": 16, "000/010": 31, "000/100": 11, "001/000": 8, "010/000": 33, "100/000": 9} | {"10": 507, "13": 13, "15": 22, "19": 1, "20": 2, "3": 11, "4": 2, "5": 146, "9": 5} | {"none": 709} | {"at least one": 88, "both": 0, "neither": 621} |
| center_only_s1 | selection | main | pairs | {"1-1": 7, "2-1": 2, "2-2": 7, "3-1": 46, "3-2": 2, "3-3": 8, "4-1": 8, "4-2": 43, "4-3": 2, "4-4": 6, "5-1": 25, "5-2": 9, "5-3": 48, "5-4": 2, "5-5": 8, "6-1": 117, "6-2": 24, "6-3": 9, "6-4": 55, "6-5": 2, "6-6": 14, "7-1": 12, "7-2": 106, "7-3": 29, "7-4": 9, "7-5": 56, "7-6": 5, "7-7": 6} | {"-1": 32, "-10": 18, "-11": 25, "-12": 16, "-13": 22, "-2": 57, "-3": 30, "-4": 145, "-5": 27, "-6": 154, "-7": 26, "-8": 32, "-9": 30, "0": 71, "1": 29, "10": 34, "12": 38, "2": 82, "3": 31, "4": 167, "5": 41, "6": 159, "7": 38, "8": 30} | {"000/000": 535, "000/001": 15, "000/010": 43, "000/100": 10, "001/000": 17, "010/000": 28, "100/000": 19} | {"10": 361, "13": 10, "14": 4, "15": 67, "18": 2, "19": 6, "20": 16, "23": 2, "25": 2, "3": 11, "4": 3, "5": 175, "9": 8} | {"0-5": 438, "6-10": 49, ">10": 44, "none": 136} | {"at least one": 103, "both": 0, "neither": 564} |
| center_only_s1 | selection | main | multi | {"1-1": 5, "2-1": 1, "2-2": 5, "3-1": 41, "3-2": 6, "3-3": 5, "4-1": 8, "4-2": 41, "4-3": 6, "4-4": 7, "5-1": 13, "5-2": 8, "5-3": 41, "5-4": 2, "5-5": 6, "6-1": 753, "6-2": 14, "6-3": 8, "6-4": 43, "6-5": 4, "6-6": 4, "7-1": 29, "7-2": 747, "7-3": 12, "7-4": 8, "7-5": 42, "7-6": 10, "7-7": 4} | {"-1": 30, "-10": 7, "-11": 25, "-12": 4, "-13": 28, "-2": 52, "-3": 12, "-4": 791, "-5": 10, "-6": 797, "-7": 12, "-8": 14, "-9": 40, "0": 54, "1": 32, "10": 18, "12": 21, "2": 66, "3": 26, "4": 812, "5": 18, "6": 835, "7": 19, "8": 23} | {"000/000": 1757, "000/001": 12, "000/010": 34, "000/100": 12, "001/000": 18, "010/000": 20, "100/000": 20} | {"10": 1623, "13": 16, "15": 54, "19": 15, "4": 2, "5": 149, "9": 14} | {"0-5": 1532, "6-10": 176, ">10": 165} | {"at least one": 84, "both": 0, "neither": 1789} |
| center_only_s1 | selection | main, equal marker | native | {"1-1": 8, "2-1": 1, "2-2": 13, "3-1": 15, "3-2": 2, "3-3": 8, "4-2": 6, "4-4": 3, "5-1": 10, "5-2": 5, "5-3": 22, "5-5": 9, "6-1": 241, "6-2": 11, "6-3": 6, "6-4": 4, "6-5": 1, "6-6": 11, "7-1": 2, "7-2": 192, "7-3": 4, "7-4": 1, "7-5": 22, "7-6": 2, "7-7": 2} | {"-10": 5, "-11": 20, "-12": 9, "-13": 7, "-2": 31, "-3": 20, "-4": 207, "-5": 5, "-6": 257, "-7": 15, "-8": 15, "-9": 12, "10": 10, "12": 11, "2": 49, "3": 14, "4": 269, "5": 9, "6": 213, "7": 11, "8": 13} | {"000/000": 601} | {"10": 487, "13": 10, "15": 22, "19": 1, "20": 2, "3": 6, "4": 1, "5": 67, "9": 5} | {"none": 601} | {"at least one": 0, "both": 0, "neither": 601} |
| center_only_s1 | selection | main, equal marker | pairs | {"1-1": 4, "2-1": 2, "2-2": 7, "3-1": 30, "3-2": 2, "3-3": 5, "4-1": 2, "4-2": 14, "4-4": 2, "5-1": 25, "5-2": 9, "5-3": 48, "5-4": 2, "5-5": 5, "6-1": 117, "6-2": 24, "6-3": 9, "6-4": 8, "6-5": 1, "6-6": 10, "7-1": 12, "7-2": 106, "7-3": 29, "7-4": 9, "7-5": 50, "7-6": 3} | {"-10": 10, "-11": 25, "-12": 16, "-13": 19, "-2": 57, "-3": 27, "-4": 129, "-5": 12, "-6": 137, "-7": 26, "-8": 32, "-9": 28, "10": 16, "12": 38, "2": 82, "3": 29, "4": 154, "5": 16, "6": 149, "7": 38, "8": 30} | {"000/000": 535} | {"10": 334, "13": 8, "14": 3, "15": 67, "18": 2, "19": 6, "20": 16, "23": 2, "25": 2, "3": 9, "5": 79, "9": 7} | {"0-5": 363, "6-10": 38, ">10": 29, "none": 105} | {"at least one": 0, "both": 0, "neither": 535} |
| center_only_s1 | selection | main, equal marker | multi | {"1-1": 4, "2-1": 1, "2-2": 5, "3-1": 24, "3-2": 6, "3-3": 2, "4-2": 19, "5-1": 13, "5-2": 8, "5-3": 41, "5-4": 2, "5-5": 5, "6-1": 753, "6-2": 14, "6-3": 8, "6-4": 5, "6-5": 2, "6-6": 3, "7-1": 29, "7-2": 747, "7-3": 12, "7-4": 8, "7-5": 39, "7-6": 5, "7-7": 2} | {"-10": 1, "-11": 25, "-12": 4, "-13": 24, "-2": 52, "-3": 12, "-4": 771, "-5": 3, "-6": 779, "-7": 12, "-8": 14, "-9": 37, "10": 1, "12": 21, "2": 66, "3": 25, "4": 801, "5": 1, "6": 823, "7": 19, "8": 23} | {"000/000": 1757} | {"10": 1600, "13": 12, "15": 54, "19": 15, "4": 1, "5": 64, "9": 11} | {"0-5": 1464, "6-10": 157, ">10": 136} | {"at least one": 0, "both": 0, "neither": 1757} |
| center_only_s1 | selection | main, changed marker | native | {"1-1": 8, "2-2": 1, "3-1": 5, "3-3": 9, "4-1": 5, "4-2": 16, "4-4": 9, "5-5": 5, "6-4": 27, "6-5": 2, "6-6": 10, "7-5": 5, "7-6": 2, "7-7": 4} | {"-1": 24, "-10": 6, "-12": 1, "-13": 3, "-3": 5, "-4": 9, "-5": 16, "-6": 9, "-8": 1, "0": 64, "1": 20, "10": 14, "4": 16, "5": 15, "6": 12, "8": 1} | {"000/001": 16, "000/010": 31, "000/100": 11, "001/000": 8, "010/000": 33, "100/000": 9} | {"10": 20, "13": 3, "3": 5, "4": 1, "5": 79} | {"none": 108} | {"at least one": 88, "both": 0, "neither": 20} |
| center_only_s1 | selection | main, changed marker | pairs | {"1-1": 3, "3-1": 16, "3-3": 3, "4-1": 6, "4-2": 29, "4-3": 2, "4-4": 4, "5-5": 3, "6-4": 47, "6-5": 1, "6-6": 4, "7-5": 6, "7-6": 2, "7-7": 6} | {"-1": 32, "-10": 8, "-13": 3, "-3": 3, "-4": 16, "-5": 15, "-6": 17, "-9": 2, "0": 71, "1": 29, "10": 18, "3": 2, "4": 13, "5": 25, "6": 10} | {"000/001": 15, "000/010": 43, "000/100": 10, "001/000": 17, "010/000": 28, "100/000": 19} | {"10": 27, "13": 2, "14": 1, "3": 2, "4": 3, "5": 96, "9": 1} | {"0-5": 75, "6-10": 11, ">10": 15, "none": 31} | {"at least one": 103, "both": 0, "neither": 29} |
| center_only_s1 | selection | main, changed marker | multi | {"1-1": 1, "3-1": 17, "3-3": 3, "4-1": 8, "4-2": 22, "4-3": 6, "4-4": 7, "5-5": 1, "6-4": 38, "6-5": 2, "6-6": 1, "7-5": 3, "7-6": 5, "7-7": 2} | {"-1": 30, "-10": 6, "-13": 4, "-4": 20, "-5": 7, "-6": 18, "-9": 3, "0": 54, "1": 32, "10": 17, "3": 1, "4": 11, "5": 17, "6": 12} | {"000/001": 12, "000/010": 34, "000/100": 12, "001/000": 18, "010/000": 20, "100/000": 20} | {"10": 23, "13": 4, "4": 1, "5": 85, "9": 3} | {"0-5": 68, "6-10": 19, ">10": 29} | {"at least one": 84, "both": 0, "neither": 32} |
| center_only_s1 | selection | broad | native | {"0-0": 7, "1-0": 5, "1-1": 16, "2-0": 28, "2-1": 1, "2-2": 14, "3-0": 6, "3-1": 20, "3-2": 2, "3-3": 17, "4-0": 4, "4-1": 5, "4-2": 22, "4-4": 12, "5-0": 169, "5-1": 10, "5-2": 5, "5-3": 22, "5-5": 14, "6-0": 7, "6-1": 241, "6-2": 11, "6-3": 6, "6-4": 31, "6-5": 3, "6-6": 21, "7-0": 18, "7-1": 2, "7-2": 192, "7-3": 4, "7-4": 1, "7-5": 27, "7-6": 4, "7-7": 6} | {"-1": 24, "-10": 14, "-11": 27, "-12": 13, "-13": 12, "-14": 7, "-15": 11, "-2": 41, "-3": 35, "-4": 222, "-5": 21, "-6": 271, "-7": 18, "-8": 222, "-9": 17, "0": 68, "1": 20, "10": 26, "12": 18, "2": 230, "3": 14, "4": 290, "5": 24, "6": 229, "7": 18, "8": 14} | {"000/000": 841, "000/001": 16, "000/010": 31, "000/100": 11, "001/000": 8, "010/000": 37, "100/000": 9} | {"10": 696, "13": 16, "15": 37, "19": 3, "20": 12, "25": 1, "3": 11, "4": 3, "5": 167, "9": 7} | {"none": 953} | {"at least one": 92, "both": 0, "neither": 861} |
| center_only_s1 | selection | broad | pairs | {"0-0": 5, "1-0": 2, "1-1": 7, "2-0": 49, "2-1": 2, "2-2": 7, "3-0": 12, "3-1": 46, "3-2": 2, "3-3": 8, "4-0": 23, "4-1": 8, "4-2": 43, "4-3": 2, "4-4": 6, "5-0": 104, "5-1": 25, "5-2": 9, "5-3": 48, "5-4": 2, "5-5": 8, "6-0": 12, "6-1": 117, "6-2": 24, "6-3": 9, "6-4": 55, "6-5": 2, "6-6": 14, "7-0": 100, "7-1": 12, "7-2": 106, "7-3": 29, "7-4": 9, "7-5": 56, "7-6": 5, "7-7": 6} | {"-1": 34, "-10": 29, "-11": 30, "-12": 22, "-13": 40, "-14": 22, "-15": 21, "-2": 68, "-3": 47, "-4": 159, "-5": 27, "-6": 156, "-7": 29, "-8": 233, "-9": 52, "0": 82, "1": 35, "10": 38, "12": 79, "2": 200, "3": 33, "4": 173, "5": 45, "6": 183, "7": 79, "8": 32} | {"000/000": 823, "000/001": 15, "000/010": 43, "000/100": 10, "001/000": 19, "010/000": 39, "100/000": 25} | {"10": 493, "13": 14, "14": 5, "15": 136, "18": 3, "19": 9, "20": 74, "23": 3, "25": 9, "3": 11, "4": 3, "5": 205, "9": 9} | {"0-5": 640, "6-10": 87, ">10": 59, "none": 188} | {"at least one": 116, "both": 0, "neither": 858} |
| center_only_s1 | selection | broad | multi | {"0-0": 2, "1-0": 8, "1-1": 5, "2-0": 38, "2-1": 1, "2-2": 5, "3-0": 8, "3-1": 41, "3-2": 6, "3-3": 5, "4-0": 12, "4-1": 8, "4-2": 41, "4-3": 6, "4-4": 7, "5-0": 743, "5-1": 13, "5-2": 8, "5-3": 41, "5-4": 2, "5-5": 6, "6-0": 30, "6-1": 753, "6-2": 14, "6-3": 8, "6-4": 43, "6-5": 4, "6-6": 4, "7-0": 124, "7-1": 29, "7-2": 747, "7-3": 12, "7-4": 8, "7-5": 42, "7-6": 10, "7-7": 4} | {"-1": 30, "-10": 14, "-11": 44, "-12": 8, "-13": 36, "-14": 15, "-15": 30, "-2": 60, "-3": 28, "-4": 796, "-5": 10, "-6": 805, "-7": 12, "-8": 884, "-9": 54, "0": 66, "1": 32, "10": 18, "12": 73, "2": 826, "3": 26, "4": 842, "5": 18, "6": 857, "7": 69, "8": 23} | {"000/000": 2710, "000/001": 12, "000/010": 34, "000/100": 12, "001/000": 18, "010/000": 32, "100/000": 20} | {"10": 2388, "13": 20, "15": 144, "19": 30, "20": 63, "4": 2, "5": 173, "9": 18} | {"0-5": 2257, "6-10": 353, ">10": 228} | {"at least one": 96, "both": 0, "neither": 2742} |
| center_only_s2 | selection | main | native | {"2-2": 1, "3-1": 5, "3-3": 1, "4-1": 2, "4-2": 5, "4-4": 1, "5-1": 4, "5-2": 2, "5-3": 4, "6-1": 41, "6-2": 4, "6-3": 2, "6-4": 6, "7-1": 2, "7-2": 39, "7-3": 4, "7-4": 2, "7-5": 6} | {"-1": 4, "-10": 1, "-11": 3, "-13": 4, "-2": 4, "-3": 1, "-4": 47, "-6": 49, "-7": 1, "-8": 3, "-9": 4, "0": 8, "1": 4, "10": 3, "12": 3, "2": 13, "3": 2, "4": 50, "5": 3, "6": 50, "7": 3, "8": 2} | {"000/000": 115, "000/001": 1, "000/010": 6, "000/100": 1, "001/000": 3, "010/000": 2, "100/000": 3} | {"10": 102, "13": 4, "15": 7, "19": 1, "5": 17} | {"none": 131} | {"at least one": 12, "both": 0, "neither": 119} |
| center_only_s2 | selection | main | pairs | {"1-1": 1, "2-2": 1, "3-1": 47, "3-3": 1, "4-1": 2, "4-2": 45, "4-4": 1, "5-1": 20, "5-2": 2, "5-3": 43, "5-5": 1, "6-1": 91, "6-2": 20, "6-3": 4, "6-4": 41, "6-6": 1, "7-1": 12, "7-2": 89, "7-3": 20, "7-4": 4, "7-5": 41, "7-7": 1} | {"-1": 23, "-10": 16, "-11": 14, "-12": 14, "-13": 14, "-2": 43, "-3": 15, "-4": 124, "-5": 14, "-6": 125, "-7": 16, "-8": 26, "-9": 14, "0": 44, "1": 23, "10": 22, "12": 26, "2": 56, "3": 17, "4": 133, "5": 19, "6": 135, "7": 23, "8": 20} | {"000/000": 398, "000/001": 9, "000/010": 25, "000/100": 9, "001/000": 14, "010/000": 19, "100/000": 14} | {"10": 299, "13": 6, "15": 47, "18": 2, "19": 2, "20": 15, "23": 2, "25": 2, "5": 113} | {"0-5": 429, "6-10": 14, ">10": 17, "none": 28} | {"at least one": 67, "both": 0, "neither": 421} |
| center_only_s2 | selection | main | multi | {"1-1": 4, "2-1": 4, "2-2": 3, "3-1": 45, "3-2": 4, "3-3": 4, "4-1": 6, "4-2": 43, "4-3": 2, "4-4": 2, "5-1": 14, "5-2": 6, "5-3": 42, "5-4": 2, "5-5": 2, "6-1": 780, "6-2": 14, "6-3": 6, "6-4": 39, "6-5": 4, "6-6": 4, "7-1": 24, "7-2": 778, "7-3": 14, "7-4": 6, "7-5": 42, "7-6": 3, "7-7": 3} | {"-1": 25, "-10": 6, "-11": 23, "-12": 4, "-13": 24, "-2": 50, "-3": 9, "-4": 823, "-5": 9, "-6": 822, "-7": 13, "-8": 13, "-9": 36, "0": 53, "1": 27, "10": 19, "12": 18, "2": 65, "3": 20, "4": 831, "5": 19, "6": 853, "7": 18, "8": 20} | {"000/000": 1795, "000/001": 7, "000/010": 36, "000/100": 8, "001/000": 18, "010/000": 17, "100/000": 19} | {"10": 1679, "13": 12, "15": 52, "19": 12, "4": 2, "5": 134, "9": 9} | {"0-5": 1522, "6-10": 163, ">10": 215} | {"at least one": 78, "both": 0, "neither": 1822} |
| center_only_s2 | selection | main, equal marker | native | {"2-2": 1, "3-1": 2, "4-2": 2, "5-1": 4, "5-2": 2, "5-3": 4, "6-1": 41, "6-2": 4, "6-3": 2, "7-1": 2, "7-2": 39, "7-3": 4, "7-4": 2, "7-5": 6} | {"-11": 3, "-13": 3, "-2": 4, "-3": 1, "-4": 44, "-6": 46, "-7": 1, "-8": 3, "-9": 4, "12": 3, "2": 13, "3": 2, "4": 49, "6": 49, "7": 3, "8": 2} | {"000/000": 115} | {"10": 98, "13": 3, "15": 7, "19": 1, "5": 6} | {"none": 115} | {"at least one": 0, "both": 0, "neither": 115} |
| center_only_s2 | selection | main, equal marker | pairs | {"2-2": 1, "3-1": 33, "4-2": 15, "5-1": 20, "5-2": 2, "5-3": 43, "5-5": 1, "6-1": 91, "6-2": 20, "6-3": 4, "6-4": 8, "6-6": 1, "7-1": 12, "7-2": 89, "7-3": 20, "7-4": 4, "7-5": 33, "7-7": 1} | {"-10": 7, "-11": 14, "-12": 14, "-13": 13, "-2": 43, "-3": 15, "-4": 110, "-5": 5, "-6": 111, "-7": 16, "-8": 26, "-9": 14, "10": 9, "12": 26, "2": 56, "3": 17, "4": 124, "5": 7, "6": 126, "7": 23, "8": 20} | {"000/000": 398} | {"10": 277, "13": 5, "15": 47, "18": 2, "19": 2, "20": 15, "23": 2, "25": 2, "5": 46} | {"0-5": 349, "6-10": 13, ">10": 15, "none": 21} | {"at least one": 0, "both": 0, "neither": 398} |
| center_only_s2 | selection | main, equal marker | multi | {"1-1": 2, "2-1": 4, "2-2": 3, "3-1": 27, "3-2": 4, "3-3": 2, "4-2": 19, "5-1": 14, "5-2": 6, "5-3": 42, "5-4": 2, "5-5": 2, "6-1": 780, "6-2": 14, "6-3": 6, "6-4": 2, "6-5": 2, "6-6": 2, "7-1": 24, "7-2": 778, "7-3": 14, "7-4": 6, "7-5": 39, "7-6": 1} | {"-11": 23, "-12": 4, "-13": 21, "-2": 50, "-3": 8, "-4": 805, "-5": 2, "-6": 804, "-7": 13, "-8": 13, "-9": 35, "10": 1, "12": 18, "2": 65, "3": 20, "4": 824, "5": 1, "6": 845, "7": 18, "8": 20} | {"000/000": 1795} | {"10": 1655, "13": 9, "15": 52, "19": 12, "4": 1, "5": 58, "9": 8} | {"0-5": 1446, "6-10": 156, ">10": 193} | {"at least one": 0, "both": 0, "neither": 1795} |
| center_only_s2 | selection | main, changed marker | native | {"3-1": 3, "3-3": 1, "4-1": 2, "4-2": 3, "4-4": 1, "6-4": 6} | {"-1": 4, "-10": 1, "-13": 1, "-4": 3, "-6": 3, "0": 8, "1": 4, "10": 3, "4": 1, "5": 3, "6": 1} | {"000/001": 1, "000/010": 6, "000/100": 1, "001/000": 3, "010/000": 2, "100/000": 3} | {"10": 4, "13": 1, "5": 11} | {"none": 16} | {"at least one": 12, "both": 0, "neither": 4} |
| center_only_s2 | selection | main, changed marker | pairs | {"1-1": 1, "3-1": 14, "3-3": 1, "4-1": 2, "4-2": 30, "4-4": 1, "6-4": 33, "7-5": 8} | {"-1": 23, "-10": 9, "-13": 1, "-4": 14, "-5": 9, "-6": 14, "0": 44, "1": 23, "10": 13, "4": 9, "5": 12, "6": 9} | {"000/001": 9, "000/010": 25, "000/100": 9, "001/000": 14, "010/000": 19, "100/000": 14} | {"10": 22, "13": 1, "5": 67} | {"0-5": 80, "6-10": 1, ">10": 2, "none": 7} | {"at least one": 67, "both": 0, "neither": 23} |
| center_only_s2 | selection | main, changed marker | multi | {"1-1": 2, "3-1": 18, "3-3": 2, "4-1": 6, "4-2": 24, "4-3": 2, "4-4": 2, "6-4": 37, "6-5": 2, "6-6": 2, "7-5": 3, "7-6": 2, "7-7": 3} | {"-1": 25, "-10": 6, "-13": 3, "-3": 1, "-4": 18, "-5": 7, "-6": 18, "-9": 1, "0": 53, "1": 27, "10": 18, "4": 7, "5": 18, "6": 8} | {"000/001": 7, "000/010": 36, "000/100": 8, "001/000": 18, "010/000": 17, "100/000": 19} | {"10": 24, "13": 3, "4": 1, "5": 76, "9": 1} | {"0-5": 76, "6-10": 7, ">10": 22} | {"at least one": 78, "both": 0, "neither": 27} |
| center_only_s2 | selection | broad | native | {"2-0": 7, "2-2": 1, "3-0": 2, "3-1": 5, "3-3": 1, "4-0": 4, "4-1": 2, "4-2": 5, "4-4": 1, "5-0": 44, "5-1": 4, "5-2": 2, "5-3": 4, "6-0": 2, "6-1": 41, "6-2": 4, "6-3": 2, "6-4": 6, "7-0": 12, "7-1": 2, "7-2": 39, "7-3": 4, "7-4": 2, "7-5": 6} | {"-1": 4, "-10": 3, "-11": 4, "-12": 1, "-13": 4, "-14": 1, "-15": 4, "-2": 6, "-3": 5, "-4": 47, "-6": 49, "-7": 1, "-8": 64, "-9": 5, "0": 12, "1": 4, "10": 3, "12": 8, "2": 60, "3": 2, "4": 52, "5": 3, "6": 52, "7": 8, "8": 2} | {"000/000": 182, "000/001": 1, "000/010": 6, "000/100": 1, "001/000": 3, "010/000": 6, "100/000": 3} | {"10": 152, "13": 5, "15": 16, "19": 2, "20": 6, "5": 21} | {"none": 202} | {"at least one": 16, "both": 0, "neither": 186} |
| center_only_s2 | selection | broad | pairs | {"0-0": 2, "1-0": 1, "1-1": 1, "2-0": 44, "2-2": 1, "3-0": 2, "3-1": 47, "3-3": 1, "4-0": 20, "4-1": 2, "4-2": 45, "4-4": 1, "5-0": 90, "5-1": 20, "5-2": 2, "5-3": 43, "5-5": 1, "6-0": 11, "6-1": 91, "6-2": 20, "6-3": 4, "6-4": 41, "6-6": 1, "7-0": 100, "7-1": 12, "7-2": 89, "7-3": 20, "7-4": 4, "7-5": 41, "7-7": 1} | {"-1": 23, "-10": 25, "-11": 18, "-12": 17, "-13": 27, "-14": 25, "-15": 14, "-2": 45, "-3": 28, "-4": 141, "-5": 14, "-6": 126, "-7": 16, "-8": 206, "-9": 38, "0": 54, "1": 27, "10": 26, "12": 66, "2": 156, "3": 19, "4": 136, "5": 23, "6": 161, "7": 63, "8": 22} | {"000/000": 654, "000/001": 9, "000/010": 25, "000/100": 9, "001/000": 14, "010/000": 29, "100/000": 18} | {"10": 411, "13": 7, "15": 113, "18": 3, "19": 3, "20": 73, "23": 3, "25": 8, "4": 1, "5": 136} | {"0-5": 629, "6-10": 59, ">10": 25, "none": 45} | {"at least one": 77, "both": 0, "neither": 681} |
| center_only_s2 | selection | broad | multi | {"0-0": 6, "1-0": 4, "1-1": 4, "2-0": 42, "2-1": 4, "2-2": 3, "3-0": 6, "3-1": 45, "3-2": 4, "3-3": 4, "4-0": 14, "4-1": 6, "4-2": 43, "4-3": 2, "4-4": 2, "5-0": 777, "5-1": 14, "5-2": 6, "5-3": 42, "5-4": 2, "5-5": 2, "6-0": 26, "6-1": 780, "6-2": 14, "6-3": 6, "6-4": 39, "6-5": 4, "6-6": 4, "7-0": 127, "7-1": 24, "7-2": 778, "7-3": 14, "7-4": 6, "7-5": 42, "7-6": 3, "7-7": 3} | {"-1": 25, "-10": 13, "-11": 38, "-12": 7, "-13": 35, "-14": 14, "-15": 25, "-2": 59, "-3": 27, "-4": 828, "-5": 9, "-6": 826, "-7": 16, "-8": 924, "-9": 52, "0": 67, "1": 27, "10": 19, "12": 70, "2": 861, "3": 20, "4": 857, "5": 19, "6": 876, "7": 70, "8": 20} | {"000/000": 2783, "000/001": 7, "000/010": 36, "000/100": 8, "001/000": 18, "010/000": 31, "100/000": 19} | {"10": 2478, "13": 15, "15": 145, "19": 25, "20": 63, "4": 2, "5": 163, "9": 11} | {"0-5": 2242, "6-10": 338, ">10": 322} | {"at least one": 92, "both": 0, "neither": 2810} |
| shuffled_dense_s0 | selection | main | native | {"2-2": 1, "3-1": 4, "4-1": 2, "4-2": 4, "5-2": 2, "5-3": 2, "6-1": 40, "6-3": 2, "6-4": 2, "7-2": 42, "7-4": 2, "7-5": 4} | {"-1": 2, "-10": 1, "-11": 1, "-13": 1, "-2": 2, "-3": 1, "-4": 47, "-6": 45, "-7": 1, "-8": 1, "-9": 1, "0": 4, "1": 2, "10": 1, "12": 2, "2": 7, "3": 1, "4": 44, "5": 1, "6": 46, "7": 2, "8": 1} | {"000/000": 99, "000/010": 2, "001/000": 2, "010/000": 2, "100/000": 2} | {"10": 94, "13": 4, "5": 9} | {"none": 107} | {"at least one": 6, "both": 0, "neither": 101} |
| shuffled_dense_s0 | selection | main | pairs | {"1-1": 2, "2-1": 2, "2-2": 3, "3-1": 42, "3-2": 1, "3-3": 2, "4-1": 2, "4-2": 42, "4-4": 2, "5-1": 26, "5-2": 2, "5-3": 42, "5-5": 1, "6-1": 87, "6-2": 26, "6-3": 4, "6-4": 42, "6-6": 1, "7-1": 14, "7-2": 88, "7-3": 26, "7-4": 2, "7-5": 43, "7-6": 4, "7-7": 2} | {"-1": 22, "-10": 22, "-11": 15, "-12": 19, "-13": 16, "-2": 44, "-3": 19, "-4": 122, "-5": 17, "-6": 121, "-7": 20, "-8": 31, "-9": 17, "0": 46, "1": 24, "10": 23, "12": 27, "2": 54, "3": 22, "4": 126, "5": 25, "6": 136, "7": 27, "8": 21} | {"000/000": 416, "000/001": 7, "000/010": 28, "000/100": 10, "001/000": 15, "010/000": 18, "100/000": 14} | {"10": 297, "13": 5, "15": 58, "18": 2, "19": 3, "20": 18, "23": 2, "25": 2, "4": 1, "5": 117, "9": 3} | {"0-5": 457, "6-10": 18, ">10": 10, "none": 23} | {"at least one": 68, "both": 0, "neither": 440} |
| shuffled_dense_s0 | selection | main | multi | {"2-1": 2, "3-1": 44, "3-2": 2, "4-1": 6, "4-2": 40, "4-3": 2, "5-1": 12, "5-2": 6, "5-3": 39, "5-4": 2, "5-5": 1, "6-1": 749, "6-2": 12, "6-3": 6, "6-4": 38, "6-5": 2, "7-1": 26, "7-2": 748, "7-3": 12, "7-4": 6, "7-5": 42, "7-6": 2} | {"-1": 22, "-10": 5, "-11": 21, "-12": 3, "-13": 23, "-2": 43, "-3": 4, "-4": 786, "-5": 4, "-6": 787, "-7": 8, "-8": 11, "-9": 35, "0": 46, "1": 21, "10": 17, "12": 19, "2": 62, "3": 18, "4": 792, "5": 17, "6": 816, "7": 19, "8": 19} | {"000/000": 1710, "000/001": 3, "000/010": 34, "000/100": 3, "001/000": 19, "010/000": 12, "100/000": 18} | {"10": 1607, "13": 12, "15": 53, "19": 13, "5": 108, "9": 6} | {"0-5": 1538, "6-10": 109, ">10": 152} | {"at least one": 68, "both": 0, "neither": 1731} |
| shuffled_dense_s0 | selection | main, equal marker | native | {"2-2": 1, "3-1": 2, "4-2": 2, "5-2": 2, "5-3": 2, "6-1": 40, "6-3": 2, "7-2": 42, "7-4": 2, "7-5": 4} | {"-11": 1, "-2": 2, "-3": 1, "-4": 45, "-6": 43, "-7": 1, "-8": 1, "-9": 1, "12": 2, "2": 7, "3": 1, "4": 44, "6": 46, "7": 2, "8": 1} | {"000/000": 99} | {"10": 92, "13": 3, "5": 4} | {"none": 99} | {"at least one": 0, "both": 0, "neither": 99} |
| shuffled_dense_s0 | selection | main, equal marker | pairs | {"1-1": 2, "2-1": 2, "2-2": 3, "3-1": 29, "3-2": 1, "3-3": 2, "4-2": 13, "5-1": 26, "5-2": 2, "5-3": 42, "6-1": 87, "6-2": 26, "6-3": 4, "6-4": 9, "6-6": 1, "7-1": 14, "7-2": 88, "7-3": 26, "7-4": 2, "7-5": 35, "7-6": 2} | {"-10": 12, "-11": 15, "-12": 19, "-13": 15, "-2": 44, "-3": 19, "-4": 108, "-5": 10, "-6": 106, "-7": 20, "-8": 31, "-9": 17, "10": 10, "12": 27, "2": 54, "3": 22, "4": 119, "5": 10, "6": 126, "7": 27, "8": 21} | {"000/000": 416} | {"10": 274, "13": 4, "15": 58, "18": 2, "19": 3, "20": 18, "23": 2, "25": 2, "4": 1, "5": 49, "9": 3} | {"0-5": 372, "6-10": 16, ">10": 10, "none": 18} | {"at least one": 0, "both": 0, "neither": 416} |
| shuffled_dense_s0 | selection | main, equal marker | multi | {"2-1": 2, "3-1": 25, "3-2": 2, "4-2": 18, "5-1": 12, "5-2": 6, "5-3": 39, "5-4": 2, "5-5": 1, "6-1": 749, "6-2": 12, "6-3": 6, "6-4": 2, "6-5": 1, "7-1": 26, "7-2": 748, "7-3": 12, "7-4": 6, "7-5": 40, "7-6": 1} | {"-11": 21, "-12": 3, "-13": 20, "-2": 43, "-3": 4, "-4": 768, "-5": 1, "-6": 768, "-7": 8, "-8": 11, "-9": 34, "12": 19, "2": 62, "3": 18, "4": 789, "6": 813, "7": 19, "8": 19} | {"000/000": 1710} | {"10": 1585, "13": 9, "15": 53, "19": 13, "5": 45, "9": 5} | {"0-5": 1465, "6-10": 106, ">10": 139} | {"at least one": 0, "both": 0, "neither": 1710} |
| shuffled_dense_s0 | selection | main, changed marker | native | {"3-1": 2, "4-1": 2, "4-2": 2, "6-4": 2} | {"-1": 2, "-10": 1, "-13": 1, "-4": 2, "-6": 2, "0": 4, "1": 2, "10": 1, "5": 1} | {"000/010": 2, "001/000": 2, "010/000": 2, "100/000": 2} | {"10": 2, "13": 1, "5": 5} | {"none": 8} | {"at least one": 6, "both": 0, "neither": 2} |
| shuffled_dense_s0 | selection | main, changed marker | pairs | {"3-1": 13, "4-1": 2, "4-2": 29, "4-4": 2, "5-5": 1, "6-4": 33, "7-5": 8, "7-6": 2, "7-7": 2} | {"-1": 22, "-10": 10, "-13": 1, "-4": 14, "-5": 7, "-6": 15, "0": 46, "1": 24, "10": 13, "4": 7, "5": 15, "6": 10} | {"000/001": 7, "000/010": 28, "000/100": 10, "001/000": 15, "010/000": 18, "100/000": 14} | {"10": 23, "13": 1, "5": 68} | {"0-5": 85, "6-10": 2, "none": 5} | {"at least one": 68, "both": 0, "neither": 24} |
| shuffled_dense_s0 | selection | main, changed marker | multi | {"3-1": 19, "4-1": 6, "4-2": 22, "4-3": 2, "6-4": 36, "6-5": 1, "7-5": 2, "7-6": 1} | {"-1": 22, "-10": 5, "-13": 3, "-4": 18, "-5": 3, "-6": 19, "-9": 1, "0": 46, "1": 21, "10": 17, "4": 3, "5": 17, "6": 3} | {"000/001": 3, "000/010": 34, "000/100": 3, "001/000": 19, "010/000": 12, "100/000": 18} | {"10": 22, "13": 3, "5": 63, "9": 1} | {"0-5": 73, "6-10": 3, ">10": 13} | {"at least one": 68, "both": 0, "neither": 21} |
| shuffled_dense_s0 | selection | broad | native | {"0-0": 1, "2-0": 5, "2-2": 1, "3-0": 2, "3-1": 4, "4-1": 2, "4-2": 4, "5-0": 43, "5-2": 2, "5-3": 2, "6-1": 40, "6-3": 2, "6-4": 2, "7-0": 14, "7-2": 42, "7-4": 2, "7-5": 4} | {"-1": 2, "-10": 1, "-11": 1, "-12": 1, "-13": 1, "-14": 1, "-15": 1, "-2": 5, "-3": 3, "-4": 47, "-6": 45, "-7": 2, "-8": 61, "-9": 2, "0": 4, "1": 2, "10": 1, "12": 8, "2": 53, "3": 1, "4": 44, "5": 1, "6": 48, "7": 8, "8": 1} | {"000/000": 164, "000/010": 2, "001/000": 2, "010/000": 2, "100/000": 2} | {"10": 141, "13": 5, "15": 7, "20": 7, "5": 12} | {"none": 172} | {"at least one": 6, "both": 0, "neither": 166} |
| shuffled_dense_s0 | selection | broad | pairs | {"0-0": 2, "1-1": 2, "2-0": 40, "2-1": 2, "2-2": 3, "3-0": 2, "3-1": 42, "3-2": 1, "3-3": 2, "4-0": 26, "4-1": 2, "4-2": 42, "4-4": 2, "5-0": 87, "5-1": 26, "5-2": 2, "5-3": 42, "5-5": 1, "6-0": 14, "6-1": 87, "6-2": 26, "6-3": 4, "6-4": 42, "6-6": 1, "7-0": 96, "7-1": 14, "7-2": 88, "7-3": 26, "7-4": 2, "7-5": 43, "7-6": 4, "7-7": 2} | {"-1": 22, "-10": 31, "-11": 18, "-12": 22, "-13": 30, "-14": 30, "-15": 15, "-2": 46, "-3": 31, "-4": 138, "-5": 17, "-6": 121, "-7": 20, "-8": 198, "-9": 45, "0": 56, "1": 32, "10": 27, "12": 64, "2": 149, "3": 24, "4": 132, "5": 29, "6": 166, "7": 64, "8": 23} | {"000/000": 665, "000/001": 7, "000/010": 28, "000/100": 10, "001/000": 15, "010/000": 28, "100/000": 22} | {"10": 405, "13": 6, "15": 127, "18": 3, "19": 6, "20": 75, "23": 3, "25": 8, "4": 1, "5": 138, "9": 3} | {"0-5": 662, "6-10": 60, ">10": 20, "none": 33} | {"at least one": 78, "both": 0, "neither": 697} |
| shuffled_dense_s0 | selection | broad | multi | {"0-0": 2, "1-0": 2, "2-0": 46, "2-1": 2, "3-0": 6, "3-1": 44, "3-2": 2, "4-0": 12, "4-1": 6, "4-2": 40, "4-3": 2, "5-0": 757, "5-1": 12, "5-2": 6, "5-3": 39, "5-4": 2, "5-5": 1, "6-0": 26, "6-1": 749, "6-2": 12, "6-3": 6, "6-4": 38, "6-5": 2, "7-0": 128, "7-1": 26, "7-2": 748, "7-3": 12, "7-4": 6, "7-5": 42, "7-6": 2} | {"-1": 22, "-10": 11, "-11": 35, "-12": 6, "-13": 34, "-14": 13, "-15": 23, "-2": 51, "-3": 25, "-4": 790, "-5": 4, "-6": 789, "-7": 10, "-8": 905, "-9": 48, "0": 58, "1": 21, "10": 17, "12": 72, "2": 840, "3": 18, "4": 818, "5": 17, "6": 838, "7": 72, "8": 19} | {"000/000": 2677, "000/001": 3, "000/010": 34, "000/100": 3, "001/000": 19, "010/000": 24, "100/000": 18} | {"10": 2385, "13": 15, "15": 147, "19": 26, "20": 64, "5": 134, "9": 7} | {"0-5": 2264, "6-10": 272, ">10": 242} | {"at least one": 80, "both": 0, "neither": 2698} |
| shuffled_dense_s1 | selection | main | native | {"2-2": 1, "3-1": 4, "4-1": 2, "4-2": 2, "5-3": 6, "5-5": 1, "6-1": 39, "6-4": 4, "6-6": 2, "7-2": 43, "7-5": 2} | {"-1": 3, "-10": 1, "-13": 1, "-2": 7, "-3": 1, "-4": 45, "-5": 1, "-6": 43, "0": 6, "1": 1, "10": 3, "12": 1, "2": 3, "3": 3, "4": 43, "5": 2, "6": 44, "7": 1, "8": 3} | {"000/000": 96, "000/001": 1, "000/010": 4, "001/000": 2, "010/000": 2, "100/000": 1} | {"10": 92, "13": 1, "3": 1, "5": 12} | {"none": 106} | {"at least one": 9, "both": 0, "neither": 97} |
| shuffled_dense_s1 | selection | main | pairs | {"1-1": 1, "2-2": 2, "3-1": 35, "3-3": 1, "4-1": 4, "4-2": 40, "4-4": 2, "5-1": 26, "5-2": 4, "5-3": 39, "5-4": 2, "5-5": 2, "6-1": 90, "6-2": 26, "6-3": 4, "6-4": 40, "6-6": 1, "7-1": 14, "7-2": 90, "7-3": 26, "7-4": 4, "7-5": 42, "7-7": 2} | {"-1": 19, "-10": 24, "-11": 15, "-12": 17, "-13": 16, "-2": 38, "-3": 18, "-4": 121, "-5": 18, "-6": 118, "-7": 19, "-8": 31, "-9": 16, "0": 48, "1": 21, "10": 22, "12": 26, "2": 57, "3": 20, "4": 126, "5": 24, "6": 135, "7": 26, "8": 19} | {"000/000": 409, "000/001": 7, "000/010": 26, "000/100": 8, "001/000": 12, "010/000": 22, "100/000": 13} | {"10": 297, "13": 8, "15": 58, "18": 2, "19": 3, "20": 18, "23": 2, "25": 2, "5": 106, "9": 1} | {"0-5": 438, "6-10": 17, ">10": 8, "none": 34} | {"at least one": 67, "both": 0, "neither": 430} |
| shuffled_dense_s1 | selection | main | multi | {"1-1": 1, "2-1": 4, "2-2": 2, "3-1": 40, "3-2": 4, "3-3": 1, "4-1": 6, "4-2": 38, "4-3": 4, "4-4": 1, "5-1": 12, "5-2": 6, "5-3": 36, "5-4": 4, "6-1": 731, "6-2": 13, "6-3": 6, "6-4": 36, "6-5": 4, "6-6": 1, "7-1": 26, "7-2": 730, "7-3": 11, "7-4": 6, "7-5": 38, "7-6": 4} | {"-1": 21, "-10": 5, "-11": 23, "-12": 3, "-13": 24, "-2": 43, "-3": 8, "-4": 767, "-5": 7, "-6": 767, "-7": 11, "-8": 11, "-9": 39, "0": 47, "1": 21, "10": 16, "12": 17, "2": 62, "3": 16, "4": 775, "5": 16, "6": 798, "7": 17, "8": 16} | {"000/000": 1676, "000/001": 4, "000/010": 32, "000/100": 4, "001/000": 17, "010/000": 15, "100/000": 17} | {"10": 1562, "13": 12, "15": 54, "19": 13, "5": 112, "9": 12} | {"0-5": 1539, "6-10": 111, ">10": 115} | {"at least one": 68, "both": 0, "neither": 1697} |
| shuffled_dense_s1 | selection | main, equal marker | native | {"2-2": 1, "3-1": 2, "4-2": 1, "5-3": 6, "5-5": 1, "6-1": 39, "6-6": 1, "7-2": 43, "7-5": 2} | {"-2": 7, "-3": 1, "-4": 44, "-5": 1, "-6": 41, "10": 1, "12": 1, "2": 3, "3": 3, "4": 42, "6": 44, "7": 1, "8": 3} | {"000/000": 96} | {"10": 89, "3": 1, "5": 6} | {"none": 96} | {"at least one": 0, "both": 0, "neither": 96} |
| shuffled_dense_s1 | selection | main, equal marker | pairs | {"1-1": 1, "2-2": 2, "3-1": 25, "3-3": 1, "4-2": 11, "5-1": 26, "5-2": 4, "5-3": 39, "5-4": 2, "6-1": 90, "6-2": 26, "6-3": 4, "6-4": 9, "6-6": 1, "7-1": 14, "7-2": 90, "7-3": 26, "7-4": 4, "7-5": 34} | {"-10": 12, "-11": 15, "-12": 17, "-13": 14, "-2": 38, "-3": 18, "-4": 108, "-5": 10, "-6": 106, "-7": 19, "-8": 31, "-9": 16, "10": 10, "12": 26, "2": 57, "3": 20, "4": 119, "5": 10, "6": 127, "7": 26, "8": 19} | {"000/000": 409} | {"10": 273, "13": 6, "15": 58, "18": 2, "19": 3, "20": 18, "23": 2, "25": 2, "5": 44, "9": 1} | {"0-5": 359, "6-10": 16, ">10": 8, "none": 26} | {"at least one": 0, "both": 0, "neither": 409} |
| shuffled_dense_s1 | selection | main, equal marker | multi | {"2-1": 4, "2-2": 2, "3-1": 23, "3-2": 4, "3-3": 1, "4-2": 17, "4-4": 1, "5-1": 12, "5-2": 6, "5-3": 36, "5-4": 4, "6-1": 731, "6-2": 13, "6-3": 6, "6-4": 2, "6-5": 2, "6-6": 1, "7-1": 26, "7-2": 730, "7-3": 11, "7-4": 6, "7-5": 36, "7-6": 2} | {"-11": 23, "-12": 3, "-13": 21, "-2": 43, "-3": 8, "-4": 750, "-5": 2, "-6": 750, "-7": 11, "-8": 11, "-9": 37, "12": 17, "2": 62, "3": 16, "4": 771, "6": 794, "7": 17, "8": 16} | {"000/000": 1676} | {"10": 1541, "13": 9, "15": 54, "19": 13, "5": 49, "9": 10} | {"0-5": 1463, "6-10": 107, ">10": 106} | {"at least one": 0, "both": 0, "neither": 1676} |
| shuffled_dense_s1 | selection | main, changed marker | native | {"3-1": 2, "4-1": 2, "4-2": 1, "6-4": 4, "6-6": 1} | {"-1": 3, "-10": 1, "-13": 1, "-4": 1, "-6": 2, "0": 6, "1": 1, "10": 2, "4": 1, "5": 2} | {"000/001": 1, "000/010": 4, "001/000": 2, "010/000": 2, "100/000": 1} | {"10": 3, "13": 1, "5": 6} | {"none": 10} | {"at least one": 9, "both": 0, "neither": 1} |
| shuffled_dense_s1 | selection | main, changed marker | pairs | {"3-1": 10, "4-1": 4, "4-2": 29, "4-4": 2, "5-5": 2, "6-4": 31, "7-5": 8, "7-7": 2} | {"-1": 19, "-10": 12, "-13": 2, "-4": 13, "-5": 8, "-6": 12, "0": 48, "1": 21, "10": 12, "4": 7, "5": 14, "6": 8} | {"000/001": 7, "000/010": 26, "000/100": 8, "001/000": 12, "010/000": 22, "100/000": 13} | {"10": 24, "13": 2, "5": 62} | {"0-5": 79, "6-10": 1, "none": 8} | {"at least one": 67, "both": 0, "neither": 21} |
| shuffled_dense_s1 | selection | main, changed marker | multi | {"1-1": 1, "3-1": 17, "4-1": 6, "4-2": 21, "4-3": 4, "6-4": 34, "6-5": 2, "7-5": 2, "7-6": 2} | {"-1": 21, "-10": 5, "-13": 3, "-4": 17, "-5": 5, "-6": 17, "-9": 2, "0": 47, "1": 21, "10": 16, "4": 4, "5": 16, "6": 4} | {"000/001": 4, "000/010": 32, "000/100": 4, "001/000": 17, "010/000": 15, "100/000": 17} | {"10": 21, "13": 3, "5": 63, "9": 2} | {"0-5": 76, "6-10": 4, ">10": 9} | {"at least one": 68, "both": 0, "neither": 21} |
| shuffled_dense_s1 | selection | broad | native | {"2-0": 5, "2-2": 1, "3-0": 2, "3-1": 4, "4-1": 2, "4-2": 2, "5-0": 41, "5-3": 6, "5-5": 1, "6-1": 39, "6-4": 4, "6-6": 2, "7-0": 16, "7-2": 43, "7-5": 2} | {"-1": 3, "-10": 1, "-12": 1, "-13": 1, "-14": 2, "-15": 1, "-2": 9, "-3": 3, "-4": 45, "-5": 1, "-6": 43, "-8": 58, "-9": 2, "0": 6, "1": 1, "10": 3, "12": 7, "2": 47, "3": 3, "4": 43, "5": 2, "6": 48, "7": 7, "8": 3} | {"000/000": 160, "000/001": 1, "000/010": 4, "001/000": 2, "010/000": 2, "100/000": 1} | {"10": 137, "13": 2, "15": 8, "20": 8, "3": 1, "5": 14} | {"none": 170} | {"at least one": 9, "both": 0, "neither": 161} |
| shuffled_dense_s1 | selection | broad | pairs | {"0-0": 1, "1-1": 1, "2-0": 37, "2-2": 2, "3-0": 4, "3-1": 35, "3-3": 1, "4-0": 26, "4-1": 4, "4-2": 40, "4-4": 2, "5-0": 91, "5-1": 26, "5-2": 4, "5-3": 39, "5-4": 2, "5-5": 2, "6-0": 14, "6-1": 90, "6-2": 26, "6-3": 4, "6-4": 40, "6-6": 1, "7-0": 95, "7-1": 14, "7-2": 90, "7-3": 26, "7-4": 4, "7-5": 42, "7-7": 2} | {"-1": 19, "-10": 33, "-11": 18, "-12": 21, "-13": 30, "-14": 27, "-15": 16, "-2": 42, "-3": 29, "-4": 136, "-5": 18, "-6": 118, "-7": 19, "-8": 200, "-9": 43, "0": 58, "1": 29, "10": 26, "12": 63, "2": 155, "3": 22, "4": 132, "5": 28, "6": 164, "7": 63, "8": 21} | {"000/000": 659, "000/001": 7, "000/010": 26, "000/100": 8, "001/000": 12, "010/000": 32, "100/000": 21} | {"10": 408, "13": 10, "15": 127, "18": 3, "19": 6, "20": 74, "23": 3, "25": 8, "5": 125, "9": 1} | {"0-5": 645, "6-10": 56, ">10": 17, "none": 47} | {"at least one": 77, "both": 0, "neither": 688} |
| shuffled_dense_s1 | selection | broad | multi | {"1-0": 4, "1-1": 1, "2-0": 41, "2-1": 4, "2-2": 2, "3-0": 6, "3-1": 40, "3-2": 4, "3-3": 1, "4-0": 14, "4-1": 6, "4-2": 38, "4-3": 4, "4-4": 1, "5-0": 750, "5-1": 12, "5-2": 6, "5-3": 36, "5-4": 4, "6-0": 26, "6-1": 731, "6-2": 13, "6-3": 6, "6-4": 36, "6-5": 4, "6-6": 1, "7-0": 134, "7-1": 26, "7-2": 730, "7-3": 11, "7-4": 6, "7-5": 38, "7-6": 4} | {"-1": 21, "-10": 12, "-11": 38, "-12": 6, "-13": 35, "-14": 13, "-15": 25, "-2": 49, "-3": 26, "-4": 771, "-5": 7, "-6": 771, "-7": 11, "-8": 899, "-9": 52, "0": 61, "1": 21, "10": 16, "12": 73, "2": 831, "3": 16, "4": 801, "5": 16, "6": 820, "7": 73, "8": 16} | {"000/000": 2637, "000/001": 4, "000/010": 32, "000/100": 4, "001/000": 17, "010/000": 29, "100/000": 17} | {"10": 2332, "13": 15, "15": 152, "19": 26, "20": 67, "5": 134, "9": 14} | {"0-5": 2276, "6-10": 274, ">10": 190} | {"at least one": 82, "both": 0, "neither": 2658} |
| shuffled_dense_s2 | selection | main | native | {"3-1": 4, "4-1": 2, "4-2": 3, "4-4": 1, "5-3": 4, "6-1": 44, "6-3": 2, "6-4": 3, "6-6": 1, "7-2": 47, "7-4": 2, "7-5": 4, "7-7": 1} | {"-1": 3, "-10": 1, "-13": 1, "-2": 4, "-4": 51, "-6": 49, "-7": 1, "-9": 1, "0": 5, "1": 2, "10": 3, "12": 2, "2": 5, "3": 2, "4": 48, "5": 2, "6": 52, "7": 2, "8": 2} | {"000/000": 108, "000/001": 1, "000/010": 3, "000/100": 1, "001/000": 2, "010/000": 2, "100/000": 1} | {"10": 104, "13": 3, "3": 1, "5": 10} | {"none": 118} | {"at least one": 8, "both": 0, "neither": 110} |
| shuffled_dense_s2 | selection | main | pairs | {"1-1": 2, "2-1": 2, "2-2": 4, "3-1": 41, "3-2": 2, "3-3": 1, "4-1": 2, "4-2": 40, "4-4": 2, "5-1": 27, "5-2": 4, "5-3": 44, "5-4": 2, "5-5": 1, "6-1": 88, "6-2": 27, "6-3": 4, "6-4": 42, "6-6": 1, "7-1": 14, "7-2": 91, "7-3": 28, "7-4": 4, "7-5": 43, "7-6": 2} | {"-1": 22, "-10": 22, "-11": 18, "-12": 18, "-13": 18, "-2": 47, "-3": 19, "-4": 126, "-5": 17, "-6": 120, "-7": 22, "-8": 31, "-9": 19, "0": 44, "1": 22, "10": 23, "12": 27, "2": 61, "3": 22, "4": 127, "5": 23, "6": 140, "7": 27, "8": 21} | {"000/000": 430, "000/001": 8, "000/010": 26, "000/100": 9, "001/000": 14, "010/000": 18, "100/000": 13} | {"10": 301, "13": 7, "15": 61, "18": 2, "19": 3, "20": 18, "23": 2, "25": 2, "5": 118, "9": 4} | {"0-5": 461, "6-10": 16, ">10": 8, "none": 33} | {"at least one": 66, "both": 0, "neither": 452} |
| shuffled_dense_s2 | selection | main | multi | {"2-1": 2, "2-2": 2, "3-1": 44, "3-2": 4, "4-1": 6, "4-2": 42, "4-3": 4, "4-4": 1, "5-1": 16, "5-2": 6, "5-3": 41, "5-4": 4, "5-5": 2, "6-1": 753, "6-2": 16, "6-3": 6, "6-4": 42, "6-5": 2, "6-6": 1, "7-1": 26, "7-2": 749, "7-3": 16, "7-4": 6, "7-5": 38, "7-6": 4} | {"-1": 25, "-10": 6, "-11": 24, "-12": 2, "-13": 25, "-2": 43, "-3": 9, "-4": 790, "-5": 7, "-6": 796, "-7": 9, "-8": 14, "-9": 38, "0": 53, "1": 21, "10": 18, "12": 18, "2": 70, "3": 19, "4": 804, "5": 18, "6": 821, "7": 18, "8": 18} | {"000/000": 1734, "000/001": 5, "000/010": 36, "000/100": 3, "001/000": 20, "010/000": 17, "100/000": 18} | {"10": 1620, "13": 12, "15": 58, "19": 13, "5": 120, "9": 10} | {"0-5": 1550, "6-10": 117, ">10": 166} | {"at least one": 78, "both": 0, "neither": 1755} |
| shuffled_dense_s2 | selection | main, equal marker | native | {"3-1": 2, "4-2": 2, "5-3": 4, "6-1": 44, "6-3": 2, "6-6": 1, "7-2": 47, "7-4": 2, "7-5": 4} | {"-2": 4, "-4": 50, "-6": 47, "-7": 1, "-9": 1, "10": 1, "12": 2, "2": 4, "3": 2, "4": 48, "5": 1, "6": 51, "7": 2, "8": 2} | {"000/000": 108} | {"10": 101, "13": 2, "5": 5} | {"none": 108} | {"at least one": 0, "both": 0, "neither": 108} |
| shuffled_dense_s2 | selection | main, equal marker | pairs | {"1-1": 2, "2-1": 2, "2-2": 4, "3-1": 29, "3-2": 2, "3-3": 1, "4-2": 12, "5-1": 27, "5-2": 4, "5-3": 44, "5-4": 2, "6-1": 88, "6-2": 27, "6-3": 4, "6-4": 8, "6-6": 1, "7-1": 14, "7-2": 91, "7-3": 28, "7-4": 4, "7-5": 35, "7-6": 1} | {"-10": 12, "-11": 18, "-12": 18, "-13": 17, "-2": 47, "-3": 19, "-4": 113, "-5": 10, "-6": 106, "-7": 22, "-8": 31, "-9": 19, "10": 10, "12": 27, "2": 61, "3": 22, "4": 119, "5": 10, "6": 131, "7": 27, "8": 21} | {"000/000": 430} | {"10": 278, "13": 6, "15": 61, "18": 2, "19": 3, "20": 18, "23": 2, "25": 2, "5": 54, "9": 4} | {"0-5": 378, "6-10": 15, ">10": 8, "none": 29} | {"at least one": 0, "both": 0, "neither": 430} |
| shuffled_dense_s2 | selection | main, equal marker | multi | {"2-1": 2, "2-2": 2, "3-1": 24, "3-2": 4, "4-2": 18, "5-1": 16, "5-2": 6, "5-3": 41, "5-4": 4, "5-5": 2, "6-1": 753, "6-2": 16, "6-3": 6, "6-4": 3, "6-5": 1, "7-1": 26, "7-2": 749, "7-3": 16, "7-4": 6, "7-5": 37, "7-6": 2} | {"-11": 24, "-12": 2, "-13": 22, "-2": 43, "-3": 9, "-4": 772, "-5": 1, "-6": 776, "-7": 9, "-8": 14, "-9": 36, "12": 18, "2": 70, "3": 19, "4": 799, "6": 818, "7": 18, "8": 18} | {"000/000": 1734} | {"10": 1596, "13": 9, "15": 58, "19": 13, "5": 50, "9": 8} | {"0-5": 1471, "6-10": 112, ">10": 151} | {"at least one": 0, "both": 0, "neither": 1734} |
| shuffled_dense_s2 | selection | main, changed marker | native | {"3-1": 2, "4-1": 2, "4-2": 1, "4-4": 1, "6-4": 3, "7-7": 1} | {"-1": 3, "-10": 1, "-13": 1, "-4": 1, "-6": 2, "0": 5, "1": 2, "10": 2, "2": 1, "5": 1, "6": 1} | {"000/001": 1, "000/010": 3, "000/100": 1, "001/000": 2, "010/000": 2, "100/000": 1} | {"10": 3, "13": 1, "3": 1, "5": 5} | {"none": 10} | {"at least one": 8, "both": 0, "neither": 2} |
| shuffled_dense_s2 | selection | main, changed marker | pairs | {"3-1": 12, "4-1": 2, "4-2": 28, "4-4": 2, "5-5": 1, "6-4": 34, "7-5": 8, "7-6": 1} | {"-1": 22, "-10": 10, "-13": 1, "-4": 13, "-5": 7, "-6": 14, "0": 44, "1": 22, "10": 13, "4": 8, "5": 13, "6": 9} | {"000/001": 8, "000/010": 26, "000/100": 9, "001/000": 14, "010/000": 18, "100/000": 13} | {"10": 23, "13": 1, "5": 64} | {"0-5": 83, "6-10": 1, "none": 4} | {"at least one": 66, "both": 0, "neither": 22} |
| shuffled_dense_s2 | selection | main, changed marker | multi | {"3-1": 20, "4-1": 6, "4-2": 24, "4-3": 4, "4-4": 1, "6-4": 39, "6-5": 1, "6-6": 1, "7-5": 1, "7-6": 2} | {"-1": 25, "-10": 6, "-13": 3, "-4": 18, "-5": 6, "-6": 20, "-9": 2, "0": 53, "1": 21, "10": 18, "4": 5, "5": 18, "6": 3} | {"000/001": 5, "000/010": 36, "000/100": 3, "001/000": 20, "010/000": 17, "100/000": 18} | {"10": 24, "13": 3, "5": 70, "9": 2} | {"0-5": 79, "6-10": 5, ">10": 15} | {"at least one": 78, "both": 0, "neither": 21} |
| shuffled_dense_s2 | selection | broad | native | {"0-0": 1, "2-0": 6, "3-0": 2, "3-1": 4, "4-1": 2, "4-2": 3, "4-4": 1, "5-0": 46, "5-3": 4, "6-1": 44, "6-3": 2, "6-4": 3, "6-6": 1, "7-0": 20, "7-2": 47, "7-4": 2, "7-5": 4, "7-7": 1} | {"-1": 3, "-10": 1, "-12": 1, "-13": 1, "-14": 2, "-15": 1, "-2": 7, "-3": 3, "-4": 51, "-6": 49, "-7": 2, "-8": 68, "-9": 3, "0": 5, "1": 2, "10": 3, "12": 10, "2": 54, "3": 2, "4": 48, "5": 2, "6": 56, "7": 10, "8": 2} | {"000/000": 183, "000/001": 1, "000/010": 3, "000/100": 1, "001/000": 2, "010/000": 2, "100/000": 1} | {"10": 154, "13": 4, "15": 10, "20": 10, "3": 1, "5": 14} | {"none": 193} | {"at least one": 8, "both": 0, "neither": 185} |
| shuffled_dense_s2 | selection | broad | pairs | {"0-0": 1, "1-1": 2, "2-0": 42, "2-1": 2, "2-2": 4, "3-0": 4, "3-1": 41, "3-2": 2, "3-3": 1, "4-0": 27, "4-1": 2, "4-2": 40, "4-4": 2, "5-0": 89, "5-1": 27, "5-2": 4, "5-3": 44, "5-4": 2, "5-5": 1, "6-0": 14, "6-1": 88, "6-2": 27, "6-3": 4, "6-4": 42, "6-6": 1, "7-0": 100, "7-1": 14, "7-2": 91, "7-3": 28, "7-4": 4, "7-5": 43, "7-6": 2} | {"-1": 22, "-10": 32, "-11": 21, "-12": 22, "-13": 32, "-14": 28, "-15": 18, "-2": 51, "-3": 31, "-4": 144, "-5": 17, "-6": 120, "-7": 22, "-8": 204, "-9": 47, "0": 55, "1": 30, "10": 27, "12": 66, "2": 158, "3": 24, "4": 133, "5": 27, "6": 170, "7": 66, "8": 23} | {"000/000": 688, "000/001": 8, "000/010": 26, "000/100": 9, "001/000": 14, "010/000": 29, "100/000": 21} | {"10": 412, "13": 9, "15": 133, "18": 3, "19": 6, "20": 77, "23": 3, "25": 8, "5": 140, "9": 4} | {"0-5": 669, "6-10": 59, ">10": 18, "none": 49} | {"at least one": 77, "both": 0, "neither": 718} |
| shuffled_dense_s2 | selection | broad | multi | {"0-0": 1, "1-0": 2, "2-0": 43, "2-1": 2, "2-2": 2, "3-0": 6, "3-1": 44, "3-2": 4, "4-0": 16, "4-1": 6, "4-2": 42, "4-3": 4, "4-4": 1, "5-0": 754, "5-1": 16, "5-2": 6, "5-3": 41, "5-4": 4, "5-5": 2, "6-0": 26, "6-1": 753, "6-2": 16, "6-3": 6, "6-4": 42, "6-5": 2, "6-6": 1, "7-0": 126, "7-1": 26, "7-2": 749, "7-3": 16, "7-4": 6, "7-5": 38, "7-6": 4} | {"-1": 25, "-10": 14, "-11": 38, "-12": 5, "-13": 36, "-14": 13, "-15": 25, "-2": 50, "-3": 29, "-4": 794, "-5": 7, "-6": 798, "-7": 10, "-8": 900, "-9": 51, "0": 69, "1": 21, "10": 18, "12": 70, "2": 843, "3": 19, "4": 830, "5": 18, "6": 843, "7": 70, "8": 18} | {"000/000": 2692, "000/001": 5, "000/010": 36, "000/100": 3, "001/000": 20, "010/000": 33, "100/000": 18} | {"10": 2395, "13": 15, "15": 153, "19": 26, "20": 63, "5": 144, "9": 11} | {"0-5": 2284, "6-10": 268, ">10": 255} | {"at least one": 94, "both": 0, "neither": 2713} |
| aligned_dense_s0 | calibration | main | native | {"2-2": 3, "3-2": 2, "4-2": 4, "4-3": 2, "5-3": 1, "5-5": 1, "6-1": 20, "6-4": 3, "6-6": 1, "7-2": 21, "7-5": 4, "7-6": 4} | {"-1": 1, "-11": 3, "-2": 1, "-3": 3, "-4": 24, "-5": 1, "-6": 21, "-9": 4, "0": 5, "1": 2, "10": 3, "12": 4, "2": 7, "3": 1, "4": 21, "5": 2, "6": 23, "7": 4, "8": 2} | {"000/000": 58, "000/010": 3, "001/000": 1, "010/000": 2, "100/000": 2} | {"10": 47, "15": 5, "5": 13, "9": 1} | {"none": 66} | {"at least one": 6, "both": 0, "neither": 60} |
| aligned_dense_s0 | calibration | main | pairs | {"1-1": 1, "3-1": 8, "4-2": 12, "5-1": 10, "5-3": 8, "6-1": 44, "6-2": 12, "6-3": 2, "6-4": 4, "7-1": 8, "7-2": 43, "7-3": 10, "7-4": 2, "7-5": 9, "7-7": 1} | {"-1": 2, "-10": 4, "-11": 5, "-12": 4, "-13": 7, "-2": 10, "-3": 3, "-4": 55, "-5": 5, "-6": 52, "-7": 5, "-8": 7, "-9": 8, "0": 9, "1": 5, "10": 4, "12": 6, "2": 15, "3": 5, "4": 56, "5": 4, "6": 63, "7": 9, "8": 5} | {"000/000": 158, "000/010": 4, "000/100": 1, "001/000": 2, "010/000": 5, "100/000": 4} | {"10": 120, "13": 2, "15": 22, "19": 3, "20": 4, "5": 23} | {"0-5": 149, "6-10": 4, ">10": 2, "none": 19} | {"at least one": 11, "both": 0, "neither": 163} |
| aligned_dense_s0 | calibration | main | multi | {"2-1": 6, "3-1": 13, "3-3": 1, "4-1": 4, "4-2": 10, "5-1": 4, "5-2": 2, "5-3": 14, "6-1": 297, "6-2": 2, "6-3": 4, "6-4": 15, "6-6": 1, "7-1": 10, "7-2": 312, "7-3": 6, "7-4": 2, "7-5": 13, "7-7": 1} | {"-1": 5, "-10": 4, "-11": 6, "-12": 2, "-13": 12, "-2": 18, "-4": 322, "-5": 2, "-6": 305, "-7": 3, "-8": 3, "-9": 19, "0": 23, "1": 3, "10": 9, "12": 7, "2": 19, "3": 7, "4": 309, "5": 8, "6": 333, "7": 8, "8": 7} | {"000/000": 686, "000/001": 1, "000/010": 15, "001/000": 4, "010/000": 8, "100/000": 3} | {"10": 644, "13": 6, "15": 21, "19": 5, "5": 38, "9": 3} | {"0-5": 663, "6-10": 18, ">10": 36} | {"at least one": 28, "both": 0, "neither": 689} |
| aligned_dense_s0 | calibration | main, equal marker | native | {"2-2": 3, "3-2": 1, "4-2": 2, "5-3": 1, "5-5": 1, "6-1": 20, "6-6": 1, "7-2": 21, "7-5": 4, "7-6": 4} | {"-11": 3, "-2": 1, "-3": 3, "-4": 22, "-6": 20, "-9": 3, "10": 1, "12": 4, "2": 7, "3": 1, "4": 21, "5": 1, "6": 23, "7": 4, "8": 2} | {"000/000": 58} | {"10": 45, "15": 5, "5": 8} | {"none": 58} | {"at least one": 0, "both": 0, "neither": 58} |
| aligned_dense_s0 | calibration | main, equal marker | pairs | {"3-1": 6, "4-2": 4, "5-1": 10, "5-3": 8, "6-1": 44, "6-2": 12, "6-3": 2, "7-1": 8, "7-2": 43, "7-3": 10, "7-4": 2, "7-5": 8, "7-7": 1} | {"-10": 2, "-11": 5, "-12": 4, "-13": 7, "-2": 10, "-3": 3, "-4": 51, "-5": 2, "-6": 50, "-7": 5, "-8": 7, "-9": 8, "10": 2, "12": 6, "2": 15, "3": 5, "4": 56, "5": 2, "6": 62, "7": 9, "8": 5} | {"000/000": 158} | {"10": 116, "13": 2, "15": 22, "19": 3, "20": 4, "5": 11} | {"0-5": 141, "6-10": 4, ">10": 2, "none": 11} | {"at least one": 0, "both": 0, "neither": 158} |
| aligned_dense_s0 | calibration | main, equal marker | multi | {"2-1": 6, "3-1": 9, "4-2": 3, "5-1": 4, "5-2": 2, "5-3": 14, "6-1": 297, "6-2": 2, "6-3": 4, "6-6": 1, "7-1": 10, "7-2": 312, "7-3": 6, "7-4": 2, "7-5": 13, "7-7": 1} | {"-11": 6, "-12": 2, "-13": 10, "-2": 18, "-4": 319, "-6": 301, "-7": 3, "-8": 3, "-9": 19, "10": 1, "12": 7, "2": 19, "3": 7, "4": 308, "5": 1, "6": 333, "7": 8, "8": 7} | {"000/000": 686} | {"10": 632, "13": 4, "15": 21, "19": 5, "5": 21, "9": 3} | {"0-5": 635, "6-10": 18, ">10": 33} | {"at least one": 0, "both": 0, "neither": 686} |
| aligned_dense_s0 | calibration | main, changed marker | native | {"3-2": 1, "4-2": 2, "4-3": 2, "6-4": 3} | {"-1": 1, "-4": 2, "-5": 1, "-6": 1, "-9": 1, "0": 5, "1": 2, "10": 2, "5": 1} | {"000/010": 3, "001/000": 1, "010/000": 2, "100/000": 2} | {"10": 2, "5": 5, "9": 1} | {"none": 8} | {"at least one": 6, "both": 0, "neither": 2} |
| aligned_dense_s0 | calibration | main, changed marker | pairs | {"1-1": 1, "3-1": 2, "4-2": 8, "6-4": 4, "7-5": 1} | {"-1": 2, "-10": 2, "-4": 4, "-5": 3, "-6": 2, "0": 9, "1": 5, "10": 2, "5": 2, "6": 1} | {"000/010": 4, "000/100": 1, "001/000": 2, "010/000": 5, "100/000": 4} | {"10": 4, "5": 12} | {"0-5": 8, "none": 8} | {"at least one": 11, "both": 0, "neither": 5} |
| aligned_dense_s0 | calibration | main, changed marker | multi | {"3-1": 4, "3-3": 1, "4-1": 4, "4-2": 7, "6-4": 15} | {"-1": 5, "-10": 4, "-13": 2, "-4": 3, "-5": 2, "-6": 4, "0": 23, "1": 3, "10": 8, "4": 1, "5": 7} | {"000/001": 1, "000/010": 15, "001/000": 4, "010/000": 8, "100/000": 3} | {"10": 12, "13": 2, "5": 17} | {"0-5": 28, ">10": 3} | {"at least one": 28, "both": 0, "neither": 3} |
| aligned_dense_s0 | calibration | broad | native | {"0-0": 1, "1-0": 2, "2-0": 3, "2-2": 3, "3-2": 2, "4-2": 4, "4-3": 2, "5-0": 21, "5-3": 1, "5-5": 1, "6-0": 2, "6-1": 20, "6-4": 3, "6-6": 1, "7-0": 6, "7-2": 21, "7-5": 4, "7-6": 4} | {"-1": 1, "-11": 5, "-13": 3, "-14": 2, "-15": 2, "-2": 1, "-3": 3, "-4": 26, "-5": 1, "-6": 23, "-8": 25, "-9": 6, "0": 5, "1": 2, "10": 3, "12": 6, "2": 29, "3": 1, "4": 23, "5": 2, "6": 25, "7": 6, "8": 2} | {"000/000": 93, "000/010": 3, "001/000": 1, "010/000": 2, "100/000": 2} | {"10": 68, "15": 11, "19": 1, "20": 3, "5": 16, "9": 2} | {"none": 101} | {"at least one": 6, "both": 0, "neither": 95} |
| aligned_dense_s0 | calibration | broad | pairs | {"1-1": 1, "2-0": 10, "3-0": 2, "3-1": 8, "4-0": 14, "4-2": 12, "5-0": 44, "5-1": 10, "5-3": 8, "6-0": 12, "6-1": 44, "6-2": 12, "6-3": 2, "6-4": 4, "7-0": 30, "7-1": 8, "7-2": 43, "7-3": 10, "7-4": 2, "7-5": 9, "7-7": 1} | {"-1": 2, "-10": 11, "-11": 9, "-12": 5, "-13": 8, "-14": 6, "-15": 12, "-2": 12, "-3": 5, "-4": 61, "-5": 5, "-6": 52, "-7": 5, "-8": 82, "-9": 14, "0": 19, "1": 7, "10": 6, "12": 20, "2": 61, "3": 5, "4": 64, "5": 6, "6": 67, "7": 23, "8": 5} | {"000/000": 258, "000/010": 4, "000/100": 1, "001/000": 2, "010/000": 15, "100/000": 6} | {"10": 175, "13": 3, "15": 50, "19": 7, "20": 22, "25": 1, "5": 28} | {"0-5": 253, "6-10": 8, ">10": 2, "none": 23} | {"at least one": 21, "both": 0, "neither": 265} |
| aligned_dense_s0 | calibration | broad | multi | {"1-0": 4, "2-0": 16, "2-1": 6, "3-0": 6, "3-1": 13, "3-3": 1, "4-0": 8, "4-1": 4, "4-2": 10, "5-0": 375, "5-1": 4, "5-2": 2, "5-3": 14, "6-0": 8, "6-1": 297, "6-2": 2, "6-3": 4, "6-4": 15, "6-6": 1, "7-0": 67, "7-1": 10, "7-2": 312, "7-3": 6, "7-4": 2, "7-5": 13, "7-7": 1} | {"-1": 5, "-10": 8, "-11": 12, "-12": 5, "-13": 16, "-14": 12, "-15": 13, "-2": 24, "-3": 5, "-4": 328, "-5": 2, "-6": 309, "-7": 3, "-8": 433, "-9": 31, "0": 31, "1": 3, "10": 9, "12": 31, "2": 399, "3": 7, "4": 317, "5": 8, "6": 351, "7": 33, "8": 7} | {"000/000": 1162, "000/001": 1, "000/010": 15, "001/000": 4, "010/000": 16, "100/000": 3} | {"10": 1030, "13": 9, "15": 67, "19": 9, "20": 33, "5": 48, "9": 5} | {"0-5": 1105, "6-10": 39, ">10": 57} | {"at least one": 36, "both": 0, "neither": 1165} |
| aligned_dense_s1 | calibration | main | native | {"1-1": 1, "2-1": 2, "2-2": 4, "3-1": 4, "3-3": 1, "4-2": 3, "4-3": 2, "4-4": 1, "6-1": 20, "6-6": 2, "7-2": 29, "7-5": 5, "7-6": 2, "7-7": 2} | {"-1": 3, "-11": 1, "-13": 1, "-2": 1, "-3": 6, "-4": 32, "-5": 1, "-6": 25, "-7": 1, "-8": 1, "-9": 3, "0": 2, "1": 4, "12": 5, "2": 11, "4": 23, "6": 32, "7": 4} | {"000/000": 69, "000/001": 1, "000/100": 2, "001/000": 2, "010/000": 2, "100/000": 2} | {"10": 56, "15": 1, "5": 20, "9": 1} | {"none": 78} | {"at least one": 5, "both": 0, "neither": 73} |
| aligned_dense_s1 | calibration | main | pairs | {"2-2": 1, "3-1": 10, "4-2": 8, "5-1": 10, "5-3": 8, "6-1": 45, "6-2": 12, "6-3": 2, "6-4": 8, "7-1": 12, "7-2": 44, "7-3": 6, "7-4": 4, "7-5": 9, "7-7": 2} | {"-1": 4, "-10": 4, "-11": 5, "-12": 5, "-13": 9, "-2": 12, "-3": 2, "-4": 53, "-5": 4, "-6": 54, "-7": 7, "-8": 6, "-9": 9, "0": 10, "1": 3, "10": 5, "12": 7, "2": 16, "3": 6, "4": 59, "5": 5, "6": 66, "7": 6, "8": 5} | {"000/000": 164, "000/001": 2, "000/010": 6, "000/100": 1, "001/000": 2, "010/000": 4, "100/000": 2} | {"10": 124, "13": 3, "15": 21, "19": 4, "20": 4, "25": 1, "4": 1, "5": 23} | {"0-5": 152, "6-10": 5, ">10": 3, "none": 21} | {"at least one": 14, "both": 0, "neither": 167} |
| aligned_dense_s1 | calibration | main | multi | {"1-1": 1, "2-1": 4, "3-1": 11, "3-3": 1, "4-1": 2, "4-2": 10, "4-4": 1, "5-1": 4, "5-2": 2, "5-3": 11, "5-5": 1, "6-1": 319, "6-2": 2, "6-3": 2, "6-4": 19, "6-6": 3, "7-1": 6, "7-2": 329, "7-3": 6, "7-4": 2, "7-5": 15, "7-7": 1} | {"-1": 4, "-10": 4, "-11": 6, "-12": 3, "-13": 8, "-2": 15, "-4": 337, "-5": 3, "-6": 325, "-7": 4, "-8": 3, "-9": 15, "0": 27, "1": 2, "10": 11, "12": 9, "2": 21, "3": 6, "4": 327, "5": 14, "6": 345, "7": 8, "8": 7} | {"000/000": 719, "000/010": 19, "001/000": 4, "010/000": 8, "100/000": 2} | {"10": 682, "13": 4, "15": 18, "19": 3, "5": 43, "9": 2} | {"0-5": 690, "6-10": 20, ">10": 42} | {"at least one": 31, "both": 0, "neither": 721} |
| aligned_dense_s1 | calibration | main, equal marker | native | {"1-1": 1, "2-1": 2, "2-2": 4, "3-1": 2, "3-3": 1, "4-2": 1, "6-1": 20, "6-6": 1, "7-2": 29, "7-5": 5, "7-6": 2, "7-7": 1} | {"-11": 1, "-13": 1, "-2": 1, "-3": 6, "-4": 30, "-6": 23, "-7": 1, "-8": 1, "-9": 2, "12": 5, "2": 11, "4": 22, "6": 30, "7": 4} | {"000/000": 69} | {"10": 56, "15": 1, "5": 12} | {"none": 69} | {"at least one": 0, "both": 0, "neither": 69} |
| aligned_dense_s1 | calibration | main, equal marker | pairs | {"2-2": 1, "3-1": 8, "4-2": 2, "5-1": 10, "5-3": 8, "6-1": 45, "6-2": 12, "6-3": 2, "6-4": 1, "7-1": 12, "7-2": 44, "7-3": 6, "7-4": 4, "7-5": 8, "7-7": 1} | {"-10": 2, "-11": 5, "-12": 5, "-13": 9, "-2": 12, "-3": 2, "-4": 51, "-5": 2, "-6": 52, "-7": 7, "-8": 6, "-9": 9, "10": 2, "12": 7, "2": 16, "3": 5, "4": 58, "5": 2, "6": 65, "7": 6, "8": 5} | {"000/000": 164} | {"10": 119, "13": 3, "15": 21, "19": 4, "20": 4, "25": 1, "5": 12} | {"0-5": 144, "6-10": 5, ">10": 3, "none": 12} | {"at least one": 0, "both": 0, "neither": 164} |
| aligned_dense_s1 | calibration | main, equal marker | multi | {"1-1": 1, "2-1": 4, "3-1": 7, "3-3": 1, "4-2": 2, "4-4": 1, "5-1": 4, "5-2": 2, "5-3": 11, "5-5": 1, "6-1": 319, "6-2": 2, "6-3": 2, "6-6": 3, "7-1": 6, "7-2": 329, "7-3": 6, "7-4": 2, "7-5": 15, "7-7": 1} | {"-11": 6, "-12": 3, "-13": 7, "-2": 15, "-4": 335, "-6": 321, "-7": 4, "-8": 3, "-9": 15, "10": 3, "12": 9, "2": 21, "3": 6, "4": 327, "5": 3, "6": 345, "7": 8, "8": 7} | {"000/000": 719} | {"10": 670, "13": 3, "15": 18, "19": 3, "5": 23, "9": 2} | {"0-5": 661, "6-10": 20, ">10": 38} | {"at least one": 0, "both": 0, "neither": 719} |
| aligned_dense_s1 | calibration | main, changed marker | native | {"3-1": 2, "4-2": 2, "4-3": 2, "4-4": 1, "6-6": 1, "7-7": 1} | {"-1": 3, "-4": 2, "-5": 1, "-6": 2, "-9": 1, "0": 2, "1": 4, "4": 1, "6": 2} | {"000/001": 1, "000/100": 2, "001/000": 2, "010/000": 2, "100/000": 2} | {"5": 8, "9": 1} | {"none": 9} | {"at least one": 5, "both": 0, "neither": 4} |
| aligned_dense_s1 | calibration | main, changed marker | pairs | {"3-1": 2, "4-2": 6, "6-4": 7, "7-5": 1, "7-7": 1} | {"-1": 4, "-10": 2, "-4": 2, "-5": 2, "-6": 2, "0": 10, "1": 3, "10": 3, "3": 1, "4": 1, "5": 3, "6": 1} | {"000/001": 2, "000/010": 6, "000/100": 1, "001/000": 2, "010/000": 4, "100/000": 2} | {"10": 5, "4": 1, "5": 11} | {"0-5": 8, "none": 9} | {"at least one": 14, "both": 0, "neither": 3} |
| aligned_dense_s1 | calibration | main, changed marker | multi | {"3-1": 4, "4-1": 2, "4-2": 8, "6-4": 19} | {"-1": 4, "-10": 4, "-13": 1, "-4": 2, "-5": 3, "-6": 4, "0": 27, "1": 2, "10": 8, "5": 11} | {"000/010": 19, "001/000": 4, "010/000": 8, "100/000": 2} | {"10": 12, "13": 1, "5": 20} | {"0-5": 29, ">10": 4} | {"at least one": 31, "both": 0, "neither": 2} |
| aligned_dense_s1 | calibration | broad | native | {"1-0": 6, "1-1": 1, "2-0": 4, "2-1": 2, "2-2": 4, "3-1": 4, "3-3": 1, "4-2": 3, "4-3": 2, "4-4": 1, "5-0": 44, "6-0": 2, "6-1": 20, "6-6": 2, "7-0": 7, "7-2": 29, "7-5": 5, "7-6": 2, "7-7": 2} | {"-1": 3, "-10": 2, "-11": 3, "-13": 3, "-14": 1, "-15": 4, "-2": 1, "-3": 7, "-4": 32, "-5": 3, "-6": 27, "-7": 1, "-8": 52, "-9": 4, "0": 4, "1": 4, "12": 8, "2": 58, "4": 25, "6": 34, "7": 6} | {"000/000": 130, "000/001": 1, "000/100": 2, "001/000": 2, "010/000": 4, "100/000": 2} | {"10": 103, "15": 8, "19": 1, "20": 4, "5": 23, "9": 2} | {"none": 141} | {"at least one": 7, "both": 0, "neither": 134} |
| aligned_dense_s1 | calibration | broad | pairs | {"0-0": 1, "1-0": 2, "2-0": 7, "2-2": 1, "3-0": 2, "3-1": 10, "4-0": 8, "4-2": 8, "5-0": 47, "5-1": 10, "5-3": 8, "6-0": 14, "6-1": 45, "6-2": 12, "6-3": 2, "6-4": 8, "7-0": 36, "7-1": 12, "7-2": 44, "7-3": 6, "7-4": 4, "7-5": 9, "7-7": 2} | {"-1": 4, "-10": 10, "-11": 11, "-12": 7, "-13": 10, "-14": 2, "-15": 14, "-2": 14, "-3": 4, "-4": 55, "-5": 4, "-6": 56, "-7": 7, "-8": 91, "-9": 11, "0": 18, "1": 3, "10": 7, "12": 24, "2": 66, "3": 6, "4": 69, "5": 7, "6": 68, "7": 23, "8": 5} | {"000/000": 273, "000/001": 2, "000/010": 6, "000/100": 1, "001/000": 2, "010/000": 12, "100/000": 2} | {"10": 179, "13": 4, "15": 50, "19": 9, "20": 24, "25": 2, "3": 1, "4": 1, "5": 27, "9": 1} | {"0-5": 254, "6-10": 9, ">10": 4, "none": 31} | {"at least one": 22, "both": 0, "neither": 276} |
| aligned_dense_s1 | calibration | broad | multi | {"1-0": 6, "1-1": 1, "2-0": 16, "2-1": 4, "3-0": 6, "3-1": 11, "3-3": 1, "4-0": 6, "4-1": 2, "4-2": 10, "4-4": 1, "5-0": 383, "5-1": 4, "5-2": 2, "5-3": 11, "5-5": 1, "6-0": 10, "6-1": 319, "6-2": 2, "6-3": 2, "6-4": 19, "6-6": 3, "7-0": 75, "7-1": 6, "7-2": 329, "7-3": 6, "7-4": 2, "7-5": 15, "7-7": 1} | {"-1": 4, "-10": 7, "-11": 14, "-12": 6, "-13": 13, "-14": 16, "-15": 14, "-2": 21, "-3": 3, "-4": 347, "-5": 3, "-6": 331, "-7": 4, "-8": 440, "-9": 31, "0": 33, "1": 2, "10": 11, "12": 36, "2": 407, "3": 6, "4": 337, "5": 14, "6": 367, "7": 34, "8": 7} | {"000/000": 1215, "000/010": 19, "001/000": 4, "010/000": 14, "100/000": 2} | {"10": 1074, "13": 7, "15": 68, "19": 8, "20": 38, "5": 54, "9": 5} | {"0-5": 1145, "6-10": 48, ">10": 61} | {"at least one": 37, "both": 0, "neither": 1217} |
| aligned_dense_s2 | calibration | main | native | {"1-1": 1, "3-1": 4, "3-2": 2, "3-3": 1, "4-2": 2, "4-3": 2, "5-3": 2, "5-5": 1, "6-1": 20, "6-4": 2, "6-5": 2, "7-2": 26, "7-5": 6} | {"-1": 5, "-11": 2, "-2": 2, "-4": 28, "-5": 3, "-6": 24, "-9": 1, "0": 8, "1": 1, "10": 2, "12": 3, "2": 6, "3": 1, "4": 23, "5": 2, "6": 27, "7": 3, "8": 1} | {"000/000": 57, "000/001": 1, "000/010": 4, "001/000": 4, "010/000": 4, "100/000": 1} | {"10": 55, "15": 1, "5": 14, "9": 1} | {"none": 71} | {"at least one": 13, "both": 0, "neither": 58} |
| aligned_dense_s2 | calibration | main | pairs | {"3-1": 8, "3-3": 2, "4-2": 6, "5-1": 10, "5-3": 4, "6-1": 37, "6-2": 8, "6-3": 2, "6-4": 11, "6-6": 1, "7-1": 6, "7-2": 40, "7-3": 10, "7-4": 2, "7-5": 8} | {"-1": 3, "-10": 3, "-11": 3, "-12": 5, "-13": 6, "-2": 10, "-3": 2, "-4": 48, "-5": 3, "-6": 42, "-7": 6, "-8": 5, "-9": 8, "0": 11, "1": 3, "10": 7, "12": 5, "2": 12, "3": 4, "4": 49, "5": 8, "6": 58, "7": 5, "8": 4} | {"000/000": 138, "000/001": 3, "000/010": 9, "000/100": 1, "010/000": 2, "100/000": 2} | {"10": 107, "13": 2, "15": 19, "19": 3, "20": 3, "5": 21} | {"0-5": 131, "6-10": 6, ">10": 2, "none": 16} | {"at least one": 14, "both": 0, "neither": 141} |
| aligned_dense_s2 | calibration | main | multi | {"1-1": 2, "2-1": 4, "2-2": 1, "3-1": 9, "4-1": 3, "4-2": 10, "5-1": 4, "5-2": 2, "5-3": 8, "6-1": 238, "6-2": 2, "6-3": 2, "6-4": 6, "7-1": 6, "7-2": 274, "7-3": 6, "7-5": 8, "7-7": 3} | {"-1": 4, "-10": 5, "-11": 5, "-12": 2, "-13": 9, "-2": 11, "-3": 1, "-4": 284, "-5": 2, "-6": 243, "-7": 3, "-8": 3, "-9": 12, "0": 13, "1": 3, "10": 3, "12": 6, "2": 15, "3": 4, "4": 246, "5": 3, "6": 289, "7": 6, "8": 4} | {"000/000": 568, "000/001": 1, "000/010": 6, "001/000": 3, "010/000": 7, "100/000": 3} | {"10": 536, "13": 3, "15": 15, "19": 3, "3": 1, "5": 28, "9": 2} | {"0-5": 539, "6-10": 12, ">10": 37} | {"at least one": 17, "both": 0, "neither": 571} |
| aligned_dense_s2 | calibration | main, equal marker | native | {"3-1": 2, "4-2": 1, "5-3": 2, "6-1": 20, "7-2": 26, "7-5": 6} | {"-11": 1, "-2": 2, "-4": 27, "-6": 21, "12": 3, "2": 6, "3": 1, "4": 22, "6": 27, "7": 3, "8": 1} | {"000/000": 57} | {"10": 52, "15": 1, "5": 4} | {"none": 57} | {"at least one": 0, "both": 0, "neither": 57} |
| aligned_dense_s2 | calibration | main, equal marker | pairs | {"3-1": 8, "4-2": 2, "5-1": 10, "5-3": 4, "6-1": 37, "6-2": 8, "6-3": 2, "6-4": 1, "6-6": 1, "7-1": 6, "7-2": 40, "7-3": 10, "7-4": 2, "7-5": 7} | {"-10": 2, "-11": 3, "-12": 5, "-13": 6, "-2": 10, "-3": 2, "-4": 46, "-5": 2, "-6": 42, "-7": 6, "-8": 5, "-9": 8, "10": 3, "12": 5, "2": 12, "3": 4, "4": 46, "5": 3, "6": 57, "7": 5, "8": 4} | {"000/000": 138} | {"10": 102, "13": 2, "15": 19, "19": 3, "20": 3, "5": 9} | {"0-5": 121, "6-10": 5, ">10": 2, "none": 10} | {"at least one": 0, "both": 0, "neither": 138} |
| aligned_dense_s2 | calibration | main, equal marker | multi | {"1-1": 2, "2-1": 4, "2-2": 1, "3-1": 6, "4-2": 3, "5-1": 4, "5-2": 2, "5-3": 8, "6-1": 238, "6-2": 2, "6-3": 2, "7-1": 6, "7-2": 274, "7-3": 6, "7-5": 8, "7-7": 2} | {"-10": 1, "-11": 5, "-12": 2, "-13": 8, "-2": 11, "-3": 1, "-4": 281, "-6": 240, "-7": 3, "-8": 3, "-9": 12, "12": 6, "2": 15, "3": 4, "4": 245, "6": 289, "7": 6, "8": 4} | {"000/000": 568} | {"10": 529, "13": 2, "15": 15, "19": 3, "3": 1, "5": 16, "9": 2} | {"0-5": 522, "6-10": 11, ">10": 35} | {"at least one": 0, "both": 0, "neither": 568} |
| aligned_dense_s2 | calibration | main, changed marker | native | {"1-1": 1, "3-1": 2, "3-2": 2, "3-3": 1, "4-2": 1, "4-3": 2, "5-5": 1, "6-4": 2, "6-5": 2} | {"-1": 5, "-11": 1, "-4": 1, "-5": 3, "-6": 3, "-9": 1, "0": 8, "1": 1, "10": 2, "4": 1, "5": 2} | {"000/001": 1, "000/010": 4, "001/000": 4, "010/000": 4, "100/000": 1} | {"10": 3, "5": 10, "9": 1} | {"none": 14} | {"at least one": 13, "both": 0, "neither": 1} |
| aligned_dense_s2 | calibration | main, changed marker | pairs | {"3-3": 2, "4-2": 4, "6-4": 10, "7-5": 1} | {"-1": 3, "-10": 1, "-4": 2, "-5": 1, "0": 11, "1": 3, "10": 4, "4": 3, "5": 5, "6": 1} | {"000/001": 3, "000/010": 9, "000/100": 1, "010/000": 2, "100/000": 2} | {"10": 5, "5": 12} | {"0-5": 10, "6-10": 1, "none": 6} | {"at least one": 14, "both": 0, "neither": 3} |
| aligned_dense_s2 | calibration | main, changed marker | multi | {"3-1": 3, "4-1": 3, "4-2": 7, "6-4": 6, "7-7": 1} | {"-1": 4, "-10": 4, "-13": 1, "-4": 3, "-5": 2, "-6": 3, "0": 13, "1": 3, "10": 3, "4": 1, "5": 3} | {"000/001": 1, "000/010": 6, "001/000": 3, "010/000": 7, "100/000": 3} | {"10": 7, "13": 1, "5": 12} | {"0-5": 17, "6-10": 1, ">10": 2} | {"at least one": 17, "both": 0, "neither": 3} |
| aligned_dense_s2 | calibration | broad | native | {"0-0": 1, "1-0": 2, "1-1": 1, "2-0": 6, "3-1": 4, "3-2": 2, "3-3": 1, "4-2": 2, "4-3": 2, "5-0": 29, "5-3": 2, "5-5": 1, "6-0": 2, "6-1": 20, "6-4": 2, "6-5": 2, "7-0": 8, "7-2": 26, "7-5": 6} | {"-1": 5, "-11": 4, "-13": 1, "-14": 1, "-15": 2, "-2": 3, "-3": 3, "-4": 28, "-5": 3, "-6": 26, "-7": 1, "-8": 40, "-9": 2, "0": 8, "1": 1, "10": 2, "12": 6, "2": 38, "3": 1, "4": 25, "5": 2, "6": 29, "7": 6, "8": 1} | {"000/000": 105, "000/001": 1, "000/010": 4, "001/000": 4, "010/000": 4, "100/000": 1} | {"10": 86, "15": 7, "19": 1, "20": 4, "5": 19, "9": 2} | {"none": 119} | {"at least one": 13, "both": 0, "neither": 106} |
| aligned_dense_s2 | calibration | broad | pairs | {"0-0": 2, "2-0": 10, "3-0": 2, "3-1": 8, "3-3": 2, "4-0": 12, "4-2": 6, "5-0": 42, "5-1": 10, "5-3": 4, "6-0": 10, "6-1": 37, "6-2": 8, "6-3": 2, "6-4": 11, "6-6": 1, "7-0": 30, "7-1": 6, "7-2": 40, "7-3": 10, "7-4": 2, "7-5": 8} | {"-1": 3, "-10": 10, "-11": 6, "-12": 6, "-13": 7, "-14": 5, "-15": 11, "-2": 12, "-3": 5, "-4": 52, "-5": 3, "-6": 42, "-7": 6, "-8": 80, "-9": 15, "0": 21, "1": 4, "10": 9, "12": 19, "2": 57, "3": 4, "4": 55, "5": 10, "6": 61, "7": 19, "8": 4} | {"000/000": 235, "000/001": 3, "000/010": 9, "000/100": 1, "010/000": 12, "100/000": 3} | {"10": 160, "13": 3, "15": 45, "19": 6, "20": 20, "25": 1, "5": 28} | {"0-5": 231, "6-10": 10, ">10": 2, "none": 20} | {"at least one": 24, "both": 0, "neither": 239} |
| aligned_dense_s2 | calibration | broad | multi | {"0-0": 1, "1-1": 2, "2-0": 16, "2-1": 4, "2-2": 1, "3-0": 6, "3-1": 9, "4-0": 4, "4-1": 3, "4-2": 10, "5-0": 319, "5-1": 4, "5-2": 2, "5-3": 8, "6-0": 4, "6-1": 238, "6-2": 2, "6-3": 2, "6-4": 6, "7-0": 59, "7-1": 6, "7-2": 274, "7-3": 6, "7-5": 8, "7-7": 3} | {"-1": 4, "-10": 7, "-11": 7, "-12": 5, "-13": 14, "-14": 12, "-15": 7, "-2": 17, "-3": 5, "-4": 291, "-5": 2, "-6": 243, "-7": 3, "-8": 368, "-9": 26, "0": 17, "1": 3, "10": 3, "12": 27, "2": 339, "3": 4, "4": 250, "5": 3, "6": 306, "7": 27, "8": 4} | {"000/000": 973, "000/001": 1, "000/010": 6, "001/000": 3, "010/000": 11, "100/000": 3} | {"10": 863, "13": 6, "15": 54, "19": 5, "20": 29, "3": 1, "5": 37, "9": 2} | {"0-5": 920, "6-10": 28, ">10": 49} | {"at least one": 21, "both": 0, "neither": 976} |
| center_only_s0 | calibration | main | native | {"1-1": 1, "3-1": 4, "3-2": 2, "3-3": 2, "4-2": 2, "4-4": 1, "5-3": 2, "5-4": 2, "5-5": 1, "6-1": 46, "6-4": 3, "6-6": 3, "7-1": 2, "7-2": 48, "7-3": 2, "7-5": 3, "7-6": 2, "7-7": 2} | {"-1": 6, "-10": 1, "-11": 3, "-12": 1, "-13": 1, "-2": 3, "-3": 5, "-4": 48, "-5": 3, "-6": 49, "-7": 3, "-8": 1, "-9": 4, "0": 6, "1": 1, "10": 1, "12": 2, "2": 6, "4": 54, "5": 1, "6": 54, "7": 3} | {"000/000": 115, "000/001": 4, "000/010": 2, "000/100": 1, "001/000": 2, "010/000": 4} | {"10": 100, "15": 8, "19": 1, "5": 18, "9": 1} | {"none": 128} | {"at least one": 12, "both": 0, "neither": 116} |
| center_only_s0 | calibration | main | pairs | {"1-1": 2, "2-1": 2, "2-2": 1, "3-1": 16, "3-3": 1, "4-1": 4, "4-2": 16, "4-4": 1, "5-1": 18, "5-2": 4, "5-3": 16, "5-5": 1, "6-1": 78, "6-2": 18, "6-3": 3, "6-4": 13, "6-6": 1, "7-1": 12, "7-2": 78, "7-3": 18, "7-4": 2, "7-5": 14} | {"-1": 8, "-10": 10, "-11": 12, "-12": 5, "-13": 19, "-2": 17, "-3": 7, "-4": 96, "-5": 7, "-6": 97, "-7": 6, "-8": 18, "-9": 15, "0": 20, "1": 8, "10": 8, "12": 10, "2": 33, "3": 7, "4": 101, "5": 8, "6": 109, "7": 10, "8": 7} | {"000/000": 283, "000/001": 2, "000/010": 8, "000/100": 3, "001/000": 6, "010/000": 12, "100/000": 5} | {"10": 215, "13": 6, "15": 36, "18": 2, "19": 4, "20": 7, "23": 2, "25": 1, "3": 4, "5": 41, "9": 1} | {"0-5": 276, "6-10": 21, ">10": 2, "none": 20} | {"at least one": 28, "both": 0, "neither": 291} |
| center_only_s0 | calibration | main | multi | {"1-1": 3, "2-2": 3, "3-1": 39, "3-3": 2, "4-1": 6, "4-2": 39, "4-4": 2, "5-1": 11, "5-2": 6, "5-3": 36, "5-5": 1, "6-1": 772, "6-2": 8, "6-3": 6, "6-4": 40, "6-6": 2, "7-1": 14, "7-2": 785, "7-3": 10, "7-4": 6, "7-5": 40, "7-7": 2} | {"-1": 23, "-10": 6, "-11": 16, "-12": 2, "-13": 16, "-2": 37, "-3": 4, "-4": 821, "-5": 6, "-6": 812, "-7": 6, "-8": 12, "-9": 24, "0": 49, "1": 19, "10": 17, "12": 19, "2": 59, "3": 16, "4": 813, "5": 17, "6": 837, "7": 19, "8": 16} | {"000/000": 1742, "000/001": 6, "000/010": 34, "000/100": 3, "001/000": 17, "010/000": 15, "100/000": 16} | {"10": 1667, "13": 12, "15": 36, "19": 7, "5": 111} | {"0-5": 1611, "6-10": 164, ">10": 58} | {"at least one": 72, "both": 0, "neither": 1761} |
| center_only_s0 | calibration | main, equal marker | native | {"3-1": 3, "3-2": 2, "3-3": 1, "5-3": 2, "5-4": 2, "6-1": 46, "6-4": 1, "6-6": 1, "7-1": 2, "7-2": 48, "7-3": 2, "7-5": 2, "7-6": 2, "7-7": 1} | {"-11": 3, "-12": 1, "-13": 1, "-2": 3, "-3": 5, "-4": 48, "-6": 47, "-7": 3, "-8": 1, "-9": 4, "12": 2, "2": 6, "4": 50, "6": 53, "7": 3} | {"000/000": 115} | {"10": 98, "15": 8, "19": 1, "5": 7, "9": 1} | {"none": 115} | {"at least one": 0, "both": 0, "neither": 115} |
| center_only_s0 | calibration | main, equal marker | pairs | {"1-1": 1, "2-1": 2, "2-2": 1, "3-1": 11, "3-3": 1, "4-2": 5, "5-1": 18, "5-2": 4, "5-3": 16, "5-5": 1, "6-1": 78, "6-2": 18, "6-3": 3, "6-4": 3, "7-1": 12, "7-2": 78, "7-3": 18, "7-4": 2, "7-5": 11} | {"-10": 5, "-11": 12, "-12": 5, "-13": 17, "-2": 17, "-3": 6, "-4": 91, "-5": 3, "-6": 91, "-7": 6, "-8": 18, "-9": 15, "10": 4, "12": 10, "2": 33, "3": 7, "4": 99, "5": 4, "6": 106, "7": 10, "8": 7} | {"000/000": 283} | {"10": 206, "13": 4, "15": 36, "18": 2, "19": 4, "20": 7, "23": 2, "25": 1, "3": 3, "5": 17, "9": 1} | {"0-5": 251, "6-10": 18, ">10": 2, "none": 12} | {"at least one": 0, "both": 0, "neither": 283} |
| center_only_s0 | calibration | main, equal marker | multi | {"1-1": 1, "2-2": 3, "3-1": 22, "3-3": 1, "4-2": 17, "4-4": 1, "5-1": 11, "5-2": 6, "5-3": 36, "6-1": 772, "6-2": 8, "6-3": 6, "6-4": 3, "6-6": 1, "7-1": 14, "7-2": 785, "7-3": 10, "7-4": 6, "7-5": 38, "7-7": 1} | {"-11": 16, "-12": 2, "-13": 13, "-2": 37, "-3": 4, "-4": 805, "-6": 795, "-7": 6, "-8": 12, "-9": 24, "12": 19, "2": 59, "3": 16, "4": 807, "6": 834, "7": 19, "8": 16} | {"000/000": 1742} | {"10": 1644, "13": 9, "15": 36, "19": 7, "5": 46} | {"0-5": 1534, "6-10": 155, ">10": 53} | {"at least one": 0, "both": 0, "neither": 1742} |
| center_only_s0 | calibration | main, changed marker | native | {"1-1": 1, "3-1": 1, "3-3": 1, "4-2": 2, "4-4": 1, "5-5": 1, "6-4": 2, "6-6": 2, "7-5": 1, "7-7": 1} | {"-1": 6, "-10": 1, "-5": 3, "-6": 2, "0": 6, "1": 1, "10": 1, "4": 4, "5": 1, "6": 1} | {"000/001": 4, "000/010": 2, "000/100": 1, "001/000": 2, "010/000": 4} | {"10": 2, "5": 11} | {"none": 13} | {"at least one": 12, "both": 0, "neither": 1} |
| center_only_s0 | calibration | main, changed marker | pairs | {"1-1": 1, "3-1": 5, "4-1": 4, "4-2": 11, "4-4": 1, "6-4": 10, "6-6": 1, "7-5": 3} | {"-1": 8, "-10": 5, "-13": 2, "-3": 1, "-4": 5, "-5": 4, "-6": 6, "0": 20, "1": 8, "10": 4, "4": 2, "5": 4, "6": 3} | {"000/001": 2, "000/010": 8, "000/100": 3, "001/000": 6, "010/000": 12, "100/000": 5} | {"10": 9, "13": 2, "3": 1, "5": 24} | {"0-5": 25, "6-10": 3, "none": 8} | {"at least one": 28, "both": 0, "neither": 8} |
| center_only_s0 | calibration | main, changed marker | multi | {"1-1": 2, "3-1": 17, "3-3": 1, "4-1": 6, "4-2": 22, "4-4": 1, "5-5": 1, "6-4": 37, "6-6": 1, "7-5": 2, "7-7": 1} | {"-1": 23, "-10": 6, "-13": 3, "-4": 16, "-5": 6, "-6": 17, "0": 49, "1": 19, "10": 17, "4": 6, "5": 17, "6": 3} | {"000/001": 6, "000/010": 34, "000/100": 3, "001/000": 17, "010/000": 15, "100/000": 16} | {"10": 23, "13": 3, "5": 65} | {"0-5": 77, "6-10": 9, ">10": 5} | {"at least one": 72, "both": 0, "neither": 19} |
| center_only_s0 | calibration | broad | native | {"0-0": 4, "1-1": 1, "2-0": 4, "3-1": 4, "3-2": 2, "3-3": 2, "4-0": 1, "4-2": 2, "4-4": 1, "5-0": 42, "5-3": 2, "5-4": 2, "5-5": 1, "6-0": 2, "6-1": 46, "6-4": 3, "6-6": 3, "7-0": 4, "7-1": 2, "7-2": 48, "7-3": 2, "7-5": 3, "7-6": 2, "7-7": 2} | {"-1": 6, "-10": 2, "-11": 4, "-12": 1, "-13": 3, "-15": 3, "-2": 5, "-3": 7, "-4": 48, "-5": 3, "-6": 49, "-7": 5, "-8": 51, "-9": 4, "0": 7, "1": 1, "10": 1, "12": 4, "2": 50, "4": 56, "5": 1, "6": 54, "7": 5} | {"000/000": 171, "000/001": 4, "000/010": 2, "000/100": 1, "001/000": 2, "010/000": 5} | {"10": 143, "15": 13, "19": 2, "20": 2, "5": 24, "9": 1} | {"none": 185} | {"at least one": 13, "both": 0, "neither": 172} |
| center_only_s0 | calibration | broad | pairs | {"0-0": 2, "1-0": 2, "1-1": 2, "2-0": 17, "2-1": 2, "2-2": 1, "3-0": 4, "3-1": 16, "3-3": 1, "4-0": 17, "4-1": 4, "4-2": 16, "4-4": 1, "5-0": 80, "5-1": 18, "5-2": 4, "5-3": 16, "5-5": 1, "6-0": 12, "6-1": 78, "6-2": 18, "6-3": 3, "6-4": 13, "6-6": 1, "7-0": 56, "7-1": 12, "7-2": 78, "7-3": 18, "7-4": 2, "7-5": 14} | {"-1": 8, "-10": 20, "-11": 17, "-12": 9, "-13": 22, "-14": 7, "-15": 18, "-2": 22, "-3": 12, "-4": 102, "-5": 7, "-6": 99, "-7": 7, "-8": 154, "-9": 22, "0": 33, "1": 10, "10": 10, "12": 36, "2": 115, "3": 9, "4": 109, "5": 10, "6": 115, "7": 36, "8": 9} | {"000/000": 458, "000/001": 2, "000/010": 8, "000/100": 3, "001/000": 6, "010/000": 25, "100/000": 7} | {"10": 309, "13": 8, "15": 78, "18": 3, "19": 8, "20": 39, "23": 3, "25": 3, "3": 4, "5": 52, "9": 2} | {"0-5": 433, "6-10": 49, ">10": 3, "none": 24} | {"at least one": 41, "both": 0, "neither": 468} |
| center_only_s0 | calibration | broad | multi | {"0-0": 1, "1-1": 3, "2-0": 41, "2-2": 3, "3-0": 6, "3-1": 39, "3-3": 2, "4-0": 12, "4-1": 6, "4-2": 39, "4-4": 2, "5-0": 786, "5-1": 11, "5-2": 6, "5-3": 36, "5-5": 1, "6-0": 14, "6-1": 772, "6-2": 8, "6-3": 6, "6-4": 40, "6-6": 2, "7-0": 120, "7-1": 14, "7-2": 785, "7-3": 10, "7-4": 6, "7-5": 40, "7-7": 2} | {"-1": 23, "-10": 12, "-11": 23, "-12": 5, "-13": 24, "-14": 15, "-15": 16, "-2": 44, "-3": 22, "-4": 825, "-5": 6, "-6": 812, "-7": 7, "-8": 921, "-9": 39, "0": 61, "1": 19, "10": 17, "12": 66, "2": 864, "3": 16, "4": 827, "5": 17, "6": 863, "7": 66, "8": 16} | {"000/000": 2710, "000/001": 6, "000/010": 34, "000/100": 3, "001/000": 17, "010/000": 27, "100/000": 16} | {"10": 2475, "13": 15, "15": 117, "19": 14, "20": 60, "5": 132} | {"0-5": 2384, "6-10": 340, ">10": 89} | {"at least one": 84, "both": 0, "neither": 2729} |
| center_only_s1 | calibration | main | native | {"1-1": 3, "2-1": 9, "2-2": 7, "3-1": 7, "3-2": 10, "3-3": 6, "4-1": 2, "4-2": 7, "4-3": 6, "4-4": 9, "5-1": 8, "5-3": 9, "5-4": 6, "5-5": 6, "6-1": 72, "6-2": 8, "6-4": 16, "6-5": 4, "6-6": 8, "7-1": 4, "7-2": 70, "7-3": 5, "7-4": 2, "7-5": 18, "7-6": 8, "7-7": 2} | {"-1": 10, "-10": 2, "-11": 9, "-12": 2, "-13": 12, "-2": 20, "-3": 21, "-4": 86, "-5": 13, "-6": 81, "-7": 13, "-8": 4, "-9": 16, "0": 30, "1": 12, "10": 14, "12": 14, "2": 39, "3": 9, "4": 88, "5": 11, "6": 94, "7": 12, "8": 12} | {"000/000": 260, "000/001": 6, "000/010": 18, "000/100": 7, "001/000": 4, "010/000": 12, "100/000": 5} | {"10": 183, "14": 1, "15": 20, "19": 3, "20": 2, "3": 1, "4": 3, "5": 78, "9": 21} | {"none": 312} | {"at least one": 40, "both": 0, "neither": 272} |
| center_only_s1 | calibration | main | pairs | {"1-1": 4, "2-2": 7, "3-1": 23, "3-3": 4, "4-2": 21, "4-4": 5, "5-1": 17, "5-2": 3, "5-3": 25, "5-5": 2, "6-1": 88, "6-2": 15, "6-3": 3, "6-4": 27, "6-6": 2, "7-1": 16, "7-2": 91, "7-3": 18, "7-4": 3, "7-5": 27, "7-6": 2, "7-7": 4} | {"-1": 16, "-10": 3, "-11": 12, "-12": 3, "-13": 16, "-2": 27, "-3": 9, "-4": 121, "-5": 7, "-6": 118, "-7": 7, "-8": 14, "-9": 17, "0": 29, "1": 15, "10": 14, "12": 17, "2": 47, "3": 15, "4": 120, "5": 16, "6": 141, "7": 16, "8": 14} | {"000/000": 347, "000/001": 6, "000/010": 24, "000/100": 7, "001/000": 10, "010/000": 5, "100/000": 8} | {"10": 264, "13": 5, "15": 36, "19": 6, "20": 5, "25": 1, "3": 5, "5": 84, "9": 1} | {"0-5": 271, "6-10": 38, ">10": 31, "none": 67} | {"at least one": 45, "both": 0, "neither": 362} |
| center_only_s1 | calibration | main | multi | {"1-1": 8, "2-2": 6, "3-1": 50, "3-3": 9, "4-1": 9, "4-2": 48, "4-4": 10, "5-1": 14, "5-2": 14, "5-3": 48, "5-5": 2, "6-1": 795, "6-2": 13, "6-3": 13, "6-4": 54, "6-6": 7, "7-1": 12, "7-2": 810, "7-3": 10, "7-4": 9, "7-5": 51, "7-6": 2, "7-7": 1} | {"-1": 31, "-10": 10, "-11": 26, "-12": 3, "-13": 20, "-2": 52, "-3": 12, "-4": 859, "-5": 8, "-6": 851, "-7": 13, "-8": 18, "-9": 30, "0": 72, "1": 28, "10": 24, "12": 24, "2": 82, "3": 23, "4": 857, "5": 24, "6": 875, "7": 23, "8": 25} | {"000/000": 1864, "000/001": 10, "000/010": 48, "000/100": 8, "001/000": 21, "010/000": 24, "100/000": 20} | {"10": 1757, "13": 23, "15": 42, "19": 6, "3": 12, "5": 154, "9": 1} | {"0-5": 1596, "6-10": 208, ">10": 191} | {"at least one": 103, "both": 0, "neither": 1892} |
| center_only_s1 | calibration | main, equal marker | native | {"1-1": 1, "2-1": 9, "2-2": 7, "3-1": 4, "3-2": 10, "3-3": 3, "4-1": 1, "4-2": 3, "4-4": 3, "5-1": 8, "5-3": 9, "5-4": 6, "5-5": 4, "6-1": 72, "6-2": 8, "6-4": 1, "6-5": 2, "6-6": 4, "7-1": 4, "7-2": 70, "7-3": 5, "7-4": 2, "7-5": 18, "7-6": 4, "7-7": 2} | {"-10": 2, "-11": 9, "-12": 2, "-13": 11, "-2": 20, "-3": 20, "-4": 82, "-5": 5, "-6": 77, "-7": 13, "-8": 4, "-9": 13, "10": 5, "12": 14, "2": 39, "3": 9, "4": 82, "5": 4, "6": 86, "7": 12, "8": 11} | {"000/000": 260} | {"10": 174, "15": 20, "19": 3, "20": 2, "4": 3, "5": 40, "9": 18} | {"none": 260} | {"at least one": 0, "both": 0, "neither": 260} |
| center_only_s1 | calibration | main, equal marker | pairs | {"1-1": 1, "2-2": 7, "3-1": 13, "3-3": 1, "4-2": 11, "4-4": 2, "5-1": 17, "5-2": 3, "5-3": 25, "5-5": 2, "6-1": 88, "6-2": 15, "6-3": 3, "6-4": 2, "6-6": 2, "7-1": 16, "7-2": 91, "7-3": 18, "7-4": 3, "7-5": 25, "7-6": 1, "7-7": 1} | {"-10": 2, "-11": 12, "-12": 3, "-13": 16, "-2": 27, "-3": 9, "-4": 113, "-5": 3, "-6": 108, "-7": 7, "-8": 14, "-9": 17, "10": 3, "12": 17, "2": 46, "3": 15, "4": 115, "5": 3, "6": 134, "7": 16, "8": 14} | {"000/000": 347} | {"10": 252, "13": 5, "15": 36, "19": 6, "20": 5, "25": 1, "3": 4, "5": 37, "9": 1} | {"0-5": 241, "6-10": 29, ">10": 24, "none": 53} | {"at least one": 0, "both": 0, "neither": 347} |
| center_only_s1 | calibration | main, equal marker | multi | {"1-1": 4, "2-2": 6, "3-1": 29, "3-3": 3, "4-2": 22, "4-4": 5, "5-1": 14, "5-2": 14, "5-3": 48, "5-5": 2, "6-1": 795, "6-2": 13, "6-3": 13, "6-4": 3, "6-6": 2, "7-1": 12, "7-2": 810, "7-3": 10, "7-4": 9, "7-5": 48, "7-6": 1, "7-7": 1} | {"-10": 2, "-11": 26, "-12": 3, "-13": 16, "-2": 52, "-3": 8, "-4": 839, "-6": 830, "-7": 13, "-8": 18, "-9": 30, "12": 24, "2": 82, "3": 23, "4": 847, "6": 867, "7": 23, "8": 25} | {"000/000": 1864} | {"10": 1725, "13": 19, "15": 42, "19": 6, "3": 8, "5": 63, "9": 1} | {"0-5": 1516, "6-10": 192, ">10": 156} | {"at least one": 0, "both": 0, "neither": 1864} |
| center_only_s1 | calibration | main, changed marker | native | {"1-1": 2, "3-1": 3, "3-3": 3, "4-1": 1, "4-2": 4, "4-3": 6, "4-4": 6, "5-5": 2, "6-4": 15, "6-5": 2, "6-6": 4, "7-6": 4} | {"-1": 10, "-13": 1, "-3": 1, "-4": 4, "-5": 8, "-6": 4, "-9": 3, "0": 30, "1": 12, "10": 9, "4": 6, "5": 7, "6": 8, "8": 1} | {"000/001": 6, "000/010": 18, "000/100": 7, "001/000": 4, "010/000": 12, "100/000": 5} | {"10": 9, "14": 1, "3": 1, "5": 38, "9": 3} | {"none": 52} | {"at least one": 40, "both": 0, "neither": 12} |
| center_only_s1 | calibration | main, changed marker | pairs | {"1-1": 3, "3-1": 10, "3-3": 3, "4-2": 10, "4-4": 3, "6-4": 25, "7-5": 2, "7-6": 1, "7-7": 3} | {"-1": 16, "-10": 1, "-4": 8, "-5": 4, "-6": 10, "0": 29, "1": 15, "10": 11, "2": 1, "4": 5, "5": 13, "6": 7} | {"000/001": 6, "000/010": 24, "000/100": 7, "001/000": 10, "010/000": 5, "100/000": 8} | {"10": 12, "3": 1, "5": 47} | {"0-5": 30, "6-10": 9, ">10": 7, "none": 14} | {"at least one": 45, "both": 0, "neither": 15} |
| center_only_s1 | calibration | main, changed marker | multi | {"1-1": 4, "3-1": 21, "3-3": 6, "4-1": 9, "4-2": 26, "4-4": 5, "6-4": 51, "6-6": 5, "7-5": 3, "7-6": 1} | {"-1": 31, "-10": 8, "-13": 4, "-3": 4, "-4": 20, "-5": 8, "-6": 21, "0": 72, "1": 28, "10": 24, "4": 10, "5": 24, "6": 8} | {"000/001": 10, "000/010": 48, "000/100": 8, "001/000": 21, "010/000": 24, "100/000": 20} | {"10": 32, "13": 4, "3": 4, "5": 91} | {"0-5": 80, "6-10": 16, ">10": 35} | {"at least one": 103, "both": 0, "neither": 28} |
| center_only_s1 | calibration | broad | native | {"0-0": 8, "1-0": 7, "1-1": 3, "2-0": 18, "2-1": 9, "2-2": 7, "3-0": 2, "3-1": 7, "3-2": 10, "3-3": 6, "4-0": 4, "4-1": 2, "4-2": 7, "4-3": 6, "4-4": 9, "5-0": 78, "5-1": 8, "5-3": 9, "5-4": 6, "5-5": 6, "6-0": 6, "6-1": 72, "6-2": 8, "6-4": 16, "6-5": 4, "6-6": 8, "7-0": 14, "7-1": 4, "7-2": 70, "7-3": 5, "7-4": 2, "7-5": 18, "7-6": 8, "7-7": 2} | {"-1": 11, "-10": 5, "-11": 16, "-12": 2, "-13": 16, "-14": 2, "-15": 13, "-2": 24, "-3": 28, "-4": 86, "-5": 13, "-6": 88, "-7": 15, "-8": 112, "-9": 18, "0": 32, "1": 12, "10": 14, "12": 21, "2": 128, "3": 9, "4": 95, "5": 11, "6": 98, "7": 17, "8": 12} | {"000/000": 394, "000/001": 6, "000/010": 18, "000/100": 7, "001/000": 5, "010/000": 14, "100/000": 5} | {"10": 272, "14": 2, "15": 33, "19": 7, "20": 11, "3": 1, "4": 4, "5": 94, "9": 25} | {"none": 449} | {"at least one": 43, "both": 0, "neither": 406} |
| center_only_s1 | calibration | broad | pairs | {"0-0": 2, "1-1": 4, "2-0": 26, "2-2": 7, "3-0": 4, "3-1": 23, "3-3": 4, "4-0": 20, "4-2": 21, "4-4": 5, "5-0": 81, "5-1": 17, "5-2": 3, "5-3": 25, "5-5": 2, "6-0": 14, "6-1": 88, "6-2": 15, "6-3": 3, "6-4": 27, "6-6": 2, "7-0": 73, "7-1": 16, "7-2": 91, "7-3": 18, "7-4": 3, "7-5": 27, "7-6": 2, "7-7": 4} | {"-1": 16, "-10": 13, "-11": 17, "-12": 5, "-13": 19, "-14": 6, "-15": 17, "-2": 33, "-3": 19, "-4": 123, "-5": 7, "-6": 118, "-7": 9, "-8": 183, "-9": 23, "0": 45, "1": 17, "10": 16, "12": 51, "2": 142, "3": 15, "4": 130, "5": 18, "6": 149, "7": 49, "8": 14} | {"000/000": 549, "000/001": 6, "000/010": 24, "000/100": 7, "001/000": 10, "010/000": 21, "100/000": 10} | {"10": 370, "13": 7, "15": 88, "19": 11, "20": 45, "25": 3, "3": 5, "5": 97, "9": 1} | {"0-5": 409, "6-10": 78, ">10": 42, "none": 98} | {"at least one": 61, "both": 0, "neither": 566} |
| center_only_s1 | calibration | broad | multi | {"0-0": 3, "1-0": 2, "1-1": 8, "2-0": 46, "2-2": 6, "3-0": 8, "3-1": 50, "3-3": 9, "4-0": 10, "4-1": 9, "4-2": 48, "4-4": 10, "5-0": 802, "5-1": 14, "5-2": 14, "5-3": 48, "5-5": 2, "6-0": 14, "6-1": 795, "6-2": 13, "6-3": 13, "6-4": 54, "6-6": 7, "7-0": 123, "7-1": 12, "7-2": 810, "7-3": 10, "7-4": 9, "7-5": 51, "7-6": 2, "7-7": 1} | {"-1": 31, "-10": 16, "-11": 34, "-12": 7, "-13": 29, "-14": 13, "-15": 18, "-2": 62, "-3": 32, "-4": 863, "-5": 8, "-6": 853, "-7": 15, "-8": 954, "-9": 43, "0": 82, "1": 28, "10": 24, "12": 75, "2": 906, "3": 23, "4": 871, "5": 24, "6": 897, "7": 73, "8": 25} | {"000/000": 2862, "000/001": 10, "000/010": 48, "000/100": 8, "001/000": 21, "010/000": 34, "100/000": 20} | {"10": 2583, "13": 27, "15": 124, "19": 13, "20": 62, "3": 12, "5": 180, "9": 2} | {"0-5": 2343, "6-10": 386, ">10": 274} | {"at least one": 113, "both": 0, "neither": 2890} |
| center_only_s2 | calibration | main | native | {"2-2": 1, "3-1": 2, "4-2": 2, "5-3": 2, "5-4": 2, "6-1": 27, "6-4": 2, "6-5": 2, "6-6": 1, "7-1": 2, "7-2": 28, "7-5": 2, "7-6": 2} | {"-1": 3, "-11": 1, "-13": 1, "-2": 2, "-3": 2, "-4": 31, "-5": 1, "-6": 28, "-7": 1, "-9": 2, "0": 2, "1": 2, "10": 1, "12": 1, "2": 4, "3": 1, "4": 31, "5": 1, "6": 33, "7": 1, "8": 1} | {"000/000": 68, "000/001": 2, "000/010": 2, "000/100": 1, "001/000": 1, "100/000": 1} | {"10": 59, "15": 2, "19": 1, "5": 10, "9": 3} | {"none": 75} | {"at least one": 5, "both": 0, "neither": 70} |
| center_only_s2 | calibration | main | pairs | {"1-1": 2, "2-2": 2, "3-1": 17, "3-3": 2, "4-1": 3, "4-2": 19, "4-4": 1, "5-1": 16, "5-2": 3, "5-3": 20, "5-5": 1, "6-1": 77, "6-2": 16, "6-3": 3, "6-4": 22, "6-6": 1, "7-1": 14, "7-2": 81, "7-3": 15, "7-4": 3, "7-5": 19, "7-7": 3} | {"-1": 11, "-10": 9, "-11": 11, "-12": 4, "-13": 17, "-2": 20, "-3": 5, "-4": 103, "-5": 6, "-6": 97, "-7": 7, "-8": 17, "-9": 15, "0": 25, "1": 10, "10": 13, "12": 15, "2": 35, "3": 10, "4": 100, "5": 13, "6": 113, "7": 14, "8": 10} | {"000/000": 294, "000/001": 4, "000/010": 18, "000/100": 2, "001/000": 7, "010/000": 7, "100/000": 8} | {"10": 224, "13": 4, "15": 34, "18": 2, "19": 5, "20": 7, "23": 2, "25": 1, "3": 7, "5": 54} | {"0-5": 299, "6-10": 20, ">10": 3, "none": 18} | {"at least one": 36, "both": 0, "neither": 304} |
| center_only_s2 | calibration | main | multi | {"1-1": 2, "2-2": 4, "3-1": 38, "3-3": 2, "4-1": 6, "4-2": 39, "4-4": 3, "5-1": 10, "5-2": 6, "5-3": 41, "5-5": 3, "6-1": 749, "6-2": 8, "6-3": 6, "6-4": 38, "6-6": 2, "7-1": 14, "7-2": 744, "7-3": 10, "7-4": 6, "7-5": 36, "7-7": 3} | {"-1": 20, "-10": 6, "-11": 16, "-12": 3, "-13": 15, "-2": 41, "-3": 6, "-4": 779, "-5": 5, "-6": 784, "-7": 6, "-8": 11, "-9": 23, "0": 46, "1": 21, "10": 19, "12": 19, "2": 57, "3": 19, "4": 788, "5": 17, "6": 800, "7": 17, "8": 22} | {"000/000": 1683, "000/001": 5, "000/010": 32, "000/100": 6, "001/000": 15, "010/000": 14, "100/000": 15} | {"10": 1605, "13": 12, "15": 37, "19": 7, "5": 109} | {"0-5": 1632, "6-10": 95, ">10": 43} | {"at least one": 66, "both": 0, "neither": 1704} |
| center_only_s2 | calibration | main, equal marker | native | {"2-2": 1, "3-1": 1, "4-2": 1, "5-3": 2, "5-4": 2, "6-1": 27, "6-5": 1, "7-1": 2, "7-2": 28, "7-5": 2, "7-6": 1} | {"-11": 1, "-13": 1, "-2": 2, "-3": 2, "-4": 30, "-5": 1, "-6": 27, "-7": 1, "-9": 2, "12": 1, "2": 4, "3": 1, "4": 29, "6": 32, "7": 1, "8": 1} | {"000/000": 68} | {"10": 58, "15": 2, "19": 1, "5": 4, "9": 3} | {"none": 68} | {"at least one": 0, "both": 0, "neither": 68} |
| center_only_s2 | calibration | main, equal marker | pairs | {"1-1": 2, "2-2": 2, "3-1": 10, "3-3": 2, "4-2": 8, "4-4": 1, "5-1": 16, "5-2": 3, "5-3": 20, "5-5": 1, "6-1": 77, "6-2": 16, "6-3": 3, "6-4": 2, "7-1": 14, "7-2": 81, "7-3": 15, "7-4": 3, "7-5": 17, "7-7": 1} | {"-10": 6, "-11": 11, "-12": 4, "-13": 16, "-2": 20, "-3": 4, "-4": 95, "-5": 4, "-6": 90, "-7": 7, "-8": 17, "-9": 15, "10": 4, "12": 15, "2": 34, "3": 10, "4": 97, "5": 4, "6": 111, "7": 14, "8": 10} | {"000/000": 294} | {"10": 212, "13": 3, "15": 34, "18": 2, "19": 5, "20": 7, "23": 2, "25": 1, "3": 5, "5": 23} | {"0-5": 263, "6-10": 16, ">10": 3, "none": 12} | {"at least one": 0, "both": 0, "neither": 294} |
| center_only_s2 | calibration | main, equal marker | multi | {"2-2": 4, "3-1": 23, "4-2": 18, "5-1": 10, "5-2": 6, "5-3": 41, "5-5": 3, "6-1": 749, "6-2": 8, "6-3": 6, "6-4": 3, "6-6": 2, "7-1": 14, "7-2": 744, "7-3": 10, "7-4": 6, "7-5": 34, "7-7": 2} | {"-11": 16, "-12": 3, "-13": 12, "-2": 41, "-3": 6, "-4": 764, "-6": 769, "-7": 6, "-8": 11, "-9": 23, "10": 2, "12": 19, "2": 57, "3": 19, "4": 783, "5": 2, "6": 794, "7": 17, "8": 22} | {"000/000": 1683} | {"10": 1582, "13": 9, "15": 37, "19": 7, "5": 48} | {"0-5": 1552, "6-10": 90, ">10": 41} | {"at least one": 0, "both": 0, "neither": 1683} |
| center_only_s2 | calibration | main, changed marker | native | {"3-1": 1, "4-2": 1, "6-4": 2, "6-5": 1, "6-6": 1, "7-6": 1} | {"-1": 3, "-4": 1, "-6": 1, "0": 2, "1": 2, "10": 1, "4": 2, "5": 1, "6": 1} | {"000/001": 2, "000/010": 2, "000/100": 1, "001/000": 1, "100/000": 1} | {"10": 1, "5": 6} | {"none": 7} | {"at least one": 5, "both": 0, "neither": 2} |
| center_only_s2 | calibration | main, changed marker | pairs | {"3-1": 7, "4-1": 3, "4-2": 11, "6-4": 20, "6-6": 1, "7-5": 2, "7-7": 2} | {"-1": 11, "-10": 3, "-13": 1, "-3": 1, "-4": 8, "-5": 2, "-6": 7, "0": 25, "1": 10, "10": 9, "2": 1, "4": 3, "5": 9, "6": 2} | {"000/001": 4, "000/010": 18, "000/100": 2, "001/000": 7, "010/000": 7, "100/000": 8} | {"10": 12, "13": 1, "3": 2, "5": 31} | {"0-5": 36, "6-10": 4, "none": 6} | {"at least one": 36, "both": 0, "neither": 10} |
| center_only_s2 | calibration | main, changed marker | multi | {"1-1": 2, "3-1": 15, "3-3": 2, "4-1": 6, "4-2": 21, "4-4": 3, "6-4": 35, "7-5": 2, "7-7": 1} | {"-1": 20, "-10": 6, "-13": 3, "-4": 15, "-5": 5, "-6": 15, "0": 46, "1": 21, "10": 17, "4": 5, "5": 15, "6": 6} | {"000/001": 5, "000/010": 32, "000/100": 6, "001/000": 15, "010/000": 14, "100/000": 15} | {"10": 23, "13": 3, "5": 61} | {"0-5": 80, "6-10": 5, ">10": 2} | {"at least one": 66, "both": 0, "neither": 21} |
| center_only_s2 | calibration | broad | native | {"0-0": 1, "1-0": 4, "2-0": 4, "2-2": 1, "3-1": 2, "4-2": 2, "5-0": 30, "5-3": 2, "5-4": 2, "6-0": 2, "6-1": 27, "6-4": 2, "6-5": 2, "6-6": 1, "7-0": 4, "7-1": 2, "7-2": 28, "7-5": 2, "7-6": 2} | {"-1": 3, "-10": 1, "-11": 3, "-13": 2, "-15": 5, "-2": 2, "-3": 4, "-4": 31, "-5": 2, "-6": 30, "-7": 1, "-8": 37, "-9": 2, "0": 3, "1": 2, "10": 1, "12": 3, "2": 36, "3": 1, "4": 33, "5": 1, "6": 33, "7": 3, "8": 1} | {"000/000": 112, "000/001": 2, "000/010": 2, "000/100": 1, "001/000": 1, "010/000": 1, "100/000": 1} | {"10": 91, "15": 7, "19": 2, "20": 2, "5": 14, "9": 4} | {"none": 120} | {"at least one": 6, "both": 0, "neither": 114} |
| center_only_s2 | calibration | broad | pairs | {"1-1": 2, "2-0": 18, "2-2": 2, "3-0": 2, "3-1": 17, "3-3": 2, "4-0": 16, "4-1": 3, "4-2": 19, "4-4": 1, "5-0": 77, "5-1": 16, "5-2": 3, "5-3": 20, "5-5": 1, "6-0": 14, "6-1": 77, "6-2": 16, "6-3": 3, "6-4": 22, "6-6": 1, "7-0": 56, "7-1": 14, "7-2": 81, "7-3": 15, "7-4": 3, "7-5": 19, "7-7": 3} | {"-1": 11, "-10": 17, "-11": 16, "-12": 7, "-13": 19, "-14": 5, "-15": 16, "-2": 22, "-3": 12, "-4": 107, "-5": 6, "-6": 97, "-7": 7, "-8": 156, "-9": 20, "0": 37, "1": 12, "10": 15, "12": 43, "2": 115, "3": 12, "4": 110, "5": 15, "6": 117, "7": 40, "8": 12} | {"000/000": 463, "000/001": 4, "000/010": 18, "000/100": 2, "001/000": 7, "010/000": 19, "100/000": 10} | {"10": 313, "13": 5, "15": 77, "18": 3, "19": 10, "20": 39, "23": 3, "25": 3, "3": 7, "5": 63} | {"0-5": 453, "6-10": 45, ">10": 3, "none": 22} | {"at least one": 48, "both": 0, "neither": 475} |
| center_only_s2 | calibration | broad | multi | {"0-0": 2, "1-1": 2, "2-0": 35, "2-2": 4, "3-0": 6, "3-1": 38, "3-3": 2, "4-0": 10, "4-1": 6, "4-2": 39, "4-4": 3, "5-0": 769, "5-1": 10, "5-2": 6, "5-3": 41, "5-5": 3, "6-0": 12, "6-1": 749, "6-2": 8, "6-3": 6, "6-4": 38, "6-6": 2, "7-0": 124, "7-1": 14, "7-2": 744, "7-3": 10, "7-4": 6, "7-5": 36, "7-7": 3} | {"-1": 20, "-10": 11, "-11": 22, "-12": 6, "-13": 24, "-14": 16, "-15": 14, "-2": 48, "-3": 19, "-4": 785, "-5": 5, "-6": 784, "-7": 7, "-8": 900, "-9": 39, "0": 56, "1": 21, "10": 19, "12": 69, "2": 842, "3": 19, "4": 800, "5": 17, "6": 826, "7": 65, "8": 22} | {"000/000": 2631, "000/001": 5, "000/010": 32, "000/100": 6, "001/000": 15, "010/000": 24, "100/000": 15} | {"10": 2393, "13": 15, "15": 117, "19": 13, "20": 63, "5": 127} | {"0-5": 2413, "6-10": 251, ">10": 64} | {"at least one": 76, "both": 0, "neither": 2652} |
| shuffled_dense_s0 | calibration | main | native | {"2-1": 2, "3-2": 2, "4-2": 1, "4-3": 2, "4-4": 1, "5-4": 2, "5-5": 1, "6-1": 19, "6-5": 4, "6-6": 1, "7-1": 2, "7-2": 19, "7-6": 2} | {"-1": 1, "-11": 2, "-13": 2, "-2": 2, "-3": 2, "-4": 21, "-5": 4, "-6": 18, "-7": 2, "-9": 4, "0": 4, "1": 2, "10": 2, "2": 2, "4": 21, "5": 2, "6": 25} | {"000/000": 51, "000/001": 1, "000/010": 1, "000/100": 2, "010/000": 3} | {"10": 37, "15": 4, "19": 1, "5": 10, "9": 6} | {"none": 58} | {"at least one": 5, "both": 0, "neither": 53} |
| shuffled_dense_s0 | calibration | main | pairs | {"1-1": 1, "2-1": 2, "3-1": 21, "3-3": 1, "4-1": 2, "4-2": 22, "5-1": 16, "5-2": 2, "5-3": 21, "5-5": 1, "6-1": 73, "6-2": 16, "6-3": 2, "6-4": 22, "7-1": 12, "7-2": 75, "7-3": 16, "7-4": 2, "7-5": 21, "7-7": 1} | {"-1": 11, "-10": 7, "-11": 10, "-12": 4, "-13": 16, "-2": 21, "-3": 4, "-4": 99, "-5": 5, "-6": 94, "-7": 5, "-8": 15, "-9": 13, "0": 25, "1": 11, "10": 13, "12": 16, "2": 35, "3": 11, "4": 97, "5": 13, "6": 106, "7": 15, "8": 12} | {"000/000": 282, "000/001": 3, "000/010": 18, "000/100": 2, "001/000": 8, "010/000": 7, "100/000": 9} | {"10": 218, "13": 4, "15": 33, "18": 2, "19": 4, "20": 7, "23": 2, "25": 1, "5": 57, "9": 1} | {"0-5": 294, "6-10": 15, "none": 20} | {"at least one": 36, "both": 0, "neither": 293} |
| shuffled_dense_s0 | calibration | main | multi | {"2-2": 2, "3-1": 39, "3-3": 1, "4-1": 6, "4-2": 40, "5-1": 10, "5-2": 4, "5-3": 39, "5-5": 2, "6-1": 729, "6-2": 10, "6-3": 4, "6-4": 38, "7-1": 12, "7-2": 734, "7-3": 10, "7-4": 4, "7-5": 41, "7-7": 1} | {"-1": 20, "-10": 6, "-11": 15, "-12": 3, "-13": 14, "-2": 39, "-3": 5, "-4": 771, "-5": 3, "-6": 764, "-7": 5, "-8": 10, "-9": 22, "0": 44, "1": 20, "10": 16, "12": 19, "2": 57, "3": 17, "4": 767, "5": 16, "6": 783, "7": 18, "8": 18} | {"000/000": 1642, "000/001": 4, "000/010": 32, "000/100": 3, "001/000": 16, "010/000": 12, "100/000": 17} | {"10": 1571, "13": 9, "15": 37, "19": 6, "5": 103} | {"0-5": 1614, "6-10": 76, ">10": 36} | {"at least one": 64, "both": 0, "neither": 1662} |
| shuffled_dense_s0 | calibration | main, equal marker | native | {"2-1": 2, "3-2": 2, "4-2": 1, "5-4": 2, "6-1": 19, "6-5": 2, "6-6": 1, "7-1": 2, "7-2": 19, "7-6": 1} | {"-11": 2, "-13": 2, "-2": 2, "-3": 2, "-4": 21, "-5": 2, "-6": 18, "-7": 2, "-9": 3, "10": 2, "2": 2, "4": 20, "5": 1, "6": 23} | {"000/000": 51} | {"10": 37, "15": 4, "19": 1, "5": 4, "9": 5} | {"none": 51} | {"at least one": 0, "both": 0, "neither": 51} |
| shuffled_dense_s0 | calibration | main, equal marker | pairs | {"2-1": 2, "3-1": 13, "4-2": 9, "5-1": 16, "5-2": 2, "5-3": 21, "5-5": 1, "6-1": 73, "6-2": 16, "6-3": 2, "6-4": 2, "7-1": 12, "7-2": 75, "7-3": 16, "7-4": 2, "7-5": 19, "7-7": 1} | {"-10": 4, "-11": 10, "-12": 4, "-13": 15, "-2": 21, "-3": 4, "-4": 90, "-5": 2, "-6": 86, "-7": 5, "-8": 15, "-9": 13, "10": 4, "12": 16, "2": 35, "3": 11, "4": 94, "5": 4, "6": 104, "7": 15, "8": 12} | {"000/000": 282} | {"10": 206, "13": 3, "15": 33, "18": 2, "19": 4, "20": 7, "23": 2, "25": 1, "5": 23, "9": 1} | {"0-5": 259, "6-10": 11, "none": 12} | {"at least one": 0, "both": 0, "neither": 282} |
| shuffled_dense_s0 | calibration | main, equal marker | multi | {"2-2": 2, "3-1": 23, "4-2": 17, "5-1": 10, "5-2": 4, "5-3": 39, "5-5": 2, "6-1": 729, "6-2": 10, "6-3": 4, "6-4": 3, "7-1": 12, "7-2": 734, "7-3": 10, "7-4": 4, "7-5": 38, "7-7": 1} | {"-11": 15, "-12": 3, "-13": 11, "-2": 39, "-3": 5, "-4": 754, "-6": 748, "-7": 5, "-8": 10, "-9": 22, "12": 19, "2": 57, "3": 17, "4": 763, "6": 780, "7": 18, "8": 18} | {"000/000": 1642} | {"10": 1549, "13": 6, "15": 37, "19": 6, "5": 44} | {"0-5": 1537, "6-10": 73, ">10": 32} | {"at least one": 0, "both": 0, "neither": 1642} |
| shuffled_dense_s0 | calibration | main, changed marker | native | {"4-3": 2, "4-4": 1, "5-5": 1, "6-5": 2, "7-6": 1} | {"-1": 1, "-5": 2, "-9": 1, "0": 4, "1": 2, "4": 1, "5": 1, "6": 2} | {"000/001": 1, "000/010": 1, "000/100": 2, "010/000": 3} | {"5": 6, "9": 1} | {"none": 7} | {"at least one": 5, "both": 0, "neither": 2} |
| shuffled_dense_s0 | calibration | main, changed marker | pairs | {"1-1": 1, "3-1": 8, "3-3": 1, "4-1": 2, "4-2": 13, "6-4": 20, "7-5": 2} | {"-1": 11, "-10": 3, "-13": 1, "-4": 9, "-5": 3, "-6": 8, "0": 25, "1": 11, "10": 9, "4": 3, "5": 9, "6": 2} | {"000/001": 3, "000/010": 18, "000/100": 2, "001/000": 8, "010/000": 7, "100/000": 9} | {"10": 12, "13": 1, "5": 34} | {"0-5": 35, "6-10": 4, "none": 8} | {"at least one": 36, "both": 0, "neither": 11} |
| shuffled_dense_s0 | calibration | main, changed marker | multi | {"3-1": 16, "3-3": 1, "4-1": 6, "4-2": 23, "6-4": 35, "7-5": 3} | {"-1": 20, "-10": 6, "-13": 3, "-4": 17, "-5": 3, "-6": 16, "0": 44, "1": 20, "10": 16, "4": 4, "5": 16, "6": 3} | {"000/001": 4, "000/010": 32, "000/100": 3, "001/000": 16, "010/000": 12, "100/000": 17} | {"10": 22, "13": 3, "5": 59} | {"0-5": 77, "6-10": 3, ">10": 4} | {"at least one": 64, "both": 0, "neither": 20} |
| shuffled_dense_s0 | calibration | broad | native | {"0-0": 1, "1-0": 2, "2-1": 2, "3-2": 2, "4-2": 1, "4-3": 2, "4-4": 1, "5-0": 19, "5-4": 2, "5-5": 1, "6-0": 2, "6-1": 19, "6-5": 4, "6-6": 1, "7-0": 4, "7-1": 2, "7-2": 19, "7-6": 2} | {"-1": 1, "-11": 4, "-13": 3, "-14": 1, "-15": 2, "-2": 3, "-3": 2, "-4": 21, "-5": 4, "-6": 20, "-7": 3, "-8": 20, "-9": 5, "0": 4, "1": 2, "10": 2, "12": 1, "2": 21, "4": 23, "5": 2, "6": 27, "7": 1} | {"000/000": 79, "000/001": 1, "000/010": 1, "000/100": 2, "010/000": 3} | {"10": 55, "15": 8, "19": 2, "20": 2, "5": 12, "9": 7} | {"none": 86} | {"at least one": 5, "both": 0, "neither": 81} |
| shuffled_dense_s0 | calibration | broad | pairs | {"0-0": 1, "1-0": 2, "1-1": 1, "2-0": 22, "2-1": 2, "3-0": 2, "3-1": 21, "3-3": 1, "4-0": 16, "4-1": 2, "4-2": 22, "5-0": 75, "5-1": 16, "5-2": 2, "5-3": 21, "5-5": 1, "6-0": 12, "6-1": 73, "6-2": 16, "6-3": 2, "6-4": 22, "7-0": 59, "7-1": 12, "7-2": 75, "7-3": 16, "7-4": 2, "7-5": 21, "7-7": 1} | {"-1": 11, "-10": 15, "-11": 15, "-12": 7, "-13": 19, "-14": 5, "-15": 16, "-2": 24, "-3": 13, "-4": 103, "-5": 5, "-6": 96, "-7": 6, "-8": 158, "-9": 18, "0": 37, "1": 13, "10": 15, "12": 45, "2": 115, "3": 13, "4": 105, "5": 15, "6": 110, "7": 43, "8": 14} | {"000/000": 457, "000/001": 3, "000/010": 18, "000/100": 2, "001/000": 8, "010/000": 19, "100/000": 11} | {"10": 307, "13": 5, "15": 76, "18": 3, "19": 8, "20": 41, "23": 3, "25": 3, "5": 70, "9": 2} | {"0-5": 447, "6-10": 47, "none": 24} | {"at least one": 48, "both": 0, "neither": 470} |
| shuffled_dense_s0 | calibration | broad | multi | {"2-0": 41, "2-2": 2, "3-0": 6, "3-1": 39, "3-3": 1, "4-0": 10, "4-1": 6, "4-2": 40, "5-0": 746, "5-1": 10, "5-2": 4, "5-3": 39, "5-5": 2, "6-0": 12, "6-1": 729, "6-2": 10, "6-3": 4, "6-4": 38, "7-0": 127, "7-1": 12, "7-2": 734, "7-3": 10, "7-4": 4, "7-5": 41, "7-7": 1} | {"-1": 20, "-10": 11, "-11": 21, "-12": 6, "-13": 22, "-14": 16, "-15": 14, "-2": 45, "-3": 22, "-4": 777, "-5": 3, "-6": 764, "-7": 5, "-8": 884, "-9": 38, "0": 54, "1": 20, "10": 16, "12": 70, "2": 821, "3": 17, "4": 779, "5": 16, "6": 809, "7": 68, "8": 18} | {"000/000": 2574, "000/001": 4, "000/010": 32, "000/100": 3, "001/000": 16, "010/000": 22, "100/000": 17} | {"10": 2338, "13": 12, "15": 119, "19": 12, "20": 64, "5": 123} | {"0-5": 2390, "6-10": 222, ">10": 56} | {"at least one": 74, "both": 0, "neither": 2594} |
| shuffled_dense_s1 | calibration | main | native | {"1-1": 1, "2-1": 2, "3-2": 2, "4-3": 2, "6-1": 24, "7-1": 2, "7-2": 21, "7-5": 1, "7-6": 2, "7-7": 1} | {"-11": 2, "-13": 2, "-2": 2, "-3": 1, "-4": 22, "-5": 2, "-6": 23, "-7": 1, "-9": 4, "0": 3, "1": 1, "12": 1, "2": 1, "4": 24, "6": 25, "7": 2} | {"000/000": 54, "000/100": 1, "010/000": 3} | {"10": 43, "15": 3, "19": 1, "5": 7, "9": 4} | {"none": 58} | {"at least one": 3, "both": 0, "neither": 55} |
| shuffled_dense_s1 | calibration | main | pairs | {"1-1": 3, "3-1": 17, "3-3": 1, "4-1": 2, "4-2": 17, "4-4": 1, "5-1": 16, "5-2": 4, "5-3": 17, "5-5": 1, "6-1": 73, "6-2": 16, "6-3": 4, "6-4": 17, "6-6": 1, "7-1": 12, "7-2": 74, "7-3": 16, "7-4": 4, "7-5": 18, "7-7": 2} | {"-1": 9, "-10": 7, "-11": 11, "-12": 4, "-13": 15, "-2": 17, "-3": 4, "-4": 92, "-5": 7, "-6": 91, "-7": 6, "-8": 16, "-9": 13, "0": 22, "1": 9, "10": 12, "12": 15, "2": 34, "3": 9, "4": 97, "5": 11, "6": 106, "7": 15, "8": 10} | {"000/000": 276, "000/001": 3, "000/010": 13, "000/100": 3, "001/000": 6, "010/000": 9, "100/000": 6} | {"10": 210, "13": 7, "15": 33, "18": 2, "19": 4, "20": 7, "23": 2, "25": 1, "5": 50} | {"0-5": 288, "6-10": 12, ">10": 3, "none": 13} | {"at least one": 31, "both": 0, "neither": 285} |
| shuffled_dense_s1 | calibration | main | multi | {"1-1": 1, "2-2": 1, "3-1": 44, "4-1": 6, "4-2": 40, "4-4": 1, "5-1": 9, "5-2": 6, "5-3": 40, "6-1": 735, "6-2": 9, "6-3": 6, "6-4": 42, "6-6": 1, "7-1": 12, "7-2": 730, "7-3": 10, "7-4": 6, "7-5": 39, "7-7": 2} | {"-1": 23, "-10": 6, "-11": 16, "-12": 4, "-13": 14, "-2": 45, "-3": 3, "-4": 767, "-5": 3, "-6": 776, "-7": 8, "-8": 11, "-9": 22, "0": 46, "1": 20, "10": 17, "12": 18, "2": 53, "3": 18, "4": 777, "5": 17, "6": 781, "7": 17, "8": 18} | {"000/000": 1651, "000/001": 5, "000/010": 34, "000/100": 3, "001/000": 18, "010/000": 12, "100/000": 17} | {"10": 1579, "13": 12, "15": 35, "19": 6, "5": 108} | {"0-5": 1624, "6-10": 73, ">10": 43} | {"at least one": 69, "both": 0, "neither": 1671} |
| shuffled_dense_s1 | calibration | main, equal marker | native | {"2-1": 2, "3-2": 2, "6-1": 24, "7-1": 2, "7-2": 21, "7-5": 1, "7-6": 1, "7-7": 1} | {"-11": 2, "-13": 2, "-2": 2, "-3": 1, "-4": 22, "-6": 23, "-7": 1, "-9": 3, "12": 1, "2": 1, "4": 24, "6": 24, "7": 2} | {"000/000": 54} | {"10": 43, "15": 3, "19": 1, "5": 4, "9": 3} | {"none": 54} | {"at least one": 0, "both": 0, "neither": 54} |
| shuffled_dense_s1 | calibration | main, equal marker | pairs | {"3-1": 11, "4-2": 7, "5-1": 16, "5-2": 4, "5-3": 17, "5-5": 1, "6-1": 73, "6-2": 16, "6-3": 4, "6-4": 2, "6-6": 1, "7-1": 12, "7-2": 74, "7-3": 16, "7-4": 4, "7-5": 16, "7-7": 2} | {"-10": 4, "-11": 11, "-12": 4, "-13": 14, "-2": 17, "-3": 4, "-4": 86, "-5": 2, "-6": 85, "-7": 6, "-8": 16, "-9": 13, "10": 5, "12": 15, "2": 34, "3": 9, "4": 94, "5": 5, "6": 103, "7": 15, "8": 10} | {"000/000": 276} | {"10": 200, "13": 6, "15": 33, "18": 2, "19": 4, "20": 7, "23": 2, "25": 1, "5": 21} | {"0-5": 256, "6-10": 10, ">10": 2, "none": 8} | {"at least one": 0, "both": 0, "neither": 276} |
| shuffled_dense_s1 | calibration | main, equal marker | multi | {"1-1": 1, "2-2": 1, "3-1": 26, "4-2": 17, "4-4": 1, "5-1": 9, "5-2": 6, "5-3": 40, "6-1": 735, "6-2": 9, "6-3": 6, "6-4": 4, "6-6": 1, "7-1": 12, "7-2": 730, "7-3": 10, "7-4": 6, "7-5": 36, "7-7": 1} | {"-11": 16, "-12": 4, "-13": 11, "-2": 45, "-3": 3, "-4": 750, "-6": 758, "-7": 8, "-8": 11, "-9": 22, "12": 18, "2": 53, "3": 18, "4": 772, "6": 778, "7": 17, "8": 18} | {"000/000": 1651} | {"10": 1556, "13": 9, "15": 35, "19": 6, "5": 45} | {"0-5": 1543, "6-10": 69, ">10": 39} | {"at least one": 0, "both": 0, "neither": 1651} |
| shuffled_dense_s1 | calibration | main, changed marker | native | {"1-1": 1, "4-3": 2, "7-6": 1} | {"-5": 2, "-9": 1, "0": 3, "1": 1, "6": 1} | {"000/100": 1, "010/000": 3} | {"5": 3, "9": 1} | {"none": 4} | {"at least one": 3, "both": 0, "neither": 1} |
| shuffled_dense_s1 | calibration | main, changed marker | pairs | {"1-1": 3, "3-1": 6, "3-3": 1, "4-1": 2, "4-2": 10, "4-4": 1, "6-4": 15, "7-5": 2} | {"-1": 9, "-10": 3, "-13": 1, "-4": 6, "-5": 5, "-6": 6, "0": 22, "1": 9, "10": 7, "4": 3, "5": 6, "6": 3} | {"000/001": 3, "000/010": 13, "000/100": 3, "001/000": 6, "010/000": 9, "100/000": 6} | {"10": 10, "13": 1, "5": 29} | {"0-5": 32, "6-10": 2, ">10": 1, "none": 5} | {"at least one": 31, "both": 0, "neither": 9} |
| shuffled_dense_s1 | calibration | main, changed marker | multi | {"3-1": 18, "4-1": 6, "4-2": 23, "6-4": 38, "7-5": 3, "7-7": 1} | {"-1": 23, "-10": 6, "-13": 3, "-4": 17, "-5": 3, "-6": 18, "0": 46, "1": 20, "10": 17, "4": 5, "5": 17, "6": 3} | {"000/001": 5, "000/010": 34, "000/100": 3, "001/000": 18, "010/000": 12, "100/000": 17} | {"10": 23, "13": 3, "5": 63} | {"0-5": 81, "6-10": 4, ">10": 4} | {"at least one": 69, "both": 0, "neither": 20} |
| shuffled_dense_s1 | calibration | broad | native | {"0-0": 1, "1-0": 2, "1-1": 1, "2-1": 2, "3-2": 2, "4-3": 2, "5-0": 24, "6-0": 2, "6-1": 24, "7-0": 2, "7-1": 2, "7-2": 21, "7-5": 1, "7-6": 2, "7-7": 1} | {"-11": 4, "-13": 3, "-14": 1, "-15": 2, "-2": 2, "-3": 1, "-4": 22, "-5": 2, "-6": 25, "-7": 1, "-8": 25, "-9": 5, "0": 3, "1": 1, "12": 1, "2": 25, "4": 26, "6": 27, "7": 2} | {"000/000": 85, "000/100": 1, "010/000": 3} | {"10": 67, "15": 5, "19": 2, "20": 1, "5": 9, "9": 5} | {"none": 89} | {"at least one": 3, "both": 0, "neither": 86} |
| shuffled_dense_s1 | calibration | broad | pairs | {"0-0": 1, "1-1": 3, "2-0": 18, "3-0": 4, "3-1": 17, "3-3": 1, "4-0": 16, "4-1": 2, "4-2": 17, "4-4": 1, "5-0": 77, "5-1": 16, "5-2": 4, "5-3": 17, "5-5": 1, "6-0": 12, "6-1": 73, "6-2": 16, "6-3": 4, "6-4": 17, "6-6": 1, "7-0": 61, "7-1": 12, "7-2": 74, "7-3": 16, "7-4": 4, "7-5": 18, "7-7": 2} | {"-1": 9, "-10": 15, "-11": 15, "-12": 8, "-13": 18, "-14": 6, "-15": 16, "-2": 22, "-3": 11, "-4": 96, "-5": 7, "-6": 91, "-7": 7, "-8": 157, "-9": 19, "0": 34, "1": 11, "10": 14, "12": 44, "2": 114, "3": 11, "4": 105, "5": 13, "6": 112, "7": 43, "8": 12} | {"000/000": 451, "000/001": 3, "000/010": 13, "000/100": 3, "001/000": 6, "010/000": 21, "100/000": 8} | {"10": 300, "13": 9, "15": 77, "18": 3, "19": 8, "20": 42, "23": 3, "25": 3, "5": 60} | {"0-5": 440, "6-10": 46, ">10": 4, "none": 15} | {"at least one": 43, "both": 0, "neither": 462} |
| shuffled_dense_s1 | calibration | broad | multi | {"0-0": 1, "1-1": 1, "2-0": 44, "2-2": 1, "3-0": 6, "3-1": 44, "4-0": 10, "4-1": 6, "4-2": 40, "4-4": 1, "5-0": 743, "5-1": 9, "5-2": 6, "5-3": 40, "6-0": 12, "6-1": 735, "6-2": 9, "6-3": 6, "6-4": 42, "6-6": 1, "7-0": 123, "7-1": 12, "7-2": 730, "7-3": 10, "7-4": 6, "7-5": 39, "7-7": 2} | {"-1": 23, "-10": 11, "-11": 22, "-12": 7, "-13": 22, "-14": 15, "-15": 14, "-2": 52, "-3": 22, "-4": 773, "-5": 3, "-6": 776, "-7": 9, "-8": 883, "-9": 37, "0": 56, "1": 20, "10": 17, "12": 68, "2": 815, "3": 18, "4": 789, "5": 17, "6": 805, "7": 66, "8": 18} | {"000/000": 2580, "000/001": 5, "000/010": 34, "000/100": 3, "001/000": 18, "010/000": 22, "100/000": 17} | {"10": 2344, "13": 15, "15": 115, "19": 12, "20": 62, "5": 131} | {"0-5": 2394, "6-10": 221, ">10": 64} | {"at least one": 79, "both": 0, "neither": 2600} |
| shuffled_dense_s2 | calibration | main | native | {"2-2": 1, "3-1": 3, "3-3": 1, "4-2": 2, "5-3": 2, "6-1": 25, "6-4": 2, "7-1": 2, "7-2": 26, "7-5": 2} | {"-1": 2, "-11": 1, "-13": 1, "-2": 2, "-3": 1, "-4": 27, "-6": 27, "-9": 2, "0": 2, "1": 1, "10": 1, "12": 1, "2": 3, "3": 1, "4": 28, "5": 1, "6": 29, "7": 1, "8": 1} | {"000/000": 61, "000/001": 1, "000/010": 2, "001/000": 1, "100/000": 1} | {"10": 55, "15": 3, "19": 1, "5": 7} | {"none": 66} | {"at least one": 4, "both": 0, "neither": 62} |
| shuffled_dense_s2 | calibration | main | pairs | {"1-1": 1, "2-2": 1, "3-1": 21, "3-3": 1, "4-1": 2, "4-2": 21, "4-4": 1, "5-1": 16, "5-2": 2, "5-3": 21, "5-5": 1, "6-1": 74, "6-2": 16, "6-3": 2, "6-4": 20, "7-1": 12, "7-2": 77, "7-3": 16, "7-4": 2, "7-5": 20, "7-7": 2} | {"-1": 11, "-10": 7, "-11": 10, "-12": 4, "-13": 15, "-2": 21, "-3": 5, "-4": 98, "-5": 5, "-6": 95, "-7": 5, "-8": 15, "-9": 12, "0": 23, "1": 11, "10": 12, "12": 17, "2": 35, "3": 11, "4": 98, "5": 12, "6": 109, "7": 15, "8": 12} | {"000/000": 284, "000/001": 3, "000/010": 16, "000/100": 3, "001/000": 8, "010/000": 7, "100/000": 8} | {"10": 220, "13": 4, "15": 33, "18": 2, "19": 4, "20": 7, "23": 2, "25": 1, "5": 56} | {"0-5": 289, "6-10": 16, ">10": 4, "none": 20} | {"at least one": 34, "both": 0, "neither": 295} |
| shuffled_dense_s2 | calibration | main | multi | {"1-1": 1, "2-2": 2, "3-1": 42, "3-3": 3, "4-1": 6, "4-2": 39, "4-4": 2, "5-1": 6, "5-2": 6, "5-3": 40, "5-5": 2, "6-1": 727, "6-2": 7, "6-3": 6, "6-4": 39, "6-6": 2, "7-1": 12, "7-2": 729, "7-3": 9, "7-4": 6, "7-5": 45, "7-7": 1} | {"-1": 21, "-10": 6, "-11": 15, "-12": 4, "-13": 12, "-2": 43, "-3": 5, "-4": 766, "-5": 4, "-6": 764, "-7": 8, "-8": 9, "-9": 22, "0": 46, "1": 20, "10": 18, "12": 21, "2": 59, "3": 18, "4": 766, "5": 17, "6": 780, "7": 20, "8": 20} | {"000/000": 1645, "000/001": 5, "000/010": 33, "000/100": 4, "001/000": 16, "010/000": 13, "100/000": 16} | {"10": 1570, "13": 12, "15": 32, "19": 6, "5": 112} | {"0-5": 1599, "6-10": 92, ">10": 41} | {"at least one": 67, "both": 0, "neither": 1665} |
| shuffled_dense_s2 | calibration | main, equal marker | native | {"2-2": 1, "3-1": 2, "4-2": 1, "5-3": 2, "6-1": 25, "7-1": 2, "7-2": 26, "7-5": 2} | {"-11": 1, "-13": 1, "-2": 2, "-3": 1, "-4": 26, "-6": 26, "-9": 2, "12": 1, "2": 3, "3": 1, "4": 27, "6": 29, "7": 1, "8": 1} | {"000/000": 61} | {"10": 54, "15": 3, "19": 1, "5": 3} | {"none": 61} | {"at least one": 0, "both": 0, "neither": 61} |
| shuffled_dense_s2 | calibration | main, equal marker | pairs | {"2-2": 1, "3-1": 13, "4-2": 9, "5-1": 16, "5-2": 2, "5-3": 21, "5-5": 1, "6-1": 74, "6-2": 16, "6-3": 2, "6-4": 2, "7-1": 12, "7-2": 77, "7-3": 16, "7-4": 2, "7-5": 18, "7-7": 2} | {"-10": 4, "-11": 10, "-12": 4, "-13": 14, "-2": 21, "-3": 5, "-4": 90, "-5": 2, "-6": 87, "-7": 5, "-8": 15, "-9": 12, "10": 4, "12": 17, "2": 35, "3": 11, "4": 95, "5": 4, "6": 106, "7": 15, "8": 12} | {"000/000": 284} | {"10": 209, "13": 3, "15": 33, "18": 2, "19": 4, "20": 7, "23": 2, "25": 1, "5": 23} | {"0-5": 256, "6-10": 12, ">10": 4, "none": 12} | {"at least one": 0, "both": 0, "neither": 284} |
| shuffled_dense_s2 | calibration | main, equal marker | multi | {"2-2": 2, "3-1": 26, "3-3": 1, "4-2": 17, "4-4": 1, "5-1": 6, "5-2": 6, "5-3": 40, "5-5": 2, "6-1": 727, "6-2": 7, "6-3": 6, "6-4": 3, "6-6": 2, "7-1": 12, "7-2": 729, "7-3": 9, "7-4": 6, "7-5": 42, "7-7": 1} | {"-11": 15, "-12": 4, "-13": 9, "-2": 43, "-3": 5, "-4": 750, "-6": 748, "-7": 8, "-8": 9, "-9": 22, "10": 1, "12": 21, "2": 59, "3": 18, "4": 761, "5": 1, "6": 776, "7": 20, "8": 20} | {"000/000": 1645} | {"10": 1547, "13": 9, "15": 32, "19": 6, "5": 51} | {"0-5": 1520, "6-10": 88, ">10": 37} | {"at least one": 0, "both": 0, "neither": 1645} |
| shuffled_dense_s2 | calibration | main, changed marker | native | {"3-1": 1, "3-3": 1, "4-2": 1, "6-4": 2} | {"-1": 2, "-4": 1, "-6": 1, "0": 2, "1": 1, "10": 1, "4": 1, "5": 1} | {"000/001": 1, "000/010": 2, "001/000": 1, "100/000": 1} | {"10": 1, "5": 4} | {"none": 5} | {"at least one": 4, "both": 0, "neither": 1} |
| shuffled_dense_s2 | calibration | main, changed marker | pairs | {"1-1": 1, "3-1": 8, "3-3": 1, "4-1": 2, "4-2": 12, "4-4": 1, "6-4": 18, "7-5": 2} | {"-1": 11, "-10": 3, "-13": 1, "-4": 8, "-5": 3, "-6": 8, "0": 23, "1": 11, "10": 8, "4": 3, "5": 8, "6": 3} | {"000/001": 3, "000/010": 16, "000/100": 3, "001/000": 8, "010/000": 7, "100/000": 8} | {"10": 11, "13": 1, "5": 33} | {"0-5": 33, "6-10": 4, "none": 8} | {"at least one": 34, "both": 0, "neither": 11} |
| shuffled_dense_s2 | calibration | main, changed marker | multi | {"1-1": 1, "3-1": 16, "3-3": 2, "4-1": 6, "4-2": 22, "4-4": 1, "6-4": 36, "7-5": 3} | {"-1": 21, "-10": 6, "-13": 3, "-4": 16, "-5": 4, "-6": 16, "0": 46, "1": 20, "10": 17, "4": 5, "5": 16, "6": 4} | {"000/001": 5, "000/010": 33, "000/100": 4, "001/000": 16, "010/000": 13, "100/000": 16} | {"10": 23, "13": 3, "5": 61} | {"0-5": 79, "6-10": 4, ">10": 4} | {"at least one": 67, "both": 0, "neither": 20} |
| shuffled_dense_s2 | calibration | broad | native | {"2-0": 3, "2-2": 1, "3-1": 3, "3-3": 1, "4-2": 2, "5-0": 27, "5-3": 2, "6-0": 2, "6-1": 25, "6-4": 2, "7-0": 6, "7-1": 2, "7-2": 26, "7-5": 2} | {"-1": 2, "-11": 2, "-13": 2, "-14": 1, "-15": 1, "-2": 2, "-3": 2, "-4": 27, "-6": 27, "-8": 33, "-9": 3, "0": 2, "1": 1, "10": 1, "12": 3, "2": 32, "3": 1, "4": 30, "5": 1, "6": 31, "7": 3, "8": 1} | {"000/000": 99, "000/001": 1, "000/010": 2, "001/000": 1, "100/000": 1} | {"10": 83, "15": 8, "19": 2, "20": 3, "5": 8} | {"none": 104} | {"at least one": 4, "both": 0, "neither": 100} |
| shuffled_dense_s2 | calibration | broad | pairs | {"0-0": 1, "1-1": 1, "2-0": 21, "2-2": 1, "3-0": 2, "3-1": 21, "3-3": 1, "4-0": 16, "4-1": 2, "4-2": 21, "4-4": 1, "5-0": 78, "5-1": 16, "5-2": 2, "5-3": 21, "5-5": 1, "6-0": 12, "6-1": 74, "6-2": 16, "6-3": 2, "6-4": 20, "7-0": 59, "7-1": 12, "7-2": 77, "7-3": 16, "7-4": 2, "7-5": 20, "7-7": 2} | {"-1": 11, "-10": 15, "-11": 14, "-12": 7, "-13": 17, "-14": 6, "-15": 15, "-2": 24, "-3": 13, "-4": 102, "-5": 5, "-6": 95, "-7": 6, "-8": 159, "-9": 18, "0": 35, "1": 13, "10": 14, "12": 46, "2": 118, "3": 13, "4": 106, "5": 14, "6": 115, "7": 41, "8": 14} | {"000/000": 459, "000/001": 3, "000/010": 16, "000/100": 3, "001/000": 8, "010/000": 19, "100/000": 10} | {"10": 312, "13": 5, "15": 76, "18": 3, "19": 8, "20": 41, "23": 3, "25": 3, "5": 67} | {"0-5": 439, "6-10": 50, ">10": 5, "none": 24} | {"at least one": 46, "both": 0, "neither": 472} |
| shuffled_dense_s2 | calibration | broad | multi | {"0-0": 1, "1-1": 1, "2-0": 45, "2-2": 2, "3-0": 6, "3-1": 42, "3-3": 3, "4-0": 8, "4-1": 6, "4-2": 39, "4-4": 2, "5-0": 740, "5-1": 6, "5-2": 6, "5-3": 40, "5-5": 2, "6-0": 12, "6-1": 727, "6-2": 7, "6-3": 6, "6-4": 39, "6-6": 2, "7-0": 123, "7-1": 12, "7-2": 729, "7-3": 9, "7-4": 6, "7-5": 45, "7-7": 1} | {"-1": 21, "-10": 10, "-11": 21, "-12": 7, "-13": 20, "-14": 17, "-15": 13, "-2": 50, "-3": 23, "-4": 774, "-5": 4, "-6": 764, "-7": 9, "-8": 875, "-9": 39, "0": 54, "1": 20, "10": 18, "12": 70, "2": 818, "3": 18, "4": 778, "5": 17, "6": 806, "7": 68, "8": 20} | {"000/000": 2572, "000/001": 5, "000/010": 33, "000/100": 4, "001/000": 16, "010/000": 21, "100/000": 16} | {"10": 2332, "13": 15, "15": 111, "19": 12, "20": 62, "5": 135} | {"0-5": 2370, "6-10": 233, ">10": 64} | {"at least one": 75, "both": 0, "neither": 2592} |

## Distinct cases over the three seeds

```
{
 "selection": {
  "actual": {
   "newly matched segments": {
    "segments": 23,
    "recordings": 18,
    "sessions": 12,
    "few_units": false
   },
   "lost segments": {
    "segments": 24,
    "recordings": 19,
    "sessions": 14,
    "few_units": false
   },
   "newly unmatched predicted segments": {
    "segments": 125,
    "recordings": 45,
    "sessions": 21,
    "few_units": false
   },
   "removed unmatched predicted segments": {
    "segments": 69,
    "recordings": 40,
    "sessions": 20,
    "few_units": false
   },
   "changed windows (fixed, all)": {
    "windows": 136,
    "sessions": 22,
    "few_units": false
   },
   "changed windows (fixed, mixed)": {
    "windows": 94,
    "sessions": 21,
    "few_units": false
   },
   "changed windows (new, all)": {
    "windows": 103,
    "sessions": 17,
    "few_units": false
   },
   "changed windows (new, mixed)": {
    "windows": 23,
    "sessions": 14,
    "few_units": false
   },
   "native clips with switches (token, argmax)": {
    "clips": 21,
    "recordings": 21,
    "sessions": 12,
    "few_units": false
   },
   "native clips with switches (token, b3)": {
    "clips": 13,
    "recordings": 13,
    "sessions": 9,
    "few_units": true
   },
   "native clips with switches (cls, argmax)": {
    "clips": 15,
    "recordings": 15,
    "sessions": 9,
    "few_units": true
   },
   "native clips with switches (cls, b3)": {
    "clips": 10,
    "recordings": 10,
    "sessions": 8,
    "few_units": true
   }
  },
  "fixed": {
   "newly matched segments": {
    "segments": 22,
    "recordings": 18,
    "sessions": 12,
    "few_units": false
   },
   "lost segments": {
    "segments": 14,
    "recordings": 11,
    "sessions": 8,
    "few_units": true
   },
   "newly unmatched predicted segments": {
    "segments": 72,
    "recordings": 36,
    "sessions": 21,
    "few_units": false
   },
   "removed unmatched predicted segments": {
    "segments": 65,
    "recordings": 38,
    "sessions": 20,
    "few_units": false
   },
   "changed windows (fixed, all)": {
    "windows": 136,
    "sessions": 22,
    "few_units": false
   },
   "changed windows (fixed, mixed)": {
    "windows": 94,
    "sessions": 21,
    "few_units": false
   },
   "changed windows (new, all)": {
    "windows": 103,
    "sessions": 17,
    "few_units": false
   },
   "changed windows (new, mixed)": {
    "windows": 23,
    "sessions": 14,
    "few_units": false
   },
   "native clips with switches (token, argmax)": {
    "clips": 21,
    "recordings": 21,
    "sessions": 12,
    "few_units": false
   },
   "native clips with switches (token, b3)": {
    "clips": 12,
    "recordings": 12,
    "sessions": 9,
    "few_units": true
   },
   "native clips with switches (cls, argmax)": {
    "clips": 15,
    "recordings": 15,
    "sessions": 9,
    "few_units": true
   },
   "native clips with switches (cls, b3)": {
    "clips": 10,
    "recordings": 10,
    "sessions": 8,
    "few_units": true
   }
  }
 },
 "calibration": {
  "actual": {
   "newly matched segments": {
    "segments": 10,
    "recordings": 10,
    "sessions": 9,
    "few_units": true
   },
   "lost segments": {
    "segments": 12,
    "recordings": 11,
    "sessions": 9,
    "few_units": true
   },
   "newly unmatched predicted segments": {
    "segments": 64,
    "recordings": 25,
    "sessions": 19,
    "few_units": false
   },
   "removed unmatched predicted segments": {
    "segments": 23,
    "recordings": 16,
    "sessions": 11,
    "few_units": false
   },
   "changed windows (fixed, all)": {
    "windows": 82,
    "sessions": 27,
    "few_units": false
   },
   "changed windows (fixed, mixed)": {
    "windows": 69,
    "sessions": 26,
    "few_units": false
   },
   "changed windows (new, all)": {
    "windows": 77,
    "sessions": 21,
    "few_units": false
   },
   "changed windows (new, mixed)": {
    "windows": 25,
    "sessions": 15,
    "few_units": false
   },
   "native clips with switches (token, argmax)": {
    "clips": 13,
    "recordings": 13,
    "sessions": 10,
    "few_units": false
   },
   "native clips with switches (token, b3)": {
    "clips": 11,
    "recordings": 11,
    "sessions": 8,
    "few_units": true
   },
   "native clips with switches (cls, argmax)": {
    "clips": 9,
    "recordings": 9,
    "sessions": 8,
    "few_units": true
   },
   "native clips with switches (cls, b3)": {
    "clips": 4,
    "recordings": 4,
    "sessions": 4,
    "few_units": true
   }
  },
  "fixed": {
   "newly matched segments": {
    "segments": 9,
    "recordings": 9,
    "sessions": 8,
    "few_units": true
   },
   "lost segments": {
    "segments": 16,
    "recordings": 16,
    "sessions": 14,
    "few_units": false
   },
   "newly unmatched predicted segments": {
    "segments": 32,
    "recordings": 19,
    "sessions": 16,
    "few_units": false
   },
   "removed unmatched predicted segments": {
    "segments": 20,
    "recordings": 15,
    "sessions": 11,
    "few_units": false
   },
   "changed windows (fixed, all)": {
    "windows": 82,
    "sessions": 27,
    "few_units": false
   },
   "changed windows (fixed, mixed)": {
    "windows": 69,
    "sessions": 26,
    "few_units": false
   },
   "changed windows (new, all)": {
    "windows": 77,
    "sessions": 21,
    "few_units": false
   },
   "changed windows (new, mixed)": {
    "windows": 25,
    "sessions": 15,
    "few_units": false
   },
   "native clips with switches (token, argmax)": {
    "clips": 13,
    "recordings": 13,
    "sessions": 10,
    "few_units": false
   },
   "native clips with switches (token, b3)": {
    "clips": 5,
    "recordings": 5,
    "sessions": 4,
    "few_units": true
   },
   "native clips with switches (cls, argmax)": {
    "clips": 9,
    "recordings": 9,
    "sessions": 8,
    "few_units": true
   },
   "native clips with switches (cls, b3)": {
    "clips": 4,
    "recordings": 4,
    "sessions": 4,
    "few_units": true
   }
  }
 }
}
```

## Fixed readings

```
{
 "rows": {
  "local_correction_without_segment_recovery": {
   "applies": true,
   "shares": [
    0.7954545454545454,
    0.8,
    0.9473684210526315,
    1.0,
    0.8571428571428571,
    0.7948717948717948
   ],
   "reading": "in every seed and half, more than half of the fixed mixed-window requests of multi streams (actual systems) lie in reference segments that the CLS system already matched; the proportions and the distance bins are reported"
  },
  "mixed_vs_pure_tradeoff": {
   "applies": false,
   "halves": {
    "selection": "not observed",
    "calibration": "not observed"
   },
   "reading": "in a half, TC - CC and CT - CC multi segment F1@50 point estimates have opposite signs in all three seeds at the fixed decoder; called consistent only when both contrasts are consistent, in opposite directions, in that half; the reference mask does not establish a learnable routing policy"
  },
  "anchor_sensitivity_observed": {
   "applies": true,
   "disagreements": [
    1118,
    1212,
    1109,
    957,
    1011,
    814
   ],
   "reading": "the aligned arm's main stability subset has at least one argmax disagreement in some seed and half; equal-marker and changed-marker support are reported apart; complete local stem equality needs an equal marker triple"
  },
  "unresolved": {
   "applies": true,
   "flags": {
    "no main-subset pair holds A03's selected output anchor in both windows": true,
    "TC - CC segment F1@50 point estimates of both signs across the six cells": true,
    "a reading rests on fewer than 10 sessions": false,
    "the actual systems use differing decoders": true
   },
   "reading": "flags that prevent attribution: no main-subset pair holds A03's selected output anchor in both windows; TC - CC segment F1@50 point estimates of both signs across the six seed and half cells; a reading resting on fewer than 10 sessions; the actual systems' differing decoders"
  },
  "end": {
   "applies": true,
   "reading": "the configuration closes after the report; no training, readout fusion, shifted request, interpolation, threshold search, new loss or label change follows"
  }
 },
 "applied": [
  "local_correction_without_segment_recovery",
  "anchor_sensitivity_observed",
  "unresolved",
  "end"
 ]
}
```

## Post hoc additions

```
[]
```

## Run identity

```
{
 "code": "e861fc8d6f86438d4f64fdf0a7a24240131005a3a98386ba238b80edf39c3163",
 "config_sha256": "f8b7898c46605d04df68c451b95026b0f8bcff5e6cbacb4c5ea162eadee0ba70",
 "task_sheet": "87f6ac5e52f6e0666873984cb40687d5daa39d01ab67cd06b911d114d2f916b0",
 "a03_config": "c2e90d605316a2c7d9679728464dbf3a534a7eafc6ea0760a966025be72fc8b4",
 "a03_summary": "1cc2936f05a50dbb084a25e9d6d98486883a5d881bfd08d3bb673aa0fa688cdc",
 "a03_readme": "4d72729291866683db99ebe7bd27f327af84d047e8a57ebfdd0dabb7a1be3cce",
 "a03_choices": "ed754376a49c383452e52ca0b99799d33a3c3796ec7fce5710a35abbcd6d51cd",
 "predictions": {
  "selection": {
   "center_only_s0": "8eabf236357cfbe5f5fa6034635600292ed9cba12b5e7f334fb4d2fbe9392eb4",
   "aligned_dense_s0": "b7275f9a245800fcd78547c019e1bec0db3762eb5443f59c63966120925a5840",
   "shuffled_dense_s0": "34915e7916017c155160ecd7e289558f26e230b98cddad2bad591e96b933fb30",
   "center_only_s1": "322eab0cd9c60a8bf24e3325dff1b819f6699a316a95ece3ca9a431466f1723b",
   "aligned_dense_s1": "0e78306632e40cf5d92f9d576f1b63711a064dd84034a6d7ec8bef2668c63281",
   "shuffled_dense_s1": "b42c8d0106a171685ea53f232ff740fd473d8dcc311ed15c07b5bd278d961890",
   "center_only_s2": "d874295c6aa3ed9fe73d8e4d78260d6c8b9a6962030f73991cfc04011ad2ee10",
   "aligned_dense_s2": "2196ae59e6e00b502bf171dd86b64a37677b2cb3e8185e01fbcf0da1b302d34f",
   "shuffled_dense_s2": "84b7502e92a4d6a7cd23cb3ecc6ea821f2744e72e82af0c50172c2a64df0cfd6"
  },
  "calibration": {
   "center_only_s0": "2d44a2f84b71e6a544d96885ef6f08d382d64517c954afb2736523df4dd2e66a",
   "aligned_dense_s0": "1c7a2383c8185c824c24d46b39327d06516da916f76e2a38636c0e410bb77058",
   "shuffled_dense_s0": "89d3bcd6ce169a8548242576edff523cb295e545d95bdcc312cacf8965cee35c",
   "center_only_s1": "ea38fa60f2d8a0bba6d057385747910dbca05d88d9ccbd82481e59e13a8da758",
   "aligned_dense_s1": "6b5e4242e97246a74fccd7a1eda294b45ff48cc58ad55ea158155135378b2730",
   "shuffled_dense_s1": "be84ba9ec3201c15a7a834307d1b1f1327cd053eb9f6b9a420ac2f1bb62298f7",
   "center_only_s2": "44b627d245506931bb0ad5a391388701738d7ca9a298fa7cdf1f85572e4571bc",
   "aligned_dense_s2": "f30bf78e93d56e86f14d1f5c10e043c10dc881a1ef5363946fd10f0d742b2e02",
   "shuffled_dense_s2": "e77c1e4c13c409c4176ad880fe4474b89ccc0bffcbb47d3f4a032748b9a7f21a"
  }
 },
 "a03_outputs": {
  "selection": {
   "0": "6088abc2685f6f83ecb3739db4a641f8a0cc55bb2c84e81c88381342701e7895",
   "1": "feca4e3f3249099a7a9e1b32015d9f5f1379c4ee4bf5fe0a8da19977a333a196",
   "2": "9abd7b7184e9f928aed1ec60c13ae763a7aae79d52c3eb1f3436d78bded99e81"
  },
  "calibration": {
   "0": "32442e2fee6473e47229e9fa6e04566693bff1cf550ba1a9b4c18012b5ffea79",
   "1": "79e2e1bf8c85d98f2f75bfc638aa48367800ee62cddd3fe9f9d3edb495c207ef",
   "2": "2147cb3473719abe1050a9f0a1c1625364c2e672eded253855c145cb51a9e07a"
  }
 }
}
```

## Stage record

```
{
 "started": "2026-10-14T22:07:07Z"
}
```

## Cost

```
{
 "cpu_hours_before_this_stage": 0.328320206006368,
 "gpu_hours": 0.0
}
```
