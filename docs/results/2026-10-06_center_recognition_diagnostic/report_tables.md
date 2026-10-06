# Recognition regression localisation: tables

## Native recognition per path (argmax, majority vote)

| model | path | half | native macro-F1 | wrong clips | A5C right | A4C right | SC4C right | cuts per clip |
|---|---|---|---|---|---|---|---|---|
| E1 (frozen) | F-dense | selection | 0.9780 | 3/284 | 10/11 | 87/87 | 12/13 | 2.9296 |
| E1 (frozen) | F-grid | selection | 0.9664 | 4/284 | 9/11 | 87/87 | 12/13 | 0.8592 |
| E1 (frozen) | F-dense | calibration | 0.9850 | 2/237 | 16/18 | 61/61 | 7/7 | 3.2869 |
| E1 (frozen) | F-grid | calibration | 0.9850 | 2/237 | 16/18 | 61/61 | 7/7 | 0.9916 |
| N_s0 | J-dense | selection | 0.9397 | 6/284 | 7/11 | 87/87 | 12/13 | 2.1585 |
| N_s0 | J-grid | selection | 0.9441 | 6/284 | 8/11 | 86/87 | 12/13 | 0.5775 |
| N_s0 | T-window | selection | 0.9304 | 7/284 | 7/11 | 86/87 | 12/13 | 0.0775 |
| N_s0 | J-dense | calibration | 0.9829 | 3/237 | 17/18 | 60/61 | 7/7 | 2.4726 |
| N_s0 | J-grid | calibration | 0.9900 | 2/237 | 17/18 | 61/61 | 7/7 | 0.7511 |
| N_s0 | T-window | calibration | 0.9753 | 4/237 | 16/18 | 60/61 | 7/7 | 0.0633 |
| S_s0 | J-dense | selection | 0.9304 | 7/284 | 7/11 | 86/87 | 12/13 | 2.4965 |
| S_s0 | J-grid | selection | 0.9566 | 5/284 | 9/11 | 86/87 | 12/13 | 0.6620 |
| S_s0 | T-window | selection | 0.9218 | 8/284 | 7/11 | 85/87 | 12/13 | 0.0810 |
| S_s0 | J-dense | calibration | 0.9856 | 2/237 | 17/18 | 60/61 | 7/7 | 2.5190 |
| S_s0 | J-grid | calibration | 0.9927 | 1/237 | 17/18 | 61/61 | 7/7 | 0.7764 |
| S_s0 | T-window | calibration | 0.9780 | 3/237 | 16/18 | 60/61 | 7/7 | 0.0675 |
| C_s0 | J-dense | selection | 0.9441 | 6/284 | 8/11 | 86/87 | 12/13 | 2.5176 |
| C_s0 | J-grid | selection | 0.9441 | 6/284 | 8/11 | 86/87 | 12/13 | 0.6866 |
| C_s0 | T-window | selection | 0.9353 | 7/284 | 8/11 | 85/87 | 12/13 | 0.0810 |
| C_s0 | J-dense | calibration | 0.9829 | 3/237 | 17/18 | 60/61 | 7/7 | 2.5401 |
| C_s0 | J-grid | calibration | 0.9927 | 1/237 | 17/18 | 61/61 | 7/7 | 0.8270 |
| C_s0 | T-window | calibration | 0.9753 | 4/237 | 16/18 | 60/61 | 7/7 | 0.1055 |
| N_s1 | J-dense | selection | 0.9304 | 7/284 | 7/11 | 86/87 | 12/13 | 2.3099 |
| N_s1 | J-grid | selection | 0.9441 | 6/284 | 8/11 | 86/87 | 12/13 | 0.6303 |
| N_s1 | T-window | selection | 0.9304 | 7/284 | 7/11 | 86/87 | 12/13 | 0.0704 |
| N_s1 | J-dense | calibration | 0.9927 | 1/237 | 17/18 | 61/61 | 7/7 | 2.5865 |
| N_s1 | J-grid | calibration | 0.9927 | 1/237 | 17/18 | 61/61 | 7/7 | 0.8101 |
| N_s1 | T-window | calibration | 0.9850 | 2/237 | 16/18 | 61/61 | 7/7 | 0.0549 |
| S_s1 | J-dense | selection | 0.9197 | 9/284 | 8/11 | 83/87 | 12/13 | 2.4331 |
| S_s1 | J-grid | selection | 0.9127 | 10/284 | 8/11 | 82/87 | 12/13 | 0.6373 |
| S_s1 | T-window | selection | 0.9000 | 11/284 | 7/11 | 82/87 | 12/13 | 0.0669 |
| S_s1 | J-dense | calibration | 0.9856 | 2/237 | 17/18 | 60/61 | 7/7 | 2.5696 |
| S_s1 | J-grid | calibration | 0.9927 | 1/237 | 17/18 | 61/61 | 7/7 | 0.7679 |
| S_s1 | T-window | calibration | 0.9780 | 3/237 | 16/18 | 60/61 | 7/7 | 0.0844 |
| C_s1 | J-dense | selection | 0.9477 | 6/284 | 9/11 | 85/87 | 12/13 | 2.6937 |
| C_s1 | J-grid | selection | 0.9441 | 6/284 | 8/11 | 86/87 | 12/13 | 0.7817 |
| C_s1 | T-window | selection | 0.9477 | 6/284 | 9/11 | 85/87 | 12/13 | 0.1162 |
| C_s1 | J-dense | calibration | 0.9856 | 2/237 | 17/18 | 60/61 | 7/7 | 2.7004 |
| C_s1 | J-grid | calibration | 0.9856 | 2/237 | 17/18 | 60/61 | 7/7 | 0.7932 |
| C_s1 | T-window | calibration | 0.9780 | 3/237 | 16/18 | 60/61 | 7/7 | 0.0970 |
| N_s2 | J-dense | selection | 0.9272 | 8/284 | 8/11 | 84/87 | 12/13 | 2.3204 |
| N_s2 | J-grid | selection | 0.9127 | 10/284 | 8/11 | 82/87 | 12/13 | 0.6373 |
| N_s2 | T-window | selection | 0.9140 | 9/284 | 7/11 | 84/87 | 12/13 | 0.1232 |
| N_s2 | J-dense | calibration | 0.9829 | 3/237 | 17/18 | 60/61 | 7/7 | 2.6414 |
| N_s2 | J-grid | calibration | 0.9873 | 3/237 | 17/18 | 61/61 | 7/7 | 0.7764 |
| N_s2 | T-window | calibration | 0.9753 | 4/237 | 16/18 | 60/61 | 7/7 | 0.0717 |
| S_s2 | J-dense | selection | 0.9244 | 9/284 | 9/11 | 82/87 | 12/13 | 2.5563 |
| S_s2 | J-grid | selection | 0.9177 | 10/284 | 9/11 | 81/87 | 12/13 | 0.7324 |
| S_s2 | T-window | selection | 0.9067 | 10/284 | 7/11 | 83/87 | 12/13 | 0.1303 |
| S_s2 | J-dense | calibration | 0.9927 | 1/237 | 17/18 | 61/61 | 7/7 | 2.8186 |
| S_s2 | J-grid | calibration | 0.9927 | 1/237 | 17/18 | 61/61 | 7/7 | 0.7595 |
| S_s2 | T-window | calibration | 0.9780 | 3/237 | 16/18 | 60/61 | 7/7 | 0.1139 |
| C_s2 | J-dense | selection | 0.9664 | 4/284 | 9/11 | 87/87 | 12/13 | 2.7535 |
| C_s2 | J-grid | selection | 0.9682 | 4/284 | 10/11 | 86/87 | 12/13 | 0.7430 |
| C_s2 | T-window | selection | 0.9414 | 6/284 | 9/11 | 85/87 | 12/13 | 0.1056 |
| C_s2 | J-dense | calibration | 0.9829 | 3/237 | 17/18 | 60/61 | 7/7 | 2.8101 |
| C_s2 | J-grid | calibration | 0.9873 | 3/237 | 17/18 | 61/61 | 7/7 | 0.8481 |
| C_s2 | T-window | calibration | 0.9753 | 4/237 | 16/18 | 60/61 | 7/7 | 0.1055 |

## Paired native comparisons (session bootstrap)

| model | comparison | d native macro-F1 | fixed / new clips | sessions fixed / new | new by class | fixed by class |
|---|---|---|---|---|---|---|
| N_s0 | F-grid - F-dense|selection | -0.0116 [-0.0422, +0.0000] | 0 / 1 | 0 / 1 | {"A5C": 1} | {} |
| N_s0 | J-grid - J-dense|selection | +0.0044 [-0.0203, +0.0452] | 1 / 1 | 1 / 1 | {"A4C": 1} | {"A5C": 1} |
| N_s0 | J-grid - F-grid|selection | -0.0223 [-0.0661, +0.0000] | 0 / 2 | 0 / 1 | {"A5C": 1, "A4C": 1} | {} |
| N_s0 | J-dense - F-dense|selection | -0.0383 [-0.0935, +0.0000] | 0 / 3 | 0 / 3 | {"A5C": 3} | {} |
| N_s0 | T-window - J-grid|selection | -0.0137 [-0.0625, +0.0169] | 1 / 2 | 1 / 2 | {"A4C": 1, "A5C": 1} | {"A4C": 1} |
| N_s0 | F-grid - F-dense|calibration | +0.0000 [+0.0000, +0.0000] | 0 / 0 | 0 / 0 | {} | {} |
| N_s0 | J-grid - J-dense|calibration | +0.0070 [+0.0000, +0.0242] | 1 / 0 | 1 / 0 | {} | {"A4C": 1} |
| N_s0 | J-grid - F-grid|calibration | +0.0050 [-0.0071, +0.0196] | 1 / 1 | 1 / 1 | {"PSAX": 1} | {"A5C": 1} |
| N_s0 | J-dense - F-dense|calibration | -0.0021 [-0.0258, +0.0168] | 1 / 2 | 1 / 2 | {"PSAX": 1, "A4C": 1} | {"A5C": 1} |
| N_s0 | T-window - J-grid|calibration | -0.0147 [-0.0389, +0.0000] | 0 / 2 | 0 / 2 | {"A5C": 1, "A4C": 1} | {} |
| S_s0 | F-grid - F-dense|selection | -0.0116 [-0.0422, +0.0000] | 0 / 1 | 0 / 1 | {"A5C": 1} | {} |
| S_s0 | J-grid - J-dense|selection | +0.0263 [+0.0000, +0.0727] | 2 / 0 | 2 / 0 | {} | {"A5C": 2} |
| S_s0 | J-grid - F-grid|selection | -0.0097 [-0.0351, +0.0000] | 1 / 2 | 1 / 2 | {"A5C": 1, "A4C": 1} | {"A5C": 1} |
| S_s0 | J-dense - F-dense|selection | -0.0476 [-0.1011, -0.0105] | 0 / 4 | 0 / 4 | {"A5C": 3, "A4C": 1} | {} |
| S_s0 | T-window - J-grid|selection | -0.0348 [-0.0798, +0.0000] | 0 / 3 | 0 / 3 | {"A4C": 1, "A5C": 2} | {} |
| S_s0 | F-grid - F-dense|calibration | +0.0000 [+0.0000, +0.0000] | 0 / 0 | 0 / 0 | {} | {} |
| S_s0 | J-grid - J-dense|calibration | +0.0070 [+0.0000, +0.0242] | 1 / 0 | 1 / 0 | {} | {"A4C": 1} |
| S_s0 | J-grid - F-grid|calibration | +0.0077 [+0.0000, +0.0213] | 1 / 0 | 1 / 0 | {} | {"A5C": 1} |
| S_s0 | J-dense - F-dense|calibration | +0.0006 [-0.0214, +0.0193] | 1 / 1 | 1 / 1 | {"A4C": 1} | {"A5C": 1} |
| S_s0 | T-window - J-grid|calibration | -0.0147 [-0.0389, +0.0000] | 0 / 2 | 0 / 2 | {"A5C": 1, "A4C": 1} | {} |
| C_s0 | F-grid - F-dense|selection | -0.0116 [-0.0422, +0.0000] | 0 / 1 | 0 / 1 | {"A5C": 1} | {} |
| C_s0 | J-grid - J-dense|selection | +0.0000 [-0.0542, +0.0564] | 2 / 2 | 2 / 1 | {"A5C": 1, "A4C": 1} | {"A4C": 1, "A5C": 1} |
| C_s0 | J-grid - F-grid|selection | -0.0223 [-0.0661, +0.0000] | 0 / 2 | 0 / 1 | {"A5C": 1, "A4C": 1} | {} |
| C_s0 | J-dense - F-dense|selection | -0.0339 [-0.0833, +0.0000] | 0 / 3 | 0 / 3 | {"A5C": 2, "A4C": 1} | {} |
| C_s0 | T-window - J-grid|selection | -0.0088 [-0.0710, +0.0473] | 2 / 3 | 1 / 3 | {"A4C": 2, "A5C": 1} | {"A5C": 1, "A4C": 1} |
| C_s0 | F-grid - F-dense|calibration | +0.0000 [+0.0000, +0.0000] | 0 / 0 | 0 / 0 | {} | {} |
| C_s0 | J-grid - J-dense|calibration | +0.0097 [+0.0000, +0.0289] | 2 / 0 | 2 / 0 | {} | {"PSAX": 1, "A4C": 1} |
| C_s0 | J-grid - F-grid|calibration | +0.0077 [+0.0000, +0.0213] | 1 / 0 | 1 / 0 | {} | {"A5C": 1} |
| C_s0 | J-dense - F-dense|calibration | -0.0021 [-0.0258, +0.0168] | 1 / 2 | 1 / 2 | {"PSAX": 1, "A4C": 1} | {"A5C": 1} |
| C_s0 | T-window - J-grid|calibration | -0.0173 [-0.0428, +0.0000] | 0 / 3 | 0 / 3 | {"PSAX": 1, "A5C": 1, "A4C": 1} | {} |
| N_s1 | F-grid - F-dense|selection | -0.0116 [-0.0422, +0.0000] | 0 / 1 | 0 / 1 | {"A5C": 1} | {} |
| N_s1 | J-grid - J-dense|selection | +0.0137 [-0.0172, +0.0607] | 2 / 1 | 2 / 1 | {"A4C": 1} | {"A4C": 1, "A5C": 1} |
| N_s1 | J-grid - F-grid|selection | -0.0223 [-0.0661, +0.0000] | 0 / 2 | 0 / 1 | {"A5C": 1, "A4C": 1} | {} |
| N_s1 | J-dense - F-dense|selection | -0.0476 [-0.1038, -0.0106] | 0 / 4 | 0 / 4 | {"A5C": 3, "A4C": 1} | {} |
| N_s1 | T-window - J-grid|selection | -0.0137 [-0.0625, +0.0169] | 1 / 2 | 1 / 2 | {"A4C": 1, "A5C": 1} | {"A4C": 1} |
| N_s1 | F-grid - F-dense|calibration | +0.0000 [+0.0000, +0.0000] | 0 / 0 | 0 / 0 | {} | {} |
| N_s1 | J-grid - J-dense|calibration | +0.0000 [+0.0000, +0.0000] | 0 / 0 | 0 / 0 | {} | {} |
| N_s1 | J-grid - F-grid|calibration | +0.0077 [+0.0000, +0.0213] | 1 / 0 | 1 / 0 | {} | {"A5C": 1} |
| N_s1 | J-dense - F-dense|calibration | +0.0077 [+0.0000, +0.0213] | 1 / 0 | 1 / 0 | {} | {"A5C": 1} |
| N_s1 | T-window - J-grid|calibration | -0.0077 [-0.0288, +0.0000] | 0 / 1 | 0 / 1 | {"A5C": 1} | {} |
| S_s1 | F-grid - F-dense|selection | -0.0116 [-0.0422, +0.0000] | 0 / 1 | 0 / 1 | {"A5C": 1} | {} |
| S_s1 | J-grid - J-dense|selection | -0.0070 [-0.0533, +0.0396] | 1 / 2 | 1 / 1 | {"A5C": 1, "A4C": 1} | {"A5C": 1} |
| S_s1 | J-grid - F-grid|selection | -0.0537 [-0.1131, +0.0000] | 0 / 6 | 0 / 2 | {"A5C": 1, "A4C": 5} | {} |
| S_s1 | J-dense - F-dense|selection | -0.0583 [-0.1313, +0.0000] | 0 / 6 | 0 / 3 | {"A5C": 2, "A4C": 4} | {} |
| S_s1 | T-window - J-grid|selection | -0.0127 [-0.0542, +0.0141] | 1 / 2 | 1 / 2 | {"A4C": 1, "A5C": 1} | {"A4C": 1} |
| S_s1 | F-grid - F-dense|calibration | +0.0000 [+0.0000, +0.0000] | 0 / 0 | 0 / 0 | {} | {} |
| S_s1 | J-grid - J-dense|calibration | +0.0070 [+0.0000, +0.0242] | 1 / 0 | 1 / 0 | {} | {"A4C": 1} |
| S_s1 | J-grid - F-grid|calibration | +0.0077 [+0.0000, +0.0213] | 1 / 0 | 1 / 0 | {} | {"A5C": 1} |
| S_s1 | J-dense - F-dense|calibration | +0.0006 [-0.0214, +0.0193] | 1 / 1 | 1 / 1 | {"A4C": 1} | {"A5C": 1} |
| S_s1 | T-window - J-grid|calibration | -0.0147 [-0.0389, +0.0000] | 0 / 2 | 0 / 2 | {"A5C": 1, "A4C": 1} | {} |
| C_s1 | F-grid - F-dense|selection | -0.0116 [-0.0422, +0.0000] | 0 / 1 | 0 / 1 | {"A5C": 1} | {} |
| C_s1 | J-grid - J-dense|selection | -0.0036 [-0.0534, +0.0481] | 2 / 2 | 2 / 1 | {"A5C": 1, "A4C": 1} | {"A4C": 2} |
| C_s1 | J-grid - F-grid|selection | -0.0223 [-0.0661, +0.0000] | 0 / 2 | 0 / 1 | {"A5C": 1, "A4C": 1} | {} |
| C_s1 | J-dense - F-dense|selection | -0.0303 [-0.0735, +0.0000] | 0 / 3 | 0 / 3 | {"A5C": 1, "A4C": 2} | {} |
| C_s1 | T-window - J-grid|selection | +0.0036 [-0.0478, +0.0552] | 2 / 2 | 1 / 2 | {"A4C": 2} | {"A5C": 1, "A4C": 1} |
| C_s1 | F-grid - F-dense|calibration | +0.0000 [+0.0000, +0.0000] | 0 / 0 | 0 / 0 | {} | {} |
| C_s1 | J-grid - J-dense|calibration | +0.0000 [-0.0195, +0.0197] | 1 / 1 | 1 / 1 | {"A4C": 1} | {"A4C": 1} |
| C_s1 | J-grid - F-grid|calibration | +0.0006 [-0.0215, +0.0195] | 1 / 1 | 1 / 1 | {"A4C": 1} | {"A5C": 1} |
| C_s1 | J-dense - F-dense|calibration | +0.0006 [-0.0214, +0.0193] | 1 / 1 | 1 / 1 | {"A4C": 1} | {"A5C": 1} |
| C_s1 | T-window - J-grid|calibration | -0.0076 [-0.0355, +0.0155] | 1 / 2 | 1 / 2 | {"A5C": 1, "A4C": 1} | {"A4C": 1} |
| N_s2 | F-grid - F-dense|selection | -0.0116 [-0.0422, +0.0000] | 0 / 1 | 0 / 1 | {"A5C": 1} | {} |
| N_s2 | J-grid - J-dense|selection | -0.0145 [-0.0589, +0.0349] | 1 / 3 | 1 / 2 | {"A5C": 1, "A4C": 2} | {"A5C": 1} |
| N_s2 | J-grid - F-grid|selection | -0.0537 [-0.1131, +0.0000] | 0 / 6 | 0 / 2 | {"A5C": 1, "A4C": 5} | {} |
| N_s2 | J-dense - F-dense|selection | -0.0508 [-0.1175, +0.0000] | 1 / 6 | 1 / 4 | {"A5C": 3, "A4C": 3} | {"A5C": 1} |
| N_s2 | T-window - J-grid|selection | +0.0013 [-0.0487, +0.0319] | 3 / 2 | 2 / 2 | {"A4C": 1, "A5C": 1} | {"A4C": 3} |
| N_s2 | F-grid - F-dense|calibration | +0.0000 [+0.0000, +0.0000] | 0 / 0 | 0 / 0 | {} | {} |
| N_s2 | J-grid - J-dense|calibration | +0.0044 [-0.0077, +0.0220] | 1 / 1 | 1 / 1 | {"PSAX": 1} | {"A4C": 1} |
| N_s2 | J-grid - F-grid|calibration | +0.0023 [-0.0147, +0.0189] | 1 / 2 | 1 / 1 | {"PSAX": 2} | {"A5C": 1} |
| N_s2 | J-dense - F-dense|calibration | -0.0021 [-0.0258, +0.0168] | 1 / 2 | 1 / 2 | {"PSAX": 1, "A4C": 1} | {"A5C": 1} |
| N_s2 | T-window - J-grid|calibration | -0.0120 [-0.0364, +0.0054] | 1 / 2 | 1 / 2 | {"A5C": 1, "A4C": 1} | {"PSAX": 1} |
| S_s2 | F-grid - F-dense|selection | -0.0116 [-0.0422, +0.0000] | 0 / 1 | 0 / 1 | {"A5C": 1} | {} |
| S_s2 | J-grid - J-dense|selection | -0.0068 [-0.0543, +0.0418] | 2 / 3 | 2 / 2 | {"A5C": 1, "A4C": 2} | {"A4C": 1, "A5C": 1} |
| S_s2 | J-grid - F-grid|selection | -0.0487 [-0.1085, +0.0000] | 1 / 7 | 1 / 3 | {"A5C": 1, "A4C": 6} | {"A5C": 1} |
| S_s2 | J-dense - F-dense|selection | -0.0535 [-0.1343, +0.0118] | 1 / 7 | 1 / 4 | {"A5C": 2, "A4C": 5} | {"A5C": 1} |
| S_s2 | T-window - J-grid|selection | -0.0110 [-0.0599, +0.0298] | 4 / 4 | 3 / 4 | {"A4C": 2, "A5C": 2} | {"A4C": 4} |
| S_s2 | F-grid - F-dense|calibration | +0.0000 [+0.0000, +0.0000] | 0 / 0 | 0 / 0 | {} | {} |
| S_s2 | J-grid - J-dense|calibration | +0.0000 [+0.0000, +0.0000] | 0 / 0 | 0 / 0 | {} | {} |
| S_s2 | J-grid - F-grid|calibration | +0.0077 [+0.0000, +0.0213] | 1 / 0 | 1 / 0 | {} | {"A5C": 1} |
| S_s2 | J-dense - F-dense|calibration | +0.0077 [+0.0000, +0.0213] | 1 / 0 | 1 / 0 | {} | {"A5C": 1} |
| S_s2 | T-window - J-grid|calibration | -0.0147 [-0.0389, +0.0000] | 0 / 2 | 0 / 2 | {"A5C": 1, "A4C": 1} | {} |
| C_s2 | F-grid - F-dense|selection | -0.0116 [-0.0422, +0.0000] | 0 / 1 | 0 / 1 | {"A5C": 1} | {} |
| C_s2 | J-grid - J-dense|selection | +0.0018 [-0.0236, +0.0427] | 1 / 1 | 1 / 1 | {"A4C": 1} | {"A5C": 1} |
| C_s2 | J-grid - F-grid|selection | +0.0018 [+0.0000, +0.0091] | 1 / 1 | 1 / 1 | {"A4C": 1} | {"A5C": 1} |
| C_s2 | J-dense - F-dense|selection | -0.0116 [-0.0642, +0.0249] | 1 / 2 | 1 / 2 | {"A5C": 2} | {"A5C": 1} |
| C_s2 | T-window - J-grid|selection | -0.0267 [-0.0807, +0.0094] | 1 / 3 | 1 / 3 | {"A4C": 2, "A5C": 1} | {"A4C": 1} |
| C_s2 | F-grid - F-dense|calibration | +0.0000 [+0.0000, +0.0000] | 0 / 0 | 0 / 0 | {} | {} |
| C_s2 | J-grid - J-dense|calibration | +0.0044 [-0.0075, +0.0221] | 1 / 1 | 1 / 1 | {"PSAX": 1} | {"A4C": 1} |
| C_s2 | J-grid - F-grid|calibration | +0.0023 [-0.0115, +0.0173] | 1 / 2 | 1 / 2 | {"PSAX": 2} | {"A5C": 1} |
| C_s2 | J-dense - F-dense|calibration | -0.0021 [-0.0258, +0.0168] | 1 / 2 | 1 / 2 | {"PSAX": 1, "A4C": 1} | {"A5C": 1} |
| C_s2 | T-window - J-grid|calibration | -0.0120 [-0.0363, +0.0052] | 1 / 2 | 1 / 2 | {"A5C": 1, "A4C": 1} | {"PSAX": 1} |
| E1 (frozen) | F-grid - F-dense|selection | -0.0116 [-0.0422, +0.0000] | 0 / 1 | 0 / 1 | {"A5C": 1} | {} |
| E1 (frozen) | F-grid - F-dense|calibration | +0.0000 [+0.0000, +0.0000] | 0 / 0 | 0 / 0 | {} | {} |

## Eight cells of the C models (F-grid, J-grid, T-window right?)

| model | half | F+ J+ T+ | F+ J+ T- | F+ J- T+ | F+ J- T- | F- J+ T+ | F- J+ T- | F- J- T+ | F- J- T- |
|---|---|---|---|---|---|---|---|---|---|
| C_s0 | selection | 275 | 3 | 2 | 0 | 0 | 0 | 0 | 4 |
| C_s0 | calibration | 232 | 3 | 0 | 0 | 1 | 0 | 0 | 1 |
| C_s1 | selection | 276 | 2 | 2 | 0 | 0 | 0 | 0 | 4 |
| C_s1 | calibration | 232 | 2 | 1 | 0 | 1 | 0 | 0 | 1 |
| C_s2 | selection | 276 | 3 | 1 | 0 | 1 | 0 | 0 | 3 |
| C_s2 | calibration | 231 | 2 | 1 | 1 | 1 | 0 | 0 | 1 |

## Target-sample error at window centres

| model | half | windows | group | F error (rec-weighted) | J error (rec-weighted) | T error (rec-weighted) |
|---|---|---|---|---|---|---|
| N_s0 | selection | 7666 | all | 0.0698 (0.0432) | 0.0651 (0.0393) | 0.0800 (0.0491) |
| N_s0 | selection | 5312 | single recording | 0.0674 (0.0422) | 0.0625 (0.0401) | 0.0523 (0.0255) |
| N_s0 | selection | 1449 | mixed views | 0.0787 (0.0387) | 0.0676 (0.0281) | 0.1994 (0.1350) |
| N_s0 | calibration | 7021 | all | 0.0534 (0.0454) | 0.0503 (0.0404) | 0.0523 (0.0454) |
| N_s0 | calibration | 4783 | single recording | 0.0516 (0.0460) | 0.0470 (0.0390) | 0.0253 (0.0150) |
| N_s0 | calibration | 1440 | mixed views | 0.0604 (0.0529) | 0.0569 (0.0450) | 0.1611 (0.1469) |
| S_s0 | selection | 7666 | all | 0.0698 (0.0432) | 0.0669 (0.0420) | 0.0610 (0.0329) |
| S_s0 | selection | 5312 | single recording | 0.0674 (0.0422) | 0.0634 (0.0424) | 0.0476 (0.0272) |
| S_s0 | selection | 1449 | mixed views | 0.0787 (0.0387) | 0.0752 (0.0364) | 0.1187 (0.0598) |
| S_s0 | calibration | 7021 | all | 0.0534 (0.0454) | 0.0456 (0.0376) | 0.0285 (0.0205) |
| S_s0 | calibration | 4783 | single recording | 0.0516 (0.0460) | 0.0424 (0.0351) | 0.0194 (0.0115) |
| S_s0 | calibration | 1440 | mixed views | 0.0604 (0.0529) | 0.0500 (0.0434) | 0.0667 (0.0586) |
| C_s0 | selection | 7666 | all | 0.0698 (0.0432) | 0.0672 (0.0424) | 0.0524 (0.0268) |
| C_s0 | selection | 5312 | single recording | 0.0674 (0.0422) | 0.0644 (0.0416) | 0.0465 (0.0267) |
| C_s0 | selection | 1449 | mixed views | 0.0787 (0.0387) | 0.0738 (0.0345) | 0.0766 (0.0317) |
| C_s0 | calibration | 7021 | all | 0.0534 (0.0454) | 0.0474 (0.0386) | 0.0231 (0.0160) |
| C_s0 | calibration | 4783 | single recording | 0.0516 (0.0460) | 0.0456 (0.0371) | 0.0199 (0.0126) |
| C_s0 | calibration | 1440 | mixed views | 0.0604 (0.0529) | 0.0542 (0.0484) | 0.0382 (0.0295) |
| N_s1 | selection | 7666 | all | 0.0698 (0.0432) | 0.0674 (0.0397) | 0.0740 (0.0437) |
| N_s1 | selection | 5312 | single recording | 0.0674 (0.0422) | 0.0634 (0.0396) | 0.0497 (0.0247) |
| N_s1 | selection | 1449 | mixed views | 0.0787 (0.0387) | 0.0752 (0.0306) | 0.1801 (0.1030) |
| N_s1 | calibration | 7021 | all | 0.0534 (0.0454) | 0.0473 (0.0376) | 0.0369 (0.0308) |
| N_s1 | calibration | 4783 | single recording | 0.0516 (0.0460) | 0.0439 (0.0338) | 0.0157 (0.0078) |
| N_s1 | calibration | 1440 | mixed views | 0.0604 (0.0529) | 0.0542 (0.0432) | 0.1215 (0.1059) |
| S_s1 | selection | 7666 | all | 0.0698 (0.0432) | 0.0733 (0.0470) | 0.0687 (0.0437) |
| S_s1 | selection | 5312 | single recording | 0.0674 (0.0422) | 0.0719 (0.0483) | 0.0606 (0.0360) |
| S_s1 | selection | 1449 | mixed views | 0.0787 (0.0387) | 0.0780 (0.0395) | 0.1159 (0.0705) |
| S_s1 | calibration | 7021 | all | 0.0534 (0.0454) | 0.0439 (0.0348) | 0.0261 (0.0191) |
| S_s1 | calibration | 4783 | single recording | 0.0516 (0.0460) | 0.0412 (0.0306) | 0.0171 (0.0102) |
| S_s1 | calibration | 1440 | mixed views | 0.0604 (0.0529) | 0.0458 (0.0435) | 0.0639 (0.0513) |
| C_s1 | selection | 7666 | all | 0.0698 (0.0432) | 0.0672 (0.0421) | 0.0496 (0.0239) |
| C_s1 | selection | 5312 | single recording | 0.0674 (0.0422) | 0.0659 (0.0420) | 0.0454 (0.0238) |
| C_s1 | selection | 1449 | mixed views | 0.0787 (0.0387) | 0.0725 (0.0364) | 0.0690 (0.0269) |
| C_s1 | calibration | 7021 | all | 0.0534 (0.0454) | 0.0423 (0.0346) | 0.0182 (0.0126) |
| C_s1 | calibration | 4783 | single recording | 0.0516 (0.0460) | 0.0408 (0.0325) | 0.0171 (0.0104) |
| C_s1 | calibration | 1440 | mixed views | 0.0604 (0.0529) | 0.0458 (0.0444) | 0.0250 (0.0222) |
| N_s2 | selection | 7666 | all | 0.0698 (0.0432) | 0.0672 (0.0431) | 0.0941 (0.0640) |
| N_s2 | selection | 5312 | single recording | 0.0674 (0.0422) | 0.0636 (0.0431) | 0.0563 (0.0326) |
| N_s2 | selection | 1449 | mixed views | 0.0787 (0.0387) | 0.0752 (0.0330) | 0.2616 (0.1680) |
| N_s2 | calibration | 7021 | all | 0.0534 (0.0454) | 0.0479 (0.0395) | 0.0645 (0.0473) |
| N_s2 | calibration | 4783 | single recording | 0.0516 (0.0460) | 0.0456 (0.0358) | 0.0234 (0.0157) |
| N_s2 | calibration | 1440 | mixed views | 0.0604 (0.0529) | 0.0507 (0.0456) | 0.2271 (0.1561) |
| S_s2 | selection | 7666 | all | 0.0698 (0.0432) | 0.0732 (0.0504) | 0.0686 (0.0439) |
| S_s2 | selection | 5312 | single recording | 0.0674 (0.0422) | 0.0712 (0.0501) | 0.0561 (0.0362) |
| S_s2 | selection | 1449 | mixed views | 0.0787 (0.0387) | 0.0801 (0.0446) | 0.1304 (0.0770) |
| S_s2 | calibration | 7021 | all | 0.0534 (0.0454) | 0.0446 (0.0355) | 0.0269 (0.0201) |
| S_s2 | calibration | 4783 | single recording | 0.0516 (0.0460) | 0.0433 (0.0325) | 0.0184 (0.0110) |
| S_s2 | calibration | 1440 | mixed views | 0.0604 (0.0529) | 0.0451 (0.0433) | 0.0632 (0.0520) |
| C_s2 | selection | 7666 | all | 0.0698 (0.0432) | 0.0670 (0.0452) | 0.0477 (0.0260) |
| C_s2 | selection | 5312 | single recording | 0.0674 (0.0422) | 0.0636 (0.0444) | 0.0431 (0.0255) |
| C_s2 | selection | 1449 | mixed views | 0.0787 (0.0387) | 0.0759 (0.0368) | 0.0725 (0.0307) |
| C_s2 | calibration | 7021 | all | 0.0534 (0.0454) | 0.0494 (0.0429) | 0.0232 (0.0178) |
| C_s2 | calibration | 4783 | single recording | 0.0516 (0.0460) | 0.0479 (0.0411) | 0.0207 (0.0141) |
| C_s2 | calibration | 1440 | mixed views | 0.0604 (0.0529) | 0.0514 (0.0484) | 0.0361 (0.0325) |

## Supplementary: one archived E1 + B3 setting on every path

| system | native_macro_f1 | segment_f1_50_multi | boundary_recall_multi | boundary_precision_multi | native_internal_cuts_per_clip |
|---|---|---|---|---|---|
| F|F-dense|selection | 0.9780 | 0.8940 | 0.9555 | 0.9432 | 0.0458 |
| F|F-grid|selection | 0.9664 | 0.8651 | 0.9555 | 0.8629 | 0.1197 |
| F|F-dense|calibration | 0.9856 | 0.9589 | 0.9774 | 0.9511 | 0.0675 |
| F|F-grid|calibration | 0.9770 | 0.9276 | 0.9673 | 0.9059 | 0.1983 |
| N_s0|J-dense|selection | 0.9397 | 0.8508 | 0.9555 | 0.8753 | 0.1232 |
| N_s0|J-grid|selection | 0.9566 | 0.8316 | 0.9686 | 0.8168 | 0.1866 |
| N_s0|T-window|selection | 0.9304 | 0.8199 | 0.8822 | 0.8321 | 0.0493 |
| N_s0|J-dense|calibration | 0.9753 | 0.9602 | 0.9874 | 0.9402 | 0.1477 |
| N_s0|J-grid|calibration | 0.9850 | 0.9048 | 0.9799 | 0.8590 | 0.2658 |
| N_s0|T-window|calibration | 0.9753 | 0.9434 | 0.9397 | 0.9235 | 0.0464 |
| S_s0|J-dense|selection | 0.9304 | 0.8610 | 0.9634 | 0.8741 | 0.1197 |
| S_s0|J-grid|selection | 0.9566 | 0.8326 | 0.9686 | 0.8061 | 0.2218 |
| S_s0|T-window|selection | 0.9304 | 0.2355 | 0.0759 | 0.5918 | 0.0000 |
| S_s0|J-dense|calibration | 0.9780 | 0.9602 | 0.9899 | 0.9426 | 0.1603 |
| S_s0|J-grid|calibration | 0.9927 | 0.9171 | 0.9899 | 0.8795 | 0.2954 |
| S_s0|T-window|calibration | 0.9780 | 0.2219 | 0.0653 | 0.6842 | 0.0042 |
| C_s0|J-dense|selection | 0.9272 | 0.8690 | 0.9712 | 0.8918 | 0.1232 |
| C_s0|J-grid|selection | 0.9441 | 0.8417 | 0.9660 | 0.8164 | 0.1655 |
| C_s0|T-window|selection | 0.9218 | 0.8539 | 0.9660 | 0.8682 | 0.0599 |
| C_s0|J-dense|calibration | 0.9753 | 0.9592 | 0.9874 | 0.9424 | 0.1477 |
| C_s0|J-grid|calibration | 0.9829 | 0.9242 | 0.9874 | 0.8851 | 0.2321 |
| C_s0|T-window|calibration | 0.9753 | 0.9684 | 0.9874 | 0.9656 | 0.0506 |
| N_s1|J-dense|selection | 0.9304 | 0.8583 | 0.9555 | 0.8946 | 0.0951 |
| N_s1|J-grid|selection | 0.9441 | 0.8147 | 0.9607 | 0.7927 | 0.1655 |
| N_s1|T-window|selection | 0.9304 | 0.8243 | 0.9084 | 0.8382 | 0.0634 |
| N_s1|J-dense|calibration | 0.9850 | 0.9499 | 0.9799 | 0.9308 | 0.1646 |
| N_s1|J-grid|calibration | 0.9850 | 0.9131 | 0.9849 | 0.8829 | 0.3249 |
| N_s1|T-window|calibration | 0.9850 | 0.9644 | 0.9548 | 0.9476 | 0.0422 |
| S_s1|J-dense|selection | 0.9197 | 0.8602 | 0.9607 | 0.8843 | 0.1197 |
| S_s1|J-grid|selection | 0.9244 | 0.8295 | 0.9686 | 0.7991 | 0.1831 |
| S_s1|T-window|selection | 0.9197 | 0.2655 | 0.1178 | 0.8036 | 0.0035 |
| S_s1|J-dense|calibration | 0.9780 | 0.9546 | 0.9849 | 0.9333 | 0.1435 |
| S_s1|J-grid|calibration | 0.9927 | 0.9214 | 0.9899 | 0.8894 | 0.3165 |
| S_s1|T-window|calibration | 0.9856 | 0.2595 | 0.0854 | 0.7083 | 0.0000 |
| C_s1|J-dense|selection | 0.9197 | 0.8824 | 0.9686 | 0.9091 | 0.1232 |
| C_s1|J-grid|selection | 0.9441 | 0.8444 | 0.9712 | 0.8226 | 0.1514 |
| C_s1|T-window|selection | 0.9591 | 0.8566 | 0.9686 | 0.8585 | 0.0810 |
| C_s1|J-dense|calibration | 0.9780 | 0.9639 | 0.9925 | 0.9495 | 0.0886 |
| C_s1|J-grid|calibration | 0.9927 | 0.9372 | 0.9824 | 0.9114 | 0.1730 |
| C_s1|T-window|calibration | 0.9780 | 0.9724 | 0.9925 | 0.9564 | 0.0633 |
| N_s2|J-dense|selection | 0.9067 | 0.8571 | 0.9607 | 0.8865 | 0.1197 |
| N_s2|J-grid|selection | 0.9394 | 0.8189 | 0.9634 | 0.8088 | 0.1655 |
| N_s2|T-window|selection | 0.9067 | 0.8086 | 0.7958 | 0.7657 | 0.0669 |
| N_s2|J-dense|calibration | 0.9674 | 0.9453 | 0.9849 | 0.9289 | 0.1013 |
| N_s2|J-grid|calibration | 0.9873 | 0.9167 | 0.9849 | 0.8869 | 0.2447 |
| N_s2|T-window|calibration | 0.9753 | 0.9135 | 0.8141 | 0.8060 | 0.0549 |
| S_s2|J-dense|selection | 0.9127 | 0.8654 | 0.9634 | 0.8954 | 0.1056 |
| S_s2|J-grid|selection | 0.9244 | 0.8263 | 0.9791 | 0.7924 | 0.1796 |
| S_s2|T-window|selection | 0.9272 | 0.2374 | 0.0916 | 0.6731 | 0.0000 |
| S_s2|J-dense|calibration | 0.9780 | 0.9602 | 0.9874 | 0.9447 | 0.1477 |
| S_s2|J-grid|calibration | 0.9927 | 0.9268 | 0.9874 | 0.8912 | 0.2447 |
| S_s2|T-window|calibration | 0.9856 | 0.2504 | 0.0854 | 0.6939 | 0.0042 |
| C_s2|J-dense|selection | 0.9477 | 0.8908 | 0.9686 | 0.9158 | 0.1232 |
| C_s2|J-grid|selection | 0.9682 | 0.8660 | 0.9738 | 0.8360 | 0.1831 |
| C_s2|T-window|selection | 0.9414 | 0.8634 | 0.9712 | 0.8588 | 0.0880 |
| C_s2|J-dense|calibration | 0.9753 | 0.9565 | 0.9950 | 0.9429 | 0.1350 |
| C_s2|J-grid|calibration | 0.9900 | 0.9331 | 0.9925 | 0.9018 | 0.2236 |
| C_s2|T-window|calibration | 0.9753 | 0.9733 | 0.9925 | 0.9634 | 0.0506 |
