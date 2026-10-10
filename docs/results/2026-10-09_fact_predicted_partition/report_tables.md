# Predicted-partition FACT: tables

Recompute: `python scripts/report_fact_predicted_partition.py --config configs/experiments/fact_predicted_partition_v1.yaml --evaluate`

## Systems

| system | half | segment F1@50 | boundary R | boundary P | native macro-F1 | switches per clip | same-view cut rate | A5C read as A4C | A4C read as A5C |
|---|---|---|---|---|---|---|---|---|---|
| Original FACT s0 | selection | 0.9062 | 0.9476 | 0.9258 | 0.9886 | 0.0000 | 0.0522 | 0 | 0 |
| Original FACT s0 | calibration | 0.9674 | 0.9824 | 0.9678 | 0.9856 | 0.0338 | 0.0249 | 1 | 1 |
| Uniform s0 | selection | 0.8930 | 0.9581 | 0.9015 | 0.9780 | 0.0070 | 0.0739 | 1 | 0 |
| Uniform s0 | calibration | 0.9648 | 0.9799 | 0.9443 | 0.9929 | 0.0211 | 0.0249 | 0 | 1 |
| Predicted s0 | selection | 0.8833 | 0.9581 | 0.9015 | 0.9780 | 0.0000 | 0.0696 | 1 | 0 |
| Predicted s0 | calibration | 0.9637 | 0.9874 | 0.9632 | 0.9829 | 0.0253 | 0.0249 | 1 | 1 |
| ASRF s0 | selection | 0.8779 | 0.9346 | 0.9273 | 0.9304 | 0.0106 | 0.0652 | 4 | 1 |
| ASRF s0 | calibration | 0.9721 | 0.9874 | 0.9752 | 0.9856 | 0.0042 | 0.0348 | 1 | 1 |
| Boundary head + frame classifier s0 | selection | 0.8803 | 0.9267 | 0.9100 | 0.9780 | 0.0775 | 0.0565 | 1 | 0 |
| Boundary head + frame classifier s0 | calibration | 0.9564 | 0.9548 | 0.9669 | 0.9856 | 0.0802 | 0.0249 | 1 | 1 |
| Original FACT s1 | selection | 0.8983 | 0.9503 | 0.9284 | 0.9780 | 0.0000 | 0.0435 | 1 | 0 |
| Original FACT s1 | calibration | 0.9605 | 0.9749 | 0.9676 | 0.9856 | 0.0295 | 0.0299 | 1 | 1 |
| Uniform s1 | selection | 0.8895 | 0.9503 | 0.8854 | 0.9886 | 0.0000 | 0.0957 | 0 | 0 |
| Uniform s1 | calibration | 0.9600 | 0.9749 | 0.9417 | 0.9829 | 0.0211 | 0.0398 | 1 | 1 |
| Predicted s1 | selection | 0.8988 | 0.9581 | 0.9015 | 0.9664 | 0.0000 | 0.0565 | 2 | 0 |
| Predicted s1 | calibration | 0.9638 | 0.9874 | 0.9539 | 0.9829 | 0.0295 | 0.0149 | 1 | 1 |
| ASRF s1 | selection | 0.8876 | 0.9398 | 0.9373 | 0.9537 | 0.0070 | 0.0478 | 3 | 0 |
| ASRF s1 | calibration | 0.9721 | 0.9849 | 0.9776 | 0.9856 | 0.0042 | 0.0348 | 1 | 1 |
| Boundary head + frame classifier s1 | selection | 0.8946 | 0.9346 | 0.9297 | 0.9780 | 0.0951 | 0.0522 | 1 | 0 |
| Boundary head + frame classifier s1 | calibration | 0.9555 | 0.9573 | 0.9621 | 0.9856 | 0.0844 | 0.0199 | 1 | 1 |
| Original FACT s2 | selection | 0.9109 | 0.9398 | 0.9253 | 0.9664 | 0.0000 | 0.0348 | 2 | 0 |
| Original FACT s2 | calibration | 0.9653 | 0.9724 | 0.9724 | 0.9929 | 0.0338 | 0.0249 | 0 | 1 |
| Uniform s2 | selection | 0.8891 | 0.9555 | 0.8990 | 0.9886 | 0.0000 | 0.0652 | 0 | 0 |
| Uniform s2 | calibration | 0.9616 | 0.9673 | 0.9530 | 0.9856 | 0.0295 | 0.0348 | 1 | 1 |
| Predicted s2 | selection | 0.9052 | 0.9581 | 0.9127 | 0.9780 | 0.0000 | 0.0522 | 1 | 0 |
| Predicted s2 | calibration | 0.9580 | 0.9774 | 0.9488 | 0.9929 | 0.0338 | 0.0348 | 0 | 1 |
| ASRF s2 | selection | 0.8825 | 0.9424 | 0.9424 | 0.9537 | 0.0106 | 0.0652 | 3 | 0 |
| ASRF s2 | calibration | 0.9662 | 0.9824 | 0.9799 | 0.9856 | 0.0042 | 0.0299 | 1 | 1 |
| Boundary head + frame classifier s2 | selection | 0.8884 | 0.9293 | 0.9293 | 0.9780 | 0.0775 | 0.0435 | 1 | 0 |
| Boundary head + frame classifier s2 | calibration | 0.9556 | 0.9573 | 0.9573 | 0.9856 | 0.0802 | 0.0299 | 1 | 1 |
| E1 + B3 | selection | 0.8940 | 0.9555 | 0.9432 | 0.9780 | 0.0458 | 0.0391 | 1 | 0 |
| E1 + B3 | calibration | 0.9589 | 0.9774 | 0.9511 | 0.9856 | 0.0675 | 0.0199 | 1 | 1 |
| E0 + B3 | selection | 0.9344 | 0.9607 | 0.9946 | 0.9650 | 0.0141 | 0.0043 | 3 | 0 |
| E0 + B3 | calibration | 0.9340 | 0.9548 | 0.9694 | 0.9660 | 0.0338 | 0.0199 | 4 | 0 |
| B7 + B3 | selection | 0.8798 | 0.9555 | 0.8902 | 0.9566 | 0.0634 | 0.0565 | 2 | 1 |
| B7 + B3 | calibration | 0.9721 | 0.9749 | 0.9628 | 0.9927 | 0.0422 | 0.0348 | 1 | 0 |

## Comparisons (seed | comparison | half)

| comparison | d segment F1@50 |
|---|---|
| s0|Predicted - Uniform|selection | -0.0097 [-0.0346, +0.0156] |
| s0|Predicted - Uniform|calibration | -0.0011 [-0.0149, +0.0128] |
| s0|Predicted - Original FACT|selection | -0.0230 [-0.0446, -0.0024] |
| s0|Predicted - Original FACT|calibration | -0.0037 [-0.0187, +0.0133] |
| s0|Uniform - Original FACT|selection | -0.0132 [-0.0329, +0.0044] |
| s0|Uniform - Original FACT|calibration | -0.0026 [-0.0177, +0.0126] |
| s0|Predicted - ASRF|selection | +0.0054 [-0.0183, +0.0287] |
| s0|Predicted - ASRF|calibration | -0.0085 [-0.0226, +0.0058] |
| s0|Uniform - ASRF|selection | +0.0151 [-0.0091, +0.0397] |
| s0|Uniform - ASRF|calibration | -0.0073 [-0.0167, +0.0020] |
| s0|Predicted - (Boundary head + frame classifier)|selection | +0.0030 [-0.0222, +0.0291] |
| s0|Predicted - (Boundary head + frame classifier)|calibration | +0.0073 [-0.0089, +0.0256] |
| s1|Predicted - Uniform|selection | +0.0093 [-0.0142, +0.0351] |
| s1|Predicted - Uniform|calibration | +0.0038 [-0.0152, +0.0210] |
| s1|Predicted - Original FACT|selection | +0.0005 [-0.0203, +0.0226] |
| s1|Predicted - Original FACT|calibration | +0.0033 [-0.0181, +0.0239] |
| s1|Uniform - Original FACT|selection | -0.0088 [-0.0315, +0.0152] |
| s1|Uniform - Original FACT|calibration | -0.0005 [-0.0155, +0.0154] |
| s1|Predicted - ASRF|selection | +0.0113 [-0.0129, +0.0370] |
| s1|Predicted - ASRF|calibration | -0.0083 [-0.0240, +0.0069] |
| s1|Uniform - ASRF|selection | +0.0020 [-0.0200, +0.0247] |
| s1|Uniform - ASRF|calibration | -0.0121 [-0.0248, +0.0011] |
| s1|Predicted - (Boundary head + frame classifier)|selection | +0.0042 [-0.0240, +0.0345] |
| s1|Predicted - (Boundary head + frame classifier)|calibration | +0.0083 [-0.0098, +0.0273] |
| s2|Predicted - Uniform|selection | +0.0161 [-0.0048, +0.0390] |
| s2|Predicted - Uniform|calibration | -0.0036 [-0.0207, +0.0137] |
| s2|Predicted - Original FACT|selection | -0.0057 [-0.0242, +0.0142] |
| s2|Predicted - Original FACT|calibration | -0.0072 [-0.0265, +0.0114] |
| s2|Uniform - Original FACT|selection | -0.0218 [-0.0432, -0.0007] |
| s2|Uniform - Original FACT|calibration | -0.0036 [-0.0143, +0.0061] |
| s2|Predicted - ASRF|selection | +0.0227 [+0.0030, +0.0451] |
| s2|Predicted - ASRF|calibration | -0.0082 [-0.0268, +0.0102] |
| s2|Uniform - ASRF|selection | +0.0066 [-0.0184, +0.0300] |
| s2|Uniform - ASRF|calibration | -0.0046 [-0.0164, +0.0067] |
| s2|Predicted - (Boundary head + frame classifier)|selection | +0.0167 [-0.0073, +0.0445] |
| s2|Predicted - (Boundary head + frame classifier)|calibration | +0.0024 [-0.0095, +0.0156] |
| s0|Predicted - (E1 + B3)|selection | -0.0107 [-0.0353, +0.0141] |
| s0|Predicted - (E1 + B3)|calibration | +0.0047 [-0.0092, +0.0178] |
| s0|Predicted - (E0 + B3)|selection | -0.0511 [-0.0833, -0.0202] |
| s0|Predicted - (E0 + B3)|calibration | +0.0297 [+0.0072, +0.0530] |
| s0|Predicted - (B7 + B3)|selection | +0.0034 [-0.0220, +0.0293] |
| s0|Predicted - (B7 + B3)|calibration | -0.0085 [-0.0263, +0.0096] |
| s1|Predicted - (E1 + B3)|selection | +0.0049 [-0.0178, +0.0292] |
| s1|Predicted - (E1 + B3)|calibration | +0.0049 [-0.0149, +0.0225] |
| s1|Predicted - (E0 + B3)|selection | -0.0356 [-0.0670, -0.0038] |
| s1|Predicted - (E0 + B3)|calibration | +0.0298 [+0.0084, +0.0529] |
| s1|Predicted - (B7 + B3)|selection | +0.0190 [-0.0011, +0.0394] |
| s1|Predicted - (B7 + B3)|calibration | -0.0083 [-0.0273, +0.0110] |
| s2|Predicted - (E1 + B3)|selection | +0.0112 [-0.0099, +0.0362] |
| s2|Predicted - (E1 + B3)|calibration | -0.0009 [-0.0171, +0.0137] |
| s2|Predicted - (E0 + B3)|selection | -0.0292 [-0.0568, -0.0004] |
| s2|Predicted - (E0 + B3)|calibration | +0.0240 [-0.0007, +0.0481] |
| s2|Predicted - (B7 + B3)|selection | +0.0253 [+0.0045, +0.0479] |
| s2|Predicted - (B7 + B3)|calibration | -0.0141 [-0.0333, +0.0057] |

## Reference units (multi streams)

| system|half|group | segments | matched | recall | missed_recordings | missed_sessions |
|---|---|---|---|---|---|
| Original FACT s0|selection|short segments | 141 | 127 | 0.9007 | 11 | 9 |
| Original FACT s0|selection|short A4C or A5C segments | 65 | 54 | 0.8308 | 10 | 8 |
| Original FACT s0|selection|X-Y-X middles | 69 | 60 | 0.8696 | 10 | 10 |
| Original FACT s0|selection|A4C/A5C X-Y-X middles | 7 | 1 | 0.1429 | 7 | 7 |
| Original FACT s0|calibration|short segments | 134 | 124 | 0.9254 | 6 | 5 |
| Original FACT s0|calibration|short A4C or A5C segments | 67 | 58 | 0.8657 | 5 | 4 |
| Original FACT s0|calibration|X-Y-X middles | 80 | 79 | 0.9875 | 1 | 1 |
| Original FACT s0|calibration|A4C/A5C X-Y-X middles | 7 | 7 | 1.0000 | 0 | 0 |
| Uniform s0|selection|short segments | 141 | 131 | 0.9291 | 8 | 7 |
| Uniform s0|selection|short A4C or A5C segments | 65 | 58 | 0.8923 | 7 | 6 |
| Uniform s0|selection|X-Y-X middles | 69 | 63 | 0.9130 | 6 | 6 |
| Uniform s0|selection|A4C/A5C X-Y-X middles | 7 | 4 | 0.5714 | 3 | 3 |
| Uniform s0|calibration|short segments | 134 | 129 | 0.9627 | 5 | 5 |
| Uniform s0|calibration|short A4C or A5C segments | 67 | 62 | 0.9254 | 5 | 5 |
| Uniform s0|calibration|X-Y-X middles | 80 | 80 | 1.0000 | 0 | 0 |
| Uniform s0|calibration|A4C/A5C X-Y-X middles | 7 | 7 | 1.0000 | 0 | 0 |
| Predicted s0|selection|short segments | 141 | 131 | 0.9291 | 8 | 7 |
| Predicted s0|selection|short A4C or A5C segments | 65 | 58 | 0.8923 | 7 | 6 |
| Predicted s0|selection|X-Y-X middles | 69 | 63 | 0.9130 | 7 | 7 |
| Predicted s0|selection|A4C/A5C X-Y-X middles | 7 | 4 | 0.5714 | 4 | 4 |
| Predicted s0|calibration|short segments | 134 | 125 | 0.9328 | 8 | 6 |
| Predicted s0|calibration|short A4C or A5C segments | 67 | 59 | 0.8806 | 7 | 5 |
| Predicted s0|calibration|X-Y-X middles | 80 | 78 | 0.9750 | 4 | 4 |
| Predicted s0|calibration|A4C/A5C X-Y-X middles | 7 | 6 | 0.8571 | 1 | 1 |
| ASRF s0|selection|short segments | 141 | 125 | 0.8865 | 14 | 9 |
| ASRF s0|selection|short A4C or A5C segments | 65 | 53 | 0.8154 | 12 | 8 |
| ASRF s0|selection|X-Y-X middles | 69 | 60 | 0.8696 | 9 | 8 |
| ASRF s0|selection|A4C/A5C X-Y-X middles | 7 | 4 | 0.5714 | 4 | 4 |
| ASRF s0|calibration|short segments | 134 | 127 | 0.9478 | 7 | 6 |
| ASRF s0|calibration|short A4C or A5C segments | 67 | 61 | 0.9104 | 6 | 5 |
| ASRF s0|calibration|X-Y-X middles | 80 | 79 | 0.9875 | 3 | 3 |
| ASRF s0|calibration|A4C/A5C X-Y-X middles | 7 | 7 | 1.0000 | 0 | 0 |
| Boundary head + frame classifier s0|selection|short segments | 141 | 122 | 0.8652 | 14 | 10 |
| Boundary head + frame classifier s0|selection|short A4C or A5C segments | 65 | 51 | 0.7846 | 11 | 9 |
| Boundary head + frame classifier s0|selection|X-Y-X middles | 69 | 60 | 0.8696 | 11 | 9 |
| Boundary head + frame classifier s0|selection|A4C/A5C X-Y-X middles | 7 | 2 | 0.2857 | 5 | 5 |
| Boundary head + frame classifier s0|calibration|short segments | 134 | 124 | 0.9254 | 10 | 10 |
| Boundary head + frame classifier s0|calibration|short A4C or A5C segments | 67 | 58 | 0.8657 | 9 | 9 |
| Boundary head + frame classifier s0|calibration|X-Y-X middles | 80 | 77 | 0.9625 | 5 | 5 |
| Boundary head + frame classifier s0|calibration|A4C/A5C X-Y-X middles | 7 | 5 | 0.7143 | 2 | 2 |
| Original FACT s1|selection|short segments | 141 | 127 | 0.9007 | 11 | 9 |
| Original FACT s1|selection|short A4C or A5C segments | 65 | 56 | 0.8615 | 8 | 7 |
| Original FACT s1|selection|X-Y-X middles | 69 | 60 | 0.8696 | 9 | 9 |
| Original FACT s1|selection|A4C/A5C X-Y-X middles | 7 | 3 | 0.4286 | 4 | 4 |
| Original FACT s1|calibration|short segments | 134 | 123 | 0.9179 | 8 | 6 |
| Original FACT s1|calibration|short A4C or A5C segments | 67 | 57 | 0.8507 | 7 | 5 |
| Original FACT s1|calibration|X-Y-X middles | 80 | 77 | 0.9625 | 3 | 3 |
| Original FACT s1|calibration|A4C/A5C X-Y-X middles | 7 | 5 | 0.7143 | 2 | 2 |
| Uniform s1|selection|short segments | 141 | 129 | 0.9149 | 10 | 9 |
| Uniform s1|selection|short A4C or A5C segments | 65 | 56 | 0.8615 | 9 | 8 |
| Uniform s1|selection|X-Y-X middles | 69 | 63 | 0.9130 | 7 | 7 |
| Uniform s1|selection|A4C/A5C X-Y-X middles | 7 | 4 | 0.5714 | 4 | 4 |
| Uniform s1|calibration|short segments | 134 | 128 | 0.9552 | 4 | 4 |
| Uniform s1|calibration|short A4C or A5C segments | 67 | 61 | 0.9104 | 4 | 4 |
| Uniform s1|calibration|X-Y-X middles | 80 | 79 | 0.9875 | 1 | 1 |
| Uniform s1|calibration|A4C/A5C X-Y-X middles | 7 | 7 | 1.0000 | 0 | 0 |
| Predicted s1|selection|short segments | 141 | 131 | 0.9291 | 8 | 7 |
| Predicted s1|selection|short A4C or A5C segments | 65 | 58 | 0.8923 | 7 | 6 |
| Predicted s1|selection|X-Y-X middles | 69 | 65 | 0.9420 | 4 | 4 |
| Predicted s1|selection|A4C/A5C X-Y-X middles | 7 | 6 | 0.8571 | 1 | 1 |
| Predicted s1|calibration|short segments | 134 | 128 | 0.9552 | 5 | 4 |
| Predicted s1|calibration|short A4C or A5C segments | 67 | 61 | 0.9104 | 5 | 4 |
| Predicted s1|calibration|X-Y-X middles | 80 | 79 | 0.9875 | 3 | 3 |
| Predicted s1|calibration|A4C/A5C X-Y-X middles | 7 | 7 | 1.0000 | 0 | 0 |
| ASRF s1|selection|short segments | 141 | 123 | 0.8723 | 15 | 10 |
| ASRF s1|selection|short A4C or A5C segments | 65 | 51 | 0.7846 | 13 | 9 |
| ASRF s1|selection|X-Y-X middles | 69 | 60 | 0.8696 | 10 | 9 |
| ASRF s1|selection|A4C/A5C X-Y-X middles | 7 | 3 | 0.4286 | 5 | 5 |
| ASRF s1|calibration|short segments | 134 | 128 | 0.9552 | 6 | 5 |
| ASRF s1|calibration|short A4C or A5C segments | 67 | 61 | 0.9104 | 6 | 5 |
| ASRF s1|calibration|X-Y-X middles | 80 | 79 | 0.9875 | 3 | 3 |
| ASRF s1|calibration|A4C/A5C X-Y-X middles | 7 | 7 | 1.0000 | 0 | 0 |
| Boundary head + frame classifier s1|selection|short segments | 141 | 125 | 0.8865 | 13 | 9 |
| Boundary head + frame classifier s1|selection|short A4C or A5C segments | 65 | 53 | 0.8154 | 11 | 8 |
| Boundary head + frame classifier s1|selection|X-Y-X middles | 69 | 63 | 0.9130 | 8 | 8 |
| Boundary head + frame classifier s1|selection|A4C/A5C X-Y-X middles | 7 | 4 | 0.5714 | 3 | 3 |
| Boundary head + frame classifier s1|calibration|short segments | 134 | 124 | 0.9254 | 10 | 9 |
| Boundary head + frame classifier s1|calibration|short A4C or A5C segments | 67 | 58 | 0.8657 | 9 | 8 |
| Boundary head + frame classifier s1|calibration|X-Y-X middles | 80 | 77 | 0.9625 | 5 | 5 |
| Boundary head + frame classifier s1|calibration|A4C/A5C X-Y-X middles | 7 | 5 | 0.7143 | 2 | 2 |
| Original FACT s2|selection|short segments | 141 | 128 | 0.9078 | 9 | 8 |
| Original FACT s2|selection|short A4C or A5C segments | 65 | 55 | 0.8462 | 8 | 7 |
| Original FACT s2|selection|X-Y-X middles | 69 | 59 | 0.8551 | 12 | 12 |
| Original FACT s2|selection|A4C/A5C X-Y-X middles | 7 | 3 | 0.4286 | 4 | 4 |
| Original FACT s2|calibration|short segments | 134 | 125 | 0.9328 | 7 | 5 |
| Original FACT s2|calibration|short A4C or A5C segments | 67 | 59 | 0.8806 | 6 | 4 |
| Original FACT s2|calibration|X-Y-X middles | 80 | 78 | 0.9750 | 2 | 2 |
| Original FACT s2|calibration|A4C/A5C X-Y-X middles | 7 | 6 | 0.8571 | 1 | 1 |
| Uniform s2|selection|short segments | 141 | 128 | 0.9078 | 10 | 9 |
| Uniform s2|selection|short A4C or A5C segments | 65 | 55 | 0.8462 | 9 | 8 |
| Uniform s2|selection|X-Y-X middles | 69 | 61 | 0.8841 | 8 | 8 |
| Uniform s2|selection|A4C/A5C X-Y-X middles | 7 | 3 | 0.4286 | 4 | 4 |
| Uniform s2|calibration|short segments | 134 | 125 | 0.9328 | 8 | 6 |
| Uniform s2|calibration|short A4C or A5C segments | 67 | 59 | 0.8806 | 7 | 5 |
| Uniform s2|calibration|X-Y-X middles | 80 | 78 | 0.9750 | 4 | 4 |
| Uniform s2|calibration|A4C/A5C X-Y-X middles | 7 | 6 | 0.8571 | 1 | 1 |
| Predicted s2|selection|short segments | 141 | 129 | 0.9149 | 9 | 8 |
| Predicted s2|selection|short A4C or A5C segments | 65 | 56 | 0.8615 | 8 | 7 |
| Predicted s2|selection|X-Y-X middles | 69 | 65 | 0.9420 | 4 | 4 |
| Predicted s2|selection|A4C/A5C X-Y-X middles | 7 | 5 | 0.7143 | 2 | 2 |
| Predicted s2|calibration|short segments | 134 | 124 | 0.9254 | 9 | 6 |
| Predicted s2|calibration|short A4C or A5C segments | 67 | 58 | 0.8657 | 8 | 5 |
| Predicted s2|calibration|X-Y-X middles | 80 | 76 | 0.9500 | 6 | 5 |
| Predicted s2|calibration|A4C/A5C X-Y-X middles | 7 | 4 | 0.5714 | 3 | 2 |
| ASRF s2|selection|short segments | 141 | 125 | 0.8865 | 14 | 10 |
| ASRF s2|selection|short A4C or A5C segments | 65 | 54 | 0.8308 | 11 | 8 |
| ASRF s2|selection|X-Y-X middles | 69 | 60 | 0.8696 | 9 | 8 |
| ASRF s2|selection|A4C/A5C X-Y-X middles | 7 | 4 | 0.5714 | 4 | 4 |
| ASRF s2|calibration|short segments | 134 | 125 | 0.9328 | 9 | 7 |
| ASRF s2|calibration|short A4C or A5C segments | 67 | 60 | 0.8955 | 7 | 6 |
| ASRF s2|calibration|X-Y-X middles | 80 | 79 | 0.9875 | 3 | 3 |
| ASRF s2|calibration|A4C/A5C X-Y-X middles | 7 | 7 | 1.0000 | 0 | 0 |
| Boundary head + frame classifier s2|selection|short segments | 141 | 123 | 0.8723 | 14 | 10 |
| Boundary head + frame classifier s2|selection|short A4C or A5C segments | 65 | 52 | 0.8000 | 11 | 9 |
| Boundary head + frame classifier s2|selection|X-Y-X middles | 69 | 60 | 0.8696 | 11 | 10 |
| Boundary head + frame classifier s2|selection|A4C/A5C X-Y-X middles | 7 | 2 | 0.2857 | 5 | 5 |
| Boundary head + frame classifier s2|calibration|short segments | 134 | 124 | 0.9254 | 10 | 10 |
| Boundary head + frame classifier s2|calibration|short A4C or A5C segments | 67 | 58 | 0.8657 | 9 | 9 |
| Boundary head + frame classifier s2|calibration|X-Y-X middles | 80 | 77 | 0.9625 | 5 | 5 |
| Boundary head + frame classifier s2|calibration|A4C/A5C X-Y-X middles | 7 | 5 | 0.7143 | 2 | 2 |
| E1 + B3|selection|short segments | 141 | 129 | 0.9149 | 10 | 7 |
| E1 + B3|selection|short A4C or A5C segments | 65 | 56 | 0.8615 | 9 | 6 |
| E1 + B3|selection|X-Y-X middles | 69 | 63 | 0.9130 | 6 | 5 |
| E1 + B3|selection|A4C/A5C X-Y-X middles | 7 | 5 | 0.7143 | 2 | 2 |
| E1 + B3|calibration|short segments | 134 | 125 | 0.9328 | 8 | 6 |
| E1 + B3|calibration|short A4C or A5C segments | 67 | 59 | 0.8806 | 7 | 5 |
| E1 + B3|calibration|X-Y-X middles | 80 | 80 | 1.0000 | 0 | 0 |
| E1 + B3|calibration|A4C/A5C X-Y-X middles | 7 | 7 | 1.0000 | 0 | 0 |

## Stage tracking (matched runs per readout; misses of the official output by earliest loss)

| system|set|group | E1 | i | u | U | final | segments | first lost at E1 | first lost at i | first lost at u | first lost at U | first lost at final |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Original FACT s0|training monitor|stream:all | 50 | 51 | 51 | 52 | 52 | 59 | 5 | 2 | 0 | 0 | 0 |
| Original FACT s0|training monitor|stream:short | 16 | 14 | 14 | 14 | 14 | 19 | 3 | 2 | 0 | 0 | 0 |
| Original FACT s0|training monitor|stream:a4c_a5c_xyx | 3 | 2 | 2 | 2 | 2 | 5 | 2 | 1 | 0 | 0 | 0 |
| Original FACT s0|training monitor|native:all | 4 | 5 | 5 | 5 | 5 | 5 | 0 | 0 | 0 | 0 | 0 |
| Original FACT s0|training monitor|native:short | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| Original FACT s0|development selection|stream:all | 410 | 461 | 461 | 460 | 459 | 502 | 30 | 8 | 2 | 0 | 3 |
| Original FACT s0|development selection|stream:short | 131 | 130 | 129 | 129 | 127 | 141 | 6 | 4 | 2 | 0 | 2 |
| Original FACT s0|development selection|stream:a4c_a5c_xyx | 6 | 5 | 3 | 3 | 1 | 7 | 1 | 1 | 2 | 0 | 2 |
| Original FACT s0|development selection|native:all | 247 | 280 | 281 | 281 | 282 | 284 | 2 | 0 | 0 | 0 | 0 |
| Original FACT s0|development selection|native:short | 30 | 30 | 31 | 31 | 31 | 31 | 0 | 0 | 0 | 0 | 0 |
| Original FACT s0|development calibration|stream:all | 459 | 509 | 506 | 503 | 504 | 518 | 9 | 2 | 1 | 1 | 1 |
| Original FACT s0|development calibration|stream:short | 118 | 129 | 127 | 126 | 124 | 134 | 6 | 1 | 1 | 1 | 1 |
| Original FACT s0|development calibration|stream:a4c_a5c_xyx | 7 | 7 | 7 | 7 | 7 | 7 | 0 | 0 | 0 | 0 | 0 |
| Original FACT s0|development calibration|native:all | 200 | 235 | 235 | 235 | 235 | 237 | 2 | 0 | 0 | 0 | 0 |
| Original FACT s0|development calibration|native:short | 17 | 20 | 20 | 20 | 20 | 20 | 0 | 0 | 0 | 0 | 0 |
| Uniform s0|training monitor|stream:all | 50 | 51 | 51 | 52 | 57 | 59 | 2 | 0 | 0 | 0 | 0 |
| Uniform s0|training monitor|stream:short | 16 | 14 | 14 | 14 | 17 | 19 | 2 | 0 | 0 | 0 | 0 |
| Uniform s0|training monitor|stream:a4c_a5c_xyx | 3 | 2 | 2 | 2 | 4 | 5 | 1 | 0 | 0 | 0 | 0 |
| Uniform s0|training monitor|native:all | 4 | 5 | 5 | 5 | 5 | 5 | 0 | 0 | 0 | 0 | 0 |
| Uniform s0|training monitor|native:short | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| Uniform s0|development selection|stream:all | 410 | 459 | 460 | 458 | 459 | 502 | 34 | 6 | 2 | 1 | 0 |
| Uniform s0|development selection|stream:short | 131 | 130 | 128 | 126 | 131 | 141 | 5 | 2 | 2 | 1 | 0 |
| Uniform s0|development selection|stream:a4c_a5c_xyx | 6 | 5 | 3 | 3 | 4 | 7 | 1 | 0 | 2 | 0 | 0 |
| Uniform s0|development selection|native:all | 247 | 280 | 280 | 281 | 281 | 284 | 3 | 0 | 0 | 0 | 0 |
| Uniform s0|development selection|native:short | 30 | 30 | 30 | 31 | 30 | 31 | 1 | 0 | 0 | 0 | 0 |
| Uniform s0|development calibration|stream:all | 459 | 508 | 505 | 502 | 507 | 518 | 9 | 1 | 1 | 0 | 0 |
| Uniform s0|development calibration|stream:short | 118 | 127 | 127 | 124 | 129 | 134 | 4 | 1 | 0 | 0 | 0 |
| Uniform s0|development calibration|stream:a4c_a5c_xyx | 7 | 7 | 7 | 7 | 7 | 7 | 0 | 0 | 0 | 0 | 0 |
| Uniform s0|development calibration|native:all | 200 | 234 | 235 | 235 | 235 | 237 | 2 | 0 | 0 | 0 | 0 |
| Uniform s0|development calibration|native:short | 17 | 20 | 20 | 20 | 20 | 20 | 0 | 0 | 0 | 0 | 0 |
| Predicted s0|training monitor|stream:all | 50 | 51 | 50 | 56 | 57 | 59 | 2 | 0 | 0 | 0 | 0 |
| Predicted s0|training monitor|stream:short | 16 | 14 | 14 | 16 | 17 | 19 | 2 | 0 | 0 | 0 | 0 |
| Predicted s0|training monitor|stream:a4c_a5c_xyx | 3 | 2 | 1 | 4 | 4 | 5 | 1 | 0 | 0 | 0 | 0 |
| Predicted s0|training monitor|native:all | 4 | 5 | 5 | 5 | 5 | 5 | 0 | 0 | 0 | 0 | 0 |
| Predicted s0|training monitor|native:short | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| Predicted s0|development selection|stream:all | 410 | 460 | 458 | 459 | 454 | 502 | 36 | 6 | 0 | 2 | 4 |
| Predicted s0|development selection|stream:short | 131 | 130 | 128 | 130 | 131 | 141 | 5 | 2 | 0 | 1 | 2 |
| Predicted s0|development selection|stream:a4c_a5c_xyx | 6 | 5 | 3 | 4 | 4 | 7 | 1 | 0 | 0 | 1 | 1 |
| Predicted s0|development selection|native:all | 247 | 279 | 280 | 280 | 281 | 284 | 3 | 0 | 0 | 0 | 0 |
| Predicted s0|development selection|native:short | 30 | 30 | 30 | 30 | 31 | 31 | 0 | 0 | 0 | 0 | 0 |
| Predicted s0|development calibration|stream:all | 459 | 508 | 505 | 506 | 504 | 518 | 9 | 1 | 0 | 1 | 3 |
| Predicted s0|development calibration|stream:short | 118 | 128 | 127 | 126 | 125 | 134 | 6 | 1 | 0 | 1 | 1 |
| Predicted s0|development calibration|stream:a4c_a5c_xyx | 7 | 7 | 7 | 7 | 6 | 7 | 0 | 0 | 0 | 0 | 1 |
| Predicted s0|development calibration|native:all | 200 | 234 | 234 | 234 | 234 | 237 | 3 | 0 | 0 | 0 | 0 |
| Predicted s0|development calibration|native:short | 17 | 20 | 20 | 20 | 20 | 20 | 0 | 0 | 0 | 0 | 0 |
| Original FACT s1|training monitor|stream:all | 50 | 51 | 51 | 51 | 52 | 59 | 5 | 2 | 0 | 0 | 0 |
| Original FACT s1|training monitor|stream:short | 16 | 14 | 14 | 14 | 14 | 19 | 3 | 2 | 0 | 0 | 0 |
| Original FACT s1|training monitor|stream:a4c_a5c_xyx | 3 | 2 | 2 | 2 | 2 | 5 | 2 | 1 | 0 | 0 | 0 |
| Original FACT s1|training monitor|native:all | 4 | 5 | 5 | 5 | 5 | 5 | 0 | 0 | 0 | 0 | 0 |
| Original FACT s1|training monitor|native:short | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| Original FACT s1|development selection|stream:all | 410 | 458 | 461 | 460 | 455 | 502 | 34 | 10 | 0 | 0 | 3 |
| Original FACT s1|development selection|stream:short | 131 | 127 | 129 | 128 | 127 | 141 | 6 | 6 | 0 | 0 | 2 |
| Original FACT s1|development selection|stream:a4c_a5c_xyx | 6 | 3 | 3 | 3 | 3 | 7 | 1 | 3 | 0 | 0 | 0 |
| Original FACT s1|development selection|native:all | 247 | 280 | 280 | 280 | 281 | 284 | 3 | 0 | 0 | 0 | 0 |
| Original FACT s1|development selection|native:short | 30 | 30 | 30 | 30 | 31 | 31 | 0 | 0 | 0 | 0 | 0 |
| Original FACT s1|development calibration|stream:all | 459 | 507 | 504 | 502 | 499 | 518 | 10 | 1 | 2 | 1 | 5 |
| Original FACT s1|development calibration|stream:short | 118 | 128 | 126 | 125 | 123 | 134 | 7 | 0 | 1 | 1 | 2 |
| Original FACT s1|development calibration|stream:a4c_a5c_xyx | 7 | 7 | 7 | 7 | 5 | 7 | 0 | 0 | 0 | 0 | 2 |
| Original FACT s1|development calibration|native:all | 200 | 233 | 234 | 234 | 234 | 237 | 3 | 0 | 0 | 0 | 0 |
| Original FACT s1|development calibration|native:short | 17 | 19 | 20 | 20 | 20 | 20 | 0 | 0 | 0 | 0 | 0 |
| Uniform s1|training monitor|stream:all | 50 | 51 | 51 | 53 | 53 | 59 | 5 | 1 | 0 | 0 | 0 |
| Uniform s1|training monitor|stream:short | 16 | 14 | 14 | 14 | 15 | 19 | 3 | 1 | 0 | 0 | 0 |
| Uniform s1|training monitor|stream:a4c_a5c_xyx | 3 | 2 | 2 | 2 | 2 | 5 | 2 | 1 | 0 | 0 | 0 |
| Uniform s1|training monitor|native:all | 4 | 5 | 5 | 5 | 5 | 5 | 0 | 0 | 0 | 0 | 0 |
| Uniform s1|training monitor|native:short | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| Uniform s1|development selection|stream:all | 410 | 461 | 460 | 457 | 459 | 502 | 33 | 5 | 0 | 1 | 4 |
| Uniform s1|development selection|stream:short | 131 | 129 | 128 | 124 | 129 | 141 | 7 | 2 | 0 | 1 | 2 |
| Uniform s1|development selection|stream:a4c_a5c_xyx | 6 | 5 | 3 | 2 | 4 | 7 | 1 | 0 | 0 | 1 | 1 |
| Uniform s1|development selection|native:all | 247 | 280 | 280 | 281 | 282 | 284 | 2 | 0 | 0 | 0 | 0 |
| Uniform s1|development selection|native:short | 30 | 30 | 30 | 31 | 31 | 31 | 0 | 0 | 0 | 0 | 0 |
| Uniform s1|development calibration|stream:all | 459 | 506 | 506 | 504 | 504 | 518 | 9 | 2 | 0 | 1 | 2 |
| Uniform s1|development calibration|stream:short | 118 | 127 | 127 | 126 | 128 | 134 | 3 | 1 | 0 | 1 | 1 |
| Uniform s1|development calibration|stream:a4c_a5c_xyx | 7 | 7 | 7 | 7 | 7 | 7 | 0 | 0 | 0 | 0 | 0 |
| Uniform s1|development calibration|native:all | 200 | 234 | 235 | 235 | 234 | 237 | 3 | 0 | 0 | 0 | 0 |
| Uniform s1|development calibration|native:short | 17 | 20 | 20 | 20 | 20 | 20 | 0 | 0 | 0 | 0 | 0 |
| Predicted s1|training monitor|stream:all | 50 | 51 | 49 | 56 | 57 | 59 | 2 | 0 | 0 | 0 | 0 |
| Predicted s1|training monitor|stream:short | 16 | 14 | 14 | 16 | 17 | 19 | 2 | 0 | 0 | 0 | 0 |
| Predicted s1|training monitor|stream:a4c_a5c_xyx | 3 | 2 | 0 | 4 | 4 | 5 | 1 | 0 | 0 | 0 | 0 |
| Predicted s1|training monitor|native:all | 4 | 5 | 5 | 5 | 5 | 5 | 0 | 0 | 0 | 0 | 0 |
| Predicted s1|training monitor|native:short | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| Predicted s1|development selection|stream:all | 410 | 458 | 459 | 465 | 462 | 502 | 31 | 7 | 0 | 2 | 0 |
| Predicted s1|development selection|stream:short | 131 | 128 | 128 | 130 | 131 | 141 | 6 | 3 | 0 | 1 | 0 |
| Predicted s1|development selection|stream:a4c_a5c_xyx | 6 | 3 | 3 | 5 | 6 | 7 | 1 | 0 | 0 | 0 | 0 |
| Predicted s1|development selection|native:all | 247 | 280 | 280 | 280 | 280 | 284 | 4 | 0 | 0 | 0 | 0 |
| Predicted s1|development selection|native:short | 30 | 30 | 30 | 30 | 30 | 31 | 1 | 0 | 0 | 0 | 0 |
| Predicted s1|development calibration|stream:all | 459 | 505 | 504 | 501 | 506 | 518 | 10 | 1 | 1 | 0 | 0 |
| Predicted s1|development calibration|stream:short | 118 | 126 | 125 | 127 | 128 | 134 | 5 | 0 | 1 | 0 | 0 |
| Predicted s1|development calibration|stream:a4c_a5c_xyx | 7 | 7 | 7 | 6 | 7 | 7 | 0 | 0 | 0 | 0 | 0 |
| Predicted s1|development calibration|native:all | 200 | 234 | 235 | 235 | 234 | 237 | 3 | 0 | 0 | 0 | 0 |
| Predicted s1|development calibration|native:short | 17 | 20 | 20 | 20 | 20 | 20 | 0 | 0 | 0 | 0 | 0 |
| Original FACT s2|training monitor|stream:all | 50 | 51 | 51 | 51 | 51 | 59 | 6 | 2 | 0 | 0 | 0 |
| Original FACT s2|training monitor|stream:short | 16 | 14 | 14 | 14 | 14 | 19 | 3 | 2 | 0 | 0 | 0 |
| Original FACT s2|training monitor|stream:a4c_a5c_xyx | 3 | 2 | 2 | 1 | 2 | 5 | 2 | 1 | 0 | 0 | 0 |
| Original FACT s2|training monitor|native:all | 4 | 5 | 5 | 5 | 5 | 5 | 0 | 0 | 0 | 0 | 0 |
| Original FACT s2|training monitor|native:short | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| Original FACT s2|development selection|stream:all | 410 | 460 | 461 | 457 | 460 | 502 | 29 | 8 | 2 | 0 | 3 |
| Original FACT s2|development selection|stream:short | 131 | 129 | 129 | 125 | 128 | 141 | 7 | 4 | 2 | 0 | 0 |
| Original FACT s2|development selection|stream:a4c_a5c_xyx | 6 | 3 | 3 | 1 | 3 | 7 | 1 | 2 | 1 | 0 | 0 |
| Original FACT s2|development selection|native:all | 247 | 280 | 281 | 281 | 280 | 284 | 4 | 0 | 0 | 0 | 0 |
| Original FACT s2|development selection|native:short | 30 | 30 | 31 | 31 | 30 | 31 | 1 | 0 | 0 | 0 | 0 |
| Original FACT s2|development calibration|stream:all | 459 | 506 | 506 | 505 | 500 | 518 | 10 | 2 | 1 | 0 | 5 |
| Original FACT s2|development calibration|stream:short | 118 | 126 | 127 | 126 | 125 | 134 | 5 | 2 | 0 | 0 | 2 |
| Original FACT s2|development calibration|stream:a4c_a5c_xyx | 7 | 7 | 7 | 7 | 6 | 7 | 0 | 0 | 0 | 0 | 1 |
| Original FACT s2|development calibration|native:all | 200 | 233 | 234 | 234 | 234 | 237 | 3 | 0 | 0 | 0 | 0 |
| Original FACT s2|development calibration|native:short | 17 | 19 | 20 | 20 | 20 | 20 | 0 | 0 | 0 | 0 | 0 |
| Uniform s2|training monitor|stream:all | 50 | 51 | 51 | 52 | 55 | 59 | 3 | 1 | 0 | 0 | 0 |
| Uniform s2|training monitor|stream:short | 16 | 14 | 14 | 14 | 16 | 19 | 2 | 1 | 0 | 0 | 0 |
| Uniform s2|training monitor|stream:a4c_a5c_xyx | 3 | 2 | 2 | 2 | 4 | 5 | 1 | 0 | 0 | 0 | 0 |
| Uniform s2|training monitor|native:all | 4 | 5 | 5 | 5 | 5 | 5 | 0 | 0 | 0 | 0 | 0 |
| Uniform s2|training monitor|native:short | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| Uniform s2|development selection|stream:all | 410 | 460 | 461 | 460 | 457 | 502 | 34 | 7 | 2 | 0 | 2 |
| Uniform s2|development selection|stream:short | 131 | 129 | 128 | 124 | 128 | 141 | 8 | 3 | 2 | 0 | 0 |
| Uniform s2|development selection|stream:a4c_a5c_xyx | 6 | 4 | 3 | 3 | 3 | 7 | 1 | 1 | 2 | 0 | 0 |
| Uniform s2|development selection|native:all | 247 | 280 | 280 | 282 | 282 | 284 | 2 | 0 | 0 | 0 | 0 |
| Uniform s2|development selection|native:short | 30 | 30 | 30 | 31 | 31 | 31 | 0 | 0 | 0 | 0 | 0 |
| Uniform s2|development calibration|stream:all | 459 | 507 | 505 | 504 | 501 | 518 | 11 | 1 | 2 | 1 | 2 |
| Uniform s2|development calibration|stream:short | 118 | 127 | 126 | 126 | 125 | 134 | 5 | 1 | 1 | 1 | 1 |
| Uniform s2|development calibration|stream:a4c_a5c_xyx | 7 | 7 | 7 | 7 | 6 | 7 | 0 | 0 | 0 | 0 | 1 |
| Uniform s2|development calibration|native:all | 200 | 234 | 234 | 234 | 234 | 237 | 3 | 0 | 0 | 0 | 0 |
| Uniform s2|development calibration|native:short | 17 | 20 | 20 | 20 | 20 | 20 | 0 | 0 | 0 | 0 | 0 |
| Predicted s2|training monitor|stream:all | 50 | 51 | 48 | 57 | 57 | 59 | 2 | 0 | 0 | 0 | 0 |
| Predicted s2|training monitor|stream:short | 16 | 14 | 13 | 17 | 17 | 19 | 2 | 0 | 0 | 0 | 0 |
| Predicted s2|training monitor|stream:a4c_a5c_xyx | 3 | 2 | 0 | 4 | 4 | 5 | 1 | 0 | 0 | 0 | 0 |
| Predicted s2|training monitor|native:all | 4 | 5 | 5 | 5 | 5 | 5 | 0 | 0 | 0 | 0 | 0 |
| Predicted s2|training monitor|native:short | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| Predicted s2|development selection|stream:all | 410 | 459 | 461 | 466 | 463 | 502 | 31 | 6 | 0 | 0 | 2 |
| Predicted s2|development selection|stream:short | 131 | 129 | 128 | 132 | 129 | 141 | 8 | 2 | 0 | 0 | 2 |
| Predicted s2|development selection|stream:a4c_a5c_xyx | 6 | 4 | 3 | 5 | 5 | 7 | 1 | 0 | 0 | 0 | 1 |
| Predicted s2|development selection|native:all | 247 | 280 | 280 | 280 | 281 | 284 | 3 | 0 | 0 | 0 | 0 |
| Predicted s2|development selection|native:short | 30 | 30 | 30 | 30 | 30 | 31 | 1 | 0 | 0 | 0 | 0 |
| Predicted s2|development calibration|stream:all | 459 | 509 | 506 | 504 | 502 | 518 | 8 | 1 | 0 | 3 | 4 |
| Predicted s2|development calibration|stream:short | 118 | 128 | 127 | 126 | 124 | 134 | 5 | 1 | 0 | 1 | 3 |
| Predicted s2|development calibration|stream:a4c_a5c_xyx | 7 | 7 | 7 | 6 | 4 | 7 | 0 | 0 | 0 | 1 | 2 |
| Predicted s2|development calibration|native:all | 200 | 233 | 234 | 235 | 236 | 237 | 1 | 0 | 0 | 0 | 0 |
| Predicted s2|development calibration|native:short | 17 | 19 | 20 | 20 | 20 | 20 | 0 | 0 | 0 | 0 | 0 |

## Position control on the Predicted models (head positions against even positions)

| seed|half | segment F1@50 (head positions) | segment F1@50 (even positions) | difference | boundary R difference | boundary P difference | runs matched by a only | runs matched by b only |
|---|---|---|---|---|---|---|---|
| s0|selection | 0.8833 | 0.8902 | -0.0069 | 0.0183 | -0.0005 | 10 | 10 |
| s1|selection | 0.8988 | 0.8800 | 0.0188 | 0.0131 | 0.0057 | 13 | 2 |
| s2|selection | 0.9052 | 0.8948 | 0.0104 | 0.0079 | -0.0063 | 12 | 4 |
| s0|calibration | 0.9637 | 0.9663 | -0.0026 | 0.0050 | -0.0118 | 2 | 0 |
| s1|calibration | 0.9638 | 0.9646 | -0.0008 | 0.0126 | 0.0006 | 5 | 3 |
| s2|calibration | 0.9580 | 0.9569 | 0.0012 | 0.0075 | -0.0043 | 7 | 4 |

## Verdicts

```
{
 "Predicted - Uniform": {
  "selection": "no consistent difference detected",
  "calibration": "no consistent difference detected"
 },
 "Predicted - Original FACT": {
  "selection": "no consistent difference detected",
  "calibration": "no consistent difference detected"
 },
 "Uniform - Original FACT": {
  "selection": "no consistent difference detected",
  "calibration": "no consistent difference detected"
 },
 "Predicted - ASRF": {
  "selection": "no consistent difference detected",
  "calibration": "no consistent difference detected"
 },
 "Uniform - ASRF": {
  "selection": "no consistent difference detected",
  "calibration": "no consistent difference detected"
 },
 "Predicted - (E1 + B3)": {
  "selection": "no consistent difference detected",
  "calibration": "no consistent difference detected"
 },
 "Predicted - (E0 + B3)": {
  "selection": "consistent negative",
  "calibration": "no consistent difference detected"
 },
 "Predicted - (B7 + B3)": {
  "selection": "no consistent difference detected",
  "calibration": "no consistent difference detected"
 },
 "Predicted - (Boundary head + frame classifier)": {
  "selection": "no consistent difference detected",
  "calibration": "no consistent difference detected"
 }
}
```

## Decision

```
{
 "consistent_beat_both_halves": {
  "Predicted - Uniform": false,
  "Predicted - Original FACT": false,
  "Uniform - Original FACT": false,
  "Predicted - ASRF": false,
  "Uniform - ASRF": false,
  "Predicted - (E1 + B3)": false,
  "Predicted - (E0 + B3)": false,
  "Predicted - (B7 + B3)": false,
  "Predicted - (Boundary head + frame classifier)": false
 },
 "position_control_stable_direction": false,
 "groups": {
  "Uniform": {
   "mean_segment_f1_50": {
    "selection": 0.8905453503453685,
    "calibration": 0.962135905663375
   },
   "constraint_failures": [],
   "goal_met": false,
   "goal_margin": {
    "selection": -0.05386433683929348,
    "calibration": -0.020006265326058204
   },
   "calibration_sensitivity_margin": -0.021757672811401285,
   "losses_at_i_u": false,
   "stable_reduction": false,
   "full_system_gain": false
  },
  "Predicted": {
   "mean_segment_f1_50": {
    "selection": 0.8957606693266289,
    "calibration": 0.9618319730307237
   },
   "constraint_failures": [
    {
     "seed": 1,
     "half": "selection",
     "boundary_recall": 0.9581151832460733,
     "native_macro_f1": 0.9663626568079124
    }
   ],
   "goal_met": false,
   "goal_margin": {
    "selection": -0.048649017858033106,
    "calibration": -0.020310197958709586
   },
   "calibration_sensitivity_margin": -0.022061605444052668,
   "losses_at_i_u": false,
   "stable_reduction": true,
   "full_system_gain": false
  }
 },
 "primary": {
  "selection": "no consistent difference detected",
  "calibration": "no consistent difference detected"
 },
 "rows_that_apply": [
  "an interval includes zero or seeds or halves disagree: uncertain; no new threshold, epoch, grid or best-seed report"
 ]
}
```

## Cost

```
{
 "formal_training_hours": {
  "Uniform_s0": 2.780483809510867,
  "Predicted_s0": 2.799611233671506,
  "Uniform_s1": 2.794980396231016,
  "Predicted_s1": 2.8079474378294416,
  "Uniform_s2": 2.827655933499336,
  "Predicted_s2": 2.8112385160393183
 },
 "head_training_hours": {
  "0": 0.023093835512797038,
  "1": 0.0238206762737698,
  "2": 0.023494194944699606
 },
 "smoke_hours": {
  "Uniform": 0.35340193079577553,
  "Predicted": 0.3548621457152896
 },
 "decode_seconds_per_sequence": {
  "Original FACT s0": 0.020286983994505577,
  "Uniform s0": 0.017876008133634432,
  "Predicted s0": 0.019094981821995302,
  "Original FACT s1": 0.017828858163598867,
  "Uniform s1": 0.018018047314429814,
  "Predicted s1": 0.018762394347048635,
  "Original FACT s2": 0.018048818640856966,
  "Uniform s2": 0.017991272398764385,
  "Predicted s2": 0.01889979769251819
 },
 "evaluation_hours": 0.18694506651825374,
 "total_gpu_hours": 17.78753517852889
}
```
