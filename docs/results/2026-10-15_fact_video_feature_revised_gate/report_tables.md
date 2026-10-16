# FACT video feature round: tables

Recompute: `python scripts/report_fact_video_feature.py --config configs/experiments/fact_video_feature_v2.yaml --selection --manifest-sha256 <sha> --manifest-commit <commit>`

## Systems

| system | half | segment F1@50 | boundary P | boundary R | boundary F1 | native macro-F1 | switches per clip | same-view join cuts | pairs F1@50 | A4C read as A5C | A5C read as A4C |
|---|---|---|---|---|---|---|---|---|---|---|---|
| frozen Ordered + B3 s0 | selection | 0.9385 | 0.9482 | 0.9581 | 0.9531 | 0.9786 | 0.0423 | 0.0174 | 0.9615 | 2 | 0 |
| frozen Ordered + B3 s0 | calibration | 0.9817 | 0.9751 | 0.9824 | 0.9787 | 0.9929 | 0.0253 | 0.0249 | 0.9825 | 1 | 0 |
| frozen Ordered argmax s0 | selection | 0.8929 | 0.8462 | 0.9791 | 0.9078 | 0.9701 | 0.0951 | 0.0435 | 0.9619 | 3 | 0 |
| frozen Ordered argmax s0 | calibration | 0.9611 | 0.9424 | 0.9874 | 0.9644 | 0.9861 | 0.0928 | 0.0348 | 0.9741 | 2 | 0 |
| A01 Linear + B3 s0 | selection | 0.9473 | 0.9530 | 0.9555 | 0.9542 | 0.9786 | 0.0458 | 0.0261 | 0.9687 | 2 | 0 |
| A01 Linear + B3 s0 | calibration | 0.9768 | 0.9749 | 0.9749 | 0.9749 | 0.9861 | 0.0127 | 0.0199 | 0.9709 | 2 | 0 |
| A01 Linear argmax s0 | selection | 0.9007 | 0.8341 | 0.9738 | 0.8986 | 0.9701 | 0.1338 | 0.0391 | 0.9547 | 3 | 0 |
| A01 Linear argmax s0 | calibration | 0.9410 | 0.9091 | 0.9799 | 0.9432 | 0.9861 | 0.1181 | 0.0398 | 0.9465 | 2 | 0 |
| A01 MS-TCN s0 | selection | 0.9199 | 0.9179 | 0.9660 | 0.9413 | 0.9626 | 0.0739 | 0.0348 | 0.9663 | 0 | 3 |
| A01 MS-TCN s0 | calibration | 0.9672 | 0.9623 | 0.9623 | 0.9623 | 0.9780 | 0.0380 | 0.0249 | 0.9738 | 1 | 2 |
| A01 ASRF s0 | selection | 0.9366 | 0.9757 | 0.9476 | 0.9615 | 0.9869 | 0.0141 | 0.0348 | 0.9709 | 0 | 1 |
| A01 ASRF s0 | calibration | 0.9796 | 0.9923 | 0.9698 | 0.9809 | 0.9780 | 0.0084 | 0.0149 | 0.9796 | 1 | 2 |
| FACT s0 | selection | 0.9610 | 0.9657 | 0.9581 | 0.9619 | 0.9656 | 0.0000 | 0.0043 | 0.9662 | 1 | 2 |
| FACT s0 | calibration | 0.9620 | 0.9769 | 0.9548 | 0.9657 | 0.9780 | 0.0084 | 0.0199 | 0.9825 | 1 | 2 |
| FACT frame branch s0 | selection | 0.9444 | 0.9378 | 0.9476 | 0.9427 | 0.9976 | 0.0035 | 0.0043 | 0.9710 | 0 | 0 |
| FACT frame branch s0 | calibration | 0.9661 | 0.9722 | 0.9648 | 0.9685 | 0.9856 | 0.0127 | 0.0149 | 0.9825 | 1 | 1 |
| frozen Ordered + B3 s1 | selection | 0.9603 | 0.9585 | 0.9686 | 0.9635 | 0.9786 | 0.0387 | 0.0043 | 0.9592 | 2 | 0 |
| frozen Ordered + B3 s1 | calibration | 0.9836 | 0.9800 | 0.9874 | 0.9837 | 0.9861 | 0.0169 | 0.0249 | 0.9709 | 2 | 0 |
| frozen Ordered argmax s1 | selection | 0.9250 | 0.8724 | 0.9843 | 0.9250 | 0.9877 | 0.1127 | 0.0304 | 0.9484 | 1 | 0 |
| frozen Ordered argmax s1 | calibration | 0.9611 | 0.9448 | 0.9899 | 0.9669 | 0.9796 | 0.1013 | 0.0398 | 0.9543 | 3 | 0 |
| A01 Linear + B3 s1 | selection | 0.9569 | 0.9707 | 0.9529 | 0.9617 | 0.9786 | 0.0387 | 0.0087 | 0.9522 | 2 | 0 |
| A01 Linear + B3 s1 | calibration | 0.9779 | 0.9726 | 0.9799 | 0.9762 | 0.9861 | 0.0169 | 0.0149 | 0.9709 | 2 | 0 |
| A01 Linear argmax s1 | selection | 0.9074 | 0.8578 | 0.9791 | 0.9144 | 0.9701 | 0.1549 | 0.0348 | 0.9409 | 3 | 0 |
| A01 Linear argmax s1 | calibration | 0.9583 | 0.9378 | 0.9849 | 0.9608 | 0.9732 | 0.0802 | 0.0398 | 0.9598 | 4 | 0 |
| A01 MS-TCN s1 | selection | 0.9534 | 0.9509 | 0.9634 | 0.9571 | 0.9771 | 0.0423 | 0.0174 | 0.9619 | 1 | 1 |
| A01 MS-TCN s1 | calibration | 0.9740 | 0.9749 | 0.9774 | 0.9762 | 0.9780 | 0.0506 | 0.0249 | 0.9565 | 1 | 2 |
| A01 ASRF s1 | selection | 0.9476 | 0.9784 | 0.9476 | 0.9628 | 0.9771 | 0.0035 | 0.0304 | 0.9612 | 1 | 1 |
| A01 ASRF s1 | calibration | 0.9747 | 0.9923 | 0.9724 | 0.9822 | 0.9723 | 0.0084 | 0.0100 | 0.9677 | 3 | 1 |
| FACT s1 | selection | 0.9570 | 0.9551 | 0.9476 | 0.9514 | 0.9869 | 0.0035 | 0.0000 | 0.9687 | 0 | 1 |
| FACT s1 | calibration | 0.9584 | 0.9747 | 0.9673 | 0.9710 | 0.9856 | 0.0042 | 0.0149 | 0.9679 | 1 | 1 |
| FACT frame branch s1 | selection | 0.9583 | 0.9557 | 0.9607 | 0.9582 | 0.9976 | 0.0000 | 0.0000 | 0.9615 | 0 | 0 |
| FACT frame branch s1 | calibration | 0.9662 | 0.9674 | 0.9698 | 0.9686 | 0.9856 | 0.0084 | 0.0199 | 0.9709 | 1 | 1 |
| frozen Ordered + B3 s2 | selection | 0.9644 | 0.9588 | 0.9738 | 0.9662 | 0.9701 | 0.0458 | 0.0043 | 0.9569 | 3 | 0 |
| frozen Ordered + B3 s2 | calibration | 0.9797 | 0.9774 | 0.9799 | 0.9787 | 0.9861 | 0.0127 | 0.0249 | 0.9708 | 2 | 0 |
| frozen Ordered argmax s2 | selection | 0.9259 | 0.8744 | 0.9843 | 0.9261 | 0.9701 | 0.1162 | 0.0261 | 0.9596 | 3 | 0 |
| frozen Ordered argmax s2 | calibration | 0.9602 | 0.9399 | 0.9824 | 0.9607 | 0.9861 | 0.0970 | 0.0348 | 0.9543 | 2 | 0 |
| A01 Linear + B3 s2 | selection | 0.9388 | 0.9115 | 0.9712 | 0.9404 | 0.9786 | 0.0845 | 0.0304 | 0.9571 | 2 | 0 |
| A01 Linear + B3 s2 | calibration | 0.9732 | 0.9532 | 0.9724 | 0.9627 | 0.9861 | 0.0422 | 0.0299 | 0.9709 | 2 | 0 |
| A01 Linear argmax s2 | selection | 0.9082 | 0.8408 | 0.9817 | 0.9058 | 0.9786 | 0.1620 | 0.0348 | 0.9368 | 2 | 0 |
| A01 Linear argmax s2 | calibration | 0.9500 | 0.9173 | 0.9749 | 0.9452 | 0.9796 | 0.1308 | 0.0398 | 0.9543 | 3 | 0 |
| A01 MS-TCN s2 | selection | 0.9427 | 0.9385 | 0.9581 | 0.9482 | 0.9753 | 0.0563 | 0.0130 | 0.9710 | 0 | 2 |
| A01 MS-TCN s2 | calibration | 0.9626 | 0.9531 | 0.9698 | 0.9614 | 0.9780 | 0.0549 | 0.0348 | 0.9825 | 1 | 2 |
| A01 ASRF s2 | selection | 0.9506 | 0.9783 | 0.9450 | 0.9614 | 0.9869 | 0.0000 | 0.0304 | 0.9709 | 0 | 1 |
| A01 ASRF s2 | calibration | 0.9698 | 0.9845 | 0.9573 | 0.9707 | 0.9929 | 0.0084 | 0.0149 | 0.9736 | 1 | 0 |
| FACT s2 | selection | 0.9489 | 0.9549 | 0.9424 | 0.9486 | 0.9976 | 0.0000 | 0.0043 | 0.9567 | 0 | 0 |
| FACT s2 | calibration | 0.9611 | 0.9744 | 0.9548 | 0.9645 | 0.9927 | 0.0084 | 0.0199 | 0.9679 | 0 | 1 |
| FACT frame branch s2 | selection | 0.9504 | 0.9456 | 0.9555 | 0.9505 | 0.9976 | 0.0000 | 0.0087 | 0.9592 | 0 | 0 |
| FACT frame branch s2 | calibration | 0.9632 | 0.9721 | 0.9623 | 0.9672 | 0.9856 | 0.0084 | 0.0199 | 0.9738 | 1 | 1 |

## Paired comparisons (seed | comparison | half; 95% bootstrap intervals)

| comparison | multi segment F1@50 | multi boundary F1 | multi boundary recall | native macro-F1 | native cuts per clip | same-class cut rate | multi FP segments | multi FN segments |
|---|---|---|---|---|---|---|---|---|
| s0|FACT - A01 ASRF|selection | +0.0245 [+0.0071, +0.0431] | +0.0004 [-0.0165, +0.0154] | +0.0105 [-0.0082, +0.0289] | -0.0213 [-0.0804, +0.0192] | -0.0141 [-0.0385, +0.0000] | -0.0304 [-0.0612, -0.0043] | -8.0000 [-17.0000, +1.0000] | -16.0000 [-27.0000, -6.0000] |
| s0|FACT - A01 Linear + B3|selection | +0.0138 [-0.0018, +0.0304] | +0.0076 [-0.0089, +0.0248] | +0.0026 [-0.0107, +0.0165] | -0.0130 [-0.0609, +0.0207] | -0.0458 [-0.0930, -0.0041] | -0.0217 [-0.0469, -0.0041] | -9.0000 [-20.0000, +2.0000] | -5.0000 [-13.0000, +3.0000] |
| s0|FACT - frozen Ordered + B3|selection | +0.0225 [+0.0015, +0.0438] | +0.0088 [-0.0065, +0.0251] | +0.0000 [-0.0181, +0.0171] | -0.0130 [-0.0609, +0.0207] | -0.0423 [-0.0712, -0.0119] | -0.0130 [-0.0300, +0.0000] | -15.0000 [-28.0000, -3.0000] | -8.0000 [-19.0000, +3.0000] |
| s0|FACT - A01 MS-TCN|selection | +0.0411 [+0.0190, +0.0665] | +0.0206 [+0.0014, +0.0413] | -0.0079 [-0.0205, +0.0027] | +0.0030 [-0.0322, +0.0318] | -0.0739 [-0.1263, -0.0226] | -0.0304 [-0.0601, -0.0044] | -33.0000 [-52.0000, -17.0000] | -10.0000 [-19.0000, -2.0000] |
| s1|FACT - A01 ASRF|selection | +0.0095 [-0.0106, +0.0321] | -0.0114 [-0.0262, +0.0031] | +0.0000 [-0.0207, +0.0204] | +0.0098 [+0.0000, +0.0313] | +0.0000 [-0.0101, +0.0115] | -0.0304 [-0.0578, -0.0084] | +0.0000 [-12.0000, +11.0000] | -9.0000 [-21.0000, +3.0000] |
| s1|FACT - A01 Linear + B3|selection | +0.0002 [-0.0131, +0.0127] | -0.0103 [-0.0252, +0.0028] | -0.0052 [-0.0180, +0.0057] | +0.0083 [-0.0267, +0.0457] | -0.0352 [-0.0836, +0.0047] | -0.0087 [-0.0223, +0.0000] | +2.0000 [-7.0000, +11.0000] | -2.0000 [-9.0000, +6.0000] |
| s1|FACT - frozen Ordered + B3|selection | -0.0033 [-0.0177, +0.0118] | -0.0122 [-0.0295, +0.0035] | -0.0209 [-0.0365, -0.0077] | +0.0083 [-0.0267, +0.0457] | -0.0352 [-0.0593, -0.0083] | -0.0043 [-0.0138, +0.0000] | -2.0000 [-12.0000, +8.0000] | +5.0000 [-2.0000, +13.0000] |
| s1|FACT - A01 MS-TCN|selection | +0.0036 [-0.0146, +0.0222] | -0.0057 [-0.0245, +0.0119] | -0.0157 [-0.0369, +0.0027] | +0.0098 [+0.0000, +0.0362] | -0.0387 [-0.0736, -0.0080] | -0.0174 [-0.0372, -0.0040] | -6.0000 [-19.0000, +6.0000] | +2.0000 [-7.0000, +12.0000] |
| s2|FACT - A01 ASRF|selection | -0.0016 [-0.0195, +0.0165] | -0.0128 [-0.0319, +0.0060] | -0.0026 [-0.0279, +0.0206] | +0.0107 [+0.0000, +0.0390] | +0.0000 [+0.0000, +0.0000] | -0.0261 [-0.0565, +0.0000] | +5.0000 [-6.0000, +15.0000] | -3.0000 [-13.0000, +8.0000] |
| s2|FACT - A01 Linear + B3|selection | +0.0102 [-0.0099, +0.0318] | +0.0082 [-0.0113, +0.0292] | -0.0288 [-0.0498, -0.0105] | +0.0190 [+0.0000, +0.0495] | -0.0845 [-0.1469, -0.0244] | -0.0261 [-0.0524, -0.0044] | -21.0000 [-37.0000, -7.0000] | +9.0000 [-1.0000, +21.0000] |
| s2|FACT - frozen Ordered + B3|selection | -0.0154 [-0.0331, +0.0006] | -0.0176 [-0.0338, -0.0020] | -0.0314 [-0.0522, -0.0130] | +0.0275 [+0.0000, +0.0546] | -0.0458 [-0.0796, -0.0090] | +0.0000 [+0.0000, +0.0000] | +2.0000 [-7.0000, +11.0000] | +13.0000 [+3.0000, +25.0250] |
| s2|FACT - A01 MS-TCN|selection | +0.0063 [-0.0136, +0.0273] | +0.0004 [-0.0195, +0.0208] | -0.0157 [-0.0395, +0.0055] | +0.0223 [+0.0000, +0.0549] | -0.0563 [-0.1228, -0.0041] | -0.0087 [-0.0261, +0.0084] | -10.0000 [-25.0000, +2.0250] | +3.0000 [-7.0000, +14.0000] |
| s0|FACT - A01 ASRF|calibration | -0.0175 [-0.0334, -0.0036] | -0.0152 [-0.0263, -0.0060] | -0.0151 [-0.0320, -0.0024] | +0.0000 [+0.0000, +0.0000] | +0.0000 [-0.0114, +0.0129] | +0.0050 [-0.0106, +0.0219] | +9.0000 [+2.0000, +17.0000] | +9.0000 [+1.0000, +19.0000] |
| s0|FACT - A01 Linear + B3|calibration | -0.0148 [-0.0304, -0.0011] | -0.0092 [-0.0241, +0.0049] | -0.0201 [-0.0428, +0.0000] | -0.0082 [-0.0410, +0.0149] | -0.0042 [-0.0190, +0.0099] | +0.0000 [-0.0201, +0.0206] | +3.0000 [-5.0000, +10.0000] | +12.0000 [+2.0000, +24.0000] |
| s0|FACT - frozen Ordered + B3|calibration | -0.0197 [-0.0340, -0.0073] | -0.0130 [-0.0258, -0.0015] | -0.0276 [-0.0480, -0.0105] | -0.0150 [-0.0438, +0.0000] | -0.0169 [-0.0376, +0.0000] | -0.0050 [-0.0227, +0.0111] | +4.0000 [-3.0000, +10.0000] | +16.0000 [+7.0000, +27.0000] |
| s0|FACT - A01 MS-TCN|calibration | -0.0052 [-0.0204, +0.0090] | +0.0034 [-0.0103, +0.0181] | -0.0075 [-0.0268, +0.0103] | +0.0000 [+0.0000, +0.0000] | -0.0295 [-0.0656, +0.0000] | -0.0050 [-0.0280, +0.0160] | -2.0000 [-9.0000, +5.0000] | +7.0000 [-2.0000, +17.0000] |
| s1|FACT - A01 ASRF|calibration | -0.0163 [-0.0355, +0.0013] | -0.0112 [-0.0225, -0.0011] | -0.0050 [-0.0210, +0.0097] | +0.0133 [+0.0000, +0.0324] | -0.0042 [-0.0138, +0.0000] | +0.0050 [-0.0106, +0.0219] | +11.0000 [+3.0000, +21.0000] | +6.0000 [-5.0000, +18.0000] |
| s1|FACT - A01 Linear + B3|calibration | -0.0195 [-0.0348, -0.0068] | -0.0052 [-0.0175, +0.0066] | -0.0126 [-0.0319, +0.0025] | -0.0005 [-0.0321, +0.0263] | -0.0127 [-0.0284, +0.0000] | +0.0000 [-0.0141, +0.0151] | +7.0000 [-1.0000, +15.0000] | +13.0000 [+5.0000, +24.0000] |
| s1|FACT - frozen Ordered + B3|calibration | -0.0253 [-0.0410, -0.0114] | -0.0127 [-0.0245, -0.0016] | -0.0201 [-0.0390, -0.0051] | -0.0005 [-0.0321, +0.0263] | -0.0127 [-0.0284, +0.0000] | -0.0100 [-0.0246, +0.0000] | +10.0000 [+2.0000, +19.0000] | +16.0000 [+7.0000, +27.0000] |
| s1|FACT - A01 MS-TCN|calibration | -0.0156 [-0.0340, +0.0016] | -0.0052 [-0.0186, +0.0076] | -0.0101 [-0.0301, +0.0051] | +0.0076 [-0.0164, +0.0367] | -0.0464 [-0.1231, +0.0000] | -0.0100 [-0.0319, +0.0097] | +6.0000 [-5.0000, +16.0250] | +10.0000 [+1.0000, +21.0000] |
| s2|FACT - A01 ASRF|calibration | -0.0087 [-0.0224, +0.0046] | -0.0062 [-0.0184, +0.0064] | -0.0025 [-0.0152, +0.0104] | -0.0003 [-0.0243, +0.0226] | +0.0000 [-0.0111, +0.0133] | +0.0050 [-0.0156, +0.0279] | +6.0000 [-3.0000, +15.0000] | +3.0000 [-4.0000, +11.0000] |
| s2|FACT - A01 Linear + B3|calibration | -0.0121 [-0.0264, -0.0003] | +0.0018 [-0.0130, +0.0155] | -0.0176 [-0.0397, +0.0000] | +0.0065 [-0.0199, +0.0299] | -0.0338 [-0.0813, -0.0040] | -0.0100 [-0.0300, +0.0101] | -2.0000 [-9.0000, +4.0000] | +14.0000 [+6.0000, +25.0000] |
| s2|FACT - frozen Ordered + B3|calibration | -0.0187 [-0.0326, -0.0067] | -0.0142 [-0.0268, -0.0027] | -0.0251 [-0.0450, -0.0100] | +0.0065 [-0.0199, +0.0299] | -0.0042 [-0.0138, +0.0000] | -0.0050 [-0.0227, +0.0111] | +5.0000 [-2.0000, +12.0000] | +14.0000 [+6.0000, +25.0000] |
| s2|FACT - A01 MS-TCN|calibration | -0.0015 [-0.0170, +0.0146] | +0.0031 [-0.0104, +0.0166] | -0.0151 [-0.0319, +0.0000] | +0.0147 [-0.0152, +0.0473] | -0.0464 [-0.1137, +0.0000] | -0.0149 [-0.0415, +0.0101] | -7.0000 [-19.0000, +3.0000] | +8.0000 [+0.0000, +17.0000] |

## Three-seed verdicts

| comparison | metric | selection | calibration |
|---|---|---|
| FACT - A01 ASRF|multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| FACT - A01 ASRF|multi boundary F1 | no consistent difference detected | no consistent difference detected |
| FACT - A01 ASRF|multi boundary recall | no consistent difference detected | no consistent difference detected |
| FACT - A01 ASRF|native macro-F1 | no consistent difference detected | no consistent difference detected |
| FACT - A01 ASRF|native cuts per clip | no consistent difference detected | no consistent difference detected |
| FACT - A01 ASRF|same-class cut rate | no consistent difference detected | no consistent difference detected |
| FACT - A01 ASRF|multi FP segments | no consistent difference detected | no consistent difference detected |
| FACT - A01 ASRF|multi FN segments | no consistent difference detected | no consistent difference detected |
| FACT - A01 Linear + B3|multi segment F1@50 | no consistent difference detected | consistent negative |
| FACT - A01 Linear + B3|multi boundary F1 | no consistent difference detected | no consistent difference detected |
| FACT - A01 Linear + B3|multi boundary recall | no consistent difference detected | no consistent difference detected |
| FACT - A01 Linear + B3|native macro-F1 | no consistent difference detected | no consistent difference detected |
| FACT - A01 Linear + B3|native cuts per clip | no consistent difference detected | no consistent difference detected |
| FACT - A01 Linear + B3|same-class cut rate | no consistent difference detected | no consistent difference detected |
| FACT - A01 Linear + B3|multi FP segments | no consistent difference detected | no consistent difference detected |
| FACT - A01 Linear + B3|multi FN segments | no consistent difference detected | consistent positive |
| FACT - frozen Ordered + B3|multi segment F1@50 | no consistent difference detected | consistent negative |
| FACT - frozen Ordered + B3|multi boundary F1 | no consistent difference detected | consistent negative |
| FACT - frozen Ordered + B3|multi boundary recall | no consistent difference detected | consistent negative |
| FACT - frozen Ordered + B3|native macro-F1 | no consistent difference detected | no consistent difference detected |
| FACT - frozen Ordered + B3|native cuts per clip | consistent negative | no consistent difference detected |
| FACT - frozen Ordered + B3|same-class cut rate | no consistent difference detected | no consistent difference detected |
| FACT - frozen Ordered + B3|multi FP segments | no consistent difference detected | no consistent difference detected |
| FACT - frozen Ordered + B3|multi FN segments | no consistent difference detected | consistent positive |
| FACT - A01 MS-TCN|multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| FACT - A01 MS-TCN|multi boundary F1 | no consistent difference detected | no consistent difference detected |
| FACT - A01 MS-TCN|multi boundary recall | no consistent difference detected | no consistent difference detected |
| FACT - A01 MS-TCN|native macro-F1 | no consistent difference detected | no consistent difference detected |
| FACT - A01 MS-TCN|native cuts per clip | consistent negative | no consistent difference detected |
| FACT - A01 MS-TCN|same-class cut rate | no consistent difference detected | no consistent difference detected |
| FACT - A01 MS-TCN|multi FP segments | no consistent difference detected | no consistent difference detected |
| FACT - A01 MS-TCN|multi FN segments | no consistent difference detected | no consistent difference detected |

## Reference segments recovered and lost by FACT against each control

| seed | pair | half | population | TP FACT | FP FACT | FN FACT | TP control | FP control | FN control | recovered | recovered sessions | lost | lost sessions | recovered <=16 | lost <=16 | recovered X-Y-X | lost X-Y-X | new unmatched pred | removed unmatched pred |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0|FACT vs A01 ASRF|selection | native | 280 | 4 | 4 | 281 | 7 | 3 | 2 | 2 | 3 | 3 | 0 | 2 | 0 | 0 | 3 | 6 |
| s0|FACT vs A01 ASRF|selection | pairs | 200 | 7 | 7 | 200 | 5 | 7 | 3 | 3 | 3 | 4 | 1 | 0 | 0 | 0 | 7 | 5 |
| s0|FACT vs A01 ASRF|selection | multi | 481 | 18 | 21 | 465 | 26 | 37 | 20 | 11 | 4 | 5 | 4 | 1 | 0 | 0 | 16 | 24 |
| s0|FACT vs A01 Linear + B3|selection | native | 280 | 4 | 4 | 277 | 20 | 7 | 5 | 4 | 2 | 2 | 0 | 2 | 0 | 0 | 2 | 18 |
| s0|FACT vs A01 Linear + B3|selection | pairs | 200 | 7 | 7 | 201 | 7 | 6 | 2 | 2 | 3 | 3 | 1 | 1 | 0 | 0 | 7 | 7 |
| s0|FACT vs A01 Linear + B3|selection | multi | 481 | 18 | 21 | 476 | 27 | 26 | 10 | 9 | 5 | 6 | 2 | 2 | 0 | 0 | 18 | 27 |
| s0|FACT vs frozen Ordered + B3|selection | native | 280 | 4 | 4 | 281 | 15 | 3 | 1 | 1 | 2 | 2 | 0 | 2 | 0 | 0 | 3 | 14 |
| s0|FACT vs frozen Ordered + B3|selection | pairs | 200 | 7 | 7 | 200 | 9 | 7 | 1 | 1 | 1 | 2 | 0 | 0 | 0 | 0 | 7 | 9 |
| s0|FACT vs frozen Ordered + B3|selection | multi | 481 | 18 | 21 | 473 | 33 | 29 | 16 | 9 | 8 | 7 | 5 | 4 | 0 | 2 | 18 | 33 |
| s0|FACT vs A01 MS-TCN|selection | native | 280 | 4 | 4 | 278 | 27 | 6 | 3 | 2 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 24 |
| s0|FACT vs A01 MS-TCN|selection | pairs | 200 | 7 | 7 | 201 | 8 | 6 | 2 | 2 | 3 | 4 | 1 | 0 | 0 | 0 | 7 | 8 |
| s0|FACT vs A01 MS-TCN|selection | multi | 481 | 18 | 21 | 471 | 51 | 31 | 14 | 9 | 4 | 5 | 2 | 1 | 0 | 0 | 16 | 49 |
| s1|FACT vs A01 ASRF|selection | native | 282 | 3 | 2 | 281 | 4 | 3 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 2 |
| s1|FACT vs A01 ASRF|selection | pairs | 201 | 7 | 6 | 198 | 7 | 9 | 4 | 4 | 1 | 1 | 2 | 0 | 0 | 0 | 7 | 7 |
| s1|FACT vs A01 ASRF|selection | multi | 479 | 20 | 23 | 470 | 20 | 32 | 15 | 12 | 6 | 4 | 5 | 3 | 0 | 2 | 20 | 20 |
| s1|FACT vs A01 Linear + B3|selection | native | 282 | 3 | 2 | 278 | 17 | 6 | 5 | 4 | 1 | 1 | 0 | 1 | 0 | 0 | 2 | 16 |
| s1|FACT vs A01 Linear + B3|selection | pairs | 201 | 7 | 6 | 199 | 12 | 8 | 2 | 2 | 0 | 0 | 1 | 0 | 0 | 0 | 6 | 11 |
| s1|FACT vs A01 Linear + B3|selection | multi | 479 | 20 | 23 | 477 | 18 | 25 | 7 | 6 | 5 | 4 | 2 | 3 | 0 | 2 | 20 | 18 |
| s1|FACT vs frozen Ordered + B3|selection | native | 282 | 3 | 2 | 280 | 15 | 4 | 3 | 3 | 1 | 1 | 0 | 1 | 0 | 0 | 2 | 14 |
| s1|FACT vs frozen Ordered + B3|selection | pairs | 201 | 7 | 6 | 200 | 10 | 7 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 9 |
| s1|FACT vs frozen Ordered + B3|selection | multi | 479 | 20 | 23 | 484 | 22 | 18 | 4 | 4 | 9 | 7 | 0 | 6 | 0 | 2 | 20 | 22 |
| s1|FACT vs A01 MS-TCN|selection | native | 282 | 3 | 2 | 280 | 16 | 4 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 13 |
| s1|FACT vs A01 MS-TCN|selection | pairs | 201 | 7 | 6 | 202 | 11 | 5 | 2 | 2 | 3 | 2 | 1 | 1 | 0 | 0 | 7 | 11 |
| s1|FACT vs A01 MS-TCN|selection | multi | 479 | 20 | 23 | 481 | 26 | 21 | 7 | 6 | 9 | 7 | 3 | 6 | 0 | 2 | 19 | 25 |
| s2|FACT vs A01 ASRF|selection | native | 283 | 1 | 1 | 282 | 2 | 2 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 |
| s2|FACT vs A01 ASRF|selection | pairs | 199 | 10 | 8 | 200 | 5 | 7 | 2 | 2 | 3 | 3 | 1 | 0 | 0 | 0 | 10 | 5 |
| s2|FACT vs A01 ASRF|selection | multi | 474 | 23 | 28 | 471 | 18 | 31 | 11 | 8 | 8 | 7 | 7 | 4 | 1 | 2 | 23 | 18 |
| s2|FACT vs A01 Linear + B3|selection | native | 283 | 1 | 1 | 277 | 31 | 7 | 6 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 30 |
| s2|FACT vs A01 Linear + B3|selection | pairs | 199 | 10 | 8 | 201 | 12 | 6 | 0 | 0 | 2 | 2 | 0 | 1 | 0 | 0 | 7 | 9 |
| s2|FACT vs A01 Linear + B3|selection | multi | 474 | 23 | 28 | 483 | 44 | 19 | 5 | 4 | 14 | 12 | 0 | 8 | 0 | 4 | 23 | 44 |
| s2|FACT vs frozen Ordered + B3|selection | native | 283 | 1 | 1 | 278 | 19 | 6 | 5 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 18 |
| s2|FACT vs frozen Ordered + B3|selection | pairs | 199 | 10 | 8 | 200 | 11 | 7 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 7 | 8 |
| s2|FACT vs frozen Ordered + B3|selection | multi | 474 | 23 | 28 | 487 | 21 | 15 | 3 | 3 | 16 | 14 | 0 | 9 | 0 | 4 | 23 | 21 |
| s2|FACT vs A01 MS-TCN|selection | native | 283 | 1 | 1 | 280 | 20 | 4 | 3 | 3 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 19 |
| s2|FACT vs A01 MS-TCN|selection | pairs | 199 | 10 | 8 | 201 | 6 | 6 | 1 | 1 | 3 | 3 | 1 | 0 | 0 | 0 | 10 | 6 |
| s2|FACT vs A01 MS-TCN|selection | multi | 474 | 23 | 28 | 477 | 33 | 25 | 7 | 6 | 10 | 9 | 4 | 5 | 1 | 2 | 21 | 31 |
| s0|FACT vs A01 ASRF|calibration | native | 234 | 5 | 3 | 234 | 5 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 3 |
| s0|FACT vs A01 ASRF|calibration | pairs | 168 | 1 | 5 | 168 | 2 | 5 | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 1 |
| s0|FACT vs A01 ASRF|calibration | multi | 494 | 15 | 24 | 503 | 6 | 15 | 3 | 5 | 12 | 12 | 1 | 6 | 0 | 1 | 15 | 6 |
| s0|FACT vs A01 Linear + B3|calibration | native | 234 | 5 | 3 | 235 | 5 | 2 | 1 | 1 | 2 | 2 | 0 | 0 | 0 | 0 | 4 | 4 |
| s0|FACT vs A01 Linear + B3|calibration | pairs | 168 | 1 | 5 | 167 | 4 | 6 | 2 | 3 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 3 |
| s0|FACT vs A01 Linear + B3|calibration | multi | 494 | 15 | 24 | 506 | 12 | 12 | 3 | 3 | 15 | 11 | 2 | 9 | 0 | 3 | 15 | 12 |
| s0|FACT vs frozen Ordered + B3|calibration | native | 234 | 5 | 3 | 236 | 7 | 1 | 0 | 0 | 2 | 2 | 0 | 0 | 0 | 0 | 4 | 6 |
| s0|FACT vs frozen Ordered + B3|calibration | pairs | 168 | 1 | 5 | 168 | 1 | 5 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| s0|FACT vs frozen Ordered + B3|calibration | multi | 494 | 15 | 24 | 510 | 11 | 8 | 0 | 0 | 16 | 13 | 0 | 9 | 0 | 3 | 15 | 11 |
| s0|FACT vs A01 MS-TCN|calibration | native | 234 | 5 | 3 | 234 | 12 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 12 |
| s0|FACT vs A01 MS-TCN|calibration | pairs | 168 | 1 | 5 | 167 | 3 | 6 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 3 |
| s0|FACT vs A01 MS-TCN|calibration | multi | 494 | 15 | 24 | 501 | 17 | 17 | 5 | 6 | 12 | 10 | 4 | 9 | 0 | 2 | 15 | 17 |
| s1|FACT vs A01 ASRF|calibration | native | 235 | 3 | 2 | 233 | 6 | 4 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 4 |
| s1|FACT vs A01 ASRF|calibration | pairs | 166 | 4 | 7 | 165 | 3 | 8 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 2 |
| s1|FACT vs A01 ASRF|calibration | multi | 495 | 20 | 23 | 501 | 9 | 17 | 8 | 11 | 14 | 12 | 2 | 10 | 0 | 2 | 20 | 9 |
| s1|FACT vs A01 Linear + B3|calibration | native | 235 | 3 | 2 | 235 | 6 | 2 | 2 | 2 | 2 | 2 | 0 | 0 | 0 | 0 | 3 | 6 |
| s1|FACT vs A01 Linear + B3|calibration | pairs | 166 | 4 | 7 | 167 | 4 | 6 | 1 | 2 | 2 | 2 | 0 | 2 | 0 | 0 | 2 | 2 |
| s1|FACT vs A01 Linear + B3|calibration | multi | 495 | 20 | 23 | 508 | 13 | 10 | 0 | 0 | 13 | 10 | 0 | 8 | 0 | 3 | 20 | 13 |
| s1|FACT vs frozen Ordered + B3|calibration | native | 235 | 3 | 2 | 235 | 6 | 2 | 2 | 2 | 2 | 2 | 0 | 0 | 0 | 0 | 3 | 6 |
| s1|FACT vs frozen Ordered + B3|calibration | pairs | 166 | 4 | 7 | 167 | 4 | 6 | 1 | 2 | 2 | 2 | 0 | 2 | 0 | 0 | 2 | 2 |
| s1|FACT vs frozen Ordered + B3|calibration | multi | 495 | 20 | 23 | 511 | 10 | 7 | 0 | 0 | 16 | 12 | 0 | 10 | 0 | 3 | 19 | 9 |
| s1|FACT vs A01 MS-TCN|calibration | native | 235 | 3 | 2 | 234 | 15 | 3 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 15 |
| s1|FACT vs A01 MS-TCN|calibration | pairs | 166 | 4 | 7 | 165 | 7 | 8 | 2 | 3 | 1 | 1 | 0 | 1 | 0 | 0 | 4 | 7 |
| s1|FACT vs A01 MS-TCN|calibration | multi | 495 | 20 | 23 | 505 | 14 | 13 | 3 | 5 | 13 | 10 | 1 | 8 | 0 | 2 | 18 | 12 |
| s2|FACT vs A01 ASRF|calibration | native | 236 | 3 | 1 | 236 | 3 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 3 | 3 |
| s2|FACT vs A01 ASRF|calibration | pairs | 166 | 4 | 7 | 166 | 2 | 7 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 4 | 2 |
| s2|FACT vs A01 ASRF|calibration | multi | 494 | 16 | 24 | 497 | 10 | 21 | 4 | 5 | 7 | 7 | 2 | 5 | 0 | 1 | 14 | 8 |
| s2|FACT vs A01 Linear + B3|calibration | native | 236 | 3 | 1 | 234 | 13 | 3 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 13 |
| s2|FACT vs A01 Linear + B3|calibration | pairs | 166 | 4 | 7 | 167 | 4 | 6 | 2 | 3 | 3 | 3 | 1 | 1 | 0 | 0 | 3 | 3 |
| s2|FACT vs A01 Linear + B3|calibration | multi | 494 | 16 | 24 | 508 | 18 | 10 | 0 | 0 | 14 | 11 | 0 | 7 | 0 | 3 | 16 | 18 |
| s2|FACT vs frozen Ordered + B3|calibration | native | 236 | 3 | 1 | 235 | 5 | 2 | 2 | 2 | 1 | 1 | 0 | 0 | 0 | 0 | 3 | 5 |
| s2|FACT vs frozen Ordered + B3|calibration | pairs | 166 | 4 | 7 | 166 | 3 | 7 | 3 | 3 | 3 | 3 | 2 | 1 | 0 | 0 | 3 | 2 |
| s2|FACT vs frozen Ordered + B3|calibration | multi | 494 | 16 | 24 | 508 | 11 | 10 | 0 | 0 | 14 | 11 | 0 | 8 | 0 | 3 | 16 | 11 |
| s2|FACT vs A01 MS-TCN|calibration | native | 236 | 3 | 1 | 234 | 16 | 3 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 16 |
| s2|FACT vs A01 MS-TCN|calibration | pairs | 166 | 4 | 7 | 168 | 1 | 5 | 0 | 0 | 2 | 2 | 0 | 0 | 0 | 0 | 4 | 1 |
| s2|FACT vs A01 MS-TCN|calibration | multi | 494 | 16 | 24 | 502 | 23 | 16 | 5 | 8 | 13 | 8 | 2 | 9 | 0 | 1 | 15 | 22 |

## Short segments and X-Y-X middles (multi streams, raw denominators)

| system | half | unit | segments | matched | recall | wrong view | no segment at IoU 0.5 | missed sessions |
|---|---|---|---|---|---|---|---|
| frozen Ordered + B3 s0 | selection | short segments | 141 | 130 | 0.9220 | 2 | 9 | 8 |
| frozen Ordered + B3 s0 | selection | short A4C or A5C segments | 65 | 54 | 0.8308 | 2 | 9 | 8 |
| frozen Ordered + B3 s0 | selection | X-Y-X middles | 69 | 66 | 0.9565 | 1 | 2 | 3 |
| frozen Ordered + B3 s0 | selection | A4C/A5C X-Y-X middles | 7 | 5 | 0.7143 | 0 | 2 | 2 |
| frozen Ordered + B3 s0 | calibration | short segments | 134 | 128 | 0.9552 | 1 | 5 | 4 |
| frozen Ordered + B3 s0 | calibration | short A4C or A5C segments | 67 | 66 | 0.9851 | 1 | 0 | 1 |
| frozen Ordered + B3 s0 | calibration | X-Y-X middles | 80 | 80 | 1.0000 | 0 | 0 | 0 |
| frozen Ordered + B3 s0 | calibration | A4C/A5C X-Y-X middles | 7 | 7 | 1.0000 | 0 | 0 | 0 |
| A01 Linear + B3 s0 | selection | short segments | 141 | 131 | 0.9291 | 0 | 10 | 6 |
| A01 Linear + B3 s0 | selection | short A4C or A5C segments | 65 | 55 | 0.8462 | 0 | 10 | 6 |
| A01 Linear + B3 s0 | selection | X-Y-X middles | 69 | 64 | 0.9275 | 1 | 4 | 6 |
| A01 Linear + B3 s0 | selection | A4C/A5C X-Y-X middles | 7 | 3 | 0.4286 | 0 | 4 | 5 |
| A01 Linear + B3 s0 | calibration | short segments | 134 | 126 | 0.9403 | 2 | 6 | 6 |
| A01 Linear + B3 s0 | calibration | short A4C or A5C segments | 67 | 63 | 0.9403 | 2 | 2 | 4 |
| A01 Linear + B3 s0 | calibration | X-Y-X middles | 80 | 80 | 1.0000 | 0 | 0 | 0 |
| A01 Linear + B3 s0 | calibration | A4C/A5C X-Y-X middles | 7 | 7 | 1.0000 | 0 | 0 | 0 |
| A01 MS-TCN s0 | selection | short segments | 141 | 130 | 0.9220 | 1 | 10 | 8 |
| A01 MS-TCN s0 | selection | short A4C or A5C segments | 65 | 56 | 0.8615 | 0 | 9 | 6 |
| A01 MS-TCN s0 | selection | X-Y-X middles | 69 | 64 | 0.9275 | 1 | 4 | 6 |
| A01 MS-TCN s0 | selection | A4C/A5C X-Y-X middles | 7 | 3 | 0.4286 | 0 | 4 | 5 |
| A01 MS-TCN s0 | calibration | short segments | 134 | 124 | 0.9254 | 2 | 8 | 8 |
| A01 MS-TCN s0 | calibration | short A4C or A5C segments | 67 | 61 | 0.9104 | 2 | 4 | 6 |
| A01 MS-TCN s0 | calibration | X-Y-X middles | 80 | 78 | 0.9750 | 1 | 1 | 4 |
| A01 MS-TCN s0 | calibration | A4C/A5C X-Y-X middles | 7 | 6 | 0.8571 | 0 | 1 | 1 |
| A01 ASRF s0 | selection | short segments | 141 | 128 | 0.9078 | 0 | 13 | 8 |
| A01 ASRF s0 | selection | short A4C or A5C segments | 65 | 52 | 0.8000 | 0 | 13 | 8 |
| A01 ASRF s0 | selection | X-Y-X middles | 69 | 63 | 0.9130 | 1 | 5 | 7 |
| A01 ASRF s0 | selection | A4C/A5C X-Y-X middles | 7 | 3 | 0.4286 | 0 | 4 | 5 |
| A01 ASRF s0 | calibration | short segments | 134 | 124 | 0.9254 | 1 | 9 | 7 |
| A01 ASRF s0 | calibration | short A4C or A5C segments | 67 | 62 | 0.9254 | 1 | 4 | 4 |
| A01 ASRF s0 | calibration | X-Y-X middles | 80 | 77 | 0.9625 | 1 | 2 | 4 |
| A01 ASRF s0 | calibration | A4C/A5C X-Y-X middles | 7 | 5 | 0.7143 | 0 | 2 | 2 |
| FACT s0 | selection | short segments | 141 | 131 | 0.9291 | 0 | 10 | 7 |
| FACT s0 | selection | short A4C or A5C segments | 65 | 55 | 0.8462 | 0 | 10 | 7 |
| FACT s0 | selection | X-Y-X middles | 69 | 63 | 0.9130 | 1 | 5 | 6 |
| FACT s0 | selection | A4C/A5C X-Y-X middles | 7 | 3 | 0.4286 | 0 | 4 | 5 |
| FACT s0 | calibration | short segments | 134 | 119 | 0.8881 | 3 | 12 | 9 |
| FACT s0 | calibration | short A4C or A5C segments | 67 | 58 | 0.8657 | 3 | 6 | 6 |
| FACT s0 | calibration | X-Y-X middles | 80 | 76 | 0.9500 | 0 | 4 | 3 |
| FACT s0 | calibration | A4C/A5C X-Y-X middles | 7 | 4 | 0.5714 | 0 | 3 | 2 |
| FACT frame branch s0 | selection | short segments | 141 | 127 | 0.9007 | 0 | 14 | 9 |
| FACT frame branch s0 | selection | short A4C or A5C segments | 65 | 52 | 0.8000 | 0 | 13 | 8 |
| FACT frame branch s0 | selection | X-Y-X middles | 69 | 60 | 0.8696 | 1 | 8 | 8 |
| FACT frame branch s0 | selection | A4C/A5C X-Y-X middles | 7 | 0 | 0.0000 | 0 | 7 | 7 |
| FACT frame branch s0 | calibration | short segments | 134 | 123 | 0.9179 | 3 | 8 | 6 |
| FACT frame branch s0 | calibration | short A4C or A5C segments | 67 | 62 | 0.9254 | 3 | 2 | 3 |
| FACT frame branch s0 | calibration | X-Y-X middles | 80 | 76 | 0.9500 | 1 | 3 | 3 |
| FACT frame branch s0 | calibration | A4C/A5C X-Y-X middles | 7 | 5 | 0.7143 | 0 | 2 | 1 |
| frozen Ordered + B3 s1 | selection | short segments | 141 | 134 | 0.9504 | 0 | 7 | 6 |
| frozen Ordered + B3 s1 | selection | short A4C or A5C segments | 65 | 58 | 0.8923 | 0 | 7 | 6 |
| frozen Ordered + B3 s1 | selection | X-Y-X middles | 69 | 64 | 0.9275 | 1 | 4 | 6 |
| frozen Ordered + B3 s1 | selection | A4C/A5C X-Y-X middles | 7 | 3 | 0.4286 | 0 | 4 | 5 |
| frozen Ordered + B3 s1 | calibration | short segments | 134 | 129 | 0.9627 | 2 | 3 | 4 |
| frozen Ordered + B3 s1 | calibration | short A4C or A5C segments | 67 | 65 | 0.9701 | 2 | 0 | 2 |
| frozen Ordered + B3 s1 | calibration | X-Y-X middles | 80 | 80 | 1.0000 | 0 | 0 | 0 |
| frozen Ordered + B3 s1 | calibration | A4C/A5C X-Y-X middles | 7 | 7 | 1.0000 | 0 | 0 | 0 |
| A01 Linear + B3 s1 | selection | short segments | 141 | 129 | 0.9149 | 0 | 12 | 6 |
| A01 Linear + B3 s1 | selection | short A4C or A5C segments | 65 | 53 | 0.8154 | 0 | 12 | 6 |
| A01 Linear + B3 s1 | selection | X-Y-X middles | 69 | 64 | 0.9275 | 1 | 4 | 6 |
| A01 Linear + B3 s1 | selection | A4C/A5C X-Y-X middles | 7 | 3 | 0.4286 | 0 | 4 | 5 |
| A01 Linear + B3 s1 | calibration | short segments | 134 | 127 | 0.9478 | 3 | 4 | 5 |
| A01 Linear + B3 s1 | calibration | short A4C or A5C segments | 67 | 63 | 0.9403 | 3 | 1 | 3 |
| A01 Linear + B3 s1 | calibration | X-Y-X middles | 80 | 79 | 0.9875 | 1 | 0 | 1 |
| A01 Linear + B3 s1 | calibration | A4C/A5C X-Y-X middles | 7 | 7 | 1.0000 | 0 | 0 | 0 |
| A01 MS-TCN s1 | selection | short segments | 141 | 131 | 0.9291 | 1 | 9 | 7 |
| A01 MS-TCN s1 | selection | short A4C or A5C segments | 65 | 55 | 0.8462 | 1 | 9 | 7 |
| A01 MS-TCN s1 | selection | X-Y-X middles | 69 | 63 | 0.9130 | 1 | 5 | 6 |
| A01 MS-TCN s1 | selection | A4C/A5C X-Y-X middles | 7 | 3 | 0.4286 | 0 | 4 | 5 |
| A01 MS-TCN s1 | calibration | short segments | 134 | 126 | 0.9403 | 2 | 6 | 7 |
| A01 MS-TCN s1 | calibration | short A4C or A5C segments | 67 | 62 | 0.9254 | 2 | 3 | 5 |
| A01 MS-TCN s1 | calibration | X-Y-X middles | 80 | 78 | 0.9750 | 1 | 1 | 4 |
| A01 MS-TCN s1 | calibration | A4C/A5C X-Y-X middles | 7 | 6 | 0.8571 | 0 | 1 | 1 |
| A01 ASRF s1 | selection | short segments | 141 | 126 | 0.8936 | 1 | 14 | 10 |
| A01 ASRF s1 | selection | short A4C or A5C segments | 65 | 52 | 0.8000 | 1 | 12 | 8 |
| A01 ASRF s1 | selection | X-Y-X middles | 69 | 63 | 0.9130 | 1 | 5 | 6 |
| A01 ASRF s1 | selection | A4C/A5C X-Y-X middles | 7 | 3 | 0.4286 | 0 | 4 | 5 |
| A01 ASRF s1 | calibration | short segments | 134 | 127 | 0.9478 | 1 | 6 | 6 |
| A01 ASRF s1 | calibration | short A4C or A5C segments | 67 | 63 | 0.9403 | 1 | 3 | 4 |
| A01 ASRF s1 | calibration | X-Y-X middles | 80 | 78 | 0.9750 | 1 | 1 | 4 |
| A01 ASRF s1 | calibration | A4C/A5C X-Y-X middles | 7 | 6 | 0.8571 | 0 | 1 | 1 |
| FACT s1 | selection | short segments | 141 | 128 | 0.9078 | 0 | 13 | 7 |
| FACT s1 | selection | short A4C or A5C segments | 65 | 52 | 0.8000 | 0 | 13 | 7 |
| FACT s1 | selection | X-Y-X middles | 69 | 61 | 0.8841 | 1 | 7 | 8 |
| FACT s1 | selection | A4C/A5C X-Y-X middles | 7 | 1 | 0.1429 | 0 | 6 | 7 |
| FACT s1 | calibration | short segments | 134 | 119 | 0.8881 | 5 | 10 | 10 |
| FACT s1 | calibration | short A4C or A5C segments | 67 | 57 | 0.8507 | 5 | 5 | 8 |
| FACT s1 | calibration | X-Y-X middles | 80 | 75 | 0.9375 | 1 | 4 | 4 |
| FACT s1 | calibration | A4C/A5C X-Y-X middles | 7 | 4 | 0.5714 | 0 | 3 | 2 |
| FACT frame branch s1 | selection | short segments | 141 | 132 | 0.9362 | 0 | 9 | 7 |
| FACT frame branch s1 | selection | short A4C or A5C segments | 65 | 56 | 0.8615 | 0 | 9 | 7 |
| FACT frame branch s1 | selection | X-Y-X middles | 69 | 63 | 0.9130 | 1 | 5 | 7 |
| FACT frame branch s1 | selection | A4C/A5C X-Y-X middles | 7 | 2 | 0.2857 | 0 | 5 | 6 |
| FACT frame branch s1 | calibration | short segments | 134 | 124 | 0.9254 | 3 | 7 | 7 |
| FACT frame branch s1 | calibration | short A4C or A5C segments | 67 | 60 | 0.8955 | 3 | 4 | 5 |
| FACT frame branch s1 | calibration | X-Y-X middles | 80 | 76 | 0.9500 | 1 | 3 | 3 |
| FACT frame branch s1 | calibration | A4C/A5C X-Y-X middles | 7 | 4 | 0.5714 | 0 | 3 | 2 |
| frozen Ordered + B3 s2 | selection | short segments | 141 | 136 | 0.9645 | 0 | 5 | 5 |
| frozen Ordered + B3 s2 | selection | short A4C or A5C segments | 65 | 60 | 0.9231 | 0 | 5 | 5 |
| frozen Ordered + B3 s2 | selection | X-Y-X middles | 69 | 66 | 0.9565 | 1 | 2 | 3 |
| frozen Ordered + B3 s2 | selection | A4C/A5C X-Y-X middles | 7 | 5 | 0.7143 | 0 | 2 | 2 |
| frozen Ordered + B3 s2 | calibration | short segments | 134 | 127 | 0.9478 | 2 | 5 | 5 |
| frozen Ordered + B3 s2 | calibration | short A4C or A5C segments | 67 | 65 | 0.9701 | 2 | 0 | 2 |
| frozen Ordered + B3 s2 | calibration | X-Y-X middles | 80 | 80 | 1.0000 | 0 | 0 | 0 |
| frozen Ordered + B3 s2 | calibration | A4C/A5C X-Y-X middles | 7 | 7 | 1.0000 | 0 | 0 | 0 |
| A01 Linear + B3 s2 | selection | short segments | 141 | 135 | 0.9574 | 0 | 6 | 5 |
| A01 Linear + B3 s2 | selection | short A4C or A5C segments | 65 | 59 | 0.9077 | 0 | 6 | 5 |
| A01 Linear + B3 s2 | selection | X-Y-X middles | 69 | 66 | 0.9565 | 1 | 2 | 3 |
| A01 Linear + B3 s2 | selection | A4C/A5C X-Y-X middles | 7 | 5 | 0.7143 | 0 | 2 | 2 |
| A01 Linear + B3 s2 | calibration | short segments | 134 | 126 | 0.9403 | 3 | 5 | 5 |
| A01 Linear + B3 s2 | calibration | short A4C or A5C segments | 67 | 64 | 0.9552 | 3 | 0 | 2 |
| A01 Linear + B3 s2 | calibration | X-Y-X middles | 80 | 79 | 0.9875 | 1 | 0 | 1 |
| A01 Linear + B3 s2 | calibration | A4C/A5C X-Y-X middles | 7 | 7 | 1.0000 | 0 | 0 | 0 |
| A01 MS-TCN s2 | selection | short segments | 141 | 128 | 0.9078 | 1 | 12 | 9 |
| A01 MS-TCN s2 | selection | short A4C or A5C segments | 65 | 53 | 0.8154 | 1 | 11 | 9 |
| A01 MS-TCN s2 | selection | X-Y-X middles | 69 | 63 | 0.9130 | 1 | 5 | 7 |
| A01 MS-TCN s2 | selection | A4C/A5C X-Y-X middles | 7 | 2 | 0.2857 | 0 | 5 | 6 |
| A01 MS-TCN s2 | calibration | short segments | 134 | 126 | 0.9403 | 1 | 7 | 5 |
| A01 MS-TCN s2 | calibration | short A4C or A5C segments | 67 | 63 | 0.9403 | 1 | 3 | 3 |
| A01 MS-TCN s2 | calibration | X-Y-X middles | 80 | 77 | 0.9625 | 1 | 2 | 4 |
| A01 MS-TCN s2 | calibration | A4C/A5C X-Y-X middles | 7 | 5 | 0.7143 | 0 | 2 | 1 |
| A01 ASRF s2 | selection | short segments | 141 | 124 | 0.8794 | 0 | 17 | 11 |
| A01 ASRF s2 | selection | short A4C or A5C segments | 65 | 51 | 0.7846 | 0 | 14 | 9 |
| A01 ASRF s2 | selection | X-Y-X middles | 69 | 62 | 0.8986 | 1 | 6 | 8 |
| A01 ASRF s2 | selection | A4C/A5C X-Y-X middles | 7 | 2 | 0.2857 | 0 | 5 | 6 |
| A01 ASRF s2 | calibration | short segments | 134 | 122 | 0.9104 | 2 | 10 | 7 |
| A01 ASRF s2 | calibration | short A4C or A5C segments | 67 | 60 | 0.8955 | 2 | 5 | 5 |
| A01 ASRF s2 | calibration | X-Y-X middles | 80 | 78 | 0.9750 | 0 | 2 | 1 |
| A01 ASRF s2 | calibration | A4C/A5C X-Y-X middles | 7 | 5 | 0.7143 | 0 | 2 | 1 |
| FACT s2 | selection | short segments | 141 | 127 | 0.9007 | 1 | 13 | 9 |
| FACT s2 | selection | short A4C or A5C segments | 65 | 52 | 0.8000 | 1 | 12 | 8 |
| FACT s2 | selection | X-Y-X middles | 69 | 58 | 0.8406 | 1 | 10 | 11 |
| FACT s2 | selection | A4C/A5C X-Y-X middles | 7 | 1 | 0.1429 | 0 | 6 | 7 |
| FACT s2 | calibration | short segments | 134 | 119 | 0.8881 | 3 | 12 | 7 |
| FACT s2 | calibration | short A4C or A5C segments | 67 | 59 | 0.8806 | 3 | 5 | 5 |
| FACT s2 | calibration | X-Y-X middles | 80 | 76 | 0.9500 | 1 | 3 | 3 |
| FACT s2 | calibration | A4C/A5C X-Y-X middles | 7 | 4 | 0.5714 | 0 | 3 | 2 |
| FACT frame branch s2 | selection | short segments | 141 | 129 | 0.9149 | 0 | 12 | 8 |
| FACT frame branch s2 | selection | short A4C or A5C segments | 65 | 53 | 0.8154 | 0 | 12 | 8 |
| FACT frame branch s2 | selection | X-Y-X middles | 69 | 62 | 0.8986 | 1 | 6 | 8 |
| FACT frame branch s2 | selection | A4C/A5C X-Y-X middles | 7 | 1 | 0.1429 | 0 | 6 | 7 |
| FACT frame branch s2 | calibration | short segments | 134 | 121 | 0.9030 | 3 | 10 | 7 |
| FACT frame branch s2 | calibration | short A4C or A5C segments | 67 | 59 | 0.8806 | 3 | 5 | 5 |
| FACT frame branch s2 | calibration | X-Y-X middles | 80 | 76 | 0.9500 | 1 | 3 | 3 |
| FACT frame branch s2 | calibration | A4C/A5C X-Y-X middles | 7 | 4 | 0.5714 | 0 | 3 | 2 |

## FACT tokens (multi streams)

| seed | half | multi_streams | tokens_equal_segments | fewer_tokens | more_tokens | mean_tokens_minus_segments | sequences | outputs_with_a_token_on_separate_intervals |
|---|---|---|---|---|---|---|---|
| s0|selection | 120 | 81 | 7 | 32 | 0.2500 | 541 | 3 |
| s1|selection | 120 | 84 | 10 | 26 | 0.1417 | 541 | 3 |
| s2|selection | 120 | 82 | 12 | 26 | 0.1417 | 541 | 5 |
| s0|calibration | 120 | 89 | 7 | 24 | 0.1583 | 472 | 0 |
| s1|calibration | 120 | 90 | 4 | 26 | 0.1833 | 472 | 0 |
| s2|calibration | 120 | 81 | 11 | 28 | 0.1667 | 472 | 1 |

## Goal

| group | selection mean | calibration mean | constraint failures | met |
|---|---|---|---|---|
| FACT | 0.9557 | 0.9605 | 4 | False |
| A01 ASRF | 0.9449 | 0.9747 | 5 | False |
| A01 Linear + B3 | 0.9476 | 0.9760 | 0 | False |
| frozen Ordered + B3 | 0.9544 | 0.9817 | 1 | False |

## Readings

```
[
 "one_half_or_local_effect",
 "trade_off",
 "uncertain_or_no_qualifying_gain",
 "end"
]
```
