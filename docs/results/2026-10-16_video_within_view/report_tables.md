# A08 within-view regularisation round: tables

Recompute: `python scripts/report_video_within_view.py --config configs/experiments/video_within_view_v1.yaml --selection --choices-sha256 <sha> --choices-commit <commit>`

## Systems

| system | half | segment F1@50 | boundary P | boundary R | boundary F1 | native macro-F1 | switches per clip | same-view join cuts | pairs F1@50 | A4C read as A5C | A5C read as A4C |
|---|---|---|---|---|---|---|---|---|---|---|---|
| frozen Ordered + B3 s0 | selection | 0.9385 | 0.9482 | 0.9581 | 0.9531 | 0.9786 | 0.0423 | 0.0174 | 0.9615 | 2 | 0 |
| frozen Ordered + B3 s0 | calibration | 0.9817 | 0.9751 | 0.9824 | 0.9787 | 0.9929 | 0.0253 | 0.0249 | 0.9825 | 1 | 0 |
| frozen Ordered argmax s0 | selection | 0.8929 | 0.8462 | 0.9791 | 0.9078 | 0.9701 | 0.0951 | 0.0435 | 0.9619 | 3 | 0 |
| frozen Ordered argmax s0 | calibration | 0.9611 | 0.9424 | 0.9874 | 0.9644 | 0.9861 | 0.0928 | 0.0348 | 0.9741 | 2 | 0 |
| F + B3 s0 | selection | 0.9531 | 0.9328 | 0.9817 | 0.9566 | 0.9786 | 0.0563 | 0.0217 | 0.9569 | 2 | 0 |
| F + B3 s0 | calibration | 0.9779 | 0.9752 | 0.9874 | 0.9813 | 0.9929 | 0.0211 | 0.0199 | 0.9797 | 1 | 0 |
| F argmax s0 | selection | 0.9132 | 0.8584 | 0.9843 | 0.9171 | 0.9877 | 0.0986 | 0.0348 | 0.9456 | 1 | 0 |
| F argmax s0 | calibration | 0.9602 | 0.9471 | 0.9899 | 0.9681 | 0.9929 | 0.1224 | 0.0398 | 0.9657 | 1 | 0 |
| F at Ordered setting s0 | selection | 0.9503 | 0.9259 | 0.9817 | 0.9530 | 0.9786 | 0.0634 | 0.0217 | 0.9569 | 2 | 0 |
| F at Ordered setting s0 | calibration | 0.9761 | 0.9681 | 0.9899 | 0.9789 | 0.9929 | 0.0211 | 0.0348 | 0.9797 | 1 | 0 |
| A + B3 s0 | selection | 0.9563 | 0.9560 | 0.9660 | 0.9609 | 0.9786 | 0.0387 | 0.0217 | 0.9688 | 2 | 0 |
| A + B3 s0 | calibration | 0.9827 | 0.9801 | 0.9899 | 0.9850 | 0.9929 | 0.0127 | 0.0249 | 0.9797 | 1 | 0 |
| A argmax s0 | selection | 0.9014 | 0.8465 | 0.9817 | 0.9091 | 0.9877 | 0.1056 | 0.0391 | 0.9526 | 1 | 0 |
| A argmax s0 | calibration | 0.9668 | 0.9518 | 0.9925 | 0.9717 | 0.9929 | 0.0886 | 0.0348 | 0.9770 | 1 | 0 |
| A at Ordered setting s0 | selection | 0.9536 | 0.9757 | 0.9450 | 0.9601 | 0.9786 | 0.0246 | 0.0130 | 0.9639 | 2 | 0 |
| A at Ordered setting s0 | calibration | 0.9807 | 0.9848 | 0.9799 | 0.9824 | 0.9929 | 0.0127 | 0.0199 | 0.9736 | 1 | 0 |
| M + B3 s0 | selection | 0.9598 | 0.9471 | 0.9843 | 0.9653 | 0.9786 | 0.0352 | 0.0174 | 0.9712 | 2 | 0 |
| M + B3 s0 | calibration | 0.9770 | 0.9728 | 0.9874 | 0.9800 | 0.9929 | 0.0169 | 0.0299 | 0.9797 | 1 | 0 |
| M argmax s0 | selection | 0.9087 | 0.8571 | 0.9895 | 0.9186 | 0.9786 | 0.1162 | 0.0348 | 0.9506 | 2 | 0 |
| M argmax s0 | calibration | 0.9556 | 0.9335 | 0.9874 | 0.9597 | 0.9929 | 0.1097 | 0.0398 | 0.9630 | 1 | 0 |
| M at Ordered setting s0 | selection | 0.9654 | 0.9640 | 0.9817 | 0.9728 | 0.9786 | 0.0282 | 0.0130 | 0.9687 | 2 | 0 |
| M at Ordered setting s0 | calibration | 0.9779 | 0.9776 | 0.9849 | 0.9812 | 0.9929 | 0.0169 | 0.0199 | 0.9797 | 1 | 0 |
| frozen Ordered + B3 s1 | selection | 0.9603 | 0.9585 | 0.9686 | 0.9635 | 0.9786 | 0.0387 | 0.0043 | 0.9592 | 2 | 0 |
| frozen Ordered + B3 s1 | calibration | 0.9836 | 0.9800 | 0.9874 | 0.9837 | 0.9861 | 0.0169 | 0.0249 | 0.9709 | 2 | 0 |
| frozen Ordered argmax s1 | selection | 0.9250 | 0.8724 | 0.9843 | 0.9250 | 0.9877 | 0.1127 | 0.0304 | 0.9484 | 1 | 0 |
| frozen Ordered argmax s1 | calibration | 0.9611 | 0.9448 | 0.9899 | 0.9669 | 0.9796 | 0.1013 | 0.0398 | 0.9543 | 3 | 0 |
| F + B3 s1 | selection | 0.9508 | 0.9442 | 0.9738 | 0.9588 | 0.9786 | 0.0352 | 0.0130 | 0.9642 | 2 | 0 |
| F + B3 s1 | calibration | 0.9885 | 0.9754 | 0.9950 | 0.9851 | 0.9929 | 0.0169 | 0.0149 | 0.9884 | 1 | 0 |
| F argmax s1 | selection | 0.9005 | 0.8366 | 0.9921 | 0.9078 | 0.9877 | 0.0986 | 0.0478 | 0.9459 | 1 | 0 |
| F argmax s1 | calibration | 0.9752 | 0.9612 | 0.9950 | 0.9778 | 0.9929 | 0.0759 | 0.0199 | 0.9857 | 1 | 0 |
| F at Ordered setting s1 | selection | 0.9482 | 0.9327 | 0.9791 | 0.9553 | 0.9877 | 0.0528 | 0.0174 | 0.9642 | 1 | 0 |
| F at Ordered setting s1 | calibration | 0.9875 | 0.9778 | 0.9950 | 0.9863 | 0.9929 | 0.0169 | 0.0149 | 0.9856 | 1 | 0 |
| A + B3 s1 | selection | 0.9545 | 0.9510 | 0.9660 | 0.9584 | 0.9877 | 0.0423 | 0.0174 | 0.9617 | 1 | 0 |
| A + B3 s1 | calibration | 0.9885 | 0.9802 | 0.9950 | 0.9875 | 0.9856 | 0.0169 | 0.0199 | 0.9826 | 1 | 1 |
| A argmax s1 | selection | 0.9077 | 0.8545 | 0.9843 | 0.9148 | 0.9877 | 0.0951 | 0.0391 | 0.9549 | 1 | 0 |
| A argmax s1 | calibration | 0.9771 | 0.9612 | 0.9950 | 0.9778 | 0.9929 | 0.0675 | 0.0249 | 0.9828 | 1 | 0 |
| A at Ordered setting s1 | selection | 0.9508 | 0.9759 | 0.9529 | 0.9642 | 0.9877 | 0.0176 | 0.0087 | 0.9687 | 1 | 0 |
| A at Ordered setting s1 | calibration | 0.9875 | 0.9825 | 0.9849 | 0.9837 | 0.9929 | 0.0084 | 0.0199 | 0.9766 | 1 | 0 |
| M + B3 s1 | selection | 0.9537 | 0.9491 | 0.9764 | 0.9626 | 0.9877 | 0.0317 | 0.0174 | 0.9712 | 1 | 0 |
| M + B3 s1 | calibration | 0.9885 | 0.9754 | 0.9950 | 0.9851 | 0.9929 | 0.0127 | 0.0149 | 0.9855 | 1 | 0 |
| M argmax s1 | selection | 0.8947 | 0.8404 | 0.9921 | 0.9100 | 0.9877 | 0.1056 | 0.0435 | 0.9504 | 1 | 0 |
| M argmax s1 | calibration | 0.9781 | 0.9588 | 0.9950 | 0.9766 | 0.9929 | 0.0928 | 0.0199 | 0.9771 | 1 | 0 |
| M at Ordered setting s1 | selection | 0.9590 | 0.9763 | 0.9686 | 0.9724 | 0.9877 | 0.0176 | 0.0087 | 0.9712 | 1 | 0 |
| M at Ordered setting s1 | calibration | 0.9875 | 0.9826 | 0.9950 | 0.9888 | 0.9929 | 0.0127 | 0.0100 | 0.9767 | 1 | 0 |
| frozen Ordered + B3 s2 | selection | 0.9644 | 0.9588 | 0.9738 | 0.9662 | 0.9701 | 0.0458 | 0.0043 | 0.9569 | 3 | 0 |
| frozen Ordered + B3 s2 | calibration | 0.9797 | 0.9774 | 0.9799 | 0.9787 | 0.9861 | 0.0127 | 0.0249 | 0.9708 | 2 | 0 |
| frozen Ordered argmax s2 | selection | 0.9259 | 0.8744 | 0.9843 | 0.9261 | 0.9701 | 0.1162 | 0.0261 | 0.9596 | 3 | 0 |
| frozen Ordered argmax s2 | calibration | 0.9602 | 0.9399 | 0.9824 | 0.9607 | 0.9861 | 0.0970 | 0.0348 | 0.9543 | 2 | 0 |
| F + B3 s2 | selection | 0.9174 | 0.8747 | 0.9869 | 0.9274 | 0.9786 | 0.0880 | 0.0304 | 0.9551 | 2 | 0 |
| F + B3 s2 | calibration | 0.9771 | 0.9634 | 0.9925 | 0.9777 | 0.9861 | 0.0633 | 0.0348 | 0.9797 | 2 | 0 |
| F argmax s2 | selection | 0.9035 | 0.8494 | 0.9895 | 0.9141 | 0.9877 | 0.1303 | 0.0391 | 0.9439 | 1 | 0 |
| F argmax s2 | calibration | 0.9706 | 0.9542 | 0.9950 | 0.9742 | 0.9929 | 0.1139 | 0.0448 | 0.9573 | 1 | 0 |
| F at Ordered setting s2 | selection | 0.9407 | 0.9214 | 0.9817 | 0.9506 | 0.9786 | 0.0634 | 0.0130 | 0.9642 | 2 | 0 |
| F at Ordered setting s2 | calibration | 0.9799 | 0.9705 | 0.9925 | 0.9814 | 0.9788 | 0.0169 | 0.0299 | 0.9797 | 2 | 1 |
| A + B3 s2 | selection | 0.9564 | 0.9536 | 0.9686 | 0.9610 | 0.9786 | 0.0423 | 0.0130 | 0.9808 | 2 | 0 |
| A + B3 s2 | calibration | 0.9837 | 0.9753 | 0.9925 | 0.9838 | 0.9856 | 0.0169 | 0.0299 | 0.9709 | 1 | 1 |
| A argmax s2 | selection | 0.9004 | 0.8529 | 0.9869 | 0.9150 | 0.9877 | 0.1162 | 0.0348 | 0.9528 | 1 | 0 |
| A argmax s2 | calibration | 0.9771 | 0.9634 | 0.9925 | 0.9777 | 0.9929 | 0.0675 | 0.0348 | 0.9742 | 1 | 0 |
| A at Ordered setting s2 | selection | 0.9629 | 0.9787 | 0.9607 | 0.9696 | 0.9786 | 0.0282 | 0.0087 | 0.9758 | 2 | 0 |
| A at Ordered setting s2 | calibration | 0.9826 | 0.9849 | 0.9849 | 0.9849 | 0.9856 | 0.0127 | 0.0149 | 0.9677 | 1 | 1 |
| M + B3 s2 | selection | 0.9644 | 0.9639 | 0.9791 | 0.9714 | 0.9786 | 0.0282 | 0.0130 | 0.9663 | 2 | 0 |
| M + B3 s2 | calibration | 0.9749 | 0.9799 | 0.9799 | 0.9799 | 0.9861 | 0.0211 | 0.0199 | 0.9797 | 2 | 0 |
| M argmax s2 | selection | 0.9070 | 0.8555 | 0.9921 | 0.9188 | 0.9877 | 0.1162 | 0.0391 | 0.9528 | 1 | 0 |
| M argmax s2 | calibration | 0.9686 | 0.9564 | 0.9925 | 0.9741 | 0.9929 | 0.1097 | 0.0398 | 0.9770 | 1 | 0 |
| M at Ordered setting s2 | selection | 0.9660 | 0.9788 | 0.9686 | 0.9737 | 0.9786 | 0.0246 | 0.0087 | 0.9783 | 2 | 0 |
| M at Ordered setting s2 | calibration | 0.9788 | 0.9824 | 0.9824 | 0.9824 | 0.9788 | 0.0127 | 0.0149 | 0.9738 | 2 | 1 |

## Paired comparisons (seed / decoding / comparison / half; 95% bootstrap intervals)

| comparison | multi segment F1@50 | native macro-F1 | multi boundary recall | multi FP segments | multi FN segments | native cuts per clip |
|---|---|---|---|---|---|---|
| s0 / b3 / M - A / selection | +0.0034 [-0.0054, +0.0147] | +0.0000 [+0.0000, +0.0000] | +0.0183 [+0.0052, +0.0345] | +4.0000 [-1.0000, +9.0000] | -7.0000 [-15.0000, -1.0000] | -0.0035 [-0.0096, +0.0000] |
| s0 / b3 / M - F / selection | +0.0066 [-0.0010, +0.0173] | +0.0000 [+0.0000, +0.0000] | +0.0026 [+0.0000, +0.0083] | -6.0000 [-15.0000, +1.0000] | -1.0000 [-3.0000, +0.0000] | -0.0211 [-0.0428, +0.0000] |
| s0 / b3 / A - F / selection | +0.0032 [-0.0104, +0.0166] | +0.0000 [+0.0000, +0.0000] | -0.0157 [-0.0310, -0.0027] | -10.0000 [-20.0000, -2.0000] | +6.0000 [+0.0000, +14.0000] | -0.0176 [-0.0394, +0.0000] |
| s0 / b3 / M - frozen Ordered / selection | +0.0213 [+0.0081, +0.0370] | +0.0000 [+0.0000, +0.0000] | +0.0262 [+0.0104, +0.0442] | -5.0000 [-14.0000, +3.0000] | -16.0000 [-26.0000, -8.0000] | -0.0070 [-0.0227, +0.0000] |
| s0 / b3 / A - frozen Ordered / selection | +0.0179 [+0.0064, +0.0318] | +0.0000 [+0.0000, +0.0000] | +0.0079 [-0.0027, +0.0201] | -9.0000 [-18.0000, -2.0000] | -9.0000 [-16.0000, -3.0000] | -0.0035 [-0.0215, +0.0081] |
| s0 / argmax / M - A / selection | +0.0073 [-0.0073, +0.0230] | -0.0091 [-0.0288, +0.0000] | +0.0079 [+0.0000, +0.0182] | -5.0000 [-17.0000, +7.0000] | -3.0000 [-9.0000, +3.0000] | +0.0106 [-0.0077, +0.0341] |
| s0 / argmax / M - F / selection | -0.0045 [-0.0123, +0.0028] | -0.0091 [-0.0288, +0.0000] | +0.0052 [+0.0000, +0.0134] | +4.0000 [-3.0000, +11.0000] | +1.0000 [-2.0000, +5.0000] | +0.0176 [-0.0242, +0.0514] |
| s0 / argmax / A - F / selection | -0.0118 [-0.0282, +0.0027] | +0.0000 [+0.0000, +0.0000] | -0.0026 [-0.0113, +0.0054] | +9.0000 [-2.0000, +22.0000] | +4.0000 [-2.0000, +10.0000] | +0.0070 [-0.0213, +0.0353] |
| s0 / argmax / M - frozen Ordered / selection | +0.0159 [+0.0033, +0.0297] | +0.0085 [+0.0000, +0.0279] | +0.0105 [+0.0025, +0.0211] | -9.0000 [-21.0000, +2.0000] | -8.0000 [-14.0000, -3.0000] | +0.0211 [-0.0105, +0.0489] |
| s0 / argmax / A - frozen Ordered / selection | +0.0086 [-0.0041, +0.0223] | +0.0177 [+0.0000, +0.0395] | +0.0026 [-0.0056, +0.0123] | -4.0000 [-16.0000, +7.0000] | -5.0000 [-11.0000, +0.0000] | +0.0106 [-0.0268, +0.0423] |
| s0 / ordered_setting / M - A / selection | +0.0118 [+0.0004, +0.0249] | +0.0000 [+0.0000, +0.0000] | +0.0366 [+0.0176, +0.0575] | +4.0000 [-1.0000, +10.0000] | -15.0000 [-25.0000, -6.0000] | +0.0035 [-0.0069, +0.0190] |
| s0 / ordered_setting / M - F / selection | +0.0150 [+0.0038, +0.0293] | +0.0000 [+0.0000, +0.0000] | +0.0000 [-0.0078, +0.0078] | -16.0000 [-30.0000, -5.9750] | +0.0000 [-3.0000, +3.0000] | -0.0352 [-0.0735, -0.0073] |
| s0 / ordered_setting / A - F / selection | +0.0033 [-0.0135, +0.0206] | +0.0000 [+0.0000, +0.0000] | -0.0366 [-0.0574, -0.0176] | -20.0000 [-34.0000, -9.0000] | +15.0000 [+6.0000, +25.0000] | -0.0387 [-0.0769, -0.0085] |
| s0 / ordered_setting / M - frozen Ordered / selection | +0.0269 [+0.0129, +0.0441] | +0.0000 [+0.0000, +0.0000] | +0.0236 [+0.0082, +0.0408] | -12.0000 [-23.0000, -3.0000] | -15.0000 [-25.0000, -7.0000] | -0.0141 [-0.0369, +0.0000] |
| s0 / ordered_setting / A - frozen Ordered / selection | +0.0151 [+0.0044, +0.0281] | +0.0000 [+0.0000, +0.0000] | -0.0131 [-0.0315, +0.0027] | -16.0000 [-26.0000, -7.0000] | +0.0000 [-6.0000, +6.0000] | -0.0176 [-0.0458, +0.0030] |
| s1 / b3 / M - A / selection | -0.0008 [-0.0078, +0.0063] | +0.0000 [+0.0000, +0.0000] | +0.0105 [+0.0025, +0.0225] | +3.0000 [-4.0000, +10.0000] | -2.0000 [-5.0000, +0.0000] | -0.0106 [-0.0317, +0.0083] |
| s1 / b3 / M - F / selection | +0.0029 [-0.0068, +0.0144] | +0.0091 [+0.0000, +0.0355] | +0.0026 [+0.0000, +0.0084] | -2.0000 [-11.0000, +5.0000] | -1.0000 [-5.0000, +2.0000] | -0.0035 [-0.0123, +0.0000] |
| s1 / b3 / A - F / selection | +0.0037 [-0.0091, +0.0171] | +0.0091 [+0.0000, +0.0355] | -0.0079 [-0.0181, +0.0000] | -5.0000 [-17.0000, +6.0000] | +1.0000 [-2.0000, +4.0000] | +0.0070 [-0.0149, +0.0290] |
| s1 / b3 / M - frozen Ordered / selection | -0.0066 [-0.0196, +0.0036] | +0.0091 [+0.0000, +0.0355] | +0.0079 [-0.0054, +0.0213] | +7.0000 [+1.0000, +15.0000] | +0.0000 [-6.0000, +7.0000] | -0.0070 [-0.0253, +0.0076] |
| s1 / b3 / A - frozen Ordered / selection | -0.0059 [-0.0203, +0.0054] | +0.0091 [+0.0000, +0.0355] | -0.0026 [-0.0186, +0.0131] | +4.0000 [-4.0000, +14.0000] | +2.0000 [-4.0000, +9.0000] | +0.0035 [-0.0243, +0.0295] |
| s1 / argmax / M - A / selection | -0.0130 [-0.0236, -0.0033] | +0.0000 [+0.0000, +0.0000] | +0.0079 [+0.0000, +0.0178] | +13.0000 [+5.0000, +22.0000] | +2.0000 [-3.0000, +7.0000] | +0.0106 [+0.0000, +0.0237] |
| s1 / argmax / M - F / selection | -0.0058 [-0.0184, +0.0057] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +2.0000 [-10.0000, +14.0000] | +4.0000 [+1.0000, +8.0000] | +0.0070 [+0.0000, +0.0270] |
| s1 / argmax / A - F / selection | +0.0073 [-0.0084, +0.0225] | +0.0000 [+0.0000, +0.0000] | -0.0079 [-0.0178, +0.0000] | -11.0000 [-26.0000, +4.0000] | +2.0000 [-3.0000, +7.0000] | -0.0035 [-0.0178, +0.0122] |
| s1 / argmax / M - frozen Ordered / selection | -0.0303 [-0.0540, -0.0095] | +0.0000 [+0.0000, +0.0000] | +0.0079 [-0.0027, +0.0193] | +27.0000 [+8.0000, +48.0000] | +7.0000 [+1.0000, +14.0000] | -0.0070 [-0.0299, +0.0163] |
| s1 / argmax / A - frozen Ordered / selection | -0.0173 [-0.0390, +0.0023] | +0.0000 [+0.0000, +0.0000] | +0.0000 [-0.0107, +0.0104] | +14.0000 [-3.0000, +33.0000] | +5.0000 [-1.0000, +12.0000] | -0.0176 [-0.0421, +0.0043] |
| s1 / ordered_setting / M - A / selection | +0.0083 [+0.0001, +0.0195] | +0.0000 [+0.0000, +0.0000] | +0.0157 [+0.0026, +0.0359] | -1.0000 [-5.0000, +2.0000] | -7.0000 [-15.0000, -1.0000] | +0.0000 [+0.0000, +0.0000] |
| s1 / ordered_setting / M - F / selection | +0.0108 [-0.0023, +0.0258] | +0.0000 [-0.0272, +0.0313] | -0.0105 [-0.0212, -0.0025] | -17.0000 [-30.0000, -7.0000] | +5.0000 [+0.0000, +11.0000] | -0.0352 [-0.0766, -0.0076] |
| s1 / ordered_setting / A - F / selection | +0.0026 [-0.0122, +0.0173] | +0.0000 [-0.0272, +0.0313] | -0.0262 [-0.0479, -0.0097] | -16.0000 [-28.0000, -6.0000] | +12.0000 [+4.0000, +22.0000] | -0.0352 [-0.0766, -0.0076] |
| s1 / ordered_setting / M - frozen Ordered / selection | -0.0013 [-0.0145, +0.0093] | +0.0091 [+0.0000, +0.0355] | +0.0000 [-0.0158, +0.0161] | -3.0000 [-10.0000, +5.0000] | +4.0000 [-3.0000, +11.0000] | -0.0211 [-0.0441, -0.0030] |
| s1 / ordered_setting / A - frozen Ordered / selection | -0.0096 [-0.0253, +0.0036] | +0.0091 [+0.0000, +0.0355] | -0.0157 [-0.0380, +0.0050] | -2.0000 [-9.0000, +7.0000] | +11.0000 [+2.0000, +21.0000] | -0.0211 [-0.0441, -0.0030] |
| s2 / b3 / M - A / selection | +0.0079 [-0.0037, +0.0209] | +0.0000 [+0.0000, +0.0000] | +0.0105 [-0.0052, +0.0262] | -4.0000 [-11.0000, +3.0000] | -4.0000 [-12.0000, +3.0000] | -0.0141 [-0.0309, +0.0000] |
| s2 / b3 / M - F / selection | +0.0470 [+0.0251, +0.0710] | +0.0000 [+0.0000, +0.0000] | -0.0079 [-0.0202, +0.0027] | -47.0000 [-69.0000, -28.0000] | -4.0000 [-11.0000, +2.0000] | -0.0599 [-0.1013, -0.0193] |
| s2 / b3 / A - F / selection | +0.0391 [+0.0180, +0.0632] | +0.0000 [+0.0000, +0.0000] | -0.0183 [-0.0345, -0.0053] | -43.0000 [-64.0000, -25.0000] | +0.0000 [-8.0000, +9.0000] | -0.0458 [-0.0893, -0.0082] |
| s2 / b3 / M - frozen Ordered / selection | +0.0000 [-0.0134, +0.0146] | +0.0085 [+0.0000, +0.0241] | +0.0052 [-0.0118, +0.0244] | +0.0000 [-8.0000, +8.0000] | +0.0000 [-9.0000, +8.0000] | -0.0176 [-0.0398, +0.0000] |
| s2 / b3 / A - frozen Ordered / selection | -0.0079 [-0.0224, +0.0042] | +0.0085 [+0.0000, +0.0241] | -0.0052 [-0.0181, +0.0079] | +4.0000 [-4.0000, +13.0000] | +4.0000 [-2.0000, +11.0000] | -0.0035 [-0.0245, +0.0123] |
| s2 / argmax / M - A / selection | +0.0067 [-0.0008, +0.0158] | +0.0000 [+0.0000, +0.0000] | +0.0052 [+0.0000, +0.0132] | -3.0000 [-9.0000, +2.0000] | -4.0000 [-8.0000, -1.0000] | +0.0000 [-0.0084, +0.0111] |
| s2 / argmax / M - F / selection | +0.0036 [-0.0009, +0.0103] | +0.0000 [+0.0000, +0.0000] | +0.0026 [+0.0000, +0.0084] | -3.0000 [-9.0000, +1.0000] | -1.0000 [-3.0000, +0.0000] | -0.0141 [-0.0646, +0.0176] |
| s2 / argmax / A - F / selection | -0.0031 [-0.0117, +0.0033] | +0.0000 [+0.0000, +0.0000] | -0.0026 [-0.0080, +0.0000] | +0.0000 [-5.0000, +6.0000] | +3.0000 [+0.0000, +7.0000] | -0.0141 [-0.0651, +0.0203] |
| s2 / argmax / M - frozen Ordered / selection | -0.0188 [-0.0349, -0.0044] | +0.0177 [+0.0000, +0.0380] | +0.0079 [-0.0028, +0.0192] | +17.0000 [+6.0000, +29.0000] | +4.0000 [-2.0000, +11.0000] | +0.0000 [-0.0287, +0.0259] |
| s2 / argmax / A - frozen Ordered / selection | -0.0255 [-0.0425, -0.0109] | +0.0177 [+0.0000, +0.0380] | +0.0026 [-0.0055, +0.0115] | +20.0000 [+8.0000, +33.0000] | +8.0000 [+2.0000, +15.0000] | +0.0000 [-0.0333, +0.0309] |
| s2 / ordered_setting / M - A / selection | +0.0031 [-0.0019, +0.0083] | +0.0000 [+0.0000, +0.0000] | +0.0079 [-0.0027, +0.0197] | +0.0000 [-4.0000, +4.0000] | -3.0000 [-7.0000, +0.0000] | -0.0035 [-0.0135, +0.0000] |
| s2 / ordered_setting / M - F / selection | +0.0253 [+0.0083, +0.0448] | +0.0000 [+0.0000, +0.0000] | -0.0131 [-0.0311, +0.0027] | -28.0000 [-43.0000, -15.0000] | +1.0000 [-7.0000, +9.0000] | -0.0387 [-0.0634, -0.0115] |
| s2 / ordered_setting / A - F / selection | +0.0222 [+0.0037, +0.0428] | +0.0000 [+0.0000, +0.0000] | -0.0209 [-0.0360, -0.0079] | -28.0000 [-44.0000, -15.0000] | +4.0000 [-4.0000, +12.0000] | -0.0352 [-0.0597, -0.0088] |
| s2 / ordered_setting / M - frozen Ordered / selection | +0.0016 [-0.0104, +0.0142] | +0.0085 [+0.0000, +0.0241] | -0.0052 [-0.0194, +0.0080] | -6.0000 [-14.0000, +2.0000] | +4.0000 [-2.0000, +10.0000] | -0.0211 [-0.0412, -0.0038] |
| s2 / ordered_setting / A - frozen Ordered / selection | -0.0015 [-0.0146, +0.0106] | +0.0085 [+0.0000, +0.0241] | -0.0131 [-0.0285, +0.0026] | -6.0000 [-14.0000, +2.0000] | +7.0000 [+0.0000, +15.0000] | -0.0176 [-0.0381, +0.0037] |
| s0 / b3 / M - A / calibration | -0.0057 [-0.0118, -0.0010] | +0.0000 [+0.0000, +0.0000] | -0.0025 [-0.0078, +0.0000] | +4.0000 [+1.0000, +8.0000] | +2.0000 [+0.0000, +5.0000] | +0.0042 [+0.0000, +0.0139] |
| s0 / b3 / M - F / calibration | -0.0009 [-0.0041, +0.0019] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +1.0000 [-2.0000, +4.0250] | +0.0000 [+0.0000, +0.0000] | -0.0042 [-0.0146, +0.0000] |
| s0 / b3 / A - F / calibration | +0.0048 [+0.0009, +0.0105] | +0.0000 [+0.0000, +0.0000] | +0.0025 [+0.0000, +0.0078] | -3.0000 [-8.0000, +1.0000] | -2.0000 [-5.0000, +0.0000] | -0.0084 [-0.0228, +0.0000] |
| s0 / b3 / M - frozen Ordered / calibration | -0.0047 [-0.0128, +0.0021] | +0.0000 [+0.0000, +0.0000] | +0.0050 [-0.0027, +0.0156] | +4.0000 [+1.0000, +8.0000] | +1.0000 [-4.0000, +6.0000] | -0.0084 [-0.0275, +0.0000] |
| s0 / b3 / A - frozen Ordered / calibration | +0.0010 [-0.0053, +0.0070] | +0.0000 [+0.0000, +0.0000] | +0.0075 [+0.0000, +0.0172] | +0.0000 [-4.0000, +4.0000] | -1.0000 [-6.0000, +3.0000] | -0.0127 [-0.0320, +0.0000] |
| s0 / argmax / M - A / calibration | -0.0111 [-0.0237, -0.0011] | +0.0000 [+0.0000, +0.0000] | -0.0050 [-0.0127, +0.0000] | +9.0000 [+0.0000, +20.0000] | +3.0000 [+0.0000, +7.0000] | +0.0211 [+0.0000, +0.0489] |
| s0 / argmax / M - F / calibration | -0.0045 [-0.0155, +0.0039] | +0.0000 [+0.0000, +0.0000] | -0.0025 [-0.0078, +0.0000] | +5.0000 [-2.0000, +15.0000] | +0.0000 [-3.0000, +3.0000] | -0.0127 [-0.0502, +0.0186] |
| s0 / argmax / A - F / calibration | +0.0066 [-0.0018, +0.0156] | +0.0000 [+0.0000, +0.0000] | +0.0025 [+0.0000, +0.0078] | -4.0000 [-12.0000, +4.0000] | -3.0000 [-7.0000, +0.0000] | -0.0338 [-0.0721, -0.0044] |
| s0 / argmax / M - frozen Ordered / calibration | -0.0055 [-0.0189, +0.0066] | +0.0068 [+0.0000, +0.0226] | +0.0000 [-0.0073, +0.0076] | +5.0000 [-5.0000, +16.0000] | +1.0000 [-3.0000, +6.0000] | +0.0169 [-0.0152, +0.0485] |
| s0 / argmax / A - frozen Ordered / calibration | +0.0056 [-0.0072, +0.0169] | +0.0068 [+0.0000, +0.0226] | +0.0050 [+0.0000, +0.0128] | -4.0000 [-13.0000, +7.0000] | -2.0000 [-6.0000, +2.0000] | -0.0042 [-0.0281, +0.0103] |
| s0 / ordered_setting / M - A / calibration | -0.0028 [-0.0088, +0.0021] | +0.0000 [+0.0000, +0.0000] | +0.0050 [+0.0000, +0.0129] | +4.0000 [+1.0000, +8.0000] | -1.0000 [-5.0000, +2.0000] | +0.0042 [+0.0000, +0.0139] |
| s0 / ordered_setting / M - F / calibration | +0.0018 [-0.0019, +0.0056] | +0.0000 [+0.0000, +0.0000] | -0.0050 [-0.0128, +0.0000] | -4.0000 [-9.0000, +0.0000] | +2.0000 [+0.0000, +5.0000] | -0.0042 [-0.0146, +0.0000] |
| s0 / ordered_setting / A - F / calibration | +0.0046 [-0.0020, +0.0115] | +0.0000 [+0.0000, +0.0000] | -0.0101 [-0.0204, -0.0024] | -8.0000 [-15.0000, -3.0000] | +3.0000 [-1.0000, +7.0000] | -0.0084 [-0.0228, +0.0000] |
| s0 / ordered_setting / M - frozen Ordered / calibration | -0.0038 [-0.0118, +0.0030] | +0.0000 [+0.0000, +0.0000] | +0.0025 [-0.0051, +0.0124] | +2.0000 [-3.0000, +7.0000] | +2.0000 [-3.0000, +7.0000] | -0.0084 [-0.0275, +0.0000] |
| s0 / ordered_setting / A - frozen Ordered / calibration | -0.0011 [-0.0069, +0.0047] | +0.0000 [+0.0000, +0.0000] | -0.0025 [-0.0109, +0.0052] | -2.0000 [-6.0000, +2.0000] | +3.0000 [-1.0000, +7.0000] | -0.0127 [-0.0320, +0.0000] |
| s1 / b3 / M - A / calibration | +0.0000 [-0.0028, +0.0029] | +0.0073 [+0.0000, +0.0268] | +0.0000 [+0.0000, +0.0000] | +1.0000 [+0.0000, +3.0000] | -1.0000 [-3.0000, +0.0000] | -0.0042 [-0.0138, +0.0000] |
| s1 / b3 / M - F / calibration | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0042 [-0.0146, +0.0000] |
| s1 / b3 / A - F / calibration | -0.0000 [-0.0029, +0.0028] | -0.0073 [-0.0268, +0.0000] | +0.0000 [+0.0000, +0.0000] | -1.0000 [-3.0000, +0.0000] | +1.0000 [+0.0000, +3.0000] | +0.0000 [-0.0133, +0.0115] |
| s1 / b3 / M - frozen Ordered / calibration | +0.0049 [-0.0028, +0.0130] | +0.0068 [+0.0000, +0.0191] | +0.0075 [+0.0000, +0.0174] | +0.0000 [-6.0000, +5.0000] | -5.0000 [-10.0000, -1.0000] | -0.0042 [-0.0146, +0.0000] |
| s1 / b3 / A - frozen Ordered / calibration | +0.0048 [-0.0037, +0.0137] | -0.0005 [-0.0252, +0.0171] | +0.0075 [+0.0000, +0.0174] | -1.0000 [-7.0000, +5.0000] | -4.0000 [-9.0000, +0.0000] | +0.0000 [-0.0129, +0.0114] |
| s1 / argmax / M - A / calibration | +0.0010 [-0.0027, +0.0057] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [-3.0000, +3.0000] | -1.0000 [-3.0000, +0.0000] | +0.0253 [+0.0000, +0.0826] |
| s1 / argmax / M - F / calibration | +0.0029 [-0.0035, +0.0110] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -1.0000 [-7.0000, +4.0000] | -2.0000 [-5.0000, +0.0000] | +0.0169 [+0.0000, +0.0550] |
| s1 / argmax / A - F / calibration | +0.0019 [-0.0057, +0.0110] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -1.0000 [-7.0000, +4.0000] | -1.0000 [-5.0000, +2.0000] | -0.0084 [-0.0275, +0.0000] |
| s1 / argmax / M - frozen Ordered / calibration | +0.0170 [+0.0056, +0.0318] | +0.0134 [+0.0000, +0.0293] | +0.0050 [+0.0000, +0.0129] | -11.0000 [-23.0000, -2.0000] | -7.0000 [-12.0000, -3.0000] | -0.0084 [-0.0451, +0.0219] |
| s1 / argmax / A - frozen Ordered / calibration | +0.0160 [+0.0048, +0.0306] | +0.0134 [+0.0000, +0.0293] | +0.0050 [+0.0000, +0.0129] | -11.0000 [-23.0000, -2.0000] | -6.0000 [-11.0000, -2.0000] | -0.0338 [-0.1000, +0.0167] |
| s1 / ordered_setting / M - A / calibration | +0.0000 [-0.0048, +0.0048] | +0.0000 [+0.0000, +0.0000] | +0.0101 [+0.0024, +0.0206] | +2.0000 [+0.0000, +5.0000] | -2.0000 [-6.0000, +2.0000] | +0.0042 [+0.0000, +0.0139] |
| s1 / ordered_setting / M - F / calibration | -0.0000 [-0.0029, +0.0029] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -1.0000 [-3.0000, +0.0000] | +1.0000 [+0.0000, +3.0000] | -0.0042 [-0.0146, +0.0000] |
| s1 / ordered_setting / A - F / calibration | -0.0001 [-0.0049, +0.0046] | +0.0000 [+0.0000, +0.0000] | -0.0101 [-0.0206, -0.0024] | -3.0000 [-7.0000, +0.0000] | +3.0000 [+0.0000, +7.0000] | -0.0084 [-0.0228, +0.0000] |
| s1 / ordered_setting / M - frozen Ordered / calibration | +0.0039 [-0.0048, +0.0137] | +0.0068 [+0.0000, +0.0191] | +0.0075 [+0.0000, +0.0174] | -1.0000 [-7.0000, +5.0000] | -3.0000 [-8.0000, +2.0000] | -0.0042 [-0.0146, +0.0000] |
| s1 / ordered_setting / A - frozen Ordered / calibration | +0.0038 [-0.0049, +0.0134] | +0.0068 [+0.0000, +0.0191] | -0.0025 [-0.0133, +0.0081] | -3.0000 [-9.0000, +2.0000] | -1.0000 [-6.0000, +4.0000] | -0.0084 [-0.0216, +0.0000] |
| s2 / b3 / M - A / calibration | -0.0088 [-0.0185, -0.0019] | +0.0005 [-0.0171, +0.0252] | -0.0126 [-0.0270, -0.0024] | +1.0000 [-2.0000, +5.0000] | +8.0000 [+2.0000, +16.0000] | +0.0042 [+0.0000, +0.0138] |
| s2 / b3 / M - F / calibration | -0.0022 [-0.0122, +0.0061] | +0.0000 [+0.0000, +0.0000] | -0.0126 [-0.0287, +0.0000] | -5.0000 [-11.0000, +0.0000] | +7.0000 [+1.0000, +15.0000] | -0.0422 [-0.1073, +0.0000] |
| s2 / b3 / A - F / calibration | +0.0066 [+0.0017, +0.0129] | -0.0005 [-0.0252, +0.0171] | +0.0000 [-0.0075, +0.0075] | -6.0000 [-11.0000, -2.0000] | -1.0000 [-5.0000, +2.0000] | -0.0464 [-0.1207, +0.0000] |
| s2 / b3 / M - frozen Ordered / calibration | -0.0048 [-0.0157, +0.0047] | +0.0000 [+0.0000, +0.0000] | +0.0000 [-0.0164, +0.0152] | +2.0000 [-2.0000, +6.0000] | +3.0000 [-5.0000, +12.0000] | +0.0084 [+0.0000, +0.0275] |
| s2 / b3 / A - frozen Ordered / calibration | +0.0040 [+0.0000, +0.0089] | -0.0005 [-0.0252, +0.0171] | +0.0126 [+0.0025, +0.0252] | +1.0000 [-2.0000, +4.0000] | -5.0000 [-10.0000, -1.0000] | +0.0042 [+0.0000, +0.0138] |
| s2 / argmax / M - A / calibration | -0.0085 [-0.0193, -0.0009] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +6.0000 [+0.0000, +14.0000] | +3.0000 [+0.0000, +7.0000] | +0.0422 [+0.0084, +0.0876] |
| s2 / argmax / M - F / calibration | -0.0020 [-0.0133, +0.0086] | +0.0000 [+0.0000, +0.0000] | -0.0025 [-0.0079, +0.0000] | +0.0000 [-8.0000, +9.0000] | +2.0000 [-2.0000, +6.0000] | -0.0042 [-0.0369, +0.0231] |
| s2 / argmax / A - F / calibration | +0.0065 [-0.0001, +0.0173] | +0.0000 [+0.0000, +0.0000] | -0.0025 [-0.0079, +0.0000] | -6.0000 [-14.0000, -1.0000] | -1.0000 [-4.0000, +2.0000] | -0.0464 [-0.0996, -0.0081] |
| s2 / argmax / M - frozen Ordered / calibration | +0.0084 [-0.0075, +0.0259] | +0.0068 [+0.0000, +0.0191] | +0.0101 [+0.0024, +0.0209] | -6.0000 [-20.0000, +7.0000] | -3.0000 [-9.0000, +3.0000] | +0.0127 [-0.0439, +0.0688] |
| s2 / argmax / A - frozen Ordered / calibration | +0.0169 [+0.0055, +0.0319] | +0.0068 [+0.0000, +0.0191] | +0.0101 [+0.0024, +0.0209] | -12.0000 [-24.0000, -3.0000] | -6.0000 [-11.0000, -2.0000] | -0.0295 [-0.0679, +0.0000] |
| s2 / ordered_setting / M - A / calibration | -0.0039 [-0.0117, +0.0011] | -0.0068 [-0.0190, +0.0000] | -0.0025 [-0.0119, +0.0051] | +2.0000 [+0.0000, +5.0000] | +2.0000 [-2.0000, +8.0000] | +0.0000 [+0.0000, +0.0000] |
| s2 / ordered_setting / M - F / calibration | -0.0011 [-0.0077, +0.0042] | +0.0000 [+0.0000, +0.0000] | -0.0101 [-0.0204, -0.0024] | -4.0000 [-8.0000, -1.0000] | +5.0000 [+1.0000, +11.0000] | -0.0042 [-0.0139, +0.0000] |
| s2 / ordered_setting / A - F / calibration | +0.0027 [-0.0020, +0.0077] | +0.0068 [+0.0000, +0.0190] | -0.0075 [-0.0172, +0.0000] | -6.0000 [-11.0000, -2.0000] | +3.0000 [+0.0000, +7.0000] | -0.0042 [-0.0139, +0.0000] |
| s2 / ordered_setting / M - frozen Ordered / calibration | -0.0010 [-0.0091, +0.0060] | -0.0073 [-0.0267, +0.0000] | +0.0025 [-0.0078, +0.0135] | +0.0000 [-4.0000, +4.0000] | +1.0000 [-4.0000, +8.0000] | +0.0000 [-0.0129, +0.0114] |
| s2 / ordered_setting / A - frozen Ordered / calibration | +0.0029 [-0.0031, +0.0095] | -0.0005 [-0.0252, +0.0171] | +0.0050 [-0.0074, +0.0179] | -2.0000 [-6.0000, +2.0000] | -1.0000 [-6.0250, +4.0000] | +0.0000 [-0.0129, +0.0114] |

## Three-seed verdicts

| decoding / comparison / metric | selection | calibration |
|---|---|---|
| b3 / M - A / multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| b3 / M - A / native macro-F1 | no consistent difference detected | no consistent difference detected |
| b3 / M - A / multi boundary recall | no consistent difference detected | no consistent difference detected |
| b3 / M - A / multi FP segments | no consistent difference detected | no consistent difference detected |
| b3 / M - A / multi FN segments | no consistent difference detected | no consistent difference detected |
| b3 / M - A / native cuts per clip | no consistent difference detected | no consistent difference detected |
| b3 / M - F / multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| b3 / M - F / native macro-F1 | no consistent difference detected | no consistent difference detected |
| b3 / M - F / multi boundary recall | no consistent difference detected | no consistent difference detected |
| b3 / M - F / multi FP segments | no consistent difference detected | no consistent difference detected |
| b3 / M - F / multi FN segments | no consistent difference detected | no consistent difference detected |
| b3 / M - F / native cuts per clip | no consistent difference detected | no consistent difference detected |
| b3 / A - F / multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| b3 / A - F / native macro-F1 | no consistent difference detected | no consistent difference detected |
| b3 / A - F / multi boundary recall | no consistent difference detected | no consistent difference detected |
| b3 / A - F / multi FP segments | no consistent difference detected | no consistent difference detected |
| b3 / A - F / multi FN segments | no consistent difference detected | no consistent difference detected |
| b3 / A - F / native cuts per clip | no consistent difference detected | no consistent difference detected |
| b3 / M - frozen Ordered / multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| b3 / M - frozen Ordered / native macro-F1 | no consistent difference detected | no consistent difference detected |
| b3 / M - frozen Ordered / multi boundary recall | no consistent difference detected | no consistent difference detected |
| b3 / M - frozen Ordered / multi FP segments | no consistent difference detected | no consistent difference detected |
| b3 / M - frozen Ordered / multi FN segments | no consistent difference detected | no consistent difference detected |
| b3 / M - frozen Ordered / native cuts per clip | no consistent difference detected | no consistent difference detected |
| b3 / A - frozen Ordered / multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| b3 / A - frozen Ordered / native macro-F1 | no consistent difference detected | no consistent difference detected |
| b3 / A - frozen Ordered / multi boundary recall | no consistent difference detected | no consistent difference detected |
| b3 / A - frozen Ordered / multi FP segments | no consistent difference detected | no consistent difference detected |
| b3 / A - frozen Ordered / multi FN segments | no consistent difference detected | no consistent difference detected |
| b3 / A - frozen Ordered / native cuts per clip | no consistent difference detected | no consistent difference detected |
| argmax / M - A / multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| argmax / M - A / native macro-F1 | no consistent difference detected | no consistent difference detected |
| argmax / M - A / multi boundary recall | no consistent difference detected | no consistent difference detected |
| argmax / M - A / multi FP segments | no consistent difference detected | no consistent difference detected |
| argmax / M - A / multi FN segments | no consistent difference detected | no consistent difference detected |
| argmax / M - A / native cuts per clip | no consistent difference detected | no consistent difference detected |
| argmax / M - F / multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| argmax / M - F / native macro-F1 | no consistent difference detected | no consistent difference detected |
| argmax / M - F / multi boundary recall | no consistent difference detected | no consistent difference detected |
| argmax / M - F / multi FP segments | no consistent difference detected | no consistent difference detected |
| argmax / M - F / multi FN segments | no consistent difference detected | no consistent difference detected |
| argmax / M - F / native cuts per clip | no consistent difference detected | no consistent difference detected |
| argmax / A - F / multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| argmax / A - F / native macro-F1 | no consistent difference detected | no consistent difference detected |
| argmax / A - F / multi boundary recall | no consistent difference detected | no consistent difference detected |
| argmax / A - F / multi FP segments | no consistent difference detected | no consistent difference detected |
| argmax / A - F / multi FN segments | no consistent difference detected | no consistent difference detected |
| argmax / A - F / native cuts per clip | no consistent difference detected | no consistent difference detected |
| argmax / M - frozen Ordered / multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| argmax / M - frozen Ordered / native macro-F1 | no consistent difference detected | no consistent difference detected |
| argmax / M - frozen Ordered / multi boundary recall | no consistent difference detected | no consistent difference detected |
| argmax / M - frozen Ordered / multi FP segments | no consistent difference detected | no consistent difference detected |
| argmax / M - frozen Ordered / multi FN segments | no consistent difference detected | no consistent difference detected |
| argmax / M - frozen Ordered / native cuts per clip | no consistent difference detected | no consistent difference detected |
| argmax / A - frozen Ordered / multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| argmax / A - frozen Ordered / native macro-F1 | no consistent difference detected | no consistent difference detected |
| argmax / A - frozen Ordered / multi boundary recall | no consistent difference detected | no consistent difference detected |
| argmax / A - frozen Ordered / multi FP segments | no consistent difference detected | no consistent difference detected |
| argmax / A - frozen Ordered / multi FN segments | no consistent difference detected | no consistent difference detected |
| argmax / A - frozen Ordered / native cuts per clip | no consistent difference detected | no consistent difference detected |
| ordered_setting / M - A / multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| ordered_setting / M - A / native macro-F1 | no consistent difference detected | no consistent difference detected |
| ordered_setting / M - A / multi boundary recall | no consistent difference detected | no consistent difference detected |
| ordered_setting / M - A / multi FP segments | no consistent difference detected | no consistent difference detected |
| ordered_setting / M - A / multi FN segments | no consistent difference detected | no consistent difference detected |
| ordered_setting / M - A / native cuts per clip | no consistent difference detected | no consistent difference detected |
| ordered_setting / M - F / multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| ordered_setting / M - F / native macro-F1 | no consistent difference detected | no consistent difference detected |
| ordered_setting / M - F / multi boundary recall | no consistent difference detected | no consistent difference detected |
| ordered_setting / M - F / multi FP segments | consistent negative | no consistent difference detected |
| ordered_setting / M - F / multi FN segments | no consistent difference detected | no consistent difference detected |
| ordered_setting / M - F / native cuts per clip | consistent negative | no consistent difference detected |
| ordered_setting / A - F / multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| ordered_setting / A - F / native macro-F1 | no consistent difference detected | no consistent difference detected |
| ordered_setting / A - F / multi boundary recall | consistent negative | no consistent difference detected |
| ordered_setting / A - F / multi FP segments | consistent negative | no consistent difference detected |
| ordered_setting / A - F / multi FN segments | no consistent difference detected | no consistent difference detected |
| ordered_setting / A - F / native cuts per clip | consistent negative | no consistent difference detected |
| ordered_setting / M - frozen Ordered / multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| ordered_setting / M - frozen Ordered / native macro-F1 | no consistent difference detected | no consistent difference detected |
| ordered_setting / M - frozen Ordered / multi boundary recall | no consistent difference detected | no consistent difference detected |
| ordered_setting / M - frozen Ordered / multi FP segments | no consistent difference detected | no consistent difference detected |
| ordered_setting / M - frozen Ordered / multi FN segments | no consistent difference detected | no consistent difference detected |
| ordered_setting / M - frozen Ordered / native cuts per clip | no consistent difference detected | no consistent difference detected |
| ordered_setting / A - frozen Ordered / multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| ordered_setting / A - frozen Ordered / native macro-F1 | no consistent difference detected | no consistent difference detected |
| ordered_setting / A - frozen Ordered / multi boundary recall | no consistent difference detected | no consistent difference detected |
| ordered_setting / A - frozen Ordered / multi FP segments | no consistent difference detected | no consistent difference detected |
| ordered_setting / A - frozen Ordered / multi FN segments | no consistent difference detected | no consistent difference detected |
| ordered_setting / A - frozen Ordered / native cuts per clip | no consistent difference detected | no consistent difference detected |

## Pooled recovered and lost reference segments (B3 systems)

| pair / half / population | recovered | lost | new_unmatched_pred | removed_unmatched_pred | recovered_short | lost_short | recovered_xyx | lost_xyx |
|---|---|---|---|---|---|---|---|---|
| M - A / selection / native | 4 | 0 | 8 | 20 | 0 | 0 | 0 | 0 |
| M - A / selection / pairs | 2 | 4 | 4 | 5 | 1 | 2 | 0 | 0 |
| M - A / selection / multi | 17 | 4 | 37 | 34 | 11 | 1 | 2 | 0 |
| M - A / calibration / native | 1 | 1 | 4 | 3 | 0 | 0 | 0 | 0 |
| M - A / calibration / pairs | 3 | 0 | 3 | 4 | 1 | 0 | 0 | 0 |
| M - A / calibration / multi | 1 | 10 | 18 | 12 | 1 | 4 | 0 | 0 |
| M - F / selection / native | 6 | 0 | 1 | 31 | 0 | 0 | 0 | 0 |
| M - F / selection / pairs | 4 | 3 | 4 | 17 | 1 | 1 | 0 | 0 |
| M - F / selection / multi | 11 | 5 | 21 | 76 | 0 | 2 | 0 | 0 |
| M - F / calibration / native | 0 | 0 | 3 | 15 | 0 | 0 | 0 | 0 |
| M - F / calibration / pairs | 2 | 3 | 2 | 2 | 2 | 3 | 0 | 0 |
| M - F / calibration / multi | 1 | 8 | 13 | 17 | 1 | 3 | 0 | 0 |
| A - F / selection / native | 3 | 1 | 11 | 29 | 0 | 0 | 0 | 0 |
| A - F / selection / pairs | 4 | 1 | 1 | 13 | 1 | 0 | 0 | 0 |
| A - F / selection / multi | 10 | 17 | 36 | 94 | 0 | 12 | 0 | 2 |
| A - F / calibration / native | 1 | 1 | 4 | 17 | 0 | 0 | 0 | 0 |
| A - F / calibration / pairs | 2 | 6 | 4 | 3 | 2 | 4 | 0 | 0 |
| A - F / calibration / multi | 4 | 2 | 10 | 20 | 3 | 2 | 0 | 0 |
| M - frozen Ordered / selection / native | 4 | 0 | 4 | 17 | 0 | 0 | 0 | 0 |
| M - frozen Ordered / selection / pairs | 7 | 2 | 5 | 13 | 4 | 0 | 0 | 0 |
| M - frozen Ordered / selection / multi | 27 | 11 | 43 | 41 | 14 | 3 | 2 | 0 |
| M - frozen Ordered / calibration / native | 1 | 1 | 4 | 5 | 0 | 0 | 0 | 0 |
| M - frozen Ordered / calibration / pairs | 10 | 3 | 4 | 4 | 8 | 3 | 0 | 0 |
| M - frozen Ordered / calibration / multi | 12 | 11 | 18 | 12 | 11 | 6 | 0 | 0 |
| A - frozen Ordered / selection / native | 2 | 2 | 16 | 17 | 0 | 0 | 0 | 0 |
| A - frozen Ordered / selection / pairs | 9 | 2 | 3 | 10 | 6 | 1 | 0 | 0 |
| A - frozen Ordered / selection / multi | 16 | 13 | 36 | 37 | 5 | 4 | 0 | 0 |
| A - frozen Ordered / calibration / native | 2 | 2 | 5 | 7 | 0 | 0 | 0 | 0 |
| A - frozen Ordered / calibration / pairs | 9 | 5 | 4 | 3 | 7 | 3 | 0 | 0 |
| A - frozen Ordered / calibration / multi | 13 | 3 | 12 | 12 | 11 | 3 | 0 | 0 |

## Shared seed cases (distinct cases over the three seeds and how many appear in 1 / 2 / 3 seeds; multi streams)

| pair / half | recovered | lost | new_unmatched_sequences | removed_unmatched_sequences |
|---|---|---|---|---|
| M - A / selection | 11 (5 / 6 / 0) | 4 (4 / 0 / 0) | 18 (10 / 6 / 2) | 16 (7 / 6 / 3) |
| M - F / selection | 9 (7 / 2 / 0) | 5 (5 / 0 / 0) | 10 (5 / 5 / 0) | 25 (16 / 7 / 2) |
| A - F / selection | 8 (6 / 2 / 0) | 12 (7 / 5 / 0) | 13 (4 / 6 / 3) | 30 (16 / 11 / 3) |
| M - frozen Ordered / selection | 16 (9 / 3 / 4) | 8 (5 / 3 / 0) | 19 (13 / 3 / 3) | 21 (13 / 5 / 3) |
| A - frozen Ordered / selection | 13 (11 / 1 / 1) | 8 (3 / 5 / 0) | 13 (4 / 7 / 2) | 19 (12 / 5 / 2) |
| M - A / calibration | 1 (1 / 0 / 0) | 9 (8 / 1 / 0) | 11 (7 / 4 / 0) | 7 (5 / 2 / 0) |
| M - F / calibration | 1 (1 / 0 / 0) | 8 (8 / 0 / 0) | 8 (6 / 2 / 0) | 10 (7 / 3 / 0) |
| A - F / calibration | 4 (4 / 0 / 0) | 2 (2 / 0 / 0) | 6 (3 / 3 / 0) | 11 (5 / 5 / 1) |
| M - frozen Ordered / calibration | 7 (2 / 5 / 0) | 10 (9 / 1 / 0) | 11 (7 / 3 / 1) | 7 (6 / 0 / 1) |
| A - frozen Ordered / calibration | 8 (3 / 5 / 0) | 3 (3 / 0 / 0) | 8 (6 / 1 / 1) | 7 (5 / 1 / 1) |

## Goal

| group | selection mean | calibration mean | constraint failures | met |
|---|---|---|---|---|
| A | 0.9557 | 0.9850 | 0 | True |
| M | 0.9593 | 0.9801 | 0 | False |
| F | 0.9404 | 0.9812 | 0 | False |
| frozen Ordered | 0.9544 | 0.9817 | 1 | False |

## Training penalties (first and last epoch means; training data only)

| run | epoch | CE | S_A | S_M | equal-label penalty | different-label penalty |
|---|---|---|---|---|---|---|
| all_s0 | 0 | 0.1876 | 3.2491 | 2.5300 | 2.3554 | 7.5248 |
| all_s0 | 9 | 0.1655 | 1.2683 | 0.7865 | 0.6972 | 4.0432 |
| masked_s0 | 0 | 0.1863 | 3.2548 | 2.5319 | 2.3569 | 7.5502 |
| masked_s0 | 9 | 0.1601 | 1.4443 | 0.7403 | 0.6649 | 5.2315 |
| all_s1 | 0 | 0.1611 | 3.1036 | 2.3559 | 2.1943 | 7.4537 |
| all_s1 | 9 | 0.1372 | 1.2565 | 0.7020 | 0.6288 | 4.2501 |
| masked_s1 | 0 | 0.1604 | 3.1274 | 2.3768 | 2.2145 | 7.4944 |
| masked_s1 | 9 | 0.1063 | 1.4902 | 0.6134 | 0.5667 | 5.8947 |
| all_s2 | 0 | 0.2019 | 3.0046 | 2.3296 | 2.1562 | 7.1796 |
| all_s2 | 9 | 0.1601 | 1.2809 | 0.7465 | 0.6567 | 4.2763 |
| masked_s2 | 0 | 0.2017 | 3.0350 | 2.3540 | 2.1780 | 7.2523 |
| masked_s2 | 9 | 0.1389 | 1.4716 | 0.6492 | 0.5857 | 5.7226 |

## Readings

```
{
 "applied": [
  "trade_off",
  "consistent_negatives",
  "no_qualifying_gain",
  "end"
 ],
 "by_half": {
  "selection": [
   "trade_off",
   "consistent_negatives"
  ],
  "calibration": [
   "trade_off"
  ]
 }
}
```
