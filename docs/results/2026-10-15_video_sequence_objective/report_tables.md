# A07 sequence-objective round: tables

Recompute: `python scripts/report_video_sequence_objective.py --config configs/experiments/video_sequence_objective_v1.yaml --selection --choices-sha256 <sha> --choices-commit <commit>`

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
| B + B3 s0 | selection | 0.9494 | 0.9236 | 0.9817 | 0.9518 | 0.9786 | 0.0563 | 0.0217 | 0.9617 | 2 | 0 |
| B + B3 s0 | calibration | 0.9770 | 0.9680 | 0.9874 | 0.9776 | 0.9929 | 0.0211 | 0.0348 | 0.9797 | 1 | 0 |
| B argmax s0 | selection | 0.9154 | 0.8529 | 0.9869 | 0.9150 | 0.9877 | 0.1056 | 0.0348 | 0.9412 | 1 | 0 |
| B argmax s0 | calibration | 0.9602 | 0.9447 | 0.9874 | 0.9656 | 0.9929 | 0.1308 | 0.0398 | 0.9602 | 1 | 0 |
| B at Ordered setting s0 | selection | 0.9494 | 0.9236 | 0.9817 | 0.9518 | 0.9786 | 0.0634 | 0.0217 | 0.9617 | 2 | 0 |
| B at Ordered setting s0 | calibration | 0.9714 | 0.9585 | 0.9874 | 0.9728 | 0.9929 | 0.0211 | 0.0398 | 0.9797 | 1 | 0 |
| O + B3 s0 | selection | 0.9541 | 0.9352 | 0.9817 | 0.9579 | 0.9786 | 0.0493 | 0.0217 | 0.9569 | 2 | 0 |
| O + B3 s0 | calibration | 0.9779 | 0.9752 | 0.9874 | 0.9813 | 0.9861 | 0.0211 | 0.0249 | 0.9797 | 2 | 0 |
| O argmax s0 | selection | 0.9154 | 0.8529 | 0.9869 | 0.9150 | 0.9877 | 0.1056 | 0.0348 | 0.9412 | 1 | 0 |
| O argmax s0 | calibration | 0.9602 | 0.9471 | 0.9899 | 0.9681 | 0.9861 | 0.1181 | 0.0398 | 0.9657 | 2 | 0 |
| O at Ordered setting s0 | selection | 0.9485 | 0.9214 | 0.9817 | 0.9506 | 0.9786 | 0.0634 | 0.0217 | 0.9569 | 2 | 0 |
| O at Ordered setting s0 | calibration | 0.9714 | 0.9610 | 0.9899 | 0.9752 | 0.9861 | 0.0211 | 0.0398 | 0.9797 | 2 | 0 |
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
| B + B3 s1 | selection | 0.9528 | 0.9467 | 0.9764 | 0.9613 | 0.9786 | 0.0352 | 0.0130 | 0.9619 | 2 | 0 |
| B + B3 s1 | calibration | 0.9885 | 0.9754 | 0.9950 | 0.9851 | 0.9929 | 0.0169 | 0.0149 | 0.9884 | 1 | 0 |
| B argmax s1 | selection | 0.9049 | 0.8422 | 0.9921 | 0.9111 | 0.9877 | 0.1056 | 0.0478 | 0.9506 | 1 | 0 |
| B argmax s1 | calibration | 0.9752 | 0.9612 | 0.9950 | 0.9778 | 0.9929 | 0.0759 | 0.0199 | 0.9857 | 1 | 0 |
| B at Ordered setting s1 | selection | 0.9454 | 0.9282 | 0.9817 | 0.9542 | 0.9877 | 0.0599 | 0.0174 | 0.9619 | 1 | 0 |
| B at Ordered setting s1 | calibration | 0.9885 | 0.9754 | 0.9950 | 0.9851 | 0.9929 | 0.0169 | 0.0149 | 0.9884 | 1 | 0 |
| O + B3 s1 | selection | 0.9528 | 0.9444 | 0.9791 | 0.9614 | 0.9786 | 0.0352 | 0.0130 | 0.9642 | 2 | 0 |
| O + B3 s1 | calibration | 0.9885 | 0.9754 | 0.9950 | 0.9851 | 0.9929 | 0.0169 | 0.0149 | 0.9855 | 1 | 0 |
| O argmax s1 | selection | 0.9021 | 0.8404 | 0.9921 | 0.9100 | 0.9877 | 0.1021 | 0.0478 | 0.9506 | 1 | 0 |
| O argmax s1 | calibration | 0.9771 | 0.9612 | 0.9950 | 0.9778 | 0.9929 | 0.0759 | 0.0199 | 0.9857 | 1 | 0 |
| O at Ordered setting s1 | selection | 0.9453 | 0.9303 | 0.9791 | 0.9541 | 0.9877 | 0.0599 | 0.0174 | 0.9642 | 1 | 0 |
| O at Ordered setting s1 | calibration | 0.9885 | 0.9754 | 0.9950 | 0.9851 | 0.9929 | 0.0169 | 0.0149 | 0.9856 | 1 | 0 |
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
| B + B3 s2 | selection | 0.9200 | 0.8808 | 0.9869 | 0.9309 | 0.9786 | 0.0775 | 0.0304 | 0.9551 | 2 | 0 |
| B + B3 s2 | calibration | 0.9781 | 0.9635 | 0.9950 | 0.9790 | 0.9861 | 0.0591 | 0.0348 | 0.9769 | 2 | 0 |
| B argmax s2 | selection | 0.9052 | 0.8533 | 0.9895 | 0.9164 | 0.9877 | 0.1268 | 0.0391 | 0.9439 | 1 | 0 |
| B argmax s2 | calibration | 0.9706 | 0.9542 | 0.9950 | 0.9742 | 0.9929 | 0.1308 | 0.0448 | 0.9545 | 1 | 0 |
| B at Ordered setting s2 | selection | 0.9398 | 0.9191 | 0.9817 | 0.9494 | 0.9786 | 0.0634 | 0.0130 | 0.9619 | 2 | 0 |
| B at Ordered setting s2 | calibration | 0.9799 | 0.9705 | 0.9925 | 0.9814 | 0.9788 | 0.0211 | 0.0299 | 0.9797 | 2 | 1 |
| O + B3 s2 | selection | 0.9210 | 0.8788 | 0.9869 | 0.9297 | 0.9786 | 0.0845 | 0.0304 | 0.9551 | 2 | 0 |
| O + B3 s2 | calibration | 0.9762 | 0.9635 | 0.9950 | 0.9790 | 0.9861 | 0.0675 | 0.0348 | 0.9769 | 2 | 0 |
| O argmax s2 | selection | 0.9043 | 0.8514 | 0.9895 | 0.9153 | 0.9877 | 0.1303 | 0.0391 | 0.9439 | 1 | 0 |
| O argmax s2 | calibration | 0.9706 | 0.9542 | 0.9950 | 0.9742 | 0.9929 | 0.1224 | 0.0448 | 0.9545 | 1 | 0 |
| O at Ordered setting s2 | selection | 0.9436 | 0.9236 | 0.9817 | 0.9518 | 0.9786 | 0.0704 | 0.0130 | 0.9642 | 2 | 0 |
| O at Ordered setting s2 | calibration | 0.9780 | 0.9705 | 0.9925 | 0.9814 | 0.9788 | 0.0211 | 0.0299 | 0.9767 | 2 | 1 |

## Paired comparisons (seed | decoding | comparison | half; 95% bootstrap intervals)

| comparison | multi segment F1@50 | native macro-F1 | multi boundary recall | multi FP segments | multi FN segments | native cuts per clip |
|---|---|---|---|---|---|---|
| s0|b3|O - B|selection | +0.0046 [+0.0010, +0.0090] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -5.0000 [-10.0000, -1.0000] | +0.0000 [+0.0000, +0.0000] | -0.0070 [-0.0233, +0.0000] |
| s0|b3|B - F|selection | -0.0037 [-0.0074, -0.0009] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +4.0000 [+1.0000, +8.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [-0.0229, +0.0200] |
| s0|b3|O - F|selection | +0.0009 [+0.0000, +0.0029] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -1.0000 [-3.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0070 [-0.0260, +0.0000] |
| s0|b3|O - frozen Ordered|selection | +0.0156 [+0.0026, +0.0304] | +0.0000 [+0.0000, +0.0000] | +0.0236 [+0.0082, +0.0407] | +0.0000 [-8.0000, +8.0000] | -15.0000 [-25.0000, -7.0000] | +0.0070 [+0.0000, +0.0192] |
| s0|b3|B - frozen Ordered|selection | +0.0109 [-0.0024, +0.0261] | +0.0000 [+0.0000, +0.0000] | +0.0236 [+0.0082, +0.0407] | +5.0000 [-3.0000, +13.0000] | -15.0000 [-25.0000, -7.0000] | +0.0141 [+0.0000, +0.0307] |
| s0|b3|F - frozen Ordered|selection | +0.0146 [+0.0019, +0.0294] | +0.0000 [+0.0000, +0.0000] | +0.0236 [+0.0082, +0.0407] | +1.0000 [-6.0000, +9.0000] | -15.0000 [-25.0000, -7.0000] | +0.0141 [+0.0000, +0.0343] |
| s0|argmax|O - B|selection | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| s0|argmax|B - F|selection | +0.0022 [-0.0035, +0.0092] | +0.0000 [+0.0000, +0.0000] | +0.0026 [+0.0000, +0.0081] | +1.0000 [-4.0000, +7.0000] | -3.0000 [-7.0000, +0.0000] | +0.0070 [+0.0000, +0.0260] |
| s0|argmax|O - F|selection | +0.0022 [-0.0035, +0.0092] | +0.0000 [+0.0000, +0.0000] | +0.0026 [+0.0000, +0.0081] | +1.0000 [-4.0000, +7.0000] | -3.0000 [-7.0000, +0.0000] | +0.0070 [+0.0000, +0.0260] |
| s0|argmax|O - frozen Ordered|selection | +0.0226 [+0.0082, +0.0384] | +0.0177 [+0.0000, +0.0395] | +0.0079 [+0.0000, +0.0177] | -12.0000 [-25.0250, +0.0000] | -12.0000 [-19.0000, -6.0000] | +0.0106 [-0.0163, +0.0483] |
| s0|argmax|B - frozen Ordered|selection | +0.0226 [+0.0082, +0.0384] | +0.0177 [+0.0000, +0.0395] | +0.0079 [+0.0000, +0.0177] | -12.0000 [-25.0250, +0.0000] | -12.0000 [-19.0000, -6.0000] | +0.0106 [-0.0163, +0.0483] |
| s0|argmax|F - frozen Ordered|selection | +0.0204 [+0.0062, +0.0358] | +0.0177 [+0.0000, +0.0395] | +0.0052 [-0.0051, +0.0158] | -13.0000 [-26.0000, -2.0000] | -9.0000 [-15.0000, -3.0000] | +0.0035 [-0.0301, +0.0433] |
| s0|ordered_setting|O - B|selection | -0.0009 [-0.0029, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +1.0000 [+0.0000, +3.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| s0|ordered_setting|B - F|selection | -0.0009 [-0.0029, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +1.0000 [+0.0000, +3.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| s0|ordered_setting|O - F|selection | -0.0018 [-0.0046, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +2.0000 [+0.0000, +5.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| s0|ordered_setting|O - frozen Ordered|selection | +0.0100 [-0.0038, +0.0253] | +0.0000 [+0.0000, +0.0000] | +0.0236 [+0.0082, +0.0407] | +6.0000 [-3.0000, +15.0000] | -15.0000 [-25.0000, -7.0000] | +0.0211 [+0.0000, +0.0435] |
| s0|ordered_setting|B - frozen Ordered|selection | +0.0109 [-0.0024, +0.0261] | +0.0000 [+0.0000, +0.0000] | +0.0236 [+0.0082, +0.0407] | +5.0000 [-3.0000, +13.0000] | -15.0000 [-25.0000, -7.0000] | +0.0211 [+0.0000, +0.0435] |
| s0|ordered_setting|F - frozen Ordered|selection | +0.0118 [-0.0014, +0.0274] | +0.0000 [+0.0000, +0.0000] | +0.0236 [+0.0082, +0.0407] | +4.0000 [-4.0000, +12.0000] | -15.0000 [-25.0000, -7.0000] | +0.0211 [+0.0000, +0.0435] |
| s1|b3|O - B|selection | +0.0001 [-0.0028, +0.0031] | +0.0000 [+0.0000, +0.0000] | +0.0026 [+0.0000, +0.0082] | +1.0000 [+0.0000, +3.0000] | -1.0000 [-3.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| s1|b3|B - F|selection | +0.0020 [-0.0071, +0.0110] | +0.0000 [+0.0000, +0.0000] | +0.0026 [-0.0055, +0.0116] | -1.0000 [-6.0000, +5.0000] | -1.0000 [-7.0000, +4.0000] | +0.0000 [+0.0000, +0.0000] |
| s1|b3|O - F|selection | +0.0021 [-0.0068, +0.0111] | +0.0000 [+0.0000, +0.0000] | +0.0052 [+0.0000, +0.0133] | +0.0000 [-4.0000, +5.0000] | -2.0000 [-8.0000, +2.0000] | +0.0000 [+0.0000, +0.0000] |
| s1|b3|O - frozen Ordered|selection | -0.0075 [-0.0206, +0.0033] | +0.0000 [+0.0000, +0.0000] | +0.0105 [+0.0000, +0.0233] | +9.0000 [+2.0000, +19.0000] | -1.0000 [-6.0000, +4.0000] | -0.0035 [-0.0203, +0.0102] |
| s1|b3|B - frozen Ordered|selection | -0.0076 [-0.0205, +0.0032] | +0.0000 [+0.0000, +0.0000] | +0.0079 [-0.0052, +0.0216] | +8.0000 [+1.0000, +18.0000] | +0.0000 [-6.0000, +6.0000] | -0.0035 [-0.0203, +0.0102] |
| s1|b3|F - frozen Ordered|selection | -0.0095 [-0.0233, +0.0013] | +0.0000 [+0.0000, +0.0000] | +0.0052 [-0.0101, +0.0192] | +9.0000 [+2.0000, +19.0000] | +1.0000 [-5.0000, +8.0000] | -0.0035 [-0.0203, +0.0102] |
| s1|argmax|O - B|selection | -0.0027 [-0.0078, +0.0014] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +2.0000 [-2.0000, +6.0000] | +1.0000 [+0.0000, +3.0000] | -0.0035 [-0.0096, +0.0000] |
| s1|argmax|B - F|selection | +0.0044 [-0.0009, +0.0105] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -4.0000 [-10.0000, +1.0000] | -1.0000 [-3.0000, +0.0000] | +0.0070 [+0.0000, +0.0192] |
| s1|argmax|O - F|selection | +0.0017 [+0.0000, +0.0051] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -2.0000 [-6.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0035 [+0.0000, +0.0096] |
| s1|argmax|O - frozen Ordered|selection | -0.0228 [-0.0416, -0.0055] | +0.0000 [+0.0000, +0.0000] | +0.0079 [-0.0027, +0.0193] | +23.0000 [+7.0000, +41.0000] | +3.0000 [-2.0000, +8.0000] | -0.0106 [-0.0315, +0.0000] |
| s1|argmax|B - frozen Ordered|selection | -0.0201 [-0.0393, -0.0035] | +0.0000 [+0.0000, +0.0000] | +0.0079 [-0.0027, +0.0193] | +21.0000 [+5.0000, +39.0000] | +2.0000 [-3.0000, +8.0000] | -0.0070 [-0.0300, +0.0068] |
| s1|argmax|F - frozen Ordered|selection | -0.0245 [-0.0435, -0.0068] | +0.0000 [+0.0000, +0.0000] | +0.0079 [-0.0027, +0.0193] | +25.0000 [+8.0000, +43.0000] | +3.0000 [-2.0000, +8.0000] | -0.0141 [-0.0332, +0.0000] |
| s1|ordered_setting|O - B|selection | -0.0001 [-0.0031, +0.0027] | +0.0000 [+0.0000, +0.0000] | -0.0026 [-0.0082, +0.0000] | -1.0000 [-3.0000, +0.0000] | +1.0000 [+0.0000, +3.0000] | +0.0000 [+0.0000, +0.0000] |
| s1|ordered_setting|B - F|selection | -0.0028 [-0.0105, +0.0030] | +0.0000 [+0.0000, +0.0000] | +0.0026 [+0.0000, +0.0082] | +3.0000 [-1.0000, +9.0000] | +0.0000 [-3.0000, +3.0000] | +0.0070 [+0.0000, +0.0192] |
| s1|ordered_setting|O - F|selection | -0.0029 [-0.0100, +0.0019] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +2.0000 [-2.0000, +7.0000] | +1.0000 [+0.0000, +3.0000] | +0.0070 [+0.0000, +0.0192] |
| s1|ordered_setting|O - frozen Ordered|selection | -0.0150 [-0.0314, -0.0017] | +0.0091 [+0.0000, +0.0293] | +0.0105 [+0.0000, +0.0233] | +16.0000 [+6.0000, +29.0000] | +0.0000 [-6.0000, +6.0000] | +0.0211 [-0.0094, +0.0613] |
| s1|ordered_setting|B - frozen Ordered|selection | -0.0149 [-0.0311, -0.0015] | +0.0091 [+0.0000, +0.0293] | +0.0131 [+0.0000, +0.0264] | +17.0000 [+7.0000, +30.0000] | -1.0000 [-7.0000, +5.0000] | +0.0211 [-0.0094, +0.0613] |
| s1|ordered_setting|F - frozen Ordered|selection | -0.0121 [-0.0278, +0.0001] | +0.0091 [+0.0000, +0.0293] | +0.0105 [+0.0000, +0.0233] | +14.0000 [+5.0000, +26.0000] | -1.0000 [-6.0000, +4.0000] | +0.0141 [-0.0134, +0.0549] |
| s2|b3|O - B|selection | +0.0010 [-0.0052, +0.0088] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [-7.0000, +6.0000] | -1.0000 [-3.0000, +0.0000] | +0.0070 [+0.0000, +0.0192] |
| s2|b3|B - F|selection | +0.0026 [-0.0017, +0.0076] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -3.0000 [-9.0000, +2.0000] | +0.0000 [+0.0000, +0.0000] | -0.0106 [-0.0242, +0.0000] |
| s2|b3|O - F|selection | +0.0036 [-0.0009, +0.0108] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -3.0000 [-9.0000, +1.0000] | -1.0000 [-3.0000, +0.0000] | -0.0035 [-0.0116, +0.0000] |
| s2|b3|O - frozen Ordered|selection | -0.0433 [-0.0693, -0.0196] | +0.0085 [+0.0000, +0.0241] | +0.0131 [+0.0000, +0.0291] | +44.0000 [+25.0000, +66.0000] | +3.0000 [-6.0000, +11.0250] | +0.0387 [+0.0039, +0.0788] |
| s2|b3|B - frozen Ordered|selection | -0.0444 [-0.0709, -0.0206] | +0.0085 [+0.0000, +0.0241] | +0.0131 [+0.0000, +0.0291] | +44.0000 [+25.0000, +66.0000] | +4.0000 [-5.0000, +13.0000] | +0.0317 [+0.0039, +0.0680] |
| s2|b3|F - frozen Ordered|selection | -0.0470 [-0.0740, -0.0224] | +0.0085 [+0.0000, +0.0241] | +0.0131 [+0.0000, +0.0291] | +47.0000 [+27.0000, +70.0000] | +4.0000 [-5.0000, +13.0000] | +0.0423 [+0.0076, +0.0820] |
| s2|argmax|O - B|selection | -0.0008 [-0.0026, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +1.0000 [+0.0000, +3.0000] | +0.0000 [+0.0000, +0.0000] | +0.0035 [+0.0000, +0.0096] |
| s2|argmax|B - F|selection | +0.0017 [+0.0000, +0.0043] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -2.0000 [-5.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0035 [-0.0096, +0.0000] |
| s2|argmax|O - F|selection | +0.0008 [-0.0017, +0.0040] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -1.0000 [-5.0000, +2.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| s2|argmax|O - frozen Ordered|selection | -0.0215 [-0.0383, -0.0066] | +0.0177 [+0.0000, +0.0380] | +0.0052 [-0.0052, +0.0157] | +19.0000 [+6.0000, +32.0000] | +5.0000 [-1.0000, +12.0000] | +0.0141 [-0.0192, +0.0583] |
| s2|argmax|B - frozen Ordered|selection | -0.0207 [-0.0378, -0.0058] | +0.0177 [+0.0000, +0.0380] | +0.0052 [-0.0052, +0.0157] | +18.0000 [+6.0000, +32.0000] | +5.0000 [-1.0000, +12.0000] | +0.0106 [-0.0211, +0.0565] |
| s2|argmax|F - frozen Ordered|selection | -0.0224 [-0.0393, -0.0077] | +0.0177 [+0.0000, +0.0380] | +0.0052 [-0.0052, +0.0157] | +20.0000 [+8.0000, +34.0000] | +5.0000 [-1.0000, +12.0000] | +0.0141 [-0.0192, +0.0583] |
| s2|ordered_setting|O - B|selection | +0.0038 [+0.0000, +0.0111] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -3.0000 [-8.0000, +0.0000] | -1.0000 [-3.0000, +0.0000] | +0.0070 [+0.0000, +0.0192] |
| s2|ordered_setting|B - F|selection | -0.0009 [-0.0030, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +1.0000 [+0.0000, +3.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| s2|ordered_setting|O - F|selection | +0.0029 [+0.0000, +0.0090] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -2.0000 [-6.0000, +0.0000] | -1.0000 [-3.0000, +0.0000] | +0.0070 [+0.0000, +0.0192] |
| s2|ordered_setting|O - frozen Ordered|selection | -0.0208 [-0.0417, -0.0021] | +0.0085 [+0.0000, +0.0241] | +0.0079 [-0.0052, +0.0234] | +20.0000 [+7.0000, +35.0000] | +2.0000 [-7.0000, +10.0000] | +0.0246 [+0.0000, +0.0521] |
| s2|ordered_setting|B - frozen Ordered|selection | -0.0246 [-0.0456, -0.0050] | +0.0085 [+0.0000, +0.0241] | +0.0079 [-0.0052, +0.0234] | +23.0000 [+9.0000, +39.0000] | +3.0000 [-6.0000, +11.0000] | +0.0176 [+0.0000, +0.0388] |
| s2|ordered_setting|F - frozen Ordered|selection | -0.0236 [-0.0448, -0.0041] | +0.0085 [+0.0000, +0.0241] | +0.0079 [-0.0052, +0.0234] | +22.0000 [+9.0000, +38.0000] | +3.0000 [-6.0000, +11.0000] | +0.0176 [+0.0000, +0.0388] |
| s0|b3|O - B|calibration | +0.0009 [-0.0001, +0.0030] | -0.0068 [-0.0191, +0.0000] | +0.0000 [+0.0000, +0.0000] | -2.0000 [-5.0000, +0.0000] | +1.0000 [+0.0000, +3.0000] | +0.0000 [+0.0000, +0.0000] |
| s0|b3|B - F|calibration | -0.0009 [-0.0040, +0.0019] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +2.0000 [-2.0000, +7.0000] | -1.0000 [-3.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| s0|b3|O - F|calibration | +0.0000 [-0.0028, +0.0028] | -0.0068 [-0.0191, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [-3.0000, +3.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| s0|b3|O - frozen Ordered|calibration | -0.0038 [-0.0121, +0.0039] | -0.0068 [-0.0191, +0.0000] | +0.0050 [-0.0027, +0.0156] | +3.0000 [-1.0000, +7.0000] | +1.0000 [-4.0000, +6.0000] | -0.0042 [-0.0246, +0.0126] |
| s0|b3|B - frozen Ordered|calibration | -0.0047 [-0.0127, +0.0022] | +0.0000 [+0.0000, +0.0000] | +0.0050 [-0.0027, +0.0156] | +5.0000 [+1.0000, +9.0000] | +0.0000 [-5.0000, +5.0000] | -0.0042 [-0.0246, +0.0126] |
| s0|b3|F - frozen Ordered|calibration | -0.0038 [-0.0123, +0.0039] | +0.0000 [+0.0000, +0.0000] | +0.0050 [-0.0027, +0.0156] | +3.0000 [-2.0000, +8.0000] | +1.0000 [-4.0000, +6.0000] | -0.0042 [-0.0246, +0.0126] |
| s0|argmax|O - B|calibration | +0.0000 [+0.0000, +0.0000] | -0.0068 [-0.0191, +0.0000] | +0.0025 [+0.0000, +0.0078] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0127 [-0.0338, +0.0000] |
| s0|argmax|B - F|calibration | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0025 [-0.0078, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0084 [+0.0000, +0.0279] |
| s0|argmax|O - F|calibration | +0.0000 [+0.0000, +0.0000] | -0.0068 [-0.0191, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0042 [-0.0138, +0.0000] |
| s0|argmax|O - frozen Ordered|calibration | -0.0010 [-0.0122, +0.0098] | +0.0000 [-0.0171, +0.0204] | +0.0025 [+0.0000, +0.0080] | +0.0000 [-9.0000, +8.0000] | +1.0000 [-3.0000, +6.0000] | +0.0253 [+0.0000, +0.0648] |
| s0|argmax|B - frozen Ordered|calibration | -0.0010 [-0.0122, +0.0098] | +0.0068 [+0.0000, +0.0226] | +0.0000 [-0.0073, +0.0076] | +0.0000 [-9.0000, +8.0000] | +1.0000 [-3.0000, +6.0000] | +0.0380 [+0.0053, +0.0830] |
| s0|argmax|F - frozen Ordered|calibration | -0.0010 [-0.0122, +0.0098] | +0.0068 [+0.0000, +0.0226] | +0.0025 [+0.0000, +0.0080] | +0.0000 [-9.0000, +8.0000] | +1.0000 [-3.0000, +6.0000] | +0.0295 [+0.0000, +0.0711] |
| s0|ordered_setting|O - B|calibration | +0.0000 [+0.0000, +0.0000] | -0.0068 [-0.0191, +0.0000] | +0.0025 [+0.0000, +0.0078] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| s0|ordered_setting|B - F|calibration | -0.0047 [-0.0139, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0025 [-0.0078, +0.0000] | +4.0000 [+0.0000, +12.0000] | +1.0000 [+0.0000, +3.0000] | +0.0000 [+0.0000, +0.0000] |
| s0|ordered_setting|O - F|calibration | -0.0047 [-0.0139, +0.0000] | -0.0068 [-0.0191, +0.0000] | +0.0000 [+0.0000, +0.0000] | +4.0000 [+0.0000, +12.0000] | +1.0000 [+0.0000, +3.0000] | +0.0000 [+0.0000, +0.0000] |
| s0|ordered_setting|O - frozen Ordered|calibration | -0.0103 [-0.0231, +0.0001] | -0.0068 [-0.0191, +0.0000] | +0.0075 [+0.0000, +0.0172] | +10.0000 [+3.0000, +19.0000] | +1.0000 [-4.0000, +7.0000] | -0.0042 [-0.0246, +0.0126] |
| s0|ordered_setting|B - frozen Ordered|calibration | -0.0103 [-0.0231, +0.0001] | +0.0000 [+0.0000, +0.0000] | +0.0050 [-0.0027, +0.0156] | +10.0000 [+3.0000, +19.0000] | +1.0000 [-4.0000, +7.0000] | -0.0042 [-0.0246, +0.0126] |
| s0|ordered_setting|F - frozen Ordered|calibration | -0.0056 [-0.0141, +0.0020] | +0.0000 [+0.0000, +0.0000] | +0.0075 [+0.0000, +0.0172] | +6.0000 [+2.0000, +11.0000] | +0.0000 [-5.0000, +5.0000] | -0.0042 [-0.0246, +0.0126] |
| s1|b3|O - B|calibration | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| s1|b3|B - F|calibration | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| s1|b3|O - F|calibration | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| s1|b3|O - frozen Ordered|calibration | +0.0049 [-0.0028, +0.0130] | +0.0068 [+0.0000, +0.0191] | +0.0075 [+0.0000, +0.0174] | +0.0000 [-6.0000, +5.0000] | -5.0000 [-10.0000, -1.0000] | +0.0000 [-0.0127, +0.0127] |
| s1|b3|B - frozen Ordered|calibration | +0.0049 [-0.0028, +0.0130] | +0.0068 [+0.0000, +0.0191] | +0.0075 [+0.0000, +0.0174] | +0.0000 [-6.0000, +5.0000] | -5.0000 [-10.0000, -1.0000] | +0.0000 [-0.0127, +0.0127] |
| s1|b3|F - frozen Ordered|calibration | +0.0049 [-0.0028, +0.0130] | +0.0068 [+0.0000, +0.0191] | +0.0075 [+0.0000, +0.0174] | +0.0000 [-6.0000, +5.0000] | -5.0000 [-10.0000, -1.0000] | +0.0000 [-0.0127, +0.0127] |
| s1|argmax|O - B|calibration | +0.0019 [+0.0000, +0.0059] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -1.0000 [-3.0000, +0.0000] | -1.0000 [-3.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| s1|argmax|B - F|calibration | +0.0000 [-0.0057, +0.0056] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [-3.0000, +3.0000] | +0.0000 [-3.0000, +3.0000] | +0.0000 [+0.0000, +0.0000] |
| s1|argmax|O - F|calibration | +0.0019 [+0.0000, +0.0059] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -1.0000 [-3.0000, +0.0000] | -1.0000 [-3.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| s1|argmax|O - frozen Ordered|calibration | +0.0160 [+0.0056, +0.0293] | +0.0134 [+0.0000, +0.0293] | +0.0050 [+0.0000, +0.0129] | -11.0000 [-21.0250, -3.0000] | -6.0000 [-11.0000, -2.0000] | -0.0253 [-0.0762, +0.0168] |
| s1|argmax|B - frozen Ordered|calibration | +0.0141 [+0.0040, +0.0270] | +0.0134 [+0.0000, +0.0293] | +0.0050 [+0.0000, +0.0129] | -10.0000 [-20.0000, -3.0000] | -5.0000 [-10.0000, -1.0000] | -0.0253 [-0.0762, +0.0168] |
| s1|argmax|F - frozen Ordered|calibration | +0.0141 [+0.0028, +0.0280] | +0.0134 [+0.0000, +0.0293] | +0.0050 [+0.0000, +0.0129] | -10.0000 [-21.0000, -2.0000] | -5.0000 [-10.0000, +0.0000] | -0.0253 [-0.0762, +0.0168] |
| s1|ordered_setting|O - B|calibration | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| s1|ordered_setting|B - F|calibration | +0.0010 [+0.0000, +0.0030] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -1.0000 [-3.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| s1|ordered_setting|O - F|calibration | +0.0010 [+0.0000, +0.0030] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -1.0000 [-3.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| s1|ordered_setting|O - frozen Ordered|calibration | +0.0049 [-0.0028, +0.0130] | +0.0068 [+0.0000, +0.0191] | +0.0075 [+0.0000, +0.0174] | +0.0000 [-6.0000, +5.0000] | -5.0000 [-10.0000, -1.0000] | +0.0000 [-0.0127, +0.0127] |
| s1|ordered_setting|B - frozen Ordered|calibration | +0.0049 [-0.0028, +0.0130] | +0.0068 [+0.0000, +0.0191] | +0.0075 [+0.0000, +0.0174] | +0.0000 [-6.0000, +5.0000] | -5.0000 [-10.0000, -1.0000] | +0.0000 [-0.0127, +0.0127] |
| s1|ordered_setting|F - frozen Ordered|calibration | +0.0039 [-0.0045, +0.0127] | +0.0068 [+0.0000, +0.0191] | +0.0075 [+0.0000, +0.0174] | +0.0000 [-6.0000, +5.0000] | -4.0000 [-9.0000, +0.0000] | +0.0000 [-0.0127, +0.0127] |
| s2|b3|O - B|calibration | -0.0019 [-0.0059, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +1.0000 [+0.0000, +3.0000] | +1.0000 [+0.0000, +3.0000] | +0.0084 [+0.0000, +0.0275] |
| s2|b3|B - F|calibration | +0.0010 [+0.0000, +0.0031] | +0.0000 [+0.0000, +0.0000] | +0.0025 [+0.0000, +0.0080] | +0.0000 [+0.0000, +0.0000] | -1.0000 [-3.0000, +0.0000] | -0.0042 [-0.0246, +0.0126] |
| s2|b3|O - F|calibration | -0.0009 [-0.0058, +0.0029] | +0.0000 [+0.0000, +0.0000] | +0.0025 [+0.0000, +0.0080] | +1.0000 [+0.0000, +3.0000] | +0.0000 [-3.0000, +3.0000] | +0.0042 [+0.0000, +0.0146] |
| s2|b3|O - frozen Ordered|calibration | -0.0036 [-0.0122, +0.0042] | +0.0000 [+0.0000, +0.0000] | +0.0151 [+0.0049, +0.0279] | +8.0000 [+2.0000, +15.0000] | -4.0000 [-10.0000, +1.0000] | +0.0549 [+0.0040, +0.1391] |
| s2|b3|B - frozen Ordered|calibration | -0.0017 [-0.0093, +0.0058] | +0.0000 [+0.0000, +0.0000] | +0.0151 [+0.0049, +0.0279] | +7.0000 [+1.0000, +13.0000] | -5.0000 [-10.0250, +0.0000] | +0.0464 [+0.0040, +0.1117] |
| s2|b3|F - frozen Ordered|calibration | -0.0027 [-0.0097, +0.0041] | +0.0000 [+0.0000, +0.0000] | +0.0126 [+0.0025, +0.0247] | +7.0000 [+1.0000, +13.0000] | -4.0000 [-9.0000, +0.0000] | +0.0506 [+0.0000, +0.1336] |
| s2|argmax|O - B|calibration | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0084 [-0.0300, +0.0000] |
| s2|argmax|B - F|calibration | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0169 [+0.0000, +0.0437] |
| s2|argmax|O - F|calibration | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0084 [+0.0000, +0.0275] |
| s2|argmax|O - frozen Ordered|calibration | +0.0104 [-0.0028, +0.0265] | +0.0068 [+0.0000, +0.0191] | +0.0126 [+0.0025, +0.0243] | -6.0000 [-18.0000, +4.0000] | -5.0000 [-11.0000, +1.0000] | +0.0253 [-0.0292, +0.0878] |
| s2|argmax|B - frozen Ordered|calibration | +0.0104 [-0.0028, +0.0265] | +0.0068 [+0.0000, +0.0191] | +0.0126 [+0.0025, +0.0243] | -6.0000 [-18.0000, +4.0000] | -5.0000 [-11.0000, +1.0000] | +0.0338 [-0.0212, +0.0990] |
| s2|argmax|F - frozen Ordered|calibration | +0.0104 [-0.0028, +0.0265] | +0.0068 [+0.0000, +0.0191] | +0.0126 [+0.0025, +0.0243] | -6.0000 [-18.0000, +4.0000] | -5.0000 [-11.0000, +1.0000] | +0.0169 [-0.0333, +0.0742] |
| s2|ordered_setting|O - B|calibration | -0.0019 [-0.0059, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +1.0000 [+0.0000, +3.0000] | +1.0000 [+0.0000, +3.0000] | +0.0000 [+0.0000, +0.0000] |
| s2|ordered_setting|B - F|calibration | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0042 [+0.0000, +0.0146] |
| s2|ordered_setting|O - F|calibration | -0.0019 [-0.0059, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +1.0000 [+0.0000, +3.0000] | +1.0000 [+0.0000, +3.0000] | +0.0042 [+0.0000, +0.0146] |
| s2|ordered_setting|O - frozen Ordered|calibration | -0.0018 [-0.0101, +0.0053] | -0.0073 [-0.0267, +0.0000] | +0.0126 [+0.0025, +0.0252] | +5.0000 [+0.0000, +10.0000] | -3.0000 [-9.0000, +2.0000] | +0.0084 [+0.0000, +0.0216] |
| s2|ordered_setting|B - frozen Ordered|calibration | +0.0002 [-0.0067, +0.0068] | -0.0073 [-0.0267, +0.0000] | +0.0126 [+0.0025, +0.0252] | +4.0000 [+0.0000, +9.0000] | -4.0000 [-9.0000, +1.0000] | +0.0084 [+0.0000, +0.0216] |
| s2|ordered_setting|F - frozen Ordered|calibration | +0.0002 [-0.0067, +0.0068] | -0.0073 [-0.0267, +0.0000] | +0.0126 [+0.0025, +0.0252] | +4.0000 [+0.0000, +9.0000] | -4.0000 [-9.0000, +1.0000] | +0.0042 [-0.0096, +0.0188] |

## Three-seed verdicts

| decoding | comparison | metric | selection | calibration |
|---|---|---|
| b3|O - B|multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| b3|O - B|native macro-F1 | no consistent difference detected | no consistent difference detected |
| b3|O - B|multi boundary recall | no consistent difference detected | no consistent difference detected |
| b3|O - B|multi FP segments | no consistent difference detected | no consistent difference detected |
| b3|O - B|multi FN segments | no consistent difference detected | no consistent difference detected |
| b3|O - B|native cuts per clip | no consistent difference detected | no consistent difference detected |
| b3|B - F|multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| b3|B - F|native macro-F1 | no consistent difference detected | no consistent difference detected |
| b3|B - F|multi boundary recall | no consistent difference detected | no consistent difference detected |
| b3|B - F|multi FP segments | no consistent difference detected | no consistent difference detected |
| b3|B - F|multi FN segments | no consistent difference detected | no consistent difference detected |
| b3|B - F|native cuts per clip | no consistent difference detected | no consistent difference detected |
| b3|O - F|multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| b3|O - F|native macro-F1 | no consistent difference detected | no consistent difference detected |
| b3|O - F|multi boundary recall | no consistent difference detected | no consistent difference detected |
| b3|O - F|multi FP segments | no consistent difference detected | no consistent difference detected |
| b3|O - F|multi FN segments | no consistent difference detected | no consistent difference detected |
| b3|O - F|native cuts per clip | no consistent difference detected | no consistent difference detected |
| b3|O - frozen Ordered|multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| b3|O - frozen Ordered|native macro-F1 | no consistent difference detected | no consistent difference detected |
| b3|O - frozen Ordered|multi boundary recall | no consistent difference detected | no consistent difference detected |
| b3|O - frozen Ordered|multi FP segments | no consistent difference detected | no consistent difference detected |
| b3|O - frozen Ordered|multi FN segments | no consistent difference detected | no consistent difference detected |
| b3|O - frozen Ordered|native cuts per clip | no consistent difference detected | no consistent difference detected |
| b3|B - frozen Ordered|multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| b3|B - frozen Ordered|native macro-F1 | no consistent difference detected | no consistent difference detected |
| b3|B - frozen Ordered|multi boundary recall | no consistent difference detected | no consistent difference detected |
| b3|B - frozen Ordered|multi FP segments | no consistent difference detected | no consistent difference detected |
| b3|B - frozen Ordered|multi FN segments | no consistent difference detected | no consistent difference detected |
| b3|B - frozen Ordered|native cuts per clip | no consistent difference detected | no consistent difference detected |
| b3|F - frozen Ordered|multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| b3|F - frozen Ordered|native macro-F1 | no consistent difference detected | no consistent difference detected |
| b3|F - frozen Ordered|multi boundary recall | no consistent difference detected | no consistent difference detected |
| b3|F - frozen Ordered|multi FP segments | no consistent difference detected | no consistent difference detected |
| b3|F - frozen Ordered|multi FN segments | no consistent difference detected | no consistent difference detected |
| b3|F - frozen Ordered|native cuts per clip | no consistent difference detected | no consistent difference detected |
| argmax|O - B|multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| argmax|O - B|native macro-F1 | no consistent difference detected | no consistent difference detected |
| argmax|O - B|multi boundary recall | no consistent difference detected | no consistent difference detected |
| argmax|O - B|multi FP segments | no consistent difference detected | no consistent difference detected |
| argmax|O - B|multi FN segments | no consistent difference detected | no consistent difference detected |
| argmax|O - B|native cuts per clip | no consistent difference detected | no consistent difference detected |
| argmax|B - F|multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| argmax|B - F|native macro-F1 | no consistent difference detected | no consistent difference detected |
| argmax|B - F|multi boundary recall | no consistent difference detected | no consistent difference detected |
| argmax|B - F|multi FP segments | no consistent difference detected | no consistent difference detected |
| argmax|B - F|multi FN segments | no consistent difference detected | no consistent difference detected |
| argmax|B - F|native cuts per clip | no consistent difference detected | no consistent difference detected |
| argmax|O - F|multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| argmax|O - F|native macro-F1 | no consistent difference detected | no consistent difference detected |
| argmax|O - F|multi boundary recall | no consistent difference detected | no consistent difference detected |
| argmax|O - F|multi FP segments | no consistent difference detected | no consistent difference detected |
| argmax|O - F|multi FN segments | no consistent difference detected | no consistent difference detected |
| argmax|O - F|native cuts per clip | no consistent difference detected | no consistent difference detected |
| argmax|O - frozen Ordered|multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| argmax|O - frozen Ordered|native macro-F1 | no consistent difference detected | no consistent difference detected |
| argmax|O - frozen Ordered|multi boundary recall | no consistent difference detected | no consistent difference detected |
| argmax|O - frozen Ordered|multi FP segments | no consistent difference detected | no consistent difference detected |
| argmax|O - frozen Ordered|multi FN segments | no consistent difference detected | no consistent difference detected |
| argmax|O - frozen Ordered|native cuts per clip | no consistent difference detected | no consistent difference detected |
| argmax|B - frozen Ordered|multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| argmax|B - frozen Ordered|native macro-F1 | no consistent difference detected | no consistent difference detected |
| argmax|B - frozen Ordered|multi boundary recall | no consistent difference detected | no consistent difference detected |
| argmax|B - frozen Ordered|multi FP segments | no consistent difference detected | no consistent difference detected |
| argmax|B - frozen Ordered|multi FN segments | no consistent difference detected | no consistent difference detected |
| argmax|B - frozen Ordered|native cuts per clip | no consistent difference detected | no consistent difference detected |
| argmax|F - frozen Ordered|multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| argmax|F - frozen Ordered|native macro-F1 | no consistent difference detected | no consistent difference detected |
| argmax|F - frozen Ordered|multi boundary recall | no consistent difference detected | no consistent difference detected |
| argmax|F - frozen Ordered|multi FP segments | no consistent difference detected | no consistent difference detected |
| argmax|F - frozen Ordered|multi FN segments | no consistent difference detected | no consistent difference detected |
| argmax|F - frozen Ordered|native cuts per clip | no consistent difference detected | no consistent difference detected |
| ordered_setting|O - B|multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| ordered_setting|O - B|native macro-F1 | no consistent difference detected | no consistent difference detected |
| ordered_setting|O - B|multi boundary recall | no consistent difference detected | no consistent difference detected |
| ordered_setting|O - B|multi FP segments | no consistent difference detected | no consistent difference detected |
| ordered_setting|O - B|multi FN segments | no consistent difference detected | no consistent difference detected |
| ordered_setting|O - B|native cuts per clip | no consistent difference detected | no consistent difference detected |
| ordered_setting|B - F|multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| ordered_setting|B - F|native macro-F1 | no consistent difference detected | no consistent difference detected |
| ordered_setting|B - F|multi boundary recall | no consistent difference detected | no consistent difference detected |
| ordered_setting|B - F|multi FP segments | no consistent difference detected | no consistent difference detected |
| ordered_setting|B - F|multi FN segments | no consistent difference detected | no consistent difference detected |
| ordered_setting|B - F|native cuts per clip | no consistent difference detected | no consistent difference detected |
| ordered_setting|O - F|multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| ordered_setting|O - F|native macro-F1 | no consistent difference detected | no consistent difference detected |
| ordered_setting|O - F|multi boundary recall | no consistent difference detected | no consistent difference detected |
| ordered_setting|O - F|multi FP segments | no consistent difference detected | no consistent difference detected |
| ordered_setting|O - F|multi FN segments | no consistent difference detected | no consistent difference detected |
| ordered_setting|O - F|native cuts per clip | no consistent difference detected | no consistent difference detected |
| ordered_setting|O - frozen Ordered|multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| ordered_setting|O - frozen Ordered|native macro-F1 | no consistent difference detected | no consistent difference detected |
| ordered_setting|O - frozen Ordered|multi boundary recall | no consistent difference detected | no consistent difference detected |
| ordered_setting|O - frozen Ordered|multi FP segments | no consistent difference detected | no consistent difference detected |
| ordered_setting|O - frozen Ordered|multi FN segments | no consistent difference detected | no consistent difference detected |
| ordered_setting|O - frozen Ordered|native cuts per clip | no consistent difference detected | no consistent difference detected |
| ordered_setting|B - frozen Ordered|multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| ordered_setting|B - frozen Ordered|native macro-F1 | no consistent difference detected | no consistent difference detected |
| ordered_setting|B - frozen Ordered|multi boundary recall | no consistent difference detected | no consistent difference detected |
| ordered_setting|B - frozen Ordered|multi FP segments | no consistent difference detected | no consistent difference detected |
| ordered_setting|B - frozen Ordered|multi FN segments | no consistent difference detected | no consistent difference detected |
| ordered_setting|B - frozen Ordered|native cuts per clip | no consistent difference detected | no consistent difference detected |
| ordered_setting|F - frozen Ordered|multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| ordered_setting|F - frozen Ordered|native macro-F1 | no consistent difference detected | no consistent difference detected |
| ordered_setting|F - frozen Ordered|multi boundary recall | no consistent difference detected | no consistent difference detected |
| ordered_setting|F - frozen Ordered|multi FP segments | no consistent difference detected | no consistent difference detected |
| ordered_setting|F - frozen Ordered|multi FN segments | no consistent difference detected | no consistent difference detected |
| ordered_setting|F - frozen Ordered|native cuts per clip | no consistent difference detected | no consistent difference detected |

## Pooled recovered and lost reference segments (B3 systems)

| pair | half | population | recovered | lost | new_unmatched_pred | removed_unmatched_pred | recovered_short | lost_short | recovered_xyx | lost_xyx |
|---|---|---|---|---|---|---|---|---|
| O - B|selection|native | 1 | 1 | 5 | 5 | 0 | 0 | 0 | 0 |
| O - B|selection|pairs | 0 | 1 | 3 | 3 | 0 | 1 | 0 | 0 |
| O - B|selection|multi | 2 | 0 | 7 | 11 | 1 | 0 | 0 | 0 |
| O - B|calibration|native | 0 | 1 | 4 | 1 | 0 | 0 | 0 | 0 |
| O - B|calibration|pairs | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| O - B|calibration|multi | 0 | 2 | 3 | 4 | 0 | 1 | 0 | 0 |
| B - F|selection|native | 2 | 1 | 3 | 7 | 0 | 0 | 0 | 0 |
| B - F|selection|pairs | 1 | 0 | 3 | 3 | 1 | 0 | 0 | 0 |
| B - F|selection|multi | 3 | 2 | 12 | 12 | 1 | 1 | 0 | 0 |
| B - F|calibration|native | 0 | 0 | 2 | 3 | 0 | 0 | 0 | 0 |
| B - F|calibration|pairs | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| B - F|calibration|multi | 2 | 0 | 6 | 4 | 1 | 0 | 0 | 0 |
| O - F|selection|native | 1 | 0 | 2 | 6 | 0 | 0 | 0 | 0 |
| O - F|selection|pairs | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| O - F|selection|multi | 4 | 1 | 8 | 12 | 1 | 0 | 0 | 0 |
| O - F|calibration|native | 0 | 1 | 4 | 2 | 0 | 0 | 0 | 0 |
| O - F|calibration|pairs | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 |
| O - F|calibration|multi | 1 | 1 | 4 | 3 | 1 | 1 | 0 | 0 |
| O - frozen Ordered|selection|native | 1 | 2 | 23 | 10 | 0 | 0 | 0 | 0 |
| O - frozen Ordered|selection|pairs | 5 | 1 | 9 | 4 | 5 | 1 | 0 | 0 |
| O - frozen Ordered|selection|multi | 26 | 13 | 91 | 38 | 14 | 0 | 2 | 0 |
| O - frozen Ordered|calibration|native | 1 | 2 | 19 | 6 | 0 | 0 | 0 | 0 |
| O - frozen Ordered|calibration|pairs | 9 | 2 | 5 | 4 | 7 | 2 | 0 | 0 |
| O - frozen Ordered|calibration|multi | 14 | 6 | 23 | 12 | 12 | 5 | 0 | 0 |
| B - frozen Ordered|selection|native | 2 | 3 | 25 | 12 | 0 | 0 | 0 | 0 |
| B - frozen Ordered|selection|pairs | 5 | 0 | 10 | 5 | 5 | 0 | 0 | 0 |
| B - frozen Ordered|selection|multi | 26 | 15 | 95 | 38 | 14 | 1 | 2 | 0 |
| B - frozen Ordered|calibration|native | 1 | 1 | 16 | 6 | 0 | 0 | 0 | 0 |
| B - frozen Ordered|calibration|pairs | 10 | 2 | 5 | 4 | 8 | 2 | 0 | 0 |
| B - frozen Ordered|calibration|multi | 14 | 4 | 22 | 10 | 12 | 4 | 0 | 0 |
| F - frozen Ordered|selection|native | 1 | 3 | 26 | 9 | 0 | 0 | 0 | 0 |
| F - frozen Ordered|selection|pairs | 5 | 1 | 9 | 4 | 5 | 1 | 0 | 0 |
| F - frozen Ordered|selection|multi | 26 | 16 | 95 | 38 | 14 | 1 | 2 | 0 |
| F - frozen Ordered|calibration|native | 1 | 1 | 17 | 6 | 0 | 0 | 0 | 0 |
| F - frozen Ordered|calibration|pairs | 10 | 2 | 4 | 4 | 8 | 2 | 0 | 0 |
| F - frozen Ordered|calibration|multi | 13 | 5 | 23 | 13 | 11 | 4 | 0 | 0 |

## Goal

| group | selection mean | calibration mean | constraint failures | met |
|---|---|---|---|---|
| F | 0.9404 | 0.9812 | 0 | False |
| B | 0.9407 | 0.9812 | 0 | False |
| O | 0.9426 | 0.9809 | 0 | False |
| frozen Ordered | 0.9544 | 0.9817 | 1 | False |

## Archived rows (listed as stored)

| system | half | segment_f1_50_multi | boundary_f1_multi | boundary_recall_multi | native_macro_f1 | native_internal_cuts_per_clip | same_class_cut_rate_multi |
|---|---|---|---|---|---|---|---|
| A01 ASRF s0 | selection | 0.9366 | 0.9615 | 0.9476 | 0.9869 | 0.0141 | 0.0348 |
| A01 ASRF s0 | calibration | 0.9796 | 0.9809 | 0.9698 | 0.9780 | 0.0084 | 0.0149 |
| A01 ASRF s1 | selection | 0.9476 | 0.9628 | 0.9476 | 0.9771 | 0.0035 | 0.0304 |
| A01 ASRF s1 | calibration | 0.9747 | 0.9822 | 0.9724 | 0.9723 | 0.0084 | 0.0100 |
| A01 ASRF s2 | selection | 0.9506 | 0.9614 | 0.9450 | 0.9869 | 0.0000 | 0.0304 |
| A01 ASRF s2 | calibration | 0.9698 | 0.9707 | 0.9573 | 0.9929 | 0.0084 | 0.0149 |
| A01 Linear + B3 s0 | selection | 0.9473 | 0.9542 | 0.9555 | 0.9786 | 0.0458 | 0.0261 |
| A01 Linear + B3 s0 | calibration | 0.9768 | 0.9749 | 0.9749 | 0.9861 | 0.0127 | 0.0199 |
| A01 Linear + B3 s1 | selection | 0.9569 | 0.9617 | 0.9529 | 0.9786 | 0.0387 | 0.0087 |
| A01 Linear + B3 s1 | calibration | 0.9779 | 0.9762 | 0.9799 | 0.9861 | 0.0169 | 0.0149 |
| A01 Linear + B3 s2 | selection | 0.9388 | 0.9404 | 0.9712 | 0.9786 | 0.0845 | 0.0304 |
| A01 Linear + B3 s2 | calibration | 0.9732 | 0.9627 | 0.9724 | 0.9861 | 0.0422 | 0.0299 |
| A05 F + B3 s0 | selection | 0.9473 | 0.9542 | 0.9555 | 0.9786 | 0.0458 | 0.0261 |
| A05 F + B3 s0 | calibration | 0.9768 | 0.9749 | 0.9749 | 0.9861 | 0.0127 | 0.0199 |
| A05 F + B3 s1 | selection | 0.9569 | 0.9617 | 0.9529 | 0.9786 | 0.0387 | 0.0087 |
| A05 F + B3 s1 | calibration | 0.9779 | 0.9762 | 0.9799 | 0.9861 | 0.0169 | 0.0149 |
| A05 F + B3 s2 | selection | 0.9388 | 0.9404 | 0.9712 | 0.9786 | 0.0845 | 0.0304 |
| A05 F + B3 s2 | calibration | 0.9732 | 0.9627 | 0.9724 | 0.9861 | 0.0422 | 0.0299 |
| A05 B + B3 s0 | selection | 0.9481 | 0.9554 | 0.9529 | 0.9786 | 0.0563 | 0.0261 |
| A05 B + B3 s0 | calibration | 0.9768 | 0.9774 | 0.9774 | 0.9861 | 0.0127 | 0.0199 |
| A05 B + B3 s1 | selection | 0.9550 | 0.9592 | 0.9529 | 0.9701 | 0.0352 | 0.0130 |
| A05 B + B3 s1 | calibration | 0.9789 | 0.9776 | 0.9849 | 0.9861 | 0.0169 | 0.0249 |
| A05 B + B3 s2 | selection | 0.9620 | 0.9658 | 0.9607 | 0.9700 | 0.0387 | 0.0130 |
| A05 B + B3 s2 | calibration | 0.9806 | 0.9747 | 0.9698 | 0.9929 | 0.0169 | 0.0149 |
| A05 O + B3 s0 | selection | 0.9512 | 0.9542 | 0.9555 | 0.9701 | 0.0563 | 0.0217 |
| A05 O + B3 s0 | calibration | 0.9778 | 0.9787 | 0.9799 | 0.9861 | 0.0127 | 0.0199 |
| A05 O + B3 s1 | selection | 0.9569 | 0.9617 | 0.9529 | 0.9701 | 0.0282 | 0.0087 |
| A05 O + B3 s1 | calibration | 0.9788 | 0.9774 | 0.9774 | 0.9861 | 0.0127 | 0.0149 |
| A05 O + B3 s2 | selection | 0.9610 | 0.9671 | 0.9607 | 0.9786 | 0.0458 | 0.0130 |
| A05 O + B3 s2 | calibration | 0.9816 | 0.9760 | 0.9724 | 0.9861 | 0.0169 | 0.0149 |
| A06-R1 FACT s0 | selection | 0.9610 | 0.9619 | 0.9581 | 0.9656 | 0.0000 | 0.0043 |
| A06-R1 FACT s0 | calibration | 0.9620 | 0.9657 | 0.9548 | 0.9780 | 0.0084 | 0.0199 |
| A06-R1 FACT s1 | selection | 0.9570 | 0.9514 | 0.9476 | 0.9869 | 0.0035 | 0.0000 |
| A06-R1 FACT s1 | calibration | 0.9584 | 0.9710 | 0.9673 | 0.9856 | 0.0042 | 0.0149 |
| A06-R1 FACT s2 | selection | 0.9489 | 0.9486 | 0.9424 | 0.9976 | 0.0000 | 0.0043 |
| A06-R1 FACT s2 | calibration | 0.9611 | 0.9645 | 0.9548 | 0.9927 | 0.0084 | 0.0199 |

## Readings

```
[
 "no_qualifying_gain",
 "end"
]
```
