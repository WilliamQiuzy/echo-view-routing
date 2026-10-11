# Raw-video context round: tables

Recompute: `python scripts/report_raw_video_context.py --config configs/experiments/raw_video_context_v1.yaml --evaluate`

## Native window recognition (same centres)

| system | half | A4C/A5C balanced accuracy | A5C-A4C margin AUC | native window accuracy | A4C recall | A5C recall | windows |
|---|---|---|---|---|---|---|---|
| target_only_s0 | selection | 0.8743 | 0.9245 | 0.9588 | 0.9368 | 0.8118 | 3444 |
| target_only_s0 | calibration | 0.8435 | 0.9264 | 0.9519 | 0.8761 | 0.8109 | 2950 |
| shuffled_s0 | selection | 0.9416 | 0.9893 | 0.9817 | 0.9656 | 0.9176 | 3444 |
| shuffled_s0 | calibration | 0.9507 | 0.9792 | 0.9881 | 0.9611 | 0.9403 | 2950 |
| ordered_s0 | selection | 0.9303 | 0.9878 | 0.9811 | 0.9665 | 0.8941 | 3444 |
| ordered_s0 | calibration | 0.9406 | 0.9794 | 0.9864 | 0.9558 | 0.9254 | 2950 |
| target_only_s1 | selection | 0.8642 | 0.9226 | 0.9570 | 0.9284 | 0.8000 | 3444 |
| target_only_s1 | calibration | 0.8395 | 0.9254 | 0.9410 | 0.8531 | 0.8259 | 2950 |
| shuffled_s1 | selection | 0.9296 | 0.9913 | 0.9849 | 0.9768 | 0.8824 | 3444 |
| shuffled_s1 | calibration | 0.9356 | 0.9831 | 0.9847 | 0.9558 | 0.9154 | 2950 |
| ordered_s1 | selection | 0.9348 | 0.9886 | 0.9808 | 0.9638 | 0.9059 | 3444 |
| ordered_s1 | calibration | 0.9381 | 0.9828 | 0.9861 | 0.9558 | 0.9204 | 2950 |
| target_only_s2 | selection | 0.8833 | 0.9252 | 0.9593 | 0.9312 | 0.8353 | 3444 |
| target_only_s2 | calibration | 0.8496 | 0.9288 | 0.9485 | 0.8832 | 0.8159 | 2950 |
| shuffled_s2 | selection | 0.9271 | 0.9884 | 0.9794 | 0.9600 | 0.8941 | 3444 |
| shuffled_s2 | calibration | 0.9205 | 0.9768 | 0.9827 | 0.9504 | 0.8905 | 2950 |
| ordered_s2 | selection | 0.9370 | 0.9897 | 0.9782 | 0.9563 | 0.9176 | 3444 |
| ordered_s2 | calibration | 0.9189 | 0.9779 | 0.9827 | 0.9522 | 0.8856 | 2950 |
| E1 (same requested samples) | selection | 0.8311 | 0.9134 | 0.9489 | 0.9210 | 0.7412 | 3444 |
| E1 (same requested samples) | calibration | 0.8168 | 0.8835 | 0.9400 | 0.8973 | 0.7363 | 2950 |

## Recognition comparisons (session-paired bootstrap)

| comparison | A4C/A5C balanced accuracy | A5C-A4C margin AUC | native window accuracy |
|---|---|---|---|
| s0|Ordered - Shuffled|selection | -0.0113 [-0.0284, +0.0021] | -0.0016 [-0.0060, +0.0017] | -0.0006 [-0.0034, +0.0027] |
| s0|Ordered - Target-only|selection | +0.0560 [+0.0243, +0.0828] | +0.0633 [+0.0116, +0.1214] | +0.0224 [+0.0152, +0.0294] |
| s0|Shuffled - Target-only|selection | +0.0673 [+0.0281, +0.1049] | +0.0648 [+0.0135, +0.1209] | +0.0229 [+0.0164, +0.0291] |
| s0|Target-only - E1|selection | +0.0432 [-0.0204, +0.0933] | +0.0111 [-0.0457, +0.0623] | +0.0099 [-0.0045, +0.0226] |
| s0|Shuffled context - E1|selection | +0.1105 [+0.0303, +0.1813] | +0.0759 [+0.0127, +0.1241] | +0.0328 [+0.0172, +0.0472] |
| s0|Ordered context - E1|selection | +0.0992 [+0.0206, +0.1656] | +0.0744 [+0.0121, +0.1195] | +0.0322 [+0.0172, +0.0454] |
| s1|Ordered - Shuffled|selection | +0.0053 [-0.0254, +0.0527] | -0.0027 [-0.0111, +0.0055] | -0.0041 [-0.0118, +0.0046] |
| s1|Ordered - Target-only|selection | +0.0706 [+0.0276, +0.1123] | +0.0660 [+0.0109, +0.1303] | +0.0238 [+0.0158, +0.0341] |
| s1|Shuffled - Target-only|selection | +0.0653 [+0.0172, +0.0974] | +0.0688 [+0.0116, +0.1305] | +0.0279 [+0.0185, +0.0367] |
| s1|Target-only - E1|selection | +0.0331 [-0.0385, +0.0812] | +0.0092 [-0.0565, +0.0696] | +0.0081 [-0.0064, +0.0217] |
| s1|Shuffled context - E1|selection | +0.0985 [+0.0012, +0.1743] | +0.0780 [+0.0125, +0.1298] | +0.0360 [+0.0168, +0.0539] |
| s1|Ordered context - E1|selection | +0.1037 [+0.0025, +0.1873] | +0.0752 [+0.0124, +0.1235] | +0.0319 [+0.0176, +0.0444] |
| s2|Ordered - Shuffled|selection | +0.0099 [-0.0202, +0.0361] | +0.0013 [-0.0037, +0.0079] | -0.0012 [-0.0067, +0.0053] |
| s2|Ordered - Target-only|selection | +0.0537 [+0.0178, +0.0953] | +0.0645 [+0.0079, +0.1348] | +0.0189 [+0.0104, +0.0294] |
| s2|Shuffled - Target-only|selection | +0.0438 [+0.0106, +0.0853] | +0.0632 [+0.0070, +0.1302] | +0.0200 [+0.0130, +0.0281] |
| s2|Target-only - E1|selection | +0.0522 [-0.0256, +0.1282] | +0.0118 [-0.0519, +0.0719] | +0.0105 [-0.0068, +0.0263] |
| s2|Shuffled context - E1|selection | +0.0960 [+0.0198, +0.1560] | +0.0750 [+0.0126, +0.1223] | +0.0305 [+0.0137, +0.0450] |
| s2|Ordered context - E1|selection | +0.1059 [+0.0181, +0.1732] | +0.0763 [+0.0125, +0.1234] | +0.0293 [+0.0166, +0.0405] |
| s0|Ordered - Shuffled|calibration | -0.0101 [-0.0205, +0.0009] | +0.0002 [-0.0080, +0.0131] | -0.0017 [-0.0047, +0.0010] |
| s0|Ordered - Target-only|calibration | +0.0970 [+0.0449, +0.1480] | +0.0530 [+0.0134, +0.1041] | +0.0346 [+0.0216, +0.0510] |
| s0|Shuffled - Target-only|calibration | +0.1072 [+0.0484, +0.1670] | +0.0528 [+0.0137, +0.1064] | +0.0363 [+0.0220, +0.0545] |
| s0|Target-only - E1|calibration | +0.0267 [-0.0050, +0.0580] | +0.0429 [+0.0033, +0.0764] | +0.0119 [-0.0032, +0.0315] |
| s0|Shuffled context - E1|calibration | +0.1338 [+0.0581, +0.2020] | +0.0957 [+0.0241, +0.1668] | +0.0481 [+0.0306, +0.0690] |
| s0|Ordered context - E1|calibration | +0.1237 [+0.0546, +0.1858] | +0.0959 [+0.0244, +0.1634] | +0.0464 [+0.0289, +0.0679] |
| s1|Ordered - Shuffled|calibration | +0.0025 [-0.0052, +0.0120] | -0.0003 [-0.0042, +0.0039] | +0.0014 [-0.0014, +0.0049] |
| s1|Ordered - Target-only|calibration | +0.0986 [+0.0389, +0.1559] | +0.0574 [+0.0151, +0.1104] | +0.0451 [+0.0280, +0.0646] |
| s1|Shuffled - Target-only|calibration | +0.0961 [+0.0354, +0.1556] | +0.0576 [+0.0147, +0.1114] | +0.0437 [+0.0256, +0.0647] |
| s1|Target-only - E1|calibration | +0.0227 [-0.0031, +0.0458] | +0.0419 [+0.0059, +0.0697] | +0.0010 [-0.0131, +0.0167] |
| s1|Shuffled context - E1|calibration | +0.1188 [+0.0500, +0.1797] | +0.0995 [+0.0251, +0.1667] | +0.0447 [+0.0272, +0.0649] |
| s1|Ordered context - E1|calibration | +0.1212 [+0.0547, +0.1810] | +0.0992 [+0.0254, +0.1675] | +0.0461 [+0.0285, +0.0676] |
| s2|Ordered - Shuffled|calibration | -0.0016 [-0.0112, +0.0059] | +0.0011 [-0.0040, +0.0087] | +0.0000 [-0.0042, +0.0030] |
| s2|Ordered - Target-only|calibration | +0.0693 [+0.0379, +0.0990] | +0.0491 [+0.0147, +0.0964] | +0.0342 [+0.0228, +0.0464] |
| s2|Shuffled - Target-only|calibration | +0.0709 [+0.0388, +0.1002] | +0.0480 [+0.0141, +0.0934] | +0.0342 [+0.0246, +0.0454] |
| s2|Target-only - E1|calibration | +0.0327 [-0.0018, +0.0654] | +0.0453 [+0.0006, +0.0835] | +0.0085 [-0.0048, +0.0269] |
| s2|Shuffled context - E1|calibration | +0.1037 [+0.0488, +0.1492] | +0.0933 [+0.0247, +0.1579] | +0.0427 [+0.0267, +0.0638] |
| s2|Ordered context - E1|calibration | +0.1021 [+0.0481, +0.1474] | +0.0944 [+0.0253, +0.1600] | +0.0427 [+0.0258, +0.0636] |

## Systems

| system | half | row | segment F1@50 | boundary P | boundary R | native macro-F1 | switches per clip | same-view join cuts | A5C read as A4C | A4C read as A5C |
|---|---|---|---|---|---|---|---|---|---|---|
| target_only_s0 | selection | argmax | 0.7213 | 0.5962 | 0.9895 | 0.9771 | 0.6655 | 0.1522 | 1 | 1 |
| target_only_s0 | calibration | argmax | 0.7779 | 0.6717 | 0.9975 | 0.9929 | 0.8692 | 0.1343 | 0 | 1 |
| target_only_s0 | selection | b3 | 0.9087 | 0.9643 | 0.9188 | 0.9680 | 0.0423 | 0.0087 | 1 | 2 |
| target_only_s0 | calibration | b3 | 0.9502 | 0.9793 | 0.9523 | 0.9856 | 0.0380 | 0.0149 | 1 | 1 |
| shuffled_s0 | selection | argmax | 0.8881 | 0.8311 | 0.9791 | 0.9786 | 0.1303 | 0.0348 | 0 | 2 |
| shuffled_s0 | calibration | argmax | 0.9484 | 0.9157 | 0.9824 | 0.9861 | 0.1097 | 0.0398 | 0 | 2 |
| shuffled_s0 | selection | b3 | 0.9341 | 0.9266 | 0.9581 | 0.9701 | 0.0458 | 0.0130 | 0 | 3 |
| shuffled_s0 | calibration | b3 | 0.9750 | 0.9629 | 0.9774 | 0.9861 | 0.0295 | 0.0199 | 0 | 2 |
| ordered_s0 | selection | argmax | 0.8929 | 0.8462 | 0.9791 | 0.9701 | 0.0951 | 0.0435 | 0 | 3 |
| ordered_s0 | calibration | argmax | 0.9611 | 0.9424 | 0.9874 | 0.9861 | 0.0928 | 0.0348 | 0 | 2 |
| ordered_s0 | selection | b3 | 0.9385 | 0.9482 | 0.9581 | 0.9786 | 0.0423 | 0.0174 | 0 | 2 |
| ordered_s0 | calibration | b3 | 0.9817 | 0.9751 | 0.9824 | 0.9929 | 0.0253 | 0.0249 | 0 | 1 |
| target_only_s1 | selection | argmax | 0.7488 | 0.6258 | 0.9895 | 0.9680 | 0.7077 | 0.1348 | 1 | 2 |
| target_only_s1 | calibration | argmax | 0.7684 | 0.6606 | 0.9975 | 0.9796 | 1.0506 | 0.1393 | 0 | 3 |
| target_only_s1 | selection | b3 | 0.8307 | 0.7383 | 0.9895 | 0.9517 | 0.4366 | 0.0957 | 1 | 4 |
| target_only_s1 | calibration | b3 | 0.9086 | 0.8458 | 0.9925 | 0.9861 | 0.6624 | 0.0846 | 0 | 2 |
| shuffled_s1 | selection | argmax | 0.8985 | 0.8462 | 0.9791 | 0.9771 | 0.1338 | 0.0565 | 1 | 1 |
| shuffled_s1 | calibration | argmax | 0.9457 | 0.9047 | 0.9774 | 0.9861 | 0.1435 | 0.0448 | 0 | 2 |
| shuffled_s1 | selection | b3 | 0.9324 | 0.9010 | 0.9764 | 0.9771 | 0.0669 | 0.0391 | 1 | 1 |
| shuffled_s1 | calibration | b3 | 0.9724 | 0.9489 | 0.9799 | 0.9861 | 0.0422 | 0.0348 | 0 | 2 |
| ordered_s1 | selection | argmax | 0.9250 | 0.8724 | 0.9843 | 0.9877 | 0.1127 | 0.0304 | 0 | 1 |
| ordered_s1 | calibration | argmax | 0.9611 | 0.9448 | 0.9899 | 0.9796 | 0.1013 | 0.0398 | 0 | 3 |
| ordered_s1 | selection | b3 | 0.9603 | 0.9585 | 0.9686 | 0.9786 | 0.0387 | 0.0043 | 0 | 2 |
| ordered_s1 | calibration | b3 | 0.9836 | 0.9800 | 0.9874 | 0.9861 | 0.0169 | 0.0249 | 0 | 2 |
| target_only_s2 | selection | argmax | 0.7508 | 0.6254 | 0.9921 | 0.9771 | 0.6620 | 0.1522 | 1 | 1 |
| target_only_s2 | calibration | argmax | 0.7997 | 0.6965 | 0.9975 | 0.9861 | 0.9409 | 0.1343 | 0 | 2 |
| target_only_s2 | selection | b3 | 0.8365 | 0.7565 | 0.9843 | 0.9771 | 0.4401 | 0.1087 | 1 | 1 |
| target_only_s2 | calibration | b3 | 0.9063 | 0.8547 | 0.9899 | 0.9861 | 0.5443 | 0.0896 | 0 | 2 |
| shuffled_s2 | selection | argmax | 0.8949 | 0.8423 | 0.9791 | 0.9621 | 0.1479 | 0.0348 | 0 | 4 |
| shuffled_s2 | calibration | argmax | 0.9575 | 0.9264 | 0.9799 | 0.9929 | 0.1224 | 0.0348 | 0 | 1 |
| shuffled_s2 | selection | b3 | 0.9392 | 0.9580 | 0.9555 | 0.9621 | 0.0352 | 0.0087 | 0 | 4 |
| shuffled_s2 | calibration | b3 | 0.9797 | 0.9748 | 0.9724 | 0.9929 | 0.0211 | 0.0199 | 0 | 1 |
| ordered_s2 | selection | argmax | 0.9259 | 0.8744 | 0.9843 | 0.9701 | 0.1162 | 0.0261 | 0 | 3 |
| ordered_s2 | calibration | argmax | 0.9602 | 0.9399 | 0.9824 | 0.9861 | 0.0970 | 0.0348 | 0 | 2 |
| ordered_s2 | selection | b3 | 0.9644 | 0.9588 | 0.9738 | 0.9701 | 0.0458 | 0.0043 | 0 | 3 |
| ordered_s2 | calibration | b3 | 0.9797 | 0.9774 | 0.9799 | 0.9861 | 0.0127 | 0.0249 | 0 | 2 |
| E0 + B3 | selection | row | 0.9344 | 0.9946 | 0.9607 | 0.9650 | 0.0141 | 0.0043 | 3 | 0 |
| E0 + B3 | calibration | row | 0.9340 | 0.9694 | 0.9548 | 0.9660 | 0.0338 | 0.0199 | 4 | 0 |
| E1 + B3 | selection | row | 0.8940 | 0.9432 | 0.9555 | 0.9780 | 0.0458 | 0.0391 | 1 | 0 |
| E1 + B3 | calibration | row | 0.9589 | 0.9511 | 0.9774 | 0.9856 | 0.0675 | 0.0199 | 1 | 1 |
| B7 + B3 | selection | row | 0.8798 | 0.8902 | 0.9555 | 0.9566 | 0.0634 | 0.0565 | 2 | 1 |
| B7 + B3 | calibration | row | 0.9721 | 0.9628 | 0.9749 | 0.9927 | 0.0422 | 0.0348 | 1 | 0 |
| E1 + ASRF s0 | selection | row | 0.9100 | 0.9524 | 0.9424 | 0.9477 | 0.0458 | 0.0304 | 2 | 2 |
| E1 + ASRF s0 | calibration | row | 0.9758 | 0.9798 | 0.9774 | 0.9929 | 0.0380 | 0.0149 | 0 | 1 |
| E1 + ASRF s1 | selection | row | 0.9089 | 0.9356 | 0.9503 | 0.9664 | 0.0070 | 0.0565 | 2 | 0 |
| E1 + ASRF s1 | calibration | row | 0.9643 | 0.9699 | 0.9724 | 0.9753 | 0.0295 | 0.0299 | 2 | 1 |
| E1 + ASRF s2 | selection | row | 0.9071 | 0.9333 | 0.9529 | 0.9664 | 0.0211 | 0.0565 | 2 | 0 |
| E1 + ASRF s2 | calibration | row | 0.9644 | 0.9701 | 0.9774 | 0.9753 | 0.0464 | 0.0348 | 2 | 1 |
| C + B3 s0 | selection | row | 0.8541 | 0.8642 | 0.9660 | 0.9218 | 0.0669 | 0.1000 | 4 | 2 |
| C + B3 s0 | calibration | row | 0.9647 | 0.9586 | 0.9899 | 0.9753 | 0.0717 | 0.0299 | 2 | 1 |
| C + B3 s1 | selection | row | 0.8566 | 0.8585 | 0.9686 | 0.9477 | 0.0915 | 0.0913 | 2 | 2 |
| C + B3 s1 | calibration | row | 0.9743 | 0.9564 | 0.9925 | 0.9850 | 0.0802 | 0.0249 | 2 | 0 |
| C + B3 s2 | selection | row | 0.8523 | 0.8281 | 0.9712 | 0.9414 | 0.0986 | 0.0913 | 1 | 2 |
| C + B3 s2 | calibration | row | 0.9695 | 0.9587 | 0.9925 | 0.9753 | 0.0844 | 0.0299 | 2 | 1 |
| Original FACT s0 | selection | row | 0.9062 | 0.9258 | 0.9476 | 0.9886 | 0.0000 | 0.0522 | 0 | 0 |
| Original FACT s0 | calibration | row | 0.9674 | 0.9678 | 0.9824 | 0.9856 | 0.0338 | 0.0249 | 1 | 1 |
| Original FACT s1 | selection | row | 0.8983 | 0.9284 | 0.9503 | 0.9780 | 0.0000 | 0.0435 | 1 | 0 |
| Original FACT s1 | calibration | row | 0.9605 | 0.9676 | 0.9749 | 0.9856 | 0.0295 | 0.0299 | 1 | 1 |
| Original FACT s2 | selection | row | 0.9109 | 0.9253 | 0.9398 | 0.9664 | 0.0000 | 0.0348 | 2 | 0 |
| Original FACT s2 | calibration | row | 0.9653 | 0.9724 | 0.9724 | 0.9929 | 0.0338 | 0.0249 | 0 | 1 |
| paired ASRF s0 | selection | row | 0.8779 | 0.9273 | 0.9346 | 0.9304 | 0.0106 | 0.0652 | 4 | 1 |
| paired ASRF s0 | calibration | row | 0.9721 | 0.9752 | 0.9874 | 0.9856 | 0.0042 | 0.0348 | 1 | 1 |
| paired ASRF s1 | selection | row | 0.8876 | 0.9373 | 0.9398 | 0.9537 | 0.0070 | 0.0478 | 3 | 0 |
| paired ASRF s1 | calibration | row | 0.9721 | 0.9776 | 0.9849 | 0.9856 | 0.0042 | 0.0348 | 1 | 1 |
| paired ASRF s2 | selection | row | 0.8825 | 0.9424 | 0.9424 | 0.9537 | 0.0106 | 0.0652 | 3 | 0 |
| paired ASRF s2 | calibration | row | 0.9662 | 0.9799 | 0.9824 | 0.9856 | 0.0042 | 0.0299 | 1 | 1 |

## System comparisons (paired bootstrap; streams reuse recordings)

| comparison | segment F1@50 | native macro-F1 | boundary recall |
|---|---|---|---|
| s0|argmax|Ordered - Shuffled|selection | +0.0048 [-0.0125, +0.0216] | -0.0085 [-0.0279, +0.0000] | +0.0000 [-0.0128, +0.0132] |
| s0|argmax|Ordered - Shuffled|calibration | +0.0128 [-0.0055, +0.0317] | +0.0000 [+0.0000, +0.0000] | +0.0050 [-0.0076, +0.0175] |
| s0|b3|Ordered - Shuffled|selection | +0.0044 [-0.0122, +0.0208] | +0.0085 [+0.0000, +0.0271] | +0.0000 [-0.0182, +0.0194] |
| s0|b3|Ordered - Shuffled|calibration | +0.0067 [-0.0031, +0.0175] | +0.0068 [+0.0000, +0.0226] | +0.0050 [-0.0082, +0.0184] |
| s0|argmax|Ordered - Target-only|selection | +0.1715 [+0.1355, +0.2074] | -0.0070 [-0.0434, +0.0271] | -0.0105 [-0.0249, +0.0028] |
| s0|argmax|Ordered - Target-only|calibration | +0.1833 [+0.1460, +0.2209] | -0.0068 [-0.0226, +0.0000] | -0.0101 [-0.0209, -0.0024] |
| s0|b3|Ordered - Target-only|selection | +0.0298 [+0.0020, +0.0578] | +0.0106 [-0.0357, +0.0466] | +0.0393 [+0.0108, +0.0682] |
| s0|b3|Ordered - Target-only|calibration | +0.0315 [+0.0128, +0.0519] | +0.0073 [+0.0000, +0.0278] | +0.0302 [+0.0055, +0.0564] |
| s0|argmax|Shuffled - Target-only|selection | +0.1667 [+0.1303, +0.2041] | +0.0015 [-0.0308, +0.0324] | -0.0105 [-0.0252, +0.0051] |
| s0|argmax|Shuffled - Target-only|calibration | +0.1705 [+0.1345, +0.2063] | -0.0068 [-0.0226, +0.0000] | -0.0151 [-0.0284, -0.0049] |
| s0|b3|Shuffled - Target-only|selection | +0.0254 [-0.0040, +0.0549] | +0.0020 [-0.0389, +0.0371] | +0.0393 [+0.0131, +0.0672] |
| s0|b3|Shuffled - Target-only|calibration | +0.0248 [+0.0055, +0.0460] | +0.0005 [-0.0195, +0.0233] | +0.0251 [-0.0024, +0.0526] |
| s0|b3|Ordered - (E1 + B3)|selection | +0.0445 [+0.0138, +0.0751] | +0.0006 [-0.0387, +0.0392] | +0.0026 [-0.0212, +0.0263] |
| s0|b3|Ordered - (E1 + B3)|calibration | +0.0228 [+0.0103, +0.0352] | +0.0073 [+0.0000, +0.0268] | +0.0050 [-0.0124, +0.0217] |
| s0|b3|Ordered - (E0 + B3)|selection | +0.0041 [-0.0199, +0.0279] | +0.0136 [-0.0365, +0.0650] | -0.0026 [-0.0232, +0.0186] |
| s0|b3|Ordered - (E0 + B3)|calibration | +0.0477 [+0.0268, +0.0716] | +0.0270 [-0.0074, +0.0649] | +0.0276 [+0.0049, +0.0517] |
| s0|b3|Ordered - (B7 + B3)|selection | +0.0586 [+0.0323, +0.0855] | +0.0220 [-0.0167, +0.0687] | +0.0026 [-0.0168, +0.0237] |
| s0|b3|Ordered - (B7 + B3)|calibration | +0.0096 [-0.0019, +0.0228] | +0.0003 [-0.0226, +0.0243] | +0.0075 [-0.0104, +0.0260] |
| s0|b3|Ordered - (C + B3)|selection | +0.0843 [+0.0531, +0.1159] | +0.0568 [+0.0000, +0.1282] | -0.0079 [-0.0315, +0.0157] |
| s0|b3|Ordered - (C + B3)|calibration | +0.0170 [+0.0036, +0.0315] | +0.0176 [+0.0000, +0.0460] | -0.0075 [-0.0222, +0.0075] |
| s0|b3|Ordered - Original FACT|selection | +0.0323 [+0.0037, +0.0615] | -0.0100 [-0.0421, +0.0196] | +0.0105 [-0.0112, +0.0350] |
| s0|b3|Ordered - Original FACT|calibration | +0.0143 [-0.0009, +0.0302] | +0.0073 [+0.0000, +0.0268] | +0.0000 [-0.0155, +0.0154] |
| s0|b3|Ordered - paired ASRF|selection | +0.0606 [+0.0291, +0.0921] | +0.0482 [-0.0132, +0.1223] | +0.0236 [-0.0100, +0.0584] |
| s0|b3|Ordered - paired ASRF|calibration | +0.0096 [-0.0021, +0.0220] | +0.0073 [+0.0000, +0.0268] | -0.0050 [-0.0206, +0.0100] |
| s0|b3|Ordered - historical E1 + ASRF|selection | +0.0285 [-0.0001, +0.0561] | +0.0309 [-0.0097, +0.0746] | +0.0157 [-0.0095, +0.0412] |
| s0|b3|Ordered - historical E1 + ASRF|calibration | +0.0059 [-0.0047, +0.0164] | +0.0000 [+0.0000, +0.0000] | +0.0050 [-0.0122, +0.0217] |
| s1|argmax|Ordered - Shuffled|selection | +0.0265 [+0.0009, +0.0531] | +0.0106 [-0.0244, +0.0507] | +0.0052 [-0.0073, +0.0182] |
| s1|argmax|Ordered - Shuffled|calibration | +0.0154 [-0.0029, +0.0344] | -0.0066 [-0.0221, +0.0000] | +0.0126 [+0.0000, +0.0262] |
| s1|b3|Ordered - Shuffled|selection | +0.0279 [+0.0075, +0.0504] | +0.0015 [-0.0371, +0.0442] | -0.0079 [-0.0263, +0.0079] |
| s1|b3|Ordered - Shuffled|calibration | +0.0113 [+0.0039, +0.0199] | +0.0000 [+0.0000, +0.0000] | +0.0075 [-0.0026, +0.0189] |
| s1|argmax|Ordered - Target-only|selection | +0.1762 [+0.1336, +0.2177] | +0.0197 [-0.0216, +0.0504] | -0.0052 [-0.0178, +0.0077] |
| s1|argmax|Ordered - Target-only|calibration | +0.1928 [+0.1588, +0.2281] | +0.0000 [-0.0156, +0.0217] | -0.0075 [-0.0175, +0.0000] |
| s1|b3|Ordered - Target-only|selection | +0.1296 [+0.0893, +0.1728] | +0.0269 [-0.0195, +0.0666] | -0.0209 [-0.0416, -0.0026] |
| s1|b3|Ordered - Target-only|calibration | +0.0750 [+0.0527, +0.0992] | +0.0000 [-0.0239, +0.0309] | -0.0050 [-0.0176, +0.0073] |
| s1|argmax|Shuffled - Target-only|selection | +0.1497 [+0.1064, +0.1924] | +0.0091 [-0.0273, +0.0335] | -0.0105 [-0.0262, +0.0053] |
| s1|argmax|Shuffled - Target-only|calibration | +0.1773 [+0.1431, +0.2126] | +0.0066 [-0.0138, +0.0328] | -0.0201 [-0.0352, -0.0077] |
| s1|b3|Shuffled - Target-only|selection | +0.1017 [+0.0638, +0.1398] | +0.0254 [-0.0140, +0.0553] | -0.0131 [-0.0296, +0.0027] |
| s1|b3|Shuffled - Target-only|calibration | +0.0638 [+0.0431, +0.0872] | +0.0000 [-0.0239, +0.0309] | -0.0126 [-0.0281, +0.0000] |
| s1|b3|Ordered - (E1 + B3)|selection | +0.0664 [+0.0389, +0.0964] | +0.0006 [-0.0387, +0.0392] | +0.0131 [-0.0080, +0.0361] |
| s1|b3|Ordered - (E1 + B3)|calibration | +0.0247 [+0.0138, +0.0367] | +0.0005 [-0.0171, +0.0252] | +0.0101 [-0.0051, +0.0256] |
| s1|b3|Ordered - (E0 + B3)|selection | +0.0259 [+0.0038, +0.0506] | +0.0136 [-0.0365, +0.0650] | +0.0079 [-0.0106, +0.0292] |
| s1|b3|Ordered - (E0 + B3)|calibration | +0.0497 [+0.0284, +0.0742] | +0.0202 [-0.0113, +0.0594] | +0.0327 [+0.0099, +0.0558] |
| s1|b3|Ordered - (B7 + B3)|selection | +0.0805 [+0.0528, +0.1105] | +0.0220 [-0.0167, +0.0687] | +0.0131 [-0.0080, +0.0361] |
| s1|b3|Ordered - (B7 + B3)|calibration | +0.0115 [-0.0010, +0.0257] | -0.0065 [-0.0299, +0.0199] | +0.0126 [-0.0028, +0.0308] |
| s1|b3|Ordered - (C + B3)|selection | +0.1037 [+0.0682, +0.1401] | +0.0309 [-0.0089, +0.0794] | +0.0000 [-0.0238, +0.0236] |
| s1|b3|Ordered - (C + B3)|calibration | +0.0093 [-0.0021, +0.0214] | +0.0011 [-0.0253, +0.0363] | -0.0050 [-0.0153, +0.0027] |
| s1|b3|Ordered - Original FACT|selection | +0.0620 [+0.0312, +0.0942] | +0.0006 [-0.0387, +0.0392] | +0.0183 [-0.0029, +0.0416] |
| s1|b3|Ordered - Original FACT|calibration | +0.0231 [+0.0077, +0.0408] | +0.0005 [-0.0171, +0.0252] | +0.0126 [-0.0026, +0.0289] |
| s1|b3|Ordered - paired ASRF|selection | +0.0728 [+0.0433, +0.1043] | +0.0249 [-0.0260, +0.0823] | +0.0288 [+0.0027, +0.0556] |
| s1|b3|Ordered - paired ASRF|calibration | +0.0115 [+0.0009, +0.0239] | +0.0005 [-0.0171, +0.0252] | +0.0025 [-0.0102, +0.0157] |
| s1|b3|Ordered - historical E1 + ASRF|selection | +0.0514 [+0.0215, +0.0822] | +0.0122 [-0.0332, +0.0621] | +0.0183 [-0.0078, +0.0456] |
| s1|b3|Ordered - historical E1 + ASRF|calibration | +0.0193 [+0.0066, +0.0336] | +0.0108 [+0.0000, +0.0306] | +0.0151 [-0.0025, +0.0339] |
| s2|argmax|Ordered - Shuffled|selection | +0.0309 [+0.0088, +0.0539] | +0.0080 [-0.0163, +0.0386] | +0.0052 [-0.0051, +0.0159] |
| s2|argmax|Ordered - Shuffled|calibration | +0.0026 [-0.0150, +0.0210] | -0.0068 [-0.0191, +0.0000] | +0.0025 [-0.0124, +0.0173] |
| s2|b3|Ordered - Shuffled|selection | +0.0252 [+0.0064, +0.0461] | +0.0080 [-0.0163, +0.0386] | +0.0183 [+0.0027, +0.0363] |
| s2|b3|Ordered - Shuffled|calibration | +0.0000 [-0.0066, +0.0071] | -0.0068 [-0.0191, +0.0000] | +0.0075 [-0.0051, +0.0205] |
| s2|argmax|Ordered - Target-only|selection | +0.1750 [+0.1397, +0.2102] | -0.0070 [-0.0438, +0.0267] | -0.0079 [-0.0192, +0.0026] |
| s2|argmax|Ordered - Target-only|calibration | +0.1605 [+0.1280, +0.1919] | +0.0000 [-0.0171, +0.0204] | -0.0151 [-0.0277, -0.0048] |
| s2|b3|Ordered - Target-only|selection | +0.1279 [+0.0952, +0.1621] | -0.0070 [-0.0438, +0.0267] | -0.0105 [-0.0300, +0.0078] |
| s2|b3|Ordered - Target-only|calibration | +0.0735 [+0.0503, +0.0990] | +0.0000 [-0.0171, +0.0204] | -0.0101 [-0.0256, +0.0051] |
| s2|argmax|Shuffled - Target-only|selection | +0.1441 [+0.1045, +0.1814] | -0.0150 [-0.0562, +0.0152] | -0.0131 [-0.0271, +0.0000] |
| s2|argmax|Shuffled - Target-only|calibration | +0.1578 [+0.1236, +0.1903] | +0.0068 [+0.0000, +0.0226] | -0.0176 [-0.0310, -0.0072] |
| s2|b3|Shuffled - Target-only|selection | +0.1027 [+0.0651, +0.1410] | -0.0150 [-0.0562, +0.0152] | -0.0288 [-0.0525, -0.0078] |
| s2|b3|Shuffled - Target-only|calibration | +0.0734 [+0.0498, +0.0987] | +0.0068 [+0.0000, +0.0226] | -0.0176 [-0.0351, +0.0000] |
| s2|b3|Ordered - (E1 + B3)|selection | +0.0704 [+0.0440, +0.0995] | -0.0079 [-0.0403, +0.0226] | +0.0183 [-0.0027, +0.0405] |
| s2|b3|Ordered - (E1 + B3)|calibration | +0.0208 [+0.0090, +0.0324] | +0.0005 [-0.0171, +0.0252] | +0.0025 [-0.0148, +0.0196] |
| s2|b3|Ordered - (E0 + B3)|selection | +0.0299 [+0.0093, +0.0534] | +0.0050 [-0.0390, +0.0551] | +0.0131 [-0.0053, +0.0337] |
| s2|b3|Ordered - (E0 + B3)|calibration | +0.0458 [+0.0243, +0.0704] | +0.0202 [-0.0113, +0.0594] | +0.0251 [+0.0000, +0.0501] |
| s2|b3|Ordered - (B7 + B3)|selection | +0.0845 [+0.0594, +0.1122] | +0.0134 [-0.0186, +0.0560] | +0.0183 [+0.0000, +0.0388] |
| s2|b3|Ordered - (B7 + B3)|calibration | +0.0076 [-0.0044, +0.0215] | -0.0065 [-0.0299, +0.0199] | +0.0050 [-0.0146, +0.0242] |
| s2|b3|Ordered - (C + B3)|selection | +0.1120 [+0.0786, +0.1466] | +0.0287 [-0.0197, +0.0937] | +0.0026 [-0.0185, +0.0233] |
| s2|b3|Ordered - (C + B3)|calibration | +0.0102 [-0.0031, +0.0247] | +0.0108 [-0.0116, +0.0429] | -0.0126 [-0.0260, +0.0000] |
| s2|b3|Ordered - Original FACT|selection | +0.0535 [+0.0255, +0.0837] | +0.0037 [-0.0355, +0.0482] | +0.0340 [+0.0109, +0.0580] |
| s2|b3|Ordered - Original FACT|calibration | +0.0145 [+0.0017, +0.0279] | -0.0068 [-0.0191, +0.0000] | +0.0075 [-0.0106, +0.0260] |
| s2|b3|Ordered - paired ASRF|selection | +0.0819 [+0.0509, +0.1145] | +0.0164 [-0.0290, +0.0681] | +0.0314 [+0.0025, +0.0612] |
| s2|b3|Ordered - paired ASRF|calibration | +0.0135 [+0.0020, +0.0258] | +0.0005 [-0.0171, +0.0252] | -0.0025 [-0.0191, +0.0132] |
| s2|b3|Ordered - historical E1 + ASRF|selection | +0.0572 [+0.0302, +0.0852] | +0.0037 [-0.0355, +0.0482] | +0.0209 [-0.0027, +0.0455] |
| s2|b3|Ordered - historical E1 + ASRF|calibration | +0.0154 [+0.0019, +0.0310] | +0.0108 [+0.0000, +0.0306] | +0.0025 [-0.0133, +0.0199] |

## Verdicts

```
{
 "recognition": {
  "Ordered - Shuffled": {
   "A4C/A5C balanced accuracy": {
    "selection": "no consistent difference detected",
    "calibration": "no consistent difference detected"
   },
   "A5C-A4C margin AUC": {
    "selection": "no consistent difference detected",
    "calibration": "no consistent difference detected"
   },
   "native window accuracy": {
    "selection": "no consistent difference detected",
    "calibration": "no consistent difference detected"
   }
  },
  "Ordered - Target-only": {
   "A4C/A5C balanced accuracy": {
    "selection": "consistent positive",
    "calibration": "consistent positive"
   },
   "A5C-A4C margin AUC": {
    "selection": "consistent positive",
    "calibration": "consistent positive"
   },
   "native window accuracy": {
    "selection": "consistent positive",
    "calibration": "consistent positive"
   }
  },
  "Shuffled - Target-only": {
   "A4C/A5C balanced accuracy": {
    "selection": "consistent positive",
    "calibration": "consistent positive"
   },
   "A5C-A4C margin AUC": {
    "selection": "consistent positive",
    "calibration": "consistent positive"
   },
   "native window accuracy": {
    "selection": "consistent positive",
    "calibration": "consistent positive"
   }
  },
  "Target-only - E1": {
   "A4C/A5C balanced accuracy": {
    "selection": "no consistent difference detected",
    "calibration": "no consistent difference detected"
   },
   "A5C-A4C margin AUC": {
    "selection": "no consistent difference detected",
    "calibration": "consistent positive"
   },
   "native window accuracy": {
    "selection": "no consistent difference detected",
    "calibration": "no consistent difference detected"
   }
  },
  "Shuffled context - E1": {
   "A4C/A5C balanced accuracy": {
    "selection": "consistent positive",
    "calibration": "consistent positive"
   },
   "A5C-A4C margin AUC": {
    "selection": "consistent positive",
    "calibration": "consistent positive"
   },
   "native window accuracy": {
    "selection": "consistent positive",
    "calibration": "consistent positive"
   }
  },
  "Ordered context - E1": {
   "A4C/A5C balanced accuracy": {
    "selection": "consistent positive",
    "calibration": "consistent positive"
   },
   "A5C-A4C margin AUC": {
    "selection": "consistent positive",
    "calibration": "consistent positive"
   },
   "native window accuracy": {
    "selection": "consistent positive",
    "calibration": "consistent positive"
   }
  }
 },
 "system": {
  "argmax": {
   "Ordered - Shuffled": {
    "selection": "no consistent difference detected",
    "calibration": "no consistent difference detected"
   },
   "Ordered - Target-only": {
    "selection": "consistent positive",
    "calibration": "consistent positive"
   },
   "Shuffled - Target-only": {
    "selection": "consistent positive",
    "calibration": "consistent positive"
   }
  },
  "b3": {
   "Ordered - Shuffled": {
    "selection": "no consistent difference detected",
    "calibration": "no consistent difference detected"
   },
   "Ordered - Target-only": {
    "selection": "consistent positive",
    "calibration": "consistent positive"
   },
   "Shuffled - Target-only": {
    "selection": "no consistent difference detected",
    "calibration": "consistent positive"
   },
   "Ordered - (E1 + B3)": {
    "selection": "consistent positive",
    "calibration": "consistent positive"
   },
   "Ordered - (E0 + B3)": {
    "selection": "no consistent difference detected",
    "calibration": "consistent positive"
   },
   "Ordered - (B7 + B3)": {
    "selection": "consistent positive",
    "calibration": "no consistent difference detected"
   },
   "Ordered - (C + B3)": {
    "selection": "consistent positive",
    "calibration": "no consistent difference detected"
   },
   "Ordered - Original FACT": {
    "selection": "consistent positive",
    "calibration": "no consistent difference detected"
   },
   "Ordered - paired ASRF": {
    "selection": "consistent positive",
    "calibration": "no consistent difference detected"
   },
   "Ordered - historical E1 + ASRF": {
    "selection": "no consistent difference detected",
    "calibration": "no consistent difference detected"
   }
  }
 }
}
```

## Goal

```
{
 "Target-only": {
  "mean_segment_f1": {
   "selection": 0.8586236887422204,
   "calibration": 0.9217065408159176
  },
  "margin": {
   "selection": -0.08578599844244161,
   "calibration": -0.06043563017351561
  },
  "calibration_sensitivity_margin": -0.06218703765885869,
  "constraint_failures": [
   {
    "seed": 0,
    "half": "selection",
    "b3_feasible": true,
    "boundary_recall": 0.918848167539267,
    "native_macro_f1": 0.9680280435721336
   },
   {
    "seed": 1,
    "half": "selection",
    "b3_feasible": true,
    "boundary_recall": 0.9895287958115183,
    "native_macro_f1": 0.9517352549699842
   }
  ],
  "met": false
 },
 "Shuffled context": {
  "mean_segment_f1": {
   "selection": 0.9352449485810466,
   "calibration": 0.9757042510077852
  },
  "margin": {
   "selection": -0.009164738603615441,
   "calibration": -0.006437919981647999
  },
  "calibration_sensitivity_margin": -0.008189327466991081,
  "constraint_failures": [
   {
    "seed": 0,
    "half": "selection",
    "b3_feasible": true,
    "boundary_recall": 0.9581151832460733,
    "native_macro_f1": 0.9700744362565338
   },
   {
    "seed": 2,
    "half": "selection",
    "b3_feasible": true,
    "boundary_recall": 0.9554973821989529,
    "native_macro_f1": 0.9621080950641865
   }
  ],
  "met": false
 },
 "Ordered context": {
  "mean_segment_f1": {
   "selection": 0.9543886531510294,
   "calibration": 0.9817001920287197
  },
  "margin": {
   "selection": 0.009978965966367381,
   "calibration": -0.0004419789607135183
  },
  "calibration_sensitivity_margin": -0.0021933864460566,
  "constraint_failures": [
   {
    "seed": 2,
    "half": "selection",
    "b3_feasible": true,
    "boundary_recall": 0.9738219895287958,
    "native_macro_f1": 0.9700744362565338
   }
  ],
  "met": false
 }
}
```

## Decision

```
{
 "rows_that_apply": [
  {
   "rule": "multiframe_gain",
   "text": "Ordered - Target-only and Shuffled - Target-only both consistent positive in both halves, and Ordered - Shuffled not consistent positive in both -> 'a gain under multi-frame input, no added value of order shown'; not equivalence; repeated frames differ from the pretraining input distribution"
  },
  {
   "rule": "system_gain",
   "text": "Ordered + B3 consistent positive against a comparator in both halves (against: Ordered - Target-only, Ordered - (E1 + B3))"
  }
 ],
 "order_gain_both_halves": false,
 "order_gain_halves": [],
 "multiframe_gain": true,
 "recognition_gains_both_halves": [
  "Ordered - Target-only",
  "Shuffled - Target-only"
 ],
 "ranking_only": [],
 "recognition_but_system_worse": [],
 "system_gains_both_halves": [
  "Ordered - Target-only",
  "Ordered - (E1 + B3)"
 ],
 "controls_better": [],
 "goal_met_groups": []
}
```
