# Segment objective round: tables

Recompute: `python scripts/report_segment_objective.py --config configs/experiments/segment_objective_v1.yaml --selection --choices-sha256 <sha> --choices-commit <commit>`

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
| F + B3 s0 | selection | 0.9473 | 0.9530 | 0.9555 | 0.9542 | 0.9786 | 0.0458 | 0.0261 | 0.9687 | 2 | 0 |
| F + B3 s0 | calibration | 0.9768 | 0.9749 | 0.9749 | 0.9749 | 0.9861 | 0.0127 | 0.0199 | 0.9709 | 2 | 0 |
| F argmax s0 | selection | 0.9007 | 0.8341 | 0.9738 | 0.8986 | 0.9701 | 0.1338 | 0.0391 | 0.9547 | 3 | 0 |
| F argmax s0 | calibration | 0.9410 | 0.9091 | 0.9799 | 0.9432 | 0.9861 | 0.1181 | 0.0398 | 0.9465 | 2 | 0 |
| F at Ordered setting s0 | selection | 0.9513 | 0.9808 | 0.9346 | 0.9571 | 0.9786 | 0.0176 | 0.0130 | 0.9685 | 2 | 0 |
| F at Ordered setting s0 | calibration | 0.9698 | 0.9845 | 0.9573 | 0.9707 | 0.9861 | 0.0084 | 0.0100 | 0.9619 | 2 | 0 |
| B + B3 s0 | selection | 0.9481 | 0.9579 | 0.9529 | 0.9554 | 0.9786 | 0.0563 | 0.0261 | 0.9687 | 2 | 0 |
| B + B3 s0 | calibration | 0.9768 | 0.9774 | 0.9774 | 0.9774 | 0.9861 | 0.0127 | 0.0199 | 0.9767 | 2 | 0 |
| B argmax s0 | selection | 0.9024 | 0.8356 | 0.9712 | 0.8983 | 0.9701 | 0.1338 | 0.0391 | 0.9619 | 3 | 0 |
| B argmax s0 | calibration | 0.9509 | 0.9265 | 0.9824 | 0.9537 | 0.9861 | 0.1139 | 0.0348 | 0.9492 | 2 | 0 |
| B at Ordered setting s0 | selection | 0.9544 | 0.9836 | 0.9398 | 0.9612 | 0.9786 | 0.0176 | 0.0087 | 0.9685 | 2 | 0 |
| B at Ordered setting s0 | calibration | 0.9737 | 0.9871 | 0.9648 | 0.9759 | 0.9861 | 0.0084 | 0.0100 | 0.9766 | 2 | 0 |
| O + B3 s0 | selection | 0.9512 | 0.9530 | 0.9555 | 0.9542 | 0.9701 | 0.0563 | 0.0217 | 0.9687 | 3 | 0 |
| O + B3 s0 | calibration | 0.9778 | 0.9774 | 0.9799 | 0.9787 | 0.9861 | 0.0127 | 0.0199 | 0.9767 | 2 | 0 |
| O argmax s0 | selection | 0.9035 | 0.8360 | 0.9738 | 0.8996 | 0.9701 | 0.1408 | 0.0391 | 0.9619 | 3 | 0 |
| O argmax s0 | calibration | 0.9456 | 0.9159 | 0.9849 | 0.9492 | 0.9861 | 0.1055 | 0.0348 | 0.9492 | 2 | 0 |
| O at Ordered setting s0 | selection | 0.9525 | 0.9782 | 0.9398 | 0.9586 | 0.9701 | 0.0176 | 0.0130 | 0.9685 | 3 | 0 |
| O at Ordered setting s0 | calibration | 0.9757 | 0.9872 | 0.9698 | 0.9785 | 0.9861 | 0.0084 | 0.0100 | 0.9766 | 2 | 0 |
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
| F + B3 s1 | selection | 0.9569 | 0.9707 | 0.9529 | 0.9617 | 0.9786 | 0.0387 | 0.0087 | 0.9522 | 2 | 0 |
| F + B3 s1 | calibration | 0.9779 | 0.9726 | 0.9799 | 0.9762 | 0.9861 | 0.0169 | 0.0149 | 0.9709 | 2 | 0 |
| F argmax s1 | selection | 0.9074 | 0.8578 | 0.9791 | 0.9144 | 0.9701 | 0.1549 | 0.0348 | 0.9409 | 3 | 0 |
| F argmax s1 | calibration | 0.9583 | 0.9378 | 0.9849 | 0.9608 | 0.9732 | 0.0802 | 0.0398 | 0.9598 | 4 | 0 |
| F at Ordered setting s1 | selection | 0.9533 | 0.9807 | 0.9293 | 0.9543 | 0.9786 | 0.0176 | 0.0087 | 0.9517 | 2 | 0 |
| F at Ordered setting s1 | calibration | 0.9689 | 0.9745 | 0.9598 | 0.9671 | 0.9861 | 0.0127 | 0.0149 | 0.9591 | 2 | 0 |
| B + B3 s1 | selection | 0.9550 | 0.9655 | 0.9529 | 0.9592 | 0.9701 | 0.0352 | 0.0130 | 0.9522 | 3 | 0 |
| B + B3 s1 | calibration | 0.9789 | 0.9703 | 0.9849 | 0.9776 | 0.9861 | 0.0169 | 0.0249 | 0.9767 | 2 | 0 |
| B argmax s1 | selection | 0.9156 | 0.8661 | 0.9817 | 0.9202 | 0.9701 | 0.1303 | 0.0348 | 0.9434 | 3 | 0 |
| B argmax s1 | calibration | 0.9639 | 0.9469 | 0.9849 | 0.9655 | 0.9929 | 0.0844 | 0.0348 | 0.9543 | 1 | 0 |
| B at Ordered setting s1 | selection | 0.9513 | 0.9780 | 0.9319 | 0.9544 | 0.9701 | 0.0211 | 0.0130 | 0.9517 | 3 | 0 |
| B at Ordered setting s1 | calibration | 0.9679 | 0.9769 | 0.9548 | 0.9657 | 0.9861 | 0.0127 | 0.0149 | 0.9619 | 2 | 0 |
| O + B3 s1 | selection | 0.9569 | 0.9707 | 0.9529 | 0.9617 | 0.9701 | 0.0282 | 0.0087 | 0.9496 | 3 | 0 |
| O + B3 s1 | calibration | 0.9788 | 0.9774 | 0.9774 | 0.9774 | 0.9861 | 0.0127 | 0.0149 | 0.9709 | 2 | 0 |
| O argmax s1 | selection | 0.9235 | 0.8821 | 0.9791 | 0.9280 | 0.9621 | 0.1444 | 0.0304 | 0.9479 | 4 | 0 |
| O argmax s1 | calibration | 0.9639 | 0.9469 | 0.9849 | 0.9655 | 0.9861 | 0.0886 | 0.0348 | 0.9543 | 2 | 0 |
| O at Ordered setting s1 | selection | 0.9556 | 0.9757 | 0.9450 | 0.9601 | 0.9701 | 0.0176 | 0.0087 | 0.9542 | 3 | 0 |
| O at Ordered setting s1 | calibration | 0.9729 | 0.9772 | 0.9673 | 0.9722 | 0.9861 | 0.0127 | 0.0149 | 0.9591 | 2 | 0 |
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
| F + B3 s2 | selection | 0.9388 | 0.9115 | 0.9712 | 0.9404 | 0.9786 | 0.0845 | 0.0304 | 0.9571 | 2 | 0 |
| F + B3 s2 | calibration | 0.9732 | 0.9532 | 0.9724 | 0.9627 | 0.9861 | 0.0422 | 0.0299 | 0.9709 | 2 | 0 |
| F argmax s2 | selection | 0.9082 | 0.8408 | 0.9817 | 0.9058 | 0.9786 | 0.1620 | 0.0348 | 0.9368 | 2 | 0 |
| F argmax s2 | calibration | 0.9500 | 0.9173 | 0.9749 | 0.9452 | 0.9796 | 0.1308 | 0.0398 | 0.9543 | 3 | 0 |
| F at Ordered setting s2 | selection | 0.9491 | 0.9833 | 0.9267 | 0.9542 | 0.9877 | 0.0000 | 0.0087 | 0.9614 | 1 | 0 |
| F at Ordered setting s2 | calibration | 0.9668 | 0.9793 | 0.9497 | 0.9643 | 0.9796 | 0.0127 | 0.0149 | 0.9588 | 3 | 0 |
| B + B3 s2 | selection | 0.9620 | 0.9709 | 0.9607 | 0.9658 | 0.9700 | 0.0387 | 0.0130 | 0.9569 | 2 | 0 |
| B + B3 s2 | calibration | 0.9806 | 0.9797 | 0.9698 | 0.9747 | 0.9929 | 0.0169 | 0.0149 | 0.9591 | 1 | 0 |
| B argmax s2 | selection | 0.9161 | 0.8565 | 0.9843 | 0.9160 | 0.9700 | 0.1514 | 0.0348 | 0.9528 | 2 | 0 |
| B argmax s2 | calibration | 0.9537 | 0.9264 | 0.9799 | 0.9524 | 0.9929 | 0.1561 | 0.0398 | 0.9489 | 1 | 0 |
| B at Ordered setting s2 | selection | 0.9480 | 0.9833 | 0.9241 | 0.9528 | 0.9792 | 0.0106 | 0.0087 | 0.9614 | 1 | 0 |
| B at Ordered setting s2 | calibration | 0.9667 | 0.9844 | 0.9497 | 0.9668 | 0.9929 | 0.0127 | 0.0149 | 0.9588 | 1 | 0 |
| O + B3 s2 | selection | 0.9610 | 0.9735 | 0.9607 | 0.9671 | 0.9786 | 0.0458 | 0.0130 | 0.9569 | 2 | 0 |
| O + B3 s2 | calibration | 0.9816 | 0.9797 | 0.9724 | 0.9760 | 0.9861 | 0.0169 | 0.0149 | 0.9679 | 2 | 0 |
| O argmax s2 | selection | 0.9151 | 0.8562 | 0.9817 | 0.9146 | 0.9700 | 0.1866 | 0.0348 | 0.9437 | 2 | 0 |
| O argmax s2 | calibration | 0.9472 | 0.9218 | 0.9774 | 0.9488 | 0.9861 | 0.1392 | 0.0398 | 0.9516 | 2 | 0 |
| O at Ordered setting s2 | selection | 0.9523 | 0.9835 | 0.9346 | 0.9584 | 0.9877 | 0.0035 | 0.0087 | 0.9614 | 1 | 0 |
| O at Ordered setting s2 | calibration | 0.9717 | 0.9819 | 0.9548 | 0.9682 | 0.9861 | 0.0127 | 0.0149 | 0.9588 | 2 | 0 |

## Paired comparisons (seed | decoding | comparison | half; 95% bootstrap intervals)

| comparison | multi segment F1@50 | native macro-F1 | multi boundary recall | multi FP segments | multi FN segments | native cuts per clip |
|---|---|---|---|---|---|---|
| s0|b3|O - B|selection | +0.0031 [-0.0019, +0.0105] | -0.0085 [-0.0308, +0.0000] | +0.0026 [+0.0000, +0.0082] | +0.0000 [-3.0000, +3.0000] | -3.0000 [-8.0000, +0.0000] | +0.0000 [-0.0106, +0.0085] |
| s0|b3|B - F|selection | +0.0008 [-0.0031, +0.0052] | +0.0000 [+0.0000, +0.0000] | -0.0026 [-0.0080, +0.0000] | -2.0000 [-7.0000, +3.0000] | +1.0000 [+0.0000, +3.0000] | +0.0106 [-0.0100, +0.0373] |
| s0|b3|O - F|selection | +0.0040 [-0.0021, +0.0121] | -0.0085 [-0.0308, +0.0000] | +0.0000 [-0.0076, +0.0079] | -2.0000 [-7.0000, +3.0000] | -2.0000 [-7.0000, +2.0000] | +0.0106 [-0.0043, +0.0336] |
| s0|b3|O - frozen Ordered|selection | +0.0128 [-0.0023, +0.0292] | -0.0085 [-0.0308, +0.0000] | -0.0026 [-0.0183, +0.0127] | -8.0000 [-18.0250, +1.0000] | -5.0000 [-14.0000, +4.0000] | +0.0141 [-0.0132, +0.0387] |
| s0|b3|B - frozen Ordered|selection | +0.0096 [-0.0051, +0.0252] | +0.0000 [+0.0000, +0.0000] | -0.0052 [-0.0203, +0.0080] | -8.0000 [-18.0000, +1.0000] | -2.0000 [-9.0000, +6.0000] | +0.0141 [-0.0043, +0.0322] |
| s0|b3|F - A01 Linear|selection | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| s0|b3|F - frozen Ordered|selection | +0.0088 [-0.0058, +0.0231] | +0.0000 [+0.0000, +0.0000] | -0.0026 [-0.0184, +0.0125] | -6.0000 [-16.0000, +3.0000] | -3.0000 [-10.0000, +6.0000] | +0.0035 [-0.0359, +0.0365] |
| s0|argmax|O - B|selection | +0.0010 [-0.0028, +0.0060] | +0.0000 [+0.0000, +0.0000] | +0.0026 [+0.0000, +0.0082] | +0.0000 [-6.0000, +5.0000] | -1.0000 [-3.0000, +0.0000] | +0.0070 [+0.0000, +0.0192] |
| s0|argmax|B - F|selection | +0.0017 [-0.0048, +0.0081] | +0.0000 [+0.0000, +0.0000] | -0.0026 [-0.0082, +0.0000] | -2.0000 [-8.0000, +5.0000] | +0.0000 [-3.0000, +3.0000] | +0.0000 [-0.0190, +0.0227] |
| s0|argmax|O - F|selection | +0.0027 [-0.0017, +0.0083] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -2.0000 [-7.0000, +2.0000] | -1.0000 [-3.0000, +0.0000] | +0.0070 [-0.0080, +0.0248] |
| s0|ordered_setting|O - B|selection | -0.0019 [-0.0090, +0.0031] | -0.0085 [-0.0308, +0.0000] | +0.0000 [-0.0078, +0.0077] | +2.0000 [+0.0000, +6.0000] | +0.0000 [-3.0000, +3.0000] | +0.0000 [-0.0106, +0.0085] |
| s0|ordered_setting|B - F|selection | +0.0031 [-0.0020, +0.0105] | +0.0000 [+0.0000, +0.0000] | +0.0052 [+0.0000, +0.0134] | -1.0000 [-6.0000, +3.0000] | -2.0000 [-5.0000, +0.0000] | +0.0000 [-0.0085, +0.0106] |
| s0|ordered_setting|O - F|selection | +0.0012 [-0.0020, +0.0046] | -0.0085 [-0.0308, +0.0000] | +0.0052 [+0.0000, +0.0132] | +1.0000 [+0.0000, +3.0000] | -2.0000 [-5.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| s1|b3|O - B|selection | +0.0019 [-0.0021, +0.0061] | +0.0000 [+0.0000, +0.0000] | +0.0000 [-0.0078, +0.0077] | -2.0000 [-5.0000, +0.0000] | +0.0000 [-3.0000, +3.0000] | -0.0070 [-0.0192, +0.0000] |
| s1|b3|B - F|selection | -0.0019 [-0.0093, +0.0048] | -0.0085 [-0.0272, +0.0000] | +0.0000 [-0.0109, +0.0104] | +2.0000 [-3.0000, +7.0000] | +0.0000 [-4.0000, +4.0000] | -0.0035 [-0.0096, +0.0000] |
| s1|b3|O - F|selection | +0.0000 [-0.0070, +0.0060] | -0.0085 [-0.0272, +0.0000] | +0.0000 [-0.0108, +0.0105] | +0.0000 [-4.0000, +4.0000] | +0.0000 [-4.0000, +4.0000] | -0.0106 [-0.0289, +0.0000] |
| s1|b3|O - frozen Ordered|selection | -0.0034 [-0.0126, +0.0052] | -0.0085 [-0.0272, +0.0000] | -0.0157 [-0.0296, -0.0050] | -4.0000 [-11.0000, +3.0000] | +7.0000 [+2.0000, +12.0000] | -0.0106 [-0.0325, +0.0053] |
| s1|b3|B - frozen Ordered|selection | -0.0054 [-0.0159, +0.0034] | -0.0085 [-0.0272, +0.0000] | -0.0157 [-0.0289, -0.0051] | -2.0000 [-9.0000, +6.0000] | +7.0000 [+2.0000, +13.0000] | -0.0035 [-0.0321, +0.0215] |
| s1|b3|F - A01 Linear|selection | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| s1|b3|F - frozen Ordered|selection | -0.0034 [-0.0128, +0.0052] | +0.0000 [+0.0000, +0.0000] | -0.0157 [-0.0287, -0.0052] | -4.0000 [-11.0000, +3.0000] | +7.0000 [+3.0000, +12.0000] | +0.0000 [-0.0321, +0.0299] |
| s1|argmax|O - B|selection | +0.0079 [-0.0091, +0.0288] | -0.0080 [-0.0257, +0.0000] | -0.0026 [-0.0159, +0.0077] | -9.0000 [-26.0000, +4.0000] | +0.0000 [-7.0000, +8.0000] | +0.0141 [-0.0084, +0.0397] |
| s1|argmax|B - F|selection | +0.0083 [-0.0079, +0.0253] | +0.0000 [-0.0220, +0.0241] | +0.0026 [-0.0077, +0.0159] | -6.0000 [-19.0000, +8.0000] | -3.0000 [-11.0000, +3.0000] | -0.0246 [-0.0628, +0.0047] |
| s1|argmax|O - F|selection | +0.0161 [+0.0055, +0.0299] | -0.0080 [-0.0244, +0.0000] | +0.0000 [+0.0000, +0.0000] | -15.0000 [-28.0000, -5.0000] | -3.0000 [-7.0000, +0.0000] | -0.0106 [-0.0239, +0.0000] |
| s1|ordered_setting|O - B|selection | +0.0043 [-0.0017, +0.0109] | +0.0000 [+0.0000, +0.0000] | +0.0131 [+0.0025, +0.0276] | +1.0000 [-2.0000, +5.0000] | -5.0000 [-11.0000, -1.0000] | -0.0035 [-0.0096, +0.0000] |
| s1|ordered_setting|B - F|selection | -0.0019 [-0.0118, +0.0050] | -0.0085 [-0.0272, +0.0000] | +0.0026 [+0.0000, +0.0081] | +2.0000 [-4.0000, +9.0250] | +0.0000 [-3.0000, +3.0000] | +0.0035 [+0.0000, +0.0096] |
| s1|ordered_setting|O - F|selection | +0.0024 [-0.0058, +0.0097] | -0.0085 [-0.0272, +0.0000] | +0.0157 [+0.0027, +0.0313] | +3.0000 [+0.0000, +9.0000] | -5.0000 [-11.0000, -1.0000] | +0.0000 [+0.0000, +0.0000] |
| s2|b3|O - B|selection | -0.0010 [-0.0053, +0.0030] | +0.0086 [+0.0000, +0.0325] | +0.0000 [-0.0078, +0.0079] | +0.0000 [-4.0000, +4.0000] | +1.0000 [+0.0000, +3.0000] | +0.0070 [+0.0000, +0.0192] |
| s2|b3|B - F|selection | +0.0232 [+0.0090, +0.0413] | -0.0086 [-0.0325, +0.0000] | -0.0105 [-0.0215, -0.0025] | -27.0000 [-45.0000, -13.0000] | +2.0000 [-3.0000, +7.0000] | -0.0458 [-0.0871, -0.0087] |
| s2|b3|O - F|selection | +0.0222 [+0.0086, +0.0390] | +0.0000 [+0.0000, +0.0000] | -0.0105 [-0.0221, -0.0025] | -27.0000 [-43.0000, -14.0000] | +3.0000 [-2.0000, +8.0000] | -0.0387 [-0.0776, -0.0087] |
| s2|b3|O - frozen Ordered|selection | -0.0034 [-0.0133, +0.0059] | +0.0085 [+0.0000, +0.0241] | -0.0131 [-0.0252, -0.0027] | -4.0000 [-13.0000, +4.0000] | +7.0000 [+3.0000, +12.0000] | +0.0000 [-0.0172, +0.0152] |
| s2|b3|B - frozen Ordered|selection | -0.0024 [-0.0118, +0.0060] | -0.0001 [-0.0271, +0.0193] | -0.0131 [-0.0258, -0.0028] | -4.0000 [-12.0000, +4.0000] | +6.0000 [+2.0000, +11.0000] | -0.0070 [-0.0181, +0.0000] |
| s2|b3|F - A01 Linear|selection | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| s2|b3|F - frozen Ordered|selection | -0.0256 [-0.0446, -0.0093] | +0.0085 [+0.0000, +0.0241] | -0.0026 [-0.0082, +0.0000] | +23.0000 [+9.0000, +40.0000] | +4.0000 [+0.0000, +9.0000] | +0.0387 [+0.0037, +0.0812] |
| s2|argmax|O - B|selection | -0.0010 [-0.0069, +0.0050] | +0.0000 [+0.0000, +0.0000] | -0.0026 [-0.0082, +0.0000] | +0.0000 [-6.0250, +6.0000] | +1.0000 [+0.0000, +3.0000] | +0.0352 [+0.0000, +0.0738] |
| s2|argmax|B - F|selection | +0.0079 [+0.0008, +0.0153] | -0.0086 [-0.0325, +0.0000] | +0.0026 [+0.0000, +0.0082] | -8.0000 [-17.0000, +0.0000] | -1.0000 [-3.0000, +0.0000] | -0.0106 [-0.0237, +0.0000] |
| s2|argmax|O - F|selection | +0.0069 [+0.0018, +0.0124] | -0.0086 [-0.0325, +0.0000] | +0.0000 [+0.0000, +0.0000] | -8.0000 [-15.0000, -2.0000] | +0.0000 [+0.0000, +0.0000] | +0.0246 [+0.0000, +0.0536] |
| s2|ordered_setting|O - B|selection | +0.0043 [+0.0000, +0.0106] | +0.0086 [+0.0000, +0.0325] | +0.0105 [+0.0000, +0.0257] | +0.0000 [+0.0000, +0.0000] | -4.0000 [-10.0000, +0.0000] | -0.0070 [-0.0192, +0.0000] |
| s2|ordered_setting|B - F|selection | -0.0011 [-0.0033, +0.0000] | -0.0086 [-0.0325, +0.0000] | -0.0026 [-0.0082, +0.0000] | +0.0000 [+0.0000, +0.0000] | +1.0000 [+0.0000, +3.0000] | +0.0106 [+0.0000, +0.0232] |
| s2|ordered_setting|O - F|selection | +0.0032 [+0.0000, +0.0092] | +0.0000 [+0.0000, +0.0000] | +0.0079 [+0.0000, +0.0226] | +0.0000 [+0.0000, +0.0000] | -3.0000 [-9.0000, +0.0000] | +0.0035 [+0.0000, +0.0114] |
| s0|b3|O - B|calibration | +0.0010 [+0.0000, +0.0031] | +0.0000 [+0.0000, +0.0000] | +0.0025 [+0.0000, +0.0079] | +0.0000 [+0.0000, +0.0000] | -1.0000 [-3.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| s0|b3|B - F|calibration | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0025 [+0.0000, +0.0078] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| s0|b3|O - F|calibration | +0.0010 [+0.0000, +0.0031] | +0.0000 [+0.0000, +0.0000] | +0.0050 [+0.0000, +0.0130] | +0.0000 [+0.0000, +0.0000] | -1.0000 [-3.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| s0|b3|O - frozen Ordered|calibration | -0.0039 [-0.0109, +0.0020] | -0.0068 [-0.0222, +0.0000] | -0.0025 [-0.0104, +0.0051] | +1.0000 [-3.0000, +6.0000] | +3.0000 [-1.0000, +7.0000] | -0.0127 [-0.0335, +0.0000] |
| s0|b3|B - frozen Ordered|calibration | -0.0049 [-0.0121, +0.0011] | -0.0068 [-0.0222, +0.0000] | -0.0050 [-0.0150, +0.0049] | +1.0000 [-3.0000, +6.0000] | +4.0000 [+0.0000, +9.0000] | -0.0127 [-0.0335, +0.0000] |
| s0|b3|F - A01 Linear|calibration | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| s0|b3|F - frozen Ordered|calibration | -0.0049 [-0.0121, +0.0011] | -0.0068 [-0.0222, +0.0000] | -0.0075 [-0.0184, +0.0026] | +1.0000 [-3.0000, +6.0000] | +4.0000 [+0.0000, +9.0000] | -0.0127 [-0.0335, +0.0000] |
| s0|argmax|O - B|calibration | -0.0054 [-0.0150, +0.0011] | +0.0000 [+0.0000, +0.0000] | +0.0025 [+0.0000, +0.0077] | +6.0000 [+0.0000, +14.0000] | +0.0000 [-3.0000, +3.0000] | -0.0084 [-0.0275, +0.0000] |
| s0|argmax|B - F|calibration | +0.0100 [+0.0000, +0.0229] | +0.0000 [+0.0000, +0.0000] | +0.0025 [+0.0000, +0.0078] | -9.0000 [-20.0000, +0.0000] | -2.0000 [-5.0000, +0.0000] | -0.0042 [-0.0310, +0.0198] |
| s0|argmax|O - F|calibration | +0.0046 [+0.0000, +0.0132] | +0.0000 [+0.0000, +0.0000] | +0.0050 [+0.0000, +0.0125] | -3.0000 [-9.0000, +0.0000] | -2.0000 [-5.0000, +0.0000] | -0.0127 [-0.0340, +0.0000] |
| s0|ordered_setting|O - B|calibration | +0.0020 [+0.0000, +0.0062] | +0.0000 [+0.0000, +0.0000] | +0.0050 [+0.0000, +0.0158] | +0.0000 [+0.0000, +0.0000] | -2.0000 [-6.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| s0|ordered_setting|B - F|calibration | +0.0040 [+0.0000, +0.0116] | +0.0000 [+0.0000, +0.0000] | +0.0075 [+0.0000, +0.0177] | -1.0000 [-3.0000, +0.0000] | -3.0000 [-8.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| s0|ordered_setting|O - F|calibration | +0.0059 [+0.0000, +0.0142] | +0.0000 [+0.0000, +0.0000] | +0.0126 [+0.0024, +0.0274] | -1.0000 [-3.0000, +0.0000] | -5.0000 [-12.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| s1|b3|O - B|calibration | -0.0001 [-0.0039, +0.0037] | +0.0000 [+0.0000, +0.0000] | -0.0075 [-0.0167, +0.0000] | -3.0000 [-7.0000, +0.0000] | +3.0000 [+0.0000, +7.0000] | -0.0042 [-0.0146, +0.0000] |
| s1|b3|B - F|calibration | +0.0010 [-0.0019, +0.0044] | +0.0000 [+0.0000, +0.0000] | +0.0050 [+0.0000, +0.0127] | +1.0000 [-2.0000, +5.0000] | -2.0000 [-5.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| s1|b3|O - F|calibration | +0.0009 [-0.0021, +0.0039] | +0.0000 [+0.0000, +0.0000] | -0.0025 [-0.0079, +0.0000] | -2.0000 [-5.0000, +0.0000] | +1.0000 [+0.0000, +3.0000] | -0.0042 [-0.0146, +0.0000] |
| s1|b3|O - frozen Ordered|calibration | -0.0049 [-0.0109, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0101 [-0.0209, -0.0024] | +1.0000 [-3.0000, +5.0000] | +4.0000 [+1.0000, +8.0000] | -0.0042 [-0.0146, +0.0000] |
| s1|b3|B - frozen Ordered|calibration | -0.0048 [-0.0102, -0.0009] | +0.0000 [+0.0000, +0.0000] | -0.0025 [-0.0078, +0.0000] | +4.0000 [+1.0000, +8.0000] | +1.0000 [+0.0000, +3.0000] | +0.0000 [+0.0000, +0.0000] |
| s1|b3|F - A01 Linear|calibration | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| s1|b3|F - frozen Ordered|calibration | -0.0058 [-0.0118, -0.0001] | +0.0000 [+0.0000, +0.0000] | -0.0075 [-0.0170, +0.0000] | +3.0000 [-2.0000, +8.0000] | +3.0000 [+0.0000, +7.0000] | +0.0000 [+0.0000, +0.0000] |
| s1|argmax|O - B|calibration | +0.0000 [+0.0000, +0.0000] | -0.0068 [-0.0191, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0042 [+0.0000, +0.0138] |
| s1|argmax|B - F|calibration | +0.0055 [+0.0000, +0.0134] | +0.0197 [+0.0000, +0.0392] | +0.0000 [+0.0000, +0.0000] | -5.0000 [-12.0000, +0.0000] | -1.0000 [-3.0000, +0.0000] | +0.0042 [-0.0094, +0.0198] |
| s1|argmax|O - F|calibration | +0.0055 [+0.0000, +0.0134] | +0.0129 [+0.0000, +0.0325] | +0.0000 [+0.0000, +0.0000] | -5.0000 [-12.0000, +0.0000] | -1.0000 [-3.0000, +0.0000] | +0.0084 [-0.0090, +0.0312] |
| s1|ordered_setting|O - B|calibration | +0.0050 [+0.0010, +0.0108] | +0.0000 [+0.0000, +0.0000] | +0.0126 [+0.0025, +0.0270] | +0.0000 [+0.0000, +0.0000] | -5.0000 [-11.0000, -1.0000] | +0.0000 [+0.0000, +0.0000] |
| s1|ordered_setting|B - F|calibration | -0.0011 [-0.0050, +0.0019] | +0.0000 [+0.0000, +0.0000] | -0.0050 [-0.0128, +0.0000] | -1.0000 [-3.0000, +0.0000] | +2.0000 [+0.0000, +5.0000] | +0.0000 [+0.0000, +0.0000] |
| s1|ordered_setting|O - F|calibration | +0.0039 [+0.0009, +0.0081] | +0.0000 [+0.0000, +0.0000] | +0.0075 [+0.0000, +0.0175] | -1.0000 [-3.0000, +0.0000] | -3.0000 [-7.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| s2|b3|O - B|calibration | +0.0010 [+0.0000, +0.0030] | -0.0068 [-0.0191, +0.0000] | +0.0025 [+0.0000, +0.0077] | +0.0000 [+0.0000, +0.0000] | -1.0000 [-3.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| s2|b3|B - F|calibration | +0.0074 [-0.0001, +0.0161] | +0.0068 [+0.0000, +0.0191] | -0.0025 [-0.0121, +0.0052] | -10.0000 [-17.0000, -4.0000] | +2.0000 [-2.0000, +6.0000] | -0.0253 [-0.0742, +0.0000] |
| s2|b3|O - F|calibration | +0.0084 [+0.0013, +0.0168] | +0.0000 [+0.0000, +0.0000] | +0.0000 [-0.0076, +0.0074] | -10.0000 [-17.0000, -4.0000] | +1.0000 [-2.0000, +4.0000] | -0.0253 [-0.0742, +0.0000] |
| s2|b3|O - frozen Ordered|calibration | +0.0019 [-0.0019, +0.0058] | +0.0000 [+0.0000, +0.0000] | -0.0075 [-0.0167, +0.0000] | -3.0000 [-7.0000, +0.0000] | +1.0000 [+0.0000, +3.0000] | +0.0042 [+0.0000, +0.0139] |
| s2|b3|B - frozen Ordered|calibration | +0.0009 [-0.0031, +0.0052] | +0.0068 [+0.0000, +0.0191] | -0.0101 [-0.0205, -0.0024] | -3.0000 [-7.0000, +0.0000] | +2.0000 [+0.0000, +5.0000] | +0.0042 [+0.0000, +0.0139] |
| s2|b3|F - A01 Linear|calibration | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| s2|b3|F - frozen Ordered|calibration | -0.0066 [-0.0144, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0075 [-0.0166, +0.0000] | +7.0000 [+1.0000, +14.0000] | +0.0000 [-3.0000, +3.0000] | +0.0295 [+0.0000, +0.0783] |
| s2|argmax|O - B|calibration | -0.0066 [-0.0188, +0.0028] | -0.0068 [-0.0234, +0.0000] | -0.0025 [-0.0078, +0.0000] | +4.0000 [-4.0000, +14.0000] | +3.0000 [+0.0000, +7.0000] | -0.0169 [-0.0408, +0.0000] |
| s2|argmax|B - F|calibration | +0.0037 [-0.0046, +0.0133] | +0.0134 [+0.0000, +0.0298] | +0.0050 [-0.0050, +0.0157] | -3.0000 [-11.0000, +4.0000] | -1.0000 [-4.0000, +2.0000] | +0.0253 [+0.0000, +0.0604] |
| s2|argmax|O - F|calibration | -0.0029 [-0.0128, +0.0037] | +0.0066 [+0.0000, +0.0188] | +0.0025 [-0.0054, +0.0118] | +1.0000 [-5.0000, +9.0000] | +2.0000 [+0.0000, +5.0000] | +0.0084 [-0.0110, +0.0383] |
| s2|ordered_setting|O - B|calibration | +0.0050 [+0.0000, +0.0127] | -0.0068 [-0.0191, +0.0000] | +0.0050 [+0.0000, +0.0128] | -1.0000 [-3.0000, +0.0000] | -4.0000 [-9.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| s2|ordered_setting|B - F|calibration | -0.0001 [-0.0043, +0.0050] | +0.0134 [+0.0000, +0.0298] | +0.0000 [-0.0074, +0.0075] | -1.0000 [-3.0000, +0.0000] | +1.0000 [-2.0000, +5.0000] | +0.0000 [+0.0000, +0.0000] |
| s2|ordered_setting|O - F|calibration | +0.0049 [+0.0000, +0.0131] | +0.0066 [+0.0000, +0.0229] | +0.0050 [+0.0000, +0.0129] | -2.0000 [-5.0000, +0.0000] | -3.0000 [-8.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |

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
| b3|F - A01 Linear|multi segment F1@50 | no consistent difference detected | no consistent difference detected |
| b3|F - A01 Linear|native macro-F1 | no consistent difference detected | no consistent difference detected |
| b3|F - A01 Linear|multi boundary recall | no consistent difference detected | no consistent difference detected |
| b3|F - A01 Linear|multi FP segments | no consistent difference detected | no consistent difference detected |
| b3|F - A01 Linear|multi FN segments | no consistent difference detected | no consistent difference detected |
| b3|F - A01 Linear|native cuts per clip | no consistent difference detected | no consistent difference detected |
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

## Reference segments recovered and lost (a against b)

| seed | pair | half | population | TP a | FP a | FN a | TP b | FP b | FN b | recovered | recovered sessions | lost | lost sessions | recovered <=16 | lost <=16 | recovered X-Y-X | lost X-Y-X | new unmatched pred | removed unmatched pred |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0|O vs B|selection | native | 276 | 24 | 8 | 278 | 22 | 6 | 0 | 0 | 2 | 2 | 0 | 0 | 0 | 0 | 3 | 1 |
| s0|O vs B|selection | pairs | 201 | 7 | 6 | 201 | 7 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| s0|O vs B|selection | multi | 478 | 25 | 24 | 475 | 25 | 27 | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 3 |
| s0|O vs F|selection | native | 276 | 24 | 8 | 277 | 20 | 7 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 7 | 3 |
| s0|O vs F|selection | pairs | 201 | 7 | 6 | 201 | 7 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| s0|O vs F|selection | multi | 478 | 25 | 24 | 476 | 27 | 26 | 3 | 3 | 1 | 1 | 0 | 0 | 0 | 0 | 6 | 8 |
| s0|O vs frozen Ordered|selection | native | 276 | 24 | 8 | 281 | 15 | 3 | 0 | 0 | 5 | 4 | 0 | 0 | 0 | 0 | 13 | 4 |
| s0|O vs frozen Ordered|selection | pairs | 201 | 7 | 6 | 200 | 9 | 7 | 2 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 3 |
| s0|O vs frozen Ordered|selection | multi | 478 | 25 | 24 | 473 | 33 | 29 | 9 | 7 | 4 | 5 | 3 | 2 | 0 | 2 | 14 | 22 |
| s0|B vs F|selection | native | 278 | 22 | 6 | 277 | 20 | 7 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 4 |
| s0|B vs F|selection | pairs | 201 | 7 | 6 | 201 | 7 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| s0|B vs F|selection | multi | 475 | 25 | 27 | 476 | 27 | 26 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 4 | 6 |
| s0|B vs frozen Ordered|selection | native | 278 | 22 | 6 | 281 | 15 | 3 | 0 | 0 | 3 | 3 | 0 | 0 | 0 | 0 | 11 | 4 |
| s0|B vs frozen Ordered|selection | pairs | 201 | 7 | 6 | 200 | 9 | 7 | 2 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 3 |
| s0|B vs frozen Ordered|selection | multi | 475 | 25 | 27 | 473 | 33 | 29 | 6 | 5 | 4 | 5 | 3 | 2 | 0 | 2 | 15 | 23 |
| s0|F vs frozen Ordered|selection | native | 277 | 20 | 7 | 281 | 15 | 3 | 0 | 0 | 4 | 3 | 0 | 0 | 0 | 0 | 12 | 7 |
| s0|F vs frozen Ordered|selection | pairs | 201 | 7 | 6 | 200 | 9 | 7 | 2 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 3 |
| s0|F vs frozen Ordered|selection | multi | 476 | 27 | 26 | 473 | 33 | 29 | 7 | 6 | 4 | 5 | 3 | 2 | 0 | 2 | 16 | 22 |
| s0|F vs A01 Linear|selection | native | 277 | 20 | 7 | 277 | 20 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| s0|F vs A01 Linear|selection | pairs | 201 | 7 | 6 | 201 | 7 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| s0|F vs A01 Linear|selection | multi | 476 | 27 | 26 | 476 | 27 | 26 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| s1|O vs B|selection | native | 278 | 14 | 6 | 277 | 17 | 7 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| s1|O vs B|selection | pairs | 198 | 12 | 9 | 199 | 12 | 8 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 |
| s1|O vs B|selection | multi | 477 | 18 | 25 | 477 | 20 | 25 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 3 |
| s1|O vs F|selection | native | 278 | 14 | 6 | 278 | 17 | 6 | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 2 | 5 |
| s1|O vs F|selection | pairs | 198 | 12 | 9 | 199 | 12 | 8 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 |
| s1|O vs F|selection | multi | 477 | 18 | 25 | 477 | 18 | 25 | 2 | 2 | 2 | 1 | 2 | 1 | 0 | 0 | 2 | 2 |
| s1|O vs frozen Ordered|selection | native | 278 | 14 | 6 | 280 | 15 | 4 | 0 | 0 | 2 | 2 | 0 | 1 | 0 | 0 | 6 | 7 |
| s1|O vs frozen Ordered|selection | pairs | 198 | 12 | 9 | 200 | 10 | 7 | 0 | 0 | 2 | 2 | 0 | 2 | 0 | 0 | 2 | 0 |
| s1|O vs frozen Ordered|selection | multi | 477 | 18 | 25 | 484 | 22 | 18 | 0 | 0 | 7 | 6 | 0 | 4 | 0 | 0 | 4 | 8 |
| s1|B vs F|selection | native | 277 | 17 | 7 | 278 | 17 | 6 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 2 | 2 |
| s1|B vs F|selection | pairs | 199 | 12 | 8 | 199 | 12 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| s1|B vs F|selection | multi | 477 | 20 | 25 | 477 | 18 | 25 | 2 | 2 | 2 | 1 | 2 | 1 | 0 | 0 | 5 | 3 |
| s1|B vs frozen Ordered|selection | native | 277 | 17 | 7 | 280 | 15 | 4 | 0 | 0 | 3 | 2 | 0 | 1 | 0 | 0 | 9 | 7 |
| s1|B vs frozen Ordered|selection | pairs | 199 | 12 | 8 | 200 | 10 | 7 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 2 | 0 |
| s1|B vs frozen Ordered|selection | multi | 477 | 20 | 25 | 484 | 22 | 18 | 0 | 0 | 7 | 5 | 0 | 4 | 0 | 0 | 6 | 8 |
| s1|F vs frozen Ordered|selection | native | 278 | 17 | 6 | 280 | 15 | 4 | 0 | 0 | 2 | 1 | 0 | 0 | 0 | 0 | 9 | 7 |
| s1|F vs frozen Ordered|selection | pairs | 199 | 12 | 8 | 200 | 10 | 7 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 2 | 0 |
| s1|F vs frozen Ordered|selection | multi | 477 | 18 | 25 | 484 | 22 | 18 | 0 | 0 | 7 | 5 | 0 | 5 | 0 | 0 | 5 | 9 |
| s1|F vs A01 Linear|selection | native | 278 | 17 | 6 | 278 | 17 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| s1|F vs A01 Linear|selection | pairs | 199 | 12 | 8 | 199 | 12 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| s1|F vs A01 Linear|selection | multi | 477 | 18 | 25 | 477 | 18 | 25 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| s2|O vs B|selection | native | 278 | 19 | 6 | 277 | 18 | 7 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 4 |
| s2|O vs B|selection | pairs | 200 | 11 | 7 | 200 | 11 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| s2|O vs B|selection | multi | 480 | 17 | 22 | 481 | 17 | 21 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 4 | 4 |
| s2|O vs F|selection | native | 278 | 19 | 6 | 277 | 31 | 7 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 15 |
| s2|O vs F|selection | pairs | 200 | 11 | 7 | 201 | 12 | 6 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 2 |
| s2|O vs F|selection | multi | 480 | 17 | 22 | 483 | 44 | 19 | 3 | 4 | 6 | 6 | 0 | 2 | 0 | 0 | 6 | 33 |
| s2|O vs frozen Ordered|selection | native | 278 | 19 | 6 | 278 | 19 | 6 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 8 | 8 |
| s2|O vs frozen Ordered|selection | pairs | 200 | 11 | 7 | 200 | 11 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 2 |
| s2|O vs frozen Ordered|selection | multi | 480 | 17 | 22 | 487 | 21 | 15 | 1 | 2 | 8 | 7 | 0 | 3 | 0 | 0 | 12 | 16 |
| s2|B vs F|selection | native | 277 | 18 | 7 | 277 | 31 | 7 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 6 | 19 |
| s2|B vs F|selection | pairs | 200 | 11 | 7 | 201 | 12 | 6 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 2 |
| s2|B vs F|selection | multi | 481 | 17 | 21 | 483 | 44 | 19 | 3 | 4 | 5 | 5 | 0 | 2 | 0 | 0 | 8 | 35 |
| s2|B vs frozen Ordered|selection | native | 277 | 18 | 7 | 278 | 19 | 6 | 1 | 1 | 2 | 2 | 0 | 0 | 0 | 0 | 6 | 7 |
| s2|B vs frozen Ordered|selection | pairs | 200 | 11 | 7 | 200 | 11 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 2 |
| s2|B vs frozen Ordered|selection | multi | 481 | 17 | 21 | 487 | 21 | 15 | 1 | 2 | 7 | 7 | 0 | 3 | 0 | 0 | 11 | 15 |
| s2|F vs frozen Ordered|selection | native | 277 | 31 | 7 | 278 | 19 | 6 | 1 | 1 | 2 | 2 | 0 | 0 | 0 | 0 | 16 | 4 |
| s2|F vs frozen Ordered|selection | pairs | 201 | 12 | 6 | 200 | 11 | 7 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 |
| s2|F vs frozen Ordered|selection | multi | 483 | 44 | 19 | 487 | 21 | 15 | 1 | 1 | 5 | 5 | 0 | 1 | 0 | 0 | 32 | 9 |
| s2|F vs A01 Linear|selection | native | 277 | 31 | 7 | 277 | 31 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| s2|F vs A01 Linear|selection | pairs | 201 | 12 | 6 | 201 | 12 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| s2|F vs A01 Linear|selection | multi | 483 | 44 | 19 | 483 | 44 | 19 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| s0|O vs B|calibration | native | 235 | 5 | 2 | 235 | 5 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| s0|O vs B|calibration | pairs | 168 | 3 | 5 | 168 | 3 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| s0|O vs B|calibration | multi | 507 | 12 | 11 | 506 | 12 | 12 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| s0|O vs F|calibration | native | 235 | 5 | 2 | 235 | 5 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| s0|O vs F|calibration | pairs | 168 | 3 | 5 | 167 | 4 | 6 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| s0|O vs F|calibration | multi | 507 | 12 | 11 | 506 | 12 | 12 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| s0|O vs frozen Ordered|calibration | native | 235 | 5 | 2 | 236 | 7 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 3 | 5 |
| s0|O vs frozen Ordered|calibration | pairs | 168 | 3 | 5 | 168 | 1 | 5 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 2 | 0 |
| s0|O vs frozen Ordered|calibration | multi | 507 | 12 | 11 | 510 | 11 | 8 | 1 | 1 | 4 | 4 | 1 | 2 | 0 | 0 | 6 | 5 |
| s0|B vs F|calibration | native | 235 | 5 | 2 | 235 | 5 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| s0|B vs F|calibration | pairs | 168 | 3 | 5 | 167 | 4 | 6 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| s0|B vs F|calibration | multi | 506 | 12 | 12 | 506 | 12 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| s0|B vs frozen Ordered|calibration | native | 235 | 5 | 2 | 236 | 7 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 3 | 5 |
| s0|B vs frozen Ordered|calibration | pairs | 168 | 3 | 5 | 168 | 1 | 5 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 2 | 0 |
| s0|B vs frozen Ordered|calibration | multi | 506 | 12 | 12 | 510 | 11 | 8 | 1 | 1 | 5 | 5 | 1 | 3 | 0 | 0 | 6 | 5 |
| s0|F vs frozen Ordered|calibration | native | 235 | 5 | 2 | 236 | 7 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 3 | 5 |
| s0|F vs frozen Ordered|calibration | pairs | 167 | 4 | 6 | 168 | 1 | 5 | 1 | 1 | 2 | 3 | 1 | 1 | 0 | 0 | 3 | 0 |
| s0|F vs frozen Ordered|calibration | multi | 506 | 12 | 12 | 510 | 11 | 8 | 1 | 1 | 5 | 5 | 1 | 3 | 0 | 0 | 6 | 5 |
| s0|F vs A01 Linear|calibration | native | 235 | 5 | 2 | 235 | 5 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| s0|F vs A01 Linear|calibration | pairs | 167 | 4 | 6 | 167 | 4 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| s0|F vs A01 Linear|calibration | multi | 506 | 12 | 12 | 506 | 12 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| s1|O vs B|calibration | native | 235 | 5 | 2 | 235 | 6 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 2 |
| s1|O vs B|calibration | pairs | 167 | 4 | 6 | 168 | 3 | 5 | 0 | 0 | 1 | 2 | 0 | 0 | 0 | 0 | 1 | 0 |
| s1|O vs B|calibration | multi | 507 | 11 | 11 | 510 | 14 | 8 | 0 | 0 | 3 | 3 | 0 | 2 | 0 | 0 | 2 | 5 |
| s1|O vs F|calibration | native | 235 | 5 | 2 | 235 | 6 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 2 |
| s1|O vs F|calibration | pairs | 167 | 4 | 6 | 167 | 4 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| s1|O vs F|calibration | multi | 507 | 11 | 11 | 508 | 13 | 10 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 2 |
| s1|O vs frozen Ordered|calibration | native | 235 | 5 | 2 | 235 | 6 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 3 |
| s1|O vs frozen Ordered|calibration | pairs | 167 | 4 | 6 | 167 | 4 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| s1|O vs frozen Ordered|calibration | multi | 507 | 11 | 11 | 511 | 10 | 7 | 0 | 0 | 4 | 4 | 0 | 3 | 0 | 0 | 7 | 6 |
| s1|B vs F|calibration | native | 235 | 6 | 2 | 235 | 6 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| s1|B vs F|calibration | pairs | 168 | 3 | 5 | 167 | 4 | 6 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| s1|B vs F|calibration | multi | 510 | 14 | 8 | 508 | 13 | 10 | 2 | 2 | 0 | 0 | 1 | 0 | 0 | 0 | 4 | 3 |
| s1|B vs frozen Ordered|calibration | native | 235 | 6 | 2 | 235 | 6 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| s1|B vs frozen Ordered|calibration | pairs | 168 | 3 | 5 | 167 | 4 | 6 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| s1|B vs frozen Ordered|calibration | multi | 510 | 14 | 8 | 511 | 10 | 7 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 6 | 2 |
| s1|F vs frozen Ordered|calibration | native | 235 | 6 | 2 | 235 | 6 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| s1|F vs frozen Ordered|calibration | pairs | 167 | 4 | 6 | 167 | 4 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| s1|F vs frozen Ordered|calibration | multi | 508 | 13 | 10 | 511 | 10 | 7 | 0 | 0 | 3 | 3 | 0 | 2 | 0 | 0 | 9 | 6 |
| s1|F vs A01 Linear|calibration | native | 235 | 6 | 2 | 235 | 6 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| s1|F vs A01 Linear|calibration | pairs | 167 | 4 | 6 | 167 | 4 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| s1|F vs A01 Linear|calibration | multi | 508 | 13 | 10 | 508 | 13 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| s2|O vs B|calibration | native | 235 | 6 | 2 | 236 | 5 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 0 |
| s2|O vs B|calibration | pairs | 166 | 4 | 7 | 164 | 5 | 9 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| s2|O vs B|calibration | multi | 507 | 8 | 11 | 506 | 8 | 12 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| s2|O vs F|calibration | native | 235 | 6 | 2 | 234 | 13 | 3 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 |
| s2|O vs F|calibration | pairs | 166 | 4 | 7 | 167 | 4 | 6 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 |
| s2|O vs F|calibration | multi | 507 | 8 | 11 | 508 | 18 | 10 | 1 | 1 | 2 | 2 | 1 | 1 | 0 | 0 | 2 | 12 |
| s2|O vs frozen Ordered|calibration | native | 235 | 6 | 2 | 235 | 5 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| s2|O vs frozen Ordered|calibration | pairs | 166 | 4 | 7 | 166 | 3 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| s2|O vs frozen Ordered|calibration | multi | 507 | 8 | 11 | 508 | 11 | 10 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 2 | 5 |
| s2|B vs F|calibration | native | 236 | 5 | 1 | 234 | 13 | 3 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 |
| s2|B vs F|calibration | pairs | 164 | 5 | 9 | 167 | 4 | 6 | 0 | 0 | 3 | 3 | 0 | 1 | 0 | 0 | 1 | 0 |
| s2|B vs F|calibration | multi | 506 | 8 | 12 | 508 | 18 | 10 | 1 | 1 | 3 | 3 | 1 | 2 | 0 | 0 | 2 | 12 |
| s2|B vs frozen Ordered|calibration | native | 236 | 5 | 1 | 235 | 5 | 2 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| s2|B vs frozen Ordered|calibration | pairs | 164 | 5 | 9 | 166 | 3 | 7 | 0 | 0 | 2 | 2 | 0 | 0 | 0 | 0 | 2 | 0 |
| s2|B vs frozen Ordered|calibration | multi | 506 | 8 | 12 | 508 | 11 | 10 | 0 | 0 | 2 | 2 | 0 | 2 | 0 | 0 | 2 | 5 |
| s2|F vs frozen Ordered|calibration | native | 234 | 13 | 3 | 235 | 5 | 2 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 8 | 0 |
| s2|F vs frozen Ordered|calibration | pairs | 167 | 4 | 6 | 166 | 3 | 7 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 |
| s2|F vs frozen Ordered|calibration | multi | 508 | 18 | 10 | 508 | 11 | 10 | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 10 | 3 |
| s2|F vs A01 Linear|calibration | native | 234 | 13 | 3 | 234 | 13 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| s2|F vs A01 Linear|calibration | pairs | 167 | 4 | 6 | 167 | 4 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| s2|F vs A01 Linear|calibration | multi | 508 | 18 | 10 | 508 | 18 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Goal

| group | selection mean | calibration mean | constraint failures | met |
|---|---|---|---|---|
| F | 0.9476 | 0.9760 | 0 | False |
| B | 0.9550 | 0.9788 | 2 | False |
| O | 0.9564 | 0.9794 | 2 | False |
| frozen Ordered | 0.9544 | 0.9817 | 1 | False |

## Readings

```
[
 "uncertain_or_no_qualifying_gain",
 "end"
]
```
