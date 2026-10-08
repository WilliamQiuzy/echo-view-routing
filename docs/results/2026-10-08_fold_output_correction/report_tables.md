# Fold-output correction: tables

## Systems

| encoder | system | half | segment F1@50 | boundary P | boundary R | native macro-F1 | cuts per clip | same-class cut rate | A5C read as A4C | A4C read as A5C | raw argmax segment F1@50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| main (B0 v3) | baseline | selection | 0.9319 | 0.9787 | 0.9634 | 0.9650 | 0.0141 | 0.0261 | 3 | 0 | 0.5278 |
| main (B0 v3) | baseline | calibration | 0.9284 | 0.9413 | 0.9673 | 0.9572 | 0.0506 | 0.0348 | 5 | 0 | 0.5769 |
| main (B0 v3) | T-in_s0 | selection | 0.8893 | 0.8849 | 0.9660 | 0.9650 | 0.0563 | 0.0696 | 3 | 0 | 0.8754 |
| main (B0 v3) | T-in_s0 | calibration | 0.9234 | 0.9212 | 0.9698 | 0.9660 | 0.0844 | 0.0597 | 4 | 0 | 0.9290 |
| main (B0 v3) | T-out_s0 | selection | 0.8864 | 0.9643 | 0.9188 | 0.9005 | 0.0211 | 0.0130 | 0 | 12 | 0.8597 |
| main (B0 v3) | T-out_s0 | calibration | 0.9066 | 0.9538 | 0.9347 | 0.9624 | 0.0211 | 0.0199 | 0 | 2 | 0.8782 |
| main (B0 v3) | P-out_s0 | selection | 0.8512 | 0.9971 | 0.9031 | 0.7965 | 0.0000 | 0.0000 | 0 | 44 | 0.4490 |
| main (B0 v3) | P-out_s0 | calibration | 0.8020 | 0.9972 | 0.8945 | 0.8089 | 0.0042 | 0.0050 | 0 | 37 | 0.4340 |
| main (B0 v3) | T-in_s1 | selection | 0.8882 | 0.9091 | 0.9686 | 0.9414 | 0.0528 | 0.0783 | 4 | 0 | 0.8785 |
| main (B0 v3) | T-in_s1 | calibration | 0.9269 | 0.9277 | 0.9673 | 0.9660 | 0.0591 | 0.0498 | 4 | 0 | 0.9280 |
| main (B0 v3) | T-out_s1 | selection | 0.8736 | 0.9619 | 0.9241 | 0.9055 | 0.0106 | 0.0130 | 0 | 11 | 0.8444 |
| main (B0 v3) | T-out_s1 | calibration | 0.8636 | 0.9554 | 0.9146 | 0.9499 | 0.0295 | 0.0249 | 0 | 4 | 0.8299 |
| main (B0 v3) | P-out_s1 | selection | 0.8574 | 0.9971 | 0.9031 | 0.7995 | 0.0000 | 0.0000 | 0 | 43 | 0.4448 |
| main (B0 v3) | P-out_s1 | calibration | 0.8040 | 0.9972 | 0.8945 | 0.8136 | 0.0042 | 0.0050 | 0 | 36 | 0.4312 |
| main (B0 v3) | T-in_s2 | selection | 0.8893 | 0.9093 | 0.9712 | 0.9486 | 0.0634 | 0.0783 | 4 | 0 | 0.8776 |
| main (B0 v3) | T-in_s2 | calibration | 0.9235 | 0.9169 | 0.9698 | 0.9660 | 0.1097 | 0.0547 | 4 | 0 | 0.9254 |
| main (B0 v3) | T-out_s2 | selection | 0.8911 | 0.9723 | 0.9188 | 0.9219 | 0.0035 | 0.0174 | 0 | 8 | 0.8516 |
| main (B0 v3) | T-out_s2 | calibration | 0.8852 | 0.9659 | 0.9246 | 0.9624 | 0.0127 | 0.0348 | 0 | 2 | 0.8507 |
| main (B0 v3) | P-out_s2 | selection | 0.8595 | 0.9971 | 0.9031 | 0.8026 | 0.0000 | 0.0000 | 0 | 42 | 0.4426 |
| main (B0 v3) | P-out_s2 | calibration | 0.8060 | 0.9972 | 0.8945 | 0.8136 | 0.0042 | 0.0050 | 0 | 36 | 0.4282 |
| auxiliary (fold 0) | baseline | selection | 0.8604 | 0.8825 | 0.9634 | 0.9308 | 0.0634 | 0.1130 | 4 | 2 | 0.6074 |
| auxiliary (fold 0) | baseline | calibration | 0.8945 | 0.8915 | 0.9497 | 0.9530 | 0.0717 | 0.0498 | 5 | 1 | 0.5864 |
| auxiliary (fold 0) | T-in_s0 | selection | 0.8580 | 0.8862 | 0.9581 | 0.9160 | 0.0599 | 0.1261 | 5 | 2 | 0.8514 |
| auxiliary (fold 0) | T-in_s0 | calibration | 0.8993 | 0.8802 | 0.9598 | 0.9409 | 0.0633 | 0.0746 | 7 | 0 | 0.8941 |
| auxiliary (fold 0) | T-out_s0 | selection | 0.8860 | 0.9173 | 0.9293 | 0.9157 | 0.0141 | 0.0652 | 4 | 4 | 0.8729 |
| auxiliary (fold 0) | T-out_s0 | calibration | 0.9487 | 0.9722 | 0.9648 | 0.9633 | 0.0042 | 0.0299 | 3 | 2 | 0.9343 |
| auxiliary (fold 0) | P-out_s0 | selection | 0.8961 | 0.9806 | 0.9241 | 0.9031 | 0.0000 | 0.0304 | 2 | 10 | 0.5389 |
| auxiliary (fold 0) | P-out_s0 | calibration | 0.9228 | 0.9892 | 0.9246 | 0.9407 | 0.0042 | 0.0100 | 2 | 7 | 0.5473 |
| auxiliary (fold 0) | T-in_s1 | selection | 0.8694 | 0.9059 | 0.9581 | 0.9085 | 0.0387 | 0.1087 | 5 | 3 | 0.8558 |
| auxiliary (fold 0) | T-in_s1 | calibration | 0.9047 | 0.8843 | 0.9598 | 0.9506 | 0.0675 | 0.0697 | 6 | 0 | 0.8914 |
| auxiliary (fold 0) | T-out_s1 | selection | 0.8974 | 0.9235 | 0.9476 | 0.9157 | 0.0246 | 0.0609 | 4 | 4 | 0.8780 |
| auxiliary (fold 0) | T-out_s1 | calibration | 0.9534 | 0.9745 | 0.9598 | 0.9568 | 0.0127 | 0.0100 | 3 | 3 | 0.9355 |
| auxiliary (fold 0) | P-out_s1 | selection | 0.8961 | 0.9806 | 0.9241 | 0.9031 | 0.0000 | 0.0304 | 2 | 10 | 0.5373 |
| auxiliary (fold 0) | P-out_s1 | calibration | 0.9228 | 0.9892 | 0.9246 | 0.9407 | 0.0042 | 0.0100 | 2 | 7 | 0.5534 |
| auxiliary (fold 0) | T-in_s2 | selection | 0.8705 | 0.9037 | 0.9581 | 0.9160 | 0.0563 | 0.1174 | 5 | 2 | 0.8577 |
| auxiliary (fold 0) | T-in_s2 | calibration | 0.9007 | 0.8860 | 0.9573 | 0.9409 | 0.0675 | 0.0746 | 7 | 0 | 0.8930 |
| auxiliary (fold 0) | T-out_s2 | selection | 0.8948 | 0.9139 | 0.9450 | 0.9157 | 0.0141 | 0.0783 | 4 | 4 | 0.8820 |
| auxiliary (fold 0) | T-out_s2 | calibration | 0.9574 | 0.9722 | 0.9673 | 0.9633 | 0.0042 | 0.0299 | 3 | 2 | 0.9309 |
| auxiliary (fold 0) | P-out_s2 | selection | 0.8961 | 0.9806 | 0.9241 | 0.9031 | 0.0000 | 0.0304 | 2 | 10 | 0.5383 |
| auxiliary (fold 0) | P-out_s2 | calibration | 0.9267 | 0.9892 | 0.9246 | 0.9407 | 0.0042 | 0.0100 | 2 | 7 | 0.5528 |
| auxiliary (fold 1) | baseline | selection | 0.8594 | 0.8949 | 0.9581 | 0.9157 | 0.0458 | 0.1000 | 4 | 4 | 0.5061 |
| auxiliary (fold 1) | baseline | calibration | 0.8627 | 0.8753 | 0.9523 | 0.9409 | 0.1013 | 0.0448 | 7 | 0 | 0.5332 |
| auxiliary (fold 1) | T-in_s0 | selection | 0.8475 | 0.8768 | 0.9503 | 0.9157 | 0.0669 | 0.1087 | 4 | 4 | 0.8343 |
| auxiliary (fold 1) | T-in_s0 | calibration | 0.8643 | 0.8699 | 0.9573 | 0.9305 | 0.0970 | 0.0697 | 8 | 0 | 0.8574 |
| auxiliary (fold 1) | T-out_s0 | selection | 0.8933 | 0.9423 | 0.9398 | 0.9216 | 0.0106 | 0.0609 | 3 | 5 | 0.8852 |
| auxiliary (fold 1) | T-out_s0 | calibration | 0.9177 | 0.9544 | 0.9472 | 0.9311 | 0.0295 | 0.0348 | 6 | 3 | 0.9058 |
| auxiliary (fold 1) | P-out_s0 | selection | 0.9212 | 0.9864 | 0.9503 | 0.8924 | 0.0000 | 0.0217 | 3 | 10 | 0.4962 |
| auxiliary (fold 1) | P-out_s0 | calibration | 0.9214 | 0.9816 | 0.9372 | 0.9387 | 0.0295 | 0.0100 | 3 | 6 | 0.4619 |
| auxiliary (fold 1) | T-in_s1 | selection | 0.8444 | 0.8741 | 0.9450 | 0.9157 | 0.0528 | 0.1043 | 4 | 4 | 0.8354 |
| auxiliary (fold 1) | T-in_s1 | calibration | 0.8659 | 0.8739 | 0.9573 | 0.9305 | 0.0886 | 0.0697 | 8 | 0 | 0.8556 |
| auxiliary (fold 1) | T-out_s1 | selection | 0.8913 | 0.9231 | 0.9424 | 0.9216 | 0.0211 | 0.0696 | 3 | 5 | 0.8861 |
| auxiliary (fold 1) | T-out_s1 | calibration | 0.9188 | 0.9520 | 0.9472 | 0.9339 | 0.0253 | 0.0597 | 5 | 4 | 0.8885 |
| auxiliary (fold 1) | P-out_s1 | selection | 0.9212 | 0.9864 | 0.9503 | 0.9031 | 0.0000 | 0.0217 | 3 | 8 | 0.4991 |
| auxiliary (fold 1) | P-out_s1 | calibration | 0.9205 | 0.9790 | 0.9372 | 0.9387 | 0.0295 | 0.0100 | 3 | 6 | 0.4686 |
| auxiliary (fold 1) | T-in_s2 | selection | 0.8475 | 0.8768 | 0.9503 | 0.9089 | 0.0599 | 0.1087 | 4 | 5 | 0.8386 |
| auxiliary (fold 1) | T-in_s2 | calibration | 0.8671 | 0.8936 | 0.9497 | 0.9305 | 0.0970 | 0.0697 | 8 | 0 | 0.8529 |
| auxiliary (fold 1) | T-out_s2 | selection | 0.8986 | 0.9349 | 0.9398 | 0.9089 | 0.0141 | 0.0652 | 4 | 5 | 0.8922 |
| auxiliary (fold 1) | T-out_s2 | calibration | 0.9168 | 0.9545 | 0.9497 | 0.9311 | 0.0169 | 0.0597 | 6 | 3 | 0.8960 |
| auxiliary (fold 1) | P-out_s2 | selection | 0.9212 | 0.9864 | 0.9503 | 0.9031 | 0.0000 | 0.0217 | 3 | 8 | 0.4991 |
| auxiliary (fold 1) | P-out_s2 | calibration | 0.9205 | 0.9790 | 0.9372 | 0.9387 | 0.0295 | 0.0100 | 3 | 6 | 0.4640 |
| auxiliary (fold 2) | baseline | selection | 0.8363 | 0.8936 | 0.9450 | 0.9159 | 0.0387 | 0.0826 | 6 | 0 | 0.5714 |
| auxiliary (fold 2) | baseline | calibration | 0.8428 | 0.9373 | 0.9397 | 0.8895 | 0.0675 | 0.0647 | 11 | 1 | 0.4884 |
| auxiliary (fold 2) | T-in_s0 | selection | 0.8307 | 0.8842 | 0.9398 | 0.8965 | 0.0528 | 0.1000 | 7 | 0 | 0.8258 |
| auxiliary (fold 2) | T-in_s0 | calibration | 0.8332 | 0.8865 | 0.9422 | 0.9017 | 0.0675 | 0.0746 | 10 | 1 | 0.8167 |
| auxiliary (fold 2) | T-out_s0 | selection | 0.8900 | 0.9276 | 0.9398 | 0.9361 | 0.0246 | 0.0783 | 3 | 3 | 0.8780 |
| auxiliary (fold 2) | T-out_s0 | calibration | 0.9061 | 0.9443 | 0.9372 | 0.9177 | 0.0295 | 0.0448 | 8 | 2 | 0.8969 |
| auxiliary (fold 2) | P-out_s0 | selection | 0.9034 | 0.9778 | 0.9241 | 0.9085 | 0.0000 | 0.0304 | 2 | 9 | 0.5058 |
| auxiliary (fold 2) | P-out_s0 | calibration | 0.9141 | 0.9840 | 0.9271 | 0.9365 | 0.0169 | 0.0199 | 4 | 5 | 0.4610 |
| auxiliary (fold 2) | T-in_s1 | selection | 0.8490 | 0.9182 | 0.9398 | 0.9159 | 0.0352 | 0.0826 | 6 | 0 | 0.8242 |
| auxiliary (fold 2) | T-in_s1 | calibration | 0.8357 | 0.9012 | 0.9397 | 0.8895 | 0.0591 | 0.0746 | 11 | 1 | 0.8232 |
| auxiliary (fold 2) | T-out_s1 | selection | 0.9030 | 0.9356 | 0.9503 | 0.9361 | 0.0211 | 0.0696 | 3 | 3 | 0.8745 |
| auxiliary (fold 2) | T-out_s1 | calibration | 0.9037 | 0.9400 | 0.9447 | 0.9117 | 0.0295 | 0.0498 | 8 | 3 | 0.8918 |
| auxiliary (fold 2) | P-out_s1 | selection | 0.9054 | 0.9778 | 0.9241 | 0.9085 | 0.0000 | 0.0304 | 2 | 9 | 0.5140 |
| auxiliary (fold 2) | P-out_s1 | calibration | 0.9102 | 0.9840 | 0.9271 | 0.9365 | 0.0169 | 0.0199 | 4 | 5 | 0.4599 |
| auxiliary (fold 2) | T-in_s2 | selection | 0.8475 | 0.9201 | 0.9346 | 0.9135 | 0.0352 | 0.0826 | 6 | 0 | 0.8285 |
| auxiliary (fold 2) | T-in_s2 | calibration | 0.8311 | 0.8966 | 0.9372 | 0.9017 | 0.0675 | 0.0746 | 10 | 1 | 0.8275 |
| auxiliary (fold 2) | T-out_s2 | selection | 0.8972 | 0.9359 | 0.9555 | 0.9530 | 0.0106 | 0.0826 | 3 | 1 | 0.8740 |
| auxiliary (fold 2) | T-out_s2 | calibration | 0.8977 | 0.9422 | 0.9422 | 0.9177 | 0.0211 | 0.0498 | 8 | 2 | 0.8802 |
| auxiliary (fold 2) | P-out_s2 | selection | 0.9054 | 0.9778 | 0.9241 | 0.9085 | 0.0000 | 0.0304 | 2 | 9 | 0.5210 |
| auxiliary (fold 2) | P-out_s2 | calibration | 0.9102 | 0.9840 | 0.9271 | 0.9365 | 0.0169 | 0.0199 | 4 | 5 | 0.4622 |

## Comparisons

| encoder | comparison | d segment F1@50 | native fixed/new |
|---|---|---|---|
| main | s0|T-out - T-in|selection | -0.0029 [-0.0467, +0.0440] | 3/14 |
| main | s0|T-out - T-in|calibration | -0.0168 [-0.0486, +0.0173] | 4/4 |
| main | s0|T-out - (P-out + B3)|selection | +0.0352 [+0.0101, +0.0599] | 32/1 |
| main | s0|T-out - (P-out + B3)|calibration | +0.1046 [+0.0770, +0.1353] | 35/2 |
| main | s0|T-out - (B0 + B3)|selection | -0.0455 [-0.0822, -0.0090] | 3/14 |
| main | s0|T-out - (B0 + B3)|calibration | -0.0218 [-0.0551, +0.0128] | 5/4 |
| main | s1|T-out - T-in|selection | -0.0146 [-0.0610, +0.0325] | 4/12 |
| main | s1|T-out - T-in|calibration | -0.0633 [-0.1018, -0.0223] | 4/6 |
| main | s1|T-out - (P-out + B3)|selection | +0.0162 [-0.0106, +0.0431] | 32/1 |
| main | s1|T-out - (P-out + B3)|calibration | +0.0596 [+0.0339, +0.0884] | 32/2 |
| main | s1|T-out - (B0 + B3)|selection | -0.0583 [-0.0974, -0.0200] | 3/13 |
| main | s1|T-out - (B0 + B3)|calibration | -0.0648 [-0.1026, -0.0258] | 5/6 |
| main | s2|T-out - T-in|selection | +0.0018 [-0.0416, +0.0458] | 5/10 |
| main | s2|T-out - T-in|calibration | -0.0383 [-0.0757, +0.0022] | 4/4 |
| main | s2|T-out - (P-out + B3)|selection | +0.0316 [+0.0084, +0.0555] | 34/1 |
| main | s2|T-out - (P-out + B3)|calibration | +0.0792 [+0.0506, +0.1094] | 34/2 |
| main | s2|T-out - (B0 + B3)|selection | -0.0407 [-0.0760, -0.0053] | 3/10 |
| main | s2|T-out - (B0 + B3)|calibration | -0.0432 [-0.0812, -0.0044] | 5/4 |
| fold 0 | s0|T-out - T-in|selection | +0.0281 [-0.0002, +0.0581] | 1/2 |
| fold 0 | s0|T-out - T-in|calibration | +0.0494 [+0.0201, +0.0790] | 4/2 |
| fold 0 | s0|T-out - (P-out + B3)|selection | -0.0101 [-0.0309, +0.0106] | 6/2 |
| fold 0 | s0|T-out - (P-out + B3)|calibration | +0.0259 [+0.0038, +0.0486] | 5/1 |
| fold 0 | s0|T-out - (B0 + B3)|selection | +0.0256 [-0.0043, +0.0573] | 0/2 |
| fold 0 | s0|T-out - (B0 + B3)|calibration | +0.0542 [+0.0236, +0.0835] | 2/1 |
| fold 0 | s1|T-out - T-in|selection | +0.0280 [-0.0019, +0.0590] | 1/1 |
| fold 0 | s1|T-out - T-in|calibration | +0.0487 [+0.0230, +0.0766] | 3/3 |
| fold 0 | s1|T-out - (P-out + B3)|selection | +0.0013 [-0.0188, +0.0244] | 6/2 |
| fold 0 | s1|T-out - (P-out + B3)|calibration | +0.0306 [+0.0112, +0.0512] | 5/2 |
| fold 0 | s1|T-out - (B0 + B3)|selection | +0.0370 [+0.0073, +0.0682] | 0/2 |
| fold 0 | s1|T-out - (B0 + B3)|calibration | +0.0589 [+0.0323, +0.0872] | 2/2 |
| fold 0 | s2|T-out - T-in|selection | +0.0243 [-0.0026, +0.0534] | 1/2 |
| fold 0 | s2|T-out - T-in|calibration | +0.0567 [+0.0295, +0.0833] | 4/2 |
| fold 0 | s2|T-out - (P-out + B3)|selection | -0.0013 [-0.0222, +0.0224] | 6/2 |
| fold 0 | s2|T-out - (P-out + B3)|calibration | +0.0307 [+0.0120, +0.0518] | 5/1 |
| fold 0 | s2|T-out - (B0 + B3)|selection | +0.0343 [+0.0076, +0.0642] | 0/2 |
| fold 0 | s2|T-out - (B0 + B3)|calibration | +0.0629 [+0.0349, +0.0906] | 2/1 |
| fold 1 | s0|T-out - T-in|selection | +0.0458 [+0.0169, +0.0762] | 1/1 |
| fold 1 | s0|T-out - T-in|calibration | +0.0534 [+0.0159, +0.0943] | 2/3 |
| fold 1 | s0|T-out - (P-out + B3)|selection | -0.0279 [-0.0500, -0.0048] | 5/0 |
| fold 1 | s0|T-out - (P-out + B3)|calibration | -0.0037 [-0.0255, +0.0180] | 3/3 |
| fold 1 | s0|T-out - (B0 + B3)|selection | +0.0340 [+0.0078, +0.0614] | 1/1 |
| fold 1 | s0|T-out - (B0 + B3)|calibration | +0.0550 [+0.0219, +0.0896] | 1/3 |
| fold 1 | s1|T-out - T-in|selection | +0.0469 [+0.0180, +0.0774] | 1/1 |
| fold 1 | s1|T-out - T-in|calibration | +0.0528 [+0.0185, +0.0903] | 3/4 |
| fold 1 | s1|T-out - (P-out + B3)|selection | -0.0299 [-0.0546, -0.0048] | 3/0 |
| fold 1 | s1|T-out - (P-out + B3)|calibration | -0.0017 [-0.0235, +0.0208] | 2/2 |
| fold 1 | s1|T-out - (B0 + B3)|selection | +0.0319 [+0.0053, +0.0598] | 1/1 |
| fold 1 | s1|T-out - (B0 + B3)|calibration | +0.0560 [+0.0242, +0.0885] | 2/4 |
| fold 1 | s2|T-out - T-in|selection | +0.0511 [+0.0209, +0.0830] | 0/0 |
| fold 1 | s2|T-out - T-in|calibration | +0.0497 [+0.0191, +0.0823] | 2/3 |
| fold 1 | s2|T-out - (P-out + B3)|selection | -0.0226 [-0.0451, +0.0002] | 3/1 |
| fold 1 | s2|T-out - (P-out + B3)|calibration | -0.0037 [-0.0278, +0.0210] | 3/3 |
| fold 1 | s2|T-out - (B0 + B3)|selection | +0.0392 [+0.0122, +0.0676] | 0/1 |
| fold 1 | s2|T-out - (B0 + B3)|calibration | +0.0541 [+0.0233, +0.0862] | 1/3 |
| fold 2 | s0|T-out - T-in|selection | +0.0593 [+0.0269, +0.0933] | 4/3 |
| fold 2 | s0|T-out - T-in|calibration | +0.0729 [+0.0434, +0.1020] | 2/1 |
| fold 2 | s0|T-out - (P-out + B3)|selection | -0.0134 [-0.0356, +0.0096] | 7/2 |
| fold 2 | s0|T-out - (P-out + B3)|calibration | -0.0080 [-0.0329, +0.0161] | 3/4 |
| fold 2 | s0|T-out - (B0 + B3)|selection | +0.0537 [+0.0210, +0.0882] | 3/3 |
| fold 2 | s0|T-out - (B0 + B3)|calibration | +0.0633 [+0.0359, +0.0904] | 3/1 |
| fold 2 | s1|T-out - T-in|selection | +0.0540 [+0.0230, +0.0876] | 3/3 |
| fold 2 | s1|T-out - T-in|calibration | +0.0680 [+0.0414, +0.0934] | 3/2 |
| fold 2 | s1|T-out - (P-out + B3)|selection | -0.0024 [-0.0251, +0.0229] | 6/1 |
| fold 2 | s1|T-out - (P-out + B3)|calibration | -0.0065 [-0.0296, +0.0148] | 2/4 |
| fold 2 | s1|T-out - (B0 + B3)|selection | +0.0667 [+0.0327, +0.1016] | 3/3 |
| fold 2 | s1|T-out - (B0 + B3)|calibration | +0.0608 [+0.0353, +0.0859] | 3/2 |
| fold 2 | s2|T-out - T-in|selection | +0.0497 [+0.0165, +0.0825] | 4/1 |
| fold 2 | s2|T-out - T-in|calibration | +0.0666 [+0.0418, +0.0933] | 2/1 |
| fold 2 | s2|T-out - (P-out + B3)|selection | -0.0082 [-0.0314, +0.0176] | 8/1 |
| fold 2 | s2|T-out - (P-out + B3)|calibration | -0.0125 [-0.0362, +0.0091] | 3/4 |
| fold 2 | s2|T-out - (B0 + B3)|selection | +0.0610 [+0.0272, +0.0959] | 3/1 |
| fold 2 | s2|T-out - (B0 + B3)|calibration | +0.0549 [+0.0294, +0.0792] | 3/1 |

## Decision (main encoder)

```
{
 "primary": {
  "selection": "no consistent difference detected",
  "calibration": "no consistent difference detected"
 },
 "secondary": {
  "T-out - (P-out + B3)": {
   "selection": "no consistent difference detected",
   "calibration": "consistent positive"
  },
  "T-out - (B0 + B3)": {
   "selection": "consistent negative",
   "calibration": "no consistent difference detected"
  }
 },
 "both_halves_positive": false,
 "beats_simple_controls": false,
 "constraints_met": false,
 "mean_segment_f1_T-out": {
  "selection": 0.8837229951001792,
  "calibration": 0.8851293643853682
 },
 "goal_met": false,
 "goal_margin": {
  "selection": -0.06068669208448285,
  "calibration": -0.09701280660406508
 },
 "calibration_sensitivity_margin": -0.09876421408940816,
 "observation": "the primary comparison is not consistent positive in both halves",
 "conclusion": "not supported in both halves; each half reported as positive, negative or uncertain",
 "action": "the round ends; no automatic search"
}
```
