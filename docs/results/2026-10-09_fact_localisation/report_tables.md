# FACT short-segment localisation: tables

Post hoc training-set diagnostic of the frozen FACT, ASRF and E1 of the paired FACT-ASRF round. No training and no development-set result. Conditions B and C of the intervention use the reference partition; they are diagnostics, never model results.

Recompute: `python scripts/diagnose_fact_localisation.py --config configs/experiments/fact_localisation_v1.yaml --run`

## Fit of every readout (pooled over sequences; per-sequence F1 beside it)

| seed | kind | readout | frame error | segment F1@50 (pooled) | per-sequence F1 mean | per-sequence F1 median | matched/reference runs | exact transcripts | boundary R | boundary P | predicted changes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | stream | E1 | 0.0655 | 0.5184 | 0.5999 | 0.5882 | 5672/6463 | 255/1500 | 0.9966 | 0.3553 | 13921 |
| 0 | stream | FACT_i | 0.0294 | 0.9539 | 0.9592 | 1.0000 | 6275/6463 | 1259/1500 | 0.9678 | 0.9247 | 5194 |
| 0 | stream | FACT_u | 0.0270 | 0.9662 | 0.9689 | 1.0000 | 6278/6463 | 1318/1500 | 0.9672 | 0.9539 | 5032 |
| 0 | stream | FACT_U | 0.0266 | 0.9691 | 0.9712 | 1.0000 | 6288/6463 | 1324/1500 | 0.9680 | 0.9581 | 5014 |
| 0 | stream | FACT_final | 0.0271 | 0.9707 | 0.9722 | 1.0000 | 6281/6463 | 1340/1500 | 0.9666 | 0.9636 | 4978 |
| 0 | stream | ASRF_frame | 0.0023 | 0.9964 | 0.9965 | 1.0000 | 6445/6463 | 1480/1500 | 0.9972 | 0.9950 | 4974 |
| 0 | stream | ASRF_final | 0.0022 | 0.9974 | 0.9975 | 1.0000 | 6444/6463 | 1484/1500 | 0.9970 | 0.9980 | 4958 |
| 0 | native | E1 | 0.0492 | 0.3658 | 0.7193 | 1.0000 | 1273/1500 | 929/1500 | None | 0.0000 | 3960 |
| 0 | native | FACT_i | 0.0101 | 0.9503 | 0.9774 | 1.0000 | 1483/1500 | 1445/1500 | None | 0.0000 | 121 |
| 0 | native | FACT_u | 0.0047 | 0.9881 | 0.9924 | 1.0000 | 1492/1500 | 1484/1500 | None | 0.0000 | 20 |
| 0 | native | FACT_U | 0.0044 | 0.9894 | 0.9928 | 1.0000 | 1493/1500 | 1484/1500 | None | 0.0000 | 18 |
| 0 | native | FACT_final | 0.0050 | 0.9894 | 0.9926 | 1.0000 | 1492/1500 | 1484/1500 | None | 0.0000 | 16 |
| 0 | native | ASRF_frame | 0.0021 | 0.9920 | 0.9957 | 1.0000 | 1497/1500 | 1490/1500 | None | 0.0000 | 18 |
| 0 | native | ASRF_final | 0.0018 | 0.9960 | 0.9972 | 1.0000 | 1497/1500 | 1494/1500 | None | 0.0000 | 6 |
| 1 | stream | E1 | 0.0655 | 0.5184 | 0.5999 | 0.5882 | 5672/6463 | 255/1500 | 0.9966 | 0.3553 | 13921 |
| 1 | stream | FACT_i | 0.0296 | 0.9527 | 0.9590 | 1.0000 | 6265/6463 | 1272/1500 | 0.9664 | 0.9243 | 5189 |
| 1 | stream | FACT_u | 0.0266 | 0.9676 | 0.9695 | 1.0000 | 6281/6463 | 1320/1500 | 0.9670 | 0.9562 | 5019 |
| 1 | stream | FACT_U | 0.0269 | 0.9707 | 0.9721 | 1.0000 | 6288/6463 | 1323/1500 | 0.9678 | 0.9619 | 4993 |
| 1 | stream | FACT_final | 0.0275 | 0.9704 | 0.9706 | 1.0000 | 6271/6463 | 1328/1500 | 0.9655 | 0.9657 | 4962 |
| 1 | stream | ASRF_frame | 0.0022 | 0.9972 | 0.9974 | 1.0000 | 6451/6463 | 1482/1500 | 0.9976 | 0.9952 | 4975 |
| 1 | stream | ASRF_final | 0.0020 | 0.9980 | 0.9980 | 1.0000 | 6449/6463 | 1488/1500 | 0.9974 | 0.9978 | 4961 |
| 1 | native | E1 | 0.0471 | 0.3742 | 0.7302 | 1.0000 | 1290/1500 | 943/1500 | None | 0.0000 | 3894 |
| 1 | native | FACT_i | 0.0084 | 0.9612 | 0.9827 | 1.0000 | 1487/1500 | 1457/1500 | None | 0.0000 | 94 |
| 1 | native | FACT_u | 0.0045 | 0.9917 | 0.9938 | 1.0000 | 1493/1500 | 1486/1500 | None | 0.0000 | 11 |
| 1 | native | FACT_U | 0.0044 | 0.9924 | 0.9942 | 1.0000 | 1493/1500 | 1488/1500 | None | 0.0000 | 9 |
| 1 | native | FACT_final | 0.0045 | 0.9920 | 0.9940 | 1.0000 | 1493/1500 | 1487/1500 | None | 0.0000 | 10 |
| 1 | native | ASRF_frame | 0.0024 | 0.9914 | 0.9954 | 1.0000 | 1496/1500 | 1490/1500 | None | 0.0000 | 18 |
| 1 | native | ASRF_final | 0.0024 | 0.9960 | 0.9969 | 1.0000 | 1496/1500 | 1494/1500 | None | 0.0000 | 4 |
| 2 | stream | E1 | 0.0655 | 0.5184 | 0.5999 | 0.5882 | 5672/6463 | 255/1500 | 0.9966 | 0.3553 | 13921 |
| 2 | stream | FACT_i | 0.0293 | 0.9518 | 0.9579 | 1.0000 | 6263/6463 | 1271/1500 | 0.9668 | 0.9232 | 5197 |
| 2 | stream | FACT_u | 0.0275 | 0.9668 | 0.9686 | 1.0000 | 6271/6463 | 1317/1500 | 0.9653 | 0.9563 | 5010 |
| 2 | stream | FACT_U | 0.0286 | 0.9695 | 0.9704 | 1.0000 | 6272/6463 | 1324/1500 | 0.9647 | 0.9624 | 4975 |
| 2 | stream | FACT_final | 0.0301 | 0.9679 | 0.9690 | 1.0000 | 6251/6463 | 1326/1500 | 0.9637 | 0.9657 | 4953 |
| 2 | stream | ASRF_frame | 0.0021 | 0.9968 | 0.9971 | 1.0000 | 6447/6463 | 1482/1500 | 0.9974 | 0.9954 | 4973 |
| 2 | stream | ASRF_final | 0.0018 | 0.9983 | 0.9983 | 1.0000 | 6449/6463 | 1489/1500 | 0.9968 | 0.9980 | 4957 |
| 2 | native | E1 | 0.0477 | 0.3745 | 0.7300 | 1.0000 | 1289/1500 | 941/1500 | None | 0.0000 | 3883 |
| 2 | native | FACT_i | 0.0113 | 0.9552 | 0.9809 | 1.0000 | 1481/1500 | 1459/1500 | None | 0.0000 | 101 |
| 2 | native | FACT_u | 0.0071 | 0.9864 | 0.9914 | 1.0000 | 1490/1500 | 1482/1500 | None | 0.0000 | 21 |
| 2 | native | FACT_U | 0.0068 | 0.9871 | 0.9923 | 1.0000 | 1491/1500 | 1484/1500 | None | 0.0000 | 21 |
| 2 | native | FACT_final | 0.0073 | 0.9887 | 0.9920 | 1.0000 | 1491/1500 | 1483/1500 | None | 0.0000 | 16 |
| 2 | native | ASRF_frame | 0.0033 | 0.9881 | 0.9939 | 1.0000 | 1494/1500 | 1486/1500 | None | 0.0000 | 24 |
| 2 | native | ASRF_final | 0.0023 | 0.9967 | 0.9973 | 1.0000 | 1497/1500 | 1494/1500 | None | 0.0000 | 4 |

## Reference runs matched (one to one, class-aware, IoU 0.5), by group

| seed | group | E1 | FACT_i | FACT_u | FACT_U | FACT_final | ASRF_frame | ASRF_final | unmatched at FACT_final (recordings/sessions) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | stream:all | 5672/6463 | 6275/6463 | 6278/6463 | 6288/6463 | 6281/6463 | 6445/6463 | 6444/6463 | 148/117 |
| 0 | stream:short | 1617/1726 | 1648/1726 | 1639/1726 | 1644/1726 | 1638/1726 | 1720/1726 | 1719/1726 | 83/73 |
| 0 | stream:xyx | 772/873 | 827/873 | 822/873 | 827/873 | 823/873 | 869/873 | 868/873 | 48/44 |
| 0 | stream:a4c_a5c_xyx | 72/99 | 69/99 | 64/99 | 68/99 | 62/99 | 97/99 | 96/99 | 37/36 |
| 0 | native:all | 1273/1500 | 1483/1500 | 1492/1500 | 1493/1500 | 1492/1500 | 1497/1500 | 1497/1500 | 8/8 |
| 1 | stream:all | 5672/6463 | 6265/6463 | 6281/6463 | 6288/6463 | 6271/6463 | 6451/6463 | 6449/6463 | 154/120 |
| 1 | stream:short | 1617/1726 | 1642/1726 | 1640/1726 | 1644/1726 | 1635/1726 | 1723/1726 | 1723/1726 | 87/78 |
| 1 | stream:xyx | 772/873 | 822/873 | 827/873 | 830/873 | 825/873 | 870/873 | 869/873 | 46/41 |
| 1 | stream:a4c_a5c_xyx | 72/99 | 67/99 | 69/99 | 71/99 | 65/99 | 98/99 | 97/99 | 34/34 |
| 1 | native:all | 1290/1500 | 1487/1500 | 1493/1500 | 1493/1500 | 1493/1500 | 1496/1500 | 1496/1500 | 7/7 |
| 2 | stream:all | 5672/6463 | 6263/6463 | 6271/6463 | 6272/6463 | 6251/6463 | 6447/6463 | 6449/6463 | 169/127 |
| 2 | stream:short | 1617/1726 | 1643/1726 | 1632/1726 | 1632/1726 | 1626/1726 | 1720/1726 | 1722/1726 | 93/81 |
| 2 | stream:xyx | 772/873 | 824/873 | 822/873 | 822/873 | 821/873 | 869/873 | 869/873 | 48/44 |
| 2 | stream:a4c_a5c_xyx | 72/99 | 68/99 | 65/99 | 65/99 | 61/99 | 97/99 | 97/99 | 38/38 |
| 2 | native:all | 1289/1500 | 1481/1500 | 1490/1500 | 1491/1500 | 1491/1500 | 1494/1500 | 1497/1500 | 9/9 |

## Earliest unmatched readout along E1 -> FACT i -> u -> U -> final

| seed | group | first unmatched at | segments | recordings | sessions | matched_at_FACT_final |
|---|---|---|---|---|---|---|
| 0 | stream:all | E1 | 791 | 542 | 288 | 681 |
| 0 | stream:all | FACT_i | 57 | 59 | 52 | 11 |
| 0 | stream:all | FACT_u | 20 | 19 | 18 | 0 |
| 0 | stream:all | FACT_U | 3 | 3 | 3 | 3 |
| 0 | stream:all | FACT_final | 6 | 7 | 7 | 0 |
| 0 | stream:all | never unmatched | 5586 | 2745 | 496 | 5586 |
| 0 | stream:short | E1 | 109 | 84 | 69 | 79 |
| 0 | stream:short | FACT_i | 40 | 39 | 37 | 5 |
| 0 | stream:short | FACT_u | 18 | 18 | 17 | 0 |
| 0 | stream:short | FACT_U | 1 | 1 | 1 | 1 |
| 0 | stream:short | FACT_final | 5 | 5 | 5 | 0 |
| 0 | stream:short | never unmatched | 1553 | 1054 | 414 | 1553 |
| 0 | stream:xyx | E1 | 101 | 106 | 85 | 77 |
| 0 | stream:xyx | FACT_i | 19 | 19 | 19 | 3 |
| 0 | stream:xyx | FACT_u | 7 | 7 | 7 | 0 |
| 0 | stream:xyx | FACT_final | 3 | 4 | 4 | 0 |
| 0 | stream:xyx | never unmatched | 743 | 781 | 356 | 743 |
| 0 | stream:a4c_a5c_xyx | E1 | 27 | 36 | 34 | 14 |
| 0 | stream:a4c_a5c_xyx | FACT_i | 17 | 17 | 17 | 2 |
| 0 | stream:a4c_a5c_xyx | FACT_u | 7 | 7 | 7 | 0 |
| 0 | stream:a4c_a5c_xyx | FACT_final | 2 | 2 | 2 | 0 |
| 0 | stream:a4c_a5c_xyx | never unmatched | 46 | 56 | 52 | 46 |
| 0 | native:all | E1 | 227 | 227 | 168 | 219 |
| 0 | native:all | never unmatched | 1273 | 1273 | 415 | 1273 |
| 1 | stream:all | E1 | 791 | 542 | 288 | 675 |
| 1 | stream:all | FACT_i | 63 | 65 | 59 | 16 |
| 1 | stream:all | FACT_u | 18 | 17 | 16 | 0 |
| 1 | stream:all | FACT_U | 2 | 2 | 2 | 2 |
| 1 | stream:all | FACT_final | 11 | 13 | 13 | 0 |
| 1 | stream:all | never unmatched | 5578 | 2743 | 496 | 5578 |
| 1 | stream:short | E1 | 109 | 84 | 69 | 75 |
| 1 | stream:short | FACT_i | 45 | 44 | 42 | 9 |
| 1 | stream:short | FACT_u | 14 | 14 | 13 | 0 |
| 1 | stream:short | FACT_U | 2 | 2 | 2 | 2 |
| 1 | stream:short | FACT_final | 7 | 7 | 7 | 0 |
| 1 | stream:short | never unmatched | 1549 | 1051 | 412 | 1549 |
| 1 | stream:xyx | E1 | 101 | 106 | 85 | 76 |
| 1 | stream:xyx | FACT_i | 21 | 21 | 21 | 6 |
| 1 | stream:xyx | FACT_u | 5 | 5 | 5 | 0 |
| 1 | stream:xyx | FACT_final | 3 | 3 | 3 | 0 |
| 1 | stream:xyx | never unmatched | 743 | 781 | 356 | 743 |
| 1 | stream:a4c_a5c_xyx | E1 | 27 | 36 | 34 | 14 |
| 1 | stream:a4c_a5c_xyx | FACT_i | 18 | 18 | 18 | 4 |
| 1 | stream:a4c_a5c_xyx | FACT_u | 5 | 5 | 5 | 0 |
| 1 | stream:a4c_a5c_xyx | FACT_final | 2 | 2 | 2 | 0 |
| 1 | stream:a4c_a5c_xyx | never unmatched | 47 | 57 | 53 | 47 |
| 1 | native:all | E1 | 210 | 210 | 145 | 203 |
| 1 | native:all | never unmatched | 1290 | 1290 | 433 | 1290 |
| 2 | stream:all | E1 | 791 | 542 | 288 | 662 |
| 2 | stream:all | FACT_i | 64 | 66 | 59 | 17 |
| 2 | stream:all | FACT_u | 24 | 22 | 21 | 1 |
| 2 | stream:all | FACT_U | 7 | 7 | 7 | 6 |
| 2 | stream:all | FACT_final | 12 | 13 | 13 | 0 |
| 2 | stream:all | never unmatched | 5565 | 2742 | 496 | 5565 |
| 2 | stream:short | E1 | 109 | 84 | 69 | 75 |
| 2 | stream:short | FACT_i | 44 | 43 | 41 | 7 |
| 2 | stream:short | FACT_u | 22 | 21 | 20 | 1 |
| 2 | stream:short | FACT_U | 5 | 5 | 5 | 4 |
| 2 | stream:short | FACT_final | 7 | 7 | 7 | 0 |
| 2 | stream:short | never unmatched | 1539 | 1049 | 411 | 1539 |
| 2 | stream:xyx | E1 | 101 | 106 | 85 | 74 |
| 2 | stream:xyx | FACT_i | 21 | 21 | 21 | 6 |
| 2 | stream:xyx | FACT_u | 6 | 6 | 6 | 0 |
| 2 | stream:xyx | FACT_U | 2 | 2 | 2 | 2 |
| 2 | stream:xyx | FACT_final | 4 | 4 | 4 | 0 |
| 2 | stream:xyx | never unmatched | 739 | 779 | 355 | 739 |
| 2 | stream:a4c_a5c_xyx | E1 | 27 | 36 | 34 | 14 |
| 2 | stream:a4c_a5c_xyx | FACT_i | 18 | 18 | 18 | 3 |
| 2 | stream:a4c_a5c_xyx | FACT_u | 6 | 6 | 6 | 0 |
| 2 | stream:a4c_a5c_xyx | FACT_U | 2 | 2 | 2 | 2 |
| 2 | stream:a4c_a5c_xyx | FACT_final | 4 | 4 | 4 | 0 |
| 2 | stream:a4c_a5c_xyx | never unmatched | 42 | 52 | 48 | 42 |
| 2 | native:all | E1 | 211 | 211 | 152 | 202 |
| 2 | native:all | never unmatched | 1289 | 1289 | 435 | 1289 |

## Lost and recovered between readouts

| seed | group | readouts | lost | lost rec/ses | recovered | recovered rec/ses |
|---|---|---|---|---|---|---|
| 0 | stream:all | E1 -> FACT_i | 57 | 59/52 | 660 | 515/283 |
| 0 | stream:all | FACT_i -> FACT_u | 25 | 25/23 | 28 | 31/28 |
| 0 | stream:all | FACT_u -> FACT_U | 6 | 7/7 | 16 | 22/22 |
| 0 | stream:all | FACT_U -> FACT_final | 25 | 34/33 | 18 | 19/19 |
| 0 | stream:all | E1 -> FACT_final | 72 | 71/64 | 681 | 519/285 |
| 0 | stream:all | E1 -> ASRF_frame | 5 | 6/6 | 778 | 537/288 |
| 0 | stream:all | ASRF_frame -> ASRF_final | 3 | 3/3 | 2 | 2/2 |
| 0 | stream:all | E1 -> ASRF_final | 6 | 7/7 | 778 | 538/288 |
| 0 | stream:short | E1 -> FACT_i | 40 | 39/37 | 71 | 54/47 |
| 0 | stream:short | FACT_i -> FACT_u | 20 | 20/19 | 11 | 11/11 |
| 0 | stream:short | FACT_u -> FACT_U | 3 | 3/3 | 8 | 8/8 |
| 0 | stream:short | FACT_U -> FACT_final | 12 | 12/12 | 6 | 6/6 |
| 0 | stream:short | E1 -> FACT_final | 58 | 55/49 | 79 | 61/53 |
| 0 | stream:short | E1 -> ASRF_frame | 3 | 3/3 | 106 | 82/68 |
| 0 | stream:short | ASRF_frame -> ASRF_final | 2 | 2/2 | 1 | 1/1 |
| 0 | stream:short | E1 -> ASRF_final | 4 | 4/4 | 106 | 82/68 |
| 0 | stream:xyx | E1 -> FACT_i | 19 | 19/19 | 74 | 88/71 |
| 0 | stream:xyx | FACT_i -> FACT_u | 8 | 8/8 | 3 | 3/3 |
| 0 | stream:xyx | FACT_u -> FACT_U | 0 | 0/0 | 5 | 5/5 |
| 0 | stream:xyx | FACT_U -> FACT_final | 8 | 10/9 | 4 | 4/4 |
| 0 | stream:xyx | E1 -> FACT_final | 26 | 27/27 | 77 | 90/74 |
| 0 | stream:xyx | E1 -> ASRF_frame | 1 | 1/1 | 98 | 102/82 |
| 0 | stream:xyx | ASRF_frame -> ASRF_final | 1 | 1/1 | 0 | 0/0 |
| 0 | stream:xyx | E1 -> ASRF_final | 2 | 2/2 | 98 | 102/82 |
| 0 | stream:a4c_a5c_xyx | E1 -> FACT_i | 17 | 17/17 | 14 | 24/23 |
| 0 | stream:a4c_a5c_xyx | FACT_i -> FACT_u | 7 | 7/7 | 2 | 2/2 |
| 0 | stream:a4c_a5c_xyx | FACT_u -> FACT_U | 0 | 0/0 | 4 | 4/4 |
| 0 | stream:a4c_a5c_xyx | FACT_U -> FACT_final | 6 | 7/6 | 0 | 0/0 |
| 0 | stream:a4c_a5c_xyx | E1 -> FACT_final | 24 | 24/24 | 14 | 23/22 |
| 0 | stream:a4c_a5c_xyx | E1 -> ASRF_frame | 1 | 1/1 | 26 | 35/33 |
| 0 | stream:a4c_a5c_xyx | ASRF_frame -> ASRF_final | 1 | 1/1 | 0 | 0/0 |
| 0 | stream:a4c_a5c_xyx | E1 -> ASRF_final | 2 | 2/2 | 26 | 35/33 |
| 0 | native:all | E1 -> FACT_i | 0 | 0/0 | 210 | 210/159 |
| 0 | native:all | FACT_i -> FACT_u | 1 | 1/1 | 10 | 10/10 |
| 0 | native:all | FACT_u -> FACT_U | 0 | 0/0 | 1 | 1/1 |
| 0 | native:all | FACT_U -> FACT_final | 1 | 1/1 | 0 | 0/0 |
| 0 | native:all | E1 -> FACT_final | 0 | 0/0 | 219 | 219/164 |
| 0 | native:all | E1 -> ASRF_frame | 0 | 0/0 | 224 | 224/167 |
| 0 | native:all | ASRF_frame -> ASRF_final | 0 | 0/0 | 0 | 0/0 |
| 0 | native:all | E1 -> ASRF_final | 0 | 0/0 | 224 | 224/167 |
| 1 | stream:all | E1 -> FACT_i | 63 | 65/59 | 656 | 521/284 |
| 1 | stream:all | FACT_i -> FACT_u | 22 | 22/21 | 38 | 39/36 |
| 1 | stream:all | FACT_u -> FACT_U | 5 | 6/6 | 12 | 14/13 |
| 1 | stream:all | FACT_U -> FACT_final | 32 | 40/36 | 15 | 13/13 |
| 1 | stream:all | E1 -> FACT_final | 76 | 77/71 | 675 | 517/286 |
| 1 | stream:all | E1 -> ASRF_frame | 2 | 3/3 | 781 | 538/288 |
| 1 | stream:all | ASRF_frame -> ASRF_final | 3 | 3/3 | 1 | 1/1 |
| 1 | stream:all | E1 -> ASRF_final | 3 | 4/4 | 780 | 538/288 |
| 1 | stream:short | E1 -> FACT_i | 45 | 44/42 | 70 | 54/47 |
| 1 | stream:short | FACT_i -> FACT_u | 15 | 15/14 | 13 | 13/13 |
| 1 | stream:short | FACT_u -> FACT_U | 3 | 3/3 | 7 | 7/6 |
| 1 | stream:short | FACT_U -> FACT_final | 15 | 15/14 | 6 | 5/5 |
| 1 | stream:short | E1 -> FACT_final | 57 | 55/50 | 75 | 59/51 |
| 1 | stream:short | E1 -> ASRF_frame | 1 | 1/1 | 107 | 82/68 |
| 1 | stream:short | ASRF_frame -> ASRF_final | 1 | 1/1 | 1 | 1/1 |
| 1 | stream:short | E1 -> ASRF_final | 1 | 1/1 | 107 | 82/68 |
| 1 | stream:xyx | E1 -> FACT_i | 21 | 21/21 | 71 | 86/70 |
| 1 | stream:xyx | FACT_i -> FACT_u | 5 | 5/5 | 10 | 10/10 |
| 1 | stream:xyx | FACT_u -> FACT_U | 1 | 1/1 | 4 | 4/4 |
| 1 | stream:xyx | FACT_U -> FACT_final | 10 | 12/11 | 5 | 5/5 |
| 1 | stream:xyx | E1 -> FACT_final | 23 | 23/22 | 76 | 89/72 |
| 1 | stream:xyx | E1 -> ASRF_frame | 0 | 0/0 | 98 | 102/82 |
| 1 | stream:xyx | ASRF_frame -> ASRF_final | 1 | 1/1 | 0 | 0/0 |
| 1 | stream:xyx | E1 -> ASRF_final | 1 | 1/1 | 98 | 102/82 |
| 1 | stream:a4c_a5c_xyx | E1 -> FACT_i | 18 | 18/18 | 13 | 23/22 |
| 1 | stream:a4c_a5c_xyx | FACT_i -> FACT_u | 5 | 5/5 | 7 | 7/7 |
| 1 | stream:a4c_a5c_xyx | FACT_u -> FACT_U | 1 | 1/1 | 3 | 3/3 |
| 1 | stream:a4c_a5c_xyx | FACT_U -> FACT_final | 7 | 8/8 | 1 | 1/1 |
| 1 | stream:a4c_a5c_xyx | E1 -> FACT_final | 21 | 21/21 | 14 | 24/22 |
| 1 | stream:a4c_a5c_xyx | E1 -> ASRF_frame | 0 | 0/0 | 26 | 35/33 |
| 1 | stream:a4c_a5c_xyx | ASRF_frame -> ASRF_final | 1 | 1/1 | 0 | 0/0 |
| 1 | stream:a4c_a5c_xyx | E1 -> ASRF_final | 1 | 1/1 | 26 | 35/33 |
| 1 | native:all | E1 -> FACT_i | 0 | 0/0 | 197 | 197/135 |
| 1 | native:all | FACT_i -> FACT_u | 0 | 0/0 | 6 | 6/6 |
| 1 | native:all | FACT_u -> FACT_U | 0 | 0/0 | 0 | 0/0 |
| 1 | native:all | FACT_U -> FACT_final | 0 | 0/0 | 0 | 0/0 |
| 1 | native:all | E1 -> FACT_final | 0 | 0/0 | 203 | 203/140 |
| 1 | native:all | E1 -> ASRF_frame | 0 | 0/0 | 206 | 206/142 |
| 1 | native:all | ASRF_frame -> ASRF_final | 0 | 0/0 | 0 | 0/0 |
| 1 | native:all | E1 -> ASRF_final | 0 | 0/0 | 206 | 206/142 |
| 2 | stream:all | E1 -> FACT_i | 64 | 66/59 | 655 | 515/283 |
| 2 | stream:all | FACT_i -> FACT_u | 28 | 27/25 | 36 | 34/29 |
| 2 | stream:all | FACT_u -> FACT_U | 13 | 15/15 | 14 | 16/16 |
| 2 | stream:all | FACT_U -> FACT_final | 40 | 46/45 | 19 | 21/21 |
| 2 | stream:all | E1 -> FACT_final | 83 | 80/69 | 662 | 510/283 |
| 2 | stream:all | E1 -> ASRF_frame | 5 | 6/6 | 780 | 538/288 |
| 2 | stream:all | ASRF_frame -> ASRF_final | 0 | 0/0 | 2 | 2/2 |
| 2 | stream:all | E1 -> ASRF_final | 4 | 5/5 | 781 | 538/288 |
| 2 | stream:short | E1 -> FACT_i | 44 | 43/41 | 70 | 54/47 |
| 2 | stream:short | FACT_i -> FACT_u | 22 | 21/20 | 11 | 11/11 |
| 2 | stream:short | FACT_u -> FACT_U | 8 | 8/8 | 8 | 8/8 |
| 2 | stream:short | FACT_U -> FACT_final | 15 | 15/15 | 9 | 9/9 |
| 2 | stream:short | E1 -> FACT_final | 66 | 62/55 | 75 | 59/51 |
| 2 | stream:short | E1 -> ASRF_frame | 3 | 3/3 | 106 | 81/68 |
| 2 | stream:short | ASRF_frame -> ASRF_final | 0 | 0/0 | 2 | 2/2 |
| 2 | stream:short | E1 -> ASRF_final | 2 | 2/2 | 107 | 82/68 |
| 2 | stream:xyx | E1 -> FACT_i | 21 | 21/21 | 73 | 87/71 |
| 2 | stream:xyx | FACT_i -> FACT_u | 6 | 6/6 | 4 | 4/4 |
| 2 | stream:xyx | FACT_u -> FACT_U | 3 | 4/4 | 3 | 3/3 |
| 2 | stream:xyx | FACT_U -> FACT_final | 9 | 11/11 | 8 | 9/9 |
| 2 | stream:xyx | E1 -> FACT_final | 25 | 25/25 | 74 | 86/69 |
| 2 | stream:xyx | E1 -> ASRF_frame | 1 | 1/1 | 98 | 102/82 |
| 2 | stream:xyx | ASRF_frame -> ASRF_final | 0 | 0/0 | 0 | 0/0 |
| 2 | stream:xyx | E1 -> ASRF_final | 1 | 1/1 | 98 | 102/82 |
| 2 | stream:a4c_a5c_xyx | E1 -> FACT_i | 18 | 18/18 | 14 | 24/23 |
| 2 | stream:a4c_a5c_xyx | FACT_i -> FACT_u | 6 | 6/6 | 3 | 3/3 |
| 2 | stream:a4c_a5c_xyx | FACT_u -> FACT_U | 3 | 4/4 | 3 | 3/3 |
| 2 | stream:a4c_a5c_xyx | FACT_U -> FACT_final | 8 | 9/9 | 4 | 5/5 |
| 2 | stream:a4c_a5c_xyx | E1 -> FACT_final | 25 | 25/25 | 14 | 23/21 |
| 2 | stream:a4c_a5c_xyx | E1 -> ASRF_frame | 1 | 1/1 | 26 | 35/33 |
| 2 | stream:a4c_a5c_xyx | ASRF_frame -> ASRF_final | 0 | 0/0 | 0 | 0/0 |
| 2 | stream:a4c_a5c_xyx | E1 -> ASRF_final | 1 | 1/1 | 26 | 35/33 |
| 2 | native:all | E1 -> FACT_i | 0 | 0/0 | 192 | 192/142 |
| 2 | native:all | FACT_i -> FACT_u | 0 | 0/0 | 9 | 9/9 |
| 2 | native:all | FACT_u -> FACT_U | 1 | 1/1 | 2 | 2/2 |
| 2 | native:all | FACT_U -> FACT_final | 2 | 2/2 | 2 | 2/2 |
| 2 | native:all | E1 -> FACT_final | 0 | 0/0 | 202 | 202/149 |
| 2 | native:all | E1 -> ASRF_frame | 0 | 0/0 | 205 | 205/149 |
| 2 | native:all | ASRF_frame -> ASRF_final | 0 | 0/0 | 3 | 3/3 |
| 2 | native:all | E1 -> ASRF_final | 0 | 0/0 | 208 | 208/150 |

## Reference runs in the partition entering U (constructed streams)

| seed | group | reference run in the partition entering U | segments | recordings | sessions |
|---|---|---|---|---|---|
| 0 | all | own interval | 4366 | 2491 | 486 |
| 0 | all | inside a larger interval | 1026 | 923 | 383 |
| 0 | all | spans intervals, an end inside one | 1003 | 965 | 405 |
| 0 | all | own intervals, cut inside | 68 | 79 | 70 |
| 0 | short | own interval | 1111 | 800 | 369 |
| 0 | short | inside a larger interval | 437 | 379 | 244 |
| 0 | short | spans intervals, an end inside one | 175 | 167 | 132 |
| 0 | short | own intervals, cut inside | 3 | 3 | 3 |
| 0 | xyx | own interval | 534 | 589 | 326 |
| 0 | xyx | spans intervals, an end inside one | 179 | 231 | 171 |
| 0 | xyx | inside a larger interval | 152 | 169 | 137 |
| 0 | xyx | own intervals, cut inside | 8 | 6 | 6 |
| 0 | a4c_a5c_xyx | spans intervals, an end inside one | 53 | 65 | 54 |
| 0 | a4c_a5c_xyx | inside a larger interval | 44 | 50 | 50 |
| 0 | a4c_a5c_xyx | own interval | 2 | 2 | 2 |
| 1 | all | own interval | 4542 | 2531 | 485 |
| 1 | all | inside a larger interval | 966 | 933 | 385 |
| 1 | all | own intervals, cut inside | 62 | 71 | 65 |
| 1 | all | spans intervals, an end inside one | 893 | 849 | 371 |
| 1 | short | spans intervals, an end inside one | 175 | 167 | 128 |
| 1 | short | own interval | 1152 | 840 | 377 |
| 1 | short | inside a larger interval | 396 | 348 | 222 |
| 1 | short | own intervals, cut inside | 3 | 3 | 3 |
| 1 | xyx | spans intervals, an end inside one | 157 | 210 | 160 |
| 1 | xyx | own interval | 576 | 621 | 323 |
| 1 | xyx | inside a larger interval | 130 | 142 | 118 |
| 1 | xyx | own intervals, cut inside | 10 | 9 | 9 |
| 1 | a4c_a5c_xyx | spans intervals, an end inside one | 51 | 65 | 57 |
| 1 | a4c_a5c_xyx | inside a larger interval | 46 | 51 | 48 |
| 1 | a4c_a5c_xyx | own interval | 2 | 3 | 3 |
| 2 | all | own interval | 4396 | 2500 | 486 |
| 2 | all | inside a larger interval | 1045 | 917 | 399 |
| 2 | all | spans intervals, an end inside one | 972 | 941 | 416 |
| 2 | all | own intervals, cut inside | 50 | 61 | 55 |
| 2 | short | own interval | 1116 | 807 | 373 |
| 2 | short | inside a larger interval | 428 | 372 | 239 |
| 2 | short | spans intervals, an end inside one | 182 | 167 | 137 |
| 2 | xyx | own interval | 558 | 607 | 323 |
| 2 | xyx | inside a larger interval | 135 | 154 | 135 |
| 2 | xyx | spans intervals, an end inside one | 174 | 225 | 178 |
| 2 | xyx | own intervals, cut inside | 6 | 5 | 5 |
| 2 | a4c_a5c_xyx | inside a larger interval | 45 | 50 | 49 |
| 2 | a4c_a5c_xyx | spans intervals, an end inside one | 53 | 67 | 56 |
| 2 | a4c_a5c_xyx | own interval | 1 | 2 | 2 |

### By the largest share of one interval a run fills (added after the run; summary only)

| seed | group | place | segments | rec/ses | matched A/B/C | U frame branch A/B/C |
|---|---|---|---|---|---|---|
| 0 | all | minority in every interval | 85 | 78/70 | 0/40/27 | 2/6/4 |
| 0 | all | majority of an interval | 6378 | 2864/499 | 6281/6303/6288 | 6286/6287/6281 |
| 0 | short | minority in every interval | 58 | 55/51 | 0/26/14 | 2/3/3 |
| 0 | short | majority of an interval | 1668 | 1116/421 | 1638/1650/1645 | 1642/1642/1637 |
| 0 | xyx | minority in every interval | 28 | 27/27 | 0/14/8 | 0/1/1 |
| 0 | xyx | majority of an interval | 845 | 866/377 | 823/824/816 | 827/828/826 |
| 0 | a4c_a5c_xyx | minority in every interval | 28 | 27/27 | 0/14/8 | 0/1/1 |
| 0 | a4c_a5c_xyx | majority of an interval | 71 | 88/75 | 62/64/58 | 68/69/67 |
| 0 | stream intervals | changes inside an interval | 6532 |  | official output 0 (label off its token's class on every frame: 0) | 493 |
| 0 | native intervals | changes inside an interval | 1520 |  | official output 0 (label off its token's class on every frame: 0) | 9 |
| 1 | all | minority in every interval | 93 | 85/75 | 0/42/30 | 3/4/3 |
| 1 | all | majority of an interval | 6370 | 2865/499 | 6271/6295/6280 | 6285/6284/6280 |
| 1 | short | minority in every interval | 63 | 60/55 | 0/27/16 | 3/4/3 |
| 1 | short | majority of an interval | 1663 | 1114/421 | 1635/1639/1636 | 1641/1641/1638 |
| 1 | xyx | minority in every interval | 26 | 25/25 | 0/10/5 | 2/1/0 |
| 1 | xyx | majority of an interval | 847 | 868/377 | 825/827/824 | 828/826/825 |
| 1 | a4c_a5c_xyx | minority in every interval | 26 | 25/25 | 0/10/5 | 2/1/0 |
| 1 | a4c_a5c_xyx | majority of an interval | 73 | 90/77 | 65/69/64 | 69/67/66 |
| 1 | stream intervals | changes inside an interval | 6519 |  | official output 0 (label off its token's class on every frame: 1) | 569 |
| 1 | native intervals | changes inside an interval | 1511 |  | official output 0 (label off its token's class on every frame: 0) | 5 |
| 2 | all | minority in every interval | 93 | 86/76 | 0/36/24 | 2/2/0 |
| 2 | all | majority of an interval | 6370 | 2865/499 | 6251/6285/6260 | 6270/6272/6263 |
| 2 | short | minority in every interval | 67 | 63/57 | 0/28/16 | 2/2/0 |
| 2 | short | majority of an interval | 1659 | 1112/418 | 1626/1635/1626 | 1630/1631/1627 |
| 2 | xyx | minority in every interval | 28 | 27/27 | 0/8/2 | 1/1/0 |
| 2 | xyx | majority of an interval | 845 | 867/377 | 821/828/819 | 821/823/819 |
| 2 | a4c_a5c_xyx | minority in every interval | 28 | 27/27 | 0/8/2 | 1/1/0 |
| 2 | a4c_a5c_xyx | majority of an interval | 71 | 88/75 | 61/69/62 | 64/66/62 |
| 2 | stream intervals | changes inside an interval | 6510 |  | official output 0 (label off its token's class on every frame: 0) | 670 |
| 2 | native intervals | changes inside an interval | 1521 |  | official output 1 (label off its token's class on every frame: 0) | 9 |

### Reference changes on a partition boundary (added after the run; summary only)

| seed | all | A4C_A5C | A4C_A5C_middle_ends |
|---|---|---|---|
| 0 | 3822/4963 | 87/541 | 18/198 |
| 1 | 3922/4963 | 97/541 | 25/198 |
| 2 | 3818/4963 | 83/541 | 22/198 |

## Intervention at the last U block (constructed streams; B and C use the reference partition)

| seed | readout | condition | segment_f1_50 | boundary_recall | boundary_precision | matched/reference runs |
|---|---|---|---|---|---|---|
| 0 | official_output | A | 0.9707 | 0.9666 | 0.9636 | 6281/6463 |
| 0 | official_output | B | 0.9764 | 0.9825 | 0.9696 | 6343/6463 |
| 0 | official_output | C | 0.9741 | 0.9680 | 0.9602 | 6315/6463 |
| 0 | official_output | B - A | 0.0057 | 0.0159 | 0.0059 |  |
| 0 | official_output | C - A | 0.0034 | 0.0014 | -0.0034 |  |
| 0 | official_output | B - C | 0.0024 | 0.0145 | 0.0094 |  |
| 0 | U_frame_branch | A | 0.9691 | 0.9680 | 0.9581 | 6288/6463 |
| 0 | U_frame_branch | B | 0.9698 | 0.9696 | 0.9595 | 6293/6463 |
| 0 | U_frame_branch | C | 0.9686 | 0.9678 | 0.9579 | 6285/6463 |
| 0 | U_frame_branch | B - A | 0.0007 | 0.0016 | 0.0014 |  |
| 0 | U_frame_branch | C - A | -0.0005 | -0.0002 | -0.0002 |  |
| 0 | U_frame_branch | B - C | 0.0012 | 0.0018 | 0.0016 |  |
| 1 | official_output | A | 0.9704 | 0.9655 | 0.9657 | 6271/6463 |
| 1 | official_output | B | 0.9758 | 0.9803 | 0.9682 | 6337/6463 |
| 1 | official_output | C | 0.9738 | 0.9682 | 0.9616 | 6310/6463 |
| 1 | official_output | B - A | 0.0055 | 0.0147 | 0.0024 |  |
| 1 | official_output | C - A | 0.0034 | 0.0026 | -0.0042 |  |
| 1 | official_output | B - C | 0.0021 | 0.0121 | 0.0066 |  |
| 1 | U_frame_branch | A | 0.9707 | 0.9678 | 0.9619 | 6288/6463 |
| 1 | U_frame_branch | B | 0.9713 | 0.9674 | 0.9631 | 6288/6463 |
| 1 | U_frame_branch | C | 0.9706 | 0.9657 | 0.9617 | 6283/6463 |
| 1 | U_frame_branch | B - A | 0.0006 | -0.0004 | 0.0011 |  |
| 1 | U_frame_branch | C - A | -0.0001 | -0.0020 | -0.0003 |  |
| 1 | U_frame_branch | B - C | 0.0007 | 0.0016 | 0.0014 |  |
| 2 | official_output | A | 0.9679 | 0.9637 | 0.9657 | 6251/6463 |
| 2 | official_output | B | 0.9739 | 0.9794 | 0.9687 | 6321/6463 |
| 2 | official_output | C | 0.9714 | 0.9641 | 0.9618 | 6284/6463 |
| 2 | official_output | B - A | 0.0059 | 0.0157 | 0.0030 |  |
| 2 | official_output | C - A | 0.0035 | 0.0004 | -0.0039 |  |
| 2 | official_output | B - C | 0.0025 | 0.0153 | 0.0069 |  |
| 2 | U_frame_branch | A | 0.9695 | 0.9647 | 0.9624 | 6272/6463 |
| 2 | U_frame_branch | B | 0.9693 | 0.9659 | 0.9621 | 6274/6463 |
| 2 | U_frame_branch | C | 0.9680 | 0.9641 | 0.9614 | 6263/6463 |
| 2 | U_frame_branch | B - A | -0.0003 | 0.0012 | -0.0003 |  |
| 2 | U_frame_branch | C - A | -0.0015 | -0.0006 | -0.0010 |  |
| 2 | U_frame_branch | B - C | 0.0013 | 0.0018 | 0.0006 |  |

### Cuts

| seed | streams | streams_with_added_cuts | added_cuts | C_counts_equal_B | streams_with_B_and_C_executed |
|---|---|---|---|---|---|
| 0 | 1500 | 768 | 1141 | True | 1500 |
| 1 | 1500 | 725 | 1041 | True | 1500 |
| 2 | 1500 | 769 | 1145 | True | 1500 |

### Matched, fixed and new reference runs by group

| seed | group | comparison | matched A/B/C | fixed | fixed rec/ses | new | new rec/ses |
|---|---|---|---|---|---|---|---|
| 0 | all | B - A | 6281/6343/6315 of 6463 | 74 | 74/67 | 12 | 15/15 |
| 0 | all | C - A | 6281/6343/6315 of 6463 | 49 | 49/45 | 15 | 17/17 |
| 0 | all | B - C | 6281/6343/6315 of 6463 | 36 | 38/36 | 8 | 8/8 |
| 0 | short | B - A | 1638/1676/1659 of 1726 | 43 | 43/40 | 5 | 5/5 |
| 0 | short | C - A | 1638/1676/1659 of 1726 | 26 | 26/24 | 5 | 5/5 |
| 0 | short | B - C | 1638/1676/1659 of 1726 | 20 | 20/20 | 3 | 3/3 |
| 0 | xyx | B - A | 823/838/824 of 873 | 18 | 17/17 | 3 | 4/4 |
| 0 | xyx | C - A | 823/838/824 of 873 | 9 | 9/9 | 8 | 8/8 |
| 0 | xyx | B - C | 823/838/824 of 873 | 14 | 14/14 | 0 | 0/0 |
| 0 | a4c_a5c_xyx | B - A | 62/78/66 of 99 | 18 | 17/17 | 2 | 2/2 |
| 0 | a4c_a5c_xyx | C - A | 62/78/66 of 99 | 9 | 9/9 | 5 | 5/5 |
| 0 | a4c_a5c_xyx | B - C | 62/78/66 of 99 | 12 | 12/12 | 0 | 0/0 |
| 1 | all | B - A | 6271/6337/6310 of 6463 | 72 | 70/62 | 6 | 6/6 |
| 1 | all | C - A | 6271/6337/6310 of 6463 | 53 | 57/49 | 14 | 16/16 |
| 1 | all | B - C | 6271/6337/6310 of 6463 | 36 | 37/36 | 9 | 9/9 |
| 1 | short | B - A | 1635/1666/1652 of 1726 | 37 | 37/37 | 6 | 6/6 |
| 1 | short | C - A | 1635/1666/1652 of 1726 | 24 | 24/24 | 7 | 7/7 |
| 1 | short | B - C | 1635/1666/1652 of 1726 | 22 | 22/21 | 8 | 8/8 |
| 1 | xyx | B - A | 825/837/829 of 873 | 14 | 13/13 | 2 | 2/2 |
| 1 | xyx | C - A | 825/837/829 of 873 | 8 | 8/8 | 4 | 5/5 |
| 1 | xyx | B - C | 825/837/829 of 873 | 11 | 11/11 | 3 | 3/3 |
| 1 | a4c_a5c_xyx | B - A | 65/79/69 of 99 | 14 | 13/13 | 0 | 0/0 |
| 1 | a4c_a5c_xyx | C - A | 65/79/69 of 99 | 7 | 7/7 | 3 | 4/4 |
| 1 | a4c_a5c_xyx | B - C | 65/79/69 of 99 | 10 | 10/10 | 0 | 0/0 |
| 2 | all | B - A | 6251/6321/6284 of 6463 | 79 | 78/65 | 9 | 9/9 |
| 2 | all | C - A | 6251/6321/6284 of 6463 | 55 | 59/50 | 22 | 22/21 |
| 2 | all | B - C | 6251/6321/6284 of 6463 | 48 | 48/43 | 11 | 12/11 |
| 2 | short | B - A | 1626/1663/1642 of 1726 | 44 | 43/40 | 7 | 7/7 |
| 2 | short | C - A | 1626/1663/1642 of 1726 | 27 | 27/25 | 11 | 11/10 |
| 2 | short | B - C | 1626/1663/1642 of 1726 | 26 | 26/25 | 5 | 5/5 |
| 2 | xyx | B - A | 821/836/821 of 873 | 17 | 18/18 | 2 | 2/2 |
| 2 | xyx | C - A | 821/836/821 of 873 | 8 | 9/9 | 8 | 8/8 |
| 2 | xyx | B - C | 821/836/821 of 873 | 16 | 16/16 | 1 | 1/1 |
| 2 | a4c_a5c_xyx | B - A | 61/77/64 of 99 | 16 | 17/17 | 0 | 0/0 |
| 2 | a4c_a5c_xyx | C - A | 61/77/64 of 99 | 8 | 9/9 | 5 | 5/5 |
| 2 | a4c_a5c_xyx | B - C | 61/77/64 of 99 | 13 | 13/13 | 0 | 0/0 |

## Tokens

| seed | kind | block | sequences | active_equals_reference_runs | fewer | more | mean_active_minus_runs | all_null |
|---|---|---|---|---|---|---|---|---|
| 0 | stream | i | 1500 | 970 | 56 | 474 | 0.3440 | 0 |
| 0 | stream | u | 1500 | 997 | 56 | 447 | 0.3167 | 0 |
| 0 | stream | U | 1500 | 1178 | 28 | 294 | 0.2040 | 0 |
| 0 | native | i | 1500 | 1472 | 12 | 16 | 0.0067 | 12 |
| 0 | native | u | 1500 | 1480 | 3 | 17 | 0.0140 | 3 |
| 0 | native | U | 1500 | 1479 | 5 | 16 | 0.0120 | 5 |
| 1 | stream | i | 1500 | 895 | 60 | 545 | 0.4060 | 0 |
| 1 | stream | u | 1500 | 915 | 56 | 529 | 0.3840 | 0 |
| 1 | stream | U | 1500 | 1125 | 38 | 337 | 0.2240 | 0 |
| 1 | native | i | 1500 | 1462 | 24 | 14 | -0.0033 | 24 |
| 1 | native | u | 1500 | 1479 | 7 | 14 | 0.0080 | 7 |
| 1 | native | U | 1500 | 1484 | 6 | 10 | 0.0053 | 6 |
| 2 | stream | i | 1500 | 1006 | 67 | 427 | 0.2947 | 0 |
| 2 | stream | u | 1500 | 1010 | 72 | 418 | 0.2873 | 0 |
| 2 | stream | U | 1500 | 1188 | 48 | 264 | 0.1607 | 0 |
| 2 | native | i | 1500 | 1451 | 35 | 14 | -0.0113 | 35 |
| 2 | native | u | 1500 | 1460 | 25 | 15 | -0.0053 | 25 |
| 2 | native | U | 1500 | 1475 | 9 | 16 | 0.0073 | 9 |

## Across seeds

```
{
 "E1": {
  "all": {
   "all_seeds": {
    "segments": 791,
    "recordings": 542,
    "sessions": 288
   },
   "any_seed": {
    "segments": 791,
    "recordings": 542,
    "sessions": 288
   }
  },
  "short": {
   "all_seeds": {
    "segments": 109,
    "recordings": 84,
    "sessions": 69
   },
   "any_seed": {
    "segments": 109,
    "recordings": 84,
    "sessions": 69
   }
  },
  "xyx": {
   "all_seeds": {
    "segments": 101,
    "recordings": 106,
    "sessions": 85
   },
   "any_seed": {
    "segments": 101,
    "recordings": 106,
    "sessions": 85
   }
  },
  "a4c_a5c_xyx": {
   "all_seeds": {
    "segments": 27,
    "recordings": 36,
    "sessions": 34
   },
   "any_seed": {
    "segments": 27,
    "recordings": 36,
    "sessions": 34
   }
  }
 },
 "FACT_u": {
  "all": {
   "all_seeds": {
    "segments": 156,
    "recordings": 116,
    "sessions": 98
   },
   "any_seed": {
    "segments": 216,
    "recordings": 165,
    "sessions": 131
   }
  },
  "short": {
   "all_seeds": {
    "segments": 72,
    "recordings": 67,
    "sessions": 62
   },
   "any_seed": {
    "segments": 107,
    "recordings": 99,
    "sessions": 87
   }
  },
  "xyx": {
   "all_seeds": {
    "segments": 44,
    "recordings": 37,
    "sessions": 37
   },
   "any_seed": {
    "segments": 54,
    "recordings": 47,
    "sessions": 46
   }
  },
  "a4c_a5c_xyx": {
   "all_seeds": {
    "segments": 29,
    "recordings": 28,
    "sessions": 28
   },
   "any_seed": {
    "segments": 36,
    "recordings": 35,
    "sessions": 35
   }
  }
 },
 "FACT_U": {
  "all": {
   "all_seeds": {
    "segments": 148,
    "recordings": 109,
    "sessions": 95
   },
   "any_seed": {
    "segments": 213,
    "recordings": 165,
    "sessions": 132
   }
  },
  "short": {
   "all_seeds": {
    "segments": 68,
    "recordings": 64,
    "sessions": 62
   },
   "any_seed": {
    "segments": 103,
    "recordings": 95,
    "sessions": 86
   }
  },
  "xyx": {
   "all_seeds": {
    "segments": 40,
    "recordings": 33,
    "sessions": 33
   },
   "any_seed": {
    "segments": 53,
    "recordings": 47,
    "sessions": 46
   }
  },
  "a4c_a5c_xyx": {
   "all_seeds": {
    "segments": 26,
    "recordings": 25,
    "sessions": 25
   },
   "any_seed": {
    "segments": 36,
    "recordings": 36,
    "sessions": 36
   }
  }
 },
 "FACT_final": {
  "all": {
   "all_seeds": {
    "segments": 141,
    "recordings": 111,
    "sessions": 91
   },
   "any_seed": {
    "segments": 257,
    "recordings": 206,
    "sessions": 149
   }
  },
  "short": {
   "all_seeds": {
    "segments": 70,
    "recordings": 66,
    "sessions": 60
   },
   "any_seed": {
    "segments": 119,
    "recordings": 111,
    "sessions": 96
   }
  },
  "xyx": {
   "all_seeds": {
    "segments": 41,
    "recordings": 39,
    "sessions": 38
   },
   "any_seed": {
    "segments": 60,
    "recordings": 57,
    "sessions": 49
   }
  },
  "a4c_a5c_xyx": {
   "all_seeds": {
    "segments": 32,
    "recordings": 32,
    "sessions": 32
   },
   "any_seed": {
    "segments": 42,
    "recordings": 42,
    "sessions": 41
   }
  }
 },
 "ASRF_final": {
  "all": {
   "all_seeds": {
    "segments": 12,
    "recordings": 11,
    "sessions": 10
   },
   "any_seed": {
    "segments": 19,
    "recordings": 17,
    "sessions": 15
   }
  },
  "short": {
   "all_seeds": {
    "segments": 2,
    "recordings": 2,
    "sessions": 2
   },
   "any_seed": {
    "segments": 7,
    "recordings": 7,
    "sessions": 6
   }
  },
  "xyx": {
   "all_seeds": {
    "segments": 3,
    "recordings": 4,
    "sessions": 4
   },
   "any_seed": {
    "segments": 5,
    "recordings": 6,
    "sessions": 6
   }
  },
  "a4c_a5c_xyx": {
   "all_seeds": {
    "segments": 1,
    "recordings": 1,
    "sessions": 1
   },
   "any_seed": {
    "segments": 3,
    "recordings": 3,
    "sessions": 3
   }
  }
 },
 "native_windows_all_seeds": {
  "segments": 4500,
  "recordings": 2798,
  "sessions": 496
 }
}
```

## Cost

```
{
 "cap_hours": 1.0,
 "check_gpu_held_seconds": 13.218894958496094,
 "run_wall_seconds_per_seed": {
  "0": 279.5286955833435,
  "1": 270.7516074180603,
  "2": 271.1943187713623
 },
 "inference_seconds_per_seed": {
  "0": {
   "FACT": 250.8325982093811,
   "ASRF": 27.404942750930786
  },
  "1": {
   "FACT": 246.16739892959595,
   "ASRF": 23.675288677215576
  },
  "2": {
   "FACT": 246.43052291870117,
   "ASRF": 24.018608808517456
  }
 },
 "gpu_held_hours_before_this_summary": 0.23335021648142074,
 "summary_seconds": 19.831263542175293,
 "gpu_held_hours_total": 0.23885890099737378
}
```
