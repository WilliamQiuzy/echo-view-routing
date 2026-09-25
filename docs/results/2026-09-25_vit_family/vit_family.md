# Vision-Transformer family on EV9V

Ladder run `20260925-130154-baselines-1b027c50-s0` (identical streams, thresholds and metrics for every row).

## Nine-code cine-level classification (official EV9V test split, 888 cines)

| model | test acc | test macro-F1 | val acc | seeds | cine aggregation |
|---|---|---|---|---|---|
| STFM (official, ResNet-18; 3 paper seeds) | 0.9377 ± 0.0017 | 0.8988 ± 0.0029 | — | — | authors' test(): 10 key frames x 5-frame clips |
| EchoViewCLIP (official, CLIP ViT-B/16) | 0.9414 | 0.907 | — | — | authors' validate_: 16 frames per video |
| **Factorised video ViT (ViT-S + temporal Transformer), sliding window** | 0.9279 ± 0.0023 | 0.8853 ± 0.0091 | 0.9406 ± 0.0037 | 3 | mean softmax over 1.6 s windows (stride 0.5 s) covering the cine |
| **Factorised video ViT, transition-aware (ours), sliding window** | 0.9223 | 0.8769 | 0.9347 | 1 | mean softmax over 1.6 s windows (stride 0.5 s) covering the cine |
| **Factorised video ViT, centre-supervised transition-aware (ours), sliding window** | 0.9043 | 0.8574 | 0.9259 | 1 | mean softmax over 1.6 s windows (stride 0.5 s) covering the cine |
| **MViTv2-S (Kinetics-400 init), sliding window** | 0.9352 ± 0.0056 | 0.8976 ± 0.0076 | 0.9444 ± 0.0062 | 2 | mean softmax over 1.6 s windows (stride 0.5 s) covering the cine |
| **Echo-MViTv2-S (EchoPrime init), sliding window** | 0.9274 ± 0.0008 | 0.8860 ± 0.0021 | 0.9559 ± 0.0000 | 2 | mean softmax over 1.6 s windows (stride 0.5 s) covering the cine |
| **Echo-MViTv2-S, centre-supervised transition-aware (ours), sliding window** | 0.9343 ± 0.0069 | 0.8983 ± 0.0061 | 0.9565 ± 0.0037 | 3 | mean softmax over 1.6 s windows (stride 0.5 s) covering the cine |

## Recognition (800 untouched test clips, five-family task)

| model | clip acc | clip macro-F1 | balanced acc | frame acc | frame macro-F1 | fragments / min | share fragmented |
|---|---|---|---|---|---|---|---|
| ResNet-18 frame encoder (B0), per frame | 0.983 | 0.957 | 0.962 | 0.951 | 0.880 | 23.14 | 0.307 |
| **ViT-S/16 frame, per frame** | 0.976 | 0.935 | 0.929 | 0.953 | 0.878 | 25.38 | 0.324 |
| EchoPrime view classifier (ConvNeXt-B, per frame) | 0.979 | 0.945 | 0.926 | 0.955 | 0.890 | 19.26 | 0.312 |
| STFM (official, sliding window) | 0.984 | 0.960 | 0.959 | 0.981 | 0.937 | 0.51 | 0.029 |
| EchoViewCLIP (official, CLIP ViT-B/16, sliding window) | 0.989 | 0.972 | 0.975 | 0.979 | 0.935 | 0.72 | 0.039 |
| **Factorised video ViT (ViT-S + temporal Transformer), sliding window** | 0.970 | 0.930 | 0.945 | 0.963 | 0.901 | 1.37 | 0.066 |
| **Factorised video ViT, transition-aware (ours), sliding window** | 0.970 | 0.932 | 0.942 | 0.960 | 0.899 | 1.45 | 0.076 |
| **Factorised video ViT, centre-supervised transition-aware (ours), sliding window** | 0.959 | 0.918 | 0.950 | 0.955 | 0.892 | 1.20 | 0.066 |
| **MViTv2-S (Kinetics-400 init), sliding window** | 0.979 | 0.951 | 0.960 | 0.972 | 0.921 | 1.56 | 0.066 |
| **Echo-MViTv2-S (EchoPrime init), sliding window** | 0.975 | 0.945 | 0.970 | 0.969 | 0.914 | 1.33 | 0.059 |
| **Echo-MViTv2-S, centre-supervised transition-aware (ours), sliding window** | 0.978 | 0.942 | 0.938 | 0.974 | 0.918 | 0.93 | 0.048 |
| MS-TCN (official) on ResNet-18 features | 0.981 | 0.956 | 0.949 | 0.977 | 0.928 | 1.01 | 0.037 |
| ASFormer (official) on ResNet-18 features | 0.984 | 0.959 | 0.951 | 0.980 | 0.936 | 0.80 | 0.030 |
| **MS-TCN (official) on ViT-S features** | 0.971 | 0.927 | 0.926 | 0.966 | 0.900 | 1.75 | 0.059 |
| **ViT-Router (ours) on ViT-S features** | 0.973 | 0.920 | 0.906 | 0.974 | 0.908 | 0.86 | 0.034 |

## Segmentation (240 two-fragment + 120 multi-fragment test streams)

| model | boundary F1 @0.25 s | @0.5 s | @1.0 s | median timing error (s) | false split same/none | same/edit | missed diff/none | diff/edit | multi-fragment boundary F1 @0.5 s |
|---|---|---|---|---|---|---|---|---|---|
| ResNet-18 frame encoder (B0), per frame | 0.499 | 0.499 | 0.499 | 0.00 | 0.033 | 0.133 | 0.000 | 0.000 | 0.473 |
| **ViT-S/16 frame, per frame** | 0.483 | 0.487 | 0.487 | 0.00 | 0.133 | 0.117 | 0.000 | 0.000 | 0.442 |
| EchoPrime view classifier (ConvNeXt-B, per frame) | 0.507 | 0.507 | 0.507 | 0.00 | 0.100 | 0.117 | 0.000 | 0.000 | 0.482 |
| STFM (official, sliding window) | 0.639 | 0.963 | 0.979 | 0.20 | 0.000 | 0.000 | 0.033 | 0.033 | 0.859 |
| EchoViewCLIP (official, CLIP ViT-B/16, sliding window) | 0.689 | 0.975 | 0.975 | 0.10 | 0.000 | 0.000 | 0.017 | 0.000 | 0.888 |
| **Factorised video ViT (ViT-S + temporal Transformer), sliding window** | 0.217 | 0.579 | 0.887 | 0.30 | 0.000 | 0.000 | 0.550 | 0.383 | 0.603 |
| **Factorised video ViT, transition-aware (ours), sliding window** | 0.192 | 0.576 | 0.847 | 0.40 | 0.000 | 0.000 | 0.450 | 0.450 | 0.596 |
| **Factorised video ViT, centre-supervised transition-aware (ours), sliding window** | 0.661 | 0.975 | 0.975 | 0.10 | 0.000 | 0.000 | 0.017 | 0.017 | 0.856 |
| **MViTv2-S (Kinetics-400 init), sliding window** | 0.357 | 0.732 | 0.911 | 0.30 | 0.000 | 0.000 | 0.217 | 0.350 | 0.657 |
| **Echo-MViTv2-S (EchoPrime init), sliding window** | 0.494 | 0.765 | 0.914 | 0.20 | 0.017 | 0.017 | 0.250 | 0.200 | 0.738 |
| **Echo-MViTv2-S, centre-supervised transition-aware (ours), sliding window** | 0.750 | 0.960 | 0.960 | 0.10 | 0.000 | 0.000 | 0.017 | 0.000 | 0.870 |
| MS-TCN (official) on ResNet-18 features | 0.992 | 0.992 | 0.992 | 0.00 | 0.000 | 0.017 | 0.000 | 0.000 | 0.921 |
| ASFormer (official) on ResNet-18 features | 0.980 | 0.980 | 0.980 | 0.00 | 0.000 | 0.017 | 0.000 | 0.000 | 0.901 |
| **MS-TCN (official) on ViT-S features** | 0.952 | 0.960 | 0.968 | 0.00 | 0.000 | 0.000 | 0.000 | 0.017 | 0.843 |
| **ViT-Router (ours) on ViT-S features** | 0.971 | 0.988 | 0.988 | 0.00 | 0.000 | 0.017 | 0.000 | 0.000 | 0.891 |

## Selective routing (thresholds chosen on validation at 5% target risk, frozen)

| model | τ @5% | test coverage @5% | achieved risk | τ @1% | val coverage @1% | multi-fragment coverage | multi-fragment risk |
|---|---|---|---|---|---|---|---|
| ResNet-18 frame encoder (B0), per frame | 0.461 | 0.967 | 0.007 | 0.740 | 0.894 | 0.921 | 0.052 |
| **ViT-S/16 frame, per frame** | 0.687 | 0.973 | 0.004 | 1.000 | 0.709 | 0.913 | 0.064 |
| EchoPrime view classifier (ConvNeXt-B, per frame) | 0.184 | 0.974 | 0.007 | 0.687 | 0.919 | 0.932 | 0.088 |
| STFM (official, sliding window) | 0.240 | 0.996 | 0.028 | 0.568 | 0.620 | 0.994 | 0.108 |
| EchoViewCLIP (official, CLIP ViT-B/16, sliding window) | 0.037 | 0.992 | 0.025 | 0.837 | 0.505 | 1.000 | 0.110 |
| **Factorised video ViT (ViT-S + temporal Transformer), sliding window** | 0.976 | 0.726 | 0.034 | 0.999 | 0.338 | 0.508 | 0.103 |
| **Factorised video ViT, transition-aware (ours), sliding window** | 0.688 | 0.724 | 0.047 | 0.928 | 0.462 | 0.512 | 0.124 |
| **Factorised video ViT, centre-supervised transition-aware (ours), sliding window** | 0.001 | 1.000 | 0.038 | 0.992 | 0.391 | 1.000 | 0.119 |
| **MViTv2-S (Kinetics-400 init), sliding window** | 0.990 | 0.696 | 0.034 | 0.999 | 0.024 | 0.493 | 0.098 |
| **Echo-MViTv2-S (EchoPrime init), sliding window** | 0.793 | 0.865 | 0.045 | 0.991 | 0.319 | 0.830 | 0.113 |
| **Echo-MViTv2-S, centre-supervised transition-aware (ours), sliding window** | 0.021 | 0.986 | 0.024 | 0.972 | 0.587 | 0.995 | 0.116 |
| MS-TCN (official) on ResNet-18 features | 0.802 | 1.000 | 0.008 | 0.999 | 0.701 | 0.996 | 0.089 |
| ASFormer (official) on ResNet-18 features | 0.564 | 1.000 | 0.007 | 0.993 | 0.685 | 0.998 | 0.087 |
| **MS-TCN (official) on ViT-S features** | 0.920 | 0.998 | 0.007 | 0.996 | 0.543 | 0.973 | 0.079 |
| **ViT-Router (ours) on ViT-S features** | 0.810 | 0.999 | 0.003 | 0.999 | 0.472 | 0.997 | 0.112 |

## Seed robustness (ladder, mean ± std over training seeds)

| model | seeds | clip macro-F1 | balanced acc | boundary F1 @0.5 s | coverage @5% | achieved risk |
|---|---|---|---|---|---|---|
| **Factorised video ViT (ViT-S + temporal Transformer), sliding window** | 3 | 0.933 ± 0.007 | 0.935 ± 0.016 | 0.685 ± 0.102 | 0.827 ± 0.100 | 0.042 ± 0.007 |
| **MViTv2-S (Kinetics-400 init), sliding window** | 2 | 0.953 ± 0.002 | 0.962 ± 0.003 | 0.747 ± 0.021 | 0.706 ± 0.014 | 0.040 ± 0.008 |
| **Echo-MViTv2-S (EchoPrime init), sliding window** | 2 | 0.946 ± 0.002 | 0.966 ± 0.005 | 0.762 ± 0.005 | 0.807 ± 0.083 | 0.042 ± 0.004 |
| **Echo-MViTv2-S, centre-supervised transition-aware (ours), sliding window** | 3 | 0.947 ± 0.008 | 0.952 ± 0.012 | 0.938 ± 0.049 | 0.984 ± 0.010 | 0.031 ± 0.012 |
