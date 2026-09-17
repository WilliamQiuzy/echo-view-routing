# Five baselines on EV9V (run `20260917-175144-baselines-dc6df608-s0`, encoder 79f4a41af6e3)

Same test data for every model: 800 untouched test clips (five-family task), 240 two-fragment streams (2×2 design: same/different view × no/gamma edit) and 120 multi-fragment streams. Thresholds chosen on the validation split at a 5% contamination target and frozen. Clip-level classifiers are applied with a 1.6 s sliding window (stride 0.5 s); EchoPrime per frame; MS-TCN and ASFormer on frozen frame features.

## Recognition (800 test clips)

| model | clip acc | clip macro-F1 | balanced acc | frame acc | frame macro-F1 | fragments / min | share fragmented |
|---|---|---|---|---|---|---|---|
| STFM (official, sliding window) | 0.984 | 0.960 | 0.959 | 0.981 | 0.937 | 0.51 | 0.029 |
| EchoViewCLIP (official, sliding window) | 0.989 | 0.972 | 0.975 | 0.979 | 0.935 | 0.72 | 0.039 |
| EchoPrime (released weights, per frame) | 0.979 | 0.945 | 0.926 | 0.955 | 0.890 | 19.26 | 0.312 |
| MS-TCN (official) | 0.981 | 0.956 | 0.949 | 0.977 | 0.928 | 1.01 | 0.037 |
| ASFormer (official) | 0.984 | 0.959 | 0.951 | 0.980 | 0.936 | 0.80 | 0.030 |

## Segmentation (constructed test streams)

| model | boundary F1 @0.25 s | @0.5 s | @1.0 s | median timing error (s) | false split same/none | same/edit | missed diff/none | diff/edit | multi-fragment boundary F1 @0.5 s |
|---|---|---|---|---|---|---|---|---|---|
| STFM (official, sliding window) | 0.639 | 0.963 | 0.979 | 0.20 | 0.000 | 0.000 | 0.033 | 0.033 | 0.859 |
| EchoViewCLIP (official, sliding window) | 0.689 | 0.975 | 0.975 | 0.10 | 0.000 | 0.000 | 0.017 | 0.000 | 0.888 |
| EchoPrime (released weights, per frame) | 0.507 | 0.507 | 0.507 | 0.00 | 0.100 | 0.117 | 0.000 | 0.000 | 0.482 |
| MS-TCN (official) | 0.992 | 0.992 | 0.992 | 0.00 | 0.000 | 0.017 | 0.000 | 0.000 | 0.921 |
| ASFormer (official) | 0.980 | 0.980 | 0.980 | 0.00 | 0.000 | 0.017 | 0.000 | 0.000 | 0.901 |

## Selective routing (validation-selected thresholds, frozen)

| model | τ @5% | test coverage @5% | achieved risk | τ @1% | validation coverage @1% | multi-fragment coverage | multi-fragment risk |
|---|---|---|---|---|---|---|---|
| STFM (official, sliding window) | 0.240 | 0.996 | 0.028 | 0.568 | 0.620 | 0.994 | 0.108 |
| EchoViewCLIP (official, sliding window) | 0.037 | 0.992 | 0.025 | 0.837 | 0.505 | 1.000 | 0.110 |
| EchoPrime (released weights, per frame) | 0.184 | 0.974 | 0.007 | 0.687 | 0.919 | 0.932 | 0.088 |
| MS-TCN (official) | 0.802 | 1.000 | 0.008 | 0.999 | 0.701 | 0.996 | 0.089 |
| ASFormer (official) | 0.564 | 1.000 | 0.007 | 0.993 | 0.685 | 0.998 | 0.087 |

## Reading

- All five models reach 0.98–0.99 clip accuracy on untouched clips; EchoViewCLIP has the highest macro-F1 (0.972).
- Temporal models (MS-TCN, ASFormer) locate view changes best on two-fragment streams (boundary F1 0.98–0.99 at every tolerance); the windowed clip classifiers reach 0.96–0.98 at 0.5 s but only 0.64–0.69 at 0.25 s because the window blurs the boundary.
- EchoPrime applied per frame flickers (19 fragments/min) and splits 10–12% of same-view joins; it is a frame classifier without temporal context.
- On multi-fragment streams every model's contamination rises to 9–11% at the 5% policy chosen on two-fragment streams; the 1% target is only met with reduced coverage (0.5–0.9).
- Windowed STFM/EchoViewCLIP miss 2–3% of true changes; no model produces false splits on unedited same-view joins.
