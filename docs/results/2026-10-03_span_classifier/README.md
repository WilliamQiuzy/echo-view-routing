# Span-level A4C/A5C training, round 6 (2026-10-03): frozen against fine-tuned visual features

This round asks one question. Does training an A4C/A5C classifier directly on actual short spans improve the span
labels and reduce false pieces, and does any gain need the visual encoder to change? Round 2 already fine-tuned the
last ViT block, through frame-level and boundary losses. The new element here is a span-level objective: the model is
trained on the decision for a whole short span, not on every frame separately.

No echo expert is available, so all public labels are kept. The paper stays paused. Both validation halves are
development data, and the test split was not read.

- **Spans.** Both models classify the same spans: the candidate spans of the frozen E1 ASRF from round 5, on the
  training split and on validation. The encoder never changes the spans or ASRF's vote.
  - A span voted A4C or A5C is classified in either direction. Every other span keeps its vote, and equal neighbours
    merge afterwards.
- **V0.** It trains a small head on the mean of the span's frozen frame embeddings (512-d).
- **V1.** It trains the same head together with the last ViT block, the final normalisation and the projection. It
  starts from the same weights and head initialisation, with the same spans, frames, batch order and budget.
- **Seeds.** Three seeds were run, with model seed k on ASRF seed k.

## Answer

The pre-declared terms and rules (`configs/experiments/span_classifier_v1.yaml`) give **"neither improves"**. Under the
user's rule, this short-span classification route ends. It does not lead automatically to a larger encoder or extra
modules.

1. **Neither model improves span classification by the pre-declared measure.** That measure is accuracy above ASRF's
   vote in all three seeds on both halves.
   - V1 is above the vote in 3 of 6 seed-half cells and below it in the other 3.
   - V0 is below the vote in all six.
2. **Neither model improves the full system.**
   - V0 is consistently lower than ASRF's default output on the selection half (−2.3 to −4.1 points of segment F1@50).
   - V1 shows no consistent difference from it on either half (−3.2 to +0.2 points).
   - Both are consistently lower than the strongest previous configuration on the selection half, and neither passes
     the success test.
3. **V1 does not beat V0 consistently.** V1 − V0 is +0.2 to +1.9 points of segment F1@50 per seed. Only one of the six
   intervals excludes zero.
4. **Both models add false pieces against ASRF's default.** Under the pre-declared count this holds in all three seeds
   on the selection half.

**Observations.** These are not conclusions under the rules.
- **V1 against V0.** V1 is numerically ahead of V0 in every seed-half cell, in segment F1@50, span accuracy (+0.7 to
  +3.1 points) and A5C recall (+1.7 to +6.4 points). It also has fewer A4C pieces inside A5C segments.
- **Against ASRF.** Against ASRF's default, V1 recovers 9 to 10 A5C segments per seed on the selection half, from 4 to
  6 clips. It also turns 4 to 5 A4C segments wrong.
- **Training fit.** V1 fits the training spans much better than V0 (final loss 0.15 against 0.24–0.28). Its
  validation gain is far smaller.
- **ASRF's vote.** The vote, which uses ASRF's own temporal context, remains a strong reference for these spans.

**What this does not show.** This round does not show that visual features cannot separate A4C from A5C. It shows only
that this objective, on these spans, with this head and budget, does not beat ASRF's own decisions.

## 1. What was fixed before training

`configs/experiments/span_classifier_v1.yaml` was written before any round-6 training. It fixes:
- the spans, labels, weights and frame rule;
- the encoder, head, trainable parts, optimiser, batches, budget and decision;
- the evaluation;
- the unchanged success criterion, with the floors fixed at 0.973 and 0.980;
- the definitions of "improves" (section `terms`) and the user's four interpretation rules.

**Checks before use.**
- Parity with the stored reference embeddings was 2.9e-6, and with the E1 cache 4.9e-4.
- ASRF's decode reproduced its stored predictions exactly.
- V0 trained no encoder parameter. V1 trained exactly `transformer.resblocks.11`, `ln_post` and `proj`, and its
  parameter digest changed from start to end.
- A short sanity run (256 spans, one epoch) ran end to end before the full runs. It is not reported.

## 2. Training and validation spans

| Part | Spans with an A4C or A5C majority, per seed | A5C share | Median length | Mixed spans | Vote differs from the majority |
|---|---|---:|---:|---:|---:|
| Training split | 6,120 / 4,681 / 5,251 | 44–47% | 16–21 | 1.8–2.4% | 6.4–9.7% |
| Selection half | 556 / 463 / 514 | 35–39% | 17–21 | 2.7–3.7% | 10.4–11.0% |
| Calibration half | 530 / 416 / 447 | 38–40% | 16–21 | 1.7–2.9% | 7.7–12.3% |

The span counts differ by seed because each seed uses its own ASRF. Training and validation spans have similar lengths
and mixing.

Training spans come from a frozen ASRF that was trained on the same streams, so its vote is slightly more often right
there. On validation it differs from the majority in 7.7–12.3% of spans; on the training split, in 6.4–9.7%.

## 3. Span classification

Accuracy on validation spans with an A4C or A5C majority, per seed. The vote is ASRF's own class for the span. The
mean is round 5's rule on the E1 probe's probabilities.

| Half | Vote | Mean rule | V0 | V1 |
|---|---|---|---|---|
| Selection | 0.896 / 0.890 / 0.891 | 0.840 / 0.860 / 0.858 | 0.858 / 0.875 / 0.875 | 0.883 / 0.901 / 0.883 |
| Calibration | 0.877 / 0.923 / 0.899 | 0.843 / 0.923 / 0.902 | 0.843 / 0.921 / 0.897 | 0.874 / 0.928 / 0.906 |

A5C recall on the selection half is 0.853 / 0.830 / 0.822 for V1, against 0.792 / 0.772 / 0.802 for V0 and 0.934 /
0.731 / 0.777 for the vote. A4C recall is 0.900 / 0.942 / 0.923 for V1, against 0.875 / 0.983 / 0.965 for the vote. All
classes and clip-level shares are in `report_tables.md`.

## 4. Full system

Segment F1@50 per seed (three-seed mean). Targets are 0.944 (selection) and 0.982 (calibration).

| System | Selection | Calibration | Passes |
|---|---|---|---|
| ASRF default | 0.910 / 0.909 / 0.907 (0.909) | 0.976 / 0.964 / 0.964 (0.968) | no |
| Mean rule | 0.856 / 0.883 / 0.879 (0.873) | 0.931 / 0.964 / 0.960 (0.952) | no |
| V0 | 0.869 / 0.885 / 0.883 (0.879) | 0.924 / 0.960 / 0.953 (0.946) | no |
| V1 | 0.887 / 0.897 / 0.884 (0.889) | 0.943 / 0.967 / 0.964 (0.958) | no |

Boundary recall is at least 0.958 for V0 and V1 on both halves.

Native macro-F1 is below the floor in some seeds.
- V1 has 0.968 / 0.978 / 0.978 on the selection half, against a floor of 0.973.
- V1 has 0.986 / 0.976 / 0.983 on the calibration half, against 0.980.
- V0 is below the selection floor in all three seeds (0.951 / 0.968 / 0.968).

Paired per-seed contrasts of segment F1@50 (points; "none" means no consistent difference detected):

| Contrast | Selection | Calibration |
|---|---|---|
| V1 − V0 | +1.8 / +1.1 / +0.2, none | +1.9 / +0.7 / +1.1, none |
| V0 − ASRF default | −4.1 / −2.3 / −2.4, consistently lower | −5.2 / −0.4 / −1.2, none |
| V1 − ASRF default | −2.3 / −1.2 / −2.3, none | −3.2 / +0.2 / −0.1, none |
| V0 − mean rule | +1.3 / +0.2 / +0.4, none | −0.7 / −0.4 / −0.7, none |
| V1 − mean rule | +3.0 / +1.3 / +0.6, none | +1.2 / +0.3 / +0.4, none |
| V1 − strongest previous | −4.8 / −3.8 / −5.0, consistently lower | −2.9 / −0.6 / −0.8, none |

**Segments recovered and lost against ASRF's default** (stream segments, per seed):

| Model | Half | A5C turned right | A4C turned wrong |
|---|---|---|---|
| V0 | selection | 5 / 4 / 4 | 0 / 3 / 4 |
| V0 | calibration | 3 / 5 / 5 | 2 / 2 / 3 |
| V1 | selection | 9 / 9 / 10 | 4 / 4 / 5 |
| V1 | calibration | 3 / 6 / 6 | 2 / 1 / 2 |

Clip counts and native clips are in `report_tables.md`.

**False pieces** (unmatched predicted segments inside a reference segment of the other class), per seed:

| System | A5C pieces in A4C, selection / calibration | A4C pieces in A5C, selection / calibration |
|---|---|---|
| ASRF default | 9, 2, 2 / 6, 3, 3 | 8, 18, 19 / 2, 7, 8 |
| V0 | 13, 9, 11 / 14, 7, 8 | 30, 28, 25 / 28, 9, 11 |
| V1 | 15, 10, 13 / 11, 6, 7 | 17, 21, 24 / 16, 6, 7 |

**Without the ten sensitivity clips** (dependence only), the result does not change. V1 − V0 is −0.1 to 0.0 on the
selection half and +0.4 to +1.4 on the calibration half, with no consistent difference. Both models are numerically
below ASRF's default in every seed and half, with no consistent difference.

## 5. Reading under the pre-declared rules

| Term (config section `terms`) | V0 | V1 |
|---|---|---|
| Span accuracy above the vote in all seeds on both halves | no | no |
| Full system consistently above ASRF's default on a half and not consistently below on the other | no | no |
| Summed false pieces above ASRF's default in at least two seeds on a half | yes | yes |
| V1 consistently above V0 on a half and not consistently below on the other | | no |

Neither model improves, so the rule "neither improves" applies, and the short-span classification route ends. Neither
the "V0 improves" rule nor the "V1 beats V0" rule applies. The "classification improves but false pieces increase"
rule does not apply either, because classification does not improve by the pre-declared measure.

**How the report applied two words the config left implicit.** "Improves" counts as either classification improving or
the full system improving, as defined in `terms`. "The full system stays behind" means that no model passes the
success test on both halves. Both readings are spelled out in `scripts/round6_report.py`. The conclusion does not
depend on them, because both models fail both improvement terms.

**Code review.** The review found that the first version made "stays behind" reachable only when classification
improved. That was fixed, and the regenerated report is identical (`runs/md_round6/report_v2/`).

## 6. Limitations

- Labels are public clip labels, and no expert has reviewed any clip. Both halves are development data.
- The training spans come from an ASRF that was trained on those streams, so they are slightly cleaner than the
  validation spans (section 2).
- One budget (8 epochs) and one head were tried, by design. Nothing was tuned on validation data, so a different
  budget might fit better. That would be a new experiment.
- The bootstrap intervals resample sessions and streams but not repeated use of a clip across streams. They are
  optimistic, and the three seeds are not independent test sets.

## 7. Files and runs

- **Configuration:** `configs/experiments/span_classifier_v1.yaml`.
- **Code:**
  - `echo_routing/temporal/span_classifier.py`, with tests in `tests/test_span_classifier.py`;
  - `scripts/round6_span_classifier.py` (one seed per run) and `scripts/round6_report.py`.
- **Runs** (server):
  - seed 0 `runs/20261003-164305-md6_span_classifier_s0-32bced97-s0`;
  - seed 1 `runs/20261003-171737-md6_span_classifier_s1-6303199e-s1`;
  - seed 2 `runs/20261003-175337-md6_span_classifier_s2-d7e3e64b-s2`;
  - report `runs/md_round6/report_v2/` (the first build, `report/`, has identical tables).
  - Each seed run holds its rows, predictions and `seed.json`. The heads and V1's trained encoder parameters are in
    the matching folder under `checkpoints/`.
- **In this folder:** `report_tables.md` and `report.json`.
