# Evaluation

## Run it

```bash
pip install -e ".[evaluation]"
echoview evaluate --images /data/EV9V/Images --out report.json --cache /data/echoview_cache
```

The command needs the EV9V `Images` folder (see [DATA.md](DATA.md)) and a GPU. It decodes about 40,000 frames, runs the three seeds over about 21,000 windows and writes the full report. `--cache` keeps the decoded frames (2 GB) for the next run. On one L40S the first run takes about 18 minutes.

It prints one line per system:

```text
C2_s0                        selection F1@50 0.9584  calibration F1@50 0.9856  recording errors 1/1  decoder h7 thr 0.1860 (feasible)
C2_s1                        selection F1@50 0.9476  calibration F1@50 0.9827  recording errors 1/1  decoder h5 thr 0.1297 (feasible)
C2_s2                        selection F1@50 0.9483  calibration F1@50 0.9778  recording errors 1/2  decoder h5 thr 0.1371 (feasible)
ensemble                     selection F1@50 0.9594  calibration F1@50 0.9837  recording errors 1/2  decoder h7 thr 0.1738 (feasible)
ensemble_published_setting   selection F1@50 0.9594  calibration F1@50 0.9837  recording errors 1/2  decoder h7 thr 0.1738 (fixed)
```

To evaluate seeds you trained yourself, put them in a folder as `C2_s0/final.pt`, `C2_s1/final.pt` and `C2_s2/final.pt`, and add `--weights <folder> --unverified`.

## What is evaluated

- **Each seed alone.** Its decoder setting is chosen on the calibration half by the rule below and applied unchanged to the selection half.
- **The ensemble.** The three seeds' probabilities are averaged. The setting is chosen in the same way.
- **The ensemble with its published setting.** The setting that ships with the model. It gives the same numbers as the rule's choice.

The evaluation uses plain float32 arithmetic on the GPU. TF32 is switched off, because it changes logits in the fourth decimal.

## Metrics

**Recordings.** A recording's view is the majority of its decoded sample labels.

- `native_macro_f1` is the mean F1 over the five views of that decision.
- `native_internal_cuts_per_clip` counts label changes inside one recording. There should be none.
- The `native` block of the report gives the recording-level decision from the mean window probability, its confusion table and its calibration error.

**Streams.**

- `segment_f1_50_multi`. Predicted and reference segments of the same view are matched one to one when their temporal intersection over union is at least 0.5, best match first. F1 is 2 × matched / (predicted + reference), pooled over all streams. This is the headline number.
- `boundary_recall_multi`, `boundary_precision_multi`, `boundary_f1_multi`. Predicted label changes are matched one to one to reference view changes at most 0.5 s away, nearest first.
- `same_class_cut_rate_multi`. Of the joins where two fragments of the same view meet, the share with a predicted change within 0.5 s. A model that cuts at every splice instead of at view changes scores high here.

**Where the errors are.** The blocks `streams_multi` and `streams_exam` describe the decoded streams further (`echoview/evaluation/scorecard.py`).

- `A1_piece`. Is the middle sample of every source recording given its view?
- `A1_segment`. Does the middle sample of every returned segment carry the segment's view?
- `wrong_split`. Every wrong sample falls in one of four classes: `edge` (the cut is a little off), `spurious` (an extra segment), `absorbed` (a piece swallowed by its neighbour) and `misread` (a whole piece given another view).
- `A4_cuts`. The signed time offsets of matched cuts.
- `A5_purity`, `A5_recovery`. Per view: the share of returned samples that are right, and the share of reference samples that are returned.
- `A6_f1`, `A6_edit`. Segment F1 at overlaps 0.10, 0.25 and 0.50 and the Edit score, as defined by MS-TCN.

## The decoder and its rule

The decoder scores every sample by the Jensen-Shannon divergence between the mean probabilities of the `half_window` samples before it and after it. A cut is a local peak of the score at or above the threshold at which the two sides also name different views.

The setting is chosen on the calibration half:

1. The grid is five half windows (5, 7, 9, 11, 15 samples) by eight thresholds. A threshold is a quantile (0.80 to 0.999) of the scores inside the single-view recordings of the calibration half.
2. A setting is feasible when the recording-level macro-F1 stays at least 0.980 and the boundary recall on multi streams at least 0.95.
3. Among the feasible settings the highest segment F1@50 on multi streams wins. An equal score goes to the earlier grid position.
4. When nothing is feasible, the report says so and shows the numbers at a fixed diagnostic setting.

## Checked against the research code

On 2026-11-17 this evaluation was run beside the research code on the same frozen weights, on one L40S.

| Check | Result |
|---|---|
| Window logits, 6 comparisons of about 9,900 to 11,000 windows each | largest difference 5.2e-6, no window changed its view |
| Decoder choice of every seed and of the ensemble | identical |
| Every decoded metric, both halves, 3 seeds and the ensemble | identical |
| The ensemble with its published setting against the rule's choice | identical numbers |

One difference is known. The published threshold of the ensemble, 0.17379, was computed on another GPU with TF32 and compiled code. Plain float32 gives 0.17383. Decoding with either gives the same numbers on both halves.

The tests of this repository repeat the comparison without a GPU: `tests/test_evaluation_parity.py` feeds stored predictions for the whole benchmark to the metrics, the scorecard and the decoder rule, and requires the research code's stored outputs.

## Not here yet

The EV9V test benchmark and the MIMIC-IV-Echo evaluations of the model card were produced by the research code. They are not part of this package yet.
