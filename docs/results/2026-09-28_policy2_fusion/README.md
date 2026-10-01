# Routing policy 2 in the ladder, segment-level fusion, and the A4C/A5C review (2026-09-28)

> **Pending recomputation (2026-09-30).** These results were computed on caches that read EV9V frames in name order (0.jpg, 1.jpg, 10.jpg, 100.jpg, …, 2.jpg), which shuffles frames in time within each clip. Every number that depends on the order of frames inside a clip is unreliable, and frame- and clip-level scores also shift because different frames were sampled. Do not cite these numbers until they are recomputed from `cache_v2`. See `docs/results/2026-09-30_frame_order_fix/`.

Ladder run `20260928-001145-baselines-1b027c50-s0`: the same config, streams and prediction directories as
`20260925-130154`. It adds routing policy 2 and four fusion methods, followed by
`scripts/bootstrap_ladder.py --reference b4` (2,000 draws, cines resampled on the 4–8 fragment streams) and
`scripts/report_policy2_fusion.py`.

## What changed

- **Routing policy 2 is part of every ladder run** (`evaluate_method`, `routing_by["p2@t"]`, `policy2` in the
  metrics).
  - Segments are scored with the method's own pipeline evidence before temporal smoothing (`registry.FRAME_STAGE`).
  - B1–B5 use their ResNet-18 frame classifier (B0). ViT-Router and MS-TCN on ViT-S use their ViT-S frame
    classifier (D1). All other methods are their own frame or window stage, so they are scored with their own outputs.
  - No model outside a method's pipeline is used.
  - Score: the segment mean of the evidence probability of the segment's view.
  - Thresholds: one per predicted view, at the 95% upper bound of validation risk under a bootstrap that resamples
    cines. They are selected on the 120 multi-fragment validation streams and then frozen.
- **Fusion methods `SEG+CLS~α`** (`registry.parse_fused`). Boundaries come from SEG (MS-TCN). Each segment is
  relabelled with the argmax of the segment mean of (1 − α)·SEG + α·CLS probabilities. Their evidence for policy 2 is
  the same mixture of the two pipelines' evidence. The partners are EchoViewCLIP (`b7`) and Echo-MViT + centre, each
  at α = 0.5 and α = 1.

## Findings

1. **Policy 2 holds the 5% target for 24 of 28 methods on the 4–8 fragment test streams.**
   - Two misses are marginal and their intervals include 5%: EchoViewCLIP 0.052 (0.029–0.085) and Echo-MViT +
     centre 0.054 (0.032–0.082).
   - The other two are the MViT-Kinetics runs. They route only 4–5% of the time, so their error rate rests on very
     few segments.
   - The legacy rule holds 5% for almost none of the methods.
2. **Own-pipeline evidence equals the ResNet-evidence result for B1–B5.** MS-TCN routes 0.85 at 2.7% error and
   ASFormer 0.85 at 2.7%.
   - Pipelines without a ResNet stage route less. ViT-Router with its own ViT-S evidence routes 0.70 at 2.8%, against
     0.82 with ResNet evidence in the 2026-09-26 study.
   - Window-level outputs of clip models are weaker evidence than frame probabilities: STFM 0.46, EchoViewCLIP 0.59.
3. **Fusion.** α is chosen on validation routing coverage, and α = 0.5 wins for both partners (validation coverage at
   5%: 0.92 against 0.82–0.86).

   | model | test clip macro-F1 | A5C F1 | change F1, 4–8 clips | policy 2, 5% | policy 2, 1% |
   |---|---|---|---|---|---|
   | MS-TCN | 0.956 | 0.842 | 0.921 | 0.85 / 0.027 | 0.78 / 0.016 |
   | MS-TCN + EchoViewCLIP, α 0.5 | 0.962 | 0.857 | 0.922 | 0.87 / 0.026 | 0.84 / 0.014 |
   | MS-TCN + Echo-MViT, α 0.5 | 0.963 | 0.857 | 0.925 | 0.88 / 0.028 | 0.80 / 0.009 |
   | MS-TCN + EchoViewCLIP, α 1 | 0.971 | 0.889 | 0.935 | 0.83 / 0.036 | 0.61 / 0.011 |

   Paired differences to MS-TCN, cine-resampled:
   - Both α = 0.5 fusions route more at the same error: +2.2 points at 5% (+0.8 to +3.8) with EchoViewCLIP and
     +3.4 points (+1.7 to +5.2) with Echo-MViT.
   - With EchoViewCLIP they also route +5.8 points (+3.5 to +8.5) at 1%.
   - α = 1 reaches EchoViewCLIP-level recognition and makes fewer false cuts (−2.5 points), but misses more
     A4C↔A5C changes (+1.6 points) and routes less.
4. **Fusion does not solve A4C↔A5C changes.** They are found in 59–66% of cases whatever the partner, the same as
   MS-TCN alone.
5. **The 1% thresholds are unstable.** MS-TCN's 1% result moved from 0.71 / 0.004 in the 2026-09-26 study to
   0.78 / 0.016 here. The only difference is the bootstrap draws used for selection, which is expected with 29 A5C
   cines in validation.

## A4C/A5C review page (local only)

`scripts/a5c_audit.py` writes `runs/_demo/a5c_audit/index.html`, which git ignores and which is not published. The
clip list is in `a5c_audit_clips.json`.

- **6 A5C test clips** are misread by at least 5 of the 6 main models, almost always as A4C. Two of them come from the
  same session (`2022-05-24_14-30-39`) as a clip that every model reads correctly as A5C.
- **1 A4C test clip** (`2022-05-05_10-38-37_320x240_1`) is called A5C by all seven models. It is a candidate label
  error.
- Clear A5C and A4C references are included for side-by-side viewing.
- Whether the six A5C clips are genuinely ambiguous or clearly A5C needs a human reader.

## Files

- `policy2_fusion.md`: both tables.
- `bootstrap.md`: every endpoint with intervals and paired differences to MS-TCN.
- `ladder.md` / `ladder.csv`.
- `a5c_audit_clips.json`.
