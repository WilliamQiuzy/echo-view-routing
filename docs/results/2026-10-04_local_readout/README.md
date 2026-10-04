# Round 7 (2026-10-04): global against local readouts of the frozen E1 encoder

This round tested one hypothesis. Do the patch tokens of the frozen E1 encoder carry A4C/A5C information that its
global vector does not readily give? This was untested, not a known defect. The ViT already processes local regions;
only the way its features are read out had not been compared. The project lead set the design on 2026-10-04, after the
CETUS intake. CETUS was not used, no encoder parameter trained, the test split was not read, and the paper stays
paused.

Three readouts classify the same spans as A4C or A5C.
- **P0** is round 6's V0, the mean of each frame's global vector, with round 6's span head.
- **P1** reads each frame's global token through a small residual module, then weighs the frames with learned
  attention.
- **P2** reads all 197 tokens of each frame (the global token and 196 patch tokens) with learned attention over the
  tokens first, then weighs the frames with the same temporal attention.

P1 and P2 read the same layer at the same width, see identical tokens from one forward pass, use the same batches and
start from the same head and temporal weights. P1's residual module has as many parameters as P2's token attention
(329,091 against 328,836 trainable parameters in all). P2 − P1 is the main comparison. Because the global token is
among P2's tokens, a gain of P2 would mean the patch tokens add something. The attention pooling follows Ilse et al.
(ICML 2018); it is an established method, not a contribution here.

The design, budget, statistics and stop rules were committed before any training (`configs/experiments/local_readout_v1.yaml`,
commit 07cd481). Spans, frames, labels, weights, seeds, optimiser and budget are round 6's.

## Answer

Under the pre-declared rule, **the hypothesis ends**. Read this way, the patch tokens add no detectable A4C/A5C
information to the global token.

1. **Span classification.** P2 − P1 shows no consistent difference in accuracy, A4C recall, A5C recall or separation
   on either half.
   - Accuracy differs by −1.1 to +0.2 points per seed on the selection half and by −0.2 to +0.9 points on the
     calibration half.
   - Separation is the AUC of the A5C − A4C margin. It is lower for P2 in all six seed-half cells, by 0.3 to 0.7
     points, though no interval excludes zero.
2. **Full system.** P2 − P1 shows no consistent difference in segment F1@50 (−0.5 to +0.6 points), boundary recall,
   native macro-F1 or cuts. Between P2 and P1, no A5C unit turns right and no A4C unit turns wrong in any seed or half.
3. **Against the references.** On the selection half, all three arms are consistently below ASRF's default (−2.3 to
   −4.5 points) and below the strongest previous configuration (−4.9 to −7.0 points). None meets the final goal on
   either half. Against the highest previous calibration mean instead of the fixed reference, no verdict changes.
4. **An observation outside the main question.** Learned temporal weighting ranks spans better than the plain mean.
   P1 − P0 separation is consistently higher on the calibration half (+1.2 to +3.4 AUC points) and higher in every
   seed on the selection half (+0.5 to +0.7, not consistent). Accuracy and the full system do not improve
   consistently. The better ranking does not turn into better decisions at the fixed threshold of 0.5.

**What this does not show.** It does not show that local image information is useless for A4C/A5C. It covers one
readout: the frozen encoder's last-layer tokens through the projection trained for the global token, attention pooling,
round 6's spans and budget. Intermediate layers, a fine-tuned encoder or a different local readout are not tested.

## 1. Start conditions

All held in every seed before the results were read.
- P0 reproduced round-6 V0's validation predictions exactly, sequence by sequence, on both halves in all three seeds.
- Token 0 of the all-token path equalled the existing global embedding within 9.5e-7, against a limit of 1e-5.
- Encoder parity matched round 6: 2.9e-6 against the stored reference embeddings and 4.9e-4 against the E1 cache.
- Trainable parameters were P0 65,922, P1 329,091 and P2 328,836, as written in the config. Seven unit tests passed.
- A sanity run (256 spans, one epoch) per seed ran end to end and is not reported.
- The training spans per seed (6,120 / 4,681 / 5,251) were round 6's. Each seed took 30 to 32 minutes.

An independent code review before the results confirmed that P1 and P2 are treated alike. It noted that the strongest
previous configuration is the fixed reference of rounds 3 to 6. The report therefore also checks the highest previous
calibration mean (0.9739, the geometry round's G1 ASRF). This was committed while the seeds ran, before any result was
read (b1c337d).

## 2. Span classification

Validation spans with an A4C or A5C majority, per seed (s0 / s1 / s2). The vote is ASRF's own class.

| Half | Rule | Accuracy | A5C recall | Separation (AUC) | A4C read as A5C, spans (clips) | A5C read as A4C, spans (clips) |
|---|---|---|---|---|---|---|
| Selection | vote | 0.896 / 0.890 / 0.891 | 0.934 / 0.731 / 0.777 | – | 45 (17) / 5 (2) / 11 (6) | 13 (4) / 46 (5) / 45 (5) |
| Selection | P0 | 0.858 / 0.875 / 0.875 | 0.792 / 0.772 / 0.802 | 0.932 / 0.944 / 0.936 | 38 (10) / 19 (8) / 24 (7) | 41 (4) / 39 (4) / 40 (4) |
| Selection | P1 | 0.863 / 0.862 / 0.866 | 0.802 / 0.743 / 0.772 | 0.939 / 0.949 / 0.942 | 37 (12) / 20 (9) / 23 (7) | 39 (4) / 44 (4) / 46 (4) |
| Selection | P2 | 0.853 / 0.864 / 0.862 | 0.782 / 0.743 / 0.767 | 0.936 / 0.942 / 0.936 | 39 (11) / 19 (8) / 24 (6) | 43 (4) / 44 (4) / 47 (4) |
| Calibration | vote | 0.877 / 0.923 / 0.899 | 0.922 / 0.863 / 0.854 | – | 49 (20) / 10 (6) / 19 (10) | 16 (3) / 22 (4) / 26 (4) |
| Calibration | P0 | 0.843 / 0.921 / 0.897 | 0.780 / 0.887 / 0.865 | 0.904 / 0.945 / 0.940 | 38 (14) / 15 (7) / 22 (10) | 45 (8) / 18 (4) / 24 (4) |
| Calibration | P1 | 0.849 / 0.923 / 0.899 | 0.810 / 0.894 / 0.871 | 0.938 / 0.957 / 0.953 | 41 (15) / 15 (7) / 22 (10) | 39 (8) / 17 (4) / 23 (4) |
| Calibration | P2 | 0.858 / 0.921 / 0.899 | 0.815 / 0.894 / 0.865 | 0.933 / 0.951 / 0.948 | 37 (13) / 16 (7) / 21 (9) | 38 (8) / 17 (4) / 24 (4) |

- On the selection half, every readout makes 39 to 47 "A5C read as A4C" errors per seed. In every arm and seed they
  come from the same 4 source clips, and ASRF's vote errs on the same 4 (plus one more in seeds 1 and 2). The errors in this direction are concentrated in a few recordings, as
  round 4 found, and no readout moves them.
- No readout beats ASRF's own vote in accuracy on the selection half. On the calibration half they roughly match it in
  two seeds and fall below it in seed 0.

## 3. Full system

Segment F1@50 per seed (three-seed mean). The strongest previous configuration is 0.9344 on selection and 0.9721 on
calibration; the final goal needs 1 point more.

| System | Selection | Calibration |
|---|---|---|
| ASRF default | 0.910 / 0.909 / 0.907 (0.909) | 0.976 / 0.964 / 0.964 (0.968) |
| P0 | 0.869 / 0.886 / 0.883 (0.879) | 0.924 / 0.960 / 0.953 (0.946) |
| P1 | 0.870 / 0.880 / 0.879 (0.876) | 0.929 / 0.962 / 0.960 (0.950) |
| P2 | 0.865 / 0.882 / 0.881 (0.876) | 0.934 / 0.962 / 0.958 (0.952) |

- Boundary recall is at least 0.958 for every arm on both halves.
- Native macro-F1 misses the selection floor (0.973) in some seed for every arm (minimum 0.951 to 0.959). P2 also
  misses the calibration floor (0.980) in two seeds (0.976 in seeds 1 and 2).
- Paired contrasts of segment F1@50 per seed:
  - P2 − P1 is −0.5 / +0.2 / +0.3 points on selection and +0.6 / 0.0 / −0.2 on calibration, with no consistent
    difference.
  - P1 − P0 and P2 − P0 show no consistent difference either.

## 4. Reading under the pre-declared terms

| Term (config `terms`) | Result |
|---|---|
| Classification gain of P2 over P1 | no |
| System gain of P2 over P1 | no |

The first rule applies: the hypothesis ends. No larger readout, intermediate layer or fine-tuning follows from this
result automatically. The final goal is unchanged and not met. Beating B3, ASRF and the strongest previous
configuration with the thresholds met remains open.

## 5. Files

- **Config and plan.** `configs/experiments/local_readout_v1.yaml` and `docs/superpowers/plans/2026-10-04-local-readout.md`.
- **Code.**
  - `echo_routing/temporal/local_readout.py` holds the readouts and the token path, with tests in
    `tests/test_local_readout.py`.
  - `scripts/round7_local_readout.py` runs one seed and calls round 6's functions.
  - `scripts/round7_report.py` writes the report.
- **Numbers.** `report.json` and `report_tables.md` hold every table, the per-seed intervals and the flips with their
  clips and sessions.
- **Server only.** The runs are in `runs/2026100?-*-md7_local_readout_s{0,1,2}-*` and the head weights in
  `checkpoints/`.
