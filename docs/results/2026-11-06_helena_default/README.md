# EX5: Helena's preprocessing as the default input

Date: 2026-11-06. Development halves only. No training. The EV9V test split was not opened.

On 2026-11-06 the project owner made Helena's preprocessing the default input for every new experiment (see `AGENTS.md`). This round measures where that leaves the reference model. The model is EX3 arm H: A08's recipe trained on Helena's input, seeds 0–2, frozen in `models_frozen/ex3_helena_input_v1`. The recipe was not retuned for the new input; that was the owner's budget choice.

## What was run

- **Frame cache.** EX3's cache step was rerun unchanged. It covers 3,931 recordings and 217,094 frames (32.7 GB). Both of EX3's checks hold: H seed 0's training identity and its first-epoch pixel digest reproduce.
- **R1, H's own decoder setting.** For each seed, the B3 setting with the highest multi-stream segment F1@50 was taken from the same 40-setting grid on calibration. Floors were reported, not required. The setting was then applied unchanged to selection. Each half is paired against A08 at its own stored setting. The calibration row is optimistic, because the setting was chosen there. Selection is the honest row.
- **R2, real exam order.** EX1's real-order streams were decoded at the R1 setting and compared with A08 at its own setting. The A08 path reproduces EX1 exactly in every seed and half (largest difference 0).
- **R3, remaining errors.** On the multi-fragment streams of each half (120 on selection), every reference segment the system missed was classed by cause and by the class it was confused with. Every extra predicted segment was classed by its class and the segment it sits in.

Not rerun:
- EX3 already has external transfer and open-set scores on H.
- EX4-H already has the saliency maps.
- Closed A-rounds.
- A14's router. It is closed, its gain was +0.0026, and porting it needs an A13 head retrained on H.

## Results

**Table 1. Multi-stream segment F1@50, H at its own setting against A08 (paired difference, 95% CI).**

| Half | Seed | H | A08 | H − A08 | Extra false segments | Native macro-F1 floor met |
|---|---|---|---|---|---|---|
| Calibration | 0 | 0.9674 | 0.9827 | −0.0153 [−0.0247, −0.0069] | +9 | no (0.9788) |
| Calibration | 1 | 0.9752 | 0.9885 | −0.0133 [−0.0242, −0.0038] | +10 | yes |
| Calibration | 2 | 0.9770 | 0.9837 | −0.0067 [−0.0134, −0.0010] | +3 | no (0.9712) |
| Selection | 0 | 0.9458 | 0.9563 | −0.0105 [−0.0225, 0.0024] | +9 | no (0.9596) |
| Selection | 1 | 0.9105 | 0.9545 | −0.0440 [−0.0704, −0.0189] | +44 | yes |
| Selection | 2 | 0.9044 | 0.9564 | −0.0520 [−0.0783, −0.0278] | +46 | yes |

The three-seed means are 0.9732 for H against 0.9850 for A08 on calibration, and 0.9202 against 0.9557 on selection. Only seed 1 has a setting that meets every floor. H misreads 2 or 3 single recordings per half by window mean, where A08 misreads 1.

**Table 2. Real exam order, pooled segment F1 (EX1's streams).**

| Half | H, seeds 0/1/2 | A08, seeds 0/1/2 | H − A08 |
|---|---|---|---|
| Calibration | 0.9375 / 0.9288 / 0.9189 | 0.9462 / 0.9416 / 0.9141 | −0.0087 / −0.0128 / +0.0049, all CIs cross 0 |
| Selection | 0.9236 / 0.9032 / 0.9091 | 0.9492 / 0.9400 / 0.9495 | −0.0256 / −0.0368 / −0.0404; only seed 2's CI excludes 0 |

Frame accuracy differs between the halves. On calibration, H gets more frames right than A08 (+97, +123 and +186 frames). On selection, it gets fewer right (−67, −87 and −60 frames).

**Table 3. Remaining errors on the selection half's multi-fragment streams (502 reference segments).**

| Seed | Missed segments, H / A08 | Share A4C↔A5C, H / A08 | Extra segments, H / A08 | Extra segments that split one view into pieces, H / A08 |
|---|---|---|---|---|
| 0 | 22 / 20 | 95% / 95% | 33 / 24 | 11 / 10 |
| 1 | 24 / 20 | 96% / 100% | 70 / 26 | 32 / 9 |
| 2 | 29 / 19 | 86% / 89% | 71 / 25 | 31 / 6 |

On calibration, H misses 14, 7 and 9 of 518 segments, against A08's 7, 3 and 5. Most errors in both are A4C↔A5C there too.

## Reading

- With A08's recipe and only its decoder setting retuned, the model on Helena's input stays below A08 on the honest half. The gap is −0.011 to −0.052 segment F1@50 on constructed streams and −0.026 to −0.040 on real exam order.
- Most of the extra loss in seeds 1 and 2 comes from extra cuts. Their models split one view into pieces and add short A4C/A5C fragments.
- Retuning the decoder helps seed 0 only. Against EX3's reported settings, the selection differences moved from −0.0476 to −0.0105 for seed 0, stayed at −0.0440 for seed 1 (the same setting), and moved from −0.0477 to −0.0520 for seed 2.
- The remaining errors are still almost all A4C↔A5C, as with A08. The new input removes screen content outside the fan (EX4-H), but it does not remove the main error.
- The switch is the owner's decision for input cleanliness. These numbers are what it costs under the current recipe. Closing the gap would most likely need a training recipe tuned for the new input. That needs new budget, and data that was not used for tuning to judge it.

## Limits

- Development halves only, and A08 was itself selected on them. This is not a confirmatory comparison.
- The recipe, the 40-setting B3 grid and the floors were all built on A08's input (EX3 bias audit).
- Helena's geometry falls back on about 23% of EV9V recordings, so screen content can remain there.
- Three seeds. The seed-to-seed spread is as large as the gap in seed 0.

## Files

- **Server run folder:** `runs/ex5_helena_default_v1/`.
  - `code/` holds `ex5_h_cache.py`, `ex5_reference.py` and `SHA256SUMS`.
  - `public/h_cache_rebuild.json` and `public/reference.json` hold aggregates only, behind an identifier guard.
  - `reference_attempt1_stopped.log` is the first attempt. It was stopped after its first row, when a review found two attribution problems in the R3 counter, and that counter was fixed before the full run.
- **GPU:** about 0.15 h for the run, plus about 0.06 h for the stopped attempt.
