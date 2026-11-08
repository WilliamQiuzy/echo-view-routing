# EX6: retuning A08's recipe for Helena's input

Dates: 2026-11-06 to 2026-11-07. Development halves only. The EV9V test split was not opened. GPU use was 8.4 h against a cap of 15 h.

## Why

EX5 found that the model on Helena's input (EX3 arm H) stays below A08 on the selection half. Most of the loss is extra cuts. In seeds 1 and 2, H adds 70 and 71 false segments, and about half of them split one view into pieces. H was trained with A08's recipe, which had never been tuned for this input.

Two settings were retuned:
- **The neighbour penalty.** It is the term in A08's second stage that discourages cuts inside one view. Its coefficient, 0.15, was taken from MS-TCN and never tuned.
- **The encoder learning rate.** The training logs show that H fits the training set as well as A08 does. Final-epoch training accuracy in the second stage was .953, .956 and .948 for H, against .948, .958 and .946 for A08. H is not underfitting, so the rate was tried in both directions.

The plan (`docs/superpowers/plans/2026-11-06-ex6-helena-retune.md`) was committed before any EX6 output. It was amended once, also before any output, to test the rate in both directions.

## What was run

| Recipe | What it retrains | Penalty coefficient | Encoder learning rate |
|---|---|---|---|
| H0 (reference) | Nothing. EX3 arm H, its frozen weights and stored outputs. | 0.15 | 1e-5 |
| S | The second stage only, starting from H's first-stage end point | 0.45 | 1e-5 |
| Lup | Both stages | 0.15 | 3e-5 |
| Ldown | Both stages | 0.15 | 3e-6 |

Every other setting is A08's: 3,750 updates per stage, 8 windows per update, encoder frozen for the first 375 updates, and AdamW. Each recipe was trained with seeds 0, 1 and 2.

The rule was fixed in advance and used the calibration half only.
1. Each recipe got its own B3 setting. It was the setting in the 40-setting grid with the highest multi-stream segment F1@50 on calibration, as in EX5.
2. A recipe qualified if its three-seed mean was above H0's and at least two of its seeds were above the same seed of H0. The highest qualifying mean would be chosen. If none qualified, H0 would be kept.
3. A recipe combining S with the better learning rate would be trained only if S and that rate both had calibration means above H0.
4. Only after `choice.json` was written was the selection half opened. Every recipe was evaluated on it once, at its calibration setting.

Checks before training:
- With H0's settings, the trainer reproduces the first epoch of EX3 H seed 0 in both stages. The largest loss difference is 2.3e-7, and training accuracy and the pixel digest are identical.
- The new learning rates and coefficient were verified after the encoder was unfrozen.
- H0 is read from EX3's stored outputs and never recomputed. Its calibration settings equal EX5's. Its rows below reproduce EX5's Tables 1 and 3 exactly.

## Results

**Table 1. Calibration, multi-stream segment F1@50 (the half the choice was made on).**

| Recipe | Seeds 0 / 1 / 2 | Mean | Seeds above H0 | Qualifies |
|---|---|---|---|---|
| H0 | 0.9674 / 0.9752 / 0.9770 | 0.97318 | — | — |
| S | 0.9594 / 0.9662 / 0.9720 | 0.96589 | 0 | no |
| Lup | 0.9732 / 0.9742 / 0.9722 | 0.97320 | 1 | no |
| Ldown | 0.9634 / 0.9741 / 0.9663 | 0.96794 | 0 | no |
| A08 | 0.9827 / 0.9885 / 0.9837 | 0.9850 | | |

No recipe qualified, so H0 was kept. Lup's mean is 0.00002 above H0's, but only one of its seeds is. S's mean is below H0's, so no combined recipe was trained.

**Table 2. Selection, multi-stream segment F1@50 (paired difference, 95% CI).**

| Recipe | Seed | F1 | Against H0 | Against A08 |
|---|---|---|---|---|
| H0 | 0 | 0.9458 | | −0.0105 [−0.0225, +0.0024] |
| H0 | 1 | 0.9105 | | −0.0440 [−0.0704, −0.0189] |
| H0 | 2 | 0.9044 | | −0.0520 [−0.0783, −0.0278] |
| S | 0 | 0.9336 | −0.0122 [−0.0279, +0.0019] | −0.0228 [−0.0395, −0.0085] |
| S | 1 | 0.9188 | +0.0083 [−0.0199, +0.0369] | −0.0356 [−0.0596, −0.0138] |
| S | 2 | 0.9306 | +0.0262 [+0.0032, +0.0506] | −0.0259 [−0.0415, −0.0125] |
| Lup | 0 | 0.9225 | −0.0233 [−0.0486, +0.0021] | −0.0339 [−0.0604, −0.0067] |
| Lup | 1 | 0.9240 | +0.0135 [−0.0004, +0.0295] | −0.0305 [−0.0582, −0.0033] |
| Lup | 2 | 0.9318 | +0.0274 [+0.0073, +0.0480] | −0.0247 [−0.0474, −0.0032] |
| Ldown | 0 | 0.9231 | −0.0227 [−0.0426, −0.0051] | −0.0333 [−0.0534, −0.0146] |
| Ldown | 1 | 0.9196 | +0.0091 [−0.0176, +0.0357] | −0.0349 [−0.0538, −0.0160] |
| Ldown | 2 | 0.9249 | +0.0205 [−0.0093, +0.0484] | −0.0315 [−0.0538, −0.0117] |

The three-seed means are 0.9202 for H0, 0.9277 for S, 0.9261 for Lup, 0.9225 for Ldown and 0.9557 for A08.

**Table 3. Selection errors on the 120 multi-fragment streams (502 reference segments), seeds 0 / 1 / 2.** A missed segment is a reference segment the system did not match. An extra segment is a predicted segment that matches nothing. A split is an extra segment with the same view as the segment it sits in.

| Recipe | Missed | Missed as A4C↔A5C | Extra | Splits |
|---|---|---|---|---|
| H0 | 22 / 24 / 29 | 21 / 23 / 25 | 33 / 70 / 71 | 11 / 32 / 31 |
| S | 31 / 38 / 33 | 23 / 25 / 24 | 36 / 44 / 37 | 13 / 12 / 10 |
| Lup | 20 / 22 / 24 | 20 / 22 / 23 | 61 / 57 / 46 | 27 / 26 / 18 |
| Ldown | 40 / 39 / 40 | 29 / 31 / 33 | 37 / 42 / 35 | 8 / 7 / 8 |
| A08 | 20 / 20 / 19 | 19 / 20 / 17 | 24 / 26 / 25 | 10 / 9 / 6 |

**Table 4. Selection, real exam order, pooled segment F1 (EX1's streams, 25 exam clusters).**

| Recipe | Seeds 0 / 1 / 2 | Against A08 | Against H0 |
|---|---|---|---|
| H0 | 0.9236 / 0.9032 / 0.9091 | −0.0256 / −0.0368 / −0.0404; only seed 2's CI excludes 0 | |
| S | 0.9329 / 0.9298 / 0.9488 | −0.0163 / −0.0102 / −0.0007; all CIs cross 0 | +0.0093 / +0.0265 / +0.0397; all CIs cross 0 |
| Lup | 0.9186 / 0.9311 / 0.9495 | −0.0306 / −0.0089 / +0.0000; all CIs cross 0 | −0.0050 / +0.0279 / +0.0404; seeds 1 and 2 exclude 0 |
| Ldown | 0.9153 / 0.9122 / 0.9553 | −0.0339 / −0.0278 / +0.0058; seeds 0 and 1 exclude 0 | −0.0083 / +0.0089 / +0.0462; seed 2 excludes 0 |

A08 scores 0.9492, 0.9400 and 0.9495.

**Native floors.** These are the macro-F1 floors on single-view recordings: .980 on calibration and .973 on selection.
- Lup meets the selection floor in every seed but misses the calibration floor in every seed.
- Ldown does the opposite.
- S misses both floors in every seed.
- H0 meets the selection floor in seeds 1 and 2.

## Reading

- **By the plan's rule, the reading is `no_improvement`.** No recipe beat H0 on calibration, so H0 stays the reference on Helena's input. Nothing new was frozen.
- **No recipe closes the gap to A08.** On selection, every seed of every retuned recipe is below A08, and every confidence interval lies wholly below 0. The differences range from −0.023 to −0.036.
- **The two halves rank the recipes differently.** S was the worst recipe on calibration, with every seed below H0. On selection it has the highest mean. The selection differences against H0 follow the seed: every recipe loses on seed 0 and gains on seeds 1 and 2. H0's seed 0 was its good seed on selection, and its seeds 1 and 2 were the ones that over-cut. Much of the selection gain is therefore the new runs not repeating that over-cutting. The spread between seeds is as large as the spread between recipes. Choosing S because of its selection mean would mean choosing on the half that is meant to judge the choice.
- **The penalty does what it is meant to do, at a cost.** With coefficient 0.45, splits on selection fall from 11 / 32 / 31 to 13 / 12 / 10, but missed segments rise from 22 / 24 / 29 to 31 / 38 / 33. A lower learning rate shows the same trade with more misses (39 to 40 per seed). A higher learning rate keeps misses near A08's (20 to 24) but splits about as often as H0 (71 splits in total, against 74). No recipe has A08's balance of about 20 missed and 25 extra segments.
- **Real exam order points the same way.** S and Lup come closer to A08 than H0 does on seeds 1 and 2. All their CIs against A08 cross 0, but these runs were not chosen by the rule.
- **These two settings alone do not explain the gap.** Two things were not tested. On Helena's input the anatomy is about 71% of its size in A08's crop. The B3 grid and the floors were built on A08's input.

## Limits

- Development halves only, and A08 was selected on them. H0's selection results were already seen in EX5, before this plan was written. This is not a confirmatory comparison.
- Two settings, one new value each for the penalty and two for the learning rate.
- Three seeds. The seed-to-seed spread is as large as most of the differences above.
- The B3 grid, the floors and A08's other settings were all built on A08's input.

## Files

- **Server run folder:** `runs/ex6_helena_retune_v1/`.
  - `plan.md` is a copy of the committed plan (sha256 starts `9f0aefb1`).
  - `code/` holds `ex6_train.py` (sha256 starts `6bdfdb0b`), `ex6_evaluate.py` (`05c5f4e1`) and `SHA256SUMS`.
  - `public/` holds aggregates only, behind an identifier guard: `train_check.json`, `calibration_S_Lup_Ldown.json`, `choice.json` and `selection.json`.
  - `ledger.jsonl` records GPU time per job.
  - `checkpoints/` holds the end points of the 9 trained runs (2.0 GB). They are not frozen, because no recipe was chosen.
- **GPU:** 0.16 h for the checks, 7.08 h for the 9 training runs, 0.58 h for calibration and 0.54 h for selection. The total is 8.36 h.
