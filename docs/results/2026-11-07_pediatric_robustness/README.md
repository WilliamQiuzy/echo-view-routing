# EX7: robustness of H against A08 on EchoNet-Pediatric

Date: 2026-11-07. Inference only, no training. GPU use was 0.39 h against a cap of 3 h. The EV9V test split was not opened.

## Question

The owner asked whether the model trained on Helena's input (EX3 arm H) is more robust than A08 on other datasets' test sets. EX3 had already compared them on LVH, a 400-recording sample of Dynamic, and CAMUS. This round adds a new population, children, and a view not yet tested externally, PSAX.

The plan (`docs/superpowers/plans/2026-11-07-ex7-pediatric-robustness.md`, commit b6d10fb) was fixed before any EX7 output.

## What was run

- **Sample.** The test folds (2–9) of the frozen `pediatric-external-v1` manifest were used.
  - For A4C and for PSAX separately, 400 patients were drawn without replacement, and one recording was drawn per patient. The seed was 2026110701.
  - 105 patients appear in both views.
  - All 800 recordings decode, and every frame count matches the manifest.
- **Systems.** A08 seeds 0–2 on A08's own centre crop, and H seeds 0–2 on Helena's processing. Both use EX3's external path.
  - H's geometry comes from Helena's code (commit 53a2246) with EX3's configuration and FastSAM weights.
  - Both systems sample at 10 Hz, use A08's window plan and average the window probabilities. The class with the highest mean probability is the reading for the recording.
- **Checks before any output.**
  - On EX3's first 8 Dynamic recordings, this path reproduces the stored A08 and H outputs exactly, with a largest difference of 0.0.
  - Helena's geometry reproduces EX3's stored crops in 8 of 8 recordings, with mask IoU 1.0.
  - The first check attempt stopped because the script compared the manifest's file hash with the protocol's row digest. That was a field error in my script and was fixed. An independent code review then asked for incremental GPU accounting, the geometry run's torch settings, and identity checks before the readout. The final check ran after those fixes.
- **Geometry.** No recording failed. On 734 of 800 (92%), Helena's analysis fell back on the fan geometry alone. EchoNet videos are released already cropped to 112×112 and masked outside the fan, so Helena's processing changes little here.

## Results

**Table 1. Recall by view, A08 against H (same seed), with the primary endpoint.** Macro recall is the mean of the two view recalls. The interval is a paired bootstrap that resamples whole patients.

| Seed | A4C recall, A08 / H | PSAX recall, A08 / H | Macro recall, A08 / H | H − A08, macro (95% CI) |
|---|---|---|---|---|
| 0 | .965 / .973 | .995 / 1.000 | .980 / .986 | +.006 (−.001, +.014) |
| 1 | .880 / .903 | .983 / .985 | .931 / .944 | +.012 (−.003, +.028) |
| 2 | .870 / .880 | .990 / .998 | .930 / .939 | +.009 (−.008, +.026) |

The plan's reading is **`no_consistent_difference`**. All three point estimates favour H, but every interval includes 0.
- **Per view.** The A4C differences are +.008, +.023 and +.010, and every CI includes 0. The PSAX differences are +.005, +.003 and +.008.
- **Exact McNemar tests on A4C.** Clips only H gets right, against clips only A08 gets right, are 5:2, 22:13 and 22:18 (p .45, .18, .64).

**Table 2. Where the 400 A4C recordings go (PLAX / PSAX / A4C / A5C / SC4C).**

| Seed | A08 | H |
|---|---|---|
| 0 | 1 / 4 / 386 / 1 / 8 | 0 / 11 / 389 / 0 / 0 |
| 1 | 16 / 4 / 352 / 4 / 24 | 6 / 16 / 361 / 4 / 13 |
| 2 | 21 / 9 / 348 / 1 / 21 | 13 / 33 / 352 / 0 / 2 |

H reads fewer A4C recordings as PLAX or SC4C, but more of them as PSAX. In seed 2, it reads 33 of 400 A4C recordings as PSAX, against A08's 9. The total number of A4C errors barely changes.

**Table 3. AUROC.** The A4C row asks how well the A4C score separates A4C recordings from PSAX recordings. The PSAX row asks the same of the PSAX score.

| Score | Seed 0, H − A08 | Seed 1 | Seed 2 |
|---|---|---|---|
| A4C | −.0003 (−.0024, +.0015) | +.0041 (+.0009, +.0087) | +.0057 (+.0018, +.0105) |
| PSAX | −.0007 (−.0018, +.0001) | −.0016 (−.0030, −.0006) | −.0018 (−.0031, −.0008) |

Every AUROC is above .99 for both systems. H separates slightly better on the A4C score and slightly worse on the PSAX score. The differences are in the third decimal.

**Seed spread.** A4C recall ranges over .095 between A08's seeds (.870 to .965) and over .093 between H's seeds (.880 to .973). That is several times the H − A08 difference. Of the 9 cross-seed comparisons on macro recall, 2 have an interval above 0, 2 below 0, and 5 include 0.

**Table 4. All external results so far, H − A08 by seed (0 / 1 / 2).**

| Dataset (round) | Endpoint | H − A08 | Note |
|---|---|---|---|
| EchoNet-LVH, 343 (EX3) | PLAX recall | +.038 / +.018 / +.015 | All CIs above 0, but only 5 to 13 recordings net per seed. Geometry fallback 6% |
| EchoNet-Dynamic, 400 (EX3) | A4C recall | −.015 / −.013 / −.008 | All CIs include 0. Fallback 80% |
| CAMUS 4CH, 500 patients (EX3) | A4C recall | +.042 / −.004 / +.160 | 2CH read as A4C also rises (+.172 / −.026 / +.302). Separating 4CH from 2CH does not improve (AUROC −.020 / −.037 / +.025). Fallback about 95% |
| EchoNet-Pediatric, 800 (EX7) | Macro recall, A4C and PSAX | +.006 / +.012 / +.009 | All CIs include 0. Fallback 92% |
| EV9V selection (EX5) | Multi-stream segment F1@50 | −.011 / −.044 / −.052 | The development task; H is lower |

## Reading

- **On Pediatric, H is not worse, and its point estimates are slightly higher in every seed.** The difference is about one percentage point of recall. It is smaller than the spread between seeds of either system, and no seed's interval excludes 0.
- **H shifts some errors rather than removing them.** It reads fewer A4C clips as PLAX or SC4C, and more as PSAX.
- **Across the four external datasets, no consistent robustness gain is visible.**
  - LVH gives a small, consistent PLAX gain.
  - Pediatric leans positive without reaching significance.
  - Dynamic leans negative without reaching significance.
  - CAMUS rises mainly because H reads more apical views as A4C.
- **Helena's processing has little to act on in most of these datasets.**
  - Dynamic and Pediatric were released cropped to 112×112 and masked outside the fan, and CAMUS has no screen content.
  - Helena's analysis fell back on fan geometry alone in 80% of Dynamic, 92% of Pediatric and about 95% of CAMUS recordings.
  - LVH is the exception, with 6% fallback. It is the one external dataset where her processing really changed the input, and also the one with a consistent gain. That gain is small (5 to 13 recordings per seed) and comes from one dataset already seen in earlier rounds, so it is a lead, not a finding.
  - The intended benefit of Helena's input is removing screen content from raw frames (EX4-H). Only raw external video with screen content would test it.
- **H's main cost stays where it was.** On EV9V, H is still 0.01 to 0.05 below A08 in segment F1@50.

## Limits

- **Pediatric is external to both models but has been used before.** It served the B0 external test (2026-10-01) and E1's head-only adaptation (2026-10-05). Neither A08 nor H was trained or selected on it.
- **The sample is 800 of 6,354 test recordings,** with one recording per patient and view.
- **Only 2 of the 5 classes are present.** PLAX, A5C and SC4C readings count as errors.
- **The plan's AUROC condition was read as "two or more seeds".** It did not affect the reading, because no macro interval lies wholly above 0.
- **Three seeds.**

## Files

- **Server run folder:** `runs/ex7_pediatric_robustness_v1/`.
  - `plan.md` (sha256 starts `4beac898`).
  - `code/ex7_pediatric.py` (`3009eccd`) and `SHA256SUMS`.
  - `public/check.json`, `public/geometry.json` and `public/result.json` hold aggregates only, behind a guard against EV9V and sample identifiers.
  - `private/rows.json`, `geometry/` and `forward/` hold per-recording data and stay on the server.
  - `ledger.jsonl` records GPU time in pieces.
  - `check_attempt1_wrong_digest_field.log` and `check_attempt2_before_review_fixes.log` are the two earlier check runs.
- **GPU:** 0.39 h in total. Geometry took about 16 minutes. The six forward passes and the checks took the rest.
