# Geometry pretraining round (2026-10-04): does tilt supervision help real 2D tasks?

This round asks one question. Beyond same-plane temporal consistency, does adding regression of the tilt difference
improve real 2D tasks? Three arms share one encoder start, E1 (the EchoViewCLIP stage-1 image encoder).
- **G0** is E1 as it is.
- **G1** trains E1's last ViT block on slices cut from 3D echo volumes. The objective is InfoNCE, a contrastive loss
  that pulls together the same plane at two times and pushes apart the other slices in the batch.
- **G2** uses the same InfoNCE term and adds regression of the tilt difference between the two slices of a pair.

G2 − G1 therefore tests the extra tilt supervision. G1 already uses one geometric relation, so G1 − G0 is not only an
effect of more data. Both arms read the same slice pairs in the same order, with the same budget (3,000 steps of 32
pairs), and differ only in the tilt term's weight (0 or 1). Three seeds were run. Success is judged only on real 2D
EV9V clips. Classification comes first, then segmentation with ASRF and B3 against the strongest previous
configuration. The tilt error on held-out 3D volumes is mechanism evidence only.

The rules, thresholds and terms were fixed in `configs/experiments/geometry_pretraining_v1.yaml` before any G1 or G2
result existed. The start review is `start_review.md`. Both validation halves are development data. The test split was
not read, and the paper stays paused.

## Answer

Under the pre-declared rules, **G2 does not pass the screening, neither G1 nor G2 is better than G0, and no system
reaches the final goal.** The config's rule for this case applies: this direction stops at this data scale.

**Post-round resource update, 2026-10-04.** The CETUS organiser's reply, shared by the project lead, grants the requested
research use but confirms only ED/ES 3D volumes and annotations are available. CETUS therefore cannot be described as
simply adding more full dynamic sequences: it would also change temporal sampling and the data distribution. The
stopping decision and this round's success criteria remain unchanged. No additional training is triggered by the
permission. Details are in `docs/data_requests/2026-10-03_cetus_request.md`.

1. **The tilt term changes nothing measurable in the 2D tasks.** G2 − G1 shows no consistent difference in any
   classification or segmentation measure on either half. For segment F1@50, the per-seed differences are −0.4 to +0.3
   points for ASRF and −0.4 to +0.3 points for B3.
2. **Neither arm is better than G0.** No measure improves consistently against G0 on either half, for G1 or G2.
3. **No system reaches the final goal.** The goal is a three-seed mean segment F1@50 at least 1 point above the
   strongest previous configuration on each half, with the thresholds met in every seed and half.
   - On the selection half, every G1 and G2 system stays 1.9 to 2.2 points below the strongest previous configuration
     (0.9344) on the three-seed mean.
   - On the calibration half, G1's ASRF comes closest (0.9739 against 0.9721, +0.2 points). It needed +1 point, and on
     the selection half it misses the native floor and boundary recall.
4. **"No consistent difference" is not "no loss".** Three signals point toward a loss. None meets the consistency rule
   except the third, which holds for G1 only.
   - **Probe decisions.** All six G1 and G2 probes shift the A4C/A5C decision toward A5C. On the calibration half, the
     same two native A4C clips turn into A5C in every encoder. Native macro-F1 there falls from 0.985 (G0) to 0.969
     to 0.979.
   - **B3 against the floors.** On G1 and G2 features, B3 falls below both native floors in every seed (0.970 on
     selection against 0.973; 0.969 to 0.978 on calibration against 0.980). On G0's features it meets both.
   - **False changes.** B3's internal cuts per native clip on the calibration half rise from 0.068 (G0) to 0.114 to
     0.118 with G1 features. The rise is consistent across the three seeds.
5. **G2 predicts tilt on held-out 3D slices, without a detected downstream gain.** On held-out people, G2's tilt head errs by 4.65° to
   4.71°. Always predicting zero errs by 6.59°, and the contrast-only baseline on the model's input errs by 5.85°. No
   G2 seed carries the shortcut warning (its threshold is 5.32°).

**What this does not show.** This round does not show that 3D geometry cannot help view classification. It covers one
form of supervision (tilt difference with InfoNCE), one trainable block, one budget and 42 volumes from 19 people. It
shows that this form, at this scale, adds nothing measurable to E1 on EV9V.

**Route closed, 2026-10-04.** The project lead closed this G1/G2 route after the results. The results, the six
encoders and the stop decision are kept, and no further training is run for it. The encoders are frozen read-only
in `models_frozen/geometry_pretraining/` on the server, with digests in `frozen_weights.json`. The next training
round waits for a new testable mechanism, distinct from this one.

## 1. What ran

- **Data.** 29 EchoSlicer volumes from 4 people and 15 STACOM 2011 volunteers. Two thin EchoSlicer volumes fail the
  coverage rule, which leaves 42 volumes from 19 people. Held out for mechanism checks are EchoSlicer person C and
  STACOM v14, v15 and v16. Pretraining therefore uses 15 people, sampled person first and then volume.
- **Pairs.** Each pair is two slices through the probe at one rotation about the beam, with tilts within ±15°. A third
  of the pairs show the same plane at two times, a third show two tilts at one time, and a third change both. Both
  slices share one fixed mask and are standardised to EV9V's sector statistics. There was no augmentation.
- **Training.** The last ViT block, the final normalisation and the projection were trained with AdamW for 3,000
  steps, and the final weights were kept. Each run took 8 to 9 minutes and peaked at 1.5 GB of GPU memory. In every
  seed, G1 and G2 read bit-identical batches. All six encoders record the same config digest (commit 3a079c1).
- **Evaluation.** Embeddings were extracted in the main environment, and the E1-recipe probe was trained per encoder.
  ASRF was then trained with the seed number of the encoder, and B0 to B3 were calibrated. All of this used the
  round-2 settings unchanged. G0 is the round-2 E1 run set, which the new extractor reproduces exactly.
- **Statistics.** Differences are paired per seed with 95% bootstrap intervals, resampling native clips by session
  and streams by stream. "Consistent" means all three seeds' intervals exclude zero in the same direction.

## 2. Classification (stage A, the probe)

Ranges are over the three seeds; G0 has one probe.

| Arm | Half | A4C/A5C AUC, samples | A4C/A5C AUC, clip mean | A4C recall | A5C recall | Native macro-F1 | Cuts per clip |
|---|---|---|---|---|---|---|---|
| G0 | selection | 0.915 | 1.000 | 1.000 | 0.909 | 0.978 | 2.93 |
| G1 | selection | 0.915–0.921 | 1.000 | 0.989 | 1.000 | 0.979 | 2.86–2.93 |
| G2 | selection | 0.913–0.920 | 1.000 | 0.989 | 1.000 | 0.979 | 2.87–2.92 |
| G0 | calibration | 0.882 | 0.998 | 1.000 | 0.889 | 0.985 | 3.29 |
| G1 | calibration | 0.881–0.883 | 0.998 | 0.967 | 0.889 | 0.969–0.971 | 3.29–3.38 |
| G2 | calibration | 0.881–0.882 | 0.998 | 0.967 | 0.889–0.944 | 0.969–0.979 | 3.28–3.35 |

The AUC is the probability that a sample of an A5C clip gets a larger A5C − A4C probe margin than a sample of an A4C
clip. The rules use it over samples, because over clip means it is at its ceiling for G0 (start review). Neither AUC
moves. Recall and macro-F1 use each clip's majority vote. The selection half has 87 native A4C and 11 native A5C clips,
and the calibration half has 61 and 18.

- G2 − G1 differs by at most 0.003 in sample AUC. Two of the six seed-half intervals lie below zero and none above, so
  there is no consistent difference.
- In all six encoders, one A5C clip that G0 missed on the selection half becomes right, and one A4C clip becomes A5C.
  On the calibration half, two A4C clips become A5C in all six encoders, and one A5C clip becomes right in G2 seed 1
  only. One PSAX clip turns PLAX in seed 0 of both arms. The AUC changes little while more clips are assigned A5C.
  The shared pattern in G1 and G2 is consistent with their shared pretraining/probe-refitting pipeline; it does not
  isolate InfoNCE from probe fitting or establish that every ranking is unchanged. No consistent incremental effect
  of the tilt term is detected.

## 3. Segmentation (stage B)

Three-seed means of segment F1@50, with ranges of boundary recall (within 0.5 s) and native macro-F1.

| Arm | System | Selection F1@50 | Calibration F1@50 | Recall, sel. / cal. | Native F1, sel. / cal. |
|---|---|---|---|---|---|
| G0 | ASRF | 0.9087 | 0.9682 | 0.942–0.953 / 0.972–0.977 | 0.948–0.966 / 0.975–0.993 |
| G1 | ASRF | 0.9130 | 0.9739 | 0.932–0.948 / 0.972–0.982 | 0.918–0.954 / 0.986 |
| G2 | ASRF | 0.9155 | 0.9726 | 0.937–0.953 / 0.975–0.980 | 0.954–0.961 / 0.9796–0.986 |
| G0 | B3 | 0.8940 | 0.9589 | 0.955 / 0.977 | 0.978 / 0.986 |
| G1 | B3 | 0.9128 | 0.9672 | 0.955–0.958 / 0.972–0.977 | 0.970 / 0.969–0.978 |
| G2 | B3 | 0.9146 | 0.9666 | 0.955–0.961 / 0.975–0.980 | 0.970 / 0.969–0.975 |
| Strongest previous | | 0.9344 | 0.9721 | | |

- **G2 − G1.** No consistent difference for ASRF or B3, in any measure, on either half.
- **Against G0.** No consistent difference in segment F1@50.
  - B3's gain on the selection half (+1.5 to +2.9 points) excludes zero in two of three seeds, for G1 and for G2 alike.
    It comes with native macro-F1 below the floor, and B3 tunes on the calibration half, so it is not a clean gain.
  - ASRF ranges from −0.6 to +1.4 points.
- **Against B3, ASRF and the strongest previous configuration.**
  - On the selection half, every G1 and G2 system is 1.1 to 3.1 points below the strongest previous configuration.
  - On the calibration half, ASRF is within ±0.4 points of it and B3 is 0.3 to 1.0 points below.
  - None of these differences is consistent.
- **The only consistent results** are more false changes for B3 on G1 features, measured as internal cuts per native
  clip. They hold against G0's B3 and ASRF on the calibration half and against the strongest previous configuration
  on both halves.

## 4. Thresholds and the final goal

| System | Selection: thresholds met / goal | Calibration: thresholds met / goal | What fails |
|---|---|---|---|
| G1 ASRF | no / no | yes / no | selection recall 0.932 and native 0.918; calibration +0.2 points, not +1 |
| G2 ASRF | no / no | no / no | selection recall 0.937 and native 0.954; calibration native 0.9796 < 0.980 |
| G1 B3 | no / no | no / no | native below both floors; F1@50 below the strongest on both halves |
| G2 B3 | no / no | no / no | native below both floors; F1@50 below the strongest on both halves |

For reference, G0's B3 meets both thresholds on both halves. G0's ASRF misses the native floor on both halves and
boundary recall on the selection half. If
boundary recall were checked on the calibration half only, no verdict would change (see the config's amendments).

## 5. Screening rules

| Rule (config, `interpretation`) | Result |
|---|---|
| 1. G2 shows no consistent loss against G0 or G1 in separation or A5C recall | holds |
| 2. G2 − G1 separation is consistently positive on one half | fails |
| 3. G2 − G1 shows no consistent increase in false changes | holds |
| 4. Stage B: G2 not below G1, and consistently above on one half | fails |
| Better than G0 (term): G1 / G2 | no / no |
| G2 adds nothing over G1 (term) | yes |

The first two readings of the config apply. G2 does not pass the screening. Neither G1 nor G2 is better than G0, so
this direction stops at this data scale. Rule 1 holding means only that no loss was detected by the rule; section 2
shows where the probe did lose clips.

## 6. Mechanism evidence (held-out people)

| Run | Tilt error | On moving planes | Cosine distance: same plane, two times / two tilts, one time |
|---|---|---|---|
| G1 (3 seeds) | 6.74°–6.82° | 9.75°–10.01° | 0.185–0.188 / 0.285–0.290 |
| G2 (3 seeds) | 4.65°–4.71° | 5.88°–5.97° | 0.201–0.206 / 0.313–0.320 |
| Zero predictor | 6.59° | 9.95° | |
| Contrast baseline, final input | 5.85° | 8.26° | |

G1's tilt head is never trained, so its error only shows the untrained head. G2's reported error is lower than the
contrast-only baseline. This does not establish that its encoder acquired additional geometric information relative
to G0 or G1: that would require equivalently trained readouts on the frozen encoders, which were not run. Not triggering
the specific contrast-baseline warning also does not rule out other shortcuts. These limitations do not change the
negative downstream result or require another experiment to close this round.

G2 also enlarges the distances in the encoder's output, but between times as much as between
tilts. The ratio of the two is 1.54 to 1.55 for G1 and 1.55 to 1.56 for G2. This does not show the hoped-for
separation of heart motion from plane change. The A4C/A5C AUC also changes little (section 2).

The contrast baseline is described in `start_review.md`. On the model's input it explains more of the tilt than on raw
slices (R² 0.21 against 0.07), mainly through the slice mean.

## 7. Limitations

- Pretraining data are small: 15 people, one gated cycle per STACOM volunteer, healthy volunteers, and two scanner
  models.
- Only one block of the encoder trains, at one learning rate and budget. A larger change might behave differently,
  but this round gives no sign that it would help.
- The 3D slices differ from EV9V's 2D clips in resolution and texture, and no step adapts them.
- Each half has only 11 or 18 native A5C clips, so A5C recall moves in steps of 0.06 to 0.09, and intervals are wide.
- Seeds are not independent test sets, and the bootstrap does not model clip reuse across streams, so intervals are
  optimistic.

## 8. Files

- **Config and plan.** `configs/experiments/geometry_pretraining_v1.yaml` (with the amendments) and
  `docs/superpowers/plans/2026-10-03-geometry-pretraining.md`.
- **Code.**
  - `scripts/geometry_bank.py` and `echo_routing/geometry/` render the slice pairs.
  - `scripts/geometry_pretrain.py` trains.
  - `scripts/extract_encoder_features.py` extracts the embeddings.
  - `scripts/geometry_round_run.sh` runs the round, and `scripts/geometry_round_report.py` writes the report.
- **Numbers.** `report.json` and `report_tables.md` hold every table and per-seed interval, and
  `contrast_baseline.json` holds the baseline.
- **Weights.** `frozen_weights.json` lists the frozen files and their digests.
- **Server only.** The banks are in `data/geometry/banks/`, the encoders in `runs/geometry/{G1,G2}_s{k}/`, the
  embeddings in `cache_v2/geo_raw/` and the logs in `logs/geometry_*.log`.
