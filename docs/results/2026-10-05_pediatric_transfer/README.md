# Cross-population view recognition (2026-10-05): the current systems on EchoNet-Pediatric

The project's task stays view recognition and temporal segmentation on EV9V, with its unchanged success criterion.
This round asks whether the current strong systems fail on another population. EchoNet-Pediatric holds A4C and PSAX
recordings of children. If they fail broadly, the round runs one conventional adaptation comparison. Pediatric results
do not replace the EV9V goal and are not a new independent final validation. The paper stays paused.

An earlier round (2026-10-01) recorded a Pediatric PSAX clip recall of 0.183 for the old B0, an ImageNet ResNet-18
frame classifier. That number says nothing about the current systems, so step 1 measured them directly.

## Round answer

1. **Do the current strong systems fail on another population? Two of four do, broadly (step 1).** E0 + B3 misses
   PSAX and E1 + B3 misses A4C. E1 + ASRF and B7 pass the frozen transfer rule.
2. **Does conventional head adaptation fix it, or must the representation change? Head adaptation fixes it.**
   - D1 trains only E1's linear head on the Pediatric calibration folds, with EV9V replay. It lifts E1 + B3 on the
     test folds from A4C recall 0.740 to 0.985–0.986 and PSAX from 0.983 to 0.994–0.995. The two-class macro-F1
     gains 7.4 points in every seed, and the transfer rule holds in all three seeds.
   - D2 also fine-tunes the encoder's last block. It adds a small but consistent 0.4 points: A4C +0.6 to +0.7 and
     PSAX +0.2 to +0.3.
   - Under the pre-declared terms, "head suffices" is false only because of this increment, and "representation
     helps" is true. Both arms are ordinary fine-tuning, so the result is a conventional adaptation effect, not a
     contribution. No stable gap remains after adaptation.
3. **Do the gains reach segmentation without an EV9V decline? Through B3 they reach the labels, not boundary
   detection; through ASRF, not consistently. No EV9V decline was detected consistently across the three seeds, and
   the final goal is not met.**
   - On the stitched Pediatric exam streams, B3 with an adapted head finds 0.97 to 0.99 of the view changes with
     both views right, against 0.726 for D0. Its segment F1@50 rises from 0.848 to 0.981–0.989, with fewer cuts away
     from clip joins.
   - B3's boundary recall does not change: 0.976 for D0, 0.976–0.977 for D1 and 0.972–0.986 for D2, with no
     consistent difference. The gain lies in labelling the spans on both sides of a change and in fewer false cuts,
     not in finding more boundaries.
   - D1's ASRF is unchanged by construction, because its features equal E1's.
   - D2's ASRF, retrained on EV9V only, reads Pediatric clips better (A4C +2.2 to +12.8 points, PSAX +1.5 to +6.5,
     consistent). It finds their boundaries unstably, though: boundary recall 0.71 / 0.91 / 0.48 by seed.
   - That ASRF result describes how an ASRF trained on the source domain transfers after visual adaptation. Every
     system could have been given the same target-domain supervision budget. This round did not do that, so it says
     nothing about the ceiling of an ASRF adapted to Pediatric.
   - On EV9V, no decline against D0 was detected consistently across the three seeds, on either half, for either arm.
     That is not evidence of no loss, and thresholds are still missed in some seeds.
     - D2 + ASRF has native macro-F1 0.954 and 0.951 on the selection half in two seeds, below the 0.973 floor.
     - D2 + B3 has boundary recall 0.932 and 0.937 there in two seeds, below 0.95.
   - No arm meets the final goal. The closest, D2 with ASRF, is 3.2 points short on the selection half (0.913 against
     0.944) and 0.9 points short on the calibration half (0.973 against 0.982).

**What was trained.**
- Six adaptation runs: D1 and D2, three seeds each, 1,500 updates, about 3 to 4 minutes each.
- The ASRF of D2 for three seeds, trained on EV9V.
- B3 re-selected on EV9V's calibration half for each of the six runs.

**Not executed.**
- Adapting ASRF with Pediatric streams. An equal target-domain supervision budget for every system was possible but
  was not given.
- Adapting E0 or B7.
- Any test of real continuous scanning, or of A4C against A5C. Pediatric holds neither.

**Closing decisions (2026-10-05).**
- The Pediatric training extension is ended. Conventional adaptation worked, and no cross-population gap needing a
  new model was found.
- A bounded fault localisation, with no training, places ASRF's Pediatric boundary losses at the candidate stage
  (`boundary_fault_localisation.md`). On EV9V, ASRF's candidates cover 96–99% of the view changes, and B3 recovers
  most of the Pediatric losses. The candidate mechanism is therefore closed.
- Pediatric authorisation under item 6 of the Research Use Agreement is not confirmed (section 1, licence).

**Remaining gap.** The EV9V final goal is unchanged and unmet. Pediatric adaptation used labelled clips of the same
dataset as the test folds. It shows adaptation to a known target population, not generalisation to an unseen one.

## Step 1 answer

The rules were committed before any of these predictions (`configs/experiments/pediatric_transfer_v2.yaml`, commit
7090174). The frozen transfer rule needs a clip recall of at least 0.80 for both A4C and PSAX on the test folds.

1. **Do the current strong systems fail on Pediatric? Two of the four do, broadly.**
   - **E0 + B3 fails on PSAX** (recall 0.293), as the old B0 did. It reads 2,228 of 3,676 PSAX clips as PLAX. The
     misread clips come from 82% of the groups holding PSAX, and the ten largest of those groups hold only 4.5% of
     them. E0 + B3 is the strongest previous configuration on EV9V's selection half.
   - **E1 + B3 fails on A4C** (recall 0.740) but reads PSAX almost perfectly (0.983). It reads 552 of 2,678 A4C clips
     as A5C. The misread clips come from 37% of the groups holding A4C.
   - **E1 + ASRF passes** (A4C 0.932, PSAX 0.959, the class most seeds give). Its seeds differ widely: A4C 0.843 /
     0.943 / 0.939 and PSAX 0.978 / 0.931 / 0.956.
   - **B7 passes** (A4C 0.855, PSAX 0.936). B7 is the official EchoViewCLIP video model. It reads 183 PSAX clips as
     PMPALA, which counts as an error here because every output is kept.
   - The calibration folds show the same pattern.
2. **Decision.** The pre-declared broad-failure rule holds for E1 + B3. A class misses 0.80, and its misread clips
   come from at least 10% of the groups holding it. The rule does not hold for E1 + ASRF. The adaptation comparison
   starting from E1 is therefore pre-registered next.
3. **The E1 failure sits in the frame classifier, while the features carry the information.** E1's linear probe
   calls 469 test A4C clips A5C. ASRF reads the same E1 features and gets 399 of them right by the class most seeds
   give (197 / 409 / 426 by seed). Whether retraining only the classification head is enough is what the comparison's
   D1 arm tests.

## 1. Data, groups and history

- **Data.** The protocol is `pediatric-external-v1`, frozen on 2026-10-01 before any prediction and checked against
  its digests before use. It holds 7,810 videos: A4C 3,284 and PSAX 4,526, all 112 × 112.
  - Folds 0–1 form the calibration part (1,456 videos, 390 groups).
  - Folds 2–9 form the test part (6,354 videos, 1,549 groups).
  - Every video was scored, and none was removed for being predicted right or wrong.
  - The dataset page states 7,643 labelled videos (A4C 3,176, PSAX 4,424). The release on the server holds 167 more.
    This difference is not explained.
- **Groups and patients.** The official file lists carry no patient identifier, and the dataset page does not say how
  the folds were made. The groups are the first file-name token.
  - The groups never cross a fold.
  - Among the 953 groups with several exams, sex is constant in 951, and ages within a group span a median of 2 years
    (at most 7).
  - The groups are therefore consistent with patients but are not confirmed as patients by the provider.
- **History.** Before this round, Pediatric was read only by the 2026-10-01 round, which evaluated the old B0 and its
  decoders on both parts and reported the results. Nothing has been trained or tuned on Pediatric. The test folds
  have been viewed and cannot become an unseen test set by re-splitting.
- **Licence: not confirmed (updated 2026-10-05).** The Research Use Agreement on the dataset page states, among
  other items:
  - item 1: "Permission is granted to view and use the EchoNet-Pediatric Dataset without charge for personal,
    non-commercial research purposes only";
  - item 4: "YOU MAY NOT DISTRIBUTE, PUBLISH, OR REPRODUCE A COPY of any portion or all of the EchoNet-Pediatric
    Dataset to others without specific prior written permission";
  - item 6: "You must not modify, reverse engineer, decompile, or create derivative works from the EchoNet-Pediatric
    Dataset."

  This round treated model training as research use. That reading did not address item 6. Keeping weights on the
  server settles sharing only, not whether training or derived features count as derivative works. The project has
  no record of the agreement as actually accepted at download, or of its version; the data reached the server on
  2026-09-13. A clarification request for the project lead to send is drafted in
  `docs/data_requests/2026-10-05_echonet_pediatric_clarification.md`.
  - Until it is clarified, there is no further training on Pediatric and nothing derived from it leaves the server.
  - Existing artefacts are kept.
  - The repository holds aggregate numbers only.

## 2. Systems and preprocessing

Every weight was trained on EV9V only. No system has seen Pediatric.

| System | What it is | Preprocessing of the 10 Hz samples |
|---|---|---|
| E0 + B3 | v3 ResNet-18 frame encoder and head, then B3 with E0's EV9V parameters | aspect-preserving letterbox to 224, ImageNet normalisation (as frozen, and as on EV9V) |
| E1 + B3 | EchoViewCLIP stage-1 image encoder and linear probe, then B3 with E1's EV9V parameters | short side 256, centre crop 224, CLIP normalisation, fp32 (as on EV9V; the 112 × 112 frames are upsampled) |
| E1 + ASRF | the E1 ASRF of rounds 2–5, seeds 0–2, official decode | as E1 |
| B7 | the official EchoViewCLIP stage-1 video model, 16-frame windows of 1.6 s with a stride of 0.5 s, all nine outputs | short side 256, centre crop 224 (as on EV9V) |

The five family outputs stay, and a clip read as any other class counts as wrong. B7's four PSAX levels count as PSAX.
Its PMPALA output is kept as an error instead of being dropped as on EV9V. A clip's label is the majority of its
per-sample labels.

## 3. Results on the test folds

6,354 clips from 1,549 groups.

| System | A4C recall | PSAX recall | Macro-F1 (A4C, PSAX) | Main error | Groups with misread PSAX / A4C |
|---|---|---|---|---|---|
| E0, per-sample argmax | 0.846 | 0.277 | 0.666 | 2,288 PSAX read as PLAX | 1,273 of 1,542 / 315 of 1,287 |
| E0 + B3 | 0.844 | 0.293 | 0.676 | 2,228 PSAX read as PLAX | 1,256 of 1,542 / 316 of 1,287 |
| E1, per-sample argmax | 0.770 | 0.982 | 0.927 | 469 A4C read as A5C | 56 of 1,542 / 427 of 1,287 |
| E1 + B3 | 0.740 | 0.983 | 0.917 | 552 A4C read as A5C | 55 of 1,542 / 478 of 1,287 |
| E1 + ASRF, seed 0 | 0.843 | 0.978 | 0.946 | 296 A4C read as A5C | 71 of 1,542 / 310 of 1,287 |
| E1 + ASRF, seed 1 | 0.943 | 0.931 | 0.949 | 202 PSAX read as A4C | 208 of 1,542 / 128 of 1,287 |
| E1 + ASRF, seed 2 | 0.939 | 0.956 | 0.963 | 98 PSAX read as A4C | 141 of 1,542 / 130 of 1,287 |
| E1 + ASRF, majority of seeds | 0.932 | 0.959 | 0.962 | 100 PSAX read as A4C | 130 of 1,542 / 149 of 1,287 |
| B7 | 0.855 | 0.936 | 0.942 | 239 A4C read as A5C; 183 PSAX read as PMPALA | 186 of 1,542 / 291 of 1,287 |

The full tables, with every target class and both parts, are in `step1_tables.md` and `step1_summary.json`. The
calibration folds give A4C 0.767 and PSAX 0.976 for E1 + B3, and PSAX 0.325 for E0 + B3.

## 4. Step 2: the adaptation comparison

The rules were committed before any adaptation training (`configs/experiments/pediatric_adapt_v1.yaml`, commit
1742989), and the report script before any result (0cda286).
- **Arms.**
  - D0 is E1 unchanged.
  - D1 trains only E1's linear head, starting from E1's probe.
  - D2 trains the same head plus the encoder's last block, ln_post and proj.
- **Training data.** Each update holds 32 frames from the Pediatric calibration folds, drawn by class, then group,
  then clip. It also holds 32 EV9V training frames as replay. Training ran for 1,500 updates, the final weights were
  kept, and three seeds were run. In every seed, D1 and D2 read bit-identical batches.
- **Checks.** D1 changed no encoder tensor, and D2 changed exactly the 15 tensors of the last block, ln_post and proj.
- **Downstream.** Every arm got B3 re-selected on EV9V's calibration half. D2 got an ASRF retrained on its EV9V
  features. D1's ASRF is E1's, because its features are E1's.
- **Isolation.** No checkpoint, epoch or threshold was chosen on Pediatric data. The test folds were only scored.

Pediatric test folds, B3 (per seed). Differences are paired over the same resampled groups.

| Arm | A4C recall | PSAX recall | Macro-F1 | Clips turned right / wrong against D0 (groups) |
|---|---|---|---|---|
| D0 | 0.740 | 0.983 | 0.917 | – |
| D1 | 0.985 / 0.985 / 0.986 | 0.995 / 0.994 / 0.995 | 0.991 | 702–706 (480–482) / 2 (2) |
| D2 | 0.991 / 0.991 / 0.993 | 0.998 / 0.998 / 0.997 | 0.995 | 732–735 (491–494) / 3 (3) |

- D1 − D0 macro-F1 is +7.4 points in every seed, and D2 − D1 is +0.4 (both consistent).
- With ASRF, D2 turns 265 to 399 clips right and 1 to 12 wrong, against E1's ASRF.

## 5. Step 3: segmentation on the Pediatric exam streams

The frozen test exam streams join whole clips of one exam in clip order (2,574 streams). Their view changes come from
stitching, so they are a constructed proxy, not continuous scanning. Values are per seed.

| Arm, system | Segment F1@50 | Boundary recall (0.5 s) | View changes found with both views right | Cuts away from clip joins, per clip |
|---|---|---|---|---|
| D0, B3 | 0.848 | 0.976 | 0.726 | 0.032 |
| D1, B3 | 0.981 / 0.981 / 0.982 | 0.976 / 0.977 / 0.977 | 0.971 / 0.972 / 0.973 | 0.012 / 0.013 / 0.012 |
| D2, B3 | 0.985 / 0.985 / 0.989 | 0.972 / 0.972 / 0.986 | 0.971 / 0.971 / 0.985 | 0.005 / 0.004 / 0.009 |
| D0 = D1, ASRF | 0.888 / 0.842 / 0.866 | 0.931 / 0.677 / 0.726 | 0.782 / 0.627 / 0.667 | 0.015 / 0.003 / 0.005 |
| D2, ASRF | 0.903 / 0.961 / 0.828 | 0.707 / 0.912 / 0.480 | 0.699 / 0.894 / 0.478 | 0.001 / 0.002 / 0.000 |

- For B3, every adapted arm is consistently ahead of D0 in segment F1@50, in changes found with both views right
  (about +25 points) and in fewer cuts.
- B3's boundary recall shows no consistent difference: D1 − D0 is 0 to +0.1 points and D2 − D0 is −0.4 to +1.0. The
  adapted heads label the spans better; they do not find more boundaries.
- For ASRF, D2 − D0 is not consistent in any segmentation measure. Its boundary recall falls in two seeds and rises
  in one. Where these boundaries are lost is analysed in `boundary_fault_localisation.md`.

## 6. Step 4: EV9V

| Arm, system | Selection: segment F1@50 per seed | Calibration: segment F1@50 per seed |
|---|---|---|
| D0, B3 | 0.894 | 0.959 |
| D1, B3 | 0.906 / 0.876 / 0.899 | 0.964 / 0.959 / 0.965 |
| D2, B3 | 0.904 / 0.907 / 0.926 | 0.971 / 0.966 / 0.972 |
| D0 = D1, ASRF | 0.910 / 0.909 / 0.907 | 0.976 / 0.964 / 0.964 |
| D2, ASRF | 0.911 / 0.917 / 0.910 | 0.970 / 0.974 / 0.975 |

- **Contrasts against D0.** No difference from D0 was detected consistently across the three seeds, on either half,
  for B3 or ASRF, in segment F1@50, native macro-F1, boundary recall or cuts. That is not evidence of no loss.
- **Thresholds.** They are missed in places, as they already were for D0's ASRF.
  - D2 + B3 has boundary recall 0.932 and 0.937 on the selection half in two seeds, below 0.95.
  - D2 + ASRF has native macro-F1 0.954 and 0.951 there in two seeds, below the 0.973 floor.
- **Final goal.** No arm meets it on either half.

## 7. Files

- **Rules.** `configs/experiments/pediatric_transfer_v2.yaml` (step 1) and `configs/experiments/pediatric_adapt_v1.yaml` (steps 2-4).
- **Code.**
  - `scripts/pediatric_extract.py` runs E0, with a checkpoint option added.
  - `scripts/pediatric_extract_e1.py` runs E1.
  - `scripts/pediatric_b7.py` runs B7.
  - `scripts/pediatric_transfer_v2.py` scores all systems.
  - `scripts/pediatric_adapt.py` trains D1 and D2, and `scripts/pediatric_adapt_report.py` writes the step 2-4 report.
- **Numbers.** `step1_summary.json`, `step1_tables.md`, `step2_report.json`, `step2_tables.md` and `boundary_faults.json` hold
  aggregates only. The fault localisation is `boundary_fault_localisation.md` (`scripts/pediatric_boundary_faults.py`).
- **Server only.** The features are in `cache_v2/pediatric/`. The per-clip table is `runs/ped2_transfer/clips.csv`, and the
  adapted weights are in `runs/pediatric_adapt/`. They are not shared.
