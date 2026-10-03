# Model development, round 3 (2026-10-02): boundaries learned on their own

This round tests one question. Can a model that only learns where views change, while a frozen classifier names each
segment, beat B3 and ASRF? Recovering the classification that M5 loses inside its own segments does not count as
success. The paper stays paused. Terms are as in the round-1 and round-2 reports
(`docs/results/2026-10-02_model_development/README.md`, `docs/results/2026-10-02_model_development_round2/README.md`).

- **E0 and E1.** E0 is the v3 ResNet-18 with its own classification head. E1 is the frozen EchoViewCLIP stage-1 image
  encoder with its five-class linear probe. "The frozen classifier" means the E0 head or the E1 probe, whichever
  matches the input.
- **Systems.** M5 is the boundary-controlled model with segment pooling. ASRF is the official ASRF trained with the
  shared recipe. B3 is the change-point decoder tuned on the calibration half. B7 is the official EchoViewCLIP stage 1
  applied with a sliding window.
- **Step 1, step 2, step 3.** Step 1 decodes the boundaries that M5 and ASRF already predict, without training. Step 2
  trains a model that only predicts boundaries. Step 3 is a diagnostic of M5's errors.
- **Metrics.** Segment F1@50 counts a predicted segment as right when it has the reference class and overlaps a
  reference segment by at least half (intersection over union 0.5 or more); it is measured on multi-fragment streams.
  Boundary recall is the share of reference view changes with a predicted cut within 0.5 s. Native macro-F1 gives each
  single-view recording the majority label of its samples and averages F1 over the five classes.
- **Validation only.** Both halves have been used during development, so every result is a development result. The
  test split was not read.

## Answer

1. **Learning boundaries on their own does not beat B3 or ASRF.**
   - On E0, the boundary-only model with calibrated decoding reaches a three-seed mean segment F1@50 of 0.919 on the
     selection half and 0.947 on the calibration half. The strongest previous configuration reaches 0.934 and 0.972.
   - Against ASRF on the same input, no consistent difference is detected on either half. Against the strongest
     previous configuration, it is consistently lower on the calibration half (−2.3 to −2.6 points). On the selection
     half it is lower in all three seeds (−0.4 to −2.7), and one of the three intervals excludes zero.
   - On E1, it is consistently lower than ASRF, B3 and step 1 on the selection half, by 4.7 to 9.2 points. It cuts
     single-view recordings far more often than either.
   - No candidate of this round meets the success criterion on either half. No candidate even has a positive margin.
2. **Neither the boundary training nor the decoding produced a gain.**
   - Step 2 is not better than step 1. On E0 no consistent difference is detected. On E1 step 2 is consistently lower
     on the selection half.
   - Step 1 itself shows no consistent difference from the models' default outputs in any of the eight
     input-method-half cells (−2.3 to +1.5 points).
   - The first pre-declared rule says a step 2 that is not better than step 1 means the gain comes from the decoding.
     Here there is no gain to assign to either.
3. **The M5 modification route ends here.** The second pre-declared rule applies. The best candidate shows no
   consistent difference from ASRF and stays below the strongest B3 configuration. The proposed next step is to study
   the recordings that B3 and ASRF both get wrong, not to add modules to M5. It has not been started.
4. **The round-2 statement about case B is withdrawn, and the neighbour-swap check supports the withdrawal.**
   - Most of M5's lost fragments (58 of 64) are A4C and A5C confused with each other.
   - When an A4C fragment called A5C has an A5C neighbour, M5's convolution stage usually still says A4C (23 of 28).
     The attention stage turns it to A5C. Removing the neighbours, or giving it A4C neighbours, makes the output right
     in 26 to 28 of 28 fragments, in all 12 recordings.
   - Without an A5C neighbour, the convolution stage is already wrong in most fragments (13 of 19). Changing the
     neighbours helps less there.
   - This is a counterfactual diagnostic. Reference fragment limits were used only to build its inputs.

## 1. What was fixed before running

- `configs/experiments/model_development_v3.yaml` was written before any round-3 decoding or training. It inherits
  everything else from round 2. That covers the protocol (`protocol.json`, SHA-256 `a897483b…`), the two validation
  halves, the 1,500-stream training bank, the optimiser, the update budget and the checkpoint rule.
- **Decoding rule (steps 1 and 2).** The model's boundary probability is read at every transition except the first.
  Peaks at or above a threshold are kept, with non-maximum suppression within 5 samples (0.5 s). The sequence is cut
  at those peaks. Each piece gets the class with the highest mean probability from the frozen classifier inside it.
  Adjacent pieces with the same class merge. The class probabilities enter as they are, with no temporal transform,
  gate or pooling.
- **Threshold choice.** The threshold (0.3 to 0.9 in steps of 0.1) is chosen per run on the calibration half by the
  shared rule. That rule takes the highest segment F1@50 among settings that keep native macro-F1 at or above the
  floor and boundary recall at or above 0.95, with differences under 0.002 broken by fewer cuts. If no setting meets
  both constraints, it falls back to recall only, then floor only, then neither. The chosen threshold is then applied
  unchanged to the selection half.
- **Native floor.** Each input's frame-majority native macro-F1 minus 0.005. That is 0.960 (selection) and 0.977
  (calibration) for E0, and 0.973 and 0.980 for E1.
- **Strongest previous configuration,** taken from the round-2 report. On the selection half it is B3 tuned on the E0
  probabilities (0.934). On the calibration half it is B7 with B3 on its posteriors (0.972).
- **Success criterion,** unchanged from round 1. On a half, the three-seed mean segment F1@50 must be at least 1 point
  above the strongest previous configuration. Boundary recall must be at least 0.95 and native macro-F1 at or above
  the floor in every seed.
- **Interpretation rules,** set by the user. If step 2 is not better than step 1, the gain belongs to the decoding. If
  the best candidate only matches ASRF and stays below the strongest B3, the M5 route ends, and the next step studies
  the recordings that B3 and ASRF both get wrong.

## 2. Step 1: the existing boundaries, named by the frozen classifier

The six M5 runs and six ASRF runs of rounds 1 and 2 were decoded with the rule above. Nothing was retrained.

| Input | Boundaries from | Thresholds chosen (s0, s1, s2) | Selection, mean | Calibration, mean | Default output, selection | Default output, calibration |
|---|---|---|---:|---:|---:|---:|
| E0 | M5 | 0.9, 0.7, 0.6 | 0.913 | 0.946 | 0.917 | 0.949 |
| E0 | ASRF | 0.3, 0.5 (floor not met), 0.9 | 0.919 | 0.948 | 0.927 | 0.952 |
| E1 | M5 | 0.7, 0.7, 0.6 | 0.900 | 0.966 | 0.896 | 0.964 |
| E1 | ASRF | 0.8, 0.8, 0.9 | 0.898 | 0.971 | 0.909 | 0.968 |

Paired by seed, no difference between step 1 and the default output is consistent in any cell. The per-seed
differences range from −2.3 to +1.5 points. Step 1 stays below the strongest previous configuration. On the selection
half it is lower in every seed (−1.1 to −3.9 points), consistently so for three of the four rows. On the calibration
half, E0 M5 is consistently lower, and E1 ASRF comes closest (0.971 against 0.972, with its threshold chosen on that
same half).

## 3. Step 2: a model that only learns boundaries

- **Model.** The frozen embeddings pass through a 1×1 projection to 128 channels, three dilated residual temporal
  convolutions (dilations 1, 2 and 4; together they see 15 samples) and M5's boundary head. There is no class head,
  attention, gate or pooling. The class of each piece comes from the frozen classifier as in step 1.
- **Supervision.** Only the boundary loss is used. A join between two fragments whose view class changes is a
  positive. A join between two fragments of the same class is a strong negative (weight 1). Every other transition
  is a weak negative (weight 0.1).
- **Training.** There are six runs, E0 and E1 with three seeds each. They use the same training bank, native windows,
  optimiser, learning rate (3e-4), batch size, epoch limit (50) and early stopping (8 epochs) as M5 and ASRF. The
  visual encoder is not fine-tuned. Checkpoints are chosen on the selection half by the shared rule, decoding at
  threshold 0.5.
- **Chosen checkpoints.** For E0 these are epochs 34, 9 and 2 (out of 43, 18 and 11 run). For E1 they are epochs 2, 6
  and 6 (out of 11, 15 and 15).

| Input | Decoding | Selection, s0 / s1 / s2 (mean) | Calibration, s0 / s1 / s2 (mean) | Cuts per single-view recording, selection | Same-class joins cut, selection |
|---|---|---|---|---|---|
| E0 | threshold 0.5 | 0.911 / 0.882 / 0.861 (0.885) | 0.940 / 0.931 / 0.929 (0.933) | 0.04–0.13 | 6–11% |
| E0 | calibrated: 0.8, 0.9, 0.8 | 0.918 / 0.931 / 0.907 (0.919) | 0.949 / 0.946 / 0.947 (0.947) | 0.01–0.06 | 2–5% |
| E1 | threshold 0.5 | 0.818 / 0.820 / 0.847 (0.828) | 0.920 / 0.952 / 0.947 (0.940) | 0.16–0.29 | 10–12% |
| E1 | calibrated: 0.5, 0.8, 0.5 | 0.818 / 0.845 / 0.847 (0.837) | 0.920 / 0.960 / 0.947 (0.942) | 0.15–0.29 | 10–11% |

For comparison, on the E1 selection half B3 cuts single-view recordings 0.05 times per recording and cuts 4% of
same-class joins. ASRF cuts them 0.01–0.05 times and cuts 3–6% of same-class joins. The E1 boundary-only model still
finds the real view changes (recall 0.97–0.98 on the selection half). Its loss comes from extra cuts. Round 2 found
that the E1 probe's per-sample output varies more than E0's. Whether that also drives these extra cuts was not tested.

For two E1 seeds the calibration half chose 0.5, the checkpoint threshold. Their calibrated and default rows are
identical, which also checks that the training-time decoding and the decoder script agree.

**The checkpoint and threshold rules do not appear to explain the shortfall.**
- The native floor rejected many epochs. Only 27 of 43, 10 of 18 and 3 of 11 E0 epochs met it, and 3 of 11, 2 of 15
  and 4 of 15 E1 epochs. On these halves one recording can decide the floor. On the E0 selection half, 0.951 against
  0.965 is one A5C recording (7 against 8 of 11 right).
- Without the checkpoint rule, the highest selection-half value among the trained epochs, all decoded at the fixed
  threshold 0.5, is 0.916 for E0 and 0.883 for E1. This is a maximum over epochs at one threshold. It is not an upper
  bound over all combinations of epoch and threshold, because the other thresholds were not scored for every epoch.
  It also uses the selection half both to choose and to score. It is still below 0.934.
- Without the floor in the threshold choice, the best calibration-half values over the threshold grid, at the chosen
  checkpoints, are 0.949, 0.946 and 0.947 for E0 and 0.944, 0.960 and 0.961 for E1. All are below 0.972. On E1,
  higher thresholds cut less and score better, but they lower native macro-F1 below the floor.
- For E0 seeds 1 and 2, no threshold met the native floor on the calibration half. The rule fell back to recall only.

## 4. Strict comparison

Three-seed mean segment F1@50. Bold marks the strongest previous configuration on that half.

| Input | System | Decoding | Selection | Calibration |
|---|---|---|---:|---:|
| E0 | M5 | default | 0.917 | 0.949 |
| E0 | M5 | step 1 | 0.913 | 0.946 |
| E0 | ASRF | default | 0.927 | 0.952 |
| E0 | ASRF | step 1 | 0.919 | 0.948 |
| E0 | boundary-only | step 2 (calibrated) | 0.919 | 0.947 |
| E0 | B3 tuned | — | **0.934** | 0.934 |
| E1 | M5 | default | 0.896 | 0.964 |
| E1 | M5 | step 1 | 0.900 | 0.966 |
| E1 | ASRF | default | 0.909 | 0.968 |
| E1 | ASRF | step 1 | 0.898 | 0.971 |
| E1 | boundary-only | step 2 (calibrated) | 0.837 | 0.942 |
| E1 | B3 tuned | — | 0.894 | 0.959 |
| — | B7 with B3 | — | 0.880 | **0.972** |

Paired contrasts of step 2, per seed (s0 / s1 / s2), in points of segment F1@50. "None" means no consistent
difference detected. "Lower" means all three 95% intervals lie below zero.

| Step 2 minus | E0 selection | E0 calibration | E1 selection | E1 calibration |
|---|---|---|---|---|
| step 1 with M5's boundaries | +0.3 / +1.5 / +0.1, none | +0.6 / −0.1 / 0.0, none | −7.7 / −5.8 / −5.4, lower | −3.8 / −0.5 / −2.9, none |
| step 1 with ASRF's boundaries | +0.1 / +0.7 / −1.0, none | +0.2 / +0.2 / −0.8, none | −7.8 / −5.5 / −5.2, lower | −4.5 / −1.3 / −2.7, lower |
| original M5 | +1.2 / +1.2 / −1.8, none | −0.5 / +0.0 / −0.2, none | −6.8 / −6.1 / −4.9, lower | −4.9 / −0.1 / −1.5, none |
| ASRF | +0.7 / −1.6 / −1.7, none | +1.3 / −1.4 / −1.3, none | −9.2 / −6.4 / −6.0, lower | −5.5 / −0.4 / −1.8, none |
| B3 tuned on the same input | −1.6 / −0.4 / −2.7, none | +1.5 / +1.2 / +1.3, none | −7.6 / −4.9 / −4.7, lower | −3.9 / +0.1 / −1.2, none |
| strongest previous configuration | −1.6 / −0.4 / −2.7, none | −2.3 / −2.6 / −2.5, lower | −11.6 / −9.0 / −8.8, lower | −5.2 / −1.2 / −2.6, none |

**Success test.** Every candidate's mean is below the strongest previous configuration on both halves. That covers
step 2 at either threshold, step 1 with either model's boundaries, and the default M5 and ASRF outputs, on both
inputs. The closest are E1 ASRF with step 1 on the calibration half (−0.1 points, threshold chosen on that half) and
E0 ASRF's default output on the selection half (−0.7). The recall and native constraints therefore decide nothing.
They are listed in `round3_tables.md`.

## 5. Reading under the pre-declared rules

- **Step 2 against step 1.** Step 2 is not better than step 1. The rule assigns any gain to the decoding. But step 1
  shows no gain over the default outputs either. So neither the separate boundary training nor the classifier-named
  decoding improved segmentation.
- **ASRF and B3.** The best round-3 candidate is the E0 boundary-only model. It shows no consistent difference from
  ASRF and stays below the strongest B3 configuration on both halves. On E1 it is consistently below ASRF on the
  selection half. Under the rule, the M5 modification route ends.
- **Next step (proposed, not started).** Study the recordings that B3 and ASRF both get wrong. A first pass could list,
  on both halves, the reference view changes and fragments that both systems miss or mislabel in every seed. It would
  deduplicate them by source recording, sort them by class pair, fragment length and contrast variant, and look at the
  frames and the frame-level probabilities. A mechanism would be proposed only after that, with its test fixed in
  advance. Step 3 already points at one pattern: short A4C fragments next to other views being called A5C.

## 6. Step 3: neighbour-swap check (correction to round 2)

**Withdrawn statement.** The round-2 report said that confident boundaries at both ends of case B excluded mixing
across the boundary. That is withdrawn. In M5, three temporal convolutions run before the boundary head and before
attention is restricted. Together they see 15 samples, 7 on each side, and case B's fragment is only 10 samples long.
The round-2 README now carries the correction.

**Method.** The check uses the frozen E1 M5 runs (three seeds) on both halves. It selects every fragment that the E1
probe labels right by its mean probability but that M5 gets wrong on at least 80% of its samples. That gives 64
fragments from 25 source recordings. Each fragment's embeddings stay fixed while its surroundings change.
- **Original.** The full original stream, as in normal inference.
- **Isolated.** The fragment on its own.
- **Same-class neighbours.** The previous and next fragments are replaced by blocks of the same length from other
  recordings of the target's class.
- **Other-recording neighbours.** The previous and next fragments are replaced by blocks from other recordings of the
  same classes as the original neighbours.

Donor blocks come from the same half's single-view recordings, drawn with a fixed seed. For each version, the
fragment's mean class probability is read at the convolution (local) classifier, the attention (context) classifier,
the mix before pooling and the output. Reference limits are used only to build these inputs. Normal inference never
sees them.

**Results.** Each cell gives fragments right, then recordings where most of their fragments are right. The full tables,
including the stage before pooling, are in `neighbor_check.md` and `neighbor_check_groups.md`. Recordings can appear
in more than one group.

| Group | Surroundings | Convolution stage | Attention stage | Output |
|---|---|---|---|---|
| All 64 fragments (25 recordings) | original | 40/64 (17/25) | 9/64 (4/25) | 0/64 (0/25) |
| | isolated | 49/64 (20/25) | 40/64 (15/25) | 43/64 (18/25) |
| | same-class neighbours | 54/64 (22/25) | 53/64 (22/25) | 59/64 (23/25) |
| | other-recording neighbours | 41/64 (17/25) | 23/64 (11/25) | 21/64 (9/25) |
| A4C called A5C, an A5C neighbour (28 fragments, 12 recordings) | original | 23/28 (10/12) | 2/28 (0/12) | 0/28 (0/12) |
| | isolated | 26/28 (11/12) | 26/28 (12/12) | 26/28 (12/12) |
| | same-class neighbours | 27/28 (12/12) | 27/28 (12/12) | 28/28 (12/12) |
| | other-recording neighbours | 22/28 (9/12) | 10/28 (4/12) | 13/28 (5/12) |
| A4C called A5C, no A5C neighbour (19 fragments, 8 recordings) | original | 6/19 (3/8) | 1/19 (0/8) | 0/19 (0/8) |
| | isolated | 10/19 (5/8) | 7/19 (3/8) | 8/19 (4/8) |
| | same-class neighbours | 10/19 (4/8) | 9/19 (4/8) | 14/19 (6/8) |
| | other-recording neighbours | 6/19 (3/8) | 2/19 (1/8) | 2/19 (1/8) |
| Other errors (17 fragments, 8 recordings) | original | 11/17 (6/8) | 6/17 (4/8) | 0/17 (0/8) |
| | isolated | 13/17 (6/8) | 7/17 (3/8) | 9/17 (5/8) |
| | same-class neighbours | 17/17 (8/8) | 17/17 (8/8) | 17/17 (8/8) |
| | other-recording neighbours | 13/17 (7/8) | 11/17 (6/8) | 6/17 (3/8) |

The other errors are mostly A5C called A4C (11 of 17).

**What it shows.**
- The neighbours, not the fragment's own embeddings, decide most of these errors. With neighbours of the fragment's
  own class, the output is right in 59 of 64 fragments.
- The largest change is between the convolution stage and the attention stage. For A4C fragments called A5C next to
  an A5C fragment, the convolution stage is right in 23 of 28 and the attention stage in 2.
- Keeping the neighbours' classes but taking them from other recordings fixes about a third of the fragments (21 of
  64). So both the neighbours' class and their particular content matter.
- Case B (selection half, stream `multi_selection_0004`, samples 126–136, A4C, M5 seed 0) has PLAX and PSAX
  neighbours, not A5C. With the original neighbours, the convolution stage is undecided (A4C 0.53, A5C 0.47). The
  attention stage says A5C (0.76), and so does the output (0.71). Both edges carry boundary probability 1.0.
  Isolated, every stage says A4C (output 0.82). With A4C neighbours the output is A4C (1.00). With PLAX and PSAX
  neighbours from other recordings it is A5C again (0.60). A confident boundary therefore does not stop the mixing.

**What it does not show.** The check does not separate two routes. Attention may reach across the boundary itself,
or it may amplify neighbour information the convolutions already mixed in. The spliced test inputs resemble M5's
training streams, which are spliced the same way, but they are still constructed inputs. The fragments were selected
because M5 got them wrong, so the original row is zero by construction.

## 7. Limitations

- Both validation halves have been used during development. The test split was not read. Every result here is a
  development result.
- Bootstrap intervals resample sessions for single-view recordings and streams for multi-fragment streams. Clip reuse
  across streams is not modelled, so the intervals are optimistic. The three seeds are not independent test sets.
- The selection half has only 11 A5C and 13 SC4C single-view recordings. One recording moves native macro-F1 by about
  1.4 points, which makes the native floor a coarse gate. The bounds in section 3 show the conclusion does not depend
  on it.
- The step-1 thresholds and the step-2 calibrated thresholds were chosen on the calibration half. Calibration-half
  values for those rows are therefore in-sample.
- Two of six boundary-only runs chose epoch 2, and early stopping ended them after 11 epochs. Two more chose epoch 6.
  This is the shared rule. The best trained epoch at threshold 0.5 (section 3) also stays below the target, but that
  is a maximum over epochs at one threshold, not a bound over all epochs and thresholds.

## 8. Files and runs

- Configuration: `configs/experiments/model_development_v3.yaml`.
- Code:
  - `echo_routing/temporal/segments.py` (`boundary_decode`);
  - `echo_routing/temporal/models/boundary_context.py` (`BoundaryOnlyNet`);
  - `echo_routing/temporal/dev_heads.py` (`BoundaryOnlyHead`);
  - `echo_routing/temporal/dev_data.py` (`Seq.prob`, the frozen classifier's probabilities, never a model input);
  - `scripts/posterior_decoders.py` (the `bdec` family);
  - `scripts/train_boundary_only.py`;
  - `scripts/round3_neighbor_check.py` and `scripts/summarize_neighbor_check.py`;
  - `scripts/report_model_development_round3.py`;
  - `scripts/model_development_queue.py` (log names now carry the feature-manifest digest).
- Tests: `tests/test_boundary_only.py`. The full suite (223 tests) passed on the server.
- Runs, under `~/echo-view-routing/runs/` on the server:
  - **Step 1:** `20261002-233821-md_posterior_decoders_r3_E0-af344a52-s0` and
    `20261002-233822-md_posterior_decoders_r3_E1-af344a52-s0`.
  - **Step 2 training, E0:** `20261002-233817-md_BO_lr0.0003-af344a52-s0`,
    `20261002-234119-md_BO_lr0.0003-15d123b3-s1` and `20261002-234223-md_BO_lr0.0003-543945ff-s2`.
  - **Step 2 training, E1:** `20261002-233817-md_BO_lr0.0003-af344a52-s0-2`,
    `20261002-233917-md_BO_lr0.0003-15d123b3-s1` and `20261002-234119-md_BO_lr0.0003-543945ff-s2`.
  - **Step 2 decoding:** `20261002-234637-md_posterior_decoders_r3_BO_E0-af344a52-s0` and
    `20261002-234637-md_posterior_decoders_r3_BO_E1-af344a52-s0`.
  - **Step 3:** `20261002-233937-md3_neighbor_check-af344a52-s0`.
  - **Report:** `runs/md_round3/report_v2/`, built from the spec `runs/md_round3/report_spec_round3.json`. The first
    build (`report/`) has identical tables. Its JSON also paired each strongest configuration with the other half's
    numbers. The code review flagged this, and the second build compares each one on its own half only.
- Run-to-input mapping: each training run was matched to its input through its own `run_manifest.json`. Two queues of
  seed 0 once wrote one log, so the status file `runs/md_round3/bo_E1_s0.json` named the E0 run. It is corrected, with
  a note. Logs now carry the feature-manifest digest.
- Frozen models: `models_frozen/boundary_only/round3_{e0,e1}_s{0,1,2}`. Each checkpoint's SHA-256 matched its run
  manifest before freezing.
- Copies in this folder:
  - `round3_tables.md`: all systems, per seed and per half, and the success test;
  - `round3.json`: metrics, contrasts, success test and spec;
  - `neighbor_check.md`, `neighbor_check.json` and `neighbor_check_groups.md`: step 3.
