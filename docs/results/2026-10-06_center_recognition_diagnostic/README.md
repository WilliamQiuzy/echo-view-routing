# Where native recognition regressed in the N/S/C round (2026-10-06)

For the frozen E1, the grid sampling of window centres costs one selection clip and nothing on the calibration half.
For the trained frame heads it moves 0 to 3 clips either way, with no consistent direction. For the primary C models, the frame path before
the temporal module loses 0 to 2 clips per seed and half, and the temporal readout later recovers most of them. The
temporal window readout itself breaks 2 to 3 clips in every C seed and half that the frozen E1 and the trained frame
head both read correctly, and fixes 0 to 2.

This was inference with the nine existing models only. Nothing was trained, tuned or relabelled. The test split and
EchoNet-Pediatric were not read. The plan is `docs/superpowers/plans/2026-10-06-recognition-regression-localisation.md`.
The paths, rules and statistics were fixed in `configs/experiments/center_recognition_diagnostic_v1.yaml` and committed
before any result (3aae4d4). A fix to the run check followed before any metric (66235aa).

## The five paths

| Path | Probabilities | Time grid |
|---|---|---|
| F-dense | the frozen E1 probe (E1 cache, float32) | every 10 Hz sample |
| F-grid | the same | read only at the window target samples, filled back to the nearest centre |
| J-dense | a trained model's auxiliary frame head on its own encoder features | every sample |
| J-grid | the same | the same target samples and fill-back as F-grid |
| T-window | the same model's temporal Transformer and window head | B7's 16-frame windows, stride 5, nearest centre |

- **How a clip is labelled.** Every row is the argmax per sample. A native clip's label is the majority of its
  samples.
- **What changes between paths.** Grid against dense isolates the sampling and fill-back. J against F changes the
  encoder and the frame head together, so it is not a test of the ViT alone. T against J changes the temporal layers,
  the LayerNorm and the window head together.
- **Checks before any result.**
  - The frozen E1 + B3 reproduces exactly from F-dense with its archived numeric setting (difference 0).
  - T-window reproduces the previous round's argmax rows for all nine models (difference 0).
  - The previous round kept no per-sequence archive, so the comparison covers its summary rows.

## 1. Native clips per path

These are wrong native clips out of 284 on the selection half and 237 on the calibration half. The frozen E1 gets 3
(F-dense) or 4 (F-grid) wrong on the selection half and 2 on the calibration half.

| Model | J-dense | J-grid | T-window | A5C right, J-grid → T (sel) | A5C right, J-grid → T (cal) |
|---|---|---|---|---|---|
| C s0 | 6 / 3 | 6 / 1 | 7 / 4 | 8 → 8 of 11 | 17 → 16 of 18 |
| C s1 | 6 / 2 | 6 / 2 | 6 / 3 | 8 → 9 | 17 → 16 |
| C s2 | 4 / 3 | 4 / 3 | 6 / 4 | 10 → 9 | 17 → 16 |
| S s0 | 7 / 2 | 5 / 1 | 8 / 3 | 9 → 7 | 17 → 16 |
| S s1 | 9 / 2 | 10 / 1 | 11 / 3 | 8 → 7 | 17 → 16 |
| S s2 | 9 / 1 | 10 / 1 | 10 / 3 | 9 → 7 | 17 → 16 |
| N s0 | 6 / 3 | 6 / 2 | 7 / 4 | 8 → 7 | 17 → 16 |
| N s1 | 7 / 1 | 6 / 1 | 7 / 2 | 8 → 7 | 17 → 16 |
| N s2 | 8 / 3 | 10 / 3 | 9 / 4 | 8 → 7 | 17 → 16 |

The first three columns give selection / calibration. For the frozen E1, A5C right is 10 (F-dense) and 9 (F-grid) of
11 on the selection half, and 16 of 18 on the calibration half.

- **Calibration half.** The trained frame head gets 1 to 3 clips wrong, against the frozen E1's 2. The window head then
  loses one A5C clip in all nine models.
- **Selection half.** The frame path already loses clips in most models. In S s1, S s2 and N s2 it newly gets 6 to 7
  clips wrong that F-grid read correctly, mostly A4C clips. The C models lose 1 to 2.

## 2. The three paired effects

Fixed and newly wrong native clips per model, against the earlier stage. Each native clip is one recording. The
session bootstrap intervals for native macro-F1 are in `report_tables.md`. With so few changed clips, almost all of
them touch or include zero.

| Model | Grid, J-grid − J-dense (sel; cal) | Frame path, J-grid − F-grid (sel; cal) | Temporal readout, T − J-grid (sel; cal) |
|---|---|---|---|
| C s0 | 2/2; 2/0 | 0/2; 1/0 | 2/3; 0/3 |
| C s1 | 2/2; 1/1 | 0/2; 1/1 | 2/2; 1/2 |
| C s2 | 1/1; 1/1 | 1/1; 1/2 | 1/3; 1/2 |
| S s0 | 2/0; 1/0 | 1/2; 1/0 | 0/3; 0/2 |
| S s1 | 1/2; 1/0 | 0/6; 1/0 | 1/2; 0/2 |
| S s2 | 2/3; 0/0 | 1/7; 1/0 | 4/4; 0/2 |
| N s0 | 1/1; 1/0 | 0/2; 1/1 | 1/2; 0/2 |
| N s1 | 2/1; 0/0 | 0/2; 1/0 | 1/2; 0/1 |
| N s2 | 1/3; 1/1 | 0/6; 1/2 | 3/2; 1/2 |

Each cell gives clips fixed / clips newly wrong.

- **Grid for the frozen E1.** F-grid − F-dense loses 1 selection clip (an A5C clip) and changes nothing on the
  calibration half.
- **Where the clips go.**
  - The frame path's losses on the selection half are mainly A4C clips in S and N. Those of C are one A4C clip per
    seed and, in seeds 0 and 1, one A5C clip.
  - The temporal readout's losses are A4C and A5C clips on both halves, plus one PSAX clip.

## 3. The C models clip by clip

Native clips by whether F-grid, J-grid and T-window read them correctly (+) or not (−). The eight cells add up to the
clip count in every seed and half. Cells that are zero everywhere are left out.

| Model, half | F+ J+ T+ | F+ J+ T− | F+ J− T+ | F+ J− T− | F− J+ T+ | F− J− T− |
|---|---|---|---|---|---|---|
| C s0, selection | 275 | 3 | 2 | 0 | 0 | 4 |
| C s1, selection | 276 | 2 | 2 | 0 | 0 | 4 |
| C s2, selection | 276 | 3 | 1 | 0 | 1 | 3 |
| C s0, calibration | 232 | 3 | 0 | 0 | 1 | 1 |
| C s1, calibration | 232 | 2 | 1 | 0 | 1 | 1 |
| C s2, calibration | 231 | 2 | 1 | 1 | 1 | 1 |

- **Lost at the temporal readout.** 2 to 3 clips per seed and half are read correctly by both earlier stages and
  wrongly by the window head (F+ J+ T−).
- **Lost before the temporal module.** 0 to 2 clips per seed and half are lost at the frame path (F+ J−). The window
  head recovers all of them except one calibration clip in seed 2.
- **Fixed by the frame head.** It fixes one clip the frozen E1 got wrong in seed 2 of the selection half and in every
  calibration seed (F− J+).
- **Persistent errors.** The clips wrong at every stage (F− J− T−) are 3 to 4 on the selection half and 1 on the
  calibration half. They include the SC4C recording read as PLAX.

## 4. Per sample, the window head is the most accurate path

The target-sample error is the share of window target samples read wrongly, over every validation sequence, native
and constructed. The figure in brackets weights each recording once. F and J are the same in their dense and grid
form at these samples.

| Model | F (sel / cal) | J (sel / cal) | T (sel / cal) |
|---|---|---|---|
| C s0 | 7.0% (4.3%) / 5.3% (4.5%) | 6.7% (4.2%) / 4.7% (3.9%) | 5.2% (2.7%) / 2.3% (1.6%) |
| C s1 | same F | 6.7% (4.2%) / 4.2% (3.5%) | 5.0% (2.4%) / 1.8% (1.3%) |
| C s2 | same F | 6.7% (4.5%) / 4.9% (4.3%) | 4.8% (2.6%) / 2.3% (1.8%) |
| S, three seeds | same F | 6.7–7.3% / 4.4–4.6% | 6.1–6.9% / 2.6–2.9% |
| N, three seeds | same F | 6.5–6.7% / 4.7–5.0% | 7.4–9.4% / 3.7–6.5% |

- **The C window head makes the fewest target-sample errors.** This holds in single-recording windows and, for C, in
  mixed-view windows too.
- **N's window head is worse than its frame head in mixed-view windows.** Its error there is 12% to 26%.
- **How the clips are still lost.** The window head removes many scattered frame errors, but in a few A4C/A5C clips
  it turns the whole clip to the other view. A native clip goes by the majority of its samples, so those few clips
  flip even though the sample error falls.

## 5. Supplementary: one archived E1 + B3 setting on every path

All paths were decoded with the frozen E1 + B3 setting (half-window 9, threshold 0.2246, NMS 5), with no new
quantiles or selection. This is a diagnostic and not a fair ranking, because the threshold was set on E1's
probabilities. Its full table is in `report_tables.md`.

| Path | Segment F1@50 (sel / cal) | Native macro-F1 (sel / cal) |
|---|---|---|
| F-dense (= E1 + B3) | 0.894 / 0.959 | 0.978 / 0.986 |
| F-grid | 0.865 / 0.928 | 0.966 / 0.977 |
| C, J-dense, three seeds | 0.869–0.891 / 0.957–0.964 | 0.920–0.948 / 0.975–0.978 |
| C, T-window, three seeds | 0.854–0.863 / 0.968–0.973 | 0.922–0.959 / 0.975–0.978 |
| S, T-window, three seeds | 0.236–0.266 / 0.222–0.260 | 0.920–0.930 / 0.978–0.986 |

- **S's window outputs fail with this setting.** They are soft frame-share estimates, so the JS scores rarely pass
  E1's threshold and boundary recall falls to 0.07–0.12.
- **The tuned rows from the last round stay the reference** for these models' segmentation (`summary.json`,
  `history_t_window_b3`).

## 6. Recommendation (no training was started)

For the primary C models, the most consistent clip-level loss sits at the temporal readout. In every seed and half it
breaks 2 to 3 clips that the frozen E1 and the trained frame head both read correctly. It also recovers most of the
frame path's own losses, and it gives the lowest target-sample error. This fits the plan's second branch.

- **The next model change to test.** Let the target frame's own frame-head logits enter the final output directly,
  and let the temporal module learn only a correction. It needs the current CLS readout as its control, with the same
  encoder, windows and budget. Residual connections are not new. Nothing here shows that the change would keep the
  clips after training, or that it would reach the final goal.
- **Not adopted alongside it.** The frame path regressed clearly in some N and S models on the selection half (S s1,
  S s2, N s2: 6 to 7 more wrong clips, mostly A4C). A frozen-E1 against joint-fine-tuning comparison would address
  that path. For C the frame-path losses are 0 to 2 clips and mostly recovered, so it is not the first candidate.
  Following the plan, the two changes are not stacked.

## 7. What this does not show

- **Large-sample effects.** Every count here is a handful of clips. On the selection half there are 11 A5C clips, and
  most native intervals touch or include zero.
- **That the frame path is the ViT alone,** or that the temporal readout is attention alone. Each stage changes
  several parts at once.
- **Independent cases.** The two validation halves were used in earlier development. The constructed streams reuse
  recordings, and the three seeds share one validation set.

## Cost and files

- **Inference.** About 82 s per model (feature extraction about 70 s), 12.5 minutes for all nine.
- `report_tables.md` holds every path row, comparison with its interval, eight-cell table, target-sample error and
  supplementary row.
- `summary.json` holds the same results without recording ids.
- Per-recording ids and per-sample predictions stay in the server directory `runs/center_recognition_diagnostic_v1`.
