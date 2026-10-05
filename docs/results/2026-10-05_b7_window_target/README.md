# Does B7's window output report the view at a given time? (2026-10-05)

A video window model gives one output for a whole 1.6 s window. Temporal segmentation then gives that output to the
time points around the window's centre. This check asks whether that use matches what the model answers. It follows
the user's plan of 2026-10-05, steps 1 to 4. Nothing was trained. The definitions were committed before any result
(`configs/experiments/b7_window_target_v1.yaml`, commit 47a4f01). Only the two EV9V validation halves were read. The
test split stayed sealed, no EchoNet-Pediatric input was used and the public labels are unchanged. This is a check of
an open question under correct inputs. It is not claimed as a method, and the old ViT grid was not restored.

## Answers

1. **Was there a valid earlier test?** No. The centre-supervised video models V2c and V4c were trained and evaluated
   on 2026-09-25, while the frame loader still read frames out of time order. No ViT-family model was run after the
   fix. Their results are neither evidence of a gain nor evidence against one.
2. **Does the mismatch exist in the current B7?** Yes, as a description. B7's errors at the window centre concentrate
   in windows that hold two or more views: 15.0% of such windows on the selection half and 11.7% on the calibration
   half, against at most 4.2% elsewhere. Windows that join two recordings of the same view are not worse than windows
   from one recording. B7 cannot locate the centre at all, because it pools its frames without their positions.
3. **Does the neighbouring view cause the errors?** Partly, at the mixes where real errors occur. With 12 of 16
   frames from the centre's view, an other-view context lowers B7's accuracy at the centre by 2.3 points (95%
   interval 1.2 to 3.7) against a same-view context, and 30 of the 44 additional errors name the context's view. That
   meets the rule fixed in advance. With 14 of 16 the drop is about 1 point and the rule is not met. At an even split
   B7 reports whichever view dominates the mix, not the centre.
4. **Is there enough room to justify training?** No. An ideal repair of every wrong mixed-views window would raise
   B7 + B3 from 0.880 to 0.907 on the selection half and from 0.972 to 0.986 on the calibration half. The calibration
   half would clear its target of 0.982, but the selection half would stay far below its target of 0.944. B7's
   native-clip macro-F1 on the selection half (0.957, floor 0.973) is untouched by this repair.
5. **Training comparison.** Not drafted, as the gate requires. Nothing was trained.

## Terms

- **B7** is the EchoViewCLIP stage-1 video model, trained by this project with the official code. It reads 16 frames
  spread over 1.6 s and is applied every 0.5 s along a sequence. Every 10 Hz sample takes the output of the window
  whose centre is nearest.
- The **centre sample** is the sample at a window's centre. Its label is the view that the window's output stands for
  in segmentation.
- A window is **single recording** when all its frames come from one recording. It is **cross-recording, one view**
  when its frames come from two or more recordings that share a label. It is **mixed views** when its frames carry two
  or more labels. Joining recordings is not the same as joining views.
- The **centre-class frames** are the window's frames that carry the centre sample's label, out of 16.
- A **recording** is one labelled EV9V clip and a **session** is the protocol's split unit. Streams reuse recordings,
  so counts are also given by recording and session, and repeated windows of one recording are never counted as
  independent cases.

## 1. Earlier evidence (step 1)

- **The runs.** V2c (factorised ViT-S, run `20260925-085555-vivit_fe_centre-4f2a5c8a-s0`) and V4c (Echo-MViTv2-S,
  `20260925-092542`, `-115626` and `-114340`) trained on mixed two-view clips with the centre frame's label
  (`docs/results/2026-09-25_vit_family/DESIGN.md`).
- **Why they do not count.** At that commit (2e7703c), `list_frame_paths` sorted frame files by name, and the ViT data
  loaders read frames through it. EV9V names frames without zero padding, so both the training clips and the
  evaluation windows used frames out of time order. That README already carries a recomputation notice.
- **No later run.** The server holds 24 ViT-family runs, all from 2026-09-25. No ViT-family run, checkpoint or
  `cache_v2` prediction exists after the fix of 2026-09-30, and the ledger has no later entry. No correct-order
  reproduction exists to cite, so the question is open.
- **What B7 can represent.** In `third_party/EchoViewCLIP/trainers/vificlip.py`, each frame is encoded on its own.
  The frame features are then pooled by a temporal mean and by a softmax attention computed from each frame's own
  feature. Neither pooling uses the frame's position, so B7 gives the same output for any order of the same frames.
  It cannot tell which frame is at the centre and can only respond to the window's mix of frames.

## 2. Where B7 is wrong at the window centre (step 2)

Every B7 window of every validation sequence was classified by its 16 frames. A window is wrong when its output
differs from the label of its centre sample. The recording-averaged error weights each centre recording once.

| Half | Window type | Windows | Wrong | Error | Recording-averaged error | Recordings with an error | Sessions with an error |
|---|---|---:|---:|---:|---:|---:|---:|
| Selection | Single recording | 5,312 | 225 | 4.2% | 2.4% | 15 of 284 | 10 |
| Selection | Cross-recording, one view | 905 | 23 | 2.5% | 0.8% | 6 of 227 | 5 |
| Selection | Mixed views | 1,449 | 218 | 15.0% | 11.2% | 88 of 238 | 32 |
| Calibration | Single recording | 4,783 | 73 | 1.5% | 0.8% | 7 of 237 | 6 |
| Calibration | Cross-recording, one view | 798 | 9 | 1.1% | 0.6% | 4 of 195 | 3 |
| Calibration | Mixed views | 1,440 | 168 | 11.7% | 11.7% | 86 of 222 | 31 |

- **Mixed views carry the errors.** Joining two recordings of one view does not raise the error. Mixing views does,
  in both halves and in many recordings and sessions.
- **The fewer frames of the centre's class, the more errors.** In mixed-views windows the error is 1.8% (calibration)
  and 7.3% (selection, 2.4% when each recording counts once) when 12 to 15 of the 16 frames carry the centre's class.
  It is about 18% with 9 to 11 frames and about 55% at 8 of 16, where the window is evenly split.
- **Most wrong outputs name a view in the window.** Of the wrong mixed-views windows, 159 of 218 (selection) and 152
  of 168 (calibration) name a view that is present among the frames.
- **Class pairs.** A4C centres are most often affected, read as PLAX, A5C, PSAX or SC4C next to those views. A5C
  centres read as A4C come from only 4 recordings on the selection half (44 windows) and 5 on the calibration half.
  On the selection half one SC4C recording that B7 reads as PLAX accounts for 25 wrong mixed windows.
- **Single-recording errors are concentrated.** On the selection half they sit mostly in A5C recordings (123 windows
  from 5 recordings) and in that SC4C recording (33 windows).

These patterns describe where B7 errs. They do not show that the neighbouring view causes the errors, because mixed
windows sit at joins, where other things also change. Step 3 tests that.

## 3. Paired context probe (step 3)

Each of the 474 validation recordings with at least 16 samples served as a target. The other 47 are shorter than
1.6 s. The target's middle 16 samples formed the window. Its centre frames were kept, and the other positions were
filled from a context recording at the same positions. The context was the target itself ("own"), another recording of
the same view from another session ("same view"), or one recording of each other view ("other view"). The context
recordings were drawn with a fixed seed and kept across centre sizes. Every frame was read in the original contrast at
the cached 10 Hz frame indices, and B7 ran with its frozen weights and preprocessing. The labels chose the contexts and
scored the outputs; B7 saw frames only.

- **The probe reproduces B7.** The probe runner gave the cached output for 47 stream windows exactly (largest
  difference 0).
- **Frame order does not matter to B7.** Reversing the frames of 40 windows changed the outputs by at most 4 × 10⁻⁷.
  A context on both sides of the centre therefore stands for the one-sided context of a real join with the same mix of
  frames.

The table pools both halves. Each target gives one same-view pair and four other-view pairs per centre size. The
differences are in points of accuracy, and the intervals resample the targets' sessions.

| Centre frames of 16 | Same-view accuracy | Other-view accuracy | Difference (95% interval) | Additional errors | Naming the context's view | Targets with an additional error |
|---|---:|---:|---|---:|---:|---:|
| 4 | 99.2% | 2.2% | −97.0 (−98.0 to −95.9) | 1,839 | 1,773 | 470 |
| 8 | 99.6% | 47.7% | −51.9 (−55.0 to −49.2) | 984 | 980 | 407 |
| 12 | 99.4% | 97.0% | −2.3 (−3.7 to −1.2) | 44 | 30 | 25 |
| 14 | 98.9% | 98.0% | −0.9 (−2.2 to −0.05) | 18 | 6 | 6 |

- **A quarter of the window is enough to pull the output.** With 12 centre frames, the other view's 4 frames cost 2.3
  points of accuracy. The additional errors come from 25 targets in 18 sessions, and 30 of the 44 name the context's
  view. The rule fixed in advance counts this as support.
- **Two frames are not.** With 14 centre frames the drop is about 1 point, and only 6 of its 18 additional errors name
  the context's view. The rule is not met.
- **At an even split B7 follows the dominant view, not the centre.** A4C centres among PLAX frames are read as PLAX for
  124 of 127 targets. PSAX centres among PLAX frames keep their view for 34%. A5C centres among A4C frames keep it for
  16 of 23.
- **A4C and A5C.** With 12 centre frames, A5C context lowers A4C centres from 98.4% to 88.2% (127 targets), and 15 of
  the outputs name A5C. A4C context lowers A5C centres from 100% to 91.3% (23 targets).
- **Joining recordings alone does no harm.** A same-view context is not worse than the target's own frames (same view
  minus own, +0.6 to +1.3 points).

This answers the caution of step 2. The centre frames and the join were the same in both conditions, and only the
context's view changed. So the context's view does change B7's output at the centre once it fills a quarter of the
window.

## 4. How much the full system loses (step 4)

The ideal repair (R1) gives every wrong mixed-views window the one-hot of its centre label. Every sample keeps its
nearest window. It is an upper bound for a model that always reported the view at the centre in mixed windows.
Native clips have no mixed-views windows, so R1 leaves them unchanged. A reference repair (R2) does the same for every
wrong window of any type, native clips included. The repaired outputs were decoded by B3 with the two existing B7
settings, the frozen one and the one chosen by the common rule of the last round. Nothing was re-tuned.

| Decoding | Rows | Selection segment F1@50 | Calibration segment F1@50 | Boundary recall (sel / cal) | Native macro-F1 (sel / cal) |
|---|---|---:|---:|---|---|
| Frozen setting | as stored | 0.8798 | 0.9721 | 0.955 / 0.975 | 0.957 / 0.993 |
| Frozen setting | R1 | 0.9070 | 0.9857 | 0.997 / 1.000 | 0.957 / 0.993 |
| Frozen setting | R2 (reference) | 1.0000 | 1.0000 | 0.997 / 1.000 | 1.000 / 1.000 |
| Common-rule setting | as stored | 0.8658 | 0.9732 | 0.961 / 0.977 | 0.957 / 0.986 |
| Common-rule setting | R1 | 0.8924 | 0.9857 | 1.000 / 1.000 | 0.957 / 0.986 |
| Common-rule setting | R2 (reference) | 1.0000 | 1.0000 | 0.997 / 1.000 | 1.000 / 1.000 |

- **What this error type costs.** R1 adds 2.7 points on the selection half and 1.3 to 1.4 points on the calibration
  half. With the frozen setting it fixes 26 units and newly breaks 2 on the selection half. The fixed units come from
  17 recordings, and 14 of them involve the 5 most affected recordings. On the calibration half it fixes 10 units from
  11 recordings and breaks none.
- **How widely it is spread.** The repaired windows are 218 on the selection half (88 recordings, 32 sessions) and
  168 on the calibration half (86 recordings, 31 sessions).
- **The gate.** The final goal needs, on both halves, segment F1@50 at least 0.01 above the strongest previous system
  (0.944 and 0.982), boundary recall of at least 0.95, and native macro-F1 of at least 0.973 and 0.980. R1 meets the
  calibration half with both settings. On the selection half it misses the segment target (0.907 and 0.892 against
  0.944) and the native floor (0.957 against 0.973). The gate is not met, so no training comparison is drafted.
- **Where B7's selection-half loss lies instead.** On that half, the gap between R1 (0.907) and R2 (1.000) comes from
  wrong windows that do not mix views. Single-recording windows hold 225 errors in 15 recordings, among them the SC4C
  recording read as PLAX and several A5C recordings. A model that reports the centre's view cannot remove those.

## 5. What this check does not show

- **That centre supervision cannot help.** The gate concerns B7 and the final goal. The calibration half shows that
  mixed-window errors cost B7 + B3 about 1.4 points there, which an ideal fix would recover.
- **What a model with temporal positions would do.** B7 cannot represent the centre, and no such model was trained
  here. A future comparison would need an architecture that sees frame positions, the same encoder and mixed windows
  for its control, and the same budget, as the plan says.
- **Held-out performance.** Both validation halves were used in earlier development. The B7 checkpoint was selected
  on EchoViewCLIP's historical validation split, which contains both halves. Windows come from constructed streams
  that reuse recordings and are not real scans.

## Files

- `census.md` and `census.json` hold the step-2 tables. The per-window table (`windows.csv`, 2.3 MB) stays in the
  server run `runs/20261005-194354-b7_window_target_census-fabf5ffb-s0`.
- `probe.md` and `probe.json` hold the step-3 results, including every target and context class pair. The
  per-window table (`probe_windows.csv`) and the probe outputs stay in the server run
  `runs/20261005-195504-b7_window_target_probe-fabf5ffb-s0` and in `runs/b7_window_target/probe/`.
- `room.md` and `room.json` hold the step-4 metrics, the affected recordings by class pair, the unit changes and the
  gate (server run `runs/20261005-194458-b7_window_target_room-fabf5ffb-s0`).
