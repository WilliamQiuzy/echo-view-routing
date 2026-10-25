# A33: real intermediate frames in A08's long support, a single-source-seed screen (2026-10-25)

**Question.** A08 sees 16 samples at 10 Hz over about 1.5 s. Keep that support and those samples, and double the input
to 32 slots. Do real intermediate raw frames (I32) give a recognition gain over repeated frames (R32) that transfers to
held-out sessions, while full segmentation holds?
- **Matched comparison.** I32 − R32 is the main comparison: same 32 slots, model, initial weights, batches, optimiser and
  compute; only the odd-slot content differs. R32 and I32 against Base16 (A08 + B3) and against their own untrained
  starts (Init) are background.
- **Scope.** One source seed (A08 all_s0) and A31's two session folds, so four models. This is a mechanism screen, not
  a multi-seed result. No development, check, test or Pediatric data was read, and no label changed.
- **Data limits.** Training data only. A08 trained on these recordings, so held-out sessions test transfer of the
  correction branch, not generalisation of the whole system.
- **Relation to A22** (9742eb2 / ef98f1f). A22 inserted real frames too, but shrank the support to 0.7 s and hurt full
  segmentation. A33 keeps the full 1.5 s support, never touches the B3 cuts (the branch only relabels Base runs), and
  uses no KL.
- Configuration `configs/experiments/video_long_support_density_recognition_v1.yaml` (c1dd857, committed before any
  output); trainer 12a1f1e. Task sheet `claude_prompts/2026-10-25-long-support-density-recognition-pilot.md`.

## Inputs

**Windows.** Every A08 request c keeps its 16 sample positions a = window_frames(c, N, 16, 16), with the same end shift
and short-sequence repeats.
- **R32.** Slots 2i and 2i+1 both show sample a_i.
- **I32.** The even slots and slot 31 are R32's. Odd slot 2i+1 shows the raw frame floor((l + r) / 2) between the frames
  of a_i and a_{i+1}, but only when both samples lie in the same fragment of the stream (one source, one gamma) and a
  raw frame lies strictly between them. Otherwise the left frame repeats.
  - No frame comes from outside a fragment the stream contains, and nothing is averaged or synthesised.
  - This is stricter than A22, which could take the next frame of a source past a fragment end.
- **Marker.** A22's planned-time rule: slot coordinates are 2a_i (even), a_i + a_{i+1} (odd) and 2a_15 (slot 31), and
  the marker is the middle slot whose coordinate is 2c. It is the same in both arms.
- **Unchanged.** A08's frame loader (gamma, crop, normalisation), support end points and targets. Outputs map back to
  the 10 Hz samples by nearest centre.

**Source and plan audit** (`audit.json`).
- All 2,476 recordings of both pools have the original video at 30/1 fps (time base 1/15360). Each has a cached 10 Hz
  plan with a step of exactly 3 raw frames, and its frame count matches the extracted frames. There were no failures.
- Over 24,585 / 25,774 windows (fold 0 / fold 1):
  - I32 inserts 14.6 real frames per window on average, out of 15 possible odd slots. Every window gets at least one.
  - The odd slot repeats its left frame at fragment boundaries (9,467 / 9,606 slots) and at repeated positions (518 /
    413). Gamma changes coincide with fragment boundaries. "No frame between" never occurs, since the step is 3.
  - I32 windows hold 30.6 distinct frames on average. They span the same ~1.5 s as R32: 32 slots mean neither 32
    distinct frames nor an even 20 Hz.
  - The marker is on an even slot in 23,763 / 24,939 windows. The rest are sequence-end or repeated targets, where the
    rule puts it on slot 31 or another tied slot.

## Model

- **Adaptation.** torchvision's MViTv2-S at temporal_size 32 differs from the 16-frame build only in the 16
  `encoder.blocks.*.attn.rel_pos_t` tables (15 → 31 rows).
  - They were resized with torchvision's own runtime rule (linear, align_corners = False). Every other tensor was
    copied bit for bit, and the state was loaded strictly. The changed names and sha256 values are in `checks.json`.
  - The adapted model's outputs equal the 16-frame model run on 32 frames through torchvision's runtime resize, with a
    difference of 0.0.
- **Branch.** It trains the classifier, block 15, the final norm and the 512-d projection (7,499,525 nominal; block
  15's rel_pos_t now has 31 rows), in eval mode. Four tensors get structurally zero gradient through the class-token
  readout, as in A30 (pool_q, rel_pos_h/w/t).
- **Starts.** Both arms start from bit-identical weights. Init_R32 is not assumed equal to Base16: the token grid at
  block 15 is (16, 7, 7), 785 tokens against 393, and the relative-position tables changed.
- **Cost.** A 32-slot forward costs about twice A08's; R32 and I32 cost the same.

## Checks (`checks.json`, all passed)

- **Base16 reproduction.**
  - A08's 16-slot forward on 64 sampled windows equals A31's cached logits, with a difference of 0.0 and no argmax
    change.
  - Base runs and full-sequence scores equal A31's.
- **Model consistency.** Prefix + suffix equals the full 32-slot forward (0.0) for both arms and both folds, and the
  token grid was read from a real forward.
- **Objective and gradients.**
  - Pooled CE equals NumPy within 4e-16.
  - Gradients from 8- and 4-interval microbatches agree within 6e-8.
  - The optimizer binds exactly the trainable tensors, updates are real, the frozen prefix and A08 stay unchanged, and
    resume is exact.
- **Forced equality.** Fed R32's inputs, the I32 path reproduces R32's loss and update exactly. With their real inputs
  the arms differ (token difference up to 48; first loss 0.254 against 0.117).
- **Schedule.** It was fixed in the check stage before any formal update: 900 updates (warm-up 90) under a 3.65 h
  projection against the 4 h cap.

**Training.**
- **Recipe.** Each model ran 900 updates on its fit fold: A31's seed-0 U plans (16 T + 16 P runs per update, all
  windows), pooled-probability CE, AdamW (head 1e-3, representation 1e-4, weight decay 1e-4), warm-up 90 then cosine,
  clip 1, final weights only.
- **Weights.** All four started from bit-identical adapted weights. The R32 weights were committed in 794044e and the
  I32 weights in f6d0e94, each before that arm's held-out readout. The single analysis ran after all four readouts.
- **Engineering stop (dated note).**
  - The first R32_f0 launch ran at 4.8 s per update with an idle GPU. The cause was the worker-to-main copy of about
    0.6 GB per update.
  - It was stopped at about update 100. Only training loss and throughput had been seen, and its partial weights were
    never saved.
  - The data path was replaced by threaded positioned reads with in-process prefetch (9e472d0). It reads the same bytes
    as the memory map (checked) and does the same arithmetic. Update 90's loss matched the stopped run exactly.
  - The run then went at 1.0 s per update. The stopped run is booked as a failed 0.14 GPU h.
  - The 900-update schedule was kept. The fallback was not needed because the re-projection stayed under the cap.

## Results: held-out P runs and complete sequences

**All held-out P runs, out-of-fold over both folds** (6,788 runs; Base wrong 254, correct A5C runs 1,150).

| Model | Fixed / broken / net | A5C read as A4C fixed (of 103) | A4C read as A5C fixed (of 135) | Correct A5C kept |
|---|---|---|---|---|
| Base16 (A08 + B3) | 0 / 0 / 0 | 0 | 0 | 1,150 |
| Init_R32 (adapted, untrained) | 62 / 167 / −105 | 0 | 58 | 1,053 |
| Init_I32 (adapted, untrained) | 26 / 12 / +14 | 2 | 21 | 1,144 |
| R32 (trained) | 63 / 88 / **−25** | 2 | 61 | 1,087 |
| I32 (trained) | 64 / 84 / **−20** | 7 | 56 | 1,079 |

**Per held-out fold** (fold 1 / fold 0).

| Model | Net fixed | A32 S1 fixed | A32 S2 kept | Segment F1@50 (multi) | Native macro-F1 | Boundary P / R | FP / short FP | FN / missed short true |
|---|---|---|---|---|---|---|---|---|
| Base16 | 0 / 0 | 0/69; 0/34 | 157/157; 84/84 | 0.9499 / 0.9639 | 0.9758 / 0.9776 | 0.942 / 0.968; 0.956 / 0.968 | 156 / 105; 106 / 73 | 102 / 35; 81 / 41 |
| Init_R32 | −53 / −52 | 0; 0 | 120; 57 | 0.9346 / 0.9455 | 0.9739 / 0.9767 | 0.951 / 0.935; 0.969 / 0.933 | 149 / 80; 102 / 41 | 182 / 89; 175 / 101 |
| Init_I32 | +7 / +7 | 2; 0 | 152; 84 | 0.9510 / 0.9650 | 0.9810 / 0.9832 | 0.944 / 0.966; 0.957 / 0.966 | 149 / 98; 99 / 67 | 103 / 37; 82 / 44 |
| R32 | −14 / −11 | 0; 2 | 117; 70 | 0.9486 / 0.9613 | 0.9762 / 0.9732 | 0.957 / 0.959; 0.965 / 0.956 | 133 / 68; 90 / 48 | 129 / 49; 109 / 57 |
| I32 | −12 / −8 | 3; 4 | 117; 71 | 0.9491 / 0.9626 | 0.9707 / 0.9734 | 0.960 / 0.957; 0.963 / 0.964 | 126 / 64; 97 / 61 | 133 / 51; 96 / 51 |

**I32 − R32, the main comparison** (fold 1 / fold 0).

| Quantity | Fold 1 | Fold 0 |
|---|---|---|
| Net fixed runs | +2 | +3 |
| Segment F1@50 (multi) | +0.0005 | +0.0013 |
| Native macro-F1 | −0.0056 | +0.0002 |
| Correct A5C kept | −9 | +1 |
| A5C read as A4C fixed | +3 | +2 |
| A4C read as A5C fixed | +5 | −10 |
| Missed short true segments | +2 | −6 |
| Short false positives | −4 | +13 |

- **Native runs** (one recording each, so one session). I32 − R32 run accuracy is +0.0013, with a descriptive 95%
  session interval of [−0.0055, +0.0086] over 384 sessions.
- **Constructed streams.** They share sources, so they get point estimates only.
- **Breakdowns** (`report.json`). Pure / mixed and short / long runs, plus pure / mixed window accuracy, show no part
  where I32 clearly separates from R32. Higher window accuracy is not counted as success.

## Answers

1. **No substantive transferable gain from real intermediate frames was established.**
   - After training, I32 − R32 is +2 / +3 net runs, about 0.06–0.09% of each fold, with segment F1 +0.0005 / +0.0013.
   - Native macro-F1 and correct-A5C retention move both ways.
   - The protocol's seed-2 rule needs, in both folds, ≥ +10 net runs, F1 not lower, missed short ≤ +3 and correct A5C
     ≥ −5. It is not met: net is only +2 / +3, and fold 1 loses 9 correct A5C runs.
2. **The screen ends; the configuration is not advanced and seed 2 is not added.**
   - The pre-registered close trigger did not fire, because the small differences are positive in both folds on net runs
     and segment F1, and the pattern is not "A4C only".
   - The advancement rule was not met either.
3. **Both trained branches stay below Base16.**
   - OOF net is −25 (R32) and −20 (I32).
   - Segment F1 is lower than Base in all four cells. Native macro-F1 is lower in three of the four, including both I32
     cells.
   - As in A31, their fixes are mostly A4C read as A5C and their breaks are correct A5C runs (correct A5C kept 1,079–1,087
     of 1,150). So the clip-code recognition training keeps moving the A4C/A5C boundary.
4. **Background on the starts** (descriptive, not used to choose anything).
   - With duplicated frames, the untrained adapted model is well below Base: Init_R32 is −105 net, with 54–60 more
     missed short segments per fold.
   - With real intermediate frames it stays near Base: Init_I32 is +14 net, segment F1 +0.0011 per fold and native
     macro-F1 +0.005–0.006.
   - Training then lifts R32 from its poor start but lowers I32 below its own start.
   - These are training recordings that A08 itself learned from, and the comparison was not pre-registered. It suggests
     only that repeated frames are an unnatural 32-slot input. It does not show that Init_I32 beats Base anywhere else.
5. **Attribution.** None of this is attributed to the LVOT, motion or cardiac phase.

## Cost and files

- **GPU 2.78 of 4 h.**
  - Check: 0.04 h.
  - Caches: R32 0.69 h, I32 0.73 h.
  - Training: R32 0.24 / 0.27 h, I32 0.25 / 0.25 h.
  - Timing: 0.02 h. Readouts: 0.13 h.
  - The stopped run: 0.14 h, booked as failed.
- **CPU.** About 1.1 of 3 h, including the agent's active wall clock with waiting excluded.
- **Records.**
  - `audit.json`, `checks.json`, `weights_R32.json`, `weights_I32.json`, `report.json`.
  - Code: `scripts/train_long_support_density.py`, `echo_routing/temporal/long_support.py`, `tests/test_long_support.py`.
- **Server only.**
  - Plans, held-out logits, the Init logits, the trained weights (`runs/video_long_support_density_recognition_v1/train/`)
    and all identifiers.
  - Both prefix caches (121 GB each) were deleted after their readouts.
