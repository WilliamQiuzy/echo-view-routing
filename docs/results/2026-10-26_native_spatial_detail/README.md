# A36: native 320 spatial detail against matched reduced detail on the frozen A08 encoder (2026-10-26)

**Question.** Keep the temporal support, frames, complete field of view, frozen A08 weights, input size, objective and
compute the same. Does native 320 × 240 spatial detail (H) give more transferable A4C/A5C recognition evidence than
the same image reduced to long side 224 and enlarged back (L)?
- **Scope.** One source seed (A08 all_s0) on A31's two session folds: four residual heads (H and L per fit fold).
  - The A08 + B3 path is fixed. Its 224 inputs, logits q, 512-d embeddings e224, cuts and settings are read only.
  - No B3, threshold, router or calibration change. No development, check, test or Pediatric data was read.
- **Data limits.** A08 trained on these recordings. The heads never saw the held-out sessions, but the whole model has
  seen them, so this is a transfer screen of the head, not an independent holdout. Labels are EV9V clip codes, not
  expert LVOT judgements.
- **History.**
  - R15 compared the old E1 image encoder on the legacy crop against fan224, a change of geometry and field of view.
  - A21 added a whole-frame auxiliary view fitted to 224 with the encoder trained; it used the same residual head form.
  - A28, A29 and A31 adapted the representation or head on the original 224 input. A33 changed temporal slots. A34
    and A35 tried clip-code bags and a frozen external recogniser.
  - A36 is the first matched-detail test. Canvas, field of view, encoder and head are shared, and only the spatial
    detail differs.
- **Records.** Configuration `configs/experiments/video_native_spatial_detail_v1.yaml` (3cadf73, committed before any
  computation). Code 63649ef. Head sha256 values c978579, committed before the held-out readout. Task sheet
  `claude_prompts/2026-10-26-native-spatial-detail-recognition.md`.

## Design

- **Inputs.** Each frame follows A08's decode and gamma rule. It is then resized, keeping its aspect, to long side 320.
  The 320 × 240 originals are unchanged. The result is centred on a 320 × 320 canvas filled with RGB(29, 28, 29).
  - H is that image itself.
  - L is the same image resized with PIL bilinear to long side 224 and back to H's size, on the same canvas.
  - Marker, slot times, gamma and normalisation are identical. Only the detail differs: over the canvas, L differs
    from H by 2.8 grey levels on average.
- **Encoder.** The A08 TargetMViT is reused frozen. Only its token-grid bookkeeping changes to 80 × 80, so
  torchvision resizes the relative-position tables at run time. No table is trained or replaced.
- **Head.** z = q + g(concat(e224, e320)), with g = Linear(1024, 32) → GELU → Linear(32, 5) (32,965 parameters).
  - The last layer starts at zero, so the head starts exactly at Base. The first layer comes from one seeded
    generator per fit fold (20261062 + fold) and is bit-identical for H and L.
- **Training.** Only g is trained, on A31's uniform plans: 1,350 updates of 16 T and 16 P runs each, pooled-probability
  CE, no KL. AdamW lr 1e-3, weight decay 1e-4, warm-up 135 then cosine, clip 1. Final weights only, CPU single thread.
- **Readout.** The unchanged Base runs are relabelled by the argmax of the multiplicity-weighted mean probability, and
  equal neighbours merge. Complete sequences are scored with the original evaluator.

**Checks** (`checks.json`, all passed on the first run).
- **Inputs.** On 64 real training frames, H's content equals the gamma-applied original bit for bit, and L differs.
  The canvas fill is identical in both.
- **224 parity.** A08 reproduces A31's cached logits and embeddings exactly (difference 0.0).
- **320 grid.** The stem gives 8 × 80 × 80 tokens and the last block 8 × 10 × 10.
  - The runtime path equals a model with all 32 relative-position tables materialised by the same rule (difference
    0.0).
  - Repeat inference is identical, and splitting a batch changes outputs by at most 1.2e-6. H and L embeddings
    differ (up to 3.3).
- **Init = Base.** In both folds the untrained head returns q exactly. Every P run and complete sequence equals Base,
  and the rows match A32's Base rows.
- **Training mechanics.**
  - Forced H = L features give identical losses and weights.
  - The first-layer gradient is zero at step 1 and nonzero at step 2.
  - The optimiser binds exactly, and 3 + 3 resumed updates equal 6.
  - The folds share no session, recording or window.

**Features and training.**
- **Features.** Feature extraction covered all 50,359 request windows for H and L. It took 1.30 GPU h (44 ms per window
  per arm) and 197 MB of disk.
- **Training.** Each head ran 1,350 real updates in about 3 s. H and L start from the same loss in each fold.
- **Fit fold.** Training CE fell from 0.132 to 0.085 (fold 0) and from 0.140 to 0.098 (fold 1). The heads fixed 21–28
  of the fit fold's 109–145 Base errors and broke 2–4 (`weights_s0.json`).

## Results (held-out folds; OOF = both folds summed)

**All held-out P runs.** Fold 0 has 3,384 runs and 109 Base errors; fold 1 has 3,404 and 145. Correct A5C runs are
561 / 589, and A32's S1 runs (true A5C read as A4C by Base) are 34 / 69.

| Model | Fixed / broken / net (f0; f1; OOF) | A4C read as A5C fixed | S1 fixed | Correct A5C kept | Segment F1@50 (f0 / f1) | Native macro-F1 | Boundary R / P (f0; f1) | FP / short FP | FN / missed short |
|---|---|---|---|---|---|---|---|---|---|
| Base (= Init) | 0 / 0 / 0 | 0 | 0 / 103 | 1,150 / 1,150 | 0.9639 / 0.9499 | 0.9776 / 0.9758 | 0.968 / 0.956; 0.968 / 0.942 | 262 / 178 | 183 / 76 |
| H | 11/6/+5; 25/15/+10; **36/21/+15** | 35 | 1 | 1,133 | 0.9644 / 0.9538 | 0.9852 / 0.9791 | 0.967 / 0.958; 0.967 / 0.950 | 236 / 151 | 185 / 77 |
| L | 11/6/+5; 24/13/+11; **35/19/+16** | 35 | 0 | 1,135 | 0.9648 / 0.9538 | 0.9832 / 0.9791 | 0.967 / 0.958; 0.966 / 0.950 | 235 / 152 |

Boundary values are rounded to three decimals; fold 1 precision is 0.9505 (H) against 0.9500 (L). 184 / 77 |

FP, FN and their short parts are summed over the two folds.

**H − L, the main comparison.**

| Fold | Net runs | Segment dF | Native macro-F1 | Boundary R | FP / short FP | FN / missed short | Correct A5C | S1 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | −0.0004 | +0.0019 | −0.0005 | +1 / 0 | +1 / 0 | 0 | +1 |
| 1 | −1 | 0.0000 | 0.0000 | +0.0005 | 0 / −1 | 0 / 0 | −2 | 0 |

**Breakdowns** (OOF fixed / broken / net).

| Model | Native | Constructed | Pure | Mixed | Short (≤ 16) | Long |
|---|---|---|---|---|---|---|
| H | 12 / 1 / +11 | 24 / 20 / +4 | 28 / 15 / +13 | 8 / 6 / +2 | 20 / 11 / +9 | 16 / 10 / +6 |
| L | 11 / 1 / +10 | 24 / 18 / +6 | 27 / 13 / +14 | 8 / 6 / +2 | 19 / 10 / +9 | 16 / 9 / +7 |

- **Direction of the changes.** 35 of H's 36 fixes are A4C runs that Base read as A5C. 17 of its 21 breaks are correct
  A5C runs moved to A4C. A32's S2 runs kept as A5C fall to 81 of 84 (both arms) and to 144 (H) and 146 (L) of 157.
  L shows the same pattern: 14 of its 19 breaks are correct A5C → A4C.
- **Coverage** (`coverage_s0.json`). H's fixes come from 35 recordings in 32 sessions and its breaks from 27 recordings
  in 25 sessions. L's come from 36 / 33 and 25 / 22. A run counts all its source recordings.
  - This pass ran after the readout on the same committed heads. It first reproduced the readout's fixed and broken
    counts, and no decision depends on it.
- **Against Base.** Both arms gain slightly in both folds: segment dF +0.0005 / +0.0039 for H and +0.0009 / +0.0039
  for L. In fold 1 both have 21 fewer false segments, and short false segments fall by 23 (H) and 22 (L). Missed short
  true segments rise by 1 in total.
  - **Background, not pre-registered.** A31's linear F_U head on e224 alone gave OOF net +13 and mean dF +0.0019 on the
    same pools and seed. A36's heads are about the same size of effect.

**Source-2 rule (fixed before any output): not met.**

| Part | Requirement | Result | Met |
|---|---|---|---|
| Detail | H − L net ≥ +5 per fold; mean dF ≥ +0.003; dF ≥ 0 per fold | 0 / −1; −0.0002; −0.0004 / 0.0000 | No |
| Useful | H − Base net ≥ 0 and dF ≥ 0 per fold; mean dF ≥ +0.003 | +5 / +10; +0.0005 / +0.0039; mean +0.0022 | No (mean) |
| Target | S1 recovered ≥ 11 of 103; correct A5C kept ≥ 98% | 1 / 103; 1,133 / 1,150 (98.5%) | No (S1) |
| Protect | native macro-F1 ≥ −0.003 per fold; missed short ≤ Base + 3; boundary recall ≥ −0.005 per fold | +0.0076 / +0.0033; 77 against 76; −0.0010 / −0.0015 | Yes |

## Answers

1. **No new spatial evidence.**
   - On the frozen A08 encoder, native 320 detail gives the same held-out result as the matched reduced input.
     H − L is 0 / −1 net runs and −0.0004 / 0.0000 in segment F1.
   - Training does not separate them either. Fit-fold CE ends at 0.0846 against 0.0848 and 0.0984 against 0.0992, so
     the head finds no extra fittable signal in e320 from H.
   - The embeddings do change with detail. The head does not turn that change into A4C/A5C evidence.
2. **A small full-segmentation gain over Base, but not from detail.**
   - Both arms raise segment F1 by about +0.002 (mean), mainly through fewer false short segments in fold 1. This is
     below the +0.003 rule.
   - Because L matches H, the gain comes from training the residual head on the shared features, not from native
     detail. It is the same size as A31's F_U head.
3. **The same A4C/A5C trade as A27–A35.** The fixes are A4C runs read as A5C. Only 1 of 103 S1 runs is recovered (H),
   and correct A5C runs are lost.
4. **Closing scope.**
   - Under the fixed rule this configuration ends at source seed 0. There is no seed 2 and no resolution, blur, crop,
     head, loss, learning-rate or encoder variant.
   - The result covers only this frozen A08 320 representation with this residual head. It does not settle
     high-resolution or spatial learning in general, such as an encoder trained at native resolution. No difference
     is attributed to the LVOT.
   - Earlier closing decisions stand. This round does not stop the project. A08 + B3 stays the working reference.
5. **Cost of deployment.** H would need the 224 forward (16.5 ms per window in A31's pass) plus a 320 forward (44.2 ms
   per window). The head itself is negligible on CPU (0.003 ms per window). That is about 3.7 times Base's GPU time for
   no gain over L.

## Cost and files

- **GPU 1.30 of 4 h:** the check (0.008 h) and seed-0 extraction (1.296 h).
- **CPU about 0.5 of 3 h:**
  - four heads, the readout and coverage: under 0.01 h;
  - one failed coverage start: the feature-cache key covered the whole script, and adding the coverage stage changed
    it. It stopped at once and is booked. The rerun checks the cache against the feature record committed with the
    heads;
  - the agent's active wall clock.
  - All of it is in `runs/video_native_spatial_detail_v1/ledger.jsonl` on the server.
- **New disk:** 197 MB of 512-d feature caches. There is no pixel or prefix cache.
- **Records.** `checks.json`, `weights_s0.json` (head manifests with fit-fold monitors and the feature record),
  `readout_s0.json`, `coverage_s0.json`.
- **Code.** `scripts/train_native_spatial_detail.py`, `echo_routing/temporal/spatial_detail.py`,
  `tests/test_spatial_detail.py`.
- **Server only.** Feature caches, trained heads, per-run predictions and all identifiers.
