# A37: wider temporal evidence on the frozen A08 encoder, doubled time offsets against the original span (2026-10-26)

**Question.** Keep the cuts and target times. Add one more frozen video representation that reads real frames further
from the target. Does it correct A5C runs read as A4C, keep correct A5C, and carry over to full segmentation?
- **The one new input.** The 16 slots keep A08's marker slot, and every time offset from the target is doubled.
  - S = window_frames(c, N, 16, 16), A08's own 16 sample positions (about 1.5 s at 10 Hz).
  - W = clip(c + 2(S − c), 0, N − 1), which covers about 3.0 s at 5 Hz.
  - Span and sampling density change together. This round does not separate them, and it claims no duration,
    motion, cardiac-phase or anatomical mechanism.
- **Scope.** One source seed (A08 all_s0) on A31's two session folds: four residual heads (S and W per fit fold).
  - The A08 + B3 path is fixed. Its 224 inputs, logits q, 512-d embeddings e224, cuts and settings are read only.
  - No B3, threshold, router or calibration change. No development, check, test or Pediatric data was read.
- **Data limits.** A08 trained on these recordings, so this is a transfer screen of a new head across A31's session
  folds, not an independent holdout of the whole model. Labels are EV9V clip codes, not expert LVOT judgements.
- **History.**
  - V01 showed that frames in the same window help, and that mixed-view context can hurt the target.
  - A22 (about 0.7 s) helped mixed windows and short true segments but made many short false segments.
  - A33 kept about 1.5 s with denser slots. A34 trained bag heads on 1.5 s window representations.
  - A36 changed spatial detail at the same time support.
  - A37 is the first input that reaches beyond A08's 1.5 s window, with the same weights, slots and head.
- **Records.** Configuration `configs/experiments/video_wider_temporal_evidence_v1.yaml` (d2cf83e, committed before
  any computation). Code 16da40c. Head sha256 values f55c1ed, committed before the held-out readout. Task sheet
  `claude_prompts/2026-10-26-wider-temporal-evidence-recognition.md`.

## Design

- **Windows.** Both arms use the complete constructed sequences of A31's pools, the original stride-5 centres and the
  nearest-centre output map. Frames come from the sequence's own refs at the given positions, with A08's gamma and
  256 → 224 crop. There are no interpolated pixels, no new raw time points and no changed centres.
- **Arms.**
  - S reads e224 twice: z_S = q + g_S(concat(e224, e224)). This matches the capacity and residual structure of W.
  - W reads a new embedding from the same frozen A08: z_W = q + g_W(concat(e224, eW)).
  - g = Linear(1024, 32) → GELU → Linear(32, 5), 32,965 parameters. The last layer starts at zero. The first layer
    is seeded 20261063 + fold and is bit-identical for S and W.
- **Training.** Only g is trained, on A31's uniform plans: 1,350 updates of 16 T and 16 P runs each, pooled-probability
  CE, no KL. AdamW lr 1e-3, weight decay 1e-4, warm-up 135 then cosine, clip 1. Final weights only, CPU single thread.
- **Readout.** The unchanged Base runs are relabelled by the argmax of the multiplicity-weighted mean probability, and
  equal neighbours merge. Complete sequences are scored with the original evaluator.

**Input audit** (`audit.json`; index only, before any feature, no labels or model output).
- **Span.** W spans 28.6 samples on average against S's 15.0. 70–72% of windows reach the full 30 samples. 0.6–0.8%
  have the same span as S.
- **New frames.** 47% of W's slots lie outside S's sample range, and 99% of windows read at least one frame that S
  does not read. No window equals S.
- **Edges.** 4.6–4.9% of W slots are clamped at a sequence edge, and 27–28% of windows repeat a position.
- **Short sequences.** 53 / 45 sequences have N < 16 and 274 / 264 have N < 31. They hold 1,275 / 1,272 requests,
  and every one is kept.

**Checks** (`checks.json`, all passed on the first run).
- **Mapping.** An independent loop over all 50,359 windows reproduces S, W and the marker slot with 0 mismatches.
  - At the marker slot, W's ref equals S's target ref. Positions are ordered, in range and inside the sequence.
  - Covered cases: N < 16 (148 / 128 windows), N = 16 (56 / 16), long sequences, the last centre, repeated positions
    (6,838 / 6,899) and constructed joins (13,553 / 14,172).
  - The first request centre is sample 2 under A08's rule, and W clamps there; no window centres on sample 0.
  - No window holds one source under two gammas.
- **Features.**
  - Online S logits and embeddings equal A31's cache exactly. Feeding S through the W path also gives e224 exactly.
  - W's normal path equals an explicitly stacked forward of the same frame plan, with bit-equal input and output.
  - Repeat inference is identical, a batch split changes outputs by at most 9.5e-7, and TF32 is off.
- **Init = Base.** The untrained head returns q exactly. Every P run and complete sequence equals Base, and the rows
  match A32's Base rows in both folds.
- **Training mechanics.**
  - Forced eW = e224 gives identical losses and weights.
  - The first-layer gradient is zero at step 1 and nonzero at step 2.
  - The optimiser binds exactly, 3 + 3 resumed updates equal 6, the source weights are unchanged, and the folds share
    nothing.

**Features and training.**
- **Features.** Extraction covered all 50,359 request windows. It took 0.26 GPU h (16 ms per window) and 101 MB.
- **Training.** Each head ran 1,350 real updates in about 3 s.
- **Fit folds.** W fits more than S.
  - Fit-fold CE fell to 0.0774 against 0.0871 (fold 0) and 0.0900 against 0.1001 (fold 1).
  - W fixed 34 and 51 of the fit folds' 109 and 145 Base errors (breaking 2 and 11). S fixed 22 and 20 (breaking 2
    and 4).

## Results (held-out folds; OOF = both folds summed)

**All held-out P runs.** Fold 0 has 3,384 runs and 109 Base errors; fold 1 has 3,404 and 145. Correct A5C runs are
561 / 589, and A32's S1 runs (true A5C read as A4C by Base) are 34 / 69.

| Model | Fixed / broken / net (f0; f1; OOF) | A4C read as A5C fixed | S1 fixed | Correct A5C kept | Segment F1@50 (f0 / f1) | Native macro-F1 | Boundary R (f0 / f1) | Boundary P (f0 / f1) | FP / short FP | FN / missed short |
|---|---|---|---|---|---|---|---|---|---|---|
| Base (= Init) | 0 / 0 / 0 | 0 | 0 / 103 | 1,150 / 1,150 | 0.9639 / 0.9499 | 0.9776 / 0.9758 | 0.9676 / 0.9681 | 0.9556 / 0.9420 | 262 / 178 | 183 / 76 |
| S | 11/4/+7; 17/12/+5; **28/16/+12** | 27 | 0 | 1,139 | 0.9650 / 0.9511 | 0.9852 / 0.9810 | 0.9671 / 0.9651 | 0.9570 / 0.9451 | 247 / 165 | 185 / 79 |
| W | 24/15/+9; 28/25/+3; **52/40/+12** | 44 | 8 | 1,118 | 0.9651 / 0.9513 | 0.9871 / 0.9789 | 0.9626 / 0.9630 | 0.9645 / 0.9518 | 224 / 137 | 205 / 93 |

FP, FN and their short parts are summed over the two folds.

**W − S, the main comparison.**

| Fold | Net runs | Segment dF | Native macro-F1 | Boundary R | Boundary P | FP / short FP | FN / missed short |
|---|---|---|---|---|---|---|---|
| 0 | +2 | +0.0000 | +0.0020 | −0.0046 | +0.0076 | −13 / −15 | +12 / +9 |
| 1 | −2 | +0.0002 | −0.0020 | −0.0021 | +0.0066 | −10 / −13 | +8 / +5 |

**Breakdowns** (OOF fixed / broken / net).

| Model | Native | Constructed | Pure | Mixed | Short (≤ 16) | Long |
|---|---|---|---|---|---|---|
| S | 12 / 0 / +12 | 16 / 16 / 0 | 23 / 9 / +14 | 5 / 7 / −2 | 12 / 10 / +2 | 16 / 6 / +10 |
| W | 19 / 2 / +17 | 33 / 38 / −5 | 43 / 24 / +19 | 9 / 16 / −7 | 31 / 29 / +2 | 21 / 11 / +10 |

- **Direction of the changes.** W moves more runs in both A4C/A5C directions.
  - It fixes 44 A4C runs read as A5C and 8 S1 runs. S fixes 27 and 0.
  - It breaks 32 correct A5C runs (to A4C) and 8 correct A4C runs (to A5C). S breaks 9 correct A5C runs to A4C.
  - Dated note 2026-10-26. S loses 11 correct A5C runs in all, as the table's 1,139 shows: 9 to A4C, 1 to PLAX and 1 to
    PSAX. The 9 above counts the A4C destination only (post-A37 diagnostic, `docs/results/2026-10-26_post_a37_context_evidence_audit/`).
  - A32's S2 runs kept as A5C fall to 75 / 84 and 137 / 157 under W, against 83 / 84 and 149 / 157 under S.
- **Coverage.** W's fixes come from 54 recordings in 49 sessions and its breaks from 48 / 43. S's come from 28 / 27
  and 23 / 22. A run counts all its source recordings.
- **Against Base.**
  - Segment dF is +0.0012 / +0.0015 for W and +0.0011 / +0.0012 for S.
  - W has 38 fewer false segments (41 fewer short ones). It has 22 more missed true segments (17 more short ones).
  - Boundary recall falls by 0.0051 in both folds, and boundary precision rises by 0.009–0.010.
- **What the extra context contains** (reference-based description after the weights were frozen; no rule or
  selection used it).
  - Slots whose label differs from the target's make up 6.4–6.8% of S windows against 12.9–13.6% of W windows.
  - W adds such slots in 37–38% of all windows, and in 39% / 49% of the windows inside S1 runs. Where S is mixed, W
    is never pure.
  - This describes the input only. It is not anatomy, a route or a deployable rule.
- **Background, not a matched comparison.** S is an e224-only residual head on the same pools and seed: OOF net +12,
  mean dF +0.0012. A37 used a different round and initialisation from A36, so S is not a matched arm for A36. It
  does not settle A36's attribution question.

**Source-2 rule (fixed before any output): not met.**

| Part | Requirement | Result | Met |
|---|---|---|---|
| Wider | W − S net ≥ +5 per fold; mean dF ≥ +0.003; dF ≥ 0 per fold | +2 / −2; +0.0001; +0.0000 / +0.0002 | No |
| Useful | W − Base net ≥ 0 and dF ≥ 0 per fold; mean dF ≥ +0.003 | +9 / +3; +0.0012 / +0.0015; mean +0.0013 | No (mean) |
| Target | S1 recovered ≥ 11 of 103; correct A5C kept ≥ 1,127 of 1,150 | 8; 1,118 (97.2%) | No |
| Protect | native macro-F1 ≥ −0.003 per fold; missed short ≤ Base + 3; boundary recall ≥ −0.005 per fold | +0.0096 / +0.0031; 93 against 76; −0.0051 / −0.0051 | No (missed short, boundary recall) |

## Answers

1. **W − S shows no usable difference.**
   - Net runs are +2 / −2, and segment F1 differs by +0.0000 / +0.0002. OOF net is +12 for both.
   - W fits its training fold better: lower CE, and 34 / 51 fixes against 22 / 20. On held-out sessions the extra
     fixes are paid for by extra breaks (52 / 40 against 28 / 16), so the additional fit does not transfer as a net
     gain.
2. **The wider input reaches some S1 runs, but at the protected cost.**
   - W recovers 8 of 103 S1 runs, where S recovers 0. That is below the 11 required.
   - It loses 21 more correct A5C runs than S (1,118 against 1,139) and breaks 8 correct A4C runs into A5C.
3. **There is no full-segmentation gain beyond S.**
   - Against Base, W and S both gain about +0.001 in segment F1.
   - W changes the error form. It has fewer false short segments but more missed short true segments (93 against 76)
     and lower boundary recall (−0.0051). That breaks two protection limits.
4. **Closing scope.**
   - Under the fixed rule this configuration ends at source seed 0. There is no seed 2 and no span, rate, head, loss or
     encoder variant.
   - The result covers only this frozen A08 + doubled time offsets + residual head configuration. Span and density
     were not separated.
   - No usable difference is not equivalence. It does not show that longer temporal support or video learning is
     useless. The shared S / W gain over Base is not attributed to wider context.
   - Earlier closing decisions stand. This round does not stop the project, and A08 + B3 stays the working reference.
5. **Cost of deployment.** W needs the Base 224 forward (16.5 ms per window in A31's pass) plus a second 224 forward on
   W (16.2 ms per window), so about twice Base's GPU time. It also decodes W's frames (2.1 ms per window on CPU
   workers).
   - S needs no second forward. Its auxiliary input is e224 itself, so its minimal deployment is the Base forward plus
     the 32,965-parameter head. W needs two forwards with different inputs.
   - Dated note 2026-10-26. The first version said that a matched deployment of S would cost one more identical
     forward. That is not a necessary cost. `readout_s0.json` keeps its first wording in `forwards_S`.
   - Parameters are A08's 34.6M, shared by both forwards, plus 32,965 for the head.

## Cost and files

- **GPU 0.26 of 2 h:** the check (0.007 h) and seed-0 extraction (0.257 h).
- **CPU 0.25 of 2 h:**
  - the audit, four heads and the readout: under 0.01 h;
  - the agent's active wall clock, including the A36 wording fix.
  - All of it is in `runs/video_wider_temporal_evidence_v1/ledger.jsonl` on the server.
- **New disk:** 101 MB of 512-d eW. There is no pixel or prefix cache.
- **Records.** `audit.json`, `checks.json`, `weights_s0.json` (head manifests with fit-fold monitors and the feature
  record), `readout_s0.json`.
- **Code.** `scripts/train_wider_temporal_evidence.py`, `echo_routing/temporal/wider_temporal.py`,
  `tests/test_wider_temporal.py`.
- **Server only.** Feature caches, trained heads, per-run predictions and all identifiers.
