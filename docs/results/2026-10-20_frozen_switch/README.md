# A13: a sustained-switch head on the frozen A08 classification path (2026-10-20)

**Question.** When A08's classification outputs are held fixed, can A11's sustained-switch evidence become a
full-segmentation gain? The development halves were reused for exploratory feedback. The test split and
EchoNet-Pediatric were not read. Configuration: `configs/experiments/video_frozen_switch_v1.yaml` (104c06b), fixed before
any A13 output.

**Design.**
- **Frozen path.** Each seed's A08 all_s{k} TargetMViT is frozen whole: eval mode, no gradients, and identical parameter
  and buffer digests before and after.
- **Head.** Only A11's central-edge head is trained (33,921 parameters, fresh initialisation with seed 3000 + seed). It is
  trained on CPU and is exactly deterministic.
- **Arms.** U and H differ only in A11's negative weights. Both read A11's sealed manifests, central edges, masks,
  hard-pool groups and weights.
- **Features.** One cache per seed of A08's eval logits and embeddings for all 1,350 × 12 training windows, shared by U
  and H.
- **Checks passed.** These covered the frozen path, head-only gradients and optimiser, identical U/H initialisation and
  features, equal masses, and an exact resume.
- **Development.** The new A08 logits equal A08's saved outputs on every window (maximum difference 0.0). Each model's
  CLS + B3 reproduces A08's stored rows exactly, and the rebuilt A11 head rows reproduce A11's.
- **Decoder.** A11's decoder without the left/right check. Calibration choices were committed (e8ccf20) before any
  selection output was read.

## Results (seeds 0 and 2; four cells; primary systems are the head decoders)

| System | Mean F | Mean F − A08 A | Selection FP (short) | FN (missed short true) | Boundary recall / precision | Native macro-F1 | A4C→A5C / A5C→A4C | Join cuts |
|---|---|---|---|---|---|---|---|---|
| A08 A + B3 (= A13 CLS + B3) | 0.9698 | – | 49 (22) | 51 (22) | 0.979 / 0.966 | 0.9839 | 6 / 1 | 19 |
| A13 U head | 0.9707 | +0.0009 | 43 (18) | 56 (26) | 0.977 / 0.973 | 0.9858 | 6 / 0 | 17 |
| A13 H head | 0.9713 | +0.0015 | 44 (18) | 54 (25) | 0.978 / 0.973 | 0.9858 | 6 / 0 | 17 |
| A11 U head (historical) | 0.9659 | −0.0039 | 66 (37) | 53 (18) | – | – | – | – |
| A11 H head (historical) | 0.9686 | −0.0012 | 62 (35) | 46 (13) | – | – | – | – |

**Per cell, against A08 A.**
- U: +0.0054 (seed 0 selection), −0.0011 (seed 0 calibration), +0.0003 (seed 2 selection), −0.0010 (seed 2 calibration).
- H: +0.0075, −0.0001, −0.0006, −0.0010.
- Every per-seed paired interval includes zero.
- **Where seed 0 gains.** On selection it comes from fewer false segments: 14 (2 short) against 24 (9 short), with
  boundary precision 0.984 against 0.956.
- **Where calibration loses.** It loses slightly through more missed segments: 11 against 7 in seed 0.
- **Native macro-F1 is measured, not assumed.** It equals A08's in seed 0. In seed 2 calibration it is slightly higher,
  because the head's cuts change the native majority (one A5C read as A4C is avoided).

**Against A11 (paired, same development data).**
- **Seed 2 selection.** U − A11 U is +0.020 [0.005, 0.037] and H − A11 H is +0.012 [−0.0004, 0.028].
- **Seed 0.** The differences are small (−0.003 to +0.001).

**H − U.** Selection +0.0021 and −0.0009; calibration +0.0010 and 0.0000. Mixed, so not established.

**Fixed third-seed rule: not met**, so A13 closes with two seeds.
- **U.** Mean +0.0009, below +0.003. It also misses 26 short true segments, above the 25 allowed.
- **H.** Mean +0.0015, below +0.003. Its no-reversal and no-harm parts pass.

## What this round shows

1. **Keeping A08's outputs fixed gives a small, non-qualifying gain.** The frozen-path head is about level with A08 A
   (+0.001 to +0.0015). It cuts false segments, with higher precision, but misses slightly more true changes.
   - A11's short-segment recall did not carry over: 25–26 missed short true segments, against 22 for A08 A and 13 for
     A11 H.
   - This closes this frozen-classifier configuration only. It does not show that frozen-feature methods in general
     fail.
2. **Seed 2's loss does not reappear under this condition.** Seed 2 selection is at A08's level, against A11's −0.013 to
   −0.020.
   - This is consistent with A11's seed-2 loss involving the change of the classification path and its decoding point.
   - But the frozen condition also changes the feature mode (eval) and the optimisation (head only). No single cause is
     credited.
3. **Hard-negative weighting is not established.** H − U is mixed.

## Cost and files

- **GPU.** 0.266 h of 3: feature extraction 0.134, calibration inference 0.063 and selection inference 0.069. Head
  training took about 2 seconds of CPU per head.
- **CPU.** Booked in `runs/video_frozen_switch_v1/ledger.jsonl`, including the agent's active wall clock.
- **Records.** `checks.json`, `choices.json` (committed before selection was read) and `screen.json` (rows, counts,
  comparisons, accounting and the third-seed decision).
- **Server only.** The heads (`runs/video_frozen_switch_v1/train/*/final.pt`) and the feature caches stay on the server.
