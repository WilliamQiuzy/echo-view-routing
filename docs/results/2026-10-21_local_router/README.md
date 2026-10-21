# A14: a local router between B3 and the frozen sustained-switch alternative (2026-10-21)

**Question.** Can the local complementarity of two segmentations keep B3's true segments while adopting A13's useful
removals of false segments? This uses a known selection/residual-correction idea. The development halves were reused for
exploratory feedback. The test split and EchoNet-Pediatric were not read. Configuration:
`configs/experiments/video_local_router_v1.yaml` (b4d0ea0), fixed before any computation.

**Design.**
- **Base and Alternative.** Base is frozen A08 + B3 at A08's committed setting. Alternative is A13's frozen H head with its
  committed threshold. Both read the same frozen A08 outputs, and each reproduces its stored rows exactly.
- **Zones.** A zone is a run of disagreement between Base and Alternative. Neighbouring runs at most 16 samples apart are
  merged.
- **Router.** A router accepts or rejects each zone. An accepted zone takes the Alternative labels; everything else stays
  Base.
- **Reproduction.** No-op reproduces Base and accept-all reproduces Alternative, both verified.
- **Training labels.** Each zone is labelled by whether replacing it alone strictly raises the sequence's segment
  F1@50, using the evaluator's matching. The pool is 300 training bank streams and 200 native clips.
- **Checks passed.** Six synthetic zone tests, label isolation, no-op and accept-all reproduction, router-only
  optimisation and an exact resume.
- **Two routers.**
  - P uses prediction and A08 probability statistics: 52 features, 1,729 parameters.
  - V adds pooled A08 embedding differences: 1,080 features, 34,625 parameters.
  - Both use one fit-set normalisation and are trained on CPU for 1,350 updates.
- **Barrier.** One calibration choice per router from 8 thresholds plus no-op, committed (a441212) before any selection
  output was read.

**Training candidates (fixed generator).**

| | Seed 0 | Seed 2 |
|---|---|---|
| Zones in the pool | 91 | 101 |
| Zones of at most 16 samples | 85 | 96 |
| Positive zones in fit | 16 | 12 |
| Positive zones in check | 0 | 0 |

- Positive zones mostly remove false segments: summed over them, FP changes by −20 and −11, and matches by +4 and +5.
- The session check set ended up with only 4 zones per seed and no positives, so it gives no generalisation estimate.
- The routers fit the small training set perfectly. The source A08 and A13 models had seen these training recordings.

## Results (seeds 0 and 2; four cells)

| System | Mean F | − Base | − Alternative | Selection FP (short) | FN (missed short true) | Mean boundary recall |
|---|---|---|---|---|---|---|
| Base (A08 A + B3) | 0.9698 | – | −0.0015 | 49 (22) | 51 (22) | 0.979 |
| Alternative (A13 H) = accept-all | 0.9713 | +0.0015 | – | 44 (18) | 54 (25) | 0.978 |
| P router | 0.9724 | +0.0026 | +0.0012 | 41 (16) | 52 (23) | 0.979 |
| V router | 0.9710 | +0.0012 | −0.0003 | 45 (22) | 52 (22) | 0.979 |

**Per seed.**
- **Seed 0, P.** The threshold is 0.8, accepting 6 of 49 zones on calibration and 14 of 46 on selection.
  - Selection: 0.9640 against Base 0.9563, +0.0077 [0.0018, 0.0152], and Alternative 0.9639.
  - Calibration: 0.9855, above both Base (0.9827) and Alternative (0.9826).
  - P keeps Base's missed-segment level (FN 20 and 8, against Alternative's 22 and 11). It removes most extra false
    segments (FP 16 and 7, against Base's 24 and 11).
  - Against Base, it removes 10 and 5 unmatched predicted segments and loses one matched reference segment in each half.
- **Seed 2, both routers.** They chose no-op on calibration, because accepting even one zone lowered F1 there. So both
  equal Base exactly. In this seed, Alternative is itself slightly below Base in both halves.
- **V against P.** V accepts fewer zones and is below P in seed 0 (−0.0038 selection, −0.0019 calibration).
- **Native recognition.** Native macro-F1 is unchanged for both routers (measured), and the A4C/A5C counts equal Base's.

**Fixed third-seed rule: not met**, so A14 closes with two seeds.
- **P.** Mean +0.0026 against the required +0.003. Its no-reversal and no-harm parts pass.
- **V.** Mean +0.0012.

## What this round shows

1. **The router can keep B3's true segments while taking the Alternative's useful removals.** In seed 0, P matches Base's
   missed segments and comes close to the Alternative's false segments, so it beats both original systems on
   calibration and Base on selection. The cost is small: one matched segment lost per half and native recognition
   unchanged.
2. **It is not yet a qualifying gain.** The mean is +0.0026 against Base and +0.0012 against the Alternative. In seed 2
   the router chose no-op, so it never does worse than Base there, but it adds nothing.
3. **The direction is local decision and routing, not visual features.** V adds nothing over P.
4. **The evidence is thin.** There are only 12–16 positive training zones per seed and no usable check set. Whether
   better candidates, a router that learned more, or combinable utilities limit the gain is not separated.

## Cost and files

- **GPU.** 0.094 h of 3, for one frozen A08 pass over the training pool. The development halves reused A13's verified
  caches.
- **CPU.** Booked in `runs/video_local_router_v1/ledger.jsonl`, including the agent's active wall clock. Routers trained
  in about 2 seconds each.
- **Records.** `preparation.json` (pool, candidates, labels), `checks.json`, `choices.json` (committed before selection)
  and `screen.json` (rows, counts, zones, comparisons, accounting and the decision).
- **Server only.** Routers and caches stay on the server.
