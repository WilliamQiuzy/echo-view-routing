# A38: the frozen Base's predicted run limits A37's wider auxiliary window, C against W (2026-10-26)

**Question.** A37's W (doubled time offsets on the frozen A08) recovered 8 of 103 S1 runs but broke 40 correct runs. The
post-A37 diagnostic found that 7 of those 8 fixes and 24 of the 40 breaks depend on eW. Does clipping W to the frozen
Base's predicted run around the request centre (C) keep more correct runs than the unlimited W, while keeping useful
A5C corrections?
- **The one new input.** For a request centred at c, [a(c), b(c)) is the run of the Base's run-length encoding that
  contains c, and C = clip(W, a(c), b(c) − 1), slot by slot. The marker slot stays c.
  - The mapping reads positions, sequence lengths and the Base's predicted labels only. It never reads reference
    labels, T boundaries, clip codes, S1 / harm sets or training targets.
  - There is one mapping per window, shared by T batches, P batches, fit monitors and the readout.
- **What it is not.** This is not a fixed span / density variant of A37 (closed). It is not A28's single interval clip
  replacing the recognition input either: q, e224, the cuts and the final merge are unchanged, and C only feeds the
  residual head's auxiliary input.
- **What it does not separate.** C changes span, repeat rate and boundary position together. This round judges the
  whole input condition. It does not show that corrections come from visible LVOT, or that cross-boundary content
  caused A37's harm.
- **Scope.** One source seed (A08 all_s0) on A31's two session folds; four heads (W and C per fit fold). No
  development, check, test or Pediatric data. A08 trained on these recordings, so this is a transfer screen of a head
  across A31's session folds. Labels are EV9V clip codes.
- **Records.** Configuration `configs/experiments/video_predicted_run_context_v1.yaml` (161c2be, committed before
  any mapping statistic). Code 5139f71, with one bookkeeping fix 1808303. Task sheet
  `claude_prompts/2026-10-26-predicted-run-context.md`.

## Design

- **Windows.** Both arms use the complete constructed sequences of A31's pools, the original stride-5 centres and the
  nearest-centre output map. Frames come from the sequence's own refs, with A08's gamma and 256 → 224 crop. S, c, the
  marker and W are exactly A37's.
- **Arms.** z_W = q + g_W(concat(e224, eW)) and z_C = q + g_C(concat(e224, eC)).
  - g is A37's residual head: Linear(1024, 32) → GELU → Linear(32, 5), 32,965 parameters, last layer zero. The first
    layer is seeded 20261063 + fold. It is bit-identical for W, C and A37.
  - eW is A37's verified cache: its record is the one committed with A37's heads, and A37's key still matches. eC
    equals eW wherever C's complete ref sequence equals W's. Every other window gets a new forward of the same frozen
    A08.
- **Training.** Only g is trained, on A31's uniform plans: 1,350 updates of 16 T and 16 P runs each,
  pooled-probability CE, no KL. AdamW lr 1e-3, weight decay 1e-4, warm-up 135 then cosine, clip 1. Final weights
  only, CPU single thread. Per fold, W is trained first and must reproduce A37's W head.
- **Readout.** The unchanged Base runs are relabelled by the argmax of the multiplicity-weighted mean probability,
  and equal neighbours merge. Complete sequences are scored with the original evaluator.

**Mapping audit** (`audit_s0.json`; positions, lengths and Base predictions only, before any feature).
- **Denominators.** Fold 0 / 1: 24,585 / 25,774 requests in 1,340 / 1,347 sequences (740 / 747 native), 1,228 / 1,248
  recordings and 200 / 200 sessions.
- **Independent loop.** It re-derives every window's Base run, W, C and marker slot with 0 mismatches. Every [a, b) is
  one of the Base P runs. C is ordered, inside [a, b) and inside the sequence, and its marker slot is c. No window that
  needs no limit has C ≠ W.
- **How much changes.** The run limit changes 10,201 / 10,485 windows (41.5% / 40.7%). They come from 1,121 / 1,098
  recordings and 198 / 199 sessions, and 205 / 279 of them are in native sequences.
  - 13.8% / 13.3% of W slots move, against 4.9% / 4.6% clamped at a sequence edge by W itself.
  - Every changed window has a different ref sequence, so each one needs a new forward. Every other window reuses
    eW.
- **Span and repeats.**
  - Mean span is 24.3 / 24.6 samples for C, against W's 28.6 and S's 15.0. On the changed windows, C spans 19.2 / 19.4
    against W's 29.4.
  - 8.5% / 7.6% of C windows span less than S. No window has C = S.
  - C repeats a position in 63.0% / 61.5% of windows (16.7% / 16.0% of slots), against W's 27.8% / 26.8% (4.1% /
    3.8%). Distinct refs per window are 13.3 / 13.4 for C, 15.3 / 15.4 for W and 16.0 for S.
  - The Base run containing the centre has a median of 43 / 45 samples (5–95%: 15–238).
  - 769 / 791 centres sit on the last sample of their run, while 0 / 1 sit on the first. This is a position fact
    only; no rule uses it.

**Checks** (`checks.json`; the first attempt failed in bookkeeping, as listed under cost).
- **Features.**
  - Online S logits and embeddings equal A31's cache exactly.
  - C's normal path equals an explicitly stacked forward, with bit-equal input and output.
  - On windows where C = W, an online forward equals the reused eW to within 9.5e-7, which is batch composition
    only. On changed windows, eC differs from eW (max abs 4.2). TF32 is off and the A08 weights are unchanged.
- **Init = Base.** The untrained head returns q exactly. Every P run and complete sequence equals Base, and the rows
  match A32's Base rows in both folds.
- **Training mechanics.**
  - Forced eC = eW gives identical losses and weights.
  - The first-layer gradient is zero at step 1.
  - The optimiser binds exactly g, and 3 + 3 resumed updates equal 6.
  - The folds share no session, recording or window. A37's W head files are unchanged.
  - All 136 loaded repository modules match the committed code.
- **Projection.** 19.2 ms per new window (decode + GPU): 0.13 GPU h for source 0 and at most 0.49 h for a source-2
  run. 0.10 GB of disk per source-0 feature set.

**Features and training.**
- **Features.** 20,686 of 50,359 windows needed a new C forward (10,201 / 10,485); the other 29,673 reuse eW
  (`weights_s0.json`, feature record). This took 0.110 GPU h (16.1 ms GPU and 2.8 ms decode per new window) and
  103 MB.
- **Training.** Each head ran 1,350 real updates in about 3 s.
- **W reproduces A37.** Both fresh W heads equal A37's W heads in file sha256, update log and tensors. Base and W
  also reproduce A37's held-out readout cell for cell in both folds (`A37_reproduced`), so the W row below is A37's W.
- **Fit folds.** Pooled CE starts at 0.1316 / 0.1404. It falls to 0.0785 / 0.0947 for C, against 0.0774 / 0.0900
  for W. C fixes 36 and 29 of the fit folds' 109 and 145 Base errors (breaking 3 and 5). W fixes 34 and 51 (breaking
  2 and 11).

## Results (held-out folds; OOF = both folds summed)

**All held-out P runs.** Fold 0 has 3,384 runs and 109 Base errors; fold 1 has 3,404 and 145. Correct A5C runs are
561 / 589, and A32's S1 runs (true A5C read as A4C by Base) are 34 / 69.

| Model | Fixed / broken / net (f0; f1; OOF) | A4C read as A5C fixed | S1 fixed | Correct A5C kept | Segment F1@50 (f0 / f1) | Native macro-F1 | Boundary R (f0 / f1) | Boundary P (f0 / f1) | FP / short FP | FN / missed short |
|---|---|---|---|---|---|---|---|---|---|---|
| Base (= Init) | 0 / 0 / 0 | 0 | 0 / 103 | 1,150 / 1,150 | 0.9639 / 0.9499 | 0.9776 / 0.9758 | 0.9676 / 0.9681 | 0.9556 / 0.9420 | 262 / 178 | 183 / 76 |
| W (= A37's W) | 24/15/+9; 28/25/+3; **52/40/+12** | 44 | 8 | 1,118 | 0.9651 / 0.9513 | 0.9871 / 0.9789 | 0.9626 / 0.9630 | 0.9645 / 0.9518 | 224 / 137 | 205 / 93 |
| C | 24/17/+7; 29/20/+9; **53/37/+16** | 51 | 1 | 1,114 | 0.9664 / 0.9560 | 0.9852 / 0.9770 | 0.9646 / 0.9666 | 0.9651 / 0.9543 | 211 / 127 | 187 / 80 |

FP, FN and their short parts are summed over the two folds.

**C − W, the matched comparison.**

| Fold | Net runs | Segment dF | Native macro-F1 | Boundary R | Boundary P | FP / short FP | FN / missed short |
|---|---|---|---|---|---|---|---|
| 0 | −2 | +0.0014 | −0.0019 | +0.0020 | +0.0006 | −2 / −4 | −5 / −6 |
| 1 | +6 | +0.0047 | −0.0019 | +0.0036 | +0.0026 | −11 / −6 | −13 / −7 |

**C − Base.**

| Fold | Net runs | Segment dF | Native macro-F1 | Boundary R | Boundary P | FP / short FP | FN / missed short |
|---|---|---|---|---|---|---|---|
| 0 | +7 | +0.0025 | +0.0076 | −0.0030 | +0.0095 | −20 / −22 | +6 / +3 |
| 1 | +9 | +0.0062 | +0.0012 | −0.0015 | +0.0123 | −31 / −29 | −2 / +1 |

- **Direction of the changes (OOF).**
  - C fixes 51 A4C runs read as A5C (W 44) and 1 S1 run (W 8). It also fixes one PLAX run read as A4C.
  - C breaks 36 correct A5C runs into A4C (W 32) and one correct A4C run into SC4C. It breaks no correct A4C run into
    A5C (W 8).
  - A32's S2 runs kept as A5C are 71 / 84 and 138 / 157 under C, against 75 / 84 and 137 / 157 under W.
- **Coverage.** C's fixes come from 50 recordings in 47 sessions and its breaks from 40 / 38. W's come from 54 / 49
  and 48 / 43. A run counts all its source recordings.
- **Initial cuts and final boundaries are different things.**
  - The initial cuts never move: every arm relabels the same Base runs (2,044 / 2,057 internal boundaries).
  - Relabelling and merging equal neighbours deletes 28 / 41 of these boundaries under C, against 37 / 44 under W
    (0 under Base).

**Source-2 rule (fixed before any output): not met.**

| Part | Requirement | Result | Met |
|---|---|---|---|
| Matched | C − W net ≥ +5 per fold; dF ≥ 0 per fold; mean dF ≥ +0.003 | −2 / +6; +0.0014 / +0.0047; mean +0.0030 | No (fold-0 net) |
| Useful | C − Base net ≥ 0 and dF ≥ 0 per fold; mean dF ≥ +0.003 | +7 / +9; +0.0025 / +0.0062; mean +0.0043 | Yes |
| Target | S1 recovered ≥ 11 of 103; correct A5C kept ≥ 1,127 of 1,150 | 1; 1,114 (96.9%) | No |
| Protect | native macro-F1 ≥ −0.003 per fold; missed short ≤ Base + 3; boundary recall ≥ −0.005 per fold | +0.0076 / +0.0012; 80 against 76 (limit 79); −0.0030 / −0.0015 | No (missed short) |

## Answers

1. **C − W: no consistent run-level gain, but better segmentation in both folds.**
   - Net runs are −2 / +6 (OOF +16 against +12), so the matched run-level requirement fails in fold 0.
   - Segment F1 is higher in both folds (+0.0014 / +0.0047). C has fewer false and fewer missed short segments than W,
     and higher boundary recall (+0.0020 / +0.0036).
2. **C does not keep the useful A5C corrections.**
   - C recovers 1 of 103 S1 runs, where W recovers 8. It keeps 1,114 correct A5C runs against W's 1,118.
   - C moves runs in one direction, toward A4C: more A4C runs read as A5C are fixed (51 against 44), and more correct
     A5C runs are broken into A4C (36 against 32). It breaks no correct A4C run into A5C, where W breaks 8.
3. **Against Base, C has the larger segment gain of the two heads but misses one protection limit.**
   - Segment dF is +0.0025 / +0.0062 (mean +0.0043), and the useful part of the rule is met.
   - Missed short true segments are 80 against Base's 76; the limit is 79. Boundary recall falls by 0.0030 / 0.0015,
     within the limit. Native macro-F1 rises in both folds.
4. **Closing scope.**
   - Under the fixed rule this configuration ends at source seed 0. There is no seed 2 and no span, limit, head, loss,
     sampling or encoder variant.
   - The result covers only this frozen A08 + Base-run-limited doubled offsets + residual head configuration. C changes
     span, repeat rate and boundary position together, and no part of the difference is attributed to one of them.
     The S1 loss is not evidence that cross-boundary frames carry LVOT information.
   - Not meeting the rule is not equivalence, and it does not stop the research line. The segment gain against Base
     is a single-seed screen on training-data folds, not a new candidate or a new strongest model. A08 + B3 stays the
     working reference, and A37's closing decision is unchanged.
5. **Cost of deployment.** C needs the Base 224 forward (16.5 ms per window in A31's pass) plus a second 224 forward on
   C for every window (16.1 ms per new forward measured here), so about twice Base's GPU time. It also decodes C's
   frames (2.8 ms per window on CPU workers) and needs the Base run before the auxiliary input can be formed.
   - The reuse of eW where C = W is a training-time cache saving only. A deployment without W has no eW to reuse.
   - Parameters are A08's 34.6M, shared by both forwards, plus 32,965 for the head.

## Cost, timeline and files

- **GPU 0.125 of 2 h:** a first check attempt that failed in bookkeeping (0.007 h; a torch-generated module reports
  `_ops.py`, fixed in 1808303), the check (0.007 h) and the source-0 C extraction (0.110 h).
- **CPU 0.35 of 2 h, as of 2026-10-26 20:49:46 UTC (ledger):**
  - the audit, four heads, freeze and the readout: 0.009 h;
  - the agent's active wall clock, 0.34 h (waiting during the extraction excluded).
  - All of it is in `runs/video_predicted_run_context_v1/ledger.jsonl` on the server.
- **New disk:** 104 MB (eC 103 MB) of 1 GB. There is no pixel or prefix cache.
- **Timeline (UTC, 2026-10-26).** Dispatch 20:25, configuration 20:29, code 20:35, audit 20:36, checks 20:38,
  extraction 20:38–20:45, heads 20:46, weights committed 20:46, readout 20:47. No helper agents were used.
- **Records.** `audit_s0.json`, `checks.json`, `weights_s0.json` (head manifests with fit-fold monitors, the A37
  reproduction and the feature record), `readout_s0.json`.
- **Code.** `scripts/train_predicted_run_context.py`, `echo_routing/temporal/predicted_run_context.py`,
  `tests/test_predicted_run_context.py`.
- **Server only.** Feature caches, trained heads, per-run predictions and all identifiers.
