# A39: projected final-token channel max (P) against channel mean (G) as A5C evidence (2026-10-27 UTC)

**Question.** A27–A38 added recognition branches that read A08's 512-d CLS summary. None of them established a
target gain that protects correct A5C (A38: S1 1 / 103, correct A5C 1,114 / 1,150). On the same final non-CLS tokens of
the unchanged A08 forward, does the per-channel maximum of the projected tokens (P) give transferable A5C recognition
evidence that the per-channel mean (G) does not, without the complete A08 + B3 system getting worse?
- **The one new comparison.** U is the `encoder.norm` output of the original forward: CLS plus 8 × 7 × 7 = 392
  non-CLS tokens, already normalised. The frozen eval projection maps every non-CLS token, V = encoder.head(U[:, 1:])
  (392 × 512). Then eG = V.mean(1) and eP = V.amax(1). Projection comes first and pooling second, with no further
  normalisation, top-k, attention or ROI.
- **What it is not.** Patch evidence has been tested before. V02 replaced the main readout with V01's global /
  target-time patch means and continued end to end (closed). R07 used frozen E1 image-token attention on ASRF
  intervals (closed). A34 pooled window class probabilities over whole recordings, not tokens inside a window. Here
  CLS, q, e224, the cuts and the clip-code training target are unchanged. eG / eP only feed the residual head's
  auxiliary input.
- **What it does not show.** The final tokens have interacted with each other and with CLS. Different channels can
  reach their maximum at different times and positions. A channel maximum is therefore not a structure location,
  cardiac phase or expert evidence. The pooling reduces the joint time-space token axis, so this is not a spatial-only
  intervention. The projection was trained for CLS, which limits what a negative result covers.
- **Scope.** One source seed (A08 all_s0) on A31's two session folds; four heads (G and P per fit fold). No
  development, check, test or Pediatric data. A08 trained on these recordings, so this is a transfer screen of a head
  across A31's session folds. Labels are EV9V clip codes.
- **Records.** Configuration `configs/experiments/video_patch_evidence_v1.yaml` (323fe07, committed before any token
  statistic, forward or training). Code 0628af9. Task sheet `claude_prompts/2026-10-26-local-token-a5c-evidence.md`.

## Design

- **Windows.** A31's pools and request order: stride-5 centres on the complete constructed sequences, A08's 16
  samples (10 Hz, about 1.5 s), A08's marker, gamma and 256 → 224 crop, and the nearest-centre output map. No A37 / A38
  window, clip or predicted-boundary limit.
- **One forward per window.** A read-only hook on `encoder.norm` returns nothing and stores a clone of the output. It
  is registered for one call and removed in `finally`. `TargetMViT.embed` and the classifier run unchanged and give e224
  and q, which are compared with A31's cache for every window. G and P share this forward and these tokens.
- **Arms.** z_G = q + g_G(concat(e224, eG)) and z_P = q + g_P(concat(e224, eP)).
  - g is the existing residual head: Linear(1024, 32) → GELU → Linear(32, 5), 32,965 parameters, last layer zero.
    The first layer is seeded 20261065 + fit fold and is bit-identical for G and P.
- **Training.** Only g is trained, on A31's uniform plans: 1,350 updates of 16 T and 16 P runs each,
  pooled-probability CE. There is no KL, error emphasis or class balancing. AdamW lr 1e-3, weight decay 1e-4, warm-up
  135 then cosine, clip 1. Final weights only, CPU single thread.
- **Readout.** The unchanged Base runs are relabelled by the argmax of the multiplicity-weighted mean probability, and
  equal neighbours merge. Complete sequences are scored with the original evaluator.

**Checks** (`checks.json`, all passed at the first attempt).
- **Forward unchanged.** With the hook, q and e224 equal the plain `embed` / classifier output bit for bit. Online q and
  e224 equal A31's cache exactly, on the 32 check windows and later on all 50,359 extracted windows (max abs 0.0).
- **Token contract.** U is 32 × 393 × 768, and the grid read from the network is 8 × 7 × 7. encoder.head(U[:, 0])
  equals e224 exactly, so CLS is index 0 and is excluded from V (32 × 392 × 512). The head is Dropout → Linear, in eval
  mode.
- **Pooling and projection recomputed independently.**
  - V equals an explicit float64 matmul to within 1.2e-6, and eG equals the projection of the token mean to within 3.8e-7.
  - eP equals a token-by-token running maximum exactly. eG equals a running float64 mean to within 4.8e-7.
  - A repeat gives 0, and a 16 + 16 batch split 3.3e-6. All values are finite, and eP ≥ eG everywhere. TF32 is off,
    no hook is left and the A08 weights are unchanged.
  - Synthetic tokens: max ≥ mean, and identical tokens give eP = eG = that token.
- **Init = Base.** The untrained head returns q exactly. Every P run and complete sequence equals Base, and the rows
  match A32's Base rows in both folds.
- **Training mechanics.** Forced eP = eG gives identical losses and weights. The first-layer gradient is zero at step 1,
  the optimiser binds exactly g, and 3 + 3 resumed updates equal 6. The folds share no session, recording or window.
  All 138 loaded repository modules match code commit 0628af9.
- **Readout dry run.** Before training, the readout code ran once with Init heads only (z = q). Nothing was published.
  Base reproduced A32's rows and A38's Base cells, and Init equalled Base in every table.

**Features and training.**
- **Features.** There was one forward per window, 50,359 in total (24,585 / 25,774). This took 0.261 GPU h (16.2 ms GPU
  and 2.3 ms decode per window) and 197 MB for eG + eP. The record is in `weights_s0.json`.
  - Label-free description: a window's 512 channel maxima come from a median of 171 different tokens (5–95%: 148–191
    of 392). 21% / 20% of maxima sit at the first / last of the 8 time positions, against 8–11% at each middle one.
    This describes the descriptor only; it is not a location or phase.
- **Training.** Each head ran 1,350 real updates in about 3 s.
- **Fit folds.** Pooled CE starts at 0.1316 / 0.1404 and falls to 0.0876 / 0.1014 for G and 0.0888 / 0.1012 for P. G
  fixes 24 and 19 of the fit folds' 109 and 145 Base errors (breaking 3 and 3); P fixes 21 and 20 (breaking 2 and 4).

## Results (held-out folds; OOF = both folds summed)

**All held-out P runs.** Fold 0 has 3,384 runs and 109 Base errors; fold 1 has 3,404 and 145. Correct A5C runs are
561 / 589 and correct A4C runs 714 / 699. A32's S1 runs (true A5C read as A4C by Base) are 34 / 69, from 12 / 26
A5C-coded source recordings in 12 / 21 sessions.

| Model | Fixed / broken / net (f0; f1; OOF) | A4C read as A5C fixed | S1 fixed | Correct A5C kept | Correct A4C kept | Segment F1@50 (f0 / f1) | Native macro-F1 | Boundary R (f0 / f1) | Boundary P (f0 / f1) | FP / short FP | FN / missed short |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Base (= Init) | 0 / 0 / 0 | 0 | 0 / 103 | 1,150 / 1,150 | 1,413 / 1,413 | 0.9639 / 0.9499 | 0.9776 / 0.9758 | 0.9676 / 0.9681 | 0.9556 / 0.9420 | 262 / 178 | 183 / 76 |
| G (mean) | 12/6/+6; 15/12/+3; **27/18/+9** | 27 | 0 | 1,138 | 1,411 | 0.9645 / 0.9507 | 0.9871 / 0.9810 | 0.9666 / 0.9656 | 0.9570 / 0.9447 | 250 / 168 | 187 / 81 |
| P (max) | 11/4/+7; 13/8/+5; **24/12/+12** | 24 | 0 | 1,143 | 1,411 | 0.9650 / 0.9512 | 0.9852 / 0.9810 | 0.9671 / 0.9661 | 0.9570 / 0.9433 | 250 / 170 | 182 / 78 |

FP, FN and their short parts are summed over the two folds.

**P − G, the matched comparison, and P − Base.**

| Contrast, fold | Net runs | S1 fixed | Correct A5C / A4C kept | Segment dF | Native macro-F1 | Boundary R | Boundary P | FP / short FP | FN / missed short |
|---|---|---|---|---|---|---|---|---|---|
| P − G, 0 | +1 | 0 | +1 / 0 | +0.0006 | −0.0019 | +0.0005 | +0.0000 | −1 / 0 | −2 / −1 |
| P − G, 1 | +2 | 0 | +4 / 0 | +0.0004 | +0.0000 | +0.0005 | −0.0014 | +1 / +2 | −3 / −2 |
| P − Base, 0 | +7 | 0 | −2 / −1 | +0.0011 | +0.0076 | −0.0005 | +0.0014 | −5 / −3 | −1 / 0 |
| P − Base, 1 | +5 | 0 | −5 / −1 | +0.0013 | +0.0051 | −0.0021 | +0.0013 | −7 / −5 | 0 / +2 |

- **Direction of the changes (OOF).**
  - Every fix of both heads is a true A4C run that Base read as A5C (G 27, P 24). Neither head fixes any other kind of
    error.
  - P breaks 12 runs: 6 correct A5C into A4C, 1 A5C into PLAX, 1 A4C into A5C, 1 A4C into PLAX, 1 PSAX into A4C, 1
    PSAX into SC4C and 1 PLAX into SC4C. G breaks 18, 10 of them correct A5C into A4C.
  - Over all 1,255 true A5C runs, recall is 0.9163 for Base, 0.9068 for G and 0.9108 for P (read as A4C: 103 / 115 /
    111). Over all 1,554 true A4C runs it is 0.9093 / 0.9254 / 0.9234 (read as A5C: 135 / 109 / 112).
  - A32's S2 runs kept as A5C are 83 / 84 and 152 / 157 under P, against 83 / 84 and 149 / 157 under G.
- **Sources.** P recovers no S1 run, so there are no A5C-coded source recordings to count. P's fixes come from 24
  recordings in 23 sessions and its breaks from 17 / 16. G's come from 26 / 25 and 25 / 24. A run counts all its
  source recordings.
- **Initial cuts and final boundaries.** The initial cuts never move (2,044 / 2,057 internal Base boundaries).
  Relabelling and merging deletes 5 / 13 of them under P and 6 / 18 under G.

**Source-2 rule (fixed before any output): not met.**

| Part | Requirement | Result | Met |
|---|---|---|---|
| Target increment | P − G S1 fixed ≥ +1 per fold and ≥ +5 in total | 0 / 0; total 0 (P 0, G 0) | No |
| A5C signal | P S1 fixed ≥ 11 of 103, from ≥ 3 A5C-coded recordings and ≥ 1 per fold | 0; 0 recordings | No |
| Retention | correct A5C ≥ 1,127 of 1,150 and ≥ G − 3; correct A4C ≥ 1,385 of 1,413 and ≥ G − 3 | 1,143 (G 1,138); 1,411 (G 1,411) | Yes |
| Net | P − Base ≥ 0 and P − G ≥ 0 per fold | +7 / +5; +1 / +2 | Yes |
| System | segment dF (P − Base) and (P − G) ≥ 0 per fold; native ≥ −0.003; boundary recall ≥ −0.005 per fold; missed short ≤ Base + 3 | +0.0011 / +0.0013 and +0.0006 / +0.0004; +0.0076 / +0.0051; −0.0005 / −0.0021; 78 against 76 (limit 79) | Yes |

## Answers

1. **The channel maximum adds no A5C recognition evidence here.** P and G each recover 0 of the 103 S1 runs. The
   target increment is 0 in both folds, so the main question is answered negatively for this configuration.
2. **P keeps correct runs slightly better than G, but the difference is small.** P breaks 12 runs against G's 18, and
   keeps 1,143 correct A5C runs against 1,138. P − G is +1 / +2 net runs and +0.0006 / +0.0004 in segment F1. This is
   one source seed on training-data folds with shared sources, so no difference is established and no significance is
   claimed.
3. **Both heads are slightly above Base in the same one-way pattern as earlier residual heads.** P is +12 net runs OOF
   (G +9), with segment dF +0.0011 / +0.0013. All of it comes from fixing A4C runs that Base read as A5C, at the cost
   of 7 correct A5C runs (6 into A4C). This resembles the shared shift of A34, A36 and A38, and it is not attributed
   to the token descriptor. Every retention and full-system protection passes (missed short true segments 78 against
   Base's 76; G has 81).
4. **Closing scope.**
   - Under the fixed rule this configuration ends at source seed 0. There is no source 2 and no pool, head, layer,
     loss or seed variant.
   - The result covers only the frozen A08's projected final-token channel max / mean with this residual head and
     clip-code labels. The tokens have interacted, the projection was trained for CLS, and a channel maximum is not a
     location. Not passing is not equivalence and does not disprove local or patch evidence in general. A08 + B3 stays
     the working reference.
5. **Cost of deployment.** One Base 224 forward per window (16.2 ms GPU here, 16.5 ms in A31's pass); eG / eP come from
   its final tokens, with no second encoder pass. On the same 1,027 check windows, the token path (hook clone,
   392-token projection and pooling) took 0.010 ms per window longer than the plain forward, which is within timing
   noise of a 16 ms forward. The head adds about 0.003 ms per window on one CPU thread.
   Parameters are A08's 34.6M plus 32,965 for the head.

## Cost, timeline and files

- **GPU 0.275 of 2 h:** the check (0.014 h) and the source-0 extraction (0.261 h). No job failed.
- **CPU 0.34 of 2 h, as of 2026-10-27 03:35:28 UTC (ledger):**
  - the Init dry run, four heads, freeze and the readout: 0.012 h;
  - the agent's active wall clock, 0.327 h in two spans (waiting during the extraction, 03:16:30–03:30:12, excluded).
  - All of it is in `runs/video_patch_evidence_v1/ledger.jsonl` on the server.
- **New disk:** 207 MB (eG + eP 197 MB) of 1 GB. There is no token, pixel or prefix cache.
- **Timeline (UTC, 2026-10-27).** Dispatch 03:02, configuration 03:07, code 03:13, checks complete 03:14, extraction
  03:14–03:30, Init dry run 03:31, heads 03:32, weights committed 03:32 (171b295), freeze and readout 03:32–03:33,
  cross-checked summary 03:35. Codex's review file had no blocking finding before the extraction or before the
  freeze. No helper agents were used.
- **Records.** `checks.json`, `weights_s0.json` (head manifests with fit-fold monitors and the feature record),
  `readout_s0.json`.
- **Code.** `scripts/train_patch_evidence.py`, `echo_routing/temporal/token_pool_evidence.py`,
  `tests/test_token_pool_evidence.py`.
- **Server only.** Feature caches, trained heads, per-run predictions and all identifiers.
