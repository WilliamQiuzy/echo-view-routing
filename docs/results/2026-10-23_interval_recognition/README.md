# A27: interval recognition supervision on the frozen A08 video representation (2026-10-23)

**Question.** On A08's frozen 512-dimensional recognition representation, does a new head trained on intervals beat the
same head trained on single windows for the class labels of the fixed Base + B3 intervals? And does that become a
full-segmentation gain after relabelling and merging?
- **Arms.** W trains with window CE averaged inside each training interval. I trains with CE on the interval's pooled
  probability.
- **Shared term.** Both add the same own-A08 retention KL with coefficient 1.
- **Origin.** This is Task 4 of `docs/superpowers/plans/2026-10-23-experiment-navigation-and-recognition.md`, authorised
  by the user and the lead.
- **How it differs from earlier rounds.**
  - R05 used span statistics on probability outputs, and R06 an older image encoder.
  - A01 and A05 trained window-level readouts on V01 features.
  - A14–A17 selected local edits, and A25 trained a four-window expected-error surrogate.
  - Here only an independent copy of A08's 512→5 classifier is trained, on cached frozen embeddings.
- **No EchoViewCLIP.** Codex's 2026-10-23 diagnostic showed that relabelling Base intervals with EchoViewCLIP lowers all
  four cells. So there is no EchoViewCLIP replacement or distillation here.
- **Data limits.** The development halves are reused exploratory feedback. The A15 check pool's recordings trained the
  encoder, so it is not a holdout. Test and Pediatric were not read.
- Configuration: `configs/experiments/video_interval_recognition_v1.yaml` (43fe344), fixed before any A27 output.

**Design.**
- **Representation.** A15's cached frozen A08 embeddings: the large fit pool of 3,187 pool sequences, of which the fit
  sequences train the heads. The cached embedding is the input of A08's classifier.
- **Head.** A copy of A08's `classifier` (2,565 parameters), the only trainable tensor. The encoder, marker, projection
  and original classifier are untouched.
- **Training intervals** (fit pool only).
  - T is the reference constant-class runs: 7,242 per seed.
  - P is the seed's Base runs labelled by their majority reference class: 7,411 and 7,366. About 46% are mixed (mean
    purity 0.954), and 274/256 carry a label other than Base's.
  - Every interval keeps its samples' nearest request windows with multiplicity weights. Every interval has at least two
    windows, and none was dropped.
- **Batches.** 1,350 updates of 16 T plus 16 P intervals, uniform with replacement, from `default_rng([20261023, 27, seed])`.
  W and I share them.
- **Optimiser.** AdamW, lr 1e-4, warm-up 135 then cosine, clip 1, CPU on one thread.
- **Deployment.** Original q makes the Base + B3 labels at A08's committed numeric setting. Each Base run takes the argmax of
  the new head's mean probability over its samples. Equal neighbours merge, and the complete sequences are scored.
- **Order.**
  - The code, preparation and checks were committed in 983e863.
  - Training ran as PID 373279 (1,350 updates per run, 1.7–1.9 s each).
  - The frozen head identities were committed in 6f6964f before evaluation.
  - The development halves and check pool were evaluated once.

**Checks passed** (`checks.json`).
- The six unit tests pass:
  - objectives against independent numpy;
  - I ≠ W in general, and I = W for equal predictions;
  - uneven and short-interval mappings;
  - gradients only in the head;
  - merging removes a boundary;
  - the majority and tie rules.
- The copied head reproduces the cached logits within 5.7e-6, with 100% argmax agreement.
- With zero updates, the adapter reproduces Base on all 3,187 pool sequences and the four stored development rows exactly.
- The original q is unchanged. Exactly 2,565 parameters are trainable and optimised.
- A 3 + 3 resume equals 6 straight updates exactly.

## Results (seeds 0 and 2)

| System | s0 sel | s0 cal | s2 sel | s2 cal | Check s0 / s2 |
|---|---|---|---|---|---|
| Base (A08 + B3) | 0.9563 | 0.9827 | 0.9564 | 0.9837 | 0.9603 / 0.9555 |
| W head | 0.9563 | 0.9827 | 0.9534 | 0.9837 | 0.9609 / 0.9555 |
| I head | 0.9563 | 0.9827 | 0.9534 | 0.9837 | 0.9609 / 0.9560 |
| A14 P (stored, context) | 0.9640 | 0.9855 | 0.9564 | 0.9837 | – |

**Fixed gate: not met by either arm**, so A27 closes with two seeds and no seed-1 run.

| Part | W | I | Required |
|---|---|---|---|
| Mean dF against Base | −0.00075 | −0.00075 | ≥ +0.003 |
| Half means (selection / calibration) | −0.0015 / 0.0 | −0.0015 / 0.0 | each ≥ −0.005 |
| Lowest cell | −0.0030 (s2 selection) | −0.0030 (s2 selection) | ≥ −0.015 |
| Native macro-F1 / boundary recall difference | 0.0 / −0.0007 | 0.0 / −0.0007 | ≥ −0.003 / ≥ −0.010 |
| Missed short true segments (four cells) | 22 against Base's 22 | 22 against 22 | ≤ Base + 3 |
| Calibration feasibility; selection native floor 0.973 | both seeds; both seeds | both seeds; both seeds | each seed |

- **I − W.** It is exactly 0 in every development cell. On check it is 0 and +0.0006 (one recovered segment in seed 2).
  On check native macro-F1, I is +0.005 above W in seed 0 (one clip). There is no interval-supervision effect to attribute.
- **Against Base.**
  - Both heads equal Base in three development cells.
  - In seed 2 selection they lose 2 matched segments (−0.0030 [−0.0081, 0.0]).
  - On check they change ±1–2 segments (+0.0006 / 0.0, intervals include 0). W lowers check native macro-F1 by one clip in
    both seeds and I in seed 2 (−0.005).
- **Interval-label accuracy on the fixed Base intervals.** It is unchanged on development (s0 0.9723/0.9839,
  s2 0.9694/0.9819). On check it moves by at most 0.0007.
- **Why so little changed.** On the training pool itself, each head relabels only 4–8 of about 7,400 Base runs. Each
  relabelled run is a fix (4–8 of the 256–274 wrong ones), and none breaks (`training_readout.json`).
  - The weights moved by about 10% of their norm.
  - The final retention KL is 0.004–0.006.
  - From the first 135 to the last 270 updates, the mean class term falls only from 0.177–0.186 to 0.164–0.176 (W) and
    from 0.126–0.131 to 0.110–0.121 (I).

## What this round shows

1. **Neither supervision changed the class labels of the Base intervals in any useful number.** Under the fixed recipe
   (2,565-parameter head, lr 1e-4, 1,350 updates, retention KL weight 1), both heads stay close to A08's classifier.
   - They relabel about 0.1% of training intervals and almost none on development.
   - So the round cannot separate interval from window supervision. I − W is 0 because neither moved.
2. **No full-segmentation gain.** The only development change is a two-segment loss in seed 2 selection. The Base
   intervals' own pooled labels are already 97–98% right. The remaining wrong ones are not corrected by this small,
   strongly retained head.
3. **What is closed.** This exact head/objective recipe is closed. It does not show that interval supervision cannot help.
   It shows that a retained linear head on fixed embeddings is too constrained to move these labels. Neither a coefficient
   or learning-rate sweep nor a seed-1 run follows from this round.

## Cost and files

- **CPU.** About 0.3 of 2 h, booked in `runs/video_interval_recognition_v1/ledger.jsonl`, including the agent's active wall clock. Training
  took about 7 s in total.
- **GPU.** 0 of 1 h.
- **Records.**
  - `preparation.json` and `checks.json` (983e863).
  - `training_summary.json` and `train/` (6f6964f).
  - `screen.json`: rows, counts, accounting, comparisons, interval-label accuracy, gate and context rows.
  - `training_readout.json`.
- **Frozen heads.** `models_frozen/video_interval_recognition_v1/{W,I}_s{0,2}` (final.pt sha 33540ad6517e, 297c1d611e75,
  07d09ed2af83 and 616de6ebf08e).
- **Server only.** Interval lists, batches, per-sequence predictions and all identifiers.
