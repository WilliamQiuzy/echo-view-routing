# A28: interval recognition with a frozen against an adapted video representation (2026-10-24)

**Question.** Each interval contributes the same video evidence from inside itself. Does letting the recognition branch's
video representation adapt classify those intervals better than the frozen representation, and does that improve the
final full segmentation?
- **Branch.** The original A08 + B3 cut path is kept unchanged. Recognition runs on an independent copy of A08 all_s{k}.
- **Arms.**
  - Frozen trains only the copied classifier (2,565 parameters).
  - Adapted also trains the last MultiscaleBlock (`encoder.blocks.15`, 7,100,160), the final norm (`encoder.norm`, 1,536)
    and the 768→512 projection (`encoder.head.1`, 393,728), for 7,497,989 in total.
  - Both run the branch in eval mode, with the same clips, batches, plain interval CE and schedule. Neither has KL or a
    teacher.
- **Scope.** The main attribution is Adapted − Frozen.
- **Data limits.** The development halves are reused exploratory feedback. The A15 check pool's recordings trained the
  encoder, so it is not a holdout. Test and Pediatric were not read.
- **Predecessors.**
  - R06 compared a frozen with a last-block span classifier on the E1 image encoder, A4C/A5C only; neither improved.
  - A27 trained a head on frozen window embeddings with a retention KL. It barely changed labels, and its causes were not
    established.
  - This round changes the input (one 16-frame clip per interval, five classes) and keeps the cut path separate.
- Configuration: `configs/experiments/video_interval_visual_adaptation_v1.yaml` (948eda4), fixed before any A28 output.
  Task sheet `claude_prompts/2026-10-23-video-recognition-finetune.md`.

**Design.**
- **Clip rule.** For interval [a, b) of length L, slot k shows sample a + ⌊(2k + 1)L / 32⌋.
  - A long interval is covered evenly. A short one repeats samples in time order, and every request lies inside the
    interval.
  - The marker sits at A08's rule for the target sample s[8].
  - Each slot uses its own frame and gamma with A08's preprocessing. An unreadable frame would abort the stage, and none
    did.
- **Training intervals.** A27's lists on A15's large fit pool, rebuilt with sample ranges and checked equal.
  - T is 7,242 reference runs. P is 7,411 and 7,366 Base runs labelled by their majority reference class.
  - In T, 1,503 runs are shorter than 16 samples; in P, 1,932 and 1,893. About 4,190 P clips span two recordings, because
    Base runs cross joins.
- **Batches.** A27's 1,350 × (16 T + 16 P) batches, for both arms.
- **Optimiser.** AdamW with weight decay 1e-4; classifier lr 1e-4, Adapted representation lr 1e-5. Warm-up 135 then
  cosine, clip 1, final weights only.
- **Frozen arm.** Trained on its cached frozen clip embeddings, which equal the online path (checked).
- **Deployment.** Each Base run gets one clip and one branch forward and takes its argmax class. Equal neighbours merge,
  and complete sequences are scored.
- **Four rows.**
  - Base is A08 + B3.
  - Init is the new clips with the untrained copy. It equals Frozen and Adapted at zero updates.
  - Frozen and Adapted are the two trained arms.
- **Order.**
  - The code, preparation and checks were committed in 5d1e533.
  - Training ran as chain PID 610660 (Adapted s0 as Python PID 616036, 1,350 real updates per run).
  - The final weights were committed in c5fddb8 before evaluation.
  - The development halves and check pool were evaluated once.

**Checks passed** (`checks.json`).
- Every clip position lies inside its interval, and the pixels equal A08's decode. A run crossing a join reads from its
  two recordings.
- The split prefix/suffix forward equals the full forward exactly (0.0).
- The four Base rows reproduce, and the A08 file is unchanged.
- The trainable sets are 7,497,989 and 2,565, and the optimizer binds exactly those. The frozen prefix is unchanged after
  training.
- With representation lr 0, the Adapted path equals the Frozen head within 4.1e-6 (the head itself moved 9.9e-4).
- A 3 + 3 resume equals 6 exactly. On a fixed batch both arms learn, and Adapted's representation moves.
- **Engineering faults before training** (booked, no result read).
  - The configured A27 path carried its description, so the code now reads the text before the first space.
  - Importing A27's CPU-only trainer hid the GPU, so the first check failed (0.003 GPU h). `CUDA_VISIBLE_DEVICES` is now
    set before that import.

## Results (seeds 0 and 2)

| System | s0 sel | s0 cal | s2 sel | s2 cal | Check s0 / s2 | Native macro-F1, cal s0 / s2 |
|---|---|---|---|---|---|---|
| Base (A08 + B3) | 0.9563 | 0.9827 | 0.9564 | 0.9837 | 0.9603 / 0.9555 | 0.9929 / 0.9856 |
| Init (new clips, untrained copy) | 0.9483 | 0.9817 | 0.9552 | 0.9846 | 0.9591 / 0.9537 | 0.9856 / 0.9929 |
| Frozen | 0.9483 | 0.9817 | 0.9523 | 0.9827 | 0.9601 / 0.9552 | 0.9788 / 0.9788 |
| Adapted | 0.9483 | 0.9817 | 0.9543 | 0.9827 | 0.9601 / 0.9564 | 0.9856 / 0.9788 |
| A27 W / I (stored) | 0.9563 | 0.9827 | 0.9534 | 0.9837 | 0.9609 / 0.9555–0.9560 | – |

**Fixed gate: not met by either arm**, so A28 closes with two seeds and no seed-1 run.

| Part | Frozen | Adapted | Required |
|---|---|---|---|
| Mean dF against Base | −0.0035 | −0.0030 | ≥ +0.003 |
| Half means (selection / calibration) | −0.0061 / −0.0010 | −0.0051 / −0.0010 | each ≥ −0.005 |
| Lowest cell | −0.0080 (s0 selection) | −0.0080 (s0 selection) | ≥ −0.015 |
| Native macro-F1 difference | −0.0074 | −0.0057 | ≥ −0.003 |
| Boundary recall difference | −0.0007 | −0.0007 | ≥ −0.010 |
| Missed short true segments (four cells) | 26 against Base's 22 | 24 against 22 | ≤ Base + 3 |
| Calibration feasibility (native ≥ 0.980) | fails both seeds (0.9788) | fails seed 2 (0.9788) | each seed |

**Adapted − Frozen** (multi segment F1@50, 95% paired intervals).

| Seed | Selection | Calibration | Check |
|---|---|---|---|
| 0 | 0.0000 [−0.0059, +0.0059] | 0.0000 | 0.0000 |
| 2 | +0.0020 [−0.0040, +0.0097] | 0.0000 | +0.0011 [−0.0004, +0.0033] |

- The direction is never negative, but every interval includes 0. Against Frozen, Adapted recovers 0–3 and loses 0–1
  matched segments per cell.
  Native macro-F1 is +0.0068 in seed 0 calibration, one clip.
- **Not an established adaptation effect.** The attribution rule needs the same sign on both halves and check with
  supporting intervals.

**The new input is the main common change** (Init − Base): −0.0080 and −0.0010 in seed 0, −0.0012 and +0.0009 in seed 2,
and −0.0012/−0.0018 on check. Frozen − Init and Adapted − Init are small (−0.0029 to +0.0027).

**Labels of the fixed Base intervals** (`interval_table`; majority reference class).
- **Development.** Accuracy stays at or slightly below Base's for both arms.
  - Seed 0 selection: 0.9693 against 0.9723 (Adapted fixes 2, breaks 5).
  - Calibration: 0.9818 against 0.9839.
- **Check.** About level: 0.9690–0.9701 against 0.9693–0.9694 (fixes 7–9 in 10–14 recordings, breaks 6–8).
- **A4C/A5C.** The interval-level confusions move by up to 5 per cell, in both directions. For example, A5C runs read as
  A4C rise from 11 to 15–16 on seed 0 selection and from 55 to 60 on seed 2 check.

**Training-pool readout** (`training_readout.json`; training data only).
- **Init.** The untrained copy on the new clips already changes 40–41 P labels: 25 fixes and 12–13 breaks of the 256–274
  wrong Base runs.
- **After training.** Frozen fixes 28–30 and breaks 14. Adapted fixes 31–32 and breaks 10–13, at accuracy 0.9656/0.9681
  against Base 0.9630/0.9652.
- **What training corrected.** On its own training intervals, training corrected about 3–7 more wrong runs than Init. It
  did not correct the high-confidence wrong classes in any useful number.
- **Final loss.** Last-270 CE is Frozen 0.080/0.063 and Adapted 0.076/0.059.

## What this round shows

1. **Adapting the last block did not beat the frozen representation by an established margin.** Adapted − Frozen is 0
   to +0.0020 with intervals including 0. On its training intervals, Adapted fixes only 2–3 more wrong runs than Frozen.
2. **Neither arm improves full segmentation.**
   - The single-clip interval input, shared by both arms, is itself a little worse than Base's pooled window probabilities
     on development (Init − Base down to −0.0080).
   - Calibration native recognition falls to 0.9788 in three of four arm-seed cells.
   - Training does not make up for this.
3. **What is closed.** This configuration is closed: one evenly spaced 16-frame clip per Base run, a copied A08 branch,
   classifier-only against last block, norm and projection, plain CE, these learning rates.
   - It does not show that representation adaptation cannot help recognition.
   - It does not establish why the labels barely move. Candidates such as the training pool being A08's own training data
     (Init CE on training intervals is already low), the clip rule or the learning rates were not separated.
   - Neither a variant nor a seed-1 run follows.

## Cost and files

- **GPU.** About 0.75 of 6 h.
  - Checks: 0.025 h, including the failed launch.
  - Clip embedding of the training intervals: 0.13 h.
  - Adapted training: 0.40 h (about 12 minutes per seed).
  - Readout: 0.07 h.
  - Evaluation: 0.12 h.
- **CPU.** About 0.7 of 3 h, booked in `runs/video_interval_visual_adaptation_v1/ledger.jsonl`, including the agent's
  active wall clock.
- **Deployment cost.** The two-branch system runs A08 once per request window, plus one branch forward per Base run
  (0.016 GPU s per clip, 34.6 M parameters). It is not a one-forward model.
- **Records.**
  - `preparation.json` and `checks.json` (5d1e533).
  - `training_summary.json` and `train/` (c5fddb8).
  - `screen.json`: rows, counts, interval tables, comparisons, accounting, gate and context.
  - `training_readout.json`.
- **Frozen branches.** `models_frozen/video_interval_visual_adaptation_v1/{Frozen,Adapted}_s{0,2}`. Each holds the
  trainable tensors only, applied onto A08 all_s{k}: Frozen 7bd949e2888a and 53aae9b59f77, Adapted 236040fbaba4 and
  5e8319c1b10a.
- **Server only.** Clip lists, cached clip embeddings, per-sequence predictions and all identifiers.
