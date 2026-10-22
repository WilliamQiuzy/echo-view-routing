# A23: training the video model on same-target context interventions (2026-10-22)

**Question.** Can the prediction for a given target time be made robust to interfering context while keeping the stable
correct predictions and the true short segments? A23 trains the A08 video model on copies of its own training targets.
Each copy keeps a protected local run of real target frames, and the context outside it is replaced by another view's
real frames.
- **Other (treatment).** The donor context comes from a training recording of another view.
- **Same (matched control).** The donor context comes from a recording of the target's own view and another session.
- The two arms share the protected runs, replaced slots, seams, offsets, extra samples, loss, compute and schedule. Only
  the donor's view differs, so Other − Same isolates training under interfering-view context.
- Same − Base is the common effect of copies and continuation. Long10 R (A10/A18 R), A14 P and A20 H are frozen
  background rows.
- **History.** R10 (2026-10-05) probed the frozen B7 with paired same-view/other-view context replacement and found that
  other-view context pulls the centre. Its proposed training was never run. No earlier round trained a video model under
  same-target context interventions.
- **Not rerun.** A08's neighbour smoothing, A02's dense token supervision, A20's replay, A21's spatial branch and A22's
  short support are different conditions.
- **Unchanged at deployment.** The input layout (16 frames at 10 Hz, target marker), the parameters and the single
  encoder forward per window are A08's.
- **Data limits.** Copies contain synthetic recording joins and are training inputs, not continuous scans. The check
  pool is not a new holdout, and the development halves are reused exploratory feedback. The test split and
  EchoNet-Pediatric were not read.
- This is a conventional counterfactual-augmentation test, and no algorithmic contribution is claimed.
- Configuration: `configs/experiments/video_context_intervention_v1.yaml` (a2ffa24), fixed before any A23 output.

**Design.**
- **Batches.** Each update has A10's 12 pinned windows (rows 0–1,349, seeds 0 and 2) plus one copy of each, 24 windows
  in one forward and backward pass.
- **Protected run.** The run is taken around the marker slot m inside [m − 4, m + 3]. It holds the slots that carry the
  target's label and stay continuous in the original stream, with no fragment join and positions equal or consecutive.
  It is 8 slots in 66% of copies and 7.25 slots on average, so 8.75 of 16 slots are replaced on average. The target
  sample is never altered.
- **Donor.** A protocol-train native recording, drawn with numpy default_rng([20261022, 23, seed, u, w]). Both arms use
  the same uniforms.
  - The other slots take one contiguous run of the donor's own 10 Hz plan in slot order, at gamma 1.
  - Other's donors always carry another label; Same's donors always carry the target's label and come from another
    session. Each arm draws about 3,300 distinct recordings.
- **Loss.** On the 12 originals, mean CE + KL to the own A08 (A18's cached teacher). On the 12 copies, mean CE with the
  target label. There is no teacher on copies and no other term.
- **Recipe.** A10's: 1,350 updates; the encoder and its projection run at 1e-5 and are frozen for the first 135
  updates; marker and classifier 1e-4; the head group is kept for layout; weight decay 1e-4 and clip 1.
- **Order.**
  - The run records, evaluator and replay check were committed in 4196e05 at 19:28:50 UTC. Calibration was then started
    in a separate command at 19:29:00.
  - The sensitivity choices were committed alone in 9f50531, with the git object's sha verified, before selection or
    the check pool was read.
  - The check pool was evaluated once.
- **Checks passed** (`checks.json`).
  - Legality for all 16,200 copies per seed (0 violations): protected labels and continuity, the marker inside the run,
    one contiguous donor run at gamma 1, and the donor label and session relation.
  - Pixels of real copies through the loader: protected slots bit-equal to the original window's, and context slots
    equal to independently decoded donor frames.
  - Teacher parity 0.0, and forced Other = Same reproduces exactly.
  - Resume within bounds. Encoder gradients are 0 before the warm-up and non-zero after.
  - Timing was 1.42 s per update, with a 36.4 GB peak.
- **One failed launch.** In the first queue, Other_s0 trained completely. Same_s0 then stopped after 1.5 minutes
  (0.025 GPU h, booked as failed). The frame-path table had been cached from Other_s0's manifest, so a Same donor
  recording had no listed directory.
  - The fix (4196e05) lists every protocol-train clip once and changes no computation.
  - Same_s0, Other_s2 and Same_s2 then ran from update 0 with the fixed code. Other_s0 keeps its pre-fix code digest.
  - The fixed code replays Other_s0's logged losses over updates 0–4 within 2.7e-8 (`replay_check.json`, 0.004 GPU h).

**Training-set readout** (`training_summary.json`).
- Other's copies start hard: 52–54% correct while the encoder is frozen, rising to 94% over the last 270 updates.
- Same's copies are easy throughout (98–99%).
- Other fits the originals slightly worse than Same: CE +0.017 and +0.011, 47 and 41 fewer of 16,200 correct, and KL to
  the teacher 0.055 against 0.02.

## Results

Multi segment F1@50. Development has four cells (2 seeds × 2 halves); the check pool has one cell per seed.

| System (one forward per window) | Development mean | s0 sel, s0 cal, s2 sel, s2 cal | Check s0 / s2 (mean) |
|---|---|---|---|
| Base (A08 + B3) | 0.9698 | 0.9563, 0.9827, 0.9564, 0.9837 | 0.9603 / 0.9555 (0.9579) |
| Long10 R@A08 | 0.9695 | 0.9559, 0.9827, 0.9567, 0.9828 | 0.9606 / 0.9578 (0.9592) |
| **Other@A08** | 0.9545 | 0.9392, 0.9630, 0.9442, 0.9714 | 0.9458 / 0.9487 (0.9473) |
| **Same@A08** | 0.9659 | 0.9540, 0.9790, 0.9519, 0.9789 | 0.9603 / 0.9582 (0.9593) |
| A14 P | 0.9724 | 0.9640, 0.9855, 0.9564, 0.9837 | not evaluated |
| A20 H@A08 | 0.9724 | 0.9577, 0.9866, 0.9644, 0.9808 | not evaluated |

**Promotion rule: not met for either arm.**

| Part | Other@A08 | Same@A08 | Required |
|---|---|---|---|
| Mean dF against Base | −0.0153 | −0.0039 | ≥ +0.003 |
| Half means (selection / calibration) | −0.0147 / −0.0160 | −0.0035 / −0.0042 | each ≥ −0.005 |
| Lowest cell | −0.0197 (s0 calibration) | −0.0048 (s2 calibration) | ≥ −0.015 |
| Native macro-F1 difference | −0.0011 | −0.0016 | ≥ −0.003 |
| Boundary recall difference | +0.0137 | +0.0059 | ≥ −0.010 |
| Missed short true segments | 10 against Base's 22 | 17 against 22 | ≤ Base + 3 |
| Calibration row feasible | s2 fails (native 0.9788) | both seeds | each seed |

So no seed 1 was trained. A23 closes with two seeds and no mechanism, coefficient, teacher, input-branch or schedule
variant.

### 1. Does training under interfering context make the target prediction robust?

**At the window level, yes: the predictions near view changes and in short runs become much more robust.**

| Check pool (842 / 795 A08-wrong windows) | Base s0 | Same s0 | Other s0 | Base s2 | Same s2 | Other s2 |
|---|---|---|---|---|---|---|
| All windows accuracy | 0.9537 | 0.9581 | 0.9647 | 0.9563 | 0.9617 | 0.9655 |
| Mixed windows accuracy | 0.9084 | 0.9178 | 0.9512 | 0.9152 | 0.9284 | 0.9555 |
| Short-run windows accuracy | 0.9150 | 0.9264 | 0.9594 | 0.9199 | 0.9351 | 0.9578 |
| Pure windows accuracy | 0.9763 | 0.9781 | 0.9715 | 0.9767 | 0.9783 | 0.9705 |
| A08-wrong windows now right | – | 104 | 359 | – | 130 | 356 |
| A08-right windows now wrong | – | 24 | 158 | – | 31 | 188 |
| A08-wrong short-run windows now right | – | 15% | 64% | – | 24% | 66% |

- **Robustness gained.** Other fixes 43–45% of the A08-wrong check windows, against Same's 12–16%. It fixes about 2/3
  of the A08-wrong windows in short runs. Development is similar: 39–45% against 9–20%.
- **Net windows.** Other's overall window accuracy and CE are the best of the three. On check it has 201 and 168 more
  net-fixed windows than Base.
- **Pure windows.** Pure-window accuracy falls, from 0.978 for Same to 0.971 for Other, and Other breaks 6–7 times as
  many A08-right windows.

### 2. Does Other beat the matched Same control in full segmentation?

**No. Other is consistently worse.**

| Other − Same | s0 selection | s0 calibration | s2 selection | s2 calibration | Check s0 / s2 |
|---|---|---|---|---|---|
| Multi segment F1@50 | −0.0147 [−0.0327, +0.0028] | −0.0160 [−0.0313, −0.0038] | −0.0077 | −0.0075 | −0.0145 [−0.0210, −0.0085] / −0.0095 [−0.0153, −0.0038] |

- **Every cell is negative.** All six cells are negative, the mean is −0.0115 on development and −0.0120 on check, and
  three intervals exclude zero.
- **Own-calibration readout.** It says the same: −0.0118 on development and −0.0169 on check (sensitivity only).
- **Reading.** Training under interfering-view context hurts full segmentation against the same training with
  same-view context. That holds even though it improves window accuracy.

### 3. Does it keep stable predictions and true short segments while removing false ones?

**It keeps the short true segments but adds far more false short segments.**

| Count (short in brackets) | Base | Long10 R@A08 | Same@A08 | Other@A08 |
|---|---|---|---|---|
| Development missed true segments | 51 (22) | 41 (15) | 45 (17) | 45 (10) |
| Development false segments | 72 (39) | 84 (53) | 95 (59) | 145 (113) |
| Check missed true segments | 187 (79) | 160 (62) | 156 (61) | 149 (42) |
| Check false segments | 248 (153) | 263 (180) | 267 (184) | 406 (314) |

- **Other.** It misses fewest short true segments (10 on development, 42 on check). But scattered flips inside pure
  segments become false short segments at the fixed decoder: 113 against Same's 59 on development, and 314 against 184
  on check. Boundary precision falls to 0.933 (development) and 0.917/0.927 (check), against 0.955 and 0.950/0.946 for
  Same.
- **Same.** It is close to Long10 R: development −0.0036 against R, check +0.0000. It is slightly below Base on
  development (−0.0039, all four cells negative) and at Base on check (+0.0014). The common copy and continuation effect
  is therefore about neutral to slightly negative.
- **Native recognition.** Native is close for both arms (−0.0011 and −0.0016). Other_s2 fails calibration feasibility
  (native 0.9788, one more A4C read as A5C).
- **Against references.** Both arms are below A14 P and A20 H@A08 (Same −0.0065 against both).

## What this round shows

1. **Same-target interventions buy the intended window robustness.** Other fixes about two in five of A08's wrong
   windows and two thirds of its wrong short-run windows. Its overall window accuracy is the best of the three.
2. **The robustness costs pure-window stability, and the fixed decoder turns that into false short segments.** Full
   segmentation is consistently worse than the matched Same control: every cell is negative, with −0.0115 on
   development and −0.0120 on check. The system is 0.015 below A08 + B3.
3. **A22 showed the same pattern.** Window gains near view changes come with scattered flips inside pure segments, and
   B3 converts those flips into false short segments. Window accuracy and short-segment recall were not good proxies for
   full segmentation in either round. The flips that matter for segmentation are isolated errors inside stable segments.
4. **What is closed.** This configuration is closed: other-view context outside a protected run of at most 8 slots, CE
   on copies and this schedule.
   - It does not show that context robustness cannot help.
   - It shows that at these settings the gain at view changes was outweighed by instability inside pure segments.
   - Any further attempt would need to keep within-segment stability (A08's own strength) as part of the objective.
     That is a reading, not a proposal.

## Cost and files

- **GPU.** 2.71 of 6 h.
  - Checks: 0.02 h.
  - Training: 2.07 h for four runs, plus 0.025 h for the failed Same_s0 launch.
  - Replay check: 0.004 h.
  - Calibration inference: 0.13 h.
  - Selection and check-pool inference: 0.47 h.
- **CPU.** 0.63 of 4 h at the final commit.
  - This includes preparation (main process), the decode/evaluation workers outside GPU time (0.21 h), the training
    readout, and the agent's active wall clock booked by hand.
  - Waiting is not booked, and GPU loaders are not counted again.
- **Records.**
  - `preparation.json`, `checks.json` and `manifests_index.json` (511debb).
  - `training_summary.json`, `replay_check.json` and `train/` (4196e05).
  - `choices.json` (9f50531).
  - `screen.json`: rows, counts, comparisons, accounting, window groups and cost.
- **Frozen weights.** `models_frozen/video_context_intervention_v1/{Other,Same}_s{0,2}` on the server (final.pt sha
  d4fdcf48ef2f, d594fe90ea27, 6d2571b17ace and ebce3644b18d).
- **Server only.** The manifests (donor mappings), frames, logits, per-sequence predictions and all identifiers.
