# A09-R1: full-schedule continuation of A09's plain arm (2026-10-18)

**Question.** A09's plain-CE arm C was stopped at update 1,250 of 3,750 by its frozen rule, while the learning rate was
still 0.84 of its peak. A09's `stop_both` stays. R1 is the lead's dated follow-up, not a promotion. It asks whether
completing the unchanged schedule keeps C's native gain and removes its extra short false segments. Training length,
data exposure and learning-rate progress change together.

**What ran.** Seeds 0 and 2 resumed through A09's unchanged trainer and ran updates 1,250 to 3,749
(`configs/experiments/video_discriminative_r1.yaml`).
- **Resume.** The resume restored the AdamW state (1,250 and 875 steps) and the RNG states. The parent identities were
  recomputed with the parent code. The restored model's eval logits equal those of the evaluated pilot snapshot.
- **Evaluation.** The final weights only, on both development halves, with one B3 choice per model on calibration.
- **Reproduction.** The pilot snapshots C1250 reuse their cached outputs and pilot choices, and these reproduce
  `pilot.json` exactly. The V01 parity was 0.0.
- **Seed 1.** It was conditional on a rule fixed before any R1 output. The rule failed, so there are two seeds and no
  three-seed verdict.

## Results (two seeds; exploratory; reused development halves)

Definitions:
- N is argmax native macro-F1.
- E counts native A4C read as A5C and A5C read as A4C, summed over the 4 cells (2 seeds × 2 halves).
- F is multi segment F1@50 with B3 at the model's calibration choice. "F fixed" uses frozen Ordered's stored setting.
- FP is false multi-stream segments, summed over two seeds on selection; "short" means at most 16 samples.
- FN is missed multi-stream segments over the 4 cells.

| System | Mean N | E (A4C→A5C / A5C→A4C) | Mean F | Mean F fixed | Selection FP (short) | FN (short) | Mean boundary recall |
|---|---|---|---|---|---|---|---|
| Frozen Ordered | 0.9781 | 10 / 0 | 0.9661 | 0.9661 | 54 (29) | 62 (29) | 0.9736 |
| A08 A | 0.9903 | 4 / 0 | 0.9698 | – | 49 (22) | 51 (22) | 0.9793 |
| C1250 (pilot) | 0.9858 | 4 / 2 | 0.9439 | 0.9534 | 141 (93) | 69 (19) | 0.9787 |
| C3750 (R1) | 0.9786 | 8 / 2 | 0.9429 | 0.9540 | 116 (79) | 77 (27) | 0.9787 |

**Per seed, selection.**
- **Seed 0 recovered.** F rose from 0.899 to 0.933, against frozen Ordered's 0.939 and A08 A's 0.956. False segments
  fell from 75 to 39, and short ones from 45 to 18.
- **Seed 2 did not.** F fell from 0.914 to 0.903, against 0.964 and 0.956. False segments rose from 66 to 77, and short
  ones from 48 to 61.
  - Its calibration choice moved to a very low threshold (quantile 0.95, threshold 0.004). Seed 0's choice was quantile
    0.999, close to Ordered's setting.
  - At Ordered's fixed setting, seed 2 reaches 0.940, still below Ordered's 0.964.

**Calibration.** F fell against the pilot in both seeds: from 0.983 to 0.973 and from 0.980 to 0.962. Missed segments
rose from 5 to 11 and from 9 to 12.

**Paired bootstrap per seed.** These are 95% intervals; with two seeds there is no verdict.
- **C3750 − A08 A.** Negative in all four cells, with every interval below zero: −0.023 and −0.053 on selection, −0.010
  and −0.022 on calibration.
- **C3750 − frozen Ordered.** Seed 2 is negative in both halves (−0.061 and −0.018). Seed 0's intervals include zero.
- **C3750 − C1250.** Seed 0 gains on selection, +0.034 [0.010, 0.060]. Every other cell is negative: −0.011 on selection
  for seed 2 (interval includes zero), and −0.010 and −0.018 on calibration (intervals below zero).

**The seed-1 rule failed**, so R1 stops.
- **Recovery fails.** The mean F change against the pilot is −0.001; at least +0.010 was required. Selection FP is 116
  against a limit of 98.7.
- **The strong-reference condition fails.** The mean difference from A08 A is −0.027; −0.005 was allowed.
- **The no-degradation condition fails on missed segments.** There are 77 against a limit of 71.3. Its native, A4C/A5C
  count, boundary-recall and short-segment parts pass.

## What was learned

1. **The pilot's native gain is absent at the final point (an observation).** At update 3,750, A4C read as A5C is back
   to 8, near frozen Ordered's 10, and the 2 A5C read as A4C errors remain. Native macro-F1 equals Ordered's (+0.0005).
   Training length, data exposure and learning-rate progress changed together, so the cause is not identified (corrected
   on 2026-10-18; see below).
2. **Finishing the schedule does not reliably remove the fragmentation.** Seed 0 approaches Ordered on selection; seed 2
   does not. Both seeds lose segments on calibration. Part of seed 2's loss is its decoder working point, but at Ordered's
   fixed setting the mean F is still 0.012 below Ordered.
3. **This recipe ends here.** The recipe is class-balanced cross-session native windows (10 of 14 per update) with 4
   constructed replay windows and plain CE. It keeps no lasting native gain and segments clearly worse than A08 A. The
   completed schedule closes this complete recipe only. Which of its parts, such as the 10-of-14 native share, causes the
   losses is not identified, and discriminative training in general is not closed.
4. **Segment-level counts worsened.** False and missed cuts rose in both halves. These counts alone do not show that
   window-level discrimination is no longer a bottleneck (corrected on 2026-10-18; see below). A08 A remains the
   strongest model.

## Cost and files

- **GPU.** 1.246 h of 4: check 0.003, the two continuations 1.110 and the evaluation 0.133. A09's 1.341 GPU h is
  unchanged.
- **CPU.** Booked in `runs/video_discriminative_r1/ledger.jsonl`, including the agent's active wall clock.
- **Frozen models.** `models_frozen/video_discriminative_r1/plain_s0` (sha256 8ddafa44…) and `plain_s2` (93de2c76…).
- **Records.** `two_seed.json` (rows, counts, choices, comparisons, accounting and the seed-1 decision) and
  `checks.json`. Per-window outputs stay on the server.
- **Sealing and commit.** The decision, configuration and code were stored on the server with their sha256 before the GPU
  started (`runs/video_discriminative_r1/decision_files.sha256`). The local branch head's object was unreadable under iCloud
  eviction until Codex restored it on 2026-10-18. The round's files were committed locally after that, which was after the
  launch. Nothing was pushed.

## Dated corrections (2026-10-18, after the lead's review)

Each item gives the original statement, then the correction and its reason.
1. **The cause of the lost native gain.** The original said: "The mid-schedule state, not the recipe, carried the
   reduction." This is now an observation only. The gain seen at the pilot is absent at update 3,750. Training length,
   data exposure and learning-rate progress changed together, so the high learning rate is not shown to have caused the
   pilot gain.
2. **What the segment counts show.** The original said: "It shows as false and missed cuts in both halves, not as
   window-level A4C/A5C discrimination." The worse FP/FN counts do not by themselves show that window discrimination is
   no longer a bottleneck. They also do not show that the 10-of-14 native share caused the losses. Only the complete
   recipe is closed.
3. **The local commit.** The original said: "The local commit came after the launch." When this was written, no local
   commit existed. The files were sealed on the server with their sha256 before the launch, and the local commit was
   pending; it was made afterwards. A README must not report a commit that does not exist.
