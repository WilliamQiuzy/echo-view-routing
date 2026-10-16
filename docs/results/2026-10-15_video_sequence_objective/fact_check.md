# Fact-check of A07

An independent fact-checker, a separate agent, checked this round from 08:55:18 to 09:19:45 UTC on 2026-10-16. That was
after the single selection evaluation, against the README draft.

**How it worked.**
- It ran no stage of the round. It wrote nothing into either repository, `runs/`, `checkpoints/`, `models_frozen/` or
  `logs/`. It used a server scratch directory, deleted at the end.
- It printed no recording, session, stream or recipe id, and it read neither the test split nor EchoNet-Pediatric.
- It wrote its own code for the statistics:
  - the sampler and the stride-5 grid of the training manifests;
  - L_F, L_B and L_O, with every placement of a transcript over the four windows enumerated;
  - B3, argmax and the stored-setting decoding;
  - segment matching at IoU 0.5, boundary matching at 5 samples, the native clip-majority rule and macro-F1, the
    per-class precision and recall, TP, FP and FN, switches, same-view join cuts and the transcript edit distance;
  - the paired bootstrap (2,000 draws, seed 20261002, native clips by session and streams by whole stream, the same draws
    for every system) and the consistency rule;
  - the goal and floors, the trade-off cells and all readings;
  - the recovered and lost accounting with its session counts;
  - its own walk of the git objects.
- It read the development halves only through the repository's half-bounded reader and the saved private outputs.
- **Helpers.** Two read-only helper agents ran inside its window under the same rules. One checked the git integrity,
  the barrier and the ledger, and the other compared the README with `summary.json`. The fact-checker re-checked each of
  their findings itself.
- **GPU.** It used one 66-second job (0.018 GPU hours) from 09:12:09 to 09:13:16, after `nvidia-smi` showed no other job,
  with the memory fraction set to 0.6. The peak allocation was 8,851 MiB.
- **Time.** The agents' wall clocks are booked separately in the round's ledger as CPU hours: the fact-checker 0.408
  hours besides its GPU job and 0.016 hours for one follow-up answer, and the helpers 0.209 and 0.284 hours.

**Verdict.** The round passes the fact-check with corrections, and no numeric error was found.
- 17,037 of the 17,039 values of `summary.json` were recomputed or compared with their sources, with 0 mismatches. Every
  float is bitwise equal, including all 2,592 bootstrap values. The two frame-decoding times cannot be recomputed.
- About 720 README values match `summary.json` as rounded.
- A GPU re-inference matched the cached logits of all nine models within 7.6e-6, with equal argmax.
- The README had two medium and nine low wording or flagging issues and several nits. All were corrected before
  publication (below).

## What was verified

1. **Order and integrity.**
   - **Commits.** The chain 5110b12, 1a76407, 3b92fe8, 095320c, f39c4b3, dcae2cd, 5ec611c and a96dfc50 is linear, with
     one parent each, at the stated times. The configuration (f8d50e07) is touched only by 1a76407, and the task sheet
     equals its pin (4418f3e6).
   - **Red tests.** At 3b92fe8 the three test files fail at collection, with ModuleNotFoundError for each of the three
     round modules.
   - **Server code.** The server copies of the nine round files equal the commits, and all 18 helper files equal their
     pins.
   - **The report fixes.** From 095320c to 5ec611c the report differs only by `archived_pin` and its use. The training
     and preparation code and `echo_routing/` are unchanged.
   - **The superseded choices.** The superseded record (c7763b22, report code 6dbea9c8) differs from the committed one
     (b4aa8fdc, report code 1894d3b4) in exactly two fields, the report-code digest and the frame-decoding time.
2. **Identities and runs.**
   - Its own rebuild of the identity payload and the training code digest gives 1733e5e660b3, as recorded. 63 identity
     comparisons across the run manifests, the queue status, the frozen manifests, `choices.json`, `summary.json` and the
     check record all agree.
   - Every run has 3,750 updates from update 0, 10 epoch records equal to its training log and 10 epoch log lines, with
     no interruption or resume.
   - The encoder stayed unchanged through the warm-up. All 90 stored learning rates equal 0 before update 375 or 1e-5
     times the cosine factor after it.
   - The final sha256 values agree across the checkpoints, the frozen copies, the queue status, `summary.json` and
     `choices.json`. The frozen directories have no writable entry.
   - The model has 34,623,872 encoder and 16,677 new parameters, 34,640,549 in all.
3. **Grouping.**
   - Its own sampler re-derived all 3,750 updates of every seed, which is 90,000 windows. It compared the sequence, start,
     centres, targets, markers, the 16 sample indices, the frame references, the kind and the repeat flags, with 0
     mismatches.
   - The three arms of each seed share one manifest sha256 and identical input digests.
   - The preparation counts recompute. They are 33,049 native starts with 230 repeating clips, 1,478 eligible streams with
     14,153 starts, 0 skipped runs (the shortest bank run is 10 samples) and repeated native targets of 311, 329 and 335.
4. **Gradient math.**
   - Its own L_F, L_B and L_O matched `segment_objective.objective` on 336 cases. 300 cases were random. The other 36 were
     adversarial: one class, X-Y, X-Y-X, X-Y-X-Y, X-Y-Z, repeated targets and two changing groups, at logit scales up to
     60.
   - The largest differences were 4.9e-16 (relative) in value and 1.7e-15 in the logit gradients. The single-class
     identity spread was 3.6e-15, with a gradient difference of 0. A finite-difference check of its L_O agreed within
     4e-9.
   - **A GPU replay of update 0 of seed 0** ran in training mode with the V01 source weights, manifest row 0 and the
     recorded random seeds. It gave the losses F 0.11502886, B 0.10221063 and O 0.06986854. These differ from the check
     record by 0, 0 and 6.9e-17. The formal runs store epoch means only, not per-update losses.
5. **Results.**
   - From the cached probabilities, V01's saved logits and the bounded reader, it recomputed 16,277 values. These were
     the system rows, TP/FP/FN, transcripts, switches, join cuts, all comparisons, the accounting, native changes,
     verdicts, pooled counts, the goal, the decision and the half counts.
   - It compared another 760 values with their sources: the choices, archived rows, runs, last epochs, identity, cost,
     parity and list fields.
   - The cached windows, identities and probabilities equal its own mapping, and V01's 12 stored rows (168 fields) were
     reproduced.
6. **GPU re-inference.** All nine frozen models were sha256-checked and strictly loaded. They were re-run on 60
   sequences: 4 native, 3 pair and 3 multi per seed and half, 2,511 windows in all. The largest logit difference was
   7.6e-6, and the argmax was equal everywhere. V01's parity record (0.0 on 64 windows per seed) was confirmed.
7. **The barrier.**
   - a96dfc50 adds exactly `choices.json`. Its own SHA-1 computation of the commit, the four trees and blob 8490a018
     matches. The blob's sha256 (b4aa8fdc) equals both server copies and the proof.
   - The proof was made at 08:21:40, before the single opening ("first opening") at 08:22:11, and `done.json` was written
     at 08:45:23.
   - The choice rule re-applied to each stored grid gives the stored status, setting and feasible count. Its own
     40-setting grids from the cached calibration probabilities equal the stored ones (9 × 520 values, 0 mismatches).
8. **Cost.**
   - The ledger then had 32 lines, with GPU 5.000071 hours and CPU 0.698035 hours. The GPU time splits into the check
     0.087183, the trainings 4.265829 and the inference stages 0.647059, which includes 0.011546 for the failed dry run
     and rerun.
   - The launch projection was 5.917748 of 8. The nine queue projections ran from 5.918 to 5.862, all within the cap.
9. **Private ids.** 1,085 development ids were checked: 521 recordings or native clips, 72 sessions and 492 streams. None
   appears in `README.md`, `summary.json`, `report_tables.md`, `choices.json`, `code_review.md`, `preparation.json` or
   `checks.json`.
10. **Wording.** The README follows the rules. No absent consistent difference is called equality, and constructed
    streams are not called scans. Sessions are not called patients, and the development halves are called reused. The
    findings below concern descriptions and flags.

## Corrections made

**Medium**
1. **Seed 2 on calibration.** The draft said that seed 2's continued models cut more often on selection and not on
   calibration. With B3 they also cut more on calibration: multi FP segments 18, 18 and 19 against 11, with intervals
   above zero, and native switches 15, 14 and 16 against 3. At the stored setting the FP counts are 15 to 16 against 11,
   with intervals starting at zero. With argmax they are lower, 24 against 30. None of this comes with a segment F1@50
   gap. Sections 4 and "What this round does not show" now say so.
2. **Session flags.** Section 3 now gives the sessions behind the added and removed predicted segments. On selection
   they come from 10 to 17 sessions per seed. On calibration O's come from 10, 2 and 12 sessions (added) and 7, 5 and 8
   (removed), and B's and F's from 2 to 11. Section 6 now gives the sessions behind the cut same-view joins: 2 to 10 per
   system, seed and half, fewer than 10 in 20 of 24 cells. The "Small counts" limit now covers both.

**Low**
3. **Runs, not classes.** The sampled orders have two runs (3,631 to 3,640) or three runs (110 to 119). Of the three-run
   orders, 17, 26 and 20 are X-Y-X with two classes, so 102, 84 and 91 hold three classes.
4. **A missing interval.** Seed 0's calibration interval of O − frozen Ordered in multi FP segments, [−1, +7], also
   includes zero.
5. **The verdict count.** There are 216 verdicts: three decoders, six comparisons and six metrics, in two halves. The
   two further bootstrapped metrics carry no verdict, and none of their 72 interval sets is consistent.
6. **A5C read as A4C.** The stored-setting rows of seed 2 each read one A5C clip as A4C on calibration. No B3 or argmax
   row does.
7. **Denominators.** Each seed's multi reference holds 502 segments on selection, 141 of them with at most 16 samples and
   7 A4C/A5C X-Y-X middles. On calibration it holds 518, with 134 and 7.
8. **A test change.** 095320c also corrected one test assertion to the configuration's readings and removed two unused
   imports. Its commit message records this, and the README now discloses it.
9. **The review's ledger line.** It was booked by hand with 0.297 hours. Its finished field (03:18:09) is the booking
   time, not the review's end.
10. **The server test runs.** Two runs at 03:16 gave no usable summary line, and a rerun at 03:17:04 exited 0 with 51
    tests. The first two are booked in one ok = false line of 0.007 CPU hours.
11. **Attributive wording.** The O − B conclusion now says only that no consistent difference between O and B was
    detected.

**Nits**
12. The headings of sections 1 to 3 now say "no consistent difference was detected".
13. Seed 0's O − F selection interval [+0.0000, +0.0029] starts at zero. The draft said it ends there.
14. Frozen Ordered's only floor failure is described as lying in a cell where the continued models have 0.9786, not as
    removed by them.
15. "So" became "and" in the sentence on the arms' shared batches. The fact-checker confirmed the claim that follows.
    The median epoch-to-epoch swing of L_F (0.025 to 0.035) exceeds the largest gap between arms (0.005 to 0.009), and F
    and O correlate at 0.992 or more.
16. Update numbers now count consistently.
17. The timing now gives the check's 0.459 to 0.466 seconds per unfrozen update and the runs' average of about 0.455
    seconds.
18. The two memory figures are now attributed. The peak of 12,380 MiB is torch's allocated peak, and 15,077 MiB is
    nvidia-smi's reading.
19. The second failed calibration run stopped before any model ran. Its "GPU inference" ledger line holds 0.003 hours of
    set-up time.
20. The README explains the figure title of `training_curves.png`. The phrase "dashed: encoder unfrozen" refers to the
    grey line, while the dashed curves are seed 1.
21. In three tables of `report_tables.md` the row keys contain "|", which shifts the Markdown cells. The README notes
    this, and `summary.json` holds the values.
22. The queue's start time of 03:19:12 is the time of the launch command. The first run started at 03:19:15.
