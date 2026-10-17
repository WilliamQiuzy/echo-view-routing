# Fact-check of A08

An independent fact-checker, a separate agent, checked this round after the single selection evaluation, against the
README draft. It took two attempts.

**The first attempt.** One fact-checker and two read-only helper agents started at 17:12 UTC on 2026-10-16. At 18:16:54
all three stopped at once, between tool calls, and the agent harness ended them after ten minutes without progress. The
stall came from the agent infrastructure, not from a step of the check. The fact-checker had written its own CPU-only
scripts. These read the two halves through the half-bounded reader and recomputed the rows, comparisons, accounting,
readings and grids. They had run, but its comparison with `summary.json` had not. It left its scratch directory on the
server. Its three wall clocks are booked in the ledger as 3.161 CPU hours, marked not ok.

**The CPU cap.** The configuration books every agent's wall clock as CPU hours. Under that rule the first attempt took
the ledger to 3.751 CPU hours, over the 3-hour cap. On 2026-10-17 the user kept the rule and authorised up to 1.5 CPU
hours beyond the cap for one lean fact-check agent without helpers.

**The second attempt.** One fact-checker, without helpers, ran from 14:17:29 to 14:34:32 UTC on 2026-10-17 (0.303 hours
by the agent harness). It read the first attempt's two scripts in full and confirmed that they use their own decoder,
matching, bootstrap, accounting, readings and goal code. They use the repository only for the half-bounded reader and the
configuration. It re-ran both scripts, which gave identical outputs, and finished the comparison. It then deleted the
server scratch directory.

**How it worked.**
- It ran no stage of the round and wrote nothing into either repository, `runs/`, `checkpoints/`, `models_frozen/` or
  `logs/`.
- It printed no recording, session, stream or recipe id, and it read neither the test split nor EchoNet-Pediatric.
- **GPU.** It used one job of 43.9 seconds (0.012 GPU hours) from 14:28:03, after `nvidia-smi` showed no compute process,
  with the memory fraction set to 0.6.

**Verdict.** The round passes the fact-check with corrections, and no numeric error was found.
- 16,910 of the 16,918 values of `summary.json` were recomputed or compared with their sources, with 0 mismatches. The
  eight values not compared are six output labels and two decoding times.
- About 750 numbers of the README draft match `summary.json` and the server records.
- The README had four low and three nit-level wording issues. All were corrected before publication (below).

## What was verified

1. **Recomputed results (15,583 values, largest difference 0.0).** These are the system rows, TP/FP/FN, transcripts,
   switches, join cuts, all comparisons, the accounting, native changes, shared cases, verdicts, pooled counts, the
   goal, the trade-off cells, the consistent list, the decision, the half counts and the choices.
   - The code reproduces V01's frozen Ordered rows (84 fields per half) and A07's F rows (144 fields per half) exactly.
     A08's F rows equal A07's.
2. **Values compared with server records (1,327 values, 0 mismatches).** These are the training records (960 and 150
   against the training logs), the runs (42), the identities (47), the cost (104 against the ledger) and the parity
   records (24).
3. **Order and integrity.**
   - The commits 6cbb67b, 19ff07a, 042b7ef, 170017c, 39bb154, eb937a0, 7ef1670 and c99bb6af are linear, at the stated
     times. The configuration has not changed since 19ff07a, and the task sheet equals its pin (db8bda36).
   - The server copies of the round's files equal their commits. A07's five reused files are unchanged.
   - The training code digest (56224ab7) and the report code digest (8548d580) recompute.
4. **Runs.**
   - The six run identities recompute. They equal the run manifests, the queue status, the frozen manifests,
     `choices.json`, `checks.json` and `summary.json`.
   - Every run has 10 epoch records and log lines, no resume and the expected final learning rates. Its epoch input
     digests equal A07's (60 of 60), and its start weights are the V01 Ordered states.
   - Each final sha256 is the same in the checkpoint, the frozen copy, the run manifest, the frozen manifest, the queue
     status and the summary. The frozen directories have no write bit.
5. **Masks.** Recounted from A07's manifests, the edge masks equal the preparation record for all three seeds. Every
   native group has three selected edges, no group lacks one, and the repeated native targets are 311, 329 and 335.
6. **The barrier.**
   - c99bb6af adds exactly `choices.json`. Its own SHA-1 computation of the commit, the four trees and blob 924efbc9
     matches. The file's sha256 (b2a731e5) is the same locally, in both server copies and in the proof.
   - The proof was made at 16:45:15, after the commit (16:45:02) and before the single opening at 16:45:43.
     `done.json` was written at 17:05:52.
   - Its own 40-setting grids from the cached calibration probabilities equal the stored ones (3,120 values, 0
     mismatches). The choice rule gives the stored choice for all six models.
7. **Ledger.** Through the selection stage the ledger has 19 lines, all ok, with 3.3917 GPU and 0.5899 CPU hours. The
   launch projection (4.4496) and the six queue projections (4.4498 to 4.4201) recompute.
8. **Private ids.** 1,095 development ids (513 on calibration, 582 on selection) were searched in seven result files,
   with no hit.
9. **Loss equations.** Its own S_A, S_M and L = CE + 0.15 S matched `within_view.objective` on 600 random and adversarial
   cases, with the cap active in 450 of them. The largest differences were 8.9e-16 in the loss and 3.0e-8 in the
   float32 logit gradient. The 73 masked cases without a selected edge gave exactly zero with finite gradients. The
   detach held.
10. **GPU re-inference.** The six frozen models, sha256-checked and strictly loaded, ran on 12 sequences per half (174
    calibration and 187 selection windows). Every argmax equalled the cached one. The logits differed by 0.0007 to
    0.0016, because the job ran with PyTorch's default TF32 setting. A probe after the fact-check re-inferred the six
    models under the round's determinism settings (TF32 off, deterministic cuDNN) on the first 64 calibration windows.
    It reproduced the cached logits exactly (0.0) with equal argmax. The probe took 10 GPU seconds and is booked in the
    ledger.

## Corrections made

**Low**
1. **A causal heading.** Section 2's heading said that both new arms "remove" F's seed-2 over-segmentation. It now says
   that on selection in seed 2 neither new arm shows it.
2. **A5C read as A4C.** The draft said no other B3 system read an A5C clip as A4C. At the stored Ordered setting, A, M
   and F each read one in seed 2 on calibration. The README now says so.
3. **M's different-label penalty.** It also fell, from 7.25 to 7.55 in the first epoch to 5.23 to 5.89. The draft said
   it "stayed".
4. **Join cuts.** No session count exists for them. The small-count limit now gives at most 7 join cuts per seed and
   half instead.

**Nits**
5. "Native macro-F1 at the floor" now reads "at or above the floor (0.973 on selection, 0.980 on calibration)".
6. In seed 1 on selection, A and M also read correctly one A4C clip that frozen Ordered reads as A5C, and one that F
   reads as A5C. The README now says so.
7. The queue's launch time (13:27:17) is the time of the launch command. Its first kept record is at 13:27:21, and the
   README now says so.

The placeholders for the fact-check's cost and section were filled, and the cost section now notes the three lines of
the stalled attempt.

## Not done

- The GPU replay of update 0 of each arm. The check record gives 0.15 S = 0.950 and 0.982 against a CE of 0.115.
- Recomputing the training penalties from forwards, which needs about 0.5 GPU hours.
- Re-running the parity of V01 and F; their records were compared instead.
- Running the failing tests of 042b7ef. The fact-checker confirmed statically that they import three modules missing
  from that tree.
- The task sheet's 1e-5 tolerance, the sentence that most of the penalty comes from non-target classes, and the launch
  figures at 13:33:04, which the README says were not kept.
- `report_tables.md` beyond the id scan.

## Time

- **Attempt 1.** 17:11:59 to 18:16:54 UTC on 2026-10-16 for the fact-checker (1.082 hours), and 1.036 and 1.043 hours
  for its two helpers. That is 3.161 CPU hours, of which 0.745 hours were commands. All are marked not ok.
- **Attempt 2.** 0.303 hours on 2026-10-17, booked as 0.291 CPU and 0.012 GPU hours.
- **The probe.** 10 GPU seconds and its CPU part, on 2026-10-17.
