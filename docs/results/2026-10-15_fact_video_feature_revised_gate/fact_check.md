# Fact-check of A06-R1

An independent fact-checker, a separate agent, checked this round from 01:02:10 to 01:23:40 UTC on 2026-10-16. That was
after the single selection evaluation, against the README and `code_review.md` drafts.

**How it worked.**
- It ran no stage of the round and wrote nothing into the repository or the round's directories. It used a server
  scratch directory, deleted at the end.
- It printed no recording, session, stream or recipe id. None of 1,095 development ids appears in the README,
  `code_review.md`, `report_tables.md` or `summary.json`.
- It wrote its own code for the statistics:
  - segment matching at IoU 0.5 and F1@50;
  - boundary matching at 5 samples;
  - the native clip-majority rule and macro-F1, cuts per clip and the same-class cut rate;
  - FP and FN counts;
  - the paired bootstrap (2,000 draws, seed 20261002, sessions then streams, the same draws for every system) and the
    consistency rule;
  - the goal and floors and all six readings;
  - the accounting with recordings and sessions, the denominators, tokens and mechanism counts;
  - its own B3 decoder and its own walk of the git objects.
- It used the repository's bounded development reader and A03's half metadata. A01's ASRF and MS-TCN ran through their
  frozen heads and official decoders as the objects under test.
- A helper agent did the local git work for the gate and code checks, read-only. The fact-checker spot-checked its
  findings.
- **GPU.** It used one 50-second job (0.014 GPU hours) after `nvidia-smi` showed no other job, with the memory fraction
  set to 0.6.
- **CPU.** About 0.03 hours of command time. The two agents' wall clocks (0.358 and 0.305 hours) are booked separately in
  the round's ledger.

**Verdict.** The round passes the fact-check, with no high-severity issue.
- Every value of `summary.json` was recomputed: 10,274 values, with 0 mismatches.
- The README tables match `summary.json` in all 336 system cells and all 80 paired-table values.
- A GPU re-inference reproduced all 120 sampled sequences exactly.
- The README had one low numeric error, two medium omissions and several low or nit-level points. All were corrected
  before publication (below).

## What was verified

1. **Order and integrity.**
   - **Commits.** The ten commits from 0053bda to cf3963e are linear, in the stated order and at the stated times. The
     configuration has not changed since 7bd5249 (sha256 d3841a09), and the task sheet equals its pin (55157622).
   - **Red tests.** bfd1dc0's tests fail at import. A copy of that tree on the server gives ModuleNotFoundError.
   - **Server code.** Every file of the round on the server equals cf3963e:
     - the 11 training files;
     - the revalidation, post-training and report modules;
     - the configuration and the tests.
     Of 457 tracked Python and YAML files, the 10 that differ belong to earlier rounds.
   - **Identities.** The run identities recompute from a fresh head. They equal the run manifests, the launch record,
     the evaluation manifest and `summary.json`. Building a model first changes all three, which confirms the disclosed
     cause of the refused launch.
   - **Runs.** Each run has 18,750 updates, no interruption, no resume and 50 epoch records with 50 log lines. The
     learning rate falls from 1e-4 to 1e-5 in epoch 31.
   - **Weights.** The final sha256 values (911a9a64, ff2c6637 and 369f7fc8) equal the checkpoints, the frozen copies, the
     queue status, the evaluation manifest and the summary. The initial states equal F02's (52ba4673, 048357d7 and
     7bb0b685).
   - **Barrier.** cf3963e adds exactly one file, the manifest. The fact-checker's own walk of the git objects (the
     commit, 4 trees and the path) leads to blob 5db60648, which matches both server copies. The proof verifies with its
     own code and with the round's.
   - **Selection opened once.** The proof was made at 00:51:22, the selection half was opened once at 00:51:46 and
     `done.json` was written at 00:54:37. The manifest's provenance equals the current code, configuration, runs,
     re-hashed development caches and official digest.
   - **v1 records.** They and their server copies are unchanged. No file under a v1 path is newer than 06:50 on
     2026-10-15.
2. **The gate exception.**
   - The v1 check (9061032e) is unchanged and still records passed = false. Its only failed requirement is
     matched_classes_right, with its aggregates.
   - The fact-checker's own predicate on the 16 final update-2,000 rows lets only rows 0, 4, 6 and 10 through the all-null
     fallback. R1 read no earlier check point, and those store only booleans.
   - All 28 inherited fields equal the v1 check:
     - isolation, with 216 sequences and 864 comparisons per state, all 0;
     - padding, 831 positions, 0.0;
     - the repeat and the resume, 0.0;
     - the optimiser;
     - the three zero-gradient parameters, with nothing non-finite;
     - the initial states.
   - The pinned `Block._eval` returns the frame-branch argmax when no token is active. The checkout is at 7bd81bda with no
     tracked change, and its 30-file digest is 59ffe584.
3. **Source and training semantics.**
   - The 11 training files at 3d2ea49 digest to ae7531732dd0. Only `scripts/train_fact_video_feature.py` differs from
     them.
   - In that script, the changed functions are exactly the 9 that the configuration allows and the added ones exactly
     its 5. None was removed, and all 44 module-level statements are identical.
   - The training loop, the run updates, the resume, the padding check, the overfit smoke, model construction and the
     preparation are AST-identical. `term_values` repeats `fact_integration.term_losses` call for call and draws no
     random numbers.
   - No training semantics changed.
4. **Results.**
   - **Controls recomputed independently.**
     - Frozen Ordered + B3 came from V01's sha-checked saved probabilities with the fact-checker's own B3 decoder.
     - A01 Linear + B3 came from the frozen linear weights on the development features. Float32 on the GPU and float64
       on the CPU gave identical labels in all six seed-and-half cells.
     - A01 ASRF and MS-TCN ran through their frozen heads.
   - **Everything compared.** All 24 system rows, all comparisons and intervals, the verdicts, the goal and the readings
     (with their items, cells and intervals), the accounting and the denominators. That is 10,274 values, with 0
     mismatches.
5. **GPU re-inference.** Each frozen FACT head was strictly loaded after a sha256 check, and FACTHead.predict ran in eval
   mode on 20 sequences per seed and half. That is 120 sequences: 69 native clips, 23 pair streams and 28 multi streams.
   The labels, frame-branch labels and token counts were all identical to the saved outputs.
6. **Cost.** The v1 ledger (sha256 7e90dc53, 9 lines) and the R1 ledger (27 lines before the fact-check) give 8.790 GPU
   and 1.475 CPU hours cumulatively. The projections 11.848, 11.268 and 10.678 recompute from the ledger.
7. **Wording.** The README meets the rules. It never calls an absent consistent difference equality and states the
   consistency definition. It claims no cause. Constructed streams are not called scans, sessions are not called
   patients, the development halves are called reused, and the results are attributed to the revised execution.

## Corrections made

**Medium**
1. **The A4C/A5C X-Y-X middles.** The draft withheld their counts as not informative. The task sheet asks for them with
   their raw denominators, so the README now gives every system's matched counts out of 7 per seed and half.
2. **Session flags.** The calibration and selection multi-stream counts in "Where the halves differ" now give the
   sessions behind the recovered and the lost segments, and they flag fewer than 10 sessions as small samples.

**Low**
3. **A training loss.** FACT_s2's last-epoch loss is 1.723 (1.72349), not 1.724.
4. **The largest point-estimate item.** Its recovered and lost segments come from different sessions (11, 12 and 8
   against 5, 4 and 7). The README now gives both.
5. **The trade-off cells.** The README now says that the recovered counts are small (1 to 18 per cell, 1 to 6 in
   calibration) and come from fewer than 10 sessions per population in every cell.
6. **Inherited versus repeated checks.** The gradient record and the strict loads were also repeated under R1. Only
   isolation and the final smoke rows are purely inherited.
7. **Unmatched tokens.** Section 3 of the task sheet asks for them to be reported. In row 3 the 11 unmatched tokens are
   active, and in every other row they are null.
8. **"Byte-identical."** Only the second preparation.json is byte-identical to the first. The second
   revalidation_prepare.json differs in its time stamp and the script's sha256. This was corrected in the README and in
   `code_review.md`.

**Nit**
9. The second fix verification overlapped the second preparation and ended before the second check.
10. The queue status keeps only the last projection, so the projection before FACT_s1 recomputes from the ledger.
11. The CPU items now also list the identity diagnostic and the queue overhead and freeze.
12. The cost section now includes this fact-check, and this file exists.
