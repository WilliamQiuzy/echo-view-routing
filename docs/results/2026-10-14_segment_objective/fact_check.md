# Fact-check of the segment objective round

An independent fact-checker, a separate agent, checked this round from 02:51 to 03:14 UTC on 2026-10-15, after the single
selection evaluation.

**How it worked.**
- It worked read-only on the repository and the server and used neither the GPU nor any stage of the round.
- It printed no recording, session, stream or recipe id.
- It used its own code for the following:
  - B3 decoding;
  - class-aware greedy IoU-0.5 matching in the evaluator's tie order;
  - the native majority rule and the 0.5 s boundary matching;
  - same-view join cuts;
  - the bootstrap and the accounting;
  - run identities;
  - the git-object walk;
  - the losses.
- It reused these repository pieces:
  - A03's `half_sets` for the half metadata;
  - A01's `TrainSet`, plan and `batch_digest` to rebuild one formal batch;
  - the report's module list for its code digest.
  
  The round's `objective`, `loss_from_batch` and `predict` were used only as the objects under test.
- It used about 0.04 CPU hours of command time, recorded in the round's ledger.

**Verdict.** The results are confirmed.
- Every recomputed value in `summary.json` matched: 0 mismatches in about 17,000 values.
- The ordering, integrity, barrier, losses and label isolation all check out.
- The fact-checker agrees with the applied readings "uncertain or no qualifying gain" and "end".
- The README had one medium factual error, one medium wording problem and four low or nit-level errors. They were
  corrected before publication. Two points on the scope of the readings' operational tests are now disclosed in the
  README.

## What was verified

1. **Order and integrity.**
   - **Commits.** They are linear: a13276e (00:47:01), 6088a5b (01:00:40), 787bb42 (01:11:04), 2fb4ffd (01:24:23),
     c7e8e70 (01:56:10) and eaff3e1 (02:37:47). At 787bb42 both test modules fail at import.
   - **Server code.** It equals c7e8e70. All 8 training-digest files and the report's 93 imported modules and 2 helpers
     match. The digests recompute to configuration 3f698094, training code c77bce94 and report code 26120990. The only
     differing server files belong to other rounds.
   - **Runs.** All 9 run identities recompute, and the frozen and checkpoint final sha256 values equal the manifests'.
     Every run has 18,750 updates, no interruption, 50 epoch records equal to its training log, and queue return code 0.
   - **Choices.**
     - The server's choice file is blob 66546ed1 in eaff3e1, with sha256 0b5034da.
     - All 5 proof objects (the commit and 4 trees) hash to their names and equal the local git objects.
     - The selection half was opened once, at 02:38:26. That was after the commit and after the proof (02:37:57), and the
       evaluation finished at 02:42:58.
     - The dry run read the calibration half (472 sequences) for both halves.
   - **Committed settings.**
     - The fact-checker's B3 reproduces the stored B3, argmax and stored-setting labels in all 18 saved files, with 0
       differing samples.
     - All 360 grid thresholds re-derive exactly from the calibration native scores.
     - The 9 choices re-derive from their grids.
2. **System rows.**
   - **Values.** 13 systems, 3 seeds and 2 halves give 3,744 row values. With them came 1,404 TP/FP/FN, 702 transcript,
     468 switch and 156 join-cut values. There were 0 mismatches.
   - **F and A01 Linear.** F's rows equal A01 Linear's stored rows, and their decodings are identical sample by sample.
   - **References.** The frozen Ordered rows were recomputed from V01's saved outputs. A01 MS-TCN and ASRF were not
     recomputed; their summary rows equal A01's stored rows.
3. **Bootstrap.**
   - The draws replayed `paired_bootstrap` exactly: 2,000 draws with seed 20261002, sessions then streams, and the same
     draws for every system.
   - 78 comparisons, 8 metrics and three values each make 1,872 values. There were 0 mismatches at 1e-12.
   - All 156 verdicts recompute, and none is consistent.
4. **Goal and readings.**
   - The goal records of F, B, O and frozen Ordered recompute exactly.
   - The applied readings follow from the code's fixed tests. The fact-checker agrees with "uncertain or no qualifying
     gain" and "end". It also agrees with not applying complete system progress, the segment balancing effect and the
     order objective effect.
   - On the transcript-only and trade-off readings it agrees only under the code's tests (see note 3 below).
5. **Source-unit counts.**
   - 8,064 accounting values and 588 native-change values gave 0 mismatches, and the 42 pooled entries are equal.
   - The fewer-than-10-sessions flags are correct. Every recovered and lost bucket rests on fewer than 10 sessions.
6. **Losses.**
   - **Toy cases.** An independent path enumeration over 3,249 toy cases (T ≤ 6, 2,349 of them with X-Y-X transcripts)
     agrees with the forward recursion. The largest value difference is 1.4e-14 and the largest gradient difference
     3.5e-15. K = 1 equals mean CE, T = K has one path, and the rejections fire.
   - **One formal batch** (seed 0, epoch 0, update 0, initial state). F is 0.2815813020, B 0.3403623465 and O
     0.2459449336.
     - These agree with an independent implementation within 3.3e-16, with gradients on the same logits within 3.2e-16.
       That implementation computes the order term by forward-backward.
     - The fp32 parameter gradients agree within 9.7e-8 (accumulation order).
     - F differs from A01's fp32 cross-entropy by 4.0e-8 on the CPU. The check measured 1.5e-8 on the GPU.
   - **Epoch-0 replay.** Epoch 0 (375 updates) of seed 0 was replayed on the CPU for each arm. The epoch-mean training
     loss differs from the recorded GPU value by 4.4e-9 (F), 4.5e-9 (B) and 3.7e-9 (O).
   - **Class weights.** They are float32-exact for all three seeds.
7. **Label isolation.**
   - **Probe.** The frozen heads on the CPU reproduce the saved probabilities within 1.4e-6, with no argmax change.
     Renamed or reordered ids, and random labels, joins and native flags, gave bit-identical probabilities.
   - **No early development reads.** Before the freeze the development caches were only hashed; their access times
     match the preparation's hashing at 01:26. The calibration stage's bounded reader was confirmed by code reading.
   - **Cache layout and parity.** Each development cache holds 541 selection sequences, then 472 calibration
     sequences. Parity reproduces (at most 8.11e-6), and the initial states equal the V01 classifier tensors.
8. **Ids.** No recording, session, stream or recipe id appears in the seven result files. 6,462 ids and sessions were
   checked on the server.

## Corrections made to the README

**Medium**
1. **FP segments against frozen Ordered (section 3).**
   - **Original claim.** All three linear heads have fewer selection FP segments than the reference.
   - **The error.** F does not in seed 2: 44 against 21, and 89 against 76 summed over the seeds.
   - **Correction.** The README now gives B (25, 20 and 17) and O (25, 18 and 17) fewer than frozen Ordered (33, 22 and
     21) in every seed, and F fewer only in seeds 0 and 1.
2. **Seed-2 wording (section 2 and "What this round does not show").**
   - **The problem.** The heading and bullets implied a cause, and they cited only the stored-setting sensitivity as
     "no change", although its intervals include zero. With direct argmax, which is also a common decoder, seed 2's
     selection B − F (+0.0079 [+0.0008, +0.0153]) and O − F (+0.0069 [+0.0018, +0.0124]) exclude zero.
   - **Correction.** The README now reports both sensitivities. It says the seed-2 gap at each head's own setting
     coincides with F's seed-2 setting, with no cause established.
3. **Scope of the readings' operational tests (section 8).** The code's tests reproduce the applied readings exactly
   but are narrower than the sheet's wording in two places. Both are now disclosed descriptively, without changing the
   applied readings.
   - **Transcript distance.** O − F, a mandatory comparison, has a lower B3 multi edit distance in all six seed–half
     cells. The test covered only O − B and B − F.
   - **Native floor.** The trade-off test counted a single-seed recall-floor failure but not a single-seed native-floor
     failure. B and O fail the native floor in two selection seeds each, in the half where they recover segments
     against F.

**Low and nits**
4. **B's last-epoch losses.** They are 0.197, 0.198 and 0.197 (seed 0 is 0.19748).
5. **Run booking gap.** The gap booked as queue overhead is 4.2 to 4.4 seconds per run, about 38 seconds in total,
   not 2 to 3 seconds. This was also corrected in `code_review.md`.
6. **Native switch counts.** They are now flagged as resting on fewer than 10 sessions (3 to 8).
7. **The dry run.** It is not a deviation, because the configuration fixed it before any code (6088a5b,
   `evaluation.dry_run`).
8. **Nits.** Parity is "at most 8.11e-6". The launch figures (646 MiB, 13,125 updates at 02:06:44) come from the live
   process list and log, not from a saved record. The fact-checker found the run's timing consistent with them.
