# Implementation review of the segment objective round

An independent reviewer reviewed the round's code before any formal training. It was a separate agent with read-only
access. It reviewed commit 2fb4ffd against the task sheet and the fixed configuration (6088a5b), with the tests committed
before the modules (787bb42). It then verified the fixes in commit c7e8e70.

**What the reviewer did not do.** It ran no stage of the round and no GPU job. It opened no development cache, V01
development output or label file, and it printed no recording, session, stream or recipe id.

**What its probes did.**
- Ran the round's tests in a scratch copy of the repository, with the editable-install finder removed so that the copy
  was what got tested.
- Compared the loss with an independent time-step forward algorithm.
- Ran the bounded reader on synthetic development caches with canary rows.
- Sent real signals to real queue and run processes on a scratch copy.

They used about 0.03 CPU hours for the review (01:28 to 01:51 UTC on 2026-10-15) and about 0.03 hours for the
verification (01:56 to 02:05). Both are recorded in the round's ledger.

**Verdict.** Pass with fixes. No finding was blocking.
- All nine review items held.
- The five should-fix findings were fixed in c7e8e70, together with the changes listed under "Other changes".
- The reviewer verified every fix with file and line evidence and cleared the queue. All 64 round tests pass in a fresh
  scratch copy.

## What the reviewer confirmed before the fixes

1. **Equations.**
   - F, B and O follow the task sheet. B's denominator is the summed run weights D_n. O is (B + 0.5 L_order) / 1.5, with
     L_order weighted by D_n.
   - The forward recursion is the task sheet's code verbatim, with explicit rejections. It has no blank, no zero-length
     state, no `zero_infinity` and no maximum.
   - The objective is accumulated in float64 from the head's fp32 log-softmax in every arm. Padding never reaches a value
     or a gradient.
   - Runs come from labels alone, so they merge across same-view joins, and X-Y-X keeps three states. Native crops get
     their runs from the crop's own labels.
   - An independent time-step forward algorithm agreed with the recursion to 1.2e-15 in value and 5.8e-15 in gradient,
     over 624 cases with T up to 256, K up to 8 and logit scale up to 30.
2. **Loss tests.** The enumeration reference and its autograd gradients are independent of the logcumsumexp path. The
   tests cover X-Y-X, K = 1, T = K, P ≤ 1, the mean-CE bound, the rejections, B = O on all-native batches, F against
   masked weighted cross-entropy, repeated identical samples and padding.
3. **Isolation.**
   - Before the freeze, development caches and V01's development outputs are only hashed.
   - Every report stage verifies the frozen runs first.
   - No encoder call is reachable. A runtime guard makes encoder construction and forward passes, and A01's eager
     development loaders, raise.
   - Heads read features only. MS-TCN and ASRF receive zero placeholder labels of the valid length.
4. **Batching.** Every epoch's batch digests are compared with A01's plan, and the arms' batches are equal.
   `seed_everything` sits where A01 put it. Resume requires the same run identity and restores the RNG states.
5. **Optimiser.**
   - AdamW binds only the head's 2,565 parameters, with A01's settings and schedule.
   - The arms' initial states are bit-identical copies of A01's pinned linear initial state, which equals the V01
     source classifier.
6. **Barrier.**
   - The choice record carries its provenance. The git-object proof and the barrier work through their real command
     lines.
   - The bounded reader validates the archive and NPY headers. It made no cross-half read in either direction in the
     canary probes.
7. **Budget.**
   - The ledger keeps GPU and CPU kinds and records failed stages.
   - The check and per-run queue projections apply the 0.75-hour reserve against 4 GPU hours.
   - Report stages split their time into a GPU part and a CPU part without overlap.
8. **Evaluation.**
   - A01's stored rows are reproduced with A01's inputs, order and batch sizes.
   - B3 is chosen once on calibration and re-derived from its own grid later.
   - The comparisons match the task sheet.
   - The bootstrap draws exactly as `scripts/report_model_development.paired_bootstrap`.
   - The recovered and lost accounting worked in a synthetic probe.

## Should-fix findings and their fixes

1. **The ledger could miss formal runs.**
   - **The problem.** The queue wrote a run's GPU line only after the child process returned. An interrupted queue would
     have booked the killed run as CPU overhead, or nothing at all. A direct run wrote no line.
   - **The fix.** Every run now books its own GPU line in a `finally`, and SIGTERM becomes an exception, so interrupted
     runs are booked too. The queue books a run only when the run wrote no line of its own, for example because it was
     killed together with the queue.
   - **Verification.** The reviewer checked this with real signals: SIGTERM to the run, to the queue and to the whole
     group, SIGKILL to the run, and a normal finish. Each case produced exactly one run line.
2. **Selection could be opened twice without a trace.**
   - **The problem.** A failure after the selection half was opened would have left no record, and a rerun was possible.
   - **The fix.** The stage now writes an `opened.json` marker and a zero-hour ledger line after the barrier and before
     any selection read.
     - A finished evaluation is never repeated.
     - A repeat after a failed one needs a stated reason, and every opening is listed in the summary.
     - Figures and the private outputs are written before the results.
     - The dry run exercises both.
3. **The access check came after the label read.**
   - **The problem.** A half's label metadata was read before the access check.
   - **The fix.** `Ctx.half` now checks access first, and a test covers it.
4. **The order-contribution claim was too broad.**
   - **The problem.** "Order contribution claimed" was set from O − B alone, even when only B had progressed.
   - **The fix.** It now also requires O itself to be a progress candidate, and a test covers this case.
5. **The static encoder scan covered only the report script.** It now also scans the training script and the loss
   module.

## Other changes in c7e8e70

These came from the reviewer's notes or from the author.
- **Padding check.** The GPU padding check now feeds the padded batch to the head without normalisation. The padded
  logits differ, and the loss and gradients must still be identical. All three arms gave exactly 0.0 on the server.
- **Training cache hash.** A formal run checks the full sha256 of its training cache against the preparation record.
- **Freeze errors.** The freeze shows `freeze_model.py`'s error output.
- **B3 grid.** The calibration, dry-run and selection stages require the inherited decoding grid to equal the configured
  40-setting grid.
- **Replication disclosure.** The summary reports F against A01 Linear: probability and argmax differences and
  final-weight differences.
- **Config wording.** The configuration's numerics sentence now says the class weights are float32-exact. Only the
  wording changed. The configuration's sha256 moved from b53bf634 (6088a5b) to 3f698094 (c7e8e70). The run identities
  were made after the change.

Because the code and the configuration changed, the preparation and the check ran again on the server at 01:57 UTC and
passed before the queue started.

## Notes kept for the report

- **GPU booking gaps.** A run's own ledger line starts after its imports, so a few seconds per run are booked as queue
  overhead, not GPU. The fact-check measured 4.2 to 4.4 seconds per run, about 38 seconds over the nine runs.
- **Interrupted stages.** The report stages, the preparation and the check have no SIGTERM handler.
- **No direct runs during the queue.** The queue overhead line would also subtract a direct run started during the
  queue. No direct run was started.
- **SIGHUP.** A hang-up signal would book nothing. The queue was therefore launched detached (`setsid nohup`), so it
  cannot receive one from a closed terminal.
- **Full commit id.** `--selection` needs the full 40-character commit id. An abbreviated id fails safely.
