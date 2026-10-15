# Fact-check of the FACT video feature stop record

An independent fact-checker, a separate agent, checked this round's stop record on 2026-10-15 from 06:33 to 06:48 UTC.

**What it checked against.**
- The saved preparation and check records and the smoke figure.
- The configuration at both of its versions, the task sheet and the round's code, including the requirement computation
  in `scripts/fact_integration.py::outcome`.
- F02's records: `smoke_FACT.json` and `fact_overfit_extension.json`.
- On the server: the ledger, the check log and F02's overfit-set kinds (no ids).
- Git, for the commit order and times.

**How it worked.** It ran no stage of the round and no GPU job, and it opened no development cache, V01 development
output or label file. It used 0.26 CPU hours of command time, recorded in the round's ledger.

**A disclosed slip.** One of its commands wrote a temporary file outside its scratch directory on the server: a
comparison of the MS-TCN patch, deleted within the same command. No repository or round file was touched and no id was
printed.

**Verdict.** The stop is confirmed. Every core number matches the records. Two medium and several low errors in the
draft were corrected before publication. None changes the stop.

## What was confirmed

- **Check points.**
  - At update 500, the transcripts and the one-to-one matching were met, while segment F1@50 = 1 and the matched-token
    requirement were not.
  - From update 750 through 2,000 only the matched-token requirement failed, at every check point.
- **Update 2,000.**
  - Frame accuracy mean 0.9954, lowest 0.9682.
  - Exact transcripts and F1@50 = 1 for all 16.
  - Matched-token share mean 0.75, lowest 0.0.
  - Active tokens equal the reference segments in 11 sequences.
- **The per-sequence pattern.**
  - The 7 multi-segment streams (3 to 6 segments) have one active token per segment and every matched token is right.
  - Of the 9 single-segment sequences, 4 have no active token (3 native crops and 1 stream) and are still right through
    the official frame-branch fallback, which was checked in the code.
  - 1 native crop has 12 active tokens and a right matched token.
- **Losses.** At update 2,000 the total was 1.7176, f2a 1.2748 (the largest) and the token term 0.3164. From update
  1,000 the token term ranged from 0.1375 to 0.6027, with its two highest values at 1,400 and 1,450.
- **Timing and projection.**
  - 5 warm-up and 50 timed updates, measured first: mean 0.6442 s, median 0.6173 s.
  - 3.3710 h per run, recomputed.
  - The post-training estimate is 0.1089 h: FACT and controls at 1.54e-4 and 1.37e-4 s per sample, over 1,003,116 and
    419,805 samples, plus 0.05 h. It uses A01's half sizes of 37,287 and 34,216.
  - Total 11.8109 of 12 GPU hours.
- **Reproducibility, resume, padding and isolation.**
  - The 6-update repeat and the 3 + 3 resume give 0.0 everywhere.
  - The optimiser matches the configuration and is bound to the model.
  - Padding: 831 positions, loss difference 0.0, predictions equal.
  - Isolation: 216 sequences and 864 comparisons at both states, with no change.
- **Gradients.** The zero-gradient parameters at initialisation are the same three that F02's smoke recorded.
- **The preparation.**
  - Sources: the 8 pins and the task sheet's sha256.
  - Caches: the cache hashes equal A05's records, and the development caches were only hashed (the code path was checked).
  - The official checkout: 7bd81bda, 0 tracked changes, 30 files, digest 59ffe584 (recomputed).
  - The token rule: 8, 2.6543 and 12.
  - Batches: 50 epochs, 18,750 batches and 150,000 sequences per seed. Spot checks: 16 sequences and 1,723 rows per seed.
  - Initial states: they equal F02's (52ba4673, 048357d7, 7bb0b685), with 326 tensors and 10,215,424 parameters.
- **Code digest.** ae753173 (recomputed) is unchanged by 2c4c791, which touched only the post module that the check
  never imports.
- **Commits (UTC).** 72d664b 04:59:15, 29dd33b 05:08:13, 576904f 05:14:12, 45fc73c 05:23:07, 3d2ea49 05:55:02 and
  2c4c791 06:08:32 (during the check, 06:08:23 to 06:28:43). At 576904f the tests fail at import.
- **Configuration.** Its sha256 is b5304032 up to 45fc73c and f9d3d51d from 3d2ea49. It was amended once, with no
  change to the model, data, metrics, thresholds or budget.
- **Cost.** GPU 0.339 h (the failed check). CPU 0.639 h before this fact-check: tests 0.004 h twice, preparations 0.002
  and 0.029 h, review 0.40 h and verification 0.20 h.
- **No queue.** The server has no training directory, queue status, checkpoint, training-fit record, frozen model, queue
  or run log, and no process of this round.
- **Server records.** The local and server preparation and check records are byte-identical.
- **Wording.** The record makes no causal claim and quotes the stopping rule exactly. It twice says FACT is not shown
  unable to learn, and it calls constructed streams not real scans.

## Corrections made

**Medium**
1. **F02's first batch and single-segment count.**
   - **The error.** The draft said both first batches held eight single-segment sequences.
   - **The facts.** F02's first batch held 5 native windows and 3 same-view-join streams with 1, 3 and 4 segments, so 6
     single-segment and 2 multi-segment sequences. F02 had 6 single-segment sequences in all, and A06 has 9.
   - **Correction.** The README now lists these as differences.
2. **A01's linear initial states.** The README listed them among the A01 records checked. This round neither uses nor
   checks them, and the claim was removed.

**Low**
3. **F02's replay.** The README now gives its counts (single-segment sequences without an active token: 4 of 6 at 500
   updates, 2 of 6 at 1,000) and the order of events: the replay (bc45d88), then the owner's decision (a086346,
   2026-10-09), then the passing rerun.
4. **`trans=false`.** It means no transcript input, with learned token queries. It does not mean no transcript loss.
5. **The f2a term.** It is the token-to-frame cross-attention term.
6. **The amendment list.** It now includes timing measured first, in both files.
7. **`code_review.md`.** The move of the batch checks is described as a placement change disclosed as an amendment, not
   as something the task sheet allows.

**Optional items applied.**
- "Projection weights" for the zero-gradient parameters.
- "Every requirement except the matched-token one" in the opening.
- The limits on recurring recordings and clip-level labels.
