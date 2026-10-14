# Implementation review of the video dense supervision round

An independent reviewer read the round's code at cd0d4c0 and its parents back to the configuration commit 8ca5fc5 before
the formal queue. The reviewer compared it with the task sheet (`docs/superpowers/plans/2026-10-13-video-dense-supervision.md`)
and the fixed configuration (`configs/experiments/video_dense_supervision_v1.yaml`). The reviewer read the preparation's
outputs and the first passing check on the server and re-derived the labels on the CPU. The reviewer started no GPU job
and edited no file.

## Verdict

The review passed the code with fixes. It found no critical or high-severity issue and nothing that would make the nine
trainings invalid or unpaired.

## Findings and what was done

1. **Medium: GPU time of a killed attempt could be lost, and the queue had no lock.** A ledger line was written only when
   a trainer returned. If the queue or the server stopped during a run, that attempt's hours were never charged, and a
   restarted queue could launch a second trainer on the same run. f120f94 adds a lock that allows one queue at a time. It
   turns SIGTERM and SIGHUP into a clean stop that terminates the trainer and charges the attempt. An attempt left running
   by a killed queue is charged at the next start, until its run manifest or its log's last write, and a live trainer
   blocks the restart. Three regression tests cover this.
2. **Low: the anchor slot is not the most influential stem slot.** The design fixes each time token's anchor as the
   centre of its three stem slots (2j - 1, 2j and 2j + 1), as the task sheet specifies. On seed 0, the reviewer measured
   the stem's time taps at L1 shares of 0.16, 0.30 and 0.55 for these three slots. The anchor slot 2j carried 5 to 7% of a
   token's input sensitivity, against 11 to 16% for slot 2j + 1, and the three slots together carried 20 to 32%. In about
   53% of the mixed training windows, one anchor's label differs from the label of slot 2j + 1 (6.8% of mixed-window
   anchors). "Aligned" therefore means the label at the anchor slot, not the label of the frame that drives the token.
   The aligned and shuffled arms share the rule and the label multiset, so their contrast is unaffected. No design change
   was made, and the report states this caveat.
3. **Low: the first preparation run hashed evaluation files.** The first preparation (02:17 UTC) also verified the
   digests of the saved development outputs of V01 and V02, A01's development features and the result summaries. Nothing
   was parsed. cd0d4c0 restricted the preparation to the training inputs, and the formal preparation record is the run of
   02:32 UTC. Its label files are identical to the first run's.
4. **Low: the timing check leaves out the epoch saving.** The projection uses timed updates only. V01's estimate by the
   same method matched its nine actual runs, so the effect is small. The report notes it.
5. **Low: run bookkeeping could go stale after an identity change.** `resumes.json` and `train_log.jsonl` could carry
   over from an earlier identity. f120f94 starts both empty unless a resume of the same identity is loaded, and rewrites
   the log from the restored state. This changed the run identities, so the check was repeated.
6. **Low: a check name and a gate.** A field named for label isolation measured whether repeated forwards agree; it is now
   called `repeated_forward_identical`, and label isolation rests on the model's `(rgb, marker)` signature and its test.
   Missing encoder gradients after unfreezing were recorded but not gated; they now fail the check.

## What the reviewer confirmed

- **The CLS path is unchanged.** The norm hook stores a clone and returns nothing, before torchvision's in-place head
  dropout acts on the original token 0. It fires exactly once per forward and is removed in `finally`. The first check
  measured a CLS difference of 0 and a time-logit recomputation difference of 0.
- **Common RNG.** The forward does not depend on the arm, and the loss, backward, clipping and optimiser step use no RNG.
  The check found identical logits and CPU and CUDA RNG states across the three arms. The permutations come from their own
  NumPy generators.
- **SC/RV pairing.** Re-checked for all 30,000 windows of every seed. Every field is equal, and every recording carries
  one label across its slots, so the label at slot j belongs to the frame shown at input slot j.
- **Geometry.** The (8, 7, 7) grid and the stem (3, 2, 1, 1) are read from the network, the norm output has 393 tokens,
  and the token flattening keeps time slowest.
- **Labels.** The aligned labels, permutations and shuffled labels were re-derived exactly. Identity permutations number 0,
  1 and 1, and 13,402, 13,437 and 13,444 of the 15,000 mixed windows per seed change their labels. Native windows do not.
- **Shared head, loss and isolation.** No parameter or optimiser group is added; the loss matches the task sheet; the
  forward takes only `(rgb, marker)`; training and preparation read only training inputs.
- **State recovery and identity.** Weights load into the existing parameters, so the optimiser stays bound. RNG states are
  restored. The identity covers the training sections (supervision and budget included), the source sections, the arm,
  the seed, the window, label and weight digests, the environment and the code.
- **Budget.** One ledger includes failed attempts, and the queue projects the total before every run against the 8-hour
  cap with the 2-hour allowance.
