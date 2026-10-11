# Implementation review before training

This is the independent implementation review that the task sheet requires before the queue
(`docs/superpowers/plans/2026-10-11-target-patch-readout.md`, Task 2). An independent code-review agent read commit
7392aa5 and the configuration of f06a5d7 against the task sheet, the source round's code it reuses unchanged and the
pinned torchvision 0.29 MViT source. It ran the round's CPU tests (12 passed) and small synthetic probes. It had no GPU
and no source weights, so the GPU checks rest on the check record, not on this review.

## Verdict

The verdict was "fail" for commit 7392aa5 as committed, with one critical and one medium finding and nothing high or
low. Both findings sit in the check script, not in the readout, the training or the queue. A fix for both was already in
the working tree when the review ended, and the reviewer verified it. The commit that adds this record ships that fix,
and the final check run used the fixed script.

## What the review confirmed

- **Token order and grid.** The grid comes from the positional-encoding grid and every block's real query-pool stride,
  and (8, 7, 7) is only asserted. The readout's reshape uses the same (T, H, W) order as `conv_proj`'s flatten and every
  `Pool`'s reshape. All 16 blocks have a query pool with time stride 1, and three of them halve the spatial size.
- **Stem alignment.** The stem's temporal kernel, stride, padding and dilation are read from `conv_proj`. The reviewer
  compared the time weights of all 16 marker slots with an independent closed-form table and found no difference.
  Repeated targets use the existing single marker plane. No label, boundary, recording or session name reaches the module.
- **Readouts.** Global and Target patches share the same spatial mean and differ only in the time weights. Continued CLS
  runs the source path with no hook. The hook is removed in `finally`, also after an exception, and nothing is detached.
  Gradients reach the marker stem, the head and, after unfreezing, the projection.
- **Pairing and recipe.** The trainer asserts that the optimiser, the update count and the warm-up equal the source
  round's. Every source file is checked against its digest before use. The three readouts of a seed share the same
  state-dict keys, parameter count and parameter digest, and the learned marker is loaded. The resume state is complete,
  and every run records whether it was interrupted. Outputs stay in this round's directories, and only allowlisted
  training recordings can be decoded.
- **Checks and queue.** The checks follow the task sheet's tolerances and the source round's recorded rule for the
  nondeterministic max-pool backward. The queue order, skip rule and budget projection are right.

## Findings and fixes

1. **Critical: every check run stopped at its first step.** `pairing()` registered a shape-capturing hook on
   `encoder.norm` of a Target patches model and then ran it. The model's own guard refuses to add its readout hook to a
   norm layer that already carries one, so the check raised at once and no check record could be written. The first
   check attempt on the server stopped this way after 14 seconds. *Fix.* The shape is now captured on the plain source
   model, and the readout model is used only for its static grid, strides and stem geometry. The reported values are
   unchanged.
2. **Medium: a crashed check attempt left no ledger line.** The check wrote its GPU hours to the round's ledger only
   after every check had passed, so a crash would have been missing from the budget. *Fix.* Every attempt now writes its
   hours in a `finally` block, finished or not. The crashed first attempt was added by hand as 0.005 hours, its measured
   14 seconds rounded up.

## The final check run

The final check run used the fixed script and passed every hard check. The details are in `checks.json`.
- Continued CLS equals the source model exactly.
- Global and Target patches equal the independent recomputation exactly (768-d readout, 512-d output and logits).
- Reproducibility and resume agree to about 1e-9 on the formal schedule.
- The unfreeze-boundary check passed under the source round's rule.
- An update takes 0.458 seconds, so the round projects to 6.4 of its 12 GPU hours.
