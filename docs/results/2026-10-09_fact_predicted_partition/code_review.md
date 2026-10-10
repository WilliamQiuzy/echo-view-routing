# Implementation review before training

This is the independent implementation review that the task sheet requires before the formal queue
(`docs/superpowers/plans/2026-10-09-fact-predicted-partition.md`, Task 2). An independent code-review agent read commit
26069b5 against the task sheet, the configuration, the pinned official FACT source (`third_party/fact` at 7bd81bd) and
the helpers it reuses. It ran the round's CPU tests (16 passed) and small synthetic checks. When it reviewed the code,
only `--prepare` had run, and it produces no head or FACT metric.

## Verdict

The verdict was "pass with fixes". There was one medium finding and one low finding, and no critical or high finding.
Both findings concern the final decision table, not training. The commit that adds this record fixes both, and that
commit precedes any head training.

## What the review confirmed

- **Data isolation.** The head and both FACT groups read only the allowlist-filtered training rows. Reference labels
  supervise the head and FACT, but never build a partition or enter a forward pass. The Uniform group never receives
  the head's cuts, and native windows and constructed streams follow one rule.
- **The head.** Its inputs come from the raw E1 features before FACT's channel and time masking. Its architecture,
  targets, loss reduction, pos_weight, plan, optimiser, schedule, final weights and decode rule match the task sheet.
  It stays frozen and outside FACT's state dict and optimiser.
- **The partition.** The uniform grid, the predicted union, the even-position control and the closed-segment
  conversion match the task sheet. The adapter's `temporal_downsample` differs from the official one only in the
  segment list. With no added cut it equals the official forward, loss and gradients bit for bit.
- **FACT training.** Both groups start from the source round's initial states and use its batch plans and recipe,
  under the same deterministic settings. The run identity covers the adapter, the partition functions, the constants,
  the configuration, the initial state, the head, the plan and the loss code.
- **Checks.** Reproducibility, resume, padding, optimiser binding, the frozen head, label isolation (predictions and
  partitions, before and after the 2,000-update fit) and the zero-gradient comparison with the official model match the
  task sheet. The queue's budget projection covers the whole round.
- **Evaluation.** The systems, references, comparisons, bootstrap settings, decision readings, subset tables, stage
  tracking, position control and the Boundary head + frame classifier control match the task sheet.

## Findings and fixes

1. **Medium: a seed and half with no miss failed the "losses at i/u" reading.** The rule asks whether most misses of a
   group's official output were first lost at the i or u block in every seed and half. The code counted a seed and
   half with no stream miss at all as failing. That could change whether decision row 3 applies.
   *Fix.* Such a seed and half now counts as meeting the reading. The configuration states this, and a test covers it.
2. **Low: the English text of decision row 3 read as "and".** The task sheet says the round ends when the losses still
   arise at i/u and the new policies do not reduce them stably *or* give no full-system gain. The code already
   implemented this reading, but the English row could be read as requiring both failures.
   *Fix.* The row now says that neither new group achieves both a stable reduction and a full-system gain.
