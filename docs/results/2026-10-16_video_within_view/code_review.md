# Implementation review of A08

An independent reviewer, a separate agent with read-only access, reviewed the round's code before the training queue
started.

**What it reviewed.**
- The task sheet. Its sha256 db8bda36 matches the working tree and commit 6cbb67b.
- AGENTS.md, the experiment index and the configuration (19ff07a, sha256 f671b2e5).
- The tests committed before the code (042b7ef) and the implementation (170017c). These are the loss module, the training
  script and the report script.
- The reused A07 sources and their helpers. Git history shows no edit to them since A07.

**What it did not do.** It ran no stage of the round, no GPU job and no ssh command. It modified no repository file.

**What its probes did.**
- They checked every pin of the task sheet, the 18 inherited helper pins and the other inherited pins.
- They compared the loss with a plain-loop reimplementation on 1,377 random cases (largest difference 3.6e-15). They found
  no non-finite value or gradient for groups without a selected edge, at coefficient 0 or 0.15, even with logits scaled
  to 1e30.
- At coefficient 0 the value and the logit gradient were bitwise equal to A07's frame loss on 300 random label sets. The
  full gradient equalled an independently written formula. The gradient is zero above the truncation.
- They compared the update step with A07's F update operation by operation. Only the penalty terms and a monitor under
  no-grad differ, and the monitor draws no random number.
- They confirmed that every new run starts from the V01 Ordered states and never from an A07-trained state. A07's
  manifests are only read, and F's saved development outputs are opened only after all six freezes.
- They checked the run identity, the code digest and the budget rules. The check record's code digest (56224ab7) equals
  the one computed from 170017c.
- An exhaustive probe of the readings over 39,366 combinations of verdicts and floor failures found no mismatch with
  the configuration.
- They ran the two test files. 45 tests passed.

**Verdict.** Pass. There was no critical or high finding, and the trainings may start. One medium finding concerns the
report only, and the remaining findings are low.

## The medium finding

**Shared seed cases.** Task 3 and the configuration ask for the seeds' shared cases to be noted. The report summed the
recovered and lost counts over seeds without saying how many of them are the same reference segments.

**The fix (eb937a0).** The report now also counts, from the private case lists, the distinct cases behind every pooled
count and how many appear in one, two or three seeds. This covers recovered and lost reference segments, sequences
with new or removed unmatched predictions and native fixed or new clips. The fix was made after the queue started and
before calibration, because the report's code digest enters the choice record's provenance. The report is not part of
the training code digest, which stayed 56224ab7, so the run identities did not change. The 49 tests pass locally and
on the server.

## Low findings and what was done

- **The coefficient-0 multi-update checks have little power.** The six-update and 3+3 comparisons run inside the
  warm-up, where the new parameters move by at most about 5.6e-6, close to the 1e-5 tolerance. The single update is
  exact, and the CPU test is bitwise. The tolerances come from the task sheet, so this limit is disclosed, not changed.
- **The coefficient and truncation are module constants.** A unit test ties them to the configuration. There is no
  runtime check. Changing the training code now would change every run identity, so this is disclosed.
- **Hook counts are checked for two models.** Every check builds fresh models, and the model's marker hook raises on a
  leaked hook, so a leak could not reach training.
- **Source parity reads A07's tolerance.** It equals A08's (1e-4).
- **A default argument.** `verify_manifest` would fail if its epoch digests were not passed. Its only caller passes
  them.
- **Consistent negatives.** The reading also applies when its only items are favourable to the candidate, for example
  fewer FN segments. Every item carries an `unfavourable` flag, and the README explains the reading.
- **Untested report helpers.** `trade_cells` and `consistent_list` had no unit test. Tests were added with the report
  fix (eb937a0).
- **Inherited from A07's accepted design.** Four behaviours were kept and noted:
  - the id deny-list reads both halves' clip lists from the protocol;
  - a selection repeat after a failure needs a disclosed reason;
  - the queue projection is conservative after a resume;
  - the CPU cap is enforced in the report stages only.

## An observation for the interpretation

At the first update the penalty term (0.15 × S, about 0.95 to 0.98) was about eight times the cross-entropy (0.115).
The penalties of equal-label and different-label edges were nearly equal (6.27 and 6.66 per edge). Most of the penalty
comes from changes in the log-probabilities of non-target classes between neighbouring windows. This follows from the
fixed MS-TCN formulation, and the task sheet rules out a coefficient search. It matters when M − A is read, and a
smaller penalty alone is not a success.

## Time

The reviewer's wall-clock time was booked in the round's ledger as CPU hours. The review ran from 13:07:11 UTC on
2026-10-16 for 0.317 hours (1,139,622 ms by the agent harness).
