# Implementation review of A06-R1

An independent reviewer, a separate agent with read-only access, reviewed the revision's code before any GPU stage. It
later verified each fix, including two fixes made after the first launch record was refused.

**What it reviewed.**
- The configuration (7bd5249), the tests committed before the code (bfd1dc0) and the implementation (4ae1d3c).
- The task sheet, against the four checks it requires. The v1 failure must stay untampered. The exception must be
  limited to single-segment rows. The inherited evidence must support the new path. The queue must not bypass the guard.
- The old training code at 3d2ea49, which produced the v1 preparation and check.

**What it did not do.** It ran no stage of the round, no GPU job and no ssh command. It modified no repository file, and
it printed no recording, session, stream or recipe id. Its probes ran in a scratch directory.

**What its probes did.**
- They recomputed the training-file digest of the code exported from git at 3d2ea49, which gives ae7531732dd0.
- They compared every training file with that code. Only `scripts/train_fact_video_feature.py` differs.
- They ran their own AST comparison of that script with the old one. The changed, added, removed and disallowed
  functions equal the configuration's lists.
- They checked the five pinned v1 files by sha256 and confirmed the official all-null fallback in the pinned
  `Block._eval`.
- They ran the three test files with the official checkout and the git history present, so no test was skipped.

**Verdict.** Pass with fixes. There was no blocking finding and two should-fix findings, and all were fixed and verified.

## The four required checks

1. **The v1 failure is not tampered with.** All five v1 pins match. `inherited_checks` requires passed = false, ok =
   false and a failed recomputed requirement. `launch_record` re-verifies the original preparation and check pins at
   every queue, run and training-fit precondition. A revalidation record never opens v1, and v1's own launch path still
   needs v1's passed check.
2. **The exception is limited to single-segment rows.** `revised_gate` copies the task sheet's section 3 predicate. The
   tests reject every named variant: a multi-segment all-null row, an active but wrong token, a failed transcript, F1 or
   matching, and a row count other than 16.
3. **The inherited evidence supports the new path.**
   - `inherited_checks` verifies every field that v1's own check used, and more. It also checks the gradient signature,
     the isolation coverage, the smoke rows against the outcome and the recomputed requirements.
   - The code comparison is sound. The training loop, run updates, resume check, padding check, model construction and
     preparation are AST-identical to the old code.
   - `term_values` repeats `fact_integration.term_losses` call for call, in the same order. It reads its token counts
     from the same forward state. A test shows identical RNG states and identical loss values against the old monitor.
   - The monitor runs after each epoch's updates and before the resume save, as before.
4. **The queue cannot bypass the guard.** The queue, run and training-fit preconditions go through `launch_record`.
   `stage_queue` calls it again itself. `--check` is refused twice under a revision. The report stages read only frozen
   runs of the current identity.

## Should-fix findings and their fixes (4ffeb92)

**S1. The namespace guard did not cover the results directory or the evaluation manifest.**
- **The fix.** The manifest must lie in the results directory. A revision's results directory may neither lie inside
  the original round's results directory nor contain it.
- **Verification.** The reviewer probed each condition alone and confirmed that both real configurations still load.

**S2. Two server-side failures had unclear messages.** A missing git binary and a missing old-code export raised raw
errors.
- **The fix.** Both now stop with an actionable message.
- **Verification.** The reviewer confirmed this, including the missing-git case through a monkeypatch.

## Fixes after the first launch record was refused

**The launch identities (b355046).** The queue refused the first launch record because its run identities differed from
the queue's own.
- **The cause.** The official FACT constructor fills the Bu and BU block sizes of the head's configuration in place, and
  the identity hashes that configuration. The revalidation check had computed the identities after building models.
- **The fix.** The revalidation now computes them from a fresh head, as the queue, train(), the freeze, the training fit
  and the report already did. The training code was not changed.
- **Verification.** The reviewer confirmed the diagnosis in the pinned official source (`FACT.__init__` and
  `update_from(..., inplace=True)`). It reproduced the refusal in a scratch probe and confirmed that the new regression
  test fails on the old behaviour.
- **What had to be rerun.** The reviewer confirmed that a new `--check` alone would produce a coherent launch record.

**The preparation version (1636ada).** On the reviewer's suggestion, the check now refuses a preparation record made by
another version of the revalidation script. The preparation was therefore run again with the final script. Its
preparation.json is byte-identical to the first, and revalidation_prepare.json differs only in its time stamp and the
script's sha256. The reviewer verified that the guard runs before any CUDA call.

## Time

The reviewer's wall-clock times were booked in the round's ledger as CPU hours.
- The review ran from 15:17:30 to 15:33:43 UTC (0.27 hours).
- The fix verification ran from 15:37:25 to 15:41:37 (0.07 hours).
- The launch-identity verification ran from 15:53:44 to 15:57:45 (0.067 hours).
- The preparation-version verification ran from 15:59:20 to 16:00:48 (0.025 hours).
