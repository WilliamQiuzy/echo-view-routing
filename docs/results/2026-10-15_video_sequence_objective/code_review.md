# Implementation review of A07

An independent reviewer reviewed the round's code before the training queue started. The reviewer was a separate agent
with read-only access.

**What it reviewed.**
- The task sheet. Its sha256 4418f3e6 matches the working tree and commit 5110b12.
- AGENTS.md and the configuration. The configuration has a single commit (1a76407) and sha256 f8d50e07.
- The tests committed before the code (3b92fe8) and the implementation (095320c).
- The pinned predecessors. These are the A05 loss module, the V01 window and model code, the V01 preparation and training
  scripts, the A01 training data, the A05 report and the A03 readout and report helpers.

**What it did not do.** It ran no stage of the round, no GPU job and no ssh command. It modified no repository file.

**What its probes did.**
- They traced the A05 objective with unit class weights and four targets per group by hand. It reduces exactly to the
  task sheet's L_F, L_B and L_O. The transcript term's internal division uses T = 4.
- They checked that the monitor reads the same detached logits once, with no extra forward pass or draw.
- They compared the grouping, the eligibility rule, the draw order and the manifest layout with the task sheet.
- They diffed the optimiser, schedule and warm-up sections of A07's configuration with V01's. The contents are equal.
- They checked that the run identity covers the configuration sections it names and that the code digest covers the
  files and functions the configuration names.
- They hashed all 18 pinned helper files and the seven pinned source files against the configuration. All match. The
  frozen weights and V01's saved outputs live only on the server, so the reviewer could not hash them.
- They confirmed that the three scripts did not exist at 3b92fe8, so the tests failed before the code was written.
- They ran the three test files. 50 tests passed.
- They checked the preparation and check records already on disk against the configuration's sha256. Every gate was
  true, and no training had started.

**Verdict.** Pass. There was no blocking or should-fix finding.

## The optional note and its follow-up

The reviewer noted that the count of skipped reference runs had no direct test. Only its building block was tested, and
the live preparation gave 0 for all three seeds.

A test now builds a constructed stream with a short run between two sampled centres. It checks that the manifest counts
report that run once. The change touches only the tests, so the run identities did not change. The code digest is still
1733e5e660b3. The three test files now hold 51 tests, and all pass locally and on the server.

The count of 0 in the real preparation is expected. An aggregate probe on the server read the 1,500 training streams.
Their 6,463 reference runs are at least 10 frames long, and consecutive grid centres are at most 5 frames apart. Every
run therefore holds a grid centre, so no run can fall between two sampled centres. The probe's CPU time is in the ledger.

## Time

The reviewer's wall-clock time was booked in the round's ledger as CPU hours. The review ran from 02:54:19 to 03:12:08
UTC on 2026-10-16 (0.297 hours). These times come from the reviewer's transcript.
