# Implementation review of the FACT video feature round

An independent reviewer reviewed the round's code before any GPU stage. It was a separate agent with read-only access.

**What it reviewed.**
- Commit 45fc73c, against the task sheet and the fixed configuration (29dd33b).
- The tests committed before the modules (576904f).
- Afterwards, the fixes in commit 3d2ea49.

**What it did not do.** It ran no stage of the round and no GPU job. It opened no development cache, V01 development output or label file, and it printed no recording, session, stream or recipe id.

**What its probes did.** They ran in scratch copies:
- the round's tests, with the copy asserted to be what was imported;
- `check_fact` end to end on CPU with synthetic data;
- a simulated three-run queue with the freeze through the real `scripts/freeze_model.py`, inside the scratch copy;
- the train-fit stage on CPU;
- the smoke check points.

**Time.** The reviewer reported 0.40 CPU hours for the review (05:24 to 05:48 UTC on 2026-10-15) and 0.20 hours for the verification (05:55 to 06:07). Both are recorded in the round's ledger.

**Verdict.** Pass with fixes.
- There was one blocking finding and three should-fix findings. All were fixed in 3d2ea49.
- The reviewer verified every fix with file and line evidence, and 44 round tests passed in a fresh scratch copy.
- It cleared the GPU check, and the formal queue only if the check passed. The check did not pass (see `README.md`).

## Blocking finding and its fix

**B1. The padding check compared predictions in training mode, so the check could never pass.**
- **The problem.** FACT's official dropout, channel masking and temporal masking change every training-mode forward pass. The check's prediction comparison therefore always failed.
- **The fix.** A helper, `padding_check`, now compares the training-mode loss under the same random state and the eval-mode predictions of the unpadded inputs. A CPU test covers it.
- **Verification.** On CPU, the loss difference was 0.0 and the predictions were equal.

## Should-fix findings and their fixes

**S1. The run identity covered post-training code.**
- **The problem.** The freeze and the training-stream decoding lived in the training script, whose digest is part of every run identity. A later repair there would have invalidated finished runs.
- **The fix.** Both moved to `scripts/fact_video_feature_post.py`, which is outside the identity. The queue imports the freeze lazily after the third run, and a CPU test covers the stream decoding.

**S2. The budget gate was tight, and the post-training estimate held an unmeasured 0.25 h.**
- **The fix.**
  - Timing is measured first in the check.
  - The estimate now uses measured FACT decoding and measured A01 ASRF + MS-TCN decoding times per sample (on training streams).
  - These are multiplied by A01's recorded half sizes (37,287 and 34,216 samples) and the real decoding passes.
  - It adds 0.05 h of stage overhead, after A05's recorded evaluation stages.
- **Unchanged.** The configured timing rule (5 + 50 updates) and the 1.25-hour reserve.

**S3. The readings' operational details were incomplete.**
- **The fix.** Point-estimate items are counted separately for native clips, pair streams and multi streams, with raw denominators. Consistent items carry their per-seed intervals. Trade cells carry their FP and native macro-F1 intervals, and the uncertain note carries the primary comparison per seed.
- **Not exercised.** The round stopped before any evaluation, so these code paths never ran on real data.

## Further changes in 3d2ea49 (from notes or the author)

- **Run manifest.** It is written atomically before the resume file is removed.
- **Id guards.** They also cover `checks.json` and `train_fit.json`.
- **Resume check.** It also compares the first half's losses.
- **Report digest.** It includes the MS-TCN official model file.
- **Batch checks.** The all-batch unpadding checks and the float32 spot checks moved from the GPU check into the CPU preparation. The task sheet's check step lists the full batch structure, so this is a placement change, disclosed as a configuration amendment.
- **Overfit smoke.** It is checked at updates 500, 750, ... and 2,000, and it stops at the first passing check. This stays within the task sheet's "at most 2,000 updates". The reviewer's probe showed that the intermediate checks leave the training path bit-identical.
- **Configuration.** These changes amended the configuration before any GPU stage: the smoke check points, timing measured first, the placement of the batch checks, the post-training estimate method, and the padding and reading-3 wording. Its sha256 moved from b5304032 (29dd33b) to f9d3d51d (3d2ea49).
- **Later commit.** 2c4c791 also records the post module's own digest in `train_fit.json`. That module is outside the run identity.

## Notes kept for the report

- **The reviewer's budget expectation.** The fixed timing rule was expected to pass only narrowly. In the event the projection was 11.81 of 12 GPU hours (within the cap), measured at 0.644 s per update.
- **Trade-cell populations.** A trade cell's count of new unmatched predictions pools all populations, while its attached FP interval covers multi streams only.
- **The MS-TCN checkout.** It carries the documented Python 3 compatibility patch that A01 and A05 used. The exact reproduction of A01's MS-TCN rows guards it.
