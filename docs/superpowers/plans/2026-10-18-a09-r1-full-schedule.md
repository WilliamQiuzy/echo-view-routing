# A09-R1 decision: complete the plain arm's schedule (2026-10-18 UTC)

**Status.** This is the lead's dated follow-up decision, made after reading A09's pilot. A09 stopped both arms at
update 1,250 of 3,750 under its frozen rule (`stop_both`). That record stays permanently, and the original gate is not
reread as a pass. R1 is not A09's promotion and not a preregistered success.

**What R1 tests.** At the pilot point the learning rate was still 0.84 of its peak. The design stays unchanged, and only
the plain-CE arm C completes its original decaying schedule. Does C then keep its native gain (fewer A4C read as A5C)
and lose the extra short false segments on selection? Training length, data exposure and learning-rate progress change
together, so no result can be credited to the learning rate alone.

**What runs.**
- C at seeds 0 and 2 resumes from the parent's update-1,250 states and runs to update 3,750 through A09's unchanged
  trainer. The state includes the AdamW moments, the update count and the RNG states. The manifest, the 14-window
  batches, plain CE, the model and the schedule stay as they were.
- D is not continued, because SupCon gave no native signal at the pilot.
- No smoothing, token, readout, temporal-head or coefficient variant is added.
- **Identity checks.** The parent identities are recomputed with the parent's code and must equal the pilot records and
  the resume states. A rename cannot bypass them.
- **New records.** The run directories, checkpoints and ledger are new (`runs/video_discriminative_r1`,
  `checkpoints/video_discriminative_r1`). Each run records its parent identity, the source resume sha256 and a new R1
  execution identity. A09's files are read only.

**Evaluation.**
- Only the final update-3,750 weights are evaluated, with no epoch choice, on both development halves.
- Each new model gets one B3 choice on calibration, with the same grid and rule.
- Comparisons: C3750 against its own C1250, frozen Ordered and A08 A.
- Reported: multi segment F1@50, native macro-F1, both A4C/A5C error directions, short false segments, missed and short
  true segments, and boundary recall.
- Argmax and Ordered's fixed decoder setting are sensitivities. The grid is not extended.
- All of this is exploration on reused development halves.

**Seed 1.** The rule is fixed in `configs/experiments/video_discriminative_r1.yaml` (`seed1_rule`) before any R1 output
exists. Seed 1 trains only if all three conditions hold:
- Segments clearly recover against the pilot.
- C3750 comes within 0.005 of A08 A in mean F1.
- Native recognition, boundary recall, missed segments and short true segments do not clearly degrade against frozen
  Ordered.

If it trains, it starts from V01 Ordered s1 with the same recipe for 3,750 updates, and the three seeds are evaluated.
Otherwise R1 stops. With two seeds there is no three-seed verdict. A useless full schedule would close this recipe
only; it would not show that the whole route fails.

**Budget.** At most 4 GPU h and 4 CPU h, including the agent's active wall clock, in a separate ledger
(`runs/video_discriminative_r1/ledger.jsonl`). A09's 1.341 GPU h and its CPU entries stay as they are. A single agent
runs everything. Only training data and the two development halves are read. The test split and EchoNet-Pediatric stay
sealed.

**Local git.** The local checkout's git objects were partly evicted to iCloud ("bad object HEAD"). The decision,
configuration and code are saved on the server with their sha256 and a launch record before the GPU starts. The local
round-only commit follows once git can read its objects again. It will state that it came after the launch.
