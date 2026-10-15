# FACT on frozen video features (2026-10-15): stopped at the overfit smoke check

This round was to train the official GTEA FACT, a segment-token model, on A01's frozen V01 Ordered 512-d video
features. It would then compare that FACT with A01's ASRF, Linear + B3 and MS-TCN and with frozen Ordered + B3.

The task sheet fixes a gate before formal training. On A01's 16 fixed training check sequences, FACT must fit every
transcript, reach segment F1@50 = 1 and have every matched token predict its segment's view, within at most 2,000
updates. If it does not, the fit curve is saved and the round stops, with no extra updates and no other optimiser.

**FACT met every requirement except the matched-token one. The round therefore stopped before any formal training.**

**What did and did not happen.**
- No FACT head was trained beyond the check, and no model was frozen.
- No development array, label or probability was opened: the development caches were only hashed.
- None of the round's comparisons or readings exists. This is a stopping record, not a model result.

The task sheet is `docs/superpowers/plans/2026-10-15-fact-video-feature-training.md` (72d664b). The configuration is
`configs/experiments/fact_video_feature_v1.yaml`.

**What kind of evidence this is.**
- **Scope.** The smoke check uses training data only. It tests whether this recipe can fit 16 fixed training sequences.
  It says nothing about generalisation, and it does not show that FACT cannot learn on these features.
- **The sequences.** The 16 sequences are A01's fixed check set: the first 8 native training crops (one segment each)
  and the first 8 constructed streams.
- **Limits.** Constructed streams are spliced from separate recordings and are not real scans, and a few recordings
  recur in them. Native labels are public clip labels, not expert frame labels.

**Order of work.** All times are UTC on 2026-10-15.
- **Before any code.** 72d664b (04:59) holds the task sheet. 29dd33b committed the configuration at
  05:08.
- **Tests and code.** 576904f committed the tests at 05:14, which failed at import, and 45fc73c committed the code at
  05:23.
- **First preparation.** It ran at 05:24 on the CPU and passed.
- **Review.** An independent implementation review found one blocking and three should-fix problems (`code_review.md`).
  3d2ea49 fixed them at 05:55, and the reviewer verified the fixes.
- **Second preparation.** It ran from 05:55:59 to 05:57:43 on the CPU and passed.
- **GPU check.** It ran from 06:08:23 to 06:28:43. The overfit smoke failed, and the stage stopped the round as the
  task sheet requires.

## What passed

**The preparation (CPU).**
- **Sources.** All 8 source pins and the task sheet's sha256 match. A01's records match: its plans, training feature
  identities, sequence file and frozen ASRF, Linear and MS-TCN weights. A01's linear initial states were not used and
  not checked here.
- **Caches.** The full cache hashes equal A05's records. The development caches were hashed only.
- **Official checkout.** The FACT checkout is at 7bd81bda, with no tracked change. All 30 tracked files were digested
  (59ffe584).
- **Token rule.** Recomputed from the training labels: longest transcript 8, average 2.654333333333333, so 12 tokens.
- **Batches.**
  - Every seed's 50 epochs of label-only batch digests equal A01's plan.
  - Every one of the 18,750 batches per seed was unpadded and compared element by element with its sequences, which is
    150,000 sequences per seed.
  - Float32 spot checks against bounded reads of the source caches are equal: 16 sequences and 1,723 rows per seed.
- **Initial states.** They are bit-identical copies of F02's untrained FACT states (52ba4673, 048357d7 and 7bb0b685),
  strictly loaded: 326 tensors and 10,215,424 parameters.

**The GPU check (seed 0).**
- **Timing.** It was measured first. 5 warm-up and 50 timed formal updates averaged 0.644 s per update (median 0.617).
  - With the per-epoch monitor and save, the projection was 3.37 GPU hours per run.
  - The measured post-training estimate was 0.11 h. FACT decodes at 0.00015 s per sample and the controls at 0.00014 s
    per sample.
  - With the 1.25-hour reserve, the projection came to 11.81 of 12 GPU hours. The budget would have allowed the runs.
- **Reproducibility.** 6 updates repeated, and 3 + 3 updates resumed through a saved file, all gave 0.0 differences in
  every loss and parameter.
- **Optimiser.** It matches the configuration (Adam, lr 1e-4, betas (0.9, 0.999), eps 1e-8, no weight decay, clip 10)
  and is bound to the trained model.
- **Padding.** Changing 831 padded labels and features left the training-mode loss (same random state) and the
  eval-mode predictions unchanged.
- **Isolation.** Five label sequences on 216 sequences (864 comparisons), at the initial and at the overfit state,
  changed no prediction and no logit.
- **Gradients at initialisation.** Three parameters of the first action-branch layer get no gradient: its norm1 weight
  and its self-attention input and output projection weights. F02 recorded the same three.

## What failed: the overfit smoke check

The smoke trained FACT from the seed-0 initial state on the 16 check sequences. It used the two fixed batches of 8,
alternating (the 8 native crops, then the 8 streams), Adam at a constant lr 1e-4 with the formal settings, and the
official augmentation. The outcome was checked at updates 500, 750, ... and 2,000.

| Requirement | Update 500 | Updates 750 to 2,000 |
|---|---|---|
| Every transcript exact (16 of 16) | met | met |
| Segment F1@50 = 1 for all 16 | not met | met |
| One-to-one matching (16 of 16) | met | met |
| Every matched token predicts its segment's view | not met | **not met at any check** |

**At update 2,000.**
- Frame accuracy averaged 0.995 (lowest 0.968), and the official output was perfect at the segment level.
- The per-sequence share of matched tokens that predicted their segment's view averaged 0.75, and the lowest was
  0.0.
- In 11 of the 16 sequences the number of active (non-null) tokens equalled the number of reference segments.

**The pattern by sequence.**
- **Multi-segment streams.** All seven (3 to 6 segments each) used exactly one active token per segment, and every
  matched token predicted its segment's view.
- **Single-segment sequences.** There are nine: the 8 native crops and 1 one-segment stream. In four of them (three
  native crops and the stream), no token was active, so the token matched to the segment predicted the null class. The
  frame branch alone produced the correct output there. In one native crop all 12 tokens were active, although its
  matched token was right.

**The losses at update 2,000** (evaluation on the 16 sequences).
- The official total was 1.72. Its largest term was the f2a term, the token-to-frame cross-attention (1.27).
- The token term was 0.32. From update 1,000 on it moved between 0.14 and 0.60, with its highest values at updates
  1,400 and 1,450.
- The curve is in `smoke_fit.png`, and the log and the per-sequence rows are in `checks.json`.

**Context, not a cause.**
- **F02's diagnostic replay.** F02 showed the same pattern earlier in training. Its replay, run to 3,000 updates
  (bc45d88, `fact_overfit_extension.json`), found single-segment sequences without an active token: 4 of its 6 at
  update 500 and 2 of 6 at update 1,000. From update 1,500 every requirement held. After that replay, the owner's
  decision (a086346, 02:03 UTC on 2026-10-09) lengthened F02's check from 1,000 to 2,000 updates, and its rerun passed.
- **What differs.**
  - **Single-segment sequences.** F02's set had 6 and A06's has 9.
  - **The first fixed batch.** F02's held 5 native windows and 3 same-view-join streams, which have 1, 3 and 4
    segments, so it held 6 single-segment and 2 multi-segment sequences. A06's holds 8 single-segment native crops, and
    its second batch adds 1 single-segment stream.
  - **Inputs.** The features differ: video context here, E1 image features in F02.

  Which difference matters is not established.
- No other batch arrangement, length or optimiser was tried. Trying one after seeing this result would change the fixed
  check post hoc.

## Stopping decision

The task sheet's rule applies: "未过则保存实际fit曲线并停止本轮，不追加updates/换优化器" (if the smoke check fails, save the
actual fit curve and stop the round; no extra updates, no other optimiser).
- The queue was not started, and none of the three formal trainings ran. Nothing was frozen.
- The calibration, the evaluation manifest and the single selection evaluation did not take place.
- The fixed readings of section 5 apply to an evaluation, so none of them was computed.
- The round's code, configuration and records are kept unchanged.
- Whether anything follows is the lead's decision. This record proposes no change to the check or the recipe.

## What this round does not show

- It does not show that FACT cannot be trained on these video features. A formal run's batches mix 4 streams and 4
  native crops, so the smoke batches differ from training. The formal recipe was never run.
- It does not compare FACT with ASRF, Linear + B3, MS-TCN or frozen Ordered + B3, and no segment-level conclusion about
  FACT on video features follows from it.
- It does not explain the token-branch behaviour. The single-segment sequences, the features and the particular
  sequences are all candidates, and none was tested.

## Fact-check

An independent fact-checker, a separate agent, checked this record against the saved records, the code and git from
06:33 to 06:48 UTC (0.26 CPU hours, in the ledger). It confirmed every core number:
- the requirements at each check point and the per-sequence pattern;
- the losses, the timing and the projection;
- reproducibility, padding, isolation and the zero-gradient parameters;
- the cost, the commit order and the configuration hashes;
- that no queue ran, nothing was frozen and no development array was opened.

It found two medium errors: F02's first batch and single-segment count, and an unchecked A01 record listed as checked.
It also found low errors (the F02 replay's counts and order, the meaning of `trans=false`, the f2a term's name and the
amendment list). All were corrected before publication (`fact_check.md`).

## Cost

- **GPU.** 0.34 of 12 hours, spent by the failed check (06:08 to 06:28).
- **CPU.** 0.90 of 4 hours:
  - two server test runs, 0.004 h each;
  - two preparations, 0.002 h and 0.029 h;
  - the independent review, 0.40 h, and its verification, 0.20 h (both reported by the reviewer);
  - the fact-check, 0.26 h.
- The ledger is `runs/fact_video_feature_v1/ledger.jsonl` on the server. Failed stages are included.

## Deviations

- **Configuration amendment.** The configuration was amended once, after the implementation review and before any GPU
  stage (29dd33b to 3d2ea49; sha256 b5304032 to f9d3d51d). The amendment covered:
  - the smoke check points (every 250 updates from 500, stopping at the first pass, at most 2,000);
  - timing measured first in the check, before the other GPU checks;
  - the placement of the all-batch checks in the CPU preparation;
  - the measured post-training estimate;
  - the padding and reading-3 wording.

  No model, data, metric, threshold or budget changed.
- **The smoke check points.** Checking from update 500 could only have shortened a passing smoke. Here no check point
  passed, so the smoke ran its full 2,000 updates, the same as checking at the end.
- **The post module's digest.** 2c4c791 (06:08:32, during the check) records it in the train-fit output. It changed
  only that module, which is outside the training identity and was not used by the check, and the module never ran.

## Provenance

- **Code.** The configuration's sha256 is f9d3d51d (3d2ea49). The training code digest is ae753173, recorded by the
  second preparation and the check.
- **Source pins.**
  - A01 configuration 3b065e7f, preparation 5f7d48db and summary 3a160151;
  - F02 configuration 5caffd56 and summary 32c2665f;
  - A05 configuration 3f698094 and preparation b02b362d;
  - A03 configuration c2e90d60.
- **Official FACT.** Commit 7bd81bda2b84618a3e23876a2637a82f24881122, base configuration `gtea.yaml`, 12 tokens,
  `trans=false` (no transcript input; learned token queries), one-to-one matching.

## Files

- `README.md`. This record.
- `preparation.json` and `checks.json`. The preparation and check records. They include the smoke log, its check
  points, the per-sequence outcome rows (kinds and counts, no ids), the timing and the projection.
- `smoke_fit.png`. The smoke's fit curve.
- `code_review.md`. The implementation review and its verification.
- `fact_check.md`. The independent fact-check of this record.
