# Problem A, round 1: phase correspondence against same-budget pooling (2026-09-30)

> **Correction (2026-10-01).** In this document the EV9V "patient groups" or "presumed patient groups" are
> recording sessions, inferred from a file-name prefix that is a timestamp shared by the clips of one session.
> Sessions never cross a data split, but patient identities are unavailable, so patient overlap across sessions or
> splits cannot be assessed. The numbers are unchanged; only the description of the grouping changes.

**Question.** Does comparing the same cardiac phase across adjacent cycles reduce extra cuts at a fixed recall of label
changes? And if it does, is the gain due to the phase correspondence, or just to pooling several frames?

**Scope.** This is a proxy task built on existing labels (`docs/research_plan.md`, problem A).

- **Positives** are joins between fragments with different source labels in constructed streams.
- **Extra cuts** are predicted label changes inside native clips that carry a single source label. A source label does
  not prove that the view never changes inside the clip, so this is a stability proxy, not a count of clinical errors.
- **Exploratory status.** The validation and test streams had already informed research decisions, so this round is
  exploratory.
- **Fixed inputs.** All methods use the fixed B0 probabilities from `cache_v2`.

**Code.**

- Method: `echo_routing/realvideo/phase_change.py`, with tests in `tests/test_phase_change.py`.
- Experiment: `scripts/problem_a.py`.
- Summary: `scripts/report_problem_a.py`.
- Outputs: `runs/_demo/problem_a/` on the server (periods, per-stream rows, `grid_b0.csv`, operating points, summary).

## Methods

All methods run offline. They see only the stream's own images and the per-sample probabilities. Source files, labels
and join positions are used for scoring only.

**Existing baselines.** These use the ladder's decoders, each with one knob:

| Method | Knob | Grid |
|---|---|---|
| B1 smoothing | Minimum run (samples) | 1, 3, 5, 7, 9, 11, 15, 21 |
| B2 HMM | Stay probability | 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995 |
| B3 change points | Threshold | 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5 |
| B4 MS-TCN (retrained on `cache_v2`) | B1's minimum-run rule applied to its output (samples) | 1, 3, 5, 7, 9, 11, 15, 21 |

**The phase method and its controls.** All three variants compare the K cycles after a sample with the K cycles before
it, using exactly the same frames. They differ only in how those frames are paired.

| Variant | How the before- and after-cycles are compared |
|---|---|
| Phase | Each phase is compared with the same phase in the other cycles. The score is the mean distance across phases. |
| Pool | The before-window and the after-window are averaged first, without matching phases. |
| Scramble | As phase, but each earlier cycle is re-paired at its own random offset of 0.25–0.75 of a cycle. Three seeds. |

The pool control removes phase correspondence, not all cardiac-cycle information: its window length is still
`K * estimated_period`, and its score calibration still distinguishes period-available samples from fallback samples.
A detector with fixed-duration windows and no period-based calibration has not yet been evaluated here.

How the three variants are run:

- **Period.** It comes from image motion in 2.4 s windows of the stream itself. A window counts only if its peak is at
  least 0.3 and its dip at least 0.6.
- **Fallback.** Where no period is available, all three variants fall back to the same fixed comparison of 0.9 s
  windows.
- **Calibration.** Scores become percentiles among validation native clips. This is done separately where a period is
  available and in the fallback, so one threshold means the same false-alarm level in both.
- **Decoding.** Cuts are placed at percentile peaks above the threshold, at least 0.5 s apart. Each segment is
  labelled by its mean probability, and neighbouring segments with the same label are merged.
- **Grid.** K is 1 or 2. The percentile threshold is 0.90, 0.95, 0.97, 0.98, 0.99, 0.993, 0.995, 0.997, 0.998 or
  0.999.

**Operating point.** On validation, each method takes the setting with the fewest extra cuts per minute in native
clips, among those that recall at least 95% of different-label joins within 0.5 s. That setting is frozen for test.

## Results

| Method | Setting | Validation recall | Validation extra/min | Test recall | Test extra/min, native clips | Test extra/min, constructed streams | Same-label join cuts (4–8 fragments) |
|---|---|---|---|---|---|---|---|
| B1 smoothing | min run 5 | 0.977 | 2.35 | 0.952 | 2.49 | 1.82 | 0.144 |
| B2 HMM | stay 0.9995 | 0.971 | 0.93 | 0.956 | 1.10 | 0.94 | 0.114 |
| B3 change points | threshold 0.3 | 0.968 | 0.97 | 0.954 | 1.45 | 1.01 | 0.099 |
| B4 MS-TCN | min run 9 | 0.954 | 0.68 | 0.946 | 0.61 | 0.44 | 0.094 |
| Phase | K 2, pct 0.998 | 0.960 | 0.35 | 0.946 | 0.61 | 0.47 | 0.064 |
| Pool | K 2, pct 0.998 | 0.962 | 0.35 | 0.946 | 0.59 | 0.47 | 0.064 |
| Scramble, 3 seeds | K 1, pct 0.998 | 0.960 | 0.33–0.35 | 0.940–0.942 | 0.58–0.70 | 0.40–0.44 | 0.064–0.074 |

- All methods reached the 95% recall target on validation. At the frozen settings, test recall was 0.952–0.956 for
  B1–B3 and 0.940–0.946 for MS-TCN and the phase family, slightly below the target.
- No method cut at same-label joins in two-fragment streams (rate at most 0.008).
- Matched cuts sat on the join in the median case.

Paired differences on test are shown below. Native extra cuts were resampled by presumed patient group, and recall by
stream.

| Phase minus | Extra cuts per minute, native clips [95% CI] | Recall [95% CI] |
|---|---|---|
| Pool | +0.012 [−0.029, +0.058] | 0.000 [0.000, 0.000] |
| Scramble, seed 0 | −0.090 [−0.181, −0.024] | +0.004 [−0.004, +0.012] |
| Scramble, seed 1 | −0.026 [−0.096, +0.026] | +0.006 [0.000, +0.013] |
| Scramble, seed 2 | +0.024 [−0.039, +0.085] | +0.004 [−0.004, +0.013] |
| B4 MS-TCN | +0.010 [−0.301, +0.298] | 0.000 [−0.021, +0.021] |
| B2 HMM | −0.491 [−0.877, −0.145] | −0.010 [−0.025, +0.004] |
| B3 change points | −0.840 [−1.262, −0.461] | −0.008 [−0.021, +0.004] |
| B1 smoothing | −1.867 [−2.685, −1.115] | −0.006 [−0.021, +0.008] |

The full trade-off curves are in `grid_b0.csv`. Phase, pool and scramble lie within a few hundredths of an extra cut
per minute of one another at every threshold and both values of K, with nearly the same recall.

At K = 1, phase is slightly below pool at many thresholds, but the scramble seeds sit at the same level. That edge
comes from the per-phase statistic, not from the correspondence.

**Coverage.** A period was available for this share of input samples:

| Split | Native clips | Two-fragment streams | 4–8-fragment streams |
|---|---|---|---|
| Validation | 39% | 21% | 15% |
| Test | 33% | 11% | 15% |

**Where the extra cuts fall** (test native clips, extra cuts per minute):

| Variant | Where a period was available (26 min) | In the fallback (52 min) |
|---|---|---|
| Phase | 0.51 | 0.66 |
| Pool | 0.47 | 0.66 |
| Scramble | 0.39–0.78 | 0.66–0.68 |

**Sensitivity analysis.** The 34 frozen sustained-change candidates were removed, which dropped 108 streams. The
ordering stayed the same.

| Method | Test recall | Test extra/min, native clips |
|---|---|---|
| Phase | 0.964 | 0.48 |
| Pool | 0.964 | 0.47 |
| Scramble | 0.959–0.961 | 0.47–0.56 |
| MS-TCN | 0.961 | 0.44 |
| B2 HMM | 0.966 | 0.85 |

## Reading

1. **No evidence for the phase correspondence.**
   - The same-budget pooling control reproduces the phase method's result.
   - Scrambling the correspondence does not consistently make it worse; the seeds disagree in direction.
   - The comparison restricted to samples that had a period shows the same.
   - By the stop rule in `docs/research_plan.md`, phase is therefore no longer the core of the method. The EchoPrime
     recheck is not triggered.
   - This holds for this implementation and this proxy task only. It does not rule out every possible use of phase.
2. **An exploratory finding is the performance of the period-adaptive pooled detector.**
   - Without pairing phases, it makes about 0.59 extra cuts per minute in native clips at test recall 0.946.
   - B1–B3 make 1.1–2.5 extra cuts per minute at test recalls 0.952–0.956. The frozen operating points do not have
     identical recall, so the table alone does not establish superiority at equal recall.
   - Its point estimates are close to retrained MS-TCN; this is not an equivalence result.
   - It still uses estimated periods to choose window lengths. Attribution needs a fixed-duration control and
     comparisons with matched windows, score calibration and decoding. B3 already labels segments using their mean
     probability, so that operation is not unique to pool. Independent confirmation would follow only after this
     component audit; see `docs/research_plan.md`.
3. **Limits.**
   - The joins are abrupt and artificial.
   - Native clips may hold unlabelled real changes.
   - Source clips recur across streams.
   - The presumed patient groups are not verified.
   - B0 and MS-TCN each come from a single training run.
