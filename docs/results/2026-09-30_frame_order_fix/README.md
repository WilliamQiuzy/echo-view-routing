# Frame-order fix and objective checks (2026-09-30)

The defect is described and verified in `docs/results/2026-09-30_realvideo_round1/` §0. In short, EV9V names frames
without zero padding, and our loader sorted them as text. Each clip was therefore read in a shuffled order, and
sampling by position picked different frames than intended. This page records the fix, what was rerun, and the
objective checks that need no expert labels.

## 1. What changed in the code

- `echo_routing.ingest.frames.list_frame_paths` now sorts by the number in the file name, which matches the MP4 frame
  order. A test covers unpadded names.
- The windowed EchoViewCLIP script, which runs in its own environment, uses the same rule.
  `eval_echoprime_views.py` uses the shared loader.
- Frame-derived caches live in a new root, `cache_v2/`, stamped with a `FRAME_ORDER` file.
  - `load_paths()` refuses a non-empty cache root without the stamp.
  - Every external prediction directory is checked when it is read, so MS-TCN, ASFormer, windowed and ViT predictions
    from before the fix cannot be mixed in.
  - The old `cache/` is kept for tracing and can be read only with `ECHO_ALLOW_LEGACY_CACHE=1`.

## 2. Where each model read its frames, and what follows

| Component | How it read frames | Action |
|---|---|---|
| B0 frame encoder (training) | Shared loader, name order | Weights kept fixed for now. Its training sample differed from the intended one. |
| B0 features | Shared loader, name order | Re-extracted into `cache_v2` for train, val and test (`orig`, `gamma090`; `gamma110` for val and test). New samples match B0 on the true frames to within 0.004; the old cache differed by up to 0.93. |
| MS-TCN (official) | Trained on B0 features | Retrained on `cache_v2` (§3). |
| ASFormer (official) | Trained on B0 features | Retrain pending. |
| ViT family | Shared loader in training and testing | Retrain pending. |
| STFM (official) | Its own `sorted(os.listdir(...))`, name order | Retrain pending, with a numeric-order patch. |
| EchoViewCLIP | Trained on MP4 files, true order | Windowed inference rerun pending; no retraining needed. |
| EchoPrime (released weights) | Windowed inference through the shared loader | Rerun pending. |
| Demo videos and site | Shared loader | Marked as pending on the public site; regeneration pending. |

## 3. Core comparison before and after the fix

The comparison uses the same ladder configuration as run `20260928-001145`. The official MS-TCN was retrained on
`cache_v2` with the authors' constants (run `20261001-013857-b4_mstcn_official-0e15acae-s0`). The core ladder is run
`20261001-021147-baselines-0dc4de8a-s0`, and its paired bootstrap is in that run's `reports/bootstrap.md`.

| Method | Run | Clip macro-F1 | Frame acc | Frag/min | Boundary F1, pairs | Boundary F1, 4–8 fragments | Coverage at 5% | Error at 5% | Policy 2 coverage | Policy 2 error |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 frame classifier | before | 0.957 | 0.951 | 23.1 | 0.499 | 0.473 | 0.921 | 0.052 | 0.773 | 0.031 |
| B0 frame classifier | after | 0.944 | 0.950 | 19.9 | 0.521 | 0.487 | 0.922 | 0.061 | 0.759 | 0.026 |
| B1 smoothing | before | 0.949 | 0.968 | 2.36 | 0.918 | 0.852 | 0.999 | 0.089 | 0.815 | 0.036 |
| B1 smoothing | after | 0.947 | 0.965 | 2.72 | 0.933 | 0.851 | 0.999 | 0.093 | 0.782 | 0.038 |
| B2 HMM | before | 0.946 | 0.967 | 2.38 | 0.906 | 0.840 | 0.984 | 0.076 | 0.810 | 0.034 |
| B2 HMM | after | 0.940 | 0.965 | 2.51 | 0.930 | 0.833 | 0.993 | 0.088 | 0.759 | 0.030 |
| B3 change points | before | 0.946 | 0.963 | 3.55 | 0.913 | 0.847 | 0.998 | 0.089 | 0.889 | 0.037 |
| B3 change points | after | 0.943 | 0.961 | 3.57 | 0.909 | 0.847 | 0.977 | 0.080 | 0.766 | 0.031 |
| B4 MS-TCN (official) | before | 0.956 | 0.977 | 1.01 | 0.992 | 0.921 | 0.999 | 0.090 | 0.851 | 0.027 |
| B4 MS-TCN (official) | after, retrained | 0.947 | 0.973 | 1.51 | 0.923 | 0.876 | 0.995 | 0.088 | 0.848 | 0.024 |

How to read the columns:

- **Clip macro-F1** gives each test clip the label most of its frames receive.
- **Frag/min** counts label changes per minute inside clips carrying one source label. It is a stability proxy;
  clip labels do not establish that every internal predicted change is false.
- **Boundary F1** counts a predicted change as correct if it falls within 0.5 s of an artificial join between
  different source labels. It measures the constructed task, not expert-verified natural view transitions.
- **Coverage and error at 5%** use thresholds chosen on 4–8-fragment validation streams.
- **Policy 2** uses class-wise thresholds at the 95% upper bound of validation error.

What changed:

- B0's clip score falls by 0.013 because different frames are now sampled.
- The smoothing baselines move by less than 0.01 on the clip and boundary scores. The 5% target still fails for them
  on 4–8-fragment streams, and policy 2 still holds it, with lower coverage than before.
- **MS-TCN loses its lead at finding view changes.**
  - Its recall of true joins is unchanged (1.000 on pairs), but its precision falls from 0.984 to 0.857. Predicted
    boundaries on the 120 two-fragment test streams rise from 122 to 140.
  - With frames in order, it makes more internal cuts relative to the constructed stream's source-label reference.
    It also makes 1.5 instead of 1.0 changes per minute inside source-labelled single-view clips. Without interval
    annotations, these internal cuts cannot all be identified as clinical errors.
- **In the paired bootstrap, MS-TCN is no longer clearly better.** Its boundary F1 differs from smoothing, HMM and
  change points by at most 0.04 on pairs and 0.04 on 4–8-fragment streams, and every interval includes zero. It still
  splits a same-view join less often on 4–8-fragment streams (B1 minus MS-TCN is +0.045, interval 0.011–0.083).
  These intervals describe this set of model runs, not equivalence or training-seed variability. The current
  bootstrap resamples pair streams or source cines, not independently verified patients.
- **MS-TCN's own evaluation (authors' `eval.py`, test split) agrees.**
  - Edit score drops from 99.2 to 97.7 on pairs and from 88.7 to 85.6 on 4–8-fragment streams.
  - F1@50 drops from 98.9 to 96.2 and from 87.3 to 82.4.
- **The earlier conclusion does not hold.** The claim that learned temporal models find view changes much better than
  simple smoothing depended on shuffled frames.

### Order versus sampled images

The old name sort had two effects. It put a clip's frames out of order, and, because frames were picked by position,
it changed which images were picked. `scripts/legacy_set_true_order.py` separates the two. It rebuilds the legacy B0
cache with its own sampled images put back into true time order (`cache_v2_legacyset/`), and the ladder was rerun on
it (run `20261001-031300-baselines-1b027c50-s0`).

- Legacy cache against this one gives the order effect, because the images are the same.
- This cache against `cache_v2` gives the set effect, because both are in order but the images differ.
- The samples in this cache keep their true frame numbers, so their spacing is irregular, while the ladder still
  treats them as 10 Hz.

| Method | Measure | Legacy | Same images, true order | New images, true order |
|---|---|---|---|---|
| B0 | clip macro-F1 | 0.957 | 0.957 | 0.944 |
| B0 | frag/min | 23.1 | 19.2 | 19.9 |
| B0 | boundary F1, pairs | 0.499 | 0.536 | 0.521 |
| B1 | frag/min | 2.36 | 2.69 | 2.72 |
| B1 | policy 2 coverage | 0.815 | 0.764 | 0.782 |
| B2 | policy 2 coverage | 0.810 | 0.774 | 0.759 |
| B3 | policy 2 coverage | 0.889 | 0.767 | 0.766 |

- **Clip-level scores change only through the images picked.** B0's clip macro-F1 does not move when only the order
  changes, because the clip label is a majority vote.
- **Order drives the stability and routing changes.** Fragments per minute and most of the policy 2 coverage drop come
  from order.
- **Boundary scores of the smoothing baselines move by about 0.02 or less** in each step.
- **MS-TCN is not decomposed.** That would need it retrained on this cache.

## 4. Objective checks without expert labels

### Overlap of change candidates and reliable phase (EV9V)

A cycle counts as reliable when the peak is at least 0.3, the dip at least 0.6, and the clip spans at least two cycles.

| Set | Clips | Patient groups | With reliable cycle | Groups with cycle | In pilot | Pilot clips with cycle |
|---|---|---|---|---|---|---|
| All clips | 1,314 | 197 | 179 | 88 | 85 | 7 |
| Sustained, either recognizer | 34 | 29 | 4 | 4 | 25 | 3 |
| Sustained, both | 13 | 13 | 1 | 1 | 7 | 1 |
| Repeated, either | 172 | 94 | 23 | 18 | 44 | 5 |
| Repeated, both | 50 | 34 | 6 | 4 | 23 | 2 |

The view breakdown for sustained candidates with a reliable cycle is A4C 1 of 14 and A5C 3 of 11; none of the PLAX or
PSAX candidates qualifies. Only four real-change candidates carry verifiable phase information. That is too few to
test the mechanism on EV9V.

### Cycle method, validated on held-out patients (MIMIC)

The thresholds were chosen on half of the 4,019 reference loops' patients and tested on the other half, for 5 random
splits and both directions.

- Held-out agreement of the period with the ECG ranges from 93.3% to 97.6%, against a 95% target.
- The chosen threshold is not stable. Some splits choose peak ≥ 0.4, which keeps about 35% of loops. Others choose
  dip ≥ 0.6, which keeps about 57%.

The same-phase frames were also checked against the ECG. One fixed phase offset per loop was allowed, because the
image method's reference phase is arbitrary.

| Loops | Boundaries | Median error | 90th percentile | Boundaries within 50 ms | Drift per beat | Missed beats | Doubled frequency |
|---|---|---|---|---|---|---|---|
| 2,035 reliable | 5,296 | 10.4 ms | 46.7 ms | 91% | 10.9 ms | 0.9% of gaps | 0.3% of gaps |
| 3,547, no threshold | 9,194 | 13.4 ms | 74.3 ms | 83% | 14.2 ms | 2.8% of gaps | 1.1% of gaps |

One frame lasts about 33 ms, so the median residual is below one frame. The offset is fitted using the evaluated
boundaries within each loop; this measures phase consistency after alignment, not independently predicted absolute
R-wave timing with sub-frame accuracy.

The table above selects loops with a threshold chosen on all patients. Choosing the threshold on half of the patients
and measuring the alignment on the other half gives the same picture over 10 held-out evaluations:

- median error 9.0–11.6 ms, and 90th percentile 33–54 ms;
- 89–95% of same-phase frames within 50 ms;
- drift 8–13 ms per beat;
- missed beats in 0.4–1.2% of gaps, and doubled frequency in 0.1–0.6%.

### Phase-locking null, simulated through the whole pipeline (MIMIC)

The test is whether the off-dominant samples of a loop cluster at one cardiac phase. Only interior runs are used.

| Recognizer | Stratum | Real phase | Synthetic off-pattern | Phase of another patient's loop |
|---|---|---|---|---|
| B0 | repeated | 14.2% of 775 | 3.2% of 633 | 11.1% |
| B0 | none | 6.5% of 983 | 2.4% of 926 | 6.4% |
| EchoPrime | repeated | 19.2% of 354 | 5.6% of 266 | 14.4% |
| EchoPrime | none | 10.1% of 497 | 1.9% of 528 | 8.9% |

The columns are as follows.

- **Real phase.** Each loop's own ECG phase.
- **Synthetic off-pattern.** Off-runs from a two-state chain matched to each loop and independent of phase. These pass
  through the same screening and edge handling.
- **Phase of another patient's loop.** Keeps each loop's real off-pattern and uses the ECG phase of a different
  patient's loop of equal length.

What the null runs show:

- The simulated configurations show no clear large inflation of rejection rates. These finite simulations do not
  prove uniform 5% calibration; the reported rates include values above 5%.
- Most of the real rate comes back when the phase belongs to another patient. R-gated loops with similar durations
  and beat counts can have nearly identical phase trajectories, so the swap can preserve both acquisition alignment
  and genuine physiological correspondence. This control alone cannot distinguish those explanations.
- With 10 donor phases per loop, a small excess remains in the repeated candidates. For B0 it is 2.6 points (95%
  interval 0.9–4.2, resampling patients). For EchoPrime it is 3.4 points (0.6–6.1).
- In loops that are not candidates, the excess is 0.8 and 1.3 points, and both intervals include zero.

There is a small positive difference in rejection rates for own-loop versus the specified donor phases. It is not an
estimate of the fraction of loops causally affected by cardiac phase, nor the total value of phase for boundary
detection. Algorithmic utility remains to be tested using the controlled comparisons in `docs/research_plan.md`.

## 5. Pending

- ASFormer, STFM and ViT retraining; EchoPrime and EchoViewCLIP windowed reruns; routing scores, policy 2 and fusion
  reruns; demo regeneration.
