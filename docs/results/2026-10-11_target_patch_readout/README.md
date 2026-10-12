# Target-anchored readout round (2026-10-11)

This round asked whether binding the final readout of the video model to the target's time position helps it name the
target view when a window crosses a view change. Each seed's Ordered context model of the raw-video context round was
trained for 3,750 more updates under three readouts of its final normalised tokens. The inputs, windows, loss and
optimiser settings were kept, and each continued run restarts the optimiser, the 375-update frozen-encoder warm-up and
the cosine schedule. Continued CLS reads the class token as before. Global patches averages all 392 patch tokens. Target
patches averages the patch tokens of the target's time slots with fixed weights from the input marker. The three
readouts have exactly the same parameters and start from the same weights. All nine models were evaluated once on the
two development halves. The task sheet is `docs/superpowers/plans/2026-10-11-target-patch-readout.md`. The round claims
no novelty. The previous round's positive result, a multi-frame and system gain without a shown value of order, stands
unchanged.

**What kind of evidence this is.**
- The nine models trained on the EV9V training split only, on the source round's frozen windows. No development data
  was read before all nine runs had finished and been frozen.
- The development halves have been used in many earlier rounds, so this evaluation is not an independent validation.
- The hold-out is by session, not by patient.
- Constructed streams are spliced from separate recordings and are not real scans. They reuse recordings, so their
  units are not independent cases. The three seeds share the same cases.
- Native clips carry the public recording label, not expert frame labels.
- The test split and EchoNet-Pediatric were not read.
- **Reproducibility.** A resume restores the full training state bit for bit. Training after a resume, and a rerun of
  the same configuration, cannot be guaranteed to give bit-identical weights, because the max-pool backward in three
  MViT blocks has no deterministic CUDA kernel. The parameter-drift check verifies the implementation only, not model
  performance. None of the nine runs was interrupted or resumed.

**Order of work.**
- f06a5d7 committed the task sheet and the fixed configuration before any code. The readout tests were written before
  the module and failed at import while it did not exist. Both went into one commit, 7392aa5, so the repository itself
  cannot show that order.
- An independent implementation review failed 7392aa5 as committed. The check script's own norm hook collided with the
  readout model's hook guard, so every check stopped at once, and a crashed check left no ledger line. b277363 fixed
  both, added the review record (`code_review.md`) and committed the evaluation code before any evaluation output.
- The final check run passed every check, and 0af197a committed its record before the queue started.
- The nine runs trained one after another from 18:22 to 22:38 UTC. The single evaluation ran from 22:39 to 23:35 UTC.
- An independent fact-check recounted the results from the saved per-window outputs (see "Fact-check").

## Answers

Values are given for seeds 0, 1 and 2 in that order. A range "a to b" gives the smallest and the largest of the three
seeds. Where a range also spans halves or groups, the sentence says so. A difference is consistent only when all three
seeds' 95% intervals exclude zero strictly on the same side.

1. **Does the target-position readout help across a view change?** No consistent effect was detected.
   - **The primary metric.** It is the accuracy at the requested sample on mixed windows whose reference view covers
     exactly 8 of the 16 frames. Selection holds 94 such windows from 72 recordings in 30 sessions, and calibration 73
     from 56 recordings in 29 sessions. No reference run contributes more than one window.
   - **Target − Global patches.** It is +0.032, +0.032 and +0.000 on selection and +0.014, +0.014 and +0.027 on
     calibration. No interval lies strictly above zero, so no consistent difference was detected in either half. On
     calibration, the target readout fixes 1 to 2 of the 73 windows per seed and newly misses none, so many bootstrap
     draws contain no changed window and every lower bound is exactly zero. The stream-resampled sensitivity agrees.
   - **Target − Continued CLS.** It is +0.053, +0.000 and −0.011 on selection and +0.082, −0.027 and +0.055 on
     calibration, with no consistent difference.
   - **Other mixed windows.** On all mixed windows, Target − Global is consistent positive on calibration only
     (+0.004 to +0.006, a net 6 to 9 of 1,440 windows). The same holds for the 646 calibration windows with 9 to 11
     target frames (+0.008 to +0.009, a net 5 to 6 windows). This rests on the session bootstrap. In the stream
     sensitivity, seed 1's lower bound is exactly zero in both subsets, and seed 2's is −0.0016 on the 9-to-11 windows.
     The fixed rule downgrades a verdict only when the stream verdict is consistent in the opposite direction, so the
     verdict stands. Neither half shows a consistent difference against Continued CLS. These subsets are secondary, and
     they do not change the reading of the primary metric.
2. **What did change the mixed windows?** The continued training itself.
   - **Accuracy on the 8-of-16 windows.** The frozen Ordered models had 0.628 to 0.702 on selection and 0.575 to 0.644
     on calibration. After 3,750 more updates, Continued CLS has 0.734 to 0.809 and 0.644 to 0.822, Global patches 0.755
     to 0.798 and 0.712 to 0.781, and Target patches 0.787 to 0.809 and 0.726 to 0.808.
   - **Against the frozen Ordered model.** On the primary windows, Target − Ordered is consistent positive in both
     halves (+0.106 to +0.164 over seeds and halves). Continued CLS − Ordered is consistent positive on selection only,
     and Global − Ordered on calibration only. On all mixed windows, all three readouts are consistent positive against
     Ordered in both halves.
   - **What this means.** The gain comes with 3,750 more updates on the same training windows, with the optimiser, the
     frozen warm-up and the schedule restarted. No readout adds a consistent gain over Continued CLS on these windows.
     This round does not show why the continued training helps, and section 3 shows that it does not reach the
     segmentation.
3. **Does anything carry into segmentation?** No. Against the frozen Ordered model the patch readouts are worse as
   systems (Global patches in both halves, Target patches on calibration). Against Continued CLS no consistent
   difference was detected.
   - **B3 settings.** Three of the six patch models found no B3 setting that meets the calibration native floor 0.980
     with boundary recall 0.95. They are Global patches seeds 0 and 1 and Target patches seed 0. For Global seed 0 and
     Target seed 0, no setting reaches the floor, and the highest calibration native macro-F1 is 0.97957, which the
     tables print as 0.980. For Global seed 1, two settings reach 0.986, but with boundary recall below 0.95. The three
     B3 rows fail and show the diagnostic setting.
   - **Segment F1@50 with B3.** Target patches has 0.902, 0.918 and 0.949 on selection and 0.962, 0.966 and 0.964 on
     calibration. Global patches has 0.903, 0.934 and 0.941 and 0.962, 0.967 and 0.970. Continued CLS has 0.943, 0.955
     and 0.935 and 0.981, 0.976 and 0.976. The frozen Ordered models keep 0.938, 0.960 and 0.964 and 0.982, 0.984 and
     0.980.
   - **Comparisons.** Target + B3 − Ordered + B3 is negative in every cell and consistent negative on calibration
     (−0.020 to −0.016). Global patches + B3 is consistent negative against Ordered in both halves. Target − Continued
     CLS has five negative point estimates of six, but neither it nor Target − Global is consistent. Continued CLS −
     Ordered shows no consistent difference. Against E1, E0 and B7 + B3, Original FACT and both ASRF references,
     Target + B3 shows no consistent difference in either half.
   - **Where the patch systems differ.** With B3, four of the six patch models switch labels inside native clips 0.099
     to 0.106 times per clip on selection, against 0.039 to 0.081 for Continued CLS and Ordered. The same four have
     boundary precision 0.866 to 0.891 on selection, against 0.900 to 0.959. These four are exactly the rows with low B3
     thresholds. The three failed rows show the diagnostic quantile 0.95, and Target patches seed 1 chose 0.80. The
     other eight rows use 0.995 to 0.999. With the fixed Ordered setting the gap is smaller (0.060 to 0.077 switches per
     clip on selection, against 0.039 to 0.060). These rows are descriptive, and this round does not show why the
     readouts behave so.
   - **Fixed Ordered B3 setting.** Decoding each seed's three readouts with that seed's Ordered setting (half-window 9,
     the stored thresholds) shows no consistent difference for any pair. This is a sensitivity only.
   - **Native recognition.** No comparison shows a consistent difference in native A4C/A5C balanced accuracy. On
     selection the patch readouts read 41 to 51 A4C windows as A5C, against 34 to 47 for Continued CLS. On calibration
     they read 14 to 20 A5C windows as A4C, against 21 to 27. These window counts carry no interval.
4. **The goal and the strong baseline.** No group meets the goal. The frozen Ordered model keeps the highest three-seed
   means, but Continued CLS − Ordered shows no consistent difference, and Ordered itself misses the calibration minimum
   and the seed-2 selection native floor.
   - **Three-seed means with B3.** Continued CLS has 0.9442 and 0.9776, Global patches 0.9261 and 0.9663, and Target
     patches 0.9230 and 0.9640, against minimums of 0.9444 and 0.9821. The frozen Ordered model keeps 0.9544 and 0.9817.
   - **Constraints.** Every group misses at least one floor. Continued CLS seed 2 has selection native macro-F1 0.970,
     below 0.973. The patch groups fail through their infeasible B3 settings, and their seed-2 models have selection
     boundary recall 0.940 (Global patches) and 0.945 (Target patches), below 0.95.
   - **The new method does not beat existing models.** That claim would need consistent positive B3 differences of
     Target patches against Continued CLS and against the frozen Ordered model. Neither holds; against Ordered the
     difference is consistent negative on calibration.
5. **The fixed decision.** Only the "no gain" row applies. There is no consistent mechanism gain and no consistent
   system gain, so this readout and training-budget configuration ends. Target-conditioned models in general are not
   rejected, and no attention, gate or auxiliary loss is added.
6. **Cost.** The round used 5.3 GPU hours of its own 12-hour cap.
   - **Training.** The nine runs took 0.472 to 0.474 hours each, 4.25 hours in all.
   - **Checks.** The check ledger holds 0.082 hours, including a first attempt that stopped after 14 seconds.
   - **Evaluation.** The single evaluation took 0.93 hours.
   - **Speed.** Inference took 16 ms per window and 3.2 to 3.3 ms per 10 Hz sample in batches of 32 windows. That is not
     a streaming latency.

## Terms

- A **mixed window** is a window of a constructed stream whose 16 frames hold at least two reference views. Its
  **target share** is the number of its frames whose view equals the requested sample's view.
- **Continued CLS**, **Global patches** and **Target patches** read token 0, the mean of all 392 patch tokens, or the
  patch tokens averaged over space and weighted over the 8 time slots by the marker. A marker on slot k gives weight to
  the time slots whose stem kernel covers frame k (slot 8 to time slot 4; slot 9 to time slots 4 and 5 at 0.5 each).
- **B3** is the simple change-point decoder. Each model chooses its setting once on the calibration half.
- The **session bootstrap** of the mixed windows resamples the sessions of the requested samples. The **stream
  bootstrap** resamples whole streams and serves as a sensitivity. Both use 2,000 draws with seed 20261002 and the
  same draws for both systems.
- A difference is **consistent** in a half when all three seeds' 95% intervals exclude zero strictly on the same side.
  Otherwise no consistent difference is detected, which does not mean the systems are equal.

## 1. Mixed windows

Accuracy at the requested sample. Every column but the last uses the mixed windows of the constructed streams (1,449 on
selection and 1,440 on calibration; none has fewer than 8 target frames). The last column uses every window whose marker
is away from slot 8 (2,018 and 1,767, mostly native or single-view windows). Each cell gives the selection half, then
the calibration half.

| System | Seed | 8 of 16 | All mixed | 9-11 of 16 | 12-15 of 16 | A4C/A5C pair windows | Marker away from slot 8 |
|---|---|---|---|---|---|---|---|
| Ordered (frozen) | 0 | 0.628 / 0.575 | 0.903 / 0.915 | 0.871 / 0.875 | 0.967 / 0.986 | 0.829 / 0.913 | 0.982 / 0.984 |
| Ordered (frozen) | 1 | 0.702 / 0.630 | 0.909 / 0.914 | 0.866 / 0.867 | 0.973 / 0.985 | 0.847 / 0.890 | 0.984 / 0.979 |
| Ordered (frozen) | 2 | 0.681 / 0.644 | 0.912 / 0.910 | 0.877 / 0.856 | 0.971 / 0.985 | 0.883 / 0.890 | 0.983 / 0.982 |
| Continued CLS | 0 | 0.734 / 0.644 | 0.920 / 0.929 | 0.893 / 0.898 | 0.967 / 0.986 | 0.829 / 0.913 | 0.983 / 0.987 |
| Continued CLS | 1 | 0.809 / 0.822 | 0.934 / 0.941 | 0.904 / 0.909 | 0.977 / 0.982 | 0.874 / 0.921 | 0.987 / 0.983 |
| Continued CLS | 2 | 0.809 / 0.753 | 0.925 / 0.933 | 0.893 / 0.896 | 0.967 / 0.983 | 0.892 / 0.921 | 0.984 / 0.981 |
| Global patches | 0 | 0.755 / 0.712 | 0.920 / 0.929 | 0.893 / 0.896 | 0.964 / 0.981 | 0.865 / 0.906 | 0.986 / 0.978 |
| Global patches | 1 | 0.777 / 0.781 | 0.930 / 0.936 | 0.906 / 0.901 | 0.971 / 0.983 | 0.865 / 0.913 | 0.985 / 0.980 |
| Global patches | 2 | 0.798 / 0.781 | 0.928 / 0.931 | 0.893 / 0.890 | 0.974 / 0.982 | 0.883 / 0.913 | 0.981 / 0.980 |
| Target patches | 0 | 0.787 / 0.726 | 0.922 / 0.935 | 0.893 / 0.906 | 0.964 / 0.982 | 0.865 / 0.921 | 0.987 / 0.982 |
| Target patches | 1 | 0.809 / 0.795 | 0.934 / 0.940 | 0.912 / 0.909 | 0.970 / 0.983 | 0.883 / 0.929 | 0.986 / 0.980 |
| Target patches | 2 | 0.798 / 0.808 | 0.930 / 0.937 | 0.901 / 0.899 | 0.971 / 0.983 | 0.892 / 0.913 | 0.984 / 0.981 |

Differences on the primary 8-of-16 windows, with the session bootstrap and the stream sensitivity.

| Comparison | Half | 8 of 16, session bootstrap (seeds 0, 1, 2) | Stream bootstrap (seeds 0, 1, 2) | Verdict |
|---|---|---|---|---|
| Target − Global | selection | +0.032 [-0.022, +0.092]; +0.032 [+0.000, +0.073]; +0.000 [-0.032, +0.030] | +0.032 [-0.011, +0.080]; +0.032 [+0.000, +0.071]; +0.000 [-0.031, +0.031] | none |
| Target − Global | calibration | +0.014 [+0.000, +0.046]; +0.014 [+0.000, +0.041]; +0.027 [+0.000, +0.068] | +0.014 [+0.000, +0.043]; +0.014 [+0.000, +0.043]; +0.027 [+0.000, +0.070] | none |
| Target − CLS | selection | +0.053 [+0.011, +0.115]; +0.000 [-0.062, +0.059]; -0.011 [-0.058, +0.041] | +0.053 [+0.011, +0.104]; +0.000 [-0.057, +0.064]; -0.011 [-0.066, +0.045] | none |
| Target − CLS | calibration | +0.082 [+0.015, +0.143]; -0.027 [-0.101, +0.048]; +0.055 [-0.014, +0.136] | +0.082 [+0.013, +0.158]; -0.027 [-0.103, +0.042]; +0.055 [+0.000, +0.123] | none |
| Global − CLS | selection | +0.021 [+0.000, +0.060]; -0.032 [-0.110, +0.036]; -0.011 [-0.058, +0.041] | +0.021 [+0.000, +0.056]; -0.032 [-0.101, +0.035]; -0.011 [-0.066, +0.045] | none |
| Global − CLS | calibration | +0.068 [+0.000, +0.123]; -0.041 [-0.114, +0.038]; +0.027 [-0.025, +0.086] | +0.068 [+0.000, +0.139]; -0.041 [-0.120, +0.029]; +0.027 [-0.026, +0.082] | none |
| Target − Ordered | selection | +0.160 [+0.083, +0.250]; +0.106 [+0.033, +0.174]; +0.117 [+0.056, +0.170] | +0.160 [+0.094, +0.237]; +0.106 [+0.042, +0.179]; +0.117 [+0.055, +0.189] | consistent positive |
| Target − Ordered | calibration | +0.151 [+0.068, +0.226]; +0.164 [+0.074, +0.260]; +0.164 [+0.080, +0.266] | +0.151 [+0.072, +0.236]; +0.164 [+0.077, +0.260]; +0.164 [+0.087, +0.250] | consistent positive |
| CLS − Ordered | selection | +0.106 [+0.052, +0.170]; +0.106 [+0.041, +0.176]; +0.128 [+0.051, +0.191] | +0.106 [+0.052, +0.171]; +0.106 [+0.042, +0.177]; +0.128 [+0.057, +0.205] | consistent positive |
| CLS − Ordered | calibration | +0.068 [+0.000, +0.138]; +0.192 [+0.088, +0.300]; +0.110 [+0.039, +0.192] | +0.068 [+0.000, +0.143]; +0.192 [+0.097, +0.293]; +0.110 [+0.042, +0.183] | none |
| Global − Ordered | selection | +0.128 [+0.068, +0.200]; +0.074 [+0.000, +0.147]; +0.117 [+0.055, +0.173] | +0.128 [+0.066, +0.200]; +0.074 [+0.010, +0.144]; +0.117 [+0.055, +0.189] | none |
| Global − Ordered | calibration | +0.137 [+0.056, +0.212]; +0.151 [+0.057, +0.250]; +0.137 [+0.062, +0.227] | +0.137 [+0.063, +0.216]; +0.151 [+0.066, +0.247]; +0.137 [+0.067, +0.217] | consistent positive |

Differences on all mixed windows (session bootstrap).

| Comparison | Half | All mixed windows, session bootstrap (seeds 0, 1, 2) | Verdict |
|---|---|---|---|
| Target − Global | selection | +0.002 [-0.002, +0.007]; +0.004 [-0.001, +0.009]; +0.002 [-0.005, +0.008] | none |
| Target − Global | calibration | +0.006 [+0.001, +0.010]; +0.004 [+0.001, +0.008]; +0.006 [+0.002, +0.012] | consistent positive |
| Target − CLS | selection | +0.002 [-0.006, +0.011]; +0.000 [-0.011, +0.009]; +0.005 [-0.003, +0.012] | none |
| Target − CLS | calibration | +0.006 [-0.001, +0.013]; -0.001 [-0.010, +0.009]; +0.004 [-0.002, +0.011] | none |
| Global − CLS | selection | +0.000 [-0.007, +0.008]; -0.004 [-0.014, +0.004]; +0.003 [-0.003, +0.009] | none |
| Global − CLS | calibration | +0.000 [-0.006, +0.007]; -0.005 [-0.014, +0.005]; -0.002 [-0.007, +0.002] | none |
| Target − Ordered | selection | +0.019 [+0.009, +0.029]; +0.026 [+0.011, +0.037]; +0.018 [+0.006, +0.028] | consistent positive |
| Target − Ordered | calibration | +0.019 [+0.012, +0.028]; +0.026 [+0.013, +0.041]; +0.027 [+0.017, +0.040] | consistent positive |
| CLS − Ordered | selection | +0.017 [+0.010, +0.023]; +0.026 [+0.016, +0.035]; +0.013 [+0.002, +0.023] | consistent positive |
| CLS − Ordered | calibration | +0.014 [+0.005, +0.022]; +0.027 [+0.015, +0.040]; +0.023 [+0.013, +0.035] | consistent positive |
| Global − Ordered | selection | +0.017 [+0.008, +0.025]; +0.021 [+0.009, +0.033]; +0.016 [+0.006, +0.025] | consistent positive |
| Global − Ordered | calibration | +0.014 [+0.005, +0.022]; +0.022 [+0.010, +0.035]; +0.021 [+0.012, +0.032] | consistent positive |

Windows of the primary subset fixed and newly missed by Target patches, per seed, with their distinct recordings and
sessions.

| Comparison on 8-of-16 windows | Half | Fixed windows (recordings, sessions) | New errors (recordings, sessions) |
|---|---|---|---|
| Target − Global | selection | 4 (4, 3); 3 (3, 3); 1 (1, 1) | 1 (1, 1); 0 (0, 0); 1 (1, 1) |
| Target − Global | calibration | 1 (1, 1); 1 (1, 1); 2 (2, 2) | 0 (0, 0); 0 (0, 0); 0 (0, 0) |
| Target − CLS | selection | 5 (5, 4); 4 (4, 4); 3 (3, 3) | 0 (0, 0); 4 (4, 4); 4 (4, 4) |
| Target − CLS | calibration | 7 (7, 7); 3 (3, 3); 5 (4, 4) | 1 (1, 1); 5 (5, 5); 1 (1, 1) |

## 2. Native window recognition

Native windows at the same centres for every system. Balanced accuracy and AUC use the A4C and A5C windows only.

| System | Seed | Balanced accuracy | Margin AUC | Native window accuracy | A4C recall | A5C recall | A4C read as A5C (windows) | A5C read as A4C (windows) |
|---|---|---|---|---|---|---|---|---|
| Ordered (frozen) | 0 | 0.930 / 0.941 | 0.988 / 0.979 | 0.981 / 0.986 | 0.967 / 0.956 | 0.894 / 0.925 | 36 / 25 | 9 / 15 |
| Ordered (frozen) | 1 | 0.935 / 0.938 | 0.989 / 0.983 | 0.981 / 0.986 | 0.964 / 0.956 | 0.906 / 0.920 | 39 / 25 | 8 / 16 |
| Ordered (frozen) | 2 | 0.937 / 0.919 | 0.990 / 0.978 | 0.978 / 0.983 | 0.956 / 0.952 | 0.918 / 0.886 | 47 / 27 | 7 / 23 |
| Continued CLS | 0 | 0.931 / 0.931 | 0.986 / 0.977 | 0.983 / 0.986 | 0.968 / 0.966 | 0.894 / 0.896 | 34 / 19 | 9 / 21 |
| Continued CLS | 1 | 0.928 / 0.928 | 0.985 / 0.979 | 0.981 / 0.985 | 0.962 / 0.961 | 0.894 / 0.896 | 41 / 22 | 9 / 21 |
| Continued CLS | 2 | 0.937 / 0.910 | 0.985 / 0.976 | 0.978 / 0.982 | 0.956 / 0.954 | 0.918 / 0.866 | 47 / 25 | 7 / 27 |
| Global patches | 0 | 0.939 / 0.937 | 0.987 / 0.980 | 0.980 / 0.984 | 0.960 / 0.943 | 0.918 / 0.930 | 43 / 31 | 7 / 14 |
| Global patches | 1 | 0.936 / 0.937 | 0.986 / 0.981 | 0.979 / 0.984 | 0.954 / 0.943 | 0.918 / 0.930 | 49 / 32 | 7 / 14 |
| Global patches | 2 | 0.929 / 0.924 | 0.985 / 0.976 | 0.977 / 0.982 | 0.953 / 0.938 | 0.906 / 0.910 | 51 / 32 | 8 / 18 |
| Target patches | 0 | 0.939 / 0.940 | 0.987 / 0.981 | 0.981 / 0.985 | 0.961 / 0.949 | 0.918 / 0.930 | 42 / 29 | 7 / 14 |
| Target patches | 1 | 0.928 / 0.927 | 0.985 / 0.980 | 0.981 / 0.984 | 0.962 / 0.949 | 0.894 / 0.905 | 41 / 29 | 9 / 19 |
| Target patches | 2 | 0.941 / 0.921 | 0.987 / 0.978 | 0.978 / 0.982 | 0.954 / 0.942 | 0.929 / 0.900 | 50 / 32 | 6 / 20 |

Differences in balanced accuracy (session-paired bootstrap of native windows).

| Comparison | Half | Balanced accuracy (seeds 0, 1, 2) | Verdict |
|---|---|---|---|
| Target − Global | selection | +0.000 [+0.000, +0.002]; -0.008 [-0.016, +0.005]; +0.012 [-0.001, +0.031] | none |
| Target − Global | calibration | +0.003 [+0.000, +0.008]; -0.010 [-0.022, +0.004]; -0.003 [-0.013, +0.005] | none |
| Target − CLS | selection | +0.008 [-0.007, +0.029]; +0.000 [-0.029, +0.043]; +0.004 [-0.015, +0.030] | none |
| Target − CLS | calibration | +0.009 [-0.011, +0.024]; -0.001 [-0.010, +0.008]; +0.011 [-0.007, +0.030] | none |
| Global − CLS | selection | +0.008 [-0.007, +0.029]; +0.008 [-0.025, +0.046]; -0.008 [-0.016, +0.000] | none |
| Global − CLS | calibration | +0.006 [-0.016, +0.023]; +0.009 [-0.009, +0.025]; +0.014 [-0.010, +0.042] | none |
| Target − Ordered | selection | +0.009 [-0.005, +0.028]; -0.007 [-0.038, +0.041]; +0.004 [-0.005, +0.014] | none |
| Target − Ordered | calibration | -0.001 [-0.009, +0.005]; -0.011 [-0.024, +0.000]; +0.002 [-0.004, +0.008] | none |
| CLS − Ordered | selection | +0.001 [-0.006, +0.005]; -0.007 [-0.016, +0.002]; +0.000 [-0.018, +0.016] | none |
| CLS − Ordered | calibration | -0.010 [-0.021, +0.004]; -0.010 [-0.026, +0.003]; -0.009 [-0.023, +0.004] | none |
| Global − Ordered | selection | +0.009 [-0.005, +0.028]; +0.001 [-0.034, +0.043]; -0.008 [-0.019, +0.002] | none |
| Global − Ordered | calibration | -0.004 [-0.015, +0.004]; -0.001 [-0.013, +0.009]; +0.005 [-0.009, +0.020] | none |

## 3. Systems

Each cell gives the selection half, then the calibration half. The frozen Ordered rows and all references reproduce
their stored rows exactly. The report checks the Ordered rows field by field and the pooled segment counts of E0, E1 and
B7 + B3 and the historical E1 + ASRF; every field of every reference was compared afterwards (see "Deviations"). The
three infeasible patch models' calibration native macro-F1 prints as 0.980 but is 0.97957, below the floor.

| B3, each model's own setting | Segment F1@50 | Boundary recall | Boundary precision | Native macro-F1 | Label switches per native clip | Cuts at same-view joins | A5C read as A4C (clips) | A4C read as A5C (clips) |
|---|---|---|---|---|---|---|---|---|
| Ordered (frozen) s0 | 0.938 / 0.982 | 0.958 / 0.982 | 0.948 / 0.975 | 0.979 / 0.993 | 0.042 / 0.025 | 0.017 / 0.025 | 0 / 0 | 2 / 1 |
| Ordered (frozen) s1 | 0.960 / 0.984 | 0.969 / 0.987 | 0.959 / 0.980 | 0.979 / 0.986 | 0.039 / 0.017 | 0.004 / 0.025 | 0 / 0 | 2 / 2 |
| Ordered (frozen) s2 | 0.964 / 0.980 | 0.974 / 0.980 | 0.959 / 0.977 | 0.970 / 0.986 | 0.046 / 0.013 | 0.004 / 0.025 | 0 / 0 | 3 / 2 |
| Continued CLS s0 | 0.943 / 0.981 | 0.971 / 0.987 | 0.932 / 0.978 | 0.979 / 0.993 | 0.042 / 0.030 | 0.022 / 0.020 | 0 / 0 | 2 / 1 |
| Continued CLS s1 | 0.955 / 0.976 | 0.974 / 0.987 | 0.944 / 0.970 | 0.979 / 0.986 | 0.067 / 0.021 | 0.013 / 0.030 | 0 / 0 | 2 / 2 |
| Continued CLS s2 | 0.935 / 0.976 | 0.984 / 0.987 | 0.900 / 0.966 | 0.970 / 0.986 | 0.081 / 0.059 | 0.022 / 0.030 | 0 / 0 | 3 / 2 |
| Global patches s0 | 0.903 / 0.962 | 0.979 / 0.987 | 0.866 / 0.945 | 0.970 / 0.980 | 0.102 / 0.076 | 0.043 / 0.040 | 0 / 0 | 3 / 3 |
| Global patches s1 | 0.934 / 0.967 | 0.987 / 0.990 | 0.891 / 0.949 | 0.968 / 0.980 | 0.102 / 0.080 | 0.022 / 0.045 | 1 / 0 | 2 / 3 |
| Global patches s2 | 0.941 / 0.970 | 0.940 / 0.970 | 0.942 / 0.977 | 0.979 / 0.986 | 0.046 / 0.017 | 0.009 / 0.020 | 0 / 0 | 2 / 2 |
| Target patches s0 | 0.902 / 0.962 | 0.982 / 0.987 | 0.866 / 0.945 | 0.979 / 0.980 | 0.099 / 0.084 | 0.043 / 0.035 | 0 / 0 | 2 / 3 |
| Target patches s1 | 0.918 / 0.966 | 0.990 / 0.992 | 0.873 / 0.950 | 0.977 / 0.986 | 0.106 / 0.076 | 0.030 / 0.045 | 1 / 0 | 1 / 2 |
| Target patches s2 | 0.949 / 0.964 | 0.945 / 0.960 | 0.953 / 0.977 | 0.979 / 0.986 | 0.046 / 0.017 | 0.004 / 0.015 | 0 / 0 | 2 / 2 |
| E1 + B3 | 0.894 / 0.959 | 0.955 / 0.977 | 0.943 / 0.951 | 0.978 / 0.986 | 0.046 / 0.068 | 0.039 / 0.020 | 1 / 1 | 0 / 1 |
| E0 + B3 | 0.934 / 0.934 | 0.961 / 0.955 | 0.995 / 0.969 | 0.965 / 0.966 | 0.014 / 0.034 | 0.004 / 0.020 | 3 / 4 | 0 / 0 |
| B7 + B3 | 0.880 / 0.972 | 0.955 / 0.975 | 0.890 / 0.963 | 0.957 / 0.993 | 0.063 / 0.042 | 0.057 / 0.035 | 2 / 1 | 1 / 0 |
| Original FACT s0 | 0.906 / 0.967 | 0.948 / 0.982 | 0.926 / 0.968 | 0.989 / 0.986 | 0.000 / 0.034 | 0.052 / 0.025 | 0 / 1 | 0 / 1 |
| Original FACT s1 | 0.898 / 0.961 | 0.950 / 0.975 | 0.928 / 0.968 | 0.978 / 0.986 | 0.000 / 0.030 | 0.043 / 0.030 | 1 / 1 | 0 / 1 |
| Original FACT s2 | 0.911 / 0.965 | 0.940 / 0.972 | 0.925 / 0.972 | 0.966 / 0.993 | 0.000 / 0.034 | 0.035 / 0.025 | 2 / 0 | 0 / 1 |
| paired ASRF s0 | 0.878 / 0.972 | 0.935 / 0.987 | 0.927 / 0.975 | 0.930 / 0.986 | 0.011 / 0.004 | 0.065 / 0.035 | 4 / 1 | 1 / 1 |
| paired ASRF s1 | 0.888 / 0.972 | 0.940 / 0.985 | 0.937 / 0.978 | 0.954 / 0.986 | 0.007 / 0.004 | 0.048 / 0.035 | 3 / 1 | 0 / 1 |
| paired ASRF s2 | 0.882 / 0.966 | 0.942 / 0.982 | 0.942 / 0.980 | 0.954 / 0.986 | 0.011 / 0.004 | 0.065 / 0.030 | 3 / 1 | 0 / 1 |
| historical E1 + ASRF s0 | 0.910 / 0.976 | 0.942 / 0.977 | 0.952 / 0.980 | 0.948 / 0.993 | 0.046 / 0.038 | 0.030 / 0.015 | 2 / 0 | 2 / 1 |
| historical E1 + ASRF s1 | 0.909 / 0.964 | 0.950 / 0.972 | 0.936 / 0.970 | 0.966 / 0.975 | 0.007 / 0.030 | 0.057 / 0.030 | 2 / 2 | 0 / 1 |
| historical E1 + ASRF s2 | 0.907 / 0.964 | 0.953 / 0.977 | 0.933 / 0.970 | 0.966 / 0.975 | 0.021 / 0.046 | 0.057 / 0.035 | 2 / 2 | 0 / 1 |

| Fixed Ordered B3 setting (sensitivity) | Segment F1@50 | Boundary recall | Boundary precision | Native macro-F1 | Label switches per native clip | Cuts at same-view joins | A5C read as A4C (clips) | A4C read as A5C (clips) |
|---|---|---|---|---|---|---|---|---|
| Ordered (frozen) s0 | 0.938 / 0.982 | 0.958 / 0.982 | 0.948 / 0.975 | 0.979 / 0.993 | 0.042 / 0.025 | 0.017 / 0.025 | 0 / 0 | 2 / 1 |
| Ordered (frozen) s1 | 0.960 / 0.984 | 0.969 / 0.987 | 0.959 / 0.980 | 0.979 / 0.986 | 0.039 / 0.017 | 0.004 / 0.025 | 0 / 0 | 2 / 2 |
| Ordered (frozen) s2 | 0.964 / 0.980 | 0.974 / 0.980 | 0.959 / 0.977 | 0.970 / 0.986 | 0.046 / 0.013 | 0.004 / 0.025 | 0 / 0 | 3 / 2 |
| Continued CLS s0 | 0.940 / 0.981 | 0.971 / 0.987 | 0.925 / 0.973 | 0.979 / 0.993 | 0.042 / 0.025 | 0.022 / 0.025 | 0 / 0 | 2 / 1 |
| Continued CLS s1 | 0.950 / 0.974 | 0.976 / 0.990 | 0.935 / 0.963 | 0.979 / 0.986 | 0.053 / 0.013 | 0.017 / 0.035 | 0 / 0 | 2 / 2 |
| Continued CLS s2 | 0.953 / 0.975 | 0.976 / 0.985 | 0.937 / 0.970 | 0.979 / 0.979 | 0.060 / 0.021 | 0.017 / 0.030 | 0 / 1 | 2 / 2 |
| Global patches s0 | 0.920 / 0.964 | 0.979 / 0.985 | 0.897 / 0.951 | 0.970 / 0.980 | 0.060 / 0.063 | 0.026 / 0.040 | 0 / 0 | 3 / 3 |
| Global patches s1 | 0.947 / 0.967 | 0.982 / 0.987 | 0.921 / 0.952 | 0.970 / 0.980 | 0.070 / 0.034 | 0.022 / 0.045 | 0 / 0 | 3 / 3 |
| Global patches s2 | 0.930 / 0.974 | 0.979 / 0.987 | 0.897 / 0.970 | 0.970 / 0.986 | 0.067 / 0.038 | 0.022 / 0.030 | 0 / 0 | 3 / 2 |
| Target patches s0 | 0.913 / 0.967 | 0.979 / 0.987 | 0.888 / 0.952 | 0.970 / 0.980 | 0.070 / 0.051 | 0.030 / 0.035 | 0 / 0 | 3 / 3 |
| Target patches s1 | 0.939 / 0.969 | 0.984 / 0.992 | 0.910 / 0.952 | 0.968 / 0.980 | 0.077 / 0.038 | 0.022 / 0.045 | 1 / 0 | 2 / 3 |
| Target patches s2 | 0.933 / 0.977 | 0.984 / 0.987 | 0.900 / 0.968 | 0.970 / 0.980 | 0.067 / 0.042 | 0.017 / 0.030 | 0 / 0 | 3 / 3 |

| Direct argmax | Segment F1@50 | Boundary recall | Boundary precision | Native macro-F1 | Label switches per native clip | Cuts at same-view joins | A5C read as A4C (clips) | A4C read as A5C (clips) |
|---|---|---|---|---|---|---|---|---|
| Ordered (frozen) s0 | 0.893 / 0.961 | 0.979 / 0.987 | 0.846 / 0.942 | 0.970 / 0.986 | 0.095 / 0.093 | 0.043 / 0.035 | 0 / 0 | 3 / 2 |
| Ordered (frozen) s1 | 0.925 / 0.961 | 0.984 / 0.990 | 0.872 / 0.945 | 0.988 / 0.980 | 0.113 / 0.101 | 0.030 / 0.040 | 0 / 0 | 1 / 3 |
| Ordered (frozen) s2 | 0.926 / 0.960 | 0.984 / 0.982 | 0.874 / 0.940 | 0.970 / 0.986 | 0.116 / 0.097 | 0.026 / 0.035 | 0 / 0 | 3 / 2 |
| Continued CLS s0 | 0.896 / 0.963 | 0.979 / 0.987 | 0.840 / 0.938 | 0.988 / 0.993 | 0.109 / 0.110 | 0.057 / 0.030 | 0 / 0 | 1 / 1 |
| Continued CLS s1 | 0.927 / 0.962 | 0.990 / 0.990 | 0.877 / 0.947 | 0.988 / 0.986 | 0.130 / 0.084 | 0.030 / 0.045 | 0 / 0 | 1 / 2 |
| Continued CLS s2 | 0.912 / 0.966 | 0.984 / 0.990 | 0.856 / 0.947 | 0.988 / 0.986 | 0.123 / 0.118 | 0.030 / 0.040 | 0 / 0 | 1 / 2 |
| Global patches s0 | 0.900 / 0.956 | 0.982 / 0.987 | 0.860 / 0.940 | 0.970 / 0.980 | 0.109 / 0.093 | 0.043 / 0.040 | 0 / 0 | 3 / 3 |
| Global patches s1 | 0.925 / 0.956 | 0.987 / 0.990 | 0.871 / 0.940 | 0.968 / 0.980 | 0.141 / 0.105 | 0.022 / 0.055 | 1 / 0 | 2 / 3 |
| Global patches s2 | 0.903 / 0.954 | 0.987 / 0.990 | 0.845 / 0.931 | 0.979 / 0.986 | 0.130 / 0.143 | 0.035 / 0.045 | 0 / 0 | 2 / 2 |
| Target patches s0 | 0.899 / 0.955 | 0.982 / 0.987 | 0.858 / 0.940 | 0.979 / 0.980 | 0.109 / 0.093 | 0.043 / 0.035 | 0 / 0 | 2 / 3 |
| Target patches s1 | 0.905 / 0.950 | 0.990 / 0.992 | 0.849 / 0.929 | 0.977 / 0.986 | 0.120 / 0.118 | 0.039 / 0.055 | 1 / 0 | 1 / 2 |
| Target patches s2 | 0.906 / 0.950 | 0.987 / 0.992 | 0.849 / 0.925 | 0.979 / 0.980 | 0.141 / 0.118 | 0.035 / 0.040 | 0 / 0 | 2 / 3 |

The B3 setting each model chose on the calibration half. An infeasible model fails its B3 row and shows the diagnostic
setting.

| Model | B3 status | Half-window | Quantile | Feasible settings of 40 |
|---|---|---|---|---|
| Ordered (frozen) s0 | feasible | 9 | 0.999 | 40 |
| Ordered (frozen) s1 | feasible | 9 | 0.999 | 6 |
| Ordered (frozen) s2 | feasible | 9 | 0.999 | 38 |
| Continued CLS s0 | feasible | 7 | 0.999 | 40 |
| Continued CLS s1 | feasible | 7 | 0.999 | 38 |
| Continued CLS s2 | feasible | 5 | 0.995 | 36 |
| Global patches s0 | infeasible (diagnostic setting shown) | 9 | 0.95 | 0 |
| Global patches s1 | infeasible (diagnostic setting shown) | 9 | 0.95 | 0 |
| Global patches s2 | feasible | 11 | 0.999 | 33 |
| Target patches s0 | infeasible (diagnostic setting shown) | 9 | 0.95 | 0 |
| Target patches s1 | feasible | 9 | 0.8 | 25 |
| Target patches s2 | feasible | 11 | 0.999 | 2 |

## 4. System comparisons

Segment F1@50 differences with 95% intervals (paired bootstrap; streams reuse recordings, so these intervals are
approximate).

| Comparison (B3 rows) | Half | Segment F1@50 (seeds 0, 1, 2) | Native macro-F1 (seeds 0, 1, 2) | Verdict |
|---|---|---|---|---|
| Target − Global | selection | -0.001 [-0.006, +0.003]; -0.016 [-0.030, -0.005]; +0.008 [-0.009, +0.025] | +0.009; +0.009; +0.000 | none |
| Target − Global | calibration | +0.000 [+0.000, +0.000]; -0.001 [-0.006, +0.003]; -0.006 [-0.020, +0.005] | +0.000; +0.007; +0.000 | none |
| Target − CLS | selection | -0.041 [-0.064, -0.020]; -0.037 [-0.063, -0.013]; +0.014 [-0.008, +0.038] | +0.000; -0.001; +0.009 | none |
| Target − CLS | calibration | -0.019 [-0.037, -0.003]; -0.010 [-0.022, -0.001]; -0.012 [-0.027, +0.001] | -0.013; +0.000; +0.000 | none |
| Global − CLS | selection | -0.040 [-0.064, -0.019]; -0.021 [-0.043, +0.000]; +0.007 [-0.013, +0.027] | -0.009; -0.011; +0.009 | none |
| Global − CLS | calibration | -0.019 [-0.037, -0.003]; -0.009 [-0.020, -0.001]; -0.006 [-0.014, +0.002] | -0.013; -0.007; +0.000 | none |
| Target − Ordered | selection | -0.036 [-0.061, -0.012]; -0.043 [-0.073, -0.015]; -0.015 [-0.035, +0.001] | +0.000; -0.001; +0.009 | none |
| Target − Ordered | calibration | -0.020 [-0.038, -0.005]; -0.018 [-0.031, -0.006]; -0.016 [-0.031, -0.003] | -0.013; +0.000; +0.000 | consistent negative |
| CLS − Ordered | selection | +0.005 [-0.009, +0.019]; -0.006 [-0.020, +0.007]; -0.030 [-0.052, -0.008] | +0.000; +0.000; +0.000 | none |
| CLS − Ordered | calibration | -0.001 [-0.007, +0.005]; -0.008 [-0.017, -0.001]; -0.004 [-0.013, +0.005] | +0.000; +0.000; +0.000 | none |
| Global − Ordered | selection | -0.035 [-0.060, -0.011]; -0.026 [-0.052, -0.002]; -0.023 [-0.043, -0.006] | -0.009; -0.011; +0.009 | consistent negative |
| Global − Ordered | calibration | -0.020 [-0.038, -0.005]; -0.017 [-0.029, -0.007]; -0.010 [-0.019, -0.002] | -0.013; -0.007; +0.000 | consistent negative |
| Target − (E1 + B3) | selection | +0.008 [-0.022, +0.039]; +0.024 [-0.008, +0.055]; +0.055 [+0.023, +0.088] | +0.001; -0.001; +0.001 | none |
| Target − (E1 + B3) | calibration | +0.003 [-0.019, +0.022]; +0.007 [-0.010, +0.022]; +0.005 [-0.014, +0.022] | -0.006; +0.001; +0.001 | none |
| Target − (E0 + B3) | selection | -0.032 [-0.061, +0.000]; -0.017 [-0.046, +0.014]; +0.015 [-0.012, +0.042] | +0.014; +0.012; +0.014 | none |
| Target − (E0 + B3) | calibration | +0.028 [-0.000, +0.057]; +0.032 [+0.008, +0.057]; +0.030 [+0.006, +0.057] | +0.014; +0.020; +0.020 | none |
| Target − (B7 + B3) | selection | +0.023 [-0.001, +0.046]; +0.038 [+0.013, +0.062]; +0.069 [+0.039, +0.098] | +0.022; +0.020; +0.022 | none |
| Target − (B7 + B3) | calibration | -0.010 [-0.032, +0.009]; -0.006 [-0.024, +0.011]; -0.008 [-0.026, +0.009] | -0.013; -0.007; -0.007 | none |
| Target − Original FACT | selection | -0.004 [-0.030, +0.022]; +0.019 [-0.012, +0.049]; +0.038 [+0.007, +0.070] | -0.010; -0.001; +0.012 | none |
| Target − Original FACT | calibration | -0.005 [-0.027, +0.016]; +0.005 [-0.013, +0.025]; -0.001 [-0.020, +0.015] | -0.006; +0.001; -0.007 | none |
| Target − paired ASRF | selection | +0.025 [-0.004, +0.054]; +0.030 [-0.001, +0.060]; +0.067 [+0.036, +0.099] | +0.048; +0.023; +0.025 | none |
| Target − paired ASRF | calibration | -0.010 [-0.030, +0.008]; -0.006 [-0.022, +0.009]; -0.002 [-0.019, +0.014] | -0.006; +0.001; +0.001 | none |
| Target − historical E1 + ASRF | selection | -0.008 [-0.037, +0.019]; +0.009 [-0.018, +0.033]; +0.042 [+0.013, +0.069] | +0.031; +0.011; +0.012 | none |
| Target − historical E1 + ASRF | calibration | -0.014 [-0.032, +0.002]; +0.002 [-0.016, +0.019]; -0.000 [-0.019, +0.018] | -0.013; +0.011; +0.011 | none |

| Comparison (fixed Ordered setting) | Half | Segment F1@50 (seeds 0, 1, 2) | Native macro-F1 (seeds 0, 1, 2) | Verdict |
|---|---|---|---|---|
| Target − Global | selection | -0.007 [-0.018, +0.000]; -0.007 [-0.017, -0.001]; +0.003 [-0.005, +0.013] | +0.000; -0.002; +0.000 | none |
| Target − Global | calibration | +0.003 [-0.002, +0.010]; +0.002 [+0.000, +0.005]; +0.003 [+0.000, +0.008] | +0.000; +0.000; -0.007 | none |
| Target − CLS | selection | -0.028 [-0.047, -0.010]; -0.011 [-0.028, +0.006]; -0.020 [-0.040, -0.003] | -0.009; -0.011; -0.009 | none |
| Target − CLS | calibration | -0.014 [-0.030, -0.001]; -0.006 [-0.016, +0.003]; +0.002 [-0.005, +0.009] | -0.013; -0.007; +0.001 | none |
| Global − CLS | selection | -0.020 [-0.041, -0.001]; -0.003 [-0.019, +0.013]; -0.023 [-0.043, -0.006] | -0.009; -0.009; -0.009 | none |
| Global − CLS | calibration | -0.017 [-0.034, -0.002]; -0.008 [-0.018, +0.001]; -0.001 [-0.007, +0.004] | -0.013; -0.007; +0.007 | none |
| Target − Ordered | selection | -0.026 [-0.048, -0.004]; -0.021 [-0.045, +0.001]; -0.032 [-0.057, -0.010] | -0.009; -0.011; +0.000 | none |
| Target − Ordered | calibration | -0.015 [-0.031, -0.003]; -0.015 [-0.027, -0.005]; -0.003 [-0.010, +0.004] | -0.013; -0.007; -0.007 | none |
| CLS − Ordered | selection | +0.002 [-0.014, +0.019]; -0.010 [-0.027, +0.004]; -0.011 [-0.026, +0.001] | +0.000; +0.000; +0.009 | none |
| CLS − Ordered | calibration | -0.001 [-0.007, +0.005]; -0.009 [-0.020, -0.001]; -0.005 [-0.014, +0.003] | +0.000; +0.000; -0.007 | none |
| Global − Ordered | selection | -0.018 [-0.039, +0.002]; -0.014 [-0.036, +0.006]; -0.035 [-0.061, -0.012] | -0.009; -0.009; +0.000 | none |
| Global − Ordered | calibration | -0.018 [-0.035, -0.005]; -0.017 [-0.029, -0.007]; -0.006 [-0.015, +0.003] | -0.013; -0.007; +0.000 | none |

| Comparison (direct argmax) | Half | Segment F1@50 (seeds 0, 1, 2) | Native macro-F1 (seeds 0, 1, 2) | Verdict |
|---|---|---|---|---|
| Target − Global | selection | -0.001 [-0.003, +0.000]; -0.020 [-0.037, -0.006]; +0.004 [+0.000, +0.008] | +0.009; +0.009; +0.000 | none |
| Target − Global | calibration | -0.002 [-0.011, +0.004]; -0.005 [-0.017, +0.003]; -0.004 [-0.014, +0.005] | +0.000; +0.007; -0.007 | none |
| Target − CLS | selection | +0.003 [-0.012, +0.019]; -0.022 [-0.043, -0.001]; -0.006 [-0.019, +0.004] | -0.009; -0.011; -0.009 | none |
| Target − CLS | calibration | -0.009 [-0.028, +0.007]; -0.012 [-0.031, +0.006]; -0.016 [-0.031, -0.002] | -0.013; +0.000; -0.007 | none |
| Global − CLS | selection | +0.004 [-0.011, +0.019]; -0.002 [-0.017, +0.014]; -0.010 [-0.023, +0.001] | -0.018; -0.020; -0.009 | none |
| Global − CLS | calibration | -0.007 [-0.027, +0.013]; -0.007 [-0.023, +0.008]; -0.012 [-0.028, +0.003] | -0.013; -0.007; +0.000 | none |
| Target − Ordered | selection | +0.006 [-0.005, +0.018]; -0.020 [-0.042, +0.002]; -0.020 [-0.038, -0.004] | +0.009; -0.011; +0.009 | none |
| Target − Ordered | calibration | -0.007 [-0.023, +0.007]; -0.011 [-0.027, +0.003]; -0.010 [-0.024, +0.003] | -0.007; +0.007; -0.007 | none |
| CLS − Ordered | selection | +0.003 [-0.013, +0.019]; +0.002 [-0.008, +0.012]; -0.014 [-0.028, -0.002] | +0.018; +0.000; +0.018 | none |
| CLS − Ordered | calibration | +0.002 [-0.009, +0.014]; +0.001 [-0.013, +0.015]; +0.006 [-0.003, +0.016] | +0.007; +0.007; +0.000 | none |
| Global − Ordered | selection | +0.007 [-0.004, +0.019]; +0.000 [-0.016, +0.018]; -0.023 [-0.040, -0.008] | +0.000; -0.020; +0.009 | none |
| Global − Ordered | calibration | -0.005 [-0.024, +0.013]; -0.006 [-0.017, +0.004]; -0.006 [-0.022, +0.008] | -0.007; +0.000; +0.000 | none |

## 5. Reference units

Matched runs of the multi streams with B3, selection then calibration. A short run has at most 16 samples. The boundary
offset is the mean absolute distance of matched predicted changes from the reference changes.

| System (B3) | Short runs kept (141 / 134) | Short A4C or A5C runs (65 / 67) | X-Y-X middles (69 / 80) | A4C/A5C middles (7 / 7) | Mean boundary offset (samples) |
|---|---|---|---|---|---|
| Ordered (frozen) s0 | 130 / 128 | 54 / 66 | 66 / 80 | 5 / 7 | 1.55 / 1.46 |
| Ordered (frozen) s1 | 134 / 129 | 58 / 65 | 64 / 80 | 3 / 7 | 1.58 / 1.47 |
| Ordered (frozen) s2 | 136 / 127 | 60 / 65 | 66 / 80 | 5 / 7 | 1.55 / 1.49 |
| Continued CLS s0 | 137 / 129 | 61 / 64 | 67 / 80 | 6 / 7 | 1.47 / 1.38 |
| Continued CLS s1 | 134 / 127 | 58 / 62 | 64 / 79 | 3 / 7 | 1.40 / 1.31 |
| Continued CLS s2 | 140 / 128 | 64 / 63 | 66 / 79 | 6 / 7 | 1.47 / 1.35 |
| Global patches s0 | 137 / 129 | 61 / 64 | 66 / 79 | 6 / 7 | 1.45 / 1.38 |
| Global patches s1 | 141 / 128 | 65 / 63 | 67 / 79 | 6 / 7 | 1.42 / 1.34 |
| Global patches s2 | 130 / 124 | 58 / 62 | 64 / 79 | 5 / 7 | 1.46 / 1.37 |
| Target patches s0 | 137 / 129 | 62 / 64 | 66 / 79 | 6 / 7 | 1.45 / 1.36 |
| Target patches s1 | 141 / 129 | 65 / 63 | 67 / 79 | 6 / 7 | 1.39 / 1.34 |
| Target patches s2 | 133 / 125 | 60 / 63 | 64 / 75 | 6 / 7 | 1.42 / 1.35 |

## 6. What this round does not show

- **No value of the target position.** Target patches did not consistently beat Global patches on the primary windows.
  The only consistent differences between them are small secondary gains on calibration, a net 6 to 9 of 1,440 mixed
  windows and 5 to 6 of the 646 windows with 9 to 11 target frames. In the stream sensitivity, not every seed's
  interval excludes zero for them, and they do not reach the systems.
- **No new system.** No readout beat the frozen Ordered model as a system. Against it, Global patches was
  consistently worse in both halves and Target patches on calibration, and Continued CLS showed no consistent
  difference. The window-level gains of continued training did not reach the segmentation.
- **No mechanism.** Patch tokens have passed through global attention and temporal convolution, so even a positive
  result would not have shown that cross-view mixing happens only at the readout. The time weights bind grid positions,
  not anatomy or receptive fields.
- **Not the goal.** No group reaches the goal or meets every floor. The development halves are reused, so these numbers
  are no independent validation, and the test split stays sealed.
- **Not clinical scanning.** Constructed streams and public recording labels are not real continuous scans or expert
  frame labels.

## Deviations and launch record

1. **A helper file beyond the task sheet's list.** The pure summaries of the report live in
   `scripts/target_patch_readout_tables.py`, beside the listed report script and its tests.
2. **The first check attempt.** It stopped after 14 seconds, because the check's own shape-capturing hook on
   `encoder.norm` collided with the readout model's hook guard. The independent review found the same fault in the
   committed code. The fix captures the shape on the plain source model. Every check attempt now writes its GPU hours to
   the ledger, and the crashed attempt was added by hand as 0.005 hours. Its ledger time was first entered as 14:24:39
   UTC by mistake. The attempt ran from about 17:57:45 to 17:57:59 UTC, right after the 7392aa5 commit, and the ledger
   line was corrected.
3. **Reference records.** The frozen Ordered rows are checked inside the report, and the run stops when they differ. The
   summary does not keep a per-row record of differences for the other references. A comparison afterwards with the
   verified rows of the raw-video context round covered 324 fields, 216 of the 12 references and 108 of the three
   Ordered models. The fact-check repeated it and added the 54 fields of the stored G1 + ASRF rows. No field differs.

## Fact-check

An independent fact-check recomputed the results from the saved per-window outputs, mostly with its own code. Every
number in `summary.json`, `report_tables.md` and the README tables matched. It found one rounding slip in the text, one
wrong table caption and several overstated or unclear sentences, and this README carries the corrections.

- **Recomputed exactly.**
  - The windows, centres, markers and orders of all 12 systems, and their per-sample probabilities.
  - The primary subsets and their support, the 168 mixed-window accuracies, 432 bootstraps, 216 fixed-and-new tables and
    72 verdicts, including the stream rule.
  - The native metrics, bootstraps and verdicts, and all 66 system rows.
  - The B3 choices, including the three infeasible models, and the within-round comparisons and units.
  - The goal, and the decision rows (only "no gain"; no claim of beating existing models).
- **Integrity.** The frozen weights match their digests, the run identities recompute, every run started from its
  seed's pinned Ordered weights, and none was interrupted. The training manifests hold training recordings only, with no
  development recording or session and no test recording. The server code equals the committed code, and all 18 tests
  pass.
- **Re-decoded.** Decoding 64 calibration windows again with two frozen models gave logits bit-identical to the saved
  ones.
- **Tables and ids.** All 884 table cells of this README match `summary.json`. `report_tables.md` regenerates
  identically, and no recording, session or stream id appears in any file here.

## Commands

Run on the server in the project `.venv`, one GPU job at a time, with `ECHO_GPU_MEM_FRACTION=0.6`.

```bash
C=configs/experiments/target_patch_readout_v1.yaml
python scripts/prepare_target_patch_readout.py --config $C                 # sources and paired starting states
python scripts/train_target_patch_readout.py --config $C --check           # the checks before the queue
python scripts/queue_target_patch_readout.py --config $C                   # the nine formal runs
python scripts/report_target_patch_readout.py --config $C --evaluate       # freezing and the single evaluation
python -m pytest tests/test_target_patch_readout.py tests/test_target_patch_readout_report.py -q
```

## Provenance

- **Code.** The readout and training code are at 7392aa5 with the check fix of b277363. The report code is at b277363.
- **Sources.** The configuration pins the task sheet, the source round's configuration, summary, README and tables, the
  three Ordered final weights, the frozen manifests and their index, the saved Ordered per-window outputs and the
  server's torchvision MViT source. Every digest was checked before use.
- **Frozen weights.** On the server, `models_frozen/target_patch_readout_v1/{cls,global_patches,target_patches}_s{0,1,2}`
  hold the final weights, the run manifest and the configuration. `summary.json` holds every run's identity and final
  sha256.
- **Server outputs.** `runs/target_patch_readout_v1/evaluation/{selection,calibration}/` holds each model's per-window
  logits and per-sample probabilities. Recording and session ids stay on the server.

## Files

- `README.md` is this file. `code_review.md` is the implementation review.
- `preparation.json` and `checks.json` hold the preparation and the final check run. `input_tokens.png` shows one
  training window and the Target patches weights of its time slots.
- `summary.json` holds every system row, the mixed-window and native tables with their bootstraps, the system
  comparisons, the verdicts, the goal, the decision, the units and the costs. `report_tables.md` holds generated tables.
- `training_curves.png` shows the training curves.
- None of these files contains a recording or session id.
