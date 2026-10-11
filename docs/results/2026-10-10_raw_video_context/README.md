# Raw-video context round (2026-10-10 to 10-11)

This round trained one video model under three input conditions and evaluated all nine models once on the two
development halves. The model is EchoPrime's MViTv2-S video encoder with its 512-d projection kept, a new five-class
head, and a zero-initialised marker stem that tells the model which of its 16 frames is the target. The three groups
differ only in the RGB frames. Target-only repeats the target frame in all 16 slots. Shuffled context feeds the same 16
frames as Ordered context, with every target slot in place and the other slots permuted. Ordered context keeps the
true time order. Each group trained three seeds for 3,750 updates, and the final weights are the models. The round is
a model comparison with an explicit input control. It claims no novelty and no mechanism in advance. The task sheet is
`docs/superpowers/plans/2026-10-10-raw-video-recognition.md`.

**What kind of evidence this is.**
- The nine models trained on the EV9V training split only, through the allowlist. No development data was read
  before all nine runs had finished and been frozen.
- The development halves have been used in many earlier rounds, so this evaluation is not an independent validation.
- The hold-out is by session, not by patient. Selection holds 284 native clips from 36 sessions and calibration 237
  from 36. Only 11 and 18 of these clips are A5C.
- Constructed streams are spliced from separate recordings and are not real scans. They reuse recordings, so their
  units are not independent cases. The three seeds share the same cases.
- Native clips carry the public recording label, not expert frame labels.
- The test split and EchoNet-Pediatric were not read.
- **Reproducibility.** A resume restores the full training state bit for bit. Training after a resume, and a rerun of
  the same configuration, cannot be guaranteed to give bit-identical weights, because the max-pool backward in three
  MViT blocks has no deterministic CUDA kernel. The parameter-drift check verifies the implementation only. It does not
  show that model performance is unaffected by this nondeterminism. None of the nine formal runs was interrupted or
  resumed.

**Order of work.**
- c527046 committed the task sheet, the lead's decision closing the FACT refinement route, the configuration, the code
  and its tests before any training metric existed.
- A first check run on that code finished at 06:42 UTC and failed one check, described under "Deviations".
- An independent implementation review passed the code with three medium findings, all in the bookkeeping of the run
  identity and the budget. c749dd2 fixed them, recorded the boundary-check rule and added the review record
  (`code_review.md`).
- The final check run passed every check, and e00634f committed the check records before the queue started.
- The nine runs trained one after another from 06:56 to 11:11 UTC, and none was interrupted. 69b3d77 and 1899fd1
  fixed the report code before and during the single evaluation (see "Deviations").
- An independent fact-check recounted the results from the saved per-window outputs (see "Fact-check").

## Answers

Values are given for seeds 0, 1 and 2 in that order. A range "a to b" gives the smallest and the largest of the three
seeds. Where a range also spans halves, groups or comparisons, the sentence says so. "Selection" and "calibration"
are the two development halves. A difference is consistent only when all three seeds' 95% intervals exclude zero on
the same side.

1. **Does multi-frame input help?** Yes for recognition within one view, under this design. The system gain is
   consistent for Ordered context in both halves and for Shuffled context on calibration only.
   - **Native A4C/A5C recognition.** On native windows of A4C and A5C clips, both multi-frame groups beat Target-only
     in every seed and half. The balanced accuracy of Ordered − Target-only is +0.054 to +0.071 on selection and +0.069
     to +0.099 on calibration. For Shuffled − Target-only it is +0.044 to +0.067 and +0.071 to +0.107. Every interval
     lies above zero, so both differences are consistent positive in both halves. The margin AUC and the accuracy of all
     native windows are consistent positive too.
   - **Levels.** The balanced accuracy is 0.864 to 0.883 and 0.839 to 0.850 for Target-only, 0.927 to 0.942 and 0.920 to
     0.951 for Shuffled context, and 0.930 to 0.937 and 0.919 to 0.941 for Ordered context. E1 read at the same
     requested samples has 0.831 and 0.817.
   - **Full systems.** With B3, Ordered − Target-only in segment F1@50 is +0.030, +0.130 and +0.128 on selection and
     +0.031, +0.075 and +0.073 on calibration, consistent positive in both halves. Shuffled − Target-only is consistent
     positive on calibration (+0.025 to +0.073) but not on selection, where the seed-0 interval includes zero. With the
     direct argmax, both differences are consistent positive in both halves (+0.144 to +0.193 over both differences
     and halves).
   - **Why this is not a pure test of extra information.** Target-only feeds 16 copies of one frame, an input unlike the
     moving clips EchoPrime was pretrained on. Part of the gap may come from that mismatch rather than from the other
     frames. Target-only itself shows no consistent balanced-accuracy difference from E1 at the same samples. Its seeds
     1 and 2 chose B3 settings with low thresholds (quantiles 0.80 and 0.97), and their rows show 0.44 to 0.66 label
     switches per native clip.
   - **Across a view change, context costs accuracy.** On mixed windows of the constructed streams, whose 16 frames hold
     at least two views, Target-only names the target correctly in 0.947 to 0.957 of windows (selection) and 0.947 to
     0.956 (calibration). The two multi-frame groups manage 0.900 to 0.912 and 0.909 to 0.920. When only 8 of the 16
     frames show the target's view, they fall to 0.534 to 0.702 over both groups, seeds and halves, against 0.947 to
     0.986 for Target-only. When 12 to 15 frames show it, they reach 0.966 to 0.986, against 0.931 to 0.952. Against
     Target-only, Ordered context fixes 36 to 59 mixed windows per seed and half and newly misses 102 to 112, the same
     direction in all six cells. These window counts carry no interval, and mixed windows reuse recordings. Within these
     limits, the multi-frame gain holds when most of the window shows the target's view and reverses when a view change
     splits the window.
2. **Does true order add value beyond the same frames?** No added value was shown.
   - **Recognition.** Ordered − Shuffled in balanced accuracy is −0.011, +0.005 and +0.010 on selection and −0.010,
     +0.002 and −0.002 on calibration. Every interval includes zero, and the seeds point in different directions. The
     margin AUC and the accuracy show no consistent difference either.
   - **Full systems.** With B3, Ordered − Shuffled is +0.004, +0.028 and +0.025 on selection and +0.007, +0.011 and +0.000
     on calibration. No point estimate is negative, and three of the six intervals exclude zero (seeds 1 and 2 on
     selection, seed 1 on calibration). That is not all three seeds in a half, so no consistent difference was detected.
     With the direct argmax the pattern is similar, with two of six intervals above zero.
   - **Working points.** Ordered context chose B3 setting 9 / 0.999 in every seed. Shuffled context chose 7 / 0.999,
     7 / 0.995 and 9 / 0.999. The system differences of seeds 0 and 1 therefore also include different decoder settings.
   - **What this does not mean.** It is no evidence that order does not matter, and the two groups are not shown to be
     equal. On mixed windows with 8 of 16 target frames, Ordered context is higher than Shuffled context in 5 of the 6
     seed-half cells (0.575 to 0.702 against 0.534 to 0.616 over seeds and halves). These windows number 73 to 94 per
     half and carry no interval.
3. **Does the recognition gain carry into segmentation?** Partly. Ordered context + B3 is consistently better than
   E1 + B3 in both halves, but it does not reach the goal.
   - **Against the references.** Ordered + B3 − (E1 + B3) is +0.045, +0.066 and +0.070 on selection and +0.023, +0.025
     and +0.021 on calibration, consistent positive in both halves. Against B7 + B3, C + B3, Original FACT and the paired
     ASRF it is consistent positive on selection only. Against E0 + B3 it is consistent positive on calibration only.
     Against the historical E1 + ASRF no consistent difference was detected in either half. All point estimates against
     these references are positive. G1 + ASRF, the higher calibration reference, has stored rows only and was not
     compared with intervals. Its segment F1@50 is 0.904, 0.920 and 0.915 on selection and 0.972, 0.974 and 0.976 on
     calibration.
   - **The goal.** It is not met. The three-seed means of Ordered + B3 are 0.9544 on selection and 0.9817 on
     calibration, against minimums of 0.9444 and 0.9821. Selection clears its minimum by 0.010. Calibration falls 0.0004
     short of 0.9821 and 0.0022 short of the sensitivity target 0.9839, which is the G1 + ASRF calibration mean plus
     0.01. Seed 2 also misses the selection native floor (macro-F1 0.970 against 0.973). Shuffled + B3 has 0.935 and
     0.976 and Target-only + B3 0.859 and 0.922, and both fail floors as well.
   - **A4C and A5C.** Ordered + B3 never reads an A5C clip as A4C (0 in all six cells, against 1 and 1 for E1 + B3). It
     reads 2, 2 and 3 A4C clips as A5C on selection and 1, 2 and 2 on calibration, against 0 and 1 for E1 + B3. It keeps
     5, 3 and 5 of the 7 A4C/A5C X → Y → X middles on selection (E1 + B3 keeps 5) and all 7 on calibration.
   - **Short runs.** Ordered + B3 keeps 130, 134 and 136 of the 141 runs of at most 16 samples on selection and 128, 129
     and 127 of 134 on calibration. E1 + B3 keeps 129 and 125, and Original FACT 127 to 128 and 123 to 125.
   - **Extra cuts.** Cuts at same-view joins are lower than for E1 + B3 on selection (0.017, 0.004 and 0.004 against
     0.039) and slightly higher on calibration (0.025 in every seed against 0.020). Label switches inside native clips are
     0.039 to 0.046 and 0.013 to 0.025 per clip (E1 + B3 0.046 and 0.068). Boundary precision is 0.948 to 0.959 and 0.975
     to 0.980 (E1 + B3 0.943 and 0.951).
   - **Native classification.** No consistent difference from E1 + B3 was detected in either half, which is not
     equivalence. The macro-F1 is 0.979, 0.979 and 0.970 on selection and 0.993, 0.986 and 0.986 on calibration, against
     0.978 and 0.986. The clips fixed and newly missed are 2/2, 2/2 and 2/3 on selection and 1/0, 1/1 and 1/1 on
     calibration. Seed 2 misses the selection floor that E1 + B3 meets, and Ordered + B3 reads 2 to 3 A4C clips as A5C on
     selection where E1 + B3 reads none. Against C + B3 the macro-F1 is higher by +0.029 to +0.057 on selection, and
     Ordered + B3 fixes 4 to 6 selection clips and newly misses 1 to 2, but no consistent difference was detected.
   - **One SC4C clip.** All nine MViT models read every selection SC4C clip correctly with B3. All 14 E1- and B7-based
     reference rows miss the same one of the 13, reading it as PLAX, and E0 + B3 reads them all correctly. Target-only
     also reads it correctly, so this fix does not need context. It is one clip, and the MViT models differ from E1 and
     B7 in pretraining, preprocessing and training at once.
   - **Boundary placement.** Matched boundaries of Ordered + B3 lie 1.46 to 1.58 samples from the reference on average
     over seeds and halves, and those of all nine MViT models 1.15 to 1.63. E1 + B3 has 0.04 to 0.05 and Original FACT
     0.31 to 0.37. The window models' outputs change only between window centres 5 samples apart, so their boundaries
     snap to that grid. They still count when they fall within the 0.5 s tolerance.
4. **The fixed decision.** The readings of the task sheet's Task 4 were fixed in the configuration before any
   evaluation. Two rows apply.
   - **Multi-frame gain.** Ordered − Target-only and Shuffled − Target-only are consistent positive in both halves, and
     Ordered − Shuffled is not. The result is "a gain under multi-frame input, no added value of order shown". It is not
     equivalence, and repeated frames differ from the pretraining input distribution.
   - **System gain.** Ordered context + B3 is consistent positive against Target-only + B3 and against E1 + B3 in both
     halves.
   - **Rows that do not apply.** There is no order gain, no ranking-only gain and no recognition gain with a worse
     system. No control group beats Ordered context, and no group meets the goal.
5. **Cost.** The round used 5.6 GPU hours of its 24-hour cap.
   - **Training.** The nine runs took 0.471 hours each, 4.24 hours in all.
   - **Checks.** The check ledger holds 0.29 hours, covering the failed first check run, the passing one and a
     one-minute timing measurement on random tensors.
   - **Evaluation.** The first pass ran 0.74 hours, from 11:15 to 11:59 UTC by its log, and stopped at the reference
     stage. The second pass took 0.31 hours and reused the saved logits.
   - **Model size.** The encoder has 34,623,872 parameters including the projection. The marker stem and the head add
     16,677.
   - **Speed.** A batch-8 training update took 0.457 seconds in the check, with a peak of 12,374 MiB. Inference took 16
     to 19 ms per window and 3.3 to 3.9 ms per 10 Hz sample over the nine models, in batches of 32 windows from decoded
     frames. That is not a streaming latency.

## Terms

- A **window** holds 16 samples of the 10 Hz axis around a requested sample c, clamped at sequence ends
  (`window_frames`). Its **target slots** are the positions whose sample is c. The **marker** is a 0/1 video whose
  plane is 1 at the middle target slot; it enters the model through a zero-initialised stem beside the RGB stem.
- **Target-only** repeats the target frame in all 16 slots. **Shuffled context** feeds the same 16 frames as Ordered
  context with every target slot in place and the other slots permuted. **Ordered context** keeps the true order.
- **Native A4C/A5C balanced accuracy** is the mean of the A4C recall and the A5C recall over native windows of those two
  views, with the five-class argmax as the decision. The **margin AUC** ranks windows by logit(A5C) minus logit(A4C),
  with A5C positive.
- **E1 at the same samples** is the frozen E1 frame classifier read at each window's requested sample.
- **B3** is the simple change-point decoder; each model chooses its setting once on the calibration half.
- A difference is **consistent** in a half when all three seeds' 95% intervals exclude zero strictly on the same side.
  Otherwise no consistent difference is detected, which does not mean the systems are equal.
- The **paired bootstrap** resamples native clips or windows by session and streams by whole stream, with the same draws
  for both systems (2,000 draws, seed 20261002). Streams share recordings, so the stream intervals are approximate.

## 1. Native window recognition

Native windows of both halves, at the same centres for every system. Each cell gives the selection half, then the calibration half. Balanced accuracy and AUC use the A4C and A5C windows only (selection 1,076 and 85, calibration 565 and 201 windows); accuracy uses all native windows (3,444 and 2,950).

| System | Seed | Balanced accuracy | Margin AUC | Native window accuracy | A4C recall | A5C recall |
|---|---|---|---|---|---|---|
| Target-only | 0 | 0.874 / 0.844 | 0.925 / 0.926 | 0.959 / 0.952 | 0.937 / 0.876 | 0.812 / 0.811 |
| Target-only | 1 | 0.864 / 0.839 | 0.923 / 0.925 | 0.957 / 0.941 | 0.928 / 0.853 | 0.800 / 0.826 |
| Target-only | 2 | 0.883 / 0.850 | 0.925 / 0.929 | 0.959 / 0.948 | 0.931 / 0.883 | 0.835 / 0.816 |
| Shuffled context | 0 | 0.942 / 0.951 | 0.989 / 0.979 | 0.982 / 0.988 | 0.966 / 0.961 | 0.918 / 0.940 |
| Shuffled context | 1 | 0.930 / 0.936 | 0.991 / 0.983 | 0.985 / 0.985 | 0.977 / 0.956 | 0.882 / 0.915 |
| Shuffled context | 2 | 0.927 / 0.920 | 0.988 / 0.977 | 0.979 / 0.983 | 0.960 / 0.950 | 0.894 / 0.891 |
| Ordered context | 0 | 0.930 / 0.941 | 0.988 / 0.979 | 0.981 / 0.986 | 0.967 / 0.956 | 0.894 / 0.925 |
| Ordered context | 1 | 0.935 / 0.938 | 0.989 / 0.983 | 0.981 / 0.986 | 0.964 / 0.956 | 0.906 / 0.920 |
| Ordered context | 2 | 0.937 / 0.919 | 0.990 / 0.978 | 0.978 / 0.983 | 0.956 / 0.952 | 0.918 / 0.886 |
| E1 at the same samples | - | 0.831 / 0.817 | 0.913 / 0.884 | 0.949 / 0.940 | 0.921 / 0.897 | 0.741 / 0.736 |

Differences with 95% intervals (session-paired bootstrap of native windows).

| Comparison | Half | Balanced accuracy (seeds 0, 1, 2) | Margin AUC (seeds 0, 1, 2) | Verdict (balanced accuracy / AUC / accuracy) |
|---|---|---|---|---|
| Ordered − Shuffled | selection | -0.011 [-0.028, +0.002]; +0.005 [-0.025, +0.053]; +0.010 [-0.020, +0.036] | -0.002 [-0.006, +0.002]; -0.003 [-0.011, +0.005]; +0.001 [-0.004, +0.008] | none / none / none |
| Ordered − Shuffled | calibration | -0.010 [-0.021, +0.001]; +0.002 [-0.005, +0.012]; -0.002 [-0.011, +0.006] | +0.000 [-0.008, +0.013]; -0.000 [-0.004, +0.004]; +0.001 [-0.004, +0.009] | none / none / none |
| Ordered − Target-only | selection | +0.056 [+0.024, +0.083]; +0.071 [+0.028, +0.112]; +0.054 [+0.018, +0.095] | +0.063 [+0.012, +0.121]; +0.066 [+0.011, +0.130]; +0.064 [+0.008, +0.135] | positive / positive / positive |
| Ordered − Target-only | calibration | +0.097 [+0.045, +0.148]; +0.099 [+0.039, +0.156]; +0.069 [+0.038, +0.099] | +0.053 [+0.013, +0.104]; +0.057 [+0.015, +0.110]; +0.049 [+0.015, +0.096] | positive / positive / positive |
| Shuffled − Target-only | selection | +0.067 [+0.028, +0.105]; +0.065 [+0.017, +0.097]; +0.044 [+0.011, +0.085] | +0.065 [+0.014, +0.121]; +0.069 [+0.012, +0.130]; +0.063 [+0.007, +0.130] | positive / positive / positive |
| Shuffled − Target-only | calibration | +0.107 [+0.048, +0.167]; +0.096 [+0.035, +0.156]; +0.071 [+0.039, +0.100] | +0.053 [+0.014, +0.106]; +0.058 [+0.015, +0.111]; +0.048 [+0.014, +0.093] | positive / positive / positive |
| Target-only − E1 | selection | +0.043 [-0.020, +0.093]; +0.033 [-0.039, +0.081]; +0.052 [-0.026, +0.128] | +0.011 [-0.046, +0.062]; +0.009 [-0.056, +0.070]; +0.012 [-0.052, +0.072] | none / none / none |
| Target-only − E1 | calibration | +0.027 [-0.005, +0.058]; +0.023 [-0.003, +0.046]; +0.033 [-0.002, +0.065] | +0.043 [+0.003, +0.076]; +0.042 [+0.006, +0.070]; +0.045 [+0.001, +0.084] | none / positive / none |
| Shuffled context − E1 | selection | +0.111 [+0.030, +0.181]; +0.098 [+0.001, +0.174]; +0.096 [+0.020, +0.156] | +0.076 [+0.013, +0.124]; +0.078 [+0.013, +0.130]; +0.075 [+0.013, +0.122] | positive / positive / positive |
| Shuffled context − E1 | calibration | +0.134 [+0.058, +0.202]; +0.119 [+0.050, +0.180]; +0.104 [+0.049, +0.149] | +0.096 [+0.024, +0.167]; +0.100 [+0.025, +0.167]; +0.093 [+0.025, +0.158] | positive / positive / positive |
| Ordered context − E1 | selection | +0.099 [+0.021, +0.166]; +0.104 [+0.003, +0.187]; +0.106 [+0.018, +0.173] | +0.074 [+0.012, +0.120]; +0.075 [+0.012, +0.124]; +0.076 [+0.013, +0.123] | positive / positive / positive |
| Ordered context − E1 | calibration | +0.124 [+0.055, +0.186]; +0.121 [+0.055, +0.181]; +0.102 [+0.048, +0.147] | +0.096 [+0.024, +0.163]; +0.099 [+0.025, +0.168]; +0.094 [+0.025, +0.160] | positive / positive / positive |

## 2. Systems

Each cell gives the selection half, then the calibration half. The references reproduce their stored rows exactly.

| System | Segment F1@50 | Boundary recall | Boundary precision | Native macro-F1 | Label switches per native clip | Cuts at same-view joins | A5C read as A4C (clips) | A4C read as A5C (clips) |
|---|---|---|---|---|---|---|---|---|
| Target-only + B3 s0 | 0.909 / 0.950 | 0.919 / 0.952 | 0.964 / 0.979 | 0.968 / 0.986 | 0.042 / 0.038 | 0.009 / 0.015 | 1 / 1 | 2 / 1 |
| Target-only + B3 s1 | 0.831 / 0.909 | 0.990 / 0.992 | 0.738 / 0.846 | 0.952 / 0.986 | 0.437 / 0.662 | 0.096 / 0.085 | 1 / 0 | 4 / 2 |
| Target-only + B3 s2 | 0.836 / 0.906 | 0.984 / 0.990 | 0.757 / 0.855 | 0.977 / 0.986 | 0.440 / 0.544 | 0.109 / 0.090 | 1 / 0 | 1 / 2 |
| Shuffled context + B3 s0 | 0.934 / 0.975 | 0.958 / 0.977 | 0.927 / 0.963 | 0.970 / 0.986 | 0.046 / 0.030 | 0.013 / 0.020 | 0 / 0 | 3 / 2 |
| Shuffled context + B3 s1 | 0.932 / 0.972 | 0.976 / 0.980 | 0.901 / 0.949 | 0.977 / 0.986 | 0.067 / 0.042 | 0.039 / 0.035 | 1 / 0 | 1 / 2 |
| Shuffled context + B3 s2 | 0.939 / 0.980 | 0.955 / 0.972 | 0.958 / 0.975 | 0.962 / 0.993 | 0.035 / 0.021 | 0.009 / 0.020 | 0 / 0 | 4 / 1 |
| Ordered context + B3 s0 | 0.938 / 0.982 | 0.958 / 0.982 | 0.948 / 0.975 | 0.979 / 0.993 | 0.042 / 0.025 | 0.017 / 0.025 | 0 / 0 | 2 / 1 |
| Ordered context + B3 s1 | 0.960 / 0.984 | 0.969 / 0.987 | 0.959 / 0.980 | 0.979 / 0.986 | 0.039 / 0.017 | 0.004 / 0.025 | 0 / 0 | 2 / 2 |
| Ordered context + B3 s2 | 0.964 / 0.980 | 0.974 / 0.980 | 0.959 / 0.977 | 0.970 / 0.986 | 0.046 / 0.013 | 0.004 / 0.025 | 0 / 0 | 3 / 2 |
| E1 + B3 | 0.894 / 0.959 | 0.955 / 0.977 | 0.943 / 0.951 | 0.978 / 0.986 | 0.046 / 0.068 | 0.039 / 0.020 | 1 / 1 | 0 / 1 |
| E0 + B3 | 0.934 / 0.934 | 0.961 / 0.955 | 0.995 / 0.969 | 0.965 / 0.966 | 0.014 / 0.034 | 0.004 / 0.020 | 3 / 4 | 0 / 0 |
| B7 + B3 | 0.880 / 0.972 | 0.955 / 0.975 | 0.890 / 0.963 | 0.957 / 0.993 | 0.063 / 0.042 | 0.057 / 0.035 | 2 / 1 | 1 / 0 |
| C + B3 s0 | 0.854 / 0.965 | 0.966 / 0.990 | 0.864 / 0.959 | 0.922 / 0.975 | 0.067 / 0.072 | 0.100 / 0.030 | 4 / 2 | 2 / 1 |
| C + B3 s1 | 0.857 / 0.974 | 0.969 / 0.992 | 0.858 / 0.956 | 0.948 / 0.985 | 0.092 / 0.080 | 0.091 / 0.025 | 2 / 2 | 2 / 0 |
| C + B3 s2 | 0.852 / 0.970 | 0.971 / 0.992 | 0.828 / 0.959 | 0.941 / 0.975 | 0.099 / 0.084 | 0.091 / 0.030 | 1 / 2 | 2 / 1 |
| Original FACT s0 | 0.906 / 0.967 | 0.948 / 0.982 | 0.926 / 0.968 | 0.989 / 0.986 | 0.000 / 0.034 | 0.052 / 0.025 | 0 / 1 | 0 / 1 |
| Original FACT s1 | 0.898 / 0.961 | 0.950 / 0.975 | 0.928 / 0.968 | 0.978 / 0.986 | 0.000 / 0.030 | 0.043 / 0.030 | 1 / 1 | 0 / 1 |
| Original FACT s2 | 0.911 / 0.965 | 0.940 / 0.972 | 0.925 / 0.972 | 0.966 / 0.993 | 0.000 / 0.034 | 0.035 / 0.025 | 2 / 0 | 0 / 1 |
| paired ASRF s0 | 0.878 / 0.972 | 0.935 / 0.987 | 0.927 / 0.975 | 0.930 / 0.986 | 0.011 / 0.004 | 0.065 / 0.035 | 4 / 1 | 1 / 1 |
| paired ASRF s1 | 0.888 / 0.972 | 0.940 / 0.985 | 0.937 / 0.978 | 0.954 / 0.986 | 0.007 / 0.004 | 0.048 / 0.035 | 3 / 1 | 0 / 1 |
| paired ASRF s2 | 0.882 / 0.966 | 0.942 / 0.982 | 0.942 / 0.980 | 0.954 / 0.986 | 0.011 / 0.004 | 0.065 / 0.030 | 3 / 1 | 0 / 1 |
| historical E1 + ASRF s0 | 0.910 / 0.976 | 0.942 / 0.977 | 0.952 / 0.980 | 0.948 / 0.993 | 0.046 / 0.038 | 0.030 / 0.015 | 2 / 0 | 2 / 1 |
| historical E1 + ASRF s1 | 0.909 / 0.964 | 0.950 / 0.972 | 0.936 / 0.970 | 0.966 / 0.975 | 0.007 / 0.030 | 0.057 / 0.030 | 2 / 2 | 0 / 1 |
| historical E1 + ASRF s2 | 0.907 / 0.964 | 0.953 / 0.977 | 0.933 / 0.970 | 0.966 / 0.975 | 0.021 / 0.046 | 0.057 / 0.035 | 2 / 2 | 0 / 1 |

The direct argmax of each model (every sample takes its nearest window, no decoder).

| Direct argmax | Segment F1@50 | Boundary recall | Boundary precision | Native macro-F1 | Label switches per native clip | Cuts at same-view joins | A5C read as A4C (clips) | A4C read as A5C (clips) |
|---|---|---|---|---|---|---|---|---|
| Target-only s0 | 0.721 / 0.778 | 0.990 / 0.997 | 0.596 / 0.672 | 0.977 / 0.993 | 0.665 / 0.869 | 0.152 / 0.134 | 1 / 0 | 1 / 1 |
| Target-only s1 | 0.749 / 0.768 | 0.990 / 0.997 | 0.626 / 0.661 | 0.968 / 0.980 | 0.708 / 1.051 | 0.135 / 0.139 | 1 / 0 | 2 / 3 |
| Target-only s2 | 0.751 / 0.800 | 0.992 / 0.997 | 0.625 / 0.696 | 0.977 / 0.986 | 0.662 / 0.941 | 0.152 / 0.134 | 1 / 0 | 1 / 2 |
| Shuffled context s0 | 0.888 / 0.948 | 0.979 / 0.982 | 0.831 / 0.916 | 0.979 / 0.986 | 0.130 / 0.110 | 0.035 / 0.040 | 0 / 0 | 2 / 2 |
| Shuffled context s1 | 0.898 / 0.946 | 0.979 / 0.977 | 0.846 / 0.905 | 0.977 / 0.986 | 0.134 / 0.143 | 0.057 / 0.045 | 1 / 0 | 1 / 2 |
| Shuffled context s2 | 0.895 / 0.958 | 0.979 / 0.980 | 0.842 / 0.926 | 0.962 / 0.993 | 0.148 / 0.122 | 0.035 / 0.035 | 0 / 0 | 4 / 1 |
| Ordered context s0 | 0.893 / 0.961 | 0.979 / 0.987 | 0.846 / 0.942 | 0.970 / 0.986 | 0.095 / 0.093 | 0.043 / 0.035 | 0 / 0 | 3 / 2 |
| Ordered context s1 | 0.925 / 0.961 | 0.984 / 0.990 | 0.872 / 0.945 | 0.988 / 0.980 | 0.113 / 0.101 | 0.030 / 0.040 | 0 / 0 | 1 / 3 |
| Ordered context s2 | 0.926 / 0.960 | 0.984 / 0.982 | 0.874 / 0.940 | 0.970 / 0.986 | 0.116 / 0.097 | 0.026 / 0.035 | 0 / 0 | 3 / 2 |

The B3 setting each model chose on the calibration half. Where two models chose different settings, the setting is
part of their system difference.

| Model | B3 status | Half-window | Quantile | Feasible settings of 40 |
|---|---|---|---|---|
| Target-only s0 | feasible | 11 | 0.999 | 22 |
| Target-only s1 | feasible | 9 | 0.8 | 6 |
| Target-only s2 | feasible | 5 | 0.97 | 5 |
| Shuffled context s0 | feasible | 7 | 0.999 | 39 |
| Shuffled context s1 | feasible | 7 | 0.995 | 39 |
| Shuffled context s2 | feasible | 9 | 0.999 | 38 |
| Ordered context s0 | feasible | 9 | 0.999 | 40 |
| Ordered context s1 | feasible | 9 | 0.999 | 6 |
| Ordered context s2 | feasible | 9 | 0.999 | 38 |

## 3. Comparisons

Segment F1@50 differences with 95% intervals (paired bootstrap; streams reuse recordings, so these intervals are approximate).

| Comparison (B3 rows) | Half | Segment F1@50 (seeds 0, 1, 2) | Native macro-F1 (seeds 0, 1, 2) | Native clips fixed/new | Verdict |
|---|---|---|---|---|---|
| Ordered − Shuffled | selection | +0.004 [-0.012, +0.021]; +0.028 [+0.007, +0.050]; +0.025 [+0.006, +0.046] | +0.009; +0.001; +0.008 | 1/0, 2/2, 2/1 | none |
| Ordered − Shuffled | calibration | +0.007 [-0.003, +0.018]; +0.011 [+0.004, +0.020]; +0.000 [-0.007, +0.007] | +0.007; +0.000; -0.007 | 1/0, 0/0, 0/1 | none |
| Ordered − Target-only | selection | +0.030 [+0.002, +0.058]; +0.130 [+0.089, +0.173]; +0.128 [+0.095, +0.162] | +0.011; +0.027; -0.007 | 3/2, 5/2, 1/2 | consistent positive |
| Ordered − Target-only | calibration | +0.031 [+0.013, +0.052]; +0.075 [+0.053, +0.099]; +0.073 [+0.050, +0.099] | +0.007; +0.000; +0.000 | 1/0, 2/2, 1/1 | consistent positive |
| Shuffled − Target-only | selection | +0.025 [-0.004, +0.055]; +0.102 [+0.064, +0.140]; +0.103 [+0.065, +0.141] | +0.002; +0.025; -0.015 | 2/2, 4/1, 2/4 | none |
| Shuffled − Target-only | calibration | +0.025 [+0.006, +0.046]; +0.064 [+0.043, +0.087]; +0.073 [+0.050, +0.099] | +0.001; +0.000; +0.007 | 1/1, 2/2, 1/0 | consistent positive |
| Ordered − (E1 + B3) | selection | +0.045 [+0.014, +0.075]; +0.066 [+0.039, +0.096]; +0.070 [+0.044, +0.099] | +0.001; +0.001; -0.008 | 2/2, 2/2, 2/3 | consistent positive |
| Ordered − (E1 + B3) | calibration | +0.023 [+0.010, +0.035]; +0.025 [+0.014, +0.037]; +0.021 [+0.009, +0.032] | +0.007; +0.001; +0.001 | 1/0, 1/1, 1/1 | consistent positive |
| Ordered − (E0 + B3) | selection | +0.004 [-0.020, +0.028]; +0.026 [+0.004, +0.051]; +0.030 [+0.009, +0.053] | +0.014; +0.014; +0.005 | 3/3, 3/3, 3/4 | none |
| Ordered − (E0 + B3) | calibration | +0.048 [+0.027, +0.072]; +0.050 [+0.028, +0.074]; +0.046 [+0.024, +0.070] | +0.027; +0.020; +0.020 | 5/1, 5/2, 5/2 | consistent positive |
| Ordered − (B7 + B3) | selection | +0.059 [+0.032, +0.086]; +0.080 [+0.053, +0.111]; +0.085 [+0.059, +0.112] | +0.022; +0.022; +0.013 | 3/1, 3/1, 3/2 | consistent positive |
| Ordered − (B7 + B3) | calibration | +0.010 [-0.002, +0.023]; +0.011 [-0.001, +0.026]; +0.008 [-0.004, +0.022] | +0.000; -0.007; -0.007 | 1/1, 1/2, 1/2 | none |
| Ordered − (C + B3) | selection | +0.084 [+0.053, +0.116]; +0.104 [+0.068, +0.140]; +0.112 [+0.079, +0.147] | +0.057; +0.031; +0.029 | 6/1, 4/1, 4/2 | consistent positive |
| Ordered − (C + B3) | calibration | +0.017 [+0.004, +0.032]; +0.009 [-0.002, +0.021]; +0.010 [-0.003, +0.025] | +0.018; +0.001; +0.011 | 3/0, 2/2, 3/1 | none |
| Ordered − Original FACT | selection | +0.032 [+0.004, +0.062]; +0.062 [+0.031, +0.094]; +0.053 [+0.026, +0.084] | -0.010; +0.001; +0.004 | 1/2, 2/2, 3/3 | consistent positive |
| Ordered − Original FACT | calibration | +0.014 [-0.001, +0.030]; +0.023 [+0.008, +0.041]; +0.014 [+0.002, +0.028] | +0.007; +0.001; -0.007 | 1/0, 1/1, 0/1 | none |
| Ordered − paired ASRF | selection | +0.061 [+0.029, +0.092]; +0.073 [+0.043, +0.104]; +0.082 [+0.051, +0.114] | +0.048; +0.025; +0.016 | 6/2, 4/2, 4/3 | consistent positive |
| Ordered − paired ASRF | calibration | +0.010 [-0.002, +0.022]; +0.012 [+0.001, +0.024]; +0.014 [+0.002, +0.026] | +0.007; +0.001; +0.001 | 1/0, 1/1, 1/1 | none |
| Ordered − historical E1 + ASRF | selection | +0.028 [-0.000, +0.056]; +0.051 [+0.021, +0.082]; +0.057 [+0.030, +0.085] | +0.031; +0.012; +0.004 | 4/1, 3/2, 3/3 | none |
| Ordered − historical E1 + ASRF | calibration | +0.006 [-0.005, +0.016]; +0.019 [+0.007, +0.034]; +0.015 [+0.002, +0.031] | +0.000; +0.011; +0.011 | 0/0, 3/1, 3/1 | none |

| Comparison (direct argmax) | Half | Segment F1@50 (seeds 0, 1, 2) | Verdict |
|---|---|---|---|
| Ordered − Shuffled | selection | +0.005 [-0.013, +0.022]; +0.026 [+0.001, +0.053]; +0.031 [+0.009, +0.054] | none |
| Ordered − Shuffled | calibration | +0.013 [-0.006, +0.032]; +0.015 [-0.003, +0.034]; +0.003 [-0.015, +0.021] | none |
| Ordered − Target-only | selection | +0.172 [+0.135, +0.207]; +0.176 [+0.134, +0.218]; +0.175 [+0.140, +0.210] | consistent positive |
| Ordered − Target-only | calibration | +0.183 [+0.146, +0.221]; +0.193 [+0.159, +0.228]; +0.160 [+0.128, +0.192] | consistent positive |
| Shuffled − Target-only | selection | +0.167 [+0.130, +0.204]; +0.150 [+0.106, +0.192]; +0.144 [+0.104, +0.181] | consistent positive |
| Shuffled − Target-only | calibration | +0.170 [+0.135, +0.206]; +0.177 [+0.143, +0.213]; +0.158 [+0.124, +0.190] | consistent positive |

## 4. Mixed windows and sequence ends

Mixed windows are windows of the constructed streams whose 16 frames hold at least two views. Accuracy is the share whose output names the requested sample's view, overall and by how many of the 16 frames show that view.

| Group | Half | Mixed windows correct (seeds 0, 1, 2) | 8 of 16 frames of the target view | 9-11 of 16 | 12-15 of 16 |
|---|---|---|---|---|---|
| Target-only | selection | 0.947, 0.953, 0.957 | 0.957, 0.947, 0.947 | 0.955, 0.965, 0.965 | 0.938, 0.944, 0.952 |
| Target-only | calibration | 0.951, 0.947, 0.956 | 0.986, 0.959, 0.973 | 0.964, 0.964, 0.971 | 0.935, 0.931, 0.940 |
| Shuffled context | selection | 0.900, 0.902, 0.903 | 0.543, 0.543, 0.585 | 0.877, 0.872, 0.869 | 0.966, 0.974, 0.973 |
| Shuffled context | calibration | 0.920, 0.917, 0.909 | 0.589, 0.616, 0.534 | 0.889, 0.876, 0.873 | 0.982, 0.985, 0.979 |
| Ordered context | selection | 0.903, 0.909, 0.912 | 0.628, 0.702, 0.681 | 0.871, 0.866, 0.877 | 0.967, 0.973, 0.971 |
| Ordered context | calibration | 0.915, 0.914, 0.910 | 0.575, 0.630, 0.644 | 0.875, 0.867, 0.856 | 0.986, 0.985, 0.985 |

Selection holds 1,449 mixed windows and calibration 1,440. Of these, 94 and 73 have 8 frames of the target view, 627
and 646 have 9 to 11, and 728 and 721 have 12 to 15. None has fewer than 8, and 27 and 23 hold three views or more.

Fixed and new errors on mixed windows, per seed, with their distinct source recordings and sessions.

| Comparison on mixed windows | Half | Fixed windows (recordings, sessions) | New errors (recordings, sessions) |
|---|---|---|---|
| Ordered − Shuffled | selection | 47 (35, 22); 56 (40, 22); 48 (36, 20) | 42 (30, 19); 46 (32, 20); 35 (30, 19) |
| Ordered − Shuffled | calibration | 45 (36, 20); 50 (43, 24); 45 (33, 20) | 52 (40, 22); 55 (50, 21); 44 (35, 21) |
| Ordered − Target-only | selection | 43 (21, 14); 39 (21, 13); 36 (20, 13) | 106 (59, 26); 103 (59, 23); 102 (57, 23) |
| Ordered − Target-only | calibration | 56 (30, 18); 59 (32, 19); 46 (28, 17) | 107 (70, 29); 107 (73, 29); 112 (72, 29) |
| Shuffled − Target-only | selection | 41 (18, 13); 40 (19, 12); 35 (20, 13) | 109 (67, 27); 114 (66, 27); 114 (62, 26) |
| Shuffled − Target-only | calibration | 52 (28, 18); 58 (31, 19); 47 (31, 18) | 96 (63, 25); 101 (65, 28); 114 (67, 28) |

This table splits all windows by whether the marker is at slot 8. A marker away from slot 8 means that the window
was clamped at a sequence end, or that the sequence is shorter than 16 samples and its window repeats samples. These
windows are not less accurate. No interval was computed.

| Group | Half | Marker away from slot 8, correct (seeds 0, 1, 2) | Marker at slot 8, correct |
|---|---|---|---|
| Target-only | selection | 0.961, 0.961, 0.959 | 0.943, 0.947, 0.951 |
| Target-only | calibration | 0.954, 0.950, 0.956 | 0.954, 0.947, 0.954 |
| Shuffled context | selection | 0.983, 0.989, 0.981 | 0.954, 0.960, 0.957 |
| Shuffled context | calibration | 0.987, 0.984, 0.984 | 0.968, 0.967, 0.964 |
| Ordered context | selection | 0.982, 0.984, 0.983 | 0.956, 0.961, 0.959 |
| Ordered context | calibration | 0.984, 0.979, 0.982 | 0.970, 0.970, 0.966 |

Of all windows, 2,018 of 7,666 on selection and 1,767 of 7,021 on calibration have the marker away from slot 8.

## 5. Reference units

Matched runs of the multi streams, selection then calibration. A short run has at most 16 samples, and an X-Y-X middle is a run whose two neighbours share a view different from its own. The boundary offset is the mean absolute distance of matched predicted changes from the reference changes.

| System | Short runs kept (141 / 134) | Short A4C or A5C runs (65 / 67) | X-Y-X middles (69 / 80) | A4C/A5C middles (7 / 7) | Mean boundary offset (samples) |
|---|---|---|---|---|---|
| Target-only + B3 s0 | 119 / 123 | 49 / 56 | 59 / 77 | 2 / 4 | 1.20 / 1.15 |
| Target-only + B3 s1 | 137 / 129 | 61 / 62 | 63 / 79 | 6 / 7 | 1.23 / 1.15 |
| Target-only + B3 s2 | 136 / 130 | 60 / 63 | 64 / 79 | 6 / 7 | 1.23 / 1.16 |
| Shuffled context + B3 s0 | 130 / 128 | 55 / 63 | 65 / 79 | 4 / 7 | 1.62 / 1.51 |
| Shuffled context + B3 s1 | 136 / 128 | 60 / 63 | 66 / 79 | 5 / 7 | 1.63 / 1.47 |
| Shuffled context + B3 s2 | 129 / 126 | 56 / 63 | 63 / 80 | 5 / 7 | 1.58 / 1.58 |
| Ordered context + B3 s0 | 130 / 128 | 54 / 66 | 66 / 80 | 5 / 7 | 1.55 / 1.46 |
| Ordered context + B3 s1 | 134 / 129 | 58 / 65 | 64 / 80 | 3 / 7 | 1.58 / 1.47 |
| Ordered context + B3 s2 | 136 / 127 | 60 / 65 | 66 / 80 | 5 / 7 | 1.55 / 1.49 |
| E1 + B3 | 129 / 125 | 56 / 59 | 63 / 80 | 5 / 7 | 0.05 / 0.04 |
| C + B3 s0 | 133 / 127 | 60 / 61 | 61 / 80 | 6 / 7 | 1.25 / 1.22 |
| C + B3 s1 | 134 / 131 | 61 / 65 | 61 / 80 | 6 / 7 | 1.23 / 1.15 |
| C + B3 s2 | 134 / 129 | 61 / 63 | 62 / 80 | 6 / 7 | 1.25 / 1.19 |
| Original FACT s0 | 127 / 124 | 54 / 58 | 60 / 79 | 1 / 7 | 0.33 / 0.31 |
| Original FACT s1 | 127 / 123 | 56 / 57 | 60 / 77 | 3 / 5 | 0.37 / 0.32 |
| Original FACT s2 | 128 / 125 | 55 / 59 | 59 / 78 | 3 / 6 | 0.34 / 0.35 |
| paired ASRF s0 | 125 / 127 | 53 / 61 | 60 / 79 | 4 / 7 | 0.05 / 0.06 |
| paired ASRF s1 | 123 / 128 | 51 / 61 | 60 / 79 | 3 / 7 | 0.06 / 0.06 |
| paired ASRF s2 | 125 / 125 | 54 / 60 | 60 / 79 | 4 / 7 | 0.09 / 0.07 |

## 6. What this round does not show

- **No value of order.** Ordered context did not consistently beat Shuffled context, so the round gives no evidence
  that the true frame order adds recognition information beyond the same frames.
- **No motion mechanism.** The multi-frame gain is not attributed to outflow-tract motion, cardiac phase or probe
  movement. Target-only differs from the other groups in its number of distinct images and in how far its input is from
  the pretraining clips, and this round cannot separate the two.
- **No mechanism for the system gain.** Ordered + B3 differs from E1 + B3, C, FACT and ASRF in pretraining,
  architecture, normalisation, training and decoder setting at once. The comparison does not show that local patches
  beat global frame vectors.
- **Not the goal.** No group reaches the goal or meets every floor. The development halves are reused, so these numbers
  are no independent validation, and the test split stays sealed.
- **Not clinical scanning.** Constructed streams and public recording labels are not real continuous scans or expert
  frame labels.

## Deviations and launch record

1. **Two helper files beyond the task sheet's list.** The pre-queue checks live in `scripts/check_raw_video_context.py`
   (called by `train_raw_video_context.py --check`), and the pure summaries live in
   `scripts/raw_video_context_tables.py`.
2. **The boundary check after the first check run.** The first check run, on the code of c527046, failed the
   short-schedule check across the unfreeze boundary. The resumed run differed from the uninterrupted one by 2.04e-5 in
   parameters, above the 1e-5 the configuration had applied, and two uninterrupted runs differed by the same amount.
   The only operator flagged without a deterministic CUDA kernel was `max_pool3d_with_indices_backward_cuda`, and the
   drift appeared only after the encoder was unfrozen. As the task sheet asks, the operator and its tolerance are
   recorded. The check now requires an exact state round-trip, losses within 1e-4 and parameter drift within 2.1 times
   each optimiser group's summed learning rate. That bound is loose, about the largest drift two Adam runs can reach
   when they differ only in the sign of near-zero gradients. The evidence for a correct resume is the exact round-trip
   and losses that agreed within 2.4e-7. The six-update check on the formal schedule kept 1e-5 and passed. The first
   run's check record was overwritten, and its numbers survive in the configuration note and `code_review.md`.
3. **Two fixed definitions.** The inference permutation's `centre_index` is the window's requested sample on the 10 Hz
   axis. The MViT keeps the constructor defaults of the official loader, stochastic depth 0.2 and dropout 0.5 before
   the projection.
4. **Report fixes.** Before the evaluation, 69b3d77 added the identity-checked reuse of saved logits and fixed two
   faults found on re-reading: the system table would have tried to print the stored B3 setting as a row, and the id
   guard would have scanned the tables for overly short stream-group names. During the evaluation, the first pass
   stopped at the reference stage because it read the stored C rows under a key that does not exist (`rows`). 1899fd1
   fixed the key. The second pass reused the saved per-window logits after checking each model's identity, sequences,
   centres, markers and slot orders, and the nine models' rows printed the same values in both passes. No result was
   chosen or changed between the passes.

## Fact-check

An independent fact-check recomputed the results from the saved per-window outputs, mostly with its own code, and
found every number right. Its corrections to the wording are in this README.

- **Recomputed exactly.** The windows, centres, markers and slot orders of all nine models, including the Shuffled
  permutation rule. The recognition metrics and their 108 bootstrap differences. The 396 fields of the argmax and B3
  system rows, and the B3 setting re-chosen by the calibration rule. The segment F1 and native macro-F1 bootstraps of
  the within-group comparisons and against E1 + B3. The verdicts, the goal and the decision rows. The native and
  mixed-window fixes and new errors with their recordings and sessions. The short-run units and offsets of the nine
  MViT rows and E1 + B3.
- **Re-decoded.** 64 selection windows of two frozen models gave bit-identical logits.
- **Integrity.** The frozen weights match their digests, the run identities recompute, and every run has 3,750 updates.
  The training split and the development halves share no recording and no session.
- **Ids.** No recording, session or sequence id appears in any file here.
- **Not recomputed.** The bootstraps against references other than E1 + B3 were checked against the consistency rule
  only, and the units of the C, FACT and ASRF rows were not recounted.
- **One limit.** The check that each native window's target frame is the frame E1's cache used holds by construction,
  because both read the same frame-index field. It is no independent pixel evidence, and streams were not checked this
  way.

## Commands

Run on the server in the project `.venv`, one GPU job at a time, with `ECHO_GPU_MEM_FRACTION=0.6`.

```bash
C=configs/experiments/raw_video_context_v1.yaml
python scripts/prepare_raw_video_context.py --config $C                  # manifests, pixel digests, initial states
python scripts/train_raw_video_context.py --config $C --check            # the checks before the queue
python scripts/queue_raw_video_context.py --config $C                    # the nine formal runs
python scripts/report_raw_video_context.py --config $C --evaluate        # freezing and the single evaluation
python -m pytest tests/test_raw_video_context.py tests/test_raw_video_context_report.py -q
```

## Provenance

- **Code.** The model and training code are at c527046 with the review fixes of c749dd2. The report code is at 1899fd1.
- **Sources.** The configuration pins the task sheet, the EchoPrime commit 03874a5f, its loader and checkpoint
  (7ca32e8b…), torchvision 0.29.0's MViT source, the environment (torch 2.14.0, CUDA 13.0), the protocol and the
  centre-supervision round's S/C window manifests. Every digest was checked before use.
- **Frozen weights.** On the server, `models_frozen/raw_video_context_v1/{target_only,shuffled,ordered}_s{0,1,2}` hold
  the final weights, the run manifest, the initial state of the new parameters, the manifest index and the
  configuration. `summary.json` holds every run's identity and final sha256.
- **Server outputs.** `runs/raw_video_context_v1/evaluation/{selection,calibration}/` holds each model's per-window
  logits, slot orders, markers and centres and its per-sample probabilities. Recording and session ids stay on the
  server.

## Files

- `README.md` is this file. `code_review.md` is the implementation review and the boundary-check record.
- `preparation.json` and `checks.json` hold the preparation and the final check run.
- `summary.json` holds every system row, the recognition tables, every comparison with its bootstrap and change tables,
  the verdicts, the goal, the decision, the units and the costs. `report_tables.md` holds the generated tables.
- `training_curves.png` shows the training curves, and `input_tokens.png` shows one training window as each group feeds
  it.
- None of these files contains a recording or session id.
