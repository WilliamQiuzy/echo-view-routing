# Trained token readout round (2026-10-14)

This round asked whether the time tokens that the video dense supervision round (A02) trained with view labels improve
the segmentation when the system reads them directly. Nothing was trained. The nine frozen A02 models keep their
weights, their 16-frame Sparse input windows and their requests. For each window, a fixed rule that uses only sample
positions picks one of the 8 trained time anchors. That anchor's saved time logits then replace the CLS logits as the
window's prediction at the requested sample. Probabilities, the nearest-centre mapping to 10 Hz and B3 are A02's. The
task sheet is `docs/superpowers/plans/2026-10-14-trained-token-readout.md`.

Five kinds of system are compared.

- **Aligned token readout** reads the time logits of A02's aligned arm. It is the only primary candidate.
- **Aligned CLS** is the same models' original A02 output with A02's stored B3 settings.
- **CLS-only token readout** and **Shuffled token readout** apply the same rule to the other two A02 arms. In training,
  their time logits had weight 0 (A02's "CLS continuation" arm) or shuffled labels.
- **CLS-only CLS** and **Shuffled CLS** are those arms' original outputs with their stored settings.
- **Frozen Ordered + B3** is the raw-video context round's (V01) saved system, the current strong reference.

This is a readout check of already trained models. It is not new training, a new architecture or a validation on real
continuous scans.

**What kind of evidence this is.**
- The development halves have been used in many earlier rounds, so this evaluation is not an independent validation.
- The hold-out is by session, not by patient.
- Constructed streams are spliced from separate recordings and are not real scans. They reuse recordings, so their units
  are not independent cases. The three seeds share the same cases.
- Native clips carry the public recording label, not expert frame labels. No label was changed.
- Each time token has interacted with every other position of its window, so its logits are not a single-frame
  recognition without context.
- The test split and EchoNet-Pediatric were not read.

**Order of work.** All times are UTC on 2026-10-14.
- a2dd798 committed the task sheet and the pending index entry A03 at 16:15. ac86622 committed the fixed configuration
  at 16:25, before any code.
- 5e19f81 committed the selector, source and adapter tests, which failed at import. 783af48 then committed the module
  with the prepare and check stages. The first preparation and check ran at 16:30 and 16:31.
- 3620640 committed the report tests, which failed at import. 95b56e4 committed the tables module, the selection stage
  and a calibration-only dry run.
- An independent implementation review passed the code with fixes (`code_review.md`). 0e62879 made the fixes, and the
  reviewer verified them. The preparation and check ran again at 17:54, and 8b0380d committed the records.
- The first calibration ran from 17:58 to 18:03 and the first dry run from 18:03 to 18:10. The dry run showed a
  table-formatting defect, which aada40b fixed. The first choice record was moved aside (see "Deviations").
- The preparation and check ran again at 18:13. The calibration ran from 18:14 to 18:19 and the dry run from 18:19 to
  18:26.
- 5c9bf49 committed the choice record at 18:26:10. The selection run started at 18:26:37 with that record's sha256 and
  finished at 18:34.
- An independent fact-check then recomputed the results (see "Fact-check").

## Answers

Values are given for seeds 0, 1 and 2 in that order. A difference is consistent only when all three seeds' 95% intervals
exclude zero strictly on the same side. All systems use B3 unless the sentence says otherwise.

1. **Does the aligned token readout improve the segmentation over the same weights' CLS?** No consistent difference was
   detected in either half.
   - **The primary metric.** Aligned token − Aligned CLS multi-stream segment F1@50 is −0.006, −0.026 and −0.008 on
     selection and −0.004, −0.014 and +0.000 on calibration. Seed 1's intervals lie below zero in both halves, and the
     other intervals contain zero. Five of the six point estimates are negative.
   - **Other metrics.** None of the five other paired metrics shows a consistent difference either. They are native
     macro-F1, label switches per native clip, boundary recall, boundary F1 and cuts at same-view joins.
2. **Does the token readout name the requested sample's view better on the same windows?** Yes, on mixed windows.
   - **All mixed windows.** Aligned token − Aligned CLS target accuracy is +0.029, +0.025 and +0.020 on selection and
     +0.016, +0.015 and +0.026 on calibration. This is consistent positive in both halves, and the stream sensitivity
     agrees. The token readout fixes 42 to 50 windows per seed on selection and 33 to 47 on calibration, from 15 to 20
     sessions. It newly misreads 8 to 13 windows, and five of these six counts rest on fewer than 10 sessions.
   - **By the target view's share of the 16 samples.** The differences are +0.085 to +0.137 on the windows where the
     target view covers 8 of 16 samples and +0.023 to +0.051 where it covers 9 to 11, over seeds and halves. Both are
     consistent positive in both halves.
   - **No consistent difference was detected on the other described subsets.** These are A4C/A5C pair windows, native
     windows, exact anchors and non-exact anchors.
     - On A4C/A5C pair windows, the selection differences are +0.063, +0.063 and +0.054. Seed 0's lower bound is exactly
       zero, so the half is not consistent. The calibration differences are 0.000 to +0.008.
     - On native windows, the calibration differences are −0.002, −0.005 and −0.004, and seed 2's interval lies below
       zero.
   - **Not A02's comparison.** A02 compared two arms' CLS outputs. On selection, its aligned − CLS continuation segment
     F1@50 was +0.011, −0.003 and +0.034. Its aligned − CLS continuation difference in all-mixed CLS target accuracy on
     selection was −0.009, −0.012 and −0.008, and two of those intervals lay below zero. This round compares two readouts
     of the same aligned weights on the same windows.
3. **What came with the local gain.** The round does not show why no consistent segment F1@50 gain was detected
   alongside the gain on mixed windows. It records the following.
   - **Direct argmax.** The token readout switches labels more often inside native clips in all six seed–half cells:
     0.135 to 0.190 switches per clip, against 0.076 to 0.134 for CLS. This difference is consistent on calibration
     only. Its stream boundary precision is lower: 0.822 to 0.929, against 0.845 to 0.954. Its segment F1@50 − CLS is
     −0.014, −0.010 and −0.015 on selection, and −0.018, −0.018 and −0.014 on calibration, where it is consistent
     negative.
   - **B3.** The token readout's boundary precision is lower in five of six seed–half cells. The largest gap is seed 1 on
     selection, 0.879 against 0.941. Its boundary recall is higher on selection in seeds 0 and 1 (+0.016 and +0.026,
     both intervals above zero) and lower in seed 2. With B3, no consistent difference in label switches per native clip
     was detected.
   - **Short units.** On selection, the token readout matched more segments of at most 16 samples in seeds 0 and 1 (137
     and 141 of 141, against 131 and 132). It matched fewer in seed 2 (135 against 138). On calibration the counts are
     within three of each other. These counts rest on fewer than 10 sessions, as the missed segments of each token or
     CLS cell come from at most 6 sessions.
4. **Does the aligned token readout beat the strong reference or the token controls?** It does not beat the strong
   reference. Its one consistent result is against the CLS-only token control.
   - **Aligned token − frozen Ordered + B3.** The differences are −0.002, −0.035 and −0.011 on selection and +0.000,
     −0.018 and +0.000 on calibration. No consistent difference was detected. Seed 1's intervals lie below zero in both
     halves.
   - **Aligned token − Shuffled token.** The differences are +0.021, −0.033 and −0.002 on selection and +0.002, −0.010
     and +0.004 on calibration. No consistent difference was detected.
   - **Aligned token − CLS-only token.** It is consistent positive on calibration (+0.028, +0.061 and +0.016), but not on
     selection (+0.018, +0.003 and +0.080).
     - CLS-only token seeds 1 and 2 had no feasible B3 setting, so they use the 9 / 0.95 diagnostic setting, and these
       comparisons are flagged.
     - The task sheet does not count a win over this control as progress.
5. **The controls' own readouts.**
   - **CLS-only token − CLS-only CLS.** All-mixed accuracy is consistent negative on selection (−0.067, −0.068 and
     −0.038). The argmax segment F1@50 is consistent negative in both halves. With B3 no consistent difference was
     detected, and its seeds 1 and 2 use diagnostic settings.
   - **Shuffled token − Shuffled CLS.** No consistent difference was detected in segment F1@50 or all-mixed accuracy.
     On the 8-of-16 windows it is consistent negative on selection.
   - **The consistent same-target gain appeared only in the aligned arm.** It is the only arm whose time tokens were
     trained with aligned labels. This round does not show that this training caused the gain.
   - **At the CLS numeric setting,** CLS-only token − CLS-only CLS is consistent negative in segment F1@50 in both
     halves.
6. **Native recognition and constraints.** The aligned token system fails one constraint cell.
   - **Native macro-F1.** It is 0.9786, 0.9869 and 0.9700 on selection and 0.9856, 0.9861 and 0.9856 on calibration.
     Seed 2's selection value is below the 0.973 floor.
   - **Boundary recall.** It is at least 0.95 in every seed and half.
   - **Changed clips against Aligned CLS.** The token readout reads one A5C clip as A4C in each of three cells: seed 1
     on selection and seeds 0 and 2 on calibration. Aligned CLS reads no A5C clip as A4C in any cell. The token readout
     fixes one A4C clip that CLS reads as A5C, in seed 1 on selection. In seed 2 on selection it newly reads one A4C
     clip as SC4C. Each count is one clip from one session.
   - **The loss note.** Native macro-F1 and boundary recall show no consistent difference from CLS, so the note for
     native or recall losses does not apply.
7. **Constraints and goal.** No system meets the goal.
   - **Three-seed means.** Aligned token has 0.9382 and 0.9758, Aligned CLS 0.9515 and 0.9817, and frozen Ordered + B3
     0.9544 and 0.9817. The minimums are 0.9444 and 0.9821. Aligned token misses them by 0.0062 and 0.0063.
   - **Other systems.** Shuffled token has 0.9428 and 0.9773 and fails seed 2's selection boundary recall (0.9476).
     CLS-only token has 0.9043 and 0.9407, and four of its cells have no feasible B3 setting.
8. **Decoding sensitivity.** Decoded with the same arm's A02 CLS numeric B3 setting, Aligned token − Aligned CLS is
   −0.006, +0.018 and −0.015 on selection and −0.005, −0.010 and −0.004 on calibration. No consistent difference was
   detected. The direct argmax rows are in answer 3. Neither row selects a system.
9. **The fixed decision.** Two readings apply: labels learned, no usable gain, and end.
   - **Labels learned, no usable gain.**
     - The same-target classification part of a usable readout holds in both halves. The segment F1@50 part does not,
       and the aligned token system fails one constraint cell.
     - The readings record this as a classification gain without a segment gain.
     - This configuration ends. An interval that contains zero is not equality, and this round does not show that every
       time-token readout fails.
   - **End.** No anchor rule, fusion, probability rule, threshold search, loss or training follows from this round.
   - **Not applying.** Complete model progress and a usable token readout do not apply.
     - No half had a stream-sensitivity conflict.
     - No consistent negative primary verdict, native loss or recall loss was detected.
     - The result is not a win over the controls only.
10. **Earlier systems.** Their stored rows are listed in section 7. No paired comparison with them was computed in this
    round.
11. **Cost.** The stages used 0.518 CPU hours of the 2-hour cap, including the superseded preparation, check,
    calibration and dry run. The review's probes added about 0.05 hours and the fact-check about 0.03 hours, about 0.60
    CPU hours in all. No GPU was used.

## Terms

- **The windows** are A02's. Requests are every fifth sample from sample 2 (`window_centres(n, 5)`), plus the last sample
  when needed. Each window takes 16 samples (`window_frames(c, n, 16, 16)`). They start 8 samples before the request
  and are clamped at the sequence ends. In the 28 selection and 19 calibration native clips shorter than 16 samples,
  the window repeats samples. The marker slot m is the middle slot that holds the request c.
- **The anchors** are input slots 0, 2, ..., 14, the slots A02 supervised. The rule takes the unique anchor j that
  minimises (|s[a_j] − c|, |a_j − m|, j). It prefers the anchor whose sample is nearest the request, then the anchor
  nearest the marker slot, then the smaller j. Labels, confidence, attention, sensitivity and native or mixed flags do
  not enter it.
- **The readout** takes `time_logits[i, j_i, :]`, applies A02's five-class softmax and maps the window probabilities to
  the 10 Hz axis by the nearest request, as A02 does. The prediction stays at the request c and is not moved to the
  anchor's sample.
- **The anchor offset** is s[a_j] − c. A window is exact when the offset is 0 and non-exact otherwise. Every window
  stays in the main evaluation.
- **B3** for each token system was chosen once on calibration from A02's 40 settings: half-window 5, 7, 9, 11 or 15
  and threshold quantile 0.80 to 0.999, with NMS radius 5. The choice needs native macro-F1 of at least 0.980 and
  boundary recall of at least 0.95, and takes the highest segment F1@50, with exact ties going to the earlier setting.
  The choices were committed before the selection run and then applied once. The CLS and Ordered systems keep their
  stored settings.
- **The sensitivity rows** are the direct argmax and the token probabilities decoded with the same seed and arm's A02
  CLS numeric B3 setting. They never select a system.
- **Same-target accuracy** is the window argmax at the request against the reference class of the requested sample. It
  compares each token readout with its own arm's CLS on the same windows. Mixed windows are windows of constructed
  streams with at least two reference views among their 16 samples. Their bootstrap resamples the source sessions of
  the requested samples (primary) or whole streams (sensitivity).
- The **paired bootstrap** of the systems resamples native clips by session and streams by stream, with 2,000 draws,
  seed 20261002, and the same draws for both systems.
- A **unit** is a reference segment of a multi stream (pair streams are not counted), matched at IoU 0.5 by a
  predicted segment of its class.
  **Segments of at most 16 samples** last at most 1.6 seconds. An **X-Y-X middle** is the middle segment of three
  consecutive reference segments whose outer two share a class.

## 1. Anchor positions

Offsets are 0 or −1 only. Every non-exact window has an odd marker slot, and it reads the anchor one slot before it.
Non-exact windows occur in three places.
- The second window of each sequence, whose request is sample 7.
- The last windows of a sequence, where the window is clamped.
- The native clips shorter than 16 samples.
Every recipe fragment is 10 to 40 samples long, so no view change falls between a request and its anchor. The anchor's
reference label therefore equals the request's in every window. Anchor slot 0 is never chosen, because the first
request is sample 2.

| Half | Windows | Exact (offset 0) | Non-exact (offset −1) | Non-exact native | Anchor label differs | Windows per anchor slot |
|---|---|---|---|---|---|---|
| selection | 7666 | 6506 | 1160 | 596 | 0 | 0: 0 / 2: 535 / 4: 6 / 6: 513 / 8: 5762 / 10: 219 / 12: 198 / 14: 433 |
| calibration | 7021 | 6004 | 1017 | 506 | 0 | 0: 0 / 2: 469 / 4: 3 / 6: 453 / 8: 5361 / 10: 187 / 12: 175 / 14: 373 |

On exact and on non-exact windows, no consistent accuracy difference between the token readout and CLS was detected.
The B3 sample accuracy below covers the samples that each kind of window serves through the nearest-centre mapping. It
is a description only.

| Half | Seed | Window accuracy, exact (token / CLS) | Window accuracy, non-exact (token / CLS) | B3 sample accuracy, exact (token / CLS) | B3 sample accuracy, non-exact (token / CLS) |
|---|---|---|---|---|---|
| selection | 0 | 0.9636 / 0.9607 | 0.9871 / 0.9828 | 0.9554 / 0.9543 | 0.9841 / 0.9785 |
| selection | 1 | 0.9706 / 0.9665 | 0.9836 / 0.9845 | 0.9595 / 0.9615 | 0.9865 / 0.9875 |
| selection | 2 | 0.9646 / 0.9619 | 0.9784 / 0.9810 | 0.9603 / 0.9557 | 0.9847 / 0.9817 |
| calibration | 0 | 0.9770 / 0.9763 | 0.9872 / 0.9882 | 0.9709 / 0.9702 | 0.9897 / 0.9863 |
| calibration | 1 | 0.9760 / 0.9775 | 0.9813 / 0.9813 | 0.9635 / 0.9687 | 0.9806 / 0.9826 |
| calibration | 2 | 0.9768 / 0.9742 | 0.9833 / 0.9843 | 0.9701 / 0.9669 | 0.9863 / 0.9822 |

`position_mapping.png` draws the rule on two synthetic windows (sample indices only, no frame).

## 2. B3 settings

The token settings were chosen on calibration only. The CLS settings are A02's stored choices, which used the same 40
settings.

| Token readout | Seed | Token B3 status | Half-window | Quantile | Feasible of 40 | CLS half-window | CLS quantile | CLS feasible of 40 |
|---|---|---|---|---|---|---|---|---|
| Aligned token | 0 | feasible | 7 | 0.999 | 37 | 7 | 0.999 | 40 |
| Aligned token | 1 | feasible | 7 | 0.97 | 10 | 7 | 0.999 | 4 |
| Aligned token | 2 | feasible | 11 | 0.999 | 40 | 5 | 0.999 | 40 |
| CLS-only token | 0 | feasible | 7 | 0.999 | 30 | 5 | 0.999 | 40 |
| CLS-only token | 1 | none feasible (9 / 0.95 diagnostic) | 9 | 0.95 | 0 | 7 | 0.999 | 40 |
| CLS-only token | 2 | none feasible (9 / 0.95 diagnostic) | 9 | 0.95 | 0 | 5 | 0.995 | 30 |
| Shuffled token | 0 | feasible | 11 | 0.8 | 32 | 7 | 0.999 | 40 |
| Shuffled token | 1 | feasible | 7 | 0.999 | 5 | 7 | 0.99 | 40 |
| Shuffled token | 2 | feasible | 7 | 0.999 | 32 | 5 | 0.999 | 40 |

The systems differ in their chosen settings as well as in their readouts, so a system difference is not due to the
readout alone.

## 3. Systems

Each cell gives seeds 0, 1 and 2. Every CLS and Ordered row equals its stored row exactly.

| System + B3 | Half | Segment F1@50 | Boundary P | Boundary R | Native macro-F1 | Switches per clip | A4C read as A5C | A5C read as A4C |
|---|---|---|---|---|---|---|---|---|
| Aligned token | selection | 0.9362 / 0.9250 / 0.9533 | 0.9370 / 0.8794 / 0.9584 | 0.9738 / 0.9921 / 0.9660 | 0.9786 / 0.9869 / 0.9700 | 0.046 / 0.102 / 0.039 | 2 / 0 / 2 | 0 / 1 / 0 |
| Aligned token | calibration | 0.9818 / 0.9658 / 0.9798 | 0.9728 / 0.9471 / 0.9752 | 0.9899 / 0.9899 / 0.9874 | 0.9856 / 0.9861 / 0.9856 | 0.013 / 0.093 / 0.013 | 1 / 2 / 1 | 1 / 0 / 1 |
| Aligned CLS | selection | 0.9423 / 0.9507 / 0.9615 | 0.9531 / 0.9413 / 0.9540 | 0.9581 / 0.9660 / 0.9764 | 0.9786 / 0.9877 / 0.9786 | 0.042 / 0.035 / 0.060 | 2 / 1 / 2 | 0 / 0 / 0 |
| Aligned CLS | calibration | 0.9856 / 0.9799 / 0.9797 | 0.9778 / 0.9728 / 0.9825 | 0.9950 / 0.9899 / 0.9849 | 0.9929 / 0.9861 / 0.9929 | 0.017 / 0.013 / 0.017 | 1 / 2 / 1 | 0 / 0 / 0 |
| Frozen Ordered | selection | 0.9385 / 0.9603 / 0.9644 | 0.9482 / 0.9585 / 0.9588 | 0.9581 / 0.9686 / 0.9738 | 0.9786 / 0.9786 / 0.9701 | 0.042 / 0.039 / 0.046 | 2 / 2 / 3 | 0 / 0 / 0 |
| Frozen Ordered | calibration | 0.9817 / 0.9836 / 0.9797 | 0.9751 / 0.9800 / 0.9774 | 0.9824 / 0.9874 / 0.9799 | 0.9929 / 0.9861 / 0.9861 | 0.025 / 0.017 / 0.013 | 1 / 2 / 2 | 0 / 0 / 0 |
| Shuffled token | selection | 0.9148 / 0.9577 / 0.9558 | 0.8771 / 0.9758 / 0.9679 | 0.9712 / 0.9503 / 0.9476 | 0.9877 / 0.9877 / 0.9786 | 0.077 / 0.021 / 0.018 | 1 / 1 / 2 | 0 / 0 / 0 |
| Shuffled token | calibration | 0.9800 / 0.9760 / 0.9758 | 0.9586 / 0.9702 / 0.9723 | 0.9899 / 0.9824 / 0.9698 | 0.9929 / 0.9861 / 0.9856 | 0.046 / 0.013 / 0.025 | 1 / 2 / 1 | 0 / 0 / 1 |
| Shuffled CLS | selection | 0.9386 / 0.9400 / 0.9603 | 0.9433 / 0.9078 / 0.9585 | 0.9581 / 0.9791 / 0.9686 | 0.9786 / 0.9786 / 0.9786 | 0.042 / 0.067 / 0.049 | 2 / 2 / 2 | 0 / 0 / 0 |
| Shuffled CLS | calibration | 0.9866 / 0.9771 / 0.9807 | 0.9754 / 0.9657 / 0.9849 | 0.9950 / 0.9899 / 0.9849 | 0.9929 / 0.9861 / 0.9929 | 0.017 / 0.046 / 0.017 | 1 / 2 / 1 | 0 / 0 / 0 |
| CLS-only token | selection | 0.9181 / 0.9217 / 0.8731 | 0.9482 / 0.9075 / 0.8341 | 0.9110 / 0.9503 / 0.9476 | 0.9786 / 0.8894 / 0.9771 | 0.011 / 0.095 / 0.088 | 2 / 7 / 1 | 0 / 0 / 1 |
| CLS-only token | calibration | 0.9537 / 0.9045 / 0.9638 | 0.9548 / 0.8721 / 0.9490 | 0.9548 / 0.9422 / 0.9824 | 0.9861 / 0.9288 / 0.9780 | 0.017 / 0.101 / 0.059 | 2 / 9 / 1 | 0 / 0 / 2 |
| CLS-only CLS | selection | 0.9312 / 0.9538 / 0.9273 | 0.9370 / 0.9418 / 0.8868 | 0.9346 / 0.9738 / 0.9843 | 0.9786 / 0.9786 / 0.9701 | 0.039 / 0.049 / 0.092 | 2 / 2 / 3 | 0 / 0 / 0 |
| CLS-only CLS | calibration | 0.9827 / 0.9770 / 0.9761 | 0.9800 / 0.9752 / 0.9656 | 0.9849 / 0.9899 / 0.9874 | 0.9929 / 0.9861 / 0.9861 | 0.025 / 0.025 / 0.059 | 1 / 2 / 2 | 0 / 0 / 0 |

`report_tables.md` holds every row, including the argmax rows, the rows at the CLS setting, the per-class recall and F1
and A02's fixed Ordered-setting rows of the CLS systems.

## 4. Comparisons

Segment F1@50 differences with 95% intervals. The diagnostic flag marks comparisons in which a system uses the 9 / 0.95
setting because it had no feasible setting.

| Comparison | Decoding | Half | Seed 0 | Seed 1 | Seed 2 | Over seeds |
|---|---|---|---|---|---|---|
| Aligned token − Aligned CLS | B3 | selection | -0.0061 [-0.0241, +0.0131] | -0.0257 [-0.0501, -0.0016] | -0.0082 [-0.0230, +0.0067] | no consistent difference detected |
| Aligned token − Aligned CLS | B3 | calibration | -0.0038 [-0.0095, +0.0009] | -0.0140 [-0.0276, -0.0034] | +0.0001 [-0.0066, +0.0069] | no consistent difference detected |
| Aligned token − Frozen Ordered | B3 | selection | -0.0023 [-0.0193, +0.0155] | -0.0353 [-0.0647, -0.0090] | -0.0110 [-0.0258, +0.0020] | no consistent difference detected |
| Aligned token − Frozen Ordered | B3 | calibration | +0.0001 [-0.0080, +0.0077] | -0.0178 [-0.0324, -0.0054] | +0.0001 [-0.0077, +0.0077] | no consistent difference detected |
| Aligned token − Shuffled token | B3 | selection | +0.0214 [+0.0061, +0.0383] | -0.0328 [-0.0640, -0.0038] | -0.0025 [-0.0149, +0.0101] | no consistent difference detected |
| Aligned token − Shuffled token | B3 | calibration | +0.0018 [-0.0066, +0.0103] | -0.0101 [-0.0240, +0.0011] | +0.0040 [-0.0029, +0.0119] | no consistent difference detected |
| Aligned token − CLS-only token | B3 | selection | +0.0181 [-0.0074, +0.0436] | +0.0033 [-0.0312, +0.0336] | +0.0802 [+0.0531, +0.1118] | no consistent difference detected (diagnostic setting) |
| Aligned token − CLS-only token | B3 | calibration | +0.0281 [+0.0152, +0.0425] | +0.0614 [+0.0366, +0.0865] | +0.0160 [+0.0037, +0.0299] | consistent positive (diagnostic setting) |
| Shuffled token − Shuffled CLS | B3 | selection | -0.0238 [-0.0460, -0.0022] | +0.0177 [-0.0020, +0.0390] | -0.0045 [-0.0205, +0.0096] | no consistent difference detected |
| Shuffled token − Shuffled CLS | B3 | calibration | -0.0066 [-0.0145, -0.0010] | -0.0011 [-0.0060, +0.0038] | -0.0048 [-0.0152, +0.0047] | no consistent difference detected |
| CLS-only token − CLS-only CLS | B3 | selection | -0.0131 [-0.0391, +0.0126] | -0.0321 [-0.0568, -0.0059] | -0.0542 [-0.0799, -0.0320] | no consistent difference detected (diagnostic setting) |
| CLS-only token − CLS-only CLS | B3 | calibration | -0.0290 [-0.0436, -0.0161] | -0.0725 [-0.0978, -0.0474] | -0.0123 [-0.0255, +0.0004] | no consistent difference detected (diagnostic setting) |
| Aligned token − Aligned CLS | argmax | selection | -0.0141 [-0.0296, -0.0010] | -0.0102 [-0.0302, +0.0092] | -0.0147 [-0.0295, +0.0009] | no consistent difference detected |
| Aligned token − Aligned CLS | argmax | calibration | -0.0176 [-0.0339, -0.0049] | -0.0184 [-0.0373, -0.0019] | -0.0138 [-0.0295, -0.0018] | consistent negative |
| Aligned token − Aligned CLS | B3 at the CLS setting | selection | -0.0061 [-0.0241, +0.0131] | +0.0176 [+0.0040, +0.0338] | -0.0155 [-0.0277, -0.0064] | no consistent difference detected |
| Aligned token − Aligned CLS | B3 at the CLS setting | calibration | -0.0048 [-0.0109, +0.0000] | -0.0105 [-0.0225, -0.0010] | -0.0039 [-0.0128, +0.0046] | no consistent difference detected |

The other paired metrics of every comparison are in `report_tables.md`.

## 5. Same-target classification

Each token readout against its own arm's CLS on the same windows, with the session bootstrap. Only the all-mixed rows
enter the readings. The other subsets are described.

| Comparison | Windows | Half | Windows (n) | Sessions | Seed 0 | Seed 1 | Seed 2 | Over seeds |
|---|---|---|---|---|---|---|---|---|
| Aligned token − own CLS | all mixed | selection | 1449 | 36 | +0.0290 [+0.0165, +0.0398] | +0.0248 [+0.0142, +0.0340] | +0.0200 [+0.0074, +0.0319] | consistent positive |
| Aligned token − own CLS | all mixed | calibration | 1440 | 35 | +0.0160 [+0.0053, +0.0278] | +0.0153 [+0.0063, +0.0233] | +0.0257 [+0.0139, +0.0376] | consistent positive |
| Aligned token − own CLS | 8 of 16 | selection | 94 | 30 | +0.1064 [+0.0380, +0.1635] | +0.0957 [+0.0286, +0.1524] | +0.0851 [+0.0127, +0.1548] | consistent positive |
| Aligned token − own CLS | 8 of 16 | calibration | 73 | 29 | +0.1370 [+0.0469, +0.2381] | +0.0959 [+0.0130, +0.1806] | +0.1233 [+0.0123, +0.2440] | consistent positive |
| Aligned token − own CLS | 9-11 of 16 | selection | 627 | 35 | +0.0510 [+0.0235, +0.0782] | +0.0351 [+0.0140, +0.0535] | +0.0319 [+0.0087, +0.0540] | consistent positive |
| Aligned token − own CLS | 9-11 of 16 | calibration | 646 | 35 | +0.0232 [+0.0075, +0.0404] | +0.0263 [+0.0107, +0.0409] | +0.0433 [+0.0237, +0.0643] | consistent positive |
| Aligned token − own CLS | A4C/A5C pair windows | selection | 111 | 20 | +0.0631 [+0.0000, +0.1346] | +0.0631 [+0.0139, +0.1319] | +0.0541 [+0.0211, +0.0952] | no consistent difference detected |
| Aligned token − own CLS | A4C/A5C pair windows | calibration | 127 | 24 | +0.0000 [-0.0400, +0.0392] | +0.0079 [-0.0182, +0.0388] | +0.0079 [-0.0254, +0.0446] | no consistent difference detected |
| Aligned token − own CLS | native | selection | 3444 | 36 | -0.0009 [-0.0036, +0.0013] | +0.0009 [-0.0014, +0.0036] | -0.0015 [-0.0083, +0.0030] | no consistent difference detected |
| Aligned token − own CLS | native | calibration | 2950 | 36 | -0.0020 [-0.0047, +0.0003] | -0.0054 [-0.0136, +0.0003] | -0.0044 [-0.0105, -0.0006] | no consistent difference detected |
| Shuffled token − own CLS | all mixed | selection | 1449 | 36 | -0.0041 [-0.0136, +0.0040] | -0.0097 [-0.0193, -0.0007] | -0.0104 [-0.0219, +0.0009] | no consistent difference detected |
| Shuffled token − own CLS | all mixed | calibration | 1440 | 35 | -0.0062 [-0.0136, +0.0014] | -0.0021 [-0.0103, +0.0059] | -0.0049 [-0.0148, +0.0062] | no consistent difference detected |
| CLS-only token − own CLS | all mixed | selection | 1449 | 36 | -0.0669 [-0.0862, -0.0490] | -0.0676 [-0.1010, -0.0352] | -0.0380 [-0.0698, -0.0101] | consistent negative |
| CLS-only token − own CLS | all mixed | calibration | 1440 | 35 | -0.0521 [-0.0770, -0.0279] | -0.0903 [-0.1209, -0.0631] | -0.0125 [-0.0283, +0.0022] | no consistent difference detected |

Target accuracy per seed (seeds 0 / 1 / 2):

| Readout | Half | All mixed | 8 of 16 | 9-11 of 16 | A4C/A5C pair windows | Native windows |
|---|---|---|---|---|---|---|
| Aligned token | selection | 0.939 / 0.951 / 0.939 | 0.840 / 0.851 / 0.851 | 0.928 / 0.941 / 0.923 | 0.856 / 0.919 / 0.919 | 0.982 / 0.981 / 0.976 |
| Aligned token | calibration | 0.956 / 0.959 / 0.961 | 0.808 / 0.863 / 0.836 | 0.941 / 0.946 / 0.949 | 0.921 / 0.929 / 0.929 | 0.983 / 0.980 / 0.980 |
| Aligned CLS | selection | 0.910 / 0.926 / 0.919 | 0.734 / 0.755 / 0.766 | 0.877 / 0.906 / 0.892 | 0.793 / 0.856 / 0.865 | 0.983 / 0.981 / 0.978 |
| Aligned CLS | calibration | 0.940 / 0.944 / 0.935 | 0.671 / 0.767 / 0.712 | 0.918 / 0.920 / 0.906 | 0.921 / 0.921 / 0.921 | 0.985 / 0.985 / 0.984 |
| Frozen Ordered | selection | 0.903 / 0.909 / 0.912 | 0.628 / 0.702 / 0.681 | 0.871 / 0.866 / 0.877 | 0.829 / 0.847 / 0.883 | 0.981 / 0.981 / 0.978 |
| Frozen Ordered | calibration | 0.915 / 0.914 / 0.910 | 0.575 / 0.630 / 0.644 | 0.875 / 0.867 / 0.856 | 0.913 / 0.890 / 0.890 | 0.986 / 0.986 / 0.983 |

Changed all-mixed windows of Aligned token − Aligned CLS. A star marks a count that rests on fewer than 10 sessions.

| Half | Seed | Fixed by the token readout (recordings / sessions) | New errors (recordings / sessions) |
|---|---|---|---|
| selection | 0 | 50 (32 / 16) | 8 (7 / 6) * |
| selection | 1 | 44 (29 / 15) | 8 (8 / 7) * |
| selection | 2 | 42 (37 / 19) | 13 (11 / 9) * |
| calibration | 0 | 34 (30 / 17) | 11 (9 / 7) * |
| calibration | 1 | 33 (31 / 18) | 11 (11 / 10) |
| calibration | 2 | 47 (40 / 20) | 10 (9 / 8) * |

## 6. Native clips and short units

Native clips that Aligned token + B3 gets right and Aligned CLS + B3 gets wrong (fixed), and the reverse (new). Every
count is one clip from one session.

| Half | Seed | A4C read as A5C: fixed / new | A5C read as A4C: fixed / new | All clips: fixed / new |
|---|---|---|---|---|
| selection | 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| selection | 1 | 1 / 0 | 0 / 1 | 1 / 1 |
| selection | 2 | 0 / 0 | 0 / 0 | 0 / 1 |
| calibration | 0 | 0 / 0 | 0 / 1 | 0 / 1 |
| calibration | 1 | 0 / 0 | 0 / 0 | 0 / 0 |
| calibration | 2 | 0 / 0 | 0 / 1 | 0 / 1 |

Matched reference units with B3, seeds 0 / 1 / 2:

| System + B3 | Half | Segments of at most 16 samples | Of which A4C or A5C | X-Y-X middles | A4C/A5C X-Y-X middles |
|---|---|---|---|---|---|
| Aligned token | selection | 137 / 141 / 135 of 141 | 61 / 65 / 59 of 65 | 67 / 68 / 64 of 69 | 6 / 7 / 3 of 7 |
| Aligned token | calibration | 131 / 130 / 130 of 134 | 64 / 63 / 63 of 67 | 80 / 79 / 80 of 80 | 7 / 7 / 7 of 7 |
| Aligned CLS | selection | 131 / 132 / 138 of 141 | 55 / 56 / 62 of 65 | 64 / 64 / 67 of 69 | 3 / 3 / 6 of 7 |
| Aligned CLS | calibration | 132 / 129 / 128 of 134 | 65 / 63 / 64 of 67 | 80 / 79 / 80 of 80 | 7 / 7 / 7 of 7 |
| Frozen Ordered | selection | 130 / 134 / 136 of 141 | 54 / 58 / 60 of 65 | 66 / 64 / 66 of 69 | 5 / 3 / 5 of 7 |
| Frozen Ordered | calibration | 128 / 129 / 127 of 134 | 66 / 65 / 65 of 67 | 80 / 80 / 80 of 80 | 7 / 7 / 7 of 7 |

The 7 A4C/A5C X-Y-X middles of each half come from 5 (selection) and 7 (calibration) streams and from 7 and 6
sessions, fewer than 10, so their counts are not stable evidence. Every miss count in this table rests on fewer than 10
sessions (at most 8 per cell).

## 7. Goal and stored references

| System + B3 | Selection mean | Calibration mean | Constraint failures | Goal met |
|---|---|---|---|---|
| Aligned token | 0.9382 | 0.9758 | seed 2 selection, native 0.9700 | no |
| Aligned CLS | 0.9515 | 0.9817 | none | no |
| Frozen Ordered | 0.9544 | 0.9817 | seed 2 selection, native 0.9701 | no |
| Shuffled token | 0.9428 | 0.9773 | seed 2 selection, recall 0.9476 | no |
| Shuffled CLS | 0.9463 | 0.9814 | none | no |
| CLS-only token | 0.9043 | 0.9407 | seed 0 selection, recall 0.9110; seed 1 selection, no feasible B3, native 0.8894; seed 1 calibration, no feasible B3, recall 0.9422, native 0.9288; seed 2 selection, no feasible B3, recall 0.9476; seed 2 calibration, no feasible B3, native 0.9780 | no |
| CLS-only CLS | 0.9374 | 0.9786 | seed 0 selection, recall 0.9346; seed 2 selection, native 0.9701 | no |

The goal needs means of at least 0.944409687184662 on selection and 0.9821421709894332 on calibration. In every seed
and half, it also needs a feasible B3 setting, boundary recall of at least 0.95 and native macro-F1 of at least 0.973
(selection) or 0.980 (calibration). The calibration sensitivity target is 0.9838935784747763. The recomputed CLS and
Ordered goals equal A02's stored goals.

Stored rows of earlier systems, as A02 stored them. They were not recomputed or paired in this round.

| Stored system | Selection segment F1@50 (mean) | Calibration segment F1@50 (mean) |
|---|---|---|
| E0 + B3 | 0.9344 | 0.9340 |
| E1 + B3 | 0.8940 | 0.9589 |
| B7 + B3 | 0.8798 | 0.9721 |
| Historical E1 + ASRF | 0.9087 | 0.9682 |
| Paired ASRF | 0.8826 | 0.9702 |
| Original FACT | 0.9051 | 0.9644 |
| A01 Linear + B3 | 0.9476 | 0.9760 |
| A01 MS-TCN | 0.9387 | 0.9679 |
| A01 ASRF | 0.9449 | 0.9747 |
| Historical Continued CLS (V02) | 0.9442 | 0.9776 |

## What this round does not show

- **No new architecture or training.** It reads outputs that A02's frozen models already produced.
- **No system gain.** No consistent segment F1@50 difference was detected between the aligned token readout and the same
  weights' CLS, the frozen Ordered reference or the shuffled token control. No system meets the goal.
- **Not why no consistent segment gain came with the local gain.** The token readout names the requested view better
  on mixed windows, and its direct argmax outputs change label more often inside native clips (consistent on
  calibration only). With B3, no consistent difference in label switches per clip was detected. This round does not
  show how these relate to the segment results.
- **Not a verdict on time-token readouts in general.** Only this configuration ends: the nearest even anchor, these
  weights and B3. Other readout rules were not tested, as the design fixed one rule.
- **Not that CLS cannot carry the information, or that a token sees only its anchor frame.** Each token has interacted
  with the whole window.
- **No cardiac or probe explanation.** Score changes are not attributed to cardiac phase or probe motion.
- **Not clinical scanning.** Constructed streams and public recording labels are not real continuous scans or expert
  frame labels, and the development halves are not an independent validation.

## Deviations and launch record

1. **Two choice records.**
   - The first calibration (17:58 to 18:03) ran with code digest fa49ee2c. The dry run that followed showed that
     comparison keys holding `|` split into extra markdown cells.
   - aada40b fixed the table layout only, but it changed the code digest, which every stage checks. The first record was
     therefore moved aside on the server (`runs/trained_token_readout_v1/superseded_fa49ee2c/`) together with its dry
     run.
   - The preparation, check, calibration and dry run then ran again with code digest 4b6930f5.
   - The second record holds the same choices, grids and CLS references as the first; only the code digest differs. It
     was committed (5c9bf49) before the selection run, and the selection run checked its sha256.
2. **Repeated preparation and check.** They ran at 783af48 (16:30), after the review fixes (17:54) and after the table
   fix (18:13). Each later record replaced the earlier one, and all runs are in the ledger.
3. **The dry run is an added stage.** It runs the selection code on the calibration half, which stands in for both
   halves. It reads no selection file and writes only under `runs/trained_token_readout_v1/dry_run/`. It is a code
   check, not a result.
4. **Tests added after the report tests.** 3620640 committed the report tests, which failed at import, before the
   tables module. Later commits added tests for the selection guards and the table layout (95b56e4, 0e62879 and
   aada40b), all before any selection-half token metric existed.
5. **No GPU spot-check.** The task sheet allows one only to verify the cache. A02's fact-check reran the three aligned
   models on the first multi stream of each half (28 and 25 windows) and reproduced their saved CLS and time logits bit
   for bit in the round's own batches. Every identity and digest checked here matches.
6. **Review probes.** The implementation review's read-only probes used about 3 minutes of server CPU outside the stage
   ledger.

## Fact-check

An independent fact-check on 2026-10-14 recomputed this round's results, mostly with its own code. It found no error in
the results. Before publication it corrected one unsupported statement of this README and seven smaller wording points.

- **Integrity and order.**
  - All 41 pins of the configuration match. All 18 prediction files carry A02's run identities, and the 9 weight pins
    equal A02's recorded digests. The weights were hashed only.
  - The checker recomputed the code digest 4b6930f5 with its own hashing of the same 83 files. The files are
    byte-identical on the server and at the local commit. Every record of the final run carries this digest.
  - 5c9bf49 committed the choice record at 18:26:10, and the selection run started at 18:26:37. The committed file's
    sha256 equals the server copy, `done.json` and the summary's identity.
  - The superseded record equals the final one except for the code digest.
  - The ledger holds 11 stage runs, all finished, for 0.518 CPU hours.
- **Re-implementation.**
  - The checker rebuilt both halves' windows from the protocol, the clip index and the window plans. They equal all 18
    A02 files and all 6 V01 files.
  - Its own anchor rule equals the saved anchors and offsets, and the token logits are bit-identical.
  - Its own softmax and mapping match the saved probabilities within 2.2e-7 (float32 rounding).
  - The position counts equal the summary. The non-exact windows are the second window of each sequence (513 and 453),
    end-clamped windows (622 and 538) and windows in clips shorter than 16 samples (25 and 26).
- **Systems.**
  - Its own B3 decoder reproduces all 18 token outputs sample for sample.
  - Its own metric code reproduces every system row exactly, and every stored A02 and V01 row.
  - It re-derived all 9 calibration choices with its own thresholds, grid and rule.
- **Statistics and readings.**
  - All 72 paired system bootstraps and all 252 same-target bootstraps reproduce exactly with its own code, as do all
    verdicts.
  - The goals, the constraint failures and the fixed readings (labels learned, no usable gain, and end) reproduce.
- **README.** All 12 tables and every number checked match `summary.json`, `choices.json`, the ledger and the git
  history.
- **Ids and frames.** No recording, session or stream id appears in any file of this directory, in the working tree or
  at its commits. The figure is a synthetic schematic without a frame.
- **Corrections before publication.**
  - A statement that the token outputs change label more often inside native clips. This holds for the direct argmax
    only.
  - The wording of the A02 all-mixed comparison, of what the round does not show, of the arm comparison, of the unit
    definition and of the scope of A02's re-inference.
  - Notes on counts that rest on fewer than 10 sessions.
- **Cost.** The check used about 0.03 CPU hours and no GPU. It ran no round stage and wrote no repository or `runs/`
  file. Its test runs created pytest temporary folders on the server, and pytest's automatic cleanup removed older
  pytest temporary folders. No project or result file was affected.

## Commands

Run on the server in the project `.venv`. All stages run on the CPU.

```bash
C=configs/experiments/trained_token_readout_v1.yaml
python scripts/trained_token_readout.py --config $C --prepare     # pins, environment, metadata, windows, positions
python scripts/trained_token_readout.py --config $C --check       # A02 CLS and V01 Ordered rows reproduced exactly
python scripts/trained_token_readout.py --config $C --calibrate   # calibration only: one B3 choice per token system
python scripts/trained_token_readout.py --config $C --dry-run     # code check; the calibration half stands in for both
python scripts/trained_token_readout.py --config $C --selection --choices-sha256 ed754376a49c383452e52ca0b99799d33a3c3796ec7fce5710a35abbcd6d51cd
python -m pytest tests/test_trained_token_readout.py tests/test_trained_token_readout_report.py -q
```

## Provenance

- **Code.**
  - The selector, source checks and metadata adapter are at 783af48 and the selection stage at 95b56e4.
  - The review fixes are at 0e62879 and the table fix at aada40b.
  - The configuration is at ac86622 (sha256 c2e90d60…).
  - The final code digest 4b6930f5 covers the script, the listed helpers and the 83 repository modules it imports. It
    was the same locally and on the server, and every record of the final run carries it.
- **Choices.** `choices.json` was committed at 5c9bf49 with sha256 ed754376… before the selection run.
- **Sources.**
  - The configuration pins the task sheet and A02's configuration, summary and README.
  - It pins the 18 saved prediction files, the nine final weights, the protocol, the clip index and the window plans.
  - It also pins V01's summary and six saved Ordered outputs.
  - Every digest was checked before use. Each prediction file's identity equals A02's run record, and the final-weight
    pins equal A02's recorded digests. The weights were hashed, never loaded.
- **Server outputs.**
  - `runs/trained_token_readout_v1/selection/{selection,calibration}/` holds every token system's per-window logits,
    anchors and offsets, per-sample probabilities and B3 outputs.
  - `runs/trained_token_readout_v1/` also holds the stage records, the ledger, the logs, the dry run and the superseded
    first record.
  - Recording, session and stream ids stay on the server.

## Files

- `README.md` is this file, and `code_review.md` is the implementation review.
- `preparation.json` and `checks.json` are the final preparation and check records. `choices.json` holds the
  calibration choices with every setting's result.
- `summary.json` holds every system row, the paired comparisons and verdicts, the same-target classification, the
  positions, the units, the goals, the decision, the stored rows, the run identity and the cost.
- `report_tables.md` holds the generated tables, and `position_mapping.png` the frame-free schematic.
- None of these files contains a recording, session or stream id or a frame.
