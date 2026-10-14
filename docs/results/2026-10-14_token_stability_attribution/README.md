# Token stability and segmentation attribution (2026-10-14)

This round asked where the trained token readout of A03 improved the requested-view accuracy on mixed windows without a
consistent segment gain. It answered three questions from the saved A02 and A03 outputs:

1. Do the corrected mixed-window decisions recover reference segments that would otherwise be missed, or do they only
   correct labels inside segments that were already matched?
2. Do the token outputs in pure windows add errors or fragments that offset the mixed-window changes at one fixed
   decoder?
3. Does the same frame, with the same local input, receive conflicting time-token predictions in overlapping windows?

Nothing was trained, and no model was loaded or run. The round read the frozen A02 models' saved logits, A03's selector,
probabilities and decoders, and the allowlist metadata of both development halves. The task sheet is
`docs/superpowers/plans/2026-10-14-token-stability-attribution.md`.

**What kind of evidence this is.**
- It is a retrospective diagnosis of development predictions whose overall results had already been inspected. The
  definitions were fixed before any new count, but this is not an independent validation or an untouched
  preregistration.
- The hold-out is by session, not by patient. Constructed streams are spliced from separate recordings and are not real
  scans; they reuse recordings, so their units are not independent cases. The three seeds share the same cases.
- Native clips carry the public recording label, not expert frame labels. No label was changed.
- The reference-conditioned cells (TC, CT) choose a readout per window from the reference labels. They are diagnostics,
  not deployable systems, attainable scores or upper bounds.
- A time token has interacted with the whole window, so it is not a single-frame embedding.
- The test split and EchoNet-Pediatric were not read.

**Order of work.** All times are UTC on 2026-10-14.
- 99b4571 committed the task sheet and the pending index entry A04 at 20:51. ae15344 committed the fixed configuration at
  20:59, before any code. 35b626a committed the tests at 21:03; they failed at import.
- 3855d7a committed the implementation at 21:14. The first preparation and check ran at 21:16.
- An independent implementation review passed the code with fixes (`code_review.md`). e4ee52e made the fixes at 21:51.
  The preparation and check ran again at 21:52, the reviewer then verified the fixes, and 27d44aa committed the records
  at 22:07.
- The single complete diagnosis ran from 22:07 to 22:14.
- An independent fact-check then recomputed the results (see "Fact-check").

## Answers

Values are given for seeds 0, 1 and 2 in that order, on the selection half and then the calibration half. A
difference is consistent only when all three seeds' 95% intervals exclude zero strictly on the same side in a half.

### 1. Mixed-window corrections rarely recover a missed segment

**Observed.**
- With the actual systems, the aligned token readout corrected 44, 40 and 38 mixed requests of multi streams on
  selection and 31, 28 and 39 on calibration, from 13 to 19 sessions per seed and half. 215 of these 220 requests lie
  within 5 samples of a reference view change.
- 185 of the 220 corrected requests lie in reference segments that both systems match, and 4 more in segments that only
  the CLS system matches. So 79.5% to 100% of the corrected requests per seed and half sit in segments the CLS system
  already matched. 19 lie in segments that only the token system matches, and 12 in segments that neither matches.
  Per seed and half, these two counts rest on fewer than 10 sessions (0 to 4).
- At the fixed decoder, the token readout newly matched 8, 10 and 0 multi-stream reference segments on selection and
  lost 5, 4 and 5. On calibration it newly matched 0, 1 and 3 and lost 3, 5 and 5. Each of these per-seed counts rests
  on fewer than 10 sessions (0 to 9).
- Over the three seeds and all populations at the fixed decoder, it newly matched 22 distinct reference segments (12
  sessions) and lost 14 (8 sessions) on selection. On calibration it newly matched 9 (8 sessions) and lost 16 (14
  sessions). The lost count on selection (8 sessions) and the newly matched count on calibration (8 sessions) rest on
  fewer than 10 sessions.

**Supported interpretation.** The mixed-window gain is mostly a local correction of labels next to view changes, inside
segments the CLS system already matched at IoU 0.5. Segment F1@50 counts such a segment once, so these corrections can
leave it unchanged.

**Not determined.** Whether these corrections improve boundary placement in a way that matters elsewhere. The distances
are distances of requests, not recovered boundaries.

### 2. No mixed-versus-pure trade-off was observed at the fixed decoder

**Observed.**
- **B3 at A02's CLS setting.**
  - With the token readout only in mixed windows (TC − CC), segment F1@50 changes by +0.004, +0.003 and −0.008 on
    selection and −0.004, −0.005 and −0.003 on calibration. With it only in pure windows (CT − CC), the changes are
    −0.006, +0.006 and −0.008 and then −0.004, −0.006 and −0.004.
  - No consistent difference was detected for either contrast, for TT − CC or for the interaction. On calibration all
    six point estimates of the two contrasts are negative, so their directions are not opposite.
- **Direct argmax.**
  - CT − CC is consistent negative on calibration (−0.014, −0.018 and −0.015). It comes with consistently more native
    label switches and lower boundary F1.
  - TC − CC shows no consistent difference (−0.002, −0.001 and −0.007, then +0.002, −0.001 and +0.001).
  - TT − CC is also consistent negative on calibration (−0.018, −0.018 and −0.014), with consistently more native
    switches and lower boundary F1.
  - Per native clip, the token readout in pure windows switches 0.135 to 0.190 times, against 0.076 to 0.134 for CLS.
- **Pure windows.** The token readout newly misreads 10 to 21 native windows per seed and half and fixes 2 to 13, each
  on fewer than 10 sessions (6 to 8 and 1 to 7). In multi streams, more than 10 samples from any view change, it newly
  misreads 5 to 16 pure windows (2 to 5 sessions) and fixes 0 to 9.
- **Probability-only changes.** At the fixed decoder, 22% to 55% of the decoded multi-stream samples that change do so
  at a sample whose window kept its argmax. Such windows move their probabilities by a mean total-variation distance of
  0.022 to 0.028.

**Supported interpretation.**
- Without smoothing, the token outputs in pure windows add label fragments (more native switches in all six cells;
  consistent on calibration only).
- B3 removes most of them, and no consistent pure-window effect remains.
- The mixed-window corrections, isolated in TC, do not show a consistent segment gain either. The absence of a system
  gain therefore does not rest on a pure-window offset alone.

**Not determined.** Whether another decoder would show a trade-off. The reference mask that defines TC and CT does not
exist at deployment, so these cells establish no routing policy.

### 3. The same frame and local input can get conflicting predictions in overlapping windows

**Observed.**
- **The main subset.**
  - Both anchors have the complete three-slot neighbourhood, and both windows hold the same sample triple.
  - In 3,155 to 9,535 pairs per population and half, the marker triples are also equal, so the complete local stem
    input is identical.
  - There, the aligned arm's anchors disagree in 0.8% to 1.4% of native pairs, 4.8% to 5.5% of pair-stream pairs and
    8.8% to 12.1% of multi-stream pairs. The rates are group-weighted, and each seed and half rests on 33 to 36
    sessions.
  - The shuffled and CLS-only arms disagree in 25% to 28% of multi-stream pairs. The aligned arm disagrees about half
    as often as the controls in pair streams and 0.34 to 0.47 times as often in multi streams. In native clips the
    aligned and shuffled rates are similar.
- **Where they disagree.**
  - In multi streams, 83% to 93% of the aligned arm's disagreeing equal-marker pairs lie within 5 samples of a
    reference view change, against 32% of all equal-marker main-subset multi pairs (30% of the whole main subset).
  - 85% to 90% of them pair anchor 6 or 7 of one window with anchor 1 or 2 of a later window. That window is 10 samples
    later in all but 5 to 10 pairs per seed and half, which are 15 samples apart at sequence ends.
- **Changed marker triples.** These pairs disagree less: 0.1% to 0.4% (native), 1.0% to 2.4% (pair streams) and 1.2% to
  1.9% (multi streams).
- **A03's output anchors.**
  - No pair holds A03's selected output anchor in both windows. This is a property of the design. The selected anchor
    holds the request or the sample before it, and requests are at least 3 samples apart.
  - 986 to 2,313 pairs per population and half hold it in one window. All of them have changed marker triples.

**Supported interpretation.**
- The time token of a frame can change class with the rest of the window and with its token position, mostly next to
  view changes.
- Window context, token position and the target marker change together in these pairs. This is joint sensitivity. It
  is not isolated cross-boundary contamination, and it is not attributed to cardiac phase.
- Aligned supervision lowers the disagreement of stream windows relative to the controls, but does not remove it.

**Not determined.** Whether A03's actual output anchors are equally unstable.
- In the equal-marker pairs, the disagreeing anchors are side anchors. 79% to 88% of their positions lie 4 or 6 slots
  from the marker.
- Only changed-marker pairs hold an output anchor, and in one window only. 44 to 64 disagreeing main-subset pairs per
  seed and half are of this kind.
- The measured instability is not evidence about the cause of A03's output switches.

### 4. The fixed readings

Four readings apply: local correction without segment recovery, anchor sensitivity observed, unresolved, and end.
- **Local correction without segment recovery.** In every seed and half, more than half of the corrected mixed requests
  of multi streams lie in segments the CLS system already matched (79.5% to 100%).
- **Anchor sensitivity observed.** The aligned arm's main subset has 814 to 1,212 disagreeing pairs per seed and half.
- **Unresolved.** Three flags apply:
  - no pair holds the selected output anchor in both windows (by design);
  - TC − CC has point estimates of both signs;
  - the actual systems use different decoders.

  The few-session flag did not fire. The local reading rests on 13 to 19 sessions per seed and half, and the anchor
  reading on 33 to 36.
- **Not applying.** The mixed-versus-pure trade-off.
- **End.** No training, readout fusion, shifted request, interpolation, threshold search, new loss or label change
  follows from this round.

### 5. For the next modeling decision

These are observations and hypotheses, not tested mechanisms.
- **Observed.**
  - A local accuracy gain at view changes mostly did not translate into newly matched segments. 19 of 220 corrected
    requests lie in segments that only the token system matches. Most lie in segments already matched.
  - The token outputs are noisier within pure stretches. B3 hides most of this noise.
  - Time-token predictions of the same frame depend on the window. The dependence concentrates near view changes. It is
    larger in the controls in pair and multi streams and similar in native clips.
  - None of the overlap pairs covers A03's output anchor in both windows.
- **Hypotheses, not shown here.**
  - A model-level change aimed at recovering missed segments (rather than relabelling matched ones) might matter more for
    segment F1@50.
  - Cross-window consistency might matter as well.
  - This diagnosis cannot show that a consistency loss would improve segmentation. A training task would need its own
    model-level hypothesis, matched controls and budget.

### 6. Cost

The round used about 0.54 CPU hours of its 2-hour cap:
- the run, 0.11 hours;
- the two preparations and checks, 0.008 hours;
- the test runs, 0.12 hours, estimated in two hand-recorded lines;
- the review and its verification, 0.26 hours, as reported;
- the fact-check, 0.045 hours, as reported.

No GPU was used.

## Terms

- **Actual systems.** The aligned token readout at its committed A03 B3 setting against the aligned CLS output at A02's
  stored setting. Their decoders differ, so a difference is a complete-system difference, not one of the readout alone.
- **Fixed decoder.** Every comparison decoded with the seed's A02 aligned CLS stored numeric B3 setting: half-window 7,
  7 and 5, thresholds 0.2294, 0.2087 and 0.1691, NMS radius 5. Nothing was refitted or chosen.
- **Mixed window.** A window of a constructed stream whose 16 input slots hold at least two reference classes. Native
  windows are pure.
- **The four cells.** The first letter names the readout of mixed windows and the second that of pure windows (C for
  CLS, T for token). The source is chosen per window before the nearest-request mapping to 10 Hz. CC equals the CLS
  output and TT the token output.
- **Corrected and newly wrong requests.** A request is corrected when the token argmax names the requested sample's
  reference class and the CLS argmax does not. It is newly wrong in the reverse case.
- **Request distance.** The number of samples between the request and the nearest reference view change; 0 means the
  request is next to the change.
- **Segment status.** A reference segment can be matched by both systems, by the token system only (newly matched), by
  the CLS system only (lost under token) or by neither. The matching is the evaluator's class-aware one-to-one matching
  at IoU 0.5, in its exact greedy order.
- **Overlapping anchors.** A frame of a sequence is held by anchors of several windows. A window keeps one anchor per
  frame: the one nearest its marker slot, then the lower index.
  - The stem of time position j reads input slots 2j − 1, 2j and 2j + 1 in both the RGB patch embedding and the marker
    stem. No later block downsamples time.
  - In the main subset, both anchors have all three slots, and both windows hold the same sample triple. With equal
    marker triples too, the complete local stem input is identical.
  - Each sequence-sample occurrence group carries weight 1, split equally over its pairs. Physical frames are counted
    separately.
- **The paired bootstrap** resamples native clips by session and streams by stream, with 2,000 draws, seed 20261002,
  and the same draws within each contrast.

## 1. Corrected mixed requests of multi streams (actual systems)

| Half | Seed | Corrected mixed requests | Within 5 samples of a change | In segments both systems match | In segments only the token system matches | In segments only CLS matches | In segments neither matches | Sessions | Newly wrong mixed requests | Segments holding both kinds |
|---|---|---|---|---|---|---|---|---|---|---|
| selection | 0 | 44 | 43 | 35 | 4 | 0 | 5 | 15 | 7 | 0 |
| selection | 1 | 40 | 38 | 30 | 8 | 2 | 0 | 13 | 8 | 2 |
| selection | 2 | 38 | 38 | 34 | 0 | 2 | 2 | 17 | 11 | 0 |
| calibration | 0 | 31 | 31 | 31 | 0 | 0 | 0 | 17 | 9 | 0 |
| calibration | 1 | 28 | 28 | 24 | 2 | 0 | 2 | 16 | 10 | 3 |
| calibration | 2 | 39 | 37 | 31 | 5 | 0 | 3 | 19 | 9 | 1 |

The per-status columns, the newly wrong requests (5 to 9 sessions) and the segments holding both kinds rest on fewer
than 10 sessions in every seed and half.

## 2. Segment accounting of multi streams

TP, FP and FN come from the exposed matching, which equals the evaluator's counts.

| Decoding | Half | Seed | Token TP / FP / FN | CLS TP / FP / FN | Token F1 | CLS F1 | Newly matched | Lost | New unmatched predicted | Removed unmatched predicted |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed decoder | selection | 0 | 477 / 40 / 25 | 474 / 30 / 28 | 0.9362 | 0.9423 | 8 | 5 | 26 | 16 |
| fixed decoder | selection | 1 | 488 / 18 / 14 | 482 / 30 / 20 | 0.9683 | 0.9507 | 10 | 4 | 9 | 21 |
| fixed decoder | selection | 2 | 482 / 35 / 20 | 487 / 24 / 15 | 0.9460 | 0.9615 | 0 | 5 | 19 | 8 |
| fixed decoder | calibration | 0 | 511 / 13 / 7 | 514 / 11 / 4 | 0.9808 | 0.9856 | 0 | 3 | 7 | 5 |
| fixed decoder | calibration | 1 | 507 / 21 / 11 | 511 / 14 / 7 | 0.9694 | 0.9799 | 1 | 5 | 13 | 6 |
| fixed decoder | calibration | 2 | 506 / 13 / 12 | 508 / 11 / 10 | 0.9759 | 0.9797 | 3 | 5 | 8 | 6 |
| actual decoders | selection | 0 | 477 / 40 / 25 | 474 / 30 / 28 | 0.9362 | 0.9423 | 8 | 5 | 26 | 16 |
| actual decoders | selection | 1 | 487 / 64 / 15 | 482 / 30 / 20 | 0.9250 | 0.9507 | 13 | 8 | 51 | 17 |
| actual decoders | selection | 2 | 480 / 25 / 22 | 487 / 24 / 15 | 0.9533 | 0.9615 | 2 | 9 | 16 | 15 |
| actual decoders | calibration | 0 | 512 / 13 / 6 | 514 / 11 / 4 | 0.9818 | 0.9856 | 0 | 2 | 7 | 5 |
| actual decoders | calibration | 1 | 509 / 27 / 9 | 511 / 14 / 7 | 0.9658 | 0.9799 | 2 | 4 | 20 | 7 |
| actual decoders | calibration | 2 | 510 / 13 / 8 | 508 / 11 / 10 | 0.9798 | 0.9797 | 3 | 1 | 8 | 6 |

The newly matched and lost counts rest on 0 to 9 sessions per seed and half, and the unmatched predicted counts on 7 to
15.

On seed 0, the token setting nearly equals the CLS setting (half-window 7 in both, threshold 0.2290 against 0.2294).
The two decodings give the same counts on selection and differ by one matched segment on calibration (512 / 13 / 6
against 511 / 13 / 7). On seed 1, the token setting's threshold is 0.0221 against 0.2087, which adds predicted
segments.

Native clips and pair streams at the fixed decoder:

| Half | Seed | Native F1 (token / CLS) | Pair-stream F1 (token / CLS) | Native newly matched / lost | Pair newly matched / lost | Native new / removed unmatched predicted |
|---|---|---|---|---|---|---|
| selection | 0 | 0.9639 / 0.9655 | 0.9761 / 0.9615 | 1 / 1 | 4 / 0 | 7 / 6 |
| selection | 1 | 0.9672 / 0.9689 | 0.9621 / 0.9594 | 0 / 0 | 2 / 0 | 5 / 4 |
| selection | 2 | 0.9454 / 0.9538 | 0.9551 / 0.9547 | 2 / 4 | 2 / 0 | 16 / 13 |
| calibration | 0 | 0.9853 / 0.9874 | 0.9827 / 0.9796 | 0 / 1 | 4 / 2 | 2 / 2 |
| calibration | 1 | 0.9749 / 0.9853 | 0.9683 / 0.9739 | 0 / 2 | 2 / 2 | 4 / 1 |
| calibration | 2 | 0.9730 / 0.9874 | 0.9739 / 0.9736 | 0 / 2 | 4 / 2 | 8 / 3 |

Native F1 here is the segment F1 of the one reference segment per native clip, not the clip-level macro-F1. Every
changed-segment count in this table rests on fewer than 10 sessions.

## 3. Pure windows and probability-only changes

| Half | Seed | Native windows fixed (sessions) | Native windows newly wrong (sessions) | Multi pure windows more than 10 samples from a change, fixed (sessions) | Same, newly wrong (sessions) |
|---|---|---|---|---|---|
| selection | 0 | 8 (3) | 11 (8) | 2 (1) | 12 (4) |
| selection | 1 | 13 (7) | 10 (6) | 9 (6) | 16 (4) |
| selection | 2 | 9 (5) | 14 (6) | 6 (2) | 13 (5) |
| calibration | 0 | 5 (1) | 11 (8) | 0 (0) | 7 (4) |
| calibration | 1 | 5 (4) | 21 (7) | 1 (1) | 7 (2) |
| calibration | 2 | 2 (1) | 15 (8) | 1 (1) | 5 (4) |

Every count in this table rests on fewer than 10 sessions.

Multi streams at the fixed decoder:

| Half | Seed | Windows, same argmax | Windows, changed argmax | Mean total variation, same argmax | Same, changed argmax | Decoded samples changed | Of which at a window with the same argmax | Streams |
|---|---|---|---|---|---|---|---|---|
| selection | 0 | 2962 | 70 | 0.028 | 0.355 | 526 | 287 | 47 |
| selection | 1 | 2950 | 82 | 0.028 | 0.330 | 426 | 194 | 47 |
| selection | 2 | 2960 | 72 | 0.025 | 0.354 | 361 | 126 | 45 |
| calibration | 0 | 2994 | 54 | 0.022 | 0.391 | 268 | 79 | 37 |
| calibration | 1 | 2996 | 52 | 0.023 | 0.388 | 245 | 55 | 36 |
| calibration | 2 | 2991 | 57 | 0.023 | 0.429 | 295 | 72 | 38 |

## 4. The four reference-conditioned cells (diagnostics only)

Multi-stream segment F1@50 of each cell, and the native switches per clip of CC and CT:

| Decoding | Half | Seed | CC | TC | CT | TT | Switches per native clip, CC | Switches per native clip, CT |
|---|---|---|---|---|---|---|---|---|
| B3 (fixed) | selection | 0 | 0.9423 | 0.9459 | 0.9365 | 0.9362 | 0.042 | 0.046 |
| B3 (fixed) | selection | 1 | 0.9507 | 0.9539 | 0.9571 | 0.9683 | 0.035 | 0.039 |
| B3 (fixed) | selection | 2 | 0.9615 | 0.9537 | 0.9538 | 0.9460 | 0.060 | 0.063 |
| B3 (fixed) | calibration | 0 | 0.9856 | 0.9817 | 0.9818 | 0.9808 | 0.017 | 0.013 |
| B3 (fixed) | calibration | 1 | 0.9799 | 0.9751 | 0.9741 | 0.9694 | 0.013 | 0.017 |
| B3 (fixed) | calibration | 2 | 0.9797 | 0.9768 | 0.9759 | 0.9759 | 0.017 | 0.030 |
| argmax | selection | 0 | 0.8973 | 0.8951 | 0.8852 | 0.8831 | 0.095 | 0.148 |
| argmax | selection | 1 | 0.9144 | 0.9135 | 0.9047 | 0.9042 | 0.099 | 0.158 |
| argmax | selection | 2 | 0.9108 | 0.9042 | 0.9009 | 0.8961 | 0.134 | 0.162 |
| argmax | calibration | 0 | 0.9734 | 0.9753 | 0.9595 | 0.9559 | 0.076 | 0.135 |
| argmax | calibration | 1 | 0.9696 | 0.9687 | 0.9521 | 0.9512 | 0.076 | 0.160 |
| argmax | calibration | 2 | 0.9705 | 0.9715 | 0.9556 | 0.9567 | 0.084 | 0.190 |

Contrasts of multi-stream segment F1@50 with 95% intervals:

| Decoding | Contrast | Half | Seed 0 | Seed 1 | Seed 2 | Over seeds |
|---|---|---|---|---|---|---|
| B3 (fixed) | TC − CC | selection | +0.0036 [-0.0085, +0.0182] | +0.0032 [-0.0077, +0.0157] | -0.0078 [-0.0140, -0.0028] | no consistent difference detected |
| B3 (fixed) | TC − CC | calibration | -0.0039 [-0.0097, +0.0009] | -0.0048 [-0.0119, +0.0010] | -0.0029 [-0.0105, +0.0050] | no consistent difference detected |
| B3 (fixed) | CT − CC | selection | -0.0058 [-0.0189, +0.0068] | +0.0064 [-0.0043, +0.0187] | -0.0077 [-0.0180, -0.0009] | no consistent difference detected |
| B3 (fixed) | CT − CC | calibration | -0.0038 [-0.0091, -0.0000] | -0.0058 [-0.0161, +0.0009] | -0.0038 [-0.0088, +0.0000] | no consistent difference detected |
| B3 (fixed) | TT − CC | selection | -0.0061 [-0.0241, +0.0131] | +0.0176 [+0.0040, +0.0338] | -0.0155 [-0.0277, -0.0064] | no consistent difference detected |
| B3 (fixed) | TT − CC | calibration | -0.0048 [-0.0109, +0.0000] | -0.0105 [-0.0225, -0.0010] | -0.0039 [-0.0128, +0.0046] | no consistent difference detected |
| B3 (fixed) | TT − TC − CT + CC | selection | -0.0039 [-0.0140, +0.0047] | +0.0079 [+0.0010, +0.0170] | +0.0000 [-0.0000, +0.0002] | no consistent difference detected |
| B3 (fixed) | TT − TC − CT + CC | calibration | +0.0029 [+0.0000, +0.0078] | +0.0001 [-0.0027, +0.0028] | +0.0029 [-0.0000, +0.0066] | no consistent difference detected |
| argmax | TC − CC | selection | -0.0021 [-0.0097, +0.0066] | -0.0009 [-0.0060, +0.0043] | -0.0066 [-0.0134, -0.0007] | no consistent difference detected |
| argmax | TC − CC | calibration | +0.0019 [-0.0066, +0.0117] | -0.0009 [-0.0045, +0.0019] | +0.0010 [-0.0037, +0.0060] | no consistent difference detected |
| argmax | CT − CC | selection | -0.0121 [-0.0252, -0.0016] | -0.0097 [-0.0290, +0.0083] | -0.0099 [-0.0224, +0.0038] | no consistent difference detected |
| argmax | CT − CC | calibration | -0.0139 [-0.0290, -0.0026] | -0.0175 [-0.0359, -0.0017] | -0.0149 [-0.0295, -0.0037] | consistent negative |
| argmax | TT − CC | selection | -0.0141 [-0.0296, -0.0010] | -0.0102 [-0.0302, +0.0092] | -0.0147 [-0.0295, +0.0009] | no consistent difference detected |
| argmax | TT − CC | calibration | -0.0176 [-0.0339, -0.0049] | -0.0184 [-0.0373, -0.0019] | -0.0138 [-0.0295, -0.0018] | consistent negative |
| argmax | TT − TC − CT + CC | selection | +0.0001 [-0.0001, +0.0003] | +0.0004 [-0.0069, +0.0076] | +0.0018 [-0.0000, +0.0054] | no consistent difference detected |
| argmax | TT − TC − CT + CC | calibration | -0.0055 [-0.0149, +0.0001] | +0.0000 [-0.0001, +0.0002] | +0.0000 [-0.0001, +0.0002] | no consistent difference detected |

The interaction is an exact contrast of the four cells' metrics, not an additive decomposition into independent causes.
`report_tables.md` holds every metric of every cell and contrast.

## 5. Overlapping anchors

Support of the subsets:

| Half | Subset | Population | Occurrence groups | Pairs | Physical frames | Recordings | Sessions | Pairs with a selected output anchor | Pairs with it in both windows |
|---|---|---|---|---|---|---|---|---|---|
| selection | main, equal marker | native | 6677 | 9535 | 6677 | 284 | 36 | 0 | 0 |
| selection | main, equal marker | pairs | 2259 | 3680 | 2259 | 274 | 36 | 0 | 0 |
| selection | main, equal marker | multi | 5932 | 7095 | 4441 | 231 | 36 | 0 | 0 |
| selection | main, changed marker | native | 1889 | 3129 | 1889 | 284 | 36 | 2313 | 0 |
| selection | main, changed marker | pairs | 984 | 1618 | 984 | 274 | 36 | 1190 | 0 |
| selection | main, changed marker | multi | 860 | 1373 | 791 | 141 | 34 | 1005 | 0 |
| selection | broad | native | 9571 | 16722 | 9571 | 284 | 36 | 2417 | 0 |
| selection | broad | pairs | 3111 | 6873 | 3111 | 274 | 36 | 1244 | 0 |
| selection | broad | multi | 8664 | 11834 | 5937 | 231 | 36 | 1059 | 0 |
| calibration | main, equal marker | native | 5689 | 8168 | 5689 | 237 | 36 | 0 | 0 |
| calibration | main, equal marker | pairs | 1937 | 3155 | 1937 | 230 | 36 | 0 | 0 |
| calibration | main, equal marker | multi | 5983 | 7126 | 4450 | 213 | 35 | 0 | 0 |
| calibration | main, changed marker | native | 1617 | 2699 | 1617 | 237 | 36 | 1956 | 0 |
| calibration | main, changed marker | pairs | 817 | 1337 | 817 | 230 | 36 | 986 | 0 |
| calibration | main, changed marker | multi | 854 | 1370 | 814 | 136 | 32 | 1020 | 0 |
| calibration | broad | native | 8171 | 14358 | 8171 | 237 | 36 | 2042 | 0 |
| calibration | broad | pairs | 2685 | 5820 | 2685 | 230 | 36 | 1046 | 0 |
| calibration | broad | multi | 8725 | 11872 | 5956 | 213 | 35 | 1080 | 0 |

Group-weighted argmax disagreement of the time tokens, by arm (seeds 0 / 1 / 2):

| Half | Subset | Population | Aligned | Shuffled | CLS-only |
|---|---|---|---|---|---|
| selection | main, equal marker | native | 0.010 / 0.010 / 0.014 | 0.014 / 0.014 / 0.015 | 0.013 / 0.076 / 0.014 |
| selection | main, equal marker | pairs | 0.050 / 0.053 / 0.052 | 0.108 / 0.107 / 0.110 | 0.106 / 0.141 / 0.110 |
| selection | main, equal marker | multi | 0.112 / 0.121 / 0.105 | 0.262 / 0.256 / 0.264 | 0.274 / 0.267 / 0.275 |
| selection | main, changed marker | native | 0.001 / 0.004 / 0.004 | 0.002 / 0.003 / 0.003 | 0.004 / 0.036 / 0.004 |
| selection | main, changed marker | pairs | 0.018 / 0.024 / 0.018 | 0.045 / 0.043 / 0.042 | 0.043 / 0.071 / 0.044 |
| selection | main, changed marker | multi | 0.013 / 0.014 / 0.017 | 0.052 / 0.052 / 0.059 | 0.067 / 0.071 / 0.065 |
| calibration | main, equal marker | native | 0.008 / 0.010 / 0.009 | 0.007 / 0.008 / 0.009 | 0.017 / 0.032 / 0.010 |
| calibration | main, equal marker | pairs | 0.054 / 0.055 / 0.048 | 0.097 / 0.095 / 0.098 | 0.099 / 0.126 / 0.101 |
| calibration | main, equal marker | multi | 0.104 / 0.111 / 0.088 | 0.254 / 0.255 / 0.254 | 0.270 / 0.284 / 0.260 |
| calibration | main, changed marker | native | 0.002 / 0.003 / 0.004 | 0.002 / 0.002 / 0.002 | 0.005 / 0.018 / 0.002 |
| calibration | main, changed marker | pairs | 0.010 / 0.010 / 0.010 | 0.028 / 0.025 / 0.027 | 0.022 / 0.040 / 0.029 |
| calibration | main, changed marker | multi | 0.018 / 0.019 / 0.012 | 0.049 / 0.053 / 0.052 | 0.056 / 0.083 / 0.053 |

Where the aligned arm's disagreeing equal-marker pairs of multi streams lie:

| Half | Seed | Disagreeing pairs | Within 5 samples of a view change | Anchor 6 or 7 against anchor 1 or 2 of a later window |
|---|---|---|---|---|
| selection | 0 | 757 | 646 (85.3%) | 648 (85.6%) |
| selection | 1 | 807 | 691 (85.6%) | 688 (85.3%) |
| selection | 2 | 702 | 582 (82.9%) | 599 (85.3%) |
| calibration | 0 | 686 | 635 (92.6%) | 609 (88.8%) |
| calibration | 1 | 719 | 661 (91.9%) | 648 (90.1%) |
| calibration | 2 | 568 | 522 (91.9%) | 512 (90.1%) |

Of all equal-marker main-subset multi pairs, 32.3% (selection) and 32.0% (calibration) lie within 5 samples of a view
change (29.9% and 29.5% of the whole main subset). In the anchor-pair column, the later window is 10 samples later in
all but 5 to 10 pairs per seed and half, which are 15 samples apart at sequence ends. The equal-marker pairs combine
interior windows 10 samples apart with clamped windows that hold the same samples and whose markers lie outside the
three slots. The changed-marker pairs all involve a clamped or short window. In 72% to 75% of them, one window's
selected output anchor meets a side anchor of the other window. `report_tables.md` holds every rate,
the broad subset, the contexts and the role distributions.

## What this round does not show

- **No new model, inference or training.** It reads outputs that A02's frozen models already produced.
- **No attainable score.** TC and CT use reference labels per window. They are not systems, policies or upper bounds,
  and no goal or constraint is claimed for them.
- **Not why the token readout fails to gain segments.** The round shows where corrections and errors sit. It does not
  establish a mechanism that a model change would fix.
- **Not the stability of A03's output anchors.** No overlap pair holds them in both windows, so output-against-output
  stability was not measured. Most disagreements are between side anchors. 44 to 64 disagreeing main-subset pairs per
  seed and half compare an output anchor with another window's side anchor, all with changed marker triples.
- **Not a single cause of anchor disagreement.** Window context, token position and the target marker change together.
  The disagreement is not attributed to cross-boundary contamination, attention or cardiac phase.
- **Not clinical scanning.** Constructed streams and public recording labels are not real continuous scans or expert
  frame labels, and the development halves are not an independent validation.

## Deviations and launch record

1. **Tests changed after the failing commit.** 35b626a committed the tests before the modules, and they failed at
   import.
   - 3855d7a changed one of them. The bootstrap parity test compared NaN values with `==`, which cannot pass, so it now
     compares NaN-aware, and its synthetic same-view joins changed so that no stream lacks one.
   - 3855d7a also added an integration test of the run path on a synthetic half.
   - e4ee52e added tests for the review fixes. It also extended the stem-geometry test of 35b626a with two failure
     cases and adapted the run-path test of 3855d7a to the new return value.
2. **The configuration's identity changed.** The review found the stem wording "no later block pools time" inexact,
   because later blocks pool time with stride 1. e4ee52e changed it to "no later block downsamples time (later blocks
   pool time with stride 1 only)" and added "equal local inputs are a statement about the stem only".
   - This moved the configuration's sha256 from e42dec3c to f8b7898c.
   - No code reads the edited text, and no definition, threshold or budget changed.
   - The final preparation, check and run all carry f8b7898c.
3. **Repeated preparation and check.** They ran at 21:16 on the reviewed code and again at 21:52 after the fixes. Both
   runs are in the ledger.
4. **Hand-recorded ledger lines.** The task sheet asks for one server and local runtime ledger. Three lines were
   therefore recorded by hand:
   - the round's test runs (an estimate);
   - the review's probes, as reported by the reviewer;
   - the review verification's probes, as reported.
5. **An extra record.** The results directory also holds `checks.json`, the passed check record.
6. **A side effect of the review.** One of the reviewer's local pytest runs used the default temporary folder. pytest's
   automatic cleanup then touched older pytest folders in the system temporary directory. No project file was affected.
7. **Corrections after the run.** These changes do not touch the run or its identity.
   - **Ledger.** The fact-check found that the hand-recorded verification line gives an impossible finishing time
     (22:20; the verification ended before 22:07). It also measured a full run of the round's tests at about 59 s, so
     the earlier test-time estimate was low. Two appended lines correct this traceably: a zero-hour note on the
     timestamp and a supplementary 0.06-hour test estimate. The fact-check's own 0.045 hours are recorded as well.
   - **Test fixture names.** Two synthetic recipe names in `tests/test_token_stability_attribution.py` coincided with
     real positional stream names (a bank name and an index, with no recording or session information). They were
     renamed to plainly synthetic names after the run. Tests are not part of the code digest.

## Fact-check

An independent fact-check recomputed this round's results with its own code (numpy, pandas and yaml only), reading
the saved A02 logits, A03's records and the server's private run data. It found no error in the results. Before
publication it corrected three statements of this README and refined twelve smaller points. `fact_check.md` holds the
full record.

- **Integrity and order.**
  - All 44 pins match, and the 18 prediction files carry A02's training identities.
  - The checker recomputed the code digest e861fc8d and the configuration sha256 f8b7898c with its own hashing.
  - The commits run in the documented order, and the records commit precedes the run.
  - The tests failed at import at 35b626a, and all 21 pass at e4ee52e.
  - The run directory is complete, and the ledger totals agree.
- **Segment counts.** The checker rebuilt the windows, the selector, the probabilities, the B3 decoder and the matching.
  Over all six seed-and-half cells, 10,648 comparisons gave 0 mismatches:
  - the actual systems, the fixed decoder and the four cells, with B3 and with argmax;
  - TP, FP, FN and F1 per population;
  - segment status and changed segments;
  - probability-only changes, switches, and the counts over seeds.
- **Contrasts.** An independent bootstrap with the same draw sequence reproduced all 288 differences and intervals
  within 1e-12. It also reproduced all 96 verdicts.
- **Windows and readings.** The distance bins, the changed requests and their segment status, the local shares, the
  anchor counts, the flags and the readings all reproduce.
- **Overlapping anchors.**
  - A separately written pair construction gives the same 35,429 and 32,050 pairs with identical attributes.
  - Group weights sum to 1 per group.
  - The rates, roles and per-pair flags of all nine arm-and-seed combinations reproduce.
  - That no pair holds the selected output anchor in both windows follows from the geometry.
- **Ids and frames.** No recording or session id appears in any results file at any commit. Two synthetic test names
  formally matched positional stream names and were renamed (deviation 7).
- **Corrections before publication.**
  - The seed-0 note under table 2. The two decodings differ by one matched segment on calibration.
  - Flags on counts that rest on fewer than 10 sessions.
  - The wording on output anchors. 44 to 64 disagreeing pairs per seed and half do involve an output anchor in one
    window.
  - Smaller points: the anchor-pair windows, the comparator share, the comparison with the controls, the wording of
    sections 3 and 5, the deviations, and the order of events.

## Commands

Run on the server in the project `.venv`. All stages run on the CPU.

```bash
C=configs/experiments/token_stability_attribution_v1.yaml
python scripts/token_stability_attribution.py --config $C --prepare   # pins, environment, stem geometry, metadata, prediction files
python scripts/token_stability_attribution.py --config $C --check     # A03's rows and saved outputs, A02's CLS probabilities
python scripts/token_stability_attribution.py --config $C --run       # the single complete diagnosis
python -m pytest tests/test_token_stability_attribution.py tests/test_token_stability_attribution_tables.py -q
```

## Provenance

- **Code.**
  - The implementation is at 3855d7a and the review fixes at e4ee52e.
  - The final code digest e861fc8d covers the script, the listed helpers and the 85 repository modules it imports.
  - The digest is the same locally and on the server, and the preparation, check and run carry it.
- **Configuration.** ae15344, amended in wording at e4ee52e (sha256 f8b7898c).
- **Sources.**
  - The configuration pins the task sheet, A03's configuration, summary, README and choice record, and A03's six saved
    aligned token outputs.
  - Through A03's configuration, it pins the protocol, the clip index, the window plans, A02's configuration and summary
    and the 18 saved prediction files.
  - Every digest and every prediction file's training identity were checked before use. No weight was loaded.
- **Server outputs.**
  - `runs/token_stability_attribution_v1/run/` holds the run's summary and tables.
  - `runs/token_stability_attribution_v1/run/private/` holds the per-case data: changed windows, segments, anchor pairs
    and per-pair disagreement flags.
  - Recording, session, stream and recipe ids stay on the server.

## Files

- `README.md` is this file, `code_review.md` the implementation review and `fact_check.md` the fact-check record.
- `preparation.json` and `checks.json` are the final preparation and check records.
- `summary.json` holds every count, contrast, verdict, stability table, reading and the run identity.
  `report_tables.md` holds the generated tables.
- None of these files contains a recording, session, stream or recipe id or a frame.
