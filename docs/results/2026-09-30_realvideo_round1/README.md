# Real-video round 1 (2026-09-30)

> **Correction (2026-10-01).** In this document the EV9V "patient groups" or "presumed patient groups" are
> recording sessions, inferred from a file-name prefix that is a timestamp shared by the clips of one session.
> Sessions never cross a data split, but patient identities are unavailable, so patient overlap across sessions or
> splits cannot be assessed. The numbers are unchanged; only the description of the grouping changes.

Follow-up checks and the frame-order fix are in `docs/results/2026-09-30_frame_order_fix/`. There are currently no
independent echo experts, so the blinded pilot in §4 is on hold.

## Question and scope

The question is whether a reader, and later a model, can tell a real view change from a stretch in which the current
frames are not enough to identify an unchanged view. This round checks whether the data can support that question. It
reports no model scores.

- Candidate counts serve to find mechanisms. They are not prevalence estimates.
- EV9V and MIMIC are reported separately.

All code is in `echo_routing/realvideo/` and in seven scripts:

- `scripts/audit_realvideo_inventory.py`
- `scripts/realvideo_probs.py`
- `scripts/realvideo_screen.py`
- `scripts/ev9v_cycles.py`
- `scripts/mimic_round1.py`
- `scripts/report_realvideo_round1.py`
- `scripts/export_annotation_pilot.py`

Tests are in `tests/test_realvideo.py`. Outputs stay on the server in `runs/_demo/realvideo/`. Files derived from
MIMIC must not leave the credentialed environment.

## Terms

| Term | Meaning |
|---|---|
| Recognizer | A model that gives view probabilities for each frame. B0 is our ResNet-18 trained on EV9V. EchoPrime is the released view classifier, never trained on EV9V; its views outside our five view families (PLAX, PSAX, A4C, A5C, SC4C) are pooled as "other". |
| Dominant view | The view a recognizer names most often in a clip. |
| Excursion | A stretch of 0.2–1.0 s in which the recognizer names another view and then returns to the dominant view. |
| Split contrast | How much the average view probabilities differ before and after the best split point of a clip. 0 means no difference and 1 means completely different. |
| Repeated candidate | A clip with at least two excursions. |
| Sustained candidate | A clip whose split contrast is at least 0.5, with different views on the two sides. |
| Mixed | A clip that is both a repeated and a sustained candidate. |
| Peak | For cycles from image motion: the average correlation between frames one cardiac period apart. |
| Dip | For cycles from image motion: how far that correlation falls in between. A cycle estimate with a high peak and a high dip is called reliable. |
| Phase locking | Whether the frames on which the recognizer disagrees with the dominant view fall at the same point of the heartbeat. |

## 0. A defect in the shared caches

EV9V names its frames `0.jpg, 1.jpg, …` without zero padding. The shared loader
`echo_routing.ingest.frames.list_frame_paths` sorts by name, so it reads a 109-frame clip in the order 0, 1, 10,
100–108, 11–19, 2, 20–29, and so on.

We checked this directly.

- Frame 50 of the MP4 matches `50.jpg`, with a mean absolute difference of 0.36. The 51st file in name order is
  `46.jpg`, with a difference of 9.40.
- The cached B0 output for frame index 90 is exactly B0 on `82.jpg`, the 91st file in name order.

The effect is large across all 5,138 EV9V clips.

- In a typical clip the cached 10 Hz samples sit 0.47 s from their intended time (median), and the worst sample sits
  3.2 s away.
- Intended and actual time correlate at only 0.31 in a typical clip.
- Only 13 clips have no sample displaced by more than 0.5 s.

The loader is used in these places:

- the B0 feature cache (`features/extract.py`) and the frame-encoder training data (`features/dataset.py`);
- the ViT code (`vit/datasets.py`, `extract_vit_features.py`, `predict_vit_streams.py`);
- the windowed EchoPrime and STFM runs and `eval_echoprime_views.py`;
- the demo videos on the public site (`make_demo_streams.py`).

The windowed EchoViewCLIP run sorts by name on its own and has the same problem. `a5c_analysis.py` shows the true frame
but labels it with the probability of a different frame. `a5c_text_cue.py` pairs text flags with probabilities from
different frames.

We also traced how the external models were trained.

- The public STFM code at the pinned commit lists frames with `sorted(os.listdir(...))`. With EV9V's unpadded names
  this sorts frames as text, so our three reproduced seeds, which ran that code, used frames in that order. The code
  alone does not show that the authors' published numbers used the same input; that would need their run records.
  The sort changes both the order of a clip's frames and, where frames are picked by position, which frames are
  picked.
- EchoViewCLIP trains and tests on the MP4 files, which decode in true order. Only our windowed inference is affected.

A fixed model gives the same output for the same image whatever the sort. Extraction, however, picks frames by position
after sorting, so different images were sampled. The existing frame-level and clip-level scores therefore cannot be
kept as they are. On the corrected cache, B0's test clip macro-F1 moves from 0.957 to 0.944. The frame encoder was also
trained through this path, so its training sample differed as well. Fragment boundaries in constructed streams are
correct. Anything that depends on the order of frames inside a clip is affected:

- the temporal models trained on cached features (MS-TCN, ASFormer and the ViT router);
- the sliding windows given to STFM and EchoViewCLIP, which saw shuffled "consecutive" frames;
- the fragmentation and flicker statistics;
- the demo videos.

This round itself read frames in numeric order and recomputed B0 and EchoPrime for the validation and test clips
(`runs/_demo/realvideo/probs_sorted/`). The shared loader was fixed on 2026-09-30, and caches now live in a new,
versioned directory. The fix and the recomputed baselines are described in `docs/results/2026-09-30_frame_order_fix/`.

## 1. Data and caches

| Source | Real videos | Per-frame outputs | Grouping | Phase signal |
|---|---|---|---|---|
| EV9V | 5,138 clips (train 3,683, val 567, test 888) at 30 fps. The median clip lasts 3.6–3.9 s depending on the split, the middle 80% last 1.5–12.2 s, and 745 clips are shorter than 2 s. | B0 exists for every split. ViT-S, EchoPrime, EchoViewCLIP and STFM exist for val and test only. All of these are shuffled (§0). Correctly ordered B0 and EchoPrime exist for 1,321 val/test clips. | Removing the clip index from the id gives 703 groups, the number of patients reported in the EV9V paper. This is a clue, not proof that each group is one patient, so we call them presumed patient groups. The split is disjoint even at the coarser export-batch level (495 batches). | The frames carry no ECG; the display shows `***bpm`. Only image motion is available. |
| MIMIC-IV-Echo | 17,173 of 524,137 DICOM files are downloaded (3.3%). These include 232 complete studies from 152 patients and 5,625 B-mode loops from 151 patients. Each file is one stored loop, and loops are never joined. | None before this round. | `subject_id` and `study_id`. | Every loop shows an ECG trace with a moving cursor. Each file also has a `HeartRate` tag. |
| EchoNet-Dynamic | 10,030 A4C videos, typically 50 fps and 171 frames long. | None. | None. | End-diastole and end-systole frames are labelled for 10,025 videos. |
| EchoXFlow sample | Two exam folders. | None. | None. | ECG arrays are stored. |

MIMIC view labels (ECHOVIEW) need a data use agreement that the user must sign; requests currently return 403. The
bulk downloader was stopped on 2026-09-25 at about 0.25 MB/s.

## 2. EV9V candidate lists

This analysis covers the validation and test clips, with frames in true order and two recognizers. It includes 1,314
clips from 197 presumed patient groups. The 7 discovery clips are left out, and 164 of the clips are shorter than 2 s.

| | B0 | EchoPrime | Either | Both | Patient groups (either) |
|---|---|---|---|---|---|
| Repeated candidates (mixed included) | 119 | 103 | 172 | 50 | 94 |
| Sustained candidates (mixed included) | 24 | 23 | 34 | 13 | 29 |
| Mixed | 9 | 10 | | | |

The counts depend on the thresholds. The table varies one setting at a time and gives counts for either recognizer
and, in brackets, for both.

| Setting | Values | Candidates |
|---|---|---|
| Shortest excursion | 0.1, 0.2, 0.3 s | repeated 302, 172, 94 (both 112, 50, 31) |
| Excursions required | 2, 3 | repeated 172, 87 |
| Split contrast required | 0.3, 0.5, 0.7 | sustained 74, 34, 9 (both 19, 13, 3) |

The two recognizers agree only moderately. On clips of at least 2 s, Cohen's κ for the candidate type is 0.38. The
Spearman correlation is 0.50 for time spent in excursions and 0.63 for split contrast.

Most sustained candidates are A4C↔A5C tilts: 20 of 24 for B0 and 18 of 23 for EchoPrime. The rest are a few PLAX↔PSAX
sweeps in long clips. These clips last 5.1 s at the median. By label, the repeated candidates are 81 A4C, 47 PLAX,
22 A5C, 17 PSAX and 5 SC4C.

The discovery clips work as a face-validity check, and they are not counted above.

- The known late A4C→A5C tilt is a sustained candidate for both recognizers, with contrast 0.79 and 0.76 and no
  excursions. With shuffled frames it had looked mixed.
- Three of the four intermittent A5C clips are repeated candidates for at least one recognizer.
- The suspected label error is not a candidate, because it reads as A5C throughout.

## 3. Cardiac cycles

### MIMIC, with the drawn ECG as the reference

The 5,625 B-mode loops come from 151 patients and last 2.6 s at the median. Every file decoded.

The ECG trace was parsed in 5,485 loops (97.5%). The failures were a cursor that did not move in a straight line
(93 loops), no R wave found (46) and no trace (1). In 5,213 loops at least two R waves were found. In 4,019 loops the
heart rate from the ECG is within 10% of the `HeartRate` tag. The tag itself is sometimes wrong; one loop has tag 73,
while the screen and the trace both show 63.

We checked 12 random loops by eye. The R markers sit on the R waves in 10. One loop with a negative QRS misses a beat,
and one has an extra marker at the end of the trace.

| Complete R–R intervals per loop | 0 | 1 | 2 | 3 | 4 or more |
|---|---|---|---|---|---|
| Loops | 412 | 376 | 1,215 | 3,217 | 405 |

The coefficient of variation of the R–R intervals exceeds 0.15 in 903 of 4,837 evaluable loops. This mixes true
arrhythmia with parse errors, and the two are not separated yet.

### Image motion against the ECG

The comparison uses the 4,019 loops whose ECG rate matches the tag. Without a threshold, the period from image motion
matches the R–R interval within 10% in 84.7% of loops. The main failure is a doubled period (8.6%). We kept a cycle
estimate only when the peak is at least 0.3 and the dip at least 0.6. This keeps 54% of loops, and 95.1% of those
agree with the ECG, with 3.5% doubled. These figures come from the same loops that chose the threshold; the follow-up
check repeats the choice on half of the patients and tests it on the other half.

### EV9V, from image motion only

Of 1,321 validation and test clips, 62 give no estimate and 179 are shorter than 2 s. Periodicity is weaker than in
MIMIC: the median peak is 0.17 (interquartile range 0.10–0.26), against 0.32 (0.19–0.45) in MIMIC.

With the threshold chosen on MIMIC, 198 clips (15%) have a reliable cycle and 179 of them span at least 2 cycles. The
median rate is 67 bpm, and 11 reliable clips have uneven cycle lengths (coefficient of variation above 0.15). The
reliable share is 18% for A4C, 20% for A5C, 16% for PLAX, 11% for PSAX and 2% for SC4C.

We checked 12 random clips with a peak of at least 0.3 by eye. In all 12 the frames marked as the same phase look
alike, and the frame half a cycle later shows the opposite phase.

### MIMIC candidates (not used in the pilot)

Of the 5,625 loops, 849 are shorter than 2 s.

| | Repeated (mixed included) | Sustained (mixed included) |
|---|---|---|
| B0 | 866 | 20 |
| EchoPrime | 413 | 46 |
| Either / both | 1,160 / 119 | 59 / 7 |

The recognizers barely agree here (κ 0.09 on loops of at least 2 s). EchoPrime's dominant view is distributed as
follows.

| PSAX | Other (A2C, A3C, SSN, Doppler) | PLAX | A4C | SC4C | A5C |
|---|---|---|---|---|---|
| 1,589 | 1,252 | 1,117 | 840 | 657 | 170 |

B0 was trained on EV9V's five view families and cannot name the "other" views. On MIMIC these lists therefore mostly
show recognizer failure, not the two phenomena. Screening MIMIC needs a recognizer that covers its views, and ECHOVIEW
labels would let us check one.

### First look at phase locking (descriptive)

The test asks whether the off-dominant frames cluster at one point of the heartbeat. Phase locking is measured by the
resultant length, which is 0 when phases are spread evenly and 1 when all fall at one phase. The null keeps the
number and length of the off runs and places them at random, so 5% of loops should reach p < 0.05 by chance.

MIMIC loops start and end on an R wave, so off runs touching either end would fake locking. Those runs are dropped.

| Data (phase source) | Recognizer | Repeated candidates with p < 0.05 | Other loops with off frames, p < 0.05 |
|---|---|---|---|
| MIMIC (ECG, loops with at least 2 beats) | B0 | 14.3% of 775 | 6.7% of 983 |
| MIMIC (ECG, loops with at least 2 beats) | EchoPrime | 18.6% of 354 | 9.3% of 497 |
| EV9V (image motion, 179 reliable clips) | B0 | 6 of 17 | 4 of 20 |
| EV9V (image motion, 179 reliable clips) | EchoPrime | 2 of 9 | 1 of 14 |

This first look should not be read as a phase effect of 2–3 times chance. The follow-up check gives each loop the ECG
phase of another patient's loop of the same length, and that reproduces most of the rate (11.1% for B0 and 14.4% for
EchoPrime in repeated candidates). R-gated loops of similar length and beat count can have nearly identical phase
trajectories, so the swap can keep both acquisition alignment and genuine physiological correspondence; this control
cannot tell them apart. The own-phase excess over the donor phases is small (about 3 points, see
`docs/results/2026-09-30_frame_order_fix/`). EV9V has too few evaluable clips to judge. None of this tests H2.

## 4. Blinded annotation pilot

**Execution status, 2026-09-30:** The user has confirmed that no independent echocardiography experts are currently
available. Keep this package as a prepared, unexecuted expert pilot; recruiting readers is not a prerequisite for the
current work. Codex/Claude review may provide exploratory AI judgments or pseudo-labels, but must not be reported as
expert annotation, inter-expert agreement, or a clinical reference standard. See the current execution constraints in
`docs/annotation/pilot_protocol.md`.

The protocol is in `docs/annotation/pilot_protocol.md` and is for the team only. The reader instructions are in
`docs/annotation/reader_instructions.md`.

The tool `demo/annotator/index.html` is a local player that steps frame by frame. Readers mark three kinds of
interval:

- A, a view change, where a transition period is allowed;
- B, the same view with locally insufficient evidence;
- C, cannot tell.

The tool saves automatically and exports JSON. We tested its logic end to end in jsdom.

The package is `runs/_demo/realvideo/pilot/annotation_pilot_v1.zip` on the server. It is 466 MB because it keeps the
original JPEG frames.

| Stratum | Clips |
|---|---|
| Random (one clip from each of 30 random patient groups) | 30 |
| Sustained candidates | 25 |
| Repeated candidates flagged by both recognizers | 15 |
| Repeated candidates flagged by one recognizer | 15 |
| Repeats for test–retest | 8 |
| Scored total | 93 |
| Practice | 2 |

The 95 clips come from 73 presumed patient groups, with at most 2 clips per group. The 93 scored presentations are 85
independent clips plus 8 repeats, and the repeats do not add independent sample size. The file `clips.js` holds only an anonymous id,
the frame count and the frame rate. The key is in a separate folder (`pilot/key/pilot_key.csv`), and the frames carry
no EXIF data or comments.

## 5. Answers for round 1

1. **Do the data support the question?** Partly.
   - EV9V gives two-recognizer sequences for 1,314 clips from 197 presumed patient groups. Clips are recorded once the view is
     obtained, so real changes inside a clip are rare: 34 sustained candidates against 172 repeated ones.
   - MIMIC loops are 2–3 beats long, start on an R wave and have no view labels. They suit phase validation and the
     study of evidence that comes and goes within one view, but not real changes.
   - We found no public continuous-scanning video.
2. **Do the candidates cover both phenomena?** Yes, but unevenly. Repeated short changes are common, and sustained
   changes are mostly A4C↔A5C tilts. Because the recognizers agree only moderately, the pilot includes clips flagged
   by one recognizer and by both.
3. **Can an independent phase be obtained?** In MIMIC, yes, from the drawn ECG. In EV9V only from image motion, which
   is reliable in 15% of clips. We validated image motion against the MIMIC ECG and checked it by eye on EV9V.
4. **Can experts tell the two apart?** Unverified. The package and analysis plan are ready, but no independent expert
   readers are available. AI-assisted review cannot answer this expert-agreement question.

## 6. Next steps

1. **Decide on the frame-order fix.** The fix is to make `list_frame_paths` sort numerically and to add the
   frame-order version to the cache namespace. The caches, the windowed baselines, the temporal models and the demo
   videos would then be regenerated. Earlier temporal results should not be used until then.
2. **Proceed without expert readers.** Preserve the pilot for possible future use. Do not wait for recruitment or use
   Codex and Claude as substitute expert readers. Any AI-assisted case review remains exploratory and separately
   identified.
3. **Run diagnostics supported by existing references.** Check the overlap between sustained/repeated candidates and
   clips with usable cycle estimates; independently evaluate period estimation and cross-cycle phase correspondence
   against available ECG references, separating threshold selection from evaluation by patient. These checks do not
   establish H1 or clinical benefit. The comparison of phase-aware methods with smoothing, confidence, duration and an
   HMM against expert interval labels remains conditional on obtaining an independent reference. If that reference
   remains unavailable, narrow or change the research question rather than treating AI agreement as validation.
4. **MIMIC.** ECHOVIEW access is optional for the current diagnostics. Reassess the need for its labels before asking
   the user to complete the agreement or downloading further studies.
