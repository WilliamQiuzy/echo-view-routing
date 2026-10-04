# Can 3D echocardiography supply a new training signal? Feasibility check (2026-10-03)

The short-span classification route ended after round 6, and no new model was trained here. Six rounds kept
re-reading the same clip labels on the same images. None of that added supervision that separates two kinds of change
in a 2D video, the heart moving and the imaging plane moving. This check asks whether 3D echocardiography can supply
that supervision.
- **Where would the new information come from?** 3D volumes can be re-sliced with known planes and times.
- **What has prior work not solved?** Echo-POSED is the closest prior method.
- **How would a model be validated?** Gains on simulated slices do not count. The check must return to real 2D videos.

No echo expert is available. Nothing here gives a view label to any re-sliced frame, and the re-sliced videos are not
real continuous scans.

**Second pass, later on 2026-10-03.** Four things changed after the first version.
- The CETUS request draft is in `docs/data_requests/2026-10-03_cetus_request.md`. The project lead reported sending
  the request on 2026-10-03; no reply or permission has been reported. Any agreement remains for the lead to sign.
- A limited control for slicing artifacts was run on the same three volumes (section 5).
- Three statements of the first version were too broad and are narrowed here. They concern the "5 times" comparison of
  plane and time changes, the claim that real 2D frames lack the contrast cue, and the missing Echo-POSED weights.
- The resource table now separates the scale each source describes from the data we actually hold. A timing error is
  also corrected. Each EchoSlicer file is a live recording of about 3 s and several heartbeats, not one cycle.

## Answer

1. **New information.** Re-slicing 3D volumes gives exactly known plane and time parameters. The supervision would be
   relative geometry, which is exact by construction and does not come from any classifier's predictions.
   - **Reproduced.** EchoSlicer's code ran on its sample and on three of its volumes.
   - **Controlled videos.** Three videos per volume keep the plane fixed, tilt the plane, or change both, and every
     frame's parameters are saved.
   - **Leaks.** The visible region reveals the tilt unless a common mask is applied. Inside the mask, image contrast
     still changes with the tilt. Section 5 shows that matching pixel-scale sharpness does not remove it, and that it
     also appears without linear-interpolation averaging. That does not fully rule out smoothing or other effects of
     the 3D acquisition. Re-slicing may therefore give a model cues that real 2D recordings do not share. That is a
     risk to control in any training. Whether real recordings share the cues has not been tested.
   - **Size of a plane change.** Image change here is the mean absolute difference in grey levels between two frames.
     A fan of planes is the set of planes reached by tilting one start plane. In the fan around the A4C plane, a 0.5°
     tilt changes the image about 5 times less than one volume step of the recording (0.11–0.16 s). In the fan around
     the A2C plane the same ratio is 5.5–11. The ratio belongs to these step sizes and planes only. Around the A4C
     plane, a tilt of about 3.5–4° changes the image as much as one volume step.
     The ratio is not a measure of how hard the task is, and it is not a signal-to-noise ratio.
2. **Data.** Few independent subjects are in hand.
   - **In hand.** EchoSlicer's 29 volumes come from 4 people. STACOM 2011's 15 volunteers, one gated cycle each, were
     downloaded after this check and verified in `docs/results/2026-10-03_stacom_verification/`. That makes 19 people.
   - **Described but not obtained.** CETUS describes 45 patients with one full cycle each. Its agreement, however,
     covers only the evaluation of endocardial segmentation, and its old download platform no longer answers. Whether
     the 45 dynamic sequences can still be obtained, and whether our use is allowed, is unconfirmed. The request asks
     both, and the project lead reports that it has been sent. MITEA has 134 subjects but only two frames per scan.
     Echo One (527 patients) is announced in the Echo-POSED paper, with no download found.
3. **Prior method.** Echo-POSED (May 2026) learns a frame-level probe pose from 3D slices, invariant to cardiac phase.
   It never tests real 2D videos, view classes or temporal change. Its code and weights are announced, but no link was
   found. That does not block this direction. A reimplementation from the paper can serve as the control if it is
   labelled as ours, comes with simple controls and claims none of the paper's results.
4. **Start conditions.** The conditions for a training round are not met now. The main reason is that the dynamic 3D
   data in hand come from 4 people. That describes the present. It is not a finding that the direction is infeasible.
   Section 7 lists what would change it, and section 8 states the one question a training round would have to answer.

## 1. What is available

| Resource | Described by the source | Obtained by us (2026-10-03) | Access and use conditions |
|---|---|---|---|
| EchoSlicer code ([github.com/echonet/3d-echo](https://github.com/echonet/3d-echo)) | Reads Philips 3D DICOM, finds standard planes by a classifier-guided search (EchoPrime view classifier, EchoNet-Dynamic LV segmentation), re-slices 2D videos | Cloned at commit `78f7085` and run | Public; **no LICENSE file** |
| EchoSlicer dataset (release v1.0, 1.1 GB) | 29 3D+t volumes from 4 people (A 15, B 5, C 5, D 4), the authors themselves; Philips EPIQ CVx | **All 29 volumes**, on the server | Public download; the README calls it open-source; no license file; the DICOM headers contain a patient-name field (not read) |
| Echo-POSED ([arXiv 2606.02634](https://arxiv.org/abs/2606.02634), 30 May 2026) | Self-supervised probe-pose model trained on 3D slices; model and data said to be released | Nothing | **No URL** in the paper, on GitHub or in web search |
| Echo One | 527 patients, 397 3D videos after filtering (348 development, 49 test), GE Vivid E95 | Nothing | Announced only |
| STACOM 2011 motion challenge | 15 healthy volunteers + 1 phantom, 4D apical volumes, DICOM | **All 15 volunteers**, downloaded and verified after this check (`docs/results/2026-10-03_stacom_verification/`); the phantom was not downloaded | Open access; the page asks users to cite Tobon-Gomez et al. 2013 |
| CETUS 2014 | 45 patients (15 healthy, 15 after myocardial infarction, 15 with dilated cardiomyopathy), one cardiac cycle each, three vendors | **Nothing.** The MIDAS link on the challenge site returns 404, and the collection on the Human Heart Project database needs a login. | The agreement allows use only to evaluate endocardial segmentation. Any other use needs the organisers' written permission. Request sent on 2026-10-03, as reported by the project lead; response and permission pending. |
| MITEA | 134 subjects, 2 scans each with **2 frames** (end-diastole, end-systole) | Nothing | Institutional request form, CC BY-NC-SA 4.0, no redistribution |

**Content of the EchoSlicer volumes.**
- **Typical volumes.** Most cover about 81° in azimuth and 64–109° in elevation, at 6–10 volumes per second.
- **Recordings, not single cycles.** Each file is a live recording of 2.0–3.2 s with 15–72 volumes. At the heart rate
  stored in the file (58–95 per minute), that spans about 3–4 heartbeats, so a beat has about 5–10 volumes in the
  typical files. The first version of this report read the volumes per file as volumes per cycle, which was wrong.
  The volume rate comes from the DICOM frame time (tag 0018,1063). The rounded display rate that EchoSlicer uses
  (0008,2144) differs from it by a few percent.
- **Narrow, fast volumes.** Five volumes trade field of view for speed. They are A_3, C_1, D_0, D_1 and D_3, at 9–30
  volumes per second, with elevation down to 19°. D_0 is a thin slab, so a 15° tilt leaves it.
- **Person D.** Only one of D's volumes is wide.

## 2. Reproduction

- **Demo.** EchoSlicer's demo ran in its own pinned environment on the bundled sample in 77 s on CPU. It produced its
  8 views with plane parameters and classifier scores.
- **Three volumes.** On A_0, B_0 and C_0 (persons A, B and C) it took 48–68 s per volume. The A4C plane its search
  selects scores 0.99, 0.91 and 0.99 on the EchoPrime classifier.
- **The A5C planes.** The planes it then selects for A5C sit 10–11° further in tilt and score only 0.31, 0.37 and
  0.72. On the sample the A5C plane scores 0.93, again at +10°.
- **What that shows.** A fixed tilt does not reliably produce an A5C-looking image. This is consistent with the
  user's point that a plane change is not a change of view class.

## 3. Controlled videos

`scripts/echoslicer_controlled.py` cuts three videos from each volume, A_0, B_0 and C_0.
- **Start plane.** All start from the A4C plane that EchoSlicer's search selects. That is a classifier's choice, so it
  is not called a label.
- **One grid.** Every frame uses one fixed in-plane grid (0.387 mm pixels), with raw intensities and no per-frame
  rescaling.

| Video | Plane | Time | Frames |
|---|---|---|---:|
| fixed_plane | start plane | every volume of the recording | 18–25 |
| fixed_time | start plane tilted 0° to 15° in 0.5° steps (EchoSlicer searches for A5C at +10° to +35°) | first volume | 31 |
| both | tilt 0° to 15° | one volume per frame, cycling | 31 |

**Outputs and storage.**
- Every frame's plane parameters (d, phi, theta, in-plane rotation and flips), time index, pixel size and visible
  fraction are in `numbers/<volume>/frames.csv`.
- Each video also exists with a common mask, keeping only pixels inside the volume in every frame of all three videos.
- The videos and contact sheets stay on the server (`data/echoslicer/controlled/`) and are not committed. The data
  carry no explicit license, and their headers hold personal fields.

## 4. Leak checks

| Check | A_0 | B_0 | C_0 | Reading |
|---|---|---|---|---|
| Visible fraction while tilting (Spearman with tilt) | −1.00 | −1.00 | −0.62 | Leak. The visible region shrinks with tilt. The common mask removes it. |
| Contrast (SD inside the common mask) against tilt, Spearman | −0.83 | −0.98 | −1.00 | Systematic in all three. It is not removed by the mask. Section 5 tests it further. |
| Contrast change over the tilt sweep, against over time at a fixed plane (relative range) | 0.08 vs 0.13 | 0.14 vs 0.06 | 0.42 vs 0.18 | In B_0 and C_0 the tilt changes contrast more than the heartbeat does. |
| Sharpness relative range: tilt sweep, against time at a fixed plane | 0.62 vs 0.06 | 0.57 vs 0.11 | 0.52 vs 0.10 | Superseded. Measured without a margin, so the edge of the volume entered the Laplacian. Section 5 repeats it with a margin. |
| Frame size across 0–15° tilt (EchoSlicer default grid) | constant | constant | constant | No size leak in this range. |
| Per-frame min–max rescaling (EchoSlicer default) | not used here | | | It would change contrast with the plane, so it was avoided. |

**What the contrast change is not.** The first version suggested a cause. Tilting moves the slice into the elevation
direction, which is sampled more coarsely (about 1.7° against 1.1° in azimuth), so interpolation would smooth it more.
Section 5 tests this. The relation also appears without linear-interpolation averaging, so that averaging alone does
not explain it. Smoothing from other sources, such as the beam width or the scanner's own processing, is not ruled
out.

**Consequence.** A model trained on such slices could learn to read the tilt from contrast instead of anatomy. Whether
real 2D recordings show the same relation is untested. A real probe tilt also changes what the plane cuts through, and
with it the contrast. Re-slicing may also add cues that real recordings do not share, for example if beams steered far
from the probe's centre give weaker signal. That is a risk any training must control. It is not an established fact.

**Change per step.** Inside the common mask, the median absolute change between consecutive frames is about 19–20
grey levels for one volume step at a fixed plane (0.11–0.16 s). It is about 3.5 for a 0.5° tilt at a fixed time. When
both change, it is about 19–20 again, dominated by the heart. The ratio of about 5 holds for these two step sizes only.
Section 5 gives the curves for other tilts and time gaps.

**Code review.** The first version did not compute the correlations against time for the fixed-plane video. They are
now in `checks.json`. The Spearman correlation of sharpness, mean and spread with the time index ranges from −0.49 to
0.74, with no consistent sign across volumes. Time spans three to four heartbeats, so these are less informative than
the ranges above. The rerun differs from the first run only by floating-point noise, at most 0.03 on sharpness values
of 350–600. Every value reported here is identical at two decimals.

## 5. Slice-artifact control

The question was whether the geometric supervision we can generate could be replaced by simple image cues such as
smoothness. The design was declared in `configs/experiments/slice_artifact_control_v1.yaml` before the first run.
Changes made after that run are listed there with their reasons and are summarised below. No network was trained, and
no slice got a view label. Controlling an image statistic here does not show that the transfer to real 2D recordings
is solved.

**Design.**
- **Sweeps.** Each sweep tilts a start plane from −15° to +15° in 0.5° steps, in both directions. The tilt changes
  θ_n, the elevation angle of the plane's normal. Sweeps run at two times, the first volume and the middle of the
  recording. Every frame of a sweep uses one fixed pixel grid (0.387 mm pixels) and raw intensities.
- **Start planes.** S1 is the A4C plane that EchoSlicer selects, used only as a geometric start. S2 rotates the azimuth
  of S1's normal by 20°. S3 sets θ_n to 90°, which makes the plane parallel to the volume's zero-elevation plane, the
  plane that contains the probe's central beam. S4, added after the first run, is EchoSlicer's A2C plane, again used
  only as a geometric start.
- **Fans of planes.** S1 to S3 turned out to share one fan. S3 keeps S1's offset and azimuth and changes only θ_n, so
  its sweep is another window on S1's fan, and S2 lies only 1.5–7° from S1. Together they cover θ_n from 75° to
  109–126°, on both sides of 90°. This A4C fan matters for the project, because EchoSlicer's own A5C plane lies on it,
  10–11° beyond the A4C plane. S4 lies 60–76° from S1 and opens a second fan, around the A2C plane.
- **Obliqueness.** Obliqueness is the angle between the slice and the zero-elevation plane. On S1–S3 it equals
  |θ_n − 90°|. It changes direction at every multiple of 90° of θ_n, and the turning point inside a sweep is called
  its fold. The fold is at θ_n = 90° for S1–S3, where obliqueness is smallest, and at 0° for S4, where it is largest.
- **Evaluation region.** All measures use only the pixels inside the volume in every frame of a start plane's sweeps,
  shrunk by 17 pixels (6.6 mm). The margin keeps every measure, including the Laplacian and the blur below, from
  reaching a pixel outside that region. The region covers 32–41% of the frame.
- **Measures.** Contrast is the standard deviation (SD) of grey levels in the region. Sharpness is the variance of the
  Laplacian, which responds to pixel-scale detail.
- **Matched smoothing.** Every frame of a sweep is blurred with the smallest Gaussian that brings its sharpness down to
  the sweep's lowest. Contrast is then measured again.
- **Nearest-neighbour check.** The same planes are also rendered so that each pixel takes one voxel's value, and no
  pixel averages several voxels. Contrast is tested the same way.

**Changes after the first run.**
- The first run measured sharpness and applied the blur without the margin. The edge of the volume, where the image
  drops to 0, therefore entered both. It inflated the sharpness of the most tilted frames, for example 566 against 330
  at −15° in A_0's first sweep. Contrast uses no neighbouring pixels and was not affected.
- The declared reading rule asks whether contrast falls on both sides of 90°. The first run did not compute that.
- The first run's time curve wrapped around the recording and used a different region from the tilt curves.
- The matching needed blurs below 0.5 pixel. The rendered pixels are 0.387 mm, while the volume's samples are about
  1–4 mm apart. Sharpness as defined here therefore mostly measures pixel-scale interpolation detail, and matching it
  leaves coarser smoothing untouched. The nearest-neighbour check was added to test interpolation directly.
- S1–S3 turned out to share one fan, so S4 was added. The per-side test then splits each sweep at its own fold, as a
  second code review pointed out. Splitting S4 at 90° would have pooled two of C_0's frames from beyond its fold.
- Seconds now come from the DICOM frame time. "Mid-cycle" in the config means the middle of the recording.

Rerunning with `--margin 0` reproduces the first run's tilt sweeps. Its numbers are kept in
`numbers/artifact_control_run1/`.

**Contrast against tilt.** Each cell gives the Spearman correlation at the first volume and at the middle of the
recording. Obliqueness is tested separately on each side of the sweep's fold.

| Volume | Start | θ_n swept (°) | Contrast vs signed tilt | Fold (°) | vs obliqueness, below the fold | vs obliqueness, above the fold | After matching, vs signed tilt | Nearest-neighbour, vs signed tilt | Largest blur (px) |
|---|---|---|---|---|---|---|---|---|---|
| A_0 | S1 | 86–116 | -0.21 / -0.62 | 90 | -1.00 / -1.00 | -0.71 / -0.82 | -0.21 / -0.62 | -0.20 / -0.60 | 0.38 |
| A_0 | S2 | 86–116 | -0.37 / -0.61 | 90 | -1.00 / -1.00 | -0.78 / -0.83 | -0.38 / -0.61 | -0.36 / -0.59 | 0.38 |
| A_0 | S3 | 75–105 | +0.88 / +0.82 | 90 | -1.00 / -0.99 | +0.09 / -0.28 | +0.88 / +0.82 | +0.87 / +0.82 | 0.39 |
| A_0 | S4 | 3–33 | +0.99 / +0.98 | 0 | n/a / n/a | -0.99 / -0.98 | +0.99 / +0.98 | +0.99 / +0.97 | 0.35 |
| B_0 | S1 | 79–109 | -0.89 / -0.89 | 90 | -0.39 / -0.35 | -0.97 / -0.99 | -0.89 / -0.89 | -0.85 / -0.88 | 0.43 |
| B_0 | S2 | 79–109 | -0.93 / -0.95 | 90 | -0.20 / +0.40 | -0.98 / -1.00 | -0.93 / -0.95 | -0.89 / -0.92 | 0.41 |
| B_0 | S3 | 75–105 | -0.48 / -0.66 | 90 | -0.83 / -0.68 | -0.97 / -0.98 | -0.49 / -0.66 | -0.39 / -0.61 | 0.44 |
| B_0 | S4 | 15–45 | +0.98 / +0.99 | 0 | n/a / n/a | -0.98 / -0.99 | +0.98 / +0.99 | +0.97 / +1.00 | 0.35 |
| C_0 | S1 | 96–126 | -1.00 / -1.00 | 90 | n/a / n/a | -1.00 / -1.00 | -1.00 / -1.00 | -1.00 / -1.00 | 0.34 |
| C_0 | S2 | 96–126 | -1.00 / -0.99 | 90 | n/a / n/a | -1.00 / -0.99 | -1.00 / -0.99 | -1.00 / -0.99 | 0.35 |
| C_0 | S3 | 75–105 | +0.79 / +0.90 | 90 | -1.00 / -1.00 | -0.66 / +0.20 | +0.79 / +0.90 | +0.79 / +0.91 | 0.36 |
| C_0 | S4 | -1–29 | -0.71 / -0.45 | 0 | n/a / n/a | +0.68 / +0.50 | -0.71 / -0.45 | -0.65 / -0.41 | 0.35 |

- **The relation holds in both directions and from every start plane.** Contrast changes with the tilt in every
  sweep. In 11 of the 24 sweeps, a frame's contrast alone ranks the frames by signed tilt almost perfectly
  (|ρ| ≥ 0.9).
- **It mostly falls with obliqueness, but not everywhere.** Take ρ ≤ −0.5 as a clear fall, a descriptive cut chosen
  after the run. In the A4C fan, contrast fell clearly with obliqueness in 25 of 32 one-sided tests. It fell clearly on
  both sides of 90° in 7 of the 14 sweeps where both sides could be tested. In the other 7 it fell clearly on one side
  and weakly, not at all, or the other way on the other side. In the A2C fan, contrast fell clearly with obliqueness in
  A_0 and B_0 (ρ −0.98 to −0.99). In C_0 it rose instead (ρ +0.68 and +0.50). The sign against the signed tilt
  therefore differs between volumes and start planes.
- **Matched smoothing changes nothing.** The largest blur any frame needed was 0.34–0.44 px. Contrast moved by at most
  0.3 grey levels, and its correlations are unchanged. Under the declared reading rule, a relation that persists after
  matching points to image content rather than smoothing. Because the matching acted only at the pixel scale, that
  reading rests on the next check rather than on the matching.
- **The relation does not need interpolation averaging.** With nearest-neighbour rendering, contrast is higher in
  every frame, by 1.7–2.4 grey levels on average per sweep, as expected without averaging. Its correlations with the
  tilt are nearly unchanged, so the relation appears without linear-interpolation averaging. This does not fully rule
  out smoothing or other acquisition effects. The voxels are already smoothed by the beam width and the scanner's
  processing, and that smoothing can vary with steering angle and depth. Anatomy, which a real probe tilt would also
  change, remains possible too. Three volumes cannot tell these apart, so the training plan carries the controls
  instead (section 8).
- **Sharpness is a weaker and less consistent cue.** With the margin, sharpness varied over a sweep by 8–41% of its
  mean in the A4C fan and 7–11% in the A2C fan. That is less than contrast in 16 of 24 sweeps. It tracks the tilt in
  some sweeps (|ρ| up to 0.90), but its direction differs between volumes. The first run's sharpness, and the
  sharpness ranges in section 4, were inflated by the volume's edge.

<details>
<summary>Sharpness against tilt, Spearman, first volume / middle of the recording</summary>

Relative sharpness is the sharpness divided by the squared SD. It does not grow with contrast, and here it mostly
mirrors contrast, because sharpness itself changes less.

| Volume | Start | Sharpness vs signed tilt, first run (no margin) | Sharpness vs signed tilt | Relative sharpness vs signed tilt | Relative sharpness vs obliqueness |
|---|---|---|---|---|---|
| A_0 | S1 | +0.44 / +0.41 | +0.15 / +0.44 | +0.20 / +0.63 | +0.30 / +0.67 |
| A_0 | S2 | +0.24 / +0.20 | +0.25 / +0.61 | +0.36 / +0.63 | +0.44 / +0.67 |
| A_0 | S3 | -0.57 / -0.46 | -0.77 / -0.36 | -0.90 / -0.79 | +0.21 / +0.40 |
| A_0 | S4 | not run | +0.42 / -0.47 | -0.98 / -0.97 | +0.98 / +0.97 |
| B_0 | S1 | +0.86 / +0.93 | +0.64 / +0.78 | +0.93 / +0.95 | +0.70 / +0.70 |
| B_0 | S2 | +0.52 / +0.80 | +0.64 / +0.81 | +0.96 / +0.98 | +0.68 / +0.67 |
| B_0 | S3 | +0.93 / +0.94 | +0.77 / +0.84 | +0.80 / +0.92 | +0.36 / +0.14 |
| B_0 | S4 | not run | +0.37 / +0.90 | -0.71 / -0.82 | +0.71 / +0.82 |
| C_0 | S1 | +0.64 / +0.67 | -0.06 / +0.32 | +0.99 / +0.99 | +0.99 / +0.99 |
| C_0 | S2 | +0.37 / +0.50 | -0.27 / +0.29 | +0.99 / +0.99 | +0.99 / +0.99 |
| C_0 | S3 | +0.58 / +0.16 | +0.13 / +0.40 | -0.90 / -0.87 | +0.20 / +0.26 |
| C_0 | S4 | not run | +0.84 / +0.84 | +0.81 / +0.56 | -0.81 / -0.56 |

</details>

**Image change against tilt size and time gap.** Image change is the mean absolute grey-level difference inside the
evaluation region. For tilts it runs from the frame at the start plane to the frame at each tilt, at the same time, and
averages both directions and both times. For time it runs between frames a given gap apart at the start plane.

| Volume | Fan | Volumes per s | Tilt 0.5° | Tilt 1° | Tilt 2° | Tilt 4° | Tilt 8° | Tilt 15° | Time 1 volume | Time 2 volumes | Largest time change | 1 volume ÷ 0.5° tilt | Tilt matching 1 volume |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_0 | A4C fan (S1–S3) | 6.2 | 3.7 | 7.2 | 12.9 | 20.4 | 27.3 | 33.3 | 19.3 (0.16 s) | 22.5 (0.32 s) | 25.5 (1.29 s) | 5.2 | 3.7° |
| A_0 | A2C fan (S4) | 6.2 | 3.4 | 6.4 | 10.6 | 14.8 | 19.4 | 24.3 | 18.7 (0.16 s) | 21.2 (0.32 s) | 24.3 (1.29 s) | 5.5 | 7.4° |
| B_0 | A4C fan (S1–S3) | 6.2 | 3.8 | 7.3 | 13.4 | 22.0 | 31.1 | 40.1 | 19.9 (0.16 s) | 23.0 (0.32 s) | 28.0 (1.45 s) | 5.2 | 3.5° |
| B_0 | A2C fan (S4) | 6.2 | 2.0 | 3.8 | 6.8 | 10.7 | 14.2 | 18.1 | 18.4 (0.16 s) | 21.2 (0.32 s) | 25.0 (1.45 s) | 9.3 | > 15° |
| C_0 | A4C fan (S1–S3) | 9.2 | 3.9 | 7.4 | 13.6 | 22.2 | 30.7 | 41.0 | 21.2 (0.11 s) | 25.7 (0.22 s) | 31.0 (1.31 s) | 5.5 | 3.8° |
| C_0 | A2C fan (S4) | 9.2 | 1.8 | 3.5 | 6.6 | 11.2 | 16.3 | 21.7 | 20.7 (0.11 s) | 25.0 (0.22 s) | 30.8 (1.31 s) | 11.4 | 13.7° |

- **The "5 times" belongs to its step sizes and planes.** In the A4C fan, one volume step (0.11–0.16 s) changes the
  image 5.2–5.5 times more than a 0.5° tilt. A tilt of 3.5–3.8° changes it as much as one volume step. A 15° tilt
  (33–41 grey levels) changes it more than any time gap up to half the recording (at most 25–31).
- **Around the A2C plane, a tilt changes the image less.** One volume step changes it 5.5–11 times more than a 0.5°
  tilt. Matching one volume step takes a tilt of 7.4° to more than 15°.
- **Time change.** The change grows over the first two volume steps and then levels off. In A_0 and C_0 it dips after
  about one heartbeat at the stored heart rate (6 and 8 volumes, against 6.1 and 8.9), when the heart returns to a
  similar shape. B_0 shows no clear dip. Later gaps rise again, which suggests slow drift over the recording.
- **What these numbers are.** The mean absolute difference mixes changes of anatomy with speckle decorrelation. These
  numbers compare sizes of image change, not amounts of useful signal. Matched smoothing moves the tilt curves by less
  than 0.5 grey levels. Plots per volume are in `numbers/artifact_control/<volume>/curves.png`.

**Answer to the question.** Could the geometric supervision we can generate be replaced easily by simple cues such as
smoothness?
- **Within a volume, partly yes.** In 11 of 24 sweeps a frame's contrast alone ranks the tilts almost perfectly, and
  sharpness tracks the tilt in some sweeps. A model that compares frames from the same volume could learn part of the
  tilt from these cues instead of anatomy.
- **Not explained by interpolation averaging alone.** Matching pixel-scale sharpness changes nothing, and the relation
  also appears with nearest-neighbour rendering. Smoothing already in the voxels, from the beam width or the scanner's
  processing, and other acquisition effects are not ruled out.
- **Across volumes, less consistently.** Contrast mostly falls with obliqueness, but the relation reverses in C_0's A2C
  fan and is weak on one side of 90° in several A4C-fan sweeps. Sharpness has no consistent direction.
- **Real 2D recordings.** Whether they share the cues is untested. If the cues come from anatomy, a real probe tilt
  changes them too. If they come from the 3D acquisition, real recordings would not share them.
- **What training would need.** Any training on slices would need at least two controls. Frames would be standardised
  to the same mean and SD inside the evaluation region. A baseline would predict the geometry from contrast and
  sharpness alone, to show how much of the target such cues explain. Passing both would not by itself show transfer
  to real 2D recordings.

## 6. What Echo-POSED has and what it does not address

| | Echo-POSED (from the paper) | This project |
|---|---|---|
| Supervision | Known relative 3D transforms of sliced views (equivariance) and invariance to cardiac phase, from random time samples; no view labels | Public clip-level view labels; constructed multi-fragment streams |
| Training data | Echo One: 397 videos after filtering (348 development, 49 test) from 527 patients; GE only; apical window | EV9V 2D clips |
| Input and output | One 2D frame (128 pixels high) → two rotations (probe pose) and ensemble uncertainty | 10 Hz 2D streams → view family per sample and segment limits |
| Validation | Angular error within volumes (x 2.1°, y 1.6°, z 11.5° on its test set); guidance simulations (8.2° within patients); external 3D sets | Segment F1@50, boundary recall and native macro-F1 on real 2D clips |
| Not addressed in the paper | Real 2D video; view classification; temporal segmentation; anatomical correctness across patients (it states that cross-subject view labels are missing); the augmentations it lists include no resolution or smoothing control | |
| Availability | Model and data announced; no link found | |

**The specific problem it leaves open.** In a real 2D video, can a geometry signal learned from 3D slices tell an
A4C→A5C plane change apart from cardiac motion, and locate it in time? This needs three things to hold.
- The signal is stable within a real clip across the cardiac cycle.
- It differs consistently between A4C and A5C clips across patients.
- It marks the A4C/A5C changes that B3 and ASRF miss (37% of them, round 4).

**What sets this apart from the paper.** In a real 2D video the plane change has to be found against cardiac motion.
Around the A4C plane, one volume step of the heart changes the image as much as a tilt of about 3.5–4° (section 5).
In re-sliced data, contrast also tracks the tilt, for reasons that may not carry over to real recordings. Echo-POSED
tests none of real video, view classes or temporal change, and it controls neither of these.

**How it would be validated.** Validation would use real EV9V 2D clips with public view labels, in four ways.
- Within-clip stability of the signal over the cardiac cycle.
- A4C/A5C separation, by clip and by session.
- Detection of the A4C/A5C joins in the constructed streams.
- Comparisons against B3 and ASRF, with a labelled reimplementation of Echo-POSED as the prior-method control
  (section 8).

Gains on simulated slices would not count.

## 7. Conditions for a training round

| Condition | Status | Evidence |
|---|---|---|
| Dynamic 3D data in hand, with enough independent subjects for the planned experiment | **Partly met** | The dynamic 3D data in hand now come from 19 people: 4 from EchoSlicer and 15 from STACOM 2011, downloaded and verified after this check. CETUS (45 patients) is described but not obtained; access and permission are pending. MITEA is static, and Echo One is not downloadable. |
| The new supervision does not use a classifier's predictions as truth | Met in principle | Relative plane geometry is exact by construction. EchoSlicer's standard-view planes come from a classifier-guided search and serve only as geometric starts, never as labels. |
| A prior method as a control | **Not a blocker** | Echo-POSED's code and weights have no download link. A reimplementation from the paper can serve if it is labelled as ours, comes with simple controls and claims none of the paper's results. |
| Slicing cues controlled | **Partly** | Section 5. Contrast tracks the tilt, and matched smoothing does not remove it. Training would need standardised frames and a contrast-only baseline. |
| A validation path back to real 2D recordings | Can be designed | Sections 6 and 8. Gains on simulated slices are not counted. |

**Verdict.** When this check was written, the conditions were not met, mainly because only 4 people's dynamic data
were in hand. That was a statement about the data, not a finding that the direction is infeasible. With STACOM 2011
added, a minimal three-arm experiment is now planned in `docs/superpowers/plans/2026-10-03-geometry-pretraining.md`
and `configs/experiments/geometry_pretraining_v1.yaml`. It starts only when the start conditions written there hold
and the project lead approves.

**Sample size.** No fixed number of subjects, such as 60, serves as a gate. Each experiment sets its own number from
its design. For example, a check of whether the slicing cues hold across people needs people held out of training. The
decisive comparison in section 8 is scored on real EV9V clips, whose number is fixed by the existing selection and
calibration halves.

**What would add data.**
- The CETUS organisers confirm that the dynamic sequences can be obtained and permit our use.
- Echo One becomes downloadable.

Either would add people to the same experiment without changing its rules. Progress does not wait on either.

## 8. The question a training round must answer

A later training round would answer one question. Does adding geometric supervision keep the A4C/A5C difference
better than ordinary pretraining with the same data and the same budget, while reducing false changes inside a
recording? The plan fixes three arms on one encoder. G0 is the E1 encoder as it is. G1 adds ordinary pretraining on
the slice pairs. G2 adds the same pretraining plus the geometric target, the signed tilt between two slices.
- **Comparison.** G1 and G2 use the same slice pairs in the same order, the same augmentation, trainable parameters
  and budget. They differ only in the geometric term.
- **Scoring.** Scores come from real 2D clips under the existing protocol, on the selection and calibration halves
  with no test split. They are A4C/A5C classification, the temporal segmentation metrics (segment F1@50, boundary
  recall within 0.5 s and native macro-F1), and false changes inside a recording, meaning predicted boundaries where
  the reference has none. The comparison is against B3 and ASRF.
- **What does not count.** A more accurate 3D angle prediction on slices does not meet the goal by itself.
- **Controls.** A fixed mask per pair, standardised frames and a contrast-only baseline come from section 5. A
  labelled reimplementation of Echo-POSED serves as the prior method if its weights remain unavailable.

The paper stays paused.

## 9. Limitations

- Only three volumes (three people) were re-sliced, with one reader and its linear interpolation.
- All tilts change θ_n. S1–S3 share one fan of planes, and S4 adds a second. Rotations of the plane about the probe's
  axis were not tested.
- The checks measure image statistics. They do not show whether a trained model would use those cues.
- The nearest-neighbour check removes only the linear interpolation done here. It does not remove smoothing already
  in the voxels, and it cannot separate anatomy from acquisition effects such as beam steering.
- Image change is a mean absolute grey-level difference, which mixes anatomy with speckle.
- Timing comes from the DICOM frame time and the stored heart rate, which were not checked against an ECG.
- Echo-POSED was read from its paper only, since no code or weights were available. Its reported numbers are not
  checked here.
- Access conditions are as found on 2026-10-03. Download pages and releases may change.

## 10. Files

- **Code.** `scripts/echoslicer_controlled.py` makes the controlled videos, and `scripts/echoslicer_artifact_control.py`
  runs the control in section 5. Both run in EchoSlicer's pinned environment, `third_party/3d-echo_run/.venv`.
  `scripts/summarize_artifact_control.py` builds the tables of section 5 from the numbers folder.
- **Config.** `configs/experiments/slice_artifact_control_v1.yaml` holds the declared design and the changes after the
  first run.
- **Data request.** `docs/data_requests/2026-10-03_cetus_request.md` holds the CETUS email draft and the agreement
  links.
- **Server setup.**
  - `third_party/3d-echo` is the pristine clone at `78f7085`. `third_party/3d-echo_run` is a working copy with the
    frozen EchoPrime view-classifier weights and the EchoNet-Dynamic LV weights.
  - The data are in `data/echoslicer/unpacked/` (zip SHA-256 `17b5f0b8…`). EchoSlicer's outputs are in
    `data/echoslicer/extracted/`, the controlled videos in `data/echoslicer/controlled/`, and the control's outputs in
    `data/echoslicer/artifact_control/`.
- **In this folder.**
  - `numbers/<volume>/` holds EchoSlicer's plane CSVs (with classifier scores), `frames.csv` (every frame's parameters
    and statistics) and `checks.json`. `numbers/sample/` holds the demo's plane CSVs.
  - `numbers/artifact_control/<volume>/` holds the control's per-frame table, tilt and time curves, summary and curve
    plots. These are statistics only, with no images of the data.
  - `numbers/artifact_control_run1/<volume>/` holds the first run's numbers, without the margin.
