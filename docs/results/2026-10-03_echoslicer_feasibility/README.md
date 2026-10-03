# Can 3D echocardiography supply a new training signal? Feasibility check (2026-10-03)

The short-span classification route ended after round 6, and no new model was trained here. Six rounds kept
re-reading the same clip labels on the same images. None of that added supervision that separates two kinds of change
in a 2D video: the heart moving, or the imaging plane moving. This check asks whether 3D echocardiography can supply
that supervision.
- **Where would the new information come from?** 3D volumes can be re-sliced with known planes and times.
- **What has prior work not solved?** Echo-POSED is the closest prior method.
- **How would a model be validated?** Gains on simulated slices do not count; the check must return to real 2D videos.

No echo expert is available. Nothing here gives a view label to any re-sliced frame, and the re-sliced videos are not
real continuous scans.

## Answer

1. **New information.** Re-slicing 3D volumes does give exactly known plane and time parameters. The supervision
   would be relative geometry, which is exact by construction. It does not come from any classifier's predictions.
   - **Reproduced.** EchoSlicer's code reproduced on its sample and on three of its volumes.
   - **Controlled videos.** Three controlled videos per volume were made with their parameters saved: plane fixed,
     plane tilting, and both changing.
   - **Leaks.**
     - The visible region reveals the tilt unless a common mask is applied.
     - Even with the mask, image contrast falls with tilt in all three volumes. This is probably interpolation
       smoothing, and real 2D frames do not carry such a cue.
   - **Change per step.** A 0.5° tilt changes the image about 5 times less than one time step of cardiac motion.
2. **Data.** Too few independent subjects are available now.
   - **Without a request:** EchoSlicer has 4 people and STACOM 2011 has 15 volunteers.
   - **With registration:** CETUS adds 45 patients. MITEA has 134 subjects but only two frames per scan.
   - **Not downloadable:** Echo One (527 patients) is announced in the Echo-POSED paper, but no download was found.
3. **Prior method.** Echo-POSED (May 2026) learns a frame-level probe pose from 3D slices, invariant to cardiac phase.
   It never tests real 2D videos, view classes or temporal change. Its code and weights are announced, but no link was
   found, so it cannot yet serve as a runnable baseline.
4. **Start conditions.** Two of the four do not hold: enough independent subjects, and a runnable prior method.
   **This direction does not meet the conditions for a training round now.** Section 6 lists what would change that.

## 1. What is available

| Resource | What it is | Access found on 2026-10-03 |
|---|---|---|
| EchoSlicer code ([github.com/echonet/3d-echo](https://github.com/echonet/3d-echo), commit `78f7085`) | Reads Philips 3D DICOM, finds standard planes by a classifier-guided search (EchoPrime view classifier, EchoNet-Dynamic LV segmentation), re-slices 2D videos | Public; **no LICENSE file** |
| EchoSlicer dataset (release v1.0, 1.1 GB) | 29 3D+t volumes from **4 people** (A 15, B 5, C 5, D 4), the authors themselves; Philips EPIQ CVx | Public download; README calls it open-source; no license file; the DICOM headers contain a patient-name field (not read) |
| Echo-POSED ([arXiv 2606.02634](https://arxiv.org/abs/2606.02634), 30 May 2026) | Self-supervised probe-pose model trained on 3D slices | The paper says the model and Echo One are released; **no URL** in the paper, on GitHub or in web search |
| Echo One | 527 patients, 397 3D videos after filtering, GE Vivid E95 | Announced only |
| STACOM 2011 motion challenge | 15 healthy volunteers + 1 phantom, 4D apical volumes, DICOM | Open access with citation (tokenised links) |
| CETUS 2014 | 45 patients, one cardiac cycle each, three vendors | Registration required |
| MITEA | 134 subjects, 2 scans each with **2 frames** (end-diastole, end-systole) | Institutional request form, CC BY-NC-SA 4.0, no redistribution |

**Content of the EchoSlicer volumes.**
- **Typical volumes.** Most cover about 81° in azimuth and 64–109° in elevation, at 6–10 volumes per second (16–31
  per cycle).
- **Narrow, fast volumes.** Five trade field of view for speed: A_3, C_1, D_0, D_1 and D_3, at 9–30 volumes per
  second, with elevation down to 19°. D_0 is a thin slab, so a 15° tilt leaves it.
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
| fixed_plane | start plane | every volume of the cycle | 18–25 |
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
| Contrast (SD inside the common mask) against tilt, Spearman | −0.83 | −0.98 | −1.00 | Systematic in all three. It is not removed by the mask. |
| Contrast change over the tilt sweep, against over time at a fixed plane (relative range) | 0.08 vs 0.13 | 0.14 vs 0.06 | 0.42 vs 0.18 | In B_0 and C_0 the tilt changes contrast more than the heartbeat does. |
| Sharpness relative range: tilt sweep, against time at a fixed plane | 0.62 vs 0.06 | 0.57 vs 0.11 | 0.52 vs 0.10 | Large, but with no consistent direction (Spearman 0.11, 0.31, 0.04). |
| Frame size across 0–15° tilt (EchoSlicer default grid) | constant | constant | constant | No size leak in this range. |
| Per-frame min–max rescaling (EchoSlicer default) | not used here | | | It would change contrast with the plane, so it was avoided. |

**What the contrast drop probably is.** The selected A4C plane lies close to the probe's native scan plane. Tilting
moves the slice into the elevation direction, which is sampled more coarsely (about 1.7° against 1.1° in azimuth).
Interpolation then smooths it more. The contact sheets show visible radial smearing at 15°. This explanation is a
hypothesis, because no blur-matched control was run.

**Consequence.** A model trained on such slices could learn "smoother means tilted". Real 2D frames have no such cue,
because they are acquired natively in every plane. Any training on slices would need a control: for example matched
smoothing, or start planes that are themselves oblique with tilts in both directions. The check would then be repeated.

**Change per step.** Inside the common mask, the median absolute change between consecutive frames is about 19–20
grey levels for one time step at a fixed plane. For a 0.5° tilt at a fixed time it is about 3.5. When both change, it
is about 19–20 again, dominated by the heart.

**Code review.** The first version did not compute the correlations against time for the fixed-plane video. They are
now in `checks.json`: Spearman of sharpness, mean and spread with the time index ranges from −0.49 to 0.74, with no
consistent sign across volumes. Time is cyclic, so these are less informative than the ranges above. The rerun differs
from the first run only by floating-point noise, at most 0.03 on sharpness values of 350–600. Every value reported
here is identical at two decimals.

## 5. What Echo-POSED has and what it does not address

| | Echo-POSED (from the paper) | This project |
|---|---|---|
| Supervision | Known relative 3D transforms of sliced views (equivariance) and invariance to cardiac phase, from random time samples; no view labels | Public clip-level view labels; constructed multi-fragment streams |
| Training data | Echo One: 397 videos after filtering (348 development, 49 test) from 527 patients; GE only; apical window | EV9V 2D clips |
| Input and output | One 2D frame (128 pixels high) → two rotations (probe pose) and ensemble uncertainty | 10 Hz 2D streams → view family per sample and segment limits |
| Validation | Angular error within volumes (x 2.1°, y 1.6°, z 11.5° on its test set); guidance simulations (8.2° within patients); external 3D sets | Segment F1@50, boundary recall and native macro-F1 on real 2D clips |
| Not addressed in the paper | Real 2D video; view classification; temporal segmentation; anatomical correctness across patients (it states that cross-subject view labels are missing); the augmentations it lists include no resolution or smoothing control | |
| Availability | Model and data announced; no link found | |

**The specific problem it leaves open.** In a real 2D video, can a geometry signal learned from 3D slices tell an
A4C→A5C plane change apart from cardiac motion, and locate it in time? This needs three things to hold:
- the signal is stable within a real clip across the cardiac cycle;
- it differs consistently between A4C and A5C clips across patients;
- it marks the A4C/A5C changes that B3 and ASRF miss (37% of them, round 4).

Two findings above show this is not a relabelling of the paper. Per step, the plane signal is about 5 times weaker than
cardiac motion. And in sliced data, tilt is confounded with smoothing that real frames do not have. Echo-POSED tests
none of real video, view classes or temporal change, and controls neither confound.

**How it would be validated.** On real EV9V 2D clips with public view labels:
- within-clip stability of the signal over the cardiac cycle;
- A4C/A5C separation, by clip and by session;
- detection of the A4C/A5C joins in the constructed streams;
- comparisons against B3, ASRF and Echo-POSED (if its weights appear).

Gains on simulated slices would not count.

## 6. Conditions for a training round

| Condition (user) | Status | Evidence |
|---|---|---|
| Data available, with enough independent subjects | **Not met** | Dynamic 3D data without a request: 4 people (EchoSlicer) + 15 volunteers (STACOM 2011). With registration: CETUS adds 45. MITEA is static. Echo One is not downloadable. Echo-POSED developed on 348 videos from its 527-patient set. |
| New supervision does not use a classifier's predictions as truth | Met in principle | Relative plane geometry is exact by construction. EchoSlicer's standard-view planes come from a classifier-guided search and must not be used as labels. |
| A clear prior method as a control | **Not met** | Echo-POSED's code and weights have no download link. A reimplementation from the paper would be a weaker stand-in. |
| A validation path back to real 2D recordings | Can be designed | Section 5. Gains on simulated slices are not counted. |

**Verdict.** The direction does not meet the start conditions now.

**What would change the verdict.**
- Echo One and the Echo-POSED weights become downloadable. That would meet the data and control conditions together.
- Alternatively, the user registers for CETUS (and uses STACOM 2011) and accepts about 60 subjects as an early
  technical test only. That is still small against the paper's development set, so no generalisation claim could rest
  on it.

## 7. Limitations

- Only three volumes (three people) were re-sliced, and only tilts about one axis from one start plane.
- The leak checks measure image statistics. They do not test whether a trained model would actually use those cues.
- Echo-POSED was read from its paper only, since no code or weights were available. Its reported numbers are not
  checked here.
- Access conditions are as found on 2026-10-03. Download pages and releases may change.

## 8. Files

- **Code:** `scripts/echoslicer_controlled.py` runs in EchoSlicer's pinned environment,
  `third_party/3d-echo_run/.venv`.
- **Server setup.**
  - `third_party/3d-echo` is the pristine clone at `78f7085`. `third_party/3d-echo_run` is a working copy with the
    frozen EchoPrime view-classifier weights and the EchoNet-Dynamic LV weights.
  - The data are in `data/echoslicer/unpacked/` (zip SHA-256 `17b5f0b8…`). EchoSlicer's outputs are in
    `data/echoslicer/extracted/` and the controlled videos in `data/echoslicer/controlled/`.
- **In this folder:** `numbers/<volume>/` holds EchoSlicer's plane CSVs (with classifier scores), `frames.csv` (every
  frame's parameters and statistics) and `checks.json`. `numbers/sample/` holds the demo's plane CSVs.
