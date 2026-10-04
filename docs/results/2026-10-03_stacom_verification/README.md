# STACOM 2011 3D echo: obtained and verified (2026-10-03)

STACOM 2011 is the open dataset of the 2011 cardiac motion-tracking challenge, hosted by the Cardiac Atlas Project
(Tobon-Gomez et al., Med Image Anal 17(6):632-648, 2013). It holds 15 healthy volunteers and one phantom. Each has
cardiac MRI and one 3D echo sequence. The check followed the plan in
`docs/superpowers/plans/2026-10-03-geometry-pretraining.md`. One volunteer was downloaded and checked first, then the
rest. The question was whether the field of view and the time sampling suit our task, not only how many people there
are.

No network was trained, and no slice got a label. Classifier scores come from EchoSlicer's view classifier, the one
used in the feasibility check. They show whether a slice looks like a standard view; they are not labels.

## Answer

1. **Obtained.** All 15 volunteers were downloaded from the public page, with no registration or agreement. The page
   asks users who publish to cite Tobon-Gomez et al. 2013.
2. **Decoding.** In all 15, the per-frame VTK volumes equal the Philips DICOM voxel for voxel. The number of frames
   matches the challenge Readme in every case (11 to 24).
3. **Time.** The DICOM frame time gives 17.8 to 21.1 volumes per second. Each sequence is one ECG-gated cycle of 0.62
   to 1.13 s. That is denser than EchoSlicer's 6 to 10 volumes per second, but it is one cycle per person.
4. **Coordinates.** The visible region fits a clean pyramid in every volunteer (R² of the width fits at least
   0.9998). Standard views appear where the geometry predicts them. A4C lies near the probe's central plane (rotated
   by −9° to +4° about the beam), A2C and A3C appear after rotating the plane one way about the beam, and A5C after
   tilting it by 10° to 26°. The challenge's expert landmarks are a weaker check (section 3).
5. **Field of view.** In one shared probe frame, the classifier recognises A4C in all 15 volunteers (0.83 to 0.99).
   It recognises A2C in 12, A3C in 12 and A5C in 10 (score at least 0.75, 0.5 and 0.5). Depending on the volunteer,
   the A4C plane keeps at least 80% of its visible area up to a tilt of 15° to 30° in each direction.
6. **Fit for the task.** STACOM can supply slice pairs that rotate and fan the plane around the left ventricle, and
   pairs across one cardiac cycle at 18 to 21 volumes per second. It also contains A5C-like planes in two thirds of
   the volunteers. Its limits are one gated cycle per person, healthy volunteers only, and one scanner model.
7. **Pipeline checks.** The slice pipeline passes the start-condition checks on all three EchoSlicer volumes with
   EchoSlicer planes and on 14 of 15 STACOM volunteers. The exception, v14, is explained in section 6.

All 15 STACOM volunteers meet the inclusion rule. With EchoSlicer's 26 eligible volumes, the experiment has 41
volumes from 19 people.

## 1. Source and access

- **Page.** https://www.cardiacatlas.org/motion-tracking-2011-challenge/ lists one zip per volunteer, the phantom, the
  ground truth (`cMac.zip`) and a Readme. Each link leads to a download page without registration or agreement.
- **Downloaded.** All 15 volunteers (v1, v2 and v4 to v16; there is no v3) and the ground truth, 6.4 GB in all. The
  phantom was not needed. The SHA-256 sums are in `numbers/zip_sha256.txt`.
- **Kept.** Only each volunteer's `3DUS` folder and the 3D echo ground truth were unpacked on the server. The MRI stays
  inside the zips.

## 2. Decoding and time

Each `3DUS` folder holds one Philips 3D DICOM (`IM_nnnn`, private class 1.2.840.113543.6.6.1.3.10002) and one VTK
file per frame. The VTK files reverse the DICOM's slice order and swap its rows and columns; after that, all voxels
are equal. The DICOM stores the slice spacing (0.580 mm), which matches the VTK. The in-plane spacing (about 0.66 mm)
appears only in the VTK headers.

| | Range over the 15 volunteers |
|---|---|
| Frames per sequence | 11 to 24, as in the Readme |
| Volumes per second (DICOM frame time) | 17.8 to 21.1 |
| Cycle length | 0.62 to 1.13 s, one ECG-gated cycle |
| Field of view (x by y by z) | 137 x 150 x 121 mm to 200 x 217 x 151 mm |
| Sector half-angles | 34° by 38° in 12 volunteers, 41° by 46° in v13 to v15 |
| Deepest visible point from the probe | 126 to 161 mm |

The per-volunteer table is in `numbers/tables.md`.

## 3. Probe sector and coordinates

**Sector.** The probe sits at one end of the box, and the visible region narrows linearly towards it. The fit finds
the probe position and the opening angles. Its centre line lies within 2° to 3° of the box axis, and the width fits
have R² of at least 0.9998 in every volunteer.

**Landmarks.** The challenge supplies 12 landmarks per volunteer in the echo frame, one per wall and level of the left
ventricle. They were transferred from MRI with a similarity transform fitted to three points marked on a four-chamber
view, so they carry registration error. Three tests were run.
- On the first frame, the landmarks are brighter than their ring centres, which sit in the cavity, in 14 of 15
  volunteers.
- Among the 48 axis permutations and flips of the volume, the volume as read ranks first in 4 volunteers and within the
  top 6 in 8. It ranks as low as 32nd in v2.
- The A4C and A2C planes that EchoSlicer's search chooses from the image both contain the left ventricle's long axis
  if the coordinates agree. Both planes pass within 10 mm of all three ring centres in 7 volunteers. The A2C plane does
  so in 12. The worst case is v15, at 32 mm.

The landmarks thus support the coordinates only loosely. The pipeline relies on the VTK geometry. Relative plane
geometry is exact given that geometry, and the view checks in sections 4 and 6 test its orientation.

## 4. Views and field of view

The probe's axes are not stored in the data, so EchoSlicer's search was run in candidate frames. These are four
directions of the azimuth axis for the A4C search, then both directions and both handednesses of the better axis for
the full search. One frame, with the azimuth along −y and mirrored relative to EchoSlicer's convention, is best on
its own in 13 of 15 volunteers. It is used for all volunteers.

For v7, v8 and v15, the A4C search alone had preferred the other axis. Their full search was then run on both axes;
for v7 the shared frame scored best. A 180° turn about the beam leaves A4C almost unchanged for the classifier but
reverses the side EchoSlicer tilts towards for A5C. In the shared frame, A5C scores a median of 0.79, against 0.14 in
the turned frame, and it is higher in 13 of 15 volunteers. This fixes which tilt direction is the A5C side.

| | In the shared frame (15 volunteers) |
|---|---|
| A4C score | 0.83 to 0.99 in all |
| A2C score at least 0.75 | 12 |
| A3C score at least 0.5 | 12 |
| A5C score at least 0.5 | 10 (v4 to v7, v10 to v12 and v14 to v16) |
| Tilt from A4C to the A5C plane | +10° to +26°, median +12° |
| Tilt keeping 80% of the A4C plane visible | +15° to +30° towards A5C, −17.5° to −30° the other way |

The five volunteers with low A5C scores (v1, v2, v8, v9 and v13) may lack the outflow tract in the field of view, or
show it poorly. EchoSlicer's own volumes scored 0.31, 0.37 and 0.72 for A5C in the feasibility check.

## 5. One probe-frame convention

`numbers/probe_frames.json` holds, for each volunteer, the shared frame as three axes (beam, elevation, azimuth) in
the volume's millimetres, the probe position, the frame time and the frame count. The slice pipeline
(`echo_routing/geometry`) loads this file. In this frame, as in EchoSlicer's volumes, the A4C plane sits near ψ = 0.
Rotating to negative ψ gives A2C, then A3C, in standard display orientation (section 6).

## 6. Pipeline checks

These are start condition 3 of `configs/experiments/geometry_pretraining_v1.yaml`.
- **EchoSlicer planes through the pipeline.** EchoSlicer's A4C planes for A_0, B_0 and C_0 were rendered by the
  pipeline. They score 0.997, 0.990 and 0.994, against EchoSlicer's own 0.991, 0.915 and 0.985.
- **Cartesian and spherical paths.** Each STACOM volunteer was rendered both directly and through a fine spherical
  resampling, in four planes (A4C, A4C tilted by ±10°, A4C rotated by 60°). On shared pixels the Pearson r is at least
  0.998, the best matching shift is zero in every case, and the mean absolute difference is below 1 grey level.
- **A4C score of the pipeline's STACOM rendering.** It falls within 0.1 of EchoSlicer's search result in 14 of 15
  volunteers. In v14 it is 0.975 against 0.847. The pre-declared check required agreement within 0.1 in both
  directions, so v14 fails it as written, although a wrong rendering would lower the score, not raise it. Whether to
  read the check one-sided is left to the start review.
- **Fixed masks.** At random tilts in ±15°, no mask pixel fell outside the volume in any of the 29 EchoSlicer volumes or
  the 15 STACOM volunteers.
- **Inclusion.** The masks cover a median of 34% to 64% of the slice in STACOM. In EchoSlicer, three thin or narrow
  volumes fall below the 20% rule (D_0 0%, D_3 10%, D_1 18%) and are excluded. That leaves 26 EchoSlicer volumes,
  person D with one of them.
- **Display orientation.** Sweeping ψ from the A4C plane, the classifier finds A2C and A3C at negative ψ in most
  volunteers and in EchoSlicer's B_0 and C_0. This is the range the pair sampler uses (−150° to +30°).

## 7. What this means for the experiment

- STACOM can enter the planned comparison as pretraining data, alongside EchoSlicer. The held-out people fixed in the
  config (EchoSlicer C; STACOM v14, v15 and v16) stay out of pretraining.
- Time pairs are denser than in EchoSlicer but cover one cycle per person, so beat-to-beat variation is absent.
- A5C-like planes are present in about two thirds of the volunteers. The pretext target is the tilt itself, not the
  view, so volunteers without them still contribute tilt pairs.

## 8. Changes made during the check

- **Display orientation.** The first version of the pipeline rendered every slice mirrored left to right. EchoSlicer's
  own A4C planes then scored 0.01 to 0.05 instead of about 0.99. The columns now run as in the standard display, and
  the ψ range was set to keep that orientation.
- **Sector fit.** A code review showed that choosing the probe side by run length could fit a box-clipped plateau and
  return a meaningless apex without error. The side is now the one whose widths fall on a straight line, and the fit
  raises unless it is straight and the apex lies within 15 mm of the visible region. The fitted sectors of all
  volunteers were unchanged.
- **Landmark ranks.** NaN scores (landmarks shifted out of the box) broke the sorting. They are now dropped before
  ranking, and all volunteers were rerun.
- **Robustness.** A failed EchoSlicer search in one candidate frame no longer stops the volunteer. The DICOM is found
  by its pattern, since its file number differs by volunteer. The mask's tilt grid is now 1° with a 2-pixel margin;
  with 2.5° steps, 7 pixels of one volunteer fell outside over 1,000 random tilts.

## 9. Limitations

- The classifier scores are checks, not labels. The classifier also reacts unevenly to mirroring. It rejected
  mirrored EchoSlicer A4C images but scored some mirrored STACOM A4C images about as high as the originals.
- The landmarks come from MRI and carry registration error, so they cannot certify the coordinates.
- The in-plane voxel spacing comes from the VTK headers only.
- The view checks use the first frame of each sequence.
- STACOM has one gated cycle per person, healthy volunteers only and one scanner model (Philips iE33).
- Volume rates come from the DICOM frame time and were not checked against an ECG.

## 10. Files

- **Code.**
  - `echo_routing/geometry/stacom.py` reads the data and fits the sector; `scripts/stacom_verify.py` runs the checks per
    volunteer.
  - `scripts/summarize_stacom_verify.py` builds the tables and the shared frame; `scripts/geometry_pipeline_check.py`
    runs the pipeline checks.
  - The tests are `tests/test_geometry_stacom.py` and `tests/test_geometry_pipeline.py`.
  - The verification scripts run in EchoSlicer's environment (`third_party/3d-echo_run/.venv`).
- **Numbers.**
  - `numbers/verify/<case>/` holds `summary.json` and EchoSlicer's view planes per frame (`views_<frame>/views.csv`);
    `numbers/verify_family_y/` holds the extra runs for v7, v8 and v15.
  - `numbers/probe_frames.json`, `numbers/pipeline_check.json` and `numbers/tables.md` hold the shared frame, the
    pipeline checks and the tables.
  - No image or video of the data is committed. They stay on the server under `data/stacom2011/verify*/`.
- **Server.** The zips and checksums are in `data/stacom2011/raw/` and the unpacked echo folders in
  `data/stacom2011/unpacked/`.
