# CETUS intake (2026-10-04): what the release holds and how much of it is usable

The CETUS organisers granted our research use on 2026-10-04 and pointed to a new collection on the Human Heart Project
database (record: `docs/data_requests/2026-10-03_cetus_request.md`). This intake followed the project lead's order.
One case was checked first: its ED and ES volumes, coordinates and annotations. Then the whole release was downloaded
and checked, and the usable people were counted.

No model was trained, and none is proposed here. The data stay on the server (`data/cetus/raw`) and are not
redistributed. This folder holds aggregate numbers only. Per-patient values, including the reference volumes derived
from the annotations, stay on the server.

## Answer

1. **Obtained.** All 45 patients were downloaded: 185 files, 647 MB, from a public collection that needs no login.
   Each patient has an ED volume, an ES volume and one LV cavity mask per phase. No 3D+t sequence and no other label
   is in the release, as the organiser's reply said.
2. **Usable.** 44 of 45 patients meet all four criteria, which were fixed after the first case and before the others
   were read.
   - patient14 fails one part of the sector criterion. Its ED and ES sector apices are 2.04 mm apart, against a limit
     of 2 mm.
   - Otherwise its two fits agree. The apices differ along the beam, not sideways (0.1 mm), and the opening angles
     differ by 0.09°.
   - It is not counted. It can still serve single-phase uses, but not pairs across phases.
3. **Coordinates.** The four files of every patient share one grid. The voxel size is 0.5763 mm in every direction for
   all 45 patients, once a header error is corrected (the size is stored in metres but labelled mm). The beam runs
   along the third array axis in every patient, and the scan sector fits a straight-sided pyramid. In the 44 usable
   patients, ED and ES share the probe frame: their fitted apices are a median 0.09 mm apart (at most 1.53 mm). Pairs
   between the two phases can therefore be built in one coordinate frame.
4. **Annotations.** Each mask holds one structure, the LV cavity, as a single connected region with the values 0 and
   255. In every patient and phase, the image is darker inside the cavity than in a 3-voxel shell around it, as blood
   pool against myocardium should be. The ES cavity is smaller than the ED cavity in all 45.
5. **Population.** ED volumes range from 80 to 408 ml (median 139) and ES volumes from 36 to 353 ml (median 87). The
   ejection fraction (EF) ranges from 0.14 to 0.57 (median 0.38), and 23 of 44 are below 0.40. This agrees with the
   challenge's mix of healthy patients and patients with infarction or dilated cardiomyopathy. The release does not say
   which patient belongs to which group or which of the three scanner vendors.
6. **A shortcut to avoid.** Outside the sector, each volume is filled with one constant grey value. It differs between
   ED and ES in 42 of 44 patients and is higher at ED in 35. The mean grey inside the sector does not differ in a
   consistent direction (higher at ED in 21 of 44). Any target involving phase must exclude everything outside the
   sector, or the fill alone gives the phase away.

## 1. Source, licence and handling

- **Collection.** https://humanheart-project.creatis.insa-lyon.fr/database/#collection/62eb991b73e9f0048c3a6c45 holds
  `dataset/` (45 patient folders, `LICENSE_TERMS.md`, `MANDATORY_CITATION.md`) and `jupyter/` (an example notebook that
  reads the files with SimpleITK).
- **Licence.** CC BY-NC-SA 4.0, for non-commercial research only.
  - Every use must cite O. Bernard et al., "Standardized Evaluation System for Left Ventricular Segmentation Algorithms
    in 3D Echocardiography", IEEE TMI 35(4):967-977, 2016, https://doi.org/10.1109/tmi.2015.2503890.
  - The organiser's reply allows sharing trained weights and derived features, and showing example slices in a thesis
    or paper. It forbids sharing the data themselves.
  - Under the licence, shared adaptations carry the same licence and the attribution.
- **Handling.** The data are kept only on the server. The repository holds code and aggregate numbers, and no image
  of the data is committed.
- **Integrity.** The server publishes no checksum (its SHA-512 field is empty). Every file was checked against its
  listed size, every gzip file decompressed fully, and the SHA-256 of each file is recorded on the server
  (`data/cetus/sha256.json`).

## 2. Criteria

| Criterion | Rule | Patients meeting it |
|---|---|---:|
| U1 decode | The four files read, share one grid and hold integer grey values 0 to 255 | 45 |
| U2 annotation | Mask values exactly 0 and 255 at both phases; the largest connected region holds at least 99% of the cavity; ES cavity smaller than ED | 45 |
| U3 alignment | At both phases, darker inside the cavity than in a 3-voxel shell outside it | 45 |
| U4 sector | At both phases one array axis gives a straight sector taper (R² ≥ 0.98, apex within 15 mm of the visible region); the same axis at ED and ES, with apices within 2 mm and opening angles within 1° | 44 |
| All four | | **44** |

The sector fit is the one used for STACOM (`echo_routing/geometry/stacom.py`), applied to each array axis in turn.

## 3. Acquisition and geometry (44 usable patients)

| Quantity | Min | Median | Max |
|---|---:|---:|---:|
| Sector half-angle, first lateral direction | 29° | 37° | 53° |
| Sector half-angle, second lateral direction | 30° | 38° | 49° |
| Deepest visible point from the apex | 101 mm | 138 mm | 204 mm |
| LV long axis against the beam (ED) | 0.8° | 9.7° | 28.3° |
| LV length along its long axis (ED) | 70 mm | 93 mm | 118 mm |
| Depth of the LV tip nearest the probe (ED) | 16 mm | 25 mm | 32 mm |
| Share of the cavity outside the fitted sector | 0% | 0% | 0.2% |

The LV long axis lies more than 20° off the beam in 7 of 44 patients. These are oblique apical acquisitions.

Grid sizes differ (33 distinct shapes, from 200 × 191 × 176 to 380 × 424 × 284 voxels), while the voxel size is
the same in all.

## 4. Quirks found on the first case

- The header stores the voxel size in metres (0.0005763) while labelling it mm. The reader multiplies it by 1,000.
- The grey value outside the sector is one constant per volume. It is that volume's most frequent value, from 0 to 36,
  and it differs between ED and ES (13 and 8 in patient01). A fixed threshold therefore misreads the sector. The first
  attempt used 13 for both phases and cut part of the ES sector.
- The last slice along each axis is 0.

## 5. What this changes

- **Two phases, not a cycle.** CETUS supplies spatial slice pairs and, in 44 people, pairs between ED and ES in one
  frame. It supplies no motion between them, no intermediate phase and no beat-to-beat variation. As recorded in the
  request file, it changes the temporal sampling and the data distribution, not only the number of people.
- **Distribution.** Most patients have impaired function, and three vendors are mixed, unlike the healthy STACOM
  volunteers and EchoSlicer's authors.
- **No restart.** None of this restarts the stopped geometry-pretraining route
  (`docs/results/2026-10-04_geometry_pretraining/README.md`). What the masks could supervise is listed in
  `supervision_inventory.md`.

## 6. Files

- **Code.**
  - `echo_routing/geometry/cetus.py` is the reader and holds the checks. `tests/test_geometry_cetus.py` has 7 tests.
  - `scripts/cetus_fetch.py` lists and downloads the release.
  - `scripts/cetus_verify.py` checks each patient against U1 to U4.
  - `scripts/cetus_mask_shape.py` measures the mask shape for the inventory.
  - These run in the separate environment `envs/cetus` on the server (nibabel).
- **Numbers.** `numbers/summary.json` holds the counts and ranges, and `numbers/mask_shape_summary.json` the mask shape.
- **Server only.** The data are in `data/cetus/raw/`, with the per-patient checks in `data/cetus/verify/`, the
  listing, the SHA-256 ledger and the per-patient mask shape.
