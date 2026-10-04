# New supervision from CETUS: inventory (2026-10-04)

This page records three things: which labels CETUS actually provides, what they could supervise, and what stays
unlabelled. It proposes no training. By the project lead's decision of 2026-10-04, the next training round waits for a
new testable mechanism, distinct from the stopped geometry-pretraining round. The counts come from the intake
(`README.md`).

## What the annotation provides

- **One structure.** The release labels the LV cavity at ED and ES. It does not label the myocardium, the right
  ventricle, the atria, the valves, the outflow tract or the aorta. It has no landmarks and no view labels.
- **Coverage.** Both phases share one probe frame in 44 people. All 45 can be used one phase at a time.
- **Geometry that follows from the cavity.**
  - Its long axis and the position of its apex.
  - Its volume at ED and ES, and hence the EF.
  - Its cross-section in any plane through the volume.
- **Rotation about the long axis.** The cavity's cross-section is nearly round (ratio of the second to the third
  principal extent: median 1.07, range 1.00 to 1.24). So the cavity barely fixes the rotation about the long axis,
  and that rotation separates A4C, A2C and A3C.
- **The base.** A check of how the cavity ends at the base could not tell a flat cut at the valve plane from a rounded
  end. The contouring protocol in Bernard et al. (2016) should be read before the base is used as a valve-plane target.

## What it could supervise

| Possible target | Built from | Real 2D check on the server | Caveats |
|---|---|---|---|
| LV cavity mask in a derived slice | The 3D mask cut with the image in the same plane | CAMUS A2C/A4C masks at ED, ES and over half a cycle; EchoNet-Dynamic A4C tracings at ED and ES | Slices are resampled from 3D and differ from native 2D in texture and resolution; 44 to 45 people |
| ED or ES of a slice, or the same plane at ED and ES | The two volumes of one person in one frame | ED/ES frames in CAMUS and EchoNet-Dynamic | The fill outside the sector gives the phase away, so mask it; only two phases; no motion in between |
| Where a slice sits relative to the LV: its angle to the long axis, its distance from the apex (foreshortening) | The mask and the plane's geometry | None: our 2D data carry no foreshortening labels | A geometric quantity, not a view |
| LV volume and EF of a person | The 3D masks | EchoNet-Dynamic EF (2D, another population) | 44 values; these are the challenge's reference values, so per-patient numbers stay on the server |

## What stays unlabelled for our task

- **A4C against A5C, and the other apical views.** No label gives the rotation about the LV axis or the outflow tract,
  which A5C shows. Neither the tilt of a plane nor its relation to the LV cavity is a view label. The 3D contours and
  tilts must not be used as A4C/A5C labels.
- **Motion and timing.** There is nothing between ED and ES, no full cycle and no beat-to-beat change.
- **Patient attributes.** Vendor and disease group are not in the files. The paper describes three vendors and three
  groups of 15.
- **Image quality.**

## Constraints that come with any use

- CC BY-NC-SA 4.0, for non-commercial research only, with the Bernard et al. (2016) citation. Shared adaptations
  carry the same licence.
- Trained weights, derived features and example slices in a thesis or paper are allowed. The data themselves may not
  be shared.
