# Fan-only 224 input: training-image preprocessing pilot

Written for the project lead and research team. This is an exploratory input check,
not a model comparison or a new segmentation method. No classifier was evaluated or
trained. No validation or test images were read.

## What was done

The fixed existing protocol supplied the eligible EV9V training recordings. With
seed 20261006, six recordings per class were sampled, for 30 recordings in total.
Sixteen evenly spaced source frames per recording were processed: 480 frames.
Numeric source-frame order was preserved. Every sampled source image was 320 x 240.

A first attempt used temporal intensity variation and a largest-component hull to
estimate the image support. Visual inspection showed that the right-hand instrument
scale was included. This attempt was rejected as an automatic fan detector.

A conservative, shared sector template was then drawn after reviewing training
images. It is fixed across all recordings and frames in this pilot. The template
parameters and exact code are in `scripts/pilot_fan224.py`. This is an exploratory,
manually specified acquisition-layout template, not an anatomical mask or a fully
automatic segmentation model. Its validity outside this sample is unverified.

Three previews were generated for each recording:

1. Existing E1 preprocessing: short side 256, then centre crop 224.
2. Sector masking, followed by that same existing transform.
3. Sector masking, its bounding-box crop, aspect-preserving resize and black padding
   to 224 x 224. No second centre crop is applied.

Thirty middle-frame comparison panels were visually inspected. All 480 transformed
frames were generated, but they were not all manually inspected. The preliminary
template removes much of the peripheral interface. Some scale marks remain near
the sector edge. It is not a complete text-removal or de-identification solution.
No expert reviewed anatomical preservation.

## Findings and limits

- The existing E1 transform already removes most peripheral text in the reviewed
  images. Removing text therefore does not necessarily change much of its input.
- Keeping the full sector changes both field of view and scale. In this pilot the
  sector bounding box is 299 x 208, resized to 224 x 156 before padding. Thus anatomy
  is smaller than under the existing centre zoom. A complete fan does not imply
  higher effective anatomical resolution at a fixed 224 x 224 input.
- Approximately 12.35% of the **geometric template area** lies outside the field
  retained by the old crop. This is not a measurement of lost anatomy, important
  A5C structures, or the cause of any classification error.
- Reusing the current E1 loader on a delivered 224 x 224 image would resize it to
  256 and centre-crop it again. A distinct `fan224` path must instead validate the
  dimensions, convert to RGB and apply the existing numerical normalisation.
- Activity outside the mask is recorded in the JSON for auditing. Interface motion
  contributes to that statistic. Neither its retention nor its removal certifies
  anatomical preservation.

These observations justify a controlled input-adaptation comparison. They do not
show that cropping improves recognition, that text causes the A4C/A5C errors, or
that the remaining errors can be fixed without new independent evidence.

## Locations and reproduction

Server-only directory:
`/home/william/echo-view-routing/runs/20261006_fan224_sector_pilot`.
It contains the source-to-output mapping, fixed mask, derived PNGs and private
comparison sheets. The earlier rejected support estimate is retained separately
in `runs/20261006_fan224_pilot`. No raw images or recording identifiers are added to
the repository. `summary.json` contains anonymous sample numbers and aggregate
geometric checks only.

From the server project root, use a **new** output directory:

```bash
.venv/bin/python scripts/pilot_fan224.py \
  --root /home/william/echo-view-routing \
  --protocol runs/20261002-051140-model_development_protocol-fa45e7f4-s0/protocol.json \
  --manifest data/manifests/ev9v_native.csv \
  --out runs/fan224_pilot_reproduction
```

The command refuses an existing output directory and unsupported source dimensions.
The current transform is a prototype. Broader training-image review and a frozen
transform specification are required before model training. The preprocessing
team has not delivered its version; this pilot is not evidence about their output.

Completed execution checks: the preview's legacy transform was pixel-identical to
`crop_uint8`; the mask was deterministic; all 480 exported frames were RGB and
224 x 224; a 224 x 224 source was refused by the raw-image prototype, preventing an
accidental second preprocessing pass; and the source manifest matched the SHA-256
recorded in the existing protocol. These checks establish implementation and
provenance, not anatomical preservation or model performance.

## Modeling decision

Do not resume C/Z/R readout search. Use the native five-class visual task to test
whether a verified fan input and ordinary adaptation improve the evidence supplied
to B3. Compare both input formats with head-only adaptation and last-block fine-
tuning, starting from the same E1 weights. The complete instructions are in
`docs/superpowers/plans/2026-10-06-fan224-visual-adaptation.md`.

This is a bounded conventional adaptation experiment, not a claim of a new model
contribution. The existing full-system targets and public labels remain unchanged.
