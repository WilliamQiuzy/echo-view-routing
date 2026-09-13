# Dataset inventory and tiers (verified 2026-09-13)

Tiering rule: **Tier 1** = pixel data with temporal structure (frame sequences with order and a frame rate), usable
for temporal routing; **Tier 2** = pixel data without temporal structure (single frames), usable only for
frame-encoder recognition / source-shift checks; **Tier 3** = no pixels (labels or metadata only). Facts below were
read from the official pages on 2026-09-13; "unverified" means the page could not be read programmatically.

| Dataset | Pixel type | Temporal structure | Views | Access / licence | Tier | Role for us |
|---|---|---|---|---|---|---|
| EV9V | MP4 + JPEG frames, 320×240, 30 fps | full cines (25–3,243 frames) | 9 codes | Hugging Face, CC-BY-4.0 | **1 (core)** | training, constructed streams, primary evaluation |
| CAMUS | 2D B-mode sequences (raw/mhd), full cardiac cycle | yes (sequences; ED/ES annotated) | A2C, A4C (some "4CH" are A5C) | public download after registration | **1** | external-source video check for A4C; A2C and mislabelled A5C as unsupported-input tests |
| EchoNet-Dynamic | videos, 112×112 | yes (with ED/ES frame numbers) | A4C only | Stanford research-use agreement | **1** | class-conditional external test (source and class confounded) |
| EchoNet-LVH | videos, native resolution, masked | yes | PLAX only | Stanford research-use agreement | **1** | same, for PLAX |
| EchoNet-Pediatric | videos | yes | A4C, PSAX | Stanford research-use agreement | **1** | paediatric shift, reported separately |
| MIMIC-IV-Echo v1.0.1 | DICOM multi-frame sequences (524,137 files, 7,228 studies, 4,572 patients) | yes; per-study many sequences, study-level timestamps, no within-file view labels | all TTE views, unlabelled | PhysioNet credentialed (CITI training + DUA) | **1 (conditional)** | only candidate for real within-file transitions; needs expert annotation |
| EchoXFlow | beamspace recordings + synchronized ECG (native acquisition geometry, not scan-converted video) | yes | not view-labelled for our task | arXiv 2605.05447; files released separately | **1 (reserve)** | only if a specific temporal gap needs it; format conversion required |
| TMED-2 | 112×112 PNG single frames (one per 2D TTE image, Doppler/M-mode removed) | **no** frame order or cine membership | PLAX, PSAX, A2C, A4C, other | public (Tufts) | **2** | frame-encoder cross-source recognition on overlapping classes |
| TTE47 | 5,447 test images with three expert annotations | no (images) — page unverified programmatically | 47 fine-grained views | request-based | **2** | fine-grained ambiguity / unsupported-input test, images only |
| Unity Imaging | task-specific image sets (e.g. PLAX measurements, strain) | no video release found | task-specific | datasets CC BY-NC-ND 4.0 (NoDerivatives: no derived streams, no redistribution) | **2** | source-shift supplement for recognition only, if terms allow |
| ECHOVIEW v0.1 | none (CSV of machine-generated 23-class probabilities for 29,196 MIMIC videos) | n/a | 23 classes, pseudo-labels averaged over 10 random frames | PhysioNet credentialed | **3** | candidate sampling inside MIMIC; never ground truth |

## Decisions

- Tier 1 is the working set. Order of adoption: EV9V (now) → CAMUS (public, next) → EchoNet families (after the
  Stanford agreement) → MIMIC-IV-Echo (after credentialing; gates the natural-transition arm).
- Tier 2 stays in the plan as a **frame-level appendix** (encoder recognition under source shift). Static images are
  never repeated into pseudo-videos and never enter temporal training, banks, or temporal metrics.
- Tier 3 (ECHOVIEW) is used only to pick candidate MIMIC recordings for expert review.
- Any image dataset that turns out to ship ordered frames per cine (with a frame rate) is promoted to Tier 1; none of
  the checked ones does today.
