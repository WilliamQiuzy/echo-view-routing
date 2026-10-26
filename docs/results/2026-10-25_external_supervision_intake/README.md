# External A4C/A5C supervision: resource intake after A34 (2026-10-25)

**What this is.** A time-boxed check of existing and directly obtainable resources (namespace
`post_a34_external_supervision_intake_v1`; scope 24947ef). It is not an A-numbered experiment.
- **Nothing was run.** No model was run, trained or downloaded, and no image, video or weight file was fetched.
- **Data not read.** No development, check, test or Pediatric data, and not the A26 runner.
- **New downloads.** Only 36 KB of public EchoForge metadata and code, pinned to commit 94131d9b. The Unity view index
  was already on the server (fetched 2026-09-29).
- **Why it was run.** A27–A34 trained on EV9V clip codes and never recovered A5C runs read as A4C. The question is
  whether a resource outside those codes can supply A4C/A5C evidence.
- **Status.** There is still no reader, and the pilot tool holds 0 real records.

## Summary table

| Source and exact release | What we actually hold | Human or machine | Unit | Real A4C / A5C and structure counts | Groups and splits | Licence (as stated) and open items | Overlap with our sources | What can be done now | What is missing |
|---|---|---|---|---|---|---|---|---|---|
| **Unity Imaging**, release 2020-12-05 (`png-cache.zip`, `labels.zip`), plus `view_index_20220705.csv` from the additional page | On the server: 7,522 PNG frames with labels, the official train / tune / ival lists, and the 2022-07-05 view index (15,500 files, no conflicts) | Labels: expert points and curves by UK cardiologists and echocardiographers (home page). View index: production method not stated | Single frames, sparse per cine | **A5C: 144 frames from 144 cines** (1 frame each; train 100 / tune 24 / ival 20), each with an expert **aortic-valve point** (`ao-valve-top-inner`) plus LV apex, MV hinges and LV endocardium. **A4C: 3,311 frames from 1,454 cines.** All 8 aortic keys are `off`, set by the plane scheme, not judged per frame. No LVOT key and no LVOT visibility label | Official train / tune / ival. A file hash identifies one DICOM; study and patient ids are not given, so patient disjointness is unknown | Release and `labels.zip` LICENSE.txt: **CC BY-NC-ND 4.0** (datasets); weights CC BY 4.0; code MIT. The additional page states **CC BY-NC-SA 4.0** for "model weights, labels, metadata, and images". It is unclear whether that covers the 2020 release; kept as an open item. The repository decision (2026-09) is evaluation only, no derived data, no redistribution | Imperial College, the same platform that annotated TTE47 (EchoFine's training data). Overlap with EV9V: none expected (different country). With EchoPrime: unknown | External, evaluation-only frame-level checks, within the existing decision | A training use needs the lead to revisit the evaluation-only decision. Single frames, not video. 144 A5C frames only. No negative structure labels |
| **TTE47 / EchoFine** default classifier (`thrive-2025/EchoFine_classifier`, HF commit 164aa2f3, file `TTE47_Classifier_Model.h5`, 251.6 MB, LFS sha256 e74c6581…) | Not downloaded. Public metadata only: `class_lookup.json`, loader, registry, requirements (EchoForge 94131d9b) | Model trained on manual labels: one cardiologist per training image; the public test set has three experts | Single 2D image → 47-class softmax | Training set 76,589 images (not public); 47 classes include **a5ch-full** and **a5ch-outflow** | Official train / validation / test, with no patient or study overlap (per THRIVE) | Weights "for research use" (README). No LICENSE file in the EchoForge repository, no licence on the HF card. Test subset CC BY-NC-SA 4.0 (not to be used as our training data) | Imperial College Healthcare NHS Trust, annotated on the Unity platform, so Unity is **not independent** of EchoFine. EV9V: none expected. EchoPrime: unknown | A frozen evidence diagnostic on EV9V training runs, like A32 (draft below) | The exact resize rule (only "resized to 224 × 224 × 3" is stated); a TF 2.13 environment; a 252 MB download |
| **TMED-2** | Nothing held | Human view labels (method not stated on the page) | 112 × 112 single frames | Views PLAX / PSAX / A2C / A4C / **other**. No A5C class and no structure labels | Not checked | Not requested (decision 2026-09-13) | — | Nothing for A5C | No A5C; "other" cannot stand in for A5C |
| **CAMUS** | Recorded in `docs/DATASETS.md` (public collection) | Human LV / myocardium / LA masks at ED / ES | 2CH / 4CH sequences | No A5C class. Some "4CH" acquisitions look like A5C but carry no label | 500 patients | Research terms, citation required | — | Nothing for A5C | The 4CH name is an acquisition label, not a per-frame A4C/A5C truth |
| **EchoNet-Pediatric** | Not read (licence item open) | — | — | — | — | Unchanged restriction | — | — | — |

## Unity in detail (official description against the files we hold)

- **Official label states** (dataset_structure page):
  - `point` / `curve` carry coordinates.
  - `off` means "the structure is definitely not in the image".
  - `blurred` means "the structure might be in the image, but there is no label available" and is masked from the
    loss.
- **What the files show.** On all 3,311 A4C frames the 8 aortic keys are `off`, which matches `plane_a4c = no` in
  `keys.json`. That is the labelling scheme's statement about the A4C plane, not a per-frame check by a person.
  - On the 144 A5C frames, only `ao-valve-top-inner` carries a point. The aortic sinus and sino-tubular junction keys
    are `off`.
  - The keys define no LVOT point and no LVOT curve. `keys.json` has plane flags for PLAX / A4C / A2C / A3C, and A5C
    appears only as a labelling task (`imp-echo-shunshin-a5c-lv-segment`).
- **Where the view label comes from.** The view index does not say how its labels were made. On our frames they agree
  with the plane-specific keys, for example the aortic-valve point on every A5C frame. Even so, they are best described
  as the plane of the expert labelling task, not an independent view read.
- **Counts** (`unity_counts.json`, aggregates recomputed by `scripts/intake_unity_counts.py`).
  - 7,523 label entries, of which 7,522 have a PNG; all file prefixes are "01" (development set).
  - Of the 15,500 view-index files, 7,978 are not in our cache (a4c 3,442, a3c 2,294, plax 2,107, a2c 135). **None of
    them is A5C**, so the 144 A5C frames are all of the A5C frames in the view index.
- **Licence.** The two pages differ, so both are recorded and neither overrides the other.
  - The repository's evaluation-only decision rests on the release licence (ND). NoDerivatives restricts sharing
    adapted material; on its face it does not forbid every internal training use. Changing the decision is not part
    of this item.

## EchoFine adapter facts (from public code and metadata only)

- **Architecture.** Keras Sequential: Input (224, 224, 3) → Rescaling(1/255) → Xception backbone → global average
  pooling → Dense(47, softmax).
  - Normalisation happens inside the model, so it takes RGB pixel values 0–255 and outputs probabilities.
  - The README says the images were "resized to 224 × 224 × 3". Interpolation, crop and aspect handling are not stated
    and must be fixed before any run.
- **Framework.** TensorFlow 2.13 (`requirements.txt`). Our server environment is PyTorch, so a separate
  `envs/echofine` would be needed. Weights come through `huggingface_hub` from `thrive-2025/EchoFine_classifier`, file
  `TTE47_Classifier_Model.h5` (251.6 MB, LFS sha256 e74c6581…, repository commit 164aa2f3, modified 2026-10-22).
- **Column order** (`class_lookup.json`, sha256 f3b0f7e6…, alphabetical):

| Columns | Classes |
|---|---|
| 0–2 | a2ch-full, a2ch-la, a2ch-lv |
| 3–6 | a3ch-full, a3ch-la, a3ch-lv, a3ch-outflow |
| 7–12 | a4ch-full, a4ch-ias, a4ch-la, a4ch-lv, a4ch-ra, a4ch-rv |
| 13–14 | **a5ch-full**, **a5ch-outflow** |
| 15 | apex |
| 16–23 | doppler-ao-descending, doppler-av, doppler-mv, doppler-pv, doppler-tissue-lateral, doppler-tissue-rv, doppler-tissue-septal, doppler-tv |
| 24–28 | mmode-a4ch-rv, mmode-ivc, mmode-plax-av, mmode-plax-lv, mmode-plax-mitral |
| 29–36 | plax-full-la, plax-full-lv, plax-full-mv, plax-full-out, plax-full-rv-ao, plax-tv, plax-valves-av, plax-valves-mv |
| 37–43 | psax-all, psax-av, psax-lv-apex, psax-lv-base, psax-lv-mid, psax-pv, psax-tv |
| 44–46 | subcostal-heart, subcostal-ivc, suprasternal |

  Column 16 is `doppler-ao-descending` in the lookup but `doppler-ao` on the dataset page.
- **What maps to A4C/A5C.**
  - Only the full-depth B-mode apical classes map directly: **a4ch-full → A4C** and **a5ch-full → A5C**.
  - **a5ch-outflow** includes the LVOT but may lack full depth or carry LVOT colour.
  - a4ch-ias / lv / rv are focused views, and a4ch-la / ra may contain colour Doppler.
  - These classes, a3ch-outflow (a different plane that also shows the LVOT), apex, all Doppler / M-mode and the
    remaining classes stay **unmapped**. Their mass is reported, not renormalised away.
  - A class name is a view category. It does not say whether a structure is visible.

## One next-model draft: a frozen EchoFine evidence diagnostic (not executed)

**Why this one.**
- **No usable human training resource right now.** Unity's 144 A5C frames are expert-labelled, but they are single
  frames, the current decision limits Unity to evaluation, and the set is small. TMED-2 and CAMUS have no A5C labels.
- **EchoFine is checkable.** Its class order and weight file are confirmed, and it is the only frozen recogniser at
  hand trained on fine-grained manual A5C labels from a source independent of EV9V.
- **Why A32 does not settle it.** A32's two recognisers (EchoViewCLIP trained on EV9V; EchoPrime with coarse views)
  do not cover this case.

**Draft** (would be A35, `a5c_echofine_evidence_v1`; needs its own authorisation).
1. **Inputs.** Exactly A32's support: every 10 Hz sample of A31's two fold pools (159,472 unique frames), with A08 +
   B3 runs for seeds 0 and 2 and A32's S1–S4 sets. No new data and no development / check / test / Pediatric
   reading.
2. **Model.** EchoFine_classifier at HF commit 164aa2f3 and LFS sha256 e74c6581…, in a pinned TF 2.13 environment,
   frozen.
   - **Preprocessing, fixed once.** The full 320 × 240 RGB frame, bilinearly resized to 224 × 224 without crop, values
     0–255; the model rescales internally.
   - **Parity check.** Same predictions for one image through the official loader and through a batched path.
3. **Rule, fixed and single.**
   - Within each Base run labelled A4C or A5C, average the per-sample probabilities of a4ch-full and a5ch-full, and
     relabel the run as the larger. All other runs are unchanged.
   - No threshold, no temperature, no use of a5ch-outflow or the focused classes in the decision.
   - Unmapped mass by set (including a5ch-outflow and the focused A4C classes) is reported descriptively.
4. **Readouts**, as in A32, at real proportions over all P runs:
   - fixed, broken and net counts;
   - S1 recovered and S2 / S3 correct A5C kept;
   - A4C ↔ A5C in both directions;
   - full sequences (segment F1@50, native macro-F1, boundary P / R, missed short true segments);
   - error overlap with A08 as reference-assisted diagnostics only.
5. **Decision rule, written before any output.** Call it complementary evidence only if, for both A08 seeds:
   - at least 10% of S1 runs are recovered;
   - at least 98% of correct A5C runs are kept;
   - net fixed over all P runs is at least 0;
   - segment F1 is not below Base in at least 3 of 4 cells.
   Otherwise, close the frozen external-recogniser route for EV9V A4C/A5C. Neither outcome licenses distillation
   automatically.
6. **Budget estimate.** Setup plus a 252 MB download; inference over 159,472 frames at about 0.1–0.3 GPU h (or about
   2 CPU h if TF 2.13 cannot use the server's GPU); readout about 0.1 CPU h.
7. **What it cannot show.** Labels stay unchanged and nothing is an expert truth. Unity cannot serve as an independent
   check of EchoFine (same platform). Agreement or disagreement says nothing about LVOT visibility.

**Unresolved before such a run.**
- Choose the resize rule (bilinear, full frame) explicitly, because the official text leaves it open.
- Confirm that the weights' "research use" covers this use, since there is no licence file.
- Decide whether a5ch-outflow belongs in the rule. This draft keeps it out to stay with full-depth B-mode classes.

## Not concluded

- These checks do not show that weak supervision, clip-label training or every external recogniser is useless.
- Unity human labels exist, but are limited in kind (single frames, aortic-valve point, no LVOT key), in number (144
  A5C frames) and by the current use decision.
- Human evidence on EV9V's own recordings (the pilot tool) is still the direct route, and it needs a reader.

## Cost and files

- **Cost.** GPU 0 h. CPU in `runs/post_a34_external_supervision_intake_v1/ledger.jsonl` (under 1 h).
- **Files.** `unity_counts.json` (aggregates), `scripts/intake_unity_counts.py`.
- **Not in the repository.** EchoForge metadata was read from the pinned commit and is not redistributed here. Unity
  files, file names and the view index stay on the server.

## Dated note (2026-10-26, from A35)

This note narrows wording only. No count, decision or file changes.

- **Platform and institution.** Unity and TTE47 share an annotation platform and an institution. That points to a
  possible common source; it does not prove that any particular image is in both.
  - Unity is therefore not described as an independent holdout for EchoFine.
  - Overlap is not described as confirmed either.
- **EV9V.** EchoFine and EV9V come from different institutions; that describes their sources. Without a checkable list,
  exact overlap, including any pretraining overlap, is unknown.
- **Model outputs.** A high class probability from EchoFine is a model prediction, not an expert annotation. A35 reads
  one fixed configuration only.
