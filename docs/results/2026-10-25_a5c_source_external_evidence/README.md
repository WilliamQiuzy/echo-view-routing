# A32: A5C error sources and frozen external recognition evidence (2026-10-25)

**Question.** A31 left two A5C problems: true A5C runs that A08 + B3 reads as A4C, and correct A5C runs that the
error-focused branches (F_E, T_E) turn into A4C. Are these runs concentrated in a few source recordings? Do they come
with a checkable label-mapping or input-support problem? Do two frozen external recognisers give usable A5C evidence on
the same inputs?

- **Diagnostic only.** Nothing was trained, fitted or searched. The original A08 weights, B3 cuts, window mapping and
  outputs did not change.
- **Data limits.**
  - Training data only: A31's two session-fold pools, built from A15's large-fit sources. No development, check, test
    or Pediatric data was read.
  - The A26 benchmark runner and its default test manifest were not used.
  - A08 and EchoViewCLIP were trained on EV9V train, so they have seen these recordings. EchoPrime's pretraining sources
    are unknown. None of this is a generalisation test, and no external system is ranked against A08.
- **Predecessors.** A31 (6ca15bc): the session cross-fit showed E moves the A4C/A5C boundary towards A4C. A26
  (frozen external benchmark): that is where the two external systems were registered.
- Configuration: `configs/experiments/a5c_source_external_evidence_v1.yaml` (4a3323f, committed before any output).
  Task sheet: `claude_prompts/2026-10-24-a5c-source-and-external-evidence-diagnostic.md`.

**Sets.** Each set is defined per A08 seed (0, 2) and held-out fold, over A31's P runs. A P run is a Base run with its
majority reference label.
- **S1.** True A5C runs that Base reads as A4C. "Persistent" means still A4C under all four A31 arms; that is the same
  set as "fixed by no arm".
- **S2.** True A5C runs that Base gets right but F_E or T_E turns into A4C. Reported per arm, as the intersection and as
  the union.
- **S3 (control).** True A5C runs that Base and all four arms get right.
- **S4 (control).** True A4C runs that Base reads as A5C and F_E or T_E fixes.

## Part 1: sources, identity and window support

**Attribution.** A run's weight is split over its source recordings by their sample shares. No run is forced onto one
majority source. Two views are reported.
- **True-class sources (main view).** Only sources whose clip label is the run's reference class.
- **Every source (`by_src` in `sets.json`).** Also counts the neighbouring fragments of other classes inside mixed runs.
  It overstates recording counts: for example, 745 instead of 169 recordings for all true-A5C runs.
- The true-class view was added after the first output showed this inflation. It changes counts and shares, not set
  membership.

**Source concentration** (true-class sources, both held-out folds; seed 0 / seed 2).

| Set | Runs | Recordings | Sessions | Mean / max uses per recording | Top-10 recordings' share of the set | Same 10 recordings' share of all runs of the class | Recordings for half the set |
|---|---|---|---|---|---|---|---|
| All true A5C | 1,255 / 1,230 | 169 / 169 | 108 / 108 | 9.9 / 9.7; 27 / 26 | 0.12 / 0.12 | 0.12 / 0.12 | 60 / 61 |
| S1 | 103 / 101 | 38 / 40 | 33 / 35 | 3.0 / 2.8; 12 / 12 | **0.60 / 0.62** | 0.11 / 0.10 | **8 / 8** |
| S1 persistent | 88 / 79 | 33 / 35 | 29 / 30 | 2.9 / 2.5; 12 / 7 | 0.67 / 0.64 | 0.11 / 0.10 | 7 / 7 |
| S2 (union) | 241 / 223 | 92 / 88 | 67 / 65 | 3.2 / 3.2; 12 / 10 | 0.29 / 0.31 | 0.10 / 0.09 | 21 / 19 |
| S2, F_E | 216 / 190 | 87 / 82 | 62 / 61 | 3.1 / 3.0; 9 / 10 | 0.30 / 0.33 | 0.09 / 0.08 | 21 / 17 |
| S2, T_E | 106 / 95 | 47 / 50 | 39 / 40 | 2.6 / 2.3; 8 / 7 | 0.50 / 0.48 | 0.09 / 0.07 | 11 / 11 |
| S3 (control) | 904 / 900 | 168 / 167 | 107 / 106 | 7.5 / 7.4; 19 / 18 | 0.13 / 0.13 | 0.10 / 0.10 | 52 / 52 |
| S4 (A4C control) | 103 / 91 | 68 / 58 | 62 / 53 | 1.6 / 1.7; 4 / 5 | 0.33 / 0.39 | 0.04 / 0.04 | 18 / 15 |
| All true A4C | 1,554 / 1,534 | 654 / 654 | 320 / 320 | 3.0 / 3.0; 11 / 11 | 0.06 / 0.05 | 0.06 / 0.05 | 186 / 188 |

**Run types** (seed 0 / seed 2).

| Set | Native / constructed | Short (≤ 16 samples) / long | Pure / mixed | Single / multi source |
|---|---|---|---|---|
| All true A5C | 170 / 1,085; 162 / 1,068 | 406 / 849; 394 / 836 | 676 / 579; 690 / 540 | 530 / 725; 534 / 696 |
| S1 | 19 / 84; 17 / 84 | 70 / 33; 57 / 44 | 75 / 28; 71 / 30 | 68 / 35; 63 / 38 |
| S2 (union) | 38 / 203; 32 / 191 | 84 / 157; 75 / 148 | 148 / 93; 129 / 94 | 122 / 119; 103 / 120 |
| S3 | 111 / 793; 113 / 787 | 246 / 658; 256 / 644 | 449 / 455; 487 / 413 | 336 / 568; 365 / 535 |
| S4 | 41 / 62; 33 / 58 | 58 / 45; 51 / 40 | 84 / 19; 70 / 21 | 78 / 25; 65 / 26 |

- **S1 is concentrated.**
  - The pools hold 169 A5C-coded recordings. S1 touches 38 / 40 of them, 41 over both seeds.
  - Ten of them carry 60–62% of S1's weight, and the same ten carry only 10–11% of all true-A5C weight.
  - Half of S1 comes from 8 recordings, against about 60 for all true-A5C runs.
  - The two seeds share most of S1's recordings:
    - fold 0: 12 of 12 / 14 (Jaccard 0.86), 26 of 34 / 34 identical runs;
    - fold 1: 25 of 26 / 26 (Jaccard 0.93), 44 of 69 / 67 identical runs.
- **But no recording is always misread.** Every A5C recording is reused in about 10 runs on average.
  - 38 / 40 of the 169 have at least one run read as A4C, and only 5 / 7 have at least half of their A5C weight read as
    A4C. None has all of it read as A4C.
  - The 5 / 7 mostly-misread recordings carry 35% / 44% of S1.
  - So S1 is a modest group of recordings that Base sometimes reads as A4C, depending on the run and the stream around
    it. It is not a list of recordings that are wrong everywhere.
- **S1 runs are short.** 57–70 of about 100 have at most 16 samples, against 32% of all true-A5C runs. Most are in
  constructed streams. S1 also has 19 / 17 native runs: parts of whole native A5C recordings that Base cuts off and
  reads as A4C.
- **S2 is broad.** E's breaks cover 92 / 88 recordings (101 over both seeds), with the top 10 at 29–31%. T_E's breaks
  are more concentrated (48–50%) than F_E's (30–33%).
- **Controls.** S3 matches the class baseline at 13%. Almost every A5C recording (168 / 167 of the 169) also has runs that
  every system gets right.

**Window support.** For each run, the A08 windows that the nearest-centre map assigns to its samples were weighted by
multiplicity. Two shares were measured over all 16 slots of those windows: slots that fall outside the run, and slots
under another reference label.

| Set (both seeds, both folds) | Slots outside the run | Slots under another reference label | Support-window centres inside the run |
|---|---|---|---|
| All true A5C | 0.150–0.158 | 0.130–0.153 | 0.995–0.999 |
| S1 | **0.266–0.313** | 0.119–0.158 | 0.956–0.980 |
| S1 persistent | 0.250–0.299 | 0.124–0.200 | 0.963–0.985 |
| S2 | 0.149–0.177 | 0.130–0.162 | 0.995–1.000 |
| S3 | 0.123–0.144 | 0.127–0.159 | 0.999–1.000 |
| S4 | 0.202–0.244 | 0.091–0.117 | 0.979–0.997 |

- S1's windows draw about twice as many slots from outside the run as the controls do, which follows from its short
  runs. Their share of slots under another reference label is about the same as the controls': 0.12–0.16 against
  0.13–0.16 for S3, with persistent S1 reaching 0.20 in one cell.
- So the extra outside support of S1 runs mostly carries the same A5C reference label. The window reaches beyond the
  Base run but stays inside A5C reference content. Contamination from another view's frames does not explain S1.
- The marker is at the window centre in every window, and the nearest-centre weights sum to 1 for every run.

**Label and mapping identity** (`sets.json`, `identity`). Every check passed.
- **Mapping.** family5_v1 puts A4C at 2 and A5C at 3. On the training rows the raw codes map exactly that way: A5C → 3
  for 217 clips, A4C → 2 for 858.
- **Stream labels.** Native streams carry their clip label throughout (1,487 / 1,487). Constructed streams carry their
  fragments' clip labels and sources (1,200 / 1,200).
- **Column order.** A08's head has 5 columns in label_index order. EchoViewCLIP's label list has rows 4 / 5 = "Apical
  4-Chamber View" / "Apical 5-Chamber View", mapped to 2 / 3. EchoPrime's A4C / A5C views map to 2 / 3.
- **Label nature.** EV9V labels are clip-level codes: one code per recording, with no frame-level annotation. The index
  columns are video_id, split, raw_label, label_index, n_samples and path. No LVOT or other structure annotation exists,
  so LVOT visibility is **unknown** for every recording.
- **Definitions.** The dataset card and the STFM README agree on "Apical 4-Chamber" / "Apical 5-Chamber". Their
  conflicts concern only the PSAX-level codes and PMPALA (`configs/labels.yaml`).
- **Anatomy.** The ASE 2019 guideline (Mitchell et al., J Am Soc Echocardiogr 2019;32:1–64, p. 26,
  https://doi.org/10.1016/j.echo.2018.06.004) says: "From the A4C view, the apical five-chamber view is obtained by
  tilting the ultrasound beam anteriorly until the LVOT, AV, and the proximal Asc Ao come into view."
  - A4C and A5C therefore differ by a tilt within one window. A recording-level code says nothing about which frames
    show the LVOT.
  - This round does not test that, and no model disagreement here is taken as evidence that a label is wrong.
- **Disclosure.** During setup, one quick check read the raw-code / label-index pairs with a CSV reader that parsed
  every row of the feature index before keeping split == train. No value from another split was used or printed. The
  committed stage (`training_raw_codes`) reads the training rows only.

## Part 2: frozen external recognisers on the same inputs

**Systems.** Both run on every sample of both pools, the same recordings and frames that A08 sees. Each uses its own
deployed preprocessing.
- **EchoViewCLIP stage 1** (best.pth a74c7b23). The original `scripts/run_windowed_echoviewclip.py`, unchanged:
  - 16 frames uniformly over 16-sample windows at stride 5, then nearest-centre mapping to samples;
  - short side 256, centre crop 224;
  - nine codes to family5, with PMPALA dropped.
  - It ran on 6 plan shards: 50,359 windows over 2,687 streams.
- **EchoPrime view classifier** (release v1.0.0, 046b7668). Per 10 Hz frame, with its own crop_and_scale and
  normalisation; 11 views to family5, unmapped mass dropped.
  - The batched pass over the 159,472 unique (recording, frame, gamma) references matched the original per-stream
    runner path on two streams, within 1.5e-6 and with identical argmax.
- **Common checks.** Both runners read the frames in the same numeric order and from the same directories as A08 (2,476
  / 2,476 recordings). The EchoViewCLIP outputs passed shape, sum and nearest-centre checks on all 2,687 streams.
- **Readout check.** Base scored by this round's evaluator reproduces A31's Base sequence rows and counts exactly, and the
  A31 arm counts reproduce A31's report.
- **Dropped mass is small.** Inside every set, EchoPrime's unmapped views carry 0.6–1.8% (A3C ≤ 0.6%, A2C ≤ 0.8%).
  EchoViewCLIP's PMPALA carries 1–4%. The mapping does not drive the results below.

**Rule.** Inside each A08 + B3 run, take the equal-weight mean of the external per-sample probabilities, then argmax;
equal neighbours merge. There is no threshold, weight, temperature or decoder change.

**Main table: all P runs at their real proportions, out-of-fold over both pools** (seed 0 / seed 2).
- The denominators are:
  - Base-wrong runs 254 / 237 and Base-correct runs 6,534 / 6,505;
  - S1 103 / 101, A4C runs read as A5C 135 / 118, and correct A5C runs 1,150 / 1,128.
- The A31 rows are the held-out branches from A31, given for reference.

| System | Fixed / broken / net | S1 fixed (A5C read as A4C) | A4C read as A5C fixed | Correct A5C kept | Multi segment F1@50 − Base (4 cells) | Native macro-F1 − Base (4 cells) |
|---|---|---|---|---|---|---|
| A08 + B3 (Base) | 0 / 0 / 0 | 0 / 0 | 0 / 0 | 1.000 / 1.000 | 0 | 0 |
| EchoViewCLIP rule | 73 / 76 / **−3**; 49 / 63 / **−14** | 5 / 5 | 63 / 37 | 0.961 / 0.971 | −0.0012, −0.0046, −0.0008, −0.0040 | +0.0008, +0.0064, +0.0013, −0.0014 |
| EchoPrime rule | 96 / 282 / **−186**; 82 / 267 / **−185** | 0 / 0 | 94 / 78 | 0.767 / 0.777 | −0.0267, −0.0360, −0.0270, −0.0353 | −0.0319, −0.0199, −0.0249, −0.0256 |
| A31 F_U (reference) | 30 / 17 / +13; 27 / 17 / +10 | 1 / 2 | 29 / 25 | 0.988 / 0.988 | see A31 | see A31 |
| A31 T_U (reference) | 67 / 58 / +9; 49 / 56 / −7 | 4 / 4 | 63 / 45 | 0.955 / 0.957 | see A31 | see A31 |
| A31 T_E (reference) | 88 / 113 / −25; 70 / 113 / −43 | 8 / 7 | 80 / 63 | 0.908 / 0.914 | see A31 | see A31 |

The four sequence cells are seed 0 held-out fold 1, seed 0 fold 0, seed 2 fold 1 and seed 2 fold 0.

**Set coverage** (seed 0 / seed 2).

| System | S1 fixed | S1 persistent fixed | S2 kept A5C | S3 kept A5C | S4 fixed |
|---|---|---|---|---|---|
| EchoViewCLIP rule | 5 / 103; 5 / 101 | 4 / 88; 4 / 79 | 209 / 241; 201 / 223 | 893 / 904; 889 / 900 | 54 / 103; 32 / 91 |
| EchoPrime rule | 0 / 103; 0 / 101 | 0 / 88; 0 / 79 | 102 / 241; 99 / 223 | 777 / 904; 772 / 900 | 80 / 103; 67 / 91 |

- **Neither recogniser finds the missing A5C runs.**
  - EchoViewCLIP fixes 5 of about 100 S1 runs per seed: 1 native and 4 constructed each time. EchoPrime fixes none.
  - Inside S1 runs, EchoPrime's mean per-frame probability is A4C 0.87 against A5C 0.09–0.10. It reads these runs as A4C
    as firmly as A08 does.
- **Their fixes are A4C runs read as A5C.**
  - EchoViewCLIP: 63 / 37 of its 73 / 49 fixes. It keeps 96–97% of correct A5C runs, about the level of A31's T_U, and
    breaks 41 / 30 of them into A4C.
  - EchoPrime: 94 / 78 of its 96 / 82 fixes, but it turns 266 / 250 correct A5C runs into A4C and keeps only 77–78%. That is the same
    boundary shift as A31's E arms, only stronger.
- **S2 looks borderline to EchoPrime as well.** On the correct A5C runs that E breaks, EchoPrime's mean probabilities
  are A4C 0.52–0.53 against A5C 0.44. On S3 they are 0.26 against 0.70. The runs E breaks are the ones an independent
  recogniser also places near the A4C/A5C boundary. This is descriptive and not tested further.
- **Error overlap** (reference-assisted, diagnostic only; not a system).
  - Of the 105 / 102 true A5C runs that A08 gets wrong, EchoViewCLIP is also wrong on 98 / 96 and EchoPrime on all.
  - All three are wrong on 98 / 96. So even an oracle choice among these three systems would recover at most 7 / 6 of
    the A5C errors.
  - Over all runs, A08 and EchoViewCLIP are both wrong on 181 / 188. EchoViewCLIP alone is wrong on 76 / 63, and A08
    alone on 73 / 49.

**Full sequences** (the rule applied to Base's runs; complete RLE scored).

| Seed, held-out fold | Base F1@50 | EchoViewCLIP | EchoPrime | Boundary P / R: Base → EchoViewCLIP | Missed short true segments: Base → EchoViewCLIP / EchoPrime | Short false positives: Base → EchoViewCLIP / EchoPrime |
|---|---|---|---|---|---|---|
| s0, fold 1 | 0.9499 | 0.9487 | 0.9231 | 0.942 / 0.968 → 0.956 / 0.954 | 35 → 49 / 52 | 105 → 71 / 64 |
| s0, fold 0 | 0.9639 | 0.9593 | 0.9280 | 0.956 / 0.968 → 0.965 / 0.954 | 41 → 67 / 71 | 73 → 53 / 48 |
| s2, fold 1 | 0.9508 | 0.9500 | 0.9237 | 0.952 / 0.966 → 0.961 / 0.954 | 42 → 54 / 55 | 90 → 66 / 59 |
| s2, fold 0 | 0.9649 | 0.9609 | 0.9295 | 0.959 / 0.970 → 0.968 / 0.958 | 42 → 64 / 67 | 68 → 52 / 42 |

- **EchoViewCLIP** is below Base in all four cells (−0.0008 to −0.0046). Relabelling merges runs into their neighbours.
  That removes false short segments (boundary precision up) but also removes true short ones: 12–26 more are missed, and
  boundary recall falls.
- **EchoPrime** is 0.027–0.036 below Base in every cell. Most of the loss is A5C turned into A4C: native A5C read as A4C
  rises from 3–6 to 16–26.
- **Native runs** (each belongs to one recording, so to one session exactly; 2,000 session draws over 384 sessions;
  descriptive):
  - EchoViewCLIP − Base run accuracy: +0.0126 [+0.0043, +0.0213] in seed 0 and +0.0038 [−0.0020, +0.0100] in seed 2;
    net +20 / +6 native runs.
  - EchoPrime: −0.0032 [−0.0158, +0.0096] and −0.0057 [−0.0178, +0.0060].
  - EchoViewCLIP's native gain is no A5C recovery: it fixes 1 of 19 / 17 native S1 runs.
- **Clip-level check.** Each model's own clip-level reading of whole native A5C clips (mean probability, argmax):
  EchoViewCLIP gets 63 / 67 and 68 / 75 right, EchoPrime 51 / 67 and 51 / 75. EchoViewCLIP trained on these clips.
  Neither recovers the A5C parts that Base cuts off and reads as A4C.

## Part 3: private review package

- **Location.** Server only: `runs/a5c_source_external_evidence_v1/review_package/`. It holds an index page, 24 case
  pages, `review_template.csv` and the private key. Nothing was copied off the server, uploaded or committed.
- **Selection.** Up to 6 unique recordings per set, by one fixed permutation (`default_rng([20261024, 32, 9])`).
  Distinct sessions come first, and no recording repeats across seeds.
  - The candidates are the sets' majority-source recordings over both seeds: S1 35, S2 90, S3 163, S4 68.
  - Each set got 6 cases from 6 distinct sessions. Four of the S4 cases are native runs; the S1–S3 cases are all from
    constructed streams.
- **Each case page** has:
  - a player over the source recording's original frames;
  - the target run's central A08 window: its 16 slots as A08 loads them, with the marker slot outlined and each slot
    marked inside or outside the run;
  - blank expert fields: view per expert, LVOT visible, image quality, label agrees with view, comments;
  - closed layers with the run's labels and outputs, the stream timeline (reference, A08 + B3, A31 F_E / T_E, both
    external rules, source letters) and the A4C / A5C probability traces of A08 and both external models.
  - Case ids are neutral, such as S1-03.
- **Status: built, not reviewed.** No expert judgement has been recorded, and every expert field is blank. Nothing here
  says whether a recording shows the LVOT or whether any label is wrong.

## Answers

1. **Concentration: yes for S1, no for S2.**
   - A08's A5C runs read as A4C come from 41 of 169 A5C-coded recordings over both seeds. Eight carry half of them, and
     the same recordings recur in both seeds.
   - These recordings are not wrong everywhere. Base reads most of their other runs correctly, and only 5 / 7 are mostly
     misread.
   - E's breaks of correct A5C spread over about 90 recordings.
2. **No checkable mapping or input-support defect was found.**
   - Every label, mapping, column-order, marker and nearest-centre check passed.
   - S1's wider window support stays inside A5C reference content.
   - The labels are clip-level codes, and LVOT visibility is unknown.
3. **No usable complementary A5C evidence from the two frozen recognisers.**
   - Both read S1 as A4C, just as A08 does: EchoViewCLIP fixes 5 per seed and EchoPrime 0.
   - What they do fix is A4C read as A5C. EchoViewCLIP breaks a similar number of correct A5C runs (net −3 / −14), and
     EchoPrime breaks many more (net −186 / −185).
   - Full segmentation falls in every cell for both.
   - This is no complementarity success. With A08 and EchoViewCLIP trained on these recordings, agreement among the
     three is also no evidence about the labels.
4. **The review package is built and unreviewed.** No expert judgement exists yet.

**Training recommendation (one).** No training round is supported on the current clip codes or on these external
outputs. The one training this evidence points to is frame-level A4C/A5C supervision, and it needs expert annotation
first.
- **Why.**
  - S1 is concentrated in a few recordings and stays A4C for three different recognisers.
  - A31 showed that training on the clip codes only moves the A4C/A5C boundary.
  - The ASE definition separates A5C from A4C by the LVOT / aortic valve coming into view with tilt. A clip-level code
    cannot place that inside a recording.
- **Annotation needed before any training.**
  - **Recordings.** About 123:
    - the 41 A5C-coded recordings behind S1 (37 behind persistent S1);
    - 41 A5C-coded control recordings from S3, by a fixed RNG with distinct sessions first;
    - 41 A4C-coded recordings from S4, of the 75 available.
  - **Classes.** A4C and A5C only.
  - **Fields per recording.**
    - Expert view: A4C, A5C, intermediate / off-axis, or other.
    - The 30 fps frame ranges in which the LVOT and the aortic valve are visible, or none.
    - Image quality: good, adequate or poor.
    - Whether the clip code fits the whole recording: yes, partly or no.
    - A second reader on at least the S1 recordings, so that agreement can be reported.
  - The 24-case package is a ready pilot for the field definitions.
- **Training once the annotations exist.** A recognition branch whose A4C/A5C targets come from the expert frame ranges,
  against a matched-compute branch on the clip codes.
  - Both relabel only the original B3 runs, as in A31, so the cuts stay protected.
  - Both are read out with A31's session cross-fit and full-sequence scoring.
  - Neither labels nor data are changed without the expert record. Nothing here extends to the rest of EV9V.

## Cost and files

- **GPU.** 0.62 of 2 h, with no failed job.
  - EchoPrime pass and parity: 0.08 h.
  - EchoViewCLIP timing on 48 streams: 0.02 h.
  - Full EchoViewCLIP run: 0.52 h.
  - The fallback session subset was not needed: the timing projected about 0.45 h for the full run.
- **CPU.** 0.76 of 3 h. Ledger stages took 0.02 h. The agent's active wall clock is booked in three spans with waiting
  excluded: 0.46 h, 0.26 h and a short final span for the index, memory and report. All of it is in
  `runs/a5c_source_external_evidence_v1/ledger.jsonl` on the server.
- **No training.** No backward pass, optimizer step or prefix-cache rebuild.
- **Records.**
  - `sets.json`: sets, concentration, identity, window support, cross-seed overlap.
  - `readout.json`: checks, run tables, coverage, overlap, sequences, native intervals.
  - `extras.json`: recordings counted three ways, S1 / S3 by run type, dropped-view mass.
  - `package.json`: review package scope and status.
- **Code.**
  - `scripts/diagnose_a5c_evidence.py`, stages plans / echoprime / evc / sets / readout / package / extras.
  - `scripts/a5c_review_package.py`.
  - `tests/test_a5c_evidence.py`.
- **Server only.** Plans, external outputs, per-run masks and predictions, the review package and every identifier.
