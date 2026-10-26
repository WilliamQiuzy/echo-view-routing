# A35: frozen EchoFine as external A4C/A5C evidence (2026-10-26)

**Question.** Take a frozen recogniser trained on a different, fine-grained manual label source: EchoFine, 47 TTE47
views, Imperial College. On the same training-source frames, can it recover A5C runs that A08 + B3 reads as A4C (A32's
S1) while keeping correct A5C runs and full segmentation?
- **Scope.** This is a diagnostic only. It is not an independent holdout, a clinical truth, a cross-dataset ranking or
  a new system. Nothing was trained, tuned or distilled.
- **Data.** Only training data: no development, check, test or Pediatric data was read.
- **Predecessors.**
  - The intake (770cdef) described the resource; A32 tested EchoViewCLIP and EchoPrime the same way.
  - A31 and A34 showed that clip-code training fixes A4C runs read as A5C but breaks correct A5C runs.
- **Files.** Protocol `configs/experiments/a5c_echofine_evidence_v1.yaml` (3eb50c4, committed before any inference).
  Code 93a6d8b. Inference identity be46081, committed before the readout. Task sheet
  `claude_prompts/2026-10-25-echofine-frozen-evidence-diagnostic.md`.

## Configuration (one, fixed)

- **Model.** `thrive-2025/EchoFine_classifier`, Hugging Face commit 164aa2f3…, `TTE47_Classifier_Model.h5` (251.6 MB).
  - **Pinned sources.** The sha256 (e74c6581…) was checked byte for byte after download. The code and the
    `class_lookup.json` (f3b0f7e6…) come from EchoForge commit 94131d9b. Column 7 is a4ch-full, 13 a5ch-full and 14
    a5ch-outflow.
  - **Use.** A non-commercial internal frozen diagnostic, under the authors' "research use" statement. There is no
    licence file, and nothing was redistributed.
- **Environment.** An isolated uv environment `envs/echofine`: TensorFlow / Keras 2.13.1, with CUDA 11.8 and cuDNN 8
  libraries installed from pip inside the environment. The system CUDA and the project environment were untouched. The
  package lock is in `env_lock.txt`.
- **Inputs.** A32's support: 159,472 unique (recording, frame, gamma) references of A31's two pools (`plan.json`), with
  A32's frame order and gamma rule.
- **Preprocessing.** The whole RGB frame in 0–255 goes through `tf.image.resize(224, 224, bilinear, antialias=False)`,
  with no crop. The model's own Rescaling(1/255) is the only normalisation. Inference uses `training=False`.
- **Rule** (a revision of the intake draft, fixed in the protocol before any EchoFine score was seen).
  - The rule averages the full 47-class softmax over a Base run's samples.
  - A run whose Base class is A4C or A5C changes only if that average's unique top class is a4ch-full (→ A4C) or
    a5ch-full (→ A5C).
  - An unmapped top class, including a5ch-outflow, or a tie keeps Base. Runs of other classes never change. Equal
    neighbours merge.

## Checks (`inference.json`, `base.json`)

- **Downloads.** The weights sha256 matches the pin. The 47 columns follow the official lookup.
- **Output.** Softmax rows sum to 1 within 3e-7. There is exactly one Rescaling layer (1/255, offset 0). Input values
  span 0–254.4.
- **Official loader against the batched path.** The comparison covered 32 fixed plan rows, with EchoForge's own
  `build_echofine_classifier` pointed at the pinned file.
  - **First run failed.** The maximum difference was 3.8e-4, and batch 16 against batch 32 also differed by 4.3e-4.
    The cause was TensorFlow's TF32 matmul / convolution mode on the GPU, not double normalisation or the training
    flag.
  - **After the fix.** TF32 was switched off for every path (full float32), and the rerun passed: 7.7e-7 with the same
    argmax, batch 16 against 32 at 6.0e-7, and repeat inference identical. The failed check is booked.
- **Weights.** Unchanged before and after inference.
- **Rule.** Synthetic cases passed (`tests/test_echofine_rule.py`): unmapped top or tie means no change, the full classes
  change the run, other Base classes never change, sample multiplicity counts, and runs merge.
- **Base.** It was reproduced from A32's records in all four cells, with the same denominators: P runs 6,788 / 6,742,
  S1 103 / 101, correct A5C 1,150 / 1,128.
- **Inference.** One full pass took 312 s on the GPU (159,472 frames; probability cache sha256 65012881…, key e6ef1fcd…).

## Results (all P runs at real proportions; source seed 0 / seed 2, both folds pooled)

| Quantity | Seed 0 | Seed 2 | Protocol threshold |
|---|---|---|---|
| Fixed / broken / net | 110 / 310 / **−200** | 90 / 290 / **−200** | net ≥ 0 |
| A5C read as A4C fixed (S1) | **1 / 103** | **1 / 101** | ≥ 11 / ≥ 11 |
| Persistent S1 fixed | 0 / 88 | 1 / 79 | — |
| A4C read as A5C fixed | 109 / 135 | 89 / 118 | — |
| Correct A5C kept | **840 / 1,150 (73.0%)** | **838 / 1,128 (74.3%)** | ≥ 1,127 / ≥ 1,106 |
| S2 kept / S3 kept | 78 / 241; 759 / 904 | 86 / 223; 747 / 900 | — |
| Non-A4C/A5C runs changed | 0 | 0 | — |
| Native / constructed net | +3 / −203 | −4 / −196 | — |
| Short (≤ 16) / long net | −17 / −183 | −19 / −181 | — |

**Full sequences** (multi segment F1@50, Base → EchoFine rule).

| Cell | Segment F1@50 | Native macro-F1 | FN / missed short true | Short FP | Native A5C read as A4C |
|---|---|---|---|---|---|
| s0, held-out fold 0 | 0.9639 → 0.9327 | 0.9776 → 0.9575 | 81 → 200; 41 → 68 | 73 → 37 | 3 → 18 |
| s0, fold 1 | 0.9499 → 0.9093 | 0.9758 → 0.9473 | 102 → 254; 35 → 57 | 105 → 59 | 3 → 25 |
| s2, fold 0 | 0.9649 → 0.9334 | 0.9832 → 0.9575 | 80 → 197; 42 → 69 | 68 → 39 | 3 → 18 |
| s2, fold 1 | 0.9508 → 0.9097 | 0.9736 → 0.9520 | 112 → 254; 42 → 59 | 90 → 59 | 6 → 23 |

Segment F1 is below Base in **all four cells** (−0.031 to −0.041).

**Coverage** (the 47-class top of each run's mean; seed 0, seed 2 alike).

| Set | Runs | Top a4ch-full / a5ch-full / a5ch-outflow / focused A4C / other | Eligible / changed | Mean a4ch-full / a5ch-full | Unmapped mass |
|---|---|---|---|---|---|
| S1 (A5C read as A4C) | 103 | 102 / 1 / 0 / 0 / 0 | 103 / 1 | 0.90 / 0.05 | 0.055 |
| All true A5C | 1,255 | 412 / 836 / 0 / 3 / 4 | 1,248 / 311 | 0.34 / 0.59 | 0.072 |
| S3 (correct under every system) | 904 | 145 / 756 / 0 / 1 / 2 | 901 / 145 | 0.22 / 0.71 | 0.072 |
| All true A4C | 1,554 | 1,511 / 24 / 0 / 10 / 9 | 1,535 / 109 | 0.89 / 0.04 | 0.070 |

- S1 recovery is 1 / 103 overall and also 1 / 103 among eligible runs (seed 2: 1 / 101 and 1 / 101).
- The rule could act on about 99% of A4C/A5C runs: the top class was a4ch-full or a5ch-full. Unmapped classes,
  including a5ch-outflow (mean probability ≤ 0.005), held about 7% of the mass.
- So the negative result does not come from evidence falling into unmapped classes.

**Context, not a matched comparison** (A32's frozen results, with different preprocessing and mapping). The net results
on the same P runs were EchoViewCLIP −3 / −14 and EchoPrime −186 / −185. Neither was rerun.

## Answers

1. **EchoFine adds no A5C evidence on these runs. It reads them as A4C, as A08 does.** On S1, EchoFine's mean
   probability is 0.90 a4ch-full against 0.05 a5ch-full, and it recovers 1 run per seed.
2. **It moves the A4C/A5C boundary further towards A4C than any earlier system.**
   - About a third of all true A5C runs have a4ch-full as their top class.
   - Applying the rule breaks 290–310 correct A5C runs and keeps only 73–74%, including 145 / 153 runs of S3 that
     every earlier system had right.
   - It fixes 89–109 of the A4C runs that A08 reads as A5C, the same direction as A31 / A34 and EchoPrime.
3. **Decision: not met in either seed** (S1, correct A5C kept and net all fail) and not met in any cell (F1 below Base
   in 4 / 4).
   - This closes **this EchoFine configuration only**: these weights, this preprocessing, the 47-class top-1 mapping and
     this acceptance rule.
   - It does not close all external recognisers. It does not show that EchoFine is wrong, or that EV9V's labels are
     wrong.
4. **What this does not tell us.**
   - The definitions behind the two label sets may differ (EV9V's A5C code against TTE47's a5ch-full).
   - The images differ in acquisition, and domain shift is untested. Whole-frame resizing distorts the aspect.
   - None of these was varied, and a class probability is not an expert finding about the LVOT.
5. **One next-step motive.**
   - Four independent ways of reading these frames now agree that S1 looks like A4C: A08, EchoViewCLIP, EchoPrime and
     EchoFine. EchoFine also turns 27% of the correct A5C runs into A4C.
   - What decides the A4C/A5C boundary on EV9V is therefore whether the clip codes match what an expert sees on these
     recordings.
   - That needs a human reader. The blind pilot tool is ready, with 24 fixed cases including the S1 recordings, and has
     0 real records. No further frozen-recogniser or clip-code variant is proposed from this result.

## Cost and files

- **GPU 0.10 of 1 h.** Two checks (one failed) and the full pass of 0.088 h.
- **CPU about 0.3 of 3 h.** Environment setup, fetch, plan, Base, readout and the agent's active wall clock, all in
  `runs/a5c_echofine_evidence_v1/ledger.jsonl`.
- **Records.** `plan.json`, `base.json`, `inference.json`, `env_lock.txt`, `readout.json`.
- **Code.**
  - `scripts/run_echofine_frames.py` (TF 2.13 runner).
  - `scripts/diagnose_echofine_evidence.py`.
  - `echo_routing/temporal/echofine_rule.py`, `tests/test_echofine_rule.py`.
- **Server only.** Weights (`models_frozen/echofine_classifier/hf_164aa2f3/`, read only), the EchoForge clone, frames,
  per-frame probabilities, per-run predictions and identifiers.
- **Intake note.** A dated wording note was added to the intake README: same platform and institution suggest a
  possible common source but do not prove overlap.

## Dated note (2026-10-26, from the residual inventory)

This note narrows wording only. No number, configuration or decision changes.

Answer 5 says "four independent ways of reading these frames", and the index row says "four readers". Both overstate
what the evidence is.
- **What they are.** These are the outputs of four models. They are not four experts.
- **Not statistically independent.** A08 and EchoViewCLIP were both trained on EV9V. The source relations of EchoPrime
  and EchoFine cannot be fully checked.
- **What agreement can settle.** Agreement among them does not establish whether a label is right or wrong, or whether
  the LVOT is visible.
