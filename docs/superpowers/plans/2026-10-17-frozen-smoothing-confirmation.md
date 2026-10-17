# Frozen all-neighbour smoothing: confirmation eligibility decision and protocol draft (2026-10-17)

**Status.** This file prepares a decision. It does not authorise opening any test data, it is not a pending model
experiment, and nothing in it has been executed. It was written from committed or kept records only. No test sample,
label, file list, prediction, per-case output or re-inference was read.

## Decision

- **Eligibility of the current sealed test for this question: not eligible.** It cannot serve as a new holdout for a
  confirmatory claim (section 1).
- **Recommendation: do not open the current test for this comparison.** The protocol draft in sections 2 to 5 is kept
  for a future holdout that meets the conditions of section 6. The final research goal is not met by A08's old-target
  result or by any single internal comparison, and a confirmatory evaluation is not an algorithmic contribution.

## 1. Audit of "the current test was not involved in development"

**Identity.**
- The current sealed test is the official EV9V test split. The model-development protocol of 2026-10-02
  (`scripts/model_development_protocol.py`; protocol a897483b) "kept the official EV9V train / validation / test split".
  It split the validation split's 72 recording sessions into the selection and calibration halves.
- No new holdout was created. Sealing the split changed its status, not its history.
- EV9V has 5,138 clips: train 3,683, validation 567 and test 888 (aggregate counts in the 2026-09-30 reports).

**Use before 2026-10-02.** The experiment ledger and records H01 to H09 show repeated use of this test split.
- Test scores of the baseline ladder: B0 to B8, STFM, EchoViewCLIP and EchoPrime views.
- The ViT family, including MViTv2-S initialised from the EchoPrime encoder. It had nine-code and ladder test scores, and
  it is the lineage of the V01 Ordered model.
- Multi-fragment streams from a frozen test bank, evaluated with thresholds, routing scores and fusions.
- Case-level error analysis of hard A5C test clips (H04), which motivated later A5C hypotheses.
- The corrected audit and the MS-TCN/ASFormer order and seed matrix.
- The index's correction 9 already says that H01 to H09 used historical EV9V test results.

**Use since 2026-10-02.** The records checked here (model-development round 1, V01, A01 to A08 and the post-A08
diagnostic) state that the test split was not read. For the rounds in between, the index records no test use. Their selection and calibration data were the two validation halves, which were reused many times.

**Sources and units.**
- Earlier audits record that recording sessions (file-name prefixes) never cross a split. Patient identities are
  unavailable, so patient overlap between splits is unknown. A session is not a patient.
- The historical test stream bank was built and evaluated before the frame-order fix. A new test stream bank would need
  the test file list, which only an authorised opening may read.
- V01 starts from EchoPrime's encoder. Whether EchoPrime's pretraining data overlaps EV9V recordings is not in the
  records.

**Verdict: not eligible.** The test split informed development before it was sealed, through model-family rankings,
thresholds and case-level error analysis. A confirmatory claim needs data that informed no development decision. A frozen
or read-only status does not mean unseen.

**Facts only the lead can confirm, unknown from the records:**
1. Whether pre-2026-10-02 test results informed choices that led to V01, A07 or A08, such as the EchoPrime MViT family.
2. Whether the hard A5C test cases inspected in H04 shaped the A5C-focused rounds.
3. Whether EchoPrime's pretraining data overlaps EV9V.
4. Whether any person or tool accessed the test split after 2026-10-02.
These facts cannot make the test eligible, because the pre-sealing use is already recorded. They would bear only on a
weaker, explicitly non-confirmatory check, which this file does not propose.

## 2. Candidate comparison (draft, for a qualified holdout only)

- **Candidate.** The frozen A08 all-neighbour models `models_frozen/video_within_view_v1/all_s{0,1,2}`, sha256 9b144925…,
  c8b09eca… and f95fed7b…. Each uses its committed calibration B3 setting (A08 `choices.json`, sha256 b2a731e5 at
  c99bb6af).
- **Reference.** The frozen V01 Ordered models `models_frozen/raw_video_context_v1/ordered_s{0,1,2}`, sha256 80f13a42…,
  74276323… and 923eb3af…. Each uses V01's stored B3 setting: half-window 9, NMS 5 and thresholds 0.29269428317686846,
  0.2845063773784935 and 0.30845250187026374.
- **Pairing.** all_s{k} is paired with ordered_s{k}, because A's seed k started from ordered_s{k}.
- **Exclusions.** No other model (no equal-label smoothing, no plain continuation), no merging and no decoder retuning.
  The inference code, configuration and choice records are pinned by commit and sha256 before any opening.
- **Selection history.** A was chosen after its development results were seen. A test of A would be a single check of a
  post-hoc candidate on new data, not a retroactive pre-registered win for A08.

## 3. Two questions, kept apart

**Q1, a performance requirement on the new data.**
- The old targets were three-seed means on two particular halves. 0.9821421709894332 came from the calibration half,
  where the decoders were tuned, so it does not transfer to unseen data.
- A single holdout therefore needs one requirement fixed in advance. The draft is a three-seed mean multi segment F1@50
  of at least 0.944409687184662, with boundary recall of at least 0.95 and native macro-F1 of at least 0.973 in every
  seed.
- Meeting Q1 says nothing about superiority.

**Q2, superiority over the strong reference.**
- **Primary endpoint.** The paired difference A − Ordered in multi segment F1@50, per seed, with a 95% interval from the
  resampling in section 4.
- **Supported** only if all three seeds' intervals lie strictly above zero and A fails no floor. An interval whose
  bound is exactly zero, or a difference of exactly zero, does not count as above zero.
- **Otherwise: superiority not demonstrated.** This is not equivalence, not ineffectiveness and not "valid only on
  development".
- **Secondary.** These are described, never used to choose: native macro-F1 and per-class recall, the A4C/A5C
  confusions, boundary precision and recall, FP and FN segments, segments of at most 16 samples, and same-view join
  cuts, each with its session count.

## 4. Inference units and feasibility

- **Seeds share cases.** The three seeds are three trained realisations evaluated on the same cases, not three
  independent samples of cases. The consistency rule addresses training-seed variation only.
- **Resampling unit: the recording session**, the smallest traceable source. Native clips belong to one session.
  Constructed streams splice several recordings and may span sessions. A strict confirmation must build streams from
  disjoint session groups, so that each session feeds at most one group, and resample those groups with their native
  clips and streams. Resampling streams independently is an approximation and could not be called a strict
  confirmation.
- **Feasibility from permitted information only.**
  - **The effect sizes on development.** A − Ordered was +0.0179, −0.0059 and −0.0079 on selection and +0.0010, +0.0048
    and +0.0040 on calibration. The sign changed between seeds, and the per-seed interval half-widths were about
    0.005 to 0.013.
  - **The size of a holdout like EV9V test.** It has 888 clips. Its session count is not in the records. The validation
    ratio of about 7.9 clips per session would suggest about 110 sessions, but that is an estimate, not a reading.
  - **What follows.** Session-disjoint stream groups would leave fewer independent streams than the development halves
    used. Intervals would therefore not be narrower than on development. With the development signs changing between
    seeds, the chance that all three intervals exclude zero on the positive side is low. The design has little power to
    separate a small true advantage from none. The likely outcome, "not demonstrated", would carry little information.
- **Consequence.** Even on an eligible holdout, this comparison would need either a larger holdout or a pre-stated
  acceptance that a negative outcome is weakly informative. This file does not tune the design to make it pass.

## 5. Opening and computation barrier (for any future execution)

**Preconditions.**
- The lead's explicit, written authorisation to open a named holdout.
- Approval of the final protocol by its sha256.
- Every choice frozen and committed with a git-object proof.
- The code tested on training or development data only. GPU engineering checks never justify reading the holdout early.

**One opening.**
- An access record is written before any holdout metadata is read (time, protocol sha256, operator).
- One evaluation runs, and a completion record follows it.
- Every opening is listed in the report.

**Failures.**
- A crash before any metric, prediction or per-case output exists may be rerun once, and only if three things hold:
  - the cause is an implementation fault independent of the data;
  - the fix changes no choice;
  - the failed attempt, its access record and the denominators it had processed are kept and reported.
- Otherwise the confirmation is void.

**Contamination.** Any of the following voids the confirmation. The run may then be reported only as an exploratory
description.
- Holdout predictions or metrics viewed before completion.
- Any change of candidate, seed, decoder, threshold, metric or comparison after opening.
- An added primary comparison.
- Any reuse of the holdout for tuning.

If superiority is not demonstrated, the holdout cannot feed back into tuning, seed or model picking, or a new primary
comparison under the same confirmation.

## 6. What would make a holdout eligible

- **Unseen.** Data that informed no decision of this project: no evaluation, ranking, threshold, error analysis or case
  inspection, before or after sealing.
- **Traceable units.** Recording-session identifiers (ideally patient identifiers) and a documented independence from
  the EV9V training and validation recordings and, if possible, from EchoPrime's pretraining data.
- **Power.** Enough sessions, and both native clips and change-containing sequences, to give the primary comparison
  power under session-group resampling. Real transitions with labelled boundaries would be preferable to constructed
  joins, which remain a proxy.

## 7. The exact actions the lead would approve

**For the current EV9V test,** the recommended decision is: do not open it for this comparison.

**If an eligible holdout is obtained later,** the lead approves three actions:
1. Naming the holdout as eligible, with the confirmations of section 1 and the conditions of section 6.
2. Approving the final protocol version by its sha256. It includes the stream-construction rule, the session-group
   resampling and the Q1 requirement.
3. Authorising one opening and one evaluation of the six pinned frozen models under section 5.
