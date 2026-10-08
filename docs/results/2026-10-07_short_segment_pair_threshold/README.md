# Short-segment discriminators with an A4C/A5C threshold: a post-hoc decision calibration check (2026-10-07)

The transfer check found that a good overall keep rate can hide heavy losses in the A4C/A5C pair. This check asked
whether a separate threshold for that pair can keep the correct A4C/A5C segments and still remove the wrong ones.

**What kind of evidence this is.** This is a post-hoc decision calibration check. The rule was set after the A4C/A5C
problem had been seen on these validation halves, and the project has used the halves repeatedly. It is exploratory,
not an independent validation. Nothing was integrated into the full system, no candidate was actually removed or
merged, and the test split stays sealed.
- The plan is `docs/superpowers/plans/2026-10-07-short-segment-pair-threshold.md`.
- The configuration, code and tests were committed before any new threshold was computed (cc4f0ae).
- The new thresholds were committed before they were applied to the validation halves (3c0d6e1).
- An independent fact-check followed the results. A reporting step (`--diagnose`) then added hindsight overlap
  diagnostics. They read the labels and set no threshold, and nothing was rescored.
- The transfer check's report and its conclusions are unchanged
  (`docs/results/2026-10-07_short_segment_transfer/`).

## Answers

1. **What was compared.** Two decision rules, fixed in advance, with no threshold grid.
   - **The original rule** applies each model's original threshold to every candidate.
   - **The pair rule** gives candidates whose predicted pair is A4C/A5C their own threshold. Both directions,
     A4C → A5C → A4C and A5C → A4C → A5C, share it. Every other pair keeps the original threshold. Only the decoded
     classes decide which threshold applies; reference labels are never read.
   - **Unchanged inputs.** The six frozen discriminators, the candidate generator (J0, κ = 8), the features and the
     candidates are those of the transfer check. The regenerated candidates equal the saved lists exactly, and the
     original rule reproduces the transfer check's counts exactly.
2. **Where the new thresholds came from.** They came from the calibration part of the original training library
   only. Each is the highest threshold that keeps at least 95% of that part's correct A4C/A5C candidates.
   - **Few candidates.** That part holds only 13, 14 and 11 correct A4C/A5C candidates (seeds 0, 1, 2), from 17, 17
     and 14 sessions. Below 20 candidates, keeping at least 95% means keeping all of them.
   - **Very low thresholds.** So each threshold equals the single lowest score among those candidates. Q0's are
     0.0011, 0.0044 and 0.0005, and Q1's are 0.0019, 0.0270 and 0.0049. The original thresholds are 0.57–0.85. The
     second-lowest scores are 1.7–19.5 times higher than the lowest.
   - **No guarantee.** Keeping 13 of 13 bounds the true keep rate from below only at 0.79 (one-sided 95%). For 14 of
     14 and 11 of 11 the bounds are 0.81 and 0.76. These bounds assume independent candidates. The candidates come
     from 11, 12 and 9 sequences, and 6, 6 and 2 of them share a recording with another, so the bounds are likely too
     high.
3. **The cost already shows on the library.** On the calibration part, keeping all correct A4C/A5C candidates leaves
   only 4–8 of the 18–22 wrong ones removed. The original thresholds remove 17–22. On the library's held-out part,
   Q0's pair thresholds remove only 9 of 40, 11 of 38 and 3 of 29 wrong ones.
4. **The A4C/A5C pair on the validation halves.** Both models now keep all 13 correct middles in every seed. Q0 misses
   most wrong ones in every seed, and Q1 does so in seeds 0 and 1.
   - **Q0** removes 1 of 14, 2 of 18 and 5 of 22 wrong A4C/A5C middles. Under the original rule it removed 13, 16 and
     21.
   - **Q1** removes 1 of 14, 8 of 18 and 12 of 22. Under the original rule it removed 14, 18 and 22.
   - **Against the original rule.** Q0 rescues 5, 4 and 4 correct middles and lets through 12, 14 and 16 wrong ones.
     Q1 rescues 6, 6 and 5 and lets through 13, 10 and 10. No correct middle is lost and no wrong middle is newly
     removed, because every pair threshold is lower than the original one.
5. **Overall.** The other pairs keep their thresholds, so their counts do not change. Over all pairs, Q0 now keeps
   100% of correct middles and removes 26.1–30.8% of wrong ones, against 96.2–96.9% and 87.0–92.3% under the
   original rule. Q1 keeps 98.4–100% and removes 21.1–57.7%.
6. **Candidates outside the binary task.** In the A4C/A5C pair there are 11, 10 and 13 of them. The pair rule keeps
   10, 4 and 8 of them that Q0's original rule removed, and 9, 7 and 10 for Q1. They are only counted. This check does
   not treat any of them as safe to remove.
7. **Q1 against Q0, both under the pair rule.** No consistent difference was detected.
   - **A4C/A5C removal.** Q1 − Q0 is 0.0, +33.3 and +31.8 points. The intervals are 0.0 to 0.0, +12.5 to +56.3 and
     +9.1 to +56.0. So Q1 removes more in seeds 1 and 2 and the same in seed 0.
   - **Where the extra removals are.** They are on the selection half (7 against 1 of 13 in seed 1, 12 against 4 of
     16 in seed 2). On the calibration half Q1 removes 1 of 5 and 0 of 6, as many as Q0 or fewer.
   - **Two limits.** Each model's pair threshold rests on its own single lowest-scoring calibration candidate. On the
     library's held-out part, the two models remove about as many wrong A4C/A5C candidates (9 and 9 of 40, 11 and 11
     of 38, 3 and 5 of 29).

**Decision.** Under the fixed rule, both models fall in the case "keeps correct, lets wrong through". Q0 misses 13 of
14, 16 of 18 and 17 of 22 wrong A4C/A5C middles (77–93%). Q1 misses 13 of 14 and 10 of 18 in seeds 0 and 1. In seed
2 it removes 12 of 22 (55%) and meets the standard, but only in that seed. No model meets the standard in most seeds.
Under the plan, the threshold check of these two discriminators therefore ends, and no further thresholds are tried.
With these pair thresholds, keeping the correct middles cost Q0 76–92% of its removals of wrong A4C/A5C middles, and
Q1 45–93%.

**What this check does not decide.** It does not separate "the scores cannot separate the pair" from "the threshold
landed badly", the distinction the plan asked for.
- Each pair threshold equals the single lowest of 11–14 library calibration scores, as answer 2 shows.
- Two hindsight diagnostics count the wrong A4C/A5C candidates that score below the lowest-scoring correct one. They
  read the labels and set no threshold, so they are not results.
  - On the library's held-out part, this is 34 of 40, 29 of 38 and 20 of 29 for Q0, but only 14 of 40, 11 of 38 and
    9 of 29 for Q1.
  - On the validation halves, it is 9 of 14, 6 of 18 and 13 of 22 for Q0, and 11 of 14, 15 of 18 and 18 of 22 for Q1.
- The lowest correct score itself varies widely between sets of 10–14 candidates. For Q0 it is 0.0005–0.0044 on the
  calibration part, 0.034–0.055 on the held-out part and 0.016–0.034 on the validation halves. For Q1 it is
  0.0019–0.027, 0.0078–0.028 and 0.059–0.088.
- So the overlap differs by model and by set. Whether a threshold fixed in advance could keep the correct middles and
  still remove most wrong ones is not shown. The plan rules out choosing one on the validation halves.

**Which parts of the rule are mine.** The standard (A4C/A5C keep ≥ 0.90 and removal ≥ 0.50) is the plan's, carried
over from the transfer check. The case names, the aggregation (both conditions in the same seed, in at least two of
three seeds) and the mapping of the plan's "almost all let through" branch to removal below 0.50 are mine, fixed
before the check. That mapping is what places Q1 in this case. Its removal of 7% in seed 0 and 44% in seed 1 counts as
letting wrong middles through, while seed 2 (55%) meets the standard. Every count is reported so the results can be
read differently.

## Terms

- A candidate's **predicted pair** is its decoded classes X and Y in X → Y → X, read without order.
- A **correct middle (a)** matches a true reference unit of its class with IoU of at least 0.5, and both outer runs'
  reference majority is the outer class.
- In a **wrong middle (b)**, every reference label inside the middle is the outer class, and both outer runs'
  reference majority is the outer class.
- Every other candidate lies **outside the binary task (c)**. These candidates are scored, but no rate includes them.
- A candidate is **kept** when its score is at or above the threshold that applies. A wrong middle that is kept is
  **missed**.
- A correct middle is **rescued** when the original rule removes it and the pair rule keeps it.
- A wrong middle is **let through** when the original rule removes it and the pair rule keeps it.
- A **case** is one of the 13 true A4C/A5C units behind the correct middles. The same 13 units appear in every seed.

## 1. The thresholds (library calibration part)

| Seed | Model | Original threshold | Pair threshold | Second-lowest score | Correct A4C/A5C kept (sessions) | One-sided 95% lower bound | Wrong A4C/A5C removed, original → pair |
|---|---|---|---|---|---|---|---|
| 0 | Q0 | 0.682 | 0.0011 | 0.0023 | 13 of 13 (17) | 0.79 | 17/18 → 4/18 |
| 0 | Q1 | 0.827 | 0.0019 | 0.0057 | 13 of 13 (17) | 0.79 | 17/18 → 6/18 |
| 1 | Q0 | 0.568 | 0.0044 | 0.0166 | 14 of 14 (17) | 0.81 | 22/22 → 8/22 |
| 1 | Q1 | 0.587 | 0.0270 | 0.0552 | 14 of 14 (17) | 0.81 | 21/22 → 8/22 |
| 2 | Q0 | 0.628 | 0.0005 | 0.0096 | 11 of 11 (14) | 0.76 | 20/20 → 5/20 |
| 2 | Q1 | 0.851 | 0.0049 | 0.0082 | 11 of 11 (14) | 0.76 | 20/20 → 8/20 |

- At the original thresholds, this part keeps only 7 of 13, 7 of 14 and 6 of 11 correct A4C/A5C candidates, for both
  models.
- The wrong A4C/A5C candidates of this part come from 10, 11 and 13 sessions.
- The lower bounds assume independent candidates and are likely too high (answer 2).

**The library's held-out part (supplementary, not part of the decision).**

| Seed | Model | Correct A4C/A5C kept, original → pair | Wrong A4C/A5C removed, original → pair |
|---|---|---|---|
| 0 | Q0 | 8/11 → 11/11 | 40/40 → 9/40 |
| 0 | Q1 | 9/11 → 11/11 | 36/40 → 9/40 |
| 1 | Q0 | 8/10 → 10/10 | 37/38 → 11/38 |
| 1 | Q1 | 8/10 → 10/10 | 37/38 → 11/38 |
| 2 | Q0 | 8/11 → 11/11 | 27/29 → 3/29 |
| 2 | Q1 | 8/11 → 11/11 | 27/29 → 5/29 |

## 2. The A4C/A5C pair on the validation halves

Each cell gives correct middles kept and wrong middles removed. The recordings and sessions are distinct counts and do
not depend on the rule.

| Seed | Half | Correct: recordings / sessions | Wrong: recordings / sessions | Q0, original | Q0, pair | Q1, original | Q1, pair |
|---|---|---|---|---|---|---|---|
| 0 | selection | 9 / 9 | 7 / 6 | 3/6, 9/9 | 6/6, 1/9 | 2/6, 9/9 | 6/6, 1/9 |
| 0 | calibration | 9 / 7 | 4 / 4 | 5/7, 4/5 | 7/7, 0/5 | 5/7, 5/5 | 7/7, 0/5 |
| 0 | both | 18 / 16 | 11 / 10 | 8/13, 13/14 | 13/13, 1/14 | 7/13, 14/14 | 13/13, 1/14 |
| 1 | selection | 9 / 9 | 7 / 6 | 3/6, 12/13 | 6/6, 1/13 | 2/6, 13/13 | 6/6, 7/13 |
| 1 | calibration | 9 / 7 | 5 / 5 | 6/7, 4/5 | 7/7, 1/5 | 5/7, 5/5 | 7/7, 1/5 |
| 1 | both | 18 / 16 | 12 / 11 | 9/13, 16/18 | 13/13, 2/18 | 7/13, 18/18 | 13/13, 8/18 |
| 2 | selection | 9 / 9 | 11 / 9 | 3/6, 16/16 | 6/6, 4/16 | 3/6, 16/16 | 6/6, 12/16 |
| 2 | calibration | 9 / 7 | 5 / 5 | 6/7, 5/6 | 7/7, 1/6 | 5/7, 6/6 | 7/7, 0/6 |
| 2 | both | 18 / 16 | 16 / 14 | 9/13, 21/22 | 13/13, 5/22 | 8/13, 22/22 | 13/13, 12/22 |

The recording counts include every recording under a middle, including neighbouring fragments that a decoded middle
runs into (see the transfer check).

## 3. What changes against the original rule

Each cell gives the count, then its distinct recordings and sessions.

| Seed | Model | Half | Correct middles rescued | Wrong middles let through | Outside candidates now kept |
|---|---|---|---|---|---|
| 0 | Q0 | selection | 3 (6, 6) | 8 (6, 5) | 6 (9, 8) |
| 0 | Q0 | calibration | 2 (3, 2) | 4 (4, 4) | 4 (5, 5) |
| 0 | Q0 | both | 5 (9, 8) | 12 (10, 9) | 10 (14, 13) |
| 0 | Q1 | selection | 4 (7, 7) | 8 (6, 5) | 5 (8, 7) |
| 0 | Q1 | calibration | 2 (2, 2) | 5 (4, 4) | 4 (5, 5) |
| 0 | Q1 | both | 6 (9, 9) | 13 (10, 9) | 9 (13, 12) |
| 1 | Q0 | selection | 3 (6, 6) | 11 (4, 4) | 3 (6, 5) |
| 1 | Q0 | calibration | 1 (2, 2) | 3 (4, 4) | 1 (1, 1) |
| 1 | Q0 | both | 4 (8, 8) | 14 (8, 8) | 4 (7, 6) |
| 1 | Q1 | selection | 4 (7, 7) | 6 (6, 5) | 4 (7, 6) |
| 1 | Q1 | calibration | 2 (2, 2) | 4 (5, 5) | 3 (4, 4) |
| 1 | Q1 | both | 6 (9, 9) | 10 (11, 10) | 7 (11, 10) |
| 2 | Q0 | selection | 3 (6, 6) | 12 (8, 7) | 4 (7, 6) |
| 2 | Q0 | calibration | 1 (2, 2) | 4 (4, 4) | 4 (4, 4) |
| 2 | Q0 | both | 4 (8, 8) | 16 (12, 11) | 8 (11, 10) |
| 2 | Q1 | selection | 3 (5, 5) | 4 (3, 3) | 6 (9, 7) |
| 2 | Q1 | calibration | 2 (2, 2) | 6 (5, 5) | 4 (4, 4) |
| 2 | Q1 | both | 5 (7, 7) | 10 (8, 8) | 10 (13, 11) |

**Which correct middles were rescued.** The cases are numbered without ids. Cases 1–7 are on the calibration half and
cases 8–13 on the selection half.
- **Q0** rescues cases 4, 9, 10 and 13 in every seed, and case 1 in seed 0. Cases 9, 10 and 13 are the three
  selection-half middles that Q0 removed in every seed of the transfer check.
- **Q1** rescues cases 1, 7, 9, 10 and 11 in every seed, and case 13 in seeds 0 and 1.
- Cases 2, 3, 5, 6, 8 and 12 are kept by both models under both rules in every seed.

**Which wrong middles were let through (both halves).**

| Seed | Model | Let through | Native clips | Pair streams | Multi-fragment streams | No join | Verified same-view join | View-change join |
|---|---|---|---|---|---|---|---|---|
| 0 | Q0 | 12 | 5 | 1 | 6 | 8 | 3 | 1 |
| 0 | Q1 | 13 | 6 | 1 | 6 | 9 | 3 | 1 |
| 1 | Q0 | 14 | 5 | 2 | 7 | 9 | 4 | 1 |
| 1 | Q1 | 10 | 3 | 1 | 6 | 6 | 4 | 0 |
| 2 | Q0 | 16 | 8 | 2 | 6 | 12 | 4 | 0 |
| 2 | Q1 | 10 | 6 | 0 | 4 | 7 | 3 | 0 |

`report_tables.md` lists every changed candidate with its half, direction, source, join, length, score and case
number.

## 4. Q1 against Q0, both under the pair rule (both halves)

| Seed | A4C/A5C removal, Q1 − Q0 (points) | A4C/A5C keep | Overall removal | Overall keep |
|---|---|---|---|---|
| 0 | 0.0 [0.0, 0.0] | 0.0 | −5.3 [−16.0, 0.0] | −0.8 [−2.5, 0.0] |
| 1 | +33.3 [+12.5, +56.3] | 0.0 | +26.1 [+9.4, +46.2] | 0.0 [0.0, 0.0] |
| 2 | +31.8 [+9.1, +56.0] | 0.0 | +26.9 [+6.2, +50.0] | −1.6 [−4.2, 0.0] |

The intervals resample streams for stream candidates and sessions for native candidates (2,000 draws). They are likely
too narrow, because streams reuse recordings that are also native clips, and the thresholds are held fixed. The
comparison only uses both models under the same rule, so the change of rule is not attributed to the visual features.

## 5. Decision under the fixed rule

Per seed the two halves are pooled, and "most seeds" means at least two of three.

| Model | Seed 0: A4C/A5C keep, removal | Seed 1 | Seed 2 | Case |
|---|---|---|---|---|
| Q0, pair rule | 1.00, 0.07 | 1.00, 0.11 | 1.00, 0.23 | keeps correct, lets wrong through |
| Q1, pair rule | 1.00, 0.07 | 1.00, 0.44 | 1.00, 0.55 | keeps correct, lets wrong through |
| Q0, original rule | 0.62, 0.93 | 0.69, 0.89 | 0.69, 0.95 | keep not reached |
| Q1, original rule | 0.54, 1.00 | 0.54, 1.00 | 0.62, 1.00 | keep not reached |

**Outcome.** No model meets the standard under the pair rule. The threshold check of these two discriminators ends,
and no further thresholds are tried. Nothing is integrated, and this check makes no claim about segment F1 or the
baselines.

## 6. What this does not show

- **Independent evidence.** The rule was set after the problem was seen on these halves, which have been used
  repeatedly in development.
- **A guaranteed keep rate.** Keeping 13 of 13 correct middles on the validation halves comes from the same 13 cases
  in every seed. The library thresholds rest on 11–14 candidates each.
- **Whether the scores or the threshold placement limit the trade-off.** Only one pre-fixed threshold per seed and
  model was tried, as the plan requires, and the hindsight diagnostics above are not results. Choosing a threshold on
  the validation halves is not allowed.
- **A full-system effect.** No candidate was removed or merged, so segment F1 was not measured.

## 7. Cost

- **Thresholds.** Scoring the library and setting the six thresholds took 0.3 seconds of computation on the CPU.
- **Application.** Both halves for three seeds took 50 seconds on the CPU.

## Files

- `thresholds.json` holds, per seed and model, both thresholds and the library counts behind them, including the
  held-out part. It was committed before the application.
- `summary.json` holds, per seed and half, every count under both rules, the transitions, the Q1 − Q0 bootstrap, the
  numbered cases, the changed candidates and the decision.
- `diagnostics.json` holds the hindsight overlap diagnostics and the spread of the lowest calibration scores. It was
  added after the fact-check by `--diagnose`, which rescores nothing.
- `report_tables.md` holds the generated tables.
- None of these files contains recording or session ids. The candidate lists and the per-candidate scores stay on the
  server, in `runs/short_segment_transfer_v1/` and `runs/short_segment_pair_threshold_v1/`.
