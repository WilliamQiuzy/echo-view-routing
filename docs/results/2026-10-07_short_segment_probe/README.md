# Short-segment probe: can frozen visual features tell a true short view segment from a wrong one? (2026-10-07)

This probe asked a narrow question. A decoder often produces a pattern X → Y → X, for example A4C → A5C → A4C. The
middle run is either a true short view segment or a wrong prediction inside one view. Do frozen visual features help
decide which, beyond the statistics that are already available?

The probe used training recordings only. Both validation halves and the test split were not read. Nothing was
integrated into the full system, no encoder or ViT was trained, and the public labels are unchanged. The plan is
`docs/superpowers/plans/2026-10-07-short-segment-probe.md`.
- **Order of commits.** The protocol was committed before the candidate library was built (16a020a). Fixes from an
  independent review and the library composition were committed before any discriminator was trained (76a8d23).
- **What this cannot show.** The frozen L1 models were trained on these same recordings. The probe therefore measures
  how a new discriminator does on held-out training sessions, not how it would do on unseen patients.
- Adding a segment-level discriminator is not new in itself; HASR (ICCV 2021) refines segments in a similar way.

## Answers

1. **What was done.**
   - The 3,364 training clips were split by session: train 301 sessions, calibration 100 and held-out 101.
   - New constructed streams were built once inside each part (900, 300 and 300) and shared by the three seeds, so no
     stream crosses parts. Each join keeps the view with probability 0.35, so about a third of the joins connect two
     recordings of the same view. They give wrong middles a chance to sit at a recording join (the splice control).
   - One fixed generator, J0 with κ = 8 on the frozen L1 probabilities, produced the X → Y → X candidates. Each seed
     used its own L1 model on the same streams.
   - Each candidate was labelled (a), (b) or (c) from the reference labels:
     - **(a) correct middle:** it matches a true unit of class Y with IoU of at least 0.5, and the reference majority
       of both outer runs is X;
     - **(b) wrong middle:** every reference label inside it is X, and the reference majority of both outer runs is X;
     - **(c) other:** everything else, including outer runs that are not X; kept out of the binary task.
   - Two logistic-regression discriminators were compared.
     - **Q0** reads lengths, L1 probabilities and B3 boundary scores.
     - **Q1** adds the mean frozen L1 features of the middle and both sides.
     - A diagnostic, Q0+sim, adds only two feature similarities to Q0.
2. **Were there enough independent examples?** Yes, by the fixed rule, but not for every question.
   - **Held-out part.** It holds 158–160 correct middles from 72–73 sessions and 40–50 wrong middles from 20–23
     sessions per seed.
   - **The A4C/A5C case.** It has only 10–11 held-out correct middles per seed.
   - **The splice control.** Only 7–10 held-out wrong middles per seed sit within 5 samples of a recording join.
3. **Do the two kinds differ in simple statistics?** Strongly, in this library.
   - **Length.** In the held-out part, correct middles have a median length of 24 samples and wrong ones 13–14.
   - **Confidence.** Correct middles are read with a median L1 confidence of 0.998, wrong ones 0.76–0.80.
   - **Source.** Every correct middle comes from a constructed stream and sits at a recording join, because a true
     view change in a constructed stream is always a join of two recordings. Of the held-out wrong middles, 65–68%
     come from native clips and only 7–10 sit at a join.
4. **Does Q1 beat Q0?** No. But the fixed rule could not have found a gain here, whatever Q1 did.
   - **Q0 is at the ceiling.** On the held-out sessions its AUC is 0.997–0.999. At the operating point that keeps 95%
     of correct middles on calibration, it keeps 97.5–98.1% of held-out correct middles and removes 100%, 97.8% and
     95.0% of wrong ones.
   - **Why the rule could not pass.** The rule needs every seed's interval for Q1 − Q0 to lie above zero.
     - In seed 0, Q0 already removes all 50 wrong middles, so Q1 − Q0 cannot be positive.
     - In seeds 1 and 2, Q0 leaves only 1 and 2 unremoved. A bootstrap draw that leaves out their groups gives a
       difference of zero or less, which happens far more often than the 2.5% the lower bound allows.
   - **What speaks against Q1.** Q1's AUC is lower in every seed (0.988–0.993 against 0.997–0.999). In seed 0 it
     removes 10 points fewer wrong middles at the same keep rate (−19.5 to −2.4, resting on 5 candidates).
     - In seed 1 both remove 45 of 46 wrong middles, and Q1 keeps one correct middle fewer (154 against 155 of 158).
     - In seed 2 the totals are equal, but Q1 misses one more wrong middle at a join and removes one more inside a
       clip.
   - **The diagnostic Q0+sim** matches Q0 in removal. Its keep rate differs by at most 0.6 points at the 95% point and
     1.3 points at the 90% point.
5. **Could a discriminator just detect splices?** The control hints at this for Q1, but on three candidates.
   - At a join, Q1 removes 8 of 10 and 6 of 7 wrong middles in seeds 0 and 2, where Q0 removes all.
   - In seed 0, Q1 also misses 3 wrong middles inside clips, so its extra misses are not all at joins.
6. **The A4C/A5C pairs.** The plan used this pair as its example, and the probe says little about it.
   - Q0 keeps only 8 of 11, 8 of 10 and 8 of 11 held-out correct A4C/A5C middles at the overall 95% point. Q1 keeps 9,
     8 and 8.
   - Of the 40, 38 and 29 wrong A4C/A5C middles, Q0 removes 100%, 97.4% and 93.1% and Q1 removes 90%, 97.4% and
     93.1%.
   - The counts are too small to conclude anything about this pair.

Under the decision fixed before training, the visual features show no extra discriminative value, so there is no
basis for a segment-level module and nothing is integrated into the full system. This outcome rests mainly on Q0's
ceiling; the direct evidence against an added value of Q1 is its lower AUC in every seed and its seed-0 deficit.

**Why this does not settle the visual question.** Two features of this library may explain Q0's ceiling, and the probe
cannot tell them apart.
- **Training exposure.** The L1 models were trained on these recordings, and they read correct fragments with
  near-certain confidence.
- **Construction.** Every correct middle comes from a constructed stream at a recording join, where the boundary
  scores and the side probabilities change abruptly. Stream fragments are also contrast-edited with probability 0.5,
  while native clips never are, and most wrong middles come from native clips.

The probe therefore cannot show whether visual features would separate the two kinds on unseen recordings with real
view changes. Nor does it show that the simple statistics would.

## Terms

- **Candidate:** three consecutive decoded runs whose outer runs share a class X that the middle's class Y differs
  from.
- **Splice:** a join between two recordings within 5 samples of the middle. Native clips have none.
- **Keep rate:** the share of correct middles (a) that a discriminator keeps.
- **Removal rate:** the share of wrong middles (b) that it removes.
- **Operating point:** the threshold set on the calibration part so that at least 95% (primary) or 90% of correct
  middles are kept. It is then applied unchanged to the held-out part.
- **Points** are hundredths of a rate. Intervals are 95% bootstrap intervals with 2,000 draws (seed 20261002),
  resampling streams for stream candidates and sessions for native candidates.

## 1. The candidate library

**Partition.**

| Part | Sessions | Clips (PLAX, PSAX, A4C, A5C, SC4C) | Constructed streams |
|---|---|---|---|
| Train | 301 | 678, 559, 526, 131, 116 | 900 |
| Calibration | 100 | 220, 178, 156, 42, 33 | 300 |
| Held-out | 101 | 222, 237, 176, 44, 46 | 300 |

**Candidates in the held-out part.**

| Seed | Category | Candidates | Recordings | Sessions | From streams / natives | At a splice | Median length | Median confidence | A4C/A5C pairs |
|---|---|---|---|---|---|---|---|---|---|
| 0 | (a) correct | 159 | 163 | 73 | 159 / 0 | 159 | 24 | 0.998 | 11 |
| 0 | (b) wrong | 50 | 30 | 23 | 17 / 33 | 10 | 14 | 0.777 | 40 |
| 0 | (c) other | 35 | 37 | 29 | 23 / 12 | 22 | 24 | 0.847 | 23 |
| 1 | (a) correct | 158 | 161 | 72 | 158 / 0 | 158 | 24 | 0.998 | 10 |
| 1 | (b) wrong | 46 | 28 | 22 | 16 / 30 | 9 | 14 | 0.800 | 38 |
| 1 | (c) other | 32 | 32 | 27 | 20 / 12 | 19 | 23.5 | 0.863 | 19 |
| 2 | (a) correct | 160 | 164 | 73 | 160 / 0 | 160 | 24 | 0.998 | 11 |
| 2 | (b) wrong | 40 | 26 | 20 | 13 / 27 | 7 | 13 | 0.757 | 29 |
| 2 | (c) other | 27 | 23 | 20 | 17 / 10 | 15 | 22 | 0.814 | 15 |

- **Train and calibration parts.** The train part holds 449–453 correct and 110–130 wrong candidates per seed, the
  calibration part 156–159 correct and 25–28 wrong. `composition.json` has every count, the length quartiles and the
  class pairs.
- **Correct middles** come only from constructed streams. A native clip has one view, so it cannot contain a true
  second view.
- **Wrong middles.**
  - Their recording count is smaller than their candidate count. A recording can carry several wrong runs, or appear
    as a native clip and in several streams; the files do not separate the two.
  - About two-thirds come from native clips and the rest from streams.
  - Most are A4C/A5C confusions.
- **Lengths.** 35, 30 and 23 held-out wrong middles fall in the 10–40 sample range where most correct middles lie.

## 2. Discriminators and choice

- **Model.** Standardised features, L2 logistic regression with balanced class weights, and C chosen by 5-fold
  stratified grouped cross-validation on the train part.
  - Q0 chose C = 10 in every seed, the largest value of the grid.
  - Q1 chose 1, 0.1 and 1.
- **Q0 features (33).**
  - The log lengths of the middle and both outer runs.
  - The two classes.
  - The mean L1 probabilities of the middle and of up to 16 samples on each side.
  - The middle's mean p(Y), p(Y) − p(X) and log-ratio.
  - The B3 boundary scores at the middle's two edges.
- **What Q0 leaves out.** ASRF outputs were excluded, because the frozen ASRF was trained on the training streams'
  boundaries and would leak the labels.
- **Q1 features (1,571).** Q0, plus the mean frozen 512-d L1 features of the middle and both side windows and their
  two cosine similarities.
- **Q0+sim** is Q0 plus the two similarities only. It is a diagnostic for discontinuity and plays no part in the
  decision.

## 3. Held-out results at the 95% operating point

| Seed | Model | AUC | Keep (a) | Remove (b) | Remove (b) at a splice | Remove (b) inside a clip | Remove (b), 10–40 samples | Keep (a), A4C/A5C |
|---|---|---|---|---|---|---|---|---|
| 0 | Q0 | 0.999 | 0.975 | 1.000 | 10 / 10 | 40 / 40 | 1.000 | 8 / 11 |
| 0 | Q1 | 0.988 | 0.975 | 0.900 | 8 / 10 | 37 / 40 | 0.914 | 9 / 11 |
| 0 | Q0+sim | 0.999 | 0.975 | 1.000 | 10 / 10 | 40 / 40 | 1.000 | 8 / 11 |
| 1 | Q0 | 0.998 | 0.981 | 0.978 | 9 / 9 | 36 / 37 | 1.000 | 8 / 10 |
| 1 | Q1 | 0.993 | 0.975 | 0.978 | 9 / 9 | 36 / 37 | 1.000 | 8 / 10 |
| 1 | Q0+sim | 0.998 | 0.987 | 0.978 | 9 / 9 | 36 / 37 | 1.000 | 9 / 10 |
| 2 | Q0 | 0.997 | 0.975 | 0.950 | 7 / 7 | 31 / 33 | 0.957 | 8 / 11 |
| 2 | Q1 | 0.990 | 0.969 | 0.950 | 6 / 7 | 32 / 33 | 0.957 | 8 / 11 |
| 2 | Q0+sim | 0.997 | 0.981 | 0.950 | 7 / 7 | 31 / 33 | 0.957 | 9 / 11 |

**Q1 − Q0 (points, held-out).**

| Seed | Removal of (b) at 95% | Keep of (a) at 95% | Removal of (b) at 90% | Keep of (a) at 90% |
|---|---|---|---|---|
| 0 | −10.0 [−19.5, −2.4] | 0.0 [−1.9, +1.9] | −2.0 [−6.7, 0.0] | −1.3 [−4.8, +2.0] |
| 1 | 0.0 [0.0, 0.0] | −0.6 [−3.0, +1.3] | −2.2 [−7.7, 0.0] | −0.6 [−3.7, +1.9] |
| 2 | 0.0 [−7.3, +7.0] | −0.6 [−2.9, +1.3] | 0.0 [0.0, 0.0] | −3.8 [−7.4, −0.6] |

- **At the 90% point,** Q0 removes every held-out wrong middle in all seeds and keeps 94.3–95.6% of correct ones.
- **The A4C/A5C pairs.** The correct A4C/A5C middles are the ones Q0 most often discards, 3, 2 and 3 of 10–11. Q1
  keeps one more in seed 0 and the same number in the other seeds.

## 4. Decision under the fixed rule

| Condition (all three seeds) | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Q1 − Q0 removal interval above zero at 95% | no | no | no |
| Q1 keep rate at most 2 points below Q0 | yes | yes | yes |
| Q1 − Q0 at a splice not below zero | no | yes | no |

The visual features show no extra discriminative value. Under the plan, there is no basis for a segment-level module,
and nothing is integrated into the full system.

The first condition could not be met in this library, because Q0 left 0, 1 and 2 wrong middles unremoved. The outcome
is therefore a ceiling result; the direct evidence against Q1 is its lower AUC in every seed and its seed-0 deficit.

## 5. What this does not show

- **That visual features cannot help on unseen recordings.**
  - Here the statistics alone nearly separate the two kinds (Q0's AUC is 0.997–0.999), which leaves Q1 almost no
    room.
  - That separation may come from the L1 models' training on these recordings, from the way the library is
    constructed (section 1), or from both. The probe has no condition that tells them apart.
  - On validation, the earlier rounds showed that the same models lose correct short segments and keep wrong ones.
    This probe does not test that setting.
- **That simple statistics would work on unseen recordings.** Q0's near-perfect separation may rest on the same
  exposure and construction.
- **Anything firm about A4C/A5C.** The held-out part has 10–11 correct A4C/A5C middles per seed.
- **That splices explain the task.** By construction, every correct middle is at a recording join and most wrong ones
  are not. The control has only 7–10 wrong middles per seed within 5 samples of a join; the flag does not check that
  the join is between same-view recordings. It hints that Q1 is drawn to joins, on three candidates, but it cannot
  measure how much.
- **Precise intervals.** The bootstrap treats streams and native sessions as independent units, but streams reuse
  recordings and every recording is also a native clip. The thresholds are held fixed, so the calibration uncertainty
  is left out. The intervals are therefore likely too narrow, and the seed-0 interval rests on 5 candidates.
- **Real view changes.** EV9V has no recordings with a view change inside one acquisition. Every true middle here is a
  join of two recordings, so the probe never sees a true short segment without a splice.

## 6. Cost

- **Library.** Building the partition, the 1,500 constructed streams (shared by the seeds) and the three per-seed
  libraries took about 1.4 minutes on the CPU.
- **Training and evaluation.** The three models per seed with cross-validation and bootstrap took about 2.5 minutes on
  the CPU.

## Files

- `composition.json` holds the partition counts and, for every seed, part and category, the candidate counts,
  recordings, sessions, lengths, class pairs, confidence and splice flags. It was committed before any discriminator
  was trained.
- `summary.json` holds every model's C choice, cross-validation scores, thresholds, held-out rates by subgroup, the
  contrasts, the sufficiency checks and the decision.
- `report_tables.md` holds the generated tables.
- None of these files contains recording or session ids. The partition, the constructed streams and the libraries stay
  on the server, in `runs/short_segment_probe_v1/`.
