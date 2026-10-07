# Joint boundary decoding: J0 and J1 on frozen L1 and ASRF outputs (2026-10-07)

This round asked whether ASRF's boundary evidence, used inside one joint decode, reduces how much correct visual
evidence the full system loses. Nothing was trained. Two decoders read only frozen per-sample outputs:

| Decoder | What decides the label sequence |
|---|---|
| J0 (fixed switch cost) | The L1 class probabilities and a fixed cost κ for every change of label |
| J1 (boundary-conditioned switch cost) | The same probabilities, plus ASRF's boundary probability: a change is cheaper where the boundary evidence is strong and dearer where it is weak |

Both find the single best label path for the whole sequence. Neither thresholds the boundary probabilities into
intervals or votes inside intervals. The objective is

  Σ_t log p_t(y_t) + λ Σ_{t≥1} [c_t log b_t + (1 − c_t) log(1 − b_t)] − κ Σ_{t≥1} c_t,

where p_t is the L1 class probability, b_t the ASRF boundary probability and c_t = 1 when the label changes at t.
J0 is λ = 0.

The plan is `docs/superpowers/plans/2026-10-07-joint-boundary-decoding.md`. This round checks another way of combining
existing outputs; it is not a new model. The configuration, decoder, code, tests and a pre-metric extraction record
were committed before any metric existed (45848e9). The parameters were chosen on the calibration half alone and
committed (c9e5022) before J0 or J1 decoded any selection sequence. Before that, the extraction step had run the
frozen ASRF on both halves to store boundary probabilities and compare them with stored predictions. Its only
label-based check, the alignment profile, used calibration and training streams, and it computed no decoder metric.
The test split and EchoNet-Pediatric were not read, and the public labels are unchanged.

## Answers

1. **What was done.**
   - **Inputs.** For each seed, p_t comes from the frozen L1_s{k} probabilities that L1 + B3 decoded in the fan224
     round, and b_t from the frozen B_s{k} + ASRF of the L1 + ASRF round (section 1).
   - **Choice.** For each seed, a 5 x 5 grid of λ and κ was decoded on the calibration half. J0 took the best feasible
     setting with λ = 0, and J1 the best feasible setting of the whole grid (section 2).
   - **Evaluation.** The six chosen decoders were applied once to both halves and compared with L1 + B3, B + ASRF and
     the references. The fixed diagnostics answer the three questions below.
2. **The three native clips that ASRF changed to wrong.** J1 reads 3, 2 and 2 of them correctly (seeds 0, 1, 2).
   - Against B + ASRF it newly misreads 2, 0 and 1 other selection clips, each from its own session.
   - Against L1 + B3, which reads all three, J1 still misreads 2, 1 and 2 selection clips and reads none that L1 + B3
     misreads.
   - J0 reads the same 3, 2 and 2.
3. **Merged changes found again, and extra changes added.** Most merged changes come back, but extra changes rise too.
   J0, which never reads the boundary probabilities, does the same on the selection half.
   - **Recovered.** On the selection half, B + ASRF merged 14, 10 and 13 true view changes that had an ASRF candidate
     cut nearby. J1 finds 12, 7 and 8 of them, all with both views right. They come from 16, 10 and 12 source
     recordings.
   - **Extra changes.** J1 makes 37, 41 and 37 extra changes on the selection streams, against 21, 20 and 27 for
     B + ASRF. It also makes 31, 27 and 32 label switches inside native clips, against 10, 6 and 4.
   - **J0 against J1.** J0 recovers the identical sets of merged selection changes (12, 7 and 8) and makes 35, 41 and
     39 extra changes. On the calibration half the two differ only in seed 0, where J1 finds 8 and J0 finds 7.
4. **Does J1 beat J0, L1 + B3 and the configurations the goal was set from, and reach the goal?** No.
   - **J1 − J0.** No consistent difference was detected. In seed 1 the choice set λ = 0, so J1 is J0. Selection-half
     segment F1@50 changes by −0.1, 0.0 and +0.3 points.
   - **J1 − (L1 + B3).** No consistent difference was detected in segment F1@50 (−0.5, −1.2 and −1.1 points on the
     selection half; every interval contains zero).
   - **J1 − (E0 + B3).** J1 is below E0 + B3 on the selection half in every seed (−3.1, −4.0 and −3.3 points). The
     goal's selection value is E0 + B3's score + 0.01, and E0 + B3 is also the highest selection reference.
   - **J1 − (B7 + B3).** No consistent difference was detected against B7 + B3 on the calibration half. The goal's
     calibration value is B7 + B3's score + 0.01. E1 + ASRF s0 (0.976) and B + ASRF (mean 0.976) score higher there,
     and J1 − (B + ASRF) shows no consistent difference either.
   - **Goal.** J1 averages 0.899 segment F1@50 on the selection half and 0.969 on the calibration half; the goal
     requires 0.944 and 0.982. Its selection native macro-F1 is 0.970 and 0.968 in seeds 0 and 2, below the 0.973
     floor. Boundary recall passes in every seed and half (0.971–0.995).
5. **Where the trade-off lies.** On the selection half, J0 and J1 match 7–15 more reference units than B + ASRF or
   L1 + B3. They also lose less correct visual evidence. But they predict more segments, and their unmatched predicted
   segments rise from 40–57 to 64–72. Segment F1@50 shows no consistent improvement.

Under the plan, J1 is not stably better than J0 or L1 + B3, so this joint-decoding route ends. No further grid,
clipping rule, decoder variant or network was started.

## Terms

- **Points** are hundredths of a metric; −0.5 points of segment F1@50 means −0.005.
- **Native clips** are whole single-view recordings. A clip is read correctly when the majority of its samples is.
- **Cuts per clip** count additional label switches within clips assigned a single reference label.
- **Reference units** are stretches of one view between two true view changes in a constructed stream. A unit is
  matched when a predicted segment of its class overlaps it with IoU of at least 0.5.
- **True view changes** are the joins of a stream where the view changes. A change is **found** when a final change
  is matched to it within 5 samples. A missed change is called **merged** when an ASRF candidate cut lies within 5
  samples. For B + ASRF this means its class vote gave both sides of the candidate one class. For the other systems it
  only means that ASRF had proposed a cut there. An **extra change** is an unmatched final change more than 5 samples
  from every true change. The same ASRF candidates serve every system in this classification.
- **Consistent** means that all three seeds' 95% intervals exclude zero in the same direction on a half. An interval
  that contains zero does not mean the systems are equal.
- **Stable** is fixed before the round. It requires every seed's selection-half interval for segment F1@50 to lie
  above zero, and no seed's calibration-half interval to lie below zero.

## 1. Inputs and checks (extraction step)

- **Class probabilities.** p_t is the stored L1_s{k} softmax of every validation sequence: 541 selection and 472
  calibration sequences per seed. These are the arrays that L1 + B3 decoded. They differ from the L1 probabilities in
  the B_s{k} feature store by at most 1.6 x 10⁻⁴.
- **Boundary probabilities.** b_t is the sigmoid of the last boundary stage of the frozen B_s{k} + ASRF checkpoint,
  run on the B_s{k} features of the same sequence.
  - The official ASRF decode of the same outputs reproduces every stored B + ASRF prediction of both halves.
  - The candidate cuts used in the diagnostics equal the pinned library's `argrelmax` peaks on every sequence.
- **Alignment.** ASRF's boundary target marks t when the label at t differs from the label at t − 1, so b_t pairs
  with c_t at the same index.
  - Around true changes, the mean b_t peaks exactly at offset 0 in every seed. On calibration streams it is 0.97 at
    offset 0 against at most 0.203 one sample away.
  - On 300 training streams the peak is 0.98 at offset 0.
- **Decoder checks.**
  - Exhaustive search over all label paths equals the dynamic programme on short random sequences, for every
    setting of the grid.
  - With λ = 0 the output ignores b exactly and equals the B2 HMM Viterbi with the same switch cost on tie-free
    inputs.
  - A boundary spike moves the switch to its exact index.
- **Fixed rules.** p is clipped to [10⁻⁶, 1] and b to [10⁻⁶, 1 − 10⁻⁶], in float64. Ties favour staying over
  switching, then the lowest class index. The decoders read p and b only, never labels, joins or source-recording
  boundaries.
- **Reproduction at evaluation.** L1 + B3 and B + ASRF reproduce their earlier rows to within 10⁻¹² on five
  metrics. B + ASRF also reproduces its earlier change categories and extra-change counts. B + ASRF reads none of the
  three clips and L1 + B3 reads all three, as before.

## 2. Choice on the calibration half

The grid had λ in {0, 0.25, 0.5, 1, 2} and κ in {0, 1, 2, 4, 8}. A setting qualifies when calibration native macro-F1
is at least 0.980 and multi-stream boundary recall at least 0.95. The highest segment F1@50 wins, and ties go to the
lower λ, then the lower κ.

| Seed | J0 | J1 | Qualifying settings (J0 / J1) | Calibration segment F1@50 (J0 / J1) |
|---|---|---|---|---|
| 0 | κ = 8 | λ = 0.5, κ = 8 | 5 of 5 / 22 of 25 | 0.970 / 0.971 |
| 1 | κ = 8 | λ = 0, κ = 8 (equals J0) | 3 of 5 / 21 of 25 | 0.966 / 0.966 |
| 2 | κ = 8 | λ = 0.25, κ = 8 | 5 of 5 / 25 of 25 | 0.968 / 0.970 |

- **Ties.** In seed 1, λ = 0.25 tied λ = 0 exactly (0.965779), and the tie went to λ = 0. In seed 2, λ = 0.25, 0.5
  and 1 tied.
- **On calibration, the boundary term adds at most 0.2 points** over the best λ = 0 setting.
- **The switch cost sat at the edge of the grid.** Every choice used the largest κ, and the calibration score was
  still rising there in every seed. For example, seed 0's λ = 0 row goes 0.568, 0.729, 0.832, 0.928 and 0.970 for
  κ = 0, 1, 2, 4 and 8. The grid therefore did not bracket the best fixed switch cost. The plan does not allow
  extending it.

## 3. Results per system

Each cell gives the selection half / the calibration half.

| System | Segment F1@50 | Boundary P | Boundary R | Native macro-F1 | Cuts per clip | A5C right |
|---|---|---|---|---|---|---|
| J1 s0 | 0.903 / 0.971 | 0.903 / 0.954 | 0.974 / 0.992 | 0.970 / 0.993 | 0.109 / 0.076 | 11/11 / 18/18 |
| J1 s1 (= J0) | 0.894 / 0.966 | 0.894 / 0.952 | 0.974 / 0.990 | 0.978 / 0.990 | 0.095 / 0.076 | 10/11 / 18/18 |
| J1 s2 | 0.901 / 0.970 | 0.905 / 0.961 | 0.971 / 0.995 | 0.968 / 0.983 | 0.113 / 0.089 | 10/11 / 18/18 |
| J0 s0 | 0.904 / 0.970 | 0.905 / 0.949 | 0.974 / 0.987 | 0.970 / 0.993 | 0.106 / 0.076 | 11/11 / 18/18 |
| J0 s2 | 0.899 / 0.968 | 0.898 / 0.957 | 0.971 / 0.995 | 0.959 / 0.983 | 0.120 / 0.084 | 10/11 / 18/18 |
| L1 + B3 s0 | 0.908 / 0.971 | 0.961 / 0.973 | 0.961 / 0.977 | 0.989 / 0.986 | 0.095 / 0.101 | 11/11 / 17/18 |
| L1 + B3 s1 | 0.906 / 0.968 | 0.970 / 0.995 | 0.937 / 0.962 | 0.989 / 0.990 | 0.007 / 0.021 | 11/11 / 18/18 |
| L1 + B3 s2 | 0.912 / 0.969 | 0.961 / 0.968 | 0.961 / 0.982 | 0.989 / 0.983 | 0.092 / 0.131 | 11/11 / 18/18 |
| B + ASRF s0 | 0.902 / 0.978 | 0.945 / 0.980 | 0.940 / 0.972 | 0.957 / 0.993 | 0.035 / 0.025 | 9/11 / 18/18 |
| B + ASRF s1 | 0.907 / 0.978 | 0.938 / 0.980 | 0.948 / 0.982 | 0.957 / 0.993 | 0.021 / 0.042 | 9/11 / 18/18 |
| B + ASRF s2 | 0.896 / 0.973 | 0.928 / 0.975 | 0.950 / 0.975 | 0.957 / 0.983 | 0.014 / 0.034 | 9/11 / 17/18 |

**Means of segment F1@50.**
- J1: 0.899 / 0.969.
- J0: 0.899 / 0.968.
- L1 + B3: 0.909 / 0.969.
- B + ASRF: 0.902 / 0.976.

**References** (reproduced from their frozen outputs).

| System | Segment F1@50 |
|---|---|
| E0 + B3 | 0.934 / 0.934 |
| E1 + B3 | 0.894 / 0.959 |
| B7 + B3 | 0.880 / 0.972 |
| E1 + ASRF s0, s1, s2 (historical) | 0.910, 0.909, 0.907 / 0.976, 0.964, 0.964 |

## 4. Paired contrasts

Values are point differences with 95% paired-bootstrap intervals (2,000 draws, seed 20261002). Native clips are
resampled by session and streams by stream.

**Segment F1@50.**

| Contrast | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| J1 − J0, selection | −0.09 [−1.04, +0.81] | 0 (identical) | +0.26 [0.00, +0.60] |
| J1 − J0, calibration | +0.19 [−0.20, +0.69] | 0 (identical) | +0.18 [0.00, +0.46] |
| J1 − (L1 + B3), selection | −0.51 [−2.21, +1.39] | −1.21 [−3.69, +1.54] | −1.12 [−2.70, +0.60] |
| J1 − (L1 + B3), calibration | +0.04 [−0.94, +0.88] | −0.20 [−1.93, +1.48] | +0.02 [−1.19, +1.07] |
| J1 − (B + ASRF), selection | +0.11 [−1.85, +2.08] | −1.27 [−3.42, +0.80] | +0.48 [−1.50, +2.48] |
| J1 − (B + ASRF), calibration | −0.63 [−1.87, +0.59] | −1.20 [−2.69, +0.11] | −0.34 [−1.14, +0.40] |
| J1 − (E0 + B3), selection | −3.11 [−5.88, −0.12] | −4.04 [−7.05, −0.96] | −3.32 [−6.16, −0.29] |
| J1 − (B7 + B3), calibration | −0.07 [−1.38, +1.37] | −0.64 [−2.08, +0.87] | −0.26 [−1.49, +0.95] |

**Other metrics, selection half.**
- **Boundary recall.** J1 has higher recall than L1 + B3 (+1.31, +3.66 and +1.05 points) and than B + ASRF (+3.40,
  +2.62 and +2.09). All six intervals lie above zero.
- **Boundary F1.** Recall does not carry through to boundary F1. Against L1 + B3, J1's boundary F1 is −2.4, −2.1 and
  −2.4 points, and two of the three intervals lie below zero.
- **Cuts per native clip.** J1 cuts more than B + ASRF in every seed (+0.07, +0.07 and +0.10 per clip, intervals
  above zero).
- **Native macro-F1.** Against L1 + B3, J1 is −1.9, −1.1 and −2.0 points. The intervals reach zero only at their upper
  end.
- **Stream units.** J1 fixes 13, 22 and 9 selection units that L1 + B3 misses (11, 18 and 9 recordings) and newly
  misses 2, 7 and 2 (3, 7 and 3 recordings). Against B + ASRF it fixes 19, 12 and 16 (13, 11 and 13 recordings) and
  newly misses 4, 5 and 5 (6, 9 and 5 recordings).
- J0 − (L1 + B3) behaves like J1 − (L1 + B3). Its segment F1@50 is −0.42, −1.21 and −1.38 points on the selection
  half.

## 5. Question 1: the three native clips

These are the three selection clips that B + ASRF misread in every seed while L1 + B3 read them correctly. One is A4C,
two are A5C, and they come from three sessions.

| Seed | J1 reads correctly | J1 new errors against B + ASRF | J1 against L1 + B3 (fixed / new) |
|---|---|---|---|
| 0 | 3 of 3 | 2 clips (2 sessions) | 0 / 2 (2 sessions) |
| 1 | 2 of 3 | 0 | 0 / 1 (1 session) |
| 2 | 2 of 3 | 1 clip (1 session) | 0 / 2 (2 sessions) |

- J0 reads the same 3, 2 and 2 of the three clips.
- J1 misreads 4, 3 and 4 of the 284 selection natives. Two are the PLAX and SC4C clips that L1 + B3 and B + ASRF also
  misread. The others are A4C read as A5C (2, 0 and 1) and A5C read as A4C (0, 1 and 1).
- **Seed 1.** The frozen L1 head's own majority misreads one of the three clips (an A5C clip), and J0 and J1 misread
  it too. L1 + B3 reads it correctly.
- **Seed 2.** The head reads all three correctly. J0 and J1 still misread one A5C clip, placing two label switches
  inside it.
- On the calibration half, J1 against B + ASRF fixes 0, 0 and 1 clips and newly misreads 0, 1 and 1.

## 6. Question 2: merged changes and extra changes

**Changes that B + ASRF merged although a candidate cut was nearby.**

| Seed | Half | Merged by B + ASRF | Found by J1 | Found by J0 | Still merged (J1) |
|---|---|---|---|---|---|
| 0 | selection | 14 (18 recordings, 12 sessions) | 12 (16 recordings, 11 sessions) | 12 | 2 |
| 1 | selection | 10 (14 recordings, 11 sessions) | 7 (10 recordings, 9 sessions) | 7 | 3 |
| 2 | selection | 13 (18 recordings, 14 sessions) | 8 (12 recordings, 9 sessions) | 8 | 5 |
| 0 | calibration | 10 | 8 | 7 | 2 |
| 1 | calibration | 5 | 4 | 4 | 1 |
| 2 | calibration | 9 | 8 | 8 | 1 |

Every recovered change has both views right.

**Extra changes on the selection streams.**

| System | Extra changes (s0, s1, s2) | Recordings | Inside a clip | At a same-view join | Native clips with switches |
|---|---|---|---|---|---|
| J1 | 37, 41, 37 | 18, 18, 19 | 23, 24, 25 | 14, 17, 12 | 10, 10, 12 (of 284) |
| J0 | 35, 41, 39 | 18, 18, 19 | 22, 24, 26 | 13, 17, 13 | 11, 10, 13 |
| B + ASRF | 21, 20, 27 | 16, 11, 18 | 13, 13, 17 | 8, 7, 10 | 6, 5, 3 |
| L1 + B3 | 15, 11, 15 | 9, 9, 9 | 8, 4, 8 | 7, 7, 7 | 16, 2, 14 |

**Segment accounting, selection half.** The counts follow exactly from segment F1@50 = 2M / (P + T), where T = 502
reference units.

| System | Matched units M (s0, s1, s2) | Predicted segments P | Unmatched predicted segments |
|---|---|---|---|
| J1 | 467, 464, 465 | 532, 536, 530 | 65, 72, 65 |
| J0 | 467, 464, 465 | 531, 536, 533 | 64, 72, 68 |
| B + ASRF | 452, 457, 454 | 500, 506, 511 | 48, 49, 57 |
| L1 + B3 | 456, 449, 458 | 502, 489, 502 | 46, 40, 44 |

**Correct visual evidence lost.** These are fragments that the frozen L1 head reads correctly as fragments and as
whole recordings but that sit in units the system does not match.
- **Selection half.**
  - J1: 20, 22 and 20 fragments, from 17 recordings in each seed.
  - J0: the same numbers of fragments and recordings (sessions 13, 12 and 13 against 12, 12 and 13).
  - B + ASRF: 34, 27 and 34 fragments, from 26, 23 and 26 recordings.
  - L1 + B3: 32, 38 and 29 fragments, from 24, 29 and 23 recordings.
- **Calibration half.**
  - J1: 4, 5 and 5 fragments.
  - B + ASRF: 12, 10 and 11.
  - L1 + B3: 11, 20 and 9.

**Unmatched units by cause, selection half.**
- J1 leaves 35, 38 and 37 of 502 units unmatched. Of these, 22, 23 and 23 have a wrong label, 10 in each seed a missed
  cut, and 3, 5 and 4 an extra cut.
- B + ASRF leaves 50, 45 and 48: 30, 27 and 29 with a wrong label, 20, 18 and 18 with a missed cut, and 0, 0 and 1
  with an extra cut.

**Reading.**
- **More of the class evidence survives with exact decoding of the L1 probabilities.**
  - J0 and J1 recover most of the changes that ASRF's class vote merged, and read two or three of the three clips
    correctly.
  - Fewer fragments of correctly read evidence are lost: 20–22 against 27–38.
- **The cost is over-segmentation.** The decoders keep runs of another class inside clips and at same-view joins.
  Against B + ASRF, extra changes rise from 21, 20 and 27 to 37, 41 and 37, and unmatched predicted segments by 17, 23
  and 8. On balance segment F1@50 changes by +0.1, −1.3 and +0.5 points, with every interval containing zero.
- **J0 shows the same pattern.**
  - J0 never reads b, yet it recovers the identical sets of merged selection changes.
  - It loses the same numbers of correct fragments and recordings, and the session counts differ by at most one.
  - Its extra changes differ from J1's by 0–2 per seed.
  - No consistent J1 − J0 difference was detected.

## 7. Goal

| | Required | J1 |
|---|---|---|
| Mean segment F1@50, selection | ≥ 0.944 | 0.899 (−4.5 points) |
| Mean segment F1@50, calibration | ≥ 0.982 | 0.969 (−1.3 points) |
| Native macro-F1 per seed, selection | ≥ 0.973 | 0.970, 0.978, 0.968 |
| Native macro-F1 per seed, calibration | ≥ 0.980 | 0.993, 0.990, 0.983 |
| Boundary recall per seed, selection | ≥ 0.95 | 0.974, 0.974, 0.971 |
| Boundary recall per seed, calibration | ≥ 0.95 | 0.992, 0.990, 0.995 |
| Feasible choice | every seed | yes |

- J1 misses both mean scores and the selection native floor in seeds 0 and 2.
- With the calibration sensitivity bar of 0.974 + 0.01 = 0.984, J1 is 1.5 points short.
- J0 averages 0.899 / 0.968 and misses the goal too.

## 8. Interpretation under the plan

- **Does the boundary term add anything (J1 − J0)?** No consistent difference was detected. The choice kept λ = 0 in
  seed 1, and in seeds 0 and 2 the selection-half differences are −0.1 and +0.3 points. No gain can be credited to the
  boundary term.
- **Is J1 better than L1 + B3?** No consistent difference was detected in segment F1@50. J1 trades higher boundary
  recall and fewer lost reference units for more extra changes and lower boundary precision.
- **Is J1 better than the configurations the goal was set from?** No. It is below E0 + B3 on the selection half in
  every seed. No consistent difference against B7 + B3 was detected on the calibration half.
- **Outcome.** J1 is not stably better than J0 or L1 + B3, so the plan's stop rule applies and the joint-decoding
  route ends. No further grid, clipping rule, decoder variant or network was started.
- **What the diagnostics add.**
  - Exact decoding of the L1 probabilities recovers most merged changes and loses less correct evidence than either
    B3 or ASRF.
  - The loss moves to over-segmentation.
  - These are observations on a twice-used development set, not a gain.

## 9. What this does not show

- **That boundary evidence cannot help a joint decoder.** One objective, one small grid and one clipping rule were
  tested. The best calibration switch cost sat at the largest κ in every seed, so the grid did not bracket the best
  fixed switch cost. How J0 and J1 compare at larger κ is unknown.
- **That J1 and J0, or J1 and L1 + B3, are equal.** The intervals contain zero, which is not evidence of equality.
- **Why the extra changes arise.** For J0, a run where the L1 head prefers another class survives when its evidence
  outweighs two switch costs of 8. For J1 the boundary log-odds at both switches also count. Run lengths were not
  recorded, and this round does not separate genuine evidence from noise in those runs.
- **Independent evidence.**
  - Both validation halves were used in earlier development rounds, and the three seeds share one validation set.
  - Calibration scores are optimistic after the choice.
  - Constructed streams reuse recordings, and the counts of changed native clips are small.
  - Seed k pairs the L1 and ASRF models of the same seed, so the spread over seeds mixes both.

## 10. Cost

- **Extraction.** The boundary probabilities of 1,013 validation sequences per seed, with every check, took about one
  minute on one NVIDIA L40S.
- **Choice.** The 25-setting grid on the calibration half for three seeds took about 11 seconds on the CPU.
- **Evaluation.** Both halves, all systems, references and diagnostics took about 4 minutes.

## Files

- `summary.json` holds every system's rows, the choices, the contrasts, the diagnostics, the stable rule, the goals
  and the references. It contains no recording ids.
- `report_tables.md` holds the generated tables.
- `choices.json` holds the calibration grid of every seed and the choices. It was committed before J0 or J1 decoded
  the selection half.
- `preparation.json` holds the pre-metric extraction record. It has the input digests, the reproduction and
  peak-rule checks, the L1 probability checks and the alignment profiles.
- Boundary probabilities, per-sequence predictions and per-recording details stay on the server, in
  `runs/joint_boundary_decoding_v1/`.
