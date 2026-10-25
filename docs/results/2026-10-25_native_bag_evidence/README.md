# A34: whole native recordings as bags, mean against power-mean head objectives (2026-10-25)

**Question.** Treat each complete native recording as a bag with its public clip code. Train a copy of A08's classifier
head with a bag objective that lets a few windows carry the class evidence: a normalised power mean with r = 4 (P).
Does it give transferable gains over the mean-probability objective with r = 1 (M), with everything else matched?
- **Scope.** A CPU screen (GPU 0 h) on cached 16-slot A08 embeddings (source seeds 0 and 2) and A31's two session
  folds: 8 heads.
  - There were no new frames, no encoder update and no new or pseudo-expert labels.
  - No development, check, test or Pediatric data was read. The A08 + B3 cuts never change.
- **Data limits.** A08 trained on these recordings. The heads never saw the held-out sessions, but the whole model has
  seen them, so this is not an independent holdout.
- **What a high-scoring window means.** The bag objective allows a few windows to carry the class. It does not mean
  those windows show the LVOT, or that frames inside a recording are really another view.
- **History.**
  - R05 (span statistics of old E1 probabilities) and R07 (old image-token / global attention readouts) are closed and
    were not reopened.
  - A27, A29 and A31 trained heads or branches on T / P intervals with the mean-probability CE.
  - A34 changes the training unit (whole native bags only) and the aggregation (r), nothing else.
- Configuration `configs/experiments/video_native_bag_evidence_v1.yaml` (65dc067, committed before any output); code
  aa2b259; head sha256 values bfe043b (committed before the held-out readout). Task sheet
  `claude_prompts/2026-10-25-native-bag-evidence-screen.md`.

## Design

- **Bags.** Every native entry of a fit fold's pool is one bag: 740 / 747 recordings from 193 / 191 sessions, with
  classes PLAX / PSAX / A4C / A5C / SC4C = 200 / 200 / 200 / 67 / 73 and 200 / 200 / 200 / 75 / 72.
  - A bag holds all of the recording's A08 request windows. Weights are the samples each window represents under the
    nearest-centre map, divided by the recording's samples.
  - A recording has a median of 38 / 40 samples and 8 windows (2–217). Nothing was cropped, and no constructed stream or
    A32 error list was used (`bags.json`).
- **Objective.** g_c(r) = (Σ w_t p_tc^r)^(1/r), B_c = g_c / Σ_k g_k, L = −log B_y, computed in log space.
  - M uses r = 1, which is A29's mean-probability CE.
  - P uses r = 4, fixed once.
- **Heads.** Copies of A08's 512 → 5 classifier (2,565 parameters), bit-identical starts.
- **Training.** AdamW lr 1e-3, weight decay 1e-4, warm-up 135 then cosine, clip 1, 1,350 updates of 16 bags drawn
  uniformly. The batch plans are fixed per seed and fold and shared by M and P. CPU, single thread, final weights only.
- **Readouts.**
  - Every head and Init (A08's own head) were read under r1 (weighted mean) and r4 (power mean), over nearest-centre
    multiplicities.
  - The unchanged Base runs are relabelled by argmax, and equal neighbours merge.
  - The main comparison is P − M under r4. Init/r4 shows what the readout alone changes.

**Checks** (`checks.json`; one run failed first, see below).
- **Objective.**
  - It matches NumPy within 2e-16, r = 1 equals A29's CE within 4e-16, and a one-window bag equals plain CE.
  - Zero-weight padding, window order and splitting a window into weighted copies change nothing.
  - Gradients stay finite on uniform and two-class extremes.
- **Init/r1.** It reproduces Base exactly (runs and full sequences) in all four seed-fold cells. The head on the
  cached embeddings matches the cached logits within 6e-6.
- **Training mechanics.** Starts and plans are identical across arms, the optimizer binds exactly, updates are real,
  3 + 3 updates equal 6, and A31's output files are unchanged.
- **First check run.** It failed on one item: r = 1 against A29's CE differed by 4e-8 because the weights had been cast
  to float32, so they summed to 1 − ε. The weights were kept in float64 for both arms, and the rerun passed (4e-16).

**Training.** Each of the 8 heads ran 1,350 real updates in about 2 s. Training loss fell from about 0.04–0.05 to
0.015–0.041 (`weights.json`, with fit-fold monitors).

## Results (held-out folds, out of fold per source seed)

**All held-out P runs** (seed 0: 6,788 runs, 254 Base-wrong, 1,150 correct A5C; seed 2: 6,742, 237, 1,128).
- ΔF1 is the multi segment F1@50 minus Base, averaged over the two folds.
- Δmissed short is the missed short true segments minus Base, summed over the two folds.

| Seed | Model / readout | Fixed / broken / net | A5C read as A4C fixed | A4C read as A5C fixed | Correct A5C kept | ΔF1 | Δ native macro-F1 | Δ missed short | Whole held-out bags accuracy (f1, f0) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | Init / r1 (= Base) | 0 / 0 / 0 | 0 | 0 | 1,150 | 0 | 0 | 0 | 0.984, 0.987 |
| 0 | Init / r4 (readout only) | 6 / 2 / +4 | 1 | 4 | 1,150 | +0.0007 | +0.0009 | 0 | 0.984, 0.988 |
| 0 | M / r1 | 42 / 18 / +24 | 0 | 41 | 1,134 | +0.0017 | +0.0072 | +2 | 0.988, 0.991 |
| 0 | M / r4 | 45 / 17 / +28 | 0 | 44 | 1,135 | +0.0026 | +0.0072 | 0 | 0.985, 0.992 |
| 0 | P / r1 | 54 / 34 / +20 | 0 | 54 | 1,121 | +0.0020 | +0.0073 | +2 | 0.988, 0.992 |
| 0 | **P / r4** | 57 / 34 / **+23** | 0 | 56 | 1,122 | +0.0032 | +0.0073 | +4 | 0.988, 0.993 |
| 2 | Init / r1 (= Base) | 0 / 0 / 0 | 0 | 0 | 1,128 | 0 | 0 | 0 | 0.984, 0.989 |
| 2 | Init / r4 (readout only) | 7 / 2 / +5 | 5 | 1 | 1,128 | +0.0008 | +0.0010 | 0 | 0.985, 0.989 |
| 2 | M / r1 | 32 / 25 / +7 | 0 | 31 | 1,106 | +0.0005 | +0.0034 | +5 | 0.989, 0.989 |
| 2 | M / r4 | 32 / 25 / +7 | 0 | 31 | 1,106 | +0.0007 | +0.0034 | +4 | 0.989, 0.991 |
| 2 | P / r1 | 48 / 59 / −11 | 0 | 48 | 1,073 | −0.0023 | +0.0042 | +15 | 0.988, 0.989 |
| 2 | **P / r4** | 50 / 61 / **−11** | 0 | 50 | 1,071 | −0.0023 | +0.0042 | +16 | 0.988, 0.991 |

**P − M under r4** (the main comparison).

| Seed | Net runs | Correct A5C kept | ΔF1 | Missed short true | Native run accuracy [descriptive 95% session interval] |
|---|---|---|---|---|---|
| 0 | −5 | −13 | +0.0006 | +4 | +0.0013 [−0.0018, +0.0044] |
| 2 | −18 | −35 | −0.0030 | +12 | +0.0032 [−0.0007, +0.0076] |

- **A32 S1.** None of the A5C runs read as A4C was recovered by any head or readout: 0 of 69 / 34 / 67 / 34 in the four
  cells. In the fold-1 cells, A32 S2 runs kept as A5C fell to 137–149 of 157 (seed 0) and 110–138 of 147 (seed 2)
  for M / P under r4.
- **Native and constructed.** Native runs gain slightly under every trained head (for example P/r4: 13 fixed, 2 broken
  in seed 0 fold 1). The breaks fall mostly on constructed and short runs: P/r4 in seed 2 fold 1 fixed 21 and broke 41
  constructed runs.
- **Whole bags against short runs.** On whole held-out native recordings both objectives transfer slightly
  (accuracy 0.984–0.989 → 0.985–0.993, near ceiling). That does not carry over to A5C on short Base runs.

## Answers

1. **The power-mean objective adds nothing over the mean objective.**
   - Under the fixed r4 readout, P − M is −5 and −18 net runs and −13 and −35 correct A5C runs. Segment F1 is +0.0006
     in seed 0 and −0.0030 in seed 2.
   - The protocol's next-step rule is not met in either seed. This closes only the whole-native power-mean head
     configuration. It does not show that weak supervision fails in general.
2. **Readout and training are separated.**
   - Init/r4 alone moves +4 / +5 net runs and segment F1 +0.0007 / +0.0008. That is a readout effect with no
     training.
   - So M/r4 − M/r1 (+4 / 0) is mostly that readout effect, not training.
3. **Every trained head moves the A4C/A5C boundary the same way as A27–A31.**
   - The fixes are A4C runs read as A5C (31–56 per seed). No A5C run read as A4C is fixed (S1 0 in all cells).
   - The breaks fall on correct A5C runs (15–57 per seed).
   - The mean-objective head M is net positive in both seeds (+24 / +7 under r1, segment F1 +0.0017 / +0.0005). That is
     background, not the tested contrast. It resembles A31's F_U (+13 / +10) and repeats the same trade.
4. **Next-step motive.**
   - Training on clip codes, whether by interval or by whole-recording bag, with mean or power-mean aggregation, keeps
     recovering A4C runs that are read as A5C. It recovers none of the A5C runs that are read as A4C.
   - The open route is unchanged: local expert evidence (the pilot tool is ready, 0 real records). There is no further
     clip-code head objective.

## Cost and files

- **GPU 0 h. CPU about 0.25 of 2 h:**
  - stages, including one failed check: 0.01 h;
  - 8 heads: about 17 s;
  - the agent's active wall clock.
  - All of it is in `runs/video_native_bag_evidence_v1/ledger.jsonl` on the server.
- **Records.** `bags.json`, `checks.json`, `weights.json` (8 manifests with fit-fold monitors), `readout.json`.
- **Code.** `scripts/train_native_bag_evidence.py`, `echo_routing/temporal/bag_objective.py`,
  `tests/test_bag_objective.py`.
- **Server only.** Bag tables and plans, trained heads, per-run predictions and all identifiers.
- **A33.** A dated scope note was appended to A33's README: the Init rows were pre-registered as background; only the
  post hoc reading of Init_I32 was not.
