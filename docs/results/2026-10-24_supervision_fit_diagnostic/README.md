# A30 (diagnostic): training input/supervision audit and small-group fit (2026-10-24)

**Question.**
1. Do the training inputs and supervision used by A27–A29 contain direct conflicts, or only shared-window tension? A
   direct conflict is an identical weighted input with different interval labels.
2. Can a fixed small group of A29's training P runs, both Base-wrong and Base-correct, be memorised under one fixed
   strong fit setting? The frozen head (F) and the last-block adaptation (T) are compared.

**Scope.**
- Training data only (A15 large fit, protocol-train). No development, check, test or Pediatric data was read. No label
  was changed.
- This is diagnostic only. There is no new segmentation candidate, no gate and no seed 1.
- Configuration: `configs/experiments/post_a29_supervision_fit_diagnostic_v1.yaml` (3c1f782), committed before any
  output. Task sheet `claude_prompts/2026-10-24-supervision-conflict-and-small-fit-diagnostic.md`.

**Correction carried over.** My A29 report and index row said T − F was zero in five of six cells. The JSON shows four
zero cells and two non-zero ones (−0.000977 in seed 0 calibration, +0.000388 on seed 2 check). Codex's dated correction
in the local index is kept unchanged.

## Stage 1: identity audit (reference level, `audit.json`)

- **Window identity.** The ordered 16 (recording, frame index, gamma as decoded) of a request window, plus its marker
  slot. The same centre frame with another context or marker is a different input.
- **Interval identity.** The weighted input multiset: window identity with count / interval length, as exact fractions.
- **Windows.** The 49,654 fit windows have 47,893 distinct identities. 3,319 rows repeat an identity found elsewhere,
  for example the same clip in a native sequence and a stream.
- **Same centre, other context.** 4,816 of 42,127 centre references appear in windows with different context or marker.
  This is described only and not counted as a conflict.

| Seed | T / P runs | Distinct interval identities | Direct conflicts | Window identities supervised by several labels | Runs touching them |
|---|---|---|---|---|---|
| 0 | 7,242 / 7,411 | 13,125 | **0 groups** (error and CE lower bounds 0) | 4,205 of 47,893 (8.8%) | 9,041 of 14,653 |
| 2 | 7,242 / 7,366 | 13,061 | **0 groups** | 4,214 of 47,893 (8.8%) | 9,009 of 14,608 |

- **No direct conflicts.** No identical weighted input carries two different interval labels, so no Base-wrong run is
  explained by a direct conflict.
- **Shared windows are common.** Many windows near class changes are shared by neighbouring runs of different labels.
  This is tension, not an impossibility.
- **Scope of the audit.** References only. Identical pixels under different references and clinical label questions are
  not excluded. In the selected small groups, 37 / 32 pairs of different identities decode to identical pixels
  (`cache.json`); duplicated frames are a candidate, not checked.

**Where the Base-wrong P runs are** (seed 0 / seed 2).

| Descriptor | Seed 0 (of 274) | Seed 2 (of 256) |
|---|---|---|
| A4C/A5C confusions (A4C→A5C / A5C→A4C) | 125 / 136 (95%) | 112 / 131 (95%) |
| Native clips / constructed within one fragment / constructed across a join | 53 / 119 / 102 | 47 / 112 / 97 |
| Pure / mixed | 190 / 84 | 177 / 79 |
| Single / multi recording | 172 / 102 | 159 / 97 |
| Short (≤ 16 samples) | 166 | 142 |
| With windows whose target sample has another label (all mixed runs) | 29 (mean weight 0.039) | 30 (0.046) |
| Recordings / sessions involved | 209 / 156 | 217 / 155 |

- **Errors sit in the source recordings.** Of the constructed single-recording errors whose recording also has a native
  clip in the pool (79 / 73), that native clip is entirely Base-correct for only 16 / 17. So most errors already sit in
  the source recording, not only in its constructed use.
- **Pure runs.** No pure Base-wrong run has an off-label target window. Their errors are not caused by windows reaching
  into another class.

## Stage 2: small-group fit (`selection.json`, `cache.json`, `fit.json`)

- **Groups.** Per seed, 16 Base-wrong and 16 Base-correct P runs outside conflict groups (none existed). They come from
  distinct sessions (22–25 per group), and the correct runs match the wrong ones' class and length bin.
- **Bystanders.** 16 + 16 bystanders share no window identity with the fit group; they share 4 / 7 recordings and 14 / 18
  sessions with it.
- **Groups chosen first.** The groups were frozen (manifest sha recorded) before any fit.
- **Setting.**
  - All original windows and multiplicities, pooled CE, eval mode.
  - AdamW: head lr 1e-3, T representation lr 1e-4, weight decay 0, clip 1, constant learning rates.
  - 1,000 full-group updates.
- **Checks.**
  - The small prefix cache reproduces A15's logits within 2.9e-6, with no argmax mismatch.
  - The CPU fake case passed: pooled CE equals numpy within 4e-16.
  - The optimizer is bound, the frozen prefix is unchanged, and the A08 files are unchanged.
- **Structurally untrainable tensors.** Four of T's 30 trainable tensors get zero gradient: block 15's `rel_pos_h/w/t`
  and `pool_q` (6,528 parameters). The readout uses only the class-token output, which these tensors cannot reach, so
  T's effective trainable set is 7,491,461. This also held in A28 and A29.

| Seed, arm | Fitted? | Base-wrong learned | Base-correct broken | Final CE mean / max | Worst true-class margin among Base-wrong runs | Head / representation moved (L2) | Bystanders: wrong fixed / correct broken (of 16 / 16) |
|---|---|---|---|---|---|---|---|
| s0 F | **no** (accuracy 0.969, CE 0.105) | 15 / 16 | 0 | 0.105 / 1.045 | −0.61 | 6.75 / – | 9 / 8 |
| s0 T | yes | 16 / 16 | 0 | 0.000 / 0.000 | +11.55 | 1.02 / 11.11 | 11 / 8 |
| s2 F | yes | 16 / 16 | 0 | 0.038 / 0.216 | +2.16 | 5.57 / – | 8 / 1 |
| s2 T | yes | 16 / 16 | 0 | 0.000 / 0.000 | +12.07 | 1.00 / 9.97 | 9 / 1 |

- **Starting point.** At update 0, the fit groups' wrong runs have CE 1.99 / 2.12 and margin −1.69 / −1.91. The worst
  margins are −3.45 / −4.15.
- **T** fits each group within 100 updates. **F** needs the full 1,000 and leaves one seed-0 error unlearned (CE 1.045).
- **Bystanders move both ways.** About half of their errors flip right (8–11 of 16). But in seed 0, 8 of 16 correct
  bystanders flip wrong, under both F and T; in seed 2, one.
  - The memorised decision mainly moves the A4C/A5C boundary.
  - It is not a transferable correction.
  - Bystanders are training runs and not a holdout.

## What this diagnostic shows

1. **The training supervision has no direct input conflicts at this exact-identity level.** The failure of A27–A29 to
   correct their training errors is not explained by identical inputs carrying contradictory labels. Shared-window
   tension exists but is not a contradiction.
2. **The errors are learnable on small groups.** Under the fixed strong setting, T memorises 32 of 32 wrong runs, and the
   linear head 31 of 32. So these training errors are not unlearnable from these inputs.
   - This does not show generalisation.
   - It does not prove that A29 failed only through dilution (3.5–3.7% of P runs are wrong) or update size.
3. **Memorising them shifts the A4C/A5C decision rather than learning a transferable rule.** On bystanders, fixes come
   with breaks: in seed 0, 8 of 16 correct runs break.
4. **Unknown.** Why the full-pool training did not correct its errors is not separated. Candidates include the error
   share in each batch, the smaller learning rates, retention of correct runs, and whether any correction can
   generalise across sessions.

## Recommendation for the next formal round (not started)

Test whether corrections of A4C/A5C errors *generalise*, before another full training.
- **Design.**
  - Session-cross-fitted folds of the training pool.
  - The recognition branch (head, or T) is trained with A29's input, objective and schedule.
  - The matched factor is error-focused against uniform P sampling: for example, half of each P batch drawn from the
    fold's Base-wrong runs.
- **Readout.** The held-out training sessions' Base-wrong runs fixed against Base-correct runs broken, at fixed rules.
- **Only then.** A development evaluation and the full-system gate.
- **Why this order.** This diagnostic shows that the errors can be fitted but that fitting moves bystanders both ways. So
  the open question is held-out correction without breakage, not capacity.

## Cost and files

- **GPU.** 0.18 of 2 h: small cache 0.017 h, four fits 0.156 h, gradient check 0.011 h.
- **CPU.** About 0.4 of 2 h, including the agent's active wall clock.
- **Records.** `audit.json` and `selection.json` (325cae9); `cache.json` and `fit.json` (this commit).
- **Diagnostic-only weights.** `runs/post_a29_supervision_fit_diagnostic_v1/fit/{F,T}_s{0,2}/final.pt` on the server.
  They are not frozen models and not candidates.
- **Server only.** Private subset and conflict manifests, the small prefix caches and all identifiers.
