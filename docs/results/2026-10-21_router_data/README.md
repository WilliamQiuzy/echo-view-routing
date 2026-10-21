# A15: does more training data help the local router? (2026-10-21)

**Question.** A14's router was trained on 12–16 positive zones per seed. With the same 1,729-parameter P router and the
same frozen Base and Alternative, does a larger training pool teach a local selection that transfers and improves full
segmentation? The only trained difference between the two arms is the data pool.
- The development halves were reused for exploratory feedback.
- The router-check set is unseen by the routers, but A08 and A13 had seen its recordings, and it is not
  patient-independent.
- The test split and EchoNet-Pediatric were not read.
- Configuration: `configs/experiments/video_router_data_v1.yaml` (a8d0d28), fixed before any computation.

**Design.**
- **Fixed from A14.**
  - Base is frozen A08 + B3 at A08's committed setting. Alternative is A13's frozen H head at its committed threshold.
  - Zones are runs of disagreement between the two, merged when at most 16 samples apart. An accepted zone takes the
    Alternative labels.
  - The router reads the same 52 P features.
  - A zone's training label is positive when replacing that zone alone strictly raises the sequence's segment F1@50.
  - No-op equals Base and accept-all equals Alternative.
- **Session-disjoint split.** The 502 training sessions were permuted with a fixed seed. 100 sessions became router-check
  and 402 router-fit. The two groups share no session and no recording, and every stream was generated from its own
  group's recordings only.
- **Pools.** Streams use A01's generator settings: 4–8 fragments of 10–40 samples, edits with probability 0.5, and
  same-view joins with probability 0.35.

| Pool | Multi streams | Native clips | Recordings | Sessions |
|---|---|---|---|---|
| Large fit | 1,500 | 887 | 2,490 | 399 of 402 |
| Small fit (inside the large pool) | 300 | 200 | 1,210 | 361 |
| Check | 600 | 200 | 678 | 99 of 100 |

- The large pool's native quota fell short in two classes. Under the cap of 2 clips per session, A5C has 142 and SC4C
  145 of the 200 asked for. The other pools met their quotas.
- A recording can appear in several streams but never twice in one. A large-fit recording appears in 3.7 streams on
  average (at most 23), a small-fit one in 1.6 (at most 9) and a check one in 5.4 (at most 26). No stream equals one of
  A01's training bank streams.
- **Two routers, small-P and large-P.** They have the same architecture, the same initialisation and the same random
  numbers for drawing batches; only the fit pool differs. Each uses its own fit normalisation and balanced loss, trains
  on CPU for 1,350 updates of 128 zones, and keeps its final weights.
- **Barrier.** Each router and seed got one calibration choice from 8 thresholds plus no-op, under A14's feasibility
  rule. The choices were committed (94c86f7) before any selection output was read.
- **Checks passed.** Six synthetic zone tests, an exact pool rebuild, session and recording disjointness, the small pool
  inside the large one, the cached outputs reproducing Base and Alternative, label isolation, the stored candidates
  reproduced, no-op and accept-all reproduction, identical starts for both arms, router-only optimisation, 1,729
  parameters and an exact resume.

**Training candidates.** A zero zone leaves the sequence's F1 unchanged when it alone is replaced, and a harmful zone
lowers it.

| | Seed 0 small fit | Seed 0 large fit | Seed 0 check | Seed 2 small fit | Seed 2 large fit | Seed 2 check |
|---|---|---|---|---|---|---|
| Zones | 78 | 429 | 167 | 94 | 483 | 182 |
| Positive | 17 | 88 | 24 | 18 | 82 | 31 |
| Zero / harmful | 50 / 11 | 296 / 45 | 121 / 22 | 61 / 15 | 332 / 69 | 126 / 25 |
| Zones of at most 16 samples | 73 | 405 | 157 | 87 | 452 | 171 |
| Positives that recover a true segment / only remove false ones | 2 / 15 | 18 / 70 | 8 / 16 | 12 / 6 | 49 / 33 | 22 / 9 |
| Sessions with zones | 211 | 364 | 90 | 235 | 365 | 91 |

- The large pool has about five times the small pool's positive zones. A14's pool had 16 and 12.
- Seed 0's positives mostly remove false segments, and seed 2's mostly recover true ones. No positive zone trades a
  matched segment for fewer false ones.

**Router check.** These sessions are unseen by both routers. It was logged only, and nothing was chosen from it. "Check
F1" is the pooled segment F1@50 over the 600 check streams.

| Router | Balanced accuracy at 0.5 | Zones accepted at 0.5 (beneficial / zero / harmful) | Check F1 at 0.5 | At its calibration choice |
|---|---|---|---|---|
| Seed 0 small-P | 0.718 | 35 (14 / 13 / 8) | 0.9617 | threshold 0.5, the same |
| Seed 0 large-P | 0.868 | 34 (20 / 4 / 10) | 0.9616 | threshold 0.9, 14 zones (10 / 0 / 4), 0.9604 |
| Seed 2 small-P | 0.715 | 29 (16 / 3 / 10) | 0.9567 | no-op, 0.9555 |
| Seed 2 large-P | 0.786 | 37 (21 / 4 / 12) | 0.9579 | no-op, 0.9555 |

On the check streams Base scores 0.9603 and 0.9555 and Alternative 0.9590 and 0.9577 (seeds 0 and 2).

## Results (seeds 0 and 2; four cells)

| System | Mean F | − Base | − Alternative | Selection FP (short) | FN, four cells (short) | Mean boundary recall |
|---|---|---|---|---|---|---|
| Base (A08 A + B3) | 0.9698 | – | −0.0015 | 49 (22) | 51 (22) | 0.979 |
| Alternative (A13 H) = accept-all | 0.9712 | +0.0015 | – | 44 (18) | 54 (25) | 0.978 |
| A14 P (frozen, its committed choice) | 0.9724 | +0.0026 | +0.0012 | 41 (16) | 52 (23) | 0.979 |
| small-P | 0.9709 | +0.0011 | −0.0003 | 44 (16) | 54 (24) | 0.978 |
| large-P | 0.9705 | +0.0007 | −0.0008 | 47 (19) | 53 (24) | 0.978 |

Base, Alternative and A14 P reproduce their stored rows exactly. Native macro-F1 and the A4C/A5C counts of both new
routers equal Base's in every cell.

**Per seed.**
- **Seed 0, calibration.** Both new routers reach 0.9846, above Base (0.9827) and Alternative (0.9826) but below A14 P
  (0.9855). small-P accepts 4 of 49 zones at 0.5 and large-P 5 at 0.9.
- **Seed 0, selection.**
  - small-P scores 0.9590 with 10 of 46 zones accepted, +0.0027 [−0.0022, 0.0079] against Base. large-P scores 0.9572
    with 3 accepted, +0.0009 [−0.0022, 0.0046].
  - Both stay below Alternative (0.9639) and A14 P (0.9640). large-P − A14 P is −0.0068 [−0.0141, −0.0009].
  - Against Base on selection, small-P removes 7 unmatched predicted segments and adds 2, and it recovers 1 matched
    segment and loses 3. large-P removes 3, adds 1 and loses 1. On calibration both remove 3 and lose 1.
  - large-P − small-P is −0.0018 [−0.0059, 0.0021] on selection and 0 on calibration.
- **Seed 2.** Both routers chose no-op on calibration, so they equal Base exactly in both halves, as A14's routers did.
  - At every threshold, small-P's 5–6 accepted calibration zones leave F1 unchanged, so the tie rule chose no-op.
  - large-P's accepted zones leave F1 unchanged at 0.7–0.8 and lower it at the other thresholds.

**Fixed third-seed rule: not met**, so A15 closes with two seeds and this data-expansion configuration closes.
- **small-P.** Mean +0.0011 against the required +0.003. Its no-reversal and no-harm parts pass, with 24 missed short
  true segments against Base's 22.
- **large-P.** Mean +0.0007; its other parts pass.
- **Attribution.** The rule for a data-expansion effect is not met either. large-P is not ahead of small-P on
  development (one cell −0.0018, three ties), and the check F1 at 0.5 is mixed (−0.0001 and +0.0011).

## Reference-assisted diagnostic (after selection; not a system and not used by any decision)

Each zone was scored by the change in its sequence's segment F1 when it alone is replaced, using the reference.
"Recover" means the zone recovers a true segment, and "remove" means it only removes false ones. The router AUC is the
probability that a router scores a beneficial zone above a non-beneficial one in the same half.

| Half and seed | Zones | Beneficial (recover / remove) | Zero | Harmful | All beneficial zones applied together | Router AUC (small-P / large-P / A14 P) |
|---|---|---|---|---|---|---|
| Seed 0 selection | 46 | 12 (3 / 9) | 28 | 6 | 0.9670 | 0.74 / 0.81 / 0.87 |
| Seed 0 calibration | 49 | 4 (0 / 4) | 40 | 5 | 0.9855 | 0.90 / 0.98 / 0.99 |
| Seed 2 selection | 51 | 8 (5 / 3) | 30 | 13 | 0.9664 | 0.90 / 0.90 / 0.68 |
| Seed 2 calibration | 37 | 4 (2 / 2) | 28 | 5 | 0.9846 | 0.90 / 0.92 / 0.52 |

- **What the candidates contain.** Applying all individually beneficial zones together gives a four-cell mean of 0.9759,
  +0.0061 above Base. Native macro-F1 is unchanged and boundary recall does not fall. This reading uses the reference,
  so it measures what the candidates contain, not what a system achieves.
- **Combining zones.** No development sequence has two individually beneficial zones, so combining cannot cancel a
  single-zone gain there. In the training pools, the 0–2 sequences per pool with two beneficial zones kept their best
  single gain when combined.
- **Seed 2.** 4 beneficial calibration zones and 8 beneficial selection zones were left unselected. The new routers
  score 3 of the 4 and 6 of the 8 at 0.5 or above. On calibration, though, every threshold also admits zero or harmful
  zones that cancel or outweigh the gain.
- **Seed 0, selection.** large-P's 0.9 threshold accepts 2 of the 12 beneficial zones, although 6 of the 12 have
  large-P scores of at least 0.5. small-P accepts 6 beneficial and 4 harmful zones, and A14 P accepts 11 beneficial
  and 3 harmful.

## What this round shows

1. **The larger pool separated useful zones better on unseen sessions, but did not improve the segmentation.**
   - On the session-disjoint check pool, large-P's balanced accuracy is 0.868 and 0.786, against small-P's 0.718 and
     0.715.
   - Its development AUC is higher in all four cells, by 0.003 to 0.08.
   - After the fixed calibration choice, large-P is not ahead of small-P (mean −0.0005). So no data-expansion effect
     on segmentation is shown.
2. **The calibration choice rests on very few useful zones.** Each calibration half has only 37–49 zones per seed, 4
   of them beneficial. One admitted harmful zone can outweigh the gain, and the rule picked a high threshold (large-P,
   seed 0) or no-op (both arms, seed 2). Whether another selection rule would carry the better separation into the
   segmentation was not tested.
3. **No-op is not the ceiling of the candidates.** Seed 2 has unselected beneficial zones in both halves. Applying all
   individually beneficial zones reaches 0.9759 on average, and combining zones did not destroy single-zone gains
   wherever that could be tested.
4. **Neither new router is stronger than both source systems.** small-P and large-P beat Base by +0.0011 and +0.0007.
   They trail Alternative (−0.0003, −0.0008) and A14's frozen P (−0.0015, −0.0019). A08 A + B3 stays the reference,
   and no new strongest model is declared.

## Cost and files

- **GPU.** 0.645 h of 3: the timing probe, then one chunked frozen A08 pass per seed over 3,187 sequences. The
  development halves reused A13's verified caches.
- **CPU.** 0.94 h of 3 at the final commit, booked in `runs/video_router_data_v1/ledger.jsonl`. This includes the 8 decode workers and
  the agent's active wall clock. Each router trained in a few seconds.
- **Records.**
  - `preparation.json` (pools, candidates and labels) and `checks.json`.
  - `choices.json`, committed before selection, and `check_log.json`.
  - `screen.json` (rows, counts, zones, comparisons, accounting and the decision) and `diagnostic.json`.
- **Server only.** The pool manifest with recording ids, the window caches, predictions, candidates and routers.
