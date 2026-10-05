# Round 8, step 1 (2026-10-04): does the training library support cross-recording supervised contrast?

The project's aim stays the same: a view-recognition and temporal-segmentation system that beats B3, ASRF and the
strongest previous configuration. The project lead proposed one new training hypothesis. It says the existing training
does not constrain view discrimination across recordings enough, so explicitly pulling together same-class recordings
from different sources, and pushing A4C and A5C apart, might help more than changing the readout again.

Before any training, this step asked whether the training library holds the problem that the hypothesis targets. The
round-8 training (C0 against C1) was to run only if it did. The decision rule was committed before any number was
computed (`configs/experiments/training_library_check_v1.yaml`, commit 7219980). No model was trained, no label was
changed, the validation clips of rounds 4 to 7 were not revisited, and the test split was not read. The paper stays
paused.

## Answer

1. **Does the training library support the hypothesis? No.** The pre-declared decision needs three conditions, and the
   third fails.
   - **Breadth holds.** The E1 probe or ASRF confuse A4C and A5C in 26 training clips from 24 sessions. The three
     sessions with most of them hold 19%.
   - **Shared failure holds.** 15 of the 26 are confused by at least two of E0, the E1 probe and ASRF.
   - **Cross-recording structure fails, clearly.** In E1's clip embeddings, same-class clips from different sessions
     have a mean cosine similarity of 0.92. A4C and A5C clips from the same session have 0.28. The embedding is
     organised by view, not by session. The deficiency the hypothesis targets is not present in the training library.
2. **What was trained? Nothing.** Following the rule, the C0/C1 comparison was not assembled.
3. **Does the new supervision help? Not tested.** The training library gave no basis for it.
4. **How far is the strong baseline? Unchanged.** No system so far meets the final goal on both halves. The closest
   points come from different systems.
   - On the selection half, round 2's E0 M3p with B3 decoding reaches 0.934, 1.1 points short of the target of 0.944.
     It also misses the native floor.
   - On the calibration half, the geometry round's G1 ASRF reaches 0.974, 0.8 points short of 0.982.
   - E1 ASRF, the reference of most rounds, is 3.6 points short on the selection half and 1.4 points short on the
     calibration half.

All predictions here are in-sample. Every training clip was seen in training by E0, by E1's encoder (EchoViewCLIP stage
1 was trained on all 3,364 of them), by the E1 probe and by ASRF. The numbers describe how the training data is fitted,
not how the models generalise.

## 1. The training library

| Class | Clips | Sessions |
|---|---:|---:|
| PLAX | 1,120 | 448 |
| PSAX | 974 | 362 |
| A4C | 858 | 405 |
| A5C | 217 | 139 |
| SC4C | 195 | 154 |
| All | 3,364 | 502 |

A session is the recording session a clip comes from. 113 sessions hold both A4C and A5C training clips.

## 2. A4C/A5C confusions (clip majority, in-sample)

| System | A5C read as A4C: clips (sessions) | A4C read as A5C: clips (sessions) | A5C clip accuracy | A4C clip accuracy |
|---|---|---|---:|---:|
| E0 head | 2 (2) | 7 (6) | 0.991 | 0.986 |
| E1 probe | 16 (16) | 5 (5) | 0.926 | 0.991 |
| E1 ASRF, majority of seeds | 13 (13) | 7 (7) | 0.940 | 0.991 |

- **Union.** The E1 probe or ASRF confuse 26 clips from 24 sessions: 18 A5C read as A4C and 8 A4C read as A5C. No
  session holds more than 2.
- **Co-failure.** 15 of the 26 are confused by at least two of the three systems; one is confused by all three. E0,
  whose encoder and head were both trained on these clips, fits them almost perfectly.
- **ASRF seeds.** The three ASRF seeds disagree on 27 of the A4C/A5C training clips.

Unlike the four validation clips of rounds 4 to 7, the training confusions are spread thinly over many sessions. They
are few in each.

## 3. Correct and confused clips

Medians (interquartile range in brackets). The margin is logit(A5C) − logit(A4C), averaged over the clip.

| Clips | E1 margin | E0 margin | Length |
|---|---|---|---|
| A5C, correct (199) | 2.53 (1.81 to 3.03) | 5.01 (3.95 to 6.41) | 3.6 s (2.2 to 6.0) |
| A5C, confused (18) | 0.31 (−0.09 to 0.46) | 3.31 (2.80 to 4.07) | 5.0 s (2.9 to 8.1) |
| A4C, correct (850) | −1.83 (−2.17 to −1.38) | −5.46 (−6.70 to −3.95) | 3.4 s (2.2 to 6.3) |
| A4C, confused (8) | 0.58 (0.27 to 1.12) | −0.57 (−1.36 to 0.04) | 3.5 s (2.9 to 5.4) |

The confused clips sit near the decision boundary, not deep inside the other class. The confused A5C clips are somewhat
longer, but the group is small.

## 4. Cross-recording similarity of E1 clip embeddings

A clip embedding is the mean of the clip's E1 frame embeddings, scaled to unit length. Pairs are taken over the 113
sessions that hold both classes.

| Pair | Pairs | Mean cosine similarity |
|---|---:|---:|
| Same class, same session | 541 | 0.953 |
| Same class, different sessions | 55,392 | 0.922 |
| A4C and A5C, different sessions | 48,184 | 0.329 |
| A4C and A5C, same session | 536 | 0.279 |

Same-class clips from different sessions are almost as similar as same-class clips from one session. Both are far more
similar than A4C and A5C clips from the same session. Recordings of one view already cluster across sources. A
loss that pulls them together and pushes A4C from A5C would act mostly on pairs that are already separated.

E1's encoder was itself trained with a label-driven contrastive objective. EchoViewCLIP stage 1 matches each video to
the text of its view name, which pulls all clips of a view towards one shared anchor. Cross-recording supervised
contrast between samples would add pairwise pulls on top of that. It was not an equivalent experiment, but its main
effect is already visible in the training data.

## 5. What the training library is missing for this objective

The candidate stops because the training data do not show its target. A training objective aimed at the cases that
remain would need cases that are not in the library.
- **More A5C recordings from independent sessions.** The library has 217 A5C clips from 139 sessions, against 858 A4C
  clips from 405.
- **A label finer than the clip.** The public labels name a view for the whole clip. They cannot show which frames of
  a borderline A5C clip display the five-chamber structures. A loss built on clip labels can therefore not target the
  near-boundary cases more precisely than the current classification loss already does.

No label was changed on the basis of model predictions, and every clip was kept.

## 6. Files

- **Rule.** `configs/experiments/training_library_check_v1.yaml`.
- **Code.** `scripts/round8_training_library.py`.
- **Numbers.** `summary.json` holds aggregates only. The per-clip table with clip ids stays on the server
  (`runs/md8_training_library/clips.csv`).
