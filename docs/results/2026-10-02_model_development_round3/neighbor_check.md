# Neighbour-swap check (round 3, step 3; diagnostic only)

Fragments the E1 probe labels right and an E1 M5 run gets wrong on at least 80% of their samples. The fragment's
embeddings are fixed; only the surroundings change. A cell counts fragments whose mean probability at that stage
picks the reference class; source clips are counted once.

| Surroundings | fragments | local (convolutions) | context (attention) | before pooling | output |
|---|---:|---|---|---|---|
| original | 64 | 40/64 (19/25 clips) | 9/64 (8/25 clips) | 13/64 (9/25 clips) | 0/64 (0/25 clips) |
| isolated | 64 | 49/64 (21/25 clips) | 40/64 (21/25 clips) | 40/64 (20/25 clips) | 43/64 (20/25 clips) |
| same class neighbours | 64 | 54/64 (23/25 clips) | 53/64 (23/25 clips) | 53/64 (23/25 clips) | 59/64 (24/25 clips) |
| other clip neighbours | 64 | 41/64 (21/25 clips) | 23/64 (15/25 clips) | 28/64 (17/25 clips) | 21/64 (14/25 clips) |

## Case B: multi_selection_0004 126-136 (A4C), M5_s0

| Surroundings | local | context | before pooling | output | boundary at the two edges |
|---|---|---|---|---|---|
| original | A4C (A4C 0.53, A5C 0.47) | A5C (A4C 0.24, A5C 0.76) | A5C (A4C 0.29, A5C 0.70) | A5C (A4C 0.29, A5C 0.71) | [1.0, 1.0] |
| isolated | A4C (A4C 0.73, A5C 0.26) | A4C (A4C 0.84, A5C 0.16) | A4C (A4C 0.82, A5C 0.18) | A4C (A4C 0.82, A5C 0.18) | [None, None] |
| same class neighbours | A4C (A4C 0.87, A5C 0.12) | A4C (A4C 1.00, A5C 0.00) | A4C (A4C 0.99, A5C 0.01) | A4C (A4C 1.00, A5C 0.00) | [0.0587, 0.3219] |
| other clip neighbours | A4C (A4C 0.50, A5C 0.49) | A5C (A4C 0.38, A5C 0.62) | A5C (A4C 0.40, A5C 0.60) | A5C (A4C 0.39, A5C 0.60) | [1.0, 1.0] |
