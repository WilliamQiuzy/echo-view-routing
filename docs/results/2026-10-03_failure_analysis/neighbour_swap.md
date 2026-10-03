# Neighbour-swap check (round 4; diagnostic only)

Targets are E0 A4C units next to an A5C unit; controls are E0 A4C units without an A5C neighbour that every main
system gets right. A cell gives the share of units whose samples mostly keep their class, the share matched by
segment F1@50 where the variant keeps different-class neighbours, and the number of source clips.

## targets

| system, originally | units | original | isolated | same_class_neighbours | other_clip_neighbours | other_a5c_only |
|---|---:|---|---|---|---|---|
| B3, wrong | 15 | 73% (0% matched), 14 clips | 100%, 14 clips | 100%, 14 clips | 93% (49% matched), 14 clips | 89% (33% matched), 14 clips |
| B3, right | 50 | 100% (100% matched), 58 clips | 100%, 58 clips | 100%, 58 clips | 99% (74% matched), 58 clips | 100% (70% matched), 58 clips |
| ASRF_s0, wrong | 20 | 70% (0% matched), 19 clips | 100%, 19 clips | 100%, 19 clips | 92% (42% matched), 19 clips | 93% (37% matched), 19 clips |
| ASRF_s0, right | 45 | 100% (100% matched), 55 clips | 100%, 55 clips | 100%, 55 clips | 97% (76% matched), 55 clips | 99% (73% matched), 55 clips |
| ASRF_s1, wrong | 13 | 54% (0% matched), 12 clips | 100%, 12 clips | 100%, 12 clips | 92% (56% matched), 12 clips | 92% (46% matched), 12 clips |
| ASRF_s1, right | 52 | 100% (100% matched), 59 clips | 100%, 59 clips | 100%, 59 clips | 96% (77% matched), 59 clips | 97% (74% matched), 59 clips |
| ASRF_s2, wrong | 7 | 100% (0% matched), 8 clips | 100%, 8 clips | 100%, 8 clips | 100% (81% matched), 8 clips | 100% (57% matched), 8 clips |
| ASRF_s2, right | 58 | 100% (100% matched), 62 clips | 100%, 62 clips | 100%, 62 clips | 99% (79% matched), 62 clips | 99% (75% matched), 62 clips |

## controls

| system, originally | units | original | inject_same | inject_a5c |
|---|---:|---|---|---|
| B3, right | 152 | 100% (100% matched), 112 clips | 100% (100% matched), 112 clips | 92% (58% matched), 112 clips |
| ASRF_s0, right | 152 | 100% (100% matched), 112 clips | 100% (99% matched), 112 clips | 92% (61% matched), 112 clips |
| ASRF_s1, right | 152 | 100% (100% matched), 112 clips | 100% (100% matched), 112 clips | 93% (71% matched), 112 clips |
| ASRF_s2, right | 152 | 100% (100% matched), 112 clips | 100% (100% matched), 112 clips | 98% (78% matched), 112 clips |

## Split by how the frozen classifier reads the inserted A5C blocks

Draw level: each row is one donor draw. 'read as A5C' means the classifier's mean probability over every
inserted A5C block's own samples picks A5C; otherwise at least one inserted A5C block looks like another class.

| kind, variant, system | read as A5C: matched | read as another class: matched |
|---|---|---|
| target, other_clip_neighbours, B3 | 80% of 138 | 39% of 57 |
| target, other_clip_neighbours, ASRF_s0 | 78% of 138 | 37% of 57 |
| target, other_clip_neighbours, ASRF_s1 | 87% of 138 | 39% of 57 |
| target, other_clip_neighbours, ASRF_s2 | 93% of 138 | 44% of 57 |
| target, other_a5c_only, B3 | 79% of 127 | 29% of 68 |
| target, other_a5c_only, ASRF_s0 | 79% of 127 | 31% of 68 |
| target, other_a5c_only, ASRF_s1 | 86% of 127 | 37% of 68 |
| target, other_a5c_only, ASRF_s2 | 91% of 127 | 40% of 68 |
| control, inject_a5c, B3 | 69% of 310 | 35% of 146 |
| control, inject_a5c, ASRF_s0 | 75% of 310 | 32% of 146 |
| control, inject_a5c, ASRF_s1 | 85% of 310 | 40% of 146 |
| control, inject_a5c, ASRF_s2 | 92% of 310 | 47% of 146 |