# Where Pediatric view changes are lost: bounded fault localisation (2026-10-05)

D2 + ASRF reads Pediatric native clips well, yet finds only 0.48 to 0.91 of the view changes in the stitched exam
streams. This note locates the stage of the decoding path where each change is lost. It compares D2 + ASRF with
D0 + ASRF and with D2 + B3 of the same seed. Nothing was trained and no threshold was changed: the existing models are
run with their own decoding rules (`scripts/pediatric_boundary_faults.py`, commit acac884).

The streams are the frozen test exam streams: 2,574 streams holding 2,558 true view changes, all between A4C and
PSAX. They are built by joining whole clips of one exam, so their changes come from stitching, not from continuous
scanning. The tolerance is 0.5 s, as for boundary recall. The counts reproduce the step-2 report: the share of changes
found is the boundary recall reported there, for every system and seed.

## How a change can be lost

Both decoders make final label changes only at candidate cuts.
- **ASRF.** The candidates are the accepted peaks of its boundary probability, at the head's own threshold of 0.5.
  Each span between candidates takes ASRF's class vote, and equal neighbours merge.
- **B3.** The candidates are its divergence detector's boundaries, and each span takes its mean-probability class.

For each true change, the first stage that loses it is recorded.

| Stage | Meaning |
|---|---|
| No candidate | No accepted candidate within 0.5 s, and no final boundary within 1.5 s |
| Merged | A candidate within 0.5 s, but the spans on both sides got the same class and merged |
| Displaced | No candidate within 0.5 s, but a final boundary 0.5–1.5 s away |
| Found | A final boundary within 0.5 s; then split by whether both views are right |

## Boundary loss stage table

Counts of the 2,558 true changes per seed, with the number of groups in brackets. Extra cuts are final boundaries
more than 0.5 s from every true change and inside a clip.

| System | Seed | Found, both views right | Found, a view wrong | Merged | Displaced | No candidate | Extra cuts |
|---|---|---|---|---|---|---|---|
| D0 + ASRF | 0 | 2,000 | 381 (303) | 63 (61) | 6 | 108 (98) | 80 |
| D0 + ASRF | 1 | 1,605 | 126 (112) | 95 (94) | 3 | 729 (509) | 16 |
| D0 + ASRF | 2 | 1,706 | 151 (133) | 34 (34) | 5 | 662 (483) | 25 |
| D2 + ASRF | 0 | 1,788 | 21 (20) | 3 (3) | 1 | 745 (512) | 3 |
| D2 + ASRF | 1 | 2,286 | 46 (41) | 10 (10) | 4 | 212 (174) | 12 |
| D2 + ASRF | 2 | 1,223 | 6 (6) | 5 (5) | 0 | 1,324 (816) | 1 |
| D2 + B3 | 0 | 2,485 | 2 (2) | 0 | 2 | 69 (63) | 24 |
| D2 + B3 | 1 | 2,485 | 2 (2) | 0 | 3 | 68 (61) | 19 |
| D2 + B3 | 2 | 2,519 | 2 (2) | 4 (4) | 1 | 32 (29) | 46 |

- **ASRF loses changes almost only at the candidate stage.** Its boundary head proposes no cut near the change.
  - For D2 + ASRF, merging and displacement together account for at most 14 changes per seed.
  - Near the changes ASRF misses, the boundary probability is far below the threshold. Its median is 0.015 / 0.021 /
    0.001 by seed for D2, and 0.18 / 0.002 / 0.002 for D0. Fewer than 4% of those changes reach 0.5, and 86–94%
    stay below 0.25 in every seed except D0's seed 0, where 57% do.
  - The head is silent, not close to the threshold.
- **The problem was there before adaptation, and it depends strongly on the seed.**
  - D0 + ASRF already loses 108, 729 and 662 changes at this stage.
  - Comparing the same changes, D2 adds 598 and 658 new losses in seeds 0 and 2, almost all at the candidate stage.
    It removes 627 in seed 1. 151, 200 and 671 changes are lost in both D0 and D2.
  - What D2 improves is the labelling of the changes ASRF does find. Changes found with a view wrong fall from 381 /
    126 / 151 to 21 / 46 / 6.
- **B3 is the strong control and solves most of it.** With the same D2 head, B3 finds 97.2–98.6% of the changes and
  loses only 32 to 69 at the candidate stage. Of the changes D2 + ASRF loses, B3 finds 685 of 749, 169 of 226 and
  1,294 of 1,329. Its price is a few more extra cuts inside clips (19 to 46 against 1 to 12).
- **What this analysis does not show is why the head is silent.** It recomputes the head's outputs with the saved
  models. It does not separate a feature shift between the populations from the nature of stitched joins or from the
  head's training, and it attributes the loss to none of them.

## Relation to the remaining EV9V problem

An earlier round already measured the same stage on EV9V (`docs/results/2026-10-03_evidence_pooling/`, step 1).
- **On EV9V the candidates are there.** E1 ASRF's boundary head proposes 96–98% of the view changes as candidate cuts
  on the selection half (0.963–0.979 by seed) and 99% on the calibration half.
- **EV9V's remaining errors arise later.** They come from the class decision on spans that already have candidate
  boundaries, mainly A5C read as A4C in a few recordings (rounds 4 to 7).
- **The two populations fail at different stages.** The Pediatric loss sits at the candidate stage, in an ASRF whose
  temporal model was trained on EV9V only and never on Pediatric. A strong simple baseline, B3 on the same adapted
  head, already finds those changes.

## Conclusion

Under the rule set for this check, **the candidate mechanism is closed**.
- **The missing boundaries are a Pediatric problem of an unadapted ASRF.** ASRF's boundary head proposes no cut near
  the change.
- **They do not explain EV9V's remaining errors.** On EV9V, candidates cover 96–99% of the changes.
- **B3 already recovers them.** With the same adapted head, B3 finds most of the changes ASRF misses.

No model experiment is proposed. The original results, public labels, success criteria and the test split are
unchanged.

The ASRF result describes how an ASRF trained on the source domain transfers after visual adaptation. This check
says nothing about an ASRF given target-domain supervision. Such training is not run, because Pediatric authorisation
is not confirmed (`README.md`, licence).
