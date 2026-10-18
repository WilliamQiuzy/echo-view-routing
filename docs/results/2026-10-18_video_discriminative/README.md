# A09: cross-session discriminative training on the V01 Ordered video model (2026-10-18)

**Question.** Can stronger cross-session view discrimination on the working multi-frame video backbone beat ordinary
classification training on the same inputs, for native recognition and for full segmentation? This is a controlled check
of a known method, supervised contrast (Khosla et al. 2020, arXiv:2004.11362). No novelty is claimed. The development
halves were reused for exploratory feedback. Nothing here is an independent confirmation. The test split and
EchoNet-Pediatric were not read.

**Design.** The configuration, including the pilot rule, was committed before any code of this round ran
(`configs/experiments/video_discriminative_v1.yaml`, 8a7a1a9).
- **Start.** Every run continues its seed's frozen V01 Ordered state (MViTv2-S, 512-d projection, target marker,
  five-class CLS) on A07's optimiser and 3,750-update schedule.
- **Batch.** Each update has 14 windows. Ten are native windows, two per class, from two different recording sessions.
  The other four are A07's change-containing constructed group of the same update, kept as CE replay.
- **C.** Plain mean CE on this class-balanced, cross-session sampling.
- **D.** The same, plus 0.5 × SupCon on the existing 512-d embedding of the ten native windows (temperature 0.1).
  Positives are same-class windows from another session. Constructed windows never enter the term.
- **λ.** It was fixed once on training batches only. The median ratio of SupCon to CE gradient norms at the start states
  was 1.69, so the predeclared ladder gave λ = 0.5.
- **Pairing.** Both arms use the same manifest, pixels, initial state, optimiser and model random state.
- **Checks.** All passed (`checks.json`):
  - D at λ = 0 reproduces C (identical first logits and loss; after 6 updates, losses within 7.5e-8 and parameters within
    1.0e-6).
  - A 3 + 3 resume reproduces 6 straight updates.
  - The optimiser binding, label isolation and source weights are correct.
  - SupCon reaches the encoder and marker stem but not the classifier.
  - Every anchor has a cross-session positive.
- **Pilot.** Seeds 0 and 2, fixed in advance, each trained to update 1,250 of 3,750. Each checkpoint got one B3 choice on
  calibration, with A08's rule and grid. Both halves were then read.
- **References.** Frozen Ordered and A08's all-neighbour model A were reproduced exactly from their saved outputs. The V01
  source parity on 64 calibration windows was 0.0.

## Pilot result (exploratory; two seeds; the pilot point is mid-schedule)

Definitions:
- N is native five-class macro-F1 of the argmax output.
- E45 and E54 count native A4C clips read as A5C and A5C clips read as A4C.
- F is multi-stream segment F1@50 with B3 at the model's calibration choice. "F fixed" uses frozen Ordered's stored B3
  setting.
- FP and FN are false and missed multi-stream segments.

| System | Mean N | E45 / E54 (sum of 4 cells) | Mean F | Mean F fixed | FP / FN selection (2 seeds) | FP / FN calibration |
|---|---|---|---|---|---|---|
| Frozen Ordered | 0.9781 | 10 / 0 | 0.9661 | 0.9661 | 54 / 44 | 22 / 18 |
| A08 A | 0.9903 | 4 / 0 | 0.9698 | – | 49 / 39 | 23 / 12 |
| C | 0.9858 | 4 / 2 | 0.9439 | 0.9534 | 141 / 55 | 25 / 14 |
| D | 0.9858 | 4 / 2 | 0.9445 | 0.9514 | 134 / 54 | 29 / 15 |

Per cell (seed, half), F of C against frozen Ordered:
- seed 0: selection 0.8989 against 0.9385; calibration 0.9828 against 0.9817.
- seed 2: selection 0.9140 against 0.9644; calibration 0.9798 against 0.9797.

**The frozen rule's decision: stop both arms.**
- **D − C.** Native N, E45 and E54 are identical in all four cells. The mean F difference is +0.0006 and the boundary
  recall difference is −0.0006. Neither signal holds.
- **C − frozen Ordered.** The native signal holds: mean N rises by 0.0077, and E45 falls by 6 while E54 rises by 2. But
  the degradation limit fails. Mean F falls by 0.0222, against an allowed loss of 0.005.

The rule allowed this one pilot. No variant, coefficient change or extra seed follows.

## Reading

1. **Did discrimination improve?** Native A4C read as A5C fell from 10 to 4 against frozen Ordered, at the cost of 2 new
   A5C read as A4C. A08 A reaches the same native A4C/A5C level (4 / 0) without the reverse errors. So against the
   stronger reference there is no native gain.
2. **Did it reach full segments?** No. On selection the C and D systems create about 2.5 times as many false segments as
   frozen Ordered (141 and 134 against 54 over two seeds).
   - Against frozen Ordered, C adds 58 and 56 unmatched predicted segments per seed, of which 36 and 41 have at most 16
     samples. It loses 10 and 15 matched reference segments and recovers 8 and 6.
   - Part of the loss is the decoder working point: the calibration choices took half-windows of 5 to 7 and low
     thresholds. Part is in the outputs: at frozen Ordered's fixed setting, C still loses 0.0127 in mean F.
   - On calibration, the half where B3 was chosen, F is level.
3. **Sampling or contrast?** The native change comes from the class-balanced cross-session continuation, which C and D
   share. The contrastive term at λ = 0.5 changed nothing measurable at the pilot point. Same-class cross-session cosine
   was already about 0.76 in C's first epoch.
4. **Against the two strong references.** Both new arms lose segment F1 to frozen Ordered and A08 A on selection, mainly
   through short false segments. Natively they match A08 A on A4C read as A5C and add two A5C read as A4C errors.

**Limitations.**
- The pilot point is a third of the schedule, with the learning rate at 0.84 of its peak. A fully annealed run might
  segment differently. The rule was fixed with this known, and it is not reopened here.
- There are two seeds, no bootstrap, and reused development halves. Counts are small; E54 changes by 2 clips.
- Sessions are not patients, and constructed streams are not real scans.

## Cost and files

- **GPU.** 1.341 h of 8, within the pilot cap of 2: checks 0.022 h, four pilot runs 1.056 h and pilot evaluation
  0.263 h.
- **CPU.** Booked in `runs/video_discriminative_v1/ledger.jsonl`, counting the agent's active wall clock (see the
  ledger's final lines).
- **Files.**
  - `preparation.json`: pools, manifest counts and repeat disclosures.
  - `checks.json`.
  - `pilot.json`: rows, choices, segment counts, accounting and the rule's evaluation.
  - Code: `scripts/train_video_discriminative.py`, `echo_routing/temporal/losses/supcon.py` and
    `scripts/report_video_discriminative.py`.
- **Server.** The pilot snapshots (`checkpoints/video_discriminative_v1/*/pilot_u1250.pt`) and the exact resume states
  are kept on the server and are not frozen models. Per-window outputs stay on the server.
- **Interrupted task.** The interrupted documentation task before this round is booked separately in
  `runs/independent_eval_requirements_prep/ledger.jsonl`.
