# Post-A08 loss diagnostic (2026-10-17)

A08's penalty was about eight times the cross-entropy at the start, mostly from non-target classes. This diagnostic
asked whether that is only a scalar size or whether it also dominates the classification-relevant gradient. It is a
bounded check on fixed training inputs at each seed's V01 start point. It trained nothing, changed no checkpoint and
searched no A08 parameter. The rules were committed before any computation (`protocol.md`, 19c56f5).

**What was computed.**
- **Inputs.** For seeds 0, 1 and 2, the first 16 updates of A07's pinned training manifest, with both groups of each,
  give 32 groups per seed. Every native group has 3 equal-label edges. The constructed groups have 32 equal-label and
  16 cross-label edges per seed, and no group lacks an equal-label edge.
- **Forward passes.** Each seed's V01 Ordered start weights were run in train mode (A08's dropout and random-number rule)
  and in eval mode. There was no optimiser or step, and parameter digests and state flags were unchanged afterwards.
- **Reproduction.** Train mode, seed 0, update 0 reproduced A08's check record exactly (CE, L_A and L_M, difference
  0.0). An independent numpy reimplementation matched A08's module within 8.9e-16. The closed-form logit gradients
  matched autograd within 1.4e-8. The penalty gradient at each group's first position was exactly zero, as the detach
  requires.

## What was found (train mode, the primary mode)

Values are per seed, 0, 1 and 2.

1. **The scalar dominance reproduces and is larger than one update suggests.** The median share of 0.15 S_A in L_A over
   the 16 updates is 0.90, 0.98 and 0.98, and 0.15 S_A is 8.6, 49 and 58 times CE. CE is small because the start model
   is confident (median true-class probability 0.998 or more).
2. **What the penalty is made of.**
   - **Roles, A.** Non-target classes on equal-label edges carry 54 to 61%. The remaining classes on cross-label edges
     carry 16 to 20%, the current label 13% and the previous label 8 to 12%. The true class on equal-label edges
     carries 1 to 2%.
   - **Truncation.** Clamped elements are 7 to 11% of the elements but carry 37, 51 and 49% of A's penalty and 23 to 34%
     of M's.
   - **Probability bins.** Elements whose current probability is below 1e-2 carry 74 to 80% of A's penalty and 85 to
     91% of M's.
3. **The penalty dominates the logit gradient in size.**
   - At the current endpoints, the median ratio of A's penalty gradient to the CE gradient is 123, 150 and 261 per
     window, and the ratio is above 1 in 84 to 94% of windows.
   - Pooled over each update's current endpoints, the ratio is 3.4, 10.8 and 37. CE's gradient is concentrated in a
     few uncertain windows.
   - M's ratios are similar (134, 157 and 254). Its gradient is exactly zero on the 16 cross-label windows per seed.
4. **Its direction relative to classification is mixed.**
   - For A, the median cosine with the CE gradient is +0.30, −0.38 and −0.17, and 42, 55 and 54% of windows conflict
     (cosine below 0). Pooled per update, the cosine is near zero (0.07, 0.05 and 0.10).
   - A's penalty lowers the true-class margin in 42, 56 and 53% of windows. It lowers the margin by more than CE raises
     it in 34, 51 and 48% of windows. For M these shares are 29, 42 and 41%.
   - Summed over windows, A's first-order margin effect is +0.76, −0.65 and +0.06, against +2.73, +1.45 and +1.15 for
     CE. Its sign changes between seeds.
5. **Which terms carry the gradient.** The terms of the softmax-coupled gradient differ from the loss shares above.
   - Non-target terms carry almost all of the penalty gradient in their windows.
   - On cross-label edges, the previous-label term carries a median 22, 41 and 25% of the window's penalty gradient and
     lowers the current margin in 83 to 100% of those windows.
   - The true-class term contributes almost nothing.
6. **Parameter gradients (start point only).**
   - **New parameters** (marker stem and classifier, the set trained during warm-up). The norm of A's penalty gradient
     is 1.8, 7.4 and 7.4 times CE's, with cosines −0.01, +0.01 and −0.16. The cosine of the total gradient with CE's
     is 0.65, 0.28 and 0.73.
   - **Encoder.** This is a counterfactual, because the encoder is frozen during warm-up. The norm ratio is 1.5, 4.2 and
     6.5, and the total gradient's cosine with CE's is 0.71, 0.31 and 0.75.
   - At the start, then, the penalty also dominates the parameter gradient in size and points nearly orthogonally to
     it. The total update direction turns away from CE's.
7. **Eval mode.** Without dropout, seed 0's penalty is smaller (median S_A 1.93 against 3.13), so dropout noise between
   windows adds to the train-mode penalty. The direction readings are similar: A's dominant-opposing share is 36, 47 and
   50%.
8. **Later training (read from A08's monitors, no replay).** The epoch-mean ratio 0.15 S/CE was 2.2 to 2.9 under A and
   1.8 to 2.2 under M in the first epoch. In the last epoch it was 1.15 to 1.37 under A and 0.69 to 0.87 under M. Epoch
   means are dominated by harder updates, and the penalty falls within the first epoch. So the start point overstates
   the ratio, and it does not describe training as a whole.

## The fixed reading

**R2 for both arms: dominant but not opposing.**
- The median gradient ratio is far above 1 in every seed, so R1 (scalar size only) is rejected. The large penalty is not
  merely a scalar.
- R3 needed the penalty to outweigh CE against the margin in more than half of the windows in every seed. This held only
  for A in seed 1 (51%). The penalty dominates the logit and parameter gradients in size, but mostly orthogonally to CE,
  and it opposes the margin in roughly half of the windows. It is not systematically against classification at the
  start.
- **What follows.** The initial motivation, that the non-target-dominated penalty works against classification, is not
  supported by this diagnostic. Nothing here explains A08's results.

**Three kinds of statement.**
- **Provable from the formulas.**
  - Both logit gradients sum to zero over the classes.
  - CE's descent always raises the margin.
  - Position 0 receives no penalty gradient.
  - M's penalty gradient is zero at cross-label windows.
  - On a cross-label edge, when the current window gives the previous label a lower log-probability than the previous
    window did and the true class leads, the previous-label term lowers the margin.
- **Observed on these 96 groups.** Everything in "What was found". These are start-point values on the first 16 updates
  of each seed's manifest.
- **Unverified conjectures.** The penalty's large component orthogonal to CE may slow or redirect CE-driven learning
  early on. Part of the train-mode penalty reflects dropout noise. Neither is tested, and neither is linked to segment
  outcomes.

## Cost and files

- **Ledger.** `runs/post_a08_loss_diagnostic/ledger.jsonl` on the server; A08's ledger is untouched. GPU 0.021 h (75 s)
  of 0.25. CPU, counting my wall clock, is booked at the end (see the final ledger line).
- **Files.** `protocol.md` (frozen rules), `diagnostic.json` (all aggregates by seed and mode, including the later-training
  monitors) and `scripts/diagnose_post_a08_loss.py`. Per-window records and the input mapping (manifest group indices)
  stay on the server in `runs/post_a08_loss_diagnostic/private_windows.json`.
- **Scope.** Single agent, no helpers and no independent fact-check. The reproduction gate, the numpy reimplementation
  and the closed-form gradients are self-checks only.

## Dated note on the R2 criterion (2026-10-17, added after the lead's review; the frozen protocol is unchanged)

The protocol's original wording, committed before any computation (19c56f5), is quoted here:
- R2: "The median ratio is above 1 in some seed, but in at most half of the current-endpoint windows does the penalty
  reduce the margin by more than CE increases it (Δm_X < 0 and |Δm_X| > Δm_CE)."
- R3: "The median ratio is above 1 and, in more than half of the windows in every seed, the penalty reduces the margin by
  more than CE increases it."

**The uncovered case.** The wording does not say how to read a set where the share is above half in some seeds and at
or below half in others. For A this happened: 51% in seed 1, against 34% and 48% in seeds 0 and 2. The script, written
before any computation, assigned R3 only when every seed is above half and R2 otherwise. A therefore received R2. That
rule is kept. It is not changed to fit these numbers.

**How R2 must be read.** R2 here means that the cross-seed criterion for systematic opposition was not met at the fixed
start point. It does not mean that the penalty never opposes classification. In seed 1 it outweighed CE against the
margin in a little over half of the windows. The start-point counterfactuals do not describe training as a whole.
