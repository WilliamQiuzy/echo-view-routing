# Post-A08 loss diagnostic: frozen protocol (written and committed before any computation)

**Question.** At the V01 start point, A08's penalty is about eight times the cross-entropy, mostly from non-target
classes. Does that scalar dominance also dominate the classification-relevant gradient? This is a hypothesis check on
fixed training inputs. It trains nothing, searches no A08 parameter and does not explain A08's results.

## Inputs (fixed)
- Seeds 0, 1 and 2. For each seed, A07's pinned manifest `VSO_s{seed}` (pins in
  `configs/experiments/video_within_view_v1.yaml`) and its **first 16 updates, u = 0 to 15**, with both groups of each
  update: one native and one constructed group. That gives 32 groups per seed and 96 in all. Nothing is chosen by
  confidence, correctness, loss or segment effect. Every denominator is kept, including groups and edges of each kind.
- Pixels come through A07's loader, and every window's digest is checked against the manifest. Only allowlisted training
  clips are read.
- Model: each seed's V01 Ordered start weights (A07 configuration `source.weights`), strictly loaded. No optimiser, no
  step and no checkpoint is written. Parameter digests, `training` flags and `requires_grad` flags are checked before
  and after.

## Modes
- **Train mode (primary).** As in A08's training: `model.train()` (dropout and stochastic depth active),
  `torch.manual_seed(seed)` and `torch.cuda.manual_seed_all(seed)` once before update 0, updates in order. fp32, TF32 off
  and deterministic cuDNN come from V01's `determinism()`. The encoder is frozen as in warm-up.
- **Eval mode (secondary).** `model.eval()` on the same inputs. The gap between the modes shows the effect of dropout and
  stochastic depth.
- **Reproduction gate.** Train mode, seed 0, update 0 must reproduce A08's check record: CE 0.11502886059133743,
  L_A 1.0649314014820694 and L_M 1.0971463198637705. The values come from A08's module and from an independent numpy
  reimplementation, which must agree within 1e-12.

## Definitions (A08's formulas, unchanged)
ell = log_softmax(z).to(float64); d[n,t,c] = clamp((ell[n,t,c] − detach(ell[n,t−1,c]))², 0, 16) for t = 1, 2 and 3;
S_A and S_M with A08's per-group normalisation; L_X = CE + 0.15·S_X.

**Penalty decomposition (a loss attribution).**
- Edge types: same-label edges (m = 1) and cross-label edges (m = 0).
- Class roles. On a same-label edge with label y, the classes are the true class y and the other four. On a cross-label
  edge (y_prev → y_cur), they are the current label, the previous label and the remaining three. The previous label is
  never pooled with ordinary non-target classes.
- Each element's weighted contribution to 0.15·S_A and to 0.15·S_M, with shares by role and edge type.
- **Truncation.** The share of elements with a raw squared difference above 16, and their share of the penalty.
- **Probability bins.** The current endpoint's class probability falls into [0, 1e-6), [1e-6, 1e-4), [1e-4, 1e-2),
  [1e-2, 0.5) or [0.5, 1]. These are reported only and choose no coefficient.

**Gradient decomposition (logits; a different attribution from the loss shares).**
- **Objects.** g_CE = ∂CE/∂z and g_X = ∂(0.15·S_X)/∂z on the (8, 5) logits, with the earlier endpoint detached. They come
  from autograd and from the closed form 2·w·δ·(e_c − p), which is zero where clamped.
- **Where.** The analysis is at the current endpoints, positions 1 to 3 of each group. Position 0 has no penalty
  gradient by construction.
- **Per current-endpoint window.**
  - The norms, the ratio ‖g_X‖/‖g_CE‖ and the cosine.
  - A conflict means a cosine below 0.
  - A norm below 1e-12 counts as zero. Its cosine is undefined, and such windows are counted separately.
- **Margin.** m = z_y − z_k, where k is the strongest competing class. The first-order change along a descent direction
  is Δm = −(g_y − g_k) per unit step. The penalty opposes the margin when Δm_X < 0.
- **Term attribution.** g_X = Σ_c G_c, where G_c is the softmax-coupled gradient of the class-c term. The sum over each
  role and its contribution to Δm are reported.
- **Batch level.** Per update, the concatenated current-endpoint gradients give ‖g_CE‖, ‖g_X‖ and their cosine.

**Parameters (train mode, start point only).** Per update, the gradients of CE and of 0.15·S_X are taken with respect to
two groups:
- the new parameters (marker stem and classifier), which train during warm-up;
- the encoder, with `requires_grad` set temporarily. This is a counterfactual, because the encoder is frozen during
  warm-up and is past the start point once unfrozen.

Norms and cosines are reported per group. Logit-level evidence alone is never claimed to dominate encoder updates.

**Later training (read only).** The epoch-mean CE, S_A, S_M and penalties of A08's six runs, and 0.15·S/CE, come from
A08's committed `summary.json`. There is no replay.

## Readings (fixed now)
- **R0, scalar dominance.** The share of 0.15·S_X in L_X, reported.
- **R1, no gradient dominance.** In every seed in train mode, the median ‖g_X‖/‖g_CE‖ at current endpoints is at most 1.
  This rejects the motivation.
- **R2, dominant but not opposing.** The median ratio is above 1 in some seed, but in at most half of the current-endpoint
  windows does the penalty reduce the margin by more than CE increases it (Δm_X < 0 and |Δm_X| > Δm_CE). The penalty is
  large but does not act against classification at the start, and the motivation is not supported.
- **R3, dominant and opposing.** The median ratio is above 1 and, in more than half of the windows in every seed, the
  penalty reduces the margin by more than CE increases it. This supports considering one new control. It does not
  explain A08.
- The parameter-level results are reported beside these readings. If they disagree with the logit level, the
  disagreement is stated, and no claim beyond the logits is made.

## Outputs and budget
- **Files.** `diagnostic.json` (aggregates), `README.md` and the script `scripts/diagnose_post_a08_loss.py`. Per-group
  results and the input mapping stay on the server in `runs/post_a08_loss_diagnostic/`. No private id appears in an
  aggregate.
- **Budget.** At most 0.25 GPU hours (estimate 0.03) and 1.0 CPU hour, counting my own wall clock from 20:54 UTC on
  2026-10-17 (estimate 0.85). At 0.95 CPU hours the diagnostic stops, saves what exists and lists what is unfinished.
  It has a new ledger, `runs/post_a08_loss_diagnostic/ledger.jsonl`, and A08's ledger is untouched.
- **Scope.** Single agent, no helpers, no independent fact-check. Small self-checks are not a fact-check.
