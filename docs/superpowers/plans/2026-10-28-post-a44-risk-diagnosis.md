# Post-A44 risk failure diagnosis

Owner: Codex, directly authorized by the user. Diagnostic only, before selecting A45. No Claude or additional training/inference. Budget GPU0, active CPU0.1h, new disk0.01GB, charged to the next-round shared cap (GPU6h/CPU3h/new3GB including failures and closeout). Preserve A43/A44 artifacts and gates.

Predecessors: A44 balanced BCE C vs primal-dual hinge N, both real600updates; N's full-fit H fails in both folds, held-out12/22 fixes/harms for C versus26/76 for N, and independent paired cache audit shows C edits are subsets of N. A43 whole-native D/M differs in supervision and max readout; A30/A31 diagnosed boundary movement and overfit. These facts do not identify anatomy or label correctness.

Permitted inputs: existing A44 source0 update logs, fixed600-step plans rebuilt from immutable A31 fit metadata, final full-fit score caches, frozen opposite-fold score cache and published readout; A31 P membership and original majority clip codes. Server only for private arrays/ids; export aggregate counts/hashes. No model forward, no new examples, no development/check/test/Pediatric, no A26 runner or Unity training.

Fixed analysis:

1. Independently reproduce N's loss derivative and all600 pre-step dual recurrences against logs; summarize multiplier, active/inactive negative windows and score-gradient mass by100-update blocks, including last50.
2. Reconstruct every protected negative's last encounter from the original plan and `r_neg` logs. Compare last-seen score to final whole-fit score, counting last-seen≤−1 but final>0 and final>−1; summarize unseen-since lag. Preserve all cases privately; no post-hoc retraining or chosen checkpoint.
3. Compare final fit and existing held-out C/N score distributions, fixed-zero outcomes, rank AUROC, rank correlation and affine score association. AUROC/rank diagnostics are not deployed thresholds and do not authorize a threshold search.
4. Break down added N-only harms/fixes by original pure/mixed/native/multi strata and source counts. These are observational correlates; do not label them causal anatomy.
5. Record verified facts separately from plausible explanation. A correct implementation can still fail nonconvex finite-horizon optimization and cross-session transfer. Do not attribute failure solely to a larger multiplier or to loss saturation without the measured trajectory.

Output: `docs/results/2026-10-28_post_a44_risk_diagnosis/audit.json` and brief README, script/config/input hashes and actual CPU timing. This is exploratory reuse of A44 outputs, not an independent efficacy experiment. No new A-number or candidate result.
