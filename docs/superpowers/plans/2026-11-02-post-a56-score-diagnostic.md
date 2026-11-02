# POST-A56-ROC: retrospective score-ranking diagnostic

Owner: the single A57 execution worker. New authorization is the user's 2026-11-02 request to replace the overly conservative strategy with new attempts. This is a **new retrospective diagnostic**, not a continuation/rescue of A55 or A56. Their failed gates and closed outcomes stay unchanged.

Question: Did the old strict-zero operating point hide useful ranking at low A4C false-positive counts, or is the low-FPR ranking itself weak? This helps choose future training and calibration; it does not select a deployed threshold.

Inputs: existing server-only, once-read A55 source score cache and A56 five-logit cache, original133 request membership/labels/patient mapping, original G-C/G-F margins. Verify against each round's committed readout/cache SHA and exact membership order before arithmetic. No raw image decode, new model forward, optimizer update, new source, or protected split access. The sample is ten repeatedly reused patients, not new validation.

Fixed analysis:

1. Reproduce all historical TP/FP/BA/AUC at each original operating point before new arithmetic; stop on mismatch.
2. Enumerate the complete empirical ROC for SF/SC/DF/DC, Z/R and the relevant frozen G reference, using each producer's own raw scalar score (A55 scalar head, A56 q3-q2). Preserve `score > threshold`, ties, +/-infinity endpoints. Never compare the absolute score scales of different heads.
3. Report maximum TP/39 under FP budgets **0, 2, 5, 10, 20** out of94. Require finite scores, preserve ties as inseparable groups, and report attained FP and the count of distinct attainable prediction sets/score intervals satisfying the budget. Real-valued cutoffs within one interval are not different states; do not interpolate fractional TP through a tied group. All maximizing thresholds remain private.
4. Report current TP/FP versus the empirical envelope, A5C/A4C patient/video denominators, and patient concentration of errors at the original point. Do not fit a calibrator or optimize a patient split in this diagnostic.

The envelope uses diagnostic labels and is an **optimistic, reference-assisted upper bound on this already inspected sample**. It is not deployable performance, a qualified candidate, an out-of-fold calibration result or grounds to rewrite a negative round. If an envelope looks promising, a separate target-task model must choose its rule entirely from its own inner training/calibration partition and then be evaluated as declared in A57.

Deliverables: `scripts/diagnose_post_a56_roc.py`, focused synthetic checks for strict ties/budget endpoints/known AUC; public `docs/results/2026-11-02_post_a56_score_diagnostic/{README.md,summary.json,verification.json}`; private arrays/thresholds/IDs stay under a new server namespace. Public aggregates contain no recording IDs. Register the non-A-number diagnostic in the index after completion, not as a new model result.

Budget: at most20CPUminutes and100MBadditional disk, GPU0. Historical costs remain recorded; the new study's actual work is separately charged. Reuse source identity/binding audits rather than rerunning old neural work. Freeze this card and the short analysis code before opening private scores for the new ROC arithmetic. Follow the diagnostic to its checked result without requesting another confirmation.
