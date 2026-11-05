# Capacity and rejection diagnosis

The primary R2 result is negative: 22 of 24 outer cells are no-op, all 24 have zero JEPA-exclusive short16 recovery, and the new A62 confirmation cells are all no-op. The two accepted R2 actions elsewhere repair one frame each without changing segment F1. Preserving the baseline through near-total rejection does not satisfy the joint recovery goal.

The frozen-N-top support analysis separates this failure from an empty action domain. It allows only the already selected original-N-admitted S-L top or no-op on each axis; it never changes the candidate, utility ranking, original N rule, or maximum-one-action constraint. The following are the **joint pooled-F1 selected solutions**, with actual joint effects recounted, under acceptance-safe filtering (H=0, dFP16<=0, no short16 true-unit loss). They are not union-recovery counts.

| A62 outer half | Seed | Base multi F1 | Joint oracle multi F1 | N-admitted tops | Safe available tops | Joint selected tops | Joint JEPA-short16 recovered | H / R | dFP16 / short16 lost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| selection | 2026110461 | 0.909632572 | 0.933333333 | 59 | 49 | 13 | 16 | 0 / 103 | -2 / 0 |
| selection | 2026110462 | 0.909632572 | 0.933333333 | 57 | 51 | 15 | 16 | 0 / 116 | -3 / 0 |
| calibration | 2026110461 | 0.912955466 | 0.933066933 | 75 | 68 | 16 | 15 | 0 / 93 | -3 / 0 |
| calibration | 2026110462 | 0.912955466 | 0.931206381 | 74 | 68 | 15 | 15 | 0 / 78 | -1 / 0 |

These are label-assisted hindsight capacities. Choosing a different safe subset on each axis is not guaranteed to be representable by one global q threshold, and the table does not demonstrate a deployable selection rule. It does show that useful safe increments remain among the frozen tops: R2 rejecting every new-confirmation action cannot be attributed solely to an empty candidate set. A larger full-G oracle domain can additionally differ in utility-top selection and original-N admission, so its gap is not a pure event-risk-loss or context-feature effect.

CAL first-unsafe thresholds are extremely small; R2 accepts no CAL actions for either seed. The unweighted all-candidate BCE training target and the N-admitted-top calibration population differ. Exact saturated q scores and infinite BCE contributions are retained in the published diagnostics. These observations support an unresolved score-ordering/calibration-and-coverage bottleneck, but do not isolate its cause or establish distribution shift, context irrelevance, or a failed model family. H1/H2 positive increment claims are unsupported; a confidence interval containing zero is not evidence of equivalence.

Both exposed development cases A and B are rejected by R1 and R2 for both seeds. Original L0 and recalibrated M0 still accept Case B for both seeds; they accept Case A for seed461 and reject it for seed462. Case A has candidate H40/R0, whereas R37 belongs to other actions in the old cell. Case B has candidate H9/R2 and loses the A4C length11 and A5C length9 true units. Rejecting these two previously inspected cases is a development regression check, not independent confirmation that the new context representation discriminates unsafe from useful safe actions.

All quantities above reuse the immutable public A62 outputs; no new forward, training, threshold, bootstrap, endpoint, or risk-safe oracle was added. The full 156 outer /26 inner /240 paired /6 new+30 reused oracle /48 support /24+2 native /16 regression matrix and both seeds remain authoritative. No A63 is authorized or launched.

Evidence: [full action-space diagnostic](action_space_diagnostic.json), [outer metrics](outer_metrics.json), [development regressions](regression_summary.json), [fixed comparisons and intervals](summary.json), [acceptance verdict](acceptance.json).

Input SHA256:
- `action_space_diagnostic.json`: `74f90cc3fb4f2a519fb22aa33f94d1e0fd2994d818f4a108f0efd952dbe120f8`
- `outer_metrics.json`: `4efa0c9780b24521cfb257aa4b4978e8c280a17a3e63533ce38fea3873d028bc`
- `regression_summary.json`: `70cfb57246001e055b50c25c4b7f9ad7b47372e089690311632a384a8742c5ee`
- `summary.json`: `5991d0467c4add6f3b320e10ced6bdc4b61ee799b97beee44fbde9f3a8ecf86f`
- `acceptance.json`: `5831023fa8bf48b477c850cfc28433caa25b6d9b22d21ebf092cb031b6d16e6f`
