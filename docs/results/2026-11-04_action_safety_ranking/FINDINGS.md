# A61 findings: frame protection improves, ranking benefit remains unestablished

A61 completed the fixed two-seed experiment: eight utility heads and two shared local-risk heads, 6000 updates each (60000 total). Independent audit verified all40 full states,60000 trace rows,24 milestone records,8 fixed thresholds and56 inner policy sets. All314 outer metric rows,240 paired5000-bootstrap comparisons,60 label-assisted oracle domains (including16 exact prior reproductions), and56 native attribution cells are complete. Resource/caller-OS/protected integration closure is recorded separately; scientific completion does not establish the publication goal.

The question was whether safety-shaped utility and whole-query pairwise ranking can retain A08 while extracting JEPA's short/true-change increment. U/S compares fixed-FIT count utility with an explicitly gross-harm-penalized target; P/L compares pointwise MSE with MSE plus all-unequal-target pairwise logistic loss. Same-seed utility initialization and query exposure are matched. E accepts the positive top action, N vetoes that same top using learned gross-count/FP evidence, and C adds the frozen reused-inner first-unsafe cutoff. No fallback, seed/checkpoint selection or outer tuning occurred. Every action retains the original whole-axis majority; that mechanical constraint alone is not frame safety.

## Predeclared primary S-L-C

On the main task, both selection replicas and calibration replica1 exactly match A08 multi segment F1 and short FN/FP. Calibration replica2 changes F1 from98.2692% to98.3670%: +0.0977 percentage points,95% CI[0,0.3026]; short FN5→4, CI for the change[−3,0], FP remains8. This is not a robust main-task segment improvement. Primary native predictions are unchanged position-by-position in both halves and both replicas, rather than merely having zero net error. Other policies' full H/R and native behavior remain visible in the complete matrix.

C is not all-noop: main accepted16/13 actions in replica1 and6/8 in replica2 (selection/calibration), repairing36/37 and12/33 propagated-reference positions with zero gross H. These source-role counts include native/multi/pair axes; primary native changes are zero, so these repairs do not establish new native recognition gains or necessarily recover whole true segments.

On the120-stream-per-half new A61 construction, selection F1 rises from87.6742% to88.7234% /88.2729% (+1.0492pp[0.2372,2.2465] /+0.5988pp[0.1229,1.1757]). Short FN72→66/67 with FP18 unchanged; gross H=0,R=41/44. Calibration rises from87.1579% to87.6050% /87.2766% (+0.4471pp[−0.3530,1.2425] /+0.1187pp[−0.5681,0.7010]); FN77→75/76 and FP13 unchanged. Both calibration replicas nevertheless accept one unsafe action: H=9,R=35/20. Gross new-JEPA short recoveries4/3 coexist with two lost true units; net FN reduction does not remove the gross harm. Thus fewer false alarms or lower total error is not a gross-frame-safety guarantee.

## Factorial controls and the cost of calibration

All four arms, both seeds and all E/N/C policies are exported in [all_policy_diagnostics.csv](all_policy_diagnostics.csv), with full fields in [metrics.csv](metrics.csv). The compact table retains each of the240 new-policy cells across all five tables: accepts, true-unit recoveries/losses, FN/FP, native/multi/pair gross H/R, net errors and top/post-veto regret. The figures show every fixed policy; no favorable arm or table replaces the predeclared primary.

| Fixed segment-F1 contrast | 95% CI strictly positive | Strictly negative | Contains zero |
|---|---:|---:|---:|
| S-L-E vs S-P-E | 0/20 | 0/20 | 20/20 |
| S-L-C vs S-P-C | 0/20 | 0/20 | 20/20 |
| U-L-C vs U-P-C | 0/20 | 0/20 | 20/20 |
| S-L-C vs S-L-N | 0/20 | 15/20 | 5/20 |
| S-L-C vs A60 J-R | 0/20 | 7/20 | 13/20 |

These are descriptive counts of the prespecified conditional, unadjusted intervals, not independent replications or multiplicity-corrected significance. In particular, the ranking-isolating S-L-E−S-P-E intervals all contain zero; A61 does not establish a pairwise-ranking benefit. C sacrifices segment F1 relative to N in15/20 cells and relative to prior A60 J-R in7/20, with no strictly positive intervals for either contrast. Lower coverage and suppressed unsafe actions therefore have a measurable recovery cost. Regret is within-query u/v surrogate regret including a fixed zero no-op; it is not a gap to pooled-bank F1.

## Candidate capacity and remaining bottleneck

All60 G-only oracles use maxONE action plus no-op. Expanded gross-safe ceilings are97.6190%/98.6564% on main, compared with A08 95.6349%/98.2692%. Their candidate unions cover5/7 and3/5 baseline short misses. On the new A61 construction, gross-safe ceilings are93.8017%/93.1174% versus87.6742%/87.1579%, with safe short-union coverage41/72 and44/77. Unrestricted union coverage is57/72 and61/77; some misses remain outside the current proposal domain, and additional recoveries conflict with true gross safety.

These are label-assisted feasibility diagnostics. Union recoveries need not be simultaneously reachable under maxONE, and the safe-only oracle is not an upper bound for an unsafe learned policy. A numerical oracle-method gap includes value estimation, ranking, risk prediction/calibration, and the difference between u/v and globally optimized pooled F1; it is not proof of learnable ranking headroom. The observed H9 false acceptance and the available gross-safe capacity remain unresolved together. No extra arm, threshold, seed or rescue was added.

## Evidence limits and provenance

New heads train only on2736 protocol-native FIT sources plus the fixed1500 A59 constructed FIT bank;628 natives plus300 constructed streams form the reused928 inner diagnostic/calibration axes. A08 historically saw inner sources, so this is not OOF or independent safety validation. The new240 constructions reuse OPEN source recordings; they are new recipes, not independent patients or clinical transitions. Bootstrap units are sessions/fixed streams/fixed pairs, with shared-source conditional intervals; sessions are not confirmed patients. Test/Pediatric/Unity/quarantine were unused. No manual annotation or clinical verification was available. H/R refer to propagated clip-family labels, not expert frame visibility. Features assay posterior trajectories, not latent anatomy or cardiac phase.

The frozen teacher inputs were produced once with6574 original-stride5 A08 windows and1127 JEPA chunks (271.649473 GPU-job seconds), at formal production0, then bound read-only after the10-model actual-Git seal. Formal head training is real CPU NN training (866.484698 wall /669.357365 measured CPU seconds), not backbone fine-tuning. Cached-head timings are not standalone A08+JEPA system latency; [source physical costs](source_physical_costs.json) retain teacher production separately. The failed V1 admission and all measured cost-refinement/RED/failed receipts remain. [Metadata notes](METADATA_NOTES.md) clarify legacy field names without rewriting sealed results. Final actual resources/caller exits and protected integration receipts follow their own immutable records. No automatic A62.
