# A61: within-sequence ranking and gross frame-harm prediction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans task by task. One server executor only: `training_execution`. Independent plan/SPEC/QUALITY/scientific reviewers are read-only and never start server jobs. The user explicitly authorizes A61 implementation, repairs, necessary checks, the complete admitted matrix, round-only commits and protected public integration without ordinary confirmation. This supersedes A60's noautomaticA61 limitation for A61 only. No automatic A62.

**Goal:** Test separately whether direct within-sequence ranking, a safety-shaped action target, and local gross frame-harm prediction can retain JEPA short-segment recovery without A60's extra wrong reference positions.

**Architecture:** Keep A60's exact expanded single-action grammar, frozen A08+B3 default and whole-axis majority guard. Train a 2×2 matched utility experiment (raw versus safety-shaped target; pointwise versus pointwise+pairwise loss), and two shared local-risk models. Every policy first chooses the same highest positive utility action; risk policies can veto it but never fall through to a different action. A fixed inner-only calibration rule strengthens the empirical veto. Freeze all models, calibration thresholds and endpoint identities before any A61 outer outcome.

**Tech Stack:** Existing Python/NumPy/PyTorch CPU FP32, immutable qualified posterior caches, full-axis five-class IoU50 matcher, actual-receipt single-queue supervisor, JSON/CSV and standard scientific PNG/PDF. Existing machines/environments only.

## 1. Evidence and falsifiable changes

Source of truth: `docs/EXPERIMENT_INDEX.md`, historical `docs/EXPERIMENT_LEDGER.md`, and corrected A57/A58/A59/A60 reports. A57's temporal heads, A58's structured decoder and A59's restricted action domain did not establish stable complete-task improvement over A08. A60 expanded useful candidate coverage, but learned value/risk decisions remained weak. Routing was already studied in A14–17; neither routing nor RankNet is novel here.

A60's 124 rows/144 comparisons/32 oracle domains/24 native attribution cells are complete and closed. New-construction G-oracle rose to95.441/95.409%, but it did not constrain actual frame or false-short harm. On main, all four J-R−A08 segment-F1 CIs include0. J-R native majority classes are unchanged mechanically, yet wrong reference positions increase25/32 and21/18. Actual A5C→A4C edits show (a) u=0,dFP16=0,dE=+25 mispredicted as beneficial and safe, and (b) u=+.891745,dFP16=−1,cuts−1,dE=+18. Thus the old count target itself can reward consolidation of wrong frames, while aggregate signed-risk regression also fails. These are reference-label, not expert frame-visibility, errors.

Changes in A61, all prospectively fixed:

1. Distinguish gross newly-wrong positions H from gross repairs R, instead of relying only on their difference dE=H−R.
2. Make harmful edits negative in a separate safety-shaped utility target; raw-count controls remain.
3. Compare within-sequence pairwise ranking plus an absolute pointwise anchor against matched pointwise regression. [RankNet](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/) is the existing probabilistic pairwise-loss precedent, not a novelty claim.
4. Train local changed-position harm/repair/neutral classification and a separate false-short-increase classifier; test an inner-calibrated veto without claiming a finite-sample guarantee. This is **not** [Learn then Test](https://arxiv.org/abs/2110.01052), conformal risk control, or independent clinical validation.

Hypotheses may fail: H1 safety-shaped training avoids proxy-only consolidation; H2 ranking reduces action/no-op regret; H3 local evidence and calibration reduce false-safe accepts while retaining nonzero true-short recovery. Fewer cuts, unchanged majority, lower training loss, or no-op everywhere cannot independently satisfy the goal.

## 2. Workspace, sources, roles and immutable scope

Only worktree `/Users/williamqiu/.config/superpowers/worktrees/echo-view-routing/a61-action-safety-ranking`, branch `a61-action-safety-ranking`, parent `0c3c7a52c395a4764976ba9a3f7c6ba0c73c381a`. Birth1793771887.0295548. Server `ssh echo-l40s`, project `/home/william/echo-view-routing`, new namespace `runs/a61_action_safety_ranking_v1`; A57/A58/A59/A60 namespaces/config/source/receipts are read-only. Never import an old CLI that writes its own PUB/RUN at import. Pure qualified libraries may be reused without broad re-audit.

B is original A08-s0 stride5+B3, source `models_frozen/video_within_view_v1/all_s0/final.pt`, SHA256 `9b144925f32fac7d725d13231a132abeb6635fa44b6fe0dbcec7f5b59d035bbc` (400 keys). Preserve original16frames/marker/gamma/256crop224/FP32, B3 half_window7/threshold.19660949291632382/nms5/quantile.999. J is A57 core J-high6000 `runs/a57_dense_sequence_v1/seed2026110257/J-high/step6000.pt`, SHA256 `2fa26f2b3ffbffd4d8481c6ac1ff2190572f18648e13f13dbf37333a17bdd985` (214 keys), original64/stride32/triangle stitch. Family order is `[PLAX,PSAX,A4C,A5C,SC4C]`.

Reuse A59/A60 qualified original-stride5 pA/pJ, B, full-axis labels/manifests: FIT2736native/402sessions+1500mixed; inner628native/100sessions+300mixed. Keep complete FIT/inner source-footprint separation. All FIT candidates, targets, position normalizers and RMS scales are FIT-only. No new FIT bank, old-source neural qualification or old-posterior forward. A08's historical inner exposure remains; no full-model OOF claim.

**A61 changes the role of inner explicitly:** all928 inner sequences are diagnostic/calibration data for the fixed final calibration rule below. They are no longer an independent evaluation of the selected veto, and no checkpoint is selected. Risk/utility gradients never use inner. Aggregate reporting names this role change, source reuse and historical development exposure.

Tables: main old10..40 task; retrospective A58stress; retrospective A59confirmation; retrospective A60confirmation; new A61confirmation. Freeze120 new construction recipes per half before the first head result, with seed2026110461, existing4..min(80,n) support,4..8fragments,gamma090prob.5,p_same_next.35; generator RNG domains `[2026110461,61,2]` and `[2026110461,61,3]`. Actual cross-join forward is mandatory for these240 new streams; never splice native logits. Constructor reading authorized class metadata and generating labels is a construction-only exception, not an assertion that labels were unaccessed. No outer utility/metrics before all model/calibration freezes. New constructions reuse OPEN recordings, not new patients or clinical holdout.

No old test/sealedtest/Pediatric/Unity training/21 quarantined patients. No raw/IDs/weights/optimizer/per-case arrays outside server. No webpage edits, push, external contact or paid service. No human annotation is available; do not invent it or use that as a routine approval blocker.

## 3. Fixed action grammar and targets

Keep A60 `proposal_actions` exactly: old RA!=B maximal components plus RJ maximal constant-class run replacements; trim only unchanged ends, retain unchanged interiors, deduplicate by complete edited-axis canonical key. No duration/confidence/newcut threshold, boundary expansion or source/kind/GT inference input. All sequences use G: edited whole-axis `bincount5.argmax()` equals B's, same class-order ties. At most one action per complete sequence.

For candidate a and full reference y, using identical full-axis class-aware IoU50 one-to-one matching:

```
H = count((B == y) & (edited != y))
R = count((B != y) & (edited == y))
dE = H - R
dFP = short_FP16(edited,y) - short_FP16(B,y)
u = 2*(M_after-M_before) - F_fit*(P_after-P_before)
safe = (H == 0 and dFP <= 0)
v = u if safe else -1 - abs(u)
```

F_fit is unchanged Base pooled F1 on the complete1500 FIT multi bank, independently recomputed/bound; expected5482/5916/6379=.8917446116307443. u retains its exact marginal numerator meaning at that FIT operating point. v is a **deliberate safety surrogate**, not exact F1 utility or a clinically established cost. Its fixed−1 margin is not searched. H=0 forbids trading a new wrong reference position for a repair elsewhere; it may sacrifice useful net improvements. The net-safe and gross-safe oracles below quantify this tradeoff. H=0 does not correct Base's retained errors or establish expert anatomical validity.

No-op has fixed target and model score0, H=R=dFP=0; it is never a trainable token. Every G candidate is retained, including unsafe/negative/zero candidates. No truth-based filtering of the inference domain. Empty candidate lists return B without a nonempty-normalizer/model call.

## 4. Matched utility experiment: eight models

Use A60's **J109 features unchanged**, in the exact existing order. These are common29+A40+J40. No source provenance/kind/ID/label/true join feature. All arms use the same FIT109 mean/std normalizer (floor1e-6), computed on the identical complete G candidate set.

Utility MLP109→32 GELU→1,3553parameters, output layer zero initialized. Four arms per seed: `U-P`, `U-L`, `S-P`, `S-L` (U target u; S target v; P pointwise only; L pointwise+pairwise). Seeds2026110461/2026110462. Per seed all four utility initial state dicts and draw/exposure plans are identical. Two independent shared-risk models are described below; utility losses cannot update them.

One common uncentered utility scale for every arm/seed:
`s=max(sqrt((mean(u**2)+mean(v**2))/2),1)` over all G FIT candidates. Physical utility predictions are network output×s. P loss is MSE to u/s or v/s. L loss is that exact MSE **plus** pairwise logistic loss with coefficient1, no grid:

```
z = concat(predicted_scaled_scores, [0])
t = concat(chosen_scaled_targets, [0])
pairs = all unordered (i,j) with t[i] != t[j]
rank_loss = mean(softplus(-sign(t[i]-t[j])*(z[i]-z[j])))
```

Tie pairs are excluded using the stored exact target values; a group with no informative pair contributes rank_loss0. Pointwise MSE averages only real candidate scores; fixed no-op has no MSE parameter. Within each source sequence average candidate MSE and informative-pair loss, then average source sequences within each nonempty kind and kinds equally. This keeps long lists from silently dominating. Label ties remain ties; canonical time/key tie-break applies only to inference.

Each update draws32 nonempty native sequence groups and32 nonempty constructed sequence groups, with replacement uniformly among groups of that kind. Use every G candidate of each drawn group; explicitly trace actual lists/candidates/positions/pairs, not a fictitious64-candidate exposure. No-op-only sequences remain in all reporting/domain denominators. If one kind's nonempty group pool is empty, fill both32 blocks from the other pool with distinct planned draws; record the fallback. If both empty, stop as structural impossibility. Seeds/plan keys must be canonical integers, no coercion that folds aliases.

All eight utility models run6000 optimizer steps =48,000 utility updates. Same AdamW1e-3/betas(.9,.999)/eps1e-8/wd.01/warm100/cos6000/FP32clip1 as A60, CPU only. No LR/width/coefficient/target-margin/threshold search or early stopping.

## 5. Local risk models: two shared models, 12,000 updates

For each changed position t only, train a mutually exclusive label:
H if B[t]==y[t]; R if edited[t]==y[t]; otherwise neutral. Because t is changed, H/R cannot both hold. Original-axis labels only; no padded/sampled-interpolation truth.

Let T be the full unpadded original-axis position count; [s,e) the canonical action interval, action_length=e−s; K=count(edited!=B) over that complete axis (changed-position count, excluding unchanged interior positions). K is neither candidate count C nor run count. For each action K>0. 47 observable features per changed position, in this fixed order:

1. pA[t]5, pJ[t]5, B[t] onehot5, proposed[t] onehot5 =20.
2. pA means over `[max(0,t−16),t)` and `(t,min(T,t+17))`; same pJ means =20 more. Empty flank means are zero; append two flank-availability flags =42.
3. `log1p(T),log1p(action_length),log1p(K),(t−s+.5)/action_length,(t+.5)/T` =47.

FIT-only per-position47 mean/std floor1e-6, fit over the complete changed-position training multiset; repeated positions across candidates are not independent observations. No source ID, sequence kind or proposal-origin feature. Position network47→32GELU→3softmax,1635parameters. A separate candidate network uses the same109 normalized features,109→32GELU→1sigmoid,3553parameters, target `I(dFP>0)`. Combined risk model5188parameters, both output layers zero initialized; one AdamW over both modules with complete named-parameter binding.

Risk loss per candidate is mean position CE + candidate BCEWithLogits; mean candidates within each kind, then equal kinds. No class weighting/search. Separate deterministic risk plans draw32native+32constructed **candidates** per update from complete eligible pools, including all changed positions of each drawn action. Empty-kind fallback follows the same explicit contract. Each risk seed6000updates,12,000 total; risk models are shared identically by all four utility arms of that seed. Utility and risk plans are distinct and both frozen; risk learning never sees utility predictions, inner labels or outer labels.

At inference sum position pH/pR as predicted gross counts and net Hhat−Rhat; report these separately. `r=max(max_changed_position(pH), sigmoid(FP_logit))` is an **uncalibrated evidence score**, not a probability of any harm (no independence product claim). All scores must be finite and in range; no clipped nonfinite repairs.

## 6. Same-choice policies and calibration freeze

First choose a_star = largest physical utility strictly>0 among G candidates, canonical action key for ties. If none, return B. Never re-rank or fall through after risk rejection. The three policies share one utility/risk forward journal:

- E: accept a_star.
- N: accept a_star only if predictedH−predictedR<=0 and pFP<.5, else B. This is an empirical point estimate veto, not a guarantee.
- C (primary): N plus `r < tau` using the fixed inner-only rule below. It is nested within N and uses no task/kind/ID/GT switch at inference.

After all final6000 weights are frozen, compute each utility arm/seed's chosen a_star on all928 inner sequences. For N-admitted tops define unsafe=H>0 or dFP>0 from complete-axis inner reference counts. Set tau to the **minimum r among unsafe N-admitted tops**; if N admits tops but none unsafe, tau=1; if N admits no tops, tau=0. Strict `<` excludes all tied unsafe cutoff scores. Use all native and constructed inner samples together under the same rule, without maximizing F1 or searching seeds/checkpoints. Publish counts/ties/threshold/hash/no-op degeneracy. Verify zero observed unsafe C accepts on these calibration sequences. This is a construction of empirical calibration safety, **not independent evidence of generalization**. Repeated sessions, candidate selection, weak labels and prior development exposure prohibit an exchangeability/finite-sample/conformal claim.

Risk models finish before utility inner readouts. Utility steps2000/4000/6000 have24 inner records with E/N diagnostics =48 policy sets. Final C calibration/readouts add8 conditional sets (56 total); no C at earlier checkpoints and no checkpoint choice. All10 final weights,30 milestone full states+10latest, all24inner records,8 tau values and endpoint plan are actual-Git SHA-manifest frozen before any new outer outcome. Full checkpoints intentionally include Adam/RNG and are budgeted as such; no A60 weight-only wording deviation.

## 7. Exact scientific matrix and conclusions

Every row uses fixed6000. Tables5×halves2:

-8 utility arms×3policies×10cells =240 policy rows.
- Base/original-raw/JEPAraw controls: main10 reference rows including historical A08s1/s2; other four tables6 each =34.
- Frozen A60 J-E/J-R, both old seeds360/361, on all five tables =40 historical-reference rows. No A60 retraining. Map old360→new0461, old361→new0462 by replicate index; do not call these equal-seed training comparisons.

**Total314 metric rows.** Stress native/pair predictions reuse the main native/pair axes exactly, only multi changes. Use decoded occupancy/class-order ties for primary native classification. No selection of the better seed/table.

Twelve prespecified contrasts per replicate×2halves×5tables =**240 comparisons**,5000 paired whole-session(native)/whole-stream(multi)/whole-pair replicates, using existing cluster/shared-source caveats, undefined numerator/denominator/null accounting:

1 S-L-C−Base; 2 S-L-C−A60J-R; 3 S-L-C−S-P-C; 4 U-L-C−U-P-C;
5 S-P-C−U-P-C; 6 S-L-C−U-L-C; 7 S-L-C−S-L-N; 8 S-L-N−S-L-E;
9 U-P-N−A60J-R; 10 S-L-E−U-P-E; 11 S-P-E−U-P-E; 12 S-L-E−S-P-E.

Ungated contrasts11/12 isolate target/ranking while C contrasts additionally include the prespecified calibration response; historical9 is a package comparison, not isolated per-position-risk causality. Preserve all systems and intervals, no multiple-endpoint win by cherry-picking. Primary scientific package is S-L-C, but all hypotheses require their matched contrasts and nonzero useful recovery.

All metrics from A60 remain: five-class native macro/confusion, reference-position confusion/accuracy, gross H/R/net per class, native cuts, A4C↔A5C, multi/pair segmentF1@50 and full M/P/T, boundaryP/R/same-view joins, short<=10/16FN/FP/gross recovered/newly lost/net, recovery/harm/veto, accepted action lengths/changedcount/fraction/fullaxis/newcuts. Explicitly report any native H>0 or short-FP increase even if macro/cuts improve. Majority protection remains only mechanical. Add fixed5 equal-width risk-score reliability bins, actual H/R versus predictions, positive safety-surrogate false accepts, within-query/no-op pair accuracy and utility regret, using outer labels **only after freeze for evaluation**, never recalibration.

**After all314 system readouts freeze**, compute60 exact label-assisted pooled-multi F1 oracle cells: old/expanded domains × unrestricted/net-safe/gross-safe ×5tables×2halves, always G/maxone+noop. net-safe is actual dE<=0 and dFP<=0; gross-safe is H==0 and dFP<=0. Unrestricted matches A60's corresponding16 old four-table G domains exactly; do not require old U domains or relabel scope. Audit domain and safety monotonicity, full matching interactions and empty cases; union short coverage is not joint attainability. No oracle labels feed models/calibration/candidate changes. Native per-action gross/net effects are diagnostics, not a fabricated native macro oracle. Store full joint5×5×5 native transitions for48 A61 mainpolicy cells +8 A60 reference cells =56 attribution cells, with exact gross-cost and predicted-gate recounts.

An acceptable outcome needs actual stable native positions (not just majority), controlled short FP, and retained true-short benefit relative to Base and A60 on fixed tables/replicates. No consistent difference is not equivalence. A positive new-construction conditional result is not new-patient or natural-clinical-transition evidence. No automatic follow-up model/threshold arm if these fail.

## 8. Actual resource envelope and runtime admission

Original shared caps/deadline remain:24GPUjobwallh/72 summed measured workerCPU accounting hours/40,000,000,000B tracked-path conservative upper; origin1793648044.1596537, deadline1793820844.1596537. Use earlier original deadline; do not reset after reboot/new round. CPU-only stages are not artificially bounded by GPU reserve.

Inherit A60 `resources.json` SHA `38a826a32c4de34bd92c411ef31d92c3e0bf6e1dc0a26bdfb9182d4f47c28103`: GPU18.41335994296604h; summed measuredCPU18.448515337222222h; CPUaccount19.198515337222222h (old separate.75h allowance retained); storageupper35,332,702,254B. Start remainder GPU5.586640057033961h **including2h closingreserve**, CPUaccount52.80148466277778h, storage4,667,297,746B. A60's measured collector/parent administrative tail.052479s remains separate and must not disappear or be recharged as a model job. Root/modelreading/integration overhead and inherited known setup/CPUtail/overlap/unknown historicalOS limits stay separately explicit.

Initial read-only local SSH probe receipt `initial_live_probe.json` has .053539 SELF+waitedSSHclient CPU seconds; remote probeCPU was not independently measured. Include measured part exactly once, preserve its scope. Every new local test/review/failedstage/engineering/source/calibration/formal/readout/bootstrap/oracle/audit/package/collector has actual rusage receipt or explicit unmetered limitation. Never count forecasts/reserve as actual, never infer OS0 from completion manifests.

New supervisor uses the A60 robust pattern: same single A61 queue flock, lock-held reload of the authoritative actual ledger and local costs, durable launch intent **before** Popen, PID/starttick identity, atomic/fsync claims/checkpoints, ENOENT-safe storage scan, bounded deadlines and fail-closed cleanup/wait. No second GPU queue or guardian. CPU imports set CUDA_VISIBLE_DEVICES='' before torch and empirically check no context. Unexpected child CUDA context immediately failclosed and charge whole occupied wall once. Foreign work is identified/read-only and never killed or billed to A61.

Before formal admission, freeze the new graph and run only its focused tests and actual cached FIT check: full target identities,47/109 schemas, no-op/ties/unsafe proxy fixture, real all-tensor finite historical gradients/drift/nonzero Adam moments, exact repeat and diskresume, canonical plans, actual64group utility and64candidate/allchangedposition risk throughput **including durable I/O**. Probe worst actual FIT K,C,C²,T×C and structural new-confirmation bounds without outer head outcomes. Engineering updates are separate from60,000 formal updates. No whole-old-library/source parity rerun just because docs changed.

Measured full forecast must cover every already-spent failure/preparation,2risk+8utility×6000, all inner/calibration checkpoints, new240-source cache,314rows,240×5000CIs,60oracles+16reproductions,56nativeattribution cells, plots, independent audits, full checkpoint/storage atomic margins and actual closure/main sync, plus2GPUh reserve. No pilot-only admission or omitted mandatory matrix. If any hard cap/deadline fails, stop/close honestly; no silent endpoint reduction. Once admitted, immediately continue all sequential stages without waiting for user or heartbeat.

## 9. Implementation file map and executable task order

Create new config `configs/experiments/action_safety_ranking_v1.json`; new pure modules `echo_routing/temporal/safety_{targets,features,ranking,risk,training,policy,data,readout_plan,resources,runtime,audit,oracle}.py`; new CLIs under `scripts/*action_safety*.py` plus `scripts/safety_common.py`; matching `tests/test_safety_*.py`. Reuse pure action/matcher/cache functions; never change completed A60 modules/identities. All new code/inputs read from this round's namespace. Exact module splits may combine tightly coupled lifecycle helpers, but contracts and no-old-write boundary cannot change.

### Task1: Freeze research, inputs and resource inheritance
- [ ] Save readable RESEARCH_REPLAN/README/pending index, birth/main preservation and this full card; round-only commit.
- [ ] Single fresh whole-plan independent review; resolve real contradictions, retain review and actual CPU scope. SPEC before QUALITY for new implemented graphs; reviewers never execute server jobs.
- [ ] Freeze config and exact314/240/60/56 endpoint plan. New240confirmation recipes before any head result; qualify only reused caches/manifests/axis/classmap and current live queue, no old forward.

### Task2: Full-axis effects, utility labels and position features (TDD)
- [ ] Write failing tests for u>0/cuts−1 yet H>0 fixture, net repairs cancelling new harm, H/R/neutral exclusivity, whole-axis matching/dFP effects, v alwaysnegative on unsafe and exact u on safe.
- [ ] Run metered RED; implement minimal target and47features/J109 reuse; test boundaries/emptyflanks/longaction/interior unchanged/classorder/GT exclusion and exact dimensionality; run metered GREEN/commit.
- [ ] Independent scoped SPEC then QUALITY; fix narrow issues before dependent real graph. Do not re-audit old qualified libraries.

### Task3: Utility ranking, local risk and lifecycle (TDD)
- [ ] RED tests: MSE equality for matched arms, all unordered unequal pairs including fixednoop, tie/noinformative list, group weighting, shared initial state/plan, zerooutputinitialization, physical unscaling and no architecture/capacity confound.
- [ ] RED local-risk tests: mean-per-action CE/BCE, gross-count aggregation, no probability-of-any-harm claim, FP strict.5 threshold, exact named optimizer state/seed/cursor/RNG and repeat6/resume3+3 across utility/risk models.
- [ ] Implement minimal models/losses/durable full checkpoints and physical journals; pass GREEN; fix pending update/readout ambiguity, directoryfsync and canonicalseed rejection. All 10 models6000, actual Adam counter checks; no uncheckpointed replay.
- [ ] Scoped independent SPEC→QUALITY; retain failures and their measured costs.

### Task4: Calibration and exact endpoint graph (TDD)
- [ ] RED for top-first/no-fallback (unsafe top must not choose runner-up), C nested in N, alltypes same inference, strict ties at firstunsafe, nounsafe tau1, noadmitted tau0, allnoop and float saturation.
- [ ] Implement whole928 calibration,8 thresholds and56conditional inner sets; demonstrate zero unsafe accepted on calibration without claiming independent safety. Freeze all calibration provenance.
- [ ] RED→GREEN exact314rows/240contrasts/60oracles/56native cells coverage, unique systems/weights, old-reference replicate mapping and stress reuse; independent SPEC→QUALITY.

### Task5: Real cached FIT, complete admission and formal execution
- [ ] Implement A61 paths/supervisor/receipt ledger/resource inheritance and cheap failed/sleep/lock tests; reuse qualified A60 source definitions without importing writing CLIs.
- [ ] Actual metadata/recipe freeze and full FIT target/features/group/risk plans; cached new-graph liveness/repeat/resume/I/O/worstcost measurement, CUDA hidden/no context. All spent costs and full remaining forecast admitted under original caps/reserve/deadline.
- [ ] Directly run two sharedrisk6000 then eightutility6000 in same queue. Latest100/fullmilestones2000/4000/6000; record realPID/startticks, physicalupdates, exact exposures/Adam, engineercounters separately. No ordinary confirmation.

### Task6: Pre-outer barrier, sources and complete scientific matrix
- [ ] Independently audit all10finals/24inner+8calibration/56sets/8taus/fullplans and score/noop contracts. Git SHA-manifest freeze models/endpoints/calibration before new outer output.
- [ ] Admit and forward only missing A61confirmation A08/J contexts; complete cache/input/journal/no missing/duplicate/changedSHA axes. All old tables read-only cache reuse; A60J-E/R weights/predictions bound as historical controls.
- [ ] Execute314 rows and56native cubes, freeze all systems; independent private original-axis recount and240×5000pairedCI; then60exact oracles and16oldG reproductions. Preserve undefined ratios/nulls and all negatives. No after-result tuning.

### Task7: Closure, independent audit and protected integration
- [ ] Standard scientific figures: matched effects, shortFN/FP versus grossframeharm, riskcalibration/acceptance, safe-domain ceilings and training/ranking curves; visually inspect exported PNG/PDF artifacts.
- [ ] Full self-contained report question→attempt/config→results→limits→next evidence; isolate target/ranking/policy effects and quantify noop degeneracy/sacrificed recovery, not just fewer cuts/macroprotection.
- [ ] Independent final scope/denominator/weights/calibration/label/risk/native/oracle/actualcost/OS audit. Direct parentwait statuses, current PID/start absence, pending0/lockavailable/GPUrelease, all actual workerCPU and legacy uncertainty retained. No manifest-derived OS success.
- [ ] Exact public/source/test/config allowlist, private weights/percase stayserver. Root selective main sync only A61 new files, append Index preserving current prefix; protect mainHEAD2f2f57b4092a98d76428894a387b8f05926933e7,8stagedbinarySHA `a9cab9907c011551149b26d9331f86d0f99776eff29f022362849b64effd5ff8` and467originalpaths via `/tmp/a61-main-preservation-before.json`. No broad staging/index overwrite/push/webpage.
- [ ] Only after every accepted stage/actualclosure/publicintegration is done (or hard-limit/external honest fullstage termination), delete only capstone-a61 monitor and state stopreason. Do not remove on core training completion; no automatic A62.
