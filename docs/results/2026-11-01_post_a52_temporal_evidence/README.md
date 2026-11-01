# Post-A52 temporal evidence and source-score audit

Completed CPU-only and independently recounted; zero pixels/model forwards/GPU or new optimization. This is an exploratory analysis of reused source data, not a new A5C recovery/full-segmentation result. A51/A52 failed source protection decisions remain unchanged.

## What was learned

Same source manual marker labels were used for every score, including Wfit outputs (never directory truth for Wonly). Exact global pairwise AUC is partitioned into same-recording and different-recording positive/negative pairs, and separately patient groups. No group-wise threshold/normalization is fitted.

| Diagnostic score | Global AUC,3666pairs | Same-video AUC,117pairs | Equal-video AUC,5eligible videos | Same-patient AUC,337pairs |
|---|---:|---:|---:|---:|
| Original A08 G |.807420|.717949|.701804|.881306|
| A51 precise L |.831151|.880342|.879509|.955490|
| A51 directory W |.819149|.709402|.698665|.896142|
| A52 full detail H |.745772|.632479|.706944|.845697|
| A52 reduced R |.757229|.760684|.793687|.890208|

There are13diagnostic videos/10patients total;5videos and6patients contain both classes and support the respective within-group statistic. Eight single-class videos have **undefined within-video AUC**, not .5. The117pairs share frames/videos and are not117independent observations. Equal-group and pooled AUC are separate. Both partitions exactly reconstruct global AUC from pair masses.26model/role/group partitions independently verified, along with all702×8token labels and physical-reference rules.

L contains a descriptive same-video ranking signal relative to W/G. This does not isolate its backbone from its trained head, establish a nuisance-offset cause, prove anatomy, or justify a deployed cutoff. All global A4C false-positive counts remain G4/94,L23/94,W21/94,H24/94,R21/94. It motivates testing a **different question**, whether frozen representations contain transferable information after discarding source heads, if separately contracted; it does not reopen A51's whole-classifier gate.

## Dense temporal-supervision qualification failed

Use the same sealed702requests with no new frames/patients. Eight even nominal temporal slots are labelled only under exact decoded ordinal/nativePTS and all origin/endpoint hypotheses; NO_VIEW/ambiguity/overlap is UNKNOWN. Informative requests require>=4known slots,>=2labels differing from center, and>=2distinct physical references for each class. Clipped duplicate slots cannot create evidence.

| Role | Requests | Known A4C slots | Known A5C slots | UNKNOWN | Requests containing both known classes | Informative requests / patients / videos |
|---|---:|---:|---:|---:|---:|---:|
| Fit |569|2453|1719|380|63|38 /12 /14|
| Diagnostic |133|675|268|121|25|15 /4 /4|

Distinct physical temporal references3346fit/782diagnostic; adjacent known-label change pairs38/20(overlappingrequests), distinct endpoint pairs34/15. These are endpoint view labels, not validated natural transition-boundary times. The predeclared informative-request minima80fit/20diagnostic fail despite patient/video minima passing. **No dense-temporal model follows; no threshold reduction, resampling, pool widening or relabelled UNKNOWNs.** This closes that input-specific qualification only, not all dense supervision.

## Sources, execution and boundaries

Primary-source refresh is archived in primary_source_refresh.json; no newly admitted adult Bmode LVOT anatomy source was obtained. D01's earlier3Dnegative stays retained. Segmentation of DopplerLVOT/VTI is not a Bmode LVOTmask; public repository visibility without found applicable licensing is not a completed training-source intake. No contacts, protected test, Pediatric/fetal or Unitytraining media operations.

Code/input contract commit3bc64d8 plus precomputation distinct-endpoint count correction0b5168b. Four focused regression tests pass. Independent verifier imports no production audit functions and repeats direct endpoint arithmetic and exact comparable-pair decompositions. Actual audit body.149sec/wholeCPU.230sec; verifier body.495sec plus interpreter CPU recorded server-side. Root/search/reviewer/closing shared phase costs in resources.json; no resource reset, between-turn idle excluded. All IDs/per-case token tables remain server-only. No new model was trained under this audit card.
