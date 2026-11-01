# A51 direct marked-frame view supervision — closed source negative

A08+B3 remains the working model. A51 trained two real600-update video models, but neither met the source A4C protection condition; **no new EV9V target model or103/1413/full-segmentation readout was run**. Source classifications below are not target-run recovery counts.

## Question and changed evidence

After A43–A48/PW-T1-C repeatedly traded A5C recovery for A4C damage, test direct view supervision rather than another loss, veto or weak clinical-purpose label. [EchoUNAL](https://gitlab.com/JSarmientoA/echoUNAL) release5e853ea4bb3eee8bb103fa44ad6ddb13fc63a5eb supplies manually relabelled frame intervals with real A4C/A5C disagreement inside directory labels. Dataset license CC-BY-NC4.0,89consented healthy volunteers,346published six-view videos; full metadata364rows includes unreleased/unannotated entries. This is view supervision, not an LVOT anatomical/visibility annotation or independent clinical validation.

L uses manual marked-frame view; W uses directory view on identical images. Original frozen A08 five-class logitA5C−logitA4C margin is historical/source G. L/W have trained fresh scalar heads, so whole-classifier gains cannot independently identify a backbone effect. Both use the complete A08 trunk/marker/projection, old five-class head frozen, real16-frame context, same initialization/exposure/600-stepBCE/compute.

## Qualification failures retained

- Q1 strict all16slots same-view contract:397fit/75diagnostic requests, discordant-patient and diagnostic-A5C minima failed. No models.
- Q2 marker-frame contract fixed before pixels:707markers: **574fit/133diagnostic initially**. Header versus decoded-frame count and native PTS gaps prevented a naive global ordinal claim. No guessed offset. Independent chronology checks found702/707marker labels consistent across decoded ordinal/native PTS and all four endpoint/index conventions; five ambiguous markers were FIT-only. Whole707source binding claim failed and remains failed.
- Separately admitted A51-P limited verified subset: **569fit/133diagnostic**,36/10patients, manual A4C/A5C330/239fit and94/39diagnostic. Same original patient allocation/AVIs, five UNKNOWN fit labels excluded in both arms, no replacement. Two excluded points still occur as unsupervised context, disclosed. Full68AVI SHA/nativePTS and all702pixel windows independently bound;336contexts mixed/unknown and920edge-clipped slots. No publisher-test replication claim: public code's test membership depends on unprovided extracted-PNG availability; metadata split here is a new fixed research split. A 21-patient quarantine is an unused metadata buffer, not proof of original test exclusion.

The marker unit is intentional because A08 predicts a point within context; mixed context is a plausible training ambiguity, not proof of human label error. All private images/identifiers/row outputs remain server-only. Only selected apical AVIs decoded; protected project dev/check/test/Pediatric/Unity were not used.

## Executed source result

Two source models L/W each600real updates,4800real training forwards and4800backwards. Modelcomplete mutable state plus immutable Base reference and full optimizer/RNG retained. Checks establish independent BCE/derivative, full/microbatch gradients, zero-head startup followed by trunk/marker/projection gradients, exact repeat/forcedW/3+3resume, complete optimizer binding, unchanged old classifier/Base. Independent audit confirms both600-step journals/all400optimizer states/steps600/live drift/frozen tensors and complete model reconstruction. CPU RNG restoration accepted and CUDA state structure checked independently; real CUDA resume was tested by production3+3check.

Complete model manifest actual Git **d971248a198de758df3e4abe56ef30cdf5a426c0** preceded freeze and the only399diagnostic forwards (G/L/W×133). Per-cell/per-row locks and durable pending guards prevent silent retries. Independent cache-only reconstruction reproduces metrics/AUC/geometry/constant/gates without another forward.

| Source classifier | A5C correct /39 | A4C false positives /94 | Balanced accuracy | AUC | Discordant correct /17 |
|---|---:|---:|---:|---:|---:|
| Original A08 G |16|4|.683852|.807420|14|
| L precise labels |31|23|.775095|.831151|7|
| W directory labels |29|21|.760093|.819149|7|
| Fit-only geometry |23|19|.693808|.749864|7|
| Fit-prior constant |0|0|.500000|.500000|12|

**Paired, from the actual raw cache:** L/G positives16shared,15L-only,0G-only; A4C mistakes4shared,19L-only,0G-only. W/G positives15shared,14W-only,1G-only; A4C mistakes4shared,17W-only,0G-only. L/W positives29shared,2L-only,0W-only; A4C mistakes15shared,8L-only,6W-only. Aggregate error counts must not be used to infer this overlap.

Fit manual-label L accuracy564/569, BCE.02253; directory-label W accuracy567/569, BCE.00772. On true fit labels W505/569. L clearly learned the fit labels but did not generalize with acceptable A4C specificity. The local pixel/anatomy that drives this behavior is not established by these numbers; attention or activations would not make it LVOT truth.

Both predeclared paths failed: original L−W precise-label increment gate and the separately fixed absolute whole-classifier source-safety gate. L−WBA=.01500 below.05, identical7/17discordant success and two extra A4C mistakes in aggregate. Neither model meets BA.80, AUC.85/original+.03, original-A08FPR+.02, or geometryBA+.10 simultaneously. **No target transfer/seed2/threshold sweep/rescue within A51.** No claim that target F1 stayed equal by measurement; it was not reread. Original A08 source0 fullF1.956885 is historical background.

## Native-label support diagnosis

`native_support.json` independently counts original native A4C/A5C bags200/67 and200/75 with all original requests. Labels are clip codes, not local anatomy annotations. Original320×240 images retain.57478area through A08 crop. Of A44 positive-run137/259membership rows,6/25marker-centre reference labels differ from the run label and37/95have mixed context; these are constructed reference/mapping mismatches, not adjudicated expert errors. More pixels or weaker gating alone have already failed in A21/A36/A43–A48.

## Reproducibility and costs

Plans: direct-view-supervision, marker-amendment, axis-check, verified-pilot, resource-amendment, target-qualification (all dated2026-11-01). Source prelaunch6e81edd; complete source model manifest/freeze d971248. Source labels/input/plan/code/full weights/oldBase digests in `pilot/input_manifest.json`, `source_models_manifest.json`; independent models/final audits, raw-cacheSHA, paired analysis and gate verdicts linked alongside. Source formal job1150.714secwall/1141.613CPU; once readout31.617wall/29.296CPU; bootstrap/checks/metadata acquisition/failures/root/reviewer audits also charged.

No budget reset: all A51/A52 work shares cumulative `runs/a51_discriminative_evidence_v1/resource_ledger.json` and the inherited older source ledger. Full aggregate `resources.json` is finalized after this authorized phase closes. A51 formal GPU work is complete; A52 is separately prospective point-frame/direct-supervision representation, not reopening this source video configuration. No push or webpage edit.
