Research proposal · 12 September 2026 · Public-data study

# 1 Research direction and proposed abstract

We propose to study whether a model can separate changes in echocardiographic view from discontinuities caused by recording boundaries or image appearance, and then return confidently identified video intervals for downstream analysis. The initial product serves researchers curating heterogeneous echocardiography archives. The primary study will use public saved cine loops and explicitly constructed multi-clip sequences. Validation of natural bedside sweeps will be an extension only if a suitable public recording set is confirmed.

The central hypothesis is that learning semantic continuity with controlled same-view and different-view joins will reduce unnecessary fragmentation without obscuring genuine view changes. We will test a compact temporal model and semantic boundary head against strong simple baselines, including probability smoothing, HMM decoding, and divergence between neighboring view predictions. The key outcome is useful routed duration at a validation-selected contamination target, accompanied by boundary accuracy, false splits, and coverage of difficult cases. Classification confidence will describe view assignment; it will not certify suitability for EF measurement or recommend physical probe movements.

The preferred development dataset is EV9V, a newly released multi-view video resource. An empirical audit has verified its small public manifests and decoded three training videos. The study remains a research proposal: no trained-model accuracy, clinical benefit, or natural-transition performance has yet been established. The first two weeks will resolve label mapping and grouping constraints and establish a real-video baseline before investing in a more complex temporal method. [^1]

**Proposed abstract.** Public echocardiography datasets predominantly contain saved single-view cine loops, whereas automated archive processing requires reliable identification and routing of intervals in heterogeneous recordings. Naive composition of labeled clips creates editing cues that can be mistaken for anatomical transitions. We propose a public-data evaluation protocol that separates semantic view changes from same-view recording discontinuities and controlled appearance changes. A lightweight temporal model is trained with interior view supervision, semantic pair supervision, and consistency under nuisance-only edits. A segment selector returns supported view intervals or defers uncertain portions. Evaluation will distinguish unmodified native cine performance, constructed-sequence segmentation, and external-domain recognition, with natural transition evaluation conditional on verified recordings and independent annotation. The study will quantify false semantic splits, missed transitions, routing contamination, retained duration, and computational cost. All improvements and the strength of the algorithmic contribution will be determined by preregistered comparisons with simple semantic-change and temporal-segmentation baselines.

# 2 Research questions and intended contribution

## 2.1 The decision problem

For an input video, the system must produce start and end positions, a candidate view, an accept or defer decision, and a confidence score for each interval. It should retain interpretable intervals in a mixed recording rather than discarding the entire input because part of it is uncertain. When recordings are already separated into files, their boundaries and provenance remain available for export and audit. Model evaluation will deliberately hide constructed splice locations from the temporal detector.

The original Project 2 slide provides a useful product specification, but several elements require a stronger research formulation. View classification is established; temporal segmentation of echocardiography has historical precedent; temporal video classification with unknown-view rejection is also established. The publishable question is therefore the interaction between semantic continuity, acquisition-related discontinuities, and selective routing under realistic public-data constraints. [^2] [^3] [^4]

An operational challenge must be resolved early: an archive of already separated, single-view cines can often be routed by classifying each native file once. The study therefore includes a mandatory native-file-aware classifier with whole-file acceptance or deferral. Temporal routing must demonstrate additional recovery of useful portions, localization of errors, or a constructed-sequence failure that predicts errors on native or external recordings. Good artificial-boundary accuracy alone would not establish added value for archive users.

## 2.2 Falsifiable hypotheses

| **Hypothesis** | **Main comparison** | **Evidence that would support it** | **Evidence that would weaken it** |
|----|----|----|----|
| H1 Semantic continuity | Proposed training versus the same model with standard semantic pair supervision | Fewer false splits at same-view joins while retaining different-view boundary recall | Improvement only over a deliberately naive splice detector |
| H2 Selective routing | Segment score versus frame thresholding and mean-probability selection | More correctly routed duration at a comparable contamination operating point | Apparent purity gain explained entirely by rejecting more data |
| H3 Robustness | Unedited native clips and a frozen external-source evaluation | Stable classification and fewer unnecessary fragments outside the constructed training setting | Gains disappear on native clips or depend on one editing family |
| H4 Natural transitions if available | Independent annotated native transitions | Useful interval recovery and boundary localization under genuine probe motion | No eligible transitions or insufficient independent evaluation units |

## 2.3 Planned paper contributions

The first contribution will be a reproducible protocol for distinguishing semantic view changes from nuisance discontinuities using public cines. It will release a documented label ontology, source and split audit, composition recipes, and metrics that preserve the distinction between real recordings and constructed sequences. This becomes a substantive contribution only if it exposes a meaningful failure mode or changes the ranking of competitive methods.

The second candidate contribution will be a small semantic-continuity model trained using controlled pair examples and consistency under appearance-only edits. The architecture and regularizers build on existing methods; their combination is not intrinsically novel. The method must outperform a parameter-free semantic-divergence baseline and standard temporal models under matched data, compute, and operating policies. If it does not, the paper will report the simpler effective method and a benchmark or reliability finding.

The third contribution will be an evaluation of routing reliability at the segment and duration levels, with frozen validation-selected policies, explicit deferral, source-shift tests, and disclosure of grouping limitations. We will not claim formal distribution-free risk control without the necessary independent calibration units and statistical construction.

# 3 Prior work and the remaining gap

## 3.1 View recognition and automated pipelines

Early echocardiography video analysis already examined temporal parsing, so the project cannot claim the first temporal segmentation of echo recordings. Modern work established accurate frame-level recognition and integration of view classification into automated interpretation pipelines. These studies provide the historical and application baselines for routing, rather than evidence that a new classifier alone is sufficient novelty. [^5] [^6]

Temporal recognition, efficient deployment, and weak supervision have also been investigated. The relevant comparison is not simply CNN versus Transformer. We must determine whether local temporal reasoning improves decisions specifically at semantic changes, while avoiding fragmentation of stable views. Cross-hospital learning from public image datasets has precedent in Heart2Heart and Fix-A-Step, so an ordinary train-on-one-dataset/test-on-another experiment is not a standalone contribution. [^7] [^8] [^9] [^10]

## 3.2 The closest competitors

| **Study** | **Relevant existing capability** | **Remaining distinction to test** |
|----|----|----|
| Jansen et al 2024 | View classification, unknown-view recognition, and quality assessment; mixed-view videos are included among unknown examples | Recover usable stable portions within a mixed input instead of assigning one global rejection label |
| EchoViewCLIP 2025 | Temporal multi-instance learning, fine-grained recognition, OOD rejection, and a quality branch | Semantic boundary robustness and selective interval export on a public-data protocol |
| STFM and EV9V 2026 | Public nine-view videos; spatial and temporal fusion with uncertainty-guided segment sampling | View changes across constructed streams, nuisance joins, and interval-level routing rather than only clip classification |
| EchoFine and TTE47 2026 | Fine-grained image classification and multiple expert test annotations | Temporal stability and segment routing; use its public test images only as a separately frozen external endpoint |
| Gao et al 2026 | Public-data view routing and downstream EF modeling using TMED, EchoNet, and MIMIC resources | Source-controlled support, independently verified evaluation, and recovery of intervals within mixed inputs |

EchoViewCLIP uses a collected dataset with 38 view categories and incorporates both temporal information and OOD rejection. Its published code is relevant, but its complete original training data are not a verified public resource for this project. STFM is a June 2026 preprint and is particularly important because it accompanies the preferred development dataset. EchoFine is a 2026 Medical Image Analysis paper; its released TTE47 subset is a test image collection, not a full public video training dataset. [^11] [^12]

Gao and colleagues provide a close example of a public-data pipeline that classifies A4C, PLAX, and Other before downstream EF modeling. Their described training sources associate A4C with EchoNet-Dynamic, PLAX with EchoNet-LVH, and Other with MIMIC. This creates a concrete opportunity to test source–class confounding, although it does not prove that their model actually relies on the source. The proposed study will test this failure mechanism rather than infer it from dataset composition alone. [^13]

## 3.3 General temporal and uncertainty methods

MS-TCN, ASFormer, and boundary-aware networks already address temporal smoothing, over-segmentation, and boundary refinement. A 2025 CVPR study also learns multi-task temporal parsing from single-task videos through constructed sequences. Accordingly, neither a temporal head nor synthetic concatenation will be presented as a new general learning principle. [^14] [^15] [^16] [^17]

Maximum softmax probability, energy scores, outlier exposure, temperature scaling, and selective classification form the relevant reliability baselines. Their assumptions and evaluation units matter: calibrated frame probabilities do not automatically yield calibrated segment scores, and confidence may deteriorate under distribution shift. Conformal risk-control results cannot simply be attached to an arbitrary ratio of wrong accepted duration to accepted duration. [^18] [^19] [^20] [^21] [^22] [^23] [^24] [^25]

The proposed gap is therefore specific: public, reproducible evaluation of semantic temporal routing that disentangles view changes from editing or source cues, and measures the cost of both confident errors and unnecessary deferral. The literature search supports this as a candidate gap, not a proven first-of-its-kind claim. A final novelty check against newly published work will precede manuscript submission.

# 4 Data strategy under the public data constraint

## 4.1 Preferred core and optional external sources

| **Resource** | **Data and labels relevant here** | **Proposed role** | **Main limitation** |
|----|----|----|----|
| EV9V | 5,138 videos with nine source view codes and official splits | Core native-video training and controlled sequence construction | Label expansion inconsistency; no released patient map or natural boundary annotations |
| TMED-2 | Multi-view PNG images with view labels | Frame baseline and external recognition on overlapping classes | Static images cannot establish temporal boundary accuracy |
| MIMIC-IV-Echo v1.0.1 | 524,137 DICOM files in 7,228 studies | Conditional external native-video audit and independent annotation | Credentialed access; no verified gold view or transition labels |
| ECHOVIEW | Machine-generated view probabilities for a MIMIC subset | Candidate sampling, weak supervision, or old-model comparator | Pseudo-label agreement is not independent clinical accuracy |
| TTE47 | Released 5,447 test images with three expert annotations | Conditional frozen external ambiguity and unsupported-input test | Images only; access request; original training set private |
| CAMUS | A2C and intended A4C sequences and structural annotations | Optional external recognition and anatomy checks | Some intended 4CH acquisitions are actually A5C |
| Unity Imaging | Images and additional view metadata | Optional source-shift supplement | Task-specific releases and differing posted reuse terms |
| EchoNet families | Mostly view-specific saved videos | Optional external class-conditional tests or downstream demonstration | Class can correlate with source, resolution, and age |
| EchoXFlow | Signal-rich short recordings from examinations | Reserve only if a specific temporal gap is filled | Large and technically different; not documented uninterrupted sweeps |

TMED-2 is an image resource and must not be converted into apparent videos by repeating static images for the scientific temporal experiments. CAMUS filenames describe intended acquisition categories and require care for A4C/A5C discrimination. Unity has different terms on different release pages; the exact subset and permissions must be recorded. These resources are useful without assuming they provide natural transitions. [^26] [^27] [^28]

MIMIC-IV-Echo v1.0.1, released in August 2026, contains DICOM sequences and patient/study linkage. Its documentation describes numerous saved sequences per study, and some examinations occurred outside ED or ICU encounters. A chronological playlist does not recreate probe motion between recordings. The documentation also does not establish that view tags are complete or correct. ECHOVIEW supplies machine-produced probabilities; its small selected A4C review does not validate all of its categories. [^29] [^30]

TTE47 provides a valuable optional external image test with retained expert disagreement. It must remain untouched during development if used as an external test. Possible overlap with Unity-derived resources must be checked before calling the sources independent. EchoXFlow offers rich acquisition information but describes short sequential recordings; it will not be downloaded wholesale merely because it is large. [^31] [^32]

## 4.2 Label ontology

Start with a coarse native-video task comprising PLAX, PSAX, A4C, A5C, and subcostal four-chamber. Preserve every original EV9V code in the manifest. PLHLA maps provisionally to PLAX; A4C, A5C, and SC4C have consistent expansions. PASA, PMASA, PMVLSA, and PPMLSA can provisionally form a broad PSAX family, while their precise level names remain unresolved. PMPALA will be excluded from the primary anatomically named task until its inconsistent expansion is resolved. This is a proposed harmonization requiring a documented clinical or author check, not a correction silently imposed on the release. [^33]

The audited manifests imply 3,364 training, 521 validation, and 800 test cines for that provisional five-family task after excluding PMPALA. A separate nine-code classification table can preserve the original benchmark, but anatomical interpretation of disputed codes must remain qualified. A2C, IVC, and suprasternal will not be promised in the initial video model merely because they appear on the original slide. Extension requires supported training and test data and a source–class support check.

This is coarse-family routing. Changes between PSAX levels deliberately remain within one category and are not positive semantic boundaries in the primary task. The original nine-code analysis is a separate sensitivity endpoint. A downstream model requiring a particular PSAX level must not consume a broad PSAX label as if that level had been verified.

Maintain three different fields. The view label describes anatomy; support_status describes whether the input belongs to the intended mode and taxonomy; routing_decision records accept or defer. Other is a dataset-specific supervised category only where its meaning is known. Unknown is an output decision, not a new anatomical class. An identifiable unsupported view need not be poor quality, and a classifiable A4C need not be suitable for measurement.

## 4.3 Splits and provenance

Retain the official EV9V train, validation, and test assignments. The release reports patient-separated splits, but its public labeled lists contain video names and view codes, not explicit patient identifiers. Our audit can establish exact video-ID separation; it cannot independently prove patient separation or construct patient-level bootstrap intervals. No patient identities will be inferred from timestamp-like filenames.

If a grouping map becomes available, create grouped development and calibration partitions and use patient-cluster uncertainty estimates. If it does not, use a small fixed validation search, freeze all policies before the official test, and call thresholds validation-selected empirical operating points. All frames, overlapping windows, and constructed derivatives of one native cine remain in that cine's official split. Near-duplicate pixel checks remain an implementation task before model evaluation; zero duplicate filenames is insufficient.

External datasets remain fully held out from training, self-supervised adaptation, and threshold selection for the primary external experiment. If adaptation is studied later, it will be a separately named experiment with disjoint adaptation and evaluation groups. All derivative-to-parent links will be saved, including native video identifiers, frame ranges, timestamps, preprocessing versions, random seeds, and editing recipes.

# 5 Benchmark construction and independent evidence

## 5.1 Four evaluation settings

| **Setting** | **Input** | **Valid conclusion** | **Conclusion it cannot support** |
|----|----|----|----|
| Native cine | Untouched held-out saved videos | Recognition, deferral, and stability within actual saved recordings | Accuracy of boundaries that are not annotated |
| Controlled multi-clip stream | Held-out cines assembled with saved provenance | Segmentation and routing under specified composition rules | Generalization to real probe trajectories |
| External source | Independently labeled held-out images or videos | Cross-source recognition and, for videos, routing stability | Temporal validity from static images or pseudo-labels alone |
| Native transition extension | Unedited within-file view changes verified and annotated by experts | Real transition interval and stable-subsegment performance | Broad bedside efficacy without representative populations |

Native cine labels are clip-level labels. Propagating them to every frame is weak supervision, not a source of newly verified per-frame ground truth. Frame metrics on these propagated labels will be named accordingly. Clip-level accuracy is the cleanest existing-label endpoint; fragmentation on purported single-view clips is a stability proxy until ambiguous frames or true changes have been audited.

## 5.2 Separating semantic changes from nuisance edits

The primary constructed benchmark uses a two-by-two design. The two independent factors are whether the view category changes and whether a controlled appearance edit is introduced. Different native cines can have different anatomy and recording appearance, so the digital edit factor must not be confused with a fully controlled scanner or patient factor.

| **View before and after** | **Additional nuisance edit** | **Semantic view boundary target** | **Expected failure to measure** |
|----|----|----|----|
| Same category | None | No change | False split at an ordinary same-view recording join |
| Same category | Applied | No change | False split or excessive deferral caused by appearance |
| Different categories | None | Change | Missed semantic change without an extra editing cue |
| Different categories | Applied | Change | Overreliance on the easy edit cue |

Include pairs from two portions of the same cine as a particularly controlled negative, and same-view pairs from distinct cines as a harder negative. All four cells must be present in training and in prespecified evaluation. Class-change and no-change examples will have matched context lengths and nuisance schedules. File borders, filenames, displayed view text, and ground-truth join locations will never be model inputs.

Primary recipes will concatenate native frame ranges using hard joins. Conservative gamma or intensity edits, re-encoding, and resampling can supply appearance perturbations after training-example inspection confirms that anatomy remains recognizable. A pilot range such as gamma 0.9–1.1 is an engineering starting point, not a validated ultrasound simulation. More aggressive blur, blank intervals, and crossfades belong in a separately labeled corruption stress test; their interiors will be marked synthetic ambiguous intervals and excluded from hard anatomical supervision.

Cross-source construction is a secondary extension requiring overlapping classes at both sources and compatible reuse terms. A matrix with one source per class cannot distinguish source recognition from anatomy recognition. Equal resizing or adversarial training does not fill missing class–source combinations. Only supported cells will be used to test actual source-change invariance; the EV9V nuisance-edit experiment will not be described as a multi-hospital controlled experiment.

## 5.3 Recipe lengths and independence

Use a small fixed recipe specification: four to eight native fragments per constructed stream, target lengths of roughly 15–45 seconds, and at least two sampled classes when testing change detection. Draw valid fragment durations within each source video's actual duration. Do not stretch a short cine to satisfy a target, and do not duplicate cycles in the primary real-video analysis. Include short segments in a separate duration-stratified challenge so smoothing cannot win by erasing difficult intervals.

Partition native cines before making any recipe. The primary test bank will use each native cine at most once across independent recipe units. Within a unit, edited and unedited variants share the same cines and form a paired comparison; they are never treated as independent samples. Additional dense reuse can improve training diversity or provide a descriptive sensitivity analysis, but it does not increase the independent test sample size. With missing patient identifiers, uncertainty estimates remain conditional on cine-level grouping and cannot be advertised as patient-level inference.

A same-view join is not a semantic view boundary, but it remains a physical file boundary in provenance. Exported clinical cines must never span separate source files or patients. The temporal semantic benchmark may merge their category run for scoring; the export layer still returns distinct native-file intervals. This separation prevents a successful benchmark segmenter from creating a medically fictitious continuous recording.

# 6 Proposed model and training procedure

## 6.1 Input processing and visual encoder

Decode native video or DICOM before preprocessing and record actual dimensions, successfully decoded frame count, and timing source. Sample at 10 Hz using timestamps for routing. Preserve all original frames and time indices for final exports. This sampling rate is a proposed routing configuration, not a validated rate for EF or wall-motion measurement. Missing or irregular timing must be represented explicitly; unknown timing yields frame-index intervals rather than invented seconds.

For EV9V, distinguish released MP4 playback time from original acquisition time. The release discusses removal of invalid extracted frames, so continuity and correspondence of alternative frame packages require checking. Decoding a 30-fps MP4 verifies its playback format; it does not reconstruct missing acquisition frames or validate physical probe-motion speed.

Maintain aspect ratio with padding to 224 by 224 pixels and standardize channels. For grayscale images, repeat the grayscale channel only as required by the chosen encoder. DICOM color interpretation and compressed transfer syntaxes require explicit decoding checks. Masks should remove identifying or view-revealing text and unnecessary borders while preserving the ultrasound sector. Compare sector-focused and full-frame conditions; a crop that removes anatomy is not an acceptable shortcut control.

Start with an ImageNet-initialized ResNet-18 and a class-balanced view head. Native cine samples, rather than all available frames, should drive minibatch balance to avoid dominance by long recordings. After training the visual baseline, cache 512-dimensional features and freeze the encoder for the primary temporal comparisons. More powerful public pretrained encoders are optional secondary experiments with documented pretraining provenance and possible test-data overlap.

## 6.2 Small temporal module and semantic boundary head

Use a non-causal three-layer residual TCN with 128 channels, kernel size 3, and dilation factors 1, 2, and 4 as the initial temporal module. This is an offline archive-processing model. The comparison with simple smoothing and a frozen-feature MS-TCN isolates temporal reasoning from encoder changes. A streaming version would require a separately measured look-ahead budget and is outside the primary claim.

At each sampled position, pool the frozen 512-dimensional frame features over left and right neighborhoods, initially five samples on each side. An MLP with 128 hidden units receives the two pooled vectors, their absolute difference, and their elementwise product, and predicts semantic view change through one sigmoid output. During training, supervise sampled known joins and stable interior locations. During inference, slide the head over every eligible position; do not supply the constructed join locations. The left/right design requires future context and is another reason not to market the initial model as zero-latency guidance.

$`b_{t}\  = \ \sigma(g_{\varphi}(\lbrack u_{t},\ v_{t},\ |u_{t}\  - \ v_{t}|,\ u_{t}\  \odot \ v_{t}\rbrack))`$

Here u and v are the left and right pooled feature vectors, g is the trainable MLP, and sigma is the sigmoid function. The product is elementwise. This compact head supplements the temporal view predictions; its added value must be measured against divergence of the view probabilities alone.

The boundary label is one when supported view categories differ on either side, and zero when they are the same. This differs from a file-edit label. Equal numbers of examples from the four construction cells prevent the training set from making the nuisance edit a reliable proxy for semantic change.

## 6.3 Objective and regularization

$`L\  = \ L_{view}\  + \ \lambda_{pair}L_{pair}\  + \ \lambda_{inv}L_{inv}`$

The view term is class-balanced cross entropy on trusted labeled interiors. The pair term is balanced binary cross entropy for semantic change, including same-view joins as negatives. The invariance term combines Jensen–Shannon divergence between class distributions for an original example and its label-preserving edited version, plus squared disagreement between their boundary probabilities. The perturbation applies to the same source frames, so this consistency target does not assume that two different patients have identical image content.

Start with pair and invariance weights of 0.5. The restricted validation search will include pair weight 0.1, 0.5, or 1.0 and invariance weight 0 or 0.5, with no more than six combinations. These are proposed starting values. Ambiguous synthetic bands contribute neither a hard target view nor a fabricated clinical transition label. Standard smoothing losses, if used, must be disabled across known semantic changes and evaluated for erasure of short segments.

The pair head and consistency loss use established principles. To claim an algorithmic contribution, compare them with a standard semantic pair head trained on the same examples, ordinary augmentation consistency, and the direct semantic-divergence baseline. The study will not attribute gains from additional training examples, a larger encoder, or more compute to the proposed objective.

## 6.4 Segment decoding and selective routing

First generate candidate changes from persistent class differences and local boundary peaks, then apply validation-selected non-maximum suppression and hysteresis. Initial boundary suppression radii of 0.2 or 0.5 seconds and minimum segment durations of 0.5 or 1.0 seconds form a small prespecified search. Preserve low-confidence intervals as deferred output rather than forcing them into a neighboring view. These duration settings are engineering parameters; they do not certify a complete cardiac cycle.

For each candidate interval, begin with a transparent score: the tenth percentile of target-class probability multiplied by the proportion of sampled frames agreeing with that class. Compare this with mean probability, calibrated maximum softmax, and max-logit or energy-based alternatives. It is a ranking score until its relationship to actual segment correctness is measured. A learned selector and an ensemble are optional upgrades, not requirements for the first paper.

Choose acceptance thresholds on the official validation partition to target 5% duration contamination on constructed validation streams, and additionally report targets of 1% and 10% when support is sufficient. Then freeze the entire pipeline before the test. If no threshold meets the target with useful coverage, report failure; do not assign perfect purity to zero accepted duration. This is empirical policy selection, not a clinical error tolerance or a distribution-free guarantee.

# 7 Experimental comparisons and statistical analysis

## 7.1 Baseline ladder

| **ID** | **Model or decision rule** | **Purpose** |
|----|----|----|
| B-file | Native-file-level probability aggregation and whole-file acceptance or deferral | Mandatory archive-utility comparator using the file boundaries available in practice |
| B0 | Frame encoder with raw argmax | Establish classification and flicker without temporal processing |
| B1 | Probability moving average or categorical majority filter plus hysteresis | Strong inexpensive temporal baseline; never take a numerical median of arbitrary class IDs |
| B2 | HMM with training-estimated transitions and Viterbi decoding | Test whether basic state persistence is sufficient |
| B3 | Jensen–Shannon divergence between left and right averaged class probabilities, with persistent label change | Essential semantic-change baseline without a learned boundary head |
| B4 | Same visual features with MS-TCN | Standard learned temporal-segmentation competitor |
| B5 | ASFormer or a BCN-style boundary baseline | Stronger temporal comparison if the pilot budget permits |
| B6 | STFM on native cines and a documented sliding-window adaptation | Recent echo-specific video and uncertainty baseline |
| P0 | TCN plus standard semantic pair BCE | Isolate the benefit of the extra boundary head |
| P1 | P0 plus balanced nuisance-pair construction | Isolate the data design |
| P2 | P1 plus prediction and boundary consistency | Full proposed method |

A deliberately naive detector trained to mark every splice will be included only as a diagnostic control. Beating it is not sufficient evidence for the method. If the contribution depends on the boundary model, B3 and P0 are mandatory. If only a benchmark contribution remains, all competitive methods should still receive the improved construction protocol where appropriate.

B-file belongs in a separate operational table. Its known recording boundaries must not be credited as learned semantic boundary detections. Constructed-stream detectors receive no join positions and do not reset context at hidden joins; native-file operational processing and exports retain the real file boundaries. This explicitly separates the laboratory segmentation problem from the information available to an archive curator.

STFM must be reproduced using its actual released data layout and code behavior. A custom implementation inspired by a paper will be labeled an adaptation, not an exact reproduction. EchoViewCLIP and Jansen provide conceptual and implementation comparators where usable code permits; published accuracy from private datasets will not be inserted into a public-data results table as if it were a matched comparison. [^34] [^35]

## 7.2 Primary and secondary endpoints

$`C(\tau)\  = \ \frac{\sum_{t}^{}{\delta_{t}A_{t}(\tau)}}{\sum_{t}^{}\delta_{t}}`$

$`R(\tau)\  = \ \frac{\sum_{t}^{}{\delta_{t}A_{t}(\tau)\ 1\lbrack ŷ_{t}\  \neq \ y_{t}\rbrack}}{\sum_{t}^{}{\delta_{t}A_{t}(\tau)}}`$

Here, sampled time has duration weight delta, A indicates acceptance into a supported view, and a wrong route includes any true unsupported or ambiguous interval incorrectly assigned to a supported view. Coverage uses all evaluable input duration as its denominator. Contamination is undefined when no duration is accepted. A known-view-only coverage measure will be secondary and explicitly labeled.

The implementation's primary outputs are known_view_routing_coverage and known_view_routing_risk. Generic accepted coverage can include an accepted Other classification and is not the primary routing metric. Synthetic ambiguous or unsupported reference intervals remain in the denominator, and routing them into a target class counts as contamination. Truly missing/unscorable data are reported separately with their duration rather than hidden through silent filtering.

The primary analysis reports achieved test contamination and coverage at the frozen validation-selected 5% target policy. Both values must be shown together: a policy that exceeds its target on test is not presented as meeting the target. Full risk–coverage curves and coverage at matched test contamination may be descriptive comparisons, but thresholds read from the test curve are oracle operating points and cannot represent deployed policy selection.

| **Endpoint** | **Required reporting detail** |
|----|----|
| Native cine classification | Macro-F1, balanced accuracy, per-class confusion, and cine-level majority prediction |
| Native cine stability | Fragments per minute and prediction changes within source-labeled single-view cines, identified as proxy metrics |
| Segment F1 | Same-class one-to-one temporal IoU matching at 0.10, 0.25, and 0.50 |
| Semantic boundary F1 | One-to-one matches within 0.25, 0.5, and 1.0 seconds; include misses and false alarms |
| Boundary timing | Median error for matched boundaries together with recall; no easy-boundary-only interpretation |
| Nuisance false splits | False semantic boundaries per same-view join, separately by edit condition |
| Selective segment error | Fraction of accepted segments with less than 95% target-view duration; definition is operational, not clinical |
| Ordering | Normalized edit score with a fixed policy for deferred gaps; never remove hard gaps silently |
| Deferral and support | Known-view retention and false acceptance of available unsupported categories |
| Reliability | Segment-score reliability, Brier score or NLL when probabilistic, and ECE with stated bins |
| Runtime | End-to-end seconds per minute of video, decoder and encoder costs, memory, and actual hardware |

Semantic boundary scoring uses the semantic head or persistent-view-change event list. An A4C–defer–A4C output does not automatically represent two anatomical changes. Acceptance/defer edges are reported separately as decision fragmentation, while the deferred duration remains visible in coverage and interval exports.

## 7.3 Required ablations and failure analyses

Compare every ablation on the same native split, sampled frames, pair counts, training steps, and base augmentations. Run three seeds for the final shortlisted methods. The central ablation table separates standard pair learning, four-cell balancing, posterior consistency, boundary consistency, and segment-level selection. Evaluate the same-view false-split rate and different-view recall together; optimizing one alone is trivial.

Perform border-only and sector-only controls, while interpreting a successful border classifier as evidence of available shortcuts rather than proof that the main model uses them. Test unedited native clips, a held-out nuisance-edit family, no-view-change streams, and short valid segments. A class–source support matrix is required before claiming cross-source invariance. Compare optional pediatric data separately from adult data. Training outliers and test unsupported-view families must be distinct if open-set rejection is studied.

Use paired resampling of independent test recipe units and native cines, keeping all edited versions and model predictions for a unit together. Do not bootstrap individual frames or reuse-heavy synthetic sequences as if they were independent. Without a patient map, these intervals are descriptive cine-group uncertainty estimates and may understate patient dependence. If patient IDs become available, upgrade to patient-cluster bootstrap and state the number of independent patients. Report effect sizes and seed variability; use one prespecified primary comparison rather than selecting the most favorable of many tests.

# 8 External validation and annotation options

## 8.1 Existing labels only

The project can start without acquiring private data or creating new annotations. Use official EV9V labels for native cine classification, the composition recipes for synthetic semantic boundaries, and existing human labels in TMED-2 or a permitted external subset for image recognition. These provide a reproducible empirical core. They do not establish frame-by-frame clinical truth, native transition timing, or general bedside safety.

TTE47, if access is obtained, can evaluate mapped supported views and expert disagreement. Freeze the mapping and model before viewing test outcomes; retain individual expert labels and report consensus and uncertain subsets separately. Report image-level outcomes, not video performance. A2C and other unsupported classes must be evaluated as unsupported inputs or in a separately trained extension, never silently mapped to an arbitrary core class.

## 8.2 Small expert review of public data

If clinician time is available, create new annotations on an authorized public MIMIC subset without collecting private images. A planning target is 300–600 cines with a population-random component and a separately reported enrichment component for rare classes and suspected transitions. Sample by confirmed patient identity where available. Experts should be blinded to model predictions and weak labels; retain their separate judgments before adjudication.

Record view, mode, whether the whole cine is assignable to one view, and any intervals of genuine within-file change. For a transition, mark the last clearly identifiable old-view frame and the first clearly identifiable new-view frame. The region between them is an uncertainty interval, not an arbitrarily precise ground-truth timestamp. Evaluate boundary distance to this interval and the amount of ambiguous duration that leaks into accepted outputs.

The sample size will be revised after a pilot measures prevalence, annotation time, and between-reader agreement. The proposed counts are feasibility targets, not a power calculation or evidence that rare error rates can be certified. Optional external annotations must be separated into policy-calibration and locked evaluation patients if local recalibration is studied. The primary external analysis keeps the original policy frozen.

## 8.3 Natural stream decision

Audit eligible long native videos rather than assuming length implies view changes. A planning threshold for pursuing the natural-transition extension is at least 30 confirmed independent patients and 100 independently reviewed transition events. This threshold decides whether a pilot is worth pursuing; it does not guarantee statistical power. If grouping is unavailable, transitions are absent, or annotation access is insufficient, remove natural bedside-transition efficacy from the paper title and main claim.

MIMIC view numbers and acquisition timestamps may help find recordings, but they do not establish the anatomical category or recover missing footage between saved cines. ECHOVIEW probabilities are useful for candidate selection, not the primary reference labels. Evaluate the random and enriched samples separately to avoid making a high-confidence preselected subset appear representative of the archive.

# 9 Implementation plan and reproducibility

## 9.1 Data contracts and software modules

Use one native-recording manifest with dataset, source version, official split, native video ID, verified patient/study IDs when present, original label, mapped label, mapping version, dimensions, frame count, timing provenance, and access conditions. Store absent identifiers as null. A second manifest describes derived training windows or evaluation recipes and links every interval to its native parent. Neither manifest contains inferred patient identities.

The implementation will separate six modules: ingestion and decoding; label and duplicate audit; frame sampling and feature extraction; composition and perturbation; temporal prediction and selection; and evaluation with native-file export. This permits the same cached features, recipe bank, and metric definitions to be used across baselines. Model outputs contain candidate view, score, decision, interval coordinates, and native provenance. Scores and reasons such as unsupported input or low confidence are separate from clinical diagnoses.

The supplied starter package already implements the metadata audit, a subset of temporal/routing metrics, and deterministic two-cine metadata recipes. It does not yet render four-to-eight-fragment streams, train an encoder or TCN, implement every planned metric, or export model-selected clinical intervals. These are explicit development tasks, not hidden assumptions of completion.

## 9.2 Concrete first training run

Train the ResNet-18 frame model on the audited official training partition with class-balanced cine sampling. A proposed initial configuration is AdamW, learning rate 0.0003, weight decay 0.0001, batch size 64 where memory permits, and a maximum of 30 epochs with validation-based early stopping. Sample up to 16 timestamp-spaced frames per selected cine visit. Do not treat those frames as 16 independent patients. These numerical values define a reproducible starting run; revise them once in the pilot if the loss or memory profile requires it and log the reason.

Evaluate native-file aggregation and B1 before introducing a learned temporal module. Cache features using the selected visual checkpoint and its hash. Train the temporal comparators on identical fixed training recipes, initially with AdamW at learning rate 0.001, batches of eight sequences, and a maximum of 50 epochs. Pad with an explicit validity mask. Native labels, synthetic ambiguous bands, and padded positions must have different supervision masks. Save the exact random seeds, optimizer state, model configuration, and inference policy.

Use three seeds only for shortlisted final comparisons after the pilot establishes that the pipeline works. The development matrix is deliberately bounded: one small encoder, one small temporal head, six loss-weight combinations, and a limited decoding search. Add a larger architecture only if the remaining scientific question requires it. Report the number of validation trials so tuning effort is visible.

## 9.3 DICOM ingestion for the conditional external arm

For MIMIC, use the documented record list for patient/study linkage and preserve the distinction between a study and its constituent files. Inspect NumberOfFrames, Rows, Columns, PhotometricInterpretation, TransferSyntaxUID, and available frame timing attributes during ingestion. Available timing can vary by object; the pipeline must explicitly select the applicable timing source or retain frame indices. A generic DICOM tag or free-text description is a candidate clue, not a verified view label. [^36] [^37]

Begin with a small approved-access sample and confirm decoded pixel shape, color handling, and timing before batch processing. Keep diagnostic logs free of identifying text and distribute code and permitted label manifests rather than restricted images. Academic or credentialed access does not imply unrestricted redistribution. No MIMIC access application or download has been performed as part of the completed audit.

## 9.4 Release package

Release versioned code, environment specification, label mappings, training configurations, permitted source IDs, composition recipes, scoring scripts, and a complete experiment ledger. Prefer regenerable recipes over redistributing third-party video. Record dataset versions and checkpoint pretraining provenance. A public checkpoint trained on private clinical data can be a useful comparator, but its results belong in an externally pretrained track rather than the strict public-training-data track.

The paper should include sufficient native-ID grouping and recipe-unit metadata to reproduce its uncertainty estimates. When the dataset does not expose patient grouping, disclose that limitation in both the methods and release README. A large number of frames or synthetic permutations does not remove it.

# 10 Feasibility checks already completed

The following results were actually obtained on 12 September 2026. They establish accessibility and engineering behavior, not model efficacy. The full audit records the pinned EV9V repository commit 094135b4da40c9aed9d9550c65305dc8715c7d3a, source hashes, and executed commands.

| **Completed check** | **Observed result** | **Interpretation and limit** |
|----|----|----|
| Public split files | 3,683 train, 567 validation, 888 test; 5,138 total; nine codes | Existing labeled video manifests are accessible |
| Manifest integrity | Zero malformed rows, duplicate IDs, conflicting labels, or exact cross-split video-ID overlap | Does not independently verify patient separation or pixel duplicates |
| Media archive access | HTTP Range access verified; reported archive size 6,606,477,312 bytes | Bounded partial access worked; full archive was not downloaded |
| Three training MP4s | All decoded successfully; 320 × 240; 30-fps playback; 133, 79, and 39 frames | Readability of three files, all from one timestamp-like source prefix; no label or population validation |
| Temporal metric scaffold | Thirty-five synthetic engineering checks passed | Arithmetic and edge-case tests, not measured model performance |
| Recipe scaffold | Sixteen synthetic metadata checks passed | Deterministic pairing, split separation, and source-reuse controls; no rendered stream or learned model |

The decode sample comprised PLHLA, PMASA, and PMPALA source labels, with playback durations of approximately 4.43, 2.63, and 1.30 seconds. The audit transferred 50 MiB of archive prefixes across an initial matching correction and a successful follow-up. No test video pixels were used for model selection, and no patient media are redistributed in the starter package.

The audit does not establish that all cines are standard views, that the released timing preserves uninterrupted acquisition, or that a temporal router improves any downstream outcome. No model has been trained or evaluated. Remaining feasibility work includes content-duplicate checks, clinical label mapping, representative native inspection, full data ingestion, GPU profiling, and real baseline evaluation. A source-code/configuration scaffold is not evidence for the proposed loss or architecture.

# 11 Twelve-week action plan and decision gates

The plan assumes one primary researcher, access to one suitable GPU for model training, and optional clinician review time. A GPU with roughly 16–24 GB memory is a planning target for the proposed small models, not a measured requirement. Verify memory and runtime on a small training subset before committing to the full comparison matrix. The audit and metric scaffold run on CPU; no GPU training benchmark has yet been performed.

| **Period** | **Work and concrete deliverable** | **Decision or acceptance check** |
|----|----|----|
| Week 1 | Obtain permitted data; freeze source versions; review code-to-view mapping; inspect stratified training cines; document file timing and grouping | Which labels and claims are supportable? Begin optional access requests through the research team |
| Week 2 | Train B-file, B0, and B1; inspect errors on development cines; prototype the four construction cells; estimate expert annotation time | Does temporal routing address within-file errors or a failure that transfers beyond artificial joins? |
| Weeks 3–4 | Complete recipe renderer and metric adapters; build non-reused test recipe units; train B2, B3, B4, and P0 | Are leakage checks clean, and are simple semantic baselines competitive? |
| Weeks 5–6 | Train P1/P2; run matched ablations; hold out a nuisance family; profile runtime | Is any gain attributable to data balance or consistency under matched resources? |
| Weeks 7–8 | Freeze the primary policy; evaluate native and constructed held-out data; run the permitted external-label arm | Report achieved risk with coverage, class-specific failures, and source-shift degradation |
| Weeks 9–10 | Complete shortlisted seeds and paired uncertainty estimates; optional blinded public-data transition review | Confirm the strength of the contribution and remove unsupported bedside claims |
| Weeks 11–12 | Assemble figures, release manifests/code, write the manuscript and limitations; rerun the final novelty search | Every claim maps to an experiment; all primary and negative results are retained |

The week-2 gate is substantive. Evidence can include verified mixed-view native intervals, clinically reviewed uncertainty islands, or a reproducible association between controlled nuisance failures and errors on untouched native/external videos. Mere prediction flicker on unreviewed source-labeled clips is insufficient to establish clinical need. If no such evidence appears, the study should focus on the constructed-sequence robustness benchmark and clearly reduce its archive-utility claim, or pivot to selective native-cine routing with independent external evaluation.

The transition extension has its own gate: eligible real within-file changes, usable patient grouping, permitted access, and independent annotations must all exist. If the gate fails, complete the public native/constructed experiments instead of presenting a playlist as a natural recording. Credentialed access delays must not block the EV9V development track.

# 12 Paper structure, success criteria, and risks

## 12.1 What a convincing paper would show

The introduction should motivate a demonstrated routing failure and distinguish semantic view intervals from ordinary saved-file boundaries. Related work should compare directly with Jansen, EchoViewCLIP, STFM, and public-data EF routing. The methods should specify the ontology, evidence levels, pair construction, model, frozen acceptance policy, and grouping limitations. Results must begin with native-file utility and strong baselines before presenting the proposed model's ablations.

Plan five main displays: a task figure with native provenance and deferred intervals; a dataset/source–class support table; a native and constructed baseline table; a four-cell false-split versus missed-change comparison; and risk–coverage plots with the actual frozen-policy operating points highlighted. An external-results table and examples of model failures should accompany them when data support it. No results cells should be filled with expected improvements.

The method contribution is supported only if the proposed model improves the prespecified trade-off over B3 and P0, with an effect not explained by extra training data or lower coverage, and if its relevance extends beyond one renderer. A simple model matching P2 is a scientifically useful finding. A benchmark contribution requires reproducible failure analysis that changes understanding or evaluation, not merely concatenation code. Full bedside transition efficacy requires the independently annotated natural-transition arm.

## 12.2 Main risks and planned responses

| **Risk** | **Consequence** | **Planned response** |
|----|----|----|
| No useful within-file temporal problem | Artificial task has weak operational value | Enforce B-file and week-2 relevance gate; narrow or pivot the claim |
| Curated EV9V has near-ceiling stability | Little evidence for uncertainty handling in difficult bedside data | Preserve native results; seek independently labeled external examples without inflating synthetic corruption |
| Ambiguous label mapping | Anatomically named results become unreliable | Preserve source codes, seek clarification, and restrict unresolved endpoints |
| Missing patient grouping | Patient leakage/uncertainty cannot be independently audited | Retain official splits, audit exact and content duplicates, disclose cine-level limits |
| Class is tied to source | Accuracy can reflect device or dataset cues | Use multi-view core data and only supported source–class comparisons |
| Simple baseline matches proposed model | Weak architectural novelty | Report the simpler result and reassess benchmark/reliability contribution |
| External policy fails its target | Validation confidence does not transfer | Report target failure, coverage loss, and class/source effects; do not retune on the test |
| Access or expert time is unavailable | Natural boundary truth remains absent | Finish the existing-label track and state exactly what remains unvalidated |

The recommended commitment is a two-week public-data pilot followed by the bounded twelve-week study if the relevance gate succeeds. This commits the team to a concrete question and comparison protocol while keeping the eventual paper's claims proportional to its evidence.

# Appendix A Focused reading guide

This is a targeted primary-source review through 12 September 2026, not a systematic review or proof of absence of competing work. The guide prioritizes what each paper changes about the proposed study. Published accuracy figures from incompatible datasets are deliberately not treated as a ranking.

**Task history and baseline pipelines.** Ebadollahi et al. (2002) already parsed echo examinations temporally using domain and ECG information. Madani et al. (2018) established CNN view recognition with frame-to-video aggregation, while Zhang et al. (2018) integrated view recognition into automated interpretation. Read these to define the task honestly and construct the frame/clip baseline. [^38] [^39] [^40]

**Video modeling and efficiency.** Howard et al. (2020) compared temporal/video models and excluded recordings with changing views or unidentifiable anatomy. Azarmehr et al. (2021) studied efficient architecture design. Naser et al. (2024) tested view classification in TTE and POCUS settings. Together they show that video models, efficient deployment, and bedside view recognition are established; temporal interval recovery is the narrower unresolved question. [^41] [^42] [^43]

**Representation learning and public transfer.** Chartsias et al. (2021) used supervised contrastive view learning. Fix-A-Step (2023) used uncurated unlabeled data and public cross-hospital echo experiments. Naidoo et al. (online 2025) studied spatiotemporal contrastive learning, and EchoFine (2026) examined fine-grained recognition and reader variability. These motivate matched contrastive/consistency comparisons and careful handling of fine labels, rather than a claim that contrastive learning itself is new. [^44] [^45] [^46] [^47]

**Unknowns and quality.** Jeon et al. (2023, version 2) studied semantic features and near-OOD view rejection. Jansen et al. (2024) combined view recognition, unknown handling, and quality assessment; mixed-view videos can receive a global unknown decision. EchoViewCLIP (2025) adds semantic and temporal aggregation with OOD and quality tasks. Recovering useful intervals must be compared with these whole-clip decisions. Their complete training cohorts are not verified as public resources for this project. [^48] [^49] [^50]

**Most recent public-data relevance.** STFM (June 2026 preprint) accompanies EV9V and must inform the native-video baseline. EchoPrime (Nature 2026, first online 2025) demonstrates large-scale view-aware study interpretation with public weights but private training data. Gao et al. (August/September 2026) already routes views in a public-resource EF pipeline. These papers prevent claiming novelty from public-data routing or downstream model selection alone. [^51] [^52] [^53]

**Temporal segmentation methods.** MS-TCN supplies the required learned temporal baseline; ASFormer and BCN provide stronger optional comparisons. BaFormer studies efficient boundary-aware segmentation. A CVPR 2025 paper on multi-task activity parsing from single-task videos directly cautions against claiming synthetic concatenation as a new general idea. Adaptation to echo needs task-specific evidence. [^54] [^55] [^56] [^57] [^58]

**Reliability methods.** Begin with maximum softmax and energy scores before complex OOD methods. Outlier Exposure informs the distinction between training outliers and unseen test classes. Guo et al. clarify temperature scaling; selective prediction and SelectiveNet formalize the coverage/error trade-off. Ovadia et al. motivate shift evaluation. Conformal Risk Control and Learn Then Test show why guarantees require an explicit loss, independent units, and valid policy-selection construction; those conditions are not automatically satisfied here. [^59] [^60] [^61] [^62] [^63] [^64] [^65] [^66] [^67]

# Sources

Primary papers, official dataset pages, and standards consulted. Dataset counts and access conditions refer to the cited versions; URLs accessed 12 September 2026.

1\. Bo Gou, Jicheng Zhang, et al. (2026). [<u>Echocardiographic Videos of Nine Views (EV9V)</u>](https://huggingface.co/datasets/bgx666/EV9V/blob/main/README.md). Hugging Face dataset repository. Dataset card and original split manifests; accessed 12 September 2026.

2\. Shahram Ebadollahi, Shih-Fu Chang, and Henry Wu (2002). [<u>Echocardiogram Videos: Summarization, Temporal Segmentation and Browsing</u>](https://www.ee.columbia.edu/dvmm/publications/02/icip02_shahram.pdf). IEEE International Conference on Image Processing (ICIP), pp. 613–616.

3\. Gino E. Jansen et al. (2024). [<u>Automated echocardiography view classification and quality assessment with recognition of unknown views</u>](https://pmc.ncbi.nlm.nih.gov/articles/PMC11364256/). Journal of Medical Imaging 11(5), 054002.

4\. Shanshan Song, Yi Qin, Honglong Yang, Taoran Huang, Hongwen Fei, and Xiaomeng Li (2025). [<u>EchoViewCLIP: Advancing Video Quality Control through High-performance View Recognition of Echocardiography</u>](https://papers.miccai.org/miccai-2025/paper/4443_paper.pdf). MICCAI 2025, pp. 181–191.

5\. Ali Madani et al. (2018). [<u>Fast and accurate view classification of echocardiograms using deep learning</u>](https://www.nature.com/articles/s41746-017-0013-1). npj Digital Medicine 1, 6.

6\. Jeffrey Zhang et al. (2018). [<u>Fully Automated Echocardiogram Interpretation in Clinical Practice: Feasibility and Diagnostic Accuracy</u>](https://pmc.ncbi.nlm.nih.gov/articles/PMC6200386/). Circulation 138, 1623–1635.

7\. Neda Azarmehr et al. (2021). [<u>Neural architecture search of echocardiography view classifiers</u>](https://doi.org/10.1117/1.JMI.8.3.034002). Journal of Medical Imaging 8(3), 034002.

8\. James P. Howard et al. (2020). [<u>Improving ultrasound video classification: an evaluation of novel deep learning methods in echocardiography</u>](https://jmai.amegroups.org/article/view/5205/html). Journal of Medical Artificial Intelligence 3, 4.

9\. Agisilaos Chartsias et al. (2021). [<u>Contrastive Learning for View Classification of Echocardiograms</u>](https://arxiv.org/abs/2108.03124). MICCAI 2021, pp. 149–158.

10\. Zhe Huang, Mary-Joy Sidhom, Benjamin S. Wessler, and Michael C. Hughes (2023). [<u>Fix-A-Step: Semi-supervised Learning From Uncurated Unlabeled Data</u>](https://proceedings.mlr.press/v206/huang23c.html). AISTATS 2023, Proceedings of Machine Learning Research 206.

11\. Bo Gou, Jicheng Zhang, et al. (2026). [<u>Spatio-Temporal Fusion Model for Standard View Classification of Echocardiographic Videos</u>](https://arxiv.org/html/2606.17437v1). arXiv:2606.17437v1. Preprint posted 16 June 2026; no journal acceptance established in this audit.

12\. Preshen Naidoo et al. (2026). [<u>Robust fine-grained echocardiographic view classification with supervised contrastive learning</u>](https://doi.org/10.1016/j.media.2026.104006). Medical Image Analysis 110, 104006.

13\. Zhiyuan Gao, Dominic Yurk, and Yaser S. Abu-Mostafa (2026). [<u>Learning from Scarce Labels: Multi-View Echocardiography for Ejection Fraction Prediction</u>](https://arxiv.org/html/2609.02969). Machine Learning for Biomedical Imaging (MELBA), 2026:025. Published 27 August 2026; arXiv posted 2 September 2026. DOI: 10.59275/j.melba.2026-8194.

14\. Yazan Abu Farha and Juergen Gall (2019). [<u>MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation</u>](https://arxiv.org/abs/1903.01945). IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

15\. Fangqiu Yi, Hongyu Wen, and Tingting Jiang (2021). [<u>ASFormer: Transformer for Action Segmentation</u>](https://arxiv.org/abs/2110.08568). British Machine Vision Conference (BMVC).

16\. Zhenzhi Wang, Ziteng Gao, Limin Wang, Zhifeng Li, and Gangshan Wu (2020). [<u>Boundary-Aware Cascade Networks for Temporal Action Segmentation</u>](https://github.com/MCG-NJU/BCN). European Conference on Computer Vision (ECCV). Official author implementation and citation record.

17\. Yuhan Shen and Ehsan Elhamifar (2025). [<u>Understanding Multi-Task Activities from Single-Task Videos</u>](https://openaccess.thecvf.com/content/CVPR2025/papers/Shen_Understanding_Multi-Task_Activities_from_Single-Task_Videos_CVPR_2025_paper.pdf). IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

18\. Dan Hendrycks and Kevin Gimpel (2017). [<u>A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks</u>](https://arxiv.org/abs/1610.02136). International Conference on Learning Representations (ICLR).

19\. Weitang Liu, Xiaoyun Wang, John D. Owens, and Yixuan Li (2020). [<u>Energy-based Out-of-distribution Detection</u>](https://arxiv.org/abs/2010.03759). Advances in Neural Information Processing Systems (NeurIPS).

20\. Dan Hendrycks, Mantas Mazeika, and Thomas Dietterich (2019). [<u>Deep Anomaly Detection with Outlier Exposure</u>](https://arxiv.org/abs/1812.04606). International Conference on Learning Representations (ICLR).

21\. Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger (2017). [<u>On Calibration of Modern Neural Networks</u>](https://proceedings.mlr.press/v70/guo17a.html). ICML, Proceedings of Machine Learning Research 70.

22\. Yonatan Geifman and Ran El-Yaniv (2017). [<u>Selective Classification for Deep Neural Networks</u>](https://arxiv.org/abs/1705.08500). Advances in Neural Information Processing Systems (NeurIPS).

23\. Yaniv Ovadia et al. (2019). [<u>Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift</u>](https://arxiv.org/abs/1906.02530). Advances in Neural Information Processing Systems (NeurIPS).

24\. Anastasios N. Angelopoulos, Stephen Bates, Adam Fisch, Lihua Lei, and Tal Schuster (2024). [<u>Conformal Risk Control</u>](https://openreview.net/forum?id=33XGfHLtZg). International Conference on Learning Representations (ICLR). Initial preprint appeared in 2022.

25\. Anastasios N. Angelopoulos, Stephen Bates, Emmanuel J. Candès, Michael I. Jordan, and Lihua Lei (2021). [<u>Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control</u>](https://arxiv.org/abs/2110.01052). arXiv:2110.01052. Initial preprint 2021; revised version 2022. No unverified journal venue asserted.

26\. Zhe Huang, Gary Long, Benjamin S. Wessler, and Michael C. Hughes (2022). [<u>TMED-2 dataset</u>](https://tmed.cs.tufts.edu/tmed_v2.html). Tufts Medical Echocardiogram Dataset project. Official resource page; related dataset paper: TMED 2: A Dataset for Semi-Supervised Classification of Echocardiograms, ICML DataPerf Workshop 2022.

27\. CAMUS project team (Simon Leclerc, Olivier Bernard, and collaborators) (2019). [<u>CAMUS project – Database</u>](https://www.creatis.insa-lyon.fr/Challenge/camus/databases.html). CREATIS / CAMUS challenge. Official database documentation; accessed 12 September 2026.

28\. Unity Imaging Collaborative (n.d.). [<u>Unity Imaging Collaborative – Open-Access Datasets for AI in Cardiology</u>](https://data.unityimaging.net/). Unity Imaging Collaborative. Living project documentation; accessed 12 September 2026. Exact release terms must be checked for each downloaded component.

29\. Brian Gow, Tom Pollard, Nathaniel Greenbaum, Benjamin Moody, Ahram Han, Jonathan W. Waks, Alistair Johnson, Elizabeth Herbst, Parastou Eslami, Ashish Chaudhari, Tanner Carbonati, Seth Berkowitz, Roger Mark, and Steven Horng (2026). [<u>MIMIC-IV-Echo: Echocardiogram Matched Subset</u>](https://physionet.org/content/mimic-iv-echo/1.0.1/). PhysioNet, version 1.0.1. Published 25 August 2026. DOI: 10.13026/307c-mr50; version-specific citation.

30\. Sampath Rapuri, Sofia Sapeta Dias, Maria Salomé Carvalho, Malcolm Lizzappi, Carl Harris, and Robert Stevens (2026). [<u>Structured Viewing Classification Annotations From the MIMIC-IV-ECHO Dataset (ECHOVIEW)</u>](https://physionet.org/content/echoview/0.1/). PhysioNet, version 0.1. Published 17 March 2026. DOI: 10.13026/ywz0-5b62. Machine-generated labels.

31\. THRIVE Centre / Preshen Naidoo and collaborators (2026). [<u>TTE47 — Dataset Card & Reference Gallery</u>](https://www.thrive-centre.com/datasets/TTE47). THRIVE Centre. Official test-set release and request page; related paper is listed under echofine.

32\. Elias Stenhede, Joanna Sulkowska, Eivind Bjørkan Orstad, Henrik Schirmer, and Arian Ranjbar (2026). [<u>EchoXFlow: A Beamspace Echocardiography Dataset for Cardiac Motion, Flow, and Function</u>](https://arxiv.org/abs/2605.05447). arXiv:2605.05447. Dataset paper posted 6 May 2026; associated files are released separately.

33\. Bo Gou, Jicheng Zhang, et al. (2026). [<u>Spatio-Temporal Fusion Model (STFM): official code and EV9V documentation</u>](https://github.com/bgx666/stfm/blob/main/README.md). GitHub: bgx666/stfm. Companion software/documentation resource, categorized with dataset resources; anatomical expansions conflict with portions of the dataset card.

34\. DICOM Standards Committee (2026). [<u>DICOM PS3.3 2026c — C.7.6.5 Cine Module</u>](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.5.html). Digital Imaging and Communications in Medicine (DICOM) Standard, Part 3. Current online standard inspected 12 September 2026; URL is a living current-version URL.

35\. DICOM Standards Committee (2026). [<u>DICOM PS3.3 2026c — C.7.6.3 Image Pixel Module</u>](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.3.html). Digital Imaging and Communications in Medicine (DICOM) Standard, Part 3. Current online standard inspected 12 September 2026; URL is a living current-version URL.

36\. Jwan A. Naser et al. (2024). [<u>Artificial intelligence-based classification of echocardiographic views</u>](https://academic.oup.com/ehjdh/article/5/3/260/7614559). European Heart Journal – Digital Health 5(3), 260–269.

37\. Preshen Naidoo et al. (2025). [<u>Spatiotemporal Contrastive Learning for Echocardiography View Classification</u>](https://doi.org/10.1007/978-3-032-00656-1_18). AIiH 2025, LNCS 16039, pp. 247–260. First online 20 August 2025; proceedings volume bears a 2026 publication year.

38\. Jaeik Jeon et al. (2023). [<u>Improving Out-of-Distribution Detection in Echocardiographic View Classication through Enhancing Semantic Features</u>](https://arxiv.org/html/2308.16483v2). arXiv:2308.16483v2. Version 2, revised 24 November 2023; title spelling follows the primary source. Version 1 has a different title and experimental formulation.

39\. Milos Vukadinovic et al. (2026). [<u>Comprehensive echocardiogram evaluation with view primed vision language AI</u>](https://www.nature.com/articles/s41586-025-09850-x). Nature 650, 970–977. First online 11 November 2025; final issue citation is 2026.

40\. Peiyao Wang, Yuewei Lin, Erik Blasch, Jie Wei, and Haibin Ling (2024). [<u>Efficient Temporal Action Segmentation via Boundary-aware Query Voting</u>](https://proceedings.neurips.cc/paper_files/paper/2024/hash/42770daf4a3384b712ea9c36e9279998-Abstract-Conference.html). Advances in Neural Information Processing Systems (NeurIPS).

41\. Yonatan Geifman and Ran El-Yaniv (2019). [<u>SelectiveNet: A Deep Neural Network with an Integrated Reject Option</u>](https://proceedings.mlr.press/v97/geifman19a.html). ICML, Proceedings of Machine Learning Research 97.

[^1]: Bo Gou, Jicheng Zhang, et al. (2026). [<u>Echocardiographic Videos of Nine Views (EV9V)</u>](https://huggingface.co/datasets/bgx666/EV9V/blob/main/README.md)

[^2]: Shahram Ebadollahi, Shih-Fu Chang, and Henry Wu (2002). [<u>Echocardiogram Videos: Summarization, Temporal Segmentation and Browsing</u>](https://www.ee.columbia.edu/dvmm/publications/02/icip02_shahram.pdf)

[^3]: Gino E. Jansen et al. (2024). [<u>Automated echocardiography view classification and quality assessment with recognition of unknown views</u>](https://pmc.ncbi.nlm.nih.gov/articles/PMC11364256/)

[^4]: Shanshan Song, Yi Qin, Honglong Yang, Taoran Huang, Hongwen Fei, and Xiaomeng Li (2025). [<u>EchoViewCLIP: Advancing Video Quality Control through High-performance View Recognition of Echocardiography</u>](https://papers.miccai.org/miccai-2025/paper/4443_paper.pdf)

[^5]: Ali Madani et al. (2018). [<u>Fast and accurate view classification of echocardiograms using deep learning</u>](https://www.nature.com/articles/s41746-017-0013-1)

[^6]: Jeffrey Zhang et al. (2018). [<u>Fully Automated Echocardiogram Interpretation in Clinical Practice: Feasibility and Diagnostic Accuracy</u>](https://pmc.ncbi.nlm.nih.gov/articles/PMC6200386/)

[^7]: Neda Azarmehr et al. (2021). [<u>Neural architecture search of echocardiography view classifiers</u>](https://doi.org/10.1117/1.JMI.8.3.034002)

[^8]: James P. Howard et al. (2020). [<u>Improving ultrasound video classification: an evaluation of novel deep learning methods in echocardiography</u>](https://jmai.amegroups.org/article/view/5205/html)

[^9]: Agisilaos Chartsias et al. (2021). [<u>Contrastive Learning for View Classification of Echocardiograms</u>](https://arxiv.org/abs/2108.03124)

[^10]: Zhe Huang, Mary-Joy Sidhom, Benjamin S. Wessler, and Michael C. Hughes (2023). [<u>Fix-A-Step: Semi-supervised Learning From Uncurated Unlabeled Data</u>](https://proceedings.mlr.press/v206/huang23c.html)

[^11]: Bo Gou, Jicheng Zhang, et al. (2026). [<u>Spatio-Temporal Fusion Model for Standard View Classification of Echocardiographic Videos</u>](https://arxiv.org/html/2606.17437v1)

[^12]: Preshen Naidoo et al. (2026). [<u>Robust fine-grained echocardiographic view classification with supervised contrastive learning</u>](https://doi.org/10.1016/j.media.2026.104006)

[^13]: Zhiyuan Gao, Dominic Yurk, and Yaser S. Abu-Mostafa (2026). [<u>Learning from Scarce Labels: Multi-View Echocardiography for Ejection Fraction Prediction</u>](https://arxiv.org/html/2609.02969)

[^14]: Yazan Abu Farha and Juergen Gall (2019). [<u>MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation</u>](https://arxiv.org/abs/1903.01945)

[^15]: Fangqiu Yi, Hongyu Wen, and Tingting Jiang (2021). [<u>ASFormer: Transformer for Action Segmentation</u>](https://arxiv.org/abs/2110.08568)

[^16]: Zhenzhi Wang, Ziteng Gao, Limin Wang, Zhifeng Li, and Gangshan Wu (2020). [<u>Boundary-Aware Cascade Networks for Temporal Action Segmentation</u>](https://github.com/MCG-NJU/BCN)

[^17]: Yuhan Shen and Ehsan Elhamifar (2025). [<u>Understanding Multi-Task Activities from Single-Task Videos</u>](https://openaccess.thecvf.com/content/CVPR2025/papers/Shen_Understanding_Multi-Task_Activities_from_Single-Task_Videos_CVPR_2025_paper.pdf)

[^18]: Dan Hendrycks and Kevin Gimpel (2017). [<u>A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks</u>](https://arxiv.org/abs/1610.02136)

[^19]: Weitang Liu, Xiaoyun Wang, John D. Owens, and Yixuan Li (2020). [<u>Energy-based Out-of-distribution Detection</u>](https://arxiv.org/abs/2010.03759)

[^20]: Dan Hendrycks, Mantas Mazeika, and Thomas Dietterich (2019). [<u>Deep Anomaly Detection with Outlier Exposure</u>](https://arxiv.org/abs/1812.04606)

[^21]: Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger (2017). [<u>On Calibration of Modern Neural Networks</u>](https://proceedings.mlr.press/v70/guo17a.html)

[^22]: Yonatan Geifman and Ran El-Yaniv (2017). [<u>Selective Classification for Deep Neural Networks</u>](https://arxiv.org/abs/1705.08500)

[^23]: Yaniv Ovadia et al. (2019). [<u>Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift</u>](https://arxiv.org/abs/1906.02530)

[^24]: Anastasios N. Angelopoulos, Stephen Bates, Adam Fisch, Lihua Lei, and Tal Schuster (2024). [<u>Conformal Risk Control</u>](https://openreview.net/forum?id=33XGfHLtZg)

[^25]: Anastasios N. Angelopoulos, Stephen Bates, Emmanuel J. Candès, Michael I. Jordan, and Lihua Lei (2021). [<u>Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control</u>](https://arxiv.org/abs/2110.01052)

[^26]: Zhe Huang, Gary Long, Benjamin S. Wessler, and Michael C. Hughes (2022). [<u>TMED-2 dataset</u>](https://tmed.cs.tufts.edu/tmed_v2.html)

[^27]: CAMUS project team (Simon Leclerc, Olivier Bernard, and collaborators) (2019). [<u>CAMUS project – Database</u>](https://www.creatis.insa-lyon.fr/Challenge/camus/databases.html)

[^28]: Unity Imaging Collaborative (n.d.). [<u>Unity Imaging Collaborative – Open-Access Datasets for AI in Cardiology</u>](https://data.unityimaging.net/)

[^29]: Brian Gow, Tom Pollard, Nathaniel Greenbaum, Benjamin Moody, Ahram Han, Jonathan W. Waks, Alistair Johnson, Elizabeth Herbst, Parastou Eslami, Ashish Chaudhari, Tanner Carbonati, Seth Berkowitz, Roger Mark, and Steven Horng (2026). [<u>MIMIC-IV-Echo: Echocardiogram Matched Subset</u>](https://physionet.org/content/mimic-iv-echo/1.0.1/)

[^30]: Sampath Rapuri, Sofia Sapeta Dias, Maria Salomé Carvalho, Malcolm Lizzappi, Carl Harris, and Robert Stevens (2026). [<u>Structured Viewing Classification Annotations From the MIMIC-IV-ECHO Dataset (ECHOVIEW)</u>](https://physionet.org/content/echoview/0.1/)

[^31]: THRIVE Centre / Preshen Naidoo and collaborators (2026). [<u>TTE47 — Dataset Card & Reference Gallery</u>](https://www.thrive-centre.com/datasets/TTE47)

[^32]: Elias Stenhede, Joanna Sulkowska, Eivind Bjørkan Orstad, Henrik Schirmer, and Arian Ranjbar (2026). [<u>EchoXFlow: A Beamspace Echocardiography Dataset for Cardiac Motion, Flow, and Function</u>](https://arxiv.org/abs/2605.05447)

[^33]: Bo Gou, Jicheng Zhang, et al. (2026). [<u>Spatio-Temporal Fusion Model (STFM): official code and EV9V documentation</u>](https://github.com/bgx666/stfm/blob/main/README.md)

[^34]: Bo Gou, Jicheng Zhang, et al. (2026). [<u>Spatio-Temporal Fusion Model for Standard View Classification of Echocardiographic Videos</u>](https://arxiv.org/html/2606.17437v1)

[^35]: Gino E. Jansen et al. (2024). [<u>Automated echocardiography view classification and quality assessment with recognition of unknown views</u>](https://pmc.ncbi.nlm.nih.gov/articles/PMC11364256/)

[^36]: DICOM Standards Committee (2026). [<u>DICOM PS3.3 2026c — C.7.6.5 Cine Module</u>](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.5.html)

[^37]: DICOM Standards Committee (2026). [<u>DICOM PS3.3 2026c — C.7.6.3 Image Pixel Module</u>](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.3.html)

[^38]: Shahram Ebadollahi, Shih-Fu Chang, and Henry Wu (2002). [<u>Echocardiogram Videos: Summarization, Temporal Segmentation and Browsing</u>](https://www.ee.columbia.edu/dvmm/publications/02/icip02_shahram.pdf)

[^39]: Ali Madani et al. (2018). [<u>Fast and accurate view classification of echocardiograms using deep learning</u>](https://www.nature.com/articles/s41746-017-0013-1)

[^40]: Jeffrey Zhang et al. (2018). [<u>Fully Automated Echocardiogram Interpretation in Clinical Practice: Feasibility and Diagnostic Accuracy</u>](https://pmc.ncbi.nlm.nih.gov/articles/PMC6200386/)

[^41]: James P. Howard et al. (2020). [<u>Improving ultrasound video classification: an evaluation of novel deep learning methods in echocardiography</u>](https://jmai.amegroups.org/article/view/5205/html)

[^42]: Neda Azarmehr et al. (2021). [<u>Neural architecture search of echocardiography view classifiers</u>](https://doi.org/10.1117/1.JMI.8.3.034002)

[^43]: Jwan A. Naser et al. (2024). [<u>Artificial intelligence-based classification of echocardiographic views</u>](https://academic.oup.com/ehjdh/article/5/3/260/7614559)

[^44]: Agisilaos Chartsias et al. (2021). [<u>Contrastive Learning for View Classification of Echocardiograms</u>](https://arxiv.org/abs/2108.03124)

[^45]: Zhe Huang, Mary-Joy Sidhom, Benjamin S. Wessler, and Michael C. Hughes (2023). [<u>Fix-A-Step: Semi-supervised Learning From Uncurated Unlabeled Data</u>](https://proceedings.mlr.press/v206/huang23c.html)

[^46]: Preshen Naidoo et al. (2025). [<u>Spatiotemporal Contrastive Learning for Echocardiography View Classification</u>](https://doi.org/10.1007/978-3-032-00656-1_18)

[^47]: Preshen Naidoo et al. (2026). [<u>Robust fine-grained echocardiographic view classification with supervised contrastive learning</u>](https://doi.org/10.1016/j.media.2026.104006)

[^48]: Jaeik Jeon et al. (2023). [<u>Improving Out-of-Distribution Detection in Echocardiographic View Classication through Enhancing Semantic Features</u>](https://arxiv.org/html/2308.16483v2)

[^49]: Gino E. Jansen et al. (2024). [<u>Automated echocardiography view classification and quality assessment with recognition of unknown views</u>](https://pmc.ncbi.nlm.nih.gov/articles/PMC11364256/)

[^50]: Shanshan Song, Yi Qin, Honglong Yang, Taoran Huang, Hongwen Fei, and Xiaomeng Li (2025). [<u>EchoViewCLIP: Advancing Video Quality Control through High-performance View Recognition of Echocardiography</u>](https://papers.miccai.org/miccai-2025/paper/4443_paper.pdf)

[^51]: Bo Gou, Jicheng Zhang, et al. (2026). [<u>Spatio-Temporal Fusion Model for Standard View Classification of Echocardiographic Videos</u>](https://arxiv.org/html/2606.17437v1)

[^52]: Milos Vukadinovic et al. (2026). [<u>Comprehensive echocardiogram evaluation with view primed vision language AI</u>](https://www.nature.com/articles/s41586-025-09850-x)

[^53]: Zhiyuan Gao, Dominic Yurk, and Yaser S. Abu-Mostafa (2026). [<u>Learning from Scarce Labels: Multi-View Echocardiography for Ejection Fraction Prediction</u>](https://arxiv.org/html/2609.02969)

[^54]: Yazan Abu Farha and Juergen Gall (2019). [<u>MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation</u>](https://arxiv.org/abs/1903.01945)

[^55]: Fangqiu Yi, Hongyu Wen, and Tingting Jiang (2021). [<u>ASFormer: Transformer for Action Segmentation</u>](https://arxiv.org/abs/2110.08568)

[^56]: Zhenzhi Wang, Ziteng Gao, Limin Wang, Zhifeng Li, and Gangshan Wu (2020). [<u>Boundary-Aware Cascade Networks for Temporal Action Segmentation</u>](https://github.com/MCG-NJU/BCN)

[^57]: Peiyao Wang, Yuewei Lin, Erik Blasch, Jie Wei, and Haibin Ling (2024). [<u>Efficient Temporal Action Segmentation via Boundary-aware Query Voting</u>](https://proceedings.neurips.cc/paper_files/paper/2024/hash/42770daf4a3384b712ea9c36e9279998-Abstract-Conference.html)

[^58]: Yuhan Shen and Ehsan Elhamifar (2025). [<u>Understanding Multi-Task Activities from Single-Task Videos</u>](https://openaccess.thecvf.com/content/CVPR2025/papers/Shen_Understanding_Multi-Task_Activities_from_Single-Task_Videos_CVPR_2025_paper.pdf)

[^59]: Dan Hendrycks and Kevin Gimpel (2017). [<u>A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks</u>](https://arxiv.org/abs/1610.02136)

[^60]: Weitang Liu, Xiaoyun Wang, John D. Owens, and Yixuan Li (2020). [<u>Energy-based Out-of-distribution Detection</u>](https://arxiv.org/abs/2010.03759)

[^61]: Dan Hendrycks, Mantas Mazeika, and Thomas Dietterich (2019). [<u>Deep Anomaly Detection with Outlier Exposure</u>](https://arxiv.org/abs/1812.04606)

[^62]: Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger (2017). [<u>On Calibration of Modern Neural Networks</u>](https://proceedings.mlr.press/v70/guo17a.html)

[^63]: Yonatan Geifman and Ran El-Yaniv (2017). [<u>Selective Classification for Deep Neural Networks</u>](https://arxiv.org/abs/1705.08500)

[^64]: Yonatan Geifman and Ran El-Yaniv (2019). [<u>SelectiveNet: A Deep Neural Network with an Integrated Reject Option</u>](https://proceedings.mlr.press/v97/geifman19a.html)

[^65]: Yaniv Ovadia et al. (2019). [<u>Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift</u>](https://arxiv.org/abs/1906.02530)

[^66]: Anastasios N. Angelopoulos, Stephen Bates, Adam Fisch, Lihua Lei, and Tal Schuster (2024). [<u>Conformal Risk Control</u>](https://openreview.net/forum?id=33XGfHLtZg)

[^67]: Anastasios N. Angelopoulos, Stephen Bates, Emmanuel J. Candès, Michael I. Jordan, and Lihua Lei (2021). [<u>Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control</u>](https://arxiv.org/abs/2110.01052)
