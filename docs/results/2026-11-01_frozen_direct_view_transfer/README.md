# A53 frozen direct-view feature supplement

Execution pending. A08 remains the working reference; no A53efficacy claim yet.

A51's manually supervised source classifier retains its failed sourceprotection verdict. POST-A52-D independently found a limited same-recording score-ranking increment(117correlatedpairs/5videos), while its proposed dense16slot supervision failed quantity qualification. A53newly asks whether source-trained **frozen features** remain useful after their classifiers are discarded; it doesnotrevise either failed gate.

Arms: G=[originalG512,zero512], L=[G512,A51L512], W=[G512,A51W512]. OriginalGfeatures/scalers are identical acrossarms. All three video trunks/marker/projection and oldheads are frozen; source classifiers are discarded. Only newFP64linear1025heads learn, sixheads across original two recording/session folds,600updates/onepositive+threenegative Base-A4C run plan. Ghas513activeparams plus512structuralzeros; L/W1025active. L−Gpractical utility has a capacitydifference; L−Wis the matched source-label contrast. No loss, cutoff, learnedgate or test-time group normalization.

Full original16frame inputs/crop/marker/gamma/normalization and complete duration/multiplicity memberships are preserved. All5030/5437rows are selected only from oldBase mapping, not newmodel/truth. FrozenL/Wbackbone extraction counts20934realwindowforwards including oppositefold covariates before headfit; this isnotzero heldout model inference. Gcachedembeddings reused onlyafter freshFITparity. Source133markerpointA5C/A4Cresults are not target103/1413results.

Fit-only checks: originalGfresh16cache difference1.68e−6, explicitwindow-vs-pooledlineargradient1.33e−15, repeat/forcedsamefeatures/disk3+3resume exact. Current engineering recheck12GPUwindows plus48previous current-science checks=60; separate4CPUFITfeature checks planned. All scientific inputs preserved through source/code provenance fixes, failures charged. Input/extractor actual Git7aeb30d precedes formal frozenfeature extraction; model/heads freeze follows onlyafter featureaudit and6×600complete.

One31401windowproduction score readout then predeclared independent31401NumPyalgebraic reconstructions of the same frozen cached-feature logits; latter is notzero classifier arithmetic and hasnofreshneuralbackbone forward. Exact pairedfix/harm/source/fullF1/native/boundary/short/gate audits follow. Whole-feature/scaler snapshots precede training; allhead/source/base/code/input/plan hashes precede productionreadout. Only strictabsolute fullsystem success and separate sourcecontribution can support respective claims. This is repeatedly used historical training-source evidence, not fresh clinical validation. No automatic deployment or seedexpansion.

SamephaseCPU6h/GPU4h/newpeak5.5GB with all inherited A51/A52costs retained, noreset. IDs/media/featuretensors/weights/per-case outputs stayserver-only. No protecteddataset, teammatefile or website change.
