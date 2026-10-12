# Implementation review of the dense target inference round

An independent reviewer read commit ff65af1 against the task sheet
(`docs/superpowers/plans/2026-10-12-dense-target-inference.md`) and the fixed configuration
(`configs/experiments/dense_target_inference_v1.yaml`) before any dense output existed. The reviewer read every new file
and the reused source modules, ran the tests on the committed snapshot, and compared the digests of all files behind the
two identities between the local copy, the commit and the server. The reviewer also read the first passed check record
and the log of a CPU dry run of the report path. No file was edited by the reviewer.

## Verdict

The review passed the code, and the full run may start. It found no critical and no high-severity problem.

## Findings and what was done

1. **Medium, process.** The local working tree held an edit made after ff65af1. It renamed the offset table headers and
   added a test assertion. The reviewer noted that committing it would change the run identity, so a new check would be
   needed. The edit was deliberate: the CPU dry run had shown that the header "Mean |offset| (samples)" broke the
   markdown columns. It was committed in 2d181ab with the test, and the check was run again before the full run.
2. **Low.** The budget projection counted 36 B3 decodes, but the report decodes 18 (three conditions, three seeds, two
   halves). The doubled count only padded the projection by about 13 seconds. 2d181ab counts 18.
3. **Low.** The configured batch size was read only into the inference identity and never compared with the code's
   batch of 32. 2d181ab stops the round when the two differ.

## What the reviewer confirmed

- **Windows.** The old centres follow `window_centres(n, 5)` and the dense centres are every sample. Each window uses
  `window_frames(c, n, 16, 16)`, and the marker comes from `s` and `c` only. The old windows equal the dense windows at
  the old centres, and every sample gets exactly one output. Constructed streams assemble their per-sample references
  from the recipe before any window is taken.
- **Mappings.**
  - Sparse is the source round's mapping and reproduces the saved per-sample probabilities bit for bit.
  - Interpolated follows the task sheet and is checked for negative values, row sums and the old centres.
  - Dense equals the old probabilities at the old centres bit for bit.
- **Inference.** The models are in eval mode with no gradient and no optimiser, built as the source evaluation built
  them and loaded strictly. The model state is compared before and after inference. The forward receives only the
  frames and the marker, and no label, boundary or native flag decides where windows are requested.
- **Identities and caches.**
  - The inference identity covers every module that can change a logit and keys the saved outputs.
  - The run identity adds the whole configuration, the decoded pixels and the report code.
  - The full run refuses to start without a passed check of the same run identity.
- **Budget.** Every attempt writes its GPU hours to the ledger, also when it fails. The run projects the remaining
  hours before each model and half.
- **Report.**
  - B3 uses the inherited numbers only, and the Sparse rows are checked against the stored rows before any Dense
    difference is computed.
  - Each pair of compared systems is resampled with the same bootstrap draws. The consistency rule, the matched
    offsets, the separation of hit 5 from near 10, the four readings, the native stages, the feasibility recheck and
    the fixed readings match the configuration.
  - Ids are guarded before every write to the repository, and the case figures stay on the server.
- **Isolation.** Neither new file reads the test split or EchoNet-Pediatric, and nothing is written outside this round's
  directories.

## Residual risk named by the reviewer

The dry run set all three conditions to the Sparse outputs. Branches that only run when the conditions differ are
covered by the unit tests, but they were not exercised at full scale before the run.
