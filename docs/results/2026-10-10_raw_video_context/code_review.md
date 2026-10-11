# Implementation review before training

This is the independent implementation review that the task sheet requires before the queue
(`docs/superpowers/plans/2026-10-10-raw-video-recognition.md`, Task 2). An independent code-review agent read commit
c527046 against the task sheet, the configuration, torchvision 0.29's MViT source and the reused helpers. It ran the
round's CPU tests (13 passed) and small synthetic probes. It had no GPU and no EchoPrime checkpoint, so the GPU checks
rest on the check record, not on this review. When it reviewed the code, only the preparation step had run, and that
step produces no training metric.

## Verdict

The verdict was "pass with fixes". There were three medium findings and no critical, high or low finding. All three
concern the bookkeeping that gates the queue, the run identity and the budget ledger. None concerns what is computed or
trained. The commit that adds this record fixes all three before the final check run and before any formal training.

## What the review confirmed

- **Model.** The encoder is `mvit_v2_s()` with `head[-1]` replaced by Linear(768, 512) and the EchoPrime checkpoint
  loaded strictly, as in `load_for_finetuning.py`. The marker stem takes its kernel, stride, padding and dilation from
  `conv_proj`, has no bias and starts at zero. The hook exists only inside `embed` and is removed in `finally`, also
  after an exception. With zero marker weights the 512-d output equals the plain encoder's bit for bit. The forward
  takes only `(rgb, marker)`.
- **Inputs.** The marker slot and the target slots come from `s` and `c` only. Every `s == c` slot keeps its frame in
  every group, and images move with their gamma. The training and inference permutations match the task sheet's RNG
  rules exactly.
- **Data.** The allowlist is enforced in preparation and training. Frame paths come only from the `video_id` and
  `frames_dir` columns. Frame labels are read in preparation only to check that each target equals its requested
  sample's label, and they are not copied into the training manifests. The target is the SC manifest's target at `c`.
- **Pixels.** Preparation and training use the same frame key and slot order, so the per-window pixel digest is a real
  end-to-end check.
- **Training.** Both optimiser groups exist from update 0. The schedule reproduces the warm-up and cosine rule. The
  encoder forward is never under `no_grad`, so frozen encoder parameters get no gradient while the marker still learns
  through the trunk. The freeze is recomputed from the update index, so a resume at any epoch boundary restores it.
- **Determinism and resume.** fp32, TF32 off, cuDNN deterministic and non-benchmark, deterministic algorithms with
  warnings, and `CUBLAS_WORKSPACE_CONFIG` set before CUDA starts. The resume state holds the model, the optimiser, the
  CPU and CUDA RNG states, the update count and the log, and it is tied to the run identity.
- **Ids.** Every field written to `index.json`, `check.json` and their copies here is a digest, a count or a version.
- **Queue.** The order and the skip and resume rules match the task sheet, and the queue refuses to start without a
  passing check record of the current identities.

## Findings and fixes

1. **Medium: the run identity hashed the whole configuration section.** Edits to the evaluation, check or budget text
   would have changed every run's identity, although no run would compute anything differently. The queue would then
   have refused to start or stopped recognising finished runs. *Fix.* The identity now covers only the training
   sections (task sheet, sources, classes, model, inputs, groups, permutation, data, training and determinism).
2. **Medium: the code digest left out parts of the training script.** It covered selected functions but not the
   training loop, the pixel-digest check, the model builder, the context or the loader. A later change there would
   not have changed the identity. *Fix.* The digest now covers the model module and the training script in full, plus
   the shared frame, path and schedule functions they call.
3. **Medium: the budget counted only the last check run.** Each check overwrote the check hours, so earlier attempts
   would have been missing from the queue's projection. *Fix.* Every check run appends its hours to a ledger, and the
   queue sums the ledger.

## The boundary check after the first check run

The first check run, made while this review was under way, found one hard failure in the short-schedule boundary
check. After the encoder was unfrozen, the resumed run differed from the uninterrupted run by 2.04e-5 in its
parameters, above the 1e-5 that the configuration had applied to this check. Two uninterrupted runs differed from each
other by the same 2.04e-5, and all losses agreed within 4.8e-7. The only operator flagged without a deterministic CUDA
kernel was `max_pool3d_with_indices_backward_cuda`, the max-pooled skip connection of three MViT blocks. Its gradient
noise can flip the sign of near-zero gradient elements, and Adam then moves those elements by up to the learning rate
in either direction.

The task sheet asks for an operator without a deterministic kernel to be recorded with its tolerance. The boundary
check therefore now has two parts.

- **Exact state round-trip.** Reloading a resume file must restore the weights, the optimiser moments and step counts,
  and the CPU and CUDA RNG states bit for bit.
- **Training agreement.** The resumed and the repeated runs must agree with one go within 1e-4 in loss. Their
  parameters must stay within 2.1 times each optimiser group's summed learning rate, the most two Adam trajectories
  can drift apart when they differ only by the sign of near-zero gradient elements.

The six-update check on the formal schedule keeps the task sheet's 1e-5. It passed in all three groups, with parameter
differences of about 1e-9, because the encoder is frozen during those updates.
