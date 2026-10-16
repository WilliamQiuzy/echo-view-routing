# Local Python environment recovery — 2026-10-15

Audience: project owner and team. This is an operations record, not a modeling experiment.

## Problem and repair

The local disk was nearly full. The original environment's `torch/__init__.py` was an iCloud `dataless` placeholder, so local Torch imports stalled. The repository remains on Desktop; this repair moves only the Python environment outside Desktop.

The replacement environment is at:

```text
/Users/williamqiu/.local/share/venvs/echo-view-routing-recovery/env
```

The repository's `.venv` is now a symlink to that directory. Existing commands such as `.venv/bin/python -m pytest ...` therefore continue to work. Packages were installed with uv's `--link-mode copy`, so this environment does not depend on package-cache files remaining present.

The original directory was renamed in place and preserved at:

```text
/Users/williamqiu/Desktop/Harvard/Capstone/echo-view-routing/.venv.icloud-backup-20261015
```

The package snapshot, recovery paths and verification record are outside the repository in `/Users/williamqiu/.local/share/venvs/echo-view-routing-recovery/`. All 48 package versions recorded from the original environment match the replacement. Dependency resolution added nine transitive packages, and the project was installed in editable mode. This restores the recorded Mac versions; it does not establish parity with the CUDA server environment.

Local `.git/info/exclude` entries hide the environment symlink and backup. No tracked dependency specification or experiment configuration changed.

## Verification

- Python 3.12.13 imports Torch 2.14.0, Torchvision 0.29.0, NumPy 2.5.3 and `echo_routing`.
- `uv pip check` reports compatible dependencies.
- A CPU ResNet-18 forward pass accepts a `1 × 3 × 224 × 224` tensor and returns a finite `1 × 1000` output. No pretrained weights were downloaded for this check.
- The following command completed with exit code 0. Its three files contain 23, 30 and 21 tests respectively: **74 tests passed**.

```sh
.venv/bin/python -m pytest \
  tests/test_fact_video_feature.py \
  tests/test_fact_video_feature_revision.py \
  tests/test_report_fact_video_feature.py -q
```

Warnings concerned the unchanged official FACT `dropout2d` usage and permission failures while pytest cleaned older shared temporary directories. Neither caused a test failure. Those older directories were not manually removed.

## Disk space and limits

Standard uv cache pruning and pip cache purging removed about 7 GB of downloadable package caches. No project data, checkpoints, result files or unrelated user documents were deleted. Available space rose from about 13 GiB to 18 GiB immediately after cleanup; a later reading was 20 GiB. These are filesystem snapshots, not reserved capacity. The disk is still about 98% full.

The old environment remains available as a backup, although its offloaded files may still need iCloud downloads. To undo the path change, first verify `.venv` is the symlink described above, remove only that symlink, and rename the preserved directory back to `.venv`. Do not delete the backup as part of routine setup.

This repair did not change the server environment, frozen weights, source data or reported model results. It did not launch training. A06-R1 remains closed under its recorded decision; its report and experiment-index entry were not rewritten.
