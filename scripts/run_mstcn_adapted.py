#!/usr/bin/env python
"""M0: the official MS-TCN (yabufarha/ms-tcn at the pinned commit, model class and loss unchanged) trained with the
shared round-1 recipe on frozen features (2026-10-02 plan, Task 4). The data, optimiser, update budget and checkpoint
rule are the task adaptation; the original-recipe B4 models stay in the system table as separate rows.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import base_parser, load_cfg, new_run_dir  # noqa: E402

from echo_routing.config import load_paths  # noqa: E402
from echo_routing.temporal.dev_heads import MSTCNHead  # noqa: E402
from echo_routing.temporal.dev_run import add_common_args, prepare, run, settings_of  # noqa: E402


def main() -> None:
    ap = base_parser("Train the official MS-TCN with the round-1 recipe"); add_common_args(ap)
    args = ap.parse_args(); cfg = load_cfg(args)
    settings = settings_of(args, cfg)
    run_dir = new_run_dir(cfg, f"md_M0_lr{settings.lr:g}")
    inputs, ckpt_dir, device = prepare(args, cfg, "M0", settings, run_dir)
    repo = load_paths().repo_root / "third_party" / "ms-tcn"
    head = MSTCNHead(repo, in_dim=int(inputs.train.streams[0].feat.shape[1]), **cfg["official_heads"]["mstcn"])
    run("M0", head, inputs, cfg, settings, run_dir, ckpt_dir, device,
        extra_manifest={"upstream": {"repo": "yabufarha/ms-tcn", "commit": head.commit, "model": "MultiStageModel",
                                     "loss": "Trainer.train: CE + 0.15 clamped T-MSE per stage", **cfg["official_heads"]["mstcn"]}})


if __name__ == "__main__":
    main()
