#!/usr/bin/env python
"""Round 3, step 2: train the boundary-only branch on frozen features (configs/experiments/model_development_v3.yaml).

The network is M5's projection, three dilated convolutions and boundary head, trained with the boundary loss only.
Joins where the view class changes are positives, same-class joins strong negatives, and other transitions weak
negatives. Interval labels come from the frozen frame-level classifier of the same visual input (the cached
probabilities), never from this network. Data, optimiser, update budget and checkpoint rule are those of rounds 1–2.
Checkpoints are chosen on the selection half with the pre-declared threshold (boundary_decoding.checkpoint_threshold);
the deployed threshold is chosen afterwards on the calibration half by scripts/posterior_decoders.py --families bdec.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import base_parser, load_cfg, new_run_dir  # noqa: E402

from echo_routing.temporal.dev_heads import BoundaryOnlyHead  # noqa: E402
from echo_routing.temporal.dev_run import add_common_args, prepare, run, settings_of  # noqa: E402


def main() -> None:
    ap = base_parser("Train the boundary-only branch"); add_common_args(ap)
    args = ap.parse_args(); cfg = load_cfg(args)
    settings = settings_of(args, cfg)
    run_dir = new_run_dir(cfg, f"md_BO_lr{settings.lr:g}")
    inputs, ckpt_dir, device = prepare(args, cfg, "BO", settings, run_dir)
    bd = cfg["boundary_decoding"]
    model_cfg = {**cfg["boundary_only"], "in_dim": inputs.feature_dim}
    head = BoundaryOnlyHead(model_cfg, threshold=float(bd["checkpoint_threshold"]), nms_radius=int(bd["nms_radius"]))
    run("BO", head, inputs, cfg, settings, run_dir, ckpt_dir, device,
        extra_manifest={"model_cfg": model_cfg, "decode_threshold": float(bd["checkpoint_threshold"]),
                        "decode_nms_radius": int(bd["nms_radius"]), "labels": "frozen frame-level classifier (cached probabilities)"})


if __name__ == "__main__":
    main()
