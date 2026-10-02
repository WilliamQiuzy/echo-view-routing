#!/usr/bin/env python
"""M1: the official ASRF (yiskw713/asrf at the pinned commit; model, losses and post-processing imported unchanged)
trained with the shared round-1 recipe on frozen features (2026-10-02 plan, Task 4).

Upstream defaults kept: 64 features, 10 layers, 4 stages per branch, CE with median-frequency class weights plus the
Gaussian-similarity TMSE on the input features, boundary BCE with pos_weight = 1 / positive share, lambda_b = 0.1,
and decoding by refinement_with_boundary at boundary_th 0.5. Class weights and pos_weight are computed by the official
rules from the epoch-0 training batches (streams and native windows). --augment adds the second-step feature masking.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import base_parser, load_cfg, new_run_dir  # noqa: E402

from echo_routing.config import load_paths  # noqa: E402
from echo_routing.temporal.dev_data import feature_mask_augment  # noqa: E402
from echo_routing.temporal.dev_heads import ASRFHead, asrf_weights  # noqa: E402
from echo_routing.temporal.dev_run import add_common_args, prepare, run, settings_of  # noqa: E402


def main() -> None:
    ap = base_parser("Train the official ASRF with the round-1 recipe"); add_common_args(ap)
    ap.add_argument("--augment", action="store_true")
    args = ap.parse_args(); cfg = load_cfg(args)
    settings = settings_of(args, cfg); row = "M1" + ("_aug" if args.augment else "")
    run_dir = new_run_dir(cfg, f"md_{row}_lr{settings.lr:g}")
    inputs, ckpt_dir, device = prepare(args, cfg, row, settings, run_dir)
    first = [s for b in inputs.train.epoch_batches(settings.seed, 0, settings.native_per_epoch, settings.native_window,
                                                    settings.streams_per_batch, settings.natives_per_batch) for s in b]
    class_weight, pos_weight = asrf_weights(first, num_classes=5)
    oc = cfg["official_heads"]["asrf"]
    head = ASRFHead(load_paths().repo_root / "third_party" / "asrf", int(first[0].feat.shape[1]), class_weight, pos_weight,
                    lambda_b=float(oc["lambda_b"]), boundary_th=float(oc["boundary_th"]), device=device)
    head.decoding = f"official refinement_with_boundary, boundary_th {oc['boundary_th']}"
    augment = feature_mask_augment(**cfg["augment"]["feature_mask"]) if args.augment else None
    run(row, head, inputs, cfg, settings, run_dir, ckpt_dir, device, augment=augment,
        extra_manifest={"upstream": {"repo": "yiskw713/asrf", "commit": head.commit, **oc},
                        "class_weight": class_weight.tolist(), "pos_weight": float(pos_weight),
                        "augment": cfg["augment"]["feature_mask"] if args.augment else None})


if __name__ == "__main__":
    main()
