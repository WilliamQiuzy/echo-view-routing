#!/usr/bin/env python
"""Train the boundary-controlled context model (M4) or its ablations M2 / M3 on frozen features (2026-10-02 plan,
Tasks 3-4).

  python scripts/train_boundary_context.py -c configs/experiments/boundary_context_v1.yaml --set seed=0 \
      --feature-manifest <feature_manifest.json> --protocol <protocol.json> --variant full

--variant plain | aux | full selects M2 | M3 | M4; --fixed-gate 0.5 replaces the learned gate (ablation); --augment adds
the training-only feature masking of the second step; --segment-pool adds the boundary-weighted segment pooling of
the output (round-1 follow-up: M5 = full + pooling, M3p = aux + pooling); --overfit N runs the small-sample overfit
check on the first N training streams instead of a training run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import base_parser, load_cfg, new_run_dir  # noqa: E402

from echo_routing.temporal.dev_data import feature_mask_augment  # noqa: E402
from echo_routing.temporal.dev_heads import BoundaryContextHead  # noqa: E402
from echo_routing.temporal.dev_run import add_common_args, prepare, run, settings_of  # noqa: E402
from echo_routing.temporal.dev_train import overfit_check  # noqa: E402

ROWS = {"plain": "M2", "aux": "M3", "full": "M4"}


def main() -> None:
    ap = base_parser("Train the boundary-controlled context model"); add_common_args(ap)
    ap.add_argument("--variant", choices=tuple(ROWS), default="full")
    ap.add_argument("--fixed-gate", type=float, default=None)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--segment-pool", action="store_true")
    ap.add_argument("--overfit", type=int, default=0)
    args = ap.parse_args(); cfg = load_cfg(args)
    v = cfg["variants"][args.variant]
    model_cfg = {**cfg["boundary_context"], "use_path": bool(v["use_path"]), "fixed_gate": args.fixed_gate,
                 "segment_pool": bool(args.segment_pool)}
    row = ({"plain": "M2p", "aux": "M3p", "full": "M5"}[args.variant] if args.segment_pool else ROWS[args.variant]) \
        + ("_fixedgate" if args.fixed_gate is not None else "") + ("_aug" if args.augment else "")
    head = BoundaryContextHead(model_cfg, float(v["boundary_weight"]),
                               name=f"boundary_context_{args.variant}{'_pool' if args.segment_pool else ''}")
    settings = settings_of(args, cfg)
    run_dir = new_run_dir(cfg, f"md_{row}{'_overfit' if args.overfit else ''}_lr{settings.lr:g}")
    inputs, ckpt_dir, device = prepare(args, cfg, row, settings, run_dir)
    if args.overfit:
        res = overfit_check(head, list(inputs.train.streams[:args.overfit]), device, seed=settings.seed)
        (run_dir / "overfit.json").write_text(json.dumps({"row": row, "model_cfg": model_cfg, **res}, indent=1))
        print(json.dumps({k: v for k, v in res.items() if k != "history"}, indent=1))
        return
    augment = feature_mask_augment(**cfg["augment"]["feature_mask"]) if args.augment else None
    run(row, head, inputs, cfg, settings, run_dir, ckpt_dir, device, augment=augment,
        extra_manifest={"variant": args.variant, "model_cfg": model_cfg, "boundary_weight": float(v["boundary_weight"]),
                        "augment": cfg["augment"]["feature_mask"] if args.augment else None})


if __name__ == "__main__":
    main()
