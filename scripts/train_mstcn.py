#!/usr/bin/env python
"""Train B4 (MS-TCN) on constructed training streams built from cached train-split features, then evaluate via the ladder."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import base_parser, load_cfg, new_run_dir  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from echo_routing.compose.recipe_builder import Pool, make_pair_recipes  # noqa: E402
from echo_routing.compose.stream_assembler import render  # noqa: E402
from echo_routing.config import gpu_memory_fraction, load_paths  # noqa: E402
from echo_routing.features.cache import load_video_features, read_index, variant_dir  # noqa: E402
from echo_routing.temporal.baselines.b4_mstcn import train_mstcn  # noqa: E402


def main() -> None:
    ap = base_parser("Train MS-TCN (B4)"); ap.add_argument("--ckpt-hash", required=True)
    args = ap.parse_args(); cfg = load_cfg(args); paths = load_paths(); rc = cfg["recipes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    idx = read_index(variant_dir(paths.cache_root, args.ckpt_hash, "orig"))
    cache = {}

    def loader(v, var):
        if (v, var) not in cache:
            cache[(v, var)] = load_video_features(variant_dir(paths.cache_root, args.ckpt_hash, var) / f"{v}.npz")
        return cache[(v, var)]

    def bank(split, n_per_cell, seed, reuse):
        sub = idx[idx["split"] == split]
        pool = Pool.from_index(sub["video_id"], sub["label_index"], sub["n_samples"])
        rec = make_pair_recipes(pool, n_per_cell, np.random.default_rng(seed), rc["edit_variant"], int(rc["min_len"]), int(rc["max_len"]), reuse=reuse)
        return [render(r, loader) for r in rec]

    train = bank("train", int(cfg["mstcn"]["n_train_per_cell"]), int(rc["seed"]) + 100, reuse=True)
    val = bank("validation", int(rc["n_per_cell"]), int(rc["seed"]), reuse=False)
    num_classes = train[0].prob.shape[1]
    run_dir = new_run_dir(cfg, "b4_mstcn")
    best = train_mstcn(train, val, num_classes, cfg, paths.checkpoints_root / run_dir.name, device)
    (run_dir / "best_checkpoint.txt").write_text(str(best))
    print(json.dumps({"train_streams": len(train), "val_streams": len(val), "best": str(best)}))


if __name__ == "__main__":
    main()
