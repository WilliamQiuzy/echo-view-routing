#!/usr/bin/env python
"""Populate the feature cache for chosen splits and variants using a frozen encoder checkpoint."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import base_parser, load_cfg  # noqa: E402

import torch  # noqa: E402

from echo_routing.config import gpu_memory_fraction, load_paths  # noqa: E402
from echo_routing.features.encoder import checkpoint_hash, load_checkpoint  # noqa: E402
from echo_routing.features.extract import extract_split  # noqa: E402
from echo_routing.manifest import load_manifest, task_subset  # noqa: E402


def main() -> None:
    ap = base_parser("Extract features/probabilities into the cache")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--splits", default="validation,test")
    ap.add_argument("--variants", default=None, help="comma list; default from config features.variants")
    ap.add_argument("--limit", type=int, default=None, help="max cines per split (smoke runs)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg(args); paths = load_paths()
    model, meta = load_checkpoint(args.ckpt)
    task = meta.get("task", cfg["task"]); label_col = "family5_index" if task == "family5" else "raw9_index"
    h = checkpoint_hash(args.ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    df = task_subset(load_manifest(paths.manifest_path), task)
    df = df[df["frames_dir"].notna() & (df["n_frames"] > 0)]
    splits = args.splits.split(","); variants = (args.variants or ",".join(cfg["features"]["variants"])).split(",")
    rows = df[df["split"].isin(splits)]
    if args.limit:
        rows = rows.groupby("split", group_keys=False).head(args.limit)
    for variant in variants:
        out = extract_split(model, device, rows, label_col, paths.cache_root, h, variant,
                            cfg["sampling"]["target_hz"], meta.get("image_size", cfg["preprocess"]["image_size"]),
                            cfg["features"]["batch_size"], cfg["features"]["num_workers"], args.overwrite,
                            meta={"ckpt": str(args.ckpt), "task": task, "classes": meta.get("classes")})
        print(f"variant={variant} -> {out} ({len(rows)} cines)", flush=True)
    print(f"ckpt_hash={h}")


if __name__ == "__main__":
    main()
