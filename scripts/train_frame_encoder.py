#!/usr/bin/env python
"""Train the ResNet-18 frame classifier (B0 encoder). Resumable via --resume <last.pt>."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import base_parser, load_cfg, new_run_dir  # noqa: E402

from echo_routing.audit.label_map import load_ontology  # noqa: E402
from echo_routing.config import load_paths  # noqa: E402
from echo_routing.features.train_encoder import TrainConfig, train  # noqa: E402
from echo_routing.manifest import load_manifest  # noqa: E402


def main() -> None:
    ap = base_parser("Train frame encoder")
    ap.add_argument("--resume", default=None, help="path to last.pt to resume from")
    ap.add_argument("--run-dir", default=None, help="reuse an existing run dir (with --resume)")
    args = ap.parse_args()
    cfg = load_cfg(args)
    paths = load_paths()
    enc = cfg["encoder"]
    tcfg = TrainConfig.from_dict({
        "task": cfg["task"], "backbone": enc["backbone"], "image_size": cfg["preprocess"]["image_size"],
        "frames_per_video": cfg["sampling"]["train_frames_per_video"], "videos_per_epoch": enc.get("videos_per_epoch"),
        "batch_size": enc["batch_size"], "lr": enc["lr"], "weight_decay": enc["weight_decay"], "epochs": enc["epochs"],
        "patience": enc["patience"], "seed": cfg["seed"], "num_workers": enc["num_workers"],
        "eval_frames_per_video": cfg["sampling"]["eval_frames_per_video"], "amp": enc["amp"],
    })
    run_dir = Path(args.run_dir) if args.run_dir else new_run_dir(cfg)
    ckpt_dir = paths.checkpoints_root / run_dir.name
    print(f"run_dir={run_dir}\nckpt_dir={ckpt_dir}\n{tcfg}", flush=True)
    best = train(tcfg, load_manifest(paths.manifest_path), load_ontology(), ckpt_dir, resume=args.resume)
    (run_dir / "best_checkpoint.txt").write_text(str(best))
    print(f"best checkpoint: {best}")


if __name__ == "__main__":
    main()
