#!/usr/bin/env python
"""Train a ViT-family model (frame or clip level) from an experiment config's `vit:` section. Resumable."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echo_routing.config import load_paths  # noqa: E402

os.environ.setdefault("HF_HOME", str(load_paths().data_root / "hf_cache"))  # before timm / huggingface_hub import

from scripts._common import base_parser, load_cfg, new_run_dir  # noqa: E402

from echo_routing.audit.label_map import load_ontology  # noqa: E402
from echo_routing.manifest import load_manifest  # noqa: E402
from echo_routing.vit.train import ViTTrainConfig, train  # noqa: E402


def resolve_train_config(cfg: dict, repo_root: Path) -> ViTTrainConfig:
    v = dict(cfg["vit"])
    v.setdefault("task", cfg["task"]); v.setdefault("seed", cfg["seed"])
    v.setdefault("image_size", cfg["preprocess"]["image_size"])
    extra = dict(v.get("extra") or {})
    if extra.get("echoprime_ckpt"):
        extra["echoprime_ckpt"] = str((repo_root / extra["echoprime_ckpt"]).resolve())
    v["extra"] = extra
    return ViTTrainConfig.from_dict(v)


def main() -> None:
    ap = base_parser("Train a ViT-family model")
    ap.add_argument("--resume", default=None, help="path to last.pt")
    ap.add_argument("--run-dir", default=None, help="reuse an existing run dir (with --resume)")
    args = ap.parse_args(); cfg = load_cfg(args); paths = load_paths()
    tcfg = resolve_train_config(cfg, paths.repo_root)
    run_dir = Path(args.run_dir) if args.run_dir else new_run_dir(cfg)
    ckpt_dir = paths.checkpoints_root / run_dir.name
    print(f"run_dir={run_dir}\nckpt_dir={ckpt_dir}\n{tcfg}", flush=True)
    best = train(tcfg, load_manifest(paths.manifest_path), load_ontology(), ckpt_dir, resume=args.resume)
    (run_dir / "best_checkpoint.txt").write_text(str(best))
    print(f"best checkpoint: {best}")


if __name__ == "__main__":
    main()
