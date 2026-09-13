#!/usr/bin/env python
"""Run EchoViewCLIP stage-1 training (official code, pinned) on EV9V in its own Python-3.10/torch-1.13 environment.

Only the dataset config differs from the authors' (third_party/adapters/echoviewclip/ev9v_stage1.yaml); the launch uses
one GPU instead of two with doubled gradient accumulation. Logs go to the run dir.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import base_parser, load_cfg, new_run_dir  # noqa: E402

from echo_routing.config import load_paths  # noqa: E402

PINNED = "f6a0d86074d7c906ae41546cd5f09f8b94da432a"


def main() -> None:
    ap = base_parser("EchoViewCLIP stage-1 on EV9V"); ap.add_argument("--only-test", action="store_true"); ap.add_argument("--resume", default=None)
    ap.add_argument("--adapter-config", default="ev9v_stage1.yaml", help="file under third_party/adapters/echoviewclip/")
    args = ap.parse_args(); cfg = load_cfg(args); paths = load_paths()
    repo = paths.repo_root / "third_party" / "EchoViewCLIP"; py = paths.repo_root / "envs" / "echoviewclip" / "bin" / "python"
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if head != PINNED:
        raise SystemExit(f"EchoViewCLIP commit {head} != pinned {PINNED}")
    run_dir = new_run_dir(cfg, "b7_echoviewclip_test" if args.only_test else "b7_echoviewclip_stage1")
    cfg_path = paths.repo_root / "third_party" / "adapters" / "echoviewclip" / args.adapter_config
    cmd = [str(py), "-m", "torch.distributed.launch", "--nproc_per_node=1", "--master_port=29264", "main_1.py", "-cfg", str(cfg_path), "--output", str(run_dir / "output")]
    if args.only_test:
        cmd += ["--only_test"]
    if args.resume:
        cmd += ["--resume", args.resume]
    env = dict(os.environ, PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="0")
    (run_dir / "command.txt").write_text(" ".join(cmd) + f"\ncwd={repo}\ncommit={head}\n")
    print("running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=repo, env=env)
    (run_dir / "exit_code.txt").write_text(str(proc.returncode)); raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
