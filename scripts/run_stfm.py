#!/usr/bin/env python
"""B6 adapter: run the vendored STFM trainer as a subprocess with our data layout and memory cap (never edits third_party)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import base_parser, load_cfg, new_run_dir  # noqa: E402

from echo_routing.config import load_paths  # noqa: E402


def main() -> None:
    ap = base_parser("Run STFM (B6)"); ap.add_argument("--test-only", action="store_true"); ap.add_argument("--resume", default="")
    args = ap.parse_args(); cfg = load_cfg(args); paths = load_paths(); st = cfg["stfm"]
    repo = paths.repo_root / st["repo"]
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if head != st["commit"]:
        raise SystemExit(f"STFM commit {head} != pinned {st['commit']}")
    run_dir = new_run_dir(cfg, "b6_stfm"); save_dir = paths.checkpoints_root / run_dir.name
    cmd = [sys.executable, "trainSelective.py", "--data_path", str((paths.repo_root / st["data_path"]).resolve()) + "/",
           "--save_dir", str(save_dir)]
    for k, v in st["args"].items():
        cmd += [f"--{k}", str(v)]
    if args.test_only:
        cmd += ["--test_flag", "1", "--resume", args.resume]
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    (run_dir / "command.txt").write_text(" ".join(cmd) + f"\ncwd={repo}\ncommit={head}\n")
    print("running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=repo, env=env)
    (run_dir / "exit_code.txt").write_text(str(proc.returncode))
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
