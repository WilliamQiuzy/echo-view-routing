#!/usr/bin/env python
"""Server-side: write demo/artifacts/ (small) = newest ladder metrics + reports, ready for remote/pull_results.sh."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echo_routing.config import load_paths  # noqa: E402

paths = load_paths()
runs = sorted(p for p in paths.runs_root.glob("*-baselines-*") if (p / "metrics").is_dir())
if not runs:
    raise SystemExit("no baselines run found")
src = runs[-1]; dst = paths.runs_root / "_demo" / src.name
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src / "metrics", dst / "metrics"); shutil.copytree(src / "reports", dst / "reports")
shutil.copy(src / "config.resolved.yaml", dst / "config.resolved.yaml")
print(f"demo bundle: {dst}")
