#!/usr/bin/env python
"""Evaluate one trained ViT-family checkpoint end to end: feature extraction (frame-wise encoders only, validation +
test, orig + gamma090) followed by predict_vit_streams.py. Prints the prediction dir and run dir of the last step."""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def run(args: list[str]) -> str:
    print("$ " + " ".join(args), flush=True)
    proc = subprocess.run([sys.executable, "-u", *args], cwd=REPO, capture_output=True, text=True)
    sys.stdout.write("".join(l + "\n" for l in proc.stdout.splitlines() if not l.startswith("  ")))
    if proc.returncode != 0:
        sys.stdout.write(proc.stderr[-4000:])
        raise SystemExit(f"step failed ({proc.returncode}): {' '.join(args)}")
    return proc.stdout


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); a = ap.parse_args()
    import torch
    spec = torch.load(REPO / a.ckpt, map_location="cpu", weights_only=False)["spec"]
    extra: list[str] = []
    if spec["name"] in ("vit_frame", "vivit_fe"):
        out = run(["scripts/extract_vit_features.py", "--ckpt", a.ckpt, "--splits", "validation,test", "--variants", "orig,gamma090"])
        extra = ["--feat-hash", re.findall(r"^ckpt_hash=(\w+)", out, re.M)[-1]]
    run(["scripts/predict_vit_streams.py", "--ckpt", a.ckpt, *extra])


if __name__ == "__main__":
    main()
