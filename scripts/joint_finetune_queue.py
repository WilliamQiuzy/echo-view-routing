#!/usr/bin/env python
"""Run the paired joint fine-tuning of round 2 (Task 5b) one job after another and keep a status file.

For each seed, the candidate and ASRF start from their own E1 runs of that seed (read from the round-2 E1 status
files) and from the same exported visual encoder. A finished job is skipped only when its command matches.
  python scripts/joint_finetune_queue.py --candidate M5 --seeds 0,1,2 --status runs/md_round2/joint_status.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", required=True, choices=("M3p", "M5")); ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--status", required=True); ap.add_argument("--e1-status", default="runs/md_round2/e1_s{seed}_status.json")
    ap.add_argument("--protocol", required=True); ap.add_argument("--feature-manifest", required=True)
    ap.add_argument("--visual-export", required=True); ap.add_argument("--config", default="configs/experiments/model_development_v2.yaml")
    a = ap.parse_args(); status_path = ROOT / a.status; status_path.parent.mkdir(parents=True, exist_ok=True)
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    for seed in [int(s) for s in a.seeds.split(",")]:
        e1 = json.loads((ROOT / a.e1_status.format(seed=seed)).read_text())
        for method in (a.candidate, "M1"):
            job = f"J_{method}_s{seed}"
            cmd = [sys.executable, "-u", "scripts/train_joint_visual_temporal.py", "-c", a.config, "--set", f"seed={seed}",
                   "--protocol", a.protocol, "--feature-manifest", a.feature_manifest, "--head-run", e1[method]["run_dir"],
                   "--visual-export", a.visual_export, "--row", f"J_{method}"]
            if status.get(job, {}).get("state") == "done" and status[job].get("cmd") == cmd:
                continue
            log = ROOT / "logs" / f"md2_joint_{method}_s{seed}.log"
            status[job] = {"state": "running", "cmd": cmd, "log": str(log), "started": dt.datetime.now().isoformat(timespec="seconds")}
            status_path.write_text(json.dumps(status, indent=1))
            with log.open("w") as fh:
                rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
            text = log.read_text(); run_dir = (re.findall(r"run_dir=(\S+)", text) or [None])[-1]
            status[job].update({"state": "done" if rc == 0 else "failed", "returncode": rc, "run_dir": run_dir,
                                "finished": dt.datetime.now().isoformat(timespec="seconds")})
            if rc != 0:
                status[job]["log_tail"] = text[-2000:]
            status_path.write_text(json.dumps(status, indent=1)); print(job, status[job]["state"], run_dir, flush=True)


if __name__ == "__main__":
    main()
