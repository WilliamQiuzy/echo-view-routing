#!/usr/bin/env python
"""Run round-1 jobs one after another on one GPU and keep a status file (2026-10-02 plan, Tasks 4 and 6).

Every job is a subprocess of an entry script with the same config, seed, protocol and feature manifest. The status
file lists each job as pending, running, done (with its run directory) or failed (with the end of its log).

  python scripts/model_development_queue.py --jobs M4,M0,M1,M2,M3 --feature-manifest F --protocol P --status S
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
CONFIG = "configs/experiments/boundary_context_v1.yaml"
JOBS = {
    "M0": ["scripts/run_mstcn_adapted.py"],
    "M1": ["scripts/run_asrf_official.py"],
    "M2": ["scripts/train_boundary_context.py", "--variant", "plain"],
    "M3": ["scripts/train_boundary_context.py", "--variant", "aux"],
    "M4": ["scripts/train_boundary_context.py", "--variant", "full"],
    "M4_fixedgate": ["scripts/train_boundary_context.py", "--variant", "full", "--fixed-gate", "0.5"],
    "M1_aug": ["scripts/run_asrf_official.py", "--augment"],
    "M4_aug": ["scripts/train_boundary_context.py", "--variant", "full", "--augment"],
    "M5": ["scripts/train_boundary_context.py", "--variant", "full", "--segment-pool"],
    "M3p": ["scripts/train_boundary_context.py", "--variant", "aux", "--segment-pool"],
    "simple": ["scripts/calibrate_simple_decoders.py"],
}


def write(path: Path, status: dict) -> None:
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(status, indent=1)); tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", required=True); ap.add_argument("--feature-manifest", required=True)
    ap.add_argument("--protocol", required=True); ap.add_argument("--status", required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--logs", default="logs")
    args = ap.parse_args()
    names = args.jobs.split(","); unknown = [n for n in names if n not in JOBS]
    if unknown:
        raise SystemExit(f"unknown jobs {unknown}; known: {sorted(JOBS)}")
    status_path = Path(args.status); status_path.parent.mkdir(parents=True, exist_ok=True)
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    for n in names:
        status.setdefault(n, {"state": "pending"})
    write(status_path, status)
    for n in names:
        if status[n].get("state") == "done":
            continue
        cmd = [sys.executable, "-u", *JOBS[n], "-c", CONFIG, "--set", f"seed={args.seed}",
               "--feature-manifest", args.feature_manifest, "--protocol", args.protocol]
        if args.lr is not None and n != "simple":
            cmd += ["--lr", str(args.lr)]
        log = ROOT / args.logs / f"md_{n}_s{args.seed}{'' if args.lr is None else f'_lr{args.lr:g}'}.log"
        status[n] = {"state": "running", "started": dt.datetime.now().isoformat(timespec="seconds"), "log": str(log), "cmd": cmd}
        write(status_path, status)
        with log.open("w") as fh:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
        text = log.read_text()
        run_dir = (re.findall(r"run_dir=(\S+)", text) or re.findall(r'"run_dir": "([^"]+)"', text) or [None])[0]
        status[n] = {**status[n], "state": "done" if rc == 0 else "failed", "returncode": rc, "run_dir": run_dir,
                     "finished": dt.datetime.now().isoformat(timespec="seconds")}
        if rc != 0:
            status[n]["log_tail"] = text[-2000:]
        write(status_path, status)
        print(f"{n}: {status[n]['state']} {run_dir}", flush=True)


if __name__ == "__main__":
    main()
