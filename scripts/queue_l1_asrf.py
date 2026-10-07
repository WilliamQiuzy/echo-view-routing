#!/usr/bin/env python
"""Run the six trainings of the L1 + ASRF round one after another (plan Task 2).

Order: seed 0, 1, 2; within a seed A, then B. Each job is scripts/train_l1_asrf.py with the same config. The status
file lists every job as pending, running, done or failed, with its log, start and end times, the chosen epoch and the
selected and final checkpoint digests. Every job is handed to the trainer, which computes the run identity itself: a
finished run of the same identity with intact checkpoints returns at once, a finished run of another identity is
refused, and an interrupted run resumes after its last finished epoch.

  python scripts/queue_l1_asrf.py --config configs/experiments/l1_asrf_integration_v1.yaml
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from echo_routing.config import load_config  # noqa: E402
from scripts.queue_center_supervision import write  # noqa: E402
from scripts.train_l1_asrf import ARMS, NS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--status", default=f"runs/{NS}/status.json"); ap.add_argument("--logs", default="logs")
    a = ap.parse_args(); cfg = load_config(ROOT / a.config); la = cfg["l1_asrf"]
    status_path = ROOT / a.status; status_path.parent.mkdir(parents=True, exist_ok=True)
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    jobs = [(arm, int(seed)) for seed in la["seeds"] for arm in ARMS]
    for arm, seed in jobs:
        status.setdefault(f"{arm}_s{seed}", {"state": "pending"})
    write(status_path, status)
    for arm, seed in jobs:
        key = f"{arm}_s{seed}"; run_dir = ROOT / "runs" / NS / "train" / key
        log = ROOT / a.logs / f"{NS}_{key}.log"
        cmd = [sys.executable, "-u", "scripts/train_l1_asrf.py", "--config", a.config, "--arm", arm, "--seed", str(seed)]
        status[key] = {"state": "running", "arm": arm, "seed": seed, "log": str(log), "cmd": cmd,
                       "started": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "run_dir": str(run_dir)}
        write(status_path, status)
        with log.open("a") as fh:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
        man_path = run_dir / "run_manifest.json"
        man = json.loads(man_path.read_text()) if rc == 0 and man_path.is_file() else {}
        status[key] = {**status[key], "state": "done" if rc == 0 and man else "failed", "returncode": rc,
                       "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "identity": man.get("identity"),
                       "chosen": man.get("chosen"), "selected_sha256": man.get("selected_sha256"), "final_sha256": man.get("final_sha256")}
        if status[key]["state"] == "failed":
            status[key]["log_tail"] = log.read_text()[-2000:]
        write(status_path, status)
        print(f"{key}: {status[key]['state']} (chosen {status[key].get('chosen')})", flush=True)
        if status[key]["state"] == "failed":
            raise SystemExit(f"{key} failed; fix the cause and rerun the queue (finished jobs return at once, the failed one resumes)")


if __name__ == "__main__":
    main()
