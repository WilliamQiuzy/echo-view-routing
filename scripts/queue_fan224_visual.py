#!/usr/bin/env python
"""Run the twelve trainings of the fan224 visual adaptation round one after another (plan Task 2).

Order: seed 0, 1, 2; within a seed L0, L1, F0, F1. Each job is scripts/train_fan224_visual.py with the same config. The
status file lists every job as pending, running, done or failed, with its run identity, start and end times, completed
updates and final checkpoint digest. A job is skipped only when the status says done with the same identity and its run
manifest exists with a final checkpoint of the recorded digest; a status file of another identity is refused. An
interrupted job resumes from its last saved state inside the trainer.

  python scripts/queue_fan224_visual.py --config configs/experiments/fan224_visual_adaptation_v1.yaml \
      --status runs/fan224_visual_adaptation_v1/status.json
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
from echo_routing.config import load_config, load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from scripts.queue_center_supervision import finished, write  # noqa: E402
from scripts.train_fan224_visual import (ARMS, NS, build, check_inputs, identity, init_state, save_or_check_init,  # noqa: E402
                                         save_or_check_plan, train_table)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--status", default=f"runs/{NS}/status.json"); ap.add_argument("--logs", default="logs")
    a = ap.parse_args(); cfg = load_config(ROOT / a.config); fv = cfg["fan224"]; paths = load_paths(); root = paths.repo_root
    import torch
    inputs = check_inputs(fv, root); table = train_table(fv, paths)
    jobs = [(arm, int(seed)) for seed in fv["seeds"] for arm in ARMS]; idents = {}
    for seed in sorted({s for _, s in jobs}):
        plan_path, _ = save_or_check_plan(fv, table, seed, root); plan_sha = sha256_file(plan_path)
        enc, head = build(fv, paths, "L1", torch.device("cpu")); _, init_sha = save_or_check_init(init_state(enc, head), seed, root); del enc, head
        for arm in ARMS:
            idents[f"{arm}_s{seed}"] = identity(cfg, arm, seed, inputs, plan_sha, init_sha, root)
    status_path = root / a.status; status_path.parent.mkdir(parents=True, exist_ok=True)
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    foreign = [k for k, v in status.items() if v.get("identity") not in (None, idents.get(k))]
    if foreign:
        raise SystemExit(f"status file belongs to another run (jobs {foreign}); choose a new --status path")
    for k in idents:
        status.setdefault(k, {"state": "pending", "identity": idents[k]})
    write(status_path, status)
    for arm, seed in jobs:
        key = f"{arm}_s{seed}"; run_dir = root / "runs" / NS / "train" / key
        if finished(status[key], run_dir / "run_manifest.json"):
            print(f"{key}: done, skipped", flush=True); continue
        log = ROOT / a.logs / f"{NS}_{key}.log"
        cmd = [sys.executable, "-u", "scripts/train_fan224_visual.py", "--config", a.config, "--arm", arm, "--seed", str(seed)]
        status[key] = {"state": "running", "identity": idents[key], "arm": arm, "seed": seed, "log": str(log), "cmd": cmd,
                       "started": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "run_dir": str(run_dir)}
        write(status_path, status)
        with log.open("a") as fh:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
        man_path = run_dir / "run_manifest.json"
        man = json.loads(man_path.read_text()) if rc == 0 and man_path.is_file() else {}
        status[key] = {**status[key], "state": "done" if rc == 0 and man.get("identity") == idents[key] else "failed", "returncode": rc,
                       "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "updates": man.get("updates"),
                       "final_checkpoint": man.get("final_checkpoint"), "final_sha256": man.get("final_sha256")}
        if status[key]["state"] == "failed":
            status[key]["log_tail"] = log.read_text()[-2000:]
        write(status_path, status)
        print(f"{key}: {status[key]['state']} ({status[key].get('updates')} updates)", flush=True)
        if status[key]["state"] == "failed":
            raise SystemExit(f"{key} failed; fix the cause and rerun the queue (finished jobs are skipped, the failed one resumes)")


if __name__ == "__main__":
    main()
