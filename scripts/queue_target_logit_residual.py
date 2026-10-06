#!/usr/bin/env python
"""Run the six trainings of the target-logit residual round one after another (plan Task 4).

Order: seed 0, 1, 2; within a seed Z then R. Each job is scripts/train_target_logit_residual.py with the same config.
The status file lists every job as pending, running, done or failed, with its run identity, start and end times,
completed updates and final checkpoint digest. A job is skipped only when the status says done with the same identity
and its run manifest exists with a final checkpoint of the recorded digest; a status file of another identity is
refused. An interrupted job resumes from its last epoch inside the trainer.

  python scripts/queue_target_logit_residual.py --config configs/experiments/target_logit_residual_v1.yaml \
      --status runs/target_logit_residual_v1/status.json
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
from scripts.queue_center_supervision import finished, write  # noqa: E402
from scripts.train_target_logit_residual import NS, check_source, identity, prepare_inits  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--status", required=True); ap.add_argument("--logs", default="logs")
    a = ap.parse_args(); cfg = load_config(ROOT / a.config); rt = cfg["target_logit_residual"]; paths = load_paths()
    source = check_source(rt, paths.repo_root); inits = prepare_inits(cfg, paths)
    index = json.loads((paths.repo_root / rt["source"]["manifest_index"]["path"]).read_text())
    jobs = [(arm, int(seed)) for seed in rt["seeds"] for arm in ("Z", "R")]
    idents = {f"{arm}_s{seed}": identity(cfg, arm, seed, index["seeds"][str(seed)]["files"]["SC"]["sha256"], inits[str(seed)]["sha256"], source, paths.repo_root)
              for arm, seed in jobs}
    status_path = Path(a.status); status_path.parent.mkdir(parents=True, exist_ok=True)
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    foreign = [k for k, v in status.items() if v.get("identity") not in (None, idents.get(k))]
    if foreign:
        raise SystemExit(f"status file belongs to another run (jobs {foreign}); choose a new --status path")
    for k in idents:
        status.setdefault(k, {"state": "pending", "identity": idents[k]})
    write(status_path, status)
    for arm, seed in jobs:
        key = f"{arm}_s{seed}"; run_dir = paths.repo_root / "runs" / NS / "train" / key
        if finished(status[key], run_dir / "run_manifest.json"):
            print(f"{key}: done, skipped", flush=True); continue
        log = ROOT / a.logs / f"{NS}_{key}.log"
        cmd = [sys.executable, "-u", "scripts/train_target_logit_residual.py", "--config", a.config, "--arm", arm, "--seed", str(seed)]
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
