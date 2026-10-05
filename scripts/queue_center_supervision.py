#!/usr/bin/env python
"""Run the nine trainings of the target-time supervision round one after another (plan Task 3).

Order: seed 0, 1, 2; within a seed N, S, C. Each job is scripts/train_center_supervision.py with the same config and
manifest root. The status file lists every job as pending, running, done or failed, with its run identity, start and
end times, completed updates and final checkpoint digest. A job is skipped only when the status says done with the
same identity and its run manifest exists with a final checkpoint of the recorded digest; a status file of another
identity is refused. An interrupted job resumes from its last epoch inside the trainer.

  python scripts/queue_center_supervision.py --config configs/experiments/ev9v_center_supervision_v1.yaml \
      --manifest-root runs/center_supervision_v1/manifests --status runs/center_supervision_v1/status.json
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
from scripts.train_center_supervision import identity  # noqa: E402


def write(path: Path, status: dict) -> None:
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(status, indent=1)); tmp.replace(path)


def finished(job: dict, run_manifest: Path) -> bool:
    if job.get("state") != "done" or not run_manifest.is_file():
        return False
    man = json.loads(run_manifest.read_text()); final = Path(man["final_checkpoint"])
    return man["identity"] == job["identity"] and final.is_file() and sha256_file(final) == job.get("final_sha256") == man["final_sha256"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--manifest-root", required=True); ap.add_argument("--status", required=True)
    ap.add_argument("--logs", default="logs")
    a = ap.parse_args(); cfg = load_config(ROOT / a.config); cs = cfg["center_supervision"]; paths = load_paths()
    index = json.loads((Path(a.manifest_root) / "index.json").read_text())
    jobs = [(arm, int(seed)) for seed in cs["seeds"] for arm in cs["arms"]]
    idents = {f"{arm}_s{seed}": identity(cfg, arm, seed, index["seeds"][str(seed)]["files"]["N" if arm == "N" else "SC"]["sha256"],
                                         index["seeds"][str(seed)]["files"]["init"]["sha256"]) for arm, seed in jobs}
    status_path = Path(a.status); status_path.parent.mkdir(parents=True, exist_ok=True)
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    foreign = [k for k, v in status.items() if v.get("identity") not in (None, idents.get(k))]
    if foreign:
        raise SystemExit(f"status file belongs to another run (jobs {foreign}); choose a new --status path")
    for k in idents:
        status.setdefault(k, {"state": "pending", "identity": idents[k]})
    write(status_path, status)
    for arm, seed in jobs:
        key = f"{arm}_s{seed}"; run_dir = paths.repo_root / "runs" / "center_supervision_v1" / "train" / key
        if finished(status[key], run_dir / "run_manifest.json"):
            print(f"{key}: done, skipped", flush=True); continue
        log = ROOT / a.logs / f"center_supervision_v1_{key}.log"
        cmd = [sys.executable, "-u", "scripts/train_center_supervision.py", "--config", a.config, "--arm", arm, "--seed", str(seed),
               "--manifest-root", a.manifest_root]
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
