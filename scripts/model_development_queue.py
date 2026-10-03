#!/usr/bin/env python
"""Run model-development jobs one after another on one GPU and keep a status file (round 1 Tasks 4/6, round 2 Task 1).

Every job is a subprocess of an entry script with the same config, seed, protocol and feature manifest. The status
file lists each job as pending, running, done (with its run directory) or failed (with the end of its log).

Each job carries a run identity: a digest of the experiment id, the resolved config, the protocol and feature-manifest
digests, the job's command arguments, the seed and the learning rate. A finished job is skipped only when its stored
identity matches; a status file holding any job with another identity is refused, so a new experiment can never
reuse or overwrite an earlier experiment's status. Logs are named by experiment, job, seed and learning rate.

  python scripts/model_development_queue.py --config configs/experiments/model_development_v2.yaml \
      --jobs M1,M3p,M5,simple --feature-manifest F --protocol P --seed 0 --status runs/md_round2/e1_s0_status.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from echo_routing.config import load_config  # noqa: E402

CONFIG = "configs/experiments/boundary_context_v1.yaml"   # round-1 default, kept for compatibility
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
    "BO": ["scripts/train_boundary_only.py"],
}


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def run_identity(experiment: str, config: dict, protocol_sha: str, features_sha: str, job: str, seed: int,
                 lr: float | None) -> str:
    payload = {"experiment": experiment, "config": config, "protocol_sha256": protocol_sha, "feature_manifest_sha256": features_sha,
               "job": job, "job_args": JOBS[job], "seed": int(seed), "lr": lr}
    return sha256_bytes(json.dumps(payload, sort_keys=True, default=str).encode())


def plan_jobs(status: dict, names: list[str], identities: dict[str, str]) -> list[str]:
    """Jobs to run. Refuses a status file that holds a job of another identity; skips a done job of the same one."""
    foreign = [n for n, v in status.items() if isinstance(v, dict) and "identity" in v and v["identity"] != identities.get(n, v["identity"])]
    legacy = [n for n, v in status.items() if isinstance(v, dict) and v.get("state") == "done" and "identity" not in v]
    if foreign or legacy:
        raise SystemExit(f"status file belongs to another run (jobs {sorted(foreign + legacy)}); choose a new --status path")
    return [n for n in names if not (status.get(n, {}).get("state") == "done" and status[n].get("identity") == identities[n])]


def write(path: Path, status: dict) -> None:
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(status, indent=1)); tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", required=True); ap.add_argument("--feature-manifest", required=True)
    ap.add_argument("--protocol", required=True); ap.add_argument("--status", required=True)
    ap.add_argument("--config", default=CONFIG); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=None); ap.add_argument("--logs", default="logs")
    args = ap.parse_args()
    names = args.jobs.split(","); unknown = [n for n in names if n not in JOBS]
    if unknown:
        raise SystemExit(f"unknown jobs {unknown}; known: {sorted(JOBS)}")
    cfg = load_config(ROOT / args.config); experiment = str(cfg.get("experiment", "round1"))
    protocol_sha = sha256_bytes(Path(args.protocol).read_bytes()); features_sha = sha256_bytes(Path(args.feature_manifest).read_bytes())
    identities = {n: run_identity(experiment, cfg, protocol_sha, features_sha, n, args.seed, args.lr) for n in names}
    status_path = Path(args.status); status_path.parent.mkdir(parents=True, exist_ok=True)
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    todo = plan_jobs(status, names, identities)
    for n in names:
        status.setdefault(n, {"state": "pending", "identity": identities[n]})
    write(status_path, status)
    for n in todo:
        cmd = [sys.executable, "-u", *JOBS[n], "-c", args.config, "--set", f"seed={args.seed}",
               "--feature-manifest", args.feature_manifest, "--protocol", args.protocol]
        if args.lr is not None and n != "simple":
            cmd += ["--lr", str(args.lr)]
        # the feature digest keeps logs of the same job on different inputs apart (two queues once shared one log)
        log = ROOT / args.logs / f"md_{experiment}_{n}_s{args.seed}_f{features_sha[:8]}{'' if args.lr is None else f'_lr{args.lr:g}'}.log"
        status[n] = {"state": "running", "identity": identities[n], "experiment": experiment, "config": args.config,
                     "protocol_sha256": protocol_sha, "feature_manifest_sha256": features_sha, "seed": args.seed,
                     "started": dt.datetime.now().isoformat(timespec="seconds"), "log": str(log), "cmd": cmd}
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
