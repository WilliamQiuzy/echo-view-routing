#!/usr/bin/env python
"""Run the nine trainings of the target-anchored readout round one after another (plan Task 2, last item).

Order: seed 0 Continued CLS, Global patches, Target patches, then seeds 1 and 2. Each job is
scripts/train_target_patch_readout.py --readout R --seed N. The queue starts only when the passing check record
(runs/target_patch_readout_v1/checks/check.json) carries the same nine identities as the current code. Before every job it
projects this round's GPU hours from its own check ledger, the finished runs and the remaining runs at the larger of the
checked timing and the slowest finished run, plus the evaluation allowance; above the 12-hour cap it stops and records the
state. A job is skipped only when its run manifest holds the same identity and an intact final checkpoint; an interrupted
job resumes inside the trainer, which records the resume.

  python scripts/queue_target_patch_readout.py --config configs/experiments/target_patch_readout_v1.yaml
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
import scripts.train_raw_video_context as rvc  # noqa: E402
import scripts.train_target_patch_readout as tr  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.queue_raw_video_context import projection, write  # noqa: E402


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--logs", default="logs")
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); ctx = tr.Ctx(cfg, a.config); b = ctx.tp["budget"]
    check = json.loads((ctx.out / "checks" / "check.json").read_text())
    jobs = [(k.rsplit("_s", 1)[0], int(k.rsplit("_s", 1)[1])) for k in ctx.tp["queue"]]
    idents = {tr.run_key(r, s): tr.identity(ctx, r, s, ctx.manifest(s)[1], ctx.tp["source"]["weights"][s]["sha256"]) for r, s in jobs}
    if not check.get("ok") or check.get("identities") != idents:
        raise SystemExit("no passing check record for the current code and inputs; rerun --check")
    status_path = ctx.out / "queue_status.json"; status = json.loads(status_path.read_text()) if status_path.exists() else {}
    foreign = [k for k, v in status.get("jobs", {}).items() if v.get("identity") not in (None, idents.get(k))]
    if foreign:
        raise SystemExit(f"the status file belongs to other identities ({foreign})")
    ledger = [json.loads(x) for x in (ctx.out / "checks" / "ledger.jsonl").read_text().splitlines() if x.strip()]
    status.setdefault("jobs", {}); status.setdefault("failed_attempt_hours", 0.0)
    status.update({"check_hours": float(sum(x["hours"] for x in ledger)), "check_attempts": len(ledger), "cap_hours": float(b["gpu_hours_max"])})
    for k in idents:
        status["jobs"].setdefault(k, {"state": "pending", "identity": idents[k]})
    write(status_path, status)
    for r, s in jobs:
        key = tr.run_key(r, s); run_dir = ctx.out / "train" / key; man = run_dir / "run_manifest.json"
        if rvc.finished(man, idents[key]):
            status["jobs"][key]["state"] = "done"; write(status_path, status); print(f"{key}: done, skipped", flush=True); continue
        done_h = {k: v["hours"] for k, v in status["jobs"].items() if v.get("state") == "done" and "hours" in v}
        per_run = max([float(check["budget"]["hours_per_run_max"]), *done_h.values()])
        proj = projection(list(idents), done_h, status["check_hours"] + status["failed_attempt_hours"], per_run, float(b["evaluation_allowance_hours"]))
        status["projection"] = {"at": now(), "before": key, "projected_total_hours": proj, "per_run_hours": per_run}
        if proj > float(b["gpu_hours_max"]):
            status["stopped"] = {"at": now(), "reason": f"projected {proj:.2f} GPU hours exceed the cap {b['gpu_hours_max']}", "before": key}
            write(status_path, status); raise SystemExit(status["stopped"]["reason"])
        log = ROOT / a.logs / f"target_patch_readout_v1_{key}.log"
        cmd = [sys.executable, "-u", "scripts/train_target_patch_readout.py", "--config", a.config, "--readout", r, "--seed", str(s)]
        status["jobs"][key] = {"state": "running", "identity": idents[key], "readout": r, "seed": s, "log": str(log), "cmd": cmd, "started": now()}
        write(status_path, status); t0 = dt.datetime.now(dt.timezone.utc)
        with log.open("a") as fh:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
        hours = (dt.datetime.now(dt.timezone.utc) - t0).total_seconds() / 3600; ok = rc == 0 and rvc.finished(man, idents[key])
        status["jobs"][key] = {**status["jobs"][key], "state": "done" if ok else "failed", "returncode": rc, "finished": now(), "hours": hours}
        if ok:
            m = json.loads(man.read_text()); status["jobs"][key].update({"updates": m["updates"], "final_sha256": m["final_sha256"], "interrupted": m["interrupted"]})
        else:
            status["failed_attempt_hours"] += hours; status["jobs"][key]["log_tail"] = log.read_text()[-2000:]
        write(status_path, status); print(f"{key}: {status['jobs'][key]['state']} ({hours:.2f} h)", flush=True)
        if not ok:
            raise SystemExit(f"{key} failed; fix the cause and rerun the queue (finished jobs are skipped, this one resumes)")
    status["finished"] = now(); write(status_path, status); print("queue finished", flush=True)


if __name__ == "__main__":
    main()
