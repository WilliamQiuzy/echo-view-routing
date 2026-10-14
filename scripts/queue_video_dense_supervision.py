#!/usr/bin/env python
"""Run the nine trainings of the video dense supervision round one after another, then freeze them (plan Task 3).

Order: seed 0 CLS continuation, aligned and shuffled token supervision, then seeds 1 and 2. Each job is
scripts/train_video_dense_supervision.py --arm A --seed N. The queue starts only when the passing check record
(runs/video_dense_supervision_v1/checks/check.json) carries the same nine identities as the current code and inputs.
Before every job it projects the round's GPU hours from its ledger (every passed or failed job), the remaining runs at the
larger of the checked timing and the slowest finished run, and the evaluation allowance; above the 8-hour cap nothing new
starts and the state is recorded. Every attempt adds a ledger line. A job is skipped only when its run manifest holds the
same identity and intact final weights; an interrupted job resumes inside the trainer, which records the resume. After the
ninth run the final weights are frozen with scripts/freeze_model.py into models_frozen/video_dense_supervision_v1/{key},
and the three source Ordered weights must still match their pins.

  python scripts/queue_video_dense_supervision.py --config configs/experiments/video_dense_supervision_v1.yaml
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
import scripts.prepare_video_dense_supervision as pd  # noqa: E402
import scripts.train_video_dense_supervision as tv  # noqa: E402
from scripts._common import load_cfg  # noqa: E402

from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402

NS = "video_dense_supervision_v1"


def projected(spent: float, remaining: int, per_run: float, allowance: float) -> float:
    return float(spent) + int(remaining) * float(per_run) + float(allowance)


def write(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(obj, indent=1, default=str)); tmp.replace(path)


def identities(ctx) -> dict:
    return {tv.run_key(a, s): tv.identity(ctx, a, s, ctx.manifest(s)[1], ctx.aux(s)[1], ctx.vd["source"]["weights"][s]["sha256"])
            for a, s in ((k.rsplit("_s", 1)[0], int(k.rsplit("_s", 1)[1])) for k in ctx.vd["queue"])}


def freeze(ctx, idents: dict) -> dict:
    out = {}
    for key, ident in idents.items():
        man_path = ctx.out / "train" / key / "run_manifest.json"; man = json.loads(man_path.read_text()); dest = ROOT / "models_frozen" / NS / key
        if man["identity"] != ident or int(man["updates"]) != ctx.total:
            raise SystemExit(f"{key}: the run manifest does not match its identity or update count")
        if not dest.exists():
            files = [man["final_checkpoint"], str(man_path), str(ROOT / ctx.config), str(ctx.out / "manifests" / f"AUX_s{man['seed']}.npz")]
            subprocess.run([sys.executable, "scripts/freeze_model.py", NS, key, *[x for f in files for x in ("--file", f)], "--meta", f"identity={ident}",
                            "--meta", f"arm={man['arm']}", "--meta", f"seed={man['seed']}", "--meta", f"lambda={man['lambda']}",
                            "--meta", f"source_weights_sha256={man['source_weights_sha256']}",
                            "--notes", "Video dense supervision round (A02, 2026-10-13); final weights after 3,750 updates from the seed's V01 Ordered model."],
                           cwd=ROOT, check=True, capture_output=True)
        if sha256_file(dest / "final.pt") != man["final_sha256"]:
            raise SystemExit(f"{key}: the frozen copy differs from the final weights")
        out[key] = {"path": str(dest.relative_to(ROOT)), "final_sha256": man["final_sha256"], "updates": man["updates"], "interrupted": man["interrupted"]}
    for s, pin in ctx.vd["source"]["weights"].items():
        pd.pinned(ROOT, pin)                                   # the V01 Ordered sources are unchanged
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--logs", default="logs")
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); ctx = tv.Ctx(cfg, a.config); b = ctx.vd["budget"]
    check = json.loads((ctx.out / "checks" / "check.json").read_text()); idents = identities(ctx)
    if not check.get("ok") or check.get("identities") != idents:
        raise SystemExit("no passing check record for the current code and inputs; rerun --check")
    status_path = ctx.out / "queue_status.json"; status = json.loads(status_path.read_text()) if status_path.exists() else {}
    foreign = [k for k, v in status.get("jobs", {}).items() if v.get("identity") not in (None, idents.get(k))]
    if foreign:
        raise SystemExit(f"the status file belongs to other identities ({foreign})")
    status.setdefault("jobs", {}); status["cap_hours"] = float(b["gpu_hours_max"])
    for k in idents:
        status["jobs"].setdefault(k, {"state": "pending", "identity": idents[k]})
    write(status_path, status)
    for key in ctx.vd["queue"]:
        arm, seed = key.rsplit("_s", 1)[0], int(key.rsplit("_s", 1)[1]); man = ctx.out / "train" / key / "run_manifest.json"
        if tv.finished(man, idents[key]):
            status["jobs"][key]["state"] = "done"; write(status_path, status); print(f"{key}: done, skipped", flush=True); continue
        done_h = [v["hours"] for v in status["jobs"].values() if v.get("state") == "done" and "hours" in v]
        per_run = max([float(check["budget"]["hours_per_run_max"]), *done_h]); remaining = sum(1 for k in ctx.vd["queue"] if not tv.finished(ctx.out / "train" / k / "run_manifest.json", idents[k]))
        proj = projected(tv.ledger_hours(ctx.ledger), remaining, per_run, float(b["evaluation_allowance_hours"]))
        status["projection"] = {"at": tv.now(), "before": key, "projected_total_hours": proj, "per_run_hours": per_run, "remaining_runs": remaining}
        if proj > float(b["gpu_hours_max"]):
            status["stopped"] = {"at": tv.now(), "reason": f"projected {proj:.2f} GPU hours exceed the cap {b['gpu_hours_max']}", "before": key}
            write(status_path, status); raise SystemExit(status["stopped"]["reason"])
        log = ROOT / a.logs / f"{NS}_{key}.log"
        cmd = [sys.executable, "-u", "scripts/train_video_dense_supervision.py", "--config", a.config, "--arm", arm, "--seed", str(seed)]
        status["jobs"][key] = {"state": "running", "identity": idents[key], "arm": arm, "seed": seed, "log": str(log), "started": tv.now()}
        write(status_path, status); t0 = dt.datetime.now(dt.timezone.utc); started = tv.now()
        with log.open("a") as fh:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
        hours = (dt.datetime.now(dt.timezone.utc) - t0).total_seconds() / 3600; ok = rc == 0 and tv.finished(man, idents[key])
        tv.ledger_append(ctx.ledger, f"train {key}", started, hours, ok, "" if ok else f"returncode {rc}")
        status["jobs"][key] = {**status["jobs"][key], "state": "done" if ok else "failed", "returncode": rc, "finished": tv.now(), "hours": hours}
        if ok:
            mj = json.loads(man.read_text()); status["jobs"][key].update({"updates": mj["updates"], "final_sha256": mj["final_sha256"], "interrupted": mj["interrupted"]})
        else:
            status["jobs"][key]["log_tail"] = log.read_text()[-2000:]
        write(status_path, status); print(f"{key}: {status['jobs'][key]['state']} ({hours:.2f} h)", flush=True)
        if not ok:
            raise SystemExit(f"{key} failed; fix the cause and rerun the queue (finished jobs are skipped, this one resumes)")
    status["frozen"] = freeze(ctx, idents); status["finished"] = tv.now(); write(status_path, status); print("queue finished and frozen", flush=True)


if __name__ == "__main__":
    main()
