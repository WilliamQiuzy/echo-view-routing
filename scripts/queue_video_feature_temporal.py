#!/usr/bin/env python
"""Run the nine trainings of the video feature temporal round one after another, then freeze them (plan Task 2).

Order: seed 0 Linear, MS-TCN, ASRF, then seeds 1 and 2; each job is scripts/train_video_feature_temporal.py --head H
--seed S. The queue starts only when the passed check record carries the nine identities of the current code and inputs.
Before every job it projects the round's GPU hours from its own ledger (preparation, checks, finished and failed runs),
the remaining runs at the larger of the checked timing and the slowest finished run of that head, and the 2-hour reserve;
above the 8-hour cap nothing new starts and the state is recorded. A job is skipped only when its manifest carries the same
identity, 18,750 updates and an intact final checkpoint. After the ninth run every manifest is verified and the final
weights are frozen with scripts/freeze_model.py into models_frozen/video_feature_temporal_v1/{head}_s{seed}.

  python scripts/queue_video_feature_temporal.py --config configs/experiments/video_feature_temporal_v1.yaml
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.prepare_video_feature_temporal as pv  # noqa: E402
import scripts.train_video_feature_temporal as tv  # noqa: E402
from scripts._common import load_cfg  # noqa: E402

from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402

NS = "video_feature_temporal_v1"


def write(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(obj, indent=1, default=str)); tmp.replace(path)


def identities(ctx: tv.Ctx) -> dict:
    out = {}
    for s in ctx.seeds:
        _, plan_sha = ctx.plan(s)
        for h in tv.HEADS:
            _, init_sha = ctx.init_state(h, s); commit = ctx.vt["heads"][h].get("commit")
            out[tv.run_key(h, s)] = tv.identity(ctx, h, s, ctx.prep["features"][str(s)]["identity"], plan_sha, init_sha, commit)
    return out


def freeze(ctx: tv.Ctx, idents: dict) -> dict:
    out = {}
    for key, ident in idents.items():
        man_path = ctx.out / "train" / key / "run_manifest.json"; man = json.loads(man_path.read_text()); dest = ctx.root / "models_frozen" / NS / key
        if not tv.finished(man_path, ident):
            raise SystemExit(f"{key}: not a finished run of this identity")
        if not dest.exists():
            h, s = key.rsplit("_s", 1); init = ctx.root / ctx.prep["inits"][key]["path"]; plan = ctx.out / "plans" / f"plan_s{s}.json"
            files = [man["final_checkpoint"], str(man_path), str(init), str(plan), str(ctx.root / ctx.config)]
            subprocess.run([sys.executable, "scripts/freeze_model.py", NS, key, *[x for f in files for x in ("--file", f)], "--meta", f"identity={ident}",
                            "--meta", f"head={h}", "--meta", f"seed={s}", "--meta", f"updates={man['updates']}",
                            "--notes", "Video feature temporal round (2026-10-12): final weights after 18,750 updates on frozen Ordered features."],
                           cwd=ctx.root, check=True, capture_output=True)
        if sha256_file(dest / "final.pt") != man["final_sha256"]:
            raise SystemExit(f"{key}: the frozen copy differs from the final weights")
        out[key] = {"path": str(dest.relative_to(ctx.root)), "final_sha256": man["final_sha256"], "updates": man["updates"], "interrupted": man["interrupted"]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--logs", default="logs")
    a = ap.parse_args(); a.set = []; ctx = tv.Ctx(load_cfg(a), a.config); b = ctx.vt["budget"]
    check = json.loads((ctx.out / "checks" / "checks.json").read_text()); idents = identities(ctx)
    if not check.get("passed") or check.get("identities") != idents:
        raise SystemExit("no passed check record for the current code and inputs; run the check again")
    status_path = ctx.out / "queue_status.json"; status = json.loads(status_path.read_text()) if status_path.exists() else {}
    foreign = [k for k, v in status.get("jobs", {}).items() if v.get("identity") not in (None, idents.get(k))]
    if foreign:
        raise SystemExit(f"the status file belongs to other identities ({foreign})")
    status.setdefault("jobs", {}); status["cap_hours"] = float(b["gpu_hours_max"])
    for k in ctx.vt["training"]["queue"]:
        status["jobs"].setdefault(k, {"state": "pending", "identity": idents[k]})
    write(status_path, status)
    for key in ctx.vt["training"]["queue"]:
        h, s = key.rsplit("_s", 1); s = int(s); man = ctx.out / "train" / key / "run_manifest.json"
        if tv.finished(man, idents[key]):
            status["jobs"][key]["state"] = "done"; write(status_path, status); print(f"{key}: done, skipped", flush=True); continue
        done = {k: v for k, v in status["jobs"].items() if v.get("state") == "done" and "hours" in v}
        per_head = {hd: max([float(check["hours_per_run"][hd]), *[v["hours"] for k, v in done.items() if k.startswith(hd + "_")]]) for hd in tv.HEADS}
        remaining = [k for k in ctx.vt["training"]["queue"] if not tv.finished(ctx.out / "train" / k / "run_manifest.json", idents[k])]
        proj = pv.ledger_hours(ctx) + sum(per_head[k.rsplit("_s", 1)[0]] for k in remaining) + float(b["reserve_hours"])
        status["projection"] = {"at": pv.now(), "before": key, "projected_total_hours": proj, "per_head_hours": per_head}
        if proj > float(b["gpu_hours_max"]):
            status["stopped"] = {"at": pv.now(), "reason": f"projected {proj:.2f} GPU hours exceed the cap {b['gpu_hours_max']}", "before": key}
            write(status_path, status); raise SystemExit(status["stopped"]["reason"])
        log = ROOT / a.logs / f"{NS}_{key}.log"
        cmd = [sys.executable, "-u", "scripts/train_video_feature_temporal.py", "--config", a.config, "--head", h, "--seed", str(s)]
        status["jobs"][key] = {"state": "running", "identity": idents[key], "head": h, "seed": s, "log": str(log), "started": pv.now()}
        write(status_path, status); started, t0 = pv.now(), time.time()
        with log.open("a") as fh:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
        hours = (time.time() - t0) / 3600; ok = rc == 0 and tv.finished(man, idents[key]); pv.ledger(ctx, f"train {key}", started, hours * 3600, ok)
        status["jobs"][key] = {**status["jobs"][key], "state": "done" if ok else "failed", "returncode": rc, "finished": pv.now(), "hours": hours}
        if ok:
            m = json.loads(man.read_text()); status["jobs"][key].update({"updates": m["updates"], "final_sha256": m["final_sha256"], "interrupted": m["interrupted"]})
        else:
            status["jobs"][key]["log_tail"] = log.read_text()[-2000:]
        write(status_path, status); print(f"{key}: {status['jobs'][key]['state']} ({hours:.2f} h)", flush=True)
        if not ok:
            raise SystemExit(f"{key} failed; fix the cause and rerun the queue (finished jobs are skipped, this one resumes)")
    status["frozen"] = freeze(ctx, idents); status["finished"] = pv.now(); write(status_path, status); print("queue finished and frozen", flush=True)


if __name__ == "__main__":
    main()
