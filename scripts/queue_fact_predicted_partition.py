#!/usr/bin/env python
"""The six formal runs of the predicted-partition round in the fixed order, one GPU job at a time
(configs/experiments/fact_predicted_partition_v1.yaml). The cap covers the GPU hours of the whole round: the three
heads' training, the smoke checks, the formal runs and an allowance for the evaluation. Before each run the queue
projects that total. A remaining run counts the mean of its group's finished runs, or else the group's smoke
estimate. A run starts only while the projection stays within the cap; otherwise the queue stops with every finished
run and resume state kept, and reports. A failed run stops the queue too.

  python scripts/queue_fact_predicted_partition.py --config configs/experiments/fact_predicted_partition_v1.yaml
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_fact_predicted_partition import GROUPS, PCtx, frozen_head, load_round, smoke_stamp  # noqa: E402


def hours_of(manifest: Path) -> float | None:
    return float(json.loads(manifest.read_text())["minutes"]) / 60 if manifest.is_file() else None


def projection(order: list[str], done: dict[str, float], smoke: dict[str, float], spent_before: float, allowance: float) -> float:
    """GPU hours before the queue, finished runs, each remaining run's group mean or smoke estimate, and the evaluation."""
    by_group: dict[str, list[float]] = {}
    for key, h in done.items():
        by_group.setdefault(key.split("_s")[0], []).append(h)
    rest = [k for k in order if k not in done]
    est = {g: (sum(v) / len(v) if (v := by_group.get(g)) else smoke[g]) for g in smoke}
    return spent_before + sum(done.values()) + sum(est[k.split("_s")[0]] for k in rest) + allowance


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    a = ap.parse_args(); cfg, fp, scfg, paths = load_round(a.config); ctx = PCtx(cfg, scfg, fp, paths)
    cap, allowance, order = float(fp["budget"]["gpu_hours_max"]), float(fp["budget"]["evaluation_allowance_hours"]), list(fp["queue"])
    heads = {s: json.loads((ctx.out / "heads" / f"head_s{s}.json").read_text()) for s in ctx.seeds}
    if not all(h["launch_check_passed"] for h in heads.values()) or not all((ctx.frozen_head_dir(s) / "MANIFEST.json").is_file() for s in ctx.seeds):
        raise SystemExit("a head is not trained, did not pass its launch check or is not frozen")
    seed = int(fp["smoke"]["seed"]); _, _, head_sha = frozen_head(ctx, seed, "cpu"); smoke, smoke_hours = {}, 0.0
    for g in GROUPS:
        rep = json.loads((ctx.out / "smoke" / f"{g}_s{seed}.json").read_text())
        if rep.get("stamp") != smoke_stamp(ctx, g, seed, head_sha):
            raise SystemExit(f"{g}: the smoke report was produced from another configuration, code, plan, initial state or head; rerun the smoke")
        if not rep["ok"]:
            raise SystemExit(f"{g}: the smoke checks did not pass")
        smoke[g] = float(rep["timing"]["estimated_hours_per_run"]); smoke_hours += float(rep["gpu_seconds"]) / 3600
    spent_before = sum(float(h["gpu_seconds"]) for h in heads.values()) / 3600 + smoke_hours
    status = ctx.out / "queue_status.json"
    for key in order:
        group, s = key.split("_s")
        done = {k: h for k in order if (h := hours_of(ctx.out / "train" / k / "run_manifest.json")) is not None}
        if key in done:
            print(json.dumps({"run": key, "state": "finished earlier", "hours": round(done[key], 2)}), flush=True); continue
        proj = projection(order, done, smoke, spent_before, allowance)
        row = {"run": key, "spent_hours": round(spent_before + sum(done.values()), 3), "projected_total_hours": round(proj, 3), "cap_hours": cap}
        if proj > cap:
            status.write_text(json.dumps({**row, "state": "stopped: the projected GPU hours exceed the cap"}, indent=1))
            print(json.dumps({**row, "state": "stopped over budget"}), flush=True)
            raise SystemExit(f"{key}: the projected GPU hours {proj:.2f} exceed the cap of {cap} h; finished runs and resume states are kept")
        print(json.dumps({**row, "state": "starting", "time": time.strftime("%Y-%m-%dT%H:%M:%S")}), flush=True)
        status.write_text(json.dumps({**row, "state": "running"}, indent=1))
        rc = subprocess.run([sys.executable, "-u", "scripts/train_fact_predicted_partition.py", "--config", a.config, "--group", group, "--seed", s], cwd=ctx.root).returncode
        if rc != 0:
            status.write_text(json.dumps({**row, "state": f"failed with exit code {rc}"}, indent=1)); raise SystemExit(f"{key} failed with exit code {rc}")
    done = {k: hours_of(ctx.out / "train" / k / "run_manifest.json") for k in order}
    status.write_text(json.dumps({"state": "all six runs finished", "hours": {k: round(v, 3) for k, v in done.items()},
                                  "formal_hours": round(sum(done.values()), 3), "hours_before_the_queue": round(spent_before, 3)}, indent=1))
    print(json.dumps({"state": "all six runs finished", "formal_hours": round(sum(done.values()), 3)}), flush=True)


if __name__ == "__main__":
    main()
