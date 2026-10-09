#!/usr/bin/env python
"""The six formal runs of the paired FACT-ASRF round in the fixed order, one GPU job at a time
(configs/experiments/fact_asrf_paired_v1.yaml). Before each run the queue projects the training hours: the hours of the
finished runs plus, for every remaining run, the mean of its arm's finished runs or else the arm's smoke estimate. A run
starts only when that projection stays within the cap. Otherwise the queue stops with every finished run and any resume
state kept, and reports. A failed run stops the queue as well.

  python scripts/queue_fact_asrf.py --config configs/experiments/fact_asrf_paired_v1.yaml
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg  # noqa: E402

from echo_routing.config import load_paths  # noqa: E402


def hours_of(manifest: Path) -> float | None:
    return float(json.loads(manifest.read_text())["minutes"]) / 60 if manifest.is_file() else None


def projection(order: list[str], done: dict[str, float], smoke: dict[str, float]) -> float:
    """Finished hours plus, for each remaining run, its arm's mean finished hours or else its smoke estimate."""
    by_arm: dict[str, list[float]] = {}
    for key, h in done.items():
        by_arm.setdefault(key.split("_")[0], []).append(h)
    rest = [k for k in order if k not in done]
    return sum(done.values()) + sum((sum(by_arm[k.split("_")[0]]) / len(by_arm[k.split("_")[0]]) if k.split("_")[0] in by_arm else smoke[k.split("_")[0]]) for k in rest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); fa = cfg["fact_asrf_paired"]; root = load_paths().repo_root
    server = root / fa["outputs"]["server_dir"]; cap = float(fa["budget"]["formal_training_gpu_hours_max"]); order = list(fa["queue"])
    smoke = {}
    for arm in ("FACT", "ASRF"):
        rep = json.loads((server / "smoke" / f"{arm}_s{int(fa['smoke']['seed'])}.json").read_text())
        if not rep["ok"]:
            raise SystemExit(f"{arm}: the smoke checks did not pass")
        smoke[arm] = float(rep["timing"]["estimated_hours_per_run"])
    status = server / "queue_status.json"
    for key in order:
        arm, seed = key.split("_s")
        done = {k: h for k in order if (h := hours_of(server / "train" / k / "run_manifest.json")) is not None}
        if key in done:
            print(json.dumps({"run": key, "state": "finished earlier", "hours": round(done[key], 2)}), flush=True); continue
        proj = projection(order, done, smoke)
        row = {"run": key, "spent_hours": round(sum(done.values()), 3), "projected_total_hours": round(proj, 3), "cap_hours": cap}
        if proj > cap:
            status.write_text(json.dumps({**row, "state": "stopped: the projected training hours exceed the cap"}, indent=1))
            print(json.dumps({**row, "state": "stopped over budget"}), flush=True); raise SystemExit(2)
        print(json.dumps({**row, "state": "starting", "time": time.strftime("%Y-%m-%dT%H:%M:%S")}), flush=True)
        status.write_text(json.dumps({**row, "state": "running"}, indent=1))
        rc = subprocess.run([sys.executable, "-u", "scripts/train_fact_asrf.py", "--config", a.config, "--arm", arm, "--seed", seed], cwd=root).returncode
        if rc != 0:
            status.write_text(json.dumps({**row, "state": f"failed with exit code {rc}"}, indent=1)); raise SystemExit(f"{key} failed with exit code {rc}")
    done = {k: hours_of(server / "train" / k / "run_manifest.json") for k in order}
    status.write_text(json.dumps({"state": "all six runs finished", "hours": {k: round(v, 3) for k, v in done.items()}, "total_hours": round(sum(done.values()), 3)}, indent=1))
    print(json.dumps({"state": "all six runs finished", "total_hours": round(sum(done.values()), 3)}), flush=True)


if __name__ == "__main__":
    main()
