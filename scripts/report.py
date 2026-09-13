#!/usr/bin/env python
"""Merge every ladder run for one encoder into runs/_reports/baseline_ladder.{md,csv}."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse  # noqa: E402

from echo_routing.config import load_paths  # noqa: E402
from echo_routing.evaluate.report import collect_metrics, write_ladder  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt-hash", default=None); a = ap.parse_args()
    paths = load_paths()
    metrics, h = collect_metrics(paths.runs_root, a.ckpt_hash)
    if not metrics:
        raise SystemExit("no ladder metrics found")
    out = write_ladder(metrics, paths.runs_root / "_reports", f"# Baseline ladder (merged) · encoder {h} · methods: {', '.join(m['method_id'] for m in metrics)}")
    print(out.read_text())


if __name__ == "__main__":
    main()
