"""Shared CLI helpers (argument parsing, run directories)."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from echo_routing.config import REPO_ROOT, load_config, load_paths


def parse_set(pairs: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in pairs:
        key, _, raw = p.partition("=")
        if not key or not _:
            raise SystemExit(f"--set expects key.path=value, got {p!r}")
        node = out
        parts = key.split(".")
        for k in parts[:-1]:
            node = node.setdefault(k, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return out


def base_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("-c", "--config", default="configs/base.yaml", help="YAML config (may `extends:`)")
    ap.add_argument("--set", action="append", default=[], help="override key.path=value (repeatable)")
    return ap


def load_cfg(args) -> dict[str, Any]:
    return load_config(REPO_ROOT / args.config, overrides=parse_set(args.set))


def config_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:8]


def new_run_dir(cfg: dict[str, Any], name: str | None = None) -> Path:
    paths = load_paths()
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}-{name or cfg.get('run_name', 'run')}-{config_hash(cfg)}-s{cfg.get('seed', 0)}"
    run_dir = paths.runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return run_dir
