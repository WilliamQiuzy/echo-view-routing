"""Configuration loading: .env (ECHO_* keys) + layered YAML configs.

Precedence (highest first): explicit overrides -> environment (ECHO_*) -> YAML -> defaults.
All paths resolve relative to the repository root unless absolute.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from echo_routing.errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PREFIX = "ECHO_"


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ without overriding existing keys."""
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if key and key not in os.environ:
            os.environ[key] = value.strip("'\"")


@dataclass(frozen=True)
class Paths:
    """Resolved filesystem layout for one machine (server or Mac)."""

    repo_root: Path
    data_root: Path
    ev9v_root: Path
    runs_root: Path
    checkpoints_root: Path
    cache_root: Path
    logs_root: Path

    @property
    def ev9v_images(self) -> Path:
        return self.ev9v_root / "Images"

    @property
    def ev9v_videos(self) -> Path:
        return self.ev9v_root / "Videos"

    @property
    def ev9v_raw(self) -> Path:
        return self.ev9v_root / "raw"

    @property
    def manifest_path(self) -> Path:
        return self.data_root / "manifests" / "ev9v_native.csv"


def _resolve(value: str | os.PathLike[str], root: Path) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else (root / p).resolve()


def load_paths(repo_root: Path | None = None, env: Mapping[str, str] | None = None) -> Paths:
    """Build Paths from ECHO_* environment overrides with repo-relative defaults."""
    root = (repo_root or REPO_ROOT).resolve()
    if env is None:
        _load_dotenv(root / ".env")
        env = os.environ
    data_root = _resolve(env.get("ECHO_DATA_ROOT", "data"), root)
    return Paths(
        repo_root=root,
        data_root=data_root,
        ev9v_root=_resolve(env.get("ECHO_EV9V_ROOT", data_root / "ev9v"), root),
        runs_root=_resolve(env.get("ECHO_RUNS_ROOT", "runs"), root),
        checkpoints_root=_resolve(env.get("ECHO_CHECKPOINTS_ROOT", "checkpoints"), root),
        cache_root=_resolve(env.get("ECHO_CACHE_ROOT", "cache"), root),
        logs_root=_resolve(env.get("ECHO_LOGS_ROOT", "logs"), root),
    )


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open() as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"config root must be a mapping: {path}")
    return loaded


def load_config(*config_paths: Path | str, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Merge one or more YAML files left-to-right, then apply explicit overrides.

    A YAML file may declare `extends: <relative path>`; it is loaded first and merged under it.
    """
    merged: dict[str, Any] = {}
    for cp in config_paths:
        path = _resolve(cp, REPO_ROOT)
        loaded = load_yaml(path)
        parent = loaded.pop("extends", None)
        if parent:
            merged = _deep_merge(merged, load_config(path.parent / parent))
        merged = _deep_merge(merged, loaded)
    if overrides:
        merged = _deep_merge(merged, overrides)
    return merged


def gpu_memory_fraction(default: float = 0.6) -> float:
    """Fraction of GPU memory this process may use (co-tenant safety)."""
    raw = os.environ.get("ECHO_GPU_MEM_FRACTION", str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"ECHO_GPU_MEM_FRACTION must be a float, got {raw!r}") from exc
    if not 0.05 <= value <= 1.0:
        raise ConfigError(f"ECHO_GPU_MEM_FRACTION out of range (0.05..1.0): {value}")
    return value
