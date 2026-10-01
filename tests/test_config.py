from __future__ import annotations

from pathlib import Path

import pytest

from echo_routing.config import ConfigError, gpu_memory_fraction, load_config, load_paths


def test_load_paths_defaults_relative_to_repo(tmp_path: Path):
    paths = load_paths(repo_root=tmp_path, env={})
    assert paths.data_root == tmp_path / "data"
    assert paths.ev9v_images == tmp_path / "data" / "ev9v" / "Images"
    assert paths.manifest_path.name == "ev9v_native.csv"


def test_load_paths_env_override(tmp_path: Path):
    paths = load_paths(repo_root=tmp_path, env={"ECHO_DATA_ROOT": "/srv/data"})
    assert paths.data_root == Path("/srv/data")
    assert paths.ev9v_root == Path("/srv/data/ev9v")


def test_load_config_extends_and_overrides(tmp_path: Path):
    (tmp_path / "base.yaml").write_text("a: 1\nnested: {x: 1, y: 2}\n")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\nnested: {y: 3}\n")
    cfg = load_config(tmp_path / "child.yaml", overrides={"a": 9})
    assert cfg == {"a": 9, "nested": {"x": 1, "y": 3}}


def test_missing_config_raises(tmp_path: Path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.yaml")


def test_gpu_memory_fraction_validation(monkeypatch):
    monkeypatch.setenv("ECHO_GPU_MEM_FRACTION", "0.5")
    assert gpu_memory_fraction() == 0.5
    monkeypatch.setenv("ECHO_GPU_MEM_FRACTION", "2")
    with pytest.raises(ConfigError):
        gpu_memory_fraction()


def test_cache_root_is_stamped_and_legacy_caches_are_refused(tmp_path):
    from echo_routing.config import CACHE_MARKER, FRAME_ORDER_VERSION, check_cache_root
    from echo_routing.errors import ConfigError
    new = check_cache_root(tmp_path / "cache_v2")
    assert (new / CACHE_MARKER).read_text().strip() == FRAME_ORDER_VERSION
    assert check_cache_root(new) == new  # stamped roots are accepted again
    legacy = tmp_path / "cache"; (legacy / "features").mkdir(parents=True)
    with pytest.raises(ConfigError):
        check_cache_root(legacy)
    assert check_cache_root(legacy, allow_legacy=True) == legacy
    (new / CACHE_MARKER).write_text("name-order\n")
    with pytest.raises(ConfigError):
        check_cache_root(new)
