from __future__ import annotations

from pathlib import Path

import pytest

from echo_routing.audit.label_map import load_ontology


@pytest.fixture(scope="session")
def ontology():
    return load_ontology()


@pytest.fixture
def tiny_raw_dir(tmp_path: Path) -> Path:
    """Three split files in EV9V format with a handful of videos."""
    (tmp_path / "train_labeled.txt").write_text("vid_a A4C\nvid_b PLHLA\nvid_c PMPALA\n")
    (tmp_path / "validation_labeled.txt").write_text("vid_d PMASA\n")
    (tmp_path / "test_labeled.txt").write_text("vid_e SC4C\nvid_f A5C\n")
    return tmp_path


@pytest.fixture
def synthetic_resource_window(monkeypatch):
    """Give synthetic children a test window without changing frozen production deadlines."""
    import importlib
    import time

    def set_window(module_name):
        module = importlib.import_module(module_name)
        monkeypatch.setattr(module, "DEADLINE_EPOCH", time.time() + 3 * 24 * 3600)

    return set_window
