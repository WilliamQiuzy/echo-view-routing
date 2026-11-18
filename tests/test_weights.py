"""Finding the weights and telling the published files from any others."""
import hashlib
from pathlib import Path

import pytest

from echoview import weights


def make_folder(root: Path, names: dict[int, tuple[str, ...]]) -> Path:
    for seed, files in names.items():
        folder = root / f"C2_s{seed}"
        folder.mkdir(parents=True)
        for name in files:
            (folder / name).write_bytes(f"{seed}:{name}".encode())
    return root


def published(monkeypatch, root: Path, seeds=(0, 1, 2)) -> None:
    table = {seed: {name: hashlib.sha256(f"{seed}:{name}".encode()).hexdigest() for name in weights.FILE_NAMES} for seed in seeds}
    monkeypatch.setattr(weights, "WEIGHT_SHA256", table)


def test_safetensors_is_preferred_over_the_training_output(tmp_path):
    folder = make_folder(tmp_path, {0: ("final.pt", "model.safetensors"), 1: ("final.pt",)})
    assert weights.seed_file(folder, 0).name == "model.safetensors"
    assert weights.seed_file(folder, 1).name == "final.pt"


def test_a_seed_without_weights_is_named(tmp_path):
    folder = make_folder(tmp_path, {0: ("final.pt",)})
    with pytest.raises(weights.WeightsError, match="C2_s1"):
        weights.seed_file(folder, 1)


def test_verify_tells_published_files_from_changed_ones(tmp_path, monkeypatch):
    folder = make_folder(tmp_path, {0: ("final.pt",), 1: ("final.pt",), 2: ("model.safetensors",)})
    published(monkeypatch, folder)
    assert weights.verify(folder) == {"C2_s0/final.pt": True, "C2_s1/final.pt": True, "C2_s2/model.safetensors": True}
    (folder / "C2_s1" / "final.pt").write_bytes(b"something else")
    assert weights.verify(folder)["C2_s1/final.pt"] is False


def test_a_file_without_a_published_checksum_is_never_published(tmp_path, monkeypatch):
    folder = make_folder(tmp_path, {0: ("final.pt",)})
    monkeypatch.setattr(weights, "WEIGHT_SHA256", {0: {"final.pt": ""}})
    assert weights.is_published(folder / "C2_s0" / "final.pt", 0) is False


def test_the_given_folder_wins_over_the_environment(tmp_path, monkeypatch):
    given, other = make_folder(tmp_path / "given", {0: ("final.pt",)}), make_folder(tmp_path / "other", {0: ("final.pt",)})
    monkeypatch.setenv(weights.ENVIRONMENT_VARIABLE, str(other))
    assert weights.resolve(given) == given
    assert weights.resolve(None) == other


def test_without_a_folder_the_weights_are_downloaded(tmp_path, monkeypatch):
    monkeypatch.delenv(weights.ENVIRONMENT_VARIABLE, raising=False)
    asked = []
    monkeypatch.setattr(weights, "download", lambda seeds=weights.SEEDS: asked.append(seeds) or tmp_path)
    assert weights.resolve(None, (0,)) == tmp_path
    assert asked == [(0,)]


def test_a_path_that_is_no_folder_is_refused(tmp_path):
    with pytest.raises(weights.WeightsError, match="not a folder"):
        weights.resolve(tmp_path / "missing")
