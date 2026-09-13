from __future__ import annotations

from pathlib import Path

import pytest

from echo_routing.manifest import (
    ManifestError, build_manifest, load_manifest, read_all_splits, save_manifest, task_subset,
)


def test_read_all_splits(tiny_raw_dir, ontology):
    entries = read_all_splits(tiny_raw_dir)
    assert [e.video_id for e in entries] == ["vid_a", "vid_b", "vid_c", "vid_d", "vid_e", "vid_f"]
    assert entries[0].split == "train" and entries[-1].split == "test"


def test_duplicate_across_splits_rejected(tiny_raw_dir):
    (tiny_raw_dir / "test_labeled.txt").write_text("vid_a A4C\n")
    with pytest.raises(ManifestError, match="duplicate"):
        read_all_splits(tiny_raw_dir)


def test_malformed_line_rejected(tiny_raw_dir):
    (tiny_raw_dir / "train_labeled.txt").write_text("vid_a A4C extra\n")
    with pytest.raises(ManifestError, match="expected"):
        read_all_splits(tiny_raw_dir)


def test_build_manifest_without_frames(tiny_raw_dir, ontology):
    df = build_manifest(tiny_raw_dir, ontology)
    assert len(df) == 6
    row = df.set_index("video_id").loc["vid_c"]
    assert row["raw_label"] == "PMPALA"
    assert row["family5_label"] is None or row["family5_label"] != row["family5_label"]  # None/NaN
    assert df["n_frames"].isna().all()


def test_build_manifest_counts_frames(tiny_raw_dir, ontology, tmp_path: Path):
    images = tmp_path / "Images"
    (images / "vid_a").mkdir(parents=True)
    for i in range(1, 31):
        (images / "vid_a" / f"frame_{i:06d}.jpg").write_bytes(b"")
    df = build_manifest(tiny_raw_dir, ontology, images_root=images)
    row = df.set_index("video_id").loc["vid_a"]
    assert row["n_frames"] == 30 and row["duration_s"] == pytest.approx(1.0)
    assert row["frames_dir"].endswith("vid_a")


def test_save_and_load_roundtrip(tiny_raw_dir, ontology, tmp_path: Path):
    df = build_manifest(tiny_raw_dir, ontology)
    path = save_manifest(df, tmp_path / "m" / "manifest.csv")
    loaded = load_manifest(path)
    assert len(loaded) == 6
    assert task_subset(loaded, "family5")["video_id"].tolist() == ["vid_a", "vid_b", "vid_d", "vid_e", "vid_f"]
    assert len(task_subset(loaded, "raw9")) == 6
