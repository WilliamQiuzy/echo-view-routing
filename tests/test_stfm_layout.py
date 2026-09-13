from __future__ import annotations

from pathlib import Path

from echo_routing.audit.label_map import load_ontology
from echo_routing.ingest.stfm_layout import build_stfm_layout
from echo_routing.manifest import build_manifest


def test_stfm_layout(tiny_raw_dir, tmp_path: Path):
    images = tmp_path / "Images"
    for v in ("vid_a", "vid_d", "vid_e"):
        (images / v).mkdir(parents=True); (images / v / "frame_000001.jpg").write_bytes(b"")
    df = build_manifest(tiny_raw_dir, load_ontology(), images_root=images)
    out = build_stfm_layout(df, images, tmp_path / "EchoData")
    assert (out / "labels.csv").read_text().splitlines()[0] == "names,labels"
    assert (out / "train.txt").read_text().split() == ["vid_a"]
    assert (out / "validation.txt").read_text().split() == ["vid_d"]
    assert (out / "Images").is_symlink() and (out / "Images" / "vid_e").is_dir()
