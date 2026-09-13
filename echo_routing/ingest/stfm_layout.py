"""Create the data layout expected by the vendored STFM code (labels.csv + split txt + Images symlink)."""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

STFM_SPLIT_FILES = {"train": "train.txt", "validation": "validation.txt", "test": "test.txt"}


def build_stfm_layout(manifest: pd.DataFrame, images_root: Path, out_dir: Path, limit_per_split: int | None = None) -> Path:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = manifest[manifest["frames_dir"].notna()]
    if limit_per_split:
        df = df.groupby("split", group_keys=False).head(limit_per_split)
    with (out_dir / "labels.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["names", "labels"])
        for r in df.itertuples(index=False):
            w.writerow([r.video_id, r.raw_label])
    for split, fname in STFM_SPLIT_FILES.items():
        (out_dir / fname).write_text("\n".join(df[df["split"] == split]["video_id"]) + "\n")
    link = out_dir / "Images"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(Path(images_root).resolve(), target_is_directory=True)
    return out_dir
