#!/usr/bin/env python
"""Build the native EV9V manifest from the split files + extracted frames/videos."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import base_parser  # noqa: E402

from echo_routing.audit.label_map import load_ontology  # noqa: E402
from echo_routing.config import load_paths  # noqa: E402
from echo_routing.manifest import build_manifest, save_manifest, summarize, task_subset  # noqa: E402


def main() -> None:
    ap = base_parser("Build data/manifests/ev9v_native.csv")
    ap.add_argument("--revision", default="094135b4da40c9aed9d9550c65305dc8715c7d3a")
    args = ap.parse_args()
    paths = load_paths()
    ont = load_ontology()
    df = build_manifest(paths.ev9v_raw, ont, images_root=paths.ev9v_images, videos_root=paths.ev9v_videos,
                        dataset_revision=args.revision)
    out = save_manifest(df, paths.manifest_path)
    print(f"wrote {out} ({len(df)} rows)")
    print(summarize(df).to_string())
    fam = task_subset(df, "family5")
    print("family5 rows by split:", fam.groupby("split")["video_id"].count().to_dict())
    print("with frames on disk:", int(df["frames_dir"].notna().sum()), "| with mp4:", int(df["video_path"].notna().sum()))
    if df["n_frames"].notna().any():
        print("n_frames: median", df["n_frames"].median(), "min", df["n_frames"].min(), "max", df["n_frames"].max())


if __name__ == "__main__":
    main()
