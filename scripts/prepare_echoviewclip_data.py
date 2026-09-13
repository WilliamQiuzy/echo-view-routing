#!/usr/bin/env python
"""Write EV9V in EchoViewCLIP's (ViFi-CLIP) layout: videos/ symlinks + train/val/test.txt ("name.mp4 <class_id>") +
labels csv ("id,name") using the dataset card's nine view names in STFM's INDEXOFLABEL order (raw9)."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echo_routing.audit.label_map import load_ontology  # noqa: E402
from echo_routing.config import load_paths  # noqa: E402
from echo_routing.manifest import load_manifest  # noqa: E402

CARD_NAMES = {"PLHLA": "Parasternal Long Axis View", "PMASA": "Parasternal Short Axis View at Apical Level",
              "PMVLSA": "Parasternal Short Axis View at Mitral Valve Level", "PASA": "Parasternal Short Axis View",
              "A4C": "Apical 4-Chamber View", "A5C": "Apical 5-Chamber View", "PMPALA": "Pulmonary Main Pulmonary Artery Long Axis View",
              "PPMLSA": "Parasternal Short Axis View at Papillary Muscle Level", "SC4C": "Subcostal 4-Chamber View"}


def main() -> None:
    paths = load_paths(); ont = load_ontology(); df = load_manifest(paths.manifest_path)
    out = paths.data_root / "echoviewclip"; (out / "videos").mkdir(parents=True, exist_ok=True)
    classes = list(ont.classes("raw9"))
    with (out / "labels_ev9v.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["id", "name"])
        for i, c in enumerate(classes):
            w.writerow([i, CARD_NAMES[c]])
    n = 0
    for split, fname in (("train", "train.txt"), ("validation", "val.txt"), ("test", "test.txt")):
        rows = df[(df["split"] == split) & df["video_path"].notna()]
        with (out / fname).open("w") as fh:
            for r in rows.itertuples(index=False):
                link = out / "videos" / f"{r.video_id}.mp4"
                if not link.exists():
                    link.symlink_to(Path(r.video_path).resolve()); n += 1
                fh.write(f"{r.video_id}.mp4 {classes.index(r.raw_label)}\n")
        print(f"{fname}: {len(rows)} videos")
    print(f"linked {n} videos into {out / 'videos'}; labels: {out / 'labels_ev9v.csv'}")


if __name__ == "__main__":
    main()
