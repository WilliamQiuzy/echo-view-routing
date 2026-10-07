#!/usr/bin/env python3
"""Training-only geometric preprocessing pilot, not a learned/anatomical segmenter.

The conservative 320x240 sector was drawn after inspecting the 30 training clips.
It is exploratory, is not a de-identification guarantee, and must not become a
production transform without broader training-set image review. Raw identifiers
and derived frames stay in the supplied server output directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CLASSES = ["PLAX", "PSAX", "A4C", "A5C", "SC4C"]
SEED = 20261006
PARAMETERS = {"source_width": 320, "source_height": 240, "apex_x": 157,
              "apex_y": 26, "radius": 207, "half_angle_degrees": 46.5}


def sector_mask() -> np.ndarray:
    yy, xx = np.mgrid[:240, :320]
    dx, dy = xx - PARAMETERS["apex_x"], yy - PARAMETERS["apex_y"]
    return ((dy >= 0) & (dx * dx + dy * dy <= PARAMETERS["radius"] ** 2)
            & (np.abs(np.arctan2(dx, dy)) <= math.radians(PARAMETERS["half_angle_degrees"])))


def legacy_crop(im: Image.Image) -> Image.Image:
    w, h = im.size
    scale = 256 / min(w, h)
    im = im.resize((max(224, round(w * scale)), max(224, round(h * scale))), Image.Resampling.BILINEAR)
    w, h = im.size
    left, top = (w - 224) // 2, (h - 224) // 2
    return im.crop((left, top, left + 224, top + 224))


def fan224(im: Image.Image, mask: np.ndarray) -> Image.Image:
    if im.size != (320, 240):
        raise ValueError("pilot template only supports 320x240 sources")
    arr = np.asarray(im.convert("RGB")).copy()
    arr[~mask] = 0
    y, x = np.where(mask)
    crop = Image.fromarray(arr).crop((int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1))
    w, h = crop.size
    scale = min(224 / w, 224 / h)
    resized = crop.resize((round(w * scale), round(h * scale)), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (224, 224))
    canvas.paste(resized, ((224 - resized.width) // 2, (224 - resized.height) // 2))
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    proto_bytes = args.protocol.read_bytes()
    proto = json.loads(proto_bytes)
    allowed = set(proto["train"]["clips"])
    with args.manifest.open() as f:
        rows = list(csv.DictReader(f))
    rng = random.Random(SEED)
    selected = []
    for label in CLASSES:
        pool = sorted((r for r in rows if r["video_id"] in allowed and r["family5_label"] == label),
                      key=lambda r: r["video_id"])
        rng.shuffle(pool)
        selected.extend(pool[:6])
    assert len(selected) == 30 and all(r["split"] == "train" for r in selected)
    mask = sector_mask()
    Image.fromarray(mask.astype(np.uint8) * 255).save(args.out / "mask.png")
    yy, xx = np.where(mask)
    bbox = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
    # The source-coordinate rectangle retained by the old 256/224 transform.
    nw, nh = round(320 * 256 / 240), 256
    left, top = (nw - 224) // 2, (nh - 224) // 2
    legacy_inside = ((xx + .5) * nw / 320 >= left) & ((xx + .5) * nw / 320 < left + 224)
    legacy_inside &= ((yy + .5) * nh / 240 >= top) & ((yy + .5) * nh / 240 < top + 224)
    private, summaries = [], []
    for k, row in enumerate(selected):
        directory = Path(row["frames_dir"])
        if not directory.is_absolute():
            directory = args.root / directory
        paths = sorted(directory.glob("*.jpg"), key=lambda p: int(p.stem))
        positions = np.unique(np.linspace(0, len(paths) - 1, 16).round().astype(int))
        frames = [Image.open(paths[int(i)]).convert("RGB") for i in positions]
        assert all(im.size == (320, 240) for im in frames)
        folder = args.out / f"sample_{k + 1:02d}"
        folder.mkdir()
        for i, im in zip(positions, frames):
            fan224(im, mask).save(folder / f"{int(i):06d}.png")
        middle = frames[len(frames) // 2]
        raw = np.asarray(middle).copy()
        masked = raw.copy(); masked[~mask] = 0
        # Retain the source only on the server. No patient/video ID is painted.
        panel = Image.new("RGB", (224 * 3, 248))
        panel.paste(legacy_crop(middle), (0, 24))
        panel.paste(legacy_crop(Image.fromarray(masked)), (224, 24))
        panel.paste(fan224(middle, mask), (448, 24))
        draw = ImageDraw.Draw(panel)
        for j, title in enumerate(["legacy", "mask + legacy", "fan + pad224"]):
            draw.text((224 * j + 4, 4), f"{k + 1:02d} {title}", fill="white")
        panel.save(folder / "comparison.png")
        gray = np.stack([np.asarray(im.convert("L"), dtype=float) for im in frames])
        activity = gray.std(0)
        stat = {"sample": k + 1, "label": row["family5_label"], "frames": len(frames),
                "outside_mask_activity_fraction": float(activity[~mask].sum() / max(activity.sum(), 1e-8))}
        summaries.append(stat)
        private.append({"sample": k + 1, "video_id": row["video_id"], "frame_positions": positions.tolist(),
                        "source_frame_names": [paths[int(i)].name for i in positions]})
    # Ten samples per sheet, including all 30 clips. No new sampling for visual review.
    for page in range(3):
        sheet = Image.new("RGB", (672, 248 * 10))
        for j in range(10):
            panel = Image.open(args.out / f"sample_{page * 10 + j + 1:02d}" / "comparison.png")
            sheet.paste(panel, (0, 248 * j))
        sheet.save(args.out / f"private_review_{page + 1}.jpg")
    result = {"kind": "exploratory_preprocessing_only", "seed": SEED, "n_clips": len(selected),
              "n_sampled_frames": sum(s["frames"] for s in summaries), "parameters": PARAMETERS,
              "mask_bbox_xyxy": bbox, "mask_area_fraction": float(mask.mean()),
              "mask_fraction_outside_legacy_field": float(1 - legacy_inside.mean()),
              "protocol_sha256": hashlib.sha256(proto_bytes).hexdigest(),
              "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(), "samples": summaries,
              "limits": "No anatomical mask ground truth, no recognition scoring, no training, no validation/test access. "
                        "Activity retention is not anatomy retention. Parameters chosen after training-image inspection."}
    (args.out / "private_manifest.json").write_text(json.dumps(private, indent=2))
    (args.out / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("samples",)}, indent=2))


if __name__ == "__main__":
    main()
