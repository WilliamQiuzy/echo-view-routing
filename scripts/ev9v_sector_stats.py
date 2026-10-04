#!/usr/bin/env python
"""Mean and SD of grey values inside the echo sector of EV9V training frames: the target to which every pretraining
slice is standardised (configs/experiments/geometry_pretraining_v1.yaml, pairs.controls.standardise).

Clips are drawn from the protocol's training clips only. Frames are the cached 10 Hz indices (at most 16 per clip),
read as grey. The sector is the convex hull of the largest region whose grey value varies over the clip (see sector),
so static overlays stay out and the dark blood pool stays in. Writes <out> (JSON).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.extract_echoviewclip_features import jobs_from  # noqa: E402
from scripts.run_windowed_echoviewclip import frame_paths  # noqa: E402


def sector(frames: np.ndarray) -> np.ndarray:
    """The fan of live image: pixels whose grey value varies over the clip (speckle; text, bars and scales stay
    still), opened to drop specks, the largest connected region, then its convex hull, which includes the dark blood
    pool. A first version closed a brightness mask with a large kernel; that eroded from the image border and turned
    every sector into the same 160 x 240 rectangle."""
    from PIL import ImageDraw
    from scipy.spatial import ConvexHull
    m = ndimage.binary_opening(frames.std(0) > 3.0, structure=np.ones((3, 3), bool))
    lab, n = ndimage.label(m)
    if not n:
        return m
    m = lab == (1 + int(np.argmax(ndimage.sum(m, lab, range(1, n + 1)))))
    yx = np.argwhere(m); hull = yx[ConvexHull(yx).vertices]
    canvas = Image.new("L", (m.shape[1], m.shape[0]), 0)
    ImageDraw.Draw(canvas).polygon([(int(x), int(y)) for y, x in hull], fill=1, outline=1)
    return np.asarray(canvas, bool)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True); ap.add_argument("--manifest", required=True); ap.add_argument("--v3-cache", required=True)
    ap.add_argument("--clips", type=int, default=300); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    a = ap.parse_args(); proto = json.loads(Path(a.protocol).read_text()); train = set(proto["train"]["clips"])
    jobs = [j for j in jobs_from(proto, Path(a.manifest), Path(a.v3_cache)) if j["video_id"] in train]
    pick = np.random.default_rng(a.seed).choice(len(jobs), size=min(a.clips, len(jobs)), replace=False)
    total = s1 = s2 = 0.0; per_clip = []
    for i in sorted(pick):
        j = jobs[i]; fp = frame_paths(Path(j["frames_dir"]))
        idx = j["frame_idx"][np.linspace(0, len(j["frame_idx"]) - 1, min(16, len(j["frame_idx"]))).astype(int)]
        f = np.stack([np.asarray(Image.open(fp[min(int(k), len(fp) - 1)]).convert("L"), np.float64) for k in idx])
        m = sector(f); x = f[:, m]
        total += x.size; s1 += x.sum(); s2 += (x ** 2).sum(); per_clip.append({"video_id": j["video_id"], "mean": float(x.mean()), "sd": float(x.std()),
                                                                               "sector_fraction": float(m.mean())})
    mean = s1 / total; sd = float(np.sqrt(s2 / total - mean ** 2))
    pc = {k: [float(np.percentile([c[k] for c in per_clip], q)) for q in (10, 50, 90)] for k in ("mean", "sd", "sector_fraction")}
    out = {"mean": float(mean), "sd": sd, "n_clips": len(per_clip), "n_pixels": int(total), "per_clip_p10_p50_p90": pc,
           "split": "train", "frames": "cached 10 Hz indices, at most 16 per clip, grey"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
