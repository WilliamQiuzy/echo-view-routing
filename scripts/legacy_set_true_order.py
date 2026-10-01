#!/usr/bin/env python
"""Order-versus-set decomposition of the frame-order defect (2026-09-30).

Rebuilds the legacy B0 cache with its own sampled images put back into true time order. Three caches then separate the
two effects of the old name sort:
- legacy `cache/`          images picked by position in name order, in name order;
- this cache               the same images, in true time order (order effect = legacy vs this);
- `cache_v2/`              images picked by position in true order, in true order (set effect = this vs cache_v2).
Each sample keeps its true 30-fps frame number, so the spacing between samples is irregular; the ladder still treats
the samples as 10 Hz, which slightly distorts within-clip timing. Output: <out-root>/features/<hash>/<variant>/.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from echo_routing.config import check_cache_root, load_paths  # noqa: E402
from echo_routing.features.cache import VideoFeatures, load_video_features, save_video_features  # noqa: E402
from echo_routing.ingest.frames import FRAME_EXTS  # noqa: E402


def name_order_numbers(frames_dir: Path) -> np.ndarray:
    """True frame number of each position in the legacy name-sorted listing."""
    names = sorted(n for n in os.listdir(frames_dir) if Path(n).suffix.lower() in FRAME_EXTS)
    return np.array([int(Path(n).stem) for n in names])


def main() -> None:
    ap = argparse.ArgumentParser(description="Legacy sampled set in true order")
    ap.add_argument("--legacy-root", default="cache"); ap.add_argument("--out-root", default="cache_v2_legacyset")
    ap.add_argument("--ckpt-hash", default="79f4a41af6e3"); ap.add_argument("--variants", default="orig,gamma090,gamma110")
    args = ap.parse_args(); paths = load_paths()
    legacy, out = paths.repo_root / args.legacy_root, check_cache_root(paths.repo_root / args.out_root)
    for variant in args.variants.split(","):
        src = legacy / "features" / args.ckpt_hash / variant; dst = out / "features" / args.ckpt_hash / variant
        dst.mkdir(parents=True, exist_ok=True)
        idx = pd.read_csv(src / "index.csv", dtype={"video_id": str})
        moved = 0
        for vid in idx["video_id"]:
            vf = load_video_features(src / f"{vid}.npz")
            true = name_order_numbers(paths.ev9v_images / vid)
            tf = true[np.minimum(vf.frame_idx, len(true) - 1)]
            order = np.argsort(tf, kind="stable"); moved += int((order != np.arange(len(order))).any())
            save_video_features(dst, VideoFeatures(vid, tf[order], (tf[order] / 30.0).astype(np.float32),
                                                   vf.feat[order], vf.prob[order], vf.logit[order]))
        idx["path"] = [str(dst / f"{v}.npz") for v in idx["video_id"]]
        idx.to_csv(dst / "index.csv", index=False)
        if (src / "meta.json").is_file():
            meta = json.loads((src / "meta.json").read_text()); meta["derived_from"] = str(src); meta["order"] = "legacy set, true order"
            (dst / "meta.json").write_text(json.dumps(meta, indent=1))
        print(f"{variant}: {len(idx)} clips, {moved} reordered -> {dst}", flush=True)
    shutil.copyfile(paths.repo_root / "cache_v2" / "FRAME_ORDER", out / "FRAME_ORDER")


if __name__ == "__main__":
    main()
