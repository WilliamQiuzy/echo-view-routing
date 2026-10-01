#!/usr/bin/env python
"""Round-1 cycle feasibility on EV9V (task 3): cardiac cycles from image motion only.

EV9V frames carry no ECG, so the cycle comes from echo_routing/realvideo/cycles.py (validated against the MIMIC ECG
in scripts/report_realvideo_round1.py). View predictions are never used. Writes <out>/ev9v_cycles.csv, the same-phase
frames per clip (<out>/ev9v_cycles.json) and <out>/ev9v_cycle_check.png: for random clips, frames at successive
same-phase boundaries and one frame half a cycle later, for a visual check.
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from echo_routing.config import load_paths  # noqa: E402
from echo_routing.ingest.frames import list_frame_paths  # noqa: E402
from echo_routing.realvideo.cycles import estimate_cycle  # noqa: E402
from echo_routing.realvideo.screening import patient_of  # noqa: E402

DOWN = 2


def grey_frames(frames_dir: str) -> np.ndarray:
    out = []
    for p in list_frame_paths(Path(frames_dir)):
        with Image.open(p) as im:
            g = np.asarray(im.convert("L"), dtype=np.float32)
        h, w = g.shape
        out.append(g[: h - h % DOWN, : w - w % DOWN].reshape(h // DOWN, DOWN, w // DOWN, DOWN).mean((1, 3)))
    return np.stack(out)


def one(args: tuple[str, str, float]) -> dict:
    vid, frames_dir, fps = args
    g = grey_frames(frames_dir)
    c = estimate_cycle(g, fps)
    return {"video_id": vid, "n_frames": len(g), "period_s": c.period_s, "peak": c.peak, "dip": c.dip, "cv": c.cv,
            "n_cycles": c.n_cycles, "boundaries": c.boundaries.tolist()}


def check_grid(rows: pd.DataFrame, out: Path, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    pick = rows.iloc[rng.choice(len(rows), size=min(n, len(rows)), replace=False)]
    w, h, pad, head = 160, 120, 4, 16
    img = Image.new("RGB", (5 * (w + pad) + pad, len(pick) * (h + head + pad) + pad), "white"); d = ImageDraw.Draw(img)
    for i, r in enumerate(pick.itertuples(index=False)):
        paths = list_frame_paths(Path(r.frames_dir)); b = r.boundaries[:4]
        half = int(round(b[0] + r.period_s * 30 / 2)) if b else 0
        y = pad + i * (h + head + pad)
        d.text((pad, y), f"{r.video_id}  period {r.period_s:.2f}s  peak {r.peak:.2f}  (4 same-phase frames | half-cycle)", fill="black")
        for j, k in enumerate(list(b) + [None] * (4 - len(b)) + [half]):
            if k is None or k >= len(paths):
                continue
            img.paste(Image.open(paths[k]).convert("RGB").resize((w, h)), (pad + j * (w + pad), y + head))
    img.save(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="EV9V image-motion cycles")
    ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--check-n", type=int, default=12); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(); paths = load_paths(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(paths.manifest_path)
    man = man[(man["dataset"] == "EV9V") & man["split"].isin(["validation", "test"]) & man["family5_label"].notna()]
    jobs = [(r.video_id, r.frames_dir, float(r.fps_playback)) for r in man.itertuples(index=False)]
    with Pool(args.workers) as pool:
        res = pool.map(one, jobs, chunksize=8)
    df = pd.DataFrame(res).merge(man[["video_id", "split", "duration_s", "frames_dir"]], on="video_id")
    df["patient"] = df["video_id"].map(patient_of)
    (out / "ev9v_cycles.json").write_text(json.dumps({r["video_id"]: r["boundaries"] for r in res}))
    df.drop(columns=["boundaries", "frames_dir"]).to_csv(out / "ev9v_cycles.csv", index=False)
    ok = df[df["peak"] >= 0.3]
    if len(ok):
        check_grid(ok, out / "ev9v_cycle_check.png", args.check_n, args.seed)
    print(df[["period_s", "peak", "dip", "cv", "n_cycles"]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
