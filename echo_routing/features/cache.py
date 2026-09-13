"""Per-video feature/probability cache produced by a frozen encoder.

Layout: {cache_root}/features/{ckpt_hash}/{variant}/{video_id}.npz
  frame_idx int32[N]   original frame indices sampled at target_hz
  t_sec     float32[N] frame_idx / fps_playback
  feat      float16[N, D]
  prob      float32[N, C]
  logit     float32[N, C]
plus {variant}/index.csv (video_id, split, raw_label, label_index, n_samples, path) and meta.json.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VideoFeatures:
    video_id: str
    frame_idx: np.ndarray
    t_sec: np.ndarray
    feat: np.ndarray
    prob: np.ndarray
    logit: np.ndarray

    @property
    def n(self) -> int:
        return int(self.prob.shape[0])


def variant_dir(cache_root: Path, ckpt_hash: str, variant: str) -> Path:
    return Path(cache_root) / "features" / ckpt_hash / variant


def save_video_features(dir_: Path, vf: VideoFeatures) -> Path:
    dir_ = Path(dir_); dir_.mkdir(parents=True, exist_ok=True)
    path = dir_ / f"{vf.video_id}.npz"
    np.savez_compressed(path, frame_idx=vf.frame_idx.astype(np.int32), t_sec=vf.t_sec.astype(np.float32),
                        feat=vf.feat.astype(np.float16), prob=vf.prob.astype(np.float32), logit=vf.logit.astype(np.float32))
    return path


def load_video_features(path: Path) -> VideoFeatures:
    with np.load(Path(path)) as z:
        return VideoFeatures(Path(path).stem, z["frame_idx"], z["t_sec"], z["feat"].astype(np.float32), z["prob"], z["logit"])


def write_index(dir_: Path, rows: list[dict]) -> Path:
    path = Path(dir_) / "index.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def read_index(dir_: Path) -> pd.DataFrame:
    return pd.read_csv(Path(dir_) / "index.csv", dtype={"video_id": str})


def write_meta(dir_: Path, meta: dict) -> Path:
    path = Path(dir_) / "meta.json"
    path.write_text(json.dumps(meta, indent=2, default=str))
    return path
