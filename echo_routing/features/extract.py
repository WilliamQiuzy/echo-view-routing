"""Run a frozen encoder over sampled frames of every cine and populate the feature cache."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from echo_routing.features.sampling import stride_indices
from echo_routing.ingest.frames import list_frame_paths, load_frame
from echo_routing.features.cache import VideoFeatures, save_video_features, variant_dir, write_index, write_meta
from echo_routing.features.encoder import FrameEncoder

VARIANT_GAMMA = {"orig": 1.0, "gamma090": 0.9, "gamma110": 1.1}  # proposal §5.2 pilot range


class _CineFrames(Dataset):
    def __init__(self, frames_dir: Path, indices: np.ndarray, image_size: int, gamma: float) -> None:
        self.paths = list_frame_paths(frames_dir); self.indices = indices; self.size = image_size; self.gamma = gamma

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        fi = int(self.indices[i])
        return torch.from_numpy(load_frame(self.paths[min(fi, len(self.paths) - 1)], self.size, self.gamma))


@torch.no_grad()
def extract_video(model: FrameEncoder, device, frames_dir: Path, video_id: str, n_frames: int, fps: float,
                  target_hz: float, image_size: int, gamma: float, batch_size: int, num_workers: int) -> VideoFeatures:
    idx = stride_indices(n_frames, fps, target_hz)
    dl = DataLoader(_CineFrames(frames_dir, idx, image_size, gamma), batch_size=batch_size, num_workers=num_workers)
    feats, logits = [], []
    for x in dl:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lg, f = model.forward_with_features(x.to(device, non_blocking=True))
        feats.append(f.float().cpu().numpy()); logits.append(lg.float().cpu().numpy())
    feat = np.concatenate(feats); logit = np.concatenate(logits)
    prob = torch.softmax(torch.from_numpy(logit), dim=1).numpy()
    return VideoFeatures(video_id, idx, (idx / fps).astype(np.float32), feat, prob, logit)


def extract_split(model: FrameEncoder, device, rows: pd.DataFrame, label_col: str, cache_root: Path, ckpt_hash: str,
                  variant: str, target_hz: float, image_size: int, batch_size: int = 128, num_workers: int = 4,
                  overwrite: bool = False, meta: dict | None = None) -> Path:
    """rows: manifest rows (any splits) with frames_dir/n_frames. Returns the variant directory."""
    gamma = VARIANT_GAMMA[variant]
    out = variant_dir(cache_root, ckpt_hash, variant)
    out.mkdir(parents=True, exist_ok=True)
    model.eval().to(device)
    index_rows = []
    for r in rows.itertuples(index=False):
        path = out / f"{r.video_id}.npz"
        if overwrite or not path.is_file():
            vf = extract_video(model, device, Path(r.frames_dir), r.video_id, int(r.n_frames), float(r.fps_playback),
                               target_hz, image_size, gamma, batch_size, num_workers)
            save_video_features(out, vf); n = vf.n
        else:
            with np.load(path) as z: n = int(z["prob"].shape[0])
        index_rows.append({"video_id": r.video_id, "split": r.split, "raw_label": r.raw_label,
                           "label_index": int(getattr(r, label_col)), "n_samples": n, "path": str(path)})
    write_index(out, index_rows)
    write_meta(out, {"ckpt_hash": ckpt_hash, "variant": variant, "gamma": gamma, "target_hz": target_hz,
                     "image_size": image_size, **(meta or {})})
    return out
