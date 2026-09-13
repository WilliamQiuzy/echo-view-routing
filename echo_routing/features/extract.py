"""Run a frozen encoder over sampled frames of every cine and populate the feature cache.

All requested (video, frame) items are streamed through ONE DataLoader (persistent workers), so
worker start-up is paid once per split, not once per video. Writes are per-video atomic.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from echo_routing.features.cache import VideoFeatures, cached_n_samples, save_video_features, variant_dir, write_index, write_meta
from echo_routing.features.encoder import FrameEncoder
from echo_routing.features.sampling import stride_indices
from echo_routing.ingest.frames import list_frame_paths, load_frame

VARIANT_GAMMA = {"orig": 1.0, "gamma090": 0.9, "gamma110": 1.1}  # proposal §5.2 pilot range


@dataclass(frozen=True)
class _Job:
    row: int
    video_id: str
    frame_idx: np.ndarray
    paths: list[Path]


class _FrameItems(Dataset):
    """Flat list of (job, position) pairs over all cines; returns (tensor, job index, position)."""

    def __init__(self, jobs: list[_Job], image_size: int, gamma: float) -> None:
        self.jobs = jobs; self.size = image_size; self.gamma = gamma
        self.items = [(j, k) for j, job in enumerate(jobs) for k in range(job.frame_idx.size)]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        j, k = self.items[i]; job = self.jobs[j]
        fi = int(job.frame_idx[k])
        x = load_frame(job.paths[min(fi, len(job.paths) - 1)], self.size, self.gamma)
        return torch.from_numpy(x), j, k


def _make_job(row: int, video_id: str, frames_dir: Path, n_frames: int, fps: float, target_hz: float) -> _Job:
    return _Job(row, video_id, stride_indices(n_frames, fps, target_hz), list_frame_paths(frames_dir))


@torch.no_grad()
def _run(model: FrameEncoder, device, jobs: list[_Job], image_size: int, gamma: float, fps: float,
         batch_size: int, num_workers: int, on_done) -> None:
    ds = _FrameItems(jobs, image_size, gamma)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    persistent_workers=num_workers > 0, pin_memory=device.type == "cuda")
    buf: dict[int, tuple[list, list]] = {}
    counts = {j: job.frame_idx.size for j, job in enumerate(jobs)}
    for x, js, ks in dl:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lg, f = model.forward_with_features(x.to(device, non_blocking=True))
        lg = lg.float().cpu().numpy(); f = f.float().cpu().numpy()
        for b, j in enumerate(js.tolist()):
            feats, logits = buf.setdefault(j, ([], []))
            feats.append(f[b]); logits.append(lg[b])
            if len(feats) == counts[j]:
                job = jobs[j]; logit = np.stack(logits); feat = np.stack(feats)
                prob = torch.softmax(torch.from_numpy(logit), dim=1).numpy()
                on_done(VideoFeatures(job.video_id, job.frame_idx, (job.frame_idx / fps).astype(np.float32), feat, prob, logit))
                del buf[j]
    if buf:
        raise RuntimeError(f"incomplete extraction for {len(buf)} cines")


def extract_video(model: FrameEncoder, device, frames_dir: Path, video_id: str, n_frames: int, fps: float,
                  target_hz: float, image_size: int, gamma: float, batch_size: int, num_workers: int = 0) -> VideoFeatures:
    """Single-cine convenience wrapper (tests, demos)."""
    out: list[VideoFeatures] = []
    _run(model.eval().to(device), device, [_make_job(0, video_id, frames_dir, n_frames, fps, target_hz)],
         image_size, gamma, fps, batch_size, num_workers, out.append)
    return out[0]


def extract_split(model: FrameEncoder, device, rows: pd.DataFrame, label_col: str, cache_root: Path, ckpt_hash: str,
                  variant: str, target_hz: float, image_size: int, batch_size: int = 128, num_workers: int = 4,
                  overwrite: bool = False, meta: dict | None = None, log_every: int = 200) -> Path:
    """rows: manifest rows (any splits) with frames_dir/n_frames. Returns the variant directory."""
    gamma = VARIANT_GAMMA[variant]
    out = variant_dir(cache_root, ckpt_hash, variant); out.mkdir(parents=True, exist_ok=True)
    model.eval().to(device)
    rows = rows.reset_index(drop=True)
    jobs = [_make_job(i, r.video_id, Path(r.frames_dir), int(r.n_frames), float(r.fps_playback), target_hz)
            for i, r in enumerate(rows.itertuples(index=False))
            if overwrite or cached_n_samples(out / f"{r.video_id}.npz") is None]  # unreadable == missing
    fps = float(rows["fps_playback"].iloc[0]) if len(rows) else 30.0
    done = {"n": 0}

    def on_done(vf: VideoFeatures) -> None:
        save_video_features(out, vf); done["n"] += 1
        if done["n"] % log_every == 0:
            print(f"  [{variant}] {done['n']}/{len(jobs)} cines", flush=True)

    if jobs:
        _run(model, device, jobs, image_size, gamma, fps, batch_size, num_workers, on_done)
    index_rows = []
    for r in rows.itertuples(index=False):
        n = cached_n_samples(out / f"{r.video_id}.npz")
        if n is None:
            raise RuntimeError(f"cache file unreadable after extraction: {out / r.video_id}.npz")
        index_rows.append({"video_id": r.video_id, "split": r.split, "raw_label": r.raw_label,
                           "label_index": int(getattr(r, label_col)), "n_samples": n, "path": str(out / f"{r.video_id}.npz")})
    write_index(out, index_rows)
    write_meta(out, {"ckpt_hash": ckpt_hash, "variant": variant, "gamma": gamma, "target_hz": target_hz,
                     "image_size": image_size, **(meta or {})})
    return out
