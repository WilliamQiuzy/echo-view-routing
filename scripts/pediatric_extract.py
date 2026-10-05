#!/usr/bin/env python
"""B0 probabilities and features for every frozen EchoNet-Pediatric video (task 2 of the 2026-10-01 round).

Requires the frozen protocol (scripts/pediatric_freeze.py) and checks its digests first. Samples follow the protocol:
nearest frame to t = k/10 s at the AVI's own frame rate; the same letterbox/ImageNet preprocessing and bfloat16
autocast as the EV9V B0 extraction. Output (isolated by preprocessing version and encoder hash):
<cache_root>/pediatric/<PREPROCESS_VERSION>/features/<B0_HASH>/<video_id>.npz plus index.csv and meta.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from echo_routing.config import gpu_memory_fraction, load_paths  # noqa: E402
from echo_routing.features.cache import VideoFeatures, cached_n_samples, save_video_features, write_index, write_meta  # noqa: E402
from echo_routing.features.encoder import checkpoint_hash, load_checkpoint  # noqa: E402
from echo_routing.ingest.frames import letterbox, to_tensor_array  # noqa: E402
from echo_routing.realvideo.pediatric import PREPROCESS_VERSION, PROTOCOL_VERSION, VIDEO_ROOT, timestamp_indices  # noqa: E402
from scripts.pediatric_freeze import B0_CKPT, B0_HASH, HZ  # noqa: E402


def verify_protocol(proto_dir: Path) -> dict:
    proto = json.loads((proto_dir / "protocol.json").read_text())
    for name, digest in proto["files_sha256"].items():
        if hashlib.sha256((proto_dir / name).read_bytes()).hexdigest() != digest:
            raise SystemExit(f"frozen file {name} changed after the freeze")
    return proto


class Clips(Dataset):
    def __init__(self, videos: pd.DataFrame, size: int) -> None:
        self.v = videos.reset_index(drop=True); self.size = size

    def __len__(self) -> int:
        return len(self.v)

    def __getitem__(self, i: int):
        r = self.v.iloc[i]
        idx = timestamp_indices(int(r.n_frames), float(r.fps), HZ)
        want = set(idx.tolist()); frames = []; k = 0
        cap = cv2.VideoCapture(str(VIDEO_ROOT / r.view / "Videos" / r.file))
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if k in want:
                frames.append(to_tensor_array(letterbox(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), self.size)))
            k += 1
        cap.release()
        if k != int(r.n_frames) or len(frames) != idx.size:
            raise RuntimeError(f"{r.video_id}: decoded {k} frames (frozen {r.n_frames}), kept {len(frames)} of {idx.size}")
        return i, idx, torch.from_numpy(np.stack(frames))


def main() -> None:
    ap = argparse.ArgumentParser(description="B0 (or another frame encoder of the same family) over the frozen EchoNet-Pediatric list")
    ap.add_argument("--ckpt", default=B0_CKPT, help="frame-encoder checkpoint; default the frozen B0. E0 (v3) uses the same preprocessing")
    ap.add_argument("--protocol-dir", default=f"data/manifests/pediatric/{PROTOCOL_VERSION}")
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(); paths = load_paths(); proto_dir = Path(args.protocol_dir)
    proto = verify_protocol(proto_dir)
    videos = pd.read_csv(proto_dir / "videos.csv", dtype={"clip_no": str})
    ckpt_hash = checkpoint_hash(Path(args.ckpt))
    if args.ckpt == B0_CKPT and ckpt_hash != B0_HASH:
        raise SystemExit("B0 checkpoint hash mismatch")
    model, meta = load_checkpoint(Path(args.ckpt)); size = int(meta.get("image_size", 224))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    model.eval().to(device)
    out = paths.cache_root / "pediatric" / PREPROCESS_VERSION / "features" / ckpt_hash; out.mkdir(parents=True, exist_ok=True)
    todo = videos if args.limit is None else videos.head(args.limit)
    todo = todo[[cached_n_samples(out / f"{v}.npz") is None for v in todo["video_id"]]]
    dl = DataLoader(Clips(todo, size), batch_size=None, shuffle=False, num_workers=args.workers,
                    persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers > 0 else None)
    t0 = time.time(); n_frames = 0
    with torch.no_grad():
        for k, (i, idx, x) in enumerate(dl):
            r = todo.iloc[int(i)]
            idx = idx.numpy() if torch.is_tensor(idx) else np.asarray(idx)  # the loader turns arrays into tensors
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                lg, f = model.forward_with_features(x.to(device, non_blocking=True))
            lg = lg.float().cpu().numpy(); f = f.float().cpu().numpy()
            prob = torch.softmax(torch.from_numpy(lg), dim=1).numpy()
            save_video_features(out, VideoFeatures(r.video_id, idx, (idx / float(r.fps)).astype(np.float32), f, prob, lg))
            n_frames += idx.size
            if (k + 1) % 500 == 0:
                print(f"  {k + 1}/{len(todo)} videos, {n_frames} samples, {time.time() - t0:.0f} s", flush=True)
    elapsed = time.time() - t0
    rows = []
    for r in videos.itertuples(index=False):
        n = cached_n_samples(out / f"{r.video_id}.npz")
        if n is None and args.limit is None:
            raise RuntimeError(f"missing features for {r.video_id}")
        if n is not None:
            if n != int(r.n_samples):
                raise RuntimeError(f"{r.video_id}: {n} samples, protocol says {r.n_samples}")
            rows.append({"video_id": r.video_id, "split": r.role, "raw_label": r.view, "label_index": int(r.label),
                         "n_samples": n, "path": str(out / f"{r.video_id}.npz")})
    write_index(out, rows)
    write_meta(out, {"ckpt": str(args.ckpt), "ckpt_hash": ckpt_hash, "classes": meta.get("classes"), "target_hz": HZ, "image_size": size,
                     "preprocess_version": PREPROCESS_VERSION, "protocol_version": PROTOCOL_VERSION,
                     "protocol_videos_digest": proto["videos_digest"], "extraction_seconds": round(elapsed, 1),
                     "samples_extracted_this_call": n_frames, "device": str(device)})
    print(f"wrote {len(rows)} videos to {out} ({n_frames} samples in {elapsed:.0f} s)")


if __name__ == "__main__":
    main()
