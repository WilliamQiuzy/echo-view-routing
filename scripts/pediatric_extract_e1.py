#!/usr/bin/env python
"""E1 features and probe outputs for every frozen EchoNet-Pediatric video (configs/experiments/pediatric_transfer_v2.yaml).

Samples follow the frozen protocol (nearest frame to t = k/10 s at the AVI's own frame rate). Each sampled frame goes
through E1's EV9V preprocessing: PIL RGB, short side to 256 (bilinear), centre crop 224 (joint_visual.crop_uint8), CLIP
normalisation inside VisualEncoder, fp32 with TF32 off. The exported E1 encoder and the round-2 E1 probe are used
unchanged. Output: <cache_root>/pediatric/ped-ts10hz-e1crop224-v1/features/<E1 cache key>/<video_id>.npz (feat float16,
logit, prob), index.csv and meta.json. Data stay on the server.
"""
from __future__ import annotations

import argparse
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
from echo_routing.realvideo.pediatric import PROTOCOL_VERSION, VIDEO_ROOT, timestamp_indices  # noqa: E402
from echo_routing.temporal.joint_visual import crop_uint8  # noqa: E402
from scripts.extract_encoder_features import load_encoder, sha256_file, strict_fp32  # noqa: E402
from scripts.geometry_g0_reproduce import load_probe  # noqa: E402
from scripts.pediatric_extract import verify_protocol  # noqa: E402

PREPROCESS = "ped-ts10hz-e1crop224-v1"
HZ = 10.0
EXPORT = "models_frozen/e1_echoviewclip_probe/v1/visual_encoder_fp32.pt"
E1_MANIFEST = "runs/20261002-142832-e1_probe-38bbcf7c-s0/feature_manifest.json"


class Clips(Dataset):
    def __init__(self, videos: pd.DataFrame) -> None:
        self.v = videos.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.v)

    def __getitem__(self, i: int):
        r = self.v.iloc[i]; idx = timestamp_indices(int(r.n_frames), float(r.fps), HZ); want = set(idx.tolist())
        frames, k = [], 0; cap = cv2.VideoCapture(str(VIDEO_ROOT / r.view / "Videos" / r.file))
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if k in want:
                frames.append(crop_uint8(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))))
            k += 1
        cap.release()
        if k != int(r.n_frames) or len(frames) != idx.size:
            raise RuntimeError(f"{r.video_id}: decoded {k} frames (frozen {r.n_frames}), kept {len(frames)} of {idx.size}")
        return i, idx, torch.from_numpy(np.stack(frames))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol-dir", default=f"data/manifests/pediatric/{PROTOCOL_VERSION}")
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--encoder", default=EXPORT, help="encoder export (E1 export format); default E1")
    ap.add_argument("--head", default=None, help="Linear(512, 5) head (state_dict, in_dim); default the E1 probe")
    ap.add_argument("--key", default=None, help="output directory name under features/; default the E1 cache key")
    a = ap.parse_args(); strict_fp32(); paths = load_paths(); proto_dir = Path(a.protocol_dir); proto = verify_protocol(proto_dir)
    videos = pd.read_csv(proto_dir / "videos.csv", dtype={"clip_no": str})
    man = json.loads(Path(E1_MANIFEST).read_text())["encoder"]; key = a.key or man["cache_key"]
    head_path = Path(a.head) if a.head else Path(man["checkpoint"])
    if a.head is None and sha256_file(head_path) != man["sha256"]:
        raise SystemExit("E1 probe digest differs from its manifest")
    device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    enc, info = load_encoder(Path(a.encoder), device); probe = load_probe(head_path).to(device)
    out = paths.cache_root / "pediatric" / PREPROCESS / "features" / key; out.mkdir(parents=True, exist_ok=True)
    todo = videos if a.limit is None else videos.head(a.limit)
    todo = todo[[cached_n_samples(out / f"{v}.npz") is None for v in todo["video_id"]]]
    dl = DataLoader(Clips(todo), batch_size=None, shuffle=False, num_workers=a.workers, persistent_workers=a.workers > 0,
                    prefetch_factor=4 if a.workers > 0 else None)
    t0, n = time.time(), 0
    with torch.no_grad():
        for k, (i, idx, u8) in enumerate(dl):
            r = todo.iloc[int(i)]; idx = idx.numpy() if torch.is_tensor(idx) else np.asarray(idx)
            f = enc(u8.to(device, non_blocking=True)); lg = probe(f)
            save_video_features(out, VideoFeatures(r.video_id, idx, (idx / float(r.fps)).astype(np.float32), f.cpu().numpy().astype(np.float16),
                                                   torch.softmax(lg, 1).cpu().numpy(), lg.cpu().numpy()))
            n += idx.size
            if (k + 1) % 1000 == 0:
                print(f"  {k + 1}/{len(todo)} videos, {n} samples, {time.time() - t0:.0f} s", flush=True)
    rows = []
    for r in videos.itertuples(index=False):
        c = cached_n_samples(out / f"{r.video_id}.npz")
        if c is None and a.limit is None:
            raise RuntimeError(f"missing features for {r.video_id}")
        if c is not None:
            if c != int(r.n_samples):
                raise RuntimeError(f"{r.video_id}: {c} samples, protocol says {r.n_samples}")
            rows.append({"video_id": r.video_id, "split": r.role, "raw_label": r.view, "label_index": int(r.label), "n_samples": c,
                         "path": str(out / f"{r.video_id}.npz")})
    write_index(out, rows)
    write_meta(out, {"encoder_export": str(a.encoder), "encoder_sha256": sha256_file(Path(a.encoder)), "probe": str(head_path),
                     "probe_sha256": sha256_file(head_path),
                     "cache_key": key, "commit": info["commit"], "preprocess_version": PREPROCESS, "protocol_version": PROTOCOL_VERSION,
                     "protocol_videos_digest": proto["videos_digest"], "target_hz": HZ, "precision": "fp32, TF32 off",
                     "classes": man.get("classes"), "seconds": round(time.time() - t0, 1), "samples_this_call": n})
    print(f"wrote {len(rows)} videos to {out} ({n} samples in {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
