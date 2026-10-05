#!/usr/bin/env python
"""B7 on EchoNet-Pediatric (configs/experiments/pediatric_transfer_v2.yaml): the official EchoViewCLIP stage-1 video
model with the EV9V sliding windows (16 samples per 1.6 s window, stride 0.5 s, 16 frames per window), its test
preprocessing, all nine outputs kept.

STANDALONE: runs under envs/echoviewclip (torch 1.13). Windowing, preprocessing and model loading come from
scripts/run_windowed_echoviewclip.py unchanged; only the frame source differs: the frozen Pediatric protocol's samples
(nearest frame to t = k/10 s at the AVI's own frame rate), decoded from the AVI. Each sample takes the softmax of the
window whose centre is nearest. Output: <out>/<video_id>.npz with raw (n, 9) in RAW9_ORDER. Data stay on the server.
  envs/echoviewclip/bin/python scripts/pediatric_b7.py --out cache_v2/pediatric/ped-ts10hz-b7window-v1/a74c7b233e79
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_windowed_echoviewclip import NUM_FRAMES, RAW9_ORDER, STRIDE, WINDOW, preprocess, window_centres, window_samples  # noqa: E402

PROTO_DIR = Path("data/manifests/pediatric/pediatric-external-v1")
VIDEO_ROOT = Path("data/echonet/pediatric/echonetpediatric/pediatric_echo_avi/pediatric_echo_avi")
HZ = 10.0


def timestamp_indices(n_frames: int, fps: float, hz: float) -> np.ndarray:
    """Same rule as echo_routing.realvideo.pediatric.timestamp_indices (checked against the frozen sample counts)."""
    idx = np.floor(np.arange(0.0, n_frames / fps, 1.0 / hz) * fps + 0.5).astype(int)
    return np.unique(idx[idx < n_frames])


def load_model(ckpt: str, config_path: str, repo: Path, out: Path):
    sys.path.insert(0, str(repo))
    import pandas as pd  # noqa: E402
    from utils.config import get_config  # noqa: E402  (official)
    from trainers import vificlip  # noqa: E402  (official)
    ns = argparse.Namespace(config=str(Path(config_path).resolve()), opts=None, batch_size=None, pretrained=None, resume=None,
                            accumulation_steps=None, output=str(out.resolve()), only_test=True, local_rank=0)
    config = get_config(ns); logger = logging.getLogger("evc"); logger.addHandler(logging.StreamHandler()); logger.setLevel(logging.INFO)
    class_names = [n for _, n in pd.read_csv(config.DATA.LABEL_LIST).values.tolist()]
    model = vificlip.returnCLIP(config, logger=logger, class_names=class_names).cuda().eval()
    sd = torch.load(ckpt, map_location="cpu")["model"]; sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    for k in ["prompt_learner.token_prefix", "prompt_learner.token_suffix", "prompt_learner.complete_text_embeddings"]:
        sd.pop(k, None)
    print("load:", model.load_state_dict(sd, strict=False))
    model = model.float()
    for mod in model.modules():
        if isinstance(getattr(mod, "dtype", None), torch.dtype):
            mod.dtype = torch.float32
    return model


def frames_of(row: dict) -> list[Image.Image]:
    idx = timestamp_indices(int(row["n_frames"]), float(row["fps"]), HZ); want = set(idx.tolist())
    if idx.size != int(row["n_samples"]):
        raise RuntimeError(f"{row['video_id']}: {idx.size} samples, frozen {row['n_samples']}")
    out, k = [], 0; cap = cv2.VideoCapture(str(VIDEO_ROOT / row["view"] / "Videos" / row["file"]))
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if k in want:
            out.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        k += 1
    cap.release()
    if k != int(row["n_frames"]) or len(out) != idx.size:
        raise RuntimeError(f"{row['video_id']}: decoded {k} frames (frozen {row['n_frames']}), kept {len(out)} of {idx.size}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="models_frozen/echoviewclip_stage1/v1/best.pth")
    ap.add_argument("--config", default="third_party/adapters/echoviewclip/ev9v_stage1_test.yaml")
    ap.add_argument("--repo", default="third_party/EchoViewCLIP"); ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    proto = json.loads((PROTO_DIR / "protocol.json").read_text())
    for name, digest in proto["files_sha256"].items():
        if hashlib.sha256((PROTO_DIR / name).read_bytes()).hexdigest() != digest:
            raise SystemExit(f"frozen file {name} changed after the freeze")
    rows = list(csv.DictReader(open(PROTO_DIR / "videos.csv")))[: a.limit] if a.limit else list(csv.DictReader(open(PROTO_DIR / "videos.csv")))
    model = load_model(a.ckpt, a.config, Path(a.repo).resolve(), out)
    with torch.no_grad():
        for n_done, row in enumerate(rows, 1):
            if (out / f"{row['video_id']}.npz").is_file():
                continue
            ims = [preprocess(im) for im in frames_of(row)]; n = len(ims); centres = window_centres(n, STRIDE); raws = []
            for c in centres:
                samp = window_samples(int(c), n, WINDOW); sel = np.linspace(0, len(samp) - 1, NUM_FRAMES).round().astype(int)
                x = torch.stack([ims[s] for s in samp[sel]]).unsqueeze(0).cuda()
                raws.append(torch.softmax(model(x).float(), dim=-1)[0].cpu().numpy())
            raw = np.asarray(raws)[np.abs(np.arange(n)[:, None] - centres[None, :]).argmin(1)]
            np.savez_compressed(out / f"{row['video_id']}.npz", raw=raw.astype(np.float32), classes=np.array(RAW9_ORDER))
            if n_done % 500 == 0:
                print(f"  {n_done}/{len(rows)}", flush=True)
    (out / "meta.json").write_text(json.dumps({"ckpt": a.ckpt, "config": a.config, "window": WINDOW, "stride": STRIDE, "frames": NUM_FRAMES,
                                               "classes": RAW9_ORDER, "protocol": proto["protocol_version"], "hz": HZ}, indent=1))
    print(f"pred_dir={out}")


if __name__ == "__main__":
    main()
