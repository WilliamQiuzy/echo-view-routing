#!/usr/bin/env python
"""B7: frozen EchoViewCLIP stage-1 applied with a sliding window (16 frames per 1.6 s window, stride 0.5 s).

STANDALONE: runs under envs/echoviewclip (torch 1.13). Uses the authors' model factory and their test preprocessing
(resize short side 256, centre crop 224, mean/std normalisation). Nine-code softmax mapped to family5 (PMPALA dropped).
  envs/echoviewclip/bin/python scripts/run_windowed_echoviewclip.py --plans data/window_plans/<hash>/plans.json \
      --ckpt models_frozen/echoviewclip_stage1/v1/best.pth --config third_party/adapters/echoviewclip/ev9v_stage1_test.yaml --out cache/echoviewclip_windowed/<hash>
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

RAW9_ORDER = ["PLHLA", "PMASA", "PMVLSA", "PASA", "A4C", "A5C", "PMPALA", "PPMLSA", "SC4C"]
RAW9_TO_FAMILY5 = {"PLHLA": 0, "PMASA": 1, "PMVLSA": 1, "PASA": 1, "A4C": 2, "A5C": 3, "PMPALA": None, "PPMLSA": 1, "SC4C": 4}
WINDOW, STRIDE, NUM_FRAMES = 16, 5, 16
MEAN = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1); STD = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)


def window_centres(n, stride):
    c = np.arange(stride // 2, n, stride, dtype=int)
    if c.size == 0 or c[-1] < n - 1 - stride // 2:
        c = np.append(c, n - 1)
    return c


def window_samples(centre, n, size):
    start = max(0, min(centre - size // 2, n - size)); return np.arange(start, min(n, start + size), dtype=int)


def apply_gamma(im, gamma):
    if abs(gamma - 1.0) < 1e-9:
        return im
    lut = [int(255 * ((i / 255.0) ** gamma) + 0.5) for i in range(256)]
    return im.point(lut * 3)


def preprocess(im: Image.Image) -> torch.Tensor:
    w, h = im.size; s = 256 / min(w, h); im = im.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BILINEAR)
    w, h = im.size; l, t = (w - 224) // 2, (h - 224) // 2; im = im.crop((l, t, l + 224, t + 224))
    x = torch.from_numpy(np.asarray(im, dtype=np.float32)).permute(2, 0, 1)
    return (x - MEAN) / STD


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--plans", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True); ap.add_argument("--repo", default="third_party/EchoViewCLIP"); ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    repo = Path(a.repo).resolve(); sys.path.insert(0, str(repo))
    import pandas as pd  # noqa: E402
    from utils.config import get_config  # noqa: E402  (official)
    from trainers import vificlip  # noqa: E402  (official)
    ns = argparse.Namespace(config=str(Path(a.config).resolve()), opts=None, batch_size=None, pretrained=None, resume=None, accumulation_steps=None,
                            output=str(Path(a.out).resolve()), only_test=True, local_rank=0)
    config = get_config(ns)
    logger = logging.getLogger("evc"); logger.addHandler(logging.StreamHandler()); logger.setLevel(logging.INFO)
    class_names = [n for _, n in pd.read_csv(config.DATA.LABEL_LIST).values.tolist()]
    model = vificlip.returnCLIP(config, logger=logger, class_names=class_names).cuda().eval()
    sd = torch.load(a.ckpt, map_location="cpu")["model"]
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    for k in ["prompt_learner.token_prefix", "prompt_learner.token_suffix", "prompt_learner.complete_text_embeddings"]:
        sd.pop(k, None)
    print("load:", model.load_state_dict(sd, strict=False))
    plans = json.loads(Path(a.plans).read_text()); frames_root = Path(plans["frames_root"]); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pcache = {}; items = list(plans["streams"].items())[: a.limit] if a.limit else list(plans["streams"].items())
    with torch.no_grad():
        for n_done, (sid, pl) in enumerate(items, 1):
            if (out / f"{sid}.npz").is_file():
                continue
            refs = pl["refs"]; n = pl["n"]; centres = window_centres(n, STRIDE); raws = []
            for c in centres:
                samp = window_samples(int(c), n, WINDOW)
                sel = np.linspace(0, len(samp) - 1, NUM_FRAMES).round().astype(int)  # 16 frames uniformly over the window (their SampleFrames)
                imgs = []
                for s in samp[sel]:
                    vid, fi, gamma = refs[s]
                    if vid not in pcache:
                        pcache[vid] = sorted(p for p in (frames_root / vid).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
                    with Image.open(pcache[vid][min(fi, len(pcache[vid]) - 1)]) as im:
                        imgs.append(preprocess(apply_gamma(im.convert("RGB"), gamma)))
                x = torch.stack(imgs).unsqueeze(0).cuda()  # (1, T, C, H, W)
                raws.append(torch.softmax(model(x).float(), dim=-1)[0].cpu().numpy())
            raw = np.asarray(raws)[np.abs(np.arange(n)[:, None] - centres[None, :]).argmin(1)]
            fam = np.zeros((n, 5))
            for j, name in enumerate(RAW9_ORDER):
                k = RAW9_TO_FAMILY5[name]
                if k is not None:
                    fam[:, k] += raw[:, j]
            np.savez_compressed(out / f"{sid}.npz", prob=fam.astype(np.float32), raw=raw.astype(np.float32))
            if n_done % 100 == 0:
                print(f"  {n_done}/{len(items)}", flush=True)
    print(f"pred_dir={out}")


if __name__ == "__main__":
    main()
