#!/usr/bin/env python
"""B7 on listed 16-frame windows, without sliding (configs/experiments/b7_window_target_v1.yaml, step 3).

The frozen EchoViewCLIP stage-1 checkpoint, the authors' model factory, and the preprocessing and nine-to-five map of
scripts/run_windowed_echoviewclip.py; one window per forward pass (batch 1, float32), as in that runner.
STANDALONE: runs under envs/echoviewclip (torch 1.13).
  envs/echoviewclip/bin/python scripts/b7_window_probe.py --windows <windows.json> \
      --ckpt models_frozen/echoviewclip_stage1/v1/best.pth \
      --config models_frozen/echoviewclip_stage1/v1/ev9v_stage1_test.yaml --out <dir>
windows.json = {"frames_root": ..., "windows": {id: [[video_id, frame_idx, gamma] x 16]}}. The windows carry frame
references only, never labels. Writes <dir>/probe.npz (ids, raw nine-code softmax, prob five-family map).
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_windowed_echoviewclip import RAW9_ORDER, RAW9_TO_FAMILY5, apply_gamma, frame_paths, preprocess  # noqa: E402


def build_model(repo: Path, config_path: str, ckpt: str, out: str):
    """As in scripts/run_windowed_echoviewclip.main: the authors' factory, the frozen weights, float32 without apex."""
    sys.path.insert(0, str(repo))
    import pandas as pd  # noqa: E402
    from trainers import vificlip  # noqa: E402  (official)
    from utils.config import get_config  # noqa: E402  (official)
    ns = argparse.Namespace(config=str(Path(config_path).resolve()), opts=None, batch_size=None, pretrained=None, resume=None,
                            accumulation_steps=None, output=str(Path(out).resolve()), only_test=True, local_rank=0)
    config = get_config(ns)
    logger = logging.getLogger("evc"); logger.addHandler(logging.StreamHandler()); logger.setLevel(logging.INFO)
    class_names = [n for _, n in pd.read_csv(config.DATA.LABEL_LIST).values.tolist()]
    model = vificlip.returnCLIP(config, logger=logger, class_names=class_names).cuda().eval()
    sd = torch.load(ckpt, map_location="cpu")["model"]
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    for k in ["prompt_learner.token_prefix", "prompt_learner.token_suffix", "prompt_learner.complete_text_embeddings"]:
        sd.pop(k, None)
    print("load:", model.load_state_dict(sd, strict=False))
    model = model.float()
    for mod in model.modules():
        if isinstance(getattr(mod, "dtype", None), torch.dtype):
            mod.dtype = torch.float32
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--config", required=True)
    ap.add_argument("--repo", default="third_party/EchoViewCLIP"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spec = json.loads(Path(a.windows).read_text()); root = Path(spec["frames_root"])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model = build_model(Path(a.repo).resolve(), a.config, a.ckpt, a.out)
    fam = np.zeros((len(RAW9_ORDER), 5))
    for j, name in enumerate(RAW9_ORDER):
        if RAW9_TO_FAMILY5[name] is not None:
            fam[j, RAW9_TO_FAMILY5[name]] = 1.0
    ids, raws, paths = sorted(spec["windows"]), [], {}
    with torch.no_grad():
        for i, wid in enumerate(ids, 1):
            imgs = []
            for vid, fi, gamma in spec["windows"][wid]:
                if vid not in paths:
                    paths[vid] = frame_paths(root / vid)
                with Image.open(paths[vid][min(int(fi), len(paths[vid]) - 1)]) as im:
                    imgs.append(preprocess(apply_gamma(im.convert("RGB"), float(gamma))))
            x = torch.stack(imgs).unsqueeze(0).cuda()   # (1, T, C, H, W)
            raws.append(torch.softmax(model(x).float(), dim=-1)[0].cpu().numpy())
            if i % 1000 == 0:
                print(f"  {i}/{len(ids)}", flush=True)
    raw = np.asarray(raws)
    np.savez_compressed(out / "probe.npz", ids=np.array(ids), raw=raw.astype(np.float32), prob=(raw @ fam).astype(np.float32))
    print(f"wrote {out / 'probe.npz'} ({len(ids)} windows)")


if __name__ == "__main__":
    main()
