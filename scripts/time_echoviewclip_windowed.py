#!/usr/bin/env python
"""Same-hardware timing of the official windowed EchoViewCLIP stage 1 (round 2, Task 6).

STANDALONE: runs under envs/echoviewclip. Uses the B7 runner's own model loading, preprocessing and windowing
(16 frames per 1.6 s window, centres every 5 samples) on the sequences chosen by scripts/time_systems.py, one sequence
at a time after warm-up, with CUDA synchronisation. Reports per sample: CPU preprocessing (frames decoded in one
process, as in the runner) and the model forward over all windows of the sequence.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_windowed_echoviewclip import (NUM_FRAMES, STRIDE, WINDOW, apply_gamma, frame_paths, preprocess,  # noqa: E402
                                               window_centres, window_samples)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plans", required=True); ap.add_argument("--chosen", required=True, help="timing JSON of time_systems.py")
    ap.add_argument("--ckpt", required=True); ap.add_argument("--config", required=True)
    ap.add_argument("--repo", default="third_party/EchoViewCLIP"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); repo = Path(a.repo).resolve(); sys.path.insert(0, str(repo))
    import pandas as pd  # noqa: E402
    from utils.config import get_config  # noqa: E402  (official)
    from trainers import vificlip  # noqa: E402  (official)
    ns = argparse.Namespace(config=str(Path(a.config).resolve()), opts=None, batch_size=None, pretrained=None, resume=None,
                            accumulation_steps=None, output="/tmp/evc_timing", only_test=True, local_rank=0)
    config = get_config(ns); logger = logging.getLogger("evc"); logger.setLevel(logging.WARNING)
    names = [n for _, n in pd.read_csv(config.DATA.LABEL_LIST).values.tolist()]
    model = vificlip.returnCLIP(config, logger=logger, class_names=names).cuda().eval()
    sd = torch.load(a.ckpt, map_location="cpu")["model"]; sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    for k in ["prompt_learner.token_prefix", "prompt_learner.token_suffix", "prompt_learner.complete_text_embeddings"]:
        sd.pop(k, None)
    model.load_state_dict(sd, strict=False); model = model.float()
    for mod in model.modules():
        if isinstance(getattr(mod, "dtype", None), torch.dtype):
            mod.dtype = torch.float32
    plans = json.loads(Path(a.plans).read_text()); chosen = json.loads(Path(a.chosen).read_text())["sequences"]
    root = Path(plans["frames_root"]); pc = {}

    def run(sid):
        refs = plans["streams"][sid]["refs"]; n = len(refs); t_pre = t_fwd = 0.0
        for c in window_centres(n, STRIDE):
            samp = window_samples(int(c), n, WINDOW); sel = np.linspace(0, len(samp) - 1, NUM_FRAMES).round().astype(int)
            t0 = time.perf_counter(); imgs = []
            for s in samp[sel]:
                vid, fi, g = refs[s]
                if vid not in pc:
                    pc[vid] = frame_paths(root / vid)
                with Image.open(pc[vid][min(fi, len(pc[vid]) - 1)]) as im:
                    imgs.append(preprocess(apply_gamma(im.convert("RGB"), g)))
            x = torch.stack(imgs).unsqueeze(0); t_pre += time.perf_counter() - t0
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.no_grad():
                torch.softmax(model(x.cuda()).float(), dim=-1).cpu()
            torch.cuda.synchronize(); t_fwd += time.perf_counter() - t0
        return n, t_pre, t_fwd

    first = next(iter(chosen.values()))[:2]
    for sid in first:
        run(sid)   # warm-up
    out = {"groups": {}, "device": torch.cuda.get_device_name(0), "note": "batch = 1 window of 16 frames; fp32; CPU preprocessing in one process"}
    for g, ids in chosen.items():
        tot_n = tot_pre = tot_fwd = 0
        for sid in ids:
            n, tp, tf = run(sid); tot_n += n; tot_pre += tp; tot_fwd += tf
        out["groups"][g] = {"n_sequences": len(ids), "n_samples": tot_n, "preprocess_ms_per_sample": 1000 * tot_pre / tot_n,
                            "model_ms_per_sample": 1000 * tot_fwd / tot_n, "system_ms_per_sample": 1000 * (tot_pre + tot_fwd) / tot_n}
        print(g, json.dumps(out["groups"][g]), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1)); print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
