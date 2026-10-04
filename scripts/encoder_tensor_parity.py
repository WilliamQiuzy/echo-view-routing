#!/usr/bin/env python
"""Encoder parity on identical input tensors (geometry pretraining, start condition 4; tolerance 1e-4 max abs).

1. make (main environment): uint8 crops of real EV9V frames through joint_visual.load_frame_u8, both gamma variants,
   saved with their clip, frame index and variant. Clips: the first 8 of the selection half, at most 16 frames each.
2. old (envs/echoviewclip, torch 1.13): the original stage-1 model (scripts/extract_echoviewclip_features.build_model,
   models_frozen/echoviewclip_stage1/v1/best.pth) embeds exactly those tensors, normalised as its test preprocessing.
   Run twice: strict fp32 (TF32 off) and torch's defaults, the setting of the original extraction.
3. new (main environment): the exported encoder (VisualEncoder, strict fp32) embeds the same tensors. It compares
   new with old, and every result with the float32 raw E1 cache rows of the same frames.
Writes <dir>/parity.json.
4. topdiff / oldcrops / diagnose: the 24 frames where the full pipeline differs most from the raw cache, cropped by
   each environment's Pillow and embedded by both models, to locate any difference above 1e-4. Writes diagnosis.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MEAN = np.array([123.675, 116.28, 103.53], np.float32).reshape(1, 3, 1, 1)
STD = np.array([58.395, 57.12, 57.375], np.float32).reshape(1, 3, 1, 1)


def make(a) -> None:
    from echo_routing.temporal.joint_visual import load_frame_u8
    from scripts.extract_echoviewclip_features import GAMMA, jobs_from
    from scripts.run_windowed_echoviewclip import frame_paths
    proto = json.loads(Path(a.protocol).read_text()); sel = set(proto["validation"]["selection"]["clips"])
    jobs = [j for j in jobs_from(proto, Path(a.manifest), Path(a.v3_cache)) if j["video_id"] in sel][:8]
    u8, rows = [], []
    for j in jobs:
        fp = frame_paths(Path(j["frames_dir"]))
        for k, fi in enumerate(j["frame_idx"][:16]):
            for var, g in GAMMA.items():
                u8.append(load_frame_u8(fp[min(int(fi), len(fp) - 1)], g)); rows.append({"video_id": j["video_id"], "row": k, "variant": var})
    d = Path(a.dir); d.mkdir(parents=True, exist_ok=True)
    np.save(d / "u8.npy", np.stack(u8)); (d / "rows.json").write_text(json.dumps(rows)); print("tensors", len(rows))


def old(a) -> None:
    import torch
    from scripts.extract_echoviewclip_features import build_model
    u8 = np.load(Path(a.dir) / "u8.npy"); x = (u8.transpose(0, 3, 1, 2).astype(np.float32) - MEAN) / STD
    model, _ = build_model(Path(a.repo).resolve(), Path(a.config).resolve(), Path(a.ckpt), Path(a.dir) / "old_build")
    for name, tf32 in (("strict", False), ("default", None)):
        if tf32 is None:   # torch 1.13 defaults: matmul TF32 off, cuDNN TF32 on
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = True
        else:
            torch.backends.cuda.matmul.allow_tf32 = tf32; torch.backends.cudnn.allow_tf32 = tf32
        with torch.no_grad():
            emb = torch.cat([model.image_encoder(torch.from_numpy(x[i:i + 64]).cuda()).float().cpu() for i in range(0, len(x), 64)]).numpy()
        np.save(Path(a.dir) / f"emb_old_{name}.npy", emb); print(name, emb.shape, "torch", torch.__version__)


def new(a) -> None:
    import torch
    from scripts.extract_encoder_features import load_encoder, strict_fp32
    strict_fp32(); d = Path(a.dir); u8 = torch.from_numpy(np.load(d / "u8.npy")); device = torch.device("cuda")
    enc, info = load_encoder(Path(a.encoder), device)
    with torch.no_grad():
        emb = torch.cat([enc(u8[i:i + 64].to(device)).float().cpu() for i in range(0, len(u8), 64)]).numpy()
    np.save(d / "emb_new.npy", emb)
    rows = json.loads((d / "rows.json").read_text())
    cache = np.stack([np.load(Path(a.raw_cache) / "raw" / r["variant"] / f"{r['video_id']}.npz")["emb"][r["row"]] for r in rows])
    res = {"n_tensors": len(rows), "tolerance": 1e-4, "torch_main": torch.__version__}
    for name in ("strict", "default"):
        o = np.load(d / f"emb_old_{name}.npy")
        res[f"new_vs_old_{name}_max_abs"] = float(np.abs(emb - o).max())
        res[f"old_{name}_vs_raw_cache_max_abs"] = float(np.abs(o - cache).max())
    res["new_vs_raw_cache_max_abs"] = float(np.abs(emb - cache).max())
    res["pass_identical_tensors"] = res["new_vs_old_strict_max_abs"] <= 1e-4
    (d / "parity.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


def topdiff(a) -> None:
    """Main environment: the frames where the full pipeline differs most from the raw cache, with their crops."""
    from echo_routing.temporal.joint_visual import load_frame_u8
    from scripts.extract_echoviewclip_features import GAMMA, jobs_from
    from scripts.run_windowed_echoviewclip import frame_paths
    proto = json.loads(Path(a.protocol).read_text()); jobs = {j["video_id"]: j for j in jobs_from(proto, Path(a.manifest), Path(a.v3_cache))}
    found = []
    for var in GAMMA:
        for f in sorted((Path(a.new_raw) / "raw" / var).glob("*.npz")):
            d = np.abs(np.load(f)["emb"] - np.load(Path(a.raw_cache) / "raw" / var / f.name)["emb"]).max(1)
            i = int(d.argmax()); found.append((float(d[i]), f.stem, i, var))
    found.sort(reverse=True); top = found[:24]; u8, rows = [], []
    for diff, v, i, var in top:
        fp = frame_paths(Path(jobs[v]["frames_dir"])); path = fp[min(int(jobs[v]["frame_idx"][i]), len(fp) - 1)]
        u8.append(load_frame_u8(path, GAMMA[var])); rows.append({"video_id": v, "row": i, "variant": var, "path": str(path), "diff": diff})
    d = Path(a.dir); d.mkdir(parents=True, exist_ok=True); np.save(d / "top_u8_main.npy", np.stack(u8)); (d / "top_rows.json").write_text(json.dumps(rows))
    import PIL; print("main Pillow", PIL.__version__, "| largest per-frame differences:", [round(r["diff"], 6) for r in rows[:6]])


def oldcrops(a) -> None:
    """Old environment: the same frames cropped by its own Pillow, and the original model on both crop sets."""
    import PIL
    import torch
    from PIL import Image
    from scripts.extract_echoviewclip_features import build_model
    from scripts.run_windowed_echoviewclip import apply_gamma
    d = Path(a.dir); rows = json.loads((d / "top_rows.json").read_text()); u8_old = []
    for r in rows:
        im = apply_gamma(Image.open(r["path"]).convert("RGB"), 1.0 if r["variant"] == "orig" else 0.9)
        w, h = im.size; s = 256 / min(w, h); im = im.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BILINEAR)
        w, h = im.size; l, t = (w - 224) // 2, (h - 224) // 2; u8_old.append(np.asarray(im.crop((l, t, l + 224, t + 224)), np.uint8))
    np.save(d / "top_u8_old.npy", np.stack(u8_old))
    model, _ = build_model(Path(a.repo).resolve(), Path(a.config).resolve(), Path(a.ckpt), d / "old_build")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    for name in ("main", "old"):
        u8 = np.load(d / f"top_u8_{name}.npy"); x = (u8.transpose(0, 3, 1, 2).astype(np.float32) - MEAN) / STD
        with torch.no_grad():
            np.save(d / f"top_emb_oldmodel_{name}crops.npy", model.image_encoder(torch.from_numpy(x).cuda()).float().cpu().numpy())
    print("old Pillow", PIL.__version__)


def batchctx(a) -> None:
    """Old environment: the top frames re-embedded inside the batch of 256 they had in the original extraction
    (items in job order, both under torch's defaults), to test whether batch context explains the cache's values."""
    import torch
    from PIL import Image
    from scripts.extract_echoviewclip_features import GAMMA, build_model, jobs_from
    from scripts.run_windowed_echoviewclip import apply_gamma, frame_paths, preprocess
    d = Path(a.dir); rows = json.loads((d / "top_rows.json").read_text())[: a.top]
    proto = json.loads(Path(a.protocol).read_text()); jobs = jobs_from(proto, Path(a.manifest), Path(a.v3_cache))
    start, at = {}, 0
    for j in jobs:
        start[j["video_id"]] = at; at += len(j["frame_idx"])
    model, _ = build_model(Path(a.repo).resolve(), Path(a.config).resolve(), Path(a.ckpt), d / "old_build"); res = []
    for r in rows:
        flat = start[r["video_id"]] + r["row"]; b0 = (flat // 256) * 256; items = []
        for j in jobs:
            s0 = start[j["video_id"]]
            if s0 + len(j["frame_idx"]) <= b0 or s0 >= b0 + 256:
                continue
            fp = frame_paths(Path(j["frames_dir"]))
            for k, fi in enumerate(j["frame_idx"]):
                if b0 <= s0 + k < b0 + 256:
                    items.append(fp[min(int(fi), len(fp) - 1)])
        x = torch.stack([preprocess(apply_gamma(Image.open(p).convert("RGB"), GAMMA[r["variant"]])) for p in items])
        with torch.no_grad():
            e_batch = model.image_encoder(x.cuda()).float().cpu().numpy()[flat - b0]
            e_alone = model.image_encoder(x[flat - b0: flat - b0 + 1].cuda()).float().cpu().numpy()[0]
        cache = np.load(Path(a.raw_cache) / "raw" / r["variant"] / f"{r['video_id']}.npz")["emb"][r["row"]]
        res.append({"video_id": r["video_id"], "variant": r["variant"], "batch_size": len(items),
                    "in_original_batch_vs_cache": float(np.abs(e_batch - cache).max()), "alone_vs_cache": float(np.abs(e_alone - cache).max())})
    (d / "batch_context.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


def diagnose(a) -> None:
    """Main environment: every combination of crops (main or old Pillow) and model, against the raw cache."""
    import torch
    from scripts.extract_encoder_features import load_encoder, strict_fp32
    strict_fp32(); d = Path(a.dir); rows = json.loads((d / "top_rows.json").read_text()); device = torch.device("cuda")
    enc, _ = load_encoder(Path(a.encoder), device)
    cache = np.stack([np.load(Path(a.raw_cache) / "raw" / r["variant"] / f"{r['video_id']}.npz")["emb"][r["row"]] for r in rows])
    m_main, m_old = np.load(d / "top_u8_main.npy"), np.load(d / "top_u8_old.npy")
    res = {"frames": len(rows), "crops_identical_frames": int(sum(np.array_equal(x, y) for x, y in zip(m_main, m_old))),
           "max_abs_crop_diff_grey": int(np.abs(m_main.astype(int) - m_old.astype(int)).max())}
    with torch.no_grad():
        for name, u8 in (("main", m_main), ("old", m_old)):
            e = enc(torch.from_numpy(u8).to(device)).float().cpu().numpy()
            res[f"newmodel_{name}crops_vs_cache"] = float(np.abs(e - cache).max())
            res[f"oldmodel_{name}crops_vs_cache"] = float(np.abs(np.load(d / f"top_emb_oldmodel_{name}crops.npy") - cache).max())
            res[f"newmodel_vs_oldmodel_{name}crops"] = float(np.abs(e - np.load(d / f"top_emb_oldmodel_{name}crops.npy")).max())
    (d / "diagnosis.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["make", "old", "new", "topdiff", "oldcrops", "diagnose", "batchctx"]); ap.add_argument("--dir", required=True)
    ap.add_argument("--protocol"); ap.add_argument("--manifest"); ap.add_argument("--v3-cache")
    ap.add_argument("--repo", default="third_party/EchoViewCLIP"); ap.add_argument("--config"); ap.add_argument("--ckpt")
    ap.add_argument("--encoder"); ap.add_argument("--raw-cache"); ap.add_argument("--new-raw"); ap.add_argument("--top", type=int, default=4)
    a = ap.parse_args(); {"make": make, "old": old, "new": new, "topdiff": topdiff, "oldcrops": oldcrops, "diagnose": diagnose, "batchctx": batchctx}[a.mode](a)


if __name__ == "__main__":
    main()
