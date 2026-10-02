#!/usr/bin/env python
"""E1 visual input: per-frame embeddings from the frozen EchoViewCLIP stage-1 image encoder (round 2, Task 3).

STANDALONE: runs under envs/echoviewclip (torch 1.13), like scripts/run_windowed_echoviewclip.py, whose preprocessing
(resize short side 256, centre crop 224, mean/std on 0-255) and gamma edit are reused unchanged.

- Model: the authors' factory (trainers.vificlip.returnCLIP, pinned commit) with the frozen checkpoint
  models_frozen/echoviewclip_stage1/v1/best.pth. A missing or unexpected visual key stops the run; only the three
  cached text-prompt tensors are excluded, as in the B7 runner.
- Layer: model.image_encoder(image) = CLIP VisionTransformer, ln_post(class token) @ proj, 512-d, float32. It runs per
  image, before ViFiCLIP's temporal mean/attention pooling and its outer projector; this is checked on real frames.
- Frames: the 10 Hz frame indices are read from the round-1 v3 cache (orig and gamma090 must agree), so E1 and E0 are
  aligned sample by sample. Clips: the protocol's train clips and both validation halves; the test split is not read.
Writes <out>/raw/<variant>/<video_id>.npz (frame_idx int32, emb float32 [n, 512]) and <out>/extract_meta.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_windowed_echoviewclip import apply_gamma, frame_paths, preprocess  # noqa: E402

PINNED = "f6a0d86074d7c906ae41546cd5f09f8b94da432a"
GAMMA = {"orig": 1.0, "gamma090": 0.9}
TEXT_CACHE_KEYS = ("prompt_learner.token_prefix", "prompt_learner.token_suffix", "prompt_learner.complete_text_embeddings")
LAYER = "model.image_encoder: CLIP VisionTransformer, ln_post(class token) @ proj (before temporal pooling and model.projector)"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def build_model(repo: Path, config_path: Path, ckpt: Path, out: Path):
    sys.path.insert(0, str(repo))
    import pandas as pd  # noqa: E402
    from utils.config import get_config  # noqa: E402  (official)
    from trainers import vificlip  # noqa: E402  (official)
    ns = argparse.Namespace(config=str(config_path), opts=None, batch_size=None, pretrained=None, resume=None, accumulation_steps=None,
                            output=str(out), only_test=True, local_rank=0)
    config = get_config(ns)
    logger = logging.getLogger("evc"); logger.addHandler(logging.StreamHandler()); logger.setLevel(logging.WARNING)
    class_names = [n for _, n in pd.read_csv(config.DATA.LABEL_LIST).values.tolist()]
    model = vificlip.returnCLIP(config, logger=logger, class_names=class_names)
    sd = torch.load(ckpt, map_location="cpu")["model"]
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    excluded = [k for k in TEXT_CACHE_KEYS if sd.pop(k, None) is not None]
    res = model.load_state_dict(sd, strict=False)
    missing, unexpected = list(res.missing_keys), list(res.unexpected_keys)
    bad = [k for k in missing + unexpected if k.startswith("image_encoder.")]
    if bad:
        raise SystemExit(f"visual weights missing or unexpected, refusing a partly random encoder: {bad[:5]}")
    model = model.float().cuda().eval()
    for mod in model.modules():  # the CLIP modules keep a dtype attribute; run everything in float32
        if isinstance(getattr(mod, "dtype", None), torch.dtype):
            mod.dtype = torch.float32
    return model, {"missing_keys": missing, "unexpected_keys": unexpected, "excluded_text_cache_keys": excluded,
                   "n_image_encoder_params": int(sum(p.numel() for n, p in model.named_parameters() if n.startswith("image_encoder.")))}


class Frames(Dataset):
    def __init__(self, items: list[tuple[int, Path, float]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        j, path, gamma = self.items[i]
        with Image.open(path) as im:
            return preprocess(apply_gamma(im.convert("RGB"), gamma)), j


def jobs_from(protocol: dict, manifest_csv: Path, v3_cache: Path) -> list[dict]:
    import pandas as pd  # noqa: E402
    man = pd.read_csv(manifest_csv, dtype={"video_id": str}).set_index("video_id")
    clips = list(protocol["train"]["clips"]) + [c for h in ("selection", "calibration") for c in protocol["validation"][h]["clips"]]
    if len(set(clips)) != len(clips):
        raise SystemExit("protocol clip lists overlap")
    jobs = []
    for v in clips:
        if man.loc[v, "split"] not in ("train", "validation"):
            raise SystemExit(f"{v} is in split {man.loc[v, 'split']}; only train and validation may be read")
        idx = {var: np.load(v3_cache / var / f"{v}.npz")["frame_idx"] for var in GAMMA}
        if not np.array_equal(idx["orig"], idx["gamma090"]):
            raise SystemExit(f"{v}: frame indices differ between variants in the v3 cache")
        jobs.append({"video_id": v, "frames_dir": str(man.loc[v, "frames_dir"]), "frame_idx": idx["orig"].astype(int)})
    return jobs


@torch.no_grad()
def embed(model, x: torch.Tensor) -> torch.Tensor:
    return model.image_encoder(x.cuda(non_blocking=True)).float().cpu()


@torch.no_grad()
def single_image_check(model, paths: list[Path]) -> dict:
    """The same frame must embed identically alone and inside batches with other frames."""
    xs = [preprocess(apply_gamma(Image.open(p).convert("RGB"), 1.0)) for p in paths[:3]]
    alone = embed(model, xs[0][None])[0]
    in_batch = embed(model, torch.stack(xs))[0]
    shuffled = embed(model, torch.stack([xs[2], xs[1], xs[0]]))[2]
    return {"max_abs_diff_batch": float((alone - in_batch).abs().max()), "max_abs_diff_reordered": float((alone - shuffled).abs().max()),
            "embedding_norm": float(alone.norm()), "dim": int(alone.numel())}


def run_variant(model, jobs: list[dict], variant: str, out: Path, batch_size: int, workers: int) -> int:
    d = out / "raw" / variant; d.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not (d / f"{j['video_id']}.npz").is_file()]
    items, paths_cache = [], {}
    for k, j in enumerate(todo):
        paths = frame_paths(Path(j["frames_dir"]))
        items += [(k, paths[min(int(fi), len(paths) - 1)], GAMMA[variant]) for fi in j["frame_idx"]]
    dl = DataLoader(Frames(items), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    buf: dict[int, list] = {}; done = 0
    for x, ks in dl:
        emb = embed(model, x)
        for e, k in zip(emb, ks.tolist()):
            buf.setdefault(k, []).append(e.numpy())
            if len(buf[k]) == len(todo[k]["frame_idx"]):
                j = todo[k]; tmp = d / f".{j['video_id']}.{os.getpid()}.tmp.npz"
                with tmp.open("wb") as fh:
                    np.savez(fh, frame_idx=np.asarray(j["frame_idx"], np.int32), emb=np.stack(buf.pop(k)).astype(np.float32))
                os.replace(tmp, d / f"{j['video_id']}.npz"); done += 1
                if done % 500 == 0:
                    print(f"  [{variant}] {done}/{len(todo)}", flush=True)
    if buf:
        raise RuntimeError(f"incomplete clips: {len(buf)}")
    return len(todo)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--v3-cache", required=True, help="cache_v2/features/<v3 key> (holds orig/ and gamma090/)")
    ap.add_argument("--ckpt", required=True); ap.add_argument("--expect-sha256", required=True)
    ap.add_argument("--config", required=True); ap.add_argument("--repo", default="third_party/EchoViewCLIP")
    ap.add_argument("--out", required=True); ap.add_argument("--variants", default="orig,gamma090")
    ap.add_argument("--batch-size", type=int, default=256); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="first N clips only (smoke)")
    ap.add_argument("--export-visual", default=None, help="save the loaded image encoder (float32 tensors only) here and stop")
    a = ap.parse_args()
    repo = Path(a.repo).resolve(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if head != PINNED:
        raise SystemExit(f"EchoViewCLIP commit {head} != pinned {PINNED}")
    ckpt_sha = sha256_file(Path(a.ckpt))
    if ckpt_sha != a.expect_sha256:
        raise SystemExit(f"checkpoint sha256 {ckpt_sha} != expected {a.expect_sha256}")
    model, load = build_model(repo, Path(a.config).resolve(), Path(a.ckpt), out)
    if a.export_visual:   # for joint fine-tuning in the main environment (round 2, Task 5b)
        enc = model.image_encoder
        arch = {"input_resolution": int(enc.input_resolution), "patch_size": int(enc.conv1.kernel_size[0]),
                "width": int(enc.conv1.out_channels), "layers": int(len(enc.transformer.resblocks)),
                "heads": int(enc.transformer.resblocks[0].attn.num_heads), "output_dim": int(enc.output_dim),
                "design_details": {"trainer": "ViFi_CLIP", "vision_depth": 0, "language_depth": 0, "vision_ctx": 0, "language_ctx": 0}}
        dest = Path(a.export_visual); dest.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": {k: v.detach().float().cpu().contiguous() for k, v in enc.state_dict().items()}, "arch": arch,
                    "source_checkpoint_sha256": ckpt_sha, "commit": head, "layer": LAYER}, dest)
        probe_frames = frame_paths(Path(jobs_from(json.loads(Path(a.protocol).read_bytes()), Path(a.manifest), Path(a.v3_cache))[0]["frames_dir"]))[:4]
        x = torch.stack([preprocess(apply_gamma(Image.open(f).convert("RGB"), 1.0)) for f in probe_frames])
        meta = {"export": str(dest), "sha256": sha256_file(dest), "arch": arch, "source_checkpoint_sha256": ckpt_sha, "commit": head,
                "reference_frames": [str(f) for f in probe_frames], "reference_embeddings": embed(model, x).tolist()}
        dest.with_suffix(".json").write_text(json.dumps(meta, indent=1))
        print(json.dumps({k: v for k, v in meta.items() if k != "reference_embeddings"}, indent=1)); return
    protocol_raw = Path(a.protocol).read_bytes(); protocol = json.loads(protocol_raw)
    jobs = jobs_from(protocol, Path(a.manifest), Path(a.v3_cache))
    if a.limit:
        jobs = jobs[: a.limit]
    check = single_image_check(model, frame_paths(Path(jobs[0]["frames_dir"])))
    if max(check["max_abs_diff_batch"], check["max_abs_diff_reordered"]) > 1e-3:
        raise SystemExit(f"embedding depends on other frames in the batch: {check}")
    print(json.dumps({"load": {k: (v if k != "missing_keys" else v[:8]) for k, v in load.items()}, "single_image_check": check}), flush=True)
    counts = {}
    for var in a.variants.split(","):
        counts[var] = run_variant(model, jobs, var, out, a.batch_size, a.workers)
        print(f"variant {var}: {counts[var]} clips extracted", flush=True)
    # online check: re-embed the shortest selection clip frame by frame and compare with what was written
    sel = [j for j in jobs if j["video_id"] in set(protocol["validation"]["selection"]["clips"])]
    j = min(sel or jobs, key=lambda r: len(r["frame_idx"])); paths = frame_paths(Path(j["frames_dir"])); online = {}
    for var in a.variants.split(","):
        saved = np.load(out / "raw" / var / f"{j['video_id']}.npz")
        emb = np.stack([embed(model, preprocess(apply_gamma(Image.open(paths[min(int(fi), len(paths) - 1)]).convert("RGB"), GAMMA[var]))[None])[0].numpy()
                        for fi in j["frame_idx"]])
        online[var] = {"video_id": j["video_id"], "n": int(len(emb)), "frame_idx_equal": bool(np.array_equal(saved["frame_idx"], j["frame_idx"])),
                       "max_abs_diff": float(np.abs(saved["emb"] - emb).max())}
    meta = {"repo": "github.com/xmed-lab/EchoViewCLIP", "commit": head, "checkpoint": str(a.ckpt), "checkpoint_sha256": ckpt_sha,
            "config": str(a.config), "layer": LAYER, "dim": check["dim"], "dtype": "float32",
            "preprocessing": "PIL RGB; gamma LUT 255*(i/255)^g (orig g=1, gamma090 g=0.9); resize short side to 256 (bilinear); "
                             "centre crop 224; (x - [123.675,116.28,103.53]) / [58.395,57.12,57.375]; float32 inference",
            "load_state_dict": load, "single_image_check": check, "online_check": online, "variants": list(counts),
            "n_clips": len(jobs), "clips_extracted_this_run": counts, "protocol": str(a.protocol),
            "protocol_sha256": hashlib.sha256(protocol_raw).hexdigest(), "frame_index_source": str(a.v3_cache),
            "splits_read": ["train", "validation"]}
    (out / "extract_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({"online_check": online, "out": str(out)}, indent=1))


if __name__ == "__main__":
    main()
