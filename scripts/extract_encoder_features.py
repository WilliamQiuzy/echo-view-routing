#!/usr/bin/env python
"""Per-frame embeddings from an exported image encoder in the main environment (geometry pretraining, Task 5).

For encoders saved in the export format of scripts/extract_echoviewclip_features.py --export-visual ({state_dict,
arch, source_checkpoint_sha256, ...}): the E1 export (G0) and the geometry-pretrained encoders (G1, G2). The frames,
preprocessing and output layout are those of the E1 extraction:
- frame indices from the round-1 v3 cache;
- PIL RGB, gamma LUT, resize of the short side to 256 (bilinear) and centre crop 224 on uint8
  (joint_visual.load_frame_u8, the same geometry as run_windowed_echoviewclip.preprocess);
- normalisation on the GPU, fp32 with TF32 off.
Only train and validation clips are read. Writes <out>/raw/<variant>/<video_id>.npz (frame_idx int32, emb float32) and
<out>/extract_meta.json with the keys scripts/train_feature_probe.py reads.
  .venv/bin/python scripts/extract_encoder_features.py --encoder models_frozen/e1_echoviewclip_probe/v1/visual_encoder_fp32.pt \
      --protocol P --manifest data/manifests/ev9v_native.csv --v3-cache cache_v2/features/beb36d80df31 --out cache_v2/geo_raw/<name>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from echo_routing.config import gpu_memory_fraction  # noqa: E402
from echo_routing.temporal.joint_visual import VisualEncoder, load_frame_u8, load_vision_transformer  # noqa: E402
from scripts.extract_echoviewclip_features import GAMMA, LAYER, jobs_from  # noqa: E402
from scripts.run_windowed_echoviewclip import frame_paths  # noqa: E402

PREPROCESSING = ("PIL RGB; gamma LUT 255*(i/255)^g (orig g=1, gamma090 g=0.9); resize short side to 256 (bilinear); "
                 "centre crop 224; (x - [123.675,116.28,103.53]) / [58.395,57.12,57.375]; float32 inference")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def strict_fp32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def load_encoder(export: Path, device) -> tuple[VisualEncoder, dict]:
    vt, info = load_vision_transformer(ROOT / "third_party" / "EchoViewCLIP", export)
    enc = VisualEncoder(vt, trainable_blocks=1).to(device).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc, info


class Frames(Dataset):
    def __init__(self, items: list[tuple[int, Path, float]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        k, path, gamma = self.items[i]
        return torch.from_numpy(load_frame_u8(path, gamma)), k


@torch.no_grad()
def embed(enc: VisualEncoder, u8: torch.Tensor, device) -> torch.Tensor:
    return enc(u8.to(device, non_blocking=True)).float().cpu()


def run_variant(enc, jobs: list[dict], variant: str, out: Path, batch: int, workers: int, device) -> int:
    d = out / "raw" / variant; d.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not (d / f"{j['video_id']}.npz").is_file()]
    items = []
    for k, j in enumerate(todo):
        paths = frame_paths(Path(j["frames_dir"]))
        items += [(k, paths[min(int(fi), len(paths) - 1)], GAMMA[variant]) for fi in j["frame_idx"]]
    dl = DataLoader(Frames(items), batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    buf: dict[int, list] = {}; done = 0
    for u8, ks in dl:
        for e, k in zip(embed(enc, u8, device), ks.tolist()):
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


@torch.no_grad()
def checks(enc, jobs: list[dict], protocol: dict, out: Path, variants: list[str], device) -> dict:
    """Alone vs in a batch, and the shortest selection clip re-embedded frame by frame against what was written."""
    paths = frame_paths(Path(jobs[0]["frames_dir"]))[:3]
    xs = torch.from_numpy(np.stack([load_frame_u8(p, 1.0) for p in paths]))
    alone = embed(enc, xs[:1], device)[0]; batch = embed(enc, xs, device)[0]; rev = embed(enc, xs.flip(0), device)[2]
    single = {"max_abs_diff_batch": float((alone - batch).abs().max()), "max_abs_diff_reordered": float((alone - rev).abs().max()),
              "embedding_norm": float(alone.norm()), "dim": int(alone.numel())}
    sel = [j for j in jobs if j["video_id"] in set(protocol["validation"]["selection"]["clips"])]
    j = min(sel or jobs, key=lambda r: len(r["frame_idx"])); fp = frame_paths(Path(j["frames_dir"])); online = {}
    for var in variants:
        saved = np.load(out / "raw" / var / f"{j['video_id']}.npz")
        emb = np.stack([embed(enc, torch.from_numpy(load_frame_u8(fp[min(int(fi), len(fp) - 1)], GAMMA[var]))[None], device)[0].numpy()
                        for fi in j["frame_idx"]])
        online[var] = {"video_id": j["video_id"], "n": int(len(emb)), "frame_idx_equal": bool(np.array_equal(saved["frame_idx"], j["frame_idx"])),
                       "max_abs_diff": float(np.abs(saved["emb"] - emb).max())}
    return {"single_image_check": single, "online_check": online}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoder", required=True, help="export blob (.pt) with state_dict, arch and source_checkpoint_sha256")
    ap.add_argument("--protocol", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--v3-cache", required=True, help="cache_v2/features/<v3 key> (holds orig/ and gamma090/)")
    ap.add_argument("--out", required=True); ap.add_argument("--variants", default="orig,gamma090")
    ap.add_argument("--batch-size", type=int, default=256); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="first N clips only (smoke)")
    a = ap.parse_args(); strict_fp32()
    device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    enc, info = load_encoder(Path(a.encoder), device)
    protocol_raw = Path(a.protocol).read_bytes(); protocol = json.loads(protocol_raw)
    jobs = jobs_from(protocol, Path(a.manifest), Path(a.v3_cache))
    if a.limit:
        jobs = jobs[: a.limit]
    variants = a.variants.split(","); counts = {}
    for var in variants:
        counts[var] = run_variant(enc, jobs, var, out, a.batch_size, a.workers, device)
        print(f"variant {var}: {counts[var]} clips extracted", flush=True)
    chk = checks(enc, jobs, protocol, out, variants, device)
    if max(chk["single_image_check"]["max_abs_diff_batch"], chk["single_image_check"]["max_abs_diff_reordered"]) > 1e-4:
        raise SystemExit(f"embedding depends on other frames in the batch: {chk['single_image_check']}")
    meta = {"extractor": "scripts/extract_encoder_features.py (main environment)", "commit": info["commit"], "checkpoint": str(a.encoder),
            "checkpoint_sha256": sha256_file(Path(a.encoder)), "source_checkpoint_sha256": info["source_checkpoint_sha256"], "layer": LAYER,
            "dim": chk["single_image_check"]["dim"], "dtype": "float32", "tf32": False, "preprocessing": PREPROCESSING, **chk,
            "variants": variants, "n_clips": len(jobs), "clips_extracted_this_run": counts, "protocol": str(a.protocol),
            "protocol_sha256": hashlib.sha256(protocol_raw).hexdigest(), "frame_index_source": str(a.v3_cache), "splits_read": ["train", "validation"],
            "torch": torch.__version__}
    (out / "extract_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("checkpoint_sha256", "single_image_check", "online_check")}, indent=1))


if __name__ == "__main__":
    main()
