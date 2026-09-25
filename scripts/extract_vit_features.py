#!/usr/bin/env python
"""Per-cine 10 Hz outputs of a frame-wise ViT encoder, written to the standard feature cache.

vit_frame : feat = CLS embedding (pre-logits), logit/prob = frame head.
vivit_fe  : feat = frame embedding fed to the temporal Transformer, logit/prob = auxiliary per-frame head.
Layout: cache/features/<ckpt_hash>/<variant>/<video_id>.npz + index.csv + meta.json (features/cache.py contract).
With --order-like <hash>, index rows follow that cache's index order, so the deterministic recipe banks built from
either cache (compose/banks.py draws from the index in row order) are identical.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import base_parser, load_cfg  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from echo_routing.config import gpu_memory_fraction, load_paths  # noqa: E402
from echo_routing.features.cache import VideoFeatures, cached_n_samples, read_index, save_video_features, variant_dir, write_index, write_meta  # noqa: E402
from echo_routing.features.encoder import checkpoint_hash  # noqa: E402
from echo_routing.features.extract import VARIANT_GAMMA  # noqa: E402
from echo_routing.features.sampling import stride_indices  # noqa: E402
from echo_routing.ingest.frames import list_frame_paths  # noqa: E402
from echo_routing.manifest import load_manifest, task_subset  # noqa: E402
from echo_routing.vit.gpu_frames import NORMS, GpuDecoder, read_bytes  # noqa: E402
from echo_routing.vit.models import load_vit_checkpoint  # noqa: E402

CHUNK = 512


class _Cines(Dataset):
    def __init__(self, rows, target_hz: float) -> None:
        self.rows, self.hz = rows, target_hz

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        idx = stride_indices(int(r.n_frames), float(r.fps_playback), self.hz)
        paths = list_frame_paths(Path(r.frames_dir))
        return r.video_id, idx, [read_bytes(paths[min(int(f), len(paths) - 1)]) for f in idx], float(r.fps_playback)


@torch.no_grad()
def encode(model, name: str, decoder: GpuDecoder, datas: list, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    feats, logits = [], []
    for s in range(0, len(datas), CHUNK):
        x = decoder(datas[s:s + CHUNK], [gamma] * len(datas[s:s + CHUNK]))
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
            if name == "vit_frame":
                lg, f = model.forward_with_features(x)
            else:
                f = model.frame_embed(x); lg = model.frame_head(f)
        feats.append(f.float().cpu().numpy()); logits.append(lg.float().cpu().numpy())
    return np.concatenate(feats), np.concatenate(logits)


def ordered(rows, order_like: Path | None):
    if order_like is None:
        return rows
    pos = {v: i for i, v in enumerate(read_index(order_like)["video_id"])}
    return sorted(rows, key=lambda r: (pos.get(r.video_id, len(pos)), r.video_id))


def check_order(index_dir: Path, order_like: Path) -> None:
    """Recipe banks draw from the index in row order: the written index must list the reference cines in the
    reference order (write_index upserts, so a second --splits batch would append rows out of order)."""
    ref = list(read_index(order_like)["video_id"]); got = list(read_index(index_dir)["video_id"])
    shared = set(ref) & set(got)
    if [v for v in got if v in shared] != [v for v in ref if v in shared]:
        raise SystemExit(f"index order of {index_dir} differs from {order_like}: extract all splits in one call")
    if set(ref) - set(got):
        print(f"warning: {len(set(ref) - set(got))} reference cines are not in {index_dir}; banks will differ", flush=True)


def main() -> None:
    ap = base_parser("Extract ViT frame features into the cache")
    ap.add_argument("--ckpt", required=True); ap.add_argument("--splits", default="train,validation,test")
    ap.add_argument("--variants", default="orig,gamma090"); ap.add_argument("--order-like", default=None, help="ckpt hash")
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--num-workers", type=int, default=6)
    args = ap.parse_args(); cfg = load_cfg(args); paths = load_paths()
    os.environ.setdefault("HF_HOME", str(paths.data_root / "hf_cache"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    model, spec, _ = load_vit_checkpoint(Path(args.ckpt)); model.eval().to(device)
    if spec.name not in ("vit_frame", "vivit_fe"):
        raise SystemExit(f"{spec.name} is not a frame-wise encoder")
    h = checkpoint_hash(Path(args.ckpt)); label_col = "family5_index" if spec.task == "family5" else "raw9_index"
    df = task_subset(load_manifest(paths.manifest_path), spec.task)
    df = df[df["frames_dir"].notna() & (df["n_frames"] > 0) & df["split"].isin(args.splits.split(","))]
    if args.limit:
        df = df.groupby("split", group_keys=False).head(args.limit)
    ref = variant_dir(paths.cache_root, args.order_like, "orig") if args.order_like else None
    rows = ordered(list(df.itertuples(index=False)), ref)
    decoder = GpuDecoder(spec.image_size, NORMS[spec.norm], device)
    for variant in args.variants.split(","):
        gamma = VARIANT_GAMMA[variant]; out = variant_dir(paths.cache_root, h, variant); out.mkdir(parents=True, exist_ok=True)
        todo = [r for r in rows if cached_n_samples(out / f"{r.video_id}.npz") is None]
        dl = DataLoader(_Cines(todo, cfg["sampling"]["target_hz"]), batch_size=None, num_workers=args.num_workers)
        for n, (vid, idx, datas, fps) in enumerate(dl, 1):
            feat, logit = encode(model, spec.name, decoder, datas, gamma)
            prob = torch.softmax(torch.from_numpy(logit), dim=1).numpy()
            save_video_features(out, VideoFeatures(vid, np.asarray(idx), (np.asarray(idx) / fps).astype(np.float32), feat, prob, logit))
            if n % 500 == 0:
                print(f"  [{variant}] {n}/{len(todo)} cines", flush=True)
        index_rows = []
        for r in rows:
            k = cached_n_samples(out / f"{r.video_id}.npz")
            if k is None:
                raise RuntimeError(f"missing cache file for {r.video_id} ({variant})")
            index_rows.append({"video_id": r.video_id, "split": r.split, "raw_label": r.raw_label,
                               "label_index": int(getattr(r, label_col)), "n_samples": k, "path": str(out / f"{r.video_id}.npz")})
        write_index(out, index_rows)
        if ref is not None:
            check_order(out, ref)   # banks are drawn from the orig index; every variant is written in the same order
        write_meta(out, {"ckpt_hash": h, "ckpt": str(args.ckpt), "variant": variant, "gamma": gamma, "model": spec.name,
                         "task": spec.task, "classes": list(spec.classes), "target_hz": cfg["sampling"]["target_hz"],
                         "image_size": spec.image_size, "order_like": args.order_like})
        print(f"variant={variant} -> {out} ({len(rows)} cines)", flush=True)
    print(f"ckpt_hash={h}")


if __name__ == "__main__":
    main()
