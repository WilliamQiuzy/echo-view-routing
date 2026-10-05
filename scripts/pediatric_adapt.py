#!/usr/bin/env python
"""Pediatric adaptation, arm D1 (frozen encoder, head only) or D2 (head plus the encoder's last block, ln_post and proj),
from E1 (configs/experiments/pediatric_adapt_v1.yaml, section training).

Every update holds 32 target frames (8 Pediatric calibration-fold clips x 4 samples; class, then group, then clip) and
32 source frames (8 EV9V training clips x 4 samples; class-balanced). The plan of all updates depends only on the seed,
so D1 and D2 read bit-identical batches; a digest per update is logged. Frames go through E1's preprocessing. The head
starts from E1's probe. Output in <out>: head.pt, visual_encoder_fp32.pt (D2, E1 export format), train_log.jsonl,
batch_digests.txt and checks.json. Pediatric data stay on the server.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from echo_routing.config import gpu_memory_fraction  # noqa: E402
from echo_routing.features.sampling import balanced_video_weights, spaced_indices  # noqa: E402
from echo_routing.realvideo.pediatric import VIDEO_ROOT, timestamp_indices  # noqa: E402
from echo_routing.temporal.joint_visual import crop_uint8, load_frame_u8  # noqa: E402
from scripts.extract_encoder_features import sha256_file, strict_fp32  # noqa: E402
from scripts.geometry_g0_reproduce import load_probe  # noqa: E402
from scripts.geometry_pretrain import EXPORT  # noqa: E402
from scripts.pediatric_extract import verify_protocol  # noqa: E402
from echo_routing.temporal.joint_visual import VisualEncoder, load_vision_transformer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/experiments/pediatric_adapt_v1.yaml"
PROTO_DIR = Path("data/manifests/pediatric/pediatric-external-v1")
E1_MANIFEST = "runs/20261002-142832-e1_probe-38bbcf7c-s0/feature_manifest.json"
E1_RAW = Path("cache_v2/e1_raw/a74c7b233e79/raw/orig")
EV9V_PROTOCOL = "runs/20261002-051140-model_development_protocol-fa45e7f4-s0/protocol.json"
CLIPS, PER_CLIP = 8, 4
TRAINABLE = ("vt.transformer.resblocks.11.", "vt.ln_post.", "vt.proj")


def samples(n: int, rng: np.random.Generator) -> np.ndarray:
    return spaced_indices(n, PER_CLIP, rng) if n >= PER_CLIP else rng.choice(n, PER_CLIP, replace=True)


def plan(seed: int, steps: int, target: pd.DataFrame, source: pd.DataFrame) -> list[dict]:
    """Every update's clips and sample positions, from the seed alone."""
    rng = np.random.default_rng([int(seed), 9])
    by_class = {c: {g: d["video_id"].tolist() for g, d in target[target["label"] == c].groupby("group")} for c in (1, 2)}
    w = balanced_video_weights(source["label"].tolist()); out = []
    for _ in range(steps):
        t = []
        for _ in range(CLIPS):
            c = int(rng.choice([1, 2])); g = rng.choice(sorted(by_class[c])); v = by_class[c][g][int(rng.integers(len(by_class[c][g])))]
            n = int(target.loc[v, "n_samples"]); t.append((v, c, samples(n, rng).tolist()))
        s = []
        for i in rng.choice(len(source), CLIPS, replace=True, p=w):
            r = source.iloc[int(i)]; s.append((r["video_id"], int(r["label"]), samples(int(r["n"]), rng).tolist()))
        out.append({"target": t, "source": s})
    return out


class Updates(Dataset):
    def __init__(self, plans: list[dict], target: pd.DataFrame, source: pd.DataFrame, frame_paths) -> None:
        self.plans, self.target, self.source, self.frame_paths = plans, target, source, frame_paths

    def __len__(self) -> int:
        return len(self.plans)

    def target_frames(self, v: str, pos: list[int]) -> list[np.ndarray]:
        r = self.target.loc[v]; idx = timestamp_indices(int(r["n_frames"]), float(r["fps"]), 10.0); want = {int(idx[p]) for p in pos}
        got, k = {}, 0; cap = cv2.VideoCapture(str(VIDEO_ROOT / r["view"] / "Videos" / r["file"]))
        while len(got) < len(want):
            ok, bgr = cap.read()
            if not ok:
                break
            if k in want:
                got[k] = crop_uint8(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            k += 1
        cap.release()
        if len(got) != len(want):
            raise RuntimeError(f"{v}: frames {sorted(want)} not all decoded")
        return [got[int(idx[p])] for p in pos]

    def __getitem__(self, i: int):
        u = self.plans[i]; frames, labels = [], []
        for v, c, pos in u["target"]:
            frames += self.target_frames(v, pos); labels += [c] * len(pos)
        for v, c, pos in u["source"]:
            fi = self.source.loc[v, "frame_idx"]; paths = self.frame_paths(v)
            frames += [load_frame_u8(paths[min(int(fi[p]), len(paths) - 1)], 1.0) for p in pos]; labels += [c] * len(pos)
        return torch.from_numpy(np.stack(frames)), torch.tensor(labels)


def lr_factor(step: int, warmup: int, total: int) -> float:
    return (step + 1) / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["D1", "D2"], required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(); strict_fp32(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    verify_protocol(PROTO_DIR)
    videos = pd.read_csv(PROTO_DIR / "videos.csv", dtype={"clip_no": str})
    target = videos[videos["role"] == "calibration"].set_index("video_id", drop=False)
    from echo_routing.config import load_paths
    from echo_routing.manifest import load_manifest
    paths = load_paths(); man = load_manifest(paths.manifest_path).set_index("video_id")
    proto = json.loads(Path(EV9V_PROTOCOL).read_text())
    from echo_routing.features.cache import read_index, variant_dir
    e1_key = json.loads(Path(E1_MANIFEST).read_text())["encoder"]["cache_key"]          # labels from E1's formal index
    idx = read_index(variant_dir(paths.cache_root, e1_key, "orig")).set_index("video_id")
    rows = []
    for v in proto["train"]["clips"]:
        fi = np.load(E1_RAW / f"{v}.npz")["frame_idx"]; rows.append({"video_id": v, "label": int(idx.loc[v, "label_index"]), "n": int(len(fi)), "frame_idx": fi})
    source = pd.DataFrame(rows).set_index("video_id", drop=False)
    from scripts.train_joint_visual_temporal import FramePaths  # noqa: E402  (frames in numeric order, as round 6)
    frame_paths = FramePaths({v: str(man.loc[v, "frames_dir"]) for v in source.index})
    plans = plan(a.seed, a.steps, target, source.reset_index(drop=True))
    device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    vt, info = load_vision_transformer(ROOT / "third_party" / "EchoViewCLIP", ROOT / EXPORT)
    enc = VisualEncoder(vt, trainable_blocks=1).to(device)
    if a.arm == "D1":
        for p in enc.parameters():
            p.requires_grad_(False)
    names = [n for n, p in enc.named_parameters() if p.requires_grad]
    if a.arm == "D2" and (not names or not all(n.startswith(TRAINABLE) for n in names)):
        raise SystemExit(f"unexpected trainable encoder tensors: {names}")
    start = {n: p.detach().clone() for n, p in enc.named_parameters()}
    head = load_probe(Path(json.loads(Path(E1_MANIFEST).read_text())["encoder"]["checkpoint"])).to(device)
    groups = [{"params": list(head.parameters()), "lr": 1e-3}]                         # config: head lr 1e-3
    if a.arm == "D2":
        groups.append({"params": [p for p in enc.parameters() if p.requires_grad], "lr": 1e-5})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4); base = [g["lr"] for g in opt.param_groups]
    params = [p for g in opt.param_groups for p in g["params"]]
    dl = DataLoader(Updates(plans, target, source, frame_paths), batch_size=None, shuffle=False, num_workers=a.workers,
                    prefetch_factor=2 if a.workers else None, pin_memory=True)
    log = (out / "train_log.jsonl").open("w"); digests = []; t0 = time.time()
    for s, (u8, y) in enumerate(dl):
        digests.append(hashlib.sha256(u8.numpy().tobytes() + y.numpy().tobytes()).hexdigest()[:16])
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * lr_factor(s, 100, a.steps)
        u8, y = u8.to(device, non_blocking=True), y.to(device)
        enc.train(a.arm == "D2"); head.train()
        if a.arm == "D1":
            with torch.no_grad():
                z = enc(u8)
        else:
            z = enc(u8)
        loss = F.cross_entropy(head(z), y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at update {s}")
        opt.zero_grad(set_to_none=True); loss.backward(); gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0)); opt.step()
        if s % 50 == 0 or s == a.steps - 1:
            with torch.no_grad():
                acc_t = float((head(z[:32]).argmax(1) == y[:32]).float().mean()); acc_s = float((head(z[32:]).argmax(1) == y[32:]).float().mean())
            rec = {"step": s, "loss": float(loss.detach()), "acc_target": acc_t, "acc_source": acc_s, "grad_norm": gn, "seconds": round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
    log.close(); (out / "batch_digests.txt").write_text("\n".join(digests) + "\n")
    changed = sorted(n for n, p in enc.named_parameters() if not torch.equal(p.detach(), start[n]))
    checks = {"arm": a.arm, "seed": a.seed, "steps": a.steps, "encoder_tensors_changed": changed,
              "ok": (not changed) if a.arm == "D1" else bool(changed) and all(n.startswith(TRAINABLE) for n in changed)}
    (out / "checks.json").write_text(json.dumps(checks, indent=1))
    if not checks["ok"]:
        raise SystemExit(f"encoder change check failed: {changed[:5]}")
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()}, "in_dim": 512, "classes": ["PLAX", "PSAX", "A4C", "A5C", "SC4C"],
                "arm": a.arm, "seed": a.seed, "config": CONFIG, "config_sha256": sha256_file(Path(CONFIG))}, out / "head.pt")
    if a.arm == "D2":
        src = torch.load(ROOT / EXPORT, map_location="cpu", weights_only=True)
        torch.save({"state_dict": {k: v.detach().float().cpu().contiguous() for k, v in enc.vt.state_dict().items()}, "arch": src["arch"],
                    "source_checkpoint_sha256": src["source_checkpoint_sha256"], "commit": info["commit"], "layer": src.get("layer"),
                    "pediatric_adapt": {"arm": a.arm, "seed": a.seed, "steps": a.steps, "config_sha256": sha256_file(Path(CONFIG))}},
                   out / "visual_encoder_fp32.pt")
    print(json.dumps({"out": str(out), "minutes": round((time.time() - t0) / 60, 1), "checks": checks["ok"]}))


if __name__ == "__main__":
    main()
