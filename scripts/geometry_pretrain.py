#!/usr/bin/env python
"""Geometry pretraining of the E1 encoder (configs/experiments/geometry_pretraining_v1.yaml): arm G1 (same-plane
temporal consistency, InfoNCE only) or G2 (the same plus regression of the tilt difference).

- Encoder: the E1 export through VisualEncoder; resblocks.11, ln_post and proj train, the prefix stays frozen. fp32,
  TF32 off.
- Data: one pair bank per seed (scripts/geometry_bank.py), read in index order: step s takes pairs 32s to 32s + 31.
  The two arms of a seed therefore see bit-identical batches; every batch's SHA-256 prefix is logged to compare.
- Loss: InfoNCE (echo_routing.geometry.losses) + lambda * tilt loss, lambda 0 for G1 and 1 for G2. The heads are created
  in the same order with the same seed in both arms.
- Optimiser: AdamW (encoder 1e-5, heads 1e-4, weight decay 1e-4), linear warm-up, cosine decay, gradient clip 1.0,
  fixed number of steps, final weights.
- Output: <out>/visual_encoder_fp32.pt in E1's export format (scripts/extract_encoder_features.py reads it),
  heads.pt, train_log.jsonl, batch_digests.txt and mechanism.json (held-out people: tilt error against the zero
  predictor, embedding change between times at one plane and between tilts at one time).
--check runs one batch and reports the gradients that each term alone sends to every trainable tensor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from echo_routing.config import gpu_memory_fraction  # noqa: E402
from echo_routing.geometry import losses  # noqa: E402
from echo_routing.temporal.joint_visual import VisualEncoder, load_vision_transformer  # noqa: E402
from scripts.extract_encoder_features import sha256_file, strict_fp32  # noqa: E402

EXPORT = "models_frozen/e1_echoviewclip_probe/v1/visual_encoder_fp32.pt"
CONFIG = "configs/experiments/geometry_pretraining_v1.yaml"
EXPECTED_TRAINABLE = ("vt.transformer.resblocks.11.", "vt.ln_post.", "vt.proj")


class Bank:
    def __init__(self, path: Path) -> None:
        self.path = Path(path); self.u8 = np.load(self.path / "u8.npy", mmap_mode="r")
        specs = pd.read_csv(self.path / "specs.csv")
        self.same_plane = torch.tensor((specs["kind"] == "same_plane").to_numpy())
        self.d_alpha = torch.tensor(specs["d_alpha"].to_numpy(), dtype=torch.float32)
        self.kind = specs["kind"].to_numpy(); self.meta = json.loads((self.path / "meta.json").read_text())

    def __len__(self) -> int:
        return len(self.u8)

    def batch(self, idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = np.ascontiguousarray(self.u8[idx]).reshape(-1, 224, 224)            # rows 2i, 2i+1 = slices of pair i
        x = np.repeat(x[..., None], 3, axis=-1)                                  # grey copied to RGB, as E1's frames
        return torch.from_numpy(x), self.same_plane[idx], self.d_alpha[idx]


def build(seed: int, device):
    torch.manual_seed(seed)
    vt, info = load_vision_transformer(ROOT / "third_party" / "EchoViewCLIP", ROOT / EXPORT)
    enc = VisualEncoder(vt, trainable_blocks=1).to(device)
    proj, head = losses.Projector().to(device), losses.TiltHead().to(device)    # same order and seed in both arms
    names = [n for n, p in enc.named_parameters() if p.requires_grad]
    if not names or not all(n.startswith(EXPECTED_TRAINABLE) for n in names):
        raise SystemExit(f"unexpected trainable encoder tensors: {names}")
    return enc, proj, head, info


def lr_factor(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def gradient_check(enc, proj, head, bank: Bank, tau: float, device) -> dict:
    """Each term alone, on the first batch: finite non-zero gradient in every trainable encoder tensor."""
    x, same, da = bank.batch(np.arange(32)); x, same, da = x.to(device), same.to(device), da.to(device); out = {}
    for term in ("tilt", "infonce"):
        for m in (enc, proj, head):
            m.zero_grad(set_to_none=True)
        z = enc(x)
        loss = losses.tilt_loss(head, z, da) if term == "tilt" else losses.info_nce(proj(z), losses.partners(same), tau)
        loss.backward()
        g = {n: (None if p.grad is None else float(p.grad.norm())) for n, p in enc.named_parameters() if p.requires_grad}
        frozen_with_grad = [n for n, p in enc.named_parameters() if not p.requires_grad and p.grad is not None]
        out[term] = {"loss": float(loss.detach()), "trainable_tensors": len(g),
                     "all_finite_nonzero": all(v is not None and math.isfinite(v) and v > 0 for v in g.values()),
                     "min_grad_norm": min(v for v in g.values() if v is not None), "frozen_with_grad": frozen_with_grad}
    return out


@torch.no_grad()
def mechanism(enc, head, bank: Bank, device, batch: int = 64) -> dict:
    """Held-out people: tilt head error against the zero predictor, and embedding change by pair kind."""
    enc.eval(); preds, dists = [], []
    for s in range(0, len(bank), batch):
        idx = np.arange(s, min(len(bank), s + batch)); x, _, _ = bank.batch(idx); z = enc(x.to(device))
        preds.append(head(z[0::2], z[1::2]).cpu() * losses.D_ALPHA_SCALE)
        dists.append(1 - torch.nn.functional.cosine_similarity(z[0::2], z[1::2]).cpu())
    pred, dist, da = torch.cat(preds).numpy(), torch.cat(dists).numpy(), bank.d_alpha.numpy()
    by_kind = {k: {"pairs": int((bank.kind == k).sum()), "mean_cosine_distance": float(dist[bank.kind == k].mean())} for k in np.unique(bank.kind)}
    return {"tilt_mae_deg": float(np.abs(pred - da).mean()), "zero_predictor_mae_deg": float(np.abs(da).mean()),
            "tilt_mae_deg_moving_plane": float(np.abs(pred - da)[bank.kind != "same_plane"].mean()),
            "embedding_change_by_kind": by_kind, "pairs": int(len(bank))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["G1", "G2"], required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--bank", required=True); ap.add_argument("--heldout", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=CONFIG); ap.add_argument("--steps", type=int, default=None, help="override (smoke runs only)")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(); strict_fp32(); cfg = yaml.safe_load(Path(a.config).read_text()); mc = cfg["model"]
    device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    bank, held = Bank(Path(a.bank)), Bank(Path(a.heldout)); tau = float(mc["infonce"]["temperature"])
    steps = int(a.steps or mc["budget"]["steps"]); per = int(mc["budget"]["pairs_per_step"]); lam = 0.0 if a.arm == "G1" else 1.0
    if len(bank) < steps * per:
        raise SystemExit(f"bank holds {len(bank)} pairs, {steps * per} needed")
    enc, proj, head, info = build(a.seed, device); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.check:
        res = gradient_check(enc, proj, head, bank, tau, device); (out / "gradient_check.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1)); return
    oc = mc["optimiser"]
    opt = torch.optim.AdamW([{"params": [p for p in enc.parameters() if p.requires_grad], "lr": float(oc["lr_encoder"])},
                             {"params": list(proj.parameters()) + list(head.parameters()), "lr": float(oc["lr_heads"])}],
                            weight_decay=float(oc["weight_decay"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, int(oc["warmup_steps"]), steps))
    params = [p for g in opt.param_groups for p in g["params"]]; log = (out / "train_log.jsonl").open("w"); digests = []
    enc.train(); t0 = time.time()
    for s in range(steps):
        x, same, da = bank.batch(np.arange(s * per, (s + 1) * per))
        digests.append(hashlib.sha256(x.numpy().tobytes()).hexdigest()[:16])
        x, same, da = x.to(device, non_blocking=True), same.to(device), da.to(device)
        z = enc(x); l_nce = losses.info_nce(proj(z), losses.partners(same), tau)
        if lam == 0.0:                       # G1: the tilt term stays out of the graph; it is only logged
            with torch.no_grad():
                l_tilt = losses.tilt_loss(head, z, da)
            loss = l_nce
        else:
            l_tilt = losses.tilt_loss(head, z, da); loss = l_nce + lam * l_tilt
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(params, float(oc["grad_clip"]))); opt.step(); sched.step()
        if s % 50 == 0 or s == steps - 1:
            rec = {"step": s, "loss": float(loss.detach()), "infonce": float(l_nce.detach()), "tilt": float(l_tilt.detach()), "grad_norm": gn,
                   "lr_encoder": opt.param_groups[0]["lr"], "same_plane_pairs": int(same.sum()), "seconds": round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
    log.close(); (out / "batch_digests.txt").write_text("\n".join(digests) + "\n")
    src = torch.load(ROOT / EXPORT, map_location="cpu", weights_only=True)
    blob = {"state_dict": {k: v.detach().float().cpu().contiguous() for k, v in enc.vt.state_dict().items()}, "arch": src["arch"],
            "source_checkpoint_sha256": src["source_checkpoint_sha256"], "commit": info["commit"], "layer": src.get("layer"),
            "geometry_pretraining": {"arm": a.arm, "seed": a.seed, "steps": steps, "lambda_geo": lam, "initial_export": EXPORT,
                                     "initial_export_sha256": sha256_file(ROOT / EXPORT), "bank": str(a.bank), "bank_sha256": bank.meta["sha256"],
                                     "config": a.config, "config_sha256": sha256_file(Path(a.config))}}
    torch.save(blob, out / "visual_encoder_fp32.pt")
    torch.save({"projector": proj.state_dict(), "tilt_head": head.state_dict()}, out / "heads.pt")
    peak_gb = torch.cuda.max_memory_allocated() / 2 ** 30
    mech = mechanism(enc, head, held, device); mech["note"] = "G1's tilt head is never trained (lambda 0)" if a.arm == "G1" else ""
    mech["peak_gpu_gb_training"] = round(peak_gb, 2)
    (out / "mechanism.json").write_text(json.dumps(mech, indent=1))
    print(json.dumps({"run_dir": str(out), "mechanism": mech, "minutes": round((time.time() - t0) / 60, 1)}, indent=1))


if __name__ == "__main__":
    main()
