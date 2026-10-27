#!/usr/bin/env python
"""A42 check attempt 1 diagnosis (check-only; no training path, no optimiser step, no held-out read): is the microbatch-1
vs batch 8 + 8 gradient difference of the formal CE + 0.5 x rank loss window coupling or FP32 round-off?

On the same fit batch (fold 0, update 7, C partner) as the failed gate:
  FP32 direct   formal microbatch 1 (16 retained graphs) and batch 8 + 8, as in check attempt 1
  FP32 / FP64   logit-gradient form: z from no-grad forwards, v = dL/dz, then sum_w J_w^T v_w by one forward / backward per
                microbatch (one graph at a time; FP64 would not fit 16 retained graphs). Same chain rule, both groupings
Coupling would persist in FP64; round-off would shrink by about 2^-29.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.train_session_matched_discrimination as t42  # noqa: E402
from scripts._common import load_cfg  # noqa: E402

import torch  # noqa: E402

import echo_routing.temporal.session_pairs as sp  # noqa: E402
import echo_routing.vit.target_mvit as tm  # noqa: E402


def grads(m) -> dict:
    return {n: (torch.zeros_like(p) if p.grad is None else p.grad.detach().double().clone()) for n, p in m.named_parameters()}


def rel(ga: dict, gb: dict) -> dict:
    mx = max(float(v.abs().max()) for v in gb.values()); k = max(gb, key=lambda n: float((ga[n] - gb[n]).abs().max())); d = float((ga[k] - gb[k]).abs().max())
    na = torch.cat([ga[n].flatten() for n in gb]); nb = torch.cat([gb[n].flatten() for n in gb])
    return {"max_abs": d, "grad_max": mx, "relative": d / mx, "normwise_relative": float((na - nb).norm() / nb.norm()), "max_abs_tensor": k}


def logit_grad(z: torch.Tensor, y: torch.Tensor, partner) -> torch.Tensor:
    z = z.detach().clone().requires_grad_(True); sp.total_loss(z, y, partner)[0].backward(); return z.grad.detach()


def decomposed(m, rgb: torch.Tensor, mk: torch.Tensor, y: torch.Tensor, partner, groups) -> dict:
    """sum_g J_g^T dL/dz_g, z and dL/dz taken from this grouping's own forwards."""
    with torch.no_grad():
        z = torch.cat([m(rgb[g], mk[g]) for g in groups])
    v = logit_grad(z, y, partner); m.zero_grad(set_to_none=True)
    for g in groups:
        m(rgb[g], mk[g]).backward(v[g])
    out = grads(m); m.zero_grad(set_to_none=True); return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--config", required=True); a = ap.parse_args(); a.set = []
    ctx = t42.Ctx(load_cfg(a), a.config); t42.t31.vsh.determinism(); torch.use_deterministic_algorithms(True, warn_only=False); device = torch.device("cuda"); t0, started = time.time(), t42.now(); ok, note = False, ""
    try:
        F0 = t42.Fold(ctx, 0); b = t42.Batches(F0, 0, [7])[0]; u, _, u8, mk, y = t42.t40.on_device(b, device); rgb = tm.to_rgb(u8); par = sp.PARTNER["C"]
        mb, b88 = [slice(w, w + 1) for w in range(16)], [slice(0, 8), slice(8, 16)]; out = {"batch": "fold 0, update 7, C partner (the failed gate's batch)"}
        m = t42.build(ctx, 0, device); sp.total_loss(t42.logits16(m, rgb, mk), y, par)[0].backward(); g32_mb = grads(m); m.zero_grad(set_to_none=True)
        sp.total_loss(torch.cat([m(rgb[g], mk[g]) for g in b88]), y, par)[0].backward(); g32_b = grads(m); m.zero_grad(set_to_none=True)
        out["fp32_direct_microbatch1_vs_batch8x2"] = rel(g32_mb, g32_b)
        g32_dmb = decomposed(m, rgb, mk, y, par, mb); out["fp32_decomposed_vs_direct_microbatch1"] = rel(g32_dmb, g32_mb)
        m.double(); r64, k64 = rgb.double(), mk.double()
        g64_mb = decomposed(m, r64, k64, y, par, mb); g64_b = decomposed(m, r64, k64, y, par, b88); out["fp64_microbatch1_vs_batch8x2"] = rel(g64_mb, g64_b)
        out["fp32_direct_microbatch1_vs_fp64"] = rel(g32_mb, g64_mb); out["fp32_direct_batch8x2_vs_fp64"] = rel(g32_b, g64_mb)
        out["limit_relative"] = t42.TOL["grad_rel"]; out["finished"] = t42.now()
        p = ctx.out / "diagnose_numerics.json"; t42.publish(ctx, "diagnose_numerics.json", out); print(json.dumps(out, indent=1), flush=True); ok = True; note = f"written {p.name}"
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        t42.ledger_append(ctx.ledger, "diagnose microbatch parity (FP32 / FP64, check only; GPU job wall time)", "gpu", started, time.time() - t0, ok, note)


if __name__ == "__main__":
    main()
