#!/usr/bin/env python
"""A41 check-failure diagnostic (fit fold 0 training windows only; no held-out data, no training run).

After check attempt 1 failed (6-update repeat / 3 + 3 resume / forced E = G over the fixed limits; FP32 microbatch-1 vs
batch-8 gradient over 1e-4 x grad max), measure where the differences arise:
  1. FP32 forward: per-window tokens / logits of the 8-window batch against single-window forwards, by SDPA backend.
  2. FP32 gradients: batch-8 against microbatch-1 accumulation per parameter group, by SDPA backend.
  3. BF16 backward repeat (same inputs twice) under use_deterministic_algorithms warn_only True and False.
  --part fp64: FP64 batch-2 against single-window forwards / gradients (separates batch coupling from round-off), and the
  FP32 single-window tokens against FP64 (round-off sensitivity).
Writes runs/video_echojepa_recognition_v1/diagnose_numerics{,_fp64}.json and the ledger.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.train_echojepa_recognition as t41  # noqa: E402
from scripts._common import load_cfg  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

CONFIG = "configs/experiments/video_echojepa_recognition_v1.yaml"


def by_group(ga: dict, gb: dict) -> dict:
    out = {}
    for name, sel in (("head", lambda k: k.startswith("head.")), ("A", lambda k: k.endswith(".A")), ("B", lambda k: k.endswith(".B"))):
        ks = [k for k in gb if sel(k)]; d = max(float((ga[k] - gb[k]).abs().max()) for k in ks); mx = max(float(gb[k].abs().max()) for k in ks)
        worst = max(ks, key=lambda k: float((ga[k] - gb[k]).abs().max()))
        out[name] = {"max_abs": d, "grad_max": mx, "rel": d / mx, "worst_tensor": worst}
    return out


def fp32_batch_vs_single(ctx, u8, mk, y, device, backend) -> dict:
    m = t41.build(ctx, "E", 0, 0, device, bf16=False); g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for x in m.loras:
            x.B.copy_(torch.randn(x.B.shape, generator=g) * 0.02)
    ctxm = sdpa_kernel([backend]) if backend is not None else torch.enable_grad()
    with ctxm:
        with torch.no_grad():
            tb = m.tokens(u8); ts = torch.cat([m.tokens(u8[w:w + 1]) for w in range(8)])
            fb, fs = t41.jl.readout_features(tb, mk), t41.jl.readout_features(ts, mk)
        F.cross_entropy(m(u8, mk), y).backward(); gb = t41.grads(m); m.zero_grad(set_to_none=True)
        for w in range(8):
            (F.cross_entropy(m(u8[w:w + 1], mk[w:w + 1]), y[w:w + 1]) / 8).backward()
        gm = t41.grads(m)
    rec = {"tokens_max_abs": float((tb - ts).abs().max()), "tokens_abs_max": float(tb.abs().max()), "token_norm_max": float(tb.norm(dim=-1).max()),
           "features_max_abs": float((fb - fs).abs().max()), "features_abs_max": float(fb.abs().max()), "logits_max_abs": float((m.head(fb) - m.head(fs)).abs().max()),
           "grad": {**t41.rel(gm, gb), "by_group": by_group(gm, gb)}}
    del m; torch.cuda.empty_cache(); return rec


def bf16_repeat(ctx, u8, mk, y, device) -> dict:
    out = []
    for _ in range(2):
        m = t41.build(ctx, "E", 0, 0, device); g = torch.Generator().manual_seed(11)
        with torch.no_grad():
            for x in m.loras:
                x.B.copy_(torch.randn(x.B.shape, generator=g) * 0.02)
        for w in range(8):
            (F.cross_entropy(m(u8[w:w + 1], mk[w:w + 1]), y[w:w + 1]) / 8).backward()
        out.append(t41.grads(m)); del m; torch.cuda.empty_cache()
    return {**t41.rel(out[0], out[1]), "bitwise_equal": all(torch.equal(out[0][k], out[1][k]) for k in out[0]), "by_group": by_group(out[0], out[1])}


def fp64_part(ctx, u8, mk, y, device) -> dict:
    """Same check-only nonzero B; the whole recogniser in FP64 (math SDPA), 2 windows."""
    m = t41.build(ctx, "E", 0, 0, device, bf16=False); g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for x in m.loras:
            x.B.copy_(torch.randn(x.B.shape, generator=g) * 0.02)
    with torch.no_grad():
        t32 = torch.cat([m.tokens(u8[w:w + 1]) for w in range(2)]).double()
    m.double(); x64 = t41.jl.jepa_input(u8[:2]).double(); logits = lambda x, k: m.head(t41.jl.readout_features(m.encoder(x), k).double())  # noqa: E731
    with torch.no_grad():
        tb = m.encoder(x64); ts = torch.cat([m.encoder(x64[w:w + 1]) for w in range(2)])
    F.cross_entropy(logits(x64, mk[:2]), y[:2]).backward(); gb = t41.grads(m); m.zero_grad(set_to_none=True)
    for w in range(2):
        (F.cross_entropy(logits(x64[w:w + 1], mk[w:w + 1]), y[w:w + 1]) / 2).backward()
    gm = t41.grads(m)
    rec = {"fp64_tokens_batch_vs_single_max_abs": float((tb - ts).abs().max()), "fp64_tokens_abs_max": float(tb.abs().max()), "fp64_grad_batch_vs_micro": {**t41.rel(gm, gb), "by_group": by_group(gm, gb)},
           "fp32_single_vs_fp64_tokens_max_abs": float((t32 - ts).abs().max()), "fp32_single_vs_fp64_tokens_rel_l2": float((t32 - ts).norm() / ts.norm()),
           "fp32_single_vs_fp64_token_rows_over_0.1": int(((t32 - ts).abs().amax(-1) > 0.1).sum())}
    return rec


def main() -> None:
    a = SimpleNamespace(config=CONFIG, set=[]); ctx = t41.Ctx(load_cfg(a), CONFIG); device = torch.device("cuda"); t41.t31.vsh.determinism()
    F0 = t41.Fold(ctx, 0); b = t41.batches(F0, 0, [3])[0]; _, _, u8, mk, y = t41.on_device(b, device)
    if "--part" in sys.argv and sys.argv[sys.argv.index("--part") + 1] == "fp64":
        rec = {"inputs": "fit fold 0, update 3's first 2 training windows", **fp64_part(ctx, u8, mk, y, device)}; t41.write_json(ctx.out / "diagnose_numerics_fp64.json", rec); print(t41.json.dumps(rec, indent=1)); return
    rec = {"inputs": "fit fold 0, update 3's 8 training windows", "fp32": {}}
    for name, be in (("default", None), ("math", SDPBackend.MATH), ("efficient", SDPBackend.EFFICIENT_ATTENTION)):
        rec["fp32"][name] = fp32_batch_vs_single(ctx, u8, mk, y, device, be)
    rec["bf16_repeat_warn_only_true"] = bf16_repeat(ctx, u8, mk, y, device)
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        rec["bf16_repeat_warn_only_false"] = bf16_repeat(ctx, u8, mk, y, device)
    except RuntimeError as e:
        rec["bf16_repeat_warn_only_false"] = {"error": str(e)[:400]}
    t41.write_json(ctx.out / "diagnose_numerics.json", rec); print(t41.json.dumps(rec, indent=1))


if __name__ == "__main__":
    t41.staged(t41.Ctx(load_cfg(SimpleNamespace(config=CONFIG, set=[])), CONFIG), f"diagnose check-1 numerics{' fp64' if '--part' in sys.argv else ''} (GPU job wall time)", "gpu", main)
