"""A41 (configs/experiments/video_echojepa_recognition_v1.yaml): the pinned EchoJEPA / V-JEPA 2 ViT-L encoder with QKV LoRA
and a target / global recogniser.

Encoder: the pinned upstream `src.models.vision_transformer.vit_large` (third_party/EchoJEPA at COMMIT), loaded strictly from
a checkpoint's `target_encoder`. Each block's fused attn.qkv becomes W x + b + (alpha / r) B (A x); everything else in the
upstream forward (RoPE, SDPA, MLP, norms, activation checkpointing) is untouched. The recogniser reads the final-norm non-CLS
tokens [B, 1568, 1024] as [B, 8, 196, 1024]: concat(tokens[:, floor(m / 2)].mean(space), tokens.mean(time, space)) ->
Linear(2048, 5). The marker slot m only selects the token time of the readout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "third_party" / "EchoJEPA"
PYDEPS = ROOT / "third_party" / "a41_pydeps"
COMMIT = "77208e1a8521c0e82f79579eebe4b22f46c5ccbd"
ENCODER_KW = dict(img_size=(224, 224), num_frames=16, patch_size=16, tubelet_size=2, use_rope=True, uniform_power=True, use_sdpa=True,
                  use_activation_checkpointing=True, use_silu=False, wide_silu=True)
DEPTH, DIM, HEADS, T_TOK, S_TOK, N_CLS = 24, 1024, 16, 8, 196, 5
N_TOKENS = T_TOK * S_TOK
RANK, ALPHA, INIT_STD = 8, 16, 0.02
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
INIT_BASE = 2026102741
LORA_PARAMS = DEPTH * RANK * (DIM + 3 * DIM)
HEAD_PARAMS = 2 * DIM * N_CLS + N_CLS
TRAINABLE = LORA_PARAMS + HEAD_PARAMS
CONTRACT = {"encoder": ENCODER_KW, "lora": {"rank": RANK, "alpha": ALPHA, "scale": ALPHA / RANK, "A_std": INIT_STD, "B": "zeros", "targets": "blocks.*.attn.qkv"},
            "readout": "concat(tokens[:, floor(m / 2)].mean(space), tokens.mean(time, space)) -> Linear(2048, 5)", "input": {"mean": MEAN, "std": STD, "scale": "1 / 255"},
            "trainable": {"lora": LORA_PARAMS, "head": HEAD_PARAMS, "total": TRAINABLE}, "init_generator": "2026102741 + 10 s + f: blocks 0..23 A, then head weight"}


def src_paths() -> list[Path]:
    mod = sys.modules.get("src")
    return [] if mod is None else [Path(p).resolve() for p in getattr(mod, "__path__", [])]


def upstream():
    """The pinned upstream vision_transformer module; `src` (a namespace package) must resolve to the EchoJEPA clone only.
    The clone is appended to sys.path and stays there (a namespace path is recomputed from sys.path), so it shadows nothing."""
    want = [(UPSTREAM / "src").resolve()]
    if "src" in sys.modules and src_paths() != want:
        raise ImportError(f"another `src` package is already loaded: {src_paths()}")
    for p in (UPSTREAM, PYDEPS):
        if str(p) not in sys.path:
            sys.path.append(str(p))
    import src.models.vision_transformer as vit
    if src_paths() != want or not Path(vit.__file__).resolve().is_relative_to(UPSTREAM.resolve()):
        raise ImportError(f"the loaded `src` is not the pinned EchoJEPA clone: {src_paths()}")
    return vit


def pydeps() -> dict:
    """The round-only dependency the upstream patch_embed imports (einops; installed with --target, the shared .venv untouched)."""
    import einops
    return {"einops": einops.__version__, "file": str(Path(einops.__file__).resolve().relative_to(ROOT)), "in_round_dir": Path(einops.__file__).resolve().is_relative_to(PYDEPS.resolve())}


def upstream_files() -> dict:
    """Repository-relative paths of every loaded upstream module file (to compare with the pinned commit's blobs)."""
    out = {}
    for name, m in list(sys.modules.items()):
        f = getattr(m, "__file__", None)
        if (name == "src" or name.startswith("src.")) and f:
            p = Path(f).resolve()
            if not p.is_relative_to(UPSTREAM.resolve()):
                raise ImportError(f"{name} loaded from outside the pinned clone: {p}")
            out[name] = str(p.relative_to(UPSTREAM.resolve()))
    return out


def build_encoder(**kw) -> nn.Module:
    return upstream().vit_large(**{**ENCODER_KW, **kw})


# ----------------------------------------------------------------------------------------------- strict loading
def normalise_keys(sd: dict) -> dict:
    """Strip leading 'module.' / 'backbone.' prefixes only; a collision is an error."""
    out = {}
    for k, v in sd.items():
        k2 = k
        while k2.startswith(("module.", "backbone.")):
            k2 = k2.split(".", 1)[1]
        if k2 in out:
            raise KeyError(f"two checkpoint keys normalise to {k2}")
        out[k2] = v
    return out


def strict_encoder_state(ckpt: dict, encoder: nn.Module, key: str = "target_encoder") -> tuple[dict, dict]:
    """The checkpoint's `key` state, normalised, against the encoder's own keys. Only an unused legacy pos_embed may be dropped
    (RoPE: encoder.pos_embed is None). Missing, extra or shape-mismatched keys raise; nothing is filled from the model."""
    if key not in ckpt:
        raise KeyError(f"checkpoint has no '{key}' (top-level keys: {sorted(ckpt)})")
    sd = normalise_keys(ckpt[key]); exp = encoder.state_dict(); dropped = []
    if "pos_embed" in sd and "pos_embed" not in exp and getattr(encoder, "pos_embed", 0) is None:
        dropped.append({"key": "pos_embed", "shape": list(sd["pos_embed"].shape), "reason": "use_rope: encoder.pos_embed is None (unused)"}); sd = {k: v for k, v in sd.items() if k != "pos_embed"}
    missing, extra = sorted(set(exp) - set(sd)), sorted(set(sd) - set(exp)); shape = sorted(k for k in exp if k in sd and tuple(exp[k].shape) != tuple(sd[k].shape))
    if missing or extra or shape:
        raise KeyError(f"strict mismatch: missing {missing[:8]} ({len(missing)}), extra {extra[:8]} ({len(extra)}), shape {shape[:8]} ({len(shape)})")
    dtypes = sorted({str(v.dtype) for v in sd.values()})
    rep = {"top_level_keys": sorted(ckpt), "raw_keys": len(ckpt[key]), "prefixes_stripped": sorted({k.split(".", 1)[0] + "." for k in ckpt[key] if k.startswith(("module.", "backbone."))}),
           "dropped": dropped, "loaded_keys": len(sd), "expected_keys": len(exp), "parameters": int(sum(v.numel() for v in sd.values())), "source_dtypes": dtypes}
    return {k: sd[k].detach().to(torch.float32).contiguous() for k in exp}, rep


def load_base(encoder: nn.Module, state: dict) -> nn.Module:
    """load_state_dict(strict=True), then verify every entry equals the given state (no silent fill)."""
    encoder.load_state_dict(state, strict=True)
    cur = encoder.state_dict()
    if sorted(cur) != sorted(state) or not all(torch.equal(cur[k].cpu(), state[k]) for k in state):
        raise RuntimeError("the encoder state differs from the loaded state")
    return encoder


# ----------------------------------------------------------------------------------------------- LoRA and recogniser
class LoRAQKV(nn.Module):
    """W x + b + (alpha / r) B (A x); W, b frozen; A [r, in], B [out, r]."""

    def __init__(self, base: nn.Linear, rank: int = RANK, alpha: int = ALPHA) -> None:
        super().__init__()
        self.base, self.rank, self.scale = base, rank, alpha / rank
        self.A = nn.Parameter(torch.zeros(rank, base.in_features, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * F.linear(F.linear(x, self.A), self.B)


def add_lora(encoder: nn.Module) -> list[LoRAQKV]:
    for p in encoder.parameters():
        p.requires_grad_(False)
    out = []
    for blk in encoder.blocks:
        if not isinstance(blk.attn.qkv, nn.Linear) or isinstance(blk.attn.qkv, LoRAQKV):
            raise TypeError("attn.qkv is not a plain Linear")
        blk.attn.qkv = LoRAQKV(blk.attn.qkv); out.append(blk.attn.qkv)
    return out


def jepa_input(u8: torch.Tensor) -> torch.Tensor:
    """uint8 [B, 16, 224, 224, 3] -> FP32 ImageNet-normalised [B, 3, 16, 224, 224]."""
    x = u8.float().div(255.0); m = x.new_tensor(MEAN); s = x.new_tensor(STD)
    return ((x - m) / s).permute(0, 4, 1, 2, 3).contiguous()


def token_grid(tokens: torch.Tensor) -> torch.Tensor:
    """[B, 1568, D] (time-major T, H, W) -> [B, 8, 196, D]."""
    b, n, d = tokens.shape
    if n != N_TOKENS:
        raise ValueError(f"{n} tokens, expected {N_TOKENS}")
    return tokens.reshape(b, T_TOK, S_TOK, d)


def readout_features(tokens: torch.Tensor, marker: torch.Tensor) -> torch.Tensor:
    g = token_grid(tokens.float()); t = torch.div(marker.long(), 2, rounding_mode="floor")
    if (t < 0).any() or (t >= T_TOK).any():
        raise ValueError("marker slot outside 0..15")
    return torch.cat([g[torch.arange(g.shape[0], device=g.device), t].mean(1), g.mean((1, 2))], 1)


class Recogniser(nn.Module):
    def __init__(self, encoder: nn.Module, bf16: bool = True) -> None:
        super().__init__()
        self.encoder, self.bf16 = encoder, bf16
        self.loras = add_lora(encoder)
        self.head = nn.Linear(2 * DIM, N_CLS)
        self.head.to(next(encoder.parameters()).device)

    def tokens(self, u8: torch.Tensor) -> torch.Tensor:
        x = jepa_input(u8)
        with torch.autocast("cuda" if x.is_cuda else "cpu", dtype=torch.bfloat16, enabled=self.bf16):
            return self.encoder(x)

    def forward(self, u8: torch.Tensor, marker: torch.Tensor) -> torch.Tensor:
        return self.head(readout_features(self.tokens(u8), marker))

    def trainable(self) -> list[tuple[str, nn.Parameter]]:
        return [*((f"lora.{i}.{n}", p) for i, m in enumerate(self.loras) for n, p in (("A", m.A), ("B", m.B))), ("head.weight", self.head.weight), ("head.bias", self.head.bias)]

    def base_named(self) -> list[tuple[str, torch.Tensor]]:
        ids = {id(p) for _, p in self.trainable()}
        return [(k, v) for k, v in self.encoder.state_dict(keep_vars=True).items() if id(v) not in ids]

    def trainable_state(self) -> dict:
        return {k: p.detach().cpu().clone() for k, p in self.trainable()}

    def load_trainable(self, st: dict) -> None:
        cur = dict(self.trainable())
        if sorted(cur) != sorted(st):
            raise KeyError("trainable keys differ")
        with torch.no_grad():
            for k, p in cur.items():
                if tuple(p.shape) != tuple(st[k].shape):
                    raise ValueError(f"{k}: shape")
                p.copy_(st[k].to(p.device))


def init_generator(s: int, f: int) -> torch.Generator:
    return torch.Generator().manual_seed(INIT_BASE + 10 * int(s) + int(f))


@torch.no_grad()
def init_trainable(model: Recogniser, s: int, f: int) -> None:
    """Fixed order from one local generator: blocks 0..23 A ~ N(0, 0.02^2), then head weight; B and head bias 0."""
    g = init_generator(s, f)
    for m in model.loras:
        m.A.copy_(torch.randn(m.A.shape, generator=g) * INIT_STD); m.B.zero_()
    model.head.weight.copy_(torch.randn(model.head.weight.shape, generator=g) * INIT_STD); model.head.bias.zero_()


def lora_delta_norms(model: Recogniser) -> list[float]:
    """Frobenius norm of each block's (alpha / r) B A (FP32)."""
    return [float((m.scale * m.B.detach().float() @ m.A.detach().float()).norm()) for m in model.loras]
