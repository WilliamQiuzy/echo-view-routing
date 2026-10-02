"""Joint fine-tuning of the E1 visual encoder with a temporal head (round 2, Task 5b).

- The encoder is the authors' CLIP VisionTransformer class from the pinned EchoViewCLIP checkout, loaded strictly with
  the image-encoder weights exported from the frozen stage-1 checkpoint (scripts/extract_echoviewclip_features.py
  --export-visual). Its output equals model.image_encoder(image), the E1 feature.
- The first blocks are frozen and run without autograd; the last `trainable_blocks` blocks, ln_post and proj are
  trained. The trainable part runs in chunks of frames under activation checkpointing, so memory stays bounded while
  the loss is still computed on the whole batch (no gradient accumulation is needed).
- Frames are decoded from the source JPEGs with the EchoViewCLIP test preprocessing (gamma LUT, resize the short side
  to 256, centre crop 224) into uint8, and normalised on the GPU; this equals the float pipeline used for E1.
- Every sequence becomes a list of SampleRef(video_id, frame_idx, gamma) from its recipe, so the model sees the frames
  of the stream itself and never the rest of a source clip.
"""
from __future__ import annotations

import hashlib
import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset

from echo_routing.compose.recipe_builder import Fragment, Recipe
from echo_routing.temporal.windowed import SampleRef, stream_plan

PINNED = "f6a0d86074d7c906ae41546cd5f09f8b94da432a"
MEAN = (123.675, 116.28, 103.53)
STD = (58.395, 57.12, 57.375)


def load_vision_transformer(repo: Path, export_path: Path) -> tuple[nn.Module, dict]:
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if head != PINNED:
        raise SystemExit(f"EchoViewCLIP commit {head} != pinned {PINNED}")
    spec = importlib.util.spec_from_file_location("evc_clip_model", Path(repo) / "clip" / "model.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    blob = torch.load(export_path, map_location="cpu", weights_only=True)
    a = blob["arch"]
    vt = mod.VisionTransformer(input_resolution=a["input_resolution"], patch_size=a["patch_size"], width=a["width"], layers=a["layers"],
                               heads=a["heads"], output_dim=a["output_dim"], design_details=a["design_details"])
    vt.load_state_dict(blob["state_dict"], strict=True)
    return vt.float(), {"arch": a, "source_checkpoint_sha256": blob["source_checkpoint_sha256"], "commit": head}


class VisualEncoder(nn.Module):
    """CLIP VisionTransformer split into a frozen prefix (no autograd) and a trainable suffix (checkpointed chunks)."""

    def __init__(self, vt: nn.Module, trainable_blocks: int = 1, chunk: int = 128) -> None:
        super().__init__()
        self.vt, self.chunk = vt, int(chunk)
        self.n_blocks = len(vt.transformer.resblocks); self.split = self.n_blocks - int(trainable_blocks)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)
        for name, p in self.vt.named_parameters():
            p.requires_grad_(self.is_trainable(name))

    def is_trainable(self, name: str) -> bool:
        if name.startswith(("ln_post.", "proj")):
            return True
        if name.startswith("transformer.resblocks."):
            return int(name.split(".")[2]) >= self.split
        return False

    def normalise(self, u8: torch.Tensor) -> torch.Tensor:
        """uint8 (N, H, W, 3) -> normalised float (N, 3, H, W), as the EchoViewCLIP test preprocessing."""
        return (u8.permute(0, 3, 1, 2).float() - self.mean) / self.std

    def prefix(self, x: torch.Tensor) -> torch.Tensor:
        """Tokens entering the first trainable block, (L, N, D) as in the official forward."""
        vt = self.vt
        x = vt.conv1(x); x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = vt.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1) + vt.positional_embedding.to(x.dtype)
        x = vt.ln_pre(x).permute(1, 0, 2)
        for blk in vt.transformer.resblocks[: self.split]:
            x = blk(x)
        return x

    def suffix(self, tokens: torch.Tensor) -> torch.Tensor:
        vt = self.vt; x = tokens
        for blk in vt.transformer.resblocks[self.split:]:
            x = blk(x)
        x = vt.ln_post(x.permute(1, 0, 2)[:, 0, :])
        return x @ vt.proj

    def forward(self, u8: torch.Tensor) -> torch.Tensor:
        outs = []
        for i in range(0, u8.shape[0], self.chunk):
            x = self.normalise(u8[i:i + self.chunk])
            with torch.no_grad():
                tok = self.prefix(x)
            if self.training and torch.is_grad_enabled():
                outs.append(checkpoint(self.suffix, tok, use_reentrant=False))
            else:
                outs.append(self.suffix(tok))
        return torch.cat(outs, 0)

    def trainable(self) -> list[tuple[str, nn.Parameter]]:
        return [(n, p) for n, p in self.vt.named_parameters() if p.requires_grad]


def trainable_blocks_from(unfreeze: list[str], n_blocks: int) -> int:
    """Number of trailing blocks to train, from a config list such as
    [image_encoder.transformer.resblocks.11, image_encoder.ln_post, image_encoder.proj]. The list must name ln_post, proj
    and a contiguous run of the last blocks; anything else is refused rather than silently ignored."""
    names = [u.removeprefix("image_encoder.") for u in unfreeze]
    blocks = sorted(int(n.split(".")[2]) for n in names if n.startswith("transformer.resblocks."))
    rest = {n for n in names if not n.startswith("transformer.resblocks.")}
    if rest != {"ln_post", "proj"} or not blocks or blocks != list(range(n_blocks - len(blocks), n_blocks)):
        raise SystemExit(f"unsupported unfreeze list {unfreeze}: expected ln_post, proj and the last k resblocks")
    return len(blocks)


def crop_uint8(im: Image.Image) -> np.ndarray:
    """Geometry of run_windowed_echoviewclip.preprocess (short side 256, centre crop 224), kept as uint8 HWC."""
    w, h = im.size; s = 256 / min(w, h)
    im = im.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BILINEAR)
    w, h = im.size; left, top = (w - 224) // 2, (h - 224) // 2
    return np.asarray(im.crop((left, top, left + 224, top + 224)), dtype=np.uint8)


def gamma_lut(gamma: float) -> list[int]:
    return [int(255 * ((i / 255.0) ** gamma) + 0.5) for i in range(256)] * 3


def load_frame_u8(path: Path, gamma: float) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if abs(gamma - 1.0) > 1e-9:
            im = im.point(gamma_lut(gamma))
        return crop_uint8(im)


def native_window_refs(video_id: str, label: int, start: int, length: int, loader) -> list[SampleRef]:
    return stream_plan(Recipe(video_id, "native", (Fragment(video_id, int(label), "orig", int(start), int(start) + int(length)),)), loader)


def refs_digest(refs: list[list[SampleRef]]) -> str:
    h = hashlib.sha256()
    for seq in refs:
        for r in seq:
            h.update(f"{r.video_id}|{r.frame_idx}|{r.gamma:.3f};".encode())
        h.update(b"#")
    return h.hexdigest()


class FrameBatches(Dataset):
    """Item i = all frames of update i as uint8 (N, 224, 224, 3) plus the per-sequence frame counts."""

    def __init__(self, plans: list[list[list[SampleRef]]], frame_paths) -> None:
        self.plans, self.frame_paths = plans, frame_paths

    def __len__(self) -> int:
        return len(self.plans)

    def __getitem__(self, i: int):
        seqs = self.plans[i]; frames = []
        for seq in seqs:
            for r in seq:
                paths = self.frame_paths(r.video_id)
                frames.append(load_frame_u8(paths[min(r.frame_idx, len(paths) - 1)], r.gamma))
        return torch.from_numpy(np.stack(frames)), torch.tensor([len(s) for s in seqs])
