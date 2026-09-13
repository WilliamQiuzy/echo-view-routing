"""Frame-level access to EV9V cines: paths, sampling, letterbox preprocessing.

Frames live at Images/{video_id}/frame_000001.jpg (320x240, 30 fps playback).
Sampling is timestamp-spaced; clip labels propagated to frames are weak supervision.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

from echo_routing.errors import FrameError

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FRAME_EXTS = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class VideoRecord:
    video_id: str
    frames_dir: Path
    n_frames: int
    label_index: int
    split: str


def list_frame_paths(frames_dir: Path) -> list[Path]:
    paths = sorted(p for p in Path(frames_dir).iterdir() if p.suffix.lower() in FRAME_EXTS)
    if not paths:
        raise FrameError(f"no frames under {frames_dir}")
    return paths


def letterbox(img: Image.Image, size: int) -> Image.Image:
    """Resize preserving aspect ratio, pad with black to size x size."""
    img = img.convert("RGB")
    return ImageOps.pad(img, (size, size), color=(0, 0, 0), method=Image.BILINEAR)


def apply_gamma(img: Image.Image, gamma: float) -> Image.Image:
    """Label-preserving appearance edit (nuisance factor). gamma=1 is identity."""
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if abs(gamma - 1.0) < 1e-9:
        return img
    lut = [int(255 * ((i / 255.0) ** gamma) + 0.5) for i in range(256)]
    return img.convert("RGB").point(lut * 3)


def to_tensor_array(img: Image.Image) -> np.ndarray:
    """HWC uint8 PIL -> CHW float32 normalized with ImageNet stats."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def load_frame(path: Path, size: int, gamma: float = 1.0) -> np.ndarray:
    with Image.open(path) as im:
        im = apply_gamma(im, gamma) if gamma != 1.0 else im
        return to_tensor_array(letterbox(im, size))
