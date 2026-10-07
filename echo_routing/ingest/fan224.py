"""Fan224 input contract: the EV9V acquisition-layout sector template, then a 224 x 224 letterbox (no centre crop).

Order of operations for one source frame (320 x 240 JPEG, RGB):
1. the gamma LUT (identity at gamma 1.0), before any masking or spatial change;
2. every pixel outside the fixed template is set to black;
3. the template's bounding box is cut out;
4. bilinear resize with preserved aspect ratio so the longer side becomes 224 (PIL, RGB);
5. the result is centred on a black 224 x 224 canvas.
The output is RGB uint8 (224, 224, 3). The model entry only applies the existing mean/std normalisation
(VisualEncoder.normalise); it must never resize or crop a fan224 image again.

The template is one fixed sector for every recording and frame (version fan224_v1). Its geometry was fitted on
training images only: the apex and both straight edges follow the region that changes over time in at least half of the
100 QC training recordings (docs/results/2026-10-06_fan224_visual_adaptation/README.md). The right edge sits 1.8 degrees
inside that region to drop the depth ticks and focus marker drawn over it; the radius stops 3-4 px short of the image's
lower arc to drop the depth-end tick at its centre; and x <= 275 drops the vertical interface lines and focal-zone
marker at the lower right. It is an acquisition-layout template, not an anatomical mask, and it is not a
de-identification guarantee.

The legacy E1 path (echo_routing.temporal.joint_visual.crop_uint8: short side 256, centre crop 224) is unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

VERSION = "fan224_v1"
SOURCE_SIZE = (320, 240)          # (width, height); anything else is refused
OUTPUT_SIZE = 224
TEMPLATE = {"apex_x": 156.7, "apex_y": 31.2, "left_half_angle_deg": 42.5, "right_half_angle_deg": 44.0,
            "radius": 200.0, "x_max": 275}
INTERPOLATION = "PIL bilinear"
PADDING = "black, centred"
GAMMA_ORDER = "gamma LUT before masking and resizing"


def template_mask(t: dict = TEMPLATE) -> np.ndarray:
    """Boolean (240, 320) mask of the sector: below the apex, between the two straight edges, within the radius."""
    w, h = SOURCE_SIZE
    yy, xx = np.mgrid[:h, :w]
    dx, dy = xx + 0.0 - t["apex_x"], yy + 0.0 - t["apex_y"]
    theta = np.degrees(np.arctan2(dx, dy))          # 0 = straight down, negative to the left
    return ((dy > 0) & (theta >= -t["left_half_angle_deg"]) & (theta <= t["right_half_angle_deg"])
            & (np.hypot(dx, dy) <= t["radius"]) & (xx <= t["x_max"]))


MASK = template_mask()
_YS, _XS = np.where(MASK)
BBOX = (int(_XS.min()), int(_YS.min()), int(_XS.max()) + 1, int(_YS.max()) + 1)   # (left, top, right, bottom)


def gamma_lut(gamma: float) -> list[int]:
    """Same LUT as the legacy path (echo_routing.temporal.joint_visual.gamma_lut)."""
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    return [int(255 * ((i / 255.0) ** gamma) + 0.5) for i in range(256)] * 3


def fan224(im: Image.Image, gamma: float = 1.0) -> np.ndarray:
    """One source frame -> RGB uint8 (224, 224, 3) under the fan224_v1 contract."""
    if im.size != SOURCE_SIZE:
        raise ValueError(f"fan224 needs a {SOURCE_SIZE[0]} x {SOURCE_SIZE[1]} source frame, got {im.size[0]} x {im.size[1]}")
    rgb = im.convert("RGB")
    if abs(gamma - 1.0) > 1e-9:
        rgb = rgb.point(gamma_lut(gamma))
    arr = np.asarray(rgb).copy()
    arr[~MASK] = 0
    crop = Image.fromarray(arr).crop(BBOX)
    w, h = crop.size; scale = min(OUTPUT_SIZE / w, OUTPUT_SIZE / h)
    resized = crop.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (OUTPUT_SIZE, OUTPUT_SIZE))
    canvas.paste(resized, ((OUTPUT_SIZE - resized.width) // 2, (OUTPUT_SIZE - resized.height) // 2))
    out = np.asarray(canvas, dtype=np.uint8)
    check_output(out)
    return out


def load_fan224_u8(path: Path, gamma: float) -> np.ndarray:
    with Image.open(path) as im:
        return fan224(im, gamma)


def check_output(arr: np.ndarray) -> None:
    if arr.dtype != np.uint8 or arr.shape != (OUTPUT_SIZE, OUTPUT_SIZE, 3):
        raise ValueError(f"fan224 output must be uint8 (224, 224, 3), got {arr.dtype} {arr.shape}")


def read_fan224_png(path: Path) -> np.ndarray:
    """A stored fan224 frame, read as is: size and mode are checked, nothing is resized or cropped."""
    with Image.open(path) as im:
        if im.size != (OUTPUT_SIZE, OUTPUT_SIZE) or im.mode != "RGB":
            raise ValueError(f"{path}: a fan224 frame must be RGB 224 x 224, got {im.mode} {im.size}")
        arr = np.asarray(im, dtype=np.uint8)
    check_output(arr)
    return arr


def spec() -> dict:
    """Everything that defines the transform; its digest names the derived frames and every cache built on them."""
    return {"version": VERSION, "source_size": list(SOURCE_SIZE), "output_size": OUTPUT_SIZE, "template": TEMPLATE,
            "bbox_ltrb": list(BBOX), "mask_sha256": hashlib.sha256(MASK.tobytes()).hexdigest(), "colour": "RGB",
            "interpolation": INTERPOLATION, "padding": PADDING, "gamma_order": GAMMA_ORDER,
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def spec_digest() -> str:
    return hashlib.sha256(json.dumps(spec(), sort_keys=True).encode()).hexdigest()


def scale_factor() -> float:
    w, h = BBOX[2] - BBOX[0], BBOX[3] - BBOX[1]
    return min(OUTPUT_SIZE / w, OUTPUT_SIZE / h)
