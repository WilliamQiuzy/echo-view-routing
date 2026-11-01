"""Label-free image interventions and fixed outcome-cohort selection.

These operators establish input sensitivity, not semantic/anatomical ground truth.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

GROUPS = ("A5C_recovered", "A5C_missed", "A4C_harmed", "A4C_retained")
GLOBAL_NAMES = ("original", "periphery", "periphery_control", "chroma", "chroma_control",
                "static_bright", "static_bright_control", "grayscale", "target_repeat",
                "full_letterbox", "retained_letterbox")


def outcome_groups(labels, base, prediction):
    y, b, p = map(np.asarray, (labels, base, prediction))
    if not (y.shape == b.shape == p.shape):
        raise ValueError("outcome shapes differ")
    el = b == 2
    masks = ((y == 3) & (p == 3), (y == 3) & (p == 2),
             (y == 2) & (p == 3), (y == 2) & (p == 2))
    return {k: np.flatnonzero(el & m) for k, m in zip(GROUPS, masks)}


def select_indices(candidates, sources, sessions, seed, n=8):
    order = np.random.default_rng(seed).permutation(np.asarray(candidates, int)).tolist()
    chosen, levels, ss, rr = [], [], set(), set()
    for level in ("session", "recording", "remaining"):
        for i in order:
            if len(chosen) == n:
                break
            if i in chosen:
                continue
            if level == "session" and sessions[i] in ss:
                continue
            if level == "recording" and sources[i] in rr:
                continue
            chosen.append(i); levels.append(level)
            ss.add(sessions[i]); rr.add(sources[i])
    return chosen, levels


def grid_masks():
    masks = np.zeros((49, 224, 224), bool)
    for j in range(49):
        y, x = divmod(j, 7)
        masks[j, y * 32:(y + 1) * 32, x * 32:(x + 1) * 32] = True
    return masks


def border_mask():
    m = np.ones((224, 224), bool)
    m[16:-16, 16:-16] = False
    return m


def area_control(mask, seed):
    m = np.asarray(mask, bool)
    out = np.zeros(m.size, bool)
    out[np.random.default_rng(seed).permutation(m.size)[:int(m.sum())]] = True
    return out.reshape(m.shape)


def proxy_masks(window):
    x = np.asarray(window, np.uint8)
    chroma = ((x.max(-1) - x.min(-1) >= 25) & (x.max(-1) >= 60)).any(0)
    gray = x.astype(np.float32).mean(-1)
    return {"periphery": border_mask(), "chroma": chroma,
            "static_bright": (gray.mean(0) >= 180) & (gray.std(0) <= 3)}


def replacement(window, fill):
    x = np.asarray(window, np.uint8)
    if fill == "mean":
        return np.broadcast_to(np.rint(x.mean((1, 2), keepdims=True)).astype(np.uint8), x.shape)
    if fill == "blur":
        return np.stack([np.asarray(Image.fromarray(f).filter(ImageFilter.BoxBlur(8))) for f in x])
    raise ValueError("unknown replacement")


def mask_window(window, mask, fill="mean", prepared=None):
    x = np.asarray(window, np.uint8)
    m = np.asarray(mask, bool)
    if x.shape != (16, 224, 224, 3) or m.shape != (224, 224):
        raise ValueError("bad input window/mask geometry")
    z = x.copy()
    rep = replacement(x, fill) if prepared is None else prepared
    z[:, m] = rep[:, m]
    return z


def weighted_mean(scores, rows, weights):
    rows, weights = np.asarray(rows, int), np.asarray(weights, np.float64)
    if rows.shape != weights.shape or np.any(weights < 0) or not np.isclose(weights.sum(), 1, atol=1e-12, rtol=0):
        raise ValueError("invalid duration weights")
    return float(np.dot(np.asarray(scores, np.float64)[rows], weights))


def letterbox_pair(raw, gamma):
    a = np.asarray(raw, np.uint8)
    im = Image.fromarray(a).convert("RGB")
    if abs(float(gamma) - 1) > 1e-9:
        lut = [int(255 * ((i / 255.) ** float(gamma)) + .5) for i in range(256)] * 3
        im = im.point(lut)
    w, h = im.size
    s = 256 / min(w, h)
    rw, rh = max(224, round(w * s)), max(224, round(h * s))
    left, top = (rw - 224) // 2, (rh - 224) // 2
    footprint = Image.new("L", (rw, rh))
    footprint.paste(255, (left, top, left + 224, top + 224))
    native_mask = footprint.resize((w, h), Image.Resampling.NEAREST)
    k = 224 / max(w, h)
    nw, nh = max(1, round(w * k)), max(1, round(h * k))
    x0, y0 = (224 - nw) // 2, (224 - nh) // 2
    full = np.zeros((224, 224, 3), np.uint8)
    full[y0:y0 + nh, x0:x0 + nw] = np.asarray(im.resize((nw, nh), Image.Resampling.BILINEAR))
    keep = np.zeros((224, 224), bool)
    keep[y0:y0 + nh, x0:x0 + nw] = np.asarray(native_mask.resize((nw, nh), Image.Resampling.NEAREST)) > 0
    retained = full.copy(); retained[~keep] = 0
    return full, retained, {"raw_shape": [h, w], "resized_shape": [rh, rw],
                            "raw_crop_retained_fraction": float((np.asarray(native_mask) > 0).mean()),
                            "retained_mask": keep}


def global_variants(window, marker, full, retained, seed):
    x = np.asarray(window, np.uint8)
    masks = proxy_masks(x)
    rep = replacement(x, "mean")
    yield "original", x
    for j, (name, mask) in enumerate(masks.items()):
        yield name, mask_window(x, mask, prepared=rep)
        yield name + "_control", mask_window(x, area_control(mask, [seed, j]), prepared=rep)
    gray = np.asarray([np.asarray(Image.fromarray(f).convert("L")) for f in x])
    yield "grayscale", np.repeat(gray[..., None], 3, -1)
    yield "target_repeat", np.repeat(x[int(marker)][None], 16, 0)
    yield "full_letterbox", full
    yield "retained_letterbox", retained


def image_proxies(window):
    x = np.asarray(window, np.float32)
    g = x.mean(-1)
    return {"intensity_mean": float(g.mean()), "intensity_std": float(g.std()),
            "dark_fraction_le10": float((g <= 10).mean()),
            "saturated_fraction_ge245": float((g >= 245).mean()),
            "spatial_abs_gradient": float((np.abs(np.diff(g, axis=1)).mean() + np.abs(np.diff(g, axis=2)).mean()) / 2),
            "temporal_abs_difference": float(np.abs(np.diff(g, axis=0)).mean()),
            **{k + "_fraction": float(v.mean()) for k, v in proxy_masks(window).items()}}
