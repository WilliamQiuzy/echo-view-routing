"""A33 (configs/experiments/video_long_support_density_recognition_v1.yaml): 32-slot windows over A08's 16-sample
long support, and the 32-frame adaptation of A08's TargetMViT.

Windows. For a request c, a = window_frames(c, N, 16, 16) are A08's 16 sample positions (about 1.5 s at 10 Hz).
  R32  slots 2i and 2i+1 both show ref[a_i].
  I32  even slots and slot 31 as R32; odd slot 2i+1 (i < 15) shows the raw frame floor((l + r) / 2) between the frames
       l, r of a_i, a_{i+1} when both samples lie in the same fragment of the stream (one source, one gamma) and a raw
       frame lies strictly between them; otherwise the left frame repeats and the reason is returned.
The marker follows A22's planned-time rule: slot coordinates 2a_i (even), a_i + a_{i+1} (odd), 2a_15 (slot 31); the
marker is the middle slot whose coordinate is 2c. Both arms share coordinates, marker, support and targets.

Model. torchvision's MViTv2-S at temporal_size 32 differs from the 16-frame build only in the 16 rel_pos_t tables
(15 -> 31 rows). They are resized once with torchvision's own runtime rule (linear, align_corners=False); every other
tensor is copied bit for bit and the state is loaded strictly. Nothing here reads a class label.
"""
from __future__ import annotations

import copy
import hashlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import echo_routing.vit.target_mvit as tm

N_SLOTS = 32
REASONS = ("inserted", "repeated_position", "fragment_boundary", "gamma_change", "no_frame_between", "last_slot")


def fragment_ids(seq: dict) -> np.ndarray:
    """Fragment index of every sample of a stream (native streams are one fragment); checks refs against fragments."""
    n = len(seq["refs"])
    if seq["kind"] == "native":
        if len({r[0] for r in seq["refs"]}) != 1:
            raise ValueError("a native stream must have one source")
        return np.zeros(n, np.int64)
    lens = [int(f[4]) - int(f[3]) for f in seq["frags"]]
    if sum(lens) != n:
        raise ValueError("fragment lengths do not cover the stream")
    ids = np.repeat(np.arange(len(lens)), lens)
    for k, fr in enumerate(seq["frags"]):
        seg = [seq["refs"][j] for j in np.flatnonzero(ids == k)]
        if any(r[0] != fr[0] for r in seg) or len({float(r[2]) for r in seg}) != 1:
            raise ValueError("a fragment's references change source or gamma")
    return ids


def slot_plan(a: np.ndarray, refs: list, frag: np.ndarray) -> dict:
    """R32 / I32 references, planned coordinates and the odd-slot reason of one window (a: 16 sample positions)."""
    a = np.asarray(a, int)
    if a.shape != (16,):
        raise ValueError("a window has 16 sample positions")
    r32, i32, coord, why = [], [], [], []
    for i in range(16):
        left = tuple(refs[a[i]]); r32 += [left, left]; i32.append(left); coord.append(2 * int(a[i]))
        if i == 15:
            i32.append(left); coord.append(2 * int(a[i])); why.append("last_slot"); continue
        coord.append(int(a[i]) + int(a[i + 1])); right = tuple(refs[a[i + 1]])
        if a[i] == a[i + 1]:
            reason = "repeated_position"
        elif frag[a[i]] != frag[a[i + 1]] or left[0] != right[0]:
            reason = "fragment_boundary"
        elif float(left[2]) != float(right[2]):
            reason = "gamma_change"
        else:
            l, r = int(left[1]), int(right[1]); mid = (l + r) // 2
            reason = "inserted" if l < mid < r else "no_frame_between"
        if reason == "inserted":
            i32.append((left[0], (int(left[1]) + int(right[1])) // 2, left[2]))
        else:
            i32.append(left)
        why.append(reason)
    return {"R32": r32, "I32": i32, "coord": np.asarray(coord), "reasons": why}


def marker32(coord: np.ndarray, c: int) -> int:
    return tm.marker_slot(np.asarray(coord), 2 * int(c))


# ----------------------------------------------------------------------------------------------- 32-frame model
def resize_rows(t: torch.Tensor, rows: int) -> torch.Tensor:
    """torchvision's runtime rel-pos resize (mvit._interpolate) with align_corners stated: linear over the row axis."""
    if t.shape[0] == rows:
        return t.clone()
    return F.interpolate(t.permute(1, 0).unsqueeze(0), size=rows, mode="linear", align_corners=False).squeeze(0).permute(1, 0).contiguous()


def mvit32() -> nn.Module:
    """MViTv2-S exactly as torchvision builds it, at temporal_size 32, with the 512-d projection."""
    from torchvision.models.video import mvit as M
    orig = M._mvit
    M._mvit = lambda **kw: orig(**{**kw, "temporal_size": N_SLOTS})
    try:
        net = M.mvit_v2_s()
    finally:
        M._mvit = orig
    net.head[-1] = nn.Linear(net.head[-1].in_features, tm.EMBED_DIM)
    return net


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def adapt(model16: tm.TargetMViT) -> tuple[tm.TargetMViT, list]:
    """A 32-slot TargetMViT holding model16's weights; returns the model and every tensor whose shape changed."""
    m = tm.TargetMViT(mvit32()); src = model16.state_dict(); dst = m.state_dict(); out, changes = {}, []
    if list(src) != list(dst):
        raise ValueError("the 16- and 32-frame models have different parameter names")
    for k, v in src.items():
        if v.shape == dst[k].shape:
            out[k] = v.clone()
        elif k.endswith("rel_pos_t") and v.ndim == 2 and v.shape[1] == dst[k].shape[1]:
            out[k] = resize_rows(v, dst[k].shape[0]); changes.append({"name": k, "from": list(v.shape), "to": list(dst[k].shape), "sha256_from": _sha(v), "sha256_to": _sha(out[k])})
        else:
            raise ValueError(f"unexpected shape change at {k}: {tuple(v.shape)} -> {tuple(dst[k].shape)}")
    m.load_state_dict(out, strict=True); m.to(next(model16.parameters()).device).eval()
    return m, changes


def on_the_fly(model16: tm.TargetMViT) -> tm.TargetMViT:
    """The 16-frame model with only its token-grid bookkeeping set to 32 frames, so that torchvision resizes rel_pos_t at
    run time: the library's own path, used to check that adapt() materialises the same computation."""
    m = copy.deepcopy(model16); m.encoder.pos_encoding.temporal_size = N_SLOTS // 2; return m.eval()
