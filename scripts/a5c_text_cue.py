#!/usr/bin/env python
"""Is on-screen Doppler text ("PW  SV4.0mm ...", set up in the LVOT in A5C) a shortcut for A5C?

For every A4C / A5C clip, each 10 Hz sample frame is checked for bright pixels in the left-margin band where the
Doppler label appears (x 0-24, y 114-128 of the 320x240 frame; the probe icon below it starts at y ~132). Reports how
often the text appears per class and split, and the ResNet-18 (B0) P(A5C) on frames with and without it, within each
true class. If P(A5C) jumps on text frames inside A4C clips too, the frame classifier reads the text.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import base_parser, load_cfg  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from echo_routing.config import load_paths  # noqa: E402
from echo_routing.features.cache import load_video_features, read_index, variant_dir  # noqa: E402

A4C, A5C = 2, 3
BOX = (0, 114, 24, 128)   # x0, y0, x1, y1 in the 320x240 frame


def text_flags(frames_dir: Path, video_id: str, frame_idx: np.ndarray) -> np.ndarray:
    out = np.zeros(frame_idx.size, bool)
    for i, f in enumerate(frame_idx):
        p = frames_dir / video_id / f"{f}.jpg"
        if p.is_file():
            a = np.asarray(Image.open(p).convert("L").crop(BOX))
            out[i] = int((a > 128).sum()) >= 8
    return out


def main() -> None:
    ap = base_parser("Doppler-text cue check")
    ap.add_argument("--ckpt-hash", required=True); ap.add_argument("--hard", required=True)
    args = ap.parse_args(); load_cfg(args); paths = load_paths()
    idx = read_index(variant_dir(paths.cache_root, args.ckpt_hash, "orig"))
    hard = {r["video_id"] for r in json.loads(Path(args.hard).read_text())["a5c_clips"] if r["n_wrong"] >= 4}
    res, frames = {}, {A4C: {"text": [], "none": []}, A5C: {"text": [], "none": []}}
    hard_rows = []
    for split in ("train", "validation", "test"):
        sub = idx[(idx["split"] == split) & idx["label_index"].isin([A4C, A5C])]
        any_text = {A4C: [], A5C: []}
        for r in sub.itertuples(index=False):
            vf = load_video_features(variant_dir(paths.cache_root, args.ckpt_hash, "orig") / f"{r.video_id}.npz")
            t = text_flags(paths.ev9v_images, r.video_id, vf.frame_idx); y = int(r.label_index)
            any_text[y].append(bool(t.any()))
            if split != "train":        # frame-level P(A5C) only on held-out clips (training clips are fitted)
                frames[y]["text"] += vf.prob[t, A5C].tolist(); frames[y]["none"] += vf.prob[~t, A5C].tolist()
            if r.video_id in hard:
                hard_rows.append({"video_id": r.video_id, "text_frames": int(t.sum()), "frames": int(t.size),
                                  "p_a5c_text": float(vf.prob[t, A5C].mean()) if t.any() else None,
                                  "p_a5c_none": float(vf.prob[~t, A5C].mean()) if (~t).any() else None})
        res[split] = {VIEW: {"clips": len(v), "with_text": int(sum(v)), "share": round(float(np.mean(v)), 3) if v else None}
                      for VIEW, v in (("A4C", any_text[A4C]), ("A5C", any_text[A5C]))}
    res["heldout_frames_p_a5c"] = {name: {k: {"frames": len(v), "mean_p_a5c": round(float(np.mean(v)), 3) if v else None}
                                          for k, v in frames[c].items()} for name, c in (("A4C", A4C), ("A5C", A5C))}
    res["misread_a5c_test_clips"] = hard_rows
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
