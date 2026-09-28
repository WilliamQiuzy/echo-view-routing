#!/usr/bin/env python
"""A4C/A5C error analysis on the native test clips (companion to scripts/a5c_audit.py).

1. Frame evidence of the ResNet-18 frame classifier (B0) for every A5C test clip, grouped by how many of the six main
   models misread the clip: share of frames called A5C, mean / 90th percentile / max P(A5C).
2. Nearest training clips (cosine similarity of mean B0 features) for every A5C test clip: labels of the 10 nearest.
3. Acquisition sessions (video id prefix before "_320x240") shared between the official train and test splits.
4. Frame grids (6 frames per clip, evenly spaced, each with its B0 P(A5C)) for misread clips and references.
Writes <out>/analysis.json and <out>/grid_*.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import base_parser, load_cfg  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from echo_routing.config import load_paths  # noqa: E402
from echo_routing.features.cache import load_video_features, read_index, variant_dir  # noqa: E402
from echo_routing.temporal.baselines.registry import decode_with_prob  # noqa: E402

VIEWS = ["PLAX", "PSAX", "A4C", "A5C", "SC4C"]
A4C, A5C = 2, 3
MAIN = ["b4", "b5", "b6", "b7", "mvit_echoprime_centre", "vit_router"]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def session(video_id: str) -> str:
    return video_id.split("_320x240")[0]


def grid(rows: list[dict], frames_dir: Path, out: Path, title: str, n_cols: int = 6) -> None:
    w, h, pad, head = 320, 240, 6, 26
    font = ImageFont.truetype(FONT, 15) if Path(FONT).is_file() else ImageFont.load_default()
    img = Image.new("RGB", (n_cols * (w + pad) + pad, 34 + len(rows) * (h + head + pad)), "white")
    d = ImageDraw.Draw(img); d.text((pad, 8), title, fill="black", font=font)
    for i, r in enumerate(rows):
        y = 34 + i * (h + head + pad)
        d.text((pad, y + 4), f"{r['video_id']}  label {VIEWS[r['truth']]}  misread by {r['n_wrong']}/6  "
                             f"B0 frames called A5C {r['frac_a5c']:.0%}", fill="black", font=font)
        picks = np.linspace(0, r["n"] - 1, n_cols).round().astype(int)
        for j, k in enumerate(picks):
            f = frames_dir / r["video_id"] / f"{r['frame_idx'][k]}.jpg"
            tile = Image.open(f).convert("RGB").resize((w, h)) if f.is_file() else Image.new("RGB", (w, h), "gray")
            x = pad + j * (w + pad); img.paste(tile, (x, y + head))
            d.rectangle([x, y + head, x + 118, y + head + 20], fill="black")
            d.text((x + 4, y + head + 2), f"P(A5C) {r['p_a5c'][k]:.2f}", fill="yellow", font=font)
    img.save(out)


def main() -> None:
    ap = base_parser("A4C/A5C error analysis")
    ap.add_argument("--ckpt-hash", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args(); cfg = load_cfg(args); paths = load_paths()
    idx = read_index(variant_dir(paths.cache_root, args.ckpt_hash, "orig"))
    load = lambda v: load_video_features(variant_dir(paths.cache_root, args.ckpt_hash, "orig") / f"{v}.npz")  # noqa: E731
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # 1. per-clip evidence for test A4C / A5C clips
    clips = []
    for r in idx[(idx["split"] == "test") & idx["label_index"].isin([A4C, A5C])].itertuples(index=False):
        vf = load(r.video_id); p = vf.prob
        wrong = sum(int(np.bincount(decode_with_prob(m, p, cfg, vf.feat, r.video_id)[0], minlength=5).argmax()) != r.label_index
                    for m in MAIN)
        clips.append({"video_id": r.video_id, "truth": int(r.label_index), "n": int(p.shape[0]), "n_wrong": int(wrong),
                      "frac_a5c": float((p.argmax(1) == A5C).mean()), "mean_a5c": float(p[:, A5C].mean()),
                      "p90_a5c": float(np.percentile(p[:, A5C], 90)), "max_a5c": float(p[:, A5C].max()),
                      "frame_idx": vf.frame_idx.tolist(), "p_a5c": p[:, A5C].round(3).tolist(),
                      "emb": (vf.feat.mean(0) / np.linalg.norm(vf.feat.mean(0))).tolist()})
    df = pd.DataFrame(clips)
    a5c = df[df.truth == A5C].copy()
    a5c["group"] = np.where(a5c.n_wrong >= 4, "misread by >=4/6", np.where(a5c.n_wrong == 0, "read by all 6", "misread by 1-3"))
    evidence = a5c.groupby("group")[["frac_a5c", "mean_a5c", "p90_a5c", "max_a5c", "n"]].median().round(3)
    evidence["clips"] = a5c.groupby("group").size()

    # 2. nearest training clips by mean B0 feature
    tr = idx[idx["split"] == "train"]
    tr_emb = np.stack([(lambda f: f / np.linalg.norm(f))(load(v).feat.mean(0)) for v in tr["video_id"]])
    tr_lab = tr["label_index"].to_numpy().astype(int)
    def nn_labels(e, k=10):
        top = np.argsort(-(tr_emb @ np.asarray(e)))[:k]
        return np.bincount(tr_lab[top], minlength=5)
    a5c["nn_a5c"] = [int(nn_labels(e)[A5C]) for e in a5c.emb]; a5c["nn_a4c"] = [int(nn_labels(e)[A4C]) for e in a5c.emb]
    nn = a5c.groupby("group")[["nn_a5c", "nn_a4c"]].mean().round(2)

    # 3. sessions shared between train and test
    idx = idx.assign(session=idx["video_id"].map(session))
    s_train, s_test = set(idx[idx.split == "train"].session), set(idx[idx.split == "test"].session)
    test = idx[idx.split == "test"]
    shared = {"test_sessions": len(s_test), "test_sessions_also_in_train": len(s_test & s_train),
              "test_clips_whose_session_is_in_train": int(test.session.isin(s_train).sum()), "test_clips": int(len(test))}
    a5c["session_in_train"] = a5c.video_id.map(session).isin(s_train)
    a5c["train_a5c_same_session"] = [int(((idx.split == "train") & (idx.session == session(v)) & (idx.label_index == A5C)).sum())
                                     for v in a5c.video_id]

    # 4. frame grids
    fd = paths.ev9v_images
    pick = lambda d, q, k=None: d.query(q).sort_values("n_wrong", ascending=False).to_dict("records")[:k]  # noqa: E731
    grid(pick(a5c, "n_wrong >= 4"), fd, out / "grid_misread_a5c.png", "A5C clips misread by at least 4 of 6 models")
    grid(a5c.query("n_wrong == 0").sort_values("mean_a5c", ascending=False).to_dict("records")[:5], fd, out / "grid_clear_a5c.png",
         "Reference: A5C clips read correctly by all 6 models")
    a4c = df[df.truth == A4C]
    grid(a4c.query("n_wrong == 0").sort_values("mean_a5c").to_dict("records")[:4], fd, out / "grid_clear_a4c.png",
         "Reference: A4C clips read correctly by all 6 models")
    grid(a4c.query("n_wrong >= 3").sort_values("n_wrong", ascending=False).to_dict("records"), fd, out / "grid_a4c_called_a5c.png",
         "A4C clips called A5C by at least 3 of 6 models")

    cols = ["video_id", "n_wrong", "n", "frac_a5c", "mean_a5c", "max_a5c", "nn_a5c", "nn_a4c", "session_in_train", "train_a5c_same_session"]
    res = {"evidence_by_group": evidence.reset_index().to_dict("records"), "nearest_train_by_group": nn.reset_index().to_dict("records"),
           "sessions": shared, "a5c_clips": a5c.sort_values("n_wrong", ascending=False)[cols].to_dict("records")}
    (out / "analysis.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "a5c_clips"}, indent=1, default=float))
    print(a5c.sort_values("n_wrong", ascending=False)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
