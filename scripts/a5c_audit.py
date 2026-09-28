#!/usr/bin/env python
"""Local review page for the A4C / A5C confusion on the native test clips (not published).

Groups: A5C clips most models call something else, A4C clips several models call A5C, and clear A5C / A4C references.
Each card: the clip (transcoded to H.264), one label strip per model over time, and the ResNet-18 per-frame P(A5C) and
P(A4C). Writes <out>/index.html, <out>/clips.json and <out>/videos/*.mp4.
"""
from __future__ import annotations

import html
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import base_parser, load_cfg  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from echo_routing.config import load_paths  # noqa: E402
from echo_routing.features.cache import load_video_features, read_index, variant_dir  # noqa: E402
from echo_routing.temporal.baselines.registry import decode_with_prob  # noqa: E402

VIEWS = ["PLAX", "PSAX", "A4C", "A5C", "SC4C"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
A4C, A5C = 2, 3
MAIN = ["b4", "b5", "b6", "b7", "mvit_echoprime_centre", "vit_router"]
SHOW = ["b0", *MAIN, "b4+b7~0.5"]
NAMES = {"b0": "ResNet-18 逐帧", "b4": "MS-TCN", "b5": "ASFormer", "b6": "STFM", "b7": "EchoViewCLIP",
         "mvit_echoprime_centre": "Echo-MViT + centre", "vit_router": "ViT-Router", "b4+b7~0.5": "MS-TCN + EchoViewCLIP"}


def clip_rows(idx: pd.DataFrame, cfg: dict, paths, ckpt: str) -> list[dict]:
    rows = []
    for r in idx[(idx["split"] == "test") & idx["label_index"].isin([A4C, A5C])].itertuples(index=False):
        vf = load_video_features(variant_dir(paths.cache_root, ckpt, "orig") / f"{r.video_id}.npz")
        preds = {}
        for m in SHOW:
            labels, prob = decode_with_prob(m, vf.prob, cfg, vf.feat, r.video_id)
            preds[m] = {"label": int(np.bincount(labels, minlength=5).argmax()), "labels": labels.tolist(),
                        "p_a5c": float(prob[:, A5C].mean()), "p_a4c": float(prob[:, A4C].mean())}
        rows.append({"video_id": r.video_id, "truth": int(r.label_index), "raw": str(r.raw_label), "n": int(vf.prob.shape[0]),
                     "frame_a5c": vf.prob[:, A5C].round(3).tolist(), "frame_a4c": vf.prob[:, A4C].round(3).tolist(), "preds": preds,
                     "n_wrong": sum(preds[m]["label"] != int(r.label_index) for m in MAIN),
                     "n_a5c": sum(preds[m]["label"] == A5C for m in MAIN)})
    return rows


def select(rows: list[dict], n_ref: int) -> dict[str, list[dict]]:
    a5c = [r for r in rows if r["truth"] == A5C]; a4c = [r for r in rows if r["truth"] == A4C]
    return {
        "A5C clips most models get wrong": sorted([r for r in a5c if r["n_wrong"] >= 4], key=lambda r: -r["n_wrong"]),
        "A4C clips several models call A5C": sorted([r for r in a4c if r["n_a5c"] >= 3], key=lambda r: -r["n_a5c"]),
        "Reference: clear A5C (all six right)": sorted([r for r in a5c if r["n_wrong"] == 0],
                                                       key=lambda r: -r["preds"]["b7"]["p_a5c"])[:n_ref],
        "Reference: clear A4C (all six right)": sorted([r for r in a4c if r["n_wrong"] == 0],
                                                       key=lambda r: -r["preds"]["b7"]["p_a4c"])[:n_ref],
    }


def transcode(src: str, dst: Path) -> None:
    if not dst.is_file():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-vf", "scale=640:-2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "20", "-movflags", "+faststart", "-an", str(dst)], check=True)


def strip(labels: list[int]) -> str:
    n = len(labels); out, s = [], 0
    for i in range(1, n + 1):
        if i == n or labels[i] != labels[s]:
            out.append(f'<span style="left:{100 * s / n:.2f}%;width:{100 * (i - s) / n:.2f}%;background:{COLORS[labels[s]]}"></span>')
            s = i
    return "".join(out)


def curve(a5c: list[float], a4c: list[float]) -> str:
    n = max(len(a5c) - 1, 1)
    pts = lambda v: " ".join(f"{100 * i / n:.2f},{30 - 28 * x:.2f}" for i, x in enumerate(v))  # noqa: E731
    return (f'<svg viewBox="0 0 100 32" preserveAspectRatio="none" class="curve"><polyline points="{pts(a4c)}" '
            f'style="fill:none;stroke:{COLORS[A4C]};stroke-width:1.2" vector-effect="non-scaling-stroke"/>'
            f'<polyline points="{pts(a5c)}" style="fill:none;stroke:{COLORS[A5C]};stroke-width:1.6" vector-effect="non-scaling-stroke"/></svg>')


def card(r: dict) -> str:
    rows = "".join(
        f'<div class="lab">{html.escape(NAMES[m])}</div><div class="strip">{strip(p["labels"])}</div>'
        f'<div class="val" style="color:{COLORS[p["label"]]}">{VIEWS[p["label"]]} · A5C {p["p_a5c"]:.2f}</div>'
        for m, p in r["preds"].items())
    return (f'<div class="card"><div class="head"><b>{html.escape(r["video_id"])}</b> · 标注 {VIEWS[r["truth"]]} '
            f'({html.escape(r["raw"])}) · {r["n"] / 10:.1f} s · 六个主要模型中 {r["n_wrong"]} 个判错</div>'
            f'<div class="body"><video src="videos/{html.escape(r["video_id"])}.mp4" controls loop muted playsinline preload="metadata"></video>'
            f'<div class="grid">{rows}<div class="lab">ResNet 逐帧 P(A5C)/P(A4C)</div><div class="strip plain">'
            f'{curve(r["frame_a5c"], r["frame_a4c"])}</div><div class="val"></div></div></div></div>')


PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>A4C / A5C review</title><style>
:root{{--bg:#f6f8f7;--surface:#fff;--ink:#17222a;--ink-2:#4e5d66;--line:#d7dedb}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0f1517;--surface:#161e21;--ink:#e6ecea;--ink-2:#b4bfba;--line:#2a353a}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,sans-serif}}
.wrap{{max-width:1180px;margin:0 auto;padding:20px 16px 60px}} h1{{font-size:22px}} h2{{font-size:17px;margin:28px 0 8px}}
p{{color:var(--ink-2);max-width:90ch}} .keys span{{margin-right:14px}} .keys i{{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:4px;vertical-align:-1px}}
.card{{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:12px;margin:10px 0}}
.head{{margin-bottom:8px;color:var(--ink-2)}} .head b{{color:var(--ink)}}
.body{{display:grid;grid-template-columns:320px 1fr;gap:14px;align-items:start}} video{{width:100%;border-radius:4px;background:#000}}
.grid{{display:grid;grid-template-columns:150px 1fr 150px;gap:4px 10px;align-items:center;font-size:12.5px}}
.strip{{position:relative;height:16px;border-radius:3px;overflow:hidden;background:var(--line)}} .strip span{{position:absolute;top:0;bottom:0}}
.strip.plain{{height:34px;background:transparent;border:1px solid var(--line)}} .curve{{width:100%;height:100%;display:block}}
.val{{font-variant-numeric:tabular-nums}} @media (max-width:760px){{.body{{grid-template-columns:1fr}}.grid{{grid-template-columns:110px 1fr 120px}}}}
</style></head><body><div class="wrap"><h1>A4C / A5C 判读复核（测试集，仅本地）</h1>
<p>每张卡片是一个测试 clip。左边是原始录像（放大到 640 宽）。右边每一行是一个模型随时间给出的视角标签，最右是它对整段的结论和平均 A5C 概率。
最后一行是 ResNet-18 逐帧的 A5C 概率（黄）和 A4C 概率（绿），范围 0 到 1。复核时请看：左室流出道（第五腔）是否清楚可见，出现了多久。</p>
<div class="keys">{keys}</div>{sections}</div></body></html>"""


def main() -> None:
    ap = base_parser("A4C/A5C review page")
    ap.add_argument("--ckpt-hash", required=True); ap.add_argument("--out", required=True); ap.add_argument("--n-ref", type=int, default=4)
    args = ap.parse_args(); cfg = load_cfg(args); paths = load_paths()
    idx = read_index(variant_dir(paths.cache_root, args.ckpt_hash, "orig"))
    man = pd.read_csv(paths.manifest_path).set_index("video_id")
    idx = idx.merge(man[["raw_label"]], left_on="video_id", right_index=True, how="left") if "raw_label" not in idx else idx
    rows = clip_rows(idx, cfg, paths, args.ckpt_hash)
    groups = select(rows, args.n_ref)
    out = Path(args.out); (out / "videos").mkdir(parents=True, exist_ok=True)
    for rs in groups.values():
        for r in rs:
            transcode(str(man.loc[r["video_id"], "video_path"]), out / "videos" / f"{r['video_id']}.mp4")
    keys = "".join(f'<span><i style="background:{c}"></i>{v}</span>' for v, c in zip(VIEWS, COLORS))
    sections = "".join(f"<h2>{html.escape(t)} · {len(rs)}</h2>" + "".join(card(r) for r in rs) for t, rs in groups.items())
    (out / "index.html").write_text(PAGE.format(keys=keys, sections=sections))
    (out / "clips.json").write_text(json.dumps({t: [{k: v for k, v in r.items() if k not in ("frame_a5c", "frame_a4c")} for r in rs]
                                                for t, rs in groups.items()}, indent=1))
    n5 = sum(r["truth"] == A5C for r in rows)
    print(json.dumps({t: len(rs) for t, rs in groups.items()}), f"A5C test clips: {n5}")


if __name__ == "__main__":
    main()
