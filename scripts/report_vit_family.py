#!/usr/bin/env python
"""Tables for the ViT-family evaluation: nine-code cine level (EV9V benchmark protocol) + the unified family5 ladder
(recognition / segmentation / routing), baselines and ViT family side by side, from one ladder run.

  scripts/report_vit_family.py --run-dir runs/<ladder run> --video-level vivit_fe=runs/<predict run>/video_level.json,... \
      --out docs/results/<date>_vit_family/vit_family.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NAMES = {
    "b0": "ResNet-18 frame encoder (B0), per frame",
    "b6": "STFM (official, sliding window)",
    "b7": "EchoViewCLIP (official, CLIP ViT-B/16, sliding window)",
    "b8": "EchoPrime view classifier (ConvNeXt-B, per frame)",
    "b4": "MS-TCN (official) on ResNet-18 features",
    "b5": "ASFormer (official) on ResNet-18 features",
    "vit_frame": "ViT-S/16 frame, per frame",
    "vivit_fe": "Factorised video ViT (ViT-S + temporal Transformer), sliding window",
    "vivit_fe_mix": "Factorised video ViT, transition-aware (ours), sliding window",
    "vivit_fe_centre": "Factorised video ViT, centre-supervised transition-aware (ours), sliding window",
    "mvit_k400": "MViTv2-S (Kinetics-400 init), sliding window",
    "mvit_echoprime": "Echo-MViTv2-S (EchoPrime init), sliding window",
    "mvit_echoprime_centre": "Echo-MViTv2-S, centre-supervised transition-aware (ours), sliding window",
    "vit_router": "ViT-Router (ours) on ViT-S features",
    "mstcn_vitfeat": "MS-TCN (official) on ViT-S features",
}
ORDER = ["b0", "vit_frame", "b8", "b6", "b7", "vivit_fe", "vivit_fe_mix", "vivit_fe_centre", "mvit_k400", "mvit_echoprime", "mvit_echoprime_centre",
         "b4", "b5", "mstcn_vitfeat", "vit_router"]
VIT = {"vit_frame", "vivit_fe", "vivit_fe_mix", "vivit_fe_centre", "mvit_k400", "mvit_echoprime", "mvit_echoprime_centre", "vit_router", "mstcn_vitfeat"}
# nine-code, cine-level test results of the two clip baselines as recorded in docs/REPRODUCTION.md (official code, EV9V)
NINE_CODE_BASELINES = [
    ("STFM (official, ResNet-18; 3 paper seeds)", "0.9377 ± 0.0017", "0.8988 ± 0.0029", "authors' test(): 10 key frames x 5-frame clips"),
    ("EchoViewCLIP (official, CLIP ViT-B/16)", "0.9414", "0.907", "authors' validate_: 16 frames per video"),
]


def g(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or d.get(k) is None:
            return default
        d = d[k]
    return d


def f(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def name(m: str) -> str:
    return f"**{NAMES.get(m, m)}**" if m in VIT else NAMES.get(m, m)


def ladder_tables(by: dict[str, dict]) -> str:
    rows = [m for m in ORDER if m in by] + sorted(m for m in by if m not in ORDER and m != "b_file")
    rec = [[name(m), f(g(by[m], "native", "test", "cine", "accuracy")), f(g(by[m], "native", "test", "cine", "macro_f1")),
            f(g(by[m], "native", "test", "cine", "balanced_accuracy")), f(g(by[m], "native", "test", "frame", "accuracy")),
            f(g(by[m], "native", "test", "frame", "macro_f1")), f(g(by[m], "native", "test", "stability", "fragments_per_minute_mean"), 2),
            f(g(by[m], "native", "test", "stability", "share_fragmented"))] for m in rows]
    seg = []
    for m in rows:
        c = g(by[m], "constructed", "cells", default={}); b = g(by[m], "constructed", "boundary", default={})
        seg.append([name(m), f(g(b, "0.25", "f1")), f(g(b, "0.5", "f1")), f(g(b, "1.0", "f1")), f(g(b, "0.5", "median_abs_error_s"), 2),
                    f(g(c, "same_none", "false_split_rate")), f(g(c, "same_edit", "false_split_rate")), f(g(c, "diff_none", "missed_rate")),
                    f(g(c, "diff_edit", "missed_rate")), f(g(by[m], "constructed_multi", "boundary", "0.5", "f1"))])
    rout = []
    for m in rows:
        pol = g(by[m], "policy", "by_target", default={})
        rout.append([name(m), f(g(pol, "0.05", "tau")), f(g(by[m], "constructed", "routing", "coverage")), f(g(by[m], "constructed", "routing", "risk")),
                     f(g(pol, "0.01", "tau")), f(g(pol, "0.01", "coverage")), f(g(by[m], "constructed_multi", "routing", "coverage")),
                     f(g(by[m], "constructed_multi", "routing", "risk"))])
    return ("## Recognition (800 untouched test clips, five-family task)\n\n" +
            table(["model", "clip acc", "clip macro-F1", "balanced acc", "frame acc", "frame macro-F1", "fragments / min", "share fragmented"], rec) +
            "\n## Segmentation (240 two-fragment + 120 multi-fragment test streams)\n\n" +
            table(["model", "boundary F1 @0.25 s", "@0.5 s", "@1.0 s", "median timing error (s)", "false split same/none", "same/edit",
                   "missed diff/none", "diff/edit", "multi-fragment boundary F1 @0.5 s"], seg) +
            "\n## Selective routing (thresholds chosen on validation at 5% target risk, frozen)\n\n" +
            table(["model", "τ @5%", "test coverage @5%", "achieved risk", "τ @1%", "val coverage @1%", "multi-fragment coverage",
                   "multi-fragment risk"], rout))


def _ms(vals: list[float], nd: int = 4) -> str:
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.{nd}f}"
    import statistics
    return f"{statistics.mean(vals):.{nd}f} ± {statistics.stdev(vals):.{nd}f}"


def _by_base(d: dict) -> dict[str, list]:
    out: dict[str, list] = {}
    for k in sorted(d, key=lambda k: (k.split("@")[0], k)):
        out.setdefault(k.split("@")[0], []).append(d[k])
    return out


def nine_code_table(video_level: dict[str, dict]) -> str:
    rows = [[n, a, fm, "—", "—", p] for n, a, fm, p in NINE_CODE_BASELINES]
    groups = _by_base(video_level)
    for m in [m for m in ORDER if m in groups] + sorted(m for m in groups if m not in ORDER):
        runs = groups[m]
        acc = [g(r, "test", "native_task", "accuracy") for r in runs]; f1 = [g(r, "test", "native_task", "macro_f1") for r in runs]
        va = [g(r, "validation", "native_task", "accuracy") for r in runs]
        rows.append([name(m), _ms(acc), _ms(f1), _ms(va), str(len(runs)), "mean softmax over 1.6 s windows (stride 0.5 s) covering the cine"])
    return ("## Nine-code cine-level classification (official EV9V test split, 888 cines)\n\n" +
            table(["model", "test acc", "test macro-F1", "val acc", "seeds", "cine aggregation"], rows))


def seed_table(by: dict[str, dict]) -> str:
    groups = _by_base({k: v for k, v in by.items() if k != "b_file"})
    rows = []
    for m in [m for m in ORDER if m in groups and len(groups[m]) > 1]:
        rs = groups[m]
        def col(*keys):
            return _ms([g(r, *keys) for r in rs if g(r, *keys) is not None], 3)
        rows.append([name(m), str(len(rs)), col("native", "test", "cine", "macro_f1"), col("native", "test", "cine", "balanced_accuracy"),
                     col("constructed", "boundary", "0.5", "f1"), col("constructed", "routing", "coverage"), col("constructed", "routing", "risk")])
    if not rows:
        return ""
    return ("\n## Seed robustness (ladder, mean ± std over training seeds)\n\n" +
            table(["model", "seeds", "clip macro-F1", "balanced acc", "boundary F1 @0.5 s", "coverage @5%", "achieved risk"], rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True); ap.add_argument("--video-level", default="")
    ap.add_argument("--out", required=True); ap.add_argument("--title", default="Vision-Transformer family on EV9V")
    a = ap.parse_args()
    by = {json.loads(p.read_text())["method_id"]: json.loads(p.read_text()) for p in sorted(Path(a.run_dir, "metrics").glob("*.json"))}
    vl = {}
    for kv in filter(None, a.video_level.split(",")):
        k, v = kv.split("=", 1); vl[k] = json.loads(Path(v).read_text())
    md = f"# {a.title}\n\nLadder run `{Path(a.run_dir).name}` (identical streams, thresholds and metrics for every row).\n\n"
    base = {k: v for k, v in by.items() if "@" not in k}
    md += (nine_code_table(vl) + "\n" if vl else "") + ladder_tables(base) + seed_table(by)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(md)
    print(md)


if __name__ == "__main__":
    main()
