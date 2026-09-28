#!/usr/bin/env python
"""Tables for routing policy 2 and the segment-level fusions of one ladder run (after scripts/bootstrap_ladder.py).

Policy 2: each method's segments scored with its own pipeline's frame-level evidence (registry.FRAME_STAGE), one
threshold per predicted class at the 95% upper bound of cine-resampled validation risk, selected on the multi-fragment
validation bank. Fusions 'SEG+CLS~a': SEG's boundaries, segment labels from (1 - a) * SEG + a * CLS probabilities.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

A4C, A5C = 2, 3
NAMES = {"b0": "ResNet-18 (B0)", "b1": "B1 smoothing", "b2": "B2 HMM", "b3": "B3 JS divergence", "b4": "MS-TCN", "b5": "ASFormer",
         "b6": "STFM", "b7": "EchoViewCLIP", "b8": "EchoPrime", "vit_frame": "ViT-S (D1)", "vivit_fe_centre": "Factorised ViT + centre",
         "mvit_echoprime_centre": "Echo-MViT + centre", "mstcn_vitfeat": "MS-TCN on ViT-S", "vit_router": "ViT-Router (D4)",
         "b4+b7~0.5": "MS-TCN + EchoViewCLIP, α 0.5", "b4+b7~1": "MS-TCN + EchoViewCLIP, α 1",
         "b4+mvit_echoprime_centre~0.5": "MS-TCN + Echo-MViT, α 0.5", "b4+mvit_echoprime_centre~1": "MS-TCN + Echo-MViT, α 1"}
MAIN = ["b0", "b1", "b2", "b3", "b8", "vit_frame", "b6", "b7", "vivit_fe_centre", "mvit_echoprime_centre", "b4", "b5", "mstcn_vitfeat",
        "vit_router"]
FUSION = ["b4", "b7", "mvit_echoprime_centre", "b4+b7~0.5", "b4+b7~1", "b4+mvit_echoprime_centre~0.5", "b4+mvit_echoprime_centre~1"]


def g(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or d.get(k) is None:
            return None
        d = d[k]
    return d


def ce(e: dict | None, nd: int = 3) -> str:
    """estimate (95% interval) from a bootstrap entry."""
    if not e or e.get("estimate") is None:
        return "—"
    lo, hi = e["ci95"]
    return f"{e['estimate']:.{nd}f} ({lo:.{nd}f}–{hi:.{nd}f})" if lo is not None else f"{e['estimate']:.{nd}f}"


def route(bs: dict, key: str) -> str:
    cov, risk = bs.get(f"coverage|{key}"), bs.get(f"risk|{key}")
    if not cov or cov.get("estimate") is None:
        return "—"
    return f"{cov['estimate']:.2f} / {ce(risk)}"


def class_f1(conf: list[list[int]], c: int) -> float:
    tp = conf[c][c]; row = sum(conf[c]); col = sum(r[c] for r in conf)
    return 2 * tp / (row + col) if row + col else 0.0


def a4c_a5c_found(frag: pd.DataFrame) -> tuple[int, float]:
    """Share of A4C <-> A5C view changes with a predicted change within 0.5 s (4-8 fragment test streams)."""
    n = found = 0
    for _, s in frag.groupby("stream", sort=False):
        labels = s["label"].tolist()
        for i, r in enumerate(s.itertuples(index=False)):
            if r.t_true and {labels[i - 1], labels[i]} == {A4C, A5C}:
                n += 1; found += 1 - r.missed
    return n, (found / n if n else float("nan"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Policy-2 and fusion tables")
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run = Path(args.run_dir)
    met = {p.name[:-5]: json.loads(p.read_text()) for p in (run / "metrics").glob("*.json")}
    bs = json.loads((run / "reports" / "bootstrap.json").read_text())["banks"]["multi_test"]["methods"]
    out = [f"Run `{run.name}`. 4–8 fragment test streams; error intervals resample cines. Coverage / actual error.", "",
           "### Routing: ladder rule versus policy 2", "",
           "| model | evidence scored | ladder rule, 5% | policy 2, 5% | policy 2, 1% |", "|---|---|---|---|---|"]
    held = {"0.05": 0, "0.01": 0}; n = 0
    for m in sorted(met):
        if m == "b_file" or "policy2" not in met[m]:
            continue
        n += 1
        for t in held:
            r = g(met[m], "constructed_multi", "routing_by", f"p2@{t}", "risk")
            held[t] += int(r is not None and r <= float(t))
    for m in MAIN + [f for f in FUSION if f not in MAIN]:
        if m not in met or m not in bs:
            continue
        out.append(f"| {NAMES.get(m, m)} | {met[m]['policy2']['evidence']} | {route(bs[m], 'pair@0.05')} | "
                   f"{route(bs[m], 'p2@0.05')} | {route(bs[m], 'p2@0.01')} |")
    out += ["", f"Policy 2 keeps the test error at or below the target for {held['0.05']}/{n} methods at 5% and "
                f"{held['0.01']}/{n} at 1% (point estimates, all methods including seeds and fusions).", "",
            "### Fusion: MS-TCN boundaries with clip-level labels", "",
            "| model | val clip macro-F1 | test clip macro-F1 | A4C F1 | A5C F1 | change F1, 4–8 clips | A4C↔A5C found | "
            "policy 2, 5% | policy 2, 1% |", "|---|---|---|---|---|---|---|---|---|"]
    for m in FUSION:
        if m not in met:
            continue
        conf = g(met[m], "native", "test", "cine", "confusion")
        frag = pd.read_csv(run / "fragments" / f"{m}.multi_test.csv")
        k, found = a4c_a5c_found(frag)
        out.append(f"| {NAMES.get(m, m)} | {g(met[m], 'native', 'validation', 'cine', 'macro_f1'):.3f} | "
                   f"{g(met[m], 'native', 'test', 'cine', 'macro_f1'):.3f} | {class_f1(conf, A4C):.3f} | {class_f1(conf, A5C):.3f} | "
                   f"{ce(bs[m]['boundary_f1'])} | {found:.2f} (n={k}) | {route(bs[m], 'p2@0.05')} | {route(bs[m], 'p2@0.01')} |")
    print("\n".join(out))


if __name__ == "__main__":
    main()
