#!/usr/bin/env python
"""Does "A5C seen at some point" separate A5C from A4C clips better than averaging over the clip?

Uses the ResNet-18 frame classifier (B0) probabilities of every A4C / A5C clip. Clip statistics of P(A5C): mean (what the
temporal models effectively do), share of frames above 0.5, longest run above 0.5 (seconds), mean of the top 30% of
frames, max. Reports ROC AUC on validation and test, then picks one threshold per statistic on validation (best
balanced accuracy over A4C vs A5C) and applies it frozen to test: A5C and A4C recall, and how many of the A5C test clips
misread by at least 4 of 6 models (scripts/a5c_analysis.py) it recovers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import base_parser, load_cfg  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from echo_routing.config import load_paths  # noqa: E402
from echo_routing.features.cache import load_video_features, read_index, variant_dir  # noqa: E402

A4C, A5C = 2, 3


def longest_run(x: np.ndarray) -> int:
    best = cur = 0
    for v in x:
        cur = cur + 1 if v else 0; best = max(best, cur)
    return best


def stats(p: np.ndarray) -> dict[str, float]:
    a = p[:, A5C]; k = max(1, int(np.ceil(0.3 * a.size)))
    return {"mean": float(a.mean()), "share_above_0.5": float((a > 0.5).mean()), "longest_run_s": longest_run(a > 0.5) / 10.0,
            "top30_mean": float(np.sort(a)[-k:].mean()), "max": float(a.max())}


def main() -> None:
    ap = base_parser("A5C aggregation check")
    ap.add_argument("--ckpt-hash", required=True); ap.add_argument("--hard", required=True, help="analysis.json from a5c_analysis.py")
    args = ap.parse_args(); load_cfg(args); paths = load_paths()
    idx = read_index(variant_dir(paths.cache_root, args.ckpt_hash, "orig"))
    hard = {r["video_id"] for r in json.loads(Path(args.hard).read_text())["a5c_clips"] if r["n_wrong"] >= 4}
    data = {}
    for split in ("validation", "test"):
        sub = idx[(idx["split"] == split) & idx["label_index"].isin([A4C, A5C])]
        rows = [(r.video_id, int(r.label_index) == A5C,
                 stats(load_video_features(variant_dir(paths.cache_root, args.ckpt_hash, "orig") / f"{r.video_id}.npz").prob))
                for r in sub.itertuples(index=False)]
        data[split] = rows
    out = {}
    for s in ("mean", "share_above_0.5", "longest_run_s", "top30_mean", "max"):
        yv = np.array([y for _, y, _ in data["validation"]]); xv = np.array([st[s] for _, _, st in data["validation"]])
        yt = np.array([y for _, y, _ in data["test"]]); xt = np.array([st[s] for _, _, st in data["test"]])
        cands = np.unique(xv)
        bal = [0.5 * (((xv >= c) & yv).sum() / yv.sum() + ((xv < c) & ~yv).sum() / (~yv).sum()) for c in cands]
        thr = float(cands[int(np.argmax(bal))])
        pred = xt >= thr
        ids = [v for v, _, _ in data["test"]]
        out[s] = {"auc_val": round(float(roc_auc_score(yv, xv)), 3), "auc_test": round(float(roc_auc_score(yt, xt)), 3),
                  "threshold_from_val": round(thr, 3),
                  "test_a5c_recall": round(float((pred & yt).sum() / yt.sum()), 3),
                  "test_a4c_recall": round(float((~pred & ~yt).sum() / (~yt).sum()), 3),
                  "test_a4c_called_a5c": int((pred & ~yt).sum()), "test_a4c_clips": int((~yt).sum()),
                  "hard_a5c_recovered": f"{sum(1 for v, p in zip(ids, pred) if v in hard and p)}/{len(hard)}"}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
