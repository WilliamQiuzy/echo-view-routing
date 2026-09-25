#!/usr/bin/env python
"""Per-view recognition and per-view-pair segmentation for any set of ladder methods (same decoders, same streams).

Recognition (800 native test clips, clip label = majority of the decoded per-sample labels, as in the ladder):
  family  : precision / recall / F1 per five-family view, Wilson 95% interval for recall.
  code    : recall per raw EV9V code (share of that code's clips routed to the right family) — every method.
  code9   : nine-code recall per raw code for methods with nine-code outputs (argmax of the mean raw probabilities).
Segmentation (120 multi-fragment test streams, 379 view changes over all 10 view pairs, 202 same-view joins):
  per unordered view pair: share of changes with a predicted boundary within ±0.5 s, and the median timing error;
  per view: spurious cuts per minute inside that view's fragments (predicted boundaries more than 0.5 s from any
  true change); per view: false splits at same-view joins.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import base_parser, load_cfg  # noqa: E402

import numpy as np  # noqa: E402

from echo_routing.config import load_paths  # noqa: E402
from echo_routing.features.cache import load_video_features, read_index, variant_dir  # noqa: E402
from echo_routing.temporal.baselines.registry import decode_with_prob  # noqa: E402
from echo_routing.temporal.windowed import RAW9_ORDER  # noqa: E402
from echo_routing.vit.infer import assemble  # noqa: E402

FAMILY5 = ["PLAX", "PSAX", "A4C", "A5C", "SC4C"]
CODES = ["PLHLA", "PASA", "PMASA", "PMVLSA", "PPMLSA", "A4C", "A5C", "SC4C"]   # the eight codes present in family5 streams
TOL = 5   # samples (0.5 s at 10 Hz)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def boundaries(labels: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.diff(labels)) + 1


def main() -> None:
    ap = base_parser("Per-view recognition and per-pair segmentation")
    ap.add_argument("--ckpt-hash", default="79f4a41af6e3"); ap.add_argument("--plans", default="data/window_plans/79f4a41af6e3/plans.json")
    ap.add_argument("--methods", required=True); ap.add_argument("--pred-dirs", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(); cfg = load_cfg(args); paths = load_paths()
    for kv in filter(None, args.pred_dirs.split(",")):
        k, v = kv.split("=", 1); cfg.setdefault(k, {})["pred_dir"] = v
    idx = read_index(variant_dir(paths.cache_root, args.ckpt_hash, "orig"))
    fam_of = dict(zip(idx["video_id"], idx["label_index"].astype(int))); raw_of = dict(zip(idx["video_id"], idx["raw_label"]))
    plans = json.loads((paths.repo_root / args.plans).read_text())["streams"]
    b0_cache: dict = {}

    def b0_loader(v, var):
        if (v, var) not in b0_cache:
            b0_cache[(v, var)] = load_video_features(variant_dir(paths.cache_root, args.ckpt_hash, var) / f"{v}.npz")
        return b0_cache[(v, var)]

    def raw9(method: str, sid: str) -> np.ndarray | None:
        d = (cfg.get(method) or cfg.get({"b6": "stfm_windowed", "b7": "echoviewclip_windowed"}.get(method, "")) or {}).get("pred_dir")
        if not d:
            return None
        with np.load(Path(d) / f"{sid}.npz") as z:
            return z["raw"] if "raw" in z.files and z["raw"].shape[1] == 9 else None

    native = sorted(s for s, p in plans.items() if p["bank"] == "native_test")
    multi = sorted(s for s, p in plans.items() if p["bank"] == "multi_test")
    out = {"n_native": len(native), "n_multi": len(multi), "families": FAMILY5, "codes": CODES, "methods": {}}
    for m in args.methods.split(","):
        def decode(sid):
            refs = plans[sid]["refs"]; prob = assemble(refs, b0_loader, "prob")
            return decode_with_prob(m, prob, cfg, None, sid)[0]
        # ---- recognition on native clips
        conf = np.zeros((5, 5), int); by_code = defaultdict(lambda: [0, 0]); by_code9 = defaultdict(lambda: [0, 0]); has9 = True
        for sid in native:
            vid = plans[sid]["refs"][0][0]; y = fam_of[vid]; code = raw_of[vid]
            pred = int(np.bincount(decode(sid), minlength=5).argmax()); conf[y, pred] += 1
            by_code[code][0] += int(pred == y); by_code[code][1] += 1
            r = raw9(m, sid) if has9 else None
            if r is None:
                has9 = False
            else:
                p9 = RAW9_ORDER[int(r.mean(0).argmax())]
                by_code9[code][0] += int(p9 == code); by_code9[code][1] += 1
        fam = {}
        for i, f in enumerate(FAMILY5):
            tp, sup, prd = conf[i, i], conf[i].sum(), conf[:, i].sum()
            p = tp / prd if prd else 0.0; rc = tp / sup if sup else 0.0
            fam[f] = {"n": int(sup), "precision": p, "recall": rc, "f1": 2 * p * rc / (p + rc) if p + rc else 0.0, "recall_ci": wilson(int(tp), int(sup))}
        res = {"family": fam, "confusion": conf.tolist(),
               "code": {c: {"n": by_code[c][1], "recall": by_code[c][0] / by_code[c][1], "recall_ci": wilson(*by_code[c])} for c in CODES if by_code[c][1]}}
        if has9:
            res["code9"] = {c: {"n": by_code9[c][1], "recall": by_code9[c][0] / by_code9[c][1], "recall_ci": wilson(*by_code9[c])} for c in CODES if by_code9[c][1]}
        # ---- segmentation on multi-fragment streams
        pair = defaultdict(lambda: {"n": 0, "hit": 0, "err": []}); spur = defaultdict(lambda: [0, 0.0]); join = defaultdict(lambda: [0, 0])
        for sid in multi:
            refs = plans[sid]["refs"]; truth = np.array([fam_of[r[0]] for r in refs]); pred_b = boundaries(decode(sid))
            vids = [r[0] for r in refs]; joins = [i for i in range(1, len(refs)) if vids[i] != vids[i - 1]]
            sem = [j for j in joins if truth[j] != truth[j - 1]]
            for j in joins:
                if truth[j] == truth[j - 1]:
                    v = FAMILY5[truth[j]]; join[v][1] += 1; join[v][0] += int(np.any(np.abs(pred_b - j) <= TOL))
            for b in sem:
                key = "–".join(sorted((FAMILY5[truth[b - 1]], FAMILY5[truth[b]]), key=FAMILY5.index)); pair[key]["n"] += 1
                d = np.abs(pred_b - b)
                if d.size and d.min() <= TOL:
                    pair[key]["hit"] += 1; pair[key]["err"].append(float(d.min()) / 10.0)
            for b in pred_b:
                if not sem or np.min(np.abs(np.array(sem) - b)) > TOL:
                    spur[FAMILY5[truth[b]]][0] += 1
            for v, n in zip(*np.unique(truth, return_counts=True)):
                spur[FAMILY5[v]][1] += n / 600.0            # minutes of that view
        res["pairs"] = {k: {"n": v["n"], "recall": v["hit"] / v["n"], "recall_ci": wilson(v["hit"], v["n"]),
                            "median_error_s": float(np.median(v["err"])) if v["err"] else None} for k, v in sorted(pair.items())}
        res["spurious_per_min"] = {v: spur[v][0] / spur[v][1] for v in FAMILY5 if spur[v][1]}
        res["same_view_false_split"] = {v: {"n": join[v][1], "rate": join[v][0] / join[v][1]} for v in FAMILY5 if join[v][1]}
        out["methods"][m] = res
        print(m, "family F1", {f: round(fam[f]["f1"], 3) for f in FAMILY5}, "| pairs", {k: round(v["recall"], 2) for k, v in res["pairs"].items()}, flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True); Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
