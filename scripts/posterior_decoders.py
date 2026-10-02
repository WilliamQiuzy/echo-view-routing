#!/usr/bin/env python
"""Calibrated decoders on top of learned heads (diagnostic of round 1, 2026-10-02 plan, Task 4).

For finished runs, recomputes the selected checkpoint's class posteriors (and boundary probabilities where the head
has them) on both validation halves, then decodes with: per-sample argmax (the default), the sticky HMM and the B3
change-point decoder on the posteriors (same grids as the simple decoders, B3 thresholds from quantiles of the
calibration natives' scores), and, for heads with a boundary output, boundary segments (peaks of b above a
threshold, NMS radius 5, each segment labelled by its mean posterior). Each decoder family's setting is chosen on the
calibration half by the shared rule and then applied unchanged to the selection half. Writes posterior_decoders.json
and grid.csv into a new run directory.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg, new_run_dir  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from echo_routing.config import load_paths  # noqa: E402
from echo_routing.evaluate.dev_metrics import evaluate_predictions, select_operating_point  # noqa: E402
from echo_routing.temporal.baselines.b3_js_divergence import boundary_scores, labels_from_boundaries  # noqa: E402
from echo_routing.temporal.baselines.registry import decode_with_prob  # noqa: E402
from echo_routing.temporal.dev_heads import head_from_manifest  # noqa: E402
from echo_routing.temporal.dev_run import HALVES, Inputs  # noqa: E402
from echo_routing.temporal.segments import non_max_suppression  # noqa: E402

FLAT = ("native_macro_f1", "segment_f1_50_multi", "boundary_recall_multi", "boundary_f1_multi", "native_internal_cuts_per_clip",
        "same_class_cut_rate_multi", "segment_f1_50_pairs")


def boundary_segments(prob: np.ndarray, b: np.ndarray, threshold: float, nms_radius: int) -> np.ndarray:
    peaks = non_max_suppression(np.where(np.arange(b.size) > 0, b, 0.0), threshold, nms_radius)
    return labels_from_boundaries(prob, peaks)


def decoders(cfg: dict, cal_native_probs: list[np.ndarray], has_boundary: bool) -> list[tuple[str, dict]]:
    g = cfg["decode_grids"]; out = [("argmax", {})]
    out += [("b2", {"stay": float(s)}) for s in g["b2"]["stay"]]
    for h in g["b3"]["half_window"]:
        scores = np.concatenate([boundary_scores(p, int(h))[1:] for p in cal_native_probs])
        for q in g["b3"]["threshold_quantile"]:
            out.append(("b3", {"half_window": int(h), "threshold": float(np.quantile(scores, float(q))), "nms_radius": 5,
                               "threshold_quantile": float(q)}))
    if has_boundary:
        bg = cfg["posterior_decoding"]["bseg"]
        out += [("bseg", {"threshold": float(t), "nms_radius": int(r)}) for t, r in itertools.product(bg["threshold"], bg["nms_radius"])]
    return out


def apply(family: str, params: dict, post: dict) -> dict[str, np.ndarray]:
    if family == "argmax":
        return {k: v["prob"].argmax(1) for k, v in post.items()}
    if family == "bseg":
        return {k: boundary_segments(v["prob"], v["boundary_prob"], params["threshold"], params["nms_radius"]) for k, v in post.items()}
    return {k: decode_with_prob(family, v["prob"], {"decode": {family: params}})[0] for k, v in post.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="", help="row=run_dir;row=run_dir")
    ap.add_argument("--pred-dirs", default="", help="row=dir;... prediction directories (<stream id>.npz with prob), e.g. B7")
    ap.add_argument("--feature-manifest", required=True); ap.add_argument("--protocol", required=True)
    ap.add_argument("-c", "--config", default="configs/experiments/boundary_context_v1.yaml")
    ap.add_argument("--tag", default="", help="run-name tag (default: the first row)")
    args = ap.parse_args(); args.set = []
    cfg = load_cfg(args); paths = load_paths(); sc = cfg["selection"]
    hz = float(cfg["protocol"]["hz"]); tol_s = float(sc["boundary_tolerance_s"])
    inputs = Inputs(cfg, Path(args.protocol), Path(args.feature_manifest), paths.cache_root)
    base = {h: inputs.frame_majority(inputs.halves[h], hz, tol_s) for h in HALVES}
    floor = base["calibration"]["native_macro_f1"] - float(sc["native_f1_tolerance"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = inputs.feature_dim
    first = (args.runs or args.pred_dirs).split(";")[0].partition("=")[0].strip()
    run_dir = new_run_dir(cfg, f"md_posterior_decoders_{args.tag or first}")   # distinct names for parallel runs
    results, grid_rows = {}, []
    sources = [(i, "run") for i in filter(None, args.runs.split(";"))] + [(i, "pred") for i in filter(None, args.pred_dirs.split(";"))]
    for item, kind in sources:
        row, _, rd = item.partition("="); rd = Path(rd.strip()); row = row.strip()
        if kind == "pred":   # an external system's per-stream probabilities; no boundary output
            post = {h: {s.seq_id: {"prob": np.load(rd / f"{s.seq_id}.npz")["prob"]} for s in inputs.halves[h].sequences()} for h in HALVES}
            has_b = False
        else:
            man = json.loads((rd / "run_manifest.json").read_text())
            head = head_from_manifest(man, paths.repo_root, in_dim, device)
            model = head.build().to(device); model.load_state_dict(torch.load(man["checkpoints"]["best"], map_location=device))
            post = {h: head.decode(model, inputs.halves[h].sequences(), device) for h in HALVES}
            has_b = "boundary_prob" in next(iter(post["calibration"].values())) and man.get("boundary_weight", 1.0) != 0.0
        cal_nat = [post["calibration"][s.seq_id]["prob"] for s in inputs.halves["calibration"].natives]
        # native floor: each system's own frame-level majority vote (an external system's argmax; else the encoder's)
        row_floor = floor if kind == "run" else evaluate_predictions(apply("argmax", {}, post["calibration"]), inputs.halves["calibration"],
                                                                         hz, tol_s)[0]["native_macro_f1"] - float(sc["native_f1_tolerance"])
        recs = []
        for i, (fam, params) in enumerate(decoders(cfg, cal_nat, has_b)):
            m, _ = evaluate_predictions(apply(fam, params, post["calibration"]), inputs.halves["calibration"], hz, tol_s)
            recs.append({"order": i, "family": fam, "params": params, **{k: m[k] for k in FLAT}})
        grid_rows += [{"row": row, **r, "params": json.dumps(r["params"])} for r in recs]
        results[row] = {"run_dir": str(rd), "source": kind, "floor_calibration": row_floor, "families": {}}
        for fam in sorted({r["family"] for r in recs}):
            op = select_operating_point([r for r in recs if r["family"] == fam], row_floor, float(sc["boundary_recall_min"]), float(sc["tie_margin"]))
            entry = {"status": op["status"], "params": op["main"]["params"]}
            for h in HALVES:
                m, tables = evaluate_predictions(apply(fam, op["main"]["params"], post[h]), inputs.halves[h], hz, tol_s)
                entry[h] = {k: m[k] for k in (*FLAT, "native_per_class_recall", "boundary_precision_multi", "native_internal_cuts_per_min")}
                for tname in ("native", "multi_seg", "multi_frag"):
                    tables[tname].to_csv(run_dir / f"rows_{row}_{fam}_{h}_{tname}.csv", index=False)
            results[row]["families"][fam] = entry
            c, s_ = entry["calibration"], entry["selection"]
            print(f"{row:4s} {fam:6s} {entry['status']:18s} {json.dumps(op['main']['params'])[:60]:60s} | cal F1 {c['native_macro_f1']:.4f} "
                  f"seg {c['segment_f1_50_multi']:.4f} rec {c['boundary_recall_multi']:.3f} cuts {c['native_internal_cuts_per_clip']:.3f} "
                  f"| sel F1 {s_['native_macro_f1']:.4f} seg {s_['segment_f1_50_multi']:.4f} rec {s_['boundary_recall_multi']:.3f} "
                  f"cuts {s_['native_internal_cuts_per_clip']:.3f}", flush=True)
    pd.DataFrame(grid_rows).to_csv(run_dir / "grid.csv", index=False)
    (run_dir / "posterior_decoders.json").write_text(json.dumps({"floor_calibration": floor, "results": results}, indent=1, default=str))
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
