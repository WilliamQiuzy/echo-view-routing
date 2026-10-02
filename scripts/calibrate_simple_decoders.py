#!/usr/bin/env python
"""Tune B1 smoothing, the B2 HMM and B3 change points on the calibration half (2026-10-02 plan, Task 4).

All decoders read the same encoder probabilities as the learned heads (feature manifest cache). The grids are fixed
in configs/experiments/model_development_v1.yaml (decode_grids). B3 thresholds are quantiles of the JS scores over
the calibration-half native clips for each window, as for the audited setting. Each decoder's main point follows the
shared rule (native macro-F1 floor from the encoder's frame-majority baseline, multi boundary recall >= 0.95, highest
multi segment F1@50, ties to fewer native cuts); its stability point is the fewest native cuts under the same
constraints. The selected settings are then applied unchanged to the selection half. Reference rows: B0 argmax, the
default B1/B2/B3 settings and the audited B3. Writes grid_calibration.csv, selected.json and validation_metrics.json.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import base_parser, load_cfg, new_run_dir  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from echo_routing.config import load_paths  # noqa: E402
from echo_routing.evaluate.dev_metrics import evaluate_predictions, select_operating_point  # noqa: E402
from echo_routing.temporal.baselines.b3_js_divergence import boundary_scores  # noqa: E402
from echo_routing.temporal.baselines.registry import AUDITED_B3, decode_with_prob  # noqa: E402
from echo_routing.temporal.dev_run import HALVES, Inputs  # noqa: E402

FLAT = ("native_macro_f1", "segment_f1_50_multi", "boundary_recall_multi", "boundary_f1_multi", "native_internal_cuts_per_clip",
        "native_internal_cuts_per_min", "same_class_cut_rate_multi", "segment_f1_50_pairs", "multi_frame_accuracy")


def probs_of(ev, store) -> dict[str, np.ndarray]:
    out = {s.seq_id: store.get(s.seq_id, "orig").prob for s in ev.natives}
    out.update({s.recipe.recipe_id: s.prob for s in (*ev.pairs, *ev.multi)})
    return out


def decode_all(method: str, params: dict, probs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    cfg = {"decode": {method: params}}
    return {k: decode_with_prob(method, p, cfg)[0] for k, p in probs.items()}


def grid(cfg: dict, cal_native_probs: list[np.ndarray]) -> list[tuple[str, dict]]:
    g = cfg["decode_grids"]; out = []
    for w, r in itertools.product(g["b1"]["window"], g["b1"]["min_run"]):
        out.append(("b1", {"window": int(w), "min_run": int(r), "mode": "prob"}))
    out += [("b2", {"stay": float(s)}) for s in g["b2"]["stay"]]
    for h in g["b3"]["half_window"]:
        scores = np.concatenate([boundary_scores(p, int(h))[1:] for p in cal_native_probs])
        for q, nms in itertools.product(g["b3"]["threshold_quantile"], g["b3"]["nms_radius"]):
            out.append(("b3", {"half_window": int(h), "threshold": float(np.quantile(scores, float(q))), "nms_radius": int(nms),
                               "threshold_quantile": float(q)}))
    return out


def main() -> None:
    ap = base_parser("Calibrate simple decoders on the calibration half")
    ap.add_argument("--feature-manifest", required=True); ap.add_argument("--protocol", required=True)
    args = ap.parse_args(); cfg = load_cfg(args); paths = load_paths()
    inputs = Inputs(cfg, Path(args.protocol), Path(args.feature_manifest), paths.cache_root)
    hz = float(cfg["protocol"]["hz"]); sc = cfg["selection"]; tol_s = float(sc["boundary_tolerance_s"])
    probs = {h: probs_of(inputs.halves[h], inputs.store) for h in HALVES}
    base = {h: inputs.frame_majority(inputs.halves[h], hz, tol_s) for h in HALVES}
    floor = base["calibration"]["native_macro_f1"] - float(sc["native_f1_tolerance"])
    cal_native = [probs["calibration"][s.seq_id] for s in inputs.halves["calibration"].natives]
    rows = []
    for i, (method, params) in enumerate(grid(cfg, cal_native)):
        m, _ = evaluate_predictions(decode_all(method, params, probs["calibration"]), inputs.halves["calibration"], hz, tol_s)
        rows.append({"order": i, "method": method, "params": params, **{k: m[k] for k in FLAT}})
        print(f"{method} {params}: native F1 {m['native_macro_f1']:.4f} segF1 {m['segment_f1_50_multi']:.4f} "
              f"rec {m['boundary_recall_multi']:.3f} cuts {m['native_internal_cuts_per_clip']:.3f}", flush=True)
    run_dir = new_run_dir(cfg, "md_simple_decoders")
    pd.DataFrame([{**r, "params": json.dumps(r["params"])} for r in rows]).to_csv(run_dir / "grid_calibration.csv", index=False)
    selected = {}
    for method in ("b1", "b2", "b3"):
        op = select_operating_point([r for r in rows if r["method"] == method], floor, float(sc["boundary_recall_min"]),
                                    float(sc["tie_margin"]))
        selected[method] = {"status": op["status"], "main": op["main"]["params"],
                            "stability": op["stability"]["params"] if op["stability"] else None}
    refs = {"B0 argmax": ("b0", {}), "B1 default": ("b1", dict(cfg["decode"]["b1"])), "B2 default": ("b2", dict(cfg["decode"]["b2"])),
            "B3 default": ("b3", dict(cfg["decode"]["b3"])), "B3 audited": ("b3", dict(AUDITED_B3))}
    for method in ("b1", "b2", "b3"):
        refs[f"{method.upper()} tuned (main)"] = (method, selected[method]["main"])
        if selected[method]["stability"] is not None:
            refs[f"{method.upper()} tuned (stability)"] = (method, selected[method]["stability"])
    results = {}
    for name, (method, params) in refs.items():
        results[name] = {"method": method, "params": params}
        for h in HALVES:
            m, tables = evaluate_predictions(decode_all(method, params, probs[h]), inputs.halves[h], hz, tol_s)
            results[name][h] = m
            tag = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            for tname, df in tables.items():
                df.to_csv(run_dir / f"rows_{tag}_{h}_{tname}.csv", index=False)
    (run_dir / "selected.json").write_text(json.dumps({"floor_calibration": floor, "rule": dict(sc), "selected": selected}, indent=1))
    (run_dir / "validation_metrics.json").write_text(json.dumps({"methods": results, "encoder_frame_majority": base,
                                                                 "feature_manifest": args.feature_manifest, "protocol": args.protocol},
                                                                indent=1, default=str))
    print(json.dumps({"run_dir": str(run_dir), "selected": selected}, indent=1))
    for name, r in results.items():
        c = r["calibration"]; s = r["selection"]
        print(f"{name:28s} cal: F1 {c['native_macro_f1']:.4f} seg {c['segment_f1_50_multi']:.4f} rec {c['boundary_recall_multi']:.3f} "
              f"cuts {c['native_internal_cuts_per_clip']:.3f} | sel: F1 {s['native_macro_f1']:.4f} seg {s['segment_f1_50_multi']:.4f} "
              f"rec {s['boundary_recall_multi']:.3f} cuts {s['native_internal_cuts_per_clip']:.3f}")


if __name__ == "__main__":
    main()
