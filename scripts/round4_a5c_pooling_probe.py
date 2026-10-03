#!/usr/bin/env python
"""Round 4 probe: does keeping intermittent A5C evidence help? (decoding only, no training; added after the inventory,
so it is not part of configs/experiments/failure_analysis_v1.yaml).

The inventory showed that in A5C units which B3 and ASRF call A4C, the frozen classifier gives A5C on part of the
samples (E0: median 90th percentile of p(A5C) 0.92, a third of the samples above 0.5), while units that are A4C rarely
do (2% with a 90th percentile above 0.8). Rule: every predicted segment labelled A4C whose 90th percentile of the
frozen classifier's p(A5C) over its own samples reaches tau is relabelled A5C; nothing else changes. It applies to
every sequence (native clips, pairs and multi-fragment streams) and never sees reference labels or limits. tau comes
from {0.50, 0.55, ..., 0.95} and is chosen per system on the calibration half by the shared rule (native floor, recall
0.95, ties under 0.002 broken by fewer cuts, then the fallback order), then applied unchanged to the selection half.
Systems: B3 tuned and the three ASRF seeds on E0 and E1. Both halves are development data; no test split is read.
Writes probe.json and probe.md into a new run directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg, new_run_dir  # noqa: E402
from scripts.round4_failure_inventory import b3  # noqa: E402

import numpy as np  # noqa: E402

from echo_routing.config import load_paths  # noqa: E402
from echo_routing.evaluate.dev_metrics import evaluate_predictions, select_operating_point  # noqa: E402
from echo_routing.temporal.dev_run import HALVES, Inputs  # noqa: E402
from echo_routing.temporal.segments import labels_to_segments  # noqa: E402

A4C, A5C, QUANTILE = 2, 3, 0.9
TAUS = [round(0.5 + 0.05 * i, 2) for i in range(10)]
FLAT = ("native_macro_f1", "segment_f1_50_multi", "boundary_recall_multi", "native_internal_cuts_per_clip")


def relabel(pred: np.ndarray, prob: np.ndarray, tau: float | None) -> np.ndarray:
    """Predicted A4C segments with strong intermittent A5C evidence become A5C (tau None: unchanged)."""
    out = np.asarray(pred, int).copy()
    if tau is None:
        return out
    for s in labels_to_segments(out):
        if s.label == A4C and np.quantile(prob[s.start:s.end, A5C], QUANTILE) >= tau:
            out[s.start:s.end] = A5C
    return out


def sequences(inputs: Inputs, half: str) -> dict[str, np.ndarray]:
    """Sequence id -> the frozen classifier's probabilities (the same arrays every decoder uses)."""
    ev = inputs.halves[half]
    return {**{s.seq_id: inputs.store.get(s.seq_id, "orig").prob for s in ev.natives},
            **{s.recipe.recipe_id: s.prob for s in (*ev.pairs, *ev.multi)}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True); ap.add_argument("--protocol", required=True)
    ap.add_argument("-c", "--config", default="configs/experiments/failure_analysis_v1.yaml")
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); paths = load_paths(); spec = json.loads(Path(a.spec).read_text())
    sc = cfg["selection"]; hz, tol = float(cfg["protocol"]["hz"]), float(sc["boundary_tolerance_s"]); results = {}
    for enc, e in spec["encoders"].items():
        inputs = Inputs(cfg, Path(a.protocol), Path(e["feature_manifest"]), paths.cache_root)
        floor = inputs.frame_majority(inputs.halves["calibration"], hz, tol)["native_macro_f1"] - float(sc["native_f1_tolerance"])
        probs = {h: sequences(inputs, h) for h in HALVES}
        b3_params = json.loads((Path(e["simple_run"]) / "selected.json").read_text())["selected"]["b3"]["main"]
        systems = {"B3": {h: {k: b3(p, b3_params) for k, p in probs[h].items()} for h in HALVES}}
        for k, rd in sorted(e["asrf"].items()):
            systems[f"ASRF_s{k}"] = {h: (lambda z: {sid: z[sid].astype(int) for sid in probs[h]})(np.load(Path(rd) / f"predictions_{h}.npz")) for h in HALVES}
        for name, preds in systems.items():
            def score(h: str, tau: float | None) -> dict:
                m, _ = evaluate_predictions({k: relabel(preds[h][k], probs[h][k], tau) for k in probs[h]}, inputs.halves[h], hz, tol)
                return {k: m[k] for k in FLAT}
            recs = [{"order": i, "params": {"tau": t}, **score("calibration", t)} for i, t in enumerate(TAUS)]
            op = select_operating_point(recs, floor, float(sc["boundary_recall_min"]), float(sc["tie_margin"])); tau = op["main"]["params"]["tau"]
            results[f"{enc}|{name}"] = {"tau": tau, "status": op["status"], "floor_calibration": floor,
                                        **{h: {"default": score(h, None), "probe": score(h, tau)} for h in HALVES}}
            r = results[f"{enc}|{name}"]
            print(f"{enc} {name}: tau {tau} ({op['status']}) | " + " | ".join(
                f"{h} segF1 {r[h]['default']['segment_f1_50_multi']:.4f} -> {r[h]['probe']['segment_f1_50_multi']:.4f}, native F1 "
                f"{r[h]['default']['native_macro_f1']:.4f} -> {r[h]['probe']['native_macro_f1']:.4f}" for h in HALVES), flush=True)
        del inputs
    run_dir = new_run_dir(cfg, "md4_a5c_pooling_probe"); (run_dir / "probe.json").write_text(json.dumps(results, indent=1))
    md = ["# A5C evidence probe (round 4; decoding only; added after the inventory)", "",
          "| input, system | tau (status) | half | segment F1@50 default -> probe | native macro-F1 default -> probe | recall default -> probe |",
          "|---|---|---|---|---|---|"]
    for k, r in results.items():
        for h in HALVES:
            d, p = r[h]["default"], r[h]["probe"]
            md.append(f"| {k.replace('|', ', ')} | {r['tau']} ({r['status']}) | {h} | {d['segment_f1_50_multi']:.3f} -> {p['segment_f1_50_multi']:.3f} | "
                      f"{d['native_macro_f1']:.3f} -> {p['native_macro_f1']:.3f} | {d['boundary_recall_multi']:.3f} -> {p['boundary_recall_multi']:.3f} |")
    (run_dir / "probe.md").write_text("\n".join(md) + "\n"); print("\n".join(md)); print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
