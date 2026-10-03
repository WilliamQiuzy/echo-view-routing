#!/usr/bin/env python
"""Round 5, step 2: does an A4C/A5C decision that reads the distribution of frame evidence help, without new A4C errors?
(configs/experiments/evidence_pooling_v1.yaml, section decider; run only because step 1 held for E1 ASRF candidate
spans).

Per seed k (0, 1, 2): the trained_mean and trained_distribution deciders (echo_routing.temporal.evidence_decider) are
fitted on training spans only; then E1 ASRF seed k decodes every validation sequence officially, its candidate spans
(accepted boundary peaks before the class vote) are kept, and four outputs are scored on both halves, with all clips
and without the ten sensitivity clips: ASRF's default, and the mean, trained_mean and trained_distribution rules on the
same spans. Reported: the success test, paired per-seed bootstrap contrasts, and units and native clips that turn
right (A5C) or wrong (A4C) against the mean rule and against the default, by units, clips and sessions.
Writes decider.json, decider.md and per-rule rows into a new run directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg, new_run_dir  # noqa: E402
from scripts.report_model_development import paired_bootstrap  # noqa: E402
from scripts.report_model_development_round2 import category, read  # noqa: E402
from scripts.round4_failure_inventory import load_asrf  # noqa: E402
from scripts.round5_reachability import asrf_decode, inputs_of, without  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from echo_routing.compose.banks import multi_bank  # noqa: E402
from echo_routing.config import load_paths  # noqa: E402
from echo_routing.evaluate.dev_metrics import evaluate_predictions  # noqa: E402
from echo_routing.evaluate.failure_units import session_of, stream_units, unit_outcomes  # noqa: E402
from echo_routing.features.cache import read_index, variant_dir  # noqa: E402
from echo_routing.temporal.dev_run import HALVES, Inputs  # noqa: E402
from echo_routing.temporal.evidence_decider import (A4C, A5C, FEATURES, Decider, decide_spans, mean_rule, span_weights,  # noqa: E402
                                                    training_spans)

SEEDS = (0, 1, 2)
RULES = ("default", "mean", "trained_mean", "trained_distribution")
FLAT = ("segment_f1_50_multi", "boundary_recall_multi", "native_macro_f1")
METRIC = "multi segment F1@50"


def fit_deciders(inputs: Inputs, cfg: dict, paths, seed: int) -> dict:
    recipes = multi_bank(read_index(variant_dir(paths.cache_root, inputs.cache_key, "orig")), "train", cfg).recipes
    if [r.recipe_id for r in recipes] != [s.seq_id for s in inputs.train.streams]:
        raise SystemExit("training-bank recipes and streams are out of order")
    spans = training_spans([(r, s.prob) for r, s in zip(recipes, inputs.train.streams)],
                           [(v, lab, inputs.store.get(v, "orig").prob) for v, lab in inputs.train.clips], seed)
    y = np.array([lab for _, lab, _ in spans]); w = span_weights([c for *_, c in spans], y); out = {}
    for name, feat in FEATURES.items():
        d = Decider(seed).fit(np.stack([feat(p) for p, _, _ in spans]), y, w)
        out[name] = {"decider": d, "rule": d.rule(feat), "train_accuracy": float((((d.prob_a5c(np.stack([feat(p) for p, _, _ in spans])) >= 0.5)
                                                                                  == (y == A5C)) * w).sum() / w.sum()), "final_loss": d.final_loss}
    return {"models": out, "spans": int(len(spans)), "A5C": int((y == A5C).sum()), "clips": len({c for *_, c in spans})}


def outputs(seqs: dict, decoded: dict, models: dict) -> dict[str, dict]:
    out = {"default": {sid: f for sid, (f, _) in decoded.items()}}
    out["mean"] = {sid: decide_spans(f, c, seqs[sid][1], mean_rule) for sid, (f, c) in decoded.items()}
    for name, m in models.items():
        out[name] = {sid: decide_spans(f, c, seqs[sid][1], m["rule"]) for sid, (f, c) in decoded.items()}
    return out


def unit_states(preds: dict, ev) -> tuple[dict, dict]:
    """(stream, unit) -> (label, matched, clips) for multi units and ('native', clip) -> (label, majority right, [clip]);
    and the false pieces: unmatched predicted A5C segments lying inside A4C units and A4C segments inside A5C units."""
    out, pieces = {}, {"A5C pieces inside A4C units": [], "A4C pieces inside A5C units": []}
    for s in ev.multi:
        units = stream_units(s.recipe); rows, spurious = unit_outcomes(preds[s.recipe.recipe_id], units)
        out.update({(s.recipe.recipe_id, u.index): (u.label, r["matched"], [v for v, *_ in u.sources]) for u, r in zip(units, rows)})
        for x in spurious:
            host = units[x["host"]]
            for name, (lab, hst) in (("A5C pieces inside A4C units", (A5C, A4C)), ("A4C pieces inside A5C units", (A4C, A5C))):
                if x["inside"] and x["label"] == lab and host.label == hst:
                    pieces[name].append([v for v, *_ in host.sources])
    for n in ev.natives:
        lab = preds[n.seq_id]; out[("native", n.seq_id)] = (int(n.labels[0]), int(np.bincount(lab, minlength=5).argmax()) == int(n.labels[0]), [n.seq_id])
    return out, {k: {"n": len(v), "clips": len({c for x in v for c in x})} for k, v in pieces.items()}


def flips(new: dict, ref: dict) -> dict:
    """A5C turning right and A4C turning wrong, by units, clips and sessions, for stream units and native clips."""
    out = {}
    for kind, sel in (("units", lambda k: k[0] != "native"), ("natives", lambda k: k[0] == "native")):
        for name, cls, cond in (("A5C fixed", A5C, lambda a, b: a and not b), ("A4C broken", A4C, lambda a, b: b and not a)):
            ks = [k for k in new if sel(k) and new[k][0] == cls and cond(new[k][1], ref[k][1])]
            clips = Counter(c for k in ks for c in set(new[k][2]))
            ranked = sorted(clips.items(), key=lambda x: (-x[1], x[0])); top2 = {c for c, _ in ranked[:2]}   # ties by clip id
            out[f"{kind}: {name}"] = {"n": len(ks), "clips": len(clips), "sessions": len({session_of(c) for c in clips}),
                                      "top2_clips_share": (sum(bool(top2 & set(new[k][2])) for k in ks) / len(ks)) if ks else None,
                                      "top_clips": ranked[:3]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True); ap.add_argument("--protocol", required=True)
    ap.add_argument("-c", "--config", default="configs/experiments/evidence_pooling_v1.yaml")
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); paths = load_paths(); spec = json.loads(Path(a.spec).read_text())
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); sc = cfg["selection"]
    hz, tol_s = float(cfg["protocol"]["hz"]), float(sc["boundary_tolerance_s"]); clips_out = set(cfg["sensitivity"]["exclude_clips"])
    e = spec["encoders"]["E1"]; inputs = Inputs(cfg, Path(a.protocol), Path(e["feature_manifest"]), paths.cache_root)
    floors = {h: inputs.frame_majority(inputs.halves[h], hz, tol_s)["native_macro_f1"] - float(sc["native_f1_tolerance"]) for h in HALVES}
    strongest = {"selection": read(Path(spec["encoders"]["E0"]["simple_run"]), "rows_b3_tuned_main_selection"),
                 "calibration": read(Path(spec["b7_run"]), "rows_B7_b3_calibration")}
    target = {h: float(2 * t["multi_seg"].seg_matched.sum() / (t["multi_seg"].seg_pred.sum() + t["multi_seg"].seg_true.sum())) + float(cfg["success"]["segment_f1_margin"])
              for h, t in strongest.items()}
    res = {"floors": floors, "target": target, "train": {}, "metrics": {}, "flips": {}, "pieces": {}, "contrasts": {}}; tables = {}; arrays = {}
    for k in SEEDS:
        fitted = fit_deciders(inputs, cfg, paths, k); models = fitted.pop("models")
        res["train"][k] = {**fitted, **{n: {"train_accuracy": m["train_accuracy"], "final_loss": m["final_loss"]} for n, m in models.items()}}
        rd = Path(e["asrf"][str(k)]); head, model = load_asrf(rd, paths.repo_root, inputs.feature_dim, device)
        for h in HALVES:
            ev = inputs.halves[h]; red = without(ev, clips_out); seqs = inputs_of(ev, inputs.store)
            decoded = {sid: asrf_decode(head, model, f, device) for sid, (f, _, _) in seqs.items()}
            stored = np.load(rd / f"predictions_{h}.npz")
            if any(not np.array_equal(decoded[sid][0], stored[sid].astype(int)) for sid in seqs):
                raise SystemExit(f"ASRF seed {k} {h}: decoding does not reproduce the stored predictions")
            outs = outputs(seqs, decoded, models); states = {}
            for rule, preds in outs.items():
                for scope, evs in (("all", ev), ("without", red)):
                    m, t = evaluate_predictions(preds, evs, hz, tol_s)
                    res["metrics"][f"{rule}|{scope}|{h}|{k}"] = {x: m[x] for x in FLAT}; tables[(rule, scope, h, k)] = t
                    res["metrics"][f"{rule}|{scope}|{h}|{k}"]["predicted_segments"] = int(t["multi_seg"].seg_pred.sum())
                states[rule], res["pieces"][f"{rule}|{h}|{k}"] = unit_states(preds, ev)
                arrays.update({f"{rule}|{h}|{k}|{sid}": v.astype(np.int8) for sid, v in preds.items()})
            for rule in ("mean", "trained_mean", "trained_distribution"):
                for ref in ("default", "mean"):
                    if rule != ref:
                        res["flips"][f"{rule} vs {ref}|{h}|{k}"] = flips(states[rule], states[ref])
            res["flips"][f"trained_distribution vs trained_mean|{h}|{k}"] = flips(states["trained_distribution"], states["trained_mean"])
            print(f"seed {k} {h}: " + "; ".join(f"{r} {res['metrics'][f'{r}|all|{h}|{k}']['segment_f1_50_multi']:.4f}" for r in RULES), flush=True)
        del head, model; torch.cuda.empty_cache()
    pairs = [("trained_distribution", "trained_mean"), ("trained_distribution", "mean"), ("trained_mean", "mean"),
             *((r, "default") for r in ("mean", "trained_mean", "trained_distribution"))]
    def contrast(new_tables, ref_tables) -> dict:
        per = {k: paired_bootstrap(new_tables(k), ref_tables(k))[METRIC] for k in SEEDS}
        return {"per_seed": per, "category": category([per[k]["ci95"] for k in SEEDS], len(SEEDS))}

    for scope in ("all", "without"):
        for new, ref in pairs:
            for h in HALVES:
                res["contrasts"][f"{new} - {ref}|{scope}|{h}"] = contrast(lambda k: tables[(new, scope, h, k)], lambda k: tables[(ref, scope, h, k)])
    for r in RULES:          # the strongest previous configuration of each half is scored on all clips only
        for h in HALVES:
            res["contrasts"][f"{r} - strongest previous|all|{h}"] = contrast(lambda k: tables[(r, "all", h, k)], lambda k: strongest[h])
    res["success"] = {}
    for r in RULES:
        ok = {}
        for h in HALVES:
            ms = [res["metrics"][f"{r}|all|{h}|{k}"] for k in SEEDS]; mean = float(np.mean([m["segment_f1_50_multi"] for m in ms]))
            ok[h] = {"mean_segment_f1_50": mean, "target": target[h], "min_recall": min(m["boundary_recall_multi"] for m in ms),
                     "min_native_f1": min(m["native_macro_f1"] for m in ms), "floor": floors[h],
                     "passes": bool(mean >= target[h] and min(m["boundary_recall_multi"] for m in ms) >= 0.95 and min(m["native_macro_f1"] for m in ms) >= floors[h])}
        res["success"][r] = ok
    run_dir = new_run_dir(cfg, "md5_evidence_decider")
    for (rule, scope, h, k), t in tables.items():
        if scope == "all":
            for name in ("native", "multi_seg", "multi_frag"):
                t[name].to_csv(run_dir / f"rows_{rule}_s{k}_{h}_{name}.csv", index=False)
    np.savez_compressed(run_dir / "predictions.npz", **arrays)
    (run_dir / "decider.json").write_text(json.dumps(res, indent=1, default=str)); print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
