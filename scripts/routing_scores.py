#!/usr/bin/env python
"""Routing confidence and threshold rules on the 4-8 fragment streams (step (a) after the 2026-09-26 calibration study).

Segments always come from each method's own decoded labels (identical to the ladder); only the confidence changes.
  score variants : own_p10 (ladder default: 10th percentile of the segment class x agreement), own_mean,
                   ownT_p10 (own probabilities temperature-scaled on multi_validation), frame_p10 and frame_mean
                   (ResNet-18 frame probabilities inside the method's segments), min_own_frame (min of own_p10, frame_p10)
  threshold rules: point (ladder rule), ucb (95% upper bound of a cine-resampled risk), point_class, ucb_class
                   (one threshold per predicted class); all selected on multi_validation and frozen
  test endpoints : coverage and actual error at 5% and 1% on multi_test with cine-level 95% intervals, the same on
                   pairs_test, AURC and hindsight coverage on multi_test, share of routed error inside A5C fragments
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import base_parser, load_cfg, new_run_dir  # noqa: E402

import numpy as np  # noqa: E402

from echo_routing.compose.banks import multi_bank, pair_bank, render_bank  # noqa: E402
from echo_routing.config import load_paths  # noqa: E402
from echo_routing.evaluate.bootstrap import resample_weights  # noqa: E402
from echo_routing.evaluate.ladder import _per_sample_routing  # noqa: E402
from echo_routing.evaluate.risk_control import (accept_mask, aurc, cine_counts, fit_temperature, oracle_coverage,  # noqa: E402
                                                select_classwise, select_threshold_ucb, temperature_scale, threshold_grid)
from echo_routing.features.cache import load_video_features, read_index, variant_dir  # noqa: E402
from echo_routing.temporal.baselines.registry import decode_with_prob  # noqa: E402
from echo_routing.temporal.decode_select import select_threshold  # noqa: E402

TARGETS = (0.05, 0.01)
RULES = ("point", "ucb", "point_class", "ucb_class")
A5C = 3


def collect(method: str, streams, cfg: dict, cfg_mean: dict, temp: float | None) -> dict:
    """Concatenated per-sample arrays for one bank: scores per variant, wrong, predicted label, truth, cine id."""
    out = {k: [] for k in ("own_p10", "own_mean", "ownT_p10", "frame_p10", "frame_mean", "wrong", "pred", "truth", "cine", "own")}
    for s in streams:
        labels, own = decode_with_prob(method, s.prob, cfg, s.feat, s.recipe.recipe_id)
        sc, wr, _ = _per_sample_routing(own, labels, s.labels, cfg)
        out["own_p10"].append(sc); out["wrong"].append(wr)
        out["own_mean"].append(_per_sample_routing(own, labels, s.labels, cfg_mean)[0])
        if temp is not None:
            out["ownT_p10"].append(_per_sample_routing(temperature_scale(own, temp), labels, s.labels, cfg)[0])
        out["frame_p10"].append(_per_sample_routing(s.prob, labels, s.labels, cfg)[0])
        out["frame_mean"].append(_per_sample_routing(s.prob, labels, s.labels, cfg_mean)[0])
        out["pred"].append(labels); out["truth"].append(s.labels); out["own"].append(own)
        out["cine"].append(np.concatenate([np.full(fr.end - fr.start, fr.video_id, dtype=object) for fr in s.recipe.fragments]))
    d = {k: np.concatenate(v) for k, v in out.items() if v}
    d["min_own_frame"] = np.minimum(d["own_p10"], d["frame_p10"])
    return d


def ci_by_cine(acc: np.ndarray, wrong: np.ndarray, cine: np.ndarray, ids: np.ndarray, weights: np.ndarray) -> dict:
    """Coverage and risk with cine-resampled 95% intervals for one accepted mask."""
    pos = {c: i for i, c in enumerate(ids.tolist())}
    inv = np.array([pos[c] for c in cine.tolist()], dtype=int)
    n = np.bincount(inv, minlength=ids.size).astype(float)
    a = np.bincount(inv, weights=acc, minlength=ids.size); w = np.bincount(inv, weights=acc & wrong, minlength=ids.size)
    wn, wa, ww = weights @ n, weights @ a, weights @ w
    with np.errstate(invalid="ignore", divide="ignore"):
        cov_rep, risk_rep = wa / wn, np.where(wa > 0, ww / np.where(wa > 0, wa, 1), np.nan)
    risk_rep = risk_rep[~np.isnan(risk_rep)]
    return {"coverage": float(a.sum() / n.sum()), "risk": float(w.sum() / a.sum()) if a.sum() else None,
            "coverage_ci": [float(x) for x in np.percentile(cov_rep, [2.5, 97.5])],
            "risk_ci": [float(x) for x in np.percentile(risk_rep, [2.5, 97.5])] if risk_rep.size else None}


def evaluate(method: str, banks: dict, cfg: dict, cfg_mean: dict, w_val: np.ndarray, w_test: np.ndarray,
             ids_val: np.ndarray, ids_test: np.ndarray, rng: np.random.Generator) -> dict:
    val0 = collect(method, banks["multi_validation"], cfg, cfg_mean, None)
    temp = fit_temperature(val0["own"], val0["truth"])
    data = {b: collect(method, banks[b], cfg, cfg_mean, temp) for b in ("multi_validation", "multi_test", "pairs_test")}
    val, test, pairs = data["multi_validation"], data["multi_test"], data["pairs_test"]
    res = {"temperature": temp, "variants": {}}
    for v in ("own_p10", "own_mean", "ownT_p10", "frame_p10", "frame_mean", "min_own_frame"):
        grid = threshold_grid(val[v]); _, acc_v, wr_v = cine_counts(val[v], val["wrong"], val["cine"], grid, ids_val)
        out = {"aurc_val": aurc(val[v], val["wrong"]), "aurc_test": aurc(test[v], test["wrong"]),
               "hindsight_coverage_val": {str(t): oracle_coverage(val[v], val["wrong"], t) for t in TARGETS},
               "hindsight_coverage_test": {str(t): oracle_coverage(test[v], test["wrong"], t) for t in TARGETS}, "rules": {}}
        for rule in RULES:
            out["rules"][rule] = {}
            for t in TARGETS:
                if rule == "point":
                    tau = select_threshold(val[v], val["wrong"], np.ones(val[v].size), t)
                elif rule == "ucb":
                    tau = select_threshold_ucb(acc_v, wr_v, grid, t, w_val)
                else:
                    tau = select_classwise(val[v], val["wrong"], val["pred"], val["cine"], t, rule=rule.split("_")[0],
                                           n_boot=w_val.shape[0], rng=rng)
                acc_val = accept_mask(val[v], val["pred"], tau); acc_t = accept_mask(test[v], test["pred"], tau)
                acc_p = accept_mask(pairs[v], pairs["pred"], tau)
                wrong_acc = acc_t & test["wrong"]
                a5c = float(wrong_acc[test["truth"] == A5C].sum() / wrong_acc.sum()) if wrong_acc.sum() else None
                out["rules"][rule][str(t)] = {
                    "tau": tau if not isinstance(tau, dict) else {str(k): x for k, x in tau.items()},
                    "val_coverage": float(acc_val.mean()),
                    "val_risk": float((acc_val & val["wrong"]).sum() / acc_val.sum()) if acc_val.sum() else None,
                    "test": ci_by_cine(acc_t, test["wrong"], test["cine"], ids_test, w_test),
                    "pairs_coverage": float(acc_p.mean()),
                    "pairs_risk": float((acc_p & pairs["wrong"]).sum() / acc_p.sum()) if acc_p.sum() else None,
                    "a5c_share_of_routed_error": a5c}
        res["variants"][v] = out
    return res


def main() -> None:
    ap = base_parser("Routing confidence and threshold rules")
    ap.add_argument("--ckpt-hash", required=True)
    ap.add_argument("--methods", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(); cfg = load_cfg(args); paths = load_paths()
    cfg_mean = copy.deepcopy(cfg); cfg_mean["policy"]["score"] = "mean"
    cfg["policy"]["score"] = "p10_agree"
    idx = read_index(variant_dir(paths.cache_root, args.ckpt_hash, "orig"))
    cache: dict = {}

    def loader(video_id: str, variant: str):
        if (video_id, variant) not in cache:
            cache[(video_id, variant)] = load_video_features(variant_dir(paths.cache_root, args.ckpt_hash, variant) / f"{video_id}.npz")
        return cache[(video_id, variant)]

    banks = {"multi_validation": render_bank(multi_bank(idx, "validation", cfg), loader),
             "multi_test": render_bank(multi_bank(idx, "test", cfg), loader),
             "pairs_test": render_bank(pair_bank(idx, "test", cfg), loader)}
    ids = {b: np.unique(np.concatenate([[fr.video_id for fr in s.recipe.fragments] for s in banks[b]]).astype(object))
           for b in ("multi_validation", "multi_test")}
    rng = np.random.default_rng(args.seed)
    w_val = resample_weights(ids["multi_validation"].size, args.n_boot, rng)
    w_test = resample_weights(ids["multi_test"].size, args.n_boot, rng)
    run_dir = new_run_dir(cfg, "routing_scores")
    results = {"ckpt_hash": args.ckpt_hash, "n_boot": args.n_boot, "seed": args.seed, "targets": list(TARGETS),
               "n_cines": {b: int(v.size) for b, v in ids.items()}, "methods": {}}
    for m in args.methods.split(","):
        results["methods"][m] = evaluate(m, banks, cfg, cfg_mean, w_val, w_test, ids["multi_validation"], ids["multi_test"], rng)
        (run_dir / "results.json").write_text(json.dumps(results, indent=1))
        print(f"done {m}", flush=True)
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
