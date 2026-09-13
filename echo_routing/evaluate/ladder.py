"""Run one decoder over the native and constructed settings and produce the metrics dict (contract §4.6)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from echo_routing.compose.stream_assembler import Stream
from echo_routing.evaluate.metrics_boundary import boundary_f1
from echo_routing.evaluate.metrics_classification import classification_summary
from echo_routing.evaluate.metrics_routing import coverage_and_risk
from echo_routing.evaluate.metrics_stability import fragments_per_minute
from echo_routing.temporal.baselines.b0_frame_argmax import clip_majority
from echo_routing.temporal.baselines.b_file import file_score
from echo_routing.temporal.baselines.registry import decode_with_prob
from echo_routing.temporal.decode_select import route, select_threshold
from echo_routing.temporal.segments import boundaries_from_labels


@dataclass(frozen=True)
class NativeCine:
    video_id: str
    label: int
    prob: np.ndarray
    feat: np.ndarray | None = None


def _per_sample_routing(prob: np.ndarray, labels: np.ndarray, truth: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray, list]:
    """Return (score per sample, wrong per sample, intervals) with tau=-inf (all accepted)."""
    pol = cfg["policy"]
    ivs = route(prob, labels, tau=-np.inf, method=pol["score"], min_len=int(pol["min_len"]))
    score = np.full(prob.shape[0], -np.inf); wrong = np.zeros(prob.shape[0], bool)
    for iv in ivs:
        if iv.accepted:  # length >= min_len
            score[iv.start:iv.end] = iv.score
            wrong[iv.start:iv.end] = truth[iv.start:iv.end] != iv.view
    return score, wrong, ivs


def native_metrics(method: str, cines: Sequence[NativeCine], num_classes: int, hz: float, cfg: dict) -> dict:
    frame_true, frame_pred, cine_true, cine_pred, fpm = [], [], [], [], []
    for c in cines:
        labels, _ = decode_with_prob(method, c.prob, cfg, c.feat, c.video_id)
        frame_true.append(np.full(labels.size, c.label)); frame_pred.append(labels)
        cine_true.append(c.label); cine_pred.append(int(np.bincount(labels, minlength=num_classes).argmax()))
        fpm.append(fragments_per_minute(labels, hz))
    return {
        "frame": classification_summary(np.concatenate(frame_true), np.concatenate(frame_pred), num_classes),
        "cine": classification_summary(np.array(cine_true), np.array(cine_pred), num_classes),
        "stability": {"fragments_per_minute_mean": float(np.mean(fpm)), "fragments_per_minute_median": float(np.median(fpm)),
                      "share_fragmented": float(np.mean(np.array(fpm) > 0)), "proxy_metric": True},
    }


def constructed_metrics(method: str, streams: Sequence[Stream], hz: float, cfg: dict, tau: float | None) -> dict:
    tols = cfg["policy"]["boundary_tolerances_s"]
    per_cell: dict[str, dict] = {}
    all_pred, all_true = {t: ([], []) for t in tols}, None
    scores, wrongs, weights = [], [], []
    for s in streams:
        labels, score_prob = decode_with_prob(method, s.prob, cfg, s.feat, s.recipe.recipe_id)
        pred_b = boundaries_from_labels(labels)
        cell = per_cell.setdefault(s.recipe.cell, {"n": 0, "false_splits": 0, "joins": 0, "missed": 0, "sem": 0, "frame_acc": []})
        cell["n"] += 1; cell["frame_acc"].append(float((labels == s.labels).mean()))
        same_joins = [j for j in s.joins if j not in set(s.semantic_boundaries.tolist())]
        tol_samples = tols[1] * hz
        cell["joins"] += len(same_joins)
        cell["false_splits"] += sum(1 for j in same_joins if np.any(np.abs(pred_b - j) <= tol_samples))
        cell["sem"] += int(s.semantic_boundaries.size)
        cell["missed"] += sum(1 for b in s.semantic_boundaries if not np.any(np.abs(pred_b - b) <= tol_samples))
        for t in tols:
            all_pred[t][0].append(pred_b / hz + 1000.0 * len(all_pred[t][0])); all_pred[t][1].append(s.semantic_boundaries / hz + 1000.0 * len(all_pred[t][1]))
        sc, wr, _ = _per_sample_routing(score_prob, labels, s.labels, cfg)
        scores.append(sc); wrongs.append(wr); weights.append(np.ones(sc.size))
    boundary = {}
    for t in tols:
        r = boundary_f1(np.concatenate(all_pred[t][0]), np.concatenate(all_pred[t][1]), t)
        boundary[str(t)] = {"precision": r.precision, "recall": r.recall, "f1": r.f1, "median_abs_error_s": r.median_abs_error,
                            "n_true": r.n_true, "n_pred": r.n_pred}
    cells = {k: {"n_streams": v["n"], "same_view_joins": v["joins"], "false_splits": v["false_splits"],
                 "false_split_rate": (v["false_splits"] / v["joins"]) if v["joins"] else None,
                 "semantic_boundaries": v["sem"], "missed": v["missed"],
                 "missed_rate": (v["missed"] / v["sem"]) if v["sem"] else None,
                 "frame_accuracy": float(np.mean(v["frame_acc"]))} for k, v in per_cell.items()}
    sc = np.concatenate(scores); wr = np.concatenate(wrongs); w = np.concatenate(weights)
    routing = None
    if tau is not None:
        routing = {"tau": tau, **coverage_and_risk(sc >= tau, wr, w)}
    return {"cells": cells, "boundary": boundary, "routing": routing, "_scores": sc, "_wrong": wr, "_weights": w}


def select_policy(method: str, val_streams: Sequence[Stream], cfg: dict) -> dict:
    m = constructed_metrics(method, val_streams, cfg["sampling"]["target_hz"], cfg, tau=None)
    targets = [cfg["policy"]["target_risk"], *cfg["policy"].get("extra_targets", [])]
    out = {}
    for t in targets:
        tau = select_threshold(m["_scores"], m["_wrong"], m["_weights"], t)
        out[str(t)] = {"tau": tau, "status": "ok" if tau is not None else "target_unmet",
                       **(coverage_and_risk(m["_scores"] >= tau, m["_wrong"], m["_weights"]) if tau is not None else {})}
    return out


def evaluate_method(method: str, native_val: Sequence[NativeCine], native_test: Sequence[NativeCine],
                    val_streams: Sequence[Stream], test_streams: Sequence[Stream], num_classes: int, cfg: dict,
                    test_multi: Sequence[Stream] | None = None) -> dict:
    hz = cfg["sampling"]["target_hz"]
    policy = select_policy(method, val_streams, cfg)
    tau = policy[str(cfg["policy"]["target_risk"])]["tau"]
    test = constructed_metrics(method, test_streams, hz, cfg, tau)
    curve = risk_coverage_points(test["_scores"], test["_wrong"])
    multi = None
    if test_multi:
        mm = constructed_metrics(method, test_multi, hz, cfg, tau)
        multi = {"test_n": len(test_multi), "cells": mm["cells"], "boundary": mm["boundary"], "routing": mm["routing"],
                 "note": "secondary setting: 4-8 fragments, cines reused across streams; same frozen tau as the pair bank"}
    return {
        "constructed_multi": multi,
        "method_id": method, "task_classes": num_classes,
        "policy": {"selected_on": "validation", "target_risk": cfg["policy"]["target_risk"], "by_target": policy,
                   "score": cfg["policy"]["score"], "min_len_samples": cfg["policy"]["min_len"]},
        "native": {"validation": native_metrics(method, native_val, num_classes, hz, cfg),
                   "test": native_metrics(method, native_test, num_classes, hz, cfg)},
        "constructed": {"validation_n": len(val_streams), "test_n": len(test_streams),
                        "cells": test["cells"], "boundary": test["boundary"], "routing": test["routing"],
                        "risk_coverage_curve": curve},
    }


def risk_coverage_points(scores: np.ndarray, wrong: np.ndarray, n_points: int = 25) -> list[dict]:
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return []
    taus = np.quantile(finite, np.linspace(0, 1, n_points))
    return [{"tau": float(t), **coverage_and_risk(scores >= t, wrong)} for t in taus]


def evaluate_b_file(native_val: Sequence[NativeCine], native_test: Sequence[NativeCine], num_classes: int, cfg: dict) -> dict:
    """Operational comparator: one decision per native file (separate table, proposal §7.1)."""
    def run(cines):
        views, scores, truths, durs = [], [], [], []
        for c in cines:
            v, s = file_score(c.prob, "mean"); views.append(v); scores.append(s); truths.append(c.label); durs.append(c.prob.shape[0])
        return np.array(views), np.array(scores), np.array(truths), np.array(durs, float)
    vv, vs, vt, vd = run(native_val); tv, ts, tt, td = run(native_test)
    targets = [cfg["policy"]["target_risk"], *cfg["policy"].get("extra_targets", [])]
    by_target = {}
    for t in targets:
        tau = select_threshold(vs, vv != vt, vd, t)
        by_target[str(t)] = {"tau": tau, "status": "ok" if tau is not None else "target_unmet",
                             "validation": coverage_and_risk(vs >= tau, vv != vt, vd) if tau is not None else None,
                             "test": coverage_and_risk(ts >= tau, tv != tt, td) if tau is not None else None}
    return {"method_id": "b_file", "task_classes": num_classes,
            "native": {"validation": {"cine": classification_summary(vt, vv, num_classes)},
                       "test": {"cine": classification_summary(tt, tv, num_classes)}},
            "policy": {"selected_on": "validation", "by_target": by_target, "score": "mean"},
            "clip_majority_test_accuracy": float(np.mean([clip_majority(c.prob) == c.label for c in native_test]))}
