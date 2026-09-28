"""Run one decoder over the native and constructed settings and produce the metrics dict (contract §4.6)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from echo_routing.compose.stream_assembler import Stream
from echo_routing.evaluate.metrics_boundary import boundary_f1, match_boundaries
from echo_routing.evaluate.metrics_classification import classification_summary
from echo_routing.evaluate.metrics_routing import coverage_and_risk
from echo_routing.evaluate.metrics_stability import fragments_per_minute
from echo_routing.evaluate.risk_control import accept_mask, select_classwise
from echo_routing.temporal.baselines.b0_frame_argmax import clip_majority
from echo_routing.temporal.baselines.b_file import file_score
from echo_routing.temporal.baselines.registry import decode_with_prob, evidence_prob, evidence_source
from echo_routing.temporal.decode_select import route, select_threshold
from echo_routing.temporal.segments import boundaries_from_labels


STREAM_OFFSET = 10**6  # samples; pooled boundary matching places stream k at k * STREAM_OFFSET
POLICY2_DEFAULT = {"score": "mean", "rule": "ucb", "level": 0.95, "n_boot": 1000, "seed": 0}


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


def _policy2_cfg(cfg: dict) -> dict:
    """Routing policy 2: segments scored with the pipeline's own frame-level evidence, one threshold per predicted
    class at the `level` upper bound of cine-resampled validation risk (docs/results/2026-09-26_routing_scores)."""
    return {**POLICY2_DEFAULT, **(cfg.get("policy2") or {})}


def _fragment_rows(s: Stream, pred_b: np.ndarray, tol: float, wrong: np.ndarray,
                   accepted: dict[str, np.ndarray]) -> list[dict]:
    """Per-fragment counts at the primary tolerance, so a bootstrap can resample streams or cines (proposal §7.3).
    A true change, its one-to-one match, a same-view join and its false split belong to the fragment the join enters;
    a predicted change belongs to the fragment it falls in; routing counts are accepted / accepted-and-wrong samples
    for every named accept mask."""
    sem = set(s.semantic_boundaries.tolist())
    matched = {int(s.semantic_boundaries[j]) for _, j in match_boundaries(pred_b, s.semantic_boundaries, tol)}
    rows, start = [], 0
    for i, fr in enumerate(s.recipe.fragments):
        end = start + (fr.end - fr.start)
        join, is_sem = i > 0, i > 0 and start in sem
        near = join and bool(np.any(np.abs(pred_b - start) <= tol))
        row = {"stream": s.recipe.recipe_id, "cell": s.recipe.cell, "video_id": fr.video_id, "label": int(fr.label),
               "variant": fr.variant, "n": end - start, "t_true": int(is_sem), "t_matched": int(is_sem and start in matched),
               "p_pred": int(np.count_nonzero((pred_b >= start) & (pred_b < end))), "same_join": int(join and not is_sem),
               "false_split": int(join and not is_sem and near), "missed": int(is_sem and not near)}
        for k, a in accepted.items():
            row[f"acc|{k}"] = int(a[start:end].sum()); row[f"wrong|{k}"] = int((a[start:end] & wrong[start:end]).sum())
        rows.append(row); start = end
    return rows


def constructed_metrics(method: str, streams: Sequence[Stream], hz: float, cfg: dict, tau: float | None,
                        taus: dict[str, float | None] | None = None,
                        class_taus: dict[str, dict[int, float | None]] | None = None) -> dict:
    """`taus` (name -> frozen threshold on the method's own score) and `class_taus` (name -> per-class thresholds on
    the policy-2 evidence score) add routing at each threshold and per-fragment count rows for resampling."""
    tols = cfg["policy"]["boundary_tolerances_s"]
    cfg2 = {**cfg, "policy": {**cfg["policy"], "score": _policy2_cfg(cfg)["score"]}}
    per_cell: dict[str, dict] = {}
    all_pred, all_true = {t: ([], []) for t in tols}, None
    scores, wrongs, weights, fragments, scores2, preds, cines = [], [], [], [], [], [], []
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
        for t in tols:  # integer sample positions, one stream per STREAM_OFFSET block: exact distances, no cross-stream matches
            all_pred[t][0].append(pred_b + STREAM_OFFSET * len(all_pred[t][0]))
            all_pred[t][1].append(s.semantic_boundaries + STREAM_OFFSET * len(all_pred[t][1]))
        sc, wr, _ = _per_sample_routing(score_prob, labels, s.labels, cfg)
        ev = evidence_prob(method, s.prob, cfg, s.feat, s.recipe.recipe_id, own=score_prob)
        sc2 = _per_sample_routing(ev, labels, s.labels, cfg2)[0]
        scores.append(sc); wrongs.append(wr); weights.append(np.ones(sc.size)); scores2.append(sc2); preds.append(labels)
        cines.append(np.concatenate([np.full(fr.end - fr.start, fr.video_id, dtype=object) for fr in s.recipe.fragments]))
        if taus is not None or class_taus:
            masks = {k: (sc >= t if t is not None else np.zeros(sc.size, bool)) for k, t in (taus or {}).items()}
            masks.update({k: accept_mask(sc2, labels, t) for k, t in (class_taus or {}).items()})
            fragments.extend(_fragment_rows(s, pred_b, tol_samples, wr, masks))
    boundary = {}
    for t in tols:
        r = boundary_f1(np.concatenate(all_pred[t][0]), np.concatenate(all_pred[t][1]), t * hz)
        boundary[str(t)] = {"precision": r.precision, "recall": r.recall, "f1": r.f1,
                            "median_abs_error_s": None if r.median_abs_error is None else r.median_abs_error / hz,
                            "n_true": r.n_true, "n_pred": r.n_pred, "matched": r.matched}
    cells = {k: {"n_streams": v["n"], "same_view_joins": v["joins"], "false_splits": v["false_splits"],
                 "false_split_rate": (v["false_splits"] / v["joins"]) if v["joins"] else None,
                 "semantic_boundaries": v["sem"], "missed": v["missed"],
                 "missed_rate": (v["missed"] / v["sem"]) if v["sem"] else None,
                 "frame_accuracy": float(np.mean(v["frame_acc"]))} for k, v in per_cell.items()}
    sc = np.concatenate(scores); wr = np.concatenate(wrongs); w = np.concatenate(weights)
    sc2 = np.concatenate(scores2); pred = np.concatenate(preds)
    routing = None
    if tau is not None:
        routing = {"tau": tau, **coverage_and_risk(sc >= tau, wr, w)}
    routing_by = None if taus is None and not class_taus else {
        k: ({"tau": t, **coverage_and_risk(sc >= t, wr, w)} if t is not None else {"tau": None}) for k, t in (taus or {}).items()}
    for k, ct in (class_taus or {}).items():
        routing_by[k] = {"tau_by_class": {str(c): v for c, v in ct.items()}, **coverage_and_risk(accept_mask(sc2, pred, ct), wr, w)}
    return {"cells": cells, "boundary": boundary, "routing": routing, "routing_by": routing_by,
            "_scores": sc, "_wrong": wr, "_weights": w, "_fragments": fragments,
            "_scores2": sc2, "_pred": pred, "_cine": np.concatenate(cines)}


def _targets(cfg: dict) -> list[float]:
    return [cfg["policy"]["target_risk"], *cfg["policy"].get("extra_targets", [])]


def _policy_from(m: dict, cfg: dict) -> dict:
    out = {}
    for t in _targets(cfg):
        tau = select_threshold(m["_scores"], m["_wrong"], m["_weights"], t)
        out[str(t)] = {"tau": tau, "status": "ok" if tau is not None else "target_unmet",
                       **(coverage_and_risk(m["_scores"] >= tau, m["_wrong"], m["_weights"]) if tau is not None else {})}
    return out


def _policy2_from(m: dict, cfg: dict) -> dict:
    p2 = _policy2_cfg(cfg); out = {}
    for t in _targets(cfg):
        ct = select_classwise(m["_scores2"], m["_wrong"], m["_pred"], m["_cine"], t, rule=p2["rule"], n_boot=int(p2["n_boot"]),
                              rng=np.random.default_rng(int(p2["seed"])), level=float(p2["level"]))
        out[str(t)] = {"tau_by_class": {str(c): v for c, v in ct.items()},
                       **coverage_and_risk(accept_mask(m["_scores2"], m["_pred"], ct), m["_wrong"], m["_weights"])}
    return out


def select_policy(method: str, val_streams: Sequence[Stream], cfg: dict) -> dict:
    return _policy_from(constructed_metrics(method, val_streams, cfg["sampling"]["target_hz"], cfg, tau=None), cfg)


def evaluate_method(method: str, native_val: Sequence[NativeCine], native_test: Sequence[NativeCine],
                    val_streams: Sequence[Stream], test_streams: Sequence[Stream], num_classes: int, cfg: dict,
                    test_multi: Sequence[Stream] | None = None, val_multi: Sequence[Stream] | None = None) -> dict:
    """Thresholds are frozen per validation bank: `policy` from the pair bank (legacy primary), `policy_multi` from the
    multi-fragment validation bank when given, and `policy2` (own frame-level evidence, per-class upper-bound thresholds)
    from the same multi-fragment bank. Test routing is reported as routing_by["pair@t" / "multi@t" / "p2@t"]."""
    hz = cfg["sampling"]["target_hz"]
    policy = select_policy(method, val_streams, cfg)
    tau = policy[str(cfg["policy"]["target_risk"])]["tau"]
    policy_multi = policy2 = None
    if val_multi:
        mv = constructed_metrics(method, val_multi, hz, cfg, tau=None)
        policy_multi, policy2 = _policy_from(mv, cfg), _policy2_from(mv, cfg)
    taus = {f"pair@{t}": v["tau"] for t, v in policy.items()}
    taus.update({f"multi@{t}": v["tau"] for t, v in (policy_multi or {}).items()})
    class_taus = {f"p2@{t}": {int(c): x for c, x in v["tau_by_class"].items()} for t, v in (policy2 or {}).items()}
    test = constructed_metrics(method, test_streams, hz, cfg, tau, taus=taus, class_taus=class_taus)
    curve = risk_coverage_points(test["_scores"], test["_wrong"])
    multi, fragments = None, {"pairs_test": test["_fragments"]}
    if test_multi:
        mm = constructed_metrics(method, test_multi, hz, cfg, tau, taus=taus, class_taus=class_taus)
        multi = {"test_n": len(test_multi), "cells": mm["cells"], "boundary": mm["boundary"], "routing": mm["routing"],
                 "routing_by": mm["routing_by"], "risk_coverage_curve": risk_coverage_points(mm["_scores"], mm["_wrong"]),
                 "note": "4-8 fragments, cines reused across streams; `routing` uses the pair-bank tau, "
                         "routing_by adds every target under both validation banks"}
        fragments["multi_test"] = mm["_fragments"]
    out = {
        "constructed_multi": multi,
        "method_id": method, "task_classes": num_classes,
        "policy": {"selected_on": "validation", "target_risk": cfg["policy"]["target_risk"], "by_target": policy,
                   "score": cfg["policy"]["score"], "min_len_samples": cfg["policy"]["min_len"]},
        "native": {"validation": native_metrics(method, native_val, num_classes, hz, cfg),
                   "test": native_metrics(method, native_test, num_classes, hz, cfg)},
        "constructed": {"validation_n": len(val_streams), "test_n": len(test_streams),
                        "cells": test["cells"], "boundary": test["boundary"], "routing": test["routing"],
                        "routing_by": test["routing_by"], "risk_coverage_curve": curve},
        "_fragments": fragments,
    }
    if policy_multi is not None:
        out["policy_multi"] = {"selected_on": "multi_validation", "validation_n": len(val_multi), "by_target": policy_multi}
        out["policy2"] = {"selected_on": "multi_validation", "evidence": evidence_source(method), **_policy2_cfg(cfg),
                          "by_target": policy2}
    return out


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
