"""Validation metrics and selection rules of model development round 1 (configs/experiments/model_development_v1.yaml).

The definitions match the joint true-order comparison (scripts/true_order_joint.py):
- native clip label = majority of the decoded per-sample labels; macro-F1, per-class recall and confusion over clips;
- native internal cuts = label changes inside a native clip, per clip and per minute;
- segment F1@50 = class-aware one-to-one matching at IoU 0.5, pooled over streams (2 * matched / (pred + true));
- boundary recall / F1 = one-to-one matching of predicted label changes to true view changes within 0.5 s, pooled;
- same-class cut rate = same-label joins with a predicted change within 0.5 s / same-label joins.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from echo_routing.compose.stream_assembler import Stream
from echo_routing.evaluate.ladder import _fragment_rows
from echo_routing.evaluate.metrics_segmentation import segment_f1
from echo_routing.temporal.segments import boundaries_from_labels, labels_to_segments


def native_table(preds: dict[str, np.ndarray], natives: Sequence, num_classes: int = 5) -> pd.DataFrame:
    rows = []
    for c in natives:
        lab = np.asarray(preds[c.seq_id]); y = int(c.labels[0])
        rows.append({"video_id": c.seq_id, "label": y, "n": int(lab.size),
                     "majority": int(np.bincount(lab, minlength=num_classes).argmax()),
                     "cuts": int(boundaries_from_labels(lab).size), "frame_correct": int((lab == y).sum())})
    return pd.DataFrame(rows)


def stream_tables(preds: dict[str, np.ndarray], streams: Sequence[Stream], tol: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-stream segment counts and per-fragment boundary counts (the ladder's fragment rows)."""
    seg_rows, frag_rows = [], []
    for s in streams:
        lab = np.asarray(preds[s.recipe.recipe_id])
        ps, ts = labels_to_segments(lab), labels_to_segments(s.labels)
        seg_rows.append({"stream": s.recipe.recipe_id, "seg_matched": segment_f1(ps, ts, 0.5)["matched"],
                         "seg_pred": len(ps), "seg_true": len(ts), "n": int(lab.size), "correct": int((lab == s.labels).sum())})
        frag_rows.extend(_fragment_rows(s, boundaries_from_labels(lab), tol, np.zeros(lab.size, bool), {}))
    return pd.DataFrame(seg_rows), pd.DataFrame(frag_rows)


def _ratio(a: float, b: float) -> float | None:
    return float(a / b) if b else None


def _clip_summary(nat: pd.DataFrame, num_classes: int) -> dict:
    conf = np.zeros((num_classes, num_classes), int)
    np.add.at(conf, (nat["label"].to_numpy(), nat["majority"].to_numpy()), 1)
    tp = np.diag(conf); fp = conf.sum(0) - tp; fn = conf.sum(1) - tp; den = 2 * tp + fp + fn
    f1 = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)
    recall = np.where(conf.sum(1) > 0, tp / np.maximum(conf.sum(1), 1), np.nan)
    return {"native_macro_f1": float(f1.mean()), "native_per_class_recall": [None if np.isnan(r) else float(r) for r in recall],
            "native_confusion": conf.tolist(), "native_accuracy": float(tp.sum() / max(conf.sum(), 1))}


def summarize(nat: pd.DataFrame, multi_seg: pd.DataFrame, multi_frag: pd.DataFrame, pairs_seg: pd.DataFrame,
              pairs_frag: pd.DataFrame, hz: float, num_classes: int = 5) -> dict:
    m = multi_frag.sum(numeric_only=True); p = pairs_frag.sum(numeric_only=True)
    ms = multi_seg.sum(numeric_only=True); ps = pairs_seg.sum(numeric_only=True)
    return {
        **_clip_summary(nat, num_classes),
        "native_frame_accuracy": float(nat["frame_correct"].sum() / nat["n"].sum()),
        "native_internal_cuts_per_clip": float(nat["cuts"].mean()),
        "native_internal_cuts_per_min": float(nat["cuts"].sum() / (nat["n"].sum() / (60.0 * hz))),
        "native_share_cut": float((nat["cuts"] > 0).mean()),
        "segment_f1_50_multi": float(2 * ms["seg_matched"] / (ms["seg_pred"] + ms["seg_true"])),
        "boundary_recall_multi": _ratio(m["t_matched"], m["t_true"]),
        "boundary_precision_multi": _ratio(m["t_matched"], m["p_pred"]),
        "boundary_f1_multi": _ratio(2 * m["t_matched"], m["p_pred"] + m["t_true"]),
        "same_class_cut_rate_multi": _ratio(m["false_split"], m["same_join"]),
        "multi_frame_accuracy": float(ms["correct"] / ms["n"]),
        "segment_f1_50_pairs": float(2 * ps["seg_matched"] / (ps["seg_pred"] + ps["seg_true"])),
        "boundary_recall_pairs": _ratio(p["t_matched"], p["t_true"]),
        "counts": {"native_clips": int(len(nat)), "multi_streams": int(len(multi_seg)), "pair_streams": int(len(pairs_seg)),
                   "multi_true_changes": int(m["t_true"]), "multi_same_joins": int(m["same_join"])},
    }


def evaluate_predictions(preds: dict[str, np.ndarray], eval_set, hz: float, tol_s: float, num_classes: int = 5) -> tuple[dict, dict]:
    """(metrics, tables) of one method on one validation half; `preds` maps sequence id -> decoded labels."""
    tol = tol_s * hz
    nat = native_table(preds, eval_set.natives, num_classes)
    ms, mf = stream_tables(preds, eval_set.multi, tol)
    ps, pf = stream_tables(preds, eval_set.pairs, tol)
    tables = {"native": nat, "multi_seg": ms, "multi_frag": mf, "pairs_seg": ps, "pairs_frag": pf}
    return summarize(nat, ms, mf, ps, pf, hz, num_classes), tables


def select_checkpoint(records: Sequence[dict], floor: float, tie: float) -> dict:
    """Checkpoint rule: among records with native macro-F1 >= floor, the highest multi segment F1@50; records within
    `tie` of it are tied and the one with fewer native internal cuts wins (then the earlier epoch). Without an eligible
    record, the highest segment F1 is kept for diagnosis and flagged."""
    eligible = [r for r in records if r["native_macro_f1"] >= floor]
    pool = eligible or list(records)
    best = max(r["segment_f1_50_multi"] for r in pool)
    tied = [r for r in pool if best - r["segment_f1_50_multi"] < tie]
    chosen = min(tied, key=lambda r: (r["native_internal_cuts_per_clip"], r.get("epoch", 0)))
    return {"chosen": chosen, "status": "ok" if eligible else "native_floor_unmet", "floor": floor}


def select_operating_point(records: Sequence[dict], floor: float, recall_min: float, tie: float) -> dict:
    """Decoding-parameter rule (calibration half). Main point: among settings with native macro-F1 >= floor and multi
    boundary recall >= recall_min, the highest multi segment F1@50, ties (< tie) to fewer native cuts. Stability point:
    among the same eligible settings, the fewest native cuts.

    Without a setting that meets both constraints the main point keeps the recall requirement and drops the native
    floor (flagged native_floor_unmet), then keeps only the floor (flagged recall_unmet), then uses all settings
    (flagged both_unmet). The order was fixed on 2026-10-02 after round-1 seed 0, when no temporal decoder met the
    calibration-half native floor; the earlier order (floor first) let settings with recall below the requirement win."""
    has_recall = [r for r in records if (r["boundary_recall_multi"] or 0.0) >= recall_min]
    ok = [r for r in has_recall if r["native_macro_f1"] >= floor]
    ok_native = [r for r in records if r["native_macro_f1"] >= floor]
    if ok:
        status, pool = "ok", ok
    elif has_recall:
        status, pool = "native_floor_unmet", has_recall
    elif ok_native:
        status, pool = "recall_unmet", ok_native
    else:
        status, pool = "both_unmet", list(records)
    best = max(r["segment_f1_50_multi"] for r in pool)
    tied = [r for r in pool if best - r["segment_f1_50_multi"] < tie]
    main = min(tied, key=lambda r: (r["native_internal_cuts_per_clip"], r.get("order", 0)))
    stable = min(ok, key=lambda r: (r["native_internal_cuts_per_clip"], -r["segment_f1_50_multi"], r.get("order", 0))) if ok else None
    return {"main": main, "stability": stable, "status": status, "floor": floor, "recall_min": recall_min}
