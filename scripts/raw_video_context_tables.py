"""Pure summaries of the raw-video context round (configs/experiments/raw_video_context_v1.yaml; plan Tasks 3-4).

- Native window recognition: the five-class argmax balanced accuracy of A4C and A5C windows (mean of the two recalls),
  the AUC of logit(A5C) - logit(A4C) on those windows (A5C positive) and the five-class accuracy of all native windows,
  with a paired bootstrap that resamples sessions (the same draws for both systems).
- Mixed windows of constructed streams: accuracy by the target class's share of the 16 frames, fixed and new errors
  with their recordings and sessions.
- Sequence ends: window accuracy when the marker is not at slot 8 (the window was clamped) against slot 8.
- Boundary offsets: one-to-one matched predicted changes within the tolerance, signed and absolute offsets.
- The goal and the fixed decision readings, and the markdown tables.
Nothing here reads a file or a recording id; callers pass arrays and keys.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.stats import rankdata

from echo_routing.evaluate.window_target import SHARE_BINS, share_bin

A4C, A5C = 2, 3
HALVES = ("selection", "calibration")
REC_METRICS = ("A4C/A5C balanced accuracy", "A5C-A4C margin AUC", "native window accuracy")
POS, NEG, NONE = "consistent positive", "consistent negative", "no consistent difference detected"


# ----------------------------------------------------------------------------------------------- recognition
def balanced_accuracy(pred: np.ndarray, label: np.ndarray) -> float | None:
    """Mean of the A4C and A5C recalls (five-class argmax) over the A4C and A5C windows."""
    pred, label = np.asarray(pred), np.asarray(label)
    r = [float((pred[label == k] == k).mean()) for k in (A4C, A5C) if (label == k).any()]
    return float(np.mean(r)) if len(r) == 2 else None


def margin_auc(margin: np.ndarray, label: np.ndarray) -> float | None:
    """AUC of the margin with A5C windows positive and A4C windows negative (average ranks for ties)."""
    margin, label = np.asarray(margin, float), np.asarray(label)
    keep = (label == A4C) | (label == A5C); m, y = margin[keep], label[keep] == A5C
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    r = rankdata(m)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def accuracy(pred: np.ndarray, label: np.ndarray) -> float:
    return float((np.asarray(pred) == np.asarray(label)).mean())


def recognition(pred: np.ndarray, margin: np.ndarray, label: np.ndarray) -> dict:
    pair = np.isin(label, (A4C, A5C))
    return {"A4C/A5C balanced accuracy": balanced_accuracy(pred[pair], label[pair]), "A5C-A4C margin AUC": margin_auc(margin[pair], label[pair]),
            "native window accuracy": accuracy(pred, label), "A4C recall": float((pred[label == A4C] == A4C).mean()) if (label == A4C).any() else None,
            "A5C recall": float((pred[label == A5C] == A5C).mean()) if (label == A5C).any() else None,
            "windows": int(label.size), "A4C windows": int((label == A4C).sum()), "A5C windows": int((label == A5C).sum())}


def _metric_values(pred, margin, label, idx) -> list[float | None]:
    p, m, y = pred[idx], margin[idx], label[idx]; pair = np.isin(y, (A4C, A5C))
    return [balanced_accuracy(p[pair], y[pair]), margin_auc(m[pair], y[pair]), accuracy(p, y)]


def window_bootstrap(a: dict, b: dict, label: np.ndarray, session: np.ndarray, draws: int = 2000, seed: int = 20261002) -> dict:
    """a - b on the same native windows; a, b = {"pred": (N,), "margin": (N,)}. Sessions are resampled with replacement
    and both systems see the same draws."""
    label, session = np.asarray(label), np.asarray(session)
    if len(a["pred"]) != label.size or len(b["pred"]) != label.size:
        raise ValueError("both systems must cover the same windows")
    keys = np.unique(session); by = {k: np.flatnonzero(session == k) for k in keys}; rng = np.random.default_rng(seed)
    pa, ma, pb, mb = (np.asarray(x) for x in (a["pred"], a["margin"], b["pred"], b["margin"]))
    full = np.arange(label.size); point = [x - y if x is not None and y is not None else None
                                           for x, y in zip(_metric_values(pa, ma, label, full), _metric_values(pb, mb, label, full))]
    rep = [[] for _ in REC_METRICS]
    for _ in range(int(draws)):
        idx = np.concatenate([by[k] for k in rng.choice(keys, size=keys.size, replace=True)])
        for j, (x, y) in enumerate(zip(_metric_values(pa, ma, label, idx), _metric_values(pb, mb, label, idx))):
            if x is not None and y is not None:
                rep[j].append(x - y)
    out = {}
    for j, name in enumerate(REC_METRICS):
        ci = [float(np.percentile(rep[j], 2.5)), float(np.percentile(rep[j], 97.5))] if rep[j] else [None, None]
        out[name] = {"delta": point[j], "ci95": ci, "draws_used": len(rep[j])}
    return out


def consistency(deltas: list[dict]) -> str:
    """'consistent positive' when every seed's lower bound is above zero, 'consistent negative' when every upper bound
    is below zero, otherwise no consistent difference detected."""
    if deltas and all(d["ci95"][0] is not None and d["ci95"][0] > 0 for d in deltas):
        return POS
    if deltas and all(d["ci95"][1] is not None and d["ci95"][1] < 0 for d in deltas):
        return NEG
    return NONE


# ----------------------------------------------------------------------------------------------- mixed windows and ends
def mixed_table(correct: np.ndarray, shares: np.ndarray) -> dict:
    """Accuracy of mixed windows overall and by the number of the 16 frames that carry the target's class."""
    correct, shares = np.asarray(correct, bool), np.asarray(shares, int); bins = np.array([share_bin(k) for k in shares])
    return {"windows": int(correct.size), "accuracy": float(correct.mean()) if correct.size else None,
            "by_share": {b: {"windows": int((bins == b).sum()), "accuracy": float(correct[bins == b].mean()) if (bins == b).any() else None} for b in SHARE_BINS}}


def changes(correct_a: np.ndarray, correct_b: np.ndarray, recordings: np.ndarray, sessions: np.ndarray) -> dict:
    """Windows right in a and wrong in b (fixed by a) and the reverse (new errors of a), with distinct recordings and sessions."""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool); out = {}
    for name, sel in (("fixed", a & ~b), ("new", ~a & b)):
        out[name] = {"windows": int(sel.sum()), "recordings": int(len(set(np.asarray(recordings)[sel].tolist()))),
                     "sessions": int(len(set(np.asarray(sessions)[sel].tolist())))}
    return out


def end_table(correct: np.ndarray, marker: np.ndarray) -> dict:
    """Window accuracy when the window was clamped at a sequence end (marker not at slot 8) and otherwise."""
    correct, marker = np.asarray(correct, bool), np.asarray(marker)
    end = marker != 8
    return {"end_windows": int(end.sum()), "end_accuracy": float(correct[end].mean()) if end.any() else None,
            "middle_windows": int((~end).sum()), "middle_accuracy": float(correct[~end].mean()) if (~end).any() else None}


def match_offsets(true_b: np.ndarray, pred_b: np.ndarray, tol: float) -> list[int]:
    """Signed offsets (pred - true) of one-to-one matches within tol, closest pairs first."""
    pairs = sorted((abs(int(p) - int(t)), int(p) - int(t), i, j) for i, t in enumerate(true_b) for j, p in enumerate(pred_b) if abs(int(p) - int(t)) <= tol)
    used_t, used_p, out = set(), set(), []
    for _, d, i, j in pairs:
        if i not in used_t and j not in used_p:
            used_t.add(i); used_p.add(j); out.append(d)
    return out


def offset_summary(offsets: list[int]) -> dict:
    o = np.asarray(offsets, float)
    return {"matched": int(o.size), "mean_abs": float(np.abs(o).mean()) if o.size else None, "mean_signed": float(o.mean()) if o.size else None,
            "exact_share": float((o == 0).mean()) if o.size else None}


# ----------------------------------------------------------------------------------------------- goal and decision
def goal(rows: list[dict], feasible: list[bool], g: dict) -> dict:
    """rows: per seed {half: summary row} of one group's B3 system; feasible: per seed whether its B3 setting was feasible."""
    mean = {h: float(np.mean([r[h]["segment_f1_50_multi"] for r in rows])) for h in HALVES}
    floors = g["native_macro_f1_min"]; fails = []
    for s, (r, f) in enumerate(zip(rows, feasible)):
        for h in HALVES:
            rec, nat = r[h]["boundary_recall_multi"] or 0.0, r[h]["native_macro_f1"]
            if not f or rec < float(g["boundary_recall_min"]) or nat < float(floors[h]):
                fails.append({"seed": s, "half": h, "b3_feasible": bool(f), "boundary_recall": rec, "native_macro_f1": nat})
    met = not fails and mean["selection"] >= float(g["selection_min"]) and mean["calibration"] >= float(g["calibration_min"])
    return {"mean_segment_f1": mean, "margin": {"selection": mean["selection"] - float(g["selection_min"]), "calibration": mean["calibration"] - float(g["calibration_min"])},
            "calibration_sensitivity_margin": mean["calibration"] - float(g["calibration_sensitivity"]), "constraint_failures": fails, "met": bool(met)}


def both(v: dict, want: str) -> bool:
    return all(v[h] == want for h in HALVES)


def decide(rules: dict, rec: dict, sysv: dict, goals: dict) -> dict:
    """The fixed readings. rec[comparison][metric][half] and sysv[comparison][half] (B3 segment F1@50) are verdicts;
    goals[group] from goal()."""
    ba, auc = REC_METRICS[0], REC_METRICS[1]; within = ("Ordered - Shuffled", "Ordered - Target-only", "Shuffled - Target-only")
    order = both(rec["Ordered - Shuffled"][ba], POS)
    order_halves = [h for h in HALVES if rec["Ordered - Shuffled"][ba][h] == POS]
    multi = both(rec["Ordered - Target-only"][ba], POS) and both(rec["Shuffled - Target-only"][ba], POS) and not order
    ranking = [c for c in within if both(rec[c][auc], POS) and not both(rec[c][ba], POS) and not both(sysv[c], POS)]
    rec_vs_sys = [c for c in within if both(rec[c][ba], POS) and any(sysv[c][h] == NEG for h in HALVES)]
    sys_gain = [c for c, v in sysv.items() if c.startswith("Ordered") and both(v, POS)]
    rec_gain = [c for c in within if both(rec[c][ba], POS)]
    control = sorted({f"{c} ({m}, {h})" for c in within if c.startswith("Ordered") for m in (ba,) for h in HALVES if rec[c][m][h] == NEG}
                     | {f"{c} (B3 segment F1@50, {h})" for c in sysv if c in within and c.startswith("Ordered") for h in HALVES if sysv[c][h] == NEG})
    applies = []
    if order:
        applies.append(("order_gain", rules["order_gain"]))
    elif order_halves:
        applies.append(("order_gain_one_half", rules["order_gain_one_half"] + f" (half: {', '.join(order_halves)})"))
    if multi:
        applies.append(("multiframe_gain", rules["multiframe_gain"]))
    if ranking:
        applies.append(("ranking_only", rules["ranking_only"] + f" (comparisons: {', '.join(ranking)})"))
    if rec_vs_sys:
        applies.append(("recognition_vs_system", rules["recognition_vs_system"] + f" (comparisons: {', '.join(rec_vs_sys)})"))
    if not rec_gain and not sys_gain:
        applies.append(("no_gain", rules["no_gain"]))
    if sys_gain:
        applies.append(("system_gain", rules["system_gain"] + f" (against: {', '.join(sys_gain)})"))
    if control:
        applies.append(("control_wins", rules["control_wins"] + f" ({'; '.join(control)})"))
    met = [g for g, v in goals.items() if v["met"]]
    if met:
        applies.append(("goal_met", rules["goal_met"] + f" (groups: {', '.join(met)})"))
    return {"rows_that_apply": [{"rule": k, "text": t} for k, t in applies], "order_gain_both_halves": order, "order_gain_halves": order_halves,
            "multiframe_gain": multi, "recognition_gains_both_halves": rec_gain, "ranking_only": ranking, "recognition_but_system_worse": rec_vs_sys,
            "system_gains_both_halves": sys_gain, "controls_better": control, "goal_met_groups": met}


# ----------------------------------------------------------------------------------------------- markdown
def md_table(rows: list[dict], cols: list[str] | None = None) -> str:
    if not rows:
        return "(none)\n"
    cols = cols or list(dict.fromkeys(k for r in rows for k in r))
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else "" if v is None else str(v)  # noqa: E731
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]) + "\n"


def ci(d: dict) -> str:
    if d.get("delta") is None:
        return "n/a"
    lo, hi = d["ci95"]
    return f"{d['delta']:+.4f} [{lo:+.4f}, {hi:+.4f}]" if lo is not None else f"{d['delta']:+.4f}"


def tables_md(res: dict) -> str:
    parts = ["# Raw-video context round: tables\n",
             "Recompute: `python scripts/report_raw_video_context.py --config configs/experiments/raw_video_context_v1.yaml --evaluate`\n"]
    rec = [{"system": k, "half": h, **{m: v[h].get(m) for m in (*REC_METRICS, "A4C recall", "A5C recall", "windows")}}
           for k, v in res.get("recognition", {}).items() for h in HALVES if h in v]
    if rec:
        parts += ["## Native window recognition (same centres)\n", md_table(rec)]
    rc = [{"comparison": k, **{m: ci(v[m]) for m in REC_METRICS}} for k, v in res.get("recognition_comparisons", {}).items()]
    if rc:
        parts += ["## Recognition comparisons (session-paired bootstrap)\n", md_table(rc)]
    sys_rows = [{"system": k, "half": h, "row": row, "segment F1@50": r.get("segment_f1_50_multi"), "boundary P": r.get("boundary_precision_multi"),
                 "boundary R": r.get("boundary_recall_multi"), "native macro-F1": r.get("native_macro_f1"), "switches per clip": r.get("native_internal_cuts_per_clip"),
                 "same-view join cuts": r.get("same_class_cut_rate_multi"), "A5C read as A4C": r.get("A5C read as A4C"), "A4C read as A5C": r.get("A4C read as A5C")}
                for k, per in res.get("systems", {}).items() for row, by in per.items() if row in ("argmax", "b3", "row") for h, r in by.items()]
    if sys_rows:
        parts += ["## Systems\n", md_table(sys_rows)]
    sc = [{"comparison": k, "segment F1@50": ci(v["bootstrap"]["multi segment F1@50"]), "native macro-F1": ci(v["bootstrap"]["native macro-F1"]),
           "boundary recall": ci(v["bootstrap"]["multi boundary recall"])} for k, v in res.get("system_comparisons", {}).items()]
    if sc:
        parts += ["## System comparisons (paired bootstrap; streams reuse recordings)\n", md_table(sc)]
    for key, title in (("verdicts", "Verdicts"), ("goal", "Goal"), ("decision", "Decision")):
        parts += [f"## {title}\n", "```\n" + json.dumps(res.get(key, "pending"), indent=1, default=str) + "\n```\n"]
    return "\n".join(parts)
