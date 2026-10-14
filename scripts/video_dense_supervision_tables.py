"""Pure summaries of the video dense supervision round (configs/experiments/video_dense_supervision_v1.yaml; plan Task 4).

- The fixed readings of the task sheet: model progress, mechanism support (with the stream-sensitivity rule), system only,
  shuffled only, trade-off, no gain and end. The readings are not exclusive.
- Pre-specified anchor diagnostics of the time logits (never a system): accuracy at the 8 anchors against the reference
  labels, class-wise recall, and whether both anchors are right at every reference class change between consecutive
  anchors, each with the distinct source recordings and sessions; fewer than 10 units are flagged.
- Markdown tables. Nothing here reads a file or prints a recording id; callers pass arrays and sets.
"""
from __future__ import annotations

import json

import numpy as np

from scripts.raw_video_context_tables import NEG, NONE, POS, ci, md_table  # noqa: F401  (re-exported for the report)
from scripts.target_patch_readout_tables import UNCERTAIN

HALVES = ("selection", "calibration")
CLASSES = ("PLAX", "PSAX", "A4C", "A5C", "SC4C")
FEW_UNITS = 10
RULE_ORDER = ("model_progress", "mechanism_support", "system_only", "shuffled_only", "tradeoff", "no_gain", "end")
SYS = ("aligned - center_only", "aligned - Ordered", "aligned - shuffled")
MIX = ("aligned - center_only", "aligned - shuffled")


# ----------------------------------------------------------------------------------------------- readings
def decide(sysv: dict, mixv: dict, constraints_ok: bool, goal_met: bool, rules: dict | None = None) -> dict:
    """sysv[c][h]: B3 multi segment F1@50 verdicts for c in SYS; mixv[c][h]: all-mixed-window accuracy verdicts (session
    bootstrap, UNCERTAIN when the stream sensitivity is consistent the other way) for c in MIX; constraints_ok: aligned
    has a feasible B3 setting, native macro-F1 at the floor and boundary recall >= 0.95 in every seed and half; goal_met:
    aligned also meets the three-seed means."""
    both = lambda v: all(v[h] == POS for h in HALVES)  # noqa: E731
    some = lambda v: any(v[h] == POS for h in HALVES)  # noqa: E731
    ac_sys, ao_sys, as_sys = (sysv[c] for c in SYS); ac_mix, as_mix = (mixv[c] for c in MIX)
    mechanism = both(ac_mix) and both(as_mix) and both(ac_sys) and bool(constraints_ok)
    local = some(ac_mix) or some(ac_sys)
    rows = {"model_progress": both(ac_sys) and both(ao_sys) and bool(constraints_ok) and bool(goal_met),
            "mechanism_support": mechanism,
            "system_only": both(ac_sys) and not mechanism,
            "shuffled_only": (some(as_mix) or some(as_sys)) and not local,
            "tradeoff": local and (not constraints_ok or not both(ac_sys)),
            "no_gain": not local and not some(ao_sys),
            "end": True}
    out = {"rows": {k: {"applies": bool(v), **({"reading": rules[k]} if rules and k in rules else {})} for k, v in rows.items()},
           "applied": [k for k in RULE_ORDER if rows[k]],
           "mechanism_uncertain_halves": [h for h in HALVES if UNCERTAIN in (ac_mix[h], as_mix[h])],
           "constraints_ok": bool(constraints_ok), "goal_met": bool(goal_met)}
    return out


# ----------------------------------------------------------------------------------------------- anchor diagnostics
def anchor_labels(labels16: np.ndarray, anchors) -> np.ndarray:
    return np.asarray(labels16)[:, list(anchors)]


def anchor_accuracy(pred: np.ndarray, labels: np.ndarray, rows: np.ndarray) -> dict:
    """Accuracy of the time logits at the anchors of the given windows, and class-wise recall over those anchors."""
    p, y = np.asarray(pred)[rows], np.asarray(labels)[rows]
    out = {"windows": int(len(rows)), "anchors": int(y.size), "accuracy": float((p == y).mean()) if y.size else None, "recall": {}}
    for k, name in enumerate(CLASSES):
        sel = y == k
        out["recall"][name] = {"anchors": int(sel.sum()), "recall": float((p[sel] == k).mean()) if sel.any() else None}
    return out


def change_points(labels: np.ndarray, rows: np.ndarray) -> list[tuple[int, int]]:
    """(window, j) for every reference class change between anchors j and j + 1 of the given windows."""
    y = np.asarray(labels); out = []
    for i in np.asarray(rows, int):
        out += [(int(i), j) for j in np.flatnonzero(y[i, :-1] != y[i, 1:]).tolist()]
    return out


def change_summary(pred: np.ndarray, labels: np.ndarray, points: list[tuple[int, int]], sources) -> dict:
    """Both anchors right at each change; sources(i, j) gives the (recordings, sessions) sets of the two anchor frames."""
    p, y = np.asarray(pred), np.asarray(labels)
    both = [(i, j) for i, j in points if p[i, j] == y[i, j] and p[i, j + 1] == y[i, j + 1]]
    def count(pts):
        recs, ses = set(), set()
        for i, j in pts:
            r, s = sources(i, j); recs |= set(r); ses |= set(s)
        return {"changes": len(pts), "recordings": len(recs), "sessions": len(ses), "few_units": len(ses) < FEW_UNITS}
    return {"all": count(points), "both_right": count(both), "both_right_share": (len(both) / len(points)) if points else None}


# ----------------------------------------------------------------------------------------------- markdown
def tables_md(res: dict) -> str:
    parts = ["# Video dense supervision round: tables\n",
             "Recompute: `python scripts/report_video_dense_supervision.py --config configs/experiments/video_dense_supervision_v1.yaml --evaluate`\n",
             "Each system cell gives the selection half, then the calibration half.\n"]
    sysd = res.get("systems", {})
    if sysd:
        rows = [{"system": k, "row": row, "half": h, "segment F1@50": r.get("segment_f1_50_multi"), "boundary P": r.get("boundary_precision_multi"),
                 "boundary R": r.get("boundary_recall_multi"), "boundary F1": r.get("boundary_f1_multi"), "native macro-F1": r.get("native_macro_f1"),
                 "switches per clip": r.get("native_internal_cuts_per_clip"), "same-view join cuts": r.get("same_class_cut_rate_multi"),
                 "A4C read as A5C": r.get("A4C read as A5C"), "A5C read as A4C": r.get("A5C read as A4C")}
                for k, per in sysd.items() for row, by in per.items() if row in ("argmax", "b3", "fixed", "row") for h, r in by.items()]
        parts += ["## Systems\n", md_table(rows)]
    sc = res.get("system_comparisons", {})
    if sc:
        rows = [{"comparison": k, "segment F1@50": ci(v["bootstrap"]["multi segment F1@50"]), "native macro-F1": ci(v["bootstrap"]["native macro-F1"]),
                 "boundary recall": ci(v["bootstrap"]["multi boundary recall"])} for k, v in sc.items()]
        parts += ["## System comparisons (paired bootstrap; streams reuse recordings)\n", md_table(rows)]
    mix = res.get("mixed", {})
    if mix.get("accuracy"):
        rows = [{"system": k, "half": h, **{s: v[h].get(s) for s in ("all mixed", "<= 7 of 16", "8 of 16", "9-11 of 16", "12-15 of 16", "A4C/A5C pair windows")}}
                for k, v in mix["accuracy"].items() for h in HALVES if h in v]
        parts += ["## Mixed-window target accuracy (CLS)\n", md_table(rows)]
    if mix.get("comparisons"):
        rows = [{"comparison": k, "session": ci(v["session"]) if v["session"]["estimable"] else "not estimable",
                 "stream": ci(v["stream"]) if v["stream"]["estimable"] else "not estimable", "sessions": v["session"]["units"], "windows": v["session"]["windows"]}
                for k, v in mix["comparisons"].items() if "|all mixed|" in k]
        parts += ["## Mixed-window comparisons (all mixed windows)\n", md_table(rows)]
    for key, title in (("anchors", "Anchor diagnostics (time logits; not systems)"), ("verdicts", "Verdicts"), ("goal", "Goal"), ("decision", "Decision"),
                       ("references_check", "Reference reproduction"), ("cost", "Cost")):
        parts += [f"## {title}\n", "```\n" + json.dumps(res.get(key, "pending"), indent=1, default=str) + "\n```\n"]
    return "\n".join(parts)
