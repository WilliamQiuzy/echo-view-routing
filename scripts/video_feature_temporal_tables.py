"""Pure summaries of the video feature temporal round (configs/experiments/video_feature_temporal_v1.yaml; Task 3).

- ASRF diagnostics: boundary peaks by the official argrelmax rule, whether each reference edge of a unit was proposed
  (a peak within 5 samples) and whether a proposed edge was merged by the refinement (no refined change within 5
  samples); whether the frame (class) branch and the source classifier name the unit's class; the readings of unmatched
  units and wrong native clips.
- The frozen diagnostic list (units frozen Ordered + B3 leaves unmatched while its own argmax majority is right): fixed
  units and new misses of every new system, with distinct recordings and sessions.
- The overlap of the cases a temporal head fixes with those Linear + B3 fixes.
- The fixed readings of the task sheet's Task 4 and the markdown tables.
Nothing here reads a file; recordings appear only as sets that are counted.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Iterable

import numpy as np

from echo_routing.evaluate.failure_units import session_of

HALVES = ("selection", "calibration")
CLASSES = ("PLAX", "PSAX", "A4C", "A5C", "SC4C")
POS, NEG, NONE = "consistent positive", "consistent negative", "no consistent difference detected"
UNIT_READINGS = ("source right, class branch wrong", "class branch right, refinement wrong", "edge not proposed", "proposed but merged")
NATIVE_READINGS = UNIT_READINGS[:2]
PRIMARY = "ASRF - (Linear + B3)"
VS_ORDERED = "ASRF - (frozen Ordered + B3)"
RULE_ORDER = ("system_gain", "selection_only", "linear_explains", "tradeoff", "no_gain", "end")


def majority(labels) -> int:
    return int(np.bincount(np.asarray(labels, int), minlength=len(CLASSES)).argmax())


def count(recs: Iterable[str]) -> dict:
    r = set(recs)
    return {"recordings": len(r), "sessions": len({session_of(v) for v in r})}


# ----------------------------------------------------------------------------------------------- ASRF diagnostics
def boundary_peaks(prob: np.ndarray, threshold: float) -> np.ndarray:
    """The official libs/metric.argrelmax on a copy: values below the threshold set to 0, the first sample always a peak,
    then strict relative maxima; the last sample never."""
    p = np.array(prob, dtype=np.float64, copy=True); p[p < threshold] = 0.0
    if p.size < 2:
        return np.zeros(min(1, p.size), int)
    peak = np.concatenate([[True], (p[:-2] < p[1:-1]) & (p[2:] < p[1:-1]), [False]])
    return np.flatnonzero(peak)


def edge_status(edges: list[int], peaks: np.ndarray, refined_changes: np.ndarray, hit: int) -> dict:
    """Per reference edge: proposed (a boundary peak other than the sequence start within `hit`), merged (proposed, but no
    refined label change within `hit`) or not proposed."""
    pk = np.asarray(peaks, int); pk = pk[pk > 0]; ch = np.asarray(refined_changes, int); out = {"edges": len(edges), "proposed": 0, "merged": 0, "not_proposed": 0}
    for e in edges:
        prop = bool(np.any(np.abs(pk - e) <= hit)); kept = bool(np.any(np.abs(ch - e) <= hit))
        out["proposed"] += int(prop); out["merged"] += int(prop and not kept); out["not_proposed"] += int(not prop)
    return out


def asrf_unit(row: dict, unit, frame_labels: np.ndarray, source_labels: np.ndarray, peaks: np.ndarray, refined_changes: np.ndarray, hit: int) -> dict:
    """One reference unit under ASRF: its unit_outcomes row, the edge status, the class-branch and source majorities and,
    for an unmatched unit, the four non-exclusive readings."""
    sl = slice(unit.start, unit.end); lab = int(unit.label)
    edges = [e for e, nb in ((unit.start, unit.left), (unit.end, unit.right)) if nb is not None]
    st = edge_status(edges, peaks, refined_changes, hit); cls_ok = majority(frame_labels[sl]) == lab; src_ok = majority(source_labels[sl]) == lab
    wrong = not row["matched"]
    readings = {"source right, class branch wrong": wrong and src_ok and not cls_ok, "class branch right, refinement wrong": wrong and cls_ok,
                "edge not proposed": wrong and st["not_proposed"] > 0, "proposed but merged": wrong and st["merged"] > 0}
    recs = sorted({s[0] for s in unit.sources})
    return {"matched": bool(row["matched"]), "cause": row["cause"], **st, "class_branch_right": bool(cls_ok), "source_right": bool(src_ok),
            "readings": {k: bool(v) for k, v in readings.items()}, "recordings": recs, "sessions": sorted({session_of(v) for v in recs})}


def asrf_native(clip: dict) -> dict:
    """A native clip under ASRF: label, decode (refined majority), frame (class-branch majority), source (source majority)."""
    wrong = clip["decode"] != clip["label"]; cls_ok = clip["frame"] == clip["label"]; src_ok = clip["source"] == clip["label"]
    return {**clip, "readings": {"source right, class branch wrong": bool(wrong and src_ok and not cls_ok),
                                 "class branch right, refinement wrong": bool(wrong and cls_ok)}}


def unit_summary(rows: list[dict]) -> dict:
    """Unmatched units with distinct sources, by edge status and by reading."""
    wrong = [r for r in rows if not r["matched"]]
    def c(sel):
        return {"units": len(sel), **count(v for r in sel for v in r["recordings"])}
    return {"units": len(rows), "unmatched": c(wrong), "edges": {k: int(sum(r[k] for r in wrong)) for k in ("edges", "proposed", "merged", "not_proposed")},
            "readings": {k: c([r for r in wrong if r["readings"][k]]) for k in UNIT_READINGS}}


def native_summary(clips: list[dict]) -> dict:
    wrong = [c for c in clips if c["decode"] != c["label"]]
    def c(sel):
        return {"clips": len(sel), **count(x["video_id"] for x in sel)}
    return {"clips": len(clips), "wrong": c(wrong), "readings": {k: c([x for x in wrong if x["readings"][k]]) for k in NATIVE_READINGS}}


# ----------------------------------------------------------------------------------------------- the frozen list and overlaps
def list_outcomes(listed: set, ordered_matched: set, new_matched: set, sources: Callable) -> dict:
    """Units of the frozen list fixed by a new system, and units Ordered + B3 matched that it misses, with their sources."""
    def c(keys):
        return {"units": len(keys), **count(v for k in keys for v in sources(k))}
    return {"listed": c(listed), "fixed": c(listed & new_matched), "new_misses": c(ordered_matched - new_matched)}


def overlap(fixed_head: set, fixed_linear: set) -> dict:
    n = len(fixed_head); m = len(fixed_head & fixed_linear)
    return {"fixed": n, "also_by_linear": m, "share": (m / n) if n else None}


# ----------------------------------------------------------------------------------------------- readings
def decide(ver: dict, goals: dict, overlap_share: float | None, trade: dict, rules: dict | None = None) -> dict:
    """The fixed readings. ver[comparison][half][metric] are verdicts; goals['ASRF'] from the goal check; overlap_share the
    pooled share of ASRF's fixes also made by Linear + B3; trade[f'{head}|{half}'] the trade-off condition."""
    f1, nat = "multi segment F1@50", "native macro-F1"
    lin, ordr = ver[PRIMARY], ver[VS_ORDERED]
    both = lambda v: all(v[h][f1] == POS for h in HALVES)  # noqa: E731
    none_pos = lambda v: all(v[h][f1] != POS for h in HALVES)  # noqa: E731
    rows = {"system_gain": {"applies": both(lin) and both(ordr) and not goals["ASRF"]["constraint_failures"]},
            "selection_only": {"applies": ordr["selection"][f1] == POS and ordr["calibration"][f1] != POS and ordr["calibration"][nat] != POS},
            "linear_explains": {"applies": none_pos(lin) and overlap_share is not None and overlap_share > 0.5, "share": overlap_share},
            "tradeoff": {"applies": any(trade.values()), "where": sorted(k for k, v in trade.items() if v)},
            "no_gain": {"applies": none_pos(lin) and none_pos(ordr),
                        "mstcn_leads_in": [h for h in HALVES if ver["ASRF - MS-TCN"][h][f1] == NEG]},
            "end": {"applies": True}}
    for k, r in rows.items():
        r["applies"] = bool(r["applies"])
        if rules and k in rules:
            r["reading"] = rules[k]
    return {"rows": rows, "applied": [k for k in RULE_ORDER if rows[k]["applies"]]}


# ----------------------------------------------------------------------------------------------- markdown
def f3(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def f4(x) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def table(head: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows] + [""]


def ci(d: dict) -> str:
    lo, hi = d["ci95"]
    return f"{d['delta']:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def _systems(res: dict) -> list[str]:
    cols = [("Segment F1@50", "segment_f1_50_multi", f3), ("Boundary recall", "boundary_recall_multi", f3), ("Boundary precision", "boundary_precision_multi", f3),
            ("Native macro-F1", "native_macro_f1", f4), ("Label switches per native clip", "native_internal_cuts_per_clip", f3),
            ("Cuts at same-view joins", "same_class_cut_rate_multi", f3), ("A4C read as A5C (clips)", "A4C read as A5C", str), ("A5C read as A4C (clips)", "A5C read as A4C", str)]
    rows = [[name, *[" / ".join(fmt(by[h][k]) if by.get(h) else "n/a" for h in HALVES) for _, k, fmt in cols]] for name, by in res["systems"].items()]
    return table(["System", *[c for c, _, _ in cols]], rows)


def _comparisons(res: dict) -> list[str]:
    rows = []
    for name, by in res["verdicts"].items():
        for h in HALVES:
            v = by[h]; seeds = sorted({k.split("|")[0] for k in res["comparisons"] if k.split("|")[1] == name and k.endswith(h)})
            rows.append([name, h, "; ".join(ci(res["comparisons"][f"{s}|{name}|{h}"]["bootstrap"]["multi segment F1@50"]) for s in seeds), v["multi segment F1@50"],
                         ", ".join(f"{m} {x}" for m, x in v.items() if m != "multi segment F1@50" and x != NONE) or "none"])
    return table(["Comparison", "Half", "Segment F1@50 (seeds 0, 1, 2)", "Verdict", "Other consistent differences"], rows)


def _goal(res: dict) -> list[str]:
    rows = [[name, " / ".join(f4(g["mean_segment_f1"][h]) for h in HALVES), " / ".join(f"{g['margin'][h]:+.4f}" for h in HALVES),
             f"{g['calibration_sensitivity_margin']:+.4f}", len(g["constraint_failures"]), g["met"]] for name, g in res["goal"].items()]
    return table(["System", "Three-seed mean", "Margin to the minimum", "Calibration sensitivity margin", "Constraint failures", "Goal met"], rows)


def _json(key: str):
    return lambda r: ["```", json.dumps(r[key], indent=1, default=str), "```", ""]


def tables_md(res: dict) -> str:
    out = ["# Video feature temporal round: tables", "",
           "Recompute: `python scripts/report_video_feature_temporal.py --config configs/experiments/video_feature_temporal_v1.yaml --evaluate`.",
           "Each cell gives the selection half, then the calibration half.", ""]
    sections = [("## Systems", ("systems",), _systems), ("## Comparisons", ("comparisons", "verdicts"), _comparisons), ("## Goal", ("goal",), _goal),
                ("## Linear + B3 settings", ("linear_b3",), _json("linear_b3")), ("## Frozen diagnostic list", ("diagnostic_list",), _json("diagnostic_list")),
                ("## ASRF decomposition", ("asrf_decomposition",), _json("asrf_decomposition")), ("## Overlap with Linear + B3", ("linear_overlap",), _json("linear_overlap")),
                ("## Short segments and X-Y-X middles", ("mechanism",), _json("mechanism")), ("## Native errors", ("native_errors",), _json("native_errors")),
                ("## Parity and counts", ("parity", "counts"), lambda r: _json("parity")(r) + _json("counts")(r)), ("## Cost", ("cost",), _json("cost")),
                ("## Decision", ("decision",), lambda r: [f"Applied: {', '.join(r['decision']['applied'])}", ""] + _json("decision")(r))]
    for head, keys, fn in sections:
        out += [head, ""] + (fn(res) if all(k in res for k in keys) else ["pending", ""])
    return "\n".join(out)
