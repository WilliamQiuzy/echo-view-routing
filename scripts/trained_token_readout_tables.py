"""Pure summaries of the trained token readout round (configs/experiments/trained_token_readout_v1.yaml; plan Task 2).

- The fixed readings of the task sheet: complete progress, a usable token readout, or labels learned without a usable
  gain, with notes for classification gains without segment gains, one half only, constraint failures, uncertain halves,
  wins over the controls only and negative results. The readings are not exclusive.
- The check of every B3 choice record against the 40-setting grid.
- Changed windows with their source recordings and sessions and a few-unit flag; anchor-offset splits.
- Markdown tables. Nothing here reads a file or prints a recording id; callers pass arrays and sets.
"""
from __future__ import annotations

import json

import numpy as np

from scripts.raw_video_context_tables import NEG, NONE, POS, ci, md_table  # noqa: F401  (re-exported for the report)
from scripts.target_patch_readout_tables import UNCERTAIN

HALVES = ("selection", "calibration")
FEW_UNITS = 10
RULE_ORDER = ("complete_progress", "useful_readout", "labels_learned_no_usable_gain", "end")
SYS = ("token - CLS", "token - Ordered", "token - CLS-only token", "token - Shuffled token")


# ----------------------------------------------------------------------------------------------- readings
def decide(sysv: dict, mixed: dict, constraints_ok: bool, goal_met: bool, rules: dict | None = None) -> dict:
    """sysv[c][h]: B3 multi segment F1@50 verdicts of the aligned token system for c in SYS; mixed[h]: the all-mixed
    same-target classification verdict of Aligned token - Aligned CLS (session bootstrap, UNCERTAIN when the stream
    sensitivity is consistent the other way); constraints_ok: a feasible B3 setting, native macro-F1 at the floor and
    boundary recall >= 0.95 in every seed and half; goal_met: also the three-seed means."""
    both = lambda x: all(x[h] == POS for h in HALVES)  # noqa: E731
    pos_halves = lambda x: [h for h in HALVES if x[h] == POS]  # noqa: E731
    cls_f1 = sysv["token - CLS"]
    complete = all(both(sysv[c]) for c in SYS) and bool(constraints_ok) and bool(goal_met)
    useful = both(mixed) and both(cls_f1) and bool(constraints_ok)
    rows = {"complete_progress": complete, "useful_readout": useful, "labels_learned_no_usable_gain": not complete and not useful, "end": True}
    one_half = sorted({*(pos_halves(mixed) if len(pos_halves(mixed)) == 1 else []), *(pos_halves(cls_f1) if len(pos_halves(cls_f1)) == 1 else [])},
                      key=HALVES.index)
    controls = any(sysv[c][h] == POS for c in ("token - CLS-only token", "token - Shuffled token") for h in HALVES)
    neg = {k: [h for h in HALVES if x[h] == NEG] for k, x in (("segment F1 vs CLS", cls_f1), ("classification vs CLS", mixed))}
    notes = {"classification_without_segments": bool(pos_halves(mixed)) and not both(cls_f1), "one_half_only": one_half,
             "constraints_failed": not constraints_ok, "uncertain_halves": [h for h in HALVES if mixed[h] == UNCERTAIN],
             "only_controls": controls and not pos_halves(cls_f1) and not pos_halves(mixed), "negative": {k: x for k, x in neg.items() if x}}
    return {"rows": {k: {"applies": bool(a), **({"reading": rules[k]} if rules and k in rules else {})} for k, a in rows.items()},
            "applied": [k for k in RULE_ORDER if rows[k]], "notes": notes, "constraints_ok": bool(constraints_ok), "goal_met": bool(goal_met)}


def check_choice_record(rec: dict, grid: dict) -> None:
    """A B3 choice record must hold the full 40-setting grid in its order and a chosen setting from it."""
    want = [(int(h), float(q)) for h in grid["half_window"] for q in grid["quantile"]]
    got = [(int(r["params"]["half_window"]), float(r["params"]["threshold_quantile"])) for r in rec["grid"]]
    if got != want or any(int(r["params"]["nms_radius"]) != int(grid["nms_radius"]) for r in rec["grid"]):
        raise SystemExit("a choice record does not hold the configured 40-setting grid")
    p = rec["params"]
    if (int(p["half_window"]), float(p["threshold_quantile"])) not in want:
        raise SystemExit("a chosen setting is not in the configured grid")


# ----------------------------------------------------------------------------------------------- case summaries
def _units(rows: np.ndarray, recordings: np.ndarray, sessions: np.ndarray) -> dict:
    ses = len(set(sessions[rows].tolist()))
    return {"windows": int(rows.size), "recordings": len(set(recordings[rows].tolist())), "sessions": ses, "few_units": ses < FEW_UNITS}


def changes(correct_a: np.ndarray, correct_b: np.ndarray, recordings: np.ndarray, sessions: np.ndarray) -> dict:
    """Windows a gets right and b wrong (fixed by a) and the reverse (new errors of a), with their sources."""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool); rec, ses = np.asarray(recordings), np.asarray(sessions)
    return {"fixed": _units(np.flatnonzero(a & ~b), rec, ses), "new": _units(np.flatnonzero(~a & b), rec, ses)}


def offset_split(offsets: np.ndarray, correct_token: np.ndarray, correct_cls: np.ndarray) -> dict:
    """Same-target accuracy of the token readout and of the CLS output on exact (offset 0) and non-exact windows."""
    off = np.asarray(offsets); out = {}
    for name, sel in (("exact", off == 0), ("non_exact", off != 0)):
        n = int(sel.sum())
        out[name] = {"windows": n, "accuracy_token": float(np.asarray(correct_token)[sel].mean()) if n else None,
                     "accuracy_cls": float(np.asarray(correct_cls)[sel].mean()) if n else None}
    return out


# ----------------------------------------------------------------------------------------------- markdown
ROWS = ("argmax", "b3", "b3_at_cls_setting", "fixed_ordered_setting", "row")
CLASSES = ("PLAX", "PSAX", "A4C", "A5C", "SC4C")


def _system_rows(sysd: dict) -> list[dict]:
    out = []
    for k, per in sysd.items():
        for row in ROWS:
            for h, r in per.get(row, {}).items():
                out.append({"system": k, "row": row, "half": h, "segment F1@50": r.get("segment_f1_50_multi"), "boundary P": r.get("boundary_precision_multi"),
                            "boundary R": r.get("boundary_recall_multi"), "boundary F1": r.get("boundary_f1_multi"), "native macro-F1": r.get("native_macro_f1"),
                            **{f"{c} recall": (r.get("native_recall") or {}).get(c) for c in CLASSES}, **{f"{c} F1": (r.get("native_f1") or {}).get(c) for c in CLASSES},
                            "switches per clip": r.get("native_internal_cuts_per_clip"), "same-view join cuts": r.get("same_class_cut_rate_multi"),
                            "A4C read as A5C": r.get("A4C read as A5C"), "A5C read as A4C": r.get("A5C read as A4C")})
    return out


def tables_md(res: dict) -> str:
    """Every table of the summary; recording, session and stream ids never reach it (callers check)."""
    parts = ["# Trained token readout round: tables\n",
             "Recompute: `python scripts/trained_token_readout.py --config configs/experiments/trained_token_readout_v1.yaml --selection`\n",
             "Seeds are not extra cases. Development halves are reused, so the intervals are approximate. A consistent difference means "
             "all three seeds' 95% intervals exclude zero on the same side in a half.\n"]
    if res.get("systems"):
        parts += ["## Systems\n", md_table(_system_rows(res["systems"]))]
    if res.get("choices"):
        rows = [{"system": k, "status": v["status"], "half_window": v["params"]["half_window"], "quantile": v["params"].get("threshold_quantile"),
                 "threshold": v["params"].get("threshold"), "feasible settings": v["n_feasible"]} for k, v in res["choices"].items()]
        rows += [{"system": k, "status": v["b3_choice"]["status"], "half_window": v["b3_choice"]["params"]["half_window"],
                  "quantile": v["b3_choice"]["params"].get("threshold_quantile"), "threshold": v["b3_choice"]["params"].get("threshold"),
                  "feasible settings": v["b3_choice"].get("n_feasible")} for k, v in res.get("systems", {}).items() if not k.startswith("token_") and "b3_choice" in v]
        parts += ["## B3 settings (token systems chosen once on calibration; CLS and Ordered stored)\n", md_table(rows)]
    if res.get("system_comparisons"):
        rows = []
        for k, v in res["system_comparisons"].items():
            b = v["bootstrap"]; seg = v.get("segment_changes", {}); nat = v.get("native_changes", {}).get("all clips", {})
            rows.append({"comparison": k, "segment F1@50": ci(b["multi segment F1@50"]), "native macro-F1": ci(b["native macro-F1"]),
                         "boundary recall": ci(b["multi boundary recall"]), "boundary F1": ci(b["multi boundary F1"]),
                         "segments matched by a only": seg.get("matched by a only", {}).get("segments"), "by b only": seg.get("matched by b only", {}).get("segments"),
                         "native clips fixed": nat.get("fixed"), "native clips newly wrong": nat.get("new")})
        parts += ["## System comparisons (seed | row | comparison | half; paired bootstrap, native clips by session, streams by stream)\n", md_table(rows)]
    if res.get("verdicts"):
        rows = [{"row": r, "comparison": n, "metric": m, **{h: x[h] for h in HALVES}} for r, by in res["verdicts"].items() for n, mm in by.items() for m, x in mm.items()]
        parts += ["## System verdicts over seeds\n", md_table(rows)]
    cl = res.get("classification", {})
    if cl.get("accuracy"):
        rows = [{"system": k, "half": h, **x} for k, by in cl["accuracy"].items() for h, x in by.items()]
        parts += ["## Same-target window accuracy at the request\n", md_table(rows)]
    if cl.get("comparisons"):
        rows = []
        for k, v in cl["comparisons"].items():
            if "session" not in v:
                continue
            rows.append({"comparison": k, "session bootstrap": ci(v["session"]) if v["session"]["estimable"] else "not estimable",
                         "stream bootstrap": ci(v["stream"]) if v["stream"]["estimable"] else "not estimable", "windows": v["session"]["windows"],
                         "sessions": v["session"]["units"], "fixed windows (rec./ses.)": "{windows} ({recordings}/{sessions})".format(**v["changes"]["fixed"]),
                         "new errors (rec./ses.)": "{windows} ({recordings}/{sessions})".format(**v["changes"]["new"]),
                         "fewer than 10 sessions": v["session"]["few_units"] or v["changes"]["fixed"]["few_units"] or v["changes"]["new"]["few_units"]})
        parts += ["## Same-target classification, token readout minus its own arm's CLS (seed | subset | half)\n", md_table(rows)]
        rows = [{"comparison": k, **{f"{e} {m}": v[e][m] for e in ("exact", "non_exact") for m in ("windows", "accuracy_token", "accuracy_cls")}}
                for k, v in cl["comparisons"].items() if "exact" in v]
        parts += ["## Same-target accuracy on exact and non-exact anchors\n", md_table(rows)]
    if cl.get("verdicts"):
        parts += ["## Same-target classification verdicts over seeds\n", md_table([{"readout | subset": k, **v} for k, v in cl["verdicts"].items()])]
    for key, title in (("positions", "Anchor positions and the per-sample split"), ("units", "Short segments and X-Y-X middles"), ("goal", "Goal and constraints"),
                       ("decision", "Fixed readings"), ("stored_rows", "Stored rows of earlier systems (from A02's summary)"), ("post_hoc", "Post hoc additions"),
                       ("cost", "Cost")):
        parts += [f"## {title}\n", "```\n" + json.dumps(res.get(key, "pending"), indent=1, default=str) + "\n```\n"]
    return "\n".join(parts)
