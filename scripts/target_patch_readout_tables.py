"""Pure summaries of the target-anchored readout round (configs/experiments/target_patch_readout_v1.yaml; plan Tasks 3-4).

- Paired accuracy differences on window subsets with a bootstrap over groups (target sessions, or whole streams as the
  sensitivity), reported as not estimable when a subset has no window or fewer than 2 groups.
- Mixed-window subsets by how many of the 16 frames carry the requested sample's reference view, A4C/A5C pair windows,
  and support counts (windows, recordings, sessions, reference runs and the largest run's share).
- Verdicts with the stream-sensitivity rule, and the fixed decision table of Task 4.
Nothing here reads a file or a recording id; callers pass arrays.
"""
from __future__ import annotations

import json

import numpy as np

from scripts.raw_video_context_tables import NEG, NONE, POS, ci, consistency, md_table

HALVES = ("selection", "calibration")
A4C, A5C = 2, 3
BINS = ("<= 7 of 16", "8 of 16", "9-11 of 16", "12-15 of 16")
UNCERTAIN = "uncertain (stream sensitivity conflicts)"
MIN_UNITS, FEW_UNITS = 2, 10


def share_bin(k: int) -> str:
    return BINS[0] if k <= 7 else BINS[1] if k == 8 else BINS[2] if k <= 11 else BINS[3]


def group_bootstrap(correct_a: np.ndarray, correct_b: np.ndarray, groups: np.ndarray, draws: int = 2000, seed: int = 20261002) -> dict:
    """Accuracy of a minus accuracy of b on the same windows, resampling groups with replacement (same draws for both)."""
    a, b, g = np.asarray(correct_a, float), np.asarray(correct_b, float), np.asarray(groups)
    if a.size != b.size or a.size != g.size:
        raise ValueError("both systems must cover the same windows")
    keys = np.unique(g)
    out = {"windows": int(a.size), "units": int(keys.size), "few_units": bool(keys.size < FEW_UNITS)}
    if a.size == 0 or keys.size < MIN_UNITS:
        return out | {"delta": None, "ci95": [None, None], "estimable": False}
    by = [np.flatnonzero(g == k) for k in keys]; rng = np.random.default_rng(seed); reps = []
    for _ in range(int(draws)):
        idx = np.concatenate([by[i] for i in rng.integers(0, keys.size, keys.size)])
        reps.append(a[idx].mean() - b[idx].mean())
    return out | {"delta": float(a.mean() - b.mean()), "ci95": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))], "estimable": True}


def verdict(session: list[dict], stream: list[dict]) -> str:
    """Session verdict across seeds, called uncertain when the stream sensitivity is consistent the other way."""
    if not session or any(not d.get("estimable", True) for d in session):
        return "not estimable"
    v, s = consistency(session), consistency(stream) if stream and all(d.get("estimable", True) for d in stream) else NONE
    if (v == POS and s == NEG) or (v == NEG and s == POS):
        return UNCERTAIN
    return v


def subset_support(rows: np.ndarray, recordings: np.ndarray, sessions: np.ndarray, runs: np.ndarray, labels: np.ndarray) -> dict:
    """Windows, distinct recordings, sessions and reference runs of a subset, per target class, and the largest run's share."""
    out = {"windows": int(rows.size), "recordings": int(len(set(recordings[rows].tolist()))), "sessions": int(len(set(sessions[rows].tolist()))),
           "reference_runs": int(len(set(runs[rows].tolist()))), "per_class": {}}
    if rows.size:
        _, counts = np.unique(runs[rows], return_counts=True); out["largest_run_share"] = float(counts.max() / rows.size)
        for k in sorted(set(labels[rows].tolist())):
            r = rows[labels[rows] == k]
            out["per_class"][str(k)] = {"windows": int(r.size), "recordings": int(len(set(recordings[r].tolist()))), "sessions": int(len(set(sessions[r].tolist())))}
    return out


def mixed_subsets(labels16: np.ndarray, label: np.ndarray, mixed: np.ndarray) -> dict[str, np.ndarray]:
    """Row indices of the mixed windows overall, by share bin, and of A4C/A5C pair windows."""
    share = (labels16 == label[:, None]).sum(1); rows = np.flatnonzero(mixed)
    out = {"all mixed": rows}
    out |= {b: rows[np.array([share_bin(int(k)) == b for k in share[rows]], bool)] if rows.size else rows for b in BINS}
    pair = np.array([set(x.tolist()) == {A4C, A5C} for x in labels16[rows]], bool) if rows.size else np.zeros(0, bool)
    out["A4C/A5C pair windows"] = rows[pair]
    return out


def decide(rules: dict, primary: dict, native: dict, system: dict, goals: dict) -> dict:
    """primary[c][h] verdicts on the 8-of-16 mixed metric for c in ('Target - Global', 'Target - CLS'); native[c][h] for
    the A4C/A5C balanced accuracy and system[c][h] for B3 segment F1@50 (c also 'Target - Ordered')."""
    tg, tc = primary["Target - Global"], primary["Target - CLS"]
    both = lambda v, w: all(v[h] == w for h in HALVES)  # noqa: E731
    pos_halves = [h for h in HALVES if tg[h] == POS and tc[h] == POS]
    native_worse = any(native[c][h] == NEG for c in ("Target - CLS", "Target - Global") for h in HALVES)
    system_gain = both(system["Target - CLS"], POS)
    applies = []
    if len(pos_halves) == 2:
        applies.append(("target_position", rules["target_position"]))
    elif pos_halves:
        applies.append(("target_position_one_half", rules["target_position_one_half"] + f" (half: {pos_halves[0]})"))
    cls_only = [h for h in HALVES if tc[h] == POS and tg[h] != POS]
    if cls_only:
        applies.append(("patch_not_position", rules["patch_not_position"] + f" (half: {', '.join(cls_only)})"))
    if pos_halves and (native_worse or not system_gain):
        applies.append(("tradeoff", rules["tradeoff"]))
    mech_any = any(tg[h] == POS for h in HALVES); sys_any = any(system[c][h] == POS for c in ("Target - CLS", "Target - Ordered") for h in HALVES)
    if not mech_any and not sys_any:
        applies.append(("no_gain", rules["no_gain"]))
    beats = both(system["Target - CLS"], POS) and both(system["Target - Ordered"], POS)
    met = [g for g, v in goals.items() if v["met"]]
    return {"rows_that_apply": [{"rule": k, "text": t} for k, t in applies], "target_position_halves": pos_halves, "cls_only_halves": cls_only,
            "native_worse": native_worse, "system_gain_over_cls_both_halves": system_gain, "beats_existing": beats,
            "beats_existing_text": rules["beats_existing"], "goal_met_groups": met, "goal_text": rules["goal_and_baselines"]}


def tables_md(res: dict) -> str:
    parts = ["# Target-anchored readout round: tables\n",
             "Recompute: `python scripts/report_target_patch_readout.py --config configs/experiments/target_patch_readout_v1.yaml --evaluate`\n"]
    mix = res.get("mixed", {}).get("accuracy", {})
    if mix:
        rows = [{"system": k, "half": h, **{s: v[h].get(s) for s in ("8 of 16", "all mixed", "<= 7 of 16", "9-11 of 16", "12-15 of 16", "A4C/A5C pair windows", "marker away from slot 8")}}
                for k, v in mix.items() for h in HALVES if h in v]
        parts += ["## Mixed-window accuracy\n", md_table(rows)]
    comp = res.get("mixed", {}).get("comparisons", {})
    if comp:
        rows = [{"comparison": k, "session": ci(v["session"]) if v["session"]["estimable"] else "not estimable",
                 "stream": ci(v["stream"]) if v["stream"]["estimable"] else "not estimable", "sessions": v["session"]["units"], "windows": v["session"]["windows"]}
                for k, v in comp.items() if "|8 of 16|" in k or "|all mixed|" in k]
        parts += ["## Mixed-window comparisons (8 of 16 and all mixed)\n", md_table(rows)]
    nat = res.get("native", {}).get("metrics", {})
    if nat:
        rows = [{"system": k, "half": h, **{m: v[h].get(m) for m in ("A4C/A5C balanced accuracy", "A5C-A4C margin AUC", "native window accuracy", "A4C recall", "A5C recall",
                                                                     "A4C read as A5C (windows)", "A5C read as A4C (windows)")}} for k, v in nat.items() for h in HALVES]
        parts += ["## Native window recognition\n", md_table(rows)]
    sysd = res.get("systems", {})
    if sysd:
        rows = [{"system": k, "row": row, "half": h, "segment F1@50": r.get("segment_f1_50_multi"), "boundary P": r.get("boundary_precision_multi"),
                 "boundary R": r.get("boundary_recall_multi"), "native macro-F1": r.get("native_macro_f1"), "switches per clip": r.get("native_internal_cuts_per_clip"),
                 "same-view join cuts": r.get("same_class_cut_rate_multi"), "A5C read as A4C": r.get("A5C read as A4C"), "A4C read as A5C": r.get("A4C read as A5C")}
                for k, per in sysd.items() for row, by in per.items() if row in ("argmax", "b3", "fixed", "row") for h, r in by.items()]
        parts += ["## Systems\n", md_table(rows)]
    sc = res.get("system_comparisons", {})
    if sc:
        rows = [{"comparison": k, "segment F1@50": ci(v["bootstrap"]["multi segment F1@50"]), "native macro-F1": ci(v["bootstrap"]["native macro-F1"]),
                 "boundary recall": ci(v["bootstrap"]["multi boundary recall"])} for k, v in sc.items()]
        parts += ["## System comparisons (paired bootstrap; streams reuse recordings)\n", md_table(rows)]
    for key, title in (("verdicts", "Verdicts"), ("goal", "Goal"), ("decision", "Decision")):
        parts += [f"## {title}\n", "```\n" + json.dumps(res.get(key, "pending"), indent=1, default=str) + "\n```\n"]
    return "\n".join(parts)
