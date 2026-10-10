"""Measures, decision and tables of the predicted-partition round (CPU only; plan
docs/superpowers/plans/2026-10-09-fact-predicted-partition.md). scripts/report_fact_predicted_partition.py decodes every
system and calls these; they read decoded records, never a model.

A stage record holds one sequence's reference labels, its source fragments, the readouts E1 (argmax of the frozen E1
probabilities), i, u and U (each FACT block's frame-branch argmax) and final (the official output), the partition the
last block used and, for the Predicted group, the head's cuts. Units are reference runs. Matching is segment_f1's
class-aware one-to-one rule at IoU 0.5, and every count carries its distinct source recordings and sessions.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np

from scripts.ev9v_complementarity import md_table
from scripts.fact_localisation_tables import own_share, segment_measures, sources, unit_groups

from echo_routing.evaluate.failure_units import session_of
from echo_routing.evaluate.metrics_boundary import match_boundaries
from echo_routing.temporal.dev_run import HALVES
from echo_routing.temporal.segments import boundaries_from_labels, labels_to_segments

CHAIN = ("E1", "i", "u", "U", "final")
GROUPS = ("all", "short", "xyx", "a4c_a5c_xyx")
A4C, A5C = 2, 3


# ----------------------------------------------------------------------------------------------- stage tracking
def stage_units(recs: list[dict], tol: float, short_max: int) -> list[dict]:
    """One unit per reference run with its groups, sources, own share of the last block's partition, and whether each
    readout of the chain matches it."""
    out = []
    for r in recs:
        ref = np.asarray(r["ref"]); true = labels_to_segments(ref); part = [tuple(x) for x in r["partition"]]
        per = {n: segment_measures(ref, np.asarray(r["stages"][n]), tol) for n in CHAIN}
        for k, t in enumerate(true):
            src = sorted({v for v, a, b in r["frags"] if a < t.end and t.start < b})
            out.append({"kind": "native" if r["native"] else "stream", "seq": r["id"], "k": k, "length": t.length, "groups": unit_groups(true, k, short_max),
                        "recordings": src, "sessions": sorted({session_of(v) for v in src}), "own_share": own_share(t, part),
                        "m": {n: bool(per[n][k]["matched"]) for n in CHAIN}})
    return out


def first_unmatched(units: list[dict]) -> dict:
    """Runs missed by the official output, by the earliest readout of the chain where they were unmatched, with sources;
    and, per earliest readout, how many runs first unmatched there the official output matches again."""
    first, again = {k: [] for k in CHAIN}, Counter()
    for u in units:
        k = next((s for s in CHAIN if not u["m"][s]), None)
        if k is None:
            continue
        if u["m"]["final"]:
            again[k] += 1
        else:
            first[k].append(u)
    return {k: {**sources(v), "matched_again_by_the_official_output": again[k]} for k, v in first.items()}


def stage_table(units: list[dict]) -> dict:
    """Per group: matched runs at each readout, the first-unmatched split of the official output's misses, the runs lost
    and recovered between consecutive readouts, and the runs that are the minority in every interval the last block used."""
    out = {}
    for kind in ("stream", "native"):
        for g in GROUPS:
            us = [u for u in units if u["kind"] == kind and g in u["groups"]]
            if not us:
                continue
            minority = [u for u in us if u["own_share"] < 0.5]
            out[f"{kind}:{g}"] = {**sources(us), "matched": {n: sum(u["m"][n] for u in us) for n in CHAIN}, "first_unmatched": first_unmatched(us),
                                  "transitions": {f"{a} -> {b}": {"lost": sources([u for u in us if u["m"][a] and not u["m"][b]]),
                                                                  "recovered": sources([u for u in us if u["m"][b] and not u["m"][a]])} for a, b in zip(CHAIN, CHAIN[1:])},
                                  "minority_in_every_interval": {**sources(minority), "matched_by_the_official_output": sum(u["m"]["final"] for u in minority)}}
    return out


def lost_at_i_u(units: list[dict]) -> tuple[int, int]:
    """(stream runs missed by the official output that were first unmatched at i or u, all stream runs it misses)."""
    miss = [u for u in units if u["kind"] == "stream" and not u["m"]["final"]]
    return sum(next(s for s in CHAIN if not u["m"][s]) in ("i", "u") for u in miss), len(miss)


# ----------------------------------------------------------------------------------------------- partitions and head cuts
def interval_counts(recs: list[dict]) -> dict:
    """Intervals the last block used, per sequence and per 100 samples (streams and natives apart)."""
    out = {}
    for kind in ("stream", "native"):
        rs = [r for r in recs if r["native"] == (kind == "native")]
        n = np.asarray([len(r["partition"]) for r in rs], float); samples = np.asarray([len(r["ref"]) for r in rs], float)
        if n.size:
            out[kind] = {"sequences": int(n.size), "intervals_mean": float(n.mean()), "intervals_q50": float(np.median(n)), "intervals_q90": float(np.percentile(n, 90)),
                         "intervals_per_100_samples": float(100 * n.sum() / samples.sum())}
    return out


def head_cut_stats(recs: list[dict], joins: dict, tol: float) -> dict:
    """The head's cuts, kept apart from the final label switches: against reference changes of the streams (one to one
    within tol), near same-view joins, and per native clip."""
    st, nat = Counter(), []
    for r in recs:
        cuts = np.asarray(r["head_cuts"], float)
        if r["native"]:
            nat.append(cuts.size); continue
        y = np.asarray(r["ref"]); ref = boundaries_from_labels(y); pairs = match_boundaries(cuts, ref, tol)
        same = [j for j in joins.get(r["id"], []) if y[j] == y[j - 1]]
        st.update({"cuts": int(cuts.size), "changes": int(ref.size), "matched": len(pairs), "same_view_joins": len(same),
                   "same_view_joins_with_a_cut": sum(bool(np.any(np.abs(cuts - j) <= tol)) for j in same)})
    n = np.asarray(nat, float)
    return {"stream_cut_precision": st["matched"] / st["cuts"] if st["cuts"] else None, "stream_cut_recall": st["matched"] / st["changes"] if st["changes"] else None,
            "same_view_join_cut_rate": st["same_view_joins_with_a_cut"] / st["same_view_joins"] if st["same_view_joins"] else None,
            "cuts_per_native_clip": float(n.mean()) if n.size else None, "native_clips_with_a_cut": int((n > 0).sum()), "native_clips": int(n.size),
            "counts": {k: int(v) for k, v in st.items()}}


# ----------------------------------------------------------------------------------------------- decision
def means(rows: dict, group: str, seeds: list[int]) -> dict:
    return {h: float(np.mean([rows[f"{group} s{s}"][h]["segment_f1_50_multi"] for s in seeds])) for h in HALVES}


def constraint_failures(rows: dict, group: str, seeds: list[int], c: dict) -> list[dict]:
    return [{"seed": s, "half": h, "boundary_recall": r["boundary_recall_multi"], "native_macro_f1": r["native_macro_f1"]}
            for s in seeds for h in HALVES for r in [rows[f"{group} s{s}"][h]]
            if (r["boundary_recall_multi"] or 0) < float(c["boundary_recall_min"]) or r["native_macro_f1"] < float(c["native_macro_f1_min"][h])]


def decide(fp: dict, verdicts: dict, rows: dict, iu: dict, position: dict) -> dict:
    """The fixed decision table, read with the config's decision_rules. iu: group -> seed -> half -> (lost at i or u,
    missed); Original FACT included. position: seed -> half -> segment F1@50 of head positions - even positions."""
    seeds = [int(s) for s in fp["seeds"]]; g, c = fp["goal"], fp["constraints"]
    beat = {k: all(v[h] == "consistent positive" for h in HALVES) for k, v in verdicts.items()}
    deltas = [position[s][h] for s in seeds for h in HALVES]
    stable = bool(deltas) and (all(d > 0 for d in deltas) or all(d < 0 for d in deltas))
    groups = {}
    for name, comp in (("Uniform", "Uniform - Original FACT"), ("Predicted", "Predicted - Original FACT")):
        mean, fails = means(rows, name, seeds), constraint_failures(rows, name, seeds, c)
        at_iu = all(iu[name][s][h][1] and iu[name][s][h][0] > iu[name][s][h][1] / 2 for s in seeds for h in HALVES)
        reduced = all(iu[name][s][h][0] < iu["Original FACT"][s][h][0] for s in seeds for h in HALVES)
        groups[name] = {"mean_segment_f1_50": mean, "constraint_failures": fails,
                        "goal_met": not fails and mean["selection"] >= float(g["selection_min"]) and mean["calibration"] >= float(g["calibration_min"]),
                        "goal_margin": {h: mean[h] - float(g[f"{h}_min"]) for h in HALVES},
                        "calibration_sensitivity_margin": mean["calibration"] - float(g["calibration_sensitivity"]),
                        "losses_at_i_u": at_iu, "stable_reduction": reduced, "full_system_gain": beat[comp]}
    primary = verdicts["Predicted - Uniform"]
    rows_ = fp["decision"]; apply = []
    if beat["Uniform - Original FACT"] and not beat["Predicted - Uniform"]:
        apply.append(rows_[0])
    if beat["Predicted - Uniform"] and beat["Predicted - Original FACT"] and stable:
        apply.append(rows_[1])
    if all(v["losses_at_i_u"] and (not v["stable_reduction"] or not v["full_system_gain"]) for v in groups.values()):
        apply.append(rows_[2])
    if any(primary[h] == "no consistent difference detected" for h in HALVES) or len({primary[h] for h in HALVES}) > 1:
        apply.append(rows_[3])
    return {"consistent_beat_both_halves": beat, "position_control_stable_direction": stable, "groups": groups, "primary": primary, "rows_that_apply": apply}


# ----------------------------------------------------------------------------------------------- tables
def tables_md(res: dict) -> str:
    """The generated tables of report_tables.md (a subset of summary.json)."""
    parts = ["# Predicted-partition FACT: tables\n",
             "Recompute: `python scripts/report_fact_predicted_partition.py --config configs/experiments/fact_predicted_partition_v1.yaml --evaluate`\n"]
    sysrows = []
    for name, per in res.get("systems", {}).items():
        for h in HALVES:
            r = per[h]["row"]
            sysrows.append({"system": name, "half": h, "segment F1@50": r["segment_f1_50_multi"], "boundary R": r["boundary_recall_multi"], "boundary P": r.get("boundary_precision_multi"),
                            "native macro-F1": r["native_macro_f1"], "switches per clip": r.get("native_internal_cuts_per_clip"), "same-view cut rate": r.get("same_class_cut_rate_multi"),
                            "A5C read as A4C": r.get("A5C read as A4C"), "A4C read as A5C": r.get("A4C read as A5C")})
    if sysrows:
        parts += ["## Systems\n", md_table(sysrows, list(dict.fromkeys(k for r in sysrows for k in r)))]
    comp = [{"comparison": k, "d segment F1@50": f"{c['bootstrap']['multi segment F1@50']['delta']:+.4f} [{c['bootstrap']['multi segment F1@50']['ci95'][0]:+.4f}, "
             f"{c['bootstrap']['multi segment F1@50']['ci95'][1]:+.4f}]"} for k, c in res.get("comparisons", {}).items()]
    if comp:
        parts += ["## Comparisons (seed | comparison | half)\n", md_table(comp)]
    units = [{"system|half|group": f"{k}|{h}|{g}", **{x: v[x] for x in ("segments", "matched", "recall", "missed_recordings", "missed_sessions")}}
             for k, per in res.get("units", {}).items() for h, gs in per.items() for g, v in gs.items()]
    if units:
        parts += ["## Reference units (multi streams)\n", md_table(units)]
    stage = [{"system|set|group": f"{k}|{st}|{g}", **{n: v["matched"][n] for n in CHAIN}, "segments": v["segments"],
              **{f"first lost at {n}": v["first_unmatched"][n]["segments"] for n in CHAIN}}
             for k, per in res.get("stage_tracking", {}).items() for st, gs in per.items() for g, v in gs.items() if g.endswith((":short", ":a4c_a5c_xyx", ":all"))]
    if stage:
        parts += ["## Stage tracking (matched runs per readout; misses of the official output by earliest loss)\n", md_table(stage)]
    pos = [{"seed|half": k, **v} for k, v in res.get("position_control", {}).get("rows", {}).items()]
    if pos:
        parts += ["## Position control on the Predicted models (head positions against even positions)\n", md_table(pos)]
    return "\n".join(parts + ["## Verdicts\n", "```\n" + json.dumps(res.get("verdicts", "pending"), indent=1) + "\n```\n",
                              "## Decision\n", "```\n" + json.dumps(res.get("decision", "pending"), indent=1, default=str) + "\n```\n",
                              "## Cost\n", "```\n" + json.dumps(res.get("cost", {}), indent=1, default=str) + "\n```\n"])


def plot_cases(cases: list[dict], path) -> None:
    """Every A4C/A5C X -> Y -> X middle of the development multi streams: the reference and each system's output around
    the middle, one panel per case (no selection by outcome)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    if not cases:
        return
    cmap = ListedColormap(["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974"]); n = len(cases)
    fig, axes = plt.subplots(n, 1, figsize=(12, 0.32 * len(cases[0]["tracks"]) * n + 1), squeeze=False)
    for ax, c in zip(axes[:, 0], cases):
        names = list(c["tracks"]); m = np.asarray([c["tracks"][k] for k in names])
        ax.imshow(m, aspect="auto", cmap=cmap, vmin=-0.5, vmax=4.5, interpolation="nearest")
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=6); ax.set_xticks([]); ax.set_title(c["title"], fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110)
