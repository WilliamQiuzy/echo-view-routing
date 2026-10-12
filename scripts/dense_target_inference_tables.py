"""Report of the dense target inference round (configs/experiments/dense_target_inference_v1.yaml; plan Task 3).

- System rows of the three conditions (Sparse, Interpolated, Dense) with direct argmax and each seed's inherited B3
  numbers, on the same full 10 Hz axis; the fixed setting's feasibility is rechecked on calibration per condition.
- Paired bootstraps (native clips by session, streams by stream) and the consistency verdicts over the three seeds.
- Matched boundary offsets within the main tolerance (5 samples): absolute mean, median and 90th percentile.
- Reference units through failure_units.unit_outcomes(hit=5, near=10) with the cause in its fixed order and flags beside
  it; a change matched within near but beyond hit is never counted as a hit.
- Native clips through three stages (argmax majority, mean probability, B3 majority) and the 10 Hz sample table.
- The goal, the fixed readings, server-only case figures and the markdown tables.
Everything returned for the repository holds counts only; recordings enter only as sets that are counted.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import scripts.raw_video_context_tables as rt
import scripts.report_center_supervision as rcs
import scripts.report_fact_asrf as ra
from scripts.ev9v_complementarity import decode
from scripts.prepare_fold_output_correction import assert_no_ids
from scripts.report_fold_output_correction import boundary_changes, segment_changes, summary, write_atomic
from scripts.report_model_development import paired_bootstrap

from echo_routing.evaluate.failure_units import CAUSES, session_of, stream_units, unit_outcomes

HALVES = ("selection", "calibration")
CONDITIONS = ("Sparse", "Interpolated", "Dense")
SEEDS = (0, 1, 2)
CLASSES = ("PLAX", "PSAX", "A4C", "A5C", "SC4C")
ROWS = ("b3", "argmax")
COMPARISONS = {"Dense - Sparse": ("Dense", "Sparse"), "Dense - Interpolated": ("Dense", "Interpolated"), "Interpolated - Sparse": ("Interpolated", "Sparse")}
METRICS = ("multi segment F1@50", "native macro-F1", "native cuts per clip", "multi boundary recall", "multi boundary F1", "same-class cut rate")
POS, NEG, NONE = rt.POS, rt.NEG, rt.NONE
SHORT = 16
READINGS = ("raw classification", "sampling position", "decoding loss", "interval mismatch")
STAGES = ("decoding", "sampling", "recognition")
PAIR_KEYS = {"A4C->A5C": (2, 3), "A5C->A4C": (3, 2), "SC4C->PLAX": (4, 0)}
RULE_ORDER = ("dense_system_gain", "goal_met", "gain_without_constraints", "offset_only", "interpolation_close", "dense_costs", "native_localisation", "end")
FLAG_TESTS = {"all edges within hit": lambda f: f["all_edges_hit"], "an edge only within near": lambda f: f["near_only_edges"] > 0,
              "missed edge": lambda f: f["missed_edges"] > 0, "extra cuts": lambda f: f["extra_cuts"] > 0,
              "own majority right": lambda f: f["own_majority_ok"], "own mean right": lambda f: f["own_mean_ok"],
              "dense majority right": lambda f: f["dense_majority_ok"]}


def report_files() -> list[Path]:
    """This module and every module the report calls, for the run identity."""
    import echo_routing.evaluate.dev_metrics as m1
    import echo_routing.evaluate.failure_units as m2
    import echo_routing.evaluate.ladder as m3
    import echo_routing.evaluate.metrics_segmentation as m4
    import echo_routing.temporal.baselines.b3_js_divergence as m5
    import echo_routing.temporal.baselines.registry as m6
    import echo_routing.temporal.segments as m7
    import scripts.ev9v_complementarity as m8
    import scripts.prepare_fold_output_correction as m9
    import scripts.report_fan224_visual as m10
    import scripts.report_fold_output_correction as m11
    import scripts.report_model_development as m12
    import scripts.round4_failure_inventory as m13
    return [Path(__file__), Path(rt.__file__), Path(rcs.__file__), Path(ra.__file__)] + [Path(m.__file__) for m in (m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13)]


# ----------------------------------------------------------------------------------------------- small helpers
def majority(labels) -> int:
    return int(np.bincount(np.asarray(labels, int), minlength=len(CLASSES)).argmax())


def sessions_of(recs) -> set:
    return {session_of(v) for v in recs}


def system_row(m: dict) -> dict:
    """The source round's stored row (summary plus native precision) and the per-sample accuracies on the 10 Hz axis."""
    return {**summary(m), "native_precision": rcs.summary(m)["native_precision"], "multi_frame_accuracy": float(m["multi_frame_accuracy"]),
            "native_frame_accuracy": float(m["native_frame_accuracy"])}


def offset_stats(offsets, hz: float) -> dict:
    """Signed offsets (predicted - reference, samples) of matched changes: absolute mean, median and 90th percentile in
    samples and seconds, the signed mean and the share of exact matches."""
    o = np.asarray(list(offsets), float)
    if not o.size:
        return {"matched": 0, **dict.fromkeys(("mean_abs", "median_abs", "p90_abs", "mean_abs_s", "median_abs_s", "p90_abs_s", "mean_signed", "exact_share"))}
    a = np.abs(o); st = {"mean_abs": float(a.mean()), "median_abs": float(np.median(a)), "p90_abs": float(np.percentile(a, 90))}
    return {"matched": int(o.size), **st, **{f"{k}_s": v / float(hz) for k, v in st.items()}, "mean_signed": float(o.mean()), "exact_share": float((o == 0).mean())}


# ----------------------------------------------------------------------------------------------- reference units
def unit_flags(row: dict, unit, own_raw: np.ndarray, own_prob: np.ndarray, dense_raw: np.ndarray, hit: int, near: int) -> dict:
    """Flags of one unit beside its unit_outcomes cause: edges matched within hit, only within near, or not at all;
    extra changes inside; whether the condition's own argmax majority, its mean probability and the Dense argmax majority
    over the unit name the reference class; and the four non-exclusive readings of a wrong unit."""
    d = list(row["edge_dist"]); hit_e = sum(x is not None and x <= hit for x in d)
    near_only = sum(x is not None and hit < x <= near for x in d); missed = sum(x is None for x in d)
    if hit_e != int(row["edges_within_hit"]) or any(x is not None and x > near for x in d):
        raise ValueError("unit_outcomes was not run with the same hit and near")
    sl, lab = slice(unit.start, unit.end), int(unit.label)
    own_maj = majority(own_raw[sl]) == lab; own_mean = int(np.asarray(own_prob[sl], float).mean(0).argmax()) == lab
    dense_maj = majority(dense_raw[sl]) == lab; wrong = not row["matched"]
    readings = {"raw classification": wrong and not dense_maj, "sampling position": wrong and not own_maj and dense_maj,
                "decoding loss": wrong and own_maj, "interval mismatch": wrong and (row["cause"] == "boundary_shift" or near_only > 0)}
    recs = sorted({s[0] for s in unit.sources})
    return {"matched": bool(row["matched"]), "cause": row["cause"], "edges": len(d), "hit_edges": hit_e, "near_only_edges": near_only,
            "missed_edges": missed, "all_edges_hit": hit_e == len(d), "extra_cuts": int(row["extra_cuts"]), "own_majority_ok": bool(own_maj),
            "own_mean_ok": bool(own_mean), "dense_majority_ok": bool(dense_maj), "readings": {k: bool(v) for k, v in readings.items()},
            "recordings": recs, "sessions": sorted(sessions_of(recs))}


def _units_count(items: list[dict]) -> dict:
    recs = set().union(*[set(x["recordings"]) for x in items]) if items else set()
    return {"units": len(items), "recordings": len(recs), "sessions": len(sessions_of(recs))}


def unit_table(flags: list[dict], spurious: list[dict]) -> dict:
    """Units, matched and wrong; wrong units by cause, by flag and by reading with distinct recordings and sessions; the
    edges of matched units; unmatched predicted segments."""
    wrong = [f for f in flags if not f["matched"]]; matched = [f for f in flags if f["matched"]]
    return {"units": len(flags), "matched": len(matched), "wrong": len(wrong),
            "by_cause": {c: _units_count([f for f in wrong if f["cause"] == c]) for c in CAUSES},
            "flags_wrong": {k: _units_count([f for f in wrong if t(f)]) for k, t in FLAG_TESTS.items()},
            "readings": {r: _units_count([f for f in wrong if f["readings"][r]]) for r in READINGS},
            "matched_edges": {"units": len(matched), "all edges within hit": int(sum(f["all_edges_hit"] for f in matched)),
                              "an edge only within near": int(sum(f["near_only_edges"] > 0 for f in matched))},
            "spurious": {"segments": len(spurious), "inside_host": int(sum(bool(s["inside"]) for s in spurious))}}


# ----------------------------------------------------------------------------------------------- native clips
def _clips_count(items: list[dict]) -> dict:
    recs = {c["video_id"] for c in items}
    return {"clips": len(items), "recordings": len(recs), "sessions": len(sessions_of(recs))}


def _pairs(items: list[dict], key: str = "decode") -> dict:
    out: dict = {}
    for c in items:
        k = f"{CLASSES[c['label']]}->{CLASSES[c[key]]}"; out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def stage_of(c: dict) -> str:
    """The stage of a B3-wrong clip by its own raw majority: decoding, sampling (Dense right) or recognition."""
    return "decoding" if c["raw"] == c["label"] else "sampling" if c["dense_raw"] == c["label"] else "recognition"


def native_stages(clips: list[dict]) -> dict:
    """clips: video_id, label, raw (own argmax majority), mean (own mean-probability label), decode (B3 majority, the
    scored label) and dense_raw (Dense argmax majority)."""
    wrong = [c for c in clips if c["decode"] != c["label"]]
    rr_dw = [c for c in wrong if c["raw"] == c["label"]]; both = [c for c in wrong if c["raw"] != c["label"]]
    rw_dr = [c for c in clips if c["decode"] == c["label"] and c["raw"] != c["label"]]
    readings = {"raw classification": [c for c in wrong if c["dense_raw"] != c["label"]],
                "sampling position": [c for c in wrong if c["raw"] != c["label"] and c["dense_raw"] == c["label"]], "decoding loss": rr_dw}
    return {"clips": len(clips), "decode wrong": {**_clips_count(wrong), "pairs": _pairs(wrong)},
            "raw right, decode wrong": {**_clips_count(rr_dw), "pairs": _pairs(rr_dw)}, "both wrong": {**_clips_count(both), "pairs": _pairs(both)},
            "raw wrong, decode right": {**_clips_count(rw_dr), "raw pairs": _pairs(rw_dr, "raw")},
            "mean label right among decode wrong": int(sum(c["mean"] == c["label"] for c in wrong)),
            "readings": {k: _clips_count(v) for k, v in readings.items()},
            "stages": {st: _clips_count([c for c in wrong if stage_of(c) == st]) for st in STAGES}}


def localisation(t: dict) -> str:
    """The stage holding more than half of the B3-wrong recordings, 'mixed' otherwise, 'no errors' without any."""
    n = t["decode wrong"]["recordings"]
    if not n:
        return "no errors"
    over = [st for st in STAGES if t["stages"][st]["recordings"] > n / 2]
    return over[0] if len(over) == 1 else "mixed"


def union_ids(lists) -> dict:
    recs = {v for x in lists for v in x}
    return {"recordings": len(recs), "sessions": len(sessions_of(recs))}


def native_union(per_seed: list[list[dict]]) -> dict:
    """The B3-wrong native clips of all seeds as distinct recordings and sessions, by class pair and stage."""
    wrong = [c for clips in per_seed for c in clips if c["decode"] != c["label"]]; pairs: dict = {}
    for c in wrong:
        pairs.setdefault(f"{CLASSES[c['label']]}->{CLASSES[c['decode']]}", set()).add(c["video_id"])
    t = {"decode wrong": {**union_ids([[c["video_id"] for c in wrong]]), "pairs": {k: union_ids([v]) for k, v in sorted(pairs.items())}},
         "stages": {st: union_ids([[c["video_id"] for c in wrong if stage_of(c) == st]]) for st in STAGES}}
    return {**t, "localisation": localisation(t)}


def native_pair_changes(a: list[dict], b: list[dict]) -> dict:
    """Native clips a fixes (b wrong, a right) and newly gets wrong, per class pair and over all clips, with recordings
    and sessions."""
    B = {c["video_id"]: c for c in b}

    def cnt(name: str, ids: list[str]) -> dict:
        return {name: len(ids), f"{name}_recordings": len(set(ids)), f"{name}_sessions": len(sessions_of(ids))}
    out = {}
    for name, (y, x) in PAIR_KEYS.items():
        fixed = [c["video_id"] for c in a if c["label"] == y and c["decode"] == y and B[c["video_id"]]["decode"] == x]
        new = [c["video_id"] for c in a if c["label"] == y and c["decode"] == x and B[c["video_id"]]["decode"] == y]
        out[name] = {**cnt("fixed", fixed), **cnt("new", new)}
    fixed = [c["video_id"] for c in a if c["decode"] == c["label"] and B[c["video_id"]]["decode"] != c["label"]]
    new = [c["video_id"] for c in a if c["decode"] != c["label"] and B[c["video_id"]]["decode"] == c["label"]]
    out["all clips"] = {**cnt("fixed", fixed), **cnt("new", new)}
    return out


# ----------------------------------------------------------------------------------------------- 10 Hz axis
def sampling_pair(y, a, b, recs) -> dict:
    """Per sample: a right and b wrong, the reverse, and both wrong, with distinct recordings and sessions."""
    y, a, b, recs = (np.asarray(x) for x in (y, a, b, recs)); ok_a, ok_b = a == y, b == y

    def cnt(sel: np.ndarray) -> dict:
        v = set(recs[sel].tolist())
        return {"samples": int(sel.sum()), "recordings": len(v), "sessions": len(sessions_of(v))}
    return {"samples": int(y.size), "a right": int(ok_a.sum()), "b right": int(ok_b.sum()), "a right, b wrong": cnt(ok_a & ~ok_b),
            "b right, a wrong": cnt(ok_b & ~ok_a), "both wrong": cnt(~ok_a & ~ok_b)}


# ----------------------------------------------------------------------------------------------- goal and readings
def goal_check(rows: list[dict], g: dict) -> dict:
    """rows: per seed {half: B3 row}. The fixed numbers are feasible for a seed when its calibration row meets the
    source round's rule (native macro-F1 >= the calibration floor and boundary recall >= 0.95)."""
    feasible = [bool((r["calibration"]["boundary_recall_multi"] or 0.0) >= float(g["boundary_recall_min"])
                     and r["calibration"]["native_macro_f1"] >= float(g["native_macro_f1_min"]["calibration"])) for r in rows]
    return {"feasible": feasible, **rt.goal(rows, feasible, g)}


def decide(ver: dict, offsets: dict, goals: dict, local: dict, rules: dict | None = None) -> dict:
    """The fixed readings. ver[f'b3|{comparison}'][half][metric] are verdicts; offsets[f'{condition}|s{seed}|{half}'] the
    B3 mean absolute matched offsets; goals[condition] from goal_check; local[f'{condition}|{half}'] the native stage."""
    ds, di = ver["b3|Dense - Sparse"], ver["b3|Dense - Interpolated"]; f1, nat = "multi segment F1@50", "native macro-F1"
    f1_both = all(ds[h][f1] == POS for h in HALVES); dense_ok = not goals["Dense"]["constraint_failures"]
    lower = all(offsets.get(f"Dense|s{s}|{h}") is not None and offsets.get(f"Sparse|s{s}|{h}") is not None
                and offsets[f"Dense|s{s}|{h}"] < offsets[f"Sparse|s{s}|{h}"] for s in SEEDS for h in HALVES)
    rows = {"dense_system_gain": {"applies": f1_both and dense_ok},
            "goal_met": {"applies": f1_both and dense_ok and bool(goals["Dense"]["met"])},
            "gain_without_constraints": {"applies": f1_both and not dense_ok},
            "offset_only": {"applies": lower and not any(ds[h][f1] == POS or ds[h][nat] == POS for h in HALVES)},
            "interpolation_close": {"halves": [h for h in HALVES if di[h][f1] != POS]},
            "dense_costs": {"halves": [h for h in HALVES if ds[h]["native cuts per clip"] == POS or ds[h]["same-class cut rate"] == POS or ds[h][nat] == NEG]},
            "native_localisation": {"by_condition_and_half": dict(local), "applies": any(v != "no errors" for v in local.values())},
            "end": {"applies": True}}
    for k in ("interpolation_close", "dense_costs"):
        rows[k]["applies"] = bool(rows[k]["halves"])
    for k, r in rows.items():
        r["applies"] = bool(r["applies"])
        if rules and k in rules:
            r["reading"] = rules[k]
    return {"rows": rows, "applied": [k for k in RULE_ORDER if rows[k]["applies"]]}


# ----------------------------------------------------------------------------------------------- the report
def systems_of(cfg: dict, b3: dict, inputs, probs: dict, seeds: list[int]) -> dict:
    """(condition, seed) -> row -> half -> {row, tables, preds}."""
    out = {}
    for c in CONDITIONS:
        for s in seeds:
            out[(c, s)] = {r: {} for r in ROWS}
            for h in HALVES:
                P = probs[s][h][c]
                for r, preds in (("argmax", {k: v.argmax(1) for k, v in P.items()}), ("b3", decode(b3[s], P))):
                    m, t = rcs.evaluate(cfg, preds, inputs.halves[h]); out[(c, s)][r][h] = {"row": system_row(m), "tables": t, "preds": preds}
    return out


def native_clips(inputs, probs: dict, sysd: dict, seeds: list[int]) -> dict:
    """(condition, seed, half) -> per native clip: label, raw, mean, decode and dense_raw."""
    out = {}
    for (c, s), v in sysd.items():
        for h in HALVES:
            out[(c, s, h)] = [{"video_id": n.seq_id, "label": int(n.labels[0]), "raw": majority(probs[s][h][c][n.seq_id].argmax(1)),
                               "mean": int(probs[s][h][c][n.seq_id].astype(float).mean(0).argmax()), "decode": majority(v["b3"][h]["preds"][n.seq_id]),
                               "dense_raw": majority(probs[s][h]["Dense"][n.seq_id].argmax(1))} for n in inputs.halves[h].natives]
            conf = np.zeros((5, 5), int)
            for x in out[(c, s, h)]:
                conf[x["label"], x["decode"]] += 1
            if conf.tolist() != v["b3"][h]["row"]["native_confusion"]:
                raise SystemExit(f"{c} s{s} {h}: the native stages disagree with the B3 row")
    return out


def units_of(inputs, probs: dict, sysd: dict, hit: int, near: int) -> tuple[dict, dict]:
    """Unit tables per condition, seed, half and stream kind (B3), and the multi-stream unit rows kept for the cases."""
    tables, keep = {}, {}
    for (c, s), v in sysd.items():
        for h in HALVES:
            ev = inputs.halves[h]
            for kind, coll in (("multi", ev.multi), ("pairs", ev.pairs)):
                flags, spur = [], []
                for st in coll:
                    sid = st.recipe.recipe_id; us = stream_units(st.recipe); P = probs[s][h][c][sid]
                    rows, sp = unit_outcomes(v["b3"][h]["preds"][sid], us, 0.5, hit, near)
                    fl = [unit_flags(r, us[r["index"]], P.argmax(1), P, probs[s][h]["Dense"][sid].argmax(1), hit, near) for r in rows]
                    flags += fl; spur += sp
                    if kind == "multi":
                        keep[(c, s, h, sid)] = (us, rows)
                tables[f"{c}|s{s}|{h}|{kind}"] = unit_table(flags, spur)
    return tables, keep


def sampling_of(seqs: dict, probs: dict, seeds: list[int]) -> dict:
    out = {}
    for s in seeds:
        for h in HALVES:
            for kind in ("native", "pairs", "multi"):
                qs = [q for q in seqs[h] if q["kind"] == kind]
                y = np.concatenate([np.asarray(q["labels"], int) for q in qs]); recs = np.array([r[0] for q in qs for r in q["refs"]])
                pred = {c: np.concatenate([probs[s][h][c][q["id"]].argmax(1) for q in qs]) for c in CONDITIONS}
                for name, (a, b) in COMPARISONS.items():
                    out[f"s{s}|{h}|{kind}|{name}"] = sampling_pair(y, pred[a], pred[b], recs)
    return out


def case_figures(out_dir: Path, inputs, probs: dict, sysd: dict, clips: dict, keep: dict, floors: dict, seeds: list[int]) -> dict:
    """Server-only probability curves: native clips a condition's B3 gets wrong in a seed and half below the native
    floor, and on calibration every wrong native clip and unmatched multi unit of Sparse or Dense. Returns counts only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    floor_cases = [(c, s, h, x) for (c, s, h), L in clips.items() if sysd[(c, s)]["b3"][h]["row"]["native_macro_f1"] < float(floors[h])
                   for x in L if x["decode"] != x["label"]]
    cal_native = [(c, s, "calibration", x) for (c, s, h), L in clips.items() if h == "calibration" and c in ("Sparse", "Dense")
                  for x in L if x["decode"] != x["label"]]
    cal_units = [(c, s, sid, us[r["index"]]) for (c, s, h, sid), (us, rows) in keep.items() if h == "calibration" and c in ("Sparse", "Dense")
                 for r in rows if not r["matched"]]
    colours = {"Sparse": "#4c72b0", "Interpolated": "#dd8452", "Dense": "#55a868"}

    def draw(path: Path, s: int, h: str, sid: str, lo: int, hi: int, classes: list[int], spans: list[tuple[int, int]], title: str) -> None:
        if path.exists():
            return
        fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}); x = np.arange(lo, hi)
        for c in CONDITIONS:
            P = probs[s][h][c][sid]
            for k, ls in zip(classes, ("-", "--")):
                ax[0].plot(x, P[lo:hi, k], color=colours[c], ls=ls, lw=1.2, label=f"{c} p({CLASSES[k]})")
            ax[1].step(x, sysd[(c, s)]["b3"][h]["preds"][sid][lo:hi] + 0.1 * CONDITIONS.index(c), where="mid", color=colours[c], lw=1.2)
        for a, b in spans:
            for axx in ax:
                axx.axvspan(a, b, color="0.85", zorder=0)
        ax[0].legend(fontsize=7, ncol=3); ax[0].set_ylim(-0.02, 1.02); ax[1].set_yticks(range(5), CLASSES); ax[1].set_xlabel("10 Hz sample")
        ax[0].set_title(title, fontsize=9); fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=110); plt.close(fig)
    for c, s, h, x in floor_cases + cal_native:
        n = len(probs[s][h][c][x["video_id"]])
        draw(out_dir / h / f"s{s}" / f"native_{x['video_id']}.png", s, h, x["video_id"], 0, n, [x["label"], x["decode"]], [],
             f"{x['video_id']} seed {s} {h}: label {CLASSES[x['label']]}; B3 majority per condition (bottom)")
    for c, s, sid, u in cal_units:
        n = len(probs[s]["calibration"][c][sid]); lo, hi = max(0, u.start - 30), min(n, u.end + 30)
        other = u.left if u.left is not None else u.right if u.right is not None else u.label
        draw(out_dir / "calibration" / f"s{s}" / f"unit_{sid}_{u.index}.png", s, "calibration", sid, lo, hi, [u.label, other], [(u.start, u.end)],
             f"{sid} unit {u.index} seed {s}: reference {CLASSES[u.label]} (shaded); B3 labels per condition (bottom)")
    floor_recs = {x["video_id"] for _, _, _, x in floor_cases}; cal_recs = {x["video_id"] for _, _, _, x in cal_native} | {r[0] for *_, u in cal_units for r in u.sources}
    return {"native floor": {"cases": len(floor_cases), **union_ids([floor_recs])},
            "calibration": {"native clip cases": len(cal_native), "unit cases": len(cal_units), **union_ids([cal_recs])},
            "note": "figures stay on the server; each case is one condition, seed and source"}


def run_report(cfg: dict, dtc: dict, b3: dict, inputs, seqs: dict, probs: dict, counts: dict, cost: dict, out_root: Path, results_dir: Path,
               ids: set, t0: float, stored: dict) -> dict:
    """Every result of the round from the per-sample probabilities of the three conditions; Sparse must first reproduce
    every stored row. Writes summary.json and report_tables.md through the id guards and the case figures on the server."""
    hz = float(cfg["protocol"]["hz"]); tol = float(cfg["selection"]["boundary_tolerance_s"]) * hz
    hit, near = int(dtc["evaluation"]["units"]["hit"]), int(dtc["evaluation"]["units"]["near"]); seeds = sorted(probs)
    if hit != int(round(tol)) or near <= hit:
        raise SystemExit("the unit hit must equal the main boundary tolerance and near must exceed it")
    sysd = systems_of(cfg, b3, inputs, probs, seeds)
    for s in seeds:
        for r in ROWS:
            for h in HALVES:
                got, want = sysd[("Sparse", s)][r][h]["row"], stored[f"ordered_s{s}"][r][h]
                if any(got.get(k) != v for k, v in want.items()):
                    raise SystemExit(f"Sparse s{s} {r} {h}: the stored row is not reproduced")
    print("Sparse rows reproduce the stored rows", flush=True)
    systems = {f"{c}|s{s}": {r: {h: sysd[(c, s)][r][h]["row"] for h in HALVES} for r in ROWS} for c, s in sysd}
    goals = {c: goal_check([systems[f"{c}|s{s}"]["b3"] for s in seeds], dtc["goal"]) for c in CONDITIONS}
    for c in CONDITIONS:
        for i, s in enumerate(seeds):
            systems[f"{c}|s{s}"]["feasible"] = goals[c]["feasible"][i]
    comp = {}
    for s in seeds:
        for r in ROWS:
            for name, (a, b) in COMPARISONS.items():
                for h in HALVES:
                    A, B, ev = sysd[(a, s)][r][h], sysd[(b, s)][r][h], inputs.halves[h]
                    comp[f"s{s}|{r}|{name}|{h}"] = {"bootstrap": paired_bootstrap(A["tables"], B["tables"]), "segment_changes": segment_changes(ev, A["preds"], B["preds"]),
                                                     "boundary_changes": boundary_changes(ev, A["preds"], B["preds"], tol)}
        print(f"seed {s}: comparisons done ({time.time() - t0:.0f} s)", flush=True)
    ver = {f"{r}|{n}": {h: {m: rt.consistency([comp[f"s{s}|{r}|{n}|{h}"]["bootstrap"][m] for s in seeds]) for m in METRICS} for h in HALVES} for r in ROWS for n in COMPARISONS}
    offsets, mech = {}, {}
    for (c, s), v in sysd.items():
        for r in ROWS:
            for h in HALVES:
                ev, offs = inputs.halves[h], []
                for st in ev.multi:
                    pred = np.flatnonzero(np.diff(np.asarray(v[r][h]["preds"][st.recipe.recipe_id])) != 0) + 1
                    offs += rt.match_offsets(np.asarray(st.semantic_boundaries, int), pred, tol)
                offsets[f"{c}|{r}|s{s}|{h}"] = offset_stats(offs, hz); mech[f"{c}|{r}|s{s}|{h}"] = ra.mechanism(ev, v[r][h]["preds"], SHORT)
    units, keep = units_of(inputs, probs, sysd, hit, near); clips = native_clips(inputs, probs, sysd, seeds)
    natives = {f"{c}|s{s}|{h}": native_stages(L) for (c, s, h), L in clips.items()}
    natives_union = {f"{c}|{h}": native_union([clips[(c, s, h)] for s in seeds]) for c in CONDITIONS for h in HALVES}
    nchanges = {f"s{s}|{n}|{h}": native_pair_changes(clips[(a, s, h)], clips[(b, s, h)]) for s in seeds for n, (a, b) in COMPARISONS.items() for h in HALVES}
    sampling = sampling_of(seqs, probs, seeds)
    local = {f"{c}|{h}": natives_union[f"{c}|{h}"]["localisation"] for c in CONDITIONS for h in HALVES}
    dec = decide(ver, {f"{c}|s{s}|{h}": offsets[f"{c}|b3|s{s}|{h}"]["mean_abs"] for c in CONDITIONS for s in seeds for h in HALVES}, goals, local, dtc["decision_rules"])
    cases = case_figures(out_root / "cases", inputs, probs, sysd, clips, keep, dtc["goal"]["native_macro_f1_min"], seeds)
    res = {"counts": counts, "systems": systems, "sparse_reproduction": "every stored argmax and B3 row of the three Ordered models is reproduced field by field",
           "comparisons": comp, "verdicts": ver, "offsets": offsets, "mechanism": mech, "units": units, "natives": natives, "natives_union": natives_union,
           "native_changes": nchanges, "sampling": sampling, "goal": goals, "decision": dec, "cases": cases,
           "cost": {**cost, "report_seconds": time.time() - t0}}
    write(res, results_dir, ids); print(json.dumps(dec["applied"]), flush=True)
    return res


def write(res: dict, out_dir: Path, ids: set) -> None:
    flat = json.loads(json.dumps(res, default=str)); assert_no_ids(flat, ids, "summary"); md = tables_md(flat)
    if any(s in md for s in ids):
        raise SystemExit("an id appears in the report tables")
    out_dir.mkdir(parents=True, exist_ok=True); write_atomic(out_dir / "summary.json", json.dumps(flat, indent=1)); write_atomic(out_dir / "report_tables.md", md)


# ----------------------------------------------------------------------------------------------- markdown
def f3(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def f4(x) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def both(get, fmt=f3) -> str:
    return " / ".join(fmt(get(h)) for h in HALVES)


def table(head: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows] + [""]


def cnt(d: dict, k: str = "units") -> str:
    return f"{d[k]} ({d['recordings']}, {d['sessions']})"


def _systems(res: dict, row: str) -> list[str]:
    cols = [("Segment F1@50", "segment_f1_50_multi", f3), ("Boundary recall", "boundary_recall_multi", f3), ("Boundary precision", "boundary_precision_multi", f3),
            ("Native macro-F1", "native_macro_f1", f4), ("Label switches per native clip", "native_internal_cuts_per_clip", f3),
            ("Same-view join cuts", "same_class_cut_rate_multi", f3), ("10 Hz accuracy, multi", "multi_frame_accuracy", f4),
            ("A4C read as A5C (clips)", "A4C read as A5C", str), ("A5C read as A4C (clips)", "A5C read as A4C", str)]
    rows = []
    for c in CONDITIONS:
        for s in SEEDS:
            x = res["systems"][f"{c}|s{s}"][row]
            rows.append([c, s, *[both(lambda h, k=k: x[h][k], fmt) for _, k, fmt in cols]] + ([str(res["systems"][f"{c}|s{s}"]["feasible"])] if row == "b3" else []))
    return table(["Condition", "Seed", *[n for n, _, _ in cols]] + (["Fixed numbers feasible"] if row == "b3" else []), rows)


def _comparisons(res: dict) -> list[str]:
    out = []
    for r, title in (("b3", "With the fixed B3 numbers"), ("argmax", "Direct argmax (secondary)")):
        rows = []
        for n in COMPARISONS:
            for h in HALVES:
                v = res["verdicts"][f"{r}|{n}"][h]
                rows.append([n, h, "; ".join(rt.ci(res["comparisons"][f"s{s}|{r}|{n}|{h}"]["bootstrap"]["multi segment F1@50"]) for s in SEEDS),
                             v["multi segment F1@50"], *[v[m] for m in METRICS[1:]]])
        out += [f"### {title}", ""] + table(["Comparison", "Half", "Segment F1@50 (seeds 0, 1, 2)", "Verdict", *[f"{m} verdict" for m in METRICS[1:]]], rows)
    rows = [[n, h, s, *[f"{c[k]['fixed']} ({c[k]['fixed_sessions']}) / {c[k]['new']} ({c[k]['new_sessions']})" for k in (*PAIR_KEYS, "all clips")]]
            for n in COMPARISONS for h in HALVES for s in SEEDS for c in [res["native_changes"][f"s{s}|{n}|{h}"]]]
    out += ["### Native clips fixed / newly wrong with B3 (sessions)", ""] + table(["Comparison", "Half", "Seed", *PAIR_KEYS, "all clips"], rows)
    rows = [[n, h, s, cnt(b["segment_changes"]["matched by a only"], "segments"), cnt(b["segment_changes"]["matched by b only"], "segments"),
             cnt(b["boundary_changes"]["found by a only"], "changes"), cnt(b["boundary_changes"]["found by b only"], "changes")]
            for n in COMPARISONS for h in HALVES for s in SEEDS for b in [res["comparisons"][f"s{s}|b3|{n}|{h}"]]]
    out += ["### Multi-stream segments and changes with B3: a only / b only (recordings, sessions)", ""] + table(
        ["Comparison", "Half", "Seed", "Segments matched by a only", "Segments matched by b only", "Changes found by a only", "Changes found by b only"], rows)
    return out


def _offsets(res: dict) -> list[str]:
    rows = []
    for c in CONDITIONS:
        for s in SEEDS:
            o = {h: res["offsets"][f"{c}|b3|s{s}|{h}"] for h in HALVES}
            rows.append([c, s, both(lambda h: o[h]["matched"], str), both(lambda h: o[h]["mean_abs"], f4), both(lambda h: o[h]["median_abs"], f3),
                         both(lambda h: o[h]["p90_abs"], f3), both(lambda h: o[h]["mean_abs_s"], f4), both(lambda h: o[h]["p90_abs_s"], f3),
                         both(lambda h: o[h]["mean_signed"], f4), both(lambda h: o[h]["exact_share"], f3)])
    return table(["Condition (B3)", "Seed", "Matched changes", "Mean absolute offset (samples)", "Median", "90th percentile", "Mean absolute offset (s)",
                  "90th percentile (s)", "Signed mean", "Exact share"], rows)


def _mechanism(res: dict) -> list[str]:
    keys = ("short segments", "short A4C or A5C segments", "X-Y-X middles", "A4C/A5C X-Y-X middles")
    rows = [[c, s, *[both(lambda h, k=k: res["mechanism"][f"{c}|b3|s{s}|{h}"][k]["matched"], str) for k in keys]] for c in CONDITIONS for s in SEEDS]
    tot = [both(lambda h, k=k: res["mechanism"][f"Sparse|b3|s0|{h}"][k]["segments"], str) for k in keys]
    return table(["Condition (B3)", "Seed", *[f"{k} kept ({t})" for k, t in zip(keys, tot)]], rows)


def _units(res: dict, kind: str) -> list[str]:
    rows, rd, fl = [], [], []
    for c in CONDITIONS:
        for s in SEEDS:
            u = {h: res["units"][f"{c}|s{s}|{h}|{kind}"] for h in HALVES}
            rows.append([c, s, both(lambda h: f"{u[h]['wrong']} of {u[h]['units']}", str), *[both(lambda h, k=k: cnt(u[h]["by_cause"][k]), str) for k in CAUSES],
                         both(lambda h: f"{u[h]['spurious']['segments']} ({u[h]['spurious']['inside_host']})", str),
                         both(lambda h: f"{u[h]['matched_edges']['an edge only within near']}", str)])
            rd.append([c, s, *[both(lambda h, k=k: cnt(u[h]["readings"][k]), str) for k in READINGS]])
            fl.append([c, s, *[both(lambda h, k=k: str(u[h]["flags_wrong"][k]["units"]), str) for k in FLAG_TESTS]])
    return (table(["Condition (B3)", "Seed", "Wrong units", *[f"{k} (rec, ses)" for k in CAUSES], "Unmatched predicted segments (inside a unit)",
                   "Matched units with an edge only within near"], rows)
            + ["Readings of wrong units, not exclusive: units (recordings, sessions).", ""] + table(["Condition (B3)", "Seed", *READINGS], rd)
            + ["Flags of wrong units, not exclusive: units.", ""] + table(["Condition (B3)", "Seed", *FLAG_TESTS], fl))


def _natives(res: dict) -> list[str]:
    rows = []
    for c in CONDITIONS:
        for s in SEEDS:
            t = {h: res["natives"][f"{c}|s{s}|{h}"] for h in HALVES}
            rows.append([c, s, both(lambda h: cnt(t[h]["decode wrong"], "clips"), str), both(lambda h: t[h]["decode wrong"]["pairs"].get("A4C->A5C", 0), str),
                         both(lambda h: t[h]["decode wrong"]["pairs"].get("A5C->A4C", 0), str), both(lambda h: t[h]["raw right, decode wrong"]["clips"], str),
                         both(lambda h: t[h]["both wrong"]["clips"], str), both(lambda h: t[h]["raw wrong, decode right"]["clips"], str),
                         both(lambda h: t[h]["mean label right among decode wrong"], str), *[both(lambda h, k=k: t[h]["stages"][k]["clips"], str) for k in STAGES]])
    out = table(["Condition", "Seed", "B3-wrong clips (rec, ses)", "A4C->A5C", "A5C->A4C", "Raw right, decode wrong", "Both wrong", "Raw wrong, decode right",
                 "Mean label right among wrong", *[f"Stage: {k}" for k in STAGES]], rows)
    rows = [[c, h, f"{u['decode wrong']['recordings']} ({u['decode wrong']['sessions']})",
             "; ".join(f"{k} {v['recordings']} ({v['sessions']})" for k, v in u["decode wrong"]["pairs"].items()) or "none",
             *[f"{u['stages'][k]['recordings']}" for k in STAGES], u["localisation"]]
            for c in CONDITIONS for h in HALVES for u in [res["natives_union"][f"{c}|{h}"]]]
    return out + ["Union over the three seeds: distinct recordings (sessions).", ""] + table(
        ["Condition", "Half", "B3-wrong recordings", "By class pair", *[f"Stage: {k}" for k in STAGES], "Localisation"], rows)


def _sampling(res: dict) -> list[str]:
    rows = [[n, s, k, both(lambda h: res["sampling"][f"s{s}|{h}|{k}|{n}"]["samples"], str),
             both(lambda h: cnt(res["sampling"][f"s{s}|{h}|{k}|{n}"]["a right, b wrong"], "samples"), str),
             both(lambda h: cnt(res["sampling"][f"s{s}|{h}|{k}|{n}"]["b right, a wrong"], "samples"), str),
             both(lambda h: cnt(res["sampling"][f"s{s}|{h}|{k}|{n}"]["both wrong"], "samples"), str)]
            for n in COMPARISONS for s in SEEDS for k in ("native", "pairs", "multi")]
    return table(["a - b", "Seed", "Sequences", "Samples", "a right, b wrong (rec, ses)", "b right, a wrong (rec, ses)", "Both wrong (rec, ses)"], rows)


def _goal(res: dict) -> list[str]:
    rows = [[c, both(lambda h: g["mean_segment_f1"][h], f4), both(lambda h: g["margin"][h], lambda x: f"{x:+.4f}"), f"{g['calibration_sensitivity_margin']:+.4f}",
             " ".join(str(x) for x in g["feasible"]), len(g["constraint_failures"]), g["met"]] for c in CONDITIONS for g in [res["goal"][c]]]
    return table(["Condition", "Three-seed mean (sel / cal)", "Margin to the minimum", "Calibration sensitivity margin", "Fixed numbers feasible (seeds)",
                  "Constraint failures", "Goal met"], rows)


def _counts(res: dict) -> list[str]:
    keys = ("old_windows", "new_windows", "short_sequences", "windows_with_marker_off_slot_8", "windows_crossing_a_join_or_gamma")
    rows = [[h, ", ".join(str(c["sequences"][k]) for k in ("native", "pairs", "multi")), c["samples"], *[c.get(k, "n/a") for k in keys]]
            for h in HALVES for c in [res["counts"][h]]]
    return table(["Half", "Sequences (native, pairs, multi)", "Samples", "Old windows", "New windows", "Short sequences",
                  "Windows with the marker off slot 8", "Windows crossing a join or gamma"], rows) + ["```", json.dumps(res["cost"], indent=1, default=str), "```", ""]


def tables_md(res: dict) -> str:
    out = ["# Dense target inference round: tables", "",
           "Recompute: `python scripts/dense_target_inference.py --config configs/experiments/dense_target_inference_v1.yaml --run` (after `--check`).",
           "Each cell gives the selection half, then the calibration half. Every condition uses each seed's inherited B3 numbers.", ""]
    sections = [("## Counts and cost", ("counts", "cost"), _counts),
                ("## Systems with the fixed B3 numbers", ("systems",), lambda r: _systems(r, "b3")),
                ("## Direct argmax", ("systems",), lambda r: _systems(r, "argmax")),
                ("## Comparisons", ("comparisons", "verdicts", "native_changes"), _comparisons),
                ("## Matched boundary offsets (within 5 samples)", ("offsets",), _offsets),
                ("## Short segments and X-Y-X middles (B3)", ("mechanism",), _mechanism),
                ("## Reference units of multi streams (B3; hit 5, near 10)", ("units",), lambda r: _units(r, "multi")),
                ("## Reference units of pair streams (B3)", ("units",), lambda r: _units(r, "pairs")),
                ("## Native clips", ("natives", "natives_union"), _natives),
                ("## The 10 Hz axis", ("sampling",), _sampling),
                ("## Goal", ("goal",), _goal),
                ("## Cases (figures on the server)", ("cases",), lambda r: ["```", json.dumps(r["cases"], indent=1), "```", ""]),
                ("## Decision", ("decision",), lambda r: [f"Applied: {', '.join(r['decision']['applied'])}", "", "```",
                                                          json.dumps(r["decision"]["rows"], indent=1, default=str), "```", ""])]
    for head, keys, fn in sections:
        out += [head, ""] + (fn(res) if all(k in res for k in keys) else ["pending", ""])
    return "\n".join(out)
