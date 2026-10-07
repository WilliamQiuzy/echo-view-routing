#!/usr/bin/env python
"""Frozen transfer check of the short-segment discriminators (configs/experiments/short_segment_transfer_v1.yaml;
docs/superpowers/plans/2026-10-07-short-segment-transfer.md). A development-set transfer check, not an independent
validation; nothing enters the full system and the test split stays sealed.

  --rebuild  Q0 and Q1 per seed, refitted with the stored C on the original train part of short_segment_probe_v1 (no
             search). The original 0.95 / 0.90 thresholds, held-out AUC and rates must reproduce; the fitted pipelines
             are pickled on the server with their sha256.
  --apply    Both validation halves: the same generator (J0, kappa 8 on the B_s{k} store), candidate pattern, categories
             and features; the frozen pipelines and the original thresholds (never re-chosen). Tables led by A4C/A5C,
             the candidates outside the binary task with their reasons, source and join diagnostics, the Q1 - Q0
             bootstrap and the fixed decision.

  python scripts/short_segment_transfer.py --config configs/experiments/short_segment_transfer_v1.yaml --rebuild
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg  # noqa: E402
from scripts.ev9v_complementarity import md_table  # noqa: E402
import scripts.short_segment_probe as sp  # noqa: E402

import numpy as np  # noqa: E402

from echo_routing.config import load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.temporal.baselines.b3_js_divergence import boundary_scores  # noqa: E402
from echo_routing.temporal.dev_run import HALVES, Inputs  # noqa: E402
from echo_routing.temporal.joint_decoding import joint_viterbi  # noqa: E402

NS = "short_segment_transfer_v1"
REASONS = ("sides not X", "a third class under the middle", "part of a Y unit (IoU < 0.5)", "a reference change under the middle")
SUBSETS = ("native clips", "pair streams", "multi streams", "no join", "same-view join", "view-change join", "middle 10-40 samples")


# ----------------------------------------------------------------------------------------------- pure helpers
def join_type(ref: np.ndarray, joins: np.ndarray, start: int, end: int, tol: int) -> str:
    """'none' without a join within tol of [start, end); 'same-view' when the reference view is the same on both sides
    of every such join; 'view-change' otherwise."""
    j = [int(k) for k in np.asarray(joins, int) if start - tol <= k <= end + tol and 0 < k < len(ref)]
    if not j:
        return "none"
    return "same-view" if all(int(ref[k - 1]) == int(ref[k]) for k in j) else "view-change"


def other_reason(left, mid, right, ref: np.ndarray) -> str:
    """Why a candidate falls outside the binary task (only called for category 'other')."""
    x, y = left.label, mid.label; ref = np.asarray(ref, int)
    if sp.majority(ref[left.start:left.end]) != x or sp.majority(ref[right.start:right.end]) != x:
        return REASONS[0]
    inside = set(ref[mid.start:mid.end].tolist())
    if inside - {x, y}:
        return REASONS[1]
    if inside == {y}:
        return REASONS[2]
    return REASONS[3]


def tally(rows: list[dict], scores: np.ndarray, t: float) -> dict:
    """Correct middles kept / wrongly removed and wrong middles removed / missed, with recordings and sessions."""
    out = {}
    for cat, keep_name, drop_name in (("kept", "kept", "wrongly_removed"), ("removed", "missed", "removed")):
        idx = [i for i, r in enumerate(rows) if r["cat"] == cat]; vids = {v for i in idx for v in rows[i]["mid_videos"]}
        kept = int(sum(scores[i] >= t for i in idx))
        out[cat] = {"n": len(idx), keep_name: kept, drop_name: len(idx) - kept, "recordings": len(vids), "sessions": len({session_of(v) for v in vids})}
    return out


def rates_of(tl: dict) -> dict:
    a, b = tl["kept"], tl["removed"]
    return {"K": a["kept"] / a["n"] if a["n"] else None, "R": b["removed"] / b["n"] if b["n"] else None}


def reproduce(stored: dict, got: dict, tol_t: float = 1e-9, tol_auc: float = 1e-12) -> dict:
    """The rebuilt model against the probe's summary: thresholds, held-out AUC and held-out rates at both points."""
    dt = max(abs(got["thresholds"][k] - stored["thresholds"][k]) for k in stored["thresholds"])
    da = abs(got["held_out_auc"] - stored["held_out_auc"])
    rates_ok = all(got["points"][k][g] == stored["points"][k][g] for k in stored["points"] for g in got["points"][k])
    return {"max_threshold_diff": dt, "auc_diff": da, "rates_identical": bool(rates_ok), "ok": bool(dt <= tol_t and da <= tol_auc and rates_ok)}


def most(flags: list[bool]) -> bool:
    return sum(bool(f) for f in flags) >= 2


def ge(v, t) -> bool:
    return v is not None and v >= t


def branch(per_seed: list[dict]) -> dict:
    """The configuration's decision on per-seed pooled rates {model: {AUC, K_all, R_all, K_aa, R_aa}} and the Q1 - Q0
    comparison {R_all_ci_low, K_all_delta, R_aa_delta}."""
    fails = {m: most([(not ge(s[m]["AUC"], 0.80)) or (not ge(s[m]["K_all"], 0.90)) or (not ge(s[m]["R_all"], 0.50)) for s in per_seed]) for m in ("Q0", "Q1")}
    practical = most([ge(s["Q0"]["K_all"], 0.90) and ge(s["Q0"]["R_all"], 0.50) and ge(s["Q0"]["K_aa"], 0.90) and ge(s["Q0"]["R_aa"], 0.50) for s in per_seed])
    high_aa = most([ge(s["Q0"]["AUC"], 0.95) and ge(s["Q0"]["K_all"], 0.95) and ge(s["Q0"]["R_all"], 0.80) and s["Q0"]["K_aa"] is not None and s["Q0"]["K_aa"] < 0.90
                    for s in per_seed])
    extra = all(s["Q1 - Q0"]["R_all_ci_low"] is not None and s["Q1 - Q0"]["R_all_ci_low"] > 0 and s["Q1 - Q0"]["K_all_delta"] >= -0.02
                and (s["Q1 - Q0"]["R_aa_delta"] is None or s["Q1 - Q0"]["R_aa_delta"] >= 0) for s in per_seed)
    if fails["Q0"] and fails["Q1"]:
        case = "both fail"
    elif extra:
        case = "Q1 extra value"
    elif practical:
        case = "Q0 practical value"
    elif high_aa:
        case = "Q0 high overall but removes correct A4C/A5C middles"
    else:
        case = "no branch"
    return {"case": case, "fails": fails, "Q0_practical": practical, "Q0_high_but_aa": high_aa, "Q1_extra": extra}


# ----------------------------------------------------------------------------------------------- inputs
def check_probe(st: dict, root: Path) -> None:
    pr = st["probe"]; specs = {"config": pr["config"], "summary": pr["summary"]}
    for seed, lib in pr["libraries"].items():
        specs[f"arrays_s{seed}"], specs[f"meta_s{seed}"] = lib["arrays"], lib["meta"]
    for name, spec in specs.items():
        if (got := sha256_file(root / spec["path"])) != spec["sha256"]:
            raise SystemExit(f"probe {name}: sha256 {got[:12]} != {spec['sha256'][:12]}")


def library(st: dict, root: Path, seed: int) -> tuple[dict, list[dict]]:
    lib = st["probe"]["libraries"][seed]
    with np.load(root / lib["arrays"]["path"]) as z:
        arrays = {k: z[k] for k in z.files}
    return arrays, json.loads((root / lib["meta"]["path"]).read_text())


# ----------------------------------------------------------------------------------------------- rebuild
def rebuild_seed(arrays: dict, meta: list[dict], stored: dict, keep_rates: list) -> tuple[dict, dict]:
    from sklearn.metrics import roc_auc_score
    part = np.array([m["part"] for m in meta]); cat = np.array([m["cat"] for m in meta]); use = cat != "other"; y = (cat == "kept").astype(int)
    splice = np.array([m["splice"] for m in meta]); pipes, out = {}, {}
    masks = {"all": np.ones(len(meta), bool), "A4C/A5C pairs": np.array([{m["x"], m["y"]} == {sp.A4C, sp.A5C} for m in meta]),
             "other pairs": np.array([{m["x"], m["y"]} != {sp.A4C, sp.A5C} for m in meta]), "middle 10-40 samples": np.array([10 <= m["len_mid"] <= 40 for m in meta])}
    for name in ("Q0", "Q1"):
        x = sp.design(arrays, name); tr = use & (part == "train"); cal = use & (part == "calibration"); ho = use & (part == "held_out")
        pipe = sp.logistic(float(stored[name]["C"])).fit(x[tr], y[tr]); s = pipe.predict_proba(x)[:, 1]
        th = {str(k): sp.threshold_for(s[cal & (y == 1)], float(k)) for k in keep_rates}
        pts = {}
        for k, t in th.items():
            pts[k] = {g: sp.rates(s[ho & m], y[ho & m], t) for g, m in masks.items()}
            pts[k]["removed at a splice"] = sp.rates(s[ho & splice & (y == 0)], y[ho & splice & (y == 0)], t)
            pts[k]["removed inside a clip"] = sp.rates(s[ho & ~splice & (y == 0)], y[ho & ~splice & (y == 0)], t)
            pts[k]["calibration"] = sp.rates(s[cal], y[cal], t)
        got = {"thresholds": th, "held_out_auc": float(roc_auc_score(y[ho], s[ho])), "points": pts}
        out[name] = {"C": stored[name]["C"], **reproduce(stored[name], got)}; pipes[name] = pipe
    return out, pipes


def run_rebuild(cfg: dict, st: dict, paths, out_dir: Path) -> None:
    root = paths.repo_root; check_probe(st, root); summary = json.loads((root / st["probe"]["summary"]["path"]).read_text())
    keep_rates = cfg["short_segment_probe"]["operating_points"]["keep_rates"]; mdir = root / "runs" / NS / "models"; mdir.mkdir(parents=True, exist_ok=True)
    rec = {"what": "Q0 and Q1 of short_segment_probe_v1 refitted with the stored C on the original train part; no search", "seeds": {}}
    for seed in [int(s) for s in cfg["short_segment_probe"]["seeds"]]:
        arrays, meta = library(st, root, seed); res, pipes = rebuild_seed(arrays, meta, summary["seeds"][str(seed)]["models"], keep_rates)
        for name, pipe in pipes.items():
            if not res[name]["ok"]:
                raise SystemExit(f"seed {seed} {name}: the rebuilt model does not reproduce the probe ({res[name]})")
            path = mdir / f"{name}_s{seed}.pkl"; path.write_bytes(pickle.dumps(pipe)); res[name]["pickle"] = str(path.relative_to(root)); res[name]["sha256"] = sha256_file(path)
        rec["seeds"][str(seed)] = res; print(json.dumps({"seed": seed, **{n: {k: v for k, v in r.items() if k != "pickle"} for n, r in res.items()}}), flush=True)
    (root / "runs" / NS / "rebuild.json").write_text(json.dumps(rec, indent=1)); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rebuild.json").write_text(json.dumps(rec, indent=1)); print(f"wrote {out_dir / 'rebuild.json'}")


# ----------------------------------------------------------------------------------------------- apply
def half_sequences(inp: Inputs, half: str):
    ev = inp.halves[half]
    for s in ev.natives:
        vf = inp.store.get(s.seq_id, "orig")
        yield f"{half}|session:{session_of(s.seq_id)}", "native clips", vf.prob, vf.feat, s.labels, np.zeros(0, int), np.array([0, vf.n]), [s.seq_id]
    for kind, coll in (("pair streams", ev.pairs), ("multi streams", ev.multi)):
        for x in coll:
            bounds = np.cumsum([0, *[f.end - f.start for f in x.recipe.fragments]])
            yield f"{half}|stream:{x.recipe.recipe_id}", kind, x.prob, x.feat, x.labels, x.joins, bounds, [f.video_id for f in x.recipe.fragments]


def candidates(sp_cfg: dict, inp: Inputs, half: str) -> tuple[dict, list[dict]]:
    gen = sp_cfg["generator"]; w, eps, tol = int(sp_cfg["features"]["side_window"]), float(gen["eps"]), int(sp_cfg["candidates"]["tolerance_samples"])
    q0, vis, sim, rows = [], [], [], []
    for group, kind, prob, feat, ref, joins, bounds, videos in half_sequences(inp, half):
        pred = joint_viterbi(prob, np.zeros(len(prob)), 0.0, float(gen["kappa"]), eps); jsd = boundary_scores(prob, 7)
        for left, mid, right in sp.sandwiches(pred):
            cat, splice = sp.categorise(left, mid, right, ref, joins, tol)
            q0.append(sp.q0_features(prob, jsd, left, mid, right, w, eps)); v, c = sp.visual_features(feat, left, mid, right, w); vis.append(v); sim.append(c)
            mv = sp.videos_over(bounds, videos, mid.start, mid.end)
            rows.append({"half": half, "group": group, "kind": kind, "cat": cat, "reason": other_reason(left, mid, right, ref) if cat == "other" else None,
                         "join": join_type(ref, joins, mid.start, mid.end, tol), "splice": splice, "x": left.label, "y": mid.label, "len_mid": mid.length,
                         "mid_videos": mv, "mid_sessions": sorted({session_of(x) for x in mv})})
    dim = 3 * np.asarray(next(iter(half_sequences(inp, half)))[3]).shape[1]   # three mean feature blocks
    return {"q0": np.asarray(q0, np.float32).reshape(-1, 33), "vis": np.asarray(vis, np.float32).reshape(-1, dim),
            "sim": np.asarray(sim, np.float32).reshape(-1, 2)}, rows


def subset_mask(rows: list[dict], name: str) -> np.ndarray:
    tests = {"native clips": lambda r: r["kind"] == "native clips", "pair streams": lambda r: r["kind"] == "pair streams",
             "multi streams": lambda r: r["kind"] == "multi streams", "no join": lambda r: r["join"] == "none",
             "same-view join": lambda r: r["join"] == "same-view", "view-change join": lambda r: r["join"] == "view-change",
             "middle 10-40 samples": lambda r: 10 <= r["len_mid"] <= 40}
    return np.array([tests[name](r) for r in rows], bool)


def half_tables(rows: list[dict], scores: dict, thresholds: dict) -> dict:
    from sklearn.metrics import roc_auc_score
    aa = np.array([{r["x"], r["y"]} == {sp.A4C, sp.A5C} for r in rows], bool); binary = np.array([r["cat"] != "other" for r in rows], bool)
    y = np.array([r["cat"] == "kept" for r in rows], int); out = {"candidates": len(rows), "other": {}}
    other = [r for r in rows if r["cat"] == "other"]
    out["other"] = {"n": len(other), "share": len(other) / len(rows) if rows else None, "reasons": dict(Counter(r["reason"] for r in other)),
                    "A4C/A5C pairs": int(sum({r["x"], r["y"]} == {sp.A4C, sp.A5C} for r in other))}
    for m, s in scores.items():
        r = {}
        for k, t in thresholds[m].items():
            r[k] = {"A4C/A5C pairs": tally([rows[i] for i in np.flatnonzero(aa)], s[aa], t),
                    "other pairs": tally([rows[i] for i in np.flatnonzero(~aa)], s[~aa], t), "all": tally(rows, s, t)}
            r[k]["subsets"] = {n: tally([rows[i] for i in np.flatnonzero(subset_mask(rows, n))], s[subset_mask(rows, n)], t) for n in SUBSETS}
        ok = binary & ((y == 1) | (y == 0))
        r["AUC"] = float(roc_auc_score(y[ok], s[ok])) if len(set(y[ok].tolist())) == 2 else None
        r["AUC A4C/A5C"] = float(roc_auc_score(y[ok & aa], s[ok & aa])) if len(set(y[ok & aa].tolist())) == 2 else None
        out[m] = r
    return out


def run_apply(cfg: dict, st: dict, paths, out_dir: Path) -> None:
    root = paths.repo_root; check_probe(st, root); spc = cfg["short_segment_probe"]; rb = json.loads((root / "runs" / NS / "rebuild.json").read_text())
    summary = json.loads((root / st["probe"]["summary"]["path"]).read_text()); boot = st["bootstrap"]; prim = st["thresholds"]["primary"]
    res = {"seeds": {}, "decision_inputs": []}; t0 = time.time()
    for seed in [int(s) for s in spc["seeds"]]:
        pipes, thresholds = {}, {}
        for name in st["models"]:
            r = rb["seeds"][str(seed)][name]
            if sha256_file(root / r["pickle"]) != r["sha256"]:
                raise SystemExit(f"seed {seed} {name}: the pickled model changed")
            pipes[name] = pickle.loads((root / r["pickle"]).read_bytes()); thresholds[name] = summary["seeds"][str(seed)]["models"][name]["thresholds"]
        inp = Inputs(cfg, root / spc["inputs"]["protocol"]["path"], root / spc["inputs"]["b_feature_manifests"][seed]["path"], paths.cache_root)
        all_rows, all_scores, seed_res = [], {m: [] for m in st["models"]}, {}
        for h in HALVES:
            arrays, rows = candidates(spc, inp, h); scores = {m: (pipes[m].predict_proba(sp.design(arrays, m))[:, 1] if rows else np.zeros(0)) for m in st["models"]}
            seed_res[h] = half_tables(rows, scores, thresholds); all_rows += rows
            for m in st["models"]:
                all_scores[m].append(scores[m])
            (root / "runs" / NS / f"candidates_s{seed}_{h}.json").write_text(json.dumps(rows))
        s_all = {m: np.concatenate(v) for m, v in all_scores.items()}; pooled = half_tables(all_rows, s_all, thresholds); seed_res["pooled halves"] = pooled
        y = np.array([r["cat"] == "kept" for r in all_rows], int); binary = np.array([r["cat"] != "other" for r in all_rows], bool)
        groups = np.array([r["group"] for r in all_rows])
        ta, tb = thresholds["Q1"][prim], thresholds["Q0"][prim]
        bd = sp.bootstrap_diff(s_all["Q1"][binary], s_all["Q0"][binary], y[binary], ta, tb, groups[binary], int(boot["draws"]), int(boot["seed"]))
        r_aa = {m: rates_of(pooled[m][prim]["A4C/A5C pairs"]) for m in st["models"]}; r_all = {m: rates_of(pooled[m][prim]["all"]) for m in st["models"]}
        seed_res["Q1 - Q0"] = {"bootstrap": bd, "R_aa_delta": None if None in (r_aa["Q1"]["R"], r_aa["Q0"]["R"]) else r_aa["Q1"]["R"] - r_aa["Q0"]["R"]}
        res["seeds"][str(seed)] = seed_res
        res["decision_inputs"].append({**{m: {"AUC": pooled[m]["AUC"], "K_all": r_all[m]["K"], "R_all": r_all[m]["R"], "K_aa": r_aa[m]["K"], "R_aa": r_aa[m]["R"]} for m in st["models"]},
                                       "Q1 - Q0": {"R_all_ci_low": bd["removal"]["ci95"][0], "K_all_delta": bd["keep"]["delta"], "R_aa_delta": seed_res["Q1 - Q0"]["R_aa_delta"]}})
        print(json.dumps({"seed": seed, "decision_inputs": res["decision_inputs"][-1]}, default=str), flush=True)
    res["decision"] = branch(res["decision_inputs"]); res["seconds"] = time.time() - t0
    (root / "runs" / NS / "details.json").write_text(json.dumps(res, indent=1, default=str)); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(res, indent=1, default=str)); (out_dir / "report_tables.md").write_text(tables_md(res, prim))
    print(json.dumps(res["decision"], indent=1)); print(f"wrote {out_dir}")


def tables_md(res: dict, prim: str) -> str:
    rows, other, subs = [], [], []
    for s, r in res["seeds"].items():
        for h in (*HALVES, "pooled halves"):
            t = r[h]; other.append({"seed": s, "half": h, "candidates": t["candidates"], "outside the binary task": t["other"]["n"],
                                    "share": t["other"]["share"], **{k: t["other"]["reasons"].get(k, 0) for k in REASONS}})
            for m in ("Q0", "Q1"):
                for g in ("A4C/A5C pairs", "other pairs"):
                    a, b = t[m][prim][g]["kept"], t[m][prim][g]["removed"]
                    rows.append({"seed": s, "half": h, "model": m, "pairs": g, "correct kept / wrongly removed": f"{a['kept']} / {a['wrongly_removed']}",
                                 "correct rec / sess": f"{a['recordings']} / {a['sessions']}", "wrong removed / missed": f"{b['removed']} / {b['missed']}",
                                 "wrong rec / sess": f"{b['recordings']} / {b['sessions']}", "AUC (all)": t[m]["AUC"]})
                if h == "pooled halves":
                    for n, x in t[m][prim]["subsets"].items():
                        subs.append({"seed": s, "model": m, "subset": n, "correct kept / n": f"{x['kept']['kept']} / {x['kept']['n']}",
                                     "wrong removed / n": f"{x['removed']['removed']} / {x['removed']['n']}", "wrong rec / sess": f"{x['removed']['recordings']} / {x['removed']['sessions']}"})
    return "\n".join(["# Short-segment transfer check: tables\n", f"## A4C/A5C first (threshold {prim})\n", md_table(rows), "## Outside the binary task\n", md_table(other),
                      "## Sources and joins (halves pooled)\n", md_table(subs), "## Decision\n", "```\n" + json.dumps(res["decision"], indent=1) + "\n```\n"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--out", required=True)
    step = ap.add_mutually_exclusive_group(required=True); step.add_argument("--rebuild", action="store_true"); step.add_argument("--apply", action="store_true")
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); st = cfg["short_segment_transfer"]; paths = load_paths(); out = paths.repo_root / a.out
    (run_rebuild if a.rebuild else run_apply)(cfg, st, paths, out)


if __name__ == "__main__":
    main()
