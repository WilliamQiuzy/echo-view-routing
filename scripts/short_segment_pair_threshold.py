#!/usr/bin/env python
"""Pair-conditional thresholds for the frozen short-segment discriminators, a post-hoc decision calibration check
(configs/experiments/short_segment_pair_threshold_v1.yaml; docs/superpowers/plans/2026-10-07-short-segment-pair-threshold.md).
Exploratory reuse of the development set; nothing enters the full system and the test split stays sealed.

  --thresholds  Per seed and model, the six frozen pipelines of short_segment_transfer_v1 score the original training
                library. The A4C/A5C threshold is the highest that keeps at least 95% of the calibration part's correct
                candidates whose predicted pair is A4C/A5C (both directions share it), by the probe's own rule; every
                other pair keeps the original threshold. Writes thresholds.json, which is committed before --apply.
  --apply       Both validation halves: the same candidates (regenerated and checked against the saved lists), the
                frozen pipelines and the committed thresholds (recomputed and checked). Both rules per seed and half,
                the A4C/A5C decisions that change, the candidates outside the binary task, Q1 - Q0 under the pair rule
                and the fixed decision.

  python scripts/short_segment_pair_threshold.py --config configs/experiments/short_segment_pair_threshold_v1.yaml \
      --out docs/results/2026-10-07_short_segment_pair_threshold --thresholds
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
import scripts.short_segment_transfer as tr  # noqa: E402

import numpy as np  # noqa: E402

from echo_routing.config import load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.temporal.dev_run import HALVES, Inputs  # noqa: E402

NS = "short_segment_pair_threshold_v1"
GROUPS = ("A4C/A5C pairs", "other pairs", "all")
CATEGORY = {"kept": "correct", "removed": "wrong", "other": "outside"}


# ----------------------------------------------------------------------------------------------- pure helpers
def pair_of(names: list[str]) -> frozenset:
    return frozenset(sp.CLASSES.index(n) for n in names)


def in_pair(r: dict, pair: frozenset) -> bool:
    """The predicted (decoded) pair of X -> Y -> X, in either direction. Reference labels are never read."""
    return frozenset((int(r["x"]), int(r["y"]))) == pair


def direction(r: dict) -> str:
    x, y = sp.CLASSES[int(r["x"])], sp.CLASSES[int(r["y"])]
    return f"{x} -> {y} -> {x}"


def lower_bound(k: int, n: int, level: float = 0.95) -> float | None:
    """One-sided Clopper-Pearson lower bound for a rate of k of n."""
    from scipy.stats import beta
    if n == 0:
        return None
    return 0.0 if k == 0 else float(beta.ppf(1 - level, k, n - k + 1))


def pair_threshold(scores: np.ndarray, keep_rate: float) -> dict:
    """The probe's rule on the calibration part's correct candidates of the pair, and what it implies."""
    s = np.asarray(scores, float); t = sp.threshold_for(s, keep_rate); k = int((s >= t).sum())
    return {"threshold": t, "n": int(s.size), "kept": k, "keeps_all": k == s.size, "keep_lower_bound_95": lower_bound(k, int(s.size))}


def rule_thresholds(rows: list[dict], pair: frozenset, t_orig: float, t_pair: float | None) -> np.ndarray:
    """Per-candidate thresholds: the pair rule when t_pair is given, the original rule otherwise."""
    return np.array([t_pair if t_pair is not None and in_pair(r, pair) else t_orig for r in rows], float)


def decisions(rows: list[dict], scores: np.ndarray, pair: frozenset, t_orig: float, t_pair: float) -> tuple[np.ndarray, np.ndarray]:
    """Kept (True) under the original rule and under the pair rule."""
    s = np.asarray(scores, float)
    return s >= rule_thresholds(rows, pair, t_orig, None), s >= rule_thresholds(rows, pair, t_orig, t_pair)


def tally_kept(rows: list[dict], keep: np.ndarray) -> dict:
    """tr.tally on decisions instead of one threshold: correct middles kept / wrongly removed and wrong middles
    removed / missed, with distinct recordings and sessions."""
    out = {}
    for cat, keep_name, drop_name in (("kept", "kept", "wrongly_removed"), ("removed", "missed", "removed")):
        idx = [i for i, r in enumerate(rows) if r["cat"] == cat]; vids = {v for i in idx for v in rows[i]["mid_videos"]}
        k = int(sum(bool(keep[i]) for i in idx))
        out[cat] = {"n": len(idx), keep_name: k, drop_name: len(idx) - k, "recordings": len(vids), "sessions": len({session_of(v) for v in vids})}
    return out


def split_tallies(rows: list[dict], keep: np.ndarray, pair: frozenset) -> dict:
    m = np.array([in_pair(r, pair) for r in rows], bool); keep = np.asarray(keep, bool)
    def part(mask: np.ndarray) -> dict:
        return tally_kept([rows[i] for i in np.flatnonzero(mask)], keep[mask])
    return {"A4C/A5C pairs": part(m), "other pairs": part(~m), "all": tally_kept(rows, keep)}


def transitions(rows: list[dict], old: np.ndarray, new: np.ndarray, pair: frozenset) -> dict:
    """Candidates of the pair whose decision differs between the rules, with distinct recordings and sessions."""
    def count(cat: str, now_kept: bool) -> dict:
        idx = [i for i, r in enumerate(rows) if in_pair(r, pair) and r["cat"] == cat and bool(new[i]) == now_kept and bool(old[i]) != now_kept]
        vids = {v for i in idx for v in rows[i]["mid_videos"]}
        return {"n": len(idx), "recordings": len(vids), "sessions": len({session_of(v) for v in vids})}
    return {"correct rescued": count("kept", True), "correct lost": count("kept", False), "wrong let through": count("removed", True),
            "wrong newly removed": count("removed", False), "outside, now kept": count("other", True), "outside, now removed": count("other", False)}


def changed(rows: list[dict], scores: np.ndarray, old: np.ndarray, new: np.ndarray, pair: frozenset) -> list[dict]:
    """The pair's candidates whose decision changes, without ids ('_i' indexes rows and is dropped before writing)."""
    return [{"_i": i, "half": r["half"], "category": CATEGORY[r["cat"]], "now": "kept" if new[i] else "removed", "direction": direction(r),
             "source": r["kind"], "join": r["join"], "middle samples": int(r["len_mid"]), "score": round(float(scores[i]), 4)}
            for i, r in enumerate(rows) if in_pair(r, pair) and bool(old[i]) != bool(new[i])]


def unit_key(r: dict, view: dict) -> tuple:
    """A correct middle's true unit: half, sequence and the recordings of its own view under it (never written)."""
    return (r["half"], r["group"], tuple(sorted(v for v in r["mid_videos"] if view[v] == int(r["y"]))))


def attach_keys(entries: list[dict], rows: list[dict], view: dict) -> list[dict]:
    """Swap each changed entry's row index for its true-unit key (correct middles only); keys are never written."""
    for e in entries:
        i = e.pop("_i"); e["_key"] = unit_key(rows[i], view) if e["category"] == "correct" else None
    return entries


def add_cases(cases: dict, rows: list[dict], dec: dict, pair: frozenset, view: dict, seed: str) -> None:
    """Each correct middle of the pair under its true unit, with the seed's decisions under both rules for every model.
    A unit met by more than one middle in a seed keeps all of them (a list per seed)."""
    for i, r in enumerate(rows):
        if r["cat"] == "kept" and in_pair(r, pair):
            entry = cases.setdefault(unit_key(r, view), {"half": r["half"], "direction": direction(r), "seeds": {}})
            entry["seeds"].setdefault(seed, []).append({m: {"original": "kept" if d[0][i] else "removed", "pair": "kept" if d[1][i] else "removed"}
                                                        for m, d in dec.items()})


def number_cases(cases: dict, changed_by_seed: dict) -> list[dict]:
    """Case numbers 1.. in key order. The keys, which hold ids, are dropped; changed entries get the case number."""
    number = {k: n + 1 for n, k in enumerate(sorted(cases))}
    for per_model in changed_by_seed.values():
        for entries in per_model.values():
            for e in entries:
                key = e.pop("_key"); e["case"] = number[key] if key is not None else None
    return [{"case": number[k], **v} for k, v in sorted(cases.items())]


def pooled_rates(t: dict) -> dict:
    aa, al = tr.rates_of(t["A4C/A5C pairs"]), tr.rates_of(t["all"])
    return {"K_aa": aa["K"], "R_aa": aa["R"], "K_all": al["K"], "R_all": al["R"]}


def model_case(per_seed: list[dict], keep_min: float, removal_min: float) -> str:
    """The configuration's per-model case on the pooled rates of each seed."""
    if tr.most([tr.ge(s["K_aa"], keep_min) and tr.ge(s["R_aa"], removal_min) for s in per_seed]):
        return "meets"
    if tr.most([tr.ge(s["K_aa"], keep_min) and not tr.ge(s["R_aa"], removal_min) for s in per_seed]):
        return "keeps correct, lets wrong through"
    return "keep not reached"


def decide(rates: dict[str, list[dict]], keep_min: float, removal_min: float) -> dict:
    cases = {m: model_case(v, keep_min, removal_min) for m, v in rates.items()}; meets = [m for m, c in cases.items() if c == "meets"]
    if meets:
        outcome = (f"{' and '.join(meets)} meet the standard under the pair rule: supports 'the current operating point was unsuitable'; "
                   "a full-system experiment comes next (not run here; no claim over the baselines)")
    else:
        outcome = "no model meets the standard under the pair rule: the threshold check of these two discriminators ends; no further thresholds"
    return {"cases": cases, "meets": meets, "outcome": outcome}


def bootstrap_decisions(ka: np.ndarray, kb: np.ndarray, y: np.ndarray, groups: np.ndarray, draws: int, seed: int) -> dict:
    """sp.bootstrap_diff on decisions: removal of wrong and keep of correct middles, a - b, resampling groups with the
    same draws. Unlike sp.bootstrap_diff, a sample without one of the classes gives None for that rate, not an error."""
    ka, kb, y, groups = np.asarray(ka, bool), np.asarray(kb, bool), np.asarray(y, int), np.asarray(groups)
    def diff(ii: np.ndarray) -> dict:
        yy = y[ii]
        return {"removal": float((~ka[ii][yy == 0]).mean() - (~kb[ii][yy == 0]).mean()) if (yy == 0).any() else None,
                "keep": float(ka[ii][yy == 1].mean() - kb[ii][yy == 1].mean()) if (yy == 1).any() else None}
    point = diff(np.arange(y.size)); vals = {"removal": [], "keep": []}
    if y.size:
        keys = np.unique(groups); idx = {k: np.flatnonzero(groups == k) for k in keys}; rng = np.random.default_rng(seed)
        for _ in range(draws):
            for k, v in diff(np.concatenate([idx[g] for g in rng.choice(keys, size=keys.size, replace=True)])).items():
                if v is not None:
                    vals[k].append(v)
    return {k: {"delta": point[k], "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [None, None]} for k, v in vals.items()}


# ----------------------------------------------------------------------------------------------- inputs
def seeds_of(cfg: dict) -> list[int]:
    return [int(s) for s in cfg["short_segment_probe"]["seeds"]]


def check_transfer(pt: dict, root: Path) -> dict:
    for name in ("rebuild", "summary"):
        spec = pt["transfer"][name]
        if (got := sha256_file(root / spec["path"])) != spec["sha256"]:
            raise SystemExit(f"transfer {name}: sha256 {got[:12]} != {spec['sha256'][:12]}")
    return {n: json.loads((root / pt["transfer"][n]["path"]).read_text()) for n in ("rebuild", "summary")}


def frozen(rb: dict, root: Path, seed: int, models: list[str]) -> dict:
    pipes = {}
    for name in models:
        r = rb["seeds"][str(seed)][name]
        if sha256_file(root / r["pickle"]) != r["sha256"]:
            raise SystemExit(f"seed {seed} {name}: the pickled model changed")
        pipes[name] = pickle.loads((root / r["pickle"]).read_bytes())
    return pipes


def check_candidates(root: Path, pt: dict, seed: int, half: str, rows: list[dict]) -> None:
    """The regenerated candidates must equal the transfer check's saved list, whose sha256 the config pins."""
    path = root / "runs" / tr.NS / f"candidates_s{seed}_{half}.json"
    if sha256_file(path) != pt["transfer"]["candidates"][seed][half]:
        raise SystemExit(f"seed {seed} {half}: the saved candidate list changed")
    if json.loads(path.read_text()) != json.loads(json.dumps(rows)):
        raise SystemExit(f"seed {seed} {half}: the regenerated candidates differ from the saved list")


def protocol_views(path: Path) -> dict[str, int]:
    proto = json.loads(path.read_text()); view = {}
    for h in HALVES:
        for f in (f for bank in proto["validation"][h]["banks"].values() for rc in bank["recipes"] for f in rc["fragments"]):
            if view.setdefault(f["video_id"], int(f["label"])) != int(f["label"]):
                raise SystemExit("a recording carries two reference views")
    return view


# ----------------------------------------------------------------------------------------------- thresholds
def seed_thresholds(arrays: dict, meta: list[dict], pipes: dict, original: dict, pair: frozenset, keep_rate: float, label: str = "") -> dict:
    """Per model: the original threshold (checked on the library), the pair threshold from the calibration part with
    its counts, and the library's held-out part under both rules (supplementary)."""
    part = np.array([m["part"] for m in meta]); cat = np.array([m["cat"] for m in meta]); aa = np.array([in_pair(m, pair) for m in meta], bool)
    cal, ho = np.flatnonzero((part == "calibration") & aa), np.flatnonzero((part == "held_out") & aa); out = {}
    for name, pipe in pipes.items():
        s = pipe.predict_proba(sp.design(arrays, name))[:, 1]; t0 = float(original[name])
        if abs(sp.threshold_for(s[(part == "calibration") & (cat == "kept")], keep_rate) - t0) > 1e-9:
            raise SystemExit(f"{label} {name}: the frozen pipeline does not reproduce the original threshold on the library")
        if not (cat[cal] == "kept").any():
            raise SystemExit(f"{label} {name}: no correct A4C/A5C candidate in the library's calibration part")
        rule = pair_threshold(s[cal][cat[cal] == "kept"], keep_rate); t1 = rule["threshold"]
        out[name] = {"original threshold": t0, "pair threshold": t1, "pair vs original": "lower" if t1 < t0 else "higher" if t1 > t0 else "equal",
                     "calibration": {"pair rule": rule, "correct by direction": dict(sorted(Counter(direction(meta[i]) for i in cal if cat[i] == "kept").items())),
                                     "at the original threshold": _at(meta, s, cal, t0), "at the pair threshold": _at(meta, s, cal, t1)},
                     "held-out, supplementary": {"original": _at(meta, s, ho, t0), "pair": _at(meta, s, ho, t1)}}
    return out


def _at(meta: list[dict], s: np.ndarray, idx: np.ndarray, t: float) -> dict:
    return tally_kept([meta[i] for i in idx], s[idx] >= t)


def original_thresholds(probe: dict, seed: int, models: list[str], keep_rate: float) -> dict:
    return {m: float(probe["seeds"][str(seed)]["models"][m]["thresholds"][str(keep_rate)]) for m in models}


def run_thresholds(cfg: dict, pt: dict, paths, out_dir: Path) -> None:
    root = paths.repo_root; st = cfg["short_segment_transfer"]; tr.check_probe(st, root); inputs = check_transfer(pt, root)
    if (out_dir / "thresholds.json").exists():
        raise SystemExit("thresholds.json exists: the thresholds are computed once, committed and never recomputed in place")
    probe = json.loads((root / st["probe"]["summary"]["path"]).read_text()); pair = pair_of(pt["pair_threshold"]["pair"]); kr = float(pt["pair_threshold"]["keep_rate"])
    rec = {"what": "per seed and model, the A4C/A5C threshold from the library's calibration part (probe rule, keep >= 95% of correct A4C/A5C); "
                   "other pairs keep the original threshold", "keep_rate": kr, "seeds": {}}; t0 = time.time()
    for seed in seeds_of(cfg):
        arrays, meta = tr.library(st, root, seed); pipes = frozen(inputs["rebuild"], root, seed, pt["models"])
        rec["seeds"][str(seed)] = seed_thresholds(arrays, meta, pipes, original_thresholds(probe, seed, pt["models"], kr), pair, kr, f"seed {seed}")
        print(json.dumps({"seed": seed, **{m: {k: rec["seeds"][str(seed)][m][k] for k in ("original threshold", "pair threshold", "pair vs original")} for m in pt["models"]}}), flush=True)
    rec["seconds"] = time.time() - t0; out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "thresholds.json").write_text(json.dumps(rec, indent=1)); print(f"wrote {out_dir / 'thresholds.json'}")


# ----------------------------------------------------------------------------------------------- apply
def rule_tables(rows: list[dict], scores: dict, t_orig: dict, t_pair: dict, pair: frozenset) -> dict:
    out = {"candidates": len(rows), "outside the binary task": {"n": sum(r["cat"] == "other" for r in rows),
                                                                "A4C/A5C pairs": sum(r["cat"] == "other" and in_pair(r, pair) for r in rows)}}
    for m, s in scores.items():
        old, new = decisions(rows, s, pair, t_orig[m], t_pair[m])
        out[m] = {"original": split_tallies(rows, old, pair), "pair": split_tallies(rows, new, pair), "transitions": transitions(rows, old, new, pair)}
    return out


def check_original(pooled: dict, saved: dict, models: list[str], prim: str) -> None:
    """The original rule must reproduce the transfer check's pooled tallies exactly."""
    for m in models:
        for g in GROUPS:
            if pooled[m]["original"][g] != saved[m][prim][g]:
                raise SystemExit(f"{m} {g}: the original rule does not reproduce the transfer check")


def seed_apply(ctx: dict, seed: int) -> tuple[dict, list[dict], np.ndarray, dict]:
    """Candidates of both halves, scores, both rules per half and pooled; returns pooled rows, decisions and scores."""
    cfg, pt, root, models = ctx["cfg"], ctx["pt"], ctx["root"], ctx["pt"]["models"]; spc = cfg["short_segment_probe"]
    pipes = frozen(ctx["rebuild"], root, seed, models); t_orig = original_thresholds(ctx["probe"], seed, models, ctx["keep_rate"])
    arrays, meta = tr.library(cfg["short_segment_transfer"], root, seed)
    again = seed_thresholds(arrays, meta, pipes, t_orig, ctx["pair"], ctx["keep_rate"], f"seed {seed}")
    t_pair = {m: float(ctx["thresholds"]["seeds"][str(seed)][m]["pair threshold"]) for m in models}   # the committed values are applied
    if any(abs(again[m]["pair threshold"] - t_pair[m]) > 1e-9 for m in models):
        raise SystemExit(f"seed {seed}: the pair thresholds differ from the committed thresholds.json")
    inp = Inputs(cfg, root / spc["inputs"]["protocol"]["path"], root / spc["inputs"]["b_feature_manifests"][seed]["path"], ctx["paths"].cache_root)
    rows_all, sc_all, res = [], {m: [] for m in models}, {"thresholds": {"original": t_orig, "pair": t_pair}}
    for h in HALVES:
        arr, rows = tr.candidates(spc, inp, h); check_candidates(root, pt, seed, h, rows)
        sc = {m: (pipes[m].predict_proba(sp.design(arr, m))[:, 1] if rows else np.zeros(0)) for m in models}
        (root / "runs" / NS / f"scores_s{seed}_{h}.json").write_text(json.dumps({m: v.tolist() for m, v in sc.items()}))
        res[h] = rule_tables(rows, sc, t_orig, t_pair, ctx["pair"]); rows_all += rows
        for m in models:
            sc_all[m].append(sc[m])
    s_all = {m: np.concatenate(v) for m, v in sc_all.items()}; res["pooled halves"] = rule_tables(rows_all, s_all, t_orig, t_pair, ctx["pair"])
    check_original(res["pooled halves"], ctx["transfer"]["seeds"][str(seed)]["pooled halves"], models, ctx["prim"])
    dec = {m: decisions(rows_all, s_all[m], ctx["pair"], t_orig[m], t_pair[m]) for m in models}
    return res, rows_all, dec, s_all


def compare(rows: list[dict], dec: dict, pair: frozenset, boot: dict) -> dict:
    """Q1 - Q0 with both models under the pair rule, overall and for the pair."""
    y = np.array([r["cat"] == "kept" for r in rows], int); binary = np.array([r["cat"] != "other" for r in rows], bool)
    aa = np.array([in_pair(r, pair) for r in rows], bool); groups = np.array([r["group"] for r in rows])
    return {g: bootstrap_decisions(dec["Q1"][1][m], dec["Q0"][1][m], y[m], groups[m], int(boot["draws"]), int(boot["seed"]))
            for g, m in (("all", binary), ("A4C/A5C pairs", binary & aa))}


def run_apply(cfg: dict, pt: dict, paths, out_dir: Path) -> None:
    root = paths.repo_root; st = cfg["short_segment_transfer"]; spc = cfg["short_segment_probe"]; tr.check_probe(st, root); inputs = check_transfer(pt, root)
    ctx = {"cfg": cfg, "pt": pt, "root": root, "paths": paths, "rebuild": inputs["rebuild"], "transfer": inputs["summary"], "prim": st["thresholds"]["primary"],
           "probe": json.loads((root / st["probe"]["summary"]["path"]).read_text()), "pair": pair_of(pt["pair_threshold"]["pair"]),
           "keep_rate": float(pt["pair_threshold"]["keep_rate"]), "thresholds": json.loads((out_dir / "thresholds.json").read_text())}
    view = protocol_views(root / spc["inputs"]["protocol"]["path"]); models = pt["models"]; (root / "runs" / NS).mkdir(parents=True, exist_ok=True)
    res = {"thresholds_sha256": sha256_file(out_dir / "thresholds.json"), "seeds": {}, "rates": {r: {m: [] for m in models} for r in ("original", "pair")}, "changed": {}}
    cases, t0 = {}, time.time()
    for seed in seeds_of(cfg):
        seed_res, rows, dec, s_all = seed_apply(ctx, seed); seed_res["Q1 - Q0, pair rule"] = compare(rows, dec, ctx["pair"], pt["bootstrap"])
        for rule in ("original", "pair"):
            for m in models:
                res["rates"][rule][m].append(pooled_rates(seed_res["pooled halves"][m][rule]))
        res["changed"][str(seed)] = {m: attach_keys(changed(rows, s_all[m], *dec[m], ctx["pair"]), rows, view) for m in models}
        add_cases(cases, rows, dec, ctx["pair"], view, str(seed))
        res["seeds"][str(seed)] = seed_res; print(json.dumps({"seed": seed, "pair rule": {m: res["rates"]["pair"][m][-1] for m in models}}), flush=True)
    res["correct A4C/A5C cases"] = number_cases(cases, res["changed"])
    std = pt["decision"]["standard"]
    res["decision"] = {**decide(res["rates"]["pair"], float(std["keep_aa_min"]), float(std["removal_aa_min"])),
                       "original rule, for reference": {m: model_case(v, float(std["keep_aa_min"]), float(std["removal_aa_min"])) for m, v in res["rates"]["original"].items()}}
    res["seconds"] = time.time() - t0; out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(res, indent=1)); (out_dir / "report_tables.md").write_text(tables_md(res, ctx["thresholds"]))
    print(json.dumps(res["decision"], indent=1)); print(f"wrote {out_dir}")


# ----------------------------------------------------------------------------------------------- report
def frac(t: dict, cat: str, key: str) -> str:
    return f"{t[cat][key]}/{t[cat]['n']}"


def columns(rows: list[dict]) -> list[str]:
    """Every key of every row in first-seen order (md_table otherwise takes the first row's keys only)."""
    return list(dict.fromkeys(k for r in rows for k in r))


def case_rows(cases: list[dict]) -> list[dict]:
    """One row per correct A4C/A5C case: per seed and model, original -> pair ('; ' between middles of one unit)."""
    return [{"case": c["case"], "half": c["half"], "direction": c["direction"],
             **{f"s{s} {m}": "; ".join(f"{d[m]['original']} -> {d[m]['pair']}" for d in lst) for s, lst in c["seeds"].items() for m in ("Q0", "Q1")}}
            for c in cases]


def tables_md(res: dict, th: dict) -> str:
    thr, aa, tra, rates, cmp_, cases, chg = [], [], [], [], [], [], []
    for s, per in th["seeds"].items():
        for m, v in per.items():
            c = v["calibration"]; r = c["pair rule"]
            thr.append({"seed": s, "model": m, "original": round(v["original threshold"], 4), "pair": round(v["pair threshold"], 4), "pair vs original": v["pair vs original"],
                        "cal correct A4C/A5C kept (pair)": f"{r['kept']}/{r['n']}", "keeps all": r["keeps_all"], "keep lower bound": round(r["keep_lower_bound_95"], 3),
                        "cal correct rec/sess": f"{c['at the pair threshold']['kept']['recordings']}/{c['at the pair threshold']['kept']['sessions']}",
                        "cal wrong removed (orig / pair)": f"{frac(c['at the original threshold'], 'removed', 'removed')} / {frac(c['at the pair threshold'], 'removed', 'removed')}"})
    for s, r in res["seeds"].items():
        for h in (*HALVES, "pooled halves"):
            for m in ("Q0", "Q1"):
                for rule in ("original", "pair"):
                    t = r[h][m][rule]["A4C/A5C pairs"]
                    aa.append({"seed": s, "half": h, "model": m, "rule": rule, "correct kept / removed": f"{t['kept']['kept']} / {t['kept']['wrongly_removed']}",
                               "correct rec / sess": f"{t['kept']['recordings']} / {t['kept']['sessions']}", "wrong removed / missed": f"{t['removed']['removed']} / {t['removed']['missed']}",
                               "wrong rec / sess": f"{t['removed']['recordings']} / {t['removed']['sessions']}"})
        for m in ("Q0", "Q1"):
            x = r["pooled halves"][m]["transitions"]
            tra.append({"seed": s, "model": m, **{k: f"{v['n']} ({v['recordings']} rec, {v['sessions']} sess)" for k, v in x.items()}})
        for g, b in r["Q1 - Q0, pair rule"].items():
            cmp_.append({"seed": s, "pairs": g, **{f"{k} delta": b[k]["delta"] for k in ("removal", "keep")}, **{f"{k} ci95": b[k]["ci95"] for k in ("removal", "keep")}})
    for rule, per in res["rates"].items():
        for m, lst in per.items():
            for s, v in zip(res["seeds"], lst):
                rates.append({"rule": rule, "model": m, "seed": s, **{k: None if x is None else round(x, 3) for k, x in v.items()}})
    cases = case_rows(res["correct A4C/A5C cases"])
    for s, per in res["changed"].items():
        for m, lst in per.items():
            chg += [{"seed": s, "model": m, **e} for e in lst]
    def tab(rows: list[dict]) -> str:
        return md_table(rows, columns(rows))
    return "\n".join(["# Short-segment pair thresholds: tables\n", "## Thresholds (library calibration part)\n", tab(thr), "## A4C/A5C under both rules\n", tab(aa),
                      "## A4C/A5C decisions that change (pooled halves)\n", tab(tra), "## Pooled rates\n", tab(rates), "## Q1 - Q0 under the pair rule\n", tab(cmp_),
                      "## Correct A4C/A5C cases (original -> pair)\n", tab(cases), "## Changed candidates of the pair\n", tab(chg),
                      "## Decision\n", "```\n" + json.dumps(res["decision"], indent=1) + "\n```\n"])


def check_out(out: str, pt: dict) -> None:
    """Only this round's own results directory; earlier rounds' results are never written."""
    if Path(out) != Path(pt["results_dir"]):
        raise SystemExit(f"--out must be {pt['results_dir']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--out", required=True)
    step = ap.add_mutually_exclusive_group(required=True)
    for name in ("--thresholds", "--apply"):
        step.add_argument(name, action="store_true")
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); pt = cfg["short_segment_pair_threshold"]; paths = load_paths(); check_out(a.out, pt)
    (run_thresholds if a.thresholds else run_apply)(cfg, pt, paths, paths.repo_root / a.out)


if __name__ == "__main__":
    main()
