#!/usr/bin/env python
"""Evaluate A15 (configs/experiments/video_router_data_v1.yaml) on the development halves, from A13's verified frozen A08
caches: Base (A08 + B3), Alternative (A13 H), accept-all, A14's frozen P at its committed choice, and small-P and large-P at
one calibration choice each; then, after selection only, the reference-assisted zone diagnostic (never a system).

  --stage calibrate               the calibration half only; writes choices.json, then router-check logging at the choices
  --stage select --choices-sha S  both halves after choices.json (sha256 S) has been committed locally
  --stage diagnostic              after select: single-zone utilities and all individually beneficial zones applied together
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.local_router as lr  # noqa: E402
import scripts.report_raw_video_context as rr  # noqa: E402
import scripts.report_segment_objective as rso  # noqa: E402
import scripts.report_video_sustained_switch as rvs  # noqa: E402
import scripts.report_video_switch_head as rsh  # noqa: E402
import scripts.report_video_within_view as rw  # noqa: E402
import scripts.segment_objective_tables as st  # noqa: E402
import scripts.train_local_router as tl  # noqa: E402
import scripts.train_router_data as trd  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402

HALVES = ("selection", "calibration")
GRID = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99)
ARMS = tuple(trd.ARMS)
A14_CHOICES = {"path": "runs/video_local_router_v1/evaluation/choices.json", "sha256": "3302adc121e58d8d540879a63dcd3a93f57d5b058739c4baf9372e5f276873e2"}
A14_SCREEN = ROOT / "runs/video_local_router_v1/evaluation/screen.json"
CMPS = {"large-P - small-P": ("large-P", "small-P"), "small-P - Base": ("small-P", "Base"), "large-P - Base": ("large-P", "Base"), "small-P - Alternative": ("small-P", "Alternative"),
        "large-P - Alternative": ("large-P", "Alternative"), "large-P - A14 P": ("large-P", "A14 P"), "small-P - A14 P": ("small-P", "A14 P")}


def router(ctx: trd.Ctx, arm: str, s: int):
    rec = json.loads((ctx.out / "routers" / f"{arm}_s{s}.json").read_text())
    if rec.get("status") != "trained":
        return None
    p = ctx.out / "routers" / f"{arm}_s{s}.pt"
    if sha256_file(p) != rec["sha256"]:
        raise SystemExit(f"{arm}_s{s}: router weights differ from their record")
    st_ = torch.load(p, map_location="cpu", weights_only=True); m = tl.build("P", int(st_["d"]), s); m.load_state_dict(st_["model"]); m.eval()
    return m, {"mu": st_["mu"].numpy(), "sd": st_["sd"].numpy()}, rec["sha256"]


def a14_router(s: int):
    ch = json.loads(tl.tf.vsh.pinned(A14_CHOICES).read_text()); p = ROOT / "runs/video_local_router_v1/routers" / f"P_s{s}.pt"
    if sha256_file(p) != ch["routers"][f"P_s{s}"]:
        raise SystemExit(f"A14 P_s{s}: the frozen router differs from A14's committed choices")
    st_ = torch.load(p, map_location="cpu", weights_only=True); m = tl.build("P", int(st_["d"]), s); m.load_state_dict(st_["model"]); m.eval()
    return m, {"mu": st_["mu"].numpy(), "sd": st_["sd"].numpy()}, ch["choices"][f"P_s{s}"]["threshold"], ch["routers"][f"P_s{s}"]


def read_half(ctx: trd.Ctx, a8, h: str, seeds: list[int]) -> dict:
    ev, seqs = a8.half(h, rso.DevAccess({h})); refs = rw.half_refs(seqs); at = {r: i for i, r in enumerate(refs)}; W = rr.half_windows(seqs, at, 5)
    orders = np.broadcast_to(np.arange(16, dtype=np.int64), W["s"].shape).copy(); a8sum = json.loads((ROOT / "docs/results/2026-10-16_video_within_view/summary.json").read_text())
    a13rows = json.loads((ROOT / "docs/results/2026-10-20_frozen_switch/screen.json").read_text())["systems"]; pins = rsh.a08_output_pins(); out = {"ev": ev, "seqs": seqs, "seed": {}}
    for s in seeds:
        with np.load(ROOT / "runs/video_frozen_switch_v1/evaluation" / h / f"a08_s{s}.npz") as z:
            if not (np.array_equal(z["seq"], W["seq"]) and np.array_equal(z["centre"], W["centre"])):
                raise SystemExit("A13's cache does not match this half's windows")
            lg, em = z["logits"].astype(np.float32), z["emb"].astype(np.float32)
        apath = ROOT / "runs/video_within_view_v1/evaluation" / h / f"all_s{s}.npz"
        if sha256_file(apath) != pins[h][s]:
            raise SystemExit("A08 saved outputs differ from their pin")
        lgA, _ = rr.cached_logits(apath, a8sum["runs"][f"all_s{s}"]["identity"], W, orders, seqs)
        if lgA is None or not tl.tf.vsh.vso.parity(lg, lgA, 1e-4)["ok"]:
            raise SystemExit("the cached frozen A08 logits differ from A08's saved outputs")
        probs = rr.per_sample(W, lg, seqs); head, thr, _ = ctx.a14.alt(s)
        edges = rvs.edge_scores(head, W, lg, em, seqs, torch.device("cpu")); base, alt = tl.systems(probs, edges, seqs, ctx.a14.base_params(s), thr)
        cand = trd.to_arrays(trd.candidates_p(seqs, probs, base, alt, with_labels=False)); scores = {}
        for arm in ARMS:
            r = router(ctx, arm, s)
            if r is not None:
                scores[arm] = tl.score(r[0], cand["fp"], r[1])
        m14, st14, thr14, _ = a14_router(s); scores["A14 P"] = tl.score(m14, cand["fp"], st14)
        out["seed"][s] = {"base": base, "alt": alt, "cand": cand, "scores": scores, "a14_threshold": thr14}
        rb, ra = rso.row_of(a8.cfg, base, ev)["row"], rso.row_of(a8.cfg, alt, ev)["row"]
        rso.check_stored(rb, a8sum["systems"][f"A + B3 s{s}"][h], f"Base s{s} {h} (A08 A + B3)"); rso.check_stored(ra, a13rows[f"H head s{s}"][h], f"Alternative s{s} {h} (A13 H)")
    return out


def routed(P: dict, s: int, accept: np.ndarray) -> dict:
    d = P["seed"][s]; c = d["cand"]; out = {}
    for qi, q in enumerate(P["seqs"]):
        sel = np.flatnonzero(c["seq"] == qi); zs = [(int(c["a"][i]), int(c["b"][i])) for i in sel]
        out[q["id"]] = lr.apply(d["base"][q["id"]], d["alt"][q["id"]], zs, [bool(accept[i]) for i in sel]) if sel.size else d["base"][q["id"]].copy()
    return out


def accept_of(scores: np.ndarray, t) -> np.ndarray:
    return np.zeros(scores.size, bool) if t == "noop" else scores >= float(t)


def calibrate(ctx: trd.Ctx, seeds: list[int]) -> None:
    t0, started, ok, note = time.time(), tl.now(), False, ""
    try:
        a8 = rsh.a08_ctx(); rw.check_grid(a8); P = read_half(ctx, a8, "calibration", seeds); choices = {}
        for s in seeds:
            for arm in ARMS:
                if arm not in P["seed"][s]["scores"]:
                    choices[f"{arm}_s{s}"] = {"status": "preparation not met", "threshold": None}; continue
                sc = P["seed"][s]["scores"][arm]; recs = []
                for t in ["noop", *GRID]:
                    acc = accept_of(sc, t); m, _ = tl.rsh.rcs.evaluate(a8.cfg, routed(P, s, acc), P["ev"])
                    recs.append({"threshold": t, "accepted": int(acc.sum()), **{k: m[k] for k in tl.rsh.rcs.FLAT}})
                feas = [r for r in recs if r["native_macro_f1"] >= 0.980 and r["boundary_recall_multi"] >= 0.95]
                best = max(feas, key=lambda r: (r["segment_f1_50_multi"], -r["accepted"])) if feas else None
                choices[f"{arm}_s{s}"] = {"status": "feasible" if best else "infeasible", "threshold": best["threshold"] if best else "noop", "grid": recs}
                print(f"{arm}_s{s}: {choices[f'{arm}_s{s}']['status']} threshold {choices[f'{arm}_s{s}']['threshold']} (accepted {best['accepted'] if best else 0} of {sc.size})", flush=True)
        routers = {f"{arm}_s{s}": router(ctx, arm, s)[2] for s in seeds for arm in ARMS if f"{arm}_s{s}" in choices and choices[f"{arm}_s{s}"]["status"] != "preparation not met"}
        a14p = {f"P_s{s}": {"threshold": a14_router(s)[2], "sha256": a14_router(s)[3]} for s in seeds}
        rec = {"stage": "calibrate", "seeds": seeds, "config_sha256": sha256_file(ROOT / ctx.config), "choices": choices, "routers": routers, "A14_P_committed": a14p, "time": tl.now()}
        flat = json.loads(json.dumps(rec, default=str)); assert_no_ids(flat, rw.ids_of(a8, {"calibration": P["seqs"]}), "choices")
        for p in (ctx.out / "evaluation" / "choices.json", ctx.results / "choices.json"):
            p.parent.mkdir(parents=True, exist_ok=True); tl.write_json(p, flat)
        log, ck = {}, trd.check_kinds(ctx)                              # written after the choices; nothing is chosen from it
        for s in seeds:
            c = trd.load_cand(ctx, s); pr = trd.load_preds(ctx, s)
            log[f"Base_s{s}"] = trd.check_log(c, pr, ck, np.zeros(c["seq"].size), "noop"); log[f"Alternative_s{s}"] = trd.check_log(c, pr, ck, np.ones(c["seq"].size), 0.5)
            for arm in ARMS:
                r = router(ctx, arm, s)
                if r is not None:
                    sc = tl.score(r[0], c["fp"], r[1]); log[f"{arm}_s{s}"] = {"at_0.5": trd.check_log(c, pr, ck, sc, 0.5), "at_choice": trd.check_log(c, pr, ck, sc, choices[f"{arm}_s{s}"]["threshold"])}
        for p in (ctx.out / "evaluation" / "check_log.json", ctx.results / "check_log.json"):
            tl.write_json(p, json.loads(json.dumps({"note": "router-check logging on the session-disjoint check pool; never used for a choice", "log": log}, default=str)))
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        tl.ledger_append(ctx.ledger, "evaluate calibrate (CPU)", "cpu", started, time.time() - t0, ok, note)


def third_seed(rows: dict, counts: dict, arms: list[str]) -> dict:
    cells = [(s, h) for s in (0, 2) for h in HALVES]; out = {}; F = lambda n, s, h: rows[f"{n} s{s}"][h]["segment_f1_50_multi"]  # noqa: E731
    for x in arms:
        dF = {f"s{s} {h}": F(x, s, h) - F("Base", s, h) for s, h in cells}; half = {h: float(np.mean([dF[f"s{s} {h}"] for s in (0, 2)])) for h in HALVES}
        dN = float(np.mean([rows[f"{x} s{s}"][h]["native_macro_f1"] - rows[f"Base s{s}"][h]["native_macro_f1"] for s, h in cells]))
        dBR = float(np.mean([rows[f"{x} s{s}"][h]["boundary_recall_multi"] - rows[f"Base s{s}"][h]["boundary_recall_multi"] for s, h in cells]))
        fs = {n: int(sum(counts[f"{n} s{s}"][h]["fn_short"] for s, h in cells)) for n in (x, "Base")}
        r = {"mean_dF": float(np.mean(list(dF.values()))), "dF": dF, "half_means": half, "mean_dN": dN, "mean_dBR": dBR, "missed_short_true": fs}
        r["signal"] = bool(r["mean_dF"] >= 0.003); r["no_reversal"] = bool(min(half.values()) >= -0.005 and min(dF.values()) >= -0.015)
        r["no_harm"] = bool(dN >= -0.003 and dBR >= -0.010 and fs[x] <= fs["Base"] + 3); r["passes"] = bool(r["signal"] and r["no_reversal"] and r["no_harm"]); out[x] = r
    out["train_seed1"] = bool(any(out[x]["passes"] for x in arms))
    if {"small-P", "large-P"} <= set(arms):
        d = {f"s{s} {h}": F("large-P", s, h) - F("small-P", s, h) for s, h in cells}
        out["development_large_minus_small"] = {"dF": d, "mean": float(np.mean(list(d.values()))), "cells_ahead": int(sum(v > 0 for v in d.values())), "cells_behind": int(sum(v < 0 for v in d.values()))}
    return out


def select(ctx: trd.Ctx, seeds: list[int], choices_sha: str) -> None:
    t0, started, ok, note = time.time(), tl.now(), False, ""
    try:
        cp = ctx.out / "evaluation" / "choices.json"
        if sha256_file(cp) != choices_sha:
            raise SystemExit("choices.json differs from the committed sha256; selection is not read")
        choices = json.loads(cp.read_text())["choices"]; a8 = rsh.a08_ctx(); per = {h: read_half(ctx, a8, h, seeds) for h in ("calibration", "selection")}; systems, zstats = {}, {}
        a14 = json.loads(A14_SCREEN.read_text())["systems"]; arms = [a for a in ARMS if all(choices[f"{a}_s{s}"]["status"] != "preparation not met" for s in seeds)]
        for h in ("calibration", "selection"):
            P = per[h]
            for s in seeds:
                d = P["seed"][s]; n = d["cand"]["seq"].size
                entries = {"Base": routed(P, s, np.zeros(n, bool)), "Alternative": d["alt"], "accept-all": routed(P, s, np.ones(n, bool)), "A14 P": routed(P, s, accept_of(d["scores"]["A14 P"], d["a14_threshold"]))}
                zstats[f"A14 P s{s} {h}"] = {"candidates": int(n), "accepted": int(accept_of(d["scores"]["A14 P"], d["a14_threshold"]).sum()), "threshold": d["a14_threshold"]}
                for arm in arms:
                    t = choices[f"{arm}_s{s}"]["threshold"]; acc = accept_of(d["scores"][arm], t); entries[arm] = routed(P, s, acc)
                    zstats[f"{arm} s{s} {h}"] = {"candidates": int(n), "accepted": int(acc.sum()), "threshold": t}
                for name, preds in entries.items():
                    systems.setdefault(f"{name} s{s}", {})[h] = rso.row_of(a8.cfg, preds, P["ev"])
                rso.check_stored(systems[f"Base s{s}"][h]["row"], rso.row_of(a8.cfg, d["base"], P["ev"])["row"], "no-op equals Base")
                rso.check_stored(systems[f"accept-all s{s}"][h]["row"], systems[f"Alternative s{s}"][h]["row"], "accept-all equals Alternative")
                rso.check_stored(systems[f"A14 P s{s}"][h]["row"], a14[f"P router s{s}"][h], f"A14 P s{s} {h} (A14 screen)")
                rso.check_stored(systems[f"Base s{s}"][h]["row"], a14[f"Base s{s}"][h], f"Base s{s} {h} (A14 screen)")
                rso.check_stored(systems[f"Alternative s{s}"][h]["row"], a14[f"Alternative s{s}"][h], f"Alternative s{s} {h} (A14 screen)")
        rows = {k: {h: v[h]["row"] for h in v} for k, v in systems.items()}; counts = {k: {h: rsh.counts_of(per[h]["seqs"], v[h]) for h in v} for k, v in systems.items()}
        cmp_, acc_ = {}, {}
        for h in HALVES:
            for s in seeds:
                draws = {}
                for c, (a, b) in CMPS.items():
                    if a not in arms + ["Base", "Alternative", "A14 P"] or b not in arms + ["Base", "Alternative", "A14 P"]:
                        continue
                    for nm in (a, b):
                        if nm not in draws:
                            draws[nm] = st.system_draws(systems[f"{nm} s{s}"][h]["tables"])
                    cmp_[f"s{s}|{c}|{h}"] = st.paired(draws[a], draws[b])
                    acc_[f"s{s}|{c}|{h}"] = st.segment_accounting(per[h]["seqs"], systems[f"{a} s{s}"][h]["preds"], systems[f"{b} s{s}"][h]["preds"])[0]
                print(f"{h} seed {s}: comparisons done", flush=True)
        res = {"stage": "select", "seeds": seeds, "config_sha256": sha256_file(ROOT / ctx.config), "choices_sha256": choices_sha, "systems": rows, "segment_counts": counts, "zones": zstats,
               "comparisons": cmp_, "accounting": acc_, "arms_evaluated": arms,
               "reproduced": ["Base equals A08 A + B3 (A08 summary, A14 screen)", "Alternative equals A13 H head (A13 screen, A14 screen)", "A14 P equals A14's P router rows",
                              "no-op equals Base", "accept-all equals Alternative"], "access": {"halves": list(HALVES), "time": started}}
        if seeds == [0, 2]:
            res["third_seed"] = third_seed(rows, counts, arms)
        ids = rw.ids_of(a8, {h: per[h]["seqs"] for h in HALVES}); flat = json.loads(json.dumps(res, default=str)); assert_no_ids(flat, ids, "select")
        for p in (ctx.out / "evaluation" / "screen.json", ctx.results / "screen.json"):
            p.parent.mkdir(parents=True, exist_ok=True); tl.write_json(p, flat)
        if "third_seed" in flat:
            tl.write_json(ctx.out / "evaluation" / "decision.json", {**flat["third_seed"], "time": tl.now()})
            print(json.dumps({k: {x: flat["third_seed"][k][x] for x in ("mean_dF", "half_means", "mean_dN", "mean_dBR", "missed_short_true", "passes")} for k in arms}, indent=1),
                  "train_seed1:", flat["third_seed"]["train_seed1"], flush=True)
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        tl.ledger_append(ctx.ledger, "evaluate select (CPU)", "cpu", started, time.time() - t0, ok, note)


# ----------------------------------------------------------------------------------------------- diagnostic
def zone_classes(du: np.ndarray, dtp: np.ndarray, dfp: np.ndarray) -> dict:
    ben = du > 1e-12
    return {"zones": int(du.size), "beneficial": int(ben.sum()), "zero": int((np.abs(du) <= 1e-12).sum()), "harmful": int((du < -1e-12).sum()),
            "beneficial_recovers_true": int((ben & (dtp > 0)).sum()), "beneficial_removes_false_only": int((ben & (dtp == 0) & (dfp < 0)).sum()), "beneficial_trades_match": int((ben & (dtp < 0)).sum()),
            "harmful_loses_match": int(((du < -1e-12) & (dtp < 0)).sum()), "harmful_adds_false_only": int(((du < -1e-12) & (dtp >= 0) & (dfp > 0)).sum())}


def auc(score: np.ndarray, y: np.ndarray):
    if not (y.any() and (~y).any()):
        return "unavailable"
    r = np.argsort(np.argsort(score, kind="stable"), kind="stable") + 1.0; npos = float(y.sum()); nneg = float((~y).sum())
    for v in np.unique(score):                                       # mid-ranks for ties
        k = score == v; r[k] = r[k].mean()
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def interaction(seq_index: np.ndarray, ben: np.ndarray, f1_after: np.ndarray, f1_base: dict, combined: dict) -> dict:
    """Sequences with two or more individually beneficial zones: does applying them together keep the best single gain?"""
    multi = [i for i in np.unique(seq_index[ben]) if int((ben & (seq_index == i)).sum()) >= 2]; below_best = below_base = 0
    for i in multi:
        best = float(f1_after[ben & (seq_index == i)].max())
        below_best += int(combined[int(i)] < best - 1e-12); below_base += int(combined[int(i)] < f1_base[int(i)] - 1e-12)
    return {"sequences_with_beneficial": int(np.unique(seq_index[ben]).size), "sequences_with_two_or_more": len(multi), "combined_below_best_single": below_best, "combined_below_base": below_base}


def diagnostic(ctx: trd.Ctx, seeds: list[int]) -> None:
    t0, started, ok, note = time.time(), tl.now(), False, ""
    try:
        if not (ctx.out / "evaluation" / "screen.json").exists():
            raise SystemExit("the diagnostic runs after the selection stage only")
        choices = json.loads((ctx.out / "evaluation" / "choices.json").read_text())["choices"]; a8 = rsh.a08_ctx(); per = {h: read_half(ctx, a8, h, seeds) for h in ("calibration", "selection")}
        out = {"note": "reference-assisted diagnostic after the selection stage; not a system and not used by any decision", "development": {}, "pools": {}}
        for h, P in per.items():
            for s in seeds:
                d = P["seed"][s]; c = d["cand"]; util = [None] * c["seq"].size
                for qi, q in enumerate(P["seqs"]):
                    sel = np.flatnonzero(c["seq"] == qi)
                    if sel.size:
                        for k, u in zip(sel, lr.zone_utilities(d["base"][q["id"]], d["alt"][q["id"]], q["labels"], [(int(c["a"][i]), int(c["b"][i])) for i in sel])):
                            util[k] = u
                du = np.asarray([u["f1_after"] - u["f1_base"] for u in util]); dtp = np.asarray([u["dtp"] for u in util]); dfp = np.asarray([u["dfp"] for u in util]); ben = du > 1e-12
                comb = routed(P, s, ben); row = rso.row_of(a8.cfg, comb, P["ev"]); base_row = rso.row_of(a8.cfg, d["base"], P["ev"])
                f1b = {qi: lr.seq_f1(d["base"][q["id"]], q["labels"])["f1"] for qi, q in enumerate(P["seqs"])}; f1c = {qi: lr.seq_f1(comb[q["id"]], q["labels"])["f1"] for qi, q in enumerate(P["seqs"])}
                rec = {"zones": zone_classes(du, dtp, dfp), "short_zones_beneficial": int((ben & ((c["b"] - c["a"]) <= 16)).sum()),
                       "all_beneficial_applied": {"row": row["row"], "counts": rsh.counts_of(P["seqs"], row), "base_counts": rsh.counts_of(P["seqs"], base_row)},
                       "sum_single_dtp_dfp": {"dtp": int(dtp[ben].sum()), "dfp": int(dfp[ben].sum())},
                       "interaction": interaction(c["seq"], ben, np.asarray([u["f1_after"] for u in util]), f1b, f1c), "routers": {}}
                for name, sc in d["scores"].items():
                    t = d["a14_threshold"] if name == "A14 P" else choices[f"{name}_s{s}"]["threshold"]; acc = accept_of(sc, t)
                    rec["routers"][name] = {"threshold": t, "accepted": int(acc.sum()), "accepted_beneficial": int((acc & ben).sum()), "beneficial_not_accepted": int((~acc & ben).sum()),
                                            "accepted_zero": int((acc & (np.abs(du) <= 1e-12)).sum()), "accepted_harmful": int((acc & (du < -1e-12)).sum()),
                                            "score_auc_beneficial_vs_rest": auc(sc, ben), "beneficial_score_median": float(np.median(sc[ben])) if ben.any() else None,
                                            "beneficial_at_or_above_0.5": int((ben & (sc >= 0.5)).sum())}
                out["development"][f"s{s} {h}"] = rec
        grp, kinds = trd.seq_meta(ctx)
        for s in seeds:
            c = trd.load_cand(ctx, s); pr = trd.load_preds(ctx, s); ben = c["du"] > 1e-12
            for g in ("large_fit", "small_fit", "check"):
                m = c[g]; members = sorted(set(np.flatnonzero((grp == ("check" if g == "check" else "fit"))).tolist())) if g != "small_fit" else None
                if g == "small_fit":
                    man = json.loads((ctx.out / "pool_manifest.json").read_text())["sequences"]; members = [i for i, q in enumerate(man) if q["small"]]
                f1b = {i: lr.seq_f1(pr["base"][i], pr["labels"][i])["f1"] for i in members}; comb = trd.routed_lists(c, pr, ben & m, members)
                f1c = {i: lr.seq_f1(comb[i], pr["labels"][i])["f1"] for i in members}; f1_after = c["du"] + np.asarray([f1b.get(int(i), 0.0) for i in c["seq"]])
                rec = {"zones": zone_classes(c["du"][m], c["dtp"][m], c["dfp"][m]), "pooled": {}}
                for kind in ("multi", "native"):
                    mem = [i for i in members if kinds[i] == kind]
                    rec["pooled"][kind] = {"base": trd.pooled_f1({i: pr["base"][i] for i in mem}, pr["labels"]), "alternative": trd.pooled_f1({i: pr["alt"][i] for i in mem}, pr["labels"]),
                                           "all_beneficial_applied": trd.pooled_f1({i: comb[i] for i in mem}, pr["labels"])}
                rec["sum_single_dtp_dfp"] = {"dtp": int(c["dtp"][m & ben].sum()), "dfp": int(c["dfp"][m & ben].sum())}
                rec["interaction"] = interaction(c["seq"][m], ben[m], f1_after[m], f1b, f1c); out["pools"][f"s{s} {g}"] = rec
        ids = rw.ids_of(a8, {h: per[h]["seqs"] for h in HALVES}); flat = json.loads(json.dumps(out, default=str)); assert_no_ids(flat, ids, "diagnostic")
        for p in (ctx.out / "evaluation" / "diagnostic.json", ctx.results / "diagnostic.json"):
            tl.write_json(p, flat)
        print(json.dumps({k: {"zones": v["zones"], "interaction": v["interaction"], "routers": {n: {x: r[x] for x in ("accepted_beneficial", "beneficial_not_accepted", "score_auc_beneficial_vs_rest")}
                                                                                                    for n, r in v["routers"].items()}} for k, v in flat["development"].items()}, indent=1), flush=True)
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        tl.ledger_append(ctx.ledger, "evaluate diagnostic (CPU)", "cpu", started, time.time() - t0, ok, note)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--stage", choices=("calibrate", "select", "diagnostic"), required=True); ap.add_argument("--choices-sha")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 2])
    a = ap.parse_args(); a.set = []; ctx = trd.Ctx(load_cfg(a), a.config)
    if a.stage == "calibrate":
        return calibrate(ctx, a.seeds)
    if a.stage == "select":
        return select(ctx, a.seeds, a.choices_sha)
    diagnostic(ctx, a.seeds)


if __name__ == "__main__":
    main()
