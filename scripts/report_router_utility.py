#!/usr/bin/env python
"""Evaluate A16 (configs/experiments/video_router_utility_v1.yaml) on the development halves from A13's verified frozen A08
caches: Base, Alternative, accept-all, A14 P and A15 large-P at their committed choices (reproduced), and C and U each
under the fixed rule (sigmoid score > 0.5) and under one calibration choice; then, after selection only, the
reference-assisted utility diagnostic (never a system).

  --stage calibrate               calibration half: fixed rows (evaluated only) and cal choices; writes choices.json
  --stage select --choices-sha S  both halves after choices.json (sha256 S) has been committed locally
  --stage diagnostic              after select: accepted utility and net counts by native / pairs / multi
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
import scripts.report_router_data as rrd  # noqa: E402
import scripts.report_segment_objective as rso  # noqa: E402
import scripts.report_video_switch_head as rsh  # noqa: E402
import scripts.report_video_within_view as rw  # noqa: E402
import scripts.segment_objective_tables as st  # noqa: E402
import scripts.train_local_router as tl  # noqa: E402
import scripts.train_router_data as trd  # noqa: E402
import scripts.train_router_utility as tru  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402

HALVES = ("selection", "calibration")
RULED = ("C-fixed", "C-cal", "U-fixed", "U-cal")
FIXED_RULE = "accept when sigmoid(score) > 0.5 (equality rejects)"
CAL_RULE = "A15 grid 0.50-0.99 and no-op, score >= threshold, calibration native macro-F1 >= 0.980 and multi boundary recall >= 0.95, highest multi F1, ties to fewer zones"
CMPS = {"U-fixed - C-fixed": ("U-fixed", "C-fixed"), "U-cal - C-cal": ("U-cal", "C-cal"), "U-fixed - U-cal": ("U-fixed", "U-cal"), "C-fixed - C-cal": ("C-fixed", "C-cal"),
        **{f"{x} - {b}": (x, b) for x in RULED for b in ("Base", "Alternative", "A14 P")}}


def router16(ctx: tru.Ctx, arm: str, s: int):
    rec = json.loads((ctx.out / "routers" / f"{arm}_s{s}.json").read_text()); p = tru.router_path(ctx, arm, s)
    if sha256_file(p) != rec["sha256"]:
        raise SystemExit(f"{arm}_s{s}: router weights differ from their record")
    st_ = torch.load(p, map_location="cpu", weights_only=True); m = tl.build("P", int(st_["d"]), s); m.load_state_dict(st_["model"]); m.eval()
    return m, {"mu": st_["mu"].numpy(), "sd": st_["sd"].numpy()}, rec["sha256"]


def read_half(ctx: tru.Ctx, a8, h: str, seeds: list[int]) -> dict:
    """A15's verified half reader (A13 caches, A08 parity, Base and Alternative stored rows, A14 P and A15 scores), plus C/U."""
    P = rrd.read_half(ctx.a15, a8, h, seeds); P["parity"] = {}
    for s in seeds:
        d = P["seed"][s]
        for arm in tru.ARMS:
            m, stats, _ = router16(ctx, arm, s); d["scores"][arm] = tl.score(m, d["cand"]["fp"], stats)
        P["parity"][s] = float(np.abs(d["scores"]["C"] - d["scores"]["large-P"]).max(initial=0.0))
    return P


def accept(name: str, d: dict, choices: dict, a15ch: dict, s: int) -> np.ndarray:
    n = d["cand"]["seq"].size
    if name == "Base":
        return np.zeros(n, bool)
    if name == "accept-all":
        return np.ones(n, bool)
    if name == "A14 P":
        return rrd.accept_of(d["scores"]["A14 P"], d["a14_threshold"])
    if name == "A15 large-P":
        return rrd.accept_of(d["scores"]["large-P"], a15ch[f"large-P_s{s}"]["threshold"])
    arm, rule = name.split("-")
    return d["scores"][arm] > 0.5 if rule == "fixed" else rrd.accept_of(d["scores"][arm], choices[f"{arm}_s{s}"]["threshold"])


def a15_choices() -> dict:
    return json.loads((ROOT / "runs/video_router_data_v1/evaluation/choices.json").read_text())["choices"]


def calibrate(ctx: tru.Ctx, seeds: list[int]) -> None:
    t0, started, ok, note = time.time(), tl.now(), False, ""
    try:
        if sha256_file(ROOT / "runs/video_router_data_v1/evaluation/choices.json") != ctx.vu["pinned_from_A15"]["choices_sha256"]:
            raise SystemExit("A15's choices differ from their pin")
        a8 = rsh.a08_ctx(); rw.check_grid(a8); P = read_half(ctx, a8, "calibration", seeds); choices, fixed_rows, a15ch = {}, {}, a15_choices()
        for s in seeds:
            d = P["seed"][s]
            for arm in tru.ARMS:
                sc = d["scores"][arm]; acc = sc > 0.5; m, _ = tl.rsh.rcs.evaluate(a8.cfg, rrd.routed(P, s, acc), P["ev"])
                fixed_rows[f"{arm}_s{s}"] = {"accepted": int(acc.sum()), "candidates": int(sc.size), **{k: m[k] for k in tl.rsh.rcs.FLAT},
                                             "feasible": bool(m["native_macro_f1"] >= 0.980 and m["boundary_recall_multi"] >= 0.95)}
                recs = []
                for t in ["noop", *rrd.GRID]:
                    a = rrd.accept_of(sc, t); mm, _ = tl.rsh.rcs.evaluate(a8.cfg, rrd.routed(P, s, a), P["ev"])
                    recs.append({"threshold": t, "accepted": int(a.sum()), **{k: mm[k] for k in tl.rsh.rcs.FLAT}})
                feas = [r for r in recs if r["native_macro_f1"] >= 0.980 and r["boundary_recall_multi"] >= 0.95]
                best = max(feas, key=lambda r: (r["segment_f1_50_multi"], -r["accepted"])) if feas else None
                choices[f"{arm}_s{s}"] = {"status": "feasible" if best else "infeasible", "threshold": best["threshold"] if best else "noop", "grid": recs}
                print(f"{arm}_s{s}: fixed accepts {fixed_rows[f'{arm}_s{s}']['accepted']} of {sc.size} (feasible {fixed_rows[f'{arm}_s{s}']['feasible']}); "
                      f"cal {choices[f'{arm}_s{s}']['status']} threshold {choices[f'{arm}_s{s}']['threshold']}", flush=True)
        flat15 = json.loads(json.dumps(a15ch, default=str)); flat16 = json.loads(json.dumps(choices, default=str))
        repro = {f"s{s}": {"threshold_equal": flat16[f"C_s{s}"]["threshold"] == flat15[f"large-P_s{s}"]["threshold"], "grid_equal": flat16[f"C_s{s}"]["grid"] == flat15[f"large-P_s{s}"]["grid"],
                           "max_abs_score_diff_vs_A15_large_P": P["parity"][s]} for s in seeds}
        rec = {"stage": "calibrate", "seeds": seeds, "config_sha256": sha256_file(ROOT / ctx.config), "rules": {"fixed": FIXED_RULE, "cal": CAL_RULE},
               "fixed_calibration_rows": fixed_rows, "choices": choices, "C_cal_reproduces_A15_large_P": repro,
               "routers": {f"{arm}_s{s}": router16(ctx, arm, s)[2] for s in seeds for arm in tru.ARMS}, "A14_P_committed": {f"P_s{s}": P["seed"][s]["a14_threshold"] for s in seeds},
               "A15_large_P_committed": {f"large-P_s{s}": a15ch[f"large-P_s{s}"]["threshold"] for s in seeds}, "time": tl.now()}
        flat = json.loads(json.dumps(rec, default=str)); assert_no_ids(flat, rw.ids_of(a8, {"calibration": P["seqs"]}), "choices")
        for p in (ctx.out / "evaluation" / "choices.json", ctx.results / "choices.json"):
            p.parent.mkdir(parents=True, exist_ok=True); tl.write_json(p, flat)
        log, ck = {}, trd.check_kinds(ctx.a15)                          # written after the choices; nothing is chosen from it
        for s in seeds:
            c = trd.load_cand(ctx.a15, s); pr = trd.load_preds(ctx.a15, s); U = tru.load_u(ctx, s)
            log[f"Base_s{s}"] = tru.edit_log(c, pr, U["u"], np.zeros(c["seq"].size, bool), ck); log[f"Alternative_s{s}"] = tru.edit_log(c, pr, U["u"], np.ones(c["seq"].size, bool), ck)
            for arm in tru.ARMS:
                m, stats, _ = router16(ctx, arm, s); sc = tl.score(m, c["fp"], stats)
                log[f"{arm}-fixed_s{s}"] = tru.edit_log(c, pr, U["u"], sc > 0.5, ck); log[f"{arm}-cal_s{s}"] = {"threshold": choices[f"{arm}_s{s}"]["threshold"],
                                                                                                                  **tru.edit_log(c, pr, U["u"], rrd.accept_of(sc, choices[f"{arm}_s{s}"]["threshold"]), ck)}
        for p in (ctx.out / "evaluation" / "check_log.json", ctx.results / "check_log.json"):
            tl.write_json(p, json.loads(json.dumps({"note": "router-check logging on A15's session-disjoint check pool (read before, router-unseen); never used for a choice", "log": log}, default=str)))
        print(json.dumps(repro), flush=True); ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        tl.ledger_append(ctx.ledger, "evaluate calibrate (CPU)", "cpu", started, time.time() - t0, ok, note)


def promotion(rows: dict, counts: dict, ch: dict) -> dict:
    """The fixed third-seed rule for U-fixed and U-cal (C rows reported with the same quantities, never qualifying)."""
    cells = [(s, h) for s in (0, 2) for h in HALVES]; out = {}; F = lambda n, s, h: rows[f"{n} s{s}"][h]["segment_f1_50_multi"]  # noqa: E731
    for x in RULED:
        dF = {f"s{s} {h}": F(x, s, h) - F("Base", s, h) for s, h in cells}; half = {h: float(np.mean([dF[f"s{s} {h}"] for s in (0, 2)])) for h in HALVES}
        dN = float(np.mean([rows[f"{x} s{s}"][h]["native_macro_f1"] - rows[f"Base s{s}"][h]["native_macro_f1"] for s, h in cells]))
        dBR = float(np.mean([rows[f"{x} s{s}"][h]["boundary_recall_multi"] - rows[f"Base s{s}"][h]["boundary_recall_multi"] for s, h in cells]))
        fs = {n: int(sum(counts[f"{n} s{s}"][h]["fn_short"] for s, h in cells)) for n in (x, "Base")}; arm, rule = x.split("-")
        feas = {f"s{s}": (ch["fixed_calibration_rows"][f"{arm}_s{s}"]["feasible"] if rule == "fixed" else ch["choices"][f"{arm}_s{s}"]["status"] == "feasible") for s in (0, 2)}
        r = {"mean_dF": float(np.mean(list(dF.values()))), "dF": dF, "half_means": half, "mean_dN": dN, "mean_dBR": dBR, "missed_short_true": fs, "calibration_feasible": feas}
        r["signal"] = bool(r["mean_dF"] >= 0.003); r["no_reversal"] = bool(min(half.values()) >= -0.005 and min(dF.values()) >= -0.015)
        r["no_harm"] = bool(dN >= -0.003 and dBR >= -0.010 and fs[x] <= fs["Base"] + 3); r["feasibility"] = bool(all(feas.values()))
        r["meets_rule"] = bool(r["signal"] and r["no_reversal"] and r["no_harm"] and r["feasibility"]); r["can_qualify"] = arm == "U"; out[x] = r
    out["train_seed1"] = bool(out["U-fixed"]["meets_rule"] or out["U-cal"]["meets_rule"])
    return out


def select(ctx: tru.Ctx, seeds: list[int], choices_sha: str) -> None:
    t0, started, ok, note = time.time(), tl.now(), False, ""
    try:
        cp = ctx.out / "evaluation" / "choices.json"
        if sha256_file(cp) != choices_sha:
            raise SystemExit("choices.json differs from the committed sha256; selection is not read")
        ch = json.loads(cp.read_text()); choices = ch["choices"]; a15ch = a15_choices(); a8 = rsh.a08_ctx(); per = {h: read_half(ctx, a8, h, seeds) for h in ("calibration", "selection")}
        a14 = json.loads(rrd.A14_SCREEN.read_text())["systems"]; a15 = json.loads((ROOT / "runs/video_router_data_v1/evaluation/screen.json").read_text())["systems"]; systems, zstats, same = {}, {}, {}
        names = ("Base", "Alternative", "accept-all", "A14 P", "A15 large-P", *RULED)
        for h in ("calibration", "selection"):
            P = per[h]
            for s in seeds:
                d = P["seed"][s]
                for name in names:
                    preds = d["alt"] if name == "Alternative" else rrd.routed(P, s, accept(name, d, choices, a15ch, s)); systems.setdefault(f"{name} s{s}", {})[h] = rso.row_of(a8.cfg, preds, P["ev"])
                    if name not in ("Base", "Alternative", "accept-all"):
                        acc = accept(name, d, choices, a15ch, s); zstats[f"{name} s{s} {h}"] = {"candidates": int(acc.size), "accepted": int(acc.sum())}
                row = lambda n: systems[f"{n} s{s}"][h]["row"]  # noqa: E731
                rso.check_stored(row("Base"), rso.row_of(a8.cfg, d["base"], P["ev"])["row"], "no-op equals Base"); rso.check_stored(row("accept-all"), row("Alternative"), "accept-all equals Alternative")
                rso.check_stored(row("Base"), a14[f"Base s{s}"][h], f"Base s{s} {h} (A14 screen)"); rso.check_stored(row("Alternative"), a14[f"Alternative s{s}"][h], f"Alternative s{s} {h} (A14 screen)")
                rso.check_stored(row("A14 P"), a14[f"P router s{s}"][h], f"A14 P s{s} {h} (A14 screen)"); rso.check_stored(row("A15 large-P"), a15[f"large-P s{s}"][h], f"A15 large-P s{s} {h} (A15 screen)")
                same[f"s{s} {h}"] = {"C-cal equals A15 large-P (row)": row("C-cal") == row("A15 large-P"),
                                     "C-cal equals A15 large-P (predictions)": all(np.array_equal(systems[f"C-cal s{s}"][h]["preds"][q["id"]], systems[f"A15 large-P s{s}"][h]["preds"][q["id"]]) for q in P["seqs"])}
        rows = {k: {h: v[h]["row"] for h in v} for k, v in systems.items()}; counts = {k: {h: rsh.counts_of(per[h]["seqs"], v[h]) for h in v} for k, v in systems.items()}
        cmp_, acc_ = {}, {}
        for h in HALVES:
            for s in seeds:
                draws = {}
                for c, (a, b) in CMPS.items():
                    for nm in (a, b):
                        if nm not in draws:
                            draws[nm] = st.system_draws(systems[f"{nm} s{s}"][h]["tables"])
                    cmp_[f"s{s}|{c}|{h}"] = st.paired(draws[a], draws[b])
                    acc_[f"s{s}|{c}|{h}"] = st.segment_accounting(per[h]["seqs"], systems[f"{a} s{s}"][h]["preds"], systems[f"{b} s{s}"][h]["preds"])[0]
                print(f"{h} seed {s}: comparisons done", flush=True)
        res = {"stage": "select", "seeds": seeds, "config_sha256": sha256_file(ROOT / ctx.config), "choices_sha256": choices_sha, "systems": rows, "segment_counts": counts, "zones": zstats,
               "comparisons": cmp_, "accounting": acc_, "C_cal_vs_A15_large_P": same, "dev_score_parity_C_vs_A15_large_P": {f"s{s}": {h: per[h]["parity"][s] for h in HALVES} for s in seeds},
               "reproduced": ["Base equals A08 A + B3 (A08 summary, A14 screen)", "Alternative equals A13 H (A13 and A14 screens)", "A14 P equals A14's rows", "A15 large-P equals A15's rows",
                              "no-op equals Base", "accept-all equals Alternative"], "access": {"halves": list(HALVES), "time": started}}
        if seeds == [0, 2]:
            res["promotion"] = promotion(rows, counts, ch)
        ids = rw.ids_of(a8, {h: per[h]["seqs"] for h in HALVES}); flat = json.loads(json.dumps(res, default=str)); assert_no_ids(flat, ids, "select")
        for p in (ctx.out / "evaluation" / "screen.json", ctx.results / "screen.json"):
            p.parent.mkdir(parents=True, exist_ok=True); tl.write_json(p, flat)
        if "promotion" in flat:
            tl.write_json(ctx.out / "evaluation" / "decision.json", {**flat["promotion"], "time": tl.now()})
            print(json.dumps({k: {x: flat["promotion"][k][x] for x in ("mean_dF", "half_means", "mean_dN", "mean_dBR", "missed_short_true", "calibration_feasible", "meets_rule")} for k in RULED}, indent=1),
                  "train_seed1:", flat["promotion"]["train_seed1"], flush=True)
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        tl.ledger_append(ctx.ledger, "evaluate select (CPU)", "cpu", started, time.time() - t0, ok, note)


# ----------------------------------------------------------------------------------------------- diagnostic
def kind_net(P: dict, preds: dict, base: dict, kind: str) -> dict:
    tp = pred = tp0 = pred0 = 0
    for q in P["seqs"]:
        if q["kind"] == kind:
            r, r0 = lr.seq_f1(preds[q["id"]], q["labels"]), lr.seq_f1(base[q["id"]], q["labels"]); tp, pred, tp0, pred0 = tp + r["tp"], pred + r["pred"], tp0 + r0["tp"], pred0 + r0["pred"]
    return {"net_dTP": tp - tp0, "net_dFP": (pred - tp) - (pred0 - tp0)}


def diagnostic(ctx: tru.Ctx, seeds: list[int]) -> None:
    t0, started, ok, note = time.time(), tl.now(), False, ""
    try:
        if not (ctx.out / "evaluation" / "screen.json").exists():
            raise SystemExit("the diagnostic runs after the selection stage only")
        ch = json.loads((ctx.out / "evaluation" / "choices.json").read_text()); choices = ch["choices"]; a15ch = a15_choices(); prep = json.loads((ctx.out / "preparation.json").read_text())
        a8 = rsh.a08_ctx(); per = {h: read_half(ctx, a8, h, seeds) for h in ("calibration", "selection")}
        out = {"note": "reference-assisted diagnostic after the selection stage; u uses each seed's training F_fit; not a system and not used by any decision", "development": {}}
        for h, P in per.items():
            for s in seeds:
                d = P["seed"][s]; c = d["cand"]; Ff = prep["seeds"][str(s)]["F_fit"]["F_fit"]; dtp = np.zeros(c["seq"].size, int); dfp = np.zeros(c["seq"].size, int)
                for qi, q in enumerate(P["seqs"]):
                    sel = np.flatnonzero(c["seq"] == qi)
                    if sel.size:
                        for k, u in zip(sel, lr.zone_utilities(d["base"][q["id"]], d["alt"][q["id"]], q["labels"], [(int(c["a"][i]), int(c["b"][i])) for i in sel])):
                            dtp[k], dfp[k] = u["dtp"], u["dfp"]
                u = np.where((dtp == 0) & (dfp == 0), 0.0, 2.0 * dtp - Ff * (dtp + dfp)); kind = np.asarray([P["seqs"][int(i)]["kind"] for i in c["seq"]]) if c["seq"].size else np.zeros(0, str)
                rec = {"zones": {k: {"zones": int((kind == k).sum()), "u_positive": int(((kind == k) & (u > 0)).sum()), "u_zero": int(((kind == k) & (u == 0)).sum()),
                                     "u_negative": int(((kind == k) & (u < 0)).sum()), "u_sum_positive": float(u[(kind == k) & (u > 0)].sum()), "u_sum_negative": float(u[(kind == k) & (u < 0)].sum())}
                                 for k in st.KINDS}, "systems": {}}
                for name in ("A14 P", "A15 large-P", *RULED):
                    acc = accept(name, d, choices, a15ch, s); preds = rrd.routed(P, s, acc); r = {}
                    for k in st.KINDS:
                        m = acc & (kind == k)
                        r[k] = {"accepted": int(m.sum()), "u_positive": int((m & (u > 0)).sum()), "u_zero": int((m & (u == 0)).sum()), "u_negative": int((m & (u < 0)).sum()),
                                "u_sum_positive": float(u[m & (u > 0)].sum()), "u_sum_negative": float(u[m & (u < 0)].sum()), "single_dTP_sum": int(dtp[m].sum()), "single_dFP_sum": int(dfp[m].sum()),
                                **kind_net(P, preds, d["base"], k)}
                    rec["systems"][name] = r
                out["development"][f"s{s} {h}"] = rec
        ids = rw.ids_of(a8, {h: per[h]["seqs"] for h in HALVES}); flat = json.loads(json.dumps(out, default=str)); assert_no_ids(flat, ids, "diagnostic")
        for p in (ctx.out / "evaluation" / "diagnostic.json", ctx.results / "diagnostic.json"):
            tl.write_json(p, flat)
        for k, v in flat["development"].items():
            print(k, json.dumps({n: {kk: (x["accepted"], x["u_positive"], x["u_negative"], round(x["u_sum_positive"], 2), round(x["u_sum_negative"], 2), x["net_dTP"], x["net_dFP"])
                                     for kk, x in r.items() if x["accepted"]} for n, r in v["systems"].items()}), flush=True)
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        tl.ledger_append(ctx.ledger, "evaluate diagnostic (CPU)", "cpu", started, time.time() - t0, ok, note)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--stage", choices=("calibrate", "select", "diagnostic"), required=True); ap.add_argument("--choices-sha")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 2])
    a = ap.parse_args(); a.set = []; ctx = tru.Ctx(load_cfg(a), a.config)
    if a.stage == "calibrate":
        return calibrate(ctx, a.seeds)
    if a.stage == "select":
        return select(ctx, a.seeds, a.choices_sha)
    diagnostic(ctx, a.seeds)


if __name__ == "__main__":
    main()
