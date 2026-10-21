#!/usr/bin/env python
"""Evaluate A17 (configs/experiments/video_router_trajectory_v1.yaml) on the development halves from A13's verified frozen
A08 caches: Base, Alternative, accept-all, A14 P, A15 large-P and A16 C-fixed / U-fixed (reproduced), and the ordered (O)
and shuffled (S) trajectory routers under the fixed rule (score > 0.5) and one calibration choice; then, after selection
only, the reference-assisted zone diagnostic and the read-only A16 supplement.

  --stage calibrate               calibration half: fixed rows (evaluated only) and cal choices; writes choices.json
  --stage select --choices-sha S  both halves after choices.json (sha256 S) has been committed locally
  --stage diagnostic              after select: accepted zones by class and kind, net counts, the A16 supplement
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
import scripts.report_router_data as rrd  # noqa: E402
import scripts.report_segment_objective as rso  # noqa: E402
import scripts.report_video_switch_head as rsh  # noqa: E402
import scripts.report_video_within_view as rw  # noqa: E402
import scripts.segment_objective_tables as st  # noqa: E402
import scripts.train_local_router as tl  # noqa: E402
import scripts.train_router_data as trd  # noqa: E402
import scripts.train_router_trajectory as trt  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.temporal.segments import labels_to_segments  # noqa: E402

HALVES = ("selection", "calibration")
SET_OF = {"selection": trt.SET_SELECTION, "calibration": trt.SET_CALIBRATION}
RULED = ("O-fixed", "O-cal", "S-fixed", "S-cal")
OLD = ("Base", "Alternative", "accept-all", "A14 P", "A15 large-P", "A16 C-fixed", "A16 U-fixed")
FIXED_RULE = "accept when sigmoid(score) > 0.5 (equality rejects)"
CAL_RULE = "A15 grid 0.50-0.99 and no-op, score >= threshold, calibration native macro-F1 >= 0.980 and multi boundary recall >= 0.95, highest multi F1, ties to fewer zones"
CMPS = {"O-fixed - S-fixed": ("O-fixed", "S-fixed"), "O-cal - S-cal": ("O-cal", "S-cal"), "O-fixed - O-cal": ("O-fixed", "O-cal"), "S-fixed - S-cal": ("S-fixed", "S-cal"),
        **{f"{x} - {b}": (x, b) for x in RULED for b in ("Base", "Alternative", "A14 P", "A16 C-fixed", "A15 large-P")}}
A16 = ROOT / "runs/video_router_utility_v1"


def u16(s: int):
    rec = json.loads((A16 / "routers" / f"U_s{s}.json").read_text()); p = A16 / "routers" / f"U_s{s}.pt"
    if sha256_file(p) != rec["sha256"]:
        raise SystemExit(f"A16 U_s{s}: weights differ from their record")
    st_ = torch.load(p, map_location="cpu", weights_only=True); m = tl.build("P", int(st_["d"]), s); m.load_state_dict(st_["model"]); m.eval()
    return m, {"mu": st_["mu"].numpy(), "sd": st_["sd"].numpy()}


def read_half(ctx: trt.Ctx, a8, h: str, seeds: list[int]) -> dict:
    """A15's verified half reader, plus the per-sample probabilities from the same A13 cache, the O/S trajectory tensors
    and the A16 U and A17 O/S scores."""
    P = rrd.read_half(ctx.a15, a8, h, seeds); seqs = P["seqs"]; refs = rw.half_refs(seqs); at = {r: i for i, r in enumerate(refs)}; W = rr.half_windows(seqs, at, 5)
    for s in seeds:
        d = P["seed"][s]
        with np.load(ROOT / "runs/video_frozen_switch_v1/evaluation" / h / f"a08_s{s}.npz") as z:
            if not (np.array_equal(z["seq"], W["seq"]) and np.array_equal(z["centre"], W["centre"])):
                raise SystemExit("A13's cache does not match this half's windows")
            probs = rr.per_sample(W, z["logits"].astype(np.float32), seqs)
        c = d["cand"]; cand = trd.to_arrays(trd.candidates_p(seqs, probs, d["base"], d["alt"], with_labels=False))
        if not (np.array_equal(cand["fp"], c["fp"]) and np.array_equal(cand["a"], c["a"])):
            raise SystemExit("the rebuilt development features differ from the half reader's")
        T = trt.tensors(c, [probs[q["id"]] for q in seqs], [d["base"][q["id"]] for q in seqs], [d["alt"][q["id"]] for q in seqs], s, SET_OF[h]); d["T"] = T
        m, stats = u16(s); d["scores"]["A16 U"] = tl.score(m, c["fp"], stats)
        for arm in trt.ARMS:
            mt, mu, sd, _ = trt.load_router(ctx, arm, s); Xn = ((c["fp"].astype(np.float32) - mu) / sd).astype(np.float32); d["scores"][arm] = trt.score(mt, T, arm, Xn)
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
    if name == "A16 C-fixed":
        return d["scores"]["large-P"] > 0.5
    if name == "A16 U-fixed":
        return d["scores"]["A16 U"] > 0.5
    arm, rule = name.split("-")
    return d["scores"][arm] > 0.5 if rule == "fixed" else rrd.accept_of(d["scores"][arm], choices[f"{arm}_s{s}"]["threshold"])


def a15_choices() -> dict:
    return json.loads((ROOT / "runs/video_router_data_v1/evaluation/choices.json").read_text())["choices"]


def calibrate(ctx: trt.Ctx, seeds: list[int]) -> None:
    t0, started, ok, note = time.time(), tl.now(), False, ""
    try:
        trt.verify_pins(ctx); a8 = rsh.a08_ctx(); rw.check_grid(a8); P = read_half(ctx, a8, "calibration", seeds); choices, fixed_rows = {}, {}
        for s in seeds:
            d = P["seed"][s]
            for arm in trt.ARMS:
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
        rec = {"stage": "calibrate", "seeds": seeds, "config_sha256": sha256_file(ROOT / ctx.config), "rules": {"fixed": FIXED_RULE, "cal": CAL_RULE}, "fixed_calibration_rows": fixed_rows,
               "choices": choices, "routers": {f"{arm}_s{s}": trt.load_router(ctx, arm, s)[3] for s in seeds for arm in trt.ARMS},
               "references": {"A14 P": {f"s{s}": P["seed"][s]["a14_threshold"] for s in seeds}, "A15 large-P": {f"s{s}": a15_choices()[f"large-P_s{s}"]["threshold"] for s in seeds},
                              "A16 C-fixed / U-fixed": "score > 0.5 on A15 large-P / A16 U"}, "time": tl.now()}
        flat = json.loads(json.dumps(rec, default=str)); assert_no_ids(flat, rw.ids_of(a8, {"calibration": P["seqs"]}), "choices")
        for p in (ctx.out / "evaluation" / "choices.json", ctx.results / "choices.json"):
            p.parent.mkdir(parents=True, exist_ok=True); tl.write_json(p, flat)
        log, ck, tol = {}, trd.check_kinds(ctx.a15), trt.tolerance()       # written after the choices; nothing is chosen from it
        for s in seeds:
            c = trd.load_cand(ctx.a15, s); pr = trd.load_preds(ctx.a15, s); T = trt.load_t(ctx, s)
            for arm in trt.ARMS:
                m, mu, sd, _ = trt.load_router(ctx, arm, s); Xn = ((c["fp"].astype(np.float32) - mu) / sd).astype(np.float32); sc = trt.score(m, T, arm, Xn)
                log[f"{arm}-fixed_s{s}"] = trt.check_log(c, pr, sc > 0.5, ck, tol); t = choices[f"{arm}_s{s}"]["threshold"]
                log[f"{arm}-cal_s{s}"] = {"threshold": t, **trt.check_log(c, pr, rrd.accept_of(sc, t), ck, tol)}
        for p in (ctx.out / "evaluation" / "check_log.json", ctx.results / "check_log.json"):
            tl.write_json(p, json.loads(json.dumps({"note": "router-check logging on A15's check pool (read before, router-unseen); never used for a choice", "log": log}, default=str)))
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        tl.ledger_append(ctx.ledger, "evaluate calibrate (CPU)", "cpu", started, time.time() - t0, ok, note)


def promotion(rows: dict, counts: dict, ch: dict) -> dict:
    """The fixed promotion rule for O-fixed and O-cal; S rows are reported with the same quantities and never qualify."""
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
        r["meets_rule"] = bool(r["signal"] and r["no_reversal"] and r["no_harm"] and r["feasibility"]); r["can_qualify"] = arm == "O"; out[x] = r
    out["train_seed1"] = bool(out["O-fixed"]["meets_rule"] or out["O-cal"]["meets_rule"])
    return out


def select(ctx: trt.Ctx, seeds: list[int], choices_sha: str) -> None:
    t0, started, ok, note = time.time(), tl.now(), False, ""
    try:
        cp = ctx.out / "evaluation" / "choices.json"
        if sha256_file(cp) != choices_sha:
            raise SystemExit("choices.json differs from the committed sha256; selection is not read")
        ch = json.loads(cp.read_text()); choices = ch["choices"]; a15ch = a15_choices(); a8 = rsh.a08_ctx(); per = {h: read_half(ctx, a8, h, seeds) for h in ("calibration", "selection")}
        a14 = json.loads(rrd.A14_SCREEN.read_text())["systems"]; a15 = json.loads((ROOT / "runs/video_router_data_v1/evaluation/screen.json").read_text())["systems"]
        a16 = json.loads((A16 / "evaluation/screen.json").read_text())["systems"]; systems, zstats = {}, {}
        for h in ("calibration", "selection"):
            P = per[h]
            for s in seeds:
                d = P["seed"][s]
                for name in (*OLD, *RULED):
                    preds = d["alt"] if name == "Alternative" else rrd.routed(P, s, accept(name, d, choices, a15ch, s)); systems.setdefault(f"{name} s{s}", {})[h] = rso.row_of(a8.cfg, preds, P["ev"])
                    if name not in ("Base", "Alternative", "accept-all"):
                        acc = accept(name, d, choices, a15ch, s); zstats[f"{name} s{s} {h}"] = {"candidates": int(acc.size), "accepted": int(acc.sum())}
                row = lambda n: systems[f"{n} s{s}"][h]["row"]  # noqa: E731
                rso.check_stored(row("Base"), rso.row_of(a8.cfg, d["base"], P["ev"])["row"], "no-op equals Base"); rso.check_stored(row("accept-all"), row("Alternative"), "accept-all equals Alternative")
                rso.check_stored(row("Base"), a14[f"Base s{s}"][h], f"Base s{s} {h} (A14 screen)"); rso.check_stored(row("Alternative"), a14[f"Alternative s{s}"][h], f"Alternative s{s} {h} (A14 screen)")
                rso.check_stored(row("A14 P"), a14[f"P router s{s}"][h], f"A14 P s{s} {h}"); rso.check_stored(row("A15 large-P"), a15[f"large-P s{s}"][h], f"A15 large-P s{s} {h}")
                rso.check_stored(row("A16 C-fixed"), a16[f"C-fixed s{s}"][h], f"A16 C-fixed s{s} {h}"); rso.check_stored(row("A16 U-fixed"), a16[f"U-fixed s{s}"][h], f"A16 U-fixed s{s} {h}")
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
               "comparisons": cmp_, "accounting": acc_,
               "reproduced": ["Base equals A08 A + B3", "Alternative equals A13 H", "A14 P, A15 large-P, A16 C-fixed and A16 U-fixed equal their stored rows", "no-op equals Base",
                              "accept-all equals Alternative"], "access": {"halves": list(HALVES), "time": started}}
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


def a16_supplement(P: dict, choices16: dict) -> dict:
    """Seed-2 selection: the short reference segments matched by A16 C-fixed but not by A16 U-fixed; each U-accepted zone
    of that sequence undone singly, recording whether the specific match returns."""
    s, d = 2, P["seed"][2]; c = d["cand"]; accU, accC = d["scores"]["A16 U"] > 0.5, d["scores"]["large-P"] > 0.5; rows = []
    for qi, q in enumerate(P["seqs"]):
        if q["kind"] != "multi":
            continue
        sel = np.flatnonzero(c["seq"] == qi)
        if not sel.size:
            continue
        zs = [(int(c["a"][k]), int(c["b"][k])) for k in sel]; ref = labels_to_segments(np.asarray(q["labels"]))
        pu = lr.apply(d["base"][q["id"]], d["alt"][q["id"]], zs, [bool(accU[k]) for k in sel]); pc = lr.apply(d["base"][q["id"]], d["alt"][q["id"]], zs, [bool(accC[k]) for k in sel])
        mu_, mc_ = st.match_pairs(labels_to_segments(pu), ref), st.match_pairs(labels_to_segments(pc), ref); ru, rc = {j for _, j in mu_}, {j for _, j in mc_}
        for j in sorted(rc - ru):
            if ref[j].length > st.SHORT:
                continue
            rec = {"segment_length": int(ref[j].length), "u_accepted_zones": int(accU[sel].sum()), "c_accepted_zones": int(accC[sel].sum()), "undo": []}
            cand_iou = [p for p in labels_to_segments(pu) if p.label == ref[j].label and min(p.end, ref[j].end) - max(p.start, ref[j].start) > 0]
            rec["u_has_same_class_overlap"] = bool(cand_iou)
            for k in sel[accU[sel]]:
                acc2 = [bool(accU[x]) and x != k for x in sel]; p2 = lr.apply(d["base"][q["id"]], d["alt"][q["id"]], zs, acc2); m2 = st.match_pairs(labels_to_segments(p2), ref)
                rec["undo"].append({"zone_length": int(c["b"][k] - c["a"][k]), "zone_in_c": bool(accC[k]), "restores_match": j in {jj for _, jj in m2}, "matches_after": len(m2), "matches_u": len(mu_)})
            n_restore = sum(u["restores_match"] for u in rec["undo"])
            rec["reading"] = ("single-zone damage" if n_restore >= 1 and all(u["matches_after"] >= u["matches_u"] for u in rec["undo"] if u["restores_match"]) else
                              "match substitution" if n_restore >= 1 else "combination effect or unknown (no single undo restores it)")
            rows.append(rec)
    return {"lost_short_segments": len(rows), "segments": rows}


def diagnostic(ctx: trt.Ctx, seeds: list[int]) -> None:
    t0, started, ok, note = time.time(), tl.now(), False, ""
    try:
        if not (ctx.out / "evaluation" / "screen.json").exists():
            raise SystemExit("the diagnostic runs after the selection stage only")
        ch = json.loads((ctx.out / "evaluation" / "choices.json").read_text()); choices = ch["choices"]; a15ch = a15_choices(); a8 = rsh.a08_ctx()
        per = {h: read_half(ctx, a8, h, seeds) for h in ("calibration", "selection")}
        out = {"note": "reference-assisted diagnostic after the selection stage; not a system and not used by any decision", "development": {}}
        for h, P in per.items():
            for s in seeds:
                d = P["seed"][s]; c = d["cand"]; du = np.zeros(c["seq"].size)
                for qi, q in enumerate(P["seqs"]):
                    sel = np.flatnonzero(c["seq"] == qi)
                    if sel.size:
                        for k, u in zip(sel, lr.zone_utilities(d["base"][q["id"]], d["alt"][q["id"]], q["labels"], [(int(c["a"][i]), int(c["b"][i])) for i in sel])):
                            du[k] = u["f1_after"] - u["f1_base"]
                kind = np.asarray([P["seqs"][int(i)]["kind"] for i in c["seq"]]) if c["seq"].size else np.zeros(0, str); ben, harm = du > 1e-12, du < -1e-12; zero = ~ben & ~harm
                rec = {"zones": {k: {"zones": int((kind == k).sum()), "beneficial": int((ben & (kind == k)).sum()), "zero": int((zero & (kind == k)).sum()), "harmful": int((harm & (kind == k)).sum())}
                                 for k in st.KINDS}, "systems": {}}
                for name in ("A14 P", "A15 large-P", "A16 C-fixed", "A16 U-fixed", *RULED):
                    acc = accept(name, d, choices, a15ch, s); preds = rrd.routed(P, s, acc)
                    rec["systems"][name] = {k: {"accepted": int((acc & (kind == k)).sum()), "beneficial": int((acc & ben & (kind == k)).sum()), "zero": int((acc & zero & (kind == k)).sum()),
                                                "harmful": int((acc & harm & (kind == k)).sum()), **kind_net(P, preds, d["base"], k)} for k in st.KINDS}
                rec["auc_beneficial_vs_rest"] = {n: rrd.auc(d["scores"][n], ben) for n in ("O", "S", "large-P", "A14 P", "A16 U")}
                out["development"][f"s{s} {h}"] = rec
        out["A16_supplement"] = a16_supplement(per["selection"], json.loads((A16 / "evaluation/choices.json").read_text()))
        ids = rw.ids_of(a8, {h: per[h]["seqs"] for h in HALVES}); flat = json.loads(json.dumps(out, default=str)); assert_no_ids(flat, ids, "diagnostic")
        for p in (ctx.out / "evaluation" / "diagnostic.json", ctx.results / "diagnostic.json"):
            tl.write_json(p, flat)
        print(json.dumps(flat["A16_supplement"], indent=1), flush=True); ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        tl.ledger_append(ctx.ledger, "evaluate diagnostic (CPU)", "cpu", started, time.time() - t0, ok, note)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--stage", choices=("calibrate", "select", "diagnostic"), required=True); ap.add_argument("--choices-sha")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 2])
    a = ap.parse_args(); a.set = []; ctx = trt.Ctx(load_cfg(a), a.config)
    if a.stage == "calibrate":
        return calibrate(ctx, a.seeds)
    if a.stage == "select":
        return select(ctx, a.seeds, a.choices_sha)
    diagnostic(ctx, a.seeds)


if __name__ == "__main__":
    main()
