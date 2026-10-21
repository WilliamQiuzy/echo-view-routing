#!/usr/bin/env python
"""A19 evaluation (configs/experiments/video_b3_coverage_v1.yaml), run once after choices.json was committed: every model's
small and large choices, its old committed B3 choice and its seed's A08 setting (references) on the router-check pool and
both development halves; large - small, E-large - R-large, E-large - A08-large, E-large - Base and E-large - A14 P."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.b3_coverage as bc  # noqa: E402
import scripts.report_router_data as rrd  # noqa: E402
import scripts.report_router_trajectory as rrt  # noqa: E402
import scripts.report_segment_objective as rso  # noqa: E402
import scripts.report_video_switch_head as rsh  # noqa: E402
import scripts.report_video_within_view as rw  # noqa: E402
import scripts.segment_objective_tables as st  # noqa: E402
import scripts.train_router_trajectory as trt  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.ev9v_complementarity import decode  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402

import numpy as np  # noqa: E402

import scripts.report_center_supervision as rcs  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402

TARGETS = ("check", "selection", "calibration")
RULES = ("large", "small", "old", "A08 fixed")


def check_row(m: dict) -> dict:
    conf = np.asarray(m["native_confusion"]); return {**{k: m[k] for k in rcs.FLAT}, "A4C read as A5C": int(conf[2, 3]), "A5C read as A4C": int(conf[3, 2])}


def params_of(ch: dict) -> dict:
    out = {}
    for key in bc.KEYS:
        s = int(key.split("_s")[1]); out[key] = {"large": ch[f"{key}|large"]["params"], "small": ch[f"{key}|small"]["params"], "old": bc.old_params(key), "A08 fixed": bc.old_params(f"A08_s{s}")}
    return out


def comparisons_for(s: int, target: str) -> dict:
    c = {f"{g} large - small": (f"{g}_s{s} large", f"{g}_s{s} small") for g in ("A08", "R", "E")}
    c.update({"E large - R large": (f"E_s{s} large", f"R_s{s} large"), "E large - A08 large": (f"E_s{s} large", f"A08_s{s} large"),
              "E large - Base": (f"E_s{s} large", f"A08_s{s} old"), "E large - E old": (f"E_s{s} large", f"E_s{s} old")})
    if target != "check" and s in (0, 2):
        c["E large - A14 P"] = (f"E_s{s} large", f"A14 P s{s}")
    return c


def evaluate(ctx: bc.Ctx, choices_sha: str) -> None:
    t0, started, ok, note = time.time(), bc.now(), False, ""
    try:
        cp = ctx.out / "choices.json"
        if sha256_file(cp) != choices_sha:
            raise SystemExit("choices.json differs from the committed sha256; nothing is evaluated")
        ch = json.loads(cp.read_text())["choices"]; prm = params_of(ch); a8 = rsh.a08_ctx(); cfg = a8.cfg
        P = bc.with_index(ctx, bc.pool(ctx)); cidx = bc.members(P, "check"); ev_c = bc.evalset(P, cidx, "check")
        chk_seqs = [{"id": P["seqs"][i]["id"], "kind": P["seqs"][i]["kind"], "labels": np.asarray(P["seqs"][i]["labels"]), "refs": P["seqs"][i]["refs"]} for i in cidx]
        systems, seqs_of = {}, {"check": chk_seqs}
        for key in bc.KEYS:
            probs = bc.pool_probs(ctx, P, key); sub = {q["id"]: probs[q["id"]] for q in chk_seqs}
            for rule, pr in prm[key].items():
                if pr is None:
                    continue
                preds = decode(pr, sub); m, t = bc.evaluate_pool(cfg, preds, ev_c); systems.setdefault(f"{key} {rule}", {})["check"] = {"row": check_row(m), "tables": t, "preds": preds}
            print(f"check {key}: done", flush=True)
        a8sum = json.loads((ROOT / "docs/results/2026-10-16_video_within_view/summary.json").read_text()); a10 = json.loads((ROOT / "docs/results/2026-10-18_video_switch_head/screen.json").read_text())["systems"]
        a18 = json.loads((ROOT / "runs/video_ensemble_teacher_v1/evaluation/screen_three_seeds.json").read_text())["systems"]; a14 = json.loads(rrd.A14_SCREEN.read_text())["systems"]
        c17 = trt.Ctx(load_cfg(SimpleNamespace(config=bc.A17_CONFIG, set=[])), bc.A17_CONFIG); ch17 = json.loads((ROOT / "runs/video_router_trajectory_v1/evaluation/choices.json").read_text())["choices"]
        reproduced = []
        for h in bc.HALVES:
            D = bc.dev_probs(ctx, a8, h); seqs_of[h] = D["seqs"]
            for key in bc.KEYS:
                for rule, pr in prm[key].items():
                    if pr is not None:
                        systems.setdefault(f"{key} {rule}", {})[h] = rso.row_of(cfg, decode(pr, D["probs"][key]), D["ev"])
                g, s = key.split("_s"); want = a8sum["systems"][f"A + B3 s{s}"][h] if g == "A08" else (a10[f"R + B3 s{s}"][h] if g == "R" and s in ("0", "2") else a18[f"{g} + B3 s{s}"][h])
                rso.check_stored(systems[f"{key} old"][h]["row"], want, f"{key} old B3 {h}"); reproduced.append(f"{key} old B3 {h}")
            P17 = rrt.read_half(c17, a8, h, [0, 2])
            if [q["id"] for q in P17["seqs"]] != [q["id"] for q in D["seqs"]]:
                raise SystemExit("A17's half differs in order")
            for s in (0, 2):
                systems[f"A14 P s{s}"] = systems.get(f"A14 P s{s}", {}); systems[f"A14 P s{s}"][h] = rso.row_of(cfg, rrd.routed(P17, s, rrt.accept("A14 P", P17["seed"][s], ch17, rrt.a15_choices(), s)), D["ev"])
                rso.check_stored(systems[f"A14 P s{s}"][h]["row"], a14[f"P router s{s}"][h], f"A14 P s{s} {h}"); reproduced.append(f"A14 P s{s} {h}")
            print(f"{h}: rows done", flush=True)
        rows = {k: {t: v[t]["row"] for t in v} for k, v in systems.items()}; counts = {k: {t: rsh.counts_of(seqs_of[t], v[t]) for t in v} for k, v in systems.items()}
        cmp_, acc_ = {}, {}
        for target in TARGETS:
            for s in (0, 1, 2):
                draws = {}
                for c, (a, b) in comparisons_for(s, target).items():
                    if target not in systems.get(a, {}) or target not in systems.get(b, {}):
                        continue
                    for nm in (a, b):
                        if nm not in draws:
                            draws[nm] = st.system_draws(systems[nm][target]["tables"])
                    cmp_[f"s{s}|{c}|{target}"] = st.paired(draws[a], draws[b]); acc_[f"s{s}|{c}|{target}"] = st.segment_accounting(seqs_of[target], systems[a][target]["preds"], systems[b][target]["preds"])[0]
            print(f"{target}: comparisons done", flush=True)
        verdicts = {}
        for c in comparisons_for(1, "check"):
            for m in st.BOOT_METRICS:
                for target in TARGETS:
                    xs = [cmp_[f"s{s}|{c}|{target}"][m] for s in (0, 1, 2) if f"s{s}|{c}|{target}" in cmp_]
                    if len(xs) == 3:
                        verdicts[f"{c}|{m}|{target}"] = st.consistency(xs)
        res = {"stage": "evaluate", "config_sha256": sha256_file(ROOT / ctx.config), "choices_sha256": choices_sha, "systems": rows, "segment_counts": counts, "comparisons": cmp_, "accounting": acc_,
               "verdicts": verdicts, "candidate_criterion": criterion(rows, counts), "reproduced": reproduced, "params_applied": prm,
               "notes": {"thresholds": "numeric thresholds from the committed choices, never recomputed on check or development", "check": "router-check sessions unseen by the choice; recordings trained the video models"},
               "time": started}
        ids = rw.ids_of(a8, {h: seqs_of[h] for h in bc.HALVES}) | {q["id"] for q in chk_seqs if q["kind"] == "native"} | {r[0] for i in cidx for r in P["seqs"][i]["refs"]}
        flat = json.loads(json.dumps(res, default=str)); assert_no_ids(flat, ids, "evaluate")
        for p in (ctx.out / "screen.json", ctx.results / "screen.json"):
            bc.write_json(p, flat)
        print(json.dumps(flat["candidate_criterion"], indent=1), flush=True); ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        bc.ledger_append(ctx.ledger, "evaluate once (CPU)", "cpu", started, time.time() - t0, ok, note)


def criterion(rows: dict, counts: dict) -> dict:
    """E-large against Base (A08 + B3 at its committed setting) over the six development cells."""
    cells = [(s, h) for s in (0, 1, 2) for h in bc.HALVES]
    if not all(h in rows.get(f"E_s{s} large", {}) for s, h in cells):
        return {"status": "E-large infeasible in some seed; not evaluated"}
    F = lambda n, s, h: rows[n][h]["segment_f1_50_multi"]  # noqa: E731
    dF = {f"s{s} {h}": F(f"E_s{s} large", s, h) - F(f"A08_s{s} old", s, h) for s, h in cells}; half = {h: float(np.mean([dF[f"s{s} {h}"] for s in (0, 1, 2)])) for h in bc.HALVES}
    dN = float(np.mean([rows[f"E_s{s} large"][h]["native_macro_f1"] - rows[f"A08_s{s} old"][h]["native_macro_f1"] for s, h in cells]))
    dBR = float(np.mean([rows[f"E_s{s} large"][h]["boundary_recall_multi"] - rows[f"A08_s{s} old"][h]["boundary_recall_multi"] for s, h in cells]))
    fs = {"E large": int(sum(counts[f"E_s{s} large"][h]["fn_short"] for s, h in cells)), "Base": int(sum(counts[f"A08_s{s} old"][h]["fn_short"] for s, h in cells))}
    feas = {f"s{s}": bool(rows[f"E_s{s} large"]["calibration"]["native_macro_f1"] >= 0.980 and rows[f"E_s{s} large"]["calibration"]["boundary_recall_multi"] >= 0.95) for s in (0, 1, 2)}
    r = {"mean_dF": float(np.mean(list(dF.values()))), "dF": dF, "half_means": half, "mean_dN": dN, "mean_dBR": dBR, "missed_short_true": fs, "calibration_feasible": feas}
    r["passes"] = bool(r["mean_dF"] >= 0.003 and min(half.values()) >= -0.005 and min(dF.values()) >= -0.015 and dN >= -0.003 and dBR >= -0.010 and fs["E large"] <= fs["Base"] + 3 and all(feas.values()))
    return r
