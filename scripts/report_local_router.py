#!/usr/bin/env python
"""Evaluate A14 (configs/experiments/video_local_router_v1.yaml) on the development halves, from A13's verified frozen A08
caches: Base (A08 + B3), Alternative (A13 H), accept-all, and the P and V routers at one calibration choice each.

  --stage calibrate               the calibration half only; writes choices.json
  --stage select --choices-sha S  both halves after choices.json (sha256 S) has been committed locally
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
from scripts._common import load_cfg  # noqa: E402
from scripts.ev9v_complementarity import decode  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402

HALVES = ("selection", "calibration")
GRID = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99)
CMPS = {"P - A08 A": ("P router", "Base"), "V - A08 A": ("V router", "Base"), "P - A13 H": ("P router", "Alternative"), "V - A13 H": ("V router", "Alternative"), "V - P": ("V router", "P router")}


def router(ctx: tl.Ctx, kind: str, s: int):
    rec = json.loads((ctx.out / "routers" / f"{kind}_s{s}.json").read_text()); p = ctx.out / "routers" / f"{kind}_s{s}.pt"
    if sha256_file(p) != rec["sha256"]:
        raise SystemExit(f"{kind}_s{s}: router weights differ from their record")
    st_ = torch.load(p, map_location="cpu", weights_only=True); m = tl.build(kind, int(st_["d"]), s); m.load_state_dict(st_["model"]); m.eval()
    return m, {"mu": st_["mu"].numpy(), "sd": st_["sd"].numpy()}, rec["sha256"]


def read_half(ctx: tl.Ctx, a8, h: str, seeds: list[int]) -> dict:
    ev, seqs = a8.half(h, rso.DevAccess({h})); ids = [q["id"] for q in seqs]; refs = rw.half_refs(seqs); at = {r: i for i, r in enumerate(refs)}; W = rr.half_windows(seqs, at, 5)
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
        probs = rr.per_sample(W, lg, seqs); embs = tl.per_sample_emb(W, em, seqs); head, thr, _ = ctx.alt(s)
        edges = rvs.edge_scores(head, W, lg, em, seqs, torch.device("cpu")); base, alt = tl.systems(probs, edges, seqs, ctx.base_params(s), thr)
        cand = tl.candidates(seqs, probs, embs, base, alt, with_labels=False); scores = {}
        for kind in ("P", "V"):
            m, stats, _ = router(ctx, kind, s); scores[kind] = tl.score(m, cand["fp" if kind == "P" else "fv"], stats)
        out["seed"][s] = {"base": base, "alt": alt, "cand": cand, "scores": scores}
        rb, ra = rso.row_of(a8.cfg, base, ev)["row"], rso.row_of(a8.cfg, alt, ev)["row"]
        rso.check_stored(rb, a8sum["systems"][f"A + B3 s{s}"][h], f"Base s{s} {h} (A08 A + B3)"); rso.check_stored(ra, a13rows[f"H head s{s}"][h], f"Alternative s{s} {h} (A13 H)")
    _ = ids
    return out


def routed(P: dict, s: int, accept: np.ndarray) -> dict:
    d = P["seed"][s]; c = d["cand"]; out = {}
    for qi, q in enumerate(P["seqs"]):
        sel = np.flatnonzero(c["seq"] == qi); zs = [(int(c["a"][i]), int(c["b"][i])) for i in sel]
        out[q["id"]] = lr.apply(d["base"][q["id"]], d["alt"][q["id"]], zs, [bool(accept[i]) for i in sel]) if sel.size else d["base"][q["id"]].copy()
    return out


def calibrate(ctx: tl.Ctx, seeds: list[int]) -> None:
    t0, started, ok, note = time.time(), tl.now(), False, ""
    try:
        a8 = rsh.a08_ctx(); rw.check_grid(a8); P = read_half(ctx, a8, "calibration", seeds); choices = {}
        for s in seeds:
            for kind in ("P", "V"):
                sc = P["seed"][s]["scores"][kind]; recs = []
                for t in ["noop", *GRID]:
                    acc = np.zeros(sc.size, bool) if t == "noop" else sc >= t; m, _ = tl.rsh.rcs.evaluate(a8.cfg, routed(P, s, acc), P["ev"])
                    recs.append({"threshold": t, "accepted": int(acc.sum()), **{k: m[k] for k in tl.rsh.rcs.FLAT}})
                feas = [r for r in recs if r["native_macro_f1"] >= 0.980 and r["boundary_recall_multi"] >= 0.95]
                best = max(feas, key=lambda r: (r["segment_f1_50_multi"], -r["accepted"])) if feas else None
                choices[f"{kind}_s{s}"] = {"status": "feasible" if best else "infeasible", "threshold": best["threshold"] if best else "noop", "grid": recs}
                print(f"{kind}_s{s}: {choices[f'{kind}_s{s}']['status']} threshold {choices[f'{kind}_s{s}']['threshold']} (accepted {best['accepted'] if best else 0} of {sc.size})", flush=True)
        rec = {"stage": "calibrate", "seeds": seeds, "config_sha256": sha256_file(ROOT / ctx.config), "choices": choices, "routers": {f"{k}_s{s}": router(ctx, k, s)[2] for s in seeds for k in ("P", "V")},
               "time": tl.now()}
        flat = json.loads(json.dumps(rec, default=str)); assert_no_ids(flat, rw.ids_of(a8, {"calibration": P["seqs"]}), "choices")
        for p in (ctx.out / "evaluation" / "choices.json", ctx.results / "choices.json"):
            p.parent.mkdir(parents=True, exist_ok=True); tl.write_json(p, flat)
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        tl.ledger_append(ctx.ledger, "evaluate calibrate (CPU)", "cpu", started, time.time() - t0, ok, note)


def third_seed(rows: dict, counts: dict) -> dict:
    cells = [(s, h) for s in (0, 2) for h in HALVES]; out = {}
    for kind in ("P", "V"):
        x = f"{kind} router"; F = lambda n, s, h: rows[f"{n} s{s}"][h]["segment_f1_50_multi"]  # noqa: E731
        dF = {f"s{s} {h}": F(x, s, h) - F("Base", s, h) for s, h in cells}; half = {h: float(np.mean([dF[f"s{s} {h}"] for s in (0, 2)])) for h in HALVES}
        dN = float(np.mean([rows[f"{x} s{s}"][h]["native_macro_f1"] - rows[f"Base s{s}"][h]["native_macro_f1"] for s, h in cells]))
        dBR = float(np.mean([rows[f"{x} s{s}"][h]["boundary_recall_multi"] - rows[f"Base s{s}"][h]["boundary_recall_multi"] for s, h in cells]))
        fs = {n: int(sum(counts[f"{n} s{s}"][h]["fn_short"] for s, h in cells)) for n in (x, "Base")}
        r = {"mean_dF": float(np.mean(list(dF.values()))), "dF": dF, "half_means": half, "mean_dN": dN, "mean_dBR": dBR, "missed_short_true": fs}
        r["signal"] = bool(r["mean_dF"] >= 0.003); r["no_reversal"] = bool(min(half.values()) >= -0.005 and min(dF.values()) >= -0.015)
        r["no_harm"] = bool(dN >= -0.003 and dBR >= -0.010 and fs[x] <= fs["Base"] + 3); r["passes"] = bool(r["signal"] and r["no_reversal"] and r["no_harm"]); out[kind] = r
    out["train_seed1"] = bool(out["P"]["passes"] or out["V"]["passes"])
    return out


def select(ctx: tl.Ctx, seeds: list[int], choices_sha: str) -> None:
    t0, started, ok, note = time.time(), tl.now(), False, ""
    try:
        cp = ctx.out / "evaluation" / "choices.json"
        if sha256_file(cp) != choices_sha:
            raise SystemExit("choices.json differs from the committed sha256; selection is not read")
        choices = json.loads(cp.read_text())["choices"]; a8 = rsh.a08_ctx(); per = {h: read_half(ctx, a8, h, seeds) for h in ("calibration", "selection")}; systems, zstats = {}, {}
        for h in ("calibration", "selection"):
            P = per[h]
            for s in seeds:
                d = P["seed"][s]; n = d["scores"]["P"].size
                entries = {"Base": routed(P, s, np.zeros(n, bool)), "Alternative": d["alt"], "accept-all": routed(P, s, np.ones(n, bool))}
                for kind in ("P", "V"):
                    t = choices[f"{kind}_s{s}"]["threshold"]; acc = np.zeros(n, bool) if t == "noop" else d["scores"][kind] >= float(t); entries[f"{kind} router"] = routed(P, s, acc)
                    zstats[f"{kind} s{s} {h}"] = {"candidates": int(n), "accepted": int(acc.sum()), "threshold": t}
                for name, preds in entries.items():
                    systems.setdefault(f"{name} s{s}", {})[h] = rso.row_of(a8.cfg, preds, P["ev"])
                rso.check_stored(systems[f"Base s{s}"][h]["row"], rso.row_of(a8.cfg, d["base"], P["ev"])["row"], "no-op equals Base")
                rso.check_stored(systems[f"accept-all s{s}"][h]["row"], systems[f"Alternative s{s}"][h]["row"], "accept-all equals Alternative")
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
               "comparisons": cmp_, "accounting": acc_, "reproduced": ["Base equals A08 A + B3", "Alternative equals A13 H head", "no-op equals Base", "accept-all equals Alternative"],
               "access": {"halves": list(HALVES), "time": started}}
        if seeds == [0, 2]:
            res["third_seed"] = third_seed(rows, counts)
        ids = rw.ids_of(a8, {h: per[h]["seqs"] for h in HALVES}); flat = json.loads(json.dumps(res, default=str)); assert_no_ids(flat, ids, "select")
        for p in (ctx.out / "evaluation" / "screen.json", ctx.results / "screen.json"):
            p.parent.mkdir(parents=True, exist_ok=True); tl.write_json(p, flat)
        if "third_seed" in flat:
            tl.write_json(ctx.out / "evaluation" / "decision.json", {**flat["third_seed"], "time": tl.now()})
            print(json.dumps({k: {x: flat["third_seed"][k][x] for x in ("mean_dF", "half_means", "mean_dN", "mean_dBR", "missed_short_true", "passes")} for k in ("P", "V")}, indent=1),
                  "train_seed1:", flat["third_seed"]["train_seed1"], flush=True)
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        tl.ledger_append(ctx.ledger, "evaluate select (CPU)", "cpu", started, time.time() - t0, ok, note)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--stage", choices=("calibrate", "select"), required=True); ap.add_argument("--choices-sha"); ap.add_argument("--seeds", nargs="+", type=int, default=[0, 2])
    a = ap.parse_args(); a.set = []; ctx = tl.Ctx(load_cfg(a), a.config)
    if a.stage == "calibrate":
        return calibrate(ctx, a.seeds)
    select(ctx, a.seeds, a.choices_sha)


if __name__ == "__main__":
    main()
