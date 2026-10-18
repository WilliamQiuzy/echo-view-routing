#!/usr/bin/env python
"""Evaluate A09 checkpoints on both development halves (configs/experiments/video_discriminative_v1.yaml).

The evaluation path is A08's: A03's half reader (one half at a time), V01's windows, inference and per-sample mapping,
the inherited B3 grid and A08's choice rule (rcs.b3_choose, once per checkpoint on calibration). Frozen Ordered (V01's
saved outputs and stored setting) and A08 A (its saved outputs and committed choices) are the references; both are
reproduced against their stored rows before use. Development halves are read for exploratory feedback, as the round
allows; every stage's access, models and decoder choices are recorded.

  --stage pilot   the pilot snapshots of pilot.queue: rows and the frozen pilot rule's decision (pilot.json)
  --stage final   every finished run of the round (final.json)
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.prepare_raw_video_context as prv  # noqa: E402
import scripts.report_center_supervision as rcs  # noqa: E402
import scripts.report_raw_video_context as rr  # noqa: E402
import scripts.report_segment_objective as rso  # noqa: E402
import scripts.report_video_within_view as rw  # noqa: E402
import scripts.segment_objective_tables as st  # noqa: E402
import scripts.train_video_discriminative as td  # noqa: E402
import scripts.trained_token_readout as ttr  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.ev9v_complementarity import decode  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402
from scripts.train_joint_visual_temporal import FramePaths  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402

HALVES = ("selection", "calibration")
PARITY_WINDOWS = 64
NAMES = {"plain": "C", "contrast": "D"}
CMPS = {"D - C": ("D", "C"), "C - frozen Ordered": ("C", "frozen Ordered"), "D - frozen Ordered": ("D", "frozen Ordered"),
        "C - A08 A": ("C", "A08 A"), "D - A08 A": ("D", "A08 A")}


def a08_ctx(ctx: td.Ctx) -> rw.Ctx:
    p = ctx.src["a08_config"]; td.pinned(p)
    return rw.Ctx(load_cfg(SimpleNamespace(config=p["path"], set=[])), p["path"])


def models_of(ctx: td.Ctx, stage: str) -> dict:
    out = {}
    if stage == "pilot":
        n = int(ctx.vd["pilot"]["updates"])
        for key in ctx.vd["pilot"]["queue"]:
            rec = json.loads((ctx.out / "train" / key / f"pilot_u{n}.json").read_text())
            out[key] = {"weights": rec["weights"], "sha256": rec["sha256"], "identity": f"{rec['identity']}@u{n}", "lambda": rec["lambda"]}
    else:
        for d in sorted((ctx.out / "train").iterdir()):
            man = d / "run_manifest.json"
            if man.is_file():
                m = json.loads(man.read_text()); fz = ROOT / "models_frozen" / td.NS / m["key"] / "final.pt"
                w = fz if fz.is_file() else Path(m["final_checkpoint"])
                out[m["key"]] = {"weights": str(w), "sha256": m["final_sha256"], "identity": m["identity"], "lambda": m["lambda"]}
    return out


def rows_of(cfg: dict, probs: dict, ev, params: dict, oparams: dict | None) -> dict:
    """Full evaluator outputs (row, tables, preds): B3 at params, argmax and, when given, B3 at the Ordered setting."""
    out = {"b3": rso.row_of(cfg, decode(params, probs), ev), "argmax": rso.row_of(cfg, {q: p.argmax(1) for q, p in probs.items()}, ev)}
    if oparams is not None:
        out["ordered_setting"] = rso.row_of(cfg, decode(oparams, probs), ev)
    return out


def compare(systems: dict, seqs: dict, seeds: list[int], names: set) -> dict:
    """Paired bootstrap of the B3 systems (the inherited draws), three-seed verdicts and segment accounting."""
    out = {"comparisons": {}, "verdicts": {}, "accounting": {}}
    for h in HALVES:
        for s in seeds:
            draws = {}
            for c, (a, b) in CMPS.items():
                if a not in names or b not in names:
                    continue
                for n in (a, b):
                    if n not in draws:
                        draws[n] = st.system_draws(systems[f"{n} + B3 s{s}"][h]["tables"])
                out["comparisons"][f"s{s}|{c}|{h}"] = st.paired(draws[a], draws[b])
                acc, _ = st.segment_accounting(seqs[h], systems[f"{a} + B3 s{s}"][h]["preds"], systems[f"{b} + B3 s{s}"][h]["preds"])
                out["accounting"][f"s{s}|{c}|{h}"] = acc
            print(f"{h} seed {s}: comparisons done", flush=True)
    for c, (a, b) in CMPS.items():
        if a in names and b in names:
            for m in st.BOOT_METRICS:
                out["verdicts"][f"{c}|{m}"] = {h: st.consistency([out["comparisons"][f"s{s}|{c}|{h}"][m] for s in seeds]) for h in HALVES}
    return out


def read_half(ctx: td.Ctx, a8: rw.Ctx, h: str, models: dict, stage: str, device, workers: int, seeds: list[int]) -> dict:
    ev, seqs = a8.half(h, rso.DevAccess({h})); ids = [q["id"] for q in seqs]; refs = rw.half_refs(seqs); at = {r: i for i, r in enumerate(refs)}
    fp = FramePaths(prv.frame_dirs(a8.paths, a8.rv, {r[0] for r in refs})); t0 = time.time(); frames = rr.decode_frames(refs, fp, workers)
    W = rr.half_windows(seqs, at, 5); orders = np.broadcast_to(np.arange(16, dtype=np.int64), W["s"].shape).copy()
    v01 = a8.v01_summary()["systems"]; a8sum = json.loads((ROOT / "docs/results/2026-10-16_video_within_view/summary.json").read_text())   # committed (f4f2358)
    a8ch = json.loads(td.pinned(ctx.src["a08_choices"]).read_text())["choices"]
    out = {"ev": ev, "seqs": seqs, "probs": {}, "refs": {}, "oparams": {}, "a08params": {}, "parity": {}, "reproduced": [], "gpu_seconds": 0.0,
           "counts": {"sequences": len(seqs), "windows": int(len(W["seq"])), "frames": len(refs), "decode_seconds": time.time() - t0}}
    k = PARITY_WINDOWS
    for s in seeds:
        saved = a8.v01_output(h, s); ttr.check_windows(W, ids, saved, f"ordered_s{s} {h}"); op = v01[f"ordered_s{s}"]["b3_choice"]["params"]; out["oparams"][s] = op
        if h == "calibration":
            state, _ = a8.source_state(s); src = td.make_model(state, device).eval(); t1 = time.time()
            with torch.no_grad():
                lg, _ = rr.model_logits(src, {"ref": W["ref"][:k], "marker": W["marker"][:k]}, orders[:k], frames, device)
            out["gpu_seconds"] += time.time() - t1; del src
            out["parity"][f"s{s}"] = td.vso.parity(lg, saved["logits"][:k], 1e-4)
            if not out["parity"][f"s{s}"]["ok"]:
                raise SystemExit(f"seed {s}: the V01 source parity failed ({out['parity'][f's{s}']})")
        P = rr.per_sample(W, saved["logits"].astype(np.float32), seqs); rows = ttr.cls_rows(a8.cfg, P, ev, op)
        for row in ("argmax", "b3"):
            ttr.check_stored(rows[row]["row"], v01[f"ordered_s{s}"][row][h], f"ordered_s{s} {row} {h} (V01)")
        out["refs"][("Ordered", s)] = P
        apath = ROOT / "runs/video_within_view_v1/evaluation" / h / f"all_s{s}.npz"
        if sha256_file(apath) != ctx.src["a08_all_outputs"][h][s]:
            raise SystemExit(f"A08 all_s{s} {h}: the saved outputs differ from this round's pin")
        lgA, _ = rr.cached_logits(apath, a8sum["runs"][f"all_s{s}"]["identity"], W, orders, seqs)
        if lgA is None:
            raise SystemExit(f"A08 all_s{s} {h}: the saved outputs do not match their identity or this half's windows")
        PA = rr.per_sample(W, lgA, seqs); ap = a8ch[f"all_s{s}"]["params"]; out["a08params"][s] = ap; out["refs"][("A08 A", s)] = PA
        got = rso.row_of(a8.cfg, decode(ap, PA), ev)["row"]; ttr.check_stored(got, a8sum["systems"][f"A + B3 s{s}"][h], f"A08 A + B3 s{s} {h}")
        out["reproduced"] += [f"frozen Ordered s{s} argmax and B3 {h}", f"A08 A s{s} B3 {h}"]
    for key, mr in models.items():
        path = ctx.out / "evaluation" / stage / h / f"{key}.npz"; lg, sec = rr.cached_logits(path, mr["identity"], W, orders, seqs)
        if lg is None:
            m = rw.load_frozen(Path(mr["weights"]), mr["sha256"], device); lg, sec = rr.model_logits(m, W, orders, frames, device); del m
            probs = rr.per_sample(W, lg, seqs); rr.save_outputs(path, mr["identity"], W, orders, lg, probs, seqs, sec); out["gpu_seconds"] += float(sec)
        else:
            probs = rr.per_sample(W, lg, seqs)
        out["probs"][key] = probs; torch.cuda.empty_cache(); print(f"{h} {key}: outputs ready", flush=True)
    del frames
    return out


# ----------------------------------------------------------------------------------------------- the frozen pilot rule
def cell_values(row_arg: dict, row_b3: dict) -> dict:
    return {"N": float(row_arg["native_macro_f1"]), "E45": int(row_arg["A4C read as A5C"]), "E54": int(row_arg["A5C read as A4C"]),
            "F": float(row_b3["segment_f1_50_multi"]), "BR": float(row_b3["boundary_recall_multi"])}


def signal(x: dict, r: dict) -> dict:
    """The configuration's pilot rule for candidate cells x against reference cells r (same seeds and halves)."""
    cells = sorted(x); dN = [x[c]["N"] - r[c]["N"] for c in cells]; dF = [x[c]["F"] - r[c]["F"] for c in cells]; dBR = [x[c]["BR"] - r[c]["BR"] for c in cells]
    E45x, E54x = sum(x[c]["E45"] for c in cells), sum(x[c]["E54"] for c in cells); E45r, E54r = sum(r[c]["E45"] for c in cells), sum(r[c]["E54"] for c in cells)
    dE45, dE54 = E45x - E45r, E54x - E54r; dE, ER = dE45 + dE54, E45r + E54r
    nat_a = bool(np.mean(dN) >= 0.003 and sum(d >= 0 for d in dN) >= 3 and dE <= 0)
    nat_b = bool(dE <= -max(2, math.ceil(0.15 * ER)) and max(dE45, dE54) <= 1)
    seg = bool(np.mean(dF) >= 0.003 and sum(d >= 0 for d in dF) >= 3)
    guard = bool(np.mean(dN) >= -0.005 and np.mean(dF) >= -0.005 and np.mean(dBR) >= -0.010 and min(dF) >= -0.015 and min(dN) >= -0.015 and dE <= 2)
    return {"cells": cells, "dN": dN, "dF": dF, "dBR": dBR, "mean_dN": float(np.mean(dN)), "mean_dF": float(np.mean(dF)), "mean_dBR": float(np.mean(dBR)),
            "E45": [E45x, E45r], "E54": [E54x, E54r], "dE45": dE45, "dE54": dE54, "dE": dE, "native_signal_a": nat_a, "native_signal_b": nat_b,
            "native_signal": nat_a or nat_b, "segment_signal": seg, "max_degradation_ok": guard, "signal": bool((nat_a or nat_b or seg) and guard)}


def pilot_decision(sysrows: dict, seeds: list[int]) -> dict:
    def cells(name: str) -> dict:
        return {f"s{s} {h}": cell_values(sysrows[f"{name} argmax s{s}"][h], sysrows[f"{name} + B3 s{s}"][h]) for s in seeds for h in HALVES}
    C, D, O, A = cells("C"), cells("D"), cells("frozen Ordered"), cells("A08 A")
    dc, co = signal(D, C), signal(C, O)
    if dc["signal"]:
        dec = "promote_both: resume C and D at seeds 0 and 2 to 3,750 and train the seed-1 pair"
    elif co["signal"]:
        dec = "promote_C: resume C at seeds 0 and 2 to 3,750 and train C seed 1; D stops"
    else:
        dec = "stop_both"
    return {"D vs C": dc, "C vs frozen Ordered": co, "described_only": {"D vs frozen Ordered": signal(D, O), "D vs A08 A": signal(D, A), "C vs A08 A": signal(C, A)},
            "decision": dec, "cells": {"C": C, "D": D, "frozen Ordered": O, "A08 A": A}}


# ----------------------------------------------------------------------------------------------- stage
def run(ctx: td.Ctx, stage: str, workers: int) -> None:
    t0, started, ok, note, gsec = time.time(), td.now(), False, "", 0.0
    try:
        a8 = a08_ctx(ctx); rw.check_grid(a8); models = models_of(ctx, stage); device = torch.device("cuda"); td.determinism()
        torch.cuda.set_per_process_memory_fraction(0.6); seeds = [int(x) for x in ctx.vd["pilot"]["seeds"]] if stage == "pilot" else ctx.seeds
        per = {}
        for h in ("calibration", "selection"):
            per[h] = read_half(ctx, a8, h, models, stage, device, workers, seeds); gsec += per[h]["gpu_seconds"]
            if h == "calibration":
                choices = {}
                for key in models:
                    ch = rcs.b3_choose(a8.cfg, rso.DIAG, per[h]["probs"][key], per[h]["ev"]); choices[key] = {x: ch[x] for x in ("status", "params", "n_feasible")}
                    print(f"{key}: {ch['status']} {ch['params']}", flush=True)
                cdir = ctx.out / "evaluation" / stage; cdir.mkdir(parents=True, exist_ok=True)
                td.write_json(cdir / "choices.json", {"stage": stage, "models": {k: {x: v[x] for x in ("sha256", "identity", "lambda")} for k, v in models.items()},
                                                      "choices": choices, "time": td.now()})
        systems, names = {}, {"frozen Ordered", "A08 A"}
        for h in ("calibration", "selection"):
            P = per[h]
            for s in seeds:
                op = P["oparams"][s]
                for name, probs, params in (("frozen Ordered", P["refs"][("Ordered", s)], op), ("A08 A", P["refs"][("A08 A", s)], P["a08params"][s])):
                    r = rows_of(a8.cfg, probs, P["ev"], params, None)
                    systems.setdefault(f"{name} + B3 s{s}", {})[h] = r["b3"]; systems.setdefault(f"{name} argmax s{s}", {})[h] = r["argmax"]
                for key in models:
                    arm, ks = key.rsplit("_s", 1)
                    if int(ks) != s:
                        continue
                    n = NAMES[arm]; names.add(n); r = rows_of(a8.cfg, P["probs"][key], P["ev"], choices[key]["params"], op)
                    systems.setdefault(f"{n} + B3 s{s}", {})[h] = r["b3"]; systems.setdefault(f"{n} argmax s{s}", {})[h] = r["argmax"]
                    systems.setdefault(f"{n} at Ordered setting s{s}", {})[h] = r["ordered_setting"]
        sysrows = {k: {h: v[h]["row"] for h in v} for k, v in systems.items()}
        counts = {k: {h: st.kind_counts(per[h]["seqs"], v[h]["preds"], v[h]["tables"]) for h in v} for k, v in systems.items() if " + B3 " in k}
        res = {"stage": stage, "config_sha256": sha256_file(ROOT / ctx.config), "models": {k: {x: v[x] for x in ("sha256", "lambda")} for k, v in models.items()},
               "choices": choices, "systems": sysrows, "segment_counts": counts, "parity": per["calibration"]["parity"], "reproduced": {h: per[h]["reproduced"] for h in HALVES},
               "half_counts": {h: per[h]["counts"] for h in HALVES}, "access": {"halves": list(HALVES), "time": started}}
        if stage == "pilot":
            res["rule"] = pilot_decision(sysrows, seeds)
        cmp = compare(systems, {h: per[h]["seqs"] for h in HALVES}, seeds, names) if stage == "final" else None
        if cmp is not None:
            res.update(cmp)
        elif stage == "pilot":                                    # described only: segment accounting at the pilot point
            res["accounting"] = {f"s{s}|{c}|{h}": st.segment_accounting(per[h]["seqs"], systems[f"{a} + B3 s{s}"][h]["preds"], systems[f"{b} + B3 s{s}"][h]["preds"])[0]
                                 for s in seeds for h in HALVES for c, (a, b) in CMPS.items() if a in names and b in names}
        ids = rw.ids_of(a8, {h: per[h]["seqs"] for h in HALVES}); flat = json.loads(json.dumps(res, default=str)); assert_no_ids(flat, ids, stage)
        for p in (ctx.out / "evaluation" / stage / f"{stage}.json", ctx.results / f"{stage}.json"):
            p.parent.mkdir(parents=True, exist_ok=True); td.write_json(p, flat)
        if stage == "pilot":
            r = res["rule"]; print(json.dumps({k: {x: r[k][x] for x in ("mean_dN", "mean_dF", "mean_dBR", "dE45", "dE54", "native_signal", "segment_signal", "max_degradation_ok", "signal")}
                                               for k in ("D vs C", "C vs frozen Ordered")}, indent=1), "\nDECISION:", r["decision"], flush=True)
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        total = time.time() - t0; g = min(gsec, total)
        td.ledger_append(ctx.ledger, f"evaluate {stage} (GPU inference)", "gpu", started, g, ok, note)
        td.ledger_append(ctx.ledger, f"evaluate {stage} (CPU)", "cpu", started, total - g, ok, note)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--stage", choices=("pilot", "final"), required=True); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(); a.set = []; run(td.Ctx(load_cfg(a), a.config), a.stage, a.workers)


if __name__ == "__main__":
    main()
