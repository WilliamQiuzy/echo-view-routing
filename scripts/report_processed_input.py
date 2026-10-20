#!/usr/bin/env python
"""Evaluate A12 (configs/experiments/video_processed_input_v1.yaml) on both development halves: stage one (frozen A08 on Pro
pixels; original from A08's saved outputs) and the screen (trained Raw and Pro, cross-input sensitivities), with full-task
rows (fixed fallback) and common-support rows (sequences without a fallback window).

  --stage screen       seeds 0 and 2 (stage one included), both fixed rules (decision.json)
  --stage three_seed   seeds 0, 1 and 2
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.prepare_raw_video_context as prv  # noqa: E402
import scripts.processed_input as pi  # noqa: E402
import scripts.report_center_supervision as rcs  # noqa: E402
import scripts.report_raw_video_context as rr  # noqa: E402
import scripts.report_segment_objective as rso  # noqa: E402
import scripts.report_video_switch_head as rsh  # noqa: E402
import scripts.report_video_within_view as rw  # noqa: E402
import scripts.segment_objective_tables as st  # noqa: E402
import scripts.train_processed_input as tp  # noqa: E402
import scripts.trained_token_readout as ttr  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.ev9v_complementarity import decode  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402
from scripts.train_joint_visual_temporal import FramePaths  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402

HALVES = ("selection", "calibration")
CMPS = {"Pro - Raw": ("Pro", "Raw"), "Pro - A08 A": ("Pro", "A08 A"), "Raw - A08 A": ("Raw", "A08 A"), "A08 on Pro (A08 setting) - A08 A": ("A08 on Pro fixed", "A08 A"),
        "A08 on Pro (recalibrated) - A08 A": ("A08 on Pro", "A08 A"), "Raw on Pro - Raw": ("Raw on Pro", "Raw"), "Pro on original - Pro": ("Pro on original", "Pro")}


def subset(ev, keep: set):
    f = lambda xs, key: tuple(x for x in xs if key(x) in keep)  # noqa: E731
    parts = dict(natives=f(ev.natives, lambda x: x.seq_id), pairs=f(ev.pairs, lambda x: x.recipe.recipe_id), multi=f(ev.multi, lambda x: x.recipe.recipe_id))
    return dataclasses.replace(ev, **parts) if dataclasses.is_dataclass(ev) else ev._replace(**parts)


def read_half(ctx: tp.Ctx, a8, h: str, trained: dict, seeds: list[int], device, workers: int) -> dict:
    ev, seqs = a8.half(h, rso.DevAccess({h})); ids = [q["id"] for q in seqs]; refs = rw.half_refs(seqs); at = {r: i for i, r in enumerate(refs)}
    W = rr.half_windows(seqs, at, 5); orders = np.broadcast_to(np.arange(16, dtype=np.int64), W["s"].shape).copy(); cache = ctx.cache()
    supported = {q["id"] for q in seqs if all(cache.is_pro(str(r[0])) for r in q["refs"])}
    v01 = a8.v01_summary()["systems"]; a8sum = json.loads((ROOT / "docs/results/2026-10-16_video_within_view/summary.json").read_text())
    a8ch = json.loads(tp.vsh.pinned(ctx.a10.src["a08_choices"]).read_text())["choices"]; pins = rsh.a08_output_pins()
    out = {"ev": ev, "seqs": seqs, "probs": {}, "refs": {}, "oparams": {}, "a08params": {}, "parity": {}, "gpu_seconds": 0.0, "supported": supported,
           "counts": {"sequences": len(seqs), "supported_sequences": len(supported), "windows": int(len(W["seq"])),
                      "fallback_videos": len({str(r[0]) for r in refs if not cache.is_pro(str(r[0]))}), "videos": len({str(r[0]) for r in refs})}}
    for s in seeds:
        saved = a8.v01_output(h, s); ttr.check_windows(W, ids, saved, f"ordered_s{s} {h}"); op = v01[f"ordered_s{s}"]["b3_choice"]["params"]; out["oparams"][s] = op
        P = rr.per_sample(W, saved["logits"].astype(np.float32), seqs); rows = ttr.cls_rows(a8.cfg, P, ev, op)
        for row in ("argmax", "b3"):
            ttr.check_stored(rows[row]["row"], v01[f"ordered_s{s}"][row][h], f"ordered_s{s} {row} {h} (V01)")
        out["refs"][("frozen Ordered", s)] = P; apath = ROOT / "runs/video_within_view_v1/evaluation" / h / f"all_s{s}.npz"
        if sha256_file(apath) != pins[h][s]:
            raise SystemExit("A08 saved outputs differ from their pin")
        lgA, _ = rr.cached_logits(apath, a8sum["runs"][f"all_s{s}"]["identity"], W, orders, seqs)
        if lgA is None:
            raise SystemExit("A08 saved outputs do not match")
        PA = rr.per_sample(W, lgA, seqs); ap = a8ch[f"all_s{s}"]["params"]; out["a08params"][s] = ap; out["refs"][("A08 A", s)] = PA
        ttr.check_stored(rso.row_of(a8.cfg, decode(ap, PA), ev)["row"], a8sum["systems"][f"A + B3 s{s}"][h], f"A08 A + B3 s{s} {h}")
    jobs = [("A08 on Pro", s, "pro", ctx.a10.start_state(s)[0], f"a08_s{s}") for s in seeds]
    for key, mr in trained.items():
        own, other = ("pro", "original") if mr["arm"] == "Pro" else ("original", "pro")
        jobs += [(mr["arm"], mr["seed"], own, mr, key), (f"{mr['arm']} on {'original' if other == 'original' else 'Pro'}", mr["seed"], other, mr, key)]
    for inp in ("original", "pro"):
        todo = [j for j in jobs if j[2] == inp]
        if not todo:
            continue
        t0 = time.time(); frames = rr.decode_frames(refs, FramePaths(prv.frame_dirs(a8.paths, a8.rv, {r[0] for r in refs})), workers) if inp == "original" else pi.frames_for_refs(refs, cache)
        out["counts"][f"decode_seconds_{inp}"] = time.time() - t0
        for name, s, _, src, key in todo:
            path = ctx.out / "evaluation" / h / f"{key}_{inp}.npz"; ident = (src["identity"] if isinstance(src, dict) else f"A08 all_s{s}") + f"|{inp}"
            lg, _ = rr.cached_logits(path, ident, W, orders, seqs)
            if lg is None:
                if isinstance(src, dict):
                    if sha256_file(Path(src["weights"])) != src["sha256"]:
                        raise SystemExit(f"{key}: weights differ from the run record")
                    model = tp.make_model(torch.load(src["weights"], map_location="cpu", weights_only=True), device).eval()
                else:
                    model = tp.make_model(src, device).eval()
                lg, sec = rr.model_logits(model, W, orders, frames, device); out["gpu_seconds"] += float(sec); del model; torch.cuda.empty_cache()
                rr.save_outputs(path, ident, W, orders, lg, rr.per_sample(W, lg, seqs), seqs, sec)
            out["probs"][(name, s)] = rr.per_sample(W, lg, seqs); print(f"{h} {name} s{s} on {inp}: ready", flush=True)
        if inp == "original" and h == "calibration":                     # the original path reproduces A08's saved outputs (source parity)
            for s in seeds:
                state, _ = ctx.a10.start_state(s); src = tp.make_model(state, device).eval(); t1 = time.time()
                lg, _ = rr.model_logits(src, {"ref": W["ref"][:64], "marker": W["marker"][:64]}, orders[:64], frames, device); out["gpu_seconds"] += time.time() - t1; del src
                lgA, _ = rr.cached_logits(ROOT / "runs/video_within_view_v1/evaluation" / h / f"all_s{s}.npz", a8sum["runs"][f"all_s{s}"]["identity"], W, orders, seqs)
                out["parity"][f"s{s}"] = tp.vsh.vso.parity(lg, lgA[:64], 1e-4)
                if not out["parity"][f"s{s}"]["ok"]:
                    raise SystemExit("the original-path parity failed")
        del frames
    return out


def rule(rows: dict, counts: dict, x: str, refs: list[str]) -> dict:
    cells = [(s, h) for s in (0, 2) for h in HALVES]; F = lambda n, s, h: rows[f"{n} + B3 s{s}"][h]["segment_f1_50_multi"]  # noqa: E731
    out = {}
    for r in refs:
        out[f"mean_dF_vs_{r}"] = float(np.mean([F(x, s, h) - F(r, s, h) for s, h in cells]))
    dA = {f"s{s} {h}": F(x, s, h) - F("A08 A", s, h) for s, h in cells}; half = {h: float(np.mean([dA[f"s{s} {h}"] for s in (0, 2)])) for h in HALVES}
    dN = float(np.mean([rows[f"{x} + B3 s{s}"][h]["native_macro_f1"] - rows[f"A08 A + B3 s{s}"][h]["native_macro_f1"] for s, h in cells]))
    dBR = float(np.mean([rows[f"{x} + B3 s{s}"][h]["boundary_recall_multi"] - rows[f"A08 A + B3 s{s}"][h]["boundary_recall_multi"] for s, h in cells]))
    fs = {n: int(sum(counts[f"{n} + B3 s{s}"][h]["fn_short"] for s, h in cells)) for n in (x, "A08 A")}
    out.update({"dF_vs_A08_cells": dA, "half_means_vs_A08": half, "mean_dN": dN, "mean_dBR": dBR, "missed_short_true": fs})
    out["signal"] = bool(all(out[f"mean_dF_vs_{r}"] >= 0.003 for r in refs)); out["no_reversal"] = bool(min(half.values()) >= -0.005 and min(dA.values()) >= -0.015)
    out["no_harm"] = bool(dN >= -0.003 and dBR >= -0.010 and fs[x] <= fs["A08 A"] + 3); out["passes"] = bool(out["signal"] and out["no_reversal"] and out["no_harm"])
    return out


def run(ctx: tp.Ctx, stage: str, workers: int) -> None:
    t0, started, ok, note, gsec = time.time(), tp.now(), False, "", 0.0
    try:
        seeds = {"screen": [0, 2], "three_seed": [0, 1, 2], "frozen_seed1": [1]}[stage]; trained = {}
        for s in (seeds if stage != "frozen_seed1" else []):
            for arm in tp.ARMS:
                key = f"{arm}_s{s}"; man = json.loads((ctx.out / "train" / key / "run_manifest.json").read_text()); fz = ROOT / "models_frozen" / "video_processed_input_v1" / key / "final.pt"
                trained[key] = {"weights": str(fz if fz.is_file() else man["final_checkpoint"]), "sha256": man["final_sha256"], "identity": man["identity"], "arm": arm, "seed": s}
        a8 = rsh.a08_ctx(); rw.check_grid(a8); device = torch.device("cuda"); tp.determinism(); torch.cuda.set_per_process_memory_fraction(0.6); per, choices = {}, {}
        for h in ("calibration", "selection"):
            per[h] = read_half(ctx, a8, h, trained, seeds, device, workers); gsec += per[h]["gpu_seconds"]
            if h == "calibration":
                for (name, s), probs in per[h]["probs"].items():
                    ch = rcs.b3_choose(a8.cfg, rso.DIAG, probs, per[h]["ev"]); choices[f"{name} s{s}"] = {x: ch[x] for x in ("status", "params", "n_feasible")}
                    print(f"{name} s{s}: {ch['status']} {ch['params']}", flush=True)
                d = ctx.out / "evaluation" / stage; d.mkdir(parents=True, exist_ok=True); tp.write_json(d / "choices.json", {"choices": choices, "time": tp.now()})
        systems, sub = {}, {}
        for h in ("calibration", "selection"):
            P = per[h]; ev, seqs = P["ev"], P["seqs"]; evs = subset(ev, P["supported"]); sseqs = [q for q in seqs if q["id"] in P["supported"]]
            for s in seeds:
                entries = [("frozen Ordered", P["refs"][("frozen Ordered", s)], P["oparams"][s]), ("A08 A", P["refs"][("A08 A", s)], P["a08params"][s]),
                           ("A08 on Pro fixed", P["probs"][("A08 on Pro", s)], P["a08params"][s])]
                entries += [(name, probs, choices[f"{name} s{ss}"]["params"]) for (name, ss), probs in P["probs"].items() if ss == s]
                for name, probs, params in entries:
                    systems.setdefault(f"{name} + B3 s{s}", {})[h] = rso.row_of(a8.cfg, decode(params, probs), ev)
                    systems.setdefault(f"{name} argmax s{s}", {})[h] = rso.row_of(a8.cfg, {q: p.argmax(1) for q, p in probs.items()}, ev)
                    sp = {q["id"]: probs[q["id"]] for q in sseqs}; sub.setdefault(f"{name} + B3 s{s}", {})[h] = rso.row_of(a8.cfg, decode(params, sp), evs)
        rows = {k: {h: v[h]["row"] for h in v} for k, v in systems.items()}; srows = {k: {h: v[h]["row"] for h in v} for k, v in sub.items()}
        counts = {k: {h: rsh.counts_of(per[h]["seqs"], v[h]) for h in v} for k, v in systems.items() if " + B3 " in k}
        cmp_, acc = {}, {}
        for h in HALVES:
            for s in seeds:
                draws = {}
                for c, (a, b) in CMPS.items():
                    if f"{a} + B3 s{s}" not in systems or f"{b} + B3 s{s}" not in systems:
                        continue
                    for n in (a, b):
                        if n not in draws:
                            draws[n] = st.system_draws(systems[f"{n} + B3 s{s}"][h]["tables"])
                    cmp_[f"s{s}|{c}|{h}"] = st.paired(draws[a], draws[b])
                    acc[f"s{s}|{c}|{h}"] = st.segment_accounting(per[h]["seqs"], systems[f"{a} + B3 s{s}"][h]["preds"], systems[f"{b} + B3 s{s}"][h]["preds"])[0]
                print(f"{h} seed {s}: comparisons done", flush=True)
        verdicts = {}
        if len(seeds) == 3:
            for c in [c for c in CMPS if f"s0|{c}|selection" in cmp_]:
                for m in st.BOOT_METRICS:
                    verdicts[f"{c}|{m}"] = {h: st.consistency([cmp_[f"s{s}|{c}|{h}"][m] for s in seeds]) for h in HALVES}
        prep = json.loads((ctx.out / "preparation.json").read_text())
        res = {"stage": stage, "config_sha256": sha256_file(ROOT / ctx.config), "seeds": seeds, "models": {k: {"sha256": v["sha256"]} for k, v in trained.items()}, "choices": choices,
               "systems": rows, "common_support_systems": srows, "segment_counts": counts, "comparisons": cmp_, "accounting": acc, "verdicts": verdicts, "parity": per["calibration"]["parity"],
               "half_counts": {h: per[h]["counts"] for h in HALVES}, "preparation_status": prep.get("status_all_allowlisted"), "access": {"halves": list(HALVES), "time": started}}
        if stage == "screen":
            res["third_seed"] = rule(rows, counts, "Pro", ["Raw", "A08 A"]); res["stage_one_seed1"] = rule(rows, counts, "A08 on Pro fixed", ["A08 A"])
        ids = rw.ids_of(a8, {h: per[h]["seqs"] for h in HALVES}); flat = json.loads(json.dumps(res, default=str)); assert_no_ids(flat, ids, stage)
        d = ctx.out / "evaluation" / stage
        for p in (d / f"{stage}.json", ctx.results / f"{stage}.json"):
            p.parent.mkdir(parents=True, exist_ok=True); tp.write_json(p, flat)
        if stage == "screen":
            tp.write_json(d / "decision.json", {"train_seed1": flat["third_seed"]["passes"], "frozen_seed1": flat["stage_one_seed1"]["passes"], "time": tp.now()})
            print(json.dumps({k: {x: flat[k][x] for x in flat[k] if x not in ("dF_vs_A08_cells",)} for k in ("third_seed", "stage_one_seed1")}, indent=1), flush=True)
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        total = time.time() - t0; g = min(gsec, total)
        tp.ledger_append(ctx.ledger, f"evaluate {stage} (GPU inference)", "gpu", started, g, ok, note)
        tp.ledger_append(ctx.ledger, f"evaluate {stage} (CPU)", "cpu", started, total - g, ok, note)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--stage", choices=("screen", "three_seed", "frozen_seed1"), required=True); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(); a.set = []; run(tp.Ctx(load_cfg(a), a.config), a.stage, a.workers)


if __name__ == "__main__":
    main()
