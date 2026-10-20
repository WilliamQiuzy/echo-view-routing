#!/usr/bin/env python
"""Evaluate A13 (configs/experiments/video_frozen_switch_v1.yaml) on the development halves.

The frozen A08 of each seed gives every window's logits and embeddings (new logits must equal A08's saved outputs within
1e-4 on every window); its probabilities name the segments. Each A13 head decodes through A11's sustained-switch decoder (no
left/right check). A11's U/H head systems are rebuilt from their frozen heads and cached outputs for paired comparisons.

  --stage calibrate               the calibration half only: one choice per head (choices.json); nothing from selection is read
  --stage select --choices-sha S  both halves, after choices.json (sha256 S) has been committed locally
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
import scripts.prepare_raw_video_context as prv  # noqa: E402
import scripts.report_raw_video_context as rr  # noqa: E402
import scripts.report_segment_objective as rso  # noqa: E402
import scripts.report_video_sustained_switch as rvs  # noqa: E402
import scripts.report_video_switch_head as rsh  # noqa: E402
import scripts.report_video_within_view as rw  # noqa: E402
import scripts.segment_objective_tables as st  # noqa: E402
import scripts.train_frozen_switch as tf  # noqa: E402
import scripts.train_video_sustained_switch as tss  # noqa: E402
import scripts.trained_token_readout as ttr  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.ev9v_complementarity import decode  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402
from scripts.train_joint_visual_temporal import FramePaths  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402

HALVES = ("selection", "calibration")
CMPS = {"U - A08 A": ("U head", "A08 A + B3"), "H - A08 A": ("H head", "A08 A + B3"), "H - U": ("H head", "U head"), "U - A11 U": ("U head", "A11 U head"),
        "H - A11 H": ("H head", "A11 H head")}


def a13_head(ctx: tf.Ctx, arm: str, s: int):
    man = json.loads((ctx.out / "train" / f"{arm}_s{s}" / "run_manifest.json").read_text()); p = Path(man["final_checkpoint"])
    if sha256_file(p) != man["final_sha256"]:
        raise SystemExit(f"{arm}_s{s}: head weights differ from the run record")
    h = tss.build_head(s); h.load_state_dict(torch.load(p, map_location="cpu", weights_only=True)); return h.eval(), man


def a11_head(arm: str, s: int):
    man = json.loads((ROOT / "runs/video_sustained_switch_v1/train" / f"{arm}_s{s}" / "run_manifest.json").read_text()); p = ROOT / "models_frozen/video_sustained_switch_v1" / f"{arm}_s{s}" / "final.pt"
    if sha256_file(p) != man["final_sha256"]:
        raise SystemExit(f"A11 {arm}_s{s}: frozen weights differ")
    h = tss.build_head(s); h.load_state_dict(torch.load(p, map_location="cpu", weights_only=True)["head"]); return h.eval(), man


def read_half(ctx: tf.Ctx, a8, h: str, seeds: list[int], device, workers: int) -> dict:
    ev, seqs = a8.half(h, rso.DevAccess({h})); ids = [q["id"] for q in seqs]; refs = rw.half_refs(seqs); at = {r: i for i, r in enumerate(refs)}
    W = rr.half_windows(seqs, at, 5); orders = np.broadcast_to(np.arange(16, dtype=np.int64), W["s"].shape).copy(); frames = None
    v01 = a8.v01_summary()["systems"]; a8sum = json.loads((ROOT / "docs/results/2026-10-16_video_within_view/summary.json").read_text())
    a8ch = json.loads(tss.vsh.pinned(ctx.a11.a10.src["a08_choices"]).read_text())["choices"]; pins = rsh.a08_output_pins(); a11ch = json.loads((ROOT / "runs/video_sustained_switch_v1/evaluation/screen/choices.json").read_text())["choices"]
    out = {"ev": ev, "seqs": seqs, "probs": {}, "edges": {}, "params": {}, "parity": {}, "gpu_seconds": 0.0, "counts": {"sequences": len(seqs), "windows": int(len(W["seq"]))}}
    for s in seeds:
        saved = a8.v01_output(h, s); ttr.check_windows(W, ids, saved, f"ordered_s{s} {h}"); op = v01[f"ordered_s{s}"]["b3_choice"]["params"]
        P = rr.per_sample(W, saved["logits"].astype(np.float32), seqs); rows = ttr.cls_rows(a8.cfg, P, ev, op)
        for row in ("argmax", "b3"):
            ttr.check_stored(rows[row]["row"], v01[f"ordered_s{s}"][row][h], f"ordered_s{s} {row} {h}")
        out["probs"][("frozen Ordered", s)] = P; out["params"][("frozen Ordered", s)] = op
        apath = ROOT / "runs/video_within_view_v1/evaluation" / h / f"all_s{s}.npz"
        if sha256_file(apath) != pins[h][s]:
            raise SystemExit("A08 saved outputs differ from their pin")
        lgA, _ = rr.cached_logits(apath, a8sum["runs"][f"all_s{s}"]["identity"], W, orders, seqs)
        if lgA is None:
            raise SystemExit("A08 saved outputs do not match")
        PA = rr.per_sample(W, lgA, seqs); ap = a8ch[f"all_s{s}"]["params"]; out["probs"][("A08 A", s)] = PA; out["params"][("A08 A", s)] = ap
        ttr.check_stored(rso.row_of(a8.cfg, decode(ap, PA), ev)["row"], a8sum["systems"][f"A + B3 s{s}"][h], f"A08 A + B3 s{s} {h}")
        fpath = ctx.out / "evaluation" / h / f"a08_s{s}.npz"; got = None
        if fpath.is_file():
            with np.load(fpath) as z:
                if np.array_equal(z["seq"], W["seq"]) and np.array_equal(z["centre"], W["centre"]):
                    got = (z["logits"], z["emb"])
        if got is None:
            if frames is None:
                frames = rr.decode_frames(refs, FramePaths(prv.frame_dirs(a8.paths, a8.rv, {r[0] for r in refs})), workers)
            model, _ = tf.frozen(ctx, s, device); d0 = tf.path_digest(model); lg, em, sec = rsh.infer(model, W, orders, frames, device); out["gpu_seconds"] += sec
            if tf.path_digest(model) != d0 or model.training:
                raise SystemExit("the frozen path changed during inference")
            del model; torch.cuda.empty_cache(); fpath.parent.mkdir(parents=True, exist_ok=True); np.savez(fpath, seq=W["seq"], centre=W["centre"], logits=lg, emb=em); got = (lg, em)
        lg, em = got[0].astype(np.float32), got[1].astype(np.float32)
        out["parity"][f"s{s}"] = tss.vsh.vso.parity(lg, lgA, 1e-4)
        if not out["parity"][f"s{s}"]["ok"]:
            raise SystemExit(f"seed {s} {h}: the frozen A08 logits differ from the saved A08 outputs ({out['parity'][f's{s}']})")
        Pn = rr.per_sample(W, lg, seqs); out["probs"][("A13 CLS", s)] = Pn; out["params"][("A13 CLS", s)] = ap
        for arm in ("U", "H"):
            hd, _ = a13_head(ctx, arm, s); out["edges"][(f"{arm} head", s)] = rvs.edge_scores(hd, W, lg, em, seqs, torch.device("cpu")); out["probs"][(f"{arm} head", s)] = Pn
            h11, man11 = a11_head(arm, s); p11 = ROOT / "runs/video_sustained_switch_v1/evaluation" / h / f"{arm}_s{s}.npz"
            with np.load(p11) as z:
                if str(z["identity"]) != man11["identity"] or not np.array_equal(z["seq"], W["seq"]) or not np.array_equal(z["centre"], W["centre"]):
                    raise SystemExit(f"A11 {arm}_s{s} {h}: cached outputs do not match")
                l11, e11 = z["logits"].astype(np.float32), z["emb"].astype(np.float32)
            out["probs"][(f"A11 {arm} head", s)] = rr.per_sample(W, l11, seqs); out["edges"][(f"A11 {arm} head", s)] = rvs.edge_scores(h11, W, l11, e11, seqs, torch.device("cpu"))
            out["params"][(f"A11 {arm} head", s)] = a11ch[f"{arm}_s{s}|head"]["params"]
        print(f"{h} seed {s}: outputs ready", flush=True)
    return out


def head_rows(cfg, P: dict, name: str, s: int, params: dict) -> dict:
    return rso.row_of(cfg, rsh.head_decode(P["probs"][(name, s)], P["edges"][(name, s)], P["seqs"], float(params["threshold"])), P["ev"])


def calibrate(ctx: tf.Ctx, seeds: list[int], workers: int) -> None:
    t0, started, ok, note, gsec = time.time(), tf.now(), False, "", 0.0
    try:
        a8 = rsh.a08_ctx(); rw.check_grid(a8); device = torch.device("cuda"); tf.vsh.determinism(); P = read_half(ctx, a8, "calibration", seeds, device, workers); gsec = P["gpu_seconds"]; choices = {}
        for s in seeds:
            for arm in ("U", "H"):
                hw = int(json.loads((ROOT / "runs/video_sustained_switch_v1/evaluation/screen/choices.json").read_text())["choices"][f"{arm}_s{s}|b3"]["params"]["half_window"])
                ch = rvs.choose(a8.cfg, P["probs"][(f"{arm} head", s)], P["edges"][(f"{arm} head", s)], P["seqs"], P["ev"], False, hw)
                choices[f"{arm}_s{s}"] = {x: ch[x] for x in ("status", "params", "n_feasible", "grid")}; print(f"{arm}_s{s}: {ch['status']} {ch['params']}", flush=True)
        rec = {"stage": "calibrate", "seeds": seeds, "config_sha256": sha256_file(ROOT / ctx.config), "choices": choices, "parity": P["parity"], "time": tf.now(),
               "heads": {f"{a}_s{s}": a13_head(ctx, a, s)[1]["final_sha256"] for s in seeds for a in ("U", "H")}}
        flat = json.loads(json.dumps(rec, default=str)); assert_no_ids(flat, rw.ids_of(a8, {"calibration": P["seqs"]}), "choices")
        for p in (ctx.out / "evaluation" / "choices.json", ctx.results / "choices.json"):
            p.parent.mkdir(parents=True, exist_ok=True); tf.write_json(p, flat)
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        total = time.time() - t0; g = min(gsec, total)
        tf.ledger_append(ctx.ledger, "evaluate calibrate (GPU inference)", "gpu", started, g, ok, note); tf.ledger_append(ctx.ledger, "evaluate calibrate (CPU)", "cpu", started, total - g, ok, note)


def third_seed(rows: dict, counts: dict) -> dict:
    cells = [(s, h) for s in (0, 2) for h in HALVES]; out = {}
    for arm in ("U", "H"):
        x = f"{arm} head"; F = lambda n, s, h: rows[f"{n} s{s}"][h]["segment_f1_50_multi"]  # noqa: E731
        dF = {f"s{s} {h}": F(x, s, h) - F("A08 A + B3", s, h) for s, h in cells}; half = {h: float(np.mean([dF[f"s{s} {h}"] for s in (0, 2)])) for h in HALVES}
        dN = float(np.mean([rows[f"{x} s{s}"][h]["native_macro_f1"] - rows[f"A08 A + B3 s{s}"][h]["native_macro_f1"] for s, h in cells]))
        dBR = float(np.mean([rows[f"{x} s{s}"][h]["boundary_recall_multi"] - rows[f"A08 A + B3 s{s}"][h]["boundary_recall_multi"] for s, h in cells]))
        fs = {n: int(sum(counts[f"{n} s{s}"][h]["fn_short"] for s, h in cells)) for n in (x, "A08 A + B3")}
        r = {"mean_dF": float(np.mean(list(dF.values()))), "dF": dF, "half_means": half, "mean_dN": dN, "mean_dBR": dBR, "missed_short_true": fs}
        r["signal"] = bool(r["mean_dF"] >= 0.003); r["no_reversal"] = bool(min(half.values()) >= -0.005 and min(dF.values()) >= -0.015)
        r["no_harm"] = bool(dN >= -0.003 and dBR >= -0.010 and fs[x] <= fs["A08 A + B3"] + 3); r["passes"] = bool(r["signal"] and r["no_reversal"] and r["no_harm"]); out[arm] = r
    out["train_seed1"] = bool(out["U"]["passes"] or out["H"]["passes"])
    return out


def select(ctx: tf.Ctx, seeds: list[int], choices_sha: str, workers: int) -> None:
    t0, started, ok, note, gsec = time.time(), tf.now(), False, "", 0.0
    try:
        cp = ctx.out / "evaluation" / "choices.json"
        if sha256_file(cp) != choices_sha:
            raise SystemExit("choices.json differs from the committed sha256; selection is not read")
        choices = json.loads(cp.read_text())["choices"]; a8 = rsh.a08_ctx(); device = torch.device("cuda"); tf.vsh.determinism(); per = {}
        for h in ("calibration", "selection"):
            per[h] = read_half(ctx, a8, h, seeds, device, workers); gsec += per[h]["gpu_seconds"]
        systems = {}
        for h in ("calibration", "selection"):
            P = per[h]
            for s in seeds:
                for name in ("frozen Ordered", "A08 A", "A13 CLS"):
                    systems.setdefault(f"{name} + B3 s{s}", {})[h] = rso.row_of(a8.cfg, decode(P["params"][(name, s)], P["probs"][(name, s)]), P["ev"])
                for arm in ("U", "H"):
                    systems.setdefault(f"{arm} head s{s}", {})[h] = head_rows(a8.cfg, P, f"{arm} head", s, choices[f"{arm}_s{s}"]["params"])
                    systems.setdefault(f"A11 {arm} head s{s}", {})[h] = head_rows(a8.cfg, P, f"A11 {arm} head", s, P["params"][(f"A11 {arm} head", s)])
        a11rows = json.loads((ROOT / "docs/results/2026-10-20_sustained_switch/screen.json").read_text())["systems"]
        for s in seeds:
            for h in HALVES:
                rso.check_stored(systems[f"A13 CLS + B3 s{s}"][h]["row"], systems[f"A08 A + B3 s{s}"][h]["row"], f"A13 CLS + B3 s{s} {h} (must equal A08)")
                for arm in ("U", "H"):
                    rso.check_stored(systems[f"A11 {arm} head s{s}"][h]["row"], a11rows[f"{arm} head s{s}"][h], f"A11 {arm} head s{s} {h} (A11 screen)")
        rows = {k: {h: v[h]["row"] for h in v} for k, v in systems.items()}
        counts = {k: {h: rsh.counts_of(per[h]["seqs"], v[h]) for h in v} for k, v in systems.items()}
        cmp_, acc = {}, {}
        for h in HALVES:
            for s in seeds:
                draws = {}
                for c, (a, b) in CMPS.items():
                    for n in (a, b):
                        if n not in draws:
                            draws[n] = st.system_draws(systems[f"{n} s{s}"][h]["tables"])
                    cmp_[f"s{s}|{c}|{h}"] = st.paired(draws[a], draws[b])
                    acc[f"s{s}|{c}|{h}"] = st.segment_accounting(per[h]["seqs"], systems[f"{a} s{s}"][h]["preds"], systems[f"{b} s{s}"][h]["preds"])[0]
                print(f"{h} seed {s}: comparisons done", flush=True)
        verdicts = {}
        if len(seeds) == 3:
            for c in CMPS:
                for m in st.BOOT_METRICS:
                    verdicts[f"{c}|{m}"] = {h: st.consistency([cmp_[f"s{s}|{c}|{h}"][m] for s in seeds]) for h in HALVES}
        res = {"stage": "select", "seeds": seeds, "config_sha256": sha256_file(ROOT / ctx.config), "choices_sha256": choices_sha, "systems": rows, "segment_counts": counts, "comparisons": cmp_,
               "accounting": acc, "verdicts": verdicts, "parity": {h: per[h]["parity"] for h in HALVES}, "half_counts": {h: per[h]["counts"] for h in HALVES},
               "reproduced": ["A13 CLS + B3 equals A08 A + B3 in every seed and half", "A11 U/H head rows equal A11's screen rows"], "access": {"halves": list(HALVES), "time": started}}
        if seeds == [0, 2]:
            res["third_seed"] = third_seed(rows, counts)
        ids = rw.ids_of(a8, {h: per[h]["seqs"] for h in HALVES}); flat = json.loads(json.dumps(res, default=str)); assert_no_ids(flat, ids, "select")
        name = "screen" if seeds == [0, 2] else "three_seed"
        for p in (ctx.out / "evaluation" / f"{name}.json", ctx.results / f"{name}.json"):
            p.parent.mkdir(parents=True, exist_ok=True); tf.write_json(p, flat)
        if "third_seed" in flat:
            tf.write_json(ctx.out / "evaluation" / "decision.json", {**flat["third_seed"], "time": tf.now()})
            print(json.dumps({a: {k: flat["third_seed"][a][k] for k in ("mean_dF", "half_means", "mean_dN", "mean_dBR", "missed_short_true", "passes")} for a in ("U", "H")}, indent=1),
                  "train_seed1:", flat["third_seed"]["train_seed1"], flush=True)
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        total = time.time() - t0; g = min(gsec, total)
        tf.ledger_append(ctx.ledger, "evaluate select (GPU inference)", "gpu", started, g, ok, note); tf.ledger_append(ctx.ledger, "evaluate select (CPU)", "cpu", started, total - g, ok, note)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--stage", choices=("calibrate", "select"), required=True); ap.add_argument("--choices-sha")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 2]); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(); a.set = []; ctx = tf.Ctx(load_cfg(a), a.config)
    if a.stage == "calibrate":
        return calibrate(ctx, a.seeds, a.workers)
    select(ctx, a.seeds, a.choices_sha, a.workers)


if __name__ == "__main__":
    main()
