#!/usr/bin/env python
"""Evaluate A18 (configs/experiments/video_ensemble_teacher_v1.yaml) on the development halves: the students E + B3 (one
calibration choice per seed) against A10's frozen R + B3 (committed choice), each seed's A08 + B3, frozen Ordered + B3,
A14 P and A17 O-fixed (reproduced); the three-model A08 probability average as its own system (one B3 choice, one row per
half); sensitivity readouts; and, after selection only, whether the students moved toward the teacher.

  --stage calibrate               E inference on calibration (outputs saved); E and ensemble B3 choices; writes choices.json
  --stage select --choices-sha S  both halves after choices.json (sha256 S) has been committed locally
  --stage diagnostic              after select: student-teacher agreement and shared errors
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.prepare_raw_video_context as prv  # noqa: E402
import scripts.report_center_supervision as rcs  # noqa: E402
import scripts.report_raw_video_context as rr  # noqa: E402
import scripts.report_router_data as rrd  # noqa: E402
import scripts.report_router_trajectory as rrt  # noqa: E402
import scripts.report_segment_objective as rso  # noqa: E402
import scripts.report_video_switch_head as rsh  # noqa: E402
import scripts.report_video_within_view as rw  # noqa: E402
import scripts.segment_objective_tables as st  # noqa: E402
import scripts.train_ensemble_teacher as tet  # noqa: E402
import scripts.train_router_trajectory as trt  # noqa: E402
import scripts.train_video_switch_head as vsh  # noqa: E402
import scripts.trained_token_readout as ttr  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.ev9v_complementarity import decode  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402
from scripts.train_joint_visual_temporal import FramePaths  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.temporal.segments import labels_to_segments  # noqa: E402

HALVES = ("selection", "calibration")
A17_CONFIG = "configs/experiments/video_router_trajectory_v1.yaml"
CMPS = {"E - R": ("E + B3", "R + B3"), "E - Base": ("E + B3", "Base"), "E - frozen Ordered": ("E + B3", "frozen Ordered + B3"), "E - A14 P": ("E + B3", "A14 P"),
        "E - A17 O-fixed": ("E + B3", "A17 O-fixed"), "R - Base": ("R + B3", "Base"), "E - ensemble": ("E + B3", "ensemble + B3"),
        "E@A08 - R@A08 (sensitivity)": ("E + B3@A08", "R + B3@A08")}


def e_models(ctx: tet.Ctx, seeds: list[int]) -> dict:
    out = {}
    for s in seeds:
        key = f"E_s{s}"; man = json.loads((ctx.out / "train" / key / "run_manifest.json").read_text())
        fz = ROOT / "models_frozen" / "video_ensemble_teacher_v1" / key / "final.pt"; w = fz if fz.is_file() else Path(man["final_checkpoint"])
        if sha256_file(w) != man["final_sha256"]:
            raise SystemExit(f"{key}: the weights differ from their run record")
        out[key] = {"weights": str(w), "sha256": man["final_sha256"], "identity": man["identity"], "seed": s, "frozen": fz.is_file()}
    return out


def windows_logits(path: Path, ident: str, W: dict) -> np.ndarray | None:
    if not path.is_file():
        return None
    with np.load(path) as z:
        if str(z["identity"]) == ident and np.array_equal(z["seq"], W["seq"]) and np.array_equal(z["centre"], W["centre"]):
            return z["logits"].astype(np.float32)
    return None


def read_half(ctx: tet.Ctx, a8, h: str, seeds: list[int], device, workers: int, routers: bool = True) -> dict:
    ev, seqs = a8.half(h, rso.DevAccess({h})); ids = [q["id"] for q in seqs]; refs = rw.half_refs(seqs); at = {r: i for i, r in enumerate(refs)}
    W = rr.half_windows(seqs, at, 5); orders = np.broadcast_to(np.arange(16, dtype=np.int64), W["s"].shape).copy(); p = ctx.ve["pinned"]
    a8sum = json.loads((ROOT / "docs/results/2026-10-16_video_within_view/summary.json").read_text()); a8ch = json.loads(vsh.pinned(ctx.a10.src["a08_choices"]).read_text())["choices"]
    v01 = a8.v01_summary()["systems"]; pins = rsh.a08_output_pins(); a10 = json.loads((ROOT / p["A10_screen"]["path"]).read_text())["systems"]
    a10ch = json.loads((ROOT / p["A10_choices"]["path"]).read_text())["choices"]
    out = {"ev": ev, "seqs": seqs, "W": W, "probs": {}, "params": {}, "wprobs": {}, "gpu_seconds": 0.0, "reproduced": [], "counts": {"sequences": len(seqs), "windows": int(len(W["seq"]))}}
    member_logits = {}
    for j in tet.TEACHERS:
        apath = ROOT / "runs/video_within_view_v1/evaluation" / h / f"all_s{j}.npz"
        if sha256_file(apath) != pins[h][j]:
            raise SystemExit(f"A08 all_s{j} {h}: the saved outputs differ from their pin")
        lg, _ = rr.cached_logits(apath, a8sum["runs"][f"all_s{j}"]["identity"], W, orders, seqs)
        if lg is None:
            raise SystemExit(f"A08 all_s{j} {h}: the saved outputs do not match their identity or this half's windows")
        member_logits[j] = lg.astype(np.float32); out["probs"][f"A08 all_s{j}"] = rr.per_sample(W, member_logits[j], seqs); out["params"][f"A08 all_s{j}"] = a8ch[f"all_s{j}"]["params"]
        rso.check_stored(rso.row_of(a8.cfg, decode(a8ch[f"all_s{j}"]["params"], out["probs"][f"A08 all_s{j}"]), ev)["row"], a8sum["systems"][f"A + B3 s{j}"][h], f"A08 all_s{j} + B3 {h}")
        out["wprobs"][f"A08 all_s{j}"] = torch.softmax(torch.from_numpy(member_logits[j]), -1).numpy()
    lq = tet.log_q(torch.from_numpy(np.stack([member_logits[j] for j in tet.TEACHERS]))).numpy()        # the teacher average, window level
    out["probs"]["ensemble"] = rr.per_sample(W, lq, seqs); out["wprobs"]["ensemble"] = np.exp(lq)
    for s in seeds:
        saved = a8.v01_output(h, s); ttr.check_windows(W, ids, saved, f"ordered_s{s} {h}"); op = v01[f"ordered_s{s}"]["b3_choice"]["params"]
        P = rr.per_sample(W, saved["logits"].astype(np.float32), seqs); rows = ttr.cls_rows(a8.cfg, P, ev, op)
        for row in ("argmax", "b3"):
            ttr.check_stored(rows[row]["row"], v01[f"ordered_s{s}"][row][h], f"ordered_s{s} {row} {h} (V01)")
        out["probs"][f"frozen Ordered s{s}"], out["params"][f"frozen Ordered s{s}"] = P, op
        rpath = ROOT / "runs/video_switch_head_v1/evaluation" / h / f"R_s{s}.npz"; rman = json.loads((ROOT / "runs/video_switch_head_v1/train" / f"R_s{s}" / "run_manifest.json").read_text())
        if sha256_file(rpath) != p["A10_R_outputs"][h][s] or rman["final_sha256"] != p["A10_R_weights"][s]:
            raise SystemExit(f"R_s{s} {h}: outputs or weights differ from their pins")
        lgR = windows_logits(rpath, rman["identity"], W)
        if lgR is None:
            raise SystemExit(f"R_s{s} {h}: the saved outputs do not match this half's windows")
        out["probs"][f"R s{s}"] = rr.per_sample(W, lgR, seqs); out["params"][f"R s{s}"] = a10ch[f"R_s{s}|b3"]["params"]; out["wprobs"][f"R s{s}"] = torch.softmax(torch.from_numpy(lgR), -1).numpy()
        rso.check_stored(rso.row_of(a8.cfg, decode(out["params"][f"R s{s}"], out["probs"][f"R s{s}"]), ev)["row"], a10[f"R + B3 s{s}"][h], f"R + B3 s{s} {h} (A10 screen)")
        rso.check_stored(rso.row_of(a8.cfg, {q: v.argmax(1) for q, v in out["probs"][f"R s{s}"].items()}, ev)["row"], a10[f"R argmax s{s}"][h], f"R argmax s{s} {h} (A10 screen)")
        out["reproduced"] += [f"A08 all_s0-2 + B3 {h}", f"frozen Ordered s{s} {h}", f"R s{s} + B3 and argmax {h}"]
    models = e_models(ctx, seeds); frames = None
    for key, mr in models.items():
        path = ctx.out / "evaluation" / h / f"{key}.npz"; lg = windows_logits(path, mr["identity"], W)
        if lg is None:
            if frames is None:
                fp = FramePaths(prv.frame_dirs(a8.paths, a8.rv, {r[0] for r in refs})); t0 = time.time(); frames = rr.decode_frames(refs, fp, workers); out["counts"]["decode_seconds"] = time.time() - t0
            st_ = torch.load(mr["weights"], map_location="cpu", weights_only=True); model = vsh.make_model(st_["model"], device).eval()
            lg, em, sec = rsh.infer(model, W, orders, frames, device); out["gpu_seconds"] += sec; out["counts"][f"{key}_inference_seconds"] = sec; del model; torch.cuda.empty_cache()
            path.parent.mkdir(parents=True, exist_ok=True); np.savez(path, identity=mr["identity"], seq=W["seq"], centre=W["centre"], logits=lg)
        out["probs"][f"E s{mr['seed']}"] = rr.per_sample(W, lg.astype(np.float32), seqs); out["wprobs"][f"E s{mr['seed']}"] = torch.softmax(torch.from_numpy(lg.astype(np.float32)), -1).numpy()
        print(f"{h} {key}: outputs ready", flush=True)
    del frames
    if routers:
        c17 = trt.Ctx(load_cfg(SimpleNamespace(config=A17_CONFIG, set=[])), A17_CONFIG); P17 = rrt.read_half(c17, a8, h, seeds)
        if [q["id"] for q in P17["seqs"]] != ids:
            raise SystemExit("A17's half differs in order")
        ch17 = json.loads((ROOT / "runs/video_router_trajectory_v1/evaluation/choices.json").read_text())["choices"]; a15ch = rrt.a15_choices(); out["routed"] = {}
        for s in seeds:
            d = P17["seed"][s]
            for name, key in (("A14 P", "A14 P"), ("A17 O-fixed", "O-fixed")):
                out["routed"][f"{name} s{s}"] = rrd.routed(P17, s, rrt.accept(key, d, ch17, a15ch, s))
    return out


def calibrate(ctx: tet.Ctx, seeds: list[int], workers: int) -> None:
    t0, started, ok, note, gsec = time.time(), tet.now(), False, "", 0.0
    try:
        a8 = rsh.a08_ctx(); rw.check_grid(a8); device = torch.device("cuda"); tet.determinism(); P = read_half(ctx, a8, "calibration", seeds, device, workers, routers=False); gsec += P["gpu_seconds"]
        choices = {}
        for key in [f"E s{s}" for s in seeds] + ["ensemble"]:
            ch = rcs.b3_choose(a8.cfg, rso.DIAG, P["probs"][key], P["ev"]); choices[f"{key}|b3"] = {x: ch[x] for x in ("status", "params", "n_feasible")}; choices[f"{key}|b3"]["grid"] = ch["grid"]
            print(f"{key}: B3 {ch['status']} {ch['params']} ({ch['n_feasible']} feasible)", flush=True)
        rec = {"stage": "calibrate", "seeds": seeds, "config_sha256": sha256_file(ROOT / ctx.config), "choices": choices, "models": {k: {x: v[x] for x in ("sha256", "identity", "frozen")} for k, v in e_models(ctx, seeds).items()},
               "references": {"R": "A10's committed R_s{k}|b3", "Base": "A08's committed all_s{k} B3 setting", "frozen Ordered": "V01's committed B3 setting", "A14 P / A17 O-fixed": "their committed rules"},
               "ensemble_members": {f"all_s{j}": ctx.ve["pinned"]["A08_teachers"][j] for j in tet.TEACHERS}, "time": tet.now()}
        flat = json.loads(json.dumps(rec, default=str)); assert_no_ids(flat, rw.ids_of(a8, {"calibration": P["seqs"]}), "choices")
        for p in (ctx.out / "evaluation" / "choices.json", ctx.results / "choices.json"):
            p.parent.mkdir(parents=True, exist_ok=True); tet.write_json(p, flat)
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        tet.ledger_append(ctx.ledger, "evaluate calibrate (E inference, GPU job wall time)", "gpu", started, time.time() - t0, ok, note)


def promotion(rows: dict, counts: dict, ch: dict) -> dict:
    cells = [(s, h) for s in (0, 2) for h in HALVES]; F = lambda n, s, h: rows[f"{n} s{s}"][h]["segment_f1_50_multi"]  # noqa: E731
    dF = {f"s{s} {h}": F("E + B3", s, h) - F("Base", s, h) for s, h in cells}; half = {h: float(np.mean([dF[f"s{s} {h}"] for s in (0, 2)])) for h in HALVES}
    dN = float(np.mean([rows[f"E + B3 s{s}"][h]["native_macro_f1"] - rows[f"Base s{s}"][h]["native_macro_f1"] for s, h in cells]))
    dBR = float(np.mean([rows[f"E + B3 s{s}"][h]["boundary_recall_multi"] - rows[f"Base s{s}"][h]["boundary_recall_multi"] for s, h in cells]))
    fs = {n: int(sum(counts[f"{n} s{s}"][h]["fn_short"] for s, h in cells)) for n in ("E + B3", "Base")}
    feas = {f"s{s}": ch["choices"][f"E s{s}|b3"]["status"] == "feasible" for s in (0, 2)}
    r = {"mean_dF": float(np.mean(list(dF.values()))), "dF": dF, "half_means": half, "mean_dN": dN, "mean_dBR": dBR, "missed_short_true": fs, "calibration_feasible": feas}
    r["signal"] = bool(r["mean_dF"] >= 0.003); r["no_reversal"] = bool(min(half.values()) >= -0.005 and min(dF.values()) >= -0.015)
    r["no_harm"] = bool(dN >= -0.003 and dBR >= -0.010 and fs["E + B3"] <= fs["Base"] + 3); r["feasibility"] = bool(all(feas.values()))
    r["meets_rule"] = bool(r["signal"] and r["no_reversal"] and r["no_harm"] and r["feasibility"]); r["train_seed1"] = r["meets_rule"]
    return r


def systems_of(a8, P: dict, seeds: list[int], choices: dict) -> dict:
    ev, out = P["ev"], {}
    for s in seeds:
        for name, preds in (("Base", decode(P["params"][f"A08 all_s{s}"], P["probs"][f"A08 all_s{s}"])), ("A08 argmax", {q: v.argmax(1) for q, v in P["probs"][f"A08 all_s{s}"].items()}),
                            ("frozen Ordered + B3", decode(P["params"][f"frozen Ordered s{s}"], P["probs"][f"frozen Ordered s{s}"])),
                            ("R + B3", decode(P["params"][f"R s{s}"], P["probs"][f"R s{s}"])), ("R argmax", {q: v.argmax(1) for q, v in P["probs"][f"R s{s}"].items()}),
                            ("R + B3@A08", decode(P["params"][f"A08 all_s{s}"], P["probs"][f"R s{s}"])),
                            ("E + B3", decode(choices[f"E s{s}|b3"]["params"], P["probs"][f"E s{s}"])), ("E argmax", {q: v.argmax(1) for q, v in P["probs"][f"E s{s}"].items()}),
                            ("E + B3@A08", decode(P["params"][f"A08 all_s{s}"], P["probs"][f"E s{s}"])),
                            ("ensemble + B3", decode(choices["ensemble|b3"]["params"], P["probs"]["ensemble"])), ("A14 P", P["routed"][f"A14 P s{s}"]), ("A17 O-fixed", P["routed"][f"A17 O-fixed s{s}"])):
            out[f"{name} s{s}"] = rso.row_of(a8.cfg, preds, ev)
    out["A08 all_s1 + B3"] = rso.row_of(a8.cfg, decode(P["params"]["A08 all_s1"], P["probs"]["A08 all_s1"]), ev)
    out["ensemble argmax"] = rso.row_of(a8.cfg, {q: v.argmax(1) for q, v in P["probs"]["ensemble"].items()}, ev)
    return out


def select(ctx: tet.Ctx, seeds: list[int], choices_sha: str, workers: int) -> None:
    t0, started, ok, note = time.time(), tet.now(), False, ""
    try:
        cp = ctx.out / "evaluation" / "choices.json"
        if sha256_file(cp) != choices_sha:
            raise SystemExit("choices.json differs from the committed sha256; selection is not read")
        ch = json.loads(cp.read_text()); choices = ch["choices"]; a8 = rsh.a08_ctx(); device = torch.device("cuda"); tet.determinism()
        per = {h: read_half(ctx, a8, h, seeds, device, workers) for h in ("calibration", "selection")}; gsec = sum(per[h]["gpu_seconds"] for h in per)
        a14 = json.loads(rrd.A14_SCREEN.read_text())["systems"]; a17 = json.loads((ROOT / "runs/video_router_trajectory_v1/evaluation/screen.json").read_text())["systems"]; systems = {}
        for h in ("calibration", "selection"):
            for k, v in systems_of(a8, per[h], seeds, choices).items():
                systems.setdefault(k, {})[h] = v
            for s in seeds:
                rso.check_stored(systems[f"A14 P s{s}"][h]["row"], a14[f"P router s{s}"][h], f"A14 P s{s} {h}"); rso.check_stored(systems[f"A17 O-fixed s{s}"][h]["row"], a17[f"O-fixed s{s}"][h], f"A17 O-fixed s{s} {h}")
        rows = {k: {h: v[h]["row"] for h in v} for k, v in systems.items()}; counts = {k: {h: rsh.counts_of(per[h]["seqs"], v[h]) for h in v} for k, v in systems.items() if "argmax" not in k}
        cmp_, acc_ = {}, {}
        for h in HALVES:
            for s in seeds:
                draws = {}
                for c, (a, b) in CMPS.items():
                    ka, kb = (f"{a} s{s}", f"{b} s{s}")
                    for nm in (ka, kb):
                        if nm not in draws:
                            draws[nm] = st.system_draws(systems[nm][h]["tables"])
                    cmp_[f"s{s}|{c}|{h}"] = st.paired(draws[ka], draws[kb]); acc_[f"s{s}|{c}|{h}"] = st.segment_accounting(per[h]["seqs"], systems[ka][h]["preds"], systems[kb][h]["preds"])[0]
            ens = st.system_draws(systems["ensemble + B3 s0"][h]["tables"])
            for j, key in ((0, "Base s0"), (1, "A08 all_s1 + B3"), (2, "Base s2")):
                cmp_[f"ensemble - A08 all_s{j}|{h}"] = st.paired(ens, st.system_draws(systems[key][h]["tables"]))
                acc_[f"ensemble - A08 all_s{j}|{h}"] = st.segment_accounting(per[h]["seqs"], systems["ensemble + B3 s0"][h]["preds"], systems[key][h]["preds"])[0]
            print(f"{h}: comparisons done", flush=True)
        mr = e_models(ctx, seeds); mcpu = vsh.make_model(torch.load(mr[f"E_s{seeds[0]}"]["weights"], map_location="cpu", weights_only=True)["model"], torch.device("cpu"))
        n_params = int(sum(x.numel() for x in mcpu.parameters())); del mcpu
        cost = {"student": {"parameters": n_params, "forwards_per_window": 1, "windows": {h: per[h]["counts"]["windows"] for h in HALVES},
                            "inference_seconds": {h: {k: v for k, v in per[h]["counts"].items() if "inference_seconds" in k} for h in HALVES}},
                "ensemble": {"forwards_per_window": 3, "note": "three A08 models; its development outputs came from their saved logits"}}
        res = {"stage": "select", "seeds": seeds, "config_sha256": sha256_file(ROOT / ctx.config), "choices_sha256": choices_sha, "systems": rows, "segment_counts": counts, "comparisons": cmp_, "accounting": acc_,
               "reproduced": {h: per[h]["reproduced"] + ["A14 P and A17 O-fixed rows"] for h in HALVES}, "cost": cost, "half_counts": {h: per[h]["counts"] for h in HALVES},
               "notes": {"ensemble": "one system per half (rows under the s0 key and the s2 key are identical copies, never separate cells)"}, "access": {"halves": list(HALVES), "time": started}}
        if seeds == [0, 2]:
            res["promotion"] = promotion(rows, counts, ch)
        ids = rw.ids_of(a8, {h: per[h]["seqs"] for h in HALVES}); flat = json.loads(json.dumps(res, default=str)); assert_no_ids(flat, ids, "select")
        for p in (ctx.out / "evaluation" / "screen.json", ctx.results / "screen.json"):
            p.parent.mkdir(parents=True, exist_ok=True); tet.write_json(p, flat)
        if "promotion" in flat:
            tet.write_json(ctx.out / "evaluation" / "decision.json", {**flat["promotion"], "time": tet.now()})
            print(json.dumps({k: flat["promotion"][k] for k in ("mean_dF", "half_means", "mean_dN", "mean_dBR", "missed_short_true", "calibration_feasible", "meets_rule")}, indent=1), flush=True)
        ok = True; note = f"GPU inference {gsec:.0f} s"
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        tet.ledger_append(ctx.ledger, "evaluate select (E inference on selection, GPU job wall time)", "gpu", started, time.time() - t0, ok, note)


# ----------------------------------------------------------------------------------------------- diagnostic
def diagnostic(ctx: tet.Ctx, seeds: list[int], workers: int) -> None:
    t0, started, ok, note = time.time(), tet.now(), False, ""
    try:
        if not (ctx.out / "evaluation" / "screen.json").exists():
            raise SystemExit("the diagnostic runs after the selection stage only")
        ch = json.loads((ctx.out / "evaluation" / "choices.json").read_text())["choices"]; a8 = rsh.a08_ctx(); device = torch.device("cuda"); out = {"note": "after selection; descriptive", "halves": {}}
        seqs_of = {}
        for h in HALVES:
            P = read_half(ctx, a8, h, seeds, device, workers, routers=False); seqs_of[h] = P["seqs"]; sysd = {k: v["preds"] for k, v in systems_of_light(a8, P, seeds, ch).items()}; q = P["wprobs"]["ensemble"]; rec = {}
            for s in seeds:
                own, E, R = P["wprobs"][f"A08 all_s{s}"], P["wprobs"][f"E s{s}"], P["wprobs"][f"R s{s}"]
                rec[f"s{s}"] = {"window_l1_to_teacher": {n: float(np.abs(x - q).sum(1).mean()) for n, x in (("Base", own), ("R", R), ("E", E))},
                                "window_argmax_agrees_with_teacher": {n: float((x.argmax(1) == q.argmax(1)).mean()) for n, x in (("Base", own), ("R", R), ("E", E))},
                                "sample_labels_equal_ensemble_b3": {n: float(np.mean(np.concatenate([sysd[f"{n} s{s}"][qq["id"]] == sysd["ensemble"][qq["id"]] for qq in P["seqs"] if qq["kind"] == "multi"])))
                                                                    for n in ("Base", "R + B3", "E + B3")}}
            seg = {"reference_segments": 0, "by_members_matched": {str(k): 0 for k in range(4)}, "ensemble_matched_given_members": {str(k): 0 for k in range(4)}}
            for s in seeds:
                seg[f"E s{s}_matched_given_members"] = {str(k): 0 for k in range(4)}
            for qq in P["seqs"]:
                if qq["kind"] != "multi":
                    continue
                ref = labels_to_segments(np.asarray(qq["labels"])); mt = lambda preds: {j for _, j in st.match_pairs(labels_to_segments(np.asarray(preds)), ref)}  # noqa: E731
                mem = [mt(sysd[f"member {j}"][qq["id"]]) for j in tet.TEACHERS]; ens = mt(sysd["ensemble"][qq["id"]]); es = {s: mt(sysd[f"E + B3 s{s}"][qq["id"]]) for s in seeds}
                for j in range(len(ref)):
                    k = str(sum(j in x for x in mem)); seg["reference_segments"] += 1; seg["by_members_matched"][k] += 1; seg["ensemble_matched_given_members"][k] += int(j in ens)
                    for s in seeds:
                        seg[f"E s{s}_matched_given_members"][k] += int(j in es[s])
            rec["segments"] = seg; out["halves"][h] = rec
        flat = json.loads(json.dumps(out, default=str)); assert_no_ids(flat, rw.ids_of(a8, seqs_of), "diagnostic")
        for p in (ctx.out / "evaluation" / "diagnostic.json", ctx.results / "diagnostic.json"):
            tet.write_json(p, flat)
        print(json.dumps(flat, indent=1), flush=True); ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        tet.ledger_append(ctx.ledger, "evaluate diagnostic (CPU)", "cpu", started, time.time() - t0, ok, note)


def systems_of_light(a8, P: dict, seeds: list[int], choices: dict) -> dict:
    out = {"ensemble": {"preds": decode(choices["ensemble|b3"]["params"], P["probs"]["ensemble"])}}
    for j in tet.TEACHERS:
        out[f"member {j}"] = {"preds": decode(P["params"][f"A08 all_s{j}"], P["probs"][f"A08 all_s{j}"])}
    for s in seeds:
        out[f"Base s{s}"] = out[f"member {s}"]; out[f"R + B3 s{s}"] = {"preds": decode(P["params"][f"R s{s}"], P["probs"][f"R s{s}"])}
        out[f"E + B3 s{s}"] = {"preds": decode(choices[f"E s{s}|b3"]["params"], P["probs"][f"E s{s}"])}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--stage", choices=("calibrate", "select", "diagnostic"), required=True); ap.add_argument("--choices-sha")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 2]); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(); a.set = []; ctx = tet.Ctx(load_cfg(a), a.config)
    if a.stage == "calibrate":
        return calibrate(ctx, a.seeds, a.workers)
    if a.stage == "select":
        return select(ctx, a.seeds, a.choices_sha, a.workers)
    diagnostic(ctx, a.seeds, a.workers)


if __name__ == "__main__":
    main()
