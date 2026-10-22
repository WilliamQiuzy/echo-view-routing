#!/usr/bin/env python
"""Evaluate A20 (configs/experiments/video_hard_replay_v1.yaml): U and H decoded at their seed's committed A08 B3 setting
(primary), one own-calibration B3 choice each and argmax (sensitivity), on both development halves and once on A15's check
pool; A08, frozen Ordered, R, A14 P and A19's check rows reproduced; window accuracy and CE by fixed reference/A08 groups.

  --stage training                training windows only: base and replay CE / accuracy per run (no evaluation data)
  --stage calibrate               U/H inference on calibration (outputs saved); the sensitivity B3 choices; choices.json
  --stage select --choices-sha S  selection inference, the check pool once, rows, comparisons, window groups, promotion
  --stage background              post hoc, no decision: A10 R's window groups on the development halves (saved outputs)
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
import scripts.b3_coverage as bc  # noqa: E402
import scripts.b3_coverage_evaluate as bce  # noqa: E402
import scripts.prepare_hard_replay as phr  # noqa: E402
import scripts.prepare_raw_video_context as prv  # noqa: E402
import scripts.report_center_supervision as rcs  # noqa: E402
import scripts.report_raw_video_context as rr  # noqa: E402
import scripts.report_router_data as rrd  # noqa: E402
import scripts.report_router_trajectory as rrt  # noqa: E402
import scripts.report_segment_objective as rso  # noqa: E402
import scripts.report_video_switch_head as rsh  # noqa: E402
import scripts.report_video_within_view as rw  # noqa: E402
import scripts.segment_objective_tables as st  # noqa: E402
import scripts.train_hard_replay as thr  # noqa: E402
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

HALVES = ("selection", "calibration")
ARMS = ("U", "H")
CHUNK = 150
A17_CONFIG = "configs/experiments/video_router_trajectory_v1.yaml"
A10_SCREEN = ROOT / "docs/results/2026-10-18_video_switch_head/screen.json"
A18_SCREEN3 = ROOT / "runs/video_ensemble_teacher_v1/evaluation/screen_three_seeds.json"
A19_SCREEN = ROOT / "runs/video_b3_coverage_v1/screen.json"
DEV_CMPS = {"H@A08 - U@A08": ("H@A08", "U@A08"), "U@A08 - Base": ("U@A08", "Base"), "H@A08 - Base": ("H@A08", "Base"), "H own-cal - U own-cal": ("H own-cal", "U own-cal"),
            "U@A08 - R + B3": ("U@A08", "R + B3"), "H@A08 - R + B3": ("H@A08", "R + B3"), "U@A08 - A14 P": ("U@A08", "A14 P"), "H@A08 - A14 P": ("H@A08", "A14 P")}
CHECK_CMPS = {"H@A08 - U@A08": ("H@A08", "U@A08"), "U@A08 - Base": ("U@A08", "Base"), "H@A08 - Base": ("H@A08", "Base"), "H own-cal - U own-cal": ("H own-cal", "U own-cal")}


def a08_params(ctx: thr.Ctx, s: int) -> dict:
    p = ctx.vh["pinned"]["A08_choices"]
    if sha256_file(ROOT / p["path"]) != p["sha256"]:
        raise SystemExit("A08's committed choices differ from their pin")
    return json.loads((ROOT / p["path"]).read_text())["choices"][f"all_s{s}"]["params"]


def r_reference(s: int) -> tuple[Path, dict, dict, dict]:
    """The R control of seed s: A10's R_s0/R_s2 or A18's R_s1 (saved development outputs, committed B3 setting, stored rows)."""
    root, screen = (ROOT / "runs/video_switch_head_v1", A10_SCREEN) if s in (0, 2) else (ROOT / "runs/video_ensemble_teacher_v1", A18_SCREEN3)
    man = json.loads((root / "train" / f"R_s{s}" / "run_manifest.json").read_text())
    return root, man, bc.old_params(f"R_s{s}"), json.loads(screen.read_text())["systems"][f"R + B3 s{s}"]


def new_models(ctx: thr.Ctx, seeds: list[int]) -> dict:
    out = {}
    for s in seeds:
        for arm in ARMS:
            key = f"{arm}_s{s}"; man = json.loads((ctx.out / "train" / key / "run_manifest.json").read_text())
            fz = ROOT / "models_frozen" / "video_hard_replay_v1" / key / "final.pt"; w = fz if fz.is_file() else Path(man["final_checkpoint"])
            if sha256_file(w) != man["final_sha256"]:
                raise SystemExit(f"{key}: the weights differ from their run record")
            out[key] = {"weights": str(w), "sha256": man["final_sha256"], "identity": man["identity"], "seed": s, "arm": arm, "frozen": fz.is_file()}
    return out


def saved(path: Path, ident: str, W: dict) -> np.ndarray | None:
    if not path.is_file():
        return None
    with np.load(path) as z:
        same = str(z["identity"]) == ident and np.array_equal(z["seq"], W["seq"]) and np.array_equal(z["centre"], W["centre"])
        return z["logits"].astype(np.float32) if same else None


def load(mr: dict, device):
    m = vsh.make_model(torch.load(mr["weights"], map_location="cpu", weights_only=True)["model"], device).eval()
    for q in m.parameters():
        q.requires_grad_(False)
    return m


def infer_set(models: dict, W: dict, refs: list, fp, where: Path, device, workers: int, acc: dict) -> dict:
    """Saved outputs reused by run identity and window plan; otherwise one frame decode for the set and one frozen pass each."""
    out, frames = {}, None; orders = np.broadcast_to(np.arange(16, dtype=np.int64), W["s"].shape).copy()
    for key, mr in models.items():
        p = where / f"{key}.npz"; lg = saved(p, mr["identity"], W)
        if lg is None:
            if frames is None:
                t0 = time.time(); frames = rr.decode_frames(refs, fp, workers); acc["decode_seconds"] += time.time() - t0
            m = load(mr, device); lg, _, sec = rsh.infer(m, W, orders, frames, device); acc["gpu_seconds"] += sec; del m; torch.cuda.empty_cache()
            p.parent.mkdir(parents=True, exist_ok=True); np.savez(p, identity=mr["identity"], weights_sha256=mr["sha256"], seq=W["seq"], centre=W["centre"], logits=lg.astype(np.float32))
        out[key] = lg.astype(np.float32); print(f"{where.name} {key}: outputs ready", flush=True)
    return out


def window_meta(seqs: list[dict], W: dict) -> dict:
    """Fixed window groups from the references: 16-frame labels holding two or more views; a target in a true run <= 16 samples."""
    rl = [phr.run_lengths(np.asarray(q["labels"])) for q in seqs]
    mixed = np.asarray([len(set(np.asarray(x).tolist())) >= 2 for x in W["labels16"]]); short = np.asarray([rl[int(q)][int(c)] <= phr.SHORT for q, c in zip(W["seq"], W["centre"])])
    return {"mixed": mixed, "short": short, "label": np.asarray(W["label"], np.int64)}


def window_groups(lg: np.ndarray, meta: dict, a08_wrong: np.ndarray, is_new: bool) -> dict:
    lp = torch.log_softmax(torch.from_numpy(lg), -1).numpy(); t = meta["label"]; ce = -lp[np.arange(len(t)), t]; ok = lp.argmax(1) == t
    groups = {"all": np.ones(len(t), bool), "mixed": meta["mixed"], "pure": ~meta["mixed"], "short_run": meta["short"], "other_run": ~meta["short"],
              "A08_wrong": a08_wrong, "A08_right": ~a08_wrong, "A08_wrong_short_run": a08_wrong & meta["short"]}
    out = {g: {"windows": int(m.sum()), "accuracy": float(ok[m].mean()) if m.any() else None, "ce": float(ce[m].mean()) if m.any() else None} for g, m in groups.items()}
    if is_new:
        out["A08_wrong"]["now_right"] = int((a08_wrong & ok).sum()); out["A08_right"]["now_wrong"] = int((~a08_wrong & ~ok).sum())
    return out


def read_half(ctx: thr.Ctx, a8, h: str, seeds: list[int], device, workers: int, routers: bool, acc: dict) -> dict:
    ev, seqs = a8.half(h, rso.DevAccess({h})); ids = [q["id"] for q in seqs]; refs = rw.half_refs(seqs); at = {r: i for i, r in enumerate(refs)}; W = rr.half_windows(seqs, at, 5)
    orders = np.broadcast_to(np.arange(16, dtype=np.int64), W["s"].shape).copy(); a8sum = json.loads((ROOT / "docs/results/2026-10-16_video_within_view/summary.json").read_text())
    v01 = a8.v01_summary()["systems"]; pins = rsh.a08_output_pins()
    out = {"ev": ev, "seqs": seqs, "W": W, "logits": {}, "probs": {}, "params": {}, "counts": {"sequences": len(seqs), "windows": int(len(W["seq"]))}, "reproduced": []}
    for s in seeds:
        apath = ROOT / "runs/video_within_view_v1/evaluation" / h / f"all_s{s}.npz"
        if sha256_file(apath) != pins[h][s]:
            raise SystemExit(f"A08 all_s{s} {h}: the saved outputs differ from their pin")
        lg, _ = rr.cached_logits(apath, a8sum["runs"][f"all_s{s}"]["identity"], W, orders, seqs)
        if lg is None:
            raise SystemExit(f"A08 all_s{s} {h}: the saved outputs do not match their identity or this half's windows")
        out["logits"][f"A08_s{s}"] = lg.astype(np.float32); out["probs"][f"A08_s{s}"] = rr.per_sample(W, out["logits"][f"A08_s{s}"], seqs); out["params"][f"A08_s{s}"] = a08_params(ctx, s)
        rso.check_stored(rso.row_of(a8.cfg, decode(out["params"][f"A08_s{s}"], out["probs"][f"A08_s{s}"]), ev)["row"], a8sum["systems"][f"A + B3 s{s}"][h], f"A08 all_s{s} + B3 {h}")
        sv = a8.v01_output(h, s); ttr.check_windows(W, ids, sv, f"ordered_s{s} {h}"); op = v01[f"ordered_s{s}"]["b3_choice"]["params"]
        Po = rr.per_sample(W, sv["logits"].astype(np.float32), seqs); rows = ttr.cls_rows(a8.cfg, Po, ev, op)
        for row in ("argmax", "b3"):
            ttr.check_stored(rows[row]["row"], v01[f"ordered_s{s}"][row][h], f"ordered_s{s} {row} {h} (V01)")
        out["probs"][f"Ordered_s{s}"], out["params"][f"Ordered_s{s}"] = Po, op
        root, man, rp, want = r_reference(s); lgR = saved(root / "evaluation" / h / f"R_s{s}.npz", man["identity"], W)
        if lgR is None:
            raise SystemExit(f"R_s{s} {h}: the saved outputs do not match their run identity or this half's windows")
        out["probs"][f"R_s{s}"], out["params"][f"R_s{s}"] = rr.per_sample(W, lgR, seqs), rp
        rso.check_stored(rso.row_of(a8.cfg, decode(rp, out["probs"][f"R_s{s}"]), ev)["row"], want[h], f"R + B3 s{s} {h}")
        out["reproduced"] += [f"A08 all_s{s} + B3 {h}", f"frozen Ordered s{s} argmax and B3 {h} (V01)", f"R s{s} + B3 {h} ({'A10' if s in (0, 2) else 'A18'})"]
    fp = FramePaths(prv.frame_dirs(a8.paths, a8.rv, {r[0] for r in refs}))
    for key, lg in infer_set(new_models(ctx, seeds), W, refs, fp, ctx.out / "evaluation" / h, device, workers, acc).items():
        out["logits"][key] = lg; out["probs"][key] = rr.per_sample(W, lg, seqs)
    out["meta"] = window_meta(seqs, W)
    if routers and (rs := [s for s in seeds if s in (0, 2)]):
        c17 = trt.Ctx(load_cfg(SimpleNamespace(config=A17_CONFIG, set=[])), A17_CONFIG); P17 = rrt.read_half(c17, a8, h, rs)
        if [q["id"] for q in P17["seqs"]] != ids:
            raise SystemExit("A17's half differs in order")
        ch17 = json.loads((ROOT / "runs/video_router_trajectory_v1/evaluation/choices.json").read_text())["choices"]; a14 = json.loads(rrd.A14_SCREEN.read_text())["systems"]; out["routed"] = {}
        for s in rs:
            out["routed"][f"A14 P s{s}"] = rrd.routed(P17, s, rrt.accept("A14 P", P17["seed"][s], ch17, rrt.a15_choices(), s))
            rso.check_stored(rso.row_of(a8.cfg, out["routed"][f"A14 P s{s}"], ev)["row"], a14[f"P router s{s}"][h], f"A14 P s{s} {h}"); out["reproduced"].append(f"A14 P s{s} {h}")
    return out


def book(ctx: thr.Ctx, what: str, started: str, total: float, acc: dict, workers: int, ok: bool, note: str) -> None:
    """A19's split: GPU inference seconds as GPU; the main process outside them and the extra decode workers as CPU."""
    g = min(acc["gpu_seconds"], total)
    thr.ledger_append(ctx.ledger, f"{what} (frozen inference, GPU inference seconds)", "gpu", started, g, ok, note)
    thr.ledger_append(ctx.ledger, f"{what} (CPU: main process outside GPU + {workers - 1} more decode workers)", "cpu", started, (total - g) + acc["decode_seconds"] * (workers - 1), ok, note)


def choices_name(seeds: list[int]) -> str:
    return "choices_seed1.json" if seeds == [1] else "choices.json"


def calibrate(ctx: thr.Ctx, seeds: list[int], workers: int) -> None:
    t0, started, ok, note, acc = time.time(), thr.now(), False, "", {"gpu_seconds": 0.0, "decode_seconds": 0.0}
    try:
        a8 = rsh.a08_ctx(); rw.check_grid(a8); device = torch.device("cuda"); thr.determinism(); P = read_half(ctx, a8, "calibration", seeds, device, workers, False, acc); ch = {}
        for s in seeds:
            for arm in ARMS:
                c = rcs.b3_choose(a8.cfg, rso.DIAG, P["probs"][f"{arm}_s{s}"], P["ev"]); ch[f"{arm}_s{s}|own"] = {x: c[x] for x in ("status", "params", "n_feasible", "grid")}
                print(f"{arm}_s{s}: own B3 {c['status']} {c['params']} ({c['n_feasible']} feasible)", flush=True)
        rec = {"stage": "calibrate", "seeds": seeds, "config_sha256": sha256_file(ROOT / ctx.config), "primary": {f"s{s}": a08_params(ctx, s) for s in seeds}, "sensitivity_choices": ch,
               "models": {k: {x: v[x] for x in ("sha256", "identity", "frozen")} for k, v in new_models(ctx, seeds).items()},
               "notes": {"primary": "each seed's committed A08 B3 numeric setting (pinned, never re-chosen)", "sensitivity": "never used for promotion or to change the primary"}, "time": thr.now()}
        flat = json.loads(json.dumps(rec, default=str)); assert_no_ids(flat, rw.ids_of(a8, {"calibration": P["seqs"]}), "choices")
        for p in (ctx.out / "evaluation" / choices_name(seeds), ctx.results / choices_name(seeds)):
            p.parent.mkdir(parents=True, exist_ok=True); thr.write_json(p, flat)
        ok = True; note = f"GPU {acc['gpu_seconds']:.0f} s, decode {acc['decode_seconds']:.0f} s"
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        book(ctx, f"evaluate calibrate seeds {seeds}", started, time.time() - t0, acc, workers, ok, note)


def systems_of(a8, P: dict, s: int, ch: dict) -> dict:
    ap = P["params"][f"A08_s{s}"]; am = lambda k: {q: v.argmax(1) for q, v in P["probs"][k].items()}  # noqa: E731
    items = [("Base", decode(ap, P["probs"][f"A08_s{s}"])), ("A08 argmax", am(f"A08_s{s}")), ("Ordered + B3", decode(P["params"][f"Ordered_s{s}"], P["probs"][f"Ordered_s{s}"])),
             ("R + B3", decode(P["params"][f"R_s{s}"], P["probs"][f"R_s{s}"]))]
    for arm in ARMS:
        k = f"{arm}_s{s}"; items += [(f"{arm}@A08", decode(ap, P["probs"][k])), (f"{arm} argmax", am(k)), (f"{arm} own-cal", decode(ch[f"{k}|own"]["params"], P["probs"][k]))]
    if f"A14 P s{s}" in P.get("routed", {}):
        items.append(("A14 P", P["routed"][f"A14 P s{s}"]))
    return {f"{n} s{s}": rso.row_of(a8.cfg, p, P["ev"]) for n, p in items}


def check_logits(ctx: thr.Ctx, models: dict, seqs: list[dict], W: dict, fp, device, workers: int, acc: dict) -> dict:
    """The check pool in chunks of 150 sequences (A19's pipeline): one decode per chunk, every pending model on it."""
    where = ctx.out / "evaluation" / "check"; out = {k: lg for k, mr in models.items() if (lg := saved(where / f"{k}.npz", mr["identity"], W)) is not None}
    todo = {k: mr for k, mr in models.items() if k not in out}
    if not todo:
        return out
    M = {k: load(mr, device) for k, mr in todo.items()}; parts = {k: [] for k in todo}; seq_, cen_ = [], []
    for c0 in range(0, len(seqs), CHUNK):
        chunk = seqs[c0:c0 + CHUNK]; refs = rw.half_refs(chunk); at = {r: i for i, r in enumerate(refs)}; Wc = rr.half_windows(chunk, at, 5)
        orders = np.broadcast_to(np.arange(16, dtype=np.int64), Wc["s"].shape).copy(); t0 = time.time(); frames = rr.decode_frames(refs, fp, workers); acc["decode_seconds"] += time.time() - t0
        for k, m in M.items():
            lg, _, sec = rsh.infer(m, Wc, orders, frames, device); acc["gpu_seconds"] += sec; parts[k].append(lg.astype(np.float32))
        seq_.append(Wc["seq"] + c0); cen_.append(Wc["centre"]); del frames
        print(f"check chunk {c0 // CHUNK + 1} of {-(-len(seqs) // CHUNK)}: GPU {acc['gpu_seconds']:.0f} s, decode {acc['decode_seconds']:.0f} s", flush=True)
    if not (np.array_equal(np.concatenate(seq_), W["seq"]) and np.array_equal(np.concatenate(cen_), W["centre"])):
        raise SystemExit("the chunked check windows differ from the check window plan")
    where.mkdir(parents=True, exist_ok=True)
    for k, mr in todo.items():
        out[k] = np.concatenate(parts[k]); np.savez(where / f"{k}.npz", identity=mr["identity"], weights_sha256=mr["sha256"], seq=W["seq"], centre=W["centre"], logits=out[k])
    del M; torch.cuda.empty_cache()
    return out


def check_pool(ctx: thr.Ctx, a8, seeds: list[int], ch: dict, device, workers: int, acc: dict) -> dict:
    """A15's 600 multi + 200 native check sequences once: Base from the pinned A08 pool logits (A19's rows reproduced)."""
    bctx = thr._B3Ctx(); P = bc.with_index(bctx, bc.pool(bctx)); cidx = bc.members(P, "check"); seqs = [P["seqs"][i] for i in cidx]; vids = {r[0] for q in seqs for r in q["refs"]}
    if vids - set(prv.allowlist(ctx.a10.rv, ROOT)):
        raise SystemExit("a check recording lies outside the training allowlist")
    refs = rw.half_refs(seqs); at = {r: i for i, r in enumerate(refs)}; W = rr.half_windows(seqs, at, 5); keep = np.isin(P["W"]["seq"], cidx)
    if not all(np.array_equal(a, b) for a, b in ((np.asarray(cidx)[W["seq"]], P["W"]["seq"][keep]), (W["centre"], P["W"]["centre"][keep]), (W["marker"], P["W"]["marker"][keep]))):
        raise SystemExit("the check windows differ from the pool's window plan")
    fp = FramePaths(prv.frame_dirs(ctx.a10.paths, ctx.a10.rv, vids)); lgs = check_logits(ctx, new_models(ctx, seeds), seqs, W, fp, device, workers, acc)
    ev = bc.evalset(P, cidx, "check"); meta = window_meta(seqs, W); a19 = json.loads(A19_SCREEN.read_text())["systems"]; out = {"systems": {}, "groups": {}, "reproduced": []}
    for s in seeds:
        la = bc.pool_logits(bctx, P, f"A08_s{s}")[keep]; wrong = la.argmax(1) != meta["label"]; ap = a08_params(ctx, s)
        items = [("Base", la, ap)] + [(f"{a}@A08", lgs[f"{a}_s{s}"], ap) for a in ARMS] + [(f"{a} own-cal", lgs[f"{a}_s{s}"], ch[f"{a}_s{s}|own"]["params"]) for a in ARMS]
        for name, lg, prm in items:
            preds = decode(prm, rr.per_sample(W, lg, seqs)); m, t = bc.evaluate_pool(a8.cfg, preds, ev); out["systems"][f"{name} s{s}"] = {"row": bce.check_row(m), "tables": t, "preds": preds}
        rso.check_stored(out["systems"][f"Base s{s}"]["row"], a19[f"A08_s{s} old"]["check"], f"A08 s{s} + B3 check (A19)"); out["reproduced"].append(f"A08 s{s} + B3 check (A19)")
        out["groups"][f"Base s{s}"] = window_groups(la, meta, wrong, False)
        for a in ARMS:
            out["groups"][f"{a} s{s}"] = window_groups(lgs[f"{a}_s{s}"], meta, wrong, True)
    out["seqs"] = [{"id": q["id"], "kind": q["kind"], "labels": np.asarray(q["labels"]), "refs": q["refs"]} for q in seqs]
    out["counts"] = {"sequences": len(seqs), "native": int(sum(q["kind"] == "native" for q in seqs)), "multi": int(sum(q["kind"] == "multi" for q in seqs)), "windows": int(len(W["seq"]))}
    return out


def promotion(rows: dict, counts: dict) -> dict:
    cells = [(s, h) for s in (0, 2) for h in HALVES]; out = {}; F = lambda n, s, h: rows[f"{n} s{s}"][h]["segment_f1_50_multi"]  # noqa: E731
    for x in ("U@A08", "H@A08"):
        dF = {f"s{s} {h}": F(x, s, h) - F("Base", s, h) for s, h in cells}; half = {h: float(np.mean([dF[f"s{s} {h}"] for s in (0, 2)])) for h in HALVES}
        dN = float(np.mean([rows[f"{x} s{s}"][h]["native_macro_f1"] - rows[f"Base s{s}"][h]["native_macro_f1"] for s, h in cells]))
        dBR = float(np.mean([rows[f"{x} s{s}"][h]["boundary_recall_multi"] - rows[f"Base s{s}"][h]["boundary_recall_multi"] for s, h in cells]))
        fs = {n: int(sum(counts[f"{n} s{s}"][h]["fn_short"] for s, h in cells)) for n in (x, "Base")}
        feas = {f"s{s}": bool(rows[f"{x} s{s}"]["calibration"]["native_macro_f1"] >= 0.980 and rows[f"{x} s{s}"]["calibration"]["boundary_recall_multi"] >= 0.95) for s in (0, 2)}
        r = {"mean_dF": float(np.mean(list(dF.values()))), "dF": dF, "half_means": half, "mean_dN": dN, "mean_dBR": dBR, "missed_short_true": fs, "calibration_feasible": feas}
        r["signal"] = bool(r["mean_dF"] >= 0.003); r["no_reversal"] = bool(min(half.values()) >= -0.005 and min(dF.values()) >= -0.015)
        r["no_harm"] = bool(dN >= -0.003 and dBR >= -0.010 and fs[x] <= fs["Base"] + 3); r["feasibility"] = bool(all(feas.values()))
        r["meets_rule"] = bool(r["signal"] and r["no_reversal"] and r["no_harm"] and r["feasibility"]); out[x] = r
    dHU = {f"s{s} {h}": F("H@A08", s, h) - F("U@A08", s, h) for s, h in cells}
    out["H - U (attribution, not a rule)"] = {"dF": dHU, "mean_dF": float(np.mean(list(dHU.values()))), "half_means": {h: float(np.mean([dHU[f"s{s} {h}"] for s in (0, 2)])) for h in HALVES}}
    out["train_seed1"] = bool(out["U@A08"]["meets_rule"] or out["H@A08"]["meets_rule"])
    return out


def compare(seqs: list[dict], systems: dict, target: str, s: int, cmps: dict, cmp_: dict, acc_: dict) -> None:
    draws = {}
    for c, (a, b) in cmps.items():
        ka, kb = f"{a} s{s}", f"{b} s{s}"
        if target not in systems.get(ka, {}) or target not in systems.get(kb, {}):
            continue
        for nm in (ka, kb):
            if nm not in draws:
                draws[nm] = st.system_draws(systems[nm][target]["tables"])
        cmp_[f"s{s}|{c}|{target}"] = st.paired(draws[ka], draws[kb]); acc_[f"s{s}|{c}|{target}"] = st.segment_accounting(seqs, systems[ka][target]["preds"], systems[kb][target]["preds"])[0]


def select(ctx: thr.Ctx, seeds: list[int], choices_sha: str, workers: int) -> None:
    t0, started, ok, note, acc = time.time(), thr.now(), False, "", {"gpu_seconds": 0.0, "decode_seconds": 0.0}
    try:
        cp = ctx.out / "evaluation" / choices_name(seeds)
        if sha256_file(cp) != choices_sha:
            raise SystemExit("the choices file differs from the committed sha256; nothing is read")
        chj = json.loads(cp.read_text()); ch = chj["sensitivity_choices"]
        if any(chj["models"][k]["sha256"] != v["sha256"] for k, v in new_models(ctx, seeds).items()):
            raise SystemExit("a model differs from the weights its choices were committed with")
        a8 = rsh.a08_ctx(); device = torch.device("cuda"); thr.determinism()
        per = {h: read_half(ctx, a8, h, seeds, device, workers, True, acc) for h in ("calibration", "selection")}; systems, groups = {}, {}
        for h in HALVES:
            P = per[h]
            for s in seeds:
                for k, v in systems_of(a8, P, s, ch).items():
                    systems.setdefault(k, {})[h] = v
                wrong = P["logits"][f"A08_s{s}"].argmax(1) != P["meta"]["label"]; groups.setdefault(f"Base s{s}", {})[h] = window_groups(P["logits"][f"A08_s{s}"], P["meta"], wrong, False)
                for a in ARMS:
                    groups.setdefault(f"{a} s{s}", {})[h] = window_groups(P["logits"][f"{a}_s{s}"], P["meta"], wrong, True)
            print(f"{h}: rows done", flush=True)
        CK = check_pool(ctx, a8, seeds, ch, device, workers, acc); print("check: rows done", flush=True)
        for k, v in CK["systems"].items():
            systems.setdefault(k, {})["check"] = v
        for k, v in CK["groups"].items():
            groups.setdefault(k, {})["check"] = v
        seqs_of = {"check": CK["seqs"], **{h: per[h]["seqs"] for h in HALVES}}
        rows = {k: {t: v[t]["row"] for t in v} for k, v in systems.items()}; counts = {k: {t: rsh.counts_of(seqs_of[t], v[t]) for t in v} for k, v in systems.items() if "argmax" not in k}
        cmp_, acc_ = {}, {}
        for t, cmps in (("selection", DEV_CMPS), ("calibration", DEV_CMPS), ("check", CHECK_CMPS)):
            for s in seeds:
                compare(seqs_of[t], systems, t, s, cmps, cmp_, acc_)
            print(f"{t}: comparisons done", flush=True)
        res = {"stage": "select", "seeds": seeds, "config_sha256": sha256_file(ROOT / ctx.config), "choices_sha256": choices_sha, "systems": rows, "segment_counts": counts, "comparisons": cmp_,
               "accounting": acc_, "window_groups": groups, "reproduced": {**{h: per[h]["reproduced"] for h in HALVES}, "check": CK["reproduced"]},
               "counts": {**{h: per[h]["counts"] for h in HALVES}, "check": CK["counts"], "gpu_seconds": acc["gpu_seconds"], "decode_seconds": acc["decode_seconds"]},
               "notes": {"primary": "U/H@A08 = each seed's committed A08 B3 numeric setting", "check": "A15 router-check sessions, unseen by the replay; its recordings trained the video encoders; not a new holdout",
                         "window_groups": "accuracy and CE of the window argmax against the reference target; A08_wrong = the seed's own A08 argmax wrong on that window"},
               "access": {"targets": ["calibration", "selection", "check"], "time": started}}
        if seeds == [0, 2]:
            res["promotion"] = promotion(rows, counts)
        ids = rw.ids_of(a8, {h: per[h]["seqs"] for h in HALVES}) | {q["id"] for q in CK["seqs"] if q["kind"] == "native"} | {r[0] for q in CK["seqs"] for r in q["refs"]}
        flat = json.loads(json.dumps(res, default=str)); assert_no_ids(flat, ids, "select"); name = "screen_seed1.json" if seeds == [1] else "screen.json"
        for p in (ctx.out / "evaluation" / name, ctx.results / name):
            p.parent.mkdir(parents=True, exist_ok=True); thr.write_json(p, flat)
        if "promotion" in flat:
            thr.write_json(ctx.out / "evaluation" / "decision.json", {**flat["promotion"], "time": thr.now()})
            print(json.dumps({k: {x: flat["promotion"][k][x] for x in ("mean_dF", "half_means", "mean_dN", "mean_dBR", "missed_short_true", "calibration_feasible", "meets_rule")} for k in ("U@A08", "H@A08")}, indent=1),
                  json.dumps(flat["promotion"]["H - U (attribution, not a rule)"]), "train_seed1:", flat["promotion"]["train_seed1"], flush=True)
        ok = True; note = f"GPU {acc['gpu_seconds']:.0f} s, decode {acc['decode_seconds']:.0f} s"
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        book(ctx, f"evaluate select + check pool seeds {seeds}", started, time.time() - t0, acc, workers, ok, note)


def training_summary(ctx: thr.Ctx, seeds: list[int]) -> None:
    """Training data only (no evaluation set): base-12 and replay-4 CE / accuracy per run, by phase, bucket and H = U."""
    t0, started, ok, note, out = time.time(), thr.now(), False, "", {}
    try:
        for s in seeds:
            same = np.all(ctx.manifest("U", s)[0]["pool_rows"] == ctx.manifest("H", s)[0]["pool_rows"], axis=1); u = np.arange(ctx.total)
            parts = {"all": u >= 0, "encoder frozen (u < warm-up)": u < ctx.warmup, "last 270 updates": u >= ctx.total - 270, "short bucket (even u)": u % 2 == 0,
                     "non-short bucket (odd u)": u % 2 == 1, "H = U updates": same, "H != U updates": ~same}
            for arm in ARMS:
                rec = [json.loads(x) for x in (ctx.out / "train" / f"{arm}_s{s}" / "updates.jsonl").read_text().splitlines()]
                if [r["u"] for r in rec] != u.tolist():
                    raise SystemExit(f"{arm}_s{s}: the update log is incomplete")
                X = {k: np.asarray([r[k] for r in rec], float) for k in ("ce_base", "ce_replay", "acc_base", "acc_replay", "kl_base", "kl_replay")}
                out[f"{arm}_s{s}"] = {p: {"updates": int(m.sum()), **{k: float(v[m].mean()) for k, v in X.items()}} for p, m in parts.items()}
        rec = {"stage": "training summary", "seeds": seeds, "runs": out, "notes": "training windows only; U and H replay windows differ on H != U updates (H's are harder by A08 CE), so their replay CE is not a matched comparison there", "time": thr.now()}
        for p in (ctx.out / "training_summary.json", ctx.results / "training_summary.json"):
            thr.write_json(p, rec)
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        thr.ledger_append(ctx.ledger, f"training summary seeds {seeds} (CPU)", "cpu", started, time.time() - t0, ok, note)


def background(ctx: thr.Ctx) -> None:
    """Post hoc, after select and outside every rule: A10 R_s0/R_s2 (the same recipe without replay) window groups on both
    development halves from their saved outputs, against the same A08-wrong groups. The check pool is not read again."""
    t0, started, ok, note, out = time.time(), thr.now(), False, "", {}
    try:
        a8 = rsh.a08_ctx(); a8sum = json.loads((ROOT / "docs/results/2026-10-16_video_within_view/summary.json").read_text()); pins = rsh.a08_output_pins()
        scr = json.loads((ctx.out / "evaluation" / "screen.json").read_text())["window_groups"]
        for h in HALVES:
            ev, seqs = a8.half(h, rso.DevAccess({h})); refs = rw.half_refs(seqs); at = {r: i for i, r in enumerate(refs)}; W = rr.half_windows(seqs, at, 5)
            orders = np.broadcast_to(np.arange(16, dtype=np.int64), W["s"].shape).copy(); meta = window_meta(seqs, W)
            for s in (0, 2):
                apath = ROOT / "runs/video_within_view_v1/evaluation" / h / f"all_s{s}.npz"
                if sha256_file(apath) != pins[h][s]:
                    raise SystemExit(f"A08 all_s{s} {h}: the saved outputs differ from their pin")
                la, _ = rr.cached_logits(apath, a8sum["runs"][f"all_s{s}"]["identity"], W, orders, seqs); wrong = la.argmax(1) != meta["label"]
                if window_groups(la, meta, wrong, False) != scr[f"Base s{s}"][h]:
                    raise SystemExit(f"Base s{s} {h}: the window groups differ from the select stage")
                root, man, _, _ = r_reference(s); lgR = saved(root / "evaluation" / h / f"R_s{s}.npz", man["identity"], W)
                if lgR is None:
                    raise SystemExit(f"R_s{s} {h}: the saved outputs do not match this half's windows")
                out.setdefault(f"R s{s}", {})[h] = window_groups(lgR, meta, wrong, True)
        rec = {"stage": "background (post hoc, no decision)", "groups": out, "notes": "A10 R = A10's 12-window recipe from the same A08 start without replay; development halves only; the check pool was not read again", "time": thr.now()}
        for p in (ctx.out / "evaluation" / "background_windows.json", ctx.results / "background_windows.json"):
            thr.write_json(p, rec)
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        thr.ledger_append(ctx.ledger, "background R window groups, post hoc (CPU)", "cpu", started, time.time() - t0, ok, note)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--stage", choices=("training", "calibrate", "select", "background"), required=True); ap.add_argument("--choices-sha")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 2]); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(); a.set = []; ctx = thr.Ctx(load_cfg(a), a.config)
    if a.stage == "training":
        return training_summary(ctx, a.seeds)
    if a.stage == "background":
        return background(ctx)
    if a.stage == "calibrate":
        return calibrate(ctx, a.seeds, a.workers)
    select(ctx, a.seeds, a.choices_sha, a.workers)


if __name__ == "__main__":
    main()
