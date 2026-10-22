#!/usr/bin/env python
"""Evaluate A24 (configs/experiments/video_context_relation_v1.yaml): OtherRel and SameRel decoded at their seed's committed
A08 B3 setting (primary) on the original 16-frame windows, with A23's frozen Other / Same (the zero-relation references),
Long10 R, A14 P and A20 H as frozen rows; one own-calibration choice and argmax (sensitivity); window flips and
boundary errors by fixed definitions; both development halves and once A15's check pool.

  --stage training                training windows only: per-run CE / Rel / accuracy, matched OtherRel - SameRel
  --stage calibrate               Other/Same inference on calibration (outputs saved); sensitivity choices
  --stage select --choices-sha S  selection inference, the check pool once, rows, comparisons, window groups, promotion
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
import scripts.prepare_raw_video_context as prv  # noqa: E402
import scripts.report_center_supervision as rcs  # noqa: E402
import scripts.report_hard_replay as rhr  # noqa: E402
import scripts.report_raw_video_context as rr  # noqa: E402
import scripts.report_router_data as rrd  # noqa: E402
import scripts.report_router_trajectory as rrt  # noqa: E402
import scripts.report_segment_objective as rso  # noqa: E402
import scripts.report_video_switch_head as rsh  # noqa: E402
import scripts.report_video_within_view as rw  # noqa: E402
import scripts.train_router_trajectory as trt  # noqa: E402
import scripts.train_context_intervention as tci  # noqa: E402
import scripts.train_context_relation as ttd  # noqa: E402
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
ARMS = tuple(ttd.ARMS)
A23_ROOT = ROOT / "runs/video_context_intervention_v1"
CHUNK = 150
A20_ROOT = ROOT / "runs/video_hard_replay_v1"
DEV_CMPS = {"OtherRel@A08 - SameRel@A08": ("OtherRel@A08", "SameRel@A08"), "OtherRel@A08 - A23 Other@A08": ("OtherRel@A08", "A23 Other@A08"), "SameRel@A08 - A23 Same@A08": ("SameRel@A08", "A23 Same@A08"),
            "OtherRel@A08 - Base": ("OtherRel@A08", "Base"), "SameRel@A08 - Base": ("SameRel@A08", "Base"), "A23 Other@A08 - A23 Same@A08": ("A23 Other@A08", "A23 Same@A08"),
            "OtherRel own-cal - SameRel own-cal": ("OtherRel own-cal", "SameRel own-cal"), "OtherRel@A08 - A14 P": ("OtherRel@A08", "A14 P"), "SameRel@A08 - A14 P": ("SameRel@A08", "A14 P"),
            "OtherRel@A08 - A20 H@A08": ("OtherRel@A08", "A20 H@A08"), "SameRel@A08 - A20 H@A08": ("SameRel@A08", "A20 H@A08")}
CHECK_CMPS = {k: DEV_CMPS[k] for k in ("OtherRel@A08 - SameRel@A08", "OtherRel@A08 - A23 Other@A08", "SameRel@A08 - A23 Same@A08", "OtherRel@A08 - Base", "SameRel@A08 - Base",
                                       "A23 Other@A08 - A23 Same@A08", "OtherRel own-cal - SameRel own-cal")}


def a08_params(ctx: ttd.Ctx, s: int) -> dict:
    p = ctx.a23.vc["pinned"]["A08_choices"]
    if sha256_file(ROOT / p["path"]) != p["sha256"]:
        raise SystemExit("A08's committed choices differ from their pin")
    return json.loads((ROOT / p["path"]).read_text())["choices"][f"all_s{s}"]["params"]


def new_models(ctx: ttd.Ctx, seeds: list[int]) -> dict:
    out = {}
    for s in seeds:
        for arm in ARMS:
            key = f"{arm}_s{s}"; man = json.loads((ctx.out / "train" / key / "run_manifest.json").read_text())
            fz = ROOT / "models_frozen" / "video_context_relation_v1" / key / "final.pt"; w = fz if fz.is_file() else Path(man["final_checkpoint"])
            if sha256_file(w) != man["final_sha256"]:
                raise SystemExit(f"{key}: the weights differ from their run record")
            out[key] = {"weights": str(w), "sha256": man["final_sha256"], "identity": man["identity"], "seed": s, "arm": arm, "frozen": fz.is_file()}
    return out


def load(mr: dict, device):
    m = vsh.make_model(torch.load(mr["weights"], map_location="cpu", weights_only=True)["model"], device).eval()
    for q in m.parameters():
        q.requires_grad_(False)
    return m


def infer_orig(models: dict, seqs: list[dict], W: dict, fp, where: Path, device, workers: int, acc: dict) -> dict:
    """Saved outputs reused by run identity and window plan; otherwise chunks of CHUNK sequences: one decode of the
    original 16-frame windows' frames, every pending model on them (one forward per window)."""
    out = {k: lg for k, mr in models.items() if (lg := rhr.saved(where / f"{k}.npz", mr["identity"], W)) is not None}; todo = {k: mr for k, mr in models.items() if k not in out}
    if not todo:
        return out
    M = {k: load(mr, device) for k, mr in todo.items()}; parts = {k: [] for k in todo}; seq_, cen_ = [], []
    for c0 in range(0, len(seqs), CHUNK):
        chunk = seqs[c0:c0 + CHUNK]; refs = rw.half_refs(chunk); at = {r: i for i, r in enumerate(refs)}; Wc = rr.half_windows(chunk, at, 5)
        orders = np.broadcast_to(np.arange(16, dtype=np.int64), Wc["s"].shape).copy(); t0 = time.time(); frames = rr.decode_frames(refs, fp, workers); acc["decode_seconds"] += time.time() - t0
        for k, m in M.items():
            lg, _, sec = rsh.infer(m, Wc, orders, frames, device); acc["gpu_seconds"] += sec; acc["windows"] += len(Wc["seq"]); parts[k].append(lg.astype(np.float32))
        seq_.append(Wc["seq"] + c0); cen_.append(Wc["centre"]); del frames
        print(f"{where.name} chunk {c0 // CHUNK + 1} of {-(-len(seqs) // CHUNK)}: GPU {acc['gpu_seconds']:.0f} s, decode {acc['decode_seconds']:.0f} s", flush=True)
    if not (np.array_equal(np.concatenate(seq_), W["seq"]) and np.array_equal(np.concatenate(cen_), W["centre"])):
        raise SystemExit(f"{where.name}: the chunked windows differ from the window plan")
    where.mkdir(parents=True, exist_ok=True)
    for k, mr in todo.items():
        out[k] = np.concatenate(parts[k]); np.savez(where / f"{k}.npz", identity=mr["identity"], weights_sha256=mr["sha256"], seq=W["seq"], centre=W["centre"], logits=out[k])
    del M; torch.cuda.empty_cache()
    return out


def read_half(ctx: ttd.Ctx, a8, h: str, seeds: list[int], device, workers: int, routers: bool, acc: dict) -> dict:
    ev, seqs = a8.half(h, rso.DevAccess({h})); ids = [q["id"] for q in seqs]; refs = rw.half_refs(seqs); at = {r: i for i, r in enumerate(refs)}; W = rr.half_windows(seqs, at, 5)
    orders = np.broadcast_to(np.arange(16, dtype=np.int64), W["s"].shape).copy(); a8sum = json.loads((ROOT / "docs/results/2026-10-16_video_within_view/summary.json").read_text())
    v01 = a8.v01_summary()["systems"]; pins = rsh.a08_output_pins(); a20 = json.loads((ROOT / ctx.a23.vc["pinned"]["A20_screen"]["path"]).read_text())["systems"]
    a18 = json.loads((ROOT / ctx.a23.vc["pinned"]["A18_screen"]["path"]).read_text())["systems"]
    out = {"ev": ev, "seqs": seqs, "W": W, "logits": {}, "probs": {}, "params": {}, "counts": {"sequences": len(seqs), "windows": int(len(W["seq"]))}, "reproduced": []}
    for s in seeds:
        apath = ROOT / "runs/video_within_view_v1/evaluation" / h / f"all_s{s}.npz"
        if sha256_file(apath) != pins[h][s]:
            raise SystemExit(f"A08 all_s{s} {h}: the saved outputs differ from their pin")
        lg, _ = rr.cached_logits(apath, a8sum["runs"][f"all_s{s}"]["identity"], W, orders, seqs)
        if lg is None:
            raise SystemExit(f"A08 all_s{s} {h}: the saved outputs do not match their identity or this half's windows")
        out["logits"][f"A08_s{s}"] = lg.astype(np.float32); out["probs"][f"A08_s{s}"] = rr.per_sample(W, out["logits"][f"A08_s{s}"], seqs); out["params"][f"A08_s{s}"] = ap = a08_params(ctx, s)
        rso.check_stored(rso.row_of(a8.cfg, decode(ap, out["probs"][f"A08_s{s}"]), ev)["row"], a8sum["systems"][f"A + B3 s{s}"][h], f"A08 all_s{s} + B3 {h}")
        sv = a8.v01_output(h, s); ttr.check_windows(W, ids, sv, f"ordered_s{s} {h}"); op = v01[f"ordered_s{s}"]["b3_choice"]["params"]
        Po = rr.per_sample(W, sv["logits"].astype(np.float32), seqs); rows = ttr.cls_rows(a8.cfg, Po, ev, op)
        for row in ("argmax", "b3"):
            ttr.check_stored(rows[row]["row"], v01[f"ordered_s{s}"][row][h], f"ordered_s{s} {row} {h} (V01)")
        out["probs"][f"Ordered_s{s}"], out["params"][f"Ordered_s{s}"] = Po, op
        root, man, rp, want = rhr.r_reference(s)
        if man["final_sha256"] != ctx.a23.vc["pinned"]["Long10_R_weights"][s]:
            raise SystemExit(f"Long10 R_s{s}: the run record differs from the pinned weights")
        lgR = rhr.saved(root / "evaluation" / h / f"R_s{s}.npz", man["identity"], W)
        if lgR is None:
            raise SystemExit(f"R_s{s} {h}: the saved outputs do not match their run identity or this half's windows")
        out["logits"][f"R_s{s}"] = lgR; out["probs"][f"R_s{s}"], out["params"][f"R_s{s}"] = rr.per_sample(W, lgR, seqs), rp
        rso.check_stored(rso.row_of(a8.cfg, decode(rp, out["probs"][f"R_s{s}"]), ev)["row"], want[h], f"R + B3 s{s} {h}")
        if s in (0, 2):
            rso.check_stored(rso.row_of(a8.cfg, decode(ap, out["probs"][f"R_s{s}"]), ev)["row"], a18[f"R + B3@A08 s{s}"][h], f"Long10 R@A08 s{s} {h} (A18)")
            for arm in ("U", "H"):
                man20 = json.loads((A20_ROOT / "train" / f"{arm}_s{s}" / "run_manifest.json").read_text()); lgA = rhr.saved(A20_ROOT / "evaluation" / h / f"{arm}_s{s}.npz", man20["identity"], W)
                if lgA is None:
                    raise SystemExit(f"A20 {arm}_s{s} {h}: the saved outputs do not match")
                out["probs"][f"A20 {arm}_s{s}"] = rr.per_sample(W, lgA, seqs)
                rso.check_stored(rso.row_of(a8.cfg, decode(ap, out["probs"][f"A20 {arm}_s{s}"]), ev)["row"], a20[f"{arm}@A08 s{s}"][h], f"A20 {arm}@A08 s{s} {h}")
            a23 = json.loads((ROOT / ctx.vr["pinned"]["A23_screen"]["path"]).read_text())["systems"]
            for arm in ("Other", "Same"):
                man23 = json.loads((A23_ROOT / "train" / f"{arm}_s{s}" / "run_manifest.json").read_text())
                if man23["final_sha256"] != ctx.vr["pinned"]["A23_weights"][f"{arm}_s{s}"]:
                    raise SystemExit(f"A23 {arm}_s{s}: the run record differs from the pinned weights")
                lgB = rhr.saved(A23_ROOT / "evaluation" / h / f"{arm}_s{s}.npz", man23["identity"], W)
                if lgB is None:
                    raise SystemExit(f"A23 {arm}_s{s} {h}: the saved outputs do not match")
                out["logits"][f"A23 {arm}_s{s}"] = lgB; out["probs"][f"A23 {arm}_s{s}"] = rr.per_sample(W, lgB, seqs)
                rso.check_stored(rso.row_of(a8.cfg, decode(ap, out["probs"][f"A23 {arm}_s{s}"]), ev)["row"], a23[f"{arm}@A08 s{s}"][h], f"A23 {arm}@A08 s{s} {h}")
            out["reproduced"] += [f"Long10 R@A08 s{s} {h} (A18)", f"A20 U/H@A08 s{s} {h}", f"A23 Other/Same@A08 s{s} {h}"]
        out["reproduced"] += [f"A08 all_s{s} + B3 {h}", f"frozen Ordered s{s} argmax and B3 {h} (V01)", f"R s{s} + B3 {h}"]
    fp = FramePaths(prv.frame_dirs(a8.paths, a8.rv, {r[0] for r in refs}))
    for key, lg in infer_orig(new_models(ctx, seeds), seqs, W, fp, ctx.out / "evaluation" / h, device, workers, acc).items():
        out["logits"][key] = lg; out["probs"][key] = rr.per_sample(W, lg, seqs)
    out["meta"] = rhr.window_meta(seqs, W)
    if routers and (rs := [s for s in seeds if s in (0, 2)]):
        c17 = trt.Ctx(load_cfg(SimpleNamespace(config=rhr.A17_CONFIG, set=[])), rhr.A17_CONFIG); P17 = rrt.read_half(c17, a8, h, rs)
        if [q["id"] for q in P17["seqs"]] != ids:
            raise SystemExit("A17's half differs in order")
        ch17 = json.loads((ROOT / "runs/video_router_trajectory_v1/evaluation/choices.json").read_text())["choices"]; a14 = json.loads(rrd.A14_SCREEN.read_text())["systems"]; out["routed"] = {}
        for s in rs:
            out["routed"][f"A14 P s{s}"] = rrd.routed(P17, s, rrt.accept("A14 P", P17["seed"][s], ch17, rrt.a15_choices(), s))
            rso.check_stored(rso.row_of(a8.cfg, out["routed"][f"A14 P s{s}"], ev)["row"], a14[f"P router s{s}"][h], f"A14 P s{s} {h}"); out["reproduced"].append(f"A14 P s{s} {h}")
    return out


def book(ctx: ttd.Ctx, what: str, started: str, total: float, acc: dict, workers: int, ok: bool, note: str) -> None:
    """A19's split: GPU inference seconds as GPU; the main process outside them and the extra decode workers as CPU."""
    g = min(acc["gpu_seconds"], total)
    ttd.ledger_append(ctx.ledger, f"{what} (frozen inference, GPU inference seconds)", "gpu", started, g, ok, note)
    ttd.ledger_append(ctx.ledger, f"{what} (CPU: main process outside GPU + {workers - 1} more decode workers)", "cpu", started, (total - g) + acc["decode_seconds"] * (workers - 1), ok, note)


def choices_name(seeds: list[int]) -> str:
    return "choices_seed1.json" if seeds == [1] else "choices.json"


def new_acc() -> dict:
    return {"gpu_seconds": 0.0, "decode_seconds": 0.0, "windows": 0}


def calibrate(ctx: ttd.Ctx, seeds: list[int], workers: int) -> None:
    t0, started, ok, note, acc = time.time(), ttd.now(), False, "", new_acc()
    try:
        a8 = rsh.a08_ctx(); rw.check_grid(a8); device = torch.device("cuda"); ttd.determinism(); P = read_half(ctx, a8, "calibration", seeds, device, workers, False, acc); ch = {}
        for s in seeds:
            for arm in ARMS:
                c = rcs.b3_choose(a8.cfg, rso.DIAG, P["probs"][f"{arm}_s{s}"], P["ev"]); ch[f"{arm}_s{s}|own"] = {x: c[x] for x in ("status", "params", "n_feasible", "grid")}
                print(f"{arm}_s{s}: own B3 {c['status']} {c['params']} ({c['n_feasible']} feasible)", flush=True)
        rec = {"stage": "calibrate", "seeds": seeds, "config_sha256": sha256_file(ROOT / ctx.config), "primary": {f"s{s}": a08_params(ctx, s) for s in seeds}, "sensitivity_choices": ch,
               "models": {k: {x: v[x] for x in ("sha256", "identity", "frozen")} for k, v in new_models(ctx, seeds).items()},
               "notes": {"primary": "each seed's committed A08 B3 numeric setting (pinned, never re-chosen)", "sensitivity": "never used for promotion or to change the primary"}, "time": ttd.now()}
        flat = json.loads(json.dumps(rec, default=str)); assert_no_ids(flat, rw.ids_of(a8, {"calibration": P["seqs"]}), "choices")
        for p in (ctx.out / "evaluation" / choices_name(seeds), ctx.results / choices_name(seeds)):
            p.parent.mkdir(parents=True, exist_ok=True); ttd.write_json(p, flat)
        ok = True; note = f"GPU {acc['gpu_seconds']:.0f} s, decode {acc['decode_seconds']:.0f} s"
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        book(ctx, f"evaluate calibrate seeds {seeds}", started, time.time() - t0, acc, workers, ok, note)


def systems_of(a8, P: dict, s: int, ch: dict) -> dict:
    ap = P["params"][f"A08_s{s}"]; am = lambda k: {q: v.argmax(1) for q, v in P["probs"][k].items()}  # noqa: E731
    items = [("Base", decode(ap, P["probs"][f"A08_s{s}"])), ("A08 argmax", am(f"A08_s{s}")), ("Ordered + B3", decode(P["params"][f"Ordered_s{s}"], P["probs"][f"Ordered_s{s}"])),
             ("R + B3", decode(P["params"][f"R_s{s}"], P["probs"][f"R_s{s}"])), ("Long10 R@A08", decode(ap, P["probs"][f"R_s{s}"])), ("Long10 R argmax", am(f"R_s{s}"))]
    for arm in ARMS:
        k = f"{arm}_s{s}"; items += [(f"{arm}@A08", decode(ap, P["probs"][k])), (f"{arm} argmax", am(k)), (f"{arm} own-cal", decode(ch[f"{k}|own"]["params"], P["probs"][k]))]
    for arm in ("U", "H"):
        if f"A20 {arm}_s{s}" in P["probs"]:
            items.append((f"A20 {arm}@A08", decode(ap, P["probs"][f"A20 {arm}_s{s}"])))
    for arm in ("Other", "Same"):
        if f"A23 {arm}_s{s}" in P["probs"]:
            items.append((f"A23 {arm}@A08", decode(ap, P["probs"][f"A23 {arm}_s{s}"])))
    if f"A14 P s{s}" in P.get("routed", {}):
        items.append(("A14 P", P["routed"][f"A14 P s{s}"]))
    return {f"{n} s{s}": rso.row_of(a8.cfg, p, P["ev"]) for n, p in items}


def check_pool(ctx: ttd.Ctx, a8, seeds: list[int], ch: dict, device, workers: int, acc: dict) -> dict:
    """A15's 600 multi + 200 native check sequences once: A08 and Long10 R from A19's pinned pool logits (rows reproduced)."""
    cfgp = "configs/experiments/video_b3_coverage_v1.yaml"; bctx = bc.Ctx(load_cfg(SimpleNamespace(config=cfgp, set=[])), cfgp); P = bc.with_index(bctx, bc.pool(bctx))
    cidx = bc.members(P, "check"); seqs = [P["seqs"][i] for i in cidx]; vids = {r[0] for q in seqs for r in q["refs"]}
    if vids - set(prv.allowlist(ctx.a23.a10.rv, ROOT)):
        raise SystemExit("a check recording lies outside the training allowlist")
    refs = rw.half_refs(seqs); at = {r: i for i, r in enumerate(refs)}; W = rr.half_windows(seqs, at, 5); keep = np.isin(P["W"]["seq"], cidx)
    if not all(np.array_equal(a, b) for a, b in ((np.asarray(cidx)[W["seq"]], P["W"]["seq"][keep]), (W["centre"], P["W"]["centre"][keep]), (W["marker"], P["W"]["marker"][keep]))):
        raise SystemExit("the check windows differ from the pool's window plan")
    fp = FramePaths(prv.frame_dirs(ctx.a23.a10.paths, ctx.a23.a10.rv, vids)); lgs = infer_orig(new_models(ctx, seeds), seqs, W, fp, ctx.out / "evaluation" / "check", device, workers, acc)
    ev = bc.evalset(P, cidx, "check"); meta = rhr.window_meta(seqs, W); a19 = json.loads((ROOT / ctx.a23.vc["pinned"]["A19_screen"]["path"]).read_text())["systems"]
    out = {"systems": {}, "groups": {}, "reproduced": []}
    for s in seeds:
        la = bc.pool_logits(bctx, P, f"A08_s{s}")[keep]; lr = bc.pool_logits(bctx, P, f"R_s{s}")[keep]; wrong = la.argmax(1) != meta["label"]; ap = a08_params(ctx, s)
        ref23 = {}
        for arm in ("Other", "Same"):
            man23 = json.loads((A23_ROOT / "train" / f"{arm}_s{s}" / "run_manifest.json").read_text()); ref23[arm] = rhr.saved(A23_ROOT / "evaluation" / "check" / f"{arm}_s{s}.npz", man23["identity"], W)
            if ref23[arm] is None or man23["final_sha256"] != ctx.vr["pinned"]["A23_weights"][f"{arm}_s{s}"]:
                raise SystemExit(f"A23 {arm}_s{s}: the saved check outputs or weights do not match")
        items = ([("Base", la, ap), ("Long10 R@A08", lr, ap), ("A23 Other@A08", ref23["Other"], ap), ("A23 Same@A08", ref23["Same"], ap)] + [(f"{a}@A08", lgs[f"{a}_s{s}"], ap) for a in ARMS]
                 + [(f"{a} own-cal", lgs[f"{a}_s{s}"], ch[f"{a}_s{s}|own"]["params"]) for a in ARMS])
        for name, lg, prm in items:
            preds = decode(prm, rr.per_sample(W, lg, seqs)); m, t = bc.evaluate_pool(a8.cfg, preds, ev); out["systems"][f"{name} s{s}"] = {"row": bce.check_row(m), "tables": t, "preds": preds}
        rso.check_stored(out["systems"][f"Base s{s}"]["row"], a19[f"A08_s{s} old"]["check"], f"A08 s{s} + B3 check (A19)")
        rso.check_stored(out["systems"][f"Long10 R@A08 s{s}"]["row"], a19[f"R_s{s} A08 fixed"]["check"], f"Long10 R@A08 s{s} check (A19)"); out["reproduced"] += [f"A08 s{s} + B3 check (A19)", f"Long10 R@A08 s{s} check (A19)"]
        a23c = json.loads((ROOT / ctx.vr["pinned"]["A23_screen"]["path"]).read_text())["systems"]
        for arm in ("Other", "Same"):
            rso.check_stored(out["systems"][f"A23 {arm}@A08 s{s}"]["row"], a23c[f"{arm}@A08 s{s}"]["check"], f"A23 {arm}@A08 s{s} check")
        out["reproduced"].append(f"A23 Other/Same@A08 s{s} check")
        out["groups"][f"Base s{s}"] = rhr.window_groups(la, meta, wrong, False); out["groups"][f"Long10 R s{s}"] = rhr.window_groups(lr, meta, wrong, True)
        out["flips"] = out.get("flips", {}); out["flips"][f"Base s{s}"] = flips(la, W, meta, la, seqs); out["flips"][f"Long10 R s{s}"] = flips(lr, W, meta, la, seqs)
        for arm in ("Other", "Same"):
            out["groups"][f"A23 {arm} s{s}"] = rhr.window_groups(ref23[arm], meta, wrong, True); out["flips"][f"A23 {arm} s{s}"] = flips(ref23[arm], W, meta, la, seqs)
        for a in ARMS:
            out["groups"][f"{a} s{s}"] = rhr.window_groups(lgs[f"{a}_s{s}"], meta, wrong, True); out["flips"][f"{a} s{s}"] = flips(lgs[f"{a}_s{s}"], W, meta, la, seqs)
    out["seqs"] = [{"id": q["id"], "kind": q["kind"], "labels": np.asarray(q["labels"]), "refs": q["refs"]} for q in seqs]
    out["counts"] = {"sequences": len(seqs), "native": int(sum(q["kind"] == "native" for q in seqs)), "multi": int(sum(q["kind"] == "multi" for q in seqs)), "windows": int(len(W["seq"]))}
    return out


def flips(lg: np.ndarray, W: dict, meta: dict, la: np.ndarray, seqs: list[dict]) -> dict:
    """Fixed definitions: a flip is an argmax change between consecutive request windows of a sequence; pure flips join two
    pure windows of the same label; new pure flips are those where the seed's A08 does not flip; boundary errors are wrong
    windows whose centre lies within 8 samples of a true label change, far errors the rest."""
    a, b, lab = lg.argmax(1), la.argmax(1), meta["label"]; same = W["seq"][1:] == W["seq"][:-1]; pure = ~meta["mixed"]
    pp = same & pure[1:] & pure[:-1] & (lab[1:] == lab[:-1]); f = same & (a[1:] != a[:-1]); fa = same & (b[1:] != b[:-1])
    dist = np.full(len(lab), np.inf)
    for qi in np.unique(W["seq"]):
        rows = np.flatnonzero(W["seq"] == qi); L = np.asarray(seqs[int(qi)]["labels"]); ch = np.flatnonzero(L[1:] != L[:-1]) + 1
        if ch.size:
            dist[rows] = np.abs(W["centre"][rows][:, None] - ch[None, :]).min(1)
    near = dist < 8; wrong = a != lab
    return {"flips": int(f.sum()), "pure_pairs": int(pp.sum()), "pure_flips": int((f & pp).sum()), "new_pure_flips": int((f & pp & ~fa).sum()),
            "near_windows": int(near.sum()), "boundary_errors": int((wrong & near).sum()), "far_errors": int((wrong & ~near).sum())}


def promotion(rows: dict, counts: dict) -> dict:
    cells = [(s, h) for s in (0, 2) for h in HALVES]; out = {}; F = lambda n, s, h: rows[f"{n} s{s}"][h]["segment_f1_50_multi"]  # noqa: E731
    for x in ("OtherRel@A08", "SameRel@A08"):
        dF = {f"s{s} {h}": F(x, s, h) - F("Base", s, h) for s, h in cells}; half = {h: float(np.mean([dF[f"s{s} {h}"] for s in (0, 2)])) for h in HALVES}
        dN = float(np.mean([rows[f"{x} s{s}"][h]["native_macro_f1"] - rows[f"Base s{s}"][h]["native_macro_f1"] for s, h in cells]))
        dBR = float(np.mean([rows[f"{x} s{s}"][h]["boundary_recall_multi"] - rows[f"Base s{s}"][h]["boundary_recall_multi"] for s, h in cells]))
        fs = {n: int(sum(counts[f"{n} s{s}"][h]["fn_short"] for s, h in cells)) for n in (x, "Base")}
        feas = {f"s{s}": bool(rows[f"{x} s{s}"]["calibration"]["native_macro_f1"] >= 0.980 and rows[f"{x} s{s}"]["calibration"]["boundary_recall_multi"] >= 0.95) for s in (0, 2)}
        r = {"mean_dF": float(np.mean(list(dF.values()))), "dF": dF, "half_means": half, "mean_dN": dN, "mean_dBR": dBR, "missed_short_true": fs, "calibration_feasible": feas}
        r["signal"] = bool(r["mean_dF"] >= 0.003); r["no_reversal"] = bool(min(half.values()) >= -0.005 and min(dF.values()) >= -0.015)
        r["no_harm"] = bool(dN >= -0.003 and dBR >= -0.010 and fs[x] <= fs["Base"] + 3); r["feasibility"] = bool(all(feas.values()))
        r["meets_rule"] = bool(r["signal"] and r["no_reversal"] and r["no_harm"] and r["feasibility"]); out[x] = r
    out["attribution (not a rule)"] = {}
    for name, (x, y) in (("OtherRel - SameRel", ("OtherRel@A08", "SameRel@A08")), ("OtherRel - A23 Other", ("OtherRel@A08", "A23 Other@A08")), ("SameRel - A23 Same", ("SameRel@A08", "A23 Same@A08"))):
        d = {f"s{s} {h}": F(x, s, h) - F(y, s, h) for s, h in cells}
        out["attribution (not a rule)"][name] = {"dF": d, "mean_dF": float(np.mean(list(d.values()))), "half_means": {h: float(np.mean([d[f"s{s} {h}"] for s in (0, 2)])) for h in HALVES}}
    out["train_seed1"] = bool(out["OtherRel@A08"]["meets_rule"] or out["SameRel@A08"]["meets_rule"])
    return out


def select(ctx: ttd.Ctx, seeds: list[int], choices_sha: str, workers: int) -> None:
    t0, started, ok, note, acc = time.time(), ttd.now(), False, "", new_acc()
    try:
        cp = ctx.out / "evaluation" / choices_name(seeds)
        if sha256_file(cp) != choices_sha:
            raise SystemExit("the choices file differs from the committed sha256; nothing is read")
        chj = json.loads(cp.read_text()); ch = chj["sensitivity_choices"]; models = new_models(ctx, seeds)
        if any(chj["models"][k]["sha256"] != v["sha256"] for k, v in models.items()):
            raise SystemExit("a model differs from the weights its choices were committed with")
        a8 = rsh.a08_ctx(); device = torch.device("cuda"); ttd.determinism()
        per = {h: read_half(ctx, a8, h, seeds, device, workers, True, acc) for h in ("calibration", "selection")}; systems, groups, fl = {}, {}, {}
        for h in HALVES:
            P = per[h]
            for s in seeds:
                for k, v in systems_of(a8, P, s, ch).items():
                    systems.setdefault(k, {})[h] = v
                wrong = P["logits"][f"A08_s{s}"].argmax(1) != P["meta"]["label"]; groups.setdefault(f"Base s{s}", {})[h] = rhr.window_groups(P["logits"][f"A08_s{s}"], P["meta"], wrong, False)
                groups.setdefault(f"Long10 R s{s}", {})[h] = rhr.window_groups(P["logits"][f"R_s{s}"], P["meta"], wrong, True)
                la = P["logits"][f"A08_s{s}"]; fl.setdefault(f"Base s{s}", {})[h] = flips(la, P["W"], P["meta"], la, P["seqs"]); fl.setdefault(f"Long10 R s{s}", {})[h] = flips(P["logits"][f"R_s{s}"], P["W"], P["meta"], la, P["seqs"])
                for arm in ("Other", "Same"):
                    groups.setdefault(f"A23 {arm} s{s}", {})[h] = rhr.window_groups(P["logits"][f"A23 {arm}_s{s}"], P["meta"], wrong, True); fl.setdefault(f"A23 {arm} s{s}", {})[h] = flips(P["logits"][f"A23 {arm}_s{s}"], P["W"], P["meta"], la, P["seqs"])
                for a in ARMS:
                    fl.setdefault(f"{a} s{s}", {})[h] = flips(P["logits"][f"{a}_s{s}"], P["W"], P["meta"], la, P["seqs"])
                for a in ARMS:
                    groups.setdefault(f"{a} s{s}", {})[h] = rhr.window_groups(P["logits"][f"{a}_s{s}"], P["meta"], wrong, True)
            print(f"{h}: rows done", flush=True)
        CK = check_pool(ctx, a8, seeds, ch, device, workers, acc); print("check: rows done", flush=True)
        for k, v in CK["systems"].items():
            systems.setdefault(k, {})["check"] = v
        for k, v in CK["groups"].items():
            groups.setdefault(k, {})["check"] = v
        for k, v in CK["flips"].items():
            fl.setdefault(k, {})["check"] = v
        seqs_of = {"check": CK["seqs"], **{h: per[h]["seqs"] for h in HALVES}}
        rows = {k: {t: v[t]["row"] for t in v} for k, v in systems.items()}; counts = {k: {t: rsh.counts_of(seqs_of[t], v[t]) for t in v} for k, v in systems.items() if "argmax" not in k}
        cmp_, acc_ = {}, {}
        for t, cmps in (("selection", DEV_CMPS), ("calibration", DEV_CMPS), ("check", CHECK_CMPS)):
            for s in seeds:
                rhr.compare(seqs_of[t], systems, t, s, cmps, cmp_, acc_)
            print(f"{t}: comparisons done", flush=True)
        res = {"stage": "select", "seeds": seeds, "config_sha256": sha256_file(ROOT / ctx.config), "choices_sha256": choices_sha, "systems": rows, "segment_counts": counts, "comparisons": cmp_,
               "accounting": acc_, "window_groups": groups, "flips": fl, "reproduced": {**{h: per[h]["reproduced"] for h in HALVES}, "check": CK["reproduced"]},
               "cost": {"parameters": 34640549, "encoder_forwards_per_window": 1, "inference_seconds_per_window": acc["gpu_seconds"] / max(1, acc["windows"]), "windows_inferred": acc["windows"]},
               "counts": {**{h: per[h]["counts"] for h in HALVES}, "check": CK["counts"], "gpu_seconds": acc["gpu_seconds"], "decode_seconds": acc["decode_seconds"]},
               "notes": {"primary": "Other/Same/Long10 R@A08 = each seed's committed A08 B3 numeric setting on the original 16-frame windows", "check": "A15 router-check sessions, unseen by A23's batches; its recordings trained the A08 encoder; not a new holdout",
                         "window_groups": "the original A08-defined groups (16-frame 10 Hz reference windows); accuracy and CE of each model's window argmax against the reference target"},
               "access": {"targets": ["calibration", "selection", "check"], "time": started}}
        if seeds == [0, 2]:
            res["promotion"] = promotion(rows, counts)
        ids = rw.ids_of(a8, {h: per[h]["seqs"] for h in HALVES}) | {q["id"] for q in CK["seqs"] if q["kind"] == "native"} | {r[0] for q in CK["seqs"] for r in q["refs"]}
        flat = json.loads(json.dumps(res, default=str)); assert_no_ids(flat, ids, "select"); name = "screen_seed1.json" if seeds == [1] else "screen.json"
        for p in (ctx.out / "evaluation" / name, ctx.results / name):
            p.parent.mkdir(parents=True, exist_ok=True); ttd.write_json(p, flat)
        if "promotion" in flat:
            ttd.write_json(ctx.out / "evaluation" / "decision.json", {**flat["promotion"], "time": ttd.now()})
            print(json.dumps({k: {x: flat["promotion"][k][x] for x in ("mean_dF", "half_means", "mean_dN", "mean_dBR", "missed_short_true", "calibration_feasible", "meets_rule")} for k in ("OtherRel@A08", "SameRel@A08")}, indent=1),
                  json.dumps(flat["promotion"]["attribution (not a rule)"]), "train_seed1:", flat["promotion"]["train_seed1"], flush=True)
        ok = True; note = f"GPU {acc['gpu_seconds']:.0f} s, decode {acc['decode_seconds']:.0f} s"
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        book(ctx, f"evaluate select + check pool seeds {seeds}", started, time.time() - t0, acc, workers, ok, note)


def training_summary(ctx: ttd.Ctx, seeds: list[int]) -> None:
    """Training data only: per run CE / KL / accuracy on originals and copies by phase, and the matched per-update Other - Same."""
    t0, started, ok, note, out = time.time(), ttd.now(), False, "", {}
    try:
        u = np.arange(ctx.total); parts = {"all": u >= 0, "encoder frozen (u < warm-up)": u < ctx.warmup, "last 270 updates": u >= ctx.total - 270}
        for s in seeds:
            X = {}
            for arm in ARMS:
                rec = [json.loads(x) for x in (ctx.out / "train" / f"{arm}_s{s}" / "updates.jsonl").read_text().splitlines()]
                if [r["u"] for r in rec] != u.tolist():
                    raise SystemExit(f"{arm}_s{s}: the update log is incomplete")
                X[arm] = {k: np.asarray([r[k] for r in rec], float) for k in ("loss", "ce_original", "kl", "ce_copy", "rel", "correct_original", "correct_copy")}
                out[f"{arm}_s{s}"] = {p: {"updates": int(m.sum()), **{k: float(v[m].mean()) for k, v in X[arm].items() if not k.startswith("correct")},
                                          "accuracy_original": float(X[arm]["correct_original"][m].sum() / (ttd.BASE * m.sum())), "accuracy_copy": float(X[arm]["correct_copy"][m].sum() / (ttd.BASE * m.sum()))}
                                      for p, m in parts.items()}
            d = X["OtherRel"]["ce_original"] - X["SameRel"]["ce_original"]
            out[f"OtherRel - SameRel s{s}"] = {p: {"mean_ce_original_difference": float(d[m].mean()), "correct_original_difference": int((X["OtherRel"]["correct_original"][m] - X["SameRel"]["correct_original"][m]).sum())}
                                         for p, m in parts.items()}
        rec = {"stage": "training summary", "seeds": seeds, "runs": out, "notes": "training windows only; both arms see the same 12 originals each update (with the relation term); their copies differ in the donor's view; models diverge after the first update", "time": ttd.now()}
        for p in (ctx.out / "training_summary.json", ctx.results / "training_summary.json"):
            ttd.write_json(p, rec)
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ttd.ledger_append(ctx.ledger, f"training summary seeds {seeds} (CPU)", "cpu", started, time.time() - t0, ok, note)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--stage", choices=("training", "calibrate", "select"), required=True); ap.add_argument("--choices-sha")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 2]); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(); a.set = []; ctx = ttd.Ctx(load_cfg(a), a.config)
    if a.stage == "training":
        return training_summary(ctx, a.seeds)
    if a.stage == "calibrate":
        return calibrate(ctx, a.seeds, a.workers)
    select(ctx, a.seeds, a.choices_sha, a.workers)


if __name__ == "__main__":
    main()
