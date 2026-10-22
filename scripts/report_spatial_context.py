#!/usr/bin/env python
"""Evaluate A21 (configs/experiments/video_spatial_context_v1.yaml): Full and Restricted decoded at their seed's committed
A08 B3 setting (primary), one own-calibration B3 choice each and argmax (sensitivity), on both development halves and once
on A15's check pool; A08, frozen Ordered, R, A14 P, A20 U/H and A19's check rows reproduced; window accuracy and CE by fixed
reference/A08 groups; measured two-view inference cost against A08's single forward.

  --stage training                training windows only: per-run and matched Full - Restricted training CE / accuracy
  --stage calibrate               Full/Restricted inference on calibration (outputs saved); sensitivity choices; choices.json
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
import scripts.train_spatial_context as tsc  # noqa: E402
import scripts.train_video_switch_head as vsh  # noqa: E402
import scripts.trained_token_readout as ttr  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.ev9v_complementarity import decode  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402
from scripts.train_joint_visual_temporal import FramePaths  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

import echo_routing.vit.target_mvit as tm  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402

HALVES = ("selection", "calibration")
ARMS = tsc.ARMS
CHUNK = 75
A20_ROOT = ROOT / "runs/video_hard_replay_v1"
DEV_CMPS = {"Full@A08 - Restricted@A08": ("Full@A08", "Restricted@A08"), "Full@A08 - Base": ("Full@A08", "Base"), "Restricted@A08 - Base": ("Restricted@A08", "Base"),
            "Full own-cal - Restricted own-cal": ("Full own-cal", "Restricted own-cal"), "Full@A08 - A14 P": ("Full@A08", "A14 P"), "Restricted@A08 - A14 P": ("Restricted@A08", "A14 P"),
            "Full@A08 - A20 H@A08": ("Full@A08", "A20 H@A08"), "Full@A08 - R + B3": ("Full@A08", "R + B3")}
CHECK_CMPS = {k: DEV_CMPS[k] for k in ("Full@A08 - Restricted@A08", "Full@A08 - Base", "Restricted@A08 - Base", "Full own-cal - Restricted own-cal")}


def a08_params(ctx: tsc.Ctx, s: int) -> dict:
    p = ctx.vc["pinned"]["A08_choices"]
    if sha256_file(ROOT / p["path"]) != p["sha256"]:
        raise SystemExit("A08's committed choices differ from their pin")
    return json.loads((ROOT / p["path"]).read_text())["choices"][f"all_s{s}"]["params"]


def new_models(ctx: tsc.Ctx, seeds: list[int]) -> dict:
    out = {}
    for s in seeds:
        for arm in ARMS:
            key = f"{arm}_s{s}"; man = json.loads((ctx.out / "train" / key / "run_manifest.json").read_text())
            fz = ROOT / "models_frozen" / "video_spatial_context_v1" / key / "final.pt"; w = fz if fz.is_file() else Path(man["final_checkpoint"])
            if sha256_file(w) != man["final_sha256"]:
                raise SystemExit(f"{key}: the weights differ from their run record")
            out[key] = {"weights": str(w), "sha256": man["final_sha256"], "identity": man["identity"], "seed": s, "arm": arm, "frozen": fz.is_file(), "parameters": man["parameters"]}
    return out


def load(mr: dict, device) -> tsc.SpatialContext:
    st = torch.load(mr["weights"], map_location="cpu", weights_only=True)["model"]
    base = vsh.make_model({k[len("base."):]: v for k, v in st.items() if k.startswith("base.")}, torch.device("cpu")); m = tsc.SpatialContext(base, mr["seed"])
    m.load_state_dict(st, strict=True); m = m.to(device).eval()
    for q in m.parameters():
        q.requires_grad_(False)
    return m


class RefViews(Dataset):
    """(crop, Full aux, Restricted aux) of each (recording, frame, gamma) reference, from one decode per frame."""

    def __init__(self, refs: list[tuple], fp, chunk: int = 128) -> None:
        self.refs, self.fp, self.chunk = refs, fp, chunk

    def __len__(self) -> int:
        return (len(self.refs) + self.chunk - 1) // self.chunk

    def __getitem__(self, i: int):
        out = []
        for v, f, g in self.refs[i * self.chunk:(i + 1) * self.chunk]:
            p = self.fp(v); out.append(tsc.frame_views(p[min(int(f), len(p) - 1)], round(float(g), 6))[:3])
        return tuple(torch.from_numpy(np.stack([x[k] for x in out])) for k in range(3))


def decode_views(refs: list[tuple], fp, workers: int) -> tuple[np.ndarray, dict]:
    crop = np.empty((len(refs), 224, 224, 3), np.uint8); aux = {a: np.empty_like(crop) for a in ARMS}; k = 0
    for c, f, r in DataLoader(RefViews(refs, fp), batch_size=None, num_workers=workers, prefetch_factor=2 if workers else None):
        n = len(c); crop[k:k + n], aux["Full"][k:k + n], aux["Restricted"][k:k + n] = c.numpy(), f.numpy(), r.numpy(); k += n
    if k != len(refs):
        raise SystemExit(f"decoded {k} of {len(refs)} frames")
    return crop, aux


@torch.no_grad()
def infer2(model: tsc.SpatialContext, W: dict, crop: np.ndarray, aux: np.ndarray, device, batch: int = 32) -> tuple[np.ndarray, float]:
    out = []; torch.cuda.synchronize(); t0 = time.time()
    for b in range(0, len(W["ref"]), batch):
        idx = W["ref"][b:b + batch]; mk = tm.marker_planes(torch.from_numpy(W["marker"][b:b + batch]).long().to(device))
        z = model.logits(tm.to_rgb(torch.from_numpy(crop[idx]).to(device, non_blocking=True)), tm.to_rgb(torch.from_numpy(aux[idx]).to(device, non_blocking=True)), mk)
        out.append(z.float().cpu().numpy())
    torch.cuda.synchronize()
    return np.concatenate(out), time.time() - t0


def infer_seqs(models: dict, seqs: list[dict], W: dict, fp, where: Path, device, workers: int, acc: dict) -> dict:
    """Saved outputs reused by run identity and window plan; otherwise chunks of CHUNK sequences: one decode of the
    three views per chunk, every pending model on its own arm's aux."""
    out = {k: lg for k, mr in models.items() if (lg := rhr.saved(where / f"{k}.npz", mr["identity"], W)) is not None}; todo = {k: mr for k, mr in models.items() if k not in out}
    if not todo:
        return out
    M = {k: load(mr, device) for k, mr in todo.items()}; parts = {k: [] for k in todo}; seq_, cen_ = [], []; torch.cuda.reset_peak_memory_stats(device)
    for c0 in range(0, len(seqs), CHUNK):
        chunk = seqs[c0:c0 + CHUNK]; refs = rw.half_refs(chunk); at = {r: i for i, r in enumerate(refs)}; Wc = rr.half_windows(chunk, at, 5)
        t0 = time.time(); crop, aux = decode_views(refs, fp, workers); acc["decode_seconds"] += time.time() - t0
        for k, m in M.items():
            lg, sec = infer2(m, Wc, crop, aux[todo[k]["arm"]], device); acc["gpu_seconds"] += sec; acc["windows"] += len(Wc["seq"]); parts[k].append(lg.astype(np.float32))
        seq_.append(Wc["seq"] + c0); cen_.append(Wc["centre"]); del crop, aux
        print(f"{where.name} chunk {c0 // CHUNK + 1} of {-(-len(seqs) // CHUNK)}: GPU {acc['gpu_seconds']:.0f} s, decode {acc['decode_seconds']:.0f} s", flush=True)
    if not (np.array_equal(np.concatenate(seq_), W["seq"]) and np.array_equal(np.concatenate(cen_), W["centre"])):
        raise SystemExit(f"{where.name}: the chunked windows differ from the window plan")
    acc["peak_gpu_memory_mib"] = max(acc.get("peak_gpu_memory_mib", 0.0), torch.cuda.max_memory_allocated(device) / 2**20); where.mkdir(parents=True, exist_ok=True)
    for k, mr in todo.items():
        out[k] = np.concatenate(parts[k]); np.savez(where / f"{k}.npz", identity=mr["identity"], weights_sha256=mr["sha256"], seq=W["seq"], centre=W["centre"], logits=out[k])
    del M; torch.cuda.empty_cache()
    return out


def a20_reference(s: int, h: str, W: dict) -> dict:
    """A20's frozen U/H development outputs (identity and window plan checked)."""
    out = {}
    for arm in ("U", "H"):
        man = json.loads((A20_ROOT / "train" / f"{arm}_s{s}" / "run_manifest.json").read_text()); lg = rhr.saved(A20_ROOT / "evaluation" / h / f"{arm}_s{s}.npz", man["identity"], W)
        if lg is None:
            raise SystemExit(f"A20 {arm}_s{s} {h}: the saved outputs do not match their identity or this half's windows")
        out[arm] = lg
    return out


def read_half(ctx: tsc.Ctx, a8, h: str, seeds: list[int], device, workers: int, routers: bool, acc: dict) -> dict:
    ev, seqs = a8.half(h, rso.DevAccess({h})); ids = [q["id"] for q in seqs]; refs = rw.half_refs(seqs); at = {r: i for i, r in enumerate(refs)}; W = rr.half_windows(seqs, at, 5)
    orders = np.broadcast_to(np.arange(16, dtype=np.int64), W["s"].shape).copy(); a8sum = json.loads((ROOT / "docs/results/2026-10-16_video_within_view/summary.json").read_text())
    v01 = a8.v01_summary()["systems"]; pins = rsh.a08_output_pins(); a20 = json.loads((ROOT / ctx.vc["pinned"]["A20_screen"]["path"]).read_text())["systems"]
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
        root, man, rp, want = rhr.r_reference(s); lgR = rhr.saved(root / "evaluation" / h / f"R_s{s}.npz", man["identity"], W)
        if lgR is None:
            raise SystemExit(f"R_s{s} {h}: the saved outputs do not match their run identity or this half's windows")
        out["probs"][f"R_s{s}"], out["params"][f"R_s{s}"] = rr.per_sample(W, lgR, seqs), rp
        rso.check_stored(rso.row_of(a8.cfg, decode(rp, out["probs"][f"R_s{s}"]), ev)["row"], want[h], f"R + B3 s{s} {h}")
        out["reproduced"] += [f"A08 all_s{s} + B3 {h}", f"frozen Ordered s{s} argmax and B3 {h} (V01)", f"R s{s} + B3 {h}"]
        if s in (0, 2):
            for arm, lgA in a20_reference(s, h, W).items():
                out["probs"][f"A20 {arm}_s{s}"] = rr.per_sample(W, lgA, seqs)
                rso.check_stored(rso.row_of(a8.cfg, decode(ap, out["probs"][f"A20 {arm}_s{s}"]), ev)["row"], a20[f"{arm}@A08 s{s}"][h], f"A20 {arm}@A08 s{s} {h}")
                out["reproduced"].append(f"A20 {arm}@A08 s{s} {h}")
    fp = FramePaths(prv.frame_dirs(a8.paths, a8.rv, {r[0] for r in refs}))
    for key, lg in infer_seqs(new_models(ctx, seeds), seqs, W, fp, ctx.out / "evaluation" / h, device, workers, acc).items():
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


def book(ctx: tsc.Ctx, what: str, started: str, total: float, acc: dict, workers: int, ok: bool, note: str) -> None:
    """A19's split: GPU inference seconds as GPU; the main process outside them and the extra decode workers as CPU."""
    g = min(acc["gpu_seconds"], total)
    tsc.ledger_append(ctx.ledger, f"{what} (frozen inference, GPU inference seconds)", "gpu", started, g, ok, note)
    tsc.ledger_append(ctx.ledger, f"{what} (CPU: main process outside GPU + {workers - 1} more decode workers)", "cpu", started, (total - g) + acc["decode_seconds"] * (workers - 1), ok, note)


def choices_name(seeds: list[int]) -> str:
    return "choices_seed1.json" if seeds == [1] else "choices.json"


def new_acc() -> dict:
    return {"gpu_seconds": 0.0, "decode_seconds": 0.0, "windows": 0}


def calibrate(ctx: tsc.Ctx, seeds: list[int], workers: int) -> None:
    t0, started, ok, note, acc = time.time(), tsc.now(), False, "", new_acc()
    try:
        a8 = rsh.a08_ctx(); rw.check_grid(a8); device = torch.device("cuda"); tsc.determinism(); P = read_half(ctx, a8, "calibration", seeds, device, workers, False, acc); ch = {}
        for s in seeds:
            for arm in ARMS:
                c = rcs.b3_choose(a8.cfg, rso.DIAG, P["probs"][f"{arm}_s{s}"], P["ev"]); ch[f"{arm}_s{s}|own"] = {x: c[x] for x in ("status", "params", "n_feasible", "grid")}
                print(f"{arm}_s{s}: own B3 {c['status']} {c['params']} ({c['n_feasible']} feasible)", flush=True)
        rec = {"stage": "calibrate", "seeds": seeds, "config_sha256": sha256_file(ROOT / ctx.config), "primary": {f"s{s}": a08_params(ctx, s) for s in seeds}, "sensitivity_choices": ch,
               "models": {k: {x: v[x] for x in ("sha256", "identity", "frozen")} for k, v in new_models(ctx, seeds).items()},
               "notes": {"primary": "each seed's committed A08 B3 numeric setting (pinned, never re-chosen)", "sensitivity": "never used for promotion or to change the primary"}, "time": tsc.now()}
        flat = json.loads(json.dumps(rec, default=str)); assert_no_ids(flat, rw.ids_of(a8, {"calibration": P["seqs"]}), "choices")
        for p in (ctx.out / "evaluation" / choices_name(seeds), ctx.results / choices_name(seeds)):
            p.parent.mkdir(parents=True, exist_ok=True); tsc.write_json(p, flat)
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
    for arm in ("U", "H"):
        if f"A20 {arm}_s{s}" in P["probs"]:
            items.append((f"A20 {arm}@A08", decode(ap, P["probs"][f"A20 {arm}_s{s}"])))
    if f"A14 P s{s}" in P.get("routed", {}):
        items.append(("A14 P", P["routed"][f"A14 P s{s}"]))
    return {f"{n} s{s}": rso.row_of(a8.cfg, p, P["ev"]) for n, p in items}


def check_pool(ctx: tsc.Ctx, a8, seeds: list[int], ch: dict, device, workers: int, acc: dict) -> dict:
    """A15's 600 multi + 200 native check sequences once: Base from the pinned A08 pool logits (A19's rows reproduced)."""
    bctx = bc.Ctx(load_cfg(SimpleNamespace(config=bc_config(), set=[])), bc_config()); P = bc.with_index(bctx, bc.pool(bctx)); cidx = bc.members(P, "check"); seqs = [P["seqs"][i] for i in cidx]
    vids = {r[0] for q in seqs for r in q["refs"]}
    if vids - set(prv.allowlist(ctx.a10.rv, ROOT)):
        raise SystemExit("a check recording lies outside the training allowlist")
    refs = rw.half_refs(seqs); at = {r: i for i, r in enumerate(refs)}; W = rr.half_windows(seqs, at, 5); keep = np.isin(P["W"]["seq"], cidx)
    if not all(np.array_equal(a, b) for a, b in ((np.asarray(cidx)[W["seq"]], P["W"]["seq"][keep]), (W["centre"], P["W"]["centre"][keep]), (W["marker"], P["W"]["marker"][keep]))):
        raise SystemExit("the check windows differ from the pool's window plan")
    fp = FramePaths(prv.frame_dirs(ctx.a10.paths, ctx.a10.rv, vids)); lgs = infer_seqs(new_models(ctx, seeds), seqs, W, fp, ctx.out / "evaluation" / "check", device, workers, acc)
    ev = bc.evalset(P, cidx, "check"); meta = rhr.window_meta(seqs, W); a19 = json.loads((ROOT / ctx.vc["pinned"]["A19_screen"]["path"]).read_text())["systems"]
    out = {"systems": {}, "groups": {}, "reproduced": []}
    for s in seeds:
        la = bc.pool_logits(bctx, P, f"A08_s{s}")[keep]; wrong = la.argmax(1) != meta["label"]; ap = a08_params(ctx, s)
        items = [("Base", la, ap)] + [(f"{a}@A08", lgs[f"{a}_s{s}"], ap) for a in ARMS] + [(f"{a} own-cal", lgs[f"{a}_s{s}"], ch[f"{a}_s{s}|own"]["params"]) for a in ARMS]
        for name, lg, prm in items:
            preds = decode(prm, rr.per_sample(W, lg, seqs)); m, t = bc.evaluate_pool(a8.cfg, preds, ev); out["systems"][f"{name} s{s}"] = {"row": bce.check_row(m), "tables": t, "preds": preds}
        rso.check_stored(out["systems"][f"Base s{s}"]["row"], a19[f"A08_s{s} old"]["check"], f"A08 s{s} + B3 check (A19)"); out["reproduced"].append(f"A08 s{s} + B3 check (A19)")
        out["groups"][f"Base s{s}"] = rhr.window_groups(la, meta, wrong, False)
        for a in ARMS:
            out["groups"][f"{a} s{s}"] = rhr.window_groups(lgs[f"{a}_s{s}"], meta, wrong, True)
    out["seqs"] = [{"id": q["id"], "kind": q["kind"], "labels": np.asarray(q["labels"]), "refs": q["refs"]} for q in seqs]
    out["counts"] = {"sequences": len(seqs), "native": int(sum(q["kind"] == "native" for q in seqs)), "multi": int(sum(q["kind"] == "multi" for q in seqs)), "windows": int(len(W["seq"]))}
    return out


def bc_config() -> str:
    return "configs/experiments/video_b3_coverage_v1.yaml"


def promotion(rows: dict, counts: dict) -> dict:
    cells = [(s, h) for s in (0, 2) for h in HALVES]; out = {}; F = lambda n, s, h: rows[f"{n} s{s}"][h]["segment_f1_50_multi"]  # noqa: E731
    for x in ("Full@A08", "Restricted@A08"):
        dF = {f"s{s} {h}": F(x, s, h) - F("Base", s, h) for s, h in cells}; half = {h: float(np.mean([dF[f"s{s} {h}"] for s in (0, 2)])) for h in HALVES}
        dN = float(np.mean([rows[f"{x} s{s}"][h]["native_macro_f1"] - rows[f"Base s{s}"][h]["native_macro_f1"] for s, h in cells]))
        dBR = float(np.mean([rows[f"{x} s{s}"][h]["boundary_recall_multi"] - rows[f"Base s{s}"][h]["boundary_recall_multi"] for s, h in cells]))
        fs = {n: int(sum(counts[f"{n} s{s}"][h]["fn_short"] for s, h in cells)) for n in (x, "Base")}
        feas = {f"s{s}": bool(rows[f"{x} s{s}"]["calibration"]["native_macro_f1"] >= 0.980 and rows[f"{x} s{s}"]["calibration"]["boundary_recall_multi"] >= 0.95) for s in (0, 2)}
        r = {"mean_dF": float(np.mean(list(dF.values()))), "dF": dF, "half_means": half, "mean_dN": dN, "mean_dBR": dBR, "missed_short_true": fs, "calibration_feasible": feas}
        r["signal"] = bool(r["mean_dF"] >= 0.003); r["no_reversal"] = bool(min(half.values()) >= -0.005 and min(dF.values()) >= -0.015)
        r["no_harm"] = bool(dN >= -0.003 and dBR >= -0.010 and fs[x] <= fs["Base"] + 3); r["feasibility"] = bool(all(feas.values()))
        r["meets_rule"] = bool(r["signal"] and r["no_reversal"] and r["no_harm"] and r["feasibility"]); out[x] = r
    dFR = {f"s{s} {h}": F("Full@A08", s, h) - F("Restricted@A08", s, h) for s, h in cells}
    out["Full - Restricted (attribution, not a rule)"] = {"dF": dFR, "mean_dF": float(np.mean(list(dFR.values()))), "half_means": {h: float(np.mean([dFR[f"s{s} {h}"] for s in (0, 2)])) for h in HALVES}}
    out["train_seed1"] = bool(out["Full@A08"]["meets_rule"] or out["Restricted@A08"]["meets_rule"])
    return out


def a08_single_forward(ctx: tsc.Ctx, device, batch: int = 32, reps: int = 3) -> dict:
    """Micro-benchmark on synthetic inputs: A08's one CLS forward per window against the two-view model's two forwards."""
    state, _ = ctx.start(0); a08 = vsh.make_model(state, device).eval(); two = tsc.make(state, 0, device).eval(); x = torch.randint(0, 256, (batch, 16, 224, 224, 3), dtype=torch.uint8)
    mk = tm.marker_planes(torch.zeros(batch, dtype=torch.long).to(device)); out = {}
    with torch.no_grad():
        for name, fn in (("A08 one forward", lambda: a08(tm.to_rgb(x.to(device)), mk)), ("two-view model", lambda: two.logits(tm.to_rgb(x.to(device)), tm.to_rgb(x.to(device)), mk))):
            fn(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(device); a = time.time()
            for _ in range(reps):
                fn()
            torch.cuda.synchronize(); out[name] = {"seconds_per_window": (time.time() - a) / (reps * batch), "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20}
    out["A18 three-model average"] = {"seconds_per_window": 3 * out["A08 one forward"]["seconds_per_window"], "note": "three A08 forwards (estimated as 3 x A08)"}
    del a08, two; torch.cuda.empty_cache()
    return out


def select(ctx: tsc.Ctx, seeds: list[int], choices_sha: str, workers: int) -> None:
    t0, started, ok, note, acc = time.time(), tsc.now(), False, "", new_acc()
    try:
        cp = ctx.out / "evaluation" / choices_name(seeds)
        if sha256_file(cp) != choices_sha:
            raise SystemExit("the choices file differs from the committed sha256; nothing is read")
        chj = json.loads(cp.read_text()); ch = chj["sensitivity_choices"]; models = new_models(ctx, seeds)
        if any(chj["models"][k]["sha256"] != v["sha256"] for k, v in models.items()):
            raise SystemExit("a model differs from the weights its choices were committed with")
        a8 = rsh.a08_ctx(); device = torch.device("cuda"); tsc.determinism()
        per = {h: read_half(ctx, a8, h, seeds, device, workers, True, acc) for h in ("calibration", "selection")}; systems, groups = {}, {}
        for h in HALVES:
            P = per[h]
            for s in seeds:
                for k, v in systems_of(a8, P, s, ch).items():
                    systems.setdefault(k, {})[h] = v
                wrong = P["logits"][f"A08_s{s}"].argmax(1) != P["meta"]["label"]; groups.setdefault(f"Base s{s}", {})[h] = rhr.window_groups(P["logits"][f"A08_s{s}"], P["meta"], wrong, False)
                for a in ARMS:
                    groups.setdefault(f"{a} s{s}", {})[h] = rhr.window_groups(P["logits"][f"{a}_s{s}"], P["meta"], wrong, True)
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
                rhr.compare(seqs_of[t], systems, t, s, cmps, cmp_, acc_)
            print(f"{t}: comparisons done", flush=True)
        cost = {"models": {k: {"parameters": v["parameters"], "encoder_forwards_per_window": 2} for k, v in models.items()}, "benchmark": a08_single_forward(ctx, device),
                "measured_inference": {"two_view_seconds_per_window": acc["gpu_seconds"] / max(1, acc["windows"]), "windows_inferred": acc["windows"], "peak_gpu_memory_mib": acc.get("peak_gpu_memory_mib")}}
        res = {"stage": "select", "seeds": seeds, "config_sha256": sha256_file(ROOT / ctx.config), "choices_sha256": choices_sha, "systems": rows, "segment_counts": counts, "comparisons": cmp_,
               "accounting": acc_, "window_groups": groups, "reproduced": {**{h: per[h]["reproduced"] for h in HALVES}, "check": CK["reproduced"]}, "cost": cost,
               "counts": {**{h: per[h]["counts"] for h in HALVES}, "check": CK["counts"], "gpu_seconds": acc["gpu_seconds"], "decode_seconds": acc["decode_seconds"]},
               "notes": {"primary": "Full/Restricted@A08 = each seed's committed A08 B3 numeric setting", "check": "A15 router-check sessions, unseen by A21's batches; its recordings trained the A08 encoder; not a new holdout",
                         "window_groups": "accuracy and CE of the window argmax against the reference target; A08_wrong = the seed's own A08 argmax wrong on that window",
                         "cost": "two encoder forwards per window; never equal compute with A08"}, "access": {"targets": ["calibration", "selection", "check"], "time": started}}
        if seeds == [0, 2]:
            res["promotion"] = promotion(rows, counts)
        ids = rw.ids_of(a8, {h: per[h]["seqs"] for h in HALVES}) | {q["id"] for q in CK["seqs"] if q["kind"] == "native"} | {r[0] for q in CK["seqs"] for r in q["refs"]}
        flat = json.loads(json.dumps(res, default=str)); assert_no_ids(flat, ids, "select"); name = "screen_seed1.json" if seeds == [1] else "screen.json"
        for p in (ctx.out / "evaluation" / name, ctx.results / name):
            p.parent.mkdir(parents=True, exist_ok=True); tsc.write_json(p, flat)
        if "promotion" in flat:
            tsc.write_json(ctx.out / "evaluation" / "decision.json", {**flat["promotion"], "time": tsc.now()})
            print(json.dumps({k: {x: flat["promotion"][k][x] for x in ("mean_dF", "half_means", "mean_dN", "mean_dBR", "missed_short_true", "calibration_feasible", "meets_rule")} for k in ("Full@A08", "Restricted@A08")}, indent=1),
                  json.dumps(flat["promotion"]["Full - Restricted (attribution, not a rule)"]), "train_seed1:", flat["promotion"]["train_seed1"], flush=True)
        ok = True; note = f"GPU {acc['gpu_seconds']:.0f} s, decode {acc['decode_seconds']:.0f} s"
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        book(ctx, f"evaluate select + check pool seeds {seeds}", started, time.time() - t0, acc, workers, ok, note)


def training_summary(ctx: tsc.Ctx, seeds: list[int]) -> None:
    """Training data only (no evaluation set): per run CE / KL / accuracy by phase, and the matched per-update Full -
    Restricted difference (both arms see the same 12 windows every update; only the aux view differs)."""
    t0, started, ok, note, out = time.time(), tsc.now(), False, "", {}
    try:
        u = np.arange(ctx.total); parts = {"all": u >= 0, "encoder frozen (u < warm-up)": u < ctx.warmup, "last 270 updates": u >= ctx.total - 270}
        for s in seeds:
            X = {}
            for arm in ARMS:
                rec = [json.loads(x) for x in (ctx.out / "train" / f"{arm}_s{s}" / "updates.jsonl").read_text().splitlines()]
                if [r["u"] for r in rec] != u.tolist():
                    raise SystemExit(f"{arm}_s{s}: the update log is incomplete")
                X[arm] = {k: np.asarray([r[k] for r in rec], float) for k in ("loss", "ce", "kl", "correct", "g_last_norm")}
                out[f"{arm}_s{s}"] = {p: {"updates": int(m.sum()), **{k: float(v[m].mean()) for k, v in X[arm].items() if k != "correct"}, "accuracy": float(X[arm]["correct"][m].sum() / (tsc.WINDOWS * m.sum()))}
                                      for p, m in parts.items()}
            d = X["Full"]["ce"] - X["Restricted"]["ce"]
            out[f"Full - Restricted s{s}"] = {p: {"mean_ce_difference": float(d[m].mean()), "share_full_lower": float((d[m] < 0).mean()), "share_equal": float((d[m] == 0).mean()),
                                                  "correct_difference": int((X["Full"]["correct"][m] - X["Restricted"]["correct"][m]).sum())} for p, m in parts.items()}
        rec = {"stage": "training summary", "seeds": seeds, "runs": out, "notes": "training windows only; both arms see the same windows and targets each update, but their models diverge after the first update", "time": tsc.now()}
        for p in (ctx.out / "training_summary.json", ctx.results / "training_summary.json"):
            tsc.write_json(p, rec)
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        tsc.ledger_append(ctx.ledger, f"training summary seeds {seeds} (CPU)", "cpu", started, time.time() - t0, ok, note)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--stage", choices=("training", "calibrate", "select"), required=True); ap.add_argument("--choices-sha")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 2]); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(); a.set = []; ctx = tsc.Ctx(load_cfg(a), a.config)
    if a.stage == "training":
        return training_summary(ctx, a.seeds)
    if a.stage == "calibrate":
        return calibrate(ctx, a.seeds, a.workers)
    select(ctx, a.seeds, a.choices_sha, a.workers)


if __name__ == "__main__":
    main()
