#!/usr/bin/env python
"""Evaluate A10 (configs/experiments/video_switch_head_v1.yaml) on both development halves.

Per half: A03's half reader, V01's windows and frames; frozen Ordered and A08 A reproduced from their saved outputs; each
A10 model's per-window logits and 512-d embeddings. R is decoded with B3 (A08's grid and rule). P and V are decoded with
their own switch head: an edge score between consecutive windows of a sequence is placed at cut sample
(c_k + c_k+1) // 2 + 1, and the existing boundary_decode (NMS, segments named by mean-probability argmax, adjacent
same-class segments merged) gives the labels; the thresholds are 8 quantiles of the model's own calibration native edge
scores, chosen once on calibration with A08's rule. P and V CLS outputs through B3 and every argmax are sensitivities.

  --stage screen       seeds 0 and 2, and the fixed third-seed rule (decision.json)
  --stage three_seed   seeds 0, 1 and 2, with three-seed verdicts
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

import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.prepare_raw_video_context as prv  # noqa: E402
import scripts.report_center_supervision as rcs  # noqa: E402
import scripts.report_raw_video_context as rr  # noqa: E402
import scripts.report_segment_objective as rso  # noqa: E402
import scripts.report_video_discriminative_r1 as rd1  # noqa: E402
import scripts.report_video_within_view as rw  # noqa: E402
import scripts.segment_objective_tables as st  # noqa: E402
import scripts.train_video_switch_head as vsh  # noqa: E402
import scripts.trained_token_readout as ttr  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.ev9v_complementarity import decode  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402
from scripts.train_joint_visual_temporal import FramePaths  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import echo_routing.vit.target_mvit as tm  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.temporal.segments import boundary_decode  # noqa: E402

HALVES = ("selection", "calibration")
QUANTILES = (0.80, 0.85, 0.90, 0.95, 0.97, 0.99, 0.995, 0.999)
NMS, DIAG_Q = 5, 0.95
PRIMARY = {"R": "R + B3", "P": "P head", "V": "V head"}
CMPS = {"R - A08 A": ("R + B3", "A08 A + B3"), "P - A08 A": ("P head", "A08 A + B3"), "V - A08 A": ("V head", "A08 A + B3"), "V - P": ("V head", "P head"),
        "P - R": ("P head", "R + B3"), "V - R": ("V head", "R + B3"), "R - frozen Ordered": ("R + B3", "frozen Ordered + B3"),
        "P - frozen Ordered": ("P head", "frozen Ordered + B3"), "V - frozen Ordered": ("V head", "frozen Ordered + B3")}


def a08_ctx() -> rw.Ctx:
    p = "configs/experiments/video_within_view_v1.yaml"
    return rw.Ctx(load_cfg(SimpleNamespace(config=p, set=[])), p)


def a08_output_pins() -> dict:
    return yaml.safe_load((ROOT / "configs/experiments/video_discriminative_v1.yaml").read_text())["video_discriminative"]["source"]["a08_all_outputs"]


def models_of(ctx: vsh.Ctx, seeds: list[int]) -> dict:
    out = {}
    for s in seeds:
        for arm in vsh.ARMS:
            key = f"{arm}_s{s}"; man = json.loads((ctx.out / "train" / key / "run_manifest.json").read_text())
            fz = ROOT / "models_frozen" / vsh.NS / key / "final.pt"; w = fz if fz.is_file() else Path(man["final_checkpoint"])
            out[key] = {"weights": str(w), "sha256": man["final_sha256"], "identity": man["identity"], "arm": arm, "seed": s}
    return out


def load(mr: dict, device):
    if sha256_file(Path(mr["weights"])) != mr["sha256"]:
        raise SystemExit(f"{mr['weights']}: the weights differ from their run record")
    st_ = torch.load(mr["weights"], map_location="cpu", weights_only=True); model = vsh.make_model(st_["model"], device).eval()
    heads = vsh.build_heads(mr["seed"]).to(device); heads.load_state_dict(st_["heads"]); heads.eval()
    return model, heads


@torch.no_grad()
def infer(model, W: dict, orders: np.ndarray, frames: np.ndarray, device, batch: int = 32) -> tuple[np.ndarray, np.ndarray, float]:
    lg, em = [], []; torch.cuda.synchronize(); t0 = time.time()
    for b in range(0, len(orders), batch):
        idx = np.take_along_axis(W["ref"][b:b + batch], orders[b:b + batch], 1); u8 = torch.from_numpy(frames[idx]).to(device, non_blocking=True)
        e = model.embed(tm.to_rgb(u8), tm.marker_planes(torch.from_numpy(W["marker"][b:b + batch]).long().to(device))); z = model.classifier(e)
        lg.append(z.float().cpu().numpy()); em.append(e.float().cpu().numpy())
    torch.cuda.synchronize()
    return np.concatenate(lg), np.concatenate(em), time.time() - t0


@torch.no_grad()
def edge_scores(head_kind: str, heads, W: dict, logits: np.ndarray, emb: np.ndarray, seqs: list[dict], device) -> dict:
    """Per sequence: (cut samples, sigmoid edge scores) between consecutive windows."""
    out = {}
    for qi, q in enumerate(seqs):
        rows = np.flatnonzero(W["seq"] == qi); rows = rows[np.argsort(W["centre"][rows], kind="stable")]
        if rows.size < 2:
            out[q["id"]] = (np.zeros(0, int), np.zeros(0)); continue
        z = torch.from_numpy(logits[rows]).to(device); e = torch.from_numpy(emb[rows]).to(device); a = np.arange(rows.size - 1); b = a + 1
        s = torch.sigmoid(heads[head_kind](vsh.edge_features(z, e, a, b, head_kind))).cpu().numpy().astype(float)
        c = W["centre"][rows]; out[q["id"]] = ((c[a] + c[b]) // 2 + 1, s)
    return out


def head_decode(probs: dict, edges: dict, seqs: list[dict], threshold: float) -> dict:
    preds = {}
    for q in seqs:
        p = probs[q["id"]]; n = p.shape[0]; cut, s = edges[q["id"]]; bnd = np.zeros(n)
        for i, v in zip(cut, s):
            if 0 < i < n:
                bnd[i] = max(bnd[i], v)
        preds[q["id"]] = boundary_decode(p, bnd, threshold, NMS)
    return preds


def head_choose(cfg: dict, probs: dict, edges: dict, seqs: list[dict], ev) -> dict:
    nat = np.concatenate([edges[q["id"]][1] for q in seqs if q["kind"] == "native" and edges[q["id"]][1].size] or [np.zeros(1)])
    recs = []
    for order, qn in enumerate(QUANTILES):
        thr = float(np.quantile(nat, qn)); m, _ = rcs.evaluate(cfg, head_decode(probs, edges, seqs, thr), ev)
        recs.append({"order": order, "params": {"threshold": thr, "nms_radius": NMS, "threshold_quantile": qn}, **{k: m[k] for k in rcs.FLAT}})
    sel = rcs.select_setting(recs, 0.980, 0.95)
    if sel["chosen"] is None:
        d = next(r for r in recs if abs(r["params"]["threshold_quantile"] - DIAG_Q) < 1e-9)
        return {"status": "infeasible (diagnostic setting shown)", "params": d["params"], "n_feasible": 0, "grid": recs}
    return {"status": "feasible", "params": sel["chosen"]["params"], "n_feasible": sel["n_feasible"], "grid": recs}


def read_half(ctx: vsh.Ctx, a8: rw.Ctx, h: str, models: dict, seeds: list[int], device, workers: int) -> dict:
    ev, seqs = a8.half(h, rso.DevAccess({h})); ids = [q["id"] for q in seqs]; refs = rw.half_refs(seqs); at = {r: i for i, r in enumerate(refs)}
    fp = FramePaths(prv.frame_dirs(a8.paths, a8.rv, {r[0] for r in refs})); t0 = time.time(); frames = rr.decode_frames(refs, fp, workers)
    W = rr.half_windows(seqs, at, 5); orders = np.broadcast_to(np.arange(16, dtype=np.int64), W["s"].shape).copy()
    v01 = a8.v01_summary()["systems"]; a8sum = json.loads((ROOT / "docs/results/2026-10-16_video_within_view/summary.json").read_text())
    a8ch = json.loads(vsh.pinned(ctx.src["a08_choices"]).read_text())["choices"]; pins = a08_output_pins()
    out = {"ev": ev, "seqs": seqs, "probs": {}, "edges": {}, "refs": {}, "oparams": {}, "a08params": {}, "parity": {}, "reproduced": [], "gpu_seconds": 0.0,
           "counts": {"sequences": len(seqs), "windows": int(len(W["seq"])), "frames": len(refs), "decode_seconds": time.time() - t0}}
    for s in seeds:
        saved = a8.v01_output(h, s); ttr.check_windows(W, ids, saved, f"ordered_s{s} {h}"); op = v01[f"ordered_s{s}"]["b3_choice"]["params"]; out["oparams"][s] = op
        P = rr.per_sample(W, saved["logits"].astype(np.float32), seqs); rows = ttr.cls_rows(a8.cfg, P, ev, op)
        for row in ("argmax", "b3"):
            ttr.check_stored(rows[row]["row"], v01[f"ordered_s{s}"][row][h], f"ordered_s{s} {row} {h} (V01)")
        out["refs"][("frozen Ordered", s)] = P
        apath = ROOT / "runs/video_within_view_v1/evaluation" / h / f"all_s{s}.npz"
        if sha256_file(apath) != pins[h][s]:
            raise SystemExit(f"A08 all_s{s} {h}: the saved outputs differ from their pin")
        lgA, _ = rr.cached_logits(apath, a8sum["runs"][f"all_s{s}"]["identity"], W, orders, seqs)
        if lgA is None:
            raise SystemExit(f"A08 all_s{s} {h}: the saved outputs do not match their identity or this half's windows")
        PA = rr.per_sample(W, lgA, seqs); ap = a8ch[f"all_s{s}"]["params"]; out["a08params"][s] = ap; out["refs"][("A08 A", s)] = PA
        ttr.check_stored(rso.row_of(a8.cfg, decode(ap, PA), ev)["row"], a8sum["systems"][f"A + B3 s{s}"][h], f"A08 A + B3 s{s} {h}")
        if h == "calibration":                                   # A08 A start state against its own saved outputs (the source parity)
            state, _ = ctx.start_state(s); src = vsh.make_model(state, device).eval(); t1 = time.time()
            lg, _ = rr.model_logits(src, {"ref": W["ref"][:64], "marker": W["marker"][:64]}, orders[:64], frames, device); out["gpu_seconds"] += time.time() - t1; del src
            out["parity"][f"s{s}"] = vsh.vso.parity(lg, lgA[:64], 1e-4)
            if not out["parity"][f"s{s}"]["ok"]:
                raise SystemExit(f"seed {s}: the A08 start-state parity failed")
        out["reproduced"] += [f"frozen Ordered s{s} argmax and B3 {h}", f"A08 A s{s} B3 {h}"]
    for key, mr in models.items():
        path = ctx.out / "evaluation" / h / f"{key}.npz"; got = None
        if path.is_file():
            with np.load(path) as z:
                if str(z["identity"]) == mr["identity"] and np.array_equal(z["seq"], W["seq"]) and np.array_equal(z["centre"], W["centre"]):
                    got = (z["logits"], z["emb"])
        if got is None:
            model, heads = load(mr, device); lg, em, sec = infer(model, W, orders, frames, device); out["gpu_seconds"] += sec; del model, heads; torch.cuda.empty_cache()
            path.parent.mkdir(parents=True, exist_ok=True); np.savez(path, identity=mr["identity"], seq=W["seq"], centre=W["centre"], logits=lg, emb=em); got = (lg, em)
        lg, em = got; out["probs"][key] = rr.per_sample(W, lg.astype(np.float32), seqs)
        if mr["arm"] in ("P", "V"):
            _, heads = load(mr, device); out["edges"][key] = edge_scores(mr["arm"], heads, W, lg.astype(np.float32), em.astype(np.float32), seqs, device); del heads; torch.cuda.empty_cache()
        print(f"{h} {key}: outputs ready", flush=True)
    del frames
    return out


def counts_of(seqs: list[dict], sysd: dict) -> dict:
    kc, sh = st.kind_counts(seqs, sysd["preds"], sysd["tables"])["multi"], rd1.short_counts(seqs, sysd["preds"]); mf = sysd["tables"]["multi_frag"]
    if (kc["fp"], kc["fn"]) != (sh["fp"], sh["fn"]):
        raise SystemExit("the short-segment counts disagree with the evaluator's matching")
    return {**kc, **sh, "same_label_join_cuts": int(mf["false_split"].sum()), "same_label_joins": int(mf["same_join"].sum())}


def third_seed(rows: dict, counts: dict) -> dict:
    cells = [(s, h) for s in (0, 2) for h in HALVES]; out = {}
    F = lambda n, s, h: rows[f"{n} s{s}"][h]["segment_f1_50_multi"]  # noqa: E731
    for arm, name in PRIMARY.items():
        dF = {f"s{s} {h}": F(name, s, h) - F("A08 A + B3", s, h) for s, h in cells}
        dN = float(np.mean([rows[f"{name} s{s}"][h]["native_macro_f1"] - rows[f"A08 A + B3 s{s}"][h]["native_macro_f1"] for s, h in cells]))
        dBR = float(np.mean([rows[f"{name} s{s}"][h]["boundary_recall_multi"] - rows[f"A08 A + B3 s{s}"][h]["boundary_recall_multi"] for s, h in cells]))
        fs = {n: int(sum(counts[f"{n} s{s}"][h]["fn_short"] for s, h in cells)) for n in (name, "A08 A + B3")}
        half_means = {h: float(np.mean([dF[f"s{s} {h}"] for s in (0, 2)])) for h in HALVES}
        r = {"mean_dF": float(np.mean(list(dF.values()))), "dF": dF, "half_means": half_means, "mean_dN": dN, "mean_dBR": dBR, "missed_short_true": fs}
        r["signal"] = bool(r["mean_dF"] >= 0.003); r["no_reversal"] = bool(min(half_means.values()) >= -0.005 and min(dF.values()) >= -0.015)
        r["no_harm"] = bool(dN >= -0.003 and dBR >= -0.010 and fs[name] <= fs["A08 A + B3"] + 3); r["passes"] = bool(r["signal"] and r["no_reversal"] and r["no_harm"]); out[arm] = r
    out["train_seed1"] = bool(any(out[a]["passes"] for a in PRIMARY))
    return out


def run(ctx: vsh.Ctx, stage: str, workers: int) -> None:
    t0, started, ok, note, gsec = time.time(), vsh.now(), False, "", 0.0
    try:
        seeds = [0, 2] if stage == "screen" else [0, 1, 2]; models = models_of(ctx, seeds); a8 = a08_ctx(); rw.check_grid(a8)
        device = torch.device("cuda"); vsh.determinism(); torch.cuda.set_per_process_memory_fraction(0.6); per, choices = {}, {}
        for h in ("calibration", "selection"):
            per[h] = read_half(ctx, a8, h, models, seeds, device, workers); gsec += per[h]["gpu_seconds"]
            if h == "calibration":
                for key, mr in models.items():
                    ch = rcs.b3_choose(a8.cfg, rso.DIAG, per[h]["probs"][key], per[h]["ev"]); choices[f"{key}|b3"] = {x: ch[x] for x in ("status", "params", "n_feasible")}
                    if mr["arm"] in ("P", "V"):
                        hc = head_choose(a8.cfg, per[h]["probs"][key], per[h]["edges"][key], per[h]["seqs"], per[h]["ev"]); choices[f"{key}|head"] = {x: hc[x] for x in ("status", "params", "n_feasible")}
                        choices[f"{key}|head"]["grid"] = hc["grid"]
                    print(f"{key}: B3 {ch['status']} {ch['params']}" + (f"; head {choices[f'{key}|head']['status']} {choices[f'{key}|head']['params']}" if mr["arm"] != "R" else ""), flush=True)
                d = ctx.out / "evaluation" / stage; d.mkdir(parents=True, exist_ok=True)
                vsh.write_json(d / "choices.json", {"stage": stage, "models": {k: {x: v[x] for x in ("sha256", "identity")} for k, v in models.items()}, "choices": choices, "time": vsh.now()})
        systems = {}
        for h in ("calibration", "selection"):
            P = per[h]; ev, seqs = P["ev"], P["seqs"]
            for s in seeds:
                for name, probs, params in (("frozen Ordered", P["refs"][("frozen Ordered", s)], P["oparams"][s]), ("A08 A", P["refs"][("A08 A", s)], P["a08params"][s])):
                    systems.setdefault(f"{name} + B3 s{s}", {})[h] = rso.row_of(a8.cfg, decode(params, probs), ev)
                    systems.setdefault(f"{name} argmax s{s}", {})[h] = rso.row_of(a8.cfg, {q: p.argmax(1) for q, p in probs.items()}, ev)
                for key, mr in models.items():
                    if mr["seed"] != s:
                        continue
                    a, probs = mr["arm"], P["probs"][key]
                    systems.setdefault(f"{a} + B3 s{s}", {})[h] = rso.row_of(a8.cfg, decode(choices[f"{key}|b3"]["params"], probs), ev)
                    systems.setdefault(f"{a} argmax s{s}", {})[h] = rso.row_of(a8.cfg, {q: p.argmax(1) for q, p in probs.items()}, ev)
                    if a in ("P", "V"):
                        systems.setdefault(f"{a} head s{s}", {})[h] = rso.row_of(a8.cfg, head_decode(probs, P["edges"][key], seqs, choices[f"{key}|head"]["params"]["threshold"]), ev)
        rows = {k: {h: v[h]["row"] for h in v} for k, v in systems.items()}
        counts = {k: {h: counts_of(per[h]["seqs"], v[h]) for h in v} for k, v in systems.items() if (" + B3 " in k or " head " in k)}
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
        res = {"stage": stage, "config_sha256": sha256_file(ROOT / ctx.config), "seeds": seeds, "models": {k: {"sha256": v["sha256"]} for k, v in models.items()},
               "choices": choices, "systems": rows, "segment_counts": counts, "comparisons": cmp_, "accounting": acc, "verdicts": verdicts,
               "parity": per["calibration"]["parity"], "reproduced": {h: per[h]["reproduced"] for h in HALVES}, "half_counts": {h: per[h]["counts"] for h in HALVES},
               "access": {"halves": list(HALVES), "time": started}, "tuning_budget": {"R": "B3 40 settings", "P_V_head": f"{len(QUANTILES)} settings", "P_V_B3_sensitivity": "B3 40 settings"}}
        if stage == "screen":
            res["third_seed"] = third_seed(rows, counts)
        ids = rw.ids_of(a8, {h: per[h]["seqs"] for h in HALVES}); flat = json.loads(json.dumps(res, default=str)); assert_no_ids(flat, ids, stage)
        d = ctx.out / "evaluation" / stage
        for p in (d / f"{stage}.json", ctx.results / f"{stage}.json"):
            p.parent.mkdir(parents=True, exist_ok=True); vsh.write_json(p, flat)
        if stage == "screen":
            vsh.write_json(d / "decision.json", {**flat["third_seed"], "config_sha256": res["config_sha256"], "time": vsh.now()})
            print(json.dumps({a: {k: flat["third_seed"][a][k] for k in ("mean_dF", "half_means", "mean_dN", "mean_dBR", "missed_short_true", "passes")} for a in PRIMARY}, indent=1),
                  "train_seed1:", flat["third_seed"]["train_seed1"], flush=True)
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        total = time.time() - t0; g = min(gsec, total)
        vsh.ledger_append(ctx.ledger, f"evaluate {stage} (GPU inference)", "gpu", started, g, ok, note)
        vsh.ledger_append(ctx.ledger, f"evaluate {stage} (CPU)", "cpu", started, total - g, ok, note)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--stage", choices=("screen", "three_seed"), required=True); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(); a.set = []; run(vsh.Ctx(load_cfg(a), a.config), a.stage, a.workers)


if __name__ == "__main__":
    main()
