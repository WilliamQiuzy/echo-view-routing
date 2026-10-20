#!/usr/bin/env python
"""A11 stage one (configs/experiments/video_sustained_switch_v1.yaml, stage_one): does B3's left/right argmax check change
A10's head decoding? CPU only, on A10's cached per-window outputs, frozen heads and committed choices.

For P_s0, P_s2, V_s0 and V_s2: the original head decoding (boundary_decode) against the same candidates, NMS 5 and segment
naming with B3's check (a peak at i is kept only if argmax(mean prob over [i-h, i)) != argmax(mean prob over [i, i+h)), h =
the model's committed P/V + B3 half-window). (a) Both at the committed numeric threshold on both halves. (b) The checked
version chosen once among the same 8 calibration quantiles with A08's rule, written before selection is read, then applied
to selection. Development halves are reused for exploration.

  python scripts/diagnose_switch_head_check.py --config configs/experiments/video_sustained_switch_v1.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.report_center_supervision as rcs  # noqa: E402
import scripts.report_raw_video_context as rr  # noqa: E402
import scripts.report_segment_objective as rso  # noqa: E402
import scripts.report_video_switch_head as rsh  # noqa: E402
import scripts.report_video_within_view as rw  # noqa: E402
import scripts.segment_objective_tables as st  # noqa: E402
import scripts.train_video_switch_head as vsh  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.ev9v_complementarity import decode  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.temporal.baselines.b3_js_divergence import labels_from_boundaries  # noqa: E402
from echo_routing.temporal.segments import non_max_suppression  # noqa: E402

KEYS = ("P_s0", "P_s2", "V_s0", "V_s2")
SHORT = 16


def checked_decode(probs: dict, edges: dict, seqs: list[dict], threshold: float, h: int) -> dict:
    """The original candidates, NMS and naming, with B3's left/right argmax check after NMS."""
    out = {}
    for q in seqs:
        p = np.asarray(probs[q["id"]], float); n = p.shape[0]; cut, s = edges[q["id"]]; b = np.zeros(n)
        for i, v in zip(cut, s):
            if 0 < i < n:
                b[i] = max(b[i], v)
        keep = []
        for i in non_max_suppression(b, threshold, rsh.NMS):
            lo, hi = max(0, i - h), min(n, i + h)
            if i > 0 and p[lo:i].mean(0).argmax() != p[i:hi].mean(0).argmax():
                keep.append(int(i))
        out[q["id"]] = labels_from_boundaries(p, np.asarray(keep, int))
    return out


def native_scores(edges: dict, seqs: list[dict]) -> np.ndarray:
    return np.concatenate([edges[q["id"]][1] for q in seqs if q["kind"] == "native" and edges[q["id"]][1].size] or [np.zeros(1)])


def kept_recovered(seqs: list[dict], ref_pred: dict, orig: dict, chk: dict) -> dict:
    """Reference segments that the original V decoding matched and A08 A did not: how many the checked decoding keeps."""
    out = {"recovered": 0, "kept": 0, "recovered_short": 0, "kept_short": 0}
    for q in seqs:
        if q["kind"] != "multi":
            continue
        a = st.segment_status(q["labels"], orig[q["id"]], ref_pred[q["id"]]); c = st.segment_status(q["labels"], chk[q["id"]], orig[q["id"]])
        for j, (sa, seg) in enumerate(zip(a["status"], a["ref"])):
            if sa == "a only":
                short = seg.length <= SHORT; kept = c["status"][j] in ("both", "a only")
                out["recovered"] += 1; out["kept"] += int(kept); out["recovered_short"] += int(short); out["kept_short"] += int(kept and short)
    return out


def summary(sysd: dict, seqs: list[dict]) -> dict:
    r, c = sysd["row"], rsh.counts_of(seqs, sysd)
    return {"F": r["segment_f1_50_multi"], "BR": r["boundary_recall_multi"], "BP": r["boundary_precision_multi"], "N": r["native_macro_f1"],
            "fp": c["fp"], "fp_short": c["fp_short"], "fn": c["fn"], "fn_short": c["fn_short"], "join_cuts": c["same_label_join_cuts"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument("--config", required=True)
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); vc = cfg["video_sustained_switch"]; out_dir = ROOT / vc["outputs"]["root"] / "stage_one"; out_dir.mkdir(parents=True, exist_ok=True)
    ledger = ROOT / vc["budget"]["ledger"]; t0, started, ok, note = time.time(), vsh.now(), False, ""
    try:
        p10 = vc["source"]["a10_config"]; vsh.pinned(p10); a10 = vsh.Ctx(load_cfg(SimpleNamespace(config=p10["path"], set=[])), p10["path"])
        a8 = rsh.a08_ctx(); rw.check_grid(a8); ch10 = json.loads((a10.out / "evaluation" / "screen" / "choices.json").read_text())
        heads, man = {}, {}
        for key in KEYS:
            m = json.loads((a10.out / "train" / key / "run_manifest.json").read_text()); man[key] = m; w = ROOT / "models_frozen" / vsh.NS / key / "final.pt"
            if sha256_file(w) != m["final_sha256"] or ch10["models"][key]["sha256"] != m["final_sha256"]:
                raise SystemExit(f"{key}: frozen weights differ from the run record or the A10 choices")
            hd = vsh.build_heads(int(m["seed"])); hd.load_state_dict(torch.load(w, map_location="cpu", weights_only=True)["heads"]); heads[key] = hd.eval()
        pins = rsh.a08_output_pins(); a8sum = json.loads((ROOT / "docs/results/2026-10-16_video_within_view/summary.json").read_text()); a8ch = json.loads(vsh.pinned(a10.src["a08_choices"]).read_text())["choices"]
        rec = {"config_sha256": sha256_file(ROOT / a.config), "started": started, "models": {k: man[k]["final_sha256"] for k in KEYS}, "cells": {}, "choices": {}}
        data = {}
        for h in ("calibration", "selection"):                                     # calibration first: its choices are written before selection is read
            ev, seqs = a8.half(h, rso.DevAccess({h})); refs = rw.half_refs(seqs); at = {r: i for i, r in enumerate(refs)}; W = rr.half_windows(seqs, at, 5)
            orders = np.broadcast_to(np.arange(16, dtype=np.int64), W["s"].shape).copy(); A = {}
            for s in (0, 2):
                apath = ROOT / "runs/video_within_view_v1/evaluation" / h / f"all_s{s}.npz"
                if sha256_file(apath) != pins[h][s]:
                    raise SystemExit("A08 saved outputs differ from their pin")
                lgA, _ = rr.cached_logits(apath, a8sum["runs"][f"all_s{s}"]["identity"], W, orders, seqs); A[s] = decode(a8ch[f"all_s{s}"]["params"], rr.per_sample(W, lgA, seqs))
            for key in KEYS:
                arm, s = key.split("_s")[0], int(key.split("_s")[1])
                with np.load(a10.out / "evaluation" / h / f"{key}.npz") as z:
                    if str(z["identity"]) != man[key]["identity"] or not np.array_equal(z["seq"], W["seq"]) or not np.array_equal(z["centre"], W["centre"]):
                        raise SystemExit(f"{key} {h}: cached outputs differ from the run identity or this half's windows")
                    lg, em = z["logits"].astype(np.float32), z["emb"].astype(np.float32)
                probs = rr.per_sample(W, lg, seqs); edges = rsh.edge_scores(arm, heads[key], W, lg, em, seqs, torch.device("cpu"))
                thr = float(ch10["choices"][f"{key}|head"]["params"]["threshold"]); hw = int(ch10["choices"][f"{key}|b3"]["params"]["half_window"])
                orig = rso.row_of(a8.cfg, rsh.head_decode(probs, edges, seqs, thr), ev); chk = rso.row_of(a8.cfg, checked_decode(probs, edges, seqs, thr, hw), ev)
                cell = {"threshold": thr, "half_window": hw, "original": summary(orig, seqs), "checked_at_original": summary(chk, seqs)}
                if h == "calibration":
                    nat = native_scores(edges, seqs); recs = []
                    for order, qn in enumerate(rsh.QUANTILES):
                        tq = float(np.quantile(nat, qn)); m_, _ = rcs.evaluate(a8.cfg, checked_decode(probs, edges, seqs, tq, hw), ev)
                        recs.append({"order": order, "params": {"threshold": tq, "threshold_quantile": qn}, **{k: m_[k] for k in rcs.FLAT}})
                    sel = rcs.select_setting(recs, 0.980, 0.95)
                    choice = sel["chosen"]["params"] if sel["chosen"] is not None else next(r["params"] for r in recs if abs(r["params"]["threshold_quantile"] - rsh.DIAG_Q) < 1e-9)
                    rec["choices"][key] = {"status": "feasible" if sel["chosen"] is not None else "infeasible (diagnostic)", "params": choice, "grid": recs}
                    vsh.write_json(out_dir / "checked_choices.json", {"choices": rec["choices"], "time": vsh.now()})
                tc = float(rec["choices"][key]["params"]["threshold"]); chs = rso.row_of(a8.cfg, checked_decode(probs, edges, seqs, tc, hw), ev); cell["checked_at_choice"] = summary(chs, seqs)
                cell["A08_A"] = summary(rso.row_of(a8.cfg, A[s], ev), seqs)
                if arm == "V":
                    cell["v_recovered_vs_A08_kept_at_original"] = kept_recovered(seqs, A[s], orig["preds"], chk["preds"])
                    cell["v_recovered_vs_A08_kept_at_choice"] = kept_recovered(seqs, A[s], orig["preds"], chs["preds"])
                rec["cells"][f"{key}|{h}"] = cell; print(f"{key} {h}: original F {cell['original']['F']:.4f}, checked@orig {cell['checked_at_original']['F']:.4f}, checked@choice {cell['checked_at_choice']['F']:.4f}", flush=True)
            data[h] = seqs
        mo = float(np.mean([rec["cells"][f"{k}|{h}"]["original"]["F"] for k in KEYS for h in ("calibration", "selection")]))
        mc = float(np.mean([rec["cells"][f"{k}|{h}"]["checked_at_choice"]["F"] for k in KEYS for h in ("calibration", "selection")]))
        rec["mapping"] = {"mean_F_original": mo, "mean_F_checked_at_choice": mc, "enable_check": bool(mc > mo)}
        ids = rw.ids_of(a8, data); flat = json.loads(json.dumps(rec, default=str)); assert_no_ids(flat, ids, "stage one")
        res = ROOT / vc["outputs"]["results_dir"]; res.mkdir(parents=True, exist_ok=True)
        for p in (out_dir / "stage_one.json", res / "stage_one.json"):
            vsh.write_json(p, flat)
        vsh.write_json(out_dir / "decision.json", {**rec["mapping"], "config_sha256": rec["config_sha256"], "time": vsh.now()})
        print(json.dumps(rec["mapping"]), flush=True); ok = True; note = json.dumps(rec["mapping"])
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        vsh.ledger_append(ledger, "stage one (CPU decoding check on cached A10 outputs)", "cpu", started, time.time() - t0, ok, note)


if __name__ == "__main__":
    main()
