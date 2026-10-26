#!/usr/bin/env python
"""post_a35_residual_error_inventory_v1 (configs/experiments/post_a35_residual_error_inventory_v1.yaml): residual
full-segmentation errors of the current Base (A08 + B3, seeds 0 / 2) on A31's training-source pools, from cached outputs.

Reproduces each seed x held-out fold cell with the original evaluator, then classifies every unmatched reference unit
with failure_units.unit_outcomes (fixed priority: label, missed_cut, extra_cut, boundary_shift; near 10, hit 5) and
counts unmatched predicted segments, reference change pairs and the existing per-sample probability evidence.
Aggregates only; nothing is re-inferred.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.b3_coverage as bc  # noqa: E402
import scripts.diagnose_a5c_evidence as d32  # noqa: E402
import scripts.report_raw_video_context as rr  # noqa: E402
import scripts.report_video_switch_head as rsh  # noqa: E402
from scripts._common import load_cfg  # noqa: E402

import numpy as np  # noqa: E402

from echo_routing.evaluate.failure_units import Unit, segment_counts, session_of, unit_outcomes  # noqa: E402
from echo_routing.evaluate.metrics_segmentation import temporal_iou  # noqa: E402
from echo_routing.temporal.segments import Segment, labels_to_segments  # noqa: E402

A32_CONFIG = "configs/experiments/a5c_source_external_evidence_v1.yaml"
CLASSES = ["PLAX", "PSAX", "A4C", "A5C", "SC4C"]
SHORT = 16


def units_of(q: dict) -> list:
    """Reference units: runs of the reference labels (same-class fragments merge); sources kept per unit."""
    lab = np.asarray(q["labels"], int); cuts = np.flatnonzero(np.diff(lab)) + 1; st = np.r_[0, cuts]; en = np.r_[cuts, len(lab)]; out = []
    for i, (a, b) in enumerate(zip(st, en)):
        src = tuple(sorted({(str(r[0]), str(r[2]), 0, 0) for r in q["refs"][a:b]}))
        out.append(Unit(q["id"], i, int(a), int(b), int(lab[a]), int(lab[st[i - 1]]) if i else None, int(lab[st[i + 1]]) if i + 1 < len(st) else None, src))
    return out


def pair_name(a: int, b: int) -> str:
    x, y = sorted((CLASSES[a], CLASSES[b])); return f"{x}<->{y}"


def cell(ctx, s: int, h: int, a8cfg, data) -> tuple[dict, dict]:
    D = d32.fold_data(ctx, s, h); seqs = D["seqs"]; base = D["base_preds"]; W = d32.t31.pool_W(seqs); O = d32.t31.outputs(ctx.a31, s, h); ps = rr.per_sample(W, O["logits"], seqs)
    P = {"seqs": seqs, "n_of": dict(data.n), "label_of": dict(data.label)}; sr = d32.t31.seq_rows(a8cfg, P, base, bc.evalset(P, list(range(len(seqs))), f"heldout_f{h}"))
    counts = rsh.counts_of(seqs, sr); rec = {"evaluator": {"row": sr["row"], "counts": counts}}
    tot = {k: Counter() for k in ("units", "fn", "pred", "fp", "units_short", "fn_short", "pred_short", "fp_short")}; forms = defaultdict(Counter); comp = Counter(); priv = []
    rank = defaultdict(list); fracc = defaultdict(list); changes = Counter(); missed = Counter(); side = Counter(); fp_host = Counter(); fp_inside = Counter(); rec_src = defaultdict(set)
    rec_cnt = Counter()
    for q in seqs:
        kind = q["kind"]; pred = np.asarray(base[q["id"]], int); units = units_of(q); rows, spur = unit_outcomes(pred, units); pr = ps[q["id"]]; amax = pr.argmax(1)
        m, npred, nunit = segment_counts(pred, units); rec_cnt[(kind, "matched")] += m; rec_cnt[(kind, "pred")] += npred; rec_cnt[(kind, "units")] += nunit
        pseg = labels_to_segments(pred)
        for u, r in zip(units, rows):
            k = (kind, CLASSES[u.label]); sh = u.length <= SHORT; tot["units"][k] += 1; tot["units_short"][k] += sh
            mean = pr[u.start:u.end].mean(0); rk = int((mean > mean[u.label]).sum()) + 1; fc = float((amax[u.start:u.end] == u.label).mean())
            if r["matched"]:
                rank[(kind, "matched")].append(rk); fracc[(kind, "matched")].append(fc); continue
            tot["fn"][k] += 1; tot["fn_short"][k] += sh; f = r["cause"]
            if f == "label":
                f = "label A4C<->A5C" if {u.label, r["best_label"]} == {2, 3} else "label other"
            forms[(kind, f)][CLASSES[u.label]] += 1; rank[(kind, f)].append(rk); fracc[(kind, f)].append(fc)
            for v, _, _, _ in u.sources:
                rec_src[(kind, f)].add(v)
            if any(p.label == u.label and temporal_iou(p, Segment(u.start, u.end, u.label)) >= 0.5 for p in pseg):
                comp[(kind, f)] += 1
            priv.append({"stream": q["id"], "unit": u.index, "form": f})
        for j in range(1, len(units)):
            L, R = units[j - 1], units[j]; pn = pair_name(L.label, R.label); changes[(kind, pn)] += 1
            hit = rows[j]["edge_dist"][0] if rows[j]["edge_dist"] else None              # left edge of unit j = change j
            if hit is None:
                missed[(kind, pn)] += 1
                lok = np.bincount(amax[L.start:L.end], minlength=5).argmax() == L.label; rok = np.bincount(amax[R.start:R.end], minlength=5).argmax() == R.label
                side[(kind, pn, "both sides argmax-correct" if lok and rok else ("one side wrong" if lok != rok else "both sides wrong"), "shorter side <= 16" if min(L.length, R.length) <= SHORT else "both sides > 16")] += 1
        for sp in spur:
            k = (kind, CLASSES[sp["label"]]); fp_host[(kind, CLASSES[sp["label"]], "host " + CLASSES[units[sp["host"]].label])] += 1; fp_inside[(kind, "inside host" if sp["inside"] else "straddles")] += 1
        for p in pseg:
            tot["pred"][(kind, CLASSES[p.label])] += 1; tot["pred_short"][(kind, CLASSES[p.label])] += (p.end - p.start) <= SHORT
        for sp in spur:
            tot["fp"][(kind, CLASSES[sp["label"]])] += 1; tot["fp_short"][(kind, CLASSES[sp["label"]])] += (sp["end"] - sp["start"]) <= SHORT
    mk = lambda c: {f"{a}|{b}": v for (a, b), v in c.items()}  # noqa: E731
    rec["reconcile"] = {"multi_matched": rec_cnt[("multi", "matched")], "evaluator_tp": counts["tp"], "multi_fp": rec_cnt[("multi", "pred")] - rec_cnt[("multi", "matched")], "evaluator_fp": counts["fp"],
                        "multi_fn": rec_cnt[("multi", "units")] - rec_cnt[("multi", "matched")], "evaluator_fn": counts["fn"], "native_units": rec_cnt[("native", "units")], "native_matched": rec_cnt[("native", "matched")],
                        "native_pred_segments": rec_cnt[("native", "pred")]}
    rec["totals"] = {k: mk(v) for k, v in tot.items()}
    rec["forms"] = {f"{a}|{b}": dict(v) for (a, b), v in forms.items()}; rec["competition"] = mk(comp)
    rec["form_recordings"] = {f"{a}|{b}": len(v) for (a, b), v in rec_src.items()}; rec["form_sessions"] = {f"{a}|{b}": len({session_of(x) for x in v}) for (a, b), v in rec_src.items()}
    rec["prob_rank"] = {f"{a}|{b}": {"units": len(v), "rank1_share": float(np.mean(np.asarray(v) == 1)), "median_rank": float(np.median(v))} for (a, b), v in rank.items()}
    rec["frac_argmax_true"] = {f"{a}|{b}": float(np.mean(v)) for (a, b), v in fracc.items()}
    rec["changes"] = mk(changes); rec["missed_changes"] = mk(missed); rec["missed_change_evidence"] = {"|".join(k): v for k, v in side.items()}
    rec["fp_by_host"] = {"|".join(k): v for k, v in fp_host.items()}; rec["fp_position"] = mk(fp_inside)
    lens = [u.length for q in seqs if q["kind"] == "multi" for u in units_of(q)]; rec["multi_unit_length_quartiles"] = np.percentile(lens, [25, 50, 75]).tolist()
    return rec, {"units": priv, "fn_recordings": {k: sorted(v) for k, v in rec_src.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); a = ap.parse_args(); t0, started = time.time(), d32.now()
    cfg = load_cfg(SimpleNamespace(config=a.config, set=[])); v = cfg["residual_error_inventory"]; out_root, res = ROOT / v["outputs"]["root"], ROOT / v["outputs"]["results_dir"]
    ctx = d32.Ctx(load_cfg(SimpleNamespace(config=A32_CONFIG, set=[])), A32_CONFIG); a8 = rsh.a08_ctx(); data = d32.t31.vpr.training_data(ctx.a31.a15.a10.a07p)
    a32 = json.loads((ROOT / "docs/results/2026-10-25_a5c_source_external_evidence/readout.json").read_text()); rec = {"cells": {}}; priv = {}; union = defaultdict(set)
    for s in d32.SEEDS:
        for h in d32.FOLDS:
            c, p = cell(ctx, s, h, a8.cfg, data); old = a32["sequences"][f"s{s}_heldout_f{h}"]["A08 Base"]
            c["reproduces_A32"] = bool(c["evaluator"]["row"] == old["sequence_row"] and c["evaluator"]["counts"] == old["counts"]); r = c["reconcile"]
            c["reconciled"] = bool(r["multi_matched"] == r["evaluator_tp"] and r["multi_fp"] == r["evaluator_fp"] and r["multi_fn"] == r["evaluator_fn"])
            c["forms_sum_to_fn"] = bool(sum(sum(x.values()) for k, x in c["forms"].items() if k.startswith("multi|")) == r["evaluator_fn"])
            rec["cells"][f"s{s}_f{h}"] = c; priv[f"s{s}_f{h}"] = p
            for k, vs in p["fn_recordings"].items():
                union[k] |= set(vs)
    rec["union_over_cells"] = {f"{a}|{b}": {"recordings": len(vs), "sessions": len({session_of(x) for x in vs})} for (a, b), vs in union.items()}
    rec["ok"] = all(c["reproduces_A32"] and c["reconciled"] and c["forms_sum_to_fn"] for c in rec["cells"].values())
    flat = json.loads(json.dumps(rec, default=float)); d32.assert_no_ids(flat, d32.private_ids(ctx), "inventory")
    out_root.mkdir(parents=True, exist_ok=True); res.mkdir(parents=True, exist_ok=True)
    for p in (out_root / "inventory.json", res / "inventory.json"):
        p.write_text(json.dumps(flat, indent=1, sort_keys=True))
    (out_root / "inventory_private.json").write_text(json.dumps({c: {"units": p["units"], "fn_recordings": {"|".join(k): vs for k, vs in p["fn_recordings"].items()}} for c, p in priv.items()}))
    os.chmod(out_root / "inventory_private.json", 0o600)
    d32.ledger_append(ROOT / v["budget"]["ledger"], "inventory (CPU)", "cpu", started, time.time() - t0, bool(rec["ok"]), "four cells")
    print(json.dumps({k: {kk: c[kk] for kk in ("reproduces_A32", "reconciled", "forms_sum_to_fn")} for k, c in rec["cells"].items()}), rec["ok"])
    if not rec["ok"]:
        raise SystemExit("reproduction, reconciliation or form sums failed")


if __name__ == "__main__":
    main()
