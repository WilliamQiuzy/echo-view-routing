#!/usr/bin/env python
"""Evaluate A09-R1 (configs/experiments/video_discriminative_r1.yaml) on both development halves.

The final update-3,750 C models (C3750) are compared with their own pilot snapshots (C1250, whose cached pilot outputs and
pilot B3 choices are reused and reproduced against pilot.json), frozen Ordered and A08 A. The half reader, inference,
references and rows are A09's pinned evaluator functions; each C3750 gets one B3 choice on calibration with the same grid
and rule. Exploratory development feedback only.

  --stage two_seed     seeds 0 and 2, and the fixed seed-1 rule (decision.json)
  --stage three_seed   seeds 0, 1 and 2 after the conditional seed-1 run, with three-seed verdicts
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.report_center_supervision as rcs  # noqa: E402
import scripts.report_segment_objective as rso  # noqa: E402
import scripts.report_video_discriminative as rd  # noqa: E402
import scripts.report_video_within_view as rw  # noqa: E402
import scripts.segment_objective_tables as st  # noqa: E402
import scripts.train_video_discriminative as td  # noqa: E402
import scripts.train_video_discriminative_r1 as tr  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402
from scripts.token_stability_attribution_tables import match_pairs  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.temporal.segments import labels_to_segments  # noqa: E402

HALVES = rd.HALVES
SHORT = 16
NS = "video_discriminative_r1"
PAIRS = {"C3750 - C1250": ("C3750", "C1250"), "C3750 - frozen Ordered": ("C3750", "frozen Ordered"), "C3750 - A08 A": ("C3750", "A08 A")}


def models_of(r1: tr.R1, seeds: list[int]) -> tuple[dict, dict]:
    """(final models, pilot models). Finals come from this round's run records (the frozen copy when present); pilot
    snapshots keep the parent's pilot identity string, so their cached pilot outputs are reused."""
    fin, pil = {}, {}
    for s in seeds:
        rec = json.loads((r1.out / "train" / f"{tr.ARM}_s{s}" / "r1_run.json").read_text()); man = json.loads((r1.out / "train" / f"{tr.ARM}_s{s}" / "run_manifest.json").read_text())
        fz = ROOT / "models_frozen" / NS / f"{tr.ARM}_s{s}" / "final.pt"; w = fz if fz.is_file() else Path(man["final_checkpoint"])
        fin[f"c3750_s{s}"] = {"weights": str(w), "sha256": man["final_sha256"], "identity": rec["r1_identity"], "lambda": 0.0}
        if s in r1.r["parent"]["runs"]:
            p = r1.r["parent"]["runs"][s]
            pil[f"c1250_s{s}"] = {"weights": str(td.pinned(p["pilot_snapshot"])), "sha256": p["pilot_snapshot"]["sha256"], "identity": f"{p['identity']}@u{tr.PILOT}", "lambda": 0.0}
    return fin, pil


def seed_cache(r1: tr.R1, pil: dict) -> None:
    """Copy the parent's cached pilot outputs (read only at their source) into this round's output directory."""
    for key in pil:
        s = int(key.rsplit("_s", 1)[1])
        for h in HALVES:
            src = r1.a09.out / "evaluation" / "pilot" / h / f"{tr.ARM}_s{s}.npz"; dst = r1.out / "evaluation" / "final" / h / f"{key}.npz"
            if not dst.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dst)
            if sha256_file(dst) != sha256_file(src):
                raise SystemExit(f"{dst.name}: the copied pilot outputs differ from their source")


def short_counts(seqs: list[dict], preds: dict) -> dict:
    """Multi-stream counts: false segments (all, at most 16 samples), missed reference segments (all, short) and the
    short reference segments with their matches."""
    out = dict.fromkeys(("fp", "fp_short", "fn", "fn_short", "ref_short", "tp_short"), 0)
    for q in seqs:
        if q["kind"] != "multi":
            continue
        pa, ref = labels_to_segments(np.asarray(preds[q["id"]])), labels_to_segments(np.asarray(q["labels"])); m = match_pairs(pa, ref)
        mp, mr = {i for i, _ in m}, {j for _, j in m}
        for i, p in enumerate(pa):
            if i not in mp:
                out["fp"] += 1; out["fp_short"] += int(p.length <= SHORT)
        for j, t in enumerate(ref):
            short = t.length <= SHORT; out["ref_short"] += int(short)
            if j in mr:
                out["tp_short"] += int(short)
            else:
                out["fn"] += 1; out["fn_short"] += int(short)
    return out


def seed1_decision(rule_cells: dict, counts: dict, acc: dict) -> dict:
    """The configuration's seed1_rule on seeds 0 and 2 (applied once)."""
    cells = [(s, h) for s in (0, 2) for h in HALVES]
    F = lambda n, s, h: rule_cells[f"{n} + B3 s{s}"][h]["segment_f1_50_multi"]  # noqa: E731
    N = lambda n, s, h: rule_cells[f"{n} argmax s{s}"][h]["native_macro_f1"]  # noqa: E731
    BR = lambda n, s, h: rule_cells[f"{n} + B3 s{s}"][h]["boundary_recall_multi"]  # noqa: E731
    E = lambda n: sum(rule_cells[f"{n} argmax s{s}"][h]["A4C read as A5C"] + rule_cells[f"{n} argmax s{s}"][h]["A5C read as A4C"] for s, h in cells)  # noqa: E731
    FPsel = lambda n: sum(counts[f"{n} + B3 s{s}"]["selection"]["fp"] for s in (0, 2))  # noqa: E731
    FN = lambda n: sum(counts[f"{n} + B3 s{s}"][h]["fn"] for s, h in cells)  # noqa: E731
    dF_pilot = float(np.mean([F("C3750", s, h) - F("C1250", s, h) for s, h in cells])); dF_a08 = float(np.mean([F("C3750", s, h) - F("A08 A", s, h) for s, h in cells]))
    dN = float(np.mean([N("C3750", s, h) - N("frozen Ordered", s, h) for s, h in cells])); dBR = float(np.mean([BR("C3750", s, h) - BR("frozen Ordered", s, h) for s, h in cells]))
    short_net = int(sum(acc[f"s{s}|C3750 - frozen Ordered|{h}"]["multi"]["lost"]["short"] - acc[f"s{s}|C3750 - frozen Ordered|{h}"]["multi"]["recovered"]["short"] for s, h in cells))
    rec = {"mean_dF_vs_C1250": dF_pilot, "selection_FP": {"C3750": FPsel("C3750"), "C1250": FPsel("C1250")}, "mean_dF_vs_A08A": dF_a08, "mean_dN_vs_Ordered": dN,
           "E45_plus_E54": {"C3750": E("C3750"), "frozen Ordered": E("frozen Ordered")}, "mean_dBR_vs_Ordered": dBR, "FN": {"C3750": FN("C3750"), "frozen Ordered": FN("frozen Ordered")},
           "short_lost_minus_recovered_vs_Ordered": short_net}
    rec["recovery"] = bool(dF_pilot >= 0.010 and FPsel("C3750") <= 0.70 * FPsel("C1250"))
    rec["strong_reference"] = bool(dF_a08 >= -0.005)
    rec["no_degradation"] = bool(dN >= -0.003 and E("C3750") <= E("frozen Ordered") and dBR >= -0.010 and FN("C3750") <= 1.15 * FN("frozen Ordered") and short_net <= 3)
    rec["train_seed1"] = bool(rec["recovery"] and rec["strong_reference"] and rec["no_degradation"])
    return rec


def run(r1: tr.R1, stage: str, workers: int) -> None:
    t0, started, ok, note, gsec = time.time(), td.now(), False, "", 0.0
    try:
        if tr.other_gpu_processes():
            raise SystemExit("another GPU process is present; the evaluation does not start")
        seeds = [0, 2] if stage == "two_seed" else [0, 1, 2]; fin, pil = models_of(r1, seeds); seed_cache(r1, pil); models = {**fin, **pil}
        a8 = rd.a08_ctx(r1.a09); rw.check_grid(a8); device = torch.device("cuda"); td.determinism(); torch.cuda.set_per_process_memory_fraction(0.6)
        pchoices = json.loads(td.pinned(r1.r["parent"]["pilot_choices"]).read_text())
        pilot_rows = json.loads(td.pinned(r1.r["parent"]["pilot_record"]).read_text())["systems"]
        per, choices = {}, {}
        for h in ("calibration", "selection"):
            per[h] = rd.read_half(r1.t, a8, h, models, "final", device, workers, seeds); gsec += per[h]["gpu_seconds"]
            if h == "calibration":
                for key in fin:
                    ch = rcs.b3_choose(a8.cfg, rso.DIAG, per[h]["probs"][key], per[h]["ev"]); choices[key] = {x: ch[x] for x in ("status", "params", "n_feasible")}
                    print(f"{key}: {ch['status']} {ch['params']}", flush=True)
                for key, mr in pil.items():
                    pc = pchoices["choices"][key.replace("c1250", tr.ARM)]
                    if pchoices["models"][key.replace("c1250", tr.ARM)]["sha256"] != mr["sha256"]:
                        raise SystemExit(f"{key}: the pilot choice belongs to other weights")
                    choices[key] = {**pc, "source": "A09 pilot choice"}
                d = r1.out / "evaluation" / stage; d.mkdir(parents=True, exist_ok=True)
                td.write_json(d / "choices.json", {"stage": stage, "models": {k: {x: v[x] for x in ("sha256", "identity")} for k, v in models.items()}, "choices": choices, "time": td.now()})
        systems = {}
        for h in ("calibration", "selection"):
            P = per[h]
            for s in seeds:
                op = P["oparams"][s]
                for name, probs, params in (("frozen Ordered", P["refs"][("Ordered", s)], op), ("A08 A", P["refs"][("A08 A", s)], P["a08params"][s])):
                    r = rd.rows_of(a8.cfg, probs, P["ev"], params, None)
                    systems.setdefault(f"{name} + B3 s{s}", {})[h] = r["b3"]; systems.setdefault(f"{name} argmax s{s}", {})[h] = r["argmax"]
                for key in models:
                    if int(key.rsplit("_s", 1)[1]) != s:
                        continue
                    n = "C3750" if key.startswith("c3750") else "C1250"; r = rd.rows_of(a8.cfg, P["probs"][key], P["ev"], choices[key]["params"], op)
                    for dec, lab in (("b3", "+ B3"), ("argmax", "argmax"), ("ordered_setting", "at Ordered setting")):
                        systems.setdefault(f"{n} {lab} s{s}", {})[h] = r[dec]
        for s in [x for x in seeds if f"c1250_s{x}" in pil]:                 # the reused pilot outputs reproduce the pilot rows
            for h in HALVES:
                rso.check_stored(systems[f"C1250 + B3 s{s}"][h]["row"], pilot_rows[f"C + B3 s{s}"][h], f"C1250 + B3 s{s} {h}")
                rso.check_stored(systems[f"C1250 argmax s{s}"][h]["row"], pilot_rows[f"C argmax s{s}"][h], f"C1250 argmax s{s} {h}")
        rows = {k: {h: v[h]["row"] for h in v} for k, v in systems.items()}
        counts = {}
        for k, v in systems.items():
            if " + B3 " not in k:
                continue
            for h in v:
                kc, sh = st.kind_counts(per[h]["seqs"], v[h]["preds"], v[h]["tables"])["multi"], short_counts(per[h]["seqs"], v[h]["preds"])
                if (kc["fp"], kc["fn"]) != (sh["fp"], sh["fn"]):
                    raise SystemExit(f"{k} {h}: the short-segment counts disagree with the evaluator's matching")
                counts.setdefault(k, {})[h] = {**kc, **sh}
        cmp_, acc = {}, {}
        for h in HALVES:
            for s in seeds:
                draws = {}
                for c, (a, b) in PAIRS.items():
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
            for c in ("C3750 - frozen Ordered", "C3750 - A08 A"):
                for m in st.BOOT_METRICS:
                    verdicts[f"{c}|{m}"] = {h: st.consistency([cmp_[f"s{s}|{c}|{h}"][m] for s in seeds]) for h in HALVES}
        res = {"stage": stage, "config_sha256": sha256_file(ROOT / r1.config), "seeds": seeds, "models": {k: {x: v[x] for x in ("sha256",)} for k, v in models.items()},
               "choices": choices, "systems": rows, "segment_counts": counts, "comparisons": cmp_, "accounting": acc, "verdicts": verdicts,
               "parity": per["calibration"]["parity"], "reproduced": {h: per[h]["reproduced"] + [f"C1250 s{s} B3 and argmax {h} (pilot.json)" for s in seeds if f"c1250_s{s}" in pil] for h in HALVES},
               "half_counts": {h: per[h]["counts"] for h in HALVES}, "access": {"halves": list(HALVES), "time": started}}
        if stage == "two_seed":
            res["seed1_decision"] = seed1_decision(rows, counts, acc)
        ids = rw.ids_of(a8, {h: per[h]["seqs"] for h in HALVES}); flat = json.loads(json.dumps(res, default=str)); assert_no_ids(flat, ids, stage)
        d = r1.out / "evaluation" / stage
        for p in (d / f"{stage}.json", r1.results / f"{stage}.json"):
            p.parent.mkdir(parents=True, exist_ok=True); td.write_json(p, flat)
        if stage == "two_seed":
            td.write_json(d / "decision.json", {**flat["seed1_decision"], "config_sha256": res["config_sha256"], "time": td.now()})
            print(json.dumps(flat["seed1_decision"], indent=1), flush=True)
        ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        total = time.time() - t0; g = min(gsec, total)
        td.ledger_append(r1.ledger, f"R1 evaluate {stage} (GPU inference)", "gpu", started, g, ok, note)
        td.ledger_append(r1.ledger, f"R1 evaluate {stage} (CPU)", "cpu", started, total - g, ok, note)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--stage", choices=("two_seed", "three_seed"), required=True); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(); a.set = []; run(tr.R1(load_cfg(a), a.config), a.stage, a.workers)


if __name__ == "__main__":
    main()
