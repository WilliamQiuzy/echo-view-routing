#!/usr/bin/env python
"""A35 (configs/experiments/a5c_echofine_evidence_v1.yaml): frozen EchoFine evidence diagnostic, project-environment side.

  --stage plan     A32's 159,472 unique (recording, frame, gamma) references of both A31 pools -> plan.npz (paths, gamma)
  --stage base     A32 / A31 Base rows and denominators reproduced from the stored records (no Base inference)
  --stage readout  once, after run.json / probs.npy are committed by key: the fixed rule over every Base run, all P runs,
                   full sequences, coverage, the protocol's decision
EchoFine itself runs in envs/echofine through scripts/run_echofine_frames.py.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.b3_coverage as bc  # noqa: E402
import scripts.diagnose_a5c_evidence as d32  # noqa: E402
import scripts.report_video_switch_head as rsh  # noqa: E402
from scripts._common import load_cfg  # noqa: E402

import numpy as np  # noqa: E402

import echo_routing.temporal.echofine_rule as er  # noqa: E402
import echo_routing.temporal.interval_recognition as ir  # noqa: E402
from echo_routing.config import load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.ingest.frames import list_frame_paths  # noqa: E402

A32_CONFIG = "configs/experiments/a5c_source_external_evidence_v1.yaml"
SEEDS, FOLDS = d32.SEEDS, d32.FOLDS
SETS = ("all", "A4C_all", "A5C_all", "S1", "S1_persistent", "S2", "S3", "S4")


class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.config, self.v = config, cfg["echofine_evidence"]; o = self.v["outputs"]
        self.out, self.results, self.ledger = ROOT / o["root"], ROOT / o["results_dir"], ROOT / self.v["budget"]["ledger"]
        self.d = d32.Ctx(load_cfg(SimpleNamespace(config=A32_CONFIG, set=[])), A32_CONFIG)


def staged(ctx: Ctx, job: str, fn) -> None:
    t0, started, ok, note = time.time(), d32.now(), False, ""
    try:
        note = fn() or ""; ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        d32.ledger_append(ctx.ledger, job, "cpu", started, time.time() - t0, ok, note)


def publish(ctx: Ctx, name: str, rec: dict) -> None:
    flat = json.loads(json.dumps(rec, default=float)); d32.assert_no_ids(flat, d32.private_ids(ctx.d), name)
    for p in (ctx.out / name, ctx.results / name):
        p.parent.mkdir(parents=True, exist_ok=True); d32.write_json(p, flat)


def refs_of(ctx: Ctx) -> list:
    """A32's ordering: sorted unique (recording, frame, gamma) over both pools."""
    return sorted({(str(r[0]), int(r[1]), float(r[2])) for f in FOLDS for q in d32.t31.load_pool(ctx.d.a31, f) for r in q["refs"]})


def plan(ctx: Ctx) -> str:
    refs = refs_of(ctx); root = load_paths().ev9v_images; cache = {}; paths = []
    for v, fi, g in refs:
        if v not in cache:
            cache[v] = list_frame_paths(root / v)
        p = cache[v]; paths.append(str(p[min(fi, len(p) - 1)]))
    a32 = json.loads((ROOT / "runs/a5c_source_external_evidence_v1/echoprime/record.json").read_text())["unique_references"]
    if len(refs) != a32:
        raise SystemExit(f"{len(refs)} references against A32's {a32}")
    ctx.out.mkdir(parents=True, exist_ok=True); np.savez(ctx.out / "plan.npz", path=np.asarray(paths), gamma=np.asarray([g for _, _, g in refs]), frame=np.asarray([f for _, f, _ in refs]))
    rec = {"rows": len(refs), "recordings": len(cache), "gamma_values": sorted({g for _, _, g in refs}), "plan_sha256": sha256_file(ctx.out / "plan.npz"), "a32_unique_references": a32,
           "frames_clamped_to_last": int(sum(fi >= len(cache[v]) for v, fi, _ in refs))}
    publish(ctx, "plan.json", rec); print(json.dumps(rec)); return f"{len(refs)} rows"


def base(ctx: Ctx) -> str:
    """Base rows, runs and denominators reproduced from A32's stored records."""
    a32 = json.loads((ROOT / "docs/results/2026-10-25_a5c_source_external_evidence/readout.json").read_text()); a8 = rsh.a08_ctx(); data = d32.t31.vpr.training_data(ctx.d.a31.a15.a10.a07p); rec = {}
    for s in SEEDS:
        tot = Counter()
        for h in FOLDS:
            D = d32.fold_data(ctx.d, s, h); M = d32.set_masks(D); y, b = D["pk"]["labels"], D["pk"]["base"]; seqs = D["seqs"]
            P = {"seqs": seqs, "n_of": dict(data.n), "label_of": dict(data.label)}; sr = d32.t31.seq_rows(a8.cfg, P, D["base_preds"], bc.evalset(P, list(range(len(seqs))), f"heldout_f{h}"))
            old = a32["sequences"][f"s{s}_heldout_f{h}"]["A08 Base"]; rec[f"s{s}_f{h}_sequences_equal_A32"] = bool(sr["row"] == old["sequence_row"] and rsh.counts_of(seqs, sr) == old["counts"])
            tot.update({"P_runs": len(y), "base_wrong": int((b != y).sum()), "S1": int(M["S1"].sum()), "correct_A5C": int(((y == 3) & (b == 3)).sum())})
        rec[f"s{s}"] = dict(tot)
    ok = all(v for k, v in rec.items() if k.endswith("A32")) and rec["s0"]["P_runs"] == 6788 and rec["s2"]["P_runs"] == 6742 and (rec["s0"]["S1"], rec["s2"]["S1"]) == (103, 101) and (rec["s0"]["correct_A5C"], rec["s2"]["correct_A5C"]) == (1150, 1128)
    rec["ok"] = bool(ok); publish(ctx, "base.json", rec); print(json.dumps(rec))
    if not ok:
        raise SystemExit("Base rows or denominators differ from A32")
    return "Base reproduced"


# ----------------------------------------------------------------------------------------------- readout
def probs_table(ctx: Ctx) -> tuple:
    run = json.loads((ctx.out / "frames" / "run.json").read_text()); com = json.loads((ctx.out / "probs_committed.json").read_text())
    if sha256_file(ctx.out / "frames" / "probs.npy") != run["probs_sha256"] or com.get("probs_sha256") != run["probs_sha256"] or run["plan_sha256"] != sha256_file(ctx.out / "plan.npz"):
        raise SystemExit("probabilities are not the committed ones or belong to another plan")
    p = np.load(ctx.out / "frames" / "probs.npy"); refs = refs_of(ctx)
    if p.shape != (len(refs), 47) or not np.isfinite(p).all():
        raise SystemExit("probability table shape / values")
    return p, {r: i for i, r in enumerate(refs)}, run


def sample_probs(seq: dict, p: np.ndarray, at: dict) -> np.ndarray:
    return p[[at[(str(r[0]), int(r[1]), float(r[2]))] for r in seq["refs"]]]


def coverage(Q: np.ndarray, cats: list, base: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict:
    idx = np.flatnonzero(mask); c = Counter(cats[i] for i in idx); elig = [i for i in idx if base[i] in (er.A4C, er.A5C) and cats[i] in ("a4ch-full", "a5ch-full")]
    q = Q[idx] if len(idx) else np.zeros((1, 47))
    return {"runs": int(len(idx)), "top_class": dict(c), "eligible": len(elig), "changed": int((pred[idx] != base[idx]).sum()),
            "mean_prob": {"a4ch-full": float(q[:, er.COL_A4C_FULL].mean()), "a5ch-full": float(q[:, er.COL_A5C_FULL].mean()), "a5ch-outflow": float(q[:, er.COL_A5C_OUTFLOW].mean()),
                          "focused_A4C": float(q[:, list(er.COLS_FOCUSED_A4C)].sum(1).mean()), "a3ch-outflow": float(q[:, 6].mean()), "apex": float(q[:, 15].mean()),
                          "unmapped_total": float(1 - q[:, er.COL_A4C_FULL].mean() - q[:, er.COL_A5C_FULL].mean())}}


def readout(ctx: Ctx) -> str:
    if (ctx.out / "readout.json").exists():
        raise SystemExit("the readout exists; it runs once")
    p, at, run = probs_table(ctx); a8 = rsh.a08_ctx(); data = d32.t31.vpr.training_data(ctx.d.a31.a15.a10.a07p); a32 = json.loads((ROOT / "docs/results/2026-10-25_a5c_source_external_evidence/readout.json").read_text())
    rec = {"model": {k: run[k] for k in ("tf", "keras", "devices", "weights_sha256", "lookup_sha256", "plan_sha256", "probs_sha256", "key", "preprocessing")}, "seeds": {}, "sequences": {}, "coverage": {}}; priv = {}
    for s in SEEDS:
        cat = {k: [] for k in ("y", "b", "pred", "kind", "len", "pure", "sess", "Q", "cats")}; masks = []
        for h in FOLDS:
            D = d32.fold_data(ctx.d, s, h); M = d32.set_masks(D); pk, seqs = D["pk"], D["seqs"]; sp = {q["id"]: sample_probs(q, p, at) for q in seqs}
            Q = np.stack([sp[seqs[i]["id"]][x:y_].mean(0) for i, x, y_ in zip(pk["seq"], pk["a"], pk["b"])]); pred = np.asarray([er.decide(q, c) for q, c in zip(Q, pk["base"])])
            rl = {q["id"]: er.relabel(D["base_preds"][q["id"]], sp[q["id"]], ir.runs) for q in seqs}
            if not all(rl[seqs[i]["id"]][x] == pred[k] for k, (i, x) in enumerate(zip(pk["seq"], pk["a"]))):
                raise SystemExit("the run rule and the sequence relabelling disagree")
            P = {"seqs": seqs, "n_of": dict(data.n), "label_of": dict(data.label)}; ev = bc.evalset(P, list(range(len(seqs))), f"heldout_f{h}")
            sr = d32.t31.seq_rows(a8.cfg, P, rl, ev); rec["sequences"][f"s{s}_f{h}"] = {"EchoFine rule": {"sequence_row": sr["row"], "counts": rsh.counts_of(seqs, sr)}, "A08 Base": a32["sequences"][f"s{s}_heldout_f{h}"]["A08 Base"]}
            for k, v in (("y", pk["labels"]), ("b", pk["base"]), ("pred", pred), ("kind", D["kind"]), ("len", D["length"]), ("pure", D["pure"]), ("sess", np.char.add(f"f{h}:", D["sessions"].astype(str))), ("Q", Q),
                         ("cats", np.asarray([er.top_category(q) for q in Q]))):
                cat[k].append(v)
            masks.append(M); priv[(s, h)] = pred
        C = {k: np.concatenate(v) for k, v in cat.items()}; y, b, pred = C["y"], C["b"], C["pred"]; Mc = {k: np.concatenate([m[k] for m in masks]) for k in masks[0]}; Mc["all"] = np.ones(len(y), bool)
        nat = C["kind"] == "native"; ok5 = (y == 3) & (b == 3); other = ~np.isin(b, [er.A4C, er.A5C])
        parts = {"native": nat, "constructed": ~nat, "short_le16": C["len"] <= 16, "long": C["len"] > 16, "pure": C["pure"], "mixed": ~C["pure"]}
        r = {"all": d32.run_counts(y, b, pred), "changes": d32.change_table(y, b, pred), "core": d32.core_coverage(Mc, y, b, pred),
             "non_A4C_A5C_runs_unchanged": bool((pred[other] == b[other]).all()), "correct_A5C_kept": {"runs": int(ok5.sum()), "kept": int((ok5 & (pred == 3)).sum())},
             "by_part": {k: {**d32.run_counts(y, b, pred, m), "sessions_with_changes": len(set(C["sess"][m & (pred != b)]))} for k, m in parts.items()}}
        rec["seeds"][f"s{s}"] = r; rec["coverage"][f"s{s}"] = {n: coverage(C["Q"], list(C["cats"]), b, pred, Mc[n]) for n in SETS}
        s1 = Mc["S1"]; el = s1 & np.isin(C["cats"], ["a4ch-full", "a5ch-full"]); rec["coverage"][f"s{s}"]["S1_recovery"] = {"unconditional": f"{int((s1 & (pred == 3)).sum())}/{int(s1.sum())}", "within_eligible": f"{int((el & (pred == 3)).sum())}/{int(el.sum())}"}
    dec = {}
    for s in SEEDS:
        x = rec["seeds"][f"s{s}"]; n1 = x["core"]["S1_fixed_to_A5C"]["runs"]; f1 = x["core"]["S1_fixed_to_A5C"]["to_target"]; nk = x["correct_A5C_kept"]["runs"]; kk = x["correct_A5C_kept"]["kept"]
        dec[f"s{s}"] = {"S1_recovered": f1, "S1_needed": math.ceil(0.10 * n1), "correct_A5C_kept": kk, "kept_needed": math.ceil(0.98 * nk), "net_fixed": x["all"]["net_fixed"],
                        "S1_ok": f1 >= math.ceil(0.10 * n1), "kept_ok": kk >= math.ceil(0.98 * nk), "net_ok": x["all"]["net_fixed"] >= 0}
    cells = {k: v["EchoFine rule"]["sequence_row"]["segment_f1_50_multi"] >= v["A08 Base"]["sequence_row"]["segment_f1_50_multi"] for k, v in rec["sequences"].items()}
    dec["cells_f1_not_below_base"] = cells; dec["cells_ok"] = sum(cells.values()) >= 3
    dec["met"] = bool(all(dec[f"s{s}"][k] for s in SEEDS for k in ("S1_ok", "kept_ok", "net_ok")) and dec["cells_ok"]); rec["decision"] = dec
    rec["a32_context"] = {f"s{s}": {m: a32["runs"][str(s)][m]["all"] for m in ("EchoViewCLIP", "EchoPrime")} for s in SEEDS}
    np.save(ctx.out / "readout_private.npy", np.array({str(k): v for k, v in priv.items()}, dtype=object), allow_pickle=True); publish(ctx, "readout.json", rec)
    print(json.dumps({"seeds": {s: {k: v[k] for k in ("all", "correct_A5C_kept", "non_A4C_A5C_runs_unchanged")} for s, v in rec["seeds"].items()}, "decision": dec}, indent=1)); return "readout"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--stage", choices=("plan", "base", "readout"), required=True)
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config); ctx.out.mkdir(parents=True, exist_ok=True)
    staged(ctx, f"{a.stage} (CPU)", {"plan": lambda: plan(ctx), "base": lambda: base(ctx), "readout": lambda: readout(ctx)}[a.stage])


if __name__ == "__main__":
    main()
