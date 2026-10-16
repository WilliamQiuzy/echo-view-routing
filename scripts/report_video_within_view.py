#!/usr/bin/env python
"""Evaluate the A08 within-view regularisation round once (docs/superpowers/plans/2026-10-16-video-within-view-regularization.md,
Tasks 3-4; configs/experiments/video_within_view_v1.yaml). Runs only after the six new models are finished and frozen.

Every half is read alone through A03's half_sets (labels of that half's clips only, references from the pinned window
plans) and only that half's frames are decoded (V01's decode_frames). The six new models give the ordinary single-window
CLS logits on V01's stride-5 windows (identity slot order, the marker slot); every 10 Hz sample takes its nearest centre's
softmax. F is A07's frozen plain-CE control: its saved per-window outputs of the half being read are checked (sha256,
run identity, windows), its probabilities recomputed from the stored logits must equal the stored ones, and its B3 row at
its A07 choice, its argmax row and its stored-setting row must equal A07's summary rows exactly. Frozen Ordered is V01's
saved outputs with V01's stored rows reproduced. The eager development loaders are forbidden throughout.

  --calibrate    calibration half only: each seed's V01 source and F re-inferred on the first 64 windows against their
                 saved logits, V01's and F's stored calibration rows reproduced, the six new models' calibration outputs
                 and the B3 grid of each (choices.json, written once; F keeps its A07 choices)
  --dry-run      the selection pipeline with the calibration half standing in for both halves (a code check)
  --proof        (local, needs git) the git-object proof of the committed choices.json
  --selection    after the barrier (choice sha256, commit and proof verified): the selection half opened once; the systems,
                 paired comparisons, verdicts, accounting, goals, training penalties and the fixed readings

  python scripts/report_video_within_view.py --config configs/experiments/video_within_view_v1.yaml --calibrate
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.prepare_raw_video_context as prv  # noqa: E402
import scripts.raw_video_context_tables as rt  # noqa: E402
import scripts.report_center_supervision as rcs  # noqa: E402
import scripts.report_raw_video_context as rr  # noqa: E402
import scripts.report_segment_objective as rso  # noqa: E402
import scripts.report_video_sequence_objective as vrp  # noqa: E402
import scripts.segment_objective_tables as st  # noqa: E402
import scripts.train_video_within_view as tw  # noqa: E402
import scripts.trained_token_readout as ttr  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.ev9v_complementarity import decode  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402
from scripts.report_fold_output_correction import write_atomic  # noqa: E402
from scripts.train_joint_visual_temporal import FramePaths  # noqa: E402
from scripts.trained_token_readout_tables import check_choice_record  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from echo_routing.config import gpu_memory_fraction  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402

NS = tw.NS
HALVES = st.HALVES
OTHER = {"selection": "calibration", "calibration": "selection"}
ARMS = tw.ARMS
NAMES = tw.NAMES
DIAG = rso.DIAG
POS, NEG, NONE = st.POS, st.NEG, st.NONE
PRIMARY = "M - A"
COMPARISONS = ("M - A", "M - F", "A - F", "M - frozen Ordered", "A - frozen Ordered")
DECODINGS = ("b3", "argmax", "ordered_setting")
SUFFIX = {"b3": "+ B3", "argmax": "argmax", "ordered_setting": "at Ordered setting"}
READING_ORDER = ("full_success", "mask_specific_signal", "smoothing_only_signal", "release_of_harm_only", "trade_off", "consistent_negatives",
                 "no_qualifying_gain", "end")
GAIN_READINGS = ("full_success", "mask_specific_signal", "smoothing_only_signal")
LOWER_IS_BETTER = ("multi FP segments", "multi FN segments", "native cuts per clip")
PARITY_WINDOWS = 64
DevAccess = rso.DevAccess
forbid_eager_loaders = vrp.forbid_eager_loaders
half_refs = vrp.half_refs
HELPER_FILES = ("scripts/freeze_model.py",)
F_SYSTEMS = ("F + B3", "F argmax", "F at Ordered setting")


def system_name(n: str, dec: str) -> str:
    """The system of one group (A, M, F or frozen Ordered) under a decoding; frozen Ordered's own setting is its B3."""
    if n == "frozen Ordered":
        return "frozen Ordered argmax" if dec == "argmax" else "frozen Ordered + B3"
    return f"{n} {SUFFIX[dec]}"


CMP = {d: {c: (system_name(c.split(" - ")[0], d), system_name(c.split(" - ")[1], d)) for c in COMPARISONS} for d in DECODINGS}
PAIRS = dict(CMP["b3"])                                  # segment accounting on the B3 systems


def _repo_modules() -> list[str]:
    out = set()
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if not f or not f.endswith(".py"):
            continue
        q = Path(f).resolve()
        if q.is_relative_to(ROOT) and q.relative_to(ROOT).parts[0] in ("scripts", "echo_routing"):
            out.add(q.relative_to(ROOT).as_posix())
    return sorted(out)


IMPORTED = _repo_modules()


def code_digest() -> str:
    h = hashlib.sha256(Path(__file__).read_bytes())
    for f in sorted(set(IMPORTED) | set(HELPER_FILES)):
        q = ROOT / f
        h.update(f.encode() + b"\0" + (q.read_bytes() if q.is_file() else b"missing"))
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------- the readings
def qualifies(v: dict) -> bool:
    """Consistent positive in at least one half and not consistent negative in the other."""
    return any(v[h] == POS for h in HALVES) and not any(v[h] == NEG for h in HALVES)


def unfavourable(metric: str, verdict: str) -> bool:
    """A consistent verdict against the candidate: lower F1, recall or native macro-F1, or more FP or FN segments or cuts."""
    return verdict == (POS if metric in LOWER_IS_BETTER else NEG)


def decide(ver: dict, goals: dict, trade: list[dict], negatives: list[dict]) -> dict:
    """The configuration's fixed, non-exclusive readings with the halves each holds in. ver[comparison][half]: the B3
    multi segment F1@50 verdicts over the three seeds; goals[group]: rt.goal of A, M, F and frozen Ordered; trade: the
    trade-off cells; negatives: every consistent verdict against a candidate (and every literal negative)."""
    ma, mf, af, mo = ver["M - A"], ver["M - F"], ver["A - F"], ver["M - frozen Ordered"]
    fail = {g: bool(goals[g]["constraint_failures"]) for g in ("A", "M")}; applied, notes, by_half = [], {}, {h: [] for h in HALVES}
    if all(v[h] == POS for v in (ma, mf, mo) for h in HALVES) and not fail["M"] and bool(goals["M"]["met"]):
        applied.append("full_success"); by_half["selection"].append("full_success"); by_half["calibration"].append("full_success")
    halves = [h for h in HALVES if ma[h] == POS and mf[h] == POS and ma[OTHER[h]] != NEG and mf[OTHER[h]] != NEG]
    if halves and not fail["M"]:
        applied.append("mask_specific_signal")
        notes["mask_specific_signal"] = {"halves": halves, "limitations": {"m_minus_frozen_ordered": dict(mo), "mean_targets_met": bool(goals["M"]["met"])}}
        for h in halves:
            by_half[h].append("mask_specific_signal")
    halves = [h for h in HALVES if af[h] == POS and mf[h] == POS and ma[h] != POS and af[OTHER[h]] != NEG and mf[OTHER[h]] != NEG]
    if halves and not fail["A"] and not fail["M"]:
        applied.append("smoothing_only_signal"); notes["smoothing_only_signal"] = {"halves": halves}
        for h in halves:
            by_half[h].append("smoothing_only_signal")
    if qualifies(ma) and not qualifies(mf):
        halves = [h for h in HALVES if ma[h] == POS]; applied.append("release_of_harm_only"); notes["release_of_harm_only"] = {"halves": halves}
        for h in halves:
            by_half[h].append("release_of_harm_only")
    if trade:
        applied.append("trade_off"); notes["trade_off"] = {"cells": trade}
        for h in HALVES:
            if any(c["half"] == h for c in trade):
                by_half[h].append("trade_off")
    if negatives:
        applied.append("consistent_negatives"); notes["consistent_negatives"] = negatives
        for h in HALVES:
            if any(c["half"] == h for c in negatives):
                by_half[h].append("consistent_negatives")
    if not set(GAIN_READINGS) & set(applied):
        applied.append("no_qualifying_gain"); notes["no_qualifying_gain"] = {"primary": dict(ma)}
    applied.append("end")
    order = lambda xs: [r for r in READING_ORDER if r in xs]  # noqa: E731
    return {"applied": order(applied), "notes": notes, "by_half": {h: order(v) for h, v in by_half.items()},
            "inputs": {"verdicts": ver, "floor_failures": fail, "m_goal_met": bool(goals["M"]["met"])}}


# ----------------------------------------------------------------------------------------------- choices, context
def load_choices(ctx, choices: dict) -> dict:
    keys = {tw.run_key(a, s) for a in ARMS for s in ctx.seeds}
    if set(choices["systems"]) != keys or set(choices["choices"]) != keys:
        raise SystemExit("the choice record does not hold exactly one record per new model")
    for k, rec in choices["systems"].items():
        check_choice_record(rec, ctx.www["evaluation"]["b3"]["grid"]); rso.rederive_choice(rec, ctx.www["goal"])
        if choices["choices"][k] != {x: rec[x] for x in ("status", "params", "n_feasible")}:
            raise SystemExit("a summary choice differs from its full record")
    a07 = ctx.a07_choices()["choices"]
    if choices.get("f_control") != {f"frame_s{s}": a07[f"frame_s{s}"] for s in ctx.seeds}:
        raise SystemExit("F's choices differ from A07's committed choices")
    return choices


class Ctx(tw.Ctx):
    def __init__(self, cfg: dict, config: str) -> None:
        super().__init__(cfg, config); self._cache = {}; self.gpu_seconds = 0.0; o = self.www["outputs"]; self.a07src = self.a07vso["source"]
        self.relpath = o["choices"]; self.twin = self.out / "calibration" / "choices.json"; self.proof = ROOT / o["choices_proof"]

    def _pinned(self, key: str, pin: dict, kind: str = "json"):
        if key not in self._cache:
            text = tw.pinned(ROOT, pin).read_text(); self._cache[key] = json.loads(text) if kind == "json" else yaml.safe_load(text)
        return self._cache[key]

    def a03src(self) -> dict:
        return self._pinned("a03", self.a07src["a03_config"], "yaml")["trained_token_readout"]["source"]

    def proto(self) -> dict:
        p = self._pinned("protocol", self.a03src()["protocol"]); ttr.check_class_order(rcs.CLASSES, p["mapping_version"])
        if self.a03src()["protocol"]["sha256"] != self.a07src["recipes"]["protocol"]["sha256"]:
            raise SystemExit("A03's protocol pin differs from A07's")
        return p

    def plans(self) -> dict:
        return self._pinned("plans", self.a03src()["window_plans"])

    def half(self, h: str, access: DevAccess) -> tuple:
        access.require(h)
        return ttr.half_sets(self.proto(), h, tw.pinned(ROOT, self.a03src()["clip_index"]), self.plans())

    def v01_summary(self) -> dict:
        return self._pinned("v01", self.a07src["raw_video_summary"])

    def v01_output(self, h: str, seed: int) -> dict:
        with np.load(tw.pinned(ROOT, self.a07src["ordered_outputs"][h][seed])) as z:
            return {k: z[k] for k in ("sequence_ids", "seq", "centre", "marker", "logits")}

    def a07_summary(self) -> dict:
        return self._pinned("a07_summary", self.src["a07_summary"])

    def a07_choices(self) -> dict:
        return self._pinned("a07_choices", self.src["a07_choices"])

    def f_output_path(self, h: str, seed: int) -> Path:
        return tw.pinned(ROOT, self.src["f_control"][seed]["outputs"][h])


def check_grid(ctx: Ctx) -> None:
    grid, want = ctx.cfg["decode_grids"]["b3"], ctx.www["evaluation"]["b3"]["grid"]
    if ([int(x) for x in grid["half_window"]], [float(x) for x in grid["threshold_quantile"]], [int(x) for x in grid["nms_radius"]]) != \
            ([int(x) for x in want["half_window"]], [float(x) for x in want["quantile"]], [int(want["nms_radius"])]):
        raise SystemExit("the inherited decoding grid differs from the configured B3 grid")


def check_fixed(ctx: Ctx, seed: int, params: dict) -> None:
    fx = ctx.www["evaluation"]["fixed_ordered_setting"]
    if (int(params["half_window"]), int(params["nms_radius"])) != (int(fx["half_window"]), int(fx["nms_radius"])) or \
            abs(float(params["threshold"]) - float(fx["thresholds"][seed])) > 1e-12:
        raise SystemExit(f"seed {seed}: V01's stored B3 setting differs from the configured fixed setting")


def verify_runs(ctx: Ctx) -> dict:
    idents = tw.identities(ctx); out = {}
    for key, ident in idents.items():
        man_path = ctx.out / "train" / key / "run_manifest.json"
        if not tw.finished(man_path, ident):
            raise SystemExit(f"{key}: not a finished run of this identity")
        man = json.loads(man_path.read_text()); frozen = ROOT / "models_frozen" / NS / key / "final.pt"
        if not frozen.is_file() or sha256_file(frozen) != man["final_sha256"]:
            raise SystemExit(f"{key}: the frozen final weights are missing or differ")
        out[key] = {"identity": ident, "updates": man["updates"], "final_sha256": man["final_sha256"], "interrupted": man["interrupted"],
                    "parameters": man["parameters"], "frozen": str(frozen.parent.relative_to(ROOT)), "epochs": man["epochs"]}
    return out


def provenance(ctx: Ctx, runs: dict) -> dict:
    a3 = ctx.a03src(); f = ctx.src["f_control"]
    return {"report_code": code_digest(), "train_code": tw.code_digest(), "config_sha256": sha256_file(ROOT / ctx.config),
            "runs": {k: {"identity": v["identity"], "final_sha256": v["final_sha256"]} for k, v in runs.items()},
            "f_control": {str(s): {"identity": f[s]["identity"], "weights": f[s]["weights"]["sha256"], "outputs": {h: f[s]["outputs"][h]["sha256"] for h in HALVES}}
                          for s in ctx.seeds},
            "a07": {k: ctx.src[k]["sha256"] for k in ("a07_config", "a07_summary", "a07_choices")},
            "metadata": {k: a3[k]["sha256"] for k in ("protocol", "clip_index", "window_plans")},
            "v01_outputs": {f"{h}_s{s}": ctx.a07src["ordered_outputs"][h][s]["sha256"] for h in HALVES for s in ctx.seeds},
            "v01_summary": ctx.a07src["raw_video_summary"]["sha256"]}


def ids_of(ctx: Ctx, seqs: dict) -> set:
    proto = ctx.proto(); ids = {c for h in HALVES for c in proto["validation"][h]["clips"]}
    ids |= {r[0] for q in (x for h in seqs for x in seqs[h]) for r in q["refs"]} | {q["id"] for h in seqs for q in seqs[h]}
    return ids | {session_of(v) for v in list(ids)}


# ----------------------------------------------------------------------------------------------- models and F's stored outputs
def load_frozen(path: Path, sha: str, device):
    if sha256_file(path) != sha:
        raise SystemExit(f"{path}: the frozen weights differ from their record")
    m = tw.make_model(torch.load(path, map_location="cpu", weights_only=True), device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def f_probabilities(path: Path, ident: str, W: dict, orders: np.ndarray, seqs: list[dict]) -> dict:
    """F's per-sample probabilities from its saved logits; the identity, sequences, windows and slot orders must match
    and the recomputed probabilities must equal the stored ones exactly."""
    lg, _ = rr.cached_logits(Path(path), ident, W, orders, seqs)
    if lg is None:
        raise SystemExit("F's saved outputs do not belong to its identity or to this half's windows")
    probs = rr.per_sample(W, lg, seqs)
    with np.load(path) as z:
        off, sp = z["sample_offsets"], z["sample_probs"]
    for i, q in enumerate(seqs):
        if not np.array_equal(np.asarray(probs[q["id"]], np.float32), sp[off[i]:off[i + 1]]):
            raise SystemExit("F's recomputed probabilities differ from its stored ones")
    return probs


def f_logits(path: Path) -> np.ndarray:
    with np.load(path) as z:
        return z["logits"].astype(np.float32)


# ----------------------------------------------------------------------------------------------- one half
def read_half(ctx: Ctx, h: str, access: DevAccess, runs: dict, choices: dict | None, device, workers: int) -> dict:
    """One half: its metadata and frames only; V01's saved outputs checked and its stored rows reproduced; F's saved
    outputs and rows reproduced; on the calibration half the V01 and F parity; every new model's per-sample probabilities
    (saved outputs of the same identity reused); with choices also the A and M systems."""
    ev, seqs = ctx.half(h, access); ids = [q["id"] for q in seqs]; refs = half_refs(seqs); at = {r: i for i, r in enumerate(refs)}
    fp = FramePaths(prv.frame_dirs(ctx.paths, ctx.rv, {r[0] for r in refs})); t0 = time.time(); frames = rr.decode_frames(refs, fp, workers)
    W = rr.half_windows(seqs, at, 5); orders = np.broadcast_to(np.arange(16, dtype=np.int64), W["s"].shape).copy()
    v01, a07, a07ch = ctx.v01_summary()["systems"], ctx.a07_summary()["systems"], ctx.a07_choices()["choices"]
    out = {"ev": ev, "seqs": seqs, "systems": {}, "probs": {}, "parity": {}, "reproduced": [], "reused": [],
           "counts": {"sequences": len(seqs), "windows": int(len(W["seq"])), "frames": len(refs), "decode_seconds": time.time() - t0}}
    tol = float(ctx.www["checks"]["source_parity_max_abs"]); k = PARITY_WINDOWS
    for s in ctx.seeds:
        saved = ctx.v01_output(h, s); ttr.check_windows(W, ids, saved, f"ordered_s{s} {h}")
        oparams = v01[f"ordered_s{s}"]["b3_choice"]["params"]; check_fixed(ctx, s, oparams); fpath = ctx.f_output_path(h, s)
        if h == "calibration":
            state, _ = ctx.source_state(s); src = tw.make_model(state, device).eval()
            lg, _ = rr.model_logits(src, {"ref": W["ref"][:k], "marker": W["marker"][:k]}, orders[:k], frames, device); del src
            fm = load_frozen(tw.pinned(ROOT, ctx.src["f_control"][s]["weights"]), ctx.src["f_control"][s]["weights"]["sha256"], device)
            lf, _ = rr.model_logits(fm, {"ref": W["ref"][:k], "marker": W["marker"][:k]}, orders[:k], frames, device); del fm
            out["parity"][f"s{s}"] = {"v01": tw.parity(lg, saved["logits"][:k], tol), "f": tw.parity(lf, f_logits(fpath)[:k], tol)}
            if not (out["parity"][f"s{s}"]["v01"]["ok"] and out["parity"][f"s{s}"]["f"]["ok"]):
                raise SystemExit(f"seed {s}: the V01 or F parity failed ({out['parity'][f's{s}']})")
        P = rr.per_sample(W, saved["logits"].astype(np.float32), seqs); rows = ttr.cls_rows(ctx.cfg, P, ev, oparams)
        for row in ("argmax", "b3"):
            ttr.check_stored(rows[row]["row"], v01[f"ordered_s{s}"][row][h], f"ordered_s{s} {row} {h} (V01)")
        sysd = {"frozen Ordered + B3": rso.row_of(ctx.cfg, decode(oparams, P), ev), "frozen Ordered argmax": rso.row_of(ctx.cfg, {q: p.argmax(1) for q, p in P.items()}, ev)}
        fprobs = f_probabilities(fpath, ctx.src["f_control"][s]["identity"], W, orders, seqs); fparams = a07ch[f"frame_s{s}"]["params"]
        fsys = {"F + B3": rso.row_of(ctx.cfg, decode(fparams, fprobs), ev), "F argmax": rso.row_of(ctx.cfg, {q: p.argmax(1) for q, p in fprobs.items()}, ev),
                "F at Ordered setting": rso.row_of(ctx.cfg, decode(oparams, fprobs), ev)}
        for name, x in fsys.items():
            rso.check_stored(x["row"], a07[f"{name} s{s}"][h], f"{name} s{s} {h} (A07)")
        out["reproduced"] += [f"frozen Ordered s{s} argmax and B3 {h}", f"F s{s} B3, argmax and stored-setting {h}"]; sysd.update(fsys)
        for arm in ARMS:
            key = tw.run_key(arm, s); path = ctx.out / "evaluation" / h / f"{key}.npz"; lg, sec = rr.cached_logits(path, runs[key]["identity"], W, orders, seqs)
            if lg is None:
                m = load_frozen(ROOT / runs[key]["frozen"] / "final.pt", runs[key]["final_sha256"], device); lg, sec = rr.model_logits(m, W, orders, frames, device); del m
                probs = rr.per_sample(W, lg, seqs); rr.save_outputs(path, runs[key]["identity"], W, orders, lg, probs, seqs, sec)
            else:
                probs = rr.per_sample(W, lg, seqs); out["reused"].append(f"{key} {h}")
            out["probs"][(arm, s)] = probs; n = NAMES[arm]
            if choices is not None:
                sysd[f"{n} + B3"] = rso.row_of(ctx.cfg, decode(choices["choices"][key]["params"], probs), ev)
                sysd[f"{n} argmax"] = rso.row_of(ctx.cfg, {q: p.argmax(1) for q, p in probs.items()}, ev)
                sysd[f"{n} at Ordered setting"] = rso.row_of(ctx.cfg, decode(oparams, probs), ev)
        for name, x in sysd.items():
            out["systems"][f"{name} s{s}"] = x
        torch.cuda.empty_cache(); print(f"{h} seed {s}: V01 and F checked, six-model outputs ready", flush=True)
    del frames
    return out


def read_halves(ctx: Ctx, data: dict, access: DevAccess, runs: dict, choices: dict | None, workers: int) -> dict:
    """data: reported half -> the half whose files are read; a half read once serves every reported half naming it."""
    device = tw.pvft.setup_gpu(); t0 = time.time(); per = {}
    try:
        for h in ("calibration", "selection"):
            if h not in data:
                continue
            if data[h] != h and data[h] in per:
                per[h] = per[data[h]]; continue
            per[h] = read_half(ctx, data[h], access, runs, choices, device, workers)
    finally:
        torch.cuda.empty_cache(); ctx.gpu_seconds += time.time() - t0
    return per


# ----------------------------------------------------------------------------------------------- summary
def check_calibration_rows(ctx: Ctx, per: dict, choices: dict) -> None:
    for key, rec in choices["systems"].items():
        arm, s = tw.split_key(key); g = next(r for r in rec["grid"] if r["params"] == rec["params"])
        row = per["systems"][f"{NAMES[arm]} + B3 s{s}"]["row"]
        if any(row[f] != g[f] for f in rcs.FLAT):
            raise SystemExit(f"{key}: the calibration row differs from the committed choice record")
    for s in ctx.seeds:
        rec = ctx.a07_choices()["systems"][f"frame_s{s}"]; g = next(r for r in rec["grid"] if r["params"] == rec["params"])
        if any(per["systems"][f"F + B3 s{s}"]["row"][f] != g[f] for f in rcs.FLAT):
            raise SystemExit(f"F s{s}: the calibration row differs from A07's choice record")


def goals(ctx: Ctx, systems: dict, choices: dict) -> dict:
    out = {}; a07 = ctx.a07_choices()["choices"]
    for n in ("A", "M", "F", "frozen Ordered"):
        rows = [{h: systems[f"{system_name(n, 'b3')} s{s}"][h] for h in HALVES} for s in ctx.seeds]
        if n in ("A", "M"):
            arm = {v: k for k, v in NAMES.items()}[n]; feas = [choices["choices"][tw.run_key(arm, s)]["status"] == "feasible" for s in ctx.seeds]
        elif n == "F":
            feas = [a07[f"frame_s{s}"]["status"] == "feasible" for s in ctx.seeds]
        else:
            feas = [True] * len(ctx.seeds)
        out[n] = rt.goal(rows, feas, ctx.www["goal"])
    return out


def trade_cells(res: dict) -> list[dict]:
    """A comparison and half where the candidate recovers reference segments or has fewer multi FP segments, together with
    a floor failure of the candidate in that half, a consistent multi boundary recall or segment F1@50 loss, or more lost
    than recovered reference segments."""
    v, K = res["verdicts"], res["kind_counts"]; out = []
    for c in COMPARISONS:
        cand, comp = c.split(" - ")
        for h in HALVES:
            pooled = res["accounting_pooled"][f"{c}|{h}|multi"]; seeds = sorted({int(k.rsplit(" s", 1)[1]) for k in K if k.startswith(system_name(cand, "b3") + " s")})
            fp = {x: sum(K[f"{system_name(x, 'b3')} s{s}"][h]["multi"]["fp"] for s in seeds) for x in (cand, comp)}
            gains = (["recovers reference segments"] if pooled["recovered"] > 0 else []) + (["fewer multi FP segments"] if fp[cand] < fp[comp] else [])
            costs = []
            if any(f["half"] == h for f in res["goal"][cand]["constraint_failures"]):
                costs.append("a floor failure of the candidate")
            if v[f"b3|{c}|multi boundary recall"][h] == NEG:
                costs.append("consistently lower multi boundary recall")
            if v[f"b3|{c}|multi segment F1@50"][h] == NEG:
                costs.append("consistently lower multi segment F1@50")
            if pooled["lost"] > pooled["recovered"]:
                costs.append("more lost than recovered reference segments")
            if gains and costs:
                out.append({"comparison": c, "half": h, "gains": gains, "costs": costs, "recovered": pooled["recovered"], "lost": pooled["lost"],
                            "multi_fp": fp, "new_unmatched_pred": pooled["new_unmatched_pred"], "removed_unmatched_pred": pooled["removed_unmatched_pred"]})
    return out


def consistent_list(res: dict) -> list[dict]:
    """Every consistent verdict against a candidate and every literal consistent negative, by decoding, comparison, metric and half."""
    out = []
    for key, v in res["verdicts"].items():
        dec, c, metric = key.split("|")
        for h in HALVES:
            if v[h] in (POS, NEG) and (v[h] == NEG or unfavourable(metric, v[h])):
                out.append({"decoding": dec, "comparison": c, "metric": metric, "half": h, "verdict": v[h], "unfavourable": unfavourable(metric, v[h])})
    return out


def pooled(ctx: Ctx, acc: dict) -> dict:
    out = {}
    for c in PAIRS:
        for h in HALVES:
            for kind in st.KINDS:
                xs = [acc[f"s{s}|{c}|{h}"]["segments"][kind] for s in ctx.seeds]
                out[f"{c}|{h}|{kind}"] = {n: int(sum(x[n]["segments"] for x in xs)) for n in ("recovered", "lost", "new_unmatched_pred", "removed_unmatched_pred")} | \
                    {"recovered_short": int(sum(x["recovered"]["short"] for x in xs)), "lost_short": int(sum(x["lost"]["short"] for x in xs)),
                     "recovered_xyx": int(sum(x["recovered"]["xyx_a4c_a5c"] for x in xs)), "lost_xyx": int(sum(x["lost"]["xyx_a4c_a5c"] for x in xs)),
                     "recovered_sessions_per_seed": [int(x["recovered"]["sessions"]) for x in xs], "lost_sessions_per_seed": [int(x["lost"]["sessions"]) for x in xs],
                     "new_unmatched_sessions_per_seed": [int(x["new_unmatched_pred"]["sessions"]) for x in xs],
                     "removed_unmatched_sessions_per_seed": [int(x["removed_unmatched_pred"]["sessions"]) for x in xs]}
    return out


def training_penalties(runs: dict) -> dict:
    keep = ("epoch", "updates", "loss", "ce", "regulariser", "S_A", "S_M", "same_label_penalty", "different_label_penalty", "same_edges", "different_edges",
            "zero_edge_groups", "native_switches", "constructed_within_view_switches", "constructed_accuracy", "native_accuracy")
    return {k: [{x: e.get(x) for x in keep} for e in r["epochs"]] for k, r in runs.items()}


def summarise(ctx: Ctx, per: dict, choices: dict) -> tuple[dict, dict]:
    res = {"systems": {}, "kind_counts": {}, "transcripts": {}, "switches": {}, "join_cuts": {}, "comparisons": {}, "accounting": {}, "native_changes": {}}
    private = {}
    for h in HALVES:
        x = per[h]; seqs = x["seqs"]
        for name, sysd in x["systems"].items():
            res["systems"].setdefault(name, {})[h] = sysd["row"]
            res["kind_counts"].setdefault(name, {})[h] = st.kind_counts(seqs, sysd["preds"], sysd["tables"])
            res["transcripts"].setdefault(name, {})[h] = st.transcript_summary(seqs, sysd["preds"])
            res["switches"].setdefault(name, {})[h] = st.switch_summary(seqs, sysd["preds"])
            mf = sysd["tables"]["multi_frag"]; res["join_cuts"].setdefault(name, {})[h] = {"cuts": int(mf["false_split"].sum()), "same_view_joins": int(mf["same_join"].sum())}
        for s in ctx.seeds:
            draws = {}
            for dec, cmps in CMP.items():
                for cname, (a, b) in cmps.items():
                    for k in (a, b):
                        if k not in draws:
                            draws[k] = st.system_draws(x["systems"][f"{k} s{s}"]["tables"])
                    res["comparisons"][f"s{s}|{dec}|{cname}|{h}"] = st.paired(draws[a], draws[b])
            for pname, (a, b) in PAIRS.items():
                A, B = x["systems"][f"{a} s{s}"], x["systems"][f"{b} s{s}"]; seg, priv = st.segment_accounting(seqs, A["preds"], B["preds"])
                nc = rcs.native_changes(A["tables"]["native"], B["tables"]["native"])
                res["accounting"][f"s{s}|{pname}|{h}"] = {"segments": seg}
                res["native_changes"][f"s{s}|{pname}|{h}"] = {k: ({"fixed": len(v["fixed"]), "new": len(v["new"]), "fixed_sessions": v["fixed_sessions"],
                                                                  "new_sessions": v["new_sessions"]} if "fixed_sessions" in v else v) for k, v in nc.items()}
                private[f"s{s}|{pname}|{h}"] = {"segments": priv, "native": {k: v for k, v in nc.items() if "fixed_sessions" in v}}
            print(f"{h} seed {s}: comparisons and accounting done", flush=True)
    res["verdicts"] = {f"{dec}|{c}|{m}": {h: st.consistency([res["comparisons"][f"s{s}|{dec}|{c}|{h}"][m] for s in ctx.seeds]) for h in HALVES}
                       for dec, cmps in CMP.items() for c in cmps for m in st.BOOT_METRICS}
    res["accounting_pooled"] = pooled(ctx, res["accounting"]); res["goal"] = goals(ctx, res["systems"], choices)
    ver = {c: res["verdicts"][f"b3|{c}|multi segment F1@50"] for c in COMPARISONS}; res["trade_cells"] = trade_cells(res); res["consistent"] = consistent_list(res)
    res["decision"] = decide(ver, res["goal"], res["trade_cells"], res["consistent"])
    res["decision"]["primary_by_seed"] = {h: {f"s{s}": res["comparisons"][f"s{s}|b3|{PRIMARY}|{h}"]["multi segment F1@50"] for s in ctx.seeds} for h in HALVES}
    return res, private


def tables_md(res: dict, config: str) -> str:
    md = st.md_table
    parts = ["# A08 within-view regularisation round: tables\n",
             f"Recompute: `python scripts/report_video_within_view.py --config {config} --selection --choices-sha256 <sha> --choices-commit <commit>`\n"]
    rows = [{"system": k, "half": h, "segment F1@50": r.get("segment_f1_50_multi"), "boundary P": r.get("boundary_precision_multi"), "boundary R": r.get("boundary_recall_multi"),
             "boundary F1": r.get("boundary_f1_multi"), "native macro-F1": r.get("native_macro_f1"), "switches per clip": r.get("native_internal_cuts_per_clip"),
             "same-view join cuts": r.get("same_class_cut_rate_multi"), "pairs F1@50": r.get("segment_f1_50_pairs"), "A4C read as A5C": r.get("A4C read as A5C"),
             "A5C read as A4C": r.get("A5C read as A4C")} for k, by in res.get("systems", {}).items() for h, r in by.items()]
    parts += ["## Systems\n", md(rows)]
    parts += ["## Paired comparisons (seed / decoding / comparison / half; 95% bootstrap intervals)\n",
              md([{"comparison": k.replace("|", " / "), **{m: st.ci(v[m]) for m in st.BOOT_METRICS}} for k, v in res.get("comparisons", {}).items()])]
    parts += ["## Three-seed verdicts\n", md([{"decoding / comparison / metric": k.replace("|", " / "), **v} for k, v in res.get("verdicts", {}).items()])]
    parts += ["## Pooled recovered and lost reference segments (B3 systems)\n", md([{"pair / half / population": k.replace("|", " / "), **{x: y for x, y in v.items() if not isinstance(y, list)}}
                                                                                   for k, v in res.get("accounting_pooled", {}).items()])]
    parts += ["## Goal\n", md([{"group": g, "selection mean": v["mean_segment_f1"]["selection"], "calibration mean": v["mean_segment_f1"]["calibration"],
                                 "constraint failures": len(v["constraint_failures"]), "met": v["met"]} for g, v in res.get("goal", {}).items()])]
    pen = [{"run": k, "epoch": e["epoch"], "CE": e["ce"], "S_A": e["S_A"], "S_M": e["S_M"], "equal-label penalty": e["same_label_penalty"],
            "different-label penalty": e["different_label_penalty"]} for k, es in res.get("training", {}).items() for e in es if e["epoch"] in (0, 9)]
    parts += ["## Training penalties (first and last epoch means; training data only)\n", md(pen)]
    if "decision" in res:
        parts += ["## Readings\n", "```\n" + json.dumps({"applied": res["decision"]["applied"], "by_half": res["decision"]["by_half"]}, indent=1) + "\n```\n"]
    return "\n".join(parts)


def figures(ctx: Ctx, runs: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.2)); col = {"all": "#dd8452", "masked": "#4c72b0"}
    for key, r in runs.items():
        arm, s = tw.split_key(key); ep = r["epochs"]; x = [e["updates"] for e in ep]; kw = {"color": col[arm], "ls": ("-", "--", ":")[s], "lw": 1.1, "label": f"{NAMES[arm]} s{s}"}
        ax[0].plot(x, [e["ce"] for e in ep], **kw); ax[1].plot(x, [e["same_label_penalty"] for e in ep], **kw)
        ax[2].plot(x, [e["different_label_penalty"] for e in ep], **kw); ax[3].plot(x, [e["constructed_accuracy"] for e in ep], **kw)
    for a, t in zip(ax, ("plain CE L_CE (epoch mean)", "equal-label edge penalty (epoch mean, logged in both arms)",
                         "different-label edge penalty (epoch mean, logged in both arms)", "target accuracy of the constructed groups")):
        a.set_title(t, fontsize=9); a.set_xlabel("updates"); a.grid(alpha=0.3); a.axvline(375, color="grey", lw=0.8, ls="--")
    ax[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Training data only; the final weights are kept and nothing is chosen from these curves (grey line: encoder unfrozen; line style: seed)", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "training_curves.png", dpi=130); plt.close(fig)


def save_private(out: Path, per: dict, private: dict) -> None:
    """Per-sequence B3 decodings of the six new models and the case lists; server only."""
    for h in HALVES:
        x = per[h]; ids = [q["id"] for q in x["seqs"]]; d = out / h; d.mkdir(parents=True, exist_ok=True)
        for (arm, s), probs in x["probs"].items():
            dec = np.concatenate([np.asarray(x["systems"][f"{NAMES[arm]} + B3 s{s}"]["preds"][i], np.int8) for i in ids])
            np.savez(d / f"{arm}_s{s}_b3.npz", sequence_ids=np.asarray(ids), sample_offsets=np.cumsum([0, *[len(probs[i]) for i in ids]]), b3=dec)
    write_atomic(out / "private_accounting.json", json.dumps(private, default=str))


def evaluate_all(ctx: Ctx, choices: dict, data: dict, access: DevAccess, runs: dict, workers: int) -> tuple[dict, dict, dict, set]:
    per = read_halves(ctx, data, access, runs, choices, workers)
    check_calibration_rows(ctx, per["calibration"], choices)
    res, private = summarise(ctx, per, choices)
    res["parity"] = {h: per[h]["parity"] for h in HALVES}; res["reproduced"] = {h: per[h]["reproduced"] for h in HALVES}
    res["reused_outputs"] = {h: per[h]["reused"] for h in HALVES}; res["half_counts"] = {h: per[h]["counts"] for h in HALVES}
    res["choices"] = choices["choices"]; res["f_choices"] = choices["f_control"]; res["training"] = training_penalties(runs)
    res["runs"] = {k: {x: v for x, v in r.items() if x != "epochs"} for k, r in runs.items()}; res["training_last_epoch"] = {k: r["epochs"][-1] for k, r in runs.items()}
    ids = ids_of(ctx, {h: per[h]["seqs"] for h in HALVES}); flat = json.loads(json.dumps(res, default=str)); assert_no_ids(flat, ids, "summary")
    if any(x in tables_md(flat, ctx.config) for x in ids):
        raise SystemExit("an id appears in the report tables")
    return flat, per, private, ids


def open_selection(ctx: Ctx, out: Path) -> list:
    """Recorded before the selection half is read; a finished evaluation is never repeated; a repeat after a failure needs
    --repeat-reason and every opening is listed in the summary."""
    if (out / "done.json").is_file():
        raise SystemExit("the selection half was already evaluated once; it is never evaluated again")
    p = out / "opened.json"; rec = json.loads(p.read_text()) if p.is_file() else {"openings": []}; reason = getattr(ctx, "repeat_reason", None)
    if rec["openings"] and not reason:
        raise SystemExit("the selection half was already opened; a repeat needs --repeat-reason and is disclosed in the summary")
    rec["openings"].append({"at": tw.now(), "choices_sha256": ctx.choices_sha256, "choices_commit": ctx.choices_commit, "reason": reason or "first opening"})
    out.mkdir(parents=True, exist_ok=True); write_atomic(p, json.dumps(rec, indent=1))
    tw.ledger_append(ctx.ledger, "selection opened", "cpu", tw.now(), 0.0, True, f"opening {len(rec['openings'])}: {rec['openings'][-1]['reason']}")
    return rec["openings"]


# ----------------------------------------------------------------------------------------------- stages
def run_stage(ctx: Ctx, name: str, fn, workers: int) -> None:
    """Every stage, finished or failed, appends its inference part as GPU hours and its remainder as CPU hours."""
    b = ctx.www["budget"]
    if tw.ledger_hours(ctx.ledger, "cpu") >= float(b["cpu_hours_max"]) or tw.ledger_hours(ctx.ledger, "gpu") >= float(b["gpu_hours_max"]):
        raise SystemExit("the CPU or GPU budget is spent")
    t0, started, ok, note = time.time(), tw.now(), False, ""; ctx.gpu_seconds = 0.0
    try:
        env = tw.vpr.environment()
        if any(str(env[k]) != str(v) for k, v in ctx.src["environment"].items()):
            raise SystemExit(f"environment {env} differs from the pinned {ctx.src['environment']}")
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction()); fn(ctx, workers); ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        total = time.time() - t0; g = min(float(ctx.gpu_seconds), total)
        if g > 0:
            tw.ledger_append(ctx.ledger, f"{name} (GPU inference)", "gpu", started, g, ok, note)
        tw.ledger_append(ctx.ledger, f"{name} (CPU)", "cpu", started, total - g, ok, note)


def stage_calibrate(ctx: Ctx, workers: int) -> None:
    undo = forbid_eager_loaders()
    try:
        check_grid(ctx); runs = verify_runs(ctx); prov = provenance(ctx, runs)
        per = read_halves(ctx, {"calibration": "calibration"}, DevAccess({"calibration"}), runs, None, workers)["calibration"]
        rec = {"provenance": prov, "half": "calibration", "parity": per["parity"], "reproduced": per["reproduced"], "counts": per["counts"], "systems": {}}
        for s in ctx.seeds:
            for arm in ARMS:
                ch = rcs.b3_choose(ctx.cfg, DIAG, per["probs"][(arm, s)], per["ev"]); key = tw.run_key(arm, s)
                rec["systems"][key] = {"status": ch["status"], "params": ch["params"], "n_feasible": ch["n_feasible"], "grid": ch["grid"]}
                print(f"{key}: {ch['status']} {ch['params']}", flush=True)
        rec["choices"] = {k: {x: v[x] for x in ("status", "params", "n_feasible")} for k, v in rec["systems"].items()}
        a07 = ctx.a07_choices()["choices"]; rec["f_control"] = {f"frame_s{s}": a07[f"frame_s{s}"] for s in ctx.seeds}
        flat = json.loads(json.dumps(rec, default=str)); assert_no_ids(flat, ids_of(ctx, {"calibration": per["seqs"]}), "choices")
        for p in (ctx.twin, ROOT / ctx.relpath):
            if p.is_file() and json.loads(p.read_text()) != flat:
                raise SystemExit("an earlier choice record differs from this one; it is never overwritten")
        for p in (ctx.twin, ROOT / ctx.relpath):
            p.parent.mkdir(parents=True, exist_ok=True); write_atomic(p, json.dumps(flat, indent=1))
    finally:
        undo()


def stage_dry_run(ctx: Ctx, workers: int) -> None:
    undo = forbid_eager_loaders()
    try:
        check_grid(ctx); runs = verify_runs(ctx); prov = provenance(ctx, runs)
        if not ctx.twin.is_file():
            raise SystemExit("run --calibrate first")
        choices = load_choices(ctx, json.loads(ctx.twin.read_text()))
        if choices["provenance"] != json.loads(json.dumps(prov, default=str)):
            raise SystemExit("the choices belong to other code, configuration, runs or sources")
        flat, per, private, _ = evaluate_all(ctx, choices, {"calibration": "calibration", "selection": "calibration"}, DevAccess({"calibration"}), runs, workers)
        out = ctx.out / "dry_run"; out.mkdir(parents=True, exist_ok=True); flat["dry_run"] = "the calibration half stands in for both halves; a code check, not a result"
        figures(ctx, runs, out); save_private(out, per, private)
        write_atomic(out / "summary.json", json.dumps(flat, indent=1)); write_atomic(out / "report_tables.md", tables_md(flat, ctx.config))
        print("dry run ok (a code check, not a result)", flush=True)
    finally:
        undo()


def stage_selection(ctx: Ctx, workers: int) -> None:
    undo = forbid_eager_loaders()
    try:
        check_grid(ctx); runs = verify_runs(ctx); prov = provenance(ctx, runs)
        choices = load_choices(ctx, rso.barrier(ROOT / ctx.relpath, ctx.twin, ctx.proof, ctx.relpath, ctx.choices_sha256, ctx.choices_commit, prov))
        out = ctx.out / "selection"; done = out / "done.json"; openings = open_selection(ctx, out)
        flat, per, private, ids = evaluate_all(ctx, choices, {"calibration": "calibration", "selection": "selection"}, DevAccess({"calibration", "selection"}), runs, workers)
        led = [json.loads(x) for x in ctx.ledger.read_text().splitlines() if x.strip()]
        flat["identity"] = {**prov, "choices_sha256": ctx.choices_sha256, "choices_commit": ctx.choices_commit,
                            "proof_made": json.loads(ctx.proof.read_text()).get("made"), "selection_openings": openings}
        flat["cost"] = {"gpu_hours_before_this_stage": tw.ledger_hours(ctx.ledger, "gpu"), "cpu_hours_before_this_stage": tw.ledger_hours(ctx.ledger, "cpu"),
                        "stages": [{k: x[k] for k in ("job", "kind", "started", "finished", "hours", "ok")} for x in led]}
        assert_no_ids(flat, ids, "results summary"); figures(ctx, runs, ctx.results); save_private(out, per, private)
        write_atomic(ctx.results / "summary.json", json.dumps(flat, indent=1)); write_atomic(ctx.results / "report_tables.md", tables_md(flat, ctx.config))
        write_atomic(done, json.dumps({"choices_sha256": ctx.choices_sha256, "choices_commit": ctx.choices_commit, "finished": tw.now()}))
        print(json.dumps(flat["decision"]["applied"]), flush=True)
    finally:
        undo()


def stage_proof(ctx: Ctx) -> None:
    if not ctx.choices_commit:
        raise SystemExit("--proof needs --choices-commit")
    proof = rso.git_proof(ROOT, ctx.choices_commit, ctx.relpath); rso.verify_proof(proof, (ROOT / ctx.relpath).read_bytes(), ctx.relpath, proof["commit"])
    ctx.proof.parent.mkdir(parents=True, exist_ok=True); write_atomic(ctx.proof, json.dumps(proof, indent=1))
    print(json.dumps({"commit": proof["commit"], "blob": proof["blob"], "file_sha256": proof["file_sha256"], "objects": len(proof["objects"])}), flush=True)


STAGES = {"calibrate": stage_calibrate, "dry_run": stage_dry_run, "selection": stage_selection}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--choices-sha256"); ap.add_argument("--choices-commit", help="the full 40-character commit")
    ap.add_argument("--repeat-reason", help="--selection only: why a failed selection evaluation is repeated (disclosed in the summary)")
    ap.add_argument("--workers", type=int, default=6); g = ap.add_mutually_exclusive_group(required=True)
    for s in ("calibrate", "dry_run", "proof", "selection"):
        g.add_argument(f"--{s.replace('_', '-')}", dest=s, action="store_true")
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config)
    ctx.choices_sha256, ctx.choices_commit, ctx.repeat_reason = a.choices_sha256, a.choices_commit, a.repeat_reason
    if a.proof:
        stage_proof(ctx); return
    stage = next(s for s in STAGES if getattr(a, s))
    tw.determinism(); run_stage(ctx, stage, STAGES[stage], a.workers)


if __name__ == "__main__":
    main()
