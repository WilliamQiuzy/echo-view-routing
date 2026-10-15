#!/usr/bin/env python
"""Evaluate the FACT video feature round once (docs/superpowers/plans/2026-10-15-fact-video-feature-training.md, Task 3;
configs/experiments/fact_video_feature_v1.yaml). Runs only after the three FACT heads are finished and frozen.

Development data come from A03's metadata-only helpers (one half's protocol entries, clip index rows and window plans)
and from A05's read-only adapter of A01's combined development caches, which copies only the open half's rows by
bounded seeks. No video encoder is built and no eager development loader is called (a guard makes them raise). Every
head sees features alone; FACT, MS-TCN and ASRF get all-zero placeholder labels of the valid length.

  --calibration  the calibration half only: parity of the source classifier against V01's saved logits, A01's stored
                 ASRF, Linear + B3, MS-TCN and frozen Ordered + B3 rows reproduced exactly, FACT's official outputs, and
                 the evaluation manifest (every model and source sha256, the fixed decoding, control settings, readings)
  --dry-run      a code check of the selection pipeline before the manifest is committed; calibration stands in for
                 both halves and nothing reaches the results
  --proof        (local) the git-object proof that a commit holds exactly the manifest; copied to the server
  --selection    with --manifest-sha256 and --manifest-commit: the barrier, then both halves once: systems, paired
                 comparisons, accounting, mechanism and tokens, goal and the fixed readings

  python scripts/report_fact_video_feature.py --config configs/experiments/fact_video_feature_v1.yaml --calibration
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
import scripts.prepare_video_feature_temporal as pv  # noqa: E402
import scripts.raw_video_context_tables as rt  # noqa: E402
import scripts.report_center_supervision as rcs  # noqa: E402
import scripts.report_fact_asrf as ra  # noqa: E402
import scripts.report_segment_objective as rs  # noqa: E402
import scripts.segment_objective_tables as st  # noqa: E402
import scripts.train_fact_video_feature as tf  # noqa: E402
import scripts.train_segment_objective as ts  # noqa: E402
import scripts.train_video_feature_temporal as tv  # noqa: E402
import scripts.trained_token_readout as ttr  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.ev9v_complementarity import decode  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402
from scripts.report_fold_output_correction import write_atomic  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

import echo_routing.temporal.fact_head as fh  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402

NS = tf.NS
HALVES = st.HALVES
POS, NEG, NONE = st.POS, st.NEG, st.NONE
DevAccess, open_half, member_layout, read_members = rs.DevAccess, rs.dev_half, rs.member_layout, rs.read_members
git_proof, verify_proof = rs.git_proof, rs.verify_proof
PRIMARY = "FACT - A01 ASRF"
COMPARE = {"FACT - A01 ASRF": ("FACT", "A01 ASRF"), "FACT - A01 Linear + B3": ("FACT", "A01 Linear + B3"),
           "FACT - frozen Ordered + B3": ("FACT", "frozen Ordered + B3"), "FACT - A01 MS-TCN": ("FACT", "A01 MS-TCN")}
CONTROLS = ("A01 ASRF", "A01 Linear + B3", "frozen Ordered + B3", "A01 MS-TCN")
POINT_CONTROLS = ("A01 ASRF", "A01 Linear + B3", "frozen Ordered + B3")
TRADE_CONTROLS = ("A01 Linear + B3", "frozen Ordered + B3")
METRICS = ("multi segment F1@50", "multi boundary F1", "multi boundary recall", "native macro-F1", "native cuts per clip", "same-class cut rate",
           "multi FP segments", "multi FN segments")
REFERENCE_ROWS = {"frozen Ordered + B3": "frozen Ordered + B3", "frozen Ordered argmax": "frozen Ordered argmax", "A01 Linear + B3": "Linear + B3",
                  "A01 Linear argmax": "Linear argmax", "A01 MS-TCN": "MS-TCN", "A01 ASRF": "ASRF"}
READING_ORDER = ("complete_system_progress", "learned_system_gain_without_strong_reference_win", "one_half_or_local_effect", "trade_off",
                 "uncertain_or_no_qualifying_gain", "end")


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
HELPER_FILES = ("scripts/freeze_model.py", "third_party/ms-tcn/model.py")


def code_digest() -> str:
    h = hashlib.sha256(Path(__file__).read_bytes())
    for f in sorted(set(IMPORTED) | set(HELPER_FILES)):
        q = ROOT / f
        h.update(f.encode() + b"\0" + (q.read_bytes() if q.is_file() else b"missing"))
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------- context
class Ctx(tf.Ctx):
    def __init__(self, cfg: dict, config: str) -> None:
        super().__init__(cfg, config); self._cache = {}; self.gpu_seconds = 0.0
        self.relpath = self.fv["outputs"]["manifest"]; self.twin = self.out / "evaluation" / "evaluation_manifest.json"
        self.proof = self.root / self.fv["outputs"]["manifest_proof"]

    def _pinned(self, key: str, pin: dict, kind: str = "json"):
        if key not in self._cache:
            text = pv.pinned(self.root, pin).read_text(); self._cache[key] = json.loads(text) if kind == "json" else yaml.safe_load(text)
        return self._cache[key]

    def a03src(self) -> dict:
        return self._pinned("a03", self.src["a03_config"], "yaml")["trained_token_readout"]["source"]

    def proto(self) -> dict:
        p = self._pinned("protocol", self.a03src()["protocol"]); ttr.check_class_order(rcs.CLASSES, p["mapping_version"])
        return p

    def plans(self) -> dict:
        return self._pinned("plans", self.a03src()["window_plans"])

    def half(self, h: str, access: DevAccess) -> tuple:
        """One half's metadata (labels of that half's clips only), after the access check."""
        access.require(h)
        return ttr.half_sets(self.proto(), h, pv.pinned(self.root, self.a03src()["clip_index"]), self.plans())

    def a01_summary(self) -> dict:
        return self._pinned("a01_summary", self.src["a01_summary"])

    def v01_summary(self) -> dict:
        return self._pinned("v01_summary", self.a01().vt["source"]["report"]["summary"])


# ----------------------------------------------------------------------------------------------- barrier and openings
def manifest_barrier(results: Path, twin: Path, proof_path: Path, relpath: str, expected_sha: str | None, commit: str | None, provenance: dict) -> dict:
    """The committed manifest: the results copy equals the calibration stage's copy, carries the given sha256, the
    git-object proof shows the given commit holds exactly these bytes, and the provenance is the current one."""
    if not expected_sha or not commit:
        raise SystemExit("--selection needs --manifest-sha256 and --manifest-commit of the committed evaluation manifest")
    for p in (results, twin, proof_path):
        if not Path(p).is_file():
            raise SystemExit(f"{Path(p).name} is missing")
    data = Path(results).read_bytes()
    if Path(twin).read_bytes() != data:
        raise SystemExit("the results copy of the manifest differs from the calibration stage's copy")
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise SystemExit("the manifest is not the committed record (its sha256 differs)")
    verify_proof(json.loads(Path(proof_path).read_text()), data, relpath, commit)
    man = json.loads(data)
    if man.get("provenance") != json.loads(json.dumps(provenance, default=str)):
        raise SystemExit("the manifest belongs to other code, configuration, runs or sources")
    return man


def open_selection(ctx, out: Path) -> list:
    """Recorded before the selection half is read. A finished evaluation is never repeated; after an interruption a
    repeat needs --repeat-reason, reuses saved outputs of the same identity and is listed in the summary."""
    out = Path(out)
    if (out / "done.json").is_file():
        raise SystemExit("the selection half was already evaluated once; it is never evaluated again")
    p = out / "opened.json"; rec = json.loads(p.read_text()) if p.is_file() else {"openings": []}; reason = getattr(ctx, "repeat_reason", None)
    if rec["openings"] and not reason:
        raise SystemExit("the selection half was already opened; a repeat needs --repeat-reason and is disclosed in the summary")
    rec["openings"].append({"at": ts.now(), "manifest_sha256": ctx.manifest_sha256, "manifest_commit": ctx.manifest_commit, "reason": reason or "first opening"})
    out.mkdir(parents=True, exist_ok=True); write_atomic(p, json.dumps(rec, indent=1))
    ts.ledger_append(ctx.ledger, "selection opened", "cpu", ts.now(), 0.0, True, f"opening {len(rec['openings'])}: {rec['openings'][-1]['reason']}")
    return rec["openings"]


# ----------------------------------------------------------------------------------------------- saved FACT outputs
def save_outputs(path: Path, ident: str, outputs: dict) -> None:
    ids = list(outputs); lab = [np.asarray(outputs[k]["labels"], np.int8) for k in ids]
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, identity=ident, ids=np.asarray(ids), offsets=np.cumsum([0, *[x.size for x in lab]]), labels=np.concatenate(lab),
             frame_labels=np.concatenate([np.asarray(outputs[k]["frame_labels"], np.int8) for k in ids]),
             tokens_used=np.asarray([outputs[k]["tokens_used"] for k in ids]), runs=np.asarray([outputs[k]["runs"] for k in ids]),
             tokens=np.asarray([outputs[k]["tokens"] for k in ids]), separate=np.asarray([bool(outputs[k]["separate_intervals"]) for k in ids]))
    tmp.replace(path)


def load_outputs(path: Path, ident: str) -> dict | None:
    """Saved outputs are reused only under the same identity; another identity is never overwritten silently."""
    path = Path(path)
    if not path.is_file():
        return None
    z = read_members(path, ("identity", "ids", "offsets", "labels", "frame_labels", "tokens_used", "runs", "tokens", "separate"))
    if str(z["identity"]) != ident:
        raise SystemExit(f"{path.name} holds outputs of another identity; it is not reused or overwritten")
    off = z["offsets"]
    return {str(k): {"labels": z["labels"][off[i]:off[i + 1]].astype(int), "frame_labels": z["frame_labels"][off[i]:off[i + 1]].astype(int),
                     "tokens_used": int(z["tokens_used"][i]), "runs": int(z["runs"][i]), "tokens": int(z["tokens"][i]), "separate_intervals": bool(z["separate"][i])}
            for i, k in enumerate(z["ids"].tolist())}


# ----------------------------------------------------------------------------------------------- references, goal, readings
def stored_rows(a01_summary: dict) -> dict:
    """E0 / E1 / B7 + B3 and F02's FACT ('Original FACT s{k}' in A01's summary) as stored rows; a missing one stops."""
    sysd = a01_summary["systems"]; names = {**{k: k for k in ("E0 + B3", "E1 + B3", "B7 + B3")}, **{f"F02 FACT s{k}": f"Original FACT s{k}" for k in (0, 1, 2)}}
    missing = [v for v in names.values() if v not in sysd]
    if missing:
        raise SystemExit(f"stored reference rows are missing from A01's summary: {missing}")
    return {k: sysd[v] for k, v in names.items()}


def goal_of(rows_by_seed: list[dict], g: dict) -> dict:
    """FACT's main output has no B3 setting, so no feasibility condition applies; the floors and means do."""
    return rt.goal(rows_by_seed, [True] * len(rows_by_seed), g)


CATEGORIES = (("all reference segments", "segments"), ("segments of at most 16 samples", "short"), ("A4C/A5C X-Y-X middles", "xyx_a4c_a5c"))


def point_estimate_items(acc: dict, seeds=(0, 1, 2)) -> list[dict]:
    """Recoveries pooled over seeds that exceed the losses against ASRF, Linear + B3 or frozen Ordered + B3, counted
    separately for native clips, pair streams and multi streams."""
    items = []
    for c in POINT_CONTROLS:
        for h in HALVES:
            for kind in st.KINDS:
                xs = [acc[f"s{s}|FACT vs {c}|{h}"]["segments"][kind] for s in seeds]
                for what, key in CATEGORIES:
                    rec, lost = (int(sum(x[side][key] for x in xs)) for side in ("recovered", "lost"))
                    if rec > lost:
                        items.append({"control": c, "half": h, "population": kind, "what": what, "recovered": rec, "lost": lost})
    return items


def denominators(seqs: list[dict]) -> dict:
    """Reference segments per population: all, at most 16 samples, and A4C/A5C X-Y-X middles (the raw denominators)."""
    from echo_routing.temporal.segments import labels_to_segments
    out = {k: {w: 0 for w, _ in CATEGORIES} for k in st.KINDS}
    for q in seqs:
        ref = labels_to_segments(np.asarray(q["labels"])); d = out[q["kind"]]
        d["all reference segments"] += len(ref); d["segments of at most 16 samples"] += sum(r.length <= 16 for r in ref)
        d["A4C/A5C X-Y-X middles"] += sum(st.xyx_middle(ref, j) for j in range(len(ref)))
    return out


def local_evidence(ver: dict, points: list[dict]) -> dict:
    return {"consistent": [{"comparison": c, "half": h} for c in COMPARE for h in HALVES if ver[c][h] == POS], "point_estimates": list(points)}


def trade_cells(acc: dict, rows: dict, failures: list[dict], seeds=(0, 1, 2)) -> list[dict]:
    """Seed-and-half cells where FACT recovers a reference segment against Linear + B3 or frozen Ordered + B3 while it
    also has a new unmatched predicted segment, a lower native macro-F1 or a floor failure in that cell."""
    out = []
    for s in seeds:
        for h in HALVES:
            fail = any(int(f["seed"]) == s and f["half"] == h for f in failures)
            for c in TRADE_CONTROLS:
                seg = acc[f"s{s}|FACT vs {c}|{h}"]["segments"]
                rec = int(sum(seg[k]["recovered"]["segments"] for k in seg)); new = int(sum(seg[k]["new_unmatched_pred"]["segments"] for k in seg))
                drop = rows[f"FACT s{s}"][h]["native_macro_f1"] < rows[f"{c} s{s}"][h]["native_macro_f1"]
                if rec > 0 and (new > 0 or drop or fail):
                    out.append({"seed": s, "half": h, "control": c, "recovered": rec, "new_unmatched_pred": new, "native_macro_f1_lower": bool(drop), "floor_failure": bool(fail)})
    return out


def readings(ver: dict, goal: dict, local: dict, trade: list[dict]) -> dict:
    both = lambda c: all(ver[c][h] == POS for h in HALVES)  # noqa: E731
    floors = not goal["constraint_failures"]; met = bool(goal["met"]); applied, notes = [], {}
    progress = both(PRIMARY) and both("FACT - A01 Linear + B3") and both("FACT - frozen Ordered + B3") and floors and met
    if progress:
        applied.append("complete_system_progress")
    if both(PRIMARY) and both("FACT - A01 Linear + B3") and not progress:
        failed = [n for n, ok in (("frozen Ordered + B3 comparison", both("FACT - frozen Ordered + B3")), ("recall or native floor", floors), ("goal", met)) if not ok]
        applied.append("learned_system_gain_without_strong_reference_win"); notes["learned_system_gain_without_strong_reference_win"] = {"failed": failed}
    if local["consistent"] or local["point_estimates"]:
        applied.append("one_half_or_local_effect"); notes["one_half_or_local_effect"] = local
    if trade:
        applied.append("trade_off"); notes["trade_off"] = {"cells": trade}
    if not progress:
        applied.append("uncertain_or_no_qualifying_gain"); notes["uncertain_or_no_qualifying_gain"] = {"primary": ver[PRIMARY]}
    applied.append("end")
    return {"applied": [r for r in READING_ORDER if r in applied], "notes": notes}


# ----------------------------------------------------------------------------------------------- runs and provenance
def verify_runs(ctx: Ctx) -> dict:
    prep = tf.read_record(ctx, "preparation.json"); head = tf.make_head(ctx, fh.TokenRule(**prep["token_rule"])); idents = tf.identities(ctx, head, prep); out = {}
    for key, ident in idents.items():
        man_path = ctx.out / "train" / key / "run_manifest.json"
        if tf.run_state(man_path, ident, ctx.total) != "done":
            raise SystemExit(f"{key}: not a finished run of this identity")
        man = json.loads(man_path.read_text()); frozen = ctx.root / "models_frozen" / NS / key / "final.pt"
        if not frozen.is_file() or sha256_file(frozen) != man["final_sha256"]:
            raise SystemExit(f"{key}: the frozen final weights are missing or differ")
        out[key] = {"identity": ident, "updates": man["completed_updates"], "final_sha256": man["final_sha256"], "interrupted": man["interrupted"],
                    "parameters": man["parameters"], "minutes": man["minutes"], "frozen": str(frozen.parent.relative_to(ctx.root)), "records": man["records"]}
    return out


def provenance(ctx: Ctx, runs: dict) -> dict:
    a3 = ctx.a03src(); a01 = ctx.a01(); prep = tf.read_record(ctx, "preparation.json")
    if a01.vt["source"]["protocol"] != a3["protocol"] or a01.vt["source"]["window_plans"] != a3["window_plans"]:
        raise SystemExit("A03's metadata pins differ from A01's protocol or window plans")
    return {"report_code": code_digest(), "train_code": tf.code_digest(), "config_sha256": ctx.config_sha(), "preparation_code": prep.get("code"),
            "official": prep["official"], "runs": {k: {"identity": v["identity"], "final_sha256": v["final_sha256"]} for k, v in runs.items()},
            "development_caches": {s: c["development"]["sha256"] for s, c in prep["caches"].items()}, "a01_summary": ctx.src["a01_summary"]["sha256"],
            "f02_summary": ctx.src["f02_summary"]["sha256"], "metadata": {k: a3[k]["sha256"] for k in ("protocol", "clip_index", "window_plans")},
            "v01_outputs": {f"{h}_s{s}": a01.vt["source"]["outputs"][h][s]["sha256"] for h in HALVES for s in ctx.seeds}}


def ids_of(ctx: Ctx, seqs: dict) -> set:
    proto = ctx.proto(); ids = {c for h in HALVES for c in proto["validation"][h]["clips"]}
    ids |= {r[0] for q in (x for h in seqs for x in seqs[h]) for r in q["refs"]} | {q["id"] for h in seqs for q in seqs[h]}
    return ids | {session_of(v) for v in list(ids)}


# ----------------------------------------------------------------------------------------------- one half
def frozen_fact(ctx: Ctx, head, seed: int, runs: dict, device) -> torch.nn.Module:
    key = tf.run_key(seed); path = ctx.root / runs[key]["frozen"] / "final.pt"
    if sha256_file(path) != runs[key]["final_sha256"]:
        raise SystemExit(f"{key}: the frozen weights differ from the run record")
    model = head.build(); model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True); model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def half_systems(ctx: Ctx, h: str, ev, seqs: list[dict], access: DevAccess, device, runs: dict) -> dict:
    a01 = ctx.a01(); stored = ctx.a01_summary()["systems"]; v01 = ctx.v01_summary()["systems"]; prep = tf.read_record(ctx, "preparation.json")
    head = tf.make_head(ctx, fh.TokenRule(**prep["token_rule"])); ids = [q["id"] for q in seqs]
    out = {"systems": {}, "parity": {}, "reproduced": [], "fact": {}, "seqs": seqs, "ev": ev, "reused_outputs": []}
    for s in ctx.seeds:
        cache = ctx.root / prep["caches"][str(s)]["development"]["path"]
        if sha256_file(cache) != prep["caches"][str(s)]["development"]["sha256"]:
            raise SystemExit(f"seed {s}: the development cache differs from its preparation record")
        d = open_half(cache, h, ids, access)
        v = read_members(pv.pinned(ctx.root, a01.vt["source"]["outputs"][h][s]), ("sequence_ids", "centre", "logits", "sample_offsets", "sample_probs"))
        if v["sequence_ids"].tolist() != ids or not np.array_equal(v["centre"], d["centres"]):
            raise SystemExit(f"seed {s} {h}: V01's saved outputs list other sequences or windows than the cache")
        init, _ = a01.init_state("linear", s); mine = d["window"].astype(np.float64) @ init["weight"].double().numpy().T + init["bias"].double().numpy()
        saved = v["logits"].astype(np.float64); par = {"windows": int(len(saved)), "max_abs": float(np.abs(mine - saved).max()), "argmax_equal": bool(np.array_equal(mine.argmax(1), saved.argmax(1)))}
        if par["max_abs"] > 1e-4 or not par["argmax_equal"]:
            raise SystemExit(f"seed {s} {h}: the source classifier's parity failed ({par})")
        out["parity"][f"s{s}"] = par; feats = d["samples"]; off = v["sample_offsets"]
        P = {sid: v["sample_probs"][off[i]:off[i + 1]] for i, sid in enumerate(ids)}; oparams = v01[f"ordered_s{s}"]["b3_choice"]["params"]
        sysd = {"frozen Ordered + B3": rs.row_of(ctx.cfg, decode(oparams, P), ev), "frozen Ordered argmax": rs.row_of(ctx.cfg, {k: p.argmax(1) for k, p in P.items()}, ev)}
        lin = rs.frozen_linear(ctx.root / "models_frozen" / "video_feature_temporal_v1" / f"linear_s{s}" / "final.pt", ctx.a01_summary()["runs"][f"linear_s{s}"]["final_sha256"], device)
        lp = ts.predict(lin, feats, device)
        sysd["A01 Linear + B3"] = rs.row_of(ctx.cfg, decode(ctx.a01_summary()["linear_b3"][str(s)]["params"], lp), ev)
        sysd["A01 Linear argmax"] = rs.row_of(ctx.cfg, {k: p.argmax(1) for k, p in lp.items()}, ev)
        for name_, label in (("mstcn", "A01 MS-TCN"), ("asrf", "A01 ASRF")):
            hd, model = rs.a01_head(ctx, name_, s, device); o = hd.decode(model, rs.placeholder_seqs(feats), device)
            sysd[label] = rs.row_of(ctx.cfg, {k: np.asarray(x["labels"], int) for k, x in o.items()}, ev); del model
        for name, key in REFERENCE_ROWS.items():
            rs.check_stored(sysd[name]["row"], stored[f"{key} s{s}"][h], f"{name} s{s} {h}"); out["reproduced"].append(f"{name} s{s} {h}")
        ident = hashlib.sha256(json.dumps({"run": runs[tf.run_key(s)], "half": h, "ids": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                                           "decoding": ctx.f["decoding"]}, sort_keys=True, default=str).encode()).hexdigest()
        path = ctx.out / "evaluation" / h / f"FACT_s{s}.npz"; fo = load_outputs(path, ident)
        if fo is None:
            torch.use_deterministic_algorithms(True, warn_only=True); model = frozen_fact(ctx, head, s, runs, device)
            fo = ra.fact_outputs(head, model, rs.placeholder_seqs(feats), device); save_outputs(path, ident, fo); del model
        else:
            out["reused_outputs"].append(f"s{s} {h}")
        out["fact"][s] = fo
        sysd["FACT"] = rs.row_of(ctx.cfg, {k: np.asarray(x["labels"], int) for k, x in fo.items()}, ev)
        sysd["FACT frame branch"] = rs.row_of(ctx.cfg, {k: np.asarray(x["frame_labels"], int) for k, x in fo.items()}, ev)
        for name, x in sysd.items():
            out["systems"][f"{name} s{s}"] = x
        torch.cuda.empty_cache(); print(f"{h} seed {s}: parity, controls reproduced and FACT decoded", flush=True)
    return out


def read_halves(ctx: Ctx, data: dict, access: DevAccess, runs: dict) -> dict:
    device = pv.setup_gpu(); t0 = time.time(); per = {}
    try:
        for h in ("calibration", "selection"):
            if h not in data:
                continue
            if data[h] != h and data[h] in per:
                per[h] = per[data[h]]; continue
            ev, seqs = ctx.half(data[h], access); per[h] = half_systems(ctx, data[h], ev, seqs, access, device, runs)
    finally:
        torch.cuda.empty_cache(); ctx.gpu_seconds += time.time() - t0
    return per


# ----------------------------------------------------------------------------------------------- summary
def summarise(ctx: Ctx, per: dict, runs: dict) -> tuple[dict, dict]:
    res = {"systems": {}, "kind_counts": {}, "transcripts": {}, "switches": {}, "join_cuts": {}, "mechanism": {}, "tokens": {}, "comparisons": {}, "accounting": {},
           "native_changes": {}}
    private = {}
    for h in HALVES:
        x = per[h]; seqs, ev = x["seqs"], x["ev"]
        for name, sysd in x["systems"].items():
            res["systems"].setdefault(name, {})[h] = sysd["row"]
            res["kind_counts"].setdefault(name, {})[h] = st.kind_counts(seqs, sysd["preds"], sysd["tables"])
            res["transcripts"].setdefault(name, {})[h] = st.transcript_summary(seqs, sysd["preds"])
            res["switches"].setdefault(name, {})[h] = st.switch_summary(seqs, sysd["preds"])
            mf = sysd["tables"]["multi_frag"]; res["join_cuts"].setdefault(name, {})[h] = {"cuts": int(mf["false_split"].sum()), "same_view_joins": int(mf["same_join"].sum())}
            if not name.startswith("frozen Ordered argmax") and not name.startswith("A01 Linear argmax"):
                res["mechanism"].setdefault(name, {})[h] = ra.mechanism(ev, sysd["preds"], 16)
        for s in ctx.seeds:
            res["tokens"][f"s{s}|{h}"] = ra.token_table(ev, x["fact"][s])
            draws = {n: st.system_draws(x["systems"][f"{n} s{s}"]["tables"]) for n in ("FACT", *CONTROLS)}
            for cname, (a, b) in COMPARE.items():
                res["comparisons"][f"s{s}|{cname}|{h}"] = st.paired(draws[a], draws[b])
            for c in CONTROLS:
                A, B = x["systems"][f"FACT s{s}"], x["systems"][f"{c} s{s}"]; seg, priv = st.segment_accounting(seqs, A["preds"], B["preds"])
                nc = rcs.native_changes(A["tables"]["native"], B["tables"]["native"])
                res["accounting"][f"s{s}|FACT vs {c}|{h}"] = {"segments": seg}
                res["native_changes"][f"s{s}|FACT vs {c}|{h}"] = {k: ({"fixed": len(v["fixed"]), "new": len(v["new"]), "fixed_sessions": v["fixed_sessions"],
                                                                         "new_sessions": v["new_sessions"]} if "fixed_sessions" in v else v) for k, v in nc.items()}
                private[f"s{s}|FACT vs {c}|{h}"] = {"segments": priv, "native": {k: v for k, v in nc.items() if "fixed_sessions" in v}}
            print(f"{h} seed {s}: comparisons and accounting done", flush=True)
    res["verdicts"] = {f"{c}|{m}": {h: st.consistency([res["comparisons"][f"s{s}|{c}|{h}"][m] for s in ctx.seeds]) for h in HALVES} for c in COMPARE for m in METRICS}
    res["accounting_pooled"] = pooled(ctx, res["accounting"])
    rows = {n: {h: res["systems"][n][h] for h in HALVES} for n in res["systems"]}
    a01 = ctx.a01_summary()
    res["goal"] = {"FACT": goal_of([rows[f"FACT s{s}"] for s in ctx.seeds], ctx.fv["goal"]),
                   "A01 ASRF": goal_of([rows[f"A01 ASRF s{s}"] for s in ctx.seeds], ctx.fv["goal"]),
                   "A01 Linear + B3": rt.goal([rows[f"A01 Linear + B3 s{s}"] for s in ctx.seeds], [a01["linear_b3"][str(s)]["status"] == "feasible" for s in ctx.seeds], ctx.fv["goal"]),
                   "frozen Ordered + B3": goal_of([rows[f"frozen Ordered + B3 s{s}"] for s in ctx.seeds], ctx.fv["goal"])}
    ver = {c: res["verdicts"][f"{c}|multi segment F1@50"] for c in COMPARE}; res["denominators"] = {h: denominators(per[h]["seqs"]) for h in HALVES}
    points = [{**i, "denominator_per_seed": res["denominators"][i["half"]][i["population"]][i["what"]], "seeds": len(ctx.seeds)}
              for i in point_estimate_items(res["accounting"], ctx.seeds)]
    trade = trade_cells(res["accounting"], rows, res["goal"]["FACT"]["constraint_failures"], ctx.seeds)
    for cell in trade:
        cell["intervals"] = {m: res["comparisons"][f"s{cell['seed']}|FACT - {cell['control']}|{cell['half']}"][m] for m in ("multi FP segments", "native macro-F1")}
    res["decision"] = readings(ver, res["goal"]["FACT"], local_evidence(ver, points), trade); notes = res["decision"]["notes"]
    by_seed = lambda c, h: {f"s{s}": res["comparisons"][f"s{s}|{c}|{h}"]["multi segment F1@50"] for s in ctx.seeds}  # noqa: E731
    if "one_half_or_local_effect" in notes:
        for item in notes["one_half_or_local_effect"]["consistent"]:
            item["intervals"] = by_seed(item["comparison"], item["half"])
    if "uncertain_or_no_qualifying_gain" in notes:
        notes["uncertain_or_no_qualifying_gain"]["primary_by_seed"] = {h: by_seed(PRIMARY, h) for h in HALVES}
    res["stored_rows"] = stored_rows(a01)
    return res, private


def pooled(ctx: Ctx, acc: dict) -> dict:
    out = {}
    for c in CONTROLS:
        for h in HALVES:
            for kind in st.KINDS:
                xs = [acc[f"s{s}|FACT vs {c}|{h}"]["segments"][kind] for s in ctx.seeds]
                out[f"FACT vs {c}|{h}|{kind}"] = {n: int(sum(x[n]["segments"] for x in xs)) for n in ("recovered", "lost", "new_unmatched_pred", "removed_unmatched_pred")} | \
                    {f"{side}_{k}": int(sum(x[side][k] for x in xs)) for side in ("recovered", "lost") for k in ("short", "xyx_a4c_a5c")}
    return out


def tables_md(res: dict) -> str:
    md = st.md_table
    parts = ["# FACT video feature round: tables\n",
             "Recompute: `python scripts/report_fact_video_feature.py --config configs/experiments/fact_video_feature_v1.yaml --selection "
             "--manifest-sha256 <sha> --manifest-commit <commit>`\n"]
    rows = [{"system": k, "half": h, "segment F1@50": r.get("segment_f1_50_multi"), "boundary P": r.get("boundary_precision_multi"), "boundary R": r.get("boundary_recall_multi"),
             "boundary F1": r.get("boundary_f1_multi"), "native macro-F1": r.get("native_macro_f1"), "switches per clip": r.get("native_internal_cuts_per_clip"),
             "same-view join cuts": r.get("same_class_cut_rate_multi"), "pairs F1@50": r.get("segment_f1_50_pairs"), "A4C read as A5C": r.get("A4C read as A5C"),
             "A5C read as A4C": r.get("A5C read as A4C")} for k, by in res.get("systems", {}).items() for h, r in by.items()]
    parts += ["## Systems\n", md(rows)]
    parts += ["## Paired comparisons (seed | comparison | half; 95% bootstrap intervals)\n",
              md([{"comparison": k, **{m: st.ci(v[m]) for m in METRICS}} for k, v in res.get("comparisons", {}).items()])]
    parts += ["## Three-seed verdicts\n", md([{"comparison | metric": k, **v} for k, v in res.get("verdicts", {}).items()])]
    acc = []
    for key, by in res.get("accounting", {}).items():
        for kind in st.KINDS:
            x = by["segments"][kind]
            acc.append({"seed | pair | half": key, "population": kind, "TP FACT": x["a"]["tp"], "FP FACT": x["a"]["fp"], "FN FACT": x["a"]["fn"], "TP control": x["b"]["tp"],
                        "FP control": x["b"]["fp"], "FN control": x["b"]["fn"], "recovered": x["recovered"]["segments"], "recovered sessions": x["recovered"]["sessions"],
                        "lost": x["lost"]["segments"], "lost sessions": x["lost"]["sessions"], "recovered <=16": x["recovered"]["short"], "lost <=16": x["lost"]["short"],
                        "recovered X-Y-X": x["recovered"]["xyx_a4c_a5c"], "lost X-Y-X": x["lost"]["xyx_a4c_a5c"], "new unmatched pred": x["new_unmatched_pred"]["segments"],
                        "removed unmatched pred": x["removed_unmatched_pred"]["segments"]})
    parts += ["## Reference segments recovered and lost by FACT against each control\n", md(acc)]
    mech = [{"system | half": f"{n} | {h}", "unit": u, "segments": v["segments"], "matched": v["matched"], "recall": v["recall"], "wrong view": v["wrong view"],
             "no segment at IoU 0.5": v["no segment at IoU 0.5"], "missed sessions": v["missed_sessions"]}
            for n, by in res.get("mechanism", {}).items() for h, m in by.items() for u, v in m.items()]
    parts += ["## Short segments and X-Y-X middles (multi streams, raw denominators)\n", md(mech)]
    parts += ["## FACT tokens (multi streams)\n", md([{"seed | half": k, **v} for k, v in res.get("tokens", {}).items()])]
    parts += ["## Goal\n", md([{"group": g, "selection mean": v["mean_segment_f1"]["selection"], "calibration mean": v["mean_segment_f1"]["calibration"],
                                 "constraint failures": len(v["constraint_failures"]), "met": v["met"]} for g, v in res.get("goal", {}).items()])]
    if "decision" in res:
        parts += ["## Readings\n", "```\n" + json.dumps(res["decision"]["applied"], indent=1) + "\n```\n"]
    return "\n".join(parts)


def figures(ctx: Ctx, runs: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for key, r in runs.items():
        s = int(key.rsplit("_s", 1)[1]); recs = r["records"]; x = [e["updates"] for e in recs]; kw = {"ls": ("-", "--", ":")[s], "lw": 1.2, "label": key}
        ax[0].plot(x, [e["loss"]["total"] for e in recs], **kw); ax[1].plot(x, [e["monitor"]["segment_f1_50"] for e in recs], **kw)
        ax[2].plot(x, [e["monitor"]["transcripts_exact"] for e in recs], **kw)
    for a_, t in zip(ax, ("official training loss (epoch mean)", "segment F1@50 on the 16 check sequences", "exact transcripts of the 16 check sequences")):
        a_.set_title(t, fontsize=9); a_.set_xlabel("updates"); a_.grid(alpha=0.3); a_.axvline(11719, color="grey", lw=0.8, ls="--")
    ax[0].set_yscale("log"); ax[0].legend(fontsize=7)
    fig.suptitle("Training data only; the final weights are kept and nothing is chosen from these curves (dashed line: learning rate x 0.1)", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "training_curves.png", dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 3.4)); ax.set_xlim(0, 60); ax.set_ylim(0, 10); ax.axis("off")
    for i in range(30):
        ax.add_patch(plt.Rectangle((1 + i * 1.6, 7.2), 1.3, 1.3, color=("#cfd8e3", "#e8d5c4", "#d5e8d4")[min(i // 10, 2)]))
    ax.text(1, 9.1, "a stream's (T, 512) frozen video features (one row per 10 Hz sample; three views here)", fontsize=9)
    for j in range(12):
        ax.add_patch(plt.Rectangle((1 + j * 4.0, 3.6), 3.2, 1.3, color="#4c72b0" if j < 3 else "#bbbbbb"))
    ax.text(1, 5.4, "12 action tokens (int(1.5 x the longest training transcript, 8)); trained tokens claim segments, the rest predict null", fontsize=9)
    ax.text(1, 2.2, "frame and token branches update each other by cross-attention over the blocks i, u, U (official GTEA configuration);", fontsize=9)
    ax.text(1, 1.0, "the official output of the last block assigns every frame to a token and names it; no B3, no threshold search", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "fact_tokens.png", dpi=130); plt.close(fig)


def save_private(out: Path, per: dict, private: dict) -> None:
    write_atomic(Path(out) / "private_accounting.json", json.dumps(private, default=str))


def evaluate_all(ctx: Ctx, data: dict, access: DevAccess, runs: dict) -> tuple[dict, dict, dict, set]:
    per = read_halves(ctx, data, access, runs); res, private = summarise(ctx, per, runs)
    res["parity"] = {h: per[h]["parity"] for h in HALVES}; res["reproduced"] = {h: per[h]["reproduced"] for h in HALVES}
    res["reused_outputs"] = {h: per[h]["reused_outputs"] for h in HALVES}
    res["runs"] = {k: {x: v for x, v in r.items() if x != "records"} for k, r in runs.items()}; res["training_last_epoch"] = {k: r["records"][-1] for k, r in runs.items()}
    ids = ids_of(ctx, {h: per[h]["seqs"] for h in HALVES}); flat = json.loads(json.dumps(res, default=str)); assert_no_ids(flat, ids, "summary")
    if any(x in tables_md(flat) for x in ids):
        raise SystemExit("an id appears in the report tables")
    return flat, per, private, ids


# ----------------------------------------------------------------------------------------------- stages
def run_stage(ctx: Ctx, name: str, fn) -> None:
    b = ctx.fv["budget"]
    if ts.ledger_hours(ctx.ledger, "cpu") >= float(b["cpu_hours_max"]) or ts.ledger_hours(ctx.ledger, "gpu") >= float(b["gpu_hours_max"]):
        raise SystemExit("the CPU or GPU budget is spent")
    t0, started, ok, note = time.time(), ts.now(), False, ""; ctx.gpu_seconds = 0.0
    try:
        env = ts.environment()
        if any(str(env[k]) != str(v) for k, v in ctx.src["environment"].items()):
            raise SystemExit(f"environment {env} differs from the pinned {ctx.src['environment']}")
        fn(ctx); ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        total = time.time() - t0; g = min(float(ctx.gpu_seconds), total)
        if g > 0:
            ts.ledger_append(ctx.ledger, f"{name} (GPU inference)", "gpu", started, g, ok, note)
        ts.ledger_append(ctx.ledger, f"{name} (CPU)", "cpu", started, total - g, ok, note)


def stage_calibration(ctx: Ctx) -> None:
    undo = rs.forbid_encoders()
    try:
        runs = verify_runs(ctx); prov = provenance(ctx, runs)
        per = read_halves(ctx, {"calibration": "calibration"}, DevAccess({"calibration"}), runs)["calibration"]
        man = {"provenance": prov, "half_opened": "calibration", "decoding": {"FACT": ctx.f["decoding"], "diagnostic": ctx.f["diagnostic_readout"]},
               "controls": {"A01 Linear + B3": {str(s): ctx.a01_summary()["linear_b3"][str(s)]["params"] for s in ctx.seeds},
                            "frozen Ordered + B3": {str(s): ctx.v01_summary()["systems"][f"ordered_s{s}"]["b3_choice"]["params"] for s in ctx.seeds},
                            "A01 ASRF": "official refinement_with_boundary at boundary_th 0.5", "A01 MS-TCN": "official last-stage argmax"},
               "comparisons": {"primary": PRIMARY, "mandatory": [c for c in COMPARE if c != PRIMARY], "metrics": list(METRICS),
                               "bootstrap": {"draws": 2000, "seed": 20261002}}, "readings": ctx.fv["readings"], "goal": ctx.fv["goal"],
               "calibration_evidence": {"parity": per["parity"], "reproduced": per["reproduced"]}}
        flat = json.loads(json.dumps(man, default=str)); assert_no_ids(flat, ids_of(ctx, {"calibration": per["seqs"]}), "manifest")
        for p in (ctx.twin, ctx.root / ctx.relpath):
            if p.is_file() and json.loads(p.read_text()) != flat:
                raise SystemExit("an earlier evaluation manifest differs from this one; it is never overwritten")
        for p in (ctx.twin, ctx.root / ctx.relpath):
            p.parent.mkdir(parents=True, exist_ok=True); write_atomic(p, json.dumps(flat, indent=1))
    finally:
        undo()


def stage_dry_run(ctx: Ctx) -> None:
    undo = rs.forbid_encoders()
    try:
        runs = verify_runs(ctx); prov = provenance(ctx, runs)
        if not ctx.twin.is_file() or json.loads(ctx.twin.read_text()).get("provenance") != json.loads(json.dumps(prov, default=str)):
            raise SystemExit("run --calibration first (the manifest is missing or belongs to other code or inputs)")
        flat, per, private, _ = evaluate_all(ctx, {"calibration": "calibration", "selection": "calibration"}, DevAccess({"calibration"}), runs)
        out = ctx.out / "dry_run"; out.mkdir(parents=True, exist_ok=True); flat["dry_run"] = "the calibration half stands in for both halves; a code check, not a result"
        figures(ctx, runs, out); save_private(out, per, private)
        write_atomic(out / "summary.json", json.dumps(flat, indent=1)); write_atomic(out / "report_tables.md", tables_md(flat)); print("dry run ok (a code check, not a result)", flush=True)
    finally:
        undo()


def stage_selection(ctx: Ctx) -> None:
    undo = rs.forbid_encoders()
    try:
        runs = verify_runs(ctx); prov = provenance(ctx, runs)
        manifest_barrier(ctx.root / ctx.relpath, ctx.twin, ctx.proof, ctx.relpath, ctx.manifest_sha256, ctx.manifest_commit, prov)
        out = ctx.out / "selection"; openings = open_selection(ctx, out)
        flat, per, private, ids = evaluate_all(ctx, {"calibration": "calibration", "selection": "selection"}, DevAccess({"calibration", "selection"}), runs)
        led = [json.loads(x) for x in ctx.ledger.read_text().splitlines() if x.strip()]
        flat["identity"] = {**prov, "manifest_sha256": ctx.manifest_sha256, "manifest_commit": ctx.manifest_commit, "proof_made": json.loads(ctx.proof.read_text()).get("made"),
                            "selection_openings": openings}
        flat["cost"] = {"gpu_hours_before_this_stage": ts.ledger_hours(ctx.ledger, "gpu"), "cpu_hours_before_this_stage": ts.ledger_hours(ctx.ledger, "cpu"),
                        "earlier_stages": [{k: x[k] for k in ("job", "kind", "started", "finished", "hours", "ok")} for x in led]}
        assert_no_ids(flat, ids, "results summary"); figures(ctx, runs, ctx.results); save_private(out, per, private)
        write_atomic(ctx.results / "summary.json", json.dumps(flat, indent=1)); write_atomic(ctx.results / "report_tables.md", tables_md(flat))
        write_atomic(out / "done.json", json.dumps({"manifest_sha256": ctx.manifest_sha256, "manifest_commit": ctx.manifest_commit, "finished": ts.now()}))
        print(json.dumps(flat["decision"]["applied"]), flush=True)
    finally:
        undo()


def stage_proof(ctx: Ctx) -> None:
    if not ctx.manifest_commit:
        raise SystemExit("--proof needs --manifest-commit")
    proof = git_proof(ctx.root, ctx.manifest_commit, ctx.relpath); verify_proof(proof, (ctx.root / ctx.relpath).read_bytes(), ctx.relpath, proof["commit"])
    ctx.proof.parent.mkdir(parents=True, exist_ok=True); write_atomic(ctx.proof, json.dumps(proof, indent=1))
    print(json.dumps({"commit": proof["commit"], "blob": proof["blob"], "file_sha256": proof["file_sha256"], "objects": len(proof["objects"])}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--manifest-sha256"); ap.add_argument("--manifest-commit", help="the full 40-character commit")
    ap.add_argument("--repeat-reason", help="--selection only: why an interrupted selection evaluation is repeated (disclosed in the summary)")
    g = ap.add_mutually_exclusive_group(required=True)
    for s in ("calibration", "dry_run", "proof", "selection"):
        g.add_argument(f"--{s.replace('_', '-')}", dest=s, action="store_true")
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config)
    ctx.manifest_sha256, ctx.manifest_commit, ctx.repeat_reason = a.manifest_sha256, a.manifest_commit, a.repeat_reason
    if a.proof:
        stage_proof(ctx); return
    stage = next(s for s in ("calibration", "dry_run", "selection") if getattr(a, s))
    run_stage(ctx, stage, {"calibration": stage_calibration, "dry_run": stage_dry_run, "selection": stage_selection}[stage])


if __name__ == "__main__":
    main()
