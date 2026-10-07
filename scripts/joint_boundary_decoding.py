#!/usr/bin/env python
"""Joint boundary decoding round (configs/experiments/joint_boundary_decoding_v1.yaml;
docs/superpowers/plans/2026-10-07-joint-boundary-decoding.md). Nothing is trained.

J0 and J1 decode the frozen L1_s{k} class probabilities with one exact Viterbi pass
(echo_routing/temporal/joint_decoding.py); J1 adds the boundary probabilities of the frozen B_s{k} + ASRF. Steps, in order:

  --extract   GPU, no metric. Boundary probabilities of every validation sequence from the frozen B_s{k} + ASRF; the
              official decode of the same outputs must reproduce the stored B + ASRF predictions; mean b around true
              changes on training and calibration streams (alignment); every L1 probability array checked against its
              sequence and against the B_s{k} store.
  --choose    Calibration half only: the 25-setting grid per seed. J1 is the best feasible setting of the grid, J0 the best
              feasible setting with lambda = 0 (native macro-F1 >= 0.980, multi boundary recall >= 0.95, then the highest
              segment F1@50, ties to the lower order). Writes choices.json; the selection half is never built.
  --evaluate  Both halves once with the frozen choices: J0, J1, L1 + B3, B + ASRF and the references; paired-bootstrap
              contrasts; the fixed diagnostics (three clips, merged changes, extra changes, correct visual evidence lost,
              unit causes); the goal and the stable rule.
  --dry-run   Training data only: the whole pipeline on a training subset; no validation data, no metric printed.

  python scripts/joint_boundary_decoding.py --config configs/experiments/joint_boundary_decoding_v1.yaml --extract
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg  # noqa: E402
from scripts.diagnose_center_recognition import native_bootstrap  # noqa: E402
from scripts.ev9v_complementarity import md_table  # noqa: E402
from scripts.report_center_supervision import FLAT, PAIRS, evaluate, references, validation  # noqa: E402
from scripts.report_fan224_visual import strip_ids  # noqa: E402
from scripts.report_l1_asrf import (TOL, classify_changes, compare_one, evaluate_system, fragment_transfer, goal, l1_b3,  # noqa: E402
                                    transfer_table, unit_table)
from scripts.report_model_development import BOOT_SEED, N_BOOT, paired_bootstrap  # noqa: E402
from scripts.train_l1_asrf import make_head  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from echo_routing.config import gpu_memory_fraction, load_config, load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import select_setting, sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of, stream_units  # noqa: E402
from echo_routing.evaluate.metrics_boundary import match_boundaries  # noqa: E402
from echo_routing.features.cache import read_index, variant_dir  # noqa: E402
from echo_routing.temporal.dev_data import EvalSet, FeatureStore, eval_set, native_seq  # noqa: E402
from echo_routing.temporal.dev_run import HALVES, Inputs, load_json  # noqa: E402
from echo_routing.temporal.joint_decoding import joint_viterbi, joint_viterbi_grid  # noqa: E402
from echo_routing.temporal.segments import boundaries_from_labels  # noqa: E402

NS = "joint_boundary_decoding_v1"
SYSTEMS = ("J0", "J1")
OFFSETS = tuple(range(-3, 4))
RR = "recording right, fragment right"
CONTRASTS = (("J1", "J0"), ("J1", "L1 + B3"), ("J1", "B + ASRF"), ("J0", "L1 + B3"))
STRONGEST = ("E0 + B3", "B7 + B3")
METRICS = ("multi segment F1@50", "multi boundary recall", "native macro-F1")
L1B3_DETAILS = "runs/fan224_visual_adaptation_v1/report/details.json"
L1B3_PROBS = "runs/fan224_visual_adaptation_v1/report/probs_L1_s{seed}.npz"


# ----------------------------------------------------------------------------------------------- pure helpers
def grid_of(jd: dict) -> list[tuple[float, float]]:
    """The settings in their fixed order: lambda ascending, then kappa ascending."""
    return [(float(lam), float(kap)) for lam in jd["grid"]["lambda"] for kap in jd["grid"]["kappa"]]


def choose_systems(recs: list[dict], native_min: float, recall_min: float) -> dict:
    """J1: the best feasible setting of the whole grid; J0: the best feasible setting with lambda = 0. No fallback."""
    out = {}
    for name, pool in (("J0", [r for r in recs if r["params"]["lambda"] == 0.0]), ("J1", recs)):
        s = select_setting(pool, native_min, recall_min); ch = s["chosen"]
        out[name] = {"status": s["status"], "n_feasible": s["n_feasible"], "n_settings": s["n_settings"],
                     "order": ch["order"] if ch else None, "params": ch["params"] if ch else None,
                     "calibration": {k: ch[k] for k in FLAT} if ch else None}
    return out


def boundary_profile(bprobs: list[np.ndarray], changes: list[np.ndarray]) -> dict:
    """Mean boundary probability at each offset from the true changes (alignment check: the peak should be offset 0)."""
    sums, counts = Counter(), Counter()
    for b, ch in zip(bprobs, changes):
        for r in np.asarray(ch, int):
            for o in OFFSETS:
                if 0 <= r + o < len(b):
                    sums[o] += float(b[r + o]); counts[o] += 1
    mean = {o: sums[o] / counts[o] for o in OFFSETS if counts[o]}
    return {"changes": int(sum(len(c) for c in changes)), "mean_b_by_offset": {str(o): v for o, v in mean.items()},
            "peak_offset": int(max(mean, key=mean.get)) if mean else None}


def candidates(b: np.ndarray, th: float = 0.5) -> list[int]:
    """ASRF's accepted peaks (the official argrelmax at boundary_th, re-implemented), the first frame excluded."""
    x = np.asarray(b, np.float64).copy(); x[x < th] = 0.0
    if x.size < 3:
        return []
    peak = np.concatenate([[True], (x[:-2] < x[1:-1]) & (x[2:] < x[1:-1]), [False]])
    return [int(i) for i in np.flatnonzero(peak) if i > 0]


def fragment_bounds(recipe) -> np.ndarray:
    return np.cumsum([0, *[f.end - f.start for f in recipe.fragments]])


def fragment_at(bounds: np.ndarray, pos: int) -> int:
    return int(np.searchsorted(bounds, pos, side="right") - 1)


def extra_positions(final: np.ndarray, joins: np.ndarray, semantic: np.ndarray) -> list[tuple[int, str]]:
    """Extra final changes as classify_changes counts them: unmatched and more than TOL from every true change; inside a
    clip (more than TOL from every join) or at a same-view join."""
    pred_b = boundaries_from_labels(np.asarray(final, int)); sem = np.asarray(semantic, int); joins = np.asarray(joins, int)
    used = {int(pred_b[i]) for i, _ in match_boundaries(pred_b, sem, TOL)}
    extra = [int(b) for b in pred_b if int(b) not in used and (sem.size == 0 or np.abs(sem - b).min() > TOL)]
    return [(b, "inside a clip" if joins.size == 0 or np.abs(joins - b).min() > TOL else "at a same-view join") for b in extra]


def change_records(final: np.ndarray, cuts: list[int], stream) -> tuple[list[dict], list[dict]]:
    """Per true view change of one multi stream (category, the recordings on both sides) and per extra change (kind,
    the recording around it, or the two recordings at the nearest join)."""
    sid = stream.recipe.recipe_id; frs = stream.recipe.fragments; bounds = fragment_bounds(stream.recipe)
    rows, _ = classify_changes(final, cuts, stream.labels, stream.joins, stream.semantic_boundaries)
    changes = [{"stream": sid, "r": int(r["r"]), "cat": r["cat"],
                "videos": sorted({frs[fragment_at(bounds, r["r"] - 1)].video_id, frs[fragment_at(bounds, r["r"])].video_id})} for r in rows]
    extras = []
    for p, kind in extra_positions(final, stream.joins, stream.semantic_boundaries):
        if kind == "inside a clip":
            vids = [frs[fragment_at(bounds, p)].video_id]
        else:
            j = int(stream.joins[np.abs(stream.joins - p).argmin()]); vids = [frs[fragment_at(bounds, j - 1)].video_id, frs[fragment_at(bounds, j)].video_id]
        extras.append({"stream": sid, "pos": p, "kind": kind, "videos": sorted(set(vids))})
    return changes, extras


def dedup(rows: list[dict]) -> dict:
    vids = {v for r in rows for v in r["videos"]}
    return {"count": len(rows), "recordings": len(vids), "sessions": len({session_of(v) for v in vids})}


def merged_recovery(ref_changes: list[dict], sys_changes: list[dict]) -> dict:
    """The reference system's merged changes (an ASRF candidate within TOL, both sides one class) and what the other
    system makes of each: found with both views right, found with a view wrong, or still missed (by category)."""
    by = {(r["stream"], r["r"]): r for r in sys_changes}
    merged = [r for r in ref_changes if r["cat"] == "merged"]
    now = [by[(r["stream"], r["r"])] for r in merged]
    found = [r for r in now if r["cat"].startswith("found")]
    return {"merged_in_reference": dedup(merged), "found": dedup(found),
            "found_both_views_right": sum(r["cat"] == "found, both views right" for r in now),
            "found_a_view_wrong": sum(r["cat"] == "found, a view wrong" for r in now),
            "still_missed": dict(Counter(r["cat"] for r in now if not r["cat"].startswith("found"))),
            "_ids": {"found": [[r["stream"], r["r"]] for r in found]}}


def extra_summary(extras: list[dict]) -> dict:
    return {"all": dedup(extras), **{k: dedup([e for e in extras if e["kind"] == k]) for k in ("inside a clip", "at a same-view join")}}


def native_cut_summary(preds: dict, natives) -> dict:
    cuts = {s.seq_id: int(boundaries_from_labels(np.asarray(preds[s.seq_id])).size) for s in natives}
    hit = [v for v, c in cuts.items() if c]
    return {"clips": len(cuts), "internal_switches": int(sum(cuts.values())), "clips_with_switches": len(hit),
            "sessions_with_switches": len({session_of(v) for v in hit})}


def clip_reads(native_table, ids: list[str]) -> dict:
    """How many of the given native clips a system reads correctly (majority of its samples)."""
    t = native_table.set_index("video_id").loc[ids]; right = t.index[t["majority"] == t["label"]].tolist()
    return {"clips": len(ids), "right": len(right), "sessions_right": len({session_of(v) for v in right})}


def native_diff(a, b) -> dict:
    """Native clips a reads correctly and b misreads (fixed), and the reverse (new), with sessions."""
    j = a.merge(b, on=["video_id", "label"], suffixes=("_a", "_b"))
    fixed = j[(j["majority_a"] == j["label"]) & (j["majority_b"] != j["label"])]["video_id"].tolist()
    new = j[(j["majority_a"] != j["label"]) & (j["majority_b"] == j["label"])]["video_id"].tolist()
    return {"fixed": len(fixed), "fixed_sessions": len({session_of(v) for v in fixed}), "new": len(new),
            "new_sessions": len({session_of(v) for v in new}), "_ids": {"fixed": fixed, "new": new}}


def visual_loss(transfer: list[dict], units: list[dict]) -> dict:
    """Fragments the frozen head reads correctly as fragment and as recording (majority rule) whose unit is unmatched."""
    outcome = {(u["stream"], u["unit"]): u["cause"] for u in units}
    lost = [r for r in transfer if r["majority_category"] == RR and outcome[(r["stream"], r["unit"])] != "matched"]
    return {"fragments": len(lost), "recordings": len({r["video"] for r in lost}), "sessions": len({session_of(r["video"]) for r in lost}),
            "of_fragments": sum(r["majority_category"] == RR for r in transfer)}


def stable(per_seed: list[dict | None]) -> dict:
    """The config's stable rule on segment F1@50 contrasts: every seed's selection interval above zero and no seed's
    calibration interval below zero. A missing seed (an infeasible choice) fails the rule."""
    if not per_seed or any(c is None for c in per_seed):
        return {"stable": False, "reason": "a seed has no feasible choice"}
    sel = [c["selection"]["bootstrap"]["multi segment F1@50"]["ci95"][0] > 0 for c in per_seed]
    rev = [c["calibration"]["bootstrap"]["multi segment F1@50"]["ci95"][1] < 0 for c in per_seed]
    return {"stable": bool(all(sel) and not any(rev)), "selection_above_zero": sel, "calibration_below_zero": rev}


def three_clips(details: dict, seeds: list[int]) -> list[str]:
    """Selection native clips that B + ASRF misreads and L1 + B3 reads correctly, the same in every seed."""
    sets = [sorted({v for cat in ("A4C read as A5C", "A5C read as A4C") for v in details["comparisons"][f"s{s}|B - (L1 + B3)"]["selection"]["_ids"][cat]["new"]})
            for s in seeds]
    if any(x != sets[0] for x in sets) or len(sets[0]) != 3:
        raise SystemExit("the L1 + ASRF round's three lost clips are not the same three in every seed")
    return sets[0]


# ----------------------------------------------------------------------------------------------- inputs
def seeds_of(jd: dict) -> list[int]:
    return [int(s) for s in jd["seeds"]]


def check_inputs(jd: dict, root: Path) -> dict:
    out = {}
    for key in ("protocol", "e1_feature_manifest", "l1_asrf_config", "l1_asrf_summary", "l1_asrf_details", "fan224_details"):
        spec = jd["inputs"][key]
        if (got := sha256_file(root / spec["path"])) != spec["sha256"]:
            raise SystemExit(f"input {key} {spec['path']}: sha256 {got[:12]} != {spec['sha256'][:12]}")
        out[key] = got
    if jd["inputs"]["fan224_details"]["path"] != L1B3_DETAILS:   # l1_b3 (report_l1_asrf.py) opens these exact paths
        raise SystemExit(f"fan224_details must be {L1B3_DETAILS}, the file that l1_b3 reads")
    for seed in seeds_of(jd):
        s = jd["inputs"]["seeds"][seed]
        if s["l1_probs"]["path"] != L1B3_PROBS.format(seed=seed):
            raise SystemExit(f"seed {seed}: l1_probs must be {L1B3_PROBS.format(seed=seed)}, the file that l1_b3 reads")
        for key in ("l1_probs", "b_feature_manifest", "asrf_preds"):
            if (got := sha256_file(root / s[key]["path"])) != s[key]["sha256"]:
                raise SystemExit(f"seed {seed} {key}: sha256 {got[:12]} != {s[key]['sha256'][:12]}")
        sel = root / s["asrf"]["frozen"] / "selected.pt"; man = json.loads((root / s["asrf"]["frozen"] / "MANIFEST.json").read_text())
        if sha256_file(sel) != s["asrf"]["selected_sha256"] or {f["file"]: f["sha256"] for f in man["files"]}.get("selected.pt") != s["asrf"]["selected_sha256"]:
            raise SystemExit(f"seed {seed}: the frozen B + ASRF checkpoint differs from the config")
        out[f"seed {seed}"] = "verified"
    return out


def l1_asrf_cfg(jd: dict, root: Path) -> dict:
    return load_config(root / jd["inputs"]["l1_asrf_config"]["path"])["l1_asrf"]


def b_inputs(cfg: dict, jd: dict, paths, seed: int) -> Inputs:
    root = paths.repo_root
    return Inputs(cfg, root / jd["inputs"]["protocol"]["path"], root / jd["inputs"]["seeds"][seed]["b_feature_manifest"]["path"], paths.cache_root)


def load_half(path: Path, half: str) -> dict[str, np.ndarray]:
    """One half's arrays of a '{half}|{sequence}' npz; the other half's entries are never read."""
    with np.load(path) as z:
        return {k.split("|", 1)[1]: z[k] for k in z.files if k.startswith(f"{half}|")}


def asrf_model(jd: dict, la: dict, paths, seed: int, device):
    frozen = paths.repo_root / jd["inputs"]["seeds"][seed]["asrf"]["frozen"]
    plan = json.loads((frozen / f"plan_s{seed}.json").read_text()); head = make_head(la, paths, plan, device)
    model = head.build().to(device); model.load_state_dict(torch.load(frozen / "selected.pt", map_location=device, weights_only=True)); model.eval()
    return head, model


# ----------------------------------------------------------------------------------------------- extract
def run_extract(cfg: dict, jd: dict, paths, device) -> None:
    root = paths.repo_root; t0 = time.time(); rec = {"inputs": check_inputs(jd, root), "seeds": {}}; la = l1_asrf_cfg(jd, root)
    out_dir = root / "runs" / NS / "boundaries"; out_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds_of(jd):
        inp = b_inputs(cfg, jd, paths, seed); head, model = asrf_model(jd, la, paths, seed, device); th = float(la["asrf"]["boundary_th"])
        from libs.metric import argrelmax   # the pinned official repository, on sys.path once the head exists
        stored = np.load(root / jd["inputs"]["seeds"][seed]["asrf_preds"]["path"]); arrays, r = {}, {"sequences": {}, "l1_probs": {}}
        for h in HALVES:
            ev = inp.halves[h]; seqs = ev.sequences(); dec = head.decode(model, seqs, device)
            for s in seqs:
                if not np.array_equal(dec[s.seq_id]["labels"], stored[f"{h}|{s.seq_id}"].astype(int)):
                    raise SystemExit(f"seed {seed} {h}: the official decode does not reproduce a stored B + ASRF prediction")
                if candidates(dec[s.seq_id]["boundary_prob"], th) != [int(i) for i in argrelmax(dec[s.seq_id]["boundary_prob"].copy(), threshold=th) if i > 0]:
                    raise SystemExit(f"seed {seed} {h}: candidates() differs from the official argrelmax")
                arrays[f"{h}|{s.seq_id}"] = dec[s.seq_id]["boundary_prob"].astype(np.float32)
            r["sequences"][h] = {"count": len(seqs), "official_decode_reproduces_stored_predictions": True, "candidates_equal_official_argrelmax": True}
            probs = load_half(root / jd["inputs"]["seeds"][seed]["l1_probs"]["path"], h)
            if set(probs) != {s.seq_id for s in seqs} or any(probs[s.seq_id].shape != (s.labels.size, 5) for s in seqs):
                raise SystemExit(f"seed {seed} {h}: the L1 probabilities do not match the sequences")
            store = {s.seq_id: s.prob for s in seqs}
            gap = max(float(np.abs(probs[k] - store[k]).max()) for k in probs)
            if gap > 5e-2:
                raise SystemExit(f"seed {seed} {h}: the L1 probabilities differ from the B store by {gap:.3e}")
            r["l1_probs"][h] = {"sequences": len(probs), "max_abs_gap_to_b_store": gap}
            if h == "calibration":   # alignment on the half that serves the choice
                strm = [*ev.pairs, *ev.multi]
                r["alignment_calibration"] = boundary_profile([dec[s.recipe.recipe_id]["boundary_prob"] for s in strm], [s.semantic_boundaries for s in strm])
        tr = list(inp.train.streams[:300]); dt = head.decode(model, tr, device)
        r["alignment_training"] = boundary_profile([dt[s.seq_id]["boundary_prob"] for s in tr], [boundaries_from_labels(s.labels) for s in tr])
        for k in ("alignment_calibration", "alignment_training"):
            if r[k]["peak_offset"] != 0:
                raise SystemExit(f"seed {seed}: the mean boundary probability peaks at offset {r[k]['peak_offset']} ({k})")
        path = out_dir / f"B_s{seed}.npz"; np.savez_compressed(path, **arrays)
        r["boundaries"] = str(path.relative_to(root)); r["boundaries_sha256"] = sha256_file(path); rec["seeds"][str(seed)] = r
        print(json.dumps({"seed": seed, **{k: r[k] for k in ("sequences", "l1_probs", "alignment_calibration", "alignment_training")}}), flush=True)
        del model; torch.cuda.empty_cache()
    rec["seconds"] = time.time() - t0; (root / "runs" / NS / "extract.json").write_text(json.dumps(rec, indent=1))


# ----------------------------------------------------------------------------------------------- choose
def calibration_set(jd: dict, paths, seed: int) -> EvalSet:
    """The calibration half alone (protocol checked); the selection half is not built."""
    root = paths.repo_root; proto, sha = load_json(root / jd["inputs"]["protocol"]["path"])
    if sha != jd["inputs"]["protocol"]["sha256"]:
        raise SystemExit("protocol sha256 differs from the config")
    fman, _ = load_json(root / jd["inputs"]["seeds"][seed]["b_feature_manifest"]["path"]); key = fman["encoder"]["cache_key"]
    idx = read_index(variant_dir(paths.cache_root, key, "orig")); labels = dict(zip(idx["video_id"], idx["label_index"].astype(int)))
    return eval_set(proto, "calibration", labels, FeatureStore(paths.cache_root, key))


def grid_records(cfg: dict, grid: list, probs: dict, bprobs: dict, ev) -> list[dict]:
    ids = [s.seq_id for s in ev.sequences()]
    if set(ids) != set(probs) or set(ids) != set(bprobs):
        raise SystemExit("probabilities and sequences differ")
    paths_ = {sid: joint_viterbi_grid(probs[sid], bprobs[sid], grid) for sid in ids}; recs = []
    for g, (lam, kap) in enumerate(grid):
        m, _ = evaluate(cfg, {sid: paths_[sid][g] for sid in ids}, ev)
        recs.append({"order": g, "params": {"lambda": lam, "kappa": kap}, **{k: m[k] for k in FLAT}})
    return recs


def extraction(root: Path) -> tuple[dict, Path]:
    ext = root / "runs" / NS / "extract.json"
    if not ext.is_file():
        raise SystemExit("run --extract first")
    rec = json.loads(ext.read_text())
    for s, r in rec["seeds"].items():
        if sha256_file(root / r["boundaries"]) != r["boundaries_sha256"]:
            raise SystemExit(f"seed {s}: the boundary file changed after extraction")
    return rec, ext


def run_choose(cfg: dict, jd: dict, paths, config_path: str) -> None:
    root = paths.repo_root; check_inputs(jd, root); extract, ext = extraction(root); grid = grid_of(jd); ch = jd["choice"]
    out = {"config_sha256": sha256_file(root / config_path), "extract_sha256": sha256_file(ext), "grid": grid, "seeds": {}}
    for seed in seeds_of(jd):
        ev = calibration_set(jd, paths, seed)
        recs = grid_records(cfg, grid, load_half(root / jd["inputs"]["seeds"][seed]["l1_probs"]["path"], "calibration"),
                            load_half(root / extract["seeds"][str(seed)]["boundaries"], "calibration"), ev)
        out["seeds"][str(seed)] = {"choice": choose_systems(recs, float(ch["native_min"]), float(ch["recall_min"])), "grid": recs}
        print(json.dumps({"seed": seed, **{k: {kk: v[kk] for kk in ("status", "params", "n_feasible")} for k, v in out["seeds"][str(seed)]["choice"].items()}}), flush=True)
    path = root / "runs" / NS / "choices.json"; path.write_text(json.dumps(out, indent=1)); print(f"wrote {path} sha256 {sha256_file(path)}")


# ----------------------------------------------------------------------------------------------- evaluate
def decode_system(probs: dict, bprobs: dict, params: dict) -> dict:
    return {sid: joint_viterbi(probs[sid], bprobs[sid], params["lambda"], params["kappa"]) for sid in probs}


def system_diagnostics(preds_h: dict, ev, cuts_h: dict, transfer: list) -> tuple[dict, list[dict]]:
    changes, extras = [], []
    for s in ev.multi:
        c, e = change_records(preds_h[s.recipe.recipe_id], cuts_h[s.recipe.recipe_id], s); changes += c; extras += e
    units, unit_sum = unit_table(preds_h, ev, transfer)
    return {"change_categories": dict(Counter(r["cat"] for r in changes)), "extra_changes": extra_summary(extras),
            "native_switches": native_cut_summary(preds_h, ev.natives), "units": unit_sum, "transfer_table": transfer_table(transfer, units),
            "correct_visual_evidence_lost": visual_loss(transfer, units)}, changes


def stored_preds(path: Path) -> dict:
    with np.load(path) as z:
        return {h: {k.split("|", 1)[1]: z[k].astype(int) for k in z.files if k.startswith(f"{h}|")} for h in HALVES}


def reproduce_b_asrf(rows: dict, diag: dict, old_sys: dict, old_diag: dict, seed: int) -> None:
    keys = ("segment_f1_50_multi", "native_macro_f1", "boundary_recall_multi", "boundary_precision_multi", "native_internal_cuts_per_clip")
    if any(abs(rows[h][k] - old_sys[h][k]) > 1e-12 for h in HALVES for k in keys):
        raise SystemExit(f"seed {seed}: B + ASRF does not reproduce the L1 + ASRF round's rows")
    for h in HALVES:
        old = old_diag["faults"][h]; ex = diag[h]["extra_changes"]
        if diag[h]["change_categories"] != old["by_category"] or (ex["inside a clip"]["count"], ex["at a same-view join"]["count"]) != (old["extra_inside_clip"], old["extra_at_same_view_join"]):
            raise SystemExit(f"seed {seed} {h}: B + ASRF change categories do not reproduce the L1 + ASRF round")


def evaluate_seed(cfg: dict, jd: dict, la: dict, paths, seed: int, choice: dict, seqs: dict, three: list[str], extract: dict) -> tuple[dict, dict, dict]:
    root = paths.repo_root; spec = jd["inputs"]["seeds"][seed]; inp = b_inputs(cfg, jd, paths, seed)
    probs = {h: load_half(root / spec["l1_probs"]["path"], h) for h in HALVES}
    bprobs = {h: load_half(root / extract["seeds"][str(seed)]["boundaries"], h) for h in HALVES}
    th = float(la["asrf"]["boundary_th"])
    cuts = {h: {s.recipe.recipe_id: candidates(bprobs[h][s.recipe.recipe_id], th) for s in inp.halves[h].multi} for h in HALVES}
    preds = {"B + ASRF": stored_preds(root / spec["asrf_preds"]["path"])}; preds["L1 + B3"], chk = l1_b3(cfg, la, paths, seed, inp)
    for name in SYSTEMS:
        if choice[name]["status"] == "feasible":
            preds[name] = {h: decode_system(probs[h], bprobs[h], choice[name]["params"]) for h in HALVES}
    transfer = {h: fragment_transfer({s.recipe.recipe_id: stream_units(s.recipe) for s in inp.halves[h].multi}, lambda v, var: inp.store.get(v, var).prob)
                for h in HALVES}
    res = {"systems": {}, "diagnostics": {}, "comparisons": {}}; tables, changes = {}, {}
    old = json.loads((root / jd["inputs"]["l1_asrf_summary"]["path"]).read_text())
    for name, p in preds.items():
        rows, tables[name] = evaluate_system(cfg, p, inp, seqs); diag = {}
        for h in HALVES:
            diag[h], changes[(name, h)] = system_diagnostics(p[h], inp.halves[h], cuts[h], transfer[h])
        if name == "B + ASRF":
            reproduce_b_asrf(rows, diag, old["systems"][f"B_s{seed} + ASRF"], old["diagnostics"][f"B_s{seed} + ASRF"], seed)
        res["systems"][name] = {**rows, **({"choice": {k: choice[name][k] for k in ("status", "params", "order", "n_feasible")}} if name in SYSTEMS else {}),
                                **({"reproduction": chk} if name == "L1 + B3" else {})}
        res["diagnostics"][name] = {**diag, "three_clips": clip_reads(tables[name]["selection"]["native"], three)}
    if res["diagnostics"]["B + ASRF"]["three_clips"]["right"] != 0 or res["diagnostics"]["L1 + B3"]["three_clips"]["right"] != 3:
        raise SystemExit(f"seed {seed}: the three clips do not reproduce (B + ASRF should read none, L1 + B3 all three)")
    for name in [n for n in SYSTEMS if n in preds]:
        res["diagnostics"][name]["merged_recovery"] = {h: merged_recovery(changes[("B + ASRF", h)], changes[(name, h)]) for h in HALVES}
        for other in ("B + ASRF", "L1 + B3"):
            res["diagnostics"][name][f"native against {other}"] = {h: native_diff(tables[name][h]["native"], tables[other][h]["native"]) for h in HALVES}
        np.savez_compressed(root / "runs" / NS / "report" / f"preds_{name}_s{seed}.npz", **{f"{h}|{k}": v.astype(np.int8) for h in HALVES for k, v in preds[name][h].items()})
    for a, b in CONTRASTS:
        if a in preds and b in preds:
            res["comparisons"][f"{a} - ({b})"] = {h: compare_one(tables[a][h], tables[b][h], preds[a][h], preds[b][h], seqs[h]) for h in HALVES}
    return res, tables, preds


def check_choices(jd: dict, root: Path, config_path: str, ext: Path) -> dict:
    path = root / "runs" / NS / "choices.json"
    if not path.is_file():
        raise SystemExit("run --choose first")
    ch = json.loads(path.read_text())
    if ch["config_sha256"] != sha256_file(root / config_path) or ch["extract_sha256"] != sha256_file(ext):
        raise SystemExit("the config or the extraction changed after the choice")
    if [tuple(g) for g in ch["grid"]] != grid_of(jd):
        raise SystemExit("the grid differs from the config")
    for s in seeds_of(jd):
        again = choose_systems(ch["seeds"][str(s)]["grid"], float(jd["choice"]["native_min"]), float(jd["choice"]["recall_min"]))
        if again != ch["seeds"][str(s)]["choice"]:
            raise SystemExit(f"seed {s}: the stored choice does not follow from the stored grid")
    return {**ch, "sha256": sha256_file(path)}


def run_evaluate(cfg: dict, jd: dict, paths, a) -> None:
    root = paths.repo_root; la = l1_asrf_cfg(jd, root); check_inputs(jd, root); extract, ext = extraction(root)
    choices = check_choices(jd, root, a.config, ext); seeds = seeds_of(jd)
    three = three_clips(json.loads((root / jd["inputs"]["l1_asrf_details"]["path"]).read_text()), seeds)
    inputs_e1, seqs = validation(cfg, {"inputs": {"protocol": jd["inputs"]["protocol"], "feature_manifest": {"path": jd["inputs"]["e1_feature_manifest"]["path"]}}}, paths)
    (root / "runs" / NS / "report").mkdir(parents=True, exist_ok=True); out = root / a.out; out.mkdir(parents=True, exist_ok=True)
    res = {"choices": {s: {k: v for k, v in choices["seeds"][str(s)]["choice"].items()} for s in seeds}, "choices_sha256": choices["sha256"],
           "seeds": {}, "timing": {}}
    tables, preds = {}, {}
    for s in seeds:
        t0 = time.time(); res["seeds"][str(s)], tables[s], preds[s] = evaluate_seed(cfg, jd, la, paths, s, choices["seeds"][str(s)]["choice"], seqs, three, extract)
        res["timing"][str(s)] = time.time() - t0; print(f"seed {s} evaluated", flush=True)
    refs = references({**cfg, "center_supervision": {"inputs": {"protocol": jd["inputs"]["protocol"]}}}, paths, inputs_e1, seqs)
    res["references"] = {k: {h: v[h]["metrics"] for h in HALVES} for k, v in refs.items()}
    for s in seeds:
        for ref in STRONGEST:
            if "J1" in preds[s]:
                res["seeds"][str(s)]["comparisons"][f"J1 - ({ref})"] = {h: {"bootstrap": paired_bootstrap(tables[s]["J1"][h], refs[ref][h]["tables"]),
                                                                           "native_session": native_bootstrap(tables[s]["J1"][h]["native"], refs[ref][h]["tables"]["native"], N_BOOT, BOOT_SEED)}
                                                                       for h in HALVES}
    res["stable"] = {f"J1 - ({b})": stable([res["seeds"][str(s)]["comparisons"].get(f"J1 - ({b})") for s in seeds]) for b in ("J0", "L1 + B3", "B + ASRF")}
    for name in SYSTEMS:
        rows = [{"chosen": {"status": res["choices"][s][name]["status"]}, **{h: res["seeds"][str(s)]["systems"][name][h] for h in HALVES}}
                for s in seeds if name in res["seeds"][str(s)]["systems"]]
        res[f"goal_{name}"] = goal(jd, rows) if len(rows) == len(seeds) else {"met": False, "reason": "a seed has no feasible choice"}
    (root / "runs" / NS / "report" / "details.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "summary.json").write_text(json.dumps(strip_ids(res), indent=1, default=str)); (out / "report_tables.md").write_text(tables_md(strip_ids(res)))
    print(json.dumps({k: res[k] for k in ("stable", "goal_J1")}, indent=1, default=str)); print(f"wrote {out}")


# ----------------------------------------------------------------------------------------------- tables
def ci(c: dict, m: str) -> str:
    b = c["bootstrap"][m]
    return f"{100 * b['delta']:+.2f} [{100 * b['ci95'][0]:+.2f}, {100 * b['ci95'][1]:+.2f}]"


def tables_md(res: dict) -> str:
    choice = [{"seed": s, "system": n, "status": c["status"], "lambda": (c["params"] or {}).get("lambda"), "kappa": (c["params"] or {}).get("kappa"),
               "feasible settings": f"{c['n_feasible']} of {c['n_settings']}", "calibration segment F1@50": (c["calibration"] or {}).get("segment_f1_50_multi")}
              for s, d in res["choices"].items() for n, c in d.items()]
    systems, comps, diags = [], [], []
    for s, r in res["seeds"].items():
        for name, row in r["systems"].items():
            for h in HALVES:
                m = row[h]; systems.append({"seed": s, "system": name, "half": h, "segment F1@50": m["segment_f1_50_multi"], "boundary P": m["boundary_precision_multi"],
                                            "boundary R": m["boundary_recall_multi"], "native macro-F1": m["native_macro_f1"], "cuts per clip": m["native_internal_cuts_per_clip"],
                                            **{p: m[p] for p in PAIRS}, "A5C right": f"{m['native_counts']['A5C'][0]}/{m['native_counts']['A5C'][1]}"})
        for name, c in r["comparisons"].items():
            for h in HALVES:
                comps.append({"seed": s, "comparison": name, "half": h, **{f"d {m} (points)": ci(c[h], m) for m in METRICS}})
        for name, d in r["diagnostics"].items():
            for h in HALVES:
                mr = d.get("merged_recovery", {}).get(h); ex = d[h]["extra_changes"]
                diags.append({"seed": s, "system": name, "half": h, "three clips right": d["three_clips"]["right"] if h == "selection" else "",
                              "merged (B + ASRF) found": f"{mr['found']['count']} of {mr['merged_in_reference']['count']}" if mr else "",
                              "extra changes": f"{ex['all']['count']} ({ex['all']['recordings']} rec)", "native switches": d[h]["native_switches"]["internal_switches"],
                              "correct evidence lost": f"{d[h]['correct_visual_evidence_lost']['fragments']} fr / {d[h]['correct_visual_evidence_lost']['recordings']} rec"})
    refs = [{"system": k, "half": h, "segment F1@50": v[h]["segment_f1_50_multi"], "boundary R": v[h]["boundary_recall_multi"], "native macro-F1": v[h]["native_macro_f1"]}
            for k, v in res["references"].items() for h in HALVES]
    return "\n".join(["# Joint boundary decoding: tables\n", "## Choices (calibration only)\n", md_table(choice), "## Systems\n", md_table(systems),
                      "## Comparisons\n", md_table(comps), "## Diagnostics\n", md_table(diags), "## References\n", md_table(refs),
                      "## Stable rule and goal\n", "```\n" + json.dumps({k: res[k] for k in ("stable", "goal_J0", "goal_J1")}, indent=1, default=str) + "\n```\n"])


# ----------------------------------------------------------------------------------------------- dry run
def dry_run(cfg: dict, jd: dict, paths, device) -> None:
    """Training data only: one seed's frozen B + ASRF on a training subset, the grid, the choice and every diagnostic."""
    root = paths.repo_root; la = l1_asrf_cfg(jd, root); seed = seeds_of(jd)[0]; inp = b_inputs(cfg, jd, paths, seed); head, model = asrf_model(jd, la, paths, seed, device)
    clips = [v for c in range(5) for v in [x for x, y in inp.train.clips if y == c][:3]]
    natives = tuple(native_seq(v, inp.labels[v], inp.store.get(v, "orig")) for v in clips)
    from echo_routing.compose.banks import multi_bank
    from echo_routing.compose.stream_assembler import render
    from echo_routing.temporal.windowed import stream_plan
    idx = read_index(variant_dir(paths.cache_root, inp.cache_key, "orig")); recipes = multi_bank(idx, "train", cfg).recipes[:6]
    streams = tuple(render(r, inp.store.get) for r in recipes); ev = EvalSet("train-subset", natives, streams[:2], streams[2:])
    seqs = {h: [{"id": v, "kind": "native", "labels": s.labels.astype(int), "refs": [(v, int(f), 1.0) for f in inp.store.get(v, "orig").frame_idx]} for v, s in zip(clips, natives)]
            + [{"id": r.recipe_id, "kind": kind, "labels": st.labels.astype(int), "refs": [(x.video_id, int(x.frame_idx), float(x.gamma)) for x in stream_plan(r, inp.store.get)]}
               for r, st, kind in zip(recipes, streams, ("pairs", "pairs", "multi", "multi", "multi", "multi"))] for h in HALVES}
    dec = head.decode(model, ev.sequences(), device); probs = {s.seq_id: s.prob for s in ev.sequences()}; bprobs = {k: v["boundary_prob"] for k, v in dec.items()}
    recs = grid_records(cfg, grid_of(jd), probs, bprobs, ev); ch = choose_systems(recs, 0.980, 0.95)
    params = {n: (ch[n]["params"] or {"lambda": 1.0 if n == "J1" else 0.0, "kappa": 2.0}) for n in SYSTEMS}
    preds = {"B + ASRF": {h: {k: v["labels"] for k, v in dec.items()} for h in HALVES}, **{n: {h: decode_system(probs, bprobs, params[n]) for h in HALVES} for n in SYSTEMS}}
    sub = SimpleNamespace(halves={h: ev for h in HALVES}, store=inp.store); cuts = {s.recipe.recipe_id: candidates(bprobs[s.recipe.recipe_id]) for s in ev.multi}
    transfer = fragment_transfer({s.recipe.recipe_id: stream_units(s.recipe) for s in ev.multi}, lambda v, var: inp.store.get(v, var).prob)
    out, tabs, chg = {}, {}, {}
    for n, p in preds.items():
        rows, tabs[n] = evaluate_system(cfg, p, sub, seqs); diag, chg[n] = system_diagnostics(p["selection"], ev, cuts, transfer)
        out[n] = {"rows": sorted(rows), "diagnostics": sorted(diag), "three_clips": clip_reads(tabs[n]["selection"]["native"], clips[:3])["clips"]}
    out["merged_recovery"] = sorted(merged_recovery(chg["B + ASRF"], chg["J1"]))
    out["comparison"] = sorted(compare_one(tabs["J1"]["selection"], tabs["J0"]["selection"], preds["J1"]["selection"], preds["J0"]["selection"], seqs["selection"]))
    out["native_diff"] = sorted(native_diff(tabs["J1"]["selection"]["native"], tabs["B + ASRF"]["selection"]["native"]))
    out["grid_settings"] = len(recs); out["choice_status"] = {n: ch[n]["status"] for n in SYSTEMS}
    print(json.dumps({"dry_run": "ok (training data only, no metric)", **out}, indent=1, default=str))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--out", default=None)
    step = ap.add_mutually_exclusive_group(required=True)
    for s in ("extract", "choose", "evaluate", "dry-run"):
        step.add_argument(f"--{s}", action="store_true")
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); jd = cfg["joint_decoding"]; paths = load_paths()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    if a.extract or a.dry_run:
        device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
        (run_extract if a.extract else dry_run)(cfg, jd, paths, device)
    elif a.choose:
        run_choose(cfg, jd, paths, a.config)
    else:
        if not a.out:
            raise SystemExit("--out is required for --evaluate")
        run_evaluate(cfg, jd, paths, a)


if __name__ == "__main__":
    main()
