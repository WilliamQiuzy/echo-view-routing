#!/usr/bin/env python
"""Evaluate, freeze and report the L1 + ASRF round (plan Tasks 3-4; configs/experiments/l1_asrf_integration_v1.yaml).

1. Checks. The six runs are finished with the identity the trainer computes now; per seed, A and B share the batch plan,
   the class weights, pos_weight and the initial state. The reference systems reproduce from their frozen outputs, and
   L1 + B3 reproduces the fan224 round's rows from its stored probabilities and parameters.
2. Outputs. Each run's checkpoint chosen on calibration decodes both validation halves with the official
   refinement_with_boundary at boundary_th 0.5 (nothing appended).
3. Results, fixed before any score: per seed and half, segment F1@50, boundary precision/recall, native macro-F1 and
   per-class counts, A4C/A5C confusions, internal cuts; B - A (main) and B - (L1 + B3) with the paired bootstrap; native
   clips and stream reference units fixed or newly wrong, by recording and session; the goal for B.
   Diagnostics, computed once with the main table:
   - fragment transfer: every reference unit of every multi-fragment stream is split into its source fragments; each
     fragment and its whole source recording (same variant, so the same gamma) is classified by the frozen visual head
     (L1 for B and L1 + B3, E1 for A) with one rule at a time: majority of per-frame argmax, and mean probability.
     The transfer table counts fragments, distinct source recordings and sessions per category, also split by the
     system's outcome on the fragment's unit (added after the first evaluation pass, which counted fragments only);
   - unit outcomes (echo_routing.evaluate.failure_units): matched, or label / missed_cut / extra_cut / boundary_shift;
   - true view changes of the ASRF systems: found (both views right / a view wrong), merged, displaced, no candidate;
     extra cuts inside a clip or at a same-view join (the rules of scripts/pediatric_boundary_faults.py, re-implemented
     here so that no Pediatric entry is imported);
   - native clips and units that the frozen L1 head reads correctly but B + ASRF gets wrong, and the reverse.
4. Freezing (scripts/freeze_model.py) of the six models (selected and final checkpoints) with their plans, initial
   states, run manifests, feature manifests and the config.

--dry-run: training data only, untrained ASRF of seed 0 for both feature sets; every step runs, no metric is printed.

  python scripts/report_l1_asrf.py --config configs/experiments/l1_asrf_integration_v1.yaml \
      --out docs/results/2026-10-07_l1_asrf_integration
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg  # noqa: E402
from scripts.diagnose_center_recognition import native_bootstrap  # noqa: E402
from scripts.ev9v_complementarity import decode as b3_decode, md_table  # noqa: E402
from scripts.report_center_supervision import PAIRS, evaluate, native_changes, references, summary, validation  # noqa: E402
from scripts.report_fan224_visual import segment_changes, strip_ids  # noqa: E402
from scripts.report_model_development import BOOT_SEED, N_BOOT, paired_bootstrap  # noqa: E402
from scripts.train_l1_asrf import ARMS, NS, feature_manifest, finished_run, identity, make_head, make_plan, save_or_check_init, save_or_check_plan  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from echo_routing.compose.banks import multi_bank  # noqa: E402
from echo_routing.compose.recipe_builder import recipe_to_dict  # noqa: E402
from echo_routing.config import gpu_memory_fraction, load_config, load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of, stream_units, unit_outcomes  # noqa: E402
from echo_routing.evaluate.metrics_boundary import match_boundaries  # noqa: E402
from echo_routing.features.cache import read_index, variant_dir  # noqa: E402
from echo_routing.temporal.dev_data import stream_seq  # noqa: E402
from echo_routing.temporal.dev_run import HALVES, Inputs, digest, load_json  # noqa: E402
from echo_routing.temporal.segments import boundaries_from_labels  # noqa: E402

TOL, WIDE, NUM_CLASSES = 5, 15, 5
FAULTS = ("found, both views right", "found, a view wrong", "merged", "displaced", "no candidate")
TRANSFER = ("recording right, fragment right", "recording right, fragment wrong", "recording wrong, fragment right", "recording wrong, fragment wrong")


# ----------------------------------------------------------------------------------------------- pure helpers
def majority(labels: np.ndarray) -> int:
    return int(np.bincount(np.asarray(labels, int), minlength=NUM_CLASSES).argmax())


def classify_rule(prob: np.ndarray, rule: str) -> int:
    prob = np.asarray(prob, float)
    if rule == "majority":
        return majority(prob.argmax(1))
    if rule == "mean":
        return int(prob.mean(0).argmax())
    raise ValueError(rule)


def fragment_transfer(units_by_stream: dict, prob_of, rules=("majority", "mean")) -> list[dict]:
    """One row per source fragment of every unit: the visual class of the fragment and of its whole source recording
    (same variant, so the same gamma) under each rule. prob_of(video, variant) -> (n, 5) probabilities."""
    rows = []
    for sid, units in units_by_stream.items():
        for u in units:
            for k, (vid, variant, a, b) in enumerate(u.sources):
                vp = prob_of(vid, variant); rec = {"stream": sid, "unit": u.index, "fragment": k, "label": u.label, "video": vid, "variant": variant,
                                                   "length": b - a, "unit_fragments": len(u.sources)}
                for r in rules:
                    rec[f"{r}_fragment"] = classify_rule(vp[a:b], r); rec[f"{r}_recording"] = classify_rule(vp, r)
                    rec[f"{r}_category"] = transfer_category(rec[f"{r}_recording"] == u.label, rec[f"{r}_fragment"] == u.label)
                rows.append(rec)
    return rows


def transfer_category(recording_right: bool, fragment_right: bool) -> str:
    return TRANSFER[(0 if recording_right else 2) + (0 if fragment_right else 1)]


def transfer_table(transfer_rows: list[dict], unit_rows: list[dict], rules=("majority", "mean")) -> dict:
    """Per rule and transfer category: fragments, distinct source recordings and sessions, and the same counts split by
    the system's outcome on the fragment's unit (matched, or the unit's cause). Fragments repeat recordings, so the
    recording and session counts are the independent ones."""
    outcome = {(u["stream"], u["unit"]): u["cause"] for u in unit_rows}
    out = {}
    for rule in rules:
        cats: dict = {}
        for r in transfer_rows:
            c = cats.setdefault(r[f"{rule}_category"], {"fragments": 0, "recordings": set(), "sessions": set(), "by_outcome": {}})
            c["fragments"] += 1; c["recordings"].add(r["video"]); c["sessions"].add(session_of(r["video"]))
            o = c["by_outcome"].setdefault(outcome[(r["stream"], r["unit"])], {"fragments": 0, "recordings": set(), "sessions": set()})
            o["fragments"] += 1; o["recordings"].add(r["video"]); o["sessions"].add(session_of(r["video"]))
        out[rule] = {k: {"fragments": v["fragments"], "recordings": len(v["recordings"]), "sessions": len(v["sessions"]),
                         "by_outcome": {o: {"fragments": w["fragments"], "recordings": len(w["recordings"]), "sessions": len(w["sessions"])}
                                        for o, w in sorted(v["by_outcome"].items())}}
                     for k, v in sorted(cats.items())}
    return out


def classify_changes(final: np.ndarray, cuts: list[int], labels: np.ndarray, joins: np.ndarray, semantic: np.ndarray) -> tuple[list[dict], dict]:
    """Every true view change of one stream: found (both views right / a view wrong), merged (a candidate within TOL
    but both sides took one class), displaced (an unmatched final change TOL < d <= WIDE away), no candidate. Extra
    final changes are split into inside-a-clip and at-a-same-view-join (the rules of pediatric_boundary_faults.py)."""
    final = np.asarray(final, int); sem = np.asarray(semantic, int); joins = np.asarray(joins, int); pred_b = boundaries_from_labels(final)
    pairs = match_boundaries(pred_b, sem, TOL); hit = {int(sem[j]) for _, j in pairs}; used = {int(pred_b[i]) for i, _ in pairs}
    starts, ends = np.r_[0, joins], np.r_[joins, len(final)]
    maj = [majority(final[a:b]) for a, b in zip(starts, ends)]; truth = [int(labels[a]) for a in starts]
    pos = {int(j): k for k, j in enumerate(joins, start=1)}; cuts = np.asarray(cuts, int); free = [int(b) for b in pred_b if int(b) not in used]
    rows = []
    for r in sem.tolist():
        k = pos[r]
        if r in hit:
            cat = FAULTS[0] if maj[k - 1] == truth[k - 1] and maj[k] == truth[k] else FAULTS[1]
        elif cuts.size and np.abs(cuts - r).min() <= TOL:
            cat = "merged"
        elif any(TOL < abs(b - r) <= WIDE for b in free):
            cat = "displaced"
        else:
            cat = "no candidate"
        rows.append({"r": r, "cat": cat, "views": (truth[k - 1], truth[k])})
    extra = [b for b in free if sem.size == 0 or np.abs(sem - b).min() > TOL]
    inside = sum(1 for b in extra if joins.size == 0 or np.abs(joins - b).min() > TOL)
    return rows, {"extra_inside_clip": inside, "extra_at_same_view_join": len(extra) - inside}


def goal(la: dict, rows: list[dict]) -> dict:
    g = la["goal"]; floors = {"selection": 0.973, "calibration": 0.980}
    mean = {h: float(np.mean([r[h]["segment_f1_50_multi"] for r in rows])) for h in HALVES}
    per_seed = [{h: {"feasible": r["chosen"]["status"] == "feasible", "recall": r[h]["boundary_recall_multi"], "native": r[h]["native_macro_f1"]}
                 for h in HALVES} for r in rows]
    ok = all(p[h]["feasible"] and (p[h]["recall"] or 0) >= 0.95 and p[h]["native"] >= floors[h] for p in per_seed for h in HALVES)
    return {"mean_segment_f1": mean, "required": {"selection": g["selection_min"], "calibration": g["calibration_min"]},
            "margin": {h: mean[h] - float(g[f"{h}_min"]) for h in HALVES}, "per_seed": per_seed,
            "met": bool(ok and all(mean[h] >= float(g[f"{h}_min"]) for h in HALVES)),
            "sensitivity_calibration": mean["calibration"] - (float(g["calibration_sensitivity"]) + 0.01)}


def compare_one(ta: dict, tb: dict, pa: dict, pb: dict, seqs_h: list) -> dict:
    ch = native_changes(ta["native"], tb["native"])
    return {"bootstrap": paired_bootstrap(ta, tb), "native_session": native_bootstrap(ta["native"], tb["native"], N_BOOT, BOOT_SEED),
            "native_changes": {k: ({"fixed": len(v["fixed"]), "new": len(v["new"]), "fixed_sessions": v["fixed_sessions"], "new_sessions": v["new_sessions"]}
                                   if isinstance(v.get("fixed"), list) else v) for k, v in ch.items()},
            "segment_changes": segment_changes(pa, pb, seqs_h), "_ids": {k: {"fixed": v["fixed"], "new": v["new"]} for k, v in ch.items() if isinstance(v.get("fixed"), list)}}


def counts_by(rows: list[dict], key: str, by: str | None = None) -> dict:
    if by is None:
        return dict(Counter(r[key] for r in rows))
    out = {}
    for r in rows:
        out.setdefault(str(r[by]), Counter())[r[key]] += 1
    return {k: dict(v) for k, v in out.items()}


# ----------------------------------------------------------------------------------------------- systems
def asrf_outputs(head, model, inputs: Inputs, device) -> dict:
    """Per half: decoded labels for every sequence and, for multi streams, the candidate cuts before the class vote."""
    from scripts.round5_reachability import asrf_decode
    out = {}
    for h in HALVES:
        ev = inputs.halves[h]; dec = head.decode(model, ev.sequences(), device)
        cuts = {s.recipe.recipe_id: asrf_decode(head, model, s.feat.astype(np.float16).astype(np.float32), device) for s in ev.multi}
        for sid, (final, _) in cuts.items():
            if not np.array_equal(final, dec[sid]["labels"]):
                raise SystemExit(f"{sid}: the candidate decode differs from the official decode")
        out[h] = {"preds": {k: np.asarray(v["labels"], int) for k, v in dec.items()}, "cuts": {k: v[1] for k, v in cuts.items()}}
    return out


def evaluate_system(cfg: dict, preds_by_half: dict, inputs: Inputs, seqs: dict) -> tuple[dict, dict]:
    rows, tables = {}, {}
    for h in HALVES:
        m, t = evaluate(cfg, preds_by_half[h], inputs.halves[h])
        rows[h] = {**summary(m), "native_counts": {c: [int(np.asarray(m["native_confusion"])[k, k]), int(np.asarray(m["native_confusion"])[k].sum())]
                                                  for k, c in enumerate(("PLAX", "PSAX", "A4C", "A5C", "SC4C"))},
                   "stream_sample_error": float(sum(int((np.asarray(preds_by_half[h][q["id"]]) != q["labels"]).sum()) for q in seqs[h] if q["kind"] == "multi")
                                                / max(1, sum(len(q["labels"]) for q in seqs[h] if q["kind"] == "multi")))}
        tables[h] = t
    return rows, tables


def unit_table(preds: dict, inputs_h, transfer_rows: list[dict]) -> tuple[list[dict], dict]:
    """Unit outcomes of one system on one half, joined with the visual status of the unit's fragments."""
    status = {}
    for r in transfer_rows:
        key = (r["stream"], r["unit"]); s = status.setdefault(key, {"majority": [], "mean": []})
        for rule in ("majority", "mean"):
            s[rule].append(r[f"{rule}_category"])
    rows = []
    for s in inputs_h.multi:
        units = stream_units(s.recipe); out, _ = unit_outcomes(preds[s.recipe.recipe_id], units)
        for u, o in zip(units, out):
            st = status[(s.recipe.recipe_id, u.index)]
            rows.append({"stream": s.recipe.recipe_id, "unit": u.index, "label": u.label, "fragments": len(u.sources), "matched": o["matched"],
                         "cause": o["cause"] or "matched", **{f"visual_{rule}": ("all fragments right" if all(c.endswith("fragment right") for c in st[rule])
                                                                                 else "some fragment wrong") for rule in ("majority", "mean")},
                         "sources": [v for v, *_ in u.sources]})
    return rows, {"by_cause": counts_by(rows, "cause"), "by_cause_and_visual_majority": counts_by(rows, "cause", "visual_majority"),
                  "by_cause_and_visual_mean": counts_by(rows, "cause", "visual_mean")}


def fault_table(preds: dict, cuts: dict, inputs_h) -> dict:
    rows, extra = [], Counter()
    for s in inputs_h.multi:
        r, e = classify_changes(preds[s.recipe.recipe_id], cuts[s.recipe.recipe_id], s.labels, s.joins, s.semantic_boundaries)
        rows += r; extra.update(e)
    return {"changes": len(rows), "by_category": dict(Counter(r["cat"] for r in rows)), **dict(extra)}


def visual_vs_system(visual_native: dict, system_native, unit_rows: list[dict]) -> dict:
    """Native clips and stream units that the frozen L1 head reads correctly while B + ASRF gets them wrong, and the
    reverse. visual_native: clip -> the head's majority vote is right; unit_rows: unit_table rows, which carry both the
    visual status of each unit's fragments (majority rule) and the system's outcome."""
    sys_right = {r.video_id: int(r.majority) == int(r.label) for r in system_native.itertuples()}
    broken = [v for v, ok in visual_native.items() if ok and not sys_right[v]]; fixed = [v for v, ok in visual_native.items() if not ok and sys_right[v]]
    unit_broken = [r for r in unit_rows if r["visual_majority"] == "all fragments right" and not r["matched"]]
    unit_fixed = [r for r in unit_rows if r["visual_majority"] != "all fragments right" and r["matched"]]
    return {"native_visual_right_system_wrong": {"clips": len(broken), "sessions": len({session_of(v) for v in broken})},
            "native_visual_wrong_system_right": {"clips": len(fixed), "sessions": len({session_of(v) for v in fixed})},
            "units_visual_right_system_unmatched": {"units": len(unit_broken), "causes": dict(Counter(r["cause"] for r in unit_broken)),
                                                   "source_recordings": len({v for r in unit_broken for v in r["sources"]})},
            "units_visual_wrong_system_matched": {"units": len(unit_fixed), "source_recordings": len({v for r in unit_fixed for v in r["sources"]})},
            "_ids": {"native_broken": broken, "native_fixed": fixed}}


# ----------------------------------------------------------------------------------------------- checks and loading
def expected_identity(cfg: dict, la: dict, paths, arm: str, seed: int) -> str:
    """The trainer's identity, computed without rendering streams (no training or decoding)."""
    root = paths.repo_root; fman_path = feature_manifest(la, arm, seed, root)
    proto, proto_sha = load_json(root / la["inputs"]["protocol"]["path"]); fman, fman_sha = load_json(fman_path); key = fman["encoder"]["cache_key"]
    idx = read_index(variant_dir(paths.cache_root, key, "orig")); bank = multi_bank(idx, "train", cfg)
    data = SimpleNamespace(proto_sha=proto_sha, fman_sha=fman_sha, cache_key=key, bank_digest=digest([recipe_to_dict(r) for r in bank.recipes]),
                           index_sha={v: sha256_file(variant_dir(paths.cache_root, key, v) / "index.csv") for v in ("orig", "gamma090")})
    plan = root / "runs" / NS / "plans" / f"plan_s{seed}.json"; init = root / "runs" / NS / "inits" / f"asrf_init_s{seed}.pt"
    return identity(cfg, arm, seed, data, sha256_file(plan), sha256_file(init), root)


def verify(cfg: dict, la: dict, paths) -> dict:
    root = paths.repo_root; runs = {}
    for seed in [int(s) for s in la["seeds"]]:
        for arm in ARMS:
            key = f"{arm}_s{seed}"; man_path = root / "runs" / NS / "train" / key / "run_manifest.json"
            ident = expected_identity(cfg, la, paths, arm, seed)
            if finished_run(man_path, ident) is not True:
                raise SystemExit(f"{key}: not a finished run of the current identity")
            runs[key] = json.loads(man_path.read_text())
        a, b = runs[f"A_s{seed}"], runs[f"B_s{seed}"]
        if not (a["plan_sha256"] == b["plan_sha256"] and a["init_sha256"] == b["init_sha256"] and a["class_weight"] == b["class_weight"]
                and a["pos_weight"] == b["pos_weight"] and a["bank_digest"] == b["bank_digest"]):
            raise SystemExit(f"seed {seed}: A and B are not paired")
        if a["cache_key"] == b["cache_key"]:
            raise SystemExit(f"seed {seed}: A and B read the same features")
    return runs


def load_model(head, man: dict, which: str, device):
    model = head.build().to(device); model.load_state_dict(torch.load(man[f"{which}_checkpoint"], map_location=device, weights_only=True)); model.eval()
    return model


def l1_b3(cfg: dict, la: dict, paths, seed: int, inputs: Inputs) -> tuple[dict, dict]:
    """L1 + B3 of the fan224 round from its stored per-sequence probabilities and its calibration-chosen parameters."""
    root = paths.repo_root; fan = json.loads((root / "runs/fan224_visual_adaptation_v1/report/details.json").read_text())
    params = fan["models"][f"L1_s{seed}"]["b3_choice"]["params"]; probs_all = np.load(root / f"runs/fan224_visual_adaptation_v1/report/probs_L1_s{seed}.npz")
    preds, check = {}, {}
    for h in HALVES:
        probs = {k.split("|", 1)[1]: probs_all[k] for k in probs_all.files if k.startswith(f"{h}|")}
        preds[h] = b3_decode(params, probs)
        m, _ = evaluate(cfg, preds[h], inputs.halves[h]); stored = fan["models"][f"L1_s{seed}"][f"b3|{h}"]
        for k in ("segment_f1_50_multi", "native_macro_f1", "boundary_recall_multi", "boundary_precision_multi", "native_internal_cuts_per_clip"):
            if abs(m[k] - stored[k]) > 1e-12:
                raise SystemExit(f"L1_s{seed} + B3 {h}: {k} {m[k]} does not reproduce {stored[k]}")
        check[h] = "reproduced"
    return preds, {"params": params, **check}


# ----------------------------------------------------------------------------------------------- main
def freeze(paths, runs: dict, config: str) -> dict:
    out = {}
    for key, man in runs.items():
        files = [man["selected_checkpoint"], man["final_checkpoint"], str(paths.repo_root / "runs" / NS / "train" / key / "run_manifest.json"), man["plan"], man["init"],
                 str(paths.repo_root / config)]
        if man["arm"] == "B":
            files.append(str(paths.repo_root / "runs" / NS / "features" / f"feature_manifest_L1_s{man['seed']}.json"))
        dest = paths.repo_root / "models_frozen" / NS / key
        if not dest.exists():
            subprocess.run([sys.executable, "scripts/freeze_model.py", NS, key, *sum((["--file", f] for f in files), []), "--meta", f"identity={man['identity']}",
                            "--meta", f"arm={man['arm']}", "--meta", f"seed={man['seed']}", "--meta", f"chosen_epoch={man['chosen']['epoch']}",
                            "--notes", "L1 + ASRF round (2026-10-07); checkpoint chosen on calibration."], cwd=paths.repo_root, check=True, capture_output=True)
        out[key] = str(dest)
    return out


def write_outputs(res: dict, details: Path, out: Path) -> None:
    (details / "details.json").write_text(json.dumps(res, indent=1, default=str)); out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(strip_ids(res), indent=1, default=str)); (out / "report_tables.md").write_text(tables_md(strip_ids(res)))


def run_evaluate(a, cfg: dict, la: dict, paths, device) -> None:
    root = paths.repo_root; runs = verify(cfg, la, paths); print("six runs verified and paired", flush=True)
    _, seqs = validation(cfg, {"inputs": {"protocol": la["inputs"]["protocol"], "feature_manifest": {"path": la["inputs"]["e1_feature_manifest"]["path"]}}}, paths)
    details = root / a.details; details.mkdir(parents=True, exist_ok=True); out = root / a.out
    res = {"systems": {}, "diagnostics": {}, "comparisons": {}, "runs": {}, "timing": {}}
    inputs = {"A": Inputs(cfg, root / la["inputs"]["protocol"]["path"], root / la["inputs"]["e1_feature_manifest"]["path"], paths.cache_root)}
    tables, preds = {}, {}
    for seed in [int(s) for s in la["seeds"]]:
        inputs[f"B_s{seed}"] = Inputs(cfg, root / la["inputs"]["protocol"]["path"], feature_manifest(la, "B", seed, root), paths.cache_root)
        for arm in ARMS:
            t0 = time.time(); key = f"{arm}_s{seed}"; man = runs[key]; inp = inputs["A" if arm == "A" else f"B_s{seed}"]
            plan = json.loads(Path(man["plan"]).read_text()); head = make_head(la, paths, plan, device); model = load_model(head, man, "selected", device)
            outs = asrf_outputs(head, model, inp, device); rows, tabs = evaluate_system(cfg, {h: outs[h]["preds"] for h in HALVES}, inp, seqs)
            prob_of = (lambda v, var, s=inp.store: s.get(v, var).prob)
            transfer = {h: fragment_transfer({s.recipe.recipe_id: stream_units(s.recipe) for s in inp.halves[h].multi}, prob_of) for h in HALVES}
            units = {h: unit_table(outs[h]["preds"], inp.halves[h], transfer[h]) for h in HALVES}
            faults = {h: fault_table(outs[h]["preds"], outs[h]["cuts"], inp.halves[h]) for h in HALVES}
            res["systems"][f"{key} + ASRF"] = {**rows, "chosen": man["chosen"]}; tables[key], preds[key] = tabs, {h: outs[h]["preds"] for h in HALVES}
            res["diagnostics"][f"{key} + ASRF"] = {"fragment_transfer": {h: {r: counts_by(transfer[h], f"{r}_category") for r in ("majority", "mean")} for h in HALVES},
                                                   "transfer_table": {h: transfer_table(transfer[h], units[h][0]) for h in HALVES},
                                                   "units": {h: units[h][1] for h in HALVES}, "faults": faults}
            if arm == "B":
                vis = {h: {s.seq_id: classify_rule(s.prob, "majority") == int(s.labels[0]) for s in inp.halves[h].natives} for h in HALVES}
                res["diagnostics"][f"{key} + ASRF"]["visual_vs_system"] = {h: visual_vs_system(vis[h], tabs[h]["native"], units[h][0]) for h in HALVES}
            np.savez_compressed(details / f"preds_{key}.npz", **{f"{h}|{k}": v.astype(np.int8) for h in HALVES for k, v in outs[h]["preds"].items()})
            res["timing"][key] = {"seconds": time.time() - t0}
            print(f"{key} + ASRF: chosen epoch {man['chosen']['epoch']} ({man['chosen']['status']}) | sel {rows['selection']['segment_f1_50_multi']:.4f} "
                  f"cal {rows['calibration']['segment_f1_50_multi']:.4f} | native {rows['selection']['native_macro_f1']:.4f} / {rows['calibration']['native_macro_f1']:.4f}", flush=True)
            del model; torch.cuda.empty_cache()
        p_l1, chk = l1_b3(cfg, la, paths, seed, inputs[f"B_s{seed}"]); rows, tabs = evaluate_system(cfg, p_l1, inputs[f"B_s{seed}"], seqs)
        res["systems"][f"L1_s{seed} + B3"] = {**rows, "reproduction": chk}; tables[f"L1_s{seed}"], preds[f"L1_s{seed}"] = tabs, p_l1
        prob_of = (lambda v, var, s=inputs[f"B_s{seed}"].store: s.get(v, var).prob)
        tr = {h: fragment_transfer({s.recipe.recipe_id: stream_units(s.recipe) for s in inputs[f"B_s{seed}"].halves[h].multi}, prob_of) for h in HALVES}
        ul = {h: unit_table(p_l1[h], inputs[f"B_s{seed}"].halves[h], tr[h]) for h in HALVES}
        res["diagnostics"][f"L1_s{seed} + B3"] = {"units": {h: ul[h][1] for h in HALVES}, "transfer_table": {h: transfer_table(tr[h], ul[h][0]) for h in HALVES}}
        for name, (x, y) in {"B - A": (f"B_s{seed}", f"A_s{seed}"), "B - (L1 + B3)": (f"B_s{seed}", f"L1_s{seed}")}.items():
            res["comparisons"][f"s{seed}|{name}"] = {h: compare_one(tables[x][h], tables[y][h], preds[x][h], preds[y][h], seqs[h]) for h in HALVES}
    res["goal_B"] = goal(la, [res["systems"][f"B_s{s} + ASRF"] for s in la["seeds"]]); res["goal_A"] = goal(la, [res["systems"][f"A_s{s} + ASRF"] for s in la["seeds"]])
    res["runs"] = {k: {x: v[x] for x in ("identity", "arm", "seed", "epochs", "updates", "chosen", "selected_sha256", "final_sha256", "plan_sha256", "init_sha256",
                                         "cache_key", "asrf_commit", "class_weight", "pos_weight")} | {"records": v["records"]} for k, v in runs.items()}
    res["references"] = "pending"; res["frozen"] = "pending"; write_outputs(res, details, out)
    refs = references({**cfg, "center_supervision": {"inputs": {"protocol": la["inputs"]["protocol"]}}}, paths, inputs["A"], seqs)
    res["references"] = {k: {h: v[h]["metrics"] for h in HALVES} for k, v in refs.items()}; write_outputs(res, details, out); print("references reproduced", flush=True)
    res["frozen"] = {} if a.no_freeze else freeze(paths, runs, a.config); write_outputs(res, details, out)
    print(json.dumps(res["goal_B"], indent=1, default=str)); print(f"wrote {out}")


def dry_run(cfg: dict, la: dict, paths, device) -> None:
    """Training data only: a training subset as both 'halves', the untrained ASRF of seed 0 on A and B features."""
    from echo_routing.compose.stream_assembler import render
    from echo_routing.temporal.dev_data import EvalSet, native_seq
    from echo_routing.temporal.windowed import stream_plan
    root = paths.repo_root; out = {}
    for arm in ARMS:
        inp = Inputs(cfg, root / la["inputs"]["protocol"]["path"], feature_manifest(la, arm, 0, root), paths.cache_root)
        clips = [v for c in range(5) for v in [x for x, y in inp.train.clips if y == c][:3]]
        idx = read_index(variant_dir(paths.cache_root, inp.cache_key, "orig")); recipes = multi_bank(idx, "train", cfg).recipes[:4]
        natives = [native_seq(v, inp.labels[v], inp.store.get(v, "orig")) for v in clips]; streams = [render(r, inp.store.get) for r in recipes]
        ev = EvalSet("train-subset", tuple(natives), tuple(streams[:2]), tuple(streams[2:]))
        sub = SimpleNamespace(halves={h: ev for h in HALVES}, store=inp.store)
        seqs_h = [{"id": v, "kind": "native", "labels": s.labels.astype(int), "refs": [(v, int(f), 1.0) for f in inp.store.get(v, "orig").frame_idx]} for v, s in zip(clips, natives)]
        seqs_h += [{"id": r.recipe_id, "kind": kind, "labels": st.labels.astype(int), "refs": [(x.video_id, int(x.frame_idx), float(x.gamma)) for x in stream_plan(r, inp.store.get)]}
                   for r, st, kind in zip(recipes, streams, ("pairs", "pairs", "multi", "multi"))]
        seqs = {h: seqs_h for h in HALVES}
        plan = make_plan(SimpleNamespace(train=inp.train), la["training"], 0); head = make_head(la, paths, plan, device)
        init, _ = save_or_check_init(head, 0, root); model = head.build().to(device); model.load_state_dict(torch.load(init, map_location=device, weights_only=True)); model.eval()
        outs = asrf_outputs(head, model, sub, device); rows, tabs = evaluate_system(cfg, {h: outs[h]["preds"] for h in HALVES}, sub, seqs)
        tr = fragment_transfer({s.recipe.recipe_id: stream_units(s.recipe) for s in ev.multi}, lambda v, var: inp.store.get(v, var).prob)
        ut = unit_table(outs["selection"]["preds"], ev, tr); ft = fault_table(outs["selection"]["preds"], outs["selection"]["cuts"], ev)
        out[arm] = {"rows": sorted(rows), "transfer_rows": len(tr), "units": len(ut[0]), "faults_changes": ft["changes"], "tables": sorted(tabs["selection"])}
        out[f"{arm}_tables"] = tabs; out[f"{arm}_preds"] = {h: outs[h]["preds"] for h in HALVES}; out["seqs"] = seqs
    c = compare_one(out["B_tables"]["selection"], out["A_tables"]["selection"], out["B_preds"]["selection"], out["A_preds"]["selection"], out["seqs"]["selection"])
    print(json.dumps({"dry_run": "ok (training data only, no metric)", "A": out["A"], "B": out["B"], "compare_keys": sorted(c)}, indent=1, default=str))


def tables_md(res: dict) -> str:
    rows = []
    for k, r in res["systems"].items():
        for h in HALVES:
            m = r[h]
            rows.append({"system": k, "half": h, "chosen": (f"{r['chosen']['epoch']} ({r['chosen']['status']})" if "chosen" in r else ""), "segment F1@50": m["segment_f1_50_multi"],
                         "boundary P": m["boundary_precision_multi"], "boundary R": m["boundary_recall_multi"], "native macro-F1": m["native_macro_f1"],
                         "cuts per clip": m["native_internal_cuts_per_clip"], **{p: m[p] for p in PAIRS}})
    comp = []
    for k, v in res["comparisons"].items():
        for h, c in v.items():
            b = c["bootstrap"]
            comp.append({"comparison": k, "half": h, **{f"d {m}": f"{b[m]['delta']:+.4f} [{b[m]['ci95'][0]:+.4f}, {b[m]['ci95'][1]:+.4f}]"
                                                         for m in ("multi segment F1@50", "multi boundary recall", "native macro-F1")},
                         "native fixed/new": f"{c['native_changes']['all clips']['fixed']}/{c['native_changes']['all clips']['new']}"})
    refs = res["references"] if isinstance(res["references"], dict) else {}
    ref = [{"system": k, "half": h, "segment F1@50": v[h]["segment_f1_50_multi"], "boundary R": v[h]["boundary_recall_multi"], "native macro-F1": v[h]["native_macro_f1"]}
           for k, v in refs.items() for h in HALVES] or [{"system": "references pending"}]
    return "\n".join(["# L1 + ASRF round: tables\n", "## Systems\n", md_table(rows), "## Comparisons\n", md_table(comp), "## References\n", md_table(ref),
                      "## Goal (B + ASRF)\n", "```\n" + json.dumps(res["goal_B"], indent=1, default=str) + "\n```\n"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--out", default=None); ap.add_argument("--details", default=f"runs/{NS}/report")
    ap.add_argument("--no-freeze", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); la = cfg["l1_asrf"]; paths = load_paths()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    if a.dry_run:
        dry_run(cfg, la, paths, device); return
    if not a.out:
        raise SystemExit("--out is required for the evaluation")
    run_evaluate(a, cfg, la, paths, device)


if __name__ == "__main__":
    main()
