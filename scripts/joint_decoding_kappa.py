#!/usr/bin/env python
"""Post-hoc switch-cost range check of the joint boundary decoding round (configs/experiments/joint_boundary_decoding_kappa_v1.yaml).

Not part of the original pre-registration: the project lead asked for it after every J0/J1 choice of
joint_boundary_decoding_v1 sat at the largest switch cost of its grid (kappa = 8). Only the kappa range changes
(0 ... 128). The class and boundary probabilities exported by the original round are reused without re-extraction, and
every other rule is inherited from the original configuration.

  --choose    Calibration half only: the 5 x 13 grid per seed with its calibration curves (segment F1@50, boundary recall,
              native macro-F1, extra changes, native switches, matched units, predicted segments, and the changes that
              B + ASRF merged which the setting finds). The 25 settings of the original grid must reproduce the original
              calibration metrics exactly. J0 and J1 are chosen with the original rule. Writes choices.json.
  --plot      The calibration curves from a choices.json (drawing only).
  --evaluate  Both halves once with the committed choices: J0, J1, the original kappa <= 8 choices, L1 + B3, B + ASRF and the
              references; contrasts, gains and losses against the kappa <= 8 choices, the goal and the fixed decisions.
  --dry-run   The evaluation code on the calibration half alone (in both slots), with placeholder settings.

  python scripts/joint_decoding_kappa.py --config configs/experiments/joint_boundary_decoding_kappa_v1.yaml --choose
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg  # noqa: E402
from scripts.diagnose_center_recognition import native_bootstrap  # noqa: E402
from scripts.ev9v_complementarity import decode as b3_decode, md_table  # noqa: E402
from scripts.joint_boundary_decoding import (SYSTEMS, b_inputs, calibration_set, candidates, change_records, check_inputs,  # noqa: E402
                                             choose_systems, clip_reads, decode_system, dedup, extraction, grid_of, l1_asrf_cfg,
                                             load_half, merged_recovery, native_diff, seeds_of, stable, system_diagnostics,
                                             three_clips)
from scripts.report_center_supervision import FLAT, PAIRS, evaluate, references, validation  # noqa: E402
from scripts.report_fan224_visual import strip_ids  # noqa: E402
from scripts.report_l1_asrf import compare_one, evaluate_system, fragment_transfer, goal, l1_b3  # noqa: E402
from scripts.report_model_development import BOOT_SEED, N_BOOT, paired_bootstrap  # noqa: E402

import numpy as np  # noqa: E402

from echo_routing.config import load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import stream_units  # noqa: E402
from echo_routing.temporal.dev_data import FeatureStore  # noqa: E402
from echo_routing.temporal.dev_run import HALVES, load_json  # noqa: E402
from echo_routing.temporal.joint_decoding import joint_viterbi_grid  # noqa: E402
from echo_routing.temporal.segments import boundaries_from_labels  # noqa: E402
from echo_routing.temporal.windowed import stream_plan  # noqa: E402

NS = "joint_boundary_decoding_kappa_v1"
OLD = {"J0": "J0 (kappa <= 8)", "J1": "J1 (kappa <= 8)"}
CONTRASTS = (("J1", "J0"), ("J1", "L1 + B3"), ("J0", "L1 + B3"), ("J1", "B + ASRF"), ("J0", OLD["J0"]), ("J1", OLD["J1"]))
STRONGEST = ("E0 + B3", "B7 + B3")
CURVES = ("segment_f1_50_multi", "boundary_recall_multi", "native_macro_f1", "extra_changes", "merged_found")


# ----------------------------------------------------------------------------------------------- pure helpers
def reproduce_original_grid(recs: list[dict], original: list[dict], seed) -> int:
    """Every setting of the original grid must give exactly its original calibration metrics; returns the count."""
    by = {(r["params"]["lambda"], r["params"]["kappa"]): r for r in recs}
    for o in original:
        r = by[(o["params"]["lambda"], o["params"]["kappa"])]
        if any(r[k] != o[k] for k in FLAT):
            raise SystemExit(f"seed {seed}: setting {o['params']} does not reproduce the original calibration metrics")
    return len(original)


def upper_limit(choice: dict, kappas: list[float]) -> dict:
    return {n: bool(c["params"] is not None and c["params"]["kappa"] == max(kappas)) for n, c in choice.items()}


def gains_losses(ch_new: list[dict], ch_old: list[dict], ch_asrf: list[dict]) -> dict:
    """True view changes of the kappa <= 8 counterpart (old) against the new choice: found by old and still found (kept),
    found by old and now missed (lost), found only now; the same for the changes old recovered after B + ASRF merged them."""
    key = lambda r: (r["stream"], r["r"])  # noqa: E731
    by_new = {key(r): r for r in ch_new}; merged = {key(r) for r in ch_asrf if r["cat"] == "merged"}
    found_old = [r for r in ch_old if r["cat"].startswith("found")]; old_keys = {key(r) for r in found_old}
    kept = [r for r in found_old if by_new[key(r)]["cat"].startswith("found")]
    lost = [by_new[key(r)] for r in found_old if not by_new[key(r)]["cat"].startswith("found")]
    newly = [r for r in ch_new if r["cat"].startswith("found") and key(r) not in old_keys]
    rec_old = [r for r in found_old if key(r) in merged]; rec_kept = [r for r in rec_old if by_new[key(r)]["cat"].startswith("found")]
    return {"found_by_old": dedup(found_old), "kept": dedup(kept), "lost": dedup(lost), "lost_categories": {c: sum(r["cat"] == c for r in lost) for c in {r["cat"] for r in lost}},
            "found_only_now": dedup(newly), "recovered_by_old": dedup(rec_old), "recovered_kept": dedup(rec_kept)}


def decisions(stab: dict, goals: dict) -> dict:
    """The configuration's decision table, applied to the stable-rule outcomes."""
    j0 = stab["J0 - (L1 + B3)"]["stable"]; j1 = stab["J1 - (L1 + B3)"]["stable"]; extra = stab["J1 - (J0)"]["stable"]
    if extra:
        case = "j1_extra"
    elif j0:
        case = "j0_only"
    elif j1:
        case = "j1_improves_without_extra_gain_over_j0"
    else:
        case = "neither_improves"
    return {"case": case, "j0_improves": j0, "j1_improves": j1, "j1_stably_better_than_j0": extra,
            "goal_met": {n: bool(goals[n].get("met")) for n in goals}}


def seqs_of(ev, store) -> list[dict]:
    """Per-sample source references of one half's sequences (as report_center_supervision.validation builds them)."""
    out = [{"id": s.seq_id, "kind": "native", "labels": s.labels.astype(int), "refs": [(s.seq_id, int(f), 1.0) for f in store.get(s.seq_id, "orig").frame_idx]}
           for s in ev.natives]
    for kind, coll in (("pairs", ev.pairs), ("multi", ev.multi)):
        out += [{"id": st.recipe.recipe_id, "kind": kind, "labels": st.labels.astype(int),
                 "refs": [(r.video_id, int(r.frame_idx), float(r.gamma)) for r in stream_plan(st.recipe, store.get)]} for st in coll]
    return out


# ----------------------------------------------------------------------------------------------- inputs
def check_original(ks: dict, root: Path) -> dict:
    o = ks["original"]; out = {}
    for key in ("config", "extract", "choices", "summary"):
        if (got := sha256_file(root / o[key]["path"])) != o[key]["sha256"]:
            raise SystemExit(f"original {key} {o[key]['path']}: sha256 {got[:12]} != {o[key]['sha256'][:12]}")
        out[key] = got
    for seed, d in o["preds"].items():
        for name, spec in d.items():
            if sha256_file(root / spec["path"]) != spec["sha256"]:
                raise SystemExit(f"original {name} predictions of seed {seed} changed")
    return out


def int_half(path: Path, half: str) -> dict:
    return {k: v.astype(int) for k, v in load_half(path, half).items()}


# ----------------------------------------------------------------------------------------------- choose
def curve_records(cfg: dict, grid: list, probs: dict, bprobs: dict, ev, cuts: dict, merged_ref: set) -> list[dict]:
    ids = [s.seq_id for s in ev.sequences()]
    if set(ids) != set(probs) or set(ids) != set(bprobs):
        raise SystemExit("probabilities and sequences differ")
    paths_ = {sid: joint_viterbi_grid(probs[sid], bprobs[sid], grid) for sid in ids}; recs = []
    for g, (lam, kap) in enumerate(grid):
        preds = {sid: paths_[sid][g] for sid in ids}; m, t = evaluate(cfg, preds, ev); found, extra = set(), 0
        for s in ev.multi:
            rows, ex = change_records(preds[s.recipe.recipe_id], cuts[s.recipe.recipe_id], s)
            found |= {(r["stream"], r["r"]) for r in rows if r["cat"].startswith("found")}; extra += len(ex)
        ms = t["multi_seg"]; matched, predicted = int(ms["seg_matched"].sum()), int(ms["seg_pred"].sum())
        recs.append({"order": g, "params": {"lambda": lam, "kappa": kap}, **{k: m[k] for k in FLAT}, "extra_changes": extra,
                     "native_switches": int(sum(boundaries_from_labels(preds[s.seq_id]).size for s in ev.natives)),
                     "matched_units": matched, "predicted_segments": predicted, "unmatched_predicted_segments": predicted - matched,
                     "merged_found": len(merged_ref & found), "merged_total": len(merged_ref)})
    return recs


def run_choose(cfg: dict, jd: dict, ks: dict, paths, config_path: str) -> None:
    root = paths.repo_root; check_inputs(jd, root); orig = check_original(ks, root); extract, ext = extraction(root)
    if str(ext.relative_to(root)) != ks["original"]["extract"]["path"]:
        raise SystemExit("the boundary probabilities must come from the original extraction")
    la = l1_asrf_cfg(jd, root); th = float(la["asrf"]["boundary_th"]); grid = grid_of(jd); ch = jd["choice"]
    orig_choices = json.loads((root / ks["original"]["choices"]["path"]).read_text()); orig_sum = json.loads((root / ks["original"]["summary"]["path"]).read_text())
    out = {"status": ks["status"], "config_sha256": sha256_file(root / config_path), "original": orig, "grid": grid, "seeds": {}}
    for seed in seeds_of(jd):
        spec = jd["inputs"]["seeds"][seed]; ev = calibration_set(jd, paths, seed)
        bprobs = load_half(root / extract["seeds"][str(seed)]["boundaries"], "calibration")
        cuts = {s.recipe.recipe_id: candidates(bprobs[s.recipe.recipe_id], th) for s in ev.multi}
        asrf = int_half(root / spec["asrf_preds"]["path"], "calibration")
        merged_ref = {(r["stream"], r["r"]) for s in ev.multi for r in change_records(asrf[s.recipe.recipe_id], cuts[s.recipe.recipe_id], s)[0] if r["cat"] == "merged"}
        if len(merged_ref) != orig_sum["seeds"][str(seed)]["diagnostics"]["B + ASRF"]["calibration"]["change_categories"].get("merged", 0):
            raise SystemExit(f"seed {seed}: B + ASRF merged changes do not reproduce the original round")
        recs = curve_records(cfg, grid, load_half(root / spec["l1_probs"]["path"], "calibration"), bprobs, ev, cuts, merged_ref)
        n = reproduce_original_grid(recs, orig_choices["seeds"][str(seed)]["grid"], seed)
        choice = choose_systems(recs, float(ch["native_min"]), float(ch["recall_min"]))
        out["seeds"][str(seed)] = {"choice": choice, "upper_limit": upper_limit(choice, jd["grid"]["kappa"]), "reproduced_original_settings": n, "grid": recs}
        print(json.dumps({"seed": seed, **{k: {kk: v[kk] for kk in ("status", "params", "n_feasible")} for k, v in choice.items()},
                          "upper_limit": out["seeds"][str(seed)]["upper_limit"]}), flush=True)
    path = root / "runs" / NS / "choices.json"; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path} sha256 {sha256_file(path)}")


def check_choices(jd: dict, root: Path, config_path: str) -> dict:
    path = root / "runs" / NS / "choices.json"
    if not path.is_file():
        raise SystemExit("run --choose first")
    ch = json.loads(path.read_text())
    if ch["config_sha256"] != sha256_file(root / config_path) or [tuple(g) for g in ch["grid"]] != grid_of(jd):
        raise SystemExit("the config or the grid changed after the choice")
    for s in seeds_of(jd):
        if choose_systems(ch["seeds"][str(s)]["grid"], float(jd["choice"]["native_min"]), float(jd["choice"]["recall_min"])) != ch["seeds"][str(s)]["choice"]:
            raise SystemExit(f"seed {s}: the stored choice does not follow from the stored grid")
    return {**ch, "sha256": sha256_file(path)}


# ----------------------------------------------------------------------------------------------- evaluate
def assemble(jd: dict, ks: dict, la: dict, paths, seed: int, choice: dict, extract: dict, ev_halves: dict, slot: dict, l1b3) -> tuple[dict, dict, dict]:
    """Predictions of every system and the ASRF candidates per half slot; slot maps a half name to the data half read."""
    root = paths.repo_root; spec = jd["inputs"]["seeds"][seed]; th = float(la["asrf"]["boundary_th"])
    probs = {h: load_half(root / spec["l1_probs"]["path"], slot[h]) for h in HALVES}
    bprobs = {h: load_half(root / extract["seeds"][str(seed)]["boundaries"], slot[h]) for h in HALVES}
    cuts = {h: {s.recipe.recipe_id: candidates(bprobs[h][s.recipe.recipe_id], th) for s in ev_halves[h].multi} for h in HALVES}
    preds = {"B + ASRF": {h: int_half(root / spec["asrf_preds"]["path"], slot[h]) for h in HALVES}, "L1 + B3": l1b3(probs)}
    for name in SYSTEMS:
        preds[OLD[name]] = {h: int_half(root / ks["original"]["preds"][seed][name]["path"], slot[h]) for h in HALVES}
        if choice[name]["status"] == "feasible":
            preds[name] = {h: decode_system(probs[h], bprobs[h], choice[name]["params"]) for h in HALVES}
    return preds, cuts, probs


def evaluate_core(cfg: dict, preds: dict, ev_halves: dict, seqs: dict, cuts: dict, transfer: dict, three: list[str]) -> tuple[dict, dict]:
    holder = SimpleNamespace(halves=ev_halves); res = {"systems": {}, "diagnostics": {}, "comparisons": {}, "gains_losses": {}}; tables, changes = {}, {}
    for name, p in preds.items():
        rows, tables[name] = evaluate_system(cfg, p, holder, seqs); diag = {}
        for h in HALVES:
            diag[h], changes[(name, h)] = system_diagnostics(p[h], ev_halves[h], cuts[h], transfer[h]); ms = tables[name][h]["multi_seg"]
            diag[h]["segments"] = {"matched": int(ms["seg_matched"].sum()), "predicted": int(ms["seg_pred"].sum()), "reference": int(ms["seg_true"].sum())}
            diag[h]["segments"]["unmatched_predicted"] = diag[h]["segments"]["predicted"] - diag[h]["segments"]["matched"]
        res["systems"][name] = rows; res["diagnostics"][name] = {**diag, "three_clips": clip_reads(tables[name]["selection"]["native"], three)}
    for a, b in CONTRASTS:
        if a in preds and b in preds:
            res["comparisons"][f"{a} - ({b})"] = {h: compare_one(tables[a][h], tables[b][h], preds[a][h], preds[b][h], seqs[h]) for h in HALVES}
    for name in [n for n in SYSTEMS if n in preds]:
        old, d = OLD[name], res["diagnostics"][name]
        d["merged_recovery"] = {h: merged_recovery(changes[("B + ASRF", h)], changes[(name, h)]) for h in HALVES}
        for other in (old, "B + ASRF", "L1 + B3"):
            d[f"native against {other}"] = {h: native_diff(tables[name][h]["native"], tables[other][h]["native"]) for h in HALVES}
        res["gains_losses"][name] = {h: {**gains_losses(changes[(name, h)], changes[(old, h)], changes[("B + ASRF", h)]),
                                         "extra_changes": {"old": res["diagnostics"][old][h]["extra_changes"]["all"], "new": d[h]["extra_changes"]["all"]},
                                         "unmatched_predicted_segments": {"old": res["diagnostics"][old][h]["segments"]["unmatched_predicted"],
                                                                          "new": d[h]["segments"]["unmatched_predicted"]},
                                         "units": res["comparisons"][f"{name} - ({old})"][h]["segment_changes"],
                                         "native_switches": {"old": res["diagnostics"][old][h]["native_switches"]["internal_switches"],
                                                             "new": d[h]["native_switches"]["internal_switches"]}} for h in HALVES}
    return res, tables


def check_reproduction(res: dict, orig: dict, seed: int) -> None:
    """B + ASRF and the kappa <= 8 choices must give exactly their rows of the original round."""
    keys = ("segment_f1_50_multi", "native_macro_f1", "boundary_recall_multi", "boundary_precision_multi", "native_internal_cuts_per_clip")
    for name, old in (("B + ASRF", "B + ASRF"), (OLD["J0"], "J0"), (OLD["J1"], "J1")):
        if any(abs(res["systems"][name][h][k] - orig["seeds"][str(seed)]["systems"][old][h][k]) > 1e-12 for h in HALVES for k in keys):
            raise SystemExit(f"seed {seed}: {name} does not reproduce the original round's rows")


def run_evaluate(cfg: dict, jd: dict, ks: dict, paths, a) -> None:
    root = paths.repo_root; la = l1_asrf_cfg(jd, root); check_inputs(jd, root); check_original(ks, root); extract, _ = extraction(root)
    choices = check_choices(jd, root, a.config); seeds = seeds_of(jd); orig = json.loads((root / ks["original"]["summary"]["path"]).read_text())
    three = three_clips(json.loads((root / jd["inputs"]["l1_asrf_details"]["path"]).read_text()), seeds)
    inputs_e1, seqs = validation(cfg, {"inputs": {"protocol": jd["inputs"]["protocol"], "feature_manifest": {"path": jd["inputs"]["e1_feature_manifest"]["path"]}}}, paths)
    rep = root / "runs" / NS / "report"; rep.mkdir(parents=True, exist_ok=True); out = root / a.out; out.mkdir(parents=True, exist_ok=True)
    res = {"status": ks["status"], "choices": {s: choices["seeds"][str(s)]["choice"] for s in seeds}, "upper_limit": {s: choices["seeds"][str(s)]["upper_limit"] for s in seeds},
           "choices_sha256": choices["sha256"], "seeds": {}, "timing": {}}
    tables, preds = {}, {}
    for s in seeds:
        t0 = time.time(); inp = b_inputs(cfg, jd, paths, s); slot = {h: h for h in HALVES}
        preds[s], cuts, _ = assemble(jd, ks, la, paths, s, choices["seeds"][str(s)]["choice"], extract, inp.halves, slot, lambda _p, s=s, inp=inp: l1_b3(cfg, la, paths, s, inp)[0])
        transfer = {h: fragment_transfer({x.recipe.recipe_id: stream_units(x.recipe) for x in inp.halves[h].multi}, lambda v, var, st=inp.store: st.get(v, var).prob) for h in HALVES}
        r, tables[s] = evaluate_core(cfg, preds[s], inp.halves, seqs, cuts, transfer, three); check_reproduction(r, orig, s)
        if r["diagnostics"]["B + ASRF"]["three_clips"]["right"] != 0 or r["diagnostics"]["L1 + B3"]["three_clips"]["right"] != 3:
            raise SystemExit(f"seed {s}: the three clips do not reproduce")
        for name in [n for n in SYSTEMS if n in preds[s]]:
            np.savez_compressed(rep / f"preds_{name}_s{s}.npz", **{f"{h}|{k}": v.astype(np.int8) for h in HALVES for k, v in preds[s][name][h].items()})
        res["seeds"][str(s)] = r; res["timing"][str(s)] = time.time() - t0; print(f"seed {s} evaluated", flush=True)
    refs = references({**cfg, "center_supervision": {"inputs": {"protocol": jd["inputs"]["protocol"]}}}, paths, inputs_e1, seqs)
    res["references"] = {k: {h: v[h]["metrics"] for h in HALVES} for k, v in refs.items()}
    for s in seeds:
        for ref in STRONGEST:
            if "J1" in preds[s]:
                res["seeds"][str(s)]["comparisons"][f"J1 - ({ref})"] = {h: {"bootstrap": paired_bootstrap(tables[s]["J1"][h], refs[ref][h]["tables"]),
                                                                           "native_session": native_bootstrap(tables[s]["J1"][h]["native"], refs[ref][h]["tables"]["native"], N_BOOT, BOOT_SEED)}
                                                                       for h in HALVES}
    res["stable"] = {f"{x} - ({y})": stable([res["seeds"][str(s)]["comparisons"].get(f"{x} - ({y})") for s in seeds]) for x, y in CONTRASTS}
    for name in SYSTEMS:
        rows = [{"chosen": {"status": res["choices"][s][name]["status"]}, **{h: res["seeds"][str(s)]["systems"][name][h] for h in HALVES}}
                for s in seeds if name in res["seeds"][str(s)]["systems"]]
        res[f"goal_{name}"] = goal(jd, rows) if len(rows) == len(seeds) else {"met": False, "reason": "a seed has no feasible choice"}
    res["decision"] = decisions(res["stable"], {n: res[f"goal_{n}"] for n in SYSTEMS})
    (rep / "details.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "summary.json").write_text(json.dumps(strip_ids(res), indent=1, default=str)); (out / "report_tables.md").write_text(tables_md(strip_ids(res)))
    print(json.dumps({k: res[k] for k in ("decision", "upper_limit")}, indent=1, default=str)); print(f"wrote {out}")


# ----------------------------------------------------------------------------------------------- dry run, plot, tables
def dry_run(cfg: dict, jd: dict, ks: dict, paths) -> None:
    """The evaluation code with the calibration half in both slots and placeholder settings; no selection data."""
    root = paths.repo_root; la = l1_asrf_cfg(jd, root); check_inputs(jd, root); check_original(ks, root); extract, _ = extraction(root); seed = seeds_of(jd)[0]
    ev = calibration_set(jd, paths, seed); fman, _ = load_json(root / jd["inputs"]["seeds"][seed]["b_feature_manifest"]["path"])
    store = FeatureStore(paths.cache_root, fman["encoder"]["cache_key"]); halves = {h: ev for h in HALVES}; slot = {h: "calibration" for h in HALVES}
    seqs = {h: seqs_of(ev, store) for h in HALVES}
    params = json.loads((root / jd["inputs"]["fan224_details"]["path"]).read_text())["models"][f"L1_s{seed}"]["b3_choice"]["params"]
    choice = {"J0": {"status": "feasible", "params": {"lambda": 0.0, "kappa": 16.0}}, "J1": {"status": "feasible", "params": {"lambda": 0.5, "kappa": 16.0}}}
    preds, cuts, _ = assemble(jd, ks, la, paths, seed, choice, extract, halves, slot, lambda p: {h: b3_decode(params, p[h]) for h in HALVES})
    transfer = {h: fragment_transfer({x.recipe.recipe_id: stream_units(x.recipe) for x in ev.multi}, lambda v, var: store.get(v, var).prob) for h in HALVES}
    res, _ = evaluate_core(cfg, preds, halves, seqs, cuts, transfer, [s.seq_id for s in ev.natives[:3]])
    print(json.dumps({"dry_run": "ok (calibration half only, placeholder settings, no metric)", "systems": sorted(res["systems"]),
                      "comparisons": sorted(res["comparisons"]), "gains_losses": {n: sorted(v["selection"]) for n, v in res["gains_losses"].items()},
                      "diagnostics": sorted(res["diagnostics"]["J1"])}, indent=1))


def plot(choices_path: str, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ch = json.loads(Path(choices_path).read_text()); kappas = sorted({g[1] for g in ch["grid"]}); lams = sorted({g[0] for g in ch["grid"]}); seeds = sorted(ch["seeds"])
    titles = {"segment_f1_50_multi": "Segment F1@50", "boundary_recall_multi": "Boundary recall", "native_macro_f1": "Native macro-F1",
              "extra_changes": "Extra changes (multi streams)", "merged_found": "Changes merged by B + ASRF, found"}
    fig, axes = plt.subplots(len(CURVES), len(seeds), figsize=(4.2 * len(seeds), 2.6 * len(CURVES)), sharex=True)
    for j, s in enumerate(seeds):
        grid = ch["seeds"][s]["grid"]; c = ch["seeds"][s]["choice"]
        for i, key in enumerate(CURVES):
            ax = axes[i, j]
            for lam in lams:
                rows = sorted((r for r in grid if r["params"]["lambda"] == lam), key=lambda r: r["params"]["kappa"])
                ax.plot(range(len(kappas)), [r[key] for r in rows], marker=".", lw=1.1, label=f"λ = {lam:g}")
            for name, mk in (("J0", "s"), ("J1", "o")):
                if c[name]["params"]:
                    r = next(r for r in grid if r["params"] == c[name]["params"])
                    ax.plot([kappas.index(c[name]["params"]["kappa"])], [r[key]], mk, mfc="none", mec="black", ms=9)
            if key == "native_macro_f1":
                ax.axhline(0.98, color="grey", lw=0.8, ls="-.")
            if key == "boundary_recall_multi":
                ax.axhline(0.95, color="grey", lw=0.8, ls="-.")
            if key == "merged_found":
                ax.axhline(grid[0]["merged_total"], color="grey", lw=0.8, ls=":")
            if key in ("segment_f1_50_multi",):
                ax.set_ylim(0.9, 0.99)
            ax.set_title(f"seed {s}: {titles[key]}", fontsize=9); ax.grid(alpha=0.3)
            ax.set_xticks(range(len(kappas))); ax.set_xticklabels([f"{k:g}" for k in kappas], fontsize=7)
        axes[-1, j].set_xlabel("switch cost κ")
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("Calibration half: every λ against κ (square: chosen J0; circle: chosen J1; dash-dot: choice floors; dotted: merged total)", fontsize=10)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); print(out_path)


def ci(c: dict, m: str) -> str:
    b = c["bootstrap"][m]
    return f"{100 * b['delta']:+.2f} [{100 * b['ci95'][0]:+.2f}, {100 * b['ci95'][1]:+.2f}]"


def tables_md(res: dict) -> str:
    choice = [{"seed": s, "system": n, "status": c["status"], "lambda": (c["params"] or {}).get("lambda"), "kappa": (c["params"] or {}).get("kappa"),
               "at upper limit": res["upper_limit"][s][n], "feasible settings": f"{c['n_feasible']} of {c['n_settings']}",
               "calibration segment F1@50": (c["calibration"] or {}).get("segment_f1_50_multi")} for s, d in res["choices"].items() for n, c in d.items()]
    systems, comps, gl = [], [], []
    for s, r in res["seeds"].items():
        for name, row in r["systems"].items():
            for h in HALVES:
                m = row[h]; seg = r["diagnostics"][name][h]["segments"]
                systems.append({"seed": s, "system": name, "half": h, "segment F1@50": m["segment_f1_50_multi"], "boundary P": m["boundary_precision_multi"],
                                "boundary R": m["boundary_recall_multi"], "native macro-F1": m["native_macro_f1"], "cuts per clip": m["native_internal_cuts_per_clip"],
                                **{p: m[p] for p in PAIRS}, "matched units": seg["matched"], "unmatched predicted": seg["unmatched_predicted"]})
        for name, c in r["comparisons"].items():
            for h in HALVES:
                comps.append({"seed": s, "comparison": name, "half": h, **{f"d {m} (points)": ci(c[h], m) for m in ("multi segment F1@50", "multi boundary recall", "native macro-F1")}})
        for name, g in r["gains_losses"].items():
            for h in HALVES:
                x = g[h]
                gl.append({"seed": s, "system": name, "half": h, "found by old": x["found_by_old"]["count"], "kept": x["kept"]["count"], "lost": x["lost"]["count"],
                           "found only now": x["found_only_now"]["count"], "recovered kept": f"{x['recovered_kept']['count']} of {x['recovered_by_old']['count']}",
                           "extra changes old/new": f"{x['extra_changes']['old']['count']}/{x['extra_changes']['new']['count']}",
                           "unmatched predicted old/new": f"{x['unmatched_predicted_segments']['old']}/{x['unmatched_predicted_segments']['new']}",
                           "units fixed/lost": f"{x['units']['fixed']['segments']}/{x['units']['new']['segments']}"})
    refs = [{"system": k, "half": h, "segment F1@50": v[h]["segment_f1_50_multi"]} for k, v in res["references"].items() for h in HALVES]
    return "\n".join(["# Joint decoding switch-cost check (post hoc): tables\n", "## Choices (calibration only)\n", md_table(choice), "## Systems\n", md_table(systems),
                      "## Comparisons\n", md_table(comps), "## Gains and losses against the kappa <= 8 choices\n", md_table(gl), "## References\n", md_table(refs),
                      "## Stable rule, goals and decision\n", "```\n" + json.dumps({k: res[k] for k in ("stable", "goal_J0", "goal_J1", "decision")}, indent=1, default=str) + "\n```\n"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--out", default=None); ap.add_argument("--choices", default=None)
    step = ap.add_mutually_exclusive_group(required=True)
    for s in ("choose", "evaluate", "dry-run", "plot"):
        step.add_argument(f"--{s}", action="store_true")
    a = ap.parse_args()
    if a.plot:
        if not (a.choices and a.out):
            raise SystemExit("--plot needs --choices and --out")
        plot(a.choices, a.out); return
    a.set = []; cfg = load_cfg(a); jd, ks = cfg["joint_decoding"], cfg["kappa_supplement"]; paths = load_paths()
    if a.choose:
        run_choose(cfg, jd, ks, paths, a.config)
    elif a.dry_run:
        dry_run(cfg, jd, ks, paths)
    else:
        if not a.out:
            raise SystemExit("--out is required for --evaluate")
        run_evaluate(cfg, jd, ks, paths, a)


if __name__ == "__main__":
    main()
