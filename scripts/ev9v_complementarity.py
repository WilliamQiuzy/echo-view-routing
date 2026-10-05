#!/usr/bin/env python
"""EV9V: are E0 and B7 complementary, and does one fixed equal-weight fusion improve the full system?
(docs/superpowers/plans/2026-10-05-ev9v-complementarity-distillation.md, tasks 1-2; nothing is trained.)

configs/experiments/ev9v_complementarity_v1.yaml fixes every input (paths and hashes), the class handling, the time
axis and the task-2 rules. Only the two validation halves are read; the test split stays sealed; no EchoNet-Pediatric
input is read.

--phase audit (task 1)
  inputs     input hashes; no test clip and no forbidden source; per-sample alignment of E0 and B7 on every
             validation sequence; B7's cached five-family rows against the frozen map; the mass the map drops
  windows    B7 windows whose frames span a join (the windows run along each input stream), and a re-run of B7 with
             the unchanged runner on a fixed subset (--write-spot-plans writes the subset and prints the command)
  reproduce  the frozen E0 + B3 and B7 + B3 against their reported metrics and segment counts, exactly
  cells      units and native clips: both right, only E0 right, only B7 right, both wrong, by recording, session,
             class and half
  opinion    where the systems differ, each teacher's class over the reference interval (diagnostic only)
--phase fusion (task 2)
  q = 0.5 p_E0 + 0.5 p_B7 per sample. E0, B7 and q each go through the same B3 grid; each setting is chosen on the
  calibration half by the fixed rule and applied once to the selection half. Metrics, fixed and new errors by source,
  paired bootstrap and the four start conditions of task 3.
Writes audit.json, audit.md, units.csv and natives.csv, or fusion.json, fusion.md, grid_calibration.csv and
changes.csv, into a new run directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg, new_run_dir  # noqa: E402
from scripts.calibrate_simple_decoders import grid as simple_grid  # noqa: E402
from scripts.report_model_development import paired_bootstrap  # noqa: E402
from scripts.round4_failure_inventory import b3, majority  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from echo_routing.compose.recipe_builder import Recipe, recipe_from_dict  # noqa: E402
from echo_routing.config import load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import (CELLS, AlignmentError, InputError, cell, changes, check_alignment,  # noqa: E402
                                                   check_distribution, check_no_test, check_sources, concentration,
                                                   files_digest, fuse, interval_class, map_b7, mapping_matrix, native_refs,
                                                   normalise_rows, opinion_kind, recipe_refs, select_setting, sha256_file)
from echo_routing.evaluate.dev_metrics import evaluate_predictions  # noqa: E402
from echo_routing.evaluate.failure_units import session_of, stream_units, unit_outcomes  # noqa: E402
from echo_routing.features.extract import VARIANT_GAMMA  # noqa: E402
from echo_routing.temporal.dev_run import HALVES, Inputs  # noqa: E402
from echo_routing.temporal.segments import boundaries_from_labels  # noqa: E402
from echo_routing.temporal.windowed import window_centres, window_samples  # noqa: E402

CLASSES = ("PLAX", "PSAX", "A4C", "A5C", "SC4C")
A4C, A5C = 2, 3
KINDS = ("native", "pairs", "multi")
SEG = ["seg_matched", "seg_pred", "seg_true"]
REPRO_KEYS = ("native_macro_f1", "segment_f1_50_multi", "boundary_recall_multi", "native_internal_cuts_per_clip")
FLAT = ("native_macro_f1", "segment_f1_50_multi", "boundary_recall_multi", "boundary_precision_multi", "boundary_f1_multi",
        "native_internal_cuts_per_clip", "same_class_cut_rate_multi", "segment_f1_50_pairs")


@dataclass
class Seq5:
    seq_id: str
    kind: str                  # native | pairs | multi
    labels: np.ndarray
    e0: np.ndarray             # E0 softmax rows as cached
    b7_stored: np.ndarray      # B7 five-family rows as cached: the input of the frozen B7 + B3
    b7: np.ndarray             # the same map divided by its row sum: the B7 teacher of task 2
    dropped: np.ndarray        # B7 mass of the dropped code per sample
    recipe: Recipe | None
    joins: np.ndarray


# ----------------------------------------------------------------------------------------------- inputs
def hashed_inputs(cfg: dict) -> dict[str, tuple[str, str]]:
    i, e, b = cfg["inputs"], cfg["inputs"]["E0"], cfg["inputs"]["B7"]
    return {"protocol": (i["protocol"]["path"], i["protocol"]["sha256"]),
            "source manifest": (i["source_manifest"]["path"], i["source_manifest"]["sha256"]),
            "E0 feature manifest": (e["feature_manifest"], e["feature_manifest_sha256"]),
            "B7 checkpoint": (b["checkpoint"], b["checkpoint_sha256"]), "B7 runner config": (b["runner_config"], b["runner_config_sha256"]),
            "B7 label list": (b["label_list"], b["label_list_sha256"]), "B7 window plans": (b["window_plans"], b["window_plans_sha256"])}


def e0_files(proto: dict) -> list[str]:
    """Every E0 cache file (<variant>/<clip>.npz) that the two halves read."""
    out = set()
    for h in HALVES:
        out |= {f"orig/{v}.npz" for v in proto["validation"][h]["clips"]}
        for kind in ("pairs", "multi"):
            for d in proto["validation"][h]["banks"][kind]["recipes"]:
                out |= {f"{fr.variant}/{fr.video_id}.npz" for fr in recipe_from_dict(d).fragments}
    return sorted(out)


def digests(cfg: dict, paths) -> dict[str, str]:
    root, i = paths.repo_root, cfg["inputs"]
    proto = json.loads((root / i["protocol"]["path"]).read_text())
    plans = json.loads((root / i["B7"]["window_plans"]).read_text())["streams"]
    return {"E0 files": files_digest(paths.cache_root / "features" / i["E0"]["cache_key"], e0_files(proto)),
            "B7 predictions": files_digest(root / i["B7"]["pred_dir"], [f"{sid}.npz" for sid in plans])}


def load(cfg: dict, paths) -> tuple[Inputs, dict, dict]:
    root, i, c = paths.repo_root, cfg["inputs"], cfg["classes"]
    named = hashed_inputs(cfg)
    check_sources([p for p, _ in named.values()] + [i["B7"]["pred_dir"], i["E0"]["frozen_b3_run"], i["B7"]["frozen_b3_run"],
                                                    cfg["audit"]["round4_clue"]], cfg["forbidden_input_substrings"])
    for name, (p, sha) in named.items():
        if (got := sha256_file(root / p)) != sha:
            raise InputError(f"{name} {p}: sha256 {got[:12]} != {sha[:12]}")
    dg = digests(cfg, paths)
    for name, want in (("E0 files", i["E0"]["files_digest"]), ("B7 predictions", i["B7"]["pred_digest"])):
        if dg[name] != want:
            raise InputError(f"{name}: digest {dg[name][:12]} != {str(want)[:12]}")
    manifest = pd.read_csv(root / i["source_manifest"]["path"], dtype={"video_id": str})
    test = set(manifest.loc[manifest["split"] == "test", "video_id"])
    inputs = Inputs(cfg, root / i["protocol"]["path"], root / i["E0"]["feature_manifest"], paths.cache_root)
    if inputs.cache_key != i["E0"]["cache_key"]:
        raise InputError(f"E0 cache key {inputs.cache_key} != {i['E0']['cache_key']}")
    plans = json.loads((root / i["B7"]["window_plans"]).read_text())["streams"]
    if {p["bank"] for p in plans.values()} != {f"{k}_{h}" for k in KINDS for h in HALVES}:
        raise InputError(f"plan banks {sorted({p['bank'] for p in plans.values()})} are not the six validation banks")
    mat, pred_dir = mapping_matrix(c["b7_raw9_order"], c["b7_raw9_to_family5"], c["family5"]), root / i["B7"]["pred_dir"]
    data: dict[str, dict[str, Seq5]] = {h: {} for h in HALVES}
    for h in HALVES:
        ev = inputs.halves[h]
        items = [(s.seq_id, "native", s.labels, inputs.store.get(s.seq_id, "orig").prob, native_refs(inputs.store.get(s.seq_id, "orig")),
                  None, np.zeros(0, int)) for s in ev.natives]
        for kind, coll in (("pairs", ev.pairs), ("multi", ev.multi)):
            items += [(st.recipe.recipe_id, kind, st.labels, st.prob, recipe_refs(st.recipe, inputs.store.get), st.recipe, st.joins) for st in coll]
        for sid, kind, labels, e0, refs, recipe, joins in items:
            plan = plans.get(sid)
            if plan is None or plan["bank"] != f"{kind}_{h}":
                raise AlignmentError(f"{sid}: no B7 plan in bank {kind}_{h}")
            check_alignment(refs, plan["refs"], VARIANT_GAMMA, sid)
            check_no_test({r[0] for r in refs} | {r[0] for r in plan["refs"]}, test)
            check_distribution(e0, c["e0_sum_tolerance"])
            with np.load(pred_dir / f"{sid}.npz") as z:
                stored, raw = z["prob"].astype(float), z["raw"].astype(float)
            mapped, dropped = map_b7(raw, mat, stored, c["b7_map_tolerance"])
            data[h][sid] = Seq5(sid, kind, np.asarray(labels, int), np.asarray(e0, float), stored,
                                normalise_rows(mapped, c["b7_min_mapped_mass"]), dropped, recipe, np.asarray(joins, int))
    loaded = {sid for h in HALVES for sid in data[h]}
    if loaded != set(plans):
        raise AlignmentError(f"B7 plans and the halves' sequences differ: {len(set(plans) - loaded)} extra, {len(loaded - set(plans))} missing")
    counts = {h: {k: {"sequences": sum(sd.kind == k for sd in data[h].values()), "samples": int(sum(sd.labels.size for sd in data[h].values() if sd.kind == k))}
                  for k in KINDS} for h in HALVES}
    print(f"inputs verified; aligned {len(loaded)} sequences: {json.dumps(counts)}", flush=True)
    return inputs, data, {"hashes": {k: v[1] for k, v in named.items()}, "digests": dg, "manifest_test_clips": len(test),
                          "test_clips_in_inputs": 0, "sequences": counts}


# ----------------------------------------------------------------------------------------------- decoding helpers
def evaluate(cfg: dict, preds: dict, ev) -> tuple[dict, dict]:
    return evaluate_predictions(preds, ev, float(cfg["protocol"]["hz"]), float(cfg["selection"]["boundary_tolerance_s"]))


def decode(params: dict, probs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {sid: b3(p, params) for sid, p in probs.items()}


def run_system(cfg: dict, inputs: Inputs, probs: dict, params: dict, halves=HALVES) -> dict:
    out = {}
    for h in halves:
        preds = decode(params, probs[h]); m, tables = evaluate(cfg, preds, inputs.halves[h])
        out[h] = {"preds": preds, "metrics": m, "tables": tables}
    return out


def frozen_systems(cfg: dict, paths) -> dict:
    root = paths.repo_root; e = root / cfg["inputs"]["E0"]["frozen_b3_run"]; b = root / cfg["inputs"]["B7"]["frozen_b3_run"]
    b7 = json.loads((b / "posterior_decoders.json").read_text())["results"]["B7"]
    if b7["run_dir"] != cfg["inputs"]["B7"]["pred_dir"]:
        raise InputError(f"the frozen B7 + B3 read {b7['run_dir']}, not {cfg['inputs']['B7']['pred_dir']}")
    e0m = json.loads((e / "validation_metrics.json").read_text())["methods"]["B3 tuned (main)"]
    return {"E0 frozen": {"params": json.loads((e / "selected.json").read_text())["selected"]["b3"]["main"], "input": "e0",
                          "metrics": {h: e0m[h] for h in HALVES}, "rows": str(e / "rows_b3_tuned_main_{half}_multi_seg.csv")},
            "B7 frozen": {"params": b7["families"]["b3"]["params"], "input": "b7_stored",
                          "metrics": {h: b7["families"]["b3"][h] for h in HALVES}, "rows": str(b / "rows_B7_b3_{half}_multi_seg.csv")}}


def reproduce(cfg: dict, inputs: Inputs, data: dict, fz: dict) -> tuple[dict, dict]:
    report, runs, tol = {}, {}, float(cfg["audit"]["reproduce"]["tolerance"])
    for name, f in fz.items():
        probs = {h: {sid: getattr(sd, f["input"]) for sid, sd in data[h].items()} for h in HALVES}
        runs[name] = run_system(cfg, inputs, probs, f["params"])
        for h in HALVES:
            m, t = runs[name][h]["metrics"], runs[name][h]["tables"]
            diff = max(abs(m[k] - f["metrics"][h][k]) for k in REPRO_KEYS)
            got, want = t["multi_seg"][SEG].sum().to_numpy(), pd.read_csv(f["rows"].format(half=h))[SEG].sum().to_numpy()
            if diff > tol or not np.array_equal(got, want):
                raise SystemExit(f"{name} {h}: not reproduced (max metric difference {diff:.2e}; segments {got.tolist()} vs {want.tolist()})")
            report[f"{name}|{h}"] = {**{k: m[k] for k in REPRO_KEYS}, "segments_matched_pred_true": got.tolist(), "max_abs_diff": diff}
    print("frozen E0 + B3 and B7 + B3 reproduced exactly", flush=True)
    return report, runs


# ----------------------------------------------------------------------------------------------- outcomes
def unit_frame(cfg: dict, data_h: dict, h: str, preds: dict) -> pd.DataFrame:
    cz, rows = cfg["audit"]["causes"], []
    for sid, sd in data_h.items():
        if sd.kind != "multi":
            continue
        units = stream_units(sd.recipe)
        outs = {name: unit_outcomes(p[sid], units, 0.5, int(cz["hit"]), int(cz["near"]))[0] for name, p in preds.items()}
        for u in units:
            srcs = sorted({v for v, *_ in u.sources})
            row = {"half": h, "key": f"{sid}#{u.index}", "stream": sid, "unit": u.index, "start": u.start, "end": u.end, "length": u.length,
                   "label": u.label, "left": u.left, "right": u.right, "sources": ";".join(srcs), "sessions": ";".join(sorted({session_of(v) for v in srcs}))}
            for name, o in outs.items():
                r = o[u.index]
                row.update({f"{name}_right": bool(r["matched"]), f"{name}_cause": r["cause"], f"{name}_majority": r["majority"],
                            f"{name}_extra_cuts": r["extra_cuts"]})
            rows.append(row)
    return pd.DataFrame(rows)


def native_frame(data_h: dict, h: str, preds: dict) -> pd.DataFrame:
    rows = []
    for sid, sd in data_h.items():
        if sd.kind != "native":
            continue
        row = {"half": h, "key": sid, "label": int(sd.labels[0]), "sources": sid, "sessions": session_of(sid)}
        for name, p in preds.items():
            maj = majority(p[sid])
            row.update({f"{name}_right": maj == int(sd.labels[0]), f"{name}_majority": maj, f"{name}_cuts": int(boundaries_from_labels(p[sid]).size)})
        rows.append(row)
    return pd.DataFrame(rows)


def with_cells(df: pd.DataFrame, data: dict, a: str = "E0", b: str = "B7") -> pd.DataFrame:
    """Outcome cell of the two frozen systems and each teacher's class over the item's reference interval."""
    ce0, cb7 = [], []
    for r in df.itertuples():
        sd = data[r.half][r.stream] if "stream" in df else data[r.half][r.key]
        lo, hi = (r.start, r.end) if "stream" in df else (0, sd.labels.size)
        ce0.append(interval_class(sd.e0, lo, hi)); cb7.append(interval_class(sd.b7, lo, hi))
    return df.assign(cell=[cell(x, y) for x, y in zip(df[f"{a}_right"], df[f"{b}_right"])], E0_interval_class=ce0, B7_interval_class=cb7,
                     opinion=[opinion_kind(x, y) for x, y in zip(ce0, cb7)])


def distinct(series: pd.Series) -> int:
    return len({x for s in series for x in str(s).split(";") if x})


def cell_table(df: pd.DataFrame, by: list[str]) -> list[dict]:
    rows = []
    for keys, g in df.groupby(by, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for c in CELLS:
            sub = g[g["cell"] == c]
            rows.append({**dict(zip(by, keys)), "cell": c, "items": len(sub), "recordings": distinct(sub["sources"]), "sessions": distinct(sub["sessions"])})
    return rows


def recording_table(df: pd.DataFrame, which: str, top: int = 10) -> dict:
    sub = df[df["cell"] == which]
    per = sub.assign(rec=sub["sources"].str.split(";")).explode("rec").groupby("rec").size().sort_values(ascending=False)
    conc = concentration(list(sub["sources"].str.split(";")), [])
    return {"items": len(sub), "recordings": int(per.size), "top": [[r, int(n)] for r, n in per.head(top).items()],
            **{f"items_involving_top{k}": conc[f"top{k}"]["fixed_involving"] for k in (1, 3, 5)}}


def opinion_table(df: pd.DataFrame) -> list[dict]:
    rows = []
    for c in CELLS[1:]:
        sub = df[df["cell"] == c]
        for kind in ("teachers disagree on class", "same class opinion"):
            s = sub[sub["opinion"] == kind]
            rows.append({"cell": c, "opinion": kind, "items": len(s), "recordings": distinct(s["sources"]),
                         "E0 class right": int((s["E0_interval_class"] == s["label"]).sum()), "B7 class right": int((s["B7_interval_class"] == s["label"]).sum())})
    return rows


# ----------------------------------------------------------------------------------------------- audit
def mass_report(data: dict) -> dict:
    out = {}
    for h in HALVES:
        d = np.concatenate([sd.dropped for sd in data[h].values()]); lab = np.concatenate([sd.labels for sd in data[h].values()])
        out[h] = {"samples": int(d.size), "mean": float(d.mean()), "median": float(np.median(d)), "p99": float(np.quantile(d, 0.99)),
                  "max": float(d.max()), "share_above_0.1": float((d > 0.1).mean()), "share_above_0.5": float((d > 0.5).mean()),
                  "mean_by_true_class": {CLASSES[k]: float(d[lab == k].mean()) for k in range(5) if (lab == k).any()}}
    return out


def window_report(cfg: dict, data: dict) -> dict:
    w, out = cfg["inputs"]["B7"]["window"], {}
    for h in HALVES:
        for kind in ("pairs", "multi"):
            n_win = n_span = n_samples = n_assigned = 0
            for sd in (x for x in data[h].values() if x.kind == kind):
                n = sd.labels.size; centres = window_centres(n, int(w["stride_samples"]))
                frag = np.searchsorted(sd.joins, np.arange(n), side="right")
                spans = np.array([len(set(frag[s[np.linspace(0, len(s) - 1, int(w["frames"])).round().astype(int)]].tolist())) > 1
                                  for s in (window_samples(int(c), n, int(w["samples"])) for c in centres)])
                nearest = np.abs(np.arange(n)[:, None] - centres[None, :]).argmin(1)
                n_win += centres.size; n_span += int(spans.sum()); n_samples += n; n_assigned += int(spans[nearest].sum())
            out[f"{h}/{kind}"] = {"windows": n_win, "windows_spanning_a_join": n_span, "samples": n_samples, "samples_given_a_spanning_window": n_assigned}
    return out


def spot_ids(plans: dict, per_bank: int) -> list[str]:
    by: dict[str, list[str]] = {}
    for sid, p in plans.items():
        by.setdefault(p["bank"], []).append(sid)
    return [sid for bank in sorted(by) for sid in sorted(by[bank])[:per_bank]]


def write_spot_plans(cfg: dict, paths) -> None:
    root, b, sp = paths.repo_root, cfg["inputs"]["B7"], cfg["audit"]["spot_check"]
    allp = json.loads((root / b["window_plans"]).read_text()); ids = spot_ids(allp["streams"], int(sp["per_bank"]))
    d = root / sp["dir"]; d.mkdir(parents=True, exist_ok=True)
    (d / "plans.json").write_text(json.dumps({**{k: v for k, v in allp.items() if k != "streams"}, "streams": {s: allp["streams"][s] for s in ids}}))
    print(f"wrote {d / 'plans.json'} ({len(ids)} streams). Run:\n  envs/echoviewclip/bin/python {b['runner']} --plans {sp['dir']}/plans.json "
          f"--ckpt {b['checkpoint']} --config {b['runner_config']} --out {sp['dir']}/pred")


def spot_report(cfg: dict, paths, plans: dict) -> dict:
    root, sp = paths.repo_root, cfg["audit"]["spot_check"]; ids = spot_ids(plans, int(sp["per_bank"])); d = root / sp["dir"] / "pred"
    missing = [s for s in ids if not (d / f"{s}.npz").is_file()]
    if missing:
        raise SystemExit(f"spot-check outputs missing ({len(missing)}): run --write-spot-plans and the printed command first")
    diffs = {}
    for s in ids:
        with np.load(d / f"{s}.npz") as a, np.load(root / cfg["inputs"]["B7"]["pred_dir"] / f"{s}.npz") as b:
            diffs[s] = float(max(np.abs(a["raw"] - b["raw"]).max(), np.abs(a["prob"] - b["prob"]).max()))
    res = {"streams": ids, "max_abs_diff": max(diffs.values()), "per_stream": diffs, "tolerance": float(sp["tolerance"])}
    if res["max_abs_diff"] > res["tolerance"]:
        raise SystemExit(f"the B7 cache is not reproduced by the unchanged runner (max difference {res['max_abs_diff']:.2e})")
    return res


def round4_clue(cfg: dict, paths, units: pd.DataFrame) -> dict:
    r4 = pd.read_csv(paths.repo_root / cfg["audit"]["round4_clue"]); r4 = r4[r4["enc"] == "E0"]
    m = r4.merge(units, left_on=["half", "stream", "unit"], right_on=["half", "stream", "unit"], how="inner", validate="one_to_one")
    agree = bool(len(m) == len(r4) == len(units) and (m["B3_matched"].astype(bool) == m["E0_right"]).all()
                 and (m["B7_B3_matched"].astype(bool) == m["B7_right"]).all())
    asrf_wrong = sum((~m[f"ASRF_s{k}_matched"].astype(bool)).astype(int) for k in range(3))
    common, common_right = ~m["E0_right"] & (asrf_wrong >= 2), m["E0_right"] & (asrf_wrong <= 1)
    out = {"agrees_with_round4_outcomes": agree}
    for name, mask in (("E0 common errors (B3 and >= 2 ASRF seeds wrong)", common), ("E0 common right (B3 and >= 2 ASRF seeds right)", common_right)):
        for h in (*HALVES, "both"):
            s = m[mask & ((m["half"] == h) if h != "both" else True)]
            out[f"{name}|{h}"] = {"units": len(s), "B7_B3_wrong": int((~s["B7_right"]).sum()), "B7_B3_right": int(s["B7_right"].sum()),
                                  "recordings": distinct(s["sources_y"]), "recordings_B7_right": distinct(s.loc[s["B7_right"], "sources_y"]),
                                  "recordings_B7_wrong": distinct(s.loc[~s["B7_right"], "sources_y"])}
    return out


def audit(cfg: dict, paths, args) -> None:
    inputs, data, meta = load(cfg, paths)
    plans = json.loads((paths.repo_root / cfg["inputs"]["B7"]["window_plans"]).read_text())["streams"]
    spot = spot_report(cfg, paths, plans)
    fz = frozen_systems(cfg, paths); repro, runs = reproduce(cfg, inputs, data, fz)
    units = pd.concat([unit_frame(cfg, data[h], h, {"E0": runs["E0 frozen"][h]["preds"], "B7": runs["B7 frozen"][h]["preds"]}) for h in HALVES])
    nats = pd.concat([native_frame(data[h], h, {"E0": runs["E0 frozen"][h]["preds"], "B7": runs["B7 frozen"][h]["preds"]}) for h in HALVES])
    units, nats = with_cells(units.reset_index(drop=True), data), with_cells(nats.reset_index(drop=True), data)
    units["class"] = units["label"].map(dict(enumerate(CLASSES))); nats["class"] = nats["label"].map(dict(enumerate(CLASSES)))
    rep = {"inputs": meta, "coverage": cfg["inputs"]["B7"]["coverage"], "b7_dropped_mass": mass_report(data), "windows": window_report(cfg, data),
           "spot_check": spot, "reproduction": repro, "frozen_params": {k: v["params"] for k, v in fz.items()},
           "units_by_half": cell_table(units, ["half"]), "units_by_class": cell_table(units, ["half", "class"]),
           "natives_by_half": cell_table(nats, ["half"]), "natives_by_class": cell_table(nats, ["half", "class"]),
           "recordings": {f"{h}|{c}": recording_table(units[units["half"] == h], c) for h in HALVES for c in CELLS[1:3]},
           "opinion_units": {h: opinion_table(units[units["half"] == h]) for h in HALVES},
           "opinion_natives": {h: opinion_table(nats[nats["half"] == h]) for h in HALVES},
           "round4_clue": round4_clue(cfg, paths, units)}
    run_dir = new_run_dir(cfg, "ev9v_complementarity_audit")
    units.to_csv(run_dir / "units.csv", index=False); nats.to_csv(run_dir / "natives.csv", index=False)
    (run_dir / "audit.json").write_text(json.dumps(rep, indent=1, default=str)); (run_dir / "audit.md").write_text(audit_md(rep))
    print(f"run_dir={run_dir}")


# ----------------------------------------------------------------------------------------------- fusion
def calibrate(cfg: dict, inputs: Inputs, probs_cal: dict) -> tuple[list[dict], dict]:
    ev = inputs.halves["calibration"]
    settings = [p for m, p in simple_grid(cfg, [probs_cal[s.seq_id] for s in ev.natives]) if m == "b3"]
    records = []
    for order, params in enumerate(settings):
        m, _ = evaluate(cfg, decode(params, probs_cal), ev)
        records.append({"order": order, "params": params, **{k: m[k] for k in FLAT}})
    r = cfg["fusion"]["rule"]
    return records, select_setting(records, float(r["native_macro_f1_min"]), float(r["boundary_recall_min"]))


def system_metrics(m: dict, units: pd.DataFrame, name: str) -> dict:
    conf = np.asarray(m["native_confusion"]); wrong = units[~units[f"{name}_right"]]
    return {**{k: m[k] for k in FLAT}, "native_A4C_as_A5C": int(conf[A4C, A5C]), "native_A5C_as_A4C": int(conf[A5C, A4C]),
            "units_A4C_as_A5C": int(((wrong["label"] == A4C) & (wrong[f"{name}_majority"] == A5C)).sum()),
            "units_A5C_as_A4C": int(((wrong["label"] == A5C) & (wrong[f"{name}_majority"] == A4C)).sum()),
            "units_wrong": len(wrong), "extra_cuts_inside_units": int(units[f"{name}_extra_cuts"].sum())}


def compare(a: str, b: str, h: str, runs: dict, units: pd.DataFrame, nats: pd.DataFrame) -> dict:
    """a minus b on one half: paired bootstrap, fixed and newly wrong items with their sources, and the totals check."""
    ta, tb = runs[a][h]["tables"], runs[b][h]["tables"]
    u, n = units[units["half"] == h], nats[nats["half"] == h]
    fu, nu = changes(dict(zip(u["key"], u[f"{b}_right"])), dict(zip(u["key"], u[f"{a}_right"])))
    fn, nn = changes(dict(zip(n["key"], n[f"{b}_right"])), dict(zip(n["key"], n[f"{a}_right"])))
    sa, sb = ta["multi_seg"][SEG].sum(), tb["multi_seg"][SEG].sum()
    conf_a, conf_b = np.trace(np.asarray(runs[a][h]["metrics"]["native_confusion"])), np.trace(np.asarray(runs[b][h]["metrics"]["native_confusion"]))
    consistent = bool(int(u[f"{a}_right"].sum()) == sa["seg_matched"] and int(u[f"{b}_right"].sum()) == sb["seg_matched"]
                      and len(u) == sa["seg_true"] == sb["seg_true"] and len(fu) - len(nu) == sa["seg_matched"] - sb["seg_matched"]
                      and int(n[f"{a}_right"].sum()) == conf_a and int(n[f"{b}_right"].sum()) == conf_b and len(fn) - len(nn) == conf_a - conf_b)
    src = dict(zip(u["key"], u["sources"].str.split(";")))
    return {"bootstrap": paired_bootstrap(ta, tb), "consistent_with_totals": consistent,
            "segments": {a: [int(x) for x in sa], b: [int(x) for x in sb]},
            "units": {"fixed": fu, "new_errors": nu, "concentration": concentration([src[k] for k in fu], [src[k] for k in nu])},
            "natives": {"fixed": fn, "new_errors": nn, "concentration": concentration([[k] for k in fn], [[k] for k in nn])}}


def start_conditions(fz: dict, systems: dict, comps: dict) -> dict:
    q, m, ref = systems["q"], float(fz["margin"]), fz["references"]
    if q["status"] != "feasible":
        return {"q_feasible": False, "met": {str(k): False for k in range(1, 5)}, "start_training": False}
    seg = {name: {h: s[h]["segment_f1_50_multi"] for h in HALVES} for name, s in systems.items() if s["status"] == "feasible"}
    c1 = {h: seg["q"][h] - float(ref[h]["value"]) for h in HALVES}
    controls = [s for s in ("E0", "B7") if systems[s]["status"] == "feasible"]
    c2 = {s: {h: seg["q"][h] - seg[s][h] for h in HALVES} for s in controls}
    c3 = {h: {"boundary_recall": q[h]["boundary_recall_multi"], "native_macro_f1": q[h]["native_macro_f1"]} for h in HALVES}
    met = {"1": all(v >= m for v in c1.values()),
           "2": bool(controls) and all(v >= m for d in c2.values() for v in d.values()),
           "3": all(c3[h]["boundary_recall"] >= float(fz["boundary_recall_floor"]) and c3[h]["native_macro_f1"] >= float(fz["native_floor"][h]) for h in HALVES),
           "4": all(comps[k][h]["consistent_with_totals"] for k in comps if k.startswith("q - ") for h in HALVES)}
    return {"q_feasible": True, "margins_vs_references": c1, "margins_vs_controls": c2, "infeasible_controls": [s for s in ("E0", "B7") if s not in controls],
            "floors": c3, "met": met, "start_training": all(met.values()),
            "sensitivity_calibration": seg["q"]["calibration"] - float(ref["calibration_sensitivity"]["value"])}


def fusion(cfg: dict, paths, args) -> None:
    inputs, data, meta = load(cfg, paths); fzc = cfg["fusion"]; w = fzc["weights"]
    probs = {"E0": {h: {s: sd.e0 for s, sd in data[h].items()} for h in HALVES},
             "B7": {h: {s: sd.b7 for s, sd in data[h].items()} for h in HALVES},
             "q": {h: {s: fuse(sd.e0, sd.b7, float(w["E0"]), float(w["B7"])) for s, sd in data[h].items()} for h in HALVES}}
    for h in HALVES:
        for s in data[h]:
            check_distribution(probs["q"][h][s], float(cfg["classes"]["e0_sum_tolerance"]))
    fz = frozen_systems(cfg, paths); _, runs = reproduce(cfg, inputs, data, fz)
    systems = {k: {"status": "frozen original", "params": v["params"], **{h: runs[k][h]["metrics"] for h in HALVES}} for k, v in fz.items()}
    grid_rows = []
    for name in fzc["systems"]:
        records, sel = calibrate(cfg, inputs, probs[name])
        grid_rows += [{"system": name, **r, "params": json.dumps(r["params"])} for r in records]
        systems[name] = {"status": sel["status"], "n_feasible": sel["n_feasible"], "n_settings": sel["n_settings"],
                         "params": sel["chosen"]["params"] if sel["chosen"] else None}
        if sel["chosen"]:
            runs[name] = run_system(cfg, inputs, probs[name], sel["chosen"]["params"])   # the selection half: this one setting only
            systems[name].update({h: runs[name][h]["metrics"] for h in HALVES})
        print(f"{name}: {sel['status']} ({sel['n_feasible']}/{sel['n_settings']} feasible) {json.dumps(systems[name]['params'])}", flush=True)
    names = [n for n in runs]
    units = pd.concat([unit_frame(cfg, data[h], h, {n: runs[n][h]["preds"] for n in names}) for h in HALVES]).reset_index(drop=True)
    nats = pd.concat([native_frame(data[h], h, {n: runs[n][h]["preds"] for n in names}) for h in HALVES]).reset_index(drop=True)
    table = {n: {h: system_metrics(runs[n][h]["metrics"], units[units["half"] == h], n) for h in HALVES} for n in names}
    comps = {}
    for a, b in fzc["comparisons"]:
        if a in runs and b in runs:
            comps[f"{a} - {b}"] = {h: compare(a, b, h, runs, units, nats) for h in HALVES}
            print(f"compared {a} - {b}", flush=True)
    rep = {"inputs": meta, "weights": w, "systems": {n: {k: v for k, v in s.items() if k not in HALVES} for n, s in systems.items()},
           "metrics": table, "comparisons": comps, "references": fzc["references"],
           "start_conditions": start_conditions(fzc, systems, comps)}
    run_dir = new_run_dir(cfg, "ev9v_complementarity_fusion")
    pd.DataFrame(grid_rows).to_csv(run_dir / "grid_calibration.csv", index=False)
    change_rows = [{"comparison": k, "half": h, "kind": kind, "change": ch, "key": key}
                   for k, v in comps.items() for h in HALVES for kind in ("units", "natives") for ch in ("fixed", "new_errors") for key in v[h][kind][ch]]
    pd.DataFrame(change_rows).to_csv(run_dir / "changes.csv", index=False)
    units.to_csv(run_dir / "units.csv", index=False); nats.to_csv(run_dir / "natives.csv", index=False)
    (run_dir / "fusion.json").write_text(json.dumps(rep, indent=1, default=str)); (run_dir / "fusion.md").write_text(fusion_md(rep))
    print(json.dumps(rep["start_conditions"], indent=1, default=str)); print(f"run_dir={run_dir}")


# ----------------------------------------------------------------------------------------------- markdown
def md_table(rows: list[dict], cols: list[str] | None = None) -> str:
    if not rows:
        return "(none)\n"
    cols = cols or list(rows[0])
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)  # noqa: E731
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |" for r in rows]) + "\n"


def audit_md(rep: dict) -> str:
    out = ["# Task 1 audit: E0 and B7 on the EV9V validation halves\n", f"Coverage: {rep['coverage']}\n",
           "## Reproduction of the frozen systems\n", md_table([{"system|half": k, **v} for k, v in rep["reproduction"].items()]),
           "## B7 mass dropped by the frozen map (PMPALA)\n", md_table([{"half": h, **{k: v for k, v in d.items() if k != "mean_by_true_class"}}
                                                                        for h, d in rep["b7_dropped_mass"].items()]),
           "## B7 windows that span a join\n", md_table([{"half/kind": k, **v} for k, v in rep["windows"].items()]),
           f"## Spot re-run of B7\n\nmax |difference| {rep['spot_check']['max_abs_diff']:.2e} over {len(rep['spot_check']['streams'])} streams\n",
           "## Units (reference segments; not independent cases)\n", md_table(rep["units_by_half"]), md_table(rep["units_by_class"]),
           "## Native clips\n", md_table(rep["natives_by_half"]), md_table(rep["natives_by_class"]),
           "## Recordings behind the one-sided units\n", md_table([{"half|cell": k, **{x: y for x, y in v.items() if x != "top"}} for k, v in rep["recordings"].items()]),
           "## Teacher class over the reference interval\n"]
    for h in HALVES:
        out += [f"Units, {h}\n", md_table(rep["opinion_units"][h]), f"Native clips, {h}\n", md_table(rep["opinion_natives"][h])]
    out += ["## Round-4 clue\n", md_table([{"group|half": k, **v} for k, v in rep["round4_clue"].items() if isinstance(v, dict)]),
            f"Outcomes agree with the round-4 inventory: {rep['round4_clue']['agrees_with_round4_outcomes']}\n"]
    return "\n".join(out)


def fusion_md(rep: dict) -> str:
    cols = ["native_macro_f1", "native_A4C_as_A5C", "native_A5C_as_A4C", "segment_f1_50_multi", "boundary_recall_multi",
            "native_internal_cuts_per_clip", "units_wrong", "units_A5C_as_A4C", "units_A4C_as_A5C", "extra_cuts_inside_units"]
    rows = [{"system": n, "status": rep["systems"][n]["status"], "half": h, **{c: rep["metrics"][n][h][c] for c in cols}} for n in rep["metrics"] for h in HALVES]
    comp = []
    for k, v in rep["comparisons"].items():
        for h in HALVES:
            bs = v[h]["bootstrap"]; cu, cn = v[h]["units"]["concentration"], v[h]["natives"]["concentration"]
            comp.append({"comparison": k, "half": h, **{f"d {m}": f"{bs[m]['delta']:+.4f} [{bs[m]['ci95'][0]:+.4f}, {bs[m]['ci95'][1]:+.4f}]"
                                                       for m in ("multi segment F1@50", "native macro-F1", "multi boundary recall")},
                         "units fixed/new": f"{cu['fixed']}/{cu['new_errors']}", "recordings fixed/new": f"{cu['recordings_in_fixed']}/{cu['recordings_in_new_errors']}",
                         "natives fixed/new": f"{cn['fixed']}/{cn['new_errors']}", "consistent": v[h]["consistent_with_totals"]})
    sc = rep["start_conditions"]
    return "\n".join(["# Task 2: fixed equal-weight fusion\n", md_table([{"system": n, **{k: json.dumps(v) if isinstance(v, dict) else v for k, v in s.items()}}
                                                                          for n, s in rep["systems"].items()]),
                      "## Metrics\n", md_table(rows), "## Paired differences (bootstrap 95% intervals; streams reuse recordings)\n", md_table(comp),
                      "## Start conditions of task 3\n", "```\n" + json.dumps(sc, indent=1, default=str) + "\n```\n"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=("audit", "fusion"), required=False)
    ap.add_argument("-c", "--config", default="configs/experiments/ev9v_complementarity_v1.yaml")
    ap.add_argument("--write-spot-plans", action="store_true", help="write the fixed B7 spot-check subset and print the runner command")
    ap.add_argument("--print-digests", action="store_true", help="hash the input files the config pins (no result is computed)")
    args = ap.parse_args(); args.set = []; cfg = load_cfg(args); paths = load_paths()
    if args.print_digests:
        print(json.dumps(digests(cfg, paths), indent=1)); return
    if args.write_spot_plans:
        write_spot_plans(cfg, paths); return
    if args.phase is None:
        ap.error("--phase audit|fusion is required")
    {"audit": audit, "fusion": fusion}[args.phase](cfg, paths, args)


if __name__ == "__main__":
    main()
