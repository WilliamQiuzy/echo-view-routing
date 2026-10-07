#!/usr/bin/env python
"""Evaluate, freeze and report the fan224 visual adaptation round (plan Tasks 3-4;
configs/experiments/fan224_visual_adaptation_v1.yaml).

1. Checks. The twelve runs are done with their identities and final checkpoints. Per seed the four arms share the
   batch plan, the initial state and the per-update sample digests; L0 and L1 read identical pixels, and so do F0 and
   F1. The reference systems (E0 + B3, E1 + B3, B7 + B3, E1 + ASRF) reproduce from their frozen outputs.
2. Outputs. Every model reads every 10 Hz sample of both validation halves (native clips at gamma 1.0; the constructed
   streams with their recipes' variants) through its own input: L by the legacy path, F by the frozen fan224_v1
   transform applied to the source frame (the same function that wrote the training PNGs). The per-sample visual head
   gives the probabilities; no window or sparse centre fill. Rows: argmax, and B3 with the 40-setting grid chosen on
   calibration only and applied once to selection (the fixed 9/0.95 diagnostic when nothing is feasible). The
   unadapted E1 head on L and on F is an input-sensitivity control only.
3. Results. Native confusion, macro-F1, per-class counts, A4C/A5C errors, per-sample error, internal cuts, segment
   F1@50, boundary precision/recall/F1. Per seed F0 - L0, F1 - L1, F1 - F0, L1 - L0 and (F1 - F0) - (L1 - L0) with the
   paired bootstrap (2,000 draws, seed 20261002; native clips by session, streams by stream). Fixed and new native clips
   by recording and session; changed stream segments and their source recordings. The final goal for F1 + B3.
4. Freezing. The twelve models go through scripts/freeze_model.py with their plans and the config (--no-freeze skips).

--phase train-features writes, per trained model, the features and probabilities of every training-split 10 Hz sample
in both variants (server only; no metric). --dry-run uses training data and the initial states only and prints no
metric.

  python scripts/report_fan224_visual.py --config configs/experiments/fan224_visual_adaptation_v1.yaml \
      --out docs/results/2026-10-06_fan224_visual_adaptation
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg  # noqa: E402
from scripts.diagnose_center_recognition import native_bootstrap  # noqa: E402
from scripts.ev9v_complementarity import decode, md_table  # noqa: E402
from scripts.report_center_supervision import PAIRS, b3_choose, evaluate, native_changes, references, summary, validation  # noqa: E402
from scripts.report_model_development import BOOT_SEED, N_BOOT, metric_fns, paired_bootstrap  # noqa: E402
from scripts.train_fan224_visual import (ARMS, NS, TRAINED_ENCODER, build, check_inputs, identity, init_state,  # noqa: E402
                                         save_or_check_init, save_or_check_plan, train_table)
from scripts.train_joint_visual_temporal import FramePaths  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from echo_routing.config import gpu_memory_fraction, load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.evaluate.metrics_segmentation import segment_f1, temporal_iou  # noqa: E402
from echo_routing.ingest import fan224  # noqa: E402
from echo_routing.manifest import load_manifest  # noqa: E402
from echo_routing.temporal.dev_run import HALVES  # noqa: E402
from echo_routing.temporal.joint_visual import load_frame_u8  # noqa: E402
from echo_routing.temporal.segments import labels_to_segments  # noqa: E402

CONTRASTS = {"F0 - L0": ("F0", "L0"), "F1 - L1": ("F1", "L1"), "F1 - F0": ("F1", "F0"), "L1 - L0": ("L1", "L0")}
INTERACTION = {"F1": 1.0, "F0": -1.0, "L1": -1.0, "L0": 1.0}
CLASSES = ("PLAX", "PSAX", "A4C", "A5C", "SC4C")


# ----------------------------------------------------------------------------------------------- checks
def verify(cfg: dict, fv: dict, paths, status_path: Path) -> dict:
    root = paths.repo_root; inputs = check_inputs(fv, root); table = train_table(fv, paths)
    status = json.loads(status_path.read_text()); total = int(fv["sampling"]["updates"]); runs = {}
    for seed in [int(s) for s in fv["seeds"]]:
        plan_path, _ = save_or_check_plan(fv, table, seed, root); plan_sha = sha256_file(plan_path)
        enc, head = build(fv, paths, "L1", torch.device("cpu")); _, init_sha = save_or_check_init(init_state(enc, head), seed, root); del enc, head
        for arm in ARMS:
            key = f"{arm}_s{seed}"; st = status.get(key, {}); ident = identity(cfg, arm, seed, inputs, plan_sha, init_sha, root)
            man = json.loads((root / "runs" / NS / "train" / key / "run_manifest.json").read_text()); final = Path(man["final_checkpoint"])
            if not (st.get("state") == "done" and st.get("identity") == ident == man["identity"] and man["updates"] == total and man["frozen_unchanged"]
                    and sha256_file(final) == man["final_sha256"] == st.get("final_sha256") and man["plan_sha256"] == plan_sha and man["init_sha256"] == init_sha):
                raise SystemExit(f"{key}: not a complete, verified run")
            runs[key] = man
        same = {k: {runs[f"{a}_s{seed}"][k] for a in ARMS} for k in ("plan_sha256", "init_sha256", "update_digests_sha256")}
        if any(len(v) != 1 for v in same.values()):
            raise SystemExit(f"seed {seed}: the four arms do not share plan, initial state and sample digests")
        for x, y in (("L0", "L1"), ("F0", "F1")):
            if runs[f"{x}_s{seed}"]["pixel_digests_sha256"] != runs[f"{y}_s{seed}"]["pixel_digests_sha256"]:
                raise SystemExit(f"seed {seed}: {x} and {y} did not read identical pixels")
        if runs[f"L0_s{seed}"]["pixel_digests_sha256"] == runs[f"F0_s{seed}"]["pixel_digests_sha256"]:
            raise SystemExit(f"seed {seed}: L and F read identical pixels")
    return runs


def load_trained(fv: dict, paths, man: dict, device):
    enc, head = build(fv, paths, man["arm"], device); st = torch.load(man["final_checkpoint"], map_location="cpu", weights_only=False)
    with torch.no_grad():
        named = dict(enc.vt.named_parameters())
        for n, v in st["encoder_trainable"].items():
            if not n.startswith(TRAINED_ENCODER):
                raise SystemExit(f"{man['arm']}: checkpoint holds an unexpected encoder tensor {n}")
            named[n].copy_(v.to(device))
    head.load_state_dict(st["head"]); enc.eval(); head.eval()
    return enc, head


# ----------------------------------------------------------------------------------------------- outputs
class Frames(Dataset):
    """Chunks of (video, frame, gamma) references read through one input type, as uint8 (N, 224, 224, 3)."""

    def __init__(self, triples: list[tuple], frame_paths, kind: str, chunk: int = 256) -> None:
        if kind not in ("legacy", "fan224_v1"):
            raise ValueError(kind)
        self.t, self.fp, self.kind, self.chunk = triples, frame_paths, kind, chunk

    def __len__(self) -> int:
        return (len(self.t) + self.chunk - 1) // self.chunk

    def one(self, v: str, f: int, g: float) -> np.ndarray:
        p = self.fp(v); path = p[min(int(f), len(p) - 1)]
        if self.kind == "legacy":
            return load_frame_u8(path, g)
        with Image.open(path) as im:
            return fan224.fan224(im, g)

    def __getitem__(self, i: int) -> torch.Tensor:
        return torch.from_numpy(np.stack([self.one(v, f, g) for v, f, g in self.t[i * self.chunk:(i + 1) * self.chunk]]))


@torch.no_grad()
def extract(enc, triples: list[tuple], frame_paths, kind: str, device, workers: int) -> np.ndarray:
    enc.eval(); feats = []
    for u8 in DataLoader(Frames(triples, frame_paths, kind), batch_size=None, num_workers=workers, prefetch_factor=2 if workers else None):
        feats.append(enc(u8.to(device, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(feats) if feats else np.zeros((0, 512), np.float32)


@torch.no_grad()
def head_probs(head, feats: np.ndarray, device) -> np.ndarray:
    out = [torch.softmax(head(torch.from_numpy(feats[i:i + 8192]).to(device)), -1).cpu().numpy() for i in range(0, len(feats), 8192)]
    return np.concatenate(out) if out else np.zeros((0, 5), np.float32)


def per_sequence(probs: np.ndarray, at: dict, seqs_h: list) -> dict:
    return {q["id"]: probs[[at[r] for r in q["refs"]]] for q in seqs_h}


def sample_errors(preds: dict, seqs_h: list) -> dict:
    out = {}
    for kind in ("native", "pairs", "multi"):
        qs = [q for q in seqs_h if q["kind"] == kind]; n = sum(len(q["labels"]) for q in qs)
        out[kind] = float(sum(int((np.asarray(preds[q["id"]]) != q["labels"]).sum()) for q in qs) / max(n, 1))
    return out


def evaluate_model(cfg: dict, fv: dict, probs: dict, inputs, seqs: dict) -> tuple[dict, dict, dict]:
    rows, tables, preds_all = {}, {}, {}
    ch = b3_choose(cfg, fv, probs["calibration"], inputs.halves["calibration"])
    rows["b3_choice"] = {k: v for k, v in ch.items() if k != "grid"}; rows["b3_grid"] = ch["grid"]
    for row in ("argmax", "b3"):
        for h in HALVES:
            preds = {k: v.argmax(1) for k, v in probs[h].items()} if row == "argmax" else decode(ch["params"], probs[h])
            m, t = evaluate(cfg, preds, inputs.halves[h])
            rows[f"{row}|{h}"] = {**summary(m), "sample_error": sample_errors(preds, seqs[h]),
                                  "native_counts": {CLASSES[k]: [int(np.asarray(m["native_confusion"])[k, k]), int(np.asarray(m["native_confusion"])[k].sum())] for k in range(5)}}
            tables[f"{row}|{h}"], preds_all[f"{row}|{h}"] = t, preds
    return rows, tables, preds_all


# ----------------------------------------------------------------------------------------------- comparisons
def contrast_bootstrap(tabs: dict, weights: dict) -> dict:
    """sum_k w_k * metric(system k) with native clips resampled by session and streams by stream, the same draws for
    every system (paired_bootstrap's resampling, generalised to a linear contrast of several systems)."""
    names = list(weights); first = tabs[names[0]]
    for n in names[1:]:
        for t in ("multi_seg", "multi_frag"):
            if set(tabs[n][t]["stream"]) != set(first[t]["stream"]):
                raise ValueError("contrast bootstrap: systems cover different streams")
        if set(tabs[n]["native"]["video_id"]) != set(first["native"]["video_id"]):
            raise ValueError("contrast bootstrap: systems cover different native clips")
    rng = np.random.default_rng(BOOT_SEED); fns = metric_fns()
    nat0 = first["native"].assign(session=first["native"]["video_id"].map(session_of))
    nat = {n: tabs[n]["native"].set_index("video_id").loc[nat0["video_id"]].reset_index() for n in names}
    sessions = nat0["session"].unique(); by_session = {s: np.flatnonzero(nat0["session"].to_numpy() == s) for s in sessions}
    streams = first["multi_seg"]["stream"].unique()
    idx = {(n, t): {s: np.flatnonzero(tabs[n][t]["stream"].to_numpy() == s) for s in streams} for n in names for t in ("multi_seg", "multi_frag")}
    point = {k: sum(w * fns[k][1](tabs[n][fns[k][0]] if fns[k][0] != "native" else nat[n]) for n, w in weights.items()) for k in fns}
    draws = {k: [] for k in fns}
    for _ in range(N_BOOT):
        ss = rng.choice(sessions, size=sessions.size, replace=True); st = rng.choice(streams, size=streams.size, replace=True)
        ni = np.concatenate([by_session[s] for s in ss])
        res = {n: {"native": nat[n].iloc[ni],
                   "multi_seg": tabs[n]["multi_seg"].iloc[np.concatenate([idx[(n, "multi_seg")][s] for s in st])],
                   "multi_frag": tabs[n]["multi_frag"].iloc[np.concatenate([idx[(n, "multi_frag")][s] for s in st])]} for n in names}
        for k, (t, f) in fns.items():
            draws[k].append(sum(w * f(res[n][t]) for n, w in weights.items()))
    return {k: {"delta": float(point[k]), "ci95": [float(np.percentile(draws[k], 2.5)), float(np.percentile(draws[k], 97.5))]} for k in fns}


def matched_true(pred: list, true: list, thr: float = 0.5) -> set:
    """Indices of true segments matched by segment_f1's one-to-one rule (same order of candidates)."""
    cands = sorted(((temporal_iou(p, t), i, j) for i, p in enumerate(pred) for j, t in enumerate(true)
                    if p.label == t.label and temporal_iou(p, t) >= thr), reverse=True)
    used_p, used_t = set(), set()
    for _, i, j in cands:
        if i not in used_p and j not in used_t:
            used_p.add(i); used_t.add(j)
    if len(used_t) != segment_f1(pred, true, thr)["matched"]:
        raise AssertionError("per-segment matching disagrees with segment_f1")
    return used_t


def segment_changes(preds_a: dict, preds_b: dict, seqs_h: list) -> dict:
    """Constructed multi-view stream segments matched by a but not b (fixed) and by b but not a (new), with the distinct
    source recordings and sessions behind them (streams reuse recordings, so segments are not independent cases)."""
    out = {"fixed": {"segments": 0, "recordings": set(), "sessions": set()}, "new": {"segments": 0, "recordings": set(), "sessions": set()}}
    for q in seqs_h:
        if q["kind"] != "multi":
            continue
        true = labels_to_segments(q["labels"]); ma = matched_true(labels_to_segments(np.asarray(preds_a[q["id"]])), true)
        mb = matched_true(labels_to_segments(np.asarray(preds_b[q["id"]])), true)
        for name, js in (("fixed", ma - mb), ("new", mb - ma)):
            for j in js:
                src = {q["refs"][i][0] for i in range(true[j].start, true[j].end)}
                out[name]["segments"] += 1; out[name]["recordings"] |= src; out[name]["sessions"] |= {session_of(v) for v in src}
    return {k: {"segments": v["segments"], "recordings": len(v["recordings"]), "sessions": len(v["sessions"])} for k, v in out.items()}


def compare(ta: dict, tb: dict, pa: dict, pb: dict, seqs: dict) -> dict:
    out = {}
    for row in ("b3", "argmax"):
        for h in HALVES:
            a, b = ta[f"{row}|{h}"], tb[f"{row}|{h}"]; ch = native_changes(a["native"], b["native"])
            out[f"{row}|{h}"] = {"bootstrap": paired_bootstrap(a, b), "native_session": native_bootstrap(a["native"], b["native"], N_BOOT, BOOT_SEED),
                                 "native_changes": {k: ({"fixed": len(v["fixed"]), "new": len(v["new"]), "fixed_sessions": v["fixed_sessions"], "new_sessions": v["new_sessions"]}
                                                        if isinstance(v.get("fixed"), list) else v) for k, v in ch.items()},
                                 "segment_changes": segment_changes(pa[f"{row}|{h}"], pb[f"{row}|{h}"], seqs[h]),
                                 "_ids": {k: {"fixed": v["fixed"], "new": v["new"]} for k, v in ch.items() if isinstance(v.get("fixed"), list)}}
    return out


def goal(fv: dict, rows: list[dict]) -> dict:
    g = fv["goal"]; floors = {"selection": 0.973, "calibration": 0.980}
    mean = {h: float(np.mean([r[f"b3|{h}"]["segment_f1_50_multi"] for r in rows])) for h in HALVES}
    per_seed = [{h: {"feasible": r["b3_choice"]["status"] == "feasible", "recall": r[f"b3|{h}"]["boundary_recall_multi"],
                     "native": r[f"b3|{h}"]["native_macro_f1"]} for h in HALVES} for r in rows]
    ok = all(p[h]["feasible"] and (p[h]["recall"] or 0) >= 0.95 and p[h]["native"] >= floors[h] for p in per_seed for h in HALVES)
    return {"mean_segment_f1": mean, "required": {"selection": g["selection_min"], "calibration": g["calibration_min"]},
            "margin": {"selection": mean["selection"] - float(g["selection_min"]), "calibration": mean["calibration"] - float(g["calibration_min"])},
            "per_seed": per_seed, "met": bool(ok and mean["selection"] >= float(g["selection_min"]) and mean["calibration"] >= float(g["calibration_min"])),
            "sensitivity_calibration": mean["calibration"] - (float(g["calibration_sensitivity"]) + 0.01)}


def strip_ids(obj):
    if isinstance(obj, dict):
        return {k: strip_ids(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [strip_ids(v) for v in obj]
    return obj


# ----------------------------------------------------------------------------------------------- freezing
def freeze(paths, runs: dict, config: str) -> dict:
    out = {}
    for key, man in runs.items():
        files = [man["final_checkpoint"], str(paths.repo_root / "runs" / NS / "train" / key / "run_manifest.json"), man["plan"], man["init"],
                 str(paths.repo_root / config)]
        dest = paths.repo_root / "models_frozen" / NS / key
        if not dest.exists():
            subprocess.run([sys.executable, "scripts/freeze_model.py", NS, key, *sum((["--file", f] for f in files), []), "--meta", f"identity={man['identity']}",
                            "--meta", f"arm={man['arm']}", "--meta", f"seed={man['seed']}", "--meta", f"input={man['input_kind']}",
                            "--notes", "Fan224 visual adaptation round (2026-10-06); final update."], cwd=paths.repo_root, check=True, capture_output=True)
        out[key] = str(dest)
    return out


# ----------------------------------------------------------------------------------------------- main
def model_list(runs: dict) -> list[tuple[str, dict | None, str]]:
    """(name, run manifest or None for the unadapted control, input kind)."""
    out = [("E1 on L (unadapted)", None, "legacy"), ("E1 on F (unadapted)", None, "fan224_v1")]
    for seed in sorted({m["seed"] for m in runs.values()}):
        for arm in ARMS:
            m = runs[f"{arm}_s{seed}"]; out.append((f"{arm}_s{seed}", m, m["input_kind"]))
    return out


def check_b3_grid(cfg: dict, fv: dict) -> None:
    """The grid b3_choose actually searches (cfg['decode_grids']['b3']) must be the one this round's config states."""
    used, stated = cfg["decode_grids"]["b3"], fv["evaluation"]["b3_grid"]
    if ([int(x) for x in used["half_window"]] != [int(x) for x in stated["half_window"]]
            or [float(x) for x in used["threshold_quantile"]] != [float(x) for x in stated["quantile"]] or [int(x) for x in used["nms_radius"]] != [int(stated["nms_radius"])]):
        raise SystemExit(f"the B3 grid in use {used} differs from the round's stated grid {stated}")


def preflight_sources(triples: list[tuple], frame_paths) -> dict:
    """Every referenced validation source frame exists and is 320 x 240 (the fan224 contract), before any model runs."""
    bad, n = [], 0
    for v, f in sorted({(t[0], int(t[1])) for t in triples}):
        p = frame_paths(v); path = p[min(f, len(p) - 1)]; n += 1
        with Image.open(path) as im:
            if im.size != fan224.SOURCE_SIZE:
                bad.append(f"{v}:{f} {im.size}")
    if bad:
        raise SystemExit(f"{len(bad)} validation source frames are not 320 x 240, e.g. {bad[:3]}")
    return {"source_frames": n}


def cached_model_eval(path: Path, key: dict, compute):
    """Rows, tables and predictions of one model, reused on a rerun only when the model and the frame list are the same."""
    import pickle
    if path.is_file():
        with path.open("rb") as fh:
            blob = pickle.load(fh)
        if blob.get("key") == key:
            return blob["value"], True
    value = compute()
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        pickle.dump({"key": key, "value": value}, fh, protocol=4)
    tmp.replace(path)
    return value, False


def write_outputs(res: dict, details: Path, out: Path) -> None:
    (details / "details.json").write_text(json.dumps(res, indent=1, default=str))
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(strip_ids(res), indent=1, default=str)); (out / "report_tables.md").write_text(tables_md(strip_ids(res)))


def run_evaluate(a, cfg: dict, fv: dict, paths, device) -> None:
    root = paths.repo_root; check_b3_grid(cfg, fv); runs = verify(cfg, fv, paths, root / a.status); print("twelve runs verified and paired", flush=True)
    inputs, seqs = validation(cfg, fv, paths)
    triples = sorted({r for h in HALVES for q in seqs[h] for r in q["refs"]}); at = {t: i for i, t in enumerate(triples)}
    man = load_manifest(paths.manifest_path).set_index("video_id"); fp = FramePaths({v: str(man.loc[v, "frames_dir"]) for v in {t[0] for t in triples}})
    pre = preflight_sources(triples, fp); print(f"preflight: {pre['source_frames']} validation source frames are 320 x 240", flush=True)
    details = root / a.details; (details / "eval_cache").mkdir(parents=True, exist_ok=True); out = root / a.out
    triples_digest = hashlib.sha256(json.dumps(triples).encode()).hexdigest()
    models, tables, preds, timing = {}, {}, {}, {}
    for name, run, kind in model_list(runs):
        t0 = time.time(); safe = name.split(" ")[0] if run else name.replace(" ", "_").replace("(", "").replace(")", "")

        def compute(run=run, kind=kind, safe=safe):
            if run is None:
                enc, head = build(fv, paths, "L0", device); enc.eval(); head.eval()
            else:
                enc, head = load_trained(fv, paths, run, device)
            t_x = time.time(); feats = extract(enc, triples, fp, kind, device, a.workers); x_s = time.time() - t_x
            p = head_probs(head, feats, device); probs = {h: per_sequence(p, at, seqs[h]) for h in HALVES}
            rows, tabs, pr = evaluate_model(cfg, fv, probs, inputs, seqs)
            np.savez_compressed(details / f"probs_{safe}.npz", **{f"{h}|{k}": v.astype(np.float32) for h in HALVES for k, v in probs[h].items()})
            del enc, head, feats; torch.cuda.empty_cache()
            return {"rows": rows, "tables": tabs, "preds": pr, "extract_s": x_s}
        key = {"model": name, "final_sha256": run["final_sha256"] if run else None, "input": kind, "triples": triples_digest,
               "transform_digest": fv["input_types"]["F"]["transform_digest"]}
        value, reused = cached_model_eval(details / "eval_cache" / f"{safe}.pkl", key, compute)
        rows = value["rows"]; models[name], tables[name], preds[name] = {**rows, "input": kind}, value["tables"], value["preds"]
        timing[name] = {"extract_s": value["extract_s"], "total_s": time.time() - t0, "frames": len(triples), "reused": reused}
        print(f"{name}: B3 {rows['b3_choice']['status']} | sel {rows['b3|selection']['segment_f1_50_multi']:.4f} cal {rows['b3|calibration']['segment_f1_50_multi']:.4f} "
              f"| native argmax {rows['argmax|selection']['native_macro_f1']:.4f} / {rows['argmax|calibration']['native_macro_f1']:.4f} ({time.time() - t0:.0f}s{', reused' if reused else ''})", flush=True)
    chk = same_bootstrap(paired_bootstrap(tables["F1_s0"]["b3|selection"], tables["F0_s0"]["b3|selection"]),
                         contrast_bootstrap({"F1": tables["F1_s0"]["b3|selection"], "F0": tables["F0_s0"]["b3|selection"]}, {"F1": 1.0, "F0": -1.0}))
    if not all(chk.values()):
        raise SystemExit(f"contrast_bootstrap does not reproduce paired_bootstrap: {chk}")
    comps = {}
    for seed in [int(s) for s in fv["seeds"]]:
        for name, (x, y) in CONTRASTS.items():
            kx, ky = f"{x}_s{seed}", f"{y}_s{seed}"
            comps[f"s{seed}|{name}"] = compare(tables[kx], tables[ky], preds[kx], preds[ky], seqs)
        comps[f"s{seed}|(F1 - F0) - (L1 - L0)"] = {f"{row}|{h}": {"bootstrap": contrast_bootstrap({arm: tables[f"{arm}_s{seed}"][f"{row}|{h}"] for arm in INTERACTION},
                                                                                                    INTERACTION)} for row in ("b3", "argmax") for h in HALVES}
    res = {"models": models, "comparisons": comps, "goal_F1": goal(fv, [models[f"F1_s{s}"] for s in fv["seeds"]]),
           **{f"goal_{arm}": goal(fv, [models[f"{arm}_s{s}"] for s in fv["seeds"]]) for arm in ("F0", "L1", "L0")},
           "references": "pending", "frozen": "pending", "timing": timing, "preflight": pre,
           "runs": {k: {x: v[x] for x in ("identity", "arm", "seed", "input_kind", "final_sha256", "plan_sha256", "init_sha256", "updates", "update_digests_sha256",
                                         "pixel_digests_sha256", "transform_digest")} | {"log": v["log"]} for k, v in runs.items()}}
    write_outputs(res, details, out); print(f"wrote the model results to {out} (references and freezing pending)", flush=True)
    refs = references({**cfg, "center_supervision": {"inputs": fv["inputs"]}}, paths, inputs, seqs); print("references reproduced", flush=True)
    res["references"] = {k: {h: v[h]["metrics"] for h in HALVES} for k, v in refs.items()}; write_outputs(res, details, out)
    res["frozen"] = {} if a.no_freeze else freeze(paths, runs, a.config); write_outputs(res, details, out)
    print(json.dumps(res["goal_F1"], indent=1, default=str)); print(f"wrote {out}")


def run_train_features(a, cfg: dict, fv: dict, paths, device) -> None:
    """Features (float16) and probabilities of every training-split 10 Hz sample in both variants, per trained model."""
    root = paths.repo_root; runs = verify(cfg, fv, paths, root / a.status); table = train_table(fv, paths)
    triples = [(v, int(f), g) for v, fr in zip(table["videos"], table["frames"]) for g in (1.0, 0.9) for f in fr]
    fp = FramePaths(table["frames_dir"]); out_root = root / "cache_v2" / "fan224_visual_adaptation_v1" / "train"
    for name, run, kind in model_list(runs):
        if run is None:
            continue
        dest = out_root / name; done = dest / "index.json"
        if done.is_file() and json.loads(done.read_text()).get("final_sha256") == run["final_sha256"]:
            print(f"{name}: train features exist, skipped", flush=True); continue
        dest.mkdir(parents=True, exist_ok=True); t0 = time.time(); enc, head = load_trained(fv, paths, run, device)
        feats = extract(enc, triples, fp, kind, device, a.workers); p = head_probs(head, feats, device); k = 0
        for v, fr in zip(table["videos"], table["frames"]):
            n = len(fr); arr = {}
            for variant in ("orig", "gamma090"):
                arr[f"{variant}_feat"] = feats[k:k + n].astype(np.float16); arr[f"{variant}_prob"] = p[k:k + n].astype(np.float32); k += n
            np.savez(dest / f"{v}.npz", frame_idx=fr, **arr)
        identity_meta = {"model": name, "identity": run["identity"], "final_sha256": run["final_sha256"], "input": kind,
                         "transform_digest": run["transform_digest"], "frame_order": "numeric (cache_v2)", "recordings": len(table["videos"]),
                         "samples": int(k), "variants": ["orig", "gamma090"], "seconds": time.time() - t0}
        done.write_text(json.dumps(identity_meta, indent=1)); print(json.dumps(identity_meta), flush=True)
        del enc, head, feats; torch.cuda.empty_cache()


def dry_run(cfg: dict, fv: dict, paths, device, workers: int) -> None:
    """Training data only: a training subset (natives of every class, two pair and two multi streams) used as both
    'halves', the unadapted E1 head on both inputs, and every evaluation step (argmax, B3 choice and decoding,
    comparisons, the interaction bootstrap, segment changes). Only structure is printed, no metric."""
    from types import SimpleNamespace

    from echo_routing.compose.banks import multi_bank
    from echo_routing.compose.stream_assembler import render
    from echo_routing.features.cache import read_index, variant_dir
    from echo_routing.temporal.dev_data import EvalSet, FeatureStore, native_seq
    from echo_routing.temporal.windowed import stream_plan
    key = fv["inputs"]["cache_key"]; store = FeatureStore(paths.cache_root, key)
    idx = read_index(variant_dir(paths.cache_root, key, "orig")); labels = dict(zip(idx["video_id"], idx["label_index"].astype(int)))
    table = train_table(fv, paths); clips = [v for c in range(5) for v in [x for x, y in zip(table["videos"], table["labels"]) if y == c][:3]]
    recipes = multi_bank(idx, "train", cfg).recipes[:4]
    natives = [native_seq(v, labels[v], store.get(v, "orig")) for v in clips]; streams = [render(r, store.get) for r in recipes]
    ev = EvalSet("train-subset", tuple(natives), tuple(streams[:2]), tuple(streams[2:]))
    seqs_h = [{"id": v, "kind": "native", "labels": s.labels.astype(int), "refs": [(v, int(f), 1.0) for f in store.get(v, "orig").frame_idx]} for v, s in zip(clips, natives)]
    seqs_h += [{"id": r.recipe_id, "kind": kind, "labels": st.labels.astype(int), "refs": [(x.video_id, int(x.frame_idx), float(x.gamma)) for x in stream_plan(r, store.get)]}
               for r, st, kind in zip(recipes, streams, ("pairs", "pairs", "multi", "multi"))]
    seqs = {h: seqs_h for h in HALVES}; inputs = SimpleNamespace(halves={h: ev for h in HALVES})
    triples = sorted({r for q in seqs_h for r in q["refs"]}); at = {t: i for i, t in enumerate(triples)}
    man = load_manifest(paths.manifest_path).set_index("video_id"); fp = FramePaths({v: str(man.loc[v, "frames_dir"]) for v in {t[0] for t in triples}})
    enc, head = build(fv, paths, "L0", device); enc.eval(); head.eval(); out, tabs, preds = {"frames": len(triples)}, {}, {}
    for kind in ("legacy", "fan224_v1"):
        f = extract(enc, triples, fp, kind, device, workers); p = head_probs(head, f, device)
        probs = {h: per_sequence(p, at, seqs_h) for h in HALVES}
        rows, tabs[kind], preds[kind] = evaluate_model(cfg, fv, probs, inputs, seqs)
        out[kind] = {"feature_shape": list(f.shape), "probabilities_sum_to_one": bool(np.allclose(p.sum(1), 1.0, atol=1e-5)),
                     "b3_settings": len(rows["b3_grid"]), "rows": sorted(k for k in rows if "|" in k)}
    c = compare(tabs["fan224_v1"], tabs["legacy"], preds["fan224_v1"], preds["legacy"], seqs)
    i = contrast_bootstrap({"F1": tabs["fan224_v1"]["b3|selection"], "F0": tabs["legacy"]["b3|selection"],
                            "L1": tabs["legacy"]["b3|selection"], "L0": tabs["fan224_v1"]["b3|selection"]}, INTERACTION)
    pb = paired_bootstrap(tabs["fan224_v1"]["b3|selection"], tabs["legacy"]["b3|selection"])
    cb = contrast_bootstrap({"a": tabs["fan224_v1"]["b3|selection"], "b": tabs["legacy"]["b3|selection"]}, {"a": 1.0, "b": -1.0})
    out["compare_keys"] = sorted(c); out["interaction_metrics"] = sorted(i)
    out["contrast_vs_paired_bootstrap"] = same_bootstrap(pb, cb)
    print(json.dumps({"dry_run": "ok (training data only, no metric)", **out}, indent=1))


def same_bootstrap(pb: dict, cb: dict) -> dict:
    """Per metric: does the generalised contrast reproduce paired_bootstrap (NaN-aware; tiny sets can give 0/0)?"""
    return {k: bool(np.isclose(pb[k]["delta"], cb[k]["delta"], rtol=0, atol=1e-12, equal_nan=True)
                    and np.allclose(pb[k]["ci95"], cb[k]["ci95"], rtol=0, atol=1e-12, equal_nan=True)) for k in pb}


def tables_md(res: dict) -> str:
    rows = []
    for key, r in res["models"].items():
        for row in ("b3", "argmax"):
            for h in HALVES:
                m = r[f"{row}|{h}"]
                rows.append({"model": key, "row": row, "half": h, "B3": r["b3_choice"]["status"] if row == "b3" else "", "segment F1@50": m["segment_f1_50_multi"],
                             "boundary P": m["boundary_precision_multi"], "boundary R": m["boundary_recall_multi"], "native macro-F1": m["native_macro_f1"],
                             "cuts per clip": m["native_internal_cuts_per_clip"], "native sample error": m["sample_error"]["native"],
                             "multi sample error": m["sample_error"]["multi"], **{p: m[p] for p in PAIRS}})
    comp = []
    for k, v in res["comparisons"].items():
        for rh, c in v.items():
            b = c["bootstrap"]
            rec = {"comparison": k, "row": rh, **{f"d {m}": f"{b[m]['delta']:+.4f} [{b[m]['ci95'][0]:+.4f}, {b[m]['ci95'][1]:+.4f}]"
                                                  for m in ("multi segment F1@50", "multi boundary recall", "native macro-F1")}}
            if "native_changes" in c:
                rec["native fixed/new"] = f"{c['native_changes']['all clips']['fixed']}/{c['native_changes']['all clips']['new']}"
                rec["segments fixed/new (recordings)"] = (f"{c['segment_changes']['fixed']['segments']} ({c['segment_changes']['fixed']['recordings']}) / "
                                                          f"{c['segment_changes']['new']['segments']} ({c['segment_changes']['new']['recordings']})")
            comp.append(rec)
    refs = res["references"] if isinstance(res["references"], dict) else {}
    ref = [{"system": k, "half": h, "segment F1@50": v[h]["segment_f1_50_multi"], "boundary R": v[h]["boundary_recall_multi"], "native macro-F1": v[h]["native_macro_f1"]}
           for k, v in refs.items() for h in HALVES] or [{"system": "references pending", "half": "", "segment F1@50": "", "boundary R": "", "native macro-F1": ""}]
    return "\n".join(["# Fan224 visual adaptation round: tables\n", "## Models\n", md_table(rows), "## Comparisons\n", md_table(comp),
                      "## References\n", md_table(ref), "## Goal (F1 + B3)\n", "```\n" + json.dumps(res["goal_F1"], indent=1, default=str) + "\n```\n"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--status", default=f"runs/{NS}/status.json"); ap.add_argument("--out", default=None)
    ap.add_argument("--details", default=f"runs/{NS}/report"); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--phase", choices=["evaluate", "train-features"], default="evaluate")
    ap.add_argument("--no-freeze", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); fv = cfg["fan224"]; paths = load_paths()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    if a.dry_run:
        dry_run(cfg, fv, paths, device, a.workers); return
    if a.phase == "train-features":
        run_train_features(a, cfg, fv, paths, device); return
    if not a.out:
        raise SystemExit("--out is required for the evaluation")
    run_evaluate(a, cfg, fv, paths, device)


if __name__ == "__main__":
    main()
