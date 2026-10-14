#!/usr/bin/env python
"""Frozen trained time-token readout (docs/superpowers/plans/2026-10-14-trained-token-readout.md;
configs/experiments/trained_token_readout_v1.yaml). Nothing is trained and no model is loaded: A02's saved per-window
outputs are read. Each window keeps its request c; a position-only rule picks one of the 8 trained time anchors, and that
anchor's saved time logits replace the CLS logits as the window's prediction at c.

Stages (each appends its wall-clock hours to the round's ledger, failed attempts included):
  --prepare    pins, environment, class order, the allowlist metadata of both halves, the windows of all 18 saved
               prediction files (identity, final weights, sequence order, centres, markers) and the anchor positions
  --check      A02's CLS probabilities and every A02 CLS row (argmax, B3, fixed Ordered setting) and the frozen Ordered
               rows reproduced exactly from the saved outputs and the metadata only
  --calibrate  the calibration half only: every 40-setting B3 result and the single choice of each token system
  --selection  after the committed choices: the chosen settings applied once, the comparisons and the fixed readings

The metadata adapter reads, for one half at a time, only that half's protocol entries, the clip index rows of that half's
clips (labels parsed for those rows only) and the pinned window plans of that half's sequences. Streams take their
labels and joins from their recipes. No feature, probability cache, training label or other-split file is read; the
validation entry, the Inputs class and A02's evaluate entry are not used.

  python scripts/trained_token_readout.py --config configs/experiments/trained_token_readout_v1.yaml --prepare
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.raw_video_context_tables as rt  # noqa: E402
import scripts.report_center_supervision as rcs  # noqa: E402
import scripts.report_raw_video_context as rr  # noqa: E402
import scripts.report_target_patch_readout as rtp  # noqa: E402
import scripts.target_patch_readout_tables as pt  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.ev9v_complementarity import decode  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402
from scripts.report_fold_output_correction import FLAT, boundary_changes, native_changes, segment_changes, summary, write_atomic  # noqa: E402
from scripts.report_model_development import paired_bootstrap  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

import echo_routing.vit.target_mvit as tm  # noqa: E402
from echo_routing.compose.recipe_builder import recipe_from_dict  # noqa: E402
from echo_routing.compose.stream_assembler import Stream  # noqa: E402
from echo_routing.config import load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.evaluate.window_target import window_frames  # noqa: E402
from echo_routing.temporal.dev_data import EvalSet, Seq  # noqa: E402
from echo_routing.temporal.windowed import window_centres  # noqa: E402

ANCHORS = (0, 2, 4, 6, 8, 10, 12, 14)
ARMS = ("center_only", "aligned_dense", "shuffled_dense")
HALVES = ("selection", "calibration")
CLASS_ORDER = ("PLAX", "PSAX", "A4C", "A5C", "SC4C")
MAPPING = "family5_v1"
DIAG = {"evaluation": {"b3_infeasible_diagnostic": {"half_window": 9, "quantile": 0.95}}}
HELPER_FILES = ("scripts/report_center_supervision.py", "scripts/ev9v_complementarity.py", "scripts/calibrate_simple_decoders.py",
                "scripts/report_raw_video_context.py", "scripts/report_target_patch_readout.py", "scripts/target_patch_readout_tables.py",
                "scripts/raw_video_context_tables.py", "scripts/report_fold_output_correction.py", "scripts/report_model_development.py",
                "echo_routing/evaluate/dev_metrics.py", "echo_routing/evaluate/complementarity.py", "echo_routing/evaluate/window_target.py",
                "echo_routing/temporal/windowed.py", "echo_routing/compose/stream_assembler.py", "scripts/trained_token_readout_tables.py")


# ----------------------------------------------------------------------------------------------- the selector
def marker_of(s, c) -> int:
    return tm.marker_slot(np.asarray(s), int(c))


def selected_anchor(s, c, marker) -> int:
    """The unique anchor index j minimising (|s[a_j] - c|, |a_j - marker|, j); positions only."""
    s = np.asarray(s)
    if len(s) != 16 or not 0 <= int(marker) < 16 or int(s[int(marker)]) != int(c):
        raise ValueError("window/marker mismatch")
    return min(range(8), key=lambda j: (abs(int(s[ANCHORS[j]]) - int(c)), abs(ANCHORS[j] - int(marker)), j))


def select_all(S, C, M) -> np.ndarray:
    return np.array([selected_anchor(s, c, m) for s, c, m in zip(S, C, M)], dtype=np.int64)


def token_logits(time_logits: np.ndarray, j: np.ndarray) -> np.ndarray:
    return np.asarray(time_logits)[np.arange(len(j)), np.asarray(j)]


def anchor_offsets(S, C, j) -> np.ndarray:
    S, j = np.asarray(S), np.asarray(j)
    return S[np.arange(len(j)), np.asarray(ANCHORS)[j]].astype(np.int64) - np.asarray(C, dtype=np.int64)


# ----------------------------------------------------------------------------------------------- source checks
def pinned(root, pin: dict) -> Path:
    path = Path(root) / pin["path"]
    if (got := sha256_file(path)) != pin["sha256"]:
        raise SystemExit(f"{pin['path']}: sha256 {got[:12]} differs from the pin {pin['sha256'][:12]}")
    return path


def check_identity(got: str, want: str, what: str) -> None:
    if str(got) != str(want):
        raise SystemExit(f"{what}: identity {str(got)[:12]} differs from {str(want)[:12]}")


def check_class_order(classes, mapping: str) -> None:
    if tuple(classes) != CLASS_ORDER or mapping != MAPPING:
        raise SystemExit(f"class order {tuple(classes)} / mapping {mapping} differ from {CLASS_ORDER} / {MAPPING}")


def check_windows(W: dict, seq_ids: list, saved: dict, what: str) -> None:
    if list(np.asarray(saved["sequence_ids"]).tolist()) != list(seq_ids):
        raise SystemExit(f"{what}: the saved sequence list differs from the metadata")
    for k in ("seq", "centre", "marker"):
        if not np.array_equal(np.asarray(saved[k]), np.asarray(W[k])):
            raise SystemExit(f"{what}: the saved window {k} differs from the metadata")


# ----------------------------------------------------------------------------------------------- the metadata adapter
def half_index(path, clips) -> pd.DataFrame:
    """video_id, split and n_samples of every row first; label_index parsed only for the given validation clips."""
    meta = pd.read_csv(path, usecols=["video_id", "split", "n_samples"], dtype={"video_id": str, "split": str})
    pos = {v: i for i, v in enumerate(meta["video_id"].tolist())}
    missing = [c for c in clips if c not in pos]
    if missing:
        raise SystemExit(f"{len(missing)} clips of the half are not in the clip index")
    rows = [pos[c] for c in clips]
    if (meta["split"].to_numpy()[rows] != "validation").any():
        raise SystemExit("a clip of the half is not a validation clip")
    keep = set(rows)
    lab = pd.read_csv(path, usecols=["video_id", "label_index"], dtype={"video_id": str}, skiprows=lambda i: i > 0 and (i - 1) not in keep)
    if lab["video_id"].tolist() != meta["video_id"].to_numpy()[sorted(keep)].tolist():
        raise SystemExit("the label rows do not line up with the half's clips")
    labels = dict(zip(lab["video_id"], lab["label_index"].astype(int)))
    return pd.DataFrame({"video_id": list(clips), "n_samples": meta["n_samples"].to_numpy()[rows].astype(int), "label_index": [labels[c] for c in clips]})


def meta_stream(d: dict, n_of: dict, label_of: dict) -> Stream:
    """A stream from its recipe alone: labels, joins and semantic boundaries; no prob or feature arrays."""
    recipe = recipe_from_dict(d); labels, joins, pos = [], [], 0
    for i, fr in enumerate(recipe.fragments):
        if fr.video_id not in n_of:
            raise SystemExit(f"{recipe.recipe_id}: a fragment comes from outside the half's clips")
        if not 0 <= fr.start < fr.end <= n_of[fr.video_id] or int(fr.label) != int(label_of[fr.video_id]):
            raise SystemExit(f"{recipe.recipe_id}: a fragment exceeds its clip or carries another label")
        labels.append(np.full(fr.end - fr.start, int(fr.label), dtype=int))
        if i > 0:
            joins.append(pos)
        pos += fr.end - fr.start
    lab = np.concatenate(labels); jn = np.array(joins, dtype=int)
    return Stream(recipe, None, None, lab, jn, np.array([j for j in jn if lab[j] != lab[j - 1]], dtype=int))


def half_sets(proto: dict, half: str, index_path, plans: dict) -> tuple[EvalSet, list[dict]]:
    """One development half: a metadata-only EvalSet and its sequences (native clips, pair streams, multi streams)."""
    v = proto["validation"][half]; clips = list(v["clips"]); idx = half_index(index_path, clips)
    n_of = dict(zip(idx["video_id"], idx["n_samples"].astype(int))); label_of = dict(zip(idx["video_id"], idx["label_index"].astype(int)))
    natives = tuple(Seq(c, True, None, np.full(n_of[c], label_of[c], np.int64), np.zeros(0, np.int64)) for c in clips)
    pairs = tuple(meta_stream(r, n_of, label_of) for r in v["banks"]["pairs"]["recipes"])
    multi = tuple(meta_stream(r, n_of, label_of) for r in v["banks"]["multi"]["recipes"])
    seqs = []
    for kind, items in (("native", [(s.seq_id, s.labels) for s in natives]), ("pairs", [(s.recipe.recipe_id, s.labels) for s in pairs]),
                        ("multi", [(s.recipe.recipe_id, s.labels) for s in multi])):
        for sid, lab in items:
            p = plans["streams"][sid]
            if int(p["n"]) != len(lab) or len(p["refs"]) != len(lab):
                raise SystemExit(f"{sid}: the window plan's length differs from the metadata")
            seqs.append({"id": sid, "kind": kind, "labels": np.asarray(lab, int), "refs": [(str(r[0]), int(r[1]), float(r[2])) for r in p["refs"]]})
    return EvalSet(half, natives, pairs, multi), seqs


def half_windows(seqs_h: list[dict], stride: int = 5) -> dict:
    """Every window of a half: sequence index, request c, sample positions s, marker slot and reference labels."""
    seq, cen, ss, mk, lab, lab16 = [], [], [], [], [], []
    for qi, q in enumerate(seqs_h):
        n = len(q["labels"]); labels = np.asarray(q["labels"], int)
        for c in window_centres(n, stride):
            s = np.asarray(window_frames(int(c), n, 16, 16))
            seq.append(qi); cen.append(int(c)); ss.append(s); mk.append(marker_of(s, int(c))); lab.append(int(labels[c])); lab16.append(labels[s])
    return {"seq": np.asarray(seq), "centre": np.asarray(cen), "s": np.asarray(ss), "marker": np.asarray(mk), "label": np.asarray(lab), "labels16": np.asarray(lab16)}


# ----------------------------------------------------------------------------------------------- the round's context
class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.cfg, self.config, self.tt = cfg, config, cfg["trained_token_readout"]
        self.paths = load_paths(); self.root = self.paths.repo_root; self.src = self.tt["source"]
        self.out = self.root / self.tt["outputs"]["root"]; self.results = self.root / self.tt["outputs"]["results_dir"]
        self.ledger = self.root / self.tt["budget"]["ledger"]; self._proto = self._plans = None; self._a02 = None

    def proto(self) -> dict:
        if self._proto is None:
            self._proto = json.loads(pinned(self.root, self.src["protocol"]).read_text())
            check_class_order(rcs.CLASSES, self._proto["mapping_version"])
        return self._proto

    def plans(self) -> dict:
        if self._plans is None:
            self._plans = json.loads(pinned(self.root, self.src["window_plans"]).read_text())
        return self._plans

    def a02(self) -> dict:
        if self._a02 is None:
            self._a02 = json.loads(pinned(self.root, self.src["a02_summary"]).read_text())
        return self._a02

    def half(self, h: str) -> tuple[EvalSet, list[dict], dict]:
        ev, seqs = half_sets(self.proto(), h, self.root / self.src["clip_index"]["path"], self.plans())
        return ev, seqs, half_windows(seqs)

    def prediction(self, h: str, key: str, W: dict, seqs_h: list[dict]) -> dict:
        """A02's saved outputs of one model and half, checked: pin, run identity, final weights, windows, shapes, values."""
        path = pinned(self.root, self.src["predictions"][h][key]); run = self.a02()["runs"][key]
        pinned(self.root, self.src["final_weights"][key])
        if self.src["final_weights"][key]["sha256"] != run["final_sha256"]:
            raise SystemExit(f"{key}: the pinned final weights differ from A02's run record")
        with np.load(path) as z:
            d = {k: z[k] for k in z.files}
        check_identity(str(d["identity"]), run["identity"], f"{key} {h}")
        check_windows(W, [q["id"] for q in seqs_h], d, f"{key} {h}")
        n = len(W["seq"])
        if d["logits"].shape != (n, 5) or d["time_logits"].shape != (n, 8, 5) or not (np.isfinite(d["logits"]).all() and np.isfinite(d["time_logits"]).all()):
            raise SystemExit(f"{key} {h}: the saved logits have other shapes or non-finite values")
        return d


def code_digest() -> str:
    h = hashlib.sha256(Path(__file__).read_bytes())
    root = Path(__file__).resolve().parents[1]
    for f in HELPER_FILES:
        p = root / f
        h.update(p.read_bytes() if p.is_file() else b"missing:" + f.encode())
    return h.hexdigest()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ledger_hours(path: Path, kind: str) -> float:
    if not Path(path).is_file():
        return 0.0
    return float(sum(x["hours"] for x in (json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()) if x.get("kind") == kind))


def run_stage(ctx: Ctx, name: str, fn) -> None:
    """Every stage, finished or failed, adds its CPU wall-clock hours to the ledger; the CPU budget is checked first."""
    spent = ledger_hours(ctx.ledger, "cpu"); cap = float(ctx.tt["budget"]["cpu_wall_hours_max"])
    if spent >= cap:
        raise SystemExit(f"the CPU budget is spent ({spent:.3f} of {cap} h)")
    t0, started, ok, note = time.time(), now(), False, ""
    try:
        fn(ctx); ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ctx.ledger.parent.mkdir(parents=True, exist_ok=True)
        with ctx.ledger.open("a") as fh:
            fh.write(json.dumps({"job": name, "kind": "cpu", "started": started, "finished": now(), "hours": (time.time() - t0) / 3600, "ok": ok, "note": note}) + "\n")


def ids_of(ctx: Ctx, seqs: dict) -> set:
    proto = ctx.proto(); ids = {c for h in HALVES for c in proto["validation"][h]["clips"]}
    ids |= {r[0] for q in (x for h in seqs for x in seqs[h]) for r in q["refs"]} | {q["id"] for h in seqs for q in seqs[h]}
    return ids | {session_of(v) for v in list(ids)}


def write_json(paths, obj: dict, ids: set, what: str) -> None:
    flat = json.loads(json.dumps(obj, default=str)); assert_no_ids(flat, ids, what)
    for p in paths:
        Path(p).parent.mkdir(parents=True, exist_ok=True); write_atomic(Path(p), json.dumps(flat, indent=1))


def position_summary(W: dict, seqs_h: list[dict]) -> dict:
    """Anchor offsets and whether the anchor's reference label differs from the requested sample's (positions only)."""
    j = select_all(W["s"], W["centre"], W["marker"]); off = anchor_offsets(W["s"], W["centre"], j)
    anchor_lab = W["labels16"][np.arange(len(j)), np.asarray(ANCHORS)[j]]; differ = anchor_lab != W["label"]
    native = np.array([seqs_h[q]["kind"] == "native" for q in W["seq"]])
    return {"windows": int(len(j)), "offset_counts": {str(int(k)): int((off == k).sum()) for k in np.unique(off)}, "exact": int((off == 0).sum()),
            "non_exact": int((off != 0).sum()), "anchor_counts": {str(int(a)): int((np.asarray(ANCHORS)[j] == a).sum()) for a in ANCHORS},
            "label_differs": int(differ.sum()), "label_differs_non_native": int((differ & ~native).sum()), "non_exact_native": int(((off != 0) & native).sum())}


# ----------------------------------------------------------------------------------------------- stages
def stage_prepare(ctx: Ctx) -> None:
    checked = {"task_sheet": sha256_file(pinned(ctx.root, ctx.tt["task_sheet"]))}
    for k in ("a02_config", "a02_summary", "a02_readme", "protocol", "clip_index", "window_plans"):
        checked[k] = sha256_file(pinned(ctx.root, ctx.src[k]))
    checked |= {f"final_{k}": sha256_file(pinned(ctx.root, v)) for k, v in ctx.src["final_weights"].items()}
    import numpy, pandas, scipy  # noqa: E401
    env = {"python": sys.version.split()[0], "numpy": numpy.__version__, "pandas": pandas.__version__, "scipy": scipy.__version__}
    if any(str(env[k]) != str(v) for k, v in ctx.src["environment"].items()):
        raise SystemExit(f"environment {env} differs from {ctx.src['environment']}")
    rec = {"config": ctx.config, "checked": checked, "environment": env, "code": code_digest(), "halves": {}}
    seqs_all = {}
    for h in HALVES:
        ev, seqs, W = ctx.half(h); seqs_all[h] = seqs
        for key in ctx.src["predictions"][h]:
            d = ctx.prediction(h, key, W, seqs); rec["checked"][f"{h}_{key}"] = ctx.src["predictions"][h][key]["sha256"]
            del d
        rec["halves"][h] = {"sequences": len(seqs), "native": len(ev.natives), "pairs": len(ev.pairs), "multi": len(ev.multi),
                            "windows": int(len(W["seq"])), "samples": int(sum(len(q["labels"]) for q in seqs)), "positions": position_summary(W, seqs)}
        print(f"{h}: {len(seqs)} sequences, {len(W['seq'])} windows; nine prediction files verified", flush=True)
    write_json([ctx.out / "prepare.json", ctx.results / "preparation.json"], rec, ids_of(ctx, seqs_all), "preparation")


def cls_rows(cfg: dict, probs: dict, ev, params: dict) -> dict:
    out = {}
    for row, preds in (("argmax", {k: v.argmax(1) for k, v in probs.items()}), ("b3", decode(params, probs))):
        m, t = rcs.evaluate(cfg, preds, ev); out[row] = {"row": summary(m), "tables": t, "preds": preds}
    return out


def stage_check(ctx: Ctx) -> None:
    a02 = ctx.a02()["systems"]
    a02cfg = yaml.safe_load(pinned(ctx.root, ctx.src["a02_config"]).read_text())["video_dense_supervision"]["evaluation"]["fixed_ordered_setting"]
    v01 = json.loads(pinned(ctx.root, ctx.src["ordered"]["summary"]).read_text())["systems"]; rec = {"reproduced": {}, "max_prob_diff": 0.0}; seqs_all = {}
    for h in HALVES:
        ev, seqs, W = ctx.half(h); seqs_all[h] = seqs
        for key in ctx.src["predictions"][h]:
            d = ctx.prediction(h, key, W, seqs); probs = rr.per_sample(W, d["logits"].astype(np.float32), seqs)
            flat = np.concatenate([probs[q["id"]] for q in seqs]); diff = float(np.abs(flat - d["sample_probs"]).max())
            if diff != 0.0 or not np.array_equal(np.cumsum([0, *[len(probs[q["id"]]) for q in seqs]]), d["sample_offsets"]):
                raise SystemExit(f"{key} {h}: the per-sample CLS probabilities are not reproduced ({diff})")
            seed = int(key.rsplit("_s", 1)[1]); params = a02[key]["b3_choice"]["params"]
            rows = cls_rows(ctx.cfg, probs, ev, params)
            fixed = {"half_window": int(a02cfg["half_window"]), "threshold": float(a02cfg["thresholds"][seed]), "nms_radius": int(a02cfg["nms_radius"])}
            m, _ = rcs.evaluate(ctx.cfg, decode(fixed, probs), ev); rows["fixed"] = {"row": summary(m)}
            for row in ("argmax", "b3", "fixed"):
                want = a02[key][row][h]; got = rows[row]["row"]; bad = {f: (got[f], want[f]) for f in FLAT if got[f] != want[f]}
                if bad:
                    raise SystemExit(f"{key} {row} {h}: A02's stored row is not reproduced ({bad})")
            rec["reproduced"][f"{key}|{h}"] = "argmax, b3 and fixed rows equal A02's"
        for s in (0, 1, 2):
            with np.load(pinned(ctx.root, ctx.src["ordered"]["outputs"][h][s])) as z:
                check_windows(W, [q["id"] for q in seqs], {"sequence_ids": z["sequence_ids"], "seq": z["seq"], "centre": z["centre"], "marker": z["marker"]}, f"ordered_s{s} {h}")
                probs = rr.per_sample(W, z["logits"].astype(np.float32), seqs)
            rows = cls_rows(ctx.cfg, probs, ev, v01[f"ordered_s{s}"]["b3_choice"]["params"])
            for row in ("argmax", "b3"):
                want = v01[f"ordered_s{s}"][row][h]; got = rows[row]["row"]; bad = {f: (got[f], want[f]) for f in FLAT if got[f] != want[f]}
                if bad:
                    raise SystemExit(f"ordered_s{s} {row} {h}: V01's stored row is not reproduced ({bad})")
            rec["reproduced"][f"ordered_s{s}|{h}"] = "argmax and b3 rows equal V01's"
        print(f"{h}: nine A02 CLS systems and three frozen Ordered systems reproduced", flush=True)
    rec.update({"code": code_digest(), "ok": True})
    write_json([ctx.out / "check.json", ctx.results / "checks.json"], rec, ids_of(ctx, seqs_all), "check")


def token_probs(d: dict, W: dict, seqs_h: list[dict]) -> tuple[dict, np.ndarray, np.ndarray]:
    j = select_all(W["s"], W["centre"], W["marker"]); lg = token_logits(d["time_logits"].astype(np.float32), j)
    return rr.per_sample(W, lg, seqs_h), lg, j


def stage_calibrate(ctx: Ctx) -> None:
    h = "calibration"; ev, seqs, W = ctx.half(h); rec = {"half": h, "code": code_digest(), "config_sha256": sha256_file(ctx.root / ctx.config),
                                                        "predictions": {k: v["sha256"] for k, v in ctx.src["predictions"][h].items()}, "systems": {}}
    for key in ctx.src["predictions"][h]:
        d = ctx.prediction(h, key, W, seqs); probs, _, _ = token_probs(d, W, seqs)
        ch = rcs.b3_choose(ctx.cfg, DIAG, probs, ev)
        rec["systems"][f"token_{key}"] = {"status": ch["status"], "params": ch["params"], "n_feasible": ch["n_feasible"],
                                         "grid": [{k: v for k, v in r.items()} for r in ch["grid"]]}
        print(f"token_{key}: {ch['status']} {ch['params']}", flush=True)
    rec["choices"] = {k: {"status": v["status"], "params": v["params"], "n_feasible": v["n_feasible"]} for k, v in rec["systems"].items()}
    write_json([ctx.out / "calibration" / "choices.json", ctx.results / "choices.json"], rec, ids_of(ctx, {h: seqs}), "choices")


def stage_selection(ctx: Ctx) -> None:
    raise SystemExit("the selection stage is written in Task 2, after the report tests and before the calibration")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    for s in ("prepare", "check", "calibrate", "selection"):
        g.add_argument(f"--{s}", action="store_true")
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config)
    stage = next(s for s in ("prepare", "check", "calibrate", "selection") if getattr(a, s))
    run_stage(ctx, stage, {"prepare": stage_prepare, "check": stage_check, "calibrate": stage_calibrate, "selection": stage_selection}[stage])


if __name__ == "__main__":
    main()
