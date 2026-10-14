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
  --dry-run    a code check of the selection pipeline with the calibration half standing in for both halves (the
               selection half is not read; nothing reaches the results)
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
import scripts.trained_token_readout_tables as tb  # noqa: E402
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
from echo_routing.evaluate.complementarity import select_setting, sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.evaluate.window_target import nearest_centre, window_frames  # noqa: E402
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
                "echo_routing/temporal/windowed.py", "echo_routing/compose/stream_assembler.py", "scripts/trained_token_readout_tables.py",
                "scripts/round4_failure_inventory.py", "scripts/report_fact_asrf.py", "scripts/prepare_fold_output_correction.py",
                "scripts/report_fan224_visual.py", "echo_routing/evaluate/failure_units.py", "echo_routing/vit/target_mvit.py",
                "echo_routing/compose/recipe_builder.py", "echo_routing/temporal/dev_data.py", "echo_routing/temporal/baselines/b3_js_divergence.py",
                "echo_routing/temporal/segments.py")
ROOT = Path(__file__).resolve().parents[1]


def _repo_modules() -> list[str]:
    """The repository's own modules (scripts/, echo_routing/) loaded together with this one."""
    out = set()
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if not f or not f.endswith(".py"):
            continue
        q = Path(f).resolve()
        if q.is_relative_to(ROOT) and q.relative_to(ROOT).parts[0] in ("scripts", "echo_routing"):
            out.add(q.relative_to(ROOT).as_posix())
    return sorted(out)


IMPORTED = _repo_modules()            # fixed when this module is imported; lazily imported helpers are in HELPER_FILES


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


def meta_stream(d: dict, n_of: dict, label_of: dict, where: str = "a recipe") -> Stream:
    """A stream from its recipe alone: labels, joins and semantic boundaries; no prob or feature arrays. Error messages
    name the recipe's position (where), never an id."""
    recipe = recipe_from_dict(d); labels, joins, pos = [], [], 0
    for i, fr in enumerate(recipe.fragments):
        if fr.video_id not in n_of:
            raise SystemExit(f"{where}: fragment {i} comes from outside the half's clips")
        if not 0 <= fr.start < fr.end <= n_of[fr.video_id] or int(fr.label) != int(label_of[fr.video_id]):
            raise SystemExit(f"{where}: fragment {i} exceeds its clip or carries another label")
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
    pairs = tuple(meta_stream(r, n_of, label_of, f"pairs recipe {k}") for k, r in enumerate(v["banks"]["pairs"]["recipes"]))
    multi = tuple(meta_stream(r, n_of, label_of, f"multi recipe {k}") for k, r in enumerate(v["banks"]["multi"]["recipes"]))
    seqs = []
    for kind, items in (("native", [(s.seq_id, s.labels) for s in natives]), ("pairs", [(s.recipe.recipe_id, s.labels) for s in pairs]),
                        ("multi", [(s.recipe.recipe_id, s.labels) for s in multi])):
        for k, (sid, lab) in enumerate(items):
            p = plans["streams"].get(sid)
            if p is None or p.get("bank") != f"{kind}_{half}":
                raise SystemExit(f"{kind} sequence {k}: no window plan of the {kind}_{half} bank")
            if int(p["n"]) != len(lab) or len(p["refs"]) != len(lab):
                raise SystemExit(f"{kind} sequence {k}: the window plan's length differs from the metadata")
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
        ev, seqs = half_sets(self.proto(), h, pinned(self.root, self.src["clip_index"]), self.plans())
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
    """This file, the listed helpers and every repository module imported with it, each with its path."""
    h = hashlib.sha256(Path(__file__).read_bytes())
    for f in sorted(set(HELPER_FILES) | set(IMPORTED)):
        q = ROOT / f
        h.update(f.encode() + b"\0" + (q.read_bytes() if q.is_file() else b"missing"))
    return h.hexdigest()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ledger_hours(path: Path, kind: str) -> float:
    if not Path(path).is_file():
        return 0.0
    return float(sum(x["hours"] for x in (json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()) if x.get("kind") == kind))


def environment() -> dict:
    import numpy, pandas, scipy, torch  # noqa: E401
    return {"python": sys.version.split()[0], "numpy": numpy.__version__, "pandas": pandas.__version__, "scipy": scipy.__version__,
            "torch": torch.__version__}


def preconditions(ctx: Ctx, stage: str) -> None:
    """Every stage needs the pinned environment. Later stages need the preparation made by this code (with the same
    torch), and calibration, the dry run and selection also need the passed check made by this code."""
    env = environment()
    if any(str(env[k]) != str(v) for k, v in ctx.src["environment"].items()):
        raise SystemExit(f"environment {env} differs from the pinned {ctx.src['environment']}")
    needs = {"check": ("prepare.json",), "calibrate": ("prepare.json", "check.json"), "dry_run": ("prepare.json", "check.json"),
             "selection": ("prepare.json", "check.json")}.get(stage, ())
    for name in needs:
        f = ctx.out / name
        rec = json.loads(f.read_text()) if f.is_file() else {}
        if rec.get("code") != code_digest():
            raise SystemExit(f"{name} is missing or was made by other code; run that stage again first")
        if name == "check.json" and rec.get("ok") is not True:
            raise SystemExit("the check did not pass")
        if name == "prepare.json" and rec.get("environment", {}).get("torch") != env["torch"]:
            raise SystemExit("torch differs from the preparation's")


def run_stage(ctx: Ctx, name: str, fn) -> None:
    """Every stage, finished or failed, adds its CPU wall-clock hours to the ledger; the CPU budget is checked first."""
    spent = ledger_hours(ctx.ledger, "cpu"); cap = float(ctx.tt["budget"]["cpu_wall_hours_max"])
    if spent >= cap:
        raise SystemExit(f"the CPU budget is spent ({spent:.3f} of {cap} h)")
    t0, started, ok, note = time.time(), now(), False, ""
    try:
        preconditions(ctx, name); ctx.started = started; fn(ctx); ok = True
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


def refuse_other(paths, obj: dict, what: str) -> None:
    """A record is written once: an existing copy must equal the new one (move it aside and report why otherwise)."""
    new = json.loads(json.dumps(obj, default=str))
    for f in paths:
        if Path(f).is_file() and json.loads(Path(f).read_text()) != new:
            raise SystemExit(f"an earlier {what} differs from this one; it is never overwritten")


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


STORED_EXTRA = ("native_confusion", "A4C read as A5C", "A5C read as A4C")


def check_stored(got: dict, want: dict, what: str) -> None:
    """The nine flat metrics, and the confusion matrix and the A4C/A5C counts where the stored row has them."""
    bad = {f: (got[f], want[f]) for f in FLAT if got[f] != want[f]}
    bad |= {f: "differs" for f in STORED_EXTRA if f in want and got.get(f) != want[f]}
    if bad:
        raise SystemExit(f"{what}: the stored row is not reproduced ({bad})")


# ----------------------------------------------------------------------------------------------- stages
def stage_prepare(ctx: Ctx) -> None:
    checked = {"task_sheet": sha256_file(pinned(ctx.root, ctx.tt["task_sheet"]))}
    for k in ("a02_config", "a02_summary", "a02_readme", "protocol", "clip_index", "window_plans"):
        checked[k] = sha256_file(pinned(ctx.root, ctx.src[k]))
    checked |= {f"final_{k}": sha256_file(pinned(ctx.root, v)) for k, v in ctx.src["final_weights"].items()}
    env = environment()                            # pinned versions checked before the stage; torch recorded here
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
    v01 = json.loads(pinned(ctx.root, ctx.src["ordered"]["summary"]).read_text())["systems"]; rec = {"reproduced": {}}; seqs_all = {}; diffs = []
    for h in HALVES:
        ev, seqs, W = ctx.half(h); seqs_all[h] = seqs
        for key in ctx.src["predictions"][h]:
            d = ctx.prediction(h, key, W, seqs); probs = rr.per_sample(W, d["logits"].astype(np.float32), seqs)
            flat = np.concatenate([probs[q["id"]] for q in seqs]); diff = float(np.abs(flat - d["sample_probs"]).max()); diffs.append(diff)
            if diff != 0.0 or not np.array_equal(np.cumsum([0, *[len(probs[q["id"]]) for q in seqs]]), d["sample_offsets"]):
                raise SystemExit(f"{key} {h}: the per-sample CLS probabilities are not reproduced ({diff})")
            seed = int(key.rsplit("_s", 1)[1]); params = a02[key]["b3_choice"]["params"]
            rows = cls_rows(ctx.cfg, probs, ev, params)
            fixed = {"half_window": int(a02cfg["half_window"]), "threshold": float(a02cfg["thresholds"][seed]), "nms_radius": int(a02cfg["nms_radius"])}
            m, _ = rcs.evaluate(ctx.cfg, decode(fixed, probs), ev); rows["fixed"] = {"row": summary(m)}
            for row in ("argmax", "b3", "fixed"):
                check_stored(rows[row]["row"], a02[key][row][h], f"{key} {row} {h} (A02)")
            rec["reproduced"][f"{key}|{h}"] = "argmax, b3 and fixed rows equal A02's"
        for s in (0, 1, 2):
            with np.load(pinned(ctx.root, ctx.src["ordered"]["outputs"][h][s])) as z:
                check_windows(W, [q["id"] for q in seqs], {"sequence_ids": z["sequence_ids"], "seq": z["seq"], "centre": z["centre"], "marker": z["marker"]}, f"ordered_s{s} {h}")
                probs = rr.per_sample(W, z["logits"].astype(np.float32), seqs)
            rows = cls_rows(ctx.cfg, probs, ev, v01[f"ordered_s{s}"]["b3_choice"]["params"])
            for row in ("argmax", "b3"):
                check_stored(rows[row]["row"], v01[f"ordered_s{s}"][row][h], f"ordered_s{s} {row} {h} (V01)")
            rec["reproduced"][f"ordered_s{s}|{h}"] = "argmax and b3 rows equal V01's"
        print(f"{h}: nine A02 CLS systems and three frozen Ordered systems reproduced", flush=True)
    rec.update({"max_prob_diff": max(diffs), "code": code_digest(), "ok": True})
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
    a02 = ctx.a02()["systems"]                    # the source CLS controls: A02's stored choice and calibration row (reproduced in --check)
    rec["cls_reference"] = {f"cls_{k}": {"b3_choice": a02[k]["b3_choice"], "calibration_b3_row": {f: a02[k]["b3"][h][f] for f in FLAT}}
                            for k in ctx.src["predictions"][h]}
    paths = [ctx.out / "calibration" / "choices.json", ctx.results / "choices.json"]
    refuse_other(paths, rec, "choice record")
    write_json(paths, rec, ids_of(ctx, {h: seqs}), "choices")


COMPARE = {"token - CLS": ("token_aligned_dense", "cls_aligned_dense"), "token - Ordered": ("token_aligned_dense", "ordered"),
           "token - CLS-only token": ("token_aligned_dense", "token_center_only"), "token - Shuffled token": ("token_aligned_dense", "token_shuffled_dense"),
           "CLS-only token - CLS-only CLS": ("token_center_only", "cls_center_only"), "Shuffled token - Shuffled CLS": ("token_shuffled_dense", "cls_shuffled_dense")}
WITHIN = ("token - CLS", "CLS-only token - CLS-only CLS", "Shuffled token - Shuffled CLS")
WITHIN_ROWS = {"argmax": "argmax", "b3": "b3", "b3_at_cls_setting": "b3"}   # token row -> the same arm's CLS row it is compared with
METRICS = ("multi segment F1@50", "native macro-F1", "native cuts per clip", "multi boundary recall", "multi boundary F1", "same-class cut rate")
SUBSETS = ("native", "all mixed", "8 of 16", "9-11 of 16", "A4C/A5C pair windows", "exact anchor", "non-exact anchor")
TOKEN_FAMILIES = ("token_center_only", "token_aligned_dense", "token_shuffled_dense")
FAMILIES = TOKEN_FAMILIES + ("cls_center_only", "cls_aligned_dense", "cls_shuffled_dense", "ordered")


def with_f1(row: dict) -> dict:
    """A summary row plus native per-class precision and F1 from its confusion matrix (rows: reference, columns: output)."""
    conf = np.asarray(row["native_confusion"], float); tp = np.diag(conf); col, ref = conf.sum(0), conf.sum(1)
    prec = {c: (float(tp[k] / col[k]) if col[k] else None) for k, c in enumerate(CLASS_ORDER)}
    f1 = {c: (float(2 * tp[k] / (col[k] + ref[k])) if col[k] + ref[k] else None) for k, c in enumerate(CLASS_ORDER)}
    return row | {"native_precision": prec, "native_f1": f1}


def rows_of(cfg: dict, probs: dict, ev, decoders: dict) -> dict:
    """decoders: row name -> B3 params, or None for the direct argmax."""
    out = {}
    for row, params in decoders.items():
        preds = {k: v.argmax(1) for k, v in probs.items()} if params is None else decode(params, probs)
        m, tbl = rcs.evaluate(cfg, preds, ev); out[row] = {"row": with_f1(summary(m)), "tables": tbl, "preds": preds}
    return out


def sample_windows(W: dict, seqs_h: list[dict]) -> dict:
    """Per sequence, the window row whose prediction each sample takes (the nearest-centre mapping of the probabilities)."""
    out = {}
    for qi, q in enumerate(seqs_h):
        rows = np.flatnonzero(W["seq"] == qi); out[q["id"]] = rows[nearest_centre(len(q["labels"]), W["centre"][rows])]
    return out


def sample_offset_split(seqs_h: list[dict], served: dict, off: np.ndarray, preds_a: dict, preds_b: dict) -> dict:
    """Per-sample accuracy of two outputs on samples served by an exact or a non-exact window (description only)."""
    ids = [q["id"] for q in seqs_h]; y = np.concatenate([np.asarray(q["labels"]) for q in seqs_h])
    o = np.concatenate([off[served[i]] for i in ids])
    a = np.concatenate([np.asarray(preds_a[i]) for i in ids]) == y; b = np.concatenate([np.asarray(preds_b[i]) for i in ids]) == y
    return tb.offset_split(o, a, b, unit="samples")


def same_target(W: dict, meta: dict, logits: dict, offsets: np.ndarray) -> tuple[dict, dict]:
    """Per system, window accuracy at the request on the fixed subsets; per token readout, the comparison with its own
    arm's CLS (session bootstrap, stream sensitivity, changed windows with their sources) and the anchor-offset split."""
    y = W["label"]; mix = pt.mixed_subsets(W["labels16"], y, meta["mixed"])
    subs = {"native": np.flatnonzero(meta["kind"] == "native"), **{n: mix[n] for n in ("all mixed", "8 of 16", "9-11 of 16", "A4C/A5C pair windows")},
            "exact anchor": np.flatnonzero(offsets == 0), "non-exact anchor": np.flatnonzero(offsets != 0)}
    correct = {k: v.argmax(1) == y for k, v in logits.items()}
    acc = {k: {n: (float(c[subs[n]].mean()) if subs[n].size else None) for n in SUBSETS} for k, c in correct.items()}
    comp = {}
    for k in logits:
        if not k.startswith("token_"):
            continue
        a, b = correct[k], correct["cls_" + k[len("token_"):]]
        for n in SUBSETS:
            r = subs[n]
            comp[f"{k}|{n}"] = {"session": pt.group_bootstrap(a[r], b[r], meta["session"][r]), "stream": pt.group_bootstrap(a[r], b[r], meta["stream"][r]),
                                "changes": tb.changes(a[r], b[r], meta["recording"][r], meta["session"][r])}
        comp[f"{k}|offset split"] = tb.offset_split(offsets, a, b)
    support = {n: pt.subset_support(subs[n], meta["recording"], meta["session"], meta["run"], y) for n in SUBSETS}
    return {"accuracy": acc, "support": support}, comp


def half_systems(ctx: Ctx, h: str, choices: dict, a02: dict, v01: dict, fx_cfg: dict) -> dict:
    """Every system of one half: the token readouts at their committed settings (and at the same arm's CLS setting), the
    CLS outputs and the frozen Ordered outputs at their stored settings, each stored row reproduced first."""
    ev, seqs, W = ctx.half(h); meta = rtp.window_meta(W, seqs); systems, logits, saved = {}, {}, {}
    j = select_all(W["s"], W["centre"], W["marker"]); off = anchor_offsets(W["s"], W["centre"], j); served = sample_windows(W, seqs); split = {}
    for key in ctx.src["predictions"][h]:
        arm, seed = key.rsplit("_s", 1)[0], int(key.rsplit("_s", 1)[1]); d = ctx.prediction(h, key, W, seqs)
        cls_p = rr.per_sample(W, d["logits"].astype(np.float32), seqs); tok_lg = token_logits(d["time_logits"].astype(np.float32), j)
        tok_p = rr.per_sample(W, tok_lg, seqs); p_cls = a02[key]["b3_choice"]["params"]; ch = choices["choices"][f"token_{key}"]
        fixed = {"half_window": int(fx_cfg["half_window"]), "threshold": float(fx_cfg["thresholds"][seed]), "nms_radius": int(fx_cfg["nms_radius"])}
        cls = rows_of(ctx.cfg, cls_p, ev, {"argmax": None, "b3": p_cls, "fixed_ordered_setting": fixed})
        for row, stored in (("argmax", "argmax"), ("b3", "b3"), ("fixed_ordered_setting", "fixed")):
            check_stored(cls[row]["row"], a02[key][stored][h], f"{key} {row} {h}")
        tok = rows_of(ctx.cfg, tok_p, ev, {"argmax": None, "b3": ch["params"], "b3_at_cls_setting": p_cls})
        systems[f"cls_{arm}_s{seed}"] = cls | {"b3_choice": a02[key]["b3_choice"]}
        systems[f"token_{arm}_s{seed}"] = tok | {"b3_choice": {k: ch[k] for k in ("status", "params", "n_feasible")}}
        logits[f"cls_{arm}_s{seed}"] = d["logits"].astype(np.float32); logits[f"token_{arm}_s{seed}"] = tok_lg
        split[f"token_{arm}_s{seed}"] = {r: sample_offset_split(seqs, served, off, tok[r]["preds"], cls[r]["preds"]) for r in ("argmax", "b3")}
        saved[f"token_{arm}_s{seed}"] = {"token_logits": tok_lg, "anchor": j, "offset": off, "probs": tok_p, "decoded": tok["b3"]["preds"]}
        del d
    for s in (0, 1, 2):
        with np.load(pinned(ctx.root, ctx.src["ordered"]["outputs"][h][s])) as z:
            check_windows(W, [q["id"] for q in seqs], {"sequence_ids": z["sequence_ids"], "seq": z["seq"], "centre": z["centre"], "marker": z["marker"]}, f"ordered_s{s} {h}")
            lg = z["logits"].astype(np.float32)
        o = rows_of(ctx.cfg, rr.per_sample(W, lg, seqs), ev, {"argmax": None, "b3": v01[f"ordered_s{s}"]["b3_choice"]["params"]})
        for row in ("argmax", "b3"):
            check_stored(o[row]["row"], v01[f"ordered_s{s}"][row][h], f"ordered_s{s} {row} {h}")
        systems[f"ordered_s{s}"] = o | {"b3_choice": v01[f"ordered_s{s}"]["b3_choice"]}; logits[f"ordered_s{s}"] = lg
    return {"ev": ev, "seqs": seqs, "W": W, "meta": meta, "systems": systems, "logits": logits, "saved": saved, "offsets": off, "sample_split": split}


def save_outputs(out: Path, seqs_h: list[dict], saved: dict) -> None:
    """Per-window token logits, anchors and offsets, per-sample probabilities and per-sequence B3 outputs (server only)."""
    out.mkdir(parents=True, exist_ok=True); ids = [q["id"] for q in seqs_h]
    for key, d in saved.items():
        np.savez(out / f"{key}.npz", sequence_ids=np.asarray(ids), token_logits=d["token_logits"], anchor=d["anchor"], offset=d["offset"],
                 sample_offsets=np.cumsum([0, *[len(d["probs"][i]) for i in ids]]), sample_probs=np.concatenate([d["probs"][i] for i in ids]).astype(np.float32),
                 decoded=np.concatenate([np.asarray(d["decoded"][i], np.int8) for i in ids]))


def compare(systems: dict, evs: dict, tol: float) -> tuple[dict, dict]:
    """Paired system comparisons per seed, row and half, and their consistency over seeds. A comparison whose B3 setting
    is an infeasible system's diagnostic setting is flagged."""
    comp = {}
    for s in (0, 1, 2):
        for name, (a, b) in COMPARE.items():
            ka, kb = f"{a}_s{s}", f"{b}_s{s}"
            for row in (WITHIN_ROWS if name in WITHIN else ("b3",)):
                rb = WITHIN_ROWS[row] if name in WITHIN else "b3"
                st = [systems["selection"][k]["b3_choice"]["status"] for k, r in ((kb if row == "b3_at_cls_setting" else ka, row), (kb, rb)) if r != "argmax"]
                for h in HALVES:
                    A = systems[h][ka][row]; B = systems[h][kb][rb]
                    comp[f"s{s}|{row}|{name}|{h}"] = {"bootstrap": paired_bootstrap(A["tables"], B["tables"]),
                                                      "native_changes": native_changes(A["tables"]["native"], B["tables"]["native"]),
                                                      "segment_changes": segment_changes(evs[h], A["preds"], B["preds"]),
                                                      "boundary_changes": boundary_changes(evs[h], A["preds"], B["preds"], tol),
                                                      "diagnostic_setting": any(x != "feasible" for x in st)}
    ver = {row: {n: {m: {h: rt.consistency([comp[f"s{s}|{row}|{n}|{h}"]["bootstrap"][m] for s in (0, 1, 2)]) for h in HALVES} for m in METRICS}
                 for n in COMPARE if all(f"s{s}|{row}|{n}|{h}" in comp for s in (0, 1, 2) for h in HALVES)} for row in WITHIN_ROWS}
    return comp, ver


def stored_rows(a02: dict) -> dict:
    """Earlier systems' rows as A02 stored them (not recomputed)."""
    names = ["E0 + B3", "E1 + B3", "B7 + B3"] + [f"{n} s{k}" for n in ("E1 + ASRF", "paired ASRF", "Original FACT", "A01 Linear + B3", "A01 MS-TCN", "A01 ASRF") for k in (0, 1, 2)]
    names += [f"continued_cls_s{k}" for k in (0, 1, 2)]
    return {n: (a02[n]["b3"] if "b3" in a02[n] else a02[n]["row"]) for n in names}


def strip(systems: dict) -> dict:
    out = {}
    for h, by in systems.items():
        for k, s in by.items():
            e = out.setdefault(k, {})
            for r, x in s.items():
                if r == "b3_choice":
                    e["b3_choice"] = x
                else:
                    e.setdefault(r, {})[h] = x["row"]
    return out


def position_figure(path: Path) -> None:
    """A schematic of the position rule on two synthetic windows (sample indices only, no frame)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(13, 4.6))
    for ax, (title, s, c) in zip(axes, (("Interior window. The request c sits at slot 8, and anchor slot 8 holds c (offset 0)", np.arange(22, 38), 30),
                                         ("Window at a sequence end. The request c sits at slot 15, and the nearest anchor is slot 14 (offset -1)", np.arange(84, 100), 99))):
        m = marker_of(s, c); j = selected_anchor(s, c, m); ax.set_xlim(-0.5, 16.5); ax.set_ylim(0, 2.1); ax.axis("off")
        for i in range(16):
            col = "#d62728" if i == m else ("#4c72b0" if i in ANCHORS else "#cfd8e3")
            ax.add_patch(plt.Rectangle((i, 0.9), 0.85, 0.7, color=col, alpha=1.0 if i in (ANCHORS[j], m) else 0.5))
            ax.text(i + 0.42, 1.25, str(int(s[i])), ha="center", va="center", fontsize=7, color="white" if i in ANCHORS or i == m else "black")
        ax.annotate("", xy=(ANCHORS[j] + 0.42, 0.88), xytext=(ANCHORS[j] + 0.42, 0.45), arrowprops={"arrowstyle": "->", "color": "#4c72b0"})
        ax.text(0, 0.15, f"{title}. The time logits of anchor slot {ANCHORS[j]} are read, and the prediction stays at sample {c}.", fontsize=8)
        ax.text(0, 1.85, "Each box is an input slot with its sample index. Blue boxes are the trained anchors (even slots). The red box is the marker slot of c.", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def norm(x):
    return json.loads(json.dumps(x, default=str))


def summarise(ctx: Ctx, choices: dict, a02: dict, per: dict, tol: float, own_halves: bool) -> dict:
    """own_halves: each reported half was read from itself (the final run); then the recomputed CLS and Ordered goals
    must equal A02's stored goals."""
    systems = {h: per[h]["systems"] for h in HALVES}; evs = {h: per[h]["ev"] for h in HALVES}
    comp, ver = compare(systems, evs, tol)
    cls_comp = {h: per[h]["cls_comp"] for h in HALVES}
    cls_ver = {f"{fam}|{n}": {h: pt.verdict([cls_comp[h][f"{fam}_s{s}|{n}"]["session"] for s in (0, 1, 2)],
                                            [cls_comp[h][f"{fam}_s{s}|{n}"]["stream"] for s in (0, 1, 2)]) for h in HALVES}
               for fam in TOKEN_FAMILIES for n in SUBSETS}
    goals = {}
    for fam in FAMILIES:
        rows = [{h: systems[h][f"{fam}_s{s}"]["b3"]["row"] for h in HALVES} for s in (0, 1, 2)]
        goals[fam] = rt.goal(rows, [systems["selection"][f"{fam}_s{s}"]["b3_choice"]["status"] == "feasible" for s in (0, 1, 2)], ctx.tt["goal"])
    if own_halves:
        stored = ctx.a02()["goal"]
        for fam, want in (*((f"cls_{arm}", arm) for arm in ARMS), ("ordered", "ordered")):
            if norm(goals[fam]) != norm(stored[want]):
                raise SystemExit(f"the recomputed goal of {fam} differs from A02's stored goal")
    sysv = {c: ver["b3"][c]["multi segment F1@50"] for c in tb.SYS}
    g = goals["token_aligned_dense"]
    losses = {m: ver["b3"]["token - CLS"][m] for m in ("native macro-F1", "multi boundary recall")}
    dec = tb.decide(sysv, cls_ver["token_aligned_dense|all mixed"], not g["constraint_failures"], g["met"], ctx.tt["decision_rules"], losses)
    units = rr.units(SimpleNamespace(halves=evs), {k: {r: {h: systems[h][k][r] for h in HALVES} for r in ("argmax", "b3")} for k in systems["selection"]}, tol)
    return {"choices": choices["choices"], "systems": strip(systems), "system_comparisons": comp, "verdicts": ver,
            "classification": {"accuracy": {k: {h: per[h]["cls_res"]["accuracy"][k] for h in HALVES} for k in per["selection"]["cls_res"]["accuracy"]},
                               "support": {h: per[h]["cls_res"]["support"] for h in HALVES},
                               "comparisons": {f"{k}|{h}": v for h in HALVES for k, v in cls_comp[h].items()}, "verdicts": cls_ver},
            "positions": {h: per[h]["positions"] | {"sample_split": per[h]["sample_split"]} for h in HALVES},
            "units": units, "goal": goals, "decision": dec, "stored_rows": stored_rows(a02), "post_hoc": []}


def read_halves(ctx: Ctx, choices: dict, data: dict, a02: dict, v01: dict, fx_cfg: dict) -> dict:
    """data: reported half -> the half whose files are read. A half read once serves every reported half that names it."""
    per = {}
    for h in ("calibration", "selection"):
        if data[h] != h and data[h] in per:
            per[h] = per[data[h]]; continue
        x = half_systems(ctx, data[h], choices, a02, v01, fx_cfg)
        if data[h] == "calibration":              # the committed choice records must be reproduced by the applied settings
            for key, rec in choices["systems"].items():
                g = next(r for r in rec["grid"] if r["params"] == rec["params"])
                if any(x["systems"][key]["b3"]["row"][f] != g[f] for f in FLAT if f in g):
                    raise SystemExit(f"{key}: the calibration row differs from the committed choice record")
        x["cls_res"], x["cls_comp"] = same_target(x["W"], x["meta"], x["logits"], x["offsets"]); x["positions"] = position_summary(x["W"], x["seqs"])
        per[h] = x; print(f"{h} (read from {data[h]}): systems, same-target classification and positions done", flush=True)
    return per


def identity(ctx: Ctx, choices_sha: str, data: dict) -> dict:
    src = ctx.src
    return {"code": code_digest(), "config_sha256": sha256_file(ctx.root / ctx.config), "choices_sha256": choices_sha, "halves_read": dict(data),
            "predictions": {h: {k: v["sha256"] for k, v in src["predictions"][h].items()} for h in HALVES},
            "ordered_outputs": {h: {str(s): v["sha256"] for s, v in src["ordered"]["outputs"][h].items()} for h in HALVES},
            **{k: src[k]["sha256"] for k in ("a02_summary", "a02_config", "protocol", "clip_index", "window_plans")},
            "v01_summary": src["ordered"]["summary"]["sha256"]}


def evaluate_all(ctx: Ctx, choices: dict, data: dict, choices_sha: str) -> tuple[dict, dict, set]:
    """Everything of a run in memory; nothing is written here."""
    a02 = ctx.a02()["systems"]; v01 = json.loads(pinned(ctx.root, ctx.src["ordered"]["summary"]).read_text())["systems"]
    fx_cfg = yaml.safe_load(pinned(ctx.root, ctx.src["a02_config"]).read_text())["video_dense_supervision"]["evaluation"]["fixed_ordered_setting"]
    tol = float(ctx.cfg["selection"]["boundary_tolerance_s"]) * float(ctx.cfg["protocol"]["hz"])
    per = read_halves(ctx, choices, data, a02, v01, fx_cfg)
    res = summarise(ctx, choices, a02, per, tol, own_halves=all(data[h] == h for h in HALVES)) | {"identity": identity(ctx, choices_sha, data)}
    ids = ids_of(ctx, {h: per[h]["seqs"] for h in HALVES}); flat = norm(res); assert_no_ids(flat, ids, "summary")
    if any(x in tb.tables_md(flat) for x in ids):
        raise SystemExit("an id appears in the report tables")
    return flat, per, ids


def publish(ctx: Ctx, flat: dict, per: dict, out: Path, final: bool, extra: dict, ids: set) -> bool:
    """Written once, after everything is computed. A final run whose earlier summary exists must reproduce it exactly, and
    after a finished run nothing is rewritten. The dry run writes only under its own directory."""
    old, done = out / "summary.json", out / "done.json"
    if final and old.is_file():
        if json.loads(old.read_text()) != flat:
            raise SystemExit("an earlier selection summary differs from this run; nothing is overwritten")
        if done.is_file():
            print("the finished selection run is reproduced exactly; nothing is rewritten", flush=True)
            return False
    for h in HALVES:
        save_outputs(out / h, per[h]["seqs"], per[h]["saved"])
    write_atomic(old, json.dumps(flat, indent=1)); write_atomic(out / "report_tables.md", tb.tables_md(flat))
    if not final:
        position_figure(out / "position_mapping.png")
        return True
    full = norm(flat | extra); assert_no_ids(full, ids, "results summary")
    write_atomic(ctx.results / "summary.json", json.dumps(full, indent=1)); write_atomic(ctx.results / "report_tables.md", tb.tables_md(full))
    position_figure(ctx.results / "position_mapping.png")
    write_atomic(done, json.dumps({"choices_sha256": flat["identity"]["choices_sha256"], "finished": now()}))
    return True


def rederive_choice(rec: dict, goal: dict) -> None:
    """The calibration rule applied again to the record's own grid must give its setting, status and feasible count."""
    sel = select_setting(rec["grid"], float(goal["native_macro_f1_min"]["calibration"]), float(goal["boundary_recall_min"]))
    if sel["chosen"] is None:
        d = DIAG["evaluation"]["b3_infeasible_diagnostic"]
        diag = next(r for r in rec["grid"] if r["params"]["half_window"] == int(d["half_window"]) and abs(r["params"]["threshold_quantile"] - float(d["quantile"])) < 1e-9)
        want = ("infeasible (diagnostic setting shown)", diag["params"], 0)
    else:
        want = ("feasible", sel["chosen"]["params"], sel["n_feasible"])
    if (rec["status"], rec["params"], rec["n_feasible"]) != want:
        raise SystemExit("a choice record does not follow the calibration rule on its own grid")


def load_choices(ctx: Ctx, expected_sha: str | None = None) -> tuple[dict, Path]:
    """The choice record written by --calibrate: both copies equal (and, for --selection, the committed file's sha256),
    made by this code and configuration on the pinned calibration predictions, one record per token system holding the
    configured 40-setting grid, and every choice re-derived from its own grid."""
    path, twin = ctx.results / "choices.json", ctx.out / "calibration" / "choices.json"
    if not path.is_file() or not twin.is_file() or sha256_file(path) != sha256_file(twin):
        raise SystemExit("the choice record is missing or differs from the calibration stage's copy")
    if expected_sha is not None and sha256_file(path) != expected_sha:
        raise SystemExit("choices.json is not the committed record (its sha256 differs)")
    choices = json.loads(path.read_text())
    if choices["code"] != code_digest() or choices["config_sha256"] != sha256_file(ctx.root / ctx.config):
        raise SystemExit("the choices belong to other code or another configuration")
    if choices["predictions"] != {k: v["sha256"] for k, v in ctx.src["predictions"]["calibration"].items()}:
        raise SystemExit("the choices were made on other predictions")
    keys = {f"token_{k}" for k in ctx.src["predictions"]["calibration"]}
    if set(choices["systems"]) != keys or set(choices["choices"]) != keys:
        raise SystemExit("the choice record does not hold exactly one record per token system")
    for k, rec in choices["systems"].items():
        tb.check_choice_record(rec, ctx.tt["b3"]["grid"]); rederive_choice(rec, ctx.tt["goal"])
        if choices["choices"][k] != {x: rec[x] for x in ("status", "params", "n_feasible")}:
            raise SystemExit("a summary choice differs from its full record")
    return choices, path


def stage_dry_run(ctx: Ctx) -> None:
    """A code check of the selection pipeline before the choices are committed: the calibration half stands in for both
    halves, the selection half is never read, and nothing reaches the results or the selection outputs."""
    choices, path = load_choices(ctx)
    flat, per, ids = evaluate_all(ctx, choices, {"calibration": "calibration", "selection": "calibration"}, sha256_file(path))
    flat["dry_run"] = "the calibration half stands in for both halves; a code check, not a result"
    publish(ctx, flat, per, ctx.out / "dry_run", False, {}, ids)
    print("dry run ok (a code check, not a result)", flush=True)


def stage_selection(ctx: Ctx) -> None:
    """Only with the committed choices (--choices-sha256): checked, then applied once; a repeat run may only reproduce."""
    want = getattr(ctx, "choices_sha256", None)
    if not want:
        raise SystemExit("--selection needs --choices-sha256, the sha256 of the committed choices.json")
    choices, path = load_choices(ctx, want)
    out = ctx.out / "selection"; done = out / "done.json"
    if done.is_file() and json.loads(done.read_text()).get("choices_sha256") != want:
        raise SystemExit("an earlier selection run used other choices; it is never overwritten")
    flat, per, ids = evaluate_all(ctx, choices, {"calibration": "calibration", "selection": "selection"}, want)
    led = [json.loads(x) for x in ctx.ledger.read_text().splitlines() if x.strip()] if ctx.ledger.is_file() else []
    extra = {"cost": {"cpu_hours_before_this_stage": ledger_hours(ctx.ledger, "cpu"), "gpu_hours": ledger_hours(ctx.ledger, "gpu")},
             "run": {"selection_started": getattr(ctx, "started", None), "earlier_stages": [{k: x[k] for k in ("job", "started", "finished", "ok")} for x in led]}}
    if publish(ctx, flat, per, out, True, extra, ids):
        print(json.dumps(flat["decision"]["applied"]), flush=True)


STAGES = {"prepare": stage_prepare, "check": stage_check, "calibrate": stage_calibrate, "dry_run": stage_dry_run, "selection": stage_selection}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--choices-sha256", help="--selection only: the sha256 of the committed choices.json")
    g = ap.add_mutually_exclusive_group(required=True)
    for s in STAGES:
        g.add_argument(f"--{s.replace('_', '-')}", dest=s, action="store_true")
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config); ctx.choices_sha256 = a.choices_sha256
    stage = next(s for s in STAGES if getattr(a, s))
    run_stage(ctx, stage, STAGES[stage])


if __name__ == "__main__":
    main()
