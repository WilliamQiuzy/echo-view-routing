#!/usr/bin/env python
"""Token stability and segmentation attribution (docs/superpowers/plans/2026-10-14-token-stability-attribution.md;
configs/experiments/token_stability_attribution_v1.yaml). A retrospective, CPU-only diagnosis of the frozen A02/A03
outputs. No model is built, loaded or run, nothing is trained, and no selector, decoder, threshold or fusion is searched.

Stages (each appends its wall-clock hours to the round's ledger, failed attempts included):
  --prepare  pins (the task sheet, A03's sources and the A02 sources A03 pins), the environment, the stem geometry read
             from the source code, the allowlist metadata of both halves and every saved prediction file (identity,
             sequence order, centres, markers, shapes, finite values)
  --check    A03's aligned token and CLS rows (argmax, actual B3, token at the CLS setting) reproduced exactly; A03's saved
             token logits, anchors, offsets, per-sample probabilities and decoded outputs equal bit for bit; the four
             cells' endpoints equal the two readouts
  --run      the single complete diagnosis: the actual systems' segment accounting, the four reference-conditioned cells
             at the fixed decoder with their contrasts, and the overlapping-anchor stability; per-case data stay on the
             server under the run directory

The metadata come from A03's allowlist adapter (scripts/trained_token_readout.py): one development half at a time, only
that half's protocol entries, clip-index rows and window plans. The validation entry, the Inputs class and A02's
evaluate entry are not used, and no feature, training label, other-split file or raw frame is read.

  python scripts/token_stability_attribution.py --config configs/experiments/token_stability_attribution_v1.yaml --prepare
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.report_center_supervision as rcs  # noqa: E402
import scripts.token_stability_attribution_tables as tb  # noqa: E402
import scripts.trained_token_readout as tt  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.ev9v_complementarity import decode  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402
from scripts.raw_video_context_tables import consistency  # noqa: E402
from scripts.report_fold_output_correction import summary, write_atomic  # noqa: E402
from scripts.report_model_development import paired_bootstrap  # noqa: E402
from scripts.trained_token_readout import (ANCHORS, CLASS_ORDER, anchor_offsets, check_identity, check_stored, check_windows,  # noqa: E402,F401
                                           half_sets, half_windows, pinned, sample_windows, select_all, token_logits, with_f1)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

import echo_routing.vit.target_mvit as tm  # noqa: E402
from echo_routing.config import load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.evaluate.window_target import nearest_centre  # noqa: E402
from echo_routing.temporal.segments import labels_to_segments  # noqa: E402

HALVES = tb.HALVES
KINDS = tb.KINDS
ROOT = Path(__file__).resolve().parents[1]
HELPER_FILES = ("scripts/token_stability_attribution_tables.py", "scripts/trained_token_readout.py", "scripts/trained_token_readout_tables.py",
                "scripts/report_fan224_visual.py", "scripts/report_fact_asrf.py")


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


IMPORTED = _repo_modules()


# ----------------------------------------------------------------------------------------------- pure helpers
def softmax(logits) -> np.ndarray:
    """A03's softmax (torch, float32), so that mapped probabilities equal A03's bit for bit."""
    return torch.softmax(torch.from_numpy(np.ascontiguousarray(logits, dtype=np.float32)), -1).numpy()


def map_probs(W: dict, P: np.ndarray, seqs_h: list[dict]) -> dict:
    """Window probabilities to the 10 Hz samples by the nearest request (A03's report_raw_video_context.per_sample)."""
    out = {}
    for qi, q in enumerate(seqs_h):
        rows = np.flatnonzero(W["seq"] == qi); out[q["id"]] = P[rows][nearest_centre(len(q["labels"]), W["centre"][rows])]
    return out


def mixed_flags(W: dict, kinds) -> np.ndarray:
    """Constructed-stream windows whose 16 input slots hold at least two reference classes; native windows are pure."""
    return (np.asarray(kinds) != "native") & np.array([len(set(x.tolist())) >= 2 for x in W["labels16"]], bool)


def half_meta(W: dict, seqs: list[dict]) -> dict:
    """Per window: population, mixed flag, the requested sample's source recording and session, and its distance bin."""
    kinds = np.array([seqs[q]["kind"] for q in W["seq"]]); rec = np.array([seqs[q]["refs"][c][0] for q, c in zip(W["seq"], W["centre"])])
    changes = [tb.view_changes(q["labels"]) for q in seqs]
    dist = [tb.request_distance(int(c), changes[q]) for q, c in zip(W["seq"], W["centre"])]
    return {"kind": kinds, "mixed": mixed_flags(W, kinds), "recording": rec, "session": np.array([session_of(v) for v in rec]),
            "bin": np.array([tb.distance_bin(d) for d in dist]), "changes": changes}


def check_mapping(classes, mapping: str) -> None:
    tt.check_class_order(classes, mapping)


def checked_prediction(root, pin: dict, identity: str, W: dict, seq_ids: list) -> dict:
    """A saved prediction file, checked: pin, training identity, sequence order, centres, markers, shapes, values."""
    path = pinned(root, pin)
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    check_identity(str(d["identity"]), identity, pin["path"])
    check_windows(W, seq_ids, d, pin["path"])
    n = len(W["seq"])
    if d["logits"].shape != (n, 5) or d["time_logits"].shape != (n, 8, 5) or not (np.isfinite(d["logits"]).all() and np.isfinite(d["time_logits"]).all()):
        raise SystemExit(f"{pin['path']}: the saved logits have other shapes or non-finite values")
    return d


def mvit_module():
    import torchvision.models.video.mvit as m
    return m


def stem_geometry(defaults: dict | None = None, v2s_source: str | None = None, target_source: str | None = None) -> dict:
    """The time geometry of the RGB patch embedding and the marker stem, read from the source code (no model is built):
    time kernel 3, stride 2 and padding 1, no later temporal pooling, a marker stem that copies the patch embedding and a
    marker plane on one slot. Anything else stops the round."""
    m = mvit_module()
    if defaults is None:
        sig = inspect.signature(m.MViT.__init__).parameters
        defaults = {k: sig[k].default for k in ("patch_embed_kernel", "patch_embed_stride", "patch_embed_padding")}
    v2s = inspect.getsource(m.mvit_v2_s) if v2s_source is None else v2s_source
    tgt = inspect.getsource(tm) if target_source is None else target_source
    k, s, p = (tuple(defaults[x]) for x in ("patch_embed_kernel", "patch_embed_stride", "patch_embed_padding"))
    found = re.search(r'"stride_q":\s*(\[.*?\]),\s*"stride_kv"', v2s, re.S)
    strides = ast.literal_eval(found.group(1)) if found else None
    problems = []
    if (k[0], s[0], p[0]) != (3, 2, 1):
        problems.append(f"patch embedding time kernel/stride/padding {(k[0], s[0], p[0])}")
    if not strides or any(int(x[0]) != 1 for x in strides):
        problems.append("a later block pools time (or the strides could not be read)")
    if "patch_embed" in v2s or "patch_embed" in inspect.getsource(m._mvit):
        problems.append("the builder overrides the patch embedding")
    if not re.search(r"self\.marker_stem = nn\.Conv3d\(1, conv\.out_channels, conv\.kernel_size, conv\.stride, conv\.padding, conv\.dilation", tgt):
        problems.append("the marker stem does not copy the patch embedding")
    if not re.search(r"0, slots\] = 1\.0", tgt):
        problems.append("the marker plane is not a single slot")
    if problems:
        raise SystemExit("the stem geometry differs from the documented 3-frame geometry: " + "; ".join(problems))
    return {"kernel_t": 3, "stride_t": 2, "padding_t": 1, "later_time_strides": [int(x[0]) for x in strides], "marker_stem": "copies conv_proj",
            "marker_plane": "one slot"}


def environment() -> dict:
    import numpy as _np, pandas, scipy  # noqa: E401
    return {"python": sys.version.split()[0], "numpy": _np.__version__, "pandas": pandas.__version__, "scipy": scipy.__version__, "torch": torch.__version__}


def code_digest() -> str:
    """This file, the listed helpers and every repository module imported with it, each with its path."""
    h = hashlib.sha256(Path(__file__).read_bytes())
    for f in sorted(set(HELPER_FILES) | set(IMPORTED)):
        q = ROOT / f
        h.update(f.encode() + b"\0" + (q.read_bytes() if q.is_file() else b"missing"))
    return h.hexdigest()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stats(x) -> dict:
    x = np.asarray(x, float)
    return {"n": int(x.size), "mean": float(x.mean()) if x.size else None, "median": float(np.median(x)) if x.size else None,
            "p90": float(np.percentile(x, 90)) if x.size else None}


def units(recordings, sessions=None) -> dict:
    rec = set(recordings); ses = set(sessions) if sessions is not None else {session_of(v) for v in rec}
    return {"recordings": len(rec), "sessions": len(ses), "few_units": len(ses) < tb.FEW}


# ----------------------------------------------------------------------------------------------- the round's context
class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.cfg, self.config, self.ts = cfg, config, cfg["token_stability_attribution"]
        self.paths = load_paths(); self.root = self.paths.repo_root; self.src = self.ts["source"]
        self.out = self.root / self.ts["outputs"]["root"]; self.results = self.root / self.ts["outputs"]["results_dir"]
        self.ledger = self.root / self.ts["budget"]["ledger"]; self._cache = {}

    def _load(self, key: str, pin: dict, kind: str = "json"):
        if key not in self._cache:
            text = pinned(self.root, pin).read_text()
            self._cache[key] = json.loads(text) if kind == "json" else yaml.safe_load(text)
        return self._cache[key]

    def a03src(self) -> dict:
        return self._load("a03_config", self.src["a03_config"], "yaml")["trained_token_readout"]["source"]

    def a03(self) -> dict:
        return self._load("a03_summary", self.src["a03_summary"])

    def choices(self) -> dict:
        return self._load("a03_choices", self.src["a03_choices"])

    def a02(self) -> dict:
        return self._load("a02_summary", self.a03src()["a02_summary"])

    def proto(self) -> dict:
        p = self._load("protocol", self.a03src()["protocol"])
        check_mapping(rcs.CLASSES, p["mapping_version"])
        return p

    def plans(self) -> dict:
        return self._load("plans", self.a03src()["window_plans"])

    def half(self, h: str) -> tuple:
        """One half's metadata: EvalSet, sequences, windows and per-window kind, mixed flag, request source and distance bin."""
        ev, seqs = half_sets(self.proto(), h, pinned(self.root, self.a03src()["clip_index"]), self.plans())
        W = half_windows(seqs)
        return ev, seqs, W, half_meta(W, seqs)

    def prediction(self, h: str, arm: str, seed: int, W: dict, seqs_h: list[dict]) -> dict:
        key = f"{arm}_s{seed}"
        return checked_prediction(self.root, self.a03src()["predictions"][h][key], self.a02()["runs"][key]["identity"], W, [q["id"] for q in seqs_h])

    def params(self, seed: int) -> tuple[dict, dict]:
        """(the A03 token choice, the A02 aligned CLS stored setting) of one seed, each checked against its source."""
        fixed = self.a02()["systems"][f"aligned_dense_s{seed}"]["b3_choice"]["params"]
        if {k: fixed[k] for k in ("half_window", "threshold", "nms_radius", "threshold_quantile")} != self.ts["decoders"]["fixed_params"][seed]:
            raise SystemExit(f"seed {seed}: A02's stored CLS setting differs from the configured fixed decoder")
        return self.choices()["choices"][f"token_aligned_dense_s{seed}"]["params"], fixed


def ids_of(ctx: Ctx, seqs: dict) -> set:
    proto = ctx.proto(); ids = {c for h in HALVES for c in proto["validation"][h]["clips"]}
    ids |= {r[0] for q in (x for h in seqs for x in seqs[h]) for r in q["refs"]} | {q["id"] for h in seqs for q in seqs[h]}
    return ids | {session_of(v) for v in list(ids)}


def write_json(paths, obj: dict, ids: set, what: str) -> None:
    flat = json.loads(json.dumps(obj, default=str)); assert_no_ids(flat, ids, what)
    for p in paths:
        Path(p).parent.mkdir(parents=True, exist_ok=True); write_atomic(Path(p), json.dumps(flat, indent=1))


# ----------------------------------------------------------------------------------------------- ledger and preconditions
def ledger_hours(path: Path, kind: str) -> float:
    if not Path(path).is_file():
        return 0.0
    return float(sum(x["hours"] for x in (json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()) if x.get("kind") == kind))


def preconditions(ctx: Ctx, stage: str) -> None:
    env = environment()
    if any(str(env[k]) != str(v) for k, v in ctx.src["environment"].items()):
        raise SystemExit(f"environment {env} differs from the pinned {ctx.src['environment']}")
    for name in {"check": ("prepare.json",), "run": ("prepare.json", "check.json")}.get(stage, ()):
        f = ctx.out / name; rec = json.loads(f.read_text()) if f.is_file() else {}
        if rec.get("code") != code_digest():
            raise SystemExit(f"{name} is missing or was made by other code; run that stage again first")
        if name == "check.json" and rec.get("ok") is not True:
            raise SystemExit("the check did not pass")
        if name == "prepare.json" and rec.get("environment", {}).get("torch") != env["torch"]:
            raise SystemExit("torch differs from the preparation's")


def run_stage(ctx: Ctx, name: str, fn) -> None:
    """Every stage, finished or failed, adds its CPU wall-clock hours to the ledger; the CPU budget is checked first."""
    spent = ledger_hours(ctx.ledger, "cpu"); cap = float(ctx.ts["budget"]["cpu_wall_hours_max"])
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


# ----------------------------------------------------------------------------------------------- stages: prepare and check
def stage_prepare(ctx: Ctx) -> None:
    ts = ctx.ts; a3 = ctx.a03src(); root = ctx.root
    checked = {"task_sheet": sha256_file(pinned(root, ts["task_sheet"]))}
    checked |= {k: sha256_file(pinned(root, ctx.src[k])) for k in ("a03_config", "a03_summary", "a03_readme", "a03_choices")}
    checked |= {k: sha256_file(pinned(root, a3[k])) for k in ("protocol", "clip_index", "window_plans", "a02_summary", "a02_config")}
    if ctx.choices()["predictions"] != {k: v["sha256"] for k, v in a3["predictions"]["calibration"].items()}:
        raise SystemExit("A03's choices were made on other predictions than A03's configuration pins")
    if ctx.a03()["identity"]["choices_sha256"] != ctx.src["a03_choices"]["sha256"]:
        raise SystemExit("A03's summary applied another choice record")
    for seed in ts["seeds"]:
        ctx.params(seed)
    checked |= {f"a03_output_{h}_{s}": sha256_file(pinned(root, ctx.src["a03_outputs"][h][s])) for h in HALVES for s in ts["seeds"]}
    rec = {"config": ctx.config, "checked": checked, "environment": environment(), "stem_geometry": stem_geometry(), "code": code_digest(), "halves": {}}
    seqs_all = {}
    for h in HALVES:
        ev, seqs, W, meta = ctx.half(h); seqs_all[h] = seqs
        for arm in (ts["arms"]["primary"], *ts["arms"]["controls"]):
            for seed in ts["seeds"]:
                ctx.prediction(h, arm, seed, W, seqs); rec["checked"][f"{h}_{arm}_s{seed}"] = a3["predictions"][h][f"{arm}_s{seed}"]["sha256"]
        rec["halves"][h] = {"sequences": {k: int(sum(q["kind"] == k for q in seqs)) for k in KINDS}, "windows": {k: int((meta["kind"] == k).sum()) for k in KINDS},
                            "mixed_windows": {k: int((meta["mixed"] & (meta["kind"] == k)).sum()) for k in KINDS},
                            "samples": int(sum(len(q["labels"]) for q in seqs))}
        print(f"{h}: {len(seqs)} sequences, {len(W['seq'])} windows, {int(meta['mixed'].sum())} mixed; nine prediction files verified", flush=True)
    write_json([ctx.out / "prepare.json", ctx.results / "preparation.json"], rec, ids_of(ctx, seqs_all), "preparation")


def readouts(ctx: Ctx, h: str, seed: int, W: dict, seqs: list[dict], d: dict | None = None) -> dict:
    """The aligned seed's window probabilities (CLS and token), anchors and per-sample probabilities."""
    d = d if d is not None else ctx.prediction(h, ctx.ts["arms"]["primary"], seed, W, seqs)
    j = select_all(W["s"], W["centre"], W["marker"]); tok_lg = token_logits(d["time_logits"].astype(np.float32), j)
    p_cls, p_tok = softmax(d["logits"].astype(np.float32)), softmax(tok_lg)
    return {"d": d, "j": j, "token_logits": tok_lg, "p_cls": p_cls, "p_tok": p_tok, "probs_cls": map_probs(W, p_cls, seqs), "probs_tok": map_probs(W, p_tok, seqs)}


def rows_of(cfg: dict, probs: dict, ev, decoders: dict) -> dict:
    out = {}
    for row, params in decoders.items():
        preds = {k: v.argmax(1) for k, v in probs.items()} if params is None else decode(params, probs)
        m, tbl = rcs.evaluate(cfg, preds, ev); out[row] = {"row": with_f1(summary(m)), "tables": tbl, "preds": preds}
    return out


def stage_check(ctx: Ctx) -> None:
    a03 = ctx.a03()["systems"]; rec = {"reproduced": {}, "parity": {}}; seqs_all = {}
    for h in HALVES:
        ev, seqs, W, meta = ctx.half(h); seqs_all[h] = seqs; ids = [q["id"] for q in seqs]
        want_mixed = ctx.a03()["classification"]["support"][h]["all mixed"]["windows"]
        if int(meta["mixed"].sum()) != int(want_mixed):
            raise SystemExit(f"{h}: {int(meta['mixed'].sum())} mixed windows, A03 counted {want_mixed}")
        for seed in ctx.ts["seeds"]:
            r = readouts(ctx, h, seed, W, seqs); choice, fixed = ctx.params(seed)
            tok = rows_of(ctx.cfg, r["probs_tok"], ev, {"argmax": None, "b3": choice, "b3_at_cls_setting": fixed})
            cls = rows_of(ctx.cfg, r["probs_cls"], ev, {"argmax": None, "b3": fixed})
            for who, rows in (("token", tok), ("cls", cls)):
                for row, x in rows.items():
                    check_stored(x["row"], a03[f"{who}_aligned_dense_s{seed}"][row][h], f"{who} s{seed} {row} {h} (A03)")
            with np.load(pinned(ctx.root, ctx.src["a03_outputs"][h][seed])) as z:
                saved = {k: z[k] for k in z.files}
            diff = float(np.abs(np.concatenate([r["probs_tok"][i] for i in ids]) - saved["sample_probs"]).max())
            same = {"sequence order": saved["sequence_ids"].tolist() == ids, "token logits": np.array_equal(saved["token_logits"], r["token_logits"]),
                    "anchors": np.array_equal(saved["anchor"], r["j"]), "offsets": np.array_equal(saved["offset"], anchor_offsets(W["s"], W["centre"], r["j"])),
                    "sample probabilities": diff == 0.0,
                    "decoded": np.array_equal(saved["decoded"], np.concatenate([np.asarray(tok["b3"]["preds"][i], np.int8) for i in ids]))}
            cells = tb.cell_probs(r["p_cls"], r["p_tok"], meta["mixed"])
            same |= {"CC endpoint": np.array_equal(cells["CC"], r["p_cls"]), "TT endpoint": np.array_equal(cells["TT"], r["p_tok"])}
            if not all(same.values()):
                raise SystemExit(f"s{seed} {h}: not reproduced: {[k for k, v in same.items() if not v]}")
            rec["reproduced"][f"s{seed}|{h}"] = "token argmax, B3 and B3 at the CLS setting, and CLS argmax and B3 rows equal A03's"
            rec["parity"][f"s{seed}|{h}"] = {k: bool(v) for k, v in same.items()} | {"max_probability_difference": diff}
        print(f"{h}: three aligned seeds reproduced and equal to A03's saved outputs", flush=True)
    rec.update({"code": code_digest(), "ok": True})
    write_json([ctx.out / "check.json", ctx.results / "checks.json"], rec, ids_of(ctx, seqs_all), "check")


# ----------------------------------------------------------------------------------------------- Task 2: accounting
def _bucket() -> dict:
    return {"segments": 0, "by_class": {c: 0 for c in CLASS_ORDER}, "short": 0, "long": 0, "xyx_a4c_a5c": 0, "_rec": set()}


def _close(b: dict) -> dict:
    rec = b.pop("_rec"); return b | units(rec)


def segment_accounting(seqs: list[dict], preds_a: dict, preds_b: dict) -> tuple[dict, dict, dict, list]:
    """a = the token system, b = the CLS system: per population the counts, every reference segment's status and the
    changed reference and unmatched predicted segments (with their source recordings); private lists for the server."""
    seg = {k: {"token": dict.fromkeys(("tp", "fp", "fn", "pred", "ref"), 0), "cls": dict.fromkeys(("tp", "fp", "fn", "pred", "ref"), 0),
               "status": dict.fromkeys(tb.STATUS, 0), **{n: _bucket() for n in ("newly_matched", "lost", "new_unmatched_pred", "removed_unmatched_pred")}} for k in KINDS}
    status_of, refs_of, private = {}, {}, []
    for qi, q in enumerate(seqs):
        cmp = tb.compare_segments(q["labels"], preds_a[q["id"]], preds_b[q["id"]]); k = q["kind"]; src = np.array([r[0] for r in q["refs"]])
        refs_of[qi] = cmp["ref"]
        for who, c in (("token", cmp["counts_a"]), ("cls", cmp["counts_b"])):
            for m, v in c.items():
                seg[k][who][m] += v
        for j, (r, st) in enumerate(zip(cmp["ref"], cmp["status"])):
            status_of[(qi, j)] = st; seg[k]["status"][st] += 1
            if st in ("token only", "cls only"):
                b = seg[k]["newly_matched" if st == "token only" else "lost"]
                b["segments"] += 1; b["by_class"][CLASS_ORDER[r.label]] += 1; b["short" if r.length <= tb.SHORT else "long"] += 1
                b["xyx_a4c_a5c"] += int(tb.xyx_middle(cmp["ref"], j)); b["_rec"] |= set(src[r.start:r.end].tolist())
                private.append({"sequence": q["id"], "segment": j, "start": r.start, "end": r.end, "label": r.label, "status": st})
        for name, segs in (("new_unmatched_pred", cmp["new_unmatched"]), ("removed_unmatched_pred", cmp["removed_unmatched"])):
            for s0, e0, lab in sorted(segs):
                b = seg[k][name]; b["segments"] += 1; b["by_class"][CLASS_ORDER[lab]] += 1; b["short" if e0 - s0 <= tb.SHORT else "long"] += 1
                b["_rec"] |= set(src[s0:e0].tolist()); private.append({"sequence": q["id"], "start": s0, "end": e0, "label": lab, "status": name})
    for k in KINDS:
        for who in ("token", "cls"):
            seg[k][who]["f1"] = tb.f1_of(seg[k][who])
        for n in ("newly_matched", "lost", "new_unmatched_pred", "removed_unmatched_pred"):
            seg[k][n] = _close(seg[k][n])
    return seg, status_of, refs_of, private


def window_accounting(W: dict, meta: dict, correct_a: np.ndarray, correct_b: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    """Changed requested-window correctness by population, mixed or pure window and distance bin, with sources."""
    fixed, new = correct_a & ~correct_b, ~correct_a & correct_b; out = {}
    for k in KINDS:
        out[k] = {}
        for ctx_name in (("pure",) if k == "native" else ("mixed", "pure")):
            m = (meta["kind"] == k) & (meta["mixed"] == (ctx_name == "mixed")); out[k][ctx_name] = {}
            for b in tb.BINS:
                mb = m & (meta["bin"] == b)
                if mb.any():
                    out[k][ctx_name][b] = {n: {"windows": int((mb & x).sum()), **units(meta["recording"][mb & x], meta["session"][mb & x])}
                                           for n, x in (("fixed", fixed), ("new", new))} | {"windows": int(mb.sum())}
    return out, fixed, new


def containing_accounting(W: dict, meta: dict, fixed: np.ndarray, new: np.ndarray, status_of: dict, refs_of: dict) -> dict:
    """For the stream windows whose correctness changed, the status of the reference segment holding the request."""
    out = {}
    for k in ("pairs", "multi"):
        out[k] = {}
        for ctx_name in ("mixed", "all"):
            m = (meta["kind"] == k) & (meta["mixed"] if ctx_name == "mixed" else True)
            res, segs = {}, {}
            for n, x in (("fixed", fixed), ("new", new)):
                rows = np.flatnonzero(m & x); by_win = dict.fromkeys(tb.STATUS, 0); keys = {}; ses = {s: set() for s in tb.STATUS}
                for i in rows:
                    q = int(W["seq"][i]); j = tb.segment_index(refs_of[q], int(W["centre"][i])); st = status_of[(q, j)]
                    by_win[st] += 1; keys[(q, j)] = st; ses[st].add(meta["session"][i])
                res[n] = {"windows_by_status": by_win, "segments_by_status": {s: sum(v == s for v in keys.values()) for s in tb.STATUS},
                          "sessions_by_status": {s: len(v) for s, v in ses.items()}, "sessions": len(set().union(*ses.values()))}
                segs[n] = set(keys)
            res["segments_with_both"] = len(segs["fixed"] & segs["new"])
            out[k][ctx_name] = res
    return out


def probability_only(seqs: list[dict], W: dict, served: dict, arg_a, arg_b, p_a, p_b, preds_a: dict, preds_b: dict) -> dict:
    """Windows whose argmax is unchanged but whose probabilities move, and decoded-sample changes split by whether the
    serving window's argmax changed."""
    same = arg_a == arg_b; tv = 0.5 * np.abs(np.asarray(p_a, float) - np.asarray(p_b, float)).sum(1)
    out = {"windows_same_argmax": int(same.sum()), "windows_changed_argmax": int((~same).sum()), "tv_same_argmax": stats(tv[same]),
           "tv_changed_argmax": stats(tv[~same]), "decoded_changes": {}}
    for k in KINDS:
        n_any, n_chg, n_same, n_seq = 0, 0, 0, 0
        for q in seqs:
            if q["kind"] != k:
                continue
            diff = np.asarray(preds_a[q["id"]]) != np.asarray(preds_b[q["id"]]); rows = served[q["id"]]; chg = ~same[rows]
            n_any += int(diff.sum()); n_chg += int((diff & chg).sum()); n_same += int((diff & ~chg).sum()); n_seq += int(diff.any())
        out["decoded_changes"][k] = {"samples": n_any, "serving_window_argmax_changed": n_chg, "serving_window_argmax_same": n_same, "sequences": n_seq}
    return out


def switch_accounting(seqs: list[dict], systems: dict) -> tuple[dict, dict]:
    """Label switches inside native clips per system and decoding, with the clips, recordings and sessions concerned."""
    out, private = {}, {}
    for who, by in systems.items():
        out[who] = {}
        for dec, preds in by.items():
            cuts = {q["id"]: int(tb.view_changes(preds[q["id"]]).size) for q in seqs if q["kind"] == "native"}
            clips = [c for c, n in cuts.items() if n]
            out[who][dec] = {"switches": int(sum(cuts.values())), "clips": len(cuts), "clips_with_switches": len(clips), **units(clips)}
            private[f"{who}|{dec}"] = clips
    return out, private


def account(seqs, W, meta, served, a: dict, b: dict) -> tuple[dict, dict]:
    """a, b: {"b3": decoded samples, "argmax": per-sample argmax, "win_arg": window argmax, "win_p": window probabilities}."""
    seg, status_of, refs_of, private_seg = segment_accounting(seqs, a["b3"], b["b3"])
    correct_a = a["win_arg"] == W["label"]; correct_b = b["win_arg"] == W["label"]
    win, fixed, new = window_accounting(W, meta, correct_a, correct_b)
    cont = containing_accounting(W, meta, fixed, new, status_of, refs_of)
    prob = probability_only(seqs, W, served, a["win_arg"], b["win_arg"], a["win_p"], b["win_p"], a["b3"], b["b3"])
    sw, private_sw = switch_accounting(seqs, {"token": {"argmax": a["argmax"], "b3": a["b3"]}, "cls": {"argmax": b["argmax"], "b3": b["b3"]}})
    private = {"segments": private_seg, "switch_clips": private_sw,
               "changed_windows": [{"sequence": seqs[int(W["seq"][i])]["id"], "centre": int(W["centre"][i]), "change": "fixed" if fixed[i] else "new",
                                    "mixed": bool(meta["mixed"][i]), "bin": str(meta["bin"][i])} for i in np.flatnonzero(fixed | new)]}
    return {"segments": seg, "windows": win, "containing": cont, "probability_only": prob, "switches": sw}, private


# ----------------------------------------------------------------------------------------------- Task 3: the four cells
def kind_counts(seqs: list[dict], preds: dict, tables: dict) -> dict:
    """TP / FP / FN per population with the exposed matching, checked against the evaluator's stream tables."""
    out = {k: dict.fromkeys(("tp", "fp", "fn", "pred", "ref"), 0) for k in KINDS}
    for q in seqs:
        c = tb.seg_counts(labels_to_segments(np.asarray(preds[q["id"]])), labels_to_segments(np.asarray(q["labels"])))
        for m, v in c.items():
            out[q["kind"]][m] += v
    for k, t in (("multi", "multi_seg"), ("pairs", "pairs_seg")):
        want = (int(tables[t]["seg_matched"].sum()), int(tables[t]["seg_pred"].sum()), int(tables[t]["seg_true"].sum()))
        if (out[k]["tp"], out[k]["pred"], out[k]["ref"]) != want:
            raise SystemExit(f"{k}: the exposed matching disagrees with the evaluator's tables")
    return {k: v | {"f1": tb.f1_of(v)} for k, v in out.items()}


def cells_of(ctx: Ctx, ev, W: dict, seqs: list[dict], meta: dict, r: dict, fixed: dict) -> dict:
    cells = tb.cell_probs(r["p_cls"], r["p_tok"], meta["mixed"]); out = {}
    for c in tb.CELLS:
        probs = map_probs(W, cells[c], seqs)
        rows = rows_of(ctx.cfg, probs, ev, {"argmax": None, "b3": fixed})
        out[c] = {dec: rows[dec] | {"segments": kind_counts(seqs, rows[dec]["preds"], rows[dec]["tables"])} for dec in ("b3", "argmax")}
        out[c]["win_arg"] = cells[c].argmax(1); out[c]["win_p"] = cells[c]
    return out


def contrasts_of(cells: dict) -> dict:
    """The three paired contrasts with the existing paired bootstrap and the interaction with the same draws; the
    contrast bootstrap must reproduce the paired one exactly on TT - CC."""
    out = {}
    for dec in ("b3", "argmax"):
        tabs = {c: cells[c][dec]["tables"] for c in tb.CELLS}
        for name, coef in tb.CONTRASTS.items():
            a = [c for c, w in coef.items() if w == 1 and c != "CC"]
            out[f"{dec}|{name}"] = paired_bootstrap(tabs[a[0]], tabs["CC"]) if len(coef) == 2 else tb.contrast_bootstrap(tabs, coef)
        if dec == "b3":
            mine = tb.contrast_bootstrap({"TT": tabs["TT"], "CC": tabs["CC"]}, {"TT": 1, "CC": -1})
            if json.dumps(mine, sort_keys=True, default=float) != json.dumps(out["b3|TT - CC"], sort_keys=True, default=float):
                raise SystemExit("the contrast bootstrap does not reproduce the paired bootstrap")
    return out


# ----------------------------------------------------------------------------------------------- Task 4: overlapping anchors
CONTEXTS = ("all", "both pure", "at least one mixed")
SUBSETS = ("main", "main, equal marker", "main, changed marker", "broad")


def pair_frame(W: dict, seqs: list[dict], meta: dict) -> dict:
    P = tb.anchor_pairs(W["s"], W["centre"], W["marker"], W["seq"], select_all(W["s"], W["centre"], W["marker"]))
    q, t = P["seq"], P["t"]
    P["kind"] = np.array([seqs[x]["kind"] for x in q]) if len(q) else np.zeros(0, str)
    P["mixed_any"] = meta["mixed"][P["i1"]] | meta["mixed"][P["i2"]] if len(q) else np.zeros(0, bool)
    P["label"] = np.array([int(seqs[a]["labels"][b]) for a, b in zip(q, t)], int)
    P["frame"] = np.array([json.dumps(list(seqs[a]["refs"][b])) for a, b in zip(q, t)])
    P["recording"] = np.array([seqs[a]["refs"][b][0] for a, b in zip(q, t)]); P["session"] = np.array([session_of(v) for v in P["recording"]])
    P["bin"] = np.array([tb.distance_bin(tb.request_distance(int(b), meta["changes"][a])) for a, b in zip(q, t)])
    return P


def subset_mask(P: dict, sub: str, kind: str, ctx_name: str) -> np.ndarray:
    m = {"main": P["main"], "main, equal marker": P["main"] & P["marker_equal"], "main, changed marker": P["main"] & ~P["marker_equal"],
         "broad": P["broad"]}[sub] & (P["kind"] == kind)
    return m & (~P["mixed_any"] if ctx_name == "both pure" else P["mixed_any"] if ctx_name == "at least one mixed" else True)


def stability_structure(P: dict) -> dict:
    support = {}
    for sub in SUBSETS:
        support[sub] = {}
        for k in KINDS:
            support[sub][k] = {}
            for c in CONTEXTS:
                m = subset_mask(P, sub, k, c)
                obs = set(zip(P["i1"][m].tolist(), P["j1"][m].tolist())) | set(zip(P["i2"][m].tolist(), P["j2"][m].tolist()))
                support[sub][k][c] = {"groups": int(np.unique(P["group"][m]).size), "observations": len(obs), "pairs": int(m.sum()),
                                      "physical_frames": int(np.unique(P["frame"][m]).size), **units(P["recording"][m], P["session"][m]),
                                      "selected_output_anchor": {"at least one": int((m & (P["sel1"] | P["sel2"])).sum()), "both": int((m & P["sel1"] & P["sel2"]).sum()),
                                                                 "neither": int((m & ~P["sel1"] & ~P["sel2"]).sum())}}
    dist = {}
    for k in KINDS:
        m = subset_mask(P, "main", k, "all")
        count = lambda xs: {str(x): int(n) for x, n in zip(*np.unique(np.asarray(xs), return_counts=True))} if len(xs) else {}  # noqa: E731
        trip = lambda a: ["".join(map(str, x)) for x in a]  # noqa: E731
        dist[k] = {"anchor pairs": count([f"{a}-{b}" for a, b in zip(P["j1"][m], P["j2"][m])]),
                   "marker-relative positions": count(np.r_[P["pos1"][m], P["pos2"][m]]),
                   "marker triples": count([f"{a}/{b}" for a, b in zip(trip(P["mtriple1"][m]), trip(P["mtriple2"][m]))]),
                   "centre distance": count(P["centre_distance"][m]), "boundary distance": count(P["bin"][m])}
    return {"support": support, "distributions": dist}


def stability_rates(P: dict, time_logits: np.ndarray) -> dict:
    out = {}
    if not len(P["group"]):
        return {sub: {k: {c: {"pairs": 0} for c in CONTEXTS} for k in KINDS} for sub in SUBSETS}
    l1, l2 = time_logits[P["i1"], P["j1"]].astype(float), time_logits[P["i2"], P["j2"]].astype(float)
    p1 = np.exp(l1 - l1.max(1, keepdims=True)); p1 /= p1.sum(1, keepdims=True); p2 = np.exp(l2 - l2.max(1, keepdims=True)); p2 /= p2.sum(1, keepdims=True)
    a1, a2 = p1.argmax(1), p2.argmax(1); dis = a1 != a2; js = tb.js_divergence(p1, p2); c1, c2 = a1 == P["label"], a2 == P["label"]
    for sub in SUBSETS:
        out[sub] = {}
        for k in KINDS:
            out[sub][k] = {}
            for c in CONTEXTS:
                m = subset_mask(P, sub, k, c); w = tb.group_weights(P["group"], m)
                out[sub][k][c] = {"pairs": int(m.sum()), "disagreeing pairs": int((m & dis).sum()), "disagreement": tb.weighted_rate(dis, w),
                                  "js_mean": tb.weighted_rate(js, w), "both correct": tb.weighted_rate(c1 & c2, w), "one correct": tb.weighted_rate(c1 ^ c2, w),
                                  "both wrong": tb.weighted_rate(~c1 & ~c2, w), "disagreeing sessions": len(set(P["session"][m & dis].tolist()))}
    return out


# ----------------------------------------------------------------------------------------------- the run
def across_seeds(private: dict, seqs: list[dict]) -> dict:
    """Distinct reference segments and native clips over the three seeds, so seeds never count as extra cases."""
    byid = {q["id"]: q for q in seqs}; out = {}
    for status, name in (("token only", "newly matched segments"), ("cls only", "lost segments")):
        keys, recs = set(), set()
        for p in private.values():
            for e in p["segments"]:
                if e["status"] == status:
                    keys.add((e["sequence"], e["segment"])); recs |= {r[0] for r in byid[e["sequence"]]["refs"][e["start"]:e["end"]]}
        out[name] = {"segments": len(keys), **units(recs)}
    for name in ("token|argmax", "token|b3", "cls|argmax", "cls|b3"):
        clips = set().union(*[set(p["switch_clips"][name]) for p in private.values()])
        out[f"native clips with switches ({name.replace('|', ', ')})"] = {"clips": len(clips), **units(clips)}
    return out


def run_half(ctx: Ctx, h: str) -> tuple[dict, dict, list]:
    ev, seqs, W, meta = ctx.half(h); served = sample_windows(W, seqs); a03 = ctx.a03()["systems"]; seeds = ctx.ts["seeds"]
    res = {"actual": {}, "fixed": {}, "cells": {}, "contrasts": {}, "rates": {}}; private = {"actual": {}, "fixed": {}}
    for seed in seeds:
        r = readouts(ctx, h, seed, W, seqs); choice, fixed = ctx.params(seed); key = f"s{seed}|{h}"
        tok = rows_of(ctx.cfg, r["probs_tok"], ev, {"argmax": None, "b3": choice}); cls = rows_of(ctx.cfg, r["probs_cls"], ev, {"argmax": None, "b3": fixed})
        for who, rows in (("token", tok), ("cls", cls)):
            for row, x in rows.items():
                check_stored(x["row"], a03[f"{who}_aligned_dense_s{seed}"][row][h], f"{who} s{seed} {row} {h} (A03)")
        side = lambda rows, p: {"b3": rows["b3"]["preds"], "argmax": rows["argmax"]["preds"], "win_arg": p.argmax(1), "win_p": p}  # noqa: E731
        res["actual"][key], private["actual"][key] = account(seqs, W, meta, served, side(tok, r["p_tok"]), side(cls, r["p_cls"]))
        cells = cells_of(ctx, ev, W, seqs, meta, r, fixed)
        for c, who, row in (("CC", "cls", "b3"), ("CC", "cls", "argmax"), ("TT", "token", "b3_at_cls_setting"), ("TT", "token", "argmax")):
            check_stored(cells[c]["b3" if row != "argmax" else "argmax"]["row"], a03[f"{who}_aligned_dense_s{seed}"][row][h], f"{c} s{seed} {row} {h} (A03)")
        res["fixed"][key], private["fixed"][key] = account(seqs, W, meta, served, side(cells["TT"], cells["TT"]["win_p"]), side(cells["CC"], cells["CC"]["win_p"]))
        res["cells"][key] = {dec: {c: {"row": cells[c][dec]["row"], "segments": cells[c][dec]["segments"]} for c in tb.CELLS} for dec in ("b3", "argmax")}
        for k, v in contrasts_of(cells).items():
            dec, name = k.split("|"); res["contrasts"][f"{dec}|s{seed}|{h}|{name}"] = v
        print(f"{h} seed {seed}: accounting, cells and contrasts done", flush=True)
    P = pair_frame(W, seqs, meta); res["structure"] = stability_structure(P)
    for arm in (ctx.ts["arms"]["primary"], *ctx.ts["arms"]["controls"]):
        for seed in seeds:
            res["rates"][f"{arm}_s{seed}|{h}"] = stability_rates(P, ctx.prediction(h, arm, seed, W, seqs)["time_logits"])
    res["across"] = {"actual": across_seeds(private["actual"], seqs), "fixed": across_seeds(private["fixed"], seqs)}
    res["inputs"] = {"sequences": {k: int(sum(q["kind"] == k for q in seqs)) for k in KINDS}, "windows": {k: int((meta["kind"] == k).sum()) for k in KINDS},
                     "mixed_windows": {k: int((meta["mixed"] & (meta["kind"] == k)).sum()) for k in KINDS}}
    private["pairs"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in P.items()}
    private["sequence_ids"] = [q["id"] for q in seqs]
    print(f"{h}: overlapping-anchor pairs {len(P['group'])}, main {int(P['main'].sum())}", flush=True)
    return res, private, seqs


METRICS = ("multi segment F1@50", "native macro-F1", "native cuts per clip", "multi boundary recall", "multi boundary F1", "same-class cut rate")


def decide(ctx: Ctx, out: dict) -> dict:
    seeds = ctx.ts["seeds"]; C = out["contrasts"]
    ver = {dec: {name: {m: {h: consistency([C[f"{dec}|s{s}|{h}|{name}"][m] for s in seeds]) for h in HALVES} for m in METRICS} for name in tb.CONTRASTS}
           for dec in ("b3", "argmax")}
    local, local_ses = [], []
    for h in HALVES:
        for s in seeds:
            x = out["actual"][f"s{s}|{h}"]["containing"]["multi"]["mixed"]["fixed"]; tot = sum(x["windows_by_status"].values())
            local.append((x["windows_by_status"]["both"] + x["windows_by_status"]["cls only"]) / tot if tot else None); local_ses.append(x["sessions"])
    seg = "multi segment F1@50"
    trade = {h: {"tc": [C[f"b3|s{s}|{h}|TC - CC"][seg]["delta"] for s in seeds], "ct": [C[f"b3|s{s}|{h}|CT - CC"][seg]["delta"] for s in seeds],
                 "tc_verdict": ver["b3"]["TC - CC"][seg][h], "ct_verdict": ver["b3"]["CT - CC"][seg][h]} for h in HALVES}
    R, S = out["stability"]["rates"], out["stability"]["structure"]
    anchor = [sum(R[f"aligned_dense_s{s}|{h}"]["main"][k]["all"].get("disagreeing pairs", 0) for k in KINDS) for h in HALVES for s in seeds]
    anchor_ses = [R[f"aligned_dense_s{s}|{h}"]["main"][k]["all"].get("disagreeing sessions", 0) for h in HALVES for s in seeds for k in KINDS
                  if R[f"aligned_dense_s{s}|{h}"]["main"][k]["all"].get("disagreeing pairs", 0)]
    tcs = [x for h in HALVES for x in trade[h]["tc"]]
    flags = {"no main-subset pair holds A03's selected output anchor in both windows":
             sum(S[h]["support"]["main"][k]["all"]["selected_output_anchor"]["both"] for h in HALVES for k in KINDS) == 0,
             "TC - CC segment F1@50 point estimates of both signs across the six cells": any(x > 0 for x in tcs) and any(x < 0 for x in tcs),
             "a reading rests on fewer than 10 sessions": min(local_ses) < tb.FEW or (bool(anchor_ses) and min(anchor_ses) < tb.FEW),
             "the actual systems use differing decoders": True}
    return {"verdicts": ver, "readings": tb.readings(local, trade, anchor, flags, ctx.ts["readings"]),
            "inputs": {"local_shares": local, "local_sessions": local_ses, "trade": trade, "anchor_disagreements": anchor}}


def identity(ctx: Ctx) -> dict:
    a3 = ctx.a03src()
    return {"code": code_digest(), "config_sha256": sha256_file(ctx.root / ctx.config), "task_sheet": ctx.ts["task_sheet"]["sha256"],
            **{k: ctx.src[k]["sha256"] for k in ("a03_config", "a03_summary", "a03_readme", "a03_choices")},
            "predictions": {h: {k: v["sha256"] for k, v in a3["predictions"][h].items()} for h in HALVES},
            "a03_outputs": {h: {str(s): v["sha256"] for s, v in ctx.src["a03_outputs"][h].items()} for h in HALVES}}


def publish(ctx: Ctx, flat: dict, private: dict, ids: set) -> bool:
    """Atomic: everything goes to a temporary run directory that is renamed when complete. A finished run is never
    overwritten; a repeat must reproduce it and writes nothing."""
    final, tmp = ctx.out / "run", ctx.out / "run.tmp"
    if (final / "summary.json").is_file():
        if json.loads((final / "summary.json").read_text()) != flat:
            raise SystemExit("an earlier complete run differs from this one; nothing is overwritten")
        print("the complete run is reproduced exactly; nothing is rewritten", flush=True)
        return False
    if tmp.exists():
        tmp.rename(ctx.out / f"run.incomplete-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    (tmp / "private").mkdir(parents=True)
    for h, p in private.items():
        (tmp / "private" / f"{h}.json").write_text(json.dumps(p, default=str))
    md = tb.tables_md(flat)
    write_atomic(tmp / "summary.json", json.dumps(flat, indent=1)); write_atomic(tmp / "report_tables.md", md)
    tmp.rename(final)
    full = json.loads(json.dumps(flat | {"cost": {"cpu_hours_before_this_stage": ledger_hours(ctx.ledger, "cpu"), "gpu_hours": 0.0},
                                         "run": {"started": getattr(ctx, "started", None)}}, default=str))
    assert_no_ids(full, ids, "results summary"); ctx.results.mkdir(parents=True, exist_ok=True)
    write_atomic(ctx.results / "summary.json", json.dumps(full, indent=1)); write_atomic(ctx.results / "report_tables.md", tb.tables_md(full))
    return True


def stage_run(ctx: Ctx) -> None:
    out = {"identity": identity(ctx), "actual": {}, "fixed": {}, "cells": {}, "contrasts": {}, "stability": {"structure": {}, "rates": {}},
           "across_seeds": {}, "inputs": {}}
    private, seqs_all = {}, {}
    for h in HALVES:
        res, private[h], seqs_all[h] = run_half(ctx, h)
        for k in ("actual", "fixed", "cells", "contrasts"):
            out[k] |= res[k]
        out["stability"]["structure"][h] = res["structure"]; out["stability"]["rates"] |= res["rates"]
        out["across_seeds"][h] = res["across"]; out["inputs"][h] = res["inputs"]
    d = decide(ctx, out); out["contrast_verdicts"] = d["verdicts"]; out["readings"] = d["readings"]; out["reading_inputs"] = d["inputs"]; out["post_hoc"] = []
    ids = ids_of(ctx, seqs_all); flat = json.loads(json.dumps(out, default=str)); assert_no_ids(flat, ids, "summary")
    if any(x in tb.tables_md(flat) for x in ids):
        raise SystemExit("an id appears in the report tables")
    if publish(ctx, flat, private, ids):
        print(json.dumps(out["readings"]["applied"]), flush=True)


STAGES = {"prepare": stage_prepare, "check": stage_check, "run": stage_run}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    for s in STAGES:
        g.add_argument(f"--{s}", dest=s, action="store_true")
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config)
    stage = next(s for s in STAGES if getattr(a, s))
    run_stage(ctx, stage, STAGES[stage])


if __name__ == "__main__":
    main()
