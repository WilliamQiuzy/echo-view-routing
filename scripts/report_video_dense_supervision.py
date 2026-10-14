#!/usr/bin/env python
"""Evaluate and report the video dense supervision round once (docs/superpowers/plans/2026-10-13-video-dense-supervision.md,
Task 4; configs/experiments/video_dense_supervision_v1.yaml). `--evaluate` runs after the nine trainings are frozen.

1. Runs: identities recomputed, 3,750 updates, final and frozen weights checked; the V01 sources unchanged. Only then are
   the evaluation references' pins checked and development frames loaded.
2. Windows: V01's (centres every 5 samples, 16 frames, the marker at the middle target slot, Ordered order). Every model's
   per-window CLS logits and time logits and its per-sample probabilities are saved on the server. Systems use the CLS
   probabilities only: direct argmax, B3 chosen once on calibration (40 settings) and the seed's Ordered numeric B3 setting
   (decoding sensitivity).
3. References, each reproducing its stored rows before use: frozen Ordered (V01 saved outputs), historical Continued CLS
   (V02 saved outputs), E0/E1/B7 + B3, historical E1 + ASRF, paired ASRF, Original FACT, and A01's Linear + B3, MS-TCN
   and ASRF (frozen heads on their saved features); G1 + ASRF from stored rows.
4. Paired system comparisons and verdicts; mixed-window target accuracy (session bootstrap with the stream sensitivity);
   units; the goal; the fixed readings; the pre-specified anchor diagnostics of the time logits.
Results go to the results directory through id guards after every stage.

  python scripts/report_video_dense_supervision.py --config configs/experiments/video_dense_supervision_v1.yaml --evaluate
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from argparse import Namespace  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.prepare_video_dense_supervision as pd  # noqa: E402
import scripts.raw_video_context_tables as rt  # noqa: E402
import scripts.report_center_supervision as rcs  # noqa: E402
import scripts.report_fact_asrf as ra  # noqa: E402
import scripts.report_raw_video_context as rr  # noqa: E402
import scripts.report_target_patch_readout as rtp  # noqa: E402
import scripts.target_patch_readout_tables as pt  # noqa: E402
import scripts.train_video_dense_supervision as tv  # noqa: E402
import scripts.video_dense_supervision_tables as vt  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.ev9v_complementarity import decode  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402
from scripts.report_fold_output_correction import FLAT, boundary_changes, native_changes, segment_changes, summary, write_atomic  # noqa: E402
from scripts.report_model_development import paired_bootstrap  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import echo_routing.vit.target_mvit as tm  # noqa: E402
import echo_routing.vit.video_dense_supervision as vds  # noqa: E402
from echo_routing.config import gpu_memory_fraction  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402

NS = "video_dense_supervision_v1"
HALVES = vt.HALVES
ARMS = vds.ARMS
PAIRS = {"aligned - center_only": ("aligned_dense", "center_only"), "aligned - Ordered": ("aligned_dense", "ordered"),
         "aligned - shuffled": ("aligned_dense", "shuffled_dense"), "center_only - Ordered": ("center_only", "ordered"),
         "shuffled - center_only": ("shuffled_dense", "center_only"), "aligned - Continued CLS": ("aligned_dense", "continued_cls")}
REFS = {"aligned - (E1 + B3)": "E1 + B3", "aligned - (E0 + B3)": "E0 + B3", "aligned - (B7 + B3)": "B7 + B3", "aligned - Original FACT": "Original FACT s{k}",
        "aligned - paired ASRF": "paired ASRF s{k}", "aligned - historical E1 + ASRF": "E1 + ASRF s{k}", "aligned - A01 ASRF": "A01 ASRF s{k}",
        "aligned - A01 MS-TCN": "A01 MS-TCN s{k}", "aligned - A01 Linear + B3": "A01 Linear + B3 s{k}"}
MIXED_PAIRS = ("aligned - center_only", "aligned - shuffled", "aligned - Ordered", "center_only - Ordered", "shuffled - center_only")
SUBSETS = ("all mixed", "<= 7 of 16", "8 of 16", "9-11 of 16", "12-15 of 16", "A4C/A5C pair windows")


# ----------------------------------------------------------------------------------------------- runs and pins
def verify_runs(ctx) -> dict:
    """Every run finished under its identity with 3,750 updates and frozen intact; the V01 sources unchanged."""
    runs = {}
    for key in ctx.vd["queue"]:
        arm, s = key.rsplit("_s", 1)[0], int(key.rsplit("_s", 1)[1]); man_path = ctx.out / "train" / key / "run_manifest.json"
        if not man_path.is_file():
            raise SystemExit(f"{key}: no finished run")
        man = json.loads(man_path.read_text()); ident = tv.identity(ctx, arm, s, ctx.manifest(s)[1], ctx.aux(s)[1], ctx.vd["source"]["weights"][s]["sha256"])
        frozen = ctx.root / "models_frozen" / NS / key / "final.pt"
        if man["identity"] != ident or int(man["updates"]) != ctx.total or sha256_file(Path(man["final_checkpoint"])) != man["final_sha256"]:
            raise SystemExit(f"{key}: identity, update count or final weights do not verify")
        if not frozen.is_file() or sha256_file(frozen) != man["final_sha256"]:
            raise SystemExit(f"{key}: the frozen copy is missing or differs")
        runs[key] = {"arm": arm, "seed": s, "manifest": man, "frozen": str(frozen.parent.relative_to(ctx.root))}
    for pin in ctx.vd["source"]["weights"].values():
        pd.pinned(ctx.root, pin)
    return runs


def check_reference_pins(ctx) -> dict:
    """The evaluation references, checked only now (after training): summaries, saved outputs and A01's config and features."""
    src = ctx.vd["source"]; pins = {k: v for k, v in src["reports"].items()} | {"a01_config": src["a01"]["config"]}
    pins |= {f"a01_features_s{k}": v for k, v in src["a01"]["features"].items()}
    for name in ("ordered_outputs", "continued_cls_outputs"):
        pins |= {f"{name}_{h}_s{k}": v for h, per in src[name].items() for k, v in per.items()}
    return {name: sha256_file(pd.pinned(ctx.root, pin)) for name, pin in pins.items()}


# ----------------------------------------------------------------------------------------------- the round's models
@torch.no_grad()
def model_outputs(model: vds.DenseMViT, W: dict, frames: np.ndarray, device, batch: int = 32) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-window CLS logits (N, 5) and time logits (N, 8, 5), Ordered order; the first batch's CLS logits must equal model()."""
    cls, tim = [], []; torch.cuda.synchronize(); t0 = time.time()
    for b in range(0, len(W["ref"]), batch):
        u8 = torch.from_numpy(frames[W["ref"][b:b + batch]]).to(device, non_blocking=True); rgb = tm.to_rgb(u8)
        mk = tm.marker_planes(torch.from_numpy(W["marker"][b:b + batch]).long().to(device)); z_cls, z_time = model.forward_with_time(rgb, mk)
        if b == 0 and not torch.equal(z_cls, model(rgb, mk)):
            raise SystemExit("the CLS logits of forward_with_time differ from the model's forward")
        cls.append(z_cls.float().cpu().numpy()); tim.append(z_time.float().cpu().numpy())
    torch.cuda.synchronize()
    return np.concatenate(cls), np.concatenate(tim), time.time() - t0


def cached_outputs(path: Path, ident: str, W: dict, seqs_h: list[dict]):
    if not path.is_file():
        return None
    with np.load(path) as z:
        same = (str(z["identity"]) == ident and z["sequence_ids"].tolist() == [q["id"] for q in seqs_h] and np.array_equal(z["seq"], W["seq"])
                and np.array_equal(z["centre"], W["centre"]) and np.array_equal(z["marker"], W["marker"]))
        return (z["logits"].astype(np.float32), z["time_logits"].astype(np.float32), float(z["seconds"])) if same else None


def save_outputs(path: Path, ident: str, W: dict, lg: np.ndarray, tl: np.ndarray, probs: dict, seqs_h: list[dict], seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); ids = [q["id"] for q in seqs_h]
    np.savez(path, identity=ident, sequence_ids=np.asarray(ids), seq=W["seq"], centre=W["centre"], marker=W["marker"], logits=lg.astype(np.float32),
             time_logits=tl.astype(np.float32), sample_offsets=np.cumsum([0, *[len(probs[i]) for i in ids]]),
             sample_probs=np.concatenate([probs[i] for i in ids]).astype(np.float32), seconds=seconds)


# ----------------------------------------------------------------------------------------------- references
def saved_reference(ctx, cfg: dict, W: dict, seqs: dict, inputs, pin_key: str, summary_key: str, stored_name: str, key_name: str) -> tuple[dict, dict]:
    """A frozen model's saved per-window outputs decoded with its stored B3 setting; the stored rows must be reproduced."""
    stored = json.loads(pd.pinned(ctx.root, ctx.vd["source"]["reports"][summary_key]).read_text())["systems"]; systems, logits = {}, {}
    for s in ctx.seeds:
        key, probs = key_name.format(k=s), {}; logits[key] = {}
        for h in HALVES:
            with np.load(pd.pinned(ctx.root, ctx.vd["source"][pin_key][h][s])) as z:
                if z["sequence_ids"].tolist() != [q["id"] for q in seqs[h]] or not (np.array_equal(z["centre"], W[h]["centre"]) and np.array_equal(z["marker"], W[h]["marker"])):
                    raise SystemExit(f"{key} {h}: the saved windows differ from this round's windows")
                lg = z["logits"].astype(np.float32)
            logits[key][h] = lg; probs[h] = rr.per_sample(W[h], lg, seqs[h])
        src = stored[stored_name.format(k=s)]; params = src["b3_choice"]["params"]; systems[key] = {"argmax": {}, "b3": {}, "b3_choice": src["b3_choice"]}
        for h in HALVES:
            for row, preds in (("argmax", {k: v.argmax(1) for k, v in probs[h].items()}), ("b3", decode(params, probs[h]))):
                m, t = rcs.evaluate(cfg, preds, inputs.halves[h]); r = summary(m); want = src[row][h]
                diff = {f: (r[f], want[f]) for f in FLAT if r[f] != want[f]}
                if diff:
                    raise SystemExit(f"{key} {row} {h}: the stored rows are not reproduced ({diff})")
                systems[key][row][h] = {"row": r, "tables": t, "preds": preds}
    return systems, logits


def a01_reference(ctx, seqs: dict, inputs, device) -> dict:
    """A01's frozen heads on A01's saved development features; Linear + B3, MS-TCN and ASRF must reproduce their stored rows."""
    import scripts.report_video_feature_temporal as rvf
    import scripts.train_video_feature_temporal as tvf
    src = ctx.vd["source"]["a01"]; path = str(pd.pinned(ctx.root, src["config"]).relative_to(ctx.root)); a_cfg = load_cfg(Namespace(config=path, set=[]))
    actx = tvf.Ctx(a_cfg, path); order = [q["id"] for h in HALVES for q in seqs[h]]; feats = {}
    for s in ctx.seeds:
        with np.load(pd.pinned(ctx.root, src["features"][s])) as z:
            if z["ids"].tolist() != order or z["samples"].dtype != np.float32:
                raise SystemExit(f"A01 features s{s}: other sequences or not float32")
            off, samples = z["sample_offsets"], z["samples"]; feats[s] = {}; i0 = 0
            for h in HALVES:
                feats[s][h] = {q["id"]: samples[off[i0 + j]:off[i0 + j + 1]] for j, q in enumerate(seqs[h])}; i0 += len(seqs[h])
    systems, _, _ = rvf.new_systems(actx, a_cfg, inputs, seqs, feats, device)
    stored = json.loads(pd.pinned(ctx.root, ctx.vd["source"]["reports"]["a01_summary"]).read_text())["systems"]; out = {}
    for s in ctx.seeds:
        for name in ("Linear + B3", "MS-TCN", "ASRF"):
            k = f"{name} s{s}"
            for h in HALVES:
                diff = {f: (systems[k][h]["row"].get(f), v) for f, v in stored[k][h].items() if systems[k][h]["row"].get(f) != v}
                if diff:
                    raise SystemExit(f"A01 {k} {h}: the stored row is not reproduced ({diff})")
            out[f"A01 {k}"] = {"row": {h: {"row": systems[k][h]["row"], "tables": systems[k][h]["tables"], "preds": systems[k][h]["preds"]} for h in HALVES}}
    return out


# ----------------------------------------------------------------------------------------------- analyses
def mixed_analysis(ctx, logits: dict, W: dict, meta: dict) -> dict:
    subs = pt.mixed_subsets(W["labels16"], W["label"], meta["mixed"]); y = W["label"]; correct = {k: v.argmax(1) == y for k, v in logits.items()}
    acc = {k: {n: (float(c[rows].mean()) if rows.size else None) for n, rows in subs.items()} for k, c in correct.items()}
    support = {n: pt.subset_support(rows, meta["recording"], meta["session"], meta["run"], y) for n, rows in subs.items()}
    comp = {}
    for s in ctx.seeds:
        for name in MIXED_PAIRS:
            a, b = PAIRS[name]; ka, kb = f"{a}_s{s}", f"{b}_s{s}"
            for n, rows in subs.items():
                ca, cb = correct[ka][rows], correct[kb][rows]
                comp[f"s{s}|{name}|{n}"] = {"session": pt.group_bootstrap(ca, cb, meta["session"][rows]), "stream": pt.group_bootstrap(ca, cb, meta["stream"][rows]),
                                            "changes": rt.changes(ca, cb, meta["recording"][rows], meta["session"][rows])}
    return {"accuracy": acc, "support": support, "comparisons": comp}


def anchor_analysis(time_logits: dict, W: dict, meta: dict, refs: list, ids_session) -> dict:
    """Pre-specified diagnostics of the time logits: accuracy and class recall at the anchors; both anchors at changes."""
    lab = vt.anchor_labels(W["labels16"], vds.ANCHORS); mixed = np.flatnonzero(meta["mixed"]); native = np.flatnonzero(meta["kind"] == "native")
    pts = vt.change_points(lab, mixed); slot = list(vds.ANCHORS)

    def sources(i: int, j: int):
        recs = {refs[W["ref"][i, slot[j]]][0], refs[W["ref"][i, slot[j + 1]]][0]}
        return recs, {ids_session(r) for r in recs}
    out = {}
    for key, tl in time_logits.items():
        pred = tl.argmax(-1)
        out[key] = {"mixed": vt.anchor_accuracy(pred, lab, mixed), "native": vt.anchor_accuracy(pred, lab, native),
                    "changes": vt.change_summary(pred, lab, pts, sources)}
    return out


def system_comparisons(ctx, systems: dict, inputs, tol: float) -> dict:
    out = {}
    for s in ctx.seeds:
        pairs = {n: (f"{a}_s{s}", f"{b}_s{s}") for n, (a, b) in PAIRS.items()} | {n: (f"aligned_dense_s{s}", r.format(k=s)) for n, r in REFS.items()}
        for name, (ka, kb) in pairs.items():
            for row in (("argmax", "b3", "fixed") if name in PAIRS else ("b3",)):
                if row != "b3" and row not in systems[kb]:
                    continue
                for h in HALVES:
                    A, B = systems[ka][row][h], rr.pick(systems, kb, row)[h]
                    c = {"bootstrap": paired_bootstrap(A["tables"], B["tables"]), "native_changes": native_changes(A["tables"]["native"], B["tables"]["native"])}
                    if B.get("preds") is not None:
                        c["segment_changes"] = segment_changes(inputs.halves[h], A["preds"], B["preds"]); c["boundary_changes"] = boundary_changes(inputs.halves[h], A["preds"], B["preds"], tol)
                    out[f"s{s}|{row}|{name}|{h}"] = c
    return out


def verdicts(ctx, mixed_comp: dict, comp: dict) -> dict:
    seeds = ctx.seeds
    out = {"mixed": {name: {n: {h: pt.verdict([mixed_comp[f"s{s}|{name}|{n}|{h}"]["session"] for s in seeds], [mixed_comp[f"s{s}|{name}|{n}|{h}"]["stream"] for s in seeds])
                               for h in HALVES} for n in SUBSETS} for name in MIXED_PAIRS}, "system": {}}
    for row in ("argmax", "b3", "fixed"):
        names = list(PAIRS) + (list(REFS) if row == "b3" else [])
        out["system"][row] = {n: {m: {h: rt.consistency([comp[f"s{s}|{row}|{n}|{h}"]["bootstrap"][m] for s in seeds]) for h in HALVES}
                                  for m in ("multi segment F1@50", "native macro-F1", "native cuts per clip", "multi boundary recall", "multi boundary F1", "same-class cut rate")}
                              for n in names if all(f"s{s}|{row}|{n}|{h}" in comp for s in seeds for h in HALVES)}
    return out


def strip(systems: dict) -> dict:
    out = {}
    for k, s in systems.items():
        out[k] = {r: {h: x["row"] for h, x in s[r].items()} for r in ("argmax", "b3", "fixed", "row") if r in s}
        if "b3_choice" in s:
            out[k]["b3_choice"] = s["b3_choice"]
    return out


def write(res: dict, out_dir: Path, ids: set) -> None:
    flat = json.loads(json.dumps(res, default=str)); assert_no_ids(flat, ids, "summary"); tables = vt.tables_md(flat)
    if any(s in tables for s in ids):
        raise SystemExit("an id appears in the report tables")
    write_atomic(out_dir / "summary.json", json.dumps(flat, indent=1)); write_atomic(out_dir / "report_tables.md", tables)


def plot_curves(runs: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.2)); col = {"center_only": "#8172b2", "aligned_dense": "#4c72b0", "shuffled_dense": "#dd8452"}
    for key, r in runs.items():
        ep = r["manifest"]["epochs"]; x = [e["updates"] for e in ep]; kw = {"color": col[r["arm"]], "ls": ("-", "--", ":")[r["seed"]], "lw": 1.2, "label": key}
        ax[0].plot(x, [e["loss_center"] for e in ep], **kw); ax[1].plot(x, [e["loss_time"] for e in ep], **kw); ax[2].plot(x, [e["train_accuracy"] for e in ep], **kw)
    for a, t in zip(ax, ("CLS cross-entropy (mean per epoch)", "time-token cross-entropy against each arm's own labels", "training-batch CLS accuracy (fit)")):
        a.set_title(t, fontsize=10); a.set_xlabel("updates"); a.grid(alpha=0.3)
    ax[0].legend(fontsize=7, ncol=3); fig.suptitle("Video dense supervision round: final weights kept, nothing chosen from these curves", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def token_schematic(path: Path) -> None:
    """The round's token supervision without any frame: 16 input slots, the 8 anchors, the CLS output and the shared head."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(13, 4.4)); ax.set_xlim(0, 17); ax.set_ylim(0, 8); ax.axis("off")
    for i in range(16):
        anchor = i in vds.ANCHORS
        ax.add_patch(plt.Rectangle((0.5 + i, 6.2), 0.85, 0.85, color="#4c72b0" if anchor else "#cfd8e3"))
        ax.text(0.92 + i, 6.62, str(i), ha="center", va="center", fontsize=8, color="white" if anchor else "black")
    ax.text(0.5, 7.4, "16 input frames (slots) of one window; blue: the stem-centre anchor slots 2j (j = 0..7)", fontsize=9)
    for j, s in enumerate(vds.ANCHORS):
        ax.annotate("", xy=(1.4 + 2 * j, 4.6), xytext=(0.92 + s, 6.1), arrowprops={"arrowstyle": "->", "color": "#4c72b0", "lw": 0.8})
        ax.add_patch(plt.Rectangle((0.9 + 2 * j, 3.9), 1.0, 0.7, color="#dd8452", alpha=0.6)); ax.text(1.4 + 2 * j, 4.25, f"t{j}", ha="center", va="center", fontsize=8)
    ax.text(0.5, 3.3, "final normalised tokens: the CLS token plus 8 time positions x 7 x 7; each time position's 49 tokens are averaged (after video interaction)", fontsize=9)
    ax.add_patch(plt.Rectangle((0.5, 1.2), 7.5, 1.3, color="#8172b2", alpha=0.25)); ax.text(0.7, 1.85, "CLS token -> dropout -> 512-d projection -> five-class head\nthe system output (with B3), in every arm", fontsize=9, va="center")
    ax.add_patch(plt.Rectangle((8.5, 1.2), 8.0, 1.3, color="#dd8452", alpha=0.25))
    ax.text(8.7, 1.85, "8 time vectors -> the same dropout, projection and head -> 8 time logits\nloss weight 0 (CLS continuation) or 0.5 (aligned or shuffled labels)", fontsize=9, va="center")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ----------------------------------------------------------------------------------------------- main
def run_evaluate(ctx, workers: int) -> None:
    t0, started, ok = time.time(), tv.now(), False
    try:
        evaluate(ctx, workers, t0); ok = True
    finally:
        tv.ledger_append(ctx.ledger, "evaluate", started, (time.time() - t0) / 3600, ok)


def evaluate(ctx, workers: int, t0: float) -> None:
    device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction()); tv.determinism()
    cfg, paths, vd = ctx.cfg, ctx.paths, ctx.vd; out_dir = ctx.root / vd["outputs"]["results_dir"]; out_dir.mkdir(parents=True, exist_ok=True)
    runs = verify_runs(ctx); print("nine runs verified and frozen", flush=True)
    queue = json.loads((ctx.out / "queue_status.json").read_text()); train_ids = set(tv.allowlist(ctx.rv, ctx.root)); ids = train_ids | {session_of(v) for v in train_ids}
    res = {"runs": {k: {x: r["manifest"][x] for x in ("identity", "arm", "seed", "lambda", "updates", "final_sha256", "source_weights_sha256", "aux_sha256", "interrupted", "parameters")}
                    | {"frozen": r["frozen"], "epochs": r["manifest"]["epochs"]} for k, r in runs.items()}, "decision": "pending"}
    res["reference_pins"] = check_reference_pins(ctx); write(res, out_dir, ids); plot_curves(runs, out_dir / "training_curves.png"); token_schematic(out_dir / "token_supervision.png")
    pin = ctx.rv["evaluation"]["reference_inputs"]["e1_feature_manifest"]; pd.pinned(ctx.root, pin)
    inputs, seqs = rcs.validation(cfg, {"inputs": {"protocol": ctx.rv["source"]["protocol"], "feature_manifest": pin}}, paths)   # development data from here on
    dev_rec = {r[0] for h in HALVES for q in seqs[h] for r in q["refs"]}; ids |= dev_rec | {session_of(v) for v in dev_rec} | {q["id"] for h in HALVES for q in seqs[h]}
    refs = sorted({r for h in HALVES for q in seqs[h] for r in q["refs"]}); at = {r: i for i, r in enumerate(refs)}
    fp = tv.FramePaths(tv.frame_dirs(paths, ctx.rv, {r[0] for r in refs}))
    t1 = time.time(); frames = rr.decode_frames(refs, fp, workers); t_dec = time.time() - t1; print(f"{len(refs)} frames decoded ({t_dec:.0f} s)", flush=True)
    W = {h: rr.half_windows(seqs[h], at, 5) for h in HALVES}; meta = {h: rtp.window_meta(W[h], seqs[h]) for h in HALVES}
    systems, logits = saved_reference(ctx, cfg, W, seqs, inputs, "ordered_outputs", "raw_video_summary", "ordered_s{k}", "ordered_s{k}")
    cc, cc_logits = saved_reference(ctx, cfg, W, seqs, inputs, "continued_cls_outputs", "readout_summary", "cls_s{k}", "continued_cls_s{k}")
    systems |= cc; logits |= cc_logits; logits = {h: {k: v[h] for k, v in logits.items()} for h in HALVES}
    for s in ctx.seeds:
        systems[f"ordered_s{s}"]["fixed"] = {h: systems[f"ordered_s{s}"]["b3"][h] for h in HALVES}
    print("frozen Ordered and historical Continued CLS reproduced", flush=True)
    fx = vd["evaluation"]["fixed_ordered_setting"]; cost, time_logits = {}, {h: {} for h in HALVES}
    for key, r in runs.items():
        model, probs, secs = None, {}, 0.0
        for h in HALVES:
            path = ctx.out / "evaluation" / h / f"{key}.npz"; got = cached_outputs(path, r["manifest"]["identity"], W[h], seqs[h])
            if got is None:
                if model is None:
                    state = torch.load(ctx.root / r["frozen"] / "final.pt", map_location="cpu", weights_only=True); model = tv.make_model(state, device).eval()
                lg, tl, sec = model_outputs(model, W[h], frames, device); save_outputs(path, r["manifest"]["identity"], W[h], lg, tl, rr.per_sample(W[h], lg, seqs[h]), seqs[h], sec)
            else:
                lg, tl, sec = got
            secs += sec; probs[h] = rr.per_sample(W[h], lg, seqs[h]); logits[h][key] = lg; time_logits[h][key] = tl
        systems[key] = rr.system_rows(cfg, probs, inputs) | {"fixed": rtp.fixed_rows(cfg, probs, inputs, fx["thresholds"][r["seed"]], fx)}
        n_win = sum(len(W[h]["seq"]) for h in HALVES); cost[key] = {"windows": n_win, "ms_per_window": 1000 * secs / n_win}
        print(f"{key}: B3 {systems[key]['b3_choice']['status']} | sel {systems[key]['b3']['selection']['row']['segment_f1_50_multi']:.4f} "
              f"cal {systems[key]['b3']['calibration']['row']['segment_f1_50_multi']:.4f}", flush=True)
        del model; torch.cuda.empty_cache()
    mixed = {h: mixed_analysis(ctx, logits[h], W[h], meta[h]) for h in HALVES}
    anchors = {h: anchor_analysis(time_logits[h], W[h], meta[h], refs, session_of) for h in HALVES}
    res.update({"systems": strip(systems), "mixed": {"accuracy": {k: {h: mixed[h]["accuracy"][k] for h in HALVES} for k in mixed["selection"]["accuracy"]},
                                                      "support": {h: mixed[h]["support"] for h in HALVES},
                                                      "comparisons": {f"{k}|{h}": v for h in HALVES for k, v in mixed[h]["comparisons"].items()}},
                "anchors": {k: {h: anchors[h][k] for h in HALVES} for k in anchors["selection"]},
                "cost": {"inference": cost, "frame_decode_seconds": t_dec}})
    write(res, out_dir, ids); print("round systems, mixed windows and anchors evaluated", flush=True)
    refs_cs, e1_preds, g1 = ra.reference_systems(cfg, {"inputs": {"protocol": ctx.rv["source"]["protocol"]}}, paths, inputs, seqs)
    for name, by in refs_cs.items():
        systems[name] = {"row": {h: {"row": by[h]["metrics"], "tables": by[h]["tables"], "preds": e1_preds[h] if name == "E1 + B3" else None} for h in HALVES}}
    old = SimpleNamespace(rv=ctx.rv, root=ctx.root, cfg=cfg)
    for name, by in rr.reference_fact_asrf(old, paths, inputs, device).items():
        if any(by[h].get("differing_fields") for h in HALVES):
            raise SystemExit(f"{name}: its stored rows are not reproduced")
        systems[name] = {"row": {h: {"row": by[h]["row"], "tables": by[h]["tables"], "preds": by[h]["preds"]} for h in HALVES}}
    systems |= a01_reference(ctx, seqs, inputs, device); tv.determinism(); del frames
    res["references_check"] = {"ordered": "stored V01 rows reproduced", "continued_cls": "stored V02 rows reproduced", "E0/E1/B7 + B3 and historical E1 + ASRF": "check_rows passed",
                               "paired ASRF and Original FACT": "no differing field", "A01 Linear + B3, MS-TCN, ASRF": "every stored field reproduced"}
    print("references reproduced", flush=True)
    tol = float(cfg["selection"]["boundary_tolerance_s"]) * float(cfg["protocol"]["hz"])
    comp = system_comparisons(ctx, systems, inputs, tol); ver = verdicts(ctx, res["mixed"]["comparisons"], comp)
    goals = {r: rt.goal([{h: systems[f"{r}_s{s}"]["b3"][h]["row"] for h in HALVES} for s in ctx.seeds],
                        [systems[f"{r}_s{s}"]["b3_choice"]["status"] == "feasible" for s in ctx.seeds], vd["goal"]) for r in (*ARMS, "ordered", "continued_cls")}
    sysv = {c: {h: ver["system"]["b3"][c]["multi segment F1@50"][h] for h in HALVES} for c in vt.SYS}
    mixv = {c: ver["mixed"][c]["all mixed"] for c in vt.MIX}
    dec = vt.decide(sysv, mixv, not goals["aligned_dense"]["constraint_failures"], goals["aligned_dense"]["met"], vd["decision_rules"])
    res.update({"systems": strip(systems), "references_g1": g1, "system_comparisons": comp, "verdicts": ver, "goal": goals, "decision": dec,
                "units": rr.units(inputs, systems, tol),
                "cost": {**res["cost"], "training_hours": {k: v.get("hours") for k, v in queue["jobs"].items()}, "ledger_hours_before_evaluation": tv.ledger_hours(ctx.ledger),
                         "evaluation_hours": (time.time() - t0) / 3600}})
    write(res, out_dir, ids); print(json.dumps(dec["applied"]), flush=True); print(f"wrote {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--evaluate", action="store_true"); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); ctx = tv.Ctx(cfg, a.config)
    if not a.evaluate:
        raise SystemExit("nothing to do: pass --evaluate after the nine runs are finished and frozen")
    run_evaluate(ctx, a.workers)


if __name__ == "__main__":
    main()
