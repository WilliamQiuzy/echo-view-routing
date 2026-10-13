#!/usr/bin/env python
"""Evaluate the video feature temporal round once (docs/superpowers/plans/2026-10-12-video-feature-temporal-training.md,
Task 3; configs/experiments/video_feature_temporal_v1.yaml). Runs only after the nine trainings are finished and frozen.

1. Runs: the nine identities, 18,750 updates and frozen final weights are verified. Only then are development labels read.
2. Development features: every native clip, pair stream and multi stream of both halves as its own whole sequence
   (streams assembled first), through each seed's frozen Ordered encoder; float32 Seq objects built directly. The source
   classifier on the window features must reproduce the source round's saved window logits (<= 1e-4, argmax equal).
3. Systems: Linear + B3 (B3 chosen once on calibration from the 40-setting grid), MS-TCN (last-stage argmax), ASRF
   (refinement_with_boundary at 0.5; the frame branch as a diagnostic); frozen Ordered + B3 from the saved per-sample
   probabilities with its stored setting (its stored rows reproduced exactly); E0/E1/B7 + B3, historical E1 + ASRF,
   paired ASRF, Original FACT and G1 + ASRF with their own inputs and settings.
4. Paired comparisons and verdicts; reference units with the frozen diagnostic list; the ASRF decomposition; the overlap
   with Linear + B3; short segments and X-Y-X middles; native errors with sources; the goal and the fixed readings.
Results go to the results directory through the id guards; case-level arrays stay on the server.

  python scripts/report_video_feature_temporal.py --config configs/experiments/video_feature_temporal_v1.yaml --evaluate
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.dense_target_inference_tables as dtt  # noqa: E402
import scripts.prepare_video_feature_temporal as pv  # noqa: E402
import scripts.queue_video_feature_temporal as qv  # noqa: E402
import scripts.raw_video_context_tables as rt  # noqa: E402
import scripts.report_center_supervision as rcs  # noqa: E402
import scripts.report_fact_asrf as ra  # noqa: E402
import scripts.report_raw_video_context as rr  # noqa: E402
import scripts.train_video_feature_temporal as tv  # noqa: E402
import scripts.video_feature_temporal_tables as vtt  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.ev9v_complementarity import decode  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402
from scripts.report_fold_output_correction import boundary_changes, native_changes, segment_changes, write_atomic  # noqa: E402
from scripts.report_model_development import paired_bootstrap  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of, stream_units, unit_outcomes  # noqa: E402
from echo_routing.temporal.dev_data import Seq  # noqa: E402

NS = "video_feature_temporal_v1"
HALVES = vtt.HALVES
DIAG = {"evaluation": {"b3_infeasible_diagnostic": {"half_window": 9, "quantile": 0.95}}}
METRICS = ("multi segment F1@50", "native macro-F1", "native cuts per clip", "multi boundary recall", "multi boundary F1", "same-class cut rate")
COMPARE = {"ASRF - (Linear + B3)": ("ASRF", "Linear + B3"), "ASRF - MS-TCN": ("ASRF", "MS-TCN"), "MS-TCN - (Linear + B3)": ("MS-TCN", "Linear + B3"),
           "ASRF - (frozen Ordered + B3)": ("ASRF", "frozen Ordered + B3"), "MS-TCN - (frozen Ordered + B3)": ("MS-TCN", "frozen Ordered + B3"),
           "(Linear + B3) - (frozen Ordered + B3)": ("Linear + B3", "frozen Ordered + B3"), "ASRF - paired ASRF": ("ASRF", "paired ASRF"),
           "ASRF - (E1 + B3)": ("ASRF", "E1 + B3*"), "ASRF - Original FACT": ("ASRF", "Original FACT"), "ASRF - historical E1 + ASRF": ("ASRF", "E1 + ASRF")}
NEW = ("Linear + B3", "MS-TCN", "ASRF")


# ----------------------------------------------------------------------------------------------- runs
def verify_runs(ctx: tv.Ctx) -> dict:
    idents = qv.identities(ctx); out = {}
    for key, ident in idents.items():
        path = ctx.out / "train" / key / "run_manifest.json"
        if not tv.finished(path, ident):
            raise SystemExit(f"{key}: not a finished run of this identity")
        man = json.loads(path.read_text()); frozen = ctx.root / "models_frozen" / NS / key / "final.pt"
        if not frozen.is_file() or sha256_file(frozen) != man["final_sha256"]:
            raise SystemExit(f"{key}: the frozen final weights are missing or differ")
        out[key] = {"identity": ident, "updates": man["updates"], "final_sha256": man["final_sha256"], "interrupted": man["interrupted"],
                    "parameters": man["parameters"], "frozen": str(frozen.parent.relative_to(ctx.root)), "records": man["records"]}
    return out


def load_head(ctx: tv.Ctx, head: str, seed: int, device):
    plan, _ = ctx.plan(seed); h = tv.make_head(ctx, head, plan, device)
    state = torch.load(ctx.root / "models_frozen" / NS / tv.run_key(head, seed) / "final.pt", map_location="cpu", weights_only=True)
    model = tv.new_model(h, state, device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return h, model


# ----------------------------------------------------------------------------------------------- development features
def dev_features(ctx: tv.Ctx, seqs: dict, device, workers: int) -> tuple[dict, dict]:
    """Whole development sequences through the three source models (cached under the feature identity), the per-seed
    float32 features of every sequence and the parity against the saved window logits."""
    lst = [{"id": q["id"], "refs": q["refs"]} for h in HALVES for q in seqs[h]]
    idents = {s: pv.feature_identity(ctx, s, lst) for s in ctx.seeds}; paths = {s: ctx.out / "features" / "development" / f"ordered_s{s}.npz" for s in ctx.seeds}
    if not all(p.is_file() and str(np.load(p)["identity"]) == idents[s] for s, p in paths.items()):
        models = {s: pv.build_source(ctx, s, device) for s in ctx.seeds}; before = {s: pv.model_digest(m) for s, m in models.items()}
        ex = pv.extract(ctx, SimpleNamespace(seqs=lst, check_ids=lambda: []), models, device, workers)
        for s in ctx.seeds:
            pv.save_features(ctx, "development", s, idents[s], ex["meta"], ex["features"][s])
        if any(pv.model_digest(m) != before[s] for s, m in models.items()):
            raise SystemExit("a source model's state changed during development extraction")
        del models; torch.cuda.empty_cache()
    feats, parity = {}, {}
    for s in ctx.seeds:
        z = np.load(paths[s]); off, woff = z["sample_offsets"], z["window_offsets"]; ids = z["ids"].tolist(); samples, window = z["samples"], z["window"]
        if samples.dtype != np.float32 or ids != [q["id"] for q in lst]:
            raise SystemExit(f"seed {s}: the development features are not float32 or list other sequences")
        st = torch.load(pv.pinned(ctx.root, ctx.vt["source"]["models"][s]["final"]), map_location="cpu", weights_only=True)
        W, b = st["classifier.weight"].double().numpy(), st["classifier.bias"].double().numpy(); feats[s] = {}; i0 = 0
        for h in HALVES:
            n = len(seqs[h]); rows = slice(int(woff[i0]), int(woff[i0 + n])); mine = window[rows].astype(np.float64) @ W.T + b
            with np.load(pv.pinned(ctx.root, ctx.vt["source"]["outputs"][h][s])) as zz:
                saved = zz["logits"].astype(np.float64)
            parity[f"s{s}|{h}"] = {"windows": int(len(saved)), "max_abs": float(np.abs(mine - saved).max()) if len(saved) == len(mine) else None,
                                   "argmax_equal": bool(len(saved) == len(mine) and np.array_equal(mine.argmax(1), saved.argmax(1)))}
            feats[s][h] = {q["id"]: samples[off[i0 + j]:off[i0 + j + 1]] for j, q in enumerate(seqs[h])}; i0 += n
    bad = {k: v for k, v in parity.items() if v["max_abs"] is None or v["max_abs"] > 1e-4 or not v["argmax_equal"]}
    if bad:
        raise SystemExit(f"development parity failed: {bad}")
    return feats, parity


def float_seqs(inputs, seqs: dict, feats: dict, seed: int, h: str) -> list[Seq]:
    """float32 Seq objects of one half (never EvalSet.sequences(), whose stream_seq casts to fp16)."""
    joins = {st.recipe.recipe_id: np.asarray(st.joins, np.int64) for st in (*inputs.halves[h].pairs, *inputs.halves[h].multi)}
    return [Seq(q["id"], q["kind"] == "native", feats[seed][h][q["id"]], np.asarray(q["labels"], np.int64), joins.get(q["id"], np.zeros(0, np.int64)))
            for q in seqs[h]]


# ----------------------------------------------------------------------------------------------- systems
def evaluate(cfg: dict, preds: dict, ev) -> dict:
    m, t = rcs.evaluate(cfg, preds, ev)
    return {"row": dtt.system_row(m), "tables": t, "preds": preds}


def new_systems(ctx: tv.Ctx, cfg: dict, inputs, seqs: dict, feats: dict, device) -> tuple[dict, dict, dict]:
    """Linear + B3 (and its argmax), MS-TCN, ASRF and its frame branch per seed and half; the raw outputs kept for diagnostics."""
    systems, raw, b3 = {}, {}, {}
    for s in ctx.seeds:
        fs = {h: float_seqs(inputs, seqs, feats, s, h) for h in HALVES}
        lh, lm = load_head(ctx, "linear", s, device); lprob = {h: {k: v["prob"] for k, v in lh.decode(lm, fs[h], device).items()} for h in HALVES}
        ch = rcs.b3_choose(cfg, DIAG, lprob["calibration"], inputs.halves["calibration"]); b3[s] = {k: v for k, v in ch.items() if k != "grid"}
        for h in HALVES:
            ev = inputs.halves[h]
            systems.setdefault(f"Linear + B3 s{s}", {})[h] = evaluate(cfg, decode(ch["params"], lprob[h]), ev)
            systems.setdefault(f"Linear argmax s{s}", {})[h] = evaluate(cfg, {k: v.argmax(1) for k, v in lprob[h].items()}, ev)
        mh, mm = load_head(ctx, "mstcn", s, device); ah, am = load_head(ctx, "asrf", s, device)
        for h in HALVES:
            ev = inputs.halves[h]; mo = mh.decode(mm, fs[h], device); ao = ah.decode(am, fs[h], device)
            systems.setdefault(f"MS-TCN s{s}", {})[h] = evaluate(cfg, {k: np.asarray(v["labels"], int) for k, v in mo.items()}, ev)
            systems.setdefault(f"ASRF s{s}", {})[h] = evaluate(cfg, {k: np.asarray(v["labels"], int) for k, v in ao.items()}, ev)
            systems.setdefault(f"ASRF frame branch s{s}", {})[h] = evaluate(cfg, {k: np.asarray(v["frame_labels"], int) for k, v in ao.items()}, ev)
            raw[(s, h)] = {"asrf": ao, "linear_prob": lprob[h]}
        del lm, mm, am; torch.cuda.empty_cache()
    return systems, raw, b3


def ordered_reference(ctx: tv.Ctx, cfg: dict, inputs, seqs: dict) -> tuple[dict, dict]:
    """Frozen Ordered + B3 from the saved per-sample probabilities with its stored setting; stored rows reproduced exactly."""
    stored = json.loads(pv.pinned(ctx.root, ctx.vt["source"]["report"]["summary"]).read_text())["systems"]; systems, probs = {}, {}
    for s in ctx.seeds:
        params = stored[f"ordered_s{s}"]["b3_choice"]["params"]
        for h in HALVES:
            with np.load(pv.pinned(ctx.root, ctx.vt["source"]["outputs"][h][s])) as z:
                off, sp, ids = z["sample_offsets"], z["sample_probs"], z["sequence_ids"].tolist()
            if ids != [q["id"] for q in seqs[h]]:
                raise SystemExit(f"ordered_s{s} {h}: the saved outputs list other sequences")
            P = {sid: sp[off[i]:off[i + 1]] for i, sid in enumerate(ids)}; probs[(s, h)] = P
            for name, row, preds in ((f"frozen Ordered + B3 s{s}", "b3", decode(params, P)), (f"frozen Ordered argmax s{s}", "argmax", {k: v.argmax(1) for k, v in P.items()})):
                e = evaluate(cfg, preds, inputs.halves[h]); want = stored[f"ordered_s{s}"][row][h]
                if any(e["row"].get(k) != v for k, v in want.items()):
                    raise SystemExit(f"{name} {h}: the stored row is not reproduced")
                systems.setdefault(name, {})[h] = e
    return systems, probs


def references(ctx: tv.Ctx, cfg: dict, inputs, seqs: dict, device) -> tuple[dict, dict]:
    refs_cs, e1_preds, g1 = ra.reference_systems(cfg, {"inputs": {"protocol": ctx.vt["source"]["protocol"]}}, ctx.paths, inputs, seqs); out = {}
    for name, by in refs_cs.items():
        out[name] = {h: {"row": by[h]["metrics"], "tables": by[h]["tables"], "preds": e1_preds[h] if name == "E1 + B3" else None} for h in HALVES}
    for name, by in rr.reference_fact_asrf(ctx, ctx.paths, inputs, device).items():
        if any(by[h]["differing_fields"] for h in HALVES):
            raise SystemExit(f"{name}: its stored rows are not reproduced ({ {h: by[h]['differing_fields'] for h in HALVES} })")
        out[name] = {h: {"row": by[h]["row"], "tables": by[h]["tables"], "preds": by[h]["preds"]} for h in HALVES}
    torch.use_deterministic_algorithms(True, warn_only=True)
    return out, g1


# ----------------------------------------------------------------------------------------------- comparisons
def pick(systems: dict, base: str, seed: int) -> str:
    if base == "E1 + B3*":
        return "E1 + B3"
    for cand in (f"{base} s{seed}", base):
        if cand in systems:
            return cand
    raise KeyError(base)


def comparisons(ctx: tv.Ctx, systems: dict, inputs, tol: float) -> tuple[dict, dict]:
    comp = {}
    for s in ctx.seeds:
        for name, (a, b) in COMPARE.items():
            ka, kb = pick(systems, a, s), pick(systems, b, s)
            for h in HALVES:
                A, B, ev = systems[ka][h], systems[kb][h], inputs.halves[h]
                c = {"bootstrap": paired_bootstrap(A["tables"], B["tables"]), "native_changes": native_changes(A["tables"]["native"], B["tables"]["native"])}
                if B["preds"] is not None:
                    c["segment_changes"] = segment_changes(ev, A["preds"], B["preds"]); c["boundary_changes"] = boundary_changes(ev, A["preds"], B["preds"], tol)
                comp[f"s{s}|{name}|{h}"] = c
        print(f"seed {s}: comparisons done", flush=True)
    ver = {n: {h: {m: rt.consistency([comp[f"s{s}|{n}|{h}"]["bootstrap"][m] for s in ctx.seeds]) for m in METRICS} for h in HALVES} for n in COMPARE}
    return comp, ver


# ----------------------------------------------------------------------------------------------- units and diagnostics
def unit_rows(systems: dict, name: str, ev) -> dict:
    out = {}
    for st in ev.multi:
        sid = st.recipe.recipe_id; us = stream_units(st.recipe); rows, spur = unit_outcomes(np.asarray(systems[name][ev.name]["preds"][sid], int), us, 0.5, 5, 10)
        out[sid] = (us, rows, spur)
    return out


def matched_keys(ur: dict) -> set:
    return {(sid, r["index"]) for sid, (_, rows, _) in ur.items() for r in rows if r["matched"]}


def cause_table(ur: dict) -> dict:
    out = {c: {"units": 0, "recs": set()} for c in ("label", "missed_cut", "extra_cut", "boundary_shift")}; n = spur = 0
    for _, (us, rows, sp) in ur.items():
        n += len(rows); spur += len(sp)
        for r in rows:
            if not r["matched"]:
                out[r["cause"]]["units"] += 1; out[r["cause"]]["recs"] |= {x[0] for x in us[r["index"]].sources}
    return {"units": n, "unmatched_predicted_segments": spur, **{c: {"units": v["units"], **vtt.count(v["recs"])} for c, v in out.items()}}


def diagnostics(ctx: tv.Ctx, cfg: dict, systems: dict, raw: dict, oprobs: dict, inputs, hz: float) -> dict:
    """Unit causes, the frozen list, the ASRF decomposition, the Linear overlap, trade-off inputs and native errors."""
    causes, dlist, dec, ovl, trade_in, nat_err = {}, {}, {}, {}, {}, {}
    fixed_all = {n: set() for n in NEW}
    for s in ctx.seeds:
        for h in HALVES:
            ev = inputs.halves[h]; ur = {n: unit_rows(systems, pick(systems, n, s), ev) for n in (*NEW, "frozen Ordered + B3")}
            for n, u in ur.items():
                causes[f"{n} s{s}|{h}"] = cause_table(u)
            ordered_ok = matched_keys(ur["frozen Ordered + B3"]); P = oprobs[(s, h)]; listed = set(); src = {}
            for sid, (us, rows, _) in ur["frozen Ordered + B3"].items():
                for r in rows:
                    u = us[r["index"]]; src[(sid, r["index"])] = [x[0] for x in u.sources]
                    if not r["matched"] and vtt.majority(P[sid][u.start:u.end].argmax(1)) == u.label:
                        listed.add((sid, r["index"]))
            dlist[f"s{s}|{h}"] = {n: vtt.list_outcomes(listed, ordered_ok, matched_keys(ur[n]), src.__getitem__) for n in NEW}
            # ASRF decomposition on units and native clips
            ao = raw[(s, h)]["asrf"]; urows = []
            for sid, (us, rows, _) in ur["ASRF"].items():
                bp = ao[sid]["boundary_prob"]; peaks = vtt.boundary_peaks(bp, 0.5)
                if not np.array_equal(peaks, official_peaks(ctx, bp)):
                    raise SystemExit(f"{sid}: the peak rule differs from the official argrelmax")
                refined = np.asarray(ao[sid]["labels"], int); ch = np.flatnonzero(np.diff(refined) != 0) + 1; src_lab = P[sid].argmax(1)
                urows += [vtt.asrf_unit(r, us[r["index"]], np.asarray(ao[sid]["frame_labels"], int), src_lab, peaks, ch, 5) for r in rows]
            clips = [vtt.asrf_native({"video_id": n.seq_id, "label": int(n.labels[0]), "decode": vtt.majority(ao[n.seq_id]["labels"]),
                                      "frame": vtt.majority(ao[n.seq_id]["frame_labels"]), "source": vtt.majority(P[n.seq_id].argmax(1))}) for n in ev.natives]
            dec[f"s{s}|{h}"] = {"units": vtt.unit_summary(urows), "natives": vtt.native_summary(clips)}
            # cases fixed against Ordered + B3, and whether Linear + B3 of the seed fixes them too
            nat_ok = {n: {x.seq_id for x in ev.natives if vtt.majority(systems[pick(systems, n, s)][h]["preds"][x.seq_id]) == int(x.labels[0])}
                      for n in (*NEW, "frozen Ordered + B3")}
            fx = {}
            for n in NEW:   # units and native clips this system gets right and frozen Ordered + B3 of the seed gets wrong
                fx[n] = {("u", s, k) for k in matched_keys(ur[n]) - ordered_ok} | {("n", s, v) for v in nat_ok[n] - nat_ok["frozen Ordered + B3"]}
                fixed_all[n] |= fx[n]; new_nat = nat_ok["frozen Ordered + B3"] - nat_ok[n]
                t = trade_in.setdefault(f"{n}|{h}", {"fixed": 0, "new_native_misses": 0}); t["fixed"] += len(fx[n]); t["new_native_misses"] += len(new_nat)
            ovl[f"s{s}|{h}"] = {n: vtt.overlap(fx[n], fx["Linear + B3"]) for n in ("ASRF", "MS-TCN")}
            for n in (*NEW, "frozen Ordered + B3"):
                wrong = [x for x in ev.natives if vtt.majority(systems[pick(systems, n, s)][h]["preds"][x.seq_id]) != int(x.labels[0])]
                for x in wrong:
                    k = f"{vtt.CLASSES[int(x.labels[0])]}->{vtt.CLASSES[vtt.majority(systems[pick(systems, n, s)][h]['preds'][x.seq_id])]}"
                    nat_err.setdefault(f"{n}|{h}", {}).setdefault(k, set()).add(x.seq_id)
    pooled = {n: vtt.overlap(fixed_all[n], fixed_all["Linear + B3"]) for n in ("ASRF", "MS-TCN")}
    nat = {k: {p: {"clips_over_seeds": len(v), **vtt.count(v)} for p, v in by.items()} for k, by in nat_err.items()}
    return {"causes": causes, "diagnostic_list": dlist, "asrf_decomposition": dec, "linear_overlap": {"per_seed_half": ovl, "pooled": pooled},
            "trade_inputs": trade_in, "native_errors": nat}


def official_peaks(ctx: tv.Ctx, bp: np.ndarray) -> np.ndarray:
    repo = str(ctx.root / ctx.vt["heads"]["asrf"]["repo"])
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from libs.metric import argrelmax  # noqa: E402  (official)
    return np.asarray(argrelmax(np.array(bp, dtype=np.float64, copy=True), threshold=0.5), int)


def trade_flags(ver: dict, trade_in: dict) -> dict:
    out = {}
    for n, cname in (("ASRF", "ASRF - (frozen Ordered + B3)"), ("MS-TCN", "MS-TCN - (frozen Ordered + B3)")):
        for h in HALVES:
            v = ver[cname][h]; t = trade_in[f"{n}|{h}"]
            cost = t["new_native_misses"] > 0 or v["native cuts per clip"] == vtt.POS or v["same-class cut rate"] == vtt.POS
            out[f"{n}|{h}"] = bool(t["fixed"] > 0 and cost and v["multi segment F1@50"] != vtt.POS)
    return out


def goals(ctx: tv.Ctx, systems: dict, b3: dict) -> dict:
    g = ctx.vt["goal"]; out = {}
    for n in (*NEW, "frozen Ordered + B3"):
        rows = [{h: systems[f"{n} s{s}"][h]["row"] for h in HALVES} for s in ctx.seeds]
        feasible = [b3[s]["status"] == "feasible" if n == "Linear + B3" else True for s in ctx.seeds]
        out[n] = rt.goal(rows, feasible, g)
    return out


# ----------------------------------------------------------------------------------------------- figures and output
def figures(ctx: tv.Ctx, runs: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2)); col = {"linear": "#8172b2", "mstcn": "#dd8452", "asrf": "#4c72b0"}
    for key, r in runs.items():
        h, s = key.rsplit("_s", 1); recs = r["records"]; x = [e["updates"] for e in recs]
        kw = {"color": col[h], "ls": ("-", "--", ":")[int(s)], "lw": 1.1, "label": key}
        ax[0].plot(x, [e["loss"]["total"] for e in recs], **kw); ax[1].plot(x, [e["check_accuracy_mean"] for e in recs], **kw)
    for a, t in zip(ax, ("training loss (mean per epoch; heads' losses differ)", "fit on the 16 check sequences (training data)")):
        a.set_title(t, fontsize=10); a.set_xlabel("updates"); a.grid(alpha=0.3)
    ax[0].set_yscale("log"); ax[0].legend(fontsize=7, ncol=3); fig.suptitle("Video feature temporal round: final weights kept, nothing chosen from these curves", fontsize=11)
    fig.tight_layout(); fig.savefig(ctx.results / "training_curves.png", dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 3.6)); ax.set_xlim(0, 60); ax.set_ylim(0, 10); ax.axis("off")
    for i in range(0, 30):
        ax.add_patch(plt.Rectangle((1 + i * 1.6, 7.6), 1.3, 1.3, color="#cfd8e3" if (i // 10) % 2 == 0 else "#e8d5c4"))
    for c in (2, 7, 12, 17, 22, 27):
        ax.annotate("", xy=(1.65 + c * 1.6, 7.5), xytext=(1.65 + c * 1.6, 6.3), arrowprops={"arrowstyle": "->", "color": "#4c72b0"})
    ax.add_patch(plt.Rectangle((1 + 4 * 1.6, 7.4), 16 * 1.6 - 0.3, 1.7, fill=False, ec="#c44e52", lw=1.5))
    ax.text(1, 9.4, "a whole stream on the 10 Hz axis (two fragments, assembled first); one 16-sample window around a stride-5 centre", fontsize=9)
    ax.text(1, 5.4, "frozen Ordered encoder: model.embed(16 frames, marker) -> one 512-d window feature per centre (2, 7, 12, ...)", fontsize=9)
    ax.text(1, 3.9, "nearest centre -> (T, 512) float32 for every sample; the same arrays and batches feed three heads per seed:", fontsize=9)
    for i, (t, cc) in enumerate((("Linear(512, 5) + B3", "#8172b2"), ("official MS-TCN", "#dd8452"), ("official ASRF (primary)", "#4c72b0"))):
        ax.add_patch(plt.Rectangle((1 + i * 19, 0.6), 17, 2.2, color=cc, alpha=0.25)); ax.text(2 + i * 19, 1.5, t, fontsize=10)
    fig.tight_layout(); fig.savefig(ctx.results / "feature_flow.png", dpi=130); plt.close(fig)


def strip(systems: dict) -> dict:
    return {name: {h: x["row"] for h, x in by.items()} for name, by in systems.items()}


def write(res: dict, out_dir: Path, ids: set) -> None:
    flat = json.loads(json.dumps(res, default=str)); assert_no_ids(flat, ids, "summary"); md = vtt.tables_md(flat)
    if any(s in md for s in ids):
        raise SystemExit("an id appears in the report tables")
    out_dir.mkdir(parents=True, exist_ok=True); write_atomic(out_dir / "summary.json", json.dumps(flat, indent=1)); write_atomic(out_dir / "report_tables.md", md)


def run_evaluate(ctx: tv.Ctx, workers: int) -> None:
    t0, started, ok = time.time(), pv.now(), False
    try:
        cfg = ctx.cfg; device = pv.setup_gpu(); hz = float(cfg["protocol"]["hz"]); tol = float(cfg["selection"]["boundary_tolerance_s"]) * hz
        grid, want = cfg["decode_grids"]["b3"], ctx.vt["evaluation"]["linear_b3"]["grid"]
        if ([int(x) for x in grid["half_window"]], [float(x) for x in grid["threshold_quantile"]], [int(x) for x in grid["nms_radius"]]) != \
                ([int(x) for x in want["half_window"]], [float(x) for x in want["quantile"]], [int(want["nms_radius"])]):
            raise SystemExit("the inherited decoding grid differs from the configured Linear + B3 grid")
        runs = verify_runs(ctx); print("nine runs verified and frozen", flush=True)
        inputs, seqs = rcs.validation(cfg, {"inputs": {"protocol": ctx.vt["source"]["protocol"], "feature_manifest": ctx.vt["source"]["e1_feature_manifest"]}}, ctx.paths)
        train_ids = set(json.loads(pv.pinned(ctx.root, ctx.vt["source"]["protocol"]).read_text())["train"]["clips"])
        dev = {r[0] for h in HALVES for q in seqs[h] for r in q["refs"]}
        ids = train_ids | dev | {session_of(v) for v in train_ids | dev} | {q["id"] for h in HALVES for q in seqs[h]}
        feats, parity = dev_features(ctx, seqs, device, workers); print("development features ready", flush=True)
        systems, raw, b3 = new_systems(ctx, cfg, inputs, seqs, feats, device)
        osys, oprobs = ordered_reference(ctx, cfg, inputs, seqs); systems.update(osys); print("frozen Ordered + B3 reproduced", flush=True)
        refs, g1 = references(ctx, cfg, inputs, seqs, device); systems.update(refs); print("references reproduced", flush=True)
        comp, ver = comparisons(ctx, systems, inputs, tol)
        diag = diagnostics(ctx, cfg, systems, raw, oprobs, inputs, hz)
        gl = goals(ctx, systems, b3); trade = trade_flags(ver, diag["trade_inputs"])
        dec = vtt.decide(ver, {"ASRF": gl["ASRF"]}, diag["linear_overlap"]["pooled"]["ASRF"]["share"], trade, ctx.vt["decision_rules"])
        mech = {f"{n}|{h}": ra.mechanism(inputs.halves[h], systems[n][h]["preds"], 16) for n in systems for h in HALVES if systems[n][h]["preds"] is not None}
        figures(ctx, runs)
        prep = json.loads((ctx.out / "preparation.json").read_text())
        res = {"runs": {k: {x: v for x, v in r.items() if x != "records"} for k, r in runs.items()}, "parity": parity,
               "counts": {h: {"sequences": len(seqs[h]), "samples": int(sum(len(q["refs"]) for q in seqs[h]))} for h in HALVES},
               "systems": strip(systems), "references_g1": g1, "linear_b3": b3, "comparisons": comp, "verdicts": ver, "goal": gl,
               "diagnostic_list": diag["diagnostic_list"], "asrf_decomposition": diag["asrf_decomposition"], "linear_overlap": diag["linear_overlap"],
               "causes": diag["causes"], "trade": {"inputs": diag["trade_inputs"], "flags": trade}, "native_errors": diag["native_errors"], "mechanism": mech,
               "decision": dec, "inits_equal_paired_asrf": {s: prep["inits"][f"asrf_s{s}"]["equals_paired_asrf_init"] for s in ctx.seeds},
               "cost": {"ledger_hours_before_evaluation": pv.ledger_hours(ctx), "evaluation_seconds": time.time() - t0}}
        write(res, ctx.results, ids); print(json.dumps(dec["applied"]), flush=True); ok = True
    finally:
        pv.ledger(ctx, "evaluate", started, time.time() - t0, ok)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--evaluate", action="store_true"); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(); a.set = []
    if not a.evaluate:
        raise SystemExit("nothing to do: pass --evaluate after the nine runs are finished and frozen")
    run_evaluate(tv.Ctx(load_cfg(a), a.config), a.workers)


if __name__ == "__main__":
    main()
