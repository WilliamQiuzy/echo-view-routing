#!/usr/bin/env python
"""Evaluate, freeze and report the predicted-partition round (docs/superpowers/plans/2026-10-09-fact-predicted-partition.md,
Tasks 3-4; configs/experiments/fact_predicted_partition_v1.yaml).

  --evaluate   only after the six runs: each run's identity recomputed, its 18,750 updates and final weights checked;
               the six final weights frozen (the heads were frozen before training); both development halves decoded
               once by Original FACT, Uniform, Predicted, the paired ASRF and Boundary head + frame classifier, with
               per-sequence probabilities, partitions and predictions saved on the server; the references reproduced
               with their stored settings; the comparisons with the paired bootstrap, the verdicts and the fixed
               decision; the reference-unit tables; the stage tracking on the 16 fixed training sequences and the
               development sequences; the position control on the Predicted models; interval counts, head cuts, the
               case figure, training curves and costs.
Nothing here chooses an epoch, a threshold, a grid or a model. Recording and session ids stay on the server.

  python scripts/report_fact_predicted_partition.py --config configs/experiments/fact_predicted_partition_v1.yaml --evaluate
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.fact_predicted_partition_tables as pt  # noqa: E402
import scripts.report_fact_asrf as ra  # noqa: E402
import scripts.train_fact_asrf as tr  # noqa: E402
import scripts.train_fact_predicted_partition as tp  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402
from scripts.report_fold_output_correction import boundary_changes, consistency, native_changes, segment_changes, summary, write_atomic  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import echo_routing.temporal.predicted_partition as pp  # noqa: E402
from echo_routing.config import gpu_memory_fraction  # noqa: E402
from echo_routing.evaluate.dev_metrics import evaluate_predictions  # noqa: E402
from echo_routing.temporal.dev_run import HALVES, sha256_file  # noqa: E402
from echo_routing.temporal.segments import labels_to_segments  # noqa: E402

NS, SRC = tp.NS, tp.SRC
SEEDED = ("Original FACT", "Uniform", "Predicted", "ASRF", "Boundary head + frame classifier")
FACT_SYSTEMS = ("Original FACT", "Uniform", "Predicted")
COMPARE = {"Predicted - Uniform": ("Predicted", "Uniform"), "Predicted - Original FACT": ("Predicted", "Original FACT"),
           "Uniform - Original FACT": ("Uniform", "Original FACT"), "Predicted - ASRF": ("Predicted", "ASRF"), "Uniform - ASRF": ("Uniform", "ASRF"),
           "Predicted - (E1 + B3)": ("Predicted", "E1 + B3"), "Predicted - (E0 + B3)": ("Predicted", "E0 + B3"), "Predicted - (B7 + B3)": ("Predicted", "B7 + B3"),
           "Predicted - (Boundary head + frame classifier)": ("Predicted", "Boundary head + frame classifier")}


# ----------------------------------------------------------------------------------------------- models
def verify_runs(ctx, device) -> dict:
    """Every queued run: finished, 18,750 updates, intact final weights and the identity recomputed from the inputs."""
    runs = {}
    for key in ctx.fp["queue"]:
        group, seed = key.split("_s"); seed = int(seed); man_path = ctx.out / "train" / key / "run_manifest.json"
        if not man_path.is_file():
            raise SystemExit(f"{key}: no finished run")
        man = json.loads(man_path.read_text()); plan, plan_sha = tp.checked_plan(ctx, seed); head = tr.make_head(ctx, "FACT", plan, device)
        _, cut_fn, head_sha = tp.frozen_head(ctx, seed, device)
        ident = tp.identity(ctx, group, seed, head, plan_sha, sha256_file(ctx.fact_init(seed)), head_sha)
        if man["identity"] != ident or int(man["completed_updates"]) != int(ctx.tr["total_updates"]) or sha256_file(Path(man["final_checkpoint"])) != man["final_sha256"]:
            raise SystemExit(f"{key}: identity, update count or final weights do not verify")
        model = tp.new_fact(ctx, head, group, cut_fn, seed, device); model.load_state_dict(torch.load(man["final_checkpoint"], map_location=device, weights_only=True))
        runs[key] = {"group": group, "seed": seed, "head": head, "model": model.eval(), "manifest": man, "cut_fn": cut_fn}
    return runs


def source_models(ctx, device) -> tuple[dict, dict]:
    """Original FACT (the paired round's frozen weights, official partition) and the paired ASRF, per seed."""
    orig, asrf = {}, {}
    for seed in ctx.seeds:
        plan, _ = tp.checked_plan(ctx, seed)
        for arm, store in (("FACT", orig), ("ASRF", asrf)):
            key = f"{arm}_s{seed}"; path = ctx.root / "models_frozen" / SRC / key / "final.pt"
            if sha256_file(path) != ctx.fp["source"]["frozen"][key]:
                raise SystemExit(f"{key}: the frozen weights differ from the pin")
            head = tr.make_head(ctx, arm, plan, device); model = head.build(); model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            if arm == "FACT":
                pp.partitioned(model, head.mods, "original")
            store[seed] = {"head": head, "model": model.to(device).eval()}
    return orig, asrf


def freeze(ctx, runs: dict, config: str) -> dict:
    out = {}
    for key, r in runs.items():
        man, dest = r["manifest"], ctx.root / "models_frozen" / NS / key
        try:
            if not dest.exists():
                files = [man["final_checkpoint"], str(ctx.out / "train" / key / "run_manifest.json"), man["plan"], man["init"], str(ctx.root / config)]
                subprocess.run([sys.executable, "scripts/freeze_model.py", NS, key, *[x for f in files for x in ("--file", f)], "--meta", f"identity={man['identity']}",
                                "--meta", f"group={r['group']}", "--meta", f"seed={r['seed']}", "--meta", f"head_sha256={man['head_sha256']}",
                                "--notes", "Predicted-partition round (2026-10-09); final weights after 18,750 updates; the head is frozen apart."],
                               cwd=ctx.root, check=True, capture_output=True)
            out[key] = str(dest) if sha256_file(dest / "final.pt") == man["final_sha256"] else f"frozen copy differs from {man['final_sha256'][:12]}"
        except (subprocess.CalledProcessError, OSError) as e:
            out[key] = f"not frozen: {type(e).__name__}"
    return out


# ----------------------------------------------------------------------------------------------- decoding
@torch.no_grad()
def decode_fact(head, model, seqs: list, device) -> dict:
    """Official output, the U frame branch's probabilities, each block's frame-branch argmax, the last block's
    partition and (Predicted) the head's cuts of every sequence."""
    model.eval(); out = {}
    for s in seqs:
        x = torch.from_numpy(s.feat.astype(np.float32)).to(device); pr = head.predict(model, x); bl = model.block_list
        out[s.seq_id] = {"labels": pr["labels"], "frame_prob": pr["frame_prob"], "i": bl[0].frame_clogit[:, 0].argmax(-1).cpu().numpy(),
                         "u": bl[1].frame_clogit[:, 0].argmax(-1).cpu().numpy(), "U": pr["frame_labels"], "partition": pp.last_partition(model),
                         "head_cuts": model.cut_fn(x) if model.cut_fn is not None else None, "tokens_used": pr["tokens_used"]}
    return out


@torch.no_grad()
def head_frame_classifier(cut_fn, seqs: list, device) -> dict:
    """The head's cuts split each sequence; every interval takes the argmax of its mean frozen E1 probabilities."""
    out = {}
    for s in seqs:
        cuts = cut_fn(torch.from_numpy(s.feat.astype(np.float32)).to(device)); prob = np.asarray(s.prob, float); lab = np.empty(len(prob), int)
        for a, b in pp.add_cuts([(0, len(prob))], cuts):
            lab[a:b] = int(prob[a:b].mean(0).argmax())
        out[s.seq_id] = {"labels": lab, "head_cuts": cuts}
    return out


@torch.no_grad()
def position_control(head, model, seqs: list, device, normal: dict) -> tuple[dict, dict]:
    """The Predicted model with the head's cuts moved evenly inside each official run, against its normal output. The
    weights and the inputs to the last block must stay unchanged, and the normal policy must reproduce afterwards."""
    model.eval(); w0 = {k: v.clone() for k, v in model.state_dict().items()}; even, same = {}, 0
    for s in seqs:
        x = torch.from_numpy(s.feat.astype(np.float32)).to(device).unsqueeze(1)
        pp.set_policy(model, "predicted_even"); oe = model._forward_one_video(x); even[s.seq_id] = model.block_list[-1].eval().cpu().numpy()
        pp.set_policy(model, "predicted"); on = model._forward_one_video(x)
        same += int(all(torch.equal(a, b) for i in range(len(oe) - 1) for a, b in zip(oe[i], on[i])))
    pp.set_policy(model, "predicted")
    again = sum(int(np.array_equal(head.predict(model, torch.from_numpy(s.feat.astype(np.float32)).to(device))["labels"], normal[s.seq_id]["labels"])) for s in seqs)
    return even, {"sequences": len(seqs), "inputs_to_U_unchanged": same, "normal_output_reproduced": again,
                  "weights_unchanged": all(torch.equal(v, w0[k]) for k, v in model.state_dict().items())}


def save_decoded(ctx, stem: str, dec: dict) -> None:
    """Per-sequence outputs on the server: concatenated arrays with offsets (npz) and the sequence ids (json)."""
    d = ctx.out / "evaluation"; d.mkdir(parents=True, exist_ok=True); ids = list(dec); arrays = {"lengths": np.asarray([len(dec[k]["labels"]) for k in ids])}
    arrays["labels"] = np.concatenate([np.asarray(dec[k]["labels"], np.int8) for k in ids])
    for f, dt in (("frame_prob", np.float16), ("prob", np.float16), ("i", np.int8), ("u", np.int8), ("U", np.int8), ("frame_labels", np.int8)):
        if f in dec[ids[0]] and dec[ids[0]][f] is not None:
            arrays[f] = np.concatenate([np.asarray(dec[k][f], dt) for k in ids])
    for f in ("partition", "head_cuts"):
        if f in dec[ids[0]] and dec[ids[0]][f] is not None:
            vals = [[a for a, _ in dec[k][f]] if f == "partition" else list(dec[k][f]) for k in ids]
            arrays[f"{f}_counts"] = np.asarray([len(v) for v in vals]); arrays[f] = np.asarray([x for v in vals for x in v], np.int32)
    np.savez_compressed(d / f"{stem}.npz", **arrays); write_atomic(d / f"{stem}_ids.json", json.dumps(ids))


def frags_map(ctx, inp) -> dict:
    """Stream id -> [recording, start, end) of its fragments, for the bank streams and both halves' streams."""
    out = {}
    for recipe in [*ctx.data.recipes, *[st.recipe for h in HALVES for st in (*inp.halves[h].pairs, *inp.halves[h].multi)]]:
        pos, fr = 0, []
        for f in recipe.fragments:
            fr.append([f.video_id, pos, pos + f.end - f.start]); pos += f.end - f.start
        out[recipe.recipe_id] = fr
    return out


def stage_records(seqs: list, dec: dict, frags: dict) -> list[dict]:
    return [{"id": s.seq_id, "native": bool(s.is_native), "ref": s.labels.tolist(), "partition": d["partition"], "head_cuts": d["head_cuts"],
             "frags": [[s.seq_id.rsplit("@", 1)[0], 0, int(s.labels.size)]] if s.is_native else frags[s.seq_id],
             "stages": {"E1": np.asarray(s.prob).argmax(1).tolist(), "i": d["i"].tolist(), "u": d["u"].tolist(), "U": np.asarray(d["U"]).tolist(), "final": d["labels"].tolist()}}
            for s in seqs for d in [dec[s.seq_id]]]


# ----------------------------------------------------------------------------------------------- comparisons
def comparisons(fp: dict, systems: dict, inp, tol: float, names) -> dict:
    from scripts.report_model_development import paired_bootstrap
    out = {}
    for seed in [int(s) for s in fp["seeds"]]:
        for name in names:
            a, b = COMPARE[name]; ka, kb = f"{a} s{seed}", (f"{b} s{seed}" if b in SEEDED else b)
            for h in HALVES:
                A, B = systems[ka][h], systems[kb][h]
                c = {"bootstrap": paired_bootstrap(A["tables"], B["tables"]),
                     "native_changes": native_changes(A["tables"]["native"], B["tables"]["native"])}
                if B["preds"] is not None:
                    c["segment_changes"] = segment_changes(inp.halves[h], A["preds"], B["preds"]); c["boundary_changes"] = boundary_changes(inp.halves[h], A["preds"], B["preds"], tol)
                out[f"s{seed}|{name}|{h}"] = c
    return out


def verdicts(fp: dict, comp: dict, names) -> dict:
    seeds = [int(s) for s in fp["seeds"]]
    return {n: {h: consistency([comp[f"s{s}|{n}|{h}"]["bootstrap"]["multi segment F1@50"] for s in seeds]) for h in HALVES} for n in names}


def plot_curves(ctx, runs: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.2)); colors = {"Uniform": "#4c72b0", "Predicted": "#c44e52"}; styles = {0: "-", 1: "--", 2: ":"}
    for key, r in runs.items():
        rec = r["manifest"]["records"]; x = [q["updates"] for q in rec]; kw = {"color": colors[r["group"]], "ls": styles[r["seed"]], "lw": 1.2, "label": key}
        ax[0].plot(x, [q["loss"]["total"] for q in rec], **kw); ax[1].plot(x, [q["monitor"]["segment_f1_50"] for q in rec], **kw); ax[2].plot(x, [q["monitor"]["U_intervals_mean"] for q in rec], **kw)
    for seed in ctx.seeds:
        rec = json.loads((ctx.out / "heads" / f"head_s{seed}.json").read_text())["records"]
        ax[3].plot([q["updates"] for q in rec], [q["loss"] for q in rec], color="#555555", ls=styles[seed], label=f"head s{seed}")
    ax[0].set_yscale("log")
    for a, t in zip(ax, ("FACT training loss (mean per epoch)", "monitor: 16 fixed training sequences, segment F1@50", "monitor: intervals entering U's averaging", "boundary head training loss")):
        a.set_title(t, fontsize=10); a.set_xlabel("updates"); a.grid(alpha=0.3, which="both"); a.legend(fontsize=7, ncol=2)
    fig.suptitle("Predicted-partition round (final weights kept; nothing chosen from these curves)", fontsize=11); fig.tight_layout(); fig.savefig(path, dpi=130)


def write(ctx, res: dict) -> None:
    """Both id guards on every write; called after each stage so that nothing finished is lost."""
    text = json.dumps(res, indent=1, default=str); assert_no_ids(json.loads(text), ctx.data.ids(), "summary"); tables = pt.tables_md(res)
    if any(x in text or x in tables for x in ctx.data.sessions()):
        raise SystemExit("a session id appears in the report")
    ctx.results.mkdir(parents=True, exist_ok=True); write_atomic(ctx.results / "summary.json", text); write_atomic(ctx.results / "report_tables.md", tables)


# ----------------------------------------------------------------------------------------------- evaluation
def run_evaluate(config: str) -> None:
    from scripts.report_center_supervision import validation
    t0 = time.time(); cfg, fp, scfg, paths = tp.load_round(config); ctx = tp.PCtx(cfg, scfg, fp, paths); fa = ctx.fa
    device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction()); tp.deterministic()
    runs = verify_runs(ctx, device); orig, asrf = source_models(ctx, device); print("runs verified", flush=True)
    hz, tol_s = float(cfg["protocol"]["hz"]), float(cfg["selection"]["boundary_tolerance_s"]); tol, short_max = tol_s * hz, int(fp["evaluation"]["units"]["short_max_samples"])
    res = {"runs": {k: {x: r["manifest"][x] for x in ("identity", "group", "seed", "policy", "completed_updates", "final_sha256", "plan_sha256", "init_sha256", "head_sha256",
                                                     "parameters", "head_parameters", "minutes")} for k, r in runs.items()},
           "frozen": freeze(ctx, runs, config), "decision": "pending"}
    write(ctx, res); plot_curves(ctx, runs, ctx.results / "training_curves.png")                            # stage 1
    inp, seqs_ref = validation(cfg, {"inputs": {"protocol": fa["inputs"]["protocol"], "feature_manifest": fa["inputs"]["e1_feature_manifest"]}}, paths)
    frags = frags_map(ctx, inp); chosen = [s for _, s in tr.monitor_set(ctx)]; systems, stage, cost, pos_rows, pos_checks, cut_stats, counts = {}, {}, {}, {}, {}, {}, {}
    iu, position = {g: {} for g in FACT_SYSTEMS}, {}
    models = {s: {"Original FACT": (orig[s]["head"], orig[s]["model"]), "Uniform": (runs[f"Uniform_s{s}"]["head"], runs[f"Uniform_s{s}"]["model"]),
                  "Predicted": (runs[f"Predicted_s{s}"]["head"], runs[f"Predicted_s{s}"]["model"])} for s in ctx.seeds}
    for seed in ctx.seeds:
        for name, (head, model) in models[seed].items():
            tp.deterministic(); dec = decode_fact(head, model, chosen, device)
            stage.setdefault(f"{name} s{seed}", {})["training monitor"] = pt.stage_table(pt.stage_units(stage_records(chosen, dec, frags), tol, short_max))
    for h in HALVES:
        ev = inp.halves[h]; seqs = ev.sequences(); dev_set = [s for s in seqs if s.is_native or s.seq_id in {st.recipe.recipe_id for st in ev.multi}]
        joins = {st.recipe.recipe_id: [int(j) for j in st.joins] for st in ev.multi}
        for seed in ctx.seeds:
            for name, (head, model) in models[seed].items():
                tp.deterministic(); torch.cuda.synchronize(); t = time.time(); dec = decode_fact(head, model, seqs, device); torch.cuda.synchronize()
                cost.setdefault(f"{name} s{seed}", []).append((time.time() - t) / len(seqs)); save_decoded(ctx, f"{name.replace(' ', '_')}_s{seed}_{h}", dec)
                preds = {k: v["labels"] for k, v in dec.items()}; m, tab = evaluate_predictions(preds, ev, hz, tol_s)
                systems.setdefault(f"{name} s{seed}", {})[h] = {"row": summary(m), "tables": tab, "preds": preds,
                                                                "frame_branch_row": summary(evaluate_predictions({k: v["U"] for k, v in dec.items()}, ev, hz, tol_s)[0])}
                recs = stage_records(dev_set, dec, frags); units = pt.stage_units(recs, tol, short_max)
                stage[f"{name} s{seed}"][f"development {h}"] = pt.stage_table(units); iu[name].setdefault(seed, {})[h] = pt.lost_at_i_u(units)
                counts.setdefault(f"{name} s{seed}", {})[h] = pt.interval_counts(recs)
                if name == "Predicted":
                    cut_stats.setdefault(f"Predicted s{seed}", {})[h] = pt.head_cut_stats(recs, joins, tol)
                    even, checks = position_control(head, model, seqs, device, dec); pos_checks[f"s{seed}|{h}"] = checks
                    me, te = evaluate_predictions({k: np.asarray(v) for k, v in even.items()}, ev, hz, tol_s); se = summary(me); sn = systems[f"Predicted s{seed}"][h]["row"]
                    pos_rows[f"s{seed}|{h}"] = {"segment F1@50 (head positions)": sn["segment_f1_50_multi"], "segment F1@50 (even positions)": se["segment_f1_50_multi"],
                                                "difference": sn["segment_f1_50_multi"] - se["segment_f1_50_multi"],
                                                "boundary R difference": (sn["boundary_recall_multi"] or 0) - (se["boundary_recall_multi"] or 0),
                                                "boundary P difference": (sn["boundary_precision_multi"] or 0) - (se["boundary_precision_multi"] or 0),
                                                **{f"runs {k}": v["segments"] for k, v in segment_changes(ev, preds, {k: np.asarray(v) for k, v in even.items()}).items()}}
                    position.setdefault(seed, {})[h] = pos_rows[f"s{seed}|{h}"]["difference"]
            tr.determinism("ASRF"); dec = asrf[seed]["head"].decode(asrf[seed]["model"], seqs, device); tp.deterministic()
            preds = {k: np.asarray(v["labels"], int) for k, v in dec.items()}; m, tab = evaluate_predictions(preds, ev, hz, tol_s)
            systems.setdefault(f"ASRF s{seed}", {})[h] = {"row": summary(m), "tables": tab, "preds": preds}
            save_decoded(ctx, f"ASRF_s{seed}_{h}", {k: {"labels": preds[k], "frame_labels": v["frame_labels"], "prob": v["prob"]} for k, v in dec.items()})
            _, cut_fn, _ = tp.frozen_head(ctx, seed, device); dec = head_frame_classifier(cut_fn, seqs, device); preds = {k: v["labels"] for k, v in dec.items()}
            m, tab = evaluate_predictions(preds, ev, hz, tol_s); systems.setdefault(f"Boundary head + frame classifier s{seed}", {})[h] = {"row": summary(m), "tables": tab, "preds": preds}
            save_decoded(ctx, f"Head_frame_classifier_s{seed}_{h}", dec)
            cut_stats.setdefault(f"Boundary head + frame classifier s{seed}", {})[h] = pt.head_cut_stats(
                [{"id": s.seq_id, "native": bool(s.is_native), "ref": s.labels.tolist(), "head_cuts": dec[s.seq_id]["head_cuts"]} for s in dev_set], joins, tol)
        print(f"{h}: every system decoded", flush=True)
    first = ("Predicted - Uniform", "Predicted - Original FACT", "Uniform - Original FACT", "Predicted - ASRF", "Uniform - ASRF", "Predicted - (Boundary head + frame classifier)")
    comp = comparisons(fp, systems, inp, tol, first)
    res.update({"systems": {k: {h: {x: v for x, v in r.items() if x in ("row", "frame_branch_row")} for h, r in per.items()} for k, per in systems.items()},
                "comparisons": comp, "verdicts": verdicts(fp, comp, first),
                "units": {k: {h: ra.mechanism(inp.halves[h], systems[k][h]["preds"], short_max) for h in HALVES} for k in systems},
                "stage_tracking": stage, "lost_first_at_i_or_u": iu, "interval_counts": counts, "head_cuts": cut_stats,
                "position_control": {"rows": pos_rows, "checks": pos_checks}, "seconds": time.time() - t0})
    write(ctx, res); print("round systems evaluated", flush=True)                                            # stage 2
    cases = []
    for h in HALVES:
        for st in inp.halves[h].multi:
            true = labels_to_segments(st.labels)
            for k in range(len(true)):
                if ra.xyx_middle((pt.A4C, pt.A5C))(true, k):
                    a, b = max(0, true[k].start - 30), min(st.labels.size, true[k].end + 30); sid = st.recipe.recipe_id
                    tracks = {"reference": st.labels[a:b], "E1 argmax": np.asarray(st.prob).argmax(1)[a:b]}
                    tracks.update({f"{n} s{s}": np.asarray(systems[f"{n} s{s}"][h]["preds"][sid])[a:b] for n in SEEDED for s in ctx.seeds})
                    cases.append({"title": f"{h}: A4C/A5C middle {len(cases) + 1}, {true[k].length} samples", "tracks": tracks})
    pt.plot_cases(cases, ctx.results / "cases_a4c_a5c_middles.png")
    try:
        refs, e1_preds, g1 = ra.reference_systems(cfg, fa, paths, inp, seqs_ref); ra.add_references(cfg, systems, refs, e1_preds, inp)
    except (SystemExit, Exception) as e:  # noqa: BLE001 - a reproduction failure is reported with its cause; the decision then stays open
        res.update({"references": {"not reproduced": f"{type(e).__name__}: {e}"}, "decision": "not computed: the references did not reproduce", "seconds": time.time() - t0})
        write(ctx, res); return
    rest = tuple(n for n in COMPARE if n not in first); comp.update(comparisons(fp, systems, inp, tol, rest)); ver = verdicts(fp, comp, tuple(COMPARE))
    res["units"]["E1 + B3"] = {h: ra.mechanism(inp.halves[h], systems["E1 + B3"][h]["preds"], short_max) for h in HALVES}
    hours = {k: r["manifest"]["minutes"] / 60 for k, r in runs.items()}
    heads = {s: json.loads((ctx.out / "heads" / f"head_s{s}.json").read_text()) for s in ctx.seeds}
    smokes = {g: json.loads((ctx.out / "smoke" / f"{g}_s{int(fp['smoke']['seed'])}.json").read_text()) for g in tp.GROUPS}
    res.update({"systems": {k: {h: {x: v for x, v in r.items() if x in ("row", "frame_branch_row")} for h, r in per.items()} for k, per in systems.items()},
                "comparisons": comp, "verdicts": ver, "decision": pt.decide(fp, ver, {k: {h: r[h]["row"] for h in HALVES} for k, r in systems.items()}, iu, position),
                "references": {k: {h: v[h]["metrics"] for h in HALVES} for k, v in refs.items()}, "references_g1": {"schema": "stored rows", **g1},
                "cost": {"formal_training_hours": hours, "head_training_hours": {s: h["gpu_seconds"] / 3600 for s, h in heads.items()},
                         "smoke_hours": {g: r["gpu_seconds"] / 3600 for g, r in smokes.items()}, "decode_seconds_per_sequence": {k: float(np.mean(v)) for k, v in cost.items()},
                         "evaluation_hours": (time.time() - t0) / 3600,
                         "total_gpu_hours": sum(hours.values()) + sum(h["gpu_seconds"] for h in heads.values()) / 3600 + sum(r["gpu_seconds"] for r in smokes.values()) / 3600 + (time.time() - t0) / 3600},
                "seconds": time.time() - t0})
    write(ctx, res); print(json.dumps(res["decision"], indent=1, default=str))                              # stage 3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--evaluate", action="store_true", required=True)
    run_evaluate(ap.parse_args().config)


if __name__ == "__main__":
    main()
