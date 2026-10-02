#!/usr/bin/env python
"""Fixed-interval relabelling diagnostic (round 2; requested after Task 4). Frozen predictions only, no retraining.

For each frozen E1 run of ASRF and M5 (seeds 0-2) and each validation half:
- The intervals are the model's own predicted segments: runs of its default output (ASRF: the official boundary
  refinement; M5: argmax of the pooled output). Reference boundaries are never used.
- Every interval is labelled two ways on exactly the same intervals: "own" = argmax of the model's own mean class
  probability inside the interval; "visual" = argmax of the mean E1 probe probability inside the interval.
- Step 1 reports the classification difference on the fixed intervals: intervals whose label changes, samples
  labelled right, intervals whose label is not the majority proxy label, and their source clips counted once.
- Step 2 merges adjacent intervals with the same label (one rule for both labellings: labels are written per sample
  and segments are recomputed) and scores the result with the round-1 metrics, paired against "own" per seed.
It also draws two cases: an A5C clip that E1 fixes, and a fragment that the E1 probe labels right but M5 gets wrong.
Writes relabel.json, relabel.md, case figures and per-row tables into a new run directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg, new_run_dir  # noqa: E402
from scripts.report_model_development import paired_bootstrap  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from echo_routing.config import load_paths  # noqa: E402
from echo_routing.evaluate.dev_metrics import evaluate_predictions  # noqa: E402
from echo_routing.temporal.dev_data import FeatureStore  # noqa: E402
from echo_routing.temporal.dev_heads import head_from_manifest  # noqa: E402
from echo_routing.temporal.dev_run import HALVES, Inputs  # noqa: E402

CLASSES = ["PLAX", "PSAX", "A4C", "A5C", "SC4C"]
KEYS = ("native_macro_f1", "segment_f1_50_multi", "boundary_recall_multi", "boundary_f1_multi", "native_internal_cuts_per_clip",
        "same_class_cut_rate_multi", "multi_frame_accuracy")


def intervals(labels: np.ndarray) -> list[tuple[int, int]]:
    labels = np.asarray(labels)
    if labels.size == 0:
        return []
    cut = np.flatnonzero(np.diff(labels)) + 1
    return list(zip(np.concatenate([[0], cut]).tolist(), np.concatenate([cut, [labels.size]]).tolist()))


def relabel(prob: np.ndarray, ivs: list[tuple[int, int]]) -> tuple[np.ndarray, list[int]]:
    out = np.empty(prob.shape[0], int); lab = []
    for a, b in ivs:
        k = int(np.asarray(prob[a:b], float).mean(0).argmax()); out[a:b] = k; lab.append(k)
    return out, lab


def sources_of(seq, ev_sources: dict, a: int, b: int) -> set[str]:
    """Source clips covered by samples a..b-1 of a sequence (a native clip is its own source)."""
    spans = ev_sources.get(seq)
    if spans is None:
        return {seq}
    return {v for (s, e, v) in spans if s < b and e > a}


def interval_table(seq_ids, default, prob_own, prob_vis, truth, ev_sources) -> dict:
    n = changed = own_wrong = vis_wrong = 0; s_own = s_vis = n_samples = 0
    src_own, src_vis = set(), set(); fixed = broken = 0
    for sid in seq_ids:
        ivs = intervals(default[sid]); own, lo = relabel(prob_own[sid], ivs); vis, lv = relabel(prob_vis[sid], ivs); y = truth[sid]
        for (a, b), ko, kv in zip(ivs, lo, lv):
            maj = int(np.bincount(y[a:b], minlength=5).argmax()); n += 1; changed += int(ko != kv)
            if ko != maj:
                own_wrong += 1; src_own |= sources_of(sid, ev_sources, a, b)
            if kv != maj:
                vis_wrong += 1; src_vis |= sources_of(sid, ev_sources, a, b)
            fixed += int(ko != maj and kv == maj); broken += int(ko == maj and kv != maj)
        s_own += int((own == y).sum()); s_vis += int((vis == y).sum()); n_samples += y.size
    return {"intervals": n, "label_changed": changed, "own_wrong_intervals": own_wrong, "visual_wrong_intervals": vis_wrong,
            "visual_fixes": fixed, "visual_breaks": broken, "own_sample_accuracy": s_own / n_samples, "visual_sample_accuracy": s_vis / n_samples,
            "own_wrong_source_clips": len(src_own), "visual_wrong_source_clips": len(src_vis)}


def decode_run(rd: Path, inputs: Inputs, paths, device) -> tuple[dict, str]:
    man = json.loads((rd / "run_manifest.json").read_text())
    head = head_from_manifest(man, paths.repo_root, inputs.feature_dim, device)
    model = head.build().to(device); model.load_state_dict(torch.load(man["checkpoints"]["best"], map_location=device))
    return {h: head.decode(model, inputs.halves[h].sequences(), device) for h in HALVES}, man["head"]


def case_figure(path: Path, title: str, truth: np.ndarray, panels: list[tuple[str, np.ndarray]], labels: list[tuple[str, np.ndarray]],
                boundary: np.ndarray | None, joins: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = len(panels) + 1 + (boundary is not None)
    fig, axes = plt.subplots(rows, 1, figsize=(12, 1.9 * rows), sharex=True)
    t = np.arange(truth.size)
    for ax, (name, prob) in zip(axes, panels):
        for k, c in enumerate(CLASSES):
            ax.plot(t, prob[:, k], label=c, lw=1.2)
        ax.set_ylim(-0.02, 1.02); ax.set_ylabel(name, fontsize=8)
    axes[0].legend(fontsize=7, ncol=5, loc="upper right")
    ax = axes[len(panels)]
    for i, (name, lab) in enumerate([("reference", truth), *labels]):
        ax.scatter(t, np.full(t.size, i), c=[f"C{int(k)}" for k in lab], s=6, marker="s")
    ax.set_yticks(range(len(labels) + 1)); ax.set_yticklabels(["reference", *[n for n, _ in labels]], fontsize=7)
    if boundary is not None:
        axes[-1].plot(t, boundary, color="k", lw=1); axes[-1].set_ylim(-0.02, 1.02); axes[-1].set_ylabel("M5 boundary b", fontsize=8)
    for ax in axes:
        for j in joins:
            ax.axvline(j - 0.5, color="grey", ls=":", lw=0.8)
    axes[-1].set_xlabel("10 Hz sample (dotted: source joins; colours: PLAX C0, PSAX C1, A4C C2, A5C C3, SC4C C4)")
    fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", required=True, help="M1_s0=dir;...;M5_s2=dir (frozen E1 runs)")
    ap.add_argument("--feature-manifest", required=True); ap.add_argument("--protocol", required=True)
    ap.add_argument("--e0-cache-key", default="beb36d80df31"); ap.add_argument("-c", "--config", default="configs/experiments/model_development_v2.yaml")
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); paths = load_paths(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hz = float(cfg["protocol"]["hz"]); tol_s = float(cfg["selection"]["boundary_tolerance_s"])
    inputs = Inputs(cfg, Path(a.protocol), Path(a.feature_manifest), paths.cache_root)
    vis = {h: {**{s.seq_id: inputs.store.get(s.seq_id, "orig").prob for s in inputs.halves[h].natives},
               **{s.recipe.recipe_id: s.prob for s in (*inputs.halves[h].pairs, *inputs.halves[h].multi)}} for h in HALVES}
    truth = {h: {**{s.seq_id: s.labels for s in inputs.halves[h].natives},
                 **{s.recipe.recipe_id: s.labels for s in (*inputs.halves[h].pairs, *inputs.halves[h].multi)}} for h in HALVES}
    spans = {}
    for h in HALVES:
        for s in (*inputs.halves[h].pairs, *inputs.halves[h].multi):
            pos, out = 0, []
            for fr in s.recipe.fragments:
                out.append((pos, pos + fr.end - fr.start, fr.video_id)); pos += fr.end - fr.start
            spans[s.recipe.recipe_id] = out
    run_dir = new_run_dir(cfg, "md_relabel_diagnostic"); results, decoded = {}, {}
    for item in a.runs.split(";"):
        name, _, rd = item.partition("="); name = name.strip(); dec, head_name = decode_run(Path(rd.strip()), inputs, paths, device)
        decoded[name] = dec; results[name] = {"run_dir": rd.strip(), "head": head_name}
        for h in HALVES:
            ev = inputs.halves[h]; default = {k: v["labels"] for k, v in dec[h].items()}; own_p = {k: v["prob"] for k, v in dec[h].items()}
            own = {k: relabel(own_p[k], intervals(default[k]))[0] for k in default}
            vlab = {k: relabel(vis[h][k], intervals(default[k]))[0] for k in default}
            multi_ids = [s.recipe.recipe_id for s in ev.multi]; native_ids = [s.seq_id for s in ev.natives]
            step1 = {"multi": interval_table(multi_ids, default, own_p, vis[h], truth[h], spans),
                     "native": interval_table(native_ids, default, own_p, vis[h], truth[h], spans)}
            scored, tables = {}, {}
            for variant, labs in (("default", default), ("own", own), ("visual", vlab)):
                m, t = evaluate_predictions(labs, ev, hz, tol_s); scored[variant] = {k: m[k] for k in KEYS}; tables[variant] = t
                for tn in ("native", "multi_seg", "multi_frag"):
                    t[tn].to_csv(run_dir / f"rows_{name}_{variant}_{h}_{tn}.csv", index=False)
            boot = {"visual-own": paired_bootstrap(tables["visual"], tables["own"]),
                    "visual-default": paired_bootstrap(tables["visual"], tables["default"])}
            results[name][h] = {"step1_fixed_intervals": step1, "step2_scored": scored, "paired": boot}
            print(f"{name} {h}: step1 multi {json.dumps(step1['multi'])}", flush=True)
            print(f"{name} {h}: step2 " + " | ".join(f"{v} segF1 {scored[v]['segment_f1_50_multi']:.4f} rec {scored[v]['boundary_recall_multi']:.3f} "
                                                      f"natF1 {scored[v]['native_macro_f1']:.4f} cuts {scored[v]['native_internal_cuts_per_clip']:.3f}"
                                                      for v in scored), flush=True)
    # cases (selection half; M5 seed 0 and ASRF seed 0 for the plotted model outputs)
    e0 = FeatureStore(paths.cache_root, a.e0_cache_key); h = "selection"; ev = inputs.halves[h]
    cases = {}
    fixed_a5c = [s.seq_id for s in ev.natives if int(s.labels[0]) == 3
                 and np.bincount(e0.get(s.seq_id, "orig").prob.argmax(1), minlength=5).argmax() != 3
                 and np.bincount(vis[h][s.seq_id].argmax(1), minlength=5).argmax() == 3]
    m5 = decoded.get("M5_s0"); asrf = decoded.get("M1_s0")
    if fixed_a5c and m5 is not None:
        v = fixed_a5c[0]; y = truth[h][v]
        case_figure(run_dir / "case_A_e1_fixed_a5c.png", f"Case A: A5C clip {v} (E0 head wrong, E1 probe right)", y,
                    [("E0 head prob", e0.get(v, "orig").prob), ("E1 probe prob", vis[h][v]), ("M5 before pooling", m5[h][v]["prepool_prob"]),
                     ("M5 output (pooled)", m5[h][v]["prob"])],
                    [("E0 argmax", e0.get(v, "orig").prob.argmax(1)), ("E1 argmax", vis[h][v].argmax(1)), ("M5 s0", m5[h][v]["labels"]),
                     ("ASRF s0", asrf[h][v]["labels"] if asrf else y)], m5[h][v]["boundary_prob"], np.zeros(0, int))
        cases["A"] = {"clip": v, "candidates_fixed_by_e1": fixed_a5c,
                      "majority": {"E0": int(np.bincount(e0.get(v, "orig").prob.argmax(1), minlength=5).argmax()),
                                   "E1": int(np.bincount(vis[h][v].argmax(1), minlength=5).argmax()),
                                   **{k: int(np.bincount(decoded[k][h][v]["labels"], minlength=5).argmax()) for k in decoded}}}
    # case B: fragments the E1 probe labels right (fragment-mean argmax) but M5 gets wrong on >= 80% of the fragment
    hits = []
    for name in [k for k in decoded if k.startswith("M5")]:
        for hh in HALVES:
            for s in inputs.halves[hh].multi:
                sid = s.recipe.recipe_id; lab = decoded[name][hh][sid]["labels"]
                for (st, en, vid), fr in zip(spans[sid], s.recipe.fragments):
                    if int(vis[hh][sid][st:en].mean(0).argmax()) == fr.label and (lab[st:en] != fr.label).mean() >= 0.8:
                        hits.append({"run": name, "half": hh, "stream": sid, "video_id": vid, "start": st, "end": en, "label": int(fr.label),
                                     "m5_label": int(np.bincount(lab[st:en], minlength=5).argmax())})
    if hits:
        by_video: dict[str, int] = {}
        for x in hits:
            by_video[x["video_id"]] = by_video.get(x["video_id"], 0) + 1
        pick = next((x for x in hits if x["run"] == "M5_s0" and x["half"] == "selection"), hits[0])
        hh, sid = pick["half"], pick["stream"]; s = next(z for z in inputs.halves[hh].multi if z.recipe.recipe_id == sid)
        d = decoded[pick["run"]][hh][sid]
        case_figure(run_dir / "case_B_probe_right_m5_wrong.png",
                    f"Case B: {sid} ({hh}), fragment {pick['start']}-{pick['end']} {CLASSES[pick['label']]} read as {CLASSES[pick['m5_label']]} by {pick['run']}",
                    s.labels, [("E1 probe prob", vis[hh][sid]), ("M5 before pooling", d["prepool_prob"]), ("M5 output (pooled)", d["prob"])],
                    [("E1 argmax", vis[hh][sid].argmax(1)), (pick["run"], d["labels"])], d["boundary_prob"], s.joins)
        cases["B"] = {"picked": pick, "n_fragment_cases": len(hits), "distinct_source_clips": len(by_video), "cases_by_source_clip": by_video,
                      "by_run": {k: sum(1 for x in hits if x["run"] == k) for k in sorted({x["run"] for x in hits})}}
    (run_dir / "relabel.json").write_text(json.dumps({"results": results, "cases": cases}, indent=1, default=str))
    md = ["# Fixed-interval relabelling (frozen E1 runs; generated by scripts/model_development_relabel_diagnostic.py)", ""]
    for hh in HALVES:
        md += [f"## {hh} half", "", "| Run | intervals (multi) | label changed | wrong intervals own / visual | visual fixes / breaks | "
               "wrong source clips own / visual | segF1 default / own / visual | recall default / own / visual | native F1 own / visual | cuts own / visual |",
               "|---|---:|---:|---|---|---|---|---|---|---|"]
        for name, r in results.items():
            st, sc = r[hh]["step1_fixed_intervals"]["multi"], r[hh]["step2_scored"]
            md.append(f"| {name} | {st['intervals']} | {st['label_changed']} | {st['own_wrong_intervals']} / {st['visual_wrong_intervals']} | "
                      f"{st['visual_fixes']} / {st['visual_breaks']} | {st['own_wrong_source_clips']} / {st['visual_wrong_source_clips']} | "
                      f"{sc['default']['segment_f1_50_multi']:.3f} / {sc['own']['segment_f1_50_multi']:.3f} / {sc['visual']['segment_f1_50_multi']:.3f} | "
                      f"{sc['default']['boundary_recall_multi']:.3f} / {sc['own']['boundary_recall_multi']:.3f} / {sc['visual']['boundary_recall_multi']:.3f} | "
                      f"{sc['own']['native_macro_f1']:.3f} / {sc['visual']['native_macro_f1']:.3f} | "
                      f"{sc['own']['native_internal_cuts_per_clip']:.3f} / {sc['visual']['native_internal_cuts_per_clip']:.3f} |")
        md.append("")
    md += ["## Paired difference visual - own (segment F1@50, 95% bootstrap)", ""]
    for name, r in results.items():
        md.append(f"- {name}: " + "; ".join(f"{hh} {r[hh]['paired']['visual-own']['multi segment F1@50']['delta']:+.3f} "
                                             f"[{r[hh]['paired']['visual-own']['multi segment F1@50']['ci95'][0]:+.3f}, "
                                             f"{r[hh]['paired']['visual-own']['multi segment F1@50']['ci95'][1]:+.3f}]" for hh in HALVES))
    (run_dir / "relabel.md").write_text("\n".join(md) + "\n")
    print("\n".join(md)); print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "cases_by_source_clip"} for k, v in cases.items()}, indent=1, default=str))
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
