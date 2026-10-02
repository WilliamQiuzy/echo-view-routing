#!/usr/bin/env python
"""Round-1 report of model development (2026-10-02 plan, Task 6).

Reads the queue status (learned rows M0-M4 and extra rows) and the simple-decoder calibration run, then writes to
--out: tables.md (both validation halves), report.json (all metrics, paired bootstrap differences, gate checks,
confusions, error runs), training_curves.png (selection-half curves of every learned row) and digests.json.

Paired differences resample the same units for both methods: native clips by recording session, multi-fragment
streams by stream (2,000 draws). The calibration half is the primary comparison: no learned checkpoint was chosen on
it, while the simple decoders were tuned on it, so their calibration numbers are optimistic for them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg  # noqa: E402
from scripts.calibrate_simple_decoders import decode_all, probs_of  # noqa: E402
from scripts.model_development_protocol import session_of  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from echo_routing.config import load_paths  # noqa: E402
from echo_routing.temporal.dev_run import HALVES, Inputs  # noqa: E402

N_BOOT, BOOT_SEED = 2000, 20261002
CLASSES = ["PLAX", "PSAX", "A4C", "A5C", "SC4C"]
TABLES = ("native", "multi_seg", "multi_frag")
LEARNED_NAMES = {"M0": "M0 MS-TCN (official, adapted)", "M1": "M1 ASRF (official, adapted)", "M2": "M2 local attention + gate",
                 "M3": "M3 = M2 + boundary loss", "M4": "M4 boundary-controlled (full)", "M4_fixedgate": "M4, gate fixed at 0.5",
                 "M1_aug": "M1 + feature masking", "M4_aug": "M4 + feature masking",
                 "M5": "M5 = M4 + segment pooling", "M3p": "M3p = M3 + segment pooling"}
NEW_ROWS = ("M4", "M5")
SIMPLE_ROWS = ("B0 argmax", "B1 tuned (main)", "B2 tuned (main)", "B3 tuned (main)", "B3 audited",
               "B1 tuned (stability)", "B2 tuned (stability)", "B3 tuned (stability)")
BASELINE_ROWS = ("M0", "M1", "B1 tuned (main)", "B2 tuned (main)", "B3 tuned (main)")


def read_rows(run_dir: Path, half: str, tag: str | None = None) -> dict[str, pd.DataFrame]:
    pre = f"rows_{half}_" if tag is None else f"rows_{tag}_{half}_"
    return {t: pd.read_csv(run_dir / f"{pre}{t}.csv", dtype={"video_id": str, "stream": str}) for t in TABLES}


def metric_fns() -> dict:
    def macro_f1(nat: pd.DataFrame) -> float:
        conf = np.zeros((5, 5)); np.add.at(conf, (nat["label"].to_numpy(), nat["majority"].to_numpy()), 1)
        tp = np.diag(conf); den = 2 * tp + (conf.sum(0) - tp) + (conf.sum(1) - tp)
        return float(np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0).mean())
    return {
        "native macro-F1": ("native", macro_f1),
        "native cuts per clip": ("native", lambda d: float(d["cuts"].mean())),
        "multi segment F1@50": ("multi_seg", lambda d: float(2 * d["seg_matched"].sum() / (d["seg_pred"].sum() + d["seg_true"].sum()))),
        "multi boundary recall": ("multi_frag", lambda d: float(d["t_matched"].sum() / d["t_true"].sum())),
        "multi boundary F1": ("multi_frag", lambda d: float(2 * d["t_matched"].sum() / (d["p_pred"].sum() + d["t_true"].sum()))),
        "same-class cut rate": ("multi_frag", lambda d: float(d["false_split"].sum() / d["same_join"].sum())),
    }


def paired_bootstrap(a: dict[str, pd.DataFrame], b: dict[str, pd.DataFrame]) -> dict:
    """a - b with native clips resampled by session and streams by stream (same draws for both)."""
    for t in ("multi_seg", "multi_frag"):
        if set(a[t]["stream"]) != set(b[t]["stream"]):
            raise ValueError(f"paired bootstrap: the two rows cover different streams in {t}")
    if set(a["native"]["video_id"]) != set(b["native"]["video_id"]):
        raise ValueError("paired bootstrap: the two rows cover different native clips")
    rng = np.random.default_rng(BOOT_SEED); fns = metric_fns()
    nat_a = a["native"].assign(session=a["native"]["video_id"].map(session_of)); nat_b = b["native"].set_index("video_id").loc[nat_a["video_id"]].reset_index()
    sessions = nat_a["session"].unique(); by_session = {s: np.flatnonzero(nat_a["session"].to_numpy() == s) for s in sessions}
    streams = a["multi_seg"]["stream"].unique()
    seg_idx = {t: {s: np.flatnonzero(d["stream"].to_numpy() == s) for s in streams} for t, d in
               (("a_seg", a["multi_seg"]), ("b_seg", b["multi_seg"]), ("a_frag", a["multi_frag"]), ("b_frag", b["multi_frag"]))}
    point = {k: fns[k][1](a[fns[k][0]]) - fns[k][1](b[fns[k][0]]) for k in fns}
    draws = {k: [] for k in fns}
    for _ in range(N_BOOT):
        ss = rng.choice(sessions, size=sessions.size, replace=True); st = rng.choice(streams, size=streams.size, replace=True)
        ni = np.concatenate([by_session[s] for s in ss])
        res = {"native": (nat_a.iloc[ni], nat_b.iloc[ni]),
               "multi_seg": (a["multi_seg"].iloc[np.concatenate([seg_idx["a_seg"][s] for s in st])],
                             b["multi_seg"].iloc[np.concatenate([seg_idx["b_seg"][s] for s in st])]),
               "multi_frag": (a["multi_frag"].iloc[np.concatenate([seg_idx["a_frag"][s] for s in st])],
                              b["multi_frag"].iloc[np.concatenate([seg_idx["b_frag"][s] for s in st])])}
        for k, (t, f) in fns.items():
            draws[k].append(f(res[t][0]) - f(res[t][1]))
    return {k: {"delta": point[k], "ci95": [float(np.percentile(draws[k], 2.5)), float(np.percentile(draws[k], 97.5))]} for k in fns}


def error_runs(pred: dict[str, np.ndarray], truth: dict[str, np.ndarray], ids: list[str]) -> dict:
    """Runs of consecutive wrong samples: count per sequence, length distribution."""
    lengths = []
    for i in ids:
        wrong = np.asarray(pred[i]) != truth[i]
        if wrong.any():
            edges = np.diff(np.concatenate([[0], wrong.astype(int), [0]]))
            lengths.extend((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).tolist())
    n = sum(truth[i].size for i in ids)
    return {"runs": len(lengths), "runs_per_sequence": len(lengths) / len(ids), "wrong_samples": int(sum(lengths)),
            "wrong_share": sum(lengths) / n, "median_run": float(np.median(lengths)) if lengths else 0.0,
            "runs_ge_10": int(sum(l >= 10 for l in lengths)), "runs_le_3": int(sum(l <= 3 for l in lengths))}


def fixed_broken(new: dict, ref: dict, truth: dict, ids: list[str]) -> dict:
    fixed = broken = 0
    for i in ids:
        a = np.asarray(new[i]) == truth[i]; b = np.asarray(ref[i]) == truth[i]
        fixed += int((a & ~b).sum()); broken += int((~a & b).sum())
    return {"fixed_samples": fixed, "broken_samples": broken}


def confusions(pred: dict, truth: dict, natives: list[str]) -> dict:
    clip = np.zeros((5, 5), int); frame = np.zeros((5, 5), int)
    for i in natives:
        y = int(truth[i][0]); p = np.asarray(pred[i])
        clip[y, int(np.bincount(p, minlength=5).argmax())] += 1; np.add.at(frame, (np.full(p.size, y), p), 1)
    return {"clip": clip.tolist(), "frame": frame.tolist(),
            "A4C->A5C clips": int(clip[2, 3]), "A5C->A4C clips": int(clip[3, 2]), "PLAX->PSAX clips": int(clip[0, 1]),
            "PSAX->PLAX clips": int(clip[1, 0]), "A4C<->A5C frames": int(frame[2, 3] + frame[3, 2]),
            "PLAX<->PSAX frames": int(frame[0, 1] + frame[1, 0])}


def fmt(x, d=3) -> str:
    return "n/a" if x is None else f"{x:.{d}f}"


def table_md(rows: dict[str, dict], half: str) -> list[str]:
    out = [f"| Method | native macro-F1 | recall PLAX/PSAX/A4C/A5C/SC4C | multi segF1@50 | boundary recall | boundary F1 | "
           f"native cuts/clip | cuts/min | same-class cut rate | pairs segF1@50 |", "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in rows.items():
        m = r[half]; rec = "/".join(fmt(v, 2) for v in m["native_per_class_recall"])
        out.append(f"| {name} | {fmt(m['native_macro_f1'])} | {rec} | {fmt(m['segment_f1_50_multi'])} | {fmt(m['boundary_recall_multi'])} | "
                   f"{fmt(m['boundary_f1_multi'])} | {fmt(m['native_internal_cuts_per_clip'])} | {fmt(m['native_internal_cuts_per_min'], 2)} | "
                   f"{fmt(m['same_class_cut_rate_multi'])} | {fmt(m['segment_f1_50_pairs'])} |")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--status", required=True); ap.add_argument("--simple-run", required=True)
    ap.add_argument("--extra", default="", help="row=run_dir;row=run_dir (lr retries, ablations)")
    ap.add_argument("--feature-manifest", required=True); ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("-c", "--config", default="configs/experiments/boundary_context_v1.yaml")
    args = ap.parse_args(); args.set = []
    cfg = load_cfg(args); paths = load_paths(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    status = json.loads(Path(args.status).read_text())
    learned = {k: Path(v["run_dir"]) for k, v in status.items() if v.get("state") == "done" and k in LEARNED_NAMES}
    for item in filter(None, args.extra.split(";")):
        k, _, v = item.partition("="); learned[k.strip()] = Path(v.strip())
    simple_dir = Path(args.simple_run); simple = json.loads((simple_dir / "validation_metrics.json").read_text())["methods"]
    inputs = Inputs(cfg, Path(args.protocol), Path(args.feature_manifest), paths.cache_root)
    rows, row_tables, preds, manifests = {}, {}, {}, {}
    for k, d in learned.items():
        vm = json.loads((d / "validation_metrics.json").read_text()); manifests[k] = json.loads((d / "run_manifest.json").read_text())
        for key, want in (("protocol_sha256", inputs.proto_sha), ("feature_manifest_sha256", inputs.fman_sha)):
            if manifests[k][key] != want:
                raise SystemExit(f"{k}: {key} {manifests[k][key][:12]} differs from this report's input {want[:12]}")
        rows[k] = vm["method"]; row_tables[k] = {h: read_rows(d, h) for h in HALVES}
        preds[k] = {h: dict(np.load(d / f"predictions_{h}.npz")) for h in HALVES}
    probs = {h: probs_of(inputs.halves[h], inputs.store) for h in HALVES}
    for name in SIMPLE_ROWS:
        if name not in simple:
            continue
        tag = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        rows[name] = {h: simple[name][h] for h in HALVES}; row_tables[name] = {h: read_rows(simple_dir, h, tag) for h in HALVES}
        preds[name] = {h: decode_all(simple[name]["method"], simple[name]["params"], probs[h]) for h in HALVES}
    truth, ids = {}, {}
    for h in HALVES:
        ev = inputs.halves[h]
        truth[h] = {**{s.seq_id: s.labels for s in ev.natives}, **{s.recipe.recipe_id: s.labels for s in (*ev.pairs, *ev.multi)}}
        ids[h] = {"native": [s.seq_id for s in ev.natives], "multi": [s.recipe.recipe_id for s in ev.multi]}
    report = {"rows": rows, "manifests": manifests, "simple_selected": json.loads((simple_dir / "selected.json").read_text())}
    # paired differences of M4 (and its variants) against every other row, both halves
    report["paired"] = {}
    for new in [k for k in rows if k.startswith(NEW_ROWS)]:
        for ref in rows:
            if ref != new:
                report["paired"][f"{new} - {ref}"] = {h: paired_bootstrap(row_tables[new][h], row_tables[ref][h]) for h in HALVES}
    # error structure
    report["errors"] = {k: {h: {"native_runs": error_runs(preds[k][h], truth[h], ids[h]["native"]),
                                "multi_runs": error_runs(preds[k][h], truth[h], ids[h]["multi"]),
                                "confusion": confusions(preds[k][h], truth[h], ids[h]["native"])} for h in HALVES} for k in preds}
    for new in [k for k in preds if k.startswith(NEW_ROWS)]:
        for ref in [r for r in preds if r != new]:
            report["errors"][new].setdefault("fixed_broken", {})[ref] = {
                h: {"native": fixed_broken(preds[new][h], preds[ref][h], truth[h], ids[h]["native"]),
                    "multi": fixed_broken(preds[new][h], preds[ref][h], truth[h], ids[h]["multi"])} for h in HALVES}
    # expansion gate (plan: M4 vs M0-M3 and tuned simple decoders, same encoder)
    gate = {}
    for h in HALVES:
        comps = [k for k in ("M0", "M1", "M2", "M3", "B1 tuned (main)", "B2 tuned (main)", "B3 tuned (main)") if k in rows]
        if "M4" in rows and comps:
            best_seg = max(comps, key=lambda k: rows[k][h]["segment_f1_50_multi"])
            best_nat = max(comps, key=lambda k: rows[k][h]["native_macro_f1"])
            m4 = rows["M4"][h]
            gate[h] = {"best_segment_comparator": best_seg, "segment_margin": m4["segment_f1_50_multi"] - rows[best_seg][h]["segment_f1_50_multi"],
                       "best_native_comparator": best_nat, "native_margin": m4["native_macro_f1"] - rows[best_nat][h]["native_macro_f1"],
                       "recall": m4["boundary_recall_multi"],
                       "passes": bool(m4["segment_f1_50_multi"] - rows[best_seg][h]["segment_f1_50_multi"] >= 0.005
                                      and m4["native_macro_f1"] - rows[best_nat][h]["native_macro_f1"] >= -0.005
                                      and (m4["boundary_recall_multi"] or 0) >= 0.95)}
    report["expansion_gate"] = gate
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    names = {k: LEARNED_NAMES.get(k, k) for k in rows}
    md = ["# Round-1 validation tables (generated by scripts/report_model_development.py)", ""]
    for h, title in (("calibration", "Calibration half (primary comparison; simple decoders were tuned here)"),
                     ("selection", "Selection half (learned checkpoints were chosen here)")):
        md += [f"## {title}", "", *table_md({names[k]: v for k, v in rows.items()}, h), ""]
    md += ["## Training and cost", "", "| Row | lr | epochs run | chosen epoch | checkpoint rule | updates | parameters | peak train GPU MB | train s | head ms/sample |",
           "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|"]
    for k, m in manifests.items():
        md.append(f"| {names[k]} | {m['settings']['lr']:g} | {m['epochs_run']} | {m['chosen_epoch']} | {m['checkpoint_status']} | {m['updates']} | "
                  f"{m['n_params']:,} | {m['max_train_peak_gpu_memory_mb']:.0f} | {m['train_seconds']:.0f} | {m['timing']['head_ms_per_sample']:.4f} |")
    md += ["", "## Paired differences, new-model row minus row (95% bootstrap interval; sessions for natives, streams for multi)", ""]
    for key, v in report["paired"].items():
        for h in HALVES:
            md.append(f"- {key} [{h}]: " + "; ".join(f"{m} {d['delta']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]" for m, d in v[h].items()))
    (out / "tables.md").write_text("\n".join(md) + "\n")
    plot(learned, manifests, out / "training_curves.png", names)
    digests = {k: {"best_sha256": m["checkpoints"]["best_sha256"], "predictions": m["predictions_digest"]} for k, m in manifests.items()}
    (out / "digests.json").write_text(json.dumps(digests, indent=1))
    print("\n".join(md))
    print(json.dumps(gate, indent=1))


def plot(learned: dict, manifests: dict, path: Path, names: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
    for k, d in learned.items():
        recs = [json.loads(l) for l in (d / "train_log.jsonl").read_text().splitlines() if l.strip()]
        ep = [r["epoch"] for r in recs]
        axes[0].plot(ep, [r["loss"]["total"] for r in recs], label=names[k])
        for ax, key in zip(axes[1:], ("segment_f1_50_multi", "native_macro_f1", "native_internal_cuts_per_clip")):
            ax.plot(ep, [r[key] for r in recs], label=names[k])
            c = manifests[k]["chosen_epoch"]; ax.scatter([c], [recs[c][key]], s=18)
    for ax, t in zip(axes, ("training loss (own objective)", "selection half: multi segment F1@50", "selection half: native macro-F1",
                            "selection half: native cuts per clip")):
        ax.set_title(t, fontsize=10); ax.set_xlabel("epoch")
    axes[0].set_yscale("log"); axes[1].legend(fontsize=7)
    fig.suptitle("Round-1 temporal heads on the v3 encoder, seed 0 (dots: selected checkpoints)"); fig.tight_layout()
    fig.savefig(path, dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
