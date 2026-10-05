#!/usr/bin/env python
"""Report of the Pediatric adaptation comparison D0/D1/D2 (configs/experiments/pediatric_adapt_v1.yaml).

For every seed and arm, three systems read the Pediatric test folds: the frame classifier (per-sample argmax of the
arm's head), B3 with the arm's EV9V-selected parameters, and ASRF (D0 and D1: E1's ASRF seed k; D2: its own). Native
clips give A4C/PSAX recall, the two-class macro-F1, and clips turned right or wrong against D0 with their groups. The
frozen test exam streams (stitched whole clips; a constructed proxy) give segment F1@50, boundary recall within 0.5 s,
cuts more than 0.5 s from every clip join, and the share of true view changes found with both views right. Intervals
resample groups (first file-name token), the same draws for every arm. EV9V comes from the arms' downstream runs
(paired bootstrap as in earlier rounds), with floors, recall and the strongest previous configuration. Then the
pre-declared terms. Writes report.json and report.md (aggregates only) into --out.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.pediatric_eval import assemble  # noqa: E402
from scripts.pediatric_extract import verify_protocol  # noqa: E402
from scripts.report_model_development import paired_bootstrap  # noqa: E402
from scripts.report_model_development_round2 import category, metrics, read  # noqa: E402
from scripts.round4_failure_inventory import load_asrf  # noqa: E402
from scripts.round5_reachability import asrf_decode, b3_decode  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from echo_routing.compose.recipe_builder import recipe_from_dict  # noqa: E402
from echo_routing.config import load_paths  # noqa: E402
from echo_routing.evaluate.metrics_boundary import match_boundaries  # noqa: E402
from echo_routing.evaluate.metrics_segmentation import segment_f1  # noqa: E402
from echo_routing.realvideo.pediatric import recipe_group  # noqa: E402
from echo_routing.temporal.segments import boundaries_from_labels, labels_to_segments  # noqa: E402

SEEDS, ARMS, SYSTEMS = (0, 1, 2), ("D0", "D1", "D2"), ("frame", "B3", "ASRF")
PSAX, A4C, TOL, N_BOOT, BOOT_SEED = 1, 2, 5, 2000, 20261002
FEAT = "pediatric/ped-ts10hz-e1crop224-v1/features"
E1_B3 = "runs/20261002-144641-md_simple_decoders-38bbcf7c-s0"
HALVES, FLOORS = ("selection", "calibration"), {"selection": 0.973, "calibration": 0.980}
EV9V_METRICS = ("multi segment F1@50", "native macro-F1", "multi boundary recall", "native cuts per clip")


def b3_params(simple_run: str) -> dict:
    return json.loads((Path(simple_run) / "selected.json").read_text())["selected"]["b3"]["main"]


def status_run(path: Path, job: str) -> str:
    s = json.loads(path.read_text())
    if s[job]["state"] != "done":
        raise SystemExit(f"{path}: {job} not done")
    return s[job]["run_dir"]


def sources(spec: dict, k: int) -> dict:
    """arm -> features dir, B3 parameters, ASRF run, EV9V simple run and EV9V ASRF run for seed k."""
    e1 = spec["encoders"]["E1"]; d = Path("runs/pediatric_adapt/downstream")
    out = {"D0": {"feat": "16efc85be645", "b3": E1_B3, "asrf": e1["asrf"][str(k)]}}
    out["D1"] = {"feat": f"D1_s{k}", "b3": status_run(d / f"D1_s{k}_status.json", "simple"), "asrf": e1["asrf"][str(k)]}
    out["D2"] = {"feat": f"D2_s{k}", "b3": status_run(d / f"D2_s{k}_status.json", "simple"), "asrf": status_run(d / f"D2_s{k}_status.json", "M1")}
    return out


def majority(labels: np.ndarray) -> int:
    return int(np.bincount(np.asarray(labels, int), minlength=5).argmax())


def decode(system: str, prob: np.ndarray, feat: np.ndarray, b3: dict, asrf, device) -> np.ndarray:
    if system == "frame":
        return prob.argmax(1)
    if system == "B3":
        return b3_decode(prob, b3)[0]
    return asrf_decode(*asrf, feat.astype(np.float16), device)[0]


def stream_counts(labels: np.ndarray, s) -> dict:
    sem = s.semantic_boundaries; pred_b = boundaries_from_labels(labels)
    matched = {int(sem[j]) for _, j in match_boundaries(pred_b, sem, TOL)}
    starts, ends = np.r_[0, s.joins], np.r_[s.joins, len(labels)]
    maj = [majority(labels[a:b]) for a, b in zip(starts, ends)]; truth = [int(s.labels[a]) for a in starts]
    right = sum(1 for k, j in enumerate(s.joins, start=1) if int(j) in matched and maj[k - 1] == truth[k - 1] and maj[k] == truth[k])
    joins = np.asarray(s.joins, int)
    extra = int(sum(1 for b in pred_b if joins.size == 0 or np.abs(joins - b).min() > TOL))
    seg = segment_f1(labels_to_segments(labels), labels_to_segments(s.labels), 0.5)
    return {"changes": int(sem.size), "found": len(matched), "found_right": right, "extra_cuts": extra, "clips": len(starts),
            "seg_matched": seg["matched"], "seg_pred": len(labels_to_segments(labels)), "seg_true": len(labels_to_segments(s.labels))}


def native_stats(c: np.ndarray) -> dict:
    """c: (..., 2, 5) counts of [PSAX, A4C] true clips by predicted class -> recalls and two-class macro-F1."""
    f = []
    for i, cls in enumerate((PSAX, A4C)):
        tp = c[..., i, cls]; fn = c[..., i, :].sum(-1) - tp; fp = c[..., 1 - i, cls]; den = 2 * tp + fp + fn
        f.append(np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"PSAX recall": c[..., 0, PSAX] / c[..., 0, :].sum(-1), "A4C recall": c[..., 1, A4C] / c[..., 1, :].sum(-1), "macro-F1": (f[0] + f[1]) / 2}


def stream_stats(x: np.ndarray) -> dict:
    """x: (..., 8) sums [changes, found, found_right, extra_cuts, clips, seg_matched, seg_pred, seg_true]."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"segment F1@50": 2 * x[..., 5] / (x[..., 6] + x[..., 7]), "boundary recall": x[..., 1] / x[..., 0],
                "changes found with both views right": x[..., 2] / x[..., 0], "extra cuts per clip": x[..., 3] / x[..., 4]}


def paired(xa: np.ndarray, xb: np.ndarray, fn, W: np.ndarray) -> dict:
    a, b = fn(xa.sum(0)), fn(xb.sum(0)); da, db = fn(np.tensordot(W, xa, 1)), fn(np.tensordot(W, xb, 1))
    return {m: {"delta": float(a[m] - b[m]), "ci95": np.nanpercentile(da[m] - db[m], [2.5, 97.5]).tolist(), "new": float(a[m]), "ref": float(b[m])} for m in a}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", default="runs/md_round4/spec.json"); ap.add_argument("--out", required=True)
    ap.add_argument("--protocol-dir", default="data/manifests/pediatric/pediatric-external-v1")
    a = ap.parse_args(); paths = load_paths(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    proto_dir = Path(a.protocol_dir); verify_protocol(proto_dir); spec = json.loads(Path(a.spec).read_text())
    videos = pd.read_csv(proto_dir / "videos.csv", dtype={"clip_no": str}); test = videos[videos["role"] == "test"].reset_index(drop=True)
    recipes = [recipe_from_dict(d) for d in json.loads((proto_dir / "recipes_test_exam.json").read_text())]
    groups = sorted(test["group"].unique()); gi = {g: i for i, g in enumerate(groups)}
    W = np.random.default_rng(BOOT_SEED).multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=N_BOOT).astype(float)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    nat, strm, clip_labels, res = {}, {}, {}, {"seeds": {}}
    for k in SEEDS:
        src = sources(spec, k); res["seeds"][k] = {"sources": src}
        for arm in ARMS:
            d = paths.cache_root / FEAT / src[arm]["feat"]; b3 = b3_params(src[arm]["b3"])
            asrf = load_asrf(Path(src[arm]["asrf"]), paths.repo_root, 512, device)
            clips = {}
            for v in videos["video_id"]:
                with np.load(d / f"{v}.npz") as z:
                    clips[v] = (z["prob"].astype(float), z["feat"].astype(np.float32))
            for sysn in SYSTEMS:
                c = np.zeros((len(groups), 2, 5)); labs = {}
                for r in test.itertuples(index=False):
                    p, f = clips[r.video_id]; m = majority(decode(sysn, p, f, b3, asrf, device)); labs[r.video_id] = m
                    c[gi[r.group], 0 if r.label == PSAX else 1, m] += 1
                nat[(k, arm, sysn)] = c; clip_labels[(k, arm, sysn)] = labs
                x = np.zeros((len(groups), 8))
                for rec in recipes:
                    s = assemble(clips, rec, "true", 0); st = stream_counts(decode(sysn, s.prob, s.feat, b3, asrf, device), s)
                    x[gi[recipe_group(rec)]] += [st["changes"], st["found"], st["found_right"], st["extra_cuts"], st["clips"],
                                                 st["seg_matched"], st["seg_pred"], st["seg_true"]]
                strm[(k, arm, sysn)] = x
            print(f"seed {k} {arm} done", flush=True)
    point = {f"{arm}|{sysn}|s{k}": {**{m: float(v) for m, v in native_stats(nat[(k, arm, sysn)].sum(0)).items()},
                                     **{m: float(v) for m, v in stream_stats(strm[(k, arm, sysn)].sum(0)).items()}}
             for k in SEEDS for arm in ARMS for sysn in SYSTEMS}
    contrasts = {}
    for new, ref in (("D1", "D0"), ("D2", "D0"), ("D2", "D1")):
        for sysn in SYSTEMS:
            per = {k: {**paired(nat[(k, new, sysn)], nat[(k, ref, sysn)], native_stats, W), **paired(strm[(k, new, sysn)], strm[(k, ref, sysn)], stream_stats, W)}
                   for k in SEEDS}
            contrasts[f"{new} - {ref}|{sysn}"] = {"per_seed": per, "category": {m: category([per[k][m]["ci95"] for k in SEEDS], len(SEEDS)) for m in per[0]}}
    changes = {}
    for k in SEEDS:
        for arm in ("D1", "D2"):
            for sysn in SYSTEMS:
                new, ref = clip_labels[(k, arm, sysn)], clip_labels[(k, "D0", sysn)]
                lab = dict(zip(test["video_id"], test["label"])); grp = dict(zip(test["video_id"], test["group"]))
                fixed = [v for v in new if new[v] == lab[v] and ref[v] != lab[v]]; broke = [v for v in new if new[v] != lab[v] and ref[v] == lab[v]]
                changes[f"{arm}|{sysn}|s{k}"] = {"turned right": {"clips": len(fixed), "groups": len({grp[v] for v in fixed})},
                                                 "turned wrong": {"clips": len(broke), "groups": len({grp[v] for v in broke})}}
    ev9v = ev9v_section(spec)
    rule = {f"{arm}|{sysn}|s{k}": bool(point[f"{arm}|{sysn}|s{k}"]["A4C recall"] >= 0.80 and point[f"{arm}|{sysn}|s{k}"]["PSAX recall"] >= 0.80)
            for k in SEEDS for arm in ARMS for sysn in SYSTEMS}
    meets = {arm: all(rule[f"{arm}|B3|s{k}"] for k in SEEDS) for arm in ARMS}
    cat = lambda key, m: contrasts[key]["category"][m]  # noqa: E731
    terms = {"meets_transfer_rule_B3_all_seeds": meets,
             "head_suffices": bool(meets["D1"] and cat("D2 - D1|B3", "macro-F1") != "consistent (higher)"),
             "representation_helps": bool(cat("D2 - D1|B3", "macro-F1") == "consistent (higher)" and meets["D2"]),
             "gap_remains": bool(not meets["D1"] and not meets["D2"]),
             "segmentation_gain": {arm: any(cat(f"{arm} - D0|{s}", "changes found with both views right") == "consistent (higher)" for s in ("B3", "ASRF")) for arm in ("D1", "D2")},
             "ev9v_kept": ev9v["kept"]}
    report = {"point": point, "transfer_rule": rule, "contrasts": contrasts, "changes": changes, "ev9v": ev9v, "terms": terms, "seeds": res["seeds"]}
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str)); md = markdown(report)
    (out / "report.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


def ev9v_section(spec: dict) -> dict:
    """EV9V both halves: B3 and ASRF of every arm and seed against D0, the floors, recall and the strongest previous."""
    e1 = spec["encoders"]["E1"]; d = Path("runs/pediatric_adapt/downstream")
    strongest_t = {"selection": read(Path(spec["encoders"]["E0"]["simple_run"]), "rows_b3_tuned_main_selection"),
                   "calibration": read(Path(spec["b7_run"]), "rows_B7_b3_calibration")}
    strongest = {h: metrics(t)["segment_f1_50_multi"] for h, t in strongest_t.items()}
    rows = {}
    for k in SEEDS:
        rows[("D0", "B3", k)] = {h: read(Path(E1_B3), f"rows_b3_tuned_main_{h}") for h in HALVES}
        rows[("D0", "ASRF", k)] = {h: read(Path(e1["asrf"][str(k)]), f"rows_{h}") for h in HALVES}
        for arm in ("D1", "D2"):
            rows[(arm, "B3", k)] = {h: read(Path(status_run(d / f"{arm}_s{k}_status.json", "simple")), f"rows_b3_tuned_main_{h}") for h in HALVES}
        rows[("D1", "ASRF", k)] = rows[("D0", "ASRF", k)]                       # same features, same ASRF
        rows[("D2", "ASRF", k)] = {h: read(Path(status_run(d / f"D2_s{k}_status.json", "M1")), f"rows_{h}") for h in HALVES}
    point = {f"{arm}|{sysn}|s{k}|{h}": metrics(rows[(arm, sysn, k)][h]) for (arm, sysn, k) in rows for h in HALVES}
    contrasts, kept = {}, {}
    for arm in ("D1", "D2"):
        loss = False
        for sysn in ("B3", "ASRF"):
            if arm == "D1" and sysn == "ASRF":
                continue                                                         # identical to D0
            for h in HALVES:
                per = {k: paired_bootstrap(rows[(arm, sysn, k)][h], rows[("D0", sysn, k)][h]) for k in SEEDS}
                cats = {m: category([per[k][m]["ci95"] for k in SEEDS], len(SEEDS)) for m in EV9V_METRICS}
                contrasts[f"{arm} - D0|{sysn}|{h}"] = {"per_seed": per, "category": cats}
                loss |= cats["native macro-F1"] == "consistent (lower)" or cats["multi segment F1@50"] == "consistent (lower)"
        kept[arm] = not loss
    goals = {}
    for arm in ARMS:
        for sysn in ("B3", "ASRF"):
            g = {}
            for h in HALVES:
                ms = [metrics(rows[(arm, sysn, k)][h]) for k in SEEDS]; mean = float(np.mean([m["segment_f1_50_multi"] for m in ms]))
                rec, nat = min(m["boundary_recall_multi"] for m in ms), min(m["native_macro_f1"] for m in ms)
                g[h] = {"mean_segment_f1_50": mean, "target": strongest[h] + 0.01, "min_recall": rec, "min_native_f1": nat,
                        "passes": bool(mean >= strongest[h] + 0.01 and rec >= 0.95 and nat >= FLOORS[h])}
            goals[f"{arm}|{sysn}"] = {**g, "passes_both": g["selection"]["passes"] and g["calibration"]["passes"]}
    return {"point": point, "contrasts": contrasts, "kept": kept, "goals": goals, "strongest": strongest}


def markdown(r: dict) -> list[str]:
    md = ["# Pediatric adaptation D0/D1/D2 (generated by scripts/pediatric_adapt_report.py)", "",
          "## Pediatric test folds (per seed s0 / s1 / s2)", "",
          "| arm | system | A4C recall | PSAX recall | macro-F1 | segment F1@50 | boundary recall | both views right | extra cuts per clip |", "|---|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        for sysn in SYSTEMS:
            v = [r["point"][f"{arm}|{sysn}|s{k}"] for k in SEEDS]
            md.append(f"| {arm} | {sysn} | " + " | ".join(" / ".join(f"{x[m]:.3f}" for x in v) for m in
                      ("A4C recall", "PSAX recall", "macro-F1", "segment F1@50", "boundary recall", "changes found with both views right", "extra cuts per clip")) + " |")
    md += ["", "Paired contrasts (points; group bootstrap; per seed):", ""]
    for key, c in r["contrasts"].items():
        for m, cat in c["category"].items():
            md.append(f"- {key}, {m}: " + "; ".join(f"s{k} {100 * c['per_seed'][k][m]['delta']:+.2f} [{100 * c['per_seed'][k][m]['ci95'][0]:+.2f}, "
                                                   f"{100 * c['per_seed'][k][m]['ci95'][1]:+.2f}]" for k in SEEDS) + f" -> {cat}")
    md += ["", "Clips turned right / wrong against D0 (clips, groups):", ""] + [f"- {k}: right {v['turned right']}; wrong {v['turned wrong']}" for k, v in r["changes"].items()]
    ev = r["ev9v"]
    md += ["", "## EV9V (both halves)", "", "| arm | system | half | segment F1@50 s0/s1/s2 | native macro-F1 | boundary recall |", "|---|---|---|---|---|---|"]
    for arm in ARMS:
        for sysn in ("B3", "ASRF"):
            for h in HALVES:
                v = [ev["point"][f"{arm}|{sysn}|s{k}|{h}"] for k in SEEDS]
                md.append(f"| {arm} | {sysn} | {h} | " + " / ".join(f"{x['segment_f1_50_multi']:.4f}" for x in v) + " | "
                          + " / ".join(f"{x['native_macro_f1']:.3f}" for x in v) + " | " + " / ".join(f"{x['boundary_recall_multi']:.3f}" for x in v) + " |")
    md += ["", "EV9V paired contrasts against D0:", ""]
    for key, c in ev["contrasts"].items():
        md.append(f"- {key}: " + "; ".join(f"{m} -> {cat}" for m, cat in c["category"].items()))
    md += ["", "Final goal (unchanged):", "", "```", json.dumps(ev["goals"], indent=1), "```", "", "## Terms", "", "```", json.dumps(r["terms"], indent=1), "```"]
    return md


if __name__ == "__main__":
    main()
