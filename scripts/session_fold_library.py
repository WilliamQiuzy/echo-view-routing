#!/usr/bin/env python
"""Session-fold prediction library (configs/experiments/session_fold_library_v1.yaml;
docs/superpowers/plans/2026-10-07-session-fold-library.md). EV9V training split only: the validation halves and the
test split take no part. Session hold-out, not patient hold-out. No segment-level discriminator is trained.

  --folds        The fixed fold rule over the protocol's training clips: folds.json (server, ids) and fold_balance.json.
  --train        B0 per fold (ResNet-18 + five-class head from ImageNet, the B0 v3 recipe) on the other two folds, 30
                 epochs, last weights kept, no validation; --fold 0,1,2 runs them in turn (resumable with --resume).
  --predict      Model k on every clip of fold k (orig and gamma090, 10 Hz) into the feature cache under model k's
                 checkpoint hash; models.json with the versions.
  --recognition  Table 1: five-class recognition per fold (sample and clip level) and the A4C/A5C errors with their
                 recordings and sessions.
  --library      Table 2: streams inside each fold with symmetric contrast, the fixed J0 (kappa 8) candidates, their
                 categories, sources, joins, lengths, confidence and concentration on recordings.

  python scripts/session_fold_library.py --config configs/experiments/session_fold_library_v1.yaml \
      --out docs/results/2026-10-07_session_fold_library --folds
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg  # noqa: E402
from scripts.ev9v_complementarity import md_table  # noqa: E402
import scripts.short_segment_probe as sp  # noqa: E402
import scripts.short_segment_transfer as tr  # noqa: E402

import numpy as np  # noqa: E402

from echo_routing.compose.recipe_builder import Pool, make_multi_recipes, recipe_from_dict, recipe_to_dict  # noqa: E402
from echo_routing.compose.stream_assembler import render  # noqa: E402
from echo_routing.config import load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.evaluate.metrics_classification import classification_summary  # noqa: E402
from echo_routing.temporal.dev_data import FeatureStore  # noqa: E402
from echo_routing.temporal.joint_decoding import joint_viterbi  # noqa: E402

NS = "session_fold_library_v1"
CLASSES = sp.CLASSES
FOCUS = {"A4C->A5C": (sp.A4C, sp.A5C), "A5C->A4C": (sp.A5C, sp.A4C)}
GROUPS = {"all pairs": lambda r: True, "A4C/A5C pairs": lambda r: {r["x"], r["y"]} == {sp.A4C, sp.A5C},
          "other pairs": lambda r: {r["x"], r["y"]} != {sp.A4C, sp.A5C}}


# ----------------------------------------------------------------------------------------------- pure helpers
def assign_folds(clips: list[str], labels: dict[str, int], n: int, seed: int) -> dict[str, int]:
    """Every session to one fold: StratifiedGroupKFold over the clips sorted by id (y = class, groups = session)."""
    from sklearn.model_selection import StratifiedGroupKFold
    clips = sorted(clips); groups = [session_of(c) for c in clips]; fold_of: dict[str, int] = {}
    for k, (_, test) in enumerate(StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed).split(np.zeros(len(clips)), [labels[c] for c in clips], groups)):
        for i in test:
            if fold_of.setdefault(groups[i], k) != k:
                raise SystemExit("a session spans folds")
    return fold_of


def fold_balance(clips: list[str], labels: dict[str, int], fold_of: dict[str, int], n: int) -> list[dict]:
    rows = []
    for k in range(n):
        cs = [c for c in clips if fold_of[session_of(c)] == k]; views = defaultdict(set)
        for c in cs:
            views[session_of(c)].add(labels[c])
        rows.append({"fold": k, "sessions": len(views), "clips": len(cs), **{f"{name} clips": sum(labels[c] == j for c in cs) for j, name in enumerate(CLASSES)},
                     **{f"sessions with {name}": sum(j in v for v in views.values()) for j, name in enumerate(CLASSES)}})
    return rows


def native_variants(clips: list[str], p: float, seed: int, edit_variant: str) -> dict[str, str]:
    """The symmetric contrast edit of the native clips: edit_variant with probability p (seeded, clips sorted)."""
    clips = sorted(clips); u = np.random.default_rng(seed).random(len(clips))
    return {c: (edit_variant if x < p else "orig") for c, x in zip(clips, u)}


def vote(pred: np.ndarray, n_classes: int) -> int:
    """Clip majority vote of per-sample predictions; ties go to the lower class index."""
    return int(np.bincount(np.asarray(pred, int), minlength=n_classes).argmax())


def recognition_table(items: list[tuple[str, int, np.ndarray]]) -> dict:
    """items: (clip id, label, per-sample argmax). Sample- and clip-level summaries, and per focus error the samples, the
    clips voted wrong, and the recordings and sessions involved."""
    nc = len(CLASSES); ys, ps, cy, cv = [], [], [], []; focus = {f: {"samples": 0, "any": set(), "voted": set(), "clips": 0, "of_samples": 0} for f in FOCUS}
    for vid, y, pred in items:
        pred = np.asarray(pred, int); v = vote(pred, nc); ys.append(np.full(pred.size, y)); ps.append(pred); cy.append(y); cv.append(v)
        for f, (a, b) in FOCUS.items():
            if y == a:
                k = int((pred == b).sum()); d = focus[f]; d["samples"] += k; d["of_samples"] += pred.size; d["clips"] += 1
                if k:
                    d["any"].add(vid)
                if v == b:
                    d["voted"].add(vid)
    y_s, p_s = (np.concatenate(ys), np.concatenate(ps)) if ys else (np.zeros(0, int), np.zeros(0, int))
    out = {"clips": len(items), "sessions": len({session_of(v) for v, _, _ in items}), "sample": classification_summary(y_s, p_s, nc),
           "clip": classification_summary(np.array(cy, int), np.array(cv, int), nc), "focus": {}}
    for f, d in focus.items():
        out["focus"][f] = {"samples": d["samples"], "of samples": d["of_samples"], "recordings with any": len(d["any"]), "sessions with any": len({session_of(v) for v in d["any"]}),
                           "clips voted": len(d["voted"]), "sessions of clips voted": len({session_of(v) for v in d["voted"]}), "of clips": d["clips"]}
    return out


def fold_candidates(seqs, kappa: float, eps: float, tol: int, edit_variant: str) -> list[dict]:
    """Candidates of every sequence (group, kind, prob, reference, joins, fragment bounds, videos, variants)."""
    rows = []
    for group, kind, prob, ref, joins, bounds, videos, variants in seqs:
        prob = np.asarray(prob, float); pred = joint_viterbi(prob, np.zeros(len(prob)), 0.0, kappa, eps)
        for left, mid, right in sp.sandwiches(pred):
            cat, splice = sp.categorise(left, mid, right, ref, joins, tol); mv = sp.videos_over(bounds, videos, mid.start, mid.end)
            var = {variants[k] for k in range(len(videos)) if bounds[k] < mid.end and bounds[k + 1] > mid.start}
            rows.append({"group": group, "kind": kind, "cat": cat, "reason": tr.other_reason(left, mid, right, ref) if cat == "other" else None,
                         "join": tr.join_type(ref, joins, mid.start, mid.end, tol), "splice": splice, "x": int(left.label), "y": int(mid.label),
                         "len_mid": int(mid.length), "len_left": int(left.length), "len_right": int(right.length),
                         "conf_mid": float(prob[mid.start:mid.end, mid.label].mean()), "contrast": "edited" if var == {edit_variant} else "orig" if var == {"orig"} else "mixed",
                         "mid_videos": mv, "mid_sessions": sorted({session_of(v) for v in mv})})
    return rows


def concentration(rows: list[dict]) -> dict:
    """How far candidates rest on few recordings: the largest recording's share (a candidate counts 1/k for each of its
    k recordings) and the effective number of recordings, 1 / sum(share^2)."""
    w: Counter = Counter()
    for r in rows:
        for v in r["mid_videos"]:
            w[v] += 1 / len(r["mid_videos"])
    total = sum(w.values())
    if not total:
        return {"largest recording share": None, "effective recordings": None, "most candidates on one recording": 0}
    shares = [x / total for x in w.values()]
    return {"largest recording share": max(shares), "effective recordings": 1 / sum(s * s for s in shares), "most candidates on one recording": max(Counter(v for r in rows for v in r["mid_videos"]).values())}


def quartiles(values: list[float]) -> list[float] | None:
    return [float(q) for q in np.percentile(values, [25, 50, 75])] if values else None


def describe(rows: list[dict]) -> dict:
    out = {}
    for g, test in GROUPS.items():
        out[g] = {}
        for cat in ("kept", "removed", "other"):
            rs = [r for r in rows if test(r) and r["cat"] == cat]; vids = {v for r in rs for v in r["mid_videos"]}
            out[g][cat] = {"candidates": len(rs), "recordings": len(vids), "sessions": len({session_of(v) for v in vids}),
                           **{f"from {k}": sum(r["kind"] == k for r in rs) for k in ("native clips", "multi streams")},
                           **{f"{j} join": sum(r["join"] == j for r in rs) for j in ("none", "same-view", "view-change")},
                           **{f"contrast {c}": sum(r["contrast"] == c for r in rs) for c in ("orig", "edited", "mixed")},
                           "len_mid quartiles": quartiles([r["len_mid"] for r in rs]), "conf_mid quartiles": quartiles([r["conf_mid"] for r in rs]),
                           "len_mid 10-40": sum(10 <= r["len_mid"] <= 40 for r in rs), "concentration": concentration(rs)}
    out["outside reasons"] = dict(Counter(r["reason"] for r in rows if r["cat"] == "other"))
    out["class pairs"] = {cat: {f"{CLASSES[x]}>{CLASSES[y]}": n for (x, y), n in sorted(Counter((r["x"], r["y"]) for r in rows if r["cat"] == cat).items())}
                          for cat in ("kept", "removed", "other")}
    return out


# ----------------------------------------------------------------------------------------------- inputs
def check_inputs(sf: dict, root: Path) -> dict:
    for name, spec in sf["inputs"].items():
        if (got := sha256_file(root / spec["path"])) != spec["sha256"]:
            raise SystemExit(f"input {name}: sha256 {got[:12]} != {spec['sha256'][:12]}")
    return json.loads((root / sf["inputs"]["protocol"]["path"]).read_text())


def train_labels(root: Path, sf: dict, proto: dict) -> dict[str, int]:
    import pandas as pd
    man = pd.read_csv(root / sf["inputs"]["manifest"]["path"]).set_index("video_id"); clips = proto["train"]["clips"]
    if man.loc[clips, "split"].ne("train").any():
        raise SystemExit("a protocol training clip is not in the manifest's training split")
    return {v: int(man.loc[v, "family5_index"]) for v in clips}


def folds_of(root: Path) -> dict[str, int]:
    return json.loads((root / "runs" / NS / "folds.json").read_text())


def check_out(out: str, sf: dict) -> None:
    if Path(out) != Path(sf["results_dir"]):
        raise SystemExit(f"--out must be {sf['results_dir']}")


# ----------------------------------------------------------------------------------------------- steps
def run_folds(cfg: dict, sf: dict, paths, out_dir: Path, _) -> None:
    root = paths.repo_root; proto = check_inputs(sf, root); labels = train_labels(root, sf, proto); f = sf["folds"]
    if (root / "runs" / NS / "folds.json").exists():
        raise SystemExit("folds.json exists: the folds are written once")
    fold_of = assign_folds(list(labels), labels, int(f["n"]), int(f["seed"])); (root / "runs" / NS).mkdir(parents=True, exist_ok=True)
    (root / "runs" / NS / "folds.json").write_text(json.dumps(fold_of, indent=0, sort_keys=True))
    import sklearn
    rows = fold_balance(sorted(labels), labels, fold_of, int(f["n"]))
    spread = {key: (max(r[key] for r in rows) - min(r[key] for r in rows)) for key in rows[0] if key != "fold"}
    rec = {"rule": f["rule"], "sklearn": sklearn.__version__, "clips": len(labels), "sessions": len(fold_of), "folds": rows, "spread (max - min)": spread}
    out_dir.mkdir(parents=True, exist_ok=True); (out_dir / "fold_balance.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec, indent=1))


def run_train(cfg: dict, sf: dict, paths, out_dir: Path, a) -> None:
    from echo_routing.audit.label_map import load_ontology
    from echo_routing.features.train_encoder import TrainConfig, records_from_manifest, train_fixed_epochs
    from echo_routing.manifest import load_manifest
    root = paths.repo_root; proto = check_inputs(sf, root); fold_of = folds_of(root); clips = set(proto["train"]["clips"]); enc = cfg["encoder"]
    if Path(paths.manifest_path).resolve() != (root / sf["inputs"]["manifest"]["path"]).resolve():
        raise SystemExit("the configured manifest is not the pinned one")
    recs = [r for r in records_from_manifest(load_manifest(paths.manifest_path), cfg["task"], "train") if r.video_id in clips]
    if len(recs) != len(clips):
        raise SystemExit(f"{len(recs)} of {len(clips)} training clips have frames on disk")
    tcfg = TrainConfig.from_dict({"task": cfg["task"], "backbone": enc["backbone"], "image_size": cfg["preprocess"]["image_size"],
                                  "frames_per_video": cfg["sampling"]["train_frames_per_video"], "videos_per_epoch": enc.get("videos_per_epoch"),
                                  "batch_size": enc["batch_size"], "lr": enc["lr"], "weight_decay": enc["weight_decay"], "epochs": int(sf["training"]["epochs"]),
                                  "patience": enc["patience"], "seed": int(sf["training"]["seed"]), "num_workers": enc["num_workers"],
                                  "eval_frames_per_video": cfg["sampling"]["eval_frames_per_video"], "amp": enc["amp"]})
    classes = list(load_ontology().classes(cfg["task"]))
    if tuple(classes) != CLASSES:
        raise SystemExit(f"class order {classes} differs from {CLASSES}")
    if enc["pretrained"] is not True:
        raise SystemExit("the protocol requires ImageNet initialisation (encoder.pretrained: true)")
    for k in [int(x) for x in a.fold.split(",")]:
        train_recs = [r for r in recs if fold_of[session_of(r.video_id)] != k]; ck = paths.checkpoints_root / NS / f"fold{k}"
        if (ck / "last.pt").exists() and not a.resume:
            raise SystemExit(f"fold {k} already has last.pt: pass --resume to continue it; a fold is never retrained silently")
        ck.mkdir(parents=True, exist_ok=True)
        (ck / "run_manifest.json").write_text(json.dumps({"fold": k, "train_on": "the two other folds", "n_train_clips": len(train_recs),
                                                          "n_train_sessions": len({session_of(r.video_id) for r in train_recs}), "config": tcfg.__dict__,
                                                          "protocol_sha256": sf["inputs"]["protocol"]["sha256"], "validation": "none"}, indent=1))
        print(f"fold {k}: {len(train_recs)} training clips -> {ck}", flush=True)
        train_fixed_epochs(tcfg, train_recs, classes, ck, resume=(ck / "last.pt") if a.resume else None, pretrained=bool(enc["pretrained"]))


def run_predict(cfg: dict, sf: dict, paths, out_dir: Path, _) -> None:
    import pandas as pd
    import torch

    from echo_routing.config import gpu_memory_fraction
    from echo_routing.features.encoder import checkpoint_hash, load_checkpoint
    from echo_routing.features.extract import extract_split
    root = paths.repo_root; proto = check_inputs(sf, root); fold_of = folds_of(root); epochs = int(sf["training"]["epochs"])
    man = pd.read_csv(root / sf["inputs"]["manifest"]["path"]); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    models = {}
    for k in range(int(sf["folds"]["n"])):
        ck = paths.checkpoints_root / NS / f"fold{k}" / "last.pt"; model, meta = load_checkpoint(ck)
        if int(meta["epoch"]) != epochs - 1:
            raise SystemExit(f"fold {k}: last.pt is from epoch {meta['epoch']}, not the final epoch {epochs - 1}")
        key = checkpoint_hash(ck); clips = sorted(v for v in proto["train"]["clips"] if fold_of[session_of(v)] == k)
        rows = man[man.video_id.isin(clips)]
        if len(rows) != len(clips):
            raise SystemExit(f"fold {k}: {len(rows)} of {len(clips)} clips in the manifest")
        for variant in sf["prediction"]["variants"]:
            extract_split(model, device, rows, "family5_index", paths.cache_root, key, variant, cfg["sampling"]["target_hz"], cfg["preprocess"]["image_size"],
                          cfg["features"]["batch_size"], cfg["features"]["num_workers"], meta={"ckpt": str(ck), "task": cfg["task"], "classes": list(CLASSES), "fold": k})
        log = [json.loads(x) for x in (ck.parent / "train_log.jsonl").read_text().splitlines()]
        models[str(k)] = {"checkpoint": str(ck.relative_to(paths.checkpoints_root.parent)) if ck.is_relative_to(paths.checkpoints_root.parent) else str(ck),
                          "sha256": sha256_file(ck), "cache_key": key, "final_epoch": int(meta["epoch"]), "init_sha256": meta.get("init_sha256"),
                          "final_train_loss": log[-1]["train_loss"], "train_seconds": sum(x["seconds"] for x in log), "epochs_logged": len(log),
                          "n_train_clips": log[-1]["n_train_records"], "predicted_clips": len(clips), "variants": sf["prediction"]["variants"]}
        print(json.dumps({"fold": k, **{x: models[str(k)][x] for x in ("cache_key", "final_train_loss", "predicted_clips")}}), flush=True)
    out_dir.mkdir(parents=True, exist_ok=True); (out_dir / "models.json").write_text(json.dumps(models, indent=1))


def fold_clips(proto: dict, fold_of: dict, k: int) -> list[str]:
    return sorted(v for v in proto["train"]["clips"] if fold_of[session_of(v)] == k)


def run_recognition(cfg: dict, sf: dict, paths, out_dir: Path, _) -> None:
    root = paths.repo_root; proto = check_inputs(sf, root); labels = train_labels(root, sf, proto); fold_of = folds_of(root)
    models = json.loads((out_dir / "models.json").read_text()); res, pooled = {"variant": "orig", "folds": {}}, []
    for k, m in models.items():
        store = FeatureStore(paths.cache_root, m["cache_key"])
        items = [(v, labels[v], store.get(v, "orig").prob.argmax(1)) for v in fold_clips(proto, fold_of, int(k))]
        res["folds"][k] = recognition_table(items); pooled += items
    res["pooled"] = recognition_table(pooled); (out_dir / "recognition.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: {"clip macro-F1": v["clip"]["macro_f1"], **{f: (x["clips voted"], x["recordings with any"]) for f, x in v["focus"].items()}}
                      for k, v in {**res["folds"], "pooled": res["pooled"]}.items()}, indent=1))


def fold_sequences(bank: dict, store, labels: dict):
    for d in bank["recipes"]:
        r = recipe_from_dict(d); s = render(r, store.get); bounds = np.cumsum([0, *[f.end - f.start for f in r.fragments]])
        yield f"stream:{r.recipe_id}", "multi streams", s.prob, s.labels, s.joins, bounds, [f.video_id for f in r.fragments], [f.variant for f in r.fragments]
    for v, variant in sorted(bank["natives"].items()):
        vf = store.get(v, variant)
        yield f"session:{session_of(v)}", "native clips", vf.prob, np.full(vf.n, labels[v]), np.zeros(0, int), np.array([0, vf.n]), [v], [variant]


def run_library(cfg: dict, sf: dict, paths, out_dir: Path, _) -> None:
    from echo_routing.features.cache import read_index, variant_dir
    root = paths.repo_root; proto = check_inputs(sf, root); labels = train_labels(root, sf, proto); fold_of = folds_of(root)
    models = json.loads((out_dir / "models.json").read_text()); st, mc, rc, gen = sf["streams"], cfg["multi"], cfg["recipes"], sf["generator"]
    res, pooled, t0 = {"folds": {}}, [], time.time()
    for k, m in models.items():
        clips = fold_clips(proto, fold_of, int(k)); idx = read_index(variant_dir(paths.cache_root, m["cache_key"], "orig")).set_index("video_id")
        pool = Pool.from_index(clips, [labels[v] for v in clips], [int(idx.loc[v, "n_samples"]) for v in clips])
        rec = make_multi_recipes(pool, int(st["n_streams"]), np.random.default_rng(int(st["seeds"][int(k)])), n_fragments=tuple(mc["n_fragments"]),
                                 edit_variant=rc["edit_variant"], edit_prob=float(mc["edit_prob"]), min_len=int(rc["min_len"]), max_len=int(rc["max_len"]),
                                 reuse=True, p_same_next=float(mc["p_same_next"]))
        if any(fold_of[session_of(f.video_id)] != int(k) for r in rec for f in r.fragments):
            raise SystemExit("a constructed stream crosses folds")
        bank = {"recipes": [recipe_to_dict(r) for r in rec], "natives": native_variants(clips, float(mc["edit_prob"]), int(st["native_contrast_seeds"][int(k)]), rc["edit_variant"])}
        rows = fold_candidates(fold_sequences(bank, FeatureStore(paths.cache_root, m["cache_key"]), labels), float(gen["kappa"]), float(gen["eps"]),
                               int(sf["candidates"]["tolerance_samples"]), rc["edit_variant"])
        (root / "runs" / NS / f"banks_f{k}.json").write_text(json.dumps(bank)); (root / "runs" / NS / f"library_f{k}.json").write_text(json.dumps(rows))
        res["folds"][k] = {"streams": len(rec), "natives": len(clips), "natives edited": sum(v != "orig" for v in bank["natives"].values()),
                           "fragments edited": sum(f.variant != "orig" for r in rec for f in r.fragments), "fragments": sum(len(r.fragments) for r in rec),
                           "candidates": len(rows), "composition": describe(rows)}
        pooled += rows; print(json.dumps({"fold": k, "candidates": len(rows), **{c: sum(r["cat"] == c for r in rows) for c in ("kept", "removed", "other")}}), flush=True)
    res["pooled"] = {"candidates": len(pooled), "composition": describe(pooled)}; res["seconds"] = time.time() - t0
    (out_dir / "composition.json").write_text(json.dumps(res, indent=1)); (out_dir / "report_tables.md").write_text(tables_md(out_dir, res))


# ----------------------------------------------------------------------------------------------- report
def tables_md(out_dir: Path, comp: dict) -> str:
    rec = json.loads((out_dir / "recognition.json").read_text()); t1, t2 = [], []
    for k, v in {**rec["folds"], "pooled": rec["pooled"]}.items():
        row = {"fold": k, "clips": v["clips"], "sessions": v["sessions"], "sample accuracy": v["sample"]["accuracy"], "sample macro-F1": v["sample"]["macro_f1"],
               "clip accuracy": v["clip"]["accuracy"], "clip macro-F1": v["clip"]["macro_f1"]}
        for f, x in v["focus"].items():
            row[f"{f} samples"] = f"{x['samples']}/{x['of samples']}"; row[f"{f} recordings / sessions with any"] = f"{x['recordings with any']} / {x['sessions with any']}"
            row[f"{f} clips voted (sessions)"] = f"{x['clips voted']}/{x['of clips']} ({x['sessions of clips voted']})"
        t1.append(row)
    for k, v in {**comp["folds"], "pooled": comp["pooled"]}.items():
        for g in GROUPS:
            for cat in ("kept", "removed", "other"):
                c = v["composition"][g][cat]
                t2.append({"fold": k, "pairs": g, "category": cat, "candidates": c["candidates"], "recordings / sessions": f"{c['recordings']} / {c['sessions']}",
                           "native / stream": f"{c['from native clips']} / {c['from multi streams']}", "no / same-view / view-change join": f"{c['none join']} / {c['same-view join']} / {c['view-change join']}",
                           "contrast orig / edited / mixed": f"{c['contrast orig']} / {c['contrast edited']} / {c['contrast mixed']}",
                           "len_mid quartiles": None if c["len_mid quartiles"] is None else [round(q, 1) for q in c["len_mid quartiles"]],
                           "conf_mid quartiles": None if c["conf_mid quartiles"] is None else [round(q, 3) for q in c["conf_mid quartiles"]],
                           "largest recording share": c["concentration"]["largest recording share"], "effective recordings": c["concentration"]["effective recordings"]})
    def tab(rows: list[dict]) -> str:
        return md_table(rows, list(dict.fromkeys(x for r in rows for x in r)))
    return "\n".join(["# Session-fold prediction library: tables\n", "## Table 1: recognition (orig variant)\n", tab(t1), "## Table 2: candidates\n", tab(t2)])


STEPS = {"folds": run_folds, "train": run_train, "predict": run_predict, "recognition": run_recognition, "library": run_library}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--fold", default="0,1,2", help="--train: comma list of folds, trained in turn"); ap.add_argument("--resume", action="store_true")
    step = ap.add_mutually_exclusive_group(required=True)
    for name in STEPS:
        step.add_argument(f"--{name}", action="store_true")
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); sf = cfg["session_fold_library"]; check_out(a.out, sf); paths = load_paths()
    name = next(n for n in STEPS if getattr(a, n)); STEPS[name](cfg, sf, paths, paths.repo_root / a.out, a)


if __name__ == "__main__":
    main()
